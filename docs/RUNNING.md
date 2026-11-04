# Running

From a cold boot to generated tokens: power up the late cards, swap in the patched UVM module, start the daemon, then run vLLM clients as often as you like.

## Rules the runners enforce, and you should too

- **Every register and page-table entry a run changes is saved, verified and restored.** This covers ACS bits, multicast registers, the worker configuration words and the remapped PTEs. Each run folder has before/after snapshots and a result JSON with explicit checks (`mc_registers_unchanged`, route cleanup, `dma_helper_unloaded`, worker sequence counters).
- **One GPU job at a time.** The examples queue them through [Pueue](https://github.com/Nukesor/pueue) in a group with one slot (`pueue add --group gpu --label <unique label> -- sudo -n …`). The runners need root for config-space access and the UVM ioctls.
- **No automatic retries.** A failed run leaves its evidence; read it before running again.
- **Never edit a source file while a queued or running job uses it.** Bash reads scripts incrementally, and the daemon compiles from `src/` at start.
- **Stop M10 processes only by exact PID.** `pgrep -f` on a service path also matches the daemon.
- **Don't raise the PTE record limit (512)**, and don't reload UVM to clean records: use `recover-peer-client`.

## 1. Late-power boot

See [HARDWARE.md](HARDWARE.md#boot-sequence): cold boot with PSU2 off, `boot/prepare-and-suspend.sh`, PSU2 on while asleep, wake, `boot/resume-and-enumerate.sh`. Afterwards `nvidia-smi` shows 24 Tesla M10 GPUs and the RTX.

## 2. Swap in the patched UVM module

The stock `nvidia_uvm` loads at boot. Replace it for the session:

```bash
sudo systemctl stop m10-fancontrol        # holds the device nodes; its stop hook sets the fans to full
# close nvtop / nvidia-smi watchers
sudo transport/driver/swap-uvm.sh build/uvm/nvidia-uvm-m10dis.ko
sudo systemctl start m10-fancontrol
```

The script reloads the module with the same parameters, restores the original on any failure and appends the loaded identity to `$M10DIS_STATE/uvm-swap.log`. Put the printed `loaded_at` time into `m10dis.local.env` as `Q38_UVM_LAST_LOAD=…`, and the module's SHA-256 as `Q38_UVM_SHA=…`. The daemon start refuses to run if the module was reloaded since the recorded swap, because a reload drops every route record the daemon depends on.

## 3. Start the resident daemon

```bash
pueue add --group gpu --label daemon_start -- sudo -n bash src/run_qwen38_full_daemon_start.sh
```

What it does, in order:

1. Checks the UVM identity, the inputs ([PREPARE.md](PREPARE.md)) and the topology against a live scan.
2. Builds from the current source into the run folder: worker services, worker cubins (routed v3, shared v2), the PLE owner, the route tool, the direct-MoE library and the write-combining helper module. Every hash goes into `build-hashes.sha256`.
3. Checks the multicast registers of all six cards before the session (`MC_PREMAP_OK`).
4. Starts `qwen38_resident_daemon.py serve`. It loads the n-gram shards and LM-head slices through the PLE owner, then the 48 layers of expert slices onto the 24 GPUs, and verifies them.
5. Waits for `READY`, 48 layers (49 with `Q38_DAEMON_LAYERS=49`, which also loads the MTP layer as layer 48 for MTP runs), workers `PARKED`. It then writes `daemon.env` (sockets, manifest, topology, snapshot root) and links it as `$M10DIS_RUNS/qwen38-full-daemon/latest`.

Loading takes several minutes. The daemon then stays up across any number of clients.

Status at any time:

```bash
.venv-vllm/bin/python src/qwen38_resident_daemon.py status --control-socket "$Q38_FULL_CONTROL_SOCKET"
```

## 4. Run vLLM against it

The full-model peer run:

```bash
pueue add --group gpu --label smoke -- sudo -n bash src/run_qwen38_rtx_peer_vllm.sh
```

It checks the daemon is `READY`, 48 layers, `PARKED`, with no client registered. It relaxes ACS on the switch ports (saving the old values), registers as the daemon's peer client, then runs `run_vllm_resident_m10_client.py` with the settings from its environment. It closes the adapter inside the engine process, restores all routes, ACS bits and worker words, unregisters and writes `result.json` with every check.

The settings are environment variables (full list: [INTERFACE.md](INTERFACE.md#settings)). Ready-made combinations:

| Runner | Use |
|---|---|
| `src/run_qwen38_bench.sh` | the frozen benchmark configuration: pipelined prefill, FP8 GDN/QSA/HC, M10 LM head, KV 2.5 GiB, `max_model_len` 90,112. Reads benchmark items ([BENCHMARKS.md](BENCHMARKS.md)) |
| `src/run_qwen38_moe_batch_integration.sh` | the ~4k-token prompt in `prompts/long_prompt_4k.txt`, 32 new tokens, once per entry of `Q38_BATCH_INTEGRATION_MODES` (0 per row, 1 batched, 2 pipelined). With `Q38_VLLM_LOGPROBS=5` it records the top-5 logprobs per generated token, for floor runs |
| `src/run_qwen38_rtx_peer_vllm.sh` alone | the HumanEval-style default prompt (`prompts/humaneval32_prompt.py`) |

Example, the reference floor run (5 identical pipelined runs):

```bash
pueue add --group gpu --label floor -- sudo -n env Q38_RTX_FP8_HC=1 VLLM_TRITON_FORCE_FIRST_CONFIG=1 \
  Q38_VLLM_LOGPROBS=5 Q38_BATCH_CHUNK=1024 Q38_BATCH_INTEGRATION_MODES=2,2,2,2,2 \
  bash src/run_qwen38_moe_batch_integration.sh
python3 src/lm_head_floor.py <integration folder> [<reference integration folder>]
```

## 5. Change worker kernels without reloading weights

Edit the worker headers, then:

```bash
pueue add --group gpu --label reload -- sudo -n bash src/run_pci_cluster_kernel_reload.sh
```

It parks the workers, builds new cubins and sends `RELOAD_KERNELS`. It refuses while any client is active. The workers stop their resident grids, load the new images into their existing contexts and restart, keeping all weights. The daemon records the image hashes and a reload generation. Changes to the service binaries or to the shared mailbox layout still need a daemon restart:

```bash
pueue add --group gpu --label restart -- sudo -n bash src/run_qwen38_daemon_restart.sh
```

The restart sends `STOP`, waits for the daemon to exit by itself (it never kills it), checks that no process of the old build is left and starts a new daemon from the current source.

## 6. When something goes wrong

| Symptom | What to do |
|---|---|
| Client crashed, daemon shows a registered peer client | `qwen38_resident_daemon.py recover-peer-client --control-socket …`: restores the journaled routes and worker words of that client |
| Route records left by a dead process | `qwen38_resident_daemon.py list-route-records` shows them. With the lifecycle patch, UVM frees a record when its VA space is destroyed. The `REAP_DEAD` ioctl frees records of already-dead VA spaces; this tree has no command-line tool for it |
| Worker exit with a reason code | the RTX reports it (sticky error and reason). Codes: [INTERFACE.md](INTERFACE.md) |
| `UVM reloaded since the recorded swap` | the module was reloaded (or the machine rebooted): swap again, update `Q38_UVM_LAST_LOAD`, restart the daemon |
| An M10 runs hot | check `systemctl status m10-fancontrol` and `fan/bin/fanstatus`; `fan/bin/m10-fan-full` sets all banks to full |

## 7. Shut down

Stop clients, then `qwen38_resident_daemon.py stop --control-socket …`. `STOP` is refused while a peer client is registered. The daemon parks the workers, releases the PLE owner after every worker import has closed and exits.
