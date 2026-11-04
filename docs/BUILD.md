# Build

Most binaries are built **per run, from the current source**, by the runners themselves: the daemon start builds the worker services, worker cubins, PLE owner, route tool, direct-MoE library and helper module, and the vLLM client builds the RTX peer library. Each run folder records the hashes of what it used. You build four things once:

## 1. Kernel with the late-enable quirks

Base: Ubuntu's `linux-source-7.0.0` package (7.0.14), configured like the distribution kernel.

```bash
cd linux-source-7.0.0
patch -p1 < /path/to/m10dis/kernel/m10-late-enable-quirks.patch
echo -m10late8 > localversion-m10late
make olddefconfig && make -j"$(nproc)" bindeb-pkg     # or your usual kernel build
```

The patch adds two PCI header fixups to `drivers/pci/quirks.c`:

- `quirk_pex8748_reserve_downstream` marks PEX8748 downstream ports in slots 0x09–0x13 as hotplug bridges, so the kernel reserves bus numbers and memory windows for cards that power up later.
- `quirk_nvidia_gm107_m10_no_io_bar` drops the GM107's 128-byte I/O BAR (BAR5) and clears its I/O decode.

Install the GRUB entry from `kernel/42_m10_eight_port.template` as `/etc/grub.d/42_m10_eight_port`. Fill in your own root and boot UUIDs, and keep the `pci=` parameters. Background: [HARDWARE.md](HARDWARE.md).

## 2. Patched UVM module

Driver 580.178.04 (`nvidia-srv`, DKMS source in `/usr/src/nvidia-srv-580.178.04`):

```bash
transport/driver/build-uvm.sh            # → build/uvm/nvidia-uvm-m10dis.ko
```

The script copies the driver source, applies the five patches in `transport/driver/patches/` in order, builds the modules with the same invocation as the driver's `dkms.conf` and prints srcversion, SHA-256 and vermagic. It does not install anything.

Expected srcversion: **`38263E1067A39300ABB966D`**. The SHA-256 depends on the build path. The runners check the srcversion of the loaded module, and the SHA-256 recorded by `swap-uvm.sh`: set `Q38_UVM_SHA` to your build's hash.

The patches, in order:

| Patch | Adds |
|---|---|
| `glm53-diagnostic-extension` | the guarded PTE query/write/restore ioctls and their record table |
| `qwen38-rtx-peer-multiroute` | external BAR1 owner, retained owner and DMA routes, requester-side PTE construction (RTX → M10 and M10 → RTX) |
| `qwen38-rtx-peer-pte-capacity` | record table raised from 128 to 512 entries |
| `qwen38-pte-lifecycle-cleanup` | releases records when their GPU VA space is destroyed |
| `qwen38-pte-reap-dead` | `REAP_DEAD`: frees records whose VA space is already dead |

Don't install the module through DKMS or the initramfs. The stock module loads at boot and `swap-uvm.sh` replaces it for the session ([RUNNING.md](RUNNING.md)).

## 3. RM ioctl capture preload

```bash
src/build_rm_trace_preload.sh            # → build/rm_ioctl_trace_preload.so
```

The daemon and the vLLM client `LD_PRELOAD` it to capture the RM object handles of their CUDA allocations, which the route setup needs.

## 4. Python environment

```bash
python3.14 -m venv .venv-vllm
.venv-vllm/bin/pip install -r requirements-vllm.txt
```

`requirements-vllm.txt` is the full freeze of the tested environment: vLLM 0.32.0, PyTorch 2.13.0+cu132, Triton 3.7.1, Transformers 5.19.0. vLLM itself is not patched. All changes go in through `src/sitecustomize.py` and the hook modules.

The M10 code needs a CUDA 12.x `nvcc` (12.4 tested; `/usr/bin/nvcc` by default, set `NVCC` otherwise). CUDA 13 no longer targets Maxwell (sm_50). Its host compiler is `g++-13` (`HOST_CXX` to change). The runners compile for `sm_50` (M10) and use PyTorch's bundled CUDA for the RTX.

## 5. Fan drivers (optional, for cooling)

- `fan/drivers/nct6687/`: `make` against the running kernel, install `nct6687.ko`, use `nct6687-modprobe.conf`.
- `fan/drivers/arctic-fan-controller/`: `scripts/prepare.sh` then `scripts/dkms-install.sh` (see its README).

Then `fan/install-m10-fancontrol.sh`. Details: [HARDWARE.md](HARDWARE.md#cooling).

## Tests that need no hardware run

```bash
CUDA_VISIBLE_DEVICES= .venv-vllm/bin/python src/test_qwen38_fp8_block_quant.py "$M10DIS_MODEL"
CUDA_VISIBLE_DEVICES= .venv-vllm/bin/python src/test_qwen38_cpu_head_hooks.py "$M10DIS_MODEL"
.venv-vllm/bin/python src/test_qwen38_daemon_peer_recovery.py
python3 src/test_qwen38_pci_discover.py          # reads sysfs only
```

`src/test_qwen38_hc_fp8.py` needs the RTX (no M10 work). The `test_*.cu` programs are hardware tests driven by the `run_*_test.sh` runners.
