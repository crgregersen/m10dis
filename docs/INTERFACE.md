# Interface reference: daemon, worker modes, settings

This is the reference for everything a client or operator can ask the resident M10 daemon (`src/qwen38_resident_daemon.py serve`) to do. The operational how-to (start, readiness, kernel reload) is in [RUNNING.md](RUNNING.md).

**Operator rule.** Do not stop the daemon to clean up after a client. Park the workers, recover the client (`RECOVER_PEER_CLIENT`) and reload kernels while parked. Stop the daemon only when its weights or services themselves must change, and only with the operator's go-ahead.

## Transport

| Item | Value |
|---|---|
| Socket | `RUNTIME_DIR/control.sock` (Unix stream, mode 0600, owner root) |
| Request | one JSON object on one line, `{"command": "...", ...fields}` |
| Response | one JSON object on one line |
| Python helper | `qwen38_resident_daemon.send_control(path, command, timeout, **fields)` |
| CLI | `qwen38_resident_daemon.py <subcommand> --control-socket PATH ...` (prints the JSON response) |

CLI exit status: `stop`, `recover-peer-client` and `list-route-records` exit 1 when the response is not `ok`/`STOP_ACCEPTED`. The older subcommands always exit 0 for compatibility; callers must read the JSON `status`.

`clients_quiesced: true` is the caller's assertion that no RTX/vLLM request is in flight. Commands that touch the workers refuse without it (`CLIENTS_NOT_QUIESCED`).

## Commands

| Command (CLI) | Request fields | Effect | Success response |
|---|---|---|---|
| `STATUS` (`status`) | – | Full daemon state (see below) | state object, `status: READY` when serving |
| `PARK_WORKERS` (`park-workers --clients-quiesced`) | `clients_quiesced` | Stop the resident polling grids; weights stay resident | `code: WORKERS_PARKED` |
| `WAKE_WORKERS` (`wake-workers --clients-quiesced`) | `clients_quiesced` | Relaunch the polling grids against the retained weights | `code: WORKERS_ACTIVE` |
| `RELOAD_KERNELS` (`reload-kernels --routed-image R --shared-image S --clients-quiesced`) | `routed_image`, `shared_image`, `clients_quiesced` | Swap the SM50 cubins and restart the grids (works while parked; weights are not reloaded). Recorded in `kernel_reload` | `status: ok` |
| `LOAD_CPU_MAILBOX_MANIFEST` (`load-cpu-mailbox --manifest M --clients-quiesced`) | `manifest_path`, `clients_quiesced` | Register a client's shared-memory result mailbox in every worker | `code: CPU_MAILBOX_LOADED` |
| `UNLOAD_CPU_MAILBOX_MANIFEST` (`unload-cpu-mailbox --clients-quiesced`) | `clients_quiesced` | Unregister the mailbox from every worker | `code: CPU_MAILBOX_UNLOADED` |
| `REGISTER_PEER_CLIENT` (adapter only) | `run_id`, `pid`, `journal` | Record that this client is about to acquire patched-UVM route records, journaled at `journal` | `code: PEER_CLIENT_REGISTERED` |
| `UNREGISTER_PEER_CLIENT` (adapter only) | `run_id` | Release ownership after a clean close; the journal must end in `CLOSED` (or `RECOVERED`) | `code: PEER_CLIENT_UNREGISTERED` |
| `RECOVER_PEER_CLIENT` (`recover-peer-client --clients-quiesced [--client-alive-ok]`) | `clients_quiesced`, optional `client_alive_ok` | After a client died or skipped its close: park workers, restore its journaled routes, unload its mailbox. The daemon keeps running | `code: PEER_CLIENT_RECOVERED`, `steps` |
| `LIST_ROUTE_RECORDS` (`list-route-records [--max-id N]`) | optional `max_id` (default 4096) | Read-only scan of the active patched-UVM multi-PTE records | `code: ROUTE_RECORDS`, `active`, `va_space_dead`, `records` |
| `STOP` (`stop [--force]`) | optional `force` | Drain and stop. **Refused while a peer client is registered** (`PEER_CLIENT_REGISTERED`) unless `force` | `status: STOP_ACCEPTED` |

Error codes: `CLIENTS_NOT_QUIESCED`, `WORKERS_NOT_READY`, `DIRECT_MAPPINGS_NOT_READY`, `INVALID_REQUEST`, `PEER_CLIENT_ALREADY_REGISTERED`, `PEER_CLIENT_NOT_REGISTERED`, `PEER_CLIENT_NOT_CLOSED`, `PEER_CLIENT_STILL_RUNNING`, `PEER_ROUTE_TOOL_MISSING`, `PARK_FAILED`, `ROUTE_RESTORE_FAILED`, `PEER_CLIENT_REGISTERED` (STOP guard), `KERNEL_RELOAD_FAILED`, `CPU_MAILBOX_LOAD_FAILED`.

### STATUS fields used by runners

| Field | Meaning |
|---|---|
| `status` | `STARTING`, `LOADING_PLE`, `LOADING_EXPERTS`, `READY`, `ERROR_CLEANUP`, `STOPPED` |
| `layers_ready`, `loaded_layer_ids` | resident expert layers |
| `worker_activity` | `ACTIVE` or `PARKED` |
| `kernel_reload` | last RELOAD_KERNELS: images, SHA-256, generation |
| `cpu_mailbox` | `{status: LOADED/UNLOADED, run_id, manifest, device_slots}` |
| `peer_client` | `{status: NONE}`, `{status: REGISTERED, run_id, pid, journal}`, or `{status: RECOVERY_FAILED, steps}`; `last` keeps the previous client's final record |
| `multicast_snapshot_root` | per-run PEX snapshot directory (a subdirectory of `Q38_MULTICAST_SNAPSHOT`) |
| `direct_moe_setup` | `status: PASS`, `gpu_map_records: 24` |
| `expert_load_seconds`, `stream_pack.layers[]` | load timing |

## Peer-client lifecycle (RTX ↔ M10 routes)

An RTX peer client (`qwen38_vllm_rtx_peer_adapter.py` + `libqwen38_vllm_rtx_peer.so`) installs routes as records in the patched UVM's **global** tables:

| Record | Where the PTE lives | Can be restored after the client dies? |
|---|---|---|
| RTX request leaves (PEX multicast publish, `RTX_PEX_PTE`) | client's own RTX address space | No: the space dies with the client (UVM lifecycle hook or `REAP_DEAD` releases the record) |
| M10 return leaves (`return_routes`) | M10 worker address spaces in the daemon's services | **Yes, while the daemon runs** |
| M10 aggregation routes (`agg_routes`) | M10 peer worker address spaces | **Yes, while the daemon runs** |
| Owner retains and DMA maps | kernel state, released by id | Yes |

These records are not tied to the client's file descriptor. If the client dies without closing, the M10-side records stay applied. They can be restored only while the daemon's services (and so the M10 page tables) still exist. Stopping the daemon first makes them unrecoverable until a UVM reload (or `REAP_DEAD`, when the loaded UVM has it).

Sequence for a peer client:

1. `REGISTER_PEER_CLIENT {run_id, pid, journal}`, **before** the library acquires any record. The daemon accepts one peer client at a time.
2. The library appends every acquired and released record to the journal (below).
3. Normal end: the client calls the adapter close **inside the vLLM engine process** (`llm.collective_rpc(_q38_close_peer_adapter)`), which parks the workers, restores every route, writes `CLOSED` and sends `UNREGISTER_PEER_CLIENT`. `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS=120` is a backstop: vLLM SIGKILLs the engine after this timeout, 5 s by default, which is what left routes behind on 2026-11-02.
4. Abnormal end (killed client, failed close): `RECOVER_PEER_CLIENT {clients_quiesced: true}`. The vLLM wrapper `run_qwen38_rtx_peer_pex_vllm_client.sh` calls it automatically when the cleanup proof fails and the peer path had opened. It restores ACS and unloads `rtx_mc_dma_map` only after the daemon reports `PEER_CLIENT_RECOVERED` and `peer_client.status == NONE`.

### RECOVER_PEER_CLIENT steps

1. Refuse unless a client is registered (`REGISTERED` or `RECOVERY_FAILED`) and `clients_quiesced` is set. Refuse if the registered pid still exists, unless `client_alive_ok`.
2. Park the workers (no M10 writes through the routes from here on).
3. Run `q38_peer_route_tool restore JOURNAL` (path given by `serve --peer-route-tool`). Output goes to `JOURNAL.recover.log`.
4. On tool failure: `peer_client.status = RECOVERY_FAILED`, the mailbox is kept, `STOP` stays refused, and the response is `ROUTE_RESTORE_FAILED`.
5. On success: unload the CPU mailbox if loaded, delete its `/dev/shm` object, set `peer_client = {status: NONE, last: {..., status: RECOVERED, steps}}`.

## Route journal

Path: `Q38_RTX_PEER_ROUTE_JOURNAL`, or by default `rtx_peer_routes.<pid>.journal` next to the client's `rtx_peer_timing` output. Every line is fsync'd before the record is used. The format is tab-separated:

| Line | Fields | Written when |
|---|---|---|
| `OPEN` | `pid`, pid | journal opened (library open or tool run) |
| `OWNER` | map_id, requester uuid (hex), owner uuid | owner retain succeeded |
| `DMA` | map_id, route_id, requester uuid, owner uuid | DMA map succeeded |
| `PTE` | mapping_id, gpu uuid, target_va, target_pte_phys, temporary_pte, original_pte, owner map_id (0 for RTX PEX leaves) | guarded PTE apply succeeded |
| `PTE_RESTORED` | mapping_id | restore verified |
| `DMA_UNMAPPED` | map_id, route_id | DMA unmap succeeded |
| `OWNER_RELEASED` | map_id | owner release succeeded |
| `CLOSED` | optional `rollback` | library close (or open rollback) completed |
| `RECOVERED` | – | `q38_peer_route_tool restore` finished without a FAILED record |

## q38_peer_route_tool

Built from `src/q38_peer_route_tool.cu` by `run_qwen38_full_daemon_start.sh` into `$RUN/build/`. It must run as root (the ioctls need `CAP_SYS_RAWIO`).

| Mode | What it does |
|---|---|
| `scan [MAX_ID]` | Read-only: `QUERY` of ids 1..MAX_ID, one `ACTIVE_RECORD` line per live record with `va_space_state` (2 = DEAD), then `SCAN_DONE` with the kernel's active count |
| `restore JOURNAL` | Restore unreleased PTEs newest first through the kernel's guarded `RESTORE`. That call re-finds the live GPU VA space and writes the saved original only if the PTE still holds the temporary value. If no PTE ends `FAILED`, it then unmaps DMA and releases owners. Exit 0, or 3 on a FAILED PTE (DMA and owner records are then left untouched) |

PTE outcomes:

| Outcome | Meaning |
|---|---|
| `RESTORED` | Original written back and read back |
| `NOT_ACTIVE` | No such record (already restored) |
| `VA_SPACE_GONE` | Nothing written; the record stays in the table |
| `PTE_CHANGED` | The PTE no longer holds our temporary value; nothing written |
| `FAILED` | Anything else; blocks the DMA and owner releases |

## Tests

| Test | Covers |
|---|---|
| `src/test_qwen38_daemon_peer_recovery.py` | Offline (no GPU): the real control server with a fake manager and fake tool. Register/second-register refused, STOP guard, unregister needs CLOSED, recovery refuses a live client, parks before restoring, keeps the mailbox on tool failure, unloads it on success, LIST_ROUTE_RECORDS. 18/18 on 2026-11-02 |
| `q38_peer_route_tool restore` on a journal with nonexistent ids | Parser and `NOT_ACTIVE` path against the loaded UVM (kernel returns before any page-table access). Run 2026-11-02, kernel active count unchanged |

## PCI-cluster exchange (worker request modes)

Every RTX → M10 request uses one pattern: the RTX **broadcasts** one request to all 24 workers (PEX hardware multicast: identical bytes on every worker of a card), every worker runs the op on its own resident data and writes its part into its direct result record, and the RTX **waits** for all 24 acknowledgements and **combines** the parts. RTX side: `pci_cluster_*` kernels in `src/qwen38_vllm_rtx_peer.cu` (one skeleton, one device sequence counter for all ops). Worker side: `src/pci_cluster_exchange.cuh` plus the mode bodies in `resident_dynamic_routed_v3.cuh` / `resident_dynamic_shared_v2.cuh`.

| Mode | Op | Request | Worker result | RTX combine |
|---|---|---|---|---|
| 1 | MoE (`PciClusterMoeOp`) | h, 10 expert ids, 10 routes | its experts' partial sum | Σ routed + gate·Σ shared |
| 4 | PLE gather, CPU-driven (direct client) | per-worker `ple_offsets` (unicast) | rows | host copies rows |
| 6 | Row gather (`PciClusterRowGatherOp`) | `ple_count` n ≤ 508 and `ple_offsets[r] = (owner << 40) \| byte offset`, broadcast in control page 0 | only its own rows, compacted in request order (≤ 64), count in `gate` | row i = slot[owner] at that worker's rank; counts cross-checked |
| 7 | LM-head top-k (`PciClusterLmHeadOp`) | x (hidden, as MoE), `ple_count` = K (1–2559), `ple_offsets[2w]` / `[2w+1]` = worker w's slice offset / rows | logits of its slice (FP32 accumulate, BF16 rounding), exact local top-K: `[count, (bf16 << 16) \| local index …]`, slice log-sum-exp in `gate` | logits row = −inf, candidates scattered to `row_start + index`; counts checked (reason 10); global log-sum-exp |
| 8 | Batched MoE (`PciClusterMoeBatchOp`) | doorbell only: `ple_count` = R (1–64); x [R][2560] (FP32 row stride; BF16 values when the format word is `Q38_BATCH_FMT_BF16`, as the RTX sends since k8), ids and routes [R][10] in the host batch area of the shared CPU mailbox | expert-major over the R rows; card leaders (routed) and worker 20 (shared) leave reduced rows [R][2560] FP32 in their part areas (shared: plus gate [R]), then a flag = seq; every worker acknowledges its slot as in mode 1 | row r = Σ routed reducers in worker order + gate[r] · shared reducer; bit-identical to R mode-1 calls |
| 9 | Pipelined batched MoE (`q38_rtx_peer_moe_batch_pipe_gpu`) | doorbell only (`control[2]` = 9, then seq); one call of R rows becomes n = ⌈R/32⌉ chunks with consecutive sequences; chunk seq uses slot seq & 1 of two 32-row slots in the mode-8 batch area; per-slot header {fmt, rows, layer, seq} | as mode 8 for the slot's rows; workers always take `last + 1` (the doorbell may already show `last + 2`); card peers wait for the leader's final credit `token(seq, R)` before acknowledging, so the slot is free for chunk seq + 2 | as mode 8 per chunk; the RTX waits for every ack ≥ seq (wrap-safe) before combining chunk n; bit-identical to mode 8 and mode 1 |
| 10 | Multi-row decode MoE (`PciClusterMoe2Op`, `q38_rtx_peer_moe2_gpu`; MTP verify) | R = 1..4 BF16 rows through the multicast pages: `ple_count` = R, `reserved` = 1; row r's x / ids / routes at `q38_mr_x_off` / `q38_mr_ids_off` / `q38_mr_routes_off` (`resident_multirow_abi.cuh`; rows 2–3 use 3 more aliased pages 0xC000–0xEFFF) | per row mode 1's arithmetic; card peers push rows 1..R−1 packed (10,244 B each) into the leader block `Q38_MR_PEER_BASE(k)`; reducers write row r to return record r (row-0 record ± r·3 pages); every worker acknowledges in its row-0 slot | per row PciClusterMoeOp's combine; checks every reducer's row-r record header (reason 12); bit-identical to R mode-1 calls |

Mode 6 per-worker configuration (written once per worker by `q38_rtx_peer_configure_row_gather` through the owner BAR mapping while the workers are parked; previous words saved, readback verified, restored and verified at peer close, `PCI_CLUSTER_ROW_GATHER_TABLE` / `PCI_CLUSTER_ROW_GATHER_RESTORE` lines): `cpu_mailbox_worker` = peer worker index (set by the peer open), `cpu_mailbox_reserved` carries `Q38_CPU_MAILBOX_FLAG_RTX_BAR1_RETURN`, `ple_reserved[1..2]` = this worker's table size in bytes (from the PLE manifest DEVICE row). Shard → worker mapping is by PCI address from the PLE manifest; nothing is hardcoded.

**Open item:** if the client dies without closing, `RECOVER_PEER_CLIENT` does not yet restore `ple_reserved[1..2]` (the route journal has no record type for it). The words are read only by mode 6, and a stale size is still a true upper bound of that worker's own table. The saved values are in the client's `PCI_CLUSTER_ROW_GATHER_TABLE` log lines.

**Multi-row decode MoE (mode 10, 2026-11-04).** Opened with `Q38_RTX_PEER_MOE_MULTIROW=1` and `Q38_RTX_PEER_MOE_MULTIROW_MAX=N` (2..4): the peer open sizes each reducer's return route to 3N pages (backwards from the slot for workers 23, and 22 if N ≥ 3, whose next VA pages are the mode-8 request area) and adds one route per card peer of ⌈(N−1)·10,244 / 4 KiB⌉ pages for its packed rows 1..N−1. Each worker gets a multi-row block at control +0xB040 {magic `Q38_MR_MAGIC`, role | N << 8, record step, ret1_va / push1_va}, saved and restored like the aggregation block; the workers refuse R > N. At N = 4: 54 of the module's 64 routes, 318 route pages and 42 request pages. The worker kernels keep the 1-row register allocation by putting mode 10 in non-inlined functions; the leaders' row blocks sit at 0x0F000, 0x17000 and 0x33000, the worker timing ring moved to 0x3C000 (256 records).

**LM head on the M10s (mode 7).** With `PLE_LM_HEAD_FILE` (set by `run_qwen38_full_daemon_start.sh`, default `$M10DIS_DATA/qwen38-lm-head/<index-sha16>/lm_head_bf16.bin` from `build_qwen38_lm_head_raw.py`, SHA-256 and checkpoint index checked before start) the PLE owner appends one contiguous row slice of `lm_head.weight` (10,346–10,347 rows, ≈53 MB) plus a 4-byte-per-row scratch to the **end** of every M10's PLE allocation, full-readback verified, and writes `LMHEAD cuda_ordinal bdf offset rows row_start row_bytes` manifest rows. The worker accepts a request only if its slice plus scratch end exactly at its table end (exit reason 4096 otherwise; 2048 for bad K / config), so its scratch writes cannot reach PLE rows. One exchange per sampled row returns 24 × K candidates and 24 log-sum-exps: exact for greedy, top-k ≤ K, top-p inside the candidates and logprobs of returned tokens; no full-vocabulary logits and no positive logit bias. Test: `src/run_pci_cluster_lm_topk_test.sh`.

**Batched MoE (mode 8).** Design and measurements: [research/BATCHED_MOE_EXCHANGE.md](research/BATCHED_MOE_EXCHANGE.md). The CPU mailbox shm grew from 24 slots (384 KiB) to `Q38_CPU_MAILBOX_BYTES` = 16,887,808 B: after the unchanged slots comes the batch area (`resident_cpu_mailbox_abi.cuh`: request x/ids/routes, gate page, 24 part areas of 64 rows + a flag page). The services register the whole region in every worker and write its worker VA into the new control field `batch_area` (region base + 384 KiB, cleared at mailbox unload); the routed service also allocates a 7.36 MB device scratch per GPU (`batch_scratch`, `batch_scratch_bytes`). These fields are appended to every control struct version, so earlier offsets are unchanged, but services and cubins must come from the same source (daemon restart). The RTX maps the same shm (`q38_rtx_peer_configure_moe_batch`, `cudaHostRegister` mapped, unregistered at peer close) and sends only the doorbell through PEX multicast, so the batch adds **no PTE route records**. Worker checks before any read: R in 1–64, `batch_area`, return flag, worker index, layer < 48, scratch size (exit reason **8192**); every id in [0, 512) and every route finite (**16384**). The RTX checks each reducer's flag against the request sequence (sticky error reason **11**, zero output). Dummy rows (an id outside [0, 512), as in mode 1) are sent as ids 0..9 with routes 0 and produce zero rows; a call of only dummy rows is skipped. Test: `src/run_q38_moe_batch_test.sh` (card 0 routed + card 5 shared, R = 1/17/64 against mode 1, invalid requests).

**Pipelined batched MoE (mode 9).** Overlaps the RTX publish and combine of one 32-row chunk with the M10 work on the next. Layout (`resident_cpu_mailbox_abi.cuh`, helpers in `resident_batch_layout.cuh`): `Q38_BATCH_PIPE_SLOTS` = 2 slots of `Q38_BATCH_PIPE_ROWS` = 32 rows inside the unchanged mode-8 batch area (no ABI size change, no daemon restart; kernel reload only). Slot s owns rows [32s, 32s + 32) of x (FP32 row stride; BF16 rows when the format word says `Q38_BATCH_FMT_BF16`), ids, routes, gate and every part area. Header of slot s at `Q38_BATCH_REQ_HDR` + 16·s = {fmt, rows, layer, seq}; slot 0 word 0 is the mode-8 format word. Flag and credit words of slot s at +32·s words in each flag page. Both sides derive the slot from the device sequence (seq & 1), never from the chunk index (t20b: an odd base sequence made all 24 workers read the wrong header, exit 8192). RTX stream order for n chunks: `begin(n) P0 D0 P1 D1 W0 C0 P2 D2 W1 C1 …` (P publish rows + header, D doorbell, W wait for all 24 acks ≥ seq, C combine). A slot is rewritten only after its previous chunk was combined, so at most two chunks are in flight. Worker checks: header seq must equal the request seq and 1 ≤ rows ≤ 32 (else exit **8192**); the rest as mode 8. Timing kinds 5 (routed) and 6 (shared). Client: `Q38_RTX_PEER_MOE_BATCH_PIPE=1` (calls of ≥ `Q38_RTX_PEER_MOE_BATCH_PIPE_MIN_ROWS`, default 33, rows; smaller calls use mode 8, single rows mode 1). Limit 4096 chunks per call.

**Return path.** The RTX never reads a worker's `direct` record during a peer session: the peer open points `cpu_mailbox_device_ptr` at a VA whose PTE is remapped onto that worker's RTX return slot (`Q38CpuMailboxSlot`). Every op therefore writes its result there and acknowledges with the MoE header convention (`worker = 0`, `layer = 1`, `status = 0`, then `seq`); mode 6 puts its owned-row count in `gate`. A worker error exit (any mode) with the RTX-return flag set also writes `status = 1`, the exit bits in `worker`, then `seq` to the slot, so the RTX reports reason 4 with the cause instead of a 2 s timeout (2026-11-03 r18: mode 6 first acknowledged only in `direct` and timed out).

Mode 6 checks run before the grid barrier, i.e. before any table read: table present, size set, return flag set, n ≤ 508 (else exit reason **512**); every owned row 4-byte aligned and inside the table, at most 64 owned (else **1024**). A failed check takes the normal worker error exit (status 1). RTX-side error reasons (sticky error word): 7 row outside every shard, 8 too many rows for one worker, 9 owned-count mismatch; 1–6 as for MoE (cycle timeout, %globaltimer deadline, abort, worker status, header, sequence space).

Worker startup (since 2026-11-03) clears `status`/`exit_reason` in both the host return struct and the direct result record before announcing `ready`: the services' relaunch only clears the host struct, so after an error exit and WAKE the RTX wait used to see a stale `exit_reason` (found by `test_pci_cluster_row_gather.cu`).

Tests: `src/run_pci_cluster_row_gather_test.sh` (both worker kernels on the RTX with a synthetic table, plain and compute-sanitizer memcheck; 2026-11-03 t2: 20/20 checks each, 0 memcheck errors). Hot swap: `src/run_pci_cluster_kernel_reload.sh` (park + RELOAD_KERNELS; no daemon stop).

## Settings

| Setting | Effect |
|---|---|
| `Q38_RTX_FP8_LINEAR_ATTN=1` | Linear-attention `in_proj_qkv`/`in_proj_z`/`out_proj` loaded as block FP8 (128x128, quantized at load by `qwen38_fp8_block_quant`, which reproduces Qwen's official FP8 byte for byte). Also sets `VLLM_GDN_DECODE_KERNEL=triton` (FP8 GDN projections hang the default cuda decode kernel under concurrency, per the primitive-ai Qwen3.8-Flash-Next mixed NVFP4/FP8 card). |
| `Q38_CPU_EMBEDDING=1`, `Q38_CPU_LM_HEAD=1` | Embedding table and LM head in host RAM |
| `Q38_VLLM_KV_CACHE_BYTES` | Fixed KV-cache reservation (vLLM `kv_cache_memory_bytes`) instead of a utilization percentage |
| `Q38_VLLM_CUDAGRAPH=1` | Breakable piecewise CUDA graphs for decode (vLLM `VLLM_USE_BREAKABLE_CUDAGRAPH`, `cudagraph_mode=PIECEWISE`, capture size 1, no torch.compile). The MoE exchange, PLE lookup and CPU embedding are eager breaks writing into cached static buffers (`q38_cudagraph_breaks.py`); everything between them is replayed from graphs. Prefill stays eager. |
| `Q38_VLLM_CUDAGRAPH_FULL=1` | With `Q38_VLLM_CUDAGRAPH=1`: `cudagraph_mode=FULL_AND_PIECEWISE`, and the FULL decode graph is captured by vLLM's `BreakableCUDAGraphWrapper` instead of the monolithic `CUDAGraphWrapper` (`install_full_breakable`; for the V2 runner that this engine uses, `CudaGraphManager.capture` is rebuilt from vLLM's source with two asserted edits: breakable capture instead of `torch.cuda.graph`, forward mode FULL instead of NONE). Logs `Q38_VLLM_CUDAGRAPH_FULL,captured,segments,N,eager_breaks,M` (r14: 3 segments, 2 breaks). FULL dispatch makes the GDN/QSA metadata builders use static decode buffers and runs vLLM's own breaks (GDN core, QSA, sparse indexer, PLE short conv) inline, so they are captured. Our host breaks stay eager (`_always_eager_break`); a q38 break inside a non-breakable capture raises instead of silently recording a no-op. |
| `Q38_RTX_PEER_MOE_BATCH=1` | With `Q38_RTX_PEER_GPU_DRIVEN=1`: MoE calls with at least `Q38_RTX_PEER_MOE_BATCH_MIN_ROWS` (default 2) rows (prefill chunks) go as one mode-8 exchange per 64 rows instead of one mode-1 exchange per row; decode (1 row) keeps mode 1. The client exports the mailbox manifest as `Q38_M10_RTX_PEER_MAILBOX_MANIFEST`. Needs services and worker kernels with mode 8 (daemon restart). |
| `Q38_RTX_PEER_MOE_BATCH_PIPE=1`, `Q38_RTX_PEER_MOE_BATCH_PIPE_MIN_ROWS` (default 33) | With `Q38_RTX_PEER_MOE_BATCH=1`: calls with at least the minimum rows go as one enqueue of 32-row mode-9 chunks (two host slots, RTX publish/combine overlapped with the M10 work). Needs worker kernels with mode 9 (kernel reload, no daemon restart). Off by default. |
| `VLLM_TRITON_FORCE_FIRST_CONFIG=1` | Disables Triton autotuning in vLLM (read in `vllm/env_override.py`): every autotuned kernel (flash linear attention `chunk_delta_h`, `chunk_o`, `solve_tril`, `wy_fast`, `l2norm`) uses its first config. Makes the autotuned kernels' config choice fixed. No measured speed cost. It does **not** make the end-to-end tokens deterministic: i12 (10 runs with it) still gave 4 of 10 runs with a different token at generated position 6. |
| `Q38_RTX_PEER_PLE_GPU=1` | With `Q38_RTX_PEER_GPU_DRIVEN=1` (and `Q38_CPU_EMBEDDING=1`): PLE rows come from the PCI-cluster row gather (worker mode 6, chunks of ≤ 64 rows) inside the CUDA graph instead of the CPU-driven PLE eager break; no `gpu_resync`. The GPU-driven sticky-error check then runs once per step in the CPU-embedding break (`qwen38_cpu_head_hooks.STEP_HOOKS`). Needs worker kernels with mode 6 (`run_pci_cluster_kernel_reload.sh`). |
| `Q38_EMBEDDING_ZERO_COPY=1` | With `Q38_CPU_EMBEDDING=1`: the host embedding table is pinned (`cudaHostRegister`, mapped, read-only; 1.27 GB, 0.19 s) and the lookup is a GPU gather that reads the rows over PCIe (`q38_zero_copy_gather` in the peer library), captured in the CUDA graph. With `Q38_RTX_PEER_PLE_GPU=1` the decode step is one graph with no eager break (r23). Per-step host hooks (sticky-error check, progress) then run in `compute_logits`; progress counts one step per call, so prefill chunks count as steps. Out-of-range ids give zero rows and are counted (`zero_copy.bad_ids` in the CPU-head timing file). Test: `src/run_q38_zero_copy_gather_test.sh`. |
| `Q38_M10_LM_HEAD=1`, `Q38_M10_LM_HEAD_TOPK` (default 1024, ≤ 2559) | With `Q38_RTX_PEER_GPU_DRIVEN=1` and a daemon started with the LM head (LMHEAD manifest rows): `ParallelLMHead` becomes `M10LmHead` (no weight; `lm_head.weight` is skipped before read, −1.27 GB on the RTX) and `compute_logits` runs one mode-7 exchange per sampled row. Exclusive with `Q38_CPU_LM_HEAD`. |
| `Q38_SKIP_NON_RTX_TENSORS` (default 1) | Skip experts, shared expert, PLE shards, MTP and vision tensors before vLLM reads them (identical 879 kept tensors) |
| `Q38_MTP_SPEC_TOKENS=k` | vLLM MTP speculative decoding (`method=mtp`, k draft tokens; the checkpoint's single MTP layer reused). The drafter's MoE is resident layer 48 on the M10s (daemon `Q38_DAEMON_LAYERS=49`), its LM head the M10 LM head, its embedding the host table; its attention, fc and hyper-connections load on the RTX. Also sets `disable_log_stats=False` (spec-decode counters in the result) and CUDA graph sizes 1 and 1 + k |
| `Q38_RTX_PEER_MOE_MULTIROW=1`, `Q38_RTX_PEER_MOE_MULTIROW_MAX` (2..4, default 2) | MoE calls of 2..max rows go as one mode-10 exchange (MTP verify) instead of mode 1 per row or mode 8 |
| `Q38_RTX_PEER_MR_STREAM` (default 1), `Q38_RTX_PEER_MOE_MULTIROW_MIN` (default 1) | Mode 10's streamed finish: each worker block pushes the elements it computed straight into the leader windows and sets one flag per block; each leader block waits only for its block's flags (no grid barrier between down and finish; bit-identical). MIN 1 also sends single-row decode calls through mode 10 (133.1 vs 141.3 µs per layer in mode 1) |
| `Q38_M10_LM_HEAD_ROWS2=1` | LM-head calls of 2..4 BF16 rows go as one mode-7 exchange (`reserved` = N; K capped at (2560 − N) // N − 1, e.g. 638 for 4 rows) |
| `Q38_DAEMON_LAYERS` (daemon start, 48 or 49) | 49 also loads the MTP layer's experts and shared expert as layer 48 (bundle `layer_48`, `src/prepare_mtp_layer_bundle.py`) |
| `Q38_RTX_PEER_PEX_OUTER=1` | Outer PEX8748 multicast group for the session ([TRANSPORT.md](TRANSPORT.md#8-optional-one-outer-multicast-group-q38_rtx_peer_pex_outer1)); the library gets the window as `Q38_RTX_PEER_PEX_OUTER_BASE` and publishes every request once |
| `Q38_RTX_PEER_PUBLISH_ORDER` (default 4) | 4: begin, publish and go of a mode-10 exchange in one RTX kernel (outer group only; without it as 0). 0: three kernels as before. 1–3: publish-order experiments ([TRANSPORT.md §9](TRANSPORT.md#9-what-limits-the-decode-exchange)): 1 request + go per card in parallel, 2 the same one card after another, 3 outer request + go per card; with `Q38_RTX_PEER_PUBLISH_SEQ` (card order, e.g. `0,1,2,3,4,5`), `Q38_RTX_PEER_GO_GAP_NS` (gap between cards), `Q38_RTX_PEER_GO_LOWFENCE=1` (order 3: sequence words only, one fence) |
| `Q38_RTX_PEER_ACK_PROFILE=1` | Measurement: the wait kernel first polls all 24 headers round-robin and records when the 9 data headers and all 24 were complete, and which header came last (timing record `pad`; harness fields `data9_ready`, `peers_after_data9`, `last_header_hist`; ≈ 5 µs per layer extra) |
| `Q38_RTX_PEER_MR_DIAG_NORTX=1` | Diagnostic, **results invalid**: card leaders and shared workers skip their result writes to the RTX (multi-row block flag `Q38_MR_FLAG_DIAG_NORTX`); measures the RTX link's share of the finish |
| `Q38_RTX_PEER_MR_PROFILE=1` | Card leaders record the phases of their finish in the worker timing record (`Q38_MR_FLAG_PROFILE`; harness fields `leader_prof_*`). Needs worker kernels built with `-DQ38_MR_PROFILE_BUILD` (otherwise ignored) |
| `Q38_RTX_PEER_MR_LL=1` | LL peer push for 1-row exchanges (`Q38_MR_FLAG_LL`, layout in `resident_multirow_abi.cuh`). Needs kernels built with `-DQ38_MR_LL_BUILD` and `Q38_RTX_PEER_MOE_MULTIROW_MAX=4` (the LL row lives in the rows ≥ 1 peer blocks). PoC, not faster overall (TRANSPORT.md §9) |
| `Q38_KV_TIER=1` | Host-memory K/V tier for the QSA layers, decode only ([plan](research/KV_HOST_TIER_PLAN.md)). `Q38_KV_TIER_TOKENS` sets the VRAM cache per layer (default 32768, multiple of 4). Turns on per layer when prefill ends, if the context is longer than the cache. |
| `Q38_KV_TIER_OWN=1` | With the tier, K/V leaves vLLM's pool, so context can exceed it (tested to 255k at 0 drafts). `Q38_KV_TIER_SUBROWS` sets the rows per attention sub-chunk (default 16). Set `Q38_VLLM_KV_CACHE_BYTES` low, or graph capture runs out of memory: 0.8 GiB at 262k and 0 drafts, 1.55 GiB at 2 drafts. With drafts, also set `VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=48` (fragmentation). |
| `Q38_KV_CONFIG_DUMP=FILE` | Saves the inputs of vLLM's K/V layout, for `analyze_kv_layout.py`. |
| `Q38_KV_TIER_VERIFY=1` | With the tier, eager decode only: compares the K/V of every selected entry with its true bytes. `Q38_KV_TIER_REPORT=FILE` writes per-layer hit rates and counts. |
| `Q38_QSA_SELECT_LOG=FILE` | Logs the indexer selections of decode rows, eager decode only (`Q38_VLLM_CUDAGRAPH=0`). `Q38_QSA_UNION_LOG=CSV` also logs distinct blocks per prefill call. Files get a `.PID` suffix. |
| `Q38_VLLM_IGNORE_EOS=1` | Generation mode always produces `--max-new-tokens` (equal-length speed comparisons) |
| `Q38_NSYS=1`, `Q38_CUDA_PROFILER_START_TOKENS`, `Q38_CUDA_PROFILER_TOKENS` | Run the client under `nsys profile --capture-range=cudaProfilerApi`; the adapter opens the range at that decode step for that many steps |
| `Q38_PYSPY_SECONDS` | Non-blocking py-spy record of the engine during steady decode |
| `Q38_PROGRESS_SECONDS` (default 10) | `Q38_PROGRESS` decode-rate line from the peer adapter |
| `VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS` (client default 120) | Backstop for the engine shutdown window; the adapter is closed explicitly before it |
