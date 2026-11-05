# Code map

Every file in `src/`, grouped by role. `src/` is flat because the runners and the Python modules find each other through `$REPO/src/<name>` and `Path(__file__).with_name(...)`. The file names keep the project's internal prefixes: `qwen38_` for the model integration, `q38_` for small tools, `resident_` for the M10 worker side.

## Runners (entry points, run as root through Pueue)

| File | Does |
|---|---|
| `m10dis_env.sh` | shared paths (`M10DIS_*`), sourced by every runner; reads `m10dis.local.env` |
| `run_qwen38_full_daemon_start.sh` | builds the M10 binaries from source, checks inputs and topology, starts the daemon, waits for 48 layers `READY` |
| `run_qwen38_daemon_restart.sh` | ordered `STOP` of the running daemon, then a fresh start |
| `run_pci_cluster_kernel_reload.sh` | rebuilds the worker cubins and hot-swaps them (`RELOAD_KERNELS`) |
| `run_qwen38_rtx_peer_vllm.sh` | one full vLLM run against the daemon, with every pre/post check |
| `run_qwen38_rtx_local_acs_trial.sh` | wrapper: relaxes and restores ACS around a peer session |
| `run_qwen38_rtx_peer_pex_vllm_client.sh` | inner step: DMA helper module, preload, the vLLM client |
| `run_qwen38_moe_batch_integration.sh` | the 4k-token prompt through vLLM once per requested mode (per row / batched / pipelined) |
| `run_qwen38_bench.sh` | benchmark run with the frozen serving configuration |
| `run_qwen38_decode_profile.sh` | decode profile / speed run with the frozen configuration: graph ceiling + per-exchange breakdown, optional py-spy and nsys, MTP and multi-row settings pass through |
| `run_bench_score_v2.sh` | scores the newest benchmark run (CPU) |
| `run_qwen38_rtx_pex_multicast.sh` | standalone check of CPU vs RTX publication into the six multicast windows |

## RTX side (vLLM process)

| File | Does |
|---|---|
| `run_vllm_resident_m10_client.py` | the vLLM client: daemon preflight, vLLM start, generation or benchmark mode, result JSON |
| `run_stage8_stage9_vllm_generation.py` | builds the vLLM engine with the M10 hooks installed |
| `sitecustomize.py` | loads the hooks in vLLM's worker processes |
| `qwen38_vllm_rtx_peer_adapter.py` | replaces the MoE forward; calls the peer library; owns open/close of the peer session |
| `qwen38_vllm_rtx_peer.cu` | **the RTX peer library**: publish, wait, combine kernels for every worker mode; route setup and restore |
| `build_qwen38_vllm_rtx_peer.sh` | builds it (called by the client) |
| `qwen38_rtx_fp8_hooks.py`, `qwen38_fp8_block_quant.py` | block-scaled FP8 for the GDN and QSA projections |
| `build_qwen38_fp8_cache.py` | precomputes those FP8 tensors once |
| `qwen38_hc_fp8.py` | FP8 hyper-connection matrices with Triton GEMM/GEMV kernels |
| `qwen38_cpu_head_hooks.py` | embedding in host RAM (zero-copy) and the LM head on the M10s |
| `q38_cudagraph_breaks.py` | eager breaks for vLLM's breakable CUDA graphs |
| `qwen38_direct_moe_adapter.py`, `qwen38_direct_resident_client.cpp`, `build_qwen38_direct_moe.sh` | the earlier CPU-driven direct path. The RTX-peer adapter still uses it for the direct n-gram (PLE) adapter when a PLE manifest is set |
| `qwen38_moe_rpc.py` | socket RPC to the daemon (control and fallback path) |
| `qwen38_cpu_mailbox_manifest.py`, `qwen38_gpu_mappings_manifest.py` | manifests of the shared host mailbox and the retained BAR1 mappings |
| `rm_ioctl_trace_preload.c`, `build_rm_trace_preload.sh` | LD_PRELOAD that captures the RM handles of CUDA allocations |

## M10 side (daemon and workers)

| File | Does |
|---|---|
| `qwen38_resident_daemon.py` | the daemon: load, park/wake, control socket, route journal, peer-client recovery |
| `qwen38_resident_service_manager.py` | host-side readiness and request gate for the worker services |
| `resident_routed_persistent_multicast_experimental.cu` | routed worker **service** (one process per routed GPU); built by `build_qwen38_resident_services.sh` |
| `resident_shared_persistent_multicast_experimental.cu` | shared worker service |
| `bench_resident_multilayer_fp8.cu`, `bench_resident_multilayer_shared.cu` | weight layouts and loaders included by the services and kernel modules |
| `resident_routed_kernel_v3_module.cu`, `resident_shared_kernel_v2_module.cu` | cubin entry points for `RELOAD_KERNELS` |
| `resident_dynamic_routed_v3.cuh` (+ `_v2`, base) | **routed worker kernel**: request poll, modes 1/6/7/8/9, card-leader sums |
| `resident_dynamic_shared_v2.cuh` (+ base) | shared worker kernel |
| `resident_routed_v2_math.cuh`, `resident_shared_v2_math.cuh` | gate/up/down math (FP8 decode, FP32 accumulate) |
| `resident_routed_batch.cuh`, `resident_batch_layout.cuh` | batched modes 8/9: staging, expert lists, tiles, host batch-area layout |
| `pci_cluster_exchange.cuh` | worker side of the generic exchange (row gather, LM-head top-K) |
| `resident_cpu_mailbox_abi.cuh`, `resident_cpu_mailbox.cuh`, `resident_direct_mailbox.cuh` | ABI of the shared host mailbox, return slots and request pages |
| `resident_multicast_mapping.cuh`, `resident_pdb_export.cuh`, `resident_pack_down_kernel.cuh` | BAR1 mapping, VA-space identity export, down-projection packing |
| `cuda_load_stage9_ple.cu`, `qwen38_parallel_load.cuh` | PLE owner: n-gram shards and LM-head slices, parallel verified loading, CUDA IPC export |
| `qwen38_m10_expert_sim.py` | layer streaming and packing helpers used by the daemon |
| `resident_multirow_abi.cuh` | layout of the multi-row decode MoE (mode 10, up to 4 rows) and the per-worker multi-row block |
| `pex_group_config.py` | programs, checks and restores the PEX8747 multicast registers |
| `check_qwen38_mc_premap.py` | read-only multicast check before the daemon starts |
| `q38_peer_route_tool.cu`, `qwen38_route_journal.h` | route records: journal, inspection, restore |
| `bench_qwen38_rtx_m10_retained_roundtrip.cu` | retained-route helpers shared with the peer library build |
| `bench_official_fp8.cu`, `bench_shared_bf16.cu` | reference kernels used by `qwen38_m10_expert_sim.py` |

## Discovery and preparation

| File | Does |
|---|---|
| `qwen38_pci_discover.py`, `qwen38_pci_env.sh` | finds the RTX, the M10s and their switch ports from sysfs |
| `discover_m10_collective_topology.py`, `prepare_six_m10_groups.py` | M10 card groups and their multicast windows → `topology.tsv` |
| `prepare_resident_expert_layer.py`, `run_stage7_full_residency.py`, `preserve_stage7_bundles.py`, `capture_stage7_manifests.py`, `combine_stage7_ranges.py` | expert-slice bundles |
| `prepare_stage9_ple_manifest.py`, `extract_qwen38_ple_shards.py`, `cuda_stage9_placement_capacity.cu` | n-gram table placement and shards |
| `build_qwen38_lm_head_raw.py` | raw LM-head file for the M10 slices |
| `prepare_mtp_layer_bundle.py` | packs the MTP layer's experts and shared expert as resident layer 48 |

## Benchmarks and analysis

| File | Does |
|---|---|
| `build_bench_items.py`, `order_bench_items.py` | benchmark item files and their fixed order |
| `score_lcb.py`, `score_mmlu.py`, `score_humaneval.py` | scorers |
| `lm_head_floor.py` | LM-head jitter floor of repeated runs |
| `bench_m10_expert_compute.cu` | 1-row routed gate/up + down on one M10 with synthetic FP8 weights: the deployed code (same headers as the worker cubin) against variants with more loads in flight or more blocks, bit-identity checked; `bench_m10_expert_compute OUT_JSON [iters]`, runs beside a parked daemon (≈ 180 MB on one GPU) |

## Tests

| File | Needs |
|---|---|
| `test_qwen38_pci_discover.py`, `test_qwen38_daemon_peer_recovery.py` | nothing (sysfs read only / fakes) |
| `test_qwen38_fp8_block_quant.py`, `test_qwen38_cpu_head_hooks.py` | the checkpoint, CPU only |
| `test_qwen38_hc_fp8.py`, `test_q38_zero_copy_gather.py` (+ runner) | the RTX |
| `test_q38_moe_batch.cu`, `test_pci_cluster_row_gather.cu`, `test_pci_cluster_lm_topk.cu` (+ `run_*_test.sh`) | M10 cards with the daemon stopped (or the RTX for the row-gather variant) |
| `qwen38_routed_v2_math_test.cu`, `qwen38_shared_v2_math_test.cu` | one GPU; worker math against a double-precision CPU reference |
