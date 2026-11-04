# Preparing the inputs

The runners read their inputs from three places, all set in `src/m10dis_env.sh` and overridable in `m10dis.local.env` at the repo root (not committed):

| Variable | Default | Holds |
|---|---|---|
| `M10DIS_DATA` | `/mnt/nvme` | large prepared files (fast local disk) |
| `M10DIS_MODEL` | `$M10DIS_DATA/models/Qwen3.8-Flash-Next-FP8` | the checkpoint |
| `M10DIS_STATE` | `$M10DIS_DATA/m10dis-state` | small machine-specific inputs: topology, placement, swap log, references |
| `M10DIS_RUNS` | `<repo>/runs` | one folder per run |
| `M10DIS_BENCH_DATA` | `$M10DIS_DATA/benchmarks` | benchmark datasets and item files |

Disk use on the test machine: checkpoint 173 GB, expert bundles 120 GB, n-gram shards 48 GB, FP8 cache 4.5 GB, LM head 1.3 GB.

## 1. Checkpoint

Download Qwen3.8-Flash-Next-FP8 into `$M10DIS_MODEL`. Every preparation step records the SHA-256 of the checkpoint index it read, and the runners check it.

## 2. PCIe topology (`$M10DIS_STATE/topology.tsv`)

```bash
sudo python3 src/prepare_six_m10_groups.py "$M10DIS_STATE/topology.tsv"
```

The script scans sysfs (`src/discover_m10_collective_topology.py`) for six complete M10 cards: a PEX8747 upstream port, four downstream ports and four GM107 each. For every card it picks an unused, 64 KiB-aligned multicast ingress window inside the card's prefetchable bridge window. One row per card: board serial, PEX8747 upstream port, its four downstream ports, the four GPUs, window start/end and ingress address.

Re-run it after any hardware change. The daemon start re-checks every row against a live scan and refuses a stale file. No PCI address is hardcoded anywhere: everything is found from vendor/device IDs and the sysfs tree.

## 3. Expert bundles (`$M10DIS_DATA/qwen38-stage9-preserved-bundles`)

Each routed GPU needs its 32-row slice of every expert, per layer, in the worker's layout (FP8 + scales), and each shared GPU its part of the shared expert:

- `src/prepare_resident_expert_layer.py --layer N --out DIR` packs one layer's 512 experts into per-GPU slice files.
- `src/run_stage7_full_residency.py` streams all 48 layers through packing, loads them onto the M10s and validates them against the checkpoint.
- `src/preserve_stage7_bundles.py` keeps the finished layer bundles (hard links) before staging is cleaned up, and `src/capture_stage7_manifests.py` / `src/combine_stage7_ranges.py` collect their manifests.

Copy the result to fast local disk. The daemon start checks that `layer_00` and `layer_47` manifests are present and reads per-layer manifests with their hashes.

## 4. N-gram embedding shards and placement

```bash
python3 src/prepare_stage9_ple_manifest.py "$M10DIS_MODEL" "$M10DIS_STATE/stage9-placement"     # memory-balanced 24-GPU plan
python3 src/extract_qwen38_ple_shards.py "$M10DIS_STATE/stage9-placement/ple_shards.tsv" "$M10DIS_DATA/qwen38-ple-shards"   # 128 × 400 MB + extract-report.json
```

`prepare_stage9_ple_manifest.py` maps the checkpoint's n-gram embedding tensors to a plan that balances memory over the 24 GM107: `gpu_allocations.tsv` (UUID, size per GPU) and `ple_shards.tsv` (byte ranges). `src/cuda_stage9_placement_capacity.cu` measures how much each GPU can actually allocate next to the expert slices. `extract_qwen38_ple_shards.py` cuts the ranges into standalone shard files and writes `extract-report.json` (`all_match: true` is required by the daemon start).

Put the placement folder at `$M10DIS_STATE/stage9-placement` (`gpu_allocations.tsv`, `ple_query_rows.txt`, manifests). The daemon start refreshes each row's PCI address from its GPU UUID, so the plan survives re-enumeration.

## 5. LM-head slices

```bash
.venv-vllm/bin/python src/build_qwen38_lm_head_raw.py "$M10DIS_MODEL" "$M10DIS_DATA/qwen38-lm-head/<index-sha16>"
```

It writes `lm_head.weight` as one raw BF16 row-major file, `lm_head_bf16.bin`, into the output folder (the runners expect `$M10DIS_DATA/qwen38-lm-head/<index-sha16>/`, where `<index-sha16>` is the first 16 hex digits of the checkpoint index SHA-256). The daemon's PLE owner appends one row slice per GPU to its allocation. `Q38_LM_HEAD_FILE=` (empty) disables the M10 LM head.

## 6. FP8 cache for the RTX layers

```bash
.venv-vllm/bin/python src/build_qwen38_fp8_cache.py "$M10DIS_MODEL" "$M10DIS_DATA/qwen38-fp8-cache/<index-sha16>-qsa" --qsa
```

It quantizes the GDN and QSA projection weights to block-scaled FP8 once, instead of at every vLLM start. `Q38_RTX_FP8_CACHE` points the runners at it. The FP8 hyper-connection matrices are quantized at load time (`src/qwen38_hc_fp8.py`).

## 7. UVM swap log and references

- `$M10DIS_STATE/uvm-swap.log` is written by `transport/driver/swap-uvm.sh` ([RUNNING.md](RUNNING.md)).
- `$M10DIS_STATE/reference/humaneval32-direct.json` and `humaneval32-rtx-peer.json` are result files of earlier known-good runs of the default prompt. `run_qwen38_rtx_peer_vllm.sh` compares new runs against them and reports, but does not fail, on differences. The RTX kernels are not bit-deterministic (see [RESULTS.md](RESULTS.md#determinism)), so compare with the floor method, not with token hashes.

Run `--help` on any script for its options. Steps 3 and 4 were run once during development and not repeated from this tree; their inputs and outputs are recorded in each script's manifest output.
