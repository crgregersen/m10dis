#!/usr/bin/env bash
# Benchmark pass (2026-11-04): one vLLM peer run (run_qwen38_rtx_peer_vllm.sh: route and cleanup checks) with the
# client's benchmark mode (Q38_BENCH_JSONL, run_vllm_resident_m10_client.py) on the current default configuration:
# pipelined batched MoE (mode 9, prefill chunk 1024), FP8 GDN + QSA + hyper-connections, M10 LM head, no prefix
# caching, KV 2.5 GiB (94,631 tokens), max_model_len 90112, QSA indexer workspace 128 MiB.
#   Q38_BENCH_ITEMS   item file (build_bench_items.py), required
#   Q38_BENCH_NAME    label for the output folder (default: item file name)
#   Q38_BENCH_OUT     results JSONL; set it to an existing file to resume (finished items are skipped)
#   other Q38_BENCH_* settings pass through (THINKING, EFFORT, MAX_TOKENS, TEMPERATURE, TOP_P, TOP_K, LIMIT, IDS,
#   DEADLINE = unix time after which no new item starts)
# No retry on failure. Queue: pueue add --group gpu --label <label> -- sudo -n env Q38_BENCH_ITEMS=... bash <this>
set -Euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/m10dis_env.sh"
ROOT=$M10DIS_HOME
REPO=$M10DIS_HOME
ITEMS=${Q38_BENCH_ITEMS:?Q38_BENCH_ITEMS is required}
NAME=${Q38_BENCH_NAME:-$(basename "$ITEMS" .jsonl)}
OUT=$M10DIS_RUNS/qwen38-bench/$NAME-$(date -u +%Y%m%dT%H%M%SZ)
[[ $EUID -eq 0 ]] || { echo 'ERROR,run as root (sudo -n)'; exit 2; }
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
RESULTS=${Q38_BENCH_OUT:-$OUT/results.jsonl}
cp "$ITEMS" "$OUT/items.jsonl"
sha256sum "$ITEMS" > "$OUT/items.sha256"
echo "BENCH_RUN_DIR,$OUT,items,$ITEMS,results,$RESULTS"
env Q38_VLLM_CPU_OFFLOAD_GB=0 Q38_VLLM_GPU_MEMORY_UTILIZATION=0.95 Q38_CPU_EMBEDDING=1 Q38_CPU_LM_HEAD=0 \
  Q38_RTX_FP8_LINEAR_ATTN=1 Q38_RTX_FP8_QSA=1 Q38_RTX_FP8_CACHE=$M10DIS_DATA/qwen38-fp8-cache/0419e2c2dfbb9252-qsa \
  Q38_RTX_FP8_HC=1 Q38_RTX_PEER_GPU_DRIVEN=1 Q38_VLLM_CUDAGRAPH=1 Q38_VLLM_CUDAGRAPH_FULL=1 Q38_PYSPY_SECONDS=0 \
  Q38_RTX_PEER_PLE_GPU=1 Q38_EMBEDDING_ZERO_COPY=1 Q38_M10_LM_HEAD=1 Q38_M10_LM_HEAD_TOPK=1024 \
  Q38_VLLM_REPEATS=1 Q38_PROGRESS_SECONDS=60 Q38_VLLM_MAX_NEW_TOKENS=8 \
  Q38_VLLM_PREFIX_CACHING=0 Q38_VLLM_KV_CACHE_BYTES=$((5 << 29)) Q38_VLLM_MAX_MODEL_LEN=${Q38_VLLM_MAX_MODEL_LEN:-90112} \
  VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128 VLLM_TRITON_FORCE_FIRST_CONFIG=1 \
  Q38_VLLM_MAX_BATCHED_TOKENS=1024 Q38_RTX_PEER_MOE_BATCH=1 Q38_RTX_PEER_MOE_BATCH_PIPE=1 \
  Q38_BENCH_JSONL="$OUT/items.jsonl" Q38_BENCH_OUT="$RESULTS" \
  bash "$REPO/src/run_qwen38_rtx_peer_vllm.sh"
rc=$?
# artifact verification (night-run rule 4): results parse, no duplicate ids, every id from the item set, >= 1 row
python3 - "$RESULTS" "$OUT/items.jsonl" <<'PY'
import json, sys
rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
items = {json.loads(l)["id"] for l in open(sys.argv[2]) if l.strip()}
ids = [r["id"] for r in rows]
gen = [r for r in rows if r.get("kind") == "generate" and "output_tokens" in r]
cho = [r for r in rows if r.get("kind") == "choice"]
print(f"BENCH_SUMMARY,results,{len(rows)},of,{len(items)},generate,{len(gen)},choice,{len(cho)},"
      f"capped,{sum(1 for r in gen if r['finish_reason'] == 'length')},"
      f"tokens,{sum(r.get('output_tokens', 0) for r in rows)},seconds,{sum(r.get('elapsed_seconds', 0) for r in rows):.0f}")
bad = []
if not rows: bad.append("no results")
if len(ids) != len(set(ids)): bad.append(f"{len(ids) - len(set(ids))} duplicate ids")
if any(i not in items for i in ids): bad.append("ids outside the item set")
if cho and len({r.get("pred") for r in cho}) == 1 and len(cho) > 20: bad.append("all choice predictions equal")
print("BENCH_VERIFY," + ("OK" if not bad else "FAIL," + ";".join(bad)))
sys.exit(1 if bad else 0)
PY
vrc=$?
[[ $rc -eq 0 && $vrc -ne 0 ]] && rc=4
echo "BENCH_DONE,rc,$rc,$OUT"
exit $rc
