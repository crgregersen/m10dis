#!/usr/bin/env bash
# Decode profile (2026-11-04): one vLLM peer run (run_qwen38_rtx_peer_vllm.sh: route and cleanup checks) in
# generation mode with the frozen benchmark configuration (run_qwen38_bench.sh), on the ~4k-token prompt.
# It collects, per run:
#   - the graph ceiling and the per-exchange breakdown of a replayed decode step (Q38_GRAPH_CEILING_ITERS,
#     Q38_GRAPH_BREAKDOWN_STEPS; RTX time before each exchange, publish, round trip, combine), after generation
#   - optionally a py-spy sample of the engine during steady decode (Q38_PYSPY_SECONDS, default 5)
#   - optionally an Nsight Systems trace of steady decode (Q38_NSYS=1; decode tokens Q38_CUDA_PROFILER_START_TOKENS
#     (default 500) to +Q38_CUDA_PROFILER_TOKENS (default 200), CUDA graphs per node)
#   Q38_PROFILE_NEW_TOKENS  decode length (default 1500, long enough for the samplers to start in steady decode)
# No retry on failure. Queue: pueue add --group gpu --label <label> -- sudo -n env ... bash <this script>
set -Euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/m10dis_env.sh"
ROOT=$M10DIS_HOME
REPO=$M10DIS_HOME
OUT=$M10DIS_RUNS/decode-profile/$(date -u +%Y%m%dT%H%M%SZ)
RUNS=$M10DIS_RUNS/qwen38-vllm-rtx-peer-optimized
[[ $EUID -eq 0 ]] || { echo 'ERROR,run as root (sudo -n)'; exit 2; }
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
PROMPT=${Q38_LONG_PROMPT:-$ROOT/prompts/long_prompt_4k.txt}
[[ -r $PROMPT ]] || { echo "ERROR,no prompt $PROMPT"; exit 3; }
cp "$PROMPT" "$OUT/prompt.txt"
echo "PROFILE_RUN_DIR,$OUT,pyspy_s,${Q38_PYSPY_SECONDS:-5},nsys,${Q38_NSYS:-0},new_tokens,${Q38_PROFILE_NEW_TOKENS:-1500}"
before=$(ls -td "$RUNS"/2026*/ 2>/dev/null | head -1 || true)
env Q38_VLLM_CPU_OFFLOAD_GB=0 Q38_VLLM_GPU_MEMORY_UTILIZATION=0.95 Q38_CPU_EMBEDDING=1 Q38_CPU_LM_HEAD=0 \
  Q38_RTX_FP8_LINEAR_ATTN=1 Q38_RTX_FP8_QSA=1 Q38_RTX_FP8_CACHE=$M10DIS_DATA/qwen38-fp8-cache/0419e2c2dfbb9252-qsa \
  Q38_RTX_FP8_HC=1 Q38_RTX_PEER_GPU_DRIVEN=1 Q38_VLLM_CUDAGRAPH=1 Q38_VLLM_CUDAGRAPH_FULL=1 \
  Q38_RTX_PEER_PLE_GPU=1 Q38_EMBEDDING_ZERO_COPY=1 Q38_M10_LM_HEAD=1 Q38_M10_LM_HEAD_TOPK=1024 \
  Q38_VLLM_REPEATS=1 Q38_PROGRESS_SECONDS=10 Q38_VLLM_MAX_NEW_TOKENS=${Q38_PROFILE_NEW_TOKENS:-1500} \
  Q38_VLLM_PREFIX_CACHING=0 Q38_VLLM_KV_CACHE_BYTES=$((5 << 29)) Q38_VLLM_MAX_MODEL_LEN=${Q38_VLLM_MAX_MODEL_LEN:-90112} \
  VLLM_SPARSE_INDEXER_MAX_LOGITS_MB=128 VLLM_TRITON_FORCE_FIRST_CONFIG=1 \
  Q38_VLLM_MAX_BATCHED_TOKENS=1024 Q38_RTX_PEER_MOE_BATCH=1 Q38_RTX_PEER_MOE_BATCH_PIPE=1 \
  Q38_GRAPH_CEILING_ITERS=${Q38_GRAPH_CEILING_ITERS:-50} Q38_GRAPH_BREAKDOWN_STEPS=${Q38_GRAPH_BREAKDOWN_STEPS:-50} \
  Q38_PYSPY_SECONDS=${Q38_PYSPY_SECONDS:-5} Q38_NSYS=${Q38_NSYS:-0} \
  Q38_VLLM_PROMPT_FILE="$OUT/prompt.txt" \
  bash "$REPO/src/run_qwen38_rtx_peer_vllm.sh" > "$OUT/runner.out" 2>&1
rc=$?
dir=$(ls -td "$RUNS"/2026*/ 2>/dev/null | head -1 || true); [[ $dir == "$before" ]] && dir=""
echo "PEER_RUN,rc,$rc,dir,$dir"
tail -3 "$OUT/runner.out"
[[ -n $dir ]] && ln -sfn "$dir" "$OUT/peer-run"
# summary: decode rate from the progress lines, graph ceiling, per-step breakdown
python3 - "$dir" <<'PY'
import json, pathlib, re, sys
d = pathlib.Path(sys.argv[1]) if sys.argv[1] else None
if d is None:
    print("PROFILE_SUMMARY,no_run_dir"); sys.exit(0)
log = (d / "client/client.log").read_text(errors="replace") if (d / "client/client.log").exists() else ""
prog = [(int(m.group(1)), m.group(0)) for m in re.finditer(r"Q38_PROGRESS,.*?decode_tokens,(\d+),[^\n]*", log)]
print(f"PROFILE_PROGRESS_LINES,{len(prog)}")
for _, line in prog[-3:]:
    print(line[:240])
try:
    r = json.loads((d / "client/vllm/resident_vllm_client_result.json").read_text())
except Exception as exc:
    print(f"PROFILE_SUMMARY,no_client_result,{exc}"); sys.exit(0)
print(f"PROFILE_CLIENT_STATUS,{r.get('status')}")
for g in r.get("generation", []):
    print(f"PROFILE_GENERATION,tokens,{len(g.get('generated_token_ids', []))},prompt,{len(g.get('prompt_token_ids', []))},elapsed_s,{g.get('elapsed_seconds')}")
def walk(x):
    if isinstance(x, dict):
        if "graph_only_ms" in x: return x
        for v in x.values():
            w = walk(v)
            if w: return w
    if isinstance(x, list):
        for v in x:
            w = walk(v)
            if w: return w
c = walk(r.get("graph_ceiling"))
if not c:
    print(f"PROFILE_CEILING,missing,{json.dumps(r.get('graph_ceiling'))[:300]}"); sys.exit(0)
print("PROFILE_CEILING," + ",".join(f"{k},{c.get(k)}" for k in ("graph_only_ms", "graph_lm_argmax_ms", "graph_lm_argmax_sync_ms", "exchange_errors")))
b = c.get("breakdown") or {}
print(f"PROFILE_BREAKDOWN,step_us,{b.get('step_us')},per_step_us,{json.dumps(b.get('per_step_us'))}")
ex = b.get("per_exchange_us") or {}
for k in ("lm_head", "moe_00", "moe_01", "moe_02", "moe_03"):
    if k in ex: print(f"PROFILE_EXCHANGE,{k},{json.dumps(ex[k])}")
others = [k for k in ex if not k.startswith("moe_") and k != "lm_head"]
for k in others: print(f"PROFILE_EXCHANGE,{k},{json.dumps(ex[k])}")
PY
echo "PROFILE_DONE,rc,$rc,$OUT"
exit $rc
