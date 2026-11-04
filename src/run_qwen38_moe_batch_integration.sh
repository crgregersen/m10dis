#!/usr/bin/env bash
# Batched MoE (worker mode 8) integration: the same ~4k-token prompt through vLLM twice, per-row MoE
# (Q38_RTX_PEER_MOE_BATCH=0) and batched (=1), prefill chunk Q38_BATCH_CHUNK (default 64), 32 new tokens,
# KV 1.0 GiB, max_model_len 8192. Each run is a normal peer run (run_qwen38_rtx_peer_vllm.sh: route and
# cleanup checks). Expected: identical generated token ids (mode 8 is bit-identical to mode 1 per row);
# reports elapsed, prefill estimate and the adapter's batch counters.
# Needs the daemon started from source with mode 8 (services + cubins). No retry on failure.
# Queue: pueue add --group gpu --label <label> -- sudo -n bash <this script>
set -Euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/m10dis_env.sh"
ROOT=$M10DIS_HOME
REPO=$M10DIS_HOME
RUNS=$M10DIS_RUNS/qwen38-vllm-rtx-peer-optimized
OUT=$M10DIS_RUNS/batched-moe-research/integration-$(date -u +%Y%m%dT%H%M%SZ)
CHUNK=${Q38_BATCH_CHUNK:-64}
# Q38_BATCH_INTEGRATION_MODES (spaces or commas, default "0 1"): 0 per row, 1 batched (mode 8), 2 pipelined
# (mode 9, Q38_RTX_PEER_MOE_BATCH_PIPE=1, 2026-11-03)
MODES=$(echo "${Q38_BATCH_INTEGRATION_MODES:-0 1}" | tr ',' ' ')
[[ $EUID -eq 0 ]] || { echo 'ERROR,run as root (sudo -n)'; exit 2; }
mkdir -p "$OUT"
exec > >(tee -a "$OUT/run.log") 2>&1
echo "RUN_DIR,$OUT"
PROMPT=${Q38_LONG_PROMPT:-$ROOT/prompts/long_prompt_4k.txt}   # ~4k tokens (HumanEval prompts)
[[ -r $PROMPT ]] || { echo 'ERROR,no long prompt from the KV-capacity probe'; exit 3; }
cp "$PROMPT" "$OUT/long_prompt.txt"
CACHE=$M10DIS_DATA/qwen38-fp8-cache/0419e2c2dfbb9252-qsa
printf 'batch\trun_dir\tstatus\tprompt_tokens\telapsed_s\tprefill_s_est\tprefill_tok_s_est\tbatch_calls\tbatch_rows\tsha_tokens\n' > "$OUT/runs.tsv"
for batch in $MODES; do
  before=$(ls -td "$RUNS"/2026*/ | head -1)
  env Q38_VLLM_CPU_OFFLOAD_GB=0 Q38_VLLM_GPU_MEMORY_UTILIZATION=0.95 Q38_CPU_EMBEDDING=1 Q38_CPU_LM_HEAD=0 \
    Q38_RTX_FP8_LINEAR_ATTN=1 Q38_RTX_FP8_QSA=1 Q38_RTX_FP8_CACHE="$CACHE" \
    Q38_RTX_PEER_GPU_DRIVEN=1 Q38_VLLM_CUDAGRAPH=1 Q38_VLLM_CUDAGRAPH_FULL=1 Q38_PYSPY_SECONDS=0 \
    Q38_RTX_PEER_PLE_GPU=1 Q38_EMBEDDING_ZERO_COPY=1 Q38_M10_LM_HEAD=1 Q38_M10_LM_HEAD_TOPK=1024 \
    Q38_VLLM_REPEATS=1 Q38_PROGRESS_SECONDS=10 Q38_VLLM_MAX_NEW_TOKENS=32 \
    Q38_VLLM_KV_CACHE_BYTES=${Q38_VLLM_KV_CACHE_BYTES:-$((1 << 30))} Q38_VLLM_MAX_MODEL_LEN=${Q38_VLLM_MAX_MODEL_LEN:-8192} Q38_VLLM_MAX_BATCHED_TOKENS="$CHUNK" \
    Q38_RTX_PEER_MOE_BATCH=$(( batch > 0 ? 1 : 0 )) Q38_RTX_PEER_MOE_BATCH_PIPE=$(( batch == 2 ? 1 : 0 )) Q38_VLLM_PROMPT_FILE="$OUT/long_prompt.txt" \
    bash "$REPO/src/run_qwen38_rtx_peer_vllm.sh" > "$OUT/batch_$batch.runner.out" 2>&1
  rc=$?
  echo "RUN_EXIT,batch,$batch,$rc"
  dir=$(ls -td "$RUNS"/2026*/ | head -1); [[ $dir == "$before" ]] && dir=""
  python3 - "$dir" "$batch" >> "$OUT/runs.tsv" <<'PY'
import hashlib, json, sys, pathlib
d, batch = sys.argv[1], sys.argv[2]
try:
    r = json.loads((pathlib.Path(d) / "client/vllm/resident_vllm_client_result.json").read_text())
    g = r["generation"][0]; el = g["elapsed_seconds"]; n = len(g["prompt_token_ids"])
    pre = el - len(g["generated_token_ids"]) / 46.7
    sha = hashlib.sha256(str(g["generated_token_ids"]).encode()).hexdigest()[:12]
    s = json.dumps(r)
    def find(key):
        i = s.find(f'"{key}": ')
        return s[i + len(key) + 4:].split(",")[0].strip("} ") if i >= 0 else ""
    print(f"{batch}\t{d}\t{r.get('status')}\t{n}\t{el:.2f}\t{pre:.2f}\t{n / pre:.1f}\t{find('moe_batch_eager_calls')}\t{find('moe_batch_eager_rows')}\t{sha}")
except Exception as exc:
    print(f"{batch}\t{d}\tERROR {exc}\t\t\t\t\t\t\t")
PY
  tail -1 "$OUT/runs.tsv"
  [[ $rc -eq 0 ]] || { echo "ERROR,run batch=$batch failed; not continuing"; exit 1; }
done
python3 - "$OUT/runs.tsv" <<'PY'
import sys
rows = [l.split("\t") for l in open(sys.argv[1]).read().splitlines()[1:]]
same = len(rows) == 2 and rows[0][9] and rows[0][9] == rows[1][9]
print(f"Q38_MOE_BATCH_TOKENS_IDENTICAL,{int(same)},sha,{rows[0][9] if rows else ''},{rows[1][9] if len(rows) > 1 else ''}")
PY
echo "Q38_MOE_BATCH_INTEGRATION_DONE,$OUT"
