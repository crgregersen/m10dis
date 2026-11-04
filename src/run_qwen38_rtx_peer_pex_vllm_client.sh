#!/usr/bin/env bash
# Inner step of the full vLLM RTX-peer run (called by run_qwen38_rtx_peer_vllm.sh inside
# run_qwen38_rtx_local_acs_trial.sh, which holds ACS cleared for the whole vLLM lifetime).
#
# Holds rtx_mc_dma_map (requester = discovered RTX) for the vLLM lifetime, runs
# run_vllm_resident_m10_client.py --rtx-peer-moe with the PEX multicast request path,
# 4-KiB leaves and leader aggregation, samples clocks on all 25 GPUs, then proves
# cleanup before unloading the DMA helper and printing the SAFE_TO_UNMAP marker that
# lets the ACS wrapper restore the ports.
set -euo pipefail
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $EUID -ne 0 || $# -ne 9 ]]; then
  echo "usage: sudo $0 GROUPS.tsv GPU_MAPPINGS CONTROL_SOCKET SNAPSHOT_ROOT LIBRARY MODEL MOE_SOCKET PLE_SOCKET OUTPUT_DIR" >&2
  exit 2
fi
GROUPS_TSV=$(realpath "$1"); MAPS=$(realpath "$2"); SOCKET=$(realpath -m "$3")
SNAPSHOTS=$(realpath "$4"); LIBRARY=$(realpath "$5"); MODEL=$(realpath "$6")
MOE_SOCKET=$(realpath -m "$7"); PLE_SOCKET=$(realpath -m "$8"); OUT=$(realpath -m "$9")
mkdir -p "$OUT"
KO_SRC="$ROOT/transport/rtx_mc_dma/rtx_mc_dma_map.ko"
PYTHON="$ROOT/.venv-vllm/bin/python"
RM_TRACE=${Q38_RM_TRACE_PRELOAD:-$ROOT/build/rm_ioctl_trace_preload.so}
PROMPT=${Q38_VLLM_PROMPT_FILE:-$ROOT/prompts/humaneval32_prompt.py}
source "$ROOT/src/qwen38_pci_env.sh"
q38_discover "$GROUPS_TSV" || exit 3

exec > >(tee -a "$OUT/experiment.log") 2>&1
echo "RUN_DIR,$OUT"; echo "UTC,$(date -u --iso-8601=ns)"
echo "RTX,$Q38_RTX_BDF,$Q38_RTX_UUID"
echo "VLLM_SETTINGS,gpu_driven,${Q38_RTX_PEER_GPU_DRIVEN:-0},ple_gpu,${Q38_RTX_PEER_PLE_GPU:-0},embedding_zero_copy,${Q38_EMBEDDING_ZERO_COPY:-0},m10_lm_head,${Q38_M10_LM_HEAD:-0},lm_topk,${Q38_M10_LM_HEAD_TOPK:-1024},moe_batch,${Q38_RTX_PEER_MOE_BATCH:-0},max_batched_tokens,${Q38_VLLM_MAX_BATCHED_TOKENS:-8},fp8_cache,${Q38_RTX_FP8_CACHE:-none},cuda_graphs,${Q38_VLLM_CUDAGRAPH:-0},cuda_graphs_full,${Q38_VLLM_CUDAGRAPH_FULL:-0},pyspy_seconds,${Q38_PYSPY_SECONDS:-0},cpu_offload_gb,${Q38_VLLM_CPU_OFFLOAD_GB:-6},gpu_memory_utilization,${Q38_VLLM_GPU_MEMORY_UTILIZATION:-0.78},kv_cache_bytes,${Q38_VLLM_KV_CACHE_BYTES:-0},cpu_embedding,${Q38_CPU_EMBEDDING:-0},cpu_lm_head,${Q38_CPU_LM_HEAD:-0},fp8_linear_attn,${Q38_RTX_FP8_LINEAR_ATTN:-0},repeats,${Q38_VLLM_REPEATS:-2}"
modinfo nvidia_uvm | rg '^(filename|srcversion):' || true
sha256sum "$LIBRARY" "$ROOT/src/qwen38_vllm_rtx_peer.cu" "$ROOT/src/run_vllm_resident_m10_client.py" \
  "$ROOT/src/qwen38_vllm_rtx_peer_adapter.py" "$ROOT/src/sitecustomize.py" "$ROOT/src/qwen38_cpu_head_hooks.py" "$ROOT/src/run_stage8_stage9_vllm_generation.py" "$PROMPT" > "$OUT/source-binary.sha256"
cp "$GROUPS_TSV" "$MAPS" "$OUT/"

"$PYTHON" - "$ROOT" "$GROUPS_TSV" "$MAPS" "$SOCKET" "$SNAPSHOTS" <<'PY'
import json, pathlib, subprocess, sys
root, groups, maps, sock, snapshots = map(pathlib.Path, sys.argv[1:])
rows = [l.split("\t") for l in groups.read_text().splitlines() if l]
assert len(rows) == 6
records = [l.split("\t") for l in maps.read_text().splitlines() if l and l[0].isdigit()]
assert len(records) == 24
for r in records:
    assert r[2] == rows[int(r[0])][3].split(",")[int(r[1])]
for g, row in enumerate(rows):
    plan = json.loads((snapshots / f"group{g}" / "plan.json").read_text())
    assert plan["base"] == int(row[6], 0) and plan["window"] == 65536
s = json.loads(subprocess.check_output([sys.executable, str(root / "src/qwen38_resident_daemon.py"),
                                        "status", "--control-socket", str(sock)], text=True))
assert s["status"] == "READY" and s["layers_ready"] in (48, 49), s.get("layers_ready")
assert s["worker_activity"] == "PARKED"
assert s["direct_moe_setup"]["gpu_map_records"] == 24
assert s["multicast_snapshot_root"] == str(snapshots)
assert "routed_v3" in json.dumps(s.get("kernel_reload") or {}), "optimized kernels not loaded"
print(f"DAEMON_PREFLIGHT,PASS,pid,{s['pid']},layers,48,activity,PARKED,kernels,routed_v3+shared_v2")
PY

if lsmod | awk '$1=="rtx_mc_dma_map" {f=1} END {exit !f}'; then
  echo "ERROR,rtx_mc_dma_map already loaded; refusing to replace its live DMA mappings"; exit 2
fi
mapfile -t F < <(python3 - "$GROUPS_TSV" <<'PY'
import sys
rows = [l.split("\t") for l in open(sys.argv[1]).read().splitlines() if l]
for c in (6, 4, 5): print(",".join(str(int(r[c], 0)) for r in rows))
PY
)
make -C "$ROOT/transport/rtx_mc_dma" KDIR="/lib/modules/$(uname -r)/build" clean all > "$OUT/module-build.log" 2>&1
cp "$KO_SRC" "$OUT/rtx_mc_dma_map.ko"; sha256sum "$KO_SRC" > "$OUT/module.sha256"
insmod "$KO_SRC" "requester_bdf=$Q38_RTX_BDF" "groups=6" "ingress=${F[0]}" "branch_start=${F[1]}" "branch_end=${F[2]}"
echo "PEX_DMA_HELPER_LOADED,$Q38_RTX_BDF,groups,6"
dmesg --ctime > "$OUT/dmesg-before.txt"

# Clock sampler on all 25 GPUs for the whole vLLM lifetime (500 ms).
nvidia-smi --query-gpu=timestamp,pci.bus_id,clocks.sm,clocks.max.sm,clocks.mem,clocks.max.mem,temperature.gpu,clocks_throttle_reasons.active \
  --format=csv,noheader -lms 500 > "$OUT/clocks.csv" 2>/dev/null &
SAMPLER=$!

# Optional py-spy profile of the vLLM EngineCore (Q38_PYSPY_SECONDS>0): waits for steady decode
# (>40 decoded tokens in the adapter's Q38_PROGRESS lines), then samples the engine process
# non-blocking (never paused) and writes collapsed stacks + a flame graph.
PYSPY_PID=
if [[ ${Q38_PYSPY_SECONDS:-0} -gt 0 ]]; then
  (
    set +e +o pipefail   # the wrapper's set -euo pipefail must not kill this background recorder
    for _ in $(seq 1 1800); do
      line=$(grep -a 'Q38_PROGRESS' "$OUT/client.log" 2>/dev/null | tail -1 || true)
      tok=$(sed -n 's/.*decode_tokens,\([0-9]*\),.*/\1/p' <<< "$line")
      [[ -n $tok && $tok -gt 40 ]] && break
      sleep 1
    done
    epid=$(grep -ao 'EngineCore pid=[0-9]*' "$OUT/client.log" | tail -1 | cut -d= -f2 || true)
    [[ -n $epid ]] || { echo "PYSPY,no_engine_pid"; exit 0; }
    echo "PYSPY,start,engine_pid,$epid,seconds,$Q38_PYSPY_SECONDS,at_decode_tokens,$tok"
    "$ROOT/.venv-vllm/bin/py-spy" record -p "$epid" -d "$Q38_PYSPY_SECONDS" -r 100 --nonblocking -f raw \
      -o "$OUT/pyspy-engine.collapsed" > "$OUT/pyspy.log" 2>&1
    "$ROOT/.venv-vllm/bin/py-spy" record -p "$epid" -d 5 -r 100 --nonblocking -f flamegraph \
      -o "$OUT/pyspy-engine.svg" >> "$OUT/pyspy.log" 2>&1
    echo "PYSPY,done,$(wc -l < "$OUT/pyspy-engine.collapsed" 2>/dev/null) stack lines"
  ) > "$OUT/pyspy-runner.log" 2>&1 &
  PYSPY_PID=$!
fi

# Optional Nsight Systems trace of steady decode (Q38_NSYS=1, 2026-11-04): the client runs under `nsys profile`
# (CUDA + NVTX, CUDA graphs traced per node, children followed). Collection is the cudaProfilerApi range the
# adapter opens at decode token Q38_CUDA_PROFILER_START_TOKENS (default 500) for Q38_CUDA_PROFILER_TOKENS
# (default 200). No --duration and no --kill: the nsys session lives as long as the client, so the wrapper's
# cleanup order is unchanged. (v2/v3 used --delay/--duration --kill=none: the CLI returned while the client still
# ran, and the client then hung at exit holding the ACS lock. Never use that mode here.)
# Report: $OUT/nsys-decode.nsys-rep.
NSYS_PREFIX=()
if [[ ${Q38_NSYS:-0} == 1 ]]; then
  export Q38_CUDA_PROFILER_START_TOKENS=${Q38_CUDA_PROFILER_START_TOKENS:-500}
  export Q38_CUDA_PROFILER_TOKENS=${Q38_CUDA_PROFILER_TOKENS:-200}
  NSYS_PREFIX=(nsys profile --trace=cuda,nvtx --cuda-graph-trace=node --trace-fork-before-exec=true
    --sample=none --cpuctxsw=none --capture-range=cudaProfilerApi --capture-range-end=stop
    --output="$OUT/nsys-decode" --force-overwrite=true)
  echo "NSYS,profile,start_tokens,$Q38_CUDA_PROFILER_START_TOKENS,tokens,$Q38_CUDA_PROFILER_TOKENS"
fi

set +e
env CUDA_HOME=/usr/local/cuda-12.4 PYTHONPATH="$ROOT/src" \
  Q38_RTX_BDF="$Q38_RTX_BDF" Q38_RTX_UUID="$Q38_RTX_UUID" \
  Q38_RTX_PEER_PEX_MULTICAST=1 Q38_M10_GROUPS_TSV="$GROUPS_TSV" \
  Q38_RTX_PEER_USE_OWNER_BUS_ADDRESS=1 \
  Q38_RTX_PEER_AGG_CARDS="${Q38_RTX_PEER_AGG_CARDS:?}" \
  Q38_RTX_PEER_PHASE_CSV="${Q38_RTX_PEER_PHASE_CSV:-}" \
  Q38_RM_TRACE_PRELOAD="$RM_TRACE" \
  Q38_CPU_EMBEDDING="${Q38_CPU_EMBEDDING:-0}" Q38_CPU_LM_HEAD="${Q38_CPU_LM_HEAD:-0}" \
  Q38_CPU_HEAD_TIMING_JSON="$OUT/vllm/cpu_head_timing" \
  Q38_LOG_LAYER_MEMORY="${Q38_LOG_LAYER_MEMORY:-0}" Q38_LAYER_MEMORY_LOG="$OUT/vllm/layer_memory" \
  Q38_RTX_FP8_LINEAR_ATTN="${Q38_RTX_FP8_LINEAR_ATTN:-0}" \
  Q38_VLLM_CUDAGRAPH="${Q38_VLLM_CUDAGRAPH:-0}" \
  Q38_VLLM_CUDAGRAPH_FULL="${Q38_VLLM_CUDAGRAPH_FULL:-0}" \
  Q38_RTX_FP8_CACHE="${Q38_RTX_FP8_CACHE:-}" \
  Q38_RTX_PEER_GPU_DRIVEN="${Q38_RTX_PEER_GPU_DRIVEN:-0}" \
  Q38_RTX_PEER_PLE_GPU="${Q38_RTX_PEER_PLE_GPU:-0}" \
  bash -c 'ulimit -c 0; exec "$@"' _ \
  "${NSYS_PREFIX[@]}" "$PYTHON" -u "$ROOT/src/run_vllm_resident_m10_client.py" \
    --model "$MODEL" --moe-socket "$MOE_SOCKET" --ple-socket "$PLE_SOCKET" --control-socket "$SOCKET" \
    --out "$OUT/vllm" --prompt-file "$PROMPT" \
    --max-model-len "${Q38_VLLM_MAX_MODEL_LEN:-512}" --max-new-tokens "${Q38_VLLM_MAX_NEW_TOKENS:-220}" \
    --cpu-offload-gb "${Q38_VLLM_CPU_OFFLOAD_GB:-6}" --repeats "${Q38_VLLM_REPEATS:-2}" \
    --gpu-memory-utilization "${Q38_VLLM_GPU_MEMORY_UTILIZATION:-0.78}" \
    --kv-cache-memory-bytes "${Q38_VLLM_KV_CACHE_BYTES:-0}" \
    --rtx-peer-moe --rtx-peer-library "$LIBRARY" \
    --park-workers-on-exit \
    > "$OUT/client.log" 2>&1
client_rc=$?
if [[ ${#NSYS_PREFIX[@]} -gt 0 ]]; then
  # Backstop: never continue to cleanup while the real client (its MainProcess pid from the log) still runs.
  cpid=$(grep -ao 'MainProcess pid=[0-9]*' "$OUT/client.log" | head -1 | cut -d= -f2 || true)
  waited=0
  while [[ -n $cpid ]] && grep -qa run_vllm_resident_m10_client.py "/proc/$cpid/cmdline" 2>/dev/null; do
    [[ $waited -ge 900 ]] && { echo "NSYS_CLIENT_STILL_RUNNING,pid,$cpid,after_s,$waited"; break; }
    sleep 2; waited=$((waited + 2))
  done
  echo "NSYS_CLIENT_WAIT,pid,${cpid:-none},seconds,$waited,nsys_rc,$client_rc"
fi
set -e
kill "$SAMPLER" 2>/dev/null || true; wait "$SAMPLER" 2>/dev/null || true
# The recorder usually exits on its own before the client does; kill/cat must not trip set -e
# (2026-11-03 tasks 44/45: an unguarded kill of the finished recorder aborted the wrapper before
# its cleanup proof and left ACS + the DMA helper in place).
if [[ -n $PYSPY_PID ]]; then kill "$PYSPY_PID" 2>/dev/null || true; wait "$PYSPY_PID" 2>/dev/null || true; cat "$OUT/pyspy-runner.log" 2>/dev/null || true; fi
echo "VLLM_CLIENT_EXIT,$client_rc"
tail -5 "$OUT/client.log"

# Cleanup proof: the client result is PASS without cleanup errors, the adapter wrote its
# timing file (only after q38_rtx_peer_close succeeded), the daemon is READY and PARKED,
# and no CUDA process other than the daemon's services remains.
set +e
"$PYTHON" - "$OUT/vllm" "$SOCKET" "$ROOT" <<'PY'
import json, pathlib, subprocess, sys
out, sock, root = pathlib.Path(sys.argv[1]), sys.argv[2], sys.argv[3]
rp = out / "resident_vllm_client_result.json"
r = json.loads(rp.read_text()) if rp.exists() else {}
timing = sorted(out.glob("rtx_peer_timing.*.json"))
# The peer library logs every RTX PTE it acquires to stderr (client.log). No such line means
# the peer path never opened (e.g. vLLM failed at model load): nothing to tear down.
never_opened = "RTX_PEX_PTE," not in (out.parent / "client.log").read_text(errors="replace")
s = json.loads(subprocess.check_output([sys.executable, f"{root}/src/qwen38_resident_daemon.py", "status",
                                        "--control-socket", sock], text=True))
allowed = {s["pid"], s.get("routed_service_pid"), s.get("shared_service_pid")}
stray = []
for p in subprocess.check_output(["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"], text=True).split():
    chain = [int(p)]
    while chain[-1] > 1:
        chain.append(int(open(f"/proc/{chain[-1]}/stat").read().rsplit(")", 1)[1].split()[1]))
    if not allowed & set(chain): stray.append(int(p))
checks = {"client_pass": r.get("status") == "PASS_VLLM_RESIDENT_M10_CLIENT",
          "no_cleanup_errors": not r.get("cleanup_errors"),
          "peer_close_timing_written": bool(timing),
          "daemon_ready_parked": s.get("status") == "READY" and s.get("worker_activity") == "PARKED",
          "no_stray_cuda_clients": not stray,
          "peer_client_unregistered": (s.get("peer_client") or {}).get("status", "NONE") == "NONE"}
for k, v in checks.items(): print("CLEANUP_CHECK", k, "OK" if v else "FAIL")
if all(checks.values()):
    sys.exit(0)
pc = s.get("peer_client") or {}
if not never_opened and pc.get("status") in ("REGISTERED", "RECOVERY_FAILED") and checks["no_stray_cuda_clients"]:
    # The client died or skipped its close: the daemon owns the journal and restores the M10-side
    # routes while its page tables still exist, parks the workers and unloads the mailbox.
    print("CLEANUP_CHECK peer_client_registered -> RECOVER_PEER_CLIENT", pc.get("run_id"), pc.get("journal"))
    rec = subprocess.run([sys.executable, f"{root}/src/qwen38_resident_daemon.py", "recover-peer-client",
                          "--control-socket", sock, "--clients-quiesced"], capture_output=True, text=True)
    print(rec.stdout)
    s3 = json.loads(subprocess.check_output([sys.executable, f"{root}/src/qwen38_resident_daemon.py", "status",
                                             "--control-socket", sock], text=True))
    ok = (rec.returncode == 0 and s3.get("status") == "READY" and s3.get("worker_activity") == "PARKED"
          and (s3.get("peer_client") or {}).get("status") == "NONE")
    print("CLEANUP_CHECK daemon_recovered_peer_client", "OK" if ok else "FAIL")
    sys.exit(0 if ok else 1)
# Peer path opened and was closed cleanly, but the client failed elsewhere (2026-11-03 task 46):
# the journal ends CLOSED with every acquired record released and the daemon holds no peer
# client, so the routes are gone; release this run's mailbox and allow SAFE_TO_UNMAP.
journals = sorted(out.glob("rtx_peer_routes.*.journal"))
def journal_released(path):
    L = [l.rstrip("\n").split("\t") for l in path.read_text().splitlines() if l.strip()]
    pte = {l[1] for l in L if l[0] == "PTE"}; pte_r = {l[1] for l in L if l[0] == "PTE_RESTORED"}
    dma = {tuple(l[1:3]) for l in L if l[0] == "DMA"}; dma_r = {tuple(l[1:3]) for l in L if l[0] == "DMA_UNMAPPED"}
    own = {l[1] for l in L if l[0] == "OWNER"}; own_r = {l[1] for l in L if l[0] == "OWNER_RELEASED"}
    return bool(L) and L[-1][0] in ("CLOSED", "RECOVERED") and pte <= pte_r and dma <= dma_r and own <= own_r
closed_clean = bool(journals) and all(journal_released(j) for j in journals)
if (not never_opened and closed_clean and checks["peer_client_unregistered"]
        and checks["daemon_ready_parked"] and checks["no_stray_cuda_clients"]):
    print("CLEANUP_CHECK peer_path_closed_clean OK (journal CLOSED, all records released)")
    never_opened = True        # same remaining cleanup as the never-opened path: own mailbox only
if never_opened and checks["daemon_ready_parked"] and checks["no_stray_cuda_clients"]:
    print("CLEANUP_CHECK peer_path_never_opened OK (client failed before acquiring any route)")
    # The client keeps its CPU mailbox registered after an unclean exit. With no route ever
    # acquired and the workers parked, release it only if the daemon's mailbox is THIS run's.
    mf = out / "direct_cpu_mailbox.manifest"
    if mf.exists():
        man = dict(l.split("\t", 1) for l in mf.read_text().splitlines() if "\t" in l)
        mb = s.get("cpu_mailbox") or {}
        if mb.get("status") == "LOADED" and mb.get("run_id") == man.get("run_id"):
            sys.path.insert(0, f"{root}/src")
            from qwen38_cpu_mailbox_manifest import unload_from_daemon
            resp = unload_from_daemon(pathlib.Path(sock))
            s2 = json.loads(subprocess.check_output([sys.executable, f"{root}/src/qwen38_resident_daemon.py", "status",
                                                     "--control-socket", sock], text=True))
            ok = resp.get("status") == "ok" and (s2.get("cpu_mailbox") or {}).get("status") == "UNLOADED"
            print("CLEANUP_CHECK own_cpu_mailbox_unloaded", "OK" if ok else "FAIL", man.get("run_id"))
            if not ok:
                sys.exit(1)
            shm = pathlib.Path("/dev/shm" + man["shm_name"])
            if shm.exists():
                shm.unlink(); print("CLEANUP_CHECK own_cpu_mailbox_shm_unlinked OK", shm)
        else:
            print("CLEANUP_CHECK own_cpu_mailbox", "not_loaded_by_this_run", mb.get("status"), mb.get("run_id"))
    sys.exit(0)
sys.exit(1)
PY
proof_rc=$?
set -e
if [[ $proof_rc -ne 0 ]]; then
  echo "PRESERVE,cleanup not proven; rtx_mc_dma_map and ACS state kept for recovery"
  exit 90
fi
rmmod rtx_mc_dma_map
echo "PEX_DMA_HELPER_RELEASED,all six resource mappings unmapped"
dmesg --ctime > "$OUT/dmesg-after.txt"
diff -u "$OUT/dmesg-before.txt" "$OUT/dmesg-after.txt" > "$OUT/dmesg-diff.txt" || true
echo "SAFE_TO_UNMAP,all acquired GPU and PTE objects released"
exit "$client_rc"
