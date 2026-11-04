#!/usr/bin/env bash
# Full vLLM integration run on the optimized path:
#   RTX attention/router -> PEX multicast publish (4 KiB RTX leaves) -> 24 M10 workers
#   (routed v3, shared v2) -> leader push aggregation on cards 0-4 -> M10 writes into
#   RTX BAR1 (4 KiB return leaves) -> RTX reduce -> next layer, for all 48 layers.
#
# Precondition: run_qwen38_full_daemon_start.sh finished (daemon READY, 48 layers, PARKED,
# optimized kernels loaded). Its daemon.env is read from Q38_FULL_DAEMON_ENV or the
# qwen38-full-daemon/latest link.
# Queue through Pueue (group gpu) only after operator go-ahead:
#   pueue add --group gpu --label <label> -- sudo -n bash <this script>
#
# Every changed register is saved, verified and restored by the wrappers:
#   ACS 0x000c on 7 outer + 20 inner ports (run_qwen38_rtx_local_acs_trial.sh)
#   RTX request PTEs and M10 return/aggregation PTEs (peer library close)
#   rtx_mc_dma_map unloaded only after cleanup is proven.
set -Eeuo pipefail
ulimit -c 0
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/m10dis_env.sh"
ROOT=$M10DIS_HOME
REPO=$M10DIS_HOME
ENV_FILE=${Q38_FULL_DAEMON_ENV:-$M10DIS_RUNS/qwen38-full-daemon/latest/daemon.env}
MODEL=$M10DIS_MODEL
REFERENCE=${Q38_REFERENCE_DIRECT:-$M10DIS_STATE/reference/humaneval32-direct.json}
REFERENCE_PEER=${Q38_REFERENCE_PEER:-$M10DIS_STATE/reference/humaneval32-rtx-peer.json}
EXPECTED_UVM_SRCVERSION=${Q38_UVM_SRCVERSION:-38263E1067A39300ABB966D}
AGG_CARDS=${Q38_RTX_PEER_AGG_CARDS:-0,1,2,3,4}
PY=$REPO/.venv-vllm/bin/python
DAEMON="$PY $REPO/src/qwen38_resident_daemon.py"
NVCC=/usr/bin/nvcc; NVIDIA_SRC=/usr/src/nvidia-srv-580.178.04
RUN=$M10DIS_RUNS/qwen38-vllm-rtx-peer-optimized/$(date -u +%Y%m%dT%H%M%SZ)

[[ $EUID -eq 0 ]] || { echo 'ERROR,run as root (sudo -n)'; exit 2; }
mkdir -p "$RUN"
exec > >(tee -a "$RUN/runner.log") 2>&1
echo "RUN_DIR,$RUN"; echo "RUN_START,$(date -u --iso-8601=ns)"
[[ -r $ENV_FILE ]] || { echo "ERROR,daemon env $ENV_FILE missing"; exit 3; }
# shellcheck disable=SC1090
source "$ENV_FILE"; cp "$ENV_FILE" "$RUN/daemon.env"
CTRL=$Q38_FULL_CONTROL_SOCKET; MANIFEST=$Q38_FULL_MANIFEST; TOPOLOGY=$Q38_FULL_TOPOLOGY; SNAP=$Q38_FULL_SNAPSHOT
echo "DAEMON_RUN,$Q38_FULL_RUN"

src=$(cat /sys/module/nvidia_uvm/srcversion)
[[ $src == "$EXPECTED_UVM_SRCVERSION" ]] || { echo "ERROR,loaded UVM $src is not the patched build"; exit 9; }
[[ -S $CTRL && -f $MANIFEST && -f $TOPOLOGY && -d $SNAP && -S $Q38_FULL_MOE_SOCKET && -S $Q38_FULL_PLE_SOCKET ]] \
  || { echo 'ERROR,daemon socket or input missing'; exit 3; }
$DAEMON status --control-socket "$CTRL" > "$RUN/daemon-status-before.json"
# The daemon writes its PEX snapshots into a per-run subdirectory of Q38_FULL_SNAPSHOT; take the
# live path from its status (as the round-trip runner does) and require it to be under that root.
SNAP=$(python3 - "$RUN/daemon-status-before.json" "$MANIFEST" "$SNAP" <<'PY'
import json, sys, re, os
s = json.load(open(sys.argv[1])); m = open(sys.argv[2]).read()
assert s.get('status') == 'READY' and s.get('layers_ready') in (48, 49) and s.get('worker_activity') == 'PARKED', s.get('status')
snap = s.get('multicast_snapshot_root') or ''
assert os.path.dirname(snap) == sys.argv[3] and os.path.isdir(snap), (snap, sys.argv[3])
rid = re.search(r'^run_id\tdaemon:(\d+)$', m, re.M); assert rid and int(rid.group(1)) == s['pid']
print(snap)
PY
) || { echo 'ERROR,daemon precondition failed'; exit 3; }
echo "DAEMON_PRECONDITION,ready_48_parked,snapshot_root,$SNAP"
[[ ! -e /sys/module/rtx_mc_dma_map ]] || { echo 'ERROR,rtx_mc_dma_map already loaded'; exit 4; }
source "$REPO/src/qwen38_pci_env.sh"
q38_discover "$TOPOLOGY" || exit 3
_extra=""
for _c in ${AGG_CARDS//,/ }; do _extra+=" $(q38_inner_ports "$_c" 1)"; done
export Q38_ACS_EXTRA_PORTS="${_extra# }"
echo "ACS_EXTRA_PORTS_FROM_DISCOVERY,cards,$AGG_CARDS,ports,$Q38_ACS_EXTRA_PORTS"

worker_seq() {  # read-only: control seq of all 24 workers must be aligned
python3 - "$MANIFEST" "$1" <<'PY'
import mmap, os, struct, sys, json
rows = [l.split('\t') for l in open(sys.argv[1]).read().splitlines() if l and l[0].isdigit()]; out = []
for g, w, bdf, _, _, _, bus in rows:
    base = int(open(f'/sys/bus/pci/devices/{bdf}/resource').read().splitlines()[1].split()[0], 16)
    fd = os.open(f'/sys/bus/pci/devices/{bdf}/resource1', os.O_RDONLY | os.O_SYNC)
    m = mmap.mmap(fd, 0x24000, mmap.MAP_SHARED, mmap.PROT_READ, offset=int(bus, 16) - base)
    seq, layer, mode = struct.unpack_from('<III', m, 0); dseq, dst = struct.unpack_from('<II', m, 0x20000)
    out.append({'bdf': bdf, 'seq': seq, 'mode': mode, 'direct_seq': dseq, 'status': dst}); m.close(); os.close(fd)
json.dump(out, open(sys.argv[2], 'w'), indent=1)
seqs = {x['seq'] for x in out}; print('WORKER_SEQ', sorted(seqs), 'aligned', len(seqs) == 1)
assert len(seqs) == 1, 'worker sequences are not aligned; the peer library would refuse to open'
PY
}
worker_seq "$RUN/worker-seq-before.json"
capture_mc() {
python3 - "$TOPOLOGY" "$1" "$REPO/src" <<'PY'
import json, sys
sys.path.insert(0, sys.argv[3]); import pex_group_config as p
rows = [l.split('\t') for l in open(sys.argv[1]).read().splitlines() if l]; out = {}
for r in rows:
    for b in [r[1]] + r[2].split(','): out[b] = {k: hex(x) for k, x in p.regs(b).items()}
json.dump(out, open(sys.argv[2], 'w'), indent=1)
print('MC_REGISTERS,ports', len(out))
PY
}
capture_mc "$RUN/mc-registers-before.json"

# ---- library: 4 KiB leaves + leader aggregation ------------------------------------------
mkdir -p "$RUN/build"
LIB=$RUN/build/libqwen38_vllm_rtx_peer_small_leaves_agg.so
"$NVCC" -std=c++17 -O2 -lineinfo -shared -Xcompiler -fPIC -DQ38_RTX_PEER_SMALL_LEAVES -DQ38_RTX_PEER_LEADER_AGG \
  -gencode arch=compute_50,code=sm_50 -gencode arch=compute_86,code=sm_86 \
  -I"$REPO/transport/driver/abi/580.178.04" -I"$REPO/src" -I"$NVIDIA_SRC/common/inc" -I"$NVIDIA_SRC/nvidia-uvm" \
  "$REPO/src/qwen38_vllm_rtx_peer.cu" -o "$LIB" -lcuda -ldl > "$RUN/build.log" 2>&1
cp "$REPO/src/qwen38_vllm_rtx_peer.cu" "$REPO/src/pci_cluster_exchange.cuh" "$REPO/src/run_vllm_resident_m10_client.py" "$REPO/src/qwen38_vllm_rtx_peer_adapter.py" \
   "$REPO/src/run_qwen38_rtx_peer_pex_vllm_client.sh" "$REPO/src/run_qwen38_rtx_local_acs_trial.sh" \
   "$REPO/src/qwen38_cpu_head_hooks.py" "$REPO/src/qwen38_rtx_fp8_hooks.py" "$REPO/src/q38_cudagraph_breaks.py" "$REPO/src/build_qwen38_fp8_cache.py" "$REPO/src/qwen38_fp8_block_quant.py" "$REPO/src/qwen38_hc_fp8.py" "$REPO/src/run_stage8_stage9_vllm_generation.py" "$0" "$RUN/"
sha256sum "$LIB" > "$RUN/build-hashes.sha256"
echo "LIBRARY,$LIB,flags,Q38_RTX_PEER_SMALL_LEAVES Q38_RTX_PEER_LEADER_AGG,agg_cards,$AGG_CARDS"

# ---- the run: ACS wrapper -> DMA helper + vLLM client --------------------------------------
set +e
env Q38_RTX_TRIAL_MODE=peer-layer Q38_ACS_ALL_M10_BRANCHES=1 Q38_RTX_PEER_AGG_CARDS="$AGG_CARDS" \
  bash "$REPO/src/run_qwen38_rtx_local_acs_trial.sh" "$TOPOLOGY" "$MANIFEST" "$CTRL" "$SNAP" \
  bash "$REPO/src/run_qwen38_rtx_peer_pex_vllm_client.sh" "$TOPOLOGY" "$MANIFEST" "$CTRL" "$SNAP" \
       "$LIB" "$MODEL" "$Q38_FULL_MOE_SOCKET" "$Q38_FULL_PLE_SOCKET" "$RUN/client" 2>&1 | tee "$RUN/acs-trial-wrapper.log"
trial_rc=${PIPESTATUS[0]}
set -e
echo "ACS_TRIAL_EXIT,$trial_rc"
if [[ $trial_rc -eq 90 ]]; then echo 'PRESERVE,cleanup not proven by the inner run; no further changes made'; exit 90; fi

# ---- after-state, token comparison, throughput, clocks ---------------------------------------
$DAEMON status --control-socket "$CTRL" > "$RUN/daemon-status-after.json" || true
capture_mc "$RUN/mc-registers-after.json"
set +e; worker_seq "$RUN/worker-seq-after.json"; seq_rc=$?; set -e
N_ACS=$(( 7 + $(wc -w <<< "$Q38_ACS_EXTRA_PORTS") ))
"$PY" - "$RUN" "$trial_rc" "$seq_rc" "$N_ACS" "$REFERENCE" "$REFERENCE_PEER" <<'PY'
import csv, json, os, re, sys, pathlib, datetime
run = pathlib.Path(sys.argv[1]); trial_rc, seq_rc, n_acs = map(int, sys.argv[2:5])
ref = json.loads(pathlib.Path(sys.argv[5]).read_text()); ref_peer = json.loads(pathlib.Path(sys.argv[6]).read_text())
wrap = (run / 'acs-trial-wrapper.log').read_text()
client_log = (run / 'client' / 'client.log').read_text() if (run / 'client' / 'client.log').exists() else ''
res_p = run / 'client' / 'vllm' / 'resident_vllm_client_result.json'
res = json.loads(res_p.read_text()) if res_p.exists() else {}
after = json.loads((run / 'daemon-status-after.json').read_text() or '{}')
mc_same = json.loads((run / 'mc-registers-before.json').read_text()) == json.loads((run / 'mc-registers-after.json').read_text())
checks = {
 'trial_exit_0': trial_rc == 0,
 'client_pass': res.get('status') == 'PASS_VLLM_RESIDENT_M10_CLIENT',
 'acs_trial_pass': 'ACS_TRIAL_RESULT,PASS' in wrap,
 'acs_restored_all': len(re.findall(r'^ACS_RESTORE,.*,match,1$', wrap, re.M)) == n_acs,
 'dma_helper_unloaded': not pathlib.Path('/sys/module/rtx_mc_dma_map').exists(),
 'mc_registers_unchanged': mc_same,
 'daemon_ready_parked_48': after.get('status') == 'READY' and after.get('worker_activity') == 'PARKED' and after.get('layers_ready') in (48, 49),
 'worker_seq_aligned_after': seq_rc == 0,
}
# Observed leaf/route counts (reported, not judged: the adapter may open once per engine process).
obs = {'request_leaves_acquired': len(re.findall(r'^RTX_PEX_PTE,.*,acquired,1,', client_log, re.M)),
       'request_leaves_restored': len(re.findall(r'^RTX_PEX_PTE_RESTORE,.*restored,1', client_log, re.M)),
       'return_routes_applied': len(re.findall(r'^MULTI_ROUTE_APPLY,.*,status,0x00000000,', client_log, re.M)),
       'return_routes_restored': len(re.findall(r'^MULTI_ROUTE_RESTORE,', client_log, re.M))}
ref_ids = ref['generation'][0]['generated_token_ids']; peer_ids = ref_peer['generation'][0]['generated_token_ids']
def cmp(a, b):
    n = min(len(a), len(b)); first = next((i for i in range(n) if a[i] != b[i]), None)
    return {'len': [len(a), len(b)], 'identical': a == b, 'first_divergence': first,
            'matching_prefix': n if first is None else first}
gens = []
for g in res.get('generation', []):
    ids = g['generated_token_ids']; n = len(ids); el = g['elapsed_seconds']
    gens.append({'repeat': g['repeat'], 'tokens': n, 'prompt_tokens': len(g.get('prompt_token_ids', [])),
                 'elapsed_seconds': el, 'tokens_per_second_incl_prefill': n / el if el else None,
                 'started_unix': g.get('started_unix'), 'ended_unix': g.get('ended_unix'),
                 'vs_oct30_cpu_path': cmp(ids, ref_ids), 'vs_oct31_rtx_peer': cmp(ids, peer_ids)})
# Clocks during each timed repeat (repeat 0 is warm-up): fraction of M10 samples at >= 0.94 x max SM.
clk = []
cp = run / 'client' / 'clocks.csv'
if cp.exists():
    for row in csv.reader(cp.read_text().splitlines(), skipinitialspace=True):
        try:
            t = datetime.datetime.strptime(row[0], '%Y/%m/%d %H:%M:%S.%f').timestamp()
            clk.append((t, row[1], float(row[2].split()[0]), float(row[3].split()[0]), float(row[4].split()[0]), row[7]))
        except (ValueError, IndexError):
            pass
for g in gens:
    if g['started_unix'] is None: continue
    s = [c for c in clk if g['started_unix'] <= c[0] <= g['ended_unix']]
    m10 = [c for c in s if c[3] < 1500]  # M10 max SM 1306 MHz; the RTX is higher
    g['clock_samples_m10'] = len(m10)
    g['clock_m10_at_ge_0p94_max'] = (sum(c[2] >= 0.94 * c[3] for c in m10) / len(m10)) if m10 else None
    g['clock_m10_min_sm_mhz'] = min((c[2] for c in m10), default=None)
    g['clock_m10_min_mem_mhz'] = min((c[4] for c in m10), default=None)
peer = res.get('rtx_peer_timing') or {}
cpu_heads = {}
for f in sorted((run / 'client' / 'vllm').glob('cpu_head_timing.*.json')):
    for role, st in json.loads(f.read_text()).get('stats', {}).items():
        a = cpu_heads.setdefault(role, {'calls': 0, 'rows': 0, 'seconds': 0.0}); [a.__setitem__(k, a[k] + st[k]) for k in a]
for a in cpu_heads.values(): a['ms_per_call'] = 1e3 * a['seconds'] / a['calls'] if a['calls'] else None
summary = {'checks': checks, 'observed_routes': obs, 'generations': gens,
           'vllm_load_seconds': res.get('vllm_load_seconds'), 'error': res.get('error'),
           'cleanup_errors': res.get('cleanup_errors'),
           'cpu_heads': cpu_heads, 'settings': {k: os.environ.get(k) for k in ('Q38_VLLM_CPU_OFFLOAD_GB', 'Q38_VLLM_GPU_MEMORY_UTILIZATION', 'Q38_VLLM_KV_CACHE_BYTES', 'Q38_CPU_EMBEDDING', 'Q38_CPU_LM_HEAD', 'Q38_RTX_FP8_LINEAR_ATTN', 'Q38_VLLM_CUDAGRAPH', 'Q38_VLLM_CUDAGRAPH_FULL', 'Q38_RTX_FP8_CACHE', 'Q38_RTX_PEER_GPU_DRIVEN', 'Q38_RTX_PEER_PLE_GPU', 'Q38_EMBEDDING_ZERO_COPY', 'Q38_M10_LM_HEAD', 'Q38_M10_LM_HEAD_TOPK', 'Q38_RTX_PEER_MOE_BATCH', 'Q38_BATCH_TIMING', 'Q38_BATCH_TIMING_ROWS', 'Q38_BATCH_TIMING_PASSES', 'Q38_BATCH_TIMING_IDS', 'Q38_RTX_PEER_MOE_BATCH_MIN_ROWS', 'Q38_BATCH_REUSE_DIR', 'Q38_RTX_PEER_MOE_BATCH_PIPE', 'Q38_RTX_PEER_MOE_BATCH_PIPE_MIN_ROWS', 'Q38_BATCH_TIMING_PATHS', 'Q38_BATCH_STRESS', 'Q38_BATCH_STRESS_PATHS', 'VLLM_TRITON_FORCE_FIRST_CONFIG', 'Q38_VLLM_LOGPROBS', 'Q38_RTX_FP8_HC', 'VLLM_SPARSE_INDEXER_MAX_LOGITS_MB', 'Q38_BENCH_JSONL', 'Q38_BENCH_MAX_TOKENS', 'Q38_BENCH_EFFORT', 'Q38_BENCH_THINKING', 'Q38_BENCH_TEMPERATURE', 'Q38_VLLM_PREFIX_CACHING', 'Q38_VLLM_KV_CACHE_BYTES', 'Q38_VLLM_MAX_MODEL_LEN', 'Q38_VLLM_MAX_BATCHED_TOKENS', 'Q38_VLLM_ASYNC_SCHEDULING', 'Q38_GRAPH_CEILING_ITERS', 'Q38_GRAPH_BREAKDOWN_STEPS', 'Q38_RTX_FP8_QSA', 'Q38_HUMANEVAL_JSONL', 'Q38_HUMANEVAL_MAX_PROMPT_TOKENS', 'Q38_HUMANEVAL_MAX_NEW_TOKENS', 'Q38_VLLM_REPEATS')},
           'rtx_peer_calls': sum(p.get('calls', 0) for p in peer.get('processes', []))}
(run / 'result.json').write_text(json.dumps(summary, indent=2) + '\n')
for k, v in checks.items(): print('CHECK', k, 'OK' if v else 'FAIL')
print('OBSERVED_ROUTES', json.dumps(obs))
print('CPU_HEADS', json.dumps(cpu_heads))
for g in gens:
    print('GENERATION', json.dumps({k: g[k] for k in g if k not in ('started_unix', 'ended_unix')}))
sys.exit(0 if all(checks.values()) else 1)
PY
echo "VLLM_RTX_PEER_RUN_OK,$RUN"
