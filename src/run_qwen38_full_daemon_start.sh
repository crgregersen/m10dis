#!/usr/bin/env bash
# Start the 48-layer resident M10 daemon from the NVMe copies, then load the optimized
# worker kernels (routed v3, shared v2) and leave the workers PARKED for the vLLM run.
#
# Queue through Pueue (group gpu) only after operator go-ahead:
#   pueue add --group gpu --label <label> -- sudo -n bash <this script>
#
# Inputs:
#   expert bundles  $M10DIS_DATA/qwen38-stage9-preserved-bundles (rsync + SHA-256 of the USB copy)
#   PLE shards      $M10DIS_DATA/qwen38-ple-shards/ple_shards.tsv  (128 extracted, SHA-verified)
#   topology        rows of Q38_FULL_TOPOLOGY_SRC, re-checked against a live sysfs scan
#                   (upstream, inner ports, GPUs) and against each PEX8747 upstream's live
#                   prefetchable bridge window; PCI addresses are never typed in here.
# Output: $RUN/daemon.env (sockets, manifest, topology, snapshot root) for the vLLM runner.
set -Eeuo pipefail
ulimit -c 0
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/m10dis_env.sh"
# Q38_DAEMON_LAYERS (2026-11-04): 48 model layers, or 49 = also the MTP layer's experts as layer 48
NL=${Q38_DAEMON_LAYERS:-48}
[[ $NL == 48 || $NL == 49 ]] || { echo "ERROR,Q38_DAEMON_LAYERS must be 48 or 49"; exit 2; }
ROOT=$M10DIS_HOME
REPO=$M10DIS_HOME
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
RUN=$M10DIS_RUNS/qwen38-full-daemon/$STAMP
RUNTIME=/run/user/1000/q38-m10-full-$STAMP
MODEL=$M10DIS_MODEL
BUNDLES=$M10DIS_DATA/qwen38-stage9-preserved-bundles
PLE_SHARDS=$M10DIS_DATA/qwen38-ple-shards/ple_shards.tsv
# LM head on the M10s (2026-11-03): raw BF16 lm_head.weight (build_qwen38_lm_head_raw.py); the PLE owner
# appends one row slice per M10 to its PLE allocation (worker mode 7). Q38_LM_HEAD_FILE= (empty) disables.
LM_HEAD_FILE=${Q38_LM_HEAD_FILE-$M10DIS_DATA/qwen38-lm-head/0419e2c2dfbb9252/lm_head_bf16.bin}
PLACEMENT=${Q38_PLACEMENT:-$M10DIS_STATE/stage9-placement}
TOPOLOGY_SRC=${Q38_FULL_TOPOLOGY_SRC:-$M10DIS_STATE/topology.tsv}
PY=$REPO/.venv-vllm/bin/python
DAEMON="$PY $REPO/src/qwen38_resident_daemon.py"
PRELOAD=${Q38_RM_TRACE_PRELOAD:-$REPO/build/rm_ioctl_trace_preload.so}   # src/build_rm_trace_preload.sh
# PLE owner and services are built from current source in $RUN/build (parallel per-card loaders;
# PLE_PARALLEL_LOAD=0 / Q38_PARALLEL_LOAD=0 select the old serial loops).
PLE=$RUN/build/cuda_load_stage9_ple_ipc_owner
NVCC=/usr/bin/nvcc; NVIDIA_SRC=/usr/src/nvidia-srv-580.178.04
EXPECTED_UVM_SRCVERSION=${Q38_UVM_SRCVERSION:-38263E1067A39300ABB966D}   # 512 multiroute + lifecycle cleanup + REAP_DEAD
EXPECTED_UVM_SHA=${Q38_UVM_SHA:-06e873a41d87eb7f1b3c6fb86335983b187260390d139285281149d5490650f8}
SWAP_LOG=${Q38_UVM_SWAP_LOG:-$M10DIS_STATE/uvm-swap.log}   # written by transport/driver/swap-uvm.sh
EXPECTED_LAST_UVM_LOAD=${Q38_UVM_LAST_LOAD:?set Q38_UVM_LAST_LOAD to the swap time (journalctl -k: nvidia_uvm: module uses)}

[[ $EUID -eq 0 ]] || { echo 'ERROR,run as root (sudo -n)'; exit 2; }
mkdir -p "$RUN" "$RUNTIME" "$RUN/daemon-pex-snapshots" "$RUN/build"
exec > >(tee -a "$RUN/runner.log") 2>&1
echo "RUN_DIR,$RUN"; echo "START_UTC,$(date -u --iso-8601=ns)"; echo "KERNEL,$(uname -a)"
cp "$0" "$RUN/"

# ---- identity: patched 512 UVM, loaded by the recorded swap and not reloaded since ----------
src=$(cat /sys/module/nvidia_uvm/srcversion)
[[ $src == "$EXPECTED_UVM_SRCVERSION" ]] || { echo "ERROR,loaded UVM srcversion $src is not the patched build"; exit 9; }
grep -q "loaded_candidate_sha256=$EXPECTED_UVM_SHA" "$SWAP_LOG" || { echo 'ERROR,swap record does not show the expected UVM build'; exit 9; }
last_load=$(journalctl -k -b -o short-iso --no-pager | grep 'nvidia_uvm: module uses' | tail -1 | cut -d' ' -f1)
echo "UVM_IDENTITY,srcversion,$src,sha256,$EXPECTED_UVM_SHA,last_load,$last_load"
[[ $last_load == "$EXPECTED_LAST_UVM_LOAD" ]] || { echo "ERROR,UVM reloaded since the recorded swap ($last_load)"; exit 9; }
echo "NVIDIA_DRIVER,$(nvidia-smi --query-gpu=driver_version --format=csv,noheader | head -1)"

# ---- inputs -------------------------------------------------------------------------
[[ -r $BUNDLES/layer_00/routed/resident_layer_manifest.json && -r $BUNDLES/layer_47/routed/resident_layer_manifest.json ]] \
  || { echo 'ERROR,NVMe bundle copy incomplete'; exit 3; }
[[ -r $PLE_SHARDS && -r $PRELOAD && -d $MODEL ]] || { echo 'ERROR,input missing'; exit 3; }
[[ $(wc -l < "$PLE_SHARDS") -eq 128 ]] || { echo 'ERROR,PLE shard table does not have 128 rows'; exit 3; }
grep -q 'all_match": true' $M10DIS_DATA/qwen38-ple-shards/extract-report.json || { echo 'ERROR,PLE extract report not all_match'; exit 3; }
if [[ -n $LM_HEAD_FILE ]]; then
  LM_DIR=$(dirname "$LM_HEAD_FILE")
  ( cd "$LM_DIR" && sha256sum -c --quiet SHA256SUMS ) || { echo 'ERROR,LM-head file SHA-256 mismatch'; exit 3; }
  python3 - "$LM_DIR/build-report.json" "$MODEL/model.safetensors.index.json" "$LM_HEAD_FILE" <<'PY' || { echo 'ERROR,LM-head file does not match this checkpoint'; exit 3; }
import hashlib, json, os, sys
r = json.load(open(sys.argv[1])); h = hashlib.sha256(open(sys.argv[2], 'rb').read()).hexdigest()
assert r['model_index_sha256'] == h and r['rows'] == 248320 and r['row_bytes'] == 5120, r
assert os.path.getsize(sys.argv[3]) == r['rows'] * r['row_bytes']
print('LM_HEAD_FILE_OK', sys.argv[3], r['rows'], r['file_sha256'][:16])
PY
fi
if [[ -f $M10DIS_DATA/copy-logs/verify.log ]]; then
  echo "BUNDLE_VERIFY_LOG,$(tail -2 $M10DIS_DATA/copy-logs/verify.log | tr '\n' ' ')"
else
  echo 'BUNDLE_VERIFY,not finished at start time (rsync exit 0 recorded)'
fi
df -h "$BUNDLES" "$MODEL" "$RUN"
[[ ! -e $RUNTIME/control.sock ]] || { echo 'ERROR,stale control socket'; exit 4; }
pgrep -f 'qwen38_resident_daemon.py serve' && { echo 'ERROR,a resident daemon is already running'; exit 4; }
apps=$(nvidia-smi --query-compute-apps=pid --format=csv,noheader | wc -l)
[[ $apps -eq 0 ]] || { echo "ERROR,$apps CUDA compute processes already running"; exit 4; }
sha256sum "$PLE_SHARDS" "$TOPOLOGY_SRC" "$PRELOAD" > "$RUN/input-hashes.sha256"

# ---- topology: TSV rows verified against the live scan and live bridge windows ----------------
source "$REPO/src/qwen38_pci_env.sh"
q38_discover "$TOPOLOGY_SRC" || exit 3
cp "$TOPOLOGY_SRC" "$RUN/topology.tsv"
TOPOLOGY=$RUN/topology.tsv
python3 - "$TOPOLOGY" <<'PY'
import sys
for r in (l.split('\t') for l in open(sys.argv[1]).read().splitlines() if l):
    up, start, end, ingress = r[1], int(r[4], 0), int(r[5], 0), int(r[6], 0)
    # resource lines 13..15 of a bridge are its windows; take the 64-bit prefetchable one.
    wins = [tuple(int(x, 16) for x in l.split()[:2]) for l in
            open(f'/sys/bus/pci/devices/{up}/resource').read().splitlines()[13:16]]
    assert (start, end) in wins, f'{up}: TSV window {start:#x}-{end:#x} not among live bridge windows {[(hex(a),hex(b)) for a,b in wins]}'
    assert start <= ingress <= end, f'{up}: ingress {ingress:#x} outside window'
    print(f'TOPOLOGY_WINDOW,{up},start,{start:#x},end,{end:#x},ingress,{ingress:#x},live_match,1')
PY
echo "RTX,$Q38_RTX_BDF,$Q38_RTX_UUID,outer_port,$Q38_RTX_OUTER_PORT"

# ---- multicast mapper (m10_qwen_mc_wc6): bases from the verified topology, built before the daemon --
python3 "$REPO/src/check_qwen38_mc_premap.py" "$TOPOLOGY" "$RUN/mc-premap" || { echo 'ERROR,pre-daemon multicast check failed'; exit 3; }
MODDIR=$RUN/m10_qwen_mc_module; mkdir -p "$MODDIR"
cp "$REPO/transport/driver/m10_qwen_mc_wc6.c" "$MODDIR/"
python3 - "$TOPOLOGY" "$MODDIR/Makefile" <<'PYGEN'
import sys
vals = [int(l.split('\t')[6], 0) for l in open(sys.argv[1]).read().splitlines() if l]
assert len(vals) == 6
open(sys.argv[2], 'w').write('obj-m += m10_qwen_mc_wc6.o\n' + ''.join(f'ccflags-y += -DQWEN_MC_BASE_{i}=0x{v:x}ULL\n' for i, v in enumerate(vals)))
print('MAPPER_BASES', *[hex(v) for v in vals])
PYGEN
MAPPER_STATE=/run/q38-m10-mapper.state   # written at insmod: bases + srcversion + module path
make -C "/lib/modules/$(uname -r)/build" M="$MODDIR" modules > "$RUN/m10-mapper-build.log" 2>&1 || { tail -20 "$RUN/m10-mapper-build.log"; exit 5; }
sha256sum "$MODDIR/m10_qwen_mc_wc6.c" "$MODDIR/Makefile" "$MODDIR/m10_qwen_mc_wc6.ko" > "$RUN/m10-mapper-hashes.sha256"
want_bases=$(grep -o 'QWEN_MC_BASE_[0-9]=0x[0-9a-f]*' "$MODDIR/Makefile" | tr '\n' ' ')
want_src=$(modinfo -F srcversion "$MODDIR/m10_qwen_mc_wc6.ko")
if [[ -e /sys/module/m10_qwen_mc_wc6 ]]; then
  # Reuse only a mapper this script loaded, with identical bases and source, and no users.
  [[ -c /dev/m10_qwen_mc_wc6 && $(cat /sys/module/m10_qwen_mc_wc6/refcnt) == 0 ]] || { echo 'ERROR,loaded mapper has no node or active references'; exit 4; }
  [[ -r $MAPPER_STATE ]] || { echo 'ERROR,mapper loaded but no state file records its bases; refusing to reuse'; exit 4; }
  got_bases=$(sed -n 's/^bases=//p' "$MAPPER_STATE"); got_src=$(sed -n 's/^srcversion=//p' "$MAPPER_STATE")
  live_src=$(cat /sys/module/m10_qwen_mc_wc6/srcversion)
  [[ $got_bases == "$want_bases" && $got_src == "$want_src" && $live_src == "$want_src" ]] \
    || { echo "ERROR,loaded mapper differs (bases '$got_bases' vs '$want_bases', srcversion $live_src vs $want_src)"; exit 4; }
  echo "MAPPER_REUSED,srcversion,$live_src,state,$MAPPER_STATE"
else
  [[ ! -e /dev/m10_qwen_mc_wc6 ]] || { echo 'ERROR,mapper node exists without loaded module'; exit 5; }
  insmod "$MODDIR/m10_qwen_mc_wc6.ko"
  printf 'bases=%s\nsrcversion=%s\nmodule=%s\nloaded_utc=%s\n' "$want_bases" "$want_src" "$MODDIR/m10_qwen_mc_wc6.ko" "$(date -u --iso-8601=s)" > "$MAPPER_STATE"
fi
[[ -c /dev/m10_qwen_mc_wc6 ]] || { echo 'ERROR,mapper loaded without char node'; exit 5; }
stat -c 'MAPPER_DEVICE,mode=%a,uid=%u,gid=%g,rdev=%t:%T' /dev/m10_qwen_mc_wc6
echo "MAPPER_LOADED,srcversion,$(cat /sys/module/m10_qwen_mc_wc6/srcversion)"

# ---- build services (current source, includes the ECAP-walk change) and worker cubins -----------
OUT_DIR=$RUN/build/resident-services bash "$REPO/src/build_qwen38_resident_services.sh" > "$RUN/build/services.log" 2>&1 \
  || { tail -30 "$RUN/build/services.log"; echo 'ERROR,service build failed'; exit 5; }
inc=( -I"$REPO/transport/driver/abi/580.178.04" -I"$NVIDIA_SRC/common/inc" -I"$NVIDIA_SRC/nvidia-uvm" -I"$REPO/src" )
nv() { "$NVCC" -O2 -std=c++17 -arch=sm_50 "$@" "${inc[@]}" --resource-usage -cubin >> "$RUN/build/cubins.log" 2>&1; }
nv "$REPO/src/resident_routed_kernel_v3_module.cu" -o "$RUN/build/routed_v3.cubin"
"$NVCC" -O2 -std=c++17 -gencode arch=compute_50,code=sm_50 -Xcompiler -pthread "$REPO/src/cuda_load_stage9_ple.cu" \
  -o "$PLE" >> "$RUN/build/cubins.log" 2>&1 || { echo "ERROR,PLE owner build failed"; exit 5; }
nv "$REPO/src/resident_shared_kernel_v2_module.cu" -o "$RUN/build/shared_v2.cubin"
"$NVCC" -std=c++17 -O2 -I"$REPO/transport/driver/abi/580.178.04" -I"$REPO/src" -I"$NVIDIA_SRC/common/inc" -I"$NVIDIA_SRC/nvidia-uvm" \
  "$REPO/src/q38_peer_route_tool.cu" -o "$RUN/build/q38_peer_route_tool" >> "$RUN/build/cubins.log" 2>&1 || { echo "ERROR,route tool build failed"; exit 5; }
OUT="$RUN/build/libqwen38_direct_moe.so" bash "$REPO/src/build_qwen38_direct_moe.sh" >> "$RUN/build/cubins.log" 2>&1 \
  || { echo "ERROR,direct-MoE library build failed"; exit 5; }   # loaded by the vLLM worker (sitecustomize.py)
sha256sum "$RUN"/build/q38_peer_route_tool "$RUN"/build/libqwen38_direct_moe.so "$RUN"/build/resident-services/* "$RUN"/build/*.cubin "$PLE" "$REPO"/src/qwen38_parallel_load.cuh "$REPO"/src/cuda_load_stage9_ple.cu "$REPO"/src/resident_dynamic_routed_v3.cuh \
  "$REPO"/src/resident_dynamic_shared_v2.cuh > "$RUN/build-hashes.sha256"
echo 'BUILD_OK,services,routed_v3.cubin,shared_v2.cubin,ple_owner'

# ---- allocation map: BDF refreshed by UUID ------------------------------------------------
python3 - "$PLACEMENT/gpu_allocations.tsv" "$RUN/gpu_allocations.current.tsv" <<'PY'
import csv, subprocess, sys
from pathlib import Path
src, dst = map(Path, sys.argv[1:])
current = {}
for row in csv.reader(subprocess.check_output(['nvidia-smi', '--query-gpu=pci.bus_id,uuid', '--format=csv,noheader'],
                                              text=True).splitlines(), skipinitialspace=True):
    bdf, uuid = (v.strip() for v in row)
    current[uuid] = bdf
rows = []
for line in src.read_text().splitlines():
    f = line.split('\t')
    if len(f) != 6: raise SystemExit(f'malformed allocation row: {line!r}')
    if f[2] not in current: raise SystemExit(f'UUID absent from current CUDA inventory: {f[2]}')
    f[1] = current[f[2]]; rows.append((int(f[0]), f))
rows.sort()
if len(rows) != 25 or [i for i, _ in rows] != list(range(25)): raise SystemExit('allocation table does not cover 25 devices')
if len({f[1] for _, f in rows}) != 25: raise SystemExit('duplicate live BDF')
dst.write_text('\n'.join('\t'.join(f) for _, f in rows) + '\n')
for i, f in rows: print('ALLOC_BDF_REFRESH', i, f[2], f[1], f[3], f[4], f[5])
PY
nvidia-smi --query-gpu=pci.bus_id,name,temperature.gpu,memory.used,pstate --format=csv,noheader > "$RUN/gpu-before.csv"

# ---- start the daemon -----------------------------------------------------------------
t0=$(date +%s.%N)
setsid env \
  CUDA_HOME=/usr/local/cuda-12.4 \
  Q38_EXPERIMENTAL_PRIVILEGED_WORKERS=1 \
  Q38_MULTICAST_GROUPS="$TOPOLOGY" \
  Q38_MULTICAST_SNAPSHOT="$RUN/daemon-pex-snapshots" \
  Q38_RM_TRACE_PRELOAD="$PRELOAD" \
  Q38_HEARTBEAT_SECONDS=0 \
  Q38_RESIDENT_IDLE_DEADLINE_CYCLES=103680000000000 \
  PLE_LM_HEAD_FILE="$LM_HEAD_FILE" \
  "$PY" -u "$REPO/src/qwen38_resident_daemon.py" serve \
    --model "$MODEL" \
    --bundle-root "$RUN/transient-bundles" \
    --persistent-bundle-root "$BUNDLES" \
    --stream-pack \
    --routed-service "$RUN/build/resident-services/resident_routed_service" \
    --shared-service "$RUN/build/resident-services/resident_shared_service" \
    --ple-binary "$PLE" \
    --allocations "$RUN/gpu_allocations.current.tsv" \
    --shards "$PLE_SHARDS" \
    --startup-rows "$PLACEMENT/ple_query_rows.txt" \
    --runtime-dir "$RUNTIME" --out "$RUN/daemon" \
    --direct-library "$RUN/build/libqwen38_direct_moe.so" \
    --direct-groups "$TOPOLOGY" \
    --peer-route-tool "$RUN/build/q38_peer_route_tool" \
    --layer-count "$NL" \
    >"$RUN/daemon.log" 2>&1 < /dev/null &
echo "$!" > "$RUN/launcher_pid.txt"
state=; prev=
printf 'phase\tfirst_seen_epoch\n' > "$RUN/load-phases.tsv"
printf 'LAUNCH\t%s\n' "$t0" >> "$RUN/load-phases.tsv"
for _ in $(seq 1 7200); do   # 2 h deadline at ~1 s polling (phase timing resolution)
  if [[ -S $RUNTIME/control.sock ]] && $DAEMON status --control-socket "$RUNTIME/control.sock" > "$RUN/daemon-status-starting.json" 2>/dev/null; then
    state=$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1])).get("status",""))' "$RUN/daemon-status-starting.json")
    if [[ $state != "$prev" ]]; then printf '%s\t%s\n' "$state" "$(date +%s.%N)" >> "$RUN/load-phases.tsv"; echo "LOAD_PHASE,$state,$(date -u +%T)"; prev=$state; fi
    [[ $state == READY ]] && break
    [[ $state == ERROR* || $state == STOPPED ]] && { cat "$RUN/daemon-status-starting.json"; exit 6; }
  fi
  if ! kill -0 "$(cat "$RUN/launcher_pid.txt")" 2>/dev/null; then tail -100 "$RUN/daemon.log"; echo 'ERROR,daemon exited during startup'; exit 7; fi
  sleep 1
done
[[ $state == READY ]] || { tail -100 "$RUN/daemon.log"; echo 'ERROR,daemon startup deadline'; exit 8; }
t1=$(date +%s.%N)
echo "DAEMON_LOAD_SECONDS,$(python3 -c "print(f'{$t1-$t0:.1f}')")"
$DAEMON status --control-socket "$RUNTIME/control.sock" > "$RUN/daemon-status-ready.json"
NL="$NL" python3 - "$RUN/daemon-status-ready.json" <<'PY'
import json, os, sys
NL = int(os.environ["NL"])
x = json.load(open(sys.argv[1]))
assert x.get('status') == 'READY' and x.get('layers_ready') == NL, x.get('layers_ready')
assert x.get('loaded_layer_ids') == list(range(NL))
assert x.get('direct_moe_setup', {}).get('status') == 'PASS'
assert x.get('direct_moe_setup', {}).get('gpu_map_records') == 24
print('DAEMON_READY', x.get('pid'), 'layers', x.get('layers_ready'), 'expert_load_seconds', x.get('expert_load_seconds'))
PY
if [[ -n $LM_HEAD_FILE ]]; then
  grep -h '^LM_HEAD_LOADED' "$RUN/daemon/resident_ple.log" || { echo 'ERROR,PLE owner did not load the LM head'; exit 6; }
  n=$(grep -c $'^LMHEAD\t[0-9]' "$RUN/daemon/ple_ipc/ple_ipc_manifest.tsv" || true)
  [[ $n -eq 24 ]] || { echo "ERROR,PLE manifest has $n LMHEAD slices, expected 24"; exit 6; }
  echo "LM_HEAD_ON_M10,slices,$n"
fi
# Load timing: PLE = LOADING_PLE -> LOADING_EXPERTS (phase polling, ~1 s resolution);
# experts = the daemon's own expert_load_seconds; per-layer stream-pack times from status.
python3 - "$RUN/load-phases.tsv" "$RUN/daemon-status-ready.json" "$PLE_SHARDS" "$BUNDLES" "$RUN/load-timing.json" <<'PY'
import json, os, sys
ph = {}
for l in open(sys.argv[1]).read().splitlines()[1:]:
    k, t = l.split('\t'); ph.setdefault(k, float(t))
x = json.load(open(sys.argv[2]))
ple_bytes = sum(int(l.split('\t')[4]) for l in open(sys.argv[3]).read().splitlines() if l)
exp_bytes = sum(os.path.getsize(os.path.join(d, f)) for d, _, fs in os.walk(sys.argv[4]) for f in fs)
ple_s = ph['LOADING_EXPERTS'] - ph['LOADING_PLE'] if {'LOADING_PLE', 'LOADING_EXPERTS'} <= ph.keys() else None
exp_s = x.get('expert_load_seconds')
layers = (x.get('stream_pack') or {}).get('layers') or []
lt = [l.get('elapsed_seconds') for l in layers if l.get('elapsed_seconds') is not None]
out = {'phases_epoch': ph, 'ple_seconds': ple_s, 'ple_bytes': ple_bytes,
       'ple_gb_per_s': ple_bytes / ple_s / 1e9 if ple_s else None,
       'expert_seconds': exp_s, 'expert_bundle_bytes': exp_bytes,
       'expert_gb_per_s': exp_bytes / exp_s / 1e9 if exp_s else None,
       'launch_to_ready_seconds': ph.get('READY', 0) - ph['LAUNCH'],
       'layer_seconds': {'count': len(lt), 'min': min(lt, default=None), 'max': max(lt, default=None),
                         'mean': sum(lt) / len(lt) if lt else None}}
json.dump(out, open(sys.argv[5], 'w'), indent=2)
f = lambda v, d=1: 'n/a' if v is None else f'{v:.{d}f}'
print(f"LOAD_TIMING,ple_s,{f(ple_s)},ple_GB,{ple_bytes/1e9:.1f},ple_GBps,{f(out['ple_gb_per_s'],2)},"
      f"expert_s,{f(exp_s)},expert_GB,{exp_bytes/1e9:.1f},expert_GBps,{f(out['expert_gb_per_s'],2)},"
      f"launch_to_ready_s,{out['launch_to_ready_seconds']:.1f},layer_s_mean,{f(out['layer_seconds']['mean'],2)},"
      f"layer_s_max,{f(out['layer_seconds']['max'],2)}")
PY
grep -h '^PLE_LOAD_MODE\|^PLE_PARALLEL_LOAD\|^PASS_STAGE9_PLE_RESIDENT_LOAD_READBACK' "$RUN/daemon/resident_ple.log" || true
grep -h 'ROUTED_PARALLEL_LOAD' "$RUN"/daemon/resident_expert_logs/* 2>/dev/null | python3 -c '
import sys
r=[l.strip().split(",") for l in sys.stdin if l.startswith("ROUTED_PARALLEL_LOAD,layer")]
s=[float(x[4]) for x in r]; g=[float(x[8]) for x in r]
print(f"ROUTED_PARALLEL_LOAD_SUMMARY,layers,{len(r)},seconds_sum,{sum(s):.1f},seconds_mean,{sum(s)/len(s):.2f},GBps_mean,{sum(g)/len(g):.2f}" if r else "ROUTED_PARALLEL_LOAD_SUMMARY,none_found")' || true

# ---- park, load optimized kernels while parked ------------------------------------------------
$DAEMON park-workers --control-socket "$RUNTIME/control.sock" --clients-quiesced > "$RUN/park-workers.json"
$DAEMON reload-kernels --control-socket "$RUNTIME/control.sock" --routed-image "$RUN/build/routed_v3.cubin" \
  --shared-image "$RUN/build/shared_v2.cubin" --clients-quiesced > "$RUN/reload-kernels.json"
python3 -c "import json,sys;r=json.load(open(sys.argv[1]));assert r.get('status')=='ok',r;print('RELOAD_KERNELS,ok,routed_v3,shared_v2')" "$RUN/reload-kernels.json" \
  || { echo 'ERROR,kernel reload failed; daemon left running with its built-in kernels'; exit 10; }
$DAEMON status --control-socket "$RUNTIME/control.sock" > "$RUN/daemon-status-parked.json"
NL="$NL" python3 - "$RUN/daemon-status-parked.json" "$RUN/build/routed_v3.cubin" <<'PY'
import json, os, sys
NL = int(os.environ["NL"])
x = json.load(open(sys.argv[1]))
assert x.get('status') == 'READY' and x.get('layers_ready') == NL, x
assert x.get('worker_activity') == 'PARKED', x.get('worker_activity')
kr = x.get('kernel_reload') or {}
print('KERNEL_RELOAD_STATE', json.dumps(kr)[:400])
assert sys.argv[2] in json.dumps(kr), 'status does not record the v3 routed image'
print('DAEMON_PARKED_READY', x.get('pid'), 'layers', NL, 'workers', 'PARKED')
PY

cat > "$RUN/daemon.env" <<EOF
Q38_FULL_RUN=$RUN
Q38_FULL_CONTROL_SOCKET=$RUNTIME/control.sock
Q38_FULL_MOE_SOCKET=$RUNTIME/moe.sock
Q38_FULL_PLE_SOCKET=$RUNTIME/ple.sock
Q38_FULL_MANIFEST=$RUN/daemon/gpu_mappings.manifest
Q38_FULL_TOPOLOGY=$TOPOLOGY
Q38_FULL_SNAPSHOT=$RUN/daemon-pex-snapshots
Q38_FULL_BUNDLES=$BUNDLES
EOF
[[ -f $RUN/daemon/gpu_mappings.manifest ]] || { echo 'ERROR,daemon did not write gpu_mappings.manifest'; exit 11; }
ln -sfn "$RUN" "$M10DIS_RUNS/qwen38-full-daemon/latest"
echo "FULL_DAEMON_RUN_OK,$RUN"
