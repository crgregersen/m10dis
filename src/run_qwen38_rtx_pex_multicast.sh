#!/usr/bin/env bash
set -euo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/m10dis_env.sh"
ROOT=$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)
if [[ $(id -u) -ne 0 ]]; then
  echo "run as root: sudo $0 ACTIVE_GROUPS.tsv GPU_MAPPINGS.tsv CONTROL_SOCKET PEX_SNAPSHOT_ROOT" >&2
  exit 2
fi
if [[ $# -ne 4 ]]; then
  echo "usage: $0 ACTIVE_GROUPS.tsv GPU_MAPPINGS.tsv CONTROL_SOCKET PEX_SNAPSHOT_ROOT" >&2
  exit 2
fi
GROUPS_TSV=$(realpath "$1")
GPU_MAPS=$(realpath "$2")
CONTROL_SOCKET=$(realpath -m "$3")
SNAPSHOT_ROOT=$(realpath "$4")
# Discovered by scanning sysfs and checked against the groups TSV (never hardcoded).
source "$ROOT/src/qwen38_pci_env.sh"
q38_discover "$GROUPS_TSV" || exit 3
RTX_BDF=$Q38_RTX_BDF
export Q38_RTX_BDF
MODE=${Q38_RTX_MODE:-both}
PEER_RESOURCE=${Q38_RTX_PEER_RESOURCE:-}
MODDIR="$ROOT/transport/rtx_mc_dma"
KO="$MODDIR/rtx_mc_dma_map.ko"
BIN="$ROOT/build/qwen38_rtx_pex_multicast"
RUN="$M10DIS_RUNS/qwen38-rtx-m10-multicast/$(date -u +%Y%m%dT%H%M%SZ)"
mkdir -p "$RUN"
exec > >(tee -a "$RUN/run.log") 2>&1

echo "RUN_DIR,$RUN"
echo "POC_MODE,$MODE"
if [[ -n ${Q38_ACS_TRIAL_FILE:-} ]]; then
  cp "$Q38_ACS_TRIAL_FILE" "$RUN/acs-controls.tsv"
  echo "ACS_ROUTE_TRIAL,temporary_local_redirect_clear,controls,$RUN/acs-controls.tsv"
fi
echo "KERNEL,$(uname -a)"
modinfo nvidia | rg '^(filename|version|srcversion|vermagic):' || true
modinfo nvidia_uvm | rg '^(filename|version|srcversion|vermagic):' || true
nvidia-smi --query-gpu=index,name,pci.bus_id,uuid,memory.used --format=csv
lspci -tv

python3 - "$ROOT" "$GROUPS_TSV" "$GPU_MAPS" "$SNAPSHOT_ROOT" "$CONTROL_SOCKET" <<'PY'
import json, pathlib, subprocess, sys
root, groups_path, maps_path, snapshots, sock = map(pathlib.Path, sys.argv[1:])
rows = [line.rstrip('\n').split('\t') for line in groups_path.read_text().splitlines() if line]
assert len(rows) == 6, f"expected six active M10 groups, got {len(rows)}"
for i, r in enumerate(rows):
    assert len(r) == 7 and len(r[2].split(',')) == 4 and len(r[3].split(',')) == 4
    lo, hi, base = (int(x, 0) for x in r[4:7])
    assert base % 65536 == 0 and lo <= base and base + 65535 <= hi
    plan = json.loads((snapshots / f'group{i}' / 'plan.json').read_text())
    assert plan['base'] == base and plan['window'] == 65536
    assert len(plan['ports']) == 5
manifest = maps_path.read_text().splitlines()
assert manifest[0] == 'QWEN38_GPU_MAPPINGS_MANIFEST\t1'
records = [x.split('\t') for x in manifest if x and x[0].isdigit()]
assert len(records) == 24, f"expected 24 daemon GPU mappings, got {len(records)}"
for r in records:
    g, w = int(r[0]), int(r[1])
    assert r[2] == rows[g][3].split(',')[w]
    assert int(r[6], 0) % 65536 == 0
status = json.loads(subprocess.check_output([sys.executable, str(root / 'src/qwen38_resident_daemon.py'),
                                             'status', '--control-socket', str(sock)], text=True))
assert status['status'] == 'READY' and status['layers_ready'] == 48 and status['worker_activity'] == 'PARKED', status
assert status['multicast_snapshot_root'] == str(snapshots)
print(f"ACTIVE_DAEMON_CHECK,PASS,pid,{status['pid']},layers,{status['layers_ready']},workers,{status['worker_activity']}")
PY

if [[ "$MODE" == "rtx-direct" && -z "$PEER_RESOURCE" ]]; then
  PEER_RESOURCE=$(python3 - "$GROUPS_TSV" <<'PY'
import pathlib, sys
row = pathlib.Path(sys.argv[1]).read_text().splitlines()[0].split('\t')
bdf = row[3].split(',')[0]
resource = pathlib.Path('/sys/bus/pci/devices', bdf, 'resource').read_text().splitlines()[1].split()[0]
value = int(resource, 16)
assert value and value % 65536 == 0, (bdf, hex(value))
print(hex(value))
PY
)
  echo "DIRECT_PEER_RESOURCE,$PEER_RESOURCE"
fi

cp "$GROUPS_TSV" "$RUN/active-groups.tsv"
cp "$GPU_MAPS" "$RUN/gpu_mappings.manifest"
cp -a "$SNAPSHOT_ROOT" "$RUN/active-pex-plans"
cp "$ROOT/src/qwen38_rtx_pex_multicast.cu" "$ROOT/src/build_qwen38_rtx_pex_multicast.sh" \
   "$MODDIR/rtx_mc_dma_map.c" "$MODDIR/Makefile" "$RUN/"
make -C "$MODDIR" KDIR="/lib/modules/$(uname -r)/build" clean all 2>&1 | tee "$RUN/module-build.log"
"$ROOT/src/build_qwen38_rtx_pex_multicast.sh" 2>&1 | tee "$RUN/cuda-build.log"
sha256sum "$KO" > "$RUN/module.sha256"
sha256sum "$ROOT/src/qwen38_rtx_pex_multicast.cu" "$BIN" \
  "$ROOT/build/qwen38_rtx_pex_multicast.sass" > "$RUN/cuda.sha256"
date --iso-8601=ns > "$RUN/kernel-log-start.txt"
dmesg --ctime > "$RUN/dmesg-before.txt"

run_groups() {
  local count=$1 label=$2
  local -a fields
  mapfile -t fields < <(python3 - "$GROUPS_TSV" "$count" <<'PY'
import sys
rows=[x.rstrip('\n').split('\t') for x in open(sys.argv[1]) if x.strip()]
n=int(sys.argv[2]); rows=rows[:n]
print(','.join(r[6] for r in rows))
print(','.join(r[4] for r in rows))
print(','.join(r[5] for r in rows))
PY
)
  [[ ${#fields[@]} -eq 3 ]] || { echo "ERROR,cannot construct module arguments"; return 1; }
  if lsmod | awk '$1=="rtx_mc_dma_map" {found=1} END {exit !found}'; then
    echo "ERROR,rtx_mc_dma_map already loaded"; return 1
  fi
  echo "MODULE_LOAD,groups,$count,requester,$RTX_BDF,resource,active-PEX-ingress"
  local -a module_args=("requester_bdf=$RTX_BDF" "groups=$count"
    "ingress=${fields[0]}" "branch_start=${fields[1]}" "branch_end=${fields[2]}")
  if [[ -n "$PEER_RESOURCE" ]]; then module_args+=("peer_resource=$PEER_RESOURCE"); fi
  insmod "$KO" "${module_args[@]}"
  local rc=0
  set +e
  "$BIN" "$GROUPS_TSV" "$GPU_MAPS" /sys/module/rtx_mc_dma_map/parameters \
    "$count" "$MODE" "$RUN" 2>&1 | tee "$RUN/${label}.log"
  rc=${PIPESTATUS[0]}
  set -e
  if ! rg -q '^SAFE_TO_UNMAP,all acquired GPU and PTE objects released$' "$RUN/${label}.log"; then
    echo "PRESERVE,module remains loaded because cleanup safety was not proved"
    return 1
  fi
  rmmod rtx_mc_dma_map
  if [[ $rc -ne 0 ]]; then echo "ERROR,client_exit,$rc"; return "$rc"; fi
  rg '^RUN_RESULT,PASS$' "$RUN/${label}.log"
}

if [[ "$MODE" == "rtx-bus-all" ]]; then
  run_groups 6 all_six_groups
else
  run_groups 1 group0
fi
if [[ "$MODE" == "both" ]]; then
  run_groups 6 all_six_groups
fi

date --iso-8601=ns > "$RUN/kernel-log-end.txt"
dmesg --ctime > "$RUN/dmesg-after.txt"
diff -u "$RUN/dmesg-before.txt" "$RUN/dmesg-after.txt" > "$RUN/dmesg-diff.txt" || true
python3 - "$ROOT" "$CONTROL_SOCKET" "$RUN/daemon-status-after.json" <<'PY'
import json, pathlib, subprocess, sys
root, sock, out = map(pathlib.Path, sys.argv[1:])
s=json.loads(subprocess.check_output([sys.executable, str(root/'src/qwen38_resident_daemon.py'),
    'status','--control-socket',str(sock)],text=True))
out.write_text(json.dumps(s,indent=2)+'\n')
assert s['status']=='READY' and s['layers_ready']==48 and s['worker_activity']=='PARKED',s
print(f"ACTIVE_DAEMON_AFTER,PASS,pid,{s['pid']},layers,{s['layers_ready']},workers,{s['worker_activity']}")
PY

cuobjdump --dump-sass "$BIN" > "$RUN/loaded-sass.txt"
rg -n -C 8 'tx_multicast_pages|STG|MEMBAR' "$RUN/loaded-sass.txt" > "$RUN/tx-sass-extract.txt" || true
{
  echo '# RTX-originated PEX8747 multicast PoC'; echo
  echo "Run directory: \`$RUN\`"; echo
  echo '## Method'; echo
  echo '- Used the live six card-local PEX8747 mappings already owned by the resident daemon. One RTX kernel block wrote to each card ingress; each PEX8747 replicated the stream to four GM107 BAR1 mappings.'
  echo '- Workers were parked and the daemon remained READY with all 48 layers resident. IOMMU and NVIDIA registry settings were unchanged.'
  if [[ -n ${Q38_ACS_TRIAL_FILE:-} ]]; then
    if [[ $MODE == "rtx-bus-all" ]]; then
      echo '- A bounded route trial temporarily cleared only ACS request/completion redirection on the RTX downstream port and the six participating M10 downstream ports. Exact original words and restoration readbacks are in `acs-controls.tsv` and the parent trial log.'
    else
      echo '- A bounded outer-PEX8749 route trial temporarily cleared only ACS request/completion redirection on the RTX downstream port and the tested M10 downstream port. Exact original words and restoration readbacks are in `acs-controls.tsv` and the parent trial log.'
    fi
  else
    echo '- No PEX/ACS/IOMMU/NVIDIA setting or module was changed.'
  fi
  echo '- Payload was the production-shaped 12 KiB request window: 2,560 FP32 H words, 10 expert IDs, 10 FP32 route weights, and deterministic padding. Scratch allocation offset was 0xC000, outside the documented daemon request fields ending at 0xA878.'
  echo '- The exact original scratch words were saved through individual GPU BAR1 resource mappings and restored byte-for-byte after requester PTE restoration.'
  if [[ $MODE == "rtx-bus" || $MODE == "rtx-bus-all" ]]; then
    echo '- RTX PTEs used the live PEX ingress PCI bus addresses for local switch routing. A separate DMA_FROM_DEVICE IOVA mapping remained active as a comparison/control but was not encoded in the PEX ingress PTE. A coherent host-RAM IOVA remained the positive control.'
  else
    echo '- CPU mode used the existing WC multicast mapper plus sfence. RTX IOVA mode used a fresh Linux DMA_FROM_DEVICE resource map for each group, a requester SYS PTE generated/validated by the matching UVM HAL, and a CUDA kernel issuing RTX stores.'
  fi
  echo
  echo '## Results'; echo
  echo '| Run | Sender | Card groups | GPU destinations | Epochs | Bytes injected/epoch | Verified receiver records |';
  echo '|---|---|---:|---:|---:|---:|---|';
  for f in group0 all_six_groups; do
    [[ -f "$RUN/$f.log" ]] || continue
    for mode in CPU RTX; do
      if ! rg -q "^(EPOCH_RESULT|VERIFY),mode,$mode," "$RUN/$f.log"; then continue; fi
      count=1; [[ $f == all_six_groups ]] && count=6
      injected=$((count*12288))
      echoes=$(rg -c "^EPOCH_RESULT,mode,$mode," "$RUN/$f.log" || true)
      result=$(rg -c "^VERIFY,mode,$mode," "$RUN/$f.log" || true)
      printf '| %s | %s | %d | %d | %s | %d | %s records; inspect log for mismatch counts |\n' "$f" "$mode" "$count" "$((count*4))" "$echoes" "$injected" "$result"
    done
  done
  echo
  echo 'Per-epoch CPU store/fence timing, RTX CUDA event timing, host submit-to-completion timing, and full BAR1 readback records are in the executed group log(s) above.'
  echo
  echo '## Interpretation'; echo
  if [[ $MODE == "rtx-bus" || $MODE == "rtx-bus-all" ]]; then
    echo 'A passing RTX bus-address mode demonstrates RTX-originated writes to the PEX ingress PCI bus apertures under the temporary local-route ACS configuration. The IOMMU IOVAs were retained as diagnostics but were not encoded in the PEX-ingress PTEs. Matching CPU BAR1 views prove multicast delivery to the 4 or 24 VRAM BAR views tested; they do not prove resident worker kernels consumed these scratch bytes or that vLLM switched ingress away from the CPU.'
  else
    echo 'A passing six-group RTX IOVA run would demonstrate RTX-originated writes through the IOMMU mappings into all six local PEX8747 ingress windows and 24 GPU BAR1 views. It does not by itself prove resident worker consumption or vLLM ingress integration.'
  fi
  echo
  echo '## Restoration'; echo
  echo '- Requester PTEs were restored in reverse order and verified against their original entries before the Linux DMA resource mappings were unmapped.'
  echo '- Receiver scratch words were restored and read back exactly. The temporary `rtx_mc_dma_map` module was removed after the client emitted `SAFE_TO_UNMAP`.'
  echo '- CUDA sanity runs allocation/kernel/readback on all 25 devices. Daemon status after the experiment is saved in `daemon-status-after.json`; kernel changes are in `dmesg-diff.txt`.'
} > "$RUN/QWEN38_RTX_PEX_MULTICAST_POC_REPORT.md"

echo "RESULT,PASS,artifacts,$RUN"
