#!/usr/bin/env bash
# Ordered restart of the resident M10 daemon (operator-approved, 2026-11-03: LM-head slices in the PLE
# owner). Follows RESIDENT_M10_DAEMON_RUNBOOK.md "Stop the resident owner": no clients, STOP, wait for
# the daemon to exit with STOPPED (never kill), confirm no process of the old build remains, then run
# run_qwen38_full_daemon_start.sh (builds services, PLE owner and cubins from current source).
# Queue: pueue add --group gpu --label <label> -- sudo -n bash <this script>
set -Eeuo pipefail
source "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/m10dis_env.sh"
ROOT=$M10DIS_HOME
REPO=$M10DIS_HOME
ENV_FILE=${Q38_FULL_DAEMON_ENV:-$M10DIS_RUNS/qwen38-full-daemon/latest/daemon.env}
PY=$REPO/.venv-vllm/bin/python
DAEMON="$PY $REPO/src/qwen38_resident_daemon.py"
LOG=$M10DIS_RUNS/qwen38-full-daemon/restart-$(date -u +%Y%m%dT%H%M%SZ).log
[[ $EUID -eq 0 ]] || { echo 'ERROR,run as root (sudo -n)'; exit 2; }
exec > >(tee -a "$LOG") 2>&1
echo "RESTART_LOG,$LOG"
# shellcheck disable=SC1090
source "$ENV_FILE"
OLD=$Q38_FULL_RUN; CTRL=$Q38_FULL_CONTROL_SOCKET
echo "OLD_DAEMON_RUN,$OLD"
if pgrep -f run_vllm_resident_m10_client.py >/dev/null; then echo 'ERROR,vLLM client running'; exit 4; fi
$DAEMON status --control-socket "$CTRL" > "$OLD/daemon-status-before-stop.json"
PID=$(python3 - "$OLD/daemon-status-before-stop.json" <<'PY'
import json, sys
s = json.load(open(sys.argv[1]))
pc = (s.get("peer_client") or {}).get("status")
assert pc in (None, "NONE"), f"peer client registered: {pc}"
assert s.get("status") == "READY", s.get("status")
print(s["pid"])
PY
)
echo "STOPPING,pid,$PID"
$DAEMON stop --control-socket "$CTRL" | tee "$OLD/stop-response.json"
for i in $(seq 1 600); do kill -0 "$PID" 2>/dev/null || break; sleep 1; done
if kill -0 "$PID" 2>/dev/null; then echo "ERROR,daemon $PID still running after 600 s; not killing (runbook)"; exit 5; fi
python3 -c "import json,sys;s=json.load(open(sys.argv[1]));print('OLD_STATE',s.get('status'));assert s.get('status')=='STOPPED',s.get('status')" "$OLD/daemon/resident_state.json" \
  || { echo 'ERROR,old daemon did not record STOPPED'; exit 5; }
left=$(for p in /proc/[0-9]*; do e=$(readlink "$p/exe" 2>/dev/null || true); [[ $e == "$OLD"/build/* ]] && echo "${p#/proc/}:$e"; done || true)
[[ -z $left ]] || { echo "ERROR,old build processes remain: $left"; exit 5; }
echo "OLD_DAEMON_STOPPED,$(date -u +%T)"
# Q38_RESTART_STOP_ONLY=1: stop only (e.g. offline M10 tests need the cards); start later with run_qwen38_full_daemon_start.sh
[[ ${Q38_RESTART_STOP_ONLY:-0} == 1 ]] && { echo "STOP_ONLY_DONE"; exit 0; }
exec bash "$REPO/src/run_qwen38_full_daemon_start.sh"
