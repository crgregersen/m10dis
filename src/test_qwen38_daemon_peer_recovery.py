#!/usr/bin/env python3
"""Offline test of the daemon peer-client commands (no GPU, no UVM).

Runs the real ControlServer on a temporary Unix socket with a fake worker manager and a fake
q38_peer_route_tool, and exercises REGISTER/UNREGISTER/RECOVER_PEER_CLIENT, LIST_ROUTE_RECORDS
and the STOP guard exactly as documented in docs/RESIDENT_DAEMON_INTERFACE.md.
Usage: python3 test_qwen38_daemon_peer_recovery.py
"""
import json, os, subprocess, sys, tempfile, threading, time
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import qwen38_resident_daemon as d

checks = {}
tmp = Path(tempfile.mkdtemp(prefix="q38_peer_test_"))
state = tmp / "state.json"; state.write_text(json.dumps({"status": "READY", "layers_ready": 48}))
tool = tmp / "fake_tool.sh"
tool.write_text("""#!/bin/bash
if [[ $1 == scan ]]; then
  echo 'ACTIVE_RECORD,id,1,slot,0,gpu,00,va,0x1,pte_phys,0x2,original,0x3,temporary,0x4,destination,0x5,va_space_state,2'
  echo 'ACTIVE_RECORD,id,2,slot,1,gpu,00,va,0x1,pte_phys,0x2,original,0x3,temporary,0x4,destination,0x5,va_space_state,1'
  echo 'SCAN_DONE,ids_checked,10,active_found,2,kernel_active_count,2'; exit 0; fi
echo "PTE_RECOVER,id,7,outcome,RESTORED"
if [[ -e "$FAKE_TOOL_FAIL" ]]; then echo 'RECOVER_RESULT,FAILED,pte_failed,1'; exit 3; fi
echo 'RECOVER_RESULT,OK,RESTORED,1'; echo RECOVERED >> "$2"; exit 0
""")
tool.chmod(0o755)
fail_flag = tmp / "fail"; os.environ["FAKE_TOOL_FAIL"] = str(fail_flag)


class FakeManager:
    def __init__(self):
        self.worker_activity = "ACTIVE"; self.mailbox_unloads = 0
    def set_workers_active(self, active):
        self.worker_activity = "ACTIVE" if active else "PARKED"
        return {"status": "ok", "code": "WORKERS_ACTIVE" if active else "WORKERS_PARKED",
                "worker_activity": self.worker_activity}
    def unload_cpu_mailbox_manifest(self):
        self.mailbox_unloads += 1; return {"status": "ok", "code": "CPU_MAILBOX_UNLOADED"}


stop_event = threading.Event()
sock = tmp / "control.sock"
ctl = d.ControlServer(sock, state, stop_event)
ctl.manager = FakeManager(); ctl.peer_route_tool = tool
send = lambda cmd, **kw: d.send_control(sock, cmd, timeout=10, **kw)

journal = tmp / "routes.journal"; journal.write_text("OPEN\tpid\t1\nPTE\t7\tgpu\t1\t2\t3\t4\t0\n")
dead = subprocess.Popen(["true"]); dead.wait()                  # a pid that no longer exists

r = send("REGISTER_PEER_CLIENT", run_id="A", pid=dead.pid, journal=str(journal))
checks["register_ok"] = r["status"] == "ok" and send("STATUS")["peer_client"]["status"] == "REGISTERED"
checks["second_register_refused"] = send("REGISTER_PEER_CLIENT", run_id="B", pid=1, journal=str(journal))["code"] == "PEER_CLIENT_ALREADY_REGISTERED"
checks["stop_refused_while_registered"] = send("STOP")["code"] == "PEER_CLIENT_REGISTERED" and not stop_event.is_set()
checks["unregister_refused_without_close"] = send("UNREGISTER_PEER_CLIENT", run_id="A")["code"] == "PEER_CLIENT_NOT_CLOSED"
checks["recover_needs_quiesced"] = send("RECOVER_PEER_CLIENT")["code"] == "CLIENTS_NOT_QUIESCED"

# client still alive -> refused
live = subprocess.Popen(["sleep", "30"])
ctl.peer_client["pid"] = live.pid
checks["recover_refuses_live_client"] = send("RECOVER_PEER_CLIENT", clients_quiesced=True)["code"] == "PEER_CLIENT_STILL_RUNNING"
live.kill(); live.wait(); ctl.peer_client["pid"] = dead.pid

# tool failure -> RECOVERY_FAILED, STOP still guarded
fail_flag.write_text("1"); ctl.cpu_mailbox = {"status": "LOADED", "manifest": str(tmp / "none.manifest")}
r = send("RECOVER_PEER_CLIENT", clients_quiesced=True)
checks["tool_failure_reported"] = r["code"] == "ROUTE_RESTORE_FAILED" and ctl.peer_client["status"] == "RECOVERY_FAILED"
checks["workers_parked_before_restore"] = ctl.manager.worker_activity == "PARKED"
checks["mailbox_kept_on_failure"] = ctl.manager.mailbox_unloads == 0
checks["stop_still_refused_after_failure"] = send("STOP")["code"] == "PEER_CLIENT_REGISTERED"

# tool success -> recovered, mailbox unloaded, peer client released
fail_flag.unlink()
r = send("RECOVER_PEER_CLIENT", clients_quiesced=True)
st = send("STATUS")
checks["recover_ok"] = r["status"] == "ok" and r["code"] == "PEER_CLIENT_RECOVERED"
checks["mailbox_unloaded_by_recover"] = ctl.manager.mailbox_unloads == 1 and st["cpu_mailbox"]["status"] == "UNLOADED"
checks["peer_client_none_after_recover"] = st["peer_client"]["status"] == "NONE" and st["peer_client"]["last"]["status"] == "RECOVERED"
checks["recover_log_written"] = journal.with_suffix(".recover.log").exists()

# clean-close path: register, journal ends CLOSED, unregister accepted
j2 = tmp / "clean.journal"; j2.write_text("OPEN\tpid\t1\nPTE\t9\tg\t1\t2\t3\t4\t0\nPTE_RESTORED\t9\nCLOSED\n")
send("REGISTER_PEER_CLIENT", run_id="C", pid=os.getpid(), journal=str(j2))
checks["unregister_wrong_run_refused"] = send("UNREGISTER_PEER_CLIENT", run_id="X")["code"] == "PEER_CLIENT_NOT_REGISTERED"
checks["unregister_after_close_ok"] = send("UNREGISTER_PEER_CLIENT", run_id="C")["code"] == "PEER_CLIENT_UNREGISTERED"

lr = send("LIST_ROUTE_RECORDS", max_id=10)
checks["list_route_records"] = lr["active"] == 2 and lr["va_space_dead"] == 1

checks["stop_allowed_when_none"] = send("STOP")["status"] == "STOP_ACCEPTED" and stop_event.is_set()
ctl.close()
for k, v in checks.items(): print("CHECK", k, "OK" if v else "FAIL")
ok = all(checks.values())
print("DAEMON_PEER_RECOVERY_TEST," + ("PASS" if ok else "FAIL"))
sys.exit(0 if ok else 1)
