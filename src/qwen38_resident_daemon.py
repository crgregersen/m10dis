#!/usr/bin/env python3
"""Keep selected Qwen3.8 M10 expert layers and PLE allocations alive independently of vLLM.

This experimental daemon owns all CUDA worker processes and both local RPC
endpoints. vLLM clients may connect and disconnect without changing weight
ownership. STOP is explicit and drains the RPC server before releasing CUDA
workers. It does not install itself as a boot service.
"""
from __future__ import annotations

import argparse
import bisect
import fcntl
import hashlib
import json
import os
import signal
import socket
import socketserver
import struct
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def valid_expert_bundle(layer_root: Path, layer: int, model_index_sha256: str) -> bool:
    """Check that routed and shared artifacts are complete for this model/layer."""
    route_dir = layer_root / "routed"
    shared_dir = layer_root / "shared"
    route_path = route_dir / "resident_layer_manifest.json"
    shared_path = shared_dir / "shared_manifest.json"
    try:
        route = json.loads(route_path.read_text())
        shared = json.loads(shared_path.read_text())
        if (route.get("status") != "PACKED_LAYER_RESIDENT_EXPERTS" or
                route.get("layer") != layer or
                route.get("model_index_sha256") != model_index_sha256 or
                route.get("expert_count") != 512 or route.get("workers") != 20 or
                route.get("slice_width") != 32 or route.get("hidden") != 2560):
            return False
        if (shared.get("layer") != layer or shared.get("workers") != 4 or
                shared.get("slice_width") != 160 or shared.get("intermediate") != 640 or
                shared.get("format") != "exact BF16 checkpoint bit patterns stored as uint16 and expanded to FP32 for GM107 accumulation"):
            return False
        route_sizes = {
            "gate.u8": 512 * 32 * 2560,
            "up.u8": 512 * 32 * 2560,
            "down.u8": 512 * 32 * 2560,
            "gate_scale.f32": 512 * 32 * 20 * 4,
            "up_scale.f32": 512 * 32 * 20 * 4,
            "down_scale.f32": 512 * 2560 * 4,
        }
        for worker in range(20):
            worker_dir = route_dir / f"gpu_{worker:02d}"
            for name, size in route_sizes.items():
                path = worker_dir / name
                if not path.is_file() or path.stat().st_size != size:
                    return False
        shared_sizes = {
            "shared_gate_weight.bin": 2560 * 2,
            **{f"shared.{worker}.{projection}.bin": 160 * 2560 * 2
               for worker in range(4) for projection in ("gate", "up", "down")},
        }
        return all((shared_dir / name).is_file() and
                   (shared_dir / name).stat().st_size == size
                   for name, size in shared_sizes.items())
    except (OSError, ValueError, TypeError):
        return False


def validate_ple_owner_binary(path: Path) -> None:
    """Reject old lookup-only binaries before they allocate any GPU memory."""
    if not path.is_file():
        raise FileNotFoundError(f"PLE binary does not exist: {path}")
    image = path.read_bytes()
    required = (b"PLE_IPC_OWNER_HOLD", b"PLE_IPC_OWNER_READY",
                b"PLE_IPC_OWNER_STOP_ACK")
    missing = [marker.decode() for marker in required if marker not in image]
    if missing:
        raise RuntimeError(
            f"PLE binary lacks resident IPC-owner protocol markers {missing}; "
            "build it from the current src/cuda_load_stage9_ple.cu before startup"
        )


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def query_m10_memory() -> list[dict[str, Any]]:
    raw = subprocess.check_output([
        "/usr/bin/nvidia-smi", "--query-gpu=pci.bus_id,name,memory.free",
        "--format=csv,noheader,nounits",
    ], text=True, timeout=15)
    rows = []
    for line in raw.splitlines():
        fields = [field.strip() for field in line.split(",")]
        if len(fields) != 3 or fields[1] != "Tesla M10":
            continue
        rows.append({"pci_bdf": fields[0], "free_mib": int(fields[2])})
    if len(rows) != 24:
        raise RuntimeError(f"expected 24 GM107 functions on six M10s, found {len(rows)}")
    return rows


class _ControlHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        owner: "ControlServer" = self.server.owner  # type: ignore[attr-defined]
        try:
            request = json.loads(self.rfile.readline(65537))
            command = request.get("command")
            if command == "STATUS":
                response = owner.status()
            elif command == "STOP" and owner.peer_client_active() and request.get("force") is not True:
                response = {"status": "error", "code": "PEER_CLIENT_REGISTERED",
                            "message": "a peer client still owns route records; run RECOVER_PEER_CLIENT "
                                       "(or UNREGISTER_PEER_CLIENT after a clean close) before STOP",
                            "peer_client": owner.peer_client}
            elif command == "STOP":
                owner.stop_event.set()
                response = {"status": "STOP_ACCEPTED", "message": "draining before release"}
            elif command == "RELOAD_KERNELS":
                if request.get("clients_quiesced") is not True:
                    response = {"status": "ERROR", "code": "CLIENTS_NOT_QUIESCED",
                                "message": "stop direct-PEX/vLLM clients before swapping kernels"}
                elif owner.manager is None:
                    response = {"status": "ERROR", "code": "WORKERS_NOT_READY"}
                else:
                    routed = Path(request.get("routed_image", "")).resolve()
                    shared = Path(request.get("shared_image", "")).resolve()
                    response = owner.manager.reload_kernel_images(routed, shared)
                    if response.get("status") == "ok":
                        state = owner.status()
                        generation = max((v.get("generation", 0)
                                          for v in response["workers"].values()), default=0)
                        state["kernel_reload"] = {
                            "generation": generation,
                            "routed_image": str(routed),
                            "routed_sha256": sha256_file(routed),
                            "shared_image": str(shared),
                            "shared_sha256": sha256_file(shared),
                            "reloaded_unix": time.time(),
                        }
                        owner.kernel_reload = state["kernel_reload"]
                        atomic_json(owner.state_path, state)
            elif command in ("PARK_WORKERS", "WAKE_WORKERS"):
                if request.get("clients_quiesced") is not True:
                    response = {"status": "error", "code": "CLIENTS_NOT_QUIESCED",
                                "message": "finish all direct MoE/PLE requests before parking or waking workers"}
                elif owner.manager is None:
                    response = {"status": "error", "code": "WORKERS_NOT_READY"}
                else:
                    active = command == "WAKE_WORKERS"
                    response = owner.manager.set_workers_active(active)
                    if response.get("status") == "ok":
                        state = owner.status()
                        state["worker_activity"] = response["worker_activity"]
                        state["worker_activity_updated_unix"] = time.time()
                        atomic_json(owner.state_path, state)
            elif command == "LOAD_CPU_MAILBOX_MANIFEST":
                if request.get("clients_quiesced") is not True:
                    response = {"status": "error", "code": "CLIENTS_NOT_QUIESCED",
                                "message": "stop direct clients before registering or restarting worker kernels"}
                elif owner.manager is None:
                    response = {"status": "error", "code": "WORKERS_NOT_READY"}
                elif owner.status().get("direct_moe_setup", {}).get("status") != "PASS":
                    response = {"status": "error", "code": "DIRECT_MAPPINGS_NOT_READY",
                                "message": "create the retained GPU BAR1 result mappings before loading CPU mailboxes"}
                else:
                    response = owner.manager.load_cpu_mailbox_manifest(
                        Path(request.get("manifest_path", "")))
                    if response.get("status") == "ok":
                        owner.cpu_mailbox = {
                            "status": "LOADED", "run_id": response["run_id"],
                            "manifest": response["manifest"],
                            "device_slots": response.get("device_slots", []),
                            "loaded_unix": time.time(),
                        }
                        state = owner.status()
                        state["cpu_mailbox"] = owner.cpu_mailbox
                        atomic_json(owner.state_path, state)
            elif command == "UNLOAD_CPU_MAILBOX_MANIFEST":
                if request.get("clients_quiesced") is not True:
                    response = {"status": "error", "code": "CLIENTS_NOT_QUIESCED",
                                "message": "stop direct clients before unloading worker mailboxes"}
                elif owner.manager is None:
                    response = {"status": "error", "code": "WORKERS_NOT_READY"}
                elif owner.status().get("direct_moe_setup", {}).get("status") != "PASS":
                    response = {"status": "error", "code": "DIRECT_MAPPINGS_NOT_READY"}
                else:
                    response = owner.manager.unload_cpu_mailbox_manifest()
                    if response.get("status") == "ok":
                        owner.cpu_mailbox = {"status": "UNLOADED", "unloaded_unix": time.time()}
                        state = owner.status()
                        state["cpu_mailbox"] = owner.cpu_mailbox
                        atomic_json(owner.state_path, state)
            elif command == "REGISTER_PEER_CLIENT":
                response = owner.register_peer_client(request)
            elif command == "UNREGISTER_PEER_CLIENT":
                response = owner.unregister_peer_client(request)
            elif command == "RECOVER_PEER_CLIENT":
                response = owner.recover_peer_client(request)
            elif command == "LIST_ROUTE_RECORDS":
                response = owner.list_route_records(request)
            else:
                response = {"status": "ERROR", "message": "unknown control command"}
        except Exception as exc:
            response = {"status": "ERROR", "message": f"{type(exc).__name__}: {exc}"}
        self.wfile.write(json.dumps(response).encode() + b"\n")


class _ControlSocketServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


class ControlServer:
    def __init__(self, path: Path, state_path: Path, stop_event: threading.Event):
        self.path = path
        self.state_path = state_path
        self.stop_event = stop_event
        self.manager: Any | None = None
        self.kernel_reload: dict[str, Any] | None = None
        self.cpu_mailbox: dict[str, Any] = {"status": "UNLOADED"}
        # Peer-route ownership (see docs/RESIDENT_DAEMON_INTERFACE.md): at most one RTX peer
        # client at a time registers its route journal here before acquiring any route record.
        self.peer_client: dict[str, Any] = {"status": "NONE"}
        self.peer_route_tool: Path | None = None
        self.peer_lock = threading.Lock()
        if path.exists():
            path.unlink()
        self.server = _ControlSocketServer(str(path), _ControlHandler)
        self.server.owner = self  # type: ignore[attr-defined]
        os.chmod(path, 0o600)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    # ---- peer-route ownership -------------------------------------------------------------
    def peer_client_active(self) -> bool:
        return self.peer_client.get("status") in ("REGISTERED", "RECOVERY_FAILED")

    def _persist(self, key: str, value: Any) -> None:
        state = self.status(); state[key] = value; atomic_json(self.state_path, state)

    @staticmethod
    def _journal_tail(path: Path) -> list[str]:
        try:
            return [l.split("\t", 1)[0] for l in path.read_text(errors="replace").splitlines() if l.strip()]
        except OSError:
            return []

    def register_peer_client(self, request: dict[str, Any]) -> dict[str, Any]:
        with self.peer_lock:
            if self.peer_client_active():
                return {"status": "error", "code": "PEER_CLIENT_ALREADY_REGISTERED", "peer_client": self.peer_client}
            run_id, journal, pid = request.get("run_id"), request.get("journal"), request.get("pid")
            if not run_id or not journal or not isinstance(pid, int) or pid <= 0:
                return {"status": "error", "code": "INVALID_REQUEST", "message": "run_id, journal and pid are required"}
            self.peer_client = {"status": "REGISTERED", "run_id": str(run_id), "pid": pid,
                                "journal": str(Path(journal).resolve()), "registered_unix": time.time()}
            self._persist("peer_client", self.peer_client)
            return {"status": "ok", "code": "PEER_CLIENT_REGISTERED", "peer_client": self.peer_client}

    def unregister_peer_client(self, request: dict[str, Any]) -> dict[str, Any]:
        with self.peer_lock:
            pc = self.peer_client
            if pc.get("status") != "REGISTERED" or request.get("run_id") != pc.get("run_id"):
                return {"status": "error", "code": "PEER_CLIENT_NOT_REGISTERED", "peer_client": pc}
            tags = self._journal_tail(Path(pc["journal"]))
            if not tags or tags[-1] not in ("CLOSED", "RECOVERED"):
                return {"status": "error", "code": "PEER_CLIENT_NOT_CLOSED",
                        "message": "journal does not end in CLOSED; use RECOVER_PEER_CLIENT", "last": tags[-1:] }
            self.peer_client = {"status": "NONE", "last": {**pc, "status": "UNREGISTERED", "unregistered_unix": time.time()}}
            self._persist("peer_client", self.peer_client)
            return {"status": "ok", "code": "PEER_CLIENT_UNREGISTERED", "peer_client": self.peer_client}

    def recover_peer_client(self, request: dict[str, Any]) -> dict[str, Any]:
        """Release everything a dead/quiesced peer client left behind, keeping the daemon up:
        park workers -> restore journaled routes (q38_peer_route_tool) -> unload CPU mailbox."""
        with self.peer_lock:
            pc = self.peer_client
            if not self.peer_client_active():
                return {"status": "error", "code": "PEER_CLIENT_NOT_REGISTERED", "peer_client": pc}
            if request.get("clients_quiesced") is not True:
                return {"status": "error", "code": "CLIENTS_NOT_QUIESCED",
                        "message": "stop the peer client (no RTX work in flight) before recovery"}
            pid = int(pc["pid"])
            try:
                os.kill(pid, 0); alive = True
            except ProcessLookupError:
                alive = False
            except PermissionError:
                alive = True
            if alive and request.get("client_alive_ok") is not True:
                return {"status": "error", "code": "PEER_CLIENT_STILL_RUNNING", "pid": pid,
                        "message": "kill the client first, or pass client_alive_ok if it is known to be stopped"}
            if self.peer_route_tool is None or not self.peer_route_tool.is_file():
                return {"status": "error", "code": "PEER_ROUTE_TOOL_MISSING",
                        "message": "start the daemon with --peer-route-tool"}
            steps: dict[str, Any] = {}
            if self.manager is not None and self.manager.worker_activity != "PARKED":
                parked = self.manager.set_workers_active(False)
                steps["park_workers"] = parked.get("code")
                if parked.get("status") != "ok":
                    return {"status": "error", "code": "PARK_FAILED", "steps": steps, "detail": parked}
            else:
                steps["park_workers"] = "ALREADY_PARKED"
            run = subprocess.run([str(self.peer_route_tool), "restore", pc["journal"]],
                                 capture_output=True, text=True, timeout=300)
            lines = run.stdout.splitlines()
            log = Path(pc["journal"]).with_suffix(".recover.log")
            log.write_text(run.stdout + run.stderr)
            outcome = next((l for l in reversed(lines) if l.startswith("RECOVER_RESULT")), "RECOVER_RESULT,NONE")
            steps["route_restore"] = {"exit": run.returncode, "result": outcome, "log": str(log),
                                      "pte_outcomes": sum(1 for l in lines if l.startswith("PTE_RECOVER"))}
            if run.returncode != 0:
                self.peer_client = {**pc, "status": "RECOVERY_FAILED", "steps": steps, "failed_unix": time.time()}
                self._persist("peer_client", self.peer_client)
                return {"status": "error", "code": "ROUTE_RESTORE_FAILED", "steps": steps}
            if self.cpu_mailbox.get("status") == "LOADED" and self.manager is not None:
                unloaded = self.manager.unload_cpu_mailbox_manifest()
                steps["cpu_mailbox"] = unloaded.get("code")
                if unloaded.get("status") == "ok":
                    manifest = Path(self.cpu_mailbox.get("manifest", ""))
                    self.cpu_mailbox = {"status": "UNLOADED", "unloaded_unix": time.time(), "by": "RECOVER_PEER_CLIENT"}
                    self._persist("cpu_mailbox", self.cpu_mailbox)
                    try:
                        name = dict(l.split("\t", 1) for l in manifest.read_text().splitlines() if "\t" in l).get("shm_name")
                        if name and Path("/dev/shm" + name).exists():
                            Path("/dev/shm" + name).unlink(); steps["cpu_mailbox_shm"] = "UNLINKED"
                    except OSError as exc:
                        steps["cpu_mailbox_shm"] = f"not removed: {exc}"
            else:
                steps["cpu_mailbox"] = self.cpu_mailbox.get("status")
            self.peer_client = {"status": "NONE", "last": {**pc, "status": "RECOVERED", "steps": steps,
                                                            "recovered_unix": time.time()}}
            self._persist("peer_client", self.peer_client)
            return {"status": "ok", "code": "PEER_CLIENT_RECOVERED", "steps": steps}

    def list_route_records(self, request: dict[str, Any]) -> dict[str, Any]:
        if self.peer_route_tool is None or not self.peer_route_tool.is_file():
            return {"status": "error", "code": "PEER_ROUTE_TOOL_MISSING"}
        max_id = int(request.get("max_id", 4096))
        run = subprocess.run([str(self.peer_route_tool), "scan", str(max_id)], capture_output=True, text=True, timeout=120)
        recs = [l for l in run.stdout.splitlines() if l.startswith("ACTIVE_RECORD")]
        done = next((l for l in run.stdout.splitlines() if l.startswith("SCAN_DONE")), "")
        dead = sum(1 for l in recs if l.endswith(",va_space_state,2"))
        return {"status": "ok" if run.returncode == 0 else "error", "code": "ROUTE_RECORDS",
                "active": len(recs), "va_space_dead": dead, "summary": done, "records": recs[:600]}

    def status(self) -> dict[str, Any]:
        state = json.loads(self.state_path.read_text())
        # The service loop periodically rewrites the status file. Overlay the
        # manager's live lifecycle state so a PARK/WAKE response cannot be
        # overwritten by a stale snapshot from that loop.
        if self.manager is not None:
            state["worker_activity"] = self.manager.worker_activity
        return state

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=5)
        self.path.unlink(missing_ok=True)


def _recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < n:
        block = sock.recv(n - len(chunks))
        if not block:
            raise EOFError(f"socket closed after {len(chunks)}/{n} bytes")
        chunks.extend(block)
    return bytes(chunks)


class _PleHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        owner: "PleRpcServer" = self.server.owner  # type: ignore[attr-defined]
        while True:
            started = time.perf_counter()
            try:
                raw = _recv_exact(self.request, 4)
            except (EOFError, ConnectionError):
                return
            count = struct.unpack("<I", raw)[0]
            # A client stop marker closes only its session. The long-lived
            # daemon continues owning both CUDA allocations and workers.
            if count == 0xffffffff:
                return
            try:
                if count > 65536:
                    raise ValueError(f"PLE request too large: {count}")
                rows = struct.unpack("<" + "Q" * count, _recv_exact(self.request, 8 * count)) if count else ()
                entries = owner.resolve(rows)
                payload = owner.manager.ple_lookup(entries, count)
                if len(payload) != count * 160:
                    raise RuntimeError("resident PLE returned an invalid byte count")
                self.request.sendall(struct.pack("<II", 0, count) + payload)
                owner.record(count, time.perf_counter() - started, 0)
            except Exception as exc:
                try:
                    self.request.sendall(struct.pack("<II", 1, 0))
                except OSError:
                    pass
                owner.record(count, time.perf_counter() - started, 1)
                print(f"PLE_RPC_ERROR,{type(exc).__name__},{exc}", flush=True)


class _PleUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = False
    block_on_close = True
    allow_reuse_address = True


class PleRpcServer:
    """Resolve global PLE rows and dispatch their gathers to resident workers."""
    def __init__(self, path: Path, manager: Any, ipc_manifest: Path):
        self.path, self.manager = path, manager
        self._lock = threading.Lock()
        self.requests = self.rows = self.errors = 0
        self.elapsed_seconds = 0.0
        self.samples_ms: list[float] = []
        self.shards: list[dict[str, Any]] = []
        self.starts: list[int] = []
        with ipc_manifest.open() as f:
            for line in f:
                fields = line.rstrip("\n").split("\t")
                if not fields or fields[0] != "SHARD":
                    continue
                if fields[1] == "id":
                    continue
                _kind, sid, ordinal, bdf, nbytes, offset, nrows, width = fields
                self.shards.append({"id": int(sid), "bdf": bdf, "bytes": int(nbytes),
                    "offset": int(offset), "rows": int(nrows), "width": int(width)})
        self.shards.sort(key=lambda s: s["id"])
        for s in self.shards:
            if s["width"] != 160 or s["bytes"] != s["rows"] * s["width"]:
                raise RuntimeError(f"invalid PLE shard metadata: {s}")
            s["global_start"] = s["id"] * s["rows"]
            self.starts.append(s["global_start"])
        self.total_rows = 128 * 2_500_012
        if not self.shards:
            raise RuntimeError("PLE manifest contains no resident shards")
        self.device_map: dict[str, tuple[str, int]] = {}
        for branch, bdfs in manager._branch_bdfs.items():
            for local, bdf in enumerate(bdfs):
                self.device_map[self._canon(bdf)] = (branch, local)
        path.unlink(missing_ok=True)
        self.server = _PleUnixServer(str(path), _PleHandler)
        self.server.owner = self  # type: ignore[attr-defined]
        os.chmod(path, 0o600)
        self.thread = threading.Thread(target=self.server.serve_forever, daemon=True)
        self.thread.start()

    @staticmethod
    def _canon(bdf: str) -> str:
        return bdf.split(":", 1)[-1].lower()

    def resolve(self, rows: tuple[int, ...]) -> list[tuple[int, str, int, int]]:
        resolved = []
        for output_index, global_row in enumerate(rows):
            if global_row >= self.total_rows:
                raise ValueError(f"PLE row out of range: {global_row} >= {self.total_rows}")
            shard_index = bisect.bisect_right(self.starts, global_row) - 1
            if shard_index < 0:
                raise ValueError(f"PLE row is not resident: {global_row}")
            shard = self.shards[shard_index]
            local_row = global_row - shard["global_start"]
            if local_row >= shard["rows"]:
                raise ValueError(f"PLE shard for row {global_row} is not loaded")
            branch_device = self.device_map.get(self._canon(shard["bdf"]))
            if branch_device is None:
                raise RuntimeError(f"PLE shard GPU is not owned by a resident service: {shard['bdf']}")
            branch, device = branch_device
            offset = shard["offset"] + local_row * shard["width"]
            resolved.append((output_index, branch, device, offset))
        return resolved

    def record(self, count: int, elapsed: float, error: int) -> None:
        with self._lock:
            self.requests += 1
            self.rows += count if not error else 0
            self.errors += error
            self.elapsed_seconds += elapsed
            self.samples_ms.append(elapsed * 1000.0)
            if len(self.samples_ms) > 10000:
                del self.samples_ms[:-10000]
        print(f"PLE_LOOKUP_REQUEST,count,{count},status,{error}", flush=True)
        print(f"PLE_LOOKUP_TIMING,rows,{count},ms,{elapsed*1000:.6f}", flush=True)

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
        self.thread.join(timeout=10)
        self.path.unlink(missing_ok=True)


class ShutdownRequested(Exception):
    """Internal cancellation used to run the normal ordered cleanup path."""


def send_control(path: Path, command: str, timeout: float = 10.0,
                 **fields: Any) -> dict[str, Any]:
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(str(path))
        sock.sendall(json.dumps({"command": command, **fields}).encode() + b"\n")
        data = bytearray()
        while b"\n" not in data:
            block = sock.recv(4096)
            if not block:
                raise RuntimeError("control socket closed before replying")
            data.extend(block)
        return json.loads(bytes(data).split(b"\n", 1)[0])
    finally:
        sock.close()


def stop_ple(proc: subprocess.Popen[bytes] | None, path: Path | None,
             log_path: Path | None = None) -> int | None:
    if proc is None:
        return None
    if proc.poll() is None and proc.stdin is not None:
        proc.stdin.write(b"STOP\n")
        proc.stdin.flush()
        proc.stdin.close()
    elif proc.poll() is None and path is not None and path.exists():
        sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        sock.settimeout(5)
        try:
            sock.connect(str(path))
            sock.sendall(struct.pack("<I", 0xFFFFFFFF))
        finally:
            sock.close()
    try:
        rc = proc.wait(timeout=180)
    except subprocess.TimeoutExpired:
        # Do not kill a CUDA-owning process based only on a host deadline.
        rc = proc.wait()
    if log_path is not None:
        log = log_path.read_text(errors="replace")
        if "PLE_IPC_OWNER_STOP_ACK" not in log:
            raise RuntimeError("PLE allocation owner exited without its STOP acknowledgement")
    return rc


def bootstrap_direct_moe_maps(moe_path: Path, out: Path, manager: Any,
                              bootstrap_layer: int) -> dict[str, Any]:
    """Create the retained BAR1 result maps once, before any vLLM client.

    The request is deliberately sent to the resident owner's setup RPC. Normal
    vLLM MoE calls use the direct PEX/BAR1 adapter and never use this socket.
    Keeping this bootstrap in the long-lived owner makes the mapping manifest valid for
    later vLLM restarts as long as the worker processes remain alive.
    """
    from qwen38_moe_rpc import REQUEST_HEADER, REQUEST_MAGIC, RESPONSE_HEADER, RESPONSE_MAGIC

    count, hidden, top_k = 1, 2560, 10
    header = REQUEST_HEADER.pack(REQUEST_MAGIC, bootstrap_layer, count, hidden, top_k)
    payload = (struct.pack("<" + "f" * hidden, *([0.0] * hidden)) +
               struct.pack("<" + "i" * top_k, *range(top_k)) +
               struct.pack("<" + "f" * top_k, *([0.1] * top_k)))
    started = time.monotonic()
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(600)
    try:
        sock.connect(str(moe_path))
        sock.sendall(header + payload)
        response = _recv_exact(sock, RESPONSE_HEADER.size)
        magic, status, size = RESPONSE_HEADER.unpack(response)
        if magic != RESPONSE_MAGIC:
            raise RuntimeError(f"direct-map bootstrap returned invalid RPC magic {magic!r}")
        body = _recv_exact(sock, size)
        if status:
            raise RuntimeError("direct-map bootstrap failed: " + body.decode("utf-8", "replace"))
        if size != hidden * 4:
            raise RuntimeError(f"direct-map bootstrap returned {size} bytes, expected {hidden * 4}")
    finally:
        sock.close()

    manager._rlog.flush()
    manager._slog.flush()
    routed_log = out / "resident_expert_logs" / "manager_routed.log"
    shared_log = out / "resident_expert_logs" / "manager_shared.log"
    for path in (routed_log, shared_log):
        if not path.is_file():
            raise RuntimeError(f"missing resident worker log after bootstrap: {path}")
    routed_text = routed_log.read_text(errors="replace")
    shared_text = shared_log.read_text(errors="replace")
    trace_log = out / "rm_mapping_trace.log"
    trace_log.write_text(routed_text + shared_text)
    from qwen38_gpu_mappings_manifest import (
        extract_gpu_mappings, write_gpu_mappings_manifest,
        extract_gpu_pdb_records, write_gpu_pdb_manifest,
    )
    records = extract_gpu_mappings(routed_text, shared_text)
    mapping_manifest = out / "gpu_mappings.manifest"
    write_gpu_mappings_manifest(mapping_manifest, records,
                                run_id=f"{out.name}:{os.getpid()}")
    pdb_records = extract_gpu_pdb_records(routed_text, shared_text, mappings=records)
    pdb_manifest = out / "gpu_pdb.manifest"
    write_gpu_pdb_manifest(pdb_manifest, pdb_records,
                           run_id=f"{out.name}:{os.getpid()}")
    map_count = len(records)
    return {"status": "PASS", "bootstrap_layer": bootstrap_layer, "bootstrap_rows": 1,
            "bootstrap_seconds": time.monotonic() - started,
            "bootstrap_rpc_calls": 1, "gpu_map_records": map_count,
            "gpu_mappings_manifest": str(mapping_manifest),
            "gpu_pdb_manifest": str(pdb_manifest),
            "rm_mapping_trace_log": str(trace_log)}


def orderly_stop_expert_manager(manager: Any) -> None:
    """Stop CUDA workers without falling back to terminate/kill on timeout."""
    manager._pool.shutdown(wait=True)
    for branch, proc in (("routed", manager._r), ("shared", manager._s)):
        if proc.poll() is not None:
            raise RuntimeError(f"{branch} worker exited before orderly STOP: {proc.returncode}")
        try:
            response = manager._send(branch, "STOP")
        except Exception:
            if proc.poll() is None:
                proc.wait()
            raise
        if response.get("status") not in ("ok", "stopped"):
            if proc.poll() is None:
                proc.wait()
            raise RuntimeError(f"{branch} worker did not acknowledge STOP: {response}")
        proc.stdin.close()
        try:
            proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            # Preserve the process and its CUDA allocations until it exits.
            # A host timeout is not authority to terminate a possibly active
            # GPU worker or release its backing allocations.
            proc.wait()
    manager._rlog.close()
    manager._slog.close()


def serve(args: argparse.Namespace) -> int:
    runtime = args.runtime_dir.resolve()
    runtime.mkdir(parents=True, exist_ok=True)
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    lock_path = runtime / "resident.lock"
    lock_fd = os.open(lock_path, os.O_CREAT | os.O_RDWR, 0o600)
    try:
        fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        os.close(lock_fd)
        raise RuntimeError(f"another resident daemon owns {runtime}") from exc

    moe_path = runtime / "moe.sock"
    ple_path = runtime / "ple.sock"
    control_path = runtime / "control.sock"
    state_path = out / "resident_state.json"
    ple_log_path = out / "resident_ple.log"
    (out / "resident_expert_logs").mkdir(exist_ok=True)
    for stale in (moe_path, ple_path, control_path):
        stale.unlink(missing_ok=True)

    stop_event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda _signum, _frame: stop_event.set())

    state: dict[str, Any] = {
        "status": "STARTING", "pid": os.getpid(), "started_unix": time.time(),
        "runtime_dir": str(runtime), "moe_socket": str(moe_path),
        "ple_socket": str(ple_path), "control_socket": str(control_path),
        "model_index_sha256": sha256_file(args.model / "model.safetensors.index.json"),
        "layers_ready": 0,
        "loaded_layer_ids": [],
        "layer_start": args.layer_start,
        "layer_count": args.layer_count,
        "sandbox_mode": args.layer_start != 0 or args.layer_count not in (48, 49),
    }
    atomic_json(state_path, state)
    manager = None
    moe_server = None
    ple_server = None
    ple_proc = None
    ple_log = None
    control = None
    stream_readers: dict[str, Any] = {}
    error: str | None = None
    try:
        # Status and stop are available during the lengthy preload as well as
        # after READY. A stop request is honored between layer loads.
        control = ControlServer(control_path, state_path, stop_event)
        if args.peer_route_tool:
            control.peer_route_tool = args.peer_route_tool.resolve()
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from qwen38_resident_service_manager import ResidentServiceManager
        import qwen38_moe_rpc as moe_rpc_module

        # The standard test runner uses daemon handler threads for convenience.
        # The long-lived owner must join client handlers before releasing CUDA
        # allocations, so configure the server class before its construction.
        moe_rpc_module._ThreadingUnixServer.daemon_threads = False
        moe_rpc_module._ThreadingUnixServer.block_on_close = True
        MoeRpcServer = moe_rpc_module.MoeRpcServer

        ple_manifest = None
        if args.skip_ple:
            state["ple"] = {"status": "SKIPPED", "reason": "expert-kernel lifecycle test"}
            state["status"] = "LOADING_EXPERTS"
            atomic_json(state_path, state)
        else:
            # PLE owns its CUDA contexts and allocations in this long-lived process.
            ple_log = ple_log_path.open("wb")
            ple_env = os.environ.copy()
            ple_env.pop("CUDA_VISIBLE_DEVICES", None)
            ple_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
            ple_env["PLE_SKIP_EXPERT_RESERVE"] = "1"
            if args.ple_shard_ids:
                ple_env["PLE_SHARD_IDS"] = args.ple_shard_ids
            ple_ipc_dir = out / "ple_ipc"
            ple_manifest = ple_ipc_dir / "ple_ipc_manifest.tsv"
            ple_env["PLE_IPC_EXPORT_DIR"] = str(ple_ipc_dir)
            ple_env["PLE_IPC_OWNER_HOLD"] = "1"
            state["status"] = "LOADING_PLE"
            atomic_json(state_path, state)
            ple_proc = subprocess.Popen(
                [str(args.ple_binary.resolve()), str(args.allocations.resolve()),
                 str(args.shards.resolve()), str(args.startup_rows.resolve())],
                cwd=out, env=ple_env, stdin=subprocess.PIPE,
                stdout=ple_log, stderr=subprocess.STDOUT,
            )
            deadline = time.monotonic() + args.startup_timeout
            while time.monotonic() < deadline:
                if stop_event.is_set():
                    raise ShutdownRequested()
                if ple_proc.poll() is not None:
                    raise RuntimeError(f"PLE process exited before ready: {ple_proc.returncode}")
                if ple_manifest.is_file() and "PLE_IPC_OWNER_READY" in ple_log_path.read_text(errors="replace"):
                    break
                time.sleep(1)
            else:
                raise TimeoutError("PLE IPC owner did not become ready")
            if stop_event.is_set():
                raise ShutdownRequested()
            state["ple"] = {"status": "IPC_OWNER_READY", "pid": ple_proc.pid,
                            "binary_sha256": sha256_file(args.ple_binary),
                            "ipc_manifest": str(ple_manifest)}
            os.environ["Q38_PLE_IPC_MANIFEST"] = str(ple_manifest)
        state["status"] = "LOADING_EXPERTS"
        atomic_json(state_path, state)

        manager = ResidentServiceManager(
            args.routed_service.resolve(), args.shared_service.resolve(), out / "resident_expert_logs",
            num_layers=49,   # 48 model layers + the MTP layer (layer 48); unloaded layers stay NOT_LOADED
        )
        control.manager = manager
        stream_pack_state: list[dict[str, Any]] = []
        stream_qwen = None
        stream_weight_map = None
        if args.stream_pack:
            # Rebuild one host staging bundle at a time. The workers retain
            # their device allocations after LOAD, so the staging can be
            # removed before packing the next layer.
            import qwen38_m10_expert_sim as stream_qwen
            import numpy as np
            stream_qwen.MODEL = args.model.resolve()
            index_path = args.model.resolve() / "model.safetensors.index.json"
            stream_weight_map = json.loads(index_path.read_text())["weight_map"]
            state["stream_pack"] = {"enabled": True, "status": "READY",
                                     "max_staged_layers": 1, "layers": stream_pack_state}
            atomic_json(state_path, state)
        load_started = time.monotonic()
        selected_layers = list(range(args.layer_start,
                                     args.layer_start + args.layer_count))
        for layer in selected_layers:
            if stop_event.is_set():
                raise ShutdownRequested()
            local_layer_root = args.bundle_root.resolve() / f"layer_{layer:02d}"
            persistent_layer_root = (args.persistent_bundle_root.resolve() / f"layer_{layer:02d}"
                                     if args.persistent_bundle_root else None)
            layer_root = local_layer_root
            pack_started = time.monotonic()
            generated_here = False
            bundle_source = "run-local"
            if args.stream_pack:
                candidates = []
                if persistent_layer_root is not None:
                    candidates.append((persistent_layer_root, "persistent-cache"))
                candidates.append((local_layer_root, "run-local-cache"))
                selected_bundle = next((item for item in candidates
                                        if valid_expert_bundle(item[0], layer,
                                                               state["model_index_sha256"])), None)
                if selected_bundle is not None:
                    layer_root, bundle_source = selected_bundle
                else:
                    layer_root = local_layer_root
                    bundle_source = "generated-temporary"
                    route_dir = layer_root / "routed"
                    shared_dir = layer_root / "shared"
                    if route_dir.exists() or shared_dir.exists():
                        raise RuntimeError(
                            f"incomplete or mismatched run-local bundle for layer {layer}: {layer_root}; "
                            "refusing to overwrite it")
                    route_dir.mkdir(parents=True)
                    shared_dir.mkdir(parents=True)
                    generated_here = True
                    pack_log = out / "stream_pack_logs" / f"layer_{layer:02d}.log"
                    pack_log.parent.mkdir(parents=True, exist_ok=True)
                    try:
                        with pack_log.open("wb") as log:
                            subprocess.run([
                                sys.executable,
                                str(Path(__file__).with_name("prepare_resident_expert_layer.py")),
                                "--model", str(args.model.resolve()),
                                "--layer", str(layer), "--out", str(route_dir),
                            ], stdout=log, stderr=subprocess.STDOUT, check=True, timeout=600)
                        # A zero activation is only used to invoke the existing
                        # BF16 shared-expert packer; it does not affect inference.
                        stream_qwen.prepare_shared(
                            layer, np.zeros(stream_qwen.HIDDEN, dtype=np.float32),
                            stream_weight_map, stream_readers, shared_dir)
                        if not valid_expert_bundle(layer_root, layer,
                                                   state["model_index_sha256"]):
                            raise RuntimeError(f"generated routed/shared bundle validation failed for layer {layer}")
                    except Exception:
                        # Preserve a failed/incomplete staging directory for
                        # diagnosis; never delete data before a successful LOAD.
                        raise
            response = manager.load_layer_async(
                layer, layer_root / "routed", layer_root / "shared",
            ).result(timeout=args.layer_timeout)
            if response.get("state") != "READY":
                raise RuntimeError(f"layer {layer} load failed: {response}")
            if args.stream_pack and generated_here:
                route_bytes = sum(p.stat().st_size for p in (layer_root / "routed").rglob("*") if p.is_file())
                shared_bytes = sum(p.stat().st_size for p in (layer_root / "shared").rglob("*") if p.is_file())
                import shutil
                shutil.rmtree(layer_root / "routed")
                shutil.rmtree(layer_root / "shared")
            else:
                route_bytes = sum(p.stat().st_size for p in (layer_root / "routed").rglob("*") if p.is_file())
                shared_bytes = sum(p.stat().st_size for p in (layer_root / "shared").rglob("*") if p.is_file())
            if args.stream_pack:
                stream_pack_state.append({"layer": layer, "status": "LOADED",
                                         "bundle_source": bundle_source,
                                         "routed_bytes": route_bytes,
                                         "shared_bytes": shared_bytes,
                                         "elapsed_seconds": time.monotonic() - pack_started})
                state["stream_pack"] = {"enabled": True, "status": "RUNNING",
                                         "max_staged_layers": 1,
                                         "persistent_bundle_root": str(args.persistent_bundle_root) if args.persistent_bundle_root else None,
                                         "layers": stream_pack_state}
            state["loaded_layer_ids"].append(layer)
            state["layers_ready"] = len(state["loaded_layer_ids"])
            state["last_layer"] = layer
            state["expert_load_seconds"] = time.monotonic() - load_started
            atomic_json(state_path, state)

        for reader in stream_readers.values():
            reader.close()
        stream_readers.clear()

        if stop_event.is_set():
            raise ShutdownRequested()
        if ple_manifest is not None:
            ple_server = PleRpcServer(ple_path, manager, ple_manifest)
        memory = query_m10_memory()
        state["m10_devices"] = memory
        state["m10_min_free_mib"] = min(row["free_mib"] for row in memory)
        if state["m10_min_free_mib"] < args.minimum_free_mib:
            raise RuntimeError(f"M10 headroom below limit: {state['m10_min_free_mib']} MiB")
        moe_server = MoeRpcServer(moe_path, manager, out / "moe_request_scratch")
        # server_close joins non-daemon handlers; if a client remains
        # connected, shutdown waits here with allocations retained until that
        # client disconnects.
        os.chmod(moe_path, 0o600)
        direct_setup = None
        if args.direct_library:
            # Create BAR1 result mappings while the resident worker owner is
            # alive. The resulting addresses and mappings remain owned by
            # these workers and are reused by direct-Pex vLLM clients after a
            # client process exits or restarts.
            direct_setup = bootstrap_direct_moe_maps(
                moe_path, out, manager, selected_layers[0])
            direct_setup.update({
                "library": str(args.direct_library.resolve()),
                "groups_tsv": str(args.direct_groups.resolve()),
            })
        state.update({
            "status": "READY", "ready_unix": time.time(),
            "worker_activity": manager.worker_activity,
            "ple_lookup_mode": ("disabled (expert-kernel lifecycle test)" if args.skip_ple
                                 else "resident-kernel CUDA IPC gather"),
            "expert_load_seconds": time.monotonic() - load_started,
            "routed_service_pid": manager._r.pid,
            "shared_service_pid": manager._s.pid,
            "multicast_snapshot_root": manager.multicast_snapshot_root,
            "routed_binary_sha256": sha256_file(args.routed_service),
            "shared_binary_sha256": sha256_file(args.shared_service),
        })
        if direct_setup is not None:
            state["direct_moe_setup"] = direct_setup
        atomic_json(state_path, state)
        print(json.dumps({"status": "READY", "state": str(state_path),
                          "moe_socket": str(moe_path), "ple_socket": str(ple_path),
                          "control_socket": str(control_path)}), flush=True)

        while not stop_event.wait(2):
            state["worker_activity"] = manager.worker_activity
            state["moe_requests"] = moe_server.calls
            state["moe_rows"] = moe_server.rows
            state["moe_timing"] = moe_server.timing_snapshot()
            if ple_server is not None:
                state["ple_requests"] = ple_server.requests
                state["ple_rows"] = ple_server.rows
                state["ple_errors"] = ple_server.errors
            if ple_server is not None and ple_server.samples_ms:
                ordered_ms = sorted(ple_server.samples_ms)
                state["ple_timing"] = {
                    "requests": len(ordered_ms), "total_ms": sum(ordered_ms),
                    "mean_ms": sum(ordered_ms) / len(ordered_ms),
                    "median_ms": ordered_ms[(len(ordered_ms) - 1) // 2],
                    "p95_ms": ordered_ms[max(0, int(0.95 * len(ordered_ms)) - 1)],
                    "max_ms": ordered_ms[-1],
                }
            if control.kernel_reload is not None:
                state["kernel_reload"] = control.kernel_reload
            state["cpu_mailbox"] = control.cpu_mailbox
            state["peer_client"] = control.peer_client
            state["last_status_update_unix"] = time.time()
            atomic_json(state_path, state)
            if (manager._r.poll() is not None or manager._s.poll() is not None or
                    (ple_proc is not None and ple_proc.poll() is not None)):
                state["status"] = "DEGRADED"
                state["worker_exit_codes"] = {
                    "routed": manager._r.poll(), "shared": manager._s.poll(),
                }
                if ple_proc is not None:
                    state["worker_exit_codes"]["ple"] = ple_proc.poll()
                atomic_json(state_path, state)
                stop_event.wait()

    except ShutdownRequested:
        state["status"] = "DRAINING"
        state["stop_reason"] = "operator request during preload"
        atomic_json(state_path, state)
    except BaseException as exc:
        error = f"{type(exc).__name__}: {exc}"
        state["status"] = "ERROR"
        state["error"] = error
        atomic_json(state_path, state)
        raise
    finally:
        state["status"] = "DRAINING" if error is None else "ERROR_CLEANUP"
        if error:
            state["error"] = error
        atomic_json(state_path, state)
        for reader in stream_readers.values():
            try:
                reader.close()
            except Exception as exc:
                state.setdefault("cleanup_errors", []).append(f"stream-pack shard reader close: {exc}")
        stream_readers.clear()
        # Stop accepting MoE requests and wait for its serving thread before
        # sending STOP to CUDA workers. Their allocations remain owned until
        # manager.close() completes.
        if moe_server is not None:
            moe_server.close()
        if ple_server is not None:
            # Join active PLE lookups before stopping their persistent kernels.
            ple_server.close()
        if manager is not None:
            try:
                orderly_stop_expert_manager(manager)
            except Exception as exc:
                state.setdefault("cleanup_errors", []).append(f"expert stop: {exc}")
        ple_rc = None
        try:
            # Imported CUDA IPC handles must be closed by every worker before
            # the allocation owner is allowed to free the PLE tables.
            ple_rc = stop_ple(ple_proc, None, ple_log_path)
            if ple_rc not in (None, 0):
                state.setdefault("cleanup_errors", []).append(f"PLE exited with status {ple_rc}")
        except Exception as exc:
            state.setdefault("cleanup_errors", []).append(f"PLE stop: {exc}")
            if ple_proc is not None and ple_proc.poll() is None:
                # Keep the process and allocations alive if shutdown could not
                # be confirmed through the documented STOP operation.
                ple_rc = ple_proc.wait()
        if ple_log is not None:
            ple_log.close()
        if control is not None:
            control.close()
        state["ple_return_code"] = ple_rc
        state["stopped_unix"] = time.time()
        state["status"] = "STOPPED" if error is None and not state.get("cleanup_errors") else "ERROR_CLEANUP"
        atomic_json(state_path, state)
        os.close(lock_fd)
    return 0 if error is None and not state.get("cleanup_errors") else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="mode", required=True)
    serve_parser = sub.add_parser("serve", help="load once and retain M10 allocations")
    serve_parser.add_argument("--model", type=Path, required=True)
    serve_parser.add_argument("--bundle-root", type=Path, required=True)
    serve_parser.add_argument("--persistent-bundle-root", type=Path,
                              default=Path(os.environ.get("M10DIS_DATA", "/mnt/nvme")) / "qwen38-stage9-preserved-bundles",
                              help="prefer complete matching bundles here before run-local staging")
    serve_parser.add_argument("--routed-service", type=Path, required=True)
    serve_parser.add_argument("--shared-service", type=Path, required=True)
    serve_parser.add_argument("--ple-binary", type=Path, required=True)
    serve_parser.add_argument("--allocations", type=Path, required=True)
    serve_parser.add_argument("--shards", type=Path, required=True)
    serve_parser.add_argument("--startup-rows", type=Path, required=True)
    serve_parser.add_argument("--direct-library", type=Path,
                              help="native PEX/BAR1 client library; enables one-time mapping bootstrap")
    serve_parser.add_argument("--direct-groups", type=Path,
                              help="fresh topology TSV matching --direct-library")
    serve_parser.add_argument("--runtime-dir", type=Path, required=True)
    serve_parser.add_argument("--out", type=Path, required=True)
    serve_parser.add_argument("--layer-start", type=int, default=0,
                              help="first transformer layer to preload (0..47)")
    serve_parser.add_argument("--layer-count", "--num-layers", dest="layer_count", type=int, default=48,
                              help="number of consecutive transformer layers to preload")
    serve_parser.add_argument("--minimum-free-mib", type=int, default=500)
    serve_parser.add_argument("--startup-timeout", type=int, default=600)
    serve_parser.add_argument("--layer-timeout", type=int, default=600)
    serve_parser.add_argument("--skip-ple", action="store_true",
                              help="skip PLE allocation owner for isolated expert-kernel lifecycle tests")
    serve_parser.add_argument("--stream-pack", action="store_true",
                              help="reuse matching persistent bundles; otherwise pack one layer, load it, and remove only generated temporary staging")
    serve_parser.add_argument("--ple-shard-ids", default="",
                              help="load only these comma-separated PLE shard IDs (for bounded tests)")
    serve_parser.add_argument("--peer-route-tool", type=Path,
                              help="q38_peer_route_tool binary used by RECOVER_PEER_CLIENT/LIST_ROUTE_RECORDS")
    status_parser = sub.add_parser("status", help="query a resident service")
    status_parser.add_argument("--control-socket", type=Path, required=True)
    stop_parser = sub.add_parser("stop", help="drain and stop a resident service")
    stop_parser.add_argument("--control-socket", type=Path, required=True)
    stop_parser.add_argument("--force", action="store_true",
                             help="stop even if a peer client is registered (its M10 route records become unrecoverable)")
    recover_parser = sub.add_parser("recover-peer-client",
                                    help="park workers, restore a dead peer client's journaled routes, unload its mailbox")
    recover_parser.add_argument("--control-socket", type=Path, required=True)
    recover_parser.add_argument("--clients-quiesced", action="store_true")
    recover_parser.add_argument("--client-alive-ok", action="store_true",
                                help="allow recovery although the registered client pid still exists")
    list_parser = sub.add_parser("list-route-records", help="read-only scan of active patched-UVM route records")
    list_parser.add_argument("--control-socket", type=Path, required=True)
    list_parser.add_argument("--max-id", type=int, default=4096)
    reload_parser = sub.add_parser("reload-kernels", help="swap SM50 cubins and restart resident grids")
    reload_parser.add_argument("--control-socket", type=Path, required=True)
    reload_parser.add_argument("--routed-image", type=Path, required=True)
    reload_parser.add_argument("--shared-image", type=Path, required=True)
    reload_parser.add_argument("--clients-quiesced", action="store_true",
                               help="assert that no external direct-PEX request is in flight")
    for command, help_text in (("park-workers", "stop polling kernels while retaining weights"),
                               ("wake-workers", "relaunch polling kernels against retained weights")):
        lifecycle_parser = sub.add_parser(command, help=help_text)
        lifecycle_parser.add_argument("--control-socket", type=Path, required=True)
        lifecycle_parser.add_argument("--clients-quiesced", action="store_true",
                                      help="assert that no external direct-PEX request is in flight")
    mailbox_parser = sub.add_parser("load-cpu-mailbox", help="register CPU shared-memory result mailboxes in workers")
    mailbox_parser.add_argument("--control-socket", type=Path, required=True)
    mailbox_parser.add_argument("--manifest", type=Path, required=True)
    mailbox_parser.add_argument("--clients-quiesced", action="store_true",
                                help="assert that no external direct-PEX request is in flight")
    unmailbox_parser = sub.add_parser("unload-cpu-mailbox", help="unregister CPU shared-memory result mailboxes")
    unmailbox_parser.add_argument("--control-socket", type=Path, required=True)
    unmailbox_parser.add_argument("--clients-quiesced", action="store_true",
                                  help="assert that no external direct-PEX request is in flight")
    args = parser.parse_args()
    if args.mode == "serve":
        # 48 model layers + the MTP layer as layer 48 (2026-11-04: --layer-count 49 loads it too)
        if args.layer_start < 0 or args.layer_start >= 49:
            parser.error("--layer-start must be in 0..48")
        if args.layer_count < 1 or args.layer_count > 49:
            parser.error("--layer-count must be in 1..49")
        if args.layer_start + args.layer_count > 49:
            parser.error("--layer-start + --layer-count must not exceed 49 (48 layers + MTP)")
        if bool(args.direct_library) != bool(args.direct_groups):
            parser.error("--direct-library and --direct-groups must be supplied together")
        if not args.skip_ple:
            validate_ple_owner_binary(args.ple_binary)
        return serve(args)
    if args.mode == "status":
        request = {"command": "STATUS"}
    elif args.mode == "reload-kernels":
        request = {"command": "RELOAD_KERNELS",
                   "routed_image": str(args.routed_image.resolve()),
                   "shared_image": str(args.shared_image.resolve()),
                   "clients_quiesced": args.clients_quiesced}
    elif args.mode == "park-workers":
        request = {"command": "PARK_WORKERS", "clients_quiesced": args.clients_quiesced}
    elif args.mode == "wake-workers":
        request = {"command": "WAKE_WORKERS", "clients_quiesced": args.clients_quiesced}
    elif args.mode == "load-cpu-mailbox":
        request = {"command": "LOAD_CPU_MAILBOX_MANIFEST",
                   "manifest_path": str(args.manifest.resolve()),
                   "clients_quiesced": args.clients_quiesced}
    elif args.mode == "unload-cpu-mailbox":
        request = {"command": "UNLOAD_CPU_MAILBOX_MANIFEST",
                   "clients_quiesced": args.clients_quiesced}
    elif args.mode == "recover-peer-client":
        request = {"command": "RECOVER_PEER_CLIENT", "clients_quiesced": args.clients_quiesced,
                   "client_alive_ok": args.client_alive_ok}
    elif args.mode == "list-route-records":
        request = {"command": "LIST_ROUTE_RECORDS", "max_id": args.max_id}
    else:
        request = {"command": "STOP", "force": bool(getattr(args, "force", False))}
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(600)
    try:
        sock.connect(str(args.control_socket))
        sock.sendall(json.dumps(request).encode() + b"\n")
        data = bytearray()
        while b"\n" not in data:
            block = sock.recv(4096)
            if not block:
                raise RuntimeError("control socket closed before replying")
            data.extend(block)
        response = json.loads(bytes(data).split(b"\n", 1)[0])
    finally:
        sock.close()
    print(json.dumps(response, indent=2))
    # New commands report failure through the exit status; older commands keep exit 0 for
    # compatibility with existing runners (they parse the JSON).
    if args.mode in ("recover-peer-client", "list-route-records", "stop") and \
            str(response.get("status", "")).lower() not in ("ok", "stop_accepted"):
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
