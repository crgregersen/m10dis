"""Local framed RPC bridge from a spawned vLLM worker to M10 services."""
from __future__ import annotations

import atexit
import json
import os
import socket
import socketserver
import struct
import tempfile
import threading
import time
import uuid
from pathlib import Path
from typing import Any

import numpy as np
import torch

REQUEST_MAGIC = b"Q38MOE1\0"
RESPONSE_MAGIC = b"Q38MOR1\0"
REQUEST_HEADER = struct.Struct("<8sIIII")
RESPONSE_HEADER = struct.Struct("<8sII")
HIDDEN_SIZE = 2560
TOP_K = 10


def _recv_exact(sock: socket.socket, size: int) -> bytes:
    data = bytearray()
    while len(data) < size:
        chunk = sock.recv(size - len(data))
        if not chunk:
            raise EOFError(f"socket closed after {len(data)}/{size} bytes")
        data.extend(chunk)
    return bytes(data)


class MoeSocketClient:
    """vLLM-worker adapter. The M10 manager remains in the parent process."""

    def __init__(self, path: Path | str):
        self.path = str(path)
        self._sock: socket.socket | None = None
        self._lock = threading.Lock()
        self._timing_lock = threading.Lock()
        self._timing = {
            "calls": 0, "rows": 0, "device_to_host_seconds": 0.0,
            "validation_serialize_seconds": 0.0, "socket_roundtrip_seconds": 0.0,
            "result_restore_seconds": 0.0,
        }
        self._timing_path = os.environ.get("Q38_MOE_TIMING_LOG")
        self._timing_emitted = False
        atexit.register(self._flush_timing)
        self.calls = 0
        self.rows = 0

    def _connect(self) -> socket.socket:
        if self._sock is None:
            sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            sock.settimeout(600)
            sock.connect(self.path)
            self._sock = sock
        return self._sock

    @torch.inference_mode()
    def forward(self, layer_id: int, hidden: torch.Tensor,
                expert_ids: torch.Tensor, route_weights: torch.Tensor) -> torch.Tensor:
        started = time.perf_counter()
        if hidden.ndim != 2 or hidden.shape[1] != HIDDEN_SIZE:
            raise ValueError(f"hidden must have shape [N,{HIDDEN_SIZE}], got {tuple(hidden.shape)}")
        rows = int(hidden.shape[0])
        if expert_ids.shape != (rows, TOP_K) or route_weights.shape != (rows, TOP_K):
            raise ValueError("expert_ids and route_weights must both have shape [N,10]")
        if hidden.device.type != "cuda":
            raise ValueError("vLLM adapter input must be on the CUDA device")
        copy_started = time.perf_counter()
        h = hidden.detach().to(torch.float32).cpu().contiguous().numpy().astype("<f4", copy=False)
        ids = expert_ids.detach().to(torch.int32).cpu().contiguous().numpy().astype("<i4", copy=False)
        weights = route_weights.detach().to(torch.float32).cpu().contiguous().numpy().astype("<f4", copy=False)
        device_to_host_s = time.perf_counter() - copy_started
        if not np.isfinite(h).all() or not np.isfinite(weights).all():
            raise ValueError("MoE request contains non-finite values")
        if np.any(ids < 0) or np.any(ids >= 512):
            raise ValueError("MoE request has out-of-range expert IDs")
        if rows and (not np.allclose(weights.sum(axis=1), 1.0, atol=1e-5) or np.any(weights < 0)):
            raise ValueError("route weights must be nonnegative and sum to one per row")
        body = h.tobytes(order="C") + ids.tobytes(order="C") + weights.tobytes(order="C")
        validation_serialize_s = time.perf_counter() - copy_started - device_to_host_s
        socket_started = time.perf_counter()
        with self._lock:
            sock = self._connect()
            sock.sendall(REQUEST_HEADER.pack(REQUEST_MAGIC, int(layer_id), rows, HIDDEN_SIZE, TOP_K))
            sock.sendall(body)
            magic, status, payload_size = RESPONSE_HEADER.unpack(_recv_exact(sock, RESPONSE_HEADER.size))
            if magic != RESPONSE_MAGIC:
                raise RuntimeError(f"bad MoE RPC response magic {magic!r}")
            payload = _recv_exact(sock, payload_size)
            if status:
                raise RuntimeError("M10 MoE RPC failed: " + payload.decode("utf-8", "replace"))
        socket_roundtrip_s = time.perf_counter() - socket_started
        expected = rows * HIDDEN_SIZE * 4
        if payload_size != expected:
            raise RuntimeError(f"M10 RPC result size {payload_size}, expected {expected}")
        output = np.frombuffer(payload, dtype="<f4").reshape(rows, HIDDEN_SIZE).copy()
        restore_started = time.perf_counter()
        result = torch.from_numpy(output).to(device=hidden.device, dtype=hidden.dtype)
        result_restore_s = time.perf_counter() - restore_started
        self.calls += 1
        self.rows += rows
        with self._timing_lock:
            self._timing["calls"] += 1
            self._timing["rows"] += rows
            self._timing["device_to_host_seconds"] += device_to_host_s
            self._timing["validation_serialize_seconds"] += validation_serialize_s
            self._timing["socket_roundtrip_seconds"] += socket_roundtrip_s
            self._timing["result_restore_seconds"] += result_restore_s
            self._timing["forward_total_seconds"] = self._timing.get("forward_total_seconds", 0.0) + time.perf_counter() - started
        return result

    def timing_snapshot(self) -> dict[str, Any]:
        with self._timing_lock:
            result = dict(self._timing)
        calls = result.get("calls", 0)
        if calls:
            for key in ("device_to_host_seconds", "validation_serialize_seconds",
                        "socket_roundtrip_seconds", "result_restore_seconds",
                        "forward_total_seconds"):
                result[key.replace("_seconds", "_mean_seconds")] = result[key] / calls
        return result

    def _flush_timing(self) -> None:
        if not self._timing_path or self._timing_emitted:
            return
        self._timing_emitted = True
        record = {"pid": os.getpid(), **self.timing_snapshot()}
        try:
            fd = os.open(self._timing_path, os.O_WRONLY | os.O_CREAT | os.O_APPEND, 0o600)
            try:
                os.write(fd, (json.dumps(record, separators=(",", ":")) + "\n").encode())
            finally:
                os.close(fd)
        except OSError:
            pass

    def close(self) -> None:
        if self._sock is not None:
            self._sock.close()
            self._sock = None
        self._flush_timing()


class _MoeRequestHandler(socketserver.BaseRequestHandler):
    def handle(self) -> None:
        server: MoeRpcServer = self.server.owner  # type: ignore[attr-defined]
        while True:
            request_started = time.perf_counter()
            try:
                header = _recv_exact(self.request, REQUEST_HEADER.size)
            except (EOFError, ConnectionError):
                return
            try:
                magic, layer, rows, hidden_size, top_k = REQUEST_HEADER.unpack(header)
                if magic != REQUEST_MAGIC or hidden_size != HIDDEN_SIZE or top_k != TOP_K:
                    raise ValueError("invalid request header")
                if not 0 < rows <= 4096 or not 0 <= layer < 48:
                    raise ValueError(f"invalid layer/row count: layer={layer}, rows={rows}")
                h_bytes = rows * HIDDEN_SIZE * 4
                id_bytes = rows * TOP_K * 4
                wt_bytes = rows * TOP_K * 4
                payload = _recv_exact(self.request, h_bytes + id_bytes + wt_bytes)
                receive_s = time.perf_counter() - request_started
                h = np.frombuffer(payload, dtype="<f4", count=rows * HIDDEN_SIZE).reshape(rows, HIDDEN_SIZE)
                off = h_bytes
                ids = np.frombuffer(payload, dtype="<i4", count=rows * TOP_K, offset=off).reshape(rows, TOP_K)
                off += id_bytes
                weights = np.frombuffer(payload, dtype="<f4", count=rows * TOP_K, offset=off).reshape(rows, TOP_K)
                result = server.infer(layer, h, ids, weights)
                send_started = time.perf_counter()
                self.request.sendall(RESPONSE_HEADER.pack(RESPONSE_MAGIC, 0, len(result)) + result)
                send_s = time.perf_counter() - send_started
                with server.lock:
                    server.calls += 1
                    server.rows += rows
                    server.timing["socket_receive_seconds"] += receive_s
                    server.timing["socket_send_seconds"] += send_s
            except Exception as exc:
                message = f"{type(exc).__name__}: {exc}".encode("utf-8", "replace")
                try:
                    self.request.sendall(RESPONSE_HEADER.pack(RESPONSE_MAGIC, 1, len(message)) + message)
                except OSError:
                    return


class _ThreadingUnixServer(socketserver.ThreadingMixIn, socketserver.UnixStreamServer):
    daemon_threads = True
    allow_reuse_address = True


class MoeRpcServer:
    def __init__(self, path: Path, manager: Any, scratch: Path):
        self.path = path
        self.manager = manager
        self.scratch = scratch.resolve()
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self.calls = 0
        self.rows = 0
        self.timing = {
            "socket_receive_seconds": 0.0, "socket_send_seconds": 0.0,
            "infer_total_seconds": 0.0, "request_file_write_seconds": 0.0,
            "tempdir_overhead_seconds": 0.0,
            "manager_pair_seconds": 0.0, "routed_service_seconds": 0.0,
            "shared_service_seconds": 0.0, "result_read_join_seconds": 0.0,
        }
        self.timing_samples: list[dict[str, float]] = []
        try:
            path.unlink()
        except FileNotFoundError:
            pass
        self._server = _ThreadingUnixServer(str(path), _MoeRequestHandler)
        self._server.owner = self  # type: ignore[attr-defined]
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)
        self._thread.start()

    def infer(self, layer: int, hidden: np.ndarray, ids: np.ndarray,
              weights: np.ndarray) -> bytes:
        infer_started = time.perf_counter()
        outputs: list[np.ndarray] = []
        file_write_s = manager_pair_s = routed_service_s = shared_service_s = read_join_s = 0.0
        tempdir_overhead_s = 0.0
        timing_samples: list[dict[str, float]] = []
        for row in range(len(hidden)):
            if not np.isfinite(hidden[row]).all() or not np.isfinite(weights[row]).all():
                raise ValueError(f"row {row} contains non-finite inputs")
            if np.any(ids[row] < 0) or np.any(ids[row] >= 512):
                raise ValueError(f"row {row} contains invalid expert IDs")
            if len(np.unique(ids[row])) != TOP_K:
                raise ValueError(f"row {row} has duplicate expert IDs")
            if np.any(weights[row] < 0) or not np.isclose(weights[row].sum(), 1.0, atol=1e-5):
                raise ValueError(f"row {row} route weights do not sum to one")
            request_id = f"vllm-rpc-{uuid.uuid4().hex}-{row}"
            row_started = time.perf_counter()
            row_file_write_s = row_manager_pair_s = row_read_join_s = 0.0
            row_routed_service_s = row_shared_service_s = 0.0
            with tempfile.TemporaryDirectory(prefix="moe-rpc-", dir=self.scratch) as td:
                root = Path(td)
                request = root / "request.bin"
                phase_started = time.perf_counter()
                with request.open("wb") as f:
                    f.write(struct.pack("<I", int(layer)))
                    f.write(np.asarray(hidden[row], dtype="<f4").tobytes())
                    f.write(np.asarray(ids[row], dtype="<i4").tobytes())
                    f.write(np.asarray(weights[row], dtype="<f4").tobytes())
                row_file_write_s += time.perf_counter() - phase_started
                paths = [root / name for name in
                         ("routed.f32", "routed_parts.f32", "shared.f32", "shared_parts.f32")]
                phase_started = time.perf_counter()
                result = self.manager.infer(request_id, layer, request, *paths)
                row_manager_pair_s += time.perf_counter() - phase_started
                row_routed_service_s = result.get("routed_wall_seconds", 0.0)
                row_shared_service_s = result.get("shared_wall_seconds", 0.0)
                routed_service_s += row_routed_service_s
                shared_service_s += row_shared_service_s
                if result.get("status") != "ok":
                    raise RuntimeError(f"resident service rejected request: {result}")
                phase_started = time.perf_counter()
                routed = np.fromfile(paths[0], dtype="<f4")
                shared = np.fromfile(paths[2], dtype="<f4")
                if routed.shape != (HIDDEN_SIZE,) or shared.shape != (HIDDEN_SIZE,):
                    raise RuntimeError("resident service returned malformed result")
                outputs.append(routed + shared)
                row_read_join_s += time.perf_counter() - phase_started
            file_write_s += row_file_write_s
            manager_pair_s += row_manager_pair_s
            read_join_s += row_read_join_s
            tempdir_overhead_s += max(
                0.0, time.perf_counter() - row_started - row_file_write_s -
                row_manager_pair_s - row_read_join_s
            )
            tempdir_row_s = max(
                0.0, time.perf_counter() - row_started - row_file_write_s -
                row_manager_pair_s - row_read_join_s
            )
            timing_samples.append({
                "request_file_write_seconds": row_file_write_s,
                "tempdir_overhead_seconds": tempdir_row_s,
                "manager_pair_seconds": row_manager_pair_s,
                "routed_service_seconds": row_routed_service_s,
                "shared_service_seconds": row_shared_service_s,
                "result_read_join_seconds": row_read_join_s,
                "row_total_seconds": time.perf_counter() - row_started,
            })
        payload = np.stack(outputs).astype("<f4", copy=False).tobytes(order="C")
        with self.lock:
            self.timing["infer_total_seconds"] += time.perf_counter() - infer_started
            self.timing["request_file_write_seconds"] += file_write_s
            self.timing["tempdir_overhead_seconds"] += tempdir_overhead_s
            self.timing["manager_pair_seconds"] += manager_pair_s
            self.timing["routed_service_seconds"] += routed_service_s
            self.timing["shared_service_seconds"] += shared_service_s
            self.timing["result_read_join_seconds"] += read_join_s
            self.timing_samples.extend(timing_samples)
            if len(self.timing_samples) > 512:
                del self.timing_samples[:-512]
        return payload

    def timing_snapshot(self) -> dict[str, Any]:
        with self.lock:
            result = {"calls": self.calls, "rows": self.rows, **self.timing,
                      "recent_samples": list(self.timing_samples)}
        calls = result["calls"]
        if calls:
            for key, value in list(result.items()):
                if key.endswith("_seconds"):
                    result[key[:-8] + "_mean_seconds"] = value / calls
        return result

    def close(self) -> None:
        self._server.shutdown()
        self._server.server_close()
        self._thread.join(timeout=10)
        try:
            self.path.unlink()
        except FileNotFoundError:
            pass
