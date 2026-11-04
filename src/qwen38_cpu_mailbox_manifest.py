"""Create and validate the POSIX shared-memory return mailbox manifest."""
from __future__ import annotations

import os
import re
import secrets
import mmap
import json
from pathlib import Path

MAGIC = "QWEN38_CPU_MAILBOX_MANIFEST"
VERSION = 1
WORKERS = 24
HIDDEN = 2560
SLOT_HEADER_BYTES = 16
SLOT_RESULT_BYTES = HIDDEN * 4
SLOT_GATE_BYTES = 4
SLOT_BYTES = SLOT_HEADER_BYTES + SLOT_RESULT_BYTES + SLOT_GATE_BYTES
SLOT_STRIDE = 16384
# 24 slots + the batched-MoE area (resident_cpu_mailbox_abi.cuh, Q38_BATCH_*; 2026-11-03)
BATCH_MAX_ROWS = 64
BATCH_PART_STRIDE = BATCH_MAX_ROWS * HIDDEN * 4 + 0x1000
BATCH_BYTES = 0xA2000 + 0x1000 + WORKERS * BATCH_PART_STRIDE
TOTAL_BYTES = WORKERS * SLOT_STRIDE + BATCH_BYTES
RTX_RETURN_MAGIC = "QWEN38_RTX_RETURN_ALIASES"
_NAME = re.compile(r"^/q38_cpu_mailbox_[a-f0-9]{16}$")


class CpuMailbox:
    def __init__(self, fd: int, name: str, mapping: mmap.mmap):
        self.fd, self.name, self.mapping = fd, name, mapping

    def close(self) -> None:
        self.mapping.close()
        os.close(self.fd)

    def unlink(self) -> None:
        os.unlink("/dev/shm" + self.name)


def create_mailbox(path: Path, run_id: str) -> tuple[CpuMailbox, dict]:
    """Create a named shm object and atomically publish its ABI manifest.

    The caller must keep the mapped file descriptor and mapping alive. Do not unlink it until
    the daemon has acknowledged UNLOAD_CPU_MAILBOX_MANIFEST.
    """
    name = "/q38_cpu_mailbox_" + secrets.token_hex(8)
    fd = os.open("/dev/shm" + name, os.O_CREAT | os.O_EXCL | os.O_RDWR, 0o600)
    try:
        os.ftruncate(fd, TOTAL_BYTES)
        mapping = mmap.mmap(fd, TOTAL_BYTES, access=mmap.ACCESS_WRITE)
        mapping[:] = bytes(TOTAL_BYTES)
    except BaseException:
        os.close(fd)
        os.unlink("/dev/shm" + name)
        raise
    manifest = {
        "magic": MAGIC,
        "version": VERSION,
        "run_id": run_id,
        "shm_name": name,
        "bytes": TOTAL_BYTES,
        "workers": WORKERS,
        "slot_stride": SLOT_STRIDE,
        "slot_bytes": SLOT_BYTES,
        "layout": {
            "seq_u32": 0,
            "status_u32": 4,
            "worker_u32": 8,
            "layer_u32": 12,
            "result_f32": SLOT_HEADER_BYTES,
            "gate_f32": SLOT_HEADER_BYTES + SLOT_RESULT_BYTES,
        },
    }
    try:
        atomic_write(path, manifest)
    except BaseException:
        mapping.close()
        os.close(fd)
        os.unlink("/dev/shm" + name)
        raise
    return CpuMailbox(fd, name, mapping), manifest


def attach_mailbox(path: Path) -> tuple[CpuMailbox, dict]:
    """Attach to a mailbox already registered by the long-lived daemon.

    The daemon owns registration and CUDA mappings; this process only opens a
    second CPU view so it can inspect/use the same shared return slots.
    """
    manifest = validate_manifest(path)
    shm_path = "/dev/shm" + manifest["shm_name"]
    fd = os.open(shm_path, os.O_RDWR | os.O_CLOEXEC)
    try:
        mapping = mmap.mmap(fd, TOTAL_BYTES, access=mmap.ACCESS_WRITE)
    except BaseException:
        os.close(fd)
        raise
    return CpuMailbox(fd, manifest["shm_name"], mapping), manifest


def atomic_write(path: Path, manifest: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    lines = [f"{MAGIC}\t{VERSION}", f"run_id\t{manifest['run_id']}",
             f"shm_name\t{manifest['shm_name']}", f"bytes\t{TOTAL_BYTES}",
             f"workers\t{WORKERS}", f"slot_stride\t{SLOT_STRIDE}",
             f"slot_bytes\t{SLOT_BYTES}"]
    temp.write_text("\n".join(lines) + "\n")
    os.replace(temp, path)


def validate_manifest(path: Path) -> dict:
    lines = path.read_text().splitlines()
    if len(lines) != 7 or lines[0] != f"{MAGIC}\t{VERSION}":
        raise ValueError("unsupported CPU mailbox manifest version or length")
    fields = {}
    for line in lines[1:]:
        key, sep, value = line.partition("\t")
        if not sep or key in fields:
            raise ValueError(f"malformed CPU mailbox manifest line: {line!r}")
        fields[key] = value
    if set(fields) != {"run_id", "shm_name", "bytes", "workers", "slot_stride", "slot_bytes"}:
        raise ValueError("CPU mailbox manifest fields differ from version 1 schema")
    if not fields["run_id"] or any(c in fields["run_id"] for c in "\t\r\n"):
        raise ValueError("CPU mailbox manifest has an invalid run_id")
    if not _NAME.fullmatch(fields["shm_name"]):
        raise ValueError("CPU mailbox manifest has an invalid shm_name")
    expected = {"bytes": TOTAL_BYTES, "workers": WORKERS,
                "slot_stride": SLOT_STRIDE, "slot_bytes": SLOT_BYTES}
    for key, wanted in expected.items():
        if int(fields[key], 10) != wanted:
            raise ValueError(f"CPU mailbox manifest field {key!r} is invalid")
    return {"magic": MAGIC, "version": VERSION, **fields,
            "bytes": TOTAL_BYTES, "workers": WORKERS,
            "slot_stride": SLOT_STRIDE, "slot_bytes": SLOT_BYTES}


def load_from_daemon(control_socket: Path, manifest_path: Path,
                     timeout: float = 30.0) -> dict:
    import socket
    request = {"command": "LOAD_CPU_MAILBOX_MANIFEST",
               "manifest_path": str(manifest_path.resolve()),
               "clients_quiesced": True}
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(str(control_socket))
        sock.sendall((json.dumps(request) + "\n").encode())
        chunks = bytearray()
        while b"\n" not in chunks:
            block = sock.recv(65536)
            if not block:
                raise EOFError("daemon closed the CPU mailbox control request")
            chunks.extend(block)
        response = json.loads(bytes(chunks).split(b"\n", 1)[0])
    finally:
        sock.close()
    if response.get("status") != "ok":
        raise RuntimeError(f"daemon rejected CPU mailbox manifest: {response}")
    return response


def unload_from_daemon(control_socket: Path, timeout: float = 30.0) -> dict:
    import socket
    request = {"command": "UNLOAD_CPU_MAILBOX_MANIFEST", "clients_quiesced": True}
    sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    sock.settimeout(timeout)
    try:
        sock.connect(str(control_socket))
        sock.sendall((json.dumps(request) + "\n").encode())
        chunks = bytearray()
        while b"\n" not in chunks:
            block = sock.recv(65536)
            if not block:
                raise EOFError("daemon closed the CPU mailbox unload request")
            chunks.extend(block)
        response = json.loads(bytes(chunks).split(b"\n", 1)[0])
    finally:
        sock.close()
    if response.get("status") != "ok":
        raise RuntimeError(f"daemon rejected CPU mailbox unload: {response}")
    return response


def write_rtx_return_alias_manifest(path: Path, registration: dict) -> None:
    """Persist daemon-owned per-GPU mailbox VAs for M10→RTX route setup."""
    if registration.get("status") != "ok" or registration.get("code") != "CPU_MAILBOX_LOADED":
        raise ValueError("CPU mailbox registration did not provide a successful response")
    slots = registration.get("device_slots")
    if not isinstance(slots, list) or len(slots) != WORKERS:
        raise ValueError(f"expected {WORKERS} daemon mailbox device aliases")
    rows: dict[tuple[int, int], tuple[str, int]] = {}
    bdfs: set[str] = set()
    for slot in slots:
        group, worker = int(slot["group"]), int(slot["worker"])
        bdf = str(slot["bdf"]).lower()
        device_va = int(slot["device_va"], 0)
        key = group, worker
        if not (0 <= group < 6 and 0 <= worker < 4) or key in rows:
            raise ValueError(f"invalid or duplicate return-alias index {key}")
        if not re.fullmatch(r"[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]", bdf):
            raise ValueError(f"invalid return-alias BDF {bdf!r}")
        if bdf in bdfs or device_va <= 0 or device_va & 0xFFF:
            raise ValueError(f"duplicate BDF or invalid/unaligned device VA for {bdf}")
        bdfs.add(bdf)
        rows[key] = bdf, device_va
    expected = {(group, worker) for group in range(6) for worker in range(4)}
    if set(rows) != expected:
        raise ValueError(f"return-alias manifest is incomplete: {sorted(expected-set(rows))}")
    run_id = str(registration.get("run_id", ""))
    if not run_id or any(ch in run_id for ch in "\t\r\n"):
        raise ValueError("invalid return-alias run_id")
    lines = [f"{RTX_RETURN_MAGIC}\t1", f"run_id\t{run_id}",
             f"records\t{WORKERS}", "group\tworker\tbdf\tdevice_va"]
    for group, worker in sorted(rows):
        bdf, device_va = rows[(group, worker)]
        lines.append(f"{group}\t{worker}\t{bdf}\t0x{device_va:x}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text("\n".join(lines) + "\n")
    os.replace(temp, path)
