"""Versioned manifest for the retained Qwen3.8 GPU BAR1 mappings.

Worker stdout/stderr remains diagnostic text in ``.log`` files.  This module
extracts only GPU_MAP records into a small, strict TSV manifest consumed by
the direct PoC and vLLM adapters.
"""
from __future__ import annotations

import csv
import os
import re
from pathlib import Path

MAGIC = "QWEN38_GPU_MAPPINGS_MANIFEST\t1"
FIELDS = ("group", "worker", "bdf", "cuda_ordinal", "nvidia_minor",
          "cuda_va", "bar1_bus")
EXPECTED_RECORDS = 24
PDB_MAGIC = "QWEN38_GPU_PDB_MANIFEST\t2"
PDB_FIELDS = ("group", "worker", "bdf", "cuda_va", "pid", "va_space",
              "gpu_va_space", "root_phys", "root_aperture")
_BDF = re.compile(r"^[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]$")


def _record_from_log_line(line: str) -> dict[str, str] | None:
    if not line.startswith("GPU_MAP,"):
        return None
    fields = next(csv.reader([line]))
    if fields[0] != "GPU_MAP" or (len(fields) - 1) % 2:
        raise ValueError(f"malformed GPU_MAP record: {line!r}")
    record = dict(zip(fields[1::2], fields[2::2]))
    if set(record) != set(FIELDS):
        raise ValueError(f"GPU_MAP fields differ from manifest schema: {record}")
    for key in ("group", "worker", "cuda_ordinal", "nvidia_minor"):
        value = int(record[key], 10)
        if value < 0:
            raise ValueError(f"negative {key} in GPU_MAP record")
        record[key] = str(value)
    for key in ("cuda_va", "bar1_bus"):
        value = int(record[key], 0)
        if value < 0:
            raise ValueError(f"negative {key} in GPU_MAP record")
        record[key] = f"0x{value:x}"
    record["bdf"] = record["bdf"].lower()
    if not _BDF.fullmatch(record["bdf"]):
        raise ValueError(f"invalid PCI BDF in GPU_MAP record: {record['bdf']!r}")
    return record


def extract_gpu_mappings(*log_texts: str) -> list[dict[str, str]]:
    """Parse and validate exactly one complete six-card/24-GPU map set."""
    records: dict[tuple[int, int], dict[str, str]] = {}
    bdfs: set[str] = set()
    for text in log_texts:
        for line in text.splitlines():
            record = _record_from_log_line(line)
            if record is None:
                continue
            key = int(record["group"]), int(record["worker"])
            if not (0 <= key[0] < 6 and 0 <= key[1] < 4):
                raise ValueError(f"GPU_MAP index outside six-M10 topology: {key}")
            if key in records:
                raise ValueError(f"duplicate GPU_MAP worker {key}")
            if record["bdf"] in bdfs:
                raise ValueError(f"duplicate GPU_MAP BDF {record['bdf']}")
            bdfs.add(record["bdf"])
            records[key] = record
    expected = {(group, worker) for group in range(6) for worker in range(4)}
    if set(records) != expected:
        missing = sorted(expected - set(records))
        extra = sorted(set(records) - expected)
        raise ValueError(f"expected all 24 GPU_MAP records; missing={missing}, extra={extra}")
    return [records[(group, worker)] for group in range(6) for worker in range(4)]


def write_gpu_mappings_manifest(path: Path, records: list[dict[str, str]],
                                run_id: str) -> None:
    if len(records) != EXPECTED_RECORDS:
        raise ValueError(f"expected {EXPECTED_RECORDS} mapping records, got {len(records)}")
    if any(char in run_id for char in "\t\n\r"):
        raise ValueError("run_id cannot contain tab or newline characters")
    lines = [MAGIC, f"run_id\t{run_id}", f"records\t{EXPECTED_RECORDS}",
             "\t".join(FIELDS)]
    lines.extend("\t".join(record[field] for field in FIELDS) for record in records)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text("\n".join(lines) + "\n")
    os.replace(temp, path)


def validate_gpu_mappings_manifest(path: Path) -> list[dict[str, str]]:
    """Validate the manifest structure before passing it to a native client."""
    lines = path.read_text().splitlines()
    if len(lines) != 4 + EXPECTED_RECORDS or lines[0] != MAGIC:
        raise ValueError(f"invalid Qwen GPU mapping manifest header/length: {path}")
    if not lines[1].startswith("run_id\t") or not lines[1].split("\t", 1)[1]:
        raise ValueError(f"mapping manifest has no run_id: {path}")
    if lines[2] != f"records\t{EXPECTED_RECORDS}" or lines[3] != "\t".join(FIELDS):
        raise ValueError(f"mapping manifest schema mismatch: {path}")
    records: dict[tuple[int, int], dict[str, str]] = {}
    bdfs: set[str] = set()
    for line in lines[4:]:
        values = line.split("\t")
        if len(values) != len(FIELDS):
            raise ValueError(f"malformed mapping manifest record: {line!r}")
        record = dict(zip(FIELDS, values))
        key = int(record["group"], 10), int(record["worker"], 10)
        if not (0 <= key[0] < 6 and 0 <= key[1] < 4) or key in records:
            raise ValueError(f"invalid or duplicate mapping index: {key}")
        if not _BDF.fullmatch(record["bdf"]) or record["bdf"] in bdfs:
            raise ValueError(f"invalid or duplicate BDF in mapping record: {record['bdf']!r}")
        bdfs.add(record["bdf"])
        for field in ("cuda_ordinal", "nvidia_minor"):
            if int(record[field], 10) < 0:
                raise ValueError(f"invalid {field} in mapping record: {record[field]!r}")
        if int(record["cuda_va"], 0) < 0 or int(record["bar1_bus"], 0) < 0:
            raise ValueError(f"invalid address in mapping record: {record}")
        records[key] = record
    if len(records) != EXPECTED_RECORDS:
        raise ValueError(f"mapping manifest has {len(records)} records, expected 24")
    return [records[(group, worker)] for group in range(6) for worker in range(4)]


def extract_gpu_pdb_records(*log_texts: str,
                            mappings: list[dict[str, str]]) -> list[dict[str, str]]:
    """Read all candidate worker PDBs; the client selects by mapped return VA."""
    import csv
    records: dict[tuple[int, int], list[dict[str, str]]] = {}
    by_key = {(int(r["group"]), int(r["worker"])): r for r in mappings}
    for text in log_texts:
        for line in text.splitlines():
            if not line.startswith("GPU_PDB_CANDIDATE,"):
                continue
            fields = next(csv.reader([line]))
            if fields[0] != "GPU_PDB_CANDIDATE" or (len(fields) - 1) % 2:
                raise ValueError(f"malformed GPU_PDB_CANDIDATE record: {line!r}")
            row = dict(zip(fields[1::2], fields[2::2]))
            if set(row) != set(PDB_FIELDS):
                raise ValueError(f"GPU_PDB_CANDIDATE fields differ from schema: {row}")
            for key in ("group", "worker", "pid", "root_aperture"):
                row[key] = str(int(row[key], 10))
            for key in ("cuda_va", "va_space", "gpu_va_space", "root_phys"):
                row[key] = f"0x{int(row[key], 0):x}"
            row["bdf"] = row["bdf"].lower()
            key = int(row["group"]), int(row["worker"])
            mapping = by_key.get(key)
            if mapping is None:
                raise ValueError(f"unknown GPU_PDB_CANDIDATE worker {key}")
            if row["bdf"] != mapping["bdf"] or row["cuda_va"] != mapping["cuda_va"]:
                raise ValueError(f"GPU_PDB_CANDIDATE does not match retained mapping {key}")
            if not all(int(row[k], 0) for k in ("va_space", "gpu_va_space", "root_phys")):
                raise ValueError(f"GPU_PDB_CANDIDATE has null page-directory identity {key}")
            identity = tuple(row[k] for k in ("va_space", "gpu_va_space", "root_phys"))
            bucket = records.setdefault(key, [])
            if any(tuple(old[k] for k in ("va_space", "gpu_va_space", "root_phys")) == identity
                   for old in bucket):
                raise ValueError(f"duplicate GPU PDB candidate for worker {key}")
            bucket.append(row)
    expected = set(by_key)
    if set(records) != expected:
        raise ValueError(f"missing worker PDB candidates: {sorted(expected-set(records))}")
    return [row for g in range(6) for w in range(4)
            for row in sorted(records[(g, w)], key=lambda x: int(x["root_phys"], 0))]


def write_gpu_pdb_manifest(path: Path, records: list[dict[str, str]], run_id: str) -> None:
    if len(records) < EXPECTED_RECORDS:
        raise ValueError(f"expected at least {EXPECTED_RECORDS} worker PDB candidates, got {len(records)}")
    worker_keys = {(int(row["group"]), int(row["worker"])) for row in records}
    if len(worker_keys) != EXPECTED_RECORDS:
        raise ValueError(f"expected candidates for {EXPECTED_RECORDS} workers, got {len(worker_keys)}")
    lines = [PDB_MAGIC, f"run_id\t{run_id}", f"workers\t{EXPECTED_RECORDS}",
             f"candidates\t{len(records)}",
             "\t".join(PDB_FIELDS)]
    lines.extend("\t".join(row[field] for field in PDB_FIELDS) for row in records)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_name(path.name + f".{os.getpid()}.tmp")
    temp.write_text("\n".join(lines) + "\n")
    os.replace(temp, path)


def validate_gpu_pdb_manifest(path: Path) -> list[dict[str, str]]:
    lines = path.read_text().splitlines()
    if len(lines) < 5 or lines[0] != PDB_MAGIC:
        raise ValueError(f"invalid worker PDB manifest header/length: {path}")
    if not lines[1].startswith("run_id\t") or not lines[1].split("\t", 1)[1]:
        raise ValueError("worker PDB manifest has no run_id")
    if lines[2] != f"workers\t{EXPECTED_RECORDS}" or not lines[3].startswith("candidates\t") or lines[4] != "\t".join(PDB_FIELDS):
        raise ValueError("worker PDB manifest schema mismatch")
    candidate_count = int(lines[3].split("\t", 1)[1])
    if candidate_count < EXPECTED_RECORDS or len(lines) != 5 + candidate_count:
        raise ValueError("worker PDB candidate count mismatch")
    rows = []
    seen = set()
    workers = set()
    for line in lines[5:]:
        values = line.split("\t")
        if len(values) != len(PDB_FIELDS):
            raise ValueError(f"malformed worker PDB row: {line!r}")
        row = dict(zip(PDB_FIELDS, values))
        key = int(row["group"]), int(row["worker"])
        identity = key + tuple(row[field] for field in ("va_space", "gpu_va_space", "root_phys"))
        if key[0] not in range(6) or key[1] not in range(4) or identity in seen:
            raise ValueError(f"invalid/duplicate worker PDB candidate: {key}")
        seen.add(identity)
        workers.add(key)
        for field in ("cuda_va", "va_space", "gpu_va_space", "root_phys"):
            if int(row[field], 0) <= 0:
                raise ValueError(f"invalid {field} in worker PDB row")
        for field in ("pid", "root_aperture"):
            if int(row[field], 10) <= 0:
                raise ValueError(f"invalid {field} in worker PDB row")
        rows.append(row)
    if len(workers) != EXPECTED_RECORDS:
        raise ValueError(f"worker PDB manifest covers {len(workers)} workers")
    return sorted(rows, key=lambda r: (int(r["group"]), int(r["worker"]),
                                       int(r["root_phys"], 0)))
