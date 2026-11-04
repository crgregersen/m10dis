#!/usr/bin/env python3
"""Read-only discovery of M10-local PEX8747 groups for Qwen collectives.

This reports candidate local four-GPU islands and bridge windows. It never
writes PCI configuration, maps BARs, loads a module, or changes GPU state.
"""
from __future__ import annotations

import json
import pathlib
import subprocess
import sys

SYS_PCI = pathlib.Path("/sys/bus/pci/devices")


def query_gpus() -> list[dict[str, str]]:
    q = subprocess.run(
        ["nvidia-smi", "--query-gpu=serial,uuid,name,pci.bus_id",
         "--format=csv,noheader,nounits"],
        text=True, capture_output=True, check=True)
    out = []
    for line in q.stdout.splitlines():
        serial, uuid, name, bdf = (part.strip() for part in line.split(",", 3))
        if "Tesla M10" in name:
            out.append({"serial": serial, "uuid": uuid, "name": name,
                        "bdf": bdf.lower().replace("00000000:", "0000:")})
    return out


def path_bdfs(gpu_bdf: str) -> list[str]:
    dev = (SYS_PCI / gpu_bdf).resolve()
    names = [p.name for p in dev.parents]
    names.reverse()
    return [n for n in names if len(n.split(":")) == 3 and "." in n]


def pex_identity(bdf: str) -> tuple[int, int] | None:
    vendor = (SYS_PCI / bdf / "vendor")
    device = (SYS_PCI / bdf / "device")
    try:
        return int(vendor.read_text(), 16), int(device.read_text(), 16)
    except OSError:
        return None


def resource_ranges(bdf: str) -> list[dict[str, int]]:
    rows = []
    for index, line in enumerate((SYS_PCI / bdf / "resource").read_text().splitlines()):
        start, end, flags = (int(value, 16) for value in line.split())
        if start or end:
            rows.append({"bar_index": index, "start": start, "end": end,
                         "length": end - start + 1, "flags": flags})
    return rows


def bridge_window(bdf: str) -> dict[str, int] | None:
    text = subprocess.run(["lspci", "-s", bdf, "-vv"], check=True,
                          text=True, capture_output=True).stdout
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("Prefetchable memory behind bridge:"):
            import re
            match = re.search(r"([0-9a-fA-F]+)-([0-9a-fA-F]+)", line)
            if match:
                a, b = (int(v, 16) for v in match.groups())
                return {"start": a, "end": b, "length": b - a + 1}
    return None


def find_multicast_capability(bdf):
    """Walk the PCIe extended capability list from 0x100 for the Multicast capability
    (ID 0x0012); its offset is never assumed. Needs root to read past 64 bytes."""
    try:
        with (SYS_PCI / bdf / "config").open("rb") as cfg:
            raw = cfg.read(4096)
    except PermissionError:
        return "requires-root-read"
    if len(raw) < 4096:
        return "requires-root-read"
    found, seen, pos = [], set(), 0x100
    while pos and pos not in seen and 0x100 <= pos <= 0xFFC:
        seen.add(pos)
        header = int.from_bytes(raw[pos:pos + 4], "little")
        if header in (0, 0xFFFFFFFF):
            break
        if header & 0xFFFF == 0x0012:
            found.append({"offset": hex(pos), "header": hex(header)})
        pos = (header >> 20) & 0xFFC
    if len(found) != 1:
        return f"multicast capability found {len(found)} times"
    return found[0]

def main() -> int:
    groups: dict[str, list[dict[str, str]]] = {}
    for gpu in query_gpus():
        groups.setdefault(gpu["serial"], []).append(gpu)

    report = []
    for serial, gpus in sorted(groups.items(), key=lambda item: min(g["bdf"] for g in item[1])):
        paths = {g["bdf"]: path_bdfs(g["bdf"]) for g in gpus}
        pex_sets = []
        for chain in paths.values():
            pex_sets.append([b for b in chain if pex_identity(b) == (0x10B5, 0x8747)])
        common = set(pex_sets[0]) if pex_sets else set()
        for candidate in pex_sets[1:]:
            common.intersection_update(candidate)
        # The nearest PEX8747 common to all four endpoints is the card-local
        # switch upstream function; per-GPU downstream ports are the closest
        # PEX8747 ancestors below it.
        local_upstream = None
        if common:
            local_upstream = min(common, key=lambda b: max(chain.index(b) for chain in paths.values()))
        ports = {}
        if local_upstream:
            for gpu in gpus:
                chain = paths[gpu["bdf"]]
                idx = chain.index(local_upstream)
                downstream = [b for b in chain[idx + 1:]
                              if pex_identity(b) == (0x10B5, 0x8747)]
                ports[gpu["bdf"]] = downstream[0] if downstream else None
        group = {"serial": serial, "gpu_count": len(gpus), "gpus": gpus,
                 "path_bdfs": paths, "local_pex_upstream_bdf": local_upstream,
                 "gpu_downstream_ports": ports,
                 "multicast_capability": None,
                 "local_prefetchable_window": bridge_window(local_upstream) if local_upstream else None,
                 "gpu_resources": {g["bdf"]: resource_ranges(g["bdf"]) for g in gpus}}
        if local_upstream and (SYS_PCI / local_upstream / "config").exists():
            group["multicast_capability"] = find_multicast_capability(local_upstream)
        report.append(group)

    json.dump({"host": __import__("socket").gethostname(), "groups": report}, sys.stdout, indent=2)
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
