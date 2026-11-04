#!/usr/bin/env python3
"""Read-only pre-daemon check of the six PEX8747 group upstreams before the mapper is built.

For each topology row: device is 10b5:8747, the Multicast extended capability is found by
walking the ECAP list (pex_group_config.mc_cap, never a fixed offset), multicast is still
disabled (the daemon captures and applies it during its direct-map bootstrap), and the
ingress base is 64 KiB aligned and inside both the TSV branch window and the live bridge
prefetchable window. Saves each upstream's 4 KiB config space and a JSON record.
Writes nothing to PCI config space. Usage: check_qwen38_mc_premap.py TOPOLOGY.tsv OUT_DIR
"""
import datetime, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent))
import pex_group_config as p

topology, out = Path(sys.argv[1]), Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
rows = [l.split('\t') for l in topology.read_text().splitlines() if l]
assert len(rows) == 6 and all(len(r) == 7 for r in rows), 'expected six 7-column topology rows'
groups = []
for r in rows:
    up, start, end, ingress = r[1], int(r[4], 0), int(r[5], 0), int(r[6], 0)
    vendor, device = p.r16(up, 0), p.r16(up, 2)
    assert (vendor, device) == (0x10b5, 0x8747), f'{up}: id {vendor:04x}:{device:04x}'
    cap = p.mc_cap(up)
    regs = p.regs(up)
    win_lo, win_hi = p.window(up)
    e = {'upstream': up, 'mc_cap_offset': hex(cap), 'regs_before': {k: hex(v) for k, v in regs.items()},
         'enabled_before': bool(regs['control'] >> 15 & 1), 'desired_ingress': hex(ingress),
         'branch': [hex(start), hex(end)], 'bridge_window': [hex(win_lo), hex(win_hi)]}
    assert not e['enabled_before'], f'{up}: multicast already enabled before daemon start: {e}'
    assert ingress % 65536 == 0 and start <= ingress and ingress + 65535 <= end, f'{up}: ingress outside branch: {e}'
    assert win_lo <= ingress and ingress + 65535 <= win_hi, f'{up}: ingress outside bridge window: {e}'
    (out / (up.replace(':', '_') + '.upstream.config.bin')).write_bytes(p.read(up, 0, 4096))
    groups.append(e)
    print(f"MC_PREMAP,{up},cap,{hex(cap)},enabled,0,ingress,{hex(ingress)},window,{hex(win_lo)}-{hex(win_hi)},ok,1")
(out / 'mc-premap.json').write_text(json.dumps({'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
                                                 'groups': groups}, indent=2) + '\n')
print('MC_PREMAP_OK,groups,6')
