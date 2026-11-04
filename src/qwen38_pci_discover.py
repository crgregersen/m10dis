#!/usr/bin/env python3
"""Discover the RTX, the M10 GPUs and their PCIe switch ports by scanning sysfs.

Operator rule: never hardcode PCI bus addresses; bus numbers change whenever
hardware is added or moved. This script is read-only (sysfs + nvidia-smi).

Identification (vendor:device, then the parent chain in sysfs):
  M10 GPU             10de:13bd (Tesla M10, GM107)
  M10 inner switch    10b5:8747 (GPU parent = downstream port, its parent = upstream port)
  outer switch        10b5:8748 (PEX8747 upstream's parent = outer downstream port)
  RTX                 the single NVIDIA display-class GPU that is not 10de:13bd;
                      its parent must be a 10b5:8748 downstream port of the same
                      outer switch as every M10 group.
Groups are the PEX8747 upstream ports, ordered by their GPUs' bus numbers, or by
the row order of --topology TSV after verifying it matches the scan.

Usage: qwen38_pci_discover.py [--topology TOPOLOGY.tsv] (--json | --env)
Exit status 2 on any ambiguity or mismatch, with an ERROR line on stderr.
"""
import argparse, csv, json, os, subprocess, sys

# Test-only overrides (used by test_qwen38_pci_discover.py with synthetic layouts):
#   Q38_PCI_SYSFS_DEVICES  directory standing in for /sys/bus/pci/devices
#   Q38_NVIDIA_SMI_CSV     file with `nvidia-smi --query-gpu=pci.bus_id,uuid,name` output
SYS = os.environ.get('Q38_PCI_SYSFS_DEVICES', '/sys/bus/pci/devices')
M10, PEX8747, PEX8748 = ('0x10de', '0x13bd'), ('0x10b5', '0x8747'), ('0x10b5', '0x8748')


def fail(msg):
    print(f'ERROR,pci_discover,{msg}', file=sys.stderr)
    sys.exit(2)


def rd(bdf, name):
    with open(os.path.join(SYS, bdf, name)) as f:
        return f.read().strip()


def ident(bdf):
    return rd(bdf, 'vendor'), rd(bdf, 'device')


def parent(bdf):
    p = os.path.basename(os.path.dirname(os.path.realpath(os.path.join(SYS, bdf))))
    return p if os.path.exists(os.path.join(SYS, p)) else None


def need(bdf, want, role):
    if bdf is None or ident(bdf) != want:
        fail(f'{role} {bdf} has id {ident(bdf) if bdf else None}, expected {want[0]}:{want[1]}')


def children(bdf):
    path = os.path.realpath(os.path.join(SYS, bdf))
    return sorted(e for e in os.listdir(path) if os.path.exists(os.path.join(SYS, e))
                  and os.path.realpath(os.path.join(SYS, e)).startswith(path + '/'))


def scan():
    devs = sorted(os.listdir(SYS))
    m10 = [d for d in devs if ident(d) == M10]
    if not m10:
        fail('no Tesla M10 (10de:13bd) GPUs found')
    rtx = [d for d in devs if rd(d, 'vendor') == '0x10de' and rd(d, 'class').startswith('0x03')
           and ident(d) != M10]
    if len(rtx) != 1:
        fail(f'expected exactly one non-M10 NVIDIA display GPU, found {rtx}')
    rtx = rtx[0]
    groups = {}
    for g in m10:
        dn = parent(g); need(dn, PEX8747, f'parent of M10 {g}')
        up = parent(dn); need(up, PEX8747, f'PEX8747 upstream above {dn}')
        outer = parent(up); need(outer, PEX8748, f'outer port above {up}')
        grp = groups.setdefault(up, {'upstream': up, 'outer_port': outer, 'inner_ports': [], 'gpus': []})
        if grp['outer_port'] != outer:
            fail(f'group {up} reached via two outer ports')
        grp['inner_ports'].append(dn); grp['gpus'].append(g)
    for grp in groups.values():
        if len(grp['gpus']) != 4:
            fail(f'group {grp["upstream"]} has {len(grp["gpus"])} GPUs, expected 4')
        order = sorted(range(4), key=lambda i: grp['gpus'][i])
        grp['gpus'] = [grp['gpus'][i] for i in order]
        grp['inner_ports'] = [grp['inner_ports'][i] for i in order]
    rtx_port = parent(rtx); need(rtx_port, PEX8748, f'parent of RTX {rtx}')
    switch_up = parent(rtx_port); need(switch_up, PEX8748, f'outer switch upstream above {rtx_port}')
    for grp in groups.values():
        if parent(grp['outer_port']) != switch_up:
            fail(f'group {grp["upstream"]} is not behind outer switch upstream {switch_up}')
    outer_ports = [d for d in children(switch_up) if ident(d) == PEX8748]
    used = {g['outer_port'] for g in groups.values()} | {rtx_port}
    free = [p for p in outer_ports if p not in used and not children(p)]
    ordered = sorted(groups.values(), key=lambda g: g['gpus'][0])
    uuid = None
    try:
        smi_file = os.environ.get('Q38_NVIDIA_SMI_CSV')
        out = open(smi_file).read() if smi_file else subprocess.check_output(
            ['nvidia-smi', '--query-gpu=pci.bus_id,uuid,name', '--format=csv,noheader'], text=True, timeout=30)
        for row in csv.reader(out.splitlines(), skipinitialspace=True):
            if len(row) == 3 and row[0].strip().lower().replace('00000000:', '0000:') == rtx:
                uuid, name = row[1].strip(), row[2].strip()
    except (OSError, subprocess.SubprocessError) as e:
        fail(f'nvidia-smi failed: {e}')
    if not uuid:
        fail(f'nvidia-smi does not list the RTX {rtx}')
    return {'rtx': {'bdf': rtx, 'uuid': uuid, 'name': name, 'id': ':'.join(x[2:] for x in ident(rtx)),
                    'outer_port': rtx_port},
            'outer_switch_upstream': switch_up, 'outer_free_ports': free, 'groups': ordered}


def apply_topology(found, path):
    rows = [l.split('\t') for l in open(path).read().splitlines() if l.strip()]
    by_up = {g['upstream']: g for g in found['groups']}
    if len(rows) != len(by_up):
        fail(f'topology {path} lists {len(rows)} groups, scan found {len(by_up)}')
    ordered = []
    for r in rows:
        if len(r) < 4 or r[1] not in by_up:
            fail(f'topology row upstream {r[1] if len(r) > 1 else r} not found by scan')
        g = by_up[r[1]]
        if sorted(r[2].split(',')) != sorted(g['inner_ports']) or sorted(r[3].split(',')) != sorted(g['gpus']):
            fail(f'topology row for {r[1]} differs from scan (ports/GPUs)')
        g['inner_ports'] = r[2].split(','); g['gpus'] = r[3].split(',')
        ordered.append(g)
    found['groups'] = ordered
    found['topology'] = os.path.abspath(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument('--topology')
    m = ap.add_mutually_exclusive_group(required=True)
    m.add_argument('--json', action='store_true'); m.add_argument('--env', action='store_true')
    a = ap.parse_args()
    found = scan()
    if a.topology:
        apply_topology(found, a.topology)
    if a.json:
        print(json.dumps(found, indent=2)); return
    g = found['groups']
    env = {
        'Q38_RTX_BDF': found['rtx']['bdf'], 'Q38_RTX_UUID': found['rtx']['uuid'],
        'Q38_RTX_OUTER_PORT': found['rtx']['outer_port'],
        'Q38_OUTER_SWITCH_UPSTREAM': found['outer_switch_upstream'],
        'Q38_GROUP_UPSTREAMS': ' '.join(x['upstream'] for x in g),
        'Q38_GROUP_OUTER_PORTS': ' '.join(x['outer_port'] for x in g),
        'Q38_GROUP_INNER_PORTS': ' '.join(p for x in g for p in x['inner_ports']),
        'Q38_GROUP0_UPSTREAM': g[0]['upstream'],
        'Q38_GROUP0_INNER_PORTS': ' '.join(g[0]['inner_ports']),
        'Q38_GROUP0_GPUS': ' '.join(g[0]['gpus']),
        'Q38_M10_GPUS': ' '.join(p for x in g for p in x['gpus']),
        'Q38_SHARED_GROUP_OUTER_PORT': g[-1]['outer_port'],
        'Q38_SHARED_GROUP_GPUS': ' '.join(g[-1]['gpus']),
        'Q38_GROUP_COUNT': str(len(g)),
    }
    for k, v in env.items():
        print(f"export {k}='{v}'")


if __name__ == '__main__':
    main()
