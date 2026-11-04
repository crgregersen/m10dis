#!/usr/bin/env python3
"""Tests for qwen38_pci_discover.py (PCI discovery by scanning; no hardcoded addresses).

Part 1, synthetic: builds fake sysfs trees shaped like this host (root port -> PEX8748
outer switch -> six PEX8747 cards x 4 Tesla M10 GPUs, RTX on its own outer port, NVMe on
the chipset) and runs the discovery against them via Q38_PCI_SYSFS_DEVICES and
Q38_NVIDIA_SMI_CSV. Covers renumbered buses (RTX moved, NVMe on the old RTX address),
the RTX audio function, and every refusal case.
Part 2, live (read-only): runs the discovery on this machine and cross-checks it against
nvidia-smi, lspci-style parent chains, the newest daemon topology.tsv and the shell helper.

Usage: test_qwen38_pci_discover.py [--synthetic-only]
Prints one CHECK line per case and TEST_PCI_DISCOVER,PASS|FAIL; exit 0 only if all pass.
"""
import json, os, shutil, subprocess, sys, tempfile, glob
from pathlib import Path

HERE = Path(__file__).resolve().parent
DISCOVER = HERE / 'qwen38_pci_discover.py'
BRIDGE, VGA, VGA3D, AUDIO, NVME = '0x060400', '0x030000', '0x030200', '0x040300', '0x010802'
results = []

def check(name, ok, detail=''):
    results.append((name, bool(ok)))
    print(f"CHECK,{name},{'OK' if ok else 'FAIL'}{',' + detail if detail else ''}", flush=True)

# ---------------------------------------------------------------- synthetic sysfs
class FakeSys:
    def __init__(self):
        self.root = Path(tempfile.mkdtemp(prefix='q38-fake-sysfs-'))
        self.devs = self.root / 'bus/pci/devices'; self.devs.mkdir(parents=True)
        self.top = self.root / 'devices/pci0000:00'; self.top.mkdir(parents=True)
        self.smi = []
    def add(self, parent_dir, bdf, vendor, device, cls):
        d = (parent_dir or self.top) / bdf; d.mkdir()
        for k, v in (('vendor', vendor), ('device', device), ('class', cls)):
            (d / k).write_text(v + '\n')
        os.symlink(d, self.devs / bdf)
        return d
    def gpu_smi(self, bdf, uuid, name):
        self.smi.append(f"{bdf.replace('0000:', '00000000:').upper()}, {uuid}, {name}")
    def run(self, *args):
        smi = self.root / 'nvidia-smi.csv'; smi.write_text('\n'.join(self.smi) + '\n')
        env = dict(os.environ, Q38_PCI_SYSFS_DEVICES=str(self.devs), Q38_NVIDIA_SMI_CSV=str(smi))
        return subprocess.run([sys.executable, str(DISCOVER), *args], env=env, capture_output=True, text=True)
    def cleanup(self):
        shutil.rmtree(self.root)

def b(bus, dev=0, fn=0):
    return f'0000:{bus:02x}:{dev:02x}.{fn}'

def build(rtx_bus=0x2d, nvme_bus=0x33, group_base=0x03, outer_bus=0x02, groups=6,
          extra_rtx=False, wrong_switch=False, short_group=False, rtx_other_switch=False,
          rtx_audio=True, rtx_in_smi=True, free_port=True):
    """Host-shaped tree. Returns (FakeSys, expected dict)."""
    f = FakeSys()
    rp = f.add(None, b(0, 1), '0x8086', '0x1901', BRIDGE)               # CPU root port
    up = f.add(rp, b(outer_bus - 1), '0x10b5', '0x8748', BRIDGE)          # outer switch upstream
    outer_devs = [0x08, 0x09, 0x0a, 0x0b, 0x10, 0x11][:groups]
    exp = {'groups': []}
    bus = group_base
    for gi, od in enumerate(outer_devs):
        op = f.add(up, b(outer_bus, od), '0x10b5', '0x8748', BRIDGE)
        g_up = f.add(op, b(bus), '0x10b5', '0x8749' if (wrong_switch and gi == 2) else '0x8747', BRIDGE)
        inner, gpus = [], []
        for i, pd in enumerate((0x08, 0x09, 0x10, 0x11)):
            if short_group and gi == 4 and i == 3:
                break
            dn = f.add(g_up, b(bus + 1, pd), '0x10b5', '0x8747', BRIDGE)
            gpu = f.add(dn, b(bus + 2 + i), '0x10de', '0x13bd', VGA3D)
            f.gpu_smi(gpu.name, f'GPU-m10-{gi}-{i}', 'Tesla M10')
            inner.append(dn.name); gpus.append(gpu.name)
        exp['groups'].append({'upstream': g_up.name, 'outer_port': op.name, 'inner_ports': inner, 'gpus': gpus})
        bus += 6
    if free_port:
        f.add(up, b(outer_bus, 0x12), '0x10b5', '0x8748', BRIDGE)
    if rtx_other_switch:                                              # RTX on a different root port
        rp2 = f.add(None, b(0, 2), '0x8086', '0x1905', BRIDGE)
        up2 = f.add(rp2, b(0x40), '0x10b5', '0x8748', BRIDGE)
        rport = f.add(up2, b(0x41, 0x13), '0x10b5', '0x8748', BRIDGE)
    else:
        rport = f.add(up, b(outer_bus, 0x13), '0x10b5', '0x8748', BRIDGE)
    rtx = f.add(rport, b(rtx_bus), '0x10de', '0x2482', VGA)
    if rtx_audio:
        f.add(rport, b(rtx_bus, 0, 1), '0x10de', '0x228b', AUDIO)
    if rtx_in_smi:
        f.gpu_smi(rtx.name, 'GPU-rtx-uuid', 'NVIDIA GeForce RTX 3070 Ti')
    if extra_rtx:
        rport2 = f.add(up, b(outer_bus, 0x14), '0x10b5', '0x8748', BRIDGE)
        f.add(rport2, b(rtx_bus + 1), '0x10de', '0x2484', VGA)
    pch = f.add(None, b(0, 0x1b), '0x8086', '0xa2e7', BRIDGE)            # chipset root port
    f.add(pch, b(nvme_bus), '0x126f', '0x2263', NVME)
    exp.update(rtx=rtx.name, rtx_port=rport.name, switch_up=up.name,
               free=[b(outer_bus, 0x12)] if free_port else [])
    return f, exp

def synthetic():
    def ok_case(name, **kw):
        f, exp = build(**kw)
        try:
            r = f.run('--json')
            if r.returncode != 0:
                return check(name, False, r.stderr.strip())
            j = json.loads(r.stdout)
            good = (j['rtx']['bdf'] == exp['rtx'] and j['rtx']['outer_port'] == exp['rtx_port']
                    and j['rtx']['uuid'] == 'GPU-rtx-uuid' and j['outer_switch_upstream'] == exp['switch_up']
                    and j['outer_free_ports'] == exp['free']
                    and [{k: g[k] for k in ('upstream', 'outer_port', 'inner_ports', 'gpus')} for g in j['groups']] == exp['groups'])
            # --env must carry the same values
            e = f.run('--env'); env = dict(l[len('export '):].split('=', 1) for l in e.stdout.splitlines())
            env = {k: v.strip("'") for k, v in env.items()}
            good &= (env['Q38_RTX_BDF'] == exp['rtx'] and env['Q38_GROUP_COUNT'] == str(len(exp['groups']))
                     and env['Q38_GROUP_OUTER_PORTS'].split() == [g['outer_port'] for g in exp['groups']]
                     and env['Q38_M10_GPUS'].split() == [x for g in exp['groups'] for x in g['gpus']])
            check(name, good, f"rtx={j['rtx']['bdf']}")
        finally:
            f.cleanup()

    def refuse_case(name, expect_text, **kw):
        f, _ = build(**kw)
        try:
            r = f.run('--json')
            check(name, r.returncode == 2 and expect_text in r.stderr, r.stderr.strip()[:120])
        finally:
            f.cleanup()

    ok_case('synthetic_host_layout')
    # The 2026-11-02 case: hardware added, buses renumbered, NVMe on the old RTX address.
    ok_case('synthetic_renumbered_rtx_moved_nvme_on_old_address', rtx_bus=0x3a, nvme_bus=0x2d, group_base=0x05)
    ok_case('synthetic_without_rtx_audio_function', rtx_audio=False)
    ok_case('synthetic_no_free_outer_port', free_port=False)
    refuse_case('refuse_two_rtx_gpus', 'exactly one non-M10', extra_rtx=True)
    refuse_case('refuse_wrong_switch_id', 'expected 0x10b5:0x8747', wrong_switch=True)
    refuse_case('refuse_group_with_3_gpus', 'expected 4', short_group=True)
    refuse_case('refuse_rtx_behind_other_switch', 'outer switch', rtx_other_switch=True)
    refuse_case('refuse_rtx_missing_from_nvidia_smi', 'nvidia-smi does not list', rtx_in_smi=False)

    # --topology: matching (reordered) TSV is accepted in its row order; a mismatch is refused.
    f, exp = build()
    try:
        rows = list(reversed(exp['groups']))
        tsv = f.root / 'topology.tsv'
        tsv.write_text(''.join(f"serial{i}\t{g['upstream']}\t{','.join(g['inner_ports'])}\t{','.join(g['gpus'])}\t0\t0\t0\n"
                               for i, g in enumerate(rows)))
        r = f.run('--json', '--topology', str(tsv))
        check('topology_reordered_accepted_in_tsv_order',
              r.returncode == 0 and [g['upstream'] for g in json.loads(r.stdout)['groups']] == [g['upstream'] for g in rows])
        bad = rows[0]['gpus'][:]; bad[0] = b(0x7f)
        tsv.write_text(''.join(f"serial{i}\t{g['upstream']}\t{','.join(g['inner_ports'])}\t{','.join(bad if i == 0 else g['gpus'])}\t0\t0\t0\n"
                               for i, g in enumerate(rows)))
        r = f.run('--json', '--topology', str(tsv))
        check('topology_mismatch_refused', r.returncode == 2 and 'differs from scan' in r.stderr, r.stderr.strip()[:120])
    finally:
        f.cleanup()

# ---------------------------------------------------------------- live, read-only
def live():
    r = subprocess.run([sys.executable, str(DISCOVER), '--json'], capture_output=True, text=True)
    check('live_discovery_runs', r.returncode == 0, r.stderr.strip()[:120])
    if r.returncode:
        return
    j = json.loads(r.stdout)
    sysd = '/sys/bus/pci/devices'
    rd = lambda d, n: open(f'{sysd}/{d}/{n}').read().strip()
    parent = lambda d: os.path.basename(os.path.dirname(os.path.realpath(f'{sysd}/{d}')))
    gpus = [g for grp in j['groups'] for g in grp['gpus']]
    check('live_24_m10_in_6_groups', len(j['groups']) == 6 and len(gpus) == 24 and len(set(gpus)) == 24)
    check('live_rtx_identity', (rd(j['rtx']['bdf'], 'vendor'), rd(j['rtx']['bdf'], 'device')) == ('0x10de', '0x2482'),
          j['rtx']['bdf'])
    chain_ok = all(parent(g) in grp['inner_ports'] and parent(parent(g)) == grp['upstream']
                   and parent(grp['upstream']) == grp['outer_port'] and parent(grp['outer_port']) == j['outer_switch_upstream']
                   for grp in j['groups'] for g in grp['gpus'])
    chain_ok &= parent(j['rtx']['bdf']) == j['rtx']['outer_port'] and parent(j['rtx']['outer_port']) == j['outer_switch_upstream']
    check('live_parent_chains', chain_ok)
    smi = subprocess.run(['nvidia-smi', '--query-gpu=pci.bus_id,name', '--format=csv,noheader'], capture_output=True, text=True).stdout
    smi_m10 = sorted(l.split(',')[0].strip().lower().replace('00000000:', '0000:') for l in smi.splitlines() if 'M10' in l)
    check('live_matches_nvidia_smi_m10_list', smi_m10 == sorted(gpus), f'{len(smi_m10)} M10 in nvidia-smi')
    # The newest daemon topology.tsv must agree with the scan.
    tsvs = sorted(glob.glob(os.path.join(os.environ.get('M10DIS_RUNS', os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'runs')), '*/*/topology.tsv')), key=os.path.getmtime)
    if tsvs:
        r2 = subprocess.run([sys.executable, str(DISCOVER), '--json', '--topology', tsvs[-1]], capture_output=True, text=True)
        check('live_matches_newest_topology_tsv', r2.returncode == 0, tsvs[-1])
    # The shell helper used by the runners.
    sh = subprocess.run(['bash', '-c', f'source {HERE}/qwen38_pci_env.sh && q38_discover >/dev/null && '
                         'echo "$Q38_RTX_BDF|$(q38_outer_ports 0 5 | wc -w)|$(q38_inner_ports 0 5 | wc -w)"'],
                        capture_output=True, text=True)
    want = f"{j['rtx']['bdf']}|5|20"
    check('live_shell_helper', sh.returncode == 0 and sh.stdout.strip() == want, sh.stdout.strip() or sh.stderr.strip()[:120])

if __name__ == '__main__':
    synthetic()
    if '--synthetic-only' not in sys.argv:
        live()
    ok = all(r for _, r in results)
    print(f"TEST_PCI_DISCOVER,{'PASS' if ok else 'FAIL'},{sum(r for _, r in results)}/{len(results)}")
    sys.exit(0 if ok else 1)
