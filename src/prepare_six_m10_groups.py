#!/usr/bin/env python3
"""Freshly discover M10 islands and choose unused aligned 64 KiB ingress windows."""
import json, pathlib, subprocess, sys
from discover_m10_collective_topology import SYS_PCI, bridge_window, path_bdfs, pex_identity, resource_ranges

def align(x,a): return (x+a-1)&~(a-1)
def main():
 out=pathlib.Path(sys.argv[1]);out.parent.mkdir(parents=True,exist_ok=True)
 data=json.loads(subprocess.run([sys.executable,str(pathlib.Path(__file__).with_name('discover_m10_collective_topology.py'))],check=True,text=True,capture_output=True).stdout)
 if len(data['groups'])!=6 or any(g['gpu_count']!=4 for g in data['groups']): raise SystemExit('expected six complete four-GM107 M10 groups')
 groups=[]
 for g in data['groups']:
  up=g['local_pex_upstream_bdf']; win=g['local_prefetchable_window']
  if not win or not up: raise SystemExit(f'{g["serial"]}: no local prefetchable bridge window')
  ports=[g['gpu_downstream_ports'][d['bdf']] for d in g['gpus']]
  if any(not p or pex_identity(p)!=(0x10b5,0x8747) for p in ports) or pex_identity(up)!=(0x10b5,0x8747): raise SystemExit(f'{g["serial"]}: incomplete PEX8747 group')
  endpoints=[]
  for p in SYS_PCI.iterdir():
   try:
    cls=int((p/'class').read_text(),16)
    if cls>>16==0x06: continue
    for r in resource_ranges(p.name):
     if r['start'] and r['end']>=r['start']: endpoints.append((r['start'],r['end'],p.name,r['bar_index']))
   except (OSError,ValueError): pass
  start=win['start']; end=win['end']; candidate=align(start,0x10000)
  while candidate+0xffff<=end:
   overlap=next((r for r in sorted(endpoints) if r[0]<=candidate+0xffff and r[1]>=candidate),None)
   if overlap is None: break
   candidate=align(overlap[1]+1,0x10000)
  if candidate+0xffff>end: raise SystemExit(f'{g["serial"]}: no 64 KiB gap in local window')
  groups.append({'serial':g['serial'],'upstream':up,'gpus':[d['bdf'] for d in g['gpus']],
    'uuids':[d['uuid'] for d in g['gpus']],'ports':ports,'window':win,'mc_base':candidate,
    'mc_bytes':0x10000,'gpu_resources':g['gpu_resources']})
 data['groups']=groups
 out.write_text(json.dumps(data,indent=2)+'\n')
 with out.with_suffix('.tsv').open('w') as f:
  for g in groups:f.write('\t'.join([g['serial'],g['upstream'],','.join(g['ports']),','.join(g['gpus']),hex(g['window']['start']),hex(g['window']['end']),hex(g['mc_base'])])+'\n')
 for g in groups:print(g['serial'],g['upstream'],'gpus='+','.join(g['gpus']),'candidate='+hex(g['mc_base']),'window='+hex(g['window']['start'])+'-'+hex(g['window']['end']))
 print('TOPOLOGY_JSON',out)
if __name__=='__main__':main()
