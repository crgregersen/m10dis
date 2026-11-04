#!/usr/bin/env python3
"""Bounded, snapshot-backed multicast configuration: one M10-local PEX8747 group, or (outer-*, 2026-11-04)
one outer PEX8748 group that copies an RTX write to the six cards' own multicast windows."""
from __future__ import annotations
import argparse, json, os, pathlib, re, struct, subprocess, sys

# Multicast extended capability (ID 0x0012). Its position is found per port by walking
# the PCIe extended capability list from 0x100 (never assumed); register offsets below
# are relative to the capability header.
MC_CAP_ID=0x0012
MC_CAP=0x4; CTRL=0x6; BASE=0x8; RECEIVE=0x10
BLOCK=0x18; BLOCK_UNTRANS=0x20; OVERLAY=0x28
ENABLE=1<<15; INDEX=16; WINDOW=1<<INDEX
SYS=pathlib.Path('/sys/bus/pci/devices')

def read(bdf,off,n):
 fd=os.open(SYS/bdf/'config',os.O_RDONLY)
 try: x=os.pread(fd,n,off)
 finally: os.close(fd)
 if len(x)!=n: raise RuntimeError(f'{bdf}: short config read at {off:#x}')
 return x
def write(bdf,off,data):
 fd=os.open(SYS/bdf/'config',os.O_WRONLY)
 try: n=os.pwrite(fd,data,off)
 finally: os.close(fd)
 if n!=len(data): raise RuntimeError(f'{bdf}: short config write at {off:#x}')
def r16(b,o): return struct.unpack('<H',read(b,o,2))[0]
def r32(b,o): return struct.unpack('<I',read(b,o,4))[0]
def r64(b,o): return struct.unpack('<Q',read(b,o,8))[0]
def w16(b,o,v): write(b,o,struct.pack('<H',v))
def w64(b,o,v): write(b,o,struct.pack('<II',v&0xffffffff,v>>32))
_mc_cache={}
def mc_cap(b):
 """Offset of the Multicast extended capability of port b; exactly one must exist."""
 if b in _mc_cache: return _mc_cache[b]
 found=[]; seen=set(); pos=0x100
 while pos and pos not in seen and 0x100<=pos<=0xffc:
  seen.add(pos); h=r32(b,pos)
  if h in (0,0xffffffff): break
  if h&0xffff==MC_CAP_ID: found.append(pos)
  pos=(h>>20)&0xffc
 if len(found)!=1: raise RuntimeError(f'{b}: multicast extended capability found {len(found)} times ({[hex(x) for x in found]})')
 _mc_cache[b]=found[0]; return found[0]
def mc(b,rel): return mc_cap(b)+rel
def regs(b,device=0x8747):
 ident=read(b,0,4)
 if struct.unpack('<HH',ident)!=(0x10b5,device): raise RuntimeError(f'{b}: not PEX{device:x}')
 if r32(b,mc(b,0))&0xffff!=MC_CAP_ID: raise RuntimeError(f'{b}: multicast capability header changed')
 return {'control':r16(b,mc(b,CTRL)),'base':r64(b,mc(b,BASE)),'receive':r64(b,mc(b,RECEIVE)),
         'block':r64(b,mc(b,BLOCK)),'block_untranslated':r64(b,mc(b,BLOCK_UNTRANS)),
         'overlay':r64(b,mc(b,OVERLAY))}
def window(b):
 s=subprocess.run(['lspci','-s',b,'-vv'],check=True,text=True,capture_output=True).stdout
 for line in s.splitlines():
  if 'Prefetchable memory behind bridge:' in line:
   m=re.search(r'([0-9a-fA-F]+)-([0-9a-fA-F]+)',line)
   if m:return int(m.group(1),16),int(m.group(2),16)
 raise RuntimeError(f'{b}: no prefetchable bridge window')
def collision_check(base):
 for p in SYS.iterdir():
  try:
   cls=int((p/'class').read_text(),16)
   if cls>>16==0x06: continue
   for i,line in enumerate((p/'resource').read_text().splitlines()):
    a,e,f=(int(x,16) for x in line.split())
    if a and e>=a and a<=base+WINDOW-1 and e>=base:
     raise RuntimeError(f'candidate overlaps endpoint {p.name} resource {i}: {a:#x}-{e:#x}')
  except FileNotFoundError: pass
def load_snapshot(d): return json.loads((d/'snapshot.json').read_text())
def restore(d):
 snap=load_snapshot(d); dev=snap.get('device',0x8747)
 regs_=lambda b: regs(b,dev)
 for b in snap['ports']: w16(b,mc(b,CTRL),snap['regs'][b]['control']&~ENABLE)
 for b in snap['ports']:
  v=snap['regs'][b]
  for off,key in ((BASE,'base'),(RECEIVE,'receive'),(BLOCK,'block'),
                  (BLOCK_UNTRANS,'block_untranslated'),(OVERLAY,'overlay')): w64(b,mc(b,off),v[key])
 for b in snap['ports']: w16(b,mc(b,CTRL),snap['regs'][b]['control'])
 for b in snap['ports']:
  now=regs_(b)
  if now!=snap['regs'][b]: raise RuntimeError(f'{b}: exact restore mismatch {now} != {snap["regs"][b]}')
 print('RESTORE,PASS')
# ---- outer PEX8748 group (2026-11-04) -------------------------------------------------------
# Ingress: the RTX's downstream port (base + enable, no receive). Egress: the six card ports (receive group 0,
# overlay = that card's own multicast window from topology.tsv, so the copy lands in the card's window and the
# card's PEX8747 copies it to its four GPUs). Window: the first 64 KiB of the prefetchable window of a downstream
# port with no device behind it (unused address space that the switch already routes); every other port's window
# and every endpoint resource must stay clear of it. Everything is discovered from sysfs.
def _children(b): return [p.name for p in pathlib.Path(os.path.realpath(SYS/b)).iterdir() if re.fullmatch(r'[0-9a-f]{4}:[0-9a-f]{2}:[0-9a-f]{2}\.[0-7]',p.name)]
def outer_discover(topology,rtx):
 rtx_port=pathlib.Path(os.path.realpath(SYS/rtx)).parent.name
 up=pathlib.Path(os.path.realpath(SYS/rtx_port)).parent.name
 rows=[l.split('\t') for l in pathlib.Path(topology).read_text().splitlines() if l]
 cards=[];bases=[]
 for r in rows:
  port=pathlib.Path(os.path.realpath(SYS/r[1])).parent.name
  if pathlib.Path(os.path.realpath(SYS/port)).parent.name!=up: raise RuntimeError(f'card {r[1]} is not below outer switch {up}')
  cards.append(port);bases.append(int(r[6],16))
 ds=[n for n in _children(up)]
 for b in [up]+ds:
  if struct.unpack('<HH',read(b,0,4))!=(0x10b5,0x8748): raise RuntimeError(f'{b}: not PEX8748')
 empty=[b for b in ds if b!=rtx_port and b not in cards and not _children(b)]
 if not empty: raise RuntimeError('no empty downstream port with a window for the outer multicast group')
 lo,hi=window(empty[0]); base=(lo+WINDOW-1)&~(WINDOW-1)
 if base+WINDOW-1>hi: raise RuntimeError(f'{empty[0]}: window too small')
 for b in ds:
  if b==empty[0]: continue
  try: wlo,whi=window(b)
  except RuntimeError: continue
  if wlo<=base+WINDOW-1 and whi>=base: raise RuntimeError(f'outer window overlaps {b} window')
 collision_check(base)
 return {'upstream':up,'rtx_port':rtx_port,'card_ports':cards,'card_mc_bases':bases,'empty_port':empty[0],'base':base}
def outer_capture(d,topology,rtx):
 info=outer_discover(topology,rtx);ports=[info['rtx_port'],*info['card_ports']]
 reg={b:regs(b,0x8748) for b in ports}
 if any(v['control']&ENABLE for v in reg.values()): raise RuntimeError('outer multicast was already enabled')
 d.mkdir(parents=True,exist_ok=False)
 for b in ports:(d/(b.replace(':','_')+'.config')).write_bytes(read(b,0,4096))
 (d/'snapshot.json').write_text(json.dumps({'device':0x8748,'ports':ports,'regs':reg,'outer':info},indent=2)+'\n')
 print('OUTER_CAPTURE,PASS,ports,',','.join(ports),'base,',hex(info['base']),'empty_port,',info['empty_port'])
def outer_apply(d):
 snap=load_snapshot(d);info=snap['outer'];ports=snap['ports'];base=info['base']
 if base%WINDOW: raise RuntimeError('outer base must be 64 KiB aligned')
 collision_check(base)
 planned={}
 for i,b in enumerate(ports):
  old=snap['regs'][b]
  if regs(b,0x8748)!=old: raise RuntimeError(f'{b}: changed since snapshot')
  if i==0: recv=overlay=0
  else:
   mb=info['card_mc_bases'][i-1]
   if mb%WINDOW: raise RuntimeError(f'card multicast base {mb:#x} not 64 KiB aligned')
   recv=1;overlay=mb|INDEX
  planned[b]={'control_disabled':old['control']&~ENABLE,'control_enabled':(old['control']&~0x803f)|ENABLE,
   'base':base|INDEX,'receive':recv,'overlay':overlay,'block':old['block'],'block_untranslated':old['block_untranslated']}
 (d/'plan.json').write_text(json.dumps({'base':base,'window':WINDOW,'ports':planned},indent=2)+'\n')
 changed=False
 try:
  for b in ports:w16(b,mc(b,CTRL),planned[b]['control_disabled'])
  changed=True
  for b in ports:
   v=planned[b]
   for off,key in ((BASE,'base'),(RECEIVE,'receive'),(OVERLAY,'overlay')):w64(b,mc(b,off),v[key])
  for b in ports:
   now=regs(b,0x8748);v=planned[b]
   if any(now[k]!=v[k] for k in ('base','receive','overlay','block','block_untranslated')) or now['control']!=v['control_disabled']:
    raise RuntimeError(f'{b}: disabled-state readback mismatch')
  for b in ports[1:]:w16(b,mc(b,CTRL),planned[b]['control_enabled'])   # egress first, then the RTX ingress
  w16(ports[0],mc(ports[0],CTRL),planned[ports[0]]['control_enabled'])
  for b in ports:
   if regs(b,0x8748)['control']!=planned[b]['control_enabled']: raise RuntimeError(f'{b}: enable readback mismatch')
  print('OUTER_APPLY,PASS,base,',hex(base))
 except Exception:
  if changed: restore(d)
  raise

def main():
 ap=argparse.ArgumentParser(); sub=ap.add_subparsers(dest='op',required=True)
 c=sub.add_parser('capture');c.add_argument('--snapshot',type=pathlib.Path,required=True);c.add_argument('--upstream',required=True);c.add_argument('--ports',nargs=4,required=True);c.add_argument('--gpus',nargs=4,required=True)
 a=sub.add_parser('apply');a.add_argument('--snapshot',type=pathlib.Path,required=True);a.add_argument('--base',type=lambda x:int(x,0),required=True);a.add_argument('--map-bus',nargs=4,type=lambda x:int(x,0),required=True)
 r=sub.add_parser('restore');r.add_argument('--snapshot',type=pathlib.Path,required=True)
 oc=sub.add_parser('outer-capture');oc.add_argument('--snapshot',type=pathlib.Path,required=True);oc.add_argument('--topology',required=True);oc.add_argument('--rtx',required=True)
 oa=sub.add_parser('outer-apply');oa.add_argument('--snapshot',type=pathlib.Path,required=True)
 od=sub.add_parser('outer-discover');od.add_argument('--topology',required=True);od.add_argument('--rtx',required=True)
 args=ap.parse_args()
 if os.geteuid()!=0: raise RuntimeError('must run as root')
 if args.op=='outer-discover': print(json.dumps({k:(hex(v) if isinstance(v,int) else v) for k,v in outer_discover(args.topology,args.rtx).items()})); return
 if args.op=='outer-capture': outer_capture(args.snapshot,args.topology,args.rtx); return
 if args.op=='outer-apply': outer_apply(args.snapshot); return
 if args.op=='capture':
  d=args.snapshot;d.mkdir(parents=True,exist_ok=False);ports=[args.upstream,*args.ports]
  reg={b:regs(b) for b in ports}
  if any(v['control']&ENABLE for v in reg.values()): raise RuntimeError('multicast was already enabled')
  for b in ports:(d/(b.replace(':','_')+'.config')).write_bytes(read(b,0,4096))
  (d/'snapshot.json').write_text(json.dumps({'upstream':args.upstream,'ports':ports,'gpus':args.gpus,'regs':reg},indent=2)+'\n')
  print('CAPTURE,PASS,ports,',','.join(ports))
 elif args.op=='restore': restore(args.snapshot)
 else:
  d=args.snapshot;snap=load_snapshot(d);base=args.base;ports=snap['ports'];up=snap['upstream']
  if base%WINDOW: raise RuntimeError('base must be 64 KiB aligned')
  lo,hi=window(up)
  if not lo<=base or base+WINDOW-1>hi: raise RuntimeError(f'base outside upstream window {lo:#x}-{hi:#x}')
  collision_check(base)
  maps=args.map_bus
  planned={}
  for i,b in enumerate(ports):
   old=snap['regs'][b]
   if regs(b)!=old: raise RuntimeError(f'{b}: changed since snapshot')
   if i==0: recv=overlay=0
   else:
    gpu=snap['gpus'][i-1]
    # BAR1 containment from current sysfs resource table.
    rows=(SYS/gpu/'resource').read_text().splitlines();bar=rows[1].split();barlo,barhi=int(bar[0],16),int(bar[1],16)
    if maps[i-1]%WINDOW or not barlo<=maps[i-1] or maps[i-1]+WINDOW-1>barhi:
     raise RuntimeError(f'{gpu}: RM BAR1 mapping not aligned/contained')
    recv=1;overlay=maps[i-1]|INDEX
   planned[b]={'control_disabled':old['control']&~ENABLE,
     'control_enabled':(old['control']&~0x803f)|ENABLE,
     'base':base|INDEX,'receive':recv,'overlay':overlay,
     'block':old['block'],'block_untranslated':old['block_untranslated']}
  (d/'plan.json').write_text(json.dumps({'base':base,'window':WINDOW,'ports':planned},indent=2)+'\n')
  changed=False
  try:
   for b in ports:w16(b,mc(b,CTRL),planned[b]['control_disabled'])
   changed=True
   for b in ports:
    v=planned[b]
    for off,key in ((BASE,'base'),(RECEIVE,'receive'),(OVERLAY,'overlay')):w64(b,mc(b,off),v[key])
   for b in ports:
    now=regs(b);v=planned[b]
    if now['control']!=v['control_disabled'] or now['base']!=v['base'] or now['receive']!=v['receive'] or now['overlay']!=v['overlay'] or now['block']!=v['block'] or now['block_untranslated']!=v['block_untranslated']:
     raise RuntimeError(f'{b}: disabled-state readback mismatch')
   for b in ports[1:]:w16(b,mc(b,CTRL),planned[b]['control_enabled'])
   w16(up,mc(up,CTRL),planned[up]['control_enabled'])
   for b in ports:
    if regs(b)['control']!=planned[b]['control_enabled']: raise RuntimeError(f'{b}: enable readback mismatch')
   print('APPLY,PASS,base,',hex(base),'window,',hex(lo),hex(hi))
  except Exception:
   if changed: restore(d)
   raise
if __name__=='__main__':
 try:main()
 except Exception as e:print('ERROR',e,file=sys.stderr);raise SystemExit(1)
