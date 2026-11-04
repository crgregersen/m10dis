#!/usr/bin/env python3
"""Hard-link completed Stage 7 layer bundles before the runner removes staging."""
from __future__ import annotations
import argparse, json, os, time
from pathlib import Path

def preserve(src: Path, dst: Path) -> int:
    dst.mkdir(parents=True,exist_ok=True); n=0
    for root,dirs,files in os.walk(src):
        rel=Path(root).relative_to(src); target=dst/rel; target.mkdir(parents=True,exist_ok=True)
        for name in files:
            a=Path(root)/name;b=target/name
            if b.exists():continue
            try:os.link(a,b)
            except FileExistsError:pass
            n+=1
    return n

def main():
    ap=argparse.ArgumentParser();ap.add_argument('pid',type=int);ap.add_argument('stage_root',type=Path);ap.add_argument('archive',type=Path);ap.add_argument('state',type=Path);ap.add_argument('--interval',type=float,default=.1);a=ap.parse_args()
    source=a.stage_root.resolve()/'staging';archive=a.archive.resolve();archive.mkdir(parents=True,exist_ok=True);done={}
    while True:
        for layer in range(48):
            tag=f'{layer:02d}';r=source/f'route_{tag}';s=source/f'shared_{tag}'
            if layer in done:continue
            if (r/'resident_layer_manifest.json').is_file() and (s/'shared_manifest.json').is_file():
                nr=preserve(r,archive/f'layer_{tag}'/'routed');ns=preserve(s,archive/f'layer_{tag}'/'shared')
                done[layer]={'routed_files_hardlinked':nr,'shared_files_hardlinked':ns,
                    'routed_manifest':str(archive/f'layer_{tag}'/'routed'/'resident_layer_manifest.json'),
                    'shared_manifest':str(archive/f'layer_{tag}'/'shared'/'shared_manifest.json')}
        state={'status':'RUNNING','stage7_pid':a.pid,'preserved_layers':sorted(done),'preserved_count':len(done),'updated_unix':time.time(),'layers':done}
        tmp=a.state.with_suffix(a.state.suffix+'.tmp');tmp.parent.mkdir(parents=True,exist_ok=True);tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(a.state)
        try:os.kill(a.pid,0)
        except ProcessLookupError:
            state['status']='STAGE7_EXITED';state['updated_unix']=time.time();tmp=a.state.with_suffix(a.state.suffix+'.tmp');tmp.write_text(json.dumps(state,indent=2)+'\n');tmp.replace(a.state);break
        time.sleep(a.interval)

if __name__=='__main__':main()
