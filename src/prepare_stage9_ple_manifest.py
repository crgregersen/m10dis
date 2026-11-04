#!/usr/bin/env python3
"""Map checkpoint PLE safetensors shards to a memory-balanced 24-GM107 plan."""
import json, pathlib, struct, subprocess, sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parent))
import qwen38_m10_expert_sim as q

model=pathlib.Path(sys.argv[1]).resolve(); out=pathlib.Path(sys.argv[2]).resolve(); out.mkdir(parents=True,exist_ok=True)
index=json.loads((model/'model.safetensors.index.json').read_text())['weight_map']
routed,shared,cards=q.choose_gpu_groups()
gpu_rows=[]
for row in subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,name,pci.bus_id','--format=csv,noheader'],text=True).splitlines():
 i,u,n,b=[x.strip() for x in row.split(',',3)];gpu_rows.append({'index':int(i),'uuid':u,'name':n,'bdf':b})
ordinal={g['uuid']:g['index'] for g in gpu_rows}
route_ord=[ordinal[u] for u in routed];shared_ord=[ordinal[u] for u in shared]
if len(route_ord)!=20 or len(shared_ord)!=4:raise RuntimeError('expected 20 routed and 4 shared GM107s')
shards=[];per_gpu={i:[] for i in range(25)}
for s in range(128):
 key=f'model.language_model.layers.1.ple.ple_embedding.ngram_embedding.shard_{s}.weight'
 fn=index[key]; p=model/fn
 with p.open('rb') as f:
  hlen=struct.unpack('<Q',f.read(8))[0]; header=json.loads(f.read(hlen))
 meta=header[key]; start,end=meta['data_offsets']; shape=meta['shape']
 if meta['dtype']!='F8_E4M3' or shape != [2_500_012,160] or end-start != 2_500_012*160:
  raise RuntimeError(f'unexpected PLE shard metadata {s}: {meta}')
 # GPU 0 exposes 280 MiB less CUDA-visible VRAM than the other GM107s.
 # Move one PLE shard from its nominal three-shard share to the first shared
 # worker, whose measured headroom is substantially larger.
 if s<60 and s != 2:
  route_slot = s//3 if s < 2 else 1 + (s-3)//3
  dev=route_ord[route_slot]
  reserve_mib=6157
 else:
  dev=shared_ord[0] if s == 2 else shared_ord[(s-60)//17]
  reserve_mib=176
 rec={'shard':s,'key':key,'path':str(p),'offset':8+hlen+start,'bytes':end-start,'rows':shape[0],'width':shape[1],'gpu_index':dev}
 shards.append(rec);per_gpu[dev].append(rec)
gpu_alloc=[]
for g in gpu_rows:
 i=g['index']; assigned=per_gpu[i]
 reserve=(6157 if g['uuid'] in routed else 176 if g['uuid'] in shared else 0)*1024*1024
 ple=sum(x['bytes'] for x in assigned); scratch=64*1024*1024 if assigned else 0
 gpu_alloc.append({**g,'role':'routed' if g['uuid'] in routed else 'shared' if g['uuid'] in shared else 'unused_rtx',
   'ple_shards':[x['shard'] for x in assigned], 'ple_bytes':ple,
   'expert_reserve_bytes':reserve,'scratch_reserve_bytes':scratch,
   'total_request_bytes':ple+reserve+scratch})
(out/'ple_shards.tsv').write_text(''.join(f"{x['shard']}\t{x['gpu_index']}\t{x['path']}\t{x['offset']}\t{x['bytes']}\t{x['rows']}\t{x['width']}\n" for x in shards))
(out/'gpu_allocations.tsv').write_text(''.join(f"{x['index']}\t{x['bdf']}\t{x['uuid']}\t{x['expert_reserve_bytes']}\t{x['ple_bytes']}\t{x['scratch_reserve_bytes']}\n" for x in gpu_alloc))
plan={'status':'PREPARED','model':str(model),'model_index_sha256':__import__('hashlib').sha256((model/'model.safetensors.index.json').read_bytes()).hexdigest(),
 'model_config_ple_layer_ids':json.loads((model/'config.json').read_text())['text_config']['ple_layer_ids'],
 'table_shard_count':len(shards),'per_shard_bytes':shards[0]['bytes'],'total_shard_bytes':sum(x['bytes'] for x in shards),
 'routed_gpu_uuids':routed,'shared_gpu_uuids':shared,'card_groups':cards,'gpu_allocations':gpu_alloc,
 'shards':shards,'notes':'PLE placement is headroom-adjusted: the first routed GM107 has two shards because its CUDA-visible VRAM is 280 MiB smaller; that shard is assigned to the first shared GM107, which holds 18 shards. The other 19 routed GM107s hold three shards each and the other three shared GM107s hold 17 each. Reserve bytes model the measured Stage7 expert/shared residency footprint, and 64 MiB per active GM107 is reserved for query work.'}
(out/'ple_placement_manifest.json').write_text(json.dumps(plan,indent=2)+'\n')
print('PLE_MANIFEST_READY',len(shards),'bytes',plan['total_shard_bytes'],'route_gpus',len(route_ord),'shared_gpus',len(shared_ord))
