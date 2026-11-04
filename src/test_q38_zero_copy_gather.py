#!/usr/bin/env python3
"""Offline test of the zero-copy embedding gather (Q38_EMBEDDING_ZERO_COPY=1) on the RTX.

Builds a pageable host table like CpuTable (BF16, rows x 2560), pins it through the peer library's
q38_zero_copy_register, and checks q38_zero_copy_gather against torch.index_select:
eager, under CUDA-graph capture + replay with new ids, and with out-of-range ids (zero rows, counted).
Usage: test_q38_zero_copy_gather.py LIBRARY   (run with only the RTX visible). Prints ZERO_COPY_GATHER_TEST,PASS|FAIL.
"""
import ctypes, os, sys
import torch

os.environ["Q38_M10_RTX_PEER_LIBRARY"] = sys.argv[1]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import qwen38_cpu_head_hooks as h

fails = 0
def check(ok, name, detail=""):
    global fails
    print(f"CHECK,{name},{'OK' if ok else 'FAIL'}{',' + detail if detail else ''}", flush=True)
    fails += 0 if ok else 1

os.environ["Q38_EMBEDDING_ZERO_COPY"] = "1"
dev = torch.device("cuda", 0)
rows, dim = 248320, 2560
emb = h.CpuTable(rows, dim, params_dtype=torch.bfloat16, role="embedding")
g = torch.Generator().manual_seed(38)
emb.load_weights([("weight", torch.randn(rows, dim, generator=g, dtype=torch.float32).to(torch.bfloat16))])

def ref(ids):
    return emb.table.index_select(0, ids.cpu().clamp(0, rows - 1)).to(dev) * (((ids >= 0) & (ids < rows)).cpu().view(-1, 1)).to(dev, torch.bfloat16)

# eager: decode (1 id) and a prefill chunk (8 ids incl. first/last row)
for name, ids in (("eager_1", torch.tensor([[12345]], device=dev)),
                  ("eager_8", torch.tensor([[0, 1, 77, 151643, 248319, 9, 9, 200000]], device=dev))):
    out = emb(ids); torch.cuda.synchronize()
    check(out.shape == (*ids.shape, dim) and torch.equal(out.view(-1, dim), ref(ids.view(-1))), name, str(tuple(out.shape)))

# graph: capture with one id buffer, replay with new ids written into it
static_ids = torch.tensor([[5]], device=dev)
s = torch.cuda.Stream(); s.wait_stream(torch.cuda.current_stream())
with torch.cuda.stream(s):
    emb(static_ids)                                   # warm-up outside capture
torch.cuda.current_stream().wait_stream(s)
graph = torch.cuda.CUDAGraph()
with torch.cuda.graph(graph):
    static_out = emb(static_ids)
for k, v in enumerate((3, 248000, 151645, 42)):
    static_ids.fill_(v); graph.replay(); torch.cuda.synchronize()
    check(torch.equal(static_out.view(-1, dim), ref(static_ids.view(-1))), f"graph_replay_{k}", f"id,{v}")

# out-of-range ids: zero rows, counted
bad_ids = torch.tensor([[-1, rows, 7]], device=dev)
out = emb(bad_ids); torch.cuda.synchronize()
z = emb._q38_zero_copy
check(torch.count_nonzero(out[0, 0]).item() == 0 and torch.count_nonzero(out[0, 1]).item() == 0
      and torch.equal(out[0, 2], ref(torch.tensor([7], device=dev))[0]) and int(z["bad"].item()) == 2,
      "out_of_range_zero_and_counted", f"bad,{int(z['bad'].item())},read_only,{z['read_only']}")

h._zero_copy_close()
check(z.get("unregistered") is True, "unregister")
print(f"ZERO_COPY_GATHER_TEST,{'FAIL' if fails else 'PASS'},failures,{fails}", flush=True)
sys.exit(1 if fails else 0)
