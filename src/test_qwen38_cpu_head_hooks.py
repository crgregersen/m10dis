"""CPU-only check of qwen38_cpu_head_hooks with the real checkpoint tensors.

1. AutoWeightsLoader delivers model.embed_tokens.weight / lm_head.weight into CpuTable via
   load_weights (the path vLLM uses), and the loaded tables equal the checkpoint bytes.
2. Embedding forward is an exact row copy.
3. LM head through vLLM's LogitsProcessor matches a float64 reference within a BF16 bound,
   and the argmax agrees (negative control: a perturbed table must break agreement).
Run: CUDA_VISIBLE_DEVICES= python test_qwen38_cpu_head_hooks.py MODEL_DIR
"""
import json, os, sys, time
from pathlib import Path
import torch
from torch import nn
from safetensors import safe_open

assert os.environ.get("CUDA_VISIBLE_DEVICES", None) == "", "run with CUDA_VISIBLE_DEVICES= (CPU only)"
sys.path.insert(0, str(Path(__file__).parent))
from qwen38_cpu_head_hooks import CpuTable
from vllm.model_executor.models.utils import AutoWeightsLoader
from vllm.model_executor.layers.logits_processor import LogitsProcessor

model = Path(sys.argv[1])
idx = json.loads((model / "model.safetensors.index.json").read_text())["weight_map"]
def tensor(key):
    with safe_open(model / idx[key], "pt") as f:
        return f.get_tensor(key).clone()
E_KEY, H_KEY = "model.language_model.embed_tokens.weight", "lm_head.weight"
emb, head = tensor(E_KEY), tensor(H_KEY)
V, D = emb.shape
print(f"CHECKPOINT,embedding,{tuple(emb.shape)},{emb.dtype},lm_head,{tuple(head.shape)},{head.dtype}")

class Inner(nn.Module):
    def __init__(self):
        super().__init__(); self.embed_tokens = CpuTable(V, D, role="embedding", params_dtype=torch.bfloat16)
class Outer(nn.Module):
    def __init__(self):
        super().__init__(); self.model = Inner(); self.lm_head = CpuTable(V, D, role="lm_head", params_dtype=torch.bfloat16)
m = Outer()
loaded = AutoWeightsLoader(m).load_weights([("model.embed_tokens.weight", emb), ("lm_head.weight", head)])
checks = {}
checks["loader_reports_both"] = {"model.embed_tokens.weight", "lm_head.weight"} <= set(loaded)
checks["no_parameters"] = len(list(m.parameters())) == 0
checks["embedding_table_exact"] = torch.equal(m.model.embed_tokens.table, emb)
checks["lm_head_table_exact"] = torch.equal(m.lm_head.table, head)

g = torch.Generator().manual_seed(0)
ids = torch.randint(0, V, (64,), generator=g)
checks["embedding_rows_exact"] = torch.equal(m.model.embed_tokens(ids), emb[ids])

from vllm.config import VllmConfig, set_current_vllm_config
try:
    with set_current_vllm_config(VllmConfig()):
        lp = LogitsProcessor(V)
except Exception as exc:  # fall back to a minimal processor with the same head call path
    print(f"NOTE,LogitsProcessor needs a model config ({type(exc).__name__}); using _apply_head path directly")
    lp = None
if lp is None:
    def lp(head, h):
        return head.quant_method.apply(head, h, bias=None)
worst, agree, n = 0.0, 0, 8
for i in range(n):
    h = (torch.randn(1, D, generator=g) * 0.5).to(torch.bfloat16)
    t = time.perf_counter(); got = lp(m.lm_head, h); ms = (time.perf_counter() - t) * 1e3
    ref = h.double() @ head.double().t()
    bound = (h.double().abs() @ head.double().abs().t()) * 2 ** -7 + 1e-6   # BF16 output rounding + accumulation
    worst = max(worst, float(((got.double() - ref).abs() / bound).max()))
    agree += int(got.argmax() == ref.argmax())
    print(f"LM_HEAD,sample,{i},ms,{ms:.1f},argmax_got,{int(got.argmax())},argmax_ref,{int(ref.argmax())}")
checks["lm_head_within_bf16_bound"] = worst <= 1.0
checks["lm_head_argmax_agree"] = agree == n
# negative control: perturb the winning row heavily; the reference argmax must no longer match the original.
h = (torch.randn(1, D, generator=g) * 0.5).to(torch.bfloat16)
win = int(lp(m.lm_head, h).argmax())
m.lm_head.table[win] *= -1
checks["negative_control_detects_change"] = int(lp(m.lm_head, h).argmax()) != win
for k, v in checks.items(): print("CHECK", k, "OK" if v else "FAIL")
print(f"WORST_ERROR_OVER_BOUND,{worst:.3f}")
print("CPU_HEAD_HOOK_TEST," + ("PASS" if all(checks.values()) else "FAIL"))
sys.exit(0 if all(checks.values()) else 1)
