#!/usr/bin/env python3
"""Run Qwen4-Exp vLLM on the RTX with all experts and PLE resident on M10s.

The experimental bridge patches the installed vLLM model in-process. Routed
and shared expert parameters are omitted from the RTX load and delegated to the
resident M10 expert service. The large PLE table is omitted from the RTX load;
the n-gram embedding module requests its FP8 rows from the M10 PLE service.
Only the RTX is visible to vLLM. M10 child services receive explicit device
masks of their own.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import importlib
import json
import os
import re
import socket
import struct
import subprocess
import sys
import threading
import time
import traceback
from pathlib import Path
from typing import Any

def discover_rtx_uuid() -> str:
    """RTX UUID from Q38_RTX_UUID or a live sysfs scan (qwen38_pci_discover.py); never hardcoded."""
    env = os.environ.get("Q38_RTX_UUID")
    if env:
        return env
    out = subprocess.run([sys.executable, str(Path(__file__).with_name("qwen38_pci_discover.py")), "--json"],
                         check=True, text=True, capture_output=True).stdout
    return json.loads(out)["rtx"]["uuid"]


def recv_exact(sock: socket.socket, n: int) -> bytes:
    chunks = bytearray()
    while len(chunks) < n:
        data = sock.recv(n - len(chunks))
        if not data:
            raise RuntimeError(f"PLE socket closed after {len(chunks)}/{n} bytes")
        chunks.extend(data)
    return bytes(chunks)


class PleSocketClient:
    def __init__(self, path: Path):
        self.path = path
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(180)
        self.sock.connect(str(path))
        self.lock = threading.Lock()
        self.requests = 0
        self.rows = 0

    def lookup(self, global_rows: list[int]) -> bytes:
        with self.lock:
            self.sock.sendall(struct.pack("<I", len(global_rows)))
            if global_rows:
                self.sock.sendall(struct.pack("<" + "Q" * len(global_rows), *global_rows))
            status, count = struct.unpack("<II", recv_exact(self.sock, 8))
            if status != 0 or count != len(global_rows):
                raise RuntimeError(
                    f"PLE lookup rejected: status={status}, count={count}, requested={len(global_rows)}"
                )
            payload = recv_exact(self.sock, count * 160)
            self.requests += 1
            self.rows += count
            return payload

    def close(self, stop: bool = False) -> None:
        try:
            if stop:
                with self.lock:
                    self.sock.sendall(struct.pack("<I", 0xFFFFFFFF))
        finally:
            self.sock.close()


def install_model_hooks(adapter: Any, ple_socket: Path | None) -> dict[str, Any]:
    import numpy as np
    import torch
    import torch.nn.functional as F
    from torch import nn
    from vllm.distributed import get_ep_group
    from vllm.model_executor.layers.fused_moe import GateLinear
    from vllm.model_executor.layers.fused_moe.router.fused_topk_router import (
        FusedTopKRouter,
    )

    model_mod = importlib.import_module("vllm.models.qwen4_exp.nvidia.model")
    ngram_mod = importlib.import_module("vllm.models.qwen4_exp.nvidia.ngram_embedding")
    common_ngram = importlib.import_module("vllm.models.qwen4_exp.common.ngram_embedding")
    # Skip, BEFORE reading them from disk, the checkpoint tensors the RTX never uses: experts and the
    # shared expert (resident on the M10s), PLE n-gram shards (M10 PLE owner), MTP head and vision
    # tower (not built). Without this vLLM reads ~172 GB to keep ~7 GB (2026-11-03: 183 s load).
    # Disable with Q38_SKIP_NON_RTX_TENSORS=0.
    if os.environ.get("Q38_SKIP_NON_RTX_TENSORS", "1") == "1":
        from vllm.model_executor.model_loader import weight_utils as _wu
        if not getattr(_wu, "_q38_skip_installed", False):
            _orig_skip = _wu.should_skip_weight
            _non_rtx = (".mlp.experts.", ".mlp.shared_expert.", ".mlp.shared_expert_gate.",
                        "ngram_embedding.shard_", "model.visual.")
            # MTP speculative decoding (Q38_MTP_SPEC_TOKENS > 0, 2026-11-04): the drafter's attention, norms,
            # fc and hyper-connection tensors load on the RTX; its experts / shared expert are on the M10s
            # (resident layer 48, skipped by the patterns above). Otherwise the whole MTP head is skipped.
            if int(os.environ.get("Q38_MTP_SPEC_TOKENS", "0") or 0) <= 0:
                _non_rtx = _non_rtx + ("mtp.",)
            if os.environ.get("Q38_M10_LM_HEAD") == "1":
                _non_rtx = _non_rtx + ("lm_head.weight",)      # held by the M10s (PLE owner slices)
            _skip_counts = {"skipped": 0, "kept": 0}

            def _q38_should_skip(name, *a, **k):
                extra = getattr(_wu, "_q38_extra_skip", None)   # e.g. BF16 originals replaced by the FP8 cache
                if any(t in name for t in _non_rtx) or (extra is not None and extra(name)):
                    _skip_counts["skipped"] += 1
                    return True
                _skip_counts["kept"] += 1
                return _orig_skip(name, *a, **k)

            _wu.should_skip_weight = _q38_should_skip
            _wu._q38_skip_installed = True
            _wu._q38_skip_counts = _skip_counts
    # Opt-in: embedding / LM head in host RAM (Q38_CPU_EMBEDDING=1, Q38_CPU_LM_HEAD=1).
    from qwen38_cpu_head_hooks import install as install_cpu_heads
    cpu_heads = install_cpu_heads(model_mod)
    # MTP drafter (vllm.models.qwen4_exp.nvidia.mtp): its own embedding / LM head are replaced by the target's
    # after loading (vLLM shares them); build them with the same host-table / M10 stand-ins, so the throwaway
    # copies never allocate 2 x 1.27 GB on the RTX. Its MoE block is Qwen4ExpSparseMoeBlock (patched below),
    # i.e. the M10 exchange with layer id 48.
    if int(os.environ.get("Q38_MTP_SPEC_TOKENS", "0") or 0) > 0:
        mtp_mod = importlib.import_module("vllm.models.qwen4_exp.nvidia.mtp")
        mtp_mod.VocabParallelEmbedding = model_mod.VocabParallelEmbedding
        mtp_mod.ParallelLMHead = model_mod.ParallelLMHead
        cpu_heads["mtp_heads"] = "shared stand-ins"

    # Q38_VLLM_CUDAGRAPH=1: the host-side MoE exchange and PLE lookup become eager breaks of
    # vLLM's breakable CUDA graphs (q38_cudagraph_breaks); everything around them is captured.
    from q38_cudagraph_breaks import make_break, install_full_breakable
    if install_full_breakable():
        print("Q38_VLLM_CUDAGRAPH_FULL,breakable_full_wrapper_installed", file=sys.stderr, flush=True)
    _dummy_skips = {"count": 0}
    _gpu_driven = os.environ.get("Q38_RTX_PEER_GPU_DRIVEN") == "1"
    # Q38_RTX_PEER_PLE_GPU=1: PLE rows come from the PCI-cluster row gather (one broadcast exchange,
    # worker mode 6) inside the CUDA graph; no PLE eager break, no CPU lookup, no sequence resync.
    _ple_gpu = _gpu_driven and os.environ.get("Q38_RTX_PEER_PLE_GPU") == "1"
    # Per-step host hooks (sticky-error check, progress) run in the CPU-embedding eager break, or -
    # when the embedding is a zero-copy GPU gather and no break is left - in compute_logits, which the
    # runner calls eagerly after every graph replay.
    from qwen38_cpu_head_hooks import STEP_HOOKS, zero_copy_enabled
    if zero_copy_enabled():
        if os.environ.get("Q38_CPU_EMBEDDING") != "1":
            raise RuntimeError("Q38_EMBEDDING_ZERO_COPY=1 needs Q38_CPU_EMBEDDING=1 (host table)")
        lm_cls = model_mod.Qwen4ExpForCausalLM
        if not getattr(lm_cls.compute_logits, "_q38_step_hooks", False):
            _orig_compute_logits = lm_cls.compute_logits

            def compute_logits(self, hidden_states):
                for hook in STEP_HOOKS:
                    hook()
                try:   # one call per scheduled step: decode windows are exact, prefill chunks count as steps
                    from qwen38_vllm_rtx_peer_adapter import _progress_note
                    _progress_note(1)
                except ImportError:
                    pass
                return _orig_compute_logits(self, hidden_states)

            compute_logits._q38_step_hooks = True
            lm_cls.compute_logits = compute_logits
    if os.environ.get("Q38_M10_LM_HEAD") == "1":
        if not _gpu_driven:
            raise RuntimeError("Q38_M10_LM_HEAD=1 needs Q38_RTX_PEER_GPU_DRIVEN=1")
        import qwen38_cpu_head_hooks as _heads
        _heads.LM_HEAD_FN = lambda x: adapter.lm_head_logits(x)
    if _ple_gpu:
        # Without the PLE break, surface GPU-driven exchange errors once per step (STEP_HOOKS above).
        if os.environ.get("Q38_CPU_EMBEDDING") != "1":
            raise RuntimeError("Q38_RTX_PEER_PLE_GPU=1 needs Q38_CPU_EMBEDDING=1 (its break or compute_logits runs the per-step error check)")
        STEP_HOOKS.append(lambda: adapter.check_error())

    def _moe_exchange(layer_id, h, ids, w):
        # vLLM's CUDA-graph warm-up runs dummy forwards whose router rows carry expert id -1.
        # Real tokens always have ids in [0, num_experts); never send anything else to the M10s
        # (2026-11-03 task 49: the M10 range check rejected id -1 and the routed workers exited).
        n_exp = 512
        if bool(((ids < 0) | (ids >= n_exp)).any().item()):
            _dummy_skips["count"] += 1
            if _dummy_skips["count"] <= 3 or _dummy_skips["count"] % 1000 == 0:
                print(f"Q38_MOE_DUMMY_SKIP,layer,{layer_id},count,{_dummy_skips['count']},"
                      f"ids_min,{int(ids.min())},ids_max,{int(ids.max())}", file=sys.stderr, flush=True)
            return torch.zeros_like(h)
        return adapter.forward(layer_id, h, ids, w)

    _moe_break = make_break(_moe_exchange, "moe",
                            lambda layer_id, h, ids, w: (tuple(h.shape), h.dtype, h.device))
    # PLE rows come back as raw FP8 [*ngram_ids.shape, embedding_dim] (see _q38_forward_impl).
    _ple_break = make_break(lambda key, module, ngram_ids: module._q38_forward_impl(ngram_ids), "ple",
                            lambda key, module, ngram_ids: ((*ngram_ids.shape, module.embedding_dim),
                                                            torch.float8_e4m3fn, ngram_ids.device))

    class RemoteMoEProxy:
        def __init__(self, gate: nn.Module, layer_id: int, config: Any):
            self.gate = gate
            self.layer_id = layer_id
            self.adapter = adapter
            self.router = FusedTopKRouter(
                top_k=int(config.num_experts_per_tok),
                global_num_experts=int(config.num_experts),
                scoring_func="softmax",
                renormalize=bool(getattr(config, "norm_topk_prob", True)),
            )

        def __call__(self, *, hidden_states: torch.Tensor, router_logits: torch.Tensor) -> torch.Tensor:
            logits = F.linear(hidden_states, self.gate.weight)
            weights, ids = self.router.select_experts(
                hidden_states=hidden_states,
                router_logits=logits,
                topk_indices_dtype=torch.int32,
            )
            if _gpu_driven:
                # Phase 2: enqueue-only exchange captured inside the CUDA graph (no eager break);
                # dummy rows are skipped on the GPU, errors surface at the next PLE break.
                return self.adapter.forward(self.layer_id, hidden_states, ids, weights)
            return _moe_break(self.layer_id, hidden_states, ids, weights)

        def update_expert_map(self) -> None:
            return None

    moe_cls = model_mod.Qwen4ExpSparseMoeBlock

    def remote_moe_init(self, vllm_config, prefix: str = "", reduce_results: bool = True):
        del reduce_results
        nn.Module.__init__(self)
        config = vllm_config.model_config.hf_text_config
        parallel = vllm_config.parallel_config
        eplb = parallel.eplb_config
        self.tp_size = 1
        self.ep_group = get_ep_group().device_group
        self.ep_size = int(self.ep_group.size())
        self.n_routed_experts = int(config.num_experts)
        self.n_logical_experts = self.n_routed_experts
        self.n_redundant_experts = int(eplb.num_redundant_experts)
        self.n_physical_experts = self.n_logical_experts + self.n_redundant_experts
        self.n_local_physical_experts = self.n_physical_experts // self.ep_size
        self.enable_eplb = bool(parallel.enable_eplb)
        self.is_sequence_parallel = False
        self.is_fused_shared_expert_enabled = False
        self.replicate_shared_expert = False
        self.shared_expert = None
        self.shared_expert_gate = None
        self.n_shared_experts = int(config.shared_expert_intermediate_size > 0)
        self.gate = GateLinear(
            int(config.hidden_size), int(config.num_experts),
            prefix=f"{prefix}.gate",
        )
        match = re.search(r"\.layers\.(\d+)\.mlp$", prefix)
        if match is None:
            match = re.search(r"layers\.(\d+)", prefix)
        if match is None:
            raise ValueError(f"cannot derive layer index from MoE prefix {prefix!r}")
        self.experts = RemoteMoEProxy(self.gate, int(match.group(1)), config)

    moe_cls.__init__ = remote_moe_init

    model_class = model_mod.Qwen4ExpModel
    original_model_load = model_class.load_weights

    def filtered_model_load(self, weights):
        skipped = 0

        def keep():
            nonlocal skipped
            for name, tensor in weights:
                if any(token in name for token in (
                    ".mlp.experts.",
                    ".mlp.shared_expert.",
                    ".mlp.shared_expert_gate.",
                )):
                    skipped += 1
                    continue
                yield name, tensor

        loaded = original_model_load(self, keep())
        self._m10_skipped_expert_checkpoint_tensors = (
            getattr(self, "_m10_skipped_expert_checkpoint_tensors", 0) + skipped
        )
        return loaded

    model_class.load_weights = filtered_model_load

    ple_base = common_ngram.Qwen4ExpPLEEmbedding
    direct_ple = bool(getattr(adapter, "supports_direct_ple", False))
    if not direct_ple and ple_socket is None:
        raise ValueError("PLE requires either the direct BAR1 adapter or a PLE socket")

    class RemotePleEmbedding(ple_base):
        supports_prefetch = False
        _client: Any = None

        def allocate_embedding_weight(self, num_embeddings: int, embedding_dim: int, dtype: torch.dtype) -> torch.Tensor:
            # The parameter exists for vLLM's FP8 scale and loader metadata;
            # the full table stays resident in the M10 service.
            del num_embeddings
            return torch.empty((1, embedding_dim), dtype=dtype)

        def start_prefetch(self, ngram_ids: torch.Tensor) -> None:
            del ngram_ids

        def finalize_prefetch(self, total_tokens: int) -> torch.Tensor:
            raise RuntimeError("remote PLE does not use asynchronous prefetch")

        def forward(self, ngram_ids: torch.Tensor) -> torch.Tensor:
            if _ple_gpu:
                return self._q38_forward_gpu(ngram_ids)
            return _ple_break(id(self), self, ngram_ids)

        def _q38_forward_gpu(self, ngram_ids: torch.Tensor) -> torch.Tensor:
            if not direct_ple:
                raise RuntimeError("Q38_RTX_PEER_PLE_GPU=1 needs the direct RTX peer adapter")
            slot_size, slot_offset = self._get_dp_gather_slot(ngram_ids.shape[0])
            gathered = self._gather_dp_ids(ngram_ids, slot_size)
            raw = adapter.ple_gather_gpu(gathered)             # uint8 [n, 160], rows in gathered order
            rows = raw.view(torch.float8_e4m3fn).reshape(*gathered.shape, self.embedding_dim)
            return self._select_embeddings(rows, ngram_ids.shape[0], slot_offset)

        def _q38_forward_impl(self, ngram_ids: torch.Tensor) -> torch.Tensor:
            if RemotePleEmbedding._client is None:
                RemotePleEmbedding._client = (adapter if direct_ple else PleSocketClient(ple_socket))
            slot_size, slot_offset = self._get_dp_gather_slot(ngram_ids.shape[0])
            gathered = self._gather_dp_ids(ngram_ids, slot_size)
            flat_ids = gathered.detach().to(device="cpu", dtype=torch.int64).contiguous().numpy().reshape(-1)
            ids = [int(x) for x in flat_ids]
            # DirectResidentM10MoEAdapter exposes the explicit lookup_ple API;
            # the legacy socket client exposes lookup. Keep the transport
            # boundary explicit so direct mode does not silently fall through
            # to the old socket contract.
            if direct_ple and hasattr(RemotePleEmbedding._client, "check_error"):
                RemotePleEmbedding._client.check_error()    # surface GPU-driven MoE errors each layer
            payload = (RemotePleEmbedding._client.lookup_ple(ids) if direct_ple
                       else RemotePleEmbedding._client.lookup(ids))
            if direct_ple and hasattr(RemotePleEmbedding._client, "gpu_resync"):
                # The PLE path advanced the workers' control sequence from the CPU; hand the new
                # value to the GPU-driven MoE exchange (the stream is idle here: ids went to host).
                RemotePleEmbedding._client.gpu_resync()
            raw = np.frombuffer(payload, dtype=np.uint8).copy()
            rows = torch.from_numpy(raw).view(torch.float8_e4m3fn).reshape(
                *gathered.shape, self.embedding_dim
            )
            rows = rows.to(device=ngram_ids.device)
            return self._select_embeddings(rows, ngram_ids.shape[0], slot_offset)

    ngram_mod.Qwen4ExpPLEDeviceEmbedding = RemotePleEmbedding
    ngram_mod.Qwen4ExpPLEPinnedHostEmbedding = RemotePleEmbedding

    ngram_cls = ngram_mod.Qwen4ExpNGramEmbedding
    original_ngram_load = ngram_cls.load_weights

    def filtered_ngram_load(self, weights):
        return original_ngram_load(
            self,
            ((name, tensor) for name, tensor in weights
             if not (name.startswith("ngram_embedding.shard_") and name.endswith(".weight"))),
        )

    ngram_cls.load_weights = filtered_ngram_load
    # Opt-in load-time FP8 for the linear-attention projections (Q38_RTX_FP8_LINEAR_ATTN=1);
    # wraps the filtered Qwen4ExpModel.load_weights installed above.
    from qwen38_rtx_fp8_hooks import install as install_rtx_fp8
    rtx_fp8 = install_rtx_fp8(model_mod)
    # Opt-in FP8 hyper-connection matrices (Q38_RTX_FP8_HC=1, qwen38_hc_fp8; 2026-11-03).
    from qwen38_hc_fp8 import install as install_hc_fp8
    rtx_fp8_hc = install_hc_fp8()
    # Opt-in logging of the QSA indexer selections (Q38_QSA_SELECT_LOG, eager decode only).
    from qwen38_qsa_select_log import install as install_qsa_select_log
    install_qsa_select_log()
    # Opt-in host-memory K/V tier for the QSA layers (Q38_KV_TIER=1).
    from qwen38_kv_tier_hook import install as install_kv_tier
    install_kv_tier()
    RemotePleEmbedding._client = None
    return {
        "moe_class": f"{moe_cls.__module__}.{moe_cls.__name__}",
        "ple_class": f"{RemotePleEmbedding.__module__}.{RemotePleEmbedding.__name__}",
        "expert_filter": [".mlp.experts.", ".mlp.shared_expert.", ".mlp.shared_expert_gate."],
        "ple_filter": "ngram_embedding.shard_*.weight",
        "cpu_heads": cpu_heads,
        "rtx_fp8": rtx_fp8,
        "rtx_fp8_hc": rtx_fp8_hc,
        "ple_transport": ("pci_cluster_row_gather_mode6" if _ple_gpu
                          else "direct_BAR1_resident_mode4" if direct_ple else "unix_socket"),
        "ple_protocol": "resident mode-4 gathers routed by PLE manifest into per-GPU BAR1 mailboxes" if direct_ple else "count:u32le + global_rows:u64le; status:u32le + count:u32le + count*160 raw FP8 bytes",
    }


def query_memory() -> list[dict[str, str]]:
    raw = subprocess.check_output([
        "nvidia-smi", "--query-gpu=index,name,uuid,pci.bus_id,memory.used,memory.free",
        "--format=csv,noheader,nounits",
    ], text=True)
    rows = []
    for line in raw.splitlines():
        cols = [x.strip() for x in line.split(",")]
        rows.append(dict(zip(("index", "name", "uuid", "bdf", "used_mib", "free_mib"), cols)))
    return rows


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def parse_ple_server_requests(path: Path) -> dict[str, Any]:
    requests = []
    if path.exists():
        for line in path.read_text(errors="replace").splitlines():
            match = re.fullmatch(r"PLE_LOOKUP_REQUEST,count,(\d+),status,(\d+)", line.strip())
            if match:
                requests.append({"rows": int(match.group(1)), "status": int(match.group(2))})
    return {
        "request_count": len(requests),
        "rows_returned": sum(x["rows"] for x in requests if x["status"] == 0),
        "successful_requests": sum(x["status"] == 0 for x in requests),
        "failed_requests": sum(x["status"] != 0 for x in requests),
        "requests": requests,
        "source": "PLE service request log (server-side; includes spawned vLLM worker)",
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=Path(os.environ.get("M10DIS_MODEL", "/mnt/nvme/models/Qwen3.8-Flash-Next-FP8")))
    ap.add_argument("--bundle-root", type=Path, required=True)
    ap.add_argument("--routed-service", type=Path, required=True)
    ap.add_argument("--shared-service", type=Path, required=True)
    ap.add_argument("--direct-library", type=Path,
                    help="optional native PEX/BAR1 MoE adapter for spawned vLLM workers")
    ap.add_argument("--direct-groups", type=Path,
                    help="fresh six-M10 PEX topology TSV required by --direct-library")
    ap.add_argument("--ple-binary", type=Path, required=True)
    ap.add_argument("--allocations", type=Path, required=True)
    ap.add_argument("--shards", type=Path, required=True)
    ap.add_argument("--startup-rows", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--cpu-offload-gb", type=float, default=6.0)
    ap.add_argument("--max-model-len", type=int, default=128)
    ap.add_argument("--max-new-tokens", type=int, default=3)
    ap.add_argument("--prompt-file", type=Path,
                    help="use an exact prompt file instead of the built-in smoke prompt")
    ap.add_argument("--repeats", type=int, default=2)
    ap.add_argument("--ignore-eos", action="store_true",
                    help="continue generation to max-new-tokens even after EOS")
    a = ap.parse_args()
    if a.repeats < 1:
        ap.error("--repeats must be positive")
    if bool(a.direct_library) != bool(a.direct_groups):
        ap.error("--direct-library and --direct-groups must be supplied together")
    model = a.model.resolve()
    out = a.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    RTX_UUID = discover_rtx_uuid()
    os.environ["CUDA_VISIBLE_DEVICES"] = RTX_UUID
    os.environ["PLE_SKIP_EXPERT_RESERVE"] = "1"
    sockpath = Path("/tmp") / f"q38ple-vllm-{os.getpid()}.sock"
    try:
        sockpath.unlink()
    except FileNotFoundError:
        pass

    result: dict[str, Any] = {
        "status": "RUNNING",
        "model": str(model),
        "model_index_sha256": sha256_file(model / "model.safetensors.index.json"),
        "vllm_gpu_uuid": RTX_UUID,
        "bundle_root": str(a.bundle_root.resolve()),
        "start_unix": time.time(),
        "stage8": "RUNNING",
        "stage9": "RUNNING",
    }
    (out / "run_state.json").write_text(json.dumps(result, indent=2) + "\n")
    manager = None
    ple_proc = None
    ple_log = None
    moe_rpc = None
    moe_socket_client = None
    ple_client: PleSocketClient | None = None
    llm = None
    error: str | None = None
    direct_timing = None
    try:
        sys.path.insert(0, str(Path(__file__).resolve().parent))
        from qwen38_resident_service_manager import ResidentServiceManager
        from vllm import LLM, SamplingParams
        import torch

        if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (8, 6):
            raise RuntimeError("vLLM process does not see only the RTX 3070 Ti (sm_86)")
        gpu_name = torch.cuda.get_device_name(0)
        if "3070 Ti" not in gpu_name:
            raise RuntimeError(f"unexpected vLLM device 0: {gpu_name}")
        result.update({"gpu_name": gpu_name, "gpu_capability": list(torch.cuda.get_device_capability(0)),
                       "torch_version": torch.__version__})

        # The PLE helper must claim its table before the expert workers load,
        # matching the already validated all-48 co-residency run. Its binary
        # must honor PLE_SKIP_EXPERT_RESERVE; the non-co-residency lookup
        # binary attempts an obsolete expert-sized reservation and will OOM.
        ple_env = os.environ.copy()
        ple_env.pop("CUDA_VISIBLE_DEVICES", None)
        ple_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        ple_env["PLE_SKIP_EXPERT_RESERVE"] = "1"
        ple_log = (out / "ple_server.log").open("wb")
        ple_cmd = [str(a.ple_binary.resolve()), str(a.allocations.resolve()),
                   str(a.shards.resolve()), str(a.startup_rows.resolve()), str(sockpath)]
        ple_proc = subprocess.Popen(ple_cmd, cwd=out, env=ple_env,
                                    stdout=ple_log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + 600
        while time.monotonic() < deadline:
            if ple_proc.poll() is not None:
                raise RuntimeError(f"PLE service exited before ready: {ple_proc.returncode}")
            if sockpath.exists() and "PLE_LOOKUP_SERVER_READY" in (out / "ple_server.log").read_text(errors="replace"):
                break
            time.sleep(1)
        else:
            raise TimeoutError("PLE service did not become ready in 600 seconds")
        result["ple_server"] = {"status": "READY", "pid": ple_proc.pid,
                                "socket": str(sockpath),
                                "binary": str(a.ple_binary.resolve()),
                                "binary_sha256": sha256_file(a.ple_binary.resolve())}

        service_log = out / "expert_service_logs"
        manager = ResidentServiceManager(a.routed_service.resolve(), a.shared_service.resolve(), service_log)
        load_records = []
        load_start = time.monotonic()
        for layer in range(48):
            layer_root = a.bundle_root.resolve() / f"layer_{layer:02d}"
            response = manager.load_layer_async(
                layer, layer_root / "routed", layer_root / "shared"
            ).result(timeout=600)
            if response.get("state") != "READY":
                raise RuntimeError(f"expert service failed to load layer {layer}: {response}")
            load_records.append(response)
            (out / "expert_load_progress.json").write_text(json.dumps({
                "status": "RUNNING", "loaded_layers": layer + 1,
                "load_seconds": time.monotonic() - load_start,
                "last_layer": response,
            }, indent=2) + "\n")
        result["expert_residency"] = {
            "status": "PASS", "layers": len(load_records),
            "load_seconds": time.monotonic() - load_start,
            "per_layer": load_records,
        }

        memory = query_memory()
        m10_free = [int(x["free_mib"]) for x in memory if "Tesla M10" in x["name"]]
        if len(m10_free) != 24 or min(m10_free) < 500:
            raise RuntimeError(f"M10 memory headroom check failed: count={len(m10_free)} min_free_mib={min(m10_free, default=-1)}")
        result["co_resident_memory"] = {"m10_count": len(m10_free),
                                        "minimum_free_mib": min(m10_free),
                                        "devices": memory}
        from qwen38_moe_rpc import MoeRpcServer, MoeSocketClient

        moe_sockpath = Path("/tmp") / f"q38moe-vllm-{os.getpid()}.sock"
        moe_rpc = MoeRpcServer(moe_sockpath, manager, out / "moe_request_scratch")
        adapter = MoeSocketClient(moe_sockpath)
        moe_socket_client = adapter
        hook_info = install_model_hooks(adapter, sockpath)
        result["model_hooks"] = hook_info
        result["moe_rpc_socket"] = str(moe_sockpath)

        direct_map_manifest = out / "gpu_mappings.manifest"
        if a.direct_library:
            # This one setup request creates retained BAR1 mappings and starts
            # resident kernels. Their idle watchdog is long enough to cover
            # vLLM startup; no concurrent heartbeat publisher runs while the
            # vLLM worker sends direct PEX/BAR1 requests.
            boot_h = torch.zeros((1, 2560), dtype=torch.float32, device="cuda")
            boot_ids = torch.arange(10, dtype=torch.int32, device="cuda").reshape(1, 10)
            boot_weights = torch.full((1, 10), 0.1, dtype=torch.float32, device="cuda")
            boot_start = time.monotonic()
            _ = adapter.forward(0, boot_h, boot_ids, boot_weights)
            torch.cuda.synchronize()
            manager._rlog.flush(); manager._slog.flush()
            routed_text = (service_log / "manager_routed.log").read_text(errors="replace")
            shared_text = (service_log / "manager_shared.log").read_text(errors="replace")
            trace_log = out / "rm_mapping_trace.log"
            trace_log.write_text(routed_text + shared_text)
            from qwen38_gpu_mappings_manifest import (
                extract_gpu_mappings, write_gpu_mappings_manifest,
            )
            mapping_records = extract_gpu_mappings(routed_text, shared_text)
            write_gpu_mappings_manifest(direct_map_manifest, mapping_records,
                                        run_id=f"{out.name}:{os.getpid()}")
            result["direct_moe_setup"] = {
                "status": "PASS", "bootstrap_layer": 0,
                "bootstrap_seconds": time.monotonic() - boot_start,
                "gpu_mappings_manifest": str(direct_map_manifest),
                "rm_mapping_trace_log": str(trace_log),
                "groups_tsv": str(a.direct_groups.resolve()),
                "library": str(a.direct_library.resolve()),
                "bootstrap_rpc_calls": moe_rpc.calls,
                "resident_idle_deadline_cycles": int(os.environ.get(
                    "Q38_RESIDENT_IDLE_DEADLINE_CYCLES", "120000000000")),
            }
        (out / "run_state.json").write_text(json.dumps(result, indent=2) + "\n")

        # vLLM 0.32 uses spawned EngineCore/worker processes. Runtime monkeypatches
        # in this parent are not inherited by spawn, so opt-in sitecustomize
        # applies the MoE/PLE hooks in those children. In direct mode, worker
        # MoE traffic goes to PEX/BAR1; the parent RPC remains setup-only.
        bootstrap_dir = out / "worker_bootstrap"
        bootstrap_dir.mkdir(parents=True, exist_ok=True)
        (bootstrap_dir / "sitecustomize.py").write_text(
            (Path(__file__).resolve().parent / "sitecustomize.py").read_text()
        )
        (bootstrap_dir / "qwen38_moe_rpc.py").symlink_to(
            Path(__file__).resolve().parent / "qwen38_moe_rpc.py"
        )
        if a.direct_library:
            (bootstrap_dir / "qwen38_direct_moe_adapter.py").symlink_to(
                Path(__file__).resolve().parent / "qwen38_direct_moe_adapter.py"
            )
        (bootstrap_dir / "run_stage8_stage9_vllm_generation.py").symlink_to(
            Path(__file__).resolve()
        )
        old_pythonpath = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [str(bootstrap_dir), str(Path(__file__).resolve().parent), old_pythonpath]
        ).rstrip(os.pathsep)
        os.environ["Q38_M10_WORKER_HOOK"] = "1"
        os.environ["Q38_M10_MOE_SOCKET"] = str(moe_sockpath)
        os.environ["Q38_M10_PLE_SOCKET"] = str(sockpath)
        os.environ["Q38_M10_BOOTSTRAP_LOG"] = str(out / "worker_bootstrap.log")
        if a.direct_library:
            os.environ["Q38_M10_DIRECT_LIBRARY"] = str(a.direct_library.resolve())
            os.environ["Q38_M10_DIRECT_GROUPS"] = str(a.direct_groups.resolve())
            os.environ["Q38_M10_DIRECT_GPU_MAPPINGS_MANIFEST"] = str(direct_map_manifest.resolve())
            os.environ["Q38_M10_DIRECT_TIMING_JSON"] = str(out / "direct_moe_timing")

        llm_start = time.monotonic()
        llm = LLM(
            model=str(model),
            tokenizer=str(model),
            trust_remote_code=True,
            tensor_parallel_size=1,
            dtype="bfloat16",
            gpu_memory_utilization=0.78,
            cpu_offload_gb=a.cpu_offload_gb,
            enforce_eager=True,
            max_model_len=a.max_model_len,
            max_num_seqs=1,
            max_num_batched_tokens=8,
            distributed_executor_backend="uni",
            # This is a text-only generation validation. hf_overrides below
            # selects the model's language branch, while this separate vLLM
            # engine option disables its multimodal encoder/cache budget.
            # Without it vLLM still profiles a maximum-size image encoder
            # during startup and can exhaust the RTX 3070 Ti before inference.
            language_model_only=True,
            hf_overrides={"language_model_only": True},
            disable_custom_all_reduce=True,
        )
        bootstrap_log = out / "worker_bootstrap.log"
        bootstrap_records = []
        if bootstrap_log.exists():
            for line in bootstrap_log.read_text(errors="replace").splitlines():
                try:
                    bootstrap_records.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
        patched_children = [x for x in bootstrap_records
                            if x.get("status") == "PATCHED" and x.get("pid") != os.getpid()]
        result["worker_bootstrap"] = {
            "records": bootstrap_records,
            "patched_child_count": len(patched_children),
            "log": str(bootstrap_log),
        }
        if not patched_children:
            raise RuntimeError(
                "vLLM initialized without a spawned child confirming the M10 model hooks; "
                "refusing generation with an unpatched PLE/MoE path"
            )
        if a.direct_library and not any(
            x.get("moe_transport") == "direct_PEX_BAR1" for x in patched_children
        ):
            raise RuntimeError("spawned vLLM workers did not confirm direct PEX/BAR1 MoE hooks")
        result["vllm_load_seconds"] = time.monotonic() - llm_start
        prompt = (a.prompt_file.read_text()
                  if a.prompt_file else "The capital of Denmark is")
        sampling = SamplingParams(
            temperature=0.0,
            max_tokens=a.max_new_tokens,
            ignore_eos=a.ignore_eos,
            seed=0,
        )
        generations = []
        for repeat in range(a.repeats):
            start = time.monotonic()
            outputs = llm.generate([prompt], sampling, use_tqdm=False)
            elapsed = time.monotonic() - start
            if len(outputs) != 1 or len(outputs[0].outputs) != 1:
                raise RuntimeError(f"unexpected vLLM output cardinality: {len(outputs)}")
            item = outputs[0]
            candidate = item.outputs[0]
            generations.append({
                "repeat": repeat,
                "prompt": prompt,
                "prompt_token_ids": list(item.prompt_token_ids or []),
                "generated_token_ids": list(candidate.token_ids),
                "text": candidate.text,
                "finish_reason": candidate.finish_reason,
                "elapsed_seconds": elapsed,
            })
        if len(generations) > 1 and any(
            x["generated_token_ids"] != generations[0]["generated_token_ids"]
            for x in generations[1:]
        ):
            raise RuntimeError("greedy repeated prompts produced different token sequences")
        ple_stats = parse_ple_server_requests(out / "ple_server.log")
        if ple_stats["successful_requests"] == 0 or ple_stats["rows_returned"] == 0:
            raise RuntimeError(f"generation did not produce a successful M10 PLE lookup: {ple_stats}")
        if a.direct_library:
            if moe_rpc.calls != 1:
                raise RuntimeError(f"expected exactly one setup-only MoE RPC bootstrap, observed {moe_rpc.calls}")
        elif moe_rpc.calls == 0 or moe_rpc.rows == 0:
            raise RuntimeError("generation did not cross the M10 MoE service boundary")
        result.update({
            "stage8": "PASS_VLLM_M10_EXPERT_BOUNDARY_IN_ENGINE",
            "stage9": "PASS_FULL_MODEL_TEXT_GENERATION",
            "vllm_load_seconds": result["vllm_load_seconds"],
            "generation": generations,
            "moe_service_calls": moe_rpc.calls,
            "moe_service_rows": moe_rpc.rows,
            "ple_service": ple_stats,
            "scope": "text-only greedy generation through vLLM on RTX 3070 Ti; routed/shared experts on six M10s; PLE table on M10s; other model operators and LM head on vLLM RTX/CPU-offload path",
            "caveats": [
                "experimental in-process vLLM monkeypatch; no production packaging",
                ("MoE bootstrap uses one setup-only service request; model MoE calls use direct PEX/BAR1"
                 if a.direct_library else
                 "M10 request transport uses per-row temporary files and is not performance optimized"),
                "only a short text prompt and two repeated greedy generations were exercised",
            ],
        })
        result["status"] = "PASS_STAGE8_STAGE9_VLLM_GENERATION"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        result["error"] = error
        result["error_traceback"] = traceback.format_exc()
        result["status"] = "PARTIAL_STAGE8_STAGE9_VLLM_GENERATION"
    finally:
        if llm is not None:
            try:
                engine = getattr(llm, "llm_engine", None)
                core = getattr(engine, "engine_core", None)
                shutdown = getattr(core, "shutdown", None)
                if callable(shutdown):
                    shutdown()
            except Exception as exc:
                result.setdefault("cleanup_errors", []).append(f"vLLM shutdown: {exc}")
            llm = None
            gc.collect()
        if a.direct_library:
            timing_paths = list(out.glob("direct_moe_timing.*.json"))
            for _ in range(50):
                if timing_paths:
                    break
                time.sleep(0.1)
                timing_paths = list(out.glob("direct_moe_timing.*.json"))
            timing_records = [json.loads(p.read_text()) for p in timing_paths]
            active_records = [x for x in timing_records if x.get("calls", 0) > 0]
            if active_records:
                direct_timing = {
                    "calls": sum(x.get("calls", 0) for x in active_records),
                    "rows": sum(x.get("rows", 0) for x in active_records),
                    "processes": active_records,
                }
            if not direct_timing or direct_timing.get("calls", 0) == 0:
                result.setdefault("cleanup_errors", []).append(
                    "vLLM worker produced no direct-MoE timing summary/calls"
                )
            else:
                result["direct_moe_timing"] = direct_timing
        ngram_mod = sys.modules.get("vllm.models.qwen4_exp.nvidia.ngram_embedding")
        if ngram_mod is not None:
            remote_cls = getattr(ngram_mod, "Qwen4ExpPLEDeviceEmbedding", None)
            client = getattr(remote_cls, "_client", None)
            if client is not None:
                ple_client = client
        if ple_client is not None:
            try:
                ple_client.close(stop=True)
                ple_client = None
            except Exception as exc:
                result.setdefault("cleanup_errors", []).append(f"PLE client stop: {exc}")
        if moe_socket_client is not None:
            try:
                moe_socket_client.close()
            except Exception as exc:
                result.setdefault("cleanup_errors", []).append(f"MoE RPC client cleanup: {exc}")
        if moe_rpc is not None:
            try:
                result["moe_rpc_stats"] = moe_rpc.timing_snapshot()
                moe_rpc.close()
            except Exception as exc:
                result.setdefault("cleanup_errors", []).append(f"MoE RPC server cleanup: {exc}")
        # Resident workers import PLE allocations through CUDA IPC. Stop the
        # workers and close those imported handles before stopping the PLE
        # allocation owner. Reversing this order can invalidate memory still
        # referenced by resident kernels and leave STOP waiting forever.
        if manager is not None:
            try:
                manager.close()
                result["expert_services_stopped"] = True
            except Exception as exc:
                result.setdefault("cleanup_errors", []).append(f"expert service cleanup: {exc}")
        if ple_proc is not None:
            try:
                if ple_proc.poll() is None and sockpath.exists():
                    # If no model call opened the client, connect only to send
                    # the service's documented stop marker after CUDA work is idle.
                    stop = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
                    stop.settimeout(5)
                    stop.connect(str(sockpath))
                    stop.sendall(struct.pack("<I", 0xFFFFFFFF))
                    stop.close()
                result["ple_server_return_code"] = ple_proc.wait(timeout=180)
                if result["ple_server_return_code"] != 0:
                    result.setdefault("cleanup_errors", []).append(
                        f"PLE service exited with status {result['ple_server_return_code']}"
                    )
            except Exception as exc:
                result.setdefault("cleanup_errors", []).append(f"PLE process cleanup: {exc}")
                if ple_proc.poll() is None:
                    ple_proc.terminate()
                    try:
                        ple_proc.wait(timeout=20)
                    except subprocess.TimeoutExpired:
                        ple_proc.kill(); ple_proc.wait()
            if ple_log is not None:
                ple_log.close()
            result["ple_service"] = parse_ple_server_requests(out / "ple_server.log") | {
                "return_code": result.get("ple_server_return_code"),
                "clean_eof": "PLE_LOOKUP_CLIENT_EOF,clean" in
                    (out / "ple_server.log").read_text(errors="replace"),
            }
        result["final_gpu_memory"] = query_memory()
        result["end_unix"] = time.time()
        if result.get("cleanup_errors"):
            result["status"] = "FAIL_CLEANUP"
        (out / "stage8_stage9_vllm_generation_result.json").write_text(
            json.dumps(result, indent=2) + "\n"
        )
        (out / "run_state.json").write_text(json.dumps(result, indent=2) + "\n")
        print(result["status"], error or "", flush=True)
    return 0 if result["status"] == "PASS_STAGE8_STAGE9_VLLM_GENERATION" else 1


if __name__ == "__main__":
    raise SystemExit(main())
