"""Opt-in vLLM worker bootstrap for the experimental Qwen38 M10 bridge."""
from __future__ import annotations

import json
import os
import traceback
from pathlib import Path

if os.environ.get("Q38_M10_WORKER_HOOK") == "1":
    _record = {
        "pid": os.getpid(),
        "status": "STARTING",
    }
    try:
        # The RTX peer route discovers/retains RM exports from this EngineCore
        # process. LD_PRELOAD can be ignored for secure-exec launches, so load
        # the tracer explicitly before importing the CUDA-backed model hooks.
        _trace_path = os.environ.get("Q38_RM_TRACE_PRELOAD")
        if os.environ.get("Q38_M10_RTX_PEER_LIBRARY"):
            if not _trace_path:
                raise RuntimeError("RTX peer mode requires Q38_RM_TRACE_PRELOAD")
            import ctypes

            _trace = Path(_trace_path).resolve(strict=True)
            ctypes.CDLL(str(_trace), mode=ctypes.RTLD_GLOBAL)
            _record["rm_trace_preload"] = str(_trace)

        from run_stage8_stage9_vllm_generation import install_model_hooks

        _ple_path = Path(os.environ["Q38_M10_PLE_SOCKET"])
        _rtx_peer_library = os.environ.get("Q38_M10_RTX_PEER_LIBRARY")
        _direct_library = os.environ.get("Q38_M10_DIRECT_LIBRARY")
        if _rtx_peer_library:
            from qwen38_vllm_rtx_peer_adapter import RtxPeerResidentM10MoEAdapter
            from qwen38_direct_moe_adapter import DirectResidentM10MoEAdapter
            class _LazyRtxPeerAdapter:
                @property
                def supports_direct_ple(self):
                    return bool(os.environ.get("Q38_M10_DIRECT_PLE_MANIFEST"))

                def __init__(self):
                    self._inner = None

                def _get(self):
                    if self._inner is None:
                        ple = DirectResidentM10MoEAdapter(
                            _direct_library, os.environ["Q38_M10_DIRECT_GROUPS"],
                            os.environ["Q38_M10_DIRECT_GPU_MAPPINGS_MANIFEST"],
                            os.environ.get("Q38_M10_DIRECT_PLE_MANIFEST"),
                        ) if self.supports_direct_ple else None
                        self._inner = RtxPeerResidentM10MoEAdapter(
                            _rtx_peer_library,
                            os.environ["Q38_M10_DIRECT_GPU_MAPPINGS_MANIFEST"],
                            ple_adapter=ple,
                            # None -> adapter uses Q38_RTX_BDF or a live scan; never a hardcoded address.
                            rtx_bdf=os.environ.get("Q38_M10_RTX_BDF") or None,
                            control_socket=os.environ.get("Q38_M10_CONTROL_SOCKET"),
                        )
                    return self._inner

                def forward(self, *args, **kwargs):
                    return self._get().forward(*args, **kwargs)

                def lookup_ple(self, global_rows):
                    return self._get().lookup_ple(global_rows)

                # Phase 2 GPU-driven exchange helpers (no-ops before the peer path opens)
                def check_error(self):
                    if self._inner is not None and getattr(self._inner, "gpu_driven", False):
                        self._inner.check_error()

                def gpu_resync(self):
                    if self._inner is not None and getattr(self._inner, "gpu_driven", False):
                        self._inner.gpu_resync()

                # PCI-cluster row gather for PLE (Q38_RTX_PEER_PLE_GPU=1); opens the peer path if needed.
                def ple_gather_gpu(self, rows):
                    return self._get().ple_gather_gpu(rows)

                # LM head on the M10s (Q38_M10_LM_HEAD=1); opens the peer path if needed.
                def lm_head_logits(self, x):
                    return self._get().lm_head_logits(x)

                def close(self):
                    if self._inner is not None:
                        self._inner.close()

            _adapter = _LazyRtxPeerAdapter()
            _record["moe_transport"] = "RTX_CUDA_to_M10_BAR1_peer"
            _record["hooks"] = install_model_hooks(_adapter, None)
            _record["ple_transport"] = ("pci_cluster_row_gather_mode6"
                                        if os.environ.get("Q38_RTX_PEER_GPU_DRIVEN") == "1"
                                        and os.environ.get("Q38_RTX_PEER_PLE_GPU") == "1"
                                        else "direct_BAR1_resident_mode4")
        elif _direct_library:
            from qwen38_direct_moe_adapter import DirectResidentM10MoEAdapter
            class _LazyDirectAdapter:
                @property
                def supports_direct_ple(self):
                    return bool(os.environ.get("Q38_M10_DIRECT_PLE_MANIFEST"))

                def __init__(self):
                    self._inner = None

                def forward(self, *args, **kwargs):
                    if self._inner is None:
                        self._inner = DirectResidentM10MoEAdapter(
                            _direct_library,
                            os.environ["Q38_M10_DIRECT_GROUPS"],
                            os.environ["Q38_M10_DIRECT_GPU_MAPPINGS_MANIFEST"],
                            os.environ["Q38_M10_DIRECT_PLE_MANIFEST"],
                        )
                    return self._inner.forward(*args, **kwargs)

                def lookup_ple(self, global_rows):
                    if self._inner is None:
                        self._inner = DirectResidentM10MoEAdapter(
                            _direct_library,
                            os.environ["Q38_M10_DIRECT_GROUPS"],
                            os.environ["Q38_M10_DIRECT_GPU_MAPPINGS_MANIFEST"],
                            os.environ["Q38_M10_DIRECT_PLE_MANIFEST"],
                        )
                    return self._inner.lookup_ple(global_rows)

                def close(self):
                    if self._inner is not None:
                        self._inner.close()

            _adapter = _LazyDirectAdapter()
            _record["moe_transport"] = "direct_PEX_BAR1"
            _record["hooks"] = install_model_hooks(_adapter, None)
            _record["ple_transport"] = "direct_BAR1_resident_mode4"
        else:
            from qwen38_moe_rpc import MoeSocketClient
            _moe_path = os.environ["Q38_M10_MOE_SOCKET"]
            _record["moe_transport"] = "unix_socket"
            _record["hooks"] = install_model_hooks(MoeSocketClient(_moe_path), _ple_path)
        _record["status"] = "PATCHED"
    except BaseException as _exc:
        _record.update({"status": "FAILED", "error": repr(_exc),
                        "traceback": traceback.format_exc()})
        _log_path = os.environ.get("Q38_M10_BOOTSTRAP_LOG")
        if _log_path:
            with open(_log_path, "a", encoding="utf-8") as _f:
                _f.write(json.dumps(_record) + "\n")
        raise
    _log_path = os.environ.get("Q38_M10_BOOTSTRAP_LOG")
    if _log_path:
        with open(_log_path, "a", encoding="utf-8") as _f:
            _f.write(json.dumps(_record) + "\n")
