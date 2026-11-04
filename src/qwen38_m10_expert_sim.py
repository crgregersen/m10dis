#!/usr/bin/env python3
"""Standalone layer-wise Qwen3.8 routed and shared expert runner for six M10s.

The first five M10s run the routed branch: top-10 experts, each sliced across
20 GM107s (32 intermediate columns per GPU). The sixth M10 runs the BF16 shared
expert over four 160-column slices. The CPU combines both outputs and verifies
them against independent FP32 references. Attention, residual/hyper-connection,
and token-generation work remain outside this phase.

Without --activations-dir, the saved representative layer-0 hidden vector is
reused for every layer. This is a layer-isolated workload sweep, not a
sequential model forward pass. Supply one true 2560-element FP32 activation
per layer as layer_NN.f32 to use externally produced layer states.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import mmap
import os
import re
import shutil
import struct
import subprocess
import sys
import tempfile
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np

HIDDEN = 2560
INTERMEDIATE = 640
TOP_K = 10
WORKERS = 20
SLICE = INTERMEDIATE // WORKERS
SHARED_WORKERS = 4
SHARED_SLICE = INTERMEDIATE // SHARED_WORKERS
PREFIX = "model.language_model.layers.{layer}.mlp."
MTP_LAYER = 48          # the MTP layer's experts / shared expert, resident as layer 48 (2026-11-04)


def layer_prefix(layer: int) -> str:
    return "mtp.layers.0.mlp." if layer == MTP_LAYER else PREFIX.format(layer=layer)
PROJECTIONS = ("gate_proj", "up_proj", "down_proj")


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 << 20), b""):
            h.update(block)
    return h.hexdigest()


class TensorReader:
    """Small mmap reader for local safetensors; reads only requested tensors."""
    def __init__(self, path: Path):
        self.path = path
        self.file = path.open("rb")
        self.length = struct.unpack("<Q", self.file.read(8))[0]
        self.header = json.loads(self.file.read(self.length))
        self.data_offset = 8 + self.length
        self.map = mmap.mmap(self.file.fileno(), 0, access=mmap.ACCESS_READ)

    def get(self, key: str) -> np.ndarray:
        meta = self.header[key]
        a, b = meta["data_offsets"]
        shape = tuple(meta["shape"])
        dtype = meta["dtype"]
        raw = memoryview(self.map)[self.data_offset + a:self.data_offset + b]
        if dtype == "F8_E4M3":
            out = np.frombuffer(raw, dtype=np.uint8).reshape(shape).copy()
        elif dtype == "BF16":
            u = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
            out = (u << 16).view(np.float32).reshape(shape)
        else:
            raise ValueError(f"unsupported tensor dtype {dtype}: {key}")
        del raw
        return out

    def close(self):
        self.map.close()
        self.file.close()


def load_tensor(key: str, weight_map: dict[str, str], readers: dict[str, TensorReader]) -> np.ndarray:
    shard = weight_map[key]
    if shard not in readers:
        readers[shard] = TensorReader(MODEL / shard)
    return readers[shard].get(key)


def fp8_lut() -> np.ndarray:
    lut = np.empty(256, dtype=np.float32)
    for code in range(256):
        sign = -1.0 if code & 0x80 else 1.0
        e, m = (code >> 3) & 0xF, code & 7
        if e == 0:
            value = m * (2.0 ** -9)
        elif e == 15 and m == 7:
            value = np.nan
        else:
            value = (1.0 + m / 8.0) * (2.0 ** (e - 7))
        lut[code] = sign * value
    return lut


def selected_routes(layer: int, x: np.ndarray, weight_map: dict[str, str],
                    readers: dict[str, TensorReader], route_override: dict | None = None):
    gate = load_tensor(layer_prefix(layer) + "gate.weight", weight_map, readers)
    if gate.shape != (512, HIDDEN):
        raise ValueError(f"layer {layer} router shape {gate.shape}, expected (512,2560)")
    logits = gate @ x
    router_ids = np.argsort(logits)[-TOP_K:][::-1].astype(np.int32)
    values = logits[router_ids].astype(np.float32)
    router_probs = np.exp(values - values.max())
    router_probs /= router_probs.sum()
    if route_override is None:
        return router_ids, router_probs.astype(np.float32), logits, router_ids, "checkpoint_router"
    ids = np.asarray(route_override.get("expert_ids"), dtype=np.int32)
    probs = np.asarray(route_override.get("route_weights"), dtype=np.float32)
    if ids.shape != (TOP_K,) or probs.shape != (TOP_K,):
        raise ValueError(f"layer {layer} override requires {TOP_K} expert IDs and route weights")
    if np.any(ids < 0) or np.any(ids >= 512) or len(np.unique(ids)) != TOP_K:
        raise ValueError(f"layer {layer} override IDs must be unique values in [0,512)")
    if not np.isfinite(probs).all() or np.any(probs < 0) or not np.isclose(probs.sum(), 1.0, atol=1e-6):
        raise ValueError(f"layer {layer} override weights must be finite, nonnegative, and sum to one")
    source = str(route_override.get("source", "explicit_fixture"))
    return ids, probs, logits, router_ids, source


def prepare_shared(layer: int, x: np.ndarray, weight_map: dict[str, str],
                   readers: dict[str, TensorReader], work: Path) -> tuple[np.ndarray, float, dict]:
    """Prepare checkpoint BF16 shared-expert weights for four GM107 slices."""
    prefix = layer_prefix(layer)
    g = load_tensor(prefix + "shared_expert.gate_proj.weight", weight_map, readers)
    u = load_tensor(prefix + "shared_expert.up_proj.weight", weight_map, readers)
    d = load_tensor(prefix + "shared_expert.down_proj.weight", weight_map, readers)
    sg = load_tensor(prefix + "shared_expert_gate.weight", weight_map, readers)
    for key in (prefix + "shared_expert.gate_proj.weight",
                prefix + "shared_expert.up_proj.weight",
                prefix + "shared_expert.down_proj.weight",
                prefix + "shared_expert_gate.weight"):
        dtype = readers[weight_map[key]].header[key]["dtype"]
        if dtype != "BF16":
            raise ValueError(f"layer {layer} expected BF16 shared tensor {key}, found {dtype}")
    if g.shape != (INTERMEDIATE, HIDDEN) or u.shape != g.shape:
        raise ValueError(f"layer {layer} shared gate/up shapes {g.shape}/{u.shape}")
    if d.shape != (HIDDEN, INTERMEDIATE) or sg.shape != (1, HIDDEN):
        raise ValueError(f"layer {layer} shared down/gate shapes {d.shape}/{sg.shape}")

    gate_logit = float(sg.reshape(-1) @ x)
    shared_gate = 1.0 / (1.0 + np.exp(-gate_logit))
    gate = g @ x
    up = u @ x
    hidden = (gate / (1.0 + np.exp(-gate))) * up
    reference = (d @ hidden).astype(np.float32) * np.float32(shared_gate)

    work.mkdir(parents=True, exist_ok=True)
    x.astype(np.float32).tofile(work / "input.bin")
    def write_bf16_words(a: np.ndarray, path: Path) -> None:
        # TensorReader expands BF16 to an exactly corresponding FP32 bit pattern.
        # Keeping the upper 16 bits preserves the original checkpoint values.
        f32 = np.ascontiguousarray(a, dtype=np.float32)
        (f32.view(np.uint32) >> 16).astype(np.uint16).tofile(path)

    write_bf16_words(sg, work / "shared_gate_weight.bin")
    for shard in range(SHARED_WORKERS):
        start, end = shard * SHARED_SLICE, (shard + 1) * SHARED_SLICE
        write_bf16_words(g[start:end], work / f"shared.{shard}.gate.bin")
        write_bf16_words(u[start:end], work / f"shared.{shard}.up.bin")
        write_bf16_words(d[:, start:end], work / f"shared.{shard}.down.bin")
        (d[:, start:end] @ hidden[start:end]).astype(np.float32).tofile(
            work / f"shared.{shard}.ref.bin")
    (work / "shared_gate_scale.reference.txt").write_text(f"{shared_gate:.9g}\n")
    manifest = {
        "layer": layer,
        "format": "exact BF16 checkpoint bit patterns stored as uint16 and expanded to FP32 for GM107 accumulation",
        "intermediate": INTERMEDIATE,
        "workers": SHARED_WORKERS,
        "slice_width": SHARED_SLICE,
        "gate_logit": gate_logit,
        "gate_sigmoid": float(shared_gate),
        "equation": "sigmoid(shared_expert_gate(x)) * down(silu(gate(x)) * up(x))",
        "gate_scalar_computed_on": "first GM107 of sixth M10",
    }
    (work / "shared_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return reference, float(shared_gate), manifest


def expand_scale(s: np.ndarray, rows: int, cols: int) -> np.ndarray:
    return np.repeat(np.repeat(s, 128, axis=-2), 128, axis=-1)[..., :rows, :cols]


def prepare_layer(layer: int, x: np.ndarray, ids: np.ndarray, probs: np.ndarray,
                  weight_map: dict[str, str], readers: dict[str, TensorReader], work: Path,
                  lut: np.ndarray) -> tuple[np.ndarray, dict]:
    prefix = layer_prefix(layer)
    qg, qu, qd, sg, su, sd = [], [], [], [], [], []
    for expert in ids.tolist():
        vals = {}
        for proj in PROJECTIONS:
            base = prefix + f"experts.{expert}.{proj}"
            vals[proj] = (load_tensor(base + ".weight", weight_map, readers),
                          load_tensor(base + ".weight_scale_inv", weight_map, readers))
        g, gs = vals["gate_proj"]
        u, us = vals["up_proj"]
        d, ds = vals["down_proj"]
        if g.shape != (INTERMEDIATE, HIDDEN) or u.shape != g.shape or d.shape != (HIDDEN, INTERMEDIATE):
            raise ValueError(f"unexpected expert matrix shape in layer {layer}, expert {expert}")
        if gs.shape != (5, 20) or us.shape != gs.shape or ds.shape != (20, 5):
            raise ValueError(f"unexpected FP8 scale shape in layer {layer}, expert {expert}")
        qg.append(g); qu.append(u); qd.append(d)
        sg.append(gs); su.append(us); sd.append(ds)

    qg, qu, qd = np.stack(qg), np.stack(qu), np.stack(qd)
    sg, su, sd = np.stack(sg), np.stack(su), np.stack(sd)
    # Independent CPU reference: full FP32 dequantized routed branch.
    full = np.zeros(HIDDEN, dtype=np.float32)
    for e in range(TOP_K):
        g = lut[qg[e]] * expand_scale(sg[e], INTERMEDIATE, HIDDEN)
        u = lut[qu[e]] * expand_scale(su[e], INTERMEDIATE, HIDDEN)
        d = lut[qd[e]] * expand_scale(sd[e], HIDDEN, INTERMEDIATE)
        a = g @ x
        b = u @ x
        act = (a / (1.0 + np.exp(-a))) * b
        full += probs[e] * (d @ act)

    work.mkdir(parents=True, exist_ok=True)
    x.astype(np.float32).tofile(work / "input.bin")
    probs.astype(np.float32).tofile(work / "routes.bin")
    full.tofile(work / "bf16_source_reference.bin")  # benchmark's full-output check input
    manifest = {"layer": layer, "expert_ids": ids.tolist(), "routes": probs.tolist(),
                "format": "official E4M3 FP8, BF16 inverse scales, 128x128 scale blocks",
                "slice_width": SLICE, "workers": WORKERS}
    (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")

    slice_refs = []
    for shard in range(WORKERS):
        start, end = shard * SLICE, (shard + 1) * SLICE
        qg[:, start:end, :].tofile(work / f"official_fp8.{shard}.gate.bin")
        qu[:, start:end, :].tofile(work / f"official_fp8.{shard}.up.bin")
        qd[:, :, start:end].transpose(0, 2, 1).copy().tofile(work / f"official_fp8.{shard}.down.bin")
        gs_part = np.empty((TOP_K, SLICE, HIDDEN // 128), dtype=np.float32)
        us_part = np.empty_like(gs_part)
        for ei in range(TOP_K):
            for r in range(SLICE):
                gs_part[ei, r] = sg[ei, (start + r) // 128]
                us_part[ei, r] = su[ei, (start + r) // 128]
        gs_part.tofile(work / f"official_fp8.{shard}.gs.bin")
        us_part.tofile(work / f"official_fp8.{shard}.us.bin")
        ds_part = np.empty((TOP_K, HIDDEN), dtype=np.float32)
        for ei in range(TOP_K):
            ds_part[ei] = sd[ei, np.arange(HIDDEN) // 128, start // 128]
        ds_part.tofile(work / f"official_fp8.{shard}.ds.bin")
        part_ref = np.zeros(HIDDEN, dtype=np.float32)
        for ei in range(TOP_K):
            gsx = lut[qg[ei, start:end]] * expand_scale(sg[ei], INTERMEDIATE, HIDDEN)[start:end]
            usx = lut[qu[ei, start:end]] * expand_scale(su[ei], INTERMEDIATE, HIDDEN)[start:end]
            dsx = lut[qd[ei, :, start:end]] * expand_scale(sd[ei, :, start // 128:start // 128 + 1], HIDDEN, SLICE)
            a, b = gsx @ x, usx @ x
            act = (a / (1.0 + np.exp(-a))) * b
            part_ref += probs[ei] * (dsx @ act)
        part_ref.tofile(work / f"official_fp8.{shard}.ref.bin")
        slice_refs.append(part_ref)
    return full, manifest


def choose_gpu_groups() -> tuple[list[str], list[str], list[dict]]:
    q = subprocess.run(["nvidia-smi", "--query-gpu=serial,uuid,name,pci.bus_id",
                        "--format=csv,noheader,nounits"], check=True, text=True, capture_output=True)
    cards: dict[str, list[tuple[str, str]]] = {}
    for line in q.stdout.splitlines():
        serial, uuid, name, bdf = [v.strip() for v in line.split(",", 3)]
        if "Tesla M10" in name:
            cards.setdefault(serial, []).append((bdf, uuid))
    ordered = sorted(cards.items(), key=lambda item: min(bdf for bdf, _ in item[1]))
    if len(ordered) < 6 or any(len(gpus) != 4 for _, gpus in ordered[:6]):
        counts = {serial: len(gpus) for serial, gpus in ordered}
        raise RuntimeError(f"need six complete M10s (four GM107s each); found {counts}")
    groups = []
    for serial, gpus in ordered[:6]:
        gpus.sort()
        groups.append({"serial": serial, "bdfs": [bdf for bdf, _ in gpus],
                       "uuids": [uuid for _, uuid in gpus]})
    routed = [uuid for card in groups[:5] for uuid in card["uuids"]]
    shared = groups[5]["uuids"]
    return routed, shared, groups


def build_kernel(source: Path, binary: Path):
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError("nvcc not found; CUDA 12.4 sm_50 compiler is required")
    cmd = [nvcc, "-O3", "-std=c++17", "-ccbin", "/usr/bin/g++-13", "-arch=sm_50",
           "-lineinfo", "-Xptxas=-v", "-DFP8_GROUP=128", "-DDOWN_UNROLL=4",
           str(source), "-o", str(binary)]
    print("BUILD", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def build_shared_kernel(source: Path, binary: Path):
    nvcc = shutil.which("nvcc")
    if not nvcc:
        raise RuntimeError("nvcc not found; CUDA 12.4 sm_50 compiler is required")
    cmd = [nvcc, "-O3", "-std=c++17", "-ccbin", "/usr/bin/g++-13", "-arch=sm_50",
           "-lineinfo", "-Xptxas=-v", str(source), "-o", str(binary)]
    print("BUILD_SHARED", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> int:
    global MODEL
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--model", type=Path, default=Path(os.environ.get("M10DIS_MODEL", "/mnt/nvme/models/Qwen3.8-Flash-Next-FP8")))
    ap.add_argument("--sample-data", type=Path, default=Path("benchmark-results/qwen38-official-fp8-block128/20261028T014135Z"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--kernel-source", type=Path, default=Path(__file__).with_name("bench_official_fp8.cu"))
    ap.add_argument("--kernel", type=Path, default=None)
    ap.add_argument("--shared-kernel-source", type=Path, default=Path(__file__).with_name("bench_shared_bf16.cu"))
    ap.add_argument("--shared-kernel", type=Path, default=None)
    ap.add_argument("--no-shared", action="store_true", help="run the routed branch only")
    ap.add_argument("--layers", default="all", help="all or comma-separated zero-based layer indices")
    ap.add_argument("--activations-dir", type=Path, help="layer_NN.f32 files, exactly 2560 FP32 values each")
    ap.add_argument("--routes-json", type=Path,
                    help="optional JSON object keyed by layer index with expert_ids and route_weights")
    ap.add_argument("--rounds", type=int, default=1, help="measured resident kernel iterations per layer")
    ap.add_argument("--build", action="store_true")
    ap.add_argument("--keep-prepared", action="store_true", help="retain per-layer packed files")
    args = ap.parse_args()
    MODEL = args.model.resolve()
    args.out = args.out.resolve()
    args.out.mkdir(parents=True, exist_ok=True)
    model_index_path = MODEL / "model.safetensors.index.json"
    config_path = MODEL / "config.json"
    config = json.loads(config_path.read_text())
    text_cfg = config.get("text_config", config)
    if (text_cfg["num_hidden_layers"], text_cfg["hidden_size"], text_cfg["moe_intermediate_size"],
        text_cfg["num_experts"], text_cfg["num_experts_per_tok"]) != (48, HIDDEN, INTERMEDIATE, 512, TOP_K):
        raise RuntimeError("checkpoint dimensions differ from this runner's validated Qwen3.8 layout")
    weight_map = json.loads(model_index_path.read_text())["weight_map"]
    route_overrides = json.loads(args.routes_json.read_text()) if args.routes_json else {}
    if not isinstance(route_overrides, dict):
        raise ValueError("--routes-json must contain an object keyed by layer index")
    layers = list(range(48)) if args.layers == "all" else [int(x) for x in args.layers.split(",")]
    if not layers or any(x < 0 or x >= 48 for x in layers):
        raise ValueError("layers must be 'all' or indices in 0..47")
    if args.rounds < 1:
        raise ValueError("rounds must be positive")
    sample = np.fromfile(args.sample_data / "input.bin", dtype=np.float32)
    if sample.shape != (HIDDEN,):
        raise RuntimeError(f"sample input must have {HIDDEN} FP32 values")
    binary = (args.kernel or args.out / "qwen38_m10_unroll4").resolve()
    if args.build or not binary.exists():
        build_kernel(args.kernel_source.resolve(), binary)
    shared_binary = (args.shared_kernel or args.out / "qwen38_shared_bf16").resolve()
    if not args.no_shared and (args.build or not shared_binary.exists()):
        build_shared_kernel(args.shared_kernel_source.resolve(), shared_binary)
    routed_gpus, shared_gpus, cards = choose_gpu_groups()
    print("ROUTED_GPU_UUID_ORDER", ",".join(routed_gpus), flush=True)
    print("SHARED_GPU_UUID_ORDER", ",".join(shared_gpus), flush=True)
    route_env = os.environ.copy()
    route_env["CUDA_VISIBLE_DEVICES"] = ",".join(routed_gpus)
    route_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    shared_env = os.environ.copy()
    shared_env["CUDA_VISIBLE_DEVICES"] = ",".join(shared_gpus)
    shared_env["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"

    metadata = {"model": str(MODEL), "model_revision": "236dfdf285828023ca3bcd3f37366c58a3469b13",
                "layers": layers, "input_mode": "per-layer external activations" if args.activations_dir else "same saved layer-0 sample reused independently for every layer",
                "shared_expert": "BF16 shared branch on sixth M10" if not args.no_shared else "disabled",
                "top_k": TOP_K, "routed_workers": WORKERS, "routed_slice_width": SLICE,
                "shared_workers": 0 if args.no_shared else SHARED_WORKERS,
                "shared_slice_width": None if args.no_shared else SHARED_SLICE,
                "kernel_rounds": args.rounds, "m10_card_groups": cards,
                "route_override_file": str(args.routes_json) if args.routes_json else None,
                "routed_gpus_uuid_order": routed_gpus,
                "shared_gpus_uuid_order": [] if args.no_shared else shared_gpus,
                "config_sha256": sha256(config_path), "index_sha256": sha256(model_index_path),
                "kernel_sha256": sha256(binary),
                "shared_kernel_sha256": None if args.no_shared else sha256(shared_binary)}
    (args.out / "run_metadata.json").write_text(json.dumps(metadata, indent=2) + "\n")
    rows = []
    with ExitStack() as stack:
        readers: dict[str, TensorReader] = {}
        for layer in layers:
            if args.activations_dir:
                x_path = args.activations_dir / f"layer_{layer:02d}.f32"
                x = np.fromfile(x_path, dtype=np.float32)
                if x.shape != (HIDDEN,):
                    raise RuntimeError(f"{x_path} must contain exactly 2560 FP32 values")
            else:
                x = sample.copy()
            ids, probs, logits, router_ids, route_source = selected_routes(
                layer, x, weight_map, readers, route_overrides.get(str(layer)))
            temp_parent = args.out / "prepared" if args.keep_prepared else None
            if temp_parent:
                work = temp_parent / f"layer_{layer:02d}"
            else:
                temp = tempfile.TemporaryDirectory(prefix=f"qwen-layer-{layer:02d}-")
                stack.callback(temp.cleanup)
                work = Path(temp.name)
            start = time.monotonic()
            expected, manifest = prepare_layer(layer, x, ids, probs, weight_map, readers, work, fp8_lut())
            manifest["route_source"] = route_source
            manifest["checkpoint_router_top10_ids"] = router_ids.tolist()
            manifest["explicit_expert_ids"] = ids.tolist()
            manifest["explicit_route_weights"] = probs.tolist()
            (work / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
            # Persist CPU references beside each run so a later full-layer
            # finalizer can distinguish the GPU result from its oracle.
            expected.astype(np.float32).tofile(args.out / f"layer_{layer:02d}.routed_reference.f32")
            if args.no_shared:
                shared_expected = None
                shared_gate = 0.0
                shared_work = None
            else:
                shared_work = work / "shared"
                shared_expected, shared_gate, _ = prepare_shared(
                    layer, x, weight_map, readers, shared_work)
                shared_expected.astype(np.float32).tofile(
                    args.out / f"layer_{layer:02d}.shared_reference.f32")
            extraction_s = time.monotonic() - start
            csv_path = args.out / f"layer_{layer:02d}.csv"
            routed_path = args.out / f"layer_{layer:02d}.routed.f32"
            output_path = args.out / f"layer_{layer:02d}.f32"
            cmd = [str(binary), str(work), "fp8", str(WORKERS), str(args.rounds), str(csv_path), str(routed_path)]
            print(f"LAYER {layer:02d} TOP10 {ids.tolist()} EXTRACT_S {extraction_s:.3f}", flush=True)
            shared_proc = None
            shared_output = args.out / f"layer_{layer:02d}.shared_partials.f32"
            if not args.no_shared:
                shared_csv = args.out / f"layer_{layer:02d}.shared.csv"
                shared_cmd = [str(shared_binary), str(shared_work), str(SHARED_WORKERS),
                              str(args.rounds), str(shared_csv), str(shared_output)]
                shared_proc = subprocess.Popen(shared_cmd, env=shared_env, text=True,
                                               stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
            proc = subprocess.run(cmd, env=route_env, text=True, capture_output=True)
            (args.out / f"layer_{layer:02d}.log").write_text(proc.stdout + proc.stderr)
            if proc.returncode:
                print(proc.stdout, end="")
                print(proc.stderr, file=sys.stderr, end="")
                raise RuntimeError(f"layer {layer} kernel/reference run failed with {proc.returncode}")
            routed_got = np.fromfile(routed_path, dtype=np.float32)
            if routed_got.shape != (HIDDEN,) or not np.allclose(routed_got, expected, rtol=1e-3, atol=2e-7, equal_nan=False):
                raise RuntimeError(f"layer {layer} aggregate output failed CPU reference comparison")
            shared_ok = None
            if shared_proc is not None:
                shared_text = shared_proc.communicate()[0]
                (args.out / f"layer_{layer:02d}.shared.log").write_text(shared_text)
                if shared_proc.returncode:
                    print(shared_text, file=sys.stderr, end="")
                    raise RuntimeError(f"layer {layer} shared kernel failed with {shared_proc.returncode}")
                partials = np.fromfile(shared_output, dtype=np.float32)
                if partials.shape != (SHARED_WORKERS * HIDDEN + 1,):
                    raise RuntimeError(f"layer {layer}: wrong shared partial output size {partials.shape}")
                gpu_shared_gate = float(partials[-1])
                partials = partials[:-1].reshape(SHARED_WORKERS, HIDDEN)
                if not np.isclose(gpu_shared_gate, shared_gate, rtol=1e-5, atol=1e-7):
                    raise RuntimeError(f"layer {layer} shared gate mismatch: GPU={gpu_shared_gate}, CPU={shared_gate}")
                shared_got = partials.sum(axis=0) * np.float32(gpu_shared_gate)
                shared_ok = bool(np.allclose(shared_got, shared_expected, rtol=1e-3, atol=2e-5))
                if not shared_ok:
                    err = np.abs(shared_got - shared_expected)
                    raise RuntimeError(f"layer {layer} shared output mismatch; max_abs={err.max():.8g}, bad={np.count_nonzero(err > 2e-5 + np.abs(shared_expected)*1e-3)}")
                shared_got.astype(np.float32).tofile(args.out / f"layer_{layer:02d}.shared.f32")
                got = routed_got + shared_got
            else:
                got = routed_got
            expected_total = expected + (shared_expected if shared_expected is not None else 0.0)
            expected_total.astype(np.float32).tofile(
                args.out / f"layer_{layer:02d}.moe_reference.f32")
            if not np.allclose(got, expected_total, rtol=1e-3, atol=2e-5, equal_nan=False):
                raise RuntimeError(f"layer {layer} combined routed+shared output failed CPU reference comparison")
            got.astype(np.float32).tofile(output_path)
            rows.append({"layer": layer, "expert_ids": json.dumps(ids.tolist()),
                         "routes": json.dumps([float(v) for v in probs]),
                         "extract_seconds": extraction_s, "routed_matches_cpu": True,
                         "shared_matches_cpu": shared_ok, "combined_matches_cpu": True,
                         "output_file": str(output_path), "log_file": str(args.out / f"layer_{layer:02d}.log")})
            (args.out / f"layer_{layer:02d}.json").write_text(json.dumps(manifest, indent=2) + "\n")
            print(f"LAYER {layer:02d} PASS; routed{' + shared' if shared_ok is not None else ''} output written to {output_path}", flush=True)
        for reader in readers.values():
            stack.callback(reader.close)
    with (args.out / "layers.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    print(f"PASS: {len(rows)} isolated routed{' + shared' if not args.no_shared else ''} expert layers; full transformer forward is not included")
    return 0


if __name__ == "__main__":
    sys.exit(main())
