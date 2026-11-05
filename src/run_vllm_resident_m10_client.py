#!/usr/bin/env python3
"""Run one vLLM client against an already-loaded M10 resident daemon.

This client owns only the RTX/vLLM lifecycle. It must never stop the remote
M10 PLE or expert services. Starting and exiting this program can therefore be
repeated while M10 CUDA allocations remain resident in qwen38_resident_daemon.
"""
from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import subprocess
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from qwen38_resident_daemon import send_control



def _q38_close_peer_adapter(worker=None) -> str:
    """Runs inside the vLLM EngineCore worker (collective_rpc): close the RTX peer adapter
    explicitly (park workers, restore every route, unregister from the daemon) before vLLM's own
    shutdown, which SIGKILLs the engine after VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS."""
    import sys as _s
    mod = _s.modules.get("sitecustomize")
    lazy = getattr(mod, "_adapter", None)
    if lazy is None:
        return "no_adapter"
    if getattr(lazy, "_inner", None) is None:
        return "never_opened"
    lazy.close()
    return "closed"


def _q38_graph_ceiling(worker=None, iters: int = 200) -> dict:
    """Runs inside the vLLM worker (collective_rpc) AFTER all generations: replay the captured FULL
    decode graph back to back to measure the step time without vLLM's host loop (the ceiling of a
    GPU-resident graph loop). It repeats the last token: it rewrites one KV slot and advances the
    GDN state of the finished request, so it must only run after generation. Three timings (ms/step):
      graph_only            graph replays, one sync at the end
      graph_lm_argmax       + compute_logits (LM head) + argmax per step, one sync at the end
      graph_lm_argmax_sync  the same with a host sync per step (as a per-step token read-back)"""
    import os as _os
    import sys as _s
    import time as _t
    import torch as _torch
    mr = worker.model_runner
    cgm = getattr(mr, "cudagraph_manager", None)
    graphs = getattr(cgm, "graphs", {}) if cgm is not None else {}
    full = [(d, g) for d, g in graphs.items() if getattr(d, "num_tokens", None) == 1 and "FULL" in str(getattr(d, "cg_mode", ""))]
    if not full:
        return {"error": f"no FULL batch-1 decode graph (have {[str(d) for d in graphs]})"}
    desc, graph = full[0]
    hs = cgm.hidden_states[:1]
    model = mr.model
    compute_logits = getattr(model, "compute_logits", None)
    out = {"descriptor": str(desc), "graph_type": type(graph).__name__, "iters": int(iters)}

    def timed(step, sync_each):
        _torch.cuda.synchronize()
        t0 = _t.perf_counter()
        for _ in range(iters):
            step()
            if sync_each:
                _torch.cuda.synchronize()
        _torch.cuda.synchronize()
        return round(1e3 * (_t.perf_counter() - t0) / iters, 3)

    def graph_lm():
        graph.replay()
        logits = compute_logits(hs)
        _torch.argmax(logits, dim=-1)

    out["graph_only_ms"] = timed(graph.replay, False)
    steps = int(_os.environ.get("Q38_GRAPH_BREAKDOWN_STEPS", "0"))
    inner = getattr(getattr(_s.modules.get("sitecustomize"), "_adapter", None), "_inner", None)
    if steps > 0 and compute_logits is not None and inner is not None and hasattr(inner, "timing_enable"):
        _torch.cuda.synchronize()
        inner.timing_enable(steps * 64 + 64)
        wt = os.environ.get("Q38_WORKER_TIMING") == "1" and hasattr(inner, "worker_timing_enable")
        if wt:
            inner.worker_timing_enable()
        for _ in range(steps):
            graph_lm()
        rtx_recs = inner.timing_read(steps * 64 + 64)
        out["breakdown"] = _q38_breakdown(rtx_recs)
        if wt:
            out["worker_breakdown"] = _q38_worker_breakdown(rtx_recs, inner.worker_timing_read())
    if compute_logits is not None:
        out["graph_lm_argmax_ms"] = timed(graph_lm, False)
        out["graph_lm_argmax_sync_ms"] = timed(graph_lm, True)
    lazy = getattr(_s.modules.get("sitecustomize"), "_adapter", None)
    try:
        if lazy is not None:
            lazy.check_error()
        out["exchange_errors"] = None
    except Exception as exc:          # report; the run's own checks decide pass/fail
        out["exchange_errors"] = str(exc)
    for k in ("graph_only_ms", "graph_lm_argmax_ms", "graph_lm_argmax_sync_ms"):
        if out.get(k):
            out[k.replace("_ms", "_tok_per_s")] = round(1e3 / out[k], 2)
    print("Q38_GRAPH_CEILING," + ",".join(f"{k},{v}" for k, v in out.items()), flush=True)
    return out


def _q38_breakdown(recs: list[dict]) -> dict:
    """Per-exchange timing of replayed decode steps (records from the peer library). A step ends with
    the LM-head exchange (layer 512). Per exchange key (MoE layer 0-47, 'ple', 'lm_head'), medians in us:
      rtx_before  previous exchange done -> this exchange begins (RTX compute in between)
      publish     begin -> sequence word published (validation + multicast payload)
      round_trip  sequence published -> all 24 workers acknowledged (PCIe + M10 compute)
      combine     acknowledged -> result written on the RTX
    Totals per step (median of per-step sums) and the step time (LM-head done -> next LM-head done)."""
    import statistics as st
    steps, cur = [], []
    for r in recs:
        cur.append(r)
        if r["layer"] == 512:
            steps.append(cur); cur = []
    steps = steps[1:]                       # first step may lack its predecessor
    if not steps:
        return {"error": "no complete steps", "records": len(recs)}
    name = lambda l: "lm_head" if l == 512 else "ple" if l == 256 else f"moe_{l:02d}"
    per, totals, step_ns = {}, {"rtx_before": [], "publish": [], "round_trip": [], "combine": []}, []
    prev_done = None
    for stp in steps:
        sums = {k: 0 for k in totals}
        for r in stp:
            if r["skip"] or not r["t_done"]:
                continue
            vals = {"rtx_before": (r["t_begin"] - prev_done) if prev_done else None,
                    "publish": r["t_seq"] - r["t_begin"], "round_trip": r["t_ack"] - r["t_seq"],
                    "combine": r["t_done"] - r["t_ack"]}
            d = per.setdefault(name(r["layer"]), {k: [] for k in totals})
            for k, v in vals.items():
                if v is not None and v >= 0:
                    d[k].append(v); sums[k] += v
            if r["layer"] == 512 and prev_done is not None:
                pass
            prev_done = r["t_done"]
        for k in totals:
            totals[k].append(sums[k])
    lm_done = [s[-1]["t_done"] for s in steps if s[-1]["t_done"]]
    step_ns = [b - a for a, b in zip(lm_done, lm_done[1:])]
    med = lambda xs: round(st.median(xs) / 1e3, 1) if xs else None
    out = {"steps": len(steps), "records": len(recs),
           "step_us": med(step_ns),
           "per_step_us": {k: med(v) for k, v in totals.items()},
           "per_exchange_us": {k: {m: med(v) for m, v in d.items()} for k, d in sorted(per.items())}}
    print("Q38_BREAKDOWN_TOTALS,steps,{},step_us,{},{}".format(out["steps"], out["step_us"],
          ",".join(f"{k},{v}" for k, v in out["per_step_us"].items())), flush=True)
    for k, d in out["per_exchange_us"].items():
        print("Q38_BREAKDOWN," + k + "," + ",".join(f"{m},{v}" for m, v in d.items()), flush=True)
    return out


def _q38_worker_breakdown(rtx_recs: list[dict], wrecs: list[dict]) -> dict:
    """Join RTX exchange records with M10 worker records by sequence (MoE layers only). Per exchange:
    rtx_round_trip (RTX: sequence published -> all acknowledged), and per worker kind the busy time
    seen -> done (M10 clock) split into validate (seen->checked), gate_up (checked->gateup),
    down (gateup->down), finish (down->done: leader aggregation / result write). critical = the
    largest busy time of the 24 workers; transport = round trip - critical (PCIe, doorbell, ack)."""
    import statistics as st
    by_seq: dict[int, list[dict]] = {}
    for w in wrecs:
        by_seq.setdefault(w["seq"], []).append(w)
    rows = []
    for r in rtx_recs:
        if r["layer"] >= 48 or r["skip"] or r["seq"] not in by_seq or not r["t_ack"]:
            continue
        ws = by_seq[r["seq"]]
        busy = [w["t_done"] - w["t_seen"] for w in ws if w["t_done"] >= w["t_seen"]]
        routed = [w for w in ws if w["kind"] == 1 and w["t_down"]]
        if not busy or not routed:
            continue
        rows.append({"rtt": r["t_ack"] - r["t_seq"], "critical": max(busy), "workers": len(ws),
                     "routed_validate": st.median(w["t_checked"] - w["t_seen"] for w in routed),
                     "routed_gate_up": st.median(w["t_gateup"] - w["t_checked"] for w in routed),
                     "routed_down": st.median(w["t_down"] - w["t_gateup"] for w in routed),
                     "routed_finish_leader": st.median([w["t_done"] - w["t_down"] for w in routed if w["role"] == 1] or [0]),
                     "routed_finish_peer": st.median([w["t_done"] - w["t_down"] for w in routed if w["role"] == 2] or [0]),
                     "shared_busy": max([w["t_done"] - w["t_seen"] for w in ws if w["kind"] == 2] or [0])})
    if not rows:
        return {"error": "no joined MoE exchanges", "worker_records": len(wrecs)}
    med = lambda k: round(st.median(x[k] for x in rows) / 1e3, 1)
    out = {"exchanges": len(rows), "workers_per_exchange": st.median(x["workers"] for x in rows),
           "median_us": {k: med(k) for k in rows[0] if k != "workers"}}
    out["median_us"]["transport_rtt_minus_critical"] = round(out["median_us"]["rtt"] - out["median_us"]["critical"], 1)
    print("Q38_WORKER_BREAKDOWN,exchanges,{},{}".format(out["exchanges"],
          ",".join(f"{k},{v}" for k, v in out["median_us"].items())), flush=True)
    return out


def _q38_batch_timing(worker=None) -> dict:
    """Runs inside the vLLM worker (collective_rpc) AFTER all generations (Q38_BATCH_TIMING=1, needs
    Q38_RTX_PEER_MOE_BATCH=1): the MoE exchange alone, batched (mode 8) vs per row (mode 1), on the same
    synthetic rows (seeded: BF16 x ~ N(0, 0.05), 10 distinct random experts per row, positive routes
    summing to 1), all 48 layers per pass. Per R in Q38_BATCH_TIMING_ROWS (default "1 8 16 32 64"):
      ms_per_layer_call   wall time of one layer's MoE for R rows (median of Q38_BATCH_TIMING_PASSES passes
                          after one warm-up pass; one sync per pass)
      rtx_us              RTX timing ring, medians per exchange: publish (t_pub-t_begin), doorbell
                          (t_seq-t_pub), wait for 24 acks (t_ack-t_seq), combine (t_done-t_ack), total,
                          gap to the next exchange (launch / host)
      worker_us (batch)   M10 stamps per exchange: routed validate / stage+gate_up / down / finish (k-sum,
                          card reduce, flag, ack) by role, shared busy; critical = max busy of 24 workers
    Also checks that the last layer's outputs of both paths are bit-identical.
    Q38_BATCH_TIMING_PATHS (default "batch per_row") may add "pipe": the pipelined call (mode 9, 32-row chunks,
    one enqueue per layer call; worker kinds 5/6); every path is compared with the first one."""
    import os as _os
    import statistics as st
    import sys as _s
    import time as _t
    import torch as _torch
    inner = getattr(getattr(_s.modules.get("sitecustomize"), "_adapter", None), "_inner", None)
    if inner is None or not hasattr(inner, "moe_batch_min_rows"):
        return {"error": "no RTX peer adapter with batched MoE"}
    if not inner.moe_batch:
        return {"error": "Q38_RTX_PEER_MOE_BATCH is not enabled"}
    rows_list = [int(x) for x in _os.environ.get("Q38_BATCH_TIMING_ROWS", "1 8 16 32 64").replace(",", " ").split()]   # spaces or commas (pueue drops quotes)
    passes = int(_os.environ.get("Q38_BATCH_TIMING_PASSES", "3"))
    paths = _os.environ.get("Q38_BATCH_TIMING_PATHS", "batch per_row").replace(",", " ").split()
    gen = _torch.Generator(device="cpu").manual_seed(1234)
    med = lambda xs: round(st.median(xs) / 1e3, 1) if xs else None

    def _q38_ack_profile_summary(recs, med):
        # Q38_RTX_PEER_ACK_PROFILE=1: pad = (t_all - t_data) ns | last header << 24; data = 5 leaders + 4 shared
        lag = [r["pad"] & 0xFFFFFF for r in recs if r.get("pad")]
        last = [r["pad"] >> 24 for r in recs if r.get("pad")]
        hist = {}
        for w in last: hist[w] = hist.get(w, 0) + 1
        data_ready = [r["t_ack"] - r["t_seq"] - (r["pad"] & 0xFFFFFF) for r in recs if r.get("pad")]
        return {"ack_profiled": len(lag), "data9_ready": med(data_ready), "peers_after_data9": med(lag),
                "peers_after_data9_max": round(max(lag) / 1e3, 1) if lag else None,
                "peers_after_data9_zero": sum(1 for x in lag if x == 0),
                "last_header_hist": dict(sorted(hist.items(), key=lambda kv: -kv[1])[:6])}

    def rtx_phases(recs):
        recs = [r for r in recs if r["layer"] < 48 and not r["skip"] and r["t_done"]]
        gaps = [b["t_begin"] - a["t_done"] for a, b in zip(recs, recs[1:]) if b["t_begin"] >= a["t_done"]]
        return {"exchanges": len(recs), "publish": med([r["t_pub"] - r["t_begin"] for r in recs]),
                "doorbell": med([r["t_seq"] - r["t_pub"] for r in recs]),
                "wait_acks": med([r["t_ack"] - r["t_seq"] for r in recs]),
                "combine": med([r["t_done"] - r["t_ack"] for r in recs]),
                "total": med([r["t_done"] - r["t_begin"] for r in recs]), "gap_next": med(gaps),
                **(_q38_ack_profile_summary(recs, med) if any(r.get("pad") for r in recs) else {})}

    def worker_phases(wrecs, routed_kind, shared_kind):
        # mode 8: routed kind 3, shared kind 4; mode 1: routed kind 1, shared kind 2 (per exchange = per row)
        by_seq = {}
        for w in wrecs:
            if w["kind"] in (routed_kind, shared_kind) and w["t_done"] >= w["t_seen"]:
                by_seq.setdefault(w["seq"], []).append(w)
        routed = [w for ws in by_seq.values() for w in ws if w["kind"] == routed_kind and w["t_down"]]
        # Q38_RTX_PEER_MR_PROFILE=1 (mode 10 leaders): t_checked carries the packed finish phases, not a timestamp
        isprof = lambda w: routed_kind == 7 and (w["t_checked"] >> 56) == 0xA5
        prof = [w for w in routed if isprof(w)]
        ph = lambda w, sh: ((w["t_checked"] >> sh) & 0x3FFF) * 32
        shared = [w for ws in by_seq.values() for w in ws if w["kind"] == shared_kind]
        return {"exchanges": len(by_seq),
                # mode 8 (since step 2): t_checked is stamped after the host x staging, so the first field is check + stage
                ("routed_check_stage" if routed_kind in (3, 5) else "routed_validate"): med([w["t_checked"] - w["t_seen"] for w in routed if not isprof(w)]),
                ("routed_gate_up" if routed_kind in (3, 5) else "routed_stage_gate_up"): med([w["t_gateup"] - w["t_checked"] for w in routed if not isprof(w)]),
                **({"leader_prof_n": len(prof), "leader_prof_reads": med([ph(w, 42) for w in prof]),
                    "leader_prof_local_write": med([ph(w, 28) for w in prof]), "leader_prof_rtx_write": med([ph(w, 14) for w in prof]),
                    "leader_prof_fence": med([ph(w, 0) for w in prof])} if prof else {}),
                # mode 8 t_extra: high 32 = validation end - t_seen, low 32 = block 0's list build (ns); stage = the rest
                **({"routed_validate8": med([(w.get("t_extra", 0) >> 32) for w in routed]),
                    "routed_stage8": med([w["t_checked"] - w["t_seen"] - (w.get("t_extra", 0) >> 32) for w in routed]),
                    "routed_lists8": med([(w.get("t_extra", 0) & 0xFFFFFFFF) for w in routed])} if routed_kind in (3, 5) else {}),
                "routed_down": med([w["t_down"] - w["t_gateup"] for w in routed]),
                "routed_finish_leader": med([w["t_done"] - w["t_down"] for w in routed if w["role"] == 1]),
                "routed_finish_peer": med([w["t_done"] - w["t_down"] for w in routed if w["role"] == 2]),
                # mode 1 / 10 finish split (2026-11-04, t_extra): leader flags seen, leader / peer block-0 sum or push written,
                # the rest of the leader finish = grid barrier (other blocks) + ack
                **({"leader_wait_flags": med([w["t_extra"] >> 32 for w in routed if w["role"] == 1]),
                    # NOTE (2026-11-04, T284): t_extra low 32 = t_sm - t_down, i.e. it INCLUDES the flag wait; the name is kept
                    # for comparability with earlier runs. leader_sum_write_only_b0 = the leader's own sum + write + fence.
                    "leader_sum_write_b0": med([w["t_extra"] & 0xFFFFFFFF for w in routed if w["role"] == 1]),
                    "leader_sum_write_only_b0": med([(w["t_extra"] & 0xFFFFFFFF) - (w["t_extra"] >> 32) for w in routed
                                                     if w["role"] == 1 and (w["t_extra"] >> 32)]),
                    "leader_barrier_ack": med([w["t_done"] - w["t_down"] - (w["t_extra"] & 0xFFFFFFFF) for w in routed if w["role"] == 1]),
                    "peer_push_b0": med([w["t_extra"] & 0xFFFFFFFF for w in routed if w["role"] == 2])}
                   if routed_kind in (1, 7) else {}),
                "shared_busy": med([w["t_done"] - w["t_seen"] for w in shared]),
                # mode 8 shared (kind 4): t_checked = after x staging, t_gateup = after hidden (gate/up + gate), t_down = after down
                **({"shared_stage8": med([w["t_checked"] - w["t_seen"] for w in shared]),
                    "shared_hidden8": med([w["t_gateup"] - w["t_checked"] for w in shared]),
                    "shared_down8": med([w["t_down"] - w["t_gateup"] for w in shared]),
                    "shared_finish8": med([w["t_done"] - w["t_down"] for w in shared])} if shared_kind in (4, 6) else {}),
                "critical": med([max(w["t_done"] - w["t_seen"] for w in ws) for ws in by_seq.values()]),
                # pipelined: worker time per chunk from one request's start to the next one's (all 24 workers)
                **({"chunk_period": med([b - a for a, b in zip(starts, starts[1:]) if 0 < b - a < 50_000_000])}
                   if (starts := sorted(min(w["t_seen"] for w in ws) for ws in by_seq.values())) and routed_kind == 5 else {})}

    # Stress identity check (2026-11-03, after the i8/i10 token changes): Q38_BATCH_STRESS=N repeats N passes of all 48
    # layers per path in Q38_BATCH_STRESS_PATHS (default "pipe batch") and compares EVERY layer's output rows with a
    # per-row (mode 1) reference, by a 64-bit checksum per row (the timing check above only sees the last layer).
    # The input differs per layer (h + layer * 2^-6, BF16) so a stale read of an earlier request is visible.
    # Reports mismatching (pass, layer, row) with the 32-row chunk and the row inside it.
    stress = int(_os.environ.get("Q38_BATCH_STRESS", "0"))
    stress_paths = _os.environ.get("Q38_BATCH_STRESS_PATHS", "pipe batch").replace(",", " ").split()

    def stress_check(R, h, ids, routes):
        wts = _torch.arange(1, 2561, device="cuda:0", dtype=_torch.int64)
        # the adapter returns the input dtype (BF16 here): checksum the raw 16- or 32-bit words of each row (t22 failed on this)
        bits = lambda o: o.contiguous().view(_torch.int16 if o.element_size() == 2 else _torch.int32).to(_torch.int64)
        hl = lambda layer: (h.float() + layer * 2.0 ** -6).to(_torch.bfloat16)
        rowsum = lambda o: (bits(o) * wts).sum(1)
        def run(path, npass):
            inner.moe_batch = path in ("batch", "pipe")
            inner.moe_batch_pipe = path == "pipe"
            inner.moe_multirow = path == "multi"
            inner.moe_batch_min_rows = inner.moe_batch_pipe_min_rows = 1
            cs = _torch.empty((npass, 48, R), dtype=_torch.int64, device="cuda:0")
            for it in range(npass):
                for layer in range(48):
                    cs[it, layer] = rowsum(inner.forward(layer, hl(layer), ids, routes))
            _torch.cuda.synchronize()
            inner.check_error()
            return cs
        ref = run("per_row", 1)[0]
        # reference checksum of all 48 layers' outputs (seeded inputs, bit-reproducible M10 path): equal across runs and
        # transports (e.g. with and without the outer PEX8748 multicast group, 2026-11-04)
        rep = {"passes": stress, "rows": R, "ref_checksum": int(ref.sum().item()) & 0xFFFFFFFFFFFF}
        for path in stress_paths:
            t0 = _t.perf_counter()
            bad = (run(path, stress) != ref.unsqueeze(0)).nonzero().tolist()     # [pass, layer, row]
            rep[path] = {"seconds": round(_t.perf_counter() - t0, 1), "bad_rows": len(bad),
                         "bad_calls": len({(p_, l_) for p_, l_, _ in bad}), "first": bad[:24],
                         "chunk32_hist": dict(sorted(__import__("collections").Counter(r_ // 32 for _, _, r_ in bad).items())),
                         "row_in_chunk_hist": dict(sorted(__import__("collections").Counter(r_ % 32 for _, _, r_ in bad).items())),
                         "layer_hist": dict(sorted(__import__("collections").Counter(l_ for _, l_, _ in bad).items()))}
        return rep

    saved = (inner.moe_batch, inner.moe_batch_min_rows, getattr(inner, "moe_batch_pipe", False),
             getattr(inner, "moe_batch_pipe_min_rows", 33), getattr(inner, "moe_multirow", False))
    out = {"passes": passes, "rows": {}}
    try:
        for R in rows_list:
            h = (_torch.randn(R, 2560, generator=gen) * 0.05).to(_torch.bfloat16).to("cuda:0")
            # Q38_BATCH_TIMING_IDS=<file.pt>: real router ids (int32 [N, 10], adapter Q38_BATCH_REUSE_DIR dump),
            # first R rows; default random distinct experts per row
            ids_file = _os.environ.get("Q38_BATCH_TIMING_IDS")
            if ids_file:
                ids = _torch.load(ids_file)[:R].to(_torch.int32).to("cuda:0")
                assert ids.shape[0] == R, f"{ids_file} has fewer than {R} rows"
            else:
                ids = _torch.stack([_torch.randperm(512, generator=gen)[:10] for _ in range(R)]).to(_torch.int32).to("cuda:0")
            wgt = _torch.rand(R, 10, generator=gen) + 0.1
            routes = (wgt / wgt.sum(1, keepdim=True)).to("cuda:0")
            res, last = {"distinct_experts": int(_torch.unique(ids).numel())}, {}
            for path in paths:
                if path == "multi" and not getattr(inner, "moe_multirow_min", 2) <= R <= getattr(inner, "moe_multirow_max", 2):
                    continue                      # the multi-row exchange takes 2..max rows
                inner.moe_batch = path in ("batch", "pipe")
                inner.moe_batch_pipe = path == "pipe"
                inner.moe_multirow = path == "multi"
                inner.moe_batch_min_rows = inner.moe_batch_pipe_min_rows = 1
                times, recs, wrecs = [], [], []
                # the stamped pass must fit the 256-record worker rings (512 before 2026-11-04): exchanges per layer =
                # ceil(R/64) batched, ceil(R/32) pipelined, ceil(R/2) multi-row, R per row
                per_layer = {"batch": -(-R // 64), "pipe": -(-R // 32), "multi": 1}.get(path, R)
                ring_layers = max(1, min(48, 240 // per_layer))
                for p in range(passes + 2):
                    ring = p == passes + 1
                    _torch.cuda.synchronize()
                    if ring:
                        inner.timing_enable(R * 48 + 64)
                        inner.worker_timing_enable()
                    t0 = _t.perf_counter()
                    for layer in range(ring_layers if ring else 48):
                        o = inner.forward(layer, h, ids, routes)
                    _torch.cuda.synchronize()
                    dt = _t.perf_counter() - t0
                    if ring:
                        recs = inner.timing_read(R * 48 + 64)
                        wrecs = inner.worker_timing_read()
                    elif p > 0:
                        times.append(dt)
                inner.check_error()
                if ring_layers < 48:    # o of the stamped pass is layer ring_layers-1; recompute layer 47 (t20c: batch / pipe
                                        # at R=1024 compared layers 29 / 14 with per-row layer 47)
                    o = inner.forward(47, h, ids, routes)
                    _torch.cuda.synchronize()
                last[path] = o.float()
                res[path] = {"ms_per_layer_call": round(st.median(times) * 1e3 / 48, 3),
                             "us_per_row_per_layer": round(st.median(times) * 1e6 / 48 / R, 1),
                             "rtx_us": rtx_phases(recs)}
                res[path]["worker_us"] = worker_phases(wrecs, *{"batch": (3, 4), "pipe": (5, 6), "multi": (7, 8)}.get(path, (1, 2)))
                res[path]["worker_stamped_layers"] = ring_layers
            ref = paths[0]
            res["identical_to_" + ref] = {p: bool(_torch.equal(last[p], last[ref])) for p in paths[1:] if p in last}
            res["max_abs_diff_to_" + ref] = {p: float((last[p] - last[ref]).abs().max()) for p in paths[1:] if p in last}
            if "multi" in last and "per_row" in last:
                res["speedup_multi_vs_per_row"] = round(res["per_row"]["ms_per_layer_call"] / res["multi"]["ms_per_layer_call"], 3)
            if "batch" in last and "per_row" in last:
                res["bit_identical_last_layer"] = bool(_torch.equal(last["batch"], last["per_row"]))
                res["speedup_batch_vs_per_row"] = round(res["per_row"]["ms_per_layer_call"] / res["batch"]["ms_per_layer_call"], 3)
            if "pipe" in last and "batch" in last:
                res["speedup_pipe_vs_batch"] = round(res["batch"]["ms_per_layer_call"] / res["pipe"]["ms_per_layer_call"], 3)
            out["rows"][R] = res
            if stress:
                res["stress"] = stress_check(R, h, ids, routes)
                print(f"Q38_BATCH_STRESS,R,{R},{res['stress']}", flush=True)
            print(f"Q38_BATCH_TIMING,R,{R}," + ",".join(f"{p}_ms,{res[p]['ms_per_layer_call']}" for p in paths if p in res)
                  + f",identical_to_{ref},{res['identical_to_' + ref]},speedup_batch_vs_per_row,{res.get('speedup_batch_vs_per_row')}"
                  + f",speedup_pipe_vs_batch,{res.get('speedup_pipe_vs_batch')}"
                  + f",speedup_multi_vs_per_row,{res.get('speedup_multi_vs_per_row')}"
                  + "".join(f",rtx_{p},{res[p]['rtx_us']},workers_{p},{res[p].get('worker_us')}" for p in paths if p in res), flush=True)
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        inner.moe_batch, inner.moe_batch_min_rows, inner.moe_batch_pipe, inner.moe_batch_pipe_min_rows, inner.moe_multirow = saved
    return out


def _q38_lm2_check(worker=None) -> dict:
    """Q38_LM2_CHECK=N (2026-11-04), inside the vLLM worker after the generations: N random BF16 row groups through the
    M10 LM head as one multi-row exchange and as single-row exchanges, for each row count in Q38_LM2_CHECK_ROWS
    (default "2"). Must hold: identical log-sum-exps (bits), and every logit the multi-row call returns (its K may be
    smaller: (2560 - n) // n - 1) equals the single-row logit; candidates beyond the smaller K are -inf. "argmax_bad"
    counts rows whose argmax differs. Times both forms (median, one sync per call)."""
    import os as _os
    import statistics as st
    import sys as _s
    import time as _t
    import torch as _torch
    inner = getattr(getattr(_s.modules.get("sitecustomize"), "_adapter", None), "_inner", None)
    if inner is None or not getattr(inner, "lm_head_m10", False):
        return {"error": "no RTX peer adapter with the M10 LM head"}
    n = int(_os.environ.get("Q38_LM2_CHECK", "0"))
    row_counts = [int(x) for x in _os.environ.get("Q38_LM2_CHECK_ROWS", "2").replace(",", " ").split()]
    gen = _torch.Generator(device="cpu").manual_seed(4321)
    saved = _os.environ.get("Q38_M10_LM_HEAD_ROWS2")
    out = {"groups": n}
    try:
        for nr in row_counts:
            res = {"bad_lse": 0, "bad_logits": 0, "argmax_bad": 0}
            t1, t2 = [], []
            for i in range(n):
                x = (_torch.randn(nr, 2560, generator=gen) * 1.5).to(_torch.bfloat16).to("cuda:0")
                _os.environ["Q38_M10_LM_HEAD_ROWS2"] = "0"
                _torch.cuda.synchronize(); t0 = _t.perf_counter()
                a = inner.lm_head_logits(x); la = inner.last_lm_lse.clone()
                _torch.cuda.synchronize(); t1.append(_t.perf_counter() - t0)
                _os.environ["Q38_M10_LM_HEAD_ROWS2"] = "1"
                t0 = _t.perf_counter()
                b = inner.lm_head_logits(x); lb = inner.last_lm_lse.clone()
                _torch.cuda.synchronize(); t2.append(_t.perf_counter() - t0)
                res["bad_lse"] += int((la.view(_torch.int32) != lb.view(_torch.int32)).sum())
                res["lse_checksum"] = (res.get("lse_checksum", 0) + int(la.view(_torch.int32).to(_torch.int64).sum().item())) & 0xFFFFFFFFFFFF
                fin = _torch.isfinite(b.float())
                res["bad_logits"] += int(((a.view(_torch.int16) != b.view(_torch.int16)) & fin).any(dim=1).sum())
                res["argmax_bad"] += int((a.float().argmax(1) != b.float().argmax(1)).sum())
            res["ms_single_rows"] = round(st.median(t1) * 1e3, 3)
            res["ms_one_multi"] = round(st.median(t2) * 1e3, 3)
            out[f"rows{nr}"] = res
        inner.check_error()
    except Exception as exc:
        out["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if saved is None:
            _os.environ.pop("Q38_M10_LM_HEAD_ROWS2", None)
        else:
            _os.environ["Q38_M10_LM_HEAD_ROWS2"] = saved
    print("Q38_LM2_CHECK," + ",".join(f"{k},{v}" for k, v in out.items()), flush=True)
    return out


def discover_rtx_uuid() -> str:
    """RTX UUID from Q38_RTX_UUID or a live sysfs/nvidia-smi scan (never hardcoded)."""
    env = os.environ.get("Q38_RTX_UUID")
    if env:
        return env
    script = Path(__file__).with_name("qwen38_pci_discover.py")
    out = subprocess.run(["python3", str(script), "--json"], capture_output=True, text=True, timeout=60)
    if out.returncode != 0:
        raise RuntimeError(f"RTX discovery failed: {out.stderr.strip()}")
    return json.loads(out.stdout)["rtx"]["uuid"]


def atomic_json(path: Path, value: dict[str, Any]) -> None:
    tmp = path.with_name(path.name + f".{os.getpid()}.tmp")
    tmp.write_text(json.dumps(value, indent=2) + "\n")
    tmp.replace(path)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(8 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--moe-socket", type=Path, required=True)
    parser.add_argument("--ple-socket", type=Path, required=True)
    parser.add_argument("--control-socket", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prompt-file", type=Path)
    parser.add_argument("--max-model-len", type=int, default=384)
    parser.add_argument("--max-new-tokens", type=int, default=32)
    parser.add_argument("--cpu-offload-gb", type=float, default=6.0)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.78)
    parser.add_argument("--kv-cache-memory-bytes", type=int, default=0,
                        help="fixed KV-cache reservation in bytes; when >0 vLLM ignores the utilization percentage")
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--direct-moe", action="store_true",
                        help="send MoE requests directly over PEX/BAR1 using daemon-owned mappings")
    parser.add_argument("--rtx-peer-moe", action="store_true",
                        help="RTX CUDA kernels publish to and reduce from resident M10 BAR1 mailboxes")
    parser.add_argument("--rtx-peer-library", type=Path,
                        help="CUDA shared library built by build_qwen38_vllm_rtx_peer.sh")
    parser.add_argument("--direct-moe-library", type=Path,
                        help="ABI-matched direct mailbox/PLE library; defaults to the daemon's recorded library")
    parser.add_argument("--direct-ple-read-width", type=int, choices=(1, 4, 8, 16, 32),
                        default=32,
                        help="CPU load width in bytes for direct BAR1 PLE result collection")
    parser.add_argument("--park-workers-on-exit", action="store_true",
                        help="wake workers before generation and park after client cleanup")
    parser.add_argument("--disable-direct-cpu-mailbox", action="store_true",
                        help="use the BAR1 result-read fallback instead of the registered CPU mailbox")
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if args.direct_moe and args.rtx_peer_moe:
        parser.error("--direct-moe and --rtx-peer-moe are mutually exclusive")
    if args.rtx_peer_moe and args.rtx_peer_library is None:
        parser.error("--rtx-peer-moe requires --rtx-peer-library")
    if args.rtx_peer_moe and args.disable_direct_cpu_mailbox:
        parser.error("--rtx-peer-moe requires the daemon-registered mailbox for its M10 return aliases")

    model = args.model.resolve()
    out = args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    state_path = out / "resident_vllm_client_result.json"
    result: dict[str, Any] = {
        "status": "RUNNING", "pid": os.getpid(), "model": str(model),
        "started_unix": time.time(), "moe_socket": str(args.moe_socket),
        "ple_socket": str(args.ple_socket),
    }
    atomic_json(state_path, result)

    try:
        manager_before = send_control(args.control_socket, "STATUS")
        if manager_before.get("status") != "READY" or manager_before.get("layers_ready") not in (48, 49):   # 49 = with the MTP layer
            raise RuntimeError(f"M10 resident service is not fully ready: {manager_before}")
        client_model_hash = sha256_file(model / "model.safetensors.index.json")
        if manager_before.get("model_index_sha256") != client_model_hash:
            raise RuntimeError("resident M10 service model index does not match this vLLM client")
        direct_setup = manager_before.get("direct_moe_setup")
        if args.direct_moe or args.rtx_peer_moe:
            if not isinstance(direct_setup, dict) or direct_setup.get("status") != "PASS":
                raise RuntimeError("resident daemon has no validated direct-MoE BAR1 setup")
            direct_library = (args.direct_moe_library.resolve()
                              if args.direct_moe_library else
                              Path(direct_setup["library"]).resolve())
            direct_groups = Path(direct_setup["groups_tsv"]).resolve()
            direct_maps = Path(direct_setup["gpu_mappings_manifest"]).resolve()
            direct_ple_manifest = Path(manager_before.get("ple", {}).get("ipc_manifest", "")).resolve()
            for path in (direct_library, direct_groups, direct_maps, direct_ple_manifest):
                if not path.is_file():
                    raise RuntimeError(f"resident direct-MoE prerequisite is missing: {path}")
            if args.rtx_peer_moe and not args.rtx_peer_library.resolve().is_file():
                raise RuntimeError(f"RTX peer library is missing: {args.rtx_peer_library}")
            from qwen38_gpu_mappings_manifest import validate_gpu_mappings_manifest
            if len(validate_gpu_mappings_manifest(direct_maps)) != 24:
                raise RuntimeError("resident direct-MoE manifest no longer has all 24 worker mappings")
    except Exception as exc:
        result.update({
            "status": "FAILED_RESIDENT_PREFLIGHT",
            "error": f"{type(exc).__name__}: {exc}",
            "traceback": traceback.format_exc(),
            "end_unix": time.time(),
        })
        atomic_json(state_path, result)
        print(json.dumps({"status": result["status"], "error": result["error"],
                          "output": str(state_path)}), flush=True)
        return 1
    result["resident_before"] = manager_before
    result["model_index_sha256"] = client_model_hash
    moe_client = None
    ple_client = None
    llm = None
    cpu_mailbox = None
    cpu_mailbox_manifest = None
    cpu_mailbox_registered = False
    bootstrap_log = None
    error: str | None = None
    try:
        # The existing model hooks are reused, but their sockets now terminate
        # in the long-lived daemon rather than this vLLM process.
        # Set the GPU mask before importing torch/vLLM: both can initialize
        # CUDA while importing optional extensions, so setting it later is
        # not a reliable device-isolation boundary.
        os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
        # Backstop for the engine shutdown window (default 5 s): peer teardown restores up to
        # ~150 route records and parks the M10 workers through the daemon.
        os.environ.setdefault("VLLM_WORKER_SHUTDOWN_TIMEOUT_SECONDS", "120")
        # Q38_VLLM_CUDAGRAPH=1: breakable piecewise CUDA graphs for decode (batch 1). Our MoE
        # exchange, PLE lookup and CPU embedding are eager breaks (q38_cudagraph_breaks).
        cuda_graphs = os.environ.get("Q38_VLLM_CUDAGRAPH") == "1"
        if cuda_graphs:
            os.environ["VLLM_USE_BREAKABLE_CUDAGRAPH"] = "1"
        cudagraph_mode = ("FULL_AND_PIECEWISE" if cuda_graphs and os.environ.get("Q38_VLLM_CUDAGRAPH_FULL") == "1"
                          else "PIECEWISE")
        # FlashInfer's sampler JIT-compiles on first use and needs ninja (not installed; 2026-11-03
        # task 46 failed in graph warm-up). vLLM's PyTorch sampler is equivalent for our greedy runs.
        os.environ.setdefault("VLLM_USE_FLASHINFER_SAMPLER", "0")
        os.environ["Q38_MODEL_DIR"] = str(args.model.resolve())   # FP8 cache binds to this checkpoint
        # collective_rpc(_q38_close_peer_adapter) sends a function to the local engine process; vLLM
        # refuses to pickle callables unless this is set (2026-11-03 run: the explicit close was
        # rejected and only the atexit fallback closed the adapter). Local, trusted processes only.
        if args.rtx_peer_moe:
            os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        # FP8 linear-attention projections need the triton GDN decode kernel: with the default cuda
        # kernel the engine hangs without error under concurrency (primitive-ai Qwen3.8-Flash-Next
        # mixed NVFP4/FP8 README, bisected to the FP8 GDN projections; measured on Blackwell).
        if os.environ.get("Q38_RTX_FP8_LINEAR_ATTN") == "1":
            os.environ.setdefault("VLLM_GDN_DECODE_KERNEL", "triton")
        RTX_UUID = discover_rtx_uuid()
        result["rtx_uuid"] = RTX_UUID
        if args.rtx_peer_moe:
            device_rows = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=uuid,pci.bus_id", "--format=csv,noheader"],
                text=True, timeout=15).splitlines()
            devices = {}
            for line in device_rows:
                fields = [field.strip() for field in line.split(",", 1)]
                if len(fields) == 2:
                    devices[fields[0]] = fields[1]
            if RTX_UUID not in devices:
                raise RuntimeError(f"RTX UUID {RTX_UUID} is not currently enumerated")
            domain, bus, devfn = devices[RTX_UUID].split(":", 2)
            rtx_bdf = f"{int(domain, 16):04x}:{bus.lower()}:{devfn.lower()}"
            os.environ["Q38_RTX_BDF"] = rtx_bdf
            os.environ["Q38_M10_RTX_BDF"] = rtx_bdf
            os.environ["CUDA_VISIBLE_DEVICES"] = ",".join(
                [RTX_UUID] + [uuid for uuid in devices if uuid != RTX_UUID])
            result["rtx_bdf"] = rtx_bdf
        else:
            os.environ["CUDA_VISIBLE_DEVICES"] = RTX_UUID
        os.environ["PLE_SKIP_EXPERT_RESERVE"] = "1"
        os.environ["Q38_MOE_TIMING_LOG"] = str(out / "moe_client_timing.jsonl")
        if args.direct_moe or args.rtx_peer_moe:
            os.environ["Q38_PLE_READ_WIDTH"] = str(args.direct_ple_read_width)
            os.environ["Q38_M10_DIRECT_LIBRARY"] = str(direct_library)
            os.environ["Q38_M10_DIRECT_GROUPS"] = str(direct_groups)
            os.environ["Q38_M10_DIRECT_GPU_MAPPINGS_MANIFEST"] = str(direct_maps)
            os.environ["Q38_M10_DIRECT_PLE_MANIFEST"] = str(direct_ple_manifest)
            os.environ["Q38_M10_DIRECT_TIMING_JSON"] = str(out / "direct_moe_timing")
            if args.direct_moe:
                result["direct_ple_read_width_bytes"] = args.direct_ple_read_width
            os.environ.pop("Q38_M10_DIRECT_CPU_MAILBOX_MANIFEST", None)
            if (args.direct_moe or args.rtx_peer_moe) and not args.disable_direct_cpu_mailbox:
                from qwen38_cpu_mailbox_manifest import create_mailbox, load_from_daemon
                mailbox_manifest_path = out / "direct_cpu_mailbox.manifest"
                cpu_mailbox, cpu_mailbox_manifest = create_mailbox(
                    mailbox_manifest_path,
                    run_id=f"vllm-peer-{os.getpid()}-{time.time_ns()}" if args.rtx_peer_moe
                    else f"vllm-direct-{os.getpid()}-{time.time_ns()}")
                registration = load_from_daemon(args.control_socket, mailbox_manifest_path)
                cpu_mailbox_registered = True
                if args.direct_moe:
                    os.environ["Q38_M10_DIRECT_CPU_MAILBOX_MANIFEST"] = str(mailbox_manifest_path)
                if args.rtx_peer_moe:
                    # batched MoE (worker mode 8): the RTX maps the same mailbox (its batch area)
                    os.environ["Q38_M10_RTX_PEER_MAILBOX_MANIFEST"] = str(mailbox_manifest_path)
                    from qwen38_cpu_mailbox_manifest import write_rtx_return_alias_manifest
                    return_aliases = out / "rtx_return_aliases.manifest"
                    write_rtx_return_alias_manifest(return_aliases, registration)
                    os.environ["Q38_M10_RTX_RETURN_ALIASES_MANIFEST"] = str(return_aliases)
                result["direct_cpu_mailbox"] = {
                    "registration": registration,
                    "manifest": str(mailbox_manifest_path),
                    "transport": ("daemon-owned M10 alias addresses for direct M10-to-RTX BAR1 result writes"
                                  if args.rtx_peer_moe else
                                  "resident GPU writes to daemon-registered mapped CPU mailbox"),
                }
            else:
                if args.direct_moe:
                    result["direct_cpu_mailbox"] = {"enabled": False, "fallback": "owner BAR1 CPU reads"}
        if args.rtx_peer_moe:
            os.environ["Q38_M10_RTX_PEER_LIBRARY"] = str(args.rtx_peer_library.resolve())
            os.environ["Q38_M10_RTX_PEER_TIMING_JSON"] = str(out / "rtx_peer_timing")
            os.environ["Q38_M10_CONTROL_SOCKET"] = str(args.control_socket.resolve())
            trace_preload = Path(os.environ.get(
                "Q38_RM_TRACE_PRELOAD",
                str(Path(__file__).resolve().parent.parent / "build" / "rm_ioctl_trace_preload.so"),
            )).resolve()
            if not trace_preload.is_file():
                raise RuntimeError(f"RM export-capture preload is missing: {trace_preload}")
            os.environ["Q38_RM_TRACE_PRELOAD"] = str(trace_preload)
            prior_preload = os.environ.get("LD_PRELOAD", "")
            preload_entries = [entry for entry in prior_preload.split(":") if entry]
            if str(trace_preload) not in preload_entries:
                preload_entries.insert(0, str(trace_preload))
            os.environ["LD_PRELOAD"] = ":".join(preload_entries)
        source_dir = Path(__file__).resolve().parent
        sys.path.insert(0, str(source_dir))
        from run_stage8_stage9_vllm_generation import install_model_hooks
        from vllm import LLM, SamplingParams
        import torch

        if not torch.cuda.is_available() or torch.cuda.get_device_capability(0) != (8, 6):
            raise RuntimeError("vLLM client must see only the RTX 3070 Ti (sm_86)")
        gpu_name = torch.cuda.get_device_name(0)
        if "3070 Ti" not in gpu_name:
            raise RuntimeError(f"unexpected vLLM device 0: {gpu_name}")

        if args.direct_moe:
            from qwen38_direct_moe_adapter import DirectResidentM10MoEAdapter
            moe_client = DirectResidentM10MoEAdapter(
                direct_library, direct_groups, direct_maps, direct_ple_manifest)
            hook_info = install_model_hooks(moe_client, None)
        elif args.rtx_peer_moe:
            # EngineCore is spawned. The child sitecustomize installs the
            # actual GPU-to-GPU adapter lazily; this parent adapter is only
            # used to patch the parent model import surface.
            from qwen38_moe_rpc import MoeSocketClient
            from run_stage8_stage9_vllm_generation import PleSocketClient
            moe_client = MoeSocketClient(args.moe_socket)
            hook_info = install_model_hooks(moe_client, args.ple_socket)
        else:
            from qwen38_moe_rpc import MoeSocketClient
            from run_stage8_stage9_vllm_generation import PleSocketClient
            moe_client = MoeSocketClient(args.moe_socket)
            hook_info = install_model_hooks(moe_client, args.ple_socket)
        bootstrap = out / "worker_bootstrap"
        bootstrap.mkdir(parents=True, exist_ok=True)
        (bootstrap / "sitecustomize.py").write_text((source_dir / "sitecustomize.py").read_text())
        for name in ("qwen38_moe_rpc.py", "run_stage8_stage9_vllm_generation.py"):
            link = bootstrap / name
            if not link.exists() and not link.is_symlink():
                link.symlink_to(source_dir / name)
        prior_pythonpath = os.environ.get("PYTHONPATH", "")
        os.environ["PYTHONPATH"] = os.pathsep.join(
            [str(bootstrap), str(source_dir), prior_pythonpath]
        ).rstrip(os.pathsep)
        os.environ["Q38_M10_WORKER_HOOK"] = "1"
        os.environ["Q38_M10_MOE_SOCKET"] = str(args.moe_socket)
        os.environ["Q38_M10_PLE_SOCKET"] = str(args.ple_socket)
        bootstrap_log = out / "worker_bootstrap.log"
        os.environ["Q38_M10_BOOTSTRAP_LOG"] = str(bootstrap_log)

        llm_started = time.monotonic()
        mtp_tokens = int(os.environ.get("Q38_MTP_SPEC_TOKENS", "0") or 0)
        result["mtp_spec_tokens"] = mtp_tokens
        llm = LLM(
            model=str(model), tokenizer=str(model), trust_remote_code=True,
            tensor_parallel_size=1, dtype="bfloat16", gpu_memory_utilization=args.gpu_memory_utilization,
            cpu_offload_gb=args.cpu_offload_gb, enforce_eager=not cuda_graphs,
            max_model_len=args.max_model_len, max_num_seqs=1,
            # Q38_VLLM_MAX_BATCHED_TOKENS: prefill chunk (default 8, the tested value since 2026-11-01)
            max_num_batched_tokens=int(os.environ.get("Q38_VLLM_MAX_BATCHED_TOKENS", "8")),
            distributed_executor_backend="uni",
            language_model_only=True,
            hf_overrides={"language_model_only": True}, disable_custom_all_reduce=True,
            **({"kv_cache_memory_bytes": args.kv_cache_memory_bytes} if args.kv_cache_memory_bytes > 0 else {}),
            # Q38_VLLM_PREFIX_CACHING=0 (2026-11-03): no prefix caching. Then the GDN (mamba) state is kept once per
            # running request instead of a snapshot per 1568-token block ('align' mode), and the KV pool holds about
            # 3x the tokens. Single-request runs never reuse a prefix. Unset keeps vLLM's default (on).
            **({"enable_prefix_caching": os.environ["Q38_VLLM_PREFIX_CACHING"] == "1"}
               if os.environ.get("Q38_VLLM_PREFIX_CACHING") in ("0", "1") else {}),
            # Q38_VLLM_ASYNC_SCHEDULING=1/0 forces vLLM's async scheduling (host prepares step N+1 while
            # the GPU runs step N); unset keeps vLLM's default. The effective value is recorded below.
            **({"async_scheduling": os.environ["Q38_VLLM_ASYNC_SCHEDULING"] == "1"}
               if os.environ.get("Q38_VLLM_ASYNC_SCHEDULING") in ("0", "1") else {}),
            # PIECEWISE: vLLM's GDN/QSA cores are eager breaks. Q38_VLLM_CUDAGRAPH_FULL=1:
            # FULL_AND_PIECEWISE with a breakable FULL wrapper (q38_cudagraph_breaks.
            # install_full_breakable) captures those cores; only our host breaks stay eager.
            **({"compilation_config": {"cudagraph_mode": cudagraph_mode,
                                       "cudagraph_capture_sizes": sorted({1, 1 + mtp_tokens})}}
               if cuda_graphs else {}),
            # Q38_MTP_SPEC_TOKENS=N (2026-11-04): MTP speculative decoding with the checkpoint's MTP layer, N draft
            # tokens per step (its experts are resident layer 48 on the M10s: daemon Q38_DAEMON_LAYERS=49).
            **({"speculative_config": {"method": "mtp", "num_speculative_tokens": mtp_tokens},
                "disable_log_stats": False} if mtp_tokens > 0 else {}),
        )
        result["cuda_graphs"] = cuda_graphs
        try:
            result["async_scheduling_effective"] = llm.llm_engine.vllm_config.scheduler_config.async_scheduling
        except Exception as exc:
            result["async_scheduling_effective"] = f"unknown: {exc}"
        print(f"Q38_ASYNC_SCHEDULING,requested,{os.environ.get('Q38_VLLM_ASYNC_SCHEDULING', 'default')},"
              f"effective,{result['async_scheduling_effective']}", flush=True)
        result["cudagraph_mode"] = cudagraph_mode if cuda_graphs else "NONE"
        result["vllm_load_seconds"] = time.monotonic() - llm_started
        result["gpu_name"] = gpu_name
        result["hook_info"] = hook_info
        result["worker_bootstrap_log"] = str(bootstrap_log)
        result["moe_transport_expected"] = (
            "RTX_CUDA_to_M10_BAR1_peer" if args.rtx_peer_moe else
            "direct_PEX_BAR1" if args.direct_moe else "unix_socket")
        result["ple_transport_expected"] = (
            "direct_BAR1_resident_mode4" if (args.direct_moe or args.rtx_peer_moe) else "unix_socket")
        prompt = args.prompt_file.read_text() if args.prompt_file else "The capital of Denmark is"
        if args.park_workers_on_exit:
            result["worker_wake"] = send_control(args.control_socket, "WAKE_WORKERS",
                                                   clients_quiesced=True)
        # Q38_VLLM_LOGPROBS=N (2026-11-03): record the top-N logprobs of every generated token (greedy is unchanged).
        # With the M10 LM head the logits are the 24 x K candidates (BF16, -inf elsewhere): the logprobs are normalized
        # over the candidates, differences between them are the exact logit differences.
        q38_logprobs = int(os.environ.get("Q38_VLLM_LOGPROBS", "0") or 0)
        # Q38_VLLM_IGNORE_EOS=1 (2026-11-04): always generate max_new_tokens (equal-length speed comparisons)
        sampling = SamplingParams(temperature=0.0, max_tokens=args.max_new_tokens, seed=0,
                                  ignore_eos=os.environ.get("Q38_VLLM_IGNORE_EOS") == "1",
                                  **({"logprobs": q38_logprobs} if q38_logprobs > 0 else {}))
        generations = []

        def spec_metrics() -> dict:
            out_m = {}
            try:
                for m in llm.get_metrics():
                    if "spec_decode" in m.name:
                        v = getattr(m, "value", None)
                        if v is None and hasattr(m, "values"):
                            v = list(m.values)
                        out_m[m.name] = v
            except Exception as exc:          # metrics need disable_log_stats=False (set with Q38_MTP_SPEC_TOKENS)
                out_m["error"] = f"{type(exc).__name__}: {exc}"
            return out_m
        for repeat in range(args.repeats):
            started_unix = time.time()
            started = time.monotonic()
            outputs = llm.generate([prompt], sampling, use_tqdm=False)
            elapsed = time.monotonic() - started
            if len(outputs) != 1 or len(outputs[0].outputs) != 1:
                raise RuntimeError("unexpected vLLM output cardinality")
            item = outputs[0]
            candidate = item.outputs[0]
            generations.append({
                "repeat": repeat, "prompt": prompt,
                "prompt_token_ids": list(item.prompt_token_ids or []),
                "generated_token_ids": list(candidate.token_ids),
                "text": candidate.text, "finish_reason": candidate.finish_reason,
                "elapsed_seconds": elapsed,
                "started_unix": started_unix, "ended_unix": started_unix + elapsed,
            })
            if mtp_tokens > 0:              # cumulative vLLM counters (drafts, draft tokens, accepted tokens)
                generations[-1]["spec_decode_metrics"] = spec_metrics()
                print("Q38_SPEC_DECODE," + json.dumps(generations[-1]["spec_decode_metrics"]), flush=True)
            if q38_logprobs > 0:
                # per generated token: [[token_id, logprob], ...] best first (top-N plus the chosen token)
                generations[-1]["top_logprobs"] = [
                    sorted(([int(t), float(lp.logprob)] for t, lp in (step or {}).items()), key=lambda e: -e[1])
                    for step in (candidate.logprobs or [])]
        result["generation"] = generations
        # Q38_HUMANEVAL_JSONL: greedy pass over every HumanEval problem whose prompt fits the context
        # (Q38_HUMANEVAL_MAX_PROMPT_TOKENS, default 380; Q38_HUMANEVAL_MAX_NEW_TOKENS, default 256;
        # Q38_HUMANEVAL_LIMIT caps the count). Standard HumanEval stop sequences. Scored separately
        # by score_humaneval.py (the completions are not executed in this process).
        he_path = os.environ.get("Q38_HUMANEVAL_JSONL")
        if he_path:
            import gzip
            opener = gzip.open if he_path.endswith(".gz") else open
            with opener(he_path, "rt") as fh:
                problems = [json.loads(line) for line in fh if line.strip()]
            tok = llm.get_tokenizer()
            max_prompt = int(os.environ.get("Q38_HUMANEVAL_MAX_PROMPT_TOKENS", "380"))
            max_new = int(os.environ.get("Q38_HUMANEVAL_MAX_NEW_TOKENS", "256"))
            limit = int(os.environ.get("Q38_HUMANEVAL_LIMIT", "0"))
            # Q38_HUMANEVAL_CHAT=1: chat template with thinking (Q38_HUMANEVAL_THINKING, default 1), no
            # stop sequences, the code is taken from the last ```python block after </think>.
            # Q38_HUMANEVAL_TASKS="HumanEval/32,..." restricts the problem set.
            chat = os.environ.get("Q38_HUMANEVAL_CHAT") == "1"
            thinking = os.environ.get("Q38_HUMANEVAL_THINKING", "1") == "1"
            # Q38_HUMANEVAL_REASONING_EFFORT = xhigh | medium | low (chat template reasoning_effort;
            # unset = template default xhigh). Ignored without thinking.
            effort = os.environ.get("Q38_HUMANEVAL_REASONING_EFFORT") or None
            wanted = [t for t in os.environ.get("Q38_HUMANEVAL_TASKS", "").split(",") if t]
            if wanted:
                problems = [p for p in problems if p["task_id"] in wanted]
            stops = [] if chat else ["\nclass", "\ndef", "\n#", "\nif", "\nprint"]

            def he_prompt(prob):
                if not chat:
                    return prob["prompt"]
                msg = ("Complete the following Python function. Reply with the complete function "
                       "(including any imports it needs) in one ```python code block.\n\n```python\n"
                       + prob["prompt"] + "```")
                kw = {"reasoning_effort": effort} if (thinking and effort) else {}
                return tok.apply_chat_template([{"role": "user", "content": msg}], add_generation_prompt=True,
                                               tokenize=False, enable_thinking=thinking, **kw)

            def he_extract(text):
                import re
                answer = text.split("</think>", 1)[1] if "</think>" in text else text
                blocks = re.findall(r"```(?:python)?\n(.*?)```", answer, re.S)
                return answer, (blocks[-1] if blocks else answer)
            items, skipped = [], []
            he_started = time.monotonic()
            for prob in problems:
                if limit and len(items) >= limit:
                    break
                prompt_text = he_prompt(prob)
                n_prompt = len(tok(prompt_text, add_special_tokens=False).input_ids)
                budget = min(max_new, args.max_model_len - n_prompt - 1)
                if n_prompt > max_prompt or budget < 32:
                    skipped.append({"task_id": prob["task_id"], "prompt_tokens": n_prompt})
                    continue
                sp = SamplingParams(temperature=0.0, max_tokens=budget, seed=0, stop=stops)
                t0 = time.monotonic()
                he_out = llm.generate([prompt_text], sp, use_tqdm=False)[0]   # not `out`: that is the output dir
                cand = he_out.outputs[0]
                extra = {}
                completion = cand.text
                if chat:
                    answer, completion = he_extract(cand.text)
                    ids = list(cand.token_ids)
                    extra = {"mode": "chat_thinking" if thinking else "chat", "raw_text": cand.text, "answer": answer,
                             "thinking_tokens": ids.index(248069) if 248069 in ids else (len(ids) if thinking else 0)}
                items.append({"task_id": prob["task_id"], "prompt_tokens": len(he_out.prompt_token_ids or []),
                              "generated_token_ids": list(cand.token_ids), "completion": completion, **extra,
                              "finish_reason": cand.finish_reason, "stop_reason": cand.stop_reason,
                              "elapsed_seconds": round(time.monotonic() - t0, 3)})
                print(f"Q38_HUMANEVAL,{len(items)},{prob['task_id']},prompt,{items[-1]['prompt_tokens']},"
                      f"tokens,{len(cand.token_ids)},finish,{cand.finish_reason},s,{items[-1]['elapsed_seconds']}", flush=True)
            result["humaneval"] = {"dataset": he_path, "dataset_sha256": sha256_file(Path(he_path)),
                                   "mode": ("chat_thinking" if thinking else "chat") if chat else "completion",
                                   "reasoning_effort": (effort or "xhigh (template default)") if (chat and thinking) else None,
                                   "max_prompt_tokens": max_prompt, "max_new_tokens": max_new, "stops": stops,
                                   "problems": len(problems), "generated": len(items), "skipped": skipped,
                                   "seconds": round(time.monotonic() - he_started, 1), "items": items}
        # Q38_BENCH_JSONL (2026-11-04): generic benchmark pass, one item at a time, results appended per item to
        # Q38_BENCH_OUT (JSONL, flushed; items already there are skipped, so a re-queued run resumes).
        # Item: {"id", "messages": [chat messages], "kind": "generate" | "choice", "choices": ["A", ...] (choice)}.
        # generate: chat template with thinking (Q38_BENCH_THINKING, default 1; Q38_BENCH_EFFORT = template
        #   reasoning_effort, unset = template default), sampling Q38_BENCH_TEMPERATURE / TOP_P / TOP_K (default the
        #   model card's thinking settings 1.0 / 0.95 / 20), seed per item (stable hash of the id), max output
        #   Q38_BENCH_MAX_TOKENS (default 32768, bounded by the context).
        # choice: no thinking, one generated token, logprobs of the choice letters (top Q38_BENCH_CHOICE_LOGPROBS, 20).
        # Q38_BENCH_LIMIT caps the items, Q38_BENCH_IDS="id1,id2" restricts them. Scoring is offline.
        bench_path = os.environ.get("Q38_BENCH_JSONL")
        if bench_path:
            import hashlib as _hl
            out_path = Path(os.environ["Q38_BENCH_OUT"])
            done = set()
            if out_path.exists():
                for line in out_path.read_text().splitlines():
                    if line.strip():
                        done.add(json.loads(line)["id"])
            with open(bench_path) as fh:
                bench = [json.loads(line) for line in fh if line.strip()]
            wanted = [t for t in os.environ.get("Q38_BENCH_IDS", "").split(",") if t]
            if wanted:
                bench = [b for b in bench if b["id"] in wanted]
            limit = int(os.environ.get("Q38_BENCH_LIMIT", "0"))
            if limit:
                bench = bench[:limit]
            tok = llm.get_tokenizer()
            thinking = os.environ.get("Q38_BENCH_THINKING", "1") == "1"
            effort = os.environ.get("Q38_BENCH_EFFORT") or None
            temp = float(os.environ.get("Q38_BENCH_TEMPERATURE", "1.0"))
            top_p = float(os.environ.get("Q38_BENCH_TOP_P", "0.95"))
            top_k = int(os.environ.get("Q38_BENCH_TOP_K", "20"))
            max_new = int(os.environ.get("Q38_BENCH_MAX_TOKENS", "32768"))
            n_lp = int(os.environ.get("Q38_BENCH_CHOICE_LOGPROBS", "20"))
            think_end = tok.convert_tokens_to_ids("</think>")
            stats = {"items": len(bench), "skipped_done": 0, "generated": 0, "too_long": 0, "tokens": 0,
                     "stopped_at_deadline": 0}
            # Q38_BENCH_DEADLINE = unix time (seconds): no new item is started after it (night runs, 2026-11-04)
            deadline = float(os.environ.get("Q38_BENCH_DEADLINE", "0") or 0)
            bench_started = time.monotonic()
            with open(out_path, "a") as outf:
                for item in bench:
                    if item["id"] in done:
                        stats["skipped_done"] += 1
                        continue
                    if deadline and time.time() > deadline:
                        stats["stopped_at_deadline"] = 1
                        print(f"Q38_BENCH_DEADLINE,reached,{deadline:.0f},generated,{stats['generated']}", flush=True)
                        break
                    kind = item.get("kind", "generate")
                    think = thinking and kind == "generate"
                    kw = {"reasoning_effort": effort} if (think and effort) else {}
                    prompt_text = tok.apply_chat_template(item["messages"], add_generation_prompt=True, tokenize=False,
                                                          enable_thinking=think, **kw)
                    n_prompt = len(tok(prompt_text, add_special_tokens=False).input_ids)
                    seed = int.from_bytes(_hl.sha256(str(item["id"]).encode()).digest()[:4], "little")
                    if kind == "choice":
                        sp = SamplingParams(temperature=0.0, max_tokens=1, seed=seed, logprobs=n_lp)
                    else:
                        budget = min(max_new, args.max_model_len - n_prompt - 1)
                        if budget < 256:
                            stats["too_long"] += 1
                            rec = {"id": item["id"], "kind": kind, "error": "prompt_too_long", "prompt_tokens": n_prompt}
                            outf.write(json.dumps(rec) + "\n"); outf.flush()
                            continue
                        sp = SamplingParams(temperature=temp, top_p=top_p, top_k=top_k, max_tokens=budget, seed=seed)
                    t0 = time.monotonic()
                    o = llm.generate([prompt_text], sp, use_tqdm=False)[0]
                    cand = o.outputs[0]
                    ids = list(cand.token_ids)
                    rec = {"id": item["id"], "kind": kind, "prompt_tokens": len(o.prompt_token_ids or []),
                           "output_tokens": len(ids), "finish_reason": cand.finish_reason,
                           "elapsed_seconds": round(time.monotonic() - t0, 3), "seed": seed}
                    if kind == "choice":
                        lp = {}
                        for t, l in (cand.logprobs[0] if cand.logprobs else {}).items():
                            lp[tok.decode([t]).strip()] = max(lp.get(tok.decode([t]).strip(), -1e9), float(l.logprob))
                        choice_lp = {c: lp[c] for c in item["choices"] if c in lp}
                        rec.update({"choice_logprobs": choice_lp, "top_token": tok.decode(ids[:1]),
                                    "pred": max(choice_lp, key=choice_lp.get) if choice_lp else None})
                    else:
                        text = cand.text
                        rec.update({"text": text, "thinking_tokens": ids.index(think_end) if think_end in ids else
                                    (len(ids) if think else 0), "max_tokens": sp.max_tokens,
                                    "sampling": {"temperature": temp, "top_p": top_p, "top_k": top_k, "thinking": think,
                                                 "effort": effort or "template default"}})
                    outf.write(json.dumps(rec) + "\n"); outf.flush()
                    stats["generated"] += 1; stats["tokens"] += len(ids)
                    el = time.monotonic() - bench_started
                    print(f"Q38_BENCH,{stats['generated']}/{len(bench) - stats['skipped_done']},{item['id']},kind,{kind},"
                          f"prompt,{rec['prompt_tokens']},out,{len(ids)},finish,{cand.finish_reason},s,{rec['elapsed_seconds']},"
                          f"total_s,{el:.0f},tok_s,{stats['tokens'] / max(el, 1e-9):.1f}", flush=True)
            stats["seconds"] = round(time.monotonic() - bench_started, 1)
            result["bench"] = {"dataset": bench_path, "dataset_sha256": sha256_file(Path(bench_path)),
                               "out": str(out_path), "thinking": thinking, "effort": effort, "temperature": temp,
                               "top_p": top_p, "top_k": top_k, "max_tokens": max_new, "deadline": deadline, **stats}
        result["status"] = "PASS_VLLM_RESIDENT_M10_CLIENT"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        result["error"] = error
        result["traceback"] = traceback.format_exc()
        result["status"] = "PARTIAL_VLLM_RESIDENT_M10_CLIENT"
    finally:
        ceiling_iters = int(os.environ.get("Q38_GRAPH_CEILING_ITERS", "0"))
        if llm is not None and ceiling_iters > 0 and "error" not in result:
            try:   # after every generation, while the peer path is still open
                result["graph_ceiling"] = llm.collective_rpc(_q38_graph_ceiling, timeout=600, args=(ceiling_iters,))
            except BaseException as exc:
                result["graph_ceiling"] = {"error": f"{type(exc).__name__}: {exc}"}
        if llm is not None and int(os.environ.get("Q38_LM2_CHECK", "0") or 0) > 0 and "error" not in result:
            try:   # after every generation, while the peer path is still open
                result["lm2_check"] = llm.collective_rpc(_q38_lm2_check, timeout=600)
            except BaseException as exc:
                result["lm2_check"] = {"error": f"{type(exc).__name__}: {exc}"}
        if llm is not None and os.environ.get("Q38_BATCH_TIMING") == "1" and "error" not in result:
            try:   # after every generation, while the peer path is still open
                result["batch_timing"] = llm.collective_rpc(_q38_batch_timing, timeout=900)
            except BaseException as exc:
                result["batch_timing"] = {"error": f"{type(exc).__name__}: {exc}"}
        if llm is not None and args.rtx_peer_moe:
            # Explicit, verified peer teardown inside the engine process, also after Ctrl-C
            # (KeyboardInterrupt reaches this finally block). Never rely on atexit here.
            try:
                result["peer_close_rpc"] = llm.collective_rpc(_q38_close_peer_adapter, timeout=300)
            except BaseException as exc:
                result.setdefault("cleanup_errors", []).append(f"peer adapter close rpc: {type(exc).__name__}: {exc}")
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
        try:
            import importlib
            ngram_mod = importlib.import_module("vllm.models.qwen4_exp.nvidia.ngram_embedding")
            remote_cls = getattr(ngram_mod, "Qwen4ExpPLEDeviceEmbedding", None)
            ple_client = getattr(remote_cls, "_client", None)
            if ple_client is not None:
                # Direct mode shares the native adapter for both paths. In
                # socket mode, close only the replaceable client connection.
                if args.direct_moe or args.rtx_peer_moe:
                    ple_client.close()
                else:
                    ple_client.close(stop=False)
        except Exception as exc:
            result.setdefault("cleanup_errors", []).append(f"PLE client close: {exc}")
        if moe_client is not None:
            try:
                moe_client.close()
            except Exception as exc:
                result.setdefault("cleanup_errors", []).append(f"MoE client close: {exc}")
        if cpu_mailbox is not None:
            if cpu_mailbox_registered and result.get("status") == "PASS_VLLM_RESIDENT_M10_CLIENT" and not result.get("cleanup_errors"):
                try:
                    from qwen38_cpu_mailbox_manifest import unload_from_daemon
                    result["direct_cpu_mailbox"]["unregistration"] = unload_from_daemon(
                        args.control_socket)
                    cpu_mailbox.unlink()
                    cpu_mailbox.close()
                    cpu_mailbox = None
                except Exception as exc:
                    result.setdefault("cleanup_errors", []).append(
                        f"direct CPU mailbox teardown: {exc}")
            elif not cpu_mailbox_registered:
                cpu_mailbox.unlink()
                cpu_mailbox.close()
                cpu_mailbox = None
            else:
                result["direct_cpu_mailbox"]["cleanup"] = (
                    "retained for recovery because vLLM/client completion was not clean")
        if (args.park_workers_on_exit and
                result.get("status") == "PASS_VLLM_RESIDENT_M10_CLIENT" and
                not result.get("cleanup_errors")):
            try:
                result["worker_park"] = send_control(
                    args.control_socket, "PARK_WORKERS", clients_quiesced=True)
            except Exception as exc:
                result.setdefault("cleanup_errors", []).append(f"worker park: {exc}")
        try:
            manager_after = send_control(args.control_socket, "STATUS")
            # The daemon refreshes counters periodically. Allow one reporting
            # interval before judging whether this client crossed both paths.
            for _ in range(5):
                moe_progress = (args.direct_moe or args.rtx_peer_moe or
                    manager_after.get("moe_requests", 0) > manager_before.get("moe_requests", 0))
                ple_progress = (args.direct_moe or args.rtx_peer_moe or
                    manager_after.get("ple_requests", 0) > manager_before.get("ple_requests", 0))
                if moe_progress and ple_progress:
                    break
                time.sleep(1)
                manager_after = send_control(args.control_socket, "STATUS")
            result["resident_after"] = manager_after
            stable_fields = ("pid", "started_unix", "model_index_sha256", "layers_ready",
                             "expert_load_seconds", "routed_service_pid", "shared_service_pid")
            changed = [key for key in stable_fields if manager_after.get(key) != manager_before.get(key)]
            if manager_after.get("status") != "READY" or changed:
                result.setdefault("cleanup_errors", []).append(
                    f"M10 service identity/readiness changed during vLLM client run: {changed}"
                )
            if (not args.direct_moe and not args.rtx_peer_moe and
                    manager_after.get("moe_requests", 0) <= manager_before.get("moe_requests", 0)):
                result.setdefault("cleanup_errors", []).append("no MoE request reached resident daemon")
            if args.direct_moe or args.rtx_peer_moe:
                records = []
                for timing_path in sorted(out.glob("direct_moe_timing.*.json")):
                    try:
                        records.append(json.loads(timing_path.read_text()))
                    except (OSError, json.JSONDecodeError) as exc:
                        result.setdefault("cleanup_errors", []).append(
                            f"invalid direct MoE timing record {timing_path}: {exc}")
                calls = sum(int(x.get("calls", 0)) for x in records)
                ple_calls = sum(int(x.get("ple_calls", 0)) for x in records)
                ple_rows = sum(int(x.get("ple_rows", 0)) for x in records)
                result["direct_moe_timing"] = {"processes": records, "calls": calls,
                    "direct_ple_calls": ple_calls, "direct_ple_rows": ple_rows}
                peer_records = []
                if args.rtx_peer_moe:
                    for timing_path in sorted(out.glob("rtx_peer_timing.*.json")):
                        try:
                            peer_records.append(json.loads(timing_path.read_text()))
                        except (OSError, json.JSONDecodeError) as exc:
                            result.setdefault("cleanup_errors", []).append(
                                f"invalid RTX peer timing record {timing_path}: {exc}")
                    calls = sum(int(x.get("calls", 0)) for x in peer_records)
                    result["rtx_peer_timing"] = {"processes": peer_records,
                        "calls": calls, "rows": sum(int(x.get("rows", 0)) for x in peer_records),
                        "host_payload_relay_bytes": 0}
                bootstrap_records = []
                if bootstrap_log is not None and bootstrap_log.is_file():
                    for line in bootstrap_log.read_text(errors="replace").splitlines():
                        try:
                            bootstrap_records.append(json.loads(line))
                        except json.JSONDecodeError:
                            pass
                result["worker_bootstrap_records"] = bootstrap_records
                expected_transport = ("RTX_CUDA_to_M10_BAR1_peer" if args.rtx_peer_moe
                                      else "direct_PEX_BAR1")
                if calls <= 0 or not any(x.get("moe_transport") == expected_transport
                                         for x in bootstrap_records):
                    result.setdefault("cleanup_errors", []).append(
                        f"vLLM worker did not execute {expected_transport} MoE path")
                expected_ple = ("pci_cluster_row_gather_mode6"
                                if args.rtx_peer_moe and os.environ.get("Q38_RTX_PEER_PLE_GPU") == "1"
                                else "direct_BAR1_resident_mode4")
                if ((not args.rtx_peer_moe and ple_calls <= 0) or not any(
                        x.get("ple_transport") == expected_ple
                        for x in bootstrap_records)):
                    result.setdefault("cleanup_errors", []).append(
                        f"vLLM worker did not execute the {expected_ple} PLE path")
                # Q38_RTX_PEER_PLE_GPU=1: PLE runs as the PCI-cluster row gather (worker mode 6) on the
                # GPU-driven path, so there are no CPU-driven mode-4 lookups; require evidence of the
                # GPU gather instead (eager calls in prefill/warm-up; graph replays are not counted).
                ple_gpu = os.environ.get("Q38_RTX_PEER_PLE_GPU") == "1"
                if args.rtx_peer_moe and ple_gpu:
                    gpu_ple_calls = sum(int(x.get("ple_gpu_eager_calls") or 0) for x in peer_records)
                    result["rtx_peer_timing"]["ple_gpu_eager_calls"] = gpu_ple_calls
                    result["rtx_peer_timing"]["ple_gpu_eager_rows"] = sum(int(x.get("ple_gpu_eager_rows") or 0) for x in peer_records)
                    if gpu_ple_calls <= 0 or not all(x.get("ple_gpu") for x in peer_records):
                        result.setdefault("cleanup_errors", []).append(
                            "vLLM worker did not execute the PCI-cluster GPU PLE row gather")
                elif args.rtx_peer_moe and ple_calls <= 0:
                    result.setdefault("cleanup_errors", []).append(
                        "vLLM worker did not execute direct BAR1 PLE lookups")
            if (not args.direct_moe and not args.rtx_peer_moe and
                    manager_after.get("ple_requests", 0) <= manager_before.get("ple_requests", 0)):
                result.setdefault("cleanup_errors", []).append("no PLE request reached resident daemon")
            timing_path = out / "moe_client_timing.jsonl"
            if timing_path.exists():
                timing_records = []
                for line in timing_path.read_text(errors="replace").splitlines():
                    try:
                        timing_records.append(json.loads(line))
                    except json.JSONDecodeError:
                        continue
                result["moe_client_timing_records"] = timing_records
        except Exception as exc:
            result.setdefault("cleanup_errors", []).append(f"resident status after client: {exc}")
        result["end_unix"] = time.time()
        if result.get("cleanup_errors"):
            result["status"] = "FAIL_CLIENT_CLEANUP"
        atomic_json(state_path, result)
    print(json.dumps({"status": result["status"], "error": error,
                      "output": str(state_path)}), flush=True)
    return 0 if result["status"] == "PASS_VLLM_RESIDENT_M10_CLIENT" else 1


if __name__ == "__main__":
    raise SystemExit(main())
