"""Aggregate Phase 13 per-cell latency CSVs into a single summary table.

Each cell's CSV (one row per fire_batch) has columns:
  step,total_ms,build_batch_ms,...,inter_call_gap_ms

This script:
  1. Skips the first 30 steps per cell (prefill / warmup transient).
  2. Reports mean, p50, p95 of total_ms, transform_gpu_ms, inter_call_gap_ms,
     batch_num_seqs.
  3. Combines with the per-cell tps JSON for end-to-end attribution:
       end_to_end_ms_per_step = (1000 / tps) * mean_batch_num_seqs
       host_overhead_ms = end_to_end_ms_per_step - transform_gpu_ms
       inter_call_share = inter_call_gap_ms / host_overhead_ms

Run::
    python phase13_aggregate.py --in-dir /path/to/results --out report.json
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median


def _percentile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    idx = max(0, min(len(sorted_vals) - 1, int(p * len(sorted_vals))))
    return sorted_vals[idx]


def _aggregate_csv(csv_path: Path, skip_steps: int = 30) -> dict:
    """Mean / p50 / p95 of selected columns, after dropping prefill steps."""
    rows = []
    with csv_path.open() as fh:
        for r in csv.DictReader(fh):
            rows.append(r)
    if len(rows) <= skip_steps:
        return {"steady_steps": 0, "raw_steps": len(rows)}

    rows = rows[skip_steps:]
    cols = ("total_ms", "transform_gpu_ms", "sample_gpu_ms",
            "embed_gpu_ms", "inference_ms", "inter_call_gap_ms")
    out = {"steady_steps": len(rows), "skipped": skip_steps}
    for c in cols:
        vals = sorted(float(r[c]) for r in rows if r.get(c))
        if vals:
            out[f"{c}_mean"] = mean(vals)
            out[f"{c}_p50"] = median(vals)
            out[f"{c}_p95"] = _percentile(vals, 0.95)

    bs_vals = sorted(int(r["batch_num_seqs"]) for r in rows
                     if r.get("batch_num_seqs"))
    if bs_vals:
        out["batch_num_seqs_mean"] = mean(bs_vals)
        out["batch_num_seqs_p50"] = median(bs_vals)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--skip-steps", type=int, default=30)
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    cells = []
    for csv_path in sorted(in_dir.glob("*_latency.csv")):
        # Filename: {label}_c{C}_t{T}_latency.csv
        stem = csv_path.stem.replace("_latency", "")
        parts = stem.split("_")
        label = parts[0]
        c = int(parts[1].lstrip("c"))
        tok = int(parts[2].lstrip("t"))
        json_path = in_dir / f"{label}_c{c}_t{tok}.json"
        cell_summary = json.loads(json_path.read_text()) if json_path.exists() else {}
        agg = _aggregate_csv(csv_path, skip_steps=args.skip_steps)

        # End-to-end per-step (real wall) at observed batch
        bns = agg.get("batch_num_seqs_mean", 0) or c
        tps = cell_summary.get("tps", 0)
        e2e_ms = (1000.0 / tps * bns) if tps > 0 else 0
        transform_ms = agg.get("transform_gpu_ms_mean", 0)
        sample_ms = agg.get("sample_gpu_ms_mean", 0)
        embed_ms = agg.get("embed_gpu_ms_mean", 0)
        py_total_ms = agg.get("total_ms_mean", 0)
        inter_ms = agg.get("inter_call_gap_ms_mean", 0)
        host_overhead_ms = max(0.0, e2e_ms - transform_ms - sample_ms - embed_ms)

        cells.append({
            "label": label, "concurrency": c, "max_tokens": tok,
            "tps": tps,
            "batch_num_seqs_mean": bns,
            "e2e_ms_per_step": e2e_ms,
            "py_total_ms_mean": py_total_ms,
            "transform_gpu_ms_mean": transform_ms,
            "sample_gpu_ms_mean": sample_ms,
            "embed_gpu_ms_mean": embed_ms,
            "inter_call_gap_ms_mean": inter_ms,
            "inter_call_gap_ms_p95": agg.get("inter_call_gap_ms_p95", 0),
            "host_overhead_ms": host_overhead_ms,
            "raw_agg": agg,
        })

    out_path = Path(args.out)
    out_path.write_text(json.dumps({"cells": cells}, indent=2))
    print(f"Wrote {out_path} with {len(cells)} cell rows")
    # Print a flat table
    print()
    print(f"{'label':<10} {'c':>3} {'tok':>5} {'tps':>7} {'bns':>5} "
          f"{'e2e':>6} {'pyT':>6} {'xform':>6} {'sample':>6} "
          f"{'inter_p50':>9} {'inter_p95':>9} {'host_oh':>8}")
    for c in cells:
        print(f"{c['label']:<10} {c['concurrency']:>3} {c['max_tokens']:>5} "
              f"{c['tps']:>7.1f} {c['batch_num_seqs_mean']:>5.1f} "
              f"{c['e2e_ms_per_step']:>6.2f} {c['py_total_ms_mean']:>6.2f} "
              f"{c['transform_gpu_ms_mean']:>6.2f} {c['sample_gpu_ms_mean']:>6.2f} "
              f"{c['inter_call_gap_ms_mean']:>9.3f} "
              f"{c['inter_call_gap_ms_p95']:>9.3f} "
              f"{c['host_overhead_ms']:>8.2f}")


if __name__ == "__main__":
    main()
