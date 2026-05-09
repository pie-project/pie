"""Split the per-step latency CSV by batch_num_seqs ranges and aggregate
per-cell stats. Phase 13 cells:
  c=4  → batch_num_seqs ∈ [1, 4]
  c=16 → batch_num_seqs ∈ [10, 16]
  c=64 → batch_num_seqs ∈ [50, 64]

The harness rotation logic was meant to give each cell its own CSV but
pie's `latency.py` keeps the file handle open across rotations, so all
fire_batch rows from c=4 + c=16 + c=64 land in one file. Splitting by
batch_num_seqs is the right post-hoc approach because batch shape is
deterministic per cell after warmup.
"""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import mean, median


CELLS = {
    "c4":  range(1, 5),
    "c16": range(10, 17),
    "c64": range(50, 65),
}


def _percentile(sorted_vals, p):
    if not sorted_vals:
        return 0.0
    idx = max(0, min(len(sorted_vals) - 1, int(p * len(sorted_vals))))
    return sorted_vals[idx]


def _stats(rows, col):
    vals = sorted(float(r[col]) for r in rows if r.get(col))
    if not vals:
        return {"mean": 0, "p50": 0, "p95": 0, "n": 0}
    return {"mean": mean(vals), "p50": median(vals),
            "p95": _percentile(vals, 0.95), "n": len(vals)}


def split_and_aggregate(csv_path: Path, skip_warmup_per_cell: int = 30):
    with csv_path.open() as fh:
        rows = list(csv.DictReader(fh))

    # Group rows by which cell their batch_num_seqs falls in
    groups: dict[str, list] = {k: [] for k in CELLS}
    for r in rows:
        try:
            bns = int(r["batch_num_seqs"])
        except (ValueError, KeyError):
            continue
        for cell, rng in CELLS.items():
            if bns in rng:
                groups[cell].append(r)
                break

    out = {}
    for cell, group in groups.items():
        # Drop the first `skip_warmup_per_cell` steps in the group to
        # exclude prefill ramp-up.
        steady = group[skip_warmup_per_cell:] if len(group) > skip_warmup_per_cell else group
        out[cell] = {
            "raw_steps": len(group),
            "steady_steps": len(steady),
            "transform_gpu_ms": _stats(steady, "transform_gpu_ms"),
            "sample_gpu_ms": _stats(steady, "sample_gpu_ms"),
            "embed_gpu_ms": _stats(steady, "embed_gpu_ms"),
            "total_ms": _stats(steady, "total_ms"),
            "inference_ms": _stats(steady, "inference_ms"),
            "inter_call_gap_ms": _stats(steady, "inter_call_gap_ms"),
            "batch_num_seqs": _stats(steady, "batch_num_seqs"),
            "batch_total_tokens": _stats(steady, "batch_total_tokens"),
        }
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in-dir", required=True)
    p.add_argument("--out", required=True)
    args = p.parse_args()

    in_dir = Path(args.in_dir)
    summary = {"variants": {}}

    for label in ("baseline", "parity"):
        # Find the combined CSV (was c4 file by accident of rotation order).
        candidates = list(in_dir.glob(f"{label}_*_latency.csv"))
        if not candidates:
            continue
        csv_path = candidates[0]
        cell_stats = split_and_aggregate(csv_path)
        # Cross-reference with the per-cell tps JSON
        for cell in CELLS:
            tag = cell.replace("c", "")
            json_path = in_dir / f"{label}_c{tag}_t200.json"
            if json_path.exists():
                cell_summary = json.loads(json_path.read_text())
                cell_stats[cell]["tps"] = cell_summary.get("tps", 0)
                cell_stats[cell]["wall_s"] = cell_summary.get("wall_s", 0)
                cell_stats[cell]["completed"] = cell_summary.get("completed", 0)
                # End-to-end real wall per step at observed batch
                bns = cell_stats[cell]["batch_num_seqs"]["mean"] or float(tag.replace("c","") or 1)
                tps = cell_summary.get("tps", 0)
                cell_stats[cell]["e2e_ms_per_step_at_obs_batch"] = (
                    1000.0 / tps * bns if tps > 0 else 0)
        summary["variants"][label] = cell_stats

    out = Path(args.out)
    out.write_text(json.dumps(summary, indent=2))
    print(f"Wrote {out}")
    print()

    # Pretty-print a comparison table
    print(f"{'variant':<10} {'cell':<5} {'tps':>7} {'bns':>5} "
          f"{'e2e':>6} {'pyT':>6} {'xform':>6} {'sample':>6} "
          f"{'inter_p50':>9} {'inter_p95':>9} {'inter_mean':>10}")
    for label, cells in summary["variants"].items():
        for cell, s in cells.items():
            tps = s.get("tps", 0)
            bns = s["batch_num_seqs"]["mean"]
            e2e = s.get("e2e_ms_per_step_at_obs_batch", 0)
            pyT = s["total_ms"]["mean"]
            xform = s["transform_gpu_ms"]["mean"]
            sample = s["sample_gpu_ms"]["mean"]
            inter_p50 = s["inter_call_gap_ms"]["p50"]
            inter_p95 = s["inter_call_gap_ms"]["p95"]
            inter_mean = s["inter_call_gap_ms"]["mean"]
            print(f"{label:<10} {cell:<5} {tps:>7.1f} {bns:>5.1f} "
                  f"{e2e:>6.2f} {pyT:>6.2f} {xform:>6.2f} {sample:>6.2f} "
                  f"{inter_p50:>9.3f} {inter_p95:>9.3f} {inter_mean:>10.3f}")


if __name__ == "__main__":
    main()
