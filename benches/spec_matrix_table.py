#!/usr/bin/env python3
"""Fold the per-arm `three_way.py --out` files into the speculation matrix:
each engine's plain tok/s beside its own speculative arm, and the arm's ratio
to that engine's plain number — so a row reads "did speculation help THIS
engine", not "which engine is faster".

The arm a file holds is its name, not anything inside it: `three_way.py`
stamps every row `engine: pie` whether the inferlet was the plain one or the
speculative loop, so the six arms are six files.

    python3 benches/spec_matrix_table.py <dir-of-{label}_{arm}.jsonl>

Arms: pie, piespec, mlx, mlxdraft, llama, llamadraft. No file for an arm
prints `n/a` (never attempted); an empty file prints `-` (attempted, no row
landed — a harness fault, not an engine's answer); a row that completed
nothing prints `refused`, which is the engine declining the shape.
"""
import json
import sys
from pathlib import Path

ARMS = ["pie", "piespec", "mlx", "mlxdraft", "llama", "llamadraft"]
HEADS = ["pie", "pie spec (k=1 loop)", "mlx-lm", "mlx-lm draft", "llama.cpp", "llama.cpp draft"]
# Each speculative arm and the plain arm it is a ratio of.
BASE = {"piespec": "pie", "mlxdraft": "mlx", "llamadraft": "llama"}
SHAPES = [("latency:8:1:64", "1 stream · 64"), ("tput:32:8:64", "8 conc · 64"), ("tput:16:8:256", "8 conc · 256")]
MODELS = ["gemma-26b-a4b", "gemma-31b", "qwen36-27b", "qwen38-27b"]
PRETTY = {"gemma-26b-a4b": "gemma-4-26B-A4B", "gemma-31b": "gemma-4-31B",
          "qwen36-27b": "Qwen3.6-27B", "qwen38-27b": "Qwen3.8-27B"}


def load(root: Path):
    """`(model, shape, arm) -> tok/s`, and the set of arms whose file exists."""
    cells, seen, empty = {}, set(), set()
    for path in sorted(root.glob("*.jsonl")):
        stem = path.stem
        arm = next((a for a in sorted(ARMS, key=len, reverse=True) if stem.endswith("_" + a)), None)
        if arm is None:
            continue
        model = stem[: -len(arm) - 1]
        seen.add((model, arm))
        if not path.read_text().strip():
            empty.add((model, arm))
        for line in path.read_text().splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            # A row that completed nothing is a refusal, not a zero.
            ok = row.get("completed") and not row.get("failed")
            cells[(model, row["shape"], arm)] = row["tok_s"] if ok else None
    return cells, seen, empty


def render(cells, seen, empty):
    def cell(model, shape, arm):
        if (model, arm) not in seen:
            return "n/a"
        if (model, arm) in empty:
            return "-"
        value = cells.get((model, shape, arm))
        if value is None:
            return "refused"
        base_arm = BASE.get(arm)
        if base_arm is None:
            return f"{value:.1f}"
        base = cells.get((model, shape, base_arm))
        return f"{value:.1f}" if not base else f"{value:.1f} ({value / base:.2f}×)"

    rows = []
    for model in MODELS:
        if not any(m == model for m, _ in seen):
            continue
        for at, (shape, label) in enumerate(SHAPES):
            rows.append([PRETTY.get(model, model) if at == 0 else "", label]
                        + [cell(model, shape, arm) for arm in ARMS])
    if not rows:
        print("no rows yet")
        return
    head = ["model", "shape"] + HEADS
    width = [max(len(head[i]), max((len(r[i]) for r in rows), default=0)) + 2 for i in range(len(head))]
    bar = lambda l, m, r: l + m.join("─" * w for w in width) + r
    print(bar("┌", "┬", "┐"))
    print("│" + "│".join(h.center(w) for h, w in zip(head, width)) + "│")
    for at, row in enumerate(rows):
        print(bar("├", "┼", "┤"))
        print("│" + "│".join(" " + c.ljust(w - 1) for c, w in zip(row, width)) + "│")
    print(bar("└", "┴", "┘"))


if __name__ == "__main__":
    root = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
    cells, seen, empty = load(root)
    render(cells, seen, empty)
