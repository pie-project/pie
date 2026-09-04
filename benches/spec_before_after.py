#!/usr/bin/env python3
"""pie's plain and speculative arms from two matrix directories, side by side.

    python3 benches/spec_before_after.py <before-dir> <after-dir>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from two_box_table import MODELS, PRETTY, SHAPES, load_m4  # noqa: E402


def cell(m, model, shape, arm):
    v = m.get((model, shape, arm))
    return f"{v:.1f}" if v else "-"


def ratio(m, model, shape):
    a, b = m.get((model, shape, "pie")), m.get((model, shape, "piespec"))
    return f"{b / a:.2f}×" if a and b else "-"


if __name__ == "__main__":
    before, after = (load_m4(Path(d)) for d in sys.argv[1:3])
    head = ["model", "shape", "pie before", "pie after", "spec before", "spec after", "spec/plain before", "spec/plain after"]
    rows = []
    for model in MODELS:
        for at, (shape, label) in enumerate(SHAPES):
            rows.append([PRETTY[model] if at == 0 else "", label,
                         cell(before, model, shape, "pie"), cell(after, model, shape, "pie"),
                         cell(before, model, shape, "piespec"), cell(after, model, shape, "piespec"),
                         ratio(before, model, shape), ratio(after, model, shape)])
    width = [max(len(head[i]), max(len(r[i]) for r in rows)) + 2 for i in range(len(head))]
    bar = lambda l, m, r: l + m.join("─" * w for w in width) + r
    print(bar("┌", "┬", "┐"))
    print("│" + "│".join(h.center(w) for h, w in zip(head, width)) + "│")
    for row in rows:
        print(bar("├", "┼", "┤"))
        print("│" + "│".join(" " + c.ljust(w - 1) for c, w in zip(row, width)) + "│")
    print(bar("└", "┴", "┘"))
