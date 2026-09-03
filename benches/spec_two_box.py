#!/usr/bin/env python3
"""The speculation matrix twice, one box above the other, same six columns.

Stacked rather than interleaved: each table then reads on its own as "what
does speculation do to each engine on THIS machine", and the two are
compared by looking down the same column, not across a doubled header.

    python3 benches/spec_two_box.py <dir-of-{label}_{arm}.jsonl>
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from two_box_table import M1, MODELS, PRETTY, SHAPES, load_m4  # noqa: E402

ARMS = ["pie", "piespec", "mlx", "mlxdraft", "llama", "llamadraft"]
HEADS = ["pie", "pie spec (k=1 loop)", "mlx-lm", "mlx-lm draft", "llama.cpp", "llama.cpp draft"]
BASE = {"piespec": "pie", "mlxdraft": "mlx", "llamadraft": "llama"}
# What a missing number means, per box: the peer recorded mlx-lm refusing the
# Qwen hybrids outright and llama.cpp running out of memory on 32 GB.
M1_NOTE = {("qwen36-27b", "mlxdraft"): "unsupported", ("qwen38-27b", "mlxdraft"): "unsupported",
           ("gemma-31b", "llama"): "OOM", ("gemma-31b", "llamadraft"): "OOM"}
M4_NOTE = {("qwen36-27b", "mlxdraft"): "refused", ("qwen38-27b", "mlxdraft"): "refused"}


def table(title, get, note):
    rows = []
    for model in MODELS:
        for at, (shape, label) in enumerate(SHAPES):
            row = [PRETTY[model] if at == 0 else "", label]
            for arm in ARMS:
                value = get(model, shape, arm)
                if not value:
                    row.append(note.get((model, arm), "-"))
                    continue
                base = get(model, shape, BASE[arm]) if arm in BASE else None
                row.append(f"{value:.1f}" if not base else f"{value:.1f} ({value / base:.2f}×)")
            rows.append(row)

    head = ["model", "shape"] + HEADS
    width = [max(len(head[i]), max(len(r[i]) for r in rows)) + 2 for i in range(len(head))]
    bar = lambda l, m, r: l + m.join("─" * w for w in width) + r
    print(title)
    print(bar("┌", "┬", "┐"))
    print("│" + "│".join(h.center(w) for h, w in zip(head, width)) + "│")
    for row in rows:
        print(bar("├", "┼", "┤"))
        print("│" + "│".join(" " + c.ljust(w - 1) for c, w in zip(row, width)) + "│")
    print(bar("└", "┴", "┘"))


if __name__ == "__main__":
    m4 = load_m4(Path(sys.argv[1] if len(sys.argv) > 1 else "."))
    table("M1 Max · 32 GB · 24 GPU cores",
          lambda model, shape, arm: M1.get((model, shape), {}).get(arm), M1_NOTE)
    print()
    table("M4 Pro · 48 GB · 20 GPU cores",
          lambda model, shape, arm: m4.get((model, shape, arm)), M4_NOTE)
