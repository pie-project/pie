#!/usr/bin/env python3
"""The two-box view: pie against the best baseline, on M1 Max and M4 Pro.

Each engine is credited with its BEST arm — plain or speculative — because
the question a deployment asks is "how fast is this engine on this box", not
"how fast is one of its code paths". Speculation helps some engines on some
shapes and hurts on others (see `spec_matrix_table.py` for the arm-by-arm
split), and a table that fixed every engine to its plain arm would flatter
whichever engine happened to have the worst speculative path.

The M4 Pro rows are read from this box's `three_way.py` output. The M1 Max
rows are the peer session's, transcribed — they cannot be recomputed here.

    python3 benches/two_box_table.py <dir-of-{label}_{arm}.jsonl>
"""
import json
import sys
from pathlib import Path

ARMS = ["pie", "piespec", "mlx", "mlxdraft", "llama", "llamadraft"]
# Which arms belong to which engine, best-of taken across each group.
ENGINES = {"pie": ["pie", "piespec"], "mlx": ["mlx", "mlxdraft"], "llama": ["llama", "llamadraft"]}
SHAPES = [("latency:8:1:64", "1 stream · 64"), ("tput:32:8:64", "8 conc · 64"), ("tput:16:8:256", "8 conc · 256")]
MODELS = ["gemma-26b-a4b", "gemma-31b", "qwen36-27b", "qwen38-27b"]
PRETTY = {"gemma-26b-a4b": "gemma-4-26B-A4B", "gemma-31b": "gemma-4-31B",
          "qwen36-27b": "Qwen3.6-27B", "qwen38-27b": "Qwen3.8-27B"}

# The peer session's M1 Max table (32 GB, 24 GPU cores), tok/s per arm.
# Their note: the gemma rows predate their BK=64 tile change, the Qwen rows
# follow it, so the gemma pie numbers there are pessimistic.
M1 = {
    ("gemma-26b-a4b", "latency:8:1:64"): dict(pie=59.1, piespec=57.6, mlx=36.0, mlxdraft=18.5, llama=41.6, llamadraft=31.1),
    ("gemma-26b-a4b", "tput:32:8:64"):   dict(pie=128.3, piespec=85.2, mlx=63.4, mlxdraft=18.6, llama=74.3, llamadraft=56.1),
    ("gemma-26b-a4b", "tput:16:8:256"):  dict(pie=141.0, piespec=97.2, mlx=76.9, mlxdraft=20.4, llama=77.9, llamadraft=63.9),
    ("gemma-31b", "latency:8:1:64"):     dict(pie=14.0, piespec=12.9, mlx=9.6, mlxdraft=5.4, llama=8.6, llamadraft=6.3),
    ("gemma-31b", "tput:32:8:64"):       dict(pie=28.2, piespec=18.2, mlx=12.3, mlxdraft=5.4, llama=None, llamadraft=None),
    ("gemma-31b", "tput:16:8:256"):      dict(pie=32.0, piespec=20.2, mlx=13.9, mlxdraft=2.6, llama=None, llamadraft=None),
    ("qwen36-27b", "latency:8:1:64"):    dict(pie=16.3, piespec=15.4, mlx=11.1, mlxdraft=None, llama=10.0, llamadraft=3.8),
    ("qwen36-27b", "tput:32:8:64"):      dict(pie=40.8, piespec=34.1, mlx=14.5, mlxdraft=None, llama=14.9, llamadraft=12.4),
    ("qwen36-27b", "tput:16:8:256"):     dict(pie=51.2, piespec=44.6, mlx=16.3, mlxdraft=None, llama=16.6, llamadraft=12.5),
    ("qwen38-27b", "latency:8:1:64"):    dict(pie=16.4, piespec=14.0, mlx=11.0, mlxdraft=None, llama=12.1, llamadraft=10.6),
    ("qwen38-27b", "tput:32:8:64"):      dict(pie=40.7, piespec=31.9, mlx=13.2, mlxdraft=None, llama=16.1, llamadraft=12.1),
    ("qwen38-27b", "tput:16:8:256"):     dict(pie=51.3, piespec=42.9, mlx=15.3, mlxdraft=None, llama=19.1, llamadraft=14.3),
}


def load_m4(root: Path):
    cells = {}
    for path in sorted(root.glob("*.jsonl")):
        arm = next((a for a in sorted(ARMS, key=len, reverse=True) if path.stem.endswith("_" + a)), None)
        if arm is None:
            continue
        model = path.stem[: -len(arm) - 1]
        for line in path.read_text().splitlines():
            if line.strip():
                row = json.loads(line)
                ok = row.get("completed") and not row.get("failed")
                cells[(model, row["shape"], arm)] = row["tok_s"] if ok else None
    return cells


def best(get, engine):
    """The better of an engine's plain and speculative arms, and which won."""
    got = [(get(arm), arm) for arm in ENGINES[engine]]
    got = [(v, a) for v, a in got if v]
    if not got:
        return None, None
    value, arm = max(got)
    return value, arm


def render(m4):
    rows = []
    for model in MODELS:
        for at, (shape, label) in enumerate(SHAPES):
            row = [PRETTY[model] if at == 0 else "", label]
            for box in ("m1", "m4"):
                get = ((lambda arm: M1.get((model, shape), {}).get(arm)) if box == "m1"
                       else (lambda arm: m4.get((model, shape, arm))))
                pie, pie_arm = best(get, "pie")
                mlx, _ = best(get, "mlx")
                llama, _ = best(get, "llama")
                base = max([x for x in (mlx, llama) if x] or [0]) or None
                row += [
                    f"{pie:.1f}{'*' if pie_arm == 'piespec' else ''}" if pie else "-",
                    f"{mlx:.1f}" if mlx else "-",
                    f"{llama:.1f}" if llama else "OOM",
                    f"{pie / base:.2f}×" if pie and base else "-",
                ]
            rows.append(row)

    head = ["model", "shape",
            "M1 pie", "M1 mlx-lm", "M1 llama", "M1 pie/best",
            "M4 pie", "M4 mlx-lm", "M4 llama", "M4 pie/best"]
    width = [max(len(head[i]), max(len(r[i]) for r in rows)) + 2 for i in range(len(head))]
    bar = lambda l, m, r: l + m.join("─" * w for w in width) + r
    print(bar("┌", "┬", "┐"))
    print("│" + "│".join(h.center(w) for h, w in zip(head, width)) + "│")
    for row in rows:
        print(bar("├", "┼", "┤"))
        print("│" + "│".join(" " + c.ljust(w - 1) for c, w in zip(row, width)) + "│")
    print(bar("└", "┴", "┘"))
    print("\n* = the speculative arm was the engine's better one on that row.")


if __name__ == "__main__":
    render(load_m4(Path(sys.argv[1] if len(sys.argv) > 1 else ".")))
