#!/usr/bin/env python3
"""Fold `three_way.py --out` rows (one JSON a line) into the preview-style
table: per model and shape, each engine's tok/s and pie's ratio to the
better baseline, plus the completed/out/prompt fields that say whether the
comparison was even (see `three_way.py`).

    python3 benches/matrix_table.py /tmp/warmstream/matrix.jsonl
"""
import json
import sys
from collections import defaultdict

rows = [json.loads(l) for l in open(sys.argv[1]) if l.strip()]
by = defaultdict(dict)
for r in rows:
    by[(r["model"], r["shape"])][r["engine"]] = r
print(f"{'model':<18}{'shape':<16}{'pie':>8}{'mlx':>8}{'llama':>8}{'pie/best':>10}  notes")
for (model, shape), engines in sorted(by.items()):
    def tok(e):
        return engines[e]["tok_s"] if e in engines and engines[e]["completed"] else None
    pie, mlx, llama = tok("pie"), tok("mlx"), tok("llamacpp")
    best = max(x for x in (mlx, llama) if x is not None) if (mlx or llama) else None
    ratio = f"{pie / best:.2f}x" if pie and best else "-"
    notes = []
    for e in ("pie", "mlx", "llamacpp"):
        if e in engines:
            r = engines[e]
            if r["failed"]:
                notes.append(f"{e}: {r['failed']} failed")
            notes.append(f"{e} out {r['out_mean']:.0f} prompt {r['prompt_mean']:.0f}")
    fmt = lambda x: f"{x:8.1f}" if x else f"{'-':>8}"
    print(f"{model:<18}{shape:<16}{fmt(pie)}{fmt(mlx)}{fmt(llama)}{ratio:>10}  {'; '.join(notes)}")
