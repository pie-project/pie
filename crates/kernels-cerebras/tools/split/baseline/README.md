# Role-split output baseline

Captures everything the role-split build and host produce for every shipping config, so that a refactor
that must not change behaviour can be checked file for file against the tree before it.

- `configs.json`: the configs. Full ones read a staged prep (`$SPLIT_JOB/<prep>`), each with the
  `PIE_CEREBRAS_SPLIT` overrides and cslc params the chip / Pie runs used (the request ids are in
  `source`). Tiny ones prep a frozen tiny checkpoint (`$SPLIT_TINY/<ckpt>`) with the tree's own prep
  script, so the prep code is covered too.
  - Qwen3.5 9B (also at RLV 5), 4B, 0.8B; Qwen3-8B (9 K4 units, gap slot, G2 12);
  - Gemma 4 E2B, E4B (as on chip: PYW 128, NES 4, WIN 0, NEG 20; and at its staged params), 12B (exact ring);
  - gpt-oss-20b (CYV 2, GLUV 1);
  - tiny: Qwen3.5 t08 and t4, Qwen3 dense with and without a gap slot, Gemma, Gemma 12B, gpt-oss.
- `capture.py <out> [names]`: per config (see its docstring for the files):
  - `host/`: the Rust host through its public entry point (`SplitSpec::parse` + `SplitModel::open`, then one
    `greedy` launch) on a recording device, `examples/split_baseline.rs`:
    - `pe.tsv`: every PE's words of every symbol, hashed;
    - `rcfg.txt`: every PE's rcfg in full;
    - `calls.log`: the device calls in order;
    - `derived.json`: the model's public fields.
  - `build/`: the family's mkfull output and `params.used` (the cslc params);
  - `prep/` (tiny only): the prep's files;
  - `meta.json` / `params.txt` (full only): the staged prep's;
  - `py_rcfg.txt`: the Python geometry's rcfg in the same format. At the baseline it equals `host/rcfg.txt`
    on every config that has one (`notes.json`).
  - `manifest.json`: sha256 of every file.
- `compare.py <reference> <candidate> [--allow globs]`: file-for-file diff of two captures, with the first
  differing lines of a text file.

## Reference

`scratch/refactor/baseline/a4c1f516d/` holds the capture of `a4c1f516d` (cerebras-model-support before
the refactor):
- big text files are gzipped (`.gz`); the manifest hashes the plain text, and compare.py reads either;
- `tiny_hf/`: the frozen tiny checkpoints (copied from scratch/split);
- `capture.out`: the run's log.

## Final verification (after the refactor)

```sh
export PY=<python with numpy> SPLIT_BASELINE_BIN=<the refactored tree's split_baseline>   # or let capture.py build it
python3 tools/split/baseline/capture.py /tmp/cand
python3 tools/split/baseline/compare.py scratch/refactor/baseline/a4c1f516d /tmp/cand
```

Entry points the harness relies on (keep them, or adapt the harness in the same commit and say so):
- `examples/split_baseline.rs`: `SplitSpec::parse("<prep>:<prefix>:<k=v,..>")`, `SplitModel::open(Box<dyn Device>,
  &spec, vocab)`, `SplitModel::{greedy, vocab, hidden, cap, positions, launches, steps, fed}`, the `Device`
  trait;
- each family's mkfull CLI (`configs.json`'s `mkfull`), writing `layout.csl` and one `.csl` a PE kind into
  the out dir;
- each family's prep CLI (`configs.json`'s `prep`), writing `<prefix>meta.json`, `params.txt` and the `.npy`
  tiles;
- `py_rcfg.txt` uses the Python geometries (`split_geo*.py`). If they are removed, that file goes
  `missing`: expected. It is the Python-vs-Rust cross-check at the baseline, not part of the contract.

Acceptable differences (nothing else):
- `*/build/*.py`: the payload's copies of the Python host tools (allowed by default);
- `*/py_rcfg.txt`, `*/notes.json`: missing or changed once the Python geometry is gone;
- a moved or renamed tool: change `configs.json` in the refactor and keep the outputs identical.

Known gaps:
- Linked ELFs are not hashed (cslc + the SDK, minutes a config). The generated sources plus `params.used`
  determine them.
- Full preps are not re-run: they need the HF checkpoints and minutes to hours each. The tiny preps cover the
  prep code; the full ones are covered as the staged files the host places.
- The gpt-oss Python geometry gives per-kind hashes only (`rcfg_hash.py`).
- The Qwen gap-slot configs have no Python rcfg (`split_geo.py` has no gap slot).
- The host's greedy runs on a zero device (every read is 0). It covers the host's writes and launches, not
  its reading of real results.
