# Wafer program output baseline

Captures everything the wafer program's compiler, build and host produce for every shipping config, so that a
refactor that must not change behaviour can be checked file for file against the tree before it.

- `configs.json`: the configs. A full one is a catalog deployment (`model`) planned through its trace with a preset
  and overrides (`preset`, `set`: what the chip / Pie runs used; the request ids are in `source`), over a staged
  prep's tiles (`$BENCH_JOB/<prep>`, `<prefix>`). A tiny one is a shape fixture (`shapes/`: the tiny checkpoints have
  no catalog deployment) with a tiny preset, and its frozen checkpoint (`$TINY_CKPTS/<ckpt>`) is prepped for that plan
  by the tree's own prep script, so the prep code is covered too. `levers`: the build's kernel levers.
  - Qwen3.5 9B (at RLV 4, as its staged tiles, and at RLV 5), 4B, 0.8B; Qwen3-8B (9 K4 units, gap slot, G2 12);
  - Gemma 4 E2B, E4B (as on chip: PYW 128, NES 4, NEG 20; and at its staged params: PYW 160, NEG 16), 12B (exact
    ring);
  - gpt-oss-20b (CYV 2, GLUV 1);
  - tiny: Qwen3.5 t08 and t4, Qwen3 dense with and without a gap slot, Gemma, Gemma 12B, gpt-oss.
- `capture.py <out> [names]`: per config (see its docstring for the files):
  - `plan/`: `wafer-compile plan`'s `plan.json` and `tiling.json`;
  - `prep/` (tiny only): the prep's files;
  - `build/`: `build.py`'s output for the plan (its `compiled/` holds the artifact) and `params.used` (the cslc params
    compare.py evaluates the layout with: none, every param a literal);
  - `host/`: the Rust host on the build's artifact (engine-cerebras `examples/artifact_baseline.rs`: `Artifact::read`,
    `ArtifactModel::open`, then one `greedy` launch on a recording device):
    - `pe.tsv`: every PE's words of every symbol, hashed;
    - `rcfg.txt`: every PE's rcfg in full;
    - `calls.log`: the device calls in order;
    - `derived.json`: the model's public fields (with `ARTIFACT_BASELINE_LOGITS=1`, a logits row's too).
  - `manifest.json`: sha256 of every file.
- `compare.py <reference> <candidate> [--allow globs]`: file-for-file diff of two captures, with the first
  differing lines of a text file; `build/` is compared by what it compiles (`layout_eval.py`: every PE the same
  source with the same params).

## Running it

```sh
export PY=<python with numpy>   # ARTIFACT_BASELINE_BIN / WAFER_COMPILE: prebuilt binaries, else capture.py builds them
python3 tools/wafer/baseline/capture.py /tmp/cand
python3 tools/wafer/baseline/compare.py <reference capture> /tmp/cand
```

Entry points the harness relies on (keep them, or adapt the harness in the same commit and say so):
- `wafer-compile plan` / `build`;
- engine-cerebras `examples/artifact_baseline.rs <artifact dir> <prep dir> <prefix> <vocab> <out dir>`;
- `build.py <out> <plan.json> [levers]`, writing `layout.csl`, one `.csl` a PE kind and `compiled/`;
- each family's prep CLI (`configs.json`'s `prep`), writing `<prefix>meta.json` and the `.npy` tiles.

## References

- `scratch/refactor/baseline/a4c1f516d/`: the capture before the layered build (big text files gzipped; the manifest
  hashes the plain text, and compare.py reads either); `tiny_hf/` there holds the frozen tiny checkpoints.
- `scratch/refactor/ma1/`: the capture before the trace-driven plan (its configs read a staged meta). Against it the
  expected differences are: `plan/` new; `notes.json`, the staged `meta.json` / `params.txt` and the tiny preps'
  `params.txt` gone; the Gemma, 12B and gpt-oss tiny preps' `meta.json` (no derived KV keys, `preset` for `geo`);
  `tiny_gptoss/host/derived.json`'s `positions` (the host now opens the program built for its plan, NE 8: 32, where
  the old host derived its own plan at NE 128). Everything else, `build/` included, is the same.

Known gaps:
- Linked ELFs are not hashed (cslc + the SDK, minutes a config). The generated sources determine them.
- Full preps are not re-run: they need the HF checkpoints and minutes to hours each. The tiny preps cover the prep
  code; the full ones are covered as the staged files the host places.
- The host's greedy runs on a zero device (every read is 0). It covers the host's writes and launches, not its
  reading of real results.
