# gpugrammar

Constrained decoding whose parser state lives on the GPU.

A constrained decoder answers one question at every step: which of the model's
tokens may come next? Every deployed system answers it on the host — which
means the answer cannot be inside the CUDA graph a serving engine records for
its decode step. A graph holds device work; host work placed inside it does not
go in at all. Attempting to capture XGrammar's mask fill produces an empty
graph, and replay then reproduces whatever the host buffer happened to hold.

This library moves the parser onto the device so that it can be captured. That
is the claim: not that a mask is built faster, but that the state is somewhere
a serving engine can reach without returning to the host.

## Install

Needs an NVIDIA GPU, PyTorch with CUDA, and a Rust toolchain.

```bash
pip install .
```

To work on it, build the extension into the source tree instead, so an edit
to a kernel takes effect without a reinstall:

```bash
maturin develop --release
```

The measurement suite and the device verifications are `gpu_lr1`, which is
research code and is not in the wheel; run them from a checkout with
`PYTHONPATH=src`:

```bash
python -m gpu_lr1.verify              # every device check against the matcher
python -m gpu_lr1.rigor.latency       # the step, against XGrammar and llguidance
```

## Use

```python
import gpugrammar, torch

engine = gpugrammar.Engine(vocabulary)            # bytes per token id
grammar = engine.compile_json_schema(schema)      # or compile_regex(...)

batch = engine.batch(size=64)
batch.set_grammars([grammar] * 64)

batch.capture()                                   # record the graphs, once

while decoding:
    mask = batch.step(sampled)                    # advance + fill, one replay
    logits.masked_fill_(...)                      # your sampler
```

`step` is `advance` then `fill_mask` as a single recording, which is the path a
decode loop wants: only the sample sits between a fill and the advance that
follows it, and nothing at all sits between that advance and the next fill.
The two are also available separately.

After `capture()`, all three are graph replays: no host work, no
synchronisation, and the recording stays valid however the batch's grammars are
reassigned between steps. That is what makes it composable with the engine's
own captured decode step.

### Sampling from the set instead of the mask

A vocabulary-wide mask is the wrong shape for what reads it. At a structural
position a schema admits a few hundred of a hundred and fifty thousand tokens,
and the sampler that follows sorts and normalises all of them anyway.

```python
ids, counts, kind = batch.shortlist()             # all on the device
```

`kind[i]` is 0 when `ids[i]` names what row `i` *admits* and 1 when it names
what it *forbids* — whichever list is shorter, decided on the device, because
a caller choosing for itself would have to read a count on the host.

Measured on Qwen3's vocabulary at batch 512, against applying the mask to every
row: **2.68x when every row is sparse, 1.05x when half are dense**, and 0.86x
for the sync-free alternative of sampling both ways and selecting. The step
distribution is bimodal — half of all steps admit under four thousand tokens
and half admit nearly everything — so a mixed batch, which is what continuous
batching gives, sits near parity. Acting on `kind` still costs one host
synchronisation, because a sampler's output shape is its row count. The set
being resident is necessary and not sufficient.

A batch may hold sequences under different grammars — which is what a serving
batch looks like, since requests bring their own — and it is still one launch.

### Checking it

Every grammar carries a host-side reference parser. The device is checked
against it rather than trusted:

```python
matcher = grammar.matcher(0)
reference = torch.zeros(engine.mask_words, dtype=torch.int32)
matcher.fill_bitmask(reference)
assert torch.equal(batch.fill_mask()[0].cpu(), reference)
```

`batch.problems()` returns `(terminated, overflow)` per sequence. `overflow`
means a ceiling was reached and the mask that follows may be *narrower* than the
grammar allows. Narrowing is the one failure this engine must never do quietly.

## What it supports

- **JSON Schema** — 510 of JSONSchemaBench's 533 compile; of the instances that
  are valid JSON satisfying their own schema, 98.0% are accepted (99.8% with a
  larger configuration ceiling).
- **Regular expressions** — compiled to the same tables.
- **EBNF** — `compile_ebnf(source, root)`, including grammars a regular
  language cannot express. On a SQL SELECT subset the parser stack grows one
  entry per level of parenthesis nesting, which is the thing a DFA cannot do.

Grammars are LALR(1), and a conflicted cell is forked at runtime rather than
refused at construction, so ambiguity is handled rather than rejected. No schema
in the corpus is refused for its grammar class.

## Where it stands

Against XGrammar 0.2.3 and llguidance 1.7.6 on one A100, charging every engine
for a mask and an advance per sequence — which is what a decode step costs:

| | batch 32 | batch 128 | batch 512 |
|---|---:|---:|---:|
| regex, vs XGrammar | 2.26x | 6.42x | 21.33x |
| regex, vs llguidance | 4.00x | 10.31x | 37.56x |
| JSON Schema, vs XGrammar | **0.94x** | 3.04x | 9.53x |
| SQL nested 32 deep, vs XGrammar | 177x | 653x | 1,739x |
| SQL nested 32 deep, vs llguidance | 1.91x | 6.10x | 17.26x |

Two things the table does not say. Our cost is flat in the nesting depth and
llguidance's is not — at batch 128, depth 1 to 32 takes them from 654 µs to
1,653 and us from 226 to 271 — which is the axis a device-resident stack is
for. And XGrammar's collapse on SQL is real rather than a misuse: one
sequence's fill reaches 70 ms where an expression can continue with almost any
token.

We lose to llguidance at batch 1 on SQL (0.15x), to XGrammar on JSON Schema at
batch 32 (0.94x), on compile time throughout (120 ms median against 16, and
1.3 s for the SQL grammar against 88 ms), and on memory (3.27 MB resident per
schema against a 52 KiB host cache, 91 MB for SQL). Those are real and they are
in `GOAL.md` with the rest.

## Reproducing

```bash
python -m unittest discover -s tests          # correctness, including the API
```

The measurements above come from probes kept outside the package; `GOAL.md`
records what each one measures, what it found, and — for a dozen of them — what
was tried and reverted, with the reason.

## Layout

| | |
|---|---|
| `src/gpugrammar/` | the public library |
| `src/gpu_lr1/` | the research tree: kernels, benchmarks, earlier prototypes |
| `rust/crates/gpugrammar-ir` | front ends — JSON Schema, regex, EBNF |
| `rust/crates/gpugrammar-lex` | lexer construction and vocabulary grouping |
| `rust/crates/gpugrammar-lr` | LALR(1) table construction |
| `rust/crates/gpugrammar-tables` | the compile pipeline and the device artifact |
| `rust/crates/gpugrammar-run` | the reference matcher the device is checked against |
| `third_party/vllm` | a vLLM backend, for end-to-end measurement |
| `GOAL.md` | the research narrative and every measurement |
| `docs/prototype-history.md` | the earlier prototype, kept for its record |

## Status

Research code under active development, and not yet a stable interface.
