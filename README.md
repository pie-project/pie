# engrain

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

*To engrain* is to work something into the fibre of a material rather than lay
it on the surface, so that it cannot be taken back out. That is the difference
this library is named for.

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

The measurement suite and the device verifications are `engrain_lab`, which is
research code and is not in the wheel; run them from a checkout with
`PYTHONPATH=src`:

```bash
python -m engrain_lab.verify              # every device check against the matcher
python -m engrain_lab.rigor.latency       # the step, against XGrammar and llguidance
```

## Use

```python
import engrain

engine  = engrain.Engine(vocabulary)          # bytes per token id
grammar = engine.compile(json_schema=schema)  # or regex=..., or ebnf=...

slots = engine.slots(64)
slots.admit(0, grammar)                       # a request arrives

while slots:
    logits  = model(...)
    tokens  = slots.sample(logits)            # the constraint is applied here
    verdict = slots.commit(tokens)            # advance; the next mask is ready
    slots.release(finished)                   # a request leaves
```

There is no `capture()` to remember: the first step records the graph and every
step after replays it, with no host work and no synchronisation. The recording
stays valid however slots are reassigned between steps, which is what makes it
composable with a serving engine's own captured decode step.

### Slots, not a batch

The rectangle is not an implementation detail, it is what makes the step
capturable, so it is the thing you are given. `admit` puts a request in a slot
and `release` takes it out — which is what continuous batching does, and what an
API that resets the whole batch cannot express. Slots may hold different
grammars, and the step is still one launch.

### What the grammar does not enforce

```python
grammar = engine.compile(json_schema=schema)
for note in grammar.relaxations:
    print(note["keyword"], "at", note["at"])
    print(" ", note["effect"])
    print(" ", note["remedy"] or "nothing here would fix it")
```

```
required at #/properties/order
  an object here may close with properties missing: 9 are required and the
  parser can carry 7 at once
  require fewer properties here, or close the object with
  `additionalProperties: false` to raise the budget
```

The mask may admit more than the source allows and never less. Each entry names
the keyword, points at the place with a JSON pointer, says what the mask now
admits, and gives the edit that would enforce it — and the list is empty when
there is nothing to check. A constrained decoder that widens a schema without
saying so is the failure this list exists to prevent; one that says only *that*
it widened sends its author looking.

### Handing the constraint to a sampler you do not own

```python
slots.apply(logits)      # -inf on every forbidden token, in place
mask = slots.mask()      # or the packed bitmask itself, (slots, words)
```

Underneath, the allowed set is a device object rather than a mask on its way
from the host, which is what would let a sampler read a few hundred candidates
instead of a hundred and fifty thousand. Measured on Qwen3's vocabulary at
batch 512 against applying the mask to every row: **2.68x when every row is
sparse, 1.05x when half are dense**, and 0.86x for the sync-free alternative.
The step distribution is bimodal — half of all steps admit under four thousand
tokens and half admit nearly everything — so a mixed batch, which is what
continuous batching gives, sits near parity. Acting on it still costs one host
synchronisation, because a sampler's output shape is its row count. The set
being resident is necessary and not sufficient, and `sample` will change when a
sampler accepts a device-side row count.

### Checking it

Every grammar carries a host-side reference parser. The device is checked
against it rather than trusted:

```python
matcher = grammar.matcher(0)
reference = torch.zeros(engine.mask_words, dtype=torch.int32)
matcher.fill_bitmask(reference)
assert torch.equal(slots.mask()[0].cpu(), reference)
```

`commit` returns a `Verdict`: `terminated` when the parser refused the token a
slot was given, and `narrowed` when a ceiling was reached and the next mask may
be *narrower* than the grammar allows. Narrowing is the one failure this engine
must never commit quietly, so it is reported rather than absorbed. Both are
device tensors; `verdict.ok` is the moment you choose to synchronise.

The layers underneath — the compiler, the pool, the arena, the batch and its
graph — are `engrain.internals`, and reaching for them is a statement that you
want to make those decisions yourself.

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
| `src/engrain/` | the public library |
| `src/engrain_lab/` | the research tree: kernels, benchmarks, earlier prototypes |
| `rust/crates/engrain-ir` | front ends — JSON Schema, regex, EBNF |
| `rust/crates/engrain-lex` | lexer construction and vocabulary grouping |
| `rust/crates/engrain-lr` | LALR(1) table construction |
| `rust/crates/engrain-tables` | the compile pipeline and the device artifact |
| `rust/crates/engrain-run` | the reference matcher the device is checked against |
| `third_party/vllm` | a vLLM backend, for end-to-end measurement |
| `GOAL.md` | the research narrative and every measurement |
| `docs/prototype-history.md` | the earlier prototype, kept for its record |

## Status

Research code under active development, and not yet a stable interface.
