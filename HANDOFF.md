# Handoff

Where gpugrammar stands, what is proven, and what is known to be broken.
Written to be read by someone picking this up cold. `GOAL.md` holds the
project's direction and the reasoning behind the design; this file holds the
current state.

## What exists

**A Rust compiler** (`rust/`, six crates) that takes a JSON Schema, EBNF or
regex and produces a device-ready artifact: LALR(1) ACTION/GOTO tables, a byte
lexer, and the vocabulary grouped by what each token does to the lexer.

| crate | what it does |
| --- | --- |
| `gpugrammar-ir` | Front end, vendored from pie-grammar: EBNF, JSON Schema, regex, FSMs |
| `gpugrammar-lex` | Regularity analysis, terminal extraction, lexer determinisation, vocabulary grouping |
| `gpugrammar-lr` | EBNF-to-BNF flattening, LALR(1) construction, a reference parser |
| `gpugrammar-tables` | Artifact emission; the `compile`, `coverage` and `explain` binaries |
| `gpugrammar-run` | Reference matcher over the artifact; the `validate` and `trace` binaries |
| `gpugrammar-py` | PyO3 bindings |

**A vLLM fork** at `third_party/vllm`, submodule of `ingim/vllm` branch
`gpugrammar`, branched from upstream `910cc8543`. It registers `gpugrammar` as a
first-class structured-output backend. Run vLLM from there, not from a
site-packages install.

**A GPU sampler** (`src/gpu_lr1/ragged_sampler.py`, `wide_sampler.py`) that
fuses masking into sampling. **It is not yet wired to the compiler.** The
measured GPU wins live here and are not in the serving path.

## What is proven

Both backends produce valid output under vLLM, with the engine in its own
subprocess and nothing monkeypatched:

```
xgrammar    64/64 valid | 1182 tokens in 0.64s = 1858 tok/s
gpugrammar  64/64 valid | 1247 tokens in 0.77s = 1619 tok/s
```

All 21 Rust test suites pass. Coverage over 533 real JSONSchemaBench schemas:
461 lower to a grammar, 434 reach LALR(1) tables. Median parser state count is
3; the median schema needs no stack at all because it is purely regular.

## What is broken, in priority order

**1. We reject documents we should accept.** This is the important one. A byte
level differential — compile each schema, feed the model-generated instance one
byte at a time — still rejects some documents that XGrammar produced under the
same schema. Reproduce a single case with:

```
cargo run --release --bin gpugrammar-trace -- results/jsonschemabench-instances.json <index>
```

It prints, per byte, the lexer state, the terminals the scan emitted, the
terminals a pending lexeme could still become, and whether the parser accepted.
That tool found every bug listed below and is the right place to start.

**2. Coverage went down while correctness went up.** LALR(1) tables: 461 → 434.
Each correctness fix changed the lexicon, and a coarser lexicon means more
reduce/reduce conflicts. The conflicts are real and unexamined. `gpugrammar-explain`
dumps the terminals and productions for one schema.

**3. Length bounds explode the lexer.** A DFA can only hold a counter by
unrolling it, so `"maxLength": 2048` over UTF-8 costs about seventy states per
counted position: one schema asked for 209,001 lexer states. 94 of 533 schemas
carry a bound, 38 above 256. `build_lexer_within` abandons construction over a
budget so the rest of the corpus stays measurable, but the real answer is a
counter-augmented lexer, where the runtime state is `(dfa_state, counter)` and
the counter is not compiled away. That is unbuilt.

**4. Emission is slow for large lexers.** Reachability is now one fixpoint
rather than a search per group, and pending-terminal lists are shared by lexer
state rather than copied per group. It is still too slow to emit a
20,000-state lexer against a byte vocabulary: a 60-instance validation run did
not finish in 900 seconds, which is why there is no single acceptance number
above. Not profiled properly yet.

**5. 72 schemas fail before the lexer.** Front-end gaps, all in the vendored
JSON Schema lowering: `allOf` with multiple schemas (18), `required` naming a
property not in `properties` (21), `pattern` with length bounds (12), and a
tail of smaller ones.

**6. We are 13% slower than XGrammar.** Expected: this path still fills a CPU
bitmask, because that is the only interface vLLM offers, and batch 64 is far
below where mask fill dominates. The GPU sampler is not connected. No claim
should be made until it is.

## Bugs fixed this session, and what each one teaches

**Exponential property expansion.** `enumerate_properties` enumerated the
powerset of optional properties — 2^k alternatives, each with a full copy of
every property's value grammar. One schema became 22,796 productions and
200,836 terminals. A linear state construction already existed and was only
used above eight optional properties; deleting the enumeration entirely fixed
15 reduce/reduce conflicts, cut the largest terminal count 306-fold, and took
the whole corpus from minutes to 0.57 seconds. **The grammar was never
un-LALR; the front end was making it un-LALR.**

**The start symbol was whatever came first.** The skeleton follows declaration
order, and `flatten` took its first rule as the start. For any schema whose
root is not declared first, the parser was parsing a subexpression. `Lexicon`
now records the root. This is the kind of bug that unit tests on hand-written
grammars will never find, because hand-written grammars declare the root first.

**Nullable terminals can never be emitted.** `__json_ws ::= [ \t\n\r]*` matched
the empty string, and a scanner cannot emit an empty lexeme, so any document
without whitespace was unparseable — as was `""`, whose body terminal was also
nullable. Nullability now moves into the skeleton as `ε | terminal`, and the
terminal's automaton is stripped of ε by giving it a fresh non-accepting start
that takes over the byte edges leaving the old start's epsilon closure.

**One accepting terminal per DFA state is not enough.** Generated grammars are
lexically ambiguous by construction: a declared property name `"id"` is also a
generic JSON string. Collapsing to the lowest-numbered terminal lost the one
the parser wanted. States now carry the whole accepting set, a scan returns
candidate terminal sequences, and the parser picks the one it can follow —
which is the LR viable-prefix property doing the disambiguation.

**Terminals must be whole lexemes, and the front end must say so.** Splitting
`'"' body '"'` into three terminals makes the body a terminal whose class
`[^"\\]*` overlaps every punctuation terminal: after a colon the scanner kept
munching as a string body and never committed the colon. Merging maximal
regular *subtrees* does not help, because the three pieces are siblings, not a
subtree. Merging maximal regular *sibling runs* over-merges — `string ':'`
became one terminal. The fix is neither: the front end now declares each
lexical unit as a rule, and the extractor's existing "a whole regular rule is
one terminal" policy does the rest.

## Reproducing

```sh
# Rust
cd rust && cargo test --release

# Coverage over real schemas
python -c "import json; d=json.load(open('results/jsonschemabench-instances.json')); \
  json.dump([i['schema'] for i in d['instances']], open('/tmp/schemas.json','w'))"
cargo run --release --bin gpugrammar-coverage -- /tmp/schemas.json 0

# Byte-level acceptance against model-generated documents
GPUGRAMMAR_MAX_LEXER_STATES=4000 \
  cargo run --release --bin gpugrammar-validate -- results/jsonschemabench-instances.json 5

# Why one document was rejected
cargo run --release --bin gpugrammar-trace -- results/jsonschemabench-instances.json 2

# Python
python -m unittest discover -s tests

# vLLM, from the submodule
PATH="$PWD/.venv/bin:$PATH" FLASHINFER_DISABLE_VERSION_CHECK=1 \
  python src/gpu_lr1/vllm_smoke.py --backend gpugrammar --prompts 64
```

## Environment

The venv is fragile and several things will re-break it.

- `python3 -m venv` fails; use `uv venv` and `uv pip install --python .venv/bin/python`.
- vLLM is installed editable from the submodule with `VLLM_USE_PRECOMPILED=1`,
  which downloads a wheel and builds only the Python parts. That is right for
  this work: every change here is pure Python. It needs `setuptools_rust` and
  `setuptools_scm` present with `--no-build-isolation`.
- `flashinfer-python` and `flashinfer-cubin` versions must match or FlashInfer
  refuses to start. They currently do not; `FLASHINFER_DISABLE_VERSION_CHECK=1`
  bypasses it. Do not fix this by downgrading `flashinfer-python`, which drags
  torch to 2.10 and breaks vLLM.
- FlashInfer JIT needs `ninja` on `PATH`, which lives in `.venv/bin`.
- Reinstalling torch can leave `nvidia-nccl-cu13` stale, which shows up as
  `undefined symbol: ncclDevCommDestroy`. Reinstall nccl, not torch.

## Next

1. Close the acceptance gap. Trace each rejection; do not guess.
2. Understand the reduce/reduce conflicts rather than trading them against
   correctness.
3. Wire the artifact to the GPU sampler. Until then no performance claim holds.
4. Counter-augmented lexer for length bounds.
5. Measure with vLLM's `benchmark_serving_structured_output.py` at batch sizes
   where mask fill actually matters.
