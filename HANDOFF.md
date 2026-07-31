# Handoff

Where engrain stands, what is proven, and what is known to be broken.
Written to be read by someone picking this up cold. `GOAL.md` holds the
project's direction and the reasoning behind the design; this file holds the
current state.

## What exists

**A Rust compiler** (`rust/`, six crates) that takes a JSON Schema, EBNF or
regex and produces a device-ready artifact: LALR(1) ACTION/GOTO tables, a byte
lexer, and the vocabulary grouped by what each token does to the lexer.

| crate | what it does |
| --- | --- |
| `engrain-ir` | Front end, vendored from pie-grammar: EBNF, JSON Schema, regex, FSMs |
| `engrain-lex` | Regularity analysis, terminal extraction, lexer determinisation, vocabulary grouping |
| `engrain-lr` | EBNF-to-BNF flattening, LALR(1) construction, a reference parser |
| `engrain-tables` | Artifact emission; the `compile`, `coverage` and `explain` binaries |
| `engrain-run` | Reference matcher over the artifact; the `validate` and `trace` binaries |
| `engrain-py` | PyO3 bindings |

**A vLLM fork** at `third_party/vllm`, submodule of `ingim/vllm` branch
`engrain`, branched from upstream `910cc8543`. It registers `engrain` as a
first-class structured-output backend. Run vLLM from there, not from a
site-packages install.

**A GPU sampler** (`src/engrain_lab/ragged_sampler.py`, `wide_sampler.py`) that
fuses masking into sampling. **It is not yet wired to the compiler.** The
measured GPU wins live here and are not in the serving path.

## What is proven

Both backends produce valid output under vLLM, with the engine in its own
subprocess and nothing monkeypatched:

```
xgrammar    64/64 valid | 1182 tokens in 0.64s = 1858 tok/s
engrain  64/64 valid | 1222 tokens in 0.88s = 1392 tok/s
```

The grammar is exact on a small schema: correct documents are accepted, and
out-of-order properties, a trailing comma and an undeclared property are each
rejected at the byte where they go wrong.

All 23 Rust test suites pass. Coverage over 533 real JSONSchemaBench schemas:
461 lower to a grammar, 434 reach LALR(1) tables. Median parser state count is
3; the median schema needs no stack at all because it is purely regular. Lexer
states: median 129, p90 356. The whole corpus compiles and validates in 5.1
seconds.

## What the lexer is for

It collapses the vocabulary, and that is all. Many byte strings map to one
terminal — `"abc"`, `"xyz"` and `"hello"` all emit `string` — so tokens that
emit the same terminals become indistinguishable to the parser and share a
group. Measured with Qwen3's 151,669 tokens, a lexer state has 1 to 13 groups,
so a decode step is a handful of ACTION lookups regardless of vocabulary size.
That is the whole point of the lexer/parser split, and it is why the left half
of `allowed(token) = lexer_ok(...) AND parser_ok(...)` can be compile-time data.

It follows that **a lexer state that merges no tokens is pure cost**. Each one
carries a token bitset per group: 19 KB with this vocabulary. The schemas that
blew up had 10,807 lexer states at 1.3 groups each and cost 252 MB; a
well-behaved one had 76 states at 4.6 groups and cost 6 MB. The extra states
were counting string length and distinguishing which use of a rule we were in —
neither of which merges a single token, and both of which a stack does for free.
That is what XGrammar gets by having no lexer at all: its grammar keeps
`{0, 2048}` as a repetition node and its automaton counts at runtime, so the
grammar is the same size whether the bound is 8 or 2048.

`DEFAULT_TERMINAL_BUDGET` is the knob. A subtree becomes one terminal only if
it fits; anything larger is left to the parser. Measured over the corpus:

| terminal budget | compiled | accepted | acceptance |
| --- | --- | --- | --- |
| 128 | 385 | 212 | 55.1% |
| 512 | 365 | **265** | 72.6% |
| 4096 | 280 | 239 | 85.4% |
| unbounded | 265 | 227 | 85.7% |

Coarse terminals are accurate but do not fit; fine terminals fit but bring back
the lexical ambiguity that coarse ones removed. The absolute number of schemas
that both compile and accept peaks at 512, which is the default.

## What is broken, in priority order

**1. We reject documents we should accept. Acceptance is 85.7%.** Of 265
schemas that compile within the lexer budget, 227 accept the document a model
produced under that same schema. A byte-level differential is the measurement —
compile the schema, feed the instance one byte at a time. Reproduce a single
case with:

```
cargo run --release --bin engrain-trace -- results/jsonschemabench-instances.json <index>
```

It prints, per byte, the lexer state, the terminals the scan emitted, the
terminals a pending lexeme could still become, and whether the parser accepted.
That tool found every bug listed below and is the right place to start.

**2. Coverage went down while correctness went up.** LALR(1) tables: 461 → 434.
Each correctness fix changed the lexicon, and a coarser lexicon means more
reduce/reduce conflicts. The conflicts are real and unexamined. `engrain-explain`
dumps the terminals and productions for one schema.

**3. Length bounds explode whatever holds them.** A DFA can only hold a counter by
unrolling it, so `"maxLength": 2048` over UTF-8 costs about seventy states per
counted position: one schema asked for 209,001 lexer states. 94 of 533 schemas
carry a bound, 38 above 256. `build_lexer_within` abandons construction over a
budget so the rest of the corpus stays measurable, but the real answer is a
counter-augmented lexer, where the runtime state is `(dfa_state, counter)` and
the counter is not compiled away. That is unbuilt.

**4. Compilation was unusably slow, and is not any more.** The vendored FSM
builder inlined a referenced rule into every use without memoisation, so a rule
graph that shares subexpressions was expanded as a tree. Lowered schemas share
heavily, and the expansion is exponential: validation over the corpus never
finished. `resolve_references` already splices rule references with sharing, so
the builder is now asked not to inline at all. The whole corpus went from never
finishing to **5.1 seconds**. Reachability is also one fixpoint rather than a
search per group, and pending-terminal lists are shared by lexer state.

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
cargo run --release --bin engrain-coverage -- /tmp/schemas.json 0

# Byte-level acceptance against model-generated documents
ENGRAIN_MAX_LEXER_STATES=4000 \
  cargo run --release --bin engrain-validate -- results/jsonschemabench-instances.json 5

# Why one document was rejected
cargo run --release --bin engrain-trace -- results/jsonschemabench-instances.json 2

# Python
python -m unittest discover -s tests

# vLLM, from the submodule
PATH="$PWD/.venv/bin:$PATH" FLASHINFER_DISABLE_VERSION_CHECK=1 \
  python src/engrain_lab/vllm_smoke.py --backend engrain --prompts 64
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
