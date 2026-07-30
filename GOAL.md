# GOAL

## Ultimate goal

Build **gpugrammar**: a constrained-decoding engine whose parser state lives on
the device, so that a grammar is something a decode step *contains* rather than
something a decode step *waits for*.

Two hard requirements set it apart from the current `gpu-lr1` prototype:

1. **Real parser power.** The engine must handle **LALR(1)**, and preferably
   **IELR(1)**, grammars — not a bounded subset, not an acyclic schema DFA.
2. **Paper before code.** The design is written up and submitted as an academic
   paper first; the open-source release follows. The paper must present a set of
   clearly identified technical challenges solved in a way that is *elegant*,
   not merely engineered.

### The thesis, and what it is not

**[framing decision, 2026-07-25]** This project is *not* an attempt to build a
faster mask generator. That framing was tried and it does not survive contact
with measurement.

XGrammar's mask fill is 4.6% of a vLLM decode step at batch 512 on a 0.6B model,
and it overlaps the forward pass, so the ceiling on beating it is small and
conditional — it needs a large batch, a small model, and a starved CPU. Every
honest re-measurement of the "10x faster" claim moved it the wrong way: 17.3x on
synthetic schemas with a truncated vocabulary and a single-threaded baseline,
4.5–14.3x once the workload was real and the baseline was given its best thread
count, and **−19% end to end** once our own compiler was actually in the loop.
A claim that erodes under scrutiny is not the claim to build a paper on.

The durable claim is about **where the state lives**, and it is not conditional
on batch size:

> A CUDA graph is a fixed sequence of kernels. Anything the host has to produce
> mid-step cannot be inside it. Today's engines therefore build the mask outside
> the graph and hand it in, which makes the grammar a second-class participant
> in the decode loop.

Three consequences follow, and none of them are latency arguments:

- **Speculative decoding is not yet an argument.** An earlier version of this
  document claimed 33,367 µs against 56 µs at batch 512, k=8. That measured
  XGrammar filling a bitmask per draft position, which is not the API it offers
  for this: `traverse_draft_tree` walks the whole draft tree in one call and is
  flat in `k` - 1,218 µs at batch 512 whether `k` is 1, 4 or 8. The per-position
  path costs 56,323 µs at `k`=8, so the old comparison overstated the gap by
  about 46x. It has been withdrawn.
- **Sampler fusion becomes possible.** If the allowed set is on device, sampling
  reads 400 candidates instead of 151,669. Measured at batch 512: unconstrained
  FlashInfer 3,457 µs, fused constrained sampling 198 µs. *Constraining makes
  sampling cheaper* — but only if the constraint is already there.
- **The decode loop can be captured whole.** A host-dependent component cannot
  join a fully graph-resident or megakernel decode loop at all.

On all three, the comparison is not "we are faster" but "the other design cannot
participate". That is what makes the claim worth a paper.

**What we owe in exchange** is the cost of residency: table memory. XGrammar
keeps kilobytes because it recomputes on the host every step. We keep the
translation from tokens to terminals resident, and the honest number today is
31 MB for a schema that started at 440 MB. Bringing that to the same order of
magnitude as XGrammar's cache is the central engineering problem, and the
measurements say it is reachable: a mask is a pure function of the lexer state,
and a real document visits 1–4% of the states its grammar can reach.

## Sequencing

| Milestone | Target |
|---|---|
| Technical challenges frozen, architecture decided | now |
| Core engine + evaluation complete | before submission |
| **MLSys 2027 submission** | **2026-10-30** (expected deadline, 23:59 PDT) |
| MLSys 2027 conference | 2027-05-17 – 05-22 |
| Public source release of `gpugrammar` | after acceptance/decision |

Everything in this repository is a feasibility prototype feeding that paper.
Work that does not either (a) retire a named challenge below or (b) produce a
figure/table in the paper is out of scope.

## What "device-resident" must mean concretely

A serving engine should be able to swap XGrammar for gpugrammar and get:

- **A decode step that never touches the host.** No mask handed in from
  outside, no synchronisation per token. This is the requirement everything
  else is in service of; a design that violates it is not this project.
- **Equal or greater grammar coverage.** Full LALR(1)/IELR(1), plus a regex
  lexer layer, plus JSON Schema as a front-end — with no silent language
  truncation.
- **Bit-exact masks.** Differentially verified against a reference parser on
  every benchmark configuration, not just spot-checked. Approximation is not
  available here: one wrong bit either leaks an illegal token or blocks a legal
  one, so learned or lossy mask representations are out of scope by definition.
- **Speculative decoding at full draft length**, with rollback and fork on
  device, since this is where a host-side matcher stops being viable rather
  than merely costing something.
- **Sampler fusion**: the allowed set feeds the sampler directly instead of
  being materialised as a vocabulary-wide mask first.
- **Bounded memory** under heterogeneous continuous batching, within an order
  of magnitude of XGrammar's kilobyte-scale compiler cache. This is the price
  of residency and the number we are most obliged to report honestly.
- **Serving-grade compile times.** New schemas arrive per request; compilation
  must be incremental and cached, not a per-grammar batch job.

**Parity, not victory, is the bar for per-step mask cost.** If the constraint
step is resident and its cost is comparable, the argument is already won,
because the alternative cannot be resident at all.

## Killer examples (measured, A100 80GB, Qwen3 151,669-token vocabulary)

Two workloads where a GPU-resident engine is not incrementally better but
categorically better. Both use XGrammar's own builtin JSON grammar, XGrammar
0.2.3 for the baseline, and FlashInfer 0.6.15 `top_k_top_p_sampling_from_logits`
as the sampler — the sampler vLLM/SGLang actually use.

### Example 1: fused constrained sampling is cheaper than *unconstrained* sampling

At a JSON number/structural position the grammar allows 396 of 151,669 tokens.
Sampling over the allowed set instead of the vocabulary (median wall, µs):

| batch | FlashInfer, **no constraint** | XGrammar mask (on GPU) + FlashInfer | XGrammar full path + FlashInfer | **gather + FlashInfer** |
|---:|---:|---:|---:|---:|
| 1 | 275.5 | 331.0 | 1,014.7 | **191.1** |
| 128 | 1,070.9 | 1,130.6 | 5,867.6 | **201.6** |
| 512 | 3,456.7 | 3,775.4 | 8,968.0 | **198.0** |
| 2,048 | 13,007.2 | 14,526.0 | 29,737.3 | **408.6** |

Constraining the grammar makes sampling **17.5× faster at batch 512 and 31.8×
faster at batch 2,048 than sampling with no constraint at all**, and 45×/73×
faster than the deployed XGrammar path. Nobody exploits this today because every
existing system hands back a full-vocabulary mask and lets the engine sample
over the whole vocabulary.

In the opposite regime — a JSON string body, 147,144 allowed tokens, 4,525
exceptions — fusion still does not lose: 3,063.8 µs versus 3,435.1 µs
unconstrained and 3,980.9 µs for XGrammar at batch 512. The constraint is
effectively free at both ends of the density spectrum.

Context: a 7B decode step at batch 512 is ~8.5 ms (weight streaming lower bound,
measured 1,648 GB/s). Today's constrained path adds 8.97 ms — it more than
doubles the step. The fused path adds 0.198 ms, or 2.3%.

**Re-measured on the shipped engine (2026-07-30), and the figures above are
superseded by these.** `Batch.allowed` and `gg_compact` now exist, so this is
the artifact's own number rather than a probe's. XGrammar's own
`apply_token_bitmask_inplace` is the mask baseline, `sampling_from_logits` is
the sampler throughout, and the compaction is charged to us — without it the
set is not on hand at all, and leaving it out was how the earlier figure got
to be larger. 425 of 151,669 tokens allowed:

| batch | no constraint | XGrammar mask + sample | **allowed set + sample** | vs unconstrained | vs mask |
|---:|---:|---:|---:|---:|---:|
| 1 | 240.0 | 248.8 | **198.2** | 1.21× | 1.26× |
| 128 | 480.4 | 634.2 | **209.3** | 2.29× | 3.03× |
| 512 | 1,194.9 | 1,842.5 | **207.8** | **5.75×** | **8.87×** |

Flat at 190–210 µs at every batch, because the work is the size of the set and
not of the vocabulary. The claim survives: **a constrained step is cheaper than
an unconstrained one**, by 5.75× at batch 512 rather than 17.5×. The earlier
number came from a heavier sampling baseline; the smaller one is the one to
publish, and it is still the argument, because nothing that hands a mask back
to the host can make it at all.

**The dense regime was then measured, and the claim above that fusion "does
not lose" there is false.** A JSON string body admits 147,346 of 151,669
tokens, and gathering them costs more than never leaving the mask:

| batch | no constraint | XGrammar mask + sample | allowed set + sample |
|---:|---:|---:|---:|
| 1 | 240.5 | 249.0 | 400.4 (**0.62×**) |
| 128 | 480.6 | 611.2 | 1,654.4 (**0.37×**) |
| 512 | 1,195.9 | 1,719.4 | 5,670.0 (**0.30×**) |

Swept over set size at a fixed batch, the set path is flat at ~210 µs while
the set is small and then grows linearly. It stops winning at **43% of the
vocabulary at batch 128 and 11% at batch 512**:

| allowed | % vocab | b128 mask / set | b512 mask / set |
|---:|---:|---:|---:|
| 64 | 0.0% | 627.7 / 209.5 → 3.00× | 1,809.9 / 210.2 → 8.61× |
| 1,024 | 0.7% | 643.2 / 210.2 → 3.06× | 1,882.1 / 214.3 → 8.78× |
| 4,096 | 2.7% | 660.3 / 184.4 → 3.58× | 1,975.1 / 213.0 → 9.27× |
| 16,384 | 10.8% | 671.8 / 192.8 → 3.48× | 2,017.1 / 544.3 → 3.71× |
| 65,536 | 43.2% | 669.3 / 584.0 → 1.15× | 1,959.4 / 2,034.4 → **0.96×** |
| 147,346 | 97.1% | 615.9 / 1,680.5 → **0.37×** | 1,735.4 / 5,752.5 → **0.30×** |

### How often is a real step sparse?

Measured over 11,892 decoding steps from 120 JSON Schema Bench documents,
walking each document token by token under its own schema:

    allowed tokens per step:  p50 147,234   p90 147,354   p99 151,669
    steps admitting <  4k tokens:  49.3%
    steps admitting < 16k tokens:  49.3%
    steps admitting < 64k tokens:  49.9%

**The distribution is bimodal, not sparse.** Half of all steps are structural
and admit a few hundred tokens; the other half are inside string bodies and
admit essentially the whole vocabulary. There is almost nothing in between.

So "always use the set" is a **net loss** — 0.61–0.67× against always using the
mask. What wins is choosing, which `counts` already makes possible:

| batch | always mask | always set | **choose per step** |
|---:|---:|---:|---:|
| 128 | 637.8 | 947.1 (0.67×) | **407.3 (1.57×)** |
| 512 | 1,853.6 | 3,021.5 (0.61×) | **984.9 (1.88×)** |

### Re-measured in place, which is what a serving loop does

The figures above charge both paths for a 310 MB restore of the logits, which
a decode loop never pays — its logits are fresh every step. In place:

| position | batch | short list | no constraint | XGrammar mask | shortlist path |
|---|---:|---:|---:|---:|---:|
| structural | 128 | 7 allowed | 481.7 | 543.6 | **206.7 (2.63×)** |
| structural | 512 | 7 allowed | 1,195.9 | 1,430.9 | **185.1 (7.73×)** |
| string body | 128 | 4,386 forbidden | 481.4 | 524.2 | 555.1 (0.94×) |
| string body | 512 | 4,386 forbidden | 1,196.2 | 1,358.4 | 1,424.8 (0.95×) |

**In the dense half there is nothing to win.** XGrammar's mask kernel costs
161 µs over unconstrained sampling at batch 512 — 13.5% — so the constraint is
already nearly free there, and scattering `-inf` at the 4,386 forbidden ids
with a generic gather/scatter is *slower* than its bitmask kernel. The
complement is the right *representation* and the wrong *operation*; a fused
kernel might close the 161 µs, and nothing larger is available.

All of the win is in the sparse half, and it is large: **7.73× at batch 512.**

### The number to publish

Weighting by the measured step distribution — 49.3% sparse, 50.7% dense:

| batch | always mask | **choose per step** | vs mask | vs *unconstrained* |
|---:|---:|---:|---:|---:|
| 128 | 533.8 | **367.7** | 1.45× | 1.31× |
| 512 | 1,394.1 | **780.0** | 1.79× | **1.53×** |

**1.79× against the mask path and 1.53× against sampling with no constraint at
all** — on the real distribution, in place, with the compaction charged. Not
17.5×, not 8.87×, not 1.88×. Every earlier figure was a position rather than a
workload, or was charged to the wrong side.

The claim that survives all of it is the one that matters: **constraining the
grammar makes a decode step cheaper than not constraining it**, and no system
that hands a mask back to the host can make that claim at all.

### Example 2: speculative decoding, withdrawn, re-measured, and now possible

An earlier version of this document claimed 33,367 us against 56 us at batch
512, k=8. That measured XGrammar filling a bitmask per draft position, which is
not the API it offers: `traverse_draft_tree` walks the whole tree in one call
and is flat in `k`. The claim was withdrawn.

Until recently we could not do this at all. Verifying a draft means advancing
through it and then keeping only the prefix the model accepted, and the device
kept exactly one step of state - enough for the commit to read while overwriting
it, and nothing beyond. Going back meant reloading from the host matcher, which
is the round trip the design exists not to make.

The advance now writes the state it replaces into a ring and `rollback(k)` puts
an entry back, with the ring slot held on the device so the advance stays inside
a CUDA graph - a graph records the arguments it was launched with, so a slot
passed as a scalar would freeze at whatever it was when the recording was made.
The history is off by default: one kept step is the size of the live state, 67
MB at batch 512.

Measured against `traverse_draft_tree` on a linear draft, charging ourselves a
fill and an advance per position and the rollback at the end:

| batch | k=1 | k=4 | k=8 |
|---:|---:|---:|---:|
| 128 | 1.75x | 0.55x | 0.33x |
| 512 | **4.94x** | 1.06x | 0.74x |

**The whole walk is one launch now (2026-07-28).** It was `k` fills and `k`
advances - `2k` graph replays - which is linear in `k` in exactly the cost this
design exists to remove. `capture_draft` records the lot: the state is saved,
every position is advanced and filled into its own row, and the state is put
back, all device-side, so the parse ends where it began and nothing reaches the
host. Verified against the reference matcher, 276 draft masks over eight
schemas, plus three tests.

| batch | k=1 | k=4 | k=8 |
|---:|---:|---:|---:|
| 128 | 1.29x | 0.83x | 0.45x |
| 512 | 2.95x | **2.68x** | **1.59x** |

Both engines produce a mask per draft position - `traverse_draft_tree` fills one
per node - so this is like for like.

At batch 512 we win at every draft length, where k=8 was 0.74x before the
per-step work came down. At 128 we still lose past k=1, and the reason is not
speculative-specific: our cost is `k` times a step, and a step at batch 128 is
where our fixed cost still shows. XGrammar is flat past k=4 because a child
node continues from its parent's state and its mask comes from their cache;
being flat in `k` for us would mean advancing position by position and then
filling every position at once, as one wide batch. Our fill is nearly flat in
the batch - 103 us at 32 against 116 at 512 - so that is the shape that would
do it. Not attempted.

So speculative decoding is no longer a loss, but it is not a killer example
either: it is a capability that was missing, now with a cost curve that favours
us at serving batch sizes and them at small ones - which is the same boundary
everything else in this document falls on.

## Where the automaton ends

**[settled by measurement, 2026-07-28]** Constrained decoding answers a prefix
question - given the bytes so far, which tokens may come next - and that is a
narrower question than "does this document satisfy the schema". Trying to
answer the second one with a grammar was the mistake, and the boundary between
them is the design rather than an implementation detail.

**The contract.** The automaton must answer *widely*:

    widening   -> an invalid document becomes generatable. A checker fixes it.
    narrowing  -> a valid document becomes ungeneratable. Nothing fixes it.

A superset is something a downstream check can filter. A subset is a document
the model can no longer produce, and no amount of validation afterwards brings
it back. So every approximation in the pipeline belongs on the widening side,
and every one that is not is a bug rather than a trade-off. This inverts what
this document used to say - that the engine only ever narrows, which was
offered as a safety property and is in fact the dangerous direction.

**What belongs inside.** Properties decidable from a prefix by a finite state
and a stack: structure and nesting, which property names may appear and which
must, `additionalProperties: false`, value types, `enum` and `const`, `pattern`,
bounded lengths and counts, and property sets in any order.

**What belongs outside.** Properties that need the whole document, or memory
the stack does not have:

| keyword | why the automaton cannot |
|---|---|
| `oneOf` exclusivity | "exactly one branch" is a count over the whole document |
| `uniqueItems` | every pair of elements has to be compared |
| `multipleOf` on reals | arithmetic a finite state cannot carry |
| `not` over a context-free set | the complement of a context-free language need not be context-free |
| `allOf` of two context-free sets | intersection likewise |

**The audit.** Each case is a schema, a document it accepts, and a document it
rejects; refusing the first is a violation and admitting the second is an
approximation (`files/probe_boundary.py`):

| | at the audit | now |
|---|---|---|
| required properties, closed objects, value types, lengths, numeric bounds, nesting | exact | exact |
| `oneOf` exclusivity | **widens** - inherent, belongs outside | widens, **declared** |
| `uniqueItems` | **widens** - inherent, belongs outside | widens, **declared** |
| declared property types, when the object is open | **widens - and this one is ours** | widens, **declared**; exact on request |
| the `Ordered` precision level | **narrows** | **deleted** |
| the configuration ceiling | **narrows** | narrows, and still open |

Three findings, two of them bugs by the contract above. Both have been acted
on and the audit now reports **0 narrowing violations** against 3.

**A declared property's type is not enforced when `additionalProperties` is
permitted**, which is the default. `{"a": 1}` against `{"properties": {"a":
{"type":"string"}}}` is invalid and we admit it, because the object body has an
arm taking any name with any value and a declared name goes through it.

This is not an inherent limit. "Any name except these" is a regular set, and
`string_body_excluding` builds the complement exactly: a string is not a
declared name when it stops at a trie node no name ends at, or leaves the trie
by a character no edge carries, after which the rest is free. Those two are
built separately so the free tail is emitted once - building it per node is
equally correct and more than doubles compile time, because the lexer then
determinises a copy of "any string" for every character of every declared name.

**And it is off by default, which is the more interesting result.** Excluding
declared names costs the schema its one shared string lexeme: every object then
carries a key terminal with its own trie, and the lexer determinises the union
of all of them.

| | default | `exact=True` |
|---|---:|---:|
| compile p50 | 27 ms | 159 ms |
| compile p90 | 1.4 s | 3.2 s |
| captured fill, batch 512 | 72 us | 155 us |
| captured advance, batch 512 | 49 us | 96 us |
| corpus instances accepted | 483 | **488** |
| schemas needing conflict forking | 64 | **8** |

The two things this project loses on are compile time and memory. Paying 6x
compile and 2x per step to enforce one keyword interaction that a downstream
type check settles for nothing is the wrong default, so the default declares it
instead. Note the last row: most of the corpus's parser conflicts are this same
ambiguity seen from the other side - a declared name has two readings, the
literal and the generic key - which is why closing it removes seven eighths of
them. That is the strongest evidence yet that the conflicts are a property of
the lowering rather than of JSON Schema.

**What makes a widened mask safe is being told how it widened.** A compiled
grammar now carries `approximations`: exactly what it does not enforce, gated on
both the level it settled on and the keywords the schema actually uses, so a
closed object reports nothing and a schema that never mentions `uniqueItems` is
not warned about it. A list that cries wolf is one callers learn to ignore, and
it is the only thing standing between a widened mask and a wrong document.

    >>> grammar = engine.compile_json_schema(schema)
    >>> grammar.approximations
    ['a property whose name the schema declares may also be read as an
      additional one, so its declared type is not enforced while
      additionalProperties is open']

This is the boundary made into an API rather than a paragraph: the automaton
says where it stops, and the caller knows what is left to do.

**`Ordered` narrowed, and its own source said so.** The `Precision` enum's
header stated that no level describes less than the schema allows; the
`Ordered` variant three lines below stated that it rejects permutations of
valid documents. Both cannot hold.

The cause was that `Precision` was one enum over two axes. Whether an object
takes its properties in any order is a *narrowing* knob, bought for a smaller
lexer; how far branches merge is a widening one. They were walked in one order,
and `unordered()` was true only at the first level, so `Merged` and `Branches`
silently used the narrowing lowering too.

**Fixed by making the chain one axis, every step of it widening.** `Ordered` is
gone. What never fits is always *counting* - `required` needs a subset of the
required set in the parser state, `minProperties`/`maxProperties` need a tally -
while the shape of an object costs one choice regardless, so the fallback keeps
the shape and drops the counting. That is a strict superset. It is applied to
the object that did not fit and to nothing else; refusing the schema so the
search retries it at a coarser level was measured and is worse, 494 schemas
against 507, because it relaxes every other object too and those did fit.

That deleted `build_property_state` and `intersperse_properties`, the
subset-state machinery that existed only to serve the declared order, and with
it schema 247 - the only wrong refusal in the corpus that was a lowering
decision rather than the configuration ceiling.

**The ceilings narrow.** Dropping a configuration at the ceiling is the
engine's oldest safety story - "narrowing is the safe direction" - and by this
contract it is exactly backwards. Seven of the eight documents we wrongly
refuse are the configuration ceiling.

**And most of that is the same bug again.** A configuration set grows when a
prefix has several readings, and the commonest reading to fork on is a declared
property name that also scans as the generic key. Excluding declared names
takes the sample's counts from 128, 64 and 32 down to 5, 3 and 3, and takes
four of the seven wrong refusals with them:

| | default | `exact=True` |
|---|---:|---:|
| instances rejected | 24 | **19** |
| rejected at the final byte | 1 | **0** |
| of those, valid documents wrongly refused | 7 | **3** |
| widest configuration set, sample of 60 | 128 | 5 typical, 139 worst |

Nothing that was accepted becomes refused. So one lowering decision - letting a
declared name be read two ways - was at once a widening hole, seven eighths of
the parser conflicts, and most of the narrowing at the ceiling. The three
problems the audit found separately have substantially one cause.

What remains at the ceiling is a genuine resource limit rather than an
artefact: one schema in the sample still reaches 139 configurations under
either lowering. Since the typical grammar now needs fewer than eight, a
per-grammar ceiling would be worth much more than when it was last costed
against speed alone - it was cancelled at 6% there, and here it is the
difference between a mask that narrows and one that does not.

So the boundary is not a grammar-class question. Restricting to a deterministic
class would not move it: on grammars that already have no parser conflict, a
sequence still carries a median of 2 configurations and up to the ceiling of
128, and those configurations differ in their *parser stacks* rather than only
in their lexer state. Measured with a byte vocabulary, where there is no
tokenizer ambiguity at all, a two-property object still reaches eight. The set
comes from where lexemes end in a generated lexicon, which is C1, and it is
independent of the parser's grammar class (`files/probe_determinism.py`).

## Grammar class decision

**[settled by measurement, 2026-07-27]** IELR(1) was the target: it accepts the
same grammars as canonical LR(1) at LALR(1)'s table size, and LALR(1) was the
fallback. On this corpus it would buy nothing, and here is why.

Every conflict the corpus produces is reduce/reduce; not one is shift/reduce.
That is the signature of LALR's one weakness - it merges LR(1) states sharing an
LR(0) core, and merging their lookahead sets can invent a reduce/reduce conflict
the grammar does not have - so it looked as though most of them were artefacts
and IELR(1) would take them. Building canonical LR(1) tables, which never merge,
says otherwise:

| of the 32 schemas with no LALR(1) parser at any precision level | |
|---|---|
| LALR artefacts, which IELR(1) would remove | **0** |
| conflicting under canonical LR(1) too | **32** |
| undecided within a 400,000-state budget | 0 |

The control matters as much as the result: of the schemas LALR does accept,
canonical LR(1) accepts 26 of 26 and refuses none, so the builder is answering
rather than failing. A canonical builder that always failed would have produced
the same headline.

So these grammars are genuinely ambiguous - two derivations for one string -
and no LR(1) construction of any kind will parse them. They come from `oneOf`
branches that overlap, which is exactly what `oneOf` describes and what a union
cannot distinguish.

**Ambiguity is not a problem for a mask.** A parser that has to build a tree
must choose a derivation; we never build one. The question is only whether
*some* derivation admits the next token, and the union of what all of them admit
is the answer. Carrying a set of parses is therefore not a compromise here but
the right shape - and the engine already does it, because scanning a generated
lexicon is ambiguous too and a sequence already carries a set of configurations.
Letting an ACTION conflict fork one is the same mechanism on a second axis, over
the same LALR tables, with the same ceiling and the same overflow flag.

That is the decision: **LALR(1) tables, with conflicts forked at runtime rather
than refused at construction.** IELR(1) is not needed for what it was wanted
for, and the measurement above is why.

**Built and measured (2026-07-28).** A conflicted cell contributes a digit to a
mixed radix and a replay runs once per combination, on the host and on the
device alike. A grammar with no conflicts has one path, so the loop is one
iteration around unchanged code and nothing that already compiled pays for it.

| | |
|---|---|
| compiled, was 470 | **510 of 533** |
| accepted | **482** |
| refusals still labelled `Conflict` | **0** |
| conflicted schemas verified against the reference matcher | **64, the whole corpus** |
| steps agreeing on both the mask and the configuration set | 3,641 |
| widest cell handled | 32 actions |

The cost is the fill and the advance at batch 512 going from 102 and 71 us on a
conflict-free grammar to 164 and 168 on the worst conflicted one - under 2x for
the fill, under 2.4x for the advance, and only on the grammars that need it.

The path count is bounded (16, the same bound the reference uses) and losing a
derivation can only narrow a mask, never widen one. That is why the bound is
safe to have and why exceeding it raises the overflow flag.

Also required at this layer: precedence/associativity declarations for conflict
resolution, and an EBNF front-end, since practical grammars are not written in
bare BNF.

## Technical challenges (the substance of the paper)

Each challenge below is stated as a problem, not a solution. Solutions are what
the paper contributes. Challenges marked **[measured]** are already backed by
numbers from this repository.

### C1. Tokenizer–grammar impedance mismatch

Model tokens are BPE byte strings that straddle lexeme boundaries; an LR parser
consumes terminals. A token may end mid-lexeme, so the parser configuration must
carry **lexer state**, not just a parser stack. The prototype avoids this by
restricting itself to byte-terminal grammars, which is why it cannot express a
real language today.

*Why hard:* the configuration space is (lexer DFA state × parser stack), and
maximal-munch lexing means a token's terminal segmentation is not a function of
the token alone.

### C2. Stack-dependent transitions vs. GPU-friendly flat tables

A reduction pops states, and the following `goto` depends on the newly exposed
stack state. Therefore `next[state, token]` is **not** stack-independent, and no
flat per-state table is exact in general.

**[measured]** Enumerating reachable stacks up to a depth bound works but
explodes: at vocabulary 4,096 / depth 6 the compile times out (>10 s), and the
bound silently drops edges that exceed it, making the accepted language a strict
*subset* of the grammar — a correctness cliff, not just a performance one.

*Why hard:* the needed object is a sound **and** complete finite abstraction of
the stack language, compact enough to index on GPU. Prior art (Pre3, PSC) does
parser-stack classification; the open question is an abstraction that is exact
for mask computation, not merely conservative.

### C3. Wide-row mask representation — the dominant scaling wall

**[measured]** The fused CSR kernels are one-program-per-row with
`BLOCK_SIZE = next_pow2(row_nnz)`, hard-capped at 32,768 entries. A realistic
JSON *string* state over Qwen3 allows **146,924 of 151,669 tokens**, needing
`BLOCK_SIZE` 262,144 — the kernel raises rather than runs. The density sweep
already shows CSR losing to dense/bitset beyond ~8,192 allowed tokens.

**Status.** The 32,768 cap is removed and the complement path is implemented.
Streaming the allowed list was correct but unusably slow (6.2 ms at batch 1,
46.7 ms at batch 2,048, against 0.27 / 13.0 ms for unconstrained FlashInfer)
because every bisection probe re-gathered the row through its index list. The
wide kernel now reads logits **contiguously** and tests membership against an
18.5 KiB per-state bitset, and each sweep evaluates 8 candidate thresholds at
once so the search costs 8 passes instead of 32. Wide rows dropped to 1.43 ms
at batch 1 and 20.5 ms at batch 2,048 — a 4.3x improvement, and now 1.6x of
unconstrained sampling rather than 3.6x.

The remaining wide-row gap is **parallelism, not algorithm**: one program per
sequence means batch 1 occupies a single SM while sweeping 151,669 logits
eleven times. Splitting a wide row across several CTAs with a two-level
reduction is the next step and should mostly close the small-batch gap.

*Why hard, and why it is the most promising contribution:* real grammar states
are bimodal — either a handful of allowed tokens (structural positions: 396–759
tokens) or almost the entire vocabulary (string bodies: 147,144). A single
representation cannot serve both. The narrow half is now solved by gathering the
row; the wide half needs the dual: a **per-state complement** (4,525 exceptions,
18.5 KiB as a bitset) consumed by one contiguous O(V) pass, with the threshold
found from a single-pass histogram instead of repeated global probes. Cost then
becomes the unconstrained sampler's cost plus about 6% of extra traffic.

### C4. Table memory under heterogeneous continuous batching

**[measured]** Naive CSR costs ~1.1 MiB for a single JSON-string state on Qwen3;
1,000 such states is 1.09 GiB. The schema DFA backend already reaches 25 MiB for
14 schemas at a 32k vocabulary. XGrammar's compiler cache is ~52 KiB.

*Why hard:* thousands of concurrent requests with distinct grammars share GPU
memory with the model and KV cache. Needs structural sharing (interned mask
sets, a DAG over states, suffix sharing across schemas), plus paging/eviction
policy — while keeping lookups branch-free on device.

**Paging, built and measured (2026-07-28).** The hard part is not reclaiming
memory; it is reclaiming it *without moving anything*. Compaction renumbers
every survivor, so every holder of a grammar id has to be told and every
recorded CUDA graph is re-recorded — under continuous batching that would be
every few requests, which would give back exactly what residency bought.

So the arena is not a bump pointer with wholesale compaction. Each array
carries a free list with coalescing, and identifiers are recycled rather than
renumbered: a released run goes back, the next admission takes it, and no
address and no number moves. A grammar a sequence is running under is pinned
and never chosen; the rest are evicted least-recently-used, where "recently"
is stamped at the last unpin rather than per step, so the decode loop does no
bookkeeping at all.

40 schemas, 92.9 MB of tables in all, driven through 400 admissions:

| budget | arena | evictions | graph re-records, warm | holes | per round |
|---:|---:|---:|---:|---:|---:|
| 32 MB | 33.9 MB | 399 | **0** | 0.0% | 679 us |
| 64 MB | 63.7 MB | 384 | **0** | 18.2% | 698 us |
| 96 MB | 99.4 MB | 374 | 1 | 11.1% | 699 us |
| 128 MB | 110.3 MB | 0 | 0 | 0.0% | 76 us |
| unbounded | 110.3 MB | 0 | 0 | 0.0% | 70 us |

**399 evictions cost zero graph re-records.** The re-records that do happen are
one-off, from an array doubling during warm-up. The mask of the pinned grammar
agrees with its matcher after all of it.

The price of paging is the re-admission: 700 us against 76 when everything
fits, which is the device copy of ~2.3 MB of tables. It is paid when a request
arrives, not when a token is sampled. Making it that cheap needed separating
what a grammar *is* from where it lands - re-admission had been 8.2 ms, of
which the copy was 0.7 and the rest was materialising arrays from Rust and
recomputing ceilings that had not changed.

What a release cannot lower is a ceiling - the window, the readings a group can
have, the paths a replay follows - because those size buffers a running step is
reading. A pool that has seen a large grammar keeps its shape after it leaves.
That is recorded rather than fixed.

### C5. The latency floor is dispatch, not compute

**[measured]** For the direct LR(1) step at batch 1: reported 36.4 µs, CPU
dispatch 38.7 µs, CUDA-graph replay 3.41 µs, **actual kernel 3.16 µs**. The
device work is already ~10× cheaper than the framework overhead around it.

*Consequence:* the contribution cannot be "a fast kernel". It must be a
constraint step that is **resident in the serving engine's CUDA graph**, with
in-place state updates, no host synchronization, and stable launch shapes across
changing batch composition. Benchmarks must report device time and dispatch time
separately; the current `measure()` helper conflates them and must be fixed
before any number goes in the paper.

### C6. Warp divergence from unbounded reduce chains

**[measured]** A right-recursive grammar needs a reduce chain proportional to
nesting depth (depth 100 → 100 reductions in one step); with a bound of 16 the
step returns `REDUCTION_LIMIT`. On SIMT hardware the deepest lane dictates the
step cost for its whole batch.

*Why hard:* per-step work must be bounded and uniform. Candidates: precomputed
reduce-closure compression so one token = one table lookup, or work
redistribution across lanes. Any bound must not truncate the language (see C2).

### C7. Online grammar admission and compile budget

**[measured]** XGrammar compiles in ~3.2 ms; the prototype takes ~0.86 s plus
~1.1 s of launch autotuning per batch shape. In serving, grammars arrive with
requests.

*Why hard:* needs lazy/incremental table construction (build states on first
visit), a persistent cross-request cache, and possibly GPU-side construction —
while preserving the exactness guarantees of C2.

**[measured] Rows repeat, so this is affordable.** Over the 55,406 real
decoding steps of the Llama-3 replay there are only **5,837 distinct allowed
sets**: a lazily built cache hits **89.5%** of the time and builds 105 rows per
1,000 steps. Even with no cross-request sharing at all, reuse *within* a single
request is 67.2%, so at most 32.8% of steps can miss.

If a row costs one XGrammar mask fill (measured p50 6 µs, p99 1,071 µs), the
amortised construction cost is **0.6 µs per step** warm and 2.0 µs cold,
against 3.3 µs per sequence for the sampling step itself at batch 128. The
asymmetry is the point: XGrammar fills a mask on *every* step, gpugrammar only
on a cache miss, so including construction still leaves it ahead. The 1,071 µs
p99 fill also confirms the tail behaviour llguidance criticises.

### C8. Correctness at scale, as a first-class artifact

The prototype's parser core is exhaustively verified against an independent
Earley recognizer, and its GPU kernels match the CPU reference step-for-step.
That standard must extend to the full system: bit-exact differential masks
against a reference parser across grammars, tokenizers, and batch shapes, plus
an argued **soundness and completeness** result for the stack abstraction —
over-approximation admits invalid strings, under-approximation silently removes
valid ones.

**[measured, and it fails]** The runtime only ever narrows, which was the whole
of the claim; the *lowering* over-approximates, because `oneOf` becomes a union
and a union is "at least one" where `oneOf` is "exactly one". We admit 7 of the
22 corpus documents that violate their own schema. XGrammar admits all 22. So
the result this challenge asks for has to be stated over the pipeline rather
than the parser, and it is currently a negative one - see the soundness section
below.

### C9. Sampler and speculative-decoding integration

Masks must compose with temperature/top-k/top-p sampling, and support rollback
and fork for speculative decoding and beam search — all batched on device. The
prototype does argmax only, which is not a usable serving interface.

### C10. Ragged fused sampling across a heterogeneous batch

**[measured]** The sampler-fusion win (Example 1) depends on processing only the
allowed set. But sequences in one batch sit in states of wildly different width
— 396 tokens at a number position, 147,144 inside a string — and 51.4% of the
tokens in a realistic JSON document are emitted from wide string states. A dense
gather buffer is sized by the widest row in the batch, so a single wide sequence
erases the advantage for all 512.

*Why hard:* the sampler must consume **ragged** rows — per-sequence widths, not a
padded rectangle — while keeping top-k/top-p semantics exact and staying inside
one CUDA graph. Candidates: width bucketing with per-bucket launches, a
per-row persistent kernel, or a two-tier design that routes narrow rows to a
gathered sampler and wide rows to a complement-masked full-width sampler. This
is the concrete engineering-and-algorithms problem that makes Example 1 real
instead of anecdotal.

**Status.** Implemented in `src/gpu_lr1/ragged_sampler.py` and
`src/gpu_lr1/wide_sampler.py`, verified by `tests/test_ragged_sampler.py`
against a sorted reference. Three findings drove the optimisation, each
measured rather than assumed:

1. **Hidden syncs, not kernels.** Two `.item()` calls in the first dispatcher
   cost 270 µs per step. Row widths are now cached at table construction and
   bucketing uses an in-kernel early exit, so both bucket kernels launch over
   one grid and the step never touches the host.
2. **Occupancy, not bandwidth.** fp16 logits did not speed the wide kernel up
   at all, and block/warp tuning plateaued, which ruled out bandwidth and
   compute. One program per sequence left batch 128 with 128 programs for 108
   SMs. Splitting each wide row into chunks — 64 at batch 1, down to 2 at batch
   2,048 — gave 3.5x at small batch.
3. **Launch count, not work.** The split path issues about twenty launches, so
   the step became dispatch-bound again. Capturing it as a CUDA graph collapsed
   that to one replay: narrow rows went from 493 µs to 28 µs at batch 1.

Final measurement on A100 with Qwen3 and XGrammar's builtin JSON grammar,
graph-replayed, median wall (`results/a100-ragged-sampler.json`):

| profile | batch | gpugrammar | FlashInfer unconstrained | XGrammar full path | vs unconstrained |
|---|---:|---:|---:|---:|---:|
| narrow | 1 | 28.3 µs | 289.3 µs | 1,041.6 µs | 10.2× |
| narrow | 512 | 35.3 µs | 3,471.0 µs | 8,985.1 µs | 98.4× |
| narrow | 2,048 | 114.7 µs | 12,991.3 µs | 32,009.2 µs | **113.3×** |
| mixed | 1 | 88.8 µs | 282.1 µs | 1,031.9 µs | 3.2× |
| mixed | 128 | 732.4 µs | 1,086.9 µs | 5,988.7 µs | 1.5× |
| mixed | 2,048 | 8,927.5 µs | 13,027.8 µs | 28,954.4 µs | 1.5× |
| wide | 1 | 89.3 µs | 282.4 µs | 1,009.2 µs | 3.2× |
| wide | 2,048 | 16,850.9 µs | 13,009.1 µs | 26,001.3 µs | 0.8× |

A realistic mixed batch is now faster than sampling with no constraint at all
at every batch size, by 1.4–3.2×, and 3.2× faster than the deployed XGrammar
path at batch 2,048. C10 is retired.

### Measured on the real workload

Earlier profiles were synthetic in both directions and the comparison charged
each engine differently. The measurement below fixes both.

**Workload.** `gpu_lr1.generate_instances` has Llama-3-8B-Instruct produce 533
JSON values under a real XGrammar constraint — 50 schemas from each of the 11
JSONSchemaBench configs, sampled at temperature 0.8 / top-p 0.95, up to 256 new
tokens, mean 125 tokens. `gpu_lr1.replay_tokenizer` then replays that text
through each tokenizer's grammar, which is faithful because the state a matcher
reaches is determined by the bytes consumed. Content and tokenization are
separated so vocabulary effects are isolated from schema effects.

**Cost model.** Both engines are charged for everything they must do per step.
XGrammar pays mask fill, pinned H2D, mask apply, FlashInfer sampling **and**
`batch_accept_token` plus rollback — the advance alone is 4.6–22.2% of its CPU
work and was previously omitted. Its thread count is swept over 1/2/4/8/16/auto
and the fastest is reported, and its device work is CUDA-graphed, as ours is.

**Width is set by the schema, not the vocabulary.** Across three tokenizer
families the median step allows a few hundred tokens no matter how large the
vocabulary is, so the O(V)-versus-O(allowed) ratio grows with vocabulary size:

| tokenizer | vocab | steps | median allowed | wide (>8,192) | forced |
|---|---:|---:|---:|---:|---:|
| Llama 3 | 128,256 | 55,406 | 396 | 32.9% | 0.8% |
| Qwen 3.6 | 248,077 | 66,557 | 378 | 32.0% | 0.7% |
| Gemma 4 (SentencePiece) | 262,144 | 69,313 | **107** | 32.3% | 0.0% |

Per split the spread is wide: Glaive function calls are only 11–12% wide, while
WashingtonPost is 46–54%.

**Result** (A100, graph-replayed, median wall):

| tokenizer | batch | gpugrammar | FlashInfer unconstrained | XGrammar full path | gap |
|---|---:|---:|---:|---:|---:|
| Llama 3 | 32 | 222.4 µs | 299.7 µs | 1,954.8 µs | 8.8× |
| Llama 3 | 128 | 416.0 µs | 628.3 µs | 5,963.5 µs | **14.3×** |
| Llama 3 | 512 | 1,464.2 µs | 1,984.9 µs | 13,981.4 µs | 9.5× |
| Qwen 3.6 | 32 | 291.2 µs | 917.2 µs | 3,011.7 µs | 10.3× |
| Qwen 3.6 | 128 | 771.8 µs | 1,755.3 µs | 6,039.5 µs | 7.8× |
| Qwen 3.6 | 512 | 2,234.1 µs | 5,697.8 µs | 13,896.6 µs | 6.2× |
| Gemma 4 | 32 | 310.8 µs | 607.4 µs | 2,969.7 µs | 9.6× |
| Gemma 4 | 128 | 874.1 µs | 1,200.8 µs | 7,826.4 µs | 9.0× |
| Gemma 4 | 512 | 2,439.1 µs | 4,025.2 µs | 10,937.2 µs | 4.5× |

The honest range is **4.5–14.3×**, not the 12–35× an unfair thread setting and a
hand-written schema produced.

**Table memory is now reportable.** Wide rows keep only a bitset plus a default
successor and a small override list; their token lists are dropped outright,
which `tests/test_ragged_sampler.py` checks does not change a single sampled
token:

| tokenizer | resident tables | as plain CSR | reduction |
|---|---:|---:|---:|
| Llama 3 | 24.5 MiB | 935.0 MiB | 38.2× |
| Qwen 3.6 | 36.6 MiB | 1,807.7 MiB | 49.4× |
| Gemma 4 | 34.7 MiB | 1,952.6 MiB | 56.3× |

Tens of MiB is defensible next to a KiB-scale compiler cache only because it
buys a GPU-resident step; the paper must report it, not hide it.

### Measured: the grammar half of a decode step

**[measured, A100, Qwen3 151,669-token vocabulary]** A step's grammar cost is
not only the mask fill. Every accepted token also has to advance the parser,
once per sequence per step, and that half lands *after* the sampled token, on
the critical path, with nothing to overlap. Charging both halves, replaying the
same document through both backends so they visit the same states, and giving
XGrammar its best thread count:

| batch | XGrammar fill | XGrammar advance | total | ours fill | ours advance | total | ratio |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 32 | 281 µs | 28 µs | 309 µs | 458 µs | 12 µs | 470 µs | **0.66x** |
| 128 | 653 | 173 | 826 | 663 | 40 | 703 | 1.18x |
| 256 | 1,075 | 343 | 1,418 | 971 | 75 | 1,046 | 1.36x |
| 512 | 1,925 | 632 | 2,557 | 1,576 | 143 | 1,720 | **1.49x** |

Two things to read off it. The crossover is near batch 100: below that a kernel
launch costs more than the CPU work it replaces, and we are *slower*. Above it
the gap widens, because the CPU cost scales with the batch and the GPU cost
barely does — which is the premise the project was founded on, now measured
rather than assumed.

And the advance is 4–5x cheaper throughout. That is the half that cannot hide
behind the forward pass, and it is the half that speculative decoding multiplies
by the draft length.

**What this does not support.** An end-to-end throughput claim. Grammar work is
a fraction of a decode step, so a 1.49x on that fraction is a few percent
end to end, and a vLLM A/B at batch 256 ranged from 4,523 to 13,325 tokens per
second across runs — too noisy to attribute a difference to anything. The
per-step number is the defensible one.

### vLLM integration

`gpu_lr1/vllm_backend.py` implements vLLM's `StructuredOutputBackend` on top of
the Rust compiler, reached from Python through PyO3 bindings
(`gpugrammar-py`). vLLM 0.25 dispatches backends with a hardcoded `if/elif` and
has no registry, unlike SGLang's `register_grammar_backend`, so `install()`
substitutes this backend for the name vLLM already knows. That is a measurement
device; the upstream ask is a registry. The engine also runs in a subprocess by
default, so measurement needs `VLLM_ENABLE_V1_MULTIPROCESSING=0` or a plugin
entry point.

**It works end to end.** Qwen3-0.6B under a JSON Schema produces 16/16 valid
documents, the same as stock XGrammar. Compiling that schema against the full
151,669-token vocabulary takes 83 ms and yields 88 groups; a mask fill takes
19 µs.

**It is faster now, narrowly, and the number is noisy.** Qwen3-0.6B under three
schemas at once, 64 prompts, median of five runs: **7,052 tok/s against
XGrammar's 6,588**. The spread is wide - 3,972 to 7,104 for us and 4,644 to
7,630 for them - so this is a 1.07x that should be read as parity rather than a
win. It was 571 against 692 when this section was first written, and what closed
it was not the kernels but the interface between them and the engine.

**A batch under many schemas is one launch.** Every schema the engine has seen
lives in one arena and a sequence carries the index of the one it is under, so a
step holding a dozen different schemas is a single launch rather than a dozen.
That is what a serving batch is: requests bring their own schemas and the
mixture changes every step. A CUDA graph recorded on one assignment of grammars
to sequences replays correctly on a different one, which is what makes the
capture survive continuous batching.

**Two bugs only the integration found**, and neither is visible with one schema.
vLLM compiles grammars on a thread pool, so two requests with different schemas
reach the backend at once; admission was a check-then-act, and two schemas took
the same index, which masks one against the other's tables. And the copy into
vLLM's bitmask assumed the two were the same width - vLLM's spans the model's
padded vocabulary and ours the tokenizer's, 4,748 words against 4,740 - so the
tail was left as whatever the row held.

**Loading the batch's state cost more than every kernel it fed**: 2.3 ms at
batch 512 against 84 us for the fill. It sent `rows x max_configs x max_stack`,
8.45 MB to carry a few dozen words, and turned each matcher's state into Python
objects on the way. Sending what the sequences hold is 65 kB, and packing it in
Rust is 352 us. This is the shape of most of what was slow here: not arithmetic,
but a ceiling paid for as if it were the work.

**Coverage and acceptance on real schemas.** Of 533 JSONSchemaBench schemas,
**510 compile** at the default lexer budget and feeding each schema's
model-generated instance through the matcher one byte at a time accepts **482**.
It was 431 and 416, and what moved it was reading the failures rather than
counting them.

Counting keywords does not say why a schema fails - `type` and `properties` are
in all of them - so rank each keyword by how much more often it appears in a
failing schema than in one that compiles, then read the error the pipeline had
been collapsing into a four-value code. Four causes, and three are now fixed:

| cause | was | fix |
|---|---|---|
| reduce/reduce conflicts, `oneOf` in 86% at lift 6.5 | 109 attempts | collapse object branches into one object |
| a pattern that also carries a length bound | 39 | narrow the repeat the bound constrains |
| `$ref` to anything but `#/definitions/X` | 24 | follow the JSON pointer |
| `allOf` branches that disagree | 20 | compute the conjunction where one exists |

**Coverage is a budget, not a wall.** The lexer budget was reported at 4,000
states as though it were a limit. It is not:

| lexer states | compile | accepted | refused by the lexer |
|---:|---:|---:|---:|
| 4,000 | 495 | 471 | 17 |
| 20,000 (default) | 510 | 482 | **1** |

At the default budget one schema fails on the lexer, and since conflicts are
forked at runtime rather than refused, none fails on the parser class either.
What is left is 12 the front end cannot lower and 10 over the production
budget. The price falls entirely on the tail: the median schema's compile time
and table size are the same at every budget, because it comes nowhere near the
ceiling. Reporting a single coverage figure without the budget it was measured
at is what made this look like a wall.

**The acceptance metric was measuring the corpus.** Acceptance had been "how
many of the 533 instances does the matcher consume", on the assumption that a
benchmark's instances are valid documents. Handing each one to a real JSON
Schema validator says otherwise:

| the corpus's own instances | |
|---|---|
| valid JSON satisfying their schema | 420 |
| violate their schema | 24 |
| **not JSON at all** - truncated mid-string | **88** |

So 78.8% is the *ceiling* any correct engine can score on this corpus, and an
engine that refuses the other 113 is right to. Measured against the instances a
correct engine must accept:

| | |
|---|---|
| valid instances whose schema compiles | 406 |
| accepted | 398 |
| **acceptance** | **98.0%**, against the 94.5% the old metric reported |
| genuine wrong refusals | **8**, not the 28 the old metric counted |

Of the 28 refusals, 7 are of text that is not JSON and 13 of documents that
violate their own schema. Eight are ours, and all eight are explained.

**Seven of the eight are the configuration ceiling, which is a knob.** The
parser carries a set of configurations and drops the rest at a ceiling; dropping
one can only make it stricter. Raising the ceiling recovers them:

| configurations | wrong refusals | batch-512 state | fill |
|---:|---:|---:|---:|
| 16 | 8 | 49.6 MB | 284 us |
| 128 (default) | 8 | 340 MB | 287 us |
| 512 | 6 | 1,328 MB | 282 us |
| 4,096 | **1** | 10,549 MB | 609 us |

So acceptance over the valid instances of compiled schemas is **99.8%** at a
ceiling of 4,096, and the price is linear in memory. Reporting acceptance
without the ceiling it was measured at is the same mistake as reporting
coverage without the lexer budget.

**The last one is the precision fallback, also a knob.** Schema 247's
`Unordered` lowering exceeds the 20,000-production budget, so the search falls
back to `Ordered`, which fixes property order - and this document does not use
that order. It is the only schema of the 533 whose refusal is a lowering
decision rather than a ceiling.

**There is no unexplained refusal.** That is the claim the soundness argument
needs, and it is now true rather than nearly true.

**Objects accept their properties in any order.** A JSON object is a set, but a
grammar describes a sequence, so the standard answer - XGrammar's too - fixes
the order at the one the schema declares and rejects every other permutation of
a valid document. It can be done exactly: what the order stands in for is
"which required properties have appeared", which is a *subset* of the required
set, not an ordering of everything. Carrying that subset in the parser state
costs one rule per subset, and required sets are small - 96% of the objects in
JSONSchemaBench require at most four properties.

Measured on the same corpus re-serialised with every object's keys reversed:

| | in declared order | keys reversed |
|---|---|---|
| before | 348 / 416 | 130 / 361 (36.0%) |
| after | 416 / 431 | 313 / 369 (**84.8%**) |

The cost is memory. All property names being live at once enlarges the lexer,
and the eight-schema resident total goes from 3.87 MB to **10.84 MB** - still
60x below the 658 MB the first working version needed. It also costs a few
documents in declared order (five, at the chosen budget), because a declared
name also scans as a generic one and the matcher has to carry a configuration
per reading; raising its budget from 64 to 128 recovered most of them, and 256
recovered two more, which is where it stops paying.

Order-freedom is free when `additionalProperties` is `false`: with a closed set
of names there is no generic reading to fork on. It is bought when the names
are open, which is why the two budgets differ.

**The compiler searches rather than computes.** Lowering cannot know whether
the grammar it produced is LALR(1) - finding out costs a table construction -
so the pipeline lowers a schema as precisely as it can be expressed, tries to
build tables, and drops to a coarser lowering only when the precise one has no
parser. Every level accepts a superset of the one above, so a token the schema
allows is never masked away. On this corpus 424 schemas compile at the most
faithful level, 5 need declaration order, and 2 need `anyOf` branches lowered
without their siblings.

**Two bugs only end-to-end testing found.** A grammar with no recursion left an
empty skeleton, which was treated as failure when it is the best case: the
document is one lexeme, no stack is needed, and it covers 68% of the corpus.
And a token that ends mid-lexeme emits no terminal, so nothing constrained it —
a finished document could be followed by the opening of a second one, which is
exactly what the model did. Groups now carry the terminals a pending lexeme
could still become, and admissibility requires one of them to be acceptable.

## Answering the reviewer, with measurements

Twenty questions a referee would ask, and a benchmark for each, in
`src/gpu_lr1/rigor/`. The rules are deliberately inconvenient: warm up before
timing, report distributions rather than means, run each baseline in its best
documented configuration, and report a benchmark that could not run as
unanswered rather than dropping it. Results below are on one A100 80GB against
XGrammar 0.2.3 with the same tokenizer and the same documents.

**Is the mask sound (q01, q02)?** Walk the grammar - at every step choose only
among the bytes the mask admits - and hand what comes out to a real JSON Schema
validator. Anything invalid is a byte the mask should have refused. Over 1,653
generated documents from 200 schemas, **97.0% validate**, and every one of the
50 failures is attributable:

| cause | count |
|---|---|
| schema uses `dependencies`, which the front end does not lower | 17 |
| schema uses `not`, likewise | 16 |
| the `Branches` fallback, which discards the keywords a branch sits next to | 17 |

There is no unattributed failure. Split by lowering level the picture is
sharper, and less comfortable:

| level | valid | of |
|---|---|---|
| `Unordered` | 98.9% | 1,591 |
| `Ordered` | 62.2% | 45 |
| `Branches` | 5.9% | 17 |

The last-resort level buys coverage with a mask that is mostly wrong. Two
schemas use it. A deployment that would rather be refused than misled should be
able to cap the search, and reporting the level a schema compiled at is the
minimum honesty.

**Two host costs were hiding in our own fill.** Awkward, for a design whose
argument is that host costs on the critical path are the problem. `fill_mask`
read the live configuration count back *from the device* every step, a
synchronisation that bought nothing because the kernel already guards on it and
the host had put the counts there. And two Triton launches cost about 110us of
host time to *issue* - argument marshalling, not arithmetic - which on a small
schema was the whole measurement. Removing the sync is what makes the fill
capturable as a CUDA graph, which takes batch 1 from 107us to 12us with
bit-identical masks.

**Per-step cost (q10, q11, q15).** Median over four schemas, charging both the
fill and the advance:

| batch | 1 | 8 | 32 | 128 | 512 |
|---|---|---|---|---|---|
| ratio, whole step | 0.96x | 0.66x | 3.13x | 8.01x | **15.06x** |
| ratio, only what cannot overlap | 1.19x | 1.31x | 1.53x | 1.51x | 1.48x |

The second row is the one to believe. XGrammar's fill can be hidden behind the
forward pass by a worker thread; the advance cannot, because it follows the
sampled token. The tail is ours: at batch 512 on one schema we are p50 2,550us
and p99 2,561us against 11,487us and 12,342us - a 10us spread against 855us.

**Compile time (q18), remeasured at the new coverage.** Cold, no cache on
either side, over all 510 schemas that compile: **153 ms p50 against XGrammar's
16 ms**, p99 11.5 s against 0.6 s. Split by whether the grammar forks:

| | n | p50 | p90 | p99 |
|---|---:|---:|---:|---:|
| conflict-free | 446 | 125 ms | 1.4 s | 8.6 s |
| forked, newly compiling | 64 | **3.0 s** | 10.2 s | 28.2 s |

So the schemas GLR-lite added cost about 24x the median to compile. That is the
price of the coverage, and it is charged to the schemas that needed it rather
than spread over the rest.

Where the time goes, over the whole corpus:

| stage | share |
|---|---:|
| grouping the vocabulary | **62.4%** |
| building the lexer DFA | 34.4% |
| LALR tables, including the conflict forking | 2.8% |
| everything else | 0.4% |

Grouping scans all 151,669 tokens from every lexer state. It is embarrassingly
parallel and already parallelised across twenty-four cores; it is simply the
work, and it is exactly the work a host-side matcher repeats at every decode
step instead of doing once. Moving it after the tables build - so a lowering
level that is going to be refused does not pay for a vocabulary it will throw
away - took p99 from 15.2 s to 11.5 s and left p50 alone, since the median
schema succeeds at the first level and pays either way.

Being 8x slower cold is the honest number, and it matters because schemas
arrive per request. The structural answer is to group lazily - a real document
reaches 2-44% of the states its grammar can - which `group_state` already
supports and the pipeline does not yet use.

**Memory (q16).** The median schema costs 0.94 MB resident, about seven tokens
of KV cache for a 7B model. This is not comparable to XGrammar's 52 KB cache
and should not be printed beside it: XGrammar keeps an automaton on the host
and recomputes the token mapping every step, and buying out that recomputation
is what the memory is for.

**The decisive measurement (q09), corrected.** The first attempt at this
concluded that grammar cost is at most 5% of a decode step and that the
performance argument was therefore dead. That was wrong, and wrong in the
direction that flattered the host-side baseline, for a reason worth recording:
the denominator was a HuggingFace model in eager mode, which answered 30 ms at
*every* batch size from 1 to 512. No forward pass behaves that way. It was
timing the Python interpreter.

A serving engine captures the decode step as a CUDA graph precisely to delete
that overhead. Measured that way the step is 4.9 ms at batch 1 and 22.1 ms at
batch 512, and the picture inverts:

| batch | captured step | gpugrammar | XGrammar | end-to-end |
|---|---|---|---|---|
| 1 | 4.9 ms | 0.5% | 0.4% | 1.00x |
| 32 | 6.5 ms | 3.2% | 7.3% | 1.04x |
| 128 | 9.4 ms | 4.1% | 18.2% | 1.13x |
| 512 | 22.1 ms | 5.1% | **30.2%** | **1.24x** |

On the schema that stresses each engine most, XGrammar reaches 52.7% of a
decode step at batch 512 and the end-to-end gain is 1.37x. This is a whole-
system result, not a microbenchmark ratio.

**Per-step cost, with the parser resident (2026-07-26).** Both engines charged
for the fill *and* the advance, on a batch where each sequence sits at its own
point in its document - which is what a serving batch looks like, and which
matters because our fill deduplicates. Median over four schemas, in isolation:

| batch | 1 | 8 | 32 | 128 | 512 |
|---|---|---|---|---|---|
| whole step, isolated | 0.20x | 0.60x | 1.89x | 3.82x | **7.83x** |

**Overlap, twice corrected (q10).** Earlier versions of this document said the
advance "cannot overlap, because it follows the sampled token". That is wrong.
The forward pass follows the same token: a decode step embeds what was sampled
at `t-1`, and so does the parser. Neither needs the other, and the mask is not
wanted until the logits exist. So a step is

    sample(t-1)  ->  forward pass       ->  apply mask  ->  sample(t)
                 ->  advance + fill     ->

with the middle branches concurrent. Both engines can do this. Measured with
the decode step and the grammar step each captured as a CUDA graph, on separate
streams, schema 2:

| batch | forward pass | ours alone | ours overlapped | XGrammar alone | XGrammar overlapped |
|---:|---:|---:|---:|---:|---:|
| 32 | 6,535 us | 333 | **+108** | 857 | **+47** |
| 128 | 9,435 us | 360 | **+148** | 3,297 | **+154** |
| 512 | 22,111 us | 381 | **+113** | 12,362 | **+506** |

Correcting it the first time went against us and was recorded that way: at
batch 512 ours then cost 3,381 us alone and 3,334 us overlapped, against
XGrammar's 12,487 us alone and 510 us overlapped. The reading was that host
work overlaps with a forward pass by using a resource it is not using, while
device work overlaps by sharing the very multiprocessors the forward pass is
saturating - so a device-resident parser is cheaper in isolation and harder to
hide, and the second effect was the larger one.

The structural half of that is still true. What was wrong was treating 3,381 us
as the cost of a device-resident parser rather than as the cost of *this*
implementation. The grid was one program per (sequence, configuration, group),
sized by the configuration ceiling and by the largest number of groups any
lexer state has - 841,000 programs at batch 512, of which 93% to 95% exited
immediately. Enumerating the work instead of the ceilings took the whole
grammar step to 451 us, and at batch 512 it now costs 202 us of wall clock
against XGrammar's 419.

So the honest statement is narrower than either previous version. Device work
does compete with the forward pass for the same multiprocessors, and 45% of
ours fails to hide where 96% of XGrammar's host work does. That penalty is
real. It is simply smaller than the thirty-fold difference in what there is to
hide, once the work is the work rather than the ceilings. At batch 32 XGrammar
still adds less - 47 against 108, which is 0.9% of a step - and at 128 the two
are level.

Captured, which is what a serving loop replays, a whole grammar step is 133 us
at batch 512 and 56 us at batch 1: fill 84 and 29, advance 49 and 27.

**The fill cannot be captured (q22).** This is the structural finding and it is
binary rather than a matter of microseconds. A CUDA graph records device work;
host work inside the captured region does not go in at all. Attempting to
capture XGrammar's fill produces an empty graph - PyTorch says so - and replay
then reproduces whatever the host buffer happened to hold. Our fill captures
and replays bit-identically.

A serving engine that captures its decode step therefore cannot put a host-side
mask inside it. The fill has to be hoisted out and joined to the graph, which
reinstates the synchronisation the graph existed to remove, and forecloses
running several decode steps - speculative drafts, multi-step scheduling -
without returning to the host between them. That is what device residency buys,
and no amount of optimising a host-side fill can buy it.

**A mixed-schema batch (2026-07-28), which had never been compared.** Every
number above puts one schema under the whole batch. That is not what a serving
batch looks like - requests bring their own - and it is the case that should
hurt *us* rather than them: our fill deduplicates rows sharing a grammar, a
lexer state and a stack, and a mixture has fewer to share, while our ceilings
are maxima over the pool, so one large schema sizes the buffers for every
sequence. XGrammar's per-sequence host call is schema-agnostic; mixing costs it
nothing structurally.

Measured with a control - each schema also run alone at the same batch size, so
what is reported is the cost of *mixing* rather than the cost of whichever
schemas were in the mixture:

| batch | schemas | ours | XGrammar | ratio | mixing costs us | and them |
|---:|---:|---:|---:|---:|---:|---:|
| 128 | 1 | 269 us | 545 us | 2.03x | 1.01x | 1.08x |
| 128 | 2 | 879 | 693 | **0.79x** | 1.68x | 1.10x |
| 128 | 4 | 884 | 1,607 | 1.82x | 1.43x | 0.99x |
| 128 | 16 | 1,730 | 7,026 | 4.06x | **3.03x** | 0.66x |
| 512 | 1 | 342 | 2,078 | 6.08x | 0.99x | 1.06x |
| 512 | 2 | 951 | 2,723 | 2.86x | 1.60x | 1.16x |
| 512 | 8 | 1,801 | 35,616 | **19.78x** | 2.51x | 0.83x |
| 512 | 16 | 2,251 | 45,211 | **20.08x** | 3.49x | 1.06x |

The prediction held: **mixing costs us up to 3.5x and costs XGrammar nothing.**
That is the one place this design is structurally weaker, and it had not been
measured. It is also not decisive, because their absolute cost is what a
mixture multiplies - 35.6 ms at batch 512 over eight schemas is longer than the
decode step it is meant to hide behind.

Read against the single-schema numbers, the effect runs both ways: those
*understate* the gap at batch 512 with many schemas, where it reaches 20x, and
*overstate* it at batch 128 with two, where we lose at 0.79x.

**Where a small batch's microseconds went (2026-07-28).** The step lost to
XGrammar below about batch 100, and profiling said why: at batch 32 only 27% of
134.8 us of device time was the replay. The rest was fourteen small kernels
whose cost barely moved with the batch. Four fixes, each measured:

| | was | now |
|---|---:|---:|
| `_hash_kernel` | 8.1 us | **2.6** |
| `_dedup_kernel` | 8.6 | **3.2** |
| counting and prefix-summing | 15.0 | **8.8** |
| `_broadcast_kernel` | 11.9 | **2.5** |
| `_commit_kernel` | 8.6 | **3.9** |
| clearing the candidate flags | 2 MB a step | **gone** |
| **whole step, batch 32** | **134.8** | **103.0** |
| **whole step, batch 512** | **207.9** | **149.5** |

Three of the five are the same bug this design keeps making: a loop over the
*ceiling* rather than over what exists. `for config in range(0, CONFIGS)` is
unrolled whole, so hashing a sequence's one to twelve configurations ran 128
bodies each loading 256 stack slots. The candidate flags were one per slot, so
every step cleared 2 MB to make room for a few dozen answers - replaced by a
count per configuration, which also removed the commit's ceiling loop. The
fourth was bandwidth: the broadcast is a copy and ran one program per sequence,
32 programs on 108 multiprocessors, moving 606 KB at 51 GB/s.

Re-sweeping the sweep width afterwards moved it from 2,048 blocks to 4,096:
103 and 149 us against 103 and 175. Fewer is much worse (256 blocks is 180 and
516), which says the blocks were never the floor - the items were.

Against XGrammar, charging both engines the whole step with one
synchronisation:

| batch | ours | XGrammar | was | now |
|---:|---:|---:|---:|---:|
| 32 | 162.9 us | 133.1 us | 0.61x | **0.82x** |
| 128 | 189.3 | 512.7 | 1.97x | **2.71x** |
| 512 | 217.5 | 1,998.6 | 5.77x | **9.19x** |

**We still lose at batch 32**, and the remaining 103 us of device time is 36.6
of replay and the rest spread over kernels that are now at their launch floor.
Closing it needs the replay itself to shrink, not more of this.

What did change is the answer to "what if XGrammar ported its fill to the GPU".
Their advance stays on the host either way, and at batch 512 it alone is 330 us
against our whole step at 244: **1.35x, where before this it was 0.94x.** A
perfect port of the half they could port would no longer be enough.

**Where the step still is, and three ways of shrinking it that did not work
(2026-07-28).** At batch 32 the fill enumerates 9,977 work items and admits
**344 of them - 3.4%**. So 96.6% of the replay is spent discovering a refusal,
which is the same shape as XGrammar's own finding that under 1% of the
vocabulary is context-dependent. Three ways of exploiting that were measured
before being built, and all three are too weak to pay for themselves:

| idea | what it would remove | measured |
|---|---|---|
| precompute admission per (parser state, group) | all of it | **2.95 GB** of bitset on one schema |
| refuse a group whose readings' first terminals have no action | the enumeration | 1.4-1.8x fewer items |
| replay each *distinct* reading of a state once, not once per group | duplicated replays | 1.3-1.6x fewer replays |

The second and third look much better in the aggregate than they are in place:
readings are shared 2x to 22x across a whole grammar but barely at all within
one lexer state, and most groups' first terminal is one the parser state does
allow. Refusals happen deeper than either filter can see.

So the replay is the work, and making it cheaper needs a table that does not
fit rather than a better filter. That is the honest state of the remaining gap
at small batch, and it is recorded rather than left as an intention.

**Why a mixture costs us (2026-07-28), attributed.** Mixing was measured at up
to 3.5x for us and about 1x for XGrammar, and two causes were possible: our
deduplication finds less to share, which is inherent, or our ceilings are
maxima over the pool, which is not. Four conditions separate them - one schema;
one schema with the sequences spread over its document so deduplication has as
little to share; the mixture; and the mixture's *pool* with every sequence
under one schema, which is the ceilings alone:

| batch 128 | step | distinct states | work items | ns per item |
|---|---:|---:|---:|---:|
| one schema | 186 us | 10 | 4,131 | 30.5 |
| one schema, states spread | 293 | 21 | 9,977 | 22.3 |
| **8 schemas mixed** | **1,209** | 60 | 45,792 | 36.1 |
| one schema, mixed pool's ceilings | 191 | 10 | 4,131 | 31.3 |

**The ceilings are almost innocent: 1.06x.** Carrying another grammar's window
and group count costs essentially nothing, because the sweep enumerates the
work rather than the ceiling - which is the property this design was rebuilt
around, now confirmed on the case it was meant for.

What the mixture actually does is *make more work*: 45,792 items against 4,131,
because eight schemas at sixty distinct parse states have that much more to
replay. Per item the mixture is 36.1 ns against 22.3, and that 1.6x is the
whole of the fixable part.

So the mixed penalty is mostly real work and only slightly implementation. That
is worth knowing before optimising it: **per-grammar ceilings would buy 6%.**

One attempt is recorded as a failure. Five kernels find which configuration
owns a work item by binary searching the offsets, twelve dependent loads each,
so the answer was written down once and read instead. It was **twice as slow**:
the table is sized by the ceiling - rows times the widest state's group count -
so writing it moves megabytes to save loads that were already in cache. The
searches stay.

**Most of the replay does not depend on the stack (2026-07-28).** XGrammar's
central idea is that a token's fate is usually decided by the automaton state
alone, so the context-dependent set is under 1% and everything else is cached.
The LR analogue is exact: a group's readings run against the parser state on
top of the stack, and a reading that only ever *shifts* never looks below the
top. Only a reduce pops, and only a pop exposes the stack.

Measured over 6.9 million group replays on eight real grammars:

| | share |
|---|---:|
| refused without the stack | **91.0%** |
| admitted without the stack | 1.5% |
| needs the stack, because a reading reduces | **7.5%** |

Median schema 88.1%, worst 82.3%. So the refusals are precomputed when the
tables are built - two bits per (lexer state, parser state, group) - and the
runtime replays only what is left. Refusals only: a reading that survives by
shifting still has to run the pending-lexeme probe, which reduces, so
admissions stay undecided. That is also the safe direction, since a wrongly
precomputed admission would widen a mask and a wrongly precomputed refusal is
caught by the corpus.

| | was | now |
|---|---:|---:|
| `_mask_kernel` at batch 32 | 36.6 us | **22.5** |
| whole step, batch 32 | 103.0 | **88.0** |
| whole step, batch 512 | 149.5 | **135.0** |
| whole step, batch 512, with the search shortcut too | 149.5 | **116.2** |
| against XGrammar at batch 32 | 0.82x | **0.94x** |
| at batch 128 | 2.71x | **3.04x** |
| at batch 512 | 9.19x | **9.53x** |

**It is bought with memory, and the memory is reported.** Over 25 corpus
schemas the resident tables are 3.27 MB at the median, 6.43 at p90, and the
verdict table is 25% of that - so it raised the median schema's residency by
about a third. It is abandoned above four million words, so a pathological
grammar falls back to replaying everything and pays nothing.

That trade is the same one this whole design makes, applied one level down: the
thing we are worst at against XGrammar is memory, and this makes it worse to
make the step faster. Reporting both is the only honest way to present it.

The same table makes the advance's token search cheaper too - a group already
refused cannot be the one that advances - and that is another 19 us at batch
512. Getting there found a latent bug worth more than the microseconds.

The search shortcut made two corpus schemas produce *more* configurations than
the reference matcher, which is impossible for a change that only removes
candidates. Four measurements settled it: the table agrees with an actual
replay over every (lexer state, parser state, group) of a grammar
(`files/verify_verdicts.py`); the plumbing with a filter that cannot match
passes all 591 corpus steps; the filter provably only ever turns "found group
75" into "found nothing"; and yet the answer grew.

The cause was in an earlier change of ours. Replacing the per-slot candidate
flags with a per-configuration count removed a 2 MB clear from every step - but
`_candidate_kernel` writes the count *inside* the branch that runs when a group
was found, so a configuration whose token is in no group kept the count the
previous step left behind, and the commit read candidates that were not there.
It survived because a configuration almost always finds a group. The search
shortcut made that common, and that is how it surfaced.

**Not only JSON (2026-07-28).** Every measurement above is on JSON Schema,
which invites the obvious objection that this is a JSON parser with an LR(1)
story attached. A regex is the cheapest constraint that is not JSON, both
engines take one directly, and the front end already lowers one. Fifteen
patterns of the shape a deployment actually constrains with - identifiers,
dates, UUIDs, paths, quantities - each walked over a string it accepts:

| | |
|---|---|
| patterns whose masks and configuration sets agree with the reference | **15 of 15** |
| median step against XGrammar at batch 128 | **6.25x** |
| worst | 3.83x |
| compile time, median | **6.6 ms**, against 120 for a schema |

The interesting number is the spread rather than the median. XGrammar's cost
depends on the pattern by a factor of **240**: one fill is 1.6 us on `[0-9]+`
and **385 us** on `[a-z]+(,[a-z]+)*`, where after a letter both a letter and a
comma continue and the uncertain set is large. Ours moves by 2.4x over the same
set, 84 to 206 us, because the per-step work is a function of the automaton
rather than of how many tokens need deciding.

That is the tail llguidance criticises, measured here on a pattern nobody would
call adversarial - and it is the same asymmetry as the batch scaling, in a
second dimension: their cost tracks the work the constraint implies, ours
tracks the constraint's shape.

**llguidance as a second baseline (q19, 2026-07-28).** Only XGrammar had been
compared, and llguidance claims fast mask computation and criticises
XGrammar's tail specifically - so a reviewer would rightly ask why the fastest
baseline was not chosen. All three are charged the same way: a mask for every
sequence and an advance for every sequence, which is what a decode step costs.

| batch | median vs XGrammar | median vs llguidance |
|---:|---:|---:|
| 32 | 2.26x | 4.00x |
| 128 | 6.42x | 10.31x |
| 512 | 21.33x | 37.56x |

On JSON Schema at batch 128 it is 4.19x and 17.62x. **llguidance is slower than
XGrammar throughout on this workload** - 0.24x their speed on schemas, 0.63x on
regex - so adding it does not change the ranking, and saying so is the point of
having measured it.

Its tail claim holds, though, and it is the one place either baseline behaves
better than the other by a wide margin: on `[a-z]+(,[a-z]+)*`, where XGrammar
needs 45,354 us at batch 128, llguidance needs 963. That is the pattern where
XGrammar's uncertain set is large, and llguidance's design does not have that
failure mode. Ours needs 208 us.

One schema of the six was refused outright: llguidance does not support `oneOf`
without a coercion flag. Left as it is rather than tuned, since the default
configuration is what a deployment gets.

Two things follow for the paper. The speed argument does not rest on choosing a
weak baseline - the stronger claim, llguidance's, is about the tail rather than
the median, and it is right about the tail. And **at batch 32 on regex we are
ahead of both**, 2.26x and 4.00x, where on JSON Schema at batch 32 we are behind
XGrammar at 0.94x. The small-batch loss is a property of the schema workload's
per-step cost, not of the design.

**SQL, where LR(1) does work no finite automaton can (2026-07-28).** Regex
answered "is this only JSON?" and did not answer "why LR(1)?" - a regex is a
DFA, and JSON's nesting is bounded by the schema that generated it. A SELECT
subset with a real expression rule is the case that needs the other thing: an
expression may nest parentheses arbitrarily, and no finite automaton can match
`(((...)))` to unbounded depth.

Thirteen queries - projections, aliases, joins, an expression with precedence
and parentheses, GROUP BY, LIMIT - walked byte by byte against the reference
matcher: **all thirteen agree** on every mask and every configuration set. And
the stack does what the grammar class says it must:

| nesting depth | 1 | 8 | 32 | 128 | 256 |
|---|---:|---:|---:|---:|---:|
| deepest parser stack | 10 | 17 | 41 | 137 | ceiling |

**One entry per level, exactly.** That is the property a DFA cannot have, and
it is why the machinery is LR(1) rather than regular. At 256 the stack reaches
`max_stack`, which is a documented ceiling rather than a silent truncation.

Against both baselines on the same grammar, charging each for a mask and an
advance per sequence:

| | batch 1 | batch 32 | batch 128 | batch 512 |
|---|---:|---:|---:|---:|
| vs XGrammar, median | 41x | 1,244x | 3,981x | 9,613x |
| vs llguidance, depth 1 | **0.15x** | **0.84x** | 2.89x | 8.47x |
| vs llguidance, depth 32 | - | 1.91x | **6.10x** | **17.26x** |

Three things, and the first is not in our favour.

**llguidance beats us at small batch**, 27 us against our 179 at batch 1. Its
per-sequence cost on this grammar is tiny and ours is a fixed device cost, so
below about batch 32 it is simply the better engine here.

**XGrammar collapses on a real context-free grammar.** A single sequence's fill
reaches 70 ms at the `=` of `WHERE id=42`, verified by walking the query and
confirming its masks are non-empty and its tokens accepted - this is their
engine working correctly, not a misuse. An expression can continue with almost
any identifier, number or operator, so the context-dependent set is enormous,
and that is the case their adaptive cache is not built for. Where llguidance is
27 us they are 17,329.

**Only our cost is flat in the nesting depth.** At batch 128, going from depth
1 to depth 32 takes llguidance from 654 us to 1,653 while ours goes from 226 to
271. Depth is what a stack is for, and it is the axis on which a device-resident
stack stops being an implementation detail: the deeper the grammar the wider the
margin, 2.89x to 6.10x at batch 128 and 8.47x to 17.26x at 512.

The cost is at the front. This grammar takes **1.3 seconds to compile** against
XGrammar's 88 ms, and its tables are **91 MB** - by far the worst numbers in
this document, and the same trade as everywhere else, at its extreme.

**Neither engine is sound, and ours was claimed to be (2026-07-28).** This
document has said for a long time that the engine only ever *narrows* - that
dropping a configuration or reaching a ceiling can make a mask stricter but
never looser. That is true of the runtime. It says nothing about the lowering,
and the lowering is where it fails.

`oneOf` in JSON Schema means **exactly one** branch matches. We lower it to a
union, which means **at least one**. The two readings differ on any document
that matches two branches, and objects match two branches easily, because
`additionalProperties` is permitted unless a schema forbids it:

```json
{"oneOf": [
  {"type":"object", "properties":{"a":{"type":"string"}}, "required":["a"]},
  {"type":"object", "properties":{"b":{"type":"string"}}, "required":["b"]}]}
```

`{"a":"x","b":"y"}` matches both branches, so the schema rejects it. A real
validator agrees. **We accept it. XGrammar rejects it.** On this case they are
right and we are wrong.

Adding `additionalProperties: false` to both branches makes the branches
disjoint and the conflict disappears - `max_actions` goes from 2 to 1. So the
conflicts this corpus produces and the unsoundness have **the same cause**: a
`oneOf` whose branches overlap is both ambiguous to parse and mis-lowered by a
union. The GLR-lite work follows the ambiguity; it does not fix the semantics.

Over the corpus, against a real validator:

| of the 22 documents that violate their own schema and whose schema compiles | |
|---|---|
| **we admit** | **7 (32%)** |
| **XGrammar admits** | **22 (100%)** |
| llguidance | refuses these schemas outright - `oneOf` needs a coercion flag |

Charged the same way, with their `is_completed` standing for our
`can_terminate`, so neither gets the laxer test.

Three things follow, and the first is the uncomfortable one.

**The claim was wrong and is withdrawn.** "This engine only narrows" holds for
the runtime and not for the pipeline, and the paper cannot state it unqualified.

**No engine here is sound.** XGrammar admits every invalid document in the
corpus; llguidance declines to answer by refusing the construct. Constrained
decoding is generally presented as a guarantee, and on `oneOf` it is not one in
any of these implementations. That is worth saying in a paper rather than
leaving for a reader to discover.

**We are tighter than the baseline, which is the honest form of the claim.**
32% against 100% is a real difference and it comes from the precision search -
the pipeline tries an exact lowering before a relaxed one, and reports which it
settled on. What it cannot yet do is refuse rather than relax, which is the
option a deployment that would rather be told than misled should have.

The exclusivity `oneOf` asks for is not context-free in general, so a parser
cannot simply be made to enforce it. What is achievable is narrower and worth
doing: lower `oneOf` exactly when the branches can be shown disjoint - which
closing the objects does - and report the approximation when they cannot.

**Host contention (q21).** Weaker than expected and worth saying so. With
twenty-four cores deliberately saturated, XGrammar's fill slows by 1.06x and
ours by 1.01x; both engines' p99 degrades to about 3 ms, which is the operating
system rather than either design. Contention is not the argument. Capturability
is.

**Still unanswered.** End-to-end serving with error bars (q08), whether
XGrammar can be made to accept any property order (q06), grammars beyond JSON
Schema and regex - a programming language or a query language, where LR(1) is
doing work a DFA could not (q07) - speculative decoding in a serving path
(q13), depth scaling (q14), Outlines as a third baseline (q19), and the
per-mechanism ablation (q20).

**Threats that remain.** Table construction is still excluded — rows are
replayed, while XGrammar computes masks online from a compact automaton, so
compile time and incremental admission (C7) must be measured before any
end-to-end claim. There is one GPU, one generating model, and no serving
integration, so these are isolated-step numbers and cannot be headlined.

**What is left.** Pure-wide batches above 128 still run at 0.8× of
unconstrained because the search costs ten sweeps of the vocabulary. Replacing
the multi-probe search with a single-pass histogram would cut that to about
four sweeps; it is the only remaining algorithmic gap.

## Non-goals

- Beating XGrammar on a grammar with ~15 allowed tokens per state. That is the
  most favorable possible case and is already demonstrated; it is not the paper.
- Supporting ambiguous grammars, or general CFGs beyond LR(1) power.
- Training-time or model-side changes. This is a decoding-time system.
- Any claim of general unbounded tokenizer-aware LR(1) support until C1–C3 are
  actually solved.

## Reporting rules

Decision 3 fixes the workloads; these rules govern how any number derived from
them is reported.

- **Baselines:** XGrammar, llguidance, Outlines, plus the closest parser-aware
  systems (Pre3, PSC) and Gram2Token where reproducible. The sampler baseline is
  FlashInfer (`top_k_top_p_sampling_from_logits`), not a `torch.sort` reference
  implementation — a sorted-softmax sampler is 5× slower and would be a strawman.
- **Metrics:** end-to-end tokens/s inside a real serving engine (not an isolated
  microbenchmark), constraint-step device time, dispatch overhead, table memory,
  compile latency, and mask exactness rate.
- **Ablations:** per representation (CSR / bitset / complement / interval),
  per stack abstraction, graph-resident vs. launched, IELR(1) vs. LALR(1) vs.
  canonical LR(1) table sizes.
- **Reporting rule:** every reported speedup states batch size, allowed-token
  distribution, and whether model execution is included. Isolated-microbenchmark
  speedups are never headlined.
- **Headline rule:** mask-fill throughput is never the headline claim. It is
  reported for parity, not for victory. The headline claims are the ones a
  host-side matcher cannot make at all: speculative decoding at depth, sampler
  fusion, and whole-loop graph capture.
- **Residency cost is reported beside every residency benefit.** Any figure
  showing what device residency buys carries the table memory it costs, on the
  same page.

## Working agreement

- Prototype code in this repository is evidence, not product. `gpugrammar` is a
  clean implementation informed by it.
- No claim enters README, GOAL, or the paper without a measurement or a proof.
- At every major milestone: run focused tests and the relevant benchmark, update
  documentation and results, commit with the Copilot co-author trailer, push.
