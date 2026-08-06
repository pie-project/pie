# Provenance

Every number in `engrain.tex` comes from a harness in this repository, run on
one NVIDIA A100 80GB PCIe. This file maps each table to what produced it, so a
number can be re-derived rather than trusted.

Numbers that were measured against code paths since deleted are **not** in the
paper. `GOAL.md` still carries some of those in its narrative sections; they
are historical record, not results.

| table | what it reports | produced by | date |
|---|---|---|---|
| 1 | full step vs XGrammar and llguidance | `files/exp_baselines_latency.py` | 2026-07-31 |
| 2 | coverage and compile time, three engines | `files/exp_baselines_coverage.py` | 2026-07-31 |
| 3 | heterogeneous batches, and the cost of mixing | `files/probe_mixed_cost.py` | 2026-07-31 |
| 4 | ablation of the memo and of graph capture | `files/exp_ablation.py` | 2026-07-31 |
| 5 | over- and under-acceptance on the exact fragment, three engines | `rigor.fragment`, `rigor.soundness` | 2026-08-06 |
| 6 | end to end in vLLM, three engines | `rigor.e2e` | 2026-08-06 |
| 7 | arena size, sharing and the verdict trade | `rigor.cost` + a live pool | 2026-08-05 |
| 8 | online serving, Poisson arrivals, four engines | `rigor.online` | 2026-08-06 |

`rigor.online` reads `results/corpus-agreed.json`, which is the exact fragment
screened once to the 417 schemas all three backends accept. Screening inside
each arm compiles the whole corpus in three engines before a single request is
served, which is minutes of nothing per arm.

## The corpus the correctness and end-to-end results use

`python -m engrain_lab.rigor.fragment` rewrites each of the 533
JSONSchemaBench schemas by applying the remedies `CompiledGrammar.relaxations`
reports, and keeps it only if the result lands inside the fragment engrain
enforces with nothing left over. **Membership is measured, not asserted**: the
schema is compiled, its relaxation list must be empty, and twenty documents are
then generated under its mask and validated. 468 survive of 533 (37 refused,
23 still relaxed, 5 reported nothing relaxed and over-accepted anyway). A
second pass under independent seeds walks 10,177 complete documents with zero
invalid.

The corpus is favourable to engrain by construction and the paper says so
wherever it is used. The under-acceptance result is not: it permutes the
baseline's own corpus documents.

## Reported in prose rather than a table

- allowed-set width (`results/rigorous-summary.json`)
- speculative decoding (`files/probe_speculative.py`)
- overlap-adjusted cost (`engrain_lab.rigor.overlap`)
- fill under CPU contention (`engrain_lab.rigor.serving`)
- mask latency by batch, vocabulary and diversity (`engrain_lab.rigor.lockstep`)
- regex and EBNF, correctness and cost (`engrain_lab.rigor.nonjson`)
- cold path from schema to first mask, three engines (session measurement,
  2026-08-06): llguidance p50 1.2 ms / max 11.5, XGrammar 20.7 / 54,473,
  engrain 227.2 / 11,901
- llguidance driven through a byte-level `LLTokenizer` for the correctness
  half, so the random walk is the same walk the other two engines take. Its
  matcher is wrapped to expose `accept_token`/`allowed_tokens`/`can_terminate`;
  the first run of that comparison reported zero complete walks because
  llguidance spells it `consume_token` and the harness swallowed the
  `AttributeError`.
- the relaxation list's completeness (`engrain_lab.rigor.relaxation`)
- `python -m engrain_lab.verify` — six differential verifications, zero failures

## Baselines

Three engines, each given its strongest interface: XGrammar with
`BatchGrammarMatcher`, the thread count swept over 1/2/4/8/16 and
**`any_order=True`**; llguidance with `LLExecutor` and its `_par` fill and
consume; outlines_core in a Python loop, because it has no batch interface.

`any_order=True` is a fairness correction: XGrammar's default fixes object
property order at the schema's declaration, so it rejects `{"b":2,"a":1}` where
engrain accepts it. It also compiles *faster* — 0.81 s against 4.89 s over 60
schemas — so it is XGrammar's better setting on both axes.

**It is also not the setting a deployment gets.** vLLM does not expose
`any_order`, so every served request uses the ordered lowering. The end-to-end
results report both, the matched one through a one-line patch to vLLM
(`XGRAMMAR_ANY_ORDER=1`, in the `ingim/vllm` fork on branch `engrain`).

XGrammar is threaded exactly where vLLM threads it: above 128 structured
requests in a step, in chunks of sixteen, read out of
`vllm/v1/structured_output/__init__.py`. An earlier version of `rigor.lockstep`
threaded above batch 16, which was wrong in both directions — at batch 32 the
pool costs more than the work, and at 128 it made XGrammar look faster than
vLLM runs it.

**llguidance has no ordering option at all.** Its JSON options are
`item_separator`, `key_separator`, `whitespace_flexible`, `whitespace_pattern`,
`coerce_one_of`, `lenient`, `json_allowed_escapes` — property order is fixed in
its lowering, so unlike XGrammar it cannot be given the order-free setting. It
is measured as it ships, which is its only setting.

outlines_core cannot be configured out of the same restriction: its lowering
emits a literal regex with the properties in declared order, so order is a
property of the language it accepts. It could not be seeded into a live state on
any of the three model-generated documents the other three engines accept, and
building its vocabulary-indexed DFA cost 33.8 s over 20 schemas against
engrain's 1.6 s, XGrammar's 0.5 s and llguidance's 0.0 s.

## Measurements that were wrong before they were right

Recorded because the paper's methodology claims depend on them.

**The ablation.** Setting `memo_slots` on a batch that has already been
captured changes a Python attribute and nothing the device does: a CUDA graph
bakes in its kernels' scalar arguments at record time. The first run therefore
reported that the memo was worth 1.00x while hitting 90% of steps. An ablation
of a graph-resident system has to be re-recorded, not re-configured.

**The soundness aggregate.** `rigor.soundness` counts a random walk that runs
out of byte budget as an invalid document, which mixes walk truncation into a
correctness metric. Decomposing it into completion and over-acceptance is what
made the failure attributable to a single keyword, and from there to a bug.

**Tokens per second.** `tok/s = tokens / makespan`, so an engine whose requests
run longer generates more tokens *and* keeps the batch fuller, and a fuller
batch spreads the per-step fixed cost over more rows. The inflation is exactly
the gap in tokens generated: 17.7% more tokens, 17% better tokens-per-second
than the makespan says. The paper quotes makespan, which needs no normalising
because both arms serve the same 512 requests.

**Resident bytes.** The number a table budget is spent in omitted the readings
and the verdict table and charged a flat 20 bytes per group: 2.9 MB reported
against a real 7.5. A 1 GiB budget was holding what a 2.6 GiB one had been
asked for. It is now counted from the arrays the arena uploads and checked
against them to 100.0%.

**A known wrong mask, reported rather than withheld.** `rigor.online` with
`ENGRAIN_VERIFY=1` shows a row whose batched mask is missing twelve words of
bits against the same row computed alone, under continuous batching only, and
three of 400 requests die with `grammar rejected tokens`. It is not
root-caused. The correctness results walk the host reference matcher and are
unaffected; the serving numbers are measured on an engine with this path live
and the paper says so.

**A cross-engine refusal is not automatically the refuser's fault.** The first
version of `rigor.nonjson` counted every string one engine generated and the
other refused against the refuser. On a wide Unicode class that reported a win
for us as a loss: XGrammar was generating overlong UTF-8, and we were right to
refuse it. A refusal now only counts when the referee agrees the string is in
the language.
