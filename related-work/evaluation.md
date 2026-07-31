# Evaluation plan derived from prior work

The current benchmark establishes an isolated GPU-kernel feasibility result.
The literature suggests the following additional work before making a
production or general-LR claim.

## 1. Correctness and schema coverage

Adopt the terminology from
[JSONSchemaBench](https://arxiv.org/abs/2501.10868):

- **declared coverage:** the compiler accepts a schema;
- **empirical coverage:** generated instances validate against the schema;
- **compliance rate:** empirical coverage divided by declared coverage;
- **over-constraint:** a schema-valid continuation is incorrectly rejected;
- **under-constraint:** a schema-invalid continuation is incorrectly accepted;
- **true coverage:** exact equivalence to the schema semantics, approximated by
  official JSON Schema test cases.

Required experiments:

1. Run the compiler over the full JSONSchemaBench corpus.
2. Report accepted/rejected schemas by corpus split and keyword.
3. Run the official JSON Schema Test Suite for every accepted keyword family.
4. Generate both valid and adversarial invalid instances for accepted schemas.
5. Require zero silent keyword drops: unsupported semantics must produce a
   compile error.
6. Report recursion, property-order, whitespace, Unicode, numeric, and regex
   behavior separately.

The project's canonical-output restriction should be treated as an explicit
language subset. It can be sound while intentionally incomplete.

## 2. Tokenizer alignment

The tokenizer/grammar mismatch is a central issue in Geng et al., SynCode,
DOMINO, Automata-based Constraints, XGrammar, and Flexible and Efficient GCD.
Testing one truncated GPT-2 vocabulary is insufficient.

Minimum matrix:

| Family | Example | Property to stress |
|---|---|---|
| Byte-level BPE | GPT-2 | Existing baseline and multi-terminal tokens |
| Large byte-level BPE | Llama 3 / Qwen | 100K+ vocabulary table growth |
| SentencePiece BPE | Llama 2-like tokenizer | Different whitespace and byte fallback behavior |
| Unigram | T5/SentencePiece unigram | Ambiguous tokenization and alternate segmentations |

For each tokenizer report:

- vocabulary size and byte encoding rules;
- compile time and peak host/device memory;
- number of reachable DFA/token pairs;
- CSR edges per state and mask-density distribution;
- cross-terminal and cross-JSON-lexeme token frequency;
- differential agreement with a byte-by-byte reference matcher.

## 3. Schema diversity and heterogeneous batching

The main engrain hypothesis is not merely large batch size. It is a large batch
containing many distinct schemas and active grammar states.

Measure:

- batch sizes `1, 8, 32, 128, 512, 2048` where memory permits;
- distinct schema counts `1, 4, 16, 64, batch_size`;
- repeated-schema and all-distinct-schema controls;
- narrow and wide state-count distributions;
- sparse, medium, and dense allowed-token rows;
- correlated states versus randomly scattered global state IDs;
- cold schemas entering a warm continuous batch;
- table-cache hit and eviction behavior.

Report both aggregate throughput and fairness/tail effects on warm co-tenants.
GRID and the SqueezeBits serving study show that cold grammar work can affect
other requests even when steady-state masks are fast.

## 4. Runtime representations

Continue comparing:

- dense boolean masks;
- packed accept bitsets;
- packed reject bitsets;
- CSR accepted-token edges;
- CSR rejected-token edges;
- compact byte-DFA advance after sampling;
- adaptive per-row storage selecting the smallest representation.

For every density bucket report:

- p50, p90, p95, p99, and p99.9 wall time;
- CUDA kernel time;
- bytes read/written per sequence;
- resident table memory;
- occupancy and achieved memory bandwidth where profiling is available;
- selected-token and next-state correctness.

MaskBench demonstrates why median-only reporting is inadequate: an engine can
win at p50 and still stall an entire batch at p99.

## 5. Compilation and caching

Offline preprocessing is a first-class metric in SynCode, DOMINO, Flexible and
Efficient GCD, XGrammar-2, PSC, and Gram2Token.

Measure:

- grammar compilation time;
- determinization time;
- tokenizer cross-product time;
- packing and device-upload time;
- time to first mask / time to first token;
- peak compiler RSS and GPU temporary memory;
- cold on-disk cache miss;
- warm process cache hit;
- persistent cache load;
- cross-schema substructure reuse;
- incremental addition of one schema to an existing table set.

Report percentiles across the full schema corpus rather than only total time for
14 or 64 hand-written schemas.

## 6. Recursive parser backend

The terminal-level recursive backend should be extended and compared across at
least three representations:

1. direct LR action/goto execution with a device stack;
2. Pre3-style prefix-conditioned DPDA edges;
3. PSC-style parser-stack classification if the full construction becomes
   publicly available.

Required stress cases:

- deeply nested objects and arrays;
- recursive `$ref`;
- long and variable-length reduction chains;
- many reductions without consuming a byte;
- divergent stack depth within one warp;
- rollback/fork for beam and speculative decoding;
- malformed-prefix rejection;
- stack overflow and configured depth limits.

The existing one-push/one-pop stack microbenchmark is useful but does not model
LR reduction loops.

## 7. Sampling semantics

The current greedy argmax path is not enough for serving integration.

Implement and compare:

- temperature sampling;
- top-k;
- top-p;
- min-p or serving-engine-equivalent filters;
- repetition/frequency/presence penalties;
- sparse-row sampling without materializing a full vocabulary mask;
- dense-row fallback;
- deterministic RNG behavior across fused and reference paths.

Quality evaluation should distinguish:

- ordinary hard-mask sampling;
- distribution-faithful goals from Grammar-Aligned Decoding;
- constrained-region switching from CRANE;
- draft-conditioned approaches such as DCCD.

## 8. Speculative and jump-forward decoding

Required capabilities:

- fork and rollback grammar state;
- verify multiple draft tokens in one kernel launch;
- traverse a speculative tree;
- preserve RNG and parser state after partial acceptance;
- fast-forward deterministic spans;
- re-tokenize or otherwise handle token boundaries after jump-forward.

Compare with:

- XGrammar draft-tree traversal;
- SGLang compressed-FSM jump-forward;
- DOMINO opportunistic/speculative masking;
- MaskBench fast-forward-token share.

## 9. End-to-end serving

Kernel latency must be connected to actual model throughput.

Recommended experiment:

- integrate as a structured-output backend in vLLM or SGLang;
- use one small model where grammar overhead is visible and one larger model
  where model compute dominates;
- test continuous batching with dynamic arrivals;
- compare unconstrained, XGrammar, llguidance, and engrain paths;
- separate grammar compile time from decode time;
- rotate schemas so cache behavior is controlled;
- report TTFT, TPOT, requests/second, tokens/second, p99 request latency, GPU
  utilization, CPU utilization, H2D traffic, and table-cache memory;
- include model execution and matcher state acceptance/rollback in every path.

The current XGrammar microbenchmark is intentionally optimistic and should
remain as an isolated lower-bound comparison, not the sole production claim.

## 10. Baseline comparability

Every result table should record:

- exact engine and commit/version;
- parser thread count;
- CPU model, core pinning, and NUMA placement;
- GPU model and clocks;
- tokenizer and full vocabulary size;
- schema corpus and accepted subset;
- batch composition;
- whether compile time is included;
- whether mask construction, H2D copy, mask apply, sampling, state acceptance,
  and rollback are included;
- warm-up, timed iteration count, synchronization boundary, and percentile
  definition;
- whether logits and tables are rotated to avoid unrealistic cache residency.

Do not compare:

- one engine's isolated mask latency with another engine's end-to-end TPOT;
- batch-1 results with batch-512 results without showing the scaling curve;
- systems that implement different schema subsets without a coverage table;
- CUDA event time with host wall time as if they were interchangeable.

## 11. Minimum publication-grade checklist

- [ ] JSONSchemaBench corpus compile and keyword-coverage report.
- [ ] Official JSON Schema Test Suite differential results.
- [ ] Zero silent unsupported-keyword acceptance.
- [ ] At least three tokenizer families.
- [ ] Homogeneous and all-distinct heterogeneous batch sweeps.
- [ ] p50 through p99.9 wall-time distributions.
- [ ] Cold/warm compile and persistent-cache measurements.
- [ ] Memory growth versus number of resident schemas.
- [ ] Full sampling semantics or an explicitly scoped greedy-only result.
- [ ] Recursive/rollback benchmark before claiming LR(1) support.
- [ ] End-to-end serving TPOT and throughput.
- [ ] Reproducible XGrammar and llguidance baselines.
- [ ] Clear separation of paper-verified, source-verified, and abstract-only
      comparisons.
