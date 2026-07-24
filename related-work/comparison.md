# Architecture comparison

This document compares the online execution boundary of the systems most
relevant to gpu-lr1. "GPU apply" means that a precomputed mask is consumed by a
GPU kernel; it does not imply that the parser or automaton executes on GPU.

## Core system matrix

| System | Constraint model | Token alignment | Runtime state | Grammar/mask computation | GPU work | Recursive CFG | Heterogeneous batching | Artifact status |
|---|---|---|---|---|---|---:|---|---|
| Outlines paper | Regex / regular language | FSM walked over vocabulary during index build | FSM state | CPU lookup over precomputed index | Backend-dependent logit mask | No in original formulation | Per-request FSM/index | Public |
| SynCode | General CFG with terminal DFAs | Offline DFA mask store over tokenizer tokens | Parser condition + live terminal DFA states | CPU table lookup | Backend-dependent logit mask | Yes | Per-request parser | Public |
| DOMINO | Formal-language constraints | Vocabulary-aligned subterminal trees | Constraint automaton state | CPU/speculative controller | Model inference and sampling | Depends on grammar backend | Not a GPU parser batch | Public |
| XGrammar | General CFG / JSON Schema | Byte-level token classification and adaptive cache | Persistent PDA/Earley state | **CPU C++** matcher and mask construction | Packed-mask application | Yes | CPU batch API + selected GPU rows | Public |
| llguidance | General CFG / regex / JSON Schema | Token trie and vocabulary slices | Lexer DFA + Earley parser | **CPU Rust** | Host-engine mask application | Yes | Per-request CPU matcher | Public |
| Pre3 | LR(1) -> DPDA | Prefix-conditioned transitions | DPDA stack | Published algorithm/reference artifact is CPU-oriented | Serving-engine dependent | Yes | Parallel stack-prefix checks; no verified GPU parser batch | Paper; limited artifact |
| PSC | Parser-stack classifier | All token acceptance conditions merged during preprocessing | Parser stack | **Not stated publicly** | Not stated publicly | Likely parser-dependent; not stated | Not stated | Abstract only |
| GRID | LALR(1) SQL | Byte-token trie + lexer/parser configuration | Lexer state + LALR stack | **CPU Rust** | vLLM-style mask application/model inference | LALR(1) | CPU request states | Preprint; no public code found |
| Gram2Token | Deterministic byte grammar -> token categories | Token trie groups equal transition outcomes | Preprocessed grammar state | **GPU claimed** | Category lookup, mask, state update | Not fully specified in public abstract | Schema-diverse continuous batching | Abstract; declared repo unavailable |
| gpu-lr1 DFA | Canonical acyclic JSON Schema -> byte DFA | Explicit DFA/tokenizer cross-product | Flat global DFA state ID | **GPU Triton** | CSR/bitset selection and next-state write | No | One global state namespace across schemas | Public prototype |
| gpu-lr1 LR(1) | Canonical LR(1) ACTION/GOTO tables | Already-segmented grammar terminals; no LLM-token bridge yet | Bounded ragged LR state stacks | **GPU Triton** | Sparse terminal selection plus reduce/goto/shift closure | Yes, at terminal level | Global state/production IDs plus per-sequence ragged stacks | Public prototype |
| gpu-lr1 bounded tokens | Bounded canonical LR(1) configurations | Token terminal trie; real bytes for byte-terminal grammars | Full LR stacks encoded as finite configuration IDs | Offline CPU expansion; **GPU runtime** | CSR token selection plus next-configuration write | Yes up to configured depth | Global configuration IDs across grammars | Public prototype |

## Device-placement evidence

### XGrammar

The public `GrammarMatcher.fill_next_token_bitmask` API states that the mask
must be on CPU. Its CUDA and Triton kernels apply a packed bitmask to logits.
The parser stack and automaton transition are not part of those kernels.
XGrammar can overlap CPU mask generation with GPU model execution, which is a
valuable serving optimization but a different architecture from eliminating
the host dependency.

### llguidance and AICI

llguidance is a Rust parser library. AICI explicitly runs Wasm controllers on
CPU while the GPU is busy with token generation. Both are designed to exploit
overlap rather than keep grammar state on device.

### SGLang, vLLM, and TensorRT-LLM

These engines provide GPU-aware scheduling and packed-mask application, but the
selected grammar backend determines where parsing occurs. Using XGrammar or
llguidance still means CPU grammar execution unless a separate GPU-native
backend is supplied.

### Gram2Token

Gram2Token's conference abstract explicitly places token-category lookup,
masking, and grammar-state update on GPU. This is the closest public claim to
gpu-lr1. The representation and coverage cannot yet be audited in detail
because the full implementation was not publicly reachable at the survey
snapshot.

### PSC

PSC's abstract is highly relevant because it turns all token acceptance
conditions into one parser-stack classifier. However, it does not say whether
that classifier executes on CPU, GPU, or both. It should not be labeled
GPU-native until the full paper or artifact establishes that fact.

## Architecture families

### 1. Candidate guards

Examples: PICARD, Synchromesh.

The model proposes one or more candidates and an external parser or semantic
oracle rejects invalid candidates. This avoids a full-vocabulary mask but does
not guarantee that every invalid vocabulary token was removed. It is attractive
for beams or semantic checks and less suitable as the sole primitive for exact
high-batch sampling.

### 2. State-to-token indexes

Examples: Outlines, SynCode, the regular subset of gpu-lr1, and gpu-lr1's
bounded LR configuration expansion.

Expensive tokenizer/grammar alignment is compiled ahead of inference. Online
execution becomes a lookup keyed by automaton or parser state. The main design
problems are:

- table size and compile latency;
- tokens spanning multiple grammar terminals;
- recursive context that cannot be represented by one finite state;
- efficient storage of sparse and dense rows;
- sharing tables across many related grammars.

### 3. CPU parser plus GPU mask application

Examples: XGrammar, llguidance, GRID, most current serving integrations.

This architecture preserves broad grammar coverage and mature rollback support.
It performs well when CPU work is small or hidden under model execution. It can
become a throughput bottleneck when:

- model steps get faster;
- batches contain many distinct parser states;
- a few complex rows create large tail latency;
- mask staging serializes sampling;
- schema compilation occurs on the request path.

### 4. Determinized stack-aware parsing

Examples: Pre3 and PSC.

These systems attempt to summarize enough stack context during preprocessing
that online parsing is deterministic or classifier-like. gpu-lr1 now provides a
baseline GPU implementation of ordinary canonical LR(1) ACTION/GOTO execution
over terminal IDs. Pre3 and PSC remain the most important route to avoiding
per-token parser work when real tokenizer tokens span multiple grammar
terminals. The unresolved systems questions are classifier memory growth,
tokenizer alignment, rollback, and mixed-grammar token masking.

### 5. GPU-resident grammar state

Examples: Gram2Token by abstract claim and both gpu-lr1 backends by public
prototype.

The goal is to keep the active grammar state in device tensors and fuse:

```text
state lookup -> valid-token selection -> sampling -> next-state update
```

The benefit is strongest for high-batch workloads where host work scales with
the number of active sequences. The risk is moving too much precomputed grammar
data into HBM or compiling recursive stack context into an impractically large
finite state space.

## Closest comparisons

### gpu-lr1 versus Gram2Token

Shared high-level ideas:

- preprocess byte-level grammar behavior into token-level transitions;
- keep grammar state and transition tables on GPU;
- target schema-diverse continuous batching;
- trade higher preprocessing/TTFT for lower online overhead.

Publicly demonstrated gpu-lr1 details not available in the Gram2Token abstract:

- flat relocation of all schema-local states into one global namespace;
- CSR `(allowed_token, next_state)` rows;
- packed-bitset and dense alternatives;
- measured CSR/bitset density crossover;
- memory scaling for 14 and 64 heterogeneous schemas;
- fused Triton token selection and state update;
- direct XGrammar wall-clock baseline.
- bounded exact stack-configuration expansion with measured state/memory growth;
- full Qwen3 151,669-token byte results and explicit compile timeouts.

Potential Gram2Token differences that remain unknown:

- how recursive stack configurations are represented;
- whether token categories are global, per grammar, or per grammar-state family;
- memory growth with vocabulary and grammar count;
- sampling semantics beyond the reported runtime pipeline;
- rollback, fork, and speculative tree support.

### gpu-lr1 versus Pre3 and PSC

Pre3 and PSC address the hard tokenizer-level form of stack-dependent recursive
parsing.

- Pre3 attaches stack-prefix conditions and operations to deterministic DPDA
  edges.
- PSC compiles the acceptance conditions of all vocabulary tokens into one
  parser-stack classifier.
- gpu-lr1's DFA backend eliminates the stack for acyclic canonical JSON.
- gpu-lr1's direct LR backend executes the stack after input has been segmented
  into grammar terminals.
- gpu-lr1's bounded token backend handles multi-terminal tokenizer tokens by
  enumerating full stack configurations, which is exact within a depth bound but
  can grow exponentially.

A tokenizer-aware recursive gpu-lr1 backend should therefore be framed as a GPU
table/kernel realization of stack-aware prior art, not as a new discovery that
LR reductions depend on the exposed stack state.

### gpu-lr1 versus XGrammar and llguidance

XGrammar and llguidance have broader grammar and JSON Schema support, mature
error handling, and serving integrations. gpu-lr1's measured advantage is in a
narrower execution regime:

- grammar tables are already compiled and resident;
- active state is a device-side integer;
- the batch may contain many different schemas;
- token selection and state update are fused;
- no recursive or dynamic schema semantics are required.

Any comparison must report both coverage and speed. A faster engine that
silently ignores schema keywords is not a valid replacement.

The repository now includes a controlled Qwen3 comparison with XGrammar:

- identical 151,669 token IDs and full-width logits;
- two finite-depth grammars compiled independently by both engines;
- identical concrete prefixes/configurations;
- every timed mask checked bit-for-bit for equality;
- fastest XGrammar thread count selected per batch;
- a stronger-than-native XGrammar baseline using fused bitset argmax and a
  captured pinned-H2D-copy CUDA Graph.

Under that isolated constraint-step boundary, gpu-lr1's graphed path is 2.0x
faster at batch 1 and 863.5x at batch 2,048 than the optimistic XGrammar path.
Including XGrammar token acceptance and rollback changes the range to 3.9x
through 1,468.9x. The non-graphed gpu-lr1 plan is slower at batch 1 and wins
from batch 8 onward.

The trade-off reverses at compilation: gpu-lr1 takes 0.856 s versus 0.0032 s for
XGrammar and uses about 292 KiB of runtime tables versus about 52 KiB of
XGrammar compiler cache. Model execution is excluded, and XGrammar can overlap
CPU matching with the model, so these numbers are not end-to-end TPOT speedups.

## Defensible claims

- gpu-lr1 demonstrates that a useful acyclic JSON Schema subset can execute
  without a per-token CPU grammar dependency.
- gpu-lr1 also demonstrates sparse canonical LR(1) ACTION/GOTO execution,
  heterogeneous global table relocation, and ragged bounded stacks on GPU over a
  terminal stream.
- Flat global state IDs avoid schema padding and device pointer chasing in a
  heterogeneous batch.
- Dense `M_next[state, token]` is memory-prohibitive at realistic state counts;
  sparse edges or a compact byte-DFA transition are practical alternatives.
- CSR is preferable for sparse rows, while packed bitsets/dense scans win as
  mask density grows; a hybrid sampler is necessary.
- The measured benefit grows with heterogeneous batch size in the current
  XGrammar comparison.

## Claims to avoid

- "gpu-lr1 is the first GPU-native grammar-constrained decoder."
  Gram2Token is direct prior art at the claim level.
- "gpu-lr1 provides unbounded, general-lexer LR(1) constrained decoding."
  Tokenizer bytes are currently supported for byte-terminal grammars, or through
  explicit terminal sequences, and bounded configuration expansion can time out
  on branching grammars.
- "All production engines synchronize GPU to CPU every token."
  The common direction is CPU-to-GPU mask staging and scheduling dependence;
  implementations may overlap work and avoid an explicit blocking device read.
- "JSON, SQL, and most programming languages are solved by one
  stack-independent transition table."
  LR reductions expose stack-dependent `goto` state.
- "A kernel-level speedup implies the same end-to-end TPOT improvement."
  Full model execution, scheduler behavior, compilation, cache hits, and tail
  latency must be measured separately.
- "Compiled successfully" means "implements the schema exactly."
  JSONSchemaBench documents both under-constraint and over-constraint in
  existing engines.
