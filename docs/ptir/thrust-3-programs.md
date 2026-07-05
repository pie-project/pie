<!-- Source of truth: wiki page `tensor-ir-plan.md` (slug tensor-ir-log). This folder is the split, on-disk copy for implementation teams. -->

# Thrust 3 — PTIR Programs: Trace, Channels, Compilation

**Status:** Ready for execution. Independent of thrusts 1 and 2 (see §6).
**Audience:** SDK (tracing frontend), runtime (channel store, registration),
driver/CUDA (interpreter, JIT, graphs) engineers.
**Realizes:** [`overview.md`](overview.md) §1–§4 (values, channels, programs,
intrinsics), §5.3 (attachment points), §7.1 (epoch rings), §7.3 (compilation),
and the appendix (the op set). Contract provided: **C3 — program identity is a
hash** ([`masterplan.md`](masterplan.md) §3).

---

## 1. Goal

Turn the shipped Sampling-IR path — one JIT'd program at one stage — into the
general PTIR substrate:

- **Programs**: closures traced once into graphs over the closed first-party
  op set (overview appendix) plus named intrinsics; `if`/`for` resolve at
  trace time; a different branch is a different program (batch-by-program).
- **Channels**: the only stateful construct — GPU-resident cells with
  full/empty bits, SPSC, capacity trace-known. Loop-carried state
  (`tok`/`len`/`rng`) never round-trips to the host; ordering comes from the
  bits.
- **Execution rules** (overview §1): per-stage readiness, dummy-run on miss,
  pass-atomic per instance, resubmission, no cancellation, poison on fault or
  deadline (engine policy).
- **Commit** (overview §7.1): epoch rings, predicated index bump at pass end,
  in-place lowering for qualifying channels, within-pass register semantics.
- **Compilation** (overview §7.3): tier 0 (op-per-kernel interpreter), tier 1
  (per-stage JIT fusion — the Sampling-IR pipeline generalized), tier 2
  (launch-erased graphs keyed by the stage tuple).
- **Attachment points** (overview §5.3): `prologue`, `on_attn_proj`,
  `attn_mask`, `on_attn` (per layer), `epilogue`; configuration sinks with
  stage-precedence checking; batching identity = the tuple of stage traces.

Standalone value: **epilogue parity first.** The first shippable artifact is a
drop-in replacement for the Sampling-IR epilogue (same workloads, same or
better tokens/s) carried by the new trace/channel machinery — usable under
today's scheduler and today's working set.

## 2. Current state

| area | today | anchor |
|---|---|---|
| JIT | NVRTC-compiled sampling epilogues; codegen, jit backend, primitive sources, kernel cache keyed by FNV-1a program hash; thread-pool compile | `driver/cuda/src/sampling_ir/{codegen,jit,jit_backend,runtime,primitives_src}.{cpp,hpp}` |
| epilogue ABI | `sampling_program_{indptr,bytes,bytes_indptr}` (program blobs per request), `sampling_input_*`, `sampling_binding_*` (incl. `SamplingBinding::Tensor{key}`), `sampling_late_*` value tables | `interface/driver/src/schema.rs` (:229–:277) |
| executor | sampling deliberately **outside** CUDA graphs; per-program partitioning of sampling rows within a batch; retained next-input buffer (proto-channel); Late-input buffers bound into graphs | `driver/cuda/src/executor/executor.cpp` (:2117, :2889, :3205, :387–:409) |
| graphs | `ForwardGraphKey{num_requests, num_tokens, variant}`, pure-decode capture | `driver/cuda/src/executor/forward_graph.hpp` (:46) |
| design lineage | Sampling-IR spec (P1–P5: ops are data; expressiveness bounded by fusion-ability) — declared "PTIR at `stage = lm-head`"; predecessor PTIR sketch | [`../docs/sampling-ir.md`](../docs/sampling-ir.md), [`../docs/tensor-ir.md`](../docs/tensor-ir.md) (superseded by `overview.md` on conflict) |
| SDK | Rust generation loop with spec/draft handling; Python P3-native async wrappers; no tracing frontend | `sdk/rust/inferlet/src/{generation,spec,forward}.rs`, `sdk/python/src/inferlet/` |

The Sampling-IR pipeline is the seed: this thrust generalizes its **program
carrier** (bytes + hash + cache), its **binding tables** (inputs, late keys,
device aliases), and its **executor integration** (row partitioning,
outside-graph placement) from one stage to all stages.

## 3. Locked design decisions

| # | decision | source |
|---|---|---|
| T1 | Traced once; `if`/`for` at trace time; `select` for data-dependent choice; a genuinely different branch is a different traced program. An **instance** is one binding of a program to its channels and working set: trace = identity, instance = state | overview §2 |
| T2 | Channels are the only stateful construct: bounded queues of cells with full/empty bits; capacity trace-known; **SPSC** with endpoints fixed at bind time (second endpoint = bind error); pairs may span pipelines | overview §1 |
| T3 | Execution rules: readiness per stage as each program starts (`take`/`read` need full; a channel whose *first* op is `put` needs empty); miss ⇒ dummy values, batch stays uniform; pass atomic per instance; runtime resubmits in submission order; no cancellation; faults/deadline poison every touched channel; deadline is engine policy | overview §1 |
| T4 | Commit = epoch rings: reads from committed cell, puts to pending cell, predicated index bump at pass end; fire-time structural predicate + on-device per-stage AND; in-place lowering (no undo after the last fallible stage; mutation-granularity undo before it); host-visible channels always keep the full ring; **within a pass a channel is a register** (double put = last wins) | overview §7.1 |
| T5 | Batching identity is the tuple of stage traces; instances co-batch stage by stage — a shared forward batches even where epilogues differ; per-program glue partitions the row space | overview §5.3 |
| T6 | Intrinsics split by contract owner: first-party (`intrinsics::*`, spec-owned, some model-gated, checked at bind) vs second-party (`intrinsics::kernel::*`, backend-owned, calling convention fixed, availability at bind, fallback = a different traced program). No intrinsic may return a time- or load-varying value | overview §4, §1 |
| T7 | The first-party op set is **closed** per the overview appendix; extension lands as named second-party ops (D5), promotion is a spec event | overview §0, §4, appendix |
| T8 | Replay determinism: `rng` noise is a pure function of `(key, ctr)`; `pivot_threshold` determinism is spec semantics | overview §3, §4 |
| T9 | The trunk is never expressed in PTIR — the descriptor binds it. Tiers 0/1/2; `top_k`, `matmul` and kin are library kernels; the JIT emits glue, never algorithms | overview §7.3 |
| T10 | `Register` is reserved, not implemented. Any proposed API that is a latest-value read (unpaired, non-replayable) is rejected or redesigned channel-shaped | overview §1 |
| T11 | Configuration sinks are ordinary ops with stage-precedence checking; pass-wide sinks are prologue-only; `attn_page_mask` at prologue = all layers, at attn-proj = that layer. Every pass contains a forward; no bare pass | overview §4, §5.3 |
| T12 | Programs never write pages; there is no memory op in the IR | overview §4, §5.2 |

## 4. Boundaries

- **Kernels behind second-party intrinsics that touch attention/KV**
  (`attn_page_mask`, `envelope_dot`, `gather_tokens`, the masked-attention
  variant) are thrust 1 (its M2b/M3). This thrust defines the intrinsic
  surface and binds kernels **when the backend provides them** — bind-time
  availability (overview §4) is the contract, and `attn_page_mask` is
  direction-only under the no-attention-kernel constraint (thrust 1 W11).
  §6.1's Quest tap is therefore optional, never a P-phase gate.
- **Retry scheduling** for resubmission is thrust 2 (a missed instance is
  "membership in a later fire"). This thrust defines the per-instance
  semantics (T3/T4) and emits the readiness words (C2's final producer).
- **Depth/pipelining**: every P phase below is testable with a *synchronous*
  submit loop (degenerate depth 0) — channels still order everything; the
  host just blocks more. Thrust 2 removes the blocking, not the semantics.
- **Metal** is a follow-on backend, not in scope: tier 0's op library is the
  entire porting surface (overview §7.3); nothing here may assume CUDA beyond
  the tier-1/2 modules.
- Multi-model channels, `Register`, spliced modalities: out of scope
  (recorded as direction in the overview).

## 5. Phases

### Phase P0 — Spec freeze and validation suite

Make the overview mechanically checkable before writing the runtime.

1. **Op table module** (shared Rust crate + C++ header, generated from one
   source of truth): op ids, arities, shape/dtype inference rules for every
   family in the overview appendix; the composed ops (`gumbel`, `mask_apply`,
   `softmax`, …) expressed as expansions over the core.
2. **Trace container:** extend the `sampling_program_bytes` encoding to carry
   stage-tagged programs (prologue / attn-proj / attn / epilogue), channel
   declarations (shape, dtype, capacity, seed), descriptor-port bindings, and
   sink calls. Program hash = FNV-1a over the canonical encoding (the existing
   cache key discipline).
3. **Validator:** SPSC endpoint check; per-channel program order; first-op
   direction table per channel (the readiness predicate input, T3); sink
   stage-precedence (T11); shape closure (all shapes trace-known; per-page
   shapes bound to caps, W10); model-gated intrinsic availability (bind-time);
   the T10 lint (no time-varying intrinsics).
4. **Golden vectors:** small traces + expected validator verdicts + expected
   tier-0 results; doubles as the conformance suite for any future backend.

Exit: overview §3's example and §6.2's epilogue serialize, validate, and
hash stably; every appendix family has shape-rule tests.

### Phase P1 — SDK tracing frontend (Rust first)

1. `Tensor`/`Channel`/`ForwardPass`/`Pipeline` builder types whose method
   surface matches the overview's examples **verbatim** (the overview is the
   API doc; deviations require an overview PR).
2. Trace-once memoization: closure traced on first submit; cache keyed by
   trace-affecting constants; per-instance data (seeds, slot ids) must flow
   through channels or data args, and the tracer rejects trace-time capture of
   per-instance values it can detect (D2 lint).
3. Trace-time errors surface with source spans: readiness-direction
   conflicts, double-endpoint, sink misplacement.
4. Python frontend after Rust parity (same trace bytes, shared validator via
   FFI or subprocess).

Exit: overview §3 and §6.2 examples compile to validated traces from the Rust
SDK; error-message snapshot tests for the lint set.

### Phase P2 — WIT + runtime carrier

1. WIT (append-only, behind the `ptir` flag): `channel` resource
   (`new(shape, dtype, capacity)`, `from(seed)`, async `put`/`take`/`read`),
   `program` registration (`register(bytes) -> program-id`, hash-deduped),
   `pipeline` (`submit(fwd)`, `close`), stage attachment on the forward pass
   (`stage(kind, program-id)`), and descriptor ports accepting channel
   references (C1 shapes).
2. Runtime: program registry (validate once, compile per backend tier, cache
   by hash); instance construction (allocate channel cells from declared
   shapes; seed `Channel::from` values); submit path carries (program-set,
   channel bindings, descriptor bindings) through the existing request path.
3. Registration-time pricing: rows, readout counts, channel bytes computed
   once per program and attached to the identity (feeds thrust 2's capacity
   accounting through C3's opaque identity).

Exit: register/instantiate/submit round-trips on the mock driver; duplicate
registration is a cache hit; malformed traces fail at bind with the P0
validator's message.

### Phase P3 — Channel store and host ops

1. Device layout: per-instance arena — cells at
   `instance base + channel offset + ring index`, bits packed in a word per
   instance (the word C2's wait/predicate reads). Capacity-1 = double buffer;
   capacity-N = ring of N+1 with head/tail (overview §7.1).
2. Host `put`/`take`/`read`: async, over the direct host→driver channel
   (thrust 2's S3b when available; the existing `copy_d2d`/`notify` path
   until then). All `put`s before a submit coalesce into one transfer (D1;
   overview §6.4 relies on it). `bool` channels travel packed (D1).
3. Poison: fault/deadline marks every touched channel; blocked host ops
   resolve to errors (`out.take().await?`); teardown reclaims parked
   instances (`close` semantics, overview §1).

Exit: host↔device ping-pong through a channel on mock and CUDA; back-pressure
observed (capacity-1 `out` blocks the producer pass, not the host thread);
poison propagates to a parked host `take`.

### Phase P4 — Tier-0 execution

1. **Reference interpreter** (host, runs under the mock driver): executes
   validated traces cell-accurately — the golden model every backend diffs
   against. Implements T3/T4 exactly: per-stage readiness, dummy values,
   pass-atomic effects, ring commit, poison.
2. **CUDA tier 0:** one prebuilt kernel per core op (the appendix families —
   a few dozen row-parallel kernels); stage runner walks the trace launch by
   launch; per-stage readiness check kernel reads the bits word; effects
   predicated on the accumulated commit flag; end-of-pass predicated bump
   kernel (fused later, T4).
3. **In-place lowering pass:** classify channels (host-visible ⇒ full ring;
   device-private linear `take`→`put` after the last fallible stage ⇒
   in-place; before it ⇒ row-granularity undo), per overview §7.1. The
   classification is computed at registration from attachment points and the
   fallible-stage analysis.
4. **Resubmission protocol:** a missed instance's pass reports "no commit";
   the runtime re-enqueues it in submission order (under today's scheduler: a
   retry loop; under thrust 2: later-fire membership).

Exit: golden-vector parity between reference interpreter and CUDA tier 0;
overview §3 runs end to end on CUDA tier 0 (greedy + grammar mask, dummy-run
on late mask observed and recovered); replay determinism test — same seeds,
same tokens, across runs and across tier changes (T8).

### Phase P5 — Tier-1 JIT fusion

1. Generalize `sampling_ir` codegen from "the epilogue" to "a stage program":
   one fused kernel per stage per program — a lane per CTA, reductions
   row-local, geometry scatters composed flat (overview §6.2's epilogue is
   the canonical stress case: gathers + two flat scatters + top-k).
2. Library kernels: per-row `top_k` (bitonic for beam widths), `matmul`
   (cuBLASLt or a hand kernel for the `[k,d]×[d,L]` contrastive shape) —
   linked, not generated (T9).
3. Fuse the readiness check into the kernel prologue and the commit bump into
   the epilogue of each stage's last kernel (overview §7.1).
4. Cache compiled kernels by program hash (existing FNV discipline); compile
   asynchronously at registration with tier-0 fallback until warm (the
   existing thread-pool compile pattern).

Exit: tier-1 == tier-0 on the golden vectors; epilogue parity gate — a
grammar-constrained decode via PTIR tier 1 within 5% tokens/s of the
Sampling-IR path (masterplan M1 demo); §6.2's epilogue fuses into ≤ 2 kernels.

### Phase P6 — Tier-2 graph capture

1. Extend the graph key with the **program-set hash** (C3) beside
   `{num_requests, num_tokens, variant}`.
2. Capture policy: trunk in-graph (as today); tier-1 glue kernels in-graph
   when all their channel inputs are device-resident; host-fed stages stay
   outside capture exactly like sampling today (`executor.cpp:2889`) until
   the C2 wait-word is provably capture-safe (`cuStreamWaitValue32` inside
   graphs — evaluate, don't assume).
3. LRU eviction and capture-on-Nth-hit, as today; probe key cardinality.

Exit: pure-decode PTIR fleet shows launch-overhead parity with today's graph
path; key-cardinality probe within budget on a mixed 4-program fleet.

### Phase P7 — Attachment-point rollout

Order chosen so each step ships alone:

1. **Epilogue** (done by P5's parity gate).
2. **Prologue + configuration sinks:** sink calling convention and
   stage-precedence enforcement (T11); `intrinsics::kernel::lora` (grouped
   GEMM over batch-gathered weights — second-party kernel, overview §6.5) and
   `minference_sparse` as the first two sinks; weight-update-variant pattern
   (RCU-style pair publish via per-instance atomic commit, overview §6.5)
   as a conformance test.
3. **Per-layer taps** (`on_attn_proj`, `on_attn`): `query()` and
   `intrinsics::layer` in scope; one readiness check covers all invocations;
   the accumulation idiom (`take`→`scatter_set(layer, ·)`→`put`) lowers to an
   in-place row write (overview §5.3). Capture-safety review before allowing
   taps inside tier-2 regions. `attn_page_mask` binds only if the backend
   ever provides it (bind-time availability; direction-only under thrust 1's
   W11 constraint).
4. **Descriptor ports as channels:** `attn_working_set` arities, `readout`,
   `positions` bound to channels (C1 final form; needs thrust 1's M5 for
   device-resident reads — until then, host-fed channels only).

Exit per step: the corresponding overview example fragment runs (LoRA §6.5
batching test: B instances, B adapters, one pass; SnapKV §4 idiom on the
per-layer tap).

### Phase P8 — Conformance and north-star examples

- Semantics suite: readiness/dummy-run/resubmission, ring commit under
  double-put (register rule), poison and `close`, SPSC bind errors,
  per-channel program order (the §6.1 `cursor` read-before-take case).
- Overview §6.1 end to end with thrusts 1+2 (masterplan M3); §6.2 with M4's
  compact; §6.4 MCTS host-loop pattern on the mock driver.
- Perf ledger: dummy-run rate, JIT compile latency at registration, channel
  arena bytes per instance, graph-key count — published per release.

## 6. Interfaces

**Provides:** C3 (the program-set hash as opaque batching identity, plus
per-program registration-time pricing); the readiness bit words that become
C2's final producer; the second-party intrinsic registry thrust 1's kernels
plug into.

**Consumes:** C1 for channel-fed descriptors (P7.4 only — earlier phases use
host-fed channels and today's descriptor); C2's wait mechanism when thrust 2
lands (until then the synchronous loop blocks on the host, which is correct,
just slower). No phase before P7.4 depends on another thrust.

## 7. Risks

| risk | mitigation |
|---|---|
| Resubmission re-runs the trunk on a readiness miss | Structural readiness (thrust 2 F5) makes misses late-host-edge-only; measure dummy-run rate (M3 gate < 1%); stage-level resubmission recorded as a later optimization, not v1 |
| Registration-time JIT latency visible to first submit | Async compile + tier-0 fallback (P5.4); hash cache persists across instances |
| Channel arena footprint (e.g. vocab-sized mask cells × capacity × instances) | Shapes are trace-known ⇒ exact accounting at registration (P2.3); packed `bool` wire (D1); capacity defaults to 1 |
| Graph-key explosion (program-set × shape buckets) | Bucket shapes as today; LRU; capture only proven-hot tuples (P6.3) |
| `cuStreamWaitValue32` inside captured graphs is backend-fragile | Keep host-fed stages outside capture (today's sampling placement) until proven; tier-1 fusion already removed most launch overhead |
| Two tracing frontends drift (Rust/Python) | One trace container + one validator (P0); Python emits the same bytes and reuses the validator |

## 8. Open questions

1. Trace container: extend `sampling_program_bytes` in place vs. a new
   versioned blob with a Sampling-IR compatibility reader — decide in P0 with
   the thrust-2 S1 merge work in view (same tables get CSR-merged).
2. Stage-level (vs pass-level) resubmission after a late-mask miss: worth the
   complexity once dummy-run rates are known? (Direction only.)
3. Channel observability: a debug `peek` tooling surface (host-side, read-only
   snapshots) — needed for support, must not become a `Register` back door
   (T10).
4. `rank_le`/`pivot_threshold` numeric contract across backends (ties,
   NaN ordering) — pin in P0's op table.
5. Per-site attachment stages for novel PEFT algebras (overview §6.5's
   honest cost) — direction; revisit after LoRA ships.

## 9. Code anchors

| area | files |
|---|---|
| JIT seed | `driver/cuda/src/sampling_ir/{codegen,jit,jit_backend,runtime,primitives_src,thread_pool}.{cpp,hpp}` |
| Program/binding ABI | `interface/driver/src/schema.rs` (`sampling_program_*`, `sampling_binding_*`, `sampling_late_*`) |
| Executor integration | `driver/cuda/src/executor/executor.cpp` (row partitioning :2117, sampling outside graphs :2889/:3205, retained next-input :387), `driver/cuda/src/executor/forward_graph.hpp`, `graph_variant.hpp`, `persistent_inputs.{cpp,hpp}` |
| SDK | `sdk/rust/inferlet/src/`, `sdk/python/src/inferlet/`, `sdk/tools/bakery/` |
| WIT | `interface/inferlet/core/wit/inference.wit` (+ new `ptir` surfaces) |
| Direct host→driver path | `runtime/src/driver/ops.rs`, `runtime/src/driver/channel.rs` |
| Design lineage | [`overview.md`](overview.md) (normative), [`../docs/sampling-ir.md`](../docs/sampling-ir.md), [`../docs/tensor-ir.md`](../docs/tensor-ir.md) |
