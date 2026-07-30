# Polymorphic batching in Pie: a plan

Date: 2026-07-29
Baseline: `dev` @ `8435eb26`. Every file:line below was verified on that commit;
the line numbers move, the claims should not.

This proposes two changes that are worth doing separately and are much cheaper
together:

1. **`model/`** — a declarative toolchain for the forward pass, so a model is a
   declaration instead of 35.5k lines of hand-written C++ per driver.
2. **Polymorphic fires** — let one fire hold requests that do not agree on what
   to compute, planned in `runtime/engine`.

The second is the interesting one. The first is what makes it affordable.

---

## 1. Pie already implements polymorphic batching. Twice.

This is not a proposal to add a mechanism. The mechanism is here, hand-written,
in two places, and Pie is the only engine that has named the principle:

> *"A program is a closure whose effects are channel `put`/`take`s; `if`/`for`
> resolve at trace time; a different branch is a different program
> (**batch-by-program**)."*
> — `compiler/dsl/src/lib.rs:4-6`

- `ptir_program_hashes` is a `Slice`: **one fire holds N different PTIR
  programs**, each owning a disjoint row range
  (`driver/abi/include/pie_native/launch_view.hpp:72-85`).
- `driver/cuda/src/pipeline/grouped_runtime.cuh` fans one launch across lanes,
  one lane per program. That is gather → per-lane work → scatter.
- `driver/cuda/src/kernels/moe_grouped_gemm.cu` is the same pattern written a
  second time, for the expert axis.
- `compiler/plan/src/lane_table.rs` already carries per-lane symbolic extents
  (`kv_len`, `row_count`, `token_count`, `query_len`) plus `active_row_mask` and
  `row_valid`.

So the vocabulary exists. What it does not do is cross the model boundary.

---

## 2. Where it stops

### 2.1 The forward pass is a cascade of fast paths that divergence switches off

```cpp
const bool fused_decode_qkv_post =
    use_fused_qkv &&
    active_stage_hooks == nullptr &&      // <-- any hook, anywhere, disables this
    decode_fused_post_enabled() &&
    is_pure_decode &&
    !has_custom_mask &&
    ...
```
— `driver/cuda/src/model/llama_like/llama_like.cpp:492-503`

**112 predicates of this shape across 23 files** in `driver/cuda/src/model/`
(grep for `active_stage_hooks == nullptr|!has_custom_mask|is_pure_decode`).

Each one is the same rule: *a fused edge cannot be a merge point.* Pie
discovered it empirically and encoded it as a boolean. It is correct. What is
missing is that the boolean is all-or-nothing.

### 2.2 The hook pointer is a global

```cpp
inline thread_local const StageHooks* active_stage_hooks = nullptr;
```
— `driver/cuda/src/model/stage_hooks.hpp:55`

**One pointer per fire.** A single request that wants an attention-score tap
disables the fused QKV kernel for every request in the batch. This is the
polymorphism penalty in its purest form, and it is a hidden parameter, not a
kernel limitation.

### 2.3 Multi-row programs cannot co-batch at all

`preserves_inner_rows()` is `wire_row_count() > 1`, and that forces
`requires_solo_submission` (`runtime/engine/src/scheduler/worker.rs:322-345`).
The driver enforces a second rule: *"ptir: dense device mask in a multi-program
batch requires solo retry"* (`driver/cuda/src/batch/frame.cpp:935-943`).

So a KV-sparsity program like `tests/inferlets/quest-attention` — which writes a
per-lane `attn_page_mask` — cannot share a fire with an ordinary request.

### 2.4 Nothing about the weights may vary

- *"the engine serves exactly one model"* — `runtime/model/src/lib.rs:97`
- `IModel::body(ws, kv, attn_ws, cublas, in)` takes no per-request weight, rank
  or depth (`driver/cuda/src/model/imodel.hpp:79-83`)
- LoRA does not exist. `("lora", SinkScope::PassWide)` is **reserved and
  unimplemented** (`compiler/ir/src/registry.rs:152`); PEFT adapters are on the
  roadmap for v0.3.0.

### 2.5 PTIR is one axis, and it is the cheapest one

PTIR is not the forward pass. It is four fixed stages — prologue, on_attn_proj,
on_attn, epilogue — whose programs run around a backbone that stays hardcoded
(`ptir-refactor.md:845-854` puts `src/model/` out of scope, *"because it has
nothing to do with PTIR"*). "Which PTIR program" is one polymorphism axis among
several, and post-logit divergence is the axis that costs least to serve.

---

## 3. What the mechanism is worth, including where it is worth nothing

Numbers below are measured on an RTX 3090 (sm_86) against a Qwen2.5-class model
in a separate prototype. They are quoted because they *price the decisions*, not
because they will reproduce on Pie unchanged. Re-measure before trusting any of
them on this codebase.

### 3.1 Splitting is the wrong baseline. Padding is the right one.

Given a batch it cannot express, the move an engineer actually makes is to
coerce every request into the batch's most general shape and run one suboptimal
kernel — paying wasted arithmetic to pay the weight traffic once. That is what
vLLM does with `max_lora_rank`, and it is a far stronger baseline than splitting.

Against padding, at 8 programs:

| axis | vs split | vs padded, thin | vs padded, wide |
|---|---|---|---|
| depth | 6.8× | **1.03×** | 1.20× |
| qlen | 7.5× | **1.14×** | 1.69× |
| rank (adapters) | 3.6× | **0.56×** | 0.93× |
| vision | 2.0× | **1.00×** | 1.04× |
| sampler | 1.2–5.5× | **1.02×** | 1.38× |
| MoE | — | 1.21× | **3.96×** |
| mixture-of-depths | — | **4.09×** | 1.72× |

Read the middle column. **On rank, padding beats exact per-rank bucketing
outright.** Do not build the bucketing path; store adapters at max rank.

The gap is not our bookkeeping: stripping every gather, scatter, clone and mask
from the padded path moves it by 0–6%, and merging still wins 1.20× (depth),
1.74× (qlen), 3.59× (MoE), 4.0× (MoD). It is not an occupancy artefact either —
at 12.5% kept the dense path does 8× the row-work in 4.0× the time, so it is
using the machine *better per row* and loses anyway.

**The rule this gives:** padding is free when it inflates a dimension the machine
was not being paid for (at one row per program a decoder layer is weight-traffic
bound, so a wider batch costs nothing), and ruinous when the pad is a multiple of
the layer itself. The second set is exactly the set of axes someone has had to
hand-write a kernel for. MoD, whose kernel nobody wrote, is the largest gap.

### 3.2 Composing axes does not compound the advantage

The natural prediction — pad factors multiply, so merging pulls ahead as axes
compose — is **wrong**, twice:

| axes live | rows | padded | merged | **planned** | vs padded | pad waste |
|---|---|---|---|---|---|---|
| depth | 8 | 6.10 | 5.74 | **5.73** | 1.06× | 1.26× |
| × rank | 8 | 6.63 | **10.08** | **6.17** | 1.07× | 2.68× |
| × rank × vision | 8 | 6.67 | **9.87** | **6.34** | 1.05× | 2.68× |
| × rank × vision | 512 | 28.7 | 24.8 | **23.0** | 1.25× | 2.68× |

Pad waste doubles and the measured advantage does not move. And pure merging
gets *worse* when rank joins depth, because it inherits the one axis merging
loses. What survives is the **planned** column — merged on depth, padded on rank
— which is never worse than either pure strategy and beats pure merging by 1.5×.

**This is the argument for a planner, and it is not "merging wins".** It is that
with three axes live the right answer is a different lowering per axis, which is
not something a point solution can express.

### 3.3 The three classes, and why the DSL is asymmetric

| class | what it is | cost |
|---|---|---|
| CORRECTION | folds into an additive fix on a materialized output | 1.01× the no-divergence floor |
| WEIGHT | same operator, per-program weights | one batched GEMM, no branch |
| STRUCTURAL | genuinely different operators | fused region must split: flat ~7%, vs 1.87× for not merging at all (K=32) |

Cost order matches which axes engines already batch. CORRECTION and WEIGHT
commute with fusion, so they were cheap enough to hand-write; STRUCTURAL forces a
fused region to split, and nobody batches those.

### 3.4 Conditional graph nodes: enumerate variants, not combinations

Measured on CUDA 12.8:

```
SWITCH with  8 bodies, 1 taken:   9.6 us
SWITCH with 16 bodies, 1 taken:   9.8 us
SWITCH with 32 bodies, 1 taken:   9.8 us      <- constant in body count
```

So enumerating variants is free **if they are mutually exclusive**. Independent
IF nodes are not:

```
K=32,  0 live:  22.3 us      <- you pay for what you enumerated
K=32, 32 live:  28.0 us
32 separate graph launches:  99.5 us   <- the alternative
```

Idle overhead is 0.66–1.3 µs per enumerated-but-absent subgraph, and a body needs
roughly 250 µs of work before the fixed cost falls under 10%. **Conditionals
belong at layer granularity, and only for STRUCTURAL divergence.** CORRECTION and
WEIGHT need no graph node at all — the variant is a pointer, not a branch.

Consequence for `ForwardGraphKey{num_requests, num_tokens, variant}`
(`driver/cuda/src/batch/forward_graph.hpp:49-59`, LRU capped at 128): with
polymorphism in the backbone, either the key multiplies by the program set (it
will not fit) or topology is fixed with conditional regions and the key gains
only a supergraph generation number. The second is the answer, but note that
capture bakes grid sizes — so row-count variation inside a graph must be padded
or conditionalized. Pie already pads `R` to a lattice for exactly this reason
(`forward_graph.hpp:60-85`).

---

## 4. Structure

```
model/                    (new top level; absorbs the existing loader/)
  schema/    weight names, shapes, dtypes, config facts       <- dependency floor
  loader/    schema + checkpoint  -> device memory            (contracts stay in drivers)
  forward/   schema + declaration -> executable forward pass  (specs stay in drivers)
compiler/                 UNTOUCHED. PTIR only.
runtime/engine/scheduler/ fire planning + row order
driver/cuda/              prepare() materializes, body() reads, capture
```

### 4.1 Why `model/` nests

`compiler/` nests because its crates form a DAG with `ir` as the floor. `model/`
earns the same shape only if `schema` is genuinely shared — and it is. A forward
declaration that says `layer[l].qkv: [hidden, qkv_out]` *is* the shape half of the
contract; the family supplies the checkpoint names. Today that agreement is an
implicit contract between the `Qwen3Weights` struct and `bind_llama_like()`.
Lifting it to a schema turns a class of mismatch bugs into type errors.

**If `schema` does not end up shared, do not nest.** Flat is better than a
directory that only expresses a theme.

### 4.2 Why the specs live in drivers

Precedent, stated explicitly in `driver/cuda/src/model/contract.hpp`:

> *"This header is **the mechanism and nothing else**. The knowledge — that Phi-3
> ships a fused QKV, that Kimi hides the decoder under `language_model.` — lives
> in each family's own directory next to its forward pass."*
>
> *"**It is CUDA's.** Metal authors its own contract... A header shared between
> them made each carry the other's vocabulary."*

`model/forward` is the mechanism; `driver/cuda/src/model/qwen3_6/` holds the
declaration, the contract, and the kernel selection. Same relationship
`compiler/` has to inferlets and `loader/` has to contracts: **a toolchain is
named for its activity, and its inputs live elsewhere.**

One nuance worth keeping in view: layer order and layer types are facts about the
*checkpoint*, not about the driver. If the declaration names *operations* rather
than *kernels*, the architecture half is driver-neutral and could be promoted
later. Do not design for that reuse up front; promote it when a second backend
asks.

### 4.3 Why the planner lives in `runtime/engine`

Planning a fire needs four things, and only the scheduler has all four:

1. the model's divergence sites → `model/forward`
2. which PTIR programs are attached and what they need → `compiler/`
3. the device cost model → driver capabilities
4. **what was admitted** → the scheduler itself

There is precedent: `runtime/engine/src/pipeline/program.rs` already orchestrates
`pie_plan::compile_bound` and `pie_codegen::launch::build`. And
`LaunchGrouping::accepts` — *"can these co-batch"* — already lives at
`scheduler/worker.rs:382-431`. *"How do they co-batch"* belongs next to it.

```
runtime/engine/src/scheduler/
    worker.rs      accepts()   — today: force solo; after: admit when a plan exists
    fire_plan.rs   (new)       — per-site lowering, row order, region mask
```

### 4.4 Emit alternatives, not decisions

`compiler/plan` already produces **two** partitions per stage — `singleton` (the
always-correct fallback) and `fused` — and lets the backend choose. Fire planning
should do the same: `model/forward` and the plan stay device-independent and emit
candidate lowerings; the runtime picks using device cost. This keeps
`compiler/codegen`'s rule (*"a pure `Plan -> String` function with no
device-architecture input"*) intact for the new toolchain too.

Class assignment (CORRECTION / WEIGHT / STRUCTURAL) is structural and stays
device-independent. Only the choice among candidates is device-dependent.

### 4.5 Naming consequences

- `runtime/model` (*"Model Service — model registration and tokenizer
  management"*, plus instruct templates and multimodal host-decode) **must be
  renamed** — `runtime/catalog` or similar. It is not the model.
- Optional, and independent: `compiler/` → `program/`, which makes the rule
  visible (`program/` for programs that live in inferlets, `model/` for models
  that live in drivers). Costs a rename of a two-day-old refactor; skip it if
  that is not worth the churn.

---

## 5. The declaration

### 5.1 Principles

**The DSL expresses what varies, never how to lower it.** We measured exact
per-rank bucketing losing to padding at 0.56×. If authors could write the
lowering, they would write that one. Fusion is the same: a fused edge cannot be a
merge point, so fusion and merging must be chosen together, by the planner.

**Two of the three classes need no syntax at all.**

- **WEIGHT** comes from binding. The author writes `matmul(x, layer[l].qkv)`. If
  the deployment binds 32 of them, the compiler sees a lane dimension and emits a
  batched GEMM. `ops::gemm_batched_act_x_w` already exists
  (`driver/cuda/src/ops/gemm.hpp:202`).
- **CORRECTION** comes from algebra. If the binding says "qkv is base plus a
  per-lane low-rank delta", the compiler distributes:
  `x(W + BA)ᵀ = xWᵀ + (xAᵀ)Bᵀ` — one shared GEMM plus a small batched GEMM at
  `beta=1`. With no adapters the term vanishes and the code is the original.
- **STRUCTURAL** needs syntax, because it changes control flow.

The asymmetry is the point: **syntax is required exactly where cost is
incurred.** No marker means free.

**Static and dynamic are Rust's `impl` and `dyn`.** Layer kind, rope variant,
qk-norm — all resolve at trace time, which for a tracing eDSL is *model load
time*. That is strictly stronger than `rustc` monomorphization here, because the
variation is driven by `config.json`, which `rustc` never sees. Only `dyn` values
reach the fire planner.

### 5.2 Qwen3.6 (hybrid GDN + full attention, dense and MoE variants)

```rust
model! { qwen3_6

  layers = cfg.layer_types;                     // [Full, Linear, Linear, ...]  static
  depth  : dyn PerRequest<u32>;                 // the only structural syntax here

  weights {
      embed, final_norm, lm_head;
      layer[l] {
          attn_norm, mlp_norm, o_proj, in_proj;
          Full   => { q_norm, k_norm },
          Linear => { conv, a_log, dt_bias, gate_norm },
          Dense  => { gate_up, down },
          Moe    => { router, expert[e]{ gate_up, down } },
      }
  }

  forward(tok, h: dyn Hooks) -> Token {
      h.prologue();
      let mut y = embed[tok];
      for l in 0..depth {
          let x = rmsnorm(y, layer[l].attn_norm, variant: gemma);
          y += match layers[l] { Full => full_attn(l, x, h), Linear => gdn(l, x, h) };
          y += mlp(l, rmsnorm(y, layer[l].mlp_norm, variant: gemma));
      }
      h.epilogue(matmul(rmsnorm(y, final_norm, variant: gemma), lm_head), y)
  }

  fn full_attn(l, x, h) {
      let (q, k, v, g) = split4(matmul(x, layer[l].in_proj));
      let q = rope(rmsnorm(q, layer[l].q_norm, variant: gemma), pos, partial: cfg.partial_rotary);
      let k = rope(rmsnorm(k, layer[l].k_norm, variant: gemma), pos, partial: cfg.partial_rotary);
      kv[l].append(k, v);

      let pages = h.on_attn_proj(q, l, kv[l].pages);        // default: identity
      let a = attention(q, kv[l] @ pages) * sigmoid(g);     // gated attention
      h.on_attn(a.scores, q, l);                            // default: no-op
      matmul(a, layer[l].o_proj)
  }

  fn gdn(l, x, h) {
      let (qkv, ab) = split_gdn(matmul(x, layer[l].in_proj));
      let q_pre     = causal_conv1d(qkv, layer[l].conv, state[l].conv);
      h.on_attn_proj(q_pre, l, ..);      // qwen3_5_forward.cpp:903 becomes this line
      let o = gated_delta(gdn_prep(q_pre, ab, layer[l].a_log, layer[l].dt_bias),
                          state[l].recurrent);
      h.on_attn(.., q_pre, l);           // :1200
      matmul(rmsnorm(o, layer[l].gate_norm, variant: gated), layer[l].o_proj)
  }

  fn mlp(l, m) = match layers[l].mlp {
      Dense => matmul(swiglu(matmul(m, layer[l].gate_up)), layer[l].down),
      Moe   => {
          let e: dyn PerToken<Expert> = topk(matmul(m, layer[l].router), cfg.top_k);
          weighted_sum(e.w, matmul(swiglu(matmul(m, layer[l].expert[e.i].gate_up)),
                                                 layer[l].expert[e.i].down))
      }
  };

  opaque { attention = flashinfer{decode, prefill},
           causal_conv1d, gated_delta, gdn_prep, rope, swiglu, rmsnorm }
}
```

Two `dyn` sites in the whole model: `depth` (structural) and `expert[e.i]`
(weight, token granularity). Everything else is static and disappears at trace
time.

### 5.3 Stages are ordinary functions

`trait Hooks` with default methods, passed as a parameter. Not special syntax.

```rust
trait Hooks {                                     // this is the ABI, versioned separately
    fn prologue(&self) {}
    fn on_attn_proj(&self, q: Tensor, l: u32, pages: Pages) -> Pages { pages }
    fn on_attn(&self, score: Tensor, q: Tensor, l: u32) {}
    fn epilogue(&self, logits: Tensor, hidden: Tensor) -> Token { argmax(logits) }
}
```

This gets three things at once:

- **The intrinsic mapping falls out of the arguments.** GDN passing `q_pre` where
  full attention passes `q` is one visible line instead of a decision buried at
  `qwen3_5_forward.cpp:903`. `ModelProfile` / `intrinsic_available` becomes
  generated rather than hand-maintained.
- **The fusion consequence is derived.** A hook that reads `q` forces `q` to be
  materialized, so the fused QKV kernel splits there. Nobody writes
  `active_stage_hooks == nullptr`; it is inferred, and priced at ~7%.
- **`thread_local` becomes a parameter, which is the actual fix.** A per-lane
  `dyn Hooks` means lanes taking the default keep the fused path while lanes with
  a program take the split one, in the same fire.

The trait must stay closed and versioned independently of models: inferlets are
compiled and hash-cached separately, and `quest-attention` has to run on any model
reporting the capability. **Do not let stages attach to model-internal names** —
that was an early mistake in this design; it would couple every inferlet to one
model's internals.

### 5.4 Two axes the DSL does not yet handle honestly

- **`state[l]` is per-request.** GDN's conv and recurrent state are per-sequence
  slots. That is why RS-buffer fires are forced solo today
  (`touches_rs_buffer()`, `worker.rs`). The declaration above does not mark it.
- **`writes { Pages }` changes downstream shape.** A per-lane page mask means
  per-lane page counts. FlashInfer takes CSR page tables so it is expressible,
  but the driver currently rejects it. This is the real gate for co-batching
  quest-class programs.

---

## 6. What dissolves

| today | after |
|---|---|
| `active_stage_hooks` (thread_local, `stage_hooks.hpp:55`) | `fn forward(tok, h: dyn Hooks)` — a parameter |
| 112 fast-path predicates across 23 files | derived: a tap materializes, otherwise fused |
| `requires_solo_submission` for multi-row programs | per-lane lowering |
| `ModelProfile` availability flags, hand-maintained | generated from the declaration |
| `IntrinsicId` closed enum + flags | `trait Hooks` signatures |
| Metal: 2 families (3,796 lines) vs CUDA: 12 (35,571) | one emitter per backend |

That last row is the strongest business case for `model/forward`, and it is
independent of polymorphic batching. Metal is closing the gap by hand -- it went
from one family to two while this was being written -- which is the cost being
paid, not evidence against it. **Justify the two changes separately.**
Bundling them puts both at risk.

---

## 7. Staging

Ordered so that each step is useful alone and none blocks on the next.

**Stage 0 — measure.** Reproduce §3 on this codebase and this hardware. Nothing
below should be built on numbers from another prototype. In particular, confirm
that padding beats bucketing here too before *not* building the bucketing path.

**Stage 1 — hooks become a parameter.** Thread a per-fire hook set through
`IModel::body` instead of `thread_local`, and make the fused-path predicates take
a row count rather than a boolean. No DSL, no new axes. This alone lets a
hook-using request share a fire with an ordinary one.

**Stage 2 — relax solo submission.** `preserves_inner_rows()` and the dense-mask
rejection. Needs per-lane page tables in the attention path.

**Stage 3 — `model/forward`, one family.** Declare `llama_like` (which covers
qwen3, mistral3, phi3, olmo3) and emit against the existing kernels. Keep the
hand-written path alive and diff against it — the parity harness in
`driver/cuda/tests/load_parity` and `tests/inferlets/naive-baseline` are the
golden models. Do not port the exotic families (nemotron_h, deepseek_v4, csm)
yet.

**Stage 4 — WEIGHT-class divergence.** Adapters, using the reserved `lora` sink.
Store at max rank; do not build exact bucketing. This is the roadmap v0.3.0 item.

**Stage 5 — fire planning.** `scheduler/fire_plan.rs`: per-site lowering, row
order, masks. Only now does `runtime` need to know about classes.

**Stage 6 — conditional regions.** Structural divergence at layer granularity,
with the supergraph captured once. This is the piece that is unbuilt in the
prototype too, so expect it to be the hardest.

---

## 8. What would change this plan

- **If padding does not beat bucketing on this hardware**, §5.1's "WEIGHT comes
  from binding, store at max rank" is wrong and the DSL needs a way to express
  bucket structure.
- **If `schema` does not end up shared between loader and forward**, do not nest
  `model/`; keep `loader/` where it is and put `forward/` at top level.
- **If a second backend never wants the declarations**, the case for
  `model/forward` shrinks to "112 predicates in one place", which is real but
  much smaller. Decide with the Metal roadmap in hand.
- **If tensor parallelism is a near-term target**, re-measure everything: a split
  baseline pays collective communication *per program* there, so the merged path
  plausibly gains more, and none of §3 probes it.

---

## 9. What this plan does not claim

The forward pass stays hand-written C++ kernels. Nothing here generates a kernel;
`model/forward` decides which existing kernels are called, in what order, with
what indirection. FlashInfer, cuBLAS, CUTLASS and the 196 kernels in
`driver/cuda/src/kernels` are untouched — and 108 of those are used twice or
less, which is why the `opaque` escape hatch is load-bearing rather than
decorative.

The declarative layer does not make anything faster by itself. The 35.5k lines
hold tuning knowledge, and moving them to a declaration risks losing some of it —
`recognize_library_dataflows`-style pattern re-folding is where that will show up
first, quietly, as a regression. The justification is Metal getting twelve
families and 112 predicates having one home, not throughput.
