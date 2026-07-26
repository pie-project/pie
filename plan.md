# North star: polymorphic batching in Pie

Status: **candidate.** Deliberately high level — this is the shape of the thing,
not the plan for building it. The evidence, the file:line citations, the measured
numbers and the staging are in `pie-application-plan.md`.

---

## The line

> **`forward/` defines what a pass computes, `compiler/` builds a program,
> `runtime/` schedules it, `driver/` fires it.**

One clause added to Pie's existing north star. Everything below follows from
where that clause sits.

---

## Why

A serving engine receives requests that want different computations. One carries
an adapter and another does not; one verifies four speculative tokens while
another decodes one; one carries an image; one is served by a shallower variant
of the same checkpoint.

Engines have converged, independently and repeatedly, on the same answer: run the
differing programs together anyway, sharing whatever operators they have in
common. ORCA batches the MLP and splits at attention. Chunked prefill mixes
prefill and decode rows. SGMV serves many adapters from one kernel. MoE routing
gathers tokens per expert. Each is hand-written for one axis, and they do not
compose.

**Pie has already built this twice** — once for PTIR programs (`grouped_runtime`
fans one launch across lanes) and once for MoE experts (grouped GEMM) — and is
the only engine that has named the principle: *batch-by-program*.

What it does not do is cross the model boundary. Inside `IModel::body` there is
exactly one model, one weight set, one depth, and any divergence anywhere in the
batch collapses the whole fire onto a slow path. That boundary is what this work
moves.

---

## Part 1 — the declaration

### The principle

> **The declaration says what varies. It never says how to lower it.**

This is not stylistic. Measured: exact per-rank bucketing *loses* to padding.
If authors could write the lowering, they would write the one that loses. Fusion
is the same — a fused edge cannot be a merge point, so fusion and merging must be
chosen together, by something that can see both.

### The consequence: two of three divergence classes need no syntax

| class | how it is expressed | cost |
|---|---|---|
| **correction** (LoRA) | nothing — derived from algebra | ~1× the no-divergence floor |
| **weight** (adapters, experts) | nothing — derived from binding | one batched GEMM, no branch |
| **structural** (depth, MoD, phase) | **syntax** | the fused region must split |

*Weight* comes from binding: the author writes `matmul(x, layer[l].qkv)`, and if
the deployment binds many of them the compiler sees a lane dimension. *Correction*
comes from algebra: if a weight is bound as "base plus a per-lane low-rank delta",
`x(W + BA)ᵀ` distributes into a shared GEMM plus a small one. With no adapters the
term vanishes and the code is what it was.

**Syntax is required exactly where cost is incurred.** No marker means free. That
asymmetry is the design, and it is why the declaration for a model with no
structural divergence reads like an ordinary forward pass.

### What it looks like

```rust
model! { qwen3_6

  layers = cfg.layer_types;            // [Full, Linear, ...] — static, from the checkpoint
  depth  : dyn PerRequest<u32>;        // the only structural syntax in this model

  forward(tok, h: dyn Hooks) -> Token {
      h.prologue();
      let mut y = embed[tok];
      for l in 0..depth {
          let x = rmsnorm(y, layer[l].attn_norm, variant: gemma);
          y += match layers[l] {                       // static match, no runtime branch
              Full   => full_attn(l, x, h),
              Linear => gdn(l, x, h),
          };
          y += mlp(l, rmsnorm(y, layer[l].mlp_norm, variant: gemma));
      }
      h.epilogue(matmul(rmsnorm(y, final_norm, variant: gemma), lm_head), y)
  }

  fn full_attn(l, x, h) {
      let (q, k, v, g) = split4(matmul(x, layer[l].in_proj));
      let q = rope(rmsnorm(q, layer[l].q_norm, variant: gemma), pos, partial: cfg.partial_rotary);
      let k = rope(rmsnorm(k, layer[l].k_norm, variant: gemma), pos, partial: cfg.partial_rotary);
      kv[l].append(k, v);

      let pages = h.on_attn_proj(q, l, kv[l].pages);   // default: identity
      let a = attention(q, kv[l] @ pages) * sigmoid(g);
      h.on_attn(a.scores, q, l);                       // default: no-op
      matmul(a, layer[l].o_proj)
  }

  fn mlp(l, m) = match layers[l].mlp {
      Dense => matmul(swiglu(matmul(m, layer[l].gate_up)), layer[l].down),
      Moe   => {
          let e: dyn PerToken<Expert> = topk(matmul(m, layer[l].router), cfg.top_k);
          weighted_sum(e.w, matmul(swiglu(matmul(m, layer[l].expert[e.i].gate_up)),
                                                 layer[l].expert[e.i].down))
      }
  };

  opaque { attention, causal_conv1d, gated_delta, rope, swiglu, rmsnorm }
}
```

Three things to notice.

**Static and dynamic are `impl` and `dyn`.** `match layers[l]` is a real match
that runs at trace time — which for a tracing eDSL is *model load time*, when
`config.json` is in hand. Rope variant, qk-norm, post-norm, fused-QKV opt-in,
quantization: all of it resolves then and leaves no trace. Only `dyn` reaches the
planner. This is strictly stronger than compile-time generics, because the
variation is driven by data the Rust compiler never sees.

**The same expression is two mechanisms.** `matmul(x, W[i])` with `i` per-token
is MoE grouped GEMM; with `i` per-request it is SGMV. Two hand-written kernels,
one expression. The observation that motivated this work becomes a syntactic
identity.

**Stages are ordinary functions.** `trait Hooks` with default methods, passed as a
parameter. Not special syntax, not a `thread_local`, not a fixed list of four
things baked into every model file. The defaults *are* the "no program attached"
case, so a lane taking the default is a call that can be inlined — which is
exactly the condition under which the fused kernel survives.

**Kernels are named as operations, not as kernels.** `rmsnorm(variant: gemma)`,
not `launch_rmsnorm_gemma_bf16`. Op→kernel selection belongs to the backend, and
keeping that line clean is what lets a second backend exist later.

### What it does not do

It generates no kernels. FlashInfer, cuBLAS, CUTLASS and the several hundred
kernels already in the driver are untouched. The declaration decides which
existing kernels are called, in what order, with what indirection — and the
`opaque` block is where anything genuinely exotic keeps its own name.

---

## Part 2 — the planner

A fire holds several programs. The planner decides how they share one pass.

### What it decides

For each **divergence site** — a point the model declared it can carry variation
at — one **lowering**:

| lowering | meaning | compiler analogue |
|---|---|---|
| uniform | every lane agrees | devirtualization |
| prefix | the agreeing lanes take the fast path; the rest do not | loop peeling |
| batched / padded | one kernel over per-lane weights | dictionary passing |
| conditional | genuinely different operators, guarded | guarded dispatch |

Plus three things that are not per-site:

- **row order** — the permutation that makes each site's divergence contiguous,
  so "fast path on the first *n* rows" is a row count rather than a gather. This
  is loop unswitching, and it is what makes `prefix` possible at all.
- **extents and masks** — per-lane row counts, active-row bitsets.
- **pointer tables** — per-lane weights, for the classes that are data rather
  than control.

### What it does not decide

**Not fusion, separately from merging.** They compete for the same tensor, so one
thing chooses both.

**Not the final answer, alone.** The plan emits *candidate* lowerings and lets the
runtime pick using device knowledge. This mirrors what Pie's PTIR planner already
does — two partitions per stage, singleton and fused, with the backend choosing —
and it keeps the device-independence rule intact for the new toolchain too.

### The rule it applies

Not "merge when possible". The measured rule is narrower and more useful:

> Padding is free when it inflates a dimension the machine was not being paid for
> anyway, and ruinous when it multiplies the layer itself.

At one row per program a decoder layer is bound by reading weights, so a wider
batch costs nothing — every row-padding axis collapses to about 1×, and the right
answer is to pad. When the pad is a multiple of the layer — dense MoE routing,
dense mixture-of-depths — it is never free at any width, and merging wins by
several times.

**And the axes in that second group are precisely the ones for which someone has
had to hand-write a kernel.** Grouped GEMM for MoE exists. For mixture-of-depths
nobody wrote one, and that is the largest gap measured.

The corollary matters more than the rule: with several axes live, the right answer
is *a different lowering per axis*. Neither pure strategy wins. That is the case
for a planner, and it is not the case that "merging is fast".

---

## Part 3 — how it lands in Pie

### Three phases, three homes Pie already has

| phase | what | where | frequency |
|---|---|---|---|
| **plan** | classify sites, choose candidate lowerings, order rows | `runtime/engine/scheduler` | per program set, cached |
| **materialize** | fill lane tables, masks, pointer arrays | `IModel::prepare()` | per step |
| **execute** | one pass | `IModel::body()` + capture | per step |

The second and third are not a new split. `body()` is already host-work-free so
that it can be captured into a CUDA graph, and `prepare()` exists to hold the host
work that was hoisted out of it. **Polymorphic batching needs exactly the same
split for exactly the same reason**, so it lands on a seam that is already there.

The planner sits in `runtime/engine/scheduler` because planning a fire needs four
things and only the scheduler has all four: the model's divergence sites, the
attached programs and what they require, the device cost model, and what was
admitted. It belongs beside `LaunchGrouping::accepts` — *can these co-batch* — as
its natural continuation: *how do they co-batch*.

### What changes

- **`forward/`** is new: the declaration eDSL, the traced form, and per-target
  emission. Shaped like `loader/` — a Rust crate with a C ABI that the driver
  calls at cold start.
- **`driver/`** gains an `IModel` implementation that runs a declared forward, and
  its fast-path conditions take a row count where they used to take a boolean.
- **`runtime/`** gains fire planning, and its admission rules stop forcing solo
  submission for cases a plan can now express.

### What does not change

**`compiler/` is untouched.** PTIR is one polymorphism axis among several, and by
measurement the cheapest one — post-logit divergence is already nearly free. The
op set stays closed, the trace container stays the same, inferlets keep compiling
against a stable surface. "Which PTIR program" simply becomes one more `dyn`
alongside "which adapter" and "which depth", and Pie's existing lane runtime is
already its lowering.

The hand-written model families stay until a declaration has been proved
bit-identical to each one. Nothing is replaced; things are added next to what they
must match.

---

## What "done" looks like

Concrete enough to argue with:

1. A declaration of one model family produces token-identical output to its
   hand-written forward, on the same checkpoints and the same fires.
2. A request that attaches a PTIR stage program shares a fire with one that does
   not, and the second keeps its fused kernel.
3. Two adapters of different rank serve in one fire without a separate pass, and
   without an exact-bucketing kernel existing anywhere.
4. The condition "any divergence, take the slow path" appears in no model file,
   because it is derived.
5. A second backend implements one emitter rather than twelve model families.

---

## Non-goals

- **Not a kernel generator.** If this ever emits a fused kernel of its own, the
  design has drifted.
- **Not faster by itself.** The declarative layer is a legibility and reach
  argument — a second backend, one home for a rule currently written out a hundred
  times. Throughput comes from the polymorphism work, and only on the axes where
  padding is not already free.
- **Not a replacement for PTIR.** Different authors, different trust, different
  lifecycle. They compose; they do not merge.
- **Not a rewrite of the tuned kernels.** The knowledge in the existing forward
  passes is real, and moving it into a declaration risks losing some of it. Where
  a declaration cannot express a tuned path, `opaque` is the correct answer, not a
  worse kernel.
