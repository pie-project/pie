//! The traced form: what one forward pass computes, as data.
//!
//! Values are SSA — each is produced by exactly one op — and shapes are
//! symbolic in the fire's extents (`Dim::Tokens`, `Dim::Requests`), because
//! the trace is taken once per model load, not per fire. Weights appear by
//! declaration name (`layer.3.qkv`); resolving names to device tensors is
//! the driver contract's job, exactly as it is for the loader.
//!
//! The op vocabulary is deliberately the *operation* vocabulary of the
//! hand-written passes, not their kernel vocabulary: `Matmul` + `SplitQkv` +
//! `RmsnormQk` + `Rope` is what the fused decode kernel computes, and
//! whether those four ops become one launch is the emitter's choice, made
//! per fire — the hook-free prefix taking the fused kernel while the tail
//! runs unfused (stage1-notes.md) is exactly that choice, and it is not
//! expressible if the trace bakes the fusion in.
//!
//! # `dyn`: the first per-token axis
//!
//! Everything above is resolved at trace time. The MoE expert axis is the
//! first thing that is not: `TopK` produces a per-token expert assignment
//! whose CONTENT exists only at fire time, and the expert-indexed `Matmul`s
//! downstream of it name a weight *template* (`layer.0.expert.{e}.gate_up`)
//! whose `{e}` the selector resolves per token. This is the first trace
//! whose lowering is not fixed at trace time — the expert dimension is
//! data — and, per the tart prototype's `ir.py`, per-token weight selection
//! IS `Div::Weight` at token granularity: gather → grouped GEMM → scatter is
//! its lowering, and `matmul(x, W[i])` with `i` per-token being MoE grouped
//! GEMM (with `i` per-request, SGMV) is the syntactic identity that
//! motivated this work (plan.md Part 1). The trace states the selection;
//! which grouped-GEMM strategy fires (cuBLAS batched, aligned blocks,
//! CUTLASS fused) stays the emitter's per-fire choice, exactly as fusion
//! does. The [`DynAxis`] marker on values and the `selector` field on
//! [`OpKind::Matmul`] are that syntax — present exactly where cost is
//! incurred, absent everywhere else.
//!
//! # The per-request state axis
//!
//! The GDN ops (`CausalConv1d`, `GatedDelta`) are the first whose semantics
//! include a store that is per-layer AND per-request: each request owns a
//! conv-window slab and a recurrent-state slab that the op reads and
//! advances in place, across fires (pie-application-plan.md §5.4's
//! "state[l] is per-request" — the axis the sketch left unmarked, and the
//! reason RS-touching fires are forced solo today, `touches_rs_buffer()`).
//! The trace marks it the way the KV cache is already marked: the ops carry
//! `layer` and the store stays implicit, NOT a traced value. That is a
//! deliberate design call, justified by the hand-written pass: state never
//! appears as an activation there — every state-touching kernel takes the
//! cache base plus a per-request slot indirection (`slot_ids_d`) and
//! mutates the slab in place — and a traced SSA value is per-fire and
//! single-assignment, so a first-class state value would misstate both the
//! lifetime (state outlives the fire) and the dataflow (state is not
//! produced by any op of this pass). What the planner needs is the FACT
//! that an op addresses such a store; [`OpKind::state_ref`] derives exactly
//! that from the vocabulary, so "does this trace touch per-request
//! recurrent state" is a query, not a name-match. (`DynAxis::PerRequest`
//! stays un-introduced: `dyn` marks values whose CONTENT selects structure,
//! and no state value exists to mark.)

use serde::{Deserialize, Serialize};

/// One symbolic extent of a value's shape.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum Dim {
    /// The fire's token rows (`N`; equals `Requests` on a pure-decode fire).
    Tokens,
    /// The fire's request rows (`R`).
    Requests,
    /// A load-time constant: hidden size, head count x head dim, vocab.
    Const(u32),
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Shape(pub Vec<Dim>);

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    BF16,
    F32,
    I32,
}

/// Index into [`ForwardPlan::values`].
pub type ValueId = u32;

/// The `dyn` marker: which fire extent a value's *selection* varies over.
///
/// Marks values whose content chooses lowering-relevant structure per
/// element of an extent — today only the per-token expert assignment a
/// [`OpKind::TopK`] produces. Ordinary activations are per-token *data* and
/// carry no marker; the marker means "the planner must look at this value's
/// content to know which weights a downstream op reads" (plan.md Part 1's
/// `dyn PerToken<Expert>`). `PerRequest` (adapters, depth) is the same
/// grammar at request granularity and lands with its own axes later.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DynAxis {
    /// Varies per token row of the fire (`Dim::Tokens` granularity).
    PerToken,
}

/// RMSNorm weight conventions that change the arithmetic, not the kernel
/// choice. `Gemma` folds `(1 + w)`; `Plain` multiplies `w` directly.
///
/// `Default` is `Plain` so the field can ride serde-additively on ops that
/// predate it ([`OpKind::RmsnormPerHead`]): a golden that never stated a
/// variant reads back as the plain fold it always meant.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormVariant {
    #[default]
    Plain,
    Gemma,
}

impl NormVariant {
    /// Serde helper: `Plain` is the resting value and is skipped on
    /// serialization, the discipline that keeps pre-variant goldens
    /// byte-identical (the same rule as `selector`/`dyn_axis`).
    pub fn is_plain(&self) -> bool {
        *self == NormVariant::Plain
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RopeKind {
    Standard,
    /// Llama3/YaRN-style frequency scaling; parameters live in the facts.
    Yarn,
}

/// The fire-shape class a LOWERED trace is specialized to (north-star-dsl.md).
///
/// What varies after model load and CHANGES WHICH OPS RUN: the toolchain
/// traces a lowered declaration once per class, so inside the
/// declaration a class arm is an ordinary trace-time `match` — the same
/// mechanism that erases static facts, applied to the axes that used to
/// be the drivers' `is_pure_decode` / `commit_advance` / `state_only`
/// booleans. Anything that changes only a kernel's PARAMETER
/// (`verify_frozen`'s write_state) is not a class; anything that changes
/// a kernel choice per fire within one op list is a [`GuardPred`].
///
/// Semantic traces ([`crate::family::llama_like`]) have no class: they
/// serve every fire shape, and kernel choice stays with their consumer.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum FireClass {
    /// Pure decode: every request contributes one token row.
    Decode,
    /// Prefill-shaped (anything else — the hand-written bodies treat
    /// mixed fires as one qo_indptr-windowed prefill, a decode row being
    /// an `Nr == 1` window).
    Prefill,
    /// The spec-decode repair pass (qwen3_5 MTP): ONLY each linear
    /// layer's conv+prep+recurrence over the confirmed prefix, fed from
    /// the verify stash — no embed, attention, MLP or epilogue. A
    /// genuinely different pass, so a genuinely different trace.
    CommitAdvance,
    /// The speculative repair's whole-backbone flavor: everything except
    /// the final-norm/lm_head epilogue.
    StateOnly,
    /// The frozen-verify service (qwen3_5 MTP): the prefill body plus a
    /// verify-stash STORE per linear layer. Reserved by the rung-5
    /// geometry; its trace is the next qwen3_5 slice.
    FrozenVerify,
    // The masked classes (wire 5/6) and the hooked classes (wire 7/8)
    // are RETIRED (A1/A2, the class-collapse amendment): a custom mask
    // is a GuardPred::HasCustomMask arm and attached stage hooks are a
    // GuardPred::HasStageHooks arm of the Decode/Prefill traces now —
    // the op-list deltas are local, so they live at op granularity.
    // What remains a class is what changes the PASS wholesale: the
    // fire's shape and the MTP services. The wire numbers stay
    // reserved (append-only ABI); the trace entries answer
    // InvalidArgument.
}

// (The short-lived `AttnKernel` enum — rung 1's `Attention.param1` tag —
// is gone: a lowered trace states its attention kernel the way it states
// every kernel, as an [`OpKind::Launch`] with the launcher's name. Raw
// signatures, not enum tags; north-star-dsl.md.)

/// A [`OpKind::Guard`] arm's predicate: the ONE kind of branch a lowered
/// trace may carry — over a per-fire RUNTIME INPUT, closed-vocabulary so
/// every predicate is emittable as a fixed C++ condition (rung 3's
/// generated form spells it; the interpreter evaluates it). Trace-time
/// facts never appear here — they resolved during tracing. Predicates
/// may carry a load-time VALUE (the token thresholds: env-tunable
/// driver constants, resolved into the trace like every fact) but never
/// an open-ended expression: a declaration cannot smuggle an arbitrary
/// runtime choice past the toolchain. Wire form: a (kind, payload) u32
/// pair; kinds are appended-only.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum GuardPred {
    /// The fire carries explicit KV-write descriptors (`w_page`/`w_off`),
    /// the graph-replay steering path — `has_write_desc` in every driver
    /// signature. Wire kind 0, payload unused.
    HasWriteDesc,
    /// `N <= k`: the fire's token rows within a threshold (the
    /// warp-tiled prefill ceiling). Wire kind 1, payload `k`.
    TokensLE(u32),
    /// `N > k` (the cached-prefill floor). Wire kind 2, payload `k`.
    TokensGT(u32),
    /// The fire's attached programs read attention scores at `OnAttn`
    /// (`StageHooks::wants_attn_score`) — the score-capturing attention
    /// dispatch runs instead of the plain one. Wire kind 3, payload
    /// unused.
    WantsAttnScore,
    /// The fire carries a custom attention mask (`custom_mask_d !=
    /// nullptr`): the masked arm runs the custom-mask prefill dispatch
    /// and, in the fused-decode deployment, the general QKV sequence —
    /// the class-collapse amendment's first predicate (a mask is a
    /// guard, not a class). Wire kind 4, payload unused.
    HasCustomMask,
    /// The fire carries attached stage-hook programs (`stage_hooks !=
    /// nullptr`) — A2 of the class-collapse amendment: the hooked arm
    /// holds the general QKV sequence, the two per-layer HookSites and
    /// the WantsAttnScore-guarded attention. The caller's gate admits
    /// only ALL-hooked fires (fast_rows == 0), so presence ⇔ every row
    /// is hooked; A3's Peel op replaces this all-or-nothing arm with
    /// the fast_rows row split. Wire kind 5, payload unused. RETIRED
    /// vocabulary since A3 (reserved, unstated).
    HasStageHooks,
    /// The fire carries usable lora lanes (`lora != nullptr &&
    /// lora->usable()`) — the §5.1 correction: the adapter delta lands
    /// on the materialized q/v projections before anything consumes
    /// them, and the fused decode-QKV epilogue (which writes V straight
    /// to the paged cache, so there is nothing to correct into) must
    /// not run. Wire kind 6, payload unused.
    HasLora,
}

impl GuardPred {
    /// The ABI (kind, payload) pair.
    pub fn wire(&self) -> (u32, u32) {
        match *self {
            GuardPred::HasWriteDesc => (0, 0),
            GuardPred::TokensLE(k) => (1, k),
            GuardPred::TokensGT(k) => (2, k),
            GuardPred::WantsAttnScore => (3, 0),
            GuardPred::HasCustomMask => (4, 0),
            GuardPred::HasStageHooks => (5, 0),
            GuardPred::HasLora => (6, 0),
        }
    }
}

/// A model-body hook site's stage (the HookSite slice,
/// north-star-dsl.md): the TWO points at which a fire's attached PTIR
/// programs observe and intervene inside the forward. PTIR's Prologue
/// and Epilogue are dispatch-side post-logits machinery and never trace
/// ops. Wire values are the ABI (`HookSite.param0`).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum HookStage {
    /// Before the layer's attention: observes q, intervenes through the
    /// page-mask sink (the driver brackets `begin_layer` → invoke →
    /// compact/pointer-swap; a narrowed page list feeds the SAME stated
    /// attention kernel as substituted arguments).
    OnAttnProj,
    /// After the layer's attention: observes the scores the (possibly
    /// capturing) attention published through the sideband.
    OnAttn,
}

/// One arm of a [`OpKind::Guard`] chain: the first arm whose predicate
/// holds runs; `ops` is the arm's flat region length.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct GuardArm {
    pub pred: GuardPred,
    pub ops: u32,
}

/// ONE SEAM STATEMENT the model text made, in text order
/// (`.wiki/tart/dsl.md` ①, migration step 4).
///
/// Three of the five seams lower to ops today (two `HookSite`s and the
/// adapter's `HasLora` guard); the two BOUNDARY seams lower to nothing
/// at all, which is why prologue and epilogue live in a different world
/// from the rest — the traced form does not record that the text has
/// them. This list records every seam the text stated, whichever way it
/// lowered, so "what does this declaration expose?" has one answer.
///
/// `op` is the index of the op carrying the seam when one does. A
/// boundary seam has none: it is a statement about the trace, not a
/// point inside it.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct SeamStatement {
    pub seam: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub layer: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub op: Option<u32>,
}

/// One operation of the traced form.
///
/// Weights are referenced by name; `layer` tags the ops that address
/// per-layer state (KV cache, layer weights) so the driver can bracket its
/// layer loop without re-deriving structure from names.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum OpKind {
    /// Token ids -> hidden rows, via the embedding table.
    Embed { weight: String },
    /// `out = act @ weight^T (+ beta * out)`. `beta_one` is the residual
    /// accumulate the hand-written passes fold into cuBLAS.
    ///
    /// With `selector` set, `weight` is a TEMPLATE (`layer.0.expert.{e}.gate_up`)
    /// whose `{e}` the selector value — a per-token expert assignment,
    /// `[Tokens, k]` of expert indices — resolves per token: row `t` of the
    /// activation is multiplied against the weights its `k` selected experts
    /// name, producing a `[Tokens, k, out]` result. This is `Div::Weight` at
    /// token granularity; grouped GEMM is its lowering (the drivers' MoE
    /// gate_up/down kernels), chosen by the emitter per fire. The selector
    /// is also the op's LAST input (the [`TraceBuilder::matmul_add`]
    /// convention for auxiliary operands), so dataflow walks need no special
    /// case; the field states which input selects rather than flows.
    Matmul {
        weight: String,
        beta_one: bool,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        selector: Option<ValueId>,
    },
    /// Row RMSNorm over the trailing dim.
    Rmsnorm {
        weight: String,
        variant: NormVariant,
    },
    /// Broadcast bias add over `[rows, width]`: `x[r, :] += bias`. The
    /// Qwen-2 family's attention biases (`{q,k,v}_proj.bias`), applied to
    /// the raw projections after the lora correction and before
    /// norms/rope — the hand-written `maybe_add_bias` position
    /// (llama_like.cpp). The kernel is 1:1 (`launch_add_bias_bf16`), so
    /// the semantic and lowered traces state the same op.
    AddBias { weight: String },
    /// Per-head RMSNorm of packed `[rows, heads * head_dim]` Q or K.
    /// `variant` selects the weight fold exactly as on [`OpKind::Rmsnorm`]:
    /// qwen3/olmo-style checkpoints multiply `w` directly (`Plain`), while
    /// qwen3.5's full-attention q/k norms fold `(1 + w)` (`Gemma` —
    /// `full_attn_layer_body` launches `launch_rmsnorm_gemma_bf16` over
    /// `N * heads` rows of `head_dim`). Serde-defaulted to `Plain` and
    /// skipped there, so every pre-variant golden stays byte-identical.
    RmsnormPerHead {
        weight: String,
        head_dim: u32,
        #[serde(default, skip_serializing_if = "NormVariant::is_plain")]
        variant: NormVariant,
    },
    /// Split packed QKV `[rows, q + 2kv]` into Q, K, V (three results).
    SplitQkv { q_width: u32, kv_width: u32 },
    /// Rotary embedding applied in place to Q and K (two operands).
    ///
    /// `partial` is the partial-rotary width: `Some(rotary_dim)` rotates
    /// only the first `rotary_dim` channels of each head and passes the
    /// rest through (qwen3.5 full attention, `launch_rope_partial_bf16`);
    /// `None` is the full rotation every earlier family traces. The trace
    /// states the resolved CHANNEL COUNT, not HF's `partial_rotary_factor`,
    /// for the same reason `SplitQkv` states widths rather than head
    /// counts: every trace-time constant is already multiplied out, and the
    /// driver's `max(2, 2 * int(0.5 * factor * head_dim))` derivation is
    /// config-parsing knowledge that belongs with the facts (the fixture
    /// pins 0.25 × 256 → 64 with its provenance). Serde-skipped when
    /// absent, so pre-partial goldens stay byte-identical.
    Rope {
        kind: RopeKind,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        partial: Option<u32>,
    },
    /// Append this fire's K/V rows to the layer's paged cache.
    KvAppend { layer: u32 },
    /// Paged attention over the layer's cache. Opaque in the SEMANTIC
    /// trace: the executor derives the path. A lowered trace states its
    /// attention kernel as an [`OpKind::Launch`] instead of this kind.
    Attention { layer: u32 },
    /// SwiGLU over packed `[rows, 2 * inter]` gate‖up.
    Swiglu { inter: u32 },
    /// Gather the sampled rows and project to logits.
    LmHead { weight: String },
    /// `residual += x`, elementwise. The post-norm residual landing
    /// (`NormPlacement::Post`): the sub-layer's normed output is added to
    /// the residual stream by its own launch, because the norm between the
    /// projection GEMM and the add is what makes the pre-norm `beta=1`
    /// fold impossible. A separate op because it is a separate launch in
    /// the hand-written pass (`launch_residual_add_bf16`).
    ResidualAdd,
    /// Router top-k over per-token logits: for each token row, the `k`
    /// highest-scoring experts, with softmaxed-and-renormalized routing
    /// weights. Two results: the expert indices (`[Tokens, k]` i32, marked
    /// [`DynAxis::PerToken`] — the `dyn` value everything expert-indexed
    /// consumes) and the routing weights (`[Tokens, k]` f32). One op
    /// because it is one launch in the hand-written MoE pass
    /// (`launch_topk_softmax_bf16`: top-k + softmax + renormalize).
    TopK { k: u32 },
    /// Per-token combine of the k routed expert outputs:
    /// `out[t] = sum_j w[t, j] * x[t, j, :]`, collapsing `[Tokens, k, d]`
    /// to `[Tokens, d]`. The hand-written MoE pass's
    /// `launch_token_batched_weighted_sum_bf16` (the prefill path's
    /// per-expert `scatter_add_weighted` loop is a lowering of the same
    /// combine, chosen with the grouped GEMM it follows).
    WeightedSum { k: u32 },
    /// Shared-expert landing: `out = base + sigmoid(gate) * x`, the scalar
    /// per-token gate broadcast over the hidden dim. Operands `[x, gate,
    /// base]` — fresh value first, the stream it lands on last, the
    /// [`TraceBuilder::residual_add`] convention. One op because it is one
    /// launch (`launch_sigmoid_scalar_gate_add_bf16`); the `[Tokens, 1]`
    /// gate logit comes from an ordinary `Matmul` the trace states
    /// separately, exactly as the hand-written pass launches it.
    SigmoidGateAdd,
    /// Split a packed `[rows, w0 + w1]` value at `w0` into two (two
    /// results). The GDN in-projection splits when the deployment binds the
    /// fused banks: `in_proj_qkvz` → (mixed qkv, z gate) and `in_proj_ba` →
    /// (b, a) — `launch_split_bf16_rows` and `launch_split_qwen_gdn_ba_bf16`
    /// respectively, one op each because each is one launch. Distinct from
    /// [`OpKind::SplitQkv`], which is the three-way attention split.
    SplitGdn { width0: u32, width1: u32 },
    /// Depthwise causal conv1d over the packed `[rows, conv_dim]` qkv, with
    /// the fused SiLU the hand-written kernels apply
    /// (`launch_causal_conv1d_{update,prefill}*`). `weight` names the conv
    /// binding (the driver binds the checkpoint's conv weight AND bias
    /// under it); `kernel` is the window width (`linear_conv_kernel_dim`).
    /// `layer` marks the implicit PER-REQUEST conv-state slab the op reads
    /// and advances — see the module doc's "the per-request state axis" and
    /// [`OpKind::state_ref`]. Decode-update vs prefill-walk vs batched
    /// slot-indirected variants are lowerings of this one op, the emitter's
    /// per-fire choice.
    CausalConv1d {
        weight: String,
        layer: u32,
        kernel: u32,
    },
    /// The post-conv GDN prep (`launch_qwen_gdn_post_conv_prep_bf16`): one
    /// launch that unpacks the conv output's `[q_raw | k_raw | v_raw]`,
    /// L2-normalizes q/k into compact per-head fp32, converts v to fp32,
    /// and folds `a`/`b` with the `a_log`/`dt_bias` parameters into the
    /// per-head gating log-decay `g` and mixing `beta`. Inputs `[qkv, a,
    /// b]` (the kernel's operand order); five results: q `[Tokens, Kh,
    /// Kd]`, k `[Tokens, Kh, Kd]`, v `[Tokens, Vh, Vd]`, g `[Tokens, Vh]`,
    /// beta `[Tokens, Vh]`, all f32. Two weight names because the launch
    /// reads two parameter tensors. (The GQA `repeat_interleave` of q/k
    /// from Kh to Vh heads is NOT an op: most recurrence kernels index the
    /// compact layout directly, so materializing it is a lowering choice.)
    GdnPrep { a_log: String, dt_bias: String },
    /// The gated-delta recurrence: fold this fire's tokens into the layer's
    /// PER-REQUEST recurrent state and produce the core attention output
    /// `[Tokens, Vh, Vd]` f32. Inputs `[q, k, v, g, beta]`. Opaque, like
    /// `Attention`: the decode-step, chunked-prefill, warp-tiled and cached
    /// kernel families (`launch_{recurrent,chunk}_gated_delta_*`) are all
    /// lowerings the backend picks per fire. `layer` marks the implicit
    /// per-request state slab ([`OpKind::state_ref`]).
    GatedDelta { layer: u32 },
    /// Gated RMSNorm (`launch_rmsnorm_gated_fp32_in_bf16`): per (row,
    /// head), `out = w * rmsnorm(x) * silu(gate)`, normalizing the trailing
    /// head dim of the rank-3 f32 core output and flattening to the gate's
    /// `[Tokens, Vh * Vd]` bf16 shape (the fp32→bf16 conversion is fused
    /// into the same launch). Inputs `[x, gate]`. NOT a [`NormVariant`]:
    /// variants select the weight arithmetic at fixed arity, while gating
    /// adds an operand and changes the launch — and the kernel's weight
    /// fold is plain (`rmsnorm.hpp`: "Plain weight (no `1+w` convention)"),
    /// so there is no variant to state.
    RmsnormGated { weight: String },
    /// The interleaved per-head `[query | gate]` split of qwen3.5 full
    /// attention's 2×-wide gated q projection
    /// (`launch_split_q_gate_bf16`): the packed `[rows, heads * 2 *
    /// head_dim]` input carries, PER HEAD, `head_dim` query channels then
    /// `head_dim` gate channels — `q[n, h*d + i] = packed[n, h*2d + i]`,
    /// `gate[n, h*d + i] = packed[n, h*2d + d + i]` — so this is NOT a row
    /// split: [`OpKind::SplitGdn`] cuts a packed row at one offset, while
    /// this op de-interleaves at head granularity. Two results, q then
    /// gate, each `[rows, heads * head_dim]`.
    SplitQGate { heads: u32, head_dim: u32 },
    /// `out = x * sigmoid(gate)`, elementwise — qwen3.5 full attention's
    /// output gate (`launch_sigmoid_gate_inplace_bf16`: `attn_out *=
    /// sigmoid(gate)` before o_proj). Operands `[x, gate]`, same shape.
    /// A multiply with NO residual and no landing: distinct from
    /// [`OpKind::SigmoidGateAdd`], whose scalar per-token gate broadcasts
    /// over the hidden dim and lands on a base stream — here the gate is
    /// full-width and nothing is added.
    SigmoidGateMul,
    /// A STATED kernel launch — the op a LOWERED trace uses wherever the
    /// declaration's class arm called a raw kernel signature
    /// (`dsl::cuda`, north-star-dsl.md). `kernel` is the driver's
    /// launcher symbol; a dumb consumer resolves it in a name→launcher
    /// registry and launches, and the ABI stops growing per kernel — this
    /// ONE kind carries every stated kernel, present and future.
    ///
    /// `weights` are the weight names the launcher consumes, in signature
    /// order. `state` marks the implicit per-layer store the kernel
    /// addresses (the fused decode-QKV kernel writes the KV cache), the
    /// same declaration [`OpKind::state_ref`] derives from vocabulary for
    /// semantic kinds. Operand values ride `inputs`/`outputs` like every
    /// other op; mechanical launcher parameters (stream, dims, workspace)
    /// are the driver's binding, not the trace's business.
    ///
    /// A SEMANTIC trace never contains one — the general arm it lowers
    /// remains the semantics the parity harness holds it to.
    Launch {
        kernel: String,
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        weights: Vec<String>,
        #[serde(default, skip_serializing_if = "Option::is_none")]
        state: Option<StateRef>,
    },
    /// The one branch a lowered trace may carry: a CHAIN of arms over
    /// per-fire RUNTIME INPUTS ([`GuardPred`], closed vocabulary) — the
    /// first arm whose predicate holds runs, the trailing `else_ops`
    /// region when none does. Regions are flat and consecutive (arm 0's
    /// ops immediately after this op, then arm 1's, …, then the else's);
    /// no nesting (the builder enforces it). The interpreter evaluates
    /// the arms in order and jumps every dead region; rung 3's emitter
    /// spells `if / else if / else` — the ONLY branch a generated file
    /// carries that the declaration wrote.
    ///
    /// A guard may PRODUCE values (rung 4c: the recurrence three-way's
    /// output): the values are the GUARD's outputs, and each region's
    /// launches are its lowerings — they bind the same output buffer and
    /// record no SSA outputs of their own, so dataflow sees one producer
    /// whichever arm ran. A side-effect-only guard (the KV write) simply
    /// has no outputs.
    Guard {
        arms: Vec<GuardArm>,
        else_ops: u32,
    },
    /// A hook site ([`HookStage`]; the HookSite slice): the point where
    /// the fire's attached PTIR programs run against this layer. The op
    /// observes its input (q — the value `invoke_stage_hook` passes) and
    /// produces nothing: interventions travel through sidebands (the
    /// page-mask sink, the score capture) and are ARGUMENT-driven — a
    /// site with nothing attached is a no-op by argument, not by branch,
    /// which is what lets the same trace serve every program. WHICH
    /// program runs is `dyn` (sideband data); this op states only WHERE
    /// and WHAT IS OBSERVABLE.
    HookSite { stage: HookStage, layer: u32 },
    /// Loop peeling as vocabulary (A3, the class-collapse amendment):
    /// TWO regions that BOTH run, over complementary row ranges — the
    /// prefix region over rows `[0, fast_rows)`, the tail region over
    /// `[fast_rows, N)`, with `fast_rows` (the hook-free prefix) a
    /// RUNTIME input of the fire, never a trace value. Regions are
    /// consecutive like a Guard's (prefix ops right after this op, then
    /// the tail's); an empty row range skips its region's launches —
    /// `fast_rows == N` is the classic all-fused fire, `fast_rows == 0`
    /// the all-hooked one, anything between the MIXED fire. This is
    /// plan.md Part 3 verbatim: the fast-path condition takes a row
    /// count where it used to take a boolean.
    ///
    /// A Peel may PRODUCE values (the fused decode-QKV's q): the values
    /// are the PEEL's outputs and both regions' launches bind disjoint
    /// row windows of the same buffers, recording no SSA outputs of
    /// their own — dataflow sees one producer, jointly lowered.
    ///
    /// `window` names WHICH runtime row count is the split
    /// ([`PeelWindow`]): the op is one region-word over every axis the
    /// scheduler seriates, not one word per axis.
    Peel {
        prefix_ops: u32,
        tail_ops: u32,
        #[serde(default, skip_serializing_if = "PeelWindow::is_hook_free")]
        window: PeelWindow,
    },
}

/// The runtime row count a [`OpKind::Peel`]'s split reads — the peel's
/// AXIS. Every axis the scheduler seriates into a prefix/suffix order
/// gets a variant here, and the regions' meaning is fixed by the op
/// (both run, complementary windows); only the split's SOURCE varies.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PeelWindow {
    /// The hook-free prefix (`fast_rows`, A3): the prefix region is the
    /// fused fast path, the tail the hook-visible general sequence.
    #[default]
    HookFreePrefix,
    /// The unmasked prefix (the spatial mask split, NS-2/NS-4): the
    /// prefix region serves the PLAIN decode rows, the tail the masked
    /// suffix rows — the work-sharing fire's one divergent op, stated.
    /// UNPLANNED (`unmasked_prefix_rows == u32::MAX`, or prepare kept
    /// the fire-level arm) means the tail region runs FULL-N with
    /// fire-level addressing and the prefix region is skipped: the
    /// fire-level custom dispatch is this peel's degenerate endpoint,
    /// not a separate op.
    UnmaskedPrefix,
}

impl PeelWindow {
    /// Serde skip for the default axis — every pre-window golden stays
    /// byte-identical.
    pub fn is_hook_free(&self) -> bool {
        matches!(self, PeelWindow::HookFreePrefix)
    }
}

/// Which implicit store an op addresses. Both stores are per-layer and
/// PER-REQUEST — the axis pie-application-plan.md §5.4 calls out — but they
/// are different resources with different lowerings: the paged KV cache
/// grows and is page-table-indirected, the recurrent store is fixed-size
/// slabs advanced in place (and is why RS fires are forced solo today).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum StateStore {
    /// The paged KV cache (`KvAppend` writes, `Attention` reads).
    KvCache,
    /// The GDN conv-window + recurrent-state slabs (`CausalConv1d` and
    /// `GatedDelta` each read AND advance their half).
    RecurrentState,
}

/// The state an op addresses: which store, at which layer. Derived from the
/// vocabulary by [`OpKind::state_ref`] — the honest marking of the
/// per-request state axis (module doc), with the store implicit exactly as
/// the KV cache always was.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct StateRef {
    pub store: StateStore,
    pub layer: u32,
}

impl OpKind {
    /// The implicit per-layer, per-request store this op addresses, if any.
    ///
    /// This is how the planner learns a trace touches per-request state
    /// without name-matching: `plan.ops.iter().any(|op|
    /// op.kind.state_ref().is_some_and(|s| s.store ==
    /// StateStore::RecurrentState))` is the traced-form statement of
    /// today's hand-maintained `touches_rs_buffer()`.
    pub fn state_ref(&self) -> Option<StateRef> {
        match *self {
            OpKind::KvAppend { layer } | OpKind::Attention { layer, .. } => Some(StateRef {
                store: StateStore::KvCache,
                layer,
            }),
            OpKind::Launch { state, .. } => state,
            OpKind::CausalConv1d { layer, .. } | OpKind::GatedDelta { layer } => Some(StateRef {
                store: StateStore::RecurrentState,
                layer,
            }),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Op {
    pub kind: OpKind,
    /// Values consumed, in operand order.
    pub inputs: Vec<ValueId>,
    /// Values produced (SplitQkv produces three, KvAppend none).
    pub outputs: Vec<ValueId>,
    /// The layer this op belongs to, or `None` for prologue/epilogue.
    pub layer: Option<u32>,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ValueInfo {
    pub shape: Shape,
    pub dtype: DType,
    /// The `dyn` marker: set on values whose content selects per-element
    /// structure (a [`OpKind::TopK`] expert assignment), `None` for
    /// ordinary data. Serde-skipped when absent so every pre-dyn traced
    /// form serializes byte-identically.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub dyn_axis: Option<DynAxis>,
}

/// The traced form of one family's forward pass, for one set of load-time
/// facts. Serializable so goldens can pin it and a driver can consume it
/// across the (future) C ABI.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct ForwardPlan {
    /// The family that traced this, plus a facts digest — a cache key, and
    /// the first thing a mismatch report prints.
    pub family: String,
    pub values: Vec<ValueInfo>,
    pub ops: Vec<Op>,
    /// STRUCTURAL S-3: the DECLARATION states the depth axis — every
    /// layer-tagged op of this trace may run over the full-depth prefix
    /// row window when the fire plans a depth split (layers `[k, L)` at
    /// rows `[0, split)`), and may be SKIPPED entirely on a uniform
    /// truncated fire. The trace is layer-unrolled while `k` is a
    /// runtime input, so the axis is a trace-level capability keyed on
    /// each op's own `layer` tag, not a region op at a static position
    /// (the [`OpKind::Peel`] doc's row-window vocabulary, applied
    /// per-layer). False for classes whose bodies cannot window
    /// (XQA-deployment, padded head dims, prefill shapes).
    #[serde(default, skip_serializing_if = "is_false")]
    pub depth_window: bool,
    /// Every seam the text stated ([`SeamStatement`]), in text order.
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub seams: Vec<SeamStatement>,
}

fn is_false(b: &bool) -> bool {
    !*b
}

impl ForwardPlan {
    /// DEPTH HAS NO SYNTAX (`.wiki/tart/dsl.md` ③, migration step 5).
    ///
    /// Every statement tagged with a layer is implicitly `rows(depth >
    /// l)`: it is skipped on a uniform truncated fire once its layer
    /// reaches `k`, and runs over the full-depth prefix rows on a union
    /// fire. The author writes nothing, and the IR carries no word —
    /// membership is the LAYER TAG plus the declaration's axis, which
    /// is what an `Op` already has.
    ///
    /// This replaces a per-op `DepthRole` enum whose `Windowed` variant
    /// was exactly this predicate, restated on every layer-tagged op of
    /// every trace.
    pub fn depth_windowed(&self, op: &Op) -> bool {
        self.depth_window && op.layer.is_some()
    }

    /// Does this op's kernel pair the depth PREFIX plan (and its
    /// dedicated workspace) on union tail layers, instead of the fire's
    /// own decode plan?
    ///
    /// The other half of the retired `DepthRole`, and it was never a
    /// property of the OP: it is a property of the KERNEL, so it lives
    /// in the kernel table beside `whole` and `needs`
    /// ([`crate::kernels::KernelSig::depth_prefix_plan`]).
    pub fn depth_prefix_plan(&self, op: &Op) -> bool {
        if !self.depth_windowed(op) {
            return false;
        }
        let OpKind::Launch { kernel, .. } = &op.kind else {
            return false;
        };
        crate::kernels::Backend::of_family(&self.family)
            .and_then(|b| crate::kernels::sig_in(b, kernel))
            .is_some_and(|k| k.depth_prefix_plan)
    }

    /// Ops belonging to layer `l`, in execution order.
    pub fn layer_ops(&self, l: u32) -> impl Iterator<Item = &Op> {
        self.ops.iter().filter(move |op| op.layer == Some(l))
    }
}

/// Records ops as a declaration executes. The declaration calls these
/// methods in computation order; the builder assigns value ids and keeps
/// the op list flat — structure (layers) is carried on the ops themselves.
pub struct TraceBuilder {
    family: String,
    values: Vec<ValueInfo>,
    ops: Vec<Op>,
    layer: Option<u32>,
    /// Seam statements in text order ([`SeamStatement`]).
    seams: Vec<SeamStatement>,
    /// Open [`Self::open_guard`] depth. Nesting is part of the
    /// vocabulary since A1 (north-star-dsl.md, the class-collapse
    /// amendment): a nested guard is an ordinary op inside a region —
    /// region lengths count it and its regions, the aux wire encoding
    /// is unchanged, the walk keeps a skip stack, the emitter recurses.
    guard_depth: u32,
    /// Open VALUE-PRODUCING regions ([`Self::open_guard`] /
    /// [`Self::open_peel`] with output shapes). A launch recorded while
    /// this is non-zero is a LOWERING of the enclosing construct's
    /// output — it binds that buffer and records no SSA output of its
    /// own — and a launch recorded at zero produces its own value.
    ///
    /// That is a property of WHERE THE STATEMENT IS, not of the kernel,
    /// and encoding it in the wrapper name is why `dsl::cuda` grew ten
    /// wrappers over five kernels (`.wiki/tart/dsl.md` ②, migration
    /// step 2). Tracking it here lets one wrapper serve both positions.
    value_region_depth: u32,
    /// V2 rung ②: the depth axis, DECLARED BY THE BODY
    /// ([`Self::declare_depth_window`]) instead of painted on after the
    /// trace (the review's smell — family.rs:64-91). While set, every
    /// layer-tagged op records its [`DepthRole`] at push time: the
    /// flashinfer decode dispatch swaps to the depth prefix plan on
    /// union tail layers, everything else windows.
    depth_axis: bool,
}

impl TraceBuilder {
    pub fn new(family: impl Into<String>) -> Self {
        Self {
            family: family.into(),
            values: Vec::new(),
            ops: Vec::new(),
            layer: None,
            seams: Vec::new(),
            guard_depth: 0,
            value_region_depth: 0,
            depth_axis: false,
        }
    }

    /// V2 rung ②: the body states the depth axis (the deployment gate
    /// lives with the statement, in the declaration text). Must precede
    /// the first layer-tagged op; the plan serializes with
    /// `depth_window` set and roles assigned exactly as the retired
    /// post-trace paint-over assigned them (the goldens pin it).
    pub fn declare_depth_window(&mut self) {
        debug_assert!(
            self.ops.iter().all(|op| op.layer.is_none()),
            "depth axis declared after layer-tagged ops were recorded"
        );
        self.depth_axis = true;
    }

    /// Bracket ops that belong to layer `l`.
    pub fn layer<T>(&mut self, l: u32, f: impl FnOnce(&mut Self) -> T) -> T {
        let previous = self.layer.replace(l);
        let out = f(self);
        self.layer = previous;
        out
    }

    /// The dsl surface's per-op layer tag ([`crate::dsl`] derives it from
    /// the handle an op touches rather than from this bracket).
    pub(crate) fn set_layer(&mut self, layer: Option<u32>) {
        self.layer = layer;
    }

    /// A value's shape, for dsl ops whose outputs mirror their inputs.
    pub(crate) fn value_shape(&self, id: ValueId) -> Shape {
        self.values[id as usize].shape.clone()
    }

    /// Open a [`OpKind::Guard`] chain: records the op with empty arms
    /// (and its output values, if any — created HERE so dataflow sees
    /// one producer whichever arm runs) and returns its index for
    /// [`Self::close_guard`] to patch once the dsl has run every region
    /// closure. Guards may NEST (A1): the inner guard op and its
    /// regions are contiguous ops inside the enclosing region, so the
    /// enclosing arm's length simply counts them.
    pub(crate) fn open_guard(&mut self, out_shapes: Vec<(Shape, DType)>) -> (usize, Vec<ValueId>) {
        self.guard_depth += 1;
        if !out_shapes.is_empty() {
            self.value_region_depth += 1;
        }
        let outs = self.push(
            OpKind::Guard {
                arms: Vec::new(),
                else_ops: 0,
            },
            vec![],
            out_shapes,
        );
        (self.ops.len() - 1, outs)
    }

    pub(crate) fn op_count_now(&self) -> usize {
        self.ops.len()
    }

    /// Is the statement being recorded a LOWERING of an enclosing
    /// construct's output rather than a producer of its own value?
    pub(crate) fn inside_value_region(&self) -> bool {
        self.value_region_depth > 0
    }

    /// Open an [`OpKind::Peel`]: records the op with empty region
    /// lengths (and its output values — created here so dataflow sees
    /// one producer, jointly lowered by both regions) and returns its
    /// index for [`Self::close_peel`]. Region ops follow consecutively,
    /// prefix first; guards may nest inside either region.
    pub(crate) fn open_peel(
        &mut self,
        out_shapes: Vec<(Shape, DType)>,
        window: PeelWindow,
    ) -> (usize, Vec<ValueId>) {
        let outs = self.push(
            OpKind::Peel {
                prefix_ops: 0,
                tail_ops: 0,
                window,
            },
            vec![],
            out_shapes,
        );
        if !outs.is_empty() {
            self.value_region_depth += 1;
        }
        (self.ops.len() - 1, outs)
    }

    pub(crate) fn close_peel(&mut self, peel_idx: usize, prefix: u32, tail: u32) {
        let OpKind::Peel {
            prefix_ops,
            tail_ops,
            ..
        } = &mut self.ops[peel_idx].kind
        else {
            panic!("close_peel: not a peel at {peel_idx}");
        };
        *prefix_ops = prefix;
        *tail_ops = tail;
        if !self.ops[peel_idx].outputs.is_empty() {
            self.value_region_depth -= 1;
        }
    }

    /// Patch a peel's AXIS after its arms have run — the axis is a
    /// consequence of the arm's row predicate ([`crate::dsl::RowPred`]),
    /// which is only known once the arm is written.
    pub(crate) fn set_peel_window(&mut self, peel_idx: usize, w: PeelWindow) {
        let OpKind::Peel { window, .. } = &mut self.ops[peel_idx].kind else {
            panic!("set_peel_window: not a peel at {peel_idx}");
        };
        *window = w;
    }

    pub(crate) fn push_hook_site(&mut self, stage: HookStage, layer: u32, q: ValueId) {
        self.push(OpKind::HookSite { stage, layer }, vec![q], vec![]);
    }

    /// Record that the text stated a seam, with the index of the op
    /// carrying it when one does.
    pub(crate) fn push_seam(&mut self, seam: &str, layer: Option<u32>, op: Option<u32>) {
        self.seams.push(SeamStatement {
            seam: seam.to_string(),
            layer,
            op,
        });
    }

    pub(crate) fn close_guard(&mut self, guard_idx: usize, arms: Vec<GuardArm>, else_ops: u32) {
        let OpKind::Guard {
            arms: a,
            else_ops: e,
        } = &mut self.ops[guard_idx].kind
        else {
            panic!("close_guard: not a guard at {guard_idx}");
        };
        *a = arms;
        *e = else_ops;
        assert!(self.guard_depth > 0, "close_guard without open_guard");
        self.guard_depth -= 1;
        if !self.ops[guard_idx].outputs.is_empty() {
            self.value_region_depth -= 1;
        }
    }

    /// The `+=` fold ([`crate::dsl`]): if `rhs` is the output of the op
    /// just recorded and that op is a plain unfused matmul, rewrite it to
    /// the `beta_one` accumulate against `residual` — id-neutral, the
    /// same op [`Self::matmul_add`] records directly. Returns false when
    /// the shape doesn't hold (rhs older than the last op, or the last op
    /// is not a plain matmul), in which case the caller lands the
    /// residual explicitly.
    pub(crate) fn try_fold_residual(&mut self, rhs: ValueId, residual: ValueId) -> bool {
        let Some(op) = self.ops.last_mut() else {
            return false;
        };
        let foldable = matches!(
            &op.kind,
            OpKind::Matmul {
                beta_one: false,
                selector: None,
                ..
            }
        ) && op.outputs == [rhs];
        if !foldable {
            return false;
        }
        let OpKind::Matmul { beta_one, .. } = &mut op.kind else {
            unreachable!("matched above");
        };
        *beta_one = true;
        op.inputs.push(residual);
        true
    }

    fn value(&mut self, shape: Shape, dtype: DType) -> ValueId {
        self.values.push(ValueInfo {
            shape,
            dtype,
            dyn_axis: None,
        });
        (self.values.len() - 1) as ValueId
    }

    /// Declare a fragment parameter: a value no op of this trace produces.
    ///
    /// Full-model traces never need this — `embed` starts them — but a
    /// traced *fragment* (`family::qwen3_5_moe_mlp_block`) takes the
    /// residual stream it lands on as a parameter, and stating that as a
    /// producer-less value keeps the dataflow honest: the composing
    /// declaration substitutes its own value where the fragment reads the
    /// parameter.
    pub fn input(&mut self, shape: Shape, dtype: DType) -> ValueId {
        self.value(shape, dtype)
    }

    fn push(
        &mut self,
        kind: OpKind,
        inputs: Vec<ValueId>,
        out_shapes: Vec<(Shape, DType)>,
    ) -> Vec<ValueId> {
        let outputs: Vec<ValueId> = out_shapes
            .into_iter()
            .map(|(shape, dtype)| self.value(shape, dtype))
            .collect();
        self.ops.push(Op {
            kind,
            inputs,
            outputs: outputs.clone(),
            layer: self.layer,
        });
        outputs
    }

    pub fn embed(&mut self, weight: &str, hidden: u32) -> ValueId {
        self.push(
            OpKind::Embed {
                weight: weight.to_string(),
            },
            vec![],
            vec![(
                Shape(vec![Dim::Tokens, Dim::Const(hidden)]),
                DType::BF16,
            )],
        )[0]
    }

    pub fn matmul(&mut self, x: ValueId, weight: &str, out_width: u32) -> ValueId {
        self.matmul_inner(x, weight, out_width, false)
    }

    /// The residual-accumulate form: `out += x @ w^T` where `out` is the
    /// residual stream. Returns the (new SSA id of the) accumulated value.
    pub fn matmul_add(
        &mut self,
        x: ValueId,
        weight: &str,
        residual: ValueId,
        out_width: u32,
    ) -> ValueId {
        let out = self.matmul_inner(x, weight, out_width, true);
        // The residual is an input of the accumulate — record it so the
        // dataflow is honest even though the lowering is one GEMM.
        self.ops
            .last_mut()
            .expect("matmul_inner pushed")
            .inputs
            .push(residual);
        out
    }

    fn matmul_inner(
        &mut self,
        x: ValueId,
        weight: &str,
        out_width: u32,
        beta_one: bool,
    ) -> ValueId {
        let rows = self.values[x as usize].shape.0[0];
        self.push(
            OpKind::Matmul {
                weight: weight.to_string(),
                beta_one,
                selector: None,
            },
            vec![x],
            vec![(Shape(vec![rows, Dim::Const(out_width)]), DType::BF16)],
        )[0]
    }

    /// The expert-indexed matmul: `weight_template` names a weight bank
    /// (`layer.0.expert.{e}.gate_up`) and `selector` — a [`Self::topk`]
    /// index value, `[Tokens, k]` — resolves `{e}` per token. Each token
    /// row is multiplied against its k selected experts' weights, so the
    /// result is `[Tokens, k, out_width]` (the driver's route-expanded
    /// `[N*K, out]` scratch, kept factored because k is a load-time
    /// constant and Tokens is not). One op = one launch: the grouped
    /// gate_up/down GEMM of the hand-written MoE pass, whatever strategy
    /// (cuBLAS batched, aligned blocks, CUTLASS fused) the emitter picks.
    pub fn matmul_per_token(
        &mut self,
        x: ValueId,
        weight_template: &str,
        selector: ValueId,
        out_width: u32,
    ) -> ValueId {
        assert!(
            weight_template.contains("{e}"),
            "per-token matmul weight must be a template with an {{e}} slot, got {weight_template:?}"
        );
        assert_eq!(
            self.values[selector as usize].dyn_axis,
            Some(DynAxis::PerToken),
            "per-token matmul selector must be a dyn PerToken value"
        );
        let rows = self.values[x as usize].shape.0[0];
        let k = self.values[selector as usize].shape.0[1];
        self.push(
            OpKind::Matmul {
                weight: weight_template.to_string(),
                beta_one: false,
                selector: Some(selector),
            },
            // The selector is an input too — its content is consumed — and
            // by convention the last one, like matmul_add's residual.
            vec![x, selector],
            vec![(
                Shape(vec![rows, k, Dim::Const(out_width)]),
                DType::BF16,
            )],
        )[0]
    }

    pub fn rmsnorm(&mut self, x: ValueId, weight: &str, variant: NormVariant) -> ValueId {
        let shape = self.values[x as usize].shape.clone();
        self.push(
            OpKind::Rmsnorm {
                weight: weight.to_string(),
                variant,
            },
            vec![x],
            vec![(shape, DType::BF16)],
        )[0]
    }

    pub fn add_bias(&mut self, x: ValueId, weight: &str) -> ValueId {
        let shape = self.values[x as usize].shape.clone();
        self.push(
            OpKind::AddBias {
                weight: weight.to_string(),
            },
            vec![x],
            vec![(shape, DType::BF16)],
        )[0]
    }

    pub fn rmsnorm_per_head(
        &mut self,
        x: ValueId,
        weight: &str,
        head_dim: u32,
        variant: NormVariant,
    ) -> ValueId {
        let shape = self.values[x as usize].shape.clone();
        self.push(
            OpKind::RmsnormPerHead {
                weight: weight.to_string(),
                head_dim,
                variant,
            },
            vec![x],
            vec![(shape, DType::BF16)],
        )[0]
    }

    pub fn split_qkv(
        &mut self,
        packed: ValueId,
        q_width: u32,
        kv_width: u32,
    ) -> (ValueId, ValueId, ValueId) {
        let rows = self.values[packed as usize].shape.0[0];
        let out = self.push(
            OpKind::SplitQkv { q_width, kv_width },
            vec![packed],
            vec![
                (Shape(vec![rows, Dim::Const(q_width)]), DType::BF16),
                (Shape(vec![rows, Dim::Const(kv_width)]), DType::BF16),
                (Shape(vec![rows, Dim::Const(kv_width)]), DType::BF16),
            ],
        );
        (out[0], out[1], out[2])
    }

    /// Rope mutates Q and K in place; SSA-wise it produces two new values.
    pub fn rope(&mut self, q: ValueId, k: ValueId, kind: RopeKind) -> (ValueId, ValueId) {
        self.rope_inner(q, k, kind, None)
    }

    /// The partial-rotary form: only the first `rotary_dim` channels of
    /// each head rotate (`launch_rope_partial_bf16`; qwen3.5's
    /// `partial_rotary_factor` resolved to a channel count — see
    /// [`OpKind::Rope`]).
    pub fn rope_partial(
        &mut self,
        q: ValueId,
        k: ValueId,
        kind: RopeKind,
        rotary_dim: u32,
    ) -> (ValueId, ValueId) {
        self.rope_inner(q, k, kind, Some(rotary_dim))
    }

    fn rope_inner(
        &mut self,
        q: ValueId,
        k: ValueId,
        kind: RopeKind,
        partial: Option<u32>,
    ) -> (ValueId, ValueId) {
        let q_shape = self.values[q as usize].shape.clone();
        let k_shape = self.values[k as usize].shape.clone();
        let out = self.push(
            OpKind::Rope { kind, partial },
            vec![q, k],
            vec![(q_shape, DType::BF16), (k_shape, DType::BF16)],
        );
        (out[0], out[1])
    }

    pub fn kv_append(&mut self, layer: u32, k: ValueId, v: ValueId) {
        self.push(OpKind::KvAppend { layer }, vec![k, v], vec![]);
    }

    pub fn attention(&mut self, layer: u32, q: ValueId, q_width: u32) -> ValueId {
        self.push(
            OpKind::Attention { layer },
            vec![q],
            vec![(
                Shape(vec![Dim::Tokens, Dim::Const(q_width)]),
                DType::BF16,
            )],
        )[0]
    }

    /// A STATED kernel launch ([`OpKind::Launch`]) — the recording half of
    /// the raw kernel signatures in [`crate::dsl::cuda`]; declarations
    /// call those, never this.
    pub fn launch(
        &mut self,
        kernel: &str,
        weights: Vec<String>,
        state: Option<StateRef>,
        inputs: Vec<ValueId>,
        out_shapes: Vec<(Shape, DType)>,
    ) -> Vec<ValueId> {
        self.push(
            OpKind::Launch {
                kernel: kernel.to_string(),
                weights,
                state,
            },
            inputs,
            out_shapes,
        )
    }

    /// SwiGLU halves the trailing gate‖up dim and keeps every leading dim,
    /// so it covers both the dense `[Tokens, 2*inter]` activation and the
    /// route-expanded `[Tokens, k, 2*inter]` one (the driver's
    /// `chunked_swiglu` over `N*K` rows).
    pub fn swiglu(&mut self, packed: ValueId, inter: u32) -> ValueId {
        let mut shape = self.values[packed as usize].shape.clone();
        *shape.0.last_mut().expect("swiglu input has a trailing dim") = Dim::Const(inter);
        self.push(OpKind::Swiglu { inter }, vec![packed], vec![(shape, DType::BF16)])[0]
    }

    /// Router top-k: `(indices, weights)`, both `[Tokens, k]`. The indices
    /// are the trace's first `dyn` value ([`DynAxis::PerToken`]); the
    /// weights are already softmaxed and renormalized, because the launch
    /// this op mirrors (`launch_topk_softmax_bf16`) does all three.
    pub fn topk(&mut self, logits: ValueId, k: u32) -> (ValueId, ValueId) {
        let rows = self.values[logits as usize].shape.0[0];
        let out = self.push(
            OpKind::TopK { k },
            vec![logits],
            vec![
                (Shape(vec![rows, Dim::Const(k)]), DType::I32),
                (Shape(vec![rows, Dim::Const(k)]), DType::F32),
            ],
        );
        self.values[out[0] as usize].dyn_axis = Some(DynAxis::PerToken);
        (out[0], out[1])
    }

    /// The top-k combine: collapse `x` (`[Tokens, k, d]`) to `[Tokens, d]`
    /// under per-token `weights` (`[Tokens, k]`). Operand order: weights,
    /// then the value they weight.
    pub fn weighted_sum(&mut self, weights: ValueId, x: ValueId) -> ValueId {
        let x_shape = &self.values[x as usize].shape.0;
        let (rows, d) = (x_shape[0], x_shape[2]);
        let k = match self.values[weights as usize].shape.0[1] {
            Dim::Const(k) => k,
            other => panic!("weighted_sum weights must have a Const k dim, got {other:?}"),
        };
        self.push(
            OpKind::WeightedSum { k },
            vec![weights, x],
            vec![(Shape(vec![rows, d]), DType::BF16)],
        )[0]
    }

    /// The shared-expert landing: `base + sigmoid(gate) * x`. Operand
    /// order mirrors [`Self::residual_add`] — the fresh value first, the
    /// stream it lands on last — and the result is the (new SSA id of the)
    /// combined value.
    pub fn sigmoid_gate_add(&mut self, x: ValueId, gate: ValueId, base: ValueId) -> ValueId {
        let shape = self.values[base as usize].shape.clone();
        self.push(
            OpKind::SigmoidGateAdd,
            vec![x, gate, base],
            vec![(shape, DType::BF16)],
        )[0]
    }

    /// The two-way GDN split: packed `[rows, w0 + w1]` into `[rows, w0]`
    /// and `[rows, w1]` at `w0`.
    pub fn split_gdn(
        &mut self,
        packed: ValueId,
        width0: u32,
        width1: u32,
    ) -> (ValueId, ValueId) {
        let rows = self.values[packed as usize].shape.0[0];
        let out = self.push(
            OpKind::SplitGdn { width0, width1 },
            vec![packed],
            vec![
                (Shape(vec![rows, Dim::Const(width0)]), DType::BF16),
                (Shape(vec![rows, Dim::Const(width1)]), DType::BF16),
            ],
        );
        (out[0], out[1])
    }

    /// The interleaved per-head `[query | gate]` split of a 2×-wide gated
    /// q projection: packed `[rows, heads * 2 * head_dim]` into (q, gate),
    /// each `[rows, heads * head_dim]`. See [`OpKind::SplitQGate`] for why
    /// this is not a [`Self::split_gdn`] row split.
    pub fn split_q_gate(
        &mut self,
        packed: ValueId,
        heads: u32,
        head_dim: u32,
    ) -> (ValueId, ValueId) {
        let rows = self.values[packed as usize].shape.0[0];
        match self.values[packed as usize].shape.0[1] {
            Dim::Const(w) if w == 2 * heads * head_dim => {}
            other => panic!(
                "split_q_gate input width {other:?} must be 2 * {heads} * {head_dim}"
            ),
        }
        let half = Shape(vec![rows, Dim::Const(heads * head_dim)]);
        let out = self.push(
            OpKind::SplitQGate { heads, head_dim },
            vec![packed],
            vec![(half.clone(), DType::BF16), (half, DType::BF16)],
        );
        (out[0], out[1])
    }

    /// The multiply-only output gate: `out = x * sigmoid(gate)`, both
    /// operands the same shape ([`OpKind::SigmoidGateMul`] — no residual,
    /// unlike [`Self::sigmoid_gate_add`]).
    pub fn sigmoid_gate_mul(&mut self, x: ValueId, gate: ValueId) -> ValueId {
        let shape = self.values[x as usize].shape.clone();
        assert_eq!(
            shape, self.values[gate as usize].shape,
            "sigmoid_gate_mul operands must share a shape"
        );
        self.push(
            OpKind::SigmoidGateMul,
            vec![x, gate],
            vec![(shape, DType::BF16)],
        )[0]
    }

    /// Depthwise causal conv1d (+ fused SiLU) over the packed qkv, against
    /// layer `layer`'s per-request conv state. Shape-preserving.
    pub fn causal_conv1d(
        &mut self,
        layer: u32,
        qkv: ValueId,
        weight: &str,
        kernel: u32,
    ) -> ValueId {
        let shape = self.values[qkv as usize].shape.clone();
        self.push(
            OpKind::CausalConv1d {
                weight: weight.to_string(),
                layer,
                kernel,
            },
            vec![qkv],
            vec![(shape, DType::BF16)],
        )[0]
    }

    /// The post-conv GDN prep: `(q, k, v, g, beta)`, all f32, with q/k in
    /// the compact `[Tokens, key_heads, key_dim]` per-head layout and v in
    /// `[Tokens, value_heads, value_dim]`. Operand order `[qkv, a, b]` is
    /// the kernel's.
    #[allow(clippy::too_many_arguments)]
    pub fn gdn_prep(
        &mut self,
        qkv: ValueId,
        a: ValueId,
        b: ValueId,
        a_log: &str,
        dt_bias: &str,
        key_heads: u32,
        key_dim: u32,
        value_heads: u32,
        value_dim: u32,
    ) -> (ValueId, ValueId, ValueId, ValueId, ValueId) {
        let rows = self.values[qkv as usize].shape.0[0];
        let qk = Shape(vec![rows, Dim::Const(key_heads), Dim::Const(key_dim)]);
        let out = self.push(
            OpKind::GdnPrep {
                a_log: a_log.to_string(),
                dt_bias: dt_bias.to_string(),
            },
            vec![qkv, a, b],
            vec![
                (qk.clone(), DType::F32),
                (qk, DType::F32),
                (
                    Shape(vec![rows, Dim::Const(value_heads), Dim::Const(value_dim)]),
                    DType::F32,
                ),
                (Shape(vec![rows, Dim::Const(value_heads)]), DType::F32),
                (Shape(vec![rows, Dim::Const(value_heads)]), DType::F32),
            ],
        );
        (out[0], out[1], out[2], out[3], out[4])
    }

    /// The gated-delta recurrence against layer `layer`'s per-request
    /// recurrent state. The core output keeps v's `[Tokens, Vh, Vd]` shape.
    pub fn gated_delta(
        &mut self,
        layer: u32,
        q: ValueId,
        k: ValueId,
        v: ValueId,
        g: ValueId,
        beta: ValueId,
    ) -> ValueId {
        let shape = self.values[v as usize].shape.clone();
        self.push(
            OpKind::GatedDelta { layer },
            vec![q, k, v, g, beta],
            vec![(shape, DType::F32)],
        )[0]
    }

    /// The gated RMSNorm landing: per-head norm of the rank-3 f32 core
    /// output, silu-gated by `gate`, flattened to `gate`'s `[Tokens,
    /// Vh * Vd]` bf16 shape (the fused fp32→bf16 conversion).
    pub fn rmsnorm_gated(&mut self, x: ValueId, gate: ValueId, weight: &str) -> ValueId {
        let x_elems: u32 = self.values[x as usize].shape.0[1..]
            .iter()
            .map(|d| match d {
                Dim::Const(c) => *c,
                other => panic!("rmsnorm_gated x must have Const head dims, got {other:?}"),
            })
            .product();
        let gate_shape = self.values[gate as usize].shape.clone();
        match gate_shape.0[1] {
            Dim::Const(w) if w == x_elems => {}
            other => panic!("rmsnorm_gated gate width {other:?} must equal x's flattened {x_elems}"),
        }
        self.push(
            OpKind::RmsnormGated {
                weight: weight.to_string(),
            },
            vec![x, gate],
            vec![(gate_shape, DType::BF16)],
        )[0]
    }

    /// The post-norm residual landing: `residual += x`. Operand order
    /// mirrors [`Self::matmul_add`] — the freshly computed value first,
    /// the residual stream it lands on appended — and the result is the
    /// (new SSA id of the) accumulated stream.
    pub fn residual_add(&mut self, x: ValueId, residual: ValueId) -> ValueId {
        let shape = self.values[residual as usize].shape.clone();
        self.push(
            OpKind::ResidualAdd,
            vec![x, residual],
            vec![(shape, DType::BF16)],
        )[0]
    }

    pub fn lm_head(&mut self, hidden: ValueId, weight: &str, vocab: u32) -> ValueId {
        self.push(
            OpKind::LmHead {
                weight: weight.to_string(),
            },
            vec![hidden],
            vec![(
                Shape(vec![Dim::Requests, Dim::Const(vocab)]),
                DType::F32,
            )],
        )[0]
    }

    pub fn finish(self) -> ForwardPlan {
        let plan = ForwardPlan {
            family: self.family,
            values: self.values,
            ops: self.ops,
            depth_window: self.depth_axis,
            seams: self.seams,
        };
        // ② The kernel signatures, checked (`.wiki/tart/dsl.md` ②,
        // migration step 3). A declaration is traced when the model
        // LOADS, so this is the load-time check the design asks for:
        // `whole` and the table's own coverage stop being rules a
        // reader has to know and become rules a build cannot violate.
        let mut problems = crate::kernels::check_plan(&plan);
        problems.extend(crate::dsl::seam::check_plan(&plan));
        assert!(
            problems.is_empty(),
            "signature violations in this declaration:\n  {}",
            problems.join("\n  ")
        );
        plan
    }
}
