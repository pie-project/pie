//! THE OP VOCABULARY — every kind of thing a traced form can say.

use super::*;
use serde::{Deserialize, Serialize};

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
    /// (llama_like.cpp). The kernel is 1:1 (`kernels::norm::add_bias_bf16`), so
    /// the semantic and lowered traces state the same op.
    AddBias { weight: String },
    /// Per-head RMSNorm of packed `[rows, heads * head_dim]` Q or K.
    /// `variant` selects the weight fold exactly as on [`OpKind::Rmsnorm`]:
    /// qwen3/olmo-style checkpoints multiply `w` directly (`Plain`), while
    /// qwen3.5's full-attention q/k norms fold `(1 + w)` (`Gemma` —
    /// `full_attn_layer_body` launches `kernels::norm::rmsnorm_gemma_bf16` over
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
    /// rest through (qwen3.5 full attention, `kernels::rope::rope_partial_bf16`);
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
    /// the hand-written pass (`kernels::norm::residual_add_bf16`).
    ResidualAdd,
    /// The WINDOW of a value along its leading dim — `x[index]`.
    ///
    /// Produces a value and launches NOTHING. gemma3n's AltUp is what
    /// asked for it: `altup_predict` produces all `k` streams and the
    /// layer body runs on ONE, which in `gemma3n.cpp` is a pointer
    /// offset — no kernel, no copy, `predictions + active * N * H`.
    ///
    /// It is an OP rather than a `Val` method because every value in this
    /// IR is an op's output, and because the thing it states is real: the
    /// text says which window the body reads, and a reader following the
    /// dataflow needs to see it. What it does NOT state is a launch, so
    /// `lower` emits no rectangle for it and `Buffers` gives its value an
    /// offset INTO the source's — which is the whole of its meaning.
    ///
    /// deepseek_v4's hyper-connections are rank-K too and never state
    /// this: they mix all K streams and select none. Two rank-K schemes,
    /// one new question.
    Select { index: u32 },
    /// Router top-k over per-token logits: for each token row, the `k`
    /// highest-scoring experts, with softmaxed-and-renormalized routing
    /// weights. Two results: the expert indices (`[Tokens, k]` i32, marked
    /// [`DynAxis::PerToken`] — the `dyn` value everything expert-indexed
    /// consumes) and the routing weights (`[Tokens, k]` f32). One op
    /// because it is one launch in the hand-written MoE pass
    /// (`kernels::moe::topk_softmax_bf16`: top-k + softmax + renormalize).
    TopK { k: u32 },
    /// Per-token combine of the k routed expert outputs:
    /// `out[t] = sum_j w[t, j] * x[t, j, :]`, collapsing `[Tokens, k, d]`
    /// to `[Tokens, d]`. The hand-written MoE pass's
    /// `kernels::moe::token_batched_weighted_sum_bf16`.
    WeightedSum { k: u32 },
    /// Shared-expert landing: `out = base + sigmoid(gate) * x`, the scalar
    /// per-token gate broadcast over the hidden dim. Operands `[x, gate,
    /// base]` — fresh value first, the stream it lands on last, the
    /// [`TraceBuilder::residual_add`] convention. One op because it is one
    /// launch (`kernels::mlp::sigmoid_scalar_gate_add_bf16`); the `[Tokens, 1]`
    /// gate logit comes from an ordinary `Matmul` the trace states
    /// separately, exactly as the hand-written pass launches it.
    SigmoidGateAdd,
    /// Split a packed `[rows, w0 + w1]` value at `w0` into two (two
    /// results). The GDN in-projection splits when the deployment binds the
    /// fused banks: `in_proj_qkvz` → (mixed qkv, z gate) and `in_proj_ba` →
    /// (b, a) — `kernels::layout::split_bf16_rows` and `kernels::layout::split_qwen_gdn_ba_bf16`
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
    /// The post-conv GDN prep (`kernels::ssm::qwen_gdn_post_conv_prep_bf16`): one
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
    /// Gated RMSNorm (`kernels::norm::rmsnorm_gated_fp32_in_bf16`): per (row,
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
    /// (`kernels::layout::split_q_gate_bf16`): the packed `[rows, heads * 2 *
    /// head_dim]` input carries, PER HEAD, `head_dim` query channels then
    /// `head_dim` gate channels — `q[n, h*d + i] = packed[n, h*2d + i]`,
    /// `gate[n, h*d + i] = packed[n, h*2d + d + i]` — so this is NOT a row
    /// split: [`OpKind::SplitGdn`] cuts a packed row at one offset, while
    /// this op de-interleaves at head granularity. Two results, q then
    /// gate, each `[rows, heads * head_dim]`.
    SplitQGate { heads: u32, head_dim: u32 },
    /// `out = x * sigmoid(gate)`, elementwise — qwen3.5 full attention's
    /// output gate (`kernels::mlp::sigmoid_gate_inplace_bf16`: `attn_out *=
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
        /// Scalar arguments the stated kernel takes that no operand
        /// shape gives — a rotary width, a padded head dim.
        ///
        /// A `Launch`'s two wire params are already spoken for (the
        /// state mark), and a scalar that has nowhere to ride is a
        /// scalar the DRIVER re-derives from its config. That is the
        /// thing this arc removes, so the channel exists rather than
        /// the derivation.
        ///
        /// Not a general escape hatch: a number belongs here only when
        /// it is a property of THIS STATEMENT that no shape spells.
        /// `eps`, `rope_theta` and a vocabulary size are properties of
        /// the deployment, and they stay the arm's parameters.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        params: Vec<u32>,
        /// Params the FIRE decides — `(index, Shape)`, meaning the scalar
        /// at `params[index]` is that shape's ELEMENT COUNT for the fire
        /// being lowered rather than the constant sitting there.
        ///
        /// The exception [`Self::Launch::params`]'s rule needs. A number
        /// belongs in `params` only when no shape spells it, and a number
        /// a shape DOES spell must not be copied there — a copy of an
        /// extent is an extent that can disagree with itself. But some
        /// kernels take an extent as a scalar because they bound a loop
        /// with it, and until the lowering could answer, the text had to
        /// write a constant and hope.
        ///
        /// Measured on `mlx-community/gpt-oss-20b-MXFP4-Q4`. `route_sort`
        /// scans `expert_ids[0 .. p.n]` and the text set `n` to
        /// `n_experts * experts_per_token` — a property of the DEPLOYMENT,
        /// 128 where the fire's pairs were 16. The kernel read 112 entries
        /// past that region and into `perm`, which the same kernel is
        /// concurrently writing, so the same fire over the same weights
        /// gave 0, 6 and 208,004 NaNs on three runs.
        ///
        /// So the channel exists rather than the constant: the lowering
        /// knows the fire's token count, the shapes already say what an
        /// extent is a function of, and the driver reads an ordinary
        /// param.
        #[serde(default, skip_serializing_if = "Vec::is_empty")]
        param_extents: Vec<(u8, Shape)>,
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
    Guard { arms: Vec<GuardArm>, else_ops: u32 },
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
