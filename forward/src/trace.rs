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
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum NormVariant {
    Plain,
    Gemma,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RopeKind {
    Standard,
    /// Llama3/YaRN-style frequency scaling; parameters live in the facts.
    Yarn,
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
    /// Per-head RMSNorm of packed `[rows, heads * head_dim]` Q or K.
    RmsnormPerHead { weight: String, head_dim: u32 },
    /// Split packed QKV `[rows, q + 2kv]` into Q, K, V (three results).
    SplitQkv { q_width: u32, kv_width: u32 },
    /// Rotary embedding applied in place to Q and K (two operands).
    Rope { kind: RopeKind },
    /// Append this fire's K/V rows to the layer's paged cache.
    KvAppend { layer: u32 },
    /// Paged attention over the layer's cache. Opaque: the backend owns
    /// plan choice (decode/prefill/FA2/XQA) entirely.
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
}

/// Which implicit store an op addresses. Both stores are per-layer and
/// PER-REQUEST — the axis pie-application-plan.md §5.4 calls out — but they
/// are different resources with different lowerings: the paged KV cache
/// grows and is page-table-indirected, the recurrent store is fixed-size
/// slabs advanced in place (and is why RS fires are forced solo today).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
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
            OpKind::KvAppend { layer } | OpKind::Attention { layer } => Some(StateRef {
                store: StateStore::KvCache,
                layer,
            }),
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
}

impl ForwardPlan {
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
}

impl TraceBuilder {
    pub fn new(family: impl Into<String>) -> Self {
        Self {
            family: family.into(),
            values: Vec::new(),
            ops: Vec::new(),
            layer: None,
        }
    }

    /// Bracket ops that belong to layer `l`.
    pub fn layer<T>(&mut self, l: u32, f: impl FnOnce(&mut Self) -> T) -> T {
        let previous = self.layer.replace(l);
        let out = f(self);
        self.layer = previous;
        out
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

    pub fn rmsnorm_per_head(&mut self, x: ValueId, weight: &str, head_dim: u32) -> ValueId {
        let shape = self.values[x as usize].shape.clone();
        self.push(
            OpKind::RmsnormPerHead {
                weight: weight.to_string(),
                head_dim,
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
        let q_shape = self.values[q as usize].shape.clone();
        let k_shape = self.values[k as usize].shape.clone();
        let out = self.push(
            OpKind::Rope { kind },
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
        ForwardPlan {
            family: self.family,
            values: self.values,
            ops: self.ops,
        }
    }
}
