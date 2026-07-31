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
