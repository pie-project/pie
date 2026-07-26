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
    Matmul { weight: String, beta_one: bool },
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
        self.values.push(ValueInfo { shape, dtype });
        (self.values.len() - 1) as ValueId
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
            },
            vec![x],
            vec![(Shape(vec![rows, Dim::Const(out_width)]), DType::BF16)],
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

    pub fn swiglu(&mut self, packed: ValueId, inter: u32) -> ValueId {
        let rows = self.values[packed as usize].shape.0[0];
        self.push(
            OpKind::Swiglu { inter },
            vec![packed],
            vec![(Shape(vec![rows, Dim::Const(inter)]), DType::BF16)],
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
