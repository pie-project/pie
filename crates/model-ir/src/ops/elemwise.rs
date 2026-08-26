use serde::{Deserialize, Serialize};

use crate::operands::Operands;
use crate::value::ValueId;

/// Per-token math — tokens are independent. Per-token reductions like
/// rmsnorm's mean-of-squares belong here.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Elementwise {
    Rmsnorm {
        x: ValueId,
        weight: ValueId,
        eps: f32,
        y: ValueId,
    },
    RmsnormPerHead {
        x: ValueId,
        weight: ValueId,
        head_dim: u32,
        eps: f32,
        y: ValueId,
    },
    /// Scales by `weight + 1` (Gemma-style).
    RmsnormPlusOne {
        x: ValueId,
        weight: ValueId,
        eps: f32,
        y: ValueId,
    },
    RmsnormPerHeadPlusOne {
        x: ValueId,
        weight: ValueId,
        head_dim: u32,
        eps: f32,
        y: ValueId,
    },
    RmsnormNoScale {
        x: ValueId,
        head_dim: u32,
        eps: f32,
        y: ValueId,
    },
    /// `x` is f32; the norm is gated by `gate`, per group of `head_dim`.
    RmsnormGated {
        x: ValueId,
        gate: ValueId,
        weight: ValueId,
        head_dim: u32,
        eps: f32,
        y: ValueId,
    },
    /// Like `RmsnormGated`, but grouped by head count instead of head width.
    RmsnormGatedBy {
        x: ValueId,
        gate: ValueId,
        weight: ValueId,
        heads: u32,
        eps: f32,
        y: ValueId,
    },
    ResidualAdd {
        x: ValueId,
        y: ValueId,
        y_out: ValueId,
    },
    AddBias {
        bias: ValueId,
        out: ValueId,
        out_out: ValueId,
    },
    MulScalar {
        s: f32,
        x: ValueId,
        x_out: ValueId,
    },
    Scale {
        s: ValueId,
        x: ValueId,
        x_out: ValueId,
    },
    /// Norms the summed blocks against the prefix, then projects the blend.
    ResBlend {
        prefix: ValueId,
        blocks: Vec<ValueId>,
        weight: ValueId,
        eps: f32,
        proj: ValueId,
        y: ValueId,
    },
    RopeFull {
        q: ValueId,
        k: ValueId,
        positions: ValueId,
        head_dim: u32,
        theta: f32,
        interleaved: bool,
        q_out: ValueId,
        k_out: ValueId,
    },
    RopePartial {
        q: ValueId,
        k: ValueId,
        positions: ValueId,
        rotary_dim: u32,
        head_dim: u32,
        theta: f32,
        q_out: ValueId,
        k_out: ValueId,
    },
    RopePartialQ {
        q: ValueId,
        positions: ValueId,
        rotary_dim: u32,
        head_dim: u32,
        theta: f32,
        q_out: ValueId,
    },
    /// Partial rope over the last `rotary_dim` lanes of each head.
    RopePartialLast {
        q: ValueId,
        positions: ValueId,
        rotary_dim: u32,
        head_dim: u32,
        theta: f32,
        interleaved: bool,
        q_out: ValueId,
    },
    RopeYarn {
        q: ValueId,
        k: ValueId,
        positions: ValueId,
        head_dim: u32,
        theta: f32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        attention_factor: f32,
        original_max_position: u32,
        interleaved: bool,
        q_out: ValueId,
        k_out: ValueId,
    },
    GateSigmoidMul {
        x: ValueId,
        gate: ValueId,
        x_out: ValueId,
    },
    // Hyper-connections: residual streams expanded, mixed by learned gates, and
    // folded back layer by layer.
    /// Tiles `x` across `streams` residual streams.
    HcExpand {
        x: ValueId,
        streams: u32,
        y: ValueId,
    },
    HcRmsnormF32 {
        streams: ValueId,
        eps: f32,
        y: ValueId,
    },
    /// Computes the layer input `x` plus the post/comb mixing matrices.
    HcGates {
        normed: ValueId,
        streams: ValueId,
        scale: ValueId,
        base: ValueId,
        stream_count: u32,
        gate_eps: f32,
        alpha: f32,
        sinkhorn: u32,
        x: ValueId,
        post_mix: ValueId,
        comb_mix: ValueId,
    },
    /// Mixes the layer output back into the streams under the gate matrices.
    HcFold {
        x: ValueId,
        streams: ValueId,
        post_mix: ValueId,
        comb_mix: ValueId,
        y: ValueId,
    },
    // `Hc::Collapse` was deleted: no plane can fire it honestly (review R5).
}

impl Operands for Elementwise {
    fn inputs(&self, sink: &mut Vec<ValueId>) {
        match self {
            Self::Rmsnorm { x, weight, .. } => sink.extend([*x, *weight]),
            Self::RmsnormPerHead { x, weight, .. } => sink.extend([*x, *weight]),
            Self::RmsnormPlusOne { x, weight, .. } => sink.extend([*x, *weight]),
            Self::RmsnormPerHeadPlusOne { x, weight, .. } => sink.extend([*x, *weight]),
            Self::RmsnormNoScale { x, .. } => sink.push(*x),
            Self::RmsnormGated { x, gate, weight, .. } => sink.extend([*x, *gate, *weight]),
            Self::RmsnormGatedBy { x, gate, weight, .. } => sink.extend([*x, *gate, *weight]),
            Self::ResidualAdd { x, y, .. } => sink.extend([*x, *y]),
            Self::AddBias { bias, out, .. } => sink.extend([*bias, *out]),
            Self::MulScalar { x, .. } => sink.push(*x),
            Self::Scale { s, x, .. } => sink.extend([*s, *x]),
            Self::ResBlend { prefix, blocks, weight, proj, .. } => {
                sink.push(*prefix);
                sink.extend_from_slice(blocks);
                sink.push(*weight);
                sink.push(*proj);
            }
            Self::RopeFull { q, k, positions, .. } => sink.extend([*q, *k, *positions]),
            Self::RopePartial { q, k, positions, .. } => sink.extend([*q, *k, *positions]),
            Self::RopePartialQ { q, positions, .. } => sink.extend([*q, *positions]),
            Self::RopePartialLast { q, positions, .. } => sink.extend([*q, *positions]),
            Self::RopeYarn { q, k, positions, .. } => sink.extend([*q, *k, *positions]),
            Self::GateSigmoidMul { x, gate, .. } => sink.extend([*x, *gate]),
            Self::HcExpand { x, .. } => sink.push(*x),
            Self::HcRmsnormF32 { streams, .. } => sink.push(*streams),
            Self::HcGates { normed, streams, scale, base, .. } => {
                sink.extend([*normed, *streams, *scale, *base]);
            }
            Self::HcFold { x, streams, post_mix, comb_mix, .. } => {
                sink.extend([*x, *streams, *post_mix, *comb_mix]);
            }
        }
    }
    fn outputs(&self, sink: &mut Vec<ValueId>) {
        match self {
            Self::Rmsnorm { y, .. } => sink.push(*y),
            Self::RmsnormPerHead { y, .. } => sink.push(*y),
            Self::RmsnormPlusOne { y, .. } => sink.push(*y),
            Self::RmsnormPerHeadPlusOne { y, .. } => sink.push(*y),
            Self::RmsnormNoScale { y, .. } => sink.push(*y),
            Self::RmsnormGated { y, .. } => sink.push(*y),
            Self::RmsnormGatedBy { y, .. } => sink.push(*y),
            Self::ResidualAdd { y_out, .. } => sink.push(*y_out),
            Self::AddBias { out_out, .. } => sink.push(*out_out),
            Self::MulScalar { x_out, .. } => sink.push(*x_out),
            Self::Scale { x_out, .. } => sink.push(*x_out),
            Self::ResBlend { y, .. } => sink.push(*y),
            Self::RopeFull { q_out, k_out, .. } => sink.extend([*q_out, *k_out]),
            Self::RopePartial { q_out, k_out, .. } => sink.extend([*q_out, *k_out]),
            Self::RopePartialQ { q_out, .. } => sink.push(*q_out),
            Self::RopePartialLast { q_out, .. } => sink.push(*q_out),
            Self::RopeYarn { q_out, k_out, .. } => sink.extend([*q_out, *k_out]),
            Self::GateSigmoidMul { x_out, .. } => sink.push(*x_out),
            Self::HcExpand { y, .. } => sink.push(*y),
            Self::HcRmsnormF32 { y, .. } => sink.push(*y),
            Self::HcGates { x, post_mix, comb_mix, .. } => sink.extend([*x, *post_mix, *comb_mix]),
            Self::HcFold { y, .. } => sink.push(*y),
        }
    }
    fn aliases(&self, sink: &mut Vec<(ValueId, ValueId)>) {
        match self {
            Self::Rmsnorm { .. } => {}
            Self::RmsnormPerHead { .. } => {}
            Self::RmsnormPlusOne { .. } => {}
            Self::RmsnormPerHeadPlusOne { .. } => {}
            Self::RmsnormNoScale { .. } => {}
            Self::RmsnormGated { .. } => {}
            Self::RmsnormGatedBy { .. } => {}
            Self::ResidualAdd { y_out, y, .. } => sink.push((*y_out, *y)),
            Self::AddBias { out_out, out, .. } => sink.push((*out_out, *out)),
            Self::MulScalar { x_out, x, .. } => sink.push((*x_out, *x)),
            Self::Scale { x_out, x, .. } => sink.push((*x_out, *x)),
            Self::ResBlend { .. } => {}
            Self::RopeFull { q_out, q, k_out, k, .. } => sink.extend([(*q_out, *q), (*k_out, *k)]),
            Self::RopePartial { q_out, q, k_out, k, .. } => {
                sink.extend([(*q_out, *q), (*k_out, *k)]);
            }
            Self::RopePartialQ { q_out, q, .. } => sink.push((*q_out, *q)),
            Self::RopePartialLast { q_out, q, .. } => sink.push((*q_out, *q)),
            Self::RopeYarn { q_out, q, k_out, k, .. } => sink.extend([(*q_out, *q), (*k_out, *k)]),
            Self::GateSigmoidMul { x_out, x, .. } => sink.push((*x_out, *x)),
            Self::HcExpand { .. } => {}
            Self::HcRmsnormF32 { .. } => {}
            Self::HcGates { .. } => {}
            Self::HcFold { .. } => {}
        }
    }
    fn name(&self) -> &'static str {
        match self {
            Self::Rmsnorm { .. } => "elementwise.rmsnorm",
            Self::RmsnormPerHead { .. } => "elementwise.rmsnorm_per_head",
            Self::RmsnormPlusOne { .. } => "elementwise.rmsnorm_plus_one",
            Self::RmsnormPerHeadPlusOne { .. } => "elementwise.rmsnorm_per_head_plus_one",
            Self::RmsnormNoScale { .. } => "elementwise.rmsnorm_no_scale",
            Self::RmsnormGated { .. } => "elementwise.rmsnorm_gated",
            Self::RmsnormGatedBy { .. } => "elementwise.rmsnorm_gated_by",
            Self::ResidualAdd { .. } => "elementwise.residual_add",
            Self::AddBias { .. } => "elementwise.add_bias",
            Self::MulScalar { .. } => "elementwise.mul_scalar",
            Self::Scale { .. } => "elementwise.scale",
            Self::ResBlend { .. } => "elementwise.res_blend",
            Self::RopeFull { .. } => "elementwise.rope_full",
            Self::RopePartial { .. } => "elementwise.rope_partial",
            Self::RopePartialQ { .. } => "elementwise.rope_partial_q",
            Self::RopePartialLast { .. } => "elementwise.rope_partial_last",
            Self::RopeYarn { .. } => "elementwise.rope_yarn",
            Self::GateSigmoidMul { .. } => "elementwise.gate_sigmoid_mul",
            Self::HcExpand { .. } => "elementwise.hc_expand",
            Self::HcRmsnormF32 { .. } => "elementwise.hc_rmsnorm_f32",
            Self::HcGates { .. } => "elementwise.hc_gates",
            Self::HcFold { .. } => "elementwise.hc_fold",
        }
    }
}
