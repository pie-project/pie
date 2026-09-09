use serde::{Deserialize, Serialize};

use crate::operands::Operands;
use crate::value::ValueId;

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Yarn {
    pub factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub original_max_position: u32,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PostNorm {
    pub weight: ValueId,
    pub plus_one: bool,
    pub eps: f32,
    pub out: ValueId,
}

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
    LayernormNoScale {
        x: ValueId,
        eps: f32,
        y: ValueId,
    },
    Layernorm {
        x: ValueId,
        weight: ValueId,
        bias: ValueId,
        eps: f32,
        y: ValueId,
    },
    RmsnormGroupedPlusOne {
        x: ValueId,
        weight: ValueId,
        group: u32,
        eps: f32,
        y: ValueId,
    },
    Clamp {
        x: ValueId,
        lo: f32,
        hi: f32,
        x_out: ValueId,
    },
    ClampLearned {
        x: ValueId,
        lo: ValueId,
        hi: ValueId,
        x_out: ValueId,
    },
    RmsnormGated {
        x: ValueId,
        gate: ValueId,
        weight: ValueId,
        head_dim: u32,
        eps: f32,
        act: GateActivation,
        y: ValueId,
    },
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
    ResidualAddRmsnorm {
        x: ValueId,
        y: ValueId,
        y_out: ValueId,
        weight: ValueId,
        plus_one: bool,
        eps: f32,
        out: ValueId,
    },
    RmsnormResidualAdd {
        x: ValueId,
        weight: ValueId,
        eps: f32,
        t: ValueId,
        y: ValueId,
        y_out: ValueId,
        scale: Option<(ValueId, ValueId)>,
        post: Option<PostNorm>,
    },
    EmbedScaleAdd {
        ids: ValueId,
        table: ValueId,
        vocab: u32,
        e: ValueId,
        embed_scale: f32,
        e_scaled: ValueId,
        y: ValueId,
        y_out: ValueId,
        out_scale: f32,
        y_scaled: ValueId,
    },
    EmbedScaleAddSelect {
        ids: ValueId,
        table: ValueId,
        vocab: u32,
        e: ValueId,
        embed_scale: f32,
        e_scaled: ValueId,
        stacked: ValueId,
        layer: u32,
        width: u32,
        y_out: ValueId,
        out_scale: f32,
        y_scaled: ValueId,
    },
    AddBias {
        bias: ValueId,
        out: ValueId,
        out_out: ValueId,
    },
    Standardize {
        x: ValueId,
        bias: ValueId,
        scale: ValueId,
        x_out: ValueId,
    },
    MulScalar {
        s: f32,
        x: ValueId,
        x_out: ValueId,
    },
    SiluScaled {
        s: f32,
        x: ValueId,
        x_out: ValueId,
    },
    Scale {
        s: ValueId,
        x: ValueId,
        x_out: ValueId,
    },
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
    RopeMrope {
        q: ValueId,
        k: ValueId,
        positions: ValueId,
        sections: [u32; 3],
        form: MropeForm,
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
    RmsnormRopePartialQ {
        x: ValueId,
        weight: ValueId,
        head_dim: u32,
        eps: f32,
        positions: ValueId,
        rotary_dim: u32,
        theta: f32,
        y: ValueId,
        q_out: ValueId,
    },
    RopePartialLast {
        q: ValueId,
        positions: ValueId,
        rotary_dim: u32,
        head_dim: u32,
        theta: f32,
        interleaved: bool,
        inverse: bool,
        yarn: Option<Yarn>,
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
    GateSigmoidMulHeads {
        x: ValueId,
        gate: ValueId,
        head_dim: u32,
        scale: f32,
        x_out: ValueId,
    },
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
    HcProject {
        normed: ValueId,
        weight: ValueId,
        stream_count: u32,
        mixes: ValueId,
    },
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
    HcFold {
        x: ValueId,
        streams: ValueId,
        post_mix: ValueId,
        comb_mix: ValueId,
        y: ValueId,
    },
    HcCollapse {
        mixes: ValueId,
        streams: ValueId,
        scale: ValueId,
        base: ValueId,
        stream_count: u32,
        hc_eps: f32,
        y: ValueId,
    },

    HcMix {
        gates: ValueId,
        normed: ValueId,
        streams: u32,
        y: ValueId,
    },
    HcInject {
        o: ValueId,
        gates: ValueId,
        streams: u32,
        hyper: ValueId,
        hyper_out: ValueId,
    },
    PleGate {
        key: ValueId,
        query: ValueId,
        value: ValueId,
        streams: u32,
        y: ValueId,
    },

    Modulate {
        x: ValueId,
        m: ValueId,
        lane_of_row: Option<ValueId>,
        form: ModulateForm,
        y: ValueId,
    },
    GatedResidualAdd {
        r: ValueId,
        g: ValueId,
        y: ValueId,
        lane_of_row: Option<ValueId>,
        r_out: ValueId,
    },
    NormModulate {
        x: ValueId,
        norm: NormKind,
        normed: ValueId,
        m: ValueId,
        lane_of_row: Option<ValueId>,
        form: ModulateForm,
        y: ValueId,
    },
    GatedResidualNormModulate {
        r: ValueId,
        g: ValueId,
        y: ValueId,
        lane_of_row: Option<ValueId>,
        r_out: ValueId,
        norm: NormKind,
        normed: ValueId,
        m: ValueId,
        form: ModulateForm,
        out: ValueId,
    },
    Sinusoid {
        t: ValueId,
        dim: u32,
        max_period: f32,
        flip_sin_cos: bool,
        scale: f32,
        y: ValueId,
    },
    RelativeBucketBias {
        embedding: ValueId,
        max_len: u32,
        num_buckets: u32,
        max_distance: f32,
        bidirectional: bool,
        y: ValueId,
    },
    Silu {
        x: ValueId,
        x_out: ValueId,
    },
    Gelu {
        x: ValueId,
        tanh: bool,
        x_out: ValueId,
    },
    Tanh {
        x: ValueId,
        x_out: ValueId,
    },
    Mul {
        x: ValueId,
        y: ValueId,
        z: ValueId,
    },
    Add {
        x: ValueId,
        y: ValueId,
        z: ValueId,
    },
    RopeAxes {
        x: ValueId,
        positions: ValueId,
        dims: [u32; 4],
        thetas: [f32; 4],
        form: RopeForm,
        rotary_dim: u32,
        head_dim: u32,
        x_out: ValueId,
    },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum ModulateForm {
    ScaleShift,
    Scale,
    TanhGate,
}

impl ModulateForm {
    #[must_use]
    pub fn slices(self) -> u64 {
        match self {
            ModulateForm::ScaleShift => 2,
            ModulateForm::Scale | ModulateForm::TanhGate => 1,
        }
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NormKind {
    Layernorm { eps: f32 },
    Rmsnorm { head_dim: u32, eps: f32 },
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum RopeForm {
    Interleaved,
    Neox,
    Split,
    SplitLadder,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GateActivation {
    Silu,
    Sigmoid,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MropeForm {
    Interleaved,
    Blocked,
    Split,
}

impl Operands for Elementwise {
    fn inputs(&self, sink: &mut Vec<ValueId>) {
        match self {
            Self::Rmsnorm { x, weight, .. } => sink.extend([*x, *weight]),
            Self::RmsnormPerHead { x, weight, .. } => sink.extend([*x, *weight]),
            Self::RmsnormPlusOne { x, weight, .. } => sink.extend([*x, *weight]),
            Self::RmsnormPerHeadPlusOne { x, weight, .. } => sink.extend([*x, *weight]),
            Self::RmsnormGroupedPlusOne { x, weight, .. } => sink.extend([*x, *weight]),
            Self::RmsnormNoScale { x, .. } => sink.push(*x),
            Self::LayernormNoScale { x, .. } => sink.push(*x),
            Self::Layernorm {
                x, weight, bias, ..
            } => sink.extend([*x, *weight, *bias]),
            Self::Clamp { x, .. } => sink.push(*x),
            Self::ClampLearned { x, lo, hi, .. } => sink.extend([*x, *lo, *hi]),
            Self::RmsnormGated {
                x, gate, weight, ..
            } => sink.extend([*x, *gate, *weight]),
            Self::RmsnormGatedBy {
                x, gate, weight, ..
            } => sink.extend([*x, *gate, *weight]),
            Self::ResidualAdd { x, y, .. } => sink.extend([*x, *y]),
            Self::ResidualAddRmsnorm { x, y, weight, .. } => sink.extend([*x, *y, *weight]),
            Self::RmsnormResidualAdd {
                x,
                weight,
                y,
                scale,
                post,
                ..
            } => {
                sink.extend([*x, *weight, *y]);
                if let Some((s, _)) = scale {
                    sink.push(*s);
                }
                if let Some(post) = post {
                    sink.push(post.weight);
                }
            }
            Self::EmbedScaleAdd { ids, table, y, .. } => sink.extend([*ids, *table, *y]),
            Self::EmbedScaleAddSelect {
                ids,
                table,
                stacked,
                ..
            } => sink.extend([*ids, *table, *stacked]),
            Self::AddBias { bias, out, .. } => sink.extend([*bias, *out]),
            Self::Standardize { x, bias, scale, .. } => sink.extend([*x, *bias, *scale]),
            Self::MulScalar { x, .. } => sink.push(*x),
            Self::SiluScaled { x, .. } => sink.push(*x),
            Self::Scale { s, x, .. } => sink.extend([*s, *x]),
            Self::ResBlend {
                prefix,
                blocks,
                weight,
                proj,
                ..
            } => {
                sink.push(*prefix);
                sink.extend_from_slice(blocks);
                sink.push(*weight);
                sink.push(*proj);
            }
            Self::RopeFull {
                q, k, positions, ..
            } => sink.extend([*q, *k, *positions]),
            Self::RopePartial {
                q, k, positions, ..
            } => sink.extend([*q, *k, *positions]),
            Self::RopeMrope {
                q, k, positions, ..
            } => sink.extend([*q, *k, *positions]),
            Self::RopePartialQ { q, positions, .. } => sink.extend([*q, *positions]),
            Self::RmsnormRopePartialQ {
                x,
                weight,
                positions,
                ..
            } => sink.extend([*x, *weight, *positions]),
            Self::RopePartialLast { q, positions, .. } => sink.extend([*q, *positions]),
            Self::RopeYarn {
                q, k, positions, ..
            } => sink.extend([*q, *k, *positions]),
            Self::GateSigmoidMul { x, gate, .. } => sink.extend([*x, *gate]),
            Self::GateSigmoidMulHeads { x, gate, .. } => sink.extend([*x, *gate]),
            Self::HcExpand { x, .. } => sink.push(*x),
            Self::HcRmsnormF32 { streams, .. } => sink.push(*streams),
            Self::HcProject { normed, weight, .. } => sink.extend([*normed, *weight]),
            Self::HcGates {
                normed,
                streams,
                scale,
                base,
                ..
            } => {
                sink.extend([*normed, *streams, *scale, *base]);
            }
            Self::HcFold {
                x,
                streams,
                post_mix,
                comb_mix,
                ..
            } => {
                sink.extend([*x, *streams, *post_mix, *comb_mix]);
            }
            Self::HcCollapse {
                mixes,
                streams,
                scale,
                base,
                ..
            } => {
                sink.extend([*mixes, *streams, *scale, *base]);
            }
            Self::HcMix { gates, normed, .. } => sink.extend([*gates, *normed]),
            Self::HcInject {
                o, gates, hyper, ..
            } => sink.extend([*o, *gates, *hyper]),
            Self::PleGate {
                key, query, value, ..
            } => sink.extend([*key, *query, *value]),
            Self::Modulate {
                x, m, lane_of_row, ..
            } => {
                sink.extend([*x, *m]);
                sink.extend(*lane_of_row);
            }
            Self::GatedResidualAdd {
                r,
                g,
                y,
                lane_of_row,
                ..
            } => {
                sink.extend([*r, *g, *y]);
                sink.extend(*lane_of_row);
            }
            Self::NormModulate {
                x, m, lane_of_row, ..
            } => {
                sink.extend([*x, *m]);
                sink.extend(*lane_of_row);
            }
            Self::GatedResidualNormModulate {
                r,
                g,
                y,
                m,
                lane_of_row,
                ..
            } => {
                sink.extend([*r, *g, *y, *m]);
                sink.extend(*lane_of_row);
            }
            Self::Sinusoid { t, .. } => sink.push(*t),
            Self::RelativeBucketBias { embedding, .. } => sink.push(*embedding),
            Self::Silu { x, .. } => sink.push(*x),
            Self::Gelu { x, .. } => sink.push(*x),
            Self::Tanh { x, .. } => sink.push(*x),
            Self::Mul { x, y, .. } => sink.extend([*x, *y]),
            Self::Add { x, y, .. } => sink.extend([*x, *y]),
            Self::RopeAxes { x, positions, .. } => sink.extend([*x, *positions]),
        }
    }
    fn outputs(&self, sink: &mut Vec<ValueId>) {
        match self {
            Self::Rmsnorm { y, .. } => sink.push(*y),
            Self::RmsnormPerHead { y, .. } => sink.push(*y),
            Self::RmsnormPlusOne { y, .. } => sink.push(*y),
            Self::RmsnormPerHeadPlusOne { y, .. } => sink.push(*y),
            Self::RmsnormGroupedPlusOne { y, .. } => sink.push(*y),
            Self::RmsnormNoScale { y, .. } => sink.push(*y),
            Self::LayernormNoScale { y, .. } => sink.push(*y),
            Self::Layernorm { y, .. } => sink.push(*y),
            Self::Clamp { x_out, .. } => sink.push(*x_out),
            Self::ClampLearned { x_out, .. } => sink.push(*x_out),
            Self::RmsnormGated { y, .. } => sink.push(*y),
            Self::RmsnormGatedBy { y, .. } => sink.push(*y),
            Self::ResidualAdd { y_out, .. } => sink.push(*y_out),
            Self::ResidualAddRmsnorm { y_out, out, .. } => sink.extend([*y_out, *out]),
            Self::RmsnormResidualAdd {
                t,
                y_out,
                scale,
                post,
                ..
            } => {
                sink.extend([*t, *y_out]);
                if let Some((_, scaled)) = scale {
                    sink.push(*scaled);
                }
                if let Some(post) = post {
                    sink.push(post.out);
                }
            }
            Self::EmbedScaleAdd {
                e,
                e_scaled,
                y_out,
                y_scaled,
                ..
            } => sink.extend([*e, *e_scaled, *y_out, *y_scaled]),
            Self::EmbedScaleAddSelect {
                e,
                e_scaled,
                y_out,
                y_scaled,
                ..
            } => sink.extend([*e, *e_scaled, *y_out, *y_scaled]),
            Self::AddBias { out_out, .. } => sink.push(*out_out),
            Self::Standardize { x_out, .. } => sink.push(*x_out),
            Self::MulScalar { x_out, .. } => sink.push(*x_out),
            Self::SiluScaled { x_out, .. } => sink.push(*x_out),
            Self::Scale { x_out, .. } => sink.push(*x_out),
            Self::ResBlend { y, .. } => sink.push(*y),
            Self::RopeFull { q_out, k_out, .. } => sink.extend([*q_out, *k_out]),
            Self::RopePartial { q_out, k_out, .. } => sink.extend([*q_out, *k_out]),
            Self::RopeMrope { q_out, k_out, .. } => sink.extend([*q_out, *k_out]),
            Self::RopePartialQ { q_out, .. } => sink.push(*q_out),
            Self::RmsnormRopePartialQ { y, q_out, .. } => sink.extend([*y, *q_out]),
            Self::RopePartialLast { q_out, .. } => sink.push(*q_out),
            Self::RopeYarn { q_out, k_out, .. } => sink.extend([*q_out, *k_out]),
            Self::GateSigmoidMul { x_out, .. } => sink.push(*x_out),
            Self::GateSigmoidMulHeads { x_out, .. } => sink.push(*x_out),
            Self::HcExpand { y, .. } => sink.push(*y),
            Self::HcRmsnormF32 { y, .. } => sink.push(*y),
            Self::HcProject { mixes, .. } => sink.push(*mixes),
            Self::HcGates {
                x,
                post_mix,
                comb_mix,
                ..
            } => sink.extend([*x, *post_mix, *comb_mix]),
            Self::HcFold { y, .. } => sink.push(*y),
            Self::HcCollapse { y, .. } => sink.push(*y),
            Self::HcMix { y, .. } => sink.push(*y),
            Self::HcInject { hyper_out, .. } => sink.push(*hyper_out),
            Self::PleGate { y, .. } => sink.push(*y),
            Self::Modulate { y, .. } => sink.push(*y),
            Self::GatedResidualAdd { r_out, .. } => sink.push(*r_out),
            Self::NormModulate { normed, y, .. } => sink.extend([*normed, *y]),
            Self::GatedResidualNormModulate {
                r_out, normed, out, ..
            } => {
                sink.extend([*r_out, *normed, *out]);
            }
            Self::Sinusoid { y, .. } => sink.push(*y),
            Self::RelativeBucketBias { y, .. } => sink.push(*y),
            Self::Silu { x_out, .. } => sink.push(*x_out),
            Self::Gelu { x_out, .. } => sink.push(*x_out),
            Self::Tanh { x_out, .. } => sink.push(*x_out),
            Self::Mul { z, .. } => sink.push(*z),
            Self::Add { z, .. } => sink.push(*z),
            Self::RopeAxes { x_out, .. } => sink.push(*x_out),
        }
    }
    fn aliases(&self, sink: &mut Vec<(ValueId, ValueId)>) {
        match self {
            Self::Rmsnorm { .. } => {}
            Self::RmsnormPerHead { .. } => {}
            Self::RmsnormPlusOne { .. } => {}
            Self::RmsnormPerHeadPlusOne { .. } => {}
            Self::RmsnormGroupedPlusOne { .. } => {}
            Self::RmsnormNoScale { .. } => {}
            Self::LayernormNoScale { .. } => {}
            Self::Layernorm { .. } => {}
            Self::Clamp { x_out, x, .. } => sink.push((*x_out, *x)),
            Self::ClampLearned { x_out, x, .. } => sink.push((*x_out, *x)),
            Self::RmsnormGated { .. } => {}
            Self::RmsnormGatedBy { .. } => {}
            Self::ResidualAdd { y_out, y, .. } => sink.push((*y_out, *y)),
            Self::ResidualAddRmsnorm { y_out, y, .. } => sink.push((*y_out, *y)),
            Self::RmsnormResidualAdd { y_out, y, .. } => sink.push((*y_out, *y)),
            Self::EmbedScaleAdd { y_out, y, .. } => sink.push((*y_out, *y)),
            Self::EmbedScaleAddSelect { .. } => {}
            Self::AddBias { out_out, out, .. } => sink.push((*out_out, *out)),
            Self::Standardize { x_out, x, .. } => sink.push((*x_out, *x)),
            Self::MulScalar { x_out, x, .. } => sink.push((*x_out, *x)),
            Self::SiluScaled { x_out, x, .. } => sink.push((*x_out, *x)),
            Self::Scale { x_out, x, .. } => sink.push((*x_out, *x)),
            Self::ResBlend { .. } => {}
            Self::RopeFull {
                q_out, q, k_out, k, ..
            } => sink.extend([(*q_out, *q), (*k_out, *k)]),
            Self::RopePartial {
                q_out, q, k_out, k, ..
            } => {
                sink.extend([(*q_out, *q), (*k_out, *k)]);
            }
            Self::RopeMrope {
                q_out, q, k_out, k, ..
            } => {
                sink.extend([(*q_out, *q), (*k_out, *k)]);
            }
            Self::RopePartialQ { q_out, q, .. } => sink.push((*q_out, *q)),
            Self::RmsnormRopePartialQ { q_out, y, .. } => sink.push((*q_out, *y)),
            Self::RopePartialLast { q_out, q, .. } => sink.push((*q_out, *q)),
            Self::RopeYarn {
                q_out, q, k_out, k, ..
            } => sink.extend([(*q_out, *q), (*k_out, *k)]),
            Self::GateSigmoidMul { x_out, x, .. } => sink.push((*x_out, *x)),
            Self::GateSigmoidMulHeads { x_out, x, .. } => sink.push((*x_out, *x)),
            Self::HcExpand { .. } => {}
            Self::HcRmsnormF32 { .. } => {}
            Self::HcProject { .. } => {}
            Self::HcGates { .. } => {}
            Self::HcFold { .. } => {}
            Self::HcCollapse { .. } => {}
            Self::HcMix { .. } => {}
            Self::HcInject {
                hyper_out, hyper, ..
            } => sink.push((*hyper_out, *hyper)),
            Self::PleGate { .. } => {}
            Self::Modulate { .. } => {}
            Self::GatedResidualAdd { r_out, r, .. } => sink.push((*r_out, *r)),
            Self::NormModulate { .. } => {}
            Self::GatedResidualNormModulate { r_out, r, .. } => sink.push((*r_out, *r)),
            Self::Sinusoid { .. } => {}
            Self::RelativeBucketBias { .. } => {}
            Self::Silu { x_out, x, .. } => sink.push((*x_out, *x)),
            Self::Gelu { x_out, x, .. } => sink.push((*x_out, *x)),
            Self::Tanh { x_out, x, .. } => sink.push((*x_out, *x)),
            Self::Mul { .. } => {}
            Self::Add { .. } => {}
            Self::RopeAxes { x_out, x, .. } => sink.push((*x_out, *x)),
        }
    }
    fn name(&self) -> &'static str {
        match self {
            Self::Rmsnorm { .. } => "elementwise.rmsnorm",
            Self::RmsnormPerHead { .. } => "elementwise.rmsnorm_per_head",
            Self::RmsnormPlusOne { .. } => "elementwise.rmsnorm_plus_one",
            Self::RmsnormPerHeadPlusOne { .. } => "elementwise.rmsnorm_per_head_plus_one",
            Self::RmsnormGroupedPlusOne { .. } => "elementwise.rmsnorm_grouped_plus_one",
            Self::RmsnormNoScale { .. } => "elementwise.rmsnorm_no_scale",
            Self::LayernormNoScale { .. } => "elementwise.layernorm_no_scale",
            Self::Layernorm { .. } => "elementwise.layernorm",
            Self::Clamp { .. } => "elementwise.clamp",
            Self::ClampLearned { .. } => "elementwise.clamp_learned",
            Self::RmsnormGated { .. } => "elementwise.rmsnorm_gated",
            Self::RmsnormGatedBy { .. } => "elementwise.rmsnorm_gated_by",
            Self::ResidualAdd { .. } => "elementwise.residual_add",
            Self::ResidualAddRmsnorm { .. } => "elementwise.residual_add_rmsnorm",
            Self::RmsnormResidualAdd { .. } => "elementwise.rmsnorm_residual_add",
            Self::EmbedScaleAdd { .. } => "elementwise.embed_scale_add",
            Self::EmbedScaleAddSelect { .. } => "elementwise.embed_scale_add_select",
            Self::AddBias { .. } => "elementwise.add_bias",
            Self::Standardize { .. } => "elementwise.standardize",
            Self::MulScalar { .. } => "elementwise.mul_scalar",
            Self::SiluScaled { .. } => "elementwise.silu_scaled",
            Self::Scale { .. } => "elementwise.scale",
            Self::ResBlend { .. } => "elementwise.res_blend",
            Self::RopeFull { .. } => "elementwise.rope_full",
            Self::RopePartial { .. } => "elementwise.rope_partial",
            Self::RopeMrope { .. } => "elementwise.rope_mrope",
            Self::RopePartialQ { .. } => "elementwise.rope_partial_q",
            Self::RmsnormRopePartialQ { .. } => "elementwise.rmsnorm_rope_partial_q",
            Self::RopePartialLast { .. } => "elementwise.rope_partial_last",
            Self::RopeYarn { .. } => "elementwise.rope_yarn",
            Self::GateSigmoidMul { .. } => "elementwise.gate_sigmoid_mul",
            Self::GateSigmoidMulHeads { .. } => "elementwise.gate_sigmoid_mul_heads",
            Self::HcExpand { .. } => "elementwise.hc_expand",
            Self::HcRmsnormF32 { .. } => "elementwise.hc_rmsnorm_f32",
            Self::HcProject { .. } => "elementwise.hc_project",
            Self::HcGates { .. } => "elementwise.hc_gates",
            Self::HcFold { .. } => "elementwise.hc_fold",
            Self::HcCollapse { .. } => "elementwise.hc_collapse",
            Self::HcMix { .. } => "elementwise.hc_mix",
            Self::HcInject { .. } => "elementwise.hc_inject",
            Self::PleGate { .. } => "elementwise.ple_gate",
            Self::Modulate { .. } => "elementwise.modulate",
            Self::GatedResidualAdd { .. } => "elementwise.gated_residual_add",
            Self::NormModulate { .. } => "elementwise.norm_modulate",
            Self::GatedResidualNormModulate { .. } => "elementwise.gated_residual_norm_modulate",
            Self::Sinusoid { .. } => "elementwise.sinusoid",
            Self::RelativeBucketBias { .. } => "elementwise.relative_bucket_bias",
            Self::Silu { .. } => "elementwise.silu",
            Self::Gelu { .. } => "elementwise.gelu",
            Self::Tanh { .. } => "elementwise.tanh",
            Self::Mul { .. } => "elementwise.mul",
            Self::Add { .. } => "elementwise.add",
            Self::RopeAxes { .. } => "elementwise.rope_axes",
        }
    }
}
