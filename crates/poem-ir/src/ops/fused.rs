use serde::{Deserialize, Serialize};

use crate::operands::Operands;
use crate::ops::elemwise::ModulateForm;
use crate::value::ValueId;

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PostNorm {
    pub weight: ValueId,
    pub plus_one: bool,
    pub eps: f32,
    pub out: ValueId,
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NormKind {
    Layernorm { eps: f32 },
    Rmsnorm { head_dim: u32, eps: f32 },
}

/// Ops only the compiler forms, from the primitive ones a model writes; each
/// is a kernel some backend ships.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum Fused {
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
    RmsnormRopePartialQ {
        x: ValueId,
        weight: ValueId,
        head_dim: u32,
        eps: f32,
        positions: ValueId,
        rotary_dim: u32,
        theta: f32,
        span: u32,
        y: ValueId,
        q_out: ValueId,
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
    /// mHC's mix projection of the f32-normed stream row, from the streams
    /// themselves: the norm (no weight) is a row scale, so it lands on the
    /// projection's sums and the normed row is never written.
    HcNormProject {
        streams: ValueId,
        eps: f32,
        weight: ValueId,
        stream_count: u32,
        mixes: ValueId,
    },
    /// A gated-delta layer's decode step from its in-projections' rows: the
    /// split of `qkvz` at `width` into `qkv` and `z`, the causal conv update
    /// of `qkv` into `conv`, the gate prep of `ba` into `gates`, and the
    /// recurrence into `y`. Every intermediate keeps its value, so a backend
    /// that cannot take the step whole runs the primitives over them.
    GdnStep {
        qkvz: ValueId,
        width: u32,
        qkv: ValueId,
        z: ValueId,
        conv_weight: ValueId,
        conv_state: ValueId,
        conv_width: u32,
        dilation: u32,
        conv: ValueId,
        ba: ValueId,
        dt_bias: ValueId,
        a_log: ValueId,
        gates: ValueId,
        state: ValueId,
        k_heads: u32,
        v_heads: u32,
        k_dim: u32,
        v_dim: u32,
        y: ValueId,
    },
    /// Qwen's gated attention prologue: the split of the `[q_h | gate_h]`
    /// projection `qg` into `q` and `gate`, q and k normed per head (`w + 1`)
    /// into `qn` and `kn`, both turned by the partial rotation into `q_out`
    /// and `k_out` (in place, as the rotation lands them), and `k_out` and
    /// `v` appended to `cache`. Every intermediate keeps its value, so a
    /// backend that cannot take the step whole runs the primitives over them.
    QGateNormRopeAppend {
        qg: ValueId,
        head_dim: u32,
        q: ValueId,
        gate: ValueId,
        q_norm: ValueId,
        q_eps: f32,
        qn: ValueId,
        k: ValueId,
        k_norm: ValueId,
        k_eps: f32,
        kn: ValueId,
        positions: ValueId,
        rotary_dim: u32,
        theta: f32,
        span: u32,
        q_out: ValueId,
        k_out: ValueId,
        v: ValueId,
        cache: ValueId,
        write_page: ValueId,
        write_offset: ValueId,
    },
    MatmulGeglu {
        act: ValueId,
        w: ValueId,
        intermediate: u32,
        packed: ValueId,
        y: ValueId,
    },
    /// A whole dense MLP: `gate_up` over `act`, swiglu, `down`. `packed` and
    /// `h` are the two intermediates, still written, so a backend may run
    /// the three primitives, or split the chain between engines.
    MlpSwiglu {
        act: ValueId,
        gate_up: ValueId,
        down: ValueId,
        intermediate: u32,
        packed: ValueId,
        h: ValueId,
        y: ValueId,
    },
    LmHeadSoftcap {
        act: ValueId,
        w: ValueId,
        cap: f32,
        y: ValueId,
        y_out: ValueId,
    },
    MatmulBias {
        act: ValueId,
        w: ValueId,
        bias: ValueId,
        y: ValueId,
        y_out: ValueId,
    },
    QkvFusedQknormRopeVnormWrite {
        packed: ValueId,
        positions: ValueId,
        q_norm_weight: ValueId,
        k_norm_weight: ValueId,
        /// The one epsilon both head norms and the v norm use.
        eps: f32,
        cache: ValueId,
        write_page: ValueId,
        write_offset: ValueId,
        kv_heads: u32,
        head_dim: u32,
        theta: f32,
        rotary_dim: u32,
        q: ValueId,
    },
}

impl Operands for Fused {
    fn inputs(&self, sink: &mut Vec<ValueId>) {
        match self {
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
            Self::RmsnormRopePartialQ {
                x,
                weight,
                positions,
                ..
            } => sink.extend([*x, *weight, *positions]),
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
            Self::HcNormProject {
                streams, weight, ..
            } => sink.extend([*streams, *weight]),
            Self::QGateNormRopeAppend {
                qg,
                q_norm,
                k,
                k_norm,
                positions,
                v,
                cache,
                write_page,
                write_offset,
                ..
            } => sink.extend([
                *qg,
                *q_norm,
                *k,
                *k_norm,
                *positions,
                *v,
                *cache,
                *write_page,
                *write_offset,
            ]),
            Self::GdnStep {
                qkvz,
                conv_weight,
                conv_state,
                ba,
                dt_bias,
                a_log,
                state,
                ..
            } => sink.extend([
                *qkvz,
                *conv_weight,
                *conv_state,
                *ba,
                *dt_bias,
                *a_log,
                *state,
            ]),
            Self::MatmulGeglu { act, w, .. } | Self::LmHeadSoftcap { act, w, .. } => {
                sink.extend([*act, *w]);
            }
            Self::MlpSwiglu {
                act, gate_up, down, ..
            } => sink.extend([*act, *gate_up, *down]),
            Self::MatmulBias { act, w, bias, .. } => sink.extend([*act, *w, *bias]),
            Self::QkvFusedQknormRopeVnormWrite {
                packed,
                positions,
                q_norm_weight,
                k_norm_weight,
                cache,
                write_page,
                write_offset,
                ..
            } => {
                sink.extend([
                    *packed,
                    *positions,
                    *q_norm_weight,
                    *k_norm_weight,
                    *cache,
                    *write_page,
                    *write_offset,
                ]);
            }
        }
    }
    fn outputs(&self, sink: &mut Vec<ValueId>) {
        match self {
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
            Self::RmsnormRopePartialQ { y, q_out, .. } => sink.extend([*y, *q_out]),
            Self::NormModulate { normed, y, .. } => sink.extend([*normed, *y]),
            Self::GatedResidualNormModulate {
                r_out, normed, out, ..
            } => {
                sink.extend([*r_out, *normed, *out]);
            }
            Self::HcNormProject { mixes, .. } => sink.push(*mixes),
            Self::GdnStep {
                qkv,
                z,
                conv,
                gates,
                y,
                ..
            } => sink.extend([*qkv, *z, *conv, *gates, *y]),
            Self::QGateNormRopeAppend {
                q,
                gate,
                qn,
                kn,
                q_out,
                k_out,
                ..
            } => sink.extend([*q, *gate, *qn, *kn, *q_out, *k_out]),
            Self::MatmulGeglu { packed, y, .. } => sink.extend([*packed, *y]),
            Self::MlpSwiglu { packed, h, y, .. } => sink.extend([*packed, *h, *y]),
            Self::LmHeadSoftcap { y, y_out, .. } => sink.extend([*y, *y_out]),
            Self::MatmulBias { y, y_out, .. } => sink.extend([*y, *y_out]),
            Self::QkvFusedQknormRopeVnormWrite { q, .. } => sink.push(*q),
        }
    }
    fn aliases(&self, sink: &mut Vec<(ValueId, ValueId)>) {
        match self {
            Self::ResidualAddRmsnorm { y_out, y, .. } => sink.push((*y_out, *y)),
            Self::RmsnormResidualAdd { y_out, y, .. } => sink.push((*y_out, *y)),
            Self::EmbedScaleAdd { y_out, y, .. } => sink.push((*y_out, *y)),
            Self::EmbedScaleAddSelect { .. } => {}
            Self::RmsnormRopePartialQ { q_out, y, .. } => sink.push((*q_out, *y)),
            Self::NormModulate { .. } => {}
            Self::GatedResidualNormModulate { r_out, r, .. } => sink.push((*r_out, *r)),
            Self::LmHeadSoftcap { y, y_out, .. } | Self::MatmulBias { y, y_out, .. } => {
                sink.push((*y_out, *y));
            }
            Self::QGateNormRopeAppend {
                qn,
                kn,
                q_out,
                k_out,
                ..
            } => sink.extend([(*q_out, *qn), (*k_out, *kn)]),
            Self::HcNormProject { .. }
            | Self::GdnStep { .. }
            | Self::MatmulGeglu { .. }
            | Self::MlpSwiglu { .. }
            | Self::QkvFusedQknormRopeVnormWrite { .. } => {}
        }
    }
    // The name a backend keys its kernel by (seat tables, `FUSED` lists), in
    // the family that kernel ships in.
    fn name(&self) -> &'static str {
        match self {
            Self::ResidualAddRmsnorm { .. } => "elementwise.residual_add_rmsnorm",
            Self::RmsnormResidualAdd { .. } => "elementwise.rmsnorm_residual_add",
            Self::EmbedScaleAdd { .. } => "elementwise.embed_scale_add",
            Self::EmbedScaleAddSelect { .. } => "elementwise.embed_scale_add_select",
            Self::RmsnormRopePartialQ { .. } => "elementwise.rmsnorm_rope_partial_q",
            Self::NormModulate { .. } => "elementwise.norm_modulate",
            Self::GatedResidualNormModulate { .. } => "elementwise.gated_residual_norm_modulate",
            Self::HcNormProject { .. } => "elementwise.hc_norm_project",
            Self::GdnStep { .. } => "attention.ssm_gdn_step",
            Self::QGateNormRopeAppend { .. } => "attention.q_gate_k_norm_rope_append",
            Self::MatmulGeglu { .. } => "linear.matmul_geglu",
            Self::MlpSwiglu { .. } => "linear.mlp_swiglu",
            Self::LmHeadSoftcap { .. } => "linear.lm_head_softcap",
            Self::MatmulBias { .. } => "linear.matmul_bias",
            Self::QkvFusedQknormRopeVnormWrite { .. } => {
                "custom_cuda.qkv_fused_qknorm_rope_vnorm_write"
            }
        }
    }
}
