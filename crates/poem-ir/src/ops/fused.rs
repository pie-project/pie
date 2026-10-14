use serde::{Deserialize, Serialize};

use crate::operands::Operands;
use crate::ops::attn::Attention;
use crate::ops::elemwise::ModulateForm;
use crate::ops::linear::Linear;
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
    MatmulGeglu {
        act: ValueId,
        w: ValueId,
        intermediate: u32,
        packed: ValueId,
        y: ValueId,
    },
    /// The packed gate-up projection and its SwiGLU; `packed` still an
    /// output, read by nothing else.
    MatmulSwiglu {
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
    /// A routed gate-up select (`MoeMatmulSelect`, `..Bias` or `..Quant`,
    /// its `y` the packed gate|up rows) and the GLU of it (`MlpSwiglu`,
    /// `..Clamp`, `..ClampAlpha` or `MlpGegluTanhPacked`): a backend may land the GLU from the
    /// select's epilogue, or run the two. The packed rows are still an
    /// output, read by nothing else.
    MoeSelectGlu {
        select: Box<Linear>,
        glu: Box<Linear>,
    },
    /// A log-sum-exp attention (`DecodeLse` or `PrefillLse`) and the sinks
    /// over its output (`Sink`): a backend lands the sinked output in
    /// `o_out`, in the attention's fold of a split or after it. The
    /// attention's `o` (read by nothing else) and `lse` are still outputs.
    LseSink {
        attn: Box<Attention>,
        sink: ValueId,
        head_dim: u32,
        o_out: ValueId,
    },
    /// YaRN over q and k, and the rotated k with its v landed in the paged
    /// cache at each row's page and offset (`kv_append`); `k_out` is read by
    /// nothing else.
    RopeYarnAppend {
        q: ValueId,
        k: ValueId,
        v: ValueId,
        positions: ValueId,
        head_dim: u32,
        theta: f32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        attention_factor: f32,
        original_max_position: u32,
        interleaved: bool,
        cache: ValueId,
        write_page: ValueId,
        write_offset: ValueId,
        q_out: ValueId,
        k_out: ValueId,
    },
    /// A routed MLP: the gate-up select and its GLU (as in `MoeSelectGlu`),
    /// the down select of the GLU on the same routes, and the weighted sum
    /// of its rows a token (`MoeWeightedSum`), and where it follows, the
    /// shared expert's row added under its sigmoid gate
    /// (`MoeSigmoidGateAdd`). A backend may keep the rows between in its own
    /// layout, or run the ops. The packed, GLU and routed rows (and the sum,
    /// under a gate) are still outputs, read by nothing else.
    MoeExperts {
        select: Box<Linear>,
        glu: Box<Linear>,
        down: Box<Linear>,
        sum: Box<Linear>,
        gate_add: Option<Box<Linear>>,
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
            Self::MatmulGeglu { act, w, .. }
            | Self::MatmulSwiglu { act, w, .. }
            | Self::LmHeadSoftcap { act, w, .. } => {
                sink.extend([*act, *w]);
            }
            Self::MlpSwiglu {
                act, gate_up, down, ..
            } => sink.extend([*act, *gate_up, *down]),
            Self::MatmulBias { act, w, bias, .. } => sink.extend([*act, *w, *bias]),
            Self::MoeSelectGlu { select, .. } => select.inputs(sink),
            Self::RopeYarnAppend {
                q,
                k,
                v,
                positions,
                cache,
                write_page,
                write_offset,
                ..
            } => sink.extend([*q, *k, *v, *positions, *cache, *write_page, *write_offset]),
            Self::LseSink {
                attn, sink: sinks, ..
            } => {
                attn.inputs(sink);
                sink.push(*sinks);
            }
            Self::MoeExperts {
                select,
                down,
                sum,
                gate_add,
                ..
            } => {
                // The select's (x, bank, its bias, routes), the sum's weights,
                // then the down bank and its bias.
                select.inputs(sink);
                if let Linear::MoeWeightedSum { weights, .. } = **sum {
                    sink.push(weights);
                }
                match **down {
                    Linear::MoeMatmulSelect { bank, .. }
                    | Linear::MoeMatmulSelectQuant { bank, .. } => {
                        sink.push(bank);
                    }
                    Linear::MoeMatmulSelectBias { bank, bias, .. } => sink.extend([bank, bias]),
                    _ => {}
                }
                // Then the gated shared expert's row and its gate.
                if let Some(Linear::MoeSigmoidGateAdd { shared, gate, .. }) = gate_add.as_deref() {
                    sink.extend([*shared, *gate]);
                }
            }
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
            Self::MatmulGeglu { packed, y, .. } | Self::MatmulSwiglu { packed, y, .. } => {
                sink.extend([*packed, *y]);
            }
            Self::MlpSwiglu { packed, h, y, .. } => sink.extend([*packed, *h, *y]),
            Self::LmHeadSoftcap { y, y_out, .. } => sink.extend([*y, *y_out]),
            Self::MatmulBias { y, y_out, .. } => sink.extend([*y, *y_out]),
            Self::MoeSelectGlu { select, glu } => {
                select.outputs(sink);
                glu.outputs(sink);
            }
            Self::RopeYarnAppend { q_out, k_out, .. } => sink.extend([*q_out, *k_out]),
            Self::LseSink { attn, o_out, .. } => {
                attn.outputs(sink);
                sink.push(*o_out);
            }
            Self::MoeExperts {
                select,
                glu,
                down,
                sum,
                gate_add,
            } => {
                select.outputs(sink);
                glu.outputs(sink);
                down.outputs(sink);
                sum.outputs(sink);
                if let Some(gate_add) = gate_add {
                    gate_add.outputs(sink);
                }
            }
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
            Self::RopeYarnAppend {
                q_out, q, k_out, k, ..
            } => sink.extend([(*q_out, *q), (*k_out, *k)]),
            // The sinked output lands in a plane of its own: the
            // attention's `o` is written by nothing a reader sees.
            Self::LseSink { attn, .. } => attn.aliases(sink),
            Self::NormModulate { .. } => {}
            Self::GatedResidualNormModulate { r_out, r, .. } => sink.push((*r_out, *r)),
            Self::LmHeadSoftcap { y, y_out, .. } | Self::MatmulBias { y, y_out, .. } => {
                sink.push((*y_out, *y));
            }
            Self::MatmulGeglu { .. }
            | Self::MatmulSwiglu { .. }
            | Self::MlpSwiglu { .. }
            | Self::MoeSelectGlu { .. }
            | Self::MoeExperts { .. }
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
            Self::MatmulGeglu { .. } => "linear.matmul_geglu",
            Self::MatmulSwiglu { .. } => "linear.matmul_swiglu",
            Self::MlpSwiglu { .. } => "linear.mlp_swiglu",
            Self::LmHeadSoftcap { .. } => "linear.lm_head_softcap",
            Self::MatmulBias { .. } => "linear.matmul_bias",
            Self::MoeSelectGlu { .. } => "linear.moe_select_glu",
            Self::RopeYarnAppend { .. } => "elementwise.rope_yarn_append",
            Self::LseSink { .. } => "attention.lse_sink",
            Self::MoeExperts { .. } => "linear.moe_experts",
            Self::QkvFusedQknormRopeVnormWrite { .. } => {
                "custom_cuda.qkv_fused_qknorm_rope_vnorm_write"
            }
        }
    }
}
