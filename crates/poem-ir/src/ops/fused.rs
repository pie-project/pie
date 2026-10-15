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
    /// A projection whose rows are cut at `width` straight after: a backend
    /// may land each side from its own slice of the weight, or run the
    /// matmul into `y` and cut it.
    MatmulSplitRows {
        act: ValueId,
        w: ValueId,
        width: u32,
        y: ValueId,
        left: ValueId,
        right: ValueId,
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
    /// DeepSeek-V4's attention whole: a windowed log-sum-exp attention
    /// (`window`, a `PrefillLse`), a compressed-pool one over the same q
    /// (`pool`, a `PoolLse` or `PoolLseSelected`), their merge (`merge`, a
    /// `MergeLse` of the two) and the sinks over the merged output: a
    /// backend lands the sinked output in `o_out` in one pass, or runs the
    /// four. Every intermediate is still an output, read by nothing else.
    PoolAttention {
        window: Box<Attention>,
        pool: Box<Attention>,
        merge: Box<Attention>,
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
    /// Qwen3.5's attention prologue: the q heads cut from their `[q | gate]`
    /// rows (`split_q_gate`), q and k normed per head with plus-one weights
    /// (`rmsnorm_per_head_plus_one`), their rotary prefixes turned
    /// (`rope_partial`), and the turned k with its v landed in the paged
    /// cache (`kv_append`). `q_split`, `q_normed`, `k_normed` and `k_out` are
    /// read by nothing else; a backend may leave them unwritten.
    QgQknormRopeAppend {
        qg: ValueId,
        k: ValueId,
        v: ValueId,
        positions: ValueId,
        q_norm: ValueId,
        k_norm: ValueId,
        head_dim: u32,
        q_eps: f32,
        k_eps: f32,
        rotary_dim: u32,
        theta: f32,
        span: u32,
        cache: ValueId,
        write_page: ValueId,
        write_offset: ValueId,
        q_split: ValueId,
        gate: ValueId,
        q_normed: ValueId,
        k_normed: ValueId,
        q_out: ValueId,
        k_out: ValueId,
    },
    /// A packed q/k/v row split (`split_qkv`), YaRN over its q and k, and
    /// the rotated k with its v landed in the paged cache (`kv_append`):
    /// `RopeYarnAppend` reading its three straight from the packed row.
    /// `k_out` and `v_out` (the split's k and v) are read by nothing else; a
    /// backend may leave them unwritten.
    SplitRopeYarnAppend {
        packed: ValueId,
        q_width: u32,
        kv_width: u32,
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
        v_out: ValueId,
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
            Self::MatmulSplitRows { act, w, .. } => sink.extend([*act, *w]),
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
            Self::SplitRopeYarnAppend {
                packed,
                positions,
                cache,
                write_page,
                write_offset,
                ..
            } => sink.extend([*packed, *positions, *cache, *write_page, *write_offset]),
            Self::QgQknormRopeAppend {
                qg,
                k,
                v,
                positions,
                q_norm,
                k_norm,
                cache,
                write_page,
                write_offset,
                ..
            } => sink.extend([
                *qg,
                *k,
                *v,
                *positions,
                *q_norm,
                *k_norm,
                *cache,
                *write_page,
                *write_offset,
            ]),
            Self::LseSink {
                attn, sink: sinks, ..
            } => {
                attn.inputs(sink);
                sink.push(*sinks);
            }
            // The window's (q, plan, cache), the pool's (q, positions,
            // requests, its selection, entries), then the sinks: the merge
            // reads only what the two land.
            Self::PoolAttention {
                window,
                pool,
                sink: sinks,
                ..
            } => {
                window.inputs(sink);
                pool.inputs(sink);
                sink.push(*sinks);
            }
            Self::MoeExperts {
                select,
                down,
                sum,
                gate_add,
                ..
            } => {
                // The select's (x, bank, its bias, routes, its input scales),
                // the sum's weights, then the down bank and its bias or input
                // scales.
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
                    Linear::MoeMatmulSelectActQuant {
                        bank, input_scales, ..
                    } => {
                        sink.push(bank);
                        sink.extend(input_scales);
                    }
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
            Self::MatmulSplitRows { y, left, right, .. } => sink.extend([*y, *left, *right]),
            Self::MoeSelectGlu { select, glu } => {
                select.outputs(sink);
                glu.outputs(sink);
            }
            Self::RopeYarnAppend { q_out, k_out, .. } => sink.extend([*q_out, *k_out]),
            Self::SplitRopeYarnAppend {
                q_out,
                k_out,
                v_out,
                ..
            } => sink.extend([*q_out, *k_out, *v_out]),
            Self::QgQknormRopeAppend {
                q_split,
                gate,
                q_normed,
                k_normed,
                q_out,
                k_out,
                ..
            } => sink.extend([*q_split, *gate, *q_normed, *k_normed, *q_out, *k_out]),
            Self::LseSink { attn, o_out, .. } => {
                attn.outputs(sink);
                sink.push(*o_out);
            }
            Self::PoolAttention {
                window,
                pool,
                merge,
                o_out,
                ..
            } => {
                window.outputs(sink);
                pool.outputs(sink);
                merge.outputs(sink);
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
            // The sinks scale the merged output where it lies, as `Sink`
            // does: the sinked output is the merge's plane.
            Self::PoolAttention {
                window,
                pool,
                merge,
                o_out,
                ..
            } => {
                window.aliases(sink);
                pool.aliases(sink);
                merge.aliases(sink);
                if let Attention::MergeLse { o, .. } = **merge {
                    sink.push((*o_out, o));
                }
            }
            Self::NormModulate { .. } => {}
            Self::GatedResidualNormModulate { r_out, r, .. } => sink.push((*r_out, *r)),
            Self::LmHeadSoftcap { y, y_out, .. } | Self::MatmulBias { y, y_out, .. } => {
                sink.push((*y_out, *y));
            }
            Self::MatmulGeglu { .. }
            | Self::MatmulSwiglu { .. }
            | Self::MatmulSplitRows { .. }
            | Self::MlpSwiglu { .. }
            | Self::MoeSelectGlu { .. }
            | Self::MoeExperts { .. }
            | Self::SplitRopeYarnAppend { .. }
            | Self::QkvFusedQknormRopeVnormWrite { .. } => {}
            // The rotation turns the normed rows in place.
            Self::QgQknormRopeAppend {
                q_out,
                q_normed,
                k_out,
                k_normed,
                ..
            } => sink.extend([(*q_out, *q_normed), (*k_out, *k_normed)]),
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
            Self::MatmulSplitRows { .. } => "linear.matmul_split_rows",
            Self::MoeSelectGlu { .. } => "linear.moe_select_glu",
            Self::RopeYarnAppend { .. } => "elementwise.rope_yarn_append",
            Self::SplitRopeYarnAppend { .. } => "elementwise.split_rope_yarn_append",
            Self::QgQknormRopeAppend { .. } => "elementwise.qg_qknorm_rope_append",
            Self::LseSink { .. } => "attention.lse_sink",
            Self::PoolAttention { .. } => "attention.pool_attention",
            Self::MoeExperts { .. } => "linear.moe_experts",
            Self::QkvFusedQknormRopeVnormWrite { .. } => {
                "custom_cuda.qkv_fused_qknorm_rope_vnorm_write"
            }
        }
    }
}
