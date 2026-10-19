use serde::{Deserialize, Serialize};

use crate::operands::Operands;
use crate::ops::attn::Attention;
use crate::ops::elemwise::{Elementwise, ModulateForm, Yarn};
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
    /// A head's RMS norm (scaled by `weight`, or scale-free) and the
    /// rotation of its last `rotary_dim` dims (`RopePartialLast`), the
    /// normed heads the rotation's own plane.
    NormRopeLast {
        x: ValueId,
        weight: Option<ValueId>,
        head_dim: u32,
        eps: f32,
        positions: ValueId,
        rotary_dim: u32,
        theta: f32,
        interleaved: bool,
        inverse: bool,
        yarn: Option<Yarn>,
        y: ValueId,
        q_out: ValueId,
    },
    /// `NormRopeLast` over a row one head wide (scaled), the rotated row
    /// then written to the cache at its page and offset
    /// (`KvAppendShared`); `y` is read by nothing else.
    NormRopeLastAppend {
        x: ValueId,
        weight: ValueId,
        head_dim: u32,
        eps: f32,
        positions: ValueId,
        rotary_dim: u32,
        theta: f32,
        interleaved: bool,
        inverse: bool,
        yarn: Option<Yarn>,
        y: ValueId,
        cache: ValueId,
        write_page: ValueId,
        write_offset: ValueId,
    },
    /// An index's ranking (`rank`, an `IndexTopk`) of queries whose tail is
    /// rotated just before it (`rotate`, a `RopePartialLast` of the queries
    /// the ranking alone reads): a backend may turn them as it loads them.
    IndexTopkRope {
        rotate: Box<Elementwise>,
        rank: Box<Attention>,
    },
    /// DeepSeek-V4's q (`NormRopeLast`, scale-free) and its kv plane
    /// (`NormRopeLastAppend`) of one layer, landed together.
    NormRopePair { q: Box<Fused>, kv: Box<Fused> },
    /// DeepSeek-V4's compressor at its boundaries: gathered entries
    /// (`PoolGather`) RMS-normed by `weight` into `latent`, its tail rotated
    /// (`rotate`) and appended to the pool (`append`); the gathered and normed
    /// rows are read by nothing else. `state`, if set, also lands the fire's
    /// compressor state written just before (a `PoolStateWrite`).
    PoolCompress {
        gather: Box<Attention>,
        weight: ValueId,
        eps: f32,
        latent: ValueId,
        rotate: Box<Elementwise>,
        append: Box<Attention>,
        state: Option<Box<Attention>>,
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
    /// A routed gate-up select (`MoeMatmulSelect`, `..Bias` or `..Quant`, `y`
    /// the packed gate|up rows) and its GLU (`MlpSwiglu`, `..Clamp`,
    /// `..ClampAlpha` or `MlpGegluTanhPacked`), landable from the select's
    /// epilogue. The packed rows are still an output, read by nothing else.
    MoeSelectGlu {
        select: Box<Linear>,
        glu: Box<Linear>,
    },
    /// A log-sum-exp attention (`DecodeLse` or `PrefillLse`) and its sinks
    /// (`Sink`), the sinked output landed in `o_out`. The attention's `o`
    /// (read by nothing else) and `lse` are still outputs.
    LseSink {
        attn: Box<Attention>,
        sink: ValueId,
        head_dim: u32,
        o_out: ValueId,
    },
    /// A sublayer's whole hyper-connection pre-mix: stream norm (`norm`, an
    /// `HcRmsnormF32`), mix projection (`project`, an `HcProject`), combined
    /// gates (`gates`, an `HcGates`) and the pre-mixed row's `rmsnorm`. The
    /// intermediates are still outputs, read by nothing else.
    HcPre {
        norm: Box<Elementwise>,
        project: Box<Elementwise>,
        gates: Box<Elementwise>,
        rmsnorm: Box<Elementwise>,
    },
    /// The combined hyper-connection gates (`gates`, an `HcGates`) and the RMS
    /// norm of the row they pre-mix (`norm`), landable in one pass. The
    /// pre-mixed row is still an output, read by nothing else.
    HcGatesNorm {
        gates: Box<Elementwise>,
        norm: Box<Elementwise>,
    },
    /// AttnRes's blend (`blend`, a `ResBlend`) and the RMS norm of its row
    /// (`norm`), landable in one pass. The blended row is still an output,
    /// read by nothing else.
    ResBlendRmsnorm {
        blend: Box<Elementwise>,
        norm: Box<Elementwise>,
    },
    /// A hyper-connection stream norm (`norm`, an `HcRmsnormF32`) and the mix
    /// projection it feeds (`project`, an `HcProject`), landable in one pass
    /// with the norm's per-row scale taken out of the dot. The normed rows are
    /// still an output, read by nothing else.
    HcPrenormProject {
        norm: Box<Elementwise>,
        project: Box<Elementwise>,
    },
    /// DeepSeek-V4's whole attention: windowed (`window`, a `PrefillLse`) and
    /// compressed-pool (`pool`, a `PoolLse` or `PoolLseSelected`) attention
    /// over one q, their `merge` (a `MergeLse`) and the sinks, landed in
    /// `o_out`; every intermediate is still an output, read by nothing else.
    /// `rotate` (the inverse rotation of `o_out`'s tail) lands too.
    PoolAttention {
        window: Box<Attention>,
        pool: Box<Attention>,
        merge: Box<Attention>,
        sink: ValueId,
        head_dim: u32,
        o_out: ValueId,
        rotate: Option<Box<Elementwise>>,
    },
    /// A latent attention's whole prefill: the optional cut of `q_b`'s rows
    /// (`split`, an `MlaSplitQB`), query absorb (`absorb_q`), scoring over
    /// every cached latent row (`prefill`) and output absorb (`absorb_out`)
    /// through one `kv_b`, whose rows a backend may land expanded (a k and v
    /// head a q head). Every intermediate is still an output, read by nothing
    /// else.
    MlaPrefillExpanded {
        split: Option<Box<Attention>>,
        absorb_q: Box<Attention>,
        prefill: Box<Attention>,
        absorb_out: Box<Attention>,
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
    /// Qwen3.5's attention prologue: q cut from `[q | gate]` (`split_q_gate`),
    /// q and k normed per head (`rmsnorm_per_head_plus_one`), their rotary
    /// prefixes turned (`rope_partial`) and k with v cached (`kv_append`).
    /// `q_split`, `q_normed`, `k_normed` and `k_out` may be left unwritten.
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
    /// `RopeYarnAppend` reading q, k and v straight from a packed row split by
    /// `split_qkv`. `k_out` and `v_out` (the split's k and v) may be left
    /// unwritten.
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
    /// A routed MLP: gate-up select and GLU (as in `MoeSelectGlu`), the down
    /// select on the same routes, the per-token weighted sum (`MoeWeightedSum`)
    /// and, where it follows, the sigmoid-gated shared expert
    /// (`MoeSigmoidGateAdd`). The rows between (and the sum, under a gate) are
    /// still outputs, read by nothing else. `norm`: each route's down row
    /// normed under a scale (`NormGate`) before the sum (Muse Spark's).
    MoeExperts {
        select: Box<Linear>,
        glu: Box<Linear>,
        down: Box<Linear>,
        norm: Option<Box<Elementwise>>,
        sum: Box<Linear>,
        gate_add: Option<Box<Linear>>,
    },
    /// `qkv` (a `QkvFusedQknormRopeVnormWrite`) on the first `width` columns
    /// of its `packed` rows, the rest of each row landing in `rest`: a row
    /// split (Muse Spark's qkv|gate) read in place.
    SplitQkvFused {
        width: u32,
        rest: ValueId,
        qkv: Box<Fused>,
    },
    /// A packed qkv row split, q and k normed per head (under weights, or
    /// none) where `qk_norm`, v normed without a scale where `v_norm` and
    /// scaled by `v_scale`, the first `rotary_dim / 2` pairs `span` apart of
    /// q and k turned (`rotary_dim` 0: none), and k and v written to the
    /// cache, the packed heads `widths` (q and k, v) wide and landing
    /// `head_dim` wide: Gemma 4's attention prologue, Muse Spark's and MiMo's.
    QkvFusedQknormRopeVnormWrite {
        packed: ValueId,
        /// The rows' positions, where they turn.
        positions: Option<ValueId>,
        q_norm_weight: Option<ValueId>,
        k_norm_weight: Option<ValueId>,
        v_norm: bool,
        /// The one epsilon both head norms and the v norm use.
        eps: f32,
        cache: ValueId,
        write_page: ValueId,
        write_offset: ValueId,
        kv_heads: u32,
        head_dim: u32,
        theta: f32,
        rotary_dim: u32,
        span: u32,
        qk_norm: bool,
        v_scale: f32,
        widths: [u32; 2],
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
            Self::NormRopeLast {
                x,
                weight,
                positions,
                ..
            } => {
                sink.extend([*x, *positions]);
                sink.extend(*weight);
            }
            Self::NormRopeLastAppend {
                x,
                weight,
                positions,
                cache,
                write_page,
                write_offset,
                ..
            } => sink.extend([*x, *weight, *positions, *cache, *write_page, *write_offset]),
            Self::NormRopePair { q, kv } => {
                q.inputs(sink);
                kv.inputs(sink);
            }
            // The raw queries and their positions, then the ranking's keys
            // and head weights.
            Self::IndexTopkRope { rotate, rank } => {
                rotate.inputs(sink);
                if let Attention::IndexTopk { keys, weights, .. } = **rank {
                    sink.push(keys);
                    sink.extend(weights);
                }
            }
            // The boundaries' positions and requests, the state's pages, the
            // norm's weight, the rope positions, the pool and its write
            // tables, then the gather's position bias where it has one.
            Self::PoolCompress {
                gather,
                weight,
                rotate,
                append,
                state,
                ..
            } => {
                if let Attention::PoolGather {
                    boundary_pos,
                    boundary_req,
                    pages,
                    ..
                } = **gather
                {
                    sink.extend([boundary_pos, boundary_req, pages]);
                }
                sink.push(*weight);
                if let Elementwise::RopePartialLast { positions, .. } = **rotate {
                    sink.push(positions);
                }
                if let Attention::PoolKvAppend {
                    pool,
                    write_page,
                    write_offset,
                    ..
                } = **append
                {
                    sink.extend([pool, write_page, write_offset]);
                }
                if let Attention::PoolGather { ape: Some(ape), .. } = **gather {
                    sink.push(ape);
                }
                if let Some(state) = state
                    && let Attention::PoolStateWrite {
                        kv,
                        score,
                        write_page,
                        write_offset,
                        ..
                    } = **state
                {
                    sink.extend([kv, score, write_page, write_offset]);
                }
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
            // The streams, the plane, the gates' scale and base, the weight.
            Self::HcPre {
                norm,
                project,
                gates,
                rmsnorm,
            } => {
                norm.inputs(sink);
                if let Elementwise::HcProject { weight, .. } = **project {
                    sink.push(weight);
                }
                if let Elementwise::HcGates { scale, base, .. } = **gates {
                    sink.extend([scale, base]);
                }
                if let Elementwise::Rmsnorm { weight, .. } = **rmsnorm {
                    sink.push(weight);
                }
            }
            Self::ResBlendRmsnorm { blend, norm } => {
                blend.inputs(sink);
                if let Elementwise::Rmsnorm { weight, .. } = **norm {
                    sink.push(weight);
                }
            }
            Self::HcGatesNorm { gates, norm } => {
                gates.inputs(sink);
                if let Elementwise::Rmsnorm { weight, .. } = **norm {
                    sink.push(weight);
                }
            }
            Self::HcPrenormProject { norm, project } => {
                norm.inputs(sink);
                if let Elementwise::HcProject { weight, .. } = **project {
                    sink.push(weight);
                }
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
            // `q_b` (or the query's `nope` half), `kv_b`, the plan, the
            // rope half where the rows come apart, the cache.
            Self::MlaPrefillExpanded {
                split,
                absorb_q,
                prefill,
                ..
            } => {
                if let Some(split) = split {
                    split.inputs(sink);
                } else if let Attention::MlaAbsorbQ { q_nope, .. } = **absorb_q {
                    sink.push(q_nope);
                }
                if let Attention::MlaAbsorbQ { kv_b, .. } = **absorb_q {
                    sink.push(kv_b);
                }
                if let Attention::MlaPrefill {
                    plan, q_pe, cache, ..
                } = **prefill
                {
                    sink.push(plan);
                    if split.is_none() {
                        sink.push(q_pe);
                    }
                    sink.push(cache);
                }
            }
            Self::MoeExperts {
                select,
                glu,
                down,
                norm,
                sum,
                gate_add,
            } => {
                // The select's (x, bank, its bias, routes, its input scales),
                // the route weights (the sum's, or the GLU's that weighs its
                // rows first), then the down bank and its bias or input
                // scales.
                select.inputs(sink);
                if let Linear::MoeWeightedSum { weights, .. } = **sum {
                    sink.push(weights);
                } else if let Linear::MlpSwigluClampRouted { weights, .. } = **glu {
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
                // Then the routes' norm scale.
                if let Some(Elementwise::NormGate { scale, .. }) = norm.as_deref() {
                    sink.extend(*scale);
                }
                // Then the gated shared expert's row and its gate, or the
                // shared row a plain sum adds.
                if let Some(Linear::MoeSigmoidGateAdd { shared, gate, .. }) = gate_add.as_deref() {
                    sink.extend([*shared, *gate]);
                }
                if let Linear::MoeSumAdd { shared, .. } = **sum {
                    sink.push(shared);
                }
            }
            Self::SplitQkvFused { qkv, .. } => qkv.inputs(sink),
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
                sink.extend([*packed, *cache, *write_page, *write_offset]);
                sink.extend(positions.iter().chain(q_norm_weight).chain(k_norm_weight));
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
            Self::NormRopeLast { y, q_out, .. } => sink.extend([*y, *q_out]),
            Self::NormRopeLastAppend { y, .. } => sink.push(*y),
            Self::NormRopePair { q, kv } => {
                q.outputs(sink);
                kv.outputs(sink);
            }
            Self::IndexTopkRope { rotate, rank } => {
                rotate.outputs(sink);
                rank.outputs(sink);
            }
            Self::PoolCompress {
                gather,
                latent,
                rotate,
                ..
            } => {
                gather.outputs(sink);
                sink.push(*latent);
                rotate.outputs(sink);
            }
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
            Self::HcPrenormProject { norm, project } => {
                norm.outputs(sink);
                project.outputs(sink);
            }
            Self::HcGatesNorm { gates, norm } => {
                gates.outputs(sink);
                norm.outputs(sink);
            }
            Self::ResBlendRmsnorm { blend, norm } => {
                blend.outputs(sink);
                norm.outputs(sink);
            }
            Self::HcPre {
                norm,
                project,
                gates,
                rmsnorm,
            } => {
                norm.outputs(sink);
                project.outputs(sink);
                gates.outputs(sink);
                rmsnorm.outputs(sink);
            }
            Self::PoolAttention {
                window,
                pool,
                merge,
                o_out,
                rotate,
                ..
            } => {
                window.outputs(sink);
                pool.outputs(sink);
                merge.outputs(sink);
                sink.push(*o_out);
                if let Some(rotate) = rotate {
                    rotate.outputs(sink);
                }
            }
            Self::MlaPrefillExpanded {
                split,
                absorb_q,
                prefill,
                absorb_out,
            } => {
                if let Some(split) = split {
                    split.outputs(sink);
                }
                absorb_q.outputs(sink);
                prefill.outputs(sink);
                absorb_out.outputs(sink);
            }
            Self::MoeExperts {
                select,
                glu,
                down,
                norm,
                sum,
                gate_add,
            } => {
                select.outputs(sink);
                glu.outputs(sink);
                down.outputs(sink);
                if let Some(norm) = norm {
                    norm.outputs(sink);
                }
                sum.outputs(sink);
                if let Some(gate_add) = gate_add {
                    gate_add.outputs(sink);
                }
            }
            Self::QkvFusedQknormRopeVnormWrite { q, .. } => sink.push(*q),
            Self::SplitQkvFused { rest, qkv, .. } => {
                qkv.outputs(sink);
                sink.push(*rest);
            }
        }
    }
    fn aliases(&self, sink: &mut Vec<(ValueId, ValueId)>) {
        match self {
            Self::ResidualAddRmsnorm { y_out, y, .. } => sink.push((*y_out, *y)),
            Self::RmsnormResidualAdd { y_out, y, .. } => sink.push((*y_out, *y)),
            Self::EmbedScaleAdd { y_out, y, .. } => sink.push((*y_out, *y)),
            Self::EmbedScaleAddSelect { .. } => {}
            Self::RmsnormRopePartialQ { q_out, y, .. } => sink.push((*q_out, *y)),
            Self::NormRopeLast { q_out, y, .. } => sink.push((*q_out, *y)),
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
                rotate,
                ..
            } => {
                window.aliases(sink);
                pool.aliases(sink);
                merge.aliases(sink);
                if let Attention::MergeLse { o, .. } = **merge {
                    sink.push((*o_out, o));
                }
                if let Some(rotate) = rotate {
                    rotate.aliases(sink);
                }
            }
            Self::NormModulate { .. }
            | Self::HcPrenormProject { .. }
            | Self::HcGatesNorm { .. }
            | Self::ResBlendRmsnorm { .. }
            | Self::HcPre { .. } => {}
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
            | Self::MatmulSwiglu { .. }
            | Self::MatmulSplitRows { .. }
            | Self::MlpSwiglu { .. }
            | Self::MoeSelectGlu { .. }
            | Self::MoeExperts { .. }
            | Self::MlaPrefillExpanded { .. }
            | Self::SplitRopeYarnAppend { .. }
            | Self::NormRopeLastAppend { .. }
            | Self::QkvFusedQknormRopeVnormWrite { .. }
            | Self::SplitQkvFused { .. } => {}
            // The rotation turns the normed rows in place.
            Self::PoolCompress { rotate, .. } => rotate.aliases(sink),
            Self::NormRopePair { q, kv } => {
                q.aliases(sink);
                kv.aliases(sink);
            }
            Self::IndexTopkRope { rotate, rank } => {
                rotate.aliases(sink);
                rank.aliases(sink);
            }
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
            Self::NormRopeLast { .. } => "elementwise.norm_rope_last",
            Self::NormRopeLastAppend { .. } => "elementwise.norm_rope_last_append",
            Self::PoolCompress { .. } => "attention.pool_compress",
            Self::NormRopePair { .. } => "elementwise.norm_rope_pair",
            Self::IndexTopkRope { .. } => "attention.index_topk_rope",
            Self::NormModulate { .. } => "elementwise.norm_modulate",
            Self::GatedResidualNormModulate { .. } => "elementwise.gated_residual_norm_modulate",
            Self::HcNormProject { .. } => "elementwise.hc_norm_project",
            Self::GdnStep { .. } => "attention.ssm_gdn_step",
            Self::QGateNormRopeAppend { .. } => "attention.q_gate_k_norm_rope_append",
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
            Self::MlaPrefillExpanded { .. } => "attention.mla_prefill_expanded",
            Self::HcPrenormProject { .. } => "elementwise.hc_prenorm_project",
            Self::HcGatesNorm { .. } => "elementwise.hc_gates_norm",
            Self::ResBlendRmsnorm { .. } => "elementwise.res_blend_rmsnorm",
            Self::HcPre { .. } => "elementwise.hc_pre",
            Self::MoeExperts { .. } => "linear.moe_experts",
            Self::QkvFusedQknormRopeVnormWrite { .. } => {
                "custom_cuda.qkv_fused_qknorm_rope_vnorm_write"
            }
            Self::SplitQkvFused { .. } => "custom_cuda.split_qkv_fused",
        }
    }
}
