use serde::{Deserialize, Serialize};

use crate::Op;
use crate::operands::Part;
use crate::ops::attn::Attn;
use crate::ops::elemwise::{Elemwise, ModulateForm, Yarn};
use crate::ops::linear::Linear;
use crate::value::ValueId;

/// The norm a [`Fused::RmsnormResidualAdd`] also lands over its last row:
/// `out = rmsnorm(row) * weight` (`1 + weight` where `plus_one`).
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct PostNorm {
    pub weight: ValueId,
    pub plus_one: bool,
    pub eps: f32,
    pub out: ValueId,
}

impl Part for PostNorm {
    fn inputs(&self, sink: &mut Vec<ValueId>) {
        sink.push(self.weight);
    }
    fn outputs(&self, sink: &mut Vec<ValueId>) {
        sink.push(self.out);
    }
}

/// The scale a [`Fused::RmsnormResidualAdd`] lands its stream row under:
/// `out = by * row`.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Scaled {
    pub by: ValueId,
    pub out: ValueId,
}

impl Part for Scaled {
    fn inputs(&self, sink: &mut Vec<ValueId>) {
        sink.push(self.by);
    }
    fn outputs(&self, sink: &mut Vec<ValueId>) {
        sink.push(self.out);
    }
}

#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum NormKind {
    Layernorm { eps: f32 },
    Rmsnorm { head_dim: u32, eps: f32 },
}

/// Ops only the compiler forms, from the primitive ones a model writes; each
/// is a kernel some backend ships.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize, Op)]
#[op(group = "fused")]
pub enum Fused {
    ResidualAddRmsnorm {
        #[input]
        x: ValueId,
        #[input]
        y: ValueId,
        #[output(alias = y)]
        y_out: ValueId,
        #[input]
        weight: ValueId,
        plus_one: bool,
        eps: f32,
        #[output]
        out: ValueId,
    },
    RmsnormResidualAdd {
        #[input]
        x: ValueId,
        #[input]
        weight: ValueId,
        eps: f32,
        #[output]
        t: ValueId,
        #[input]
        y: ValueId,
        #[output(alias = y)]
        y_out: ValueId,
        #[part]
        scale: Option<Scaled>,
        #[part]
        post: Option<PostNorm>,
    },
    EmbedScaleAdd {
        #[input(I32)]
        ids: ValueId,
        #[input]
        table: ValueId,
        vocab: u32,
        #[output]
        e: ValueId,
        embed_scale: f32,
        #[output]
        e_scaled: ValueId,
        #[input]
        y: ValueId,
        #[output(alias = y)]
        y_out: ValueId,
        out_scale: f32,
        #[output]
        y_scaled: ValueId,
    },
    EmbedScaleAddSelect {
        #[input(I32)]
        ids: ValueId,
        #[input]
        table: ValueId,
        vocab: u32,
        #[output]
        e: ValueId,
        embed_scale: f32,
        #[output]
        e_scaled: ValueId,
        #[input]
        stacked: ValueId,
        layer: u32,
        width: u32,
        #[output]
        y_out: ValueId,
        out_scale: f32,
        #[output]
        y_scaled: ValueId,
    },
    RmsnormRopePartialQ {
        #[input]
        x: ValueId,
        #[input]
        weight: ValueId,
        head_dim: u32,
        eps: f32,
        #[input(I32)]
        positions: ValueId,
        rotary_dim: u32,
        theta: f32,
        span: u32,
        #[output]
        y: ValueId,
        #[output(alias = y)]
        q_out: ValueId,
    },
    /// A head's RMS norm (scaled by `weight`, or scale-free) and the
    /// rotation of its last `rotary_dim` dims (a `Rope` turning `Last`), the
    /// normed heads the rotation's own plane.
    NormRopeLast {
        #[input]
        x: ValueId,
        #[input]
        weight: Option<ValueId>,
        head_dim: u32,
        eps: f32,
        #[input(I32)]
        positions: ValueId,
        rotary_dim: u32,
        theta: f32,
        interleaved: bool,
        inverse: bool,
        yarn: Option<Yarn>,
        #[output]
        y: ValueId,
        #[output(alias = y)]
        q_out: ValueId,
    },
    /// `NormRopeLast` over a row one head wide (scaled), the rotated row
    /// then written to the cache at its page and offset
    /// (`KvAppend` of one plane); `y` is read by nothing else.
    NormRopeLastAppend {
        #[input]
        x: ValueId,
        #[input]
        weight: ValueId,
        head_dim: u32,
        eps: f32,
        #[input(I32)]
        positions: ValueId,
        rotary_dim: u32,
        theta: f32,
        interleaved: bool,
        inverse: bool,
        yarn: Option<Yarn>,
        #[output]
        y: ValueId,
        #[input(CACHE)]
        cache: ValueId,
        #[input(I32)]
        write_page: ValueId,
        #[input(I32)]
        write_offset: ValueId,
    },
    /// An index's ranking (`rank`, an `IndexTopk`) of queries whose tail is
    /// rotated just before it (`rotate`, a `Rope` turning the queries' `Last`
    /// the ranking alone reads): a backend may turn them as it loads them.
    IndexTopkRope {
        #[part]
        rotate: Box<Elemwise>,
        #[part]
        rank: Box<Attn>,
    },
    /// DeepSeek-V4's q (`NormRopeLast`, scale-free) and its kv plane
    /// (`NormRopeLastAppend`) of one layer, landed together.
    NormRopePair {
        #[part]
        q: Box<Fused>,
        #[part]
        kv: Box<Fused>,
    },
    /// DeepSeek-V4's compressor at its boundaries: gathered entries
    /// (`PoolGather`) RMS-normed by `weight` into `latent`, its tail rotated
    /// (`rotate`) and appended to the pool (`append`); the gathered and normed
    /// rows are read by nothing else. `state`, if set, also lands the fire's
    /// compressor state written just before (a `PoolStateWrite`).
    PoolCompress {
        #[part]
        gather: Box<Attn>,
        #[input]
        weight: ValueId,
        eps: f32,
        #[output]
        latent: ValueId,
        #[part]
        rotate: Box<Elemwise>,
        #[part]
        append: Box<Attn>,
        #[part]
        state: Option<Box<Attn>>,
    },
    NormModulate {
        #[input]
        x: ValueId,
        norm: NormKind,
        #[output]
        normed: ValueId,
        #[input]
        m: ValueId,
        #[input(I32)]
        lane_of_row: Option<ValueId>,
        form: ModulateForm,
        #[output]
        y: ValueId,
    },
    GatedResidualNormModulate {
        #[input]
        r: ValueId,
        #[input]
        g: ValueId,
        #[input]
        y: ValueId,
        #[input(I32)]
        lane_of_row: Option<ValueId>,
        #[output(alias = r)]
        r_out: ValueId,
        norm: NormKind,
        #[output]
        normed: ValueId,
        #[input]
        m: ValueId,
        form: ModulateForm,
        #[output]
        out: ValueId,
    },
    /// mHC's mix projection of the f32-normed stream row, from the streams
    /// themselves: the norm (no weight) is a row scale, so it lands on the
    /// projection's sums and the normed row is never written.
    HcNormProject {
        #[input]
        streams: ValueId,
        eps: f32,
        #[input(F32)]
        weight: ValueId,
        stream_count: u32,
        #[output(F32)]
        mixes: ValueId,
    },
    /// A gated-delta layer's decode step from its in-projections' rows: the
    /// split of `qkvz` at `width` into `qkv` and `z`, the causal conv update
    /// of `qkv` into `conv`, the gate prep of `ba` into `gates`, and the
    /// recurrence into `y`. Every intermediate keeps its value, so a backend
    /// that cannot take the step whole runs the primitives over them.
    GdnStep {
        #[input]
        qkvz: ValueId,
        width: u32,
        #[output]
        qkv: ValueId,
        #[output]
        z: ValueId,
        #[input]
        conv_weight: ValueId,
        #[input(CACHE)]
        conv_state: ValueId,
        conv_width: u32,
        dilation: u32,
        #[output]
        conv: ValueId,
        #[input]
        ba: ValueId,
        #[input]
        dt_bias: ValueId,
        #[input]
        a_log: ValueId,
        #[output(F32)]
        gates: ValueId,
        #[input(CACHE)]
        state: ValueId,
        k_heads: u32,
        v_heads: u32,
        k_dim: u32,
        v_dim: u32,
        #[output(F32)]
        y: ValueId,
    },
    /// Qwen's gated attention prologue: the split of the `[q_h | gate_h]`
    /// projection `qg` into `q` and `gate`, q and k normed per head (`w + 1`)
    /// into `qn` and `kn`, both turned by the partial rotation into `q_out`
    /// and `k_out` (in place, as the rotation lands them), and `k_out` and
    /// `v` appended to `cache`. Every intermediate keeps its value, so a
    /// backend that cannot take the step whole runs the primitives over them.
    QGateNormRopeAppend {
        #[input]
        qg: ValueId,
        head_dim: u32,
        #[output]
        q: ValueId,
        #[output]
        gate: ValueId,
        #[input]
        q_norm: ValueId,
        q_eps: f32,
        #[output]
        qn: ValueId,
        #[input]
        k: ValueId,
        #[input]
        k_norm: ValueId,
        k_eps: f32,
        #[output]
        kn: ValueId,
        #[input(I32)]
        positions: ValueId,
        rotary_dim: u32,
        theta: f32,
        span: u32,
        #[output(alias = qn)]
        q_out: ValueId,
        #[output(alias = kn)]
        k_out: ValueId,
        #[input]
        v: ValueId,
        #[input(CACHE)]
        cache: ValueId,
        #[input(I32)]
        write_page: ValueId,
        #[input(I32)]
        write_offset: ValueId,
    },
    MatmulGeglu {
        #[input]
        act: ValueId,
        #[input]
        w: ValueId,
        intermediate: u32,
        #[output]
        packed: ValueId,
        #[output]
        y: ValueId,
    },
    /// The packed gate-up projection and its SwiGLU; `packed` still an
    /// output, read by nothing else.
    MatmulSwiglu {
        #[input]
        act: ValueId,
        #[input]
        w: ValueId,
        intermediate: u32,
        #[output]
        packed: ValueId,
        #[output]
        y: ValueId,
    },
    /// A whole dense MLP: `gate_up` over `act`, swiglu, `down`. `packed` and
    /// `h` are the two intermediates, still written, so a backend may run
    /// the three primitives, or split the chain between engines.
    MatmulSwigluMatmul {
        #[input]
        act: ValueId,
        #[input]
        gate_up: ValueId,
        #[input]
        down: ValueId,
        intermediate: u32,
        #[output]
        packed: ValueId,
        #[output]
        h: ValueId,
        #[output]
        y: ValueId,
    },
    /// The logits' product (a `Matmul` with `logits`) and its softcap
    /// (`Softcap`); `y` is still an output, read by nothing else.
    MatmulSoftcap {
        #[input]
        act: ValueId,
        #[input]
        w: ValueId,
        cap: f32,
        #[output]
        y: ValueId,
        #[output(alias = y)]
        y_out: ValueId,
    },
    MatmulBias {
        #[input]
        act: ValueId,
        #[input]
        w: ValueId,
        #[input]
        bias: ValueId,
        #[output]
        y: ValueId,
        #[output(alias = y)]
        y_out: ValueId,
    },
    /// A projection whose rows are cut at `width` straight after: a backend
    /// may land each side from its own slice of the weight, or run the
    /// matmul into `y` and cut it.
    MatmulSplitRows {
        #[input]
        act: ValueId,
        #[input]
        w: ValueId,
        width: u32,
        #[output]
        y: ValueId,
        #[output]
        left: ValueId,
        #[output]
        right: ValueId,
    },
    /// A routed gate-up select (`MoeSelect`, `y` the packed gate|up rows) and
    /// its packed `Glu` (`Swiglu`, `SwigluAlpha` or `GegluTanh`), landable
    /// from the select's epilogue. The packed rows are still an output, read
    /// by nothing else.
    MoeSelectGlu {
        #[part]
        select: Box<Linear>,
        #[part]
        glu: Box<Linear>,
    },
    /// A log-sum-exp attention (`DecodeLse` or `PrefillLse`) and its sinks
    /// (`Sink`), the sinked output landed in `o_out`. The attention's `o`
    /// (read by nothing else) and `lse` are still outputs.
    LseSink {
        #[part]
        attn: Box<Attn>,
        #[input]
        sink: ValueId,
        head_dim: u32,
        #[output]
        o_out: ValueId,
    },
    /// A sublayer's whole hyper-connection pre-mix: stream norm (`norm`, an
    /// `HcRmsnormF32`), mix projection (`project`, an `HcProject`), combined
    /// gates (`gates`, an `HcGates`) and the pre-mixed row's `rmsnorm`. The
    /// intermediates are still outputs, read by nothing else.
    HcPre {
        #[part]
        norm: Box<Elemwise>,
        #[part]
        project: Box<Elemwise>,
        #[part]
        gates: Box<Elemwise>,
        #[part]
        rmsnorm: Box<Elemwise>,
    },
    /// The combined hyper-connection gates (`gates`, an `HcGates`) and the RMS
    /// norm of the row they pre-mix (`norm`), landable in one pass. The
    /// pre-mixed row is still an output, read by nothing else.
    HcGatesNorm {
        #[part]
        gates: Box<Elemwise>,
        #[part]
        norm: Box<Elemwise>,
    },
    /// AttnRes's blend (`blend`, a `ResBlend`) and the RMS norm of its row
    /// (`norm`), landable in one pass. The blended row is still an output,
    /// read by nothing else.
    ResBlendRmsnorm {
        #[part]
        blend: Box<Elemwise>,
        #[part]
        norm: Box<Elemwise>,
    },
    /// A hyper-connection stream norm (`norm`, an `HcRmsnormF32`) and the mix
    /// projection it feeds (`project`, an `HcProject`), landable in one pass
    /// with the norm's per-row scale taken out of the dot. The normed rows are
    /// still an output, read by nothing else.
    HcPrenormProject {
        #[part]
        norm: Box<Elemwise>,
        #[part]
        project: Box<Elemwise>,
    },
    /// DeepSeek-V4's whole attention: windowed (`window`, a `PrefillLse`) and
    /// compressed-pool (`pool`, a `PoolLse` or `PoolLseSelected`) attention
    /// over one q, their `merge` (a `MergeLse`) and its sinks (`sink`, a
    /// `Sink` over the merged plane); every intermediate is still an output,
    /// read by nothing else. `rotate` (the inverse rotation of the sinked
    /// output's tail) lands too.
    PoolAttention {
        #[part]
        window: Box<Attn>,
        #[part]
        pool: Box<Attn>,
        #[part]
        merge: Box<Attn>,
        #[part]
        sink: Box<Attn>,
        #[part]
        rotate: Option<Box<Elemwise>>,
    },
    /// A latent attention's whole prefill: the optional cut of `q_b`'s rows
    /// (`split`, an `MlaSplitQB`), query absorb (`absorb_q`), scoring over
    /// every cached latent row (`prefill`) and output absorb (`absorb_out`)
    /// through one `kv_b`, whose rows a backend may land expanded (a k and v
    /// head a q head). Every intermediate is still an output, read by nothing
    /// else.
    MlaPrefillExpanded {
        #[part]
        split: Option<Box<Attn>>,
        #[part]
        absorb_q: Box<Attn>,
        #[part]
        prefill: Box<Attn>,
        #[part]
        absorb_out: Box<Attn>,
    },
    /// YaRN over q and k, and the rotated k with its v landed in the paged
    /// cache at each row's page and offset (`kv_append`); `k_out` is read by
    /// nothing else.
    RopeYarnAppend {
        #[input]
        q: ValueId,
        #[input]
        k: ValueId,
        #[input]
        v: ValueId,
        #[input(I32)]
        positions: ValueId,
        head_dim: u32,
        theta: f32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        attention_factor: f32,
        original_max_position: u32,
        interleaved: bool,
        #[input(CACHE)]
        cache: ValueId,
        #[input(I32)]
        write_page: ValueId,
        #[input(I32)]
        write_offset: ValueId,
        #[output(alias = q)]
        q_out: ValueId,
        #[output(alias = k)]
        k_out: ValueId,
    },
    /// Qwen3.5's attention prologue: q cut from `[q | gate]` (`split_q_gate`),
    /// q and k normed per head (`rmsnorm_per_head_plus_one`), their rotary
    /// prefixes turned (`rope_partial`) and k with v cached (`kv_append`).
    /// `q_split`, `q_normed`, `k_normed` and `k_out` may be left unwritten.
    QgQknormRopeAppend {
        #[input]
        qg: ValueId,
        #[input]
        k: ValueId,
        #[input]
        v: ValueId,
        #[input(I32)]
        positions: ValueId,
        #[input]
        q_norm: ValueId,
        #[input]
        k_norm: ValueId,
        head_dim: u32,
        q_eps: f32,
        k_eps: f32,
        rotary_dim: u32,
        theta: f32,
        span: u32,
        #[input(CACHE)]
        cache: ValueId,
        #[input(I32)]
        write_page: ValueId,
        #[input(I32)]
        write_offset: ValueId,
        #[output]
        q_split: ValueId,
        #[output]
        gate: ValueId,
        #[output]
        q_normed: ValueId,
        #[output]
        k_normed: ValueId,
        #[output(alias = q_normed)]
        q_out: ValueId,
        #[output(alias = k_normed)]
        k_out: ValueId,
    },
    /// `RopeYarnAppend` reading q, k and v straight from a packed row split by
    /// `split_qkv`. `k_out` and `v_out` (the split's k and v) may be left
    /// unwritten.
    SplitRopeYarnAppend {
        #[input]
        packed: ValueId,
        q_width: u32,
        kv_width: u32,
        #[input(I32)]
        positions: ValueId,
        head_dim: u32,
        theta: f32,
        factor: f32,
        beta_fast: f32,
        beta_slow: f32,
        attention_factor: f32,
        original_max_position: u32,
        interleaved: bool,
        #[input(CACHE)]
        cache: ValueId,
        #[input(I32)]
        write_page: ValueId,
        #[input(I32)]
        write_offset: ValueId,
        #[output]
        q_out: ValueId,
        #[output]
        k_out: ValueId,
        #[output]
        v_out: ValueId,
    },
    /// A routed MLP: gate-up select and GLU (as in `MoeSelectGlu`), the down
    /// select on the same routes, the per-token weighted sum (`MoeCombine`)
    /// and, where it follows, the sigmoid-gated shared expert
    /// (`MoeSigmoidGateAdd`). The rows between (and the sum, under a gate) are
    /// still outputs, read by nothing else. `norm`: each route's down row
    /// normed under a scale (`NormGate`) before the sum (Muse Spark's).
    MoeExperts {
        #[part]
        select: Box<Linear>,
        #[part]
        glu: Box<Linear>,
        #[part]
        down: Box<Linear>,
        #[part]
        norm: Option<Box<Elemwise>>,
        #[part]
        sum: Box<Linear>,
        #[part]
        gate_add: Option<Box<Linear>>,
    },
    /// `qkv` (a `QkvFusedQknormRopeVnormWrite`) on the first columns of its
    /// `packed` rows, the rest of each row (`rest`'s width) landing in
    /// `rest`: a row split (Muse Spark's qkv|gate) read in place.
    SplitQkvFused {
        #[output]
        rest: ValueId,
        #[part]
        qkv: Box<Fused>,
    },
    /// A packed qkv row split, q and k normed per head (under weights, or
    /// none) where `qk_norm`, v normed without a scale where `v_norm` and
    /// scaled by `v_scale`, the first `rotary_dim / 2` pairs `span` apart of
    /// q and k turned (`rotary_dim` 0: none), and k and v written to the
    /// cache, the packed heads `widths` (q and k, v) wide and landing
    /// `head_dim` wide: Gemma 4's attention prologue, Muse Spark's and MiMo's.
    QkvFusedQknormRopeVnormWrite {
        #[input]
        packed: ValueId,
        /// The rows' positions, where they turn.
        #[input(I32)]
        positions: Option<ValueId>,
        #[input]
        q_norm_weight: Option<ValueId>,
        #[input]
        k_norm_weight: Option<ValueId>,
        v_norm: bool,
        /// The one epsilon both head norms and the v norm use.
        eps: f32,
        #[input(CACHE)]
        cache: ValueId,
        #[input(I32)]
        write_page: ValueId,
        #[input(I32)]
        write_offset: ValueId,
        kv_heads: u32,
        head_dim: u32,
        theta: f32,
        rotary_dim: u32,
        span: u32,
        qk_norm: bool,
        v_scale: f32,
        widths: [u32; 2],
        #[output]
        q: ValueId,
    },
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::Operands;
    use crate::ops::elemwise::{NormAxis, NormWeight};

    fn v(id: u32) -> ValueId {
        ValueId(id)
    }

    fn operands(op: &Fused) -> (Vec<u32>, Vec<u32>, Vec<(u32, u32)>) {
        let (mut ins, mut outs, mut aliases) = (Vec::new(), Vec::new(), Vec::new());
        op.inputs(&mut ins);
        op.outputs(&mut outs);
        op.aliases(&mut aliases);
        (
            ins.iter().map(|id| id.0).collect(),
            outs.iter().map(|id| id.0).collect(),
            aliases.iter().map(|(a, b)| (a.0, b.0)).collect(),
        )
    }

    /// The chain's optional scale and post-norm each read one value and
    /// land one.
    #[test]
    fn a_norm_residual_chain_reads_and_lands_its_options() {
        let chain = |scale: bool, post: bool| Fused::RmsnormResidualAdd {
            x: v(1),
            weight: v(2),
            eps: 1e-6,
            t: v(3),
            y: v(4),
            y_out: v(5),
            scale: scale.then_some(Scaled {
                by: v(6),
                out: v(7),
            }),
            post: post.then(|| PostNorm {
                weight: v(8),
                plus_one: false,
                eps: 1e-6,
                out: v(9),
            }),
        };
        assert_eq!(
            operands(&chain(false, false)),
            (vec![1, 2, 4], vec![3, 5], vec![(5, 4)])
        );
        assert_eq!(
            operands(&chain(true, true)),
            (vec![1, 2, 4, 6, 8], vec![3, 5, 7, 9], vec![(5, 4)])
        );
    }

    /// The rotation lands over the normed heads, an output of the same op.
    #[test]
    fn a_rotation_lands_over_the_heads_it_normed() {
        let op = Fused::QgQknormRopeAppend {
            qg: v(1),
            k: v(2),
            v: v(3),
            positions: v(4),
            q_norm: v(5),
            k_norm: v(6),
            head_dim: 128,
            q_eps: 1e-6,
            k_eps: 1e-6,
            rotary_dim: 64,
            theta: 1e7,
            span: 32,
            cache: v(7),
            write_page: v(8),
            write_offset: v(9),
            q_split: v(10),
            gate: v(11),
            q_normed: v(12),
            k_normed: v(13),
            q_out: v(14),
            k_out: v(15),
        };
        assert_eq!(
            operands(&op),
            (
                vec![1, 2, 3, 4, 5, 6, 7, 8, 9],
                vec![10, 11, 12, 13, 14, 15],
                vec![(14, 12), (15, 13)]
            )
        );
    }

    /// A pre-mix reads the streams, the projection's weight, the gates'
    /// scale and base and the norm's weight; what its parts pass among
    /// themselves stays inside it.
    #[test]
    fn a_hyper_connection_pre_mix_reads_only_what_its_parts_do_not_make() {
        let norm = Elemwise::HcRmsnormF32 {
            streams: v(1),
            eps: 1e-6,
            y: v(10),
        };
        let project = Elemwise::HcProject {
            normed: v(10),
            weight: v(2),
            stream_count: 4,
            mixes: v(11),
        };
        let gates = Elemwise::HcGates {
            normed: v(11),
            streams: v(1),
            scale: v(3),
            base: v(4),
            stream_count: 4,
            gate_eps: 1e-6,
            alpha: 1.0,
            sinkhorn: 20,
            comb: true,
            x: v(12),
            post_mix: v(13),
            comb_mix: v(14),
        };
        let rmsnorm = Elemwise::Rmsnorm {
            x: v(12),
            weight: NormWeight::Scale(v(5)),
            axis: NormAxis::Row,
            eps: 1e-6,
            y: v(15),
        };
        let pre = Fused::HcPre {
            norm: Box::new(norm.clone()),
            project: Box::new(project.clone()),
            gates: Box::new(gates),
            rmsnorm: Box::new(rmsnorm),
        };
        assert_eq!(
            operands(&pre),
            (vec![1, 2, 3, 4, 5], vec![10, 11, 12, 13, 14, 15], vec![])
        );
        let prenorm = Fused::HcPrenormProject {
            norm: Box::new(norm),
            project: Box::new(project),
        };
        assert_eq!(operands(&prenorm), (vec![1, 2], vec![10, 11], vec![]));
    }

    /// The sinks scale the merged plane where it lies; a log-sum-exp
    /// attention's sinked output has a plane of its own.
    #[test]
    fn the_sinks_land_where_their_op_says() {
        let merge = Attn::MergeLse {
            o1: v(20),
            lse1: v(21),
            o2: v(22),
            lse2: v(23),
            heads: 64,
            head_dim: 512,
            o: v(24),
            lse: v(25),
        };
        let lse_sink = Fused::LseSink {
            attn: Box::new(merge),
            sink: v(9),
            head_dim: 512,
            o_out: v(26),
        };
        assert_eq!(
            operands(&lse_sink),
            (vec![20, 21, 22, 23, 9], vec![24, 25, 26], vec![])
        );
        let sink = Attn::Sink {
            o: v(24),
            lse: v(25),
            sink: v(9),
            head_dim: 512,
            o_out: v(26),
        };
        let (mut ins, mut aliases) = (Vec::new(), Vec::new());
        sink.inputs(&mut ins);
        sink.aliases(&mut aliases);
        assert_eq!(
            (ins, aliases),
            (vec![v(24), v(25), v(9)], vec![(v(26), v(24))])
        );
    }
}
