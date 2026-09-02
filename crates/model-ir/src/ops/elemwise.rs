use serde::{Deserialize, Serialize};

use crate::operands::Operands;
use crate::value::ValueId;

/// Per-token math — tokens are independent. Per-token reductions like
/// rmsnorm's mean-of-squares belong here.
/// The YaRN interpolation a partial rope states beside its theta: the
/// reference's `precompute_freqs(dim, original_seq_len, base, factor,
/// beta_fast, beta_slow)` with `original_seq_len > 0`. The ramp bounds are
/// derived on the device side from these and the rotated width.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct Yarn {
    pub factor: f32,
    pub beta_fast: f32,
    pub beta_slow: f32,
    pub original_max_position: u32,
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
    /// The centred norm: `y = (x - mean(x)) / rms(x - mean(x))`, no scale, no
    /// bias. Separate from [`RmsnormNoScale`](Elementwise::RmsnormNoScale)
    /// since the mean subtraction is a second reduction, a different kernel.
    LayernormNoScale {
        x: ValueId,
        eps: f32,
        y: ValueId,
    },
    /// The whole `nn.LayerNorm` in one row: `y = (x - mean(x)) *
    /// rsqrt(var(x) + eps) * w + b`, scale/bias read as `[width]` planes.
    /// Exists beside [`LayernormNoScale`](Elementwise::LayernormNoScale)
    /// because the import fold can't express this pair.
    Layernorm {
        x: ValueId,
        weight: ValueId,
        bias: ValueId,
        eps: f32,
        y: ValueId,
    },
    /// Hyper-connection norm (qwen4): moments per `group`-wide slice, scaled
    /// by `weight + 1` over the full row width (per-stream weight, unlike
    /// [`RmsnormPerHeadPlusOne`](Elementwise::RmsnormPerHeadPlusOne)).
    RmsnormGroupedPlusOne {
        x: ValueId,
        weight: ValueId,
        group: u32,
        eps: f32,
        y: ValueId,
    },
    /// `x = min(max(x, lo), hi)`, in place, bounds as trace constants
    /// (gemma4's `use_clipped_linears`).
    Clamp {
        x: ValueId,
        lo: f32,
        hi: f32,
        x_out: ValueId,
    },
    /// The same clamp, with `lo`/`hi` as `[1]` device-held planes instead of
    /// trace constants (checkpoints shipping per-linear QAT bounds).
    ClampLearned {
        x: ValueId,
        lo: ValueId,
        hi: ValueId,
        x_out: ValueId,
    },
    /// `x` is f32; the norm is gated by `act(gate)`, per group of
    /// `head_dim`. `act` is the checkpoint's `output_gate_type` (qwen3.5:
    /// silu, qwen4: sigmoid).
    RmsnormGated {
        x: ValueId,
        gate: ValueId,
        weight: ValueId,
        head_dim: u32,
        eps: f32,
        act: GateActivation,
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
    /// Vision tower output standardization (`vision_config.standardize`):
    /// `y = (x - bias) * scale`, per column, both planes `[width]`, in place.
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
    /// `silu(s * x)`, in place. The scalar is inside the activation
    /// (`silu(s*x) != s*silu(x)`), so this is one launch where
    /// [`MulScalar`](Elementwise::MulScalar) before a bare silu would be
    /// two.
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
    /// [`RopePartial`](Elementwise::RopePartial) over a position triple:
    /// `positions` is `[rows, 3]` `i32` (one `(t, h, w)` per row); `sections`
    /// is the checkpoint's `mrope_section`. `form` says which section layout
    /// applies; see [`MropeForm`].
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
    /// Partial rope over the last `rotary_dim` lanes of each head.
    ///
    /// **`inverse` UNROTATES** — the angle is negated — for the one place a
    /// value carries a key's rope: MLA's shared latent is both key and value,
    /// so the attention output's rope lanes come back rotated by the query's
    /// own position and the reference undoes it (`apply_rotary_emb(o[...,
    /// -rd:], freqs, inverse=True)`, the official `Attention.forward`).
    ///
    /// **`yarn` IS THE LAYER'S OWN RULE, NOT THE MODEL'S.** DeepSeek-V4-Flash
    /// ropes its compressor layers at `compress_rope_theta` WITH the YaRN
    /// ramp and its pure sliding-window layers at `rope_theta` without one
    /// (`if self.compress_ratio: original_seq_len, rope_theta =
    /// args.original_seq_len, args.compress_rope_theta else 0, args.rope_theta`),
    /// so the ramp rides the op beside the theta and is `None` where the
    /// layer states none.
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
    /// The per-token mix row: `rmsnorm(streams) . hc_fn^T`, the row
    /// [`Self::HcGates`] splits into pre, post and the combiner. Not
    /// [`Linear::Matmul`](crate::ops::Linear): kept f32, too sensitive for bf16.
    /// The row is as wide as the plane says: `2M + M²` for a layer's
    /// `{attn,ffn}_hc.fn`, `M` for the trunk's `hc_head.fn` ([`Self::HcCollapse`]).
    HcProject {
        normed: ValueId,
        weight: ValueId,
        stream_count: u32,
        mixes: ValueId,
    },
    /// Computes the layer input `x` plus the post/comb mixing matrices.
    ///
    /// `normed` is the mix row [`Self::HcProject`] lands — `[N, 2M + M²]`,
    /// which is the stride this op has always read its operand at.
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
    /// The trunk collapse (`hc_head`): the `M` streams folded into the row the
    /// final norm reads under `M` sigmoid gates off the `[N, M]` mix row
    /// [`Self::HcProject`] lands through `hc_head.fn` — no post, no combiner,
    /// no Sinkhorn. `y[h] = Σₛ (σ(mixes[s]·scale[0] + base[s]) + hc_eps) · streams[s·H + h]`.
    HcCollapse {
        mixes: ValueId,
        streams: ValueId,
        scale: ValueId,
        base: ValueId,
        stream_count: u32,
        hc_eps: f32,
        y: ValueId,
    },

    // The gated-residual flavor (qwen4): mixes through per-element sigmoid
    // gates instead of a sinkhorn-normalized matrix. The GEMMs stay
    // `linear.matmul` nodes; these two ops are the arithmetic around them.
    /// `y[h] = mean_s(sigmoid(gates[s*H + h]) * normed[s*H + h])` — one
    /// `hidden`-wide layer input mixed out of `streams` normed residual
    /// streams under per-element sigmoid gates.
    HcMix {
        gates: ValueId,
        normed: ValueId,
        streams: u32,
        y: ValueId,
    },
    /// `hyper[s*H + h] += 2*sigmoid(gates[s] / streams) * o[h]` — the layer
    /// output injected back into every stream under its own scalar gate. In
    /// place on `hyper`.
    HcInject {
        o: ValueId,
        gates: ValueId,
        streams: u32,
        hyper: ValueId,
        hyper_out: ValueId,
    },
    /// PLE gate (qwen4): `y[s*H+h] = sigmoid(sgn(d)*sqrt(|d|)) * value[h]`
    /// where `d = sum_j key[s*H+j] * query[s*H+j] / sqrt(H)`.
    PleGate {
        key: ValueId,
        query: ValueId,
        value: ValueId,
        streams: u32,
        y: ValueId,
    },
}

/// Which activation gates a [`RmsnormGated`](Elementwise::RmsnormGated) —
/// the checkpoint's `output_gate_type`, as a form rather than a string.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum GateActivation {
    Silu,
    Sigmoid,
}

/// Which section layout a [`RopeMrope`](Elementwise::RopeMrope) turns by —
/// how `(t, h, w)` frequency pairs are handed out. Both arms pair
/// `(d, d + head_dim/2)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MropeForm {
    /// The trunk's (`mrope_interleaved: true`): pairs alternate `t, h, w, ...`;
    /// pair `p` turns at `theta^(-2p/head_dim)` whichever axis it took.
    Interleaved,
    /// The tower's (`apply_rotary_pos_emb_vision`): each section is a
    /// contiguous block of pairs, each restarting the frequency ladder
    /// (`sections[0] == 0`, no time axis).
    Blocked,
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
            Self::Layernorm { x, weight, bias, .. } => sink.extend([*x, *weight, *bias]),
            Self::Clamp { x, .. } => sink.push(*x),
            Self::ClampLearned { x, lo, hi, .. } => sink.extend([*x, *lo, *hi]),
            Self::RmsnormGated { x, gate, weight, .. } => sink.extend([*x, *gate, *weight]),
            Self::RmsnormGatedBy { x, gate, weight, .. } => sink.extend([*x, *gate, *weight]),
            Self::ResidualAdd { x, y, .. } => sink.extend([*x, *y]),
            Self::AddBias { bias, out, .. } => sink.extend([*bias, *out]),
            Self::Standardize { x, bias, scale, .. } => sink.extend([*x, *bias, *scale]),
            Self::MulScalar { x, .. } => sink.push(*x),
            Self::SiluScaled { x, .. } => sink.push(*x),
            Self::Scale { s, x, .. } => sink.extend([*s, *x]),
            Self::ResBlend { prefix, blocks, weight, proj, .. } => {
                sink.push(*prefix);
                sink.extend_from_slice(blocks);
                sink.push(*weight);
                sink.push(*proj);
            }
            Self::RopeFull { q, k, positions, .. } => sink.extend([*q, *k, *positions]),
            Self::RopePartial { q, k, positions, .. } => sink.extend([*q, *k, *positions]),
            Self::RopeMrope { q, k, positions, .. } => sink.extend([*q, *k, *positions]),
            Self::RopePartialQ { q, positions, .. } => sink.extend([*q, *positions]),
            Self::RopePartialLast { q, positions, .. } => sink.extend([*q, *positions]),
            Self::RopeYarn { q, k, positions, .. } => sink.extend([*q, *k, *positions]),
            Self::GateSigmoidMul { x, gate, .. } => sink.extend([*x, *gate]),
            Self::HcExpand { x, .. } => sink.push(*x),
            Self::HcRmsnormF32 { streams, .. } => sink.push(*streams),
            Self::HcProject { normed, weight, .. } => sink.extend([*normed, *weight]),
            Self::HcGates { normed, streams, scale, base, .. } => {
                sink.extend([*normed, *streams, *scale, *base]);
            }
            Self::HcFold { x, streams, post_mix, comb_mix, .. } => {
                sink.extend([*x, *streams, *post_mix, *comb_mix]);
            }
            Self::HcCollapse { mixes, streams, scale, base, .. } => {
                sink.extend([*mixes, *streams, *scale, *base]);
            }
            Self::HcMix { gates, normed, .. } => sink.extend([*gates, *normed]),
            Self::HcInject { o, gates, hyper, .. } => sink.extend([*o, *gates, *hyper]),
            Self::PleGate { key, query, value, .. } => sink.extend([*key, *query, *value]),
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
            Self::RopePartialLast { q_out, .. } => sink.push(*q_out),
            Self::RopeYarn { q_out, k_out, .. } => sink.extend([*q_out, *k_out]),
            Self::GateSigmoidMul { x_out, .. } => sink.push(*x_out),
            Self::HcExpand { y, .. } => sink.push(*y),
            Self::HcRmsnormF32 { y, .. } => sink.push(*y),
            Self::HcProject { mixes, .. } => sink.push(*mixes),
            Self::HcGates { x, post_mix, comb_mix, .. } => sink.extend([*x, *post_mix, *comb_mix]),
            Self::HcFold { y, .. } => sink.push(*y),
            Self::HcCollapse { y, .. } => sink.push(*y),
            Self::HcMix { y, .. } => sink.push(*y),
            Self::HcInject { hyper_out, .. } => sink.push(*hyper_out),
            Self::PleGate { y, .. } => sink.push(*y),
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
            Self::AddBias { out_out, out, .. } => sink.push((*out_out, *out)),
            Self::Standardize { x_out, x, .. } => sink.push((*x_out, *x)),
            Self::MulScalar { x_out, x, .. } => sink.push((*x_out, *x)),
            Self::SiluScaled { x_out, x, .. } => sink.push((*x_out, *x)),
            Self::Scale { x_out, x, .. } => sink.push((*x_out, *x)),
            Self::ResBlend { .. } => {}
            Self::RopeFull { q_out, q, k_out, k, .. } => sink.extend([(*q_out, *q), (*k_out, *k)]),
            Self::RopePartial { q_out, q, k_out, k, .. } => {
                sink.extend([(*q_out, *q), (*k_out, *k)]);
            }
            Self::RopeMrope { q_out, q, k_out, k, .. } => {
                sink.extend([(*q_out, *q), (*k_out, *k)]);
            }
            Self::RopePartialQ { q_out, q, .. } => sink.push((*q_out, *q)),
            Self::RopePartialLast { q_out, q, .. } => sink.push((*q_out, *q)),
            Self::RopeYarn { q_out, q, k_out, k, .. } => sink.extend([(*q_out, *q), (*k_out, *k)]),
            Self::GateSigmoidMul { x_out, x, .. } => sink.push((*x_out, *x)),
            Self::HcExpand { .. } => {}
            Self::HcRmsnormF32 { .. } => {}
            Self::HcProject { .. } => {}
            Self::HcGates { .. } => {}
            Self::HcFold { .. } => {}
            Self::HcCollapse { .. } => {}
            Self::HcMix { .. } => {}
            Self::HcInject { hyper_out, hyper, .. } => sink.push((*hyper_out, *hyper)),
            Self::PleGate { .. } => {}
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
            Self::RopePartialLast { .. } => "elementwise.rope_partial_last",
            Self::RopeYarn { .. } => "elementwise.rope_yarn",
            Self::GateSigmoidMul { .. } => "elementwise.gate_sigmoid_mul",
            Self::HcExpand { .. } => "elementwise.hc_expand",
            Self::HcRmsnormF32 { .. } => "elementwise.hc_rmsnorm_f32",
            Self::HcProject { .. } => "elementwise.hc_project",
            Self::HcGates { .. } => "elementwise.hc_gates",
            Self::HcFold { .. } => "elementwise.hc_fold",
            Self::HcCollapse { .. } => "elementwise.hc_collapse",
            Self::HcMix { .. } => "elementwise.hc_mix",
            Self::HcInject { .. } => "elementwise.hc_inject",
            Self::PleGate { .. } => "elementwise.ple_gate",
        }
    }
}
