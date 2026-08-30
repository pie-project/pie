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
    /// **THE CENTRED NORM, AND THE ONLY PART OF A `LayerNorm` THAT DOES NOT
    /// BAKE** (multimodal §6.1).
    ///
    /// `y = (x - mean(x)) / rms(x - mean(x))`, with no scale and no bias.
    /// Every qwen vision block is `nn.LayerNorm` — the checkpoints publish
    /// `blocks.{l}.norm1.bias` beside `.weight` to prove it, and an RMSNorm
    /// has no bias — but the two learned vectors are not this op's business:
    /// for the GEMM `M` that reads the norm,
    /// `LN(x)·Mᵀ = (c/rms(c))·diag(w)·Mᵀ + b·Mᵀ` with `c = x − mean(x)`, so
    /// `w` folds into `M` at import and `b·Mᵀ` folds into that GEMM's bias.
    /// The merger's own norm folds the same way through the 2×2 merge, which
    /// is a view. What is left over is exactly this row.
    ///
    /// A SEPARATE VARIANT FROM [`RmsnormNoScale`](Elementwise::RmsnormNoScale)
    /// and not a flag on it: the mean subtraction is a second reduction over
    /// the row, so the two are different kernels and not one kernel under a
    /// boolean. gemma4's tower is RMSNorm (weight only) and reads the older
    /// row unchanged.
    ///
    /// No `head_dim`: the towers norm whole rows. The per-head spelling is
    /// the rms family's because a trunk norms its heads, and a variant with
    /// a field nothing states would be a promise this campaign cannot check.
    LayernormNoScale {
        x: ValueId,
        eps: f32,
        y: ValueId,
    },
    /// **THE WHOLE `nn.LayerNorm`, IN ONE ROW** (multimodal §9.1's owed
    /// saving, next.md B5).
    ///
    /// `y = (x − mean(x)) · rsqrt(var(x) + eps) · w + b`, whole rows, the
    /// scale and the bias read as `[width]` planes.
    ///
    /// **WHY THIS EXISTS BESIDE
    /// [`LayernormNoScale`](Elementwise::LayernormNoScale).** §9.1 found the
    /// import fold half-expressible — `Expr::Scale` bakes `w` into the GEMM
    /// behind the norm, `Expr::Bias` adds one compile-time constant where
    /// `b·Mᵀ` is a matrix-vector product, and the two do not compose — so the
    /// towers spell the norm at RUNTIME. The spelling that stood until this
    /// row is three nodes:
    ///
    /// ```text
    /// add_bias(b, rmsnorm(layernorm_no_scale(x, eps), w, eps))
    /// ```
    ///
    /// which is TWO elementwise passes and a third launch for the bias, times
    /// twenty-five norms per qwen35 tower fire (`norm1`/`norm2` on twelve
    /// blocks, plus `merger.norm`). This row is those three, and the
    /// centred row it computes is never rounded to a storage type on the way
    /// through — which is the one thing the composition cannot claim.
    ///
    /// The scale-less variant STAYS: it is what a text writes when the scale
    /// really does bake, and the two differ in what they read and not in a
    /// flag.
    Layernorm {
        x: ValueId,
        weight: ValueId,
        bias: ValueId,
        eps: f32,
        y: ValueId,
    },
    /// **THE CLIPPED LINEAR'S HALF THAT IS NOT A GEMM** (multimodal §6.5).
    ///
    /// `x = min(max(x, lo), hi)`, in place, with both bounds TRACE CONSTANTS.
    /// gemma4's `vision_config.use_clipped_linears: true` publishes
    /// `{input,output}_{min,max}` as scalars beside every
    /// `encoder.layers.{l}.*.linear.weight`, so each projection clamps what it
    /// reads and what it writes; the scalars are the checkpoint's and are
    /// baked at import like every other number a text states.
    ///
    /// The only clamp this IR had was FUSED inside `linear.mlp_swiglu_clamp`,
    /// which is a swiglu and not a projection. This one is free-standing for
    /// the reason the clamp sites are: they sit on both sides of an ordinary
    /// matmul, and a fused spelling would need one fusion per projection
    /// shape.
    Clamp {
        x: ValueId,
        lo: f32,
        hi: f32,
        x_out: ValueId,
    },
    /// **THE SAME CLAMP, WITH THE BOUNDS THE CHECKPOINT STATES**
    /// (multimodal §12.2): `lo` and `hi` are `[1]` weight planes read on the
    /// device instead of two trace constants.
    ///
    /// `Gemma4ClippableLinear` clamps its input to `[input_min, input_max]`
    /// and its output to `[output_min, output_max]`, and the E4B checkpoint
    /// ships all of them FINITE — 448 scalars over the vision tower alone
    /// (16 layers × 7 linears × 4), `mlp.down_proj.input_max = 12.1875` and so
    /// on down. They are saturating bounds from quantization-aware training
    /// and they differ per linear.
    ///
    /// **SO A TEXT CANNOT KNOW THEM.** A trace is built by `Model::new(w, kv,
    /// tp, dims)` with no checkpoint in the room — that is the whole point of
    /// the split — and a catalog row carrying 448 of them would be a
    /// checkpoint transcribed into a `const`.
    ///
    /// **THE PRECEDENT IS IN THE FAMILY ALREADY**:
    /// [`Scale`](Elementwise::Scale) reads a DEVICE-HELD scalar where
    /// [`MulScalar`](Elementwise::MulScalar) states one, for exactly this
    /// reason. Two rows and not one with an `Option`, for that pair's reason
    /// too: a bound the CONFIG states is a different fact from a bound the
    /// checkpoint ships, and `swiglu_limit` is the first kind — which is why
    /// [`Clamp`](Elementwise::Clamp) stays and `linear.mlp_swiglu_clamp` keeps
    /// its `f32`.
    ///
    /// Same kernel, same launch, one argument apart: the bounds ride the
    /// activation's element, as `Scale`'s scalar does.
    ClampLearned {
        x: ValueId,
        lo: ValueId,
        hi: ValueId,
        x_out: ValueId,
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
    /// **THE MULTIMODAL ROTARY**: [`RopePartial`](Elementwise::RopePartial)
    /// over a position that is a TRIPLE (multimodal §2's second op).
    ///
    /// `positions` is `[rows, 3]` `i32` — one `(t, h, w)` per rotated row —
    /// where the scalar arms read `[rows, 1]`; `sections` is the checkpoint's
    /// own `mrope_section` (qwen36 states `[11, 11, 10]`) and is a TRACE
    /// CONSTANT, so it arrives stated rather than read from device memory.
    /// A separate variant rather than an `Option<[u32; 3]>` on `RopePartial`,
    /// because "a fourth axis with a fourth fact, not a flag" is the ruling
    /// this whole design descends from: scalar-rope lanes and triple-rope
    /// lanes are CLASSES, and a class is what a guard splits on.
    ///
    /// Rotates in place, like every rope arm here — see [`Operands::aliases`].
    ///
    /// **AND `form` SAYS WHICH SECTION LAYOUT** (multimodal §6.3). The trunk's
    /// rotation and the tower's disagree about how the sections map onto the
    /// frequency pairs, and both checkpoints state which they mean
    /// (`text_config.rope_parameters.mrope_interleaved`); see [`MropeForm`].
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
    // `Hc::Collapse` was deleted: no platform can fire it honestly (review R5).
}

/// **WHICH SECTION LAYOUT A [`RopeMrope`](Elementwise::RopeMrope) TURNS BY**
/// (multimodal §6.3).
///
/// The sections say WHICH of `(t, h, w)` a frequency pair turns by; this says
/// HOW the pairs are handed out, and the two checkpoints this campaign serves
/// disagree. A form and not a `bool` because the word "interleaved" is already
/// spent in this enum — [`RopeFull`](Elementwise::RopeFull) and
/// [`RopeYarn`](Elementwise::RopeYarn) carry one, and it means the PAIR layout
/// (`(d, d+1)` against `(d, d+half)`), which is a different question about a
/// different index. Both arms here pair `(d, d + head_dim/2)`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum MropeForm {
    /// **THE TRUNK'S**, and what `mrope_interleaved: true` states. Pairs
    /// alternate `t, h, w, t, h, w, …` for as far as the sections reach, and
    /// the frequency ladder is the head's own: pair `p` turns at
    /// `theta^(-2p/head_dim)` whichever axis it took. Both qwen SKUs'
    /// `text_config.rope_parameters` state it, with `mrope_section`
    /// `[11, 11, 10]`.
    Interleaved,
    /// **THE TOWER'S**, and what `apply_rotary_pos_emb_vision` does. Each
    /// section is a CONTIGUOUS block of pairs — `sections[0]` of `t`, then
    /// `sections[1]` of `h`, then `sections[2]` of `w` — and each block
    /// RESTARTS the frequency ladder over the stated pairs as a whole:
    /// the `i`-th pair of a block turns at `theta^(-2i/Σsections)`.
    ///
    /// That second half is the part a reader would not guess and the part a
    /// wrong kernel would still look plausible under.
    /// `Qwen3_5VisionRotaryEmbedding(head_dim // 2)` builds `head_dim/4`
    /// frequencies over a `head_dim/2`-wide ladder and `freqs[pos_ids]`
    /// indexes it once per AXIS before flattening, so the exponent's
    /// numerator counts within the block and its denominator is the ladder —
    /// which is `Σsections` exactly when the sections tile the rotated pairs,
    /// as the tower's `[0, head_dim/4, head_dim/4]` does.
    ///
    /// The tower rotates over `(h, w)` and has no time axis, which it states
    /// as `sections[0] == 0` rather than as a two-wide position stream: the
    /// stream is `[rows, 3]` on both axes, and a patch's `t` is read by
    /// nothing.
    Blocked,
}

impl Operands for Elementwise {
    fn inputs(&self, sink: &mut Vec<ValueId>) {
        match self {
            Self::Rmsnorm { x, weight, .. } => sink.extend([*x, *weight]),
            Self::RmsnormPerHead { x, weight, .. } => sink.extend([*x, *weight]),
            Self::RmsnormPlusOne { x, weight, .. } => sink.extend([*x, *weight]),
            Self::RmsnormPerHeadPlusOne { x, weight, .. } => sink.extend([*x, *weight]),
            Self::RmsnormNoScale { x, .. } => sink.push(*x),
            Self::LayernormNoScale { x, .. } => sink.push(*x),
            Self::Layernorm { x, weight, bias, .. } => sink.extend([*x, *weight, *bias]),
            Self::Clamp { x, .. } => sink.push(*x),
            Self::ClampLearned { x, lo, hi, .. } => sink.extend([*x, *lo, *hi]),
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
            Self::RopeMrope { q, k, positions, .. } => sink.extend([*q, *k, *positions]),
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
            Self::LayernormNoScale { y, .. } => sink.push(*y),
            Self::Layernorm { y, .. } => sink.push(*y),
            Self::Clamp { x_out, .. } => sink.push(*x_out),
            Self::ClampLearned { x_out, .. } => sink.push(*x_out),
            Self::RmsnormGated { y, .. } => sink.push(*y),
            Self::RmsnormGatedBy { y, .. } => sink.push(*y),
            Self::ResidualAdd { y_out, .. } => sink.push(*y_out),
            Self::AddBias { out_out, .. } => sink.push(*out_out),
            Self::MulScalar { x_out, .. } => sink.push(*x_out),
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
            Self::LayernormNoScale { .. } => {}
            Self::Layernorm { .. } => {}
            Self::Clamp { x_out, x, .. } => sink.push((*x_out, *x)),
            Self::ClampLearned { x_out, x, .. } => sink.push((*x_out, *x)),
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
            Self::RopeMrope { q_out, q, k_out, k, .. } => {
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
            Self::LayernormNoScale { .. } => "elementwise.layernorm_no_scale",
            Self::Layernorm { .. } => "elementwise.layernorm",
            Self::Clamp { .. } => "elementwise.clamp",
            Self::ClampLearned { .. } => "elementwise.clamp_learned",
            Self::RmsnormGated { .. } => "elementwise.rmsnorm_gated",
            Self::RmsnormGatedBy { .. } => "elementwise.rmsnorm_gated_by",
            Self::ResidualAdd { .. } => "elementwise.residual_add",
            Self::AddBias { .. } => "elementwise.add_bias",
            Self::MulScalar { .. } => "elementwise.mul_scalar",
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
            Self::HcGates { .. } => "elementwise.hc_gates",
            Self::HcFold { .. } => "elementwise.hc_fold",
        }
    }
}
