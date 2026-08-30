//! The `Elementwise` family: the norms and residual algebra, rope, the sigmoid
//! gate, and the hyper-connection stream ops.

use super::*;

pub fn rmsnorm(x: &Value, weight: &Weight, eps: f32) -> Value {
    let r = x.rec();
    let y = r.fresh(x.ty().clone());
    r.push(
        Elementwise::Rmsnorm {
            x: x.id(),
            weight: r.weight(weight),
            eps,
            y: y.id(),
        },
        &[x],
    );
    y
}

pub fn rmsnorm_per_head(x: &Value, weight: &Weight, head_dim: u32, eps: f32) -> Value {
    let r = x.rec();
    let y = r.fresh(x.ty().clone());
    r.push(
        Elementwise::RmsnormPerHead {
            x: x.id(),
            weight: r.weight(weight),
            head_dim,
            eps,
            y: y.id(),
        },
        &[x],
    );
    y
}

pub fn rmsnorm_plus_one(x: &Value, weight: &Weight, eps: f32) -> Value {
    let r = x.rec();
    let y = r.fresh(x.ty().clone());
    r.push(
        Elementwise::RmsnormPlusOne {
            x: x.id(),
            weight: r.weight(weight),
            eps,
            y: y.id(),
        },
        &[x],
    );
    y
}

pub fn rmsnorm_per_head_plus_one(x: &Value, weight: &Weight, head_dim: u32, eps: f32) -> Value {
    let r = x.rec();
    let y = r.fresh(x.ty().clone());
    r.push(
        Elementwise::RmsnormPerHeadPlusOne {
            x: x.id(),
            weight: r.weight(weight),
            head_dim,
            eps,
            y: y.id(),
        },
        &[x],
    );
    y
}

pub fn rmsnorm_no_scale(x: &Value, head_dim: u32, eps: f32) -> Value {
    let r = x.rec();
    let y = r.fresh(x.ty().clone());
    r.push(
        Elementwise::RmsnormNoScale {
            x: x.id(),
            head_dim,
            eps,
            y: y.id(),
        },
        &[x],
    );
    y
}

/// **THE CENTRED NORM** (multimodal §6.1): mean-subtract, then rms-normalize,
/// with no scale and no bias.
///
/// The vision towers' `nn.LayerNorm`, minus the two learned vectors, which
/// BAKE: `LN(x)·Mᵀ = (c/rms(c))·diag(w)·Mᵀ + b·Mᵀ` for `c = x − mean(x)`, so
/// a text folds `w` into the GEMM that reads the norm and `b·Mᵀ` into that
/// GEMM's bias at import, and writes this. Whole rows, no head grouping.
pub fn layernorm_no_scale(x: &Value, eps: f32) -> Value {
    let r = x.rec();
    let y = r.fresh(x.ty().clone());
    r.push(
        Elementwise::LayernormNoScale {
            x: x.id(),
            eps,
            y: y.id(),
        },
        &[x],
    );
    y
}

/// **THE WHOLE `nn.LayerNorm`, IN ONE NODE** (multimodal §9.1's owed saving,
/// next.md B5): centred, scaled by `weight`, biased by `bias`.
///
/// `y = (x − mean(x)) · rsqrt(var(x) + eps) · w + b`. What a qwen vision block
/// writes, and what it wrote before this row existed is three nodes —
/// `add_bias(b, rmsnorm(layernorm_no_scale(x, eps), w, eps))` — because §9.1
/// found the import fold half-expressible and the halves non-composing. Three
/// launches and two extra device rectangles per norm, twenty-five norms per
/// qwen35 tower fire; this is the one node they collapse to.
///
/// [`layernorm_no_scale`] stays for the text whose scale genuinely bakes.
pub fn layernorm(x: &Value, weight: &Weight, bias: &Weight, eps: f32) -> Value {
    let r = x.rec();
    let y = r.fresh(x.ty().clone());
    r.push(
        Elementwise::Layernorm {
            x: x.id(),
            weight: r.weight(weight),
            bias: r.weight(bias),
            eps,
            y: y.id(),
        },
        &[x],
    );
    y
}

/// **THE CLIPPED LINEAR'S CLAMP** (multimodal §6.5): `min(max(x, lo), hi)`,
/// in place, both bounds trace constants.
///
/// gemma4's `use_clipped_linears` publishes `{input,output}_{min,max}` beside
/// every vision projection; a text writes `clamp` before the matmul and after
/// it. Both bounds are the checkpoint's own numbers, so they are stated here
/// and never read from device memory.
/// **THE SAME CLAMP, WITH THE BOUNDS THE CHECKPOINT SHIPS** (multimodal
/// §12.2): `lo` and `hi` are `[1]` weight planes.
///
/// gemma4's `use_clipped_linears` is not a flag, it is 448 learned scalars —
/// `input_min`/`input_max` and `output_min`/`output_max` beside every vision
/// projection, all finite in the E4B checkpoint and all different. A text
/// cannot state them: `Model::new(w, kv, tp, dims)` has no checkpoint in the
/// room, and a catalog row carrying 448 numbers would be a checkpoint
/// transcribed into a `const`.
///
/// [`clamp`] stays for a bound the CONFIG states — `swiglu_limit` is one — the
/// way [`mul_scalar`] and [`scale`] both stay. Which one a text writes is a
/// question about where the number lives, and that is the only question.
pub fn clamp_learned(x: &Value, lo: &Weight, hi: &Weight) -> Value {
    let r = x.rec();
    let x_out = r.fresh(x.ty().clone());
    r.push(
        Elementwise::ClampLearned {
            x: x.id(),
            lo: r.weight(lo),
            hi: r.weight(hi),
            x_out: x_out.id(),
        },
        &[x],
    );
    x_out
}

pub fn clamp(x: &Value, lo: f32, hi: f32) -> Value {
    let r = x.rec();
    let x_out = r.fresh(x.ty().clone());
    r.push(
        Elementwise::Clamp {
            x: x.id(),
            lo,
            hi,
            x_out: x_out.id(),
        },
        &[x],
    );
    x_out
}

pub fn rmsnorm_gated(x: &Value, gate: &Value, weight: &Weight, head_dim: u32, eps: f32) -> Value {
    let r = x.rec();
    let y = r.fresh(gate.ty().clone());
    r.push(
        Elementwise::RmsnormGated {
            x: x.id(),
            gate: gate.id(),
            weight: r.weight(weight),
            head_dim,
            eps,
            y: y.id(),
        },
        &[x, gate],
    );
    y
}

pub fn rmsnorm_gated_by(x: &Value, gate: &Value, weight: &Weight, heads: u32, eps: f32) -> Value {
    let r = x.rec();
    let y = r.fresh(gate.ty().clone());
    r.push(
        Elementwise::RmsnormGatedBy {
            x: x.id(),
            gate: gate.id(),
            weight: r.weight(weight),
            heads,
            eps,
            y: y.id(),
        },
        &[x, gate],
    );
    y
}

pub fn residual_add(x: &Value, y: &Value) -> Value {
    let r = x.rec();
    let y_out = r.fresh(y.ty().clone());
    r.push(
        Elementwise::ResidualAdd {
            x: x.id(),
            y: y.id(),
            y_out: y_out.id(),
        },
        &[x, y],
    );
    y_out
}

pub fn add_bias(bias: &Weight, out: &Value) -> Value {
    let r = out.rec();
    let out_out = r.fresh(out.ty().clone());
    r.push(
        Elementwise::AddBias {
            bias: r.weight(bias),
            out: out.id(),
            out_out: out_out.id(),
        },
        &[out],
    );
    out_out
}

pub fn mul_scalar(s: f32, x: &Value) -> Value {
    let r = x.rec();
    let x_out = r.fresh(x.ty().clone());
    r.push(
        Elementwise::MulScalar {
            s,
            x: x.id(),
            x_out: x_out.id(),
        },
        &[x],
    );
    x_out
}

pub fn scale(s: &Weight, x: &Value) -> Value {
    let r = x.rec();
    let x_out = r.fresh(x.ty().clone());
    r.push(
        Elementwise::Scale {
            s: r.weight(s),
            x: x.id(),
            x_out: x_out.id(),
        },
        &[x],
    );
    x_out
}

pub fn res_blend(
    prefix: &Value,
    blocks: &[Value],
    norm: &Weight,
    eps: f32,
    proj: &Weight,
) -> Value {
    let r = prefix.rec();
    let y = r.fresh(prefix.ty().clone());
    let mut ins: Vec<&Value> = Vec::with_capacity(1 + blocks.len());
    ins.push(prefix);
    ins.extend(blocks);
    r.push(
        Elementwise::ResBlend {
            prefix: prefix.id(),
            blocks: blocks.iter().map(Value::id).collect(),
            weight: r.weight(norm),
            eps,
            proj: r.weight(proj),
            y: y.id(),
        },
        &ins,
    );
    y
}

pub fn rope_full(
    q: &Value,
    k: &Value,
    positions: &Value,
    head_dim: u32,
    theta: f32,
    interleaved: bool,
) -> (Value, Value) {
    let r = q.rec();
    let q_out = r.fresh(q.ty().clone());
    let k_out = r.fresh(k.ty().clone());
    r.push(
        Elementwise::RopeFull {
            q: q.id(),
            k: k.id(),
            positions: positions.id(),
            head_dim,
            theta,
            interleaved,
            q_out: q_out.id(),
            k_out: k_out.id(),
        },
        &[q, k, positions],
    );
    (q_out, k_out)
}

pub fn rope_partial(
    q: &Value,
    k: &Value,
    positions: &Value,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
) -> (Value, Value) {
    let r = q.rec();
    let q_out = r.fresh(q.ty().clone());
    let k_out = r.fresh(k.ty().clone());
    r.push(
        Elementwise::RopePartial {
            q: q.id(),
            k: k.id(),
            positions: positions.id(),
            rotary_dim,
            head_dim,
            theta,
            q_out: q_out.id(),
            k_out: k_out.id(),
        },
        &[q, k, positions],
    );
    (q_out, k_out)
}

pub fn rope_partial_q(
    q: &Value,
    positions: &Value,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
) -> Value {
    let r = q.rec();
    let q_out = r.fresh(q.ty().clone());
    r.push(
        Elementwise::RopePartialQ {
            q: q.id(),
            positions: positions.id(),
            rotary_dim,
            head_dim,
            theta,
            q_out: q_out.id(),
        },
        &[q, positions],
    );
    q_out
}

pub fn rope_partial_last(
    q: &Value,
    positions: &Value,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
    interleaved: bool,
) -> Value {
    let r = q.rec();
    let q_out = r.fresh(q.ty().clone());
    r.push(
        Elementwise::RopePartialLast {
            q: q.id(),
            positions: positions.id(),
            rotary_dim,
            head_dim,
            theta,
            interleaved,
            q_out: q_out.id(),
        },
        &[q, positions],
    );
    q_out
}

pub fn rope_yarn(
    q: &Value,
    k: &Value,
    positions: &Value,
    head_dim: u32,
    theta: f32,
    factor: f32,
    beta_fast: f32,
    beta_slow: f32,
    attention_factor: f32,
    original_max_position: u32,
    interleaved: bool,
) -> (Value, Value) {
    let r = q.rec();
    let q_out = r.fresh(q.ty().clone());
    let k_out = r.fresh(k.ty().clone());
    r.push(
        Elementwise::RopeYarn {
            q: q.id(),
            k: k.id(),
            positions: positions.id(),
            head_dim,
            theta,
            factor,
            beta_fast,
            beta_slow,
            attention_factor,
            original_max_position,
            interleaved,
            q_out: q_out.id(),
            k_out: k_out.id(),
        },
        &[q, k, positions],
    );
    (q_out, k_out)
}

pub fn gate_sigmoid_mul(x: &Value, gate: &Value) -> Value {
    let r = x.rec();
    let x_out = r.fresh(x.ty().clone());
    r.push(
        Elementwise::GateSigmoidMul {
            x: x.id(),
            gate: gate.id(),
            x_out: x_out.id(),
        },
        &[x, gate],
    );
    x_out
}

pub fn hc_expand(x: &Value, streams: u32) -> Value {
    let r = x.rec();
    let y = r.fresh(tensor(x.rows(), x.width() * u64::from(streams), x.dtype()));
    r.push(
        Elementwise::HcExpand {
            x: x.id(),
            streams,
            y: y.id(),
        },
        &[x],
    );
    y
}

pub fn hc_rmsnorm_f32(streams: &Value, eps: f32) -> Value {
    let r = streams.rec();
    let y = r.fresh(tensor(streams.rows(), streams.width(), Dtype::F32));
    r.push(
        Elementwise::HcRmsnormF32 {
            streams: streams.id(),
            eps,
            y: y.id(),
        },
        &[streams],
    );
    y
}

pub fn hc_gates(
    normed: &Value,
    streams: &Value,
    scale: &Weight,
    base: &Weight,
    stream_count: u32,
    gate_eps: f32,
    alpha: f32,
    sinkhorn: u32,
) -> (Value, Value, Value) {
    let r = normed.rec();
    let count = u64::from(stream_count);
    let x = r.fresh(tensor(
        streams.rows(),
        streams.width() / count,
        streams.dtype(),
    ));
    let post_mix = r.fresh(tensor(streams.rows(), count, Dtype::F32));
    let comb_mix = r.fresh(tensor(streams.rows(), count * count, Dtype::F32));
    r.push(
        Elementwise::HcGates {
            normed: normed.id(),
            streams: streams.id(),
            scale: r.weight(scale),
            base: r.weight(base),
            stream_count,
            gate_eps,
            alpha,
            sinkhorn,
            x: x.id(),
            post_mix: post_mix.id(),
            comb_mix: comb_mix.id(),
        },
        &[normed, streams],
    );
    (x, post_mix, comb_mix)
}

pub fn hc_fold(x: &Value, streams: &Value, post_mix: &Value, comb_mix: &Value) -> Value {
    let r = x.rec();
    let y = r.fresh(streams.ty().clone());
    r.push(
        Elementwise::HcFold {
            x: x.id(),
            streams: streams.id(),
            post_mix: post_mix.id(),
            comb_mix: comb_mix.id(),
            y: y.id(),
        },
        &[x, streams, post_mix, comb_mix],
    );
    y
}

/// **THE MULTIMODAL ROTARY**: [`rope_partial`] over a position that is a
/// triple (multimodal §2's second op).
///
/// `positions` is `[rows, 3]` `i32`, one `(t, h, w)` per rotated row, and
/// `sections` is the checkpoint's own `mrope_section` — a trace constant, so
/// it is stated here rather than read from device memory. Rotates in place,
/// like every rope arm beside it.
///
/// `form` is the section LAYOUT (multimodal §6.3): the trunk states
/// [`MropeForm::Interleaved`] (`mrope_interleaved: true`), the tower states
/// [`MropeForm::Blocked`], and the feeder is
/// [`Input::mrope_positions`](crate::Input::mrope_positions) on the token axis
/// or [`Input::patch_positions`](crate::Input::patch_positions) on the patch
/// one.
pub fn rope_mrope(
    q: &Value,
    k: &Value,
    positions: &Value,
    sections: [u32; 3],
    form: MropeForm,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
) -> (Value, Value) {
    let r = q.rec();
    let q_out = r.fresh(q.ty().clone());
    let k_out = r.fresh(k.ty().clone());
    r.push(
        Elementwise::RopeMrope {
            q: q.id(),
            k: k.id(),
            positions: positions.id(),
            sections,
            form,
            rotary_dim,
            head_dim,
            theta,
            q_out: q_out.id(),
            k_out: k_out.id(),
        },
        &[q, k, positions],
    );
    (q_out, k_out)
}
