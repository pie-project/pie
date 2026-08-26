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
