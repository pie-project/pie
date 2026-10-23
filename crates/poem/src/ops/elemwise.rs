use super::*;
use poem_ir::{NormAxis, NormWeight, Turn};

pub use poem_ir::ops::elemwise::Yarn;

pub use crate::rope::RopeScaling;

use crate::Recorder;

pub fn rmsnorm(x: &Value, weight: &Weight, eps: f32) -> Value {
    let r = x.rec();
    let y = r.fresh(x.ty().clone());
    r.push(
        Elemwise::Rmsnorm {
            x: x.id(),
            weight: NormWeight::Scale(r.weight(weight)),
            eps,
            y: y.id(),
            axis: NormAxis::Row,
        },
        &[x],
    );
    y
}

pub fn rmsnorm_per_head(x: &Value, weight: &Weight, head_dim: u32, eps: f32) -> Value {
    let r = x.rec();
    let y = r.fresh(x.ty().clone());
    r.push(
        Elemwise::Rmsnorm {
            x: x.id(),
            weight: NormWeight::Scale(r.weight(weight)),
            axis: NormAxis::Head(head_dim),
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
        Elemwise::Rmsnorm {
            x: x.id(),
            weight: NormWeight::PlusOne(r.weight(weight)),
            eps,
            y: y.id(),
            axis: NormAxis::Row,
        },
        &[x],
    );
    y
}

pub fn rmsnorm_per_head_plus_one(x: &Value, weight: &Weight, head_dim: u32, eps: f32) -> Value {
    let r = x.rec();
    let y = r.fresh(x.ty().clone());
    r.push(
        Elemwise::Rmsnorm {
            x: x.id(),
            weight: NormWeight::PlusOne(r.weight(weight)),
            axis: NormAxis::Head(head_dim),
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
        Elemwise::Rmsnorm {
            x: x.id(),
            axis: NormAxis::Head(head_dim),
            eps,
            y: y.id(),
            weight: NormWeight::None,
        },
        &[x],
    );
    y
}

pub fn layernorm_no_scale(x: &Value, eps: f32) -> Value {
    let r = x.rec();
    let y = r.fresh(x.ty().clone());
    r.push(
        Elemwise::Layernorm {
            x: x.id(),
            eps,
            y: y.id(),
            weight: None,
            bias: None,
        },
        &[x],
    );
    y
}

pub fn layernorm(x: &Value, weight: &Weight, bias: &Weight, eps: f32) -> Value {
    let r = x.rec();
    let y = r.fresh(x.ty().clone());
    r.push(
        Elemwise::Layernorm {
            x: x.id(),
            weight: Some(r.weight(weight)),
            bias: Some(r.weight(bias)),
            eps,
            y: y.id(),
        },
        &[x],
    );
    y
}

pub fn clamp_learned(x: &Value, lo: &Weight, hi: &Weight) -> Value {
    let r = x.rec();
    let x_out = r.fresh(x.ty().clone());
    r.push(
        Elemwise::Clamp {
            x: x.id(),
            lo: Some(r.weight(lo)),
            hi: Some(r.weight(hi)),
            x_out: x_out.id(),
            bounds: poem_ir::Bounds::Learned,
        },
        &[x],
    );
    x_out
}

pub fn clamp(x: &Value, lo: f32, hi: f32) -> Value {
    let r = x.rec();
    let x_out = r.fresh(x.ty().clone());
    r.push(
        Elemwise::Clamp {
            x: x.id(),
            x_out: x_out.id(),
            lo: None,
            hi: None,
            bounds: poem_ir::Bounds::Fixed { lo, hi },
        },
        &[x],
    );
    x_out
}

pub fn rmsnorm_gated(
    x: &Value,
    gate: &Value,
    weight: Option<&Weight>,
    head_dim: u32,
    eps: f32,
    act: GateActivation,
) -> Value {
    let r = x.rec();
    let y = r.fresh(gate.ty().clone());
    r.push(
        Elemwise::RmsnormGated {
            x: x.id(),
            gate: gate.id(),
            weight: weight.map(|w| r.weight(w)),
            head_dim,
            eps,
            act,
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
        Elemwise::RmsnormGatedBy {
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

pub fn rmsnorm_grouped_plus_one(x: &Value, weight: &Weight, group: u32, eps: f32) -> Value {
    let r = x.rec();
    let y = r.fresh(x.ty().clone());
    r.push(
        Elemwise::Rmsnorm {
            x: x.id(),
            weight: NormWeight::PlusOne(r.weight(weight)),
            axis: NormAxis::Group(group),
            eps,
            y: y.id(),
        },
        &[x],
    );
    y
}

pub fn residual_add(x: &Value, y: &Value) -> Value {
    let r = x.rec();
    let y_out = r.fresh(y.ty().clone());
    r.push(
        Elemwise::ResidualAdd {
            x: x.id(),
            y: y.id(),
            y_out: y_out.id(),
        },
        &[x, y],
    );
    y_out
}

pub fn copy(x: &Value) -> Value {
    mul_scalar(0.5, &add(x, x))
}

pub fn add_bias(bias: &Weight, out: &Value) -> Value {
    let r = out.rec();
    let out_out = r.fresh(out.ty().clone());
    r.push(
        Elemwise::AddBias {
            bias: r.weight(bias),
            out: out.id(),
            out_out: out_out.id(),
        },
        &[out],
    );
    out_out
}

pub fn standardize(x: &Value, bias: &Weight, scale: &Weight) -> Value {
    let r = x.rec();
    let x_out = r.fresh(x.ty().clone());
    r.push(
        Elemwise::Standardize {
            x: x.id(),
            bias: r.weight(bias),
            scale: r.weight(scale),
            x_out: x_out.id(),
        },
        &[x],
    );
    x_out
}

pub fn mul_scalar(s: f32, x: &Value) -> Value {
    let r = x.rec();
    let x_out = r.fresh(x.ty().clone());
    r.push(
        Elemwise::MulScalar {
            s,
            x: x.id(),
            x_out: x_out.id(),
        },
        &[x],
    );
    x_out
}

pub fn silu_scaled(s: f32, x: &Value) -> Value {
    let r = x.rec();
    let x_out = r.fresh(x.ty().clone());
    r.push(
        Elemwise::SiluScaled {
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
        Elemwise::Scale {
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
        Elemwise::ResBlend {
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

/// `scaling`, if any, turns each pair at its own frequency (a checkpoint's
/// `rope_scaling`), the rest of the op as stated.
pub fn rope_full(
    q: &Value,
    k: &Value,
    positions: &Value,
    head_dim: u32,
    theta: f32,
    interleaved: bool,
    scaling: Option<RopeScaling>,
) -> (Value, Value) {
    let freqs = scaled(scaling.as_ref(), theta, head_dim / 2, head_dim / 2);
    let r = q.rec();
    let q_out = r.fresh(q.ty().clone());
    let k_out = r.fresh(k.ty().clone());
    r.push(
        Elemwise::Rope {
            q: q.id(),
            k: Some(k.id()),
            positions: positions.id(),
            head_dim,
            theta,
            turn: Turn::Full { interleaved },
            freqs,
            q_out: q_out.id(),
            k_out: Some(k_out.id()),
        },
        &[q, k, positions],
    );
    (q_out, k_out)
}

/// The frequencies `scaling` turns `pairs` pairs `span` apart at.
fn scaled(
    scaling: Option<&RopeScaling>,
    theta: f32,
    pairs: u32,
    span: u32,
) -> Option<poem_ir::RopeFreqs> {
    scaling.map(|s| {
        s.freqs(theta, pairs, span)
            .unwrap_or_else(|why| panic!("{why}"))
    })
}

/// A pair sits `span` apart, half the rotary width unless stated.
pub fn rope_partial(
    q: &Value,
    k: &Value,
    positions: &Value,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
    span: Option<u32>,
    scaling: Option<RopeScaling>,
) -> (Value, Value) {
    let span = span.unwrap_or(rotary_dim / 2);
    let freqs = scaled(scaling.as_ref(), theta, rotary_dim / 2, span);
    let r = q.rec();
    let q_out = r.fresh(q.ty().clone());
    let k_out = r.fresh(k.ty().clone());
    r.push(
        Elemwise::Rope {
            q: q.id(),
            k: Some(k.id()),
            positions: positions.id(),
            head_dim,
            theta,
            turn: Turn::Partial { rotary_dim, span },
            freqs,
            q_out: q_out.id(),
            k_out: Some(k_out.id()),
        },
        &[q, k, positions],
    );
    (q_out, k_out)
}

/// A pair sits `span` apart, half the rotary width unless stated.
pub fn rope_partial_q(
    q: &Value,
    positions: &Value,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
    span: Option<u32>,
    scaling: Option<RopeScaling>,
) -> Value {
    let span = span.unwrap_or(rotary_dim / 2);
    let freqs = scaled(scaling.as_ref(), theta, rotary_dim / 2, span);
    let r = q.rec();
    let q_out = r.fresh(q.ty().clone());
    r.push(
        Elemwise::Rope {
            q: q.id(),
            k: None,
            positions: positions.id(),
            head_dim,
            theta,
            turn: Turn::Partial { rotary_dim, span },
            freqs,
            q_out: q_out.id(),
            k_out: None,
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
    rope_partial_last_yarn(
        q,
        positions,
        rotary_dim,
        head_dim,
        theta,
        interleaved,
        false,
        None,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn rope_partial_last_yarn(
    q: &Value,
    positions: &Value,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
    interleaved: bool,
    inverse: bool,
    yarn: Option<Yarn>,
) -> Value {
    let r = q.rec();
    let q_out = r.fresh(q.ty().clone());
    r.push(
        Elemwise::Rope {
            q: q.id(),
            k: None,
            positions: positions.id(),
            head_dim,
            theta,
            turn: Turn::Last {
                rotary_dim,
                interleaved,
                inverse,
                yarn,
            },
            freqs: None,
            q_out: q_out.id(),
            k_out: None,
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
        Elemwise::Rope {
            q: q.id(),
            k: Some(k.id()),
            positions: positions.id(),
            head_dim,
            theta,
            turn: Turn::Yarn {
                yarn: Yarn {
                    factor,
                    beta_fast,
                    beta_slow,
                    original_max_position,
                },
                attention_factor,
                interleaved,
            },
            freqs: None,
            q_out: q_out.id(),
            k_out: Some(k_out.id()),
        },
        &[q, k, positions],
    );
    (q_out, k_out)
}

pub fn gate_sigmoid_mul(x: &Value, gate: &Value) -> Value {
    let r = x.rec();
    let x_out = r.fresh(x.ty().clone());
    r.push(
        Elemwise::GateSigmoidMul {
            x: x.id(),
            gate: gate.id(),
            x_out: x_out.id(),
        },
        &[x, gate],
    );
    x_out
}

pub fn gate_sigmoid_mul_heads(x: &Value, gate: &Value, head_dim: u32, scale: f32) -> Value {
    assert!(head_dim > 0, "a head is at least one channel wide");
    assert!(
        x.width().is_multiple_of(u64::from(head_dim)),
        "x is {} wide, not a whole number of {head_dim}-wide heads",
        x.width()
    );
    assert_eq!(
        gate.width(),
        x.width() / u64::from(head_dim),
        "one gate logit per head"
    );
    assert_eq!(gate.rows(), x.rows(), "the gate is per row");
    assert_eq!(
        gate.dtype(),
        x.dtype(),
        "the gate rides the rectangle's dtype"
    );
    let r = x.rec();
    let x_out = r.fresh(x.ty().clone());
    r.push(
        Elemwise::GateSigmoidMulHeads {
            x: x.id(),
            gate: gate.id(),
            head_dim,
            scale,
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
        Elemwise::HcExpand {
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
        Elemwise::HcRmsnormF32 {
            streams: streams.id(),
            eps,
            y: y.id(),
        },
        &[streams],
    );
    y
}

pub fn hc_project(normed: &Value, dynamic: &Weight, stream_count: u32) -> Value {
    let r = normed.rec();
    let count = u64::from(stream_count);
    let mix_hc = 2 * count + count * count;
    assert!(
        dynamic.dim(0) == mix_hc || dynamic.dim(0) == 2 * count || dynamic.dim(0) == count,
        "`{}` lands {} rows; a {stream_count}-stream mix row is {mix_hc} wide ({} without \
         its combine), and a trunk collapse row is {count}",
        dynamic.name,
        dynamic.dim(0),
        2 * count,
    );
    let mixes = r.fresh(tensor(normed.rows(), dynamic.dim(0), Dtype::F32));
    r.push(
        Elemwise::HcProject {
            normed: normed.id(),
            weight: r.weight(dynamic),
            stream_count,
            mixes: mixes.id(),
        },
        &[normed],
    );
    mixes
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
    gates(
        normed,
        streams,
        scale,
        base,
        stream_count,
        gate_eps,
        alpha,
        sinkhorn,
        true,
    )
}

/// `hc_gates` in the independent form: the mix row is the pre and post logits
/// alone, the post gate carries `gate_eps` as the pre gate does, and the
/// combine landed is the identity, so a fold adds each stream to itself.
#[allow(clippy::too_many_arguments)]
pub fn hc_gates_independent(
    mixes: &Value,
    streams: &Value,
    scale: &Weight,
    base: &Weight,
    stream_count: u32,
    gate_eps: f32,
    alpha: f32,
) -> (Value, Value, Value) {
    gates(
        mixes,
        streams,
        scale,
        base,
        stream_count,
        gate_eps,
        alpha,
        0,
        false,
    )
}

#[allow(clippy::too_many_arguments)]
fn gates(
    normed: &Value,
    streams: &Value,
    scale: &Weight,
    base: &Weight,
    stream_count: u32,
    gate_eps: f32,
    alpha: f32,
    sinkhorn: u32,
    comb: bool,
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
        Elemwise::HcGates {
            normed: normed.id(),
            streams: streams.id(),
            scale: r.weight(scale),
            base: r.weight(base),
            stream_count,
            gate_eps,
            alpha,
            sinkhorn,
            comb,
            x: x.id(),
            post_mix: post_mix.id(),
            comb_mix: comb_mix.id(),
        },
        &[normed, streams],
    );
    (x, post_mix, comb_mix)
}

pub fn hc_collapse(
    mixes: &Value,
    streams: &Value,
    scale: &Weight,
    base: &Weight,
    stream_count: u32,
    hc_eps: f32,
) -> Value {
    let r = mixes.rec();
    let count = u64::from(stream_count);
    let y = r.fresh(tensor(
        streams.rows(),
        streams.width() / count,
        streams.dtype(),
    ));
    r.push(
        Elemwise::HcCollapse {
            mixes: mixes.id(),
            streams: streams.id(),
            scale: r.weight(scale),
            base: r.weight(base),
            stream_count,
            hc_eps,
            y: y.id(),
        },
        &[mixes, streams],
    );
    y
}

pub fn hc_fold(x: &Value, streams: &Value, post_mix: &Value, comb_mix: &Value) -> Value {
    let r = x.rec();
    let y = r.fresh(streams.ty().clone());
    r.push(
        Elemwise::HcFold {
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

pub fn hc_mix(gates: &Value, normed: &Value, streams: u32) -> Value {
    let r = gates.rec();
    let y = r.fresh(tensor(
        normed.rows(),
        normed.width() / u64::from(streams),
        normed.dtype(),
    ));
    r.push(
        Elemwise::HcMix {
            gates: gates.id(),
            normed: normed.id(),
            streams,
            y: y.id(),
        },
        &[gates, normed],
    );
    y
}

pub fn hc_inject(o: &Value, gates: &Value, streams: u32, hyper: &Value) -> Value {
    let r = o.rec();
    let hyper_out = r.fresh(hyper.ty().clone());
    r.push(
        Elemwise::HcInject {
            o: o.id(),
            gates: gates.id(),
            streams,
            hyper: hyper.id(),
            hyper_out: hyper_out.id(),
        },
        &[o, gates, hyper],
    );
    hyper_out
}

pub fn ple_gate(key: &Value, query: &Value, value: &Value, streams: u32) -> Value {
    let r = key.rec();
    let y = r.fresh(key.ty().clone());
    r.push(
        Elemwise::PleGate {
            key: key.id(),
            query: query.id(),
            value: value.id(),
            streams,
            y: y.id(),
        },
        &[key, query, value],
    );
    y
}

pub fn rope_mrope(
    q: &Value,
    k: &Value,
    positions: &Value,
    sections: [u32; 3],
    form: MropeForm,
    rotary_dim: u32,
    head_dim: u32,
    theta: f32,
    scaling: Option<RopeScaling>,
) -> (Value, Value) {
    assert!(
        scaling.is_none() || form == MropeForm::Interleaved,
        "a {form:?} M-RoPE turns a still's grid, which no rope scaling stretches"
    );
    let freqs = scaled(scaling.as_ref(), theta, rotary_dim / 2, rotary_dim / 2);
    let r = q.rec();
    let q_out = r.fresh(q.ty().clone());
    let k_out = r.fresh(k.ty().clone());
    r.push(
        Elemwise::Rope {
            q: q.id(),
            k: Some(k.id()),
            positions: positions.id(),
            head_dim,
            theta,
            turn: Turn::Mrope {
                sections,
                form,
                rotary_dim,
            },
            freqs,
            q_out: q_out.id(),
            k_out: Some(k_out.id()),
        },
        &[q, k, positions],
    );
    (q_out, k_out)
}

pub fn modulate(x: &Value, m: &Value, lane_of_row: Option<&Value>, form: ModulateForm) -> Value {
    let r = x.rec();
    assert_eq!(
        m.width(),
        form.slices() * x.width(),
        "a {form:?} modulation of a {}-wide row wants a {}-wide vector, not {}",
        x.width(),
        form.slices() * x.width(),
        m.width()
    );
    match lane_of_row {
        Some(lanes) => {
            assert_eq!(m.rows(), Dim::Lanes, "a lane-broadcast vector is per lane");
            assert_eq!(
                lanes.rows(),
                x.rows(),
                "the lane map is over the rows it maps"
            );
        }
        None => assert_eq!(
            m.rows(),
            x.rows(),
            "a per-row vector shares the rows it modulates"
        ),
    }
    let y = r.fresh(x.ty().clone());
    let mut ins = vec![x, m];
    ins.extend(lane_of_row);
    r.push(
        Elemwise::Modulate {
            x: x.id(),
            m: m.id(),
            lane_of_row: lane_of_row.map(Value::id),
            form,
            y: y.id(),
        },
        &ins,
    );
    y
}

pub fn gated_residual_add(
    r_in: &Value,
    g: &Value,
    y: &Value,
    lane_of_row: Option<&Value>,
) -> Value {
    let r = r_in.rec();
    assert_eq!(
        r_in.ty(),
        y.ty(),
        "the stream and what joins it share a type"
    );
    assert_eq!(g.width(), r_in.width(), "one gate per column");
    match lane_of_row {
        Some(lanes) => {
            assert_eq!(g.rows(), Dim::Lanes, "a lane-broadcast gate is per lane");
            assert_eq!(
                lanes.rows(),
                r_in.rows(),
                "the lane map is over the rows it maps"
            );
        }
        None => assert_eq!(
            g.rows(),
            r_in.rows(),
            "a per-row gate shares the rows it gates"
        ),
    }
    let r_out = r.fresh(r_in.ty().clone());
    let mut ins = vec![r_in, g, y];
    ins.extend(lane_of_row);
    r.push(
        Elemwise::GatedResidualAdd {
            r: r_in.id(),
            g: g.id(),
            y: y.id(),
            lane_of_row: lane_of_row.map(Value::id),
            r_out: r_out.id(),
        },
        &ins,
    );
    r_out
}

pub fn sinusoid(t: &Value, dim: u32, max_period: f32, flip_sin_cos: bool, scale: f32) -> Value {
    let r = t.rec();
    assert_eq!(t.width(), 1, "a timestep is one scalar per row");
    assert_eq!(t.dtype(), Dtype::F32, "timesteps are fp32");
    assert!(
        dim > 0 && dim.is_multiple_of(2),
        "a sinusoid of {dim} has no equal halves"
    );
    let y = r.fresh(tensor(t.rows(), dim, Dtype::F32));
    r.push(
        Elemwise::Sinusoid {
            t: t.id(),
            dim,
            max_period,
            flip_sin_cos,
            scale,
            y: y.id(),
        },
        &[t],
    );
    y
}

pub fn relative_bucket_bias(
    r: &Recorder,
    embedding: &Weight,
    max_len: u32,
    num_buckets: u32,
    max_distance: f32,
    bidirectional: bool,
) -> Value {
    assert!(
        embedding.shape.len() == 2 && embedding.shape[0] == u64::from(num_buckets),
        "`{}` is {:?}; a bucket embedding is `[num_buckets = {num_buckets}, heads]`",
        embedding.name,
        embedding.shape
    );
    assert!(
        matches!(embedding.dtype, Dtype::Bf16 | Dtype::F32),
        "`{}` is {:?}; the table reads a bf16 or f32 embedding",
        embedding.name,
        embedding.dtype
    );
    let directional = if bidirectional {
        num_buckets / 2
    } else {
        num_buckets
    };
    let max_exact = directional / 2;
    assert!(max_exact > 0, "{num_buckets} buckets leave no exact band");
    assert!(
        max_distance > max_exact as f32,
        "max_distance {max_distance} is at or below max_exact {max_exact}"
    );
    assert!(max_len > 0, "a table over no positions has no width");
    let heads = embedding.shape[1];
    let y = r.fresh(Ty::Tensor {
        shape: vec![Dim::Const(heads), Dim::Const(2 * u64::from(max_len) - 1)],
        dtype: Dtype::F32,
    });
    r.push(
        Elemwise::RelativeBucketBias {
            embedding: r.weight(embedding),
            max_len,
            num_buckets,
            max_distance,
            bidirectional,
            y: y.id(),
        },
        &[],
    );
    y
}

pub fn silu(x: &Value) -> Value {
    let r = x.rec();
    let x_out = r.fresh(x.ty().clone());
    r.push(
        Elemwise::Silu {
            x: x.id(),
            x_out: x_out.id(),
        },
        &[x],
    );
    x_out
}

pub fn gelu(x: &Value, tanh: bool) -> Value {
    let r = x.rec();
    let x_out = r.fresh(x.ty().clone());
    r.push(
        Elemwise::Gelu {
            x: x.id(),
            tanh,
            x_out: x_out.id(),
        },
        &[x],
    );
    x_out
}

pub fn tanh(x: &Value) -> Value {
    let r = x.rec();
    let x_out = r.fresh(x.ty().clone());
    r.push(
        Elemwise::Tanh {
            x: x.id(),
            x_out: x_out.id(),
        },
        &[x],
    );
    x_out
}

/// A blockwise Hadamard (butterfly FWHT) over the last dim. `block` is the
/// power-of-two transform size; each contiguous `block`-vector is turned by the
/// normalized Sylvester Hadamard matrix. When `signs` is `Some`, its ±1 entries
/// multiply the activation on load, before the butterfly — the Randomized
/// Hadamard `H·S`; the sign vector repeats block-wise if it is wider than one
/// block.
pub fn hadamard(x: &Value, block: u32, signs: Option<&Value>) -> Value {
    let r = x.rec();
    let x_out = r.fresh(x.ty().clone());
    let mut ins: Vec<&Value> = vec![x];
    if let Some(s) = signs {
        ins.push(s);
    }
    r.push(
        Elemwise::Hadamard {
            x: x.id(),
            x_out: x_out.id(),
            block,
            signs: signs.map(Value::id),
        },
        &ins,
    );
    x_out
}

/// A pure blockwise Hadamard, with no sign diagonal.
pub fn hadamard_plain(x: &Value, block: u32) -> Value {
    hadamard(x, block, None)
}

/// The Randomized Hadamard `H·(S·x)` with the ±1 sign diagonal `S` supplied as a
/// parameter bank (a registered/checkpoint [`Weight`]) rather than a live value.
/// The signs multiply the activation BEFORE the butterfly, matching the fork's
/// `MUL(input, prism.hadamard.signs.<W>)` → `MUL_MAT(prism.hadamard.<block>, …)`
/// chain exactly. The sign bank's element type must equal the activation's
/// compute dtype (`±1` is exact in both `bf16` and `f32`).
pub fn hadamard_signed(x: &Value, block: u32, signs: &Weight) -> Value {
    let r = x.rec();
    // A parameter bank is resolved from the op's `ValueId` field (as `matmul`
    // and `scale` resolve their weights) — it has no producer, so it does not
    // ride the `ins` scheduling list.
    let signs_id = r.weight(signs);
    let x_out = r.fresh(x.ty().clone());
    r.push(
        Elemwise::Hadamard {
            x: x.id(),
            x_out: x_out.id(),
            block,
            signs: Some(signs_id),
        },
        &[x],
    );
    x_out
}

/// Muse Spark's post-norm and residual gate (`elemwise.norm_gate`):
/// `x` times `scale` (rounded), RMS-normed unscaled, and with `res` and its
/// `gate` the residual's gated sum; else the norm alone.
/// With `next` (`weight, eps, plus_one`), also the norm the next sublayer
/// reads over the output, landed in the same pass.
pub fn norm_gate(
    x: &Value,
    scale: Option<&Weight>,
    res: Option<(&Value, &Weight)>,
    eps: f32,
    temperature: f32,
    next: Option<(&Weight, f32, bool)>,
) -> (Value, Option<Value>) {
    let r = x.rec();
    let y = r.fresh(x.ty().clone());
    let z = next.map(|_| r.fresh(x.ty().clone()));
    let mut ins = vec![x];
    if let Some((stream, _)) = res {
        assert_eq!(
            stream.ty(),
            x.ty(),
            "the residual and the normed row share a type"
        );
        ins.push(stream);
    }
    r.push(
        Elemwise::NormGate {
            x: x.id(),
            scale: scale.map(|s| r.weight(s)),
            res: res.map(|(stream, _)| stream.id()),
            gate: res.map(|(_, gate)| r.weight(gate)),
            eps,
            temperature,
            next: next
                .zip(z.as_ref())
                .map(|((w, eps, plus_one), z)| poem_ir::NormNext {
                    weight: r.weight(w),
                    eps,
                    plus_one,
                    y: z.id(),
                }),
            y: y.id(),
        },
        &ins,
    );
    (y, z)
}

pub fn mul(x: &Value, y: &Value) -> Value {
    let r = x.rec();
    assert_eq!(x.ty(), y.ty(), "a product's operands share a type");
    let z = r.fresh(x.ty().clone());
    r.push(
        Elemwise::Mul {
            x: x.id(),
            y: y.id(),
            z: z.id(),
        },
        &[x, y],
    );
    z
}

pub fn add(x: &Value, y: &Value) -> Value {
    let r = x.rec();
    assert_eq!(x.ty(), y.ty(), "a sum's operands share a type");
    let z = r.fresh(x.ty().clone());
    r.push(
        Elemwise::Add {
            x: x.id(),
            y: y.id(),
            z: z.id(),
        },
        &[x, y],
    );
    z
}

pub fn rope_axes(
    x: &Value,
    positions: &Value,
    dims: [u32; 4],
    thetas: [f32; 4],
    form: RopeForm,
    rotary_dim: u32,
    head_dim: u32,
) -> Value {
    let r = x.rec();
    let axes = dims.iter().take_while(|d| **d > 0).count();
    assert!(axes > 0, "a rope turns at least one axis");
    assert!(
        dims[axes..].iter().all(|d| *d == 0),
        "the axes of {dims:?} are stated first, the zeros last"
    );
    assert_eq!(
        positions.width(),
        axes as u64,
        "{axes} axes of dims want {axes} position columns, not {}",
        positions.width()
    );
    assert_eq!(
        positions.rows(),
        x.rows(),
        "positions are over the rows they turn"
    );
    assert_eq!(positions.dtype(), Dtype::F32, "axis positions are fp32");
    assert!(
        rotary_dim <= head_dim
            && rotary_dim.is_multiple_of(2)
            && dims.iter().all(|d| d.is_multiple_of(2)),
        "rotary_dim {rotary_dim} within head_dim {head_dim}, every axis an even count"
    );
    assert!(
        x.width().is_multiple_of(u64::from(head_dim)),
        "x is {} wide, not a whole number of {head_dim}-wide heads",
        x.width()
    );
    if form == RopeForm::SplitLadder {
        assert_eq!(
            rotary_dim, head_dim,
            "the ladder pairs (i, i + head_dim/2), so it turns the whole head"
        );
        let span: u64 = dims.iter().map(|d| u64::from(*d)).sum();
        assert!(
            span <= x.width() && (x.width() - span).is_multiple_of(2),
            "the ladder's {span} channels leave no whole pad in a {}-wide row",
            x.width()
        );
        assert!(
            dims[..axes].iter().all(|d| *d == dims[0]),
            "one ladder hands its axes out round-robin, so {dims:?} must be flat"
        );
    } else {
        assert_eq!(
            dims.iter().sum::<u32>(),
            rotary_dim,
            "the axes of {dims:?} do not sum to rotary_dim {rotary_dim}"
        );
    }
    let x_out = r.fresh(x.ty().clone());
    r.push(
        Elemwise::RopeAxes {
            x: x.id(),
            positions: positions.id(),
            dims,
            thetas,
            form,
            rotary_dim,
            head_dim,
            x_out: x_out.id(),
        },
        &[x, positions],
    );
    x_out
}
