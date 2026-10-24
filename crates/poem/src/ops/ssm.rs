use super::*;

pub fn causal_conv1d(
    x: &Value,
    weight: &Weight,
    bias: Option<&Weight>,
    state: ValueId,
    conv_width: u32,
) -> Value {
    causal_conv1d_dilated(x, weight, bias, state, conv_width, 1)
}

pub fn causal_conv1d_dilated(
    x: &Value,
    weight: &Weight,
    bias: Option<&Weight>,
    state: ValueId,
    conv_width: u32,
    dilation: u32,
) -> Value {
    let r = x.rec();
    let y = r.fresh(x.ty().clone());
    r.push(
        Ssm::CausalConv1d {
            x: x.id(),
            weight: r.weight(weight),
            bias: bias.map(|b| r.weight(b)),
            state,
            conv_width,
            dilation,
            y: y.id(),
        },
        &[x],
    );
    y
}

pub fn causal_conv1d_chunked(
    x: &Value,
    weight: &Weight,
    bias: Option<&Weight>,
    state: ValueId,
    conv_width: u32,
) -> Value {
    causal_conv1d_chunked_dilated(x, weight, bias, state, conv_width, 1)
}

pub fn causal_conv1d_chunked_dilated(
    x: &Value,
    weight: &Weight,
    bias: Option<&Weight>,
    state: ValueId,
    conv_width: u32,
    dilation: u32,
) -> Value {
    let r = x.rec();
    let y = r.fresh(x.ty().clone());
    r.push(
        Ssm::CausalConv1dChunked {
            x: x.id(),
            weight: r.weight(weight),
            bias: bias.map(|b| r.weight(b)),
            state,
            conv_width,
            dilation,
            y: y.id(),
        },
        &[x],
    );
    y
}

pub fn short_conv(x: &Value, weight: &Weight, state: ValueId, conv_width: u32) -> Value {
    let r = x.rec();
    let y = r.fresh(x.ty().clone());
    r.push(
        Ssm::ShortConv {
            x: x.id(),
            weight: r.weight(weight),
            state,
            conv_width,
            y: y.id(),
        },
        &[x],
    );
    y
}

pub fn short_conv_chunked(x: &Value, weight: &Weight, state: ValueId, conv_width: u32) -> Value {
    let r = x.rec();
    let y = r.fresh(x.ty().clone());
    r.push(
        Ssm::ShortConvChunked {
            x: x.id(),
            weight: r.weight(weight),
            state,
            conv_width,
            y: y.id(),
        },
        &[x],
    );
    y
}

pub fn dyn_conv(
    x: &Value,
    coeff: &Value,
    base: &Weight,
    side: u32,
    taps: u32,
    group: u32,
) -> Value {
    let r = x.rec();
    let y = r.fresh(x.ty().clone());
    r.push(
        Ssm::DynConv {
            x: x.id(),
            coeff: coeff.id(),
            base: r.weight(base),
            side,
            taps,
            group,
            y: y.id(),
        },
        &[x, coeff],
    );
    y
}

pub fn gdn_gates(ba: &Value, dt_bias: &Weight, a_log: &Weight) -> Value {
    let r = ba.rec();
    let gates = r.fresh(tensor(ba.rows(), ba.width(), Dtype::F32));
    r.push(
        Ssm::GdnGates {
            ba: ba.id(),
            dt_bias: r.weight(dt_bias),
            a_log: r.weight(a_log),
            gates: gates.id(),
        },
        &[ba],
    );
    gates
}

pub fn gated_delta(
    qkv: &Value,
    z: &Value,
    gates: &Value,
    state: ValueId,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
) -> Value {
    let r = qkv.rec();
    let width = u64::from(v_heads) * u64::from(v_dim);
    let y = r.fresh(tensor(qkv.rows(), width, Dtype::F32));
    r.push(
        Ssm::GatedDelta {
            qkv: qkv.id(),
            z: z.id(),
            gates: gates.id(),
            state,
            k_heads,
            v_heads,
            k_dim,
            v_dim,
            y: y.id(),
        },
        &[qkv, z, gates],
    );
    y
}

pub fn gated_delta_chunked(
    qkv: &Value,
    z: &Value,
    gates: &Value,
    state: ValueId,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
) -> Value {
    let r = qkv.rec();
    let width = u64::from(v_heads) * u64::from(v_dim);
    let y = r.fresh(tensor(qkv.rows(), width, Dtype::F32));
    r.push(
        Ssm::GatedDeltaChunked {
            qkv: qkv.id(),
            z: z.id(),
            gates: gates.id(),
            state,
            k_heads,
            v_heads,
            k_dim,
            v_dim,
            y: y.id(),
        },
        &[qkv, z, gates],
    );
    y
}

pub fn kda(
    mixed: &Value,
    f: &Value,
    b: &Value,
    dt_bias: &Weight,
    a_log: &Weight,
    state: ValueId,
    heads: u32,
    head_dim: u32,
    norm_eps: f32,
    gate_floor: f32,
) -> Value {
    let r = mixed.rec();
    let width = u64::from(heads) * u64::from(head_dim);
    let y = r.fresh(tensor(mixed.rows(), width, Dtype::F32));
    r.push(
        Ssm::Kda {
            mixed: mixed.id(),
            f: f.id(),
            b: b.id(),
            dt_bias: r.weight(dt_bias),
            a_log: r.weight(a_log),
            state,
            heads,
            head_dim,
            norm_eps,
            gate_floor,
            y: y.id(),
        },
        &[mixed, f, b],
    );
    y
}

pub fn kda_chunked(
    mixed: &Value,
    f: &Value,
    b: &Value,
    dt_bias: &Weight,
    a_log: &Weight,
    state: ValueId,
    heads: u32,
    head_dim: u32,
    norm_eps: f32,
    gate_floor: f32,
) -> Value {
    let r = mixed.rec();
    let width = u64::from(heads) * u64::from(head_dim);
    let y = r.fresh(tensor(mixed.rows(), width, Dtype::F32));
    r.push(
        Ssm::KdaChunked {
            mixed: mixed.id(),
            f: f.id(),
            b: b.id(),
            dt_bias: r.weight(dt_bias),
            a_log: r.weight(a_log),
            state,
            heads,
            head_dim,
            norm_eps,
            gate_floor,
            y: y.id(),
        },
        &[mixed, f, b],
    );
    y
}

/// Mamba-2's scan over the convolved `xbc` rows, gated by `silu(z)` and
/// RMS-normed a group; bf16 rows `z` wide. `chunked` walks prompts, else one
/// step a request.
#[allow(clippy::too_many_arguments)]
pub fn mamba2(
    xbc: &Value,
    dt: &Value,
    z: &Value,
    a_log: &Weight,
    dt_bias: &Weight,
    d: &Weight,
    norm: &Weight,
    eps: f32,
    state: ValueId,
    heads: u32,
    groups: u32,
    state_dim: u32,
    chunked: bool,
) -> Value {
    let r = xbc.rec();
    let y = r.fresh(tensor(xbc.rows(), z.width(), Dtype::Bf16));
    macro_rules! mamba2 {
        ($variant:ident) => {
            Ssm::$variant {
                xbc: xbc.id(),
                dt: dt.id(),
                z: z.id(),
                a_log: r.weight(a_log),
                dt_bias: r.weight(dt_bias),
                d: r.weight(d),
                norm: r.weight(norm),
                eps,
                state,
                heads,
                groups,
                state_dim,
                y: y.id(),
            }
        };
    }
    let op = if chunked {
        mamba2!(Mamba2Chunked)
    } else {
        mamba2!(Mamba2)
    };
    r.push(op, &[xbc, dt, z]);
    y
}
