#![allow(unused_variables)]
#![allow(clippy::too_many_arguments)]

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
use crate::error::Error;
use crate::tensor::{RaggedTensor, RecurrentPool, Tensor};
use dtype::Dtype;

const CONV_GROUP: u32 = 256;

const SCAN_WIDTH: u32 = 128;

const SCAN_HEAD_MAX: u32 = 256;

const CONV_HIST_MAX: u32 = 64;

const KDA: &str = "ssm/kda.slang";

fn conv_grid(channels: u32, rows: u32) -> Grid {
    Grid::of([channels, rows, 1], [CONV_GROUP, 1, 1])
}

const fn recurrence_grid(heads: u32, rows: u32) -> Grid {
    Grid::of([SCAN_WIDTH, heads, rows], [SCAN_WIDTH, 1, 1])
}

fn head_width(op: &'static str, width: u32, what: &'static str) -> Result<(), Error> {
    if width > SCAN_HEAD_MAX {
        return Err(refuse(
            op,
            format!("{what} is {width}, above the {SCAN_HEAD_MAX}-wide row this scan stages"),
        ));
    }
    Ok(())
}

fn requests(op: &'static str, x: RaggedTensor) -> Result<u32, Error> {
    if x.indptr.dtype != Dtype::I32 {
        return Err(refuse(
            op,
            format!(
                "the query CSR's boundaries are {:?}, and this scan walks an i32 indptr",
                x.indptr.dtype
            ),
        ));
    }
    match x.indptr.rows.checked_sub(1) {
        Some(lanes) if lanes > 0 => Ok(lanes),
        _ => Err(refuse(op, "the query CSR this fire names spans no request")),
    }
}

struct Delta {
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
}

impl Delta {
    fn of(
        op: &'static str,
        qkv: Tensor,
        gates: Tensor,
        y: Tensor,
        k_heads: u32,
        v_heads: u32,
        k_dim: u32,
        v_dim: u32,
    ) -> Result<Self, Error> {
        nonzero(op, "the key heads this statement states", k_heads)?;
        nonzero(op, "the value heads this statement states", v_heads)?;
        nonzero(op, "the key head width this statement states", k_dim)?;
        nonzero(op, "the value head width this statement states", v_dim)?;
        if !v_heads.is_multiple_of(k_heads) {
            return Err(refuse(
                op,
                format!(
                    "the {v_heads} value heads are not a whole number of the {k_heads} key heads"
                ),
            ));
        }
        head_width(op, k_dim, "the key head width")?;
        debug_assert_eq!(
            u64::from(qkv.width),
            2 * u64::from(k_heads) * u64::from(k_dim) + u64::from(v_heads) * u64::from(v_dim),
            "the post-convolution qkv's row is the four stated head numbers"
        );
        debug_assert!(
            gates.rows == qkv.rows && gates.width == 2 * v_heads,
            "the fused `[g_log | beta]` row is two entries per value head"
        );
        debug_assert!(
            y.rows == qkv.rows && u64::from(y.width) == u64::from(v_heads) * u64::from(v_dim),
            "the recurrence lands one value plane per row"
        );
        Ok(Self {
            k_heads,
            v_heads,
            k_dim,
            v_dim,
        })
    }
}

fn conv_history(op: &'static str, conv_width: u32, dilation: u32) -> Result<u32, Error> {
    nonzero(op, "the conv width this statement states", conv_width)?;
    nonzero(op, "the dilation this statement states", dilation)?;
    let hist = (conv_width - 1)
        .checked_mul(dilation)
        .and_then(|span| span.checked_add(1))
        .ok_or_else(|| {
            refuse(
                op,
                format!(
                    "a width of {conv_width} at dilation {dilation} keeps no countable history"
                ),
            )
        })?;
    if hist > CONV_HIST_MAX {
        return Err(refuse(
            op,
            format!("a history of {hist} rows is above the {CONV_HIST_MAX} a conv column keeps"),
        ));
    }
    Ok(hist)
}

fn conv_bank(
    op: &'static str,
    state: &RecurrentPool,
    hist: u32,
    channels: u32,
) -> Result<Tensor, Error> {
    if state.conv_state.buf != state.new_conv_state.buf {
        return Err(refuse(
            op,
            "this plane rolls the conv state in place, and the pool names two banks",
        ));
    }
    debug_assert_eq!(
        u64::from(state.conv_state.width),
        u64::from(hist) * u64::from(channels),
        "the state bank a dilated conv reads is `(width - 1) * dilation + 1` rows of channels"
    );
    Ok(state.new_conv_state)
}

pub fn causal_conv1d(
    ctx: &Ctx<'_>,
    x: Tensor,
    weight: Tensor,
    state: &RecurrentPool,
    conv_width: u32,
    dilation: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ssm_causal_conv1d";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "causal_conv1d_bf16" });
    let channels = nonzero(OP, "the conv's channel count", x.width)?;
    let rows = nonzero(OP, "rows", x.rows)?;
    debug_assert!(
        y.rows == x.rows && y.width == x.width,
        "the conv lands the row it convolves"
    );
    let taps = stated(OP, conv_width)?;
    let hist = conv_history(OP, conv_width, dilation)?;
    let bank = conv_bank(OP, state, hist, channels)?;
    ctx.fire(
        Fire::at("ssm/causal_conv1d.slang", entry).apply(conv_grid(channels, rows)),
        &[
            x.arg(),
            weight.arg(),
            bank.arg_mut(),
            state.slots.arg(),
            y.arg_mut(),
            stated(OP, channels)?.arg(),
            taps.arg(),
            stated(OP, dilation)?.arg(),
        ],
    )
}

pub fn causal_conv1d_chunked(
    ctx: &Ctx<'_>,
    x: RaggedTensor,
    weight: Tensor,
    state: &RecurrentPool,
    conv_width: u32,
    dilation: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ssm_causal_conv1d_chunked";
    let entry = dtype_dispatch!(OP, x.data.dtype, { Bf16 => "causal_conv1d_chunked_bf16" });
    let channels = nonzero(OP, "the conv's channel count", x.data.width)?;
    nonzero(OP, "rows", x.data.rows)?;
    debug_assert!(
        y.rows == x.data.rows && y.width == x.data.width,
        "the conv lands the row it convolves"
    );
    let taps = stated(OP, conv_width)?;
    let hist = conv_history(OP, conv_width, dilation)?;
    let bank = conv_bank(OP, state, hist, channels)?;
    let lanes = requests(OP, x)?;
    ctx.fire(
        Fire::at("ssm/causal_conv1d.slang", entry).apply(conv_grid(channels, lanes)),
        &[
            x.data.arg(),
            x.indptr.arg(),
            weight.arg(),
            bank.arg_mut(),
            state.slots.arg(),
            y.arg_mut(),
            stated(OP, channels)?.arg(),
            taps.arg(),
            stated(OP, dilation)?.arg(),
        ],
    )
}

pub fn gdn_prep(
    ctx: &Ctx<'_>,
    ba: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    gates: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ssm_gdn_prep";
    let entry = dtype_dispatch!(OP, ba.dtype, { Bf16 => "gdn_ba_gates_bf16" });
    debug_assert_eq!(a_log.dtype, Dtype::F32, "`{OP}` reads an f32 decay bank");
    debug_assert_eq!(gates.dtype, Dtype::F32, "`{OP}` lands an f32 decay row");
    if ba.width == 0 || !ba.width.is_multiple_of(2) {
        return Err(refuse(
            OP,
            format!(
                "the {}-wide `[b | a]` projection does not halve into value heads",
                ba.width
            ),
        ));
    }
    let v_heads = ba.width / 2;
    debug_assert!(
        gates.rows == ba.rows && gates.width == ba.width,
        "the fused `[g_log | beta]` row rides the projection it is derived from"
    );
    let rows = nonzero(OP, "rows", ba.rows)?;
    ctx.fire(
        Fire::at("ssm/gdn_prep.slang", entry)
            .apply(Grid::of([v_heads, rows, 1], [CONV_GROUP, 1, 1])),
        &[
            ba.arg(),
            a_log.arg(),
            dt_bias.arg(),
            gates.arg_mut(),
            stated(OP, v_heads)?.arg(),
        ],
    )
}

pub fn gated_delta(
    ctx: &Ctx<'_>,
    qkv: Tensor,
    z: Tensor,
    gates: Tensor,
    state: &RecurrentPool,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ssm_gated_delta";
    let entry = dtype_dispatch!(OP, qkv.dtype, { Bf16 => "gated_delta_bf16" });

    let entry = if k_dim == 128 && v_dim <= 128 {
        "gated_delta_r128_bf16"
    } else {
        entry
    };
    debug_assert_eq!(gates.dtype, Dtype::F32, "`{OP}` reads an f32 decay row");
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Delta::of(OP, qkv, gates, y, k_heads, v_heads, k_dim, v_dim)?;
    let rows = nonzero(OP, "rows", qkv.rows)?;
    ctx.fire(
        Fire::at("ssm/gated_delta.slang", entry).apply(recurrence_grid(shape.v_heads, rows)),
        &[
            qkv.arg(),
            gates.arg(),
            state.state.arg_mut(),
            state.slots.arg(),
            y.arg_mut(),
            stated(OP, shape.k_heads)?.arg(),
            stated(OP, shape.v_heads)?.arg(),
            stated(OP, shape.k_dim)?.arg(),
            stated(OP, shape.v_dim)?.arg(),
        ],
    )
}

pub fn gated_delta_chunked(
    ctx: &Ctx<'_>,
    qkv: RaggedTensor,
    z: Tensor,
    gates: Tensor,
    state: &RecurrentPool,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ssm_gated_delta_chunked";
    let entry = dtype_dispatch!(OP, qkv.data.dtype, { Bf16 => "gated_delta_chunked_bf16" });

    let entry = if k_dim == 128 && v_dim <= 128 {
        "gated_delta_chunked_r128_bf16"
    } else {
        entry
    };
    debug_assert_eq!(gates.dtype, Dtype::F32, "`{OP}` reads an f32 decay row");
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Delta::of(OP, qkv.data, gates, y, k_heads, v_heads, k_dim, v_dim)?;
    let lanes = requests(OP, qkv)?;
    ctx.fire(
        Fire::at("ssm/gated_delta.slang", entry).apply(recurrence_grid(shape.v_heads, lanes)),
        &[
            qkv.data.arg(),
            qkv.indptr.arg(),
            gates.arg(),
            state.state.arg_mut(),
            state.slots.arg(),
            y.arg_mut(),
            stated(OP, shape.k_heads)?.arg(),
            stated(OP, shape.v_heads)?.arg(),
            stated(OP, shape.k_dim)?.arg(),
            stated(OP, shape.v_dim)?.arg(),
        ],
    )
}

struct Kda {
    heads: u32,
    head_dim: u32,
}

impl Kda {
    fn of(
        op: &'static str,
        mixed: Tensor,
        f: Tensor,
        b: Tensor,
        y: Tensor,
        heads: u32,
        head_dim: u32,
    ) -> Result<Self, Error> {
        nonzero(op, "the KDA heads this statement states", heads)?;
        nonzero(op, "the KDA head width this statement states", head_dim)?;
        head_width(op, head_dim, "the KDA head width")?;
        let plane = u64::from(heads) * u64::from(head_dim);
        debug_assert_eq!(
            u64::from(mixed.width),
            3 * plane,
            "the post-convolution `[q | k | v]` row is three head planes"
        );
        debug_assert!(
            f.rows == mixed.rows && u64::from(f.width) == plane,
            "the forget projection's row is one head plane"
        );
        debug_assert!(
            b.rows == mixed.rows && b.width == heads,
            "the beta projection's row is one entry per head"
        );
        debug_assert!(
            y.rows == mixed.rows && u64::from(y.width) == plane,
            "the accumulator lands one head plane per row"
        );
        Ok(Self { heads, head_dim })
    }
}

pub fn kda_step(
    ctx: &Ctx<'_>,
    mixed: Tensor,
    f: Tensor,
    b: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    state: &RecurrentPool,
    heads: u32,
    head_dim: u32,
    norm_eps: f32,
    gate_floor: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ssm_kda_step";
    let entry = dtype_dispatch!(OP, mixed.dtype, { Bf16 => "kda_step_bf16" });
    debug_assert_eq!(dt_bias.dtype, Dtype::F32, "`{OP}` reads an f32 decay bias");
    debug_assert_eq!(a_log.dtype, Dtype::F32, "`{OP}` reads an f32 decay bank");
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Kda::of(OP, mixed, f, b, y, heads, head_dim)?;
    let rows = nonzero(OP, "rows", mixed.rows)?;
    ctx.fire(
        Fire::at(KDA, entry).apply(recurrence_grid(shape.heads, rows)),
        &[
            mixed.arg(),
            f.arg(),
            b.arg(),
            dt_bias.arg(),
            a_log.arg(),
            state.state.arg_mut(),
            state.slots.arg(),
            y.arg_mut(),
            stated(OP, shape.heads)?.arg(),
            stated(OP, shape.head_dim)?.arg(),
            norm_eps.arg(),
            gate_floor.arg(),
        ],
    )
}

pub fn kda_chunked(
    ctx: &Ctx<'_>,
    mixed: RaggedTensor,
    f: Tensor,
    b: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    state: &RecurrentPool,
    heads: u32,
    head_dim: u32,
    norm_eps: f32,
    gate_floor: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ssm_kda_chunked";
    let entry = dtype_dispatch!(OP, mixed.data.dtype, { Bf16 => "kda_chunked_bf16" });
    debug_assert_eq!(dt_bias.dtype, Dtype::F32, "`{OP}` reads an f32 decay bias");
    debug_assert_eq!(a_log.dtype, Dtype::F32, "`{OP}` reads an f32 decay bank");
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Kda::of(OP, mixed.data, f, b, y, heads, head_dim)?;
    let lanes = requests(OP, mixed)?;
    ctx.fire(
        Fire::at(KDA, entry).apply(recurrence_grid(shape.heads, lanes)),
        &[
            mixed.data.arg(),
            mixed.indptr.arg(),
            f.arg(),
            b.arg(),
            dt_bias.arg(),
            a_log.arg(),
            state.state.arg_mut(),
            state.slots.arg(),
            y.arg_mut(),
            stated(OP, shape.heads)?.arg(),
            stated(OP, shape.head_dim)?.arg(),
            norm_eps.arg(),
            gate_floor.arg(),
        ],
    )
}

pub fn kda_committed(
    ctx: &Ctx<'_>,
    mixed: Tensor,
    indptr: Tensor,
    committed: &Committed,
    f: Tensor,
    b: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    state: &RecurrentPool,
    work: Tensor,
    heads: u32,
    head_dim: u32,
    norm_eps: f32,
    gate_floor: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ssm_kda_committed";
    let entry = dtype_dispatch!(OP, mixed.dtype, { Bf16 => "kda_committed_bf16" });
    debug_assert_eq!(dt_bias.dtype, Dtype::F32, "`{OP}` reads an f32 decay bias");
    debug_assert_eq!(a_log.dtype, Dtype::F32, "`{OP}` reads an f32 decay bank");
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Kda::of(OP, mixed, f, b, y, heads, head_dim)?;
    let lanes = committed_lanes(OP, indptr)?;
    ctx.fire(
        Fire::at(KDA, entry).apply(recurrence_grid(shape.heads, lanes)),
        &[
            mixed.arg(),
            indptr.arg(),
            committed.replay.arg(),
            committed.commit.arg(),
            committed.slots.arg(),
            stated(OP, committed.lane0)?.arg(),
            f.arg(),
            b.arg(),
            dt_bias.arg(),
            a_log.arg(),
            state.state.arg_mut(),
            work.arg_mut(),
            y.arg_mut(),
            stated(OP, shape.heads)?.arg(),
            stated(OP, shape.head_dim)?.arg(),
            norm_eps.arg(),
            gate_floor.arg(),
        ],
    )
}

#[derive(Clone, Copy, Debug)]
pub struct Committed {
    pub replay: Tensor,

    pub commit: Tensor,

    pub slots: Tensor,

    pub lane0: u32,
}

fn committed_lanes(op: &'static str, indptr: Tensor) -> Result<u32, Error> {
    if indptr.dtype != Dtype::I32 {
        return Err(refuse(
            op,
            format!(
                "the window CSR's boundaries are {:?}, and this arm walks an i32 indptr",
                indptr.dtype
            ),
        ));
    }
    match indptr.rows.checked_sub(1) {
        Some(lanes) if lanes > 0 => Ok(lanes),
        _ => Err(refuse(
            op,
            "the window CSR this fire names spans no request",
        )),
    }
}

#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d_committed(
    ctx: &Ctx<'_>,
    x: Tensor,
    indptr: Tensor,
    committed: &Committed,
    weight: Tensor,
    state: &RecurrentPool,
    conv_width: u32,
    dilation: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ssm_causal_conv1d_committed";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "causal_conv1d_committed_bf16" });
    let channels = nonzero(OP, "the conv's channel count", x.width)?;
    nonzero(OP, "extended rows", x.rows)?;
    let taps = stated(OP, conv_width)?;
    let hist = conv_history(OP, conv_width, dilation)?;
    let bank = conv_bank(OP, state, hist, channels)?;
    let lanes = committed_lanes(OP, indptr)?;
    ctx.fire(
        Fire::at("ssm/causal_conv1d.slang", entry).apply(conv_grid(channels, lanes)),
        &[
            x.arg(),
            indptr.arg(),
            committed.replay.arg(),
            committed.commit.arg(),
            committed.slots.arg(),
            stated(OP, committed.lane0)?.arg(),
            weight.arg(),
            bank.arg_mut(),
            y.arg_mut(),
            stated(OP, channels)?.arg(),
            taps.arg(),
            stated(OP, dilation)?.arg(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn gated_delta_committed(
    ctx: &Ctx<'_>,
    qkv: Tensor,
    indptr: Tensor,
    committed: &Committed,
    gates: Tensor,
    state: &RecurrentPool,
    work: Tensor,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "attention.ssm_gated_delta_committed";
    let entry = dtype_dispatch!(OP, qkv.dtype, { Bf16 => "gated_delta_committed_bf16" });

    let entry = if k_dim == 128 && v_dim <= 128 {
        "gated_delta_committed_r128_bf16"
    } else {
        entry
    };
    let shape = Delta::of(OP, qkv, gates, y, k_heads, v_heads, k_dim, v_dim)?;
    let lanes = committed_lanes(OP, indptr)?;
    ctx.fire(
        Fire::at("ssm/gated_delta.slang", entry).apply(recurrence_grid(shape.v_heads, lanes)),
        &[
            qkv.arg(),
            indptr.arg(),
            committed.replay.arg(),
            committed.commit.arg(),
            committed.slots.arg(),
            stated(OP, committed.lane0)?.arg(),
            gates.arg(),
            state.state.arg_mut(),
            work.arg_mut(),
            y.arg_mut(),
            stated(OP, shape.k_heads)?.arg(),
            stated(OP, shape.v_heads)?.arg(),
            stated(OP, shape.k_dim)?.arg(),
            stated(OP, shape.v_dim)?.arg(),
        ],
    )
}
