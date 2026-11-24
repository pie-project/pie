use crate::error::Error;
use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
use crate::tensor::{RaggedTensor, RecurrentPool, Tensor};
use crate::tuning::GdnScan;

pub mod dyn_conv;

const CONV_GROUP: u32 = 256;

/// The rows one thread of a prompt's conv runs in turn. One thread a channel
/// running a whole 555-row prompt left most of the GPU idle.
const CONV_RUN_ROWS: u32 = 16;

/// The widest conv the four-channel prompt conv holds (`PIE_CONV_X4_TAPS`).
const CONV_X4_TAPS: u32 = 8;

const SCAN_WIDTH: u32 = 128;

const GDN_SCAN_FILE: &str = "ssm/gdn_scan.metal";

const GDN_SCAN_LANES: u32 = 32;

const GDN_SCAN_TG_ROWS: u32 = 4;

const SCAN_HEAD_MAX: u32 = 256;

fn conv_grid(channels: u32, rows: u32) -> Grid {
    Grid::of([channels, rows, 1], [channels.min(CONV_GROUP), 1, 1])
}

const fn recurrence_grid(heads: u32, rows: u32) -> Grid {
    Grid::of([SCAN_WIDTH, heads, rows], [SCAN_WIDTH, 1, 1])
}

macro_rules! gdn_scan_points {
    ($(($l:literal, $v:literal, $p:literal)),+ $(,)?) => {
        fn gdn_scan_point(lanes: u32, vrows: u32, per: u32) -> Option<(&'static str, &'static str)> {
            match (lanes, vrows, per) {
                $(($l, $v, $p) => Some((
                    concat!("gated_delta_scan_bfloat16_l_", $l, "_v_", $v, "_p_", $p),
                    concat!(
                        "PIE_STAMP_gdn_scan(\"gated_delta_scan_bfloat16_l_", $l, "_v_",
                        $v, "_p_", $p, "\", ", $l, ", ", $v, ", ", $p, ")"
                    ),
                )),)+
                _ => None,
            }
        }

        fn gdn_scan_step_point(lanes: u32, vrows: u32, per: u32) -> Option<(&'static str, &'static str)> {
            match (lanes, vrows, per) {
                $(($l, $v, $p) => Some((
                    concat!("gated_delta_scan_step_bfloat16_l_", $l, "_v_", $v, "_p_", $p),
                    concat!(
                        "PIE_STAMP_gdn_scan_step(\"gated_delta_scan_step_bfloat16_l_", $l, "_v_",
                        $v, "_p_", $p, "\", ", $l, ", ", $v, ", ", $p, ")"
                    ),
                )),)+
                _ => None,
            }
        }

        fn gdn_scan_committed_point(lanes: u32, vrows: u32, per: u32) -> Option<(&'static str, &'static str)> {
            match (lanes, vrows, per) {
                $(($l, $v, $p) => Some((
                    concat!("gated_delta_scan_committed_bfloat16_l_", $l, "_v_", $v, "_p_", $p),
                    concat!(
                        "PIE_STAMP_gdn_scan_committed(\"gated_delta_scan_committed_bfloat16_l_", $l, "_v_",
                        $v, "_p_", $p, "\", ", $l, ", ", $v, ", ", $p, ")"
                    ),
                )),)+
                _ => None,
            }
        }
    };
}

gdn_scan_points!(
    (8, 2, 16),
    (32, 1, 2),
    (32, 2, 2),
    (32, 4, 2),
    (32, 8, 2),
    (32, 1, 4),
    (32, 2, 4),
    (32, 4, 4),
    (32, 8, 4),
    (32, 1, 8),
    (32, 2, 8),
    (32, 4, 8),
    (32, 8, 8),
);

/// `(lanes a head, value rows a simdgroup)` of a scan.
fn lanes_rows(scan: GdnScan) -> (u32, u32) {
    match scan {
        GdnScan::Simd { rows } => (GDN_SCAN_LANES, rows),
        GdnScan::Staged { .. } => (8, 2),
    }
}

fn gdn_staged_launch_at(shape: &Delta, requests: u32, rows: u32, scan: GdnScan) -> Option<Grid> {
    let GdnScan::Staged { min_rows } = scan else {
        return None;
    };
    if shape.k_dim != 128 || shape.v_dim != 128 || requests == 0 || rows / requests < min_rows {
        return None;
    }
    Some(Grid::of(
        [8, 64, shape.v_heads.checked_mul(requests)?],
        [8, 16, 1],
    ))
}

fn gdn_scan_launch_at(
    shape: &Delta,
    requests: u32,
    (lanes, vrows): (u32, u32),
    point: fn(u32, u32, u32) -> Option<(&'static str, &'static str)>,
) -> Option<(&'static str, &'static str, Grid)> {
    if ![8, GDN_SCAN_LANES].contains(&lanes)
        || vrows == 0
        || !shape.v_dim.is_multiple_of(vrows)
        || !shape.k_dim.is_multiple_of(lanes)
    {
        return None;
    }
    let (entry, stamp) = point(lanes, vrows, shape.k_dim / lanes)?;
    let row_groups = shape.v_dim / vrows;
    let target_tg_rows = GDN_SCAN_LANES * GDN_SCAN_TG_ROWS / lanes;
    let tg_rows = if row_groups.is_multiple_of(target_tg_rows) {
        target_tg_rows
    } else {
        1
    };
    let z = shape.v_heads.checked_mul(requests)?;
    Some((
        entry,
        stamp,
        Grid::of([lanes, row_groups, z], [lanes, tg_rows, 1]),
    ))
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
    #[allow(clippy::too_many_arguments)]
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
            "the recurrence lands one head plane per row"
        );
        Ok(Self { heads, head_dim })
    }
}

fn conv_history(op: &'static str, conv_width: u32, dilation: u32) -> Result<u32, Error> {
    nonzero(op, "the conv width this statement states", conv_width)?;
    nonzero(op, "the dilation this statement states", dilation)?;
    (conv_width - 1)
        .checked_mul(dilation)
        .and_then(|span| span.checked_add(1))
        .ok_or_else(|| {
            refuse(
                op,
                format!(
                    "a width of {conv_width} at dilation {dilation} keeps no countable history"
                ),
            )
        })
}

#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d(
    ctx: &Ctx<'_>,
    x: Tensor,
    weight: Tensor,
    bias: Option<Tensor>,
    state: &RecurrentPool,
    conv_width: u32,
    dilation: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "ssm.causal_conv1d";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "causal_conv1d_bfloat16" });
    let channels = nonzero(OP, "the conv's channel count", x.width)?;
    let rows = nonzero(OP, "rows", x.rows)?;
    debug_assert!(
        y.rows == x.rows && y.width == x.width,
        "the conv lands the row it convolves"
    );
    let taps = stated(
        OP,
        nonzero(OP, "the conv width this statement states", conv_width)?,
    )?;
    let hist = conv_history(OP, conv_width, dilation)?;
    debug_assert_eq!(
        u64::from(state.conv_state.width),
        u64::from(hist) * u64::from(channels),
        "the state bank a dilated conv reads is `(width - 1) * dilation + 1` rows of channels"
    );
    ctx.fire(
        Fire::at("ssm/conv1d.metal", entry).apply(conv_grid(channels, rows)),
        &[
            x.arg(),
            weight.arg(),
            state.conv_state.arg(),
            state.new_conv_state.arg_mut(),
            state.slots.arg(),
            y.arg_mut(),
            stated(OP, x.width)?.arg(),
            taps.arg(),
            stated(OP, dilation)?.arg(),
            conv_bias(ctx, bias)?,
            i32::from(bias.is_some()).arg(),
            state.segments.arg(),
        ],
    )
}

/// A conv's per-channel bias (Mamba-2's), added before the activation.
fn conv_bias(ctx: &Ctx<'_>, bias: Option<Tensor>) -> Result<crate::encode::ArgValue, Error> {
    match bias {
        Some(b) => Ok(b.arg()),
        None => ctx.absent(),
    }
}

/// The rows of one lane whose recurrent state a prompt's scans land in slots
/// of their own as they pass them (`Lane::rs_snapshots`), at most.
pub const RS_SNAPSHOTS: u32 = 4;

/// Where a prompt's scans land the state as of rows inside it: per lane,
/// [`RS_SNAPSHOTS`] rows into the lane (ascending, 0 ends them) and then as
/// many slots; the scan's request `r` is lane `lane0 + r`.
#[derive(Clone, Copy, Debug)]
pub struct Snapshots {
    pub table: Tensor,
    pub lane0: u32,
}

fn snapshot_args(
    ctx: &Ctx<'_>,
    op: &'static str,
    snapshots: Option<Snapshots>,
) -> Result<[crate::encode::ArgValue; 2], Error> {
    Ok(match snapshots {
        Some(snapshots) => [snapshots.table.arg(), stated(op, snapshots.lane0)?.arg()],
        None => [ctx.absent()?, (-1i32).arg()],
    })
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
    causal_conv1d_chunked_landing(ctx, x, weight, None, state, conv_width, dilation, y, None)
}

/// [`causal_conv1d_chunked`], landing the window as of each of `snapshots`'
/// rows in its slot too.
#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d_chunked_landing(
    ctx: &Ctx<'_>,
    x: RaggedTensor,
    weight: Tensor,
    bias: Option<Tensor>,
    state: &RecurrentPool,
    conv_width: u32,
    dilation: u32,
    y: Tensor,
    snapshots: Option<Snapshots>,
) -> Result<(), Error> {
    const OP: &str = "ssm.causal_conv1d_chunked";
    let entry = dtype_dispatch!(OP, x.data.dtype, { Bf16 => "causal_conv1d_chunked_bfloat16" });
    let channels = nonzero(OP, "the conv's channel count", x.data.width)?;
    nonzero(OP, "rows", x.data.rows)?;
    debug_assert!(
        y.rows == x.data.rows && y.width == x.data.width,
        "the conv lands the row it convolves"
    );
    let taps = stated(
        OP,
        nonzero(OP, "the conv width this statement states", conv_width)?,
    )?;
    let hist = conv_history(OP, conv_width, dilation)?;
    debug_assert_eq!(
        u64::from(state.conv_state.width),
        u64::from(hist) * u64::from(channels),
        "the state bank a dilated conv reads is `(width - 1) * dilation + 1` rows of channels"
    );
    let lanes = requests(OP, x)?;
    let [snaps, lane0] = snapshot_args(ctx, OP, snapshots)?;
    // A lane holds at most every row, so this many runs cover the longest.
    let runs = x.data.rows.div_ceil(CONV_RUN_ROWS);
    // Four channels a thread where they divide the row and the taps fit.
    let (entry, threads) = if channels.is_multiple_of(4) && conv_width <= CONV_X4_TAPS {
        ("causal_conv1d_chunked_x4_bfloat16", channels / 4)
    } else {
        (entry, channels)
    };
    ctx.fire(
        Fire::at("ssm/conv1d.metal", entry).apply(Grid::of(
            [threads, lanes, runs],
            [threads.min(CONV_GROUP), 1, 1],
        )),
        &[
            x.data.arg(),
            x.indptr.arg(),
            weight.arg(),
            state.conv_state.arg(),
            state.new_conv_state.arg_mut(),
            state.slots.arg(),
            y.arg_mut(),
            stated(OP, x.data.width)?.arg(),
            taps.arg(),
            stated(OP, dilation)?.arg(),
            stated(OP, CONV_RUN_ROWS)?.arg(),
            snaps,
            lane0,
            conv_bias(ctx, bias)?,
            i32::from(bias.is_some()).arg(),
            state.segments.arg(),
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
    const OP: &str = "ssm.gdn_gates";
    let entry = dtype_dispatch!(OP, ba.dtype, { Bf16 => "gdn_prep_bfloat16" });
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
        Fire::at("ssm/gdn_prep.metal", entry)
            .apply(Grid::of([v_heads, rows, 1], [v_heads.min(256), 1, 1])),
        &[
            ba.arg(),
            a_log.arg(),
            dt_bias.arg(),
            gates.arg_mut(),
            stated(OP, v_heads)?.arg(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn gated_delta(
    ctx: &Ctx<'_>,
    tuning: &crate::Tuning,
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
    const OP: &str = "ssm.gated_delta";
    let _ = z;
    let entry = dtype_dispatch!(OP, qkv.dtype, { Bf16 => "gated_delta_bfloat16" });
    debug_assert_eq!(gates.dtype, Dtype::F32, "`{OP}` reads an f32 decay row");
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Delta::of(OP, qkv, gates, y, k_heads, v_heads, k_dim, v_dim)?;
    let rows = nonzero(OP, "rows", qkv.rows)?;
    if let Some((point, stamp, grid)) = gdn_scan_launch_at(
        &shape,
        rows,
        lanes_rows(tuning.gdn_scan()),
        gdn_scan_step_point,
    ) {
        return ctx.fire(
            Fire::at(GDN_SCAN_FILE, point).stamp(stamp).apply(grid),
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
                state.segments.arg(),
            ],
        );
    }
    let splits = (shape.v_dim / 32).max(1);
    let grid = Grid::of(
        [SCAN_WIDTH, shape.v_heads, rows.saturating_mul(splits)],
        [SCAN_WIDTH, 1, 1],
    );
    ctx.fire(
        Fire::at("ssm/gated_delta.metal", entry).apply(grid),
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
            stated(OP, splits)?.arg(),
            state.segments.arg(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn gated_delta_chunked(
    ctx: &Ctx<'_>,
    tuning: &crate::Tuning,
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
    gated_delta_chunked_landing(
        ctx, tuning, qkv, z, gates, state, k_heads, v_heads, k_dim, v_dim, y, None,
    )
}

/// [`gated_delta_chunked`], landing the state as of each of `snapshots`'
/// rows in its slot too.
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_chunked_landing(
    ctx: &Ctx<'_>,
    tuning: &crate::Tuning,
    qkv: RaggedTensor,
    z: Tensor,
    gates: Tensor,
    state: &RecurrentPool,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    y: Tensor,
    snapshots: Option<Snapshots>,
) -> Result<(), Error> {
    const OP: &str = "ssm.gated_delta_chunked";
    let _ = z;
    let entry = dtype_dispatch!(OP, qkv.data.dtype, { Bf16 => "gated_delta_chunked_bfloat16" });
    debug_assert_eq!(gates.dtype, Dtype::F32, "`{OP}` reads an f32 decay row");
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Delta::of(OP, qkv.data, gates, y, k_heads, v_heads, k_dim, v_dim)?;
    let lanes = requests(OP, qkv)?;
    let [snaps, lane0] = snapshot_args(ctx, OP, snapshots)?;
    let args = [
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
        snaps,
        lane0,
        state.segments.arg(),
    ];
    if let Some(grid) = gdn_staged_launch_at(&shape, lanes, qkv.data.rows, tuning.gdn_scan()) {
        return ctx.fire(
            Fire::at("ssm/gdn_staged.metal", "gated_delta_staged_bfloat16_b16")
                .stamp("PIE_GDN_STAGED(\"gated_delta_staged_bfloat16_b16\", 16)")
                .apply(grid),
            &args,
        );
    }
    if let Some((point, stamp, grid)) =
        gdn_scan_launch_at(&shape, lanes, lanes_rows(tuning.gdn_scan()), gdn_scan_point)
    {
        return ctx.fire(
            Fire::at(GDN_SCAN_FILE, point).stamp(stamp).apply(grid),
            &args,
        );
    }
    ctx.fire(
        Fire::at("ssm/gated_delta.metal", entry).apply(recurrence_grid(shape.v_heads, lanes)),
        &args,
    )
}

#[allow(clippy::too_many_arguments)]
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
    const OP: &str = "ssm.kda";
    let entry = dtype_dispatch!(OP, mixed.dtype, { Bf16 => "kda_step_bfloat16" });
    debug_assert_eq!(dt_bias.dtype, Dtype::F32, "`{OP}` reads an f32 decay bias");
    debug_assert_eq!(a_log.dtype, Dtype::F32, "`{OP}` reads an f32 decay bank");
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Kda::of(OP, mixed, f, b, y, heads, head_dim)?;
    let rows = nonzero(OP, "rows", mixed.rows)?;
    ctx.fire(
        Fire::at("ssm/kda.metal", entry).apply(recurrence_grid(shape.heads, rows)),
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
            state.segments.arg(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
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
    const OP: &str = "ssm.kda_chunked";
    let entry = dtype_dispatch!(OP, mixed.data.dtype, { Bf16 => "kda_chunked_bfloat16" });
    debug_assert_eq!(dt_bias.dtype, Dtype::F32, "`{OP}` reads an f32 decay bias");
    debug_assert_eq!(a_log.dtype, Dtype::F32, "`{OP}` reads an f32 decay bank");
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Kda::of(OP, mixed.data, f, b, y, heads, head_dim)?;
    let lanes = requests(OP, mixed)?;
    ctx.fire(
        Fire::at("ssm/kda.metal", entry).apply(recurrence_grid(shape.heads, lanes)),
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
            state.segments.arg(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
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
    const OP: &str = "ssm.kda_committed";
    let entry = dtype_dispatch!(OP, mixed.dtype, { Bf16 => "kda_committed_bfloat16" });
    debug_assert_eq!(dt_bias.dtype, Dtype::F32, "`{OP}` reads an f32 decay bias");
    debug_assert_eq!(a_log.dtype, Dtype::F32, "`{OP}` reads an f32 decay bank");
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Kda::of(OP, mixed, f, b, y, heads, head_dim)?;
    let lanes = committed_lanes(OP, indptr)?;
    ctx.fire(
        Fire::at("ssm/kda.metal", entry).apply(recurrence_grid(shape.heads, lanes)),
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
            state.segments.arg(),
        ],
    )
}

/// Mamba-2's per-head weights: `A_log`, `dt_bias` and `D` (f32 `[heads]`),
/// and the gated group norm's bf16 `[heads * 64]` weight and epsilon.
#[derive(Clone, Copy, Debug)]
pub struct Mamba2 {
    pub a_log: Tensor,
    pub dt_bias: Tensor,
    pub d: Tensor,
    pub norm: Tensor,
    pub eps: f32,
    pub heads: u32,
    pub groups: u32,
}

impl Mamba2 {
    /// Refuses all but 64-wide heads over a 128-wide state, as CUDA does.
    fn check(
        &self,
        op: &'static str,
        xbc: Tensor,
        dt: Tensor,
        z: Tensor,
        y: Tensor,
        state_dim: u32,
    ) -> Result<(), Error> {
        let inner = self.heads * 64;
        if state_dim != 128
            || self.groups == 0
            || !self.heads.is_multiple_of(self.groups)
            || xbc.width != inner + 2 * self.groups * state_dim
            || dt.width != self.heads
            || z.width != inner
            || y.width != inner
        {
            return Err(refuse(
                op,
                format!(
                    "Mamba-2 runs 64-wide heads over a 128-wide state in whole groups; this \
                     fire states {} heads in {} groups of a {}-wide state, a {}-wide xbc, \
                     {}-wide dt, a {}-wide z and a {}-wide output",
                    self.heads, self.groups, state_dim, xbc.width, dt.width, z.width, y.width
                ),
            ));
        }
        for (what, t) in [
            ("xbc", xbc.dtype),
            ("dt", dt.dtype),
            ("z", z.dtype),
            ("output", y.dtype),
        ] {
            if t != Dtype::Bf16 {
                return Err(refuse(
                    op,
                    format!("Mamba-2 reads and lands bf16, and its {what} is {t:?}"),
                ));
            }
        }
        Ok(())
    }

    /// The gated output's group norm, in place.
    fn norm(&self, ctx: &Ctx<'_>, y: Tensor) -> Result<(), Error> {
        crate::elemwise::norm::rmsnorm_grouped(
            ctx,
            y,
            self.norm,
            y.width / self.groups,
            self.eps,
            false,
            y,
        )
    }
}

/// One Mamba-2 step a row over the convolved `xbc` rows: lands `(C S + D x)
/// * silu(z)`, group-normed, in `y`.
#[allow(clippy::too_many_arguments)]
pub fn mamba2(
    ctx: &Ctx<'_>,
    xbc: Tensor,
    dt: Tensor,
    z: Tensor,
    m: Mamba2,
    state: &RecurrentPool,
    state_dim: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "ssm.mamba2";
    m.check(OP, xbc, dt, z, y, state_dim)?;
    let rows = nonzero(OP, "rows", xbc.rows)?;
    ctx.fire(
        Fire::at("ssm/mamba2.metal", "mamba2_step_bfloat16")
            .apply(Grid::of([256, m.heads, rows], [256, 1, 1])),
        &[
            xbc.arg(),
            dt.arg(),
            z.arg(),
            m.a_log.arg(),
            m.dt_bias.arg(),
            m.d.arg(),
            state.state.arg_mut(),
            state.slots.arg(),
            y.arg_mut(),
            stated(OP, m.heads)?.arg(),
            stated(OP, m.groups)?.arg(),
            state.segments.arg(),
        ],
    )?;
    m.norm(ctx, y)
}

/// A prompt's Mamba-2 walk a request, landing the state as of each of
/// `snapshots`' rows in its slot too; then the group norm.
#[allow(clippy::too_many_arguments)]
pub fn mamba2_chunked(
    ctx: &Ctx<'_>,
    xbc: RaggedTensor,
    dt: Tensor,
    z: Tensor,
    m: Mamba2,
    state: &RecurrentPool,
    state_dim: u32,
    y: Tensor,
    snapshots: Option<Snapshots>,
) -> Result<(), Error> {
    const OP: &str = "ssm.mamba2_chunked";
    m.check(OP, xbc.data, dt, z, y, state_dim)?;
    let lanes = requests(OP, xbc)?;
    let [snaps, lane0] = snapshot_args(ctx, OP, snapshots)?;
    ctx.fire(
        Fire::at("ssm/mamba2.metal", "mamba2_chunked_bfloat16")
            .apply(Grid::of([256, m.heads, lanes], [256, 1, 1])),
        &[
            xbc.data.arg(),
            xbc.indptr.arg(),
            dt.arg(),
            z.arg(),
            m.a_log.arg(),
            m.dt_bias.arg(),
            m.d.arg(),
            state.state.arg_mut(),
            state.slots.arg(),
            y.arg_mut(),
            stated(OP, m.heads)?.arg(),
            stated(OP, m.groups)?.arg(),
            snaps,
            lane0,
            state.segments.arg(),
        ],
    )?;
    m.norm(ctx, y)
}

#[cfg(test)]
mod tests {
    use super::*;

    const STAGED: GdnScan = GdnScan::Staged { min_rows: 32 };

    const D27B: Delta = Delta {
        k_heads: 16,
        v_heads: 48,
        k_dim: 128,
        v_dim: 128,
    };

    #[test]
    fn ssm_every_case() {
        the_three_scans_share_one_geometry();
        a_shape_the_stamp_does_not_name_falls_back();
        assert!(gdn_staged_launch_at(&D27B, 4, 127, STAGED).is_none());
        assert_eq!(
            gdn_staged_launch_at(&D27B, 4, 128, STAGED),
            Some(Grid::of([8, 64, 4 * 48], [8, 16, 1]))
        );
        assert!(gdn_staged_launch_at(&D27B, 4, 2048, GdnScan::Simd { rows: 4 }).is_none());
        assert!(gdn_staged_launch_at(&D27B, 0, 0, STAGED).is_none());
        assert!(gdn_staged_launch_at(&Delta { v_dim: 64, ..D27B }, 4, 2048, STAGED).is_none());
        assert!(gdn_staged_launch_at(&Delta { k_dim: 64, ..D27B }, 4, 2048, STAGED).is_none());
    }

    fn the_three_scans_share_one_geometry() {
        let plain = gdn_scan_launch_at(&D27B, 3, (32, 4), gdn_scan_point).expect("stamped");
        let step = gdn_scan_launch_at(&D27B, 3, (32, 4), gdn_scan_step_point).expect("stamped");
        let committed =
            gdn_scan_launch_at(&D27B, 3, (32, 4), gdn_scan_committed_point).expect("stamped");
        assert_eq!(plain.0, "gated_delta_scan_bfloat16_l_32_v_4_p_4");
        assert_eq!(step.0, "gated_delta_scan_step_bfloat16_l_32_v_4_p_4");
        assert_eq!(
            committed.0,
            "gated_delta_scan_committed_bfloat16_l_32_v_4_p_4"
        );
        assert!(committed.1.starts_with("PIE_STAMP_gdn_scan_committed("));
        assert_eq!(plain.2, step.2);
        assert_eq!(plain.2, committed.2);
        assert_eq!(plain.2, Grid::of([32, 32, 3 * 48], [32, 4, 1]));
        let packed = gdn_scan_launch_at(&D27B, 3, (8, 2), gdn_scan_point).expect("packed");
        let packed_step =
            gdn_scan_launch_at(&D27B, 3, (8, 2), gdn_scan_step_point).expect("packed");
        let packed_commit =
            gdn_scan_launch_at(&D27B, 3, (8, 2), gdn_scan_committed_point).expect("packed");
        assert_eq!(packed.0, "gated_delta_scan_bfloat16_l_8_v_2_p_16");
        assert_eq!(packed.2, Grid::of([8, 64, 3 * 48], [8, 16, 1]));
        assert_eq!(packed.2, packed_step.2);
        assert_eq!(packed.2, packed_commit.2);
        assert!(gdn_scan_launch_at(&D27B, 3, (8, 4), gdn_scan_point).is_none());
        assert!(gdn_scan_launch_at(&D27B, 3, (0, 2), gdn_scan_point).is_none());
    }

    fn a_shape_the_stamp_does_not_name_falls_back() {
        let odd = Delta { k_dim: 100, ..D27B };
        assert!(gdn_scan_launch_at(&odd, 1, (32, 4), gdn_scan_point).is_none());
        let deep = Delta { k_dim: 512, ..D27B };
        assert!(gdn_scan_launch_at(&deep, 1, (32, 4), gdn_scan_point).is_none());
        let ragged = Delta { v_dim: 66, ..D27B };
        assert!(gdn_scan_launch_at(&ragged, 1, (32, 4), gdn_scan_point).is_none());
    }
}

#[derive(Clone, Copy, Debug)]
pub struct Committed {
    pub replay: Tensor,
    pub commit: Tensor,
    pub slots: Tensor,
    pub lane0: u32,
    /// Each lane's carried rows among the extended rows: its replay, or
    /// none for a compact decode lane, which reads them from its buffer
    /// (the verify conv, the verify and carried scans).
    pub ext_replay: Tensor,
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
    causal_conv1d_committed_landing(
        ctx, x, indptr, committed, weight, state, conv_width, dilation, y, None,
    )
}

/// [`causal_conv1d_committed`], landing the window as of each of
/// `snapshots`' rows in its slot too.
#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d_committed_landing(
    ctx: &Ctx<'_>,
    x: Tensor,
    indptr: Tensor,
    committed: &Committed,
    weight: Tensor,
    state: &RecurrentPool,
    conv_width: u32,
    dilation: u32,
    y: Tensor,
    snapshots: Option<Snapshots>,
) -> Result<(), Error> {
    const OP: &str = "ssm.causal_conv1d_committed";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "causal_conv1d_committed_bfloat16" });
    let channels = nonzero(OP, "the conv's channel count", x.width)?;
    nonzero(OP, "extended rows", x.rows)?;
    let taps = stated(
        OP,
        nonzero(OP, "the conv width this statement states", conv_width)?,
    )?;
    let hist = conv_history(OP, conv_width, dilation)?;
    if hist > 64 {
        return Err(refuse(
            OP,
            format!("a history of {hist} rows is past the 64 this arm stages"),
        ));
    }
    let lanes = committed_lanes(OP, indptr)?;
    let [snaps, lane0] = snapshot_args(ctx, OP, snapshots)?;
    ctx.fire(
        Fire::at("ssm/conv1d.metal", entry).apply(conv_grid(channels, lanes)),
        &[
            x.arg(),
            indptr.arg(),
            committed.replay.arg(),
            committed.commit.arg(),
            committed.slots.arg(),
            stated(OP, committed.lane0)?.arg(),
            weight.arg(),
            state.conv_state.arg(),
            state.new_conv_state.arg_mut(),
            y.arg_mut(),
            stated(OP, x.width)?.arg(),
            taps.arg(),
            stated(OP, dilation)?.arg(),
            snaps,
            lane0,
            state.segments.arg(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn gated_delta_committed(
    ctx: &Ctx<'_>,
    tuning: &crate::Tuning,
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
    gated_delta_committed_landing(
        ctx, tuning, qkv, indptr, committed, gates, state, work, k_heads, v_heads, k_dim, v_dim, y,
        None,
    )
}

/// [`gated_delta_committed`], landing the state as of each of `snapshots`'
/// rows in its slot too.
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_committed_landing(
    ctx: &Ctx<'_>,
    tuning: &crate::Tuning,
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
    snapshots: Option<Snapshots>,
) -> Result<(), Error> {
    const OP: &str = "ssm.gated_delta_committed";
    let entry = dtype_dispatch!(OP, qkv.dtype, { Bf16 => "gated_delta_committed_bfloat16" });
    debug_assert_eq!(gates.dtype, Dtype::F32, "`{OP}` reads an f32 decay row");
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Delta::of(OP, qkv, gates, y, k_heads, v_heads, k_dim, v_dim)?;
    let lanes = committed_lanes(OP, indptr)?;
    if let Some((point, stamp, grid)) = gdn_scan_launch_at(
        &shape,
        lanes,
        lanes_rows(tuning.gdn_scan()),
        gdn_scan_committed_point,
    ) {
        let [snaps, lane0] = snapshot_args(ctx, OP, snapshots)?;
        return ctx.fire(
            Fire::at(GDN_SCAN_FILE, point).stamp(stamp).apply(grid),
            &[
                qkv.arg(),
                indptr.arg(),
                committed.replay.arg(),
                committed.commit.arg(),
                committed.slots.arg(),
                stated(OP, committed.lane0)?.arg(),
                gates.arg(),
                state.state.arg_mut(),
                y.arg_mut(),
                stated(OP, shape.k_heads)?.arg(),
                stated(OP, shape.v_heads)?.arg(),
                stated(OP, shape.k_dim)?.arg(),
                stated(OP, shape.v_dim)?.arg(),
                snaps,
                lane0,
                state.segments.arg(),
            ],
        );
    }
    let [snaps, lane0] = snapshot_args(ctx, OP, snapshots)?;
    let splits = (shape.v_dim / 32).max(1);
    let grid = Grid::of(
        [SCAN_WIDTH, shape.v_heads, lanes.saturating_mul(splits)],
        [SCAN_WIDTH, 1, 1],
    );
    ctx.fire(
        Fire::at("ssm/gated_delta.metal", entry).apply(grid),
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
            stated(OP, splits)?.arg(),
            snaps,
            lane0,
            state.segments.arg(),
        ],
    )
}

/// A recurrent buffer's plane as a verify arm reads it through the fire's
/// page table: `slab` from the plane's first row in the buffer's first
/// slot, and each lane's row of `pages` the in-page token of its first
/// extended row, then the slots of the pages its rows touch.
#[derive(Clone, Copy, Debug)]
pub struct Paged {
    pub slab: Tensor,
    pub pages: Tensor,
    pub page_tokens: u32,
    pub page_bytes: u32,
    pub row_bytes: u32,
    /// The ints of a lane's row of `pages`, the load's width (nonzero).
    pub page_ints: u32,
}

/// The fewest lanes a short carry takes the prepared form at.
const GDN_PREP_MIN_LANES: u32 = 4;

/// The floats a lane's short-carry record takes (`GdnShortRecord` in
/// `ssm/gdn_scan.metal`): the own row's normed q and k a key head, each of
/// up to [`GDN_CARRIED_INLINE`] carried rows' normed keys a key head and
/// corrected values a value head, their decays, the own row's scalars, and
/// the lane's facts.
fn short_record_floats(k_heads: u32, v_heads: u32) -> u64 {
    let (k, v, r, d) = (
        u64::from(k_heads),
        u64::from(v_heads),
        u64::from(GDN_CARRIED_INLINE),
        u64::from(GDN_VERIFY_DIM),
    );
    (2 + r) * k * d + r * v * d + r * v + 4 * v + v * d + 8
}

/// The pipelines [`gated_delta_carried`] dispatches whatever its lanes
/// carry: both forms (16 value rows a threadgroup, a head's 128) and the
/// per-head table's prepass. A load that buffers recurrent rows builds them
/// before ready, a beam's long carry being one its warm walk does not reach.
#[must_use]
pub fn carried_pipelines() -> [Fire; 10] {
    [
        Fire::at(GDN_SCAN_FILE, "gated_delta_carried_short_bfloat16_s_8_v_4"),
        Fire::at(GDN_SCAN_FILE, "gated_delta_carried_prep_bfloat16"),
        Fire::at(GDN_SCAN_FILE, "gated_delta_carried_prepped_bfloat16_s_8_v_4"),
        Fire::at(GDN_SCAN_FILE, "gated_delta_carried_land_bfloat16"),
        Fire::at("ssm/conv1d.metal", "causal_conv1d_carried_bfloat16"),
        Fire::at(GDN_SCAN_FILE, "gated_delta_group_fold_bfloat16"),
        Fire::at("ssm/conv1d.metal", "causal_conv1d_group_fold_bfloat16"),
        Fire::at(GDN_SCAN_FILE, "gated_delta_carried_bfloat16_s_8_v_2"),
        Fire::at(GDN_SCAN_FILE, "gated_delta_carried_bfloat16_s_8_v_16"),
        Fire::at(GDN_SCAN_FILE, "gated_delta_carried_dots_bfloat16"),
    ]
}

/// The carried rows past which [`gated_delta_carried`] runs a head's value
/// rows a threadgroup (each row's work read once a head) rather than 16.
pub const GDN_CARRIED_WIDE: u32 = 32;

/// The most carried rows [`gated_delta_carried`] reduces in the scan itself
/// (a chunk's, `kVerifyChunk`), with no per-head table launched.
pub const GDN_CARRIED_INLINE: u32 = 16;

/// The per-head table [`gated_delta_carried`] reads each carried row's
/// reductions from (`gated_delta_carried_dots`): `rows` entries a (lane, key
/// head), at least the window's most carried rows (`max_rep`), in `table`.
#[derive(Clone, Copy, Debug)]
pub struct DotsTable {
    pub table: Tensor,
    pub rows: u32,
    pub max_rep: u32,
}

impl DotsTable {
    /// Whether every lane's carried rows fit the table: else the scan makes
    /// each chunk's reductions itself, whatever a lane carries, and no table
    /// is launched (a carry is served up to its context, never refused).
    #[must_use]
    pub fn fits(&self) -> bool {
        self.max_rep <= self.rows
    }
}

/// A buffer's corrected planes (`rs_corrected`), as a verify or carried
/// scan lands and reads them through the fire's page table: each row's raw
/// keys (bf16, `k_heads * k_dim`) and corrected value `u` (f32, `v_heads *
/// v_dim`), each from the plane's first row in the buffer's first slot.
#[derive(Clone, Copy, Debug)]
pub struct Corrected {
    pub keys: Tensor,
    pub key_row_bytes: u32,
    pub u: Tensor,
    pub u_row_bytes: u32,
    /// The rows' per-head scalars (`v_heads` log-gates, then `v_heads` key
    /// norm inverses, f32): landed with the keys and `u`, read by the
    /// carried scan in place of recomputing them.
    pub scalars: Tensor,
    pub scalars_row_bytes: u32,
}

fn corrected_args(
    ctx: &Ctx<'_>,
    corrected: Option<Corrected>,
) -> Result<[crate::encode::ArgValue; 7], Error> {
    Ok(match corrected {
        Some(c) => [
            c.keys.arg_mut(),
            c.key_row_bytes.arg(),
            c.u.arg_mut(),
            c.u_row_bytes.arg(),
            1i32.arg(),
            c.scalars.arg_mut(),
            c.scalars_row_bytes.arg(),
        ],
        None => [
            ctx.absent()?,
            0u32.arg(),
            ctx.absent()?,
            0u32.arg(),
            0i32.arg(),
            ctx.absent()?,
            0u32.arg(),
        ],
    })
}

fn paged_args(ctx: &Ctx<'_>, paged: Option<Paged>) -> Result<[crate::encode::ArgValue; 6], Error> {
    Ok(match paged {
        Some(paged) => [
            paged.slab.arg_mut(),
            paged.pages.arg(),
            paged.page_tokens.arg(),
            paged.page_bytes.arg(),
            paged.row_bytes.arg(),
            i32::try_from(paged.page_ints.max(1)).unwrap_or(i32::MAX).arg(),
        ],
        None => [
            ctx.absent()?,
            ctx.absent()?,
            0u32.arg(),
            0u32.arg(),
            0u32.arg(),
            0i32.arg(),
        ],
    })
}

/// The `(simdgroups, value rows a simdgroup)` of a verify scan's
/// threadgroup. On the M4 Pro a verify round's layer (5 replayed rows and 8
/// of its own) walks in 36.6 us at (8, 2), 40.7 us at (4, 4) and 54.8 us
/// in the gate prep and the committed scan, against a floor of 25 us for
/// one row: the bank's read and write.
const GDN_VERIFY_GROUP: (u32, u32) = (8, 2);

/// The key and value head width the verify scan stages.
const GDN_VERIFY_DIM: u32 = 128;

/// The verify scan stamped at `(simdgroups, value rows)`.
#[must_use]
pub fn gated_delta_verify_entry(groups: u32, vrows: u32) -> Option<&'static str> {
    Some(match (groups, vrows) {
        (8, 2) => "gated_delta_verify_bfloat16_l_32_s_8_v_2",
        (4, 4) => "gated_delta_verify_bfloat16_l_32_s_4_v_4",
        (8, 4) => "gated_delta_verify_bfloat16_l_32_s_8_v_4",
        _ => return None,
    })
}

/// The launch of a verify scan at `(simdgroups, value rows)` over `heads`
/// value heads of every lane.
#[must_use]
pub fn gated_delta_verify_grid(groups: u32, vrows: u32, heads: u32) -> Grid {
    Grid::of(
        [GDN_SCAN_LANES, GDN_VERIFY_DIM / vrows, heads],
        [GDN_SCAN_LANES, groups, 1],
    )
}

/// Whether [`gated_delta_verify`] walks these heads.
#[must_use]
pub fn gated_delta_verify_fits(k_dim: u32, v_dim: u32) -> bool {
    k_dim == GDN_VERIFY_DIM && v_dim == GDN_VERIFY_DIM
}

/// The `[b | a]` projection a verify scan derives its gates from, as
/// [`gdn_prep`] does: the extended rows, and the window's own rows' gates
/// it lands.
#[derive(Clone, Copy, Debug)]
pub struct VerifyGates {
    pub ba: Tensor,
    pub dt_bias: Tensor,
    pub a_log: Tensor,
    pub gates: Tensor,
}

/// [`gated_delta_committed_landing`] over 128-wide heads, a threadgroup's
/// rows staged before it walks them, the gates [`gdn_prep`] derives
/// derived as they stage. A replayed row only folds in; a row of the
/// window's own lands its readout in `y` and its gates in `gates.gates`,
/// the window's own rows, where the committed scan landed every row in the
/// extended plane. Through `paged`, `gates.ba` is the window's own rows:
/// the replayed rows' are read from the lane's buffer, and the window's own
/// land there.
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_verify(
    ctx: &Ctx<'_>,
    qkv: Tensor,
    indptr: Tensor,
    committed: &Committed,
    gates: &VerifyGates,
    paged: Option<Paged>,
    corrected: Option<Corrected>,
    state: &RecurrentPool,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    y: Tensor,
    snapshots: Option<Snapshots>,
) -> Result<(), Error> {
    gated_delta_verify_at(
        ctx,
        GDN_VERIFY_GROUP,
        qkv,
        indptr,
        committed,
        gates,
        paged,
        corrected,
        state,
        k_heads,
        v_heads,
        k_dim,
        v_dim,
        y,
        snapshots,
    )
}

/// [`gated_delta_verify`] at a stated `(lanes, simdgroups, value rows)`.
#[allow(clippy::too_many_arguments)]
fn gated_delta_verify_at(
    ctx: &Ctx<'_>,
    (groups, vrows): (u32, u32),
    qkv: Tensor,
    indptr: Tensor,
    committed: &Committed,
    gates: &VerifyGates,
    paged: Option<Paged>,
    corrected: Option<Corrected>,
    state: &RecurrentPool,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    y: Tensor,
    snapshots: Option<Snapshots>,
) -> Result<(), Error> {
    const OP: &str = "ssm.gated_delta_committed";
    let _ = dtype_dispatch!(OP, qkv.dtype, { Bf16 => "gated_delta_verify_bfloat16" });
    debug_assert_eq!(
        gates.ba.dtype, qkv.dtype,
        "`{OP}` reads a `[b | a]` row as its qkv"
    );
    debug_assert_eq!(
        gates.a_log.dtype,
        Dtype::F32,
        "`{OP}` reads an f32 decay bank"
    );
    debug_assert_eq!(
        gates.gates.dtype,
        Dtype::F32,
        "`{OP}` lands an f32 decay row"
    );
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    nonzero(OP, "the key heads this statement states", k_heads)?;
    nonzero(OP, "the value heads this statement states", v_heads)?;
    if !v_heads.is_multiple_of(k_heads) {
        return Err(refuse(
            OP,
            format!("the {v_heads} value heads are not a whole number of the {k_heads} key heads"),
        ));
    }
    if gates.ba.width != 2 * v_heads {
        return Err(refuse(
            OP,
            format!(
                "the {}-wide `[b | a]` projection is not two entries of each of the {v_heads} value heads",
                gates.ba.width
            ),
        ));
    }
    if !gated_delta_verify_fits(k_dim, v_dim) {
        return Err(refuse(
            OP,
            format!(
                "the verify scan stages {GDN_VERIFY_DIM}-wide heads, and these are {k_dim} by {v_dim}"
            ),
        ));
    }
    let entry = gated_delta_verify_entry(groups, vrows)
        .ok_or_else(|| refuse(OP, "no verify scan is stamped at this geometry"))?;
    let lanes = committed_lanes(OP, indptr)?;
    let [snaps, lane0] = snapshot_args(ctx, OP, snapshots)?;
    let [slab, pages, page_tokens, page_bytes, row_bytes, is_paged] = paged_args(ctx, paged)?;
    let [key_slab, key_row_bytes, u_slab, u_row_bytes, is_corrected, scalar_slab, scalar_row_bytes] =
        corrected_args(ctx, corrected)?;
    let z = v_heads
        .checked_mul(lanes)
        .ok_or_else(|| refuse(OP, "the value heads of every lane overflow a launch"))?;
    ctx.fire(
        Fire::at(GDN_SCAN_FILE, entry).apply(gated_delta_verify_grid(groups, vrows, z)),
        &[
            qkv.arg(),
            indptr.arg(),
            committed.replay.arg(),
            committed.commit.arg(),
            committed.slots.arg(),
            stated(OP, committed.lane0)?.arg(),
            gates.ba.arg(),
            gates.a_log.arg(),
            gates.dt_bias.arg(),
            state.state.arg_mut(),
            y.arg_mut(),
            gates.gates.arg_mut(),
            stated(OP, k_heads)?.arg(),
            stated(OP, v_heads)?.arg(),
            snaps,
            lane0,
            slab,
            pages,
            page_tokens,
            page_bytes,
            row_bytes,
            is_paged,
            state.segments.arg(),
            key_slab,
            key_row_bytes,
            u_slab,
            u_row_bytes,
            is_corrected,
            committed.ext_replay.arg(),
            scalar_slab,
            scalar_row_bytes,
        ],
    )
}

/// A decode step over each lane's buffered rows read as a linear attention
/// (`gated_delta_carried`, new-memory/designs/lagged-fold.md): every lane a
/// row of its own and any number of carried rows (their gates and weights a
/// chunk at a time), read through
/// the fire's page table from the buffer's `[b | a]` plane (`ba_paged`) and
/// corrected planes (`corrected`), never re-stepped. Lands the own rows'
/// readout in `y`, gates in `gates.gates`, and their keys, `u` and `[b | a]`
/// in the buffer; folds the committed rows into the state (the verify
/// scan's state, layout and threadgroup).
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_carried(
    ctx: &Ctx<'_>,
    qkv: Tensor,
    indptr: Tensor,
    committed: &Committed,
    gates: &VerifyGates,
    ba_paged: Paged,
    corrected: Corrected,
    state: &RecurrentPool,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    y: Tensor,
    table: DotsTable,
) -> Result<(), Error> {
    const OP: &str = "ssm.gated_delta_carried";
    let _ = dtype_dispatch!(OP, qkv.dtype, { Bf16 => "gated_delta_carried_bfloat16" });
    nonzero(OP, "the key heads this statement states", k_heads)?;
    nonzero(OP, "the value heads this statement states", v_heads)?;
    if !v_heads.is_multiple_of(k_heads) {
        return Err(refuse(
            OP,
            format!("the {v_heads} value heads are not a whole number of the {k_heads} key heads"),
        ));
    }
    if !gated_delta_verify_fits(k_dim, v_dim) {
        return Err(refuse(
            OP,
            format!("the carried step reads {GDN_VERIFY_DIM}-wide heads, and these are {k_dim} by {v_dim}"),
        ));
    }
    // A short carry (a decode at its run-ahead lag) runs 16 value rows a
    // threadgroup; a long one (a beam in a parents pass carries every row
    // since its fold) a head's 128 a threadgroup, so each carried row's
    // gates, weights and values are read once a head, not once a group of
    // rows. On Qwen3.5-0.8B beam-4 over 256 steps the wide form takes the
    // step's growth from 58 to 24 us a step.
    let (groups, vrows, name): (u32, u32, &'static str) = if table.max_rep >= GDN_CARRIED_WIDE {
        (8, 16, "gated_delta_carried_bfloat16_s_8_v_16")
    } else {
        (8, 2, "gated_delta_carried_bfloat16_s_8_v_2")
    };
    let lanes = committed_lanes(OP, indptr)?;
    let z = v_heads
        .checked_mul(lanes)
        .ok_or_else(|| refuse(OP, "the value heads of every lane overflow a launch"))?;
    // A carry of one chunk (a decode at its run-ahead lag): the fused
    // step's tiling, each row's scalars once a threadgroup
    // (new-memory/designs/carried-kernel.md §1), a head's value tiles over
    // as many threadgroups as the lanes and heads leave room for.
    if table.max_rep <= GDN_CARRIED_INLINE {
        // Four value rows a simdgroup, as the scan step's, eight simdgroups
        // a threadgroup: a head over four threadgroups. On the M1 at c1 / c4
        // it beat 4 x 4, 8 x 2, 8 x 8, 8 x 16 and 2 x 16.
        let (sg, vrows) = (8, 4);
        // A record a lane in the seat's work area, when it holds one: the
        // prep makes each lane and value head's inputs once (the own row's
        // normed q and k, its gates, its landing, the carried rows dense),
        // which every simdgroup of the short scan otherwise makes for itself.
        let record = short_record_floats(k_heads, v_heads);
        let holds = u64::from(k_heads) * u64::from(table.rows) * 4 >= record;
        // Its two more launches a layer are a fixed cost a fire of few
        // lanes does not win back: below [`GDN_PREP_MIN_LANES`] the short
        // scan makes its own (M1 Max, Qwen3.5-0.8B: prep -9.2% against the
        // short scan's -4.5% at one lane, ahead from eight).
        if holds && lanes >= GDN_PREP_MIN_LANES {
            let args = |table_arg| -> Result<Vec<_>, Error> {
                Ok(vec![
                    qkv.arg(),
                    indptr.arg(),
                    committed.replay.arg(),
                    committed.commit.arg(),
                    committed.slots.arg(),
                    stated(OP, committed.lane0)?.arg(),
                    gates.ba.arg(),
                    gates.a_log.arg(),
                    gates.dt_bias.arg(),
                    state.state.arg_mut(),
                    y.arg_mut(),
                    gates.gates.arg_mut(),
                    stated(OP, k_heads)?.arg(),
                    stated(OP, v_heads)?.arg(),
                    ba_paged.slab.arg_mut(),
                    ba_paged.pages.arg(),
                    ba_paged.page_tokens.arg(),
                    ba_paged.page_bytes.arg(),
                    ba_paged.row_bytes.arg(),
                    corrected.keys.arg_mut(),
                    corrected.key_row_bytes.arg(),
                    corrected.u.arg_mut(),
                    corrected.u_row_bytes.arg(),
                    state.segments.arg(),
                    ba_paged.page_ints.arg(),
                    table_arg,
                    table.rows.arg(),
                    committed.ext_replay.arg(),
                    corrected.scalars.arg_mut(),
                    corrected.scalars_row_bytes.arg(),
                ])
            };
            ctx.fire(
                Fire::at(GDN_SCAN_FILE, "gated_delta_carried_prep_bfloat16")
                    .apply(Grid::of([GDN_SCAN_LANES, 1, z], [GDN_SCAN_LANES, 1, 1])),
                &args(table.table.arg_mut())?,
            )?;
            ctx.fire(
                Fire::at(GDN_SCAN_FILE, "gated_delta_carried_prepped_bfloat16_s_8_v_4")
                    .apply(gated_delta_verify_grid(sg, vrows, z)),
                &args(table.table.arg_mut())?,
            )?;
            return ctx.fire(
                Fire::at(GDN_SCAN_FILE, "gated_delta_carried_land_bfloat16")
                    .apply(Grid::of([GDN_SCAN_LANES, 1, z], [GDN_SCAN_LANES, 1, 1])),
                &args(table.table.arg())?,
            );
        }
        let name = "gated_delta_carried_short_bfloat16_s_8_v_4";
        return ctx.fire(
            Fire::at(GDN_SCAN_FILE, name).apply(gated_delta_verify_grid(sg, vrows, z)),
            &[
                qkv.arg(),
                indptr.arg(),
                committed.replay.arg(),
                committed.commit.arg(),
                committed.slots.arg(),
                stated(OP, committed.lane0)?.arg(),
                gates.ba.arg(),
                gates.a_log.arg(),
                gates.dt_bias.arg(),
                state.state.arg_mut(),
                y.arg_mut(),
                gates.gates.arg_mut(),
                stated(OP, k_heads)?.arg(),
                stated(OP, v_heads)?.arg(),
                ba_paged.slab.arg_mut(),
                ba_paged.pages.arg(),
                ba_paged.page_tokens.arg(),
                ba_paged.page_bytes.arg(),
                ba_paged.row_bytes.arg(),
                corrected.keys.arg_mut(),
                corrected.key_row_bytes.arg(),
                corrected.u.arg_mut(),
                corrected.u_row_bytes.arg(),
                state.segments.arg(),
                ba_paged.page_ints.arg(),
                table.table.arg(),
                0u32.arg(),
                committed.ext_replay.arg(),
                corrected.scalars.arg_mut(),
                corrected.scalars_row_bytes.arg(),
            ],
        );
    }
    // Each carried row's reductions, once a (lane, key head): the table
    // every value head and row group of the scan reads. A carry of one
    // chunk (a decode at its lag) makes them in the scan itself, a
    // simdgroup a row, and launches no table (its rows stated 0).
    let short = table.max_rep <= GDN_CARRIED_INLINE || !table.fits();
    let table_rows = if short { 0 } else { table.rows };
    if table.max_rep > 0 && !short {
        let zk = k_heads
            .checked_mul(lanes)
            .ok_or_else(|| refuse(OP, "the key heads of every lane overflow a launch"))?;
        ctx.fire(
            Fire::at(GDN_SCAN_FILE, "gated_delta_carried_dots_bfloat16").apply(Grid::of(
                [GDN_SCAN_LANES, table.max_rep, zk],
                [GDN_SCAN_LANES, 1, 1],
            )),
            &[
                qkv.arg(),
                indptr.arg(),
                committed.replay.arg(),
                stated(OP, committed.lane0)?.arg(),
                corrected.keys.arg_mut(),
                ba_paged.pages.arg(),
                ba_paged.page_tokens.arg(),
                ba_paged.page_bytes.arg(),
                corrected.key_row_bytes.arg(),
                ba_paged.page_ints.arg(),
                stated(OP, k_heads)?.arg(),
                stated(OP, v_heads)?.arg(),
                table.table.arg_mut(),
                table.rows.arg(),
                committed.ext_replay.arg(),
            ],
        )?;
    }
    ctx.fire(
        Fire::at(GDN_SCAN_FILE, name)
            .apply(gated_delta_verify_grid(groups, vrows, z)),
        &[
            qkv.arg(),
            indptr.arg(),
            committed.replay.arg(),
            committed.commit.arg(),
            committed.slots.arg(),
            stated(OP, committed.lane0)?.arg(),
            gates.ba.arg(),
            gates.a_log.arg(),
            gates.dt_bias.arg(),
            state.state.arg_mut(),
            y.arg_mut(),
            gates.gates.arg_mut(),
            stated(OP, k_heads)?.arg(),
            stated(OP, v_heads)?.arg(),
            ba_paged.slab.arg_mut(),
            ba_paged.pages.arg(),
            ba_paged.page_tokens.arg(),
            ba_paged.page_bytes.arg(),
            ba_paged.row_bytes.arg(),
            corrected.keys.arg_mut(),
            corrected.key_row_bytes.arg(),
            corrected.u.arg_mut(),
            corrected.u_row_bytes.arg(),
            state.segments.arg(),
            ba_paged.page_ints.arg(),
            table.table.arg(),
            table_rows.arg(),
            committed.ext_replay.arg(),
            corrected.scalars.arg_mut(),
            corrected.scalars_row_bytes.arg(),
        ],
    )
}

/// A fire's beam groups' folds (BEAM-PAGE-LISTS §11.3): `[lane, n]` a group,
/// each folding the `n` rows every live beam shares into its set's state
/// from its first fire lane `lane`'s buffer before the layer's step (that
/// lane's page table row starts `n` rows past them). The folding groups lead
/// the table and the first entry of no rows ends it: the kernels fold each
/// in turn, so a fire with none launches nothing. `count` is how many fold.
#[derive(Clone, Copy, Debug)]
pub struct GroupFolds {
    pub table: Tensor,
    pub count: u32,
}

/// The value rows a threadgroup of [`gated_delta_group_fold`] folds.
const GDN_GROUP_FOLD_VROWS: u32 = 8;

/// Each beam group's gated-delta fold before the layer's step
/// (`gated_delta_group_fold`): the carried scan's commit of the rows every
/// live beam shares, read through the group's first lane's page table row,
/// for the groups whose lane lies in this window (`lane0`, `lanes`).
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_group_fold(
    ctx: &Ctx<'_>,
    folds: GroupFolds,
    (lane0, lanes): (u32, u32),
    slots: Tensor,
    (a_log, dt_bias): (Tensor, Tensor),
    ba_paged: Paged,
    corrected: Corrected,
    state: &RecurrentPool,
    (k_heads, v_heads, k_dim, v_dim): (u32, u32, u32, u32),
) -> Result<(), Error> {
    const OP: &str = "ssm.gated_delta_group_fold";
    if folds.count == 0 {
        return Ok(());
    }
    nonzero(OP, "the key heads this statement states", k_heads)?;
    nonzero(OP, "the value heads this statement states", v_heads)?;
    if !v_heads.is_multiple_of(k_heads) || !gated_delta_verify_fits(k_dim, v_dim) {
        return Err(refuse(
            OP,
            format!(
                "the group fold reads {GDN_VERIFY_DIM}-wide heads, the value heads a whole \
                 number of the key heads; these are {k_heads} key and {v_heads} value heads, \
                 {k_dim} by {v_dim}"
            ),
        ));
    }
    ctx.fire(
        Fire::at(GDN_SCAN_FILE, "gated_delta_group_fold_bfloat16").apply(Grid::of(
            [GDN_SCAN_LANES, v_dim / GDN_GROUP_FOLD_VROWS, v_heads],
            [GDN_SCAN_LANES, 1, 1],
        )),
        &[
            folds.table.arg(),
            stated(OP, lane0)?.arg(),
            stated(OP, lanes)?.arg(),
            slots.arg(),
            a_log.arg(),
            dt_bias.arg(),
            state.state.arg_mut(),
            stated(OP, k_heads)?.arg(),
            stated(OP, v_heads)?.arg(),
            ba_paged.slab.arg_mut(),
            ba_paged.pages.arg(),
            ba_paged.page_tokens.arg(),
            ba_paged.page_bytes.arg(),
            ba_paged.row_bytes.arg(),
            corrected.keys.arg_mut(),
            corrected.key_row_bytes.arg(),
            corrected.u.arg_mut(),
            corrected.u_row_bytes.arg(),
            state.segments.arg(),
            ba_paged.page_ints.arg(),
        ],
    )
}

/// Each beam group's conv fold before the layer's step
/// (`causal_conv1d_group_fold`): its window moved on by the rows every live
/// beam shares, read from its first lane's buffer, for the groups whose lane
/// lies in this window (`lane0`, `lanes`).
#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d_group_fold(
    ctx: &Ctx<'_>,
    folds: GroupFolds,
    (lane0, lanes): (u32, u32),
    slots: Tensor,
    paged: Paged,
    state: &RecurrentPool,
    channels: u32,
    conv_width: u32,
    dilation: u32,
) -> Result<(), Error> {
    const OP: &str = "ssm.causal_conv1d_group_fold";
    if folds.count == 0 {
        return Ok(());
    }
    if !causal_conv1d_verify_fits(channels, conv_width, dilation) {
        return Err(refuse(
            OP,
            format!(
                "the group fold runs four of a multiple of four channels, at most \
                 {CONV_X4_TAPS} taps and {CONV_VERIFY_HIST} rows of history; this one is \
                 {channels} channels, {conv_width} taps at dilation {dilation}"
            ),
        ));
    }
    ctx.fire(
        Fire::at("ssm/conv1d.metal", "causal_conv1d_group_fold_bfloat16").apply(Grid::of(
            [channels / 4, 1, 1],
            [(channels / 4).min(CONV_VERIFY_CHANNELS * CONV_VERIFY_ROWS), 1, 1],
        )),
        &[
            folds.table.arg(),
            stated(OP, lane0)?.arg(),
            stated(OP, lanes)?.arg(),
            slots.arg(),
            state.new_conv_state.arg_mut(),
            stated(OP, channels)?.arg(),
            stated(OP, conv_width)?.arg(),
            stated(OP, dilation)?.arg(),
            paged.slab.arg_mut(),
            paged.pages.arg(),
            paged.page_tokens.arg(),
            paged.page_bytes.arg(),
            paged.row_bytes.arg(),
            paged.page_ints.arg(),
            state.segments.arg(),
        ],
    )
}

/// The rows of a lane one threadgroup of the verify conv runs side by side,
/// and the threads of four channels beside them.
const CONV_VERIFY_ROWS: u32 = 4;
const CONV_VERIFY_CHANNELS: u32 = 32;

/// The history the verify conv commits, at most: a thread holds
/// `kConvVerifyHeld` rows of it, four.
const CONV_VERIFY_HIST: u32 = 4 * CONV_VERIFY_ROWS;

/// Whether [`causal_conv1d_verify`] holds a conv this wide over this many
/// channels: four a thread, as the prompt's four-channel conv.
#[must_use]
pub fn causal_conv1d_verify_fits(channels: u32, conv_width: u32, dilation: u32) -> bool {
    channels.is_multiple_of(4)
        && conv_width <= CONV_X4_TAPS
        && conv_history("", conv_width, dilation).is_ok_and(|hist| hist <= CONV_VERIFY_HIST)
}

/// [`causal_conv1d_committed_landing`] with a thread a row: every row lands
/// in the extended plane `y`, and a row of the window's own in `land`, the
/// window's own rows. Through `paged`, `x` is the window's own rows: the
/// replayed rows are read from the lane's buffer, and the window's own land
/// there.
#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d_verify(
    ctx: &Ctx<'_>,
    x: Tensor,
    paged: Option<Paged>,
    indptr: Tensor,
    committed: &Committed,
    weight: Tensor,
    state: &RecurrentPool,
    conv_width: u32,
    dilation: u32,
    y: Tensor,
    land: Tensor,
    snapshots: Option<Snapshots>,
    own_only: bool,
) -> Result<(), Error> {
    const OP: &str = "ssm.causal_conv1d_committed";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "causal_conv1d_verify_bfloat16" });
    let channels = nonzero(OP, "the conv's channel count", y.width)?;
    nonzero(OP, "extended rows", y.rows)?;
    debug_assert!(
        land.width == y.width && land.dtype == x.dtype,
        "the window's own rows land as the conv's row"
    );
    let taps = stated(
        OP,
        nonzero(OP, "the conv width this statement states", conv_width)?,
    )?;
    if !causal_conv1d_verify_fits(channels, conv_width, dilation) {
        return Err(refuse(
            OP,
            format!(
                "the verify conv runs four of a multiple of four channels, at most \
                 {CONV_X4_TAPS} taps and {CONV_VERIFY_HIST} rows of history; this one is \
                 {channels} channels, {conv_width} taps at dilation {dilation}"
            ),
        ));
    }
    let lanes = committed_lanes(OP, indptr)?;
    // A decode window through a page table: a thread a channel, as the
    // prompt conv (`causal_conv1d_carried`).
    if let (true, Some(paged), None) = (own_only, paged, snapshots.as_ref()) {
        return ctx.fire(
            Fire::at("ssm/conv1d.metal", "causal_conv1d_carried_bfloat16").apply(Grid::of(
                [channels, lanes, 1],
                [channels.min(CONV_GROUP), 1, 1],
            )),
            &[
                x.arg(),
                indptr.arg(),
                committed.replay.arg(),
                committed.commit.arg(),
                committed.slots.arg(),
                stated(OP, committed.lane0)?.arg(),
                weight.arg(),
                state.new_conv_state.arg_mut(),
                y.arg_mut(),
                land.arg_mut(),
                stated(OP, channels)?.arg(),
                taps.arg(),
                stated(OP, dilation)?.arg(),
                paged.slab.arg_mut(),
                paged.pages.arg(),
                paged.page_tokens.arg(),
                paged.page_bytes.arg(),
                paged.row_bytes.arg(),
                i32::try_from(paged.page_ints.max(1)).unwrap_or(i32::MAX).arg(),
                state.segments.arg(),
                committed.ext_replay.arg(),
            ],
        );
    }
    let [snaps, lane0] = snapshot_args(ctx, OP, snapshots)?;
    let [slab, pages, page_tokens, page_bytes, row_bytes, is_paged] = paged_args(ctx, paged)?;
    // A decode window that makes its own row's output alone runs one row of
    // threads: the conv's rows are its one own row, and a thread commits
    // every row of a history it holds whole (`kConvVerifyHeld`).
    let hist = (conv_width - 1) * dilation + 1;
    let grid = if own_only && hist <= CONV_VERIFY_HIST / CONV_VERIFY_ROWS {
        Grid::of(
            [channels / 4, 1, lanes],
            [(channels / 4).min(CONV_VERIFY_CHANNELS * CONV_VERIFY_ROWS), 1, 1],
        )
    } else {
        Grid::of(
            [channels / 4, CONV_VERIFY_ROWS, lanes],
            [CONV_VERIFY_CHANNELS, CONV_VERIFY_ROWS, 1],
        )
    };
    ctx.fire(
        Fire::at("ssm/conv1d.metal", entry).apply(grid),
        &[
            x.arg(),
            indptr.arg(),
            committed.replay.arg(),
            committed.commit.arg(),
            committed.slots.arg(),
            stated(OP, committed.lane0)?.arg(),
            weight.arg(),
            state.conv_state.arg(),
            state.new_conv_state.arg_mut(),
            y.arg_mut(),
            land.arg_mut(),
            stated(OP, channels)?.arg(),
            taps.arg(),
            stated(OP, dilation)?.arg(),
            snaps,
            lane0,
            slab,
            pages,
            page_tokens,
            page_bytes,
            row_bytes,
            is_paged,
            state.segments.arg(),
            i32::from(own_only).arg(),
            committed.ext_replay.arg(),
        ],
    )
}
