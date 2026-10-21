use crate::error::Error;
use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
use crate::tensor::{RaggedTensor, RecurrentPool, Tensor};
use crate::tuning::GdnScan;

const CONV_GROUP: u32 = 256;

/// The rows one thread of a prompt's conv runs in turn. One thread a channel
/// running a whole 555-row prompt left most of the GPU idle.
const CONV_RUN_ROWS: u32 = 16;

/// The widest conv the four-channel prompt conv holds (`PIE_CONV_X4_TAPS`).
const CONV_X4_TAPS: u32 = 8;

const SCAN_WIDTH: u32 = 128;

const GDN_SCAN_FILE: &str = "attn/ssm_gdn_scan.metal";

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
        Fire::at("attn/ssm_causal_conv1d.metal", entry).apply(conv_grid(channels, rows)),
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
        ],
    )
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
    causal_conv1d_chunked_landing(ctx, x, weight, state, conv_width, dilation, y, None)
}

/// [`causal_conv1d_chunked`], landing the window as of each of `snapshots`'
/// rows in its slot too.
#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d_chunked_landing(
    ctx: &Ctx<'_>,
    x: RaggedTensor,
    weight: Tensor,
    state: &RecurrentPool,
    conv_width: u32,
    dilation: u32,
    y: Tensor,
    snapshots: Option<Snapshots>,
) -> Result<(), Error> {
    const OP: &str = "attention.ssm_causal_conv1d_chunked";
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
        Fire::at("attn/ssm_causal_conv1d.metal", entry).apply(Grid::of(
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
    let entry = dtype_dispatch!(OP, ba.dtype, { Bf16 => "qwen_gdn_ba_gates_bfloat16" });
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
        Fire::at("attn/ssm_gdn_prep.metal", entry)
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
    const OP: &str = "attention.ssm_gated_delta";
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
            ],
        );
    }
    let splits = (shape.v_dim / 32).max(1);
    let grid = Grid::of(
        [SCAN_WIDTH, shape.v_heads, rows.saturating_mul(splits)],
        [SCAN_WIDTH, 1, 1],
    );
    ctx.fire(
        Fire::at("attn/ssm_gated_delta.metal", entry).apply(grid),
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
    const OP: &str = "attention.ssm_gated_delta_chunked";
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
    ];
    if let Some(grid) = gdn_staged_launch_at(&shape, lanes, qkv.data.rows, tuning.gdn_scan()) {
        return ctx.fire(
            Fire::at(
                "attn/ssm_gdn_staged.metal",
                "gated_delta_staged_bfloat16_b16",
            )
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
        Fire::at("attn/ssm_gated_delta.metal", entry).apply(recurrence_grid(shape.v_heads, lanes)),
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
    const OP: &str = "attention.ssm_kda_step";
    let entry = dtype_dispatch!(OP, mixed.dtype, { Bf16 => "kda_step_bfloat16" });
    debug_assert_eq!(dt_bias.dtype, Dtype::F32, "`{OP}` reads an f32 decay bias");
    debug_assert_eq!(a_log.dtype, Dtype::F32, "`{OP}` reads an f32 decay bank");
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Kda::of(OP, mixed, f, b, y, heads, head_dim)?;
    let rows = nonzero(OP, "rows", mixed.rows)?;
    ctx.fire(
        Fire::at("attn/ssm_kda.metal", entry).apply(recurrence_grid(shape.heads, rows)),
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
    const OP: &str = "attention.ssm_kda_chunked";
    let entry = dtype_dispatch!(OP, mixed.data.dtype, { Bf16 => "kda_chunked_bfloat16" });
    debug_assert_eq!(dt_bias.dtype, Dtype::F32, "`{OP}` reads an f32 decay bias");
    debug_assert_eq!(a_log.dtype, Dtype::F32, "`{OP}` reads an f32 decay bank");
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Kda::of(OP, mixed.data, f, b, y, heads, head_dim)?;
    let lanes = requests(OP, mixed)?;
    ctx.fire(
        Fire::at("attn/ssm_kda.metal", entry).apply(recurrence_grid(shape.heads, lanes)),
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
    const OP: &str = "attention.ssm_kda_committed";
    let entry = dtype_dispatch!(OP, mixed.dtype, { Bf16 => "kda_committed_bfloat16" });
    debug_assert_eq!(dt_bias.dtype, Dtype::F32, "`{OP}` reads an f32 decay bias");
    debug_assert_eq!(a_log.dtype, Dtype::F32, "`{OP}` reads an f32 decay bank");
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Kda::of(OP, mixed, f, b, y, heads, head_dim)?;
    let lanes = committed_lanes(OP, indptr)?;
    ctx.fire(
        Fire::at("attn/ssm_kda.metal", entry).apply(recurrence_grid(shape.heads, lanes)),
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
    const OP: &str = "attention.ssm_causal_conv1d_committed";
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
        Fire::at("attn/ssm_causal_conv1d.metal", entry).apply(conv_grid(channels, lanes)),
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
    const OP: &str = "attention.ssm_gated_delta_committed";
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
        Fire::at("attn/ssm_gated_delta.metal", entry).apply(grid),
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
        ],
    )
}

/// The ints of one lane's row in a recurrent buffer's page table
/// (`PIE_RS_PAGE_INTS`).
pub const RS_PAGE_INTS: u32 = 8;

/// A recurrent buffer's plane as a verify arm reads it through the fire's
/// page table: each lane's row of `pages` names the in-page token of its
/// first extended row, then the slots of the pages its rows touch.
#[derive(Clone, Copy, Debug)]
pub struct Paged {
    pub slab: Tensor,
    pub pages: Tensor,
    pub page_tokens: u32,
    pub page_bytes: u32,
    pub plane_at: u32,
    pub row_bytes: u32,
}

fn paged_args(ctx: &Ctx<'_>, paged: Option<Paged>) -> Result<[crate::encode::ArgValue; 7], Error> {
    Ok(match paged {
        Some(paged) => [
            paged.slab.arg_mut(),
            paged.pages.arg(),
            paged.page_tokens.arg(),
            paged.page_bytes.arg(),
            paged.plane_at.arg(),
            paged.row_bytes.arg(),
            1i32.arg(),
        ],
        None => [
            ctx.absent()?,
            ctx.absent()?,
            0u32.arg(),
            0u32.arg(),
            0u32.arg(),
            0u32.arg(),
            0i32.arg(),
        ],
    })
}

/// The `(lanes a value row, simdgroups, value rows a lane)` of a verify
/// scan's threadgroup.
const GDN_VERIFY_GROUP: (i32, u32, u32) = (32, 4, 4);

/// The key and value head width the verify scan stages.
const GDN_VERIFY_DIM: u32 = 128;

/// The batched verify scan stamped at `(simdgroups, value rows, batches)`.
#[must_use]
pub fn gated_delta_verify_batched_entry(
    groups: u32,
    vrows: u32,
    batches: u32,
) -> Option<&'static str> {
    Some(match (groups, vrows, batches) {
        (8, 4, 4) => "gated_delta_verify_batched_bfloat16_s_8_v_4_b_4",
        (8, 2, 8) => "gated_delta_verify_batched_bfloat16_s_8_v_2_b_8",
        (4, 4, 4) => "gated_delta_verify_batched_bfloat16_s_4_v_4_b_4",
        (8, 4, 2) => "gated_delta_verify_batched_bfloat16_s_8_v_4_b_2",
        (4, 4, 2) => "gated_delta_verify_batched_bfloat16_s_4_v_4_b_2",
        (8, 2, 4) => "gated_delta_verify_batched_bfloat16_s_8_v_2_b_4",
        _ => return None,
    })
}

/// The launch of a batched verify scan over `heads` value heads of every
/// lane.
#[must_use]
pub fn gated_delta_verify_batched_grid(groups: u32, vrows: u32, batches: u32, heads: u32) -> Grid {
    let tiles = GDN_VERIFY_DIM / (groups * vrows * batches);
    Grid::of(
        [GDN_SCAN_LANES, groups * tiles, heads],
        [GDN_SCAN_LANES, groups, 1],
    )
}

/// The verify scan stamped at `(lanes, simdgroups, value rows)`, a lane's
/// keys side by side; at negative lanes, every `-lanes`-th.
#[must_use]
pub fn gated_delta_verify_entry(lanes: i32, groups: u32, vrows: u32) -> Option<&'static str> {
    Some(match (lanes, groups, vrows) {
        (32, 4, 4) => "gated_delta_verify_bfloat16_l_32_s_4_v_4",
        (32, 8, 4) => "gated_delta_verify_bfloat16_l_32_s_8_v_4",
        (32, 8, 2) => "gated_delta_verify_bfloat16_l_32_s_8_v_2",
        (8, 4, 1) => "gated_delta_verify_bfloat16_l_8_s_4_v_1",
        (8, 4, 2) => "gated_delta_verify_bfloat16_l_8_s_4_v_2",
        (-8, 4, 1) => "gated_delta_verify_bfloat16_l_8_s_4_v_1_strided",
        (-8, 4, 2) => "gated_delta_verify_bfloat16_l_8_s_4_v_2_strided",
        (-4, 4, 1) => "gated_delta_verify_bfloat16_l_4_s_4_v_1_strided",
        (-4, 8, 1) => "gated_delta_verify_bfloat16_l_4_s_8_v_1_strided",
        (-4, 4, 2) => "gated_delta_verify_bfloat16_l_4_s_4_v_2_strided",
        (-16, 4, 1) => "gated_delta_verify_bfloat16_l_16_s_4_v_1_strided",
        _ => return None,
    })
}

/// The launch of a verify scan at `(lanes, simdgroups, value rows)` over
/// `heads` value heads of every lane.
#[must_use]
pub fn gated_delta_verify_grid(lanes: i32, groups: u32, vrows: u32, heads: u32) -> Grid {
    let rows_a_simdgroup = (GDN_SCAN_LANES / lanes.unsigned_abs()) * vrows;
    Grid::of(
        [GDN_SCAN_LANES, GDN_VERIFY_DIM / rows_a_simdgroup, heads],
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
pub fn gated_delta_verify_at(
    ctx: &Ctx<'_>,
    (row_lanes, groups, vrows): (i32, u32, u32),
    qkv: Tensor,
    indptr: Tensor,
    committed: &Committed,
    gates: &VerifyGates,
    paged: Option<Paged>,
    state: &RecurrentPool,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    y: Tensor,
    snapshots: Option<Snapshots>,
) -> Result<(), Error> {
    const OP: &str = "attention.ssm_gated_delta_committed";
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
    let entry = gated_delta_verify_entry(row_lanes, groups, vrows)
        .ok_or_else(|| refuse(OP, "no verify scan is stamped at this geometry"))?;
    let lanes = committed_lanes(OP, indptr)?;
    let [snaps, lane0] = snapshot_args(ctx, OP, snapshots)?;
    let [
        slab,
        pages,
        page_tokens,
        page_bytes,
        plane_at,
        row_bytes,
        is_paged,
    ] = paged_args(ctx, paged)?;
    let z = v_heads
        .checked_mul(lanes)
        .ok_or_else(|| refuse(OP, "the value heads of every lane overflow a launch"))?;
    ctx.fire(
        Fire::at(GDN_SCAN_FILE, entry).apply(gated_delta_verify_grid(row_lanes, groups, vrows, z)),
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
            plane_at,
            row_bytes,
            is_paged,
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
) -> Result<(), Error> {
    const OP: &str = "attention.ssm_causal_conv1d_committed";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "causal_conv1d_verify_bfloat16" });
    let channels = nonzero(OP, "the conv's channel count", x.width)?;
    nonzero(OP, "extended rows", x.rows)?;
    debug_assert!(
        land.width == x.width && land.dtype == x.dtype,
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
    let [snaps, lane0] = snapshot_args(ctx, OP, snapshots)?;
    let [
        slab,
        pages,
        page_tokens,
        page_bytes,
        plane_at,
        row_bytes,
        is_paged,
    ] = paged_args(ctx, paged)?;
    ctx.fire(
        Fire::at("attn/ssm_causal_conv1d.metal", entry).apply(Grid::of(
            [channels / 4, CONV_VERIFY_ROWS, lanes],
            [CONV_VERIFY_CHANNELS, CONV_VERIFY_ROWS, 1],
        )),
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
            stated(OP, x.width)?.arg(),
            taps.arg(),
            stated(OP, dilation)?.arg(),
            snaps,
            lane0,
            slab,
            pages,
            page_tokens,
            page_bytes,
            plane_at,
            row_bytes,
            is_paged,
        ],
    )
}
