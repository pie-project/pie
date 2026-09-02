//! Recurrent-state mixers — causal conv, gated delta nets, KDA. Chunked
//! forms are the prefill path: one scan per request instead of per token.

use crate::error::Error;
use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
use crate::tensor::{RaggedTensor, RecurrentPool, Tensor};

const CONV_GROUP: u32 = 256;

const SCAN_WIDTH: u32 = 128;

/// Metal file holding the register scan stamped by `gdn_scan_points!`.
const GDN_SCAN_FILE: &str = "attn/ssm_gdn_scan.metal";

/// The only lane count `ssm_gdn_scan.metal` is stamped at: one simdgroup, so
/// the shuffle tree needs no barrier. Any other tuned value falls back to
/// [`gated_delta_chunked`]'s untemplated kernel.
const GDN_SCAN_LANES: u32 = 32;

/// Lane groups per threadgroup — four simdgroups, 128 threads, which is the
/// threadgroup the reference launches this shape at.
const GDN_SCAN_TG_ROWS: u32 = 4;

/// The widest head row a scan stages in threadgroup memory.
const SCAN_HEAD_MAX: u32 = 256;

fn conv_grid(channels: u32, rows: u32) -> Grid {
    Grid::of([channels, rows, 1], [channels.min(CONV_GROUP), 1, 1])
}

const fn recurrence_grid(heads: u32, rows: u32) -> Grid {
    Grid::of([SCAN_WIDTH, heads, rows], [SCAN_WIDTH, 1, 1])
}

/// Every (VROWS, PER) point the register scan is stamped at, as literal
/// entry names and stamps. `PER` is `k_dim / 32` (head widths 64/128/256);
/// `VROWS` is [`crate::tuning::DeviceTuning::gdn_scan_rows`].
macro_rules! gdn_scan_points {
    ($(($v:literal, $p:literal)),+ $(,)?) => {
        fn gdn_scan_point(vrows: u32, per: u32) -> Option<(&'static str, &'static str)> {
            match (vrows, per) {
                $(($v, $p) => Some((
                    concat!("gated_delta_scan_bfloat16_l_32_v_", $v, "_p_", $p),
                    concat!(
                        "PIE_STAMP_gdn_scan(\"gated_delta_scan_bfloat16_l_32_v_",
                        $v, "_p_", $p, "\", 32, ", $v, ", ", $p, ")"
                    ),
                )),)+
                _ => None,
            }
        }
    };
}

gdn_scan_points!(
    (1, 2),
    (2, 2),
    (4, 2),
    (8, 2),
    (1, 4),
    (2, 4),
    (4, 4),
    (8, 4),
    (1, 8),
    (2, 8),
    (4, 8),
    (8, 8),
);

/// The register scan's point and geometry, or `None` to fall back to the
/// threadgroup kernel (shapes the register scan was not stamped for).
///
/// Launch is one lane group (one simdgroup) per `VROWS` value rows:
/// threads `[32, v_dim / VROWS, requests * v_heads]`, threadgroup `[32, 4, 1]`.
fn gdn_scan_launch(shape: &Delta, requests: u32) -> Option<(&'static str, &'static str, Grid)> {
    let tuned = crate::tuning::current();
    if tuned.gdn_scan_lanes != GDN_SCAN_LANES {
        return None;
    }
    gdn_scan_launch_at(shape, requests, tuned.gdn_scan_rows)
}

/// [`gdn_scan_launch`] with the fold stated rather than read, so a test can
/// reach a fold this machine's table does not name.
fn gdn_scan_launch_at(
    shape: &Delta,
    requests: u32,
    vrows: u32,
) -> Option<(&'static str, &'static str, Grid)> {
    if vrows == 0 || shape.v_dim % vrows != 0 || shape.k_dim % GDN_SCAN_LANES != 0 {
        return None;
    }
    let (entry, stamp) = gdn_scan_point(vrows, shape.k_dim / GDN_SCAN_LANES)?;
    let row_groups = shape.v_dim / vrows;
    // Threadgroup row extent must divide row_groups: this scan is a
    // read-modify-write, so a spare lane group sharing a row is wrong, not wasteful.
    let tg_rows = if row_groups % GDN_SCAN_TG_ROWS == 0 {
        GDN_SCAN_TG_ROWS
    } else {
        1
    };
    let z = shape.v_heads.checked_mul(requests)?;
    Some((
        entry,
        stamp,
        Grid::of(
            [GDN_SCAN_LANES, row_groups, z],
            [GDN_SCAN_LANES, tg_rows, 1],
        ),
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

/// Request count a ragged fire spans: indptr is `[lanes + 1]`.
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

/// The gated-delta shape: four stated head numbers against the fused rows.
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
        if v_heads % k_heads != 0 {
            return Err(refuse(
                op,
                format!("the {v_heads} value heads are not a whole number of the {k_heads} key heads"),
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

/// The KDA shape: two stated head numbers against the mixed rows.
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

/// History a dilated convolution keeps: `(conv_width − 1)·dilation + 1` rows
/// of channels. Also the state-bank slab stride the shader indexes by.
fn conv_history(op: &'static str, conv_width: u32, dilation: u32) -> Result<u32, Error> {
    nonzero(op, "the conv width this statement states", conv_width)?;
    nonzero(op, "the dilation this statement states", dilation)?;
    (conv_width - 1)
        .checked_mul(dilation)
        .and_then(|span| span.checked_add(1))
        .ok_or_else(|| {
            refuse(
                op,
                format!("a width of {conv_width} at dilation {dilation} keeps no countable history"),
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
    let taps = stated(OP, nonzero(OP, "the conv width this statement states", conv_width)?)?;
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

/// Prefill form: walks the fire's request boundaries, one threadgroup row
/// per request.
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
    let entry = dtype_dispatch!(OP, x.data.dtype, { Bf16 => "causal_conv1d_chunked_bfloat16" });
    let channels = nonzero(OP, "the conv's channel count", x.data.width)?;
    nonzero(OP, "rows", x.data.rows)?;
    debug_assert!(
        y.rows == x.data.rows && y.width == x.data.width,
        "the conv lands the row it convolves"
    );
    let taps = stated(OP, nonzero(OP, "the conv width this statement states", conv_width)?)?;
    let hist = conv_history(OP, conv_width, dilation)?;
    debug_assert_eq!(
        u64::from(state.conv_state.width),
        u64::from(hist) * u64::from(channels),
        "the state bank a dilated conv reads is `(width - 1) * dilation + 1` rows of channels"
    );
    let lanes = requests(OP, x)?;
    ctx.fire(
        Fire::at("attn/ssm_causal_conv1d.metal", entry).apply(conv_grid(channels, lanes)),
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
        ],
    )
}

/// Folds `ba` with the dt bias and A-log into per-head decay gates.
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
    if ba.width == 0 || ba.width % 2 != 0 {
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

/// `z` is unread here; this shader gates afterwards via `elementwise.rmsnorm_gated`.
#[allow(clippy::too_many_arguments)]
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
    let _ = z;
    let entry = dtype_dispatch!(OP, qkv.dtype, { Bf16 => "gated_delta_bfloat16" });
    debug_assert_eq!(gates.dtype, Dtype::F32, "`{OP}` reads an f32 decay row");
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Delta::of(OP, qkv, gates, y, k_heads, v_heads, k_dim, v_dim)?;
    let rows = nonzero(OP, "rows", qkv.rows)?;
    // The value columns of one (head, row) split across threadgroups down
    // z, 32 columns each (a simdgroup a column, four simdgroups a group, so
    // eight turns), so a one-row fire still spreads a head over the device.
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

/// Prefill form of [`gated_delta`]: one scan per request.
#[allow(clippy::too_many_arguments)]
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
    let _ = z;
    let entry = dtype_dispatch!(OP, qkv.data.dtype, { Bf16 => "gated_delta_chunked_bfloat16" });
    debug_assert_eq!(gates.dtype, Dtype::F32, "`{OP}` reads an f32 decay row");
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Delta::of(OP, qkv.data, gates, y, k_heads, v_heads, k_dim, v_dim)?;
    let lanes = requests(OP, qkv)?;
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
    ];
    // Register scan for stamped shapes: same operands/recurrence, but the
    // cell stays in registers and per-token folds are simdgroup shuffles.
    if let Some((point, stamp, grid)) = gdn_scan_launch(&shape, lanes) {
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

/// Prefill form of [`kda_step`]: one scan per request.
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

/// [`kda_chunked`] over the extended run, on a work copy of each lane's
/// bank, persisting the bank as of the lane's `commit`
/// ([`gated_delta_committed`]'s twin). `work` is
/// `[fire lanes][heads][head_dim][head_dim]` f32.
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

    // qwen3.6-27B's gated-delta shape.
    const D27B: Delta = Delta {
        k_heads: 16,
        v_heads: 48,
        k_dim: 128,
        v_dim: 128,
    };

    #[test]
    fn a_shape_the_stamp_does_not_name_falls_back() {
        // k_dim not divisible by 32 lanes.
        let odd = Delta {
            k_dim: 100,
            ..D27B
        };
        assert!(gdn_scan_launch(&odd, 1).is_none());
        // Divides evenly but PER=16 is outside the stamped packs (max 8).
        let deep = Delta {
            k_dim: 512,
            ..D27B
        };
        assert!(gdn_scan_launch(&deep, 1).is_none());
        // v_dim not divisible by the fold width.
        let ragged = Delta { v_dim: 66, ..D27B };
        assert!(gdn_scan_launch(&ragged, 1).is_none());
    }

}


// ── the committed arm ──────────────────────────────────────────────────────
//
// The buffered recurrence's launches (`engine_metal::rs`): the same conv and
// scan over an EXTENDED row run — buffered tokens replayed ahead of each
// lane's own rows — with the state persisted only as far as each lane's
// `commit`. The tables are the fire's, indexed at `lane0 + r`; the CSR is the
// window's own.

/// The per-lane tables a committed launch reads, fire-wide, at `lane0 + r`.
#[derive(Clone, Copy, Debug)]
pub struct Committed {
    /// `i32`, `[fire lanes]`: buffered rows replayed ahead of each lane's own.
    pub replay: Tensor,
    /// `i32`, `[fire lanes]`: rows (of the extended run) whose state persists.
    pub commit: Tensor,
    /// `i32`, `[fire lanes]`: which bank each lane owns.
    pub slots: Tensor,
    /// The window's first fire lane.
    pub lane0: u32,
}

fn committed_lanes(op: &'static str, indptr: Tensor) -> Result<u32, Error> {
    if indptr.dtype != Dtype::I32 {
        return Err(refuse(
            op,
            format!("the window CSR's boundaries are {:?}, and this arm walks an i32 indptr", indptr.dtype),
        ));
    }
    match indptr.rows.checked_sub(1) {
        Some(lanes) if lanes > 0 => Ok(lanes),
        _ => Err(refuse(op, "the window CSR this fire names spans no request")),
    }
}

/// [`causal_conv1d_chunked`] over the extended run, persisting each lane's
/// state as of its `commit`. `x` and `y` hold the extended rows end to end.
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
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "causal_conv1d_committed_bfloat16" });
    let channels = nonzero(OP, "the conv's channel count", x.width)?;
    nonzero(OP, "extended rows", x.rows)?;
    let taps = stated(OP, nonzero(OP, "the conv width this statement states", conv_width)?)?;
    let hist = conv_history(OP, conv_width, dilation)?;
    if hist > 64 {
        return Err(refuse(OP, format!("a history of {hist} rows is past the 64 this arm stages")));
    }
    let lanes = committed_lanes(OP, indptr)?;
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
        ],
    )
}

/// [`gated_delta_chunked`] over the extended run, on a work copy of each
/// lane's bank, persisting the bank as of the lane's `commit`. `work` is
/// `[fire lanes][v_heads][v_dim][k_dim]` f32.
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
    let entry = dtype_dispatch!(OP, qkv.dtype, { Bf16 => "gated_delta_committed_bfloat16" });
    debug_assert_eq!(gates.dtype, Dtype::F32, "`{OP}` reads an f32 decay row");
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Delta::of(OP, qkv, gates, y, k_heads, v_heads, k_dim, v_dim)?;
    let lanes = committed_lanes(OP, indptr)?;
    ctx.fire(
        Fire::at("attn/ssm_gated_delta.metal", entry).apply(recurrence_grid(shape.v_heads, lanes)),
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
