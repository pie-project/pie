mod dyn_conv;
pub use dyn_conv::dyn_conv;

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{
    Arg, ArgValue, Ctx, Fire, Launch, dtype_dispatch, kernel, nonzero, refuse, stated,
};
use crate::tensor::{RaggedTensor, RecurrentPool, StateRows, Tensor};

const FILE: &str = "ssm/mixers.hip";

/// A decode step's carried rows (`ssm/carry.hip`'s `Carry`), passed by
/// value: the staged rs tables a lane's buffered rows are read through
/// (`runtime::channel::RsTables`), each lane's commit length over its run
/// (carried rows, then its own), where the op's buffered planes lie in a
/// page's slot, and the carried scratch the conv lands its carried rows'
/// outputs in for a recurrence that re-steps them (row `RsLane::ext + j`,
/// the lane's `j`-th carried row). [`Carry::NONE`]: no lane carries a row,
/// lands one, or commits short of its own. `kernels-cuda`'s lays the same.
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Carry {
    pub lanes: u64,
    pub pages: u64,
    pub replay_spans: u64,
    pub commit_len: u64,
    pub plane_off: [u64; 4],
    pub plane_stride: [u64; 4],
    pub page_tokens: u32,
    pub pad: u32,
    /// The fire's beams' rows ([`crate::runtime::channel::RsRows`]), a
    /// fire lane each, read by a lane whose page offset names one
    /// ([`crate::runtime::channel::RsLane::ROWS`]); 0 where no lane is a beam.
    pub lane_rows: u64,
}

const _: () = assert!(size_of::<Carry>() == 112);

impl Carry {
    pub const NONE: Carry = Carry {
        lanes: 0,
        pages: 0,
        replay_spans: 0,
        commit_len: 0,
        plane_off: [0; 4],
        plane_stride: [0; 4],
        page_tokens: 1,
        pad: 0,
        lane_rows: 0,
    };

    /// The kernels' argument.
    #[must_use]
    pub fn arg(&self) -> ArgValue<'_> {
        ArgValue::of(self)
    }

    /// Whether lanes may carry rows: the fire buffers.
    #[must_use]
    pub const fn buffers(&self) -> bool {
        self.lanes != 0
    }
}

/// A beam group's fold (`BEAM-PAGE-LISTS.md` §11.3, `ssm/carry.hip`'s
/// `GroupFold`): its representative lane (any of its lanes; they share the
/// set's state), where its fold rows start in the replay table, and how
/// many. The fold kernels read a device table of them: the groups that fold
/// lead it, and the first entry of no rows ends it (a block folds each group
/// in turn, so an empty table costs a read, not a block a group).
#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct GroupFold {
    pub lane: u32,
    pub fold_off: u32,
    pub rows: u32,
}

const _: () = assert!(size_of::<GroupFold>() == 12);

/// A thread a channel for the group folds of the conv window.
const FOLD_CONV_BLOCK: u32 = 256;

/// Folds each group's rows into its conv window (§11.3): the window stored as
/// it stands after them, from the committed cells and the rows' inputs
/// (`carry`'s plane 0) (`ssm_conv1d_fold`).
#[allow(clippy::too_many_arguments)]
pub fn conv1d_fold(
    ctx: &Ctx,
    state: &RecurrentPool,
    channels: u32,
    conv_width: u32,
    dilation: u32,
    carry: &Carry,
    folds: Tensor,
    groups: u32,
) -> Result<(), Error> {
    const OP: &str = "ssm.causal_conv1d";
    if groups == 0 {
        return Ok(());
    }
    if (conv_width.saturating_sub(1)) * dilation + 1 > 8 {
        return Err(refuse(OP, format!("a {conv_width}-tap window of dilation {dilation}, past the 8 cells a fold stores")));
    }
    let rows = state.rows(OP, state.conv_slab, state.conv_stride, channels)?;
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::attn::ssm_conv1d_fold<::pie::bf16>"))
            .apply(Launch::grid([channels.div_ceil(FOLD_CONV_BLOCK), 1, 1], [FOLD_CONV_BLOCK, 1, 1])),
        &[
            rows.arg(),
            stated(OP, channels)?.arg(),
            stated(OP, conv_width)?.arg(),
            stated(OP, dilation)?.arg(),
            carry.arg(),
            folds.arg(),
            stated(OP, groups)?.arg(),
            ctx.stage(),
        ],
    )
}

/// Folds each group's rows into its gated-delta state (§11.3), from the kept
/// planes (`carry`: `[b | a]`, keys, corrected values), rounded as a token's
/// step stores them (`ssm_gdn_fold`).
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_fold(
    ctx: &Ctx,
    state: &RecurrentPool,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    dt_bias: Tensor,
    a_log: Tensor,
    carry: &Carry,
    folds: Tensor,
    groups: u32,
) -> Result<(), Error> {
    const OP: &str = "ssm.gated_delta";
    if groups == 0 {
        return Ok(());
    }
    if !(k_dim <= BLOCK * 8 / CARRIED_BV && v_dim.is_multiple_of(CARRIED_BV)) {
        return Err(refuse(
            OP,
            format!("a group's fold steps in the carried tile only: key heads of {k_dim} and value heads of {v_dim} do not lie in it"),
        ));
    }
    let rows = state.rows(OP, state.slab, state.slot_stride_elems, v_heads)?;
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::attn::ssm_gdn_fold<16>"))
            .apply(Launch::grid([v_dim / CARRIED_BV, v_heads, 1], [BLOCK, 1, 1])),
        &[
            rows.arg(),
            dt_bias.arg(),
            a_log.arg(),
            stated(OP, k_heads)?.arg(),
            stated(OP, v_heads)?.arg(),
            stated(OP, k_dim)?.arg(),
            stated(OP, v_dim)?.arg(),
            carry.arg(),
            folds.arg(),
            stated(OP, groups)?.arg(),
            ctx.stage(),
        ],
    )
}

/// The gate prep a gated-delta step's `gates` come from: its `[b | a]` rows
/// and its banks, which a carried row's gates are derived from as
/// `gdn_gates` derives them.
#[derive(Clone, Copy, Debug)]
pub struct GdnGated {
    pub ba: Tensor,
    pub dt_bias: Tensor,
    pub a_log: Tensor,
}

/// A value tile of the carried steps' blocks.
const CARRIED_BV: u32 = 16;

const BLOCK: u32 = 256;

const WAVE: u32 = 64;

const FLOAT: u32 = 4;

const PREP_BLOCK: u32 = 128;

/// A thread a channel for the decode conv.
const CONV_UPDATE_BLOCK: u32 = 256;

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

#[derive(Clone, Copy)]
enum StagePlanes {
    Chunk,
}

impl StagePlanes {
    const fn names(self) -> (&'static str, &'static str, &'static str) {
        match self {
            StagePlanes::Chunk => (
                "attn.ssm_gdn_chunk_qk",
                "attn.ssm_gdn_chunk_v",
                "attn.ssm_gdn_chunk_gates",
            ),
        }
    }
}

fn plane(ctx: &Ctx, op: &'static str, name: &'static str, elems: u64) -> Result<u64, Error> {
    let bytes = elems.checked_mul(u64::from(FLOAT)).ok_or_else(|| {
        refuse(
            op,
            format!("the {elems}-element staging plane will not size"),
        )
    })?;
    let bytes = usize::try_from(bytes)
        .map_err(|_| refuse(op, format!("{bytes} staging bytes do not fit this host")))?;
    Ok(ctx.scratch_shared(op, name, bytes)? as u64)
}

fn seated(
    op: &'static str,
    state: &RecurrentPool,
    arm: &'static str,
    mask: bool,
    commit: bool,
    begin: bool,
) -> Result<(), Error> {
    if !mask && !state.write_state_mask.is_absent() {
        return Err(refuse(
            op,
            format!(
                "this fire carries a per-request fold predicate and `{arm}` has no seat for \
                 one, so a refused pass would fold anyway"
            ),
        ));
    }
    if !commit && !state.commit_len.is_absent() {
        return Err(refuse(
            op,
            format!(
                "this fire carries a commit length and `{arm}` has no seat for one, so the \
                 replay would fold the whole buffered window instead of its accepted prefix"
            ),
        ));
    }
    if !begin && !state.begin_at.is_absent() {
        return Err(refuse(
            op,
            format!(
                "this fire cuts a row at an interior fold boundary and `{arm}` has no seat for \
                 the segment's origin, so the tail would replay the head's tokens from the state \
                 the head just folded"
            ),
        ));
    }
    if !state.write_state && !mask {
        return Err(refuse(
            op,
            format!("`{arm}` folds its boundary unconditionally and this fire asked it not to"),
        ));
    }
    Ok(())
}

fn conv_extents(
    op: &'static str,
    x: Tensor,
    y: &Tensor,
    conv_width: u32,
) -> Result<(u32, i32, i32), Error> {
    let channels = nonzero(op, "the conv's channel count", x.width)?;
    debug_assert!(
        y.rows == x.rows && y.width == x.width,
        "the conv lands the row it convolves"
    );
    Ok((
        channels,
        stated(op, channels)?,
        stated(
            op,
            nonzero(op, "the conv width this statement states", conv_width)?,
        )?,
    ))
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum ConvOut {
    Silu,
    Residual,
}

impl ConvOut {
    const fn args(self) -> &'static str {
        match self {
            ConvOut::Silu => "<::pie::bf16>",
            ConvOut::Residual => "<::pie::bf16, false, true>",
        }
    }

    /// The template arguments with the tap count after the element type.
    fn args_with(self, taps: &str) -> String {
        match self {
            ConvOut::Silu => format!("<::pie::bf16, {taps}>"),
            ConvOut::Residual => format!("<::pie::bf16, {taps}, false, true>"),
        }
    }
}

#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d(
    ctx: &Ctx,
    x: Tensor,
    weight: Tensor,
    bias: Option<Tensor>,
    state: &RecurrentPool,
    conv_width: u32,
    dilation: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "ssm.causal_conv1d";
    conv1d_update(
        ctx,
        OP,
        x,
        weight,
        bias,
        state,
        conv_width,
        dilation,
        y,
        ConvOut::Silu,
    )
}

pub fn short_conv(
    ctx: &Ctx,
    x: Tensor,
    weight: Tensor,
    state: &RecurrentPool,
    conv_width: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "ssm.short_conv";
    conv1d_update(
        ctx,
        OP,
        x,
        weight,
        None,
        state,
        conv_width,
        1,
        y,
        ConvOut::Residual,
    )
}

#[allow(clippy::too_many_arguments)]
fn conv1d_update(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    weight: Tensor,
    bias: Option<Tensor>,
    state: &RecurrentPool,
    conv_width: u32,
    dilation: u32,
    y: &mut Tensor,
    out: ConvOut,
) -> Result<(), Error> {
    dtype_dispatch!(op, x.dtype, { Bf16 => () });
    let (channels, c, k) = conv_extents(op, x, y, conv_width)?;
    let dil = stated(op, nonzero(op, "the conv's dilation", dilation)?)?;
    let n = nonzero(op, "rows", x.rows)?;
    seated(
        op,
        state,
        "ssm_causal_conv1d_update_batched",
        true,
        false,
        false,
    )?;
    // The window cuts along its channels.
    let rows = state.rows(op, state.conv_slab, state.conv_stride, channels)?;
    // A thread owns one channel of a row.
    let entrypoint = format!("::pie::attn::ssm_conv1d_update{}", out.args());
    let launch = Launch::grid(
        [channels.div_ceil(CONV_UPDATE_BLOCK), n, 1],
        [CONV_UPDATE_BLOCK, 1, 1],
    );
    let args = [
        x.arg(),
        weight.arg(),
        bias.map_or(ArgValue::ABSENT, |b| b.arg()),
        rows.arg(),
        y.arg(),
        stated(op, n)?.arg(),
        c.arg(),
        k.arg(),
        dil.arg(),
        state.write_state.arg(),
        state.write_state_mask.arg(),
        ctx.stage(),
    ];
    ctx.fire(op, Fire::at(kernel(FILE, &entrypoint)).apply(launch), &args)
}

/// The causal conv's decode step over each lane's carried rows then its own
/// ([`Carry`]): `ssm_conv1d_update`'s sums a row, the own row's window from
/// the committed cells and at most `conv_width - 1` carried rows, the carried
/// rows' outputs into the carried scratch where it is laid (a recurrence
/// that re-steps them), the window stored at the lane's commit length; the
/// own row's input lands in the lane's buffer (plane 0), and its output's
/// channels `keys` (offset, count: a gated-delta layer's keys) in plane 1.
#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d_carried(
    ctx: &Ctx,
    x: Tensor,
    weight: Tensor,
    bias: Option<Tensor>,
    state: &RecurrentPool,
    conv_width: u32,
    dilation: u32,
    y: &mut Tensor,
    carry: &Carry,
    keys: Option<(u32, u32)>,
) -> Result<(), Error> {
    const OP: &str = "ssm.causal_conv1d";
    conv1d_carried(
        ctx,
        OP,
        x,
        weight,
        bias,
        state,
        conv_width,
        dilation,
        y,
        ConvOut::Silu,
        carry,
        keys,
    )
}

/// [`short_conv`] over each lane's carried rows then its own, as
/// [`causal_conv1d_carried`].
#[allow(clippy::too_many_arguments)]
pub fn short_conv_carried(
    ctx: &Ctx,
    x: Tensor,
    weight: Tensor,
    state: &RecurrentPool,
    conv_width: u32,
    y: &mut Tensor,
    carry: &Carry,
) -> Result<(), Error> {
    const OP: &str = "ssm.short_conv";
    conv1d_carried(
        ctx,
        OP,
        x,
        weight,
        None,
        state,
        conv_width,
        1,
        y,
        ConvOut::Residual,
        carry,
        None,
    )
}

/// The taps the carried conv sums a row (`ssm_conv1d_carried`'s `KMAX`).
const CONV_CARRIED_TAPS: u32 = 8;

#[allow(clippy::too_many_arguments)]
fn conv1d_carried(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    weight: Tensor,
    bias: Option<Tensor>,
    state: &RecurrentPool,
    conv_width: u32,
    dilation: u32,
    y: &mut Tensor,
    out: ConvOut,
    carry: &Carry,
    keys: Option<(u32, u32)>,
) -> Result<(), Error> {
    dtype_dispatch!(op, x.dtype, { Bf16 => () });
    let (channels, c, k) = conv_extents(op, x, y, conv_width)?;
    let dil = stated(op, nonzero(op, "the conv's dilation", dilation)?)?;
    let n = nonzero(op, "rows", x.rows)?;
    if conv_width > CONV_CARRIED_TAPS {
        return Err(refuse(
            op,
            format!("a {conv_width}-tap conv, past the {CONV_CARRIED_TAPS} taps the carried step sums"),
        ));
    }
    seated(op, state, "ssm_conv1d_carried", true, false, false)?;
    let rows = state.rows(op, state.conv_slab, state.conv_stride, channels)?;
    let entrypoint = format!("::pie::attn::ssm_conv1d_carried{}", out.args());
    let launch = Launch::grid(
        [channels.div_ceil(CONV_UPDATE_BLOCK), n, 1],
        [CONV_UPDATE_BLOCK, 1, 1],
    );
    let args = [
        x.arg(),
        weight.arg(),
        bias.map_or(ArgValue::ABSENT, |b| b.arg()),
        rows.arg(),
        y.arg(),
        stated(op, n)?.arg(),
        c.arg(),
        k.arg(),
        dil.arg(),
        state.write_state.arg(),
        state.write_state_mask.arg(),
        carry.arg(),
        stated(op, keys.map_or(0, |(off, _)| off))?.arg(),
        stated(op, keys.map_or(0, |(_, len)| len))?.arg(),
        ctx.stage(),
    ];
    ctx.fire(op, Fire::at(kernel(FILE, &entrypoint)).apply(launch), &args)
}

#[allow(clippy::too_many_arguments)]
pub fn causal_conv1d_chunked(
    ctx: &Ctx,
    x: RaggedTensor,
    weight: Tensor,
    bias: Option<Tensor>,
    state: &RecurrentPool,
    conv_width: u32,
    dilation: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "ssm.causal_conv1d_chunked";
    conv1d_chunked(
        ctx,
        OP,
        x,
        weight,
        bias,
        state,
        conv_width,
        dilation,
        y,
        ConvOut::Silu,
    )
}

pub fn short_conv_chunked(
    ctx: &Ctx,
    x: RaggedTensor,
    weight: Tensor,
    state: &RecurrentPool,
    conv_width: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "ssm.short_conv_chunked";
    conv1d_chunked(
        ctx,
        OP,
        x,
        weight,
        None,
        state,
        conv_width,
        1,
        y,
        ConvOut::Residual,
    )
}

#[allow(clippy::too_many_arguments)]
fn conv1d_chunked(
    ctx: &Ctx,
    op: &'static str,
    x: RaggedTensor,
    weight: Tensor,
    bias: Option<Tensor>,
    state: &RecurrentPool,
    conv_width: u32,
    dilation: u32,
    y: &mut Tensor,
    out: ConvOut,
) -> Result<(), Error> {
    const TILE_BLOCK: u32 = 128;
    const CONV_TT: u32 = 32;

    dtype_dispatch!(op, x.data.dtype, { Bf16 => () });
    let (channels, c, k) = conv_extents(op, x.data, y, conv_width)?;
    let dil = stated(op, nonzero(op, "the conv's dilation", dilation)?)?;
    let lanes = requests(op, x)?;
    seated(
        op,
        state,
        "ssm_causal_conv1d_chunked_batched",
        true,
        true,
        true,
    )?;
    let rows = state.rows(op, state.conv_slab, state.conv_stride, channels)?;
    // Dilation 1 at 2..=4 taps over whole 16-byte channel runs (a piece's
    // too) takes the eight-channel sliding walk (`ssm_conv1d_chunked_v8`).
    let vectored = dil == 1
        && (2..=4).contains(&k)
        && channels % 8 == 0
        && crate::jit::aligned16(x.data.ptr)
        && crate::jit::aligned16(y.ptr)
        && crate::jit::aligned16(state.conv_slab.ptr)
        && rows.keeps(8)
        && rows.atoms().is_multiple_of(8);
    if vectored {
        let entrypoint = format!(
            "::pie::attn::ssm_conv1d_chunked_v8{}",
            out.args_with(&format!("{k}"))
        );
        let launch = Launch::grid(
            [
                (channels / 8).div_ceil(TILE_BLOCK),
                x.data.rows.div_ceil(CONV_TT).max(1),
                1,
            ],
            [TILE_BLOCK, 1, 1],
        );
        ctx.fire(
            op,
            Fire::at(kernel(FILE, &entrypoint)).apply(launch),
            &[
                x.data.arg(),
                weight.arg(),
                bias.map_or(ArgValue::ABSENT, |b| b.arg()),
                y.arg(),
                rows.arg(),
                x.indptr.arg(),
                c.arg(),
                state.commit_len.arg(),
                state.begin_at.arg(),
                ctx.stage(),
                stated(op, lanes)?.arg(),
            ],
        )?;
    } else {
        let entrypoint = format!("::pie::attn::ssm_conv1d_chunked{}", out.args());
        let launch = Launch::grid(
            [
                channels.div_ceil(TILE_BLOCK),
                x.data.rows.div_ceil(CONV_TT).max(1),
                1,
            ],
            [TILE_BLOCK, 1, 1],
        );
        ctx.fire(
            op,
            Fire::at(kernel(FILE, &entrypoint)).apply(launch),
            &[
                x.data.arg(),
                weight.arg(),
                bias.map_or(ArgValue::ABSENT, |b| b.arg()),
                y.arg(),
                rows.arg(),
                x.indptr.arg(),
                c.arg(),
                k.arg(),
                dil.arg(),
                state.write_state.arg(),
                state.write_state_mask.arg(),
                state.commit_len.arg(),
                state.begin_at.arg(),
                ctx.stage(),
                stated(op, lanes)?.arg(),
            ],
        )?;
    }
    if !state.write_state {
        return Ok(());
    }
    let [snap, snap_rows, snap_slots] = state.snapshots.args();
    ctx.fire(
        op,
        Fire::at(kernel(FILE, "::pie::attn::ssm_conv1d_state_commit")).apply(Launch::grid(
            [channels.div_ceil(TILE_BLOCK), lanes, 1],
            [TILE_BLOCK, 1, 1],
        )),
        &[
            x.data.arg(),
            rows.arg(),
            x.indptr.arg(),
            c.arg(),
            k.arg(),
            dil.arg(),
            state.write_state_mask.arg(),
            state.commit_len.arg(),
            state.begin_at.arg(),
            snap,
            snap_rows,
            snap_slots,
            ctx.stage(),
            stated(op, lanes)?.arg(),
        ],
    )
}

pub fn gdn_gates(
    ctx: &Ctx,
    ba: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    gates: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "ssm.gdn_gates";
    dtype_dispatch!(OP, ba.dtype, { Bf16 => () });
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
        OP,
        Fire::at(kernel(FILE, "::pie::attn::ssm_gdn_prep_ba_gates")).apply(Launch::grid(
            [rows, v_heads.div_ceil(BLOCK), 1],
            [BLOCK, 1, 1],
        )),
        &[
            ba.arg(),
            a_log.arg(),
            dt_bias.arg(),
            gates.arg(),
            stated(OP, rows)?.arg(),
            stated(OP, v_heads)?.arg(),
            ctx.stage(),
        ],
    )
}

#[derive(Clone, Copy)]
struct Delta {
    n: u32,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    conv_dim: u32,
}

#[derive(Clone, Copy)]
struct DeltaStaged {
    q_norm: u64,
    k_norm: u64,
    v: u64,
    g_log: u64,
    beta: u64,
}

impl Delta {
    #[allow(clippy::too_many_arguments)]
    fn of(
        op: &'static str,
        qkv: Tensor,
        gates: Tensor,
        y: &Tensor,
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
        nonzero(op, "rows", qkv.rows)?;
        Ok(Self {
            n: qkv.rows,
            k_heads,
            v_heads,
            k_dim,
            v_dim,
            conv_dim: qkv.width,
        })
    }

    const fn elems(self, heads: u32, width: u32) -> u64 {
        self.n as u64 * heads as u64 * width as u64
    }

    fn stage(
        self,
        ctx: &Ctx,
        op: &'static str,
        qkv: Tensor,
        gates: Tensor,
        planes: StagePlanes,
    ) -> Result<DeltaStaged, Error> {
        let key = self.elems(self.k_heads, self.k_dim);
        let val = self.elems(self.v_heads, self.v_dim);
        let decay = self.elems(self.v_heads, 1);

        let (qk_name, v_name, gates_name) = planes.names();
        let qk = plane(ctx, op, qk_name, 2 * key)?;
        let v = plane(ctx, op, v_name, val)?;
        let gb = plane(ctx, op, gates_name, 2 * decay)?;
        let staged = DeltaStaged {
            q_norm: qk,
            k_norm: qk + key * u64::from(FLOAT),
            v,
            g_log: gb,
            beta: gb + decay * u64::from(FLOAT),
        };

        #[allow(clippy::cast_precision_loss)]
        let q_scale = (self.k_dim as f32).sqrt().recip();
        ctx.fire(
            op,
            Fire::at(kernel(FILE, "::pie::attn::ssm_gdn_prep_qk_norm<128>"))
                .apply(Launch::grid([self.n, self.k_heads, 1], [PREP_BLOCK, 1, 1])),
            &[
                qkv.arg(),
                ArgValue::Ptr(staged.q_norm),
                ArgValue::Ptr(staged.k_norm),
                stated(op, self.k_heads)?.arg(),
                stated(op, self.k_dim)?.arg(),
                stated(op, self.conv_dim)?.arg(),
                q_scale.arg(),
                ctx.stage(),
            ],
        )?;
        ctx.fire(
            op,
            Fire::at(kernel(FILE, "::pie::attn::ssm_gdn_prep_v_gates<128>"))
                .apply(Launch::grid([self.n, self.v_heads, 1], [PREP_BLOCK, 1, 1])),
            &[
                qkv.arg(),
                gates.arg(),
                ArgValue::Ptr(staged.v),
                ArgValue::Ptr(staged.g_log),
                ArgValue::Ptr(staged.beta),
                stated(op, self.k_heads)?.arg(),
                stated(op, self.v_heads)?.arg(),
                stated(op, self.k_dim)?.arg(),
                stated(op, self.v_dim)?.arg(),
                stated(op, self.conv_dim)?.arg(),
                ctx.stage(),
            ],
        )?;
        Ok(staged)
    }
}

#[allow(clippy::too_many_arguments)]
pub fn gated_delta(
    ctx: &Ctx,
    qkv: Tensor,
    z: Tensor,
    gates: Tensor,
    state: &RecurrentPool,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "ssm.gated_delta";

    let _ = z;
    dtype_dispatch!(OP, qkv.dtype, { Bf16 => () });
    debug_assert_eq!(gates.dtype, Dtype::F32, "`{OP}` reads an f32 decay row");
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Delta::of(OP, qkv, gates, y, k_heads, v_heads, k_dim, v_dim)?;
    seated(
        OP,
        state,
        "ssm_gated_delta_step_batched_gqa",
        true,
        false,
        false,
    )?;
    // The state cuts along its value heads.
    let rows = state.rows(OP, state.slab, state.slot_stride_elems, v_heads)?;
    // A block a (head, row) at any batch: even one row of 48 heads runs in
    // under half the tiled step's time.
    if shape.k_dim == CHUNK_HEAD
        && shape.v_dim == CHUNK_HEAD
        && rows.keeps(8)
        && shape.conv_dim.is_multiple_of(8)
    {
        return shape.decode_rows(ctx, OP, qkv, gates, &rows, state, y);
    }
    if fused_fits(&shape, &rows) {
        return shape.decode_fused(ctx, OP, qkv, gates, &rows, state, y);
    }
    Err(refuse(
        OP,
        format!(
            "the decode step is built for 8-aligned rows, a value head in whole 16-wide \
             tiles and a key head of at most 128; this one is {k_dim} x {v_dim}"
        ),
    ))
}

/// Whether the carried steps' tile holds a head: a key head of at most a
/// block's state rows, a value head in whole 16-wide tiles, 8-aligned rows.
fn carried_fits(shape: &Delta, rows: &StateRows) -> bool {
    const VEC: u32 = 8;
    shape.k_dim <= BLOCK * VEC / CARRIED_BV
        && shape.v_dim.is_multiple_of(CARRIED_BV)
        && shape.conv_dim.is_multiple_of(VEC)
        && (shape.k_heads * shape.k_dim).is_multiple_of(VEC)
        && rows.keeps(i64::from(VEC))
}

/// The gated-delta decode step over each lane's buffered rows then its own
/// ([`Carry`]: planes `[b | a]`, keys, corrected values): the rows the lane
/// commits fold first, the rest read as a linear attention for the own
/// row's corrected value and output; the state is stored at the lane's
/// commit, and the own row's `[b | a]` and corrected value land in its
/// buffer (`ssm_gdn_step_carried`).
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_carried(
    ctx: &Ctx,
    qkv: Tensor,
    gates: Tensor,
    state: &RecurrentPool,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    y: &mut Tensor,
    gated: GdnGated,
    carry: &Carry,
) -> Result<(), Error> {
    const OP: &str = "ssm.gated_delta";
    dtype_dispatch!(OP, qkv.dtype, { Bf16 => () });
    dtype_dispatch!(OP, gated.ba.dtype, { Bf16 => () });
    dtype_dispatch!(OP, gated.dt_bias.dtype, { Bf16 => () });
    debug_assert_eq!(gates.dtype, Dtype::F32, "`{OP}` reads an f32 decay row");
    debug_assert_eq!(gated.a_log.dtype, Dtype::F32, "`{OP}` reads an f32 decay bank");
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Delta::of(OP, qkv, gates, y, k_heads, v_heads, k_dim, v_dim)?;
    seated(OP, state, "ssm_gdn_step_carried", true, false, false)?;
    let rows = state.rows(OP, state.slab, state.slot_stride_elems, v_heads)?;
    if !carried_fits(&shape, &rows) {
        return Err(refuse(
            OP,
            format!(
                "a lane's carried rows step in a tile of 8-aligned rows, a value head in whole \
                 16-wide tiles and a key head of at most 128: {k_heads} key heads of {k_dim} \
                 and {v_heads} value heads of {v_dim} do not lie in it"
            ),
        ));
    }
    let n = nonzero(OP, "rows", qkv.rows)?;
    #[allow(clippy::cast_precision_loss)]
    let q_scale = (k_dim as f32).sqrt().recip();
    // A block a (head, row) holding the whole state, as the step's own
    // batched form, where the head is the one it is built for.
    if shape.k_dim == CHUNK_HEAD && shape.v_dim == CHUNK_HEAD {
        return ctx.fire(
            OP,
            Fire::at(kernel(FILE, "::pie::attn::ssm_gdn_rows_carried<128, 128, 256>"))
                .apply(Launch::grid([v_heads, n, 1], [FUSED_BLOCK, 1, 1])),
            &[
                qkv.arg(),
                gates.arg(),
                gated.ba.arg(),
                gated.dt_bias.arg(),
                gated.a_log.arg(),
                rows.arg(),
                y.arg(),
                stated(OP, k_heads)?.arg(),
                stated(OP, v_heads)?.arg(),
                stated(OP, shape.conv_dim)?.arg(),
                q_scale.arg(),
                state.write_state.arg(),
                state.write_state_mask.arg(),
                carry.arg(),
                ctx.stage(),
            ],
        );
    }
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::attn::ssm_gdn_step_carried<16>")).apply(Launch::grid(
            [v_dim / CARRIED_BV, n, v_heads],
            [BLOCK, 1, 1],
        )),
        &[
            qkv.arg(),
            gates.arg(),
            gated.ba.arg(),
            gated.dt_bias.arg(),
            gated.a_log.arg(),
            rows.arg(),
            y.arg(),
            stated(OP, k_heads)?.arg(),
            stated(OP, v_heads)?.arg(),
            stated(OP, k_dim)?.arg(),
            stated(OP, v_dim)?.arg(),
            stated(OP, shape.conv_dim)?.arg(),
            q_scale.arg(),
            state.write_state.arg(),
            state.write_state_mask.arg(),
            carry.arg(),
            ctx.stage(),
        ],
    )
}

/// The rows a chunked gated-delta fire's lanes land in their buffer past
/// their commit, made ready for a later decode ([`gated_delta_carried`]):
/// each row's corrected value and keys into the lane's buffer (planes 2 and
/// 1; its `[b | a]`, plane 0, the gate prep lands), stepped in order from
/// the state the fire left. `qkv` and `gates` are the fire's own rows by
/// `qo_indptr`, or with `ext` its extended rows (row `RsLane::ext + j`).
#[allow(clippy::too_many_arguments)]
pub fn gated_delta_tail(
    ctx: &Ctx,
    qkv: Tensor,
    gates: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    state: &RecurrentPool,
    qo_indptr: Tensor,
    ext: bool,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    carry: &Carry,
) -> Result<(), Error> {
    const OP: &str = "ssm.gated_delta_chunked";
    if !carry.buffers() {
        return Ok(());
    }
    dtype_dispatch!(OP, qkv.dtype, { Bf16 => () });
    let lanes = qo_indptr.rows.saturating_sub(1);
    if lanes == 0 {
        return Ok(());
    }
    if !(k_dim <= BLOCK * 8 / CARRIED_BV && v_dim.is_multiple_of(CARRIED_BV)) {
        return Err(refuse(
            OP,
            format!(
                "a buffered row's corrected value steps in the carried tile only: key heads of \
                 {k_dim} and value heads of {v_dim} do not lie in it"
            ),
        ));
    }
    let rows = state.rows(OP, state.slab, state.slot_stride_elems, v_heads)?;
    let conv_dim = 2 * k_heads * k_dim + v_heads * v_dim;
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::attn::ssm_gdn_tail_corrected<16>")).apply(Launch::grid(
            [v_dim / CARRIED_BV, lanes, v_heads],
            [BLOCK, 1, 1],
        )),
        &[
            qkv.arg(),
            gates.arg(),
            dt_bias.arg(),
            a_log.arg(),
            rows.arg(),
            qo_indptr.arg(),
            ext.arg(),
            stated(OP, k_heads)?.arg(),
            stated(OP, v_heads)?.arg(),
            stated(OP, k_dim)?.arg(),
            stated(OP, v_dim)?.arg(),
            stated(OP, conv_dim)?.arg(),
            state.write_state.arg(),
            state.write_state_mask.arg(),
            stated(OP, lanes)?.arg(),
            carry.arg(),
            ctx.stage(),
        ],
    )
}

/// The tickets every fused step launch draws from (a key head a row).
const STEP_TICKETS: u32 = crate::jit::TICKETS;

/// A gated-delta layer's decode step whole, from the in-projections' rows:
/// `[qkv | z] = split(qkvz)`, the conv update of `qkv`, the gate prep of
/// `ba` and the recurrence, in one launch (`ssm_gdn_step_fused`), landing
/// the split's halves (`qkv`, `z`) and the recurrence's f32 row `y`.
/// `Ok(false)` where the shape is not the one it is built for (128-wide
/// heads, a conv of at most four taps, no dilation), which leaves the step
/// to its primitives.
#[allow(clippy::too_many_arguments)]
pub fn gdn_step_fused(
    ctx: &Ctx,
    qkvz: Tensor,
    conv_weight: Tensor,
    conv_state: &RecurrentPool,
    conv_width: u32,
    dilation: u32,
    ba: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    state: &RecurrentPool,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qkv: &mut Tensor,
    z: &mut Tensor,
    y: &mut Tensor,
) -> Result<bool, Error> {
    const OP: &str = "fused.gdn_step";
    let channels = 2 * k_heads * k_dim + v_heads * v_dim;
    let covered = k_dim == CHUNK_HEAD
        && v_dim == CHUNK_HEAD
        && k_heads > 0
        && v_heads.is_multiple_of(k_heads)
        && (2..=4).contains(&conv_width)
        && dilation == 1
        && qkvz.dtype == Dtype::Bf16
        && ba.dtype == Dtype::Bf16
        && dt_bias.dtype == Dtype::Bf16
        && a_log.dtype == Dtype::F32
        && y.dtype == Dtype::F32
        && qkvz.width == channels + v_heads * v_dim
        && qkv.width == channels
        && z.width == v_heads * v_dim
        && ba.width == 2 * v_heads
        && u64::from(conv_weight.rows) * u64::from(conv_weight.width)
            == u64::from(channels) * u64::from(conv_width)
        && state.write_state_mask.is_absent() == conv_state.write_state_mask.is_absent()
        && conv_state.commit_len.is_absent()
        && conv_state.begin_at.is_absent()
        && state.commit_len.is_absent()
        && state.begin_at.is_absent()
        && (conv_state.write_state || !conv_state.write_state_mask.is_absent())
        && (state.write_state || !state.write_state_mask.is_absent());
    if !covered {
        return Ok(false);
    }
    // The state cuts along its value heads, the conv window along its
    // channels.
    let rows = state.rows(OP, state.slab, state.slot_stride_elems, v_heads)?;
    let conv_rows = conv_state.rows(OP, conv_state.conv_slab, conv_state.conv_stride, channels)?;
    if !rows.keeps(8) {
        return Ok(false);
    }
    let n = nonzero(OP, "rows", qkvz.rows)?;
    if u64::from(n) * u64::from(k_heads) > u64::from(STEP_TICKETS) {
        return Ok(false);
    }
    let tickets = ctx.tickets(OP, STEP_TICKETS)?;
    #[allow(clippy::cast_precision_loss)]
    let q_scale = (k_dim as f32).sqrt().recip();
    ctx.fire(
        OP,
        Fire::at(kernel(
            FILE,
            "::pie::attn::ssm_gdn_step_fused<128, 128, 256, 4>",
        ))
        .apply(Launch::grid([v_heads, n, 1], [FUSED_BLOCK, 1, 1])),
        &[
            qkvz.arg(),
            conv_weight.arg(),
            conv_rows.arg(),
            conv_state.write_state.arg(),
            conv_state.write_state_mask.arg(),
            ba.arg(),
            a_log.arg(),
            dt_bias.arg(),
            rows.arg(),
            state.write_state.arg(),
            state.write_state_mask.arg(),
            qkv.arg(),
            z.arg(),
            y.arg(),
            ArgValue::Ptr(tickets),
            stated(OP, k_heads)?.arg(),
            stated(OP, v_heads)?.arg(),
            stated(OP, conv_width)?.arg(),
            q_scale.arg(),
            ctx.stage(),
        ],
    )?;
    Ok(true)
}

/// [`gdn_step_fused`] over each lane's carried rows then its own: the conv's
/// window reads the committed cells, then the carried rows' inputs (`conv`:
/// plane 0 the conv's input rows, plane 1 its output's keys) and is stored
/// at the lane's commit; the recurrence reads the carried rows as
/// [`gated_delta_carried`] does (`step`: `[b | a]`, keys, corrected values).
/// The own row's conv input, keys, `[b | a]` and corrected value land in its
/// buffer. `Ok(false)` where the shape is not the fused step's, which leaves
/// the step to its carried primitives.
#[allow(clippy::too_many_arguments)]
pub fn gdn_step_fused_carried(
    ctx: &Ctx,
    qkvz: Tensor,
    conv_weight: Tensor,
    conv_state: &RecurrentPool,
    conv_width: u32,
    dilation: u32,
    ba: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    state: &RecurrentPool,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    qkv: &mut Tensor,
    z: &mut Tensor,
    y: &mut Tensor,
    conv: &Carry,
    step: &Carry,
) -> Result<bool, Error> {
    const OP: &str = "fused.gdn_step";
    let channels = 2 * k_heads * k_dim + v_heads * v_dim;
    let covered = k_dim == CHUNK_HEAD
        && v_dim == CHUNK_HEAD
        && k_heads > 0
        && v_heads.is_multiple_of(k_heads)
        && (2..=4).contains(&conv_width)
        && dilation == 1
        && qkvz.dtype == Dtype::Bf16
        && ba.dtype == Dtype::Bf16
        && dt_bias.dtype == Dtype::Bf16
        && a_log.dtype == Dtype::F32
        && y.dtype == Dtype::F32
        && qkvz.width == channels + v_heads * v_dim
        && qkv.width == channels
        && z.width == v_heads * v_dim
        && ba.width == 2 * v_heads
        && u64::from(conv_weight.rows) * u64::from(conv_weight.width)
            == u64::from(channels) * u64::from(conv_width)
        && state.write_state_mask.is_absent() == conv_state.write_state_mask.is_absent()
        && conv_state.begin_at.is_absent()
        && state.begin_at.is_absent()
        && (conv_state.write_state || !conv_state.write_state_mask.is_absent())
        && (state.write_state || !state.write_state_mask.is_absent());
    if !covered {
        return Ok(false);
    }
    let rows = state.rows(OP, state.slab, state.slot_stride_elems, v_heads)?;
    let conv_rows = conv_state.rows(OP, conv_state.conv_slab, conv_state.conv_stride, channels)?;
    if !rows.keeps(8) {
        return Ok(false);
    }
    let n = nonzero(OP, "rows", qkvz.rows)?;
    if u64::from(n) * u64::from(k_heads) > u64::from(STEP_TICKETS) {
        return Ok(false);
    }
    let tickets = ctx.tickets(OP, STEP_TICKETS)?;
    #[allow(clippy::cast_precision_loss)]
    let q_scale = (k_dim as f32).sqrt().recip();
    ctx.fire(
        OP,
        Fire::at(kernel(
            FILE,
            "::pie::attn::ssm_gdn_step_fused_carried<128, 128, 256, 4>",
        ))
        .apply(Launch::grid([v_heads, n, 1], [FUSED_BLOCK, 1, 1])),
        &[
            qkvz.arg(),
            conv_weight.arg(),
            conv_rows.arg(),
            conv_state.write_state.arg(),
            conv_state.write_state_mask.arg(),
            ba.arg(),
            a_log.arg(),
            dt_bias.arg(),
            rows.arg(),
            state.write_state.arg(),
            state.write_state_mask.arg(),
            qkv.arg(),
            z.arg(),
            y.arg(),
            ArgValue::Ptr(tickets),
            stated(OP, k_heads)?.arg(),
            stated(OP, v_heads)?.arg(),
            stated(OP, conv_width)?.arg(),
            q_scale.arg(),
            conv.arg(),
            step.arg(),
            ctx.stage(),
        ],
    )?;
    Ok(true)
}

const FUSED_BLOCK: u32 = 256;

const FUSED_BV: u32 = 16;

fn fused_fits(shape: &Delta, rows: &StateRows) -> bool {
    const VEC: u32 = 8;
    shape.conv_dim.is_multiple_of(VEC)
        && (shape.k_heads * shape.k_dim).is_multiple_of(VEC)
        && shape.v_dim.is_multiple_of(FUSED_BV)
        && (shape.k_dim * shape.v_dim).is_multiple_of(VEC)
        && rows.keeps(i64::from(VEC))
        && shape.k_dim <= FUSED_BLOCK * VEC / FUSED_BV
}

/// From this many (row, head) pairs the KDA step's block a whole head fills
/// the device; under it each head's rows split over D / 32 blocks.
const ROWS_STEP_FROM: u32 = 256;

impl Delta {
    #[allow(clippy::too_many_arguments)]
    fn decode_rows(
        self,
        ctx: &Ctx,
        op: &'static str,
        qkv: Tensor,
        gates: Tensor,
        rows: &StateRows,
        state: &RecurrentPool,
        y: &mut Tensor,
    ) -> Result<(), Error> {
        #[allow(clippy::cast_precision_loss)]
        let q_scale = (self.k_dim as f32).sqrt().recip();
        ctx.fire(
            op,
            Fire::at(kernel(
                FILE,
                "::pie::attn::ssm_gdn_decode_rows<128, 128, 256>",
            ))
            .apply(Launch::grid([self.v_heads, self.n, 1], [FUSED_BLOCK, 1, 1])),
            &[
                qkv.arg(),
                gates.arg(),
                rows.arg(),
                y.arg(),
                stated(op, self.k_heads)?.arg(),
                stated(op, self.v_heads)?.arg(),
                stated(op, self.conv_dim)?.arg(),
                q_scale.arg(),
                state.write_state.arg(),
                state.write_state_mask.arg(),
                ctx.stage(),
            ],
        )
    }

    #[allow(clippy::too_many_arguments)]
    fn decode_fused(
        self,
        ctx: &Ctx,
        op: &'static str,
        qkv: Tensor,
        gates: Tensor,
        rows: &StateRows,
        state: &RecurrentPool,
        y: &mut Tensor,
    ) -> Result<(), Error> {
        #[allow(clippy::cast_precision_loss)]
        let q_scale = (self.k_dim as f32).sqrt().recip();
        ctx.fire(
            op,
            Fire::at(kernel(FILE, "::pie::attn::ssm_gdn_decode_step<16>")).apply(Launch::grid(
                [self.v_dim / FUSED_BV, self.n, self.v_heads],
                [FUSED_BLOCK, 1, 1],
            )),
            &[
                qkv.arg(),
                gates.arg(),
                rows.arg(),
                y.arg(),
                stated(op, self.k_heads)?.arg(),
                stated(op, self.v_heads)?.arg(),
                stated(op, self.k_dim)?.arg(),
                stated(op, self.v_dim)?.arg(),
                stated(op, self.conv_dim)?.arg(),
                q_scale.arg(),
                state.write_state.arg(),
                state.write_state_mask.arg(),
                ctx.stage(),
            ],
        )
    }
}

#[allow(clippy::too_many_arguments)]
pub fn gated_delta_chunked_serial(
    ctx: &Ctx,
    qkv: RaggedTensor,
    z: Tensor,
    gates: Tensor,
    state: &RecurrentPool,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    SERIAL_SCAN.with(|flag| flag.set(true));
    let out = gated_delta_chunked(ctx, qkv, z, gates, state, k_heads, v_heads, k_dim, v_dim, y);
    SERIAL_SCAN.with(|flag| flag.set(false));
    out
}

thread_local! {
    static SERIAL_SCAN: core::cell::Cell<bool> = const { core::cell::Cell::new(false) };
}

#[allow(clippy::too_many_arguments)]
pub fn gated_delta_chunked(
    ctx: &Ctx,
    qkv: RaggedTensor,
    z: Tensor,
    gates: Tensor,
    state: &RecurrentPool,
    k_heads: u32,
    v_heads: u32,
    k_dim: u32,
    v_dim: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "ssm.gated_delta_chunked";

    /// State columns a scan block owns.
    const SCAN_BV: u32 = 128;

    let _ = z;
    dtype_dispatch!(OP, qkv.data.dtype, { Bf16 => () });
    debug_assert_eq!(gates.dtype, Dtype::F32, "`{OP}` reads an f32 decay row");
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Delta::of(OP, qkv.data, gates, y, k_heads, v_heads, k_dim, v_dim)?;
    let lanes = requests(OP, qkv)?;
    if k_dim == CHUNK_HEAD && v_dim == CHUNK_HEAD && !SERIAL_SCAN.with(core::cell::Cell::get) {
        return chunk_parallel(ctx, OP, &shape, qkv, gates, state, y);
    }
    let staged = shape.stage(ctx, OP, qkv.data, gates, StagePlanes::Chunk)?;
    // The recurrence, token by token per request: a thread per state
    // column, its key-head entries in registers.
    let bk = match k_dim {
        0..=128 => 128,
        129..=256 => 256,
        _ => {
            return Err(refuse(
                OP,
                format!("the scan holds a key head of at most 256 and this one is {k_dim}"),
            ));
        }
    };
    seated(OP, state, "ssm_gdn_scan", true, true, true)?;
    let rows = state.rows(OP, state.slab, state.slot_stride_elems, v_heads)?;
    let bv = v_dim.min(SCAN_BV);
    let [snap, snap_rows, snap_slots] = state.snapshots.args();
    ctx.fire(
        OP,
        Fire::at(kernel(
            FILE,
            &format!("::pie::attn::ssm_gdn_scan<{bk}, {bv}>"),
        ))
        .apply(Launch::grid(
            [v_dim.div_ceil(bv), lanes, v_heads],
            [bv, 1, 1],
        )),
        &[
            ArgValue::Ptr(staged.q_norm),
            ArgValue::Ptr(staged.k_norm),
            ArgValue::Ptr(staged.v),
            ArgValue::Ptr(staged.g_log),
            ArgValue::Ptr(staged.beta),
            rows.arg(),
            qkv.indptr.arg(),
            y.arg(),
            stated(OP, k_heads)?.arg(),
            stated(OP, v_heads)?.arg(),
            stated(OP, k_dim)?.arg(),
            stated(OP, v_dim)?.arg(),
            state.write_state.arg(),
            state.commit_len.arg(),
            state.write_state_mask.arg(),
            state.begin_at.arg(),
            state.fused_decay.arg(),
            snap,
            snap_rows,
            snap_slots,
            ctx.stage(),
        ],
    )
}

/// The head width the chunk-parallel passes are built for.
const CHUNK_HEAD: u32 = 128;

/// Tokens per chunk.
const CHUNK: u32 = 64;

/// Value columns per state-walking block.
const CHUNK_SLICE: u32 = 16;

/// The recurrence as three chunk-parallel passes (see `ssm_gdn_chunk_*`).
fn chunk_parallel(
    ctx: &Ctx,
    op: &'static str,
    shape: &Delta,
    qkv: RaggedTensor,
    gates: Tensor,
    state: &RecurrentPool,
    y: &mut Tensor,
) -> Result<(), Error> {
    let lanes = requests(op, qkv)?;
    seated(op, state, "ssm_gdn_chunk", true, true, true)?;
    let rows = state.rows(op, state.slab, state.slot_stride_elems, shape.v_heads)?;
    let Delta {
        k_dim,
        v_dim,
        v_heads,
        k_heads,
        ..
    } = *shape;
    // A request's chunks start at ⌊t0 / CH⌋ + r, so `n / CH + lanes` slots
    // cover them all; the row planes carry a chunk of tail for the tiles
    // that read past a request's last row.
    let slots = shape.n.div_ceil(CHUNK) + lanes + state.snapshots.count;
    let padded = Delta {
        n: shape.n + CHUNK,
        ..*shape
    };
    let per_slot = |heads: u32, width: u32| u64::from(slots) * u64::from(heads) * u64::from(width);
    let gcum = plane(ctx, op, "attn.ssm_gdn_chunk_gcum", shape.elems(v_heads, 1))?;
    let qb = plane_bf16(
        ctx,
        op,
        "attn.ssm_gdn_chunk_qb",
        padded.elems(k_heads, k_dim),
    )?;
    let kb = plane_bf16(
        ctx,
        op,
        "attn.ssm_gdn_chunk_kb",
        padded.elems(k_heads, k_dim),
    )?;
    let kt = plane_bf16(
        ctx,
        op,
        "attn.ssm_gdn_chunk_kt",
        per_slot(k_heads, k_dim * CHUNK),
    )?;
    let w = plane_bf16(
        ctx,
        op,
        "attn.ssm_gdn_chunk_w",
        padded.elems(v_heads, k_dim),
    )?;
    let u = plane_bf16(
        ctx,
        op,
        "attn.ssm_gdn_chunk_ut",
        per_slot(v_heads, v_dim * CHUNK),
    )?;
    let vnt = plane_bf16(
        ctx,
        op,
        "attn.ssm_gdn_chunk_vnt",
        per_slot(v_heads, v_dim * CHUNK),
    )?;
    let hct = plane_bf16(
        ctx,
        op,
        "attn.ssm_gdn_chunk_hct",
        per_slot(v_heads, v_dim * k_dim),
    )?;
    #[allow(clippy::cast_precision_loss)]
    let q_scale = (k_dim as f32).sqrt().recip();
    let snap = state.snapshots.args();
    let common = |args: &mut Vec<ArgValue>| -> Result<(), Error> {
        args.push(stated(op, lanes)?.arg());
        args.push(stated(op, k_heads)?.arg());
        args.push(stated(op, v_heads)?.arg());
        args.extend(snap);
        args.push(ctx.stage());
        Ok(())
    };

    let args = vec![
        qkv.data.arg(),
        gates.arg(),
        rows.arg(),
        qkv.indptr.arg(),
        state.commit_len.arg(),
        state.begin_at.arg(),
        ArgValue::Ptr(gcum),
        ArgValue::Ptr(qb),
        ArgValue::Ptr(kb),
        ArgValue::Ptr(kt),
        ArgValue::Ptr(w),
        ArgValue::Ptr(u),
        stated(op, lanes)?.arg(),
        stated(op, k_heads)?.arg(),
        stated(op, v_heads)?.arg(),
        stated(op, shape.conv_dim)?.arg(),
        q_scale.arg(),
        snap[0],
        snap[1],
        ctx.stage(),
    ];
    ctx.fire(
        op,
        Fire::at(kernel(FILE, "::pie::attn::ssm_gdn_chunk_wu<128, 128>"))
            .apply(Launch::grid([slots, v_heads, 1], [2 * BLOCK, 1, 1])),
        &args,
    )?;

    let mut args = vec![
        ArgValue::Ptr(gcum),
        ArgValue::Ptr(kt),
        ArgValue::Ptr(w),
        ArgValue::Ptr(u),
        ArgValue::Ptr(vnt),
        ArgValue::Ptr(hct),
        rows.arg(),
        qkv.indptr.arg(),
        state.commit_len.arg(),
        state.begin_at.arg(),
        state.write_state.arg(),
        state.write_state_mask.arg(),
    ];
    common(&mut args)?;
    let from = ctx.tuning().ssm_gdn_wide_from;
    let slices = lanes * v_heads * (v_dim / CHUNK_SLICE);
    let bvt = if from > 0 && slices >= from {
        2 * CHUNK_SLICE
    } else {
        CHUNK_SLICE
    };
    ctx.fire(
        op,
        Fire::at(kernel(
            FILE,
            &format!("::pie::attn::ssm_gdn_chunk_h<128, 128, {bvt}>"),
        ))
        .apply(Launch::grid(
            [lanes * v_heads * (v_dim / bvt), 1, 1],
            [BLOCK, 1, 1],
        )),
        &args,
    )?;

    let mut args = vec![
        ArgValue::Ptr(gcum),
        ArgValue::Ptr(qb),
        ArgValue::Ptr(kb),
        ArgValue::Ptr(vnt),
        ArgValue::Ptr(hct),
        rows.arg(),
        qkv.indptr.arg(),
        state.commit_len.arg(),
        state.begin_at.arg(),
        y.arg(),
    ];
    common(&mut args)?;
    ctx.fire(
        op,
        Fire::at(kernel(FILE, "::pie::attn::ssm_gdn_chunk_o<128, 128>"))
            .apply(Launch::grid([slots, v_heads, 1], [BLOCK, 1, 1])),
        &args,
    )
}

fn plane_bf16(ctx: &Ctx, op: &'static str, name: &'static str, elems: u64) -> Result<u64, Error> {
    let bytes = usize::try_from(elems.saturating_mul(2))
        .map_err(|_| refuse(op, format!("{elems} bf16 elements do not fit this host")))?;
    Ok(ctx.scratch_shared(op, name, bytes)? as u64)
}

#[derive(Clone, Copy)]
struct Kda {
    n: u32,
    heads: u32,
    head_dim: u32,
    width: u32,
}

#[derive(Clone, Copy)]
struct KdaStaged {
    q_norm: u64,
    k_norm: u64,
    v: u64,
    gate: u64,
    beta: u64,
}

impl Kda {
    fn of(
        op: &'static str,
        mixed: Tensor,
        f: Tensor,
        b: Tensor,
        y: &Tensor,
        heads: u32,
        head_dim: u32,
    ) -> Result<Self, Error> {
        nonzero(op, "the KDA heads this statement states", heads)?;
        nonzero(op, "the KDA head width this statement states", head_dim)?;
        let width = heads.checked_mul(head_dim).ok_or_else(|| {
            refuse(
                op,
                format!("the KDA plane will not size: {heads} heads x {head_dim} wide"),
            )
        })?;
        debug_assert_eq!(
            u64::from(mixed.width),
            3 * u64::from(width),
            "the post-convolution `[q | k | v]` row is three head planes"
        );
        debug_assert!(
            f.rows == mixed.rows && f.width == width,
            "the forget projection's row is one head plane"
        );
        debug_assert!(
            b.rows == mixed.rows && b.width == heads,
            "the beta projection's row is one entry per head"
        );
        debug_assert!(
            y.rows == mixed.rows && y.width == width,
            "the recurrence lands one head plane per row"
        );
        nonzero(op, "rows", mixed.rows)?;
        Ok(Self {
            n: mixed.rows,
            heads,
            head_dim,
            width,
        })
    }

    #[allow(clippy::too_many_arguments)]
    fn stage(
        self,
        ctx: &Ctx,
        op: &'static str,
        mixed: Tensor,
        f: Tensor,
        b: Tensor,
        dt_bias: Tensor,
        a_log: Tensor,
        norm_eps: f32,
        gate_floor: f32,
    ) -> Result<KdaStaged, Error> {
        const PLANES: u32 = 3;
        /// Heads a staging wavefront norms.
        const PREP_HEADS: u32 = 4;

        let wide = u64::from(self.n) * u64::from(self.width);
        let decay = u64::from(self.n) * u64::from(self.heads);

        let qkv = plane(ctx, op, "attn.ssm_kda_qkv", 3 * wide)?;
        let gb = plane(ctx, op, "attn.ssm_kda_gates", wide + decay)?;
        let staged = KdaStaged {
            q_norm: qkv,
            k_norm: qkv + wide * u64::from(FLOAT),
            v: qkv + 2 * wide * u64::from(FLOAT),
            gate: gb,
            beta: gb + wide * u64::from(FLOAT),
        };

        ctx.fire(
            op,
            Fire::at(kernel(FILE, "::pie::attn::ssm_kda_qkv_prep<128>")).apply(Launch::grid(
                [
                    self.n,
                    PLANES,
                    self.heads.div_ceil(PREP_HEADS * (PREP_BLOCK / WAVE)),
                ],
                [PREP_BLOCK, 1, 1],
            )),
            &[
                mixed.arg(),
                ArgValue::Ptr(staged.q_norm),
                ArgValue::Ptr(staged.k_norm),
                ArgValue::Ptr(staged.v),
                stated(op, self.width)?.arg(),
                stated(op, self.head_dim)?.arg(),
                norm_eps.arg(),
                ctx.stage(),
            ],
        )?;
        ctx.fire(
            op,
            Fire::at(kernel(FILE, "::pie::attn::ssm_kda_gate_beta")).apply(Launch::grid(
                [self.n, self.width.div_ceil(4 * PREP_BLOCK), 1],
                [PREP_BLOCK, 1, 1],
            )),
            &[
                f.arg(),
                b.arg(),
                a_log.arg(),
                dt_bias.arg(),
                ArgValue::Ptr(staged.gate),
                ArgValue::Ptr(staged.beta),
                stated(op, self.n)?.arg(),
                stated(op, self.heads)?.arg(),
                stated(op, self.head_dim)?.arg(),
                gate_floor.arg(), // the decay's lower bound; zero leaves it unbounded
                ctx.stage(),
            ],
        )?;
        Ok(staged)
    }
}

fn f32_state(op: &'static str, state: &RecurrentPool) -> Result<(), Error> {
    if state.slab.dtype != Dtype::F32 {
        return Err(refuse(
            op,
            format!(
                "the KDA recurrence keeps an f32 state and this cache row is declared {:?}",
                state.slab.dtype
            ),
        ));
    }
    Ok(())
}

/// The token kernels hold a KDA state row on eight lanes in whole 32-wide
/// pieces, up to this wide.
const KDA_MAX_HEAD: u32 = 128;

/// Value rows a decode block walks when the batch is too small for a block
/// per row and head.
const KDA_TILE_ROWS: u32 = 32;

fn kda_head(op: &'static str, head_dim: u32) -> Result<(), Error> {
    if !head_dim.is_multiple_of(32) || head_dim > KDA_MAX_HEAD {
        return Err(refuse(
            op,
            format!(
                "the KDA recurrence holds a head of whole 32-wide pieces, at most \
                 {KDA_MAX_HEAD}, and this one is {head_dim}"
            ),
        ));
    }
    Ok(())
}

#[allow(clippy::too_many_arguments)]
pub fn kda(
    ctx: &Ctx,
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
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "ssm.kda";

    const STEP_BLOCK: u32 = 256;

    dtype_dispatch!(OP, mixed.dtype, { Bf16 => () });
    debug_assert_eq!(dt_bias.dtype, Dtype::F32, "`{OP}` reads an f32 decay bias");
    debug_assert_eq!(a_log.dtype, Dtype::F32, "`{OP}` reads an f32 decay bank");
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Kda::of(OP, mixed, f, b, y, heads, head_dim)?;
    f32_state(OP, state)?;
    kda_head(OP, head_dim)?;
    seated(OP, state, "ssm_kda_step_batched", true, false, false)?;
    // The state cuts along its heads.
    let state_rows = state.rows(OP, state.slab, state.slot_stride_elems, shape.heads)?;
    // A block a (row, head) once the batch fills the device; under that,
    // each head's rows split over D / 32 blocks.
    let rows = if shape.n.saturating_mul(shape.heads) >= ROWS_STEP_FROM {
        head_dim
    } else {
        KDA_TILE_ROWS
    };
    ctx.fire(
        OP,
        Fire::at(kernel(
            FILE,
            &format!("::pie::attn::ssm_kda_step_batched<{head_dim}, {rows}>"),
        ))
        .apply(Launch::grid(
            [shape.heads, shape.n, head_dim / rows],
            [STEP_BLOCK, 1, 1],
        )),
        &[
            mixed.arg(),
            f.arg(),
            b.arg(),
            a_log.arg(),
            dt_bias.arg(),
            state_rows.arg(),
            y.arg(),
            stated(OP, shape.heads)?.arg(),
            norm_eps.arg(),
            gate_floor.arg(),
            state.write_state.arg(),
            state.write_state_mask.arg(),
            ctx.stage(),
        ],
    )
}

/// [`kda`] over each lane's carried rows then its own ([`Carry`]): each
/// head's state rows, held in registers, step the carried rows (their
/// `[q | k | v]` from the carried scratch, their forget and beta rows from
/// their buffered rows, planes 0 and 1) then the own row, and are stored at
/// the lane's commit length; the own row's forget and beta rows land in the
/// lane's buffer (`ssm_kda_step_carried`).
#[allow(clippy::too_many_arguments)]
pub fn kda_carried(
    ctx: &Ctx,
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
    y: &mut Tensor,
    carry: &Carry,
) -> Result<(), Error> {
    const OP: &str = "ssm.kda";

    const STEP_BLOCK: u32 = 256;

    dtype_dispatch!(OP, mixed.dtype, { Bf16 => () });
    debug_assert_eq!(dt_bias.dtype, Dtype::F32, "`{OP}` reads an f32 decay bias");
    debug_assert_eq!(a_log.dtype, Dtype::F32, "`{OP}` reads an f32 decay bank");
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Kda::of(OP, mixed, f, b, y, heads, head_dim)?;
    f32_state(OP, state)?;
    kda_head(OP, head_dim)?;
    seated(OP, state, "ssm_kda_step_carried", true, false, false)?;
    let state_rows = state.rows(OP, state.slab, state.slot_stride_elems, shape.heads)?;
    // As the step: a block a (row, head) once the batch fills the device.
    let rows = if shape.n.saturating_mul(shape.heads) >= ROWS_STEP_FROM {
        head_dim
    } else {
        KDA_TILE_ROWS
    };
    ctx.fire(
        OP,
        Fire::at(kernel(
            FILE,
            &format!("::pie::attn::ssm_kda_step_carried<{head_dim}, {rows}>"),
        ))
        .apply(Launch::grid(
            [shape.heads, shape.n, head_dim / rows],
            [STEP_BLOCK, 1, 1],
        )),
        &[
            mixed.arg(),
            f.arg(),
            b.arg(),
            a_log.arg(),
            dt_bias.arg(),
            state_rows.arg(),
            y.arg(),
            stated(OP, shape.heads)?.arg(),
            norm_eps.arg(),
            gate_floor.arg(),
            state.write_state.arg(),
            state.write_state_mask.arg(),
            carry.arg(),
            ctx.stage(),
        ],
    )
}

/// The rows a chunked KDA fire's lanes land in their buffer past their
/// commit, made ready for a later decode ([`kda_carried`]): each row's keys
/// and corrected value (planes 2 and 3; its forget and beta rows, planes 0
/// and 1, the fire lands), stepped in order from the state the fire left.
/// The own rows are the fire's by `qo_indptr`, or with `ext` its extended
/// rows (row `RsLane::ext + j`) (`ssm_kda_tail_corrected`).
#[allow(clippy::too_many_arguments)]
pub fn kda_tail(
    ctx: &Ctx,
    mixed: Tensor,
    f: Tensor,
    b: Tensor,
    dt_bias: Tensor,
    a_log: Tensor,
    state: &RecurrentPool,
    qo_indptr: Tensor,
    ext: bool,
    heads: u32,
    head_dim: u32,
    norm_eps: f32,
    gate_floor: f32,
    carry: &Carry,
) -> Result<(), Error> {
    const OP: &str = "ssm.kda_chunked";
    const STEP_BLOCK: u32 = 256;
    if !carry.buffers() {
        return Ok(());
    }
    dtype_dispatch!(OP, mixed.dtype, { Bf16 => () });
    let lanes = qo_indptr.rows.saturating_sub(1);
    if lanes == 0 {
        return Ok(());
    }
    f32_state(OP, state)?;
    kda_head(OP, head_dim)?;
    let state_rows = state.rows(OP, state.slab, state.slot_stride_elems, heads)?;
    let rows = KDA_TILE_ROWS;
    ctx.fire(
        OP,
        Fire::at(kernel(
            FILE,
            &format!("::pie::attn::ssm_kda_tail_corrected<{head_dim}, {rows}>"),
        ))
        .apply(Launch::grid([heads, lanes, head_dim / rows], [STEP_BLOCK, 1, 1])),
        &[
            mixed.arg(),
            f.arg(),
            b.arg(),
            a_log.arg(),
            dt_bias.arg(),
            state_rows.arg(),
            qo_indptr.arg(),
            ext.arg(),
            stated(OP, heads)?.arg(),
            norm_eps.arg(),
            gate_floor.arg(),
            state.write_state.arg(),
            state.write_state_mask.arg(),
            stated(OP, lanes)?.arg(),
            carry.arg(),
            ctx.stage(),
        ],
    )
}

/// Folds each group's rows into its KDA state (§11.3), from the kept planes
/// (`carry`: forget rows, beta, keys, corrected values) (`ssm_kda_fold`).
#[allow(clippy::too_many_arguments)]
pub fn kda_fold(
    ctx: &Ctx,
    state: &RecurrentPool,
    heads: u32,
    head_dim: u32,
    norm_eps: f32,
    gate_floor: f32,
    dt_bias: Tensor,
    a_log: Tensor,
    carry: &Carry,
    folds: Tensor,
    groups: u32,
) -> Result<(), Error> {
    const OP: &str = "ssm.kda";
    const STEP_BLOCK: u32 = 256;
    if groups == 0 {
        return Ok(());
    }
    f32_state(OP, state)?;
    kda_head(OP, head_dim)?;
    let state_rows = state.rows(OP, state.slab, state.slot_stride_elems, heads)?;
    let rows = KDA_TILE_ROWS;
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, &format!("::pie::attn::ssm_kda_fold<{head_dim}, {rows}>")))
            .apply(Launch::grid([heads, head_dim / rows, 1], [STEP_BLOCK, 1, 1])),
        &[
            state_rows.arg(),
            a_log.arg(),
            dt_bias.arg(),
            stated(OP, heads)?.arg(),
            norm_eps.arg(),
            gate_floor.arg(),
            carry.arg(),
            folds.arg(),
            stated(OP, groups)?.arg(),
            ctx.stage(),
        ],
    )
}

/// `kda_chunked` through the token-by-token scan whatever the head: the
/// reference the chunk-parallel passes answer to.
#[allow(clippy::too_many_arguments)]
pub fn kda_chunked_serial(
    ctx: &Ctx,
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
    y: &mut Tensor,
) -> Result<(), Error> {
    SERIAL_SCAN.with(|flag| flag.set(true));
    let out = kda_chunked(
        ctx, mixed, f, b, dt_bias, a_log, state, heads, head_dim, norm_eps, gate_floor, y,
    );
    SERIAL_SCAN.with(|flag| flag.set(false));
    out
}

#[allow(clippy::too_many_arguments)]
pub fn kda_chunked(
    ctx: &Ctx,
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
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "ssm.kda_chunked";

    dtype_dispatch!(OP, mixed.data.dtype, { Bf16 => () });
    debug_assert_eq!(dt_bias.dtype, Dtype::F32, "`{OP}` reads an f32 decay bias");
    debug_assert_eq!(a_log.dtype, Dtype::F32, "`{OP}` reads an f32 decay bank");
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` lands an f32 accumulator");
    let shape = Kda::of(OP, mixed.data, f, b, y, heads, head_dim)?;
    let lanes = requests(OP, mixed)?;
    f32_state(OP, state)?;
    kda_head(OP, head_dim)?;
    seated(OP, state, "ssm_kda_chunked_batched", true, true, true)?;
    let rows = state.rows(OP, state.slab, state.slot_stride_elems, shape.heads)?;
    if head_dim == CHUNK_HEAD && !SERIAL_SCAN.with(core::cell::Cell::get) {
        return kda_chunk_parallel(
            ctx,
            OP,
            &shape,
            mixed,
            [f, b, dt_bias, a_log],
            state,
            &rows,
            lanes,
            [norm_eps, gate_floor],
            y,
        );
    }
    let staged = shape.stage(
        ctx, OP, mixed.data, f, b, dt_bias, a_log, norm_eps, gate_floor,
    )?;
    let [snap, snap_rows, snap_slots] = state.snapshots.args();
    // A request a block, a value row to eight lanes.
    ctx.fire(
        OP,
        Fire::at(kernel(
            FILE,
            &format!("::pie::attn::ssm_kda_chunked_batched<{head_dim}>"),
        ))
        .apply(Launch::grid([lanes, shape.heads, 1], [8 * head_dim, 1, 1])),
        &[
            ArgValue::Ptr(staged.q_norm),
            ArgValue::Ptr(staged.k_norm),
            ArgValue::Ptr(staged.v),
            ArgValue::Ptr(staged.gate),
            ArgValue::Ptr(staged.beta),
            rows.arg(),
            mixed.indptr.arg(),
            y.arg(),
            stated(OP, shape.heads)?.arg(),
            stated(OP, shape.head_dim)?.arg(),
            state.write_state.arg(),
            state.commit_len.arg(),
            state.write_state_mask.arg(),
            state.begin_at.arg(),
            snap,
            snap_rows,
            snap_slots,
            ctx.stage(),
        ],
    )
}

/// The recurrence as two chunk-parallel passes (`ssm_kda_chunk_wu`, then
/// `ssm_kda_chunk_ho`),
/// over the chunk slots the gated delta rule's passes use.
#[allow(clippy::too_many_arguments)]
fn kda_chunk_parallel(
    ctx: &Ctx,
    op: &'static str,
    shape: &Kda,
    mixed: RaggedTensor,
    [f, b, dt_bias, a_log]: [Tensor; 4],
    state: &RecurrentPool,
    rows: &StateRows,
    lanes: u32,
    [norm_eps, gate_floor]: [f32; 2],
    y: &mut Tensor,
) -> Result<(), Error> {
    let (d, heads) = (shape.head_dim, shape.heads);
    let slots = shape.n.div_ceil(CHUNK) + lanes + state.snapshots.count;
    let per_slot = |width: u32| u64::from(slots) * u64::from(heads) * u64::from(width);
    let kt = plane_bf16(ctx, op, "attn.ssm_kda_chunk_kt", per_slot(d * CHUNK))?;
    let w = plane_bf16(ctx, op, "attn.ssm_kda_chunk_w", per_slot(CHUNK * d))?;
    let ut = plane_bf16(ctx, op, "attn.ssm_kda_chunk_ut", per_slot(d * CHUNK))?;
    let qg = plane_bf16(ctx, op, "attn.ssm_kda_chunk_qg", per_slot(CHUNK * d))?;
    let p = plane_bf16(ctx, op, "attn.ssm_kda_chunk_p", per_slot(CHUNK * CHUNK))?;
    let gl = plane(ctx, op, "attn.ssm_kda_chunk_gl", per_slot(d))?;
    let snap = state.snapshots.args();
    let tail = |args: &mut Vec<ArgValue>| -> Result<(), Error> {
        args.push(stated(op, lanes)?.arg());
        args.push(stated(op, heads)?.arg());
        args.extend(snap);
        args.push(ctx.stage());
        Ok(())
    };

    let args = vec![
        mixed.data.arg(),
        f.arg(),
        b.arg(),
        a_log.arg(),
        dt_bias.arg(),
        rows.arg(),
        mixed.indptr.arg(),
        state.commit_len.arg(),
        state.begin_at.arg(),
        ArgValue::Ptr(kt),
        ArgValue::Ptr(w),
        ArgValue::Ptr(ut),
        ArgValue::Ptr(qg),
        ArgValue::Ptr(p),
        ArgValue::Ptr(gl),
        stated(op, lanes)?.arg(),
        stated(op, heads)?.arg(),
        norm_eps.arg(),
        gate_floor.arg(),
        snap[0],
        snap[1],
        ctx.stage(),
    ];
    ctx.fire(
        op,
        Fire::at(kernel(FILE, "::pie::attn::ssm_kda_chunk_wu<128>"))
            .apply(Launch::grid([slots, heads, 1], [2 * BLOCK, 1, 1])),
        &args,
    )?;

    // The state walk lands each chunk's output on the way: its entering
    // state and V' stay in the walk's LDS.
    let mut args = vec![
        ArgValue::Ptr(kt),
        ArgValue::Ptr(w),
        ArgValue::Ptr(ut),
        ArgValue::Ptr(gl),
        ArgValue::Ptr(qg),
        ArgValue::Ptr(p),
        y.arg(),
        rows.arg(),
        mixed.indptr.arg(),
        state.commit_len.arg(),
        state.begin_at.arg(),
        state.write_state.arg(),
        state.write_state_mask.arg(),
    ];
    tail(&mut args)?;
    ctx.fire(
        op,
        Fire::at(kernel(FILE, "::pie::attn::ssm_kda_chunk_ho<128>")).apply(Launch::grid(
            [lanes * heads * (d / CHUNK_SLICE), 1, 1],
            [BLOCK, 1, 1],
        )),
        &args,
    )
}

/// Mamba-2's per-head weights: `A_log`, `dt_bias` and `D` (f32 `[heads]`),
/// and the gated group norm's bf16 `[heads * 64]` weight and epsilon.
#[derive(Clone, Copy)]
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
    /// Refuses all but 64-wide heads over a 128-wide state, the width of
    /// `xbc` (`x`, then each group's `B`, then its `C`) and `dt` against
    /// the heads, and a group count that does not split them.
    fn check(
        &self,
        op: &'static str,
        xbc: Tensor,
        dt: Tensor,
        z: Tensor,
        y: &Tensor,
        state_dim: u32,
    ) -> Result<(), Error> {
        dtype_dispatch!(op, xbc.dtype, { Bf16 => () });
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
        for (what, t) in [("dt", dt.dtype), ("z", z.dtype), ("output", y.dtype)] {
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
    fn norm(&self, ctx: &Ctx, op: &'static str, y: &mut Tensor) -> Result<(), Error> {
        let group = y.width / self.groups;
        crate::elemwise::norm::rmsnorm_grouped(ctx, op, *y, self.norm, group, self.eps, false, y)
    }
}

/// One Mamba-2 step a request over the convolved `xbc` rows: lands
/// `(C S + D x) * silu(z)`, group-normed, in `y`.
#[allow(clippy::too_many_arguments)]
pub fn mamba2(
    ctx: &Ctx,
    xbc: Tensor,
    dt: Tensor,
    z: Tensor,
    m: Mamba2,
    state: &RecurrentPool,
    state_dim: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "ssm.mamba2";
    m.check(OP, xbc, dt, z, y, state_dim)?;
    f32_state(OP, state)?;
    seated(OP, state, "ssm_mamba2_step", true, false, false)?;
    let n = nonzero(OP, "rows", xbc.rows)?;
    // The state cuts along its heads.
    let rows = state.rows(OP, state.slab, state.slot_stride_elems, m.heads)?;
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::attn::ssm_mamba2_step"))
            .apply(Launch::grid([n, m.heads, 1], [BLOCK, 1, 1])),
        &[
            xbc.arg(),
            dt.arg(),
            z.arg(),
            m.a_log.arg(),
            m.dt_bias.arg(),
            m.d.arg(),
            rows.arg(),
            y.arg(),
            stated(OP, m.heads)?.arg(),
            stated(OP, m.groups)?.arg(),
            state.write_state.arg(),
            state.write_state_mask.arg(),
            ctx.stage(),
        ],
    )?;
    m.norm(ctx, OP, y)
}

/// [`mamba2`] over each lane's carried rows then its own ([`Carry`]): each
/// head's state steps the carried rows (their convolved `xbc` from the
/// carried scratch, their step sizes from their buffered rows, plane 0)
/// then the own row, and is stored at the lane's commit length; the own
/// row's step size lands in the lane's buffer (`ssm_mamba2_step_carried`).
#[allow(clippy::too_many_arguments)]
pub fn mamba2_carried(
    ctx: &Ctx,
    xbc: Tensor,
    dt: Tensor,
    z: Tensor,
    m: Mamba2,
    state: &RecurrentPool,
    state_dim: u32,
    y: &mut Tensor,
    carry: &Carry,
) -> Result<(), Error> {
    const OP: &str = "ssm.mamba2";
    m.check(OP, xbc, dt, z, y, state_dim)?;
    f32_state(OP, state)?;
    seated(OP, state, "ssm_mamba2_step_carried", true, false, false)?;
    let n = nonzero(OP, "rows", xbc.rows)?;
    let rows = state.rows(OP, state.slab, state.slot_stride_elems, m.heads)?;
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::attn::ssm_mamba2_step_carried"))
            .apply(Launch::grid([n, m.heads, 1], [BLOCK, 1, 1])),
        &[
            xbc.arg(),
            dt.arg(),
            z.arg(),
            m.a_log.arg(),
            m.dt_bias.arg(),
            m.d.arg(),
            rows.arg(),
            y.arg(),
            stated(OP, m.heads)?.arg(),
            stated(OP, m.groups)?.arg(),
            state.write_state.arg(),
            state.write_state_mask.arg(),
            carry.arg(),
            ctx.stage(),
        ],
    )?;
    m.norm(ctx, OP, y)
}

/// The rows a chunked Mamba-2 fire's lanes land in their buffer past their
/// commit, made ready for a later decode ([`mamba2_carried`]): each row's
/// convolved `x | B` (plane 1; its step size, plane 0, the fire lands). The
/// own rows are the fire's by `qo_indptr`, or with `ext` its extended rows
/// (`ssm_mamba2_tail_rows`).
#[allow(clippy::too_many_arguments)]
pub fn mamba2_tail(
    ctx: &Ctx,
    xbc: Tensor,
    m: Mamba2,
    state: &RecurrentPool,
    qo_indptr: Tensor,
    ext: bool,
    state_dim: u32,
    carry: &Carry,
) -> Result<(), Error> {
    const OP: &str = "ssm.mamba2_chunked";
    const TAIL_BLOCK: u32 = 256;
    if !carry.buffers() {
        return Ok(());
    }
    dtype_dispatch!(OP, xbc.dtype, { Bf16 => () });
    let lanes = qo_indptr.rows.saturating_sub(1);
    if lanes == 0 {
        return Ok(());
    }
    let inner = m.heads * (xbc.width.saturating_sub(2 * m.groups * state_dim) / m.heads.max(1));
    let xb = inner + m.groups * state_dim;
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::attn::ssm_mamba2_tail_rows"))
            .apply(Launch::grid([lanes, 1, 1], [TAIL_BLOCK, 1, 1])),
        &[
            xbc.arg(),
            qo_indptr.arg(),
            ext.arg(),
            stated(OP, xbc.width)?.arg(),
            stated(OP, xb)?.arg(),
            state.write_state.arg(),
            state.write_state_mask.arg(),
            stated(OP, lanes)?.arg(),
            carry.arg(),
            ctx.stage(),
        ],
    )
}

/// Folds each group's rows into its Mamba-2 state (§11.3), from the kept
/// planes (`carry`: step sizes, convolved `x | B`) (`ssm_mamba2_fold`).
pub fn mamba2_fold(
    ctx: &Ctx,
    state: &RecurrentPool,
    m: Mamba2,
    carry: &Carry,
    folds: Tensor,
    groups: u32,
) -> Result<(), Error> {
    const OP: &str = "ssm.mamba2";
    if groups == 0 {
        return Ok(());
    }
    f32_state(OP, state)?;
    let rows = state.rows(OP, state.slab, state.slot_stride_elems, m.heads)?;
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::attn::ssm_mamba2_fold"))
            .apply(Launch::grid([m.heads, 1, 1], [BLOCK, 1, 1])),
        &[
            rows.arg(),
            m.a_log.arg(),
            m.dt_bias.arg(),
            stated(OP, m.heads)?.arg(),
            stated(OP, m.groups)?.arg(),
            carry.arg(),
            folds.arg(),
            stated(OP, groups)?.arg(),
            ctx.stage(),
        ],
    )
}

/// A prompt's Mamba-2 walk, token by token: a block a (request, head) lands
/// `(C S + D x) * silu(z)` for each row, then the group norm. A `tail` pass
/// (rows past a lane's commit, from the state its first pass folded) walks
/// before the norm, which every row then takes once.
#[allow(clippy::too_many_arguments)]
pub fn mamba2_chunked(
    ctx: &Ctx,
    xbc: RaggedTensor,
    dt: Tensor,
    z: Tensor,
    m: Mamba2,
    state: &RecurrentPool,
    tail: Option<&RecurrentPool>,
    state_dim: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "ssm.mamba2_chunked";
    for pass in core::iter::once(state).chain(tail) {
        mamba2_scan(ctx, xbc, dt, z, m, pass, state_dim, y)?;
    }
    m.norm(ctx, OP, y)
}

/// One pass of [`mamba2_chunked`]'s walk, the norm not taken.
#[allow(clippy::too_many_arguments)]
fn mamba2_scan(
    ctx: &Ctx,
    xbc: RaggedTensor,
    dt: Tensor,
    z: Tensor,
    m: Mamba2,
    state: &RecurrentPool,
    state_dim: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "ssm.mamba2_chunked";
    m.check(OP, xbc.data, dt, z, y, state_dim)?;
    let lanes = requests(OP, xbc)?;
    f32_state(OP, state)?;
    seated(OP, state, "ssm_mamba2_scan", true, true, true)?;
    let rows = state.rows(OP, state.slab, state.slot_stride_elems, m.heads)?;
    let [snap, snap_rows, snap_slots] = state.snapshots.args();
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::attn::ssm_mamba2_scan"))
            .apply(Launch::grid([lanes, m.heads, 1], [BLOCK, 1, 1])),
        &[
            xbc.data.arg(),
            dt.arg(),
            z.arg(),
            m.a_log.arg(),
            m.dt_bias.arg(),
            m.d.arg(),
            rows.arg(),
            xbc.indptr.arg(),
            state.commit_len.arg(),
            state.begin_at.arg(),
            state.write_state.arg(),
            state.write_state_mask.arg(),
            snap,
            snap_rows,
            snap_slots,
            y.arg(),
            stated(OP, m.heads)?.arg(),
            stated(OP, m.groups)?.arg(),
            ctx.stage(),
        ],
    )
}
