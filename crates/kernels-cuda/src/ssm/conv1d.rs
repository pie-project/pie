use super::*;
use crate::jit::kernel;

const CONV1D: &str = "ssm/conv1d.cu";

const CONV_VEC: u32 = 8;

const CONV_K_MAX: u32 = 8;

const CONV_BLOCK: u32 = 64;

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
    seated(op, state, "conv1d_update", true, false, false)?;
    let rows = state.rows(op, state.conv_slab, state.conv_stride, channels)?;
    let vectors = channels % CONV_VEC == 0
        && conv_width <= CONV_K_MAX
        && dilation == 1
        && rows.keeps(i64::from(CONV_VEC))
        && rows.atoms().is_multiple_of(CONV_VEC)
        && aligned16(x.ptr)
        && aligned16(y.ptr)
        && aligned16(weight.ptr)
        && bias.is_none_or(|b| aligned16(b.ptr))
        && aligned16(state.conv_slab.ptr);
    let mut args = vec![
        x.arg(),
        weight.arg(),
        bias.map_or(ArgValue::ABSENT, |b| b.arg()),
        rows.arg(),
        y.arg(),
        stated(op, n)?.arg(),
        c.arg(),
        k.arg(),
    ];
    let (entrypoint, launch) = if vectors {
        (
            format!("::pie::ssm::conv1d_update_vec8{}", out.args()),
            Launch::grid(
                [(channels / CONV_VEC).div_ceil(CONV_BLOCK), n, 1],
                [CONV_BLOCK, 1, 1],
            )
            .programmatic(ctx.tuning().pdl()),
        )
    } else {
        args.push(dil.arg());
        (
            format!("::pie::ssm::conv1d_update{}", out.args()),
            Launch::grid([channels.div_ceil(BLOCK), n, 1], [BLOCK, 1, 1]),
        )
    };
    args.extend([
        state.write_state.arg(),
        state.write_state_mask.arg(),
        ctx.stage(),
    ]);
    ctx.fire(
        op,
        Fire::at(kernel(CONV1D, &entrypoint)).apply(launch),
        &args,
    )
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
    const CHANNEL_TILE_FROM: u32 = 8;

    const TILE_BLOCK: u32 = 128;
    const CONV_TT: u32 = 32;

    const PER_CHANNEL_BLOCK: u32 = 64;

    dtype_dispatch!(op, x.data.dtype, { Bf16 => () });
    let (channels, c, k) = conv_extents(op, x.data, y, conv_width)?;
    let dil = stated(op, nonzero(op, "the conv's dilation", dilation)?)?;
    let lanes = requests(op, x)?;
    seated(op, state, "conv1d_chunked", true, true, true)?;
    let rows = state.rows(op, state.conv_slab, state.conv_stride, channels)?;
    let reach = (conv_width.saturating_sub(1)).saturating_mul(dilation);
    let row_tiled = reach <= CONV_TT;
    let vec8 = conv_width == 4
        && dilation == 1
        && channels.is_multiple_of(8)
        && rows.keeps(8)
        && rows.atoms().is_multiple_of(8)
        && [x.data.ptr, y.ptr, weight.ptr, state.conv_slab.ptr]
            .iter()
            .all(|&at| aligned16(at));
    let (entrypoint, launch) = if vec8 && ctx.tuning().ssm_conv_vec4() {
        (
            format!("::pie::ssm::conv1d_chunked_row_tile_vec4{}", out.args()),
            Launch::grid(
                [
                    (channels / 4).div_ceil(TILE_BLOCK),
                    x.data.rows.div_ceil(CONV_TT).max(1),
                    1,
                ],
                [TILE_BLOCK, 1, 1],
            ),
        )
    } else if vec8 {
        (
            format!("::pie::ssm::conv1d_chunked_row_tile_vec8{}", out.args()),
            Launch::grid(
                [
                    (channels / 8).div_ceil(TILE_BLOCK),
                    x.data.rows.div_ceil(CONV_TT).max(1),
                    1,
                ],
                [TILE_BLOCK, 1, 1],
            ),
        )
    } else if row_tiled {
        (
            format!("::pie::ssm::conv1d_chunked_row_tile{}", out.args()),
            Launch::grid(
                [
                    channels.div_ceil(TILE_BLOCK),
                    x.data.rows.div_ceil(CONV_TT).max(1),
                    1,
                ],
                [TILE_BLOCK, 1, 1],
            ),
        )
    } else if lanes >= CHANNEL_TILE_FROM {
        (
            format!("::pie::ssm::conv1d_chunked_channel_tile{}", out.args()),
            Launch::grid(
                [channels.div_ceil(TILE_BLOCK), lanes, 1],
                [TILE_BLOCK, 1, 1],
            ),
        )
    } else {
        (
            format!("::pie::ssm::conv1d_chunked{}", out.args()),
            Launch::grid([channels, lanes, 1], [PER_CHANNEL_BLOCK, 1, 1]),
        )
    };
    ctx.fire(
        op,
        Fire::at(kernel(CONV1D, &entrypoint)).apply(launch),
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
    if !row_tiled || !state.write_state {
        if state.snapshots.count > 0 && !row_tiled {
            return Err(refuse(
                op,
                format!(
                    "this fire lands state snapshots and a conv reaching {reach} rows folds its \
                     window inline, with no seat for them"
                ),
            ));
        }
        return Ok(());
    }
    let [snap, snap_rows, snap_slots] = state.snapshots.args();
    ctx.fire(
        op,
        Fire::at(kernel(
            CONV1D,
            "::pie::ssm::conv1d_state_commit<::pie::bf16>",
        ))
        .apply(Launch::grid(
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

/// The causal conv's decode step over each lane's carried rows then its own
/// ([`Carry`]): `conv1d_update`'s sums a row, the own row's window from the
/// committed cells and at most `conv_width - 1` carried rows, the window
/// stored at the lane's commit length; the own row's input lands in the
/// lane's buffer (plane 0), and its output's channels `keys` (offset,
/// count: the channels its recurrence reads its buffered rows by) in plane
/// 1.
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
    if conv_width > CONV_K_MAX {
        return Err(refuse(
            op,
            format!("a {conv_width}-tap conv, past the {CONV_K_MAX} taps the step sums"),
        ));
    }
    seated(op, state, "conv1d_carried", true, false, false)?;
    let rows = state.rows(op, state.conv_slab, state.conv_stride, channels)?;
    let vectors = channels % CONV_VEC == 0
        && dilation == 1
        && conv_width == 4
        && rows.keeps(i64::from(CONV_VEC))
        && rows.atoms().is_multiple_of(CONV_VEC)
        && aligned16(x.ptr)
        && aligned16(y.ptr)
        && aligned16(weight.ptr)
        && bias.is_none_or(|b| aligned16(b.ptr))
        && aligned16(state.conv_slab.ptr)
        && carry.plane_off.iter().all(|off| off.is_multiple_of(16))
        && carry.plane_stride.iter().all(|stride| stride.is_multiple_of(16))
        && keys.is_none_or(|(off, len)| off % CONV_VEC == 0 && len % CONV_VEC == 0);
    let mut args = vec![
        x.arg(),
        weight.arg(),
        bias.map_or(ArgValue::ABSENT, |b| b.arg()),
        rows.arg(),
        y.arg(),
        stated(op, n)?.arg(),
        c.arg(),
        k.arg(),
    ];
    let (entrypoint, launch) = if vectors {
        (
            format!("::pie::ssm::conv1d_carried_vec8{}", out.args()),
            Launch::grid(
                [(channels / CONV_VEC).div_ceil(CONV_BLOCK), n, 1],
                [CONV_BLOCK, 1, 1],
            )
            .programmatic(ctx.tuning().pdl()),
        )
    } else {
        args.push(dil.arg());
        (
            format!("::pie::ssm::conv1d_carried{}", out.args()),
            Launch::grid([channels.div_ceil(BLOCK), n, 1], [BLOCK, 1, 1]),
        )
    };
    args.extend([
        state.write_state.arg(),
        state.write_state_mask.arg(),
        carry.arg(),
        stated(op, keys.map_or(0, |(off, _)| off))?.arg(),
        stated(op, keys.map_or(0, |(_, len)| len))?.arg(),
        ctx.stage(),
    ]);
    ctx.fire(
        op,
        Fire::at(kernel(CONV1D, &entrypoint)).apply(launch),
        &args,
    )
}

/// Folds each beam group's fold rows (plane 0 of
/// `carry`: the rows' conv inputs, through the group's representative lane's
/// pages) into its conv window: the window stored as it stands after them.
/// The groups that fold lead `folds` (`groups` entries at most, the first of
/// no rows ending them); a block folds each in turn, so a fire none of whose
/// groups folds costs one read a block.
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
    const OP: &str = "ssm.causal_conv1d group fold";
    const BLOCK: u32 = 256;
    if groups == 0 {
        return Ok(());
    }
    let span = conv_width
        .checked_sub(1)
        .and_then(|k| k.checked_mul(dilation))
        .and_then(|cells| cells.checked_add(1))
        .unwrap_or(u32::MAX);
    let channels = nonzero(OP, "the conv's channel count", channels)?;
    let rows = state.rows(OP, state.conv_slab, state.conv_stride, channels)?;
    ctx.fire(
        OP,
        Fire::at(fold_kernel())
            .apply(Launch::grid([channels.div_ceil(BLOCK), 1, 1], [BLOCK, 1, 1])),
        &[
            rows.arg(),
            stated(OP, channels)?.arg(),
            stated(OP, span)?.arg(),
            carry.arg(),
            folds.arg(),
            stated(OP, groups)?.arg(),
        ],
    )
}

/// The conv window's group fold kernel ([`conv1d_fold`]).
pub(super) fn fold_kernel() -> &'static crate::jit::Kernel {
    kernel(CONV1D, "::pie::ssm::conv1d_fold<::pie::bf16>")
}
