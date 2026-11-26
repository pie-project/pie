//! The causal depthwise conv with its rolling history (`ssm.causal_conv1d`,
//! its chunked and committed forms), from `kernels/ssm/conv1d.slang`.

use super::{Carry, CarryPlane, Committed, Csr, RowSpan, StatePool, rows};
use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;

const CONV1D: &str = "ssm/conv1d.slang";
const CONV1D_CARRIED: &str = "ssm/conv1d_carried.slang";

const GROUP: u32 = 128;

/// The longest history a lane keeps.
const HIST_MAX: u32 = 64;

/// The conv's stated shape, checked: its channel pairs (a lane's) and taps.
struct Shape {
    channels: i32,
    pairs: u32,
    taps: i32,
    dilation: i32,
}

impl Shape {
    fn of(
        op: &'static str,
        x: Tensor,
        weight: Tensor,
        state: &StatePool,
        conv_width: u32,
        dilation: u32,
        y: Tensor,
    ) -> Result<Self, Error> {
        let channels = nonzero(op, "the conv's channel count", x.width)?;
        if !channels.is_multiple_of(2) {
            return Err(refuse(
                op,
                format!("{channels} channels are odd, and this conv lands a bf16 pair a lane"),
            ));
        }
        if state.piece_atoms < channels && !state.piece_atoms.is_multiple_of(2) {
            return Err(refuse(
                op,
                format!(
                    "the history is cut {} channels a piece, which splits the channel pair a \
                     lane lands",
                    state.piece_atoms
                ),
            ));
        }
        nonzero(op, "the conv width this statement states", conv_width)?;
        nonzero(op, "the dilation this statement states", dilation)?;
        let hist = (conv_width - 1)
            .checked_mul(dilation)
            .and_then(|span| span.checked_add(1))
            .filter(|&hist| hist <= HIST_MAX)
            .ok_or_else(|| {
                refuse(
                    op,
                    format!(
                        "a width of {conv_width} at dilation {dilation} keeps more than the \
                         {HIST_MAX} rows a conv lane holds"
                    ),
                )
            })?;
        debug_assert_eq!(
            u64::from(state.bank.width),
            u64::from(hist) * u64::from(channels),
            "a slot holds `(width - 1) * dilation + 1` rows of channels"
        );
        debug_assert!(
            u64::from(weight.rows) * u64::from(weight.width)
                >= u64::from(channels) * u64::from(conv_width),
            "the weight is `[channels, conv_width]`"
        );
        debug_assert!(
            y.rows == x.rows && y.width == x.width,
            "the conv lands the row it convolves"
        );
        Ok(Self {
            channels: stated(op, channels)?,
            pairs: channels / 2,
            taps: stated(op, conv_width)?,
            dilation: stated(op, dilation)?,
        })
    }

    fn grid(&self, rows: u32) -> Grid {
        Grid::of([self.pairs, rows, 1], [GROUP, 1, 1])
    }
}

/// One decode step a row: each row convolved over its slot's history, which
/// then rolls by the row.
#[allow(clippy::too_many_arguments)]
pub fn conv1d(
    ctx: &Ctx<'_>,
    x: Tensor,
    weight: Tensor,
    state: &StatePool,
    conv_width: u32,
    dilation: u32,
    y: Tensor,
    span: RowSpan,
) -> Result<(), Error> {
    const OP: &str = "ssm.causal_conv1d";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "ssm_conv1d_bf16" });
    let shape = Shape::of(OP, x, weight, state, conv_width, dilation, y)?;
    let launched = span.rows(OP, &[x, y, state.slots])?;
    ctx.fire(
        Fire::at(CONV1D, entry).apply(shape.grid(launched)),
        &[
            &[x.arg(), weight.arg(), state.slots.arg(), y.arg_mut()][..],
            &state.piece_args(true),
            &[
                shape.channels.arg(),
                shape.taps.arg(),
                shape.dilation.arg(),
                span.row0.arg(),
            ],
        ]
        .concat(),
    )
}

/// Each request's rows of the query CSR, convolved in order over its slot,
/// which then holds the request's last rows.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_chunked(
    ctx: &Ctx<'_>,
    x: Tensor,
    csr: Csr,
    weight: Tensor,
    state: &StatePool,
    conv_width: u32,
    dilation: u32,
    y: Tensor,
    row0: u32,
) -> Result<(), Error> {
    const OP: &str = "ssm.causal_conv1d_chunked";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "ssm_conv1d_chunked_bf16" });
    let shape = Shape::of(OP, x, weight, state, conv_width, dilation, y)?;
    rows(OP, x)?;
    ctx.fire(
        Fire::at(CONV1D, state.entry(OP, entry)?).apply(shape.grid(csr.lanes(OP)?)),
        &[
            &[
                x.arg(),
                csr.table.arg(),
                weight.arg(),
                state.slots.arg(),
                y.arg_mut(),
            ][..],
            &state.snap_table(),
            &state.piece_args(true),
            &[
                shape.channels.arg(),
                shape.taps.arg(),
                shape.dilation.arg(),
                row0.arg(),
            ],
            &[csr.at.arg()],
            &state.snap_lane0(OP)?,
        ]
        .concat(),
    )
}

/// A speculative window's rows (the replayed ones first), each request's
/// slot kept as of its committed row.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_committed(
    ctx: &Ctx<'_>,
    x: Tensor,
    csr: Csr,
    committed: &Committed,
    weight: Tensor,
    state: &StatePool,
    conv_width: u32,
    dilation: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "ssm.causal_conv1d_committed";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "ssm_conv1d_committed_bf16" });
    let shape = Shape::of(OP, x, weight, state, conv_width, dilation, y)?;
    nonzero(OP, "extended rows", x.rows)?;
    ctx.fire(
        Fire::at(CONV1D, state.entry(OP, entry)?).apply(shape.grid(csr.lanes(OP)?)),
        &[
            &[
                x.arg(),
                csr.table.arg(),
                committed.replay.arg(),
                committed.commit.arg(),
                committed.slots.arg(),
                weight.arg(),
                y.arg_mut(),
            ][..],
            &state.snap_table(),
            &state.piece_args(true),
            &[
                stated(OP, committed.lane0)?.arg(),
                shape.channels.arg(),
                shape.taps.arg(),
                shape.dilation.arg(),
                0u32.arg(),
            ],
            &[csr.at.arg()],
            &state.snap_lane0(OP)?,
        ]
        .concat(),
    )
}

/// One decode step a lane over its buffered rows then its own
/// (`kernels/ssm/conv1d_carried.slang`): the own row's window from the
/// committed history and the lane's last buffered rows' inputs, read where
/// they lie in its rs buffer pages (plane `x_plane`); the history kept as
/// of the lane's commit; the own row's input landed in its buffer.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_carried(
    ctx: &Ctx<'_>,
    x: Tensor,
    weight: Tensor,
    state: &StatePool,
    carry: &Carry,
    x_plane: CarryPlane,
    conv_width: u32,
    dilation: u32,
    y: Tensor,
    span: RowSpan,
) -> Result<(), Error> {
    const OP: &str = "ssm.causal_conv1d_carried";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "ssm_conv1d_carried_bf16" });
    conv1d_carried_as(
        entry, ctx, x, weight, state, carry, x_plane, conv_width, dilation, y, span,
    )
}

/// A beam group's fold (`BEAM-PAGE-LISTS.md` §11.3): each of `carry`'s
/// `folds` lanes from its `lane0` is a group's set, its carried rows the
/// ones every live beam shares, folded into its history ahead of the step
/// that reads it; no own row, no input read, no output written. `x` and `y`
/// are the step's, bound as it binds them.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_carried_fold(
    ctx: &Ctx<'_>,
    x: Tensor,
    weight: Tensor,
    state: &StatePool,
    carry: &Carry,
    x_plane: CarryPlane,
    conv_width: u32,
    dilation: u32,
    y: Tensor,
    folds: u32,
) -> Result<(), Error> {
    const OP: &str = "ssm.causal_conv1d_carried_fold";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "ssm_conv1d_carried_fold_bf16" });
    conv1d_carried_as(
        entry,
        ctx,
        x,
        weight,
        state,
        carry,
        x_plane,
        conv_width,
        dilation,
        y,
        RowSpan {
            row0: 0,
            rows: folds,
        },
    )
}

#[allow(clippy::too_many_arguments)]
fn conv1d_carried_as(
    entry: &'static str,
    ctx: &Ctx<'_>,
    x: Tensor,
    weight: Tensor,
    state: &StatePool,
    carry: &Carry,
    x_plane: CarryPlane,
    conv_width: u32,
    dilation: u32,
    y: Tensor,
    span: RowSpan,
) -> Result<(), Error> {
    const OP: &str = "ssm.causal_conv1d_carried";
    let shape = Shape::of(OP, x, weight, state, conv_width, dilation, y)?;
    let launched = span.rows(OP, &[x, y])?;
    ctx.fire(
        Fire::at(CONV1D_CARRIED, entry).apply(shape.grid(launched)),
        &[
            &[
                x.arg(),
                weight.arg(),
                carry.slots.arg(),
                carry.table.arg(),
                y.arg_mut(),
            ][..],
            &state.piece_args(true),
            &[
                stated(OP, carry.lane0)?.arg(),
                shape.channels.arg(),
                shape.taps.arg(),
                shape.dilation.arg(),
                span.row0.arg(),
            ],
            &carry.page_args(OP)?,
            &x_plane.args(),
        ]
        .concat(),
    )
}
