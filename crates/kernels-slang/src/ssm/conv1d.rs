//! The causal depthwise conv with its rolling history (`ssm.causal_conv1d`,
//! its chunked and committed forms), from `kernels/ssm/conv1d.slang`.

use super::{Committed, StatePool, requests, rows, window_lanes};
use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, nonzero, refuse, stated};
use crate::error::Error;
use crate::tensor::{RaggedTensor, Tensor};

const CONV1D: &str = "ssm/conv1d.slang";

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
pub fn conv1d(
    ctx: &Ctx<'_>,
    x: Tensor,
    weight: Tensor,
    state: &StatePool,
    conv_width: u32,
    dilation: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "ssm.causal_conv1d";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "ssm_conv1d_bf16" });
    let shape = Shape::of(OP, x, weight, state, conv_width, dilation, y)?;
    ctx.fire(
        Fire::at(CONV1D, entry).apply(shape.grid(rows(OP, x)?)),
        &[
            x.arg(),
            weight.arg(),
            state.bank.arg_mut(),
            state.slots.arg(),
            y.arg_mut(),
            shape.channels.arg(),
            shape.taps.arg(),
            shape.dilation.arg(),
        ],
    )
}

/// Each request's rows of the query CSR, convolved in order over its slot,
/// which then holds the request's last rows.
pub fn conv1d_chunked(
    ctx: &Ctx<'_>,
    x: RaggedTensor,
    weight: Tensor,
    state: &StatePool,
    conv_width: u32,
    dilation: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "ssm.causal_conv1d_chunked";
    let entry = dtype_dispatch!(OP, x.data.dtype, { Bf16 => "ssm_conv1d_chunked_bf16" });
    let shape = Shape::of(OP, x.data, weight, state, conv_width, dilation, y)?;
    rows(OP, x.data)?;
    ctx.fire(
        Fire::at(CONV1D, entry).apply(shape.grid(requests(OP, x)?)),
        &[
            x.data.arg(),
            x.indptr.arg(),
            weight.arg(),
            state.bank.arg_mut(),
            state.slots.arg(),
            y.arg_mut(),
            shape.channels.arg(),
            shape.taps.arg(),
            shape.dilation.arg(),
        ],
    )
}

/// A speculative window's rows (the replayed ones first), each request's
/// slot kept as of its committed row.
#[allow(clippy::too_many_arguments)]
pub fn conv1d_committed(
    ctx: &Ctx<'_>,
    x: Tensor,
    indptr: Tensor,
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
        Fire::at(CONV1D, entry).apply(shape.grid(window_lanes(OP, indptr)?)),
        &[
            x.arg(),
            indptr.arg(),
            committed.replay.arg(),
            committed.commit.arg(),
            committed.slots.arg(),
            weight.arg(),
            state.bank.arg_mut(),
            y.arg_mut(),
            stated(OP, committed.lane0)?.arg(),
            shape.channels.arg(),
            shape.taps.arg(),
            shape.dilation.arg(),
        ],
    )
}
