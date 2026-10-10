//! The GPU's side of a split MLP: the kernels in `linear/ane.metal` and the
//! sequences a layer runs them in. The Neural Engine's side is
//! [`crate::ane::ffn`]; the two meet at the hand-off event, which the GPU
//! raises after packing its inputs ([`prepare`]) and waits on before adding
//! its partial back ([`join`]).
//!
//! Everything here is over resolved tensors; who owns them, which layer is
//! staged, and whether the Neural Engine is ready are the engine's business.

use crate::ane::ffn::{INPUT_BLOCK, INTERMEDIATE_BLOCK, SEGMENT, Shape};
use crate::encode::{Arg, ArgValue, Ctx, Fire, Grid};
use crate::error::Error;
use crate::tensor::{Bank, Tensor};

pub const FILE: &str = "linear/ane.metal";

const GROUP: u32 = 256;

/// Rotates bf16 activations in 128-blocks, quantizes each token to int8 by
/// its own peak, and writes the per-token scale.
pub fn rotate(
    ctx: &Ctx<'_>,
    x: Tensor,
    sign: Tensor,
    rotated: Tensor,
    token_scale: Tensor,
) -> Result<(), Error> {
    ctx.fire(
        Fire::at(FILE, "ane_rotate").apply(Grid::of([x.rows * GROUP, 1, 1], [GROUP, 1, 1])),
        &[
            x.arg(),
            sign.arg(),
            rotated.arg_mut(),
            token_scale.arg_mut(),
            x.width.arg(),
        ],
    )
}

/// One hidden segment of the Neural Engine's channel-major input surface.
pub struct Pack {
    /// The first hidden channel of the segment.
    pub channel: u32,
    pub width: u32,
    /// The surface's row stride in bytes.
    pub stride: u32,
}

/// Transposes one segment of the rotated activations into the surface.
pub fn pack(ctx: &Ctx<'_>, rotated: Tensor, packed: Tensor, at: &Pack) -> Result<(), Error> {
    let tiles = rotated.rows.div_ceil(32);
    ctx.fire(
        Fire::at(FILE, "ane_pack").apply(Grid::of([tiles * 32, at.width / 32 * 8, 1], [32, 8, 1])),
        &[
            rotated.arg(),
            packed.arg_mut(),
            rotated.width.arg(),
            at.channel.arg(),
            at.stride.arg(),
            rotated.rows.arg(),
        ],
    )
}

/// A run of a Q4 bank's rows and the span of their inputs to take.
pub struct Rows {
    pub first: u32,
    pub rows: u32,
    pub input: u32,
    pub span: u32,
    /// Whether the inputs are the swiglu intermediate, rotated in 512-blocks,
    /// rather than the hidden state, rotated in 128-blocks.
    pub intermediate: bool,
}

fn planes(bank: &Bank) -> Result<[ArgValue; 3], Error> {
    let biases = bank.biases.ok_or(Error::Unsupported {
        op: "linear.ane_weights",
    })?;
    Ok([bank.codes.arg(), bank.scales.arg(), biases.arg()])
}

/// The per-row fp16 scale that quantizes the rotated rows to int8.
pub fn row_scale(
    ctx: &Ctx<'_>,
    bank: &Bank,
    at: &Rows,
    sign: Tensor,
    scale: Tensor,
) -> Result<(), Error> {
    let entry = if at.intermediate {
        "ane_row_scale_intermediate"
    } else {
        "ane_row_scale_inputs"
    };
    let [codes, scales, biases] = planes(bank)?;
    ctx.fire(
        Fire::at(FILE, entry).apply(Grid::of([at.rows / 8 * GROUP, 1, 1], [GROUP, 1, 1])),
        &[
            codes,
            scales,
            biases,
            scale.arg_mut(),
            sign.arg(),
            bank.ld.arg(),
            at.first.arg(),
            at.input.arg(),
            at.span.arg(),
        ],
    )
}

/// One weight surface of the Neural Engine's and the scale surface beside it.
pub struct Target {
    pub weights: Tensor,
    /// The weight surface's row stride in bytes.
    pub stride: u32,
    pub scale: Tensor,
    /// The scale surface's row stride in elements.
    pub scale_stride: u32,
}

/// Dequantizes, rotates and requantizes the rows into the target.
pub fn weights(
    ctx: &Ctx<'_>,
    bank: &Bank,
    at: &Rows,
    sign: Tensor,
    row_scale: Tensor,
    target: &Target,
) -> Result<(), Error> {
    let (entry, block) = if at.intermediate {
        ("ane_weights_intermediate", INTERMEDIATE_BLOCK)
    } else {
        ("ane_weights_inputs", INPUT_BLOCK)
    };
    let [codes, scales, biases] = planes(bank)?;
    ctx.fire(
        Fire::at(FILE, entry).apply(Grid::of(
            [at.rows / 8 * GROUP, at.span / block, 1],
            [GROUP, 1, 1],
        )),
        &[
            codes,
            scales,
            biases,
            row_scale.arg(),
            target.weights.arg_mut(),
            target.scale.arg_mut(),
            sign.arg(),
            bank.ld.arg(),
            at.first.arg(),
            at.input.arg(),
            target.stride.arg(),
            target.scale_stride.arg(),
        ],
    )
}

/// Adds the Neural Engine's partial, rescaled per token, into the output,
/// raising `status` on any non-finite value.
pub fn split_join(
    ctx: &Ctx<'_>,
    out: Tensor,
    partial: Tensor,
    token_scale: Tensor,
    status: Tensor,
    stride: u32,
) -> Result<(), Error> {
    let tiles = out.rows.div_ceil(32);
    ctx.fire(
        Fire::at(FILE, "ane_join").apply(Grid::of([tiles * 32, out.width / 32 * 8, 1], [32, 8, 1])),
        &[
            out.arg_mut(),
            partial.arg(),
            token_scale.arg(),
            status.arg_mut(),
            out.width.arg(),
            stride.arg(),
            out.rows.arg(),
        ],
    )
}

/// One layer's MLP weights as the Neural Engine's slice is cut from them:
/// the Q4 banks and the three row scales [`row_scales`] computes once.
pub struct Source {
    pub gate_up: Bank,
    pub down: Bank,
    /// Gate rows, up rows, down rows.
    pub scales: [Tensor; 3],
}

/// One weight set of the Neural Engine's: a target per gate segment, per up
/// segment, and per down slice.
pub struct Set {
    pub gate: Vec<Target>,
    pub up: Vec<Target>,
    pub down: Vec<Target>,
}

/// The surfaces every layer shares: the input segments with their strides,
/// the token scale, the partial with its stride.
pub struct Shared {
    pub inputs: Vec<(Tensor, u32)>,
    pub token_scale: Tensor,
    pub partial: (Tensor, u32),
    pub signs: Tensor,
    pub status: Tensor,
}

/// Computes the layer's three row scales. Once, at load.
pub fn row_scales(
    ctx: &Ctx<'_>,
    shape: &Shape,
    source: &Source,
    signs: Tensor,
) -> Result<(), Error> {
    let gate = Rows {
        first: shape.gpu,
        rows: shape.ane,
        input: 0,
        span: shape.hidden,
        intermediate: false,
    };
    let up = Rows {
        first: shape.intermediate + shape.gpu,
        ..gate
    };
    let down = Rows {
        first: 0,
        rows: shape.hidden,
        input: shape.gpu,
        span: shape.ane,
        intermediate: true,
    };
    row_scale(ctx, &source.gate_up, &gate, signs, source.scales[0])?;
    row_scale(ctx, &source.gate_up, &up, signs, source.scales[1])?;
    row_scale(ctx, &source.down, &down, signs, source.scales[2])
}

/// Materializes the layer's slice of the weights into `set`: the trailing
/// `shape.ane` rows of gate and up, and the matching columns of down.
pub fn stage(
    ctx: &Ctx<'_>,
    shape: &Shape,
    source: &Source,
    signs: Tensor,
    set: &Set,
) -> Result<(), Error> {
    for (k, (gate, up)) in set.gate.iter().zip(&set.up).enumerate() {
        let rows = Rows {
            first: shape.gpu,
            rows: shape.ane,
            input: k as u32 * SEGMENT,
            span: SEGMENT,
            intermediate: false,
        };
        weights(ctx, &source.gate_up, &rows, signs, source.scales[0], gate)?;
        let rows = Rows {
            first: shape.intermediate + shape.gpu,
            ..rows
        };
        weights(ctx, &source.gate_up, &rows, signs, source.scales[1], up)?;
    }
    let mut input = shape.gpu;
    for (&span, down) in shape.down.iter().zip(&set.down) {
        let rows = Rows {
            first: 0,
            rows: shape.hidden,
            input,
            span,
            intermediate: true,
        };
        weights(ctx, &source.down, &rows, signs, source.scales[2], down)?;
        input += span;
    }
    Ok(())
}

/// Rotates and packs `x` into the Neural Engine's inputs, then raises the
/// hand-off event to `ready` so it may start.
pub fn prepare(
    ctx: &Ctx<'_>,
    shared: &Shared,
    x: Tensor,
    rotated: Tensor,
    ready: u64,
) -> Result<(), Error> {
    rotate(ctx, x, shared.signs, rotated, shared.token_scale)?;
    for (k, &(packed, stride)) in shared.inputs.iter().enumerate() {
        pack(
            ctx,
            rotated,
            packed,
            &Pack {
                channel: k as u32 * SEGMENT,
                width: SEGMENT,
                stride,
            },
        )?;
    }
    ctx.signal(ready)
}

/// Waits for the hand-off event to reach `done`, then adds the Neural
/// Engine's partial into `out`.
pub fn join(ctx: &Ctx<'_>, shared: &Shared, out: Tensor, done: u64) -> Result<(), Error> {
    ctx.wait(done)?;
    let (partial, stride) = shared.partial;
    split_join(ctx, out, partial, shared.token_scale, shared.status, stride)
}
