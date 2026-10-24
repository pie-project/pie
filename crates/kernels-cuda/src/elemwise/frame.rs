//! The row frame (`elemwise/frame.cuh`): one row a block, held whole in
//! registers, around a body that computes it in hooks over the frame. The
//! bodies here are written by hand (`elemwise/frame.cu`); each lands what the
//! hand kernel of its op lands.

use super::BLOCK;
use crate::error::Error;
use crate::jit::{Arg, ArgValue, Ctx, Fire, Launch, aligned16, kernel, nonzero, refuse, stated};
use crate::tensor::Tensor;
use dtype::Dtype;

const FRAME: &str = "elemwise/frame.cu";

const VEC_WIDTH: u32 = 8;

/// The most vectors of 8 a thread holds of each plane: past it a row no
/// longer fits the registers.
const MAX_PER: u32 = 4;

/// `pie::elemwise::Plane`, field for field.
#[repr(C)]
#[derive(Clone, Copy, Debug)]
struct Plane {
    base: u64,
    row_stride: i64,
}

impl Plane {
    /// A plane whose every row is its own.
    fn rows(t: &Tensor) -> Self {
        Self {
            base: t.ptr,
            row_stride: i64::from(t.width),
        }
    }

    /// One row every row reads (a norm's weight).
    const fn shared(t: &Tensor) -> Self {
        Self {
            base: t.ptr,
            row_stride: 0,
        }
    }
}

/// `pie::elemwise::Planes<IN, OUT>`, field for field.
#[repr(C)]
struct Planes<const IN: usize, const OUT: usize> {
    ins: [Plane; IN],
    outs: [Plane; OUT],
    width: i32,
}

/// Fires `body` (a `pie::elemwise` type, `fields` its members by value)
/// over `rows` rows of bf16 planes `width` wide: `ins[k]` is the body's
/// parameter `k` and `outs[k]` its result `k`.
#[allow(clippy::too_many_arguments)]
fn fire<const IN: usize, const OUT: usize>(
    ctx: &Ctx,
    op: &'static str,
    body: &str,
    fields: ArgValue,
    ins: [Plane; IN],
    outs: [Plane; OUT],
    rows: u32,
    width: u32,
) -> Result<(), Error> {
    let rows = nonzero(op, "rows", rows)?;
    let width = nonzero(op, "the row's width", width)?;
    if !width.is_multiple_of(VEC_WIDTH) {
        return Err(refuse(
            op,
            format!("a {width}-wide row is not whole vectors of {VEC_WIDTH}"),
        ));
    }
    let per = (width / VEC_WIDTH).div_ceil(BLOCK);
    if per > MAX_PER {
        return Err(refuse(
            op,
            format!(
                "a {width}-wide row is wider than the frame holds ({})",
                BLOCK * VEC_WIDTH * MAX_PER
            ),
        ));
    }
    let planes = ins.iter().chain(&outs);
    if !planes
        .clone()
        .all(|p| aligned16(p.base) && p.row_stride % i64::from(VEC_WIDTH) == 0)
    {
        return Err(refuse(op, "a plane is not 16-byte aligned at every row"));
    }
    let planes = Planes {
        ins,
        outs,
        width: stated(op, width)?,
    };
    let entrypoint = format!("::pie::elemwise::row_frame<{body}, {BLOCK}, {per}, {IN}, {OUT}>");
    ctx.fire(
        op,
        // The frame waits on its producer in-kernel (`kernel.pdl`).
        Fire::at(kernel(FRAME, &entrypoint))
            .apply(Launch::per_row(rows, BLOCK).programmatic(ctx.tuning().pdl)),
        &[ArgValue::of(&planes), fields, ctx.stage()],
    )
}

/// `norm_residual::residual_add_rmsnorm` as a row frame's body: `y += x` in
/// place, then `out` the RMS norm of `y` by `weight` (by `1 + weight` with
/// `plus_one`), bit for bit as `residual_add_rmsnorm_reg`. Rows up to 8192
/// bf16 wide, every plane 16-byte aligned.
pub fn residual_add_rmsnorm(
    ctx: &Ctx,
    x: Tensor,
    y: &mut Tensor,
    weight: Tensor,
    plus_one: bool,
    eps: f32,
    out: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "fused.residual_add_rmsnorm";
    if y.dtype != Dtype::Bf16 {
        return Err(Error::DtypeUnsupported {
            op: OP,
            dtype: y.dtype,
        });
    }
    debug_assert!(
        x.rows == y.rows && x.width == y.width && out.rows == y.rows && out.width == y.width,
        "the residual, the stream and the normed row share one shape"
    );
    fire(
        ctx,
        OP,
        &format!("::pie::elemwise::ResidualAddRmsnorm<{plus_one}>"),
        eps.arg(),
        [Plane::rows(&x), Plane::rows(y), Plane::shared(&weight)],
        [Plane::rows(out), Plane::rows(y)],
        y.rows,
        y.width,
    )
}
