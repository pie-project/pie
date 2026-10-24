//! The row frame (`elemwise/frame.cuh`): one row a block, held whole in
//! registers, around a body that computes it in hooks over the frame. The
//! bodies here are written by hand (`elemwise/frame.cu`), each landing what
//! the hand kernel of its op lands; generated ones ([`crate::synth`]) fire
//! through the same launch.

use super::BLOCK;
use crate::error::Error;
use crate::jit::{Arg, ArgValue, Ctx, Fire, Kernel, Launch, aligned16, kernel, nonzero, refuse};
use crate::tensor::Tensor;
use dtype::Dtype;

const FRAME: &str = "elemwise/frame.cu";

/// The elements of one vector a thread loads: a row's width is whole
/// vectors of it.
pub const VEC_WIDTH: u32 = 8;

/// The most vectors of 8 a thread holds of each plane: past it a row no
/// longer fits the registers.
const MAX_PER: u32 = 4;

/// The widest row the frame holds.
pub const MAX_WIDTH: u32 = BLOCK * VEC_WIDTH * MAX_PER;

/// `pie::elemwise::Plane`, field for field: a bf16 plane's first element
/// and how far apart its rows sit, in elements.
#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Plane {
    base: u64,
    row_stride: i64,
}

impl Plane {
    /// A plane whose every row is its own.
    #[must_use]
    pub fn rows(t: &Tensor) -> Self {
        Self {
            base: t.ptr,
            row_stride: i64::from(t.width),
        }
    }

    /// One row every row reads (a norm's weight).
    #[must_use]
    pub const fn shared(t: &Tensor) -> Self {
        Self {
            base: t.ptr,
            row_stride: 0,
        }
    }

    /// The plane's first element.
    #[must_use]
    pub const fn base(&self) -> u64 {
        self.base
    }
}

/// The vectors of 8 a thread holds of each plane of a `width`-wide row, or
/// why the frame does not hold such a row.
pub(crate) fn per(op: &'static str, width: u32) -> Result<u32, Error> {
    let width = nonzero(op, "the row's width", width)?;
    if !width.is_multiple_of(VEC_WIDTH) {
        return Err(refuse(
            op,
            format!("a {width}-wide row is not whole vectors of {VEC_WIDTH}"),
        ));
    }
    if width > MAX_WIDTH {
        return Err(refuse(
            op,
            format!("a {width}-wide row is wider than the frame holds ({MAX_WIDTH})"),
        ));
    }
    Ok((width / VEC_WIDTH).div_ceil(BLOCK))
}

/// The frame's kernel around `body` (a type the unit defines) for `ins`
/// parameters and `outs` results, `per` vectors a thread, the parameters
/// `ones` sets (by bit) read as one element. A `read_only` frame's results
/// land over none of its parameters, so it reads them through the
/// read-only path.
pub(crate) fn entrypoint(
    body: &str,
    per: u32,
    ins: usize,
    outs: usize,
    ones: u32,
    read_only: bool,
) -> String {
    if ones == 0 && !read_only {
        format!("::pie::elemwise::row_frame<{body}, {BLOCK}, {per}, {ins}, {outs}>")
    } else {
        format!(
            "::pie::elemwise::row_frame<{body}, {BLOCK}, {per}, {ins}, {outs}, {ones}u, {read_only}>"
        )
    }
}

/// Fires `kernel`, a row frame of [`entrypoint`] (`fields` its body's
/// members by value), over `rows` rows of bf16 planes `width` wide:
/// `ins[k]` is the body's parameter `k` and `outs[k]` its result `k`, a
/// parameter `ones` sets (by bit) one element, read as is.
#[allow(clippy::too_many_arguments)]
pub(crate) fn fire(
    ctx: &Ctx,
    op: &'static str,
    kernel: &'static Kernel,
    fields: ArgValue,
    ins: &[Plane],
    outs: &[Plane],
    rows: u32,
    width: u32,
    ones: u32,
) -> Result<(), Error> {
    let rows = nonzero(op, "rows", rows)?;
    let rowed = (0u32..)
        .zip(ins)
        .filter(|(k, _)| (ones >> k) & 1 == 0)
        .map(|(_, p)| p);
    if !rowed
        .chain(outs)
        .all(|p| aligned16(p.base) && p.row_stride % i64::from(VEC_WIDTH) == 0)
    {
        return Err(refuse(op, "a plane is not 16-byte aligned at every row"));
    }
    // `pie::elemwise::Planes<IN, OUT>`: the planes, then the width in the
    // low half of the last 8-byte word (the struct's tail padding the rest).
    let planes: Vec<u64> = ins
        .iter()
        .chain(outs)
        .flat_map(|p| [p.base, p.row_stride.cast_unsigned()])
        .chain([u64::from(width)])
        .collect();
    ctx.fire(
        op,
        // The frame waits on its producer in-kernel (`kernel.pdl`).
        Fire::at(kernel).apply(Launch::per_row(rows, BLOCK).programmatic(ctx.tuning().pdl)),
        &[ArgValue::of_slice(&planes), fields, ctx.stage()],
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
    let per = per(OP, y.width)?;
    let body = format!("::pie::elemwise::ResidualAddRmsnorm<{plus_one}>");
    fire(
        ctx,
        OP,
        kernel(FRAME, &entrypoint(&body, per, 3, 2, 0, false)),
        eps.arg(),
        &[Plane::rows(&x), Plane::rows(y), Plane::shared(&weight)],
        &[Plane::rows(out), Plane::rows(y)],
        y.rows,
        y.width,
        0,
    )
}
