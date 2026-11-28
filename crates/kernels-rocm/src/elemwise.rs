pub mod gate;
pub mod hc;
pub mod hc_project;
pub mod modulate;
pub mod norm;
pub mod norm_residual;
pub mod norm_rope;
pub mod pointwise;
pub mod posenc;
pub mod rope;
pub mod skeleton;

use dtype::Dtype;

use crate::error::Error;
use crate::jit::{Launch, nonzero, refuse};
use crate::tensor::Tensor;

const NORM_UNIT: &str = "elemwise/norm.hip";

const BLOCK: u32 = 256;

/// Lanes per wavefront on CDNA: a block is a whole number of them.
const WARP: u32 = 64;

const VEC_WIDTH: u32 = 8;

/// A plane's element count as the flat kernels take it: any count a plane
/// holds, refused at zero.
pub(crate) fn element_count(
    op: &'static str,
    axis: &'static str,
    t: &Tensor,
) -> Result<i64, Error> {
    let n = t.elements();
    if n == 0 {
        return Err(refuse(op, format!("`{axis}` is zero")));
    }
    i64::try_from(n).map_err(|_| refuse(op, format!("{n} elements are past any plane")))
}

/// A flat grid-stride sweep over `n` elements (or lanes), a thread each,
/// and the index type its kernel runs at: unsigned 32-bit while `n` fits
/// a signed 32-bit count (a thread's next index then stays under 2^32),
/// 64-bit past it.
#[derive(Clone, Copy, Debug)]
pub(crate) struct Sweep {
    pub launch: Launch,
    pub n: i64,
}

impl Sweep {
    #[allow(clippy::cast_sign_loss)]
    pub(crate) fn of(n: i64, block: u32) -> Sweep {
        Sweep {
            launch: Launch::strided(n as u64, block),
            n,
        }
    }

    fn narrow(self) -> bool {
        self.n <= i64::from(i32::MAX)
    }

    /// The kernel's index type, its last template argument.
    pub(crate) fn index(self) -> &'static str {
        if self.narrow() {
            "unsigned"
        } else {
            "long long"
        }
    }

    /// The count as the kernel's index type takes it.
    #[allow(clippy::cast_possible_truncation, clippy::cast_sign_loss)]
    pub(crate) fn count(self) -> crate::jit::ArgValue<'static> {
        if self.narrow() {
            crate::jit::ArgValue::U32(self.n as u32)
        } else {
            crate::jit::ArgValue::I64(self.n)
        }
    }
}

/// A flat sweep over `t`'s elements, a thread an element.
pub(crate) fn elementwise(op: &'static str, t: Tensor) -> Result<Sweep, Error> {
    Ok(Sweep::of(
        element_count(op, "the element count", &t)?,
        BLOCK,
    ))
}

fn route_rows(rows: u32, width: u32) -> Launch {
    const MAX_BLOCK: u32 = 1024;

    Launch::per_row(
        rows,
        width
            .div_ceil(WARP)
            .max(1)
            .saturating_mul(WARP)
            .min(MAX_BLOCK),
    )
}

const ROPE_UNIT: &str = "elemwise/rope.hip";

fn heads(op: &'static str, width: u32, head_dim: u32) -> Result<u32, Error> {
    nonzero(op, "the head width this rotation states", head_dim)?;
    if !width.is_multiple_of(head_dim) {
        return Err(refuse(
            op,
            format!("the {width}-wide row is not a whole number of {head_dim}-wide heads"),
        ));
    }
    Ok(width / head_dim)
}

fn positions_stream(op: &'static str, positions: Tensor, x: &Tensor) {
    debug_assert_eq!(
        positions.dtype,
        Dtype::I32,
        "`{op}` reads an i32 position stream"
    );
    debug_assert_eq!(
        positions.rows, x.rows,
        "the position stream is one entry per rotated row"
    );
}
