//! `Layernorm`: the CENTRED norm, and the one part of a `LayerNorm` that no
//! import can bake away (`.wiki/alto/multimodal.md` §6.1).
//!
//! A file of its own beside `norm.rs` rather than an arm inside it, for
//! `rope_mrope.rs`'s reason one door over: `norm.rs` is the RMS family — eight
//! arms that differ in how they READ A WEIGHT and not in what they reduce —
//! and this one subtracts a mean and reads no weight at all. Sharing the file
//! would have made the family's one shared helper (`rms_row`) grow a boolean
//! that changes what it reduces, which is the shape "a fourth axis with a
//! fourth fact, not a flag" says not to build.
//!
//! **WHY THERE IS NO SCALE AND NO BIAS.** Every qwen vision block is
//! `nn.LayerNorm`, and the checkpoints publish `blocks.{l}.norm1.bias` beside
//! `.weight` to prove it. For the GEMM `M` that reads the norm,
//! `LN(x)·Mᵀ = (c/rms(c))·diag(w)·Mᵀ + b·Mᵀ` with `c = x − mean(x)`, so `w`
//! folds into `M` at import and `b·Mᵀ` folds into that GEMM's bias — and the
//! merger's own norm folds the same way through the 2×2 merge, which is a
//! view. What is left is exactly this entry.

use crate::error::Error;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "elemwise/layernorm.cuh";

const BLOCK: u32 = 256;

/// **THE CENTRED NORM**: `y = (x − mean(x)) · rsqrt(var(x) + eps)`, whole
/// rows, no scale and no bias.
///
/// One block per row, as the rms arms are. There is no `head_dim`: the towers
/// norm whole rows, and a per-head spelling would be a promise no checkpoint
/// this campaign serves makes.
///
/// # Errors
///
/// [`Error::DtypeUnsupported`] for anything but bf16 and f16; a refusal for a
/// zero-wide row or a zero-row rectangle, which are the two launches that
/// would leave the destination unwritten rather than normed.
pub fn layernorm_no_scale(ctx: &Ctx, x: Tensor, eps: f32, y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.layernorm_no_scale";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let hidden = stated(OP, nonzero(OP, "the normed width", y.width)?)?;
    let rows = nonzero(OP, "rows", y.rows)?;
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!("::pie::elemwise::layernorm_no_scale<{t}, 256>")),
        )
        .apply(Launch::per_row(rows, BLOCK)),
        &[x.arg(), y.arg(), hidden.arg(), eps.arg()],
    )
}
