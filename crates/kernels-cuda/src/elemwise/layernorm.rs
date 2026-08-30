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
//!
//! **AND WHY THERE IS A SECOND ENTRY THAT READS BOTH** (`.wiki/alto/next.md`
//! B5). §9.1 read that fold again against `checkpoint::contract::Expr` and
//! found it HALF expressible — the scale is `Expr::Scale`'s per-block factor,
//! the bias is a matrix-vector product no `Expr` computes, and the two do not
//! compose behind one another. So the qwen towers say the whole norm at
//! runtime. [`layernorm`] is that sentence in one launch;
//! [`layernorm_no_scale`] stays for the text whose scale genuinely bakes.

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

/// **AND THE WHOLE `nn.LayerNorm`, IN ONE LAUNCH**: `y = (x − mean(x)) ·
/// rsqrt(var(x) + eps) · w + b`, whole rows, both planes `[width]`
/// (`.wiki/alto/next.md` B5).
///
/// The three-op spelling this replaces —
/// `add_bias(b, rmsnorm(layernorm_no_scale(x, eps), w, eps))` — was written
/// because multimodal §9.1 found the import fold half expressible:
/// `Expr::Scale` bakes `w` into the GEMM behind the norm, `Expr::Bias` adds
/// one compile-time constant where `b·Mᵀ` is a matrix-vector product, and the
/// two do not compose. So the towers say the whole norm at runtime, and until
/// this entry they said it three times: two elementwise passes and a bias,
/// twenty-five norms per qwen35 tower fire.
///
/// **THE ARITHMETIC IS THE IDEAL ONE AND NOT THE COMPOSITION'S**, which is a
/// decision and not an accident. The middle `rmsnorm` of the three-op form
/// reduces over the centred row AFTER it has been rounded to the storage
/// type, so it multiplies by the reciprocal rms of a row whose rms is 1 only
/// up to that rounding — a uniform per-row factor of `1 ± 1.4e-4`, thirty
/// times inside bf16's own quantum and no part of what `torch.nn.LayerNorm`
/// computes. This entry has no such intermediate to reduce over: reproducing
/// the factor would mean inventing a round-trip through a storage type the
/// fused op no longer names, and the invented number would move if the
/// destination were f16 instead of bf16. The centred row therefore stays f32
/// to the single rounding at the store, which lands STRICTLY NEARER the f32
/// reference than the form it replaces — the direction a saving is allowed to
/// move a number in, and the one `tower_centred_norm` measures.
///
/// # Errors
///
/// [`Error::DtypeUnsupported`] for anything but bf16 and f16; a refusal for a
/// zero-wide row or a zero-row rectangle, which are the two launches that
/// would leave the destination unwritten rather than normed.
pub fn layernorm(
    ctx: &Ctx,
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    eps: f32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.layernorm";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let hidden = stated(OP, nonzero(OP, "the normed width", y.width)?)?;
    let rows = nonzero(OP, "rows", y.rows)?;
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!("::pie::elemwise::layernorm<{t}, 256>")),
        )
        .apply(Launch::per_row(rows, BLOCK)),
        &[
            x.arg(),
            weight.arg(),
            bias.arg(),
            y.arg(),
            hidden.arg(),
            eps.arg(),
        ],
    )
}
