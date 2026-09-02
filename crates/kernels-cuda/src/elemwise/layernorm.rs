//! `Layernorm`: the centred norm (subtracts a mean), the one part of a
//! `LayerNorm` an import can't always bake away. Separate from `norm.rs`
//! (the RMS family) because it reduces differently, not just reads its
//! weight differently. [`layernorm_no_scale`] is for weights that do fold
//! into the surrounding GEMM at import; [`layernorm`] is the whole op in one
//! launch for the qwen vision towers, whose scale+bias fold is only half
//! expressible against `checkpoint::contract::Expr`.

use crate::error::Error;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "elemwise/layernorm.cuh";

const BLOCK: u32 = 256;

/// `y = (x − mean(x)) · rsqrt(var(x) + eps)`, whole rows, no scale or bias.
/// One block per row; no `head_dim` since towers norm whole rows.
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
        &[
            x.arg(),
            y.arg(),
            hidden.arg(),
            eps.arg(),
            // Staged-geometry seat: live-rows word when a body replay armed
            // one, ABSENT otherwise.
            ctx.stage(),
        ],
    )
}

/// The whole `nn.LayerNorm` in one launch: `y = (x − mean(x)) ·
/// rsqrt(var(x) + eps) · w + b`, whole rows, both planes `[width]`. Replaces
/// a three-op form (`add_bias(b, rmsnorm(layernorm_no_scale(x, eps), w,
/// eps))`) whose middle `rmsnorm` reduced over an already-rounded row; this
/// entry keeps the centred row in f32 to a single rounding at the store, so
/// it lands nearer the f32 reference than the composition it replaces.
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
            // Staged-geometry seat: live-rows word when a body replay armed
            // one, ABSENT otherwise.
            ctx.stage(),
        ],
    )
}
