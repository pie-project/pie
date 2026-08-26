//! `Hc`: hyper-connections — residual streams expanded, mixed by learned
//! gates, and collapsed back to one stream. One entry per IR variant, all
//! over `norm/dsv4_hc.cuh` (the launches lived beside the norms in the old
//! plane; the family gets its own file here because the IR gives it one).
//!
//! The stream count `M` rides the row width — a `[N, M·H]` rectangle beside
//! a `[N, H]` one — and the mixers hold their `M` (or `M²`) coefficients in
//! registers and shared arrays, which is why [`MAX_HC_MULT`] is a hard
//! refusal and not a shape check.

use new_kernels::KernelError;
use new_model_ir::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "norm/dsv4_hc.cuh";

const BLOCK: u32 = 256;

/// The largest stream count the mixers hold in registers (`hc_post`'s
/// in-place read-before-write array) and shared gate vectors.
const MAX_HC_MULT: u32 = 8;

/// The stream fan `M`: how many `hidden`-wide streams the wide row holds.
/// The variants that state a count beside their rows assert agreement at
/// their own entries; `fold` states none.
fn stream_fan(op: &'static str, wide: u32, hidden: u32) -> Result<u32, KernelError> {
    nonzero(op, "the hidden width", hidden)?;
    if wide == 0 || wide % hidden != 0 {
        return Err(refuse(
            op,
            format!(
                "the {wide}-wide row is not a whole number of {hidden}-wide \
                 hyper-connection streams"
            ),
        ));
    }
    let fan = wide / hidden;
    if fan > MAX_HC_MULT {
        return Err(refuse(
            op,
            format!(
                "the stream count is {fan}, above the {MAX_HC_MULT} the mixers unroll into \
                 register and shared arrays"
            ),
        ));
    }
    Ok(fan)
}

/// One thread per INPUT element — these kernels' grids cover the `[N, H]`
/// side and each thread writes its `M` outputs (the old `ElementwiseIn`
/// rule). Refused rather than clamped past a 32-bit launch.
fn elementwise_in(op: &'static str, rows: u32, width: u32) -> Result<Launch, KernelError> {
    nonzero(op, "rows", rows)?;
    nonzero(op, "width", width)?;
    let n = u64::from(rows) * u64::from(width);
    let lanes = u32::try_from(n).map_err(|_| {
        refuse(
            op,
            format!("{n} elements do not fit a 32-bit launch extent"),
        )
    })?;
    Ok(Launch::flat(lanes, BLOCK))
}

/// Tiles `x` across `streams` residual streams.
pub fn expand(ctx: &Ctx, x: Tensor, streams: u32, y: &mut Tensor) -> Result<(), KernelError> {
    const OP: &str = "hc.expand";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(y.rows, x.rows, "the expansion lands one wide row per row");
    let fan = stream_fan(OP, y.width, x.width)?;
    debug_assert_eq!(
        fan, streams,
        "the row's stream fan is the count the statement states"
    );
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::norm::hc_expand<{t}>")))
            .apply(elementwise_in(OP, x.rows, x.width)?),
        &[
            x.arg(),
            y.arg(),
            stated(OP, x.rows)?.arg(),
            stated(OP, fan)?.arg(),
            stated(OP, x.width)?.arg(),
        ],
    )
}

/// RMS-normalises the wide stream row and widens it to f32 — the mix
/// coefficients derived downstream are too sensitive for a bf16 round-trip.
pub fn rmsnorm_f32(
    ctx: &Ctx,
    streams: Tensor,
    eps: f32,
    y: &mut Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "hc.rmsnorm_f32";
    dtype_dispatch!(OP, streams.dtype, { Bf16 => () });
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` widens to f32");
    debug_assert!(
        y.rows == streams.rows && y.width == streams.width,
        "the normed rectangle is the stream rectangle"
    );
    nonzero(OP, "rows", y.rows)?;
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::norm::hc_rmsnorm_to_f32<::pie::bf16, 256>")
            .apply(Launch::per_row(y.rows, BLOCK)),
        &[
            streams.arg(),
            y.arg(),
            stated(OP, nonzero(OP, "the normed row's width", y.width)?)?.arg(),
            eps.arg(),
        ],
    )
}

/// Splits the mix row, Sinkhorn-normalises the combiner, and collapses the
/// `M` streams into the layer's input — one block per token, the gate
/// matrices landing beside it.
#[allow(clippy::too_many_arguments)]
pub fn gates(
    ctx: &Ctx,
    normed: Tensor,
    streams: Tensor,
    scale: Tensor,
    base: Tensor,
    stream_count: u32,
    gate_eps: f32,
    alpha: f32,
    sinkhorn: u32,
    x: &mut Tensor,
    post_mix: &mut Tensor,
    comb_mix: &mut Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "hc.gates";
    let t = dtype_dispatch!(OP, streams.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(normed.dtype, Dtype::F32, "`{OP}` reads an f32 mix row");
    debug_assert_eq!(scale.dtype, Dtype::F32, "`{OP}` reads f32 mix scales");
    debug_assert_eq!(base.dtype, Dtype::F32, "`{OP}` reads f32 mix bases");
    debug_assert!(
        post_mix.dtype == Dtype::F32 && comb_mix.dtype == Dtype::F32,
        "`{OP}` lands f32 gate matrices"
    );
    let fan = stream_fan(OP, streams.width, x.width)?;
    debug_assert_eq!(
        fan, stream_count,
        "the row's stream fan is the count the statement states"
    );
    debug_assert!(
        post_mix.width == fan && comb_mix.width == fan * fan,
        "the gate matrices are `[N, M]` and `[N, M, M]`"
    );
    nonzero(OP, "rows", x.rows)?;
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!("::pie::norm::hc_pre_postprocess<{t}, 256>")),
        )
        .apply(Launch::per_row(x.rows, BLOCK)),
        &[
            normed.arg(),
            scale.arg(),
            base.arg(),
            streams.arg(),
            post_mix.arg(),
            comb_mix.arg(),
            x.arg(),
            stated(OP, fan)?.arg(),
            stated(OP, nonzero(OP, "the hidden width", x.width)?)?.arg(),
            gate_eps.arg(),
            alpha.arg(),
            stated(OP, sinkhorn)?.arg(),
        ],
    )
}

/// Mixes the layer's output back into the streams under the gate matrices —
/// in place across the wide row, which is why each thread owns its whole
/// `(n, h)` column.
pub fn fold(
    ctx: &Ctx,
    x: Tensor,
    streams: Tensor,
    post_mix: Tensor,
    comb_mix: Tensor,
    y: &mut Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "hc.fold";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert!(
        y.rows == streams.rows && y.width == streams.width,
        "the fold lands the stream rectangle it mixes"
    );
    let fan = stream_fan(OP, y.width, x.width)?;
    debug_assert!(
        post_mix.dtype == Dtype::F32 && comb_mix.dtype == Dtype::F32,
        "`{OP}` reads f32 gate matrices"
    );
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::norm::hc_post<{t}>")))
            .apply(elementwise_in(OP, y.rows, x.width)?),
        &[
            x.arg(),
            streams.arg(),
            post_mix.arg(),
            comb_mix.arg(),
            y.arg(),
            stated(OP, y.rows)?.arg(),
            stated(OP, fan)?.arg(),
            stated(OP, x.width)?.arg(),
        ],
    )
}

// MENLO-SEAM: `hc.collapse` has no honest source. The old plane never
// claimed the point — its CANON row (`hc.collapse` →
// `norm::hc_head_postprocess`) stayed a backlog line, because the device
// kernel reads a `[N, M]` f32 mix plane ("after GEMM") that no `Collapse`
// operand carries and no launcher ever staged. The kernel is carried and
// waiting; what is missing is the mix plane's producer.
pub fn collapse(
    _ctx: &Ctx,
    _streams: Tensor,
    _head_scale: Tensor,
    _head_base: Tensor,
    _stream_count: u32,
    _gate_eps: f32,
    _y: &mut Tensor,
) -> Result<(), KernelError> {
    Err(KernelError::Unsupported { op: "hc.collapse" })
}
