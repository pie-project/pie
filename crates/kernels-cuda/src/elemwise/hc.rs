//! `Hc`: hyper-connections — residual streams expanded, mixed by learned
//! gates, and folded back layer by layer. One entry per IR variant, all
//! over `elemwise/hc.cuh` (the launches lived beside the norms in
//! the old plane; the family gets its own file here because the IR gives it
//! one).
//!
//! The stream count `M` rides the row width — a `[N, M·H]` rectangle beside
//! a `[N, H]` one — and the mixers hold their `M` (or `M²`) coefficients in
//! registers and shared arrays, which is why [`MAX_HC_MULT`] is a hard
//! refusal and not a shape check.

use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "elemwise/hc.cuh";

const BLOCK: u32 = 256;

/// The largest stream count the mixers hold in registers (`hc_fold`'s
/// in-place read-before-write array) and shared gate vectors.
const MAX_HC_MULT: u32 = 8;

/// The stream fan `M`: how many `hidden`-wide streams the wide row holds.
/// The variants that state a count beside their rows assert agreement at
/// their own entries; `fold` states none.
fn stream_fan(op: &'static str, wide: u32, hidden: u32) -> Result<u32, Error> {
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
fn elementwise_in(op: &'static str, rows: u32, width: u32) -> Result<Launch, Error> {
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
pub fn expand(ctx: &Ctx, x: Tensor, streams: u32, y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.hc_expand";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(y.rows, x.rows, "the expansion lands one wide row per row");
    let fan = stream_fan(OP, y.width, x.width)?;
    debug_assert_eq!(
        fan, streams,
        "the row's stream fan is the count the statement states"
    );
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::elemwise::hc_expand<{t}>")))
            .apply(elementwise_in(OP, x.rows, x.width)?),
        &[
            x.arg(),
            y.arg(),
            stated(OP, x.rows)?.arg(),
            stated(OP, fan)?.arg(),
            stated(OP, x.width)?.arg(),
            // The staged-geometry seat: the region's live-rows word when a
            // body replay armed one, and the null seat (`ABSENT`) otherwise.
            ctx.stage(),
        ],
    )
}

/// RMS-normalises the wide stream row and widens it to f32 — the mix
/// coefficients derived downstream are too sensitive for a bf16 round-trip.
pub fn rmsnorm_f32(ctx: &Ctx, streams: Tensor, eps: f32, y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.hc_rmsnorm_f32";
    dtype_dispatch!(OP, streams.dtype, { Bf16 => () });
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` widens to f32");
    debug_assert!(
        y.rows == streams.rows && y.width == streams.width,
        "the normed rectangle is the stream rectangle"
    );
    nonzero(OP, "rows", y.rows)?;
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::elemwise::hc_rmsnorm_f32<::pie::bf16, 256>")
            .apply(Launch::per_row(y.rows, BLOCK)),
        &[
            streams.arg(),
            y.arg(),
            stated(OP, nonzero(OP, "the normed row's width", y.width)?)?.arg(),
            eps.arg(),
            // The staged-geometry seat: the region's live-rows word when a
            // body replay armed one, and the null seat (`ABSENT`) otherwise.
            ctx.stage(),
        ],
    )
}

/// The per-token mix row: `mixes = normed · hc_fn^T`, `[N, M·H]` against a
/// `[2M + M², M·H]` plane, all f32 — the layer's dynamic hyper plane
/// (`{attn,ffn}_hc.fn`) fired, which is what [`gates`] below splits.
///
/// **NOT `linear.matmul`**: both operands are f32 and the dense gemm points
/// are bf16; the sinkhorn is f32 by this family's design, which a bf16 detour
/// would undo. One block per `(row, mix column)`.
pub fn project(
    ctx: &Ctx,
    normed: Tensor,
    hc_fn: Tensor,
    stream_count: u32,
    mixes: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.hc_project";
    debug_assert!(
        normed.dtype == Dtype::F32 && hc_fn.dtype == Dtype::F32 && mixes.dtype == Dtype::F32,
        "`{OP}` projects an f32 row through an f32 plane into an f32 row"
    );
    debug_assert_eq!(
        mixes.rows, normed.rows,
        "the projection lands one mix row per stream row"
    );
    let fan = nonzero(OP, "the stream row this projection contracts", normed.width)?;
    if hc_fn.width != fan {
        return Err(refuse(
            OP,
            format!(
                "the dynamic plane contracts {} and the stream row is {fan} wide",
                hc_fn.width
            ),
        ));
    }
    if stream_count == 0 || stream_count > MAX_HC_MULT {
        return Err(refuse(
            OP,
            format!(
                "the stream count is {stream_count}, not one of the {MAX_HC_MULT} the \
                 mixers unroll"
            ),
        ));
    }
    let mix_hc = 2 * stream_count + stream_count * stream_count;
    if mixes.width != mix_hc || hc_fn.rows != mix_hc {
        return Err(refuse(
            OP,
            format!(
                "a {stream_count}-stream mix row is {mix_hc} wide; the plane lands {} \
                 rows into a {}-wide row",
                hc_fn.rows, mixes.width
            ),
        ));
    }
    let rows = nonzero(OP, "rows", mixes.rows)?;
    ctx.fire(
        OP,
        // One block per `(row, column)`, laid on ONE axis — the shader's own
        // point derives the pair, mirroring `elemwise/hc.metal`.
        Fire::at(FILE, "::pie::elemwise::hc_project<256>")
            .apply(Launch::grid([rows * mix_hc, 1, 1], [BLOCK, 1, 1])),
        &[
            normed.arg(),
            hc_fn.arg(),
            mixes.arg(),
            stated(OP, fan)?.arg(),
            stated(OP, mix_hc)?.arg(),
            // The staged-geometry seat: the region's live-rows word when a
            // body replay armed one, and the null seat (`ABSENT`) otherwise.
            ctx.stage(),
        ],
    )
}

/// Splits the mix row, Sinkhorn-normalises the combiner, and collapses the
/// `M` streams into the layer's input — one block per token, the gate
/// matrices landing beside it.
///
/// `normed` is the mix row [`project`] lands, `[N, 2M + M²]` — the stride the
/// kernel reads it at, which is now the width of what it is handed.
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
) -> Result<(), Error> {
    const OP: &str = "elementwise.hc_gates";
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
            symbol(&format!("::pie::elemwise::hc_gates<{t}, 256>")),
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
            // The staged-geometry seat: the region's live-rows word when a
            // body replay armed one, and the null seat (`ABSENT`) otherwise.
            ctx.stage(),
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
) -> Result<(), Error> {
    const OP: &str = "elementwise.hc_fold";
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
        Fire::at(FILE, symbol(&format!("::pie::elemwise::hc_fold<{t}>")))
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
            // The staged-geometry seat: the region's live-rows word when a
            // body replay armed one, and the null seat (`ABSENT`) otherwise.
            ctx.stage(),
        ],
    )
}

// `hc.collapse` was deleted with its IR variant (review R5): no plane could
// fire it honestly — `elemwise::hc_head_postprocess` still waits in the device
// text for a producer of its `[N, M]` f32 mix plane.

// ---- The gated-residual flavor (qwen4) ----------------------------------

/// `y[h] = meanₛ(σ(gates[s·H+h]) · normed[s·H+h])`: one `hidden`-wide layer
/// input mixed out of the stream fan under per-element sigmoid gates.
pub fn mix(ctx: &Ctx, gates: Tensor, normed: Tensor, streams: u32, y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.hc_mix";
    let t = dtype_dispatch!(OP, normed.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let fan = stream_fan(OP, normed.width, y.width)?;
    debug_assert_eq!(fan, streams, "the row's stream fan is the count the statement states");
    debug_assert!(
        gates.rows == normed.rows && gates.width == normed.width,
        "the gate rectangle is the stream rectangle"
    );
    debug_assert_eq!(y.rows, normed.rows, "the mix lands one narrow row per row");
    nonzero(OP, "rows", y.rows)?;
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::elemwise::hc_mix<{t}, 256>")))
            .apply(Launch::per_row(y.rows, BLOCK)),
        &[
            gates.arg(),
            normed.arg(),
            y.arg(),
            stated(OP, fan)?.arg(),
            stated(OP, y.width)?.arg(),
            // The staged-geometry seat: the region's live-rows word when a
            // body replay armed one, and the null seat (`ABSENT`) otherwise.
            ctx.stage(),
        ],
    )
}

/// `hyper[s·H+h] += 2·σ(gates[s]/streams)·o[h]`, in place on the wide
/// residual: the layer output injected back into every stream under its own
/// scalar gate.
pub fn inject(ctx: &Ctx, o: Tensor, gates: Tensor, streams: u32, hyper: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.hc_inject";
    let t = dtype_dispatch!(OP, o.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let fan = stream_fan(OP, hyper.width, o.width)?;
    debug_assert_eq!(fan, streams, "the row's stream fan is the count the statement states");
    debug_assert!(
        gates.rows == o.rows && gates.width == fan && hyper.rows == o.rows,
        "one gate logit per stream per row, one wide row per row"
    );
    nonzero(OP, "rows", o.rows)?;
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::elemwise::hc_inject<{t}, 256>")))
            .apply(Launch::per_row(o.rows, BLOCK)),
        &[
            o.arg(),
            gates.arg(),
            hyper.arg(),
            stated(OP, fan)?.arg(),
            stated(OP, o.width)?.arg(),
            // The staged-geometry seat: the region's live-rows word when a
            // body replay armed one, and the null seat (`ABSENT`) otherwise.
            ctx.stage(),
        ],
    )
}

/// The PLE gate: per stream, `σ(signed_sqrt(key·query / √H)) · value`. One
/// block per (row, stream), the grouped norms' own flattening.
pub fn ple_gate(
    ctx: &Ctx,
    key: Tensor,
    query: Tensor,
    value: Tensor,
    streams: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.ple_gate";
    let t = dtype_dispatch!(OP, key.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let fan = stream_fan(OP, key.width, value.width)?;
    debug_assert_eq!(fan, streams, "the row's stream fan is the count the statement states");
    debug_assert!(
        query.rows == key.rows && query.width == key.width,
        "the query rectangle is the key rectangle"
    );
    debug_assert!(
        value.rows == key.rows && y.rows == key.rows && y.width == key.width,
        "one value row and one wide answer per key row"
    );
    let rows = nonzero(OP, "rows", key.rows)?;
    let blocks = rows.checked_mul(fan).ok_or_else(|| {
        refuse(OP, format!("the grid will not launch: {rows} rows x {fan} streams"))
    })?;
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::elemwise::ple_gate<{t}, 256>")))
            .apply(Launch::per_row(blocks, BLOCK)),
        &[
            key.arg(),
            query.arg(),
            value.arg(),
            y.arg(),
            stated(OP, fan)?.arg(),
            stated(OP, value.width)?.arg(),
            // The staged-geometry seat: the region's live-rows word when a
            // body replay armed one, and the null seat (`ABSENT`) otherwise.
            ctx.stage(),
        ],
    )
}
