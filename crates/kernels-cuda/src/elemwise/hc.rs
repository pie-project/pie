use crate::error::Error;
use dtype::Dtype;

use crate::elemwise::hc_project::TILE_FROM;
use crate::elemwise::hc_project::prenorm_fits;
use crate::elemwise::hc_project::prenorm_launch;
use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, kernel, nonzero, refuse, stated};
use crate::tensor::Tensor;

const HC: &str = "elemwise/hc.cu";

const BLOCK: u32 = 256;

pub(crate) const MAX_HC_MULT: u32 = 8;

fn stream_fan(op: &'static str, wide: u32, hidden: u32) -> Result<u32, Error> {
    nonzero(op, "the hidden width", hidden)?;
    if wide == 0 || !wide.is_multiple_of(hidden) {
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

pub fn expand(ctx: &Ctx, x: Tensor, streams: u32, y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.hc_expand";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(y.rows, x.rows, "the expansion lands one wide row per row");
    let fan = stream_fan(OP, y.width, x.width)?;
    debug_assert_eq!(
        fan, streams,
        "the row's stream fan is the count the statement states"
    );
    ctx.fire(
        OP,
        Fire::at(kernel(HC, &format!("::pie::elemwise::hc_expand<{t}>")))
            .apply(elementwise_in(OP, x.rows, x.width)?),
        &[
            x.arg(),
            y.arg(),
            stated(OP, x.rows)?.arg(),
            stated(OP, fan)?.arg(),
            stated(OP, x.width)?.arg(),
            ctx.stage(),
        ],
    )
}

pub fn rmsnorm_f32(ctx: &Ctx, streams: Tensor, eps: f32, y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.hc_rmsnorm_f32";
    dtype_dispatch!(OP, streams.dtype, { Bf16 => () });
    debug_assert_eq!(y.dtype, Dtype::F32, "`{OP}` widens to f32");
    debug_assert!(
        y.rows == streams.rows && y.width == streams.width,
        "the normed rectangle is the stream rectangle"
    );
    nonzero(OP, "rows", y.rows)?;
    // Eight elements a thread, 1024 threads a row, where the row is whole
    // 16-byte runs: a 16384-wide stream row is two loads a thread, not 64
    // serial ones (DeepSeek-V4-Flash decode8: 28 us a call).
    let vec8 =
        y.width.is_multiple_of(8) && streams.ptr.is_multiple_of(16) && y.ptr.is_multiple_of(16);
    let (entry, block) = if vec8 {
        (
            "::pie::elemwise::hc_rmsnorm_f32_vec8<::pie::bf16, 1024>",
            1024,
        )
    } else {
        ("::pie::elemwise::hc_rmsnorm_f32<::pie::bf16, 256>", BLOCK)
    };
    ctx.fire(
        OP,
        Fire::at(kernel(HC, entry)).apply(Launch::per_row(y.rows, block)),
        &[
            streams.arg(),
            y.arg(),
            stated(OP, nonzero(OP, "the normed row's width", y.width)?)?.arg(),
            eps.arg(),
            ctx.stage(),
        ],
    )
}

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
    comb: bool,
    x: &mut Tensor,
    post_mix: &mut Tensor,
    comb_mix: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.hc_gates";
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
    // Without a combine the mix row is the pre and post logits alone, and
    // the combine landed is the identity the fold carries each stream by.
    let mix_hc = if comb { 2 * fan + fan * fan } else { 2 * fan };
    if normed.width != mix_hc {
        return Err(refuse(
            OP,
            format!(
                "a {fan}-stream mix row {} its combine is {mix_hc} wide and this one is {}",
                if comb { "with" } else { "without" },
                normed.width
            ),
        ));
    }
    nonzero(OP, "rows", x.rows)?;
    // The combined form with its Sinkhorn walk in one warp and the row's
    // width over blocks, enough of them to fill the SMs (DeepSeek-V4-Flash
    // decode8: 32 us a call on the block-barrier walk, a block a row).
    if comb && fan * fan <= 32 {
        let hidden = nonzero(OP, "the hidden width", x.width)?;
        let want = 2 * ctx.multiprocessors().unwrap_or(132);
        let vec8 = ctx.tuning().hc_tile && whole_runs(hidden, &[streams.ptr, x.ptr]);
        let columns = if vec8 { hidden / 8 } else { hidden };
        let slices = want.div_ceil(x.rows).clamp(1, columns.div_ceil(BLOCK));
        // Eight columns a thread: the pre-mix on `slices` blocks a row and the
        // Sinkhorn walk on one more, so the pre-mix waits on no walk.
        let (entry, grid_y) = match vec8 {
            true => (
                format!("::pie::elemwise::hc_gates_split<{t}, 256>"),
                slices + 1,
            ),
            false => (format!("::pie::elemwise::hc_gates_warp<{t}, 256>"), slices),
        };
        return ctx.fire(
            OP,
            Fire::at(kernel(HC, &entry)).apply(Launch::grid([x.rows, grid_y, 1], [BLOCK, 1, 1])),
            &[
                normed.arg(),
                scale.arg(),
                base.arg(),
                streams.arg(),
                post_mix.arg(),
                comb_mix.arg(),
                x.arg(),
                stated(OP, fan)?.arg(),
                stated(OP, hidden)?.arg(),
                gate_eps.arg(),
                alpha.arg(),
                stated(OP, sinkhorn)?.arg(),
                ctx.stage(),
            ],
        );
    }
    ctx.fire(
        OP,
        Fire::at(kernel(HC, &format!("::pie::elemwise::hc_gates<{t}, 256>")))
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
            comb.arg(),
            ctx.stage(),
        ],
    )
}

/// The gates' scale and base planes and constants, as `HcGates` states them.
#[derive(Clone, Copy)]
pub struct Gate {
    pub scale: Tensor,
    pub base: Tensor,
    pub gate_eps: f32,
    pub alpha: f32,
    pub sinkhorn: u32,
}

/// Whether `hc_gates_norm` takes these: a combine of at most 32 entries over
/// rows a whole number of 256 up to 4096 wide, the weight the streams'
/// element.
fn gates_norm_fits(
    mixes: &Tensor,
    streams: &Tensor,
    stream_count: u32,
    weight: &Tensor,
    y: &Tensor,
) -> bool {
    let hidden = y.width;
    matches!(streams.dtype, Dtype::Bf16 | Dtype::F16)
        && stream_count != 0
        && stream_count * stream_count <= 32
        && streams.width == stream_count * hidden
        && mixes.width == 2 * stream_count + stream_count * stream_count
        && weight.dtype == streams.dtype
        && y.dtype == streams.dtype
        && hidden.is_multiple_of(BLOCK)
        && (1..=16).contains(&(hidden / BLOCK))
}

#[allow(clippy::too_many_arguments)]
fn gates_norm_launch(
    ctx: &Ctx,
    op: &'static str,
    mixes: crate::jit::ArgValue,
    partial: Option<(u32, u32, f32)>,
    streams: &Tensor,
    stream_count: u32,
    gate: Gate,
    weight: &Tensor,
    norm_eps: f32,
    held: Option<[u64; 2]>,
    post_mix: &mut Tensor,
    comb_mix: &mut Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    let t = dtype_dispatch!(op, streams.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    nonzero(op, "rows", y.rows)?;
    let (slices, fan, mix_eps) = partial.unwrap_or((1, streams.width, 0.0));
    // A row of 1024 or wider pre-mixes 1024 threads a block, its squares
    // still folded in the 256-thread norm's order.
    let threads = if y.width.is_multiple_of(1024) {
        1024
    } else {
        BLOCK
    };
    ctx.fire(
        op,
        Fire::at(kernel(
            HC,
            &format!(
                "::pie::elemwise::hc_gates_norm<{t}, {}, {}, {threads}, {}>",
                y.width / BLOCK,
                partial.is_some(),
                held.is_some()
            ),
        ))
        .apply(Launch::grid([y.rows, 2, 1], [threads, 1, 1]).programmatic(ctx.tuning().pdl)),
        &[
            mixes,
            gate.scale.arg(),
            gate.base.arg(),
            streams.arg(),
            post_mix.arg(),
            comb_mix.arg(),
            weight.arg(),
            y.arg(),
            stated(op, stream_count)?.arg(),
            gate.gate_eps.arg(),
            gate.alpha.arg(),
            stated(op, gate.sinkhorn)?.arg(),
            norm_eps.arg(),
            stated(op, slices)?.arg(),
            stated(op, fan)?.arg(),
            mix_eps.arg(),
            crate::jit::ArgValue::Ptr(held.map_or(0, |[codes, _]| codes)),
            crate::jit::ArgValue::Ptr(held.map_or(0, |[_, atoms]| atoms)),
            ctx.stage(),
        ],
    )
}

/// The combined gates and the RMS norm over the row they pre-mix in one
/// launch (`hc_gates_norm`), the pre-mixed row never landed: every value the
/// pair lands, bit for bit. `Ok(false)` where the shapes are not the
/// kernel's (no combine, more than 32 combine entries, a width not a whole
/// number of 256 up to 4096, a weight of another element): the caller runs
/// the two.
#[allow(clippy::too_many_arguments)]
pub fn gates_norm(
    ctx: &Ctx,
    mixes: Tensor,
    streams: Tensor,
    stream_count: u32,
    gate: Gate,
    weight: Tensor,
    norm_eps: f32,
    post_mix: &mut Tensor,
    comb_mix: &mut Tensor,
    y: &mut Tensor,
) -> Result<bool, Error> {
    if !gates_norm_fits(&mixes, &streams, stream_count, &weight, y) {
        return Ok(false);
    }
    gates_norm_launch(
        ctx,
        "fused.hc_gates_norm",
        mixes.arg(),
        None,
        &streams,
        stream_count,
        gate,
        &weight,
        norm_eps,
        None,
        post_mix,
        comb_mix,
        y,
    )?;
    Ok(true)
}

pub fn fold(
    ctx: &Ctx,
    x: Tensor,
    streams: Tensor,
    post_mix: Tensor,
    comb_mix: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.hc_fold";
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
    // A prefill's rows of four streams a block a row (`hc_fold_rows4`, under
    // `kernel.elemwise.hc_mma`): the gates read once a thread, the same sums
    // in the same order.
    if ctx.tuning().hc_mma
        && y.rows >= 128
        && fan == 4
        && whole_runs(x.width, &[x.ptr, streams.ptr, y.ptr])
        && [post_mix.ptr, comb_mix.ptr].iter().all(|at| at % 16 == 0)
    {
        return ctx.fire(
            OP,
            Fire::at(kernel(HC, &format!("::pie::elemwise::hc_fold_rows4<{t}>")))
                .apply(Launch::per_row(y.rows, 128)),
            &[
                x.arg(),
                streams.arg(),
                post_mix.arg(),
                comb_mix.arg(),
                y.arg(),
                stated(OP, y.rows)?.arg(),
                stated(OP, x.width)?.arg(),
                ctx.stage(),
            ],
        );
    }
    // Eight columns a thread where the rows are whole 16-byte runs: the same
    // sums in the same order.
    let vec8 = ctx.tuning().hc_tile && whole_runs(x.width, &[x.ptr, streams.ptr, y.ptr]);
    let (entry, width) = if vec8 {
        (format!("::pie::elemwise::hc_fold_vec8<{t}>"), x.width / 8)
    } else {
        (format!("::pie::elemwise::hc_fold<{t}>"), x.width)
    };
    // The vec8 fold waits on its producer in-kernel (`kernel.pdl`).
    ctx.fire(
        OP,
        Fire::at(kernel(HC, &entry))
            .apply(elementwise_in(OP, y.rows, width)?.programmatic(ctx.tuning().pdl && vec8)),
        &[
            x.arg(),
            streams.arg(),
            post_mix.arg(),
            comb_mix.arg(),
            y.arg(),
            stated(OP, y.rows)?.arg(),
            stated(OP, fan)?.arg(),
            stated(OP, x.width)?.arg(),
            ctx.stage(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn collapse(
    ctx: &Ctx,
    mixes: Tensor,
    streams: Tensor,
    scale: Tensor,
    base: Tensor,
    stream_count: u32,
    hc_eps: f32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.hc_collapse";
    let t = dtype_dispatch!(OP, streams.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(mixes.dtype, Dtype::F32, "`{OP}` reads an f32 mix row");
    debug_assert_eq!(scale.dtype, Dtype::F32, "`{OP}` reads an f32 mix scale");
    debug_assert_eq!(base.dtype, Dtype::F32, "`{OP}` reads f32 mix bases");
    debug_assert_eq!(y.dtype, streams.dtype, "`{OP}` lands the streams' element");
    let fan = stream_fan(OP, streams.width, y.width)?;
    debug_assert_eq!(
        fan, stream_count,
        "the row's stream fan is the count the statement states"
    );
    if mixes.width != fan || mixes.rows != y.rows {
        return Err(refuse(
            OP,
            format!(
                "the trunk collapse folds {fan} streams under a {}-wide mix row over {} of {} rows",
                mixes.width, mixes.rows, y.rows
            ),
        ));
    }
    nonzero(OP, "rows", y.rows)?;
    ctx.fire(
        OP,
        Fire::at(kernel(
            HC,
            &format!("::pie::elemwise::hc_head_postprocess<{t}, 256>"),
        ))
        .apply(Launch::per_row(y.rows, BLOCK)),
        &[
            mixes.arg(),
            scale.arg(),
            base.arg(),
            streams.arg(),
            y.arg(),
            stated(OP, fan)?.arg(),
            stated(OP, nonzero(OP, "the hidden width", y.width)?)?.arg(),
            hc_eps.arg(),
            ctx.stage(),
        ],
    )
}

/// Whether rows `width` wide (two-byte elements) at these planes split into
/// whole aligned 16-byte runs.
/// The threads a block of the spread 16-byte mix and inject.
const SPREAD: u32 = 64;

fn whole_runs(width: u32, planes: &[u64]) -> bool {
    width.is_multiple_of(8) && planes.iter().all(|p| p % 16 == 0)
}

pub fn mix(
    ctx: &Ctx,
    gates: Tensor,
    normed: Tensor,
    streams: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.hc_mix";
    let t = dtype_dispatch!(OP, normed.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let fan = stream_fan(OP, normed.width, y.width)?;
    debug_assert_eq!(
        fan, streams,
        "the row's stream fan is the count the statement states"
    );
    debug_assert!(
        gates.rows == normed.rows && gates.width == normed.width,
        "the gate rectangle is the stream rectangle"
    );
    debug_assert_eq!(y.rows, normed.rows, "the mix lands one narrow row per row");
    nonzero(OP, "rows", y.rows)?;
    // Whole 16-byte runs where the width and the three planes allow them:
    // the same sums, element for element.
    let vec8 = whole_runs(y.width, &[gates.ptr, normed.ptr, y.ptr]);
    // The 16-byte form spreads a row's columns over blocks of 64 threads, so
    // a decode fire's few rows still fill the SMs; each element's sum is the
    // same one, in the same order.
    let (entry, launch) = if vec8 {
        (
            format!("::pie::elemwise::hc_mix_vec8<{t}, {SPREAD}>"),
            Launch::grid([y.rows, (y.width / 8).div_ceil(SPREAD), 1], [SPREAD, 1, 1]),
        )
    } else {
        (
            format!("::pie::elemwise::hc_mix<{t}, 256>"),
            Launch::per_row(y.rows, BLOCK),
        )
    };
    ctx.fire(
        OP,
        Fire::at(kernel(HC, &entry)).apply(launch),
        &[
            gates.arg(),
            normed.arg(),
            y.arg(),
            stated(OP, fan)?.arg(),
            stated(OP, y.width)?.arg(),
            ctx.stage(),
        ],
    )
}

pub fn inject(
    ctx: &Ctx,
    o: Tensor,
    gates: Tensor,
    streams: u32,
    hyper: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.hc_inject";
    let t = dtype_dispatch!(OP, o.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let fan = stream_fan(OP, hyper.width, o.width)?;
    debug_assert_eq!(
        fan, streams,
        "the row's stream fan is the count the statement states"
    );
    debug_assert!(
        gates.rows == o.rows && gates.width == fan && hyper.rows == o.rows,
        "one gate logit per stream per row, one wide row per row"
    );
    nonzero(OP, "rows", o.rows)?;
    let vec8 = whole_runs(o.width, &[o.ptr, hyper.ptr]);
    // As `mix`: a row's streams over blocks of 64 threads, the same sums.
    let (entry, launch) = if vec8 {
        (
            format!("::pie::elemwise::hc_inject_vec8<{t}, {SPREAD}>"),
            Launch::grid(
                [o.rows, (fan * o.width / 8).div_ceil(SPREAD), 1],
                [SPREAD, 1, 1],
            ),
        )
    } else {
        (
            format!("::pie::elemwise::hc_inject<{t}, 256>"),
            Launch::per_row(o.rows, BLOCK),
        )
    };
    ctx.fire(
        OP,
        Fire::at(kernel(HC, &entry)).apply(launch),
        &[
            o.arg(),
            gates.arg(),
            hyper.arg(),
            stated(OP, fan)?.arg(),
            stated(OP, o.width)?.arg(),
            ctx.stage(),
        ],
    )
}

pub fn ple_gate(
    ctx: &Ctx,
    key: Tensor,
    query: Tensor,
    value: Tensor,
    streams: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.ple_gate";
    let t = dtype_dispatch!(OP, key.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let fan = stream_fan(OP, key.width, value.width)?;
    debug_assert_eq!(
        fan, streams,
        "the row's stream fan is the count the statement states"
    );
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
        refuse(
            OP,
            format!("the grid will not launch: {rows} rows x {fan} streams"),
        )
    })?;
    ctx.fire(
        OP,
        Fire::at(kernel(HC, &format!("::pie::elemwise::ple_gate<{t}, 256>")))
            .apply(Launch::per_row(blocks, BLOCK)),
        &[
            key.arg(),
            query.arg(),
            value.arg(),
            y.arg(),
            stated(OP, fan)?.arg(),
            stated(OP, value.width)?.arg(),
            ctx.stage(),
        ],
    )
}

/// Whether `pre` lands these and with them the e4m3 operand planes a held
/// GEMM reads: the caller holds the planes only then, so a GEMM never reads
/// planes nothing wrote.
#[must_use]
pub fn pre_emits(
    streams: &Tensor,
    hc_fn: &Tensor,
    mixes: &Tensor,
    stream_count: u32,
    weight: &Tensor,
    y: &Tensor,
) -> bool {
    mixes.rows < PRE_WIDE_FROM
        && prenorm_fits(streams, hc_fn, mixes, stream_count)
        && gates_norm_fits(mixes, streams, stream_count, weight, y)
        && y.width.is_multiple_of(256)
}

/// The rows from which a sublayer's pre-mix runs as the one-pass projection
/// then the gates and norm apart: `pre`'s block-a-row pre-mix wins a
/// decode's few rows (DeepSeek-V4-Flash, 8 rows: 13.2 us where the parts
/// take ~17.7) and loses a prefill's (16384 rows: 1054 us where they take
/// ~836).
pub const PRE_WIDE_FROM: u32 = TILE_FROM;

/// A sublayer's whole pre-mix in two launches: the stream norm and mix
/// projection (`prenorm_launch`), then the gates and the RMS norm over the
/// row they pre-mix (`hc_gates_norm`), which sums the projection's slices
/// itself where it left several. With `held` (a GEMM's e4m3 1x128 operand
/// planes, codes then atoms) the normed row also lands there, as the GEMM's
/// quantize would land it. `Ok(false)` where the shapes are not the
/// kernels': the caller runs the four ops.
#[allow(clippy::too_many_arguments)]
pub fn pre(
    ctx: &Ctx,
    streams: Tensor,
    mix_eps: f32,
    hc_fn: Tensor,
    stream_count: u32,
    mixes: &mut Tensor,
    gate: Gate,
    weight: Tensor,
    norm_eps: f32,
    held: Option<[u64; 2]>,
    post_mix: &mut Tensor,
    comb_mix: &mut Tensor,
    y: &mut Tensor,
) -> Result<bool, Error> {
    const OP: &str = "fused.hc_pre";
    if !prenorm_fits(&streams, &hc_fn, mixes, stream_count)
        || !gates_norm_fits(mixes, &streams, stream_count, &weight, y)
    {
        return Ok(false);
    }
    let landed = prenorm_launch(ctx, OP, &streams, mix_eps, &hc_fn, mixes)?;
    gates_norm_launch(
        ctx,
        OP,
        crate::jit::ArgValue::Ptr(landed.out),
        (landed.slices > 1).then_some((landed.slices, streams.width, mix_eps)),
        &streams,
        stream_count,
        gate,
        &weight,
        norm_eps,
        held.filter(|_| y.width.is_multiple_of(256)),
        post_mix,
        comb_mix,
        y,
    )?;
    Ok(true)
}

#[cfg(feature = "cuda")]
const HC_SM100: &str = "elemwise/hc_sm100.cu";

/// `pie::elemwise::HcMegaParams`, field for field.
#[cfg(feature = "cuda")]
#[repr(C)]
struct MegaParams {
    x: u64,
    residual: u64,
    post_in: u64,
    comb_in: u64,
    folded: u64,
    scale: u64,
    base: u64,
    post_out: u64,
    comb_out: u64,
    weight: u64,
    y: u64,
    codes: u64,
    atoms: u64,
    part: u64,
    sync: u64,
    maps: u64,
    win: u64,
    rows: i32,
    splits: i32,
    sinkhorn: i32,
    mix_eps: f32,
    gate_eps: f32,
    alpha: f32,
    norm_eps: f32,
}

/// `pie::elemwise::hc_mega_smem()`: three stages of a task's streams
/// (32 KB), x (8 KB), the weights' f32 box and its bf16 halves (12 KB
/// each), two tasks' gates (10 KB each), then the rebased maps, barriers
/// and scratch (8 KB) and the 1 KB alignment.
#[cfg(feature = "cuda")]
const MEGA_SMEM: u32 = 3 * (32768 + 8192 + 12288 + 12288) + 2 * 10240 + 8192 + 1024;

/// The row maps a block rebases at a staged window (`kMegaMaps`).
#[cfg(feature = "cuda")]
const MEGA_MAPS: u32 = 5;

/// `pie::elemwise::kMegaThreads`.
#[cfg(feature = "cuda")]
const MEGA_THREADS: u32 = 480;

/// A task's rows (the MMA's M) and a K-block's columns of each stream.
#[cfg(feature = "cuda")]
const MEGA_ROWS: u32 = 128;
#[cfg(feature = "cuda")]
const MEGA_COLS: u32 = 32;

/// The most splits a row is cut into: a task keeps two K-blocks at least,
/// which `hc_fold_pre_tc5`'s one slot of pending gates asks.
#[cfg(feature = "cuda")]
const MEGA_SPLITS: u32 = 64;

/// The tasks a launch aims for, a multiple of the SMs: enough that the
/// dynamic queue's tail is a fraction of a task.
#[cfg(feature = "cuda")]
const MEGA_TASKS_PER_SM: u32 = 4;

/// The counters' words before the per-block arrivals
/// (`pie::elemwise::kSyncBlocks`).
#[cfg(feature = "cuda")]
const MEGA_SYNC_HEAD: u32 = 8;

/// A sublayer's fold and the next sublayer's whole pre-mix as
/// `hc_fold_pre_tc5` takes them.
#[derive(Clone, Copy)]
pub struct Seam {
    /// The sublayer's output, `[N, H]`.
    pub x: Tensor,
    /// The streams it folds into, `[N, 4H]`.
    pub streams: Tensor,
    /// The sublayer's post and combine gates, `[N, 4]` and `[N, 16]`.
    pub post_mix: Tensor,
    pub comb_mix: Tensor,
    /// The next sublayer's stream-norm eps and mix projection, `[24, 4H]`.
    pub mix_eps: f32,
    pub hc_fn: Tensor,
    pub stream_count: u32,
    pub gate: Gate,
    /// Its input norm's weight `[H]` and eps.
    pub weight: Tensor,
    pub norm_eps: f32,
}

/// Whether `hc_fold_pre_tc5` takes these: tcgen05, four streams whose
/// width is whole 1024s, the 24-mix plane, a norm weight of the row's
/// width in any rectangle (a load hands its `[H]` as H rows of one), bf16
/// rows and f32 gates 16-byte aligned, `kernel.elemwise.hc_mega_from` to
/// `hc_mega_until` rows.
#[cfg(feature = "cuda")]
fn fold_pre_fits(ctx: &Ctx, s: &Seam, folded: &Tensor, y: &Tensor) -> bool {
    let hidden = y.width;
    let wide = 4 * hidden;
    crate::linear::gemm_sm100::supported()
        && y.rows != 0
        && (ctx.tuning().hc_mega_from..=ctx.tuning().hc_mega_until).contains(&y.rows)
        && s.stream_count == 4
        && hidden != 0
        && hidden.is_multiple_of(1024)
        && [
            s.x.dtype,
            s.streams.dtype,
            folded.dtype,
            s.weight.dtype,
            y.dtype,
        ]
        .iter()
        .all(|d| *d == Dtype::Bf16)
        && [s.post_mix.dtype, s.comb_mix.dtype, s.hc_fn.dtype]
            .iter()
            .all(|d| *d == Dtype::F32)
        && s.x.width == hidden
        && s.streams.width == wide
        && folded.width == wide
        && s.post_mix.width == 4
        && s.comb_mix.width == 16
        && s.hc_fn.rows == 24
        && s.hc_fn.width == wide
        && s.weight.elements() == u64::from(hidden)
        && [
            s.x.ptr,
            s.streams.ptr,
            folded.ptr,
            s.post_mix.ptr,
            s.comb_mix.ptr,
            s.hc_fn.ptr,
            s.weight.ptr,
            y.ptr,
        ]
        .iter()
        .all(|at| at.is_multiple_of(16))
}

/// The splits a row is cut into for `rows` rows on `sms` multiprocessors:
/// about `MEGA_TASKS_PER_SM` tasks an SM, every task two K-blocks at least.
#[cfg(feature = "cuda")]
fn mega_splits(rows: u32, hidden: u32, sms: u32) -> u32 {
    let blocks = rows.div_ceil(MEGA_ROWS);
    let most = (hidden / MEGA_COLS / 2).clamp(1, MEGA_SPLITS);
    (MEGA_TASKS_PER_SM * sms).div_ceil(blocks).clamp(1, most)
}

/// A sublayer's fold (`fold`, bit for bit) and the next sublayer's whole
/// pre-mix (`pre`: the gates and the normed row `y`, to the projection's
/// rounding) in one tcgen05 launch, `folded` the folded streams. With
/// `held` (a GEMM's e4m3 1x128 planes, codes then atoms) the normed row
/// also lands there. `Ok(false)` where the shapes or the profile are not
/// the kernel's (`fold_pre_fits`): the caller runs the fold and `pre`.
pub fn fold_pre(
    ctx: &Ctx,
    seam: Seam,
    folded: &mut Tensor,
    held: Option<[u64; 2]>,
    post_mix: &mut Tensor,
    comb_mix: &mut Tensor,
    y: &mut Tensor,
) -> Result<bool, Error> {
    #[cfg(feature = "cuda")]
    {
        const OP: &str = "fused.hc_fold_pre";
        use crate::jit::ArgValue;
        use crate::linear::gemm_sm100::{Map, encode};
        use cudarc::driver::sys as dr;
        if !fold_pre_fits(ctx, &seam, folded, y) {
            return Ok(false);
        }
        let rows = y.rows;
        let hidden = y.width;
        let sms = ctx.multiprocessors().unwrap_or(148);
        let splits = mega_splits(rows, hidden, sms);
        let blocks = rows.div_ceil(MEGA_ROWS);
        let part = ctx.scratch_shared(
            OP,
            "elemwise.hc_mega_partial",
            (blocks * splits * MEGA_ROWS * 32) as usize * core::mem::size_of::<f32>(),
        )? as usize as u64;
        let sync = ctx.scratch_stream_zeroed(
            OP,
            "elemwise.hc_mega_counts",
            (MEGA_SYNC_HEAD + blocks) as usize * core::mem::size_of::<u32>(),
        )? as usize as u64;
        let grid = (blocks * splits).max(rows).min(sms);
        // Each block's row maps rebased at a staged window.
        let maps = ctx.scratch_shared(
            OP,
            "elemwise.hc_mega_maps",
            (grid * MEGA_MAPS * 128) as usize,
        )? as usize as u64;
        // The weights' 32-column box of every mix of every stream: x the
        // column, y the mix, z the stream.
        let map_fn = encode(Map {
            ptr: seam.hc_fn.ptr,
            ty: dr::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
            dims: [hidden, 24, 4],
            strides: [u64::from(4 * hidden) * 4, u64::from(hidden) * 4],
            boxes: [MEGA_COLS, 24, 4],
            swizzle: dr::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        })?;
        // x's and the streams' 32-column boxes of 128 rows, the streams'
        // stream-major (x the column, y the row, z the stream), 64-byte
        // swizzled as the MMA reads them.
        let bf16 = dr::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
        let sw64 = dr::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_64B;
        let map_x = encode(Map {
            ptr: seam.x.ptr,
            ty: bf16,
            dims: [hidden, rows, 0],
            strides: [u64::from(hidden) * 2, 0],
            boxes: [MEGA_COLS, MEGA_ROWS, 1],
            swizzle: sw64,
        })?;
        let streams = |ptr: u64| {
            encode(Map {
                ptr,
                ty: bf16,
                dims: [hidden, rows, 4],
                strides: [u64::from(4 * hidden) * 2, u64::from(hidden) * 2],
                boxes: [MEGA_COLS, MEGA_ROWS, 4],
                swizzle: sw64,
            })
        };
        let (map_res, map_out) = (streams(seam.streams.ptr)?, streams(folded.ptr)?);
        // A task's rows of the post and combine gates, unswizzled.
        let gates = |ptr: u64, width: u32| {
            encode(Map {
                ptr,
                ty: dr::CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_FLOAT32,
                dims: [width, rows, 0],
                strides: [u64::from(width) * 4, 0],
                boxes: [width, MEGA_ROWS, 1],
                swizzle: dr::CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
            })
        };
        let (map_post, map_comb) = (gates(seam.post_mix.ptr, 4)?, gates(seam.comb_mix.ptr, 16)?);
        let win = match ctx.stage() {
            ArgValue::Ptr(at) => at,
            _ => 0,
        };
        let [codes, atoms] = held.unwrap_or([0, 0]);
        let params = MegaParams {
            x: seam.x.ptr,
            residual: seam.streams.ptr,
            post_in: seam.post_mix.ptr,
            comb_in: seam.comb_mix.ptr,
            folded: folded.ptr,
            scale: seam.gate.scale.ptr,
            base: seam.gate.base.ptr,
            post_out: post_mix.ptr,
            comb_out: comb_mix.ptr,
            weight: seam.weight.ptr,
            y: y.ptr,
            codes,
            atoms,
            part,
            sync,
            maps,
            win,
            rows: stated(OP, rows)?,
            splits: stated(OP, splits)?,
            sinkhorn: stated(OP, seam.gate.sinkhorn)?,
            mix_eps: seam.mix_eps,
            gate_eps: seam.gate.gate_eps,
            alpha: seam.gate.alpha,
            norm_eps: seam.norm_eps,
        };
        ctx.fire(
            OP,
            Fire::at(kernel(
                HC_SM100,
                &format!(
                    "::pie::elemwise::hc_fold_pre_tc5<{hidden}, {}>",
                    held.is_some()
                ),
            ))
            .apply(
                Launch::grid([grid, 1, 1], [MEGA_THREADS, 1, 1])
                    .smem(MEGA_SMEM)
                    .programmatic(ctx.tuning().pdl),
            ),
            &[
                ArgValue::of(&map_fn),
                ArgValue::of(&map_x),
                ArgValue::of(&map_res),
                ArgValue::of(&map_out),
                ArgValue::of(&map_post),
                ArgValue::of(&map_comb),
                ArgValue::of(&params),
            ],
        )?;
        Ok(true)
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = (ctx, seam, folded, held, post_mix, comb_mix, y);
        Ok(false)
    }
}
