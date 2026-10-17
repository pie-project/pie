use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "elemwise/hc.cuh";

const BLOCK: u32 = 256;

const MAX_HC_MULT: u32 = 8;

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
            ctx.stage(),
        ],
    )
}

pub fn rmsnorm_f32(ctx: &Ctx, streams: Tensor, eps: f32, y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.hc_rmsnorm_f32";
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
    let vec8 = y.width.is_multiple_of(8) && streams.ptr % 16 == 0 && y.ptr % 16 == 0;
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
        Fire::at(FILE, entry).apply(Launch::per_row(y.rows, block)),
        &[
            streams.arg(),
            y.arg(),
            stated(OP, nonzero(OP, "the normed row's width", y.width)?)?.arg(),
            eps.arg(),
            ctx.stage(),
        ],
    )
}

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
    let mix_hc = hc_fn.rows;
    let layer_row = 2 * stream_count + stream_count * stream_count;
    let independent_row = 2 * stream_count;
    if mixes.width != mix_hc
        || (mix_hc != layer_row && mix_hc != independent_row && mix_hc != stream_count)
    {
        return Err(refuse(
            OP,
            format!(
                "a {stream_count}-stream mix row is {layer_row} wide ({independent_row} without \
                 its combine) and a trunk collapse row is {stream_count}; the plane lands {} \
                 rows into a {}-wide row",
                hc_fn.rows, mixes.width
            ),
        ));
    }
    let rows = nonzero(OP, "rows", mixes.rows)?;
    // Four rows a block, a warp's mix columns every eighth (three of a
    // layer's 24), the row's width cut into slices across blocks where the
    // rows alone would leave the SMs idle (a decode fire). The kernel below
    // read a row once a mix column and walked it alone (DeepSeek-V4-Flash:
    // 8.3 s of a 32x4096 prefill, 3.3 ms a decode8 step).
    if ctx.tuning().hc_tile
        && fan.is_multiple_of(TILE_K)
        && rows >= TILE_FROM
        && mix_hc <= TILE_COLS
        && normed.ptr % 16 == 0
        && hc_fn.ptr % 16 == 0
    {
        return project_tile(ctx, OP, normed, hc_fn, mixes, rows, fan, mix_hc);
    }
    if fan.is_multiple_of(4) && mix_hc <= 8 * PROJECT_COLS {
        return project_split(ctx, OP, normed, hc_fn, mixes, rows, fan, mix_hc);
    }
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::elemwise::hc_project<256>")
            .apply(Launch::grid([rows * mix_hc, 1, 1], [BLOCK, 1, 1])),
        &[
            normed.arg(),
            hc_fn.arg(),
            mixes.arg(),
            stated(OP, fan)?.arg(),
            stated(OP, mix_hc)?.arg(),
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
        // The pre-mix on `slices` blocks a row and the Sinkhorn walk on one
        // more, so the pre-mix waits on no walk.
        if vec8 {
            return ctx.fire(
                OP,
                Fire::at(
                    FILE,
                    symbol(&format!("::pie::elemwise::hc_gates_split<{t}, 256>")),
                )
                .apply(Launch::grid([x.rows, slices + 1, 1], [BLOCK, 1, 1])),
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
        return ctx.fire(
            OP,
            Fire::at(
                FILE,
                symbol(&format!("::pie::elemwise::hc_gates_warp<{t}, 256, {vec8}>")),
            )
            .apply(Launch::grid([x.rows, slices, 1], [BLOCK, 1, 1])),
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
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::elemwise::hc_gates_norm<{t}, {}, {}, {threads}, {}>",
                y.width / BLOCK,
                partial.is_some(),
                held.is_some()
            )),
        )
        .apply(Launch::grid([y.rows, 2, 1], [threads, 1, 1])),
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
        "elementwise.hc_gates_norm",
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
            Fire::at(
                FILE,
                symbol(&format!("::pie::elemwise::hc_fold_rows4<{t}>")),
            )
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
    let (entry, width) =
        if ctx.tuning().hc_tile && whole_runs(x.width, &[x.ptr, streams.ptr, y.ptr]) {
            (format!("::pie::elemwise::hc_fold_vec8<{t}>"), x.width / 8)
        } else {
            (format!("::pie::elemwise::hc_fold<{t}>"), x.width)
        };
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&entry)).apply(elementwise_in(OP, y.rows, width)?),
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
    const OP: &str = "elementwise.hc_collapse";
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
        Fire::at(
            FILE,
            symbol(&format!("::pie::elemwise::hc_head_postprocess<{t}, 256>")),
        )
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
    const OP: &str = "elementwise.hc_mix";
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
        Fire::at(FILE, symbol(&entry)).apply(launch),
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
    const OP: &str = "elementwise.hc_inject";
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
        Fire::at(FILE, symbol(&entry)).apply(launch),
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
    const OP: &str = "elementwise.ple_gate";
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
        Fire::at(
            FILE,
            symbol(&format!("::pie::elemwise::ple_gate<{t}, 256>")),
        )
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

const PROJECT_ROWS: u32 = 4;
const PROJECT_COLS: u32 = 3;

/// `hc_project_tile`'s rows a block, column ceiling, shared tile, thread
/// groups over the tile and tiles in flight: 64 x 4 x 64 x 2 projects
/// DeepSeek-V4-Flash's 16384 rows of 16384 into 24 in 427 us where the
/// four-row split walk took 1339 (64 x 4 x 32 x 3: 506; 64 x 4 x 128 x 2,
/// past the default shared memory: 409).
const TILE_ROWS: u32 = 64;
const TILE_COLS: u32 = 24;
const TILE_K: u32 = 64;
const TILE_STAGES: u32 = 2;
const TILE_GROUPS: u32 = 4;
/// Below this many rows the four-row split walk serves: a tile of 64 would
/// mostly repeat its last row.
const TILE_FROM: u32 = 128;

/// Whether `hc_prenorm_project_tile` (and its tensor-core form) take
/// these planes: bf16 streams whole 64-element tiles wide, an f32 plane of
/// at most 24 mix columns over them, 16-byte aligned.
fn prenorm_fits(streams: &Tensor, hc_fn: &Tensor, mixes: &Tensor, stream_count: u32) -> bool {
    streams.dtype == Dtype::Bf16
        && hc_fn.dtype == Dtype::F32
        && mixes.dtype == Dtype::F32
        && hc_fn.width == streams.width
        && streams.width.is_multiple_of(TILE_K)
        && hc_fn.rows <= TILE_COLS
        && mixes.width == hc_fn.rows
        && mixes.rows == streams.rows
        && stream_count != 0
        && stream_count <= MAX_HC_MULT
        && [streams.ptr, hc_fn.ptr].iter().all(|at| at % 16 == 0)
}

/// Where `prenorm_launch` landed the projection: the mixes themselves, or
/// `slices` partial slices at `out`.
struct Landed {
    out: u64,
    slices: u32,
}

/// The projection's tile launch: a prefill's rows whole, a decode's few cut
/// across the row.
fn prenorm_launch(
    ctx: &Ctx,
    op: &'static str,
    streams: &Tensor,
    eps: f32,
    hc_fn: &Tensor,
    mixes: &Tensor,
) -> Result<Landed, Error> {
    // At most 8192 partial rows (800 KiB at 24 mixes), in shared scratch.
    const PARTIAL_ROWS: u32 = 8192;
    let fan = streams.width;
    let mix_hc = hc_fn.rows;
    let rows = nonzero(op, "rows", mixes.rows)?;
    let wide = rows >= TILE_FROM;
    // A decode's few rows (one block of them) cut the row 128 columns a
    // block: the 128 slices that leaves read the projection plane, cold at a
    // decode, over as many SMs (DeepSeek-V4 at 8 rows, planes cold: 7.0 ->
    // 5.0 us), and the fold's warps (or the gates', under `pre`) take them.
    // Past one block of rows, 256 columns a block, the 64 slices measured
    // no slower.
    let floor = if wide {
        512
    } else if rows <= 32 {
        128
    } else {
        256
    };
    if wide && ctx.tuning().hc_mma {
        return prenorm_launch_mma(ctx, op, streams, eps, hc_fn, mixes);
    }
    let (br, kt) = (if wide { TILE_ROWS } else { 32 }, TILE_K);
    let row_blocks = rows.div_ceil(br);
    let want = 2 * ctx.multiprocessors().unwrap_or(132);
    let slices = want
        .div_ceil(row_blocks)
        .clamp(1, (fan / floor).max(1))
        .min((PARTIAL_ROWS / rows).max(1));
    let slice = fan.div_ceil(slices).next_multiple_of(kt);
    let slices = fan.div_ceil(slice);
    let out = if slices > 1 {
        ctx.scratch_shared(
            op,
            "elementwise.hc_project_partial",
            (slices * rows * (mix_hc + 1)) as usize * core::mem::size_of::<f32>(),
        )? as usize as u64
    } else {
        mixes.ptr
    };
    // A prefill's rows 64 a block over four thread groups, a decode's 32 over
    // eight.
    let (kg, stages) = (if wide { TILE_GROUPS } else { 8 }, TILE_STAGES);
    let tiles = stages * (br * (kt + 8) * 2 + TILE_COLS * (kt + 4) * 4);
    let fold = (kg - 1) * br * (TILE_COLS + 4) * 4;
    let (entry, threads, smem) = (
        format!("::pie::elemwise::hc_prenorm_project_tile<{br}, {kg}, {kt}, {stages}>"),
        br * kg,
        tiles.max(fold),
    );
    ctx.fire(
        op,
        Fire::at(FILE, symbol(&entry)).apply(Launch {
            smem,
            ..Launch::grid([row_blocks, slices, 1], [threads, 1, 1])
        }),
        &[
            streams.arg(),
            hc_fn.arg(),
            crate::jit::ArgValue::Ptr(out),
            stated(op, fan)?.arg(),
            stated(op, mix_hc)?.arg(),
            stated(op, rows)?.arg(),
            stated(op, slice)?.arg(),
            eps.arg(),
            ctx.stage(),
        ],
    )?;
    Ok(Landed { out, slices })
}

/// [`prenorm_launch`] on the tensor cores (`hc_prenorm_project_mma_t`): the
/// weights split into two bf16 halves first (`hc_fn_split`), then 256 rows
/// a block, the row cut into as many slices as fill the multiprocessors
/// four times over (DeepSeek-V4-Flash, 16384 rows: 64 rows a warp and two
/// stages, 134 us where 16 rows a warp and three stages took 152).
fn prenorm_launch_mma(
    ctx: &Ctx,
    op: &'static str,
    streams: &Tensor,
    eps: f32,
    hc_fn: &Tensor,
    mixes: &Tensor,
) -> Result<Landed, Error> {
    const STAGES: u32 = 2;
    // A warp's row tiles of 8: 64 rows a warp, four warps a block.
    const TILES: u32 = 8;
    const ROWS: u32 = 32 * TILES;
    const PARTIAL_ROWS: u32 = 65536;
    let fan = streams.width;
    let mix_hc = hc_fn.rows;
    let rows = nonzero(op, "rows", mixes.rows)?;
    let split =
        ctx.scratch_shared(op, "elementwise.hc_fn_split", (48 * fan) as usize * 2)? as usize as u64;
    ctx.fire(
        op,
        Fire::at(FILE, "::pie::elemwise::hc_fn_split").apply(Launch::grid(
            [(24 * fan).div_ceil(4 * 256), 1, 1],
            [256, 1, 1],
        )),
        &[
            hc_fn.arg(),
            crate::jit::ArgValue::Ptr(split),
            stated(op, fan)?.arg(),
            stated(op, mix_hc)?.arg(),
        ],
    )?;
    let row_blocks = rows.div_ceil(ROWS);
    let want = 4 * ctx.multiprocessors().unwrap_or(132);
    let slices = want
        .div_ceil(row_blocks)
        .clamp(1, (fan / 1024).max(1))
        .min((PARTIAL_ROWS / rows).max(1));
    let slice = fan.div_ceil(slices).next_multiple_of(TILE_K);
    let slices = fan.div_ceil(slice);
    let out = if slices > 1 {
        ctx.scratch_shared(
            op,
            "elementwise.hc_project_partial",
            (slices * rows * (mix_hc + 1)) as usize * core::mem::size_of::<f32>(),
        )? as usize as u64
    } else {
        mixes.ptr
    };
    let smem = STAGES * (ROWS + 48) * (TILE_K + 8) * 2;
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::elemwise::hc_prenorm_project_mma_t<{STAGES}, {TILES}>"
            )),
        )
        .apply(Launch {
            smem,
            ..Launch::grid([row_blocks, slices, 1], [128, 1, 1])
        }),
        &[
            streams.arg(),
            crate::jit::ArgValue::Ptr(split),
            crate::jit::ArgValue::Ptr(out),
            stated(op, fan)?.arg(),
            stated(op, mix_hc)?.arg(),
            stated(op, rows)?.arg(),
            stated(op, slice)?.arg(),
            eps.arg(),
            ctx.stage(),
        ],
    )?;
    Ok(Landed { out, slices })
}

/// The stream norm and the mix projection over it in one pass over the bf16
/// `streams` (`hc_prenorm_project_tile`, and
/// `hc_prenorm_project_fold` past one slice): `mixes = (x . w) *
/// rsqrt(mean(x^2) + eps)`. `Ok(false)` where the shapes are not the
/// tile's: the caller runs the two.
pub fn prenorm_project(
    ctx: &Ctx,
    streams: Tensor,
    eps: f32,
    hc_fn: Tensor,
    stream_count: u32,
    mixes: &mut Tensor,
) -> Result<bool, Error> {
    const OP: &str = "elementwise.hc_prenorm_project";
    if !prenorm_fits(&streams, &hc_fn, mixes, stream_count) {
        return Ok(false);
    }
    let landed = prenorm_launch(ctx, OP, &streams, eps, &hc_fn, mixes)?;
    if landed.slices > 1 {
        let (rows, mix_hc) = (mixes.rows, hc_fn.rows);
        // A few slices over many rows (the tensor-core form's) fold a
        // thread a row; a decode's many slices a warp a mix.
        let fold = if ctx.tuning().hc_mma && landed.slices <= 8 && rows >= 1024 {
            Fire::at(FILE, "::pie::elemwise::hc_prenorm_project_fold_rows")
                .apply(Launch::grid([rows.div_ceil(256), 1, 1], [256, 1, 1]))
        } else {
            Fire::at(FILE, "::pie::elemwise::hc_prenorm_project_fold").apply(Launch::grid(
                [(rows * mix_hc).div_ceil(8), 1, 1],
                [BLOCK, 1, 1],
            ))
        };
        ctx.fire(
            OP,
            fold,
            &[
                crate::jit::ArgValue::Ptr(landed.out),
                mixes.arg(),
                stated(OP, mix_hc)?.arg(),
                stated(OP, rows)?.arg(),
                stated(OP, landed.slices)?.arg(),
                stated(OP, streams.width)?.arg(),
                eps.arg(),
                ctx.stage(),
            ],
        )?;
    }
    Ok(true)
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
    const OP: &str = "elementwise.hc_pre";
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

/// `hc_project` on `hc_project_tile` (and `hc_project_fold` past one slice):
/// a decode fire's few rows cut the row into slices across blocks.
#[allow(clippy::too_many_arguments)]
fn project_tile(
    ctx: &Ctx,
    op: &'static str,
    normed: Tensor,
    hc_fn: Tensor,
    mixes: &mut Tensor,
    rows: u32,
    fan: u32,
    mix_hc: u32,
) -> Result<(), Error> {
    const SLICE_FLOOR: u32 = 512;
    const PARTIAL_ROWS: u32 = 2048;
    let want = 2 * ctx.multiprocessors().unwrap_or(132);
    let (br, kg, kt, stages) = (TILE_ROWS, TILE_GROUPS, TILE_K, TILE_STAGES);
    let smem = stages * (br + TILE_COLS) * (kt + 4) * 4;
    let row_blocks = rows.div_ceil(br);
    let slices = want
        .div_ceil(row_blocks)
        .clamp(1, (fan / SLICE_FLOOR).max(1))
        .min((PARTIAL_ROWS / rows).max(1));
    let slice = fan.div_ceil(slices).next_multiple_of(kt);
    let slices = fan.div_ceil(slice);
    let out = if slices > 1 {
        ctx.scratch_shared(
            op,
            "elementwise.hc_project_partial",
            (slices * rows * mix_hc) as usize * core::mem::size_of::<f32>(),
        )? as usize as u64
    } else {
        mixes.ptr
    };
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::elemwise::hc_project_tile<{br}, {kg}, {kt}, {stages}>"
            )),
        )
        .apply(Launch {
            smem,
            ..Launch::grid([row_blocks, slices, 1], [br * kg, 1, 1])
        }),
        &[
            normed.arg(),
            hc_fn.arg(),
            crate::jit::ArgValue::Ptr(out),
            stated(op, fan)?.arg(),
            stated(op, mix_hc)?.arg(),
            stated(op, rows)?.arg(),
            stated(op, slice)?.arg(),
            ctx.stage(),
        ],
    )?;
    if slices == 1 {
        return Ok(());
    }
    ctx.fire(
        op,
        Fire::at(FILE, "::pie::elemwise::hc_project_fold")
            .apply(Launch::flat(rows * mix_hc, BLOCK)),
        &[
            crate::jit::ArgValue::Ptr(out),
            mixes.arg(),
            stated(op, mix_hc)?.arg(),
            stated(op, rows)?.arg(),
            stated(op, slices)?.arg(),
            ctx.stage(),
        ],
    )
}

/// `hc_project` on `hc_project_split` (and `hc_project_fold` past one slice).
#[allow(clippy::too_many_arguments)]
fn project_split(
    ctx: &Ctx,
    op: &'static str,
    normed: Tensor,
    hc_fn: Tensor,
    mixes: &mut Tensor,
    rows: u32,
    fan: u32,
    mix_hc: u32,
) -> Result<(), Error> {
    const SLICE_FLOOR: u32 = 256;
    let row_blocks = rows.div_ceil(PROJECT_ROWS);
    let want = 2 * ctx.multiprocessors().unwrap_or(132);
    // At most 2048 partial rows: a region captured on its own holds its own
    // partial slab.
    const PARTIAL_ROWS: u32 = 2048;
    let slices = want
        .div_ceil(row_blocks)
        .clamp(1, (fan / SLICE_FLOOR).max(1))
        .min(64)
        .min((PARTIAL_ROWS / rows).max(1));
    let slice = fan.div_ceil(slices).next_multiple_of(4);
    let slices = fan.div_ceil(slice);
    let out = if slices > 1 {
        ctx.scratch_shared(
            op,
            "elementwise.hc_project_partial",
            (slices * rows * mix_hc) as usize * core::mem::size_of::<f32>(),
        )? as usize as u64
    } else {
        mixes.ptr
    };
    ctx.fire(
        op,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::elemwise::hc_project_split<{PROJECT_ROWS}, {PROJECT_COLS}>"
            )),
        )
        .apply(Launch::grid([row_blocks, slices, 1], [BLOCK, 1, 1])),
        &[
            normed.arg(),
            hc_fn.arg(),
            crate::jit::ArgValue::Ptr(out),
            stated(op, fan)?.arg(),
            stated(op, mix_hc)?.arg(),
            stated(op, rows)?.arg(),
            stated(op, slice)?.arg(),
            ctx.stage(),
        ],
    )?;
    if slices == 1 {
        return Ok(());
    }
    ctx.fire(
        op,
        Fire::at(FILE, "::pie::elemwise::hc_project_fold")
            .apply(Launch::flat(rows * mix_hc, BLOCK)),
        &[
            crate::jit::ArgValue::Ptr(out),
            mixes.arg(),
            stated(op, mix_hc)?.arg(),
            stated(op, rows)?.arg(),
            stated(op, slices)?.arg(),
            ctx.stage(),
        ],
    )
}

/// A prefill's RMS norm of a 4096-wide row landing, beside the bf16 row, its
/// e4m3 1x128 operand in `planes` (`[codes, atoms]`, act_quant's layout) for
/// the GEMM that reads it next (`rmsnorm_emit_e4m3`). `Ok(false)` where the
/// row is not that shape: the caller norms as before.
pub fn rmsnorm_emit(
    ctx: &Ctx,
    x: Tensor,
    weight: Tensor,
    eps: f32,
    y: &mut Tensor,
    planes: [u64; 2],
) -> Result<bool, Error> {
    const OP: &str = "elementwise.rmsnorm_emit";
    if x.dtype != Dtype::Bf16
        || y.dtype != Dtype::Bf16
        || weight.dtype != Dtype::Bf16
        || x.width != 4096
        || y.width != 4096
        || weight.rows * weight.width != 4096
        || ![x.ptr, y.ptr, weight.ptr].iter().all(|at| at % 16 == 0)
    {
        return Ok(false);
    }
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::elemwise::rmsnorm_emit_e4m3<::pie::bf16>")
            .apply(Launch::per_row(y.rows, 256)),
        &[
            x.arg(),
            weight.arg(),
            y.arg(),
            stated(OP, y.width)?.arg(),
            eps.arg(),
            crate::jit::ArgValue::Ptr(planes[0]),
            crate::jit::ArgValue::Ptr(planes[1]),
            ctx.stage(),
        ],
    )?;
    Ok(true)
}
