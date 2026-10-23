use crate::error::Error;
use dtype::Dtype;

use crate::jit::{Arg, Ctx, Fire, Launch, dtype_dispatch, nonzero, refuse, stated, symbol};
use crate::tensor::Tensor;

const FILE: &str = "elemwise/hc.hip";

const BLOCK: u32 = 256;

pub(crate) const MAX_HC_MULT: u32 = 8;

/// The stream kernels' threads a block. A thread takes 8 columns, so a
/// 4096-wide stream row is four blocks.
const CHUNK_BLOCK: u32 = 128;

/// The row norm's threads a block, and the 8-column packs a thread holds in
/// registers between the norm's sum and its store.
const NORM_BLOCK: u32 = 512;
const NORM_CAP: u32 = 8;

/// Whether a kernel over `hidden`-wide 2-byte rows moves 16-byte packs:
/// every row starts 16-byte aligned when the width is a multiple of 8 and
/// every plane does.
fn packs(hidden: u32, planes: &[&Tensor]) -> &'static str {
    if hidden.is_multiple_of(8) && planes.iter().all(|t| t.ptr.is_multiple_of(16)) {
        "::pie::true_type::value"
    } else {
        "::pie::false_type::value"
    }
}

/// A stream kernel's launch: block (n, lead + c) over chunk c of row n,
/// after `lead` blocks a row that do something else.
fn per_chunk(op: &'static str, rows: u32, hidden: u32, lead: u32) -> Result<Launch, Error> {
    nonzero(op, "rows", rows)?;
    nonzero(op, "the hidden width", hidden)?;
    Ok(Launch::grid(
        [rows, lead + hidden.div_ceil(8 * CHUNK_BLOCK), 1],
        [CHUNK_BLOCK, 1, 1],
    ))
}

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

pub fn expand(ctx: &Ctx, x: Tensor, streams: u32, y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.hc_expand";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(y.rows, x.rows, "the expansion lands one wide row per row");
    let fan = stream_fan(OP, y.width, x.width)?;
    debug_assert_eq!(
        fan, streams,
        "the row's stream fan is the count the statement states"
    );
    let vec = packs(x.width, &[&x, y]);
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::elemwise::hc_expand<{t}, {fan}, {CHUNK_BLOCK}, {vec}>"
            )),
        )
        .apply(per_chunk(OP, x.rows, x.width, 0)?),
        &[x.arg(), y.arg(), stated(OP, x.width)?.arg(), ctx.stage()],
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
    let dim = nonzero(OP, "the normed row's width", y.width)?;
    // The row held in registers needs whole, aligned packs that fit.
    let vec = dim.is_multiple_of(8)
        && dim / 8 <= NORM_BLOCK * NORM_CAP
        && streams.ptr.is_multiple_of(16)
        && y.ptr.is_multiple_of(16);
    let vec = if vec {
        "::pie::true_type::value"
    } else {
        "::pie::false_type::value"
    };
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::elemwise::hc_rmsnorm_f32<{NORM_BLOCK}, {NORM_CAP}, {vec}>"
            )),
        )
        .apply(Launch::per_row(y.rows, NORM_BLOCK)),
        &[
            streams.arg(),
            y.arg(),
            stated(OP, dim)?.arg(),
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
    // Without a combine the mix row is the pre and post logits alone.
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
    let vec = packs(x.width, &[&streams, x]);
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::elemwise::hc_gates<{t}, {fan}, {CHUNK_BLOCK}, {vec}, {comb}>"
            )),
        )
        .apply(per_chunk(OP, x.rows, x.width, 1)?),
        &[
            normed.arg(),
            scale.arg(),
            base.arg(),
            streams.arg(),
            post_mix.arg(),
            comb_mix.arg(),
            x.arg(),
            stated(OP, x.width)?.arg(),
            gate_eps.arg(),
            alpha.arg(),
            stated(OP, sinkhorn)?.arg(),
            ctx.stage(),
        ],
    )
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
    let vec = packs(x.width, &[&x, &streams, y]);
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::elemwise::hc_fold<{t}, {fan}, {CHUNK_BLOCK}, {vec}>"
            )),
        )
        .apply(per_chunk(OP, y.rows, x.width, 0)?),
        &[
            x.arg(),
            streams.arg(),
            post_mix.arg(),
            comb_mix.arg(),
            y.arg(),
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
    let vec = packs(y.width, &[&streams, y]);
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::elemwise::hc_head_postprocess<{t}, {fan}, {CHUNK_BLOCK}, {vec}>"
            )),
        )
        .apply(per_chunk(OP, y.rows, y.width, 0)?),
        &[
            mixes.arg(),
            scale.arg(),
            base.arg(),
            streams.arg(),
            y.arg(),
            stated(OP, y.width)?.arg(),
            hc_eps.arg(),
            ctx.stage(),
        ],
    )
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
    let vec = packs(y.width, &[&gates, &normed, y]);
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::elemwise::hc_mix<{t}, {fan}, {CHUNK_BLOCK}, {vec}>"
            )),
        )
        .apply(per_chunk(OP, y.rows, y.width, 0)?),
        &[
            gates.arg(),
            normed.arg(),
            y.arg(),
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
    let vec = packs(o.width, &[&o, hyper]);
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!(
                "::pie::elemwise::hc_inject<{t}, {fan}, {CHUNK_BLOCK}, {vec}>"
            )),
        )
        .apply(per_chunk(OP, o.rows, o.width, 0)?),
        &[
            o.arg(),
            gates.arg(),
            hyper.arg(),
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
