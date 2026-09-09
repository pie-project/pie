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
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::elemwise::hc_rmsnorm_f32<::pie::bf16, 256>")
            .apply(Launch::per_row(y.rows, BLOCK)),
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
    if mixes.width != mix_hc || (mix_hc != layer_row && mix_hc != stream_count) {
        return Err(refuse(
            OP,
            format!(
                "a {stream_count}-stream mix row is {layer_row} wide and a trunk collapse row \
                 is {stream_count}; the plane lands {} rows into a {}-wide row",
                hc_fn.rows, mixes.width
            ),
        ));
    }
    let rows = nonzero(OP, "rows", mixes.rows)?;
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
            ctx.stage(),
        ],
    )
}

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
            ctx.stage(),
        ],
    )
}
