//! Hyper-connections (`elemwise.hc_expand`, `hc_rmsnorm_f32`, `hc_gates`,
//! `hc_fold`, `hc_collapse`, `hc_mix`, `hc_inject`, `ple_gate`) from
//! `kernels/elemwise/hc.slang`: a row carried as bf16 streams side by side.
#![allow(clippy::too_many_arguments)]

use crate::encode::{
    Arg, Ctx, Fire, Grid, dtype_dispatch, elementwise_rows, nonzero, refuse, stated,
};
use crate::error::Error;
use crate::tensor::Tensor;

const HC: &str = "elemwise/hc.slang";

/// The workgroup width every instantiation is built at.
const BLOCK: u32 = 256;

fn even_hidden(op: &'static str, hidden: u32) -> Result<u32, Error> {
    if !hidden.is_multiple_of(2) {
        return Err(refuse(
            op,
            format!("the hidden width {hidden} is odd: a bf16 word holds a pair"),
        ));
    }
    Ok(hidden / 2)
}

/// The most streams the mixers unroll into register and workgroup arrays.
pub const MAX_HC_MULT: u32 = 8;

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

fn per_row(op: &'static str, rows: u32) -> Result<Grid, Error> {
    let rows = nonzero(op, "rows", rows)?;
    let lanes = rows.checked_mul(BLOCK).ok_or_else(|| {
        refuse(
            op,
            format!("the grid will not launch: {rows} rows x {BLOCK}"),
        )
    })?;
    Ok(Grid::of([lanes, 1, 1], [BLOCK, 1, 1]))
}

pub fn expand(ctx: &Ctx<'_>, x: Tensor, streams: u32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.hc_expand";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "elemwise_hc_expand_bf16" });
    debug_assert_eq!(y.rows, x.rows, "the expansion lands one wide row per row");
    let fan = stream_fan(OP, y.width, x.width)?;
    debug_assert_eq!(
        fan, streams,
        "the row's stream fan is the count the statement states"
    );
    let words = even_hidden(OP, x.width)?;
    ctx.fire(
        Fire::at(HC, entry).apply(Grid::of(
            elementwise_rows(OP, words, x.rows)?,
            [BLOCK, 1, 1],
        )),
        &[
            x.arg(),
            y.arg_mut(),
            stated(OP, x.rows)?.arg(),
            stated(OP, fan)?.arg(),
            stated(OP, x.width)?.arg(),
        ],
    )
}

pub fn rmsnorm_f32(ctx: &Ctx<'_>, streams: Tensor, eps: f32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.hc_rmsnorm_f32";
    let entry = dtype_dispatch!(OP, streams.dtype, { Bf16 => "elemwise_hc_rmsnorm_f32_bf16" });
    debug_assert_eq!(y.dtype, dtype::Dtype::F32, "`{OP}` widens to f32");
    debug_assert!(
        y.rows == streams.rows && y.width == streams.width,
        "the normed rectangle is the stream rectangle"
    );
    ctx.fire(
        Fire::at(HC, entry).apply(per_row(OP, y.rows)?),
        &[
            streams.arg(),
            y.arg_mut(),
            stated(OP, nonzero(OP, "the normed row's width", y.width)?)?.arg(),
            eps.arg(),
        ],
    )
}

pub fn gates(
    ctx: &Ctx<'_>,
    normed: Tensor,
    streams: Tensor,
    scale: Tensor,
    base: Tensor,
    stream_count: u32,
    gate_eps: f32,
    alpha: f32,
    sinkhorn: u32,
    x: Tensor,
    post_mix: Tensor,
    comb_mix: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.hc_gates";
    let entry = dtype_dispatch!(OP, streams.dtype, { Bf16 => "elemwise_hc_gates_bf16" });
    debug_assert_eq!(
        normed.dtype,
        dtype::Dtype::F32,
        "`{OP}` reads an f32 mix row"
    );
    debug_assert_eq!(
        scale.dtype,
        dtype::Dtype::F32,
        "`{OP}` reads f32 mix scales"
    );
    debug_assert_eq!(base.dtype, dtype::Dtype::F32, "`{OP}` reads f32 mix bases");
    debug_assert!(
        post_mix.dtype == dtype::Dtype::F32 && comb_mix.dtype == dtype::Dtype::F32,
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
    even_hidden(OP, x.width)?;
    ctx.fire(
        Fire::at(HC, entry).apply(per_row(OP, x.rows)?),
        &[
            normed.arg(),
            scale.arg(),
            base.arg(),
            streams.arg(),
            post_mix.arg_mut(),
            comb_mix.arg_mut(),
            x.arg_mut(),
            stated(OP, fan)?.arg(),
            stated(OP, nonzero(OP, "the hidden width", x.width)?)?.arg(),
            gate_eps.arg(),
            alpha.arg(),
            stated(OP, sinkhorn)?.arg(),
        ],
    )
}

pub fn fold(
    ctx: &Ctx<'_>,
    x: Tensor,
    streams: Tensor,
    post_mix: Tensor,
    comb_mix: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.hc_fold";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "elemwise_hc_fold_bf16" });
    debug_assert!(
        y.rows == streams.rows && y.width == streams.width,
        "the fold lands the stream rectangle it mixes"
    );
    let fan = stream_fan(OP, y.width, x.width)?;
    debug_assert!(
        post_mix.dtype == dtype::Dtype::F32 && comb_mix.dtype == dtype::Dtype::F32,
        "`{OP}` reads f32 gate matrices"
    );
    let words = even_hidden(OP, x.width)?;
    ctx.fire(
        Fire::at(HC, entry).apply(Grid::of(
            elementwise_rows(OP, words, y.rows)?,
            [BLOCK, 1, 1],
        )),
        &[
            x.arg(),
            streams.arg(),
            post_mix.arg(),
            comb_mix.arg(),
            y.arg_mut(),
            stated(OP, y.rows)?.arg(),
            stated(OP, fan)?.arg(),
            stated(OP, x.width)?.arg(),
        ],
    )
}

pub fn collapse(
    ctx: &Ctx<'_>,
    mixes: Tensor,
    streams: Tensor,
    scale: Tensor,
    base: Tensor,
    stream_count: u32,
    hc_eps: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.hc_collapse";
    let entry = dtype_dispatch!(OP, streams.dtype, { Bf16 => "elemwise_hc_collapse_bf16" });
    debug_assert_eq!(
        mixes.dtype,
        dtype::Dtype::F32,
        "`{OP}` reads an f32 mix row"
    );
    debug_assert_eq!(
        scale.dtype,
        dtype::Dtype::F32,
        "`{OP}` reads an f32 mix scale"
    );
    debug_assert_eq!(base.dtype, dtype::Dtype::F32, "`{OP}` reads f32 mix bases");
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
    even_hidden(OP, y.width)?;
    ctx.fire(
        Fire::at(HC, entry).apply(per_row(OP, y.rows)?),
        &[
            mixes.arg(),
            scale.arg(),
            base.arg(),
            streams.arg(),
            y.arg_mut(),
            stated(OP, fan)?.arg(),
            stated(OP, nonzero(OP, "the hidden width", y.width)?)?.arg(),
            hc_eps.arg(),
        ],
    )
}

pub fn mix(
    ctx: &Ctx<'_>,
    gates: Tensor,
    normed: Tensor,
    streams: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.hc_mix";
    let entry = dtype_dispatch!(OP, y.dtype, { Bf16 => "elemwise_hc_mix_bf16" });
    let fan = stream_fan(OP, normed.width, y.width)?;
    if fan != streams {
        return Err(refuse(
            OP,
            format!("the wide row fans {fan} ways and the statement states {streams}"),
        ));
    }
    debug_assert!(
        gates.width == normed.width && gates.rows == normed.rows,
        "`{OP}` gates the normed rectangle element for element"
    );
    let words = even_hidden(OP, y.width)?;
    ctx.fire(
        Fire::at(HC, entry).apply(Grid::of(
            elementwise_rows(OP, words, y.rows)?,
            [BLOCK, 1, 1],
        )),
        &[
            gates.arg(),
            normed.arg(),
            y.arg_mut(),
            stated(OP, y.rows)?.arg(),
            stated(OP, fan)?.arg(),
            stated(OP, y.width)?.arg(),
        ],
    )
}

pub fn inject(
    ctx: &Ctx<'_>,
    o: Tensor,
    gates: Tensor,
    streams: u32,
    hyper: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.hc_inject";
    let entry = dtype_dispatch!(OP, hyper.dtype, { Bf16 => "elemwise_hc_inject_bf16" });
    let fan = stream_fan(OP, hyper.width, o.width)?;
    if fan != streams {
        return Err(refuse(
            OP,
            format!("the wide row fans {fan} ways and the statement states {streams}"),
        ));
    }
    debug_assert_eq!(
        gates.width, fan,
        "`{OP}` reads one gate logit per stream, not one per element"
    );
    let words = even_hidden(OP, o.width)?;
    ctx.fire(
        Fire::at(HC, entry).apply(Grid::of(
            elementwise_rows(OP, words, hyper.rows)?,
            [BLOCK, 1, 1],
        )),
        &[
            o.arg(),
            gates.arg(),
            hyper.arg_mut(),
            stated(OP, hyper.rows)?.arg(),
            stated(OP, fan)?.arg(),
            stated(OP, o.width)?.arg(),
        ],
    )
}

pub fn ple_gate(
    ctx: &Ctx<'_>,
    key: Tensor,
    query: Tensor,
    value: Tensor,
    streams: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.ple_gate";
    let entry = dtype_dispatch!(OP, y.dtype, { Bf16 => "elemwise_ple_gate_bf16" });
    let fan = stream_fan(OP, y.width, value.width)?;
    if fan != streams {
        return Err(refuse(
            OP,
            format!("the wide row fans {fan} ways and the statement states {streams}"),
        ));
    }
    debug_assert!(
        key.width == y.width && query.width == y.width,
        "`{OP}` dots the key and query over the stream row it gates"
    );
    even_hidden(OP, value.width)?;
    let groups = y.rows.checked_mul(fan).ok_or_else(|| {
        refuse(
            OP,
            format!("the grid will not launch: {} rows x {fan} streams", y.rows),
        )
    })?;
    ctx.fire(
        Fire::at(HC, entry).apply(per_row(OP, groups)?),
        &[
            key.arg(),
            query.arg(),
            value.arg(),
            y.arg_mut(),
            stated(OP, fan)?.arg(),
            stated(OP, value.width)?.arg(),
        ],
    )
}
