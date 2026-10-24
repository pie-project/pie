#![allow(unsafe_code)]
use crate::error::Error;
use crate::jit::{
    Arg, Ctx, Fire, Launch, aligned16, dtype_dispatch, kernel, nonzero, refuse, stated,
};
use crate::tensor::Tensor;
use dtype::Dtype;

const FILE: &str = "gather/embed.cu";

const VEC_WIDTH: u32 = 8;

fn vectorisable(hidden: u32, table: u64, y: u64) -> bool {
    hidden.is_multiple_of(VEC_WIDTH) && aligned16(table) && aligned16(y)
}

pub fn embed(
    ctx: &Ctx,
    ids: Tensor,
    table: Tensor,
    vocab: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.embed";
    dtype_dispatch!(OP, table.dtype, { Bf16 => () });
    debug_assert_eq!(ids.dtype, Dtype::I32, "`{OP}` gathers by i32 token ids");
    debug_assert_eq!(
        ids.rows, y.rows,
        "the token ids handed over are the rows this gather lands"
    );
    let vocab = stated(OP, nonzero(OP, "the embedding table's row count", vocab)?)?;
    let hidden = stated(OP, nonzero(OP, "the embedded row's width", y.width)?)?;
    let rows = stated(OP, nonzero(OP, "rows", y.rows)?)?;

    let vec = vectorisable(y.width, table.ptr, y.ptr);
    let per_row = if vec { y.width / VEC_WIDTH } else { y.width };
    let total = u64::from(y.rows) * u64::from(per_row);
    let blocks = u32::try_from(total.div_ceil(u64::from(BLOCK)))
        .map_err(|_| refuse(OP, format!("{total} gather lanes do not fit a 32-bit grid")))?;
    let instantiation = if vec {
        "::pie::gather::embed<::pie::true_type::value>"
    } else {
        "::pie::gather::embed<::pie::false_type::value>"
    };
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, instantiation)).apply(Launch::grid([blocks, 1, 1], [BLOCK, 1, 1])),
        &[
            ids.arg(),
            table.arg(),
            y.arg(),
            hidden.arg(),
            vocab.arg(),
            rows.arg(),
            stated(OP, per_row)?.arg(),
            ctx.stage(),
        ],
    )
}

pub fn embed_vocab_shard(
    ctx: &Ctx,
    ids: Tensor,
    table: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.embed_vocab_shard";
    dtype_dispatch!(OP, table.dtype, { Bf16 => () });
    debug_assert_eq!(ids.dtype, Dtype::I32, "`{OP}` gathers by i32 token ids");
    debug_assert_eq!(ids.rows, y.rows, "the ids handed over are the rows landed");
    let local = nonzero(OP, "this rank's band of the embedding table", table.rows)?;
    let hidden = stated(OP, nonzero(OP, "the embedded row's width", y.width)?)?;
    let rows = nonzero(OP, "rows", y.rows)?;
    let rank = comm_rank(ctx, OP)?;
    ctx.fire(
        OP,
        Fire::at(kernel(
            FILE,
            "::pie::gather::embed_vocab_shard<::pie::bf16>",
        ))
        .apply(Launch::grid([rows, 1, 1], [BLOCK, 1, 1])),
        &[
            ids.arg(),
            table.arg(),
            y.arg(),
            hidden.arg(),
            stated(OP, local)?.arg(),
            stated(OP, rank.saturating_mul(local))?.arg(),
            ctx.stage(),
        ],
    )
}

pub(crate) fn comm_rank(ctx: &Ctx, op: &'static str) -> Result<u32, Error> {
    if let Some(rank) = ctx.rank() {
        return Ok(rank);
    }
    let comm = ctx.comm(op)?;
    #[cfg(feature = "cuda")]
    {
        use cudarc::nccl::sys as nccl;
        let mut rank: i32 = 0;
        // SAFETY: `comm` is the live communicator this context fires its
        // collectives on; the out-parameter is a stack i32.
        let code = unsafe { nccl::ncclCommUserRank(comm.cast(), &mut rank) };
        crate::collective::answered(op, "ncclCommUserRank", code)?;
        Ok(u32::try_from(rank).unwrap_or(0))
    }
    #[cfg(not(feature = "cuda"))]
    {
        let _ = comm;
        Err(crate::jit::runtimeless(op))
    }
}

#[allow(clippy::too_many_arguments)]
pub fn embed_scale_add(
    ctx: &Ctx,
    ids: Tensor,
    table: Tensor,
    vocab: u32,
    e: &mut Tensor,
    embed_scale: f32,
    e_scaled: &mut Tensor,
    y: &mut Tensor,
    out_scale: f32,
    y_scaled: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.embed_scale_add";
    dtype_dispatch!(OP, table.dtype, { Bf16 => () });
    debug_assert_eq!(ids.dtype, Dtype::I32, "`{OP}` gathers by i32 token ids");
    debug_assert!(
        ids.rows == y.rows && e.rows == y.rows && e.width == y.width,
        "the token ids and every row plane share the fire's rows"
    );
    let vocab = stated(OP, nonzero(OP, "the embedding table's row count", vocab)?)?;
    let hidden = stated(OP, nonzero(OP, "the embedded row's width", y.width)?)?;
    let rows = stated(OP, nonzero(OP, "rows", y.rows)?)?;
    let total = u64::from(y.rows) * u64::from(y.width);
    let blocks = u32::try_from(total.div_ceil(u64::from(BLOCK)))
        .map_err(|_| refuse(OP, format!("{total} gather lanes do not fit a 32-bit grid")))?;
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::gather::embed_scale_add"))
            .apply(Launch::grid([blocks, 1, 1], [BLOCK, 1, 1])),
        &[
            ids.arg(),
            table.arg(),
            e.arg(),
            embed_scale.arg(),
            e_scaled.arg(),
            y.arg(),
            out_scale.arg(),
            y_scaled.arg(),
            hidden.arg(),
            vocab.arg(),
            rows.arg(),
            ctx.stage(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn embed_scale_add_select(
    ctx: &Ctx,
    ids: Tensor,
    table: Tensor,
    vocab: u32,
    e: &mut Tensor,
    embed_scale: f32,
    e_scaled: &mut Tensor,
    stacked: Tensor,
    layer: u32,
    width: u32,
    y_out: &mut Tensor,
    out_scale: f32,
    y_scaled: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.embed_scale_add_select";
    dtype_dispatch!(OP, table.dtype, { Bf16 => () });
    debug_assert_eq!(ids.dtype, Dtype::I32, "`{OP}` gathers by i32 token ids");
    debug_assert!(
        ids.rows == y_out.rows && e.rows == y_out.rows && e.width == y_out.width,
        "the token ids and every row plane share the fire's rows"
    );
    let col = layer.checked_mul(width).ok_or_else(|| {
        refuse(
            OP,
            format!("layer {layer}'s slice starts beyond any column: {layer} x {width}"),
        )
    })?;
    if width != y_out.width || col.checked_add(width).is_none_or(|end| end > stacked.width) {
        return Err(refuse(
            OP,
            format!(
                "layer {layer}'s {width}-wide slice does not sit in a {}-wide stacked row landing a \
                 {}-wide row",
                stacked.width, y_out.width
            ),
        ));
    }
    let vocab = stated(OP, nonzero(OP, "the embedding table's row count", vocab)?)?;
    let hidden = stated(OP, nonzero(OP, "the embedded row's width", y_out.width)?)?;
    let rows = stated(OP, nonzero(OP, "rows", y_out.rows)?)?;
    let stacked_width = stated(OP, stacked.width)?;
    let col = stated(OP, col)?;
    let total = u64::from(y_out.rows) * u64::from(y_out.width);
    let blocks = u32::try_from(total.div_ceil(u64::from(BLOCK)))
        .map_err(|_| refuse(OP, format!("{total} gather lanes do not fit a 32-bit grid")))?;
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::gather::embed_scale_add_select"))
            .apply(Launch::grid([blocks, 1, 1], [BLOCK, 1, 1])),
        &[
            ids.arg(),
            table.arg(),
            e.arg(),
            embed_scale.arg(),
            e_scaled.arg(),
            stacked.arg(),
            stacked_width.arg(),
            col.arg(),
            y_out.arg(),
            out_scale.arg(),
            y_scaled.arg(),
            hidden.arg(),
            vocab.arg(),
            rows.arg(),
            ctx.stage(),
        ],
    )
}

const BLOCK: u32 = 256;

pub fn embed_concat(
    ctx: &Ctx,
    ids: Tensor,
    table: Tensor,
    vocab: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.embed";
    debug_assert_eq!(ids.dtype, Dtype::I32, "`{OP}` gathers by i32 rows");
    let t = dtype_dispatch!(OP, table.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let heads = nonzero(OP, "the hashed head count", ids.width)?;
    let width = nonzero(OP, "the table row width", table.width)?;
    debug_assert_eq!(
        y.width,
        heads * width,
        "the output row is every head's table row, side by side"
    );
    debug_assert_eq!(y.rows, ids.rows, "one output row per id row");
    let rows = nonzero(OP, "rows", ids.rows)?;
    let total = u64::from(rows) * u64::from(heads) * u64::from(width);
    let lanes = u32::try_from(total).map_err(|_| {
        refuse(
            OP,
            format!("{total} elements do not fit a 32-bit launch extent"),
        )
    })?;
    // a table the ranks split by row: this rank holds its band, the others' rows land zero
    let held = table.rows.min(vocab);
    let base = if held < vocab {
        comm_rank(ctx, OP)?.saturating_mul(held)
    } else {
        0
    };
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, &format!("::pie::gather::embed_concat<{t}>")))
            .apply(Launch::flat(lanes, BLOCK)),
        &[
            ids.arg(),
            table.arg(),
            y.arg(),
            stated(OP, rows)?.arg(),
            stated(OP, heads)?.arg(),
            stated(OP, width)?.arg(),
            stated(OP, vocab)?.arg(),
            stated(OP, held)?.arg(),
            stated(OP, base)?.arg(),
            ctx.stage(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn embed_concat_mlxu4(
    ctx: &Ctx,
    ids: Tensor,
    codes: Tensor,
    scales: Tensor,
    biases: Option<Tensor>,
    vocab: u32,
    seat: crate::linear::moe::GroupSeat,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.embed";
    debug_assert_eq!(ids.dtype, Dtype::I32, "`{OP}` gathers by i32 rows");
    let t = dtype_dispatch!(OP, y.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let Some(biases) = biases else {
        return Err(refuse(
            OP,
            "an mxfp4 table has no gather point on this plane; the affine \
             triplet is the packed landing this gather reads",
        ));
    };
    let heads = nonzero(OP, "the hashed head count", ids.width)?;
    if y.width == 0 || !y.width.is_multiple_of(heads) {
        return Err(refuse(
            OP,
            format!(
                "the {}-wide output is not a whole number of {heads} head slices",
                y.width
            ),
        ));
    }
    let width = y.width / heads;
    let factor_rows = nonzero(OP, "the factor plane's rows", scales.rows)?;
    let per_row = scales.width / 2;
    if factor_rows != vocab || per_row == 0 || !width.is_multiple_of(per_row) {
        return Err(refuse(
            OP,
            format!(
                "a [{factor_rows} x {per_row}] factor plane does not group a \
                 [{vocab} x {width}] table"
            ),
        ));
    }
    let group = width / per_row;
    debug_assert_eq!(y.rows, ids.rows, "one output row per id row");
    let rows = nonzero(OP, "rows", ids.rows)?;
    let total = u64::from(rows) * u64::from(heads) * u64::from(width);
    let lanes = u32::try_from(total).map_err(|_| {
        refuse(
            OP,
            format!("{total} elements do not fit a 32-bit launch extent"),
        )
    })?;
    ctx.fire(
        OP,
        Fire::at(kernel(
            FILE,
            &format!("::pie::gather::embed_concat_mlxu4<{t}>"),
        ))
        .apply(Launch::flat(lanes, BLOCK)),
        &[
            ids.arg(),
            codes.arg(),
            scales.arg(),
            biases.arg(),
            y.arg(),
            stated(OP, rows)?.arg(),
            stated(OP, heads)?.arg(),
            stated(OP, width)?.arg(),
            stated(OP, group)?.arg(),
            stated(OP, vocab)?.arg(),
            stated(OP, 0u32)?.arg(),
            crate::jit::ArgValue::Ptr(seat.cell),
            crate::jit::ArgValue::Ptr(seat.hits),
            ctx.stage(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn embed_mlx_affine(
    ctx: &Ctx,
    ids: Tensor,
    codes: Tensor,
    scales: Tensor,
    biases: Option<Tensor>,
    vocab: u32,
    seat: crate::linear::moe::GroupSeat,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.embed";
    debug_assert_eq!(ids.dtype, Dtype::I32, "`{OP}` gathers by i32 token ids");
    let t = dtype_dispatch!(OP, y.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let Some(biases) = biases else {
        return Err(refuse(
            OP,
            "an mxfp4 table has no gather point on this plane; the affine \
             triplet is the packed landing this gather reads",
        ));
    };
    let width = nonzero(OP, "the embedded row's width", y.width)?;
    let entry = if codes.width == width {
        "embed_concat_mlxu8"
    } else if codes.width * 2 == width {
        "embed_concat_mlxu4"
    } else {
        return Err(refuse(
            OP,
            format!(
                "a {}-byte code row stores a {width}-wide row at neither four \
                 nor eight bits",
                codes.width
            ),
        ));
    };
    let band = nonzero(OP, "the factor plane's rows", scales.rows)?;
    let per_row = scales.width / 2;
    let Some(group) = affine_band_group(vocab, band, width, per_row) else {
        return Err(refuse(
            OP,
            format!(
                "a [{band} x {per_row}] factor plane does not group a \
                 [{vocab} x {width}] table"
            ),
        ));
    };
    // A tensor-parallel vocab band gathers only its own ids and lands zeros for
    // the rest; the model's all-reduce after the embed sums the bands.
    let offset = if band < vocab {
        crate::gather::comm_rank(ctx, OP)?.saturating_mul(band)
    } else {
        0
    };
    debug_assert_eq!(y.rows, ids.rows, "one output row per id row");
    let rows = nonzero(OP, "rows", ids.rows)?;
    let total = u64::from(rows) * u64::from(width);
    let lanes = u32::try_from(total).map_err(|_| {
        refuse(
            OP,
            format!("{total} elements do not fit a 32-bit launch extent"),
        )
    })?;
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, &format!("::pie::gather::{entry}<{t}>")))
            .apply(Launch::flat(lanes, BLOCK)),
        &[
            ids.arg(),
            codes.arg(),
            scales.arg(),
            biases.arg(),
            y.arg(),
            stated(OP, rows)?.arg(),
            stated(OP, 1u32)?.arg(),
            stated(OP, width)?.arg(),
            stated(OP, group)?.arg(),
            stated(OP, band)?.arg(),
            stated(OP, offset)?.arg(),
            crate::jit::ArgValue::Ptr(seat.cell),
            crate::jit::ArgValue::Ptr(seat.hits),
            ctx.stage(),
        ],
    )
}

fn affine_band_group(vocab: u32, band: u32, width: u32, per_row: u32) -> Option<u32> {
    let whole_or_band = band == vocab || (band < vocab && vocab.is_multiple_of(band));
    (whole_or_band && per_row != 0 && width.is_multiple_of(per_row)).then(|| width / per_row)
}

// A hashed table stored e4m3 under one f32 scale (`Dtype::E4m3tensor`) or
// one ue8m0 exponent a 32 of each row (`Dtype::Mxfp8`), gathered and
// widened only at the rows its ids name, so the table stays at a byte a
// value on the device.

/// The heads `ids` names a row and each one's width of `y`.
fn head_width(op: &'static str, ids: Tensor, y: &Tensor) -> Result<(u32, u32), Error> {
    debug_assert_eq!(ids.dtype, Dtype::I32, "`{op}` gathers by i32 rows");
    let heads = nonzero(op, "the hashed head count", ids.width)?;
    if y.width == 0 || !y.width.is_multiple_of(heads) {
        return Err(refuse(
            op,
            format!(
                "the {}-wide output is not a whole number of {heads} head slices",
                y.width
            ),
        ));
    }
    debug_assert_eq!(y.rows, ids.rows, "one output row per id row");
    Ok((heads, y.width / heads))
}

/// The gather a thread `per` of each head's slice of a row.
#[allow(clippy::too_many_arguments)]
fn gather(
    ctx: &Ctx,
    op: &'static str,
    entry: &str,
    ids: Tensor,
    codes: Tensor,
    scale: Tensor,
    vocab: u32,
    (heads, width, per): (u32, u32, u32),
    y: &mut Tensor,
) -> Result<(), Error> {
    let t = dtype_dispatch!(op, y.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let rows = nonzero(op, "rows", ids.rows)?;
    let total = u64::from(rows) * u64::from(heads) * u64::from(per);
    let lanes = u32::try_from(total).map_err(|_| {
        refuse(
            op,
            format!("{total} elements do not fit a 32-bit launch extent"),
        )
    })?;
    let entry = entry.replace('T', t);
    ctx.fire(
        op,
        Fire::at(kernel(FILE, &format!("::pie::gather::{entry}")))
            .apply(Launch::flat(lanes, BLOCK)),
        &[
            ids.arg(),
            codes.arg(),
            scale.arg(),
            y.arg(),
            stated(op, rows)?.arg(),
            stated(op, heads)?.arg(),
            stated(op, width)?.arg(),
            stated(op, vocab)?.arg(),
            ctx.stage(),
        ],
    )
}

/// `y[r, h * width ..]` is `codes[ids[r, h]] · scale`: `codes` the `[vocab,
/// width]` table bound as bytes, `scale` its one f32 bound as four bytes.
pub fn embed_concat_e4m3(
    ctx: &Ctx,
    ids: Tensor,
    codes: Tensor,
    scale: Tensor,
    vocab: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.embed";
    let (heads, width) = head_width(OP, ids, y)?;
    let held = u64::from(codes.rows) * u64::from(codes.width);
    if held != u64::from(vocab) * u64::from(width) || scale.rows * scale.width != 4 {
        return Err(refuse(
            OP,
            format!(
                "a {held}-byte table under a {}-byte scale is not [{vocab} x {width}] e4m3 \
                 under one f32",
                scale.rows * scale.width
            ),
        ));
    }
    // Sixteen codes a thread where every row is whole 16-byte runs of an
    // aligned table: the same widening in a sixteenth of the requests.
    let (entry, per) = if width.is_multiple_of(16) && codes.ptr.is_multiple_of(16) {
        ("embed_concat_e4m3_vec16<T, false>", width / 16)
    } else {
        ("embed_concat_e4m3<T>", width)
    };
    gather(
        ctx,
        OP,
        entry,
        ids,
        codes,
        scale,
        vocab,
        (heads, width, per),
        y,
    )
}

/// `y[r, h * width ..]` is `codes[ids[r, h]]` under its row's exponents:
/// `codes` the `[vocab, width]` MXFP8 table's e4m3 codes, `exps` its
/// `[vocab, width / 32]` ue8m0 exponents.
pub fn embed_concat_mxfp8(
    ctx: &Ctx,
    ids: Tensor,
    codes: Tensor,
    exps: Tensor,
    vocab: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.embed";
    let (heads, width) = head_width(OP, ids, y)?;
    if !width.is_multiple_of(32) || !codes.ptr.is_multiple_of(16) {
        return Err(refuse(
            OP,
            format!("an MXFP8 row is whole 32-wide blocks on 16-byte runs, not {width} wide"),
        ));
    }
    let held = u64::from(codes.rows) * u64::from(codes.width);
    let scales = u64::from(exps.rows) * u64::from(exps.width);
    if held != u64::from(vocab) * u64::from(width) || scales != held / 32 {
        return Err(refuse(
            OP,
            format!(
                "{held} codes under {scales} exponents are not a [{vocab} x {width}] MXFP8 table"
            ),
        ));
    }
    let entry = "embed_concat_e4m3_vec16<T, true>";
    gather(
        ctx,
        OP,
        entry,
        ids,
        codes,
        exps,
        vocab,
        (heads, width, width / 16),
        y,
    )
}

const WARP: u32 = 32;

const MAX_BLOCK: u32 = 1024;

pub fn embed_weighted(
    ctx: &Ctx,
    ids: Tensor,
    weights: Tensor,
    table: Tensor,
    vocab: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.embed";
    let t = dtype_dispatch!(OP, table.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(
        y.dtype, table.dtype,
        "`{OP}` gathers into the table's element"
    );

    if ids.dtype != Dtype::I32 {
        return Err(refuse(
            OP,
            format!(
                "the taps this gather is handed are {:?}, and it reads i32 rows",
                ids.dtype
            ),
        ));
    }
    if weights.dtype != Dtype::F32 {
        return Err(refuse(
            OP,
            format!(
                "the interpolation weights are {:?}, and this gather reads f32 — they are the \
                 preprocessor's arithmetic and not the activation's",
                weights.dtype
            ),
        ));
    }
    if ids.rows != weights.rows || ids.width != weights.width {
        return Err(refuse(
            OP,
            format!(
                "{} x {} taps and {} x {} weights; every tap is weighted and every weight taps",
                ids.rows, ids.width, weights.rows, weights.width
            ),
        ));
    }
    if ids.rows != y.rows {
        return Err(refuse(
            OP,
            format!(
                "{} rows of taps land {} rows; a gather answers one row per index row",
                ids.rows, y.rows
            ),
        ));
    }
    let taps = stated(OP, nonzero(OP, "the taps per row", ids.width)?)?;
    let vocab = stated(OP, nonzero(OP, "the table's row count", vocab)?)?;
    let hidden = stated(OP, nonzero(OP, "the gathered row's width", y.width)?)?;
    let rows = nonzero(OP, "rows", y.rows)?;
    let threads = y
        .width
        .div_ceil(WARP)
        .max(1)
        .saturating_mul(WARP)
        .min(MAX_BLOCK);
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, &format!("::pie::gather::embed_weighted<{t}>")))
            .apply(Launch::per_row(rows, threads)),
        &[
            ids.arg(),
            weights.arg(),
            table.arg(),
            y.arg(),
            hidden.arg(),
            vocab.arg(),
            taps.arg(),
            ctx.stage(),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::affine_band_group;

    #[test]
    fn a_tp2_vocab_band_groups_its_half_of_the_table() {
        assert_eq!(affine_band_group(262_144, 262_144, 5376, 84), Some(64));
        assert_eq!(affine_band_group(262_144, 131_072, 5376, 84), Some(64));
        assert_eq!(affine_band_group(262_144, 100_000, 5376, 84), None);
        assert_eq!(affine_band_group(262_144, 131_072, 5376, 0), None);
    }
}
