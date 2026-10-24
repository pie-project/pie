//! Embedding lookups (`gather.embed`, `gather.embed`,
//! `gather.embed`) from `kernels/gather/embed.slang`, into bf16 rows
//! a word a lane: from a bf16 table or an affine-quantized bank.

use dtype::Dtype;

use crate::encode::{
    Arg, Ctx, Fire, Grid, dtype_dispatch, elementwise_rows, nonzero, refuse, stated,
};
use crate::error::Error;
use crate::tensor::{Bank, Tensor};

const EMBED: &str = "gather/embed.slang";

/// The workgroup width every instantiation is built at.
const GROUP: u32 = 256;

/// The words of a `width`-wide bf16 row.
fn words(op: &'static str, what: &str, width: u32) -> Result<u32, Error> {
    if width == 0 || !width.is_multiple_of(2) {
        return Err(refuse(
            op,
            format!("the {what} is {width} wide, and a bf16 row lands as whole words"),
        ));
    }
    Ok(width / 2)
}

/// `y[n] = table[ids[n]]`; an id outside `vocab` reads row 0.
pub fn embed(
    ctx: &Ctx<'_>,
    ids: Tensor,
    table: Tensor,
    vocab: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.embed";
    debug_assert_eq!(ids.dtype, Dtype::I32, "`{OP}` gathers by i32 token ids");
    let entry = dtype_dispatch!(OP, table.dtype, { Bf16 => "gather_embed_bf16" });
    nonzero(OP, "the row count this embedding table states", vocab)?;
    debug_assert_eq!(
        ids.rows, y.rows,
        "the token ids handed over are the rows this gather lands"
    );
    let words = words(OP, "embedding row", y.width)?;
    ctx.fire(
        Fire::at(EMBED, entry).apply(Grid::of(
            elementwise_rows(OP, words, y.rows)?,
            [GROUP, 1, 1],
        )),
        &[
            ids.arg(),
            table.arg(),
            y.arg_mut(),
            stated(OP, y.width)?.arg(),
            stated(OP, vocab)?.arg(),
        ],
    )
}

/// The ids of each row and the width of one table row, for `ids.width`
/// table rows side by side in each landed row.
fn concat_slices(op: &'static str, ids: Tensor, y: Tensor) -> Result<(u32, u32), Error> {
    let heads = nonzero(op, "the ids per row", ids.width)?;
    if y.width == 0 || !y.width.is_multiple_of(heads) {
        return Err(refuse(
            op,
            format!(
                "the {}-wide landing is not {heads} table rows side by side",
                y.width
            ),
        ));
    }
    debug_assert_eq!(
        ids.rows, y.rows,
        "the token rows handed over are the rows this gather lands"
    );
    let slices = ids.rows.checked_mul(heads).ok_or_else(|| {
        refuse(
            op,
            format!("the grid will not launch: {} rows x {heads} ids", ids.rows),
        )
    })?;
    Ok((slices, y.width / heads))
}

/// `ids.width` table rows side by side in each row of `y`; an id outside
/// `vocab` lands zeros.
pub fn embed_concat(
    ctx: &Ctx<'_>,
    ids: Tensor,
    table: Tensor,
    vocab: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.embed";
    debug_assert_eq!(ids.dtype, Dtype::I32, "`{OP}` gathers by i32 token ids");
    let entry = dtype_dispatch!(OP, table.dtype, { Bf16 => "gather_embed_concat_bf16" });
    nonzero(OP, "the row count this embedding table states", vocab)?;
    let (slices, width) = concat_slices(OP, ids, y)?;
    let words = words(OP, "table row", width)?;
    ctx.fire(
        Fire::at(EMBED, entry).apply(Grid::of(
            elementwise_rows(OP, words, slices)?,
            [GROUP, 1, 1],
        )),
        &[
            ids.arg(),
            table.arg(),
            y.arg_mut(),
            stated(OP, width)?.arg(),
            stated(OP, vocab)?.arg(),
        ],
    )
}

/// [`embed`] from an affine-quantized bank; an id outside `vocab` lands
/// zeros.
pub fn embed_banked(
    ctx: &Ctx<'_>,
    ids: Tensor,
    table: Bank,
    vocab: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.embed";
    debug_assert_eq!(
        ids.rows, y.rows,
        "the token ids handed over are the rows this gather lands"
    );
    banked(ctx, OP, ids, table, vocab, y, y.rows, y.width)
}

/// [`embed_concat`] from an affine-quantized bank.
pub fn embed_concat_banked(
    ctx: &Ctx<'_>,
    ids: Tensor,
    table: Bank,
    vocab: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.embed";
    let (slices, width) = concat_slices(OP, ids, y)?;
    banked(ctx, OP, ids, table, vocab, y, slices, width)
}

#[allow(clippy::too_many_arguments)]
fn banked(
    ctx: &Ctx<'_>,
    op: &'static str,
    ids: Tensor,
    table: Bank,
    vocab: u32,
    y: Tensor,
    slices: u32,
    width: u32,
) -> Result<(), Error> {
    debug_assert_eq!(ids.dtype, Dtype::I32, "`{op}` gathers by i32 token ids");
    nonzero(op, "the row count this embedding table states", vocab)?;
    let Some(biases) = table.biases else {
        return Err(refuse(
            op,
            format!(
                "the table is a symmetric {}-bit bank in groups of {}, and the gather \
                 is instantiated for affine banks alone",
                table.bits, table.group
            ),
        ));
    };
    let entry = match (table.group, table.bits) {
        (32, 2) => "gather_embed_affine_bf16_gs_32_b_2",
        (32, 4) => "gather_embed_affine_bf16_gs_32_b_4",
        (32, 8) => "gather_embed_affine_bf16_gs_32_b_8",
        (64, 2) => "gather_embed_affine_bf16_gs_64_b_2",
        (64, 4) => "gather_embed_affine_bf16_gs_64_b_4",
        (64, 8) => "gather_embed_affine_bf16_gs_64_b_8",
        (128, 2) => "gather_embed_affine_bf16_gs_128_b_2",
        (128, 4) => "gather_embed_affine_bf16_gs_128_b_4",
        (128, 8) => "gather_embed_affine_bf16_gs_128_b_8",
        (group, bits) => {
            return Err(refuse(
                op,
                format!("no gather is instantiated at group size {group}, {bits} bits"),
            ));
        }
    };
    if !width.is_multiple_of(table.group) {
        return Err(refuse(
            op,
            format!(
                "the {width}-wide row is not a whole number of {}-code groups",
                table.group
            ),
        ));
    }
    dtype_dispatch!(op, y.dtype, { Bf16 => () });
    let words = words(op, "gathered row", width)?;
    ctx.fire(
        Fire::at(EMBED, entry).apply(Grid::of(
            elementwise_rows(op, words, slices)?,
            [GROUP, 1, 1],
        )),
        &[
            table.codes.arg(),
            table.scales.arg(),
            biases.arg(),
            ids.arg(),
            y.arg_mut(),
            stated(op, width)?.arg(),
            stated(op, vocab)?.arg(),
        ],
    )
}

/// `y[n] = sum_t weights[n, t] * table[ids[n, t]]`; an id outside `vocab`
/// reads row 0.
pub fn embed_weighted(
    ctx: &Ctx<'_>,
    ids: Tensor,
    weights: Tensor,
    table: Tensor,
    vocab: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.embed";
    let entry = dtype_dispatch!(OP, table.dtype, { Bf16 => "gather_embed_weighted_bf16" });
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
                "the interpolation weights are {:?}, and this gather reads f32",
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
    let rows = nonzero(OP, "rows", y.rows)?;
    let words = words(OP, "gathered row", y.width)?;
    ctx.fire(
        Fire::at(EMBED, entry).apply(Grid::of(elementwise_rows(OP, words, rows)?, [GROUP, 1, 1])),
        &[
            ids.arg(),
            weights.arg(),
            table.arg(),
            y.arg_mut(),
            stated(OP, y.width)?.arg(),
            vocab.arg(),
            taps.arg(),
            stated(OP, rows)?.arg(),
        ],
    )
}
