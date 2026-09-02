//! `Layout`: gathers, cuts, and slices — data movement with no arithmetic.
//! One entry per IR variant, plus the quantized embed-gather the driver
//! selects when the table is an affine bank.

use crate::error::Error;
use dtype::Dtype;

use crate::encode::{
    Arg, Ctx, Fire, Grid, dtype_dispatch, elementwise_rows, head_grid, head_group, nonzero,
    refuse, stated,
};
use crate::tensor::{Bank, Tensor};

pub fn embed(
    ctx: &Ctx<'_>,
    ids: Tensor,
    table: Tensor,
    vocab: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.embed";
    debug_assert_eq!(ids.dtype, Dtype::I32, "`{OP}` gathers by i32 token ids");
    let entry = dtype_dispatch!(OP, table.dtype, { Bf16 => "embed_bfloat16" });
    nonzero(OP, "the row count this embedding table states", vocab)?;
    debug_assert_eq!(
        ids.rows, y.rows,
        "the token ids handed over are the rows this gather lands"
    );
    ctx.fire(
        Fire::at("layout/embed.metal", entry).apply(Grid::of(
            elementwise_rows(OP, y.width, y.rows)?,
            [256, 1, 1],
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

/// The concatenating gather's geometry (qwen4's PLE): `ids` is one row of
/// `heads` ids per token, `y` is `heads` table rows laid side by side, so this
/// is [`embed`] with the head axis folded into the row axis — `(rows · heads)`
/// slices of `y.width / heads`.
fn concat_slices(op: &'static str, ids: Tensor, y: Tensor) -> Result<(u32, u32), Error> {
    let heads = nonzero(op, "the ids per row", ids.width)?;
    if y.width == 0 || y.width % heads != 0 {
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

/// `heads` gathers per row, concatenated — the dense table.
pub fn embed_concat(
    ctx: &Ctx<'_>,
    ids: Tensor,
    table: Tensor,
    vocab: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.embed_concat";
    debug_assert_eq!(ids.dtype, Dtype::I32, "`{OP}` gathers by i32 token ids");
    let entry = dtype_dispatch!(OP, table.dtype, { Bf16 => "embed_concat_bfloat16" });
    nonzero(OP, "the row count this embedding table states", vocab)?;
    let (slices, width) = concat_slices(OP, ids, y)?;
    ctx.fire(
        Fire::at("layout/embed.metal", entry).apply(Grid::of(
            elementwise_rows(OP, width, slices)?,
            [256, 1, 1],
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

/// `heads` gathers per row, concatenated — the affine bank, dequantized for
/// exactly the rows touched and never landed dense. Otherwise identical to
/// [`embed_gather_mb_4bit`].
pub fn embed_concat_mb_4bit(
    ctx: &Ctx<'_>,
    ids: Tensor,
    table: Bank,
    vocab: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.embed_concat";
    debug_assert_eq!(ids.dtype, Dtype::I32, "`{OP}` gathers by i32 token ids");
    let (slices, width) = concat_slices(OP, ids, y)?;
    gather_mb_4bit(ctx, OP, ids, table, vocab, y, slices, width)
}

pub fn split_qkv(
    ctx: &Ctx<'_>,
    packed: Tensor,
    q_width: u32,
    kv_width: u32,
    q: Tensor,
    k: Tensor,
    v: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.split_qkv";
    let entry = dtype_dispatch!(OP, packed.dtype, { Bf16 => "split_qkv_bf16" });
    ctx.fire(
        Fire::at("attn/split_qkv.metal", entry).apply(Grid::of(
            elementwise_rows(OP, packed.width, packed.rows)?,
            [256, 1, 1],
        )),
        &[
            packed.arg(),
            q.arg_mut(),
            k.arg_mut(),
            v.arg_mut(),
            q_width.arg(),
            kv_width.arg(),
        ],
    )
}

/// Deinterleaves per-head `(q, gate)` pairs from the packed projection.
pub fn split_q_gate(
    ctx: &Ctx<'_>,
    packed: Tensor,
    head_dim: u32,
    q: Tensor,
    gate: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.split_q_gate";
    let entry = dtype_dispatch!(OP, packed.dtype, { Bf16 => "q_gate_split_bfloat16" });
    nonzero(OP, "the head width this cut walks", head_dim)?;
    if q.width == 0 || q.width % head_dim != 0 {
        return Err(refuse(
            OP,
            format!(
                "the {}-wide query half does not divide by the stated head width {head_dim}",
                q.width
            ),
        ));
    }
    let lanes = head_grid(OP, head_dim, q.width / head_dim, packed.rows)?;
    ctx.fire(
        Fire::at("elemwise/gate.metal", entry).apply(Grid::of(lanes, head_group(lanes))),
        &[
            packed.arg(),
            q.arg_mut(),
            gate.arg_mut(),
            stated(OP, head_dim)?.arg(),
            stated(OP, packed.width)?.arg(),
            stated(OP, q.width)?.arg(),
        ],
    )
}

/// Splits each row at column `width`.
pub fn split_rows(
    ctx: &Ctx<'_>,
    x: Tensor,
    width: u32,
    left: Tensor,
    right: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.split_rows";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "split_rows_bfloat16" });
    nonzero(OP, "the left half of this cut", left.width)?;
    nonzero(OP, "the right half of this cut", right.width)?;
    debug_assert_eq!(left.width, width, "the left half is the width this cut states");
    debug_assert_eq!(
        left.width + right.width,
        x.width,
        "the two halves cover the packed row"
    );
    ctx.fire(
        Fire::at("layout/deinterleave.metal", entry).apply(Grid::of(
            elementwise_rows(OP, x.width, x.rows)?,
            [256, 1, 1],
        )),
        &[
            x.arg(),
            left.arg_mut(),
            right.arg_mut(),
            stated(OP, left.width)?.arg(),
            stated(OP, right.width)?.arg(),
        ],
    )
}

/// Copies layer `layer`'s `width`-wide slice out of a stacked table.
pub fn select(
    ctx: &Ctx<'_>,
    table: Tensor,
    layer: u32,
    width: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.select";
    let entry = dtype_dispatch!(OP, table.dtype, { Bf16 => "select_slice_bfloat16" });
    nonzero(OP, "the slice width this select states", width)?;
    debug_assert_eq!(
        y.width, width,
        "the selected slice is the width the statement states"
    );
    let offset = layer.checked_mul(width).ok_or_else(|| {
        refuse(
            OP,
            format!("layer {layer}'s slice starts beyond any column: {layer} x {width}"),
        )
    })?;
    if offset
        .checked_add(width)
        .is_none_or(|end| end > table.width)
    {
        return Err(refuse(
            OP,
            format!(
                "the {}-wide relayed row does not reach layer {layer}'s slice at {offset}",
                table.width
            ),
        ));
    }
    ctx.fire(
        Fire::at("layout/deinterleave.metal", entry).apply(Grid::of(
            elementwise_rows(OP, y.width, y.rows)?,
            [256, 1, 1],
        )),
        &[
            table.arg(),
            y.arg_mut(),
            stated(OP, table.width)?.arg(),
            stated(OP, offset)?.arg(),
            stated(OP, width)?.arg(),
        ],
    )
}

fn affine_point(op: &'static str, group: i32, bits: i32) -> Result<usize, Error> {
    let g = match group {
        32 => 0,
        64 => 1,
        128 => 2,
        _ => {
            return Err(refuse(op, format!("no affine point at group size {group}")));
        }
    };
    let b = match bits {
        4 => 0,
        8 => 1,
        _ => {
            return Err(refuse(op, format!("no affine point at bit width {bits}")));
        }
    };
    Ok(g * 2 + b)
}

/// The embed gather over an affine-quantized table — what `layout.embed`
/// becomes when the driver resolves the table to a bank instead of a dense
/// weight. Dequantizes one row per token; the six stamped points differ only
/// in the `(group, bits)` pair.
pub fn embed_gather_mb_4bit(
    ctx: &Ctx<'_>,
    ids: Tensor,
    table: Bank,
    vocab: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.embed";
    debug_assert_eq!(
        ids.rows, y.rows,
        "the token ids handed over are the rows this gather lands"
    );
    gather_mb_4bit(ctx, OP, ids, table, vocab, y, y.rows, y.width)
}

/// The banked gather both embed points fire, over a slice count/width the
/// caller carved. `vocab` is passed since a bank reads three planes (codes,
/// scales, biases) per id, so an out-of-range id is three reads past the end.
fn gather_mb_4bit(
    ctx: &Ctx<'_>,
    op: &'static str,
    ids: Tensor,
    table: Bank,
    vocab: u32,
    y: Tensor,
    slices: u32,
    width: u32,
) -> Result<(), Error> {
    const ENTRIES: [&str; 6] = [
        "embed_gather_mb_4bit_bfloat16_gs_32_b_4",
        "embed_gather_mb_4bit_bfloat16_gs_32_b_8",
        "embed_gather_mb_4bit_bfloat16_gs_64_b_4",
        "embed_gather_mb_4bit_bfloat16_gs_64_b_8",
        "embed_gather_mb_4bit_bfloat16_gs_128_b_4",
        "embed_gather_mb_4bit_bfloat16_gs_128_b_8",
    ];
    debug_assert_eq!(ids.dtype, Dtype::I32, "`{op}` gathers by i32 token ids");
    nonzero(op, "the row count this embedding table states", vocab)?;
    // All six points are affine (`bias + code * scale` unconditionally), so a
    // symmetric table has no valid biases to read.
    let Some(biases) = table.biases else {
        return Err(refuse(
            op,
            format!(
                "the table is a symmetric {}-bit bank in groups of {}, and \
                 `embed_gather.metal` stamps the affine gather alone",
                table.bits, table.group
            ),
        ));
    };
    let point = affine_point(
        op,
        i32::try_from(table.group).unwrap_or(i32::MAX),
        i32::try_from(table.bits).unwrap_or(i32::MAX),
    )?;
    let _ = y;
    ctx.fire(
        Fire::at("layout/embed_gather.metal", ENTRIES[point]).apply(Grid::of(
            elementwise_rows(op, width, slices)?,
            [256, 1, 1],
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

/// The two halves of one row permutation, which differ only in which way the
/// index is read, so they share one body. The index dtype is required to be
/// `i32` to match the CUDA twin's convention, even though the shader itself
/// reads it as `uint` (same bits, non-negative).
fn move_rows(
    ctx: &Ctx<'_>,
    op: &'static str,
    entry: &'static str,
    wide: Tensor,
    tight: Tensor,
    index: Tensor,
    args: [Tensor; 2],
) -> Result<(), Error> {
    if index.dtype != Dtype::I32 {
        return Err(refuse(
            op,
            format!(
                "the fire rows this copy is handed are {:?}, and it reads an i32 row map",
                index.dtype
            ),
        ));
    }
    if index.rows != tight.rows {
        return Err(refuse(
            op,
            format!("{} rows to move and {} rows named", tight.rows, index.rows),
        ));
    }
    if wide.dtype != tight.dtype || wide.width != tight.width {
        return Err(refuse(
            op,
            format!(
                "the fire-wide rectangle is {} x {:?} and the compacted one {} x {:?}; \
                 a row copy does not reshape",
                wide.width, wide.dtype, tight.width, tight.dtype
            ),
        ));
    }
    let rows = nonzero(op, "rows to move", tight.rows)?;
    let width = nonzero(op, "the width of a row this copy moves", tight.width)?;
    ctx.fire(
        Fire::at("layout/row_gather.metal", entry)
            .apply(Grid::of(elementwise_rows(op, width, rows)?, [256, 1, 1])),
        &[
            args[0].arg(),
            args[1].arg_mut(),
            index.arg(),
            width.arg(),
            rows.arg(),
        ],
    )
}

/// Gather: the rows a fragmented window covers, laid down as one
/// (`Fallback::Copy`'s first half). Reads rows of `wide` in the order `index`
/// names and writes them contiguously into `tight`. `index` is `i32`, one
/// entry per row of `tight`, naming the fire row it stands at.
///
/// # Errors
///
/// bf16/f32 only; a refusal for a mismatched index or rectangle.
pub fn gather_rows(
    ctx: &Ctx<'_>,
    wide: Tensor,
    index: Tensor,
    tight: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.gather_rows";
    let entry = dtype_dispatch!(OP, tight.dtype, {
        Bf16 => "row_gather_bfloat16",
        F32 => "row_gather_float32",
    });
    move_rows(ctx, OP, entry, wide, tight, index, [wide, tight])
}

/// Scatter: the answers put back where their rows came from
/// (`Fallback::Copy`'s second half). The same map as [`gather_rows`] read the
/// other way: row `i` of `tight` lands at fire row `index[i]` of `wide`. Rows
/// the window does not cover are not written.
///
/// # Errors
///
/// As [`gather_rows`].
pub fn scatter_rows(
    ctx: &Ctx<'_>,
    tight: Tensor,
    index: Tensor,
    wide: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.scatter_rows";
    let entry = dtype_dispatch!(OP, tight.dtype, {
        Bf16 => "row_scatter_bfloat16",
        F32 => "row_scatter_float32",
    });
    move_rows(ctx, OP, entry, wide, tight, index, [tight, wide])
}

/// The two folds' shared arithmetic: how many rows one block is, and how many
/// whole blocks the source has. Floors rather than refusing a partial tail:
/// the preprocessor always rounds image dimensions down to a whole number of
/// blocks, so the dropped rows are padding.
fn fold_extent(op: &'static str, x: Tensor, side: u32) -> Result<(u32, u32), Error> {
    nonzero(op, "the folding square's side", side)?;
    nonzero(op, "the folded row's width", x.width)?;
    let block = side.checked_mul(side).ok_or_else(|| {
        refuse(
            op,
            format!("a {side}-wide folding square has no row count that fits a u32"),
        )
    })?;
    if x.rows < block {
        return Err(refuse(
            op,
            format!(
                "{} rows do not fill one {side}x{side} fold, and a fold with no whole block \
                 would leave the destination unwritten",
                x.rows
            ),
        ));
    }
    Ok((block, x.rows / block))
}

/// The spatial pool: `y[j]` is the mean of rows `[j·side², (j+1)·side²)` of
/// `x`, over `x.rows / side²` output rows. Requires the submission's patches
/// to be in pool-block-major order. Compacting: the tail of `y` past
/// `x.rows / side²` rows is not written.
///
/// # Errors
///
/// bf16 only; refusals for zero `side`, mismatched widths, too few rows, or
/// a destination too short.
pub fn pool_rows(ctx: &Ctx<'_>, x: Tensor, side: u32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "layout.pool_rows";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "pool_rows_bfloat16" });
    debug_assert_eq!(y.dtype, x.dtype, "`{OP}` pools into the element it reads");

    let (block, out) = fold_extent(OP, x, side)?;
    if y.width != x.width {
        return Err(refuse(
            OP,
            format!(
                "the source rows are {} wide and the destination's are {}; a pool folds rows \
                 and never a row",
                x.width, y.width
            ),
        ));
    }
    if y.rows < out {
        return Err(refuse(
            OP,
            format!(
                "{} source rows fold into {out} pooled rows and the destination holds {}",
                x.rows, y.rows
            ),
        ));
    }
    let lanes = elementwise_rows(OP, x.width, out)?;
    ctx.fire(
        Fire::at("layout/fold.metal", entry).apply(Grid::of(lanes, [lanes[0].min(256), 1, 1])),
        &[
            x.arg(),
            y.arg_mut(),
            stated(OP, x.width)?.arg(),
            stated(OP, block)?.arg(),
        ],
    )
}

/// The merging fold: `y[j]` is rows `[j·side², (j+1)·side²)` of `x` laid end
/// to end — `side²` rows of `width` becoming one row of `side²·width`. Same
/// patch order and compacting tail rule as [`pool_rows`].
///
/// # Errors
///
/// As [`pool_rows`], plus a refusal if the destination width is not `side²`
/// times the source's.
pub fn merge_rows(ctx: &Ctx<'_>, x: Tensor, side: u32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "layout.merge_rows";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "merge_rows_bfloat16" });
    debug_assert_eq!(y.dtype, x.dtype, "`{OP}` merges into the element it reads");

    let (block, out) = fold_extent(OP, x, side)?;
    let merged = block.checked_mul(x.width).ok_or_else(|| {
        refuse(
            OP,
            format!(
                "{block} rows of {} do not concatenate into a row that fits a u32",
                x.width
            ),
        )
    })?;
    if y.width != merged {
        return Err(refuse(
            OP,
            format!(
                "{block} rows of {} concatenate into {merged}, and the destination's rows \
                 are {} wide",
                x.width, y.width
            ),
        ));
    }
    if y.rows < out {
        return Err(refuse(
            OP,
            format!(
                "{} source rows fold into {out} merged rows and the destination holds {}",
                x.rows, y.rows
            ),
        ));
    }
    let lanes = elementwise_rows(OP, merged, out)?;
    ctx.fire(
        Fire::at("layout/fold.metal", entry).apply(Grid::of(lanes, [lanes[0].min(256), 1, 1])),
        &[x.arg(), y.arg_mut(), stated(OP, merged)?.arg()],
    )
}

/// The embed merge, with a drop sentinel: row `i` of `src` lands at token row
/// `routes[i]` of `y`; any negative `routes[i]` places it nowhere (`-1` is
/// the sentinel a submission writes). The upper bound (`route < y.rows`) is
/// checked upstream, not here.
///
/// # Errors
///
/// As [`scatter_rows`].
pub fn scatter_live_rows(
    ctx: &Ctx<'_>,
    src: Tensor,
    routes: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.scatter_live_rows";
    let entry = dtype_dispatch!(OP, src.dtype, {
        Bf16 => "row_scatter_live_bfloat16",
        F32 => "row_scatter_live_float32",
    });
    move_rows(ctx, OP, entry, y, src, routes, [src, y])
}

/// The gather that interpolates: `y[r] = Σₜ weights[r, t] · table[ids[r, t]]`.
/// `ids` is `[rows, taps]` `i32`, `weights` is `[rows, taps]` `f32`; `taps` is
/// read off their width (2 for a separable read, 4 for bilinear, 16 for
/// bicubic). `vocab` is the table's row count.
///
/// # Errors
///
/// Table must be bf16, `ids` i32, `weights` f32; refusals for mismatched
/// geometry, zero taps, or an empty output.
pub fn embed_weighted(
    ctx: &Ctx<'_>,
    ids: Tensor,
    weights: Tensor,
    table: Tensor,
    vocab: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.embed_weighted";
    let entry = dtype_dispatch!(OP, table.dtype, { Bf16 => "embed_weighted_bfloat16" });
    debug_assert_eq!(y.dtype, table.dtype, "`{OP}` gathers into the table's element");

    if ids.dtype != Dtype::I32 {
        return Err(refuse(
            OP,
            format!(
                "the taps this gather is handed are {:?}, and it reads i32 rows",
                ids.dtype
            ),
        ));
    }
    // Weights are the preprocessor's arithmetic, not the activation's: a bf16
    // weight would move the resample more than the gather it feeds.
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
    ctx.fire(
        Fire::at("layout/embed_weighted.metal", entry).apply(Grid::of(
            elementwise_rows(OP, y.width, y.rows)?,
            [256, 1, 1],
        )),
        &[
            ids.arg(),
            weights.arg(),
            table.arg(),
            y.arg_mut(),
            stated(OP, y.width)?.arg(),
            vocab.arg(),
            taps.arg(),
        ],
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    
    use crate::probe::Probe;

    fn bf16(buf: u32, rows: u32, width: u32) -> Tensor {
        Tensor::new(buf, rows, width, Dtype::Bf16)
    }

    fn map(rows: u32) -> Tensor {
        Tensor::new(7, rows, 1, Dtype::I32)
    }

    /// A four-bit affine bank of `vocab` rows at `width`, in groups of 32 —
    /// the shape qwen4's PLE table lands as.
    fn u4_bank(vocab: u32, width: u32) -> Bank {
        Bank {
            codes: Tensor::new(10, vocab, width / 8, Dtype::U32),
            scales: Tensor::new(11, vocab, width / 32, Dtype::Bf16),
            biases: Some(Tensor::new(12, vocab, width / 32, Dtype::Bf16)),
            group: 32,
            bits: 4,
        }
    }

    /// A table of no rows is refused up front, not on the device.
    #[test]
    fn a_banked_table_of_no_rows_is_refused_by_name() {
        let probe = Probe::default();
        let why = embed_gather_mb_4bit(
            &probe,
            Tensor::new(3, 8, 1, Dtype::I32),
            u4_bank(4096, 64),
            0,
            bf16(2, 8, 64),
        )
        .expect_err("a table of no rows");
        assert!(format!("{why}").contains("row count this embedding table states"), "{why}");
        assert!(probe.fires().is_empty());
    }

    #[test]
    fn a_row_map_that_is_not_an_i32_vector_is_refused_by_name() {
        let probe = Probe::default();
        let why = scatter_rows(
            &probe,
            bf16(2, 8, 64),
            Tensor::new(7, 8, 1, Dtype::U32),
            bf16(1, 64, 64),
        )
        .expect_err("the row map is i32 on both planes");
        assert!(format!("{why}").contains("i32 row map"), "{why}");
        assert!(probe.fires().is_empty());
    }

    #[test]
    fn a_map_that_names_a_different_number_of_rows_is_refused_by_name() {
        let probe = Probe::default();
        let why = gather_rows(&probe, bf16(1, 64, 64), map(7), bf16(2, 8, 64))
            .expect_err("eight rows to move and seven named");
        assert!(format!("{why}").contains("rows named"), "{why}");
    }

    #[test]
    fn a_copy_does_not_reshape() {
        let probe = Probe::default();
        let why = gather_rows(&probe, bf16(1, 64, 128), map(8), bf16(2, 8, 64))
            .expect_err("a row copy does not reshape");
        assert!(format!("{why}").contains("does not reshape"), "{why}");
    }

    /// Unlike the CUDA twin (dtype-blind), this plane only stamps the
    /// elements a copied region can hold.
    #[test]
    fn an_element_with_no_instantiation_is_refused_by_dtype() {
        let probe = Probe::default();
        let why = gather_rows(
            &probe,
            Tensor::new(1, 64, 8, Dtype::F16),
            map(8),
            Tensor::new(2, 8, 8, Dtype::F16),
        )
        .expect_err("this plane stamps the row copy for bf16 and f32");
        assert!(matches!(why, Error::DtypeUnsupported { .. }), "{why}");
    }

    /// A fold with no whole block is refused rather than launched at zero
    /// rows.
    #[test]
    fn a_rectangle_thinner_than_one_block_is_refused_by_name() {
        let probe = Probe::default();
        let why = pool_rows(&probe, bf16(1, 8, 64), 3, bf16(2, 8, 64))
            .expect_err("eight rows do not fill one 3x3 fold");
        assert!(format!("{why}").contains("do not fill one 3x3 fold"), "{why}");

        let merged = merge_rows(&probe, bf16(1, 3, 64), 2, bf16(2, 8, 256))
            .expect_err("three rows do not fill one 2x2 fold");
        assert!(format!("{merged}").contains("do not fill one 2x2 fold"), "{merged}");
        assert!(probe.fires().is_empty());
    }

    /// The two folds disagree about the destination's width.
    #[test]
    fn a_destination_of_the_wrong_width_is_refused_in_the_folds_own_words() {
        let probe = Probe::default();
        let pooled = pool_rows(&probe, bf16(1, 36, 64), 3, bf16(2, 4, 128))
            .expect_err("a pool does not widen a row");
        assert!(format!("{pooled}").contains("never a row"), "{pooled}");

        let merged = merge_rows(&probe, bf16(1, 36, 64), 3, bf16(2, 4, 64))
            .expect_err("a merge does widen a row, by exactly side²");
        assert!(format!("{merged}").contains("concatenate into 576"), "{merged}");
    }

    /// A destination too short to hold the blocks the source has is refused.
    #[test]
    fn a_destination_too_short_for_the_blocks_is_refused_by_name() {
        let probe = Probe::default();
        let why = pool_rows(&probe, bf16(1, 90, 64), 3, bf16(2, 4, 64))
            .expect_err("ten pooled rows do not fit four");
        assert!(format!("{why}").contains("the destination holds 4"), "{why}");
    }

}


/// **A run of bytes, moved on the device**, one 32-bit word a thread — the
/// recurrent buffer's scatter and gather (`engine_metal::rs`). `src` and
/// `dst` are minted at the run's own offsets; `bytes` must be a multiple of
/// four, which every activation row this crate lands is.
pub fn copy_words(ctx: &Ctx<'_>, src: Tensor, dst: Tensor, bytes: u64) -> Result<(), Error> {
    const OP: &str = "layout.rs_copy";
    if bytes == 0 {
        return Ok(());
    }
    if bytes % 4 != 0 {
        return Err(refuse(OP, format!("{bytes} bytes is not a whole number of 32-bit words")));
    }
    let words = u32::try_from(bytes / 4)
        .map_err(|_| refuse(OP, format!("{bytes} bytes is more than one launch addresses")))?;
    ctx.fire(
        Fire::at("layout/blit.metal", "rs_copy_words").apply(Grid::of([words, 1, 1], [words.min(256), 1, 1])),
        &[src.arg(), dst.arg_mut(), stated(OP, words)?.arg()],
    )
}
