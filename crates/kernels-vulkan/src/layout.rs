#![allow(unused_variables)]
#![allow(clippy::too_many_arguments)]

use crate::encode::{
    Arg, Ctx, Fire, Grid, dtype_dispatch, elementwise_rows, head_grid, nonzero, refuse, stated,
};
use crate::error::Error;
use crate::tensor::{Bank, Tensor};
use dtype::Dtype;

const GROUP: u32 = 256;

pub fn embed(
    ctx: &Ctx<'_>,
    ids: Tensor,
    table: Tensor,
    vocab: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.embed";
    debug_assert_eq!(ids.dtype, Dtype::I32, "`{OP}` gathers by i32 token ids");
    let entry = dtype_dispatch!(OP, table.dtype, { Bf16 => "embed_bf16" });
    nonzero(OP, "the row count this embedding table states", vocab)?;
    debug_assert_eq!(
        ids.rows, y.rows,
        "the token ids handed over are the rows this gather lands"
    );
    ctx.fire(
        Fire::at("layout/embed.slang", entry).apply(Grid::of(
            elementwise_rows(OP, y.width, y.rows)?,
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

pub fn argmax(ctx: &Ctx<'_>, x: Tensor, column: u32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "layout.argmax";
    debug_assert_eq!(y.dtype, Dtype::I32, "`{OP}` writes i32 column indices");
    let entry = dtype_dispatch!(OP, x.dtype, {
        Bf16 => "argmax_rows_bf16",
        F32 => "argmax_rows_f32",
    });
    let rows = nonzero(OP, "rows", x.rows)?;
    nonzero(OP, "width", x.width)?;
    if column >= y.width {
        return Err(refuse(
            OP,
            format!(
                "column {column} is outside the {}-wide plane it writes",
                y.width
            ),
        ));
    }
    debug_assert_eq!(x.rows, y.rows, "an argmax lands one entry per row");
    ctx.fire(
        Fire::at("layout/argmax.slang", entry).apply(Grid::of([GROUP, rows, 1], [GROUP, 1, 1])),
        &[
            x.arg(),
            y.arg_mut(),
            stated(OP, x.width)?.arg(),
            stated(OP, y.width)?.arg(),
            stated(OP, column)?.arg(),
        ],
    )
}

pub fn embed_concat(
    ctx: &Ctx<'_>,
    ids: Tensor,
    table: Tensor,
    vocab: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.embed_concat";
    debug_assert_eq!(ids.dtype, Dtype::I32, "`{OP}` gathers by i32 token ids");
    let entry = dtype_dispatch!(OP, table.dtype, { Bf16 => "embed_concat_bf16" });
    nonzero(OP, "the row count this embedding table states", vocab)?;
    let (slices, width) = concat_slices(OP, ids, y)?;
    ctx.fire(
        Fire::at("layout/embed_concat.slang", entry).apply(Grid::of(
            elementwise_rows(OP, width, slices)?,
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
    if q_width.saturating_add(kv_width.saturating_mul(2)) != packed.width {
        return Err(refuse(
            OP,
            format!(
                "the {}-wide packed row is not q {q_width} + 2 x kv {kv_width}",
                packed.width
            ),
        ));
    }
    ctx.fire(
        Fire::at("layout/split_qkv.slang", entry).apply(Grid::of(
            elementwise_rows(OP, packed.width, packed.rows)?,
            [GROUP, 1, 1],
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

pub fn split_q_gate(
    ctx: &Ctx<'_>,
    packed: Tensor,
    head_dim: u32,
    q: Tensor,
    gate: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.split_q_gate";
    let entry = dtype_dispatch!(OP, packed.dtype, { Bf16 => "q_gate_split_bf16" });
    nonzero(OP, "the head width this cut walks", head_dim)?;
    if q.width == 0 || !q.width.is_multiple_of(head_dim) {
        return Err(refuse(
            OP,
            format!(
                "the {}-wide query half does not divide by the stated head width {head_dim}",
                q.width
            ),
        ));
    }
    if head_dim > GROUP {
        return Err(refuse(
            OP,
            format!("the head width {head_dim} is above the {GROUP} lanes a group holds"),
        ));
    }
    let lanes = head_grid(OP, head_dim, q.width / head_dim, packed.rows)?;
    ctx.fire(
        Fire::at("layout/gate_split.slang", entry).apply(Grid::of(lanes, [GROUP, 1, 1])),
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

pub fn split_rows(
    ctx: &Ctx<'_>,
    x: Tensor,
    width: u32,
    left: Tensor,
    right: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.split_rows";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "split_rows_bf16" });
    nonzero(OP, "the left half of this cut", left.width)?;
    nonzero(OP, "the right half of this cut", right.width)?;
    debug_assert_eq!(
        left.width, width,
        "the left half is the width this cut states"
    );
    if left.width.saturating_add(right.width) != x.width {
        return Err(refuse(
            OP,
            format!(
                "the halves {} + {} do not cover the {}-wide packed row",
                left.width, right.width, x.width
            ),
        ));
    }
    ctx.fire(
        Fire::at("layout/deinterleave.slang", entry).apply(Grid::of(
            elementwise_rows(OP, x.width, x.rows)?,
            [GROUP, 1, 1],
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

pub fn select(
    ctx: &Ctx<'_>,
    table: Tensor,
    layer: u32,
    width: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.select";
    let entry = dtype_dispatch!(OP, table.dtype, { Bf16 => "select_slice_bf16" });
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
        Fire::at("layout/deinterleave.slang", entry).apply(Grid::of(
            elementwise_rows(OP, y.width, y.rows)?,
            [GROUP, 1, 1],
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

#[allow(clippy::too_many_arguments)]
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
    debug_assert_eq!(ids.dtype, Dtype::I32, "`{op}` gathers by i32 token ids");
    nonzero(op, "the row count this embedding table states", vocab)?;
    let Some(biases) = table.biases else {
        return Err(refuse(
            op,
            format!(
                "the table is a symmetric {}-bit bank in groups of {}, and \
                 `embed_gather.slang` instantiates the affine gather alone",
                table.bits, table.group
            ),
        ));
    };
    let entry = match (table.group, table.bits) {
        (32, 2) => "embed_gather_mb_bf16_gs_32_b_2",
        (32, 4) => "embed_gather_mb_bf16_gs_32_b_4",
        (32, 8) => "embed_gather_mb_bf16_gs_32_b_8",
        (64, 2) => "embed_gather_mb_bf16_gs_64_b_2",
        (64, 4) => "embed_gather_mb_bf16_gs_64_b_4",
        (64, 8) => "embed_gather_mb_bf16_gs_64_b_8",
        (128, 2) => "embed_gather_mb_bf16_gs_128_b_2",
        (128, 4) => "embed_gather_mb_bf16_gs_128_b_4",
        (128, 8) => "embed_gather_mb_bf16_gs_128_b_8",
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
    ctx.fire(
        Fire::at("layout/embed_gather.slang", entry).apply(Grid::of(
            elementwise_rows(op, width, slices)?,
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
                "the fire-wide rectangle is {} x {:?} and the compacted one {} x {:?}; a row copy \
                 does not reshape",
                wide.width, wide.dtype, tight.width, tight.dtype
            ),
        ));
    }
    let rows = nonzero(op, "rows to move", tight.rows)?;
    let width = nonzero(op, "the width of a row this copy moves", tight.width)?;
    ctx.fire(
        Fire::at("layout/row_gather.slang", entry)
            .apply(Grid::of(elementwise_rows(op, width, rows)?, [GROUP, 1, 1])),
        &[
            args[0].arg(),
            args[1].arg_mut(),
            index.arg(),
            width.arg(),
            rows.arg(),
        ],
    )
}

pub fn gather_rows(ctx: &Ctx<'_>, wide: Tensor, index: Tensor, tight: Tensor) -> Result<(), Error> {
    const OP: &str = "layout.gather_rows";
    let entry = dtype_dispatch!(OP, tight.dtype, {
        Bf16 => "row_gather_bf16",
        F32 => "row_gather_f32",
    });
    move_rows(ctx, OP, entry, wide, tight, index, [wide, tight])
}

pub fn scatter_rows(
    ctx: &Ctx<'_>,
    tight: Tensor,
    index: Tensor,
    wide: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.scatter_rows";
    let entry = dtype_dispatch!(OP, tight.dtype, {
        Bf16 => "row_scatter_bf16",
        F32 => "row_scatter_f32",
    });
    move_rows(ctx, OP, entry, wide, tight, index, [tight, wide])
}

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
                "{} rows do not fill one {side}x{side} fold, and a fold with no whole block would \
                 leave the destination unwritten",
                x.rows
            ),
        ));
    }
    Ok((block, x.rows / block))
}

pub fn pool_rows(ctx: &Ctx<'_>, x: Tensor, side: u32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "layout.pool_rows";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "pool_rows_bf16" });
    debug_assert_eq!(y.dtype, x.dtype, "`{OP}` pools into the element it reads");
    let (block, out) = fold_extent(OP, x, side)?;
    if y.width != x.width {
        return Err(refuse(
            OP,
            format!(
                "the source rows are {} wide and the destination's are {}; a pool folds rows and \
                 never a row",
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
    ctx.fire(
        Fire::at("layout/fold.slang", entry)
            .apply(Grid::of(elementwise_rows(OP, x.width, out)?, [GROUP, 1, 1])),
        &[
            x.arg(),
            y.arg_mut(),
            stated(OP, x.width)?.arg(),
            stated(OP, block)?.arg(),
            stated(OP, out)?.arg(),
        ],
    )
}

pub fn merge_rows(ctx: &Ctx<'_>, x: Tensor, side: u32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "layout.merge_rows";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "merge_rows_bf16" });
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
                "{block} rows of {} concatenate into {merged}, and the destination's rows are {} wide",
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
    ctx.fire(
        Fire::at("layout/fold.slang", entry)
            .apply(Grid::of(elementwise_rows(OP, merged, out)?, [GROUP, 1, 1])),
        &[
            x.arg(),
            y.arg_mut(),
            stated(OP, merged)?.arg(),
            stated(OP, out)?.arg(),
        ],
    )
}

pub fn scatter_live_rows(
    ctx: &Ctx<'_>,
    src: Tensor,
    routes: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.scatter_live_rows";
    let entry = dtype_dispatch!(OP, src.dtype, {
        Bf16 => "row_scatter_live_bf16",
        F32 => "row_scatter_live_f32",
    });
    move_rows(ctx, OP, entry, y, src, routes, [src, y])
}

pub fn embed_weighted(
    ctx: &Ctx<'_>,
    ids: Tensor,
    weights: Tensor,
    table: Tensor,
    vocab: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "layout.embed_weighted";
    let entry = dtype_dispatch!(OP, table.dtype, { Bf16 => "embed_weighted_bf16" });
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
    ctx.fire(
        Fire::at("layout/embed_weighted.slang", entry).apply(Grid::of(
            elementwise_rows(OP, y.width, rows)?,
            [GROUP, 1, 1],
        )),
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

pub fn copy_words(ctx: &Ctx<'_>, src: Tensor, dst: Tensor, bytes: u64) -> Result<(), Error> {
    const OP: &str = "layout.rs_copy";
    if bytes == 0 {
        return Ok(());
    }
    if !bytes.is_multiple_of(4) {
        return Err(refuse(
            OP,
            format!("{bytes} bytes is not a whole number of 32-bit words"),
        ));
    }
    let words = u32::try_from(bytes / 4).map_err(|_| {
        refuse(
            OP,
            format!("{bytes} bytes is more than one launch addresses"),
        )
    })?;
    ctx.fire(
        Fire::at("layout/copy.slang", "copy_words").threads([words, 1, 1], [256, 1, 1]),
        &[src.arg(), dst.arg_mut(), stated(OP, words)?.arg()],
    )
}
