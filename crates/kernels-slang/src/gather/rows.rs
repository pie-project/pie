//! Row movement (`gather.rows`, `scatter_rows`, `scatter_live_rows`,
//! `pack_rows`, `unpack_rows`, `pool_rows`, `merge_rows`, `select`,
//! `split_rows`, `split_qkv`, `split_q_gate`, `rs_copy`) from
//! `kernels/gather/rows.slang`. The movers copy 32-bit words: a bf16 row is
//! half as many words as values (so an even width), an f32 or i32 row one
//! word a value.

use dtype::Dtype;

use crate::encode::{
    Arg, Ctx, Fire, Grid, dtype_dispatch, elementwise_rows, head_grid, nonzero, refuse, stated,
};
use crate::error::Error;
use crate::tensor::Tensor;

const ROWS: &str = "gather/rows.slang";

/// The workgroup width every instantiation is built at.
const GROUP: u32 = 256;

/// The words a `width`-wide row of `dtype` spans.
fn words(op: &'static str, what: &str, dtype: Dtype, width: u32) -> Result<u32, Error> {
    match dtype {
        Dtype::Bf16 if width.is_multiple_of(2) => Ok(width / 2),
        Dtype::Bf16 => Err(refuse(
            op,
            format!("the {what} is {width} wide, and a bf16 row moves as whole words"),
        )),
        Dtype::F32 | Dtype::I32 => Ok(width),
        other => Err(Error::DtypeUnsupported { op, dtype: other }),
    }
}

fn grid(op: &'static str, words: u32, rows: u32) -> Result<Grid, Error> {
    Ok(Grid::of(elementwise_rows(op, words, rows)?, [GROUP, 1, 1]))
}

/// `bytes` copied from `src` to `dst`, whole words.
pub fn copy_words(ctx: &Ctx<'_>, src: Tensor, dst: Tensor, bytes: u64) -> Result<(), Error> {
    const OP: &str = "gather.rs_copy";
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
        Fire::at(ROWS, "gather_rows_copy_words").threads([words, 1, 1], [GROUP, 1, 1]),
        &[src.arg(), dst.arg_mut(), stated(OP, words)?.arg()],
    )
}

/// Rows moved between the fire-wide rectangle and a compacted one by an i32
/// row map: `args` are (read, written).
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
    let pitch = words(op, "row this copy moves", tight.dtype, width)?;
    ctx.fire(
        Fire::at(ROWS, entry).apply(grid(op, pitch, rows)?),
        &[
            args[0].arg(),
            args[1].arg_mut(),
            index.arg(),
            pitch.arg(),
            rows.arg(),
        ],
    )
}

/// `tight[i] = wide[index[i]]`.
pub fn gather_rows(ctx: &Ctx<'_>, wide: Tensor, index: Tensor, tight: Tensor) -> Result<(), Error> {
    move_rows(
        ctx,
        "gather.rows",
        "gather_rows_words",
        wide,
        tight,
        index,
        [wide, tight],
    )
}

/// `wide[index[i]] = tight[i]`.
pub fn scatter_rows(
    ctx: &Ctx<'_>,
    tight: Tensor,
    index: Tensor,
    wide: Tensor,
) -> Result<(), Error> {
    move_rows(
        ctx,
        "gather.scatter_rows",
        "gather_rows_scatter_words",
        wide,
        tight,
        index,
        [tight, wide],
    )
}

/// `y[routes[i]] = src[i]`, a negative route dropping its row.
pub fn scatter_live_rows(
    ctx: &Ctx<'_>,
    src: Tensor,
    routes: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    move_rows(
        ctx,
        "gather.scatter_live_rows",
        "gather_rows_scatter_live_words",
        y,
        src,
        routes,
        [src, y],
    )
}

fn permutation(op: &'static str, perm: Tensor, rows: u32) -> Result<Tensor, Error> {
    if u64::from(perm.rows) * u64::from(perm.width) < u64::from(rows) {
        return Err(refuse(
            op,
            format!(
                "a {} x {} permutation for {rows} rows",
                perm.rows, perm.width
            ),
        ));
    }
    Ok(Tensor::new(perm.buf, rows, 1, perm.dtype))
}

/// `o[i] = x[perm[i]]`.
pub fn pack_rows(ctx: &Ctx<'_>, x: Tensor, perm: Tensor, o: Tensor) -> Result<(), Error> {
    let index = permutation("gather.pack_rows", perm, o.rows)?;
    gather_rows(ctx, x, index, o)
}

/// `o[perm[i]] = x[i]`.
pub fn unpack_rows(ctx: &Ctx<'_>, x: Tensor, perm: Tensor, o: Tensor) -> Result<(), Error> {
    let index = permutation("gather.unpack_rows", perm, x.rows)?;
    scatter_rows(ctx, x, index, o)
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

/// Each `side * side` block of rows averaged into one.
pub fn pool_rows(ctx: &Ctx<'_>, x: Tensor, side: u32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "gather.pool_rows";
    dtype_dispatch!(OP, x.dtype, { Bf16 => () });
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
    let pitch = words(OP, "folded row", x.dtype, x.width)?;
    ctx.fire(
        Fire::at(ROWS, "gather_rows_pool_bf16").apply(grid(OP, pitch, out)?),
        &[x.arg(), y.arg_mut(), pitch.arg(), block.arg(), out.arg()],
    )
}

/// Each `side * side` block of rows concatenated into one: in memory, the
/// rows as they lie, so a flat copy of the whole blocks.
pub fn merge_rows(ctx: &Ctx<'_>, x: Tensor, side: u32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "gather.merge_rows";
    dtype_dispatch!(OP, x.dtype, { Bf16 => () });
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
    let pitch = words(OP, "merged row", x.dtype, merged)?;
    let total = pitch
        .checked_mul(out)
        .ok_or_else(|| refuse(OP, format!("{out} rows of {pitch} words will not launch")))?;
    ctx.fire(
        Fire::at(ROWS, "gather_rows_copy_words").threads([total, 1, 1], [GROUP, 1, 1]),
        &[x.arg(), y.arg_mut(), stated(OP, total)?.arg()],
    )
}

/// Layer `layer`'s `width`-wide slice of each row of a relayed table.
pub fn select(
    ctx: &Ctx<'_>,
    table: Tensor,
    layer: u32,
    width: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.select";
    dtype_dispatch!(OP, table.dtype, { Bf16 => () });
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
    let slice = words(OP, "slice", table.dtype, width)?;
    let start = words(OP, "slice offset", table.dtype, offset)?;
    let stride = words(OP, "relayed row", table.dtype, table.width)?;
    ctx.fire(
        Fire::at(ROWS, "gather_rows_select_words").apply(grid(OP, slice, y.rows)?),
        &[
            table.arg(),
            y.arg_mut(),
            stride.arg(),
            start.arg(),
            slice.arg(),
        ],
    )
}

/// `xs` side by side into `y`, a row each's columns after the one before's
/// (kernels-cuda `gather::concat_cols`): one placement an input.
pub fn concat_cols(ctx: &Ctx<'_>, xs: &[Tensor], y: Tensor) -> Result<(), Error> {
    const OP: &str = "gather.concat_cols";
    let total: u32 = xs.iter().map(|x| x.width).sum();
    if total != y.width || xs.iter().any(|x| x.dtype != y.dtype || x.rows < y.rows) {
        return Err(refuse(
            OP,
            "the inputs share the output's element and rows, and their widths sum to its",
        ));
    }
    if y.rows == 0 {
        return Ok(());
    }
    let stride = words(OP, "output row", y.dtype, y.width)?;
    let mut at = 0u32;
    for x in xs {
        let width = words(OP, "input row", x.dtype, x.width)?;
        let offset = words(OP, "column offset", y.dtype, at)?;
        if width > 0 {
            ctx.fire(
                Fire::at(ROWS, "gather_rows_place_words").apply(grid(OP, width, y.rows)?),
                &[
                    x.arg(),
                    y.arg_mut(),
                    stride.arg(),
                    offset.arg(),
                    width.arg(),
                ],
            )?;
        }
        at += x.width;
    }
    Ok(())
}

/// Each row of `x` cut into its first `width` columns and the rest.
pub fn split_rows(
    ctx: &Ctx<'_>,
    x: Tensor,
    width: u32,
    left: Tensor,
    right: Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.split_rows";
    dtype_dispatch!(OP, x.dtype, { Bf16 => () });
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
    let lw = words(OP, "left half", x.dtype, left.width)?;
    let rw = words(OP, "right half", x.dtype, right.width)?;
    ctx.fire(
        Fire::at(ROWS, "gather_rows_split_words").apply(grid(OP, lw + rw, x.rows)?),
        &[x.arg(), left.arg_mut(), right.arg_mut(), lw.arg(), rw.arg()],
    )
}

/// A packed `q | k | v` row cut into its three parts.
pub fn split_qkv(
    ctx: &Ctx<'_>,
    packed: Tensor,
    q_width: u32,
    kv_width: u32,
    q: Tensor,
    k: Tensor,
    v: Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.split_qkv";
    dtype_dispatch!(OP, packed.dtype, { Bf16 => () });
    if q_width.saturating_add(kv_width.saturating_mul(2)) != packed.width {
        return Err(refuse(
            OP,
            format!(
                "the {}-wide packed row is not q {q_width} + 2 x kv {kv_width}",
                packed.width
            ),
        ));
    }
    let qw = words(OP, "q part", packed.dtype, q_width)?;
    let kvw = words(OP, "kv part", packed.dtype, kv_width)?;
    ctx.fire(
        Fire::at(ROWS, "gather_rows_split_qkv_words").apply(grid(OP, qw + 2 * kvw, packed.rows)?),
        &[
            packed.arg(),
            q.arg_mut(),
            k.arg_mut(),
            v.arg_mut(),
            qw.arg(),
            kvw.arg(),
        ],
    )
}

/// Each `2 * head_dim`-wide head of a packed row cut into its query half and
/// its gate half.
pub fn split_q_gate(
    ctx: &Ctx<'_>,
    packed: Tensor,
    head_dim: u32,
    q: Tensor,
    gate: Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.split_q_gate";
    dtype_dispatch!(OP, packed.dtype, { Bf16 => () });
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
    let head = words(OP, "head", packed.dtype, head_dim)?;
    let packed_words = words(OP, "packed row", packed.dtype, packed.width)?;
    let q_words = words(OP, "query half", packed.dtype, q.width)?;
    let lanes = head_grid(OP, head, q.width / head_dim, packed.rows)?;
    ctx.fire(
        Fire::at(ROWS, "gather_rows_split_q_gate_words").apply(Grid::of(lanes, [GROUP, 1, 1])),
        &[
            packed.arg(),
            q.arg_mut(),
            gate.arg_mut(),
            head.arg(),
            packed_words.arg(),
            q_words.arg(),
        ],
    )
}
