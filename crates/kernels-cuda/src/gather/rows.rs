use crate::error::Error;
use crate::jit::{
    Arg, ArgValue, Ctx, Fire, Launch, aligned16, dtype_dispatch, kernel, nonzero, refuse, stated,
};
use crate::tensor::Tensor;
use dtype::Dtype;

const FILE: &str = "gather/rows.cu";

const BLOCK: u32 = 256;

const WARP: u32 = 32;

const VEC_WIDTH: u32 = 8;

fn route_rows(rows: u32, width: u32) -> Launch {
    const MAX_BLOCK: u32 = 1024;

    Launch::per_row(
        rows,
        width
            .div_ceil(WARP)
            .max(1)
            .saturating_mul(WARP)
            .min(MAX_BLOCK),
    )
}

#[cfg_attr(not(feature = "cuda"), allow(dead_code))]
pub(crate) fn gather_width_concat(
    ctx: &Ctx,
    src: u64,
    y: &mut Tensor,
    shard_width: u32,
    world: u32,
) -> Result<(), Error> {
    const OP: &str = "gather.gather_width_concat";
    dtype_dispatch!(OP, y.dtype, { Bf16 => () });
    let rows = nonzero(OP, "rows", y.rows)?;
    let shard = nonzero(OP, "the shard width", shard_width)?;
    let world = nonzero(OP, "the rank count", world)?;
    if shard.checked_mul(world) != Some(y.width) {
        return Err(refuse(
            OP,
            format!(
                "a {}-wide destination is not {world} shards of {shard}",
                y.width
            ),
        ));
    }
    let total = u64::from(rows) * u64::from(y.width);
    let blocks = u32::try_from(total.div_ceil(u64::from(BLOCK)))
        .map_err(|_| refuse(OP, format!("{total} lanes do not fit a 32-bit grid")))?;
    ctx.fire(
        OP,
        Fire::at(kernel(
            FILE,
            "::pie::gather::gather_width_concat<::pie::bf16>",
        ))
        .apply(Launch::grid([blocks, 1, 1], [BLOCK, 1, 1])),
        &[
            ArgValue::Ptr(src),
            y.arg(),
            stated(OP, rows)?.arg(),
            stated(OP, shard)?.arg(),
            stated(OP, world)?.arg(),
        ],
    )
}

/// `packed`'s `heads` (query, key-value) heads, queries and keys
/// `widths[0]` wide and values `widths[1]`, cut into `q`, `k` and `v` heads
/// `head_dim` wide, zero past their widths.
#[allow(clippy::too_many_arguments)]
pub fn split_qkv_padded(
    ctx: &Ctx,
    packed: Tensor,
    heads: [u32; 2],
    widths: [u32; 2],
    head_dim: u32,
    q: &mut Tensor,
    k: &mut Tensor,
    v: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.split_qkv_padded";
    dtype_dispatch!(OP, packed.dtype, { Bf16 => () });
    let runs = [widths[0], widths[1], head_dim]
        .iter()
        .all(|w| w.is_multiple_of(8))
        && widths.iter().all(|&w| w <= head_dim)
        && [packed.ptr, q.ptr, k.ptr, v.ptr]
            .iter()
            .all(|p| p % 16 == 0);
    if !runs {
        return Err(refuse(
            OP,
            format!("{widths:?}-wide heads into {head_dim} are not whole 16-byte runs"),
        ));
    }
    let out8 = (heads[0] + 2 * heads[1]) * head_dim / 8;
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::gather::split_qkv_padded_vec8")).apply(Launch::grid(
            [q.rows, out8.div_ceil(BLOCK), 1],
            [BLOCK, 1, 1],
        )),
        &[
            packed.arg(),
            q.arg(),
            k.arg(),
            v.arg(),
            stated(OP, heads[0])?.arg(),
            stated(OP, heads[1])?.arg(),
            stated(OP, widths[0])?.arg(),
            stated(OP, widths[1])?.arg(),
            stated(OP, head_dim)?.arg(),
            ctx.stage(),
        ],
    )
}

pub fn split_qkv(
    ctx: &Ctx,
    packed: Tensor,
    q_width: u32,
    kv_width: u32,
    q: &mut Tensor,
    k: &mut Tensor,
    v: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.split_qkv";
    dtype_dispatch!(OP, packed.dtype, { Bf16 => () });
    debug_assert_eq!(q.width, q_width, "the q half is the width this cut states");
    debug_assert_eq!(
        k.width, kv_width,
        "the kv halves are the width this cut states"
    );
    let q_dim = stated(OP, q.width)?;
    let kv_dim = stated(OP, k.width)?;
    if q_dim <= 0 && kv_dim <= 0 {
        return Err(refuse(OP, "both halves of this cut are zero-wide"));
    }
    // Eight columns a thread where both cuts and every plane take whole
    // 16-byte runs: a copy, so the same bits land either way.
    let vec8 = q.width.is_multiple_of(8)
        && k.width.is_multiple_of(8)
        && [packed.ptr, q.ptr, k.ptr, v.ptr]
            .iter()
            .all(|p| p % 16 == 0);
    let (entry, width) = if vec8 {
        ("split_qkv_vec8", q.width.max(k.width) / 8)
    } else {
        ("split_qkv", q.width.max(k.width))
    };
    ctx.fire(
        OP,
        Fire::at(kernel(
            FILE,
            &format!("::pie::gather::{entry}<::pie::bf16>"),
        ))
        .apply(Launch::grid(
            [q.rows, width.div_ceil(BLOCK), 1],
            [BLOCK, 1, 1],
        )),
        &[
            packed.arg(),
            q.arg(),
            k.arg(),
            v.arg(),
            q_dim.arg(),
            kv_dim.arg(),
            ctx.stage(),
        ],
    )
}

pub fn split_q_gate(
    ctx: &Ctx,
    packed: Tensor,
    head_dim: u32,
    q: &mut Tensor,
    gate: &mut Tensor,
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
    let heads = q.width / head_dim;
    if ctx.tuning().rows_vec
        && head_dim.is_multiple_of(8)
        && [packed.ptr, q.ptr, gate.ptr]
            .iter()
            .all(|&at| aligned16(at))
    {
        let runs = head_dim / 8;
        let total = u64::from(q.rows) * u64::from(heads) * u64::from(runs);
        let blocks = u32::try_from(total.div_ceil(256))
            .map_err(|_| refuse(OP, format!("{total} runs overflow a grid")))?;
        return ctx.fire(
            OP,
            Fire::at(kernel(
                FILE,
                "::pie::gather::split_q_gate_vec8<::pie::bf16>",
            ))
            .apply(Launch::grid([blocks, 1, 1], [256, 1, 1])),
            &[
                packed.arg(),
                q.arg(),
                gate.arg(),
                (total as i64).arg(),
                stated(OP, heads)?.arg(),
                stated(OP, runs)?.arg(),
                ctx.stage(),
            ],
        );
    }
    let block = if head_dim < 128 { 64 } else { 128 };
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::gather::split_q_gate<::pie::bf16>"))
            .apply(Launch::grid([q.rows, heads, 1], [block, 1, 1])),
        &[
            packed.arg(),
            q.arg(),
            gate.arg(),
            stated(OP, q.rows)?.arg(),
            stated(OP, heads)?.arg(),
            stated(OP, head_dim)?.arg(),
            ctx.stage(),
        ],
    )
}

pub fn split_rows(
    ctx: &Ctx,
    x: Tensor,
    width: u32,
    left: &mut Tensor,
    right: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.split_rows";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F32 => "float" });
    debug_assert_eq!(
        left.width, width,
        "the left half is the width this cut states"
    );
    debug_assert_eq!(
        left.width + right.width,
        x.width,
        "the two halves cover the packed row"
    );
    let left_dim = stated(OP, nonzero(OP, "the left half of this cut", left.width)?)?;
    let right_dim = stated(OP, nonzero(OP, "the right half of this cut", right.width)?)?;
    let vectors = x.dtype == Dtype::Bf16
        && left.width.is_multiple_of(VEC_WIDTH)
        && right.width.is_multiple_of(VEC_WIDTH)
        && aligned16(x.ptr)
        && aligned16(left.ptr)
        && aligned16(right.ptr);
    let (entrypoint, launch) = if vectors {
        (
            "::pie::gather::split_rows_vec8<::pie::bf16>".to_string(),
            Launch::grid(
                [left.rows, (x.width / VEC_WIDTH).div_ceil(BLOCK), 1],
                [BLOCK, 1, 1],
            )
            .programmatic(ctx.tuning().pdl),
        )
    } else {
        (
            format!("::pie::gather::split_rows<{t}>"),
            route_rows(left.rows, left.width),
        )
    };
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, &entrypoint)).apply(launch),
        &[
            x.arg(),
            left.arg(),
            right.arg(),
            left_dim.arg(),
            right_dim.arg(),
            ctx.stage(),
        ],
    )
}

pub fn select(
    ctx: &Ctx,
    table: Tensor,
    layer: u32,
    width: u32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.select";
    let t = dtype_dispatch!(OP, table.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
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
        OP,
        Fire::at(kernel(FILE, &format!("::pie::gather::select<{t}>")))
            .apply(route_rows(y.rows, width)),
        &[
            table.arg(),
            y.arg(),
            stated(OP, table.width)?.arg(),
            stated(OP, offset)?.arg(),
            stated(OP, width)?.arg(),
            ctx.stage(),
        ],
    )
}

fn unit(bytes: u64, a: u64, b: u64) -> (&'static str, u64) {
    if bytes.is_multiple_of(16) && aligned16(a) && aligned16(b) {
        ("::int4", 16)
    } else if bytes.is_multiple_of(4) && a.is_multiple_of(4) && b.is_multiple_of(4) {
        ("::pie::i32", 4)
    } else {
        ("::pie::u8", 1)
    }
}

fn row_bytes(op: &'static str, handle: Tensor) -> Result<u64, Error> {
    let elem = match handle.dtype {
        Dtype::Bf16 | Dtype::F16 => 2,
        Dtype::F32 | Dtype::I32 | Dtype::U32 => 4,
        Dtype::U8 | Dtype::I8 | Dtype::E4m3 | Dtype::E8m0 => 1,
        other => return Err(Error::DtypeUnsupported { op, dtype: other }),
    };
    Ok(u64::from(handle.width) * elem)
}

fn move_rows(
    ctx: &Ctx,
    op: &'static str,
    entry: &str,
    wide: Tensor,
    tight: Tensor,
    index: Tensor,
    args: [Tensor; 3],
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
    let bytes = row_bytes(op, tight)?;
    let (unit, width) = unit(bytes, wide.ptr, tight.ptr);
    let per_row = u32::try_from(bytes / width).unwrap_or(u32::MAX);
    ctx.fire(
        op,
        Fire::at(kernel(FILE, &format!("::pie::gather::{entry}<{unit}>")))
            .apply(route_rows(rows, per_row)),
        &[
            args[0].arg(),
            args[1].arg(),
            args[2].arg(),
            stated(op, per_row)?.arg(),
        ],
    )
}

pub fn gather_rows(
    ctx: &Ctx,
    wide: Tensor,
    index: Tensor,
    tight: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.rows";
    move_rows(
        ctx,
        OP,
        "gather_rows",
        wide,
        *tight,
        index,
        [wide, *tight, index],
    )
}

pub fn scatter_rows(
    ctx: &Ctx,
    tight: Tensor,
    index: Tensor,
    wide: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.scatter_rows";
    move_rows(
        ctx,
        OP,
        "scatter_rows",
        *wide,
        tight,
        index,
        [tight, *wide, index],
    )
}

fn permute_rows(
    ctx: &Ctx,
    op: &'static str,
    entry: &str,
    x: Tensor,
    perm: Tensor,
    o: &mut Tensor,
) -> Result<(), Error> {
    if perm.dtype != Dtype::I32 {
        return Err(refuse(
            op,
            format!(
                "the permutation is {:?}, and a row map is i32 — one row named per moved row",
                perm.dtype
            ),
        ));
    }
    let rows = nonzero(op, "rows to move", o.rows)?;
    if perm.elements() < u64::from(rows) {
        return Err(refuse(
            op,
            format!(
                "the permutation is {} x {} and this launch moves {rows} rows",
                perm.rows, perm.width
            ),
        ));
    }
    if x.dtype != o.dtype || x.width != o.width {
        return Err(refuse(
            op,
            format!(
                "the source rectangle is {} x {:?} and the destination {} x {:?}; a row \
                 permutation does not reshape",
                x.width, x.dtype, o.width, o.dtype
            ),
        ));
    }
    let bytes = row_bytes(op, *o)?;
    let (unit, width) = unit(bytes, x.ptr, o.ptr);
    let per_row = u32::try_from(bytes / width).unwrap_or(u32::MAX);
    ctx.fire(
        op,
        Fire::at(kernel(FILE, &format!("::pie::gather::{entry}<{unit}>")))
            .apply(route_rows(rows, per_row)),
        &[
            x.arg(),
            perm.arg(),
            o.arg(),
            stated(op, per_row)?.arg(),
            ctx.stage(),
        ],
    )
}

pub fn pack_rows(ctx: &Ctx, x: Tensor, perm: Tensor, o: &mut Tensor) -> Result<(), Error> {
    permute_rows(ctx, "gather.pack_rows", "pack_rows", x, perm, o)
}

pub fn unpack_rows(ctx: &Ctx, x: Tensor, perm: Tensor, o: &mut Tensor) -> Result<(), Error> {
    permute_rows(ctx, "gather.unpack_rows", "unpack_rows", x, perm, o)
}

// `layout.concat_cols`: inputs side by side, a row each's columns after
// the one before's.

/// `xs` side by side into `y`: each a row's columns after the one before's.
pub fn concat_cols(ctx: &Ctx, xs: &[Tensor], y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "gather.concat_cols";
    let elem = y.dtype.bits();
    if !elem.is_multiple_of(8) {
        return Err(refuse(OP, "an element of whole bytes"));
    }
    let elem = elem / 8;
    let stride = u64::from(y.width) * elem;
    let int = |v: u64| i32::try_from(v).map_err(|_| refuse(OP, "a row past 32-bit offsets"));
    let mut at = 0u64;
    for x in xs {
        if x.dtype != y.dtype || x.rows != y.rows {
            return Err(refuse(OP, "the inputs share the output's element and rows"));
        }
        let width = u64::from(x.width) * elem;
        let wide = width.is_multiple_of(16)
            && stride.is_multiple_of(16)
            && at.is_multiple_of(16)
            && x.ptr % 16 == 0
            && y.ptr.is_multiple_of(16);
        let per = if wide { width / 16 } else { width };
        let threads = u32::try_from(u64::from(y.rows) * per)
            .map_err(|_| refuse(OP, "the rows pass a 32-bit launch"))?;
        if threads > 0 {
            ctx.fire(
                OP,
                Fire::at(kernel(FILE, "::pie::gather::concat_cols"))
                    .apply(Launch::flat(threads, BLOCK)),
                &[
                    x.arg(),
                    y.arg(),
                    int(u64::from(y.rows))?.arg(),
                    int(width)?.arg(),
                    int(stride)?.arg(),
                    int(at)?.arg(),
                    i32::from(wide).arg(),
                ],
            )?;
        }
        at += width;
    }
    if at != stride {
        return Err(refuse(
            OP,
            format!("the inputs fill {at} bytes of a {stride}-byte row"),
        ));
    }
    Ok(())
}

const MAX_BLOCK: u32 = 1024;

pub fn scatter_live_rows(
    ctx: &Ctx,
    src: Tensor,
    routes: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.scatter_live_rows";
    let unit = dtype_dispatch!(OP, src.dtype, {
        Bf16 => "::pie::bf16",
        F16 => "::pie::f16",
        F32 => "float"
    });
    if routes.dtype != Dtype::I32 {
        return Err(refuse(
            OP,
            format!(
                "the destination rows this merge is handed are {:?}, and it reads an i32 row map",
                routes.dtype
            ),
        ));
    }
    if routes.rows != src.rows {
        return Err(refuse(
            OP,
            format!(
                "{} rows to place and {} destinations named",
                src.rows, routes.rows
            ),
        ));
    }
    if y.dtype != src.dtype || y.width != src.width {
        return Err(refuse(
            OP,
            format!(
                "the token rectangle is {} x {:?} and the tower's is {} x {:?}; a row copy \
                 does not reshape",
                y.width, y.dtype, src.width, src.dtype
            ),
        ));
    }
    let rows = nonzero(OP, "rows to place", src.rows)?;
    let units = stated(OP, nonzero(OP, "the placed row's width", src.width)?)?;
    let threads = src
        .width
        .div_ceil(WARP)
        .max(1)
        .saturating_mul(WARP)
        .min(MAX_BLOCK);
    ctx.fire(
        OP,
        Fire::at(kernel(
            FILE,
            &format!("::pie::gather::scatter_live_rows<{unit}>"),
        ))
        .apply(Launch::per_row(rows, threads)),
        &[src.arg(), y.arg(), routes.arg(), units.arg(), ctx.stage()],
    )
}

pub fn pool_rows(ctx: &Ctx, x: Tensor, side: u32, y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "gather.pool_rows";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert_eq!(y.dtype, x.dtype, "`{OP}` pools into the element it reads");

    let (block, out) = fold_extent(OP, x, side)?;
    let width = stated(OP, x.width)?;
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
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, &format!("::pie::gather::pool_rows<{t}>")))
            .apply(Launch::per_row(out, BLOCK)),
        &[x.arg(), y.arg(), width.arg(), stated(OP, block)?.arg()],
    )
}

pub fn merge_rows(ctx: &Ctx, x: Tensor, side: u32, y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "gather.merge_rows";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
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
                "{block} rows of {} concatenate into {merged}, and the destination's rows are {} \
                 wide",
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
        OP,
        Fire::at(kernel(FILE, &format!("::pie::gather::merge_rows<{t}>")))
            .apply(Launch::per_row(out, BLOCK)),
        &[
            x.arg(),
            y.arg(),
            stated(OP, x.width)?.arg(),
            stated(OP, block)?.arg(),
        ],
    )
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
                "{} rows do not fill one {side}x{side} fold, and a fold with no whole block \
                 would leave the destination unwritten",
                x.rows
            ),
        ));
    }
    Ok((block, x.rows / block))
}

/// `tight[row] = wide[routes[row]]` for each live row whose route is not
/// negative, the live rows read through the stage word as every row op
/// reads them: [`scatter_live_rows`] the other way.
pub fn gather_live_rows(
    ctx: &Ctx,
    wide: Tensor,
    routes: Tensor,
    tight: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.gather_live_rows";
    if routes.dtype != Dtype::I32 {
        return Err(refuse(
            OP,
            format!(
                "the source rows this gather is handed are {:?}, and it reads an i32 row map",
                routes.dtype
            ),
        ));
    }
    if routes.rows != tight.rows {
        return Err(refuse(
            OP,
            format!(
                "{} rows to fill and {} sources named",
                tight.rows, routes.rows
            ),
        ));
    }
    if tight.dtype != wide.dtype || tight.width != wide.width {
        return Err(refuse(
            OP,
            format!(
                "the source rectangle is {} x {:?} and the destination {} x {:?}; a row copy \
                 does not reshape",
                wide.width, wide.dtype, tight.width, tight.dtype
            ),
        ));
    }
    let rows = nonzero(OP, "rows to fill", tight.rows)?;
    let bytes = row_bytes(OP, *tight)?;
    let (unit, width) = unit(bytes, wide.ptr, tight.ptr);
    let per_row = u32::try_from(bytes / width).unwrap_or(u32::MAX);
    ctx.fire(
        OP,
        Fire::at(kernel(
            FILE,
            &format!("::pie::gather::gather_live_rows<{unit}>"),
        ))
        .apply(route_rows(rows, per_row)),
        &[
            wide.arg(),
            tight.arg(),
            routes.arg(),
            stated(OP, per_row)?.arg(),
            ctx.stage(),
        ],
    )
}

/// One plane of a paged buffer: token `g` of it (its page slot times
/// `page_tokens`, plus its token within the page) is the `width`-element row
/// `g / page_tokens * per_slot + at + g % page_tokens * width` elements past
/// `base`, every element `dtype`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PagedPlane {
    pub base: u64,
    pub per_slot: u64,
    pub at: u64,
    pub width: u32,
    pub page_tokens: u32,
    pub dtype: Dtype,
}

impl PagedPlane {
    fn checked(&self, op: &'static str, rows: Tensor) -> Result<(), Error> {
        if rows.dtype != self.dtype {
            return Err(refuse(
                op,
                format!(
                    "the paged plane holds {:?} and the rows are {:?}",
                    self.dtype, rows.dtype
                ),
            ));
        }
        if rows.width != self.width {
            return Err(refuse(
                op,
                format!(
                    "the paged plane holds {}-element rows and the rows are {} wide",
                    self.width, rows.width
                ),
            ));
        }
        if self.page_tokens == 0 {
            return Err(refuse(op, "the paged plane's pages hold no tokens"));
        }
        Ok(())
    }

    /// The plane in the widest copy unit every row of it and of `rows` lands
    /// on: `(unit, units a row, per_slot, at)`, the last two in units.
    fn in_units(
        &self,
        op: &'static str,
        rows: Tensor,
    ) -> Result<(&'static str, u32, i64, i64), Error> {
        let elem = row_bytes(op, Tensor::new(0, 1, 1, self.dtype))?;
        let bytes = [u64::from(self.width), self.per_slot, self.at].map(|elems| elems * elem);
        let fits = |unit: u64| {
            bytes.iter().all(|b| b.is_multiple_of(unit))
                && self.base.is_multiple_of(unit)
                && rows.ptr.is_multiple_of(unit)
        };
        let (name, unit) = if fits(16) {
            ("::int4", 16)
        } else if fits(4) {
            ("::pie::i32", 4)
        } else {
            ("::pie::u8", 1)
        };
        let units = u32::try_from(bytes[0] / unit)
            .map_err(|_| refuse(op, "a paged row wider than a u32 of copy units"))?;
        let int = |v: u64| {
            i64::try_from(v / unit).map_err(|_| refuse(op, "a paged plane offset past an i64"))
        };
        Ok((name, units, int(bytes[1])?, int(bytes[2])?))
    }
}

fn token_map(op: &'static str, tokens: Tensor, rows: u32) -> Result<(), Error> {
    if tokens.dtype != Dtype::I32 {
        return Err(refuse(
            op,
            format!(
                "the paged tokens this copy is handed are {:?}, and it reads an i32 token map",
                tokens.dtype
            ),
        ));
    }
    if tokens.rows != rows {
        return Err(refuse(
            op,
            format!("{rows} rows to move and {} tokens named", tokens.rows),
        ));
    }
    Ok(())
}

/// `pages[tokens[row]] = rows[row]` for each live row whose token is not
/// negative, the live rows read through the stage word as every row op
/// reads them.
pub fn page_scatter_rows(
    ctx: &Ctx,
    rows: Tensor,
    tokens: Tensor,
    pages: PagedPlane,
) -> Result<(), Error> {
    const OP: &str = "gather.page_scatter_rows";
    pages.checked(OP, rows)?;
    token_map(OP, tokens, rows.rows)?;
    let count = nonzero(OP, "rows to page", rows.rows)?;
    let (unit, units, per_slot, at) = pages.in_units(OP, rows)?;
    ctx.fire(
        OP,
        Fire::at(kernel(
            FILE,
            &format!("::pie::gather::page_scatter_rows<{unit}>"),
        ))
        .apply(route_rows(count, units)),
        &[
            rows.arg(),
            ArgValue::Ptr(pages.base),
            tokens.arg(),
            stated(OP, units)?.arg(),
            per_slot.arg(),
            at.arg(),
            stated(OP, pages.page_tokens)?.arg(),
            ctx.stage(),
        ],
    )
}

/// `rows[e] = pages[tokens[e]]` for every row whose token is not negative;
/// the rest are left as they were.
pub fn page_gather_rows(
    ctx: &Ctx,
    pages: PagedPlane,
    tokens: Tensor,
    rows: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "gather.page_gather_rows";
    pages.checked(OP, *rows)?;
    token_map(OP, tokens, rows.rows)?;
    let count = nonzero(OP, "rows to unpage", rows.rows)?;
    let (unit, units, per_slot, at) = pages.in_units(OP, *rows)?;
    ctx.fire(
        OP,
        Fire::at(kernel(
            FILE,
            &format!("::pie::gather::page_gather_rows<{unit}>"),
        ))
        .apply(route_rows(count, units)),
        &[
            ArgValue::Ptr(pages.base),
            rows.arg(),
            tokens.arg(),
            stated(OP, units)?.arg(),
            per_slot.arg(),
            at.arg(),
            stated(OP, pages.page_tokens)?.arg(),
        ],
    )
}
