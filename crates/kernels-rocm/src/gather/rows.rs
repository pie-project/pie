#![allow(unsafe_code)]
use crate::error::Error;
use dtype::Dtype;

use crate::jit::{
    Arg, ArgValue, Ctx, Fire, Launch, aligned16, dtype_dispatch, kernel, nonzero, refuse, stated,
};
use crate::tensor::Tensor;

const FILE: &str = "gather/rows.hip";

const BLOCK: u32 = 256;

/// Lanes per wavefront on CDNA: a block is a whole number of them.
const WAVE: u32 = 64;

const VEC_WIDTH: u32 = 8;

fn route_rows(rows: u32, width: u32) -> Launch {
    const MAX_BLOCK: u32 = 1024;

    Launch::per_row(
        rows,
        width
            .div_ceil(WAVE)
            .max(1)
            .saturating_mul(WAVE)
            .min(MAX_BLOCK),
    )
}

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
        "::pie::layout::embed<::pie::true_type::value>"
    } else {
        "::pie::layout::embed<::pie::false_type::value>"
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
        Fire::at(kernel(FILE, "::pie::layout::embed_vocab_shard"))
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
    #[cfg(feature = "rocm")]
    {
        let nccl = crate::collective::rccl(op)?;
        let mut rank: i32 = 0;
        // SAFETY: `comm` is the live communicator this context fires its
        // collectives on; the out-parameter is a stack i32.
        let code = unsafe { (nccl.ncclCommUserRank)(comm, &raw mut rank) };
        crate::collective::answered(op, "ncclCommUserRank", code)?;
        Ok(u32::try_from(rank).unwrap_or(0))
    }
    #[cfg(not(feature = "rocm"))]
    {
        let _ = comm;
        Err(crate::jit::runtimeless(op))
    }
}

#[cfg_attr(not(feature = "rocm"), allow(dead_code))]
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
        Fire::at(kernel(FILE, "::pie::layout::gather_width_concat"))
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
        Fire::at(kernel(FILE, "::pie::layout::embed_scale_add"))
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
        Fire::at(kernel(FILE, "::pie::layout::embed_scale_add_select"))
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
    let width = q.width.max(k.width);
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::layout::split_qkv")).apply(Launch::grid(
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
        Fire::at(kernel(FILE, "::pie::layout::split_qkv_padded")).apply(Launch::grid(
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
    let block = if head_dim < 128 { 64 } else { 128 };
    ctx.fire(
        OP,
        Fire::at(kernel(FILE, "::pie::layout::split_q_gate"))
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
    // One thread per 8 elements; the kernel falls back to single elements
    // when the row's halves do not allow 16-byte moves.
    let entrypoint = format!("::pie::layout::split_rows<{t}>");
    let launch = Launch::grid(
        [left.rows, x.width.div_ceil(VEC_WIDTH * BLOCK), 1],
        [BLOCK, 1, 1],
    );
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
        Fire::at(kernel(FILE, &format!("::pie::layout::select<{t}>")))
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
        ("::pie::b128", 16)
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
        Fire::at(kernel(FILE, &format!("::pie::layout::{entry}<{unit}>")))
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
        Fire::at(kernel(FILE, &format!("::pie::layout::{entry}<{unit}>")))
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
