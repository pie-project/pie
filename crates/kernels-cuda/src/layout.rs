//! `Layout`: gathers, cuts, and slices — data movement with no arithmetic.
//! One entry per IR variant. The embed gather picks its vectorised
//! instantiation from alignment alone; that choice never leaves this file.

use kernels::KernelError;
use model_ir::Dtype;

use crate::jit::{
    Arg, Ctx, Fire, Launch, aligned16, dtype_dispatch, nonzero, refuse, stated, symbol,
};
use crate::tensor::Tensor;

const FILE: &str = "layout/layout.cuh";

const BLOCK: u32 = 256;

const WARP: u32 = 32;

const VEC_WIDTH: u32 = 8;

/// One block per row, sized to the row in whole warps.
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

/// Whether the embed gather may move eight elements at a time.
fn vectorisable(hidden: u32, table: u64, y: u64) -> bool {
    hidden % VEC_WIDTH == 0 && aligned16(table) && aligned16(y)
}

pub fn embed(
    ctx: &Ctx,
    ids: Tensor,
    table: Tensor,
    vocab: u32,
    y: &mut Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "layout.embed";
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
        Fire::at(FILE, instantiation).apply(Launch::grid([blocks, 1, 1], [BLOCK, 1, 1])),
        &[
            ids.arg(),
            table.arg(),
            y.arg(),
            hidden.arg(),
            vocab.arg(),
            rows.arg(),
            stated(OP, per_row)?.arg(),
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
) -> Result<(), KernelError> {
    const OP: &str = "layout.split_qkv";
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
        Fire::at(FILE, "::pie::layout::split_qkv<::pie::bf16>").apply(Launch::grid(
            [width.div_ceil(BLOCK), q.rows, 1],
            [BLOCK, 1, 1],
        )),
        &[
            packed.arg(),
            q.arg(),
            k.arg(),
            v.arg(),
            q_dim.arg(),
            kv_dim.arg(),
        ],
    )
}

/// Deinterleaves per-head `(q, gate)` pairs from the packed projection.
pub fn split_q_gate(
    ctx: &Ctx,
    packed: Tensor,
    head_dim: u32,
    q: &mut Tensor,
    gate: &mut Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "layout.split_q_gate";
    dtype_dispatch!(OP, packed.dtype, { Bf16 => () });
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
    let heads = q.width / head_dim;
    let block = if head_dim < 128 { 64 } else { 128 };
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::layout::split_q_gate<::pie::bf16>")
            .apply(Launch::grid([q.rows, heads, 1], [block, 1, 1])),
        &[
            packed.arg(),
            q.arg(),
            gate.arg(),
            stated(OP, q.rows)?.arg(),
            stated(OP, heads)?.arg(),
            stated(OP, head_dim)?.arg(),
        ],
    )
}

/// Splits each row at column `width`.
pub fn split_rows(
    ctx: &Ctx,
    x: Tensor,
    width: u32,
    left: &mut Tensor,
    right: &mut Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "layout.split_rows";
    dtype_dispatch!(OP, x.dtype, { Bf16 => () });
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
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::layout::split_rows<::pie::bf16>")
            .apply(route_rows(left.rows, left.width)),
        &[
            x.arg(),
            left.arg(),
            right.arg(),
            left_dim.arg(),
            right_dim.arg(),
        ],
    )
}

/// Copies layer `layer`'s `width`-wide slice out of a stacked table.
pub fn select(
    ctx: &Ctx,
    table: Tensor,
    layer: u32,
    width: u32,
    y: &mut Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "layout.select";
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
        Fire::at(FILE, symbol(&format!("::pie::layout::select<{t}>")))
            .apply(route_rows(y.rows, width)),
        &[
            table.arg(),
            y.arg(),
            stated(OP, table.width)?.arg(),
            stated(OP, offset)?.arg(),
            stated(OP, width)?.arg(),
        ],
    )
}

/// The copy unit a row of `bytes` bytes at `a` and `b` may move in, and the
/// template argument that names it.
///
/// **WIDTH IS AN OPTIMIZATION; THE BYTES ARE THE CONTRACT.** [`gather_rows`]
/// and [`scatter_rows`] are a permutation, so the only thing that may vary
/// with this choice is how fast it runs — a 16-byte unit when both addresses
/// and the row's width admit one, a 4-byte unit when they admit that, and a
/// byte otherwise, which every row admits. There is no arithmetic in the
/// kernel and no dtype in its signature, which is what lets a bf16 activation
/// and an f32 log-sum-exp move through one instantiation and neither be
/// rounded on the way.
fn unit(bytes: u64, a: u64, b: u64) -> (&'static str, u64) {
    if bytes.is_multiple_of(16) && aligned16(a) && aligned16(b) {
        ("::int4", 16)
    } else if bytes.is_multiple_of(4) && a.is_multiple_of(4) && b.is_multiple_of(4) {
        ("::pie::i32", 4)
    } else {
        ("::pie::u8", 1)
    }
}

/// How wide one row of this handle is, in bytes.
fn row_bytes(op: &'static str, handle: Tensor) -> Result<u64, KernelError> {
    let elem = match handle.dtype {
        Dtype::Bf16 | Dtype::F16 => 2,
        Dtype::F32 | Dtype::I32 | Dtype::U32 => 4,
        Dtype::U8 | Dtype::I8 | Dtype::Fp8E4m3 | Dtype::E8m0 => 1,
        other => return Err(KernelError::DtypeUnsupported { op, dtype: other }),
    };
    Ok(u64::from(handle.width) * elem)
}

/// The two halves of one `Fallback::Copy`, which differ only in which way the
/// index is read — so they are one body, and the pair cannot drift apart into
/// a gather and a scatter that disagree about what the map means.
fn move_rows(
    ctx: &Ctx,
    op: &'static str,
    entry: &str,
    wide: Tensor,
    tight: Tensor,
    index: Tensor,
    args: [Tensor; 3],
) -> Result<(), KernelError> {
    debug_assert_eq!(index.dtype, Dtype::I32, "`{op}` reads i32 fire rows");
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
        Fire::at(FILE, symbol(&format!("::pie::layout::{entry}<{unit}>")))
            .apply(route_rows(rows, per_row)),
        &[
            args[0].arg(),
            args[1].arg(),
            args[2].arg(),
            stated(op, per_row)?.arg(),
        ],
    )
}

/// **GATHER: the rows a fragmented window covers, laid down as one.**
///
/// `Fallback::Copy`'s first half (design §3; `model_compiler::layout`'s
/// `menu` is what asks for it below the crossover). A windowed consumer P4
/// could not seat stands over several intervals of the fire's rows; this
/// reads them out of `wide` in the order `index` names and writes them
/// contiguously into `tight`, so the consumer behind it is ONE launch over a
/// rectangle rather than one launch per interval.
///
/// `index` is `i32`, one entry per row of `tight`: the FIRE row that row
/// stands at. It is the caller's span list flattened, and the caller is the
/// only one who knows it — this entry checks the shapes agree and moves
/// bytes.
///
/// # Errors
///
/// [`KernelError::DtypeUnsupported`] for a packed element with no byte size,
/// and a refusal for an index vector or a rectangle that does not match the
/// one beside it.
pub fn gather_rows(
    ctx: &Ctx,
    wide: Tensor,
    index: Tensor,
    tight: &mut Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "layout.gather_rows";
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

/// **SCATTER: the answers put back where their rows came from.**
///
/// `Fallback::Copy`'s second half, and the same map as [`gather_rows`] read
/// the other way: row `i` of `tight` lands at fire row `index[i]` of `wide`.
/// The rows the window does NOT cover are not written, which is what keeps a
/// copy one consumer's slow path rather than a fact about the arena.
///
/// # Errors
///
/// As [`gather_rows`].
pub fn scatter_rows(
    ctx: &Ctx,
    tight: Tensor,
    index: Tensor,
    wide: &mut Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "layout.scatter_rows";
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
