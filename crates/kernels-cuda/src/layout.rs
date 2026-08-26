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
