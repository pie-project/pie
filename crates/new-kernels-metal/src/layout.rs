//! `Layout`: gathers, cuts, and slices — data movement with no arithmetic.
//! One entry per IR variant, plus the quantized embed-gather the driver
//! selects when the table is an affine bank.

use new_kernels::KernelError;
use new_model_ir::Dtype;

use crate::encode::{
    Arg, Ctx, Fire, Grid, dtype_dispatch, elementwise_rows, head_grid, head_group, nonzero,
    refuse, stated,
};
use crate::tensor::Tensor;

pub fn embed(
    ctx: &Ctx<'_>,
    ids: Tensor,
    table: Tensor,
    vocab: u32,
    y: Tensor,
) -> Result<(), KernelError> {
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

pub fn split_qkv(
    ctx: &Ctx<'_>,
    packed: Tensor,
    q_width: u32,
    kv_width: u32,
    q: Tensor,
    k: Tensor,
    v: Tensor,
) -> Result<(), KernelError> {
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
) -> Result<(), KernelError> {
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
        Fire::at("attn/gate.metal", entry).apply(Grid::of(lanes, head_group(lanes))),
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
) -> Result<(), KernelError> {
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
) -> Result<(), KernelError> {
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

fn affine_point(op: &'static str, group: i32, bits: i32) -> Result<usize, KernelError> {
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
/// becomes when the driver resolves the table to a `(codes, scales, biases)`
/// bank instead of a dense weight.
#[allow(clippy::too_many_arguments)]
pub fn embed_gather_mb_4bit(
    ctx: &Ctx<'_>,
    w: Tensor,
    scales: Tensor,
    biases: Tensor,
    out: Tensor,
    group: i32,
    bits: i32,
    token_ids: Tensor,
) -> Result<(), KernelError> {
    const OP: &str = "layout.embed";
    const ENTRIES: [&str; 6] = [
        "embed_gather_mb_4bit_bfloat16_gs_32_b_4",
        "embed_gather_mb_4bit_bfloat16_gs_32_b_8",
        "embed_gather_mb_4bit_bfloat16_gs_64_b_4",
        "embed_gather_mb_4bit_bfloat16_gs_64_b_8",
        "embed_gather_mb_4bit_bfloat16_gs_128_b_4",
        "embed_gather_mb_4bit_bfloat16_gs_128_b_8",
    ];
    debug_assert_eq!(token_ids.dtype, Dtype::I32, "`{OP}` gathers by i32 token ids");
    ctx.fire(
        Fire::at("layout/embed_gather.metal", ENTRIES[affine_point(OP, group, bits)?]).apply(
            Grid::of(elementwise_rows(OP, out.width, out.rows)?, [256, 1, 1]),
        ),
        &[
            w.arg(),
            scales.arg(),
            biases.arg(),
            token_ids.arg(),
            out.arg_mut(),
            stated(OP, out.width)?.arg(),
        ],
    )
}
