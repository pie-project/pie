use crate::error::Error;

use crate::encode::{Arg, Ctx, Fire, Grid, dtype_dispatch, elementwise, elementwise_rows, refuse};
use crate::tensor::Tensor;

const GROUP: u32 = 256;

/// The widest row a table-laying activation stages in threadgroup memory.
const TABLE_WIDEST: u32 = 8192;

struct Halves {
    packed: Tensor,
    y: Tensor,
    grid: Grid,
    intermediate: u32,
}

fn halves(op: &'static str, packed: Tensor, intermediate: u32, y: Tensor) -> Result<Halves, Error> {
    debug_assert_eq!(
        packed.width,
        intermediate.saturating_mul(2),
        "the packed `[gate | up]` row is twice the intermediate width it states"
    );
    debug_assert_eq!(
        y.width, intermediate,
        "the activation's row is the intermediate width it states"
    );
    debug_assert_eq!(
        y.rows, packed.rows,
        "the activation lands one row per packed row"
    );
    let lanes = elementwise_rows(op, y.width, y.rows)?;
    Ok(Halves {
        packed,
        y,
        grid: Grid::of(lanes, [lanes[0].min(GROUP), 1, 1]),
        intermediate,
    })
}

pub fn swiglu(ctx: &Ctx<'_>, packed: Tensor, intermediate: u32, y: Tensor) -> Result<(), Error> {
    const OP: &str = "linear.swiglu";
    let entry = dtype_dispatch!(OP, packed.dtype, { Bf16 => "packed_swiglu_bfloat16" });
    let cut = halves(OP, packed, intermediate, y)?;
    ctx.fire(
        Fire::at("linear/mlp_packed.metal", entry).apply(cut.grid),
        &[cut.packed.arg(), cut.y.arg_mut(), cut.intermediate.arg()],
    )
}

/// [`swiglu`] over at most eight rows that also lays the register tile's
/// input table of `y` in `table`, with `counters` arrival counters zeroed:
/// the table `linear::quant`'s register matmul lays over `y`, bit for bit,
/// so the down projection reads it and runs no table pass.
pub fn swiglu_table(
    ctx: &Ctx<'_>,
    packed: Tensor,
    intermediate: u32,
    y: Tensor,
    table: Tensor,
    counters: u32,
) -> Result<(), Error> {
    const OP: &str = "linear.swiglu";
    dtype_dispatch!(OP, packed.dtype, { Bf16 => () });
    let cut = halves(OP, packed, intermediate, y)?;
    if y.rows > 8 || !intermediate.is_multiple_of(64) || intermediate > TABLE_WIDEST {
        return Err(refuse(
            OP,
            format!(
                "a register table holds at most 8 rows of whole 64-input groups, and this one \
                 stages at most 8192, not {} x {intermediate}",
                y.rows
            ),
        ));
    }
    ctx.fire(
        Fire::at("linear/mlp_packed.metal", "packed_swiglu_table").apply(Grid::of(
            [intermediate.min(1024), 8, 1],
            [intermediate.min(1024), 1, 1],
        )),
        &[
            cut.packed.arg(),
            cut.y.arg_mut(),
            cut.intermediate.arg(),
            table.arg_mut(),
            y.rows.arg(),
            counters.arg(),
        ],
    )
}

/// [`swiglu`] over a routed stack's sorted rows: padding rows, whose `perm`
/// is negative, are left as they are.
pub fn swiglu_sorted(
    ctx: &Ctx<'_>,
    packed: Tensor,
    intermediate: u32,
    perm: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.swiglu";
    dtype_dispatch!(OP, packed.dtype, { Bf16 => () });
    let cut = halves(OP, packed, intermediate, y)?;
    ctx.fire(
        Fire::at("linear/mlp_packed.metal", "packed_swiglu_sorted").apply(cut.grid),
        &[
            cut.packed.arg(),
            cut.y.arg_mut(),
            cut.intermediate.arg(),
            perm.arg(),
        ],
    )
}

/// [`swiglu_clamp_alpha`] over a routed stack's sorted rows, padding rows left as they are.
pub fn swiglu_clamp_alpha_sorted(
    ctx: &Ctx<'_>,
    packed: Tensor,
    intermediate: u32,
    limit: f32,
    alpha: f32,
    perm: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.swiglu_clamp_alpha";
    dtype_dispatch!(OP, packed.dtype, { Bf16 => () });
    let cut = halves(OP, packed, intermediate, y)?;
    ctx.fire(
        Fire::at("linear/mlp_packed.metal", "packed_gptoss_swiglu_sorted").apply(cut.grid),
        &[
            cut.packed.arg(),
            cut.y.arg_mut(),
            cut.intermediate.arg(),
            limit.arg(),
            alpha.arg(),
            perm.arg(),
        ],
    )
}

/// [`geglu_tanh_packed`] over a routed stack's sorted rows, padding rows left as they are.
pub fn geglu_tanh_sorted(
    ctx: &Ctx<'_>,
    packed: Tensor,
    intermediate: u32,
    perm: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.geglu_tanh_packed";
    dtype_dispatch!(OP, packed.dtype, { Bf16 => () });
    let cut = halves(OP, packed, intermediate, y)?;
    ctx.fire(
        Fire::at("linear/mlp_packed.metal", "packed_geglu_tanh_sorted").apply(cut.grid),
        &[
            cut.packed.arg(),
            cut.y.arg_mut(),
            cut.intermediate.arg(),
            perm.arg(),
        ],
    )
}

/// [`geglu_tanh_sorted`] that also lands each row's 64-wide group sums in `sums` over
/// `tile`-row tiles, as the MPP routed matmul that reads the rows next folds them.
#[allow(clippy::too_many_arguments)]
pub fn geglu_tanh_sorted_sums(
    ctx: &Ctx<'_>,
    packed: Tensor,
    intermediate: u32,
    perm: Tensor,
    y: Tensor,
    sums: Tensor,
    tile: u32,
) -> Result<(), Error> {
    const OP: &str = "linear.geglu_tanh_packed";
    dtype_dispatch!(OP, packed.dtype, { Bf16 => () });
    let cut = halves(OP, packed, intermediate, y)?;
    debug_assert!(intermediate.is_multiple_of(64), "whole 64-wide groups");
    ctx.fire(
        Fire::at("linear/mlp_packed.metal", "packed_geglu_tanh_sorted_sums")
            .apply(Grid::of([256, y.rows, 1], [256, 1, 1])),
        &[
            cut.packed.arg(),
            cut.y.arg_mut(),
            cut.intermediate.arg(),
            perm.arg(),
            sums.arg_mut(),
            tile.arg(),
        ],
    )
}

pub fn swiglu_clamp(
    ctx: &Ctx<'_>,
    packed: Tensor,
    intermediate: u32,
    limit: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.swiglu_clamp";
    let entry = dtype_dispatch!(OP, packed.dtype, { Bf16 => "packed_swiglu_clamp_bfloat16" });
    let cut = halves(OP, packed, intermediate, y)?;
    ctx.fire(
        Fire::at("linear/mlp_packed.metal", entry).apply(cut.grid),
        &[
            cut.packed.arg(),
            cut.y.arg_mut(),
            cut.intermediate.arg(),
            limit.arg(),
        ],
    )
}

pub fn swiglu_clamp_alpha(
    ctx: &Ctx<'_>,
    packed: Tensor,
    intermediate: u32,
    limit: f32,
    alpha: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.swiglu_clamp_alpha";
    let entry = dtype_dispatch!(OP, packed.dtype, { Bf16 => "packed_gptoss_swiglu_bfloat16" });
    let cut = halves(OP, packed, intermediate, y)?;
    ctx.fire(
        Fire::at("linear/mlp_packed.metal", entry).apply(cut.grid),
        &[
            cut.packed.arg(),
            cut.y.arg_mut(),
            cut.intermediate.arg(),
            limit.arg(),
            alpha.arg(),
        ],
    )
}

pub fn swiglu_clamp_split(
    ctx: &Ctx<'_>,
    gate: Tensor,
    up: Tensor,
    limit: f32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.swiglu_clamp_split";
    let entry = dtype_dispatch!(OP, gate.dtype, { Bf16 => "swiglu_clamp_bfloat16" });
    debug_assert_eq!(
        (up.rows, up.width),
        (gate.rows, gate.width),
        "the unfused halves are one rectangle each and they are the same rectangle"
    );
    debug_assert_eq!(
        (y.rows, y.width),
        (gate.rows, gate.width),
        "the activation lands on the halves' own rectangle"
    );
    ctx.fire(
        Fire::at("linear/mlp_gated.metal", entry).apply(Grid::of(
            elementwise(OP, gate.width, gate.rows)?,
            [GROUP, 1, 1],
        )),
        &[gate.arg(), up.arg(), y.arg_mut(), limit.arg()],
    )
}

pub fn geglu_tanh(ctx: &Ctx<'_>, gate: Tensor, up: Tensor, y: Tensor) -> Result<(), Error> {
    const OP: &str = "linear.geglu_tanh";
    let entry = dtype_dispatch!(OP, gate.dtype, { Bf16 => "geglu_tanh_bfloat16" });
    ctx.fire(
        Fire::at("linear/mlp_gated.metal", entry).apply(Grid::of(
            elementwise(OP, gate.width, gate.rows)?,
            [GROUP, 1, 1],
        )),
        &[gate.arg(), up.arg(), y.arg_mut()],
    )
}

pub fn gelu_tanh(ctx: &Ctx<'_>, x: Tensor, y: Tensor) -> Result<(), Error> {
    const OP: &str = "linear.gelu_tanh";
    let entry = dtype_dispatch!(OP, x.dtype, { Bf16 => "mlp_gelu_tanh_bfloat16" });
    ctx.fire(
        Fire::at("linear/mlp_gated.metal", entry)
            .apply(Grid::of(elementwise(OP, x.width, x.rows)?, [GROUP, 1, 1])),
        &[x.arg(), y.arg_mut()],
    )
}

pub fn geglu_tanh_packed(
    ctx: &Ctx<'_>,
    packed: Tensor,
    intermediate: u32,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.geglu_tanh_packed";
    let entry = dtype_dispatch!(OP, packed.dtype, { Bf16 => "packed_geglu_tanh_bfloat16" });
    let cut = halves(OP, packed, intermediate, y)?;
    ctx.fire(
        Fire::at("linear/mlp_packed.metal", entry).apply(cut.grid),
        &[cut.packed.arg(), cut.y.arg_mut(), cut.intermediate.arg()],
    )
}

pub fn situ(
    ctx: &Ctx<'_>,
    packed: Tensor,
    intermediate: u32,
    beta: f32,
    up_cap: Option<f32>,
    y: Tensor,
) -> Result<(), Error> {
    const OP: &str = "linear.situ";
    if beta == 0.0 {
        return Err(refuse(OP, "beta is zero, and the gate divides by it"));
    }
    let entry = dtype_dispatch!(OP, packed.dtype, { Bf16 => "packed_situ_bfloat16" });
    let cut = halves(OP, packed, intermediate, y)?;
    ctx.fire(
        Fire::at("linear/mlp_packed.metal", entry).apply(cut.grid),
        &[
            cut.packed.arg(),
            cut.y.arg_mut(),
            cut.intermediate.arg(),
            beta.arg(),
            up_cap.unwrap_or(0.0).arg(),
        ],
    )
}
