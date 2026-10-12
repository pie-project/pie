use crate::error::Error;
use crate::jit::{Arg, Ctx, Fire, Launch, count, dtype_dispatch, refuse, stated};
use crate::spatial::{flat_elements, lanes_of};
use crate::tensor::Tensor;
use dtype::Dtype;

const FILE: &str = "spatial/norm.cuh";

const OP: &str = "spatial.group_norm";

const STATS_BLOCK: u32 = 1024;

const APPLY_BLOCK: u32 = 256;

const MAX_SPLITS: u32 = 512;

const ROWS_PER_SPLIT: u32 = 256;

#[allow(clippy::too_many_arguments)]
pub fn group_norm(
    ctx: &Ctx,
    x: Tensor,
    grid: Tensor,
    groups: u32,
    weight: Tensor,
    bias: Tensor,
    eps: f32,
    silu: bool,
    o: &mut Tensor,
) -> Result<(), Error> {
    dtype_dispatch!(OP, x.dtype, { Bf16 => () });
    debug_assert!(
        o.rows == x.rows && o.width == x.width && o.dtype == x.dtype,
        "`{OP}` lands one row per input row"
    );
    let lanes = lanes_of(OP, "input", grid)?;
    let c = count(OP, "the channel count", x.width)?;
    let groups = count(OP, "the group count", groups)?;
    if x.width > STATS_BLOCK || !x.width.is_multiple_of(groups.unsigned_abs()) {
        return Err(refuse(
            OP,
            format!(
                "{} channels in {groups} groups: the width must divide by the groups and fit {STATS_BLOCK} threads",
                x.width
            ),
        ));
    }
    for (what, t) in [("weight", weight), ("bias", bias)] {
        if t.dtype != Dtype::F32 || t.elements() != u64::from(x.width) {
            return Err(refuse(
                OP,
                format!(
                    "the {what} is {}x{} {:?}; expected {} f32",
                    t.rows, t.width, t.dtype, x.width
                ),
            ));
        }
    }
    let rows = count(OP, "rows", x.rows)?;
    let splits = rows
        .unsigned_abs()
        .div_ceil(ROWS_PER_SPLIT)
        .clamp(1, MAX_SPLITS);
    let cells = usize::try_from(lanes)
        .ok()
        .and_then(|l| l.checked_mul(splits as usize))
        .and_then(|n| n.checked_mul(groups.unsigned_abs() as usize))
        .ok_or_else(|| refuse(OP, "the moments table overflows"))?;
    let partials = ctx.scratch(OP, "spatial.group_norm.partials", cells * 16)? as u64;
    let stats = ctx.scratch(OP, "spatial.group_norm.stats", cells * 8)? as u64;
    let splits = stated(OP, splits)?;

    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::spatial::group_norm_stats<1024>").apply(Launch::grid(
            [splits.unsigned_abs(), lanes.unsigned_abs(), 1],
            [STATS_BLOCK, 1, 1],
        )),
        &[
            x.arg(),
            grid.arg(),
            partials.arg(),
            c.arg(),
            groups.arg(),
            splits.arg(),
        ],
    )?;
    ctx.fire(
        OP,
        Fire::at(FILE, "::pie::spatial::group_norm_finalize").apply(Launch::grid(
            [groups.unsigned_abs(), lanes.unsigned_abs(), 1],
            [32, 1, 1],
        )),
        &[
            partials.arg(),
            stats.arg(),
            groups.arg(),
            splits.arg(),
            eps.arg(),
        ],
    )?;
    let (blocks, total) = flat_elements(OP, x, APPLY_BLOCK)?;
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            if silu {
                "::pie::spatial::group_norm_apply<true>"
            } else {
                "::pie::spatial::group_norm_apply<false>"
            },
        )
        .apply(Launch::grid([blocks, 1, 1], [APPLY_BLOCK, 1, 1])),
        &[
            x.arg(),
            grid.arg(),
            stats.arg(),
            weight.arg(),
            bias.arg(),
            o.arg(),
            c.arg(),
            groups.arg(),
            lanes.arg(),
            total.arg(),
        ],
    )
}
