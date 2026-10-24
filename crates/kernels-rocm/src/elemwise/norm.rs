use super::*;
use crate::error::Error;
use crate::jit::{
    Arg, ArgValue, Ctx, Fire, Launch, count, dtype_dispatch, kernel, nonzero, refuse, stated,
};
use crate::tensor::Tensor;
use crate::voxels::{flat_elements, lanes_of};
use dtype::Dtype;

const ABSOLUTE_BANK: &str = "rmsnorm";

const OFFSET_BANK: &str = "rmsnorm_plus_one";

fn rows_per_head(op: &'static str, rows: u32, width: u32, head_dim: u32) -> Result<Launch, Error> {
    nonzero(op, "rows", rows)?;
    if head_dim == 0 {
        return Ok(Launch::per_row(rows, BLOCK));
    }
    if width == 0 || !width.is_multiple_of(head_dim) {
        return Err(refuse(
            op,
            format!("the {width}-wide row is not a whole number of {head_dim}-wide heads"),
        ));
    }
    let blocks = rows.checked_mul(width / head_dim).ok_or_else(|| {
        refuse(
            op,
            format!(
                "the grid will not launch: {rows} rows x {} heads",
                width / head_dim
            ),
        )
    })?;
    Ok(Launch::per_row(blocks, BLOCK))
}

const fn per_head_split(width: u32, head_dim: u32) -> u32 {
    #[allow(clippy::manual_checked_ops)]
    if head_dim == 0 { 1 } else { width / head_dim }
}

/// A per-head norm on its own lanes: a head of 8 to 512 elements (a power
/// of two) in planes that allow 16-byte reads, eight elements a lane. The
/// lanes a head, the launch and the head rows in all.
fn head_lanes(
    op: &'static str,
    y: &Tensor,
    head_dim: u32,
    planes: &[u64],
) -> Result<Option<(u32, Launch, i64)>, Error> {
    let lph = head_dim / VEC_WIDTH;
    if head_dim == 0
        || !head_dim.is_multiple_of(VEC_WIDTH)
        || !lph.is_power_of_two()
        || lph > WARP
        || !planes.iter().all(|p| p.is_multiple_of(16))
    {
        return Ok(None);
    }
    let total = u64::from(y.rows) * u64::from(y.width / head_dim);
    let blocks = u32::try_from(total.div_ceil(u64::from(BLOCK / lph)))
        .map_err(|_| refuse(op, format!("the grid will not launch: {total} head rows")))?;
    Ok(Some((
        lph,
        Launch::per_row(blocks, BLOCK),
        i64::try_from(total).unwrap_or(i64::MAX),
    )))
}

#[allow(clippy::too_many_arguments)]
fn rms_row(
    ctx: &Ctx,
    op: &'static str,
    template: &'static str,
    x: Tensor,
    weight: Tensor,
    y: &mut Tensor,
    per_head_dim: u32,
    eps: f32,
) -> Result<(), Error> {
    let t = dtype_dispatch!(op, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let hidden = stated(
        op,
        nonzero(
            op,
            "the normed width",
            if per_head_dim == 0 {
                y.width
            } else {
                per_head_dim
            },
        )?,
    )?;
    let launch = rows_per_head(op, y.rows, y.width, per_head_dim)?;
    let heads = stated(op, per_head_split(y.width, per_head_dim))?;
    if let Some((lph, lanes, total)) = head_lanes(op, y, per_head_dim, &[x.ptr, weight.ptr, y.ptr])?
    {
        let plus = template == OFFSET_BANK;
        return ctx.fire(
            op,
            Fire::at(kernel(
                NORM_UNIT,
                &format!("::pie::elemwise::rmsnorm_lanes<{t}, {lph}, {plus}>"),
            ))
            .apply(lanes),
            &[
                x.arg(),
                weight.arg(),
                y.arg(),
                total.arg(),
                eps.arg(),
                heads.arg(),
                ctx.stage(),
            ],
        );
    }
    ctx.fire(
        op,
        Fire::at(kernel(
            NORM_UNIT,
            &format!("::pie::elemwise::{template}<{t}, 256>"),
        ))
        .apply(launch),
        &[
            x.arg(),
            weight.arg(),
            y.arg(),
            hidden.arg(),
            hidden.arg(),
            hidden.arg(),
            eps.arg(),
            heads.arg(),
            ctx.stage(),
        ],
    )
}

pub fn rmsnorm(
    ctx: &Ctx,
    x: Tensor,
    weight: Tensor,
    eps: f32,
    y: &mut Tensor,
) -> Result<(), Error> {
    rms_row(ctx, "elemwise.rmsnorm", ABSOLUTE_BANK, x, weight, y, 0, eps)
}

pub fn rmsnorm_per_head(
    ctx: &Ctx,
    x: Tensor,
    weight: Tensor,
    head_dim: u32,
    eps: f32,
    y: &mut Tensor,
) -> Result<(), Error> {
    rms_row(
        ctx,
        "elemwise.rmsnorm",
        ABSOLUTE_BANK,
        x,
        weight,
        y,
        head_dim,
        eps,
    )
}

pub fn rmsnorm_plus_one(
    ctx: &Ctx,
    x: Tensor,
    weight: Tensor,
    eps: f32,
    y: &mut Tensor,
) -> Result<(), Error> {
    rms_row(ctx, "elemwise.rmsnorm", OFFSET_BANK, x, weight, y, 0, eps)
}

pub fn rmsnorm_per_head_plus_one(
    ctx: &Ctx,
    x: Tensor,
    weight: Tensor,
    head_dim: u32,
    eps: f32,
    y: &mut Tensor,
) -> Result<(), Error> {
    rms_row(
        ctx,
        "elemwise.rmsnorm",
        OFFSET_BANK,
        x,
        weight,
        y,
        head_dim,
        eps,
    )
}

pub fn rmsnorm_no_scale(
    ctx: &Ctx,
    x: Tensor,
    head_dim: u32,
    eps: f32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.rmsnorm";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let hidden = stated(
        OP,
        nonzero(
            OP,
            "the normed width",
            if head_dim == 0 { y.width } else { head_dim },
        )?,
    )?;
    let launch = rows_per_head(OP, y.rows, y.width, head_dim)?;
    let heads = stated(OP, per_head_split(y.width, head_dim))?;
    if let Some((lph, lanes, total)) = head_lanes(OP, y, head_dim, &[x.ptr, y.ptr])? {
        return ctx.fire(
            OP,
            Fire::at(kernel(
                NORM_UNIT,
                &format!("::pie::elemwise::rmsnorm_lanes<{t}, {lph}, false>"),
            ))
            .apply(lanes),
            &[
                x.arg(),
                ArgValue::ABSENT,
                y.arg(),
                total.arg(),
                eps.arg(),
                heads.arg(),
                ctx.stage(),
            ],
        );
    }
    ctx.fire(
        OP,
        Fire::at(kernel(
            NORM_UNIT,
            &format!("::pie::elemwise::rmsnorm_no_scale<{t}, 256>"),
        ))
        .apply(launch),
        &[
            x.arg(),
            y.arg(),
            hidden.arg(),
            eps.arg(),
            heads.arg(),
            ctx.stage(),
        ],
    )
}

#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_gated(
    ctx: &Ctx,
    x: Tensor,
    gate: Tensor,
    weight: Option<Tensor>,
    head_dim: u32,
    eps: f32,
    sigmoid_gate: bool,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.rmsnorm_gated";
    debug_assert_eq!(x.dtype, Dtype::F32, "`{OP}` norms an f32 accumulator");
    debug_assert!(
        weight.is_none_or(|w| w.dtype == Dtype::F32),
        "`{OP}` scales by an f32 weight"
    );
    let t = dtype_dispatch!(OP, gate.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let hidden = stated(
        OP,
        nonzero(
            OP,
            "the normed width",
            if head_dim == 0 { y.width } else { head_dim },
        )?,
    )?;
    let launch = rows_per_head(OP, y.rows, y.width, head_dim)?;
    let heads = stated(OP, per_head_split(y.width, head_dim))?;
    if let Some((lph, lanes, total)) = head_lanes(
        OP,
        y,
        head_dim,
        &[x.ptr, gate.ptr, y.ptr, weight.map_or(0, |w| w.ptr)],
    )? {
        return ctx.fire(
            OP,
            Fire::at(kernel(
                NORM_UNIT,
                &format!("::pie::elemwise::rmsnorm_gated_f32_lanes<{t}, {lph}>"),
            ))
            .apply(lanes),
            &[
                x.arg(),
                gate.arg(),
                weight.map_or(ArgValue::ABSENT, |w| w.arg()),
                y.arg(),
                total.arg(),
                eps.arg(),
                i32::from(sigmoid_gate).arg(),
                heads.arg(),
                ctx.stage(),
            ],
        );
    }
    ctx.fire(
        OP,
        Fire::at(kernel(
            NORM_UNIT,
            &format!("::pie::elemwise::rmsnorm_gated_f32_in<{t}, 256>"),
        ))
        .apply(launch),
        &[
            x.arg(),
            gate.arg(),
            weight.map_or(ArgValue::ABSENT, |w| w.arg()),
            y.arg(),
            hidden.arg(),
            eps.arg(),
            i32::from(sigmoid_gate).arg(),
            heads.arg(),
            ctx.stage(),
        ],
    )
}

pub fn rmsnorm_grouped_plus_one(
    ctx: &Ctx,
    x: Tensor,
    weight: Tensor,
    group: u32,
    eps: f32,
    y: &mut Tensor,
) -> Result<(), Error> {
    rmsnorm_grouped(ctx, "elemwise.rmsnorm", x, weight, group, eps, true, y)
}

/// Each `group`-wide run of `x`'s rows RMS-normed into `y` and scaled by
/// its own `weight`s (`1 + weight` under `plus_one`).
#[allow(clippy::too_many_arguments)]
pub(crate) fn rmsnorm_grouped(
    ctx: &Ctx,
    op: &'static str,
    x: Tensor,
    weight: Tensor,
    group: u32,
    eps: f32,
    plus_one: bool,
    y: &mut Tensor,
) -> Result<(), Error> {
    let t = dtype_dispatch!(op, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let group = nonzero(op, "the group width", group)?;
    if y.width == 0 || !y.width.is_multiple_of(group) {
        return Err(refuse(
            op,
            format!(
                "the {}-wide row is not a whole number of {group}-wide groups",
                y.width
            ),
        ));
    }
    debug_assert_eq!(
        weight.width * weight.rows.max(1),
        y.width,
        "the weight spans the row's full width"
    );
    let groups = y.width / group;
    let launch = rows_per_head(op, y.rows, y.width, group)?;
    ctx.fire(
        op,
        Fire::at(kernel(
            NORM_UNIT,
            &format!("::pie::elemwise::rmsnorm_grouped<{t}, 256, {plus_one}>"),
        ))
        .apply(launch),
        &[
            x.arg(),
            weight.arg(),
            y.arg(),
            stated(op, group)?.arg(),
            stated(op, groups)?.arg(),
            eps.arg(),
            ctx.stage(),
        ],
    )
}

pub fn rmsnorm_gated_by(
    ctx: &Ctx,
    x: Tensor,
    gate: Tensor,
    weight: Tensor,
    heads: u32,
    eps: f32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.rmsnorm_gated_by";

    const KDA_BLOCK_MIN: u32 = WARP;

    const KDA_BLOCK_MAX: u32 = 128;

    debug_assert_eq!(x.dtype, Dtype::F32, "`{OP}` norms an f32 accumulator");
    debug_assert_eq!(weight.dtype, Dtype::F32, "`{OP}` scales by an f32 weight");
    let t = dtype_dispatch!(OP, gate.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    nonzero(OP, "the stated head count", heads)?;
    nonzero(OP, "rows", y.rows)?;
    if x.width == 0 || !x.width.is_multiple_of(heads) {
        return Err(refuse(
            OP,
            format!(
                "the {}-wide normed row does not divide by the stated head count {heads}",
                x.width
            ),
        ));
    }
    let d = x.width / heads;
    ctx.fire(
        OP,
        Fire::at(kernel(
            "elemwise/norm.hip",
            &format!("::pie::elemwise::rmsnorm_gated_by<{t}>"),
        ))
        .apply(Launch::grid(
            [y.rows, heads, 1],
            // Whole wavefronts: the block's sum reduces wave by wave.
            [
                d.next_multiple_of(WARP).clamp(KDA_BLOCK_MIN, KDA_BLOCK_MAX),
                1,
                1,
            ],
        )),
        &[
            x.arg(),
            gate.arg(),
            weight.arg(),
            y.arg(),
            stated(OP, heads)?.arg(),
            stated(OP, d)?.arg(),
            eps.arg(),
            ctx.stage(),
        ],
    )
}

pub fn standardize(ctx: &Ctx, bias: Tensor, scale: Tensor, out: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.standardize";
    let t = dtype_dispatch!(OP, out.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    nonzero(OP, "rows", out.rows)?;
    let width = stated(OP, nonzero(OP, "the standardized row's width", out.width)?)?;
    for (what, plane) in [("bias", bias), ("scale", scale)] {
        if plane.dtype != out.dtype {
            return Err(refuse(
                OP,
                format!(
                    "the {what} plane is {:?} and the rows it standardizes are {:?}; \
                     both planes ride the activation's element",
                    plane.dtype, out.dtype
                ),
            ));
        }
        if plane.elements() != u64::from(out.width) {
            return Err(refuse(
                OP,
                format!(
                    "the {what} plane is a {} x {} plane, and this row reads one \
                     scalar per column of a {}-wide rectangle",
                    plane.rows, plane.width, out.width
                ),
            ));
        }
    }
    ctx.fire(
        OP,
        Fire::at(kernel(
            NORM_UNIT,
            &format!("::pie::elemwise::standardize<{t}>"),
        ))
        .apply(route_rows(out.rows, out.width)),
        &[out.arg(), bias.arg(), scale.arg(), width.arg()],
    )
}

/// The norm a gated residual lands the next sublayer's input through.
pub struct Next {
    pub weight: Tensor,
    pub eps: f32,
    pub plus_one: bool,
    pub y: Tensor,
}

const LAYERNORM_UNIT: &str = "elemwise/layernorm.hip";

const BLOCK: u32 = 256;

pub fn layernorm_no_scale(ctx: &Ctx, x: Tensor, eps: f32, y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "elemwise.layernorm";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let hidden = stated(OP, nonzero(OP, "the normed width", y.width)?)?;
    let rows = nonzero(OP, "rows", y.rows)?;
    ctx.fire(
        OP,
        Fire::at(kernel(
            LAYERNORM_UNIT,
            &format!("::pie::elemwise::layernorm_no_scale<{t}, 256>"),
        ))
        .apply(Launch::per_row(rows, BLOCK)),
        &[x.arg(), y.arg(), hidden.arg(), eps.arg(), ctx.stage()],
    )
}

pub fn layernorm(
    ctx: &Ctx,
    x: Tensor,
    weight: Tensor,
    bias: Tensor,
    eps: f32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elemwise.layernorm";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let hidden = stated(OP, nonzero(OP, "the normed width", y.width)?)?;
    let rows = nonzero(OP, "rows", y.rows)?;
    ctx.fire(
        OP,
        Fire::at(kernel(
            LAYERNORM_UNIT,
            &format!("::pie::elemwise::layernorm<{t}, 256>"),
        ))
        .apply(Launch::per_row(rows, BLOCK)),
        &[
            x.arg(),
            weight.arg(),
            bias.arg(),
            y.arg(),
            hidden.arg(),
            eps.arg(),
            ctx.stage(),
        ],
    )
}

const STATS_BLOCK: u32 = 1024;

/// One wavefront merges a (lane, group)'s splits.
const FINALIZE_BLOCK: u32 = 64;

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
    const OP: &str = "conv.group_norm";
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
    let partials = ctx.scratch(OP, "conv.group_norm.partials", cells * 16)? as u64;
    let stats = ctx.scratch(OP, "conv.group_norm.stats", cells * 8)? as u64;
    let splits = stated(OP, splits)?;

    ctx.fire(
        OP,
        Fire::at(kernel(NORM_UNIT, "::pie::elemwise::group_norm_stats<1024>")).apply(Launch::grid(
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
        Fire::at(kernel(NORM_UNIT, "::pie::elemwise::group_norm_finalize")).apply(Launch::grid(
            [groups.unsigned_abs(), lanes.unsigned_abs(), 1],
            [FINALIZE_BLOCK, 1, 1],
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
        Fire::at(kernel(
            NORM_UNIT,
            if silu {
                "::pie::elemwise::group_norm_apply<true>"
            } else {
                "::pie::elemwise::group_norm_apply<false>"
            },
        ))
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
