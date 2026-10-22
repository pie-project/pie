use super::*;
use crate::error::Error;
use crate::jit::{
    Arg, Ctx, Fire, Launch, aligned16, count, dtype_dispatch, nonzero, refuse, stated, symbol,
};
use crate::tensor::Tensor;
use crate::voxels::{flat_elements, lanes_of};
use dtype::Dtype;

const NORM: &str = "elemwise/norm.cu";

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
    let width = if per_head_dim == 0 {
        y.width
    } else {
        per_head_dim
    };
    // A whole row's norm (no heads) in 16-byte runs a thread, read once.
    let runs = y.width.div_ceil(8 * BLOCK);
    if ctx.tuning().rows_vec
        && per_head_dim == 0
        && y.width > 256
        && y.width.is_multiple_of(8)
        && runs <= 4
        && x.width.is_multiple_of(8)
        && [x.ptr, weight.ptr, y.ptr].iter().all(|&at| aligned16(at))
    {
        let plus_one = template == OFFSET_BANK;
        let runs = runs.next_power_of_two();
        return ctx.fire(
            op,
            Fire::at(
                NORM,
                symbol(&format!(
                    "::pie::elemwise::rmsnorm_runs<{t}, {runs}, {plus_one}>"
                )),
            )
            .apply(Launch::grid([y.rows, 1, 1], [BLOCK, 1, 1]).programmatic(ctx.tuning().pdl)),
            &[
                x.arg(),
                weight.arg(),
                y.arg(),
                hidden.arg(),
                hidden.arg(),
                hidden.arg(),
                eps.arg(),
                ctx.stage(),
            ],
        );
    }
    if ctx.tuning().rows_vec
        && (width == 128 || width == 256)
        && [x.ptr, weight.ptr, y.ptr].iter().all(|&at| aligned16(at))
    {
        let units = y
            .rows
            .checked_mul(per_head_split(y.width, per_head_dim))
            .ok_or_else(|| {
                refuse(
                    op,
                    format!(
                        "{} rows of {} heads overflow a grid",
                        y.rows,
                        y.width / width
                    ),
                )
            })?;
        let plus_one = template == OFFSET_BANK;
        return ctx.fire(
            op,
            Fire::at(
                NORM,
                symbol(&format!(
                    "::pie::elemwise::rmsnorm_in_warp<{t}, {}, {plus_one}>",
                    width / 32
                )),
            )
            .apply(Launch::grid([units.div_ceil(8), 1, 1], [BLOCK, 1, 1])),
            &[
                x.arg(),
                weight.arg(),
                y.arg(),
                stated(op, units)?.arg(),
                eps.arg(),
                heads.arg(),
                ctx.stage(),
            ],
        );
    }
    // A whole row of 2048 to 8192 held in registers, read once.
    if per_head_dim == 0 && matches!(width, 2048 | 4096 | 8192) {
        let plus_one = template == OFFSET_BANK;
        return ctx.fire(
            op,
            Fire::at(
                NORM,
                symbol(&format!(
                    "::pie::elemwise::rmsnorm_held<{t}, {}, {plus_one}>",
                    width / 256
                )),
            )
            .apply(launch),
            &[x.arg(), weight.arg(), y.arg(), eps.arg(), ctx.stage()],
        );
    }
    ctx.fire(
        op,
        Fire::at(
            NORM,
            symbol(&format!("::pie::elemwise::{template}<{t}, 256>")),
        )
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
    rms_row(
        ctx,
        "elementwise.rmsnorm",
        ABSOLUTE_BANK,
        x,
        weight,
        y,
        0,
        eps,
    )
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
        "elementwise.rmsnorm_per_head",
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
    rms_row(
        ctx,
        "elementwise.rmsnorm_plus_one",
        OFFSET_BANK,
        x,
        weight,
        y,
        0,
        eps,
    )
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
        "elementwise.rmsnorm_per_head_plus_one",
        OFFSET_BANK,
        x,
        weight,
        y,
        head_dim,
        eps,
    )
}

/// The norm [`norm_gate`] lands beside its output: `y = rmsnorm(out) *
/// weight` (`1 + weight` where `plus_one`).
#[derive(Clone, Copy)]
pub struct Next {
    pub weight: Tensor,
    pub eps: f32,
    pub plus_one: bool,
    pub y: Tensor,
}

pub fn rmsnorm_no_scale(
    ctx: &Ctx,
    x: Tensor,
    head_dim: u32,
    eps: f32,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.rmsnorm_no_scale";
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
    // A warp a head over 16-byte runs, where the head is whole 256-element
    // runs (DeepSeek-V4's 512-wide q heads); 8 or 16 lanes a 64- or 128-wide
    // one (Muse Spark's).
    let lanes = (hidden / 8).min(32);
    if ctx.tuning().rows_vec
        && hidden % (8 * lanes) == 0
        && matches!(lanes, 8 | 16 | 32)
        && (1..=4).contains(&(hidden / (8 * lanes)))
        && [x.ptr, y.ptr].iter().all(|&at| aligned16(at))
    {
        let units = y
            .rows
            .checked_mul(per_head_split(y.width, head_dim))
            .ok_or_else(|| refuse(OP, format!("{} rows of heads overflow a grid", y.rows)))?;
        return ctx.fire(
            OP,
            Fire::at(
                NORM,
                symbol(&format!(
                    "::pie::elemwise::rmsnorm_no_scale_vec<{t}, {}, {lanes}>",
                    hidden / (8 * lanes)
                )),
            )
            .apply(Launch::grid(
                [units.div_ceil(256 / lanes.unsigned_abs()), 1, 1],
                [BLOCK, 1, 1],
            )),
            &[
                x.arg(),
                y.arg(),
                stated(OP, units)?.arg(),
                eps.arg(),
                heads.arg(),
                ctx.stage(),
            ],
        );
    }
    // A warp a head, the block form's sums in its order, where the head is
    // whole 256-element runs.
    if ctx.tuning().hc_tile
        && hidden % 256 == 0
        && (1..=4).contains(&(hidden / 256))
        && [x.ptr, y.ptr].iter().all(|&at| aligned16(at))
    {
        let units = y
            .rows
            .checked_mul(per_head_split(y.width, head_dim))
            .ok_or_else(|| refuse(OP, format!("{} rows of heads overflow a grid", y.rows)))?;
        return ctx.fire(
            OP,
            Fire::at(
                NORM,
                symbol(&format!(
                    "::pie::elemwise::rmsnorm_no_scale_warp<{t}, {}>",
                    hidden / 256
                )),
            )
            .apply(Launch::grid([units.div_ceil(8), 1, 1], [BLOCK, 1, 1])),
            &[
                x.arg(),
                y.arg(),
                stated(OP, units)?.arg(),
                eps.arg(),
                heads.arg(),
                ctx.stage(),
            ],
        );
    }
    ctx.fire(
        OP,
        Fire::at(
            NORM,
            symbol(&format!("::pie::elemwise::rmsnorm_no_scale<{t}, 256>")),
        )
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
    const OP: &str = "elementwise.rmsnorm_gated";
    debug_assert!(
        matches!(x.dtype, Dtype::F32 | Dtype::Bf16),
        "`{OP}` norms an f32 accumulator, or one landed in bf16"
    );
    let xin = if x.dtype == Dtype::Bf16 {
        "::pie::bf16"
    } else {
        "float"
    };
    debug_assert!(
        weight.is_none_or(|w| w.dtype == Dtype::F32),
        "`{OP}` scales by an f32 weight"
    );
    let w_ptr = weight.map_or(0, |w| w.ptr);
    let w_arg = weight.map_or(crate::jit::ArgValue::ABSENT, |w| w.arg());
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
    let width = if head_dim == 0 { y.width } else { head_dim };
    let aligned = [x.ptr, gate.ptr, w_ptr, y.ptr]
        .iter()
        .all(|&at| aligned16(at));
    // 32 lanes a 128- or 256-wide head (four or eight values a lane); 8 a
    // 64-wide one (eight a lane).
    let (per, lanes) = match width {
        64 => (8, 8),
        128 | 256 => (width / 32, 32),
        _ => (0, 0),
    };
    if aligned && lanes != 0 {
        let units = y
            .rows
            .checked_mul(per_head_split(y.width, head_dim))
            .ok_or_else(|| {
                refuse(
                    OP,
                    format!(
                        "{} rows of {} heads overflow a grid",
                        y.rows,
                        y.width / width
                    ),
                )
            })?;
        return ctx.fire(
            OP,
            Fire::at(
                NORM,
                symbol(&format!(
                    "::pie::elemwise::rmsnorm_gated_in_warp<{xin}, {t}, {per}, {lanes}>"
                )),
            )
            .apply(Launch::grid(
                [units.div_ceil(256 / lanes), 1, 1],
                [BLOCK, 1, 1],
            )),
            &[
                x.arg(),
                gate.arg(),
                w_arg,
                y.arg(),
                stated(OP, units)?.arg(),
                eps.arg(),
                i32::from(sigmoid_gate).arg(),
                heads.arg(),
                ctx.stage(),
            ],
        );
    }
    ctx.fire(
        OP,
        Fire::at(
            NORM,
            symbol(&format!(
                "::pie::elemwise::rmsnorm_gated_in<{xin}, {t}, 256>"
            )),
        )
        .apply(launch),
        &[
            x.arg(),
            gate.arg(),
            w_arg,
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
    rmsnorm_grouped(
        ctx,
        "elementwise.rmsnorm_grouped_plus_one",
        x,
        weight,
        group,
        eps,
        true,
        y,
    )
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
        Fire::at(
            NORM,
            symbol(&format!(
                "::pie::elemwise::rmsnorm_grouped<{t}, 256, {plus_one}>"
            )),
        )
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
    const OP: &str = "elementwise.rmsnorm_gated_by";

    debug_assert_eq!(x.dtype, Dtype::F32, "`{OP}` norms an f32 accumulator");
    debug_assert_eq!(weight.dtype, Dtype::F32, "`{OP}` scales by an f32 weight");
    let t = dtype_dispatch!(OP, gate.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    nonzero(OP, "the stated head count", heads)?;
    nonzero(OP, "rows", y.rows)?;
    let d = x.width / heads;
    if x.width == 0 || !x.width.is_multiple_of(heads) || !d.is_multiple_of(WARP) || d > 8 * WARP {
        return Err(refuse(
            OP,
            format!(
                "the {}-wide normed row is not {heads} heads of a whole number of 32 up to 256",
                x.width
            ),
        ));
    }
    let pairs = y
        .rows
        .checked_mul(heads)
        .ok_or_else(|| refuse(OP, "the rows' heads do not count"))?;
    // A warp's (row, head) pairs.
    let per_warp: u32 = if ctx.tuning().rows_vec { 2 } else { 1 };
    ctx.fire(
        OP,
        Fire::at(
            NORM,
            symbol(&format!(
                "::pie::elemwise::rmsnorm_gated_by<{t}, {per_warp}>"
            )),
        )
        .apply(
            Launch::grid([pairs.div_ceil(8 * per_warp), 1, 1], [256, 1, 1])
                .programmatic(ctx.tuning().pdl),
        ),
        &[
            x.arg(),
            gate.arg(),
            weight.arg(),
            y.arg(),
            stated(OP, y.rows)?.arg(),
            stated(OP, heads)?.arg(),
            stated(OP, d)?.arg(),
            eps.arg(),
            ctx.stage(),
        ],
    )
}

pub fn standardize(ctx: &Ctx, bias: Tensor, scale: Tensor, out: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.standardize";
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
        Fire::at(NORM, symbol(&format!("::pie::elemwise::standardize<{t}>")))
            .apply(route_rows(out.rows, out.width)),
        &[out.arg(), bias.arg(), scale.arg(), width.arg()],
    )
}

const BLOCK: u32 = 256;

pub fn layernorm_no_scale(ctx: &Ctx, x: Tensor, eps: f32, y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.layernorm_no_scale";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let hidden = stated(OP, nonzero(OP, "the normed width", y.width)?)?;
    let rows = nonzero(OP, "rows", y.rows)?;
    ctx.fire(
        OP,
        Fire::at(
            NORM,
            symbol(&format!("::pie::elemwise::layernorm_no_scale<{t}, 256>")),
        )
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
    const OP: &str = "elementwise.layernorm";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let hidden = stated(OP, nonzero(OP, "the normed width", y.width)?)?;
    let rows = nonzero(OP, "rows", y.rows)?;
    ctx.fire(
        OP,
        Fire::at(
            NORM,
            symbol(&format!("::pie::elemwise::layernorm<{t}, 256>")),
        )
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
        Fire::at(NORM, "::pie::elemwise::group_norm_stats<1024>").apply(Launch::grid(
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
        Fire::at(NORM, "::pie::elemwise::group_norm_finalize").apply(Launch::grid(
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
            NORM,
            if silu {
                "::pie::elemwise::group_norm_apply<true>"
            } else {
                "::pie::elemwise::group_norm_apply<false>"
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
