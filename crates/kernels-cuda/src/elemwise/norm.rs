use crate::error::Error;
use dtype::Dtype;

use crate::jit::{
    Arg, Ctx, Fire, Launch, aligned16, dtype_dispatch, nonzero, refuse, stated, symbol,
};
use crate::tensor::Tensor;

const FILE: &str = "elemwise/norm.cuh";

const BLOCK: u32 = 256;

const WARP: u32 = 32;

const VEC_WIDTH: u32 = 8;

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

fn elementwise(op: &'static str, t: Tensor) -> Result<(Launch, u64), Error> {
    let n = t.elements();
    let lanes = u32::try_from(n).map_err(|_| {
        refuse(
            op,
            format!("{n} elements do not fit a 32-bit launch extent"),
        )
    })?;
    nonzero(op, "the element count", lanes)?;
    Ok((Launch::flat(lanes, BLOCK), n))
}

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
                FILE,
                symbol(&format!(
                    "::pie::elemwise::rmsnorm_runs<{t}, {runs}, {plus_one}>"
                )),
            )
            .apply(Launch::grid([y.rows, 1, 1], [BLOCK, 1, 1])),
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
                FILE,
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
                FILE,
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
            FILE,
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

/// Muse Spark's post-norm and residual gate (`elementwise.norm_gate`): a
/// row's `x` times `scale` (rounded), RMS-normed unscaled (rounded), and with
/// `res` and its f32 `gate` the residual's gated sum; with `next`, the next
/// sublayer's input norm over it in the same pass.
#[allow(clippy::too_many_arguments)]
pub fn norm_gate(
    ctx: &Ctx,
    x: Tensor,
    scale: Option<Tensor>,
    res: Option<Tensor>,
    gate: Option<Tensor>,
    eps: f32,
    temperature: f32,
    fan: u32,
    next: Option<Next>,
    y: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.norm_gate";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    if res.is_some() != gate.is_some() || gate.is_some_and(|g| g.dtype != Dtype::F32) {
        return Err(refuse(OP, "a residual comes with its f32 gate"));
    }
    let rows = nonzero(OP, "rows", y.rows)?;
    let absent = |t: Option<Tensor>| t.map_or(crate::jit::ArgValue::ABSENT, |t| t.arg());
    // A 16-byte run a thread, the row in the block's registers.
    let threads = y.width / 8;
    let planes = [
        Some(x),
        scale,
        res,
        gate,
        Some(*y),
        next.map(|n| n.weight),
        next.map(|n| n.y),
    ];
    if !y.width.is_multiple_of(8)
        || !threads.is_power_of_two()
        || !(32..=1024).contains(&threads)
        || !planes.into_iter().flatten().all(|t| aligned16(t.ptr))
    {
        return Err(refuse(
            OP,
            "a row of 256 to 8192 elements, a power of two, 16-byte aligned",
        ));
    }
    let tail = next.is_some();
    // A long fire's block walks sixteen rows; a short one's keeps a row.
    const WALK: u32 = 16;
    let walk = rows >= 8192;
    let (entry, launch) = if walk {
        (
            format!("::pie::elemwise::norm_gate_rows<{t}, {threads}, {tail}, {WALK}>"),
            Launch::per_row(rows.div_ceil(WALK), threads),
        )
    } else {
        (
            format!("::pie::elemwise::norm_gate<{t}, {threads}, {tail}>"),
            Launch::per_row(rows, threads),
        )
    };
    let mut args = vec![
        x.arg(),
        absent(scale),
        absent(res),
        absent(gate),
        y.arg(),
        eps.arg(),
        temperature.arg(),
        stated(OP, fan)?.arg(),
        absent(next.map(|n| n.weight)),
        next.map_or(0.0, |n| n.eps).arg(),
        i32::from(next.is_some_and(|n| n.plus_one)).arg(),
        absent(next.map(|n| n.y)),
    ];
    if walk {
        args.push(stated(OP, rows)?.arg());
    }
    args.push(ctx.stage());
    ctx.fire(OP, Fire::at(FILE, symbol(&entry)).apply(launch), &args)
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
                FILE,
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
                FILE,
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
            FILE,
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
                FILE,
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
            FILE,
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
            FILE,
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
            "elemwise/norm.cuh",
            symbol(&format!(
                "::pie::elemwise::rmsnorm_gated_by<{t}, {per_warp}>"
            )),
        )
        .apply(Launch::grid(
            [pairs.div_ceil(8 * per_warp), 1, 1],
            [256, 1, 1],
        )),
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

pub fn residual_add(ctx: &Ctx, x: Tensor, y: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.residual_add";
    let t = dtype_dispatch!(OP, y.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let (launch, n) = elementwise(OP, *y)?;
    // Rows of whole 16-byte groups add eight lanes a thread.
    if y.dtype == Dtype::Bf16 && y.width.is_multiple_of(8) && (y.ptr | x.ptr).is_multiple_of(16) {
        let lanes = u32::try_from(n / 8).unwrap_or(u32::MAX);
        return ctx.fire(
            OP,
            Fire::at(FILE, "::pie::elemwise::residual_add_vec8").apply(Launch::flat(lanes, BLOCK)),
            &[
                y.arg(),
                x.arg(),
                (n / 8).arg(),
                stated(OP, y.width / 8)?.arg(),
                ctx.stage(),
            ],
        );
    }
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::elemwise::residual_add<{t}>"))).apply(launch),
        &[
            y.arg(),
            x.arg(),
            n.arg(),
            stated(OP, y.width)?.arg(),
            ctx.stage(),
        ],
    )
}

pub fn residual_add_rmsnorm(
    ctx: &Ctx,
    x: Tensor,
    y: &mut Tensor,
    weight: Tensor,
    plus_one: bool,
    eps: f32,
    out: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "elementwise.residual_add_rmsnorm";
    let t = dtype_dispatch!(OP, y.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert!(
        x.rows == y.rows && x.width == y.width && out.rows == y.rows && out.width == y.width,
        "the residual, the stream and the normed row share one shape"
    );
    let hidden = stated(OP, nonzero(OP, "the normed width", y.width)?)?;
    let rows = nonzero(OP, "rows", y.rows)?;
    let plus = if plus_one { "true" } else { "false" };
    let vectors = y.dtype == Dtype::Bf16
        && y.width.is_multiple_of(VEC_WIDTH)
        && [x.ptr, y.ptr, weight.ptr, out.ptr]
            .iter()
            .all(|&at| aligned16(at));
    let per = (y.width / VEC_WIDTH).div_ceil(BLOCK);
    let (entrypoint, block) = if vectors && ctx.tuning().rows_vec && (1..=4).contains(&per) {
        (
            format!("::pie::elemwise::residual_add_rmsnorm_reg<{per}, {plus}>"),
            BLOCK,
        )
    } else if vectors {
        let block = (y.width / VEC_WIDTH).clamp(WARP, BLOCK).next_power_of_two();
        (
            format!("::pie::elemwise::residual_add_rmsnorm_vec8<{block}, {plus}>"),
            block,
        )
    } else {
        (
            format!("::pie::elemwise::residual_add_rmsnorm<{t}, 256, {plus}>"),
            BLOCK,
        )
    };
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&entrypoint)).apply(Launch::per_row(rows, block)),
        &[
            x.arg(),
            y.arg(),
            weight.arg(),
            out.arg(),
            hidden.arg(),
            eps.arg(),
            ctx.stage(),
        ],
    )
}

pub struct PostNorm<'a> {
    pub weight: Tensor,
    pub plus_one: bool,
    pub eps: f32,
    pub out: &'a mut Tensor,
}

/// The fewest rows a fused norm-and-fold takes at `kernel.elemwise.norm_fold_block`.
const FOLD_BLOCK_FROM: u32 = 1024;

/// Which of a fused norm-and-fold's rows land: the normed row `t` and the
/// stream `y` it folds into in place. A row nothing reads past the chain need
/// not be written; the 16-byte form honours it, the element form writes all.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Keep {
    pub t: bool,
    pub y: bool,
}

impl Keep {
    pub const ALL: Keep = Keep { t: true, y: true };
}

#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_residual_add(
    ctx: &Ctx,
    x: Tensor,
    w0: Tensor,
    eps0: f32,
    t: &mut Tensor,
    y: &mut Tensor,
    scale: Option<(Tensor, &mut Tensor)>,
    post: Option<PostNorm<'_>>,
) -> Result<(), Error> {
    rmsnorm_residual_add_keeping(ctx, x, w0, eps0, t, y, scale, post, Keep::ALL)
}

/// [`rmsnorm_residual_add`] landing only the rows `keep` names (of `t` and
/// `y`; `scaled` and the post-norm's `out` always land).
#[allow(clippy::too_many_arguments)]
pub fn rmsnorm_residual_add_keeping(
    ctx: &Ctx,
    x: Tensor,
    w0: Tensor,
    eps0: f32,
    t: &mut Tensor,
    y: &mut Tensor,
    scale: Option<(Tensor, &mut Tensor)>,
    post: Option<PostNorm<'_>>,
    keep: Keep,
) -> Result<(), Error> {
    const OP: &str = "elementwise.rmsnorm_residual_add";
    let per_thread = [8u32, 16, 32]
        .into_iter()
        .find(|per| y.width <= BLOCK * per)
        .ok_or_else(|| {
            refuse(
                OP,
                format!(
                    "a {}-wide row is wider than the fused chain seats ({})",
                    y.width,
                    BLOCK * 32
                ),
            )
        })?;
    let ty = dtype_dispatch!(OP, y.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    debug_assert!(
        x.rows == y.rows && x.width == y.width && t.rows == y.rows && t.width == y.width,
        "the normed row, its scaled copy and the stream share one shape"
    );
    let hidden = stated(OP, nonzero(OP, "the normed width", y.width)?)?;
    let rows = nonzero(OP, "rows", y.rows)?;
    let (s_arg, scaled_arg, has_scale) = match &scale {
        Some((s, scaled)) => (s.arg(), scaled.arg(), "true"),
        None => (
            crate::jit::ArgValue::ABSENT,
            crate::jit::ArgValue::ABSENT,
            "false",
        ),
    };
    let (w1_arg, out_arg, eps1, has_post, plus) = match &post {
        Some(post) => (
            post.weight.arg(),
            post.out.arg(),
            post.eps,
            "true",
            if post.plus_one { "true" } else { "false" },
        ),
        None => (
            crate::jit::ArgValue::ABSENT,
            crate::jit::ArgValue::ABSENT,
            0.0,
            "false",
            "false",
        ),
    };
    let mut aligned = vec![x.ptr, w0.ptr, t.ptr, y.ptr];
    if let Some((_, scaled)) = &scale {
        aligned.push(scaled.ptr);
    }
    if let Some(post) = &post {
        aligned.extend([post.weight.ptr, post.out.ptr]);
    }
    let vectors = y.dtype == Dtype::Bf16
        && y.width.is_multiple_of(VEC_WIDTH)
        && aligned.iter().all(|&at| aligned16(at));
    let (entrypoint, block) = if vectors {
        let nvec = y.width / VEC_WIDTH;
        let block = match ctx.tuning().norm_fold_block {
            // a prefill's many rows seat more an SM at a smaller block, several
            // runs a thread (at most 16: a 32-thread block takes a 4096-wide
            // row); a decode's few finish sooner a run a thread
            stated if stated > 0 && y.rows >= FOLD_BLOCK_FROM => {
                stated.max(nvec.div_ceil(16).next_power_of_two()).min(512)
            }
            _ => nvec.clamp(WARP, 512).next_power_of_two(),
        };
        let chunks = nvec.div_ceil(block);
        // a row nothing reads goes unwritten; with both kept the name is the
        // one every chip has always instantiated
        let keeping = if keep == Keep::ALL {
            String::new()
        } else {
            format!(", {}, {}", keep.t, keep.y)
        };
        (
            format!(
                "::pie::elemwise::rmsnorm_residual_add_vec8<{block}, {chunks}, {has_scale}, {has_post}, {plus}{keeping}>"
            ),
            block,
        )
    } else {
        (
            format!(
                "::pie::elemwise::rmsnorm_residual_add<{ty}, {BLOCK}, {per_thread}, {has_scale}, {has_post}, {plus}>"
            ),
            BLOCK,
        )
    };
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&entrypoint)).apply(Launch::per_row(rows, block)),
        &[
            x.arg(),
            w0.arg(),
            t.arg(),
            y.arg(),
            s_arg,
            scaled_arg,
            w1_arg,
            out_arg,
            hidden.arg(),
            eps0.arg(),
            eps1.arg(),
            ctx.stage(),
        ],
    )
}

pub fn add_bias(ctx: &Ctx, bias: Tensor, out: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.add_bias";
    let t = dtype_dispatch!(OP, out.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16", F32 => "float" });
    let tb = dtype_dispatch!(OP, bias.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16", F32 => "float" });
    nonzero(OP, "rows", out.rows)?;
    let width = stated(OP, nonzero(OP, "the biased row's width", out.width)?)?;
    ctx.fire(
        OP,
        Fire::at(
            FILE,
            symbol(&format!("::pie::elemwise::add_bias<{t}, {tb}>")),
        )
        .apply(route_rows(out.rows, out.width)),
        &[out.arg(), bias.arg(), width.arg(), ctx.stage()],
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
        Fire::at(FILE, symbol(&format!("::pie::elemwise::standardize<{t}>")))
            .apply(route_rows(out.rows, out.width)),
        &[out.arg(), bias.arg(), scale.arg(), width.arg()],
    )
}

pub fn silu_scaled(ctx: &Ctx, s: f32, x: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.silu_scaled";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let (launch, n) = elementwise(OP, *x)?;
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::elemwise::silu_scaled<{t}>"))).apply(launch),
        &[
            x.arg(),
            s.arg(),
            n.arg(),
            stated(OP, x.width)?.arg(),
            ctx.stage(),
        ],
    )
}

pub fn mul_scalar(ctx: &Ctx, s: f32, x: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.mul_scalar";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16", F32 => "float" });
    let (launch, n) = elementwise(OP, *x)?;
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::elemwise::mul_scalar<{t}>"))).apply(launch),
        &[
            x.arg(),
            s.arg(),
            n.arg(),
            stated(OP, x.width)?.arg(),
            ctx.stage(),
        ],
    )
}

pub fn scale(ctx: &Ctx, s: Tensor, x: &mut Tensor) -> Result<(), Error> {
    const OP: &str = "elementwise.scale";
    let t = dtype_dispatch!(OP, x.dtype, { Bf16 => "::pie::bf16", F16 => "::pie::f16" });
    let (launch, n) = elementwise(OP, *x)?;
    ctx.fire(
        OP,
        Fire::at(FILE, symbol(&format!("::pie::elemwise::scale<{t}>"))).apply(launch),
        &[
            x.arg(),
            s.arg(),
            n.arg(),
            stated(OP, x.width)?.arg(),
            ctx.stage(),
        ],
    )
}

pub fn res_blend(
    ctx: &Ctx,
    prefix: Tensor,
    blocks: &[Tensor],
    weight: Tensor,
    eps: f32,
    proj: Tensor,
    y: &mut Tensor,
) -> Result<(), Error> {
    crate::attn::res_blend(ctx, prefix, blocks, weight, eps, proj, y)
}

/// [`res_blend`] with the RMS norm of the blended row in one pass; see
/// [`crate::attn::res_blend_norm`].
#[allow(clippy::too_many_arguments)]
pub fn res_blend_norm(
    ctx: &Ctx,
    prefix: Tensor,
    blocks: &[Tensor],
    weight: Tensor,
    eps: f32,
    proj: Tensor,
    out_weight: Tensor,
    out_eps: f32,
    y: &mut Tensor,
) -> Result<bool, Error> {
    crate::attn::res_blend_norm(
        ctx, prefix, blocks, weight, eps, proj, out_weight, out_eps, y,
    )
}
