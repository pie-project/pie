//! Affine-quantized matmuls (`linear.matmul`, `linear.lm_head` over a bank):
//! the engine's `kernel.qmm` ladder, its rungs in order, and the first that
//! fits the shape runs. The tiled rung is `kernels/linear/qmm_tiled.slang`
//! on SPIR-V (with its cooperative-matrix tier) and `qmm_quad.slang` on
//! WGSL, which lands whole words; the register-tiled rung is
//! `qmm_register.slang` (SPIR-V); the vector rungs are `linear::qmv`'s.

use crate::encode::{Arg, Ctx, Fire, Target, Tier, dtype_dispatch, instantiated, refuse, stated};
use crate::error::Error;
use crate::tensor::{Bank, Tensor};

const TILED: &str = "linear/qmm_tiled.slang";

const QUAD: &str = "linear/qmm_quad.slang";

const REGISTER: &str = "linear/qmm_register.slang";

/// The tiled rung's row tiles, smallest first, and its column tiles.
const BM_RUNGS: [u32; 4] = [8, 16, 32, 64];

const BN_RUNGS: [u32; 2] = [16, 32];

/// The cooperative-matrix tile, square.
const COOPMAT_TILE: u32 = 64;

/// The register-tiled rung's column tile.
const REGISTER_COLS: u32 = 64;

/// A `kernel.qmm` rung.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Qmm {
    /// The tiled kernel at SPIR-V's cooperative-matrix tier, 64 x 64 tiles.
    Coopmat,
    /// The register-tiled kernel (SPIR-V): `bm` rows (32 or 64) a tile, each
    /// invocation holding a 4 x 4 block; `k16` walks 16 codes a step.
    RegisterTiled { bm: u32, k16: bool },
    /// The tiled kernel, through workgroup memory; it declines a batch whose
    /// padded row tiles the row capacity cannot hold. A 32-wide column tile
    /// pays once the grid fills `crossover_tg` workgroups.
    Tiled { crossover_tg: u32 },
    /// The vector kernel at `vecs` activation rows a workgroup; it fits
    /// every shape.
    Qmv { vecs: u32 },
    /// The one-row kernel (SPIR-V's subgroup tier): `lanes` subgroup lanes a
    /// column, `rows` columns a lane.
    OneRow { lanes: u32, rows: u32 },
}

/// A matmul's operands and extents, checked once for every rung.
pub(crate) struct Bounds {
    pub act: Tensor,
    pub w: Bank,
    pub biases: Tensor,
    pub y: Tensor,
    /// Activation rows, output columns, contraction.
    pub m: u32,
    pub n: u32,
    pub k: u32,
    pub group: u32,
    pub bits: u32,
}

fn bounds(op: &'static str, act: Tensor, w: Bank, y: Tensor) -> Result<Bounds, Error> {
    dtype_dispatch!(op, act.dtype, { Bf16 => () });
    let Some(biases) = w.biases else {
        return Err(refuse(
            op,
            format!(
                "the weight is a symmetric {}-bit bank in groups of {}, and no point is \
                 instantiated for one",
                w.bits, w.group
            ),
        ));
    };
    if y.width == 0 {
        return Err(refuse(op, "the columns this projection lands are zero"));
    }
    if act.width == 0 {
        return Err(refuse(op, "the contraction this projection walks is zero"));
    }
    debug_assert_eq!(
        act.rows, y.rows,
        "the activation's rows are the rows the result lands"
    );
    if !matches!(w.group, 32 | 64 | 128) || !matches!(w.bits, 2 | 4 | 8) {
        return Err(refuse(
            op,
            format!(
                "no point is instantiated at group size {} and bit width {}",
                w.group, w.bits
            ),
        ));
    }
    if !act.width.is_multiple_of(w.group) {
        return Err(refuse(
            op,
            format!(
                "the contraction is {}, not a whole number of {}-code groups: every point \
                 indexes its scales as `k / group`",
                act.width, w.group
            ),
        ));
    }
    Ok(Bounds {
        act,
        w,
        biases,
        y,
        m: y.rows,
        n: y.width,
        k: act.width,
        group: w.group,
        bits: w.bits,
    })
}

/// `y = act * dequant(w)^T` at the first rung of `ladder` that fits;
/// `capacity_rows` is the rows `act` and `y` can hold, padding included.
pub fn matmul(
    ctx: &Ctx<'_>,
    ladder: &[Qmm],
    act: Tensor,
    w: Bank,
    y: Tensor,
    capacity_rows: u32,
) -> Result<(), Error> {
    fire(ctx, "linear.matmul", ladder, act, w, y, capacity_rows)
}

pub fn lm_head(
    ctx: &Ctx<'_>,
    ladder: &[Qmm],
    act: Tensor,
    w: Bank,
    y: Tensor,
    capacity_rows: u32,
) -> Result<(), Error> {
    fire(ctx, "linear.lm_head", ladder, act, w, y, capacity_rows)
}

fn fire(
    ctx: &Ctx<'_>,
    op: &'static str,
    ladder: &[Qmm],
    act: Tensor,
    w: Bank,
    y: Tensor,
    capacity_rows: u32,
) -> Result<(), Error> {
    let b = bounds(op, act, w, y)?;
    if b.m == 0 {
        return Ok(());
    }
    for rung in ladder {
        match *rung {
            Qmm::Coopmat => return coopmat(ctx, op, &b),
            Qmm::RegisterTiled { bm, k16 } => return register(ctx, op, &b, bm, k16),
            Qmm::Tiled { crossover_tg } => {
                if let Some((bm, padded)) = row_tile(b.m, capacity_rows)
                    && let Some(bn) = column_tile(b.n, padded / bm, crossover_tg)
                {
                    return tiled(ctx, op, &b, bm, bn, padded);
                }
            }
            Qmm::OneRow { lanes, rows } => {
                return crate::linear::qmv::fire_row(ctx, op, &b, lanes, rows);
            }
            Qmm::Qmv { vecs } => return crate::linear::qmv::fire(ctx, op, &b, vecs),
        }
    }
    Err(refuse(
        op,
        "no `kernel.qmm` rung fits: a ladder ends on the vector kernel, which fits every shape",
    ))
}

/// The tiled rung's arguments, in binding order.
fn args(op: &'static str, b: &Bounds) -> Result<[crate::encode::ArgValue; 8], Error> {
    Ok([
        b.w.codes.arg(),
        b.w.scales.arg(),
        b.biases.arg(),
        b.act.arg(),
        b.y.arg_mut(),
        stated(op, b.k)?.arg(),
        stated(op, b.n)?.arg(),
        stated(op, b.m)?.arg(),
    ])
}

/// The largest row tile at most the rows whose padding the capacity holds,
/// and the padded rows; else the smallest tile, if its padding fits.
fn row_tile(rows: u32, capacity: u32) -> Option<(u32, u32)> {
    let fits = |rung: u32| {
        let padded = rows.max(1).div_ceil(rung) * rung;
        (padded <= capacity.max(1)).then_some((rung, padded))
    };
    BM_RUNGS
        .iter()
        .rev()
        .copied()
        .filter(|rung| rows >= *rung)
        .find_map(fits)
        .or_else(|| fits(BM_RUNGS[0]))
}

/// A 32-wide column tile once the grid fills `crossover_tg` workgroups, else
/// 16; none when the width is no whole number of 16.
fn column_tile(n: u32, row_tiles: u32, crossover_tg: u32) -> Option<u32> {
    if !n.is_multiple_of(BN_RUNGS[0]) {
        return None;
    }
    if n.is_multiple_of(32) && (n / 32) * row_tiles.max(1) >= crossover_tg {
        return Some(32);
    }
    Some(BN_RUNGS[0])
}

fn tiled(
    ctx: &Ctx<'_>,
    op: &'static str,
    b: &Bounds,
    bm: u32,
    bn: u32,
    padded: u32,
) -> Result<(), Error> {
    let (g, bits) = (b.group, b.bits);
    let (unit, name) = match ctx.target() {
        Target::Spirv => (
            TILED,
            format!("linear_qmm_tiled_bf16_gs_{g}_b_{bits}_bm_{bm}_bn_{bn}"),
        ),
        Target::Wgsl => (
            QUAD,
            format!("linear_qmm_quad_bf16_gs_{g}_b_{bits}_bm_{bm}_bn_{bn}"),
        ),
    };
    let entry = instantiated(op, name)?;
    ctx.fire(
        Fire::at(unit, entry)
            .groups([b.n.div_ceil(bn), padded / bm, 1])
            .group([32, 2, if bm < 16 { 1 } else { 2 }]),
        &args(op, b)?,
    )
}

fn coopmat(ctx: &Ctx<'_>, op: &'static str, b: &Bounds) -> Result<(), Error> {
    let entry = instantiated(
        op,
        format!(
            "linear_qmm_tiled_bf16_gs_{}_b_{}_bm_{COOPMAT_TILE}_bn_{COOPMAT_TILE}",
            b.group, b.bits
        ),
    )?;
    ctx.fire(
        Fire::at(TILED, entry)
            .tier(Tier::Coopmat)
            .groups([b.n.div_ceil(COOPMAT_TILE), b.m.div_ceil(COOPMAT_TILE), 1])
            .group([32, 4, 1]),
        &args(op, b)?,
    )
}

fn register(ctx: &Ctx<'_>, op: &'static str, b: &Bounds, bm: u32, k16: bool) -> Result<(), Error> {
    if !matches!(bm, 32 | 64) {
        return Err(refuse(
            op,
            format!("no register-tiled point holds {bm} rows a tile"),
        ));
    }
    let entry = instantiated(
        op,
        format!(
            "linear_qmm_register{}_bf16_gs_{}_b_{}_bm_{bm}",
            if k16 { "_k16" } else { "" },
            b.group,
            b.bits
        ),
    )?;
    ctx.fire(
        Fire::at(REGISTER, entry)
            .groups([b.n.div_ceil(REGISTER_COLS), b.m.div_ceil(bm), 1])
            .group([REGISTER_COLS / 4, bm / 4, 1]),
        &args(op, b)?,
    )
}
