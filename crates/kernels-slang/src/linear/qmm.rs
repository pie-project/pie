//! Affine-quantized matmuls (`linear.matmul`, `linear.matmul` over a bank):
//! the engine's `kernel.qmm` ladder, its rungs in order, and the first that
//! fits the shape runs. The tiled rung is `kernels/linear/qmm_tiled.slang`
//! on SPIR-V (with its cooperative-matrix tier) and `qmm_quad.slang` on
//! WGSL, which lands whole words; the register-tiled rung is
//! `qmm_register.slang` (SPIR-V); the vector rungs are `linear::qmv`'s.

use crate::encode::{
    Arg, Ctx, Fire, Target, Tier, distinct, dtype_dispatch, instantiated, refuse, stated,
};
use crate::error::Error;
use crate::tensor::{Bank, Tensor};

const TILED: &str = "linear/qmm_tiled.slang";

const QUAD: &str = "linear/qmm_quad.slang";

const REGISTER: &str = "linear/qmm_register.slang";

/// The tiled rung's row tiles, smallest first, and its column tiles.
const BM_RUNGS: [u32; 4] = [8, 16, 32, 64];

const BN_RUNGS: [u32; 2] = [16, 32];

/// The contraction the tiled rungs walk a step (`PIE_BK`): a shape whose
/// contraction is not a whole number of them falls to the vector kernel.
pub const TILE_K: u32 = 32;

/// The cooperative-matrix tile, square.
const COOPMAT_TILE: u32 = 64;

/// Whether the tiled rungs are instantiated for banks of `group`-code
/// groups at `bits` bits.
#[must_use]
pub const fn stamped(group: u32, bits: u32) -> bool {
    matches!(group, 32 | 64 | 128) && matches!(bits, 2 | 4 | 8)
}

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
#[derive(Clone, Copy)]
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
    if !stamped(w.group, w.bits) {
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
    fire(ctx, "linear.matmul", ladder, act, w, y, capacity_rows)
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
    let target = ctx.target();
    if let Some(names) = ctx.naming() {
        return names.either(&kernels(target, op, ladder, &b, capacity_rows)?);
    }
    if b.m == 0 {
        return Ok(());
    }
    for rung in ladder {
        if let Some(kernel) = kernel(target, op, *rung, &b, capacity_rows)? {
            return ctx.fire(kernel, &args(op, &b)?);
        }
    }
    Err(refuse(
        op,
        "no `kernel.qmm` rung fits: a ladder ends on the vector kernel, which fits every shape",
    ))
}

/// The kernel `rung` runs the batch with, at its grid; `None` where the rung
/// declines the batch.
fn kernel(
    target: Target,
    op: &'static str,
    rung: Qmm,
    b: &Bounds,
    capacity_rows: u32,
) -> Result<Option<Fire>, Error> {
    Ok(match rung {
        Qmm::Coopmat => Some(coopmat(op, b)?),
        Qmm::RegisterTiled { bm, k16 } => Some(register(op, b, bm, k16)?),
        Qmm::Tiled { crossover_tg } => {
            let tiles = row_tile(b.m, capacity_rows).and_then(|(bm, padded)| {
                column_tile(b.n, padded / bm, crossover_tg).map(|bn| (bm, bn, padded))
            });
            match tiles {
                Some((bm, bn, padded)) => Some(tiled(target, op, b, bm, bn, padded)?),
                None => None,
            }
        }
        Qmm::OneRow { lanes, rows } => Some(crate::linear::qmv::row_kernel(op, b, lanes, rows)?),
        Qmm::Qmv { vecs } => Some(crate::linear::qmv::kernel(target, op, b, vecs)?),
    })
}

/// Every kernel a rung of `ladder` runs some batch the capacity holds with,
/// whatever its rows: `ladder` is every rung some row count reaches.
fn kernels(
    target: Target,
    op: &'static str,
    ladder: &[Qmm],
    b: &Bounds,
    capacity_rows: u32,
) -> Result<Vec<Fire>, Error> {
    let mut all = Vec::new();
    for rung in ladder {
        for m in breaks(*rung, capacity_rows) {
            if let Some(kernel) = kernel(target, op, *rung, &Bounds { m, ..*b }, capacity_rows)? {
                distinct(&mut all, kernel);
            }
        }
    }
    Ok(all)
}

/// Row counts at which the kernel `rung` runs changes: [`kernel`]'s answers
/// at these are every answer it has over the batches the capacity holds.
fn breaks(rung: Qmm, capacity_rows: u32) -> Vec<u32> {
    match rung {
        // Each row tile's first batch, the first that pads to a second
        // tile, and the last whose padding the capacity holds. A tile below
        // the largest pads to no other count (a larger tile takes the rows
        // between), the largest pads to every count between these, and the
        // column tile only widens with the count.
        Qmm::Tiled { .. } => BM_RUNGS
            .iter()
            .flat_map(|&bm| [bm, bm + 1, capacity_rows / bm * bm])
            .filter(|&m| m > 0 && m <= capacity_rows)
            .collect(),
        // The one-row rung reads one row; the others run every batch with
        // one kernel.
        Qmm::Coopmat | Qmm::RegisterTiled { .. } | Qmm::OneRow { .. } | Qmm::Qmv { .. } => {
            vec![1]
        }
    }
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
    target: Target,
    op: &'static str,
    b: &Bounds,
    bm: u32,
    bn: u32,
    padded: u32,
) -> Result<Fire, Error> {
    let (g, bits) = (b.group, b.bits);
    let (unit, name) = match target {
        Target::Spirv => (
            TILED,
            format!("linear_qmm_tiled_bf16_gs_{g}_b_{bits}_bm_{bm}_bn_{bn}"),
        ),
        Target::Wgsl => (
            QUAD,
            format!("linear_qmm_quad_bf16_gs_{g}_b_{bits}_bm_{bm}_bn_{bn}"),
        ),
    };
    Ok(Fire::at(unit, instantiated(op, name)?)
        .groups([b.n.div_ceil(bn), padded / bm, 1])
        .group([32, 2, if bm < 16 { 1 } else { 2 }]))
}

fn coopmat(op: &'static str, b: &Bounds) -> Result<Fire, Error> {
    let entry = instantiated(
        op,
        format!(
            "linear_qmm_tiled_bf16_gs_{}_b_{}_bm_{COOPMAT_TILE}_bn_{COOPMAT_TILE}",
            b.group, b.bits
        ),
    )?;
    Ok(Fire::at(TILED, entry)
        .tier(Tier::Coopmat)
        .groups([b.n.div_ceil(COOPMAT_TILE), b.m.div_ceil(COOPMAT_TILE), 1])
        .group([32, 4, 1]))
}

fn register(op: &'static str, b: &Bounds, bm: u32, k16: bool) -> Result<Fire, Error> {
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
    Ok(Fire::at(REGISTER, entry)
        .groups([b.n.div_ceil(REGISTER_COLS), b.m.div_ceil(bm), 1])
        .group([REGISTER_COLS / 4, bm / 4, 1]))
}

/// A `kernel.qmm` rung's kernel, whatever its parameters: what a fired
/// module key tells of the rung that ran it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Rung {
    Coopmat,
    RegisterTiled,
    Tiled,
    OneRow,
    Qmv,
}

/// The `kernel.qmm` rung a fired module key belongs to, for a tuner that
/// watches what a forced ladder ran.
#[must_use]
pub fn ran(key: &str) -> Option<Rung> {
    if key.starts_with("linear_qmm_tiled_") && key.ends_with(".coopmat") {
        Some(Rung::Coopmat)
    } else if key.starts_with("linear_qmm_register_") {
        Some(Rung::RegisterTiled)
    } else if key.starts_with("linear_qmm_tiled_") || key.starts_with("linear_qmm_quad_") {
        Some(Rung::Tiled)
    } else if key.starts_with("linear_qmv_row_") {
        Some(Rung::OneRow)
    } else if key.starts_with("linear_qmv_") {
        Some(Rung::Qmv)
    } else {
        None
    }
}

#[cfg(test)]
mod tests {
    use dtype::Dtype;

    use super::*;

    fn batch(n: u32, k: u32, group: u32, bits: u32) -> Bounds {
        let t = |rows, width, dtype| Tensor::new(0, rows, width, dtype);
        let bank = Bank {
            codes: t(n, k, Dtype::U8),
            scales: t(n, k / group, Dtype::Bf16),
            biases: Some(t(n, k / group, Dtype::Bf16)),
            group,
            bits,
        };
        bounds(
            "linear.matmul",
            t(1, k, Dtype::Bf16),
            bank,
            t(1, n, Dtype::Bf16),
        )
        .unwrap()
    }

    /// The kernels `ladder`'s rungs run batches of every row count up to
    /// `capacity` with, asked a row count at a time.
    fn asked(target: Target, ladder: &[Qmm], b: &Bounds, capacity: u32) -> Vec<Fire> {
        let mut all = Vec::new();
        for rung in ladder {
            for m in 1..=capacity {
                if matches!(rung, Qmm::OneRow { .. }) && m != 1 {
                    continue;
                }
                let at = Bounds { m, ..*b };
                if let Some(fire) = kernel(target, "linear.matmul", *rung, &at, capacity).unwrap() {
                    distinct(&mut all, fire);
                }
            }
        }
        all
    }

    fn names(mut fires: Vec<Fire>) -> Vec<(&'static str, Tier)> {
        fires.sort_by_key(|f| f.entrypoint);
        fires.iter().map(|f| (f.entrypoint, f.tier)).collect()
    }

    #[test]
    fn the_kernels_a_ladder_names_are_the_kernels_its_rows_pick() {
        let ladders: [(Target, Vec<Qmm>); 3] = [
            (
                Target::Spirv,
                vec![
                    Qmm::OneRow { lanes: 32, rows: 1 },
                    Qmm::Tiled { crossover_tg: 160 },
                    Qmm::Qmv { vecs: 4 },
                ],
            ),
            (
                Target::Spirv,
                vec![
                    Qmm::Coopmat,
                    Qmm::RegisterTiled { bm: 32, k16: true },
                    Qmm::Tiled { crossover_tg: 1 },
                    Qmm::Qmv { vecs: 2 },
                ],
            ),
            (
                Target::Wgsl,
                vec![Qmm::Tiled { crossover_tg: 24 }, Qmm::Qmv { vecs: 2 }],
            ),
        ];
        for (target, ladder) in &ladders {
            for n in [16, 48, 64, 1024, 1030] {
                let b = batch(n, 128, 64, 4);
                for capacity in (1..=70).chain([95, 96, 97, 127, 128, 129, 191, 192, 193, 300]) {
                    let named = kernels(*target, "linear.matmul", ladder, &b, capacity).unwrap();
                    assert_eq!(
                        names(named),
                        names(asked(*target, ladder, &b, capacity)),
                        "{target:?} {ladder:?}: {n} columns, capacity {capacity}"
                    );
                }
            }
        }
    }
}
