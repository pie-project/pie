//! Dense matmuls (`linear.matmul`, `linear.matmul` over a bf16 weight, and
//! an f32 activation's lane form). SPIR-V fires `kernels/linear/gemm.slang`
//! (the vector kernels below four rows, the tiled kernel or its
//! cooperative-matrix tier above; `gemm_register.slang`'s from 96 rows
//! without one) and `gemm_lane.slang` for f32
//! activations; WGSL fires `gemm_pairs.slang`, which lands column pairs as
//! words (its vector kernel at the subgroup tier where the device has one).

use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Target, Tier, dtype_dispatch, every, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;

const GEMM: &str = "linear/gemm.slang";

const PAIRS: &str = "linear/gemm_pairs.slang";

const LANE: &str = "linear/gemm_lane.slang";

/// Rows below this take the vector kernels.
const VECTOR_MAX_ROWS: u32 = 4;

const REGISTER: &str = "linear/gemm_register.slang";

/// Fewer rows leave most of a register tile idle; SPIR-V's 32 x 32 tile
/// serves them.
const RT_MIN_ROWS: u32 = 96;

/// Rows below this take WGSL's 16-row tile.
const NARROW_MAX_ROWS: u32 = 48;

/// How an engine runs dense matmuls on its device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tactic {
    /// The highest tier the family may use: SPIR-V tiles at
    /// `Tier::Coopmat`, WGSL's vector kernel folds by shuffles at
    /// `Tier::Subgroup`.
    pub tier: Tier,
}

/// The kernel a bf16 matmul of `m` rows, `n` columns and depth `k` runs
/// on `target` under `tactic`: the one pick [`act_x_wt`] dispatches by and
/// [`padded_rows`] counts by, so the two cannot drift.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// SPIR-V's vector kernel, eight columns a group (`k` a multiple of 8).
    GemvV,
    /// SPIR-V's vector kernel.
    Gemv,
    /// SPIR-V's cooperative-matrix tile, 64 rows.
    Coopmat,
    /// SPIR-V's register tile: `cols` columns at `per` a lane, 256 lanes'
    /// rows of 8 (`256 / (cols / per) · 8` rows a tile).
    Register { cols: u32, per: u32 },
    /// SPIR-V's 32 x 32 tile.
    Tiled,
    /// WGSL's vector kernel.
    PairsGemv,
    /// WGSL's 16-row tile.
    Pairs16,
    /// WGSL's 64-row tile.
    Pairs64,
}

impl Kernel {
    /// The rows one tile of it computes: what a matmul's rows pad to.
    #[must_use]
    pub fn row_tile(self) -> u32 {
        match self {
            Kernel::GemvV | Kernel::Gemv | Kernel::PairsGemv => 1,
            Kernel::Coopmat | Kernel::Pairs64 => 64,
            Kernel::Register { cols, per } => 256 / (cols / per) * 8,
            Kernel::Tiled => 32,
            Kernel::Pairs16 => 16,
        }
    }
}

/// The kernel a matmul of `m` x `n` x `k` runs ([`Kernel`]).
#[must_use]
pub fn tile_for(target: Target, tactic: Tactic, m: u32, n: u32, k: u32) -> Kernel {
    match target {
        Target::Spirv if m < VECTOR_MAX_ROWS && k.is_multiple_of(8) => Kernel::GemvV,
        Target::Spirv if m < VECTOR_MAX_ROWS => Kernel::Gemv,
        Target::Spirv if tactic.tier == Tier::Coopmat => Kernel::Coopmat,
        // Sixty-four columns a tile at eight a lane, or thirty-two at four
        // when that leaves none idle.
        Target::Spirv if m >= RT_MIN_ROWS && k.is_multiple_of(16) => {
            if n.is_multiple_of(32) && !n.is_multiple_of(64) {
                Kernel::Register { cols: 32, per: 4 }
            } else {
                Kernel::Register { cols: 64, per: 8 }
            }
        }
        Target::Spirv => Kernel::Tiled,
        Target::Wgsl if m < VECTOR_MAX_ROWS => Kernel::PairsGemv,
        Target::Wgsl if m < NARROW_MAX_ROWS => Kernel::Pairs16,
        Target::Wgsl => Kernel::Pairs64,
    }
}

/// The rows a matmul of `rows` over a `width`-wide weight computes on
/// `target` under `tactic`: `rows` padded to the row tile of the kernel
/// [`tile_for`] picks. What a fire's estimate counts a narrow fire's
/// matmuls by, as it pays for the tile it under-fills.
#[must_use]
pub fn padded_rows(target: Target, tactic: Tactic, rows: u32, width: u32) -> u32 {
    let tile = tile_for(target, tactic, rows, width, width).row_tile();
    rows.div_ceil(tile).saturating_mul(tile)
}

pub fn matmul(
    ctx: &Ctx<'_>,
    tactic: Tactic,
    act: Tensor,
    w: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    act_x_wt(ctx, tactic, "linear.matmul", act, w, y)
}

pub fn lm_head(
    ctx: &Ctx<'_>,
    tactic: Tactic,
    act: Tensor,
    w: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    act_x_wt(ctx, tactic, "linear.matmul", act, w, y)
}

/// `y = act * w^T` on bf16.
pub fn act_x_wt(
    ctx: &Ctx<'_>,
    tactic: Tactic,
    op: &'static str,
    act: Tensor,
    w: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    let (m, n, k) = extent(op, act, w, y)?;
    fire_columns(ctx, tactic, op, act, w, y, (m, n, k), 0)
}

/// `y[:, y0 .. y0 + w.rows] = act * w^T` on bf16: one piece of a weight
/// laid in pieces along its rows (the result's columns) landing its own
/// columns of the whole result, so no piece sums into another's. The kernel
/// is the one the whole weight's matmul takes at these rows (picked by the
/// whole result's width), so a piece's columns are the whole's bits.
pub fn matmul_slice(
    ctx: &Ctx<'_>,
    tactic: Tactic,
    act: Tensor,
    w: Tensor,
    y: Tensor,
    y0: u32,
) -> Result<(), Error> {
    const OP: &str = "linear.matmul";
    let extent = extent_slice(OP, act, w, y, y0)?;
    if ctx.target() == Target::Wgsl && !(y0.is_multiple_of(2) && y.width.is_multiple_of(2)) {
        return Err(refuse(
            OP,
            format!(
                "a piece lands from column {y0} of a {}-wide result: on WGSL every output \
                 pair is one bf16 word",
                y.width
            ),
        ));
    }
    fire_columns(ctx, tactic, OP, act, w, y, extent, y0)
}

/// Fires `act * w^T` into `n` columns of `y` from `y0`, by the kernel the
/// whole result's width picks.
#[allow(clippy::too_many_arguments)]
fn fire_columns(
    ctx: &Ctx<'_>,
    tactic: Tactic,
    op: &'static str,
    act: Tensor,
    w: Tensor,
    y: Tensor,
    (m, n, k): (u32, u32, u32),
    y0: u32,
) -> Result<(), Error> {
    dtype_dispatch!(op, act.dtype, { Bf16 => () });
    let target = ctx.target();
    let width = y.width;
    let tactic = fitting(ctx, target, tactic, op, width, k)?;
    if let Some(names) = ctx.naming() {
        return names.either(&kernels(target, tactic, op, width, k)?);
    }
    if m == 0 {
        return Ok(());
    }
    ctx.fire(
        picked(target, tactic, op, m, n, k, width)?,
        &[
            act.arg(),
            w.arg(),
            y.arg_mut(),
            stated(op, m)?.arg(),
            stated(op, n)?.arg(),
            stated(op, k)?.arg(),
            stated(op, width)?.arg(),
            stated(op, y0)?.arg(),
        ],
    )
}

/// `tactic`, or the baseline tier where a kernel the projection takes at
/// its tier declares more workgroup memory than `ctx`'s device gives one
/// workgroup: the cooperative-matrix tile is passed over for the tiles
/// below it.
fn fitting(
    ctx: &Ctx<'_>,
    target: Target,
    tactic: Tactic,
    op: &'static str,
    n: u32,
    k: u32,
) -> Result<Tactic, Error> {
    if tactic.tier == Tier::Baseline
        || kernels(target, tactic, op, n, k)?
            .iter()
            .all(|fire| crate::workgroup::fits(ctx, fire))
    {
        return Ok(tactic);
    }
    Ok(Tactic {
        tier: Tier::Baseline,
    })
}

/// The rows at which the kernel a projection takes changes: [`tile_for`]'s
/// own thresholds, so its answers at these are every answer it has (a test
/// holds every row count's pick to them).
const ROW_BREAKS: [u32; 4] = [1, VECTOR_MAX_ROWS, NARROW_MAX_ROWS, RT_MIN_ROWS];

/// Every kernel a projection of `n` columns over a contraction of `k`
/// takes at some row count.
pub fn kernels(
    target: Target,
    tactic: Tactic,
    op: &'static str,
    n: u32,
    k: u32,
) -> Result<Vec<Fire>, Error> {
    every(&ROW_BREAKS, |m| kernel(target, tactic, op, m, n, k))
}

/// The kernel a projection of `m` rows takes, at its grid: [`tile_for`]'s
/// pick, the one [`kernels`] and [`padded_rows`] read too.
fn kernel(
    target: Target,
    tactic: Tactic,
    op: &'static str,
    m: u32,
    n: u32,
    k: u32,
) -> Result<Fire, Error> {
    picked(target, tactic, op, m, n, k, n)
}

/// [`kernel`] picked at a result `width` wide, its grid over the `n`
/// columns it lands (`n` = `width` but for a piece of a weight).
fn picked(
    target: Target,
    tactic: Tactic,
    op: &'static str,
    m: u32,
    n: u32,
    k: u32,
    width: u32,
) -> Result<Fire, Error> {
    if target == Target::Wgsl && !(n.is_multiple_of(2) && width.is_multiple_of(2)) {
        return Err(refuse(
            op,
            format!(
                "the projection lands {n} columns, odd: on WGSL every output pair is one \
                 bf16 word"
            ),
        ));
    }
    Ok(match tile_for(target, tactic, m, width, k) {
        Kernel::GemvV => Fire::at(GEMM, "linear_gemm_gemv_v_bf16")
            .groups([n.div_ceil(8), 1, 1])
            .group([32, 8, 1]),
        Kernel::Gemv => Fire::at(GEMM, "linear_gemm_gemv_bf16")
            .groups([1, n.div_ceil(8), m])
            .group([32, 8, 1]),
        Kernel::Coopmat => Fire::at(GEMM, "linear_gemm_tiled_bm_64_bf16")
            .tier(Tier::Coopmat)
            .groups([n.div_ceil(64), m.div_ceil(64), 1])
            .group([32, 4, 1]),
        kernel @ Kernel::Register { cols, per } => {
            let entry = if cols == 32 {
                "linear_gemm_register_bn_32_bf16"
            } else {
                "linear_gemm_register_bf16"
            };
            let (tx, ty) = (cols / per, 256 / (cols / per));
            let tiles = m.div_ceil(kernel.row_tile());
            if tiles > 65_535 {
                return Err(refuse(op, format!("the grid will not launch: {m} rows")));
            }
            // A workgroup a tile: the shader takes its tile from
            // `SV_GroupID`.
            Fire::at(REGISTER, entry)
                .groups([n.div_ceil(cols), tiles, 1])
                .group([tx, ty, 1])
        }
        Kernel::Tiled => Fire::at(GEMM, "linear_gemm_tiled_bf16")
            .groups([n.div_ceil(32), m.div_ceil(32), 1])
            .group([32, 2, 2]),
        Kernel::PairsGemv => Fire::at(PAIRS, "linear_gemm_pairs_gemv_bf16")
            .tier(if tactic.tier == Tier::Subgroup {
                Tier::Subgroup
            } else {
                Tier::Baseline
            })
            .groups([1, n.div_ceil(8), m])
            .group([32, 8, 1]),
        Kernel::Pairs16 => Fire::at(PAIRS, "linear_gemm_pairs_tiled_bm_16_bf16")
            .groups([n.div_ceil(64), m.div_ceil(16), 1])
            .group([16, 16, 1]),
        Kernel::Pairs64 => Fire::at(PAIRS, "linear_gemm_pairs_tiled_bf16")
            .groups([n.div_ceil(64), m.div_ceil(64), 1])
            .group([16, 16, 1]),
    })
}

/// `y = act * w^T` for an f32 activation, an invocation an output:
/// a bf16 or f32 weight, an f32 or bf16 result.
pub fn lane(
    ctx: &Ctx<'_>,
    op: &'static str,
    act: Tensor,
    w: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    let extent = extent(op, act, w, y)?;
    lane_columns(ctx, op, act, w, y, extent, 0)
}

/// [`lane`] for one piece of a weight laid in pieces along its rows: its
/// columns of the whole result, from `y0` ([`matmul_slice`]).
pub fn lane_slice(
    ctx: &Ctx<'_>,
    op: &'static str,
    act: Tensor,
    w: Tensor,
    y: Tensor,
    y0: u32,
) -> Result<(), Error> {
    let extent = extent_slice(op, act, w, y, y0)?;
    lane_columns(ctx, op, act, w, y, extent, y0)
}

fn lane_columns(
    ctx: &Ctx<'_>,
    op: &'static str,
    act: Tensor,
    w: Tensor,
    y: Tensor,
    (m, n, k): (u32, u32, u32),
    y0: u32,
) -> Result<(), Error> {
    if m == 0 {
        return Ok(());
    }
    let entry = match (act.dtype, w.dtype, y.dtype) {
        (Dtype::F32, Dtype::Bf16, Dtype::F32) => "linear_gemm_lane_f32_bf16_f32",
        (Dtype::F32, Dtype::F32, Dtype::F32) => "linear_gemm_lane_f32_f32_f32",
        (Dtype::F32, Dtype::Bf16, Dtype::Bf16) => "linear_gemm_lane_f32_bf16_bf16",
        (Dtype::F32, Dtype::F32, Dtype::Bf16) => "linear_gemm_lane_f32_f32_bf16",
        (a, weight, out) => {
            return Err(refuse(
                op,
                format!(
                    "no lane point for a {a:?} activation against a {weight:?} weight into {out:?}"
                ),
            ));
        }
    };
    ctx.fire(
        Fire::at(LANE, entry)
            .groups([n.div_ceil(64), m.div_ceil(4), 1])
            .group([64, 4, 1]),
        &[
            act.arg(),
            w.arg(),
            y.arg_mut(),
            stated(op, m)?.arg(),
            stated(op, n)?.arg(),
            stated(op, k)?.arg(),
            stated(op, y.width)?.arg(),
            stated(op, y0)?.arg(),
        ],
    )
}

/// Rows, a piece's output columns and contraction: the piece's rows land
/// columns `y0..` of the result, within it.
pub(crate) fn extent_slice(
    op: &'static str,
    act: Tensor,
    w: Tensor,
    y: Tensor,
    y0: u32,
) -> Result<(u32, u32, u32), Error> {
    if act.width == 0 || w.rows == 0 {
        return Err(refuse(op, "a piece of this projection is empty"));
    }
    if w.width != act.width || y0.checked_add(w.rows).is_none_or(|end| end > y.width) {
        return Err(refuse(
            op,
            format!(
                "a {} x {} piece landing from column {y0} does not lie in a projection {} in, \
                 {} out",
                w.rows, w.width, act.width, y.width
            ),
        ));
    }
    Ok((y.rows, w.rows, act.width))
}

/// Rows, output columns and contraction, the weight checked against them.
pub(crate) fn extent(
    op: &'static str,
    act: Tensor,
    w: Tensor,
    y: Tensor,
) -> Result<(u32, u32, u32), Error> {
    if y.width == 0 {
        return Err(refuse(op, "the columns this projection lands are zero"));
    }
    if act.width == 0 {
        return Err(refuse(op, "the contraction this projection walks is zero"));
    }
    if w.width != act.width || w.rows != y.width {
        return Err(refuse(
            op,
            format!(
                "the weight is {} x {} and the projection is {} in, {} out",
                w.rows, w.width, act.width, y.width
            ),
        ));
    }
    debug_assert_eq!(
        act.rows, y.rows,
        "the activation's rows are the rows the result lands"
    );
    Ok((y.rows, y.width, act.width))
}

#[cfg(test)]
mod tests {
    use super::*;

    /// A projection's naming is every kernel its rows pick: the kernel of
    /// each row count is one of [`kernels`]'.
    #[test]
    fn the_kernels_a_projection_names_hold_the_kernel_of_every_row_count() {
        for target in [Target::Spirv, Target::Wgsl] {
            for tier in [Tier::Baseline, Tier::Subgroup, Tier::Coopmat] {
                let tactic = Tactic { tier };
                for (n, k) in [(64, 64), (1024, 1024), (1030, 1001), (48, 16), (8, 24)] {
                    let named: Vec<(&str, Tier)> = kernels(target, tactic, "linear.matmul", n, k)
                        .unwrap()
                        .iter()
                        .map(|fire| (fire.entrypoint, fire.tier))
                        .collect();
                    for m in 1..=300 {
                        let picked = kernel(target, tactic, "linear.matmul", m, n, k).unwrap();
                        assert!(
                            named.contains(&(picked.entrypoint, picked.tier)),
                            "{target:?} {tier:?}: {m} rows of {n} columns over {k} pick {}, \
                             which the projection does not name",
                            picked.entrypoint
                        );
                    }
                }
            }
        }
    }

    #[test]
    fn a_matmul_pads_its_rows_to_the_tile_of_the_kernel_it_runs() {
        let (spirv, wgsl) = (Target::Spirv, Target::Wgsl);
        let base = Tactic {
            tier: Tier::Baseline,
        };
        let coop = Tactic {
            tier: Tier::Coopmat,
        };
        // A vector kernel pads nothing.
        assert_eq!(padded_rows(spirv, coop, 3, 4096), 3);
        assert_eq!(padded_rows(spirv, coop, 5, 4096), 64);
        assert_eq!(tile_for(spirv, base, 40, 4096, 4096), Kernel::Tiled);
        assert_eq!(padded_rows(spirv, base, 40, 4096), 64);
        assert_eq!(padded_rows(spirv, base, 100, 4096), 256);
        assert_eq!(padded_rows(wgsl, base, 20, 4096), 32);
        assert_eq!(padded_rows(wgsl, base, 100, 4096), 128);
    }

    /// The pick changes only at a row break: between two breaks every row
    /// count picks what the lower break does.
    #[test]
    fn the_pick_changes_only_at_a_row_break() {
        for target in [Target::Spirv, Target::Wgsl] {
            for tier in [Tier::Baseline, Tier::Subgroup, Tier::Coopmat] {
                let tactic = Tactic { tier };
                for (n, k) in [(64, 64), (1024, 1024), (1030, 1001), (48, 16)] {
                    for m in 2..=1024u32 {
                        let (was, is) = (
                            tile_for(target, tactic, m - 1, n, k),
                            tile_for(target, tactic, m, n, k),
                        );
                        assert!(
                            was == is || ROW_BREAKS.contains(&m),
                            "{target:?} {tier:?} {n}x{k}: the pick changes at {m} rows, no break"
                        );
                    }
                }
            }
        }
    }
}
