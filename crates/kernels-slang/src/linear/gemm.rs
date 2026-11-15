//! Dense matmuls (`linear.matmul`, `linear.matmul` over a bf16 weight, and
//! an f32 activation's lane form). SPIR-V fires `kernels/linear/gemm.slang`
//! (the vector kernels below four rows, the tiled kernel or its
//! cooperative-matrix tier above; `gemm_register.slang`'s from 96 rows
//! without one) and `gemm_lane.slang` for f32
//! activations; WGSL fires `gemm_pairs.slang`, which lands column pairs as
//! words (its vector kernel at the subgroup tier where the device has one).

use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Target, Tier, dtype_dispatch, refuse, stated};
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
    dtype_dispatch!(op, act.dtype, { Bf16 => () });
    let (m, n, k) = extent(op, act, w, y)?;
    if m == 0 {
        return Ok(());
    }
    let target = ctx.target();
    if target == Target::Wgsl && !n.is_multiple_of(2) {
        return Err(refuse(
            op,
            format!(
                "the projection lands {n} columns, odd: on WGSL every output pair is one \
                 bf16 word"
            ),
        ));
    }
    let fire = match tile_for(target, tactic, m, n, k) {
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
            Fire::at(REGISTER, entry)
                .groups([n.div_ceil(cols) * tx, tiles * ty, 1])
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
    };
    ctx.fire(
        fire,
        &[
            act.arg(),
            w.arg(),
            y.arg_mut(),
            stated(op, m)?.arg(),
            stated(op, n)?.arg(),
            stated(op, k)?.arg(),
        ],
    )
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
    let (m, n, k) = extent(op, act, w, y)?;
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
        ],
    )
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

    fn tactic(tier: Tier) -> Tactic {
        Tactic { tier }
    }

    #[test]
    fn a_matmul_pads_its_rows_to_the_tile_of_the_kernel_it_runs() {
        let (spirv, wgsl) = (Target::Spirv, Target::Wgsl);
        let base = tactic(Tier::Baseline);
        let coop = tactic(Tier::Coopmat);
        // A vector kernel pads nothing.
        assert_eq!(padded_rows(spirv, coop, 3, 4096), 3);
        assert_eq!(padded_rows(spirv, coop, 5, 4096), 64);
        assert_eq!(tile_for(spirv, base, 40, 4096, 4096), Kernel::Tiled);
        assert_eq!(padded_rows(spirv, base, 40, 4096), 64);
        assert_eq!(padded_rows(spirv, base, 100, 4096), 256);
        assert_eq!(padded_rows(wgsl, base, 20, 4096), 32);
        assert_eq!(padded_rows(wgsl, base, 100, 4096), 128);
    }
}
