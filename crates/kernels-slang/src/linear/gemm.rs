//! Dense matmuls (`linear.matmul`, `linear.matmul` over a bf16 weight, and
//! an f32 activation's lane form). SPIR-V fires `kernels/linear/gemm.slang`
//! (the vector kernels where the rung's tactic takes them, else the tiled
//! kernel or its cooperative-matrix tier; `gemm_register.slang`'s where
//! the rows fill its 256-row tile without one) and `gemm_lane.slang` for f32
//! activations; WGSL fires `gemm_pairs.slang`, which lands column pairs as
//! words (its vector kernel at the subgroup tier where the device has one).

use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Target, Tier, dtype_dispatch, every, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;

const GEMM: &str = "linear/gemm.slang";

const PAIRS: &str = "linear/gemm_pairs.slang";

const LANE: &str = "linear/gemm_lane.slang";

/// The rows one workgroup of the eight-column vector kernel can carry over
/// one read of the weight (the shader's `PIE_VROWS`, one entry each): its
/// grid takes a row of workgroups a block of them.
pub const VECTOR_ROWS: [u32; 3] = [4, 8, 16];

/// The rows a vector workgroup carries for a matmul of `rows`: the fewest
/// of [`VECTOR_ROWS`] that hold them, the most past them all.
#[must_use]
pub fn vector_rows(rows: u32) -> u32 {
    VECTOR_ROWS
        .into_iter()
        .find(|&held| held >= rows)
        .unwrap_or(VECTOR_ROWS[VECTOR_ROWS.len() - 1])
}

const REGISTER: &str = "linear/gemm_register.slang";

/// The rows one register tile computes at either form (256 lanes over
/// `cols / per` columns, 8 rows a lane): a matmul takes it only where its
/// rows fill one, else SPIR-V's 32 x 32 tile serves them, so no fire pays a
/// tile its rows leave mostly idle.
const REGISTER_ROWS: u32 = 256;

/// The rows WGSL's wide tile computes: a matmul takes it only where its
/// rows fill one, else the 16-row tile serves them.
const PAIRS_WIDE_ROWS: u32 = 64;

/// How an engine runs dense matmuls on its device.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Tactic {
    /// The highest tier the family may use: SPIR-V tiles at
    /// `Tier::Coopmat`, WGSL's vector kernel folds by shuffles at
    /// `Tier::Subgroup`.
    pub tier: Tier,
    /// The vector kernels, a workgroup carrying this many activation rows
    /// (one of [`VECTOR_ROWS`]) over one read of the weight, at whatever
    /// rows the matmul lands; 0 for the tiled kernels. A load chooses it a
    /// rung, by the estimate (`engine_gpu::dense`).
    pub vector: u32,
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
/// The kernel a bf16 matmul of `m` rows, `n` columns and depth `k` runs
/// on `target` under `tactic`: the pick [`act_x_wt`]'s dispatch makes
/// (`kernel`, whose conditions [`tile_for`] repeats; a test holds the two
/// together), and [`padded_rows`] counts by.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Kernel {
    /// SPIR-V's vector kernel, eight columns a group (`k` a multiple of 8),
    /// `rows` activation rows a workgroup.
    GemvV { rows: u32 },
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
            Kernel::GemvV { .. } | Kernel::Gemv | Kernel::PairsGemv => 1,
            Kernel::Coopmat | Kernel::Pairs64 => 64,
            Kernel::Register { cols, per } => 256 / (cols / per) * 8,
            Kernel::Tiled => 32,
            Kernel::Pairs16 => 16,
        }
    }
    /// The rows that share one read of the weight: a workgroup's rows. A
    /// matmul of `m` rows reads its weight `ceil(m / rows_a_read)` times
    /// (the plain vector kernels once a row, the eight-column one once a
    /// workgroup's rows, a tiled kernel once a row tile).
    #[must_use]
    pub fn rows_a_read(self) -> u32 {
        match self {
            Kernel::Gemv | Kernel::PairsGemv => 1,
            Kernel::GemvV { rows } => rows,
            _ => self.row_tile(),
        }
    }
}

/// The kernel a matmul of `m` x `n` x `k` runs ([`Kernel`]).
#[must_use]
pub fn tile_for(target: Target, tactic: Tactic, m: u32, n: u32, k: u32) -> Kernel {
    match target {
        Target::Spirv if tactic.vector > 0 && k.is_multiple_of(8) => Kernel::GemvV {
            rows: vector_rows(tactic.vector),
        },
        Target::Spirv if tactic.vector > 0 => Kernel::Gemv,
        Target::Spirv if tactic.tier == Tier::Coopmat => Kernel::Coopmat,
        // Sixty-four columns a tile at eight a lane, or thirty-two at four
        // when that leaves none idle.
        Target::Spirv if m >= REGISTER_ROWS && k.is_multiple_of(16) => {
            if n.is_multiple_of(32) && !n.is_multiple_of(64) {
                Kernel::Register { cols: 32, per: 4 }
            } else {
                Kernel::Register { cols: 64, per: 8 }
            }
        }
        Target::Spirv => Kernel::Tiled,
        Target::Wgsl if tactic.vector > 0 => Kernel::PairsGemv,
        Target::Wgsl if m < PAIRS_WIDE_ROWS => Kernel::Pairs16,
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
    let target = ctx.target();
    if let Some(names) = ctx.naming() {
        return names.either(&kernels(target, tactic, op, n, k)?);
    }
    if m == 0 {
        return Ok(());
    }
    ctx.fire(
        kernel(target, tactic, op, m, n, k)?,
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

/// The rows at which the kernel a tiled projection takes changes:
/// [`kernel`]'s own thresholds, so its answers at these are every answer it
/// has.
const ROW_BREAKS: [u32; 3] = [1, PAIRS_WIDE_ROWS, REGISTER_ROWS];

/// Every kernel a projection of `n` columns over a contraction of `k`
/// takes at some row count under `tactic`'s tier, vector or tiled: which
/// of the two a rung takes is the load's choice, made after its kernels
/// are named.
pub fn kernels(
    target: Target,
    tactic: Tactic,
    op: &'static str,
    n: u32,
    k: u32,
) -> Result<Vec<Fire>, Error> {
    let tiled = Tactic {
        vector: 0,
        ..tactic
    };
    let mut all = every(&ROW_BREAKS, |m| kernel(target, tiled, op, m, n, k))?;
    for rows in VECTOR_ROWS {
        let vector = Tactic {
            vector: rows,
            ..tactic
        };
        crate::encode::distinct(&mut all, kernel(target, vector, op, 1, n, k)?);
    }
    Ok(all)
}

/// The kernel a projection of `m` rows takes, at its grid.
fn kernel(
    target: Target,
    tactic: Tactic,
    op: &'static str,
    m: u32,
    n: u32,
    k: u32,
) -> Result<Fire, Error> {
    Ok(match target {
        Target::Spirv if tactic.vector > 0 && k.is_multiple_of(8) => {
            let rows = vector_rows(tactic.vector);
            let entry = match rows {
                4 => "linear_gemm_gemv_v_bf16",
                8 => "linear_gemm_gemv_v8_bf16",
                _ => "linear_gemm_gemv_v16_bf16",
            };
            Fire::at(GEMM, entry)
                .groups([n.div_ceil(8), m.div_ceil(rows), 1])
                .group([32, 8, 1])
        }
        Target::Spirv if tactic.vector > 0 => Fire::at(GEMM, "linear_gemm_gemv_bf16")
            .groups([1, n.div_ceil(8), m])
            .group([32, 8, 1]),
        Target::Spirv if tactic.tier == Tier::Coopmat => {
            Fire::at(GEMM, "linear_gemm_tiled_bm_64_bf16")
                .tier(Tier::Coopmat)
                .groups([n.div_ceil(64), m.div_ceil(64), 1])
                .group([32, 4, 1])
        }
        // Sixty-four columns a tile at eight a lane, or thirty-two at four
        // when that leaves none idle.
        Target::Spirv if m >= REGISTER_ROWS && k.is_multiple_of(16) => {
            let (entry, cols, per) = if n.is_multiple_of(32) && !n.is_multiple_of(64) {
                ("linear_gemm_register_bn_32_bf16", 32, 4)
            } else {
                ("linear_gemm_register_bf16", 64, 8)
            };
            let (tx, ty) = (cols / per, 256 / (cols / per));
            let tiles = m.div_ceil(ty * 8);
            if tiles > 65_535 {
                return Err(refuse(op, format!("the grid will not launch: {m} rows")));
            }
            // A workgroup a tile: the shader takes its tile from
            // `SV_GroupID`.
            Fire::at(REGISTER, entry)
                .groups([n.div_ceil(cols), tiles, 1])
                .group([tx, ty, 1])
        }
        Target::Spirv => Fire::at(GEMM, "linear_gemm_tiled_bf16")
            .groups([n.div_ceil(32), m.div_ceil(32), 1])
            .group([32, 2, 2]),
        Target::Wgsl => {
            if !n.is_multiple_of(2) {
                return Err(refuse(
                    op,
                    format!(
                        "the projection lands {n} columns, odd: on WGSL every output pair is one \
                         bf16 word"
                    ),
                ));
            }
            if tactic.vector > 0 {
                Fire::at(PAIRS, "linear_gemm_pairs_gemv_bf16")
                    .tier(if tactic.tier == Tier::Subgroup {
                        Tier::Subgroup
                    } else {
                        Tier::Baseline
                    })
                    .groups([1, n.div_ceil(8), m])
                    .group([32, 8, 1])
            } else if m < PAIRS_WIDE_ROWS {
                Fire::at(PAIRS, "linear_gemm_pairs_tiled_bm_16_bf16")
                    .groups([n.div_ceil(64), m.div_ceil(16), 1])
                    .group([16, 16, 1])
            } else {
                Fire::at(PAIRS, "linear_gemm_pairs_tiled_bf16")
                    .groups([n.div_ceil(64), m.div_ceil(64), 1])
                    .group([16, 16, 1])
            }
        }
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

    /// A projection's naming is every kernel its rows pick: the kernel of
    /// each row count is one of [`kernels`]'.
    #[test]
    fn the_kernels_a_projection_names_hold_the_kernel_of_every_row_count() {
        for target in [Target::Spirv, Target::Wgsl] {
            for tier in [Tier::Baseline, Tier::Subgroup, Tier::Coopmat] {
                for vector in [0, 4, 8, 16] {
                    let tactic = Tactic { tier, vector };
                    for (n, k) in [(64, 64), (1024, 1024), (1030, 1001), (48, 16), (8, 24)] {
                        let named: Vec<(&str, Tier)> =
                            kernels(target, tactic, "linear.matmul", n, k)
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
    }
}

#[cfg(test)]
mod tile_tests {
    use super::*;

    #[test]
    fn the_row_tile_is_the_tile_the_dispatch_picks() {
        // `kernel`'s grid rows a group, at every row break and width, are
        // `tile_for`'s row tile.
        for target in [Target::Spirv, Target::Wgsl] {
            for tier in [Tier::Baseline, Tier::Subgroup, Tier::Coopmat] {
                for vector in [0, 4, 8, 16] {
                    let tactic = Tactic { tier, vector };
                    for m in [1, 4, 8, 16, 47, 48, 64, 255, 256, 1024, 4096] {
                        for (n, k) in [(1024, 1024), (96, 4096), (4096, 1000)] {
                            let Ok(fire) = kernel(target, tactic, "test", m, n, k) else {
                                continue;
                            };
                            let tile = tile_for(target, tactic, m, n, k).row_tile();
                            let groups = fire.groups;
                            let rows = match tile {
                                1 => continue,
                                _ => m.div_ceil(tile),
                            };
                            assert!(
                                groups.contains(&rows),
                                "{target:?} {tier:?} {m}x{n}x{k}: groups {groups:?}, tile {tile}"
                            );
                        }
                    }
                }
            }
        }
    }

    /// A tile wider than the smallest of its family is taken only where the
    /// rows fill one: no row count pays a tile it leaves mostly idle, and
    /// the rows a matmul computes never fall as its rows grow.
    #[test]
    fn a_wide_tile_is_taken_only_where_the_rows_fill_it() {
        for target in [Target::Spirv, Target::Wgsl] {
            for tier in [Tier::Baseline, Tier::Coopmat] {
                let tactic = Tactic { tier, vector: 0 };
                for (n, k) in [(1024, 1024), (96, 4096), (4096, 1000)] {
                    let mut computed = 0;
                    for m in 1..=4096 {
                        let tile = tile_for(target, tactic, m, n, k);
                        if matches!(tile, Kernel::Register { .. } | Kernel::Pairs64) {
                            assert!(m >= tile.row_tile(), "{target:?} {m} rows take {tile:?}");
                        }
                        let padded = padded_rows(target, tactic, m, n);
                        assert!(
                            padded >= computed,
                            "{target:?} {tier:?}: {m} rows compute {padded}, fewer than {computed}"
                        );
                        computed = padded;
                    }
                }
            }
        }
    }
}
