//! Dense matmuls (`linear.matmul`, `linear.matmul` over a bf16 weight, and
//! an f32 activation's lane form). SPIR-V fires `kernels/linear/gemm.slang`
//! (the vector kernels where the rung's tactic takes them, else the tiled
//! kernel or its cooperative-matrix tier; `gemm_register.slang`'s where the
//! tactic takes the wide tile without one) and `gemm_lane.slang` for f32
//! activations; WGSL fires `gemm_pairs.slang`, which lands column pairs as
//! words (its vector kernel at the subgroup tier where the device has one).

use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Target, Tier, dtype_dispatch, refuse, stated};
use crate::error::Error;
use crate::ssm::RowSpan;
use crate::tensor::Tensor;

const GEMM: &str = "linear/gemm.slang";

const PAIRS: &str = "linear/gemm_pairs.slang";

const LANE: &str = "linear/gemm_lane.slang";

/// The rows one workgroup of the eight-column vector kernel can carry over
/// one read of the weight (the shader's `PIE_VROWS`, one entry each): its
/// grid takes a row of workgroups a block of them.
pub const VECTOR_ROWS: [u32; 4] = [1, 4, 8, 16];

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
    /// Of the tiled kernels, the wide tile (SPIR-V's register tile, where
    /// the contraction is whole 16-element steps; WGSL's 64-row tile), else
    /// the narrow one (SPIR-V's 32 x 32, WGSL's 16-row). No choice under
    /// SPIR-V's cooperative-matrix tier. A load chooses it a rung, by the
    /// estimate over each kernel's measured rate (`engine_gpu::dense`).
    pub wide: bool,
}

pub fn matmul(
    ctx: &Ctx<'_>,
    tactic: Tactic,
    act: Tensor,
    w: Tensor,
    y: Tensor,
    window: RowSpan,
) -> Result<(), Error> {
    act_x_wt(ctx, tactic, "linear.matmul", act, w, y, window)
}

pub fn lm_head(
    ctx: &Ctx<'_>,
    tactic: Tactic,
    act: Tensor,
    w: Tensor,
    y: Tensor,
    window: RowSpan,
) -> Result<(), Error> {
    act_x_wt(ctx, tactic, "linear.matmul", act, w, y, window)
}

/// The columns a bf16 projection of `n` columns lands on `target`: its own,
/// except a one-column projection on WGSL, whose kernels land column pairs
/// as words. That one is laid two wide: its weight a zero row before its
/// own, so the column lands as the high half of each row's word, which read
/// as an f32 is the column's value exactly (`bf16.slang`'s `bf16_to_f32`).
/// A wider odd projection lands nowhere on WGSL ([`matmul`] refuses it).
#[must_use]
pub fn laid_columns(target: Target, n: u32) -> u32 {
    if target == Target::Wgsl && n == 1 {
        2
    } else {
        n
    }
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
    /// WGSL's vector kernel, `rows` activation rows a workgroup.
    PairsGemv { rows: u32 },
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
            Kernel::GemvV { .. } | Kernel::Gemv | Kernel::PairsGemv { .. } => 1,
            Kernel::Coopmat | Kernel::Pairs64 => 64,
            Kernel::Register { cols, per } => 256 / (cols / per) * 8,
            Kernel::Tiled => 32,
            Kernel::Pairs16 => 16,
        }
    }
    /// The columns one workgroup of it computes: a matmul over a slice of
    /// a plane's columns that is a whole number of these runs the plane's
    /// own tiles.
    #[must_use]
    pub fn col_tile(self) -> u32 {
        match self {
            Kernel::GemvV { .. } | Kernel::Gemv | Kernel::PairsGemv { .. } => 8,
            Kernel::Coopmat | Kernel::Pairs16 | Kernel::Pairs64 => 64,
            Kernel::Register { cols, .. } => cols,
            Kernel::Tiled => 32,
        }
    }
    /// The rows that share one read of the weight: a workgroup's rows. A
    /// matmul of `m` rows reads its weight `ceil(m / rows_a_read)` times
    /// (the plain vector kernels once a row, the eight-column one once a
    /// workgroup's rows, a tiled kernel once a row tile).
    #[must_use]
    pub fn rows_a_read(self) -> u32 {
        match self {
            Kernel::Gemv => 1,
            Kernel::GemvV { rows } | Kernel::PairsGemv { rows } => rows,
            _ => self.row_tile(),
        }
    }
}

/// The kernel a matmul of `n` columns over `k` runs under `tactic`, at any
/// rows ([`Kernel`]).
#[must_use]
pub fn tile_for(target: Target, tactic: Tactic, n: u32, k: u32) -> Kernel {
    match target {
        Target::Spirv if tactic.vector > 0 && k.is_multiple_of(8) => Kernel::GemvV {
            rows: vector_rows(tactic.vector),
        },
        Target::Spirv if tactic.vector > 0 => Kernel::Gemv,
        Target::Spirv if tactic.tier == Tier::Coopmat => Kernel::Coopmat,
        // Sixty-four columns a tile at eight a lane, or thirty-two at four
        // when that leaves none idle.
        Target::Spirv if tactic.wide && k.is_multiple_of(16) => {
            if n.is_multiple_of(32) && !n.is_multiple_of(64) {
                Kernel::Register { cols: 32, per: 4 }
            } else {
                Kernel::Register { cols: 64, per: 8 }
            }
        }
        Target::Spirv => Kernel::Tiled,
        Target::Wgsl if tactic.vector > 0 => Kernel::PairsGemv {
            rows: vector_rows(tactic.vector),
        },
        Target::Wgsl if !tactic.wide => Kernel::Pairs16,
        Target::Wgsl => Kernel::Pairs64,
    }
}

/// The rows a matmul of `rows` over a `width`-wide weight computes on
/// `target` under `tactic`: `rows` padded to the row tile of the kernel
/// [`tile_for`] picks. What a fire's estimate counts a narrow fire's
/// matmuls by, as it pays for the tile it under-fills.
#[must_use]
pub fn padded_rows(target: Target, tactic: Tactic, rows: u32, width: u32) -> u32 {
    let tile = tile_for(target, tactic, width, width).row_tile();
    rows.div_ceil(tile).saturating_mul(tile)
}

pub fn act_x_wt(
    ctx: &Ctx<'_>,
    tactic: Tactic,
    op: &'static str,
    act: Tensor,
    w: Tensor,
    y: Tensor,
    window: RowSpan,
) -> Result<(), Error> {
    dtype_dispatch!(op, act.dtype, { Bf16 => () });
    let (_, n, k) = extent(op, act, w, y)?;
    let target = ctx.target();
    if let Some(names) = ctx.naming() {
        return names.either(&kernels(target, tactic, op, n, k)?);
    }
    // The rows are bound whole and the window's first row is the fire's
    // data (each kernel's `row0`): one body serves every window of a fire.
    let m = window.rows.min(y.rows.saturating_sub(window.row0));
    if m == 0 {
        return Ok(());
    }
    // A fire whose rows are its commands' own carries M so; one whose rows
    // vary within its body reads M from the fire block
    // (`linear/gemm.slang`'s kernels; the others carry their rows).
    let fixed = if ctx.fixed_rows() { m } else { 0 };
    let fire = kernel(target, tactic, op, m, n, k)?;
    let mut args = vec![
        act.arg(),
        w.arg(),
        y.arg_mut(),
        stated(op, m)?.arg(),
        stated(op, n)?.arg(),
        stated(op, k)?.arg(),
    ];
    if fire.file == GEMM {
        args.push(stated(op, fixed)?.arg());
    }
    args.push(window.row0.arg());
    ctx.fire(fire, &args)
}

/// Every kernel a projection of `n` columns over a contraction of `k`
/// takes under `tactic`'s tier, vector, narrow or wide: which a rung takes
/// is the load's choice, made after its kernels are named.
pub fn kernels(
    target: Target,
    tactic: Tactic,
    op: &'static str,
    n: u32,
    k: u32,
) -> Result<Vec<Fire>, Error> {
    let mut all = Vec::new();
    for wide in [false, true] {
        let tiled = Tactic {
            vector: 0,
            wide,
            ..tactic
        };
        crate::encode::distinct(&mut all, kernel(target, tiled, op, 1, n, k)?);
    }
    for rows in VECTOR_ROWS {
        let vector = Tactic {
            vector: rows,
            ..tactic
        };
        crate::encode::distinct(&mut all, kernel(target, vector, op, 1, n, k)?);
    }
    Ok(all)
}

/// The kernel a projection of `n` columns over a contraction of `k` fires
/// under `tactic`, at any rows: what a measurement of it is keyed by.
pub fn entry(target: Target, tactic: Tactic, n: u32, k: u32) -> Result<Fire, Error> {
    kernel(target, tactic, "linear.entry", 1, n, k)
}

/// The workgroups a projection of `m` rows over `n` columns and a
/// contraction of `k` launches under `tactic`: its kernel's grid.
pub fn workgroups(target: Target, tactic: Tactic, m: u32, n: u32, k: u32) -> Result<u64, Error> {
    let fire = kernel(target, tactic, "linear.workgroups", m, n, k)?;
    Ok(fire.groups.iter().map(|&g| u64::from(g)).product())
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
                1 => "linear_gemm_gemv_v1_bf16",
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
        Target::Spirv if tactic.wide && k.is_multiple_of(16) => {
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
                let rows = vector_rows(tactic.vector);
                let entry = match rows {
                    1 => "linear_gemm_pairs_gemv_v1_bf16",
                    4 => "linear_gemm_pairs_gemv_bf16",
                    8 => "linear_gemm_pairs_gemv_v8_bf16",
                    _ => "linear_gemm_pairs_gemv_v16_bf16",
                };
                Fire::at(PAIRS, entry)
                    .tier(if tactic.tier == Tier::Subgroup {
                        Tier::Subgroup
                    } else {
                        Tier::Baseline
                    })
                    .groups([1, n.div_ceil(8), m.div_ceil(rows)])
                    .group([32, 8, 1])
            } else if !tactic.wide {
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
