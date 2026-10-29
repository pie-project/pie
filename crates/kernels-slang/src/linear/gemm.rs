//! Dense matmuls (`linear.matmul`, `linear.matmul` over a bf16 weight, and
//! an f32 activation's lane form). SPIR-V fires `kernels/linear/gemm.slang`
//! (the vector kernels below four rows, the tiled kernel or its
//! cooperative-matrix tier above) and `gemm_lane.slang` for f32
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
    let fire = match ctx.target() {
        Target::Spirv if m < VECTOR_MAX_ROWS && k.is_multiple_of(8) => {
            Fire::at(GEMM, "linear_gemm_gemv_v_bf16")
                .groups([n.div_ceil(8), 1, 1])
                .group([32, 8, 1])
        }
        Target::Spirv if m < VECTOR_MAX_ROWS => Fire::at(GEMM, "linear_gemm_gemv_bf16")
            .groups([1, n.div_ceil(8), m])
            .group([32, 8, 1]),
        Target::Spirv if tactic.tier == Tier::Coopmat => {
            Fire::at(GEMM, "linear_gemm_tiled_bm_64_bf16")
                .tier(Tier::Coopmat)
                .groups([n.div_ceil(64), m.div_ceil(64), 1])
                .group([32, 4, 1])
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
            if m < VECTOR_MAX_ROWS {
                Fire::at(PAIRS, "linear_gemm_pairs_gemv_bf16")
                    .tier(if tactic.tier == Tier::Subgroup {
                        Tier::Subgroup
                    } else {
                        Tier::Baseline
                    })
                    .groups([1, n.div_ceil(8), m])
                    .group([32, 8, 1])
            } else if m < NARROW_MAX_ROWS {
                Fire::at(PAIRS, "linear_gemm_pairs_tiled_bm_16_bf16")
                    .groups([n.div_ceil(64), m.div_ceil(16), 1])
                    .group([16, 16, 1])
            } else {
                Fire::at(PAIRS, "linear_gemm_pairs_tiled_bf16")
                    .groups([n.div_ceil(64), m.div_ceil(64), 1])
                    .group([16, 16, 1])
            }
        }
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
