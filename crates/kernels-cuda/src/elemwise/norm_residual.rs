use super::*;
use crate::error::Error;
use crate::jit::{
    Arg, Ctx, Fire, Launch, aligned16, dtype_dispatch, kernel, nonzero, refuse, stated,
};
use crate::tensor::Tensor;
use dtype::Dtype;

const NORM_RESIDUAL: &str = "elemwise/norm_residual.cu";

const VEC_WIDTH: u32 = 8;

pub fn residual_add_rmsnorm(
    ctx: &Ctx,
    x: Tensor,
    y: &mut Tensor,
    weight: Tensor,
    plus_one: bool,
    eps: f32,
    out: &mut Tensor,
) -> Result<(), Error> {
    const OP: &str = "fused.residual_add_rmsnorm";
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
        // The register kernel waits on its producer in-kernel (`kernel.pdl`).
        Fire::at(kernel(NORM_RESIDUAL, &entrypoint)).apply(
            Launch::per_row(rows, block)
                .programmatic(ctx.tuning().pdl && entrypoint.contains("_reg<")),
        ),
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
    const OP: &str = "fused.rmsnorm_residual_add";
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
        // The vec8 form waits on its producer in-kernel (`kernel.pdl`).
        Fire::at(kernel(NORM_RESIDUAL, &entrypoint))
            .apply(Launch::per_row(rows, block).programmatic(ctx.tuning().pdl && vectors)),
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
