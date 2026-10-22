use super::*;
use crate::error::Error;
use crate::jit::{
    Arg, Ctx, Fire, Launch, aligned16, dtype_dispatch, nonzero, refuse, stated, symbol,
};
use crate::tensor::Tensor;
use dtype::Dtype;

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
    let entrypoint = format!("::pie::elemwise::residual_add_rmsnorm<{t}, 256, {plus}>");
    let block = BLOCK;
    ctx.fire(
        OP,
        Fire::at(NORM_UNIT, symbol(&entrypoint)).apply(Launch::per_row(rows, block)),
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
        let block = nvec.clamp(WARP, 512).next_power_of_two();
        let chunks = nvec.div_ceil(block);
        (
            format!(
                "::pie::elemwise::rmsnorm_residual_add_vec8<{block}, {chunks}, {has_scale}, {has_post}, {plus}>"
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
        Fire::at(NORM_UNIT, symbol(&entrypoint)).apply(Launch::per_row(rows, block)),
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
