use kernels::{Bind, Fire};

use crate::jit::abi::Tensor;
use crate::jit::abi::bf16;
use crate::jit::{Ctx, Launch};
use kernels::Refusal;
use kernels::plane::{Const, In, Out};

#[allow(clippy::too_many_arguments)]
pub fn qwen_gdn_post_conv_prep_bf16(
    ctx: &Ctx<'_>,
    qkv_post: In<Tensor<bf16>>,
    a: In<Tensor<bf16>>,
    b: In<Tensor<bf16>>,
    a_log: Const<Tensor<f32>>,
    dt_bias: Const<Tensor<bf16>>,
    q_norm_kh: Out<Tensor<f32>>,
    k_norm_kh: Out<Tensor<f32>>,
    v_fp32: Out<Tensor<f32>>,
    g_log_out: Out<Tensor<f32>>,
    beta_out: Out<Tensor<f32>>,
    k_h: Const<i32>,
    v_h: Const<i32>,
    k_d: Const<i32>,
    v_d: Const<i32>,
    conv_dim: Const<i32>,
) -> Result<(), Refusal> {
    let k_h = *k_h;
    let v_h = *v_h;
    let k_d = *k_d;
    let v_d = *v_d;
    let conv_dim = *conv_dim;

    const PREP_BLOCK: u32 = 128;
    let n = qkv_post.all("the post-convolution qkv")?.rows;
    #[allow(clippy::cast_precision_loss)]
    let q_scale = (k_d as f32).sqrt().recip();

    ctx.fire(
        Fire::at(
            "ssm/gated_delta_net_prep.cuh",
            "::pie::ssm::qwen_gdn_qk_norm<::pie::bf16, 128>",
        )
        .apply(Launch::grid(
            [n.unsigned_abs(), k_h.unsigned_abs(), 1],
            [PREP_BLOCK, 1, 1],
        )),
        &[
            qkv_post.arg(),
            q_norm_kh.arg(),
            k_norm_kh.arg(),
            k_h.arg(),
            k_d.arg(),
            conv_dim.arg(),
            q_scale.arg(),
        ],
    )?;

    ctx.fire(
        Fire::at(
            "ssm/gated_delta_net_prep.cuh",
            "::pie::ssm::qwen_gdn_v_g_beta<::pie::bf16, 128>",
        )
        .apply(Launch::grid(
            [n.unsigned_abs(), v_h.unsigned_abs(), 1],
            [PREP_BLOCK, 1, 1],
        )),
        &[
            qkv_post.arg(),
            a.arg(),
            b.arg(),
            a_log.arg(),
            dt_bias.arg(),
            v_fp32.arg(),
            g_log_out.arg(),
            beta_out.arg(),
            k_h.arg(),
            v_h.arg(),
            k_d.arg(),
            v_d.arg(),
            conv_dim.arg(),
        ],
    )
}
