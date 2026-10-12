use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, refuse, stated};
use crate::error::Error;
use crate::tensor::Tensor;

const FILE: &str = "gemm/lane_gemm.slang";

pub fn act_x_wt(
    ctx: &Ctx<'_>,
    op: &'static str,
    act: Tensor,
    w: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    let (rows, columns, contraction) = super::gemm::extent(op, act, w, y)?;
    if rows == 0 {
        return Ok(());
    }
    let entry = match (act.dtype, w.dtype, y.dtype) {
        (Dtype::F32, Dtype::Bf16, Dtype::F32) => "lane_gemm_f32_bf16_f32",
        (Dtype::F32, Dtype::F32, Dtype::F32) => "lane_gemm_f32_f32_f32",
        (Dtype::F32, Dtype::Bf16, Dtype::Bf16) => "lane_gemm_f32_bf16_bf16",
        (Dtype::F32, Dtype::F32, Dtype::Bf16) => "lane_gemm_f32_f32_bf16",
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
        Fire::at(FILE, entry).apply(Grid::of([columns, rows, 1], [64, 4, 1])),
        &[
            act.arg(),
            w.arg(),
            y.arg_mut(),
            stated(op, rows)?.arg(),
            stated(op, columns)?.arg(),
            stated(op, contraction)?.arg(),
        ],
    )
}
