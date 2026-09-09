use dtype::Dtype;

use crate::encode::{Arg, Ctx, Fire, Grid, nonzero, refuse};
use crate::error::Error;
use crate::tensor::Tensor;

const FILE: &str = "linear/lane_gemm.metal";

const THREADS: u32 = 256;

const SIMDS: u32 = THREADS / 32;

pub fn act_x_wt(
    ctx: &Ctx<'_>,
    op: &'static str,
    act: Tensor,
    w: Tensor,
    y: Tensor,
) -> Result<(), Error> {
    if y.rows == 0 {
        return Ok(());
    }
    let entry = match (act.dtype, w.dtype, y.dtype) {
        (Dtype::F32, Dtype::Bf16, Dtype::F32) => "lane_gemm_f32_bf16_f32",
        (Dtype::F32, Dtype::Bf16, Dtype::Bf16) => "lane_gemm_f32_bf16_bf16",
        (Dtype::F32, Dtype::F32, Dtype::F32) => "lane_gemm_f32_f32_f32",
        (Dtype::Bf16, Dtype::Bf16, Dtype::F32) => "lane_gemm_bf16_bf16_f32",
        (a, weight, out) => {
            return Err(refuse(
                op,
                format!(
                    "no lane-axis point is stamped for a {a:?} activation against a \
                     {weight:?} weight into a {out:?} result"
                ),
            ));
        }
    };
    let k = nonzero(op, "the contracted width", act.width)?;
    let n = nonzero(op, "the projected width", y.width)?;
    if w.width != k || w.rows != n {
        return Err(refuse(
            op,
            format!(
                "the weight is {} x {} and this projection contracts {k} into {n}",
                w.rows, w.width
            ),
        ));
    }
    if y.rows != act.rows {
        return Err(refuse(
            op,
            format!(
                "the activation has {} rows and the result {}",
                act.rows, y.rows
            ),
        ));
    }
    let columns = n.div_ceil(SIMDS);
    let lanes = columns.checked_mul(THREADS).ok_or_else(|| {
        refuse(
            op,
            format!("the grid will not launch: {n} columns x {} rows", y.rows),
        )
    })?;
    ctx.fire(
        Fire::at(FILE, entry).apply(Grid::of([lanes, y.rows, 1], [THREADS, 1, 1])),
        &[
            act.arg(),
            w.arg(),
            y.arg_mut(),
            y.rows.arg(),
            n.arg(),
            k.arg(),
        ],
    )
}
