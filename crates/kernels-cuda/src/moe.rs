use crate::jit::abi::Bank;
use crate::jit::abi::Tensor;
use crate::jit::abi::bf16;
use crate::jit::{Ctx, Launch};
use kernels::plane::{Const, In, Out};
use kernels::points::{Form, Repr};
use kernels::{Bind, Fire};

use kernels::Refusal;

const BLOCK: u32 = 256;

const WARP: u32 = 32;

const FLOAT: u32 = 4;

const MOE_VEC_WIDTH: i32 = 8;

const GEMV_WARPS: i32 = 4;

#[must_use]
const fn rms(rows: u32) -> Launch {
    Launch::per_row(rows, BLOCK).smem((BLOCK / WARP) * FLOAT)
}

#[must_use]
const fn elementwise_rows(rows: u32, width: u32) -> Launch {
    Launch::grid([rows, width.div_ceil(BLOCK), 1], [BLOCK, 1, 1])
}

#[must_use]
const fn router_lane(rows: u32) -> Launch {
    const ROUTER_BLOCK: u32 = 64;

    Launch::per_row(rows, ROUTER_BLOCK)
}

const MAX_EXPERTS: i32 = 512;

fn too_many_experts(e: i32) -> Refusal {
    Refusal::Wide {
        what: "num_experts, which the router stages in shared memory",
        at: i64::from(e),
        max: i64::from(MAX_EXPERTS),
    }
}

#[allow(clippy::too_many_arguments)]
fn ranked_router<T: crate::RoutineElem>(
    ctx: &Ctx<'_>,
    root: &'static str,
    symbol: &str,
    logits: In<Tensor<T>>,
    routes: Out<Tensor<i32>>,
    weights: Out<Tensor<f32>>,
    correction_bias: Option<Const<Tensor<f32>>>,
    renormalize: bool,
    routed_scaling_factor: f32,
) -> Result<(), Refusal> {
    let rect = logits.all("num_experts")?;
    let routed = routes.all("the routed fanout")?;
    let (e, k) = (rect.width, routed.width);
    if e > MAX_EXPERTS {
        return Err(too_many_experts(e));
    }
    ctx.fire(
        Fire::at(root, crate::jit::symbol(symbol)).apply(rms(rect.rows.unsigned_abs())),
        &[
            rect.ptr.arg(),
            routed.ptr.arg(),
            weights.arg(),
            correction_bias.arg(),
            e.arg(),
            k.arg(),
            renormalize.arg(),
            routed_scaling_factor.arg(),
        ],
    )
}

#[kernels_macros::claims]
impl kernels::points::Moe for Ctx<'_> {
    fn topk_softmax<T: kernels::points::Scalar>(
        &self,
        logits: In<Tensor<T>>,
        experts: u32,
        top_k: u32,
        routes: Out<Tensor<i32>>,
        weights: Out<Tensor<f32>>,
    ) -> Result<(), Refusal> {
        let _ = (experts, top_k);
        let rect = logits.all("num_experts")?;
        let routed = routes.all("the routed fanout")?;
        let (num_experts, k) = (rect.width, routed.width);
        if num_experts > MAX_EXPERTS {
            return Err(too_many_experts(num_experts));
        }
        self.fire(
            Fire::at(
                "moe/topk_softmax.cuh",
                crate::jit::symbol(&format!("::pie::moe::topk_softmax<{}>", T::CPP)),
            )
            .apply(router_lane(rect.rows.unsigned_abs())),
            &[
                rect.ptr.arg(),
                core::ptr::null::<bf16>().arg(),
                core::ptr::null::<bf16>().arg(),
                routed.ptr.arg(),
                weights.arg(),
                num_experts.arg(),
                k.arg(),
                0_i32.arg(),
            ],
        )
    }

    fn topk_sigmoid<T: kernels::points::Scalar>(
        &self,
        logits: In<Tensor<T>>,
        experts: u32,
        top_k: u32,
        renormalize: bool,
        scaling: f32,
        routes: Out<Tensor<i32>>,
        weights: Out<Tensor<f32>>,
    ) -> Result<(), Refusal> {
        let _ = (experts, top_k);
        ranked_router(
            self,
            "moe/topk_sigmoid.cuh",
            &format!("::pie::moe::topk_sigmoid<{}>", T::CPP),
            logits,
            routes,
            weights,
            None,
            renormalize,
            scaling,
        )
    }

    fn topk_sqrt_softplus<T: kernels::points::Scalar>(
        &self,
        logits: In<Tensor<T>>,
        bias: Const<Tensor<f32>>,
        experts: u32,
        top_k: u32,
        renormalize: bool,
        scaling: f32,
        routes: Out<Tensor<i32>>,
        weights: Out<Tensor<f32>>,
    ) -> Result<(), Refusal> {
        let _ = (experts, top_k);
        ranked_router(
            self,
            "moe/dsv4_routing.cuh",
            &format!("::pie::moe::topk_sqrtsoftplus<{}>", T::CPP),
            logits,
            routes,
            weights,
            Some(bias),
            renormalize,
            scaling,
        )
    }

    fn weighted_sum<T: kernels::points::Scalar>(
        &self,
        routed: In<Tensor<T>>,
        weights: In<Tensor<f32>>,
        y: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        let fan = weights.all("the routed fanout")?;
        self.fire(
            Fire::at(
                "moe/moe_dispatch.cuh",
                crate::jit::symbol(&format!(
                    "::pie::moe::token_batched_weighted_sum<{}>",
                    T::CPP
                )),
            )
            .apply(elementwise_rows(
                y.rows.unsigned_abs(),
                y.width.unsigned_abs(),
            )),
            &[
                y.arg(),
                routed.ptr.arg(),
                fan.ptr.arg(),
                fan.width.arg(),
                y.width.arg(),
            ],
        )
    }

    fn matmul_select<T: kernels::points::Scalar>(
        &self,
        x: In<Tensor<T>>,
        bank: Const<Tensor<T>>,
        routes: In<Tensor<i32>>,
        y: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        const MAX_GRID_Y: i32 = 65_535;

        let dst = y.all("N, the bank's output width")?;
        let act = x.all("K, the activation's width")?;
        let fan = routes.all("the routed fanout")?;
        let (n, k) = (dst.width, act.width);
        let top_k = fan.width;

        if top_k <= 0 {
            return Err(Refusal::Empty {
                what: "the routed fanout",
            });
        }
        let route_count = fan.rows.saturating_mul(top_k);
        if dst.rows != route_count {
            return Err(Refusal::Narrow {
                what: "the result's rows against one row per route",
                at: i64::from(dst.rows),
            });
        }
        if route_count > MAX_GRID_Y {
            return Err(Refusal::Wide {
                what: "the route run, which this GEMV puts on the grid's y axis; the \
                       aligned batched leg is what a wider fire wants",
                at: i64::from(route_count),
                max: i64::from(MAX_GRID_Y),
            });
        }
        let by_token = if act.rows == route_count {
            false
        } else if act.rows.saturating_mul(top_k) == route_count {
            true
        } else {
            return Err(Refusal::Narrow {
                what: "the activation's rows, which are the fire's tokens or its routes and \
                       neither here",
                at: i64::from(act.rows),
            });
        };

        if k <= 0 || k % MOE_VEC_WIDTH != 0 {
            return Err(Refusal::Narrow {
                what: "K, in whole float4 loads of 8",
                at: i64::from(k),
            });
        }
        if n <= 0 {
            return Err(Refusal::Empty {
                what: "N, the bank's output width",
            });
        }
        let form = if by_token { "by_token" } else { "by_route" };
        self.fire(
            Fire::at(
                "moe/moe_dispatch.cuh",
                crate::jit::symbol(&format!("::pie::moe::moe_decode_gemv_{form}<{}>", T::CPP)),
            )
            .apply(Launch::grid(
                [
                    n.unsigned_abs().div_ceil(GEMV_WARPS.unsigned_abs()),
                    route_count.unsigned_abs(),
                    1,
                ],
                [WARP, GEMV_WARPS.unsigned_abs(), 1],
            )),
            &[
                fan.ptr.arg(),
                act.ptr.arg(),
                bank.arg(),
                dst.ptr.arg(),
                top_k.arg(),
                k.arg(),
                n.arg(),
                (i64::from(n) * i64::from(k)).arg(),
            ],
        )
    }

    fn sigmoid_gate_add<T: kernels::points::Scalar>(
        &self,
        routed: In<Tensor<T>>,
        shared: In<Tensor<T>>,
        gate: In<Tensor<T>>,
        y: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        let dst = y.all("the combined row's width")?;
        let sum = routed.over(dst.rows, "the routed row's width")?;
        let side = shared.over(dst.rows, "the shared expert's row width")?;
        if sum.width != dst.width || side.width != dst.width {
            return Err(Refusal::Narrow {
                what: "the two rows this combine adds, which are the result's width",
                at: i64::from(sum.width.min(side.width)),
            });
        }
        let col = gate.over(dst.rows, "the gate column")?;
        self.fire(
            Fire::at(
                "mlp/swiglu.cuh",
                crate::jit::symbol(&format!("::pie::mlp::sigmoid_scalar_gate_add<{}>", T::CPP)),
            )
            .apply(elementwise_rows(
                dst.rows.unsigned_abs(),
                dst.width.unsigned_abs(),
            )),
            &[
                dst.ptr.arg(),
                sum.ptr.arg(),
                side.ptr.arg(),
                col.ptr.arg(),
                dst.width.arg(),
                (*col.stride).arg(),
            ],
        )
    }

    fn matmul_select_bias<T: kernels::points::Scalar, R: Repr>(
        &self,
        x: In<Tensor<T>>,
        bank: Const<Bank<R>>,
        bias: Const<Tensor<T>>,
        routes: In<Tensor<i32>>,
        y: Out<Tensor<T>>,
    ) -> Result<(), Refusal> {
        match R::FORM {
            Form::Mxfp4 => mxfp4_matmul_select_bias(self, x, bank, bias, routes, y),
        }
    }
}

fn mxfp4_matmul_select_bias<T: kernels::points::Scalar, R: Repr>(
    ctx: &Ctx<'_>,
    x: In<Tensor<T>>,
    bank: Const<Bank<R>>,
    bias: Const<Tensor<T>>,
    routes: In<Tensor<i32>>,
    y: Out<Tensor<T>>,
) -> Result<(), Refusal> {
    const ROWS_PER_WARP: i32 = 4;
    const DECODE_BLOCK: u32 = 128;

    let dst = y.all("N, the bank's output width")?;
    let act = x.all("K, the activation's width")?;
    let fan = routes.all("the routed fanout")?;
    let (n, k) = (dst.width, act.width);
    let top_k = fan.width;

    if top_k <= 0 {
        return Err(Refusal::Empty {
            what: "the routed fanout",
        });
    }

    let route_count = fan.rows.saturating_mul(top_k);
    if dst.rows != route_count {
        return Err(Refusal::Narrow {
            what: "the result's rows against one row per route",
            at: i64::from(dst.rows),
        });
    }

    let act_div = if act.rows == route_count {
        1
    } else if act.rows.saturating_mul(top_k) == route_count {
        top_k
    } else {
        return Err(Refusal::Narrow {
            what: "the activation's rows, which are the fire's tokens or its routes and \
                   neither here",
            at: i64::from(act.rows),
        });
    };

    if k <= 0 || k % 32 != 0 {
        return Err(Refusal::Narrow {
            what: "K, in whole 32-code MXFP4 blocks",
            at: i64::from(k),
        });
    }
    if n <= 0 {
        return Err(Refusal::Empty {
            what: "N, the bank's output width",
        });
    }
    let planes = bank.get();
    if planes.codes.is_null() || planes.scales.is_null() {
        return Err(Refusal::Null {
            what: "an MXFP4 bank plane; a bank slot binds its codes AND its block scales",
        });
    }

    let tile = (DECODE_BLOCK / WARP) * ROWS_PER_WARP.unsigned_abs();
    ctx.fire(
        Fire::at(
            "quant/dequant_fp4.cuh",
            crate::jit::symbol(&format!(
                "::pie::quant::mxfp4_matmul_select_bias<{}, ::pie::i32({ROWS_PER_WARP})>",
                T::CPP
            )),
        )
        .apply(Launch::grid(
            [
                route_count.unsigned_abs(),
                n.unsigned_abs().div_ceil(tile),
                1,
            ],
            [DECODE_BLOCK, 1, 1],
        )),
        &[
            act.ptr.arg(),
            fan.ptr.arg(),
            planes.codes.arg(),
            planes.scales.arg(),
            bias.arg(),
            dst.ptr.arg(),
            act_div.arg(),
            n.arg(),
            k.arg(),
        ],
    )
}
