use crate::plane::{Bind, Const, Ctx, Fire, In, Out};
use crate::points::{Bank, Payload, at_bf16};
use kernels::plane::Refusal;
use kernels::points::{Form, Repr, Scalar};

const ROUTER_LANES: u32 = 256;

const ROUTER_MAX_EXPERTS: u32 = 1024;

const ROUTER_MAX_TOP_K: u32 = 16;

const SELECT_LANES: u32 = 32;

const QMV_LANES: u32 = 32;

const MXFP4_BLOCK: i32 = 32;

const SOFTMAX_OVER_SELECTED: u32 = 0;

const LOGITS_TIGHTLY_PACKED: u32 = 0;

struct Ranked {
    lanes: [u32; 3],
}

fn ranked<T: Scalar>(
    logits: In<Payload<T>>,
    experts: u32,
    top_k: u32,
    routes: Out<Payload<i32>>,
    weights: Out<Payload<f32>>,
    what: &'static str,
) -> Result<Ranked, Refusal> {
    at_bf16::<T>(what)?;
    if experts == 0 {
        return Err(Refusal::Empty {
            what: "the expert count this router states",
        });
    }
    if top_k == 0 {
        return Err(Refusal::Empty {
            what: "the fan-out this router states",
        });
    }
    if experts > ROUTER_MAX_EXPERTS {
        return Err(Refusal::Wide {
            what: "the expert count, which this router stages in a workgroup array",
            at: i64::from(experts),
            max: i64::from(ROUTER_MAX_EXPERTS),
        });
    }
    if top_k > ROUTER_MAX_TOP_K {
        return Err(Refusal::Wide {
            what: "the fan-out, which this router stages in a workgroup array",
            at: i64::from(top_k),
            max: i64::from(ROUTER_MAX_TOP_K),
        });
    }
    if logits.width <= 0 || logits.width.unsigned_abs() != experts {
        return Err(Refusal::Narrow {
            what: "the router's row is not the expert count the statement states",
            at: i64::from(logits.width),
        });
    }
    if routes.width <= 0 || routes.width.unsigned_abs() != top_k || weights.width != routes.width {
        return Err(Refusal::Narrow {
            what: "a routed result is not the fan-out the statement states",
            at: i64::from(routes.width),
        });
    }
    if logits.rows <= 0 {
        return Err(Refusal::Empty {
            what: "the token rows this router reads",
        });
    }
    Ok(Ranked {
        lanes: [ROUTER_LANES, logits.rows.unsigned_abs(), 1],
    })
}

struct Selected {
    in_width: i32,
    out_width: i32,
    top_k: i32,
    tokens: i32,
    routed: i32,
    x_row_stride: i32,
    x_slot_stride: i32,
}

fn selected<T: Scalar>(
    x: In<Payload<T>>,
    routes: In<Payload<i32>>,
    y: Out<Payload<T>>,
    what: &'static str,
) -> Result<Selected, Refusal> {
    at_bf16::<T>(what)?;
    if routes.width <= 0 || routes.rows <= 0 {
        return Err(Refusal::Empty {
            what: "the routed fanout",
        });
    }
    if x.width <= 0 {
        return Err(Refusal::Empty {
            what: "K, the activation's width",
        });
    }
    if y.width <= 0 {
        return Err(Refusal::Empty {
            what: "N, the bank's output width",
        });
    }
    let routed = routes.rows.checked_mul(routes.width).ok_or(Refusal::Grid {
        what: "the route run, which is the tokens times the fan-out",
        at: i64::from(routes.rows) * i64::from(routes.width),
    })?;
    if y.rows != routed {
        return Err(Refusal::Narrow {
            what: "the result's rows against one row per route",
            at: i64::from(y.rows),
        });
    }
    let (x_row_stride, x_slot_stride) = if x.rows == routes.rows {
        (x.width, 0)
    } else if x.rows == routed {
        (
            x.width.checked_mul(routes.width).ok_or(Refusal::Grid {
                what: "the activation's row, which is the fan-out times its slot",
                at: i64::from(x.width) * i64::from(routes.width),
            })?,
            x.width,
        )
    } else {
        return Err(Refusal::Narrow {
            what: "the activation's rows, which are the fire's tokens or its routes and \
                   neither here",
            at: i64::from(x.rows),
        });
    };
    Ok(Selected {
        in_width: x.width,
        out_width: y.width,
        top_k: routes.width,
        tokens: routes.rows,
        routed,
        x_row_stride,
        x_slot_stride,
    })
}

#[kernels_macros::claims]
impl kernels::points::Moe for Ctx<'_> {
    fn topk_softmax<T: Scalar>(
        &self,
        logits: In<Payload<T>>,
        experts: u32,
        top_k: u32,
        routes: Out<Payload<i32>>,
        weights: Out<Payload<f32>>,
    ) -> Result<(), Refusal> {
        let row = ranked(
            logits,
            experts,
            top_k,
            routes,
            weights,
            "moe.topk_softmax at an element other than bf16",
        )?;
        self.fire(
            Fire::at("moe/route.wgsl", "router_topk_f32w_bfloat16").apply(row.lanes),
            &[
                logits.arg(),
                routes.arg(),
                weights.arg(),
                experts.arg(),
                top_k.arg(),
                SOFTMAX_OVER_SELECTED.arg(),
                LOGITS_TIGHTLY_PACKED.arg(),
            ],
        )
    }

    fn topk_sigmoid<T: Scalar>(
        &self,
        logits: In<Payload<T>>,
        experts: u32,
        top_k: u32,
        renormalize: bool,
        scaling: f32,
        routes: Out<Payload<i32>>,
        weights: Out<Payload<f32>>,
    ) -> Result<(), Refusal> {
        let row = ranked(
            logits,
            experts,
            top_k,
            routes,
            weights,
            "moe.topk_sigmoid at an element other than bf16",
        )?;
        self.fire(
            Fire::at("moe/route.wgsl", "router_topk_sigmoid").apply(row.lanes),
            &[
                logits.arg(),
                routes.arg(),
                weights.arg(),
                experts.arg(),
                top_k.arg(),
                u32::from(renormalize).arg(),
                scaling.arg(),
            ],
        )
    }

    fn topk_sqrt_softplus<T: Scalar>(
        &self,
        logits: In<Payload<T>>,
        bias: Const<Payload<f32>>,
        experts: u32,
        top_k: u32,
        renormalize: bool,
        scaling: f32,
        routes: Out<Payload<i32>>,
        weights: Out<Payload<f32>>,
    ) -> Result<(), Refusal> {
        let row = ranked(
            logits,
            experts,
            top_k,
            routes,
            weights,
            "moe.topk_sqrt_softplus at an element other than bf16",
        )?;
        self.fire(
            Fire::at("moe/route.wgsl", "router_topk_sqrt_softplus").apply(row.lanes),
            &[
                logits.arg(),
                routes.arg(),
                weights.arg(),
                bias.get().arg(),
                experts.arg(),
                top_k.arg(),
                u32::from(renormalize).arg(),
                scaling.arg(),
            ],
        )
    }

    fn matmul_select<T: Scalar>(
        &self,
        x: In<Payload<T>>,
        bank: Const<Payload<T>>,
        routes: In<Payload<i32>>,
        y: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        let fan = selected(
            x,
            routes,
            y,
            "moe.matmul_select at an element other than bf16",
        )?;
        self.fire(
            Fire::at("moe/select.wgsl", "select_gemv").apply([
                SELECT_LANES.saturating_mul(fan.out_width.unsigned_abs()),
                fan.routed.unsigned_abs(),
                1,
            ]),
            &[
                x.arg(),
                bank.get().arg(),
                routes.arg(),
                y.arg(),
                fan.in_width.arg(),
                fan.out_width.arg(),
                fan.top_k.arg(),
                fan.x_row_stride.arg(),
                fan.x_slot_stride.arg(),
            ],
        )
    }

    fn matmul_select_bias<T: Scalar, R: Repr>(
        &self,
        x: In<Payload<T>>,
        bank: Const<Bank<R>>,
        bias: Const<Payload<T>>,
        routes: In<Payload<i32>>,
        y: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        let fan = selected(
            x,
            routes,
            y,
            "moe.matmul_select_bias at an element other than bf16",
        )?;
        let planes = bank.get();
        match R::FORM {
            Form::Mxfp4 => {
                if fan.in_width % MXFP4_BLOCK != 0 {
                    return Err(Refusal::Narrow {
                        what: "K, in whole 32-code MXFP4 blocks",
                        at: i64::from(fan.in_width),
                    });
                }
                self.fire(
                    Fire::at(
                        "moe/qmv_routed.wgsl",
                        "mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4",
                    )
                    .apply([
                        QMV_LANES.saturating_mul(fan.tokens.unsigned_abs()),
                        fan.out_width.unsigned_abs(),
                        fan.top_k.unsigned_abs(),
                    ]),
                    &[
                        planes.codes.arg(),
                        planes.scales.arg(),
                        x.arg(),
                        y.arg(),
                        bias.get().arg(),
                        routes.arg(),
                        fan.in_width.arg(),
                        fan.out_width.arg(),
                        fan.x_slot_stride.arg(),
                        fan.x_row_stride.arg(),
                        fan.top_k.arg(),
                    ],
                )
            }
        }
    }

    fn weighted_sum<T: Scalar>(
        &self,
        routed: In<Payload<T>>,
        weights: In<Payload<f32>>,
        y: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("moe.weighted_sum at an element other than bf16")?;
        if y.rows <= 0 || y.width <= 0 {
            return Err(Refusal::Empty {
                what: "the token rows this fold lands on",
            });
        }
        if routed.rows % y.rows != 0 {
            return Err(Refusal::Narrow {
                what: "the routed rectangle, against the token rows it folds into",
                at: i64::from(routed.rows),
            });
        }
        let top_k = routed.rows / y.rows;
        if top_k <= 0 {
            return Err(Refusal::Empty {
                what: "the routed fanout",
            });
        }
        if routed.width != y.width {
            return Err(Refusal::Narrow {
                what: "the routed row's width, which the fold does not change",
                at: i64::from(routed.width),
            });
        }
        if weights.rows != y.rows || weights.width != top_k {
            return Err(Refusal::Narrow {
                what: "the weight plane, which is one weight per route",
                at: i64::from(weights.width),
            });
        }
        self.fire(
            Fire::at("moe/route.wgsl", "expert_combine").apply(rows_by_width(y.width, y.rows)?),
            &[
                routed.arg(),
                weights.arg(),
                y.arg(),
                y.width.arg(),
                top_k.arg(),
            ],
        )
    }

    fn sigmoid_gate_add<T: Scalar>(
        &self,
        routed: In<Payload<T>>,
        shared: In<Payload<T>>,
        gate: In<Payload<T>>,
        y: Out<Payload<T>>,
    ) -> Result<(), Refusal> {
        at_bf16::<T>("moe.sigmoid_gate_add at an element other than bf16")?;
        let width = routed.width;
        self.fire(
            Fire::at("moe/route.wgsl", "shared_expert_combine")
                .apply(rows_by_width(width, routed.rows)?),
            &[routed.arg(), shared.arg(), gate.arg(), y.arg(), width.arg()],
        )
    }
}

fn rows_by_width(width: i32, rows: i32) -> Result<[u32; 3], Refusal> {
    if width <= 0 {
        return Err(Refusal::Empty { what: "width" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    Ok([width.unsigned_abs(), rows.unsigned_abs(), 1])
}
