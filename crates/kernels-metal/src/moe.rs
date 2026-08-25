use kernels::Grid;
use kernels::plane::Refusal;
use kernels::points::{Form, Repr, Scalar};

use crate::plane::{Bind, Const, Ctx, Fire, In, Out, Tensor, bf16};
use crate::points::{self, Handle, Planes};

fn router_lanes(n_experts: u32) -> Result<u32, Refusal> {
    if n_experts == 0 {
        return Err(Refusal::Empty { what: "n_experts" });
    }
    Ok(n_experts.min(1024).div_ceil(32) * 32)
}

fn route_rows(width: i32, rows: i32) -> Result<([u32; 3], [u32; 3]), Refusal> {
    if width <= 0 {
        return Err(Refusal::Empty { what: "width" });
    }
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    let w = width.unsigned_abs();
    Ok(([w, rows.unsigned_abs(), 1], [w.min(256), 1, 1]))
}

const ROUTER_MAX_EXPERTS: u32 = 1024;

const ROUTER_MAX_TOP_K: u32 = 16;

const SELECT_GROUP: u32 = 128;

const QMV_GROUP: [u32; 3] = [32, 2, 1];

const MXFP4_BLOCK: i32 = 32;

struct Ranked {
    logits: In<Tensor<bf16>>,
    routes: Out<Tensor<i32>>,
    weights: Out<Tensor<f32>>,
    grid: Grid,
}

fn ranked<T: Scalar>(
    logits: In<Handle<T>>,
    experts: u32,
    top_k: u32,
    routes: Out<Handle<i32>>,
    weights: Out<Handle<f32>>,
    what: &'static str,
) -> Result<Ranked, Refusal> {
    let logits = points::input::<T, bf16>(logits, what)?;
    let routes = points::result::<i32, i32>(routes, what)?;
    let weights = points::result::<f32, f32>(weights, what)?;
    let width = points::stated(experts, "the expert count this router states")?;
    let k = points::stated(top_k, "the fan-out this router states")?;
    if experts > ROUTER_MAX_EXPERTS {
        return Err(Refusal::Wide {
            what: "the expert count, which this router gives one lane apiece",
            at: i64::from(experts),
            max: i64::from(ROUTER_MAX_EXPERTS),
        });
    }
    if top_k > ROUTER_MAX_TOP_K {
        return Err(Refusal::Wide {
            what: "the fan-out, which this router stages in a threadgroup array",
            at: i64::from(top_k),
            max: i64::from(ROUTER_MAX_TOP_K),
        });
    }
    if logits.width != width {
        return Err(Refusal::Narrow {
            what: "the router's row is not the expert count the statement states",
            at: i64::from(logits.width),
        });
    }
    if routes.width != k || weights.width != k {
        return Err(Refusal::Narrow {
            what: "a routed result is not the fan-out the statement states",
            at: i64::from(routes.width),
        });
    }
    let lanes = router_lanes(experts)?;
    let rows = u32::try_from(logits.rows).map_err(|_| Refusal::Empty { what: "rows" })?;
    Ok(Ranked {
        logits,
        routes,
        weights,
        grid: Grid::of([lanes, rows, 1], [lanes, 1, 1]),
    })
}

struct Selected {
    x: In<Tensor<bf16>>,
    routes: In<Tensor<i32>>,
    y: Out<Tensor<bf16>>,
    in_width: i32,
    out_width: i32,
    top_k: i32,
    tokens: i32,
    x_row_stride: i32,
    x_slot_stride: i32,
}

fn selected<T: Scalar>(
    x: In<Handle<T>>,
    routes: In<Handle<i32>>,
    y: Out<Handle<T>>,
    what: &'static str,
) -> Result<Selected, Refusal> {
    let x = points::input::<T, bf16>(x, what)?;
    let routes = points::input::<i32, i32>(routes, what)?;
    let y = points::result::<T, bf16>(y, what)?;
    if routes.width <= 0 {
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
        x,
        routes,
        y,
        in_width: x.width,
        out_width: y.width,
        top_k: routes.width,
        tokens: routes.rows,
        x_row_stride,
        x_slot_stride,
    })
}

fn select_gemv_grid(out_width: i32, routes: i32) -> Result<Grid, Refusal> {
    let lanes = out_width
        .unsigned_abs()
        .checked_mul(32)
        .ok_or(Refusal::Grid {
            what: "the output rows times the simdgroup width",
            at: i64::from(out_width) * 32,
        })?;
    Ok(Grid::of(
        [lanes, routes.unsigned_abs(), 1],
        [SELECT_GROUP, 1, 1],
    ))
}

pub fn routed_qmv_grid(rows: i32, out_vec_size: i32, slots: i32) -> Result<[u32; 3], Refusal> {
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if out_vec_size <= 0 {
        return Err(Refusal::Empty {
            what: "out_vec_size",
        });
    }
    if slots <= 0 {
        return Err(Refusal::Empty {
            what: "slots_per_row",
        });
    }
    let x = rows.unsigned_abs().checked_mul(32).ok_or(Refusal::Grid {
        what: "rows * the simdgroup width",
        at: i64::from(rows) * 32,
    })?;
    Ok([
        x,
        out_vec_size.unsigned_abs().div_ceil(4),
        slots.unsigned_abs(),
    ])
}

pub fn routed_qmv_widths(
    x_slot_stride: i32,
    y_width: i32,
    slots: i32,
) -> Result<(i32, i32), Refusal> {
    if x_slot_stride <= 0 {
        return Err(Refusal::Empty {
            what: "x_slot_stride",
        });
    }
    if y_width <= 0 {
        return Err(Refusal::Empty {
            what: "out_vec_size",
        });
    }
    if slots <= 0 {
        return Err(Refusal::Empty {
            what: "slots_per_row",
        });
    }
    if !y_width.unsigned_abs().is_multiple_of(slots.unsigned_abs()) {
        return Err(Refusal::Narrow {
            what: "an output width the slot count does not divide",
            at: i64::from(y_width),
        });
    }
    Ok((x_slot_stride, y_width / slots))
}

pub fn routed_qmm_grid(rows: i32, n: i32, tile_m: i32, tile_n: i32) -> Result<[u32; 3], Refusal> {
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if n <= 0 {
        return Err(Refusal::Empty { what: "n" });
    }
    let (m, bn) = (tile_m.unsigned_abs(), tile_n.unsigned_abs());
    if m == 0 || !rows.unsigned_abs().is_multiple_of(m) {
        return Err(Refusal::Narrow {
            what: "rows the row tile does not divide",
            at: i64::from(rows),
        });
    }
    if bn == 0 || !n.unsigned_abs().is_multiple_of(bn) {
        return Err(Refusal::Narrow {
            what: "an output width the column tile does not divide",
            at: i64::from(n),
        });
    }
    Ok([
        32 * (n.unsigned_abs() / bn),
        2 * (rows.unsigned_abs() / m),
        2,
    ])
}

pub fn tile_point(tile_m: i32, tile_n: i32) -> Result<usize, Refusal> {
    let axis = |v: i32, what: &'static str| match v {
        16 => Ok(0),
        32 => Ok(1),
        64 => Ok(2),
        _ => Err(Refusal::Narrow {
            what,
            at: i64::from(v),
        }),
    };
    Ok(axis(tile_m, "the routed qmm's row tile")? * 3
        + axis(tile_n, "the routed qmm's column tile")?)
}

#[allow(clippy::too_many_arguments)]
pub fn combine_sorted(
    ctx: &Ctx<'_>,
    y: In<Tensor<bf16>>,
    expert_weights: In<Tensor<bf16>>,
    out: Out<Tensor<bf16>>,
    inv: In<Tensor<i32>>,
    width: Const<u32>,
    experts_per_token: Const<u32>,
    out_pitch: Const<u32>,
    tokens: Const<i32>,
) -> Result<(), Refusal> {
    let y_width = y.width;
    let tokens = *tokens;
    let (lanes, group) = route_rows(y_width, tokens)?;
    ctx.fire(
        Fire::at("moe/route.metal", "combine_sorted").apply(Grid::of(lanes, group)),
        &[
            y.arg(),
            expert_weights.arg(),
            out.arg(),
            inv.arg(),
            width.arg(),
            experts_per_token.arg(),
            out_pitch.arg(),
        ],
    )
}

const SOFTMAX_OVER_SELECTED: u32 = 0;

#[kernels_macros::claims]
impl kernels::points::Moe for Ctx<'_> {
    fn topk_softmax<T: Scalar>(
        &self,
        logits: In<Handle<T>>,
        experts: u32,
        top_k: u32,
        routes: Out<Handle<i32>>,
        weights: Out<Handle<f32>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`moe.topk_softmax`, at an element this plane does not stamp";
        let logits = points::input::<T, bf16>(logits, WHAT)?;
        let routes = points::result::<i32, i32>(routes, "`moe.topk_softmax`'s route plane")?;
        let weights = points::result::<f32, f32>(weights, "`moe.topk_softmax`'s weight plane")?;
        let width = points::stated(experts, "the expert count this router states")?;
        let k = points::stated(top_k, "the fan-out this router states")?;
        if logits.width != width {
            return Err(Refusal::Narrow {
                what: "the router's row is not the expert count the statement states",
                at: i64::from(logits.width),
            });
        }

        if routes.width != k || weights.width != k {
            return Err(Refusal::Narrow {
                what: "a routed result is not the fan-out the statement states",
                at: i64::from(routes.width),
            });
        }
        let lanes = router_lanes(experts)?;
        let rows = u32::try_from(logits.rows).map_err(|_| Refusal::Empty { what: "rows" })?;
        self.fire(
            Fire::at("moe/route.metal", "router_topk_f32w_bfloat16")
                .apply(Grid::of([lanes, rows, 1], [lanes, 1, 1])),
            &[
                logits.arg(),
                routes.arg(),
                weights.arg(),
                self.absent()?,
                experts.arg(),
                top_k.arg(),
                SOFTMAX_OVER_SELECTED.arg(),
                experts.arg(),
            ],
        )
    }

    fn topk_sigmoid<T: Scalar>(
        &self,
        logits: In<Handle<T>>,
        experts: u32,
        top_k: u32,
        renormalize: bool,
        scaling: f32,
        routes: Out<Handle<i32>>,
        weights: Out<Handle<f32>>,
    ) -> Result<(), Refusal> {
        let row = ranked(
            logits,
            experts,
            top_k,
            routes,
            weights,
            "`moe.topk_sigmoid`, at an element this plane does not stamp",
        )?;
        self.fire(
            Fire::at("moe/route.metal", "router_topk_sigmoid").apply(row.grid),
            &[
                row.logits.arg(),
                row.routes.arg(),
                row.weights.arg(),
                experts.arg(),
                top_k.arg(),
                u32::from(renormalize).arg(),
                scaling.arg(),
            ],
        )
    }

    fn topk_sqrt_softplus<T: Scalar>(
        &self,
        logits: In<Handle<T>>,
        bias: Const<Handle<f32>>,
        experts: u32,
        top_k: u32,
        renormalize: bool,
        scaling: f32,
        routes: Out<Handle<i32>>,
        weights: Out<Handle<f32>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`moe.topk_sqrt_softplus`, at an element this plane does not stamp";
        let row = ranked(logits, experts, top_k, routes, weights, WHAT)?;
        let bias = points::weight::<f32, f32>(bias, "`moe.topk_sqrt_softplus`'s correction bias")?;
        self.fire(
            Fire::at("moe/route.metal", "router_topk_sqrt_softplus").apply(row.grid),
            &[
                row.logits.arg(),
                bias.arg(),
                row.routes.arg(),
                row.weights.arg(),
                experts.arg(),
                top_k.arg(),
                u32::from(renormalize).arg(),
                scaling.arg(),
            ],
        )
    }

    fn matmul_select<T: Scalar>(
        &self,
        x: In<Handle<T>>,
        bank: Const<Handle<T>>,
        routes: In<Handle<i32>>,
        y: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`moe.matmul_select`, at an element this plane does not stamp";
        let fan = selected(x, routes, y, WHAT)?;
        let bank = points::weight::<T, bf16>(bank, WHAT)?;
        self.fire(
            Fire::at("moe/select.metal", "select_gemv")
                .apply(select_gemv_grid(fan.out_width, fan.y.rows)?),
            &[
                fan.x.arg(),
                bank.arg(),
                fan.routes.arg(),
                fan.y.arg(),
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
        x: In<Handle<T>>,
        bank: Const<Planes<R>>,
        bias: Const<Handle<T>>,
        routes: In<Handle<i32>>,
        y: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`moe.matmul_select_bias`, at an element this plane does not stamp";
        let fan = selected(x, routes, y, WHAT)?;
        let bias = points::weight::<T, bf16>(bias, "`moe.matmul_select_bias`'s expert bias")?;
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
                        "quant/qmv.metal",
                        "mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4",
                    )
                    .apply(Grid::of(
                        routed_qmv_grid(fan.tokens, fan.out_width, fan.top_k)?,
                        QMV_GROUP,
                    )),
                    &[
                        Tensor::<u8>::new(planes.codes).arg(),
                        Tensor::<u8>::new(planes.scales).arg(),
                        self.absent()?,
                        fan.x.arg(),
                        fan.y.arg(),
                        fan.in_width.arg(),
                        fan.out_width.arg(),
                        bias.arg(),
                        fan.routes.arg(),
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
        routed: In<Handle<T>>,
        weights: In<Handle<f32>>,
        y: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`moe.weighted_sum`, at an element this plane does not stamp";
        let routed = points::input::<T, bf16>(routed, WHAT)?;
        let weights = points::input::<f32, f32>(weights, "`moe.weighted_sum`'s weight plane")?;
        let y = points::result::<T, bf16>(y, WHAT)?;
        if y.rows <= 0 {
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
        let (lanes, group) = route_rows(y.width, y.rows)?;
        self.fire(
            Fire::at("moe/route.metal", "expert_combine").apply(Grid::of(lanes, group)),
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
        routed: In<Handle<T>>,
        shared: In<Handle<T>>,
        gate: In<Handle<T>>,
        y: Out<Handle<T>>,
    ) -> Result<(), Refusal> {
        const WHAT: &str = "`moe.sigmoid_gate_add`, at an element this plane does not stamp";
        let routed = points::input::<T, bf16>(routed, WHAT)?;
        let (lanes, group) = route_rows(routed.width, routed.rows)?;
        self.fire(
            Fire::at("moe/route.metal", "shared_expert_combine").apply(Grid::of(lanes, group)),
            &[
                routed.arg(),
                points::input::<T, bf16>(shared, WHAT)?.arg(),
                points::input::<T, bf16>(gate, WHAT)?.arg(),
                points::result::<T, bf16>(y, WHAT)?.arg(),
                routed.width.arg(),
            ],
        )
    }
}
