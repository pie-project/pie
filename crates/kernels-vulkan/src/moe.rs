use crate::plane::{Bind, Const, Ctx, Fire, In, Out, elementwise_rows};
use kernels::plane::Refusal;

/// **RENORMALISE OVER THE CHOSEN k, NOT OVER EVERY EXPERT.**
///
/// `route.slang`'s `softmax_over_all` field, and it was a bare `1` here — so
/// this plane's router divided each chosen logit by the sum over ALL thirty-two
/// experts where every other plane divides by the sum over the four it kept.
/// The four weights came out summing to 0.31 instead of 1, the mixture's output
/// was three and a half times small, and the tower still answered a plausible
/// token: `gptoss-20b`'s argmax was RIGHT and its logit 10.1250 against a
/// banked 14.4375.
///
/// The experts chosen were identical, which is what made it invisible to
/// everything but a value-by-value bisect: `moe.topk_softmax`'s route plane
/// agreed EXACTLY between this plane and `driver-wgpu`, and only the weight
/// plane beside it moved.
///
/// Named rather than written as a literal for exactly that reason. A `1` and a
/// `0` in an argument list are two plausible numbers and neither says which
/// question it is answering; `kernels-wgpu` names both and this now does too.
const SOFTMAX_OVER_SELECTED: u32 = 0;

/// The router's logit row has no pitch of its own — it is `n_experts` wide.
const LOGITS_TIGHTLY_PACKED: u32 = 0;

pub fn routed_qmv_grid(rows: i32, out_vec_size: i32, slots: i32) -> Result<[u32; 3], Refusal> {
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if out_vec_size <= 0 {
        return Err(Refusal::Empty {
            what: "the output vector",
        });
    }
    if slots <= 0 {
        return Err(Refusal::Empty { what: "slots" });
    }
    let x = rows.unsigned_abs().checked_mul(32).ok_or(Refusal::Grid {
        what: "rows * the matvec's lane count",
        at: i64::from(rows) * 32,
    })?;
    Ok([x, out_vec_size.unsigned_abs(), slots.unsigned_abs()])
}

pub fn routed_qmm_grid(rows: i32, n: i32, tile_m: i32, tile_n: i32) -> Result<[u32; 3], Refusal> {
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    if n <= 0 {
        return Err(Refusal::Empty {
            what: "the output width",
        });
    }
    let axis = |extent: i32, tile: i32, what: &'static str| -> Result<u32, Refusal> {
        if tile <= 0 {
            return Err(Refusal::Narrow {
                what,
                at: i64::from(tile),
            });
        }
        let tiles = extent.unsigned_abs().div_ceil(tile.unsigned_abs());
        tiles.checked_mul(16).ok_or(Refusal::Grid {
            what: "a tile count times the workgroup",
            at: i64::from(tiles) * 16,
        })
    };
    Ok([
        axis(n, tile_n, "the routed qmm's column tile")?,
        axis(rows, tile_m, "the routed qmm's row tile")?,
        1,
    ])
}

pub fn router_grid(rows: i32) -> Result<[u32; 3], Refusal> {
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    Ok([1024, rows.unsigned_abs(), 1])
}

/// The activation's two strides, from the shape it actually arrives in.
///
/// **A ROUTED MATVEC IS HANDED ONE OF TWO RECTANGLES** and the pair says which:
///
/// * one row per TOKEN, shared by every route — `(width, 0)`, so every slot of
///   a row reads the same activation.
/// * one row per ROUTE, already fanned out — `(width * top_k, width)`, so slot
///   `s` of token `n` is at `n * top_k + s`.
///
/// This plane passed `(width, 0)` UNCONDITIONALLY. `gptoss-20b` hands it the
/// second rectangle — `mlp.swiglu_clamp_alpha` lands four rows for one token —
/// so all four experts contracted against route zero's activations and three
/// of the four answers were of the wrong vector. The tower still answered the
/// banked TOKEN, at a logit a whole point low.
///
/// `kernels-wgpu`'s `selected` derives the same pair for the same reason and
/// refuses a third shape by name, which is what this now does.
///
/// # Errors
///
/// A rectangle that is neither, which is a plan this point cannot serve.
fn activation_strides(
    x_rows: i32,
    x_width: i32,
    tokens: i32,
    top_k: i32,
) -> Result<(i32, i32), Refusal> {
    let routed = tokens.checked_mul(top_k).ok_or(Refusal::Grid {
        what: "the route run, which is the tokens times the fan-out",
        at: i64::from(tokens) * i64::from(top_k),
    })?;
    if x_rows == tokens {
        return Ok((x_width, 0));
    }
    if x_rows == routed {
        let row = x_width.checked_mul(top_k).ok_or(Refusal::Grid {
            what: "the activation's row, which is the fan-out times its slot",
            at: i64::from(x_width) * i64::from(top_k),
        })?;
        return Ok((row, x_width));
    }
    Err(Refusal::Narrow {
        what: "the activation's rows, which are the fire's tokens or its routes and \
               neither here",
        at: i64::from(x_rows),
    })
}

#[kernels_macros::claims]
impl kernels::points::Moe for Ctx<'_> {
    fn topk_softmax<T: kernels::points::Scalar>(
        &self,
        logits: In<crate::points::Handle<T>>,
        experts: u32,
        top_k: u32,
        routes: Out<crate::points::Handle<i32>>,
        weights: Out<crate::points::Handle<f32>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>(
            "moe.topk_softmax, at an element this plane does not instantiate",
        )?;
        let row = logits.all("the router's logit row")?;
        let n = crate::points::stated("the expert count this router states", experts)?;
        if row.width != n {
            return Err(Refusal::Narrow {
                what: "the router's logit row, against the expert count it states",
                at: i64::from(row.width),
            });
        }

        self.fire(
            Fire::at(
                crate::plane::module_path("router_topk_f32w_bfloat16", self.best()),
                // THE `f32w` ARM, because the point declares the weight plane
                // `Out<Self::Tensor<f32>>`. `kernels-wgpu` fires its twin for
                // the same reason; this plane fired the bf16 one, which writes
                // half as many bytes into the slot and leaves the reader
                // decoding two weights as one float.
                "router_topk_f32w_bfloat16",
            )
            .apply(router_grid(row.rows)?),
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

    fn matmul_select<T: kernels::points::Scalar>(
        &self,
        x: In<crate::points::Handle<T>>,
        bank: Const<crate::points::Handle<T>>,
        routes: In<crate::points::Handle<i32>>,
        y: Out<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        use crate::points::Staged;

        crate::points::at_bf16::<T>(
            "moe.matmul_select, at an element this plane does not instantiate",
        )?;
        let act = x.all("the activation row this route selects against")?;
        let out = y.all("the routed result's row")?;
        let route = routes.all("the router's chosen experts")?;
        let (row_stride, slot_stride) =
            activation_strides(act.rows, act.width, route.rows, route.width)?;

        let bank = self.bank(bank)?;
        if bank.exponents.is_some() || bank.group != 64 || bank.bits != 4 {
            return Err(Refusal::Absent {
                what: "moe.matmul_select against a bank this plane does not \
                       stamp: `qmv_routed.slang` instantiates affine gs_64/b_4 \
                       alone, and the biased mxfp4 arm is `matmul_select_bias`",
            });
        }
        self.fire(
            Fire::at(
                crate::plane::module_path("affine_qmv_routed_bfloat16_gs_64_b_4", self.best()),
                "affine_qmv_routed_bfloat16_gs_64_b_4",
            )
            .apply(routed_qmv_grid(act.rows, out.width, route.width)?),
            &[
                bank.words.arg(),
                bank.scales.arg(),
                bank.biases.arg(),
                x.arg(),
                y.arg(),
                act.width.arg(),
                out.width.arg(),
                routes.arg(),
                slot_stride.arg(),
                row_stride.arg(),
                route.width.arg(),
            ],
        )
    }

    fn matmul_select_bias<T: kernels::points::Scalar, R: kernels::points::Repr>(
        &self,
        x: In<crate::points::Handle<T>>,
        bank: Const<crate::points::Planes<R>>,
        bias: Const<crate::points::Handle<T>>,
        routes: In<crate::points::Handle<i32>>,
        y: Out<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>(
            "moe.matmul_select_bias, at an element this plane does not instantiate",
        )?;
        let act = x.all("the activation row this route selects against")?;
        let out = y.all("the routed result's row")?;
        let route = routes.all("the router's chosen experts")?;
        let (row_stride, slot_stride) =
            activation_strides(act.rows, act.width, route.rows, route.width)?;
        let planes = bank.get();
        match R::FORM {
            kernels::points::Form::Mxfp4 => self.fire(
                Fire::at(
                    crate::plane::module_path(
                        "mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4",
                        self.best(),
                    ),
                    "mxfp4_qmv_routed_bias_bfloat16_gs_32_b_4",
                )
                .apply(routed_qmv_grid(act.rows, out.width, route.width)?),
                &[
                    planes.codes.arg(),
                    planes.scales.arg(),
                    x.arg(),
                    y.arg(),
                    act.width.arg(),
                    out.width.arg(),
                    bias.arg(),
                    routes.arg(),
                    slot_stride.arg(),
                    row_stride.arg(),
                    route.width.arg(),
                ],
            ),
        }
    }

    fn weighted_sum<T: kernels::points::Scalar>(
        &self,
        routed: In<crate::points::Handle<T>>,
        weights: In<crate::points::Handle<f32>>,
        y: Out<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>(
            "moe.weighted_sum, at an element this plane does not instantiate",
        )?;
        let src = routed.all("the routed expert rows")?;
        let out = y.all("the folded row")?;
        if out.rows <= 0 || src.rows % out.rows != 0 {
            return Err(Refusal::Narrow {
                what: "the routed rectangle, against the token rows it folds into",
                at: i64::from(src.rows),
            });
        }
        // IT STAMPS ONE NOW. This refused for want of the inverse permutation
        // `combine_sorted` folds through — `inv[row * top_k + e]`, written by
        // `route_sort`, which no point of this plane claims. That refusal was
        // right and the arm beside it was missing: `routed` arrives in (token,
        // slot) order, so slot `e` of token `n` is at row `n * k + e` and
        // there is nothing to permute. `route.slang`'s `PIE_EXPERT_COMBINE` is
        // that fold, and `kernels-wgpu` has had its twin all along.
        let top_k = src.rows / out.rows;
        if top_k <= 0 {
            return Err(Refusal::Empty {
                what: "the routed fanout",
            });
        }
        if src.width != out.width {
            return Err(Refusal::Narrow {
                what: "the routed row's width, which the fold does not change",
                at: i64::from(src.width),
            });
        }
        let w = weights.all("the router's weight plane")?;
        if w.rows != out.rows || w.width != top_k {
            return Err(Refusal::Narrow {
                what: "the weight plane, which is one weight per route",
                at: i64::from(w.width),
            });
        }
        self.fire(
            Fire::at(
                crate::plane::module_path("expert_combine", self.best()),
                "expert_combine",
            )
            .apply(elementwise_rows(out.width, out.rows)?),
            &[
                routed.arg(),
                weights.arg(),
                y.arg(),
                out.width.arg(),
                out.rows.arg(),
                top_k.arg(),
            ],
        )
    }

    fn sigmoid_gate_add<T: kernels::points::Scalar>(
        &self,
        routed: In<crate::points::Handle<T>>,
        shared: In<crate::points::Handle<T>>,
        gate: In<crate::points::Handle<T>>,
        y: Out<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        crate::points::at_bf16::<T>(
            "moe.sigmoid_gate_add, at an element this plane does not instantiate",
        )?;
        let row = routed.all("the routed sum's rectangle")?;
        self.fire(
            Fire::at(
                crate::plane::module_path("shared_expert_combine", self.best()),
                "shared_expert_combine",
            )
            .apply(elementwise_rows(row.width, row.rows)?),
            &[
                routed.arg(),
                shared.arg(),
                gate.arg(),
                y.arg(),
                row.width.arg(),
            ],
        )
    }
}
