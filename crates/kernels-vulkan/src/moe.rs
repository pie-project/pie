use crate::plane::{Bind, Const, Ctx, Fire, In, Out, elementwise_rows};
use kernels::plane::Refusal;

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
                crate::plane::module_path("router_topk_bfloat16", self.best()),
                "router_topk_bfloat16",
            )
            .apply(router_grid(row.rows)?),
            &[
                logits.arg(),
                routes.arg(),
                weights.arg(),
                experts.arg(),
                top_k.arg(),
                1u32.arg(),
                0u32.arg(),
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
                0i32.arg(),
                act.width.arg(),
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
                    0i32.arg(),
                    act.width.arg(),
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
        let _ = weights;
        Err(Refusal::Unstated {
            what: "the inverse permutation `combine_sorted` folds through: it \
                   reads `inv[row * top_k + e]` and `route_sort` is what writes \
                   one, which no point of this plane claims — so the slab is \
                   not a slab this fire has forgotten to size, it is a table \
                   nothing on this plane fills, and `route.slang` stamps no \
                   unsorted combine to fold without it",
        })
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
