use crate::routine::{Bind, Const, Ctx, Fire, In, Out, elementwise_rows};
use kernels::routine::Refusal;

/// The routed matvec's grid: one 32-lane group per (row, output, slot).
///
/// `qmv_routed.slang` runs `[numthreads(32, 8, 1)]` and addresses its
/// selection as `sel = row * slots_per_row + slot`, so the fan-out rides z
/// and the token rows are multiplied out onto x with the lane count. This is
/// the arm the claimed `matmul_select` fires.
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

/// The SORTED arm's grid: column tiles by row tiles, at the 16x16
/// workgroup `qmm_t_routed.slang` declares.
///
/// Kept without a caller because the arm it grids is the one this plane
/// cannot fire yet — see the impl header. The tiles are ceil-divided, not
/// exact: the routed gemm pads its row count to whole tiles at the SORT,
/// which is what `route_sort`'s `padded` word is for.
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

/// The router's grid: one 1024-wide workgroup per token row.
///
/// `route.slang` reduces a whole logit row inside one group, so the expert
/// count never reaches the grid — it is a push word the group loops over.
pub fn router_grid(rows: i32) -> Result<[u32; 3], Refusal> {
    if rows <= 0 {
        return Err(Refusal::Empty { what: "rows" });
    }
    Ok([1024, rows.unsigned_abs(), 1])
}

/// The `Moe` family, claimed. Four of seven points land, one is written
/// whole against a missing instantiation, and two are measured backlog
/// rows.
///
/// # The routed matmuls take the UNSORTED arm, and that is the whole
/// reason they can be claimed
///
/// This plane carries two routed-matmul shapes.
/// `affine_qmm_t_routed_bfloat16_gs_*_b_*_bm_*_bn_*` — seventy-two
/// entrypoints over `(group, bits, tile_m, tile_n)` — is the throughput
/// arm, and it wants rows SORTED by expert. That is a four-launch pipeline
/// and the order is the knowledge: `route_sort` writes the permutation, the
/// row/tile expert maps and the INVERSE map in one 1024-wide group;
/// `route_gather` materialises the permuted activations padded to whole
/// tiles; the tiled gemm runs one expert per tile off `tile_expert`
/// ([`routed_qmm_grid`]); and `combine_sorted` unpermutes through `inv`.
/// Five buffers of staging that no statement carries and this plane has no
/// arena door to mint (see [`crate::points::Staged::scratch`]).
///
/// `affine_qmv_routed_bfloat16_gs_64_b_4` needs none of it. Its addressing is
/// `sel = row * slots_per_row + slot`, `expert = expert_ids[sel]`,
/// `x_base = row * x_row_stride + slot * x_slot_stride`, and it writes
/// `y[sel * out_vec_size + out_row]` — which IS the declaration's layout,
/// exactly: `routes` is `[tokens, top_k]` read flat, `x` is one row per
/// token (so `x_slot_stride` is zero and the same activation feeds every
/// slot), and `y` is the `[tokens * top_k, N]` fan-out
/// `.wiki/baker-todo.md`'s MoE rows algebra settled — "k READ off the
/// routes width never restated". So `matmul_select` is one launch against
/// the operands the statement already carries.
///
/// The tiled arm is a PERFORMANCE choice this plane cannot make yet, not a
/// correctness one. That is worth stating plainly: nothing below is
/// waiting on shader work, only on a scratch door.
///
/// # SEAM: `Bank<R: Repr>`, twice
///
/// Both routed matmuls read a quantised bank — three planes and two
/// numbers behind one `Const` — and this plane has no dense expert gemm at
/// all. See [`crate::points::Bank`]. The two bodies also constrain the
/// bank harder than the ledger type does: `qmv_routed.slang` stamps
/// exactly `gs_64_b_4` affine and `gs_32_b_4` mxfp4, so a bank at any
/// other `(group, bits)` is refused by name rather than mis-unpacked.
///
/// # Two points stay on the floor's default body
///
/// * `moe.topk_sigmoid` and `moe.topk_sqrt_softplus` — `route.slang`
///   stamps one scoring, `PIE_ROUTER_TOPK`, and it is softmax. The
///   `PIE_SCALED` variant multiplies by a per-expert plane afterwards,
///   which is neither point: `topk_sigmoid` renormalises the KEPT weights
///   and `topk_sqrt_softplus` adds a learned bias BEFORE the selection, so
///   both change which experts are chosen. Two instantiations of that
///   shader close them.
#[kernels_macros::claims]
impl kernels::points::Moe for Ctx<'_> {
    /// Softmax over all the router logits, then the top `top_k`.
    ///
    /// `softmax_over_all = 1` is the point's own name spelled as a push
    /// word: the shader's zero reading normalises over the KEPT logits
    /// instead, which is a different denominator and a different point
    /// (one this plane does not stamp — see the impl header).
    ///
    /// `logits_pitch = 0` for the reason `route.slang` states: zero means
    /// "the pitch IS `n_experts`", and a statement hands over a DENSE
    /// `[tokens, experts]` rectangle, so restating the count would be a
    /// second place for the same number to be wrong.
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
        // SEAM: `weights` is declared `Out<Tensor<f32>>` and
        // `route.slang` writes it through `PIE_BUFFER_RW(2,
        // expert_weights)` — `PIE_ACT`, `bfloat16`. The pair is
        // self-consistent on this plane, because `combine_sorted` reads it
        // back at the same element; what disagrees is the FLOOR, which
        // says a routing weight is accumulator arithmetic. Closing it is a
        // `_f32_weights` instantiation of this shader and of the two
        // combines, which is shader work.
        self.fire(
            Fire::at(
                crate::routine::module_path("router_topk_bfloat16", self.best()),
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

    /// `y[r] = x[r] @ bank[routes[r]]`, one matvec per route.
    ///
    /// See the impl header for why this is one launch and not five. The
    /// three numbers the shader wants beyond the operands are all read:
    /// `slots_per_row` is `routes`' width (the fan-out is a SET, never
    /// restated), `x_row_stride` is the activation's own row, and
    /// `x_slot_stride` is ZERO because every slot of a token reads the
    /// same activation — which is what `matmul_select`'s `x` being one row
    /// per token means.
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
        // SEAM: the bank triple. See [`crate::points::Bank`].
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
                crate::routine::module_path("affine_qmv_routed_bfloat16_gs_64_b_4", self.best()),
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

    /// [`Self::matmul_select`] with the expert's own bias row added.
    ///
    /// The bank slot is the floor's `Bank<R>` — [`crate::points::Planes`]
    /// here — so the body picks the instantiation by `R::FORM`, not by a
    /// staged flag. `qmv_routed.slang` also stamps an affine `gs_64/b_4`
    /// `_bias` form; it becomes reachable when the repr axis grows an
    /// affine member, and not before.
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
                    crate::routine::module_path(
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

    /// Fold a token's `top_k` expert rows into one, by the router's
    /// weights.
    ///
    /// SEAM, AND IT IS ONE INSTANTIATION WIDE. `combine_sorted` reads
    /// `y[inv[row * k + e] * width + c]` — the inverse of the sort's
    /// permutation. With the SORTED matmul arm that plane is real; with
    /// the arm [`Self::matmul_select`] actually fires it is the IDENTITY,
    /// because `qmv_routed` already writes row `row * k + e` at
    /// `row * k + e`. So what is missing is either a `PIE_COMBINE_DENSE`
    /// instantiation with no `inv` binding, or an identity plane in
    /// scratch — and the second is the cheaper statement of the same fact
    /// only until the first exists.
    ///
    /// `top_k` is READ and never stated: the routed rectangle holds
    /// `tokens * top_k` rows against the result's `tokens`, so the fan-out
    /// is their ratio. That is the MoE rows algebra, applied.
    fn weighted_sum<T: kernels::points::Scalar>(
        &self,
        routed: In<crate::points::Handle<T>>,
        weights: In<crate::points::Handle<f32>>,
        y: Out<crate::points::Handle<T>>,
    ) -> Result<(), Refusal> {
        use crate::points::Staged;

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
        let top_k = src.rows / out.rows;
        // SEAM: the inverse permutation. See this method's doc — the plane
        // this asks for is the identity under the arm `matmul_select`
        // fires, and the honest fix is a combine that does not bind one.
        let inv = self.scratch::<i32>("moe.route_inv", i64::from(src.rows))?;
        self.fire(
            Fire::at(
                crate::routine::module_path("combine_sorted", self.best()),
                "combine_sorted",
            )
            .apply(elementwise_rows(out.width, out.rows)?),
            &[
                routed.arg(),
                weights.arg(),
                y.arg(),
                inv.arg(),
                out.width.arg(),
                top_k.arg(),
                // Zero means "the pitch IS `width`", which a dense result
                // is. `route.slang` states the convention.
                0u32.arg(),
            ],
        )
    }

    /// `y = routed + shared * sigmoid(gate)`.
    ///
    /// The gate is one value per ROW and the shader reads it as `gate[r]`
    /// in this unstrided arm — `route.slang` records the bug that came of
    /// folding that index into the data base, and the strided sibling is
    /// not a copy of this one for that exact reason. A statement hands
    /// over a dense `[tokens, 1]` column, so this is the arm.
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
                crate::routine::module_path("shared_expert_combine", self.best()),
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
