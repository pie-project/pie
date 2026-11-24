use poem_exec::{DispatchLinear, KernelError};
use poem_ir::{Linear, Operands, ValueId};

use crate::backend::{Backend, Tactics as _};
use crate::run::Run;
use kernels_slang::linear::moe::select;

impl<B: Backend> DispatchLinear for Run<'_, B> {
    fn dispatch(&mut self, op: &Linear) -> Result<(), KernelError> {
        self.linear(op).map_err(crate::error::kernel)
    }
}

impl<B: Backend> Run<'_, B> {
    fn linear(&mut self, op: &Linear) -> Result<(), kernels_slang::Error> {
        match op {
            Linear::Relu2 { .. } => Err(kernels_slang::Error::Unsupported { op: op.name() }),
            Linear::Matmul {
                act,
                w,
                y,
                quant: None,
                logits: false,
                ..
            } if self.tensor(*act).dtype == poem_ir::Dtype::F32 && self.banked(*w).is_none() => {
                kernels_slang::linear::gemm::lane(
                    self.ctx,
                    "linear.matmul",
                    self.tensor(*act),
                    self.tensor(*w),
                    self.tensor(*y),
                )
            }
            Linear::Matmul {
                act,
                w,
                y,
                quant: None,
                logits: false,
                ..
            } => match self.banked(*w) {
                Some(bank) => {
                    let capacity = self.capacity(*act).min(self.capacity(*y));
                    let (act, y) = (self.tensor(*act), self.tensor(*y));
                    let ladder = self.qmm_ladder(act, y, bank, capacity);
                    kernels_slang::linear::qmm::matmul(self.ctx, &ladder, act, bank, y, capacity)
                }

                None => match self.maybe_stored(*w) {
                    Some(block) => kernels_slang::linear::kquant::matmul(
                        self.ctx,
                        self.tuning().kquant(),
                        self.tensor(*act),
                        block,
                        self.tensor(*y),
                    ),
                    None => self.gemm(*act, *w, |tactic| {
                        let ([act, y], window) = self.windowed([*act, *y]);
                        kernels_slang::linear::gemm::matmul(
                            self.ctx,
                            tactic,
                            act,
                            self.tensor(*w),
                            y,
                            window,
                        )
                    }),
                },
            },
            Linear::Matmul {
                act,
                w,
                y,
                quant: None,
                logits: true,
                ..
            } => match self.banked(*w) {
                Some(bank) => {
                    let capacity = self.capacity(*act).min(self.capacity(*y));
                    let (act, y) = (self.tensor(*act), self.tensor(*y));
                    let ladder = self.qmm_ladder(act, y, bank, capacity);
                    kernels_slang::linear::qmm::lm_head(self.ctx, &ladder, act, bank, y, capacity)
                }

                None => match self.maybe_stored(*w) {
                    Some(block) => kernels_slang::linear::kquant::lm_head(
                        self.ctx,
                        self.tuning().kquant(),
                        self.tensor(*act),
                        block,
                        self.tensor(*y),
                    ),
                    None => self.gemm(*act, *w, |tactic| {
                        let ([act, y], window) = self.windowed([*act, *y]);
                        kernels_slang::linear::gemm::lm_head(
                            self.ctx,
                            tactic,
                            act,
                            self.tensor(*w),
                            y,
                            window,
                        )
                    }),
                },
            },

            Linear::Glu {
                packed: Some(packed),
                intermediate,
                y,
                gate: None,
                weights: None,
                limit: None,
                act: poem_ir::GluAct::Swiglu,
                ..
            } => kernels_slang::linear::glu::swiglu(
                self.ctx,
                self.tensor(*packed),
                *intermediate,
                self.tensor(*y),
            ),
            Linear::Glu {
                packed: Some(packed),
                intermediate,
                limit: Some(limit),
                y,
                gate: None,
                weights: None,
                act: poem_ir::GluAct::Swiglu,
                ..
            } => kernels_slang::linear::glu::swiglu_clamp(
                self.ctx,
                self.tensor(*packed),
                *intermediate,
                *limit,
                self.tensor(*y),
            ),
            // An unclamped unit clamps at the largest finite f32, which keeps
            // every finite value (a shader may assume no infinity).
            Linear::Glu {
                packed: Some(packed),
                intermediate,
                limit,
                y,
                gate: None,
                weights: None,
                act: poem_ir::GluAct::SwigluAlpha { alpha, .. },
                ..
            } => kernels_slang::linear::glu::swiglu_clamp_alpha(
                self.ctx,
                self.tensor(*packed),
                *intermediate,
                limit.unwrap_or(f32::MAX),
                *alpha,
                self.tensor(*y),
            ),

            // Legs the pack left apart: the packed forms over a gate and an
            // up plane, an unclamped SwiGLU clamped at the largest finite f32.
            Linear::Glu {
                gate: Some(gate),
                up: Some(up),
                limit,
                y,
                packed: None,
                weights: None,
                act: poem_ir::GluAct::Swiglu,
                ..
            } => kernels_slang::linear::glu::swiglu_clamp_split(
                self.ctx,
                self.tensor(*gate),
                self.tensor(*up),
                limit.unwrap_or(f32::MAX),
                self.tensor(*y),
            ),
            Linear::Glu {
                gate: Some(gate),
                up: Some(up),
                limit,
                y,
                packed: None,
                weights: None,
                act: poem_ir::GluAct::SwigluAlpha { alpha, .. },
                ..
            } => kernels_slang::linear::glu::swiglu_clamp_alpha_split(
                self.ctx,
                self.tensor(*gate),
                self.tensor(*up),
                limit.unwrap_or(f32::MAX),
                *alpha,
                self.tensor(*y),
            ),
            Linear::Glu {
                gate: Some(gate),
                up: Some(up),
                y,
                packed: None,
                weights: None,
                limit: None,
                act: poem_ir::GluAct::Situ { beta, up_cap, .. },
                ..
            } => kernels_slang::linear::glu::situ_split(
                self.ctx,
                self.tensor(*gate),
                self.tensor(*up),
                *beta,
                *up_cap,
                self.tensor(*y),
            ),
            Linear::Glu {
                gate: Some(gate),
                up: Some(up),
                y,
                packed: None,
                weights: None,
                limit: None,
                act: poem_ir::GluAct::GegluTanh,
                ..
            } => kernels_slang::linear::glu::geglu_tanh(
                self.ctx,
                self.tensor(*gate),
                self.tensor(*up),
                self.tensor(*y),
            ),
            Linear::GeluTanh { x, y } => {
                kernels_slang::linear::glu::gelu_tanh(self.ctx, self.tensor(*x), self.tensor(*y))
            }
            // A product that quantizes its activations runs only on CUDA's
            // tcgen05 GEMMs.
            Linear::RelBias { .. }
            | Linear::MoeRoute {
                score: poem_ir::RouteScore::SigmoidSink { .. },
                ..
            }
            | Linear::Matmul {
                quant: Some(_),
                logits: false,
                ..
            }
            | Linear::MoeSelect {
                bias: None,
                quantized: true,
                quant: Some(_),
                ..
            }
            | Linear::Glu {
                packed: None,
                gate: Some(_),
                weights: Some(_),
                limit: Some(_),
                act: poem_ir::GluAct::Swiglu,
                ..
            }
            | Linear::Glu {
                packed: Some(_),
                gate: None,
                weights: Some(_),
                limit: Some(_),
                act: poem_ir::GluAct::Swiglu,
                ..
            }
            | Linear::MoeCombine {
                weights: None,
                shared: Some(_),
                ..
            } => Err(kernels_slang::Error::Unsupported { op: op.name() }),
            // The forms no builder makes.
            Linear::Matmul { .. } => Err(kernels_slang::Error::Unsupported { op: op.name() }),
            Linear::Glu {
                packed: Some(packed),
                intermediate,
                y,
                gate: None,
                weights: None,
                limit: None,
                act: poem_ir::GluAct::GegluTanh,
                ..
            } => kernels_slang::linear::glu::geglu_tanh_packed(
                self.ctx,
                self.tensor(*packed),
                *intermediate,
                self.tensor(*y),
            ),
            Linear::Glu {
                packed: Some(packed),
                intermediate,
                y,
                gate: None,
                weights: None,
                limit: None,
                act: poem_ir::GluAct::Situ { beta, up_cap, .. },
                ..
            } => kernels_slang::linear::glu::situ(
                self.ctx,
                self.tensor(*packed),
                *intermediate,
                *beta,
                *up_cap,
                self.tensor(*y),
            ),

            // The forms no builder makes.
            Linear::Glu { .. } => Err(kernels_slang::Error::Unsupported { op: op.name() }),
            Linear::MoeRoute {
                logits,
                experts,
                top_k,
                routes,
                weights,
                scale: None,
                score: poem_ir::RouteScore::Softmax,
                ..
            } => kernels_slang::linear::moe::route::topk_softmax(
                self.ctx,
                self.tensor(*logits),
                *experts,
                *top_k,
                self.tensor(*routes),
                self.tensor(*weights),
            ),
            Linear::MoeRoute {
                logits,
                scale: Some(scale),
                experts,
                top_k,
                routes,
                weights,
                score: poem_ir::RouteScore::Softmax,
                ..
            } => kernels_slang::linear::moe::route::topk_softmax_scaled(
                self.ctx,
                self.tensor(*logits),
                self.tensor(*scale),
                *experts,
                *top_k,
                self.tensor(*routes),
                self.tensor(*weights),
            ),

            Linear::MoeRoute {
                logits,
                bias,
                experts,
                top_k,
                routes,
                weights,
                hint: _,
                score:
                    poem_ir::RouteScore::Sigmoid {
                        renormalize,
                        scaling,
                        ..
                    },
                ..
            } => match bias {
                Some(bias) => kernels_slang::linear::moe::route::topk_sigmoid_biased(
                    self.ctx,
                    self.tensor(*logits),
                    self.tensor(*bias),
                    *experts,
                    *top_k,
                    *renormalize,
                    *scaling,
                    self.tensor(*routes),
                    self.tensor(*weights),
                ),
                None => kernels_slang::linear::moe::route::topk_sigmoid(
                    self.ctx,
                    self.tensor(*logits),
                    *experts,
                    *top_k,
                    *renormalize,
                    *scaling,
                    self.tensor(*routes),
                    self.tensor(*weights),
                ),
            },
            Linear::MoeRoute {
                logits,
                bias: Some(bias),
                experts,
                top_k,
                routes,
                weights,
                score: poem_ir::RouteScore::Predict,
                ..
            } => kernels_slang::linear::moe::route::predict_route(
                self.ctx,
                self.tensor(*logits),
                self.tensor(*bias),
                *experts,
                *top_k,
                self.tensor(*routes),
                self.tensor(*weights),
            ),

            Linear::MoeRoute {
                logits,
                bias: Some(bias),
                experts,
                top_k,
                hint: _,
                routes,
                weights,
                score:
                    poem_ir::RouteScore::SqrtSoftplus {
                        renormalize,
                        scaling,
                        ..
                    },
                ..
            } => kernels_slang::linear::moe::route::topk_sqrt_softplus(
                self.ctx,
                self.tensor(*logits),
                self.tensor(*bias),
                *experts,
                *top_k,
                *renormalize,
                *scaling,
                self.tensor(*routes),
                self.tensor(*weights),
            ),

            Linear::MoeRoute {
                ids: Some(ids),
                table: Some(tid2eid),
                logits,
                experts: _,
                top_k,
                routes,
                weights,
                score:
                    poem_ir::RouteScore::Hash {
                        vocab,
                        renormalize,
                        scaling,
                        ..
                    },
                ..
            } => kernels_slang::linear::moe::route::hash_route(
                self.ctx,
                self.tensor(*ids),
                self.tensor(*tid2eid),
                self.tensor(*logits),
                *vocab,
                *top_k,
                *renormalize,
                *scaling,
                self.tensor(*routes),
                self.tensor(*weights),
            ),
            // The forms no builder makes.
            Linear::MoeRoute { .. } => Err(kernels_slang::Error::Unsupported { op: op.name() }),
            Linear::GroupRoutes { groups, routes } => {
                kernels_slang::linear::moe::route::group_routes(
                    self.ctx,
                    *groups,
                    self.tensor(*routes),
                )
            }

            Linear::MatmulGrouped {
                x,
                w,
                routes,
                groups,
                y,
            } => kernels_slang::linear::moe::select::matmul_grouped(
                self.ctx,
                self.tuning().routed_tier(self.tier()),
                self.tensor(*x),
                match self.banked(*w) {
                    Some(bank) => kernels_slang::linear::moe::select::GroupedPlane::Bank(bank),
                    None => {
                        kernels_slang::linear::moe::select::GroupedPlane::Dense(self.tensor(*w))
                    }
                },
                self.tensor(*routes),
                *groups,
                self.tensor(*y),
            ),
            Linear::MoeSelect {
                x,
                bank,
                routes,
                y,
                bias: None,
                quantized: false,
                quant: None,
                ..
            } => {
                const OP: &str = "linear.moe_select";
                let routes = self.tensor(*routes);
                match self.expert_table(OP, Some(*bank), None, routes)? {
                    Some(table) => kernels_slang::linear::moe::select::matmul_select_tabled(
                        self.ctx,
                        self.tuning().routed_tier(self.tier()),
                        self.tensor(*x),
                        &table,
                        routes,
                        self.tensor(*y),
                    ),
                    None => kernels_slang::linear::moe::select::matmul_select(
                        self.ctx,
                        self.tuning().routed_tier(self.tier()),
                        self.tensor(*x),
                        self.tensor(*bank),
                        routes,
                        self.tensor(*y),
                    ),
                }
            }

            Linear::MoeSelect {
                x,
                bank,
                bias,
                routes,
                y,
                quantized,
                quant: None,
                ..
            } if bias.is_some() != *quantized => {
                const OP: &str = "linear.moe_select";
                let routed = self.tensor(*routes);
                if let Some(table) = self.expert_table(OP, Some(*bank), *bias, routed)? {
                    let scheme = self.scheme(*bank);
                    let experts = self.experts(*routes);
                    let capacity = self.capacity(*y);
                    let (x, y) = (self.tensor(*x), self.tensor(*y));
                    return self.routed(
                        capacity,
                        y.rows,
                        experts,
                        |scratch, batch| {
                            select::matmul_select_batched_tabled(
                                self.ctx,
                                OP,
                                x,
                                scheme,
                                bias.is_some(),
                                &table,
                                routed,
                                experts,
                                scratch,
                                y,
                                batch,
                            )
                        },
                        || {
                            select::matmul_select_quant_tabled(
                                self.ctx,
                                self.tuning().routed_tier(self.tier()),
                                x,
                                scheme,
                                bias.is_some(),
                                &table,
                                routed,
                                y,
                            )
                        },
                    );
                }
                let planes = self.planes(*bank);
                let bias_plane = bias.map(|b| self.tensor(b));
                let experts = self.experts(*routes);
                let capacity = self.capacity(*y);
                let (x, y) = (self.tensor(*x), self.tensor(*y));
                self.routed(
                    capacity,
                    y.rows,
                    experts,
                    |scratch, batch| {
                        select::matmul_select_batched(
                            self.ctx, OP, x, planes, bias_plane, routed, experts, scratch, y, batch,
                        )
                    },
                    || match bias_plane {
                        Some(bias_plane) => select::matmul_select_bias(
                            self.ctx,
                            self.tuning().routed_tier(self.tier()),
                            x,
                            planes,
                            bias_plane,
                            routed,
                            y,
                        ),
                        None => select::matmul_select_quant(
                            self.ctx,
                            self.tuning().routed_tier(self.tier()),
                            x,
                            planes,
                            routed,
                            y,
                        ),
                    },
                )
            }

            // The forms no builder makes.
            Linear::MoeSelect { .. } => Err(kernels_slang::Error::Unsupported { op: op.name() }),
            Linear::LoraCorrect {
                x,
                bank_a,
                bank_b,
                routes,
                y: _,
                y_out,
            } => kernels_slang::linear::lora::correct(
                self.ctx,
                self.tuning().lora_tier(),
                self.tensor(*x),
                self.tensor(*bank_a),
                self.tensor(*bank_b),
                self.tensor(*routes),
                self.tensor(*y_out),
            ),
            Linear::MoeCombine {
                routed,
                weights: Some(weights),
                y,
                shared: None,
                ..
            } => kernels_slang::linear::moe::combine::weighted_sum(
                self.ctx,
                self.tensor(*routed),
                self.tensor(*weights),
                self.tensor(*y),
            ),

            // The forms no builder makes.
            Linear::MoeCombine { .. } => Err(kernels_slang::Error::Unsupported { op: op.name() }),
            Linear::MoeBiasSum {
                x,
                bias,
                routes,
                weights,
                y,
            } => {
                let routes = self.tensor(*routes);
                match self.expert_table("linear.moe_bias_sum", None, Some(*bias), routes)? {
                    Some(table) => kernels_slang::linear::moe::combine::bias_sum_tabled(
                        self.ctx,
                        self.tensor(*x),
                        &table,
                        routes,
                        self.tensor(*weights),
                        self.tensor(*y),
                    ),
                    None => kernels_slang::linear::moe::combine::bias_sum(
                        self.ctx,
                        self.tensor(*x),
                        self.tensor(*bias),
                        routes,
                        self.tensor(*weights),
                        self.tensor(*y),
                    ),
                }
            }
            Linear::MoeSigmoidGateAdd {
                routed,
                shared,
                gate,
                y,
            } => kernels_slang::linear::moe::combine::sigmoid_gate_add(
                self.ctx,
                self.tensor(*routed),
                self.tensor(*shared),
                self.tensor(*gate),
                self.tensor(*y),
            ),
        }
    }
}

impl<B: Backend> Run<'_, B> {
    /// A dense matmul from `act` by the weight `w` at its plane's tactic at
    /// the rows ([`crate::dense`]). A naming sink is told every way it goes: `matmul`
    /// at each tactic some fire of up to the load's most rows takes, each
    /// naming the vector kernels besides its tiled ones.
    fn gemm(
        &self,
        act: ValueId,
        w: ValueId,
        matmul: impl Fn(kernels_slang::linear::gemm::Tactic) -> Result<(), kernels_slang::Error>,
    ) -> Result<(), kernels_slang::Error> {
        let Some(names) = self.ctx.naming() else {
            let (rows, plane) = (self.tactic_rows(act), self.tensor(w));
            let tactic = self.dense().at(rows, plane.rows, plane.width);
            self.pick(|| {
                format!(
                    "dense at {rows} rows of a {}x{} plane: {}",
                    plane.rows,
                    plane.width,
                    crate::dense::name(&tactic)
                )
            });
            return matmul(tactic);
        };
        // The walk fires at one rung; a fire of the load brings up to the
        // most rows its rungs hold.
        let rows = names.most_rows().max(self.tensor(act).rows);
        names.fork();
        for (way, tactic) in self
            .tuning()
            .gemm_tactics(self.tier(), rows)
            .into_iter()
            .enumerate()
        {
            if way > 0 {
                names.or();
            }
            matmul(tactic)?;
        }
        names.join();
        Ok(())
    }

    /// The `kernel.qmm` rungs a matmul tries, in order, at its rows; to a
    /// naming sink every rung a batch of up to the load's most rows reaches.
    fn qmm_ladder(
        &self,
        act: kernels_slang::Tensor,
        y: kernels_slang::Tensor,
        bank: kernels_slang::Bank,
        capacity: u32,
    ) -> Vec<kernels_slang::linear::qmm::Qmm> {
        match self.ctx.naming() {
            Some(names) => {
                self.tuning()
                    .qmm_rungs(names.most_rows().max(capacity), y.width, act.width, bank)
            }
            None => self.tuning().qmm_ladder(act.rows, y.width, act.width, bank),
        }
    }

    /// The rows a matmul of `act` picks its kernel by: its own where they
    /// are its commands' constants, else the power of two that holds them
    /// (within what `act` holds), so every fire of a body whose rows vary
    /// between those picks the same kernel.
    fn tactic_rows(&self, act: poem_ir::ValueId) -> u32 {
        let rows = self.tensor(act).rows;
        if self.ctx.fixed_rows() {
            rows
        } else {
            rows.next_power_of_two().min(self.capacity(act).max(rows))
        }
    }
}

impl<B: Backend> Run<'_, B> {
    /// A routed matmul of `pairs` pairs: `batched` over sorted expert tiles
    /// at the device's tile rows, sort width and tier where the load
    /// reserved the routed scratch and the shape serves it (it answers
    /// whether it fired), else `select`, a pair a row. A naming sink is told
    /// every way a route of up to `capacity` pairs goes: `batched` at each
    /// tile some count of them sorts into, over the fewest rows it sorts,
    /// and `select`.
    fn routed(
        &self,
        capacity: u32,
        pairs: u32,
        experts: u32,
        batched: impl Fn(select::RoutedScratch, select::Batch) -> Result<bool, kernels_slang::Error>,
        select: impl FnOnce() -> Result<(), kernels_slang::Error>,
    ) -> Result<(), kernels_slang::Error> {
        let Some(scratch) = self.routed_scratch() else {
            return select();
        };
        let batch = self.tuning().batch(pairs, experts);
        let Some(names) = self.ctx.naming() else {
            if batched(scratch, batch)? {
                return Ok(());
            }
            return select();
        };
        names.fork();
        for tile in self.tuning().moe_tiles(capacity, experts) {
            batched(
                scratch,
                select::Batch {
                    tile,
                    padded: tile,
                    ..batch
                },
            )?;
            names.or();
        }
        select()?;
        names.join();
        Ok(())
    }
}
