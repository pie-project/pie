use kernels_xla::linear;
use poem_exec::{DispatchLinear, KernelError};
use poem_ir::{Linear, Operands};

use crate::run::Run;

impl DispatchLinear for Run<'_> {
    fn dispatch(&mut self, op: &Linear) -> Result<(), KernelError> {
        self.ctx().scope(op.name());
        self.linear(op).map_err(crate::error::kernel)
    }
}

impl Run<'_> {
    fn linear(&mut self, op: &Linear) -> Result<(), kernels_xla::Error> {
        match op {
            Linear::Relu2 { .. } => Err(kernels_xla::Error::Unsupported { op: op.name() }),
            // A product that quantizes its activations runs only on CUDA's
            // tcgen05 GEMMs.
            Linear::Matmul {
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
            } => Err(kernels_xla::Error::Unsupported { op: op.name() }),
            Linear::Matmul {
                act,
                w,
                y,
                quant: None,
                logits: false,
                ..
            } => match self.banked(*w) {
                Some(bank) => crate::lower::linear::quant(
                    self.ctx(),
                    false,
                    self.tensor(*act),
                    bank,
                    self.tensor(*y),
                ),

                None => match self.maybe_stored(*w) {
                    Some(block) => crate::lower::linear::kquant(
                        self.ctx(),
                        false,
                        self.tensor(*act),
                        block,
                        self.tensor(*y),
                    ),
                    None => crate::lower::lower(self.ctx(), |l| {
                        let act_v = l.read(self.tensor(*act))?;
                        let w_v = l.read(self.tensor(*w))?;
                        let o_y = self.tensor(*y);
                        if let Some(v) = linear::gemm::matmul(l.f(), act_v, w_v, o_y.landing())? {
                            l.land(o_y, v)?;
                        }
                        Ok(())
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
                Some(bank) => crate::lower::linear::quant(
                    self.ctx(),
                    true,
                    self.tensor(*act),
                    bank,
                    self.tensor(*y),
                ),

                None => match self.maybe_stored(*w) {
                    Some(block) => crate::lower::linear::kquant(
                        self.ctx(),
                        true,
                        self.tensor(*act),
                        block,
                        self.tensor(*y),
                    ),
                    None => crate::lower::lower(self.ctx(), |l| {
                        let act_v = l.read(self.tensor(*act))?;
                        let w_v = l.read(self.tensor(*w))?;
                        let o_y = self.tensor(*y);
                        if let Some(v) = linear::gemm::lm_head(l.f(), act_v, w_v, o_y.landing())? {
                            l.land(o_y, v)?;
                        }
                        Ok(())
                    }),
                },
            },

            // The forms no builder makes.
            Linear::Matmul { .. } => Err(kernels_xla::Error::Unsupported { op: op.name() }),
            Linear::Glu {
                packed: Some(packed),
                intermediate,
                y,
                gate: None,
                weights: None,
                limit: None,
                act: poem_ir::GluAct::Swiglu,
                ..
            } => crate::lower::lower(self.ctx(), |l| {
                let packed_v = l.read(self.tensor(*packed))?;
                let o_y = self.tensor(*y);
                let v = linear::mlp::swiglu(l.f(), packed_v, *intermediate, o_y.landing())?;
                l.land(o_y, v)
            }),
            Linear::Glu {
                packed: Some(packed),
                intermediate,
                limit: Some(limit),
                y,
                gate: None,
                weights: None,
                act: poem_ir::GluAct::Swiglu,
                ..
            } => crate::lower::lower(self.ctx(), |l| {
                let packed_v = l.read(self.tensor(*packed))?;
                let o_y = self.tensor(*y);
                let v = linear::mlp::swiglu_clamp(
                    l.f(),
                    packed_v,
                    *intermediate,
                    *limit,
                    o_y.landing(),
                )?;
                l.land(o_y, v)
            }),
            Linear::Glu {
                packed: Some(packed),
                intermediate,
                limit: Some(limit),
                y,
                gate: None,
                weights: None,
                act: poem_ir::GluAct::SwigluAlpha { alpha, .. },
                ..
            } => crate::lower::lower(self.ctx(), |l| {
                let packed_v = l.read(self.tensor(*packed))?;
                let o_y = self.tensor(*y);
                let v = linear::mlp::swiglu_clamp_alpha(
                    l.f(),
                    packed_v,
                    *intermediate,
                    *limit,
                    *alpha,
                    o_y.landing(),
                )?;
                l.land(o_y, v)
            }),

            Linear::Glu {
                gate: Some(gate),
                up: Some(up),
                limit: Some(limit),
                y,
                packed: None,
                weights: None,
                act: poem_ir::GluAct::Swiglu,
                ..
            } => crate::lower::lower(self.ctx(), |l| {
                let gate_v = l.read(self.tensor(*gate))?;
                let up_v = l.read(self.tensor(*up))?;
                let o_y = self.tensor(*y);
                let v =
                    linear::mlp::swiglu_clamp_split(l.f(), gate_v, up_v, *limit, o_y.landing())?;
                l.land(o_y, v)
            }),
            Linear::Glu {
                gate: Some(gate),
                up: Some(up),
                y,
                packed: None,
                weights: None,
                limit: None,
                act: poem_ir::GluAct::GegluTanh,
                ..
            } => crate::lower::lower(self.ctx(), |l| {
                let gate_v = l.read(self.tensor(*gate))?;
                let up_v = l.read(self.tensor(*up))?;
                let o_y = self.tensor(*y);
                let v = linear::mlp::geglu_tanh(l.f(), gate_v, up_v, o_y.landing())?;
                l.land(o_y, v)
            }),
            Linear::GeluTanh { x, y } => crate::lower::lower(self.ctx(), |l| {
                let x_v = l.read(self.tensor(*x))?;
                let o_y = self.tensor(*y);
                let v = linear::mlp::gelu_tanh(l.f(), x_v, o_y.landing())?;
                l.land(o_y, v)
            }),
            Linear::RelBias {
                x,
                w,
                heads,
                d_rel,
                extent,
                y,
            } => crate::lower::lower(self.ctx(), |l| {
                let x_v = l.read(self.tensor(*x))?;
                let w_v = l.read(self.tensor(*w))?;
                let o_y = self.tensor(*y);
                let v = linear::gemm::rel_bias(
                    l.f(),
                    x_v,
                    w_v,
                    *heads,
                    *d_rel,
                    *extent,
                    o_y.landing(),
                )?;
                l.land(o_y, v)
            }),
            Linear::MoeRoute {
                logits,
                bias,
                scale,
                experts,
                top_k,
                routes,
                weights,
                score: poem_ir::RouteScore::SigmoidSink { sink, scaling, .. },
                ..
            } => crate::lower::lower(self.ctx(), |l| {
                let logits_v = l.read(self.tensor(*logits))?;
                let correction_bias_v = l.read_opt(bias.map(|b| self.tensor(b)))?;
                let global_scale_v = l.read_opt(scale.map(|s| self.tensor(s)))?;
                let o_routes = self.tensor(*routes);
                let o_weights = self.tensor(*weights);
                let (v0, v1) = linear::moe::topk_sigmoid_sink(
                    l.f(),
                    logits_v,
                    correction_bias_v,
                    global_scale_v,
                    *experts,
                    *top_k,
                    *sink,
                    *scaling,
                    o_routes.landing(),
                    o_weights.landing(),
                )?;
                l.land(o_routes, v0)?;
                l.land(o_weights, v1)?;
                Ok(())
            }),
            Linear::Glu {
                packed: Some(packed),
                intermediate,
                y,
                gate: None,
                weights: None,
                limit: None,
                act: poem_ir::GluAct::GegluTanh,
                ..
            } => crate::lower::lower(self.ctx(), |l| {
                let packed_v = l.read(self.tensor(*packed))?;
                let o_y = self.tensor(*y);
                let v =
                    linear::mlp::geglu_tanh_packed(l.f(), packed_v, *intermediate, o_y.landing())?;
                l.land(o_y, v)
            }),
            Linear::Glu {
                packed: Some(packed),
                intermediate,
                y,
                gate: None,
                weights: None,
                limit: None,
                act: poem_ir::GluAct::Situ { beta, up_cap, .. },
                ..
            } => crate::lower::lower(self.ctx(), |l| {
                let packed_v = l.read(self.tensor(*packed))?;
                let o_y = self.tensor(*y);
                let v = linear::mlp::situ(
                    l.f(),
                    packed_v,
                    *intermediate,
                    *beta,
                    *up_cap,
                    o_y.landing(),
                )?;
                l.land(o_y, v)
            }),

            // The forms no builder makes.
            Linear::Glu { .. } => Err(kernels_xla::Error::Unsupported { op: op.name() }),
            Linear::MoeRoute {
                logits,
                experts,
                top_k,
                routes,
                weights,
                scale: None,
                score: poem_ir::RouteScore::Softmax,
                ..
            } => crate::lower::lower(self.ctx(), |l| {
                let logits_v = l.read(self.tensor(*logits))?;
                let o_routes = self.tensor(*routes);
                let o_weights = self.tensor(*weights);
                let (v0, v1) = linear::moe::topk_softmax(
                    l.f(),
                    logits_v,
                    *experts,
                    *top_k,
                    o_routes.landing(),
                    o_weights.landing(),
                )?;
                l.land(o_routes, v0)?;
                l.land(o_weights, v1)?;
                Ok(())
            }),
            Linear::MoeRoute {
                logits,
                scale: Some(scale),
                experts,
                top_k,
                routes,
                weights,
                score: poem_ir::RouteScore::Softmax,
                ..
            } => crate::lower::lower(self.ctx(), |l| {
                let logits_v = l.read(self.tensor(*logits))?;
                let scale_v = l.read(self.tensor(*scale))?;
                let o_routes = self.tensor(*routes);
                let o_weights = self.tensor(*weights);
                let (v0, v1) = linear::moe::topk_softmax_scaled(
                    l.f(),
                    logits_v,
                    scale_v,
                    *experts,
                    *top_k,
                    o_routes.landing(),
                    o_weights.landing(),
                )?;
                l.land(o_routes, v0)?;
                l.land(o_weights, v1)?;
                Ok(())
            }),

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
                Some(bias) => crate::lower::lower(self.ctx(), |l| {
                    let logits_v = l.read(self.tensor(*logits))?;
                    let bias_v = l.read(self.tensor(*bias))?;
                    let o_routes = self.tensor(*routes);
                    let o_weights = self.tensor(*weights);
                    let (v0, v1) = linear::moe::topk_sigmoid_biased(
                        l.f(),
                        logits_v,
                        bias_v,
                        *experts,
                        *top_k,
                        *renormalize,
                        *scaling,
                        o_routes.landing(),
                        o_weights.landing(),
                    )?;
                    l.land(o_routes, v0)?;
                    l.land(o_weights, v1)?;
                    Ok(())
                }),
                None => crate::lower::lower(self.ctx(), |l| {
                    let logits_v = l.read(self.tensor(*logits))?;
                    let o_routes = self.tensor(*routes);
                    let o_weights = self.tensor(*weights);
                    let (v0, v1) = linear::moe::topk_sigmoid(
                        l.f(),
                        logits_v,
                        *experts,
                        *top_k,
                        *renormalize,
                        *scaling,
                        o_routes.landing(),
                        o_weights.landing(),
                    )?;
                    l.land(o_routes, v0)?;
                    l.land(o_weights, v1)?;
                    Ok(())
                }),
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
            } => crate::lower::lower(self.ctx(), |l| {
                let logits_v = l.read(self.tensor(*logits))?;
                let bias_v = l.read(self.tensor(*bias))?;
                let o_routes = self.tensor(*routes);
                let o_weights = self.tensor(*weights);
                let (v0, v1) = linear::moe::predict_route(
                    l.f(),
                    logits_v,
                    bias_v,
                    *experts,
                    *top_k,
                    o_routes.landing(),
                    o_weights.landing(),
                )?;
                l.land(o_routes, v0)?;
                l.land(o_weights, v1)?;
                Ok(())
            }),

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
            } => crate::lower::lower(self.ctx(), |l| {
                let logits_v = l.read(self.tensor(*logits))?;
                let bias_v = l.read(self.tensor(*bias))?;
                let o_routes = self.tensor(*routes);
                let o_weights = self.tensor(*weights);
                let (v0, v1) = linear::moe::topk_sqrt_softplus(
                    l.f(),
                    logits_v,
                    bias_v,
                    *experts,
                    *top_k,
                    *renormalize,
                    *scaling,
                    o_routes.landing(),
                    o_weights.landing(),
                )?;
                l.land(o_routes, v0)?;
                l.land(o_weights, v1)?;
                Ok(())
            }),

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
            } => crate::lower::lower(self.ctx(), |l| {
                let ids_v = l.read(self.tensor(*ids))?;
                let tid2eid_v = l.read(self.tensor(*tid2eid))?;
                let logits_v = l.read(self.tensor(*logits))?;
                let o_routes = self.tensor(*routes);
                let o_weights = self.tensor(*weights);
                let (v0, v1) = linear::moe::hash_route(
                    l.f(),
                    ids_v,
                    tid2eid_v,
                    logits_v,
                    *vocab,
                    *top_k,
                    *renormalize,
                    *scaling,
                    o_routes.landing(),
                    o_weights.landing(),
                )?;
                l.land(o_routes, v0)?;
                l.land(o_weights, v1)?;
                Ok(())
            }),
            // The forms no builder makes.
            Linear::MoeRoute { .. } => Err(kernels_xla::Error::Unsupported { op: op.name() }),
            Linear::GroupRoutes { groups, routes } => crate::lower::lower(self.ctx(), |l| {
                let o_routes = self.tensor(*routes);
                let v = linear::moe::group_routes(l.f(), *groups, o_routes.landing())?;
                l.land(o_routes, v)
            }),

            Linear::LoraCorrect {
                x,
                bank_a,
                bank_b,
                routes,
                y: _,
                y_out,
            } => crate::lower::lower(self.ctx(), |l| {
                let x_v = l.read(self.tensor(*x))?;
                let bank_a_v = l.read(self.tensor(*bank_a))?;
                let routes_v = l.read(self.tensor(*routes))?;
                let bank_b_v = l.read(self.tensor(*bank_b))?;
                let y_v = l.read(self.tensor(*y_out))?;
                let o_y = self.tensor(*y_out);
                if let Some(v) =
                    linear::lora::correct(l.f(), x_v, bank_a_v, bank_b_v, routes_v, y_v)?
                {
                    l.land(o_y, v)?;
                }
                Ok(())
            }),
            Linear::MatmulGrouped {
                x,
                w,
                routes,
                groups,
                y,
            } => {
                // A readout-rowed `x` (an MTP head) against token-rowed
                // routes and result: the GPU kernel counts rows off `x`.
                let routes = self.first_rows(self.tensor(*routes), *x);
                let y = self.first_rows(self.tensor(*y), *x);
                crate::lower::lower(self.ctx(), |l| {
                    let xv = l.read(self.tensor(*x))?;
                    let rv = l.read(routes)?;
                    let yv = l.read(y)?;
                    let plane = match self.banked(*w) {
                        Some(bank) => linear::moe::GroupedPlane::Bank(l.bank(bank)?),
                        None => linear::moe::GroupedPlane::Dense(l.read(self.tensor(*w))?),
                    };
                    let tuning = l.tuning();
                    let v = linear::moe::matmul_grouped(l.f(), xv, plane, rv, *groups, yv, tuning)?;
                    l.land(y, v)
                })
            }
            Linear::MoeSelect {
                x,
                bank,
                routes,
                y,
                bias: None,
                quantized: false,
                quant: None,
                ..
            } => crate::lower::lower(self.ctx(), |l| {
                let x_v = l.read(self.tensor(*x))?;
                let routes_v = l.read(self.tensor(*routes))?;
                let y_v = l.read(self.tensor(*y))?;
                let bank_v = l.read(self.tensor(*bank))?;
                let o_y = self.tensor(*y);
                let tuning = l.tuning();
                let v = linear::moe::matmul_select(l.f(), x_v, bank_v, routes_v, y_v, tuning)?;
                l.land(o_y, v)
            }),
            Linear::MoeSelect {
                x,
                bank,
                bias: Some(bias),
                routes,
                y,
                quantized: false,
                quant: None,
                ..
            } => crate::lower::lower(self.ctx(), |l| {
                let x_v = l.read(self.tensor(*x))?;
                let routes_v = l.read(self.tensor(*routes))?;
                let bias_v = l.read(self.tensor(*bias))?;
                let y_v = l.read(self.tensor(*y))?;
                let bank_v = l.bank(self.planes(*bank))?;
                let o_y = self.tensor(*y);
                let tuning = l.tuning();
                let v = linear::moe::matmul_select_bias(
                    l.f(),
                    x_v,
                    bank_v,
                    bias_v,
                    routes_v,
                    y_v,
                    tuning,
                )?;
                l.land(o_y, v)
            }),
            Linear::MoeSelect {
                x,
                bank,
                routes,
                y,
                bias: None,
                quantized: true,
                quant: None,
                ..
            } => crate::lower::lower(self.ctx(), |l| {
                let x_v = l.read(self.tensor(*x))?;
                let routes_v = l.read(self.tensor(*routes))?;
                let y_v = l.read(self.tensor(*y))?;
                let bank_v = l.bank(self.planes(*bank))?;
                let o_y = self.tensor(*y);
                let tuning = l.tuning();
                let v =
                    linear::moe::matmul_select_quant(l.f(), x_v, bank_v, routes_v, y_v, tuning)?;
                l.land(o_y, v)
            }),
            // The forms no builder makes.
            Linear::MoeSelect { .. } => Err(kernels_xla::Error::Unsupported { op: op.name() }),
            Linear::MoeCombine {
                routed,
                weights: Some(weights),
                y,
                shared: None,
                ..
            } => crate::lower::lower(self.ctx(), |l| {
                let routed_v = l.read(self.tensor(*routed))?;
                let weights_v = l.read(self.tensor(*weights))?;
                let o_y = self.tensor(*y);
                let v = linear::moe::weighted_sum(l.f(), routed_v, weights_v, o_y.landing())?;
                l.land(o_y, v)
            }),
            // The forms no builder makes.
            Linear::MoeCombine { .. } => Err(kernels_xla::Error::Unsupported { op: op.name() }),
            Linear::MoeBiasSum {
                x,
                bias,
                routes,
                weights,
                y,
            } => crate::lower::lower(self.ctx(), |l| {
                let x_v = l.read(self.tensor(*x))?;
                let bias_v = l.read(self.tensor(*bias))?;
                let routes_v = l.read(self.tensor(*routes))?;
                let weights_v = l.read(self.tensor(*weights))?;
                let o_y = self.tensor(*y);
                let v =
                    linear::moe::bias_sum(l.f(), x_v, bias_v, routes_v, weights_v, o_y.landing())?;
                l.land(o_y, v)
            }),
            Linear::MoeSigmoidGateAdd {
                routed,
                shared,
                gate,
                y,
            } => crate::lower::lower(self.ctx(), |l| {
                let routed_v = l.read(self.tensor(*routed))?;
                let shared_v = l.read(self.tensor(*shared))?;
                let gate_v = l.read(self.tensor(*gate))?;
                let o_y = self.tensor(*y);
                let v = linear::moe::sigmoid_gate_add(
                    l.f(),
                    routed_v,
                    shared_v,
                    gate_v,
                    o_y.landing(),
                )?;
                l.land(o_y, v)
            }),
        }
    }
}
