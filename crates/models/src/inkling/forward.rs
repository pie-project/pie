use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, Predicate, Request, Value, Weight, ops, seam,
};

use super::model::{Layer, Mlp, Model, Reading};

pub struct Facts {
    pub qo_one: bool,
    pub has_adapter: bool,
}

impl Facts {
    pub fn qo_one() -> Predicate {
        Predicate::fact(0)
    }

    pub fn has_adapter() -> Predicate {
        Predicate::fact(1)
    }
}

impl Classify for Facts {
    fn of(r: &Request) -> Facts {
        Facts {
            qo_one: r.query_len() == 1,
            has_adapter: r.has_adapter(),
        }
    }

    fn word(&self) -> u64 {
        u64::from(self.qo_one) | (u64::from(self.has_adapter) << 1)
    }
}

impl ForwardHybrid for Model {
    type Facts = Facts;

    fn caches(&self) -> HybridSpec {
        let mut c = HybridSpec::new();
        let kv = c.kv_space(self.kv);
        let hidden = u64::from(self.hidden);
        let taps = u64::from(self.conv_width);
        for w in &self.layers {
            let plane = u64::from(w.kv_heads) * u64::from(self.head_dim);
            c.kv(kv, w.kv.clone(), [plane, plane]);
            c.state(w.k_state.clone(), [taps, plane], Dtype::Bf16);
            c.state(w.v_state.clone(), [taps, plane], Dtype::Bf16);
            c.state(w.attn_state.clone(), [taps, hidden], Dtype::Bf16);
            c.state(w.mlp_state.clone(), [taps, hidden], Dtype::Bf16);
        }
        c
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let m = self;
        let d = m.head_dim;
        let one = Facts::qo_one();

        let (input_d, input_p) = inputs.split(&one);
        let geometry = [
            (
                m.layers
                    .iter()
                    .find(|w| w.reading as usize == 0)
                    .map_or(0, |w| w.kv_heads),
                Some(m.window),
            ),
            (
                m.layers
                    .iter()
                    .find(|w| w.reading as usize == 1)
                    .map_or(0, |w| w.kv_heads),
                None,
            ),
        ];
        let plan_d = geometry.map(|(kv_heads, win)| {
            (kv_heads > 0).then(|| ops::attn::plan_decode(&input_d, m.heads, kv_heads, d, win))
        });
        let plan_p = geometry.map(|(kv_heads, win)| {
            (kv_heads > 0).then(|| ops::attn::plan_prefill(&input_p, m.heads, kv_heads, d, win))
        });

        let ids = inputs.tokens();
        let mut y = ops::elemwise::rmsnorm(
            &ops::layout::embed(&ids, &m.embed, m.vocab),
            &m.embed_norm,
            m.norm_eps,
        );

        let routes = inputs.adapter_routes();
        for (_, w) in inputs.walk_layers(&m.layers) {
            let reading = w.reading as usize;
            let win = geometry[reading].1;
            let x = ops::elemwise::rmsnorm(&y, &w.attn_norm, w.attn_norm_eps_or(m.norm_eps));
            let pages = inputs.kv(&w.kv);

            let q = ops::linear::matmul(&x, &w.q_proj);
            let k = conv(
                &ops::linear::matmul(&x, &w.k_proj),
                &w.k_conv,
                &w.k_state,
                &inputs,
                m,
            );
            let v = conv(
                &ops::linear::matmul(&x, &w.v_proj),
                &w.v_conv,
                &w.v_state,
                &inputs,
                m,
            );
            let r = ops::linear::matmul(&x, &w.r_proj);
            let q = ops::elemwise::rmsnorm_per_head(&q, &w.q_norm, d, m.norm_eps);
            let k = ops::elemwise::rmsnorm_per_head(&k, &w.k_norm, d, m.norm_eps);
            ops::attn::kv_append(
                &k,
                &v,
                pages,
                &inputs.write_page(&w.kv),
                &inputs.write_offset(&w.kv),
            );
            seam::at(seam::ATTN_Q, &[&q]);

            let bias = ops::linear::rel_bias(&r, &w.rel_proj, m.heads, m.d_rel, w.extent);
            let (dq, pq) = q.split(&one);
            let (db, pb) = bias.split(&one);
            let plan_d = plan_d[reading]
                .as_ref()
                .expect("a layer of this reading built its plan");
            let plan_p = plan_p[reading]
                .as_ref()
                .expect("a layer of this reading built its plan");
            let log_scaling = match w.reading {
                Reading::Local => None,
                Reading::Global => Some(m.log_scaling),
            };
            let a = Value::merge(vec![
                ops::attn::decode_rel(
                    &dq,
                    plan_d,
                    pages,
                    &db,
                    win,
                    d,
                    w.extent,
                    m.sm_scale,
                    log_scaling,
                ),
                ops::attn::prefill_rel(
                    &pq,
                    plan_p,
                    pages,
                    &pb,
                    win,
                    d,
                    w.kv_heads,
                    w.extent,
                    m.sm_scale,
                    log_scaling,
                ),
            ]);
            seam::at(seam::ATTN_OUT, &[&a]);
            let o = ops::linear::matmul(&a, &w.o_proj);
            let o = if m.tp > 1 {
                ops::collective::all_reduce(&o)
            } else {
                o
            };
            let o = {
                let (adapted, _) = o.split(&Facts::has_adapter());
                let (px, _) = x.split(&Facts::has_adapter());
                ops::linear::lora_correct(&px, &w.lora_a, &w.lora_b, &routes, &adapted)
            };
            let o = conv(&o, &w.attn_conv, &w.attn_state, &inputs, m);
            y = ops::elemwise::residual_add(&o, &y);

            let x = ops::elemwise::rmsnorm(&y, &w.mlp_norm, m.norm_eps);
            let f = match &w.mlp {
                Mlp::Dense {
                    gate_up,
                    inter,
                    down,
                    scale,
                } => {
                    let act = ops::linear::mlp_swiglu(&ops::linear::matmul(&x, gate_up), *inter);
                    let f = ops::linear::matmul(&act, down);
                    let f = if m.tp > 1 {
                        ops::collective::all_reduce(&f)
                    } else {
                        f
                    };
                    ops::elemwise::scale(scale, &f)
                }
                Mlp::Routed {
                    router,
                    bias,
                    scale,
                    gate_up,
                    down,
                    experts,
                    top_k,
                    sink,
                    inter,
                    scaling,
                } => {
                    let fan = top_k + sink;
                    let (routes, weights) = ops::linear::moe_topk_sigmoid_sink(
                        &ops::linear::matmul(&x, router),
                        bias,
                        Some(scale),
                        *experts,
                        *top_k,
                        *sink,
                        *scaling,
                    );
                    let select = |act: &Value, bank: &Weight| {
                        if matches!(bank.dtype, Dtype::Bf16 | Dtype::F16 | Dtype::F32) {
                            ops::linear::moe_matmul_select(act, bank, &routes, fan)
                        } else {
                            ops::linear::moe_matmul_select_quant(act, bank, &routes, fan)
                        }
                    };
                    let hidden = ops::linear::mlp_swiglu(&select(&x, gate_up), *inter);
                    let routed = ops::linear::moe_weighted_sum(&select(&hidden, down), &weights);
                    if m.tp > 1 {
                        ops::collective::all_reduce(&routed)
                    } else {
                        routed
                    }
                }
            };
            let f = conv(&f, &w.mlp_conv, &w.mlp_state, &inputs, m);
            y = ops::elemwise::residual_add(&f, &y);
        }

        let x = ops::elemwise::rmsnorm(&y, &m.final_norm, m.norm_eps) * m.head_scale;
        ops::linear::lm_head(&x, &m.unembed)
    }
}

impl Layer {
    fn attn_norm_eps_or(&self, eps: f32) -> f32 {
        eps
    }
}

fn conv(v: &Value, weight: &Weight, state: &str, inputs: &Input<Facts>, m: &Model) -> Value {
    let slab = inputs.state(state);
    let (vd, vp) = v.split(&Facts::qo_one());
    Value::merge(vec![
        ops::attn::short_conv(&vd, weight, slab, m.conv_width),
        ops::attn::short_conv_chunked(&vp, weight, slab, m.conv_width),
    ])
}
