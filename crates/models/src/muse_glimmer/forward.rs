use model_dsl::{Classify, ForwardHybrid, HybridSpec, Input, Predicate, Request, Value, ops, seam};

use super::model::{Model, Reading};

pub struct Facts {
    pub qo_one: bool,
    pub masked: bool,
    pub has_adapter: bool,
    pub captures_scores: bool,
}

impl Facts {
    pub fn qo_one() -> Predicate {
        Predicate::fact(0)
    }

    pub fn masked() -> Predicate {
        Predicate::fact(1)
    }

    pub fn has_adapter() -> Predicate {
        Predicate::fact(2)
    }

    pub fn captures_scores() -> Predicate {
        Predicate::fact(3)
    }
}

impl Classify for Facts {
    fn of(r: &Request) -> Facts {
        Facts {
            qo_one: r.query_len() == 1,
            masked: r.has_custom_mask(),
            has_adapter: r.has_adapter(),
            captures_scores: r.captures_scores(),
        }
    }

    fn word(&self) -> u64 {
        u64::from(self.qo_one)
            | (u64::from(self.masked) << 1)
            | (u64::from(self.has_adapter) << 2)
            | (u64::from(self.captures_scores) << 3)
    }
}

impl ForwardHybrid for Model {
    type Facts = Facts;

    fn caches(&self) -> HybridSpec {
        let mut c = HybridSpec::new();
        let kv = c.kv_space(self.kv);
        let plane = u64::from(self.kv_heads) * u64::from(self.head_dim);
        for w in &self.layers {
            c.kv(kv, w.kv.clone(), [plane, plane]);
        }
        c
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let m = self;
        let d = m.head_dim;

        let windows = [Some(m.window), None];
        let classes = [
            Facts::masked(),
            Facts::captures_scores(),
            Facts::qo_one(),
            Predicate::rest(),
        ];
        let [input_m, input_s, input_d, input_p] = inputs.split(classes.clone());
        let plans = |input: &Input<Facts>, decode: bool| {
            windows.map(|win| {
                if decode {
                    ops::attn::plan_decode(input, m.q_heads, m.kv_heads, d, win)
                } else {
                    ops::attn::plan_prefill(input, m.q_heads, m.kv_heads, d, win)
                }
            })
        };
        let plan_m = plans(&input_m, false);
        let plan_s = plans(&input_s, false);
        let plan_d = plans(&input_d, true);
        let plan_p = plans(&input_p, false);
        let mask = inputs.mask();
        let positions = inputs.positions();

        let ids = inputs.tokens();
        let mut y = ops::elemwise::rmsnorm_no_scale(
            &ops::layout::embed(&ids, &m.embed, m.vocab),
            m.hidden,
            m.norm_eps,
        );

        let routes = inputs.adapter_routes();
        for (_, w) in inputs.walk_layers(&m.layers) {
            let reading = w.reading as usize;
            let win = windows[reading];
            let normed = ops::elemwise::rmsnorm_plus_one(&y, &w.attn_norm, w.attn_norm_eps);
            let pages = inputs.kv(&w.kv);

            let (q, k, v) = ops::layout::split_qkv(
                &ops::linear::matmul(&normed, &w.qkv),
                m.q_heads * d,
                m.kv_heads * d,
            );
            let q = ops::elemwise::rmsnorm_no_scale(&q, d, m.norm_eps);
            let k = ops::elemwise::rmsnorm_no_scale(&k, d, m.norm_eps);
            let (q, k) = match w.reading {
                Reading::Sliding => ops::elemwise::rope_full(&q, &k, &positions, d, m.theta, false),
                Reading::Full => (q, k),
            };
            ops::attn::kv_append(
                &k,
                &v,
                pages,
                &inputs.write_page(&w.kv),
                &inputs.write_offset(&w.kv),
            );
            seam::at(seam::ATTN_Q, &[&q]);

            let [mq, sq, dq, p] = q.split(classes.clone());
            let so = match w.reading {
                Reading::Sliding => {
                    ops::attn::prefill(&sq, &plan_s[reading], pages, win, d, m.kv_heads, m.sm_scale)
                }
                Reading::Full => {
                    let (so, lse) = ops::attn::prefill_lse(
                        &sq,
                        &plan_s[reading],
                        pages,
                        win,
                        d,
                        m.kv_heads,
                        m.sm_scale,
                    );
                    seam::at(seam::SCORES, &[&lse]);
                    so
                }
            };
            let a = Value::merge(vec![
                ops::attn::masked(
                    &mq,
                    &plan_m[reading],
                    &mask,
                    pages,
                    win,
                    d,
                    m.kv_heads,
                    true,
                    m.sm_scale,
                ),
                so,
                ops::attn::decode(&dq, &plan_d[reading], pages, win, d, m.sm_scale),
                ops::attn::prefill(&p, &plan_p[reading], pages, win, d, m.kv_heads, m.sm_scale),
            ]);
            seam::at(seam::ATTN_OUT, &[&a]);

            let gate = ops::linear::matmul(&normed, &w.gate);
            let o = ops::linear::matmul(&ops::elemwise::gate_sigmoid_mul(&a, &gate), &w.o_proj);
            let o = if m.tp > 1 {
                ops::collective::all_reduce(&o)
            } else {
                o
            };
            let o = {
                let (adapted, _) = o.split(&Facts::has_adapter());
                let (px, _) = normed.split(&Facts::has_adapter());
                ops::linear::lora_correct(&px, &w.lora_a, &w.lora_b, &routes, &adapted)
            };

            y = ops::elemwise::residual_add(
                &ops::elemwise::rmsnorm_plus_one(&o, &w.post_attn_norm, w.post_attn_norm_eps),
                &y,
            );
            let mlp_in = ops::elemwise::rmsnorm_plus_one(&y, &w.pre_ffw_norm, w.pre_ffw_norm_eps);
            let act = ops::linear::mlp_swiglu(&ops::linear::matmul(&mlp_in, &w.gate_up), w.inter);
            let f = ops::linear::matmul(&act, &w.down);
            let f = if m.tp > 1 {
                ops::collective::all_reduce(&f)
            } else {
                f
            };
            y = ops::elemwise::residual_add(
                &ops::elemwise::rmsnorm_plus_one(&f, &w.post_ffw_norm, w.post_ffw_norm_eps),
                &y,
            );
        }

        let x = ops::elemwise::rmsnorm(&y, &m.final_norm, m.final_norm_eps) * m.output_multiplier;
        let logits = ops::linear::lm_head(&x, &m.lm_head);
        let logits = if m.lm_head.dim(0) < u64::from(m.vocab) {
            ops::collective::all_gather(&logits, m.tp)
        } else {
            logits
        };
        ops::attn::logit_softcap(&logits, m.softcap)
    }
}
