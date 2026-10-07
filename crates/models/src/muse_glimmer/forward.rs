use poem_dsl::fact;
use poem_dsl::{ForwardHybrid, HybridSpec, Input, Value, ops, seam};

use super::model::{Model, Reading};

impl ForwardHybrid for Model {
    fn caches(&self) -> HybridSpec {
        let mut c = HybridSpec::new();
        let kv = c.kv_space(self.kv);
        let plane = u64::from(self.kv_heads) * u64::from(self.head_dim);
        for w in &self.layers {
            c.kv(kv, w.kv.clone(), [plane, plane], self.head_dim)
                .heads();
        }
        c
    }

    fn forward(&self, inputs: Input) -> Value {
        let m = self;
        let d = m.head_dim;

        let windows = [Some(m.window), None];
        let classes = [fact::has(fact::Mask), fact::scores(), fact::single_token()];
        let ([input_m, input_s, input_d], input_p) = inputs.partition(classes.clone());
        let plans = |input: &Input, decode: bool| {
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

            let ([mq, sq, dq], p) = q.partition(classes.clone());
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
            let o = {
                let adapted = o.on(fact::has(fact::Adapter));
                let px = normed.on(fact::has(fact::Adapter));
                ops::linear::lora_correct(&px, &w.lora_a, &w.lora_b, &routes, &adapted)
            };

            y = ops::elemwise::residual_add(
                &ops::elemwise::rmsnorm_plus_one(&o, &w.post_attn_norm, w.post_attn_norm_eps),
                &y,
            );
            let mlp_in = ops::elemwise::rmsnorm_plus_one(&y, &w.pre_ffw_norm, w.pre_ffw_norm_eps);
            let act = ops::linear::mlp_swiglu(&ops::linear::matmul(&mlp_in, &w.gate_up), w.inter);
            let f = ops::linear::matmul(&act, &w.down);
            y = ops::elemwise::residual_add(
                &ops::elemwise::rmsnorm_plus_one(&f, &w.post_ffw_norm, w.post_ffw_norm_eps),
                &y,
            );
        }

        let x = ops::elemwise::rmsnorm(&y, &m.final_norm, m.final_norm_eps) * m.output_multiplier;
        let x = ops::layout::gather_rows(&x, &inputs.readout_rows());
        let logits = ops::linear::lm_head(&x, &m.lm_head);
        ops::attn::logit_softcap(&logits, m.softcap)
    }
}
