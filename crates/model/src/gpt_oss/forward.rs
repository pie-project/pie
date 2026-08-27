use model_dsl::{Classify, ForwardHybrid, HybridSpec, Input, Predicate, Request, Value, ops, seam};

use super::model::{Model, Reading};

pub struct Facts {
    pub qo_one: bool,
}

impl Facts {
    pub fn qo_one() -> Predicate {
        Predicate::fact(0)
    }
}

impl Classify for Facts {
    fn of(r: &Request) -> Facts {
        Facts {
            qo_one: r.query_len() == 1,
        }
    }

    fn word(&self) -> u64 {
        self.qo_one as u64
    }
}

impl ForwardHybrid for Model {
    type Facts = Facts;

    fn caches(&self) -> HybridSpec {
        let mut c = HybridSpec::new();
        let kv = c.kv_space(self.kv);
        let plane = self.kv_heads as u64 * self.head_dim as u64;
        for w in &self.layers {
            c.kv(kv, w.attn.kv.clone(), [plane, plane]);
        }
        c
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let m = self;

        let positions = inputs.positions();
        // ONE SCHEDULE PER READING, PER CLASS. gpt-oss alternates a 128-wide
        // sliding window with full attention over one page-id space, and a
        // schedule is carved for one window or the other: `sched_prefill`
        // sizes its kv chunking from `window_left` and the kernel recomputes
        // the same count when it merges the partials, so a schedule shared
        // between the two would put the work items and the merge at two
        // different numbers. Each pair below IS that reading axis: its two
        // lines differ by the window the schedule states, `Some(m.window)`
        // against `None`, and a layer indexes its pair by the discriminant of
        // the `Reading` it carries (0 windowed, 1 full). The class is the arm
        // the plan is built off — a decode schedule reads the decode arm's
        // geometry, a prefill schedule the prefill arm's, so each plan node's
        // guard IS the class it was carved for and the recorder refuses a
        // reader from the other arm.
        let (input_d, input_p) = inputs.split(&Facts::qo_one());
        let plan_d = [
            ops::attn::plan_decode(&input_d, m.q_heads, m.kv_heads, m.head_dim, Some(m.window)),
            ops::attn::plan_decode(&input_d, m.q_heads, m.kv_heads, m.head_dim, None),
        ];
        let plan_p = [
            ops::attn::plan_prefill(&input_p, m.q_heads, m.kv_heads, m.head_dim, Some(m.window)),
            ops::attn::plan_prefill(&input_p, m.q_heads, m.kv_heads, m.head_dim, None),
        ];
        let ids = inputs.tokens();
        let mut y = ops::layout::embed(&ids, &m.embed, m.vocab);
        let d = m.head_dim;

        for (_, w) in inputs.walk_layers(&m.layers) {
            let at = &w.attn;
            let pages = inputs.kv(&at.kv);
            let write_page = inputs.write_page(&at.kv);
            let write_offset = inputs.write_offset(&at.kv);

            let x = ops::elemwise::rmsnorm(&y, &w.attn_norm, w.attn_norm_eps);
            let q = ops::elemwise::add_bias(&at.q_bias, &ops::linear::matmul(&x, &at.q_proj));
            let k = ops::elemwise::add_bias(&at.k_bias, &ops::linear::matmul(&x, &at.k_proj));
            let v = ops::elemwise::add_bias(&at.v_bias, &ops::linear::matmul(&x, &at.v_proj));
            seam::at(seam::ATTN_QV, &[&q, &v]);

            let (q, k) = ops::elemwise::rope_yarn(
                &q,
                &k,
                &positions,
                d,
                at.theta,
                at.factor,
                at.beta_fast,
                at.beta_slow,
                at.attention_factor,
                at.original_max_position,
                false,
            );
            ops::attn::kv_append(&k, &v, pages, &write_page, &write_offset);
            seam::at(seam::ATTN_Q, &[&q]);

            let win = match at.reading {
                Reading::Windowed => Some(m.window),
                Reading::Full => None,
            };
            let (dq, p) = q.split(&Facts::qo_one());
            let a = Value::merge(vec![
                {
                    let (o, lse) = ops::attn::decode_lse(
                        &dq,
                        &plan_d[at.reading as usize],
                        pages,
                        win,
                        d,
                        at.sm_scale,
                    );
                    ops::attn::sink(&o, &lse, &at.sinks, d)
                },
                {
                    let (o, lse) = ops::attn::prefill_lse(
                        &p,
                        &plan_p[at.reading as usize],
                        pages,
                        win,
                        d,
                        m.kv_heads,
                        at.sm_scale,
                    );
                    ops::attn::sink(&o, &lse, &at.sinks, d)
                },
            ]);
            seam::at(seam::ATTN_OUT, &[&a]);

            let o = ops::linear::matmul(&a, &at.o_proj);
            let o = if m.tp > 1 {
                ops::collective::all_reduce(&o)
            } else {
                o
            };
            y = ops::elemwise::residual_add(&ops::elemwise::add_bias(&at.o_bias, &o), &y);

            let e = &w.mlp;
            let x = ops::elemwise::rmsnorm(&y, &w.mlp_norm, w.mlp_norm_eps);
            let (routes, weights) = ops::linear::moe_topk_softmax(
                &ops::elemwise::add_bias(&e.router_bias, &ops::linear::matmul(&x, &e.router)),
                e.experts,
                e.top_k,
            );
            let packed = ops::linear::moe_matmul_select_bias(
                &x,
                &e.gate_up,
                &e.gate_up_bias,
                &routes,
                e.top_k,
            );
            let act = ops::linear::mlp_swiglu_clamp_alpha(
                &packed,
                e.inter,
                e.swiglu_limit,
                e.swiglu_alpha,
            );
            let routed = ops::linear::moe_matmul_select_quant(&act, &e.down, &routes, e.top_k);
            let f = ops::linear::moe_weighted_sum(&routed, &weights);
            let f = if m.tp > 1 {
                ops::collective::all_reduce(&f)
            } else {
                f
            };
            let f = ops::linear::moe_bias_sum(&f, &e.down_bias, &routes, &weights);
            y = ops::elemwise::residual_add(&f, &y);
        }

        let x = ops::elemwise::rmsnorm(&y, &m.final_norm, m.final_norm_eps);
        ops::linear::lm_head(&x, &m.head)
    }
}
