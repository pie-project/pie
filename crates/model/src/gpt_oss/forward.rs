use model_dsl::{
    Classify, ForwardHybrid, GeomKind, HybridSpec, Input, Predicate, Request, Value, ops, seam,
};

use super::model::Model;

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
        for w in &self.layers {
            let a = &w.attn;
            c.kv(kv, a.kv.clone(), [2, a.kv_heads as u64 * a.head_dim as u64]);
        }
        c
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let m = self;

        let positions = inputs.positions();
        let plan_d = inputs.plan_decode();
        let plan_p = inputs.plan_prefill();
        let write_page = inputs.geometry(inputs.kv_space(), GeomKind::WritePage);
        let write_offset = inputs.geometry(inputs.kv_space(), GeomKind::WriteOffset);
        let ids = inputs.tokens();
        let mut y = ops::layout::embed(&ids, &m.embed, m.vocab);

        for (_, w) in inputs.walk_layers(&m.layers) {
            let at = &w.attn;
            let d = at.head_dim;
            let pages = inputs.kv(&at.kv);

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

            let win = at.window;
            let (dq, p) = q.split(&Facts::qo_one());
            let a = Value::merge(vec![
                {
                    let (o, lse) = ops::attn::decode_lse(&dq, &plan_d, pages, win, d, at.sm_scale);
                    ops::attn::sink(&o, &lse, &at.sinks, d)
                },
                {
                    let (o, lse) = ops::attn::prefill_lse(
                        &p,
                        &plan_p,
                        pages,
                        win,
                        d,
                        at.kv_heads,
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
