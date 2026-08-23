use model_dsl::axes::{Dtype, KvDtype};
use model_dsl::{kernels, Facts, merge, seam, Classify, Forward, Input, KvSpec, Request, Value};

use super::model::Model;

#[derive(Facts)]
pub struct Facts {
    pub qo_one: bool,
}

impl Classify for Facts {
    fn of(r: &Request) -> Self {
        Self { qo_one: r.query_len() == 1 }
    }
}

impl<W1: Dtype, W2: Dtype, K: KvDtype, const TP: usize> Forward for Model<W1, W2, K, TP> {
    type Facts = Facts;

    fn caches(&self) -> KvSpec {
        let mut c = KvSpec::new();
        for (l, w) in self.layers.iter().enumerate() {
            let a = &w.attn;
            c.kv(format!("kv.{l}"), [2, a.kv_heads as u64 * a.head_dim as u64]);
        }
        c
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let m = self;
        let ids = inputs.token_ids();
        let mut y = kernels::embed(&ids, &m.embed);

        for (l, w) in m.layers.iter().enumerate() {
            let l = l as u32;
            let at = &w.attn;
            let d = at.head_dim;
            let pages = inputs.kv(&at.kv);

            let x = kernels::rmsnorm(&y, &w.attn_norm);
            let q = kernels::add_bias(&kernels::matmul(&x, &at.q_proj), &at.q_bias);
            let k = kernels::add_bias(&kernels::matmul(&x, &at.k_proj), &at.k_bias);
            let v = kernels::add_bias(&kernels::matmul(&x, &at.v_proj), &at.v_bias);
            seam::at(seam::ATTN_QV, (&q, &v), l);

            let (q, k) = kernels::rope_yarn(&q, &k, d, &at.rope, &inputs.positions());
            kernels::kv_append(&k, &v, &pages);
            seam::at(seam::ATTN_Q, (&q,), l);

            let win = at.kind.window();
            let (dq, p) = q.split(&Facts::qo_one());
            let a = merge![
                {
                    let (o, lse) = kernels::attention_decode_lse(&dq, &pages, win, d, at.sm_scale);
                    kernels::attention_sink(&o, &lse, &at.sinks, d)
                },
                {
                    let (o, lse) = kernels::attention_prefill_lse(
                        &kernels::query_windows(&p),
                        &pages,
                        win,
                        d,
                        at.kv_heads,
                        at.sm_scale,
                    );
                    kernels::attention_sink(&o, &lse, &at.sinks, d)
                },
            ];
            seam::at(seam::ATTN_OUT, (&a,), l);

            let o = kernels::attention_landing(&a, &at.o_proj, l);
            let o = if TP > 1 { kernels::all_reduce(&o) } else { o };
            y = kernels::add(&y, &kernels::add_bias(&o, &at.o_bias));

            let e = &w.mlp;
            let x = kernels::rmsnorm(&y, &w.mlp_norm);
            let routes = kernels::topk_softmax(
                &kernels::add_bias(&kernels::matmul(&x, &e.router), &e.router_bias),
                e.experts,
                e.top_k,
            );
            let hidden = kernels::matmul_select_bias(&x, &e.gate_up, &e.gate_up_bias, &routes);
            let act = kernels::swiglu_clamp_alpha(&hidden, e.inter, e.swiglu_limit, e.swiglu_alpha);
            let routed = kernels::matmul_select_bias(&act, &e.down, &e.down_bias, &routes);
            let f = kernels::weighted_sum(&routed, &routes);
            let f = if TP > 1 { kernels::all_reduce(&f) } else { f };
            y = kernels::add(&y, &f);
        }

        let x = kernels::rmsnorm(&y, &m.final_norm);
        let logits = kernels::lm_head(&x, &m.head);

        logits
    }
}
