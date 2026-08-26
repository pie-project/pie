use model_dsl::axes::{Dtype, KvDtype};
use model_dsl::{Classify, Facts, Forward, Input, KvSpec, Request, Value, kernels, merge, seam};

use super::model::Model;

#[derive(Facts)]
pub struct Facts {
    pub qo_one: bool,
}

impl Classify for Facts {
    fn of(r: &Request) -> Self {
        Self {
            qo_one: r.query_len() == 1,
        }
    }
}

impl<W1: Dtype, W2: Dtype, K: KvDtype, const TP: usize> Forward for Model<W1, W2, K, TP> {
    type Facts = Facts;

    fn caches(&self) -> KvSpec {
        let mut c = KvSpec::new();
        for (l, w) in self.layers.iter().enumerate() {
            let a = &w.attn;
            c.kv(
                format!("kv.{l}"),
                [2, a.kv_heads as u64 * a.head_dim as u64],
            );
        }
        c
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let m = self;
        let ids = inputs.token_ids();
        let mut y = kernels::layout::embed(&ids, &m.embed, m.vocab);

        for (_, w) in inputs.layers(&m.layers) {
            let at = &w.attn;
            let d = at.head_dim;
            let pages = inputs.kv(&at.kv);

            let x = kernels::norm::rmsnorm(&y, &w.attn_norm.weight, w.attn_norm.eps);
            // ONE MATVEC AND ONE CUT, where this was three of each.
            //
            // The three projections read the SAME row and write three
            // rectangles, so they were three dispatches that could not overlap
            // — and on `driver-metal` the two small ones ran at 24% of the
            // machine's streaming roof against 68% for `q` and 90% for the
            // lm_head, because 2.95 MB is not enough work to fill a GPU. The
            // packed bank makes them one 29.5 MB matvec, and
            // `layout.split_qkv` cuts the result back along the axis
            // `Source::Pack` joined it on. `gemma_4` has always read its
            // attention this way.
            let (q, k, v) = kernels::layout::split_qkv(
                &kernels::norm::add_bias(&at.qkv_bias, &kernels::gemm::matmul(&x, &at.qkv_proj)),
                at.q_heads * d,
                at.kv_heads * d,
            );
            seam::at(seam::ATTN_QV, (&q, &v));

            // NeoX pairing, and the YaRN block unpacked: a builder mirrors its
            // declaration one parameter at a time, and a struct is not a slot.
            let r = &at.rope;
            let (q, k) = kernels::rope::yarn(
                &q,
                &k,
                &inputs.positions(),
                d,
                r.theta,
                r.factor,
                r.beta_fast,
                r.beta_slow,
                r.attention_factor,
                r.original_max_position,
                false,
            );
            kernels::attention::kv_append(&k, &v, &pages);
            seam::at(seam::ATTN_Q, (&q,));

            let win = at.window;
            let (dq, p) = q.split(&Facts::qo_one());
            let a = merge![
                {
                    let (o, lse) = kernels::attention::decode_lse(&dq, &pages, win, d, at.sm_scale);
                    kernels::attention::sink(&o, &lse, &at.sinks, d)
                },
                {
                    let (o, lse) = kernels::attention::prefill_lse(
                        &kernels::query_windows(&p),
                        &pages,
                        win,
                        d,
                        at.kv_heads,
                        at.sm_scale,
                    );
                    kernels::attention::sink(&o, &lse, &at.sinks, d)
                },
            ];
            seam::at(seam::ATTN_OUT, (&a,));

            let o = kernels::gemm::attention_landing(&a, &at.o_proj);
            let o = kernels::dist::reduce::<TP>(o);
            y = kernels::norm::residual_add(&kernels::norm::add_bias(&at.o_bias, &o), &y);

            let e = &w.mlp;
            let x = kernels::norm::rmsnorm(&y, &w.mlp_norm.weight, w.mlp_norm.eps);
            let (routes, weights) = kernels::moe::topk_softmax(
                &kernels::norm::add_bias(&e.router_bias, &kernels::gemm::matmul(&x, &e.router)),
                e.experts,
                e.top_k,
            );
            let hidden = kernels::moe::matmul_select_bias(&x, &e.gate_up, &e.gate_up_bias, &routes);
            let act =
                kernels::mlp::swiglu_clamp_alpha(&hidden, e.inter, e.swiglu_limit, e.swiglu_alpha);
            let routed = kernels::moe::matmul_select_bias(&act, &e.down, &e.down_bias, &routes);
            let f = kernels::moe::weighted_sum(&routed, &weights);
            let f = kernels::dist::reduce::<TP>(f);
            y = kernels::norm::residual_add(&f, &y);
        }

        let fin = &m.final_norm;
        let x = kernels::norm::rmsnorm(&y, &fin.weight, fin.eps);
        kernels::gemm::lm_head(&x, &m.head)
    }
}
