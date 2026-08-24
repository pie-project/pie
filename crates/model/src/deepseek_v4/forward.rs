use model_dsl::axes::{Dtype, KvDtype};
use model_dsl::{kernels, Facts, merge, seam, Classify, Forward, Input, KvSpec, Request, Value};

use super::model::{Head, Hyper, Mix, Mlp, Model};

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

impl<W1: Dtype, K: KvDtype, const TP: usize> Forward for Model<W1, K, TP> {
    type Facts = Facts;

    fn caches(&self) -> KvSpec {
        let mut c = KvSpec::new();
        for (l, w) in self.layers.iter().enumerate() {
            let at = &w.attn;
            c.kv(format!("kv.{l}"), [1, at.heads as u64 * at.head_dim as u64]);
            if at.pool.is_some() {
                c.kv(format!("pool.{l}"), [1, at.head_dim as u64]);
            }
        }
        c
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let m = self;
        let hy = &m.hyper;
        let ids = inputs.token_ids();
        let mut streams = kernels::hc::expand(&kernels::layout::embed(&ids, &m.embed, m.vocab), hy.streams);

        for (l, w) in m.layers.iter().enumerate() {
            let l = l as u32;
            let at = &w.attn;
            let pages = inputs.kv(&at.kv);
            let pos = inputs.positions();

            let (x, post_mix, comb_mix) = gate(&streams, &w.attn_mix, hy);

            let q = kernels::gemm::matmul(&x, &at.q_down);
            let q = kernels::norm::rmsnorm(&q, &at.q_norm.weight, at.q_norm.eps);
            let q = kernels::gemm::matmul(&q, &at.q_up);
            let q = kernels::norm::rmsnorm_no_scale(&q, at.head_dim, at.q_norm.eps);
            // GPT-J pairing (`2d` with `2d + 1`), not NeoX: `is_neox_style=False`
            // in vLLM's `build_deepseek_v4_rope`. All three `partial_last`
            // statements here — q, the shared plane, the pooled rows — are the
            // same rotation and state the same convention.
            let q = kernels::rope::partial_last(&q, &pos, at.rope_dim, at.head_dim, at.theta, true);
            seam::at(seam::ATTN_Q, (&q,), l);

            let plane = kernels::gemm::matmul(&x, &at.kv_down);
            let plane = kernels::norm::rmsnorm(&plane, &at.kv_norm.weight, at.kv_norm.eps);
            let plane = kernels::rope::partial_last(&plane, &pos, at.rope_dim, at.head_dim, at.theta, true);
            kernels::attention::kv_append_shared(&plane, &pages);

            let (o, lse) = kernels::attention::prefill_lse(
                &kernels::query_windows(&q),
                &pages,
                Some(at.window),
                at.head_dim,
                at.heads,
                at.sm_scale,
            );

            let (o, lse) = match &at.pool {
                Some(p) => {
                    let entries = inputs.kv(&p.entries);
                    let (bpos, breq) = boundaries(&inputs, p.ratio);
                    let pooled = kernels::pool::gather(&bpos, &breq, &pages, at.head_dim, p.ratio);
                    let pooled = kernels::rope::partial_last(&pooled, &pos, at.rope_dim, at.head_dim, at.theta, true);
                    kernels::pool::kv_append(&pooled, &bpos, &breq, &entries);
                    let (po, plse) = kernels::pool::attention_lse(
                        &q,
                        &pos,
                        &entries,
                        p.ratio,
                        at.heads,
                        at.head_dim,
                        at.sm_scale,
                    );
                    kernels::attention::merge_lse(&o, &lse, &po, &plse, at.heads, at.head_dim)
                }
                None => (o, lse),
            };
            let o = kernels::attention::sink(&o, &lse, &at.sink, at.head_dim);
            seam::at(seam::ATTN_OUT, (&o,), l);

            let o = kernels::gemm::matmul(&o, &at.o_down);
            let o = if TP > 1 { kernels::dist::all_reduce(&o) } else { o };
            let o = kernels::gemm::matmul(&o, &at.o_up);
            streams = kernels::hc::fold(&o, &streams, &post_mix, &comb_mix);

            let (x, post_mix, comb_mix) = gate(&streams, &w.mlp_mix, hy);
            let f = match &w.mlp {
                Mlp::Dense { gate_up, down, inter, limit } => kernels::gemm::matmul(
                    &kernels::mlp::swiglu_clamp(&kernels::gemm::matmul(&x, gate_up), *inter, *limit),
                    down,
                ),
                Mlp::Routed {
                    router,
                    bias,
                    gate_up,
                    down,
                    experts,
                    top_k,
                    inter,
                    limit,
                    renorm,
                    scaling,
                } => {
                    let (routes, weights) = kernels::moe::topk_sqrt_softplus(
                        &kernels::gemm::matmul(&x, router),
                        bias,
                        *experts,
                        *top_k,
                        *renorm,
                        *scaling,
                    );
                    let hidden = kernels::moe::matmul_select(&x, gate_up, &routes);
                    let act = kernels::mlp::swiglu_clamp(&hidden, *inter, *limit);
                    kernels::moe::weighted_sum(
                        &kernels::moe::matmul_select(&act, down, &routes),
                        &weights,
                    )
                }
            };
            let f = if TP > 1 { kernels::dist::all_reduce(&f) } else { f };
            streams = kernels::hc::fold(&f, &streams, &post_mix, &comb_mix);
        }

        let y = kernels::hc::collapse(&streams, &hy.head_scale, &hy.head_base, hy.streams, hy.gate_eps);
        let fin = &m.final_norm;
        let x = kernels::norm::rmsnorm(&y, &fin.weight, fin.eps);
        let logits = match &m.head {
            Head::Tied => kernels::gemm::lm_head(&x, &m.embed),
            Head::Bank(bank) => kernels::gemm::lm_head(&x, bank),
        };

        logits
    }
}

fn gate<W1: Dtype>(streams: &Value, mix: &Mix<W1>, hy: &Hyper<W1>) -> (Value, Value, Value) {
    let normed = kernels::hc::rmsnorm_f32(streams, hy.norm_eps);
    kernels::hc::gates(
        &normed,
        streams,
        &mix.scale,
        &mix.base,
        hy.streams,
        hy.gate_eps,
        hy.alpha,
        hy.sinkhorn,
    )
}

fn boundaries(inputs: &Input<Facts>, ratio: u32) -> (Value, Value) {
    let (one, many) = inputs.positions().split(&Facts::qo_one());
    let (dpos, dreq) = kernels::pool::boundary_decode(&one, ratio);
    let (ppos, preq) = kernels::pool::boundary_prefill(&kernels::query_windows(&many), ratio);
    (merge![dpos, ppos], merge![dreq, preq])
}
