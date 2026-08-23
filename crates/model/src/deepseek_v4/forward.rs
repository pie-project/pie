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
        let mut streams = kernels::hc_expand(&kernels::embed(&ids, &m.embed), hy.streams);

        for (l, w) in m.layers.iter().enumerate() {
            let l = l as u32;
            let at = &w.attn;
            let pages = inputs.kv(&at.kv);
            let pos = inputs.positions();

            let (x, post_mix, comb_mix) = gate(&streams, &w.attn_mix, hy);

            let q = kernels::rmsnorm(&kernels::matmul(&x, &at.q_down), &at.q_norm);
            let q = kernels::rmsnorm_no_scale(&kernels::matmul(&q, &at.q_up), at.head_dim);
            let q = kernels::rope_partial_last(&q, at.rope_dim, at.head_dim, at.theta, &pos);
            seam::at(seam::ATTN_Q, (&q,), l);

            let plane = kernels::rmsnorm(&kernels::matmul(&x, &at.kv_down), &at.kv_norm);
            let plane = kernels::rope_partial_last(&plane, at.rope_dim, at.head_dim, at.theta, &pos);
            kernels::kv_append_shared(&plane, &pages);

            let (o, lse) = kernels::attention_prefill_lse(
                &kernels::query_windows(&q),
                &pages,
                Some(at.window),
                at.head_dim,
                at.heads,
                at.sm_scale,
            );
            let lse = kernels::lse_ln(&lse);

            let (o, lse) = match &at.pool {
                Some(p) => {
                    let entries = inputs.kv(&p.entries);
                    let (bpos, breq) = boundaries(&inputs, p.ratio);
                    let pooled = kernels::pool_gather(&bpos, &breq, &pages, at.head_dim, p.ratio);
                    let pooled = kernels::rope_partial_last(&pooled, at.rope_dim, at.head_dim, at.theta, &pos);
                    kernels::kv_append_pool(&pooled, &bpos, &breq, &entries);
                    let (po, plse) = kernels::attention_pooled_lse(
                        &q,
                        &entries,
                        p.ratio,
                        at.heads,
                        at.head_dim,
                        at.sm_scale,
                        &pos,
                    );
                    kernels::attention_merge_lse(&o, &lse, &po, &plse, at.heads, at.head_dim)
                }
                None => (o, lse),
            };
            let o = kernels::attention_sink(&o, &lse, &at.sink, at.head_dim);
            seam::at(seam::ATTN_OUT, (&o,), l);

            let o = kernels::matmul(&o, &at.o_down);
            let o = if TP > 1 { kernels::all_reduce(&o) } else { o };
            let o = kernels::matmul(&o, &at.o_up);
            streams = kernels::hc_fold(&o, &streams, &post_mix, &comb_mix);

            let (x, post_mix, comb_mix) = gate(&streams, &w.mlp_mix, hy);
            let f = match &w.mlp {
                Mlp::Dense { gate_up, down, inter, limit } => kernels::matmul(
                    &kernels::swiglu_clamp(&kernels::matmul(&x, gate_up), *inter, *limit),
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
                    let routes = kernels::topk_sqrt_softplus(
                        &kernels::matmul(&x, router),
                        bias,
                        *experts,
                        *top_k,
                        *renorm,
                        *scaling,
                    );
                    let hidden = kernels::matmul_select(&x, gate_up, &routes);
                    let act = kernels::swiglu_clamp(&hidden, *inter, *limit);
                    kernels::weighted_sum(&kernels::matmul_select(&act, down, &routes), &routes)
                }
            };
            let f = if TP > 1 { kernels::all_reduce(&f) } else { f };
            streams = kernels::hc_fold(&f, &streams, &post_mix, &comb_mix);
        }

        let y = kernels::hc_collapse(&streams, &hy.head_scale, &hy.head_base, hy.streams, hy.gate_eps);
        let x = kernels::rmsnorm(&y, &m.final_norm);
        let logits = match &m.head {
            Head::Tied => kernels::lm_head(&x, &m.embed),
            Head::Bank(bank) => kernels::lm_head(&x, bank),
        };

        logits
    }
}

fn gate<W1: Dtype>(streams: &Value, mix: &Mix<W1>, hy: &Hyper<W1>) -> (Value, Value, Value) {
    let normed = kernels::hc_rmsnorm_f32(streams, hy.norm_eps);
    kernels::hc_gates(
        &normed,
        &mix.scale,
        &mix.base,
        streams,
        hy.streams,
        hy.gate_eps,
        hy.alpha,
        hy.sinkhorn,
    )
}

fn boundaries(inputs: &Input<Facts>, ratio: u32) -> (Value, Value) {
    let (one, many) = inputs.positions().split(&Facts::qo_one());
    let (dpos, dreq) = kernels::pool_boundary_decode(&one, ratio);
    let (ppos, preq) = kernels::pool_boundary_prefill(&kernels::query_windows(&many), ratio);
    (merge![dpos, ppos], merge![dreq, preq])
}
