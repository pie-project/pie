use model_dsl::axes::{Dtype, KvDtype};
use model_dsl::{kernels, Facts, merge, seam, Classify, Forward, Input, KvSpec, Request, Value};

use super::model::{Attn, Head, Mlp, Model};

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
                [1, (a.kv_lora_rank + a.qk_rope_head_dim) as u64],
            );
            c.kv(format!("index.{l}"), [1, a.indexer.head_dim as u64]);
        }
        c
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let m = self;
        let ids = inputs.token_ids();
        let mut y = kernels::embed(&ids, &m.embed);

        for (l, w) in m.layers.iter().enumerate() {
            let l = l as u32;
            let x = kernels::rmsnorm(&y, &w.attn_norm);
            let o = latent_attention(&x, &inputs, &w.attn, l);
            let o = if TP > 1 { kernels::all_reduce(&o) } else { o };
            y = kernels::add(&y, &o);

            let x = kernels::rmsnorm(&y, &w.mlp_norm);
            let f = match &w.mlp {
                Mlp::Dense { gate_up, down, inter } => kernels::matmul(
                    &kernels::swiglu(&kernels::matmul(&x, gate_up), *inter),
                    down,
                ),
                Mlp::Routed {
                    router,
                    gate_up,
                    down,
                    shared,
                    experts,
                    top_k,
                    inter,
                    norm_weights,
                    scaling,
                } => {
                    let routes = kernels::topk_sigmoid(
                        &kernels::matmul(&x, router),
                        *experts,
                        *top_k,
                        *norm_weights,
                        *scaling,
                    );
                    let shared = shared.as_ref().map(|s| {
                        let act =
                            kernels::swiglu(&kernels::matmul(&x, &s.gate_up), s.inter);
                        kernels::matmul(&act, &s.down)
                    });
                    let hidden = kernels::matmul_select(&x, gate_up, &routes);
                    let act = kernels::swiglu(&hidden, *inter);
                    let routed =
                        kernels::weighted_sum(&kernels::matmul_select(&act, down, &routes), &routes);
                    match shared {
                        None => routed,
                        Some(dense) => kernels::add(&routed, &dense),
                    }
                }
            };
            let f = if TP > 1 { kernels::all_reduce(&f) } else { f };
            y = kernels::add(&y, &f);
        }

        let x = kernels::rmsnorm(&y, &m.final_norm);
        let logits = match &m.head {
            Head::Tied => kernels::lm_head(&x, &m.embed),
            Head::Bank(bank) => kernels::lm_head(&x, bank),
        };

        logits
    }
}

fn latent_attention<W1: Dtype>(x: &Value, inputs: &Input<Facts>, a: &Attn<W1>, l: u32) -> Value {
    let pages = inputs.kv(&a.kv);
    let positions = inputs.positions();

    let q_a = kernels::rmsnorm(&kernels::matmul(x, &a.q_a_proj), &a.q_a_norm);
    let q_b = kernels::matmul(&q_a, &a.q_b_proj);
    let kv_a = kernels::matmul(x, &a.kv_a_proj);
    seam::at(seam::ATTN_QV, (&q_b, &kv_a), l);

    let selection = index_select(x, &q_a, inputs, a, &positions);

    let (kv_c, k_pe) = kernels::mla_latents_rope(
        &kv_a,
        &a.kv_a_norm,
        a.kv_lora_rank,
        a.qk_rope_head_dim,
        a.theta,
        &positions,
    );
    kernels::kv_append_mla(&kv_c, &k_pe, &pages);

    let (q_nope, q_pe) = kernels::split_q_b(&q_b, a.heads, a.qk_nope_head_dim, a.qk_rope_head_dim);
    let q_pe = kernels::rope_partial_q(&q_pe, a.qk_rope_head_dim, a.qk_rope_head_dim, a.theta, &positions);
    let q = kernels::mla_absorb_q_pe(
        &q_nope,
        &q_pe,
        &a.kv_b_proj,
        a.heads,
        a.kv_lora_rank,
        a.qk_nope_head_dim,
    );
    seam::at(seam::ATTN_Q, (&q,), l);

    let one = Facts::qo_one();
    let (dq, pq) = q.split(&one);
    let (d_sel, p_sel) = selection.split(&one);
    let scored = merge![
        kernels::mla_attention_decode_selected(&dq, &pages, &d_sel, a.heads, a.kv_lora_rank, a.sm_scale),
        kernels::mla_attention_prefill_selected(
            &kernels::query_windows(&pq),
            &pages,
            &p_sel,
            a.heads,
            a.kv_lora_rank,
            a.sm_scale,
        ),
    ];

    let v = kernels::mla_absorb_out(
        &scored,
        &a.kv_b_proj,
        a.heads,
        a.kv_lora_rank,
        a.v_head_dim,
    );
    seam::at(seam::ATTN_OUT, (&v,), l);
    kernels::attention_landing(&v, &a.o_proj, l)
}

fn index_select<W1: Dtype>(
    x: &Value,
    q_a: &Value,
    inputs: &Input<Facts>,
    a: &Attn<W1>,
    positions: &Value,
) -> Value {
    let ix = &a.indexer;
    let keys = inputs.kv(&ix.keys);
    let k = kernels::index_layernorm_rope(
        &kernels::matmul(x, &ix.k_proj),
        &ix.k_norm,
        &ix.k_norm_bias,
        a.qk_rope_head_dim,
        a.theta,
        positions,
    );
    kernels::kv_append_index(&k, &keys);
    let q = kernels::index_rope(
        &kernels::matmul(q_a, &ix.q_proj),
        ix.heads,
        ix.head_dim,
        a.qk_rope_head_dim,
        a.theta,
        positions,
    );
    let weights = kernels::matmul(q_a, &ix.weights_proj);
    kernels::index_topk(&q, &weights, &keys, ix.heads, ix.head_dim, ix.top_k)
}
