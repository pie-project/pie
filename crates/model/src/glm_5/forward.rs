use model_dsl::axes::{Dtype, KvDtype};
use model_dsl::{Classify, Facts, Forward, Input, KvSpec, Request, Value, kernels, merge, seam};

use super::model::{Attn, Mlp, Model};

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
        let mut y = kernels::layout::embed(&ids, &m.embed, m.vocab);

        for (_, w) in inputs.layers(&m.layers) {
            let x = kernels::norm::rmsnorm(&y, &w.attn_norm.weight, w.attn_norm.eps);
            let o = latent_attention(&x, &inputs, &w.attn);
            let o = kernels::dist::reduce::<TP>(o);
            y = kernels::norm::residual_add(&o, &y);

            let x = kernels::norm::rmsnorm(&y, &w.mlp_norm.weight, w.mlp_norm.eps);
            let f = match &w.mlp {
                Mlp::Dense {
                    gate_up,
                    down,
                    inter,
                } => kernels::gemm::matmul(
                    &kernels::mlp::swiglu(&kernels::gemm::matmul(&x, gate_up), *inter),
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
                    let (routes, weights) = kernels::moe::topk_sigmoid(
                        &kernels::gemm::matmul(&x, router),
                        *experts,
                        *top_k,
                        *norm_weights,
                        *scaling,
                    );
                    let shared = shared.as_ref().map(|s| {
                        let act =
                            kernels::mlp::swiglu(&kernels::gemm::matmul(&x, &s.gate_up), s.inter);
                        kernels::gemm::matmul(&act, &s.down)
                    });
                    let hidden = kernels::moe::matmul_select(&x, gate_up, &routes);
                    let act = kernels::mlp::swiglu(&hidden, *inter);
                    let routed = kernels::moe::weighted_sum(
                        &kernels::moe::matmul_select(&act, down, &routes),
                        &weights,
                    );
                    match shared {
                        None => routed,
                        Some(dense) => kernels::norm::residual_add(&dense, &routed),
                    }
                }
            };
            let f = kernels::dist::reduce::<TP>(f);
            y = kernels::norm::residual_add(&f, &y);
        }

        let fin = &m.final_norm;
        let x = kernels::norm::rmsnorm(&y, &fin.weight, fin.eps);
        kernels::gemm::lm_head(&x, &m.head)
    }
}

fn latent_attention<W1: Dtype>(x: &Value, inputs: &Input<Facts>, a: &Attn<W1>) -> Value {
    let pages = inputs.kv(&a.kv);
    let positions = inputs.positions();

    let q_a = kernels::gemm::matmul(x, &a.q_a_proj);
    let q_a = kernels::norm::rmsnorm(&q_a, &a.q_a_norm.weight, a.q_a_norm.eps);
    let q_b = kernels::gemm::matmul(&q_a, &a.q_b_proj);
    let kv_a = kernels::gemm::matmul(x, &a.kv_a_proj);
    seam::at(seam::ATTN_QV, (&q_b, &kv_a));

    let selection = index_select(x, &q_a, inputs, a, &positions);

    let (kv_c, k_pe) = kernels::mla::latents_rope(
        &kv_a,
        &positions,
        &a.kv_a_norm,
        a.kv_lora_rank,
        a.qk_rope_head_dim,
        a.theta,
    );
    kernels::mla::kv_append(&kv_c, &k_pe, &pages);

    let (q_nope, q_pe) =
        kernels::mla::split_q_b(&q_b, a.heads, a.qk_nope_head_dim, a.qk_rope_head_dim);
    let q_pe = kernels::rope::partial_q(
        &q_pe,
        &positions,
        a.qk_rope_head_dim,
        a.qk_rope_head_dim,
        a.theta,
    );
    // THE ABSORB IS KIMI'S, AND SO IS THE QUERY PAIR. This text used to
    // state `mla.absorb_q_pe` — the absorb with the rotated half folded into
    // the result's tail — and hand the fused row to a selected attention
    // that stated no `q_pe`. Nothing in the tree ever wrote such a tail and
    // nothing ever read one: the absorb is a strided batched gemm with a
    // single activation operand, and both latent attention kernels address
    // `q_nope` and `q_pe` as two planes with two pitches. The legacy glm
    // text did not fold either. The point is gone and the pair is here.
    let q = kernels::mla::absorb_q(
        &q_nope,
        &a.kv_b_proj,
        a.heads,
        a.kv_lora_rank,
        a.qk_nope_head_dim,
        a.v_head_dim,
    );
    seam::at(seam::ATTN_Q, (&q,));

    let one = Facts::qo_one();
    let (dq, pq) = q.split(&one);
    let (dpe, ppe) = q_pe.split(&one);
    let (d_sel, p_sel) = selection.split(&one);
    let scored = merge![
        kernels::mla::attention_decode_selected(
            &dq,
            &dpe,
            &d_sel,
            &pages,
            a.heads,
            a.kv_lora_rank,
            a.sm_scale,
        ),
        kernels::mla::attention_prefill_selected(
            &kernels::query_windows(&pq),
            &ppe,
            &p_sel,
            &pages,
            a.heads,
            a.kv_lora_rank,
            a.sm_scale,
        ),
    ];

    let v = kernels::mla::absorb_out(
        &scored,
        &a.kv_b_proj,
        a.heads,
        a.kv_lora_rank,
        a.v_head_dim,
        a.qk_nope_head_dim,
    );
    seam::at(seam::ATTN_OUT, (&v,));
    kernels::gemm::attention_landing(&v, &a.o_proj)
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
    let k = kernels::index::layernorm_rope(
        &kernels::gemm::matmul(x, &ix.k_proj),
        positions,
        &ix.k_norm,
        &ix.k_norm_bias,
        a.qk_rope_head_dim,
        a.theta,
    );
    kernels::index::kv_append(&k, &keys);
    let q = kernels::index::rope(
        &kernels::gemm::matmul(q_a, &ix.q_proj),
        positions,
        ix.heads,
        ix.head_dim,
        a.qk_rope_head_dim,
        a.theta,
    );
    let weights = kernels::gemm::matmul(q_a, &ix.weights_proj);
    kernels::index::topk(&q, &weights, &keys, ix.heads, ix.head_dim, ix.top_k)
}
