//! The GLM 5 forward pass over the typed IR — the old forward minus the
//! machinery the design killed (design §10): the one MLA plan is built up
//! front and shared visibly across layers (§6), the write geometry of both
//! kv-append families — the latent pages and the indexer's key cache — is a
//! declared input fetched once per forward (§7), raggedness is ambient so the
//! prefill arm loses its `query_windows` plumbing (§5), and tensor
//! parallelism is plain control flow on `m.tp` (§9, decision #18).

use new_model_dsl::{
    Classify, Facts, ForwardHybrid, HybridSpec, Input, Request, Value, kernels, merge, seam,
};
use new_model_ir::GeomKind;

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

impl ForwardHybrid for Model {
    type Facts = Facts;

    fn caches(&self) -> HybridSpec {
        let mut c = HybridSpec::new();
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
        let attn = AttnShared::of(m, &inputs);
        let ids = inputs.tokens();
        let mut y = kernels::layout::embed(&ids, &m.embed, m.vocab);

        for (l, w) in inputs.layers(&m.layers) {
            let x = kernels::norm::rmsnorm(&y, &w.attn_norm.weight, w.attn_norm.eps);
            let o = latent_attention(&x, &inputs, &attn, &w.attn, l);
            let o = if m.tp > 1 {
                kernels::dist::all_reduce(&o)
            } else {
                o
            };
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
                    let hidden = kernels::moe::matmul_select(&x, gate_up, &routes, *top_k);
                    let act = kernels::mlp::swiglu(&hidden, *inter);
                    let routed = kernels::moe::weighted_sum(
                        &kernels::moe::matmul_select(&act, down, &routes, *top_k),
                        &weights,
                    );
                    match shared {
                        None => routed,
                        Some(dense) => kernels::norm::residual_add(&dense, &routed),
                    }
                }
            };
            let f = if m.tp > 1 {
                kernels::dist::all_reduce(&f)
            } else {
                f
            };
            y = kernels::norm::residual_add(&f, &y);
        }

        let fin = &m.final_norm;
        let x = kernels::norm::rmsnorm(&y, &fin.weight, fin.eps);
        kernels::gemm::lm_head(&x, &m.head)
    }
}

/// What every attention layer shares, stated once per forward: the one MLA
/// plan decode and prefill both take (§6), the page indices `mla::kv_append`
/// and `index::kv_append` write through (§7), and the positions every rope
/// reads. The first layer's two cache spaces stand in for all of them — the
/// fire lays every layer's pages out identically.
struct AttnShared {
    positions: Value,
    kv_indices: Value,
    index_indices: Value,
    plan: Value,
}

impl AttnShared {
    fn of(m: &Model, inputs: &Input<Facts>) -> AttnShared {
        let a = &m.layers.first().expect("glm 5 has at least one layer").attn;
        let kv = inputs.cache_index(a.kv.name());
        let keys = inputs.cache_index(a.indexer.keys.name());
        let positions = inputs.positions();
        AttnShared {
            plan: kernels::mla::plan(positions.rec(), kv),
            kv_indices: inputs.geometry(kv, GeomKind::Indices),
            index_indices: inputs.geometry(keys, GeomKind::Indices),
            positions,
        }
    }
}

fn latent_attention(
    x: &Value,
    inputs: &Input<Facts>,
    s: &AttnShared,
    a: &Attn,
    layer: u32,
) -> Value {
    let pages = inputs.kv(a.kv.name());

    let q_a = kernels::gemm::matmul(x, &a.q_a_proj);
    let q_a = kernels::norm::rmsnorm(&q_a, &a.q_a_norm.weight, a.q_a_norm.eps);
    let q_b = kernels::gemm::matmul(&q_a, &a.q_b_proj);
    let kv_a = kernels::gemm::matmul(x, &a.kv_a_proj);
    seam::at(seam::ATTN_QV, (&q_b, &kv_a));

    let selection = index_select(x, &q_a, inputs, s, a);

    let (kv_c, k_pe) = kernels::mla::latents_rope(
        &kv_a,
        &s.positions,
        &a.kv_a_norm,
        a.kv_lora_rank,
        a.qk_rope_head_dim,
        a.theta,
    );
    kernels::mla::kv_append(&kv_c, &k_pe, pages, &s.kv_indices, &s.positions);

    let (q_nope, q_pe) =
        kernels::mla::split_q_b(&q_b, a.heads, a.qk_nope_head_dim, a.qk_rope_head_dim);
    let q_pe = kernels::rope::partial_q(
        &q_pe,
        &s.positions,
        a.qk_rope_head_dim,
        a.qk_rope_head_dim,
        a.theta,
    );

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
            &s.plan,
            &dpe,
            &d_sel,
            pages,
            a.heads,
            a.kv_lora_rank,
            a.sm_scale,
        ),
        kernels::mla::attention_prefill_selected(
            &pq,
            &s.plan,
            &ppe,
            &p_sel,
            pages,
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
    kernels::gemm::attention_landing(&v, &a.o_proj, layer)
}

fn index_select(x: &Value, q_a: &Value, inputs: &Input<Facts>, s: &AttnShared, a: &Attn) -> Value {
    let ix = &a.indexer;
    let keys = inputs.kv(ix.keys.name());
    let k = kernels::index::layernorm_rope(
        &kernels::gemm::matmul(x, &ix.k_proj),
        &s.positions,
        &ix.k_norm,
        &ix.k_norm_bias,
        a.qk_rope_head_dim,
        a.theta,
    );
    kernels::index::kv_append(&k, keys, &s.index_indices, &s.positions);
    let q = kernels::index::rope(
        &kernels::gemm::matmul(q_a, &ix.q_proj),
        &s.positions,
        ix.heads,
        ix.head_dim,
        a.qk_rope_head_dim,
        a.theta,
    );
    let weights = kernels::gemm::matmul(q_a, &ix.weights_proj);
    kernels::index::topk(&q, &weights, keys, ix.heads, ix.head_dim, ix.top_k)
}
