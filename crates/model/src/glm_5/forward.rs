//! The GLM 5 forward pass over the typed IR — the old forward minus the
//! machinery the design killed (design §10): the one MLA plan is built up
//! front and shared visibly across layers (§6), the write geometry of both
//! kv-append families — the latent pages and the indexer's key cache — is a
//! declared input fetched where it is used (§7), raggedness is ambient so the
//! prefill arm loses its `query_windows` plumbing (§5), and tensor
//! parallelism is plain control flow on `m.tp` (§9, decision #18).

use model_dsl::{
    Classify, ForwardHybrid, GeomKind, HybridSpec, Input, Request, Value, kernels, merge, seam,
};

use super::model::{Attn, Mlp, Model};

model_dsl::facts! {
    pub struct Facts { qo_one }
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
        // Two paged spaces: the latent kv pages, and the indexer's key cache
        // with its own geometry — the fire lays every layer's pages out
        // identically within each.
        let kv = c.kv_space(self.kv);
        let index = c.kv_space(self.kv);
        for (l, w) in self.layers.iter().enumerate() {
            let a = &w.attn;
            c.kv(
                kv,
                format!("kv.{l}"),
                [1, (a.kv_lora_rank + a.qk_rope_head_dim) as u64],
            );
            c.kv(index, format!("index.{l}"), [1, a.indexer.head_dim as u64]);
        }
        c
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let m = self;
        // The one MLA plan decode and prefill both take, built once and
        // shared visibly by every layer (§6).
        let positions = inputs.positions();
        let plan = kernels::attn::mla_plan(positions.rec(), inputs.kv_space());
        let ids = inputs.tokens();
        let mut y = kernels::layout::embed(&ids, &m.embed, m.vocab);

        for (l, w) in inputs.layers(&m.layers) {
            let x = kernels::elemwise::rmsnorm(&y, &w.attn_norm, w.attn_norm_eps);
            let o = latent_attention(&x, &inputs, &plan, &w.attn, l);
            let o = if m.tp > 1 {
                kernels::collective::all_reduce(&o)
            } else {
                o
            };
            y = kernels::elemwise::residual_add(&o, &y);

            let x = kernels::elemwise::rmsnorm(&y, &w.mlp_norm, w.mlp_norm_eps);
            let f = match &w.mlp {
                Mlp::Dense {
                    gate_up,
                    down,
                    inter,
                } => kernels::linear::matmul(
                    &kernels::linear::mlp_swiglu(&kernels::linear::matmul(&x, gate_up), *inter),
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
                    let (routes, weights) = kernels::linear::moe_topk_sigmoid(
                        &kernels::linear::matmul(&x, router),
                        *experts,
                        *top_k,
                        *norm_weights,
                        *scaling,
                    );
                    let shared = shared.as_ref().map(|s| {
                        let act = kernels::linear::mlp_swiglu(
                            &kernels::linear::matmul(&x, &s.gate_up),
                            s.inter,
                        );
                        kernels::linear::matmul(&act, &s.down)
                    });
                    let hidden = kernels::linear::moe_matmul_select(&x, gate_up, &routes, *top_k);
                    let act = kernels::linear::mlp_swiglu(&hidden, *inter);
                    let routed = kernels::linear::moe_weighted_sum(
                        &kernels::linear::moe_matmul_select(&act, down, &routes, *top_k),
                        &weights,
                    );
                    match shared {
                        None => routed,
                        Some(dense) => kernels::elemwise::residual_add(&dense, &routed),
                    }
                }
            };
            let f = if m.tp > 1 {
                kernels::collective::all_reduce(&f)
            } else {
                f
            };
            y = kernels::elemwise::residual_add(&f, &y);
        }

        let x = kernels::elemwise::rmsnorm(&y, &m.final_norm, m.final_norm_eps);
        kernels::linear::lm_head(&x, &m.head)
    }
}

fn latent_attention(x: &Value, inputs: &Input<Facts>, plan: &Value, a: &Attn, layer: u32) -> Value {
    let pages = inputs.kv(&a.kv);
    let positions = inputs.positions();
    let write_page = inputs.geometry(inputs.kv_space(), GeomKind::WritePage);
    let write_offset = inputs.geometry(inputs.kv_space(), GeomKind::WriteOffset);

    let q_a = kernels::linear::matmul(x, &a.q_a_proj);
    let q_a = kernels::elemwise::rmsnorm(&q_a, &a.q_a_norm, a.q_a_norm_eps);
    let q_b = kernels::linear::matmul(&q_a, &a.q_b_proj);
    let kv_a = kernels::linear::matmul(x, &a.kv_a_proj);
    seam::at(seam::ATTN_QV, (&q_b, &kv_a));

    let selection = index_select(x, &q_a, inputs, a);

    let (kv_c, k_pe) = kernels::attn::mla_latents_rope(
        &kv_a,
        &positions,
        &a.kv_a_norm,
        a.kv_a_norm_eps,
        a.kv_lora_rank,
        a.qk_rope_head_dim,
        a.theta,
    );
    kernels::attn::mla_kv_append(&kv_c, &k_pe, pages, &write_page, &write_offset);

    let (q_nope, q_pe) =
        kernels::attn::mla_split_q_b(&q_b, a.heads, a.qk_nope_head_dim, a.qk_rope_head_dim);
    let q_pe = kernels::elemwise::rope_partial_q(
        &q_pe,
        &positions,
        a.qk_rope_head_dim,
        a.qk_rope_head_dim,
        a.theta,
    );

    let q = kernels::attn::mla_absorb_q(
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
        kernels::attn::mla_decode_selected(
            &dq,
            plan,
            &dpe,
            &d_sel,
            pages,
            a.heads,
            a.kv_lora_rank,
            a.sm_scale,
        ),
        kernels::attn::mla_prefill_selected(
            &pq,
            plan,
            &ppe,
            &p_sel,
            pages,
            a.heads,
            a.kv_lora_rank,
            a.sm_scale,
        ),
    ];

    let v = kernels::attn::mla_absorb_out(
        &scored,
        &a.kv_b_proj,
        a.heads,
        a.kv_lora_rank,
        a.v_head_dim,
        a.qk_nope_head_dim,
    );
    seam::at(seam::ATTN_OUT, (&v,));
    kernels::linear::attention_landing(&v, &a.o_proj, layer)
}

fn index_select(x: &Value, q_a: &Value, inputs: &Input<Facts>, a: &Attn) -> Value {
    let ix = &a.indexer;
    let keys = inputs.kv(&ix.keys);
    let positions = inputs.positions();
    let index_space = inputs.space_of(&ix.keys);
    let write_page = inputs.geometry(index_space, GeomKind::WritePage);
    let write_offset = inputs.geometry(index_space, GeomKind::WriteOffset);
    let k = kernels::attn::index_layernorm_rope(
        &kernels::linear::matmul(x, &ix.k_proj),
        &positions,
        &ix.k_norm,
        ix.k_norm_eps,
        &ix.k_norm_bias,
        a.qk_rope_head_dim,
        a.theta,
    );
    kernels::attn::index_kv_append(&k, keys, &write_page, &write_offset);
    let q = kernels::attn::index_rope(
        &kernels::linear::matmul(q_a, &ix.q_proj),
        &positions,
        ix.heads,
        ix.head_dim,
        a.qk_rope_head_dim,
        a.theta,
    );
    let weights = kernels::linear::matmul(q_a, &ix.weights_proj);
    kernels::attn::index_topk(&q, &weights, keys, ix.heads, ix.head_dim, ix.top_k)
}
