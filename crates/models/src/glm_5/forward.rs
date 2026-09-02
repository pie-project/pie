use model_dsl::{Classify, ForwardHybrid, HybridSpec, Input, Predicate, Request, Value, ops, seam};

use super::model::{Attn, Indexer, Mlp, Model};

pub struct Facts {
    pub qo_one: bool,
    pub has_adapter: bool,
}

impl Facts {
    pub fn qo_one() -> Predicate {
        Predicate::fact(0)
    }

    /// True for rows routed to a registered adapter. A fire with no adapter
    /// rows has zero rows in this class, so the correction dispatches nothing.
    pub fn has_adapter() -> Predicate {
        Predicate::fact(1)
    }
}

impl Classify for Facts {
    fn of(r: &Request) -> Facts {
        Facts {
            qo_one: r.query_len() == 1,
            has_adapter: r.has_adapter(),
        }
    }

    fn word(&self) -> u64 {
        u64::from(self.qo_one) | (u64::from(self.has_adapter) << 1)
    }
}

impl ForwardHybrid for Model {
    type Facts = Facts;

    fn caches(&self) -> HybridSpec {
        let mut c = HybridSpec::new();

        let kv = c.kv_space(self.kv_dtype);
        let index = c.kv_space(self.kv_dtype);
        for w in &self.layers {
            let a = &w.attn;
            c.kv(
                kv,
                a.kv.clone(),
                [self.kv_lora_rank as u64, a.qk_rope_head_dim as u64],
            );
            c.kv(index, a.indexer.keys.clone(), [a.indexer.head_dim as u64]);
        }
        c
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let m = self;

        // Two schedules, [decode, prefill], split by Facts::qo_one(); latent_attention
        // splits q the same way so each reader finds its matching plan.
        let (input_d, input_p) = inputs.split(&Facts::qo_one());
        let plan = [
            ops::attn::mla_plan(&input_d, m.heads, m.kv_lora_rank),
            ops::attn::mla_plan(&input_p, m.heads, m.kv_lora_rank),
        ];
        let ids = inputs.tokens();
        let mut y = ops::layout::embed(&ids, &m.embed, m.vocab);

        let routes = inputs.adapter_routes();
        for (_, w) in inputs.walk_layers(&m.layers) {
            let x = ops::elemwise::rmsnorm(&y, &w.attn_norm, w.attn_norm_eps);
            let o = latent_attention(&x, &inputs, &plan, m, &w.attn);
            let o = if m.tp > 1 {
                ops::collective::all_reduce(&o)
            } else {
                o
            };
            // Applied after all_reduce: on a tp-split partial product the
            // correction would be summed tp times.
            let o = {
                let (adapted, _) = o.split(&Facts::has_adapter());
                let (px, _) = x.split(&Facts::has_adapter());
                ops::linear::lora_correct(&px, &w.lora_a, &w.lora_b, &routes, &adapted)
            };
            y = ops::elemwise::residual_add(&o, &y);

            let x = ops::elemwise::rmsnorm(&y, &w.mlp_norm, w.mlp_norm_eps);
            let f = match &w.mlp {
                Mlp::Dense {
                    gate_up,
                    down,
                    inter,
                } => ops::linear::matmul(
                    &ops::linear::mlp_swiglu(&ops::linear::matmul(&x, gate_up), *inter),
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
                    let (routes, weights) = ops::linear::moe_topk_sigmoid(
                        &ops::linear::matmul(&x, router),
                        *experts,
                        *top_k,
                        *norm_weights,
                        *scaling,
                    );
                    let shared = shared.as_ref().map(|s| {
                        let act =
                            ops::linear::mlp_swiglu(&ops::linear::matmul(&x, &s.gate_up), s.inter);
                        ops::linear::matmul(&act, &s.down)
                    });
                    let packed = ops::linear::moe_matmul_select(&x, gate_up, &routes, *top_k);
                    let act = ops::linear::mlp_swiglu(&packed, *inter);
                    let routed = ops::linear::moe_weighted_sum(
                        &ops::linear::moe_matmul_select(&act, down, &routes, *top_k),
                        &weights,
                    );
                    match shared {
                        None => routed,
                        Some(dense) => ops::elemwise::residual_add(&dense, &routed),
                    }
                }
            };
            let f = if m.tp > 1 {
                ops::collective::all_reduce(&f)
            } else {
                f
            };
            y = ops::elemwise::residual_add(&f, &y);
        }

        let x = ops::elemwise::rmsnorm(&y, &m.final_norm, m.final_norm_eps);
        ops::linear::lm_head(&x, &m.head)
    }
}

/// `m.heads` and `m.kv_lora_rank` come from the trunk; `a` carries only what
/// varies per layer (head widths, rope theta, scale, weights, spaces).
fn latent_attention(
    x: &Value,
    inputs: &Input<Facts>,
    plan: &[Value; 2],
    m: &Model,
    a: &Attn,
) -> Value {
    let pages = inputs.kv(&a.kv);
    let positions = inputs.positions();
    let write_page = inputs.write_page(&a.kv);
    let write_offset = inputs.write_offset(&a.kv);

    let q_a = ops::linear::matmul(x, &a.q_a_proj);
    let q_a = ops::elemwise::rmsnorm(&q_a, &a.q_a_norm, a.q_a_norm_eps);
    let q_b = ops::linear::matmul(&q_a, &a.q_b_proj);
    let kv_a = ops::linear::matmul(x, &a.kv_a_proj);
    seam::at(seam::ATTN_QV, &[&q_b, &kv_a]);

    let selection = index_select(x, &q_a, inputs, &a.indexer);

    let (kv_c, k_pe) = ops::attn::mla_latents_rope(
        &kv_a,
        &positions,
        &a.kv_a_norm,
        a.kv_a_norm_eps,
        m.kv_lora_rank,
        a.qk_rope_head_dim,
        a.theta,
    );
    ops::attn::mla_kv_append(&kv_c, &k_pe, pages, &write_page, &write_offset);

    let (q_nope, q_pe) =
        ops::attn::mla_split_q_b(&q_b, m.heads, a.qk_nope_head_dim, a.qk_rope_head_dim);
    let q_pe = ops::elemwise::rope_partial_q(
        &q_pe,
        &positions,
        a.qk_rope_head_dim,
        a.qk_rope_head_dim,
        a.theta,
    );

    let q = ops::attn::mla_absorb_q(
        &q_nope,
        &a.kv_b_proj,
        m.heads,
        m.kv_lora_rank,
        a.qk_nope_head_dim,
        a.v_head_dim,
    );
    seam::at(seam::ATTN_Q, &[&q]);

    let one = Facts::qo_one();
    let (dq, pq) = q.split(&one);
    let (dpe, ppe) = q_pe.split(&one);
    let (d_sel, p_sel) = selection.split(&one);
    let scored = Value::merge(vec![
        ops::attn::mla_decode_selected(
            &dq,
            &plan[0],
            &dpe,
            &d_sel,
            pages,
            m.heads,
            m.kv_lora_rank,
            a.sm_scale,
        ),
        ops::attn::mla_prefill_selected(
            &pq,
            &plan[1],
            &ppe,
            &p_sel,
            pages,
            m.heads,
            m.kv_lora_rank,
            a.sm_scale,
        ),
    ]);

    let v = ops::attn::mla_absorb_out(
        &scored,
        &a.kv_b_proj,
        m.heads,
        m.kv_lora_rank,
        a.qk_nope_head_dim,
        a.v_head_dim,
    );
    seam::at(seam::ATTN_OUT, &[&v]);
    ops::linear::matmul(&v, &a.o_proj)
}

fn index_select(x: &Value, q_a: &Value, inputs: &Input<Facts>, ix: &Indexer) -> Value {
    let keys = inputs.kv(&ix.keys);
    let positions = inputs.positions();
    let write_page = inputs.write_page(&ix.keys);
    let write_offset = inputs.write_offset(&ix.keys);
    let k = ops::attn::index_layernorm_rope(
        &ops::linear::matmul(x, &ix.k_proj),
        &positions,
        &ix.k_norm,
        ix.k_norm_eps,
        &ix.k_norm_bias,
        ix.rope_dim,
        ix.theta,
    );
    ops::attn::index_kv_append(&k, keys, &write_page, &write_offset);
    let q = ops::attn::index_rope(
        &ops::linear::matmul(q_a, &ix.q_proj),
        &positions,
        ix.heads,
        ix.head_dim,
        ix.rope_dim,
        ix.theta,
    );
    let weights = ops::linear::matmul(q_a, &ix.weights_proj);
    // One key per token: stride 1, published ids are token positions.
    ops::attn::index_topk(&q, &weights, keys, ix.heads, ix.head_dim, ix.top_k, 1)
}
