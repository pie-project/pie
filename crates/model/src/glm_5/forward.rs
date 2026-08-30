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

    /// **THE ADAPTER WINDOW** (palo design §8, §0; campaign A-6).
    ///
    /// A lane that routed its rows to a registered adapter. §8 puts the
    /// correction "over the adapter window", and §0 defines a window as the
    /// rows of the lanes whose word satisfies the guard — so the axis needs a
    /// bit, and the bit is what makes it FREE when nobody uses it: a fire no
    /// lane routed has zero rows in this class, `engine::fire::walk` skips a
    /// zero-row region before it dispatches anything, and the correction costs
    /// that fire no launch, no empty grid and no instruction. A `Guard::Always`
    /// correction would instead launch two kernels per layer over every row of
    /// every fire to add zero to them, which is 1.0x nothing.
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

        // ONE SCHEDULE PER READER. A plan struct is a CARVING — request count,
        // rebased query boundaries, work-item split — and it carves for ONE
        // class. Which class is no longer left to be inferred: each plan is
        // built off that class's own arm of the inputs, so it carries the arm's
        // cond as its guard, and a reader in the other arm is refused by the
        // recorder at the line that mixed them. Hence two schedules over the
        // same shape, [decode, prefill]. `latent_attention` cuts its q with the
        // same `Facts::qo_one()`, so its arms' conds are exactly the ones these
        // two plans were built under, and each reader finds its own.
        //
        // A carving is also carved for ONE READING, and each line states it:
        // `m.heads` queries against the `m.kv_lora_rank`-wide absorbed plane,
        // which is exactly what every layer's `mla_decode_selected` and
        // `mla_prefill_selected` restate. The two lines are otherwise the same
        // numbers — the only thing that differs is the ARM they are built off,
        // and that is the truth: one reading, two classes.
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
            // **THE CORRECTION, OVER ITS WINDOW** (design §8, campaign A-6).
            // One statement: the mixer's output, plus this row's adapter's
            // `B·(A·x)`, in place. No merge and no arm — the op writes THROUGH
            // `o`'s arena column, so a class outside the window never runs the
            // node and reads the uncorrected value at the same address, which
            // is the identity for free.
            //
            // AFTER the reduce, and `Layer::lora_a`'s own note argues why: a
            // correction on a rows-cut partial product would be summed `tp`
            // times.
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

/// The MLA reading — `m.heads` and `m.kv_lora_rank` — is the trunk's, stated
/// on the schedule these two arms read; the layer's `a` carries only what
/// varies below the reading (the head widths it splits and absorbs at, its
/// rope theta, its scale, its weights, its spaces).
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
    ops::attn::index_topk(&q, &weights, keys, ix.heads, ix.head_dim, ix.top_k)
}
