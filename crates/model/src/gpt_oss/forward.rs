//! The gpt-oss forward pass over the typed IR — the old forward minus the
//! machinery the design killed (design §10): attention plans are built once
//! up front and shared visibly across layers (§6), kv-append geometry is a
//! declared input fetched once per forward (§7), raggedness is ambient so the
//! prefill arm loses its `query_windows` plumbing (§5), and tensor
//! parallelism is plain control flow on `m.tp` (§9, decision #18). The
//! sink-corrected lse arms and the clamped-alpha swiglu transcribe verbatim.

use model_dsl::{
    Classify, ForwardHybrid, GeomKind, HybridSpec, Input, Request, Value, kernels, merge, seam,
};

use super::model::Model;

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
        let kv = c.kv_space(self.kv);
        for (l, w) in self.layers.iter().enumerate() {
            let a = &w.attn;
            c.kv(
                kv,
                format!("kv.{l}"),
                [2, a.kv_heads as u64 * a.head_dim as u64],
            );
        }
        c
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let m = self;
        // The decode and prefill plans, built once and shared visibly by
        // every layer (§6), and the write addressing `kv_append` lands in.
        let positions = inputs.positions();
        let plan_d = kernels::attn::plan_decode(positions.rec(), inputs.kv_space());
        let plan_p = kernels::attn::plan_prefill(positions.rec(), inputs.kv_space());
        let write_page = inputs.geometry(inputs.kv_space(), GeomKind::WritePage);
        let write_offset = inputs.geometry(inputs.kv_space(), GeomKind::WriteOffset);
        let ids = inputs.tokens();
        let mut y = kernels::layout::embed(&ids, &m.embed, m.vocab);

        for (l, w) in inputs.layers(&m.layers) {
            let at = &w.attn;
            let d = at.head_dim;
            let pages = inputs.kv(&at.kv);

            let x = kernels::elemwise::rmsnorm(&y, &w.attn_norm, w.attn_norm_eps);
            let q =
                kernels::elemwise::add_bias(&at.q_bias, &kernels::linear::matmul(&x, &at.q_proj));
            let k =
                kernels::elemwise::add_bias(&at.k_bias, &kernels::linear::matmul(&x, &at.k_proj));
            let v =
                kernels::elemwise::add_bias(&at.v_bias, &kernels::linear::matmul(&x, &at.v_proj));
            seam::at(seam::ATTN_QV, (&q, &v));

            let (q, k) = kernels::elemwise::rope_yarn(
                &q,
                &k,
                &positions,
                d,
                at.theta,
                at.factor,
                at.beta_fast,
                at.beta_slow,
                at.attention_factor,
                at.original_max_position,
                false,
            );
            kernels::attn::kv_append(&k, &v, pages, &write_page, &write_offset);
            seam::at(seam::ATTN_Q, (&q,));

            let win = at.window;
            let (dq, p) = q.split(&Facts::qo_one());
            let a = merge![
                {
                    let (o, lse) =
                        kernels::attn::decode_lse(&dq, &plan_d, pages, win, d, at.sm_scale);
                    kernels::attn::sink(&o, &lse, &at.sinks, d)
                },
                {
                    let (o, lse) = kernels::attn::prefill_lse(
                        &p,
                        &plan_p,
                        pages,
                        win,
                        d,
                        at.kv_heads,
                        at.sm_scale,
                    );
                    kernels::attn::sink(&o, &lse, &at.sinks, d)
                },
            ];
            seam::at(seam::ATTN_OUT, (&a,));

            let o = kernels::linear::attention_landing(&a, &at.o_proj, l);
            let o = if m.tp > 1 {
                kernels::collective::all_reduce(&o)
            } else {
                o
            };
            y = kernels::elemwise::residual_add(&kernels::elemwise::add_bias(&at.o_bias, &o), &y);

            let e = &w.mlp;
            let x = kernels::elemwise::rmsnorm(&y, &w.mlp_norm, w.mlp_norm_eps);
            let (routes, weights) = kernels::linear::moe_topk_softmax(
                &kernels::elemwise::add_bias(
                    &e.router_bias,
                    &kernels::linear::matmul(&x, &e.router),
                ),
                e.experts,
                e.top_k,
            );
            let hidden = kernels::linear::moe_matmul_select_bias(
                &x,
                &e.gate_up,
                &e.gate_up_bias,
                &routes,
                e.top_k,
            );
            let act = kernels::linear::mlp_swiglu_clamp_alpha(
                &hidden,
                e.inter,
                e.swiglu_limit,
                e.swiglu_alpha,
            );
            let routed = kernels::linear::moe_matmul_select_bias(
                &act,
                &e.down,
                &e.down_bias,
                &routes,
                e.top_k,
            );
            let f = kernels::linear::moe_weighted_sum(&routed, &weights);
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
