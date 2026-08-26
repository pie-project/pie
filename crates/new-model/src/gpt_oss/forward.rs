//! The gpt-oss forward pass over the typed IR — the old forward minus the
//! machinery the design killed (design §10): attention plans are built once
//! up front and shared visibly across layers (§6), kv-append geometry is a
//! declared input fetched once per forward (§7), raggedness is ambient so the
//! prefill arm loses its `query_windows` plumbing (§5), and tensor
//! parallelism is plain control flow on `m.tp` (§9, decision #18). The
//! sink-corrected lse arms and the clamped-alpha swiglu transcribe verbatim.

use new_model_dsl::{
    Classify, Facts, ForwardHybrid, HybridSpec, Input, Request, Value, kernels, merge, seam,
};
use new_model_ir::GeomKind;

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

impl ForwardHybrid for Model {
    type Facts = Facts;

    fn caches(&self) -> HybridSpec {
        let mut c = HybridSpec::new();
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
        let attn = AttnShared::of(m, &inputs);
        let ids = inputs.tokens();
        let mut y = kernels::layout::embed(&ids, &m.embed, m.vocab);

        for (l, w) in inputs.layers(&m.layers) {
            let at = &w.attn;
            let d = at.head_dim;
            let pages = inputs.kv(at.kv.name());

            let x = kernels::norm::rmsnorm(&y, &w.attn_norm.weight, w.attn_norm.eps);
            let q = kernels::norm::add_bias(&at.q_bias, &kernels::gemm::matmul(&x, &at.q_proj));
            let k = kernels::norm::add_bias(&at.k_bias, &kernels::gemm::matmul(&x, &at.k_proj));
            let v = kernels::norm::add_bias(&at.v_bias, &kernels::gemm::matmul(&x, &at.v_proj));
            seam::at(seam::ATTN_QV, (&q, &v));

            let r = &at.rope;
            let (q, k) = kernels::rope::yarn(
                &q,
                &k,
                &attn.positions,
                d,
                r.theta,
                r.factor,
                r.beta_fast,
                r.beta_slow,
                r.attention_factor,
                r.original_max_position,
                false,
            );
            kernels::attention::kv_append(&k, &v, pages, &attn.kv_indices, &attn.positions);
            seam::at(seam::ATTN_Q, (&q,));

            let win = at.window;
            let (dq, p) = q.split(&Facts::qo_one());
            let a = merge![
                {
                    let (o, lse) = kernels::attention::decode_lse(
                        &dq,
                        &attn.plan_d,
                        pages,
                        win,
                        d,
                        at.sm_scale,
                    );
                    kernels::attention::sink(&o, &lse, &at.sinks, d)
                },
                {
                    let (o, lse) = kernels::attention::prefill_lse(
                        &p,
                        &attn.plan_p,
                        pages,
                        win,
                        d,
                        at.kv_heads,
                        at.sm_scale,
                    );
                    kernels::attention::sink(&o, &lse, &at.sinks, d)
                },
            ];
            seam::at(seam::ATTN_OUT, (&a,));

            let o = kernels::gemm::attention_landing(&a, &at.o_proj, l);
            let o = if m.tp > 1 {
                kernels::dist::all_reduce(&o)
            } else {
                o
            };
            y = kernels::norm::residual_add(&kernels::norm::add_bias(&at.o_bias, &o), &y);

            let e = &w.mlp;
            let x = kernels::norm::rmsnorm(&y, &w.mlp_norm.weight, w.mlp_norm.eps);
            let (routes, weights) = kernels::moe::topk_softmax(
                &kernels::norm::add_bias(&e.router_bias, &kernels::gemm::matmul(&x, &e.router)),
                e.experts,
                e.top_k,
            );
            let hidden =
                kernels::moe::matmul_select_bias(&x, &e.gate_up, &e.gate_up_bias, &routes, e.top_k);
            let act =
                kernels::mlp::swiglu_clamp_alpha(&hidden, e.inter, e.swiglu_limit, e.swiglu_alpha);
            let routed =
                kernels::moe::matmul_select_bias(&act, &e.down, &e.down_bias, &routes, e.top_k);
            let f = kernels::moe::weighted_sum(&routed, &weights);
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

/// What every attention layer shares, stated once per forward: the decode and
/// prefill plans (§6), the page indices `kv_append` writes through (§7), and
/// the positions rope reads. The first layer's kv space stands in for all of
/// them — the fire lays every layer's kv pages out identically.
struct AttnShared {
    positions: Value,
    kv_indices: Value,
    plan_d: Value,
    plan_p: Value,
}

impl AttnShared {
    fn of(m: &Model, inputs: &Input<Facts>) -> AttnShared {
        let kv = m
            .layers
            .first()
            .map(|w| inputs.cache_index(w.attn.kv.name()))
            .expect("gpt-oss is all attention layers");
        let positions = inputs.positions();
        AttnShared {
            plan_d: kernels::attention::plan_decode(positions.rec(), kv),
            plan_p: kernels::attention::plan_prefill(positions.rec(), kv),
            kv_indices: inputs.geometry(kv, GeomKind::Indices),
            positions,
        }
    }
}
