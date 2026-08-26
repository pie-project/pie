//! The DeepSeek V4 forward pass over the typed IR — the old forward minus the
//! machinery the design killed (design §10): the prefill plan is built once
//! up front and shared visibly across layers (§6), kv-append geometry — the
//! shared plane's and each pool space's — is a declared input (§7),
//! raggedness is ambient so the attention and boundary statements lose their
//! `query_windows` plumbing (§5), and tensor parallelism is plain control
//! flow on `m.tp` (§9, decision #18).

use new_model_dsl::{
    Classify, Facts, ForwardHybrid, HybridSpec, Input, Request, Value, kernels, merge, seam,
};
use new_model_ir::GeomKind;

use super::model::{Hyper, Mix, Mlp, Model};

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
        let attn = AttnShared::of(m, &inputs);
        let ids = inputs.tokens();
        let mut streams =
            kernels::hc::expand(&kernels::layout::embed(&ids, &m.embed, m.vocab), hy.streams);

        for (_, w) in inputs.layers(&m.layers) {
            let at = &w.attn;
            let pages = inputs.kv(at.kv.name());
            let pos = &attn.positions;

            let (x, post_mix, comb_mix) = gate(&streams, &w.attn_mix, hy);

            let q = kernels::gemm::matmul(&x, &at.q_down);
            let q = kernels::norm::rmsnorm(&q, &at.q_norm.weight, at.q_norm.eps);
            let q = kernels::gemm::matmul(&q, &at.q_up);
            let q = kernels::norm::rmsnorm_no_scale(&q, at.head_dim, at.q_norm.eps);

            let q = kernels::rope::partial_last(&q, pos, at.rope_dim, at.head_dim, at.theta, true);
            seam::at(seam::ATTN_Q, (&q,));

            let plane = kernels::gemm::matmul(&x, &at.kv_down);
            let plane = kernels::norm::rmsnorm(&plane, &at.kv_norm.weight, at.kv_norm.eps);
            let plane =
                kernels::rope::partial_last(&plane, pos, at.rope_dim, at.head_dim, at.theta, true);
            kernels::attention::kv_append_shared(&plane, pages, &attn.kv_indices, pos);

            let (o, lse) = kernels::attention::prefill_lse(
                &q,
                &attn.plan_p,
                pages,
                Some(at.window),
                at.head_dim,
                at.heads,
                at.sm_scale,
            );

            let (o, lse) = match &at.pool {
                Some(p) => {
                    let entries = inputs.kv(p.entries.name());
                    let entry_indices =
                        inputs.geometry(inputs.cache_index(p.entries.name()), GeomKind::Indices);
                    let (bpos, breq) = boundaries(pos, p.ratio);
                    let pooled =
                        kernels::pool::gather(&bpos, &breq, pages, at.head_dim, p.ratio, q.dtype());
                    let pooled = kernels::rope::partial_last(
                        &pooled,
                        pos,
                        at.rope_dim,
                        at.head_dim,
                        at.theta,
                        true,
                    );
                    kernels::pool::kv_append(&pooled, &bpos, &breq, entries, &entry_indices);
                    let (po, plse) = kernels::pool::attention_lse(
                        &q,
                        pos,
                        entries,
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
            seam::at(seam::ATTN_OUT, (&o,));

            let o = kernels::gemm::matmul(&o, &at.o_down);
            let o = if m.tp > 1 {
                kernels::dist::all_reduce(&o)
            } else {
                o
            };
            let o = kernels::gemm::matmul(&o, &at.o_up);
            streams = kernels::hc::fold(&o, &streams, &post_mix, &comb_mix);

            let (x, post_mix, comb_mix) = gate(&streams, &w.mlp_mix, hy);
            let f = match &w.mlp {
                Mlp::Dense {
                    gate_up,
                    down,
                    inter,
                    limit,
                } => kernels::gemm::matmul(
                    &kernels::mlp::swiglu_clamp(
                        &kernels::gemm::matmul(&x, gate_up),
                        *inter,
                        *limit,
                    ),
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
                    let hidden = kernels::moe::matmul_select(&x, gate_up, &routes, *top_k);
                    let act = kernels::mlp::swiglu_clamp(&hidden, *inter, *limit);
                    kernels::moe::weighted_sum(
                        &kernels::moe::matmul_select(&act, down, &routes, *top_k),
                        &weights,
                    )
                }
            };
            let f = if m.tp > 1 {
                kernels::dist::all_reduce(&f)
            } else {
                f
            };
            streams = kernels::hc::fold(&f, &streams, &post_mix, &comb_mix);
        }

        let y = kernels::hc::collapse(
            &streams,
            &hy.head_scale,
            &hy.head_base,
            hy.streams,
            hy.gate_eps,
        );
        let fin = &m.final_norm;
        let x = kernels::norm::rmsnorm(&y, &fin.weight, fin.eps);
        kernels::gemm::lm_head(&x, &m.embed)
    }
}

/// What every attention layer shares, stated once per forward: the prefill
/// plan (§6) — the old forward attends every row, decode included, through
/// `prefill_lse`, so it is the only plan — the page indices
/// `kv_append_shared` writes through (§7), and the positions each rope and
/// boundary statement reads. The first layer's kv space stands in for all of
/// them — the fire lays every layer's kv pages out identically. The pool
/// spaces share nothing here: each layer's ratio sets its own entry count,
/// so its geometry is fetched beside its cache.
struct AttnShared {
    positions: Value,
    kv_indices: Value,
    plan_p: Value,
}

impl AttnShared {
    fn of(m: &Model, inputs: &Input<Facts>) -> AttnShared {
        let kv = inputs.cache_index(
            m.layers
                .first()
                .map(|w| w.attn.kv.name())
                .expect("deepseek v4 attends in every layer"),
        );
        let positions = inputs.positions();
        AttnShared {
            plan_p: kernels::attention::plan_prefill(positions.rec(), kv),
            kv_indices: inputs.geometry(kv, GeomKind::Indices),
            positions,
        }
    }
}

fn gate(streams: &Value, mix: &Mix, hy: &Hyper) -> (Value, Value, Value) {
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

fn boundaries(positions: &Value, ratio: u32) -> (Value, Value) {
    let (one, many) = positions.split(&Facts::qo_one());
    let (dpos, dreq) = kernels::pool::boundary_decode(&one, ratio);
    let (ppos, preq) = kernels::pool::boundary_prefill(&many, ratio);
    (merge![dpos, ppos], merge![dreq, preq])
}
