//! The DeepSeek V4 forward pass over the typed IR — the old forward minus the
//! machinery the design killed (design §10): the prefill plan is built once
//! up front and shared visibly across layers (§6), kv-append geometry — the
//! shared plane's and each pool space's — is a declared input fetched where
//! it is used (§7), raggedness is ambient so the attention and boundary
//! statements lose their `query_windows` plumbing (§5), and tensor
//! parallelism is plain control flow on `m.tp` (§9, decision #18).

use model_dsl::{
    Classify, ForwardHybrid, GeomKind, HybridSpec, Input, Request, Value, kernels, merge, seam,
};

use super::model::{Hyper, Mix, Mlp, Model};

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
        // One paged space for the shared kv plane — the fire lays every
        // layer's kv pages out identically. The pool spaces share nothing:
        // each layer's ratio sets its own entry count, so every pooled layer
        // declares a space of its own.
        let kv = c.kv_space(self.kv);
        for (l, w) in self.layers.iter().enumerate() {
            let at = &w.attn;
            c.kv(
                kv,
                format!("kv.{l}"),
                [1, at.heads as u64 * at.head_dim as u64],
            );
            if at.pool.is_some() {
                let pool = c.kv_space(self.kv);
                c.kv(pool, format!("pool.{l}"), [1, at.head_dim as u64]);
            }
        }
        c
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let m = self;
        let hy = &m.hyper;
        // The old forward attends every row, decode included, through
        // `prefill_lse`, so the prefill plan is the only plan — built once
        // and shared visibly by every layer (§6).
        let positions = inputs.positions();
        let plan_p = kernels::attn::plan_prefill(positions.rec(), inputs.kv_space());
        let write_page = inputs.geometry(inputs.kv_space(), GeomKind::WritePage);
        let write_offset = inputs.geometry(inputs.kv_space(), GeomKind::WriteOffset);
        let ids = inputs.tokens();
        let mut streams = kernels::elemwise::hc_expand(
            &kernels::layout::embed(&ids, &m.embed, m.vocab),
            hy.streams,
        );

        for (_, w) in inputs.layers(&m.layers) {
            let at = &w.attn;
            let pages = inputs.kv(&at.kv);
            let pos = &positions;

            let (x, post_mix, comb_mix) = gate(&streams, &w.attn_mix, hy);

            let q = kernels::linear::matmul(&x, &at.q_down);
            let q = kernels::elemwise::rmsnorm(&q, &at.q_norm, at.q_norm_eps);
            let q = kernels::linear::matmul(&q, &at.q_up);
            let q = kernels::elemwise::rmsnorm_no_scale(&q, at.head_dim, at.q_norm_eps);

            let q = kernels::elemwise::rope_partial_last(
                &q,
                pos,
                at.rope_dim,
                at.head_dim,
                at.theta,
                true,
            );
            seam::at(seam::ATTN_Q, (&q,));

            let plane = kernels::linear::matmul(&x, &at.kv_down);
            let plane = kernels::elemwise::rmsnorm(&plane, &at.kv_norm, at.kv_norm_eps);
            let plane = kernels::elemwise::rope_partial_last(
                &plane,
                pos,
                at.rope_dim,
                at.head_dim,
                at.theta,
                true,
            );
            kernels::attn::kv_append_shared(&plane, pages, &write_page, &write_offset);

            let (o, lse) = kernels::attn::prefill_lse(
                &q,
                &plan_p,
                pages,
                Some(at.window),
                at.head_dim,
                at.heads,
                at.sm_scale,
            );

            let (o, lse) = match &at.pool {
                Some(p) => {
                    let entries = inputs.kv(&p.entries);
                    let pool_space = inputs.space_of(&p.entries);
                    let entry_page = inputs.geometry(pool_space, GeomKind::WritePage);
                    let entry_offset = inputs.geometry(pool_space, GeomKind::WriteOffset);
                    // The padding mask and the token→lane map are fire
                    // tables, not page-table geometry: every space sees the
                    // same rows, so the shared kv space is their honest seat.
                    let row_valid = inputs.geometry(inputs.kv_space(), GeomKind::RowValid);
                    let request_of_token =
                        inputs.geometry(inputs.kv_space(), GeomKind::RequestOfToken);
                    let (bpos, breq) = boundaries(pos, &row_valid, p.ratio);
                    let pooled = kernels::attn::pool_gather(
                        &bpos,
                        &breq,
                        pages,
                        at.head_dim,
                        p.ratio,
                        m.act,
                    );
                    let pooled = kernels::elemwise::rope_partial_last(
                        &pooled,
                        pos,
                        at.rope_dim,
                        at.head_dim,
                        at.theta,
                        true,
                    );
                    kernels::attn::pool_kv_append(
                        &pooled,
                        &bpos,
                        &breq,
                        entries,
                        &entry_page,
                        &entry_offset,
                    );
                    let (po, plse) = kernels::attn::pool_lse(
                        &q,
                        pos,
                        &request_of_token,
                        entries,
                        p.ratio,
                        at.heads,
                        at.head_dim,
                        at.sm_scale,
                    );
                    kernels::attn::merge_lse(&o, &lse, &po, &plse, at.heads, at.head_dim)
                }
                None => (o, lse),
            };
            let o = kernels::attn::sink(&o, &lse, &at.sink, at.head_dim);
            seam::at(seam::ATTN_OUT, (&o,));

            let o = kernels::linear::matmul(&o, &at.o_down);
            let o = if m.tp > 1 {
                kernels::collective::all_reduce(&o)
            } else {
                o
            };
            let o = kernels::linear::matmul(&o, &at.o_up);
            streams = kernels::elemwise::hc_fold(&o, &streams, &post_mix, &comb_mix);

            let (x, post_mix, comb_mix) = gate(&streams, &w.mlp_mix, hy);
            let f = match &w.mlp {
                Mlp::Dense {
                    gate_up,
                    down,
                    inter,
                    limit,
                } => kernels::linear::matmul(
                    &kernels::linear::mlp_swiglu_clamp(
                        &kernels::linear::matmul(&x, gate_up),
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
                    let (routes, weights) = kernels::linear::moe_topk_sqrt_softplus(
                        &kernels::linear::matmul(&x, router),
                        bias,
                        *experts,
                        *top_k,
                        *renorm,
                        *scaling,
                    );
                    let hidden = kernels::linear::moe_matmul_select(&x, gate_up, &routes, *top_k);
                    let act = kernels::linear::mlp_swiglu_clamp(&hidden, *inter, *limit);
                    kernels::linear::moe_weighted_sum(
                        &kernels::linear::moe_matmul_select(&act, down, &routes, *top_k),
                        &weights,
                    )
                }
            };
            let f = if m.tp > 1 {
                kernels::collective::all_reduce(&f)
            } else {
                f
            };
            streams = kernels::elemwise::hc_fold(&f, &streams, &post_mix, &comb_mix);
        }

        // `Hc::Collapse` is deleted (review R5): the head-gate plane it read
        // has no producer, and the import ships no bank one could come from.
        // The tail is the base hyper-connections collapse — the streams
        // summed back into one — spelled with the splits and adds the IR
        // already has.
        let (mut y, mut rest) = kernels::layout::split_rows(&streams, m.hidden);
        for _ in 2..hy.streams {
            let (stream, more) = kernels::layout::split_rows(&rest, m.hidden);
            y = kernels::elemwise::residual_add(&stream, &y);
            rest = more;
        }
        let y = kernels::elemwise::residual_add(&rest, &y);
        let x = kernels::elemwise::rmsnorm(&y, &m.final_norm, m.final_norm_eps);
        kernels::linear::lm_head(&x, &m.embed)
    }
}

fn gate(streams: &Value, mix: &Mix, hy: &Hyper) -> (Value, Value, Value) {
    let normed = kernels::elemwise::hc_rmsnorm_f32(streams, hy.norm_eps);
    kernels::elemwise::hc_gates(
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

fn boundaries(positions: &Value, row_valid: &Value, ratio: u32) -> (Value, Value) {
    let (one, many) = positions.split(&Facts::qo_one());
    let (dpos, dreq) = kernels::attn::pool_boundary_decode(&one, row_valid, ratio);
    let (ppos, preq) = kernels::attn::pool_boundary_prefill(&many, row_valid, ratio);
    (merge![dpos, ppos], merge![dreq, preq])
}
