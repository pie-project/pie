use model_dsl::{
    Classify, ForwardHybrid, GeomKind, HybridSpec, Input, Request, Value, merge, ops, seam,
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

        let positions = inputs.positions();
        let plan_p = ops::attn::plan_prefill(positions.rec(), inputs.kv_space());
        let write_page = inputs.geometry(inputs.kv_space(), GeomKind::WritePage);
        let write_offset = inputs.geometry(inputs.kv_space(), GeomKind::WriteOffset);
        let ids = inputs.tokens();
        let mut streams =
            ops::elemwise::hc_expand(&ops::layout::embed(&ids, &m.embed, m.vocab), hy.streams);

        for (_, w) in inputs.layers(&m.layers) {
            let at = &w.attn;
            let pages = inputs.kv(&at.kv);
            let pos = &positions;

            let (x, post_mix, comb_mix) = gate(&streams, &w.attn_mix, hy);

            let q = ops::linear::matmul(&x, &at.q_down);
            let q = ops::elemwise::rmsnorm(&q, &at.q_norm, at.q_norm_eps);
            let q = ops::linear::matmul(&q, &at.q_up);
            let q = ops::elemwise::rmsnorm_no_scale(&q, at.head_dim, at.q_norm_eps);

            let q =
                ops::elemwise::rope_partial_last(&q, pos, at.rope_dim, at.head_dim, at.theta, true);
            seam::at(seam::ATTN_Q, (&q,));

            let plane = ops::linear::matmul(&x, &at.kv_down);
            let plane = ops::elemwise::rmsnorm(&plane, &at.kv_norm, at.kv_norm_eps);
            let plane = ops::elemwise::rope_partial_last(
                &plane,
                pos,
                at.rope_dim,
                at.head_dim,
                at.theta,
                true,
            );
            ops::attn::kv_append_shared(&plane, pages, &write_page, &write_offset);

            let (o, lse) = ops::attn::prefill_lse(
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

                    let row_valid = inputs.geometry(inputs.kv_space(), GeomKind::RowValid);
                    let request_of_token =
                        inputs.geometry(inputs.kv_space(), GeomKind::RequestOfToken);
                    let (bpos, breq) = boundaries(pos, &row_valid, p.ratio);
                    let pooled =
                        ops::attn::pool_gather(&bpos, &breq, pages, at.head_dim, p.ratio, m.act);
                    let pooled = ops::elemwise::rope_partial_last(
                        &pooled,
                        pos,
                        at.rope_dim,
                        at.head_dim,
                        at.theta,
                        true,
                    );
                    ops::attn::pool_kv_append(
                        &pooled,
                        &bpos,
                        &breq,
                        entries,
                        &entry_page,
                        &entry_offset,
                    );
                    let (po, plse) = ops::attn::pool_lse(
                        &q,
                        pos,
                        &request_of_token,
                        entries,
                        p.ratio,
                        at.heads,
                        at.head_dim,
                        at.sm_scale,
                    );
                    ops::attn::merge_lse(&o, &lse, &po, &plse, at.heads, at.head_dim)
                }
                None => (o, lse),
            };
            let o = ops::attn::sink(&o, &lse, &at.sink, at.head_dim);
            seam::at(seam::ATTN_OUT, (&o,));

            let o = ops::linear::matmul(&o, &at.o_down);
            let o = if m.tp > 1 {
                ops::collective::all_reduce(&o)
            } else {
                o
            };
            let o = ops::linear::matmul(&o, &at.o_up);
            streams = ops::elemwise::hc_fold(&o, &streams, &post_mix, &comb_mix);

            let (x, post_mix, comb_mix) = gate(&streams, &w.mlp_mix, hy);
            let f = match &w.mlp {
                Mlp::Dense {
                    gate_up,
                    down,
                    inter,
                    limit,
                } => ops::linear::matmul(
                    &ops::linear::mlp_swiglu_clamp(
                        &ops::linear::matmul(&x, gate_up),
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
                    let (routes, weights) = ops::linear::moe_topk_sqrt_softplus(
                        &ops::linear::matmul(&x, router),
                        bias,
                        *experts,
                        *top_k,
                        *renorm,
                        *scaling,
                    );
                    let hidden = ops::linear::moe_matmul_select(&x, gate_up, &routes, *top_k);
                    let act = ops::linear::mlp_swiglu_clamp(&hidden, *inter, *limit);
                    ops::linear::moe_weighted_sum(
                        &ops::linear::moe_matmul_select(&act, down, &routes, *top_k),
                        &weights,
                    )
                }
            };
            let f = if m.tp > 1 {
                ops::collective::all_reduce(&f)
            } else {
                f
            };
            streams = ops::elemwise::hc_fold(&f, &streams, &post_mix, &comb_mix);
        }

        let (mut y, mut rest) = ops::layout::split_rows(&streams, m.hidden);
        for _ in 2..hy.streams {
            let (stream, more) = ops::layout::split_rows(&rest, m.hidden);
            y = ops::elemwise::residual_add(&stream, &y);
            rest = more;
        }
        let y = ops::elemwise::residual_add(&rest, &y);
        let x = ops::elemwise::rmsnorm(&y, &m.final_norm, m.final_norm_eps);
        ops::linear::lm_head(&x, &m.embed)
    }
}

fn gate(streams: &Value, mix: &Mix, hy: &Hyper) -> (Value, Value, Value) {
    let normed = ops::elemwise::hc_rmsnorm_f32(streams, hy.norm_eps);
    ops::elemwise::hc_gates(
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
    let (dpos, dreq) = ops::attn::pool_boundary_decode(&one, row_valid, ratio);
    let (ppos, preq) = ops::attn::pool_boundary_prefill(&many, row_valid, ratio);
    (merge![dpos, ppos], merge![dreq, preq])
}
