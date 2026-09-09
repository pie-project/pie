use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, Predicate, Request, Value, ValueId, Weight,
    ops, seam,
};

use super::model::{Gate, GateUp, Hyper, Indexer, Mix, Mlp, Model};

pub struct Facts {
    pub qo_one: bool,
    pub has_adapter: bool,
    pub drafts: bool,
}

impl Facts {
    pub fn qo_one() -> Predicate {
        Predicate::fact(0)
    }

    pub fn has_adapter() -> Predicate {
        Predicate::fact(1)
    }

    pub fn drafts() -> Predicate {
        Predicate::fact(2)
    }
}

impl Classify for Facts {
    fn of(r: &Request) -> Facts {
        Facts {
            qo_one: r.query_len() == 1,
            has_adapter: r.has_adapter(),
            drafts: r.drafts(),
        }
    }

    fn word(&self) -> u64 {
        u64::from(self.qo_one) | (u64::from(self.has_adapter) << 1) | (u64::from(self.drafts) << 2)
    }
}

impl ForwardHybrid for Model {
    type Facts = Facts;

    fn caches(&self) -> HybridSpec {
        let mut c = HybridSpec::new();

        let kv = c.kv_space(self.kv);
        for w in &self.layers {
            let at = &w.attn;
            c.kv(kv, at.kv.clone(), [at.kv_down.dim(0)]);
            if let Some(p) = &at.pool {
                let pool = c.kv_space(self.kv);
                c.kv(pool, p.entries.clone(), [self.head_dim as u64]);
            }
            if let Some(ix) = &at.indexer {
                let index = c.kv_space(self.kv);
                c.kv(index, ix.keys.clone(), [ix.head_dim as u64]);
            }
        }
        if let Some(mtp) = &self.mtp {
            c.kv(
                kv,
                mtp.block.attn.kv.clone(),
                [mtp.block.attn.kv_down.dim(0)],
            );
        }
        c
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let m = self;
        let hy = &m.hyper;

        let positions = inputs.positions();
        let kv_heads = kv_heads(m);
        let plan_p =
            ops::attn::plan_prefill(&inputs, m.heads, kv_heads, m.head_dim, Some(m.window));
        let ids = inputs.tokens();
        let mut streams =
            ops::elemwise::hc_expand(&ops::layout::embed(&ids, &m.embed, m.vocab), hy.streams);

        let adapter_routes = inputs.adapter_routes();
        for (l, w) in inputs.walk_layers(&m.layers) {
            let next = m.layers.get(l as usize + 1);
            streams = layer(
                m,
                &inputs,
                &plan_p,
                &positions,
                &adapter_routes,
                w,
                next,
                &streams,
                &ids,
                false,
            );
        }

        let y = match &m.hc_head {
            Some(hc) => {
                let normed = ops::elemwise::hc_rmsnorm_f32(&streams, hy.norm_eps);
                let mixes = ops::elemwise::hc_project(&normed, &hc.dynamic, hy.streams);
                ops::elemwise::hc_collapse(
                    &mixes,
                    &streams,
                    &hc.scale,
                    &hc.base,
                    hy.streams,
                    hy.gate_eps,
                )
            }
            None => {
                let (mut y, mut rest) = ops::layout::split_rows(&streams, m.hidden);
                for _ in 1..hy.streams - 1 {
                    let (stream, more) = ops::layout::split_rows(&rest, m.hidden);
                    y = ops::elemwise::residual_add(&stream, &y);
                    rest = more;
                }
                ops::elemwise::residual_add(&rest, &y)
            }
        };
        let x = ops::elemwise::rmsnorm(&y, &m.final_norm, m.final_norm_eps);
        let logits = match &m.head {
            Some(head) => ops::linear::lm_head(&x, head),
            None => ops::linear::lm_head(&x, &m.embed),
        };

        if let (Some(mtp), Some(head)) = (&m.mtp, &m.head) {
            let (input_mtp, _) = inputs.split(&Facts::drafts());
            let plan_mtp =
                ops::attn::plan_prefill(&input_mtp, m.heads, kv_heads, m.head_dim, Some(m.window));
            let (dstreams, _) = streams.split(&Facts::drafts());
            let (dpos, _) = positions.split(&Facts::drafts());
            let (dlogits, _) = logits.split(&Facts::drafts());

            let mut token = ops::layout::argmax(&[&dlogits]);
            let mut hidden = dstreams;
            let mut chain: Vec<Value> = Vec::with_capacity(mtp.depth as usize);
            for step in 0..mtp.depth {
                let e = ops::layout::embed(&token, &m.embed, m.vocab);
                let e = ops::elemwise::rmsnorm(&e, &mtp.enorm, mtp.norm_eps);
                let e = ops::elemwise::hc_expand(&ops::linear::matmul(&e, &mtp.e_proj), hy.streams);
                let h =
                    ops::elemwise::rmsnorm_per_head(&hidden, &mtp.hnorm, m.hidden, mtp.norm_eps);
                let routes = ops::linear::group_routes(&h, hy.streams);
                let h = ops::linear::matmul_grouped(&h, &mtp.h_proj, &routes, hy.streams);
                let fused = ops::elemwise::residual_add(&e, &h);

                let out = layer(
                    m,
                    &input_mtp,
                    &plan_mtp,
                    &dpos,
                    &adapter_routes,
                    &mtp.block,
                    None,
                    &fused,
                    &token,
                    step > 0,
                );
                let normed = ops::elemwise::hc_rmsnorm_f32(&out, hy.norm_eps);
                let mixes = ops::elemwise::hc_project(&normed, &mtp.hc_head.dynamic, hy.streams);
                let dy = ops::elemwise::hc_collapse(
                    &mixes,
                    &out,
                    &mtp.hc_head.scale,
                    &mtp.hc_head.base,
                    hy.streams,
                    hy.gate_eps,
                );
                let read = ops::elemwise::rmsnorm(&dy, &mtp.norm, mtp.norm_eps);
                let draft = ops::linear::lm_head(&read, head);
                if step == 0 {
                    seam::at(seam::MTP, &[&draft]);
                }
                token = ops::layout::argmax(&[&draft]);
                hidden = out;
                chain.push(draft);
            }
            let steps: Vec<&Value> = chain.iter().collect();
            seam::at(seam::MTP_DRAFTS, &[&ops::layout::argmax(&steps)]);
        }

        logits
    }
}

#[allow(clippy::too_many_arguments)]
fn layer(
    m: &Model,
    inputs: &Input<Facts>,
    plan_p: &Value,
    positions: &Value,
    adapter_routes: &Value,
    w: &super::model::Layer,
    next: Option<&super::model::Layer>,
    streams: &Value,
    ids: &Value,
    chain: bool,
) -> Value {
    let hy = &m.hyper;
    let kv_heads = kv_heads(m);
    let pos = positions;
    let at = &w.attn;
    let pages = inputs.kv(&at.kv);
    let write_page = inputs.write_page(&at.kv);
    let write_offset = inputs.write_offset(&at.kv);

    let (x, post_mix, comb_mix) = gate(streams, &w.attn_mix, hy);
    let x = match &w.attn_norm {
        Some(n) => ops::elemwise::rmsnorm(&x, n, hy.norm_eps),
        None => x,
    };

    let q_a = ops::linear::matmul(&x, &at.q_down);
    let q_a = ops::elemwise::rmsnorm(&q_a, &at.q_norm, at.q_norm_eps);
    let q = ops::linear::matmul(&q_a, &at.q_up);
    let q = ops::elemwise::rmsnorm_no_scale(&q, m.head_dim, at.q_norm_eps);

    let q = ops::elemwise::rope_partial_last_yarn(
        &q,
        pos,
        at.rope_dim,
        m.head_dim,
        at.theta,
        true,
        false,
        at.yarn,
    );
    seam::at(seam::ATTN_Q, &[&q]);

    let plane = ops::linear::matmul(&x, &at.kv_down);
    let plane = ops::elemwise::rmsnorm(&plane, &at.kv_norm, at.kv_norm_eps);
    let plane = ops::elemwise::rope_partial_last_yarn(
        &plane,
        pos,
        at.rope_dim,
        m.head_dim,
        at.theta,
        true,
        false,
        at.yarn,
    );
    if !chain {
        ops::attn::kv_append_shared(&plane, pages, &write_page, &write_offset);
    }

    let (o, lse) = ops::attn::prefill_lse(
        &q,
        plan_p,
        pages,
        Some(m.window),
        m.head_dim,
        kv_heads,
        at.sm_scale,
    );

    let (o, lse) = match &at.pool {
        Some(p) => {
            let ape = p.compressor.as_ref().map(|c| {
                if !chain {
                    let state_kv = ops::linear::matmul(&x, &c.wkv);
                    let state_score = ops::linear::matmul(&x, &c.wgate);
                    ops::attn::pool_state_write(
                        &state_kv,
                        &state_score,
                        pages,
                        &write_page,
                        &write_offset,
                        m.head_dim,
                        p.ratio,
                    );
                }
                &c.ape
            });
            let entries = inputs.kv(&p.entries);
            let entry_page = inputs.write_page(&p.entries);
            let entry_offset = inputs.write_offset(&p.entries);

            let row_valid = inputs.row_valid();
            let request_of_token = inputs.request_of_token();
            let (bpos, breq, brope) = boundaries(pos, &row_valid, p.ratio);
            let pooled =
                ops::attn::pool_gather(&bpos, &breq, pages, ape, m.head_dim, p.ratio, m.act);
            let pooled = match &p.compressor {
                Some(c) => ops::elemwise::rmsnorm(&pooled, &c.norm, c.norm_eps),
                None => pooled,
            };
            let pooled = ops::elemwise::rope_partial_last_yarn(
                &pooled,
                &brope,
                at.rope_dim,
                m.head_dim,
                at.theta,
                true,
                false,
                at.yarn,
            );
            if !chain {
                ops::attn::pool_kv_append(
                    &pooled,
                    &bpos,
                    &breq,
                    entries,
                    &entry_page,
                    &entry_offset,
                );
            }
            let selection = at.indexer.as_ref().map(|ix| {
                indexer(
                    &x,
                    &q_a,
                    ix,
                    pos,
                    &bpos,
                    &breq,
                    &brope,
                    inputs.kv(&ix.keys),
                    &inputs.write_page(&ix.keys),
                    &inputs.write_offset(&ix.keys),
                    m.act,
                    chain,
                )
            });
            let (po, plse) = match (&selection, &at.indexer) {
                (Some(selection), Some(ix)) => ops::attn::pool_lse_selected(
                    &q,
                    pos,
                    &request_of_token,
                    selection,
                    entries,
                    p.ratio,
                    ix.top_k,
                    m.heads,
                    m.head_dim,
                    at.sm_scale,
                ),
                _ => ops::attn::pool_lse(
                    &q,
                    pos,
                    &request_of_token,
                    entries,
                    p.ratio,
                    m.heads,
                    m.head_dim,
                    at.sm_scale,
                ),
            };
            ops::attn::merge_lse(&o, &lse, &po, &plse, m.heads, m.head_dim)
        }
        None => (o, lse),
    };
    let o = ops::attn::sink(&o, &lse, &at.sink, m.head_dim);
    let o = ops::elemwise::rope_partial_last_yarn(
        &o,
        pos,
        at.rope_dim,
        m.head_dim,
        at.theta,
        true,
        true,
        at.yarn,
    );
    seam::at(seam::ATTN_OUT, &[&o]);

    let o = if at.o_groups > 1 {
        let routes = ops::linear::group_routes(&o, at.o_groups);
        ops::linear::matmul_grouped(&o, &at.o_down, &routes, at.o_groups)
    } else {
        ops::linear::matmul(&o, &at.o_down)
    };
    let o = if m.tp > 1 {
        ops::collective::all_reduce(&o)
    } else {
        o
    };
    let o = ops::linear::matmul(&o, &at.o_up);
    let o = {
        let (adapted, _) = o.split(&Facts::has_adapter());
        let (px, _) = x.split(&Facts::has_adapter());
        ops::linear::lora_correct(&px, &w.lora_a, &w.lora_b, adapter_routes, &adapted)
    };
    let streams = ops::elemwise::hc_fold(&o, streams, &post_mix, &comb_mix);

    let (x, post_mix, comb_mix) = gate(&streams, &w.mlp_mix, hy);
    let x = match &w.mlp_norm {
        Some(n) => ops::elemwise::rmsnorm(&x, n, hy.norm_eps),
        None => x,
    };
    let f = mlp(&x, ids, &w.mlp, &streams, next, hy);
    let f = if m.tp > 1 {
        ops::collective::all_reduce(&f)
    } else {
        f
    };
    ops::elemwise::hc_fold(&f, &streams, &post_mix, &comb_mix)
}

const PREDICT_K: u32 = 16;

fn predict_next(streams: &Value, next: Option<&super::model::Layer>, hy: &Hyper) -> Option<Value> {
    let next = next?;
    let Mlp::MoeFlash {
        router,
        gate: Gate::Bias { bias },
        experts,
        ..
    } = &next.mlp
    else {
        return None;
    };
    let (px, _, _) = gate(streams, &next.mlp_mix, hy);
    let px = match &next.mlp_norm {
        Some(n) => ops::elemwise::rmsnorm(&px, n, hy.norm_eps),
        None => px,
    };
    let logits = ops::linear::matmul(&px, router);
    Some(ops::linear::moe_predict_route(
        &logits, bias, *experts, PREDICT_K,
    ))
}

fn mlp(
    x: &Value,
    ids: &Value,
    mlp: &Mlp,
    streams: &Value,
    next: Option<&super::model::Layer>,
    hy: &Hyper,
) -> Value {
    match mlp {
        Mlp::Dense {
            gate_up,
            down,
            inter,
            limit,
        } => ops::linear::matmul(
            &ops::linear::mlp_swiglu_clamp(&ops::linear::matmul(x, gate_up), *inter, *limit),
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
                &ops::linear::matmul(x, router),
                bias,
                *experts,
                *top_k,
                *renorm,
                *scaling,
            );
            let hidden = ops::linear::moe_matmul_select(x, gate_up, &routes, *top_k);
            let act = ops::linear::mlp_swiglu_clamp(&hidden, *inter, *limit);
            ops::linear::moe_weighted_sum(
                &ops::linear::moe_matmul_select(&act, down, &routes, *top_k),
                &weights,
            )
        }
        Mlp::MoeFlash {
            router,
            gate,
            gate_up,
            down,
            shared_gate_up,
            shared_down,
            experts,
            top_k,
            inter,
            shared_inter,
            limit,
            renorm,
            scaling,
        } => {
            let (routes, weights) = match gate {
                Gate::Bias { bias } => {
                    let hint = predict_next(streams, next, hy);
                    ops::linear::moe_topk_sqrt_softplus_hinted(
                        &ops::linear::matmul(x, router),
                        bias,
                        *experts,
                        *top_k,
                        *renorm,
                        *scaling,
                        hint.as_ref(),
                    )
                }
                Gate::Hash { tid2eid } => {
                    let vocab = u32::try_from(tid2eid.dim(0)).expect("a vocabulary no u32 holds");
                    ops::linear::moe_hash_route(
                        ids,
                        tid2eid,
                        &ops::linear::matmul(x, router),
                        vocab,
                        *experts,
                        *top_k,
                        *renorm,
                        *scaling,
                    )
                }
            };
            let shared = ops::linear::matmul(
                &ops::linear::mlp_swiglu_clamp(
                    &ops::linear::matmul(x, shared_gate_up),
                    *shared_inter,
                    *limit,
                ),
                shared_down,
            );
            let select = |act: &Value, bank: &Weight| {
                if matches!(bank.dtype, Dtype::Bf16 | Dtype::F16 | Dtype::F32) {
                    ops::linear::moe_matmul_select(act, bank, &routes, *top_k)
                } else {
                    ops::linear::moe_matmul_select_quant(act, bank, &routes, *top_k)
                }
            };
            let act = match gate_up {
                GateUp::Fused(bank) => {
                    ops::linear::mlp_swiglu_clamp(&select(x, bank), *inter, *limit)
                }
                GateUp::Split { gate, up } => {
                    ops::linear::mlp_swiglu_clamp_split(&select(x, gate), &select(x, up), *limit)
                }
            };
            let routed = ops::linear::moe_weighted_sum(&select(&act, down), &weights);
            ops::elemwise::residual_add(&shared, &routed)
        }
    }
}

fn kv_heads(m: &Model) -> u32 {
    let Some(w) = m.layers.first() else {
        return m.heads;
    };
    let row = w.attn.kv_down.dim(0);
    let head = u64::from(m.head_dim);
    assert!(
        head > 0 && row % head == 0,
        "the cached row is {row} wide and the head width is {head}, which is no \
         whole number of heads"
    );
    u32::try_from(row / head).expect("a head count inside u32")
}

#[allow(clippy::too_many_arguments)]
fn indexer(
    x: &Value,
    q_a: &Value,
    ix: &Indexer,
    positions: &Value,
    boundary_pos: &Value,
    boundary_req: &Value,
    boundary_rope: &Value,
    keys: ValueId,
    write_page: &Value,
    write_offset: &Value,
    act: Dtype,
    chain: bool,
) -> Value {
    let c = &ix.compressor;
    let ratio = c.ape.dim(0);
    let ratio = u32::try_from(ratio).expect("a pooling ratio inside u32");

    if !chain {
        let state_kv = ops::linear::matmul(x, &c.wkv);
        let state_score = ops::linear::matmul(x, &c.wgate);
        ops::attn::pool_state_write(
            &state_kv,
            &state_score,
            keys,
            write_page,
            write_offset,
            ix.head_dim,
            ratio,
        );
    }
    let k = ops::attn::pool_gather(
        boundary_pos,
        boundary_req,
        keys,
        Some(&c.ape),
        ix.head_dim,
        ratio,
        act,
    );
    let k = ops::elemwise::rmsnorm(&k, &c.norm, c.norm_eps);
    let k = ops::elemwise::rope_partial_last_yarn(
        &k,
        boundary_rope,
        ix.rope_dim,
        ix.head_dim,
        ix.theta,
        true,
        false,
        ix.yarn,
    );
    if !chain {
        ops::attn::pool_kv_append(
            &k,
            boundary_pos,
            boundary_req,
            keys,
            write_page,
            write_offset,
        );
    }

    let q = ops::linear::matmul(q_a, &ix.wq_b);
    let q = ops::elemwise::rope_partial_last_yarn(
        &q,
        positions,
        ix.rope_dim,
        ix.head_dim,
        ix.theta,
        true,
        false,
        ix.yarn,
    );
    let weights = ops::linear::matmul(x, &ix.weights_proj);
    ops::attn::index_topk(&q, &weights, keys, ix.heads, ix.head_dim, ix.top_k, ratio)
}

fn gate(streams: &Value, mix: &Mix, hy: &Hyper) -> (Value, Value, Value) {
    let normed = ops::elemwise::hc_rmsnorm_f32(streams, hy.norm_eps);
    let mixes = match &mix.dynamic {
        Some(dynamic) => ops::elemwise::hc_project(&normed, dynamic, hy.streams),
        None => normed,
    };
    ops::elemwise::hc_gates(
        &mixes,
        streams,
        &mix.scale,
        &mix.base,
        hy.streams,
        hy.gate_eps,
        hy.alpha,
        hy.sinkhorn,
    )
}

fn boundaries(positions: &Value, row_valid: &Value, ratio: u32) -> (Value, Value, Value) {
    let (one, many) = positions.split(&Facts::qo_one());
    let (dpos, dreq, drope) = ops::attn::pool_boundary_decode(&one, row_valid, ratio);
    let (ppos, preq, prope) = ops::attn::pool_boundary_prefill(&many, row_valid, ratio);
    (
        Value::merge(vec![dpos, ppos]),
        Value::merge(vec![dreq, preq]),
        Value::merge(vec![drope, prope]),
    )
}
