use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, Predicate, Request, Value, ValueId, Weight,
    ops, seam,
};

use super::model::{Engram, Gate, GateUp, Hyper, Indexer, Mix, Mlp, Model, Pool, Selection};

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

/// What one sub-block hands the next under V4.1: the mixing coefficients it
/// predicted (Single-Pass mHC reads the residual once, so a sub-block's input
/// mix is the previous sub-block's prediction) and the latest sparse selection
/// (CSA2 Reuse Mode layers attend over the ranking a preceding layer made).
struct Carry<'m> {
    mix: Option<PrevMix<'m>>,
    selection: Option<(Value, u32)>,
}

struct PrevMix<'m> {
    mixes: Value,
    scale: &'m Weight,
    base: &'m Weight,
}

impl ForwardHybrid for Model {
    type Facts = Facts;

    fn caches(&self) -> HybridSpec {
        let mut c = HybridSpec::new();

        let kv = c.kv_space(self.kv);
        for w in &self.layers {
            let at = &w.attn;
            c.kv(kv, at.kv.clone(), [at.kv_down.dim(0)]);
            if let Some(p) = &at.pool
                && p.owner
            {
                let pool = c.kv_space(self.kv);
                c.kv(pool, p.entries.clone(), [self.head_dim as u64]);
            }
            if let Some(ix) = &at.indexer
                && ix.owns_keys
            {
                let index = c.kv_space(self.kv);
                c.kv(index, ix.keys.clone(), [ix.head_dim as u64]);
            }
            if let Some(e) = &w.engram {
                c.state(e.ids_state.clone(), [u64::from(e.ngram) - 1], Dtype::I32);
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
        let mut carry = Carry {
            mix: None,
            selection: None,
        };
        for (l, w) in inputs.walk_layers(&m.layers) {
            if let Some(e) = &w.engram {
                streams = engram(&streams, &ids, &inputs, m, e);
            }
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
                &mut carry,
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
            None if hy.single_pass => premix(&streams, carry.mix.as_ref(), m),
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
        let x = ops::layout::gather_rows(&x, &inputs.readout_rows());
        let logits = match &m.head {
            Some(head) => ops::linear::lm_head(&x, head),
            None => ops::linear::lm_head(&x, &m.embed),
        };

        if let (Some(mtp), Some(head)) = (&m.mtp, &m.head) {
            let (input_mtp, _) = inputs.split(&Facts::drafts());
            let plan_mtp =
                ops::attn::plan_prefill(&input_mtp, m.heads, kv_heads, m.head_dim, Some(m.window));
            let (dstreams, _) =
                ops::layout::gather_rows(&streams, &inputs.readout_rows()).split(&Facts::drafts());
            let (dpos, _) = positions.split(&Facts::drafts());
            let (dlogits, _) = logits.split(&Facts::drafts());

            let mut token = ops::layout::argmax(&[&dlogits]);
            let mut hidden = dstreams;
            let mut chain: Vec<Value> = Vec::with_capacity(mtp.depth as usize);
            let mut draft_carry = Carry {
                mix: None,
                selection: None,
            };
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
                    &mut draft_carry,
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
fn layer<'m>(
    m: &'m Model,
    inputs: &Input<Facts>,
    plan_p: &Value,
    positions: &Value,
    adapter_routes: &Value,
    w: &'m super::model::Layer,
    next: Option<&super::model::Layer>,
    streams: &Value,
    ids: &Value,
    chain: bool,
    carry: &mut Carry<'m>,
) -> Value {
    let hy = &m.hyper;
    let kv_heads = kv_heads(m);
    let pos = positions;
    let at = &w.attn;
    let pages = inputs.kv(&at.kv);
    let write_page = inputs.write_page(&at.kv);
    let write_offset = inputs.write_offset(&at.kv);

    let (x, post_mix, comb_mix) = sublayer_input(streams, &w.attn_mix, m, carry);
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
            let entries = inputs.kv(&p.entries);
            let row_valid = inputs.row_valid();
            let request_of_token = inputs.request_of_token();
            let (bpos, breq, brope) = boundaries(pos, &row_valid, p.ratio);

            if p.owner {
                let latent = compress(
                    m,
                    &x,
                    p,
                    pages,
                    &write_page,
                    &write_offset,
                    &bpos,
                    &breq,
                    chain,
                );
                if let Some(ix) = &at.indexer
                    && let (Some(wk), Some(k_norm)) = (&ix.wk, &ix.k_norm)
                    && ix.owns_keys
                    && !chain
                {
                    // CSA2: the index keys are a projection of the compressed
                    // latent, taken before its rotation.
                    let k = ops::elemwise::rmsnorm(
                        &ops::linear::matmul(&latent, wk),
                        k_norm,
                        hy.norm_eps,
                    );
                    let k = ops::elemwise::rope_partial_last_yarn(
                        &k,
                        &brope,
                        ix.rope_dim,
                        ix.head_dim,
                        ix.theta,
                        true,
                        false,
                        ix.yarn,
                    );
                    ops::attn::pool_kv_append(
                        &k,
                        &bpos,
                        &breq,
                        inputs.kv(&ix.keys),
                        &inputs.write_page(&ix.keys),
                        &inputs.write_offset(&ix.keys),
                    );
                }
                let pooled = ops::elemwise::rope_partial_last_yarn(
                    &latent,
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
                        &inputs.write_page(&p.entries),
                        &inputs.write_offset(&p.entries),
                    );
                }
            }

            let selection = match at.selection {
                Selection::None => None,
                Selection::Own => {
                    let ix = at
                        .indexer
                        .as_ref()
                        .expect("a layer that ranks its own selection carries an indexer");
                    let s = match &ix.compressor {
                        Some(_) => indexer(
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
                        ),
                        None => rank(&x, &q_a, ix, pos, inputs.kv(&ix.keys), p.ratio),
                    };
                    carry.selection = Some((s.clone(), ix.top_k));
                    Some((s, ix.top_k))
                }
                Selection::Shared => Some(
                    carry
                        .selection
                        .clone()
                        .expect("a CSA2 Reuse Mode layer follows a layer that ranked"),
                ),
            };
            let (po, plse) = match &selection {
                Some((selection, top_k)) => ops::attn::pool_lse_selected(
                    &q,
                    pos,
                    &request_of_token,
                    selection,
                    entries,
                    p.ratio,
                    *top_k,
                    m.heads,
                    m.head_dim,
                    at.sm_scale,
                ),
                None => ops::attn::pool_lse(
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

    let (x, post_mix, comb_mix) = sublayer_input(&streams, &w.mlp_mix, m, carry);
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

/// The compressed KV latent this layer owns, before its rotation. V4 pools a
/// gated, positionally embedded window; V4.1 pools `ratio` positions under a
/// softmax gate (ratio 2) or projects every token (ratio 1); the base model
/// averages. A gated compressor writes its per-token state beside the window
/// KV and gathers it at each group's closing position.
#[allow(clippy::too_many_arguments)]
fn compress(
    m: &Model,
    x: &Value,
    p: &Pool,
    pages: ValueId,
    write_page: &Value,
    write_offset: &Value,
    bpos: &Value,
    breq: &Value,
    chain: bool,
) -> Value {
    match &p.compressor {
        Some(c) => match &c.wgate {
            Some(wgate) => {
                if !chain {
                    let state_kv = ops::linear::matmul(x, &c.wkv);
                    let state_score = ops::linear::matmul(x, wgate);
                    ops::attn::pool_state_write(
                        &state_kv,
                        &state_score,
                        pages,
                        write_page,
                        write_offset,
                        m.head_dim,
                        p.ratio,
                    );
                }
                let pooled = ops::attn::pool_gather(
                    bpos,
                    breq,
                    pages,
                    c.ape.as_ref(),
                    m.head_dim,
                    p.ratio,
                    m.act,
                );
                ops::elemwise::rmsnorm(&pooled, &c.norm, c.norm_eps)
            }
            None => {
                let kv = ops::linear::matmul(x, &c.wkv);
                ops::elemwise::rmsnorm(&kv, &c.norm, c.norm_eps)
            }
        },
        None => ops::attn::pool_gather(bpos, breq, pages, None, m.head_dim, p.ratio, m.act),
    }
}

/// The sub-block's input and the coefficients that fold its output back.
/// Under Single-Pass mHC the input is mixed with the previous sub-block's
/// prediction, and this sub-block's own prediction is kept for the next; the
/// very first sub-block reads stream 0 (the one-hot initial mix).
fn sublayer_input<'m>(
    streams: &Value,
    mix: &'m Mix,
    m: &'m Model,
    carry: &mut Carry<'m>,
) -> (Value, Value, Value) {
    let hy = &m.hyper;
    if !hy.single_pass {
        return gate(streams, mix, hy);
    }
    let normed = ops::elemwise::hc_rmsnorm_f32(streams, hy.norm_eps);
    let mixes = match &mix.dynamic {
        Some(dynamic) => ops::elemwise::hc_project(&normed, dynamic, hy.streams),
        None => normed,
    };
    let (_, post_mix, comb_mix) = ops::elemwise::hc_gates(
        &mixes,
        streams,
        &mix.scale,
        &mix.base,
        hy.streams,
        hy.gate_eps,
        hy.alpha,
        hy.sinkhorn,
    );
    let x = premix(streams, carry.mix.as_ref(), m);
    carry.mix = Some(PrevMix {
        mixes,
        scale: &mix.scale,
        base: &mix.base,
    });
    (x, post_mix, comb_mix)
}

/// `pre · streams` for the previous sub-block's `pre`: the gates kernel over
/// the previous mixes and this residual lands exactly that as its input row
/// (its other two outputs restate the previous sub-block's and are dropped).
/// Before any sub-block ran, the one-hot initial mix reads stream 0.
fn premix(streams: &Value, prev: Option<&PrevMix<'_>>, m: &Model) -> Value {
    let hy = &m.hyper;
    match prev {
        Some(p) => {
            let (x, _, _) = ops::elemwise::hc_gates(
                &p.mixes,
                streams,
                p.scale,
                p.base,
                hy.streams,
                hy.gate_eps,
                hy.alpha,
                hy.sinkhorn,
            );
            x
        }
        None => ops::layout::split_rows(streams, m.hidden).0,
    }
}

/// Engram: the position's n-gram hashes fetch table rows; `wkv` turns them
/// into one key per residual stream and a shared value; every stream adds the
/// value under a gate that is a normalised dot product of the stream against
/// its key (signed square root, then sigmoid).
fn engram(streams: &Value, ids: &Value, inputs: &Input<Facts>, m: &Model, e: &Engram) -> Value {
    let hy = &m.hyper;
    let state = inputs.state(&e.ids_state);
    let map = m.token_map.as_ref();

    let one = Facts::qo_one();
    let (ids_d, ids_p) = ids.split(&one);
    let grams = Value::merge(vec![
        ops::attn::ple_ngram_ids(
            &ids_d,
            state,
            e.pad,
            &e.mults,
            &e.primes,
            &e.offsets,
            e.heads_per_ngram,
            map,
        ),
        ops::attn::ple_ngram_ids_chunked(
            &ids_p,
            state,
            e.pad,
            &e.mults,
            &e.primes,
            &e.offsets,
            e.heads_per_ngram,
            map,
        ),
    ]);
    let rows: u64 = e.primes.iter().sum();
    let fetched = ops::layout::embed_concat(
        &grams,
        &e.table,
        u32::try_from(rows).expect("the Engram table addresses in i32"),
    );
    let kv = ops::linear::matmul(&fetched, &e.wkv);
    let (key, value) = ops::layout::split_rows(&kv, hy.streams * m.hidden);
    let key = ops::elemwise::rmsnorm_grouped_plus_one(&key, &e.k_weight, m.hidden, e.eps);
    let query = ops::elemwise::rmsnorm_grouped_plus_one(streams, &e.q_weight, m.hidden, e.eps);
    let gated = ops::elemwise::ple_gate(&key, &query, &value, hy.streams);
    ops::elemwise::residual_add(&gated, streams)
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
    if hy.single_pass {
        // The next layer's input mix is this layer's own prediction, which is
        // not settled here: the hint would score the wrong row.
        return None;
    }
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

/// V4's indexer: its own gated compressor over the hidden state makes the keys.
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
    let c = ix
        .compressor
        .as_ref()
        .expect("the V4 indexer compresses its own keys");
    let ape = c
        .ape
        .as_ref()
        .expect("the V4 index compressor is positionally embedded");
    let wgate = c.wgate.as_ref().expect("the V4 index compressor is gated");
    let ratio = ape.dim(0);
    let ratio = u32::try_from(ratio).expect("a pooling ratio inside u32");

    if !chain {
        let state_kv = ops::linear::matmul(x, &c.wkv);
        let state_score = ops::linear::matmul(x, wgate);
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
        Some(ape),
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
    ops::attn::index_topk(
        &q,
        Some(&weights),
        keys,
        ix.heads,
        ix.head_dim,
        ix.top_k,
        ratio,
    )
}

/// CSA2's ranking over keys a KV source published (this layer's own or a
/// preceding layer's): a rotated index query per head against every visible
/// key, rectified, combined by the head weights, the top entries kept.
fn rank(
    x: &Value,
    q_a: &Value,
    ix: &Indexer,
    positions: &Value,
    keys: ValueId,
    ratio: u32,
) -> Value {
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
    ops::attn::index_topk(
        &q,
        Some(&weights),
        keys,
        ix.heads,
        ix.head_dim,
        ix.top_k,
        ratio,
    )
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
