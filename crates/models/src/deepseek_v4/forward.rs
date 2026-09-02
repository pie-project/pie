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

    /// A lane whose rows routed to a registered adapter; the zero-row class
    /// lets `engine::fire::walk` skip the correction entirely.
    pub fn has_adapter() -> Predicate {
        Predicate::fact(1)
    }

    /// Lanes that want the draft head run over their rows.
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
            // Cached row width is `kv_down`'s output width (the MLA latent, `at.kv_down.dim(0)`), not `heads * head_dim`.
            c.kv(kv, at.kv.clone(), [at.kv_down.dim(0)]);
            if let Some(p) = &at.pool {
                let pool = c.kv_space(self.kv);
                c.kv(pool, p.entries.clone(), [self.head_dim as u64]);
            }
            // Indexer cache: one `index_head_dim`-wide row per cell, written only at boundary cells since index keys are pooled per block, not per token.
            if let Some(ix) = &at.indexer {
                let index = c.kv_space(self.kv);
                c.kv(index, ix.keys.clone(), [ix.head_dim as u64]);
            }
        }
        // The draft head's kv row: the same latent width as a trunk layer's,
        // in the trunk's page-id space — it attends the same sequence.
        if let Some(mtp) = &self.mtp {
            c.kv(kv, mtp.block.attn.kv.clone(), [mtp.block.attn.kv_down.dim(0)]);
        }
        c
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let m = self;
        let hy = &m.hyper;

        let positions = inputs.positions();
        // Built off the whole inputs, not one class's arm, so no reader falls outside its guard.
        let kv_heads = kv_heads(m);
        let plan_p = ops::attn::plan_prefill(&inputs, m.heads, kv_heads, m.head_dim, Some(m.window));
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
            );
        }

        // **THE TRUNK COLLAPSE.** Flash folds its `M` streams under `M`
        // learned sigmoid gates (`hc_head`: `rmsnorm(streams) · hc_head.fn^T`
        // scaled and based, sigmoid, `+ hc_eps`, weighted sum — no post, no
        // combiner, no Sinkhorn); the toy, which ships no trunk plane, sums
        // them.
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
        // Flash ships a distinct `lm_head`; the toy ties the embedding.
        let logits = match &m.head {
            Some(head) => ops::linear::lm_head(&x, head),
            None => ops::linear::lm_head(&x, &m.embed),
        };

        // **THE DRAFT HEAD**, over the draft window's rows, off the trunk's
        // STREAMS before the collapse (the official `MTPBlock.forward` takes
        // the residual, not the readout): each stream is normed and projected
        // by `h_proj`, the next token's embedding is normed, projected by
        // `e_proj` and added to every stream, one block runs, and the head's
        // own hyper head and norm read out through the base `lm_head`. Row
        // alignment is the runtime's: lane row `r` carries the token one
        // position past the streams the trunk leaves at `r`.
        if let (Some(mtp), Some(head)) = (&m.mtp, &m.head) {
            let (input_mtp, _) = inputs.split(&Facts::drafts());
            let plan_mtp =
                ops::attn::plan_prefill(&input_mtp, m.heads, kv_heads, m.head_dim, Some(m.window));
            let (dstreams, _) = streams.split(&Facts::drafts());
            let (dids, _) = ids.split(&Facts::drafts());
            let (dpos, _) = positions.split(&Facts::drafts());

            let e = ops::layout::embed(&dids, &m.embed, m.vocab);
            let e = ops::elemwise::rmsnorm(&e, &mtp.enorm, mtp.norm_eps);
            let e = ops::elemwise::hc_expand(&ops::linear::matmul(&e, &mtp.e_proj), hy.streams);
            let h = ops::elemwise::rmsnorm_per_head(&dstreams, &mtp.hnorm, m.hidden, mtp.norm_eps);
            let routes = ops::linear::group_routes(&h, hy.streams);
            let h = ops::linear::matmul_grouped(&h, &mtp.h_proj, &routes, hy.streams);
            let fused = ops::elemwise::residual_add(&e, &h);

            let out = layer(
                m,
                &input_mtp,
                &plan_mtp,
                &dpos,
                // Unsplit, as the trunk hands it: the correction inside
                // splits its own operands on the adapter fact, and a routes
                // column already cut on the draft fact would be a second arm.
                &adapter_routes,
                &mtp.block,
                None,
                &fused,
                &dids,
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
            seam::at(seam::MTP, &[&draft]);
        }

        logits
    }
}

/// **ONE FLASH BLOCK OVER THE STREAMS**: the attention sublayer and the MoE
/// sublayer, each gated in and folded back under its own hyper mix. The
/// trunk runs it forty-three times over the fire's rows; the draft head runs
/// it once, over the draft window's rows, against its own cache row.
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
) -> Value {
    let hy = &m.hyper;
    let kv_heads = kv_heads(m);
    let pos = positions;
        let at = &w.attn;
        let pages = inputs.kv(&at.kv);
        let write_page = inputs.write_page(&at.kv);
        let write_offset = inputs.write_offset(&at.kv);
        
        let (x, post_mix, comb_mix) = gate(streams, &w.attn_mix, hy);
        // Flash carries a per-sublayer pre-norm beside the hyper mix.
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
        ops::attn::kv_append_shared(&plane, pages, &write_page, &write_offset);

        let (o, lse) = ops::attn::prefill_lse(
            &q,
            &plan_p,
            pages,
            Some(m.window),
            m.head_dim,
            kv_heads,
            at.sm_scale,
        );

        let (o, lse) = match &at.pool {
            Some(p) => {
                // wkv/wgate are `coff * head_dim` wide (the state's row pitch); pool_state_write scatters them
                // into the source cache's cell so a later boundary can reach back past this fire.
                let ape = p.compressor.as_ref().map(|c| {
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
                    &c.ape
                });
                let entries = inputs.kv(&p.entries);
                let entry_page = inputs.write_page(&p.entries);
                let entry_offset = inputs.write_offset(&p.entries);

                let row_valid = inputs.row_valid();
                let request_of_token = inputs.request_of_token();
                let (bpos, breq, brope) = boundaries(pos, &row_valid, p.ratio);
                let pooled = ops::attn::pool_gather(
                    &bpos, &breq, pages, ape, m.head_dim, p.ratio, m.act,
                );
                // Ropes at the compressed row's position (`brope`, block's first token), not `pos` (block's last token on a boundary row).
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
                ops::attn::pool_kv_append(
                    &pooled,
                    &bpos,
                    &breq,
                    entries,
                    &entry_page,
                    &entry_offset,
                );
                // Scores this layer's compressed rows and selects the top-`index_topk` for the pooled reader; the sliding window is fixed, only the compressed set grows with context.
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
        // **THE VALUE CARRIED THE KEY'S ROPE, AND IT COMES BACK OUT.**
        // MLA's cached latent is both key and value, so the rope lanes
        // of every attended row arrive rotated by that row's position and
        // the output is un-rotated at the query's own (`apply_rotary_emb(
        // o[..., -rd:], freqs_cis, True)`, the official `Attention.forward`).
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

        // The o-projection: on flash `wo_a` is `[o_groups · o_lora, heads · head_dim / o_groups]` and
        // each slice of the head plane projects through its own band (the official `einsum("bsgd,grd->bsgr")`).
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

/// How many experts a route prediction ranks: the tier scores its top 6, 8, 12 and 16 against the router's true six.
const PREDICT_K: u32 = 16;

/// The route prediction for the next layer: `next`'s mlp gate, norm and router applied to `streams` after this
/// layer's attention fold and before its experts land. The streamed tier reads it at this layer's segment cut.
/// `None` where `next` routes by table or by nothing.
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
    Some(ops::linear::moe_predict_route(&logits, bias, *experts, PREDICT_K))
}

/// `ids` is the token-id column; only the flash MoE hash gate reads it (a lookup keyed by token id, not by `x`).
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
            // `noaux_tc` bias layers rank the sqrt-softplus scores plus a correction bias. Hash layers CHOOSE via
            // `ffn.gate.tid2eid`, a `[vocab, top_k]` token-id lookup, but their weights are still the gate's scores
            // at the chosen experts (the official `Gate.forward`), so the router is read on every layer.
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
                    let vocab = u32::try_from(tid2eid.dim(0))
                        .expect("a vocabulary no u32 holds");
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
            // Which routed op runs depends on the bank's declared dtype (bf16 vs. quantized).
            let select = |act: &Value, bank: &Weight| {
                if matches!(bank.dtype, Dtype::Bf16 | Dtype::F16 | Dtype::F32) {
                    ops::linear::moe_matmul_select(act, bank, &routes, *top_k)
                } else {
                    ops::linear::moe_matmul_select_quant(act, bank, &routes, *top_k)
                }
            };
            // Fused bank: one routed matmul over the packed activation. Split pair: two routed matmuls, one per
            // plane, then combined — needed when the two halves use different quantization points.
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

/// kv head count, derived from the cached row width over the head width. The toy caches a full
/// `[heads, head_dim]` plane (answer: `heads`); flash caches the MLA latent, one shared entry per token (answer: 1).
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

/// The indexer's keys come from its own compressor, one pooled entry per `ratio` tokens; the selected reader narrows only the compressed branch.
/// Scoring: `I(t, s) = sum_h w_h * relu(q_h . k_s)`; the query is `wq_b * q_a` off the same normed q-lora the main q-up reads.
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
) -> Value {
    let c = &ix.compressor;
    // Pooling ratio from the indexer's own `ape`, matching the blocks the attention compressor pools.
    let ratio = c.ape.dim(0);
    let ratio = u32::try_from(ratio).expect("a pooling ratio inside u32");

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
    // The key ropes at the compressed row's position (`boundary_rope`, block's first token); the query below ropes at its own per-token position.
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
    ops::attn::pool_kv_append(&k, boundary_pos, boundary_req, keys, write_page, write_offset);

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
        &weights,
        keys,
        ix.heads,
        ix.head_dim,
        ix.top_k,
        ratio,
    )
}

/// `hc_gates` splits a `2M + M^2` row into pre weights, post weights, and the Sinkhorn combiner, from `rmsnorm(streams) * hc_fn^T` (projected by `hc_project` when a `fn` plane exists).
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

/// The three boundary columns, decode and prefill merged: the cell each pooled entry is cached at, its lane, and the compressed row's rope position.
/// Cache cell is `(c + 1) * ratio - 1` (window close); rope position is the block start `c * ratio` — the two differ.
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
