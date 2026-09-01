use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, Predicate, Request, Value, ValueId, Weight,
    ops, seam,
};

use super::model::{Gate, GateUp, HcHead, Hyper, Indexer, Mix, Mlp, Model};

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

        let kv = c.kv_space(self.kv);
        for w in &self.layers {
            let at = &w.attn;
            // **THE CACHED ROW IS THE APPENDED ROW, AND THE APPENDED ROW IS
            // `kv_down`'S OWN OUTPUT.** It was `heads · head_dim`, which is
            // the toy's answer and only the toy's: `base` projects a full
            // per-head `[heads, head_dim]` plane and caches it, so the two
            // readings coincide there. The FLASH rows cache the MLA LATENT —
            // `kv_down` is `[kv_latent, hidden]` and `kv_append_shared` writes
            // 512 elements a token — and declaring `heads · head_dim` reserved
            // a row SIXTY-FOUR TIMES the one the appender writes. Nothing
            // caught it because nothing had fired one; the kernel's own stride
            // check is what finally did.
            c.kv(kv, at.kv.clone(), [at.kv_down.dim(0)]);
            if let Some(p) = &at.pool {
                let pool = c.kv_space(self.kv);
                c.kv(pool, p.entries.clone(), [self.head_dim as u64]);
            }
            // **THE INDEXER'S OWN CACHE, AND ITS OWN COMPRESSOR STATE WITH
            // IT.** One `index_head_dim`-wide row per cell, written at the
            // BOUNDARY cells alone (`pool_kv_append`) because this family's
            // index keys are pooled per block and not projected per token.
            // The space is its own because the rolling compressor state is
            // reserved per POOLED SPACE (`engine_metal::scratch::pool_state`
            // keys on the gather's `pages`), and this layer's attention
            // compressor already holds the kv space's slabs.
            if let Some(ix) = &at.indexer {
                let index = c.kv_space(self.kv);
                c.kv(index, ix.keys.clone(), [ix.head_dim as u64]);
            }
        }
        c
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let m = self;
        let hy = &m.hyper;

        let positions = inputs.positions();
        // ONE SCHEDULE, UNSPLIT — every class here reads prefill, so the carving
        // is built off the whole inputs rather than one class's arm, and no
        // reader falls outside its guard.
        let kv_heads = kv_heads(m);
        let plan_p = ops::attn::plan_prefill(&inputs, m.heads, kv_heads, m.head_dim, Some(m.window));
        let ids = inputs.tokens();
        let mut streams =
            ops::elemwise::hc_expand(&ops::layout::embed(&ids, &m.embed, m.vocab), hy.streams);

        // **THE TRUNK HYPER HEAD**, stated once for the whole tower
        // (`model.hc_head.*`). Its dynamic plane is read against the expanded
        // streams; fusing the trunk mixing into the gate op is the deferred
        // fire, so the read is structural today.
        if let Some(hc) = &m.hc_head {
            trunk_hyper(&streams, hc);
        }

        let adapter_routes = inputs.adapter_routes();
        for (_, w) in inputs.walk_layers(&m.layers) {
            let at = &w.attn;
            let pages = inputs.kv(&at.kv);
            let write_page = inputs.write_page(&at.kv);
            let write_offset = inputs.write_offset(&at.kv);
            let pos = &positions;

            let (x, post_mix, comb_mix) = gate(&streams, &w.attn_mix, hy);
            // Flash carries a per-sublayer pre-norm beside the hyper mix.
            let x = match &w.attn_norm {
                Some(n) => ops::elemwise::rmsnorm(&x, n, hy.norm_eps),
                None => x,
            };

            let q_a = ops::linear::matmul(&x, &at.q_down);
            let q_a = ops::elemwise::rmsnorm(&q_a, &at.q_norm, at.q_norm_eps);
            let q = ops::linear::matmul(&q_a, &at.q_up);
            let q = ops::elemwise::rmsnorm_no_scale(&q, m.head_dim, at.q_norm_eps);

            let q =
                ops::elemwise::rope_partial_last(&q, pos, at.rope_dim, m.head_dim, at.theta, true);
            seam::at(seam::ATTN_Q, &[&q]);

            let plane = ops::linear::matmul(&x, &at.kv_down);
            let plane = ops::elemwise::rmsnorm(&plane, &at.kv_norm, at.kv_norm_eps);
            let plane = ops::elemwise::rope_partial_last(
                &plane,
                pos,
                at.rope_dim,
                m.head_dim,
                at.theta,
                true,
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
                    // **THE LEARNED COMPRESSOR, WRITING THE STATE ITS OWN
                    // POOL READS** (`v4mlx/compressor.py`). `wkv` is the
                    // window's value plane and `wgate` its gate logits, both
                    // `coff · head_dim` wide — the state's row pitch, which is
                    // what those two planes' declared shapes have always been
                    // — and `pool_state_write` scatters them into the source
                    // cache's own cell so the window a later boundary closes
                    // can reach back past this fire. The gather then folds
                    // `ape` into the logits, softmaxes the window, and the
                    // compressor's `norm` closes the entry.
                    //
                    // For as long as no op wrote the state, all four planes
                    // were interned and the gather pooled zeros; the pool was
                    // "parameter-free" only in the sense that its parameters
                    // reached nothing.
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
                    // The compressor's own norm closes the pooled entry, before
                    // the rope at the COMPRESSED row's position — `brope`, the
                    // block's first token, and not `pos`, which on a boundary
                    // row is the block's last.
                    let pooled = match &p.compressor {
                        Some(c) => ops::elemwise::rmsnorm(&pooled, &c.norm, c.norm_eps),
                        None => pooled,
                    };
                    let pooled = ops::elemwise::rope_partial_last(
                        &pooled,
                        &brope,
                        at.rope_dim,
                        m.head_dim,
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
                    // **THE NSA FINE BRANCH, AND IT NARROWS THE COMPRESSED
                    // ONE.** The indexer scores this layer's compressed rows
                    // — its keys are its OWN compressor's pooled entries, one
                    // per ratio-4 block, in 1:1 correspondence with the
                    // attention compressor's — and the top-`index_topk` of
                    // them is what the pooled reader walks. The sliding
                    // window above is fixed at `m.window`; the compressed set
                    // is the only one that grows with the context, which is
                    // why the budget caps THAT and why the ratio-128 layers
                    // carry no indexer at all.
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
            seam::at(seam::ATTN_OUT, &[&o]);

            // The o-projection. On the toy it reads the whole head plane; on
            // flash the plane is reduced over `o_groups` before the low-rank
            // pair (`wo_a`/`wo_b`, out `o_groups · o_lora`).
            let o = group_reduce(&o, at.o_groups, m.hidden);
            let o = ops::linear::matmul(&o, &at.o_down);
            let o = if m.tp > 1 {
                ops::collective::all_reduce(&o)
            } else {
                o
            };
            let o = ops::linear::matmul(&o, &at.o_up);
            // **THE CORRECTION, OVER ITS WINDOW** (design §8, campaign A-6).
            let o = {
                let (adapted, _) = o.split(&Facts::has_adapter());
                let (px, _) = x.split(&Facts::has_adapter());
                ops::linear::lora_correct(&px, &w.lora_a, &w.lora_b, &adapter_routes, &adapted)
            };
            streams = ops::elemwise::hc_fold(&o, &streams, &post_mix, &comb_mix);

            let (x, post_mix, comb_mix) = gate(&streams, &w.mlp_mix, hy);
            let x = match &w.mlp_norm {
                Some(n) => ops::elemwise::rmsnorm(&x, n, hy.norm_eps),
                None => x,
            };
            let f = mlp(&x, &ids, &w.mlp);
            let f = if m.tp > 1 {
                ops::collective::all_reduce(&f)
            } else {
                f
            };
            streams = ops::elemwise::hc_fold(&f, &streams, &post_mix, &comb_mix);
        }

        let (mut y, mut rest) = ops::layout::split_rows(&streams, m.hidden);
        for _ in 1..hy.streams - 1 {
            let (stream, more) = ops::layout::split_rows(&rest, m.hidden);
            y = ops::elemwise::residual_add(&stream, &y);
            rest = more;
        }
        let y = ops::elemwise::residual_add(&rest, &y);
        let x = ops::elemwise::rmsnorm(&y, &m.final_norm, m.final_norm_eps);
        // Flash ships a distinct `lm_head`; the toy ties the embedding.
        match &m.head {
            Some(head) => ops::linear::lm_head(&x, head),
            None => ops::linear::lm_head(&x, &m.embed),
        }
    }
}

/// `ids` is the fire's own token-id column, which only the flash MoE's HASH
/// gate reads: that gate is a lookup keyed by token identity and not a
/// projection of the hidden state, so the row it needs is the id and not `x`.
fn mlp(x: &Value, ids: &Value, mlp: &Mlp) -> Value {
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
            // **TWO GATE KINDS, AND ONLY ONE OF THEM PROJECTS.** The
            // `noaux_tc` bias layers score the router's logits with the
            // correction bias. The first `num_hash_layers` route by
            // `ffn.gate.tid2eid`, a `[vocab, top_k]` I64 table read off the
            // TOKEN IDS — a lookup, not a projection — so those layers
            // compute no logits at all and their `ffn.gate.weight` is a plane
            // this text does not read (`import::Read::Named` says the other
            // half of that sentence).
            let (routes, weights) = match gate {
                Gate::Bias { bias } => ops::linear::moe_topk_sqrt_softplus(
                    &ops::linear::matmul(x, router),
                    bias,
                    *experts,
                    *top_k,
                    *renorm,
                    *scaling,
                ),
                Gate::Hash { tid2eid } => {
                    let vocab = u32::try_from(tid2eid.dim(0))
                        .expect("a vocabulary no u32 holds");
                    ops::linear::moe_hash_route(ids, tid2eid, vocab, *experts, *top_k)
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
            // **A PACKED EXPERT BANK RESOLVES THROUGH ITS PLANES, AND THE
            // MODEL TEXT IS WHAT PICKS THE POINT** — `qwen_3::forward`'s
            // sentence, for this family's reason: dsv4 ships the SAME flash
            // layers as a dense-handle bf16 row and as a split-plane 2-bit
            // one, so which of the two routed ops runs is a fact about the
            // SKU's declared representation and is read off it here.
            let select = |act: &Value, bank: &Weight| {
                if matches!(bank.dtype, Dtype::Bf16 | Dtype::F16 | Dtype::F32) {
                    ops::linear::moe_matmul_select(act, bank, &routes, *top_k)
                } else {
                    ops::linear::moe_matmul_select_quant(act, bank, &routes, *top_k)
                }
            };
            // The fused bank is one routed fire and one packed activation; the
            // split pair is two routed fires over two banks — each read at its
            // own affine point — and the two-value combine. Same arithmetic,
            // and only the second is stateable when the halves disagree.
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

/// **HOW MANY kv HEADS ONE CACHED ENTRY CARRIES**: the appended row over the
/// head width, which is the only place the number is derivable rather than
/// assumed.
///
/// The toy caches a full `[heads, head_dim]` plane and answers `heads`; the
/// flash rows cache the MLA latent, which is ONE shared entry per token that
/// every query head attends, and answer one. Saying `m.heads` here — which is
/// what this text said — is the toy's answer given to both, and it is the same
/// mistake as declaring the cache row `heads · head_dim`: sixty-four kv heads
/// where the appender writes one.
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

/// **THE NSA LIGHTNING INDEXER, FIRED** — the last interned organ of this
/// family, and the two blockers this text used to name are answered rather
/// than deferred.
///
/// # 1. The keys are this indexer's OWN COMPRESSOR'S POOLED ENTRIES
///
/// glm_5's indexer keys one row per TOKEN: a `k_proj` layernormed by a
/// `(weight, bias)` pair, roped, and appended at the token's own cell
/// (`attention.index_layernorm_rope` + `attention.index_kv_append`). This
/// family has no such projection and the guess that it needed one is what
/// stalled the branch. Its keys come from `indexer.compressor.*`, and a
/// compressor emits ONE ENTRY PER `ratio` TOKENS — the reference's
/// `compressor_prefill(..., rotate=True)` returns `[b, s / ratio, head_dim]`
/// (`v4mlx/compressor.py`), and the checkpoint states exactly that shape law
/// at this indexer's width: `indexer.compressor.wkv [2 · 128, hidden]`,
/// `ape [ratio 4, 2 · 128]`, `norm [128]`, which is `entries = coff · D` at
/// `D = index_head_dim = 128, coff = 2` — the same `2 · D` overlap the
/// ATTENTION compressor states at `D = 512` one screen up, and which
/// `pool_gather` already reads as the window's two halves.
///
/// So the key organ is the pool organ, at a narrower width and into its own
/// cache: `pool_state_write` scatters `wkv · x` and `wgate · x` at the index
/// space's own cell, `pool_gather` closes the `2 · ratio` window with the
/// learned gate, the compressor's `norm` and the rope at the compressed row
/// close the entry, and `pool_kv_append` lands it at the boundary cell. There
/// is no per-token key and `attention.index_kv_append` is not this family's
/// writer.
///
/// # 2. The selected reader narrows the COMPRESSED branch, not the latent
///
/// The reference oracle's attention is ONE softmax over
/// `concat(the 128-wide sliding window over the per-token latent, every
/// visible compressed row)` with the per-head sink in the denominator
/// (`oracle/step12_glue.py`'s `widx + cidx` into `sparse_attn`) — which is
/// what `prefill_lse` at `m.window` merged with the pooled reader and closed
/// by `attention.sink` already computes above. The window is FIXED; the only
/// key set that grows with the context is the compressed one
/// (`nvis = (pos + 1) / ratio`), so the trained `index_topk` is what caps
/// that, and the ratio-128 layers carry no indexer because `S / 128` needs no
/// capping. The kernel wanted was therefore the selected twin of
/// `attention.pool_lse`, not of the shared-latent reader; the absorbed
/// `mla_*_selected` pair remains a different attention and is still not this
/// family's.
///
/// # The scoring
///
/// `I(t, s) = Σ_h w_h · relu(q_h · k_s)` over `heads = 64` indexer heads of
/// `head_dim = 128` — `attention.index_topk`'s own statement. The query is
/// `wq_b · q_a` off the SAME normed q-lora the main q-up reads, roped on the
/// last `rope_dim` lanes of each head at the compressor's `theta` because
/// that is where the KEY's rope lives; the per-head weights are
/// `weights_proj · x`, one per head. No indexer q-norm plane exists in the
/// checkpoint and none is invented here.
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
    // The ratio the indexer's own `ape` states — one pooled key per block,
    // and the same blocks the attention compressor pools.
    let ratio = c.ape.dim(0);
    let ratio = u32::try_from(ratio).expect("a pooling ratio inside u32");

    // The keys: this indexer's compressor, into this indexer's cache.
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
    // The KEY is a compressed row and ropes at the compressed row's position
    // (`boundary_rope`, the block's first token) — the same `arange(0, cutoff,
    // ratio)` the attention compressor ropes at, because this is the same
    // organ. The QUERY below is a per-token row and ropes at its own position.
    let k = ops::elemwise::rope_partial_last(
        &k,
        boundary_rope,
        ix.rope_dim,
        ix.head_dim,
        ix.theta,
        true,
    );
    ops::attn::pool_kv_append(&k, boundary_pos, boundary_req, keys, write_page, write_offset);

    // The query and the per-head combine weights.
    let q = ops::linear::matmul(q_a, &ix.wq_b);
    let q = ops::elemwise::rope_partial_last(
        &q,
        positions,
        ix.rope_dim,
        ix.head_dim,
        ix.theta,
        true,
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

/// The trunk hyper head's planes (`hc_head.{base,fn,scale}`), stated and read
/// as checkpoint params. Fusing the trunk mixing into the gate op is the
/// deferred fire.
fn trunk_hyper(streams: &Value, hc: &HcHead) {
    let r = streams.rec();
    let _ = r.weight(&hc.base);
    let _ = r.weight(&hc.dynamic);
    let _ = r.weight(&hc.scale);
}

/// **THE MIX ROW IS PROJECTED WHERE A PLANE SAYS IT IS.**
///
/// `hc_gates` splits a `2M + M²` row into the pre weights, the post weights
/// and the Sinkhorn combiner, and it has always read its operand at that
/// stride. What produces the row is `rmsnorm(streams) · hc_fn^T`
/// (`v4mlx/hc.py`), and the flash rows ship that plane as `{attn,ffn}_hc.fn`.
/// While no op could fire it, this text interned the plane and handed the gate
/// the NORMED ROW itself — whose leading `2M + M²` floats are the first
/// columns of a `M·hidden`-wide rectangle and are nobody's mixing function.
/// `elementwise.hc_project` is the missing GEMM.
///
/// The TOY rows carry no `fn` plane (`Mix::dynamic` is `None`) — their mixing
/// is the static `scale`/`base` pair alone — so they keep handing the normed
/// row over, which is the reading they were traced under.
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

/// Reduce a head plane over `groups` even blocks to `width` before the
/// o-projection. `groups == 1` is the identity (the toy).
fn group_reduce(o: &Value, groups: u32, width: u32) -> Value {
    if groups <= 1 {
        return o.clone();
    }
    let (mut acc, mut rest) = ops::layout::split_rows(o, width);
    for _ in 1..groups - 1 {
        let (block, more) = ops::layout::split_rows(&rest, width);
        acc = ops::elemwise::residual_add(&block, &acc);
        rest = more;
    }
    ops::elemwise::residual_add(&rest, &acc)
}

/// The three boundary columns, decode and prefill merged: the CELL each
/// pooled entry is cached at, its lane, and the COMPRESSED ROW'S POSITION it
/// is roped at.
///
/// **THE THIRD IS NOT THE FIRST.** The entry lands at the cell its window
/// closes on — `(c + 1) · ratio - 1`, which is what `pool_lse` reads back —
/// but `compressor_prefill` ropes the pooled plane at `rows = arange(0,
/// cutoff, ratio)`, the block STARTS `c · ratio` (`v4mlx/compressor.py`).
/// This text roped at the raw token positions, which on a boundary row ARE
/// the closing cell, so every compressed key it cached — the attention
/// branch's and the indexer's alike — carried an angle `ratio - 1` positions
/// too far.
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
