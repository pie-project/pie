use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, Predicate, Request, Value, ops, seam,
};

use super::model::{Hyper, Indexer, Kda, Mix, Mixer, Mla, Mlp, Model, Tower};
use model_dsl::MropeForm;

/// How a DSA mixer's rows are read: the trunk splits them on `qo_one` into
/// a decode arm and a prefill arm; the draft head reads every row through
/// the prefill (chunked) forms, since its rows already stand in one split
/// arm (`drafts`) and a value may not sit in two.
#[derive(Clone, Copy)]
enum Arms {
    Split,
    Whole,
}

pub struct Facts {
    pub qo_one: bool,
    pub has_adapter: bool,
    pub drafts: bool,
    /// Lanes whose fire carries an image span.
    pub media: bool,
}

impl Facts {
    pub fn qo_one() -> Predicate {
        Predicate::fact(0)
    }

    /// Rows whose request routed to a registered adapter; a fire with none
    /// costs nothing.
    pub fn has_adapter() -> Predicate {
        Predicate::fact(1)
    }

    /// Lanes that want the draft head run over their rows.
    pub fn drafts() -> Predicate {
        Predicate::fact(2)
    }

    pub fn media() -> Predicate {
        Predicate::fact(3)
    }
}

impl Classify for Facts {
    fn of(r: &Request) -> Facts {
        Facts {
            qo_one: r.query_len() == 1,
            has_adapter: r.has_adapter(),
            drafts: r.drafts(),
            media: r.has_media(),
        }
    }

    fn word(&self) -> u64 {
        u64::from(self.qo_one)
            | (u64::from(self.has_adapter) << 1)
            | (u64::from(self.drafts) << 2)
            | (u64::from(self.media) << 3)
    }
}

impl ForwardHybrid for Model {
    type Facts = Facts;

    fn caches(&self) -> HybridSpec {
        let mut c = HybridSpec::new();
        let kv = c.kv_space(self.kv);
        for w in &self.layers {
            match &w.mixer {
                Mixer::Mla(a) => {
                    // One space per layer: the compressor state is keyed by it.
                    let index = c.kv_space(self.kv);
                    c.kv(
                        kv,
                        a.kv.clone(),
                        [self.kv_lora_rank as u64, a.qk_rope_head_dim as u64],
                    );
                    c.kv(index, a.indexer.keys.clone(), [a.indexer.head_dim as u64]);
                }
                Mixer::Kda(k) => {
                    let width = (k.heads * k.head_dim) as u64;
                    c.state(
                        k.conv_state.clone(),
                        [k.conv_kernel as u64, 3 * width],
                        Dtype::Bf16,
                    );
                    // The KDA recurrence keeps its state in f32 (`ssm_kda_*`).
                    c.state(
                        k.delta_state.clone(),
                        [k.heads as u64, k.head_dim as u64, k.head_dim as u64],
                        Dtype::F32,
                    );
                }
            }
        }
        // The draft head's rows: its own latent row in the trunk's page-id
        // space (it attends the same sequence) and its own indexer keys.
        if let Some(mtp) = &self.mtp {
            let index = c.kv_space(self.kv);
            c.kv(
                kv,
                mtp.attn.kv.clone(),
                [self.kv_lora_rank as u64, mtp.attn.qk_rope_head_dim as u64],
            );
            c.kv(index, mtp.attn.indexer.keys.clone(), [mtp.attn.indexer.head_dim as u64]);
        }
        c
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let m = self;
        let hy = &m.hyper;

        // plan[0] is the decode schedule, plan[1] the prefill one; the mixer
        // splits q by the same predicate, so each reader finds its own plan.
        let (input_d, input_p) = inputs.split(&Facts::qo_one());
        let plan = [
            ops::attn::mla_plan(&input_d, m.heads, m.kv_lora_rank),
            ops::attn::mla_plan(&input_p, m.heads, m.kv_lora_rank),
        ];
        let positions = inputs.positions();
        // Tower nodes must all be emitted before any trunk node, or
        // `model_compiler` refuses the plan (`Error::UnitsInterleave`).
        let towered = m.tower.as_ref().map(|t| tower(&inputs, t));
        let ids = inputs.tokens();
        let mut narrow = ops::layout::embed(&ids, &m.embed, m.vocab);
        // Tower rows written over the token rows the image placeholders
        // occupy, BEFORE the streams fan out. `scatter_live_rows`, not a
        // plain scatter: `merge_rows` compacts and the tail routes carry a
        // `-1` sentinel.
        if let Some(t) = &towered {
            let (imaged, _) = narrow.split(&Facts::media());
            narrow = ops::layout::scatter_live_rows(t, &inputs.patch_routes(), &imaged).everywhere();
        }
        let mut streams = ops::elemwise::hc_expand(&narrow, hy.streams);

        let routes = inputs.adapter_routes();
        for (_, w) in inputs.walk_layers(&m.layers) {
            let (x, post_mix, comb_mix) = gate(&streams, &w.attn_mix, hy);
            let x = ops::elemwise::rmsnorm(&x, &w.mixer_norm, w.mixer_norm_eps);
            let o = match &w.mixer {
                Mixer::Mla(a) => mla_mixer(&x, &inputs, &plan, &positions, m, a, Arms::Split),
                Mixer::Kda(k) => kda_mixer(&x, &inputs, k),
            };
            let o = if m.tp > 1 {
                ops::collective::all_reduce(&o)
            } else {
                o
            };
            // Must run after all_reduce: a rows-cut partial product would sum
            // the correction tp times.
            let o = {
                let (adapted, _) = o.split(&Facts::has_adapter());
                let (px, _) = x.split(&Facts::has_adapter());
                ops::linear::lora_correct(&px, &w.lora_a, &w.lora_b, &routes, &adapted)
            };
            streams = ops::elemwise::hc_fold(&o, &streams, &post_mix, &comb_mix);

            let (x, post_mix, comb_mix) = gate(&streams, &w.mlp_mix, hy);
            let x = ops::elemwise::rmsnorm(&x, &w.mlp_norm, w.mlp_norm_eps);
            let f = mlp(&x, &w.mlp);
            let f = if m.tp > 1 {
                ops::collective::all_reduce(&f)
            } else {
                f
            };
            streams = ops::elemwise::hc_fold(&f, &streams, &post_mix, &comb_mix);
        }

        // No trunk hyper head ships with this checkpoint, so the streams sum.
        let (mut y, mut rest) = ops::layout::split_rows(&streams, m.hidden);
        for _ in 1..hy.streams - 1 {
            let (stream, more) = ops::layout::split_rows(&rest, m.hidden);
            y = ops::elemwise::residual_add(&stream, &y);
            rest = more;
        }
        let y = ops::elemwise::residual_add(&rest, &y);

        let x = ops::elemwise::rmsnorm(&y, &m.final_norm, m.final_norm_eps);
        let logits = ops::linear::lm_head(&x, &m.head);

        // **THE DRAFT HEAD**, over the draft window's rows, off the trunk's
        // collapsed residual `y` (the official `MTP` block takes the last
        // layer's output, not the readout). **The head's token is the
        // trunk's argmax**: the module is trained on `(y_i, t_{i+1}) →
        // t_{i+2}`, and inside a verify fire the only `t_{i+1}` a row can
        // name is the token the trunk just chose there — the same argmax the
        // verifier reads, so the chain at the row it accepts continues its
        // next window. Row alignment is the runtime's: lane row `r` carries
        // the token one position past the residual the trunk leaves at `r`.
        if let Some(mtp) = &m.mtp {
            let (input_mtp, _) = inputs.split(&Facts::drafts());
            // One prefill plan over every draft row (`Arms::Whole`).
            let plan_one = ops::attn::mla_plan(&input_mtp, m.heads, m.kv_lora_rank);
            let plan_mtp = [plan_one.clone(), plan_one];
            let (dy, _) = y.split(&Facts::drafts());
            let (dpos, _) = positions.split(&Facts::drafts());
            let (dlogits, _) = logits.split(&Facts::drafts());
            let mut token = ops::layout::argmax(&[&dlogits]);
            let mut hidden = dy;
            let mut chain: Vec<Value> = Vec::with_capacity(mtp.depth as usize);
            for step in 0..mtp.depth {
                let e = ops::layout::embed(&token, &m.embed, m.vocab);
                let e = ops::elemwise::rmsnorm(&e, &mtp.enorm, mtp.norm_eps);
                let h = ops::elemwise::rmsnorm(&hidden, &mtp.hnorm, mtp.norm_eps);
                // `eh_proj([e; h])` as its two column halves.
                let fused = ops::elemwise::residual_add(
                    &ops::linear::matmul(&e, &mtp.e_proj),
                    &ops::linear::matmul(&h, &mtp.h_proj),
                );
                let x = ops::elemwise::rmsnorm(&fused, &mtp.mixer_norm, mtp.mixer_norm_eps);
                let o = mla_mixer(&x, &input_mtp, &plan_mtp, &dpos, m, &mtp.attn, Arms::Whole);
                let o = if m.tp > 1 {
                    ops::collective::all_reduce(&o)
                } else {
                    o
                };
                let r = ops::elemwise::residual_add(&o, &fused);
                let x = ops::elemwise::rmsnorm(&r, &mtp.mlp_norm, mtp.mlp_norm_eps);
                let f = mlp(&x, &mtp.mlp);
                let f = if m.tp > 1 {
                    ops::collective::all_reduce(&f)
                } else {
                    f
                };
                let r = ops::elemwise::residual_add(&f, &r);
                let read = ops::elemwise::rmsnorm(&r, &mtp.norm, mtp.norm_eps);
                let draft = ops::linear::lm_head(&read, &m.head);
                if step == 0 {
                    seam::at(seam::MTP, &[&draft]);
                }
                token = ops::layout::argmax(&[&draft]);
                hidden = r;
                chain.push(draft);
            }
            // The token plane: every step's argmax side by side, `[rows,
            // depth]`, what `mtp_drafts` reads.
            let steps: Vec<&Value> = chain.iter().collect();
            seam::at(seam::MTP_DRAFTS, &[&ops::layout::argmax(&steps)]);
        }
        logits
    }
}

fn mlp(x: &Value, mlp: &Mlp) -> Value {
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
            shared,
            experts,
            top_k,
            inter,
            limit,
            renorm,
            scaling,
        } => {
            let (routes, weights) = ops::linear::moe_topk_sigmoid_biased(
                &ops::linear::matmul(x, router),
                bias,
                *experts,
                *top_k,
                *renorm,
                *scaling,
            );
            let packed = ops::linear::moe_matmul_select_quant(x, gate_up, &routes, *top_k);
            let act = ops::linear::mlp_swiglu_clamp(&packed, *inter, *limit);
            let routed = ops::linear::moe_weighted_sum(
                &ops::linear::moe_matmul_select_quant(&act, down, &routes, *top_k),
                &weights,
            );
            match shared {
                None => routed,
                Some(s) => {
                    let act = ops::linear::mlp_swiglu_clamp(
                        &ops::linear::matmul(x, &s.gate_up),
                        s.inter,
                        *limit,
                    );
                    ops::elemwise::residual_add(&ops::linear::matmul(&act, &s.down), &routed)
                }
            }
        }
    }
}

/// `mla_use_nope`: nothing here ropes — `qk_rope_head_dim` is zero, so the
/// latents, the query and the indexer all read their planes unrotated.
fn mla_mixer(
    x: &Value,
    inputs: &Input<Facts>,
    plan: &[Value; 2],
    positions: &Value,
    m: &Model,
    a: &Mla,
    arms: Arms,
) -> Value {
    let pages = inputs.kv(&a.kv);
    let write_page = inputs.write_page(&a.kv);
    let write_offset = inputs.write_offset(&a.kv);

    let q_a = ops::linear::matmul(x, &a.q_a_proj);
    let q_a = ops::elemwise::rmsnorm(&q_a, &a.q_a_norm, a.q_a_norm_eps);
    let q_b = ops::linear::matmul(&q_a, &a.q_b_proj);
    let kv_a = ops::linear::matmul(x, &a.kv_a_proj);
    seam::at(seam::ATTN_QV, &[&q_b, &kv_a]);

    let selection = index_select(x, &q_a, inputs, positions, m.act, &a.indexer, arms);

    let (kv_c, k_pe) = ops::attn::mla_latents(&kv_a, &a.kv_a_norm, a.kv_a_norm_eps, m.kv_lora_rank);
    ops::attn::mla_kv_append(&kv_c, &k_pe, pages, &write_page, &write_offset);

    let (q_nope, q_pe) =
        ops::attn::mla_split_q_b(&q_b, m.heads, a.qk_nope_head_dim, a.qk_rope_head_dim);
    let q = ops::attn::mla_absorb_q(
        &q_nope,
        &a.kv_b_proj,
        m.heads,
        m.kv_lora_rank,
        a.qk_nope_head_dim,
        a.v_head_dim,
    );
    seam::at(seam::ATTN_Q, &[&q]);

    let scored = match arms {
        Arms::Split => {
            let one = Facts::qo_one();
            let (dq, pq) = q.split(&one);
            let (dpe, ppe) = q_pe.split(&one);
            let (d_sel, p_sel) = selection.split(&one);
            Value::merge(vec![
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
            ])
        }
        Arms::Whole => ops::attn::mla_prefill_selected(
            &q,
            &plan[1],
            &q_pe,
            &selection,
            pages,
            m.heads,
            m.kv_lora_rank,
            a.sm_scale,
        ),
    };

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

/// `index_kpool_compress`: keys are the gated pool of every `kpool` tokens,
/// so the ranked set is compressed rows, not tokens.
fn index_select(
    x: &Value,
    q_a: &Value,
    inputs: &Input<Facts>,
    positions: &Value,
    act: Dtype,
    ix: &Indexer,
    arms: Arms,
) -> Value {
    let keys = inputs.kv(&ix.keys);
    let write_page = inputs.write_page(&ix.keys);
    let write_offset = inputs.write_offset(&ix.keys);
    let row_valid = inputs.row_valid();

    // GLM normalizes each key before pooling, so the norm runs per token here
    // and the pooled entry is appended as gathered.
    let state_kv = ops::attn::index_layernorm_rope(
        &ops::linear::matmul(x, &ix.wk),
        positions,
        &ix.k_norm,
        ix.k_norm_eps,
        &ix.k_norm_bias,
        ix.rope_dim,
        ix.theta,
    );
    let state_score = ops::linear::matmul(x, &ix.kpool_gate);
    ops::attn::pool_state_write(
        &state_kv,
        &state_score,
        keys,
        &write_page,
        &write_offset,
        ix.head_dim,
        ix.kpool,
    );

    let (bpos, breq, _brope) = boundaries(positions, &row_valid, ix.kpool, arms);
    let k = ops::attn::pool_gather(
        &bpos,
        &breq,
        keys,
        Some(&ix.kpool_ape),
        ix.head_dim,
        ix.kpool,
        act,
    );
    ops::attn::pool_kv_append(&k, &bpos, &breq, keys, &write_page, &write_offset);

    let q = ops::attn::index_rope(
        &ops::linear::matmul(q_a, &ix.wq_b),
        positions,
        ix.heads,
        ix.head_dim,
        ix.rope_dim,
        ix.theta,
    );
    let weights = ops::linear::matmul(x, &ix.weights_proj);
    ops::attn::index_topk(
        &q,
        &weights,
        keys,
        ix.heads,
        ix.head_dim,
        ix.top_k,
        ix.kpool,
    )
}

/// The three boundary columns, decode and prefill merged: the cell each pooled
/// entry is cached at, its lane, and the compressed row's position.
fn boundaries(
    positions: &Value,
    row_valid: &Value,
    ratio: u32,
    arms: Arms,
) -> (Value, Value, Value) {
    if let Arms::Whole = arms {
        return ops::attn::pool_boundary_prefill(positions, row_valid, ratio);
    }
    let (one, many) = positions.split(&Facts::qo_one());
    let (dpos, dreq, drope) = ops::attn::pool_boundary_decode(&one, row_valid, ratio);
    let (ppos, preq, prope) = ops::attn::pool_boundary_prefill(&many, row_valid, ratio);
    (
        Value::merge(vec![dpos, ppos]),
        Value::merge(vec![dreq, preq]),
        Value::merge(vec![drope, prope]),
    )
}

fn kda_mixer(x: &Value, inputs: &Input<Facts>, k: &Kda) -> Value {
    let conv = inputs.state(&k.conv_state);
    let delta = inputs.state(&k.delta_state);
    let qkv = ops::linear::matmul(x, &k.qkv);
    let f = ops::linear::matmul(&ops::linear::matmul(x, &k.f_a), &k.f_b);
    let b = ops::linear::matmul(x, &k.b);
    seam::at(seam::RECURRENT, &[&qkv]);

    let one = Facts::qo_one();
    let (qkv_d, qkv_p) = qkv.split(&one);
    let (f_d, f_p) = f.split(&one);
    let (b_d, b_p) = b.split(&one);
    let core = Value::merge(vec![
        {
            let mixed = ops::attn::ssm_causal_conv1d(&qkv_d, &k.conv, conv, k.conv_kernel);
            ops::attn::ssm_kda_step(
                &mixed, &f_d, &b_d, &k.dt_bias, &k.a_log, delta, k.heads, k.head_dim, k.norm_eps,
                k.gate_floor,
            )
        },
        {
            // Chunked: the prefill's conv reads its own rows, not the state.
            let mixed = ops::attn::ssm_causal_conv1d_chunked(&qkv_p, &k.conv, conv, k.conv_kernel);
            ops::attn::ssm_kda_chunked(
                &mixed, &f_p, &b_p, &k.dt_bias, &k.a_log, delta, k.heads, k.head_dim, k.norm_eps,
                k.gate_floor,
            )
        },
    ]);

    // The output gate is low-rank here (`g_a_proj` then `g_b_proj`), not one plane.
    let g = ops::linear::matmul(&ops::linear::matmul(x, &k.g_a), &k.g_b);
    let o = ops::elemwise::rmsnorm_gated_by(&core, &g, &k.o_norm, k.heads, k.o_norm_eps);
    ops::linear::matmul(&o, &k.o_proj)
}

/// The vision tower over every patch row of the fire, answering one merged
/// row per `merge²` patches (`rows / merge²` live rows; the caller scatters
/// them with `scatter_live_rows` and a `-1`-sentinel route vector).
fn tower(inputs: &Input<Facts>, t: &Tower) -> Value {
    let d = t.head_dim;
    // Pre-unfolded patch vectors, the per-image indptr the bidirectional
    // attention is block-diagonal over, and each patch's (t, h, w).
    let x = inputs.patches(t.patch_width);
    let segments = inputs.patch_segments();
    let grid = inputs.patch_positions();
    let mut y = ops::elemwise::add_bias(&t.patch_embed_bias, &ops::linear::matmul(&x, &t.patch_embed));
    for b in &t.blocks {
        let n = ops::elemwise::rmsnorm(&y, &b.norm1, t.norm_eps);
        let (q, k, v) = ops::layout::split_qkv(
            &ops::elemwise::add_bias(&b.qkv_bias, &ops::linear::matmul(&n, &b.qkv)),
            t.hidden,
            t.hidden,
        );
        let q = ops::elemwise::rmsnorm_per_head(&q, &b.q_norm, d, t.norm_eps);
        let k = ops::elemwise::rmsnorm_per_head(&k, &b.k_norm, d, t.norm_eps);
        // Two axes and no time: `rotary_pos_emb(head_dim / 2)` over (h, w),
        // the (h, w) frequencies side by side and the pair repeated across
        // the head, rotated by `rotate_half` — qwen's tower form.
        let (q, k) = ops::elemwise::rope_mrope(
            &q,
            &k,
            &grid,
            [0, d / 4, d / 4],
            MropeForm::Blocked,
            d,
            d,
            t.theta,
        );
        let o = ops::attn::dense(&q, &k, &v, &segments, d, t.sm_scale);
        y = ops::elemwise::residual_add(
            &ops::elemwise::add_bias(&b.proj_bias, &ops::linear::matmul(&o, &b.proj)),
            &y,
        );
        let n = ops::elemwise::rmsnorm(&y, &b.norm2, t.norm_eps);
        let h = ops::elemwise::add_bias(&b.gate_up_bias, &ops::linear::matmul(&n, &b.gate_up));
        let a = ops::linear::mlp_swiglu_clamp(&h, t.inter, t.limit);
        y = ops::elemwise::residual_add(
            &ops::elemwise::add_bias(&b.down_bias, &ops::linear::matmul(&a, &b.down)),
            &y,
        );
    }
    let y = ops::elemwise::rmsnorm(&y, &t.post_norm, t.norm_eps);
    // The 2×2 downsample conv: one merge block's rows side by side, times
    // the kernel laid out in the same `(kh, kw, c)` order.
    let folded = ops::layout::merge_rows(&y, t.merge);
    let y = ops::elemwise::add_bias(&t.downsample_bias, &ops::linear::matmul(&folded, &t.downsample));
    let m = &t.merger;
    let p = ops::linear::matmul(&y, &m.proj);
    let n = ops::elemwise::layernorm(&p, &m.norm, &m.norm_bias, t.norm_eps);
    let g = ops::linear::mlp_gelu_tanh(&n);
    let h = ops::linear::matmul(&g, &m.gate_up);
    let a = ops::linear::mlp_swiglu_clamp(&h, t.merger_inter, t.limit);
    ops::linear::matmul(&a, &m.down)
}

/// `hc_gates` splits a `2M + M²` row into pre weights, post weights and the
/// Sinkhorn combiner, from `rmsnorm(streams)` projected through `hc.fn`.
fn gate(streams: &Value, mix: &Mix, hy: &Hyper) -> (Value, Value, Value) {
    let normed = ops::elemwise::hc_rmsnorm_f32(streams, hy.norm_eps);
    let mixes = ops::elemwise::hc_project(&normed, &mix.dynamic, hy.streams);
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
