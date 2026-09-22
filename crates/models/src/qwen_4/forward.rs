use model_dsl::{
    Classify, Dtype, ForwardHybrid, GateActivation, HybridSpec, Input, MropeForm, Predicate,
    Request, Value, ops, seam,
};

use super::model::{Attn, Gdn, Indexer, Mixer, Mlp, Model, Ple, Residual, Tower};

const MROPE_SECTIONS: [u32; 3] = [11, 11, 10];

pub struct Facts {
    pub qo_one: bool,
    pub captures_scores: bool,
    pub masked: bool,
    pub drafts: bool,
    pub media: bool,
}

impl Facts {
    #[must_use]
    pub fn qo_one() -> Predicate {
        Predicate::fact(0)
    }

    #[must_use]
    pub fn captures_scores() -> Predicate {
        Predicate::fact(1)
    }

    #[must_use]
    pub fn masked() -> Predicate {
        Predicate::fact(2)
    }

    #[must_use]
    pub fn drafts() -> Predicate {
        Predicate::fact(3)
    }

    #[must_use]
    pub fn media() -> Predicate {
        Predicate::fact(4)
    }
}

impl Classify for Facts {
    fn of(r: &Request) -> Facts {
        Facts {
            qo_one: r.query_len() == 1,
            captures_scores: r.captures_scores(),
            masked: r.has_custom_mask(),
            drafts: r.drafts(),
            media: r.has_media(),
        }
    }

    fn word(&self) -> u64 {
        u64::from(self.qo_one)
            | (u64::from(self.captures_scores) << 1)
            | (u64::from(self.masked) << 2)
            | (u64::from(self.drafts) << 3)
            | (u64::from(self.media) << 4)
    }
}

impl ForwardHybrid for Model {
    type Facts = Facts;

    fn caches(&self) -> HybridSpec {
        let mut c = HybridSpec::new();
        let kv = c.kv_space(self.kv);
        let plane = u64::from(self.kv_heads) * u64::from(self.head_dim);
        for w in &self.layers {
            match &w.mixer {
                Mixer::Attn { attn, indexer } => {
                    c.kv(kv, attn.kv.clone(), [plane, plane]);
                    if let Some(ix) = indexer {
                        // One row holds both the raw indexer keys and, at every block's
                        // closing cell, the mean that stands for the block: `pool_kv_append`
                        // writes `boundary_pos` = cell (b+1)*ratio-1 and `index_topk` reads
                        // block j at (j+1)*ratio-1. Block b+1 means over [(b+1)r, (b+2)r-1],
                        // which never includes (b+1)r-1, and within a fire the appends run
                        // before the mean, which runs before the write-back.
                        let index = c.kv_space(self.kv);
                        c.kv(index, ix.keys.clone(), [u64::from(ix.head_dim)]);
                    }
                }
                Mixer::Gdn(g) => {
                    let conv_ch = u64::from(Gdn::qkv_width(g.k_heads, g.v_heads, g.k_dim, g.v_dim));
                    c.state(
                        g.conv_state.clone(),
                        [u64::from(g.conv_kernel), conv_ch],
                        Dtype::Bf16,
                    );
                    c.state(
                        g.delta_state.clone(),
                        [u64::from(g.v_heads), u64::from(g.k_dim), u64::from(g.v_dim)],
                        Dtype::Bf16,
                    );
                }
            }
        }
        if self.mtp.is_some() {
            c.kv(kv, "kv.mtp".to_string(), [plane, plane]);
        }
        if let Some(p) = &self.ple {
            let wide = u64::from(self.streams) * u64::from(self.hidden);
            c.state(p.ids_state.clone(), [u64::from(p.dilation) - 1], Dtype::I32);
            c.state(
                p.conv_state.clone(),
                [u64::from((p.conv_kernel - 1) * p.dilation) + 1, wide],
                Dtype::Bf16,
            );
        }
        c
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let m = self;

        let classes = [
            Facts::masked(),
            Facts::captures_scores(),
            Facts::qo_one(),
            Predicate::rest(),
        ];
        let [input_m, input_s, input_d, input_p] = inputs.split(classes);
        let plan_m = ops::attn::plan_prefill(&input_m, m.q_heads, m.kv_heads, m.head_dim, None);
        let plan_d = ops::attn::plan_decode(&input_d, m.q_heads, m.kv_heads, m.head_dim, None);
        let plan_p = ops::attn::plan_prefill(&input_p, m.q_heads, m.kv_heads, m.head_dim, None);
        let plan_s = ops::attn::plan_prefill(&input_s, m.q_heads, m.kv_heads, m.head_dim, None);
        let mask = inputs.mask();

        let towered = m.tower.as_ref().map(|t| tower(&inputs, t));

        let ids = inputs.tokens();
        let mut narrow = ops::layout::embed(&ids, &m.embed, m.vocab);
        if let Some(t) = &towered {
            let (imaged, _) = narrow.split(&Facts::media());
            narrow =
                ops::layout::scatter_live_rows(t, &inputs.patch_routes(), &imaged).everywhere();
        }
        let mut y = ops::elemwise::hc_expand(&narrow, m.streams);

        for (l, w) in m.layers.iter().enumerate() {
            if let Some(p) = m.ple.as_ref().filter(|p| p.layer as usize == l) {
                let enriched = ple(&y, &ids, &inputs, m, p);
                y = ops::elemwise::residual_add(&enriched, &y);
            }

            let (x, normed) = mix_in(&y, &w.attn_res, m);
            let o = match &w.mixer {
                Mixer::Attn { attn, indexer } => attn_mixer(
                    &x,
                    &inputs,
                    m,
                    &plan_m,
                    &plan_d,
                    &plan_p,
                    &plan_s,
                    &mask,
                    attn,
                    indexer.as_ref(),
                ),
                Mixer::Gdn(g) => gdn_mixer(&x, &inputs, g),
            };
            y = inject(&o, &normed, &w.attn_res, m, &y);

            let (x, normed) = mix_in(&y, &w.mlp_res, m);
            let f = moe(&x, &w.mlp);
            y = inject(&f, &normed, &w.mlp_res, m, &y);
        }

        let (x, _) = mix_in(&y, &m.mixer, m);
        let x = ops::layout::gather_rows(&x, &inputs.readout_rows());
        let logits = ops::linear::lm_head(&x, &m.head);

        if let Some(mtp) = &m.mtp {
            let (input_mtp, _) = inputs.split(&Facts::drafts());
            let plan_mtp =
                ops::attn::plan_prefill(&input_mtp, m.q_heads, m.kv_heads, m.head_dim, None);
            let (dy, _) = y.split(&Facts::drafts());
            let (dpos, _) = rotation_positions(&inputs, m).split(&Facts::drafts());
            let (dlogits, _) = logits.split(&Facts::drafts());

            let w = &mtp.block;
            let Mixer::Attn { attn: a, .. } = &w.mixer else {
                unreachable!("the draft head's block is full attention");
            };
            let mut token = ops::layout::argmax(&[&dlogits]);
            let mut hidden = dy;
            let mut chain: Vec<Value> = Vec::with_capacity(mtp.depth as usize);
            for step in 0..mtp.depth {
                let e = ops::layout::embed(&token, &m.embed, m.vocab);
                let e = ops::elemwise::rmsnorm_plus_one(&e, &mtp.norm_embed, mtp.eps);
                let e =
                    ops::elemwise::hc_expand(&ops::linear::matmul(&e, &mtp.fc_embed), m.streams);
                let h = ops::elemwise::rmsnorm_grouped_plus_one(
                    &hidden,
                    &mtp.norm_hidden,
                    m.hidden,
                    mtp.eps,
                );
                let routes = ops::linear::group_routes(&h, m.streams);
                let h = ops::linear::matmul_grouped(&h, &mtp.fc_hidden, &routes, m.streams);
                let mut r = ops::elemwise::residual_add(&e, &h);

                let (x, normed) = mix_in(&r, &w.attn_res, m);
                let o = mtp_attn(&x, &input_mtp, m, &plan_mtp, &dpos, a, step > 0);
                r = inject(&o, &normed, &w.attn_res, m, &r);
                let (x, normed) = mix_in(&r, &w.mlp_res, m);
                let f = moe(&x, &w.mlp);
                r = inject(&f, &normed, &w.mlp_res, m, &r);

                let (x, _) = mix_in(&r, &mtp.mixer, m);
                let draft = ops::linear::lm_head(&x, &m.head);
                if step == 0 {
                    seam::at(seam::MTP, &[&draft]);
                }
                token = ops::layout::argmax(&[&draft]);
                hidden = r;
                chain.push(draft);
            }
            let steps: Vec<&Value> = chain.iter().collect();
            seam::at(seam::MTP_DRAFTS, &[&ops::layout::argmax(&steps)]);
        }

        logits
    }
}

#[allow(clippy::too_many_arguments)]
fn mtp_attn(
    x: &Value,
    inputs: &Input<Facts>,
    m: &Model,
    plan: &Value,
    positions: &Value,
    a: &Attn,
    chain: bool,
) -> Value {
    let pages = inputs.kv(&a.kv);
    let write_page = inputs.write_page(&a.kv);
    let write_offset = inputs.write_offset(&a.kv);
    let d = m.head_dim;
    let (q, gate) = ops::layout::split_q_gate(&ops::linear::matmul(x, &a.qg_proj), d);
    let k = ops::linear::matmul(x, &a.k_proj);
    let v = ops::linear::matmul(x, &a.v_proj);
    let q = ops::elemwise::rmsnorm_per_head_plus_one(&q, &a.q_norm, d, a.q_norm_eps);
    let k = ops::elemwise::rmsnorm_per_head_plus_one(&k, &a.k_norm, d, a.k_norm_eps);
    let (q, k) = rotate(&q, &k, positions, m, a, d);
    if !chain {
        ops::attn::kv_append(&k, &v, pages, &write_page, &write_offset);
    }
    let o = ops::attn::prefill(&q, plan, pages, None, d, m.kv_heads, a.sm_scale);
    ops::linear::matmul(&ops::elemwise::gate_sigmoid_mul(&o, &gate), &a.o_proj)
}

fn mix_in(y: &Value, res: &Residual, m: &Model) -> (Value, Value) {
    let normed = ops::elemwise::rmsnorm_grouped_plus_one(y, &res.norm, m.hidden, res.eps);
    let mix = ops::linear::matmul(
        &ops::elemwise::silu_scaled(
            (m.streams as f32).recip(),
            &ops::linear::matmul(&normed, &res.down),
        ),
        &res.up,
    );
    (ops::elemwise::hc_mix(&mix, &normed, m.streams), normed)
}

fn inject(o: &Value, normed: &Value, res: &Residual, m: &Model, y: &Value) -> Value {
    let gates = ops::linear::matmul(
        normed,
        res.inject
            .as_ref()
            .expect("every layer site injects; only the final mixer does not"),
    );
    ops::elemwise::hc_inject(o, &gates, m.streams, y)
}

fn ple(y: &Value, ids: &Value, inputs: &Input<Facts>, m: &Model, p: &Ple) -> Value {
    let ids_state = inputs.state(&p.ids_state);
    let conv_state = inputs.state(&p.conv_state);

    let one = Facts::qo_one();
    let (ids_d, ids_p) = ids.split(&one);
    let grams = Value::merge(vec![
        ops::attn::ple_ngram_ids(
            &ids_d,
            ids_state,
            p.eos,
            &p.mults,
            &p.primes,
            &p.offsets,
            p.heads_per_ngram,
        ),
        ops::attn::ple_ngram_ids_chunked(
            &ids_p,
            ids_state,
            p.eos,
            &p.mults,
            &p.primes,
            &p.offsets,
            p.heads_per_ngram,
        ),
    ]);
    let e = ops::layout::embed_concat(
        &grams,
        &p.table,
        u32::try_from(p.padded_vocab).expect("the padded n-gram vocabulary addresses in i32"),
    );

    let key = ops::elemwise::rmsnorm_grouped_plus_one(
        &ops::linear::matmul(&e, &p.key_proj),
        &p.norm_key,
        m.hidden,
        p.eps,
    );
    let query = ops::elemwise::rmsnorm_grouped_plus_one(y, &p.norm_query, m.hidden, p.eps);
    let value = ops::linear::matmul(&e, &p.value_proj);
    let gated = ops::elemwise::ple_gate(&key, &query, &value, m.streams);

    let normed = ops::elemwise::rmsnorm_grouped_plus_one(&gated, &p.norm_conv, m.hidden, p.eps);
    let (normed_d, normed_p) = normed.split(&one);
    let conv = Value::merge(vec![
        ops::attn::ssm_causal_conv1d_dilated(
            &normed_d,
            &p.conv,
            conv_state,
            p.conv_kernel,
            p.dilation,
        ),
        ops::attn::ssm_causal_conv1d_chunked_dilated(
            &normed_p,
            &p.conv,
            conv_state,
            p.conv_kernel,
            p.dilation,
        ),
    ]);
    ops::elemwise::residual_add(&conv, &gated)
}

fn rotation_positions(inputs: &Input<Facts>, m: &Model) -> Value {
    match &m.tower {
        None => inputs.positions(),
        Some(_) => inputs.mrope_positions(),
    }
}

fn rotate(q: &Value, k: &Value, positions: &Value, m: &Model, a: &Attn, d: u32) -> (Value, Value) {
    match &m.tower {
        None => ops::elemwise::rope_partial(q, k, positions, a.rotary_dim, d, a.theta),
        Some(_) => ops::elemwise::rope_mrope(
            q,
            k,
            positions,
            MROPE_SECTIONS,
            MropeForm::Interleaved,
            a.rotary_dim,
            d,
            a.theta,
        ),
    }
}

fn tower(inputs: &Input<Facts>, t: &Tower) -> Value {
    let d = t.head_dim;
    let x = inputs.patches(t.patch_width);
    let segments = inputs.patch_segments();
    let grid = inputs.patch_positions();

    let mut y = ops::elemwise::add_bias(
        &t.patch_embed_bias,
        &ops::linear::matmul(&x, &t.patch_embed),
    );
    let ids = inputs.patch_embed_rows(t.taps);
    let pos = if t.taps == 1 {
        ops::layout::embed(&ids, &t.pos_embed, t.positions)
    } else {
        let weights = inputs.patch_embed_weights(t.taps);
        ops::layout::embed_weighted(&ids, &weights, &t.pos_embed, t.positions)
    };
    y = ops::elemwise::residual_add(&pos, &y);

    for b in &t.blocks {
        let n = ops::elemwise::layernorm(&y, &b.norm1, &b.norm1_bias, t.norm_eps);
        let (q, k, v) = ops::layout::split_qkv(
            &ops::elemwise::add_bias(&b.qkv_bias, &ops::linear::matmul(&n, &b.qkv)),
            t.hidden,
            t.hidden,
        );
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
        let n = ops::elemwise::layernorm(&y, &b.norm2, &b.norm2_bias, t.norm_eps);
        let h = ops::elemwise::add_bias(&b.fc1_bias, &ops::linear::matmul(&n, &b.fc1));
        let a = ops::linear::mlp_gelu_tanh(&h);
        y = ops::elemwise::residual_add(
            &ops::elemwise::add_bias(&b.fc2_bias, &ops::linear::matmul(&a, &b.fc2)),
            &y,
        );
    }

    let mg = &t.merger;
    let n = ops::elemwise::layernorm(&y, &mg.norm, &mg.norm_bias, t.norm_eps);
    let folded = ops::layout::merge_rows(&n, t.merge);
    let h = ops::elemwise::add_bias(&mg.fc1_bias, &ops::linear::matmul(&folded, &mg.fc1));
    let a = ops::linear::mlp_gelu_tanh(&h);
    ops::elemwise::add_bias(&mg.fc2_bias, &ops::linear::matmul(&a, &mg.fc2))
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

fn qsa_select(x: &Value, inputs: &Input<Facts>, m: &Model, a: &Attn, ix: &Indexer) -> Value {
    let keys = inputs.kv(&ix.keys);
    let write_page = inputs.write_page(&ix.keys);
    let write_offset = inputs.write_offset(&ix.keys);
    let positions = inputs.positions();

    let qk = ops::linear::matmul(x, &ix.qk_proj);
    let (q, k) = ops::layout::split_rows(&qk, ix.heads * ix.head_dim);

    ops::attn::index_kv_append(&k, keys, &write_page, &write_offset);
    let (bpos, breq, brope) = boundaries(&positions, &inputs.row_valid(), ix.ratio);
    let blk = ops::attn::index_block_mean(&bpos, &breq, keys, ix.head_dim, ix.ratio, m.kv);
    let blk = ops::elemwise::rmsnorm_plus_one(&blk, &ix.k_norm, ix.norm_eps);
    let blk = ops::elemwise::rope_partial_q(&blk, &brope, a.rotary_dim, ix.head_dim, a.theta);
    ops::attn::pool_kv_append(&blk, &bpos, &breq, keys, &write_page, &write_offset);

    let q = ops::elemwise::rmsnorm_per_head_plus_one(&q, &ix.q_norm, ix.head_dim, ix.norm_eps);
    let q = ops::elemwise::rope_partial_q(&q, &positions, a.rotary_dim, ix.head_dim, a.theta);
    ops::attn::index_topk(&q, None, keys, ix.heads, ix.head_dim, ix.top_k, ix.ratio)
}

#[allow(clippy::too_many_arguments)]
fn attn_mixer(
    x: &Value,
    inputs: &Input<Facts>,
    m: &Model,
    plan_m: &Value,
    plan_d: &Value,
    plan_p: &Value,
    plan_s: &Value,
    mask: &Value,
    a: &Attn,
    ix: Option<&Indexer>,
) -> Value {
    let pages = inputs.kv(&a.kv);
    let write_page = inputs.write_page(&a.kv);
    let write_offset = inputs.write_offset(&a.kv);
    let d = m.head_dim;
    let (q, gate) = ops::layout::split_q_gate(&ops::linear::matmul(x, &a.qg_proj), d);
    let k = ops::linear::matmul(x, &a.k_proj);
    let v = ops::linear::matmul(x, &a.v_proj);
    seam::at(seam::ATTN_QV, &[&q, &v]);
    let q = ops::elemwise::rmsnorm_per_head_plus_one(&q, &a.q_norm, d, a.q_norm_eps);
    let k = ops::elemwise::rmsnorm_per_head_plus_one(&k, &a.k_norm, d, a.k_norm_eps);
    let (q, k) = rotate(&q, &k, &rotation_positions(inputs, m), m, a, d);
    ops::attn::kv_append(&k, &v, pages, &write_page, &write_offset);
    seam::at(seam::ATTN_Q, &[&q]);

    let classes = || {
        [
            Facts::masked(),
            Facts::captures_scores(),
            Facts::qo_one(),
            Predicate::rest(),
        ]
    };
    let [mq, sq, dq, p] = q.split(classes());
    let (so, lse) = ops::attn::prefill_lse(&sq, plan_s, pages, None, d, m.kv_heads, a.sm_scale);
    seam::at(seam::SCORES, &[&lse]);
    let (od, op) = match ix.map(|ix| (ix, qsa_select(x, inputs, m, a, ix))) {
        Some((ix, selection)) => {
            let [_, _, d_sel, p_sel] = selection.split(classes());
            (
                ops::attn::decode_selected(
                    &dq, plan_d, &d_sel, pages, None, d, a.sm_scale, ix.ratio,
                ),
                ops::attn::prefill_selected(
                    &p, plan_p, &p_sel, pages, None, d, m.kv_heads, a.sm_scale, ix.ratio,
                ),
            )
        }
        None => (
            ops::attn::decode(&dq, plan_d, pages, None, d, a.sm_scale),
            ops::attn::prefill(&p, plan_p, pages, None, d, m.kv_heads, a.sm_scale),
        ),
    };
    let o = Value::merge(vec![
        ops::attn::masked(
            &mq, plan_m, mask, pages, None, d, m.kv_heads, true, a.sm_scale,
        ),
        so,
        od,
        op,
    ]);
    seam::at(seam::ATTN_OUT, &[&o]);
    ops::linear::matmul(&ops::elemwise::gate_sigmoid_mul(&o, &gate), &a.o_proj)
}

fn gdn_mixer(x: &Value, inputs: &Input<Facts>, g: &Gdn) -> Value {
    let conv_state = inputs.state(&g.conv_state);
    let delta_state = inputs.state(&g.delta_state);
    let qkvz = ops::linear::matmul(x, &g.in_qkvz);
    let ba = ops::linear::matmul(x, &g.in_ba);
    seam::at(seam::RECURRENT, &[&qkvz]);
    let width = Gdn::qkv_width(g.k_heads, g.v_heads, g.k_dim, g.v_dim);
    let one = Facts::qo_one();
    let (qkvz_d, qkvz_p) = qkvz.split(&one);
    let (ba_d, ba_p) = ba.split(&one);
    let (core_d, z_d) = {
        let (qkv, z) = ops::layout::split_rows(&qkvz_d, width);
        let qkv = ops::attn::ssm_causal_conv1d(&qkv, &g.conv, conv_state, g.conv_kernel);
        let gates = ops::attn::ssm_gdn_prep(&ba_d, &g.dt_bias, &g.a_log);
        let core = ops::attn::ssm_gated_delta(
            &qkv,
            &z,
            &gates,
            delta_state,
            g.k_heads,
            g.v_heads,
            g.k_dim,
            g.v_dim,
        );
        (core, z)
    };
    let (core_p, z_p) = {
        let (qkv, z) = ops::layout::split_rows(&qkvz_p, width);
        let qkv = ops::attn::ssm_causal_conv1d_chunked(&qkv, &g.conv, conv_state, g.conv_kernel);
        let gates = ops::attn::ssm_gdn_prep(&ba_p, &g.dt_bias, &g.a_log);
        let core = ops::attn::ssm_gated_delta_chunked(
            &qkv,
            &z,
            &gates,
            delta_state,
            g.k_heads,
            g.v_heads,
            g.k_dim,
            g.v_dim,
        );
        (core, z)
    };
    let o = Value::merge(vec![core_d, core_p]);
    let z = Value::merge(vec![z_d, z_p]);

    let o = ops::elemwise::rmsnorm_gated(
        &o,
        &z,
        &g.norm,
        g.v_dim,
        g.norm_eps,
        GateActivation::Sigmoid,
    );
    ops::linear::matmul(&o, &g.out_proj)
}

fn moe(x: &Value, mlp: &Mlp) -> Value {
    match mlp {
        Mlp::Dense { .. } => unreachable!("every qwen4 layer routes"),
        Mlp::Routed {
            router,
            gate_up,
            down,
            shared_gate_up,
            shared_down,
            shared_gate,
            experts,
            top_k,
            inter,
            shared_inter,
        } => {
            let (routes, weights) =
                ops::linear::moe_topk_softmax(&ops::linear::matmul(x, router), *experts, *top_k);
            let select = |act: &Value, bank: &model_dsl::Weight| {
                if matches!(bank.dtype, Dtype::Bf16 | Dtype::F16 | Dtype::F32) {
                    ops::linear::moe_matmul_select(act, bank, &routes, *top_k)
                } else {
                    ops::linear::moe_matmul_select_quant(act, bank, &routes, *top_k)
                }
            };
            let hidden = ops::linear::mlp_swiglu(&select(x, gate_up), *inter);
            let routed = ops::linear::moe_weighted_sum(&select(&hidden, down), &weights);
            let shared = ops::linear::matmul(
                &ops::linear::mlp_swiglu(&ops::linear::matmul(x, shared_gate_up), *shared_inter),
                shared_down,
            );
            let gate = ops::linear::matmul(x, shared_gate);
            ops::linear::moe_sigmoid_gate_add(&routed, &shared, &gate)
        }
    }
}
