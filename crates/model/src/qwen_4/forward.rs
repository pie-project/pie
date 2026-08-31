//! The qwen4 forward: qwen_3's hybrid walk under a residual that is not a
//! sum.
//!
//! Every sublayer is bracketed by its [`Residual`] site — grouped norm,
//! low-rank sigmoid mix in, per-stream gated injection out — so the wide
//! `[streams · hidden]` row is what rides the layer loop, and the narrow
//! `hidden` row exists only between a site's mix and its inject. The PLE
//! folds its n-gram enrichment into the wide row at its one declared layer;
//! the final mixer folds the wide row down for the head, and is the last
//! normalization the logits see (there is no `final_norm` — see
//! [`Model::mixer`]).

use model_dsl::{
    Classify, Dtype, ForwardHybrid, GateActivation, HybridSpec, Input, Predicate, Request, Value,
    ops, seam,
};

use super::model::{Attn, Gdn, Mixer, Mlp, Model, Ple, Residual};

pub struct Facts {
    pub qo_one: bool,
    pub captures_scores: bool,
    pub masked: bool,
}

impl Facts {
    #[must_use]
    pub fn qo_one() -> Predicate {
        Predicate::fact(0)
    }

    /// The observability window — `qwen_3::forward::Facts::captures_scores`'
    /// bit, one family over: a lane that asked for its attention scores gets
    /// its own prefill schedule and an lse the seam captures.
    #[must_use]
    pub fn captures_scores() -> Predicate {
        Predicate::fact(1)
    }

    /// The custom-mask window, constraining attention layers only — the GDN
    /// and PLE state walks are sequential by construction and take no mask.
    #[must_use]
    pub fn masked() -> Predicate {
        Predicate::fact(2)
    }
}

impl Classify for Facts {
    fn of(r: &Request) -> Facts {
        Facts {
            qo_one: r.query_len() == 1,
            captures_scores: r.captures_scores(),
            masked: r.has_custom_mask(),
        }
    }

    fn word(&self) -> u64 {
        u64::from(self.qo_one)
            | (u64::from(self.captures_scores) << 1)
            | (u64::from(self.masked) << 2)
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
                Mixer::Attn(a) => {
                    c.kv(kv, a.kv.clone(), [plane, plane]);
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
        if let Some(p) = &self.ple {
            let wide = u64::from(self.streams) * u64::from(self.hidden);
            // The hasher's window holds token IDS — the one state row whose
            // element is an integer (see `CacheRow::State`'s doc). Its length
            // is the n-gram context, and the dilation IS `ngram_size` (the
            // model.rs declaration says why).
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

        // The four-way attention carve, `qwen_3::forward`'s own: one
        // schedule per class, each built off that class's arm of `inputs`.
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

        let ids = inputs.tokens();
        let narrow = ops::layout::embed(&ids, &m.embed, m.vocab);
        // The stream fan opens here and closes at the final mixer: the
        // embedding is tiled across every stream, which is the reference's
        // `repeat(1, 1, hc_count)`.
        let mut y = ops::elemwise::hc_expand(&narrow, m.streams);

        for (l, w) in m.layers.iter().enumerate() {
            if let Some(p) = m.ple.as_ref().filter(|p| p.layer as usize == l) {
                let enriched = ple(&y, &ids, &inputs, m, p);
                y = ops::elemwise::residual_add(&enriched, &y);
            }

            let (x, normed) = mix_in(&y, &w.attn_res, m);
            let o = match &w.mixer {
                Mixer::Attn(a) => {
                    attn_mixer(&x, &inputs, m, &plan_m, &plan_d, &plan_p, &plan_s, &mask, a)
                }
                Mixer::Gdn(g) => gdn_mixer(&x, &inputs, g),
            };
            y = inject(&o, &normed, &w.attn_res, m, &y);

            let (x, normed) = mix_in(&y, &w.mlp_res, m);
            let f = moe(&x, &w.mlp);
            y = inject(&f, &normed, &w.mlp_res, m, &y);
        }

        let (x, _) = mix_in(&y, &m.mixer, m);
        ops::linear::lm_head(&x, &m.head)
    }
}

/// One residual site's inward half: the grouped norm over the wide row, and
/// the low-rank sigmoid mix that folds it to the sublayer's input. Returns
/// the normed wide row too — the injection gates read it.
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

/// The outward half: the sublayer output returns into every stream under its
/// own gate. The final mixer never comes here — it has no `inject` bank.
fn inject(o: &Value, normed: &Value, res: &Residual, m: &Model, y: &Value) -> Value {
    let gates = ops::linear::matmul(
        normed,
        res.inject
            .as_ref()
            .expect("every layer site injects; only the final mixer does not"),
    );
    ops::elemwise::hc_inject(o, &gates, m.streams, y)
}

/// The PLE enrichment, transcribed from `Qwen4ExpTextPLELayer.forward`:
/// hash, gather, gate per stream, and mix locally through the dilated
/// depthwise convolution. Answers the wide addend; the trunk adds it.
fn ple(y: &Value, ids: &Value, inputs: &Input<Facts>, m: &Model, p: &Ple) -> Value {
    let ids_state = inputs.state(&p.ids_state);
    let conv_state = inputs.state(&p.conv_state);

    // The hasher is a state walk, so it splits the way the GDN mixers do:
    // one-token lanes step, the rest walk their request's rows.
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

/// The gated-attention arm — `qwen_3::forward::attn_mixer`, minus the tower
/// (this SKU is text-only, so the rotation is the scalar one) and minus the
/// adapter window (this family declares no adapter banks).
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
    let (q, k) =
        ops::elemwise::rope_partial(&q, &k, &inputs.positions(), a.rotary_dim, d, a.theta);
    ops::attn::kv_append(&k, &v, pages, &write_page, &write_offset);
    seam::at(seam::ATTN_Q, &[&q]);

    let [mq, sq, dq, p] = q.split([
        Facts::masked(),
        Facts::captures_scores(),
        Facts::qo_one(),
        Predicate::rest(),
    ]);
    let (so, lse) = ops::attn::prefill_lse(&sq, plan_s, pages, None, d, m.kv_heads, a.sm_scale);
    seam::at(seam::SCORES, &[&lse]);
    let o = Value::merge(vec![
        ops::attn::masked(&mq, plan_m, mask, pages, None, d, a.sm_scale),
        so,
        ops::attn::decode(&dq, plan_d, pages, None, d, a.sm_scale),
        ops::attn::prefill(&p, plan_p, pages, None, d, m.kv_heads, a.sm_scale),
    ]);
    seam::at(seam::ATTN_OUT, &[&o]);
    ops::linear::matmul(&ops::elemwise::gate_sigmoid_mul(&o, &gate), &a.o_proj)
}

/// The GatedDeltaNet arm — `qwen_3::forward::gdn_mixer` with the one
/// difference the config states: `output_gate_type: "sigmoid"`, so the gated
/// norm squashes its gate instead of siluing it.
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

/// The routed arm — `qwen_3::forward`'s, dtype-selected select and all.
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
            let (routes, weights) = ops::linear::moe_topk_softmax(
                &ops::linear::matmul(x, router),
                *experts,
                *top_k,
            );
            let select = |act: &Value, bank: &model_dsl::Weight| match bank.dtype {
                Dtype::Mxfp4 | Dtype::MlxU4 | Dtype::MlxU8 => {
                    ops::linear::moe_matmul_select_quant(act, bank, &routes, *top_k)
                }
                _ => ops::linear::moe_matmul_select(act, bank, &routes, *top_k),
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
