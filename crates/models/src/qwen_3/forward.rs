use model_dsl::{
    Classify, Dtype, ForwardHybrid, GateActivation, HybridSpec, Input, MropeForm, Predicate,
    Request, Value, Weight, ops, seam,
};

use super::model::{Attn, DRAFT_DEPTH, Gdn, Head, Mixer, Mlp, Model, Tower};

const MROPE_SECTIONS: [u32; 3] = [11, 11, 10];

pub struct Facts {
    pub qo_one: bool,
    pub has_adapter: bool,
    pub drafts: bool,
    pub captures_scores: bool,
    pub masked: bool,
    pub media: bool,
    pub block_draft: bool,
}

impl Facts {
    #[must_use]
    pub fn qo_one() -> Predicate {
        Predicate::fact(0)
    }

    #[must_use]
    pub fn has_adapter() -> Predicate {
        Predicate::fact(1)
    }

    #[must_use]
    pub fn drafts() -> Predicate {
        Predicate::fact(2)
    }

    #[must_use]
    pub fn captures_scores() -> Predicate {
        Predicate::fact(3)
    }

    #[must_use]
    pub fn masked() -> Predicate {
        Predicate::fact(4)
    }

    #[must_use]
    pub fn media() -> Predicate {
        Predicate::fact(5)
    }

    #[must_use]
    pub fn block_draft() -> Predicate {
        Predicate::fact(6)
    }
}

impl Classify for Facts {
    fn of(r: &Request) -> Facts {
        Facts {
            qo_one: r.query_len() == 1,
            has_adapter: r.has_adapter(),
            drafts: r.drafts(),
            captures_scores: r.captures_scores(),
            masked: r.has_custom_mask(),
            media: r.has_media(),
            block_draft: r.drafts_a_block(),
        }
    }

    fn word(&self) -> u64 {
        u64::from(self.qo_one)
            | (u64::from(self.has_adapter) << 1)
            | (u64::from(self.drafts) << 2)
            | (u64::from(self.captures_scores) << 3)
            | (u64::from(self.masked) << 4)
            | (u64::from(self.media) << 5)
            | (u64::from(self.block_draft) << 6)
    }
}

impl ForwardHybrid for Model {
    type Facts = Facts;

    fn caches(&self) -> HybridSpec {
        let mut c = HybridSpec::new();
        let kv = c.kv_space(self.kv);
        let plane = self.kv_heads as u64 * self.head_dim as u64;
        for w in &self.layers {
            match &w.mixer {
                Mixer::Attn(a) => {
                    c.kv(kv, a.kv.clone(), [plane, plane]);
                }

                Mixer::Gdn(g) => {
                    let conv_ch = u64::from(Gdn::qkv_width(g.k_heads, g.v_heads, g.k_dim, g.v_dim));
                    c.state(
                        g.conv_state.clone(),
                        [g.conv_kernel as u64, conv_ch],
                        Dtype::Bf16,
                    );
                    c.state(
                        g.delta_state.clone(),
                        [g.v_heads as u64, g.k_dim as u64, g.v_dim as u64],
                        Dtype::Bf16,
                    );
                }
            }
        }
        if let Some(mtp) = &self.mtp {
            let a = &mtp.attn;
            c.kv(kv, a.kv.clone(), [plane, plane]);
        }
        if let Some(dflash) = &self.dflash {
            dflash.declare_caches(&mut c, kv);
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
        let (_, trunk_inputs) = match &m.dflash {
            Some(_) => inputs.split(&Facts::block_draft()),
            None => (inputs.clone(), inputs.clone()),
        };
        let [input_m, input_s, input_d, input_p] = trunk_inputs.split(classes);
        let plan_m = ops::attn::plan_prefill(&input_m, m.q_heads, m.kv_heads, m.head_dim, None);
        let plan_d = ops::attn::plan_decode(&input_d, m.q_heads, m.kv_heads, m.head_dim, None);
        let plan_p = ops::attn::plan_prefill(&input_p, m.q_heads, m.kv_heads, m.head_dim, None);
        let plan_s = ops::attn::plan_prefill(&input_s, m.q_heads, m.kv_heads, m.head_dim, None);
        let mask = inputs.mask();

        let towered = m.tower.as_ref().map(|t| tower(&inputs, t));

        let ids = inputs.tokens();
        let mut y = ops::layout::embed(&ids, &m.embed, m.vocab);

        if let Some(t) = &towered {
            let (imaged, _) = y.split(&Facts::media());
            y = ops::layout::scatter_live_rows(t, &inputs.patch_routes(), &imaged).everywhere();
        }

        let (h_block, mut y) = match &m.dflash {
            Some(_) => {
                let (block, rest) = y.split(&Facts::block_draft());
                (Some(block), rest)
            }
            None => (None, y),
        };

        let mut fused: Option<Value> = None;
        let routes = inputs.adapter_routes();
        for (l, w) in inputs.walk_layers(&m.layers) {
            let x = ops::elemwise::rmsnorm_plus_one(&y, &w.mixer_norm, w.mixer_norm_eps);
            let o = match &w.mixer {
                Mixer::Attn(a) => {
                    attn_mixer(&x, &inputs, m, &plan_m, &plan_d, &plan_p, &plan_s, &mask, a)
                }
                Mixer::Gdn(g) => gdn_mixer(&x, &inputs, g),
            };
            let o = if m.tp > 1 {
                ops::collective::all_reduce(&o)
            } else {
                o
            };
            let o = {
                let (adapted, _) = o.split(&Facts::has_adapter());
                let (px, _) = x.split(&Facts::has_adapter());
                ops::linear::lora_correct(&px, &w.lora_a, &w.lora_b, &routes, &adapted)
            };
            y = ops::elemwise::residual_add(&o, &y);

            let x = ops::elemwise::rmsnorm_plus_one(&y, &w.mlp_norm, w.mlp_norm_eps);
            let f = match &w.mlp {
                Mlp::Dense {
                    gate_up,
                    down,
                    inter,
                } => ops::linear::matmul(
                    &ops::linear::mlp_swiglu(&ops::linear::matmul(&x, gate_up), *inter),
                    down,
                ),
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
                        &ops::linear::matmul(&x, router),
                        *experts,
                        *top_k,
                    );
                    let select = |act: &Value, bank: &Weight| {
                        if matches!(bank.dtype, Dtype::Bf16 | Dtype::F16 | Dtype::F32) {
                            ops::linear::moe_matmul_select(act, bank, &routes, *top_k)
                        } else {
                            ops::linear::moe_matmul_select_quant(act, bank, &routes, *top_k)
                        }
                    };
                    let hidden = ops::linear::mlp_swiglu(&select(&x, gate_up), *inter);
                    let routed = ops::linear::moe_weighted_sum(&select(&hidden, down), &weights);
                    let shared = ops::linear::matmul(
                        &ops::linear::mlp_swiglu(
                            &ops::linear::matmul(&x, shared_gate_up),
                            *shared_inter,
                        ),
                        shared_down,
                    );
                    ops::linear::moe_sigmoid_gate_add(
                        &routed,
                        &shared,
                        &ops::linear::matmul(&x, shared_gate),
                    )
                }
            };
            let f = if m.tp > 1 {
                ops::collective::all_reduce(&f)
            } else {
                f
            };
            y = ops::elemwise::residual_add(&f, &y);
            if let Some(d) = &m.dflash {
                d.tap(l, &y, &mut fused);
            }
        }

        let x = ops::elemwise::rmsnorm_plus_one(&y, &m.final_norm, m.final_norm_eps);
        let head = match &m.head {
            Head::Tied => &m.embed,
            Head::Bank(bank) => bank,
        };

        let (x, hb) = match (&m.dflash, h_block) {
            (Some(d), Some(block)) => {
                let fused = fused.as_ref().expect("a block drafter tapped the trunk");
                let hb = d.arm(&inputs, fused, &block, &mask, &Facts::block_draft());
                (Value::merge(vec![hb.clone(), x]), Some(hb))
            }
            _ => (x, None),
        };

        let x = ops::layout::gather_rows(&x, &inputs.readout_rows());
        let logits = ops::linear::lm_head(&x, head);
        let logits = if head_banded(m, head) {
            ops::collective::all_gather(&logits, m.tp)
        } else {
            logits
        };

        if let Some(d) = &m.dflash {
            d.plant_readout(&logits, &inputs, hb.as_ref(), &Facts::block_draft());
        }

        if let Some(mtp) = &m.mtp {
            let (input_mtp, _) = inputs.split(&Facts::drafts());
            let plan_mtp =
                ops::attn::plan_prefill(&input_mtp, m.q_heads, m.kv_heads, m.head_dim, None);
            let (dx, _) = x.split(&Facts::drafts());
            let (dlogits, _) = logits.split(&Facts::drafts());
            let mut chosen = ops::layout::argmax(&[&dlogits]);
            let mut hidden = dx;
            let mut chain: Vec<Value> = Vec::with_capacity(DRAFT_DEPTH as usize);
            for step in 0..DRAFT_DEPTH {
                let e = ops::layout::embed(&chosen, &m.embed, m.vocab);
                let (e, h) = match &mtp.pre_fc {
                    Some(pre) => (
                        ops::elemwise::rmsnorm_plus_one(&e, &pre.embedding, pre.eps),
                        ops::elemwise::rmsnorm_plus_one(&hidden, &pre.hidden, pre.eps),
                    ),
                    None => (e, hidden.clone()),
                };
                let mut dy = ops::elemwise::residual_add(
                    &ops::linear::matmul(&e, &mtp.fc_embed),
                    &ops::linear::matmul(&h, &mtp.fc_hidden),
                );

                let a = &mtp.attn;
                let nx = ops::elemwise::rmsnorm_plus_one(&dy, &mtp.mixer_norm, mtp.mixer_norm_eps);
                let o = mtp_attn(&nx, &inputs, m, &plan_mtp, a, step > 0);
                let o = if m.tp > 1 {
                    ops::collective::all_reduce(&o)
                } else {
                    o
                };
                dy = ops::elemwise::residual_add(&o, &dy);

                let nx = ops::elemwise::rmsnorm_plus_one(&dy, &mtp.mlp_norm, mtp.mlp_norm_eps);
                let Mlp::Dense {
                    gate_up,
                    down,
                    inter,
                } = &mtp.mlp
                else {
                    panic!("a draft head is one block and routes to no experts");
                };
                let f = ops::linear::matmul(
                    &ops::linear::mlp_swiglu(&ops::linear::matmul(&nx, gate_up), *inter),
                    down,
                );
                let f = if m.tp > 1 {
                    ops::collective::all_reduce(&f)
                } else {
                    f
                };
                dy = ops::elemwise::residual_add(&f, &dy);

                let read = match &mtp.norm {
                    Some(norm) => ops::elemwise::rmsnorm_plus_one(&dy, norm, mtp.norm_eps),
                    None => dy.clone(),
                };
                let draft = ops::linear::lm_head(&read, head);
                let draft = if head_banded(m, head) {
                    ops::collective::all_gather(&draft, m.tp)
                } else {
                    draft
                };
                if step == 0 {
                    seam::at(seam::MTP, &[&draft]);
                }
                chosen = ops::layout::argmax(&[&draft]);
                hidden = dy;
                chain.push(draft);
            }
            let steps: Vec<&Value> = chain.iter().collect();
            seam::at(seam::MTP_DRAFTS, &[&ops::layout::argmax(&steps)]);
        }

        logits
    }
}

fn rotate(
    q: &Value,
    k: &Value,
    inputs: &Input<Facts>,
    m: &Model,
    a: &Attn,
    d: u32,
) -> (Value, Value) {
    match &m.tower {
        None => ops::elemwise::rope_partial(q, k, &inputs.positions(), a.rotary_dim, d, a.theta),
        Some(_) => ops::elemwise::rope_mrope(
            q,
            k,
            &inputs.mrope_positions(),
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

    let m = &t.merger;
    let n = ops::elemwise::layernorm(&y, &m.norm, &m.norm_bias, t.norm_eps);
    let folded = ops::layout::merge_rows(&n, t.merge);
    let h = ops::elemwise::add_bias(&m.fc1_bias, &ops::linear::matmul(&folded, &m.fc1));
    let a = ops::linear::mlp_gelu_tanh(&h);
    ops::elemwise::add_bias(&m.fc2_bias, &ops::linear::matmul(&a, &m.fc2))
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
    let (q, k) = rotate(&q, &k, inputs, m, a, d);
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
        ops::attn::masked(
            &mq, plan_m, mask, pages, None, d, m.kv_heads, true, a.sm_scale,
        ),
        so,
        ops::attn::decode(&dq, plan_d, pages, None, d, a.sm_scale),
        ops::attn::prefill(&p, plan_p, pages, None, d, m.kv_heads, a.sm_scale),
    ]);
    seam::at(seam::ATTN_OUT, &[&o]);
    ops::linear::matmul(&ops::elemwise::gate_sigmoid_mul(&o, &gate), &a.o_proj)
}

fn mtp_attn(
    x: &Value,
    inputs: &Input<Facts>,
    m: &Model,
    plan: &Value,
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
    let (q, k) = rotate(&q, &k, inputs, m, a, d);
    if !chain {
        ops::attn::kv_append(&k, &v, pages, &write_page, &write_offset);
    }
    let o = ops::attn::prefill(&q, plan, pages, None, d, m.kv_heads, a.sm_scale);
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

    let o =
        ops::elemwise::rmsnorm_gated(&o, &z, &g.norm, g.v_dim, g.norm_eps, GateActivation::Silu);
    ops::linear::matmul(&o, &g.out_proj)
}

fn head_banded(m: &Model, head: &Weight) -> bool {
    head.dim(0) < u64::from(m.vocab)
}
