use poem_dsl::fact;
use poem_dsl::{
    Dtype, ForwardHybrid, GateActivation, HybridSpec, Input, MropeForm, Value, Weight, ops, seam,
};

use super::model::{Attn, BonsaiSigns, DRAFT_DEPTH, Gdn, Head, Mixer, Mlp, Model, Tower};

const MROPE_SECTIONS: [u32; 3] = [11, 11, 10];

/// The Bonsai online-Hadamard block: the fork's shared `prism.hadamard.1024`
/// normalized Sylvester–Walsh matrix (oracle §2). Every rotated width — 5120,
/// 6144, 17408 — is a whole multiple of it.
const BONSAI_BLOCK: u32 = 1024;

/// The forward RHT applied to a rotated projection's INPUT: `H·(S·x)`, on a COPY
/// of `x`. The Hadamard folds in place and `x` is still live — a sibling
/// un-rotated projection (`in_ba`, `attn_v`), the LoRA correction, or the
/// residual all read the unrotated residual — so the copy is load-bearing.
fn rot_copy(x: &Value, signs: &Weight) -> Value {
    ops::elemwise::hadamard_signed(&ops::elemwise::copy(x), BONSAI_BLOCK, signs)
}

/// The forward RHT `H·(S·x)` consuming `x` (no copy) — for an activation that is
/// dead after the rotated matmul (a gated attention output, a swiglu residue).
fn rot_take(x: &Value, signs: &Weight) -> Value {
    ops::elemwise::hadamard_signed(x, BONSAI_BLOCK, signs)
}

impl ForwardHybrid for Model {
    fn caches(&self) -> HybridSpec {
        let mut c = HybridSpec::new();
        let kv = c.kv_space(self.kv);
        let plane = self.kv_heads as u64 * self.head_dim as u64;
        for w in &self.layers {
            match &w.mixer {
                Mixer::Attn(a) => {
                    c.kv(kv, a.kv.clone(), [plane, plane], self.head_dim)
                        .heads();
                }

                Mixer::Gdn(g) => {
                    let conv_ch = u64::from(Gdn::qkv_width(g.k_heads, g.v_heads, g.k_dim, g.v_dim));
                    c.state(
                        g.conv_state.clone(),
                        [g.conv_kernel as u64, conv_ch],
                        Dtype::Bf16,
                    )
                    .split(1);
                    c.state(
                        g.delta_state.clone(),
                        [g.v_heads as u64, g.k_dim as u64, g.v_dim as u64],
                        Dtype::Bf16,
                    )
                    .split(0);
                }
            }
        }
        if let Some(mtp) = &self.mtp {
            let a = &mtp.attn;
            c.kv(kv, a.kv.clone(), [plane, plane], self.head_dim)
                .heads();
        }
        if let Some(dflash) = &self.dflash {
            dflash.declare_caches(&mut c, kv);
        }
        c
    }

    fn forward(&self, inputs: Input) -> Value {
        let m = self;

        let classes = [fact::has(fact::Mask), fact::scores(), fact::single_token()];
        let (_, trunk_inputs) = match &m.dflash {
            Some(_) => (
                inputs.on(fact::block_draft()),
                inputs.on(!fact::block_draft()),
            ),
            None => (inputs.clone(), inputs.clone()),
        };
        let ([input_m, input_s, input_d], input_p) = trunk_inputs.partition(classes.clone());
        let plan_m = ops::attn::plan_prefill(&input_m, m.q_heads, m.kv_heads, m.head_dim, None);
        let plan_d = ops::attn::plan_decode(&input_d, m.q_heads, m.kv_heads, m.head_dim, None);
        let plan_p = ops::attn::plan_prefill(&input_p, m.q_heads, m.kv_heads, m.head_dim, None);
        let plan_s = ops::attn::plan_prefill(&input_s, m.q_heads, m.kv_heads, m.head_dim, None);
        let mask = inputs.mask();

        let towered = m.tower.as_ref().map(|t| tower(&inputs, t));

        let ids = inputs.tokens();
        let mut y = ops::layout::embed(&ids, &m.embed, m.vocab);

        // Bonsai token-embedding INVERSE-after-lookup (oracle §6): the stored
        // `token_embd` rows are ROTATED (`H·S·true`), so the residual stream needs
        // `true = S·(H·row)`. This is the inverse order `S·H` — the sign AFTER the
        // butterfly — not the forward `H·S` that `hadamard_signed` emits. With `H`
        // symmetric-orthonormal (`H·H = I`) and `S` a ±1 involution,
        //     S·H = H·(H·S·H),   so   S·H·y = H( H·S( H·y ) ),
        // i.e. three FWHT passes over the existing op (plain H, then `H·S`, then
        // plain H). No signs-after-butterfly kernel is needed; the paired plain
        // passes cancel to the exact inverse. The fork spells this as
        // GET_ROWS → H-matmul → MUL(signs.5120); this reproduces its value.
        if let Some(b) = &m.bonsai {
            let hy = ops::elemwise::hadamard_plain(&y, BONSAI_BLOCK);
            let hshy = ops::elemwise::hadamard_signed(&hy, BONSAI_BLOCK, &b.hidden);
            y = ops::elemwise::hadamard_plain(&hshy, BONSAI_BLOCK);
        }

        if let Some(t) = &towered {
            let imaged = y.on(fact::has(fact::Media));
            y = ops::layout::scatter_live_rows(t, &inputs.patch_routes(), &imaged).everywhere();
        }

        let (h_block, mut y) = match &m.dflash {
            Some(_) => {
                let (block, rest) = (y.on(fact::block_draft()), y.on(!fact::block_draft()));
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
                Mixer::Gdn(g) => gdn_mixer(&x, &inputs, g, m.bonsai.as_ref()),
            };
            let o = {
                let adapted = o.on(fact::has(fact::Adapter));
                let px = x.on(fact::has(fact::Adapter));
                ops::linear::lora_correct(&px, &w.lora_a, &w.lora_b, &routes, &adapted)
            };
            y = ops::elemwise::residual_add(&o, &y);

            let x = ops::elemwise::rmsnorm_plus_one(&y, &w.mlp_norm, w.mlp_norm_eps);
            let f = match &w.mlp {
                Mlp::Dense {
                    gate_up,
                    down,
                    inter,
                } => {
                    // Bonsai: `ffn_gate`+`ffn_up` share the residual input (5120),
                    // rotated once; `ffn_down` rotates the swiglu intermediate
                    // (17408). (oracle §6)
                    let gx;
                    let gate_in: &Value = match &m.bonsai {
                        Some(b) => {
                            gx = rot_copy(&x, &b.hidden);
                            &gx
                        }
                        None => &x,
                    };
                    let h = ops::linear::mlp_swiglu(&ops::linear::matmul(gate_in, gate_up), *inter);
                    let hx;
                    let down_in: &Value = match &m.bonsai {
                        Some(b) => {
                            hx = rot_take(&h, &b.ffn_down);
                            &hx
                        }
                        None => &h,
                    };
                    ops::linear::matmul(down_in, down)
                }
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
                let hb = d.arm(&inputs, fused, &block, &mask, &fact::block_draft());
                (Value::merge(vec![hb.clone(), x]), Some(hb))
            }
            _ => (x, None),
        };

        let x = ops::layout::gather_rows(&x, &inputs.readout_rows());
        // Bonsai output head (oracle §6): `result_norm` is rotated (5120) before
        // the (rotated) `output.weight` matmul: MUL(signs.5120) → H → MUL_MAT.
        // Rotating the gathered readout rows equals rotating pre-gather (the RHT
        // is per-row over the hidden dim). Copy: `x` is still read by the MTP
        // path below.
        let hx;
        let head_in: &Value = match &m.bonsai {
            Some(b) => {
                hx = rot_copy(&x, &b.hidden);
                &hx
            }
            None => &x,
        };
        let logits = ops::linear::lm_head(head_in, head);

        if let Some(d) = &m.dflash {
            d.plant_readout(&logits, &inputs, hb.as_ref(), &fact::block_draft());
        }

        if let Some(mtp) = &m.mtp {
            let input_mtp = inputs.on(fact::drafts());
            let plan_mtp =
                ops::attn::plan_prefill(&input_mtp, m.q_heads, m.kv_heads, m.head_dim, None);
            let dx = x.on(fact::drafts());
            let dlogits = logits.on(fact::drafts());
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
                dy = ops::elemwise::residual_add(&f, &dy);

                let read = match &mtp.norm {
                    Some(norm) => ops::elemwise::rmsnorm_plus_one(&dy, norm, mtp.norm_eps),
                    None => dy.clone(),
                };
                let draft = ops::linear::lm_head(&read, head);
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

fn rotate(q: &Value, k: &Value, inputs: &Input, m: &Model, a: &Attn, d: u32) -> (Value, Value) {
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

fn tower(inputs: &Input, t: &Tower) -> Value {
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
    inputs: &Input,
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
    // Bonsai (full-attention layers): `attn_q` (fused q+gate), `attn_k`, and
    // `attn_v` all rotate on the residual input (5120), sharing one rotated copy
    // — the fork builds q, k, AND v from the same Hadamard-rotated input. The
    // unrotated `x` still feeds the LoRA correction (adapters are not rotated).
    let rx;
    let qkv_in: &Value = match m.bonsai.as_ref() {
        Some(b) => {
            rx = rot_copy(x, &b.hidden);
            &rx
        }
        None => x,
    };
    let (q, gate) = ops::layout::split_q_gate(&ops::linear::matmul(qkv_in, &a.qg_proj), d);
    let k = ops::linear::matmul(qkv_in, &a.k_proj);
    let v = ops::linear::matmul(qkv_in, &a.v_proj);
    seam::at(seam::ATTN_QV, &[&q, &v]);
    let q = ops::elemwise::rmsnorm_per_head_plus_one(&q, &a.q_norm, d, a.q_norm_eps);
    let k = ops::elemwise::rmsnorm_per_head_plus_one(&k, &a.k_norm, d, a.k_norm_eps);
    let (q, k) = rotate(&q, &k, inputs, m, a, d);
    // C1: incoherence-processing rotation. H is orthonormal, so (Hq)·(Hk)ᵀ = q·kᵀ
    // — the scores, and therefore the softmax, are unchanged, and no read-side op
    // needs to know the cache is rotated. K and V are turned before they are
    // cached; Q is turned to keep the scores invariant (it is never cached, so it
    // is free); the block is the whole head, which is exact only because Q and K
    // turn identically over the RoPE'd sub-block. Each of q/k/v is single-use at
    // this point, so the fresh-value rotations cannot clobber a still-live buffer.
    let (q, k, v) = if m.rotate_kv {
        (
            ops::elemwise::hadamard_plain(&q, d),
            ops::elemwise::hadamard_plain(&k, d),
            ops::elemwise::hadamard_plain(&v, d),
        )
    } else {
        (q, k, v)
    };
    ops::attn::kv_append(&k, &v, pages, &write_page, &write_offset);
    seam::at(seam::ATTN_Q, &[&q]);

    let ([mq, sq, dq], p) =
        q.partition([fact::has(fact::Mask), fact::scores(), fact::single_token()]);
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
    // C1: undo V's turn. With V cached as V·H, each head's output is
    // O = P·(V·H) = (P·V)·H = O_true·H, so one more Hadamard (H·H = I) recovers
    // O_true before the gate and o_proj. `o` is single-use here.
    let o = if m.rotate_kv {
        ops::elemwise::hadamard_plain(&o, d)
    } else {
        o
    };
    // Bonsai `attn_output` (o-proj): the gated attention output has width
    // `q_heads·head_dim` = 6144, so it rotates with the SAME diagonal as `ssm_out`
    // (`signs.6144`) — and, unlike `ssm_out`, with NO v-head reorder (oracle node
    // `node_275`: MUL(attn_gated, signs.6144) → H → MUL_MAT). The gated output is
    // dead after, so consume it.
    let gated = ops::elemwise::gate_sigmoid_mul(&o, &gate);
    let o_in = match m.bonsai.as_ref() {
        Some(b) => rot_take(&gated, &b.ssm),
        None => gated,
    };
    ops::linear::matmul(&o_in, &a.o_proj)
}

fn mtp_attn(x: &Value, inputs: &Input, m: &Model, plan: &Value, a: &Attn, chain: bool) -> Value {
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
    // C1: the draft head owns a separate `kv.mtp` cache. To never mix bases in one
    // cache, it takes the SAME four rotations as the trunk — K/V rotated in when
    // appended, Q rotated for score-invariance every (chained) step, O un-rotated
    // out — rather than reading a plain cache with rotated queries.
    let (q, k) = if m.rotate_kv {
        (
            ops::elemwise::hadamard_plain(&q, d),
            ops::elemwise::hadamard_plain(&k, d),
        )
    } else {
        (q, k)
    };
    if !chain {
        let v = if m.rotate_kv {
            ops::elemwise::hadamard_plain(&v, d)
        } else {
            v
        };
        ops::attn::kv_append(&k, &v, pages, &write_page, &write_offset);
    }
    let o = ops::attn::prefill(&q, plan, pages, None, d, m.kv_heads, a.sm_scale);
    let o = if m.rotate_kv {
        ops::elemwise::hadamard_plain(&o, d)
    } else {
        o
    };
    ops::linear::matmul(&ops::elemwise::gate_sigmoid_mul(&o, &gate), &a.o_proj)
}

fn gdn_mixer(x: &Value, inputs: &Input, g: &Gdn, bonsai: Option<&BonsaiSigns>) -> Value {
    let conv_state = inputs.state(&g.conv_state);
    let delta_state = inputs.state(&g.delta_state);
    // Bonsai: `in_qkvz` (fused attn_qkv + attn_gate z) rotates on the residual
    // input (5120); `in_ba` (ssm_beta/alpha) does NOT rotate (oracle §6). One
    // rotated copy feeds `in_qkvz`; `in_ba` reads the unrotated `x`.
    let rx;
    let qkvz_in: &Value = match bonsai {
        Some(b) => {
            rx = rot_copy(x, &b.hidden);
            &rx
        }
        None => x,
    };
    let qkvz = ops::linear::matmul(qkvz_in, &g.in_qkvz);
    let ba = ops::linear::matmul(x, &g.in_ba);
    seam::at(seam::RECURRENT, &[&qkvz]);
    let width = Gdn::qkv_width(g.k_heads, g.v_heads, g.k_dim, g.v_dim);
    let one = fact::single_token();
    let (qkvz_d, qkvz_p) = (qkvz.on(one.clone()), qkvz.on(!one.clone()));
    let (ba_d, ba_p) = (ba.on(one.clone()), ba.on(!one.clone()));
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
    // Bonsai `ssm_out` (width 6144, `signs.6144`): sign→H rotation-undo, then the
    // block-canonical `out_proj`.
    //
    // The fork reorders the GDN output's v-heads before the sign→H (oracle §6,
    // `gdn_v_grouped=true`): `final_output{6144}` → reshape{128,16,3} →
    // PERMUTE(0,2,1,3)→{128,3,16} → MUL(signs.6144) → H → MUL_MAT(ssm_out). The
    // tiled→block permute is folded into the weights at IMPORT (the GDN scan's
    // v-head-indexed inputs are reordered when `gdn_v_grouped` is set), so pie's
    // block-pairing scan already emits block-ordered heads — the same order the
    // fork's reordered sign→H and the block-canonical `out_proj` expect. No
    // output-side reorder here; the shared GDN scan kernel stays untouched.
    let o_in = match bonsai {
        Some(b) => rot_take(&o, &b.ssm),
        None => o,
    };
    ops::linear::matmul(&o_in, &g.out_proj)
}
