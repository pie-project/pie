use poem_dsl::fact;
use poem_dsl::{Dtype, ForwardHybrid, HybridSpec, Input, Value, ops, seam};

use super::model::{AttnRes, Kda, Mixer, Mla, Mlp, Model};

impl ForwardHybrid for Model {
    fn caches(&self) -> HybridSpec {
        let mut c = HybridSpec::new();
        let kv = c.kv_space(self.kv);
        for w in &self.layers {
            match &w.mixer {
                Mixer::Mla(a) => {
                    c.kv(
                        kv,
                        a.kv.clone(),
                        [self.kv_lora_rank as u64, a.qk_rope_head_dim as u64],
                        a.qk_rope_head_dim,
                    );
                }

                Mixer::Kda(k) => {
                    let width = (k.heads * k.head_dim) as u64;
                    c.state(
                        k.conv_state.clone(),
                        [k.conv_kernel as u64, 3 * width],
                        Dtype::Bf16,
                    )
                    .split(1);
                    // the KDA recurrence keeps its state in f32 on every backend
                    c.state(
                        k.delta_state.clone(),
                        [k.heads as u64, k.head_dim as u64, k.head_dim as u64],
                        Dtype::F32,
                    )
                    .split(0);
                }
            }
        }
        c
    }

    fn forward(&self, inputs: Input) -> Value {
        let m = self;

        let (input_d, input_p) = (
            inputs.on(fact::single_token()),
            inputs.on(!fact::single_token()),
        );
        let plan = [
            ops::attn::mla_plan(&input_d, m.mla_heads, m.kv_lora_rank),
            ops::attn::mla_plan(&input_p, m.mla_heads, m.kv_lora_rank),
        ];
        let ids = inputs.tokens();
        let mut y = ops::layout::embed(&ids, &m.embed, m.vocab);
        let mut blocks: Vec<Value> = Vec::new();
        let every = m.attn_res == AttnRes::Every;

        let routes = inputs.adapter_routes();
        for (l, w) in inputs.walk_layers(&m.layers) {
            // The sublayer input: under `Every`, a blend of the closed blocks and
            // the running prefix sum `y` (the released model); under
            // `AtBlockStart`, `y` itself, re-blended where a block opens.
            let mut h = y.clone();
            if let Some(b) = &w.res_blend {
                if every {
                    if !blocks.is_empty() {
                        h = ops::elemwise::res_blend(&y, &blocks, &b.norm, b.norm_eps, &b.proj);
                    }
                } else {
                    y = ops::elemwise::res_blend(&y, &blocks, &b.norm, b.norm_eps, &b.proj);
                    blocks.push(y.clone());
                    h = y.clone();
                }
            }
            // Where a block opens the prefix sum is banked and restarts from
            // this layer's attention output.
            let mut fresh = false;
            if every && m.opens_block(l) {
                blocks.push(y.clone());
                fresh = true;
            }

            let x = ops::elemwise::rmsnorm(&h, &w.mixer_norm, w.mixer_norm_eps);
            let o = match &w.mixer {
                Mixer::Mla(a) => mla_mixer(&x, &inputs, &plan, m, a),
                Mixer::Kda(k) => kda_mixer(&x, &inputs, k),
            };
            let o = {
                let adapted = o.on(fact::has(fact::Adapter));
                let px = x.on(fact::has(fact::Adapter));
                ops::linear::lora_correct(&px, &w.lora_a, &w.lora_b, &routes, &adapted)
            };
            y = if fresh {
                o
            } else {
                ops::elemwise::residual_add(&o, &y)
            };

            let h = match &w.mlp_res {
                Some(b) if !blocks.is_empty() => {
                    ops::elemwise::res_blend(&y, &blocks, &b.norm, b.norm_eps, &b.proj)
                }
                _ => y.clone(),
            };
            let x = ops::elemwise::rmsnorm(&h, &w.mlp_norm, w.mlp_norm_eps);
            let f = match &w.mlp {
                Mlp::Dense {
                    gate_up,
                    down,
                    inter,
                    beta,
                    up_cap,
                } => ops::linear::matmul(
                    &ops::linear::mlp_situ(
                        &ops::linear::matmul(&x, gate_up),
                        *inter,
                        *beta,
                        *up_cap,
                    ),
                    down,
                ),
                Mlp::Routed {
                    router,
                    bias,
                    gate_up,
                    down,
                    shared,
                    latent,
                    experts,
                    top_k,
                    renorm,
                    routed_scaling,
                    inter,
                    beta,
                    up_cap,
                } => {
                    let logits = ops::linear::matmul(&x, router);
                    let (routes, weights) = match bias {
                        Some(bias) => ops::linear::moe_topk_sigmoid_biased(
                            &logits,
                            bias,
                            *experts,
                            *top_k,
                            *renorm,
                            *routed_scaling,
                        ),
                        None => ops::linear::moe_topk_sigmoid(
                            &logits,
                            *experts,
                            *top_k,
                            *renorm,
                            *routed_scaling,
                        ),
                    };
                    let z = match latent {
                        Some(lat) => ops::linear::matmul(&x, &lat.down),
                        None => x.clone(),
                    };
                    let hidden = ops::linear::moe_matmul_select_quant(&z, gate_up, &routes, *top_k);
                    let act = ops::linear::mlp_situ(&hidden, *inter, *beta, *up_cap);
                    let routed = ops::linear::moe_weighted_sum(
                        &ops::linear::moe_matmul_select_quant(&act, down, &routes, *top_k),
                        &weights,
                    );
                    let routed = match latent {
                        Some(lat) => {
                            let r = match &lat.norm {
                                Some(norm) => ops::elemwise::rmsnorm(&routed, norm, lat.norm_eps),
                                None => routed,
                            };
                            ops::linear::matmul(&r, &lat.up)
                        }
                        None => routed,
                    };
                    match shared {
                        None => routed,
                        Some(s) => {
                            let act = ops::linear::mlp_situ(
                                &ops::linear::matmul(&x, &s.gate_up),
                                s.inter,
                                *beta,
                                *up_cap,
                            );
                            ops::elemwise::residual_add(
                                &ops::linear::matmul(&act, &s.down),
                                &routed,
                            )
                        }
                    }
                }
            };
            y = ops::elemwise::residual_add(&f, &y);
        }

        let y = match &m.output_res {
            Some(b) if !blocks.is_empty() => {
                ops::elemwise::res_blend(&y, &blocks, &b.norm, b.norm_eps, &b.proj)
            }
            _ => y,
        };
        let x = ops::elemwise::rmsnorm(&y, &m.final_norm, m.final_norm_eps);
        let x = ops::layout::gather_rows(&x, &inputs.readout_rows());
        ops::linear::lm_head(&x, &m.head)
    }
}

fn mla_mixer(x: &Value, inputs: &Input, plan: &[Value; 2], m: &Model, a: &Mla) -> Value {
    let pages = inputs.kv(&a.kv);
    let write_page = inputs.write_page(&a.kv);
    let write_offset = inputs.write_offset(&a.kv);
    let q_a = ops::linear::matmul(x, &a.q_a_proj);
    let q_a = ops::elemwise::rmsnorm(&q_a, &a.q_a_norm, a.q_a_norm_eps);
    let (kv_c, k_pe) = ops::attn::mla_latents(
        &ops::linear::matmul(x, &a.kv_a_proj),
        &a.kv_a_norm,
        a.kv_a_norm_eps,
        m.kv_lora_rank,
    );
    let (q_nope, q_pe) = ops::attn::mla_split_q_b(
        &ops::linear::matmul(&q_a, &a.q_b_proj),
        m.mla_heads,
        a.qk_nope_head_dim,
        a.qk_rope_head_dim,
    );
    ops::attn::mla_kv_append(&kv_c, &k_pe, pages, &write_page, &write_offset);

    let q = ops::attn::mla_absorb_q(
        &q_nope,
        &a.kv_b_proj,
        m.mla_heads,
        m.kv_lora_rank,
        a.qk_nope_head_dim,
        a.v_head_dim,
    );
    seam::at(seam::ATTN_Q, &[&q]);

    let one = fact::single_token();
    let (dq, p) = (q.on(one.clone()), q.on(!one.clone()));
    let (dpe, ppe) = (q_pe.on(one.clone()), q_pe.on(!one.clone()));
    let latent = Value::merge(vec![
        ops::attn::mla_decode(
            &dq,
            &plan[0],
            &dpe,
            pages,
            m.mla_heads,
            m.kv_lora_rank,
            a.sm_scale,
        ),
        ops::attn::mla_prefill(
            &p,
            &plan[1],
            &ppe,
            pages,
            m.mla_heads,
            m.kv_lora_rank,
            a.sm_scale,
        ),
    ]);
    let o = ops::attn::mla_absorb_out(
        &latent,
        &a.kv_b_proj,
        m.mla_heads,
        m.kv_lora_rank,
        a.qk_nope_head_dim,
        a.v_head_dim,
    );
    let o = match &a.gate {
        None => o,
        Some(g) => ops::elemwise::gate_sigmoid_mul(&o, &ops::linear::matmul(x, g)),
    };
    seam::at(seam::ATTN_OUT, &[&o]);
    ops::linear::matmul(&o, &a.o_proj)
}

fn kda_mixer(x: &Value, inputs: &Input, k: &Kda) -> Value {
    let conv = inputs.state(&k.conv_state);
    let delta = inputs.state(&k.delta_state);
    let qkv = ops::linear::matmul(x, &k.qkv);
    let f = ops::linear::matmul(&ops::linear::matmul(x, &k.f_a), &k.f_b);
    let b = ops::linear::matmul(x, &k.b);
    seam::at(seam::RECURRENT, &[&qkv]);

    let one = fact::single_token();
    let (qkv_d, qkv_p) = (qkv.on(one.clone()), qkv.on(!one.clone()));
    let (f_d, f_p) = (f.on(one.clone()), f.on(!one.clone()));
    let (b_d, b_p) = (b.on(one.clone()), b.on(!one.clone()));
    let core = Value::merge(vec![
        {
            let mixed = ops::attn::ssm_causal_conv1d(&qkv_d, &k.conv, conv, k.conv_kernel);
            ops::attn::ssm_kda_step(
                &mixed,
                &f_d,
                &b_d,
                &k.dt_bias,
                &k.a_log,
                delta,
                k.heads,
                k.head_dim,
                k.norm_eps,
                k.gate_floor,
            )
        },
        {
            let mixed = ops::attn::ssm_causal_conv1d_chunked(&qkv_p, &k.conv, conv, k.conv_kernel);
            ops::attn::ssm_kda_chunked(
                &mixed,
                &f_p,
                &b_p,
                &k.dt_bias,
                &k.a_log,
                delta,
                k.heads,
                k.head_dim,
                k.norm_eps,
                k.gate_floor,
            )
        },
    ]);

    let g = ops::linear::matmul(x, &k.gate);
    let o = ops::elemwise::rmsnorm_gated_by(&core, &g, &k.o_norm, k.heads, k.o_norm_eps);
    ops::linear::matmul(&o, &k.o_proj)
}
