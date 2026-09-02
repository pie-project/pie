use model_dsl::{
    Classify, Dtype, ForwardHybrid, HybridSpec, Input, Predicate, Request, Value, ops, seam,
};

use super::model::{Kda, Mixer, Mla, Mlp, Model};

pub struct Facts {
    pub qo_one: bool,
    pub has_adapter: bool,
}

impl Facts {
    pub fn qo_one() -> Predicate {
        Predicate::fact(0)
    }

    /// True for rows whose request routed to a registered adapter; a fire with no such rows costs nothing.
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
            match &w.mixer {
                Mixer::Mla(a) => {
                    c.kv(
                        kv,
                        a.kv.clone(),
                        [self.kv_lora_rank as u64, a.qk_rope_head_dim as u64],
                    );
                }

                Mixer::Kda(k) => {
                    let width = (k.heads * k.head_dim) as u64;
                    c.state(k.conv_state.clone(), [k.conv_kernel as u64, 3 * width], Dtype::Bf16);
                    c.state(
                        k.delta_state.clone(),
                        [k.heads as u64, k.head_dim as u64, k.head_dim as u64],
                        Dtype::Bf16,
                    );
                }
            }
        }
        c
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let m = self;

        // plan[0] is the decode schedule, plan[1] the prefill schedule; each is built from its own arm.
        let (input_d, input_p) = inputs.split(&Facts::qo_one());
        let plan = [
            ops::attn::mla_plan(&input_d, m.mla_heads, m.kv_lora_rank),
            ops::attn::mla_plan(&input_p, m.mla_heads, m.kv_lora_rank),
        ];
        let ids = inputs.tokens();
        let mut y = ops::layout::embed(&ids, &m.embed, m.vocab);
        let mut blocks: Vec<Value> = Vec::new();

        let routes = inputs.adapter_routes();
        for (_, w) in inputs.walk_layers(&m.layers) {
            if let Some(b) = &w.res_blend {
                y = ops::elemwise::res_blend(&y, &blocks, &b.norm, b.norm_eps, &b.proj);
                blocks.push(y.clone());
            }

            let x = ops::elemwise::rmsnorm(&y, &w.mixer_norm, w.mixer_norm_eps);
            let o = match &w.mixer {
                Mixer::Mla(a) => mla_mixer(&x, &inputs, &plan, m, a),
                Mixer::Kda(k) => kda_mixer(&x, &inputs, k),
            };
            let o = if m.tp > 1 {
                ops::collective::all_reduce(&o)
            } else {
                o
            };
            // Must run after all_reduce: correcting a tp-split partial product would sum the correction tp times.
            let o = {
                let (adapted, _) = o.split(&Facts::has_adapter());
                let (px, _) = x.split(&Facts::has_adapter());
                ops::linear::lora_correct(&px, &w.lora_a, &w.lora_b, &routes, &adapted)
            };
            y = ops::elemwise::residual_add(&o, &y);

            let x = ops::elemwise::rmsnorm(&y, &w.mlp_norm, w.mlp_norm_eps);
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
                    gate_up,
                    down,
                    shared,
                    experts,
                    top_k,
                    routed_scaling,
                    inter,
                    beta,
                    up_cap,
                } => {
                    let (routes, weights) = ops::linear::moe_topk_sigmoid(
                        &ops::linear::matmul(&x, router),
                        *experts,
                        *top_k,
                        false,
                        *routed_scaling,
                    );
                    let hidden = ops::linear::moe_matmul_select_quant(&x, gate_up, &routes, *top_k);
                    let act = ops::linear::mlp_situ(&hidden, *inter, *beta, *up_cap);
                    let routed = ops::linear::moe_weighted_sum(
                        &ops::linear::moe_matmul_select_quant(&act, down, &routes, *top_k),
                        &weights,
                    );
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
            let f = if m.tp > 1 {
                ops::collective::all_reduce(&f)
            } else {
                f
            };
            y = ops::elemwise::residual_add(&f, &y);
        }

        let x = ops::elemwise::rmsnorm(&y, &m.final_norm, m.final_norm_eps);
        ops::linear::lm_head(&x, &m.head)
    }
}

fn mla_mixer(x: &Value, inputs: &Input<Facts>, plan: &[Value; 2], m: &Model, a: &Mla) -> Value {
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

    // Must split by the same predicate `forward` used to build plan[0]/plan[1].
    let one = Facts::qo_one();
    let (dq, p) = q.split(&one);
    let (dpe, ppe) = q_pe.split(&one);
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
                &mixed, &f_d, &b_d, &k.dt_bias, &k.a_log, delta, k.heads, k.head_dim, k.norm_eps, 0.0,
            )
        },
        {
            let mixed = ops::attn::ssm_causal_conv1d_chunked(&qkv_p, &k.conv, conv, k.conv_kernel);
            ops::attn::ssm_kda_chunked(
                &mixed, &f_p, &b_p, &k.dt_bias, &k.a_log, delta, k.heads, k.head_dim, k.norm_eps, 0.0,
            )
        },
    ]);

    let g = ops::linear::matmul(x, &k.gate);
    let o = ops::elemwise::rmsnorm_gated_by(&core, &g, &k.o_norm, k.heads, k.o_norm_eps);
    ops::linear::matmul(&o, &k.o_proj)
}
