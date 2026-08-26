//! The Kimi K3 forward pass over the typed IR — the old forward minus the
//! machinery the design killed (design §10): the one MLA plan is built up
//! front and shared visibly across the full-attention layers (§6), kv-append
//! geometry is a declared input fetched where it is used (§7), raggedness is
//! ambient so the prefill/chunked arms lose their `query_windows` plumbing
//! (§5), and tensor parallelism is plain control flow on `m.tp` (§9,
//! decision #18). The K3 particulars — residual blending every `res_block`
//! layers, the situ activation, the gated MLA output — carry over 1:1.

use model_dsl::{
    Classify, ForwardHybrid, GeomKind, HybridSpec, Input, Request, Value, kernels, merge, seam,
};

use super::model::{Kda, Mixer, Mla, Mlp, Model};

model_dsl::facts! {
    pub struct Facts { qo_one }
}

impl Classify for Facts {
    fn of(r: &Request) -> Self {
        Self {
            qo_one: r.query_len() == 1,
        }
    }
}

impl ForwardHybrid for Model {
    type Facts = Facts;

    fn caches(&self) -> HybridSpec {
        let mut c = HybridSpec::new();
        let kv = c.kv_space(self.kv);
        for (l, w) in self.layers.iter().enumerate() {
            match &w.mixer {
                Mixer::Mla(a) => {
                    c.kv(
                        kv,
                        format!("kv.{l}"),
                        [1, (a.kv_lora_rank + a.qk_rope_head_dim) as u64],
                    );
                }

                Mixer::Kda(k) => {
                    let width = (k.heads * k.head_dim) as u64;
                    c.state(format!("conv.{l}"), [k.conv_kernel as u64, 3 * width]);
                    c.state(
                        format!("delta.{l}"),
                        [k.heads as u64, k.head_dim as u64, k.head_dim as u64],
                    );
                }
            }
        }
        c
    }

    fn forward(&self, inputs: Input<Facts>) -> Value {
        let m = self;
        // The one MLA plan decode and prefill both take, built once and
        // shared visibly by every full-attention layer (§6).
        let positions = inputs.positions();
        let plan = kernels::attn::mla_plan(positions.rec(), inputs.kv_space());
        let ids = inputs.tokens();
        let mut y = kernels::layout::embed(&ids, &m.embed, m.vocab);
        let mut blocks: Vec<Value> = Vec::new();

        for (l, w) in inputs.layers(&m.layers) {
            if let Some(b) = &w.res_blend {
                blocks.push(y.clone());
                y = kernels::elemwise::res_blend(&y, &blocks, &b.norm, b.norm_eps, &b.proj);
            }

            let x = kernels::elemwise::rmsnorm(&y, &w.mixer_norm, w.mixer_norm_eps);
            let o = match &w.mixer {
                Mixer::Mla(a) => mla_mixer(&x, &inputs, &plan, a, l),
                Mixer::Kda(k) => kda_mixer(&x, &inputs, k),
            };
            let o = if m.tp > 1 {
                kernels::collective::all_reduce(&o)
            } else {
                o
            };
            y = kernels::elemwise::residual_add(&o, &y);

            let x = kernels::elemwise::rmsnorm(&y, &w.mlp_norm, w.mlp_norm_eps);
            let f = match &w.mlp {
                Mlp::Dense {
                    gate_up,
                    down,
                    inter,
                    beta,
                    up_cap,
                } => kernels::linear::matmul(
                    &kernels::linear::mlp_situ(
                        &kernels::linear::matmul(&x, gate_up),
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
                    let (routes, weights) = kernels::linear::moe_topk_sigmoid(
                        &kernels::linear::matmul(&x, router),
                        *experts,
                        *top_k,
                        false,
                        *routed_scaling,
                    );
                    let hidden = kernels::linear::moe_matmul_select(&x, gate_up, &routes, *top_k);
                    let act = kernels::linear::mlp_situ(&hidden, *inter, *beta, *up_cap);
                    let routed = kernels::linear::moe_weighted_sum(
                        &kernels::linear::moe_matmul_select(&act, down, &routes, *top_k),
                        &weights,
                    );
                    match shared {
                        None => routed,
                        Some(s) => {
                            let act = kernels::linear::mlp_situ(
                                &kernels::linear::matmul(&x, &s.gate_up),
                                s.inter,
                                *beta,
                                *up_cap,
                            );
                            kernels::elemwise::residual_add(
                                &kernels::linear::matmul(&act, &s.down),
                                &routed,
                            )
                        }
                    }
                }
            };
            let f = if m.tp > 1 {
                kernels::collective::all_reduce(&f)
            } else {
                f
            };
            y = kernels::elemwise::residual_add(&f, &y);
        }

        let x = kernels::elemwise::rmsnorm(&y, &m.final_norm, m.final_norm_eps);
        kernels::linear::lm_head(&x, &m.head)
    }
}

fn mla_mixer(x: &Value, inputs: &Input<Facts>, plan: &Value, a: &Mla, layer: u32) -> Value {
    let pages = inputs.kv(&a.kv);
    let write_page = inputs.geometry(inputs.kv_space(), GeomKind::WritePage);
    let write_offset = inputs.geometry(inputs.kv_space(), GeomKind::WriteOffset);
    let q_a = kernels::linear::matmul(x, &a.q_a_proj);
    let q_a = kernels::elemwise::rmsnorm(&q_a, &a.q_a_norm, a.q_a_norm_eps);
    let (kv_c, k_pe) = kernels::attn::mla_latents(
        &kernels::linear::matmul(x, &a.kv_a_proj),
        &a.kv_a_norm,
        a.kv_a_norm_eps,
        a.kv_lora_rank,
    );
    let (q_nope, q_pe) = kernels::attn::mla_split_q_b(
        &kernels::linear::matmul(&q_a, &a.q_b_proj),
        a.heads,
        a.qk_nope_head_dim,
        a.qk_rope_head_dim,
    );
    kernels::attn::mla_kv_append(&kv_c, &k_pe, pages, &write_page, &write_offset);

    let q = kernels::attn::mla_absorb_q(
        &q_nope,
        &a.kv_b_proj,
        a.heads,
        a.kv_lora_rank,
        a.qk_nope_head_dim,
        a.v_head_dim,
    );
    seam::at(seam::ATTN_Q, (&q,));

    let one = Facts::qo_one();
    let (dq, p) = q.split(&one);
    let (dpe, ppe) = q_pe.split(&one);
    let latent = merge![
        kernels::attn::mla_decode(&dq, plan, &dpe, pages, a.heads, a.kv_lora_rank, a.sm_scale,),
        kernels::attn::mla_prefill(&p, plan, &ppe, pages, a.heads, a.kv_lora_rank, a.sm_scale,),
    ];
    let o = kernels::attn::mla_absorb_out(
        &latent,
        &a.kv_b_proj,
        a.heads,
        a.kv_lora_rank,
        a.v_head_dim,
        a.qk_nope_head_dim,
    );
    let o = match &a.gate {
        None => o,
        Some(g) => kernels::elemwise::gate_sigmoid_mul(&o, &kernels::linear::matmul(x, g)),
    };
    kernels::linear::attention_landing(&o, &a.o_proj, layer)
}

fn kda_mixer(x: &Value, inputs: &Input<Facts>, k: &Kda) -> Value {
    let conv = inputs.state(&k.conv_state);
    let delta = inputs.state(&k.delta_state);
    let qkv = kernels::linear::matmul(x, &k.qkv);
    let f = kernels::linear::matmul(&kernels::linear::matmul(x, &k.f_a), &k.f_b);
    let b = kernels::linear::matmul(x, &k.b);
    seam::at(seam::RECURRENT, (&qkv,));

    let one = Facts::qo_one();
    let (qkv_d, qkv_p) = qkv.split(&one);
    let (f_d, f_p) = f.split(&one);
    let (b_d, b_p) = b.split(&one);
    let core = merge![
        {
            let mixed = kernels::attn::ssm_causal_conv1d(&qkv_d, &k.conv, conv, k.conv_kernel);
            kernels::attn::ssm_kda_step(
                &mixed, &f_d, &b_d, &k.dt_bias, &k.a_log, delta, k.heads, k.head_dim, k.norm_eps,
            )
        },
        {
            let mixed = kernels::attn::ssm_causal_conv1d(&qkv_p, &k.conv, conv, k.conv_kernel);
            kernels::attn::ssm_kda_chunked(
                &mixed, &f_p, &b_p, &k.dt_bias, &k.a_log, delta, k.heads, k.head_dim, k.norm_eps,
            )
        },
    ];

    let g = kernels::linear::matmul(x, &k.gate);
    let o = kernels::elemwise::rmsnorm_gated_by(&core, &g, &k.o_norm, k.heads, k.o_norm_eps);
    seam::at(seam::ATTN_OUT, (&o,));
    kernels::linear::matmul(&o, &k.o_proj)
}
