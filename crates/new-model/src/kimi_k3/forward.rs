//! The Kimi K3 forward pass over the typed IR — the old forward minus the
//! machinery the design killed (design §10): the one MLA plan is built up
//! front and shared visibly across the full-attention layers (§6), kv-append
//! geometry is a declared input fetched once per forward (§7), raggedness is
//! ambient so the prefill/chunked arms lose their `query_windows` plumbing
//! (§5), and tensor parallelism is plain control flow on `m.tp` (§9,
//! decision #18). The K3 particulars — residual blending every `res_block`
//! layers, the situ activation, the gated MLA output — carry over 1:1.

use new_model_dsl::{
    Classify, Facts, ForwardHybrid, HybridSpec, Input, Request, Value, kernels, merge, seam,
};
use new_model_ir::GeomKind;

use super::model::{Kda, Mixer, Mla, Mlp, Model};

#[derive(Facts)]
pub struct Facts {
    pub qo_one: bool,
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
        for (l, w) in self.layers.iter().enumerate() {
            match &w.mixer {
                Mixer::Mla(a) => {
                    c.kv(
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
        let attn = AttnShared::of(m, &inputs);
        let ids = inputs.tokens();
        let mut y = kernels::layout::embed(&ids, &m.embed, m.vocab);
        let mut blocks: Vec<Value> = Vec::new();

        for (l, w) in inputs.layers(&m.layers) {
            if let Some(b) = &w.res_blend {
                blocks.push(y.clone());
                y = kernels::norm::res_blend(&y, &blocks, &b.norm, &b.proj);
            }

            let x = kernels::norm::rmsnorm(&y, &w.mixer_norm.weight, w.mixer_norm.eps);
            let o = match &w.mixer {
                Mixer::Mla(a) => mla_mixer(&x, &inputs, &attn, a, l),
                Mixer::Kda(k) => kda_mixer(&x, &inputs, k),
            };
            let o = if m.tp > 1 {
                kernels::dist::all_reduce(&o)
            } else {
                o
            };
            y = kernels::norm::residual_add(&o, &y);

            let x = kernels::norm::rmsnorm(&y, &w.mlp_norm.weight, w.mlp_norm.eps);
            let f = match &w.mlp {
                Mlp::Dense {
                    gate_up,
                    down,
                    inter,
                    beta,
                    up_cap,
                } => kernels::gemm::matmul(
                    &kernels::mlp::situ(
                        &kernels::gemm::matmul(&x, gate_up),
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
                    let (routes, weights) = kernels::moe::topk_sigmoid(
                        &kernels::gemm::matmul(&x, router),
                        *experts,
                        *top_k,
                        false,
                        *routed_scaling,
                    );
                    let hidden = kernels::moe::matmul_select(&x, gate_up, &routes, *top_k);
                    let act = kernels::mlp::situ(&hidden, *inter, *beta, *up_cap);
                    let routed = kernels::moe::weighted_sum(
                        &kernels::moe::matmul_select(&act, down, &routes, *top_k),
                        &weights,
                    );
                    match shared {
                        None => routed,
                        Some(s) => {
                            let act = kernels::mlp::situ(
                                &kernels::gemm::matmul(&x, &s.gate_up),
                                s.inter,
                                *beta,
                                *up_cap,
                            );
                            kernels::norm::residual_add(
                                &kernels::gemm::matmul(&act, &s.down),
                                &routed,
                            )
                        }
                    }
                }
            };
            let f = if m.tp > 1 {
                kernels::dist::all_reduce(&f)
            } else {
                f
            };
            y = kernels::norm::residual_add(&f, &y);
        }

        let fin = &m.final_norm;
        let x = kernels::norm::rmsnorm(&y, &fin.weight, fin.eps);
        kernels::gemm::lm_head(&x, &m.head)
    }
}

/// What every full-attention layer shares, stated once per forward: the one
/// MLA plan decode and prefill both take (§6), the page indices
/// `mla::kv_append` writes through (§7), and the positions that geometry
/// keys on. The first kv space's geometry stands in for all of them — the
/// fire lays every layer's kv pages out identically.
struct AttnShared {
    positions: Value,
    kv_indices: Value,
    plan: Value,
}

impl AttnShared {
    fn of(m: &Model, inputs: &Input<Facts>) -> AttnShared {
        let kv = m
            .layers
            .iter()
            .find_map(|w| match &w.mixer {
                Mixer::Mla(a) => Some(inputs.cache_index(a.kv.name())),
                Mixer::Kda(_) => None,
            })
            .expect("kimi k3 interleaves full-attention layers");
        let positions = inputs.positions();
        AttnShared {
            plan: kernels::mla::plan(positions.rec(), kv),
            kv_indices: inputs.geometry(kv, GeomKind::Indices),
            positions,
        }
    }
}

fn mla_mixer(x: &Value, inputs: &Input<Facts>, s: &AttnShared, a: &Mla, layer: u32) -> Value {
    let pages = inputs.kv(a.kv.name());
    let q_a = kernels::gemm::matmul(x, &a.q_a_proj);
    let q_a = kernels::norm::rmsnorm(&q_a, &a.q_a_norm.weight, a.q_a_norm.eps);
    let (kv_c, k_pe) = kernels::mla::latents(
        &kernels::gemm::matmul(x, &a.kv_a_proj),
        &a.kv_a_norm,
        a.kv_lora_rank,
    );
    let (q_nope, q_pe) = kernels::mla::split_q_b(
        &kernels::gemm::matmul(&q_a, &a.q_b_proj),
        a.heads,
        a.qk_nope_head_dim,
        a.qk_rope_head_dim,
    );
    kernels::mla::kv_append(&kv_c, &k_pe, pages, &s.kv_indices, &s.positions);

    let q = kernels::mla::absorb_q(
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
        kernels::mla::attention_decode(
            &dq,
            &s.plan,
            &dpe,
            pages,
            a.heads,
            a.kv_lora_rank,
            a.sm_scale,
        ),
        kernels::mla::attention_prefill(
            &p,
            &s.plan,
            &ppe,
            pages,
            a.heads,
            a.kv_lora_rank,
            a.sm_scale,
        ),
    ];
    let o = kernels::mla::absorb_out(
        &latent,
        &a.kv_b_proj,
        a.heads,
        a.kv_lora_rank,
        a.v_head_dim,
        a.qk_nope_head_dim,
    );
    let o = match &a.gate {
        None => o,
        Some(g) => kernels::gate::sigmoid_mul(&o, &kernels::gemm::matmul(x, g)),
    };
    kernels::gemm::attention_landing(&o, &a.o_proj, layer)
}

fn kda_mixer(x: &Value, inputs: &Input<Facts>, k: &Kda) -> Value {
    let conv = inputs.state(k.conv_state.name());
    let delta = inputs.state(k.delta_state.name());
    let qkv = kernels::gemm::matmul(x, &k.qkv);
    let f = kernels::gemm::matmul(&kernels::gemm::matmul(x, &k.f_a), &k.f_b);
    let b = kernels::gemm::matmul(x, &k.b);
    seam::at(seam::RECURRENT, (&qkv,));

    let one = Facts::qo_one();
    let (qkv_d, qkv_p) = qkv.split(&one);
    let (f_d, f_p) = f.split(&one);
    let (b_d, b_p) = b.split(&one);
    let core = merge![
        {
            let mixed = kernels::ssm::causal_conv1d(&qkv_d, &k.conv, conv, k.conv_kernel);
            kernels::ssm::kda_step(
                &mixed, &f_d, &b_d, &k.dt_bias, &k.a_log, delta, k.heads, k.head_dim, k.norm_eps,
            )
        },
        {
            let mixed = kernels::ssm::causal_conv1d(&qkv_p, &k.conv, conv, k.conv_kernel);
            kernels::ssm::kda_chunked(
                &mixed, &f_p, &b_p, &k.dt_bias, &k.a_log, delta, k.heads, k.head_dim, k.norm_eps,
            )
        },
    ];

    let g = kernels::gemm::matmul(x, &k.gate);
    let o = kernels::norm::rmsnorm_gated_by(&core, &g, &k.o_norm.weight, k.heads, k.o_norm.eps);
    seam::at(seam::ATTN_OUT, (&o,));
    kernels::gemm::matmul(&o, &k.o_proj)
}
