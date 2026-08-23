use model_dsl::axes::{Dtype, KvDtype};
use model_dsl::{kernels, Facts, merge, seam, Classify, ForwardHybrid, HybridInput, HybridSpec, Request, Value};

use super::model::{Head, Kda, Mixer, Mla, Mlp, Model};

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

impl<W1: Dtype, W2: Dtype, K: KvDtype, const TP: usize> ForwardHybrid for Model<W1, W2, K, TP> {
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
                    c.state(format!("conv.{l}"), [3 * width, (k.conv_kernel - 1) as u64]);
                    c.state(
                        format!("delta.{l}"),
                        [k.heads as u64, k.head_dim as u64, k.head_dim as u64],
                    );
                }
            }
        }
        c
    }

    fn forward(&self, inputs: HybridInput<Facts>) -> Value {
        let m = self;
        let ids = inputs.token_ids();
        let mut y = kernels::embed(&ids, &m.embed);
        let mut blocks: Vec<Value> = Vec::new();

        for (l, w) in m.layers.iter().enumerate() {
            let l = l as u32;

            if let Some(b) = &w.res_blend {
                blocks.push(y.clone());
                y = kernels::res_blend(&y, &blocks, &b.norm, &b.proj);
            }

            let x = kernels::rmsnorm(&y, &w.mixer_norm);
            let o = match &w.mixer {
                Mixer::Mla(a) => mla_mixer(&x, &inputs, a, l),
                Mixer::Kda(k) => kda_mixer(&x, &inputs, k, l),
            };
            let o = if TP > 1 { kernels::all_reduce(&o) } else { o };
            y = kernels::add(&y, &o);

            let x = kernels::rmsnorm(&y, &w.mlp_norm);
            let f = match &w.mlp {
                Mlp::Dense { gate_up, down, inter, beta, up_cap } => kernels::matmul(
                    &kernels::situ(&kernels::matmul(&x, gate_up), *inter, *beta, *up_cap),
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
                    let routes = kernels::topk_sigmoid(
                        &kernels::matmul(&x, router),
                        *experts,
                        *top_k,
                        false,
                        *routed_scaling,
                    );
                    let hidden = kernels::matmul_select(&x, gate_up, &routes);
                    let act = kernels::situ(&hidden, *inter, *beta, *up_cap);
                    let routed =
                        kernels::weighted_sum(&kernels::matmul_select(&act, down, &routes), &routes);
                    match shared {
                        None => routed,
                        Some(s) => {
                            let act = kernels::situ(
                                &kernels::matmul(&x, &s.gate_up),
                                s.inter,
                                *beta,
                                *up_cap,
                            );
                            kernels::add(&routed, &kernels::matmul(&act, &s.down))
                        }
                    }
                }
            };
            let f = if TP > 1 { kernels::all_reduce(&f) } else { f };
            y = kernels::add(&y, &f);
        }

        let x = kernels::rmsnorm(&y, &m.final_norm);
        let logits = match &m.head {
            Head::Tied => kernels::lm_head(&x, &m.embed),
            Head::Bank(bank) => kernels::lm_head(&x, bank),
        };

        logits
    }
}

fn mla_mixer<W1: Dtype>(x: &Value, inputs: &HybridInput<Facts>, a: &Mla<W1>, l: u32) -> Value {
    let pages = inputs.kv(&a.kv);
    let q_a = kernels::rmsnorm(&kernels::matmul(x, &a.q_a_proj), &a.q_a_norm);
    let (kv_c, k_pe) = kernels::mla_latents(
        &kernels::matmul(x, &a.kv_a_proj),
        &a.kv_a_norm,
        a.kv_lora_rank,
    );
    let (q_nope, q_pe) = kernels::split_q_b(
        &kernels::matmul(&q_a, &a.q_b_proj),
        a.heads,
        a.qk_nope_head_dim,
        a.qk_rope_head_dim,
    );
    kernels::kv_append_mla(&kv_c, &k_pe, &pages);

    let q = kernels::mla_absorb_q(
        &q_nope,
        &a.kv_b_proj,
        a.heads,
        a.kv_lora_rank,
        a.qk_nope_head_dim,
    );
    seam::at(seam::ATTN_Q, (&q,), l);

    let one = Facts::qo_one();
    let (dq, p) = q.split(&one);
    let (dpe, ppe) = q_pe.split(&one);
    let latent = merge![
        kernels::mla_attention_decode(&dq, &dpe, &pages, a.heads, a.kv_lora_rank, a.sm_scale),
        kernels::mla_attention_prefill(
            &kernels::query_windows(&p),
            &kernels::query_windows(&ppe),
            &pages,
            a.heads,
            a.kv_lora_rank,
            a.sm_scale,
        ),
    ];
    let o = kernels::mla_absorb_out(&latent, &a.kv_b_proj, a.heads, a.kv_lora_rank, a.v_head_dim);
    let o = match &a.gate {
        None => o,
        Some(g) => kernels::sigmoid_gate_mul(&o, &kernels::matmul(x, g)),
    };
    kernels::attention_landing(&o, &a.o_proj, l)
}

fn kda_mixer<W1: Dtype>(x: &Value, inputs: &HybridInput<Facts>, k: &Kda<W1>, l: u32) -> Value {
    let conv = inputs.state(&k.conv_state);
    let delta = inputs.state(&k.delta_state);
    let qkv = kernels::matmul(x, &k.qkv);
    let f = kernels::matmul(&kernels::matmul(x, &k.f_a), &k.f_b);
    let b = kernels::matmul(x, &k.b);
    seam::at(seam::RECURRENT, (&qkv,), l);

    let one = Facts::qo_one();
    let (qkv_d, qkv_p) = qkv.split(&one);
    let (f_d, f_p) = f.split(&one);
    let (b_d, b_p) = b.split(&one);
    let core = merge![
        {
            let mixed = kernels::causal_conv1d(&qkv_d, &k.conv, &conv);
            kernels::kda_step(&mixed, &f_d, &b_d, &k.dt_bias, &k.a_log, &delta, k.heads, k.head_dim, k.norm_eps)
        },
        {
            let mixed = kernels::causal_conv1d(&qkv_p, &k.conv, &conv);
            kernels::kda_chunked(
                &kernels::query_windows(&mixed), &f_p, &b_p, &k.dt_bias, &k.a_log, &delta,
                k.heads, k.head_dim, k.norm_eps,
            )
        },
    ];

    let g = kernels::matmul(x, &k.gate);
    let o = kernels::rmsnorm_gated_by(&core, &g, &k.o_norm);
    seam::at(seam::ATTN_OUT, (&o,), l);
    kernels::matmul(&o, &k.o_proj)
}
