use model_dsl::axes::{Dtype, KvDtype};
use model_dsl::{kernels, Facts, merge, seam, Classify, ForwardHybrid, HybridInput, HybridSpec, Request, Value};

use super::model::{Attn, Gdn, Head, Mixer, Mlp, Model};

#[derive(Facts)]
pub struct Facts {
    pub qo_one: bool,
}

impl Classify for Facts {
    fn of(r: &Request) -> Self {
        Self { qo_one: r.query_len() == 1 }
    }
}

impl<W1: Dtype, K: KvDtype, const TP: usize> ForwardHybrid for Model<W1, K, TP> {
    type Facts = Facts;

    fn caches(&self) -> HybridSpec {
        let mut c = HybridSpec::new();
        for (l, w) in self.layers.iter().enumerate() {
            match &w.mixer {
                Mixer::Attn(a) => {
                    c.kv(format!("kv.{l}"), [2, a.kv_heads as u64 * a.head_dim as u64]);
                }
                Mixer::Gdn(g) => {
                    let conv_ch = (2 * g.k_heads * g.k_dim + g.v_heads * g.v_dim) as u64;
                    c.state(format!("conv.{l}"), [conv_ch, (g.conv_kernel - 1) as u64]);
                    c.state(
                        format!("delta.{l}"),
                        [g.v_heads as u64, g.k_dim as u64, g.v_dim as u64],
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

        for (l, w) in m.layers.iter().enumerate() {
            let l = l as u32;
            let x = kernels::rmsnorm(&y, &w.mixer_norm);
            let o = match &w.mixer {
                Mixer::Attn(a) => attn_mixer(&x, &inputs, a, l),
                Mixer::Gdn(g) => gdn_mixer(&x, &inputs, g, l),
            };
            let o = if TP > 1 { kernels::all_reduce(&o) } else { o };
            y = kernels::add(&y, &o);

            let x = kernels::rmsnorm(&y, &w.mlp_norm);
            let f = match &w.mlp {
                Mlp::Dense { gate_up, down, inter } => {
                    kernels::matmul(&kernels::swiglu(&kernels::matmul(&x, gate_up), *inter), down)
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
                    let routes = kernels::topk_softmax(&kernels::matmul(&x, router), *experts, *top_k);
                    let hidden = kernels::swiglu(&kernels::matmul_select(&x, gate_up, &routes), *inter);
                    let routed = kernels::weighted_sum(&kernels::matmul_select(&hidden, down, &routes), &routes);
                    let shared = kernels::matmul(
                        &kernels::swiglu(&kernels::matmul(&x, shared_gate_up), *shared_inter),
                        shared_down,
                    );
                    kernels::sigmoid_gate_add(&routed, &shared, &kernels::matmul(&x, shared_gate))
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

fn attn_mixer<W1: Dtype>(x: &Value, inputs: &HybridInput<Facts>, a: &Attn<W1>, l: u32) -> Value {
    let pages = inputs.kv(&a.kv);
    let d = a.head_dim;
    let (q, gate) = kernels::split_q_gate(&kernels::matmul(x, &a.qg_proj), d);
    let k = kernels::matmul(x, &a.k_proj);
    let v = kernels::matmul(x, &a.v_proj);
    seam::at(seam::ATTN_QV, (&q, &v), l);
    let q = kernels::rmsnorm_per_head(&q, &a.q_norm);
    let k = kernels::rmsnorm_per_head(&k, &a.k_norm);
    let (q, k) = kernels::rope_partial(&q, &k, a.rotary_dim, d, a.theta, &inputs.positions());
    kernels::kv_append(&k, &v, &pages);
    seam::at(seam::ATTN_Q, (&q,), l);
    let (dq, p) = q.split(&Facts::qo_one());
    let o = merge![
        kernels::attention_decode(&dq, &pages, None, d, a.sm_scale),
        kernels::attention_prefill(&kernels::query_windows(&p), &pages, None, d, a.kv_heads, a.sm_scale),
    ];
    seam::at(seam::ATTN_OUT, (&o,), l);
    kernels::attention_landing(&kernels::sigmoid_gate_mul(&o, &gate), &a.o_proj, l)
}

fn gdn_mixer<W1: Dtype>(x: &Value, inputs: &HybridInput<Facts>, g: &Gdn<W1>, l: u32) -> Value {
    let conv_state = inputs.state(&g.conv_state);
    let delta_state = inputs.state(&g.delta_state);
    let qkvz = kernels::matmul(x, &g.in_qkvz);
    let ba = kernels::matmul(x, &g.in_ba);
    seam::at(seam::RECURRENT, (&qkvz,), l);
    let width = 2 * g.k_heads * g.k_dim + g.v_heads * g.v_dim;
    let one = Facts::qo_one();
    let (qkvz_d, qkvz_p) = qkvz.split(&one);
    let (ba_d, ba_p) = ba.split(&one);
    let o = merge![
        {
            let (qkv, z) = kernels::split_rows(&qkvz_d, width);
            let qkv = kernels::causal_conv1d(&qkv, &g.conv, &conv_state);
            let gates = kernels::gdn_prep(&ba_d, &g.dt_bias, &g.a_log);
            kernels::gated_delta(&qkv, &z, &gates, &delta_state, g.k_heads, g.v_heads, g.k_dim, g.v_dim)
        },
        {
            let (qkv, z) = kernels::split_rows(&qkvz_p, width);
            let qkv = kernels::causal_conv1d_chunked(&kernels::query_windows(&qkv), &g.conv, &conv_state);
            let gates = kernels::gdn_prep(&ba_p, &g.dt_bias, &g.a_log);
            kernels::gated_delta_chunked(&kernels::query_windows(&qkv), &z, &gates, &delta_state, g.k_heads, g.v_heads, g.k_dim, g.v_dim)
        },
    ];
    kernels::matmul(&kernels::rmsnorm_gated(&o, &g.norm), &g.out_proj)
}
