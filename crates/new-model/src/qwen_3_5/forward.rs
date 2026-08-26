//! The Qwen 3.5 forward pass over the typed IR — the old forward minus the
//! machinery the design killed (design §10): attention plans are built once
//! up front and shared visibly across layers (§6), kv-append geometry is a
//! declared input fetched once per forward (§7), raggedness is ambient so
//! the prefill/chunked arms lose their `query_windows` plumbing (§5), and
//! tensor parallelism is plain control flow on `m.tp` (§9, decision #18).

use new_model_dsl::{
    Classify, Facts, ForwardHybrid, HybridSpec, Input, Request, Value, kernels, merge, seam,
};
use new_model_ir::GeomKind;

use super::model::{Attn, Gdn, Head, Mixer, Mlp, Model};

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
                Mixer::Attn(a) => {
                    c.kv(
                        format!("kv.{l}"),
                        [2, a.kv_heads as u64 * a.head_dim as u64],
                    );
                }

                Mixer::Gdn(g) => {
                    let conv_ch = (2 * g.k_heads * g.k_dim + g.v_heads * g.v_dim) as u64;
                    c.state(format!("conv.{l}"), [g.conv_kernel as u64, conv_ch]);
                    c.state(
                        format!("delta.{l}"),
                        [g.v_heads as u64, g.k_dim as u64, g.v_dim as u64],
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

        for (l, w) in inputs.layers(&m.layers) {
            let x = kernels::norm::rmsnorm_plus_one(&y, &w.mixer_norm.weight, w.mixer_norm.eps);
            let o = match &w.mixer {
                Mixer::Attn(a) => attn_mixer(&x, &inputs, &attn, a, l),
                Mixer::Gdn(g) => gdn_mixer(&x, &inputs, g),
            };
            let o = if m.tp > 1 {
                kernels::dist::all_reduce(&o)
            } else {
                o
            };
            y = kernels::norm::residual_add(&o, &y);

            let x = kernels::norm::rmsnorm_plus_one(&y, &w.mlp_norm.weight, w.mlp_norm.eps);
            let f = match &w.mlp {
                Mlp::Dense {
                    gate_up,
                    down,
                    inter,
                } => kernels::gemm::matmul(
                    &kernels::mlp::swiglu(&kernels::gemm::matmul(&x, gate_up), *inter),
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
                    let (routes, weights) = kernels::moe::topk_softmax(
                        &kernels::gemm::matmul(&x, router),
                        *experts,
                        *top_k,
                    );
                    let hidden = kernels::mlp::swiglu(
                        &kernels::moe::matmul_select(&x, gate_up, &routes, *top_k),
                        *inter,
                    );
                    let routed = kernels::moe::weighted_sum(
                        &kernels::moe::matmul_select(&hidden, down, &routes, *top_k),
                        &weights,
                    );
                    let shared = kernels::gemm::matmul(
                        &kernels::mlp::swiglu(
                            &kernels::gemm::matmul(&x, shared_gate_up),
                            *shared_inter,
                        ),
                        shared_down,
                    );
                    kernels::moe::sigmoid_gate_add(
                        &routed,
                        &shared,
                        &kernels::gemm::matmul(&x, shared_gate),
                    )
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
        let x = kernels::norm::rmsnorm_plus_one(&y, &fin.weight, fin.eps);
        match &m.head {
            Head::Tied => kernels::gemm::lm_head(&x, &m.embed),
            Head::Bank(bank) => kernels::gemm::lm_head(&x, bank),
        }
    }
}

/// What every attention layer shares, stated once per forward: the decode and
/// prefill plans (§6), the page indices `kv_append` writes through (§7), and
/// the positions rope reads. The first kv space's geometry stands in for all
/// of them — the fire lays every layer's kv pages out identically.
struct AttnShared {
    positions: Value,
    kv_indices: Value,
    plan_d: Value,
    plan_p: Value,
}

impl AttnShared {
    fn of(m: &Model, inputs: &Input<Facts>) -> AttnShared {
        let kv = m
            .layers
            .iter()
            .find_map(|w| match &w.mixer {
                Mixer::Attn(a) => Some(inputs.cache_index(a.kv.name())),
                Mixer::Gdn(_) => None,
            })
            .expect("qwen 3.5 interleaves attention layers");
        let positions = inputs.positions();
        AttnShared {
            plan_d: kernels::attention::plan_decode(positions.rec(), kv),
            plan_p: kernels::attention::plan_prefill(positions.rec(), kv),
            kv_indices: inputs.geometry(kv, GeomKind::Indices),
            positions,
        }
    }
}

fn attn_mixer(x: &Value, inputs: &Input<Facts>, s: &AttnShared, a: &Attn, layer: u32) -> Value {
    let pages = inputs.kv(a.kv.name());
    let d = a.head_dim;
    let (q, gate) = kernels::layout::split_q_gate(&kernels::gemm::matmul(x, &a.qg_proj), d);
    let k = kernels::gemm::matmul(x, &a.k_proj);
    let v = kernels::gemm::matmul(x, &a.v_proj);
    seam::at(seam::ATTN_QV, (&q, &v));
    let q = kernels::norm::rmsnorm_per_head_plus_one(&q, &a.q_norm.weight, d, a.q_norm.eps);
    let k = kernels::norm::rmsnorm_per_head_plus_one(&k, &a.k_norm.weight, d, a.k_norm.eps);
    let (q, k) = kernels::rope::partial(&q, &k, &s.positions, a.rotary_dim, d, a.theta);
    kernels::attention::kv_append(&k, &v, pages, &s.kv_indices, &s.positions);
    seam::at(seam::ATTN_Q, (&q,));
    let (dq, p) = q.split(&Facts::qo_one());
    let o = merge![
        kernels::attention::decode(&dq, &s.plan_d, pages, None, d, a.sm_scale),
        kernels::attention::prefill(&p, &s.plan_p, pages, None, d, a.kv_heads, a.sm_scale),
    ];
    seam::at(seam::ATTN_OUT, (&o,));
    kernels::gemm::attention_landing(&kernels::gate::sigmoid_mul(&o, &gate), &a.o_proj, layer)
}

fn gdn_mixer(x: &Value, inputs: &Input<Facts>, g: &Gdn) -> Value {
    let conv_state = inputs.state(g.conv_state.name());
    let delta_state = inputs.state(g.delta_state.name());
    let qkvz = kernels::gemm::matmul(x, &g.in_qkvz);
    let ba = kernels::gemm::matmul(x, &g.in_ba);
    seam::at(seam::RECURRENT, (&qkvz,));
    let width = 2 * g.k_heads * g.k_dim + g.v_heads * g.v_dim;
    let one = Facts::qo_one();
    let (qkvz_d, qkvz_p) = qkvz.split(&one);
    let (ba_d, ba_p) = ba.split(&one);
    let (core_d, z_d) = {
        let (qkv, z) = kernels::layout::split_rows(&qkvz_d, width);
        let qkv = kernels::ssm::causal_conv1d(&qkv, &g.conv, conv_state, g.conv_kernel);
        let gates = kernels::ssm::gdn_prep(&ba_d, &g.dt_bias, &g.a_log);
        let core = kernels::ssm::gated_delta(
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
        let (qkv, z) = kernels::layout::split_rows(&qkvz_p, width);
        let qkv = kernels::ssm::causal_conv1d_chunked(&qkv, &g.conv, conv_state, g.conv_kernel);
        let gates = kernels::ssm::gdn_prep(&ba_p, &g.dt_bias, &g.a_log);
        let core = kernels::ssm::gated_delta_chunked(
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
    let o = merge![core_d, core_p];
    let z = merge![z_d, z_p];

    let o = kernels::norm::rmsnorm_gated(&o, &z, &g.norm.weight, g.v_dim, g.norm.eps);
    kernels::gemm::matmul(&o, &g.out_proj)
}
