use model_dsl::axes::{Dtype, KvDtype};
use model_dsl::{
    Classify, Facts, ForwardHybrid, HybridInput, HybridSpec, Request, Value, kernels, merge, seam,
};

use super::model::{Attn, Gdn, Head, Mixer, Mlp, Model};

// EVERY NORM OF THIS FAMILY IS AN OFFSET BANK. Qwen3.5 stores its RMSNorm
// scales the way Gemma does -- the kernel applies `1 + weight`, not `weight`
// -- and the one exception is the GDN gated out-norm, whose bank is plain.
// The legacy deployment says the same thing in its own vocabulary
// (`model-legacy/src/qwen_3_5/spec.rs`: `norm_variant: NormVariant::Gemma`
// on every fact struct, `NormVariant::Plain` on the gate norm alone), and a
// live census of the legacy fire confirms it: `norm::rmsnorm_gemma` for
// `attn_norm`, `mlp_norm`, `q_norm`, `k_norm` and `final_norm`,
// `norm::rmsnorm_gated_fp32_in` for the gate norm. Stating the plain point
// here computed a different model and nothing refused.

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

impl<W1: Dtype, K: KvDtype, const TP: usize> ForwardHybrid for Model<W1, K, TP> {
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
                // THE CONV SLAB IS `[K, C]`, WHICH IS NOT THE ROLLING
                // WINDOW. `causal_conv1d_update_batched` holds the last
                // `conv_kernel` input rows oldest-first, convolves over rows
                // `1..K-1` plus the incoming column, shifts every row down
                // one and lands the new column at row `K - 1`
                // (`kernels-cuda/kernels/ssm/causal_conv1d.cuh:396-411`). So
                // `K - 1` rows are LIVE between fires and row 0 is where the
                // shift's tail goes -- written every step, read by nothing --
                // and the rectangle the kernel indexes is `K` rows of `C`
                // channels. A declaration states the rectangle: this row used
                // to say `[C, K - 1]`, the live window, which is neither the
                // extent the pool must allocate nor the axis order the kernel
                // walks (`state[k * C + c]`, so `C` is the fast axis).
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

    fn forward(&self, inputs: HybridInput<Facts>) -> Value {
        let m = self;
        let ids = inputs.token_ids();
        let mut y = kernels::layout::embed(&ids, &m.embed, m.vocab);

        for (_, w) in inputs.layers(&m.layers) {
            let x = kernels::norm::rmsnorm_plus_one(&y, &w.mixer_norm.weight, w.mixer_norm.eps);
            let o = match &w.mixer {
                Mixer::Attn(a) => attn_mixer(&x, &inputs, a),
                Mixer::Gdn(g) => gdn_mixer(&x, &inputs, g),
            };
            let o = kernels::dist::reduce::<TP>(o);
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
                        &kernels::moe::matmul_select(&x, gate_up, &routes),
                        *inter,
                    );
                    let routed = kernels::moe::weighted_sum(
                        &kernels::moe::matmul_select(&hidden, down, &routes),
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
            let f = kernels::dist::reduce::<TP>(f);
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

fn attn_mixer<W1: Dtype>(x: &Value, inputs: &HybridInput<Facts>, a: &Attn<W1>) -> Value {
    let pages = inputs.kv(&a.kv);
    let d = a.head_dim;
    let (q, gate) = kernels::layout::split_q_gate(&kernels::gemm::matmul(x, &a.qg_proj), d);
    let k = kernels::gemm::matmul(x, &a.k_proj);
    let v = kernels::gemm::matmul(x, &a.v_proj);
    seam::at(seam::ATTN_QV, (&q, &v));
    let q = kernels::norm::rmsnorm_per_head_plus_one(&q, &a.q_norm.weight, d, a.q_norm.eps);
    let k = kernels::norm::rmsnorm_per_head_plus_one(&k, &a.k_norm.weight, d, a.k_norm.eps);
    let (q, k) = kernels::rope::partial(&q, &k, &inputs.positions(), a.rotary_dim, d, a.theta);
    kernels::attention::kv_append(&k, &v, &pages);
    seam::at(seam::ATTN_Q, (&q,));
    let (dq, p) = q.split(&Facts::qo_one());
    let o = merge![
        kernels::attention::decode(&dq, &pages, None, d, a.sm_scale),
        kernels::attention::prefill(
            &kernels::query_windows(&p),
            &pages,
            None,
            d,
            a.kv_heads,
            a.sm_scale
        ),
    ];
    seam::at(seam::ATTN_OUT, (&o,));
    kernels::gemm::attention_landing(&kernels::gate::sigmoid_mul(&o, &gate), &a.o_proj)
}

fn gdn_mixer<W1: Dtype>(x: &Value, inputs: &HybridInput<Facts>, g: &Gdn<W1>) -> Value {
    let conv_state = inputs.state(&g.conv_state);
    let delta_state = inputs.state(&g.delta_state);
    let qkvz = kernels::gemm::matmul(x, &g.in_qkvz);
    let ba = kernels::gemm::matmul(x, &g.in_ba);
    seam::at(seam::RECURRENT, (&qkvz,));
    let width = 2 * g.k_heads * g.k_dim + g.v_heads * g.v_dim;
    let one = Facts::qo_one();
    let (qkvz_d, qkvz_p) = qkvz.split(&one);
    let (ba_d, ba_p) = ba.split(&one);
    let (core_d, z_d) = {
        let (qkv, z) = kernels::layout::split_rows(&qkvz_d, width);
        let qkv = kernels::ssm::causal_conv1d(&qkv, &g.conv, &conv_state, g.conv_kernel);
        let gates = kernels::ssm::gdn_prep(&ba_d, &g.dt_bias, &g.a_log);
        let core = kernels::ssm::gated_delta(
            &qkv,
            &z,
            &gates,
            &delta_state,
            g.k_heads,
            g.v_heads,
            g.k_dim,
            g.v_dim,
        );
        (core, z)
    };
    let (core_p, z_p) = {
        let (qkv, z) = kernels::layout::split_rows(&qkvz_p, width);
        let qkv = kernels::ssm::causal_conv1d_chunked(
            &kernels::query_windows(&qkv),
            &g.conv,
            &conv_state,
            g.conv_kernel,
        );
        let gates = kernels::ssm::gdn_prep(&ba_p, &g.dt_bias, &g.a_log);
        let core = kernels::ssm::gated_delta_chunked(
            &kernels::query_windows(&qkv),
            &z,
            &gates,
            &delta_state,
            g.k_heads,
            g.v_heads,
            g.k_dim,
            g.v_dim,
        );
        (core, z)
    };
    let o = merge![core_d, core_p];
    let z = merge![z_d, z_p];
    // `g.v_dim`, because the gate norm is PER HEAD: its weight is one row of
    // `value_head_dim` floats and the mean of the square is taken across a
    // single head's channels, not across the mixer's whole
    // `value_heads * value_head_dim` output.
    let o = kernels::norm::rmsnorm_gated(&o, &z, &g.norm.weight, g.v_dim, g.norm.eps);
    kernels::gemm::matmul(&o, &g.out_proj)
}
