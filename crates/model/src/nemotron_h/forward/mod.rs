//! nemotron_h's forward, declared.
//!
//! Transcribed from `nemotron_h_forward.cpp`. The third hybrid, and the
//! only one whose schedule is a LIST rather than an interval — and the
//! only one with THREE layer kinds, because a bare `mlp` layer has no
//! mixer at all. An interval cannot say that, which is why
//! `layer_types` is carried verbatim.
//!
//! Mamba is the new vocabulary. GDN and KDA carry a decaying state over
//! key/value pairs; mamba's selective scan carries an explicit
//! `[heads, head_dim, state_size]` state and reads a per-token `dt` that
//! decides how much of it to keep. `prepare_mamba_dt_da` is its own
//! statement for that reason — the softplus-and-scale on `dt` and the
//! decay `A` it pairs with are computed once per token, before the scan,
//! not inside it.
//!
//! ReLU² where the other families have swiglu, and a GATED norm on the
//! mamba output (`zamba_rmsnorm_gated`) rather than a plain one.

pub mod facts;

use self::facts::{NemotronHFacts, NemotronLayerKind};
use model_dsl::{self as dsl, MatW, NormW, WeightRepr, matmul};
use model_ir::trace::{FireClass, ForwardPlan, NormVariant};

struct NhLayerW {
    norm: NormW,
    // mamba
    in_proj: MatW,
    out_proj: MatW,
    gate_norm: NormW,
    // attention
    q_proj: MatW,
    k_proj: MatW,
    v_proj: MatW,
    o_proj: MatW,
    // mlp / moe
    up_proj: MatW,
    down_proj: MatW,
    router: MatW,
    shared_up: MatW,
    shared_down: MatW,
}

impl NhLayerW {
    fn new(l: u32, f: &NemotronHFacts) -> Self {
        let w = |name: &str| format!("layer.{l}.{name}");
        let m = |name: &str, width: u32| MatW {
            name: w(name),
            width,
            layer: Some(l),
            repr: WeightRepr::Bf16,
        };
        let n = |name: &str| NormW {
            name: w(name),
            variant: NormVariant::Plain,
            per_head: None,
            layer: Some(l),
        };
        Self {
            norm: n("norm"),
            in_proj: m("mamba_in_proj", f.mamba.in_proj_width()),
            out_proj: m("mamba_out_proj", f.hidden),
            gate_norm: n("mamba_norm"),
            q_proj: m("q_proj", f.attn.q_width()),
            k_proj: m("k_proj", f.attn.kv_width()),
            v_proj: m("v_proj", f.attn.kv_width()),
            o_proj: m("o_proj", f.hidden),
            // ReLU², so ONE projection up and one down — there is no
            // gate half to pair with.
            up_proj: m("up_proj", f.moe.moe_intermediate),
            down_proj: m("down_proj", f.hidden),
            router: m("router", f.moe.num_experts),
            shared_up: m("shared_expert.up", f.moe.shared_intermediate),
            shared_down: m("shared_expert.down", f.hidden),
        }
    }
}

/// nemotron_h's CUDA text for one fire class.
pub fn nemotron_h_cuda(facts: &NemotronHFacts, class: FireClass) -> ForwardPlan {
    // DECODE AND PREFILL, and the difference is one call. This family
    // served Decode only and PANICKED on anything else — not refused,
    // panicked, on the first prefill a serving deployment sends. The
    // class-dependent sites in this text number exactly one, the
    // attention op, which `dsl::cuda::attention_for` now holds.
    let family = format!("nemotron_h.cuda.{}", class.suffix());
    let mb = facts.mamba;
    let _at = facts.attn;
    dsl::trace_named(&family, |t| {
        let mut y = dsl::embedded_prologue(t, facts.hidden);

        for l in 0..facts.layers() {
            // THIS LAYER's sliding window, `-1` for none — a
            // load-time fact the dispatch statements carry, where four
            // executors used to re-derive it per launch.
            let window_left = model_ir::facts::window_left_at(facts.window_left, l);
            let w = NhLayerW::new(l, facts);
            let x = dsl::cuda::rmsnorm(&y, &w.norm);

            match facts.kind(l) {
                NemotronLayerKind::Mamba => {
                    // One projection, three things: `[z | conv_dim | dt]`.
                    let packed = matmul(&x, &w.in_proj);
                    let (z, conv_in, dt_raw) = dsl::cuda::nemotron_mamba_split(
                        &packed,
                        mb.intermediate(),
                        mb.conv_dim(),
                        mb.num_heads,
                    );
                    let rs = dsl::Rs::at(t, l);
                    let conv_out = dsl::cuda::gdn_conv_update_batched(
                        &conv_in,
                        &dsl::ConvW {
                            name: format!("layer.{l}.mamba_conv"),
                            bias: Some(format!("layer.{l}.mamba_conv_bias")),
                            kernel: mb.conv_kernel,
                            layer: l,
                        },
                        &rs,
                        // mamba's conv carries the fused `[z | conv_dim | dt]`
                        // block's middle third. The GDN recurrence numbers are
                        // not this family's and are stated zero.
                        dsl::cuda::GdnShape {
                            k_heads: 0,
                            v_heads: mb.num_heads,
                            k_dim: 0,
                            v_dim: mb.head_dim,
                            conv_dim: mb.conv_dim(),
                            conv_k: mb.conv_kernel,
                        },
                    );
                    // `dt` and the decay `A` are per-token and computed
                    // ONCE, before the scan — a separate statement
                    // because the scan reads them, it does not make them.
                    let (a_par, d_par, dt_bias) = dsl::cuda::nemotron_prepare_mamba_params(
                        t,
                        l,
                        &format!("layer.{l}.mamba_a_log"),
                        &format!("layer.{l}.mamba_d"),
                        &format!("layer.{l}.mamba_dt_bias"),
                        mb.num_heads,
                    );
                    let (dt, da) = dsl::cuda::nemotron_prepare_mamba_dt_da(
                        &dt_raw,
                        &a_par,
                        &dt_bias,
                        mb.num_heads,
                    );
                    let core = dsl::cuda::nemotron_mamba_ssm(
                        &conv_out,
                        &dt,
                        &dt_raw,
                        &a_par,
                        &d_par,
                        &dt_bias,
                        &da,
                        l,
                        mb.intermediate(),
                    );
                    dsl::seam(core.trace(), &dsl::seam::ATTN_OUT, &[&core], Some(l));
                    // A GATED norm, not a plain one: `z` is the gate the
                    // split produced and the norm applies it.
                    let o = dsl::cuda::zamba_rmsnorm_gated(
                        &core,
                        &z,
                        &w.gate_norm.name,
                        mb.intermediate(),
                        mb.n_groups,
                    );
                    y += matmul(&o, &w.out_proj);
                }
                NemotronLayerKind::Attention => {
                    let q = matmul(&x, &w.q_proj);
                    let k = matmul(&x, &w.k_proj);
                    let v = matmul(&x, &w.v_proj);
                    let (q, k) = dsl::cuda::rope(&q, &k, facts.attn.heads, facts.attn.kv_heads, facts.attn.head_dim);
                    let kv = dsl::Kv::at(t, l);
                    dsl::cuda::write_kv_to_pages(&k, &v, &kv);
                    dsl::seam(q.trace(), &dsl::seam::ATTN_Q, &[&q], Some(l));
                    let o = dsl::cuda::attention_for(class, &q, &kv, window_left, facts.attn.head_dim, 0.0, 0.0)
                        .expect("a plain attention statement produces its value");
                    dsl::seam(o.trace(), &dsl::seam::ATTN_OUT, &[&o], Some(l));
                    y += matmul(&o, &w.o_proj);
                }
                NemotronLayerKind::Mlp => {
                    // No mixer. The layer IS its MLP, and the norm above
                    // is the only thing before it.
                    let up = matmul(&x, &w.up_proj);
                    let act = dsl::cuda::relu2(&up, facts.moe.moe_intermediate);
                    y += matmul(&act, &w.down_proj);
                    continue;
                }
            }

            // ── MoE, on the mixer layers ─────────────────────────────
            // A DENSE stack stops here. Every published Nemotron-H is
            // dense (`num_experts` is null in all three configs), and
            // this block ran unconditionally: a dense row would trace a
            // router of width 0, a top-k of 0 and an expert GEMV over a
            // bank of no experts, which is not a shape error anywhere —
            // it is a fire that reads a zero-length weight and returns
            // zeros. The mixer layers of a dense stack are their mixer
            // and their residual, and nothing else.
            if facts.moe.num_experts == 0 {
                continue;
            }
            let m = dsl::cuda::rmsnorm(&y, &w.norm);
            // FP32 logits — `nemotron_h_forward.cpp` fires
            // `act_x_wt_bf16_out_fp32` for the router because
            // `topk_sigmoid_bias_fp32` consumes fp32. The first
            // transcription stated a plain (bf16) matmul here; the
            // executor port caught the dtype seam before it ever ran.
            let logits = dsl::cuda::gemm_out_fp32(&m, &w.router.name, facts.moe.num_experts);
            let (experts, weights) = dsl::cuda::topk_sigmoid_bias(
                &logits,
                &format!("layer.{l}.router_bias"),
                facts.moe.top_k,
            );
            let gate_up = dsl::cuda::moe_gate_up_gemv(
                &m,
                &MatW {
                    name: format!("layer.{l}.expert.{{e}}.up"),
                    width: facts.moe.moe_intermediate,
                    layer: Some(l),
                    repr: WeightRepr::Bf16,
                },
                &experts,
                facts.moe.top_k,
            );
            let act = dsl::cuda::relu2(&gate_up, facts.moe.moe_intermediate);
            let route_out = dsl::cuda::moe_down_gemv(
                &act,
                &MatW {
                    name: format!("layer.{l}.expert.{{e}}.down"),
                    width: facts.hidden,
                    layer: Some(l),
                    repr: WeightRepr::Bf16,
                },
                &experts,
                facts.moe.top_k,
            );
            let routed = dsl::cuda::weighted_sum(&weights, &route_out, facts.hidden, None);

            let moe_out = if facts.moe.shared_intermediate > 0 {
                let sup = matmul(&m, &w.shared_up);
                let sact = dsl::cuda::relu2(&sup, facts.moe.shared_intermediate);
                let shared = matmul(&sact, &w.shared_down);
                dsl::cuda::residual_add(&routed, &shared, facts.hidden)
            } else {
                routed
            };
            y = dsl::cuda::residual_add(&y, &moe_out, facts.hidden);
        }

        dsl::logits_epilogue(t, &y, NormVariant::Plain, false, facts.vocab, false);
    })
}
