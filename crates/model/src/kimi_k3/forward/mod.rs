//! kimi_k3's forward, declared.
//!
//! Transcribed from `kimi_k3_forward.cpp`. A hybrid like qwen3_5, but
//! neither half is qwen3_5's:
//!
//! * **KDA, not GDN.** Kimi Delta Attention decays per KEY CHANNEL where
//!   qwen3_5's gated delta net decays per head — one scalar becomes a
//!   vector, and the gate is produced through a rank-`head_dim`
//!   bottleneck (`kda_f_a` then `kda_f_b`) rather than directly.
//!
//! * **MLA with no rope.** The full-attention half is MLA, and the text
//!   states no rope on it. That is not an omission: `kimi_k3_forward.cpp`
//!   says so in its own words ("there is deliberately no
//!   `kernels::rope::rope_bf16` here"), because this family's positional
//!   information rides the KDA layers instead.
//!
//! * **SITU, not swiglu.** Every MLP activation here is `kernels::mlp::situ`
//!   / its chunked twin.
//!
//! * **An attention-residual BLOCK that spans layers.**
//!   `kernels::attn::attn_res_blend` blends a block's accumulated prefix
//!   back in every `attn_res_block` layers. It is the one statement here
//!   whose operands are not this layer's — and the reason the block size
//!   is a fact rather than a loop bound.

pub mod facts;

use self::facts::KimiK3Facts;
use model_dsl::axes::{Bf16Ax, DtypeAxis, KvAxis, Mxfp4Ax, NativeKv};
use model_dsl::{self as dsl, MatW, NormW, WeightRepr, matmul};
use model_ir::trace::{FireClass, ForwardPlan, NormVariant};

/// SITU's gate blend. No committed config states `activation_situ_beta`,
/// and the hand-written pass read it with a default of `1.0`
/// (`config.cpp`: `optional<float>(j, "activation_situ_beta", 1.f)`) —
/// the normalizer's default, the same honest reading dsv4's theta gives.
const SITU_BETA: f32 = 1.0;
/// The optional tanh cap on the up half; `0` is the kernel's own word for
/// "no cap", and the hand-written default (`activation_situ_linear_beta`,
/// default `0.f`).
const SITU_LINEAR_BETA: f32 = 0.0;

struct K3LayerW {
    attn_norm: NormW,
    mlp_norm: NormW,
    // MLA
    q_a_proj: MatW,
    q_a_norm: NormW,
    q_b_proj: MatW,
    kv_a_proj: MatW,
    o_proj: MatW,
    // KDA
    kda_q: MatW,
    kda_k: MatW,
    kda_v: MatW,
    kda_f_a: MatW,
    kda_f_b: MatW,
    kda_b: MatW,
    kda_g: MatW,
    kda_o: MatW,
    kda_o_norm: NormW,
    // MLP / MoE
    dense_gate: MatW,
    dense_up: MatW,
    dense_down: MatW,
    router: MatW,
    shared_gate: MatW,
    shared_up: MatW,
    shared_down: MatW,
}

impl K3LayerW {
    fn new(l: u32, f: &KimiK3Facts, norm_eps: f32, repr: WeightRepr) -> Self {
        let w = |name: &str| format!("layer.{l}.{name}");
        let m = |name: &str, width: u32| MatW {
            name: w(name),
            width,
            layer: Some(l),
            repr,
        };
        let n = |name: &str| NormW {
            name: w(name),
            variant: NormVariant::Plain,
            per_head: None,
            layer: Some(l),
            eps: norm_eps,
        };
        let a = &f.attn;
        let k = &f.kda;
        Self {
            attn_norm: n("attn_norm"),
            mlp_norm: n("mlp_norm"),
            q_a_proj: m("q_a_proj", a.q_lora_rank),
            q_a_norm: n("q_a_norm"),
            q_b_proj: m("q_b_proj", a.q_b_width()),
            kv_a_proj: m("kv_a_proj_with_mqa", a.kv_a_width()),
            o_proj: m("o_proj", a.hidden),
            kda_q: m("kda_q_proj", k.width()),
            kda_k: m("kda_k_proj", k.width()),
            kda_v: m("kda_v_proj", k.width()),
            kda_f_a: m("kda_f_a_proj", k.value_head_dim),
            kda_f_b: m("kda_f_b_proj", k.width()),
            kda_b: m("kda_b_proj", k.value_heads),
            kda_g: m("kda_g_proj", k.width()),
            kda_o: m("kda_o_proj", f.hidden),
            kda_o_norm: n("kda_o_norm"),
            dense_gate: m("dense_gate_proj", f.dense_intermediate),
            dense_up: m("dense_up_proj", f.dense_intermediate),
            dense_down: m("dense_down_proj", f.hidden),
            router: m("router", f.moe.num_experts),
            shared_gate: m("shared_expert.gate", f.moe.shared_intermediate),
            shared_up: m("shared_expert.up", f.moe.shared_intermediate),
            shared_down: m("shared_expert.down", f.hidden),
        }
    }
}

/// kimi_k3's CUDA text for one fire class.
///
/// **Both shaped classes, one body.** Like `kimi_k2`, the attention is MLA's
/// single planned dispatch — `attn::plan_attention_mla_bf16` takes a
/// `qo_indptr`, so a decode is the case where each request contributes one
/// query row rather than a different kernel — and the KDA half is a
/// recurrence over whatever rows the fire brought. Nothing here reads the
/// class except the trace's name.
pub fn kimi_k3_cuda<W1: DtypeAxis, W2: DtypeAxis, A: DtypeAxis, K: KvAxis>(
    facts: &KimiK3Facts,
    class: FireClass,
    norm_eps: f32,
) -> ForwardPlan {
    // The activation axis is DECLARED but pinned until the launch wrappers
    // take a dtype: every statement below states BF16 outs, so a point
    // instantiated at another A would lie. The pin is a compile refusal,
    // not a comment. Same for K: MLA's latent cache and the KDA state are
    // stated bf16 and nothing here forks on the scheme yet. And W2 is
    // pinned to the marlin repr because the routed leg below states
    // `quant::mxfp4_moe_*` by name — a bf16 expert bank has no leg in
    // this text.
    const {
        assert!(matches!(A::DTYPE, model_ir::trace::DType::BF16));
        assert!(K::NATIVE_BF16);
        assert!(matches!(W2::REPR, WeightRepr::Mxfp4Marlin));
    }
    // The SKU joins the family's FIRST segment ('.'-separated segment two
    // stays the backend, which `Backend::of_family` parses).
    let family = format!(
        "kimi_k3-{}-{}-{}.cuda.{}",
        W1::NAME,
        W2::NAME,
        K::NAME,
        class.suffix()
    );
    let a = facts.attn.clone();
    let kd = facts.kda.clone();
    dsl::trace_named(&family, |t| {
        let mut y = dsl::embedded_prologue(t, facts.hidden, facts.vocab);

        for l in 0..facts.layers {
            let w = K3LayerW::new(l, facts, norm_eps, W1::REPR);
            // The attention-residual block: every `attn_res_block` layers
            // the accumulated prefix blends back in. Layer 0 opens the
            // first block, so there is nothing to blend yet.
            if facts.attn_res_block > 0 && l > 0 && l % facts.attn_res_block == 0 {
                y = dsl::cuda::attn_res_blend(
                    &y,
                    &y,
                    &format!("layer.{l}.attn_res_norm"),
                    &format!("layer.{l}.attn_res_proj"),
                    facts.hidden,
                );
            }
            let x = dsl::cuda::rmsnorm(&y, &w.attn_norm);

            if facts.is_full_attn(l) {
                let (q_b, kv_a, _q_a_n) = dsl::mla_latents(
                    &x,
                    None,
                    &w.q_a_proj,
                    &w.q_a_norm,
                    &w.q_b_proj,
                    &w.kv_a_proj,
                    a.q_lora_rank,
                );
                // The split pair, NOT the fused prepare: this family's
                // MLA carries no rope, and `kernels::attn::mla_prepare_bf16` does
                // the rope as part of what it fuses.
                let (kv_c, k_pe) = dsl::cuda::kimi_split_kv_a_norm(
                    &kv_a,
                    &format!("layer.{l}.kv_a_norm"),
                    a.kv_lora_rank,
                    a.qk_rope_head_dim,
                    norm_eps,
                );
                let (q_nope, q_pe) = dsl::cuda::kimi_split_q_b(
                    &q_b,
                    a.heads,
                    a.qk_nope_head_dim,
                    a.qk_rope_head_dim,
                );
                dsl::cuda::write_mla_to_pages(&kv_c, &k_pe, l);
                let attn_v = dsl::mla_absorbed_attention(
                    &q_nope,
                    &q_pe,
                    &format!("layer.{l}.kv_b_proj"),
                    l,
                    dsl::MlaWidths {
                        heads: a.heads,
                        kv_lora_rank: a.kv_lora_rank,
                        qk_nope_head_dim: a.qk_nope_head_dim,
                        v_head_dim: a.v_head_dim,
                    },
                );
                // The MLA output gate is NOT stated, and refuses rather
                // than approximating. `SigmoidGateMul` is a SEMANTIC op
                // and requires its operands to share a Shape; MLA's
                // absorb produces `[Tokens, heads, v_head_dim]` while a
                // projection produces `[Tokens, width]`. Same elements
                // per token, different rank, and the DSL has no rank-3
                // projection and no `cuda::` twin for
                // `kernels::mlp::sigmoid_gate_inplace_bf16` (it is an EMITTED
                // symbol, produced by the lowering from the semantic op,
                // so it is deliberately not a `kernel!` row either).
                //
                // Stating it anyway would need one of those two holes
                // filled, and filling them to make one text pass is how a
                // declaration starts describing the DSL instead of the
                // model. gpt-oss's `attn.qv` sets the precedent: a
                // binding a text cannot state honestly is refused, out
                // loud, at the boundary.
                //
                // The WEIGHT for it is gone from `K3LayerW` too, and its
                // removal is the same point one level down: it was never
                // read — nothing below this assert could reach it — and
                // it named `layer.{}.mla_g_proj`, which no checkpoint
                // publishes. The manifest's gate is
                // `layer.{}.self_attn.g_proj`. A bound name no arm reads
                // and no publication carries is a claim that this text
                // has a gate, standing directly above the refusal that
                // says it does not.
                assert!(
                    !a.output_gate,
                    "kimi_k3: `mla_output_gate` is not stated yet — the \
                     semantic SigmoidGateMul wants equal Shapes and MLA's \
                     absorb is rank-3. See this arm's comment."
                );
                y += dsl::attention_landing(&attn_v, &w.o_proj, l);
            } else {
                // ── KDA ──────────────────────────────────────────────
                let q = matmul(&x, &w.kda_q);
                let k = matmul(&x, &w.kda_k);
                let v = matmul(&x, &w.kda_v);
                // One depthwise causal conv per projection, each with a
                // fused SiLU — the batched (slot-indirected) form, which
                // is the one a decode fire takes.
                let rs = dsl::Rs::at(t, l);
                let conv = |x: &dsl::Val, name: &str| {
                    dsl::cuda::gdn_conv_update_batched(
                        x,
                        &dsl::ConvW {
                            name: format!("layer.{l}.{name}"),
                            bias: None,
                            kernel: kd.conv_kernel,
                            layer: l,
                        },
                        &rs,
                        // KDA's conv walks ONE projection at a time, so its
                        // channel count is that projection's width and not a
                        // fused `conv_dim`. The recurrence numbers are unused
                        // by this launch and stated zero rather than guessed.
                        dsl::cuda::GdnShape {
                            k_heads: 0,
                            v_heads: kd.value_heads,
                            k_dim: 0,
                            v_dim: 0,
                            conv_dim: kd.value_heads,
                            conv_k: kd.conv_kernel,
                        },
                    )
                };
                let q = conv(&q, "kda_q_conv");
                let k = conv(&k, "kda_k_conv");
                let v = conv(&v, "kda_v_conv");
                // The decay gate: a rank-`head_dim` bottleneck, then out
                // to every head channel — which is what makes the decay
                // per-CHANNEL rather than qwen3_5's per-head scalar.
                let f_a = matmul(&x, &w.kda_f_a);
                let f_b = matmul(&f_a, &w.kda_f_b);
                let b = matmul(&x, &w.kda_b);
                let (gate, beta) = dsl::cuda::kda_gate_beta(
                    &f_b,
                    &b,
                    &format!("layer.{l}.kda_a_log"),
                    &format!("layer.{l}.kda_dt_bias"),
                    kd.value_heads,
                    kd.value_head_dim,
                );
                let q = dsl::cuda::l2norm_scale_to_f32(&q, kd.width(), kd.norm_eps());
                let k = dsl::cuda::l2norm_scale_to_f32(&k, kd.width(), kd.norm_eps());
                let v = dsl::cuda::bf16_to_f32(&v, kd.width());
                let core = dsl::cuda::kda_recurrent_step(
                    &q,
                    &k,
                    &v,
                    &gate,
                    &beta,
                    l,
                    kd.value_heads,
                    kd.value_head_dim,
                );
                let g = matmul(&x, &w.kda_g);
                let o = dsl::cuda::kda_o_norm_gated(
                    &core,
                    &g,
                    &w.kda_o_norm.name,
                    kd.width(),
                    kd.value_heads,
                    kd.value_head_dim,
                    kd.norm_eps(),
                );
                dsl::seam(o.trace(), &dsl::seam::ATTN_OUT, &[&o], Some(l));
                y += matmul(&o, &w.kda_o);
            }

            // ── MLP / MoE ────────────────────────────────────────────
            let m = dsl::cuda::rmsnorm(&y, &w.mlp_norm);
            if !facts.is_moe_layer(l) {
                y += dsl::dense_gated_mlp(
                    &m,
                    &w.dense_gate,
                    &w.dense_up,
                    &w.dense_down,
                    facts.dense_intermediate,
                    dsl::GatedAct::Situ {
                        beta: SITU_BETA,
                        linear_beta: SITU_LINEAR_BETA,
                    },
                );
                continue;
            }

            let logits = matmul(&m, &w.router);
            let (experts, weights) = dsl::cuda::topk_sigmoid(
                &logits,
                facts.moe.top_k,
                facts.moe.norm_topk_prob,
                facts.moe.routed_scaling,
            );
            let (gate_up, _up) = dsl::cuda::mxfp4_moe_gate_up_decode(
                &m,
                &experts,
                &MatW {
                    name: format!("layer.{l}.expert.{{e}}.gate_up"),
                    width: 2 * facts.moe.moe_intermediate,
                    layer: Some(l),
                    repr: W2::REPR,
                },
                facts.moe.top_k,
                facts.moe.moe_intermediate,
                // The split-output form leaves the fused epilogue's clamp
                // unread (the kernel applies `glu_limit`/`glu_alpha` only
                // when it writes the fused fp16 activation, and this
                // statement takes gate and up apart) — and kimi states no
                // GLU clamp: SITU's own cap is `linear_beta` below.
                0.0,
                0.0,
            );
            let act = dsl::cuda::situ(
                &gate_up,
                facts.moe.moe_intermediate,
                SITU_BETA,
                SITU_LINEAR_BETA,
            );
            let route_out = dsl::cuda::mxfp4_moe_down_decode(
                &act,
                &experts,
                &MatW {
                    name: format!("layer.{l}.expert.{{e}}.down"),
                    width: facts.hidden,
                    layer: Some(l),
                    repr: W2::REPR,
                },
                facts.moe.top_k,
                facts.hidden,
            );
            let routed = dsl::cuda::weighted_sum(&weights, &route_out, facts.hidden, None);

            let moe_out = if facts.moe.shared_intermediate > 0 {
                let sgate = matmul(&m, &w.shared_gate);
                let sup = matmul(&m, &w.shared_up);
                let sact = dsl::cuda::situ_pair(
                    &sgate,
                    &sup,
                    facts.moe.shared_intermediate,
                    SITU_BETA,
                    SITU_LINEAR_BETA,
                );
                let shared = matmul(&sact, &w.shared_down);
                dsl::cuda::residual_add(&routed, &shared, facts.hidden)
            } else {
                routed
            };
            y = dsl::cuda::residual_add(&y, &moe_out, facts.hidden);
        }

        dsl::logits_epilogue(
            t,
            &y,
            NormVariant::Plain,
            false,
            facts.vocab,
            None,
            norm_eps,
        );
    })
}

/// One shipping SKU: its name and the monomorphized trace it instantiates.
pub type TraceFn = fn(&KimiK3Facts, FireClass, f32) -> ForwardPlan;

/// The shipped SKU's axes — the family's ONE spelling of its point.
/// [`CATALOG`], `project::trace`, `project::manifest`'s repr claims and
/// `contract::author_kimi_k3`'s MXFP4 pin all derive from these; a
/// second SKU adds a row, not a respelling.
pub type ShippedW1 = Bf16Ax;
/// The routed expert banks' axis: MXFP4, the release's packing.
pub type ShippedW2 = Mxfp4Ax;
/// The activation axis of the shipped point (pinned BF16 in the text).
pub type ShippedA = Bf16Ax;
/// The KV axis of the shipped point.
pub type ShippedKv = NativeKv;

/// The family's catalogue — every SKU this build ships, enumerated. The
/// routed experts are MXFP4 (the module doc's `mxfp4_moe_*` leg), so W2's
/// shipped value is the marlin axis. The coverage test
/// (`model/tests/catalogue_coverage.rs`) traces each row at both fire
/// classes; `TraceBuilder::finish`'s `check_plan` then refuses a row whose
/// statements reach a routine point that does not exist.
pub const CATALOG: &[(&str, TraceFn)] = model_dsl::catalogue![
    (
        "kimi_k3-bf16-mxfp4-kv-bf16",
        kimi_k3_cuda::<ShippedW1, ShippedW2, ShippedA, ShippedKv>,
    ),
];
