//! kimi's forward, declared.
//!
//! Transcribed from `driver-cuda/csrc/src/model/kimi/kimi_forward.cpp`.
//! The second MLA family; what differs from [`crate::glm_5`] is worth
//! naming, because the shapes are otherwise the same statement:
//!
//! * **No DSA.** kimi's attention reads the whole context; there is no
//!   lightning indexer and no page mask.
//!
//! * **The latents may be ONE projection.** With `q_kv_a_fused` bound,
//!   `[q_lora | kv_lora | rope]` land row-major in one buffer, so the
//!   query half is normed with a STRIDED kernel rather than a plain one
//!   — neither latent is a contiguous block of the result. That is a
//!   BINDING fact, so it is a fact here and both readings are stated.
//!
//! * **The experts are WNA16.** The decode leg is
//!   `launch_wna16_{gate_up,down}_decode_bf16` over packed weights and
//!   scales, with `bf16_to_fp16` casts on the activation either side —
//!   the kernel reads fp16. Same rectangle shape as glm5's GEMV leg,
//!   different kernels, and the casts are real launches that a text
//!   omitting them would be wrong about.

pub mod facts;

use self::facts::{KimiCudaFacts, KimiFacts};
use model_dsl::axes::{Bf16Ax, DtypeAxis, KvAxis, NativeKv, Wna16Ax};
use model_dsl::{self as dsl, MatW, NormW, WeightRepr, matmul};
use model_ir::trace::{FireClass, ForwardPlan, NormVariant};

struct KimiLayerW {
    attn_norm: NormW,
    mlp_norm: NormW,
    q_kv_a: MatW,
    q_a_proj: MatW,
    kv_a_proj: MatW,
    q_a_norm: NormW,
    q_b_proj: MatW,
    o_proj: MatW,
    dense_gate: MatW,
    dense_up: MatW,
    dense_down: MatW,
    router: MatW,
    shared_gate: MatW,
    shared_up: MatW,
    shared_down: MatW,
}

impl KimiLayerW {
    fn new(l: u32, f: &KimiFacts, norm_eps: f32, repr: WeightRepr) -> Self {
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
        Self {
            attn_norm: n("attn_norm"),
            mlp_norm: n("mlp_norm"),
            q_kv_a: m("q_kv_a_fused", a.q_kv_a_width()),
            q_a_proj: m("q_a_proj", a.q_lora_rank),
            kv_a_proj: m("kv_a_proj_with_mqa", a.kv_a_width()),
            q_a_norm: n("q_a_norm"),
            q_b_proj: m("q_b_proj", a.q_b_width()),
            o_proj: m("o_proj", a.hidden),
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

/// kimi's CUDA text for one fire class.
///
/// **Both shaped classes, and the body is the same text for each.** MLA's
/// attention is one planned dispatch — `attn::plan_attention_mla_bf16` takes
/// a `qo_indptr` and a `causal` flag, so a decode is the special case where
/// every request contributes one query row, not a different kernel. Nothing
/// else here reads the class. So the class reaches only the trace's NAME,
/// which is what a lowering keys its cache by.
///
/// It used to `panic!` on anything but Decode. That was not a statement about
/// this text — it was the absence of one, and it made every prefill a failed
/// request.
pub fn kimi_cuda<W1: DtypeAxis, W2: DtypeAxis, A: DtypeAxis, K: KvAxis>(
    facts: &KimiFacts,
    cuda: &KimiCudaFacts,
    class: FireClass,
    norm_eps: f32,
) -> ForwardPlan {
    // The activation axis is DECLARED but pinned until the launch wrappers
    // take a dtype: every statement below states BF16 outs, so a point
    // instantiated at another A would lie. The pin is a compile refusal,
    // not a comment. Same for K: MLA's latent cache is stated bf16 and
    // nothing here forks on the scheme yet.
    const {
        assert!(matches!(A::DTYPE, model_ir::trace::DType::BF16));
        assert!(K::NATIVE_BF16);
    }
    // The W4A16 expert packing's quantisation group — W2's, and the ONE
    // number the wna16 statements read. The family's own value is 32:
    // `contract.rs` builds the scale slabs at `GROUP = 32` and the
    // committed contract golden pins `"group_size": 32` on every expert
    // tensor — which is what `Wna16Ax::REPR` states. A W2 that is not
    // per-group scaled has no wna16 leg to state, and the refusal is a
    // monomorphization error, not a fire's.
    let wna16_group = const {
        match W2::REPR {
            WeightRepr::Scaled { group, .. } => group,
            _ => panic!("kimi's routed experts are WNA16; W2 must be a Scaled axis"),
        }
    };
    // The SKU joins the family's FIRST segment ('.'-separated segment two
    // stays the backend, which `Backend::of_family` parses).
    let family = format!(
        "kimi-{}-{}-{}.cuda.{}",
        W1::NAME,
        W2::NAME,
        K::NAME,
        class.suffix()
    );
    let a = facts.attn.clone();
    dsl::trace_named(&family, |t| {
        let mut y = dsl::embedded_prologue(t, facts.hidden, facts.vocab);

        for l in 0..facts.layers {
            let w = KimiLayerW::new(l, facts, norm_eps, W1::REPR);
            let x = dsl::cuda::rmsnorm(&y, &w.attn_norm);

            // The two latents, fused or not. The FUSED arm norms the
            // query half in place with a pitch, which is a different
            // kernel and not a buffer detail — `kernels::norm::rmsnorm_strided_bf16`
            // reads a row stride the plain one has no parameter for.
            let (q_b, kv_a, _q_a_n) = dsl::mla_latents(
                &x,
                cuda.q_kv_a_fused.then_some(&w.q_kv_a),
                &w.q_a_proj,
                &w.q_a_norm,
                &w.q_b_proj,
                &w.kv_a_proj,
                a.q_lora_rank,
            );

            let (_kv_c, _k_pe, q_nope, q_pe) = dsl::cuda::mla_prepare(
                &kv_a,
                &q_b,
                a.heads,
                a.kv_lora_rank,
                a.qk_nope_head_dim,
                a.qk_rope_head_dim,
            );
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
            y += dsl::attention_landing(&attn_v, &w.o_proj, l);

            let m = dsl::cuda::rmsnorm(&y, &w.mlp_norm);
            if !facts.is_moe_layer(l) {
                y += dsl::dense_gated_mlp(
                    &m,
                    &w.dense_gate,
                    &w.dense_up,
                    &w.dense_down,
                    facts.dense_intermediate,
                    dsl::GatedAct::SwiGlu,
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

            // WNA16: the kernel reads fp16, so the cast either side is a
            // real launch. Omitting it would make the text claim a
            // dtype the deployment never has at that point.
            let m_fp16 = dsl::cuda::bf16_to_fp16(&m);
            let (gate, up) = dsl::cuda::wna16_gate_up_decode(
                &m_fp16,
                &experts,
                facts.moe.moe_intermediate,
                &format!("layer.{l}.experts"),
                wna16_group,
            );
            let act = dsl::cuda::swiglu_pair(&gate, &up, facts.moe.moe_intermediate);
            let act_fp16 = dsl::cuda::bf16_to_fp16(&act);
            let route_out = dsl::cuda::wna16_down_decode(
                &act_fp16,
                &experts,
                facts.hidden,
                &format!("layer.{l}.experts"),
                wna16_group,
            );
            let routed = dsl::cuda::weighted_sum(&weights, &route_out, facts.hidden, None);

            let moe_out = if facts.moe.shared_intermediate > 0 {
                let sgate = matmul(&m, &w.shared_gate);
                let sup = matmul(&m, &w.shared_up);
                let sact = dsl::cuda::swiglu_pair(&sgate, &sup, facts.moe.shared_intermediate);
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
pub type TraceFn = fn(&KimiFacts, &KimiCudaFacts, FireClass, f32) -> ForwardPlan;

/// The shipped SKU's axes — the family's ONE spelling of its point.
///
/// Every consumer derives from these aliases rather than restating them:
/// [`CATALOG`]'s turbofish, `project::trace`'s instantiation,
/// `project::manifest`'s repr claims, and `contract`'s expert-stack pass
/// (its group and its load check read `ShippedW2`). A second SKU adds a
/// second row to the table, not a second spelling of the first.
pub type ShippedW1 = Bf16Ax;
/// The routed expert banks' axis: WNA16, the release's packing.
pub type ShippedW2 = Wna16Ax;
/// The activation axis of the shipped point (pinned BF16 in the text).
pub type ShippedA = Bf16Ax;
/// The KV axis of the shipped point.
pub type ShippedKv = NativeKv;

/// The family's catalogue — every SKU this build ships, enumerated. ONE
/// row, and its expert axis is WNA16: the checkpoint ships the routed bank
/// both as bf16 `.weight` and as W4A16 `.weight_packed` (two ENCODINGS of
/// one model — `project.rs`'s manifest deliberately names neither), but
/// this text states exactly one routed leg, the wna16 GEMVs, so one point
/// is what ships. The coverage test (`model/tests/catalogue_coverage.rs`)
/// traces each row at both fire classes; `TraceBuilder::finish`'s
/// `check_plan` then refuses a row whose statements reach a routine point
/// that does not exist.
pub const CATALOG: &[(&str, TraceFn)] = model_dsl::catalogue![
    (
        "kimi-bf16-wna16-kv-bf16",
        kimi_cuda::<ShippedW1, ShippedW2, ShippedA, ShippedKv>,
    ),
];
