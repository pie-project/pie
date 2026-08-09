//! Static C++ emission for the qwen3_5 HYBRID — rung 3's second family.
//!
//! This module starts with the DIGEST (the llama mechanism's port: one
//! format, two printers — this and `declared_facts.cpp` — held together
//! by the live static-form gate, which corrects any guessed emission
//! fact on first boot). The class-function emitter follows incrementally
//! (decode first), transliterating `qwen3_5/declared_forward.cpp`'s walk
//! arms exactly as `emit_cuda` transliterated llama's.

use super::facts::{Qwen35CudaFacts, Qwen35HybridFacts, Qwen35MlpKind};
use model_compiler::trace::NormVariant;

/// The digest naming what a generated qwen3_5 TU is emitted FROM.
/// Field-for-field the C++ printer in `declared_facts.cpp`.
pub fn facts_digest(facts: &Qwen35HybridFacts, cuda: &Qwen35CudaFacts) -> String {
    let nv = |v: NormVariant| match v {
        NormVariant::Plain => 0,
        NormVariant::Gemma => 1,
    };
    let (moe, dense_intermediate) = match &facts.mlp {
        Qwen35MlpKind::Dense { intermediate } => (0, *intermediate),
        Qwen35MlpKind::Moe(_) => (1, 0),
    };
    format!(
        "qwen3_5/l{}/int{}/v{}/te{}/nv{}/ah{}/aqh{}/akvh{}/ahd{}/arot{}/afq{}\
         /gkh{}/gvh{}/gkd{}/gvd{}/gck{}/gfi{}/moe{}/di{}/sb{}/wt{}/wtm{}/cm{}/vs{}/pd{}/pr{}",
        facts.layers,
        facts.full_attn_interval,
        facts.vocab,
        u8::from(facts.tied_embeddings),
        nv(facts.norm_variant),
        facts.attn.hidden,
        facts.attn.q_heads,
        facts.attn.kv_heads,
        facts.attn.head_dim,
        facts.attn.rotary_dim,
        u8::from(facts.attn.fused_qkv),
        facts.gdn.key_heads,
        facts.gdn.value_heads,
        facts.gdn.key_head_dim,
        facts.gdn.value_head_dim,
        facts.gdn.conv_kernel,
        u8::from(facts.gdn.fused_in_proj),
        moe,
        dense_intermediate,
        u8::from(cuda.state_bf16),
        u8::from(cuda.warp_tiled),
        cuda.warp_tiled_max,
        cuda.cached_max,
        u8::from(cuda.verify_stash),
        // The prefill-decode redirect changes the decode class's
        // attention op, so a body emitted with it on must not be
        // served to a deployment that turned it off.
        u8::from(cuda.prefill_decode),
        // The WEIGHT REPRESENTATION: a scaled projection states a
        // launcher and names scale tensors, so the emitted body differs
        // and the digest has to say which it was emitted from.
        match cuda.proj_repr {
            WeightRepr::Bf16 => 0,
            WeightRepr::Scaled {
                layout: ScaleLayout::PerTensor,
                ..
            } => 1,
            WeightRepr::Scaled {
                layout: ScaleLayout::PerChannel,
                ..
            } => 2,
            WeightRepr::Scaled {
                layout: ScaleLayout::PerGroup,
                ..
            } => 3,
            WeightRepr::Mxfp4Marlin => 4,
        },
    )
}

use super::qwen3_5_hybrid_cuda;
use model_compiler::dsl::{ScaleLayout, WeightRepr};
use model_compiler::trace::{FireClass, ForwardPlan, OpKind};

/// The parameter list shared with `qwen3_5_forward_declared`, minus the
/// plan and the per-class-constant inputs (`is_pure_decode`,
/// `commit_lens`).
///
/// `commit_lens` is not "later" — `.wiki/driver/graph.md` §4.2 RETIRED
/// the service classes, and a speculative decode now buffers its tokens
/// and folds only the accepted prefix, so there is no repair pass to
/// emit. The emitter used to carry a whole second axis for it: an
/// `emit_class_fn_commit` entry point, a `commit: bool` threaded
/// through five functions, a signature variant that appended
/// `commit_lens`, and a reset stage that was skipped under it. Nothing
/// ever called it, so every one of those branches emitted the `false`
/// side — removing the axis left all 153,887 lines of
/// `committed_cuda_emissions` byte-identical. What it had been emitting
/// instead was a claim: that this emitter can produce a commit-advance
/// class, standing beside a retirement note saying that class is gone.
const PARAMS: &str = "\
    const Qwen3_5Weights& w,\n\
    const HfConfig& cfg,\n\
    const Qwen3_5ForwardCfg& fwd_cfg,\n\
    const Qwen3_5PlanState& plan_state,\n\
    Workspace& ws,\n\
    Qwen3_5LinearAttnWorkspace& la,\n\
    KvCache& cache,\n\
    RecurrentStateCache& state_cache,\n\
    AttentionWorkspace& attn_ws,\n\
    kernels::gemm::CublasHandle& cublas,\n\
    const std::int32_t* token_ids,\n\
    const std::int32_t* positions,\n\
    const std::uint32_t* qo_indptr,\n\
    const std::uint32_t* kv_page_indices,\n\
    const std::uint32_t* kv_page_indptr,\n\
    const std::uint32_t* kv_last_page_lens,\n\
    const std::uint32_t* qo_indptr_h,\n\
    const std::uint32_t* kv_page_indptr_h,\n\
    int total_tokens,\n\
    int num_requests,\n\
    const std::uint32_t* w_page_d,\n\
    const std::uint32_t* w_off_d,\n\
    const std::uint8_t* row_valid_d,\n\
    bool has_write_desc,\n\
    const std::int32_t* slot_ids_h,\n\
    const std::uint8_t* is_fresh_h,\n\
    const std::int32_t* slot_ids_d,\n\
    const std::uint8_t* is_fresh_d,\n\
    const std::int32_t* logit_row_indices_d,\n\
    int num_logit_rows,\n\
    const StageHooks* hooks";

/// Emit the committed qwen3_5 `.inc`: the decode + prefill class
/// functions plus the digest constant. The service classes stay on the
/// interpreter walk for now (the incremental-widening pattern rung 3
/// itself used).
pub fn emit_qwen35_cuda_inc(
    facts: &Qwen35HybridFacts,
    cuda: &Qwen35CudaFacts,
    tag: &str,
) -> String {
    let decode = qwen3_5_hybrid_cuda(facts, cuda, FireClass::Decode);
    let prefill = qwen3_5_hybrid_cuda(facts, cuda, FireClass::Prefill);
    let digest = facts_digest(facts, cuda);
    let mut out = String::new();
    out.push_str(&format!(
        "// GENERATED by `cargo run -p pie-forward --bin emit-cuda` — DO NOT EDIT.\n\
         // The static C++ form of the lowered qwen3_5 hybrid class traces\n\
         // (north-star-dsl.md rung 3, second family): one statement per traced\n\
         // op, choices already made at trace time. Included by the qwen3_5\n\
         // declared_forward.cpp inside its namespace; regenerate after any\n\
         // change to the declaration, the facts below, or the emitter, and\n\
         // re-run the A/B battery. Decode + prefill only; the MTP service\n\
         // classes stay on the interpreter walk.\n\
         //\n\
         // Emitted from: {digest}\n\n\
         constexpr const char* kQ35GeneratedDigest_{tag} =\n    \"{digest}\";\n\n"
    ));
    out.push_str(&emit_class_fn(
        &decode,
        facts,
        cuda,
        &format!("generated_qwen35_decode_{tag}"),
        true,
    ));
    out.push('\n');
    out.push_str(&emit_class_fn(
        &prefill,
        facts,
        cuda,
        &format!("generated_qwen35_prefill_{tag}"),
        false,
    ));
    // The three SERVICE passes are not emitted: `.wiki/driver/graph.md`
    // §4.2 retired them. A speculative decode buffers its tokens and folds
    // only the accepted prefix, so there is no repair to generate.
    out
}

#[derive(Default)]
struct Body {
    out: String,
    indent: usize,
}

impl Body {
    fn line(&mut self, s: &str) {
        self.out.push_str(s);
        self.out.push('\n');
    }
    fn stmt(&mut self, s: &str) {
        for l in s.lines() {
            self.out.push_str("    ");
            for _ in 0..self.indent {
                self.out.push_str("    ");
            }
            self.out.push_str(l);
            self.out.push('\n');
        }
    }
}

fn split_layer_weight(name: &str) -> Option<(u32, &str)> {
    let rest = name.strip_prefix("layer.")?;
    let dot = rest.find('.')?;
    let layer = rest[..dot].parse().ok()?;
    Some((layer, &rest[dot + 1..]))
}

fn emit_class_fn(
    plan: &ForwardPlan,
    facts: &Qwen35HybridFacts,
    cuda: &Qwen35CudaFacts,
    fn_name: &str,
    is_decode: bool,
) -> String {
    let mut b = Body::default();
    b.line(&format!("inline void {fn_name}(\n{PARAMS})\n{{"));
    b.line("    // Locals mirror the interpreter's preamble.");
    b.line("    const int N = total_tokens;");
    b.line("    const int R = num_requests;");
    b.line("    const int H = cfg.hidden_size;");
    b.line("    const int num_q_heads = cfg.num_attention_heads;");
    b.line("    const int num_kv_heads = cfg.num_key_value_heads;");
    b.line("    const int d = cfg.head_dim;");
    b.line("    const int Hq = num_q_heads * d;");
    b.line("    const int Hk = num_kv_heads * d;");
    b.line("    const int I = cfg.intermediate_size;");
    b.line("    const int K_h = cfg.linear_num_key_heads;");
    b.line("    const int V_h = cfg.linear_num_value_heads;");
    b.line("    const int K_d = cfg.linear_key_head_dim;");
    b.line("    const int V_d = cfg.linear_value_head_dim;");
    b.line("    const int K_dim = K_h * K_d;");
    b.line("    const int V_dim = V_h * V_d;");
    b.line("    const int conv_dim = 2 * K_dim + V_dim;");
    b.line("    const int conv_K = cfg.linear_conv_kernel_dim;");
    b.line("    const float eps = cfg.rms_norm_eps;");
    b.line("    cudaStream_t stream = cublas.stream();");
    b.line("    (void)fwd_cfg; (void)attn_ws; (void)qo_indptr_h;");
    b.line("    (void)kv_page_indptr_h; (void)K_dim;");
    b.line("    if (std::getenv(\"PIE_DECLARED_FORWARD_TRACE\")) {");
    b.line("        std::fprintf(stderr,");
    b.line(&format!(
        "                     \"[declared-qwen35-generated] N=%d R=%d decode={}\\n\",",
        i32::from(is_decode)
    ));
    b.line("                     N, R);");
    b.line("    }");
    b.line("");
    b.line("    // Per-slot reset for freshly (re)assigned rs slots — the");
    b.line("    // interpreter's reset stage, verbatim (no commit-advance");
    b.line("    // arm: the service classes stay on the interpreter walk).");
    b.line("    if (slot_ids_h != nullptr && is_fresh_h != nullptr) {");
    b.line("        if (std::any_of(is_fresh_h, is_fresh_h + R,");
    b.line("                        [](auto fresh) { return fresh != 0; })) {");
    b.line("            if (slot_ids_d != nullptr && is_fresh_d != nullptr) {");
    b.line("                state_cache.reset_slots_if_fresh(");
    b.line("                    slot_ids_d, is_fresh_d, R, stream);");
    b.line("            } else {");
    b.line("                for (int r = 0; r < R; ++r) {");
    b.line("                    if (is_fresh_h[r]) {");
    b.line("                        state_cache.reset_slot(slot_ids_h[r], stream);");
    b.line("                    }");
    b.line("                }");
    b.line("            }");
    b.line("        }");
    if !is_decode {
        b.line("    } else {");
        b.line("        // Legacy null-slot prefill: reset all.");
        b.line("        state_cache.reset(stream);");
    }
    b.line("    }");
    b.line("");
    b.line("    const kernels::attn::DecodePlanCache* decode_plan =");
    b.line("        plan_state.decode_plan ? plan_state.decode_plan.get() : nullptr;");
    b.line("    const kernels::attn::PrefillPlanCache* prefill_plan =");
    b.line("        (plan_state.use_prefill_plan && plan_state.prefill_plan)");
    b.line("            ? plan_state.prefill_plan.get()");
    b.line("            : nullptr;");
    b.line("    const auto slot_stride = static_cast<long long>(");
    b.line("        state_cache.recurrent_slot_stride_floats());");
    b.line("    const bool write_state = !state_cache.verify_frozen();");
    b.line("    (void)write_state;");
    // q_recur/k_recur: the V_h == K_h choice is a FACT — emit the chosen
    // buffer directly.
    let gqa = facts.gdn.value_heads != facts.gdn.key_heads;
    if gqa {
        b.line("    const float* q_recur_full = la.q_norm.data();");
        b.line("    const float* k_recur_full = la.k_norm.data();");
    } else {
        b.line("    const float* q_recur_full = la.q_pre.data();");
        b.line("    const float* k_recur_full = la.k_pre.data();");
    }
    b.line("    (void)q_recur_full; (void)k_recur_full;");
    b.line("");
    let mut repeat_next_is_k = false;
    emit_range(
        &mut b,
        plan,
        facts,
        cuda,
        is_decode,
        0,
        plan.ops.len(),
        &mut repeat_next_is_k,
    );
    b.line("}");
    b.out
}

// The emitter's context, threaded -- see the note on `emit_range` in
// `shared::llama_like::forward::emit`, which this file mirrors.
#[allow(clippy::too_many_arguments)]
fn emit_range(
    b: &mut Body,
    plan: &ForwardPlan,
    facts: &Qwen35HybridFacts,
    cuda: &Qwen35CudaFacts,
    is_decode: bool,
    start: usize,
    end: usize,
    repeat_next_is_k: &mut bool,
) {
    let mut i = start;
    while i < end {
        let op = &plan.ops[i];
        if let OpKind::Guard { arms, else_ops } = &op.kind {
            let cond_of = |pred: &model_compiler::trace::GuardPred| match pred {
                model_compiler::trace::GuardPred::HasWriteDesc => "has_write_desc".to_string(),
                model_compiler::trace::GuardPred::TokensLE(k) => format!("N <= {k}"),
                model_compiler::trace::GuardPred::TokensGT(k) => format!("N > {k}"),
                other => panic!("emitter(q35): guard pred {other:?} out of scope"),
            };
            let mut region = i + 1;
            for (n, arm) in arms.iter().enumerate() {
                let kw = if n == 0 { "if" } else { "} else if" };
                b.stmt(&format!("{kw} ({}) {{", cond_of(&arm.pred)));
                b.indent += 1;
                emit_range(
                    b,
                    plan,
                    facts,
                    cuda,
                    is_decode,
                    region,
                    region + arm.ops as usize,
                    repeat_next_is_k,
                );
                b.indent -= 1;
                region += arm.ops as usize;
            }
            b.stmt("} else {");
            b.indent += 1;
            emit_range(
                b,
                plan,
                facts,
                cuda,
                is_decode,
                region,
                region + *else_ops as usize,
                repeat_next_is_k,
            );
            b.indent -= 1;
            b.stmt("}");
            i = region + *else_ops as usize;
            continue;
        }
        emit_op(b, op, plan, facts, cuda, is_decode, repeat_next_is_k);
        i += 1;
    }
}

/// The ROW norm's emission, shared by the semantic kind and the stated
/// symbol.
///
/// A CUDA text states `norm::rmsnorm_gemma_bf16` now
/// (`dsl::cuda::rmsnorm`); a semantic one still records the kind. Same
/// buffers, same weights, one body.
fn emit_row_norm_q35(b: &mut Body, weight: &str) {
    // Gemma fold everywhere (the walk's drift check is emission-
    // time here: the trace carries the variant the facts stated).
    if weight == "final_norm" {
        b.stmt("kernels::norm::rmsnorm_gemma_bf16(");
        b.stmt("    ws.y.data(), require(w.final_norm, \"final_norm\")->data(),");
        b.stmt("    ws.norm_x.data(), N, H, eps, stream);");
        return;
    }
    let (layer, field) =
        split_layer_weight(weight).unwrap_or_else(|| panic!("emitter(q35): norm weight {weight}"));
    let member = match field {
        "attn_norm" => "attn_norm_pre",
        "mlp_norm" => "mlp_norm_pre",
        other => panic!("emitter(q35): row-norm field {other}"),
    };
    b.stmt("kernels::norm::rmsnorm_gemma_bf16(");
    b.stmt(&format!(
        "    ws.y.data(), require(w.layers[{layer}].{member}, \"{weight}\")->data(),"
    ));
    b.stmt("    ws.norm_x.data(), N, H, eps, stream);");
}

fn emit_op(
    b: &mut Body,
    op: &model_compiler::trace::Op,
    plan: &ForwardPlan,
    facts: &Qwen35HybridFacts,
    cuda: &Qwen35CudaFacts,
    is_decode: bool,
    repeat_next_is_k: &mut bool,
) {
    let _ = (plan, is_decode);
    let is_full = |l: u32| facts.is_full_attn(l);
    match &op.kind {
        OpKind::Embed { weight } => {
            assert_eq!(weight, "embed");
            b.stmt("kernels::layout::embed_bf16(");
            b.stmt("    token_ids, require(w.embed, \"embed\")->data(), ws.y.data(),");
            b.stmt("    N, H, cfg.vocab_size, stream);");
        }
        OpKind::Rmsnorm { weight, .. } => {
            emit_row_norm_q35(b, weight);
        }
        OpKind::Matmul {
            weight,
            beta_one,
            selector,
        } => {
            assert!(selector.is_none(), "emitter(q35): dyn matmul out of scope");
            let (layer, field) = split_layer_weight(weight)
                .unwrap_or_else(|| panic!("emitter(q35): matmul weight {weight}"));
            let beta = if *beta_one { "1.f" } else { "0.f" };
            let raw = |member: &str, out: &str, width: &str| {
                format!(
                    "kernels::gemm::act_x_w(cublas.handle(),\n    ws.norm_x.data(),\n    *require(w.layers[{layer}].{member}, \"{weight}\"),\n    {out}, N, {width}, H);"
                )
            };
            match field {
                "in_proj_qkvz" => b.stmt(&raw(
                    "la_in_proj_qkvz",
                    "la.mixed_qkvz.data()",
                    "conv_dim + V_dim",
                )),
                "in_proj_ba" => b.stmt(&raw("la_in_proj_ba", "la.ba.data()", "2 * V_h")),
                "in_proj_qkv" => b.stmt(&raw("la_in_proj_qkv", "la.mixed_qkv.data()", "conv_dim")),
                "in_proj_z" => b.stmt(&raw("la_in_proj_z", "la.z.data()", "V_dim")),
                "in_proj_a" => b.stmt(&raw("la_in_proj_a", "la.a.data()", "V_h")),
                "in_proj_b" => b.stmt(&raw("la_in_proj_b", "la.b.data()", "V_h")),
                "qgkv" => {
                    b.stmt("kernels::gemm::act_x_w(cublas.handle(),");
                    b.stmt("    ws.norm_x.data(),");
                    b.stmt(&format!(
                        "    WeightView(*require(w.layers[{layer}].fa_qgkv_proj_fused, \"{weight}\")),"
                    ));
                    b.stmt("    ws.gate_up_fused.data(), N, 2 * Hq + 2 * Hk, H);");
                }
                "q_proj" => {
                    b.stmt("kernels::gemm::act_x_w(cublas.handle(),");
                    b.stmt("    ws.norm_x.data(),");
                    b.stmt(&format!(
                        "    make_weight_view(require(w.layers[{layer}].fa_q_proj, \"{weight}\"), w.layers[{layer}].fa_q_proj_quant),"
                    ));
                    b.stmt("    la.fa_qg_packed.data(), N, 2 * Hq, H);");
                }
                "k_proj" => {
                    b.stmt("kernels::gemm::act_x_w(cublas.handle(),");
                    b.stmt("    ws.norm_x.data(),");
                    b.stmt(&format!(
                        "    make_weight_view(require(w.layers[{layer}].fa_k_proj, \"{weight}\"), w.layers[{layer}].fa_k_proj_quant),"
                    ));
                    b.stmt("    ws.k.data(), N, Hk, H);");
                }
                "v_proj" => {
                    b.stmt("kernels::gemm::act_x_w(cublas.handle(),");
                    b.stmt("    ws.norm_x.data(),");
                    b.stmt(&format!(
                        "    make_weight_view(require(w.layers[{layer}].fa_v_proj, \"{weight}\"), w.layers[{layer}].fa_v_proj_quant),"
                    ));
                    b.stmt("    ws.v.data(), N, Hk, H);");
                }
                "o_proj" => {
                    if is_full(layer) {
                        b.stmt("kernels::gemm::act_x_w(cublas.handle(),");
                        b.stmt("    ws.attn_out.data(),");
                        b.stmt(&format!(
                            "    make_weight_view(require(w.layers[{layer}].fa_o_proj, \"{weight}\"), w.layers[{layer}].fa_o_proj_quant),"
                        ));
                        b.stmt(&format!("    ws.y.data(), N, H, Hq, {beta});"));
                    } else {
                        b.stmt("kernels::gemm::act_x_w(cublas.handle(),");
                        b.stmt("    la.core_out_bf16.data(),");
                        b.stmt(&format!(
                            "    *require(w.layers[{layer}].la_out_proj, \"{weight}\"),"
                        ));
                        b.stmt(&format!("    ws.y.data(), N, H, V_dim, {beta});"));
                    }
                }
                // The projection half of the same binding the activation
                // reads. The trace declares ONE packed matmul either way
                // — one buffer or two is `lower::Buffers`' question — so
                // the emitter is where the answer belongs, and the fact
                // is what it answers from. This used to re-derive
                // `gate_up_proj_fused != nullptr && !ws.gate_up_fused
                // .empty()` per layer and branch on it at runtime.
                "gate_up" if cuda.gate_up_fused => {
                    b.stmt("kernels::gemm::act_x_w(cublas.handle(),");
                    b.stmt("    ws.norm_x.data(),");
                    b.stmt(&format!(
                        "    WeightView(*require(w.layers[{layer}].gate_up_proj_fused, \"{weight}\")),"
                    ));
                    b.stmt("    ws.gate_up_fused.data(), N, 2 * I, H);");
                }
                "gate_up" => {
                    b.stmt("kernels::gemm::act_x_w(cublas.handle(),");
                    b.stmt("    ws.norm_x.data(),");
                    b.stmt(&format!(
                        "    make_weight_view(require(w.layers[{layer}].gate_proj, \"{weight}\"), w.layers[{layer}].gate_proj_quant),"
                    ));
                    b.stmt("    ws.gate.data(), N, I, H);");
                    b.stmt("kernels::gemm::act_x_w(cublas.handle(),");
                    b.stmt("    ws.norm_x.data(),");
                    b.stmt(&format!(
                        "    make_weight_view(require(w.layers[{layer}].up_proj, \"{weight}\"), w.layers[{layer}].up_proj_quant),"
                    ));
                    b.stmt("    ws.up.data(), N, I, H);");
                }
                "down" => {
                    b.stmt("kernels::gemm::act_x_w(cublas.handle(),");
                    b.stmt("    ws.gate.data(),");
                    b.stmt(&format!(
                        "    make_weight_view(require(w.layers[{layer}].down_proj, \"{weight}\"), w.layers[{layer}].down_proj_quant),"
                    ));
                    b.stmt(&format!("    ws.y.data(), N, H, I, {beta});"));
                }
                other => panic!("emitter(q35): matmul field {other}"),
            }
        }
        OpKind::SplitQkv { .. } => {
            b.stmt("kernels::attn::split_qkv_bf16(");
            b.stmt("    ws.gate_up_fused.data(),");
            b.stmt("    la.fa_qg_packed.data(), ws.k.data(), ws.v.data(),");
            b.stmt("    N, 2 * Hq, Hk, stream);");
        }
        OpKind::GdnPrep { a_log, .. } => {
            let (layer, _) = split_layer_weight(a_log)
                .unwrap_or_else(|| panic!("emitter(q35): prep weight {a_log}"));
            b.stmt("kernels::ssm::qwen_gdn_post_conv_prep_bf16(");
            b.stmt("    la.mixed_qkv_post.data(), la.a.data(), la.b.data(),");
            b.stmt(&format!("    w.layers[{layer}].la_A_log_fp32,"));
            b.stmt(&format!(
                "    require(w.layers[{layer}].la_dt_bias, \"layer.{layer}.dt_bias\")->data(),"
            ));
            b.stmt("    la.q_pre.data(), la.k_pre.data(), la.v_fp32.data(),");
            b.stmt("    la.g_log.data(), la.beta.data(),");
            b.stmt("    N, K_h, V_h, K_d, V_d, conv_dim, stream);");
        }
        OpKind::RmsnormGated { weight } => {
            let (layer, _) = split_layer_weight(weight)
                .unwrap_or_else(|| panic!("emitter(q35): gate_norm {weight}"));
            b.stmt("kernels::norm::rmsnorm_gated_fp32_in_bf16(");
            b.stmt(&format!(
                "    la.core_out.data(), la.z.data(), w.layers[{layer}].la_norm_w_fp32,"
            ));
            b.stmt("    la.core_out_bf16.data(),");
            b.stmt("    N * V_h, V_d, /*eps=*/eps, stream);");
        }
        OpKind::SplitQGate { .. } => {
            b.stmt("kernels::layout::split_q_gate_bf16(");
            b.stmt("    la.fa_qg_packed.data(), ws.q.data(), la.fa_gate.data(),");
            b.stmt("    N, num_q_heads, d, stream);");
        }
        OpKind::RmsnormPerHead { weight, .. } => {
            let (layer, field) = split_layer_weight(weight)
                .unwrap_or_else(|| panic!("emitter(q35): per-head norm {weight}"));
            let (buf, heads, member) = match field {
                "q_norm" => ("ws.q.data()", "num_q_heads", "fa_q_norm"),
                "k_norm" => ("ws.k.data()", "num_kv_heads", "fa_k_norm"),
                other => panic!("emitter(q35): per-head field {other}"),
            };
            b.stmt("kernels::norm::rmsnorm_gemma_bf16(");
            b.stmt(&format!(
                "    {buf}, require(w.layers[{layer}].{member}, \"{weight}\")->data(),"
            ));
            b.stmt(&format!("    {buf}, N * {heads}, d, eps, stream);"));
        }
        OpKind::SigmoidGateMul => {
            b.stmt("kernels::mlp::sigmoid_gate_inplace_bf16(");
            b.stmt("    ws.attn_out.data(), la.fa_gate.data(), N * Hq, stream);");
        }
        OpKind::HookSite { stage, layer } => {
            // qwen3_5 sites are observation-only (A4): the invoke with
            // the layer-kind buffer/width as constants.
            let (buf, width, f32flag) = if is_full(*layer) {
                ("ws.q.data()", "Hq".to_string(), "false")
            } else {
                ("la.q_pre.data()", "K_h * K_d".to_string(), "true")
            };
            let point = match stage {
                model_compiler::trace::HookStage::OnAttnProj => "OnAttnProj",
                model_compiler::trace::HookStage::OnAttn => "OnAttn",
            };
            b.stmt("if (hooks != nullptr) {");
            b.stmt("    invoke_stage_hook(");
            b.stmt(&format!("        hooks, StageHookPoint::{point}, {buf},"));
            b.stmt("        static_cast<std::uint32_t>(N),");
            b.stmt(&format!("        static_cast<std::uint32_t>({width}),"));
            b.stmt(&format!(
                "        {layer}u, stream, /*query_is_f32=*/{f32flag});"
            ));
            b.stmt("}");
        }
        OpKind::LmHead { weight } => {
            let member = match weight.as_str() {
                "embed" => "w.embed",
                "lm_head" => "w.lm_head",
                other => panic!("emitter(q35): lm_head weight {other}"),
            };
            b.stmt("// The epilogue: final norm already landed ALL rows in");
            b.stmt("// norm_x (its op above); gather compact rows, multiply,");
            b.stmt("// and copy the normed hidden back to ws.y (MTP/state");
            b.stmt("// plumbing) — the interpreter's arm verbatim.");
            b.stmt("if (logit_row_indices_d != nullptr &&");
            b.stmt("    num_logit_rows > 0 &&");
            b.stmt("    num_logit_rows < N) {");
            b.stmt("    kernels::layout::gather_bf16_rows(");
            b.stmt("        static_cast<const std::uint16_t*>(ws.norm_x.data()),");
            b.stmt("        logit_row_indices_d,");
            b.stmt("        static_cast<std::uint16_t*>(ws.norm_y.data()),");
            b.stmt("        num_logit_rows, H, stream);");
            b.stmt("    kernels::gemm::act_x_w(cublas.handle(),");
            b.stmt(&format!(
                "        ws.norm_y.data(), *require({member}, \"{weight}\"),"
            ));
            b.stmt("        ws.logits.data(), num_logit_rows, cfg.vocab_size, H);");
            b.stmt("} else {");
            b.stmt("    kernels::gemm::act_x_w(cublas.handle(),");
            b.stmt(&format!(
                "        ws.norm_x.data(), *require({member}, \"{weight}\"),"
            ));
            b.stmt("        ws.logits.data(), N, cfg.vocab_size, H);");
            b.stmt("}");
            b.stmt("CUDA_CHECK(cudaMemcpyAsync(");
            b.stmt("    ws.y.data(), ws.norm_x.data(),");
            b.stmt("    static_cast<std::size_t>(N) * H * sizeof(std::uint16_t),");
            b.stmt("    cudaMemcpyDeviceToDevice, stream));");
        }
        OpKind::Launch {
            kernel,
            weights,
            state,
            params,
        } => emit_launch(
            b,
            kernel,
            weights,
            state.as_ref(),
            params,
            op,
            facts,
            repeat_next_is_k,
        ),
        other => panic!("emitter(q35): op kind {other:?} out of scope"),
    }
}

// The emitter's context, threaded -- see `emit_range` above.
#[allow(clippy::too_many_arguments)]
fn emit_launch(
    b: &mut Body,
    kernel: &str,
    weights: &[String],
    state: Option<&model_compiler::trace::StateRef>,
    // The statement's scalar arguments -- see `OpKind::Launch::params`.
    params: &[u32],
    op: &model_compiler::trace::Op,
    facts: &Qwen35HybridFacts,
    repeat_next_is_k: &mut bool,
) {
    let _ = op;
    let sl = state.map(|s| s.layer).unwrap_or(0);
    // The interpreter's binding lambdas, emitted per site with the layer
    // constant: the conv weight bank and the model-layer→kv-slot map.
    let conv_pre = |b: &mut Body| {
        let (layer, _) = split_layer_weight(&weights[0])
            .unwrap_or_else(|| panic!("emitter(q35): conv weight {}", weights[0]));
        b.stmt(&format!("{{ const auto& cl = w.layers[{layer}];"));
        layer
    };
    let kv_view_pre = |b: &mut Body| {
        b.stmt(&format!(
            "{{ auto kv_view = cache.layer_view(w.layers[{sl}].kv_layer);"
        ));
    };
    match kernel {
        // The activation, with the binding's answer already in it. This
        // used to be a per-layer `if (gate_up_fused_N)` in the emitted
        // C++ — a runtime read of a workspace to recover something the
        // load already knew. The fact deletes the branch here, in the
        // interpreter's arm, and in the flat list's residue at once.
        "mlp::chunked_swiglu_bf16" => {
            b.stmt("kernels::mlp::chunked_swiglu_bf16(");
            b.stmt("    ws.gate_up_fused.data(), ws.gate.data(), N, I, stream);");
        }
        "mlp::swiglu_bf16" => {
            b.stmt("kernels::mlp::swiglu_bf16(");
            b.stmt("    ws.gate.data(), ws.up.data(), ws.gate.data(),");
            b.stmt("    N * I, stream);");
        }
        "ssm::causal_conv1d_update_batched_bf16" => {
            conv_pre(b);
            b.stmt("  kernels::ssm::causal_conv1d_update_batched_bf16(");
            b.stmt("      la.mixed_qkv.data(), cl.la_conv1d_w->data(),");
            b.stmt("      cl.la_conv1d_b ? cl.la_conv1d_b->data() : nullptr,");
            b.stmt(&format!("      state_cache.conv_state({sl}, /*slot=*/0),"));
            b.stmt("      slot_ids_d,");
            b.stmt("      static_cast<long long>(state_cache.conv_kernel()) *");
            b.stmt("          state_cache.conv_dim(),");
            b.stmt("      la.mixed_qkv_post.data(),");
            b.stmt("      R, conv_dim, conv_K, stream); }");
        }
        "ssm::causal_conv1d_prefill_batched_bf16" => {
            conv_pre(b);
            b.stmt("  kernels::ssm::causal_conv1d_prefill_batched_bf16(");
            b.stmt("      la.mixed_qkv.data(), cl.la_conv1d_w->data(),");
            b.stmt("      cl.la_conv1d_b ? cl.la_conv1d_b->data() : nullptr,");
            b.stmt("      la.mixed_qkv_post.data(),");
            b.stmt(&format!("      state_cache.conv_state({sl}, /*slot=*/0),"));
            b.stmt("      slot_ids_d, qo_indptr,");
            b.stmt("      static_cast<long long>(state_cache.conv_kernel()) *");
            b.stmt("          state_cache.conv_dim(),");
            b.stmt("      R, conv_dim, conv_K, stream, write_state,");
            b.stmt("      /*commit_lens=*/nullptr); }");
        }
        k @ ("ssm::recurrent_gated_delta_step_batched"
        | "ssm::recurrent_gated_delta_step_batched_state_bf16"
        | "ssm::recurrent_gated_delta_step_batched_gqa"
        | "ssm::recurrent_gated_delta_step_batched_gqa_state_bf16") => {
            let gqa = k.contains("_gqa");
            let bf16 = k.ends_with("state_bf16");
            let cast = if bf16 { "" } else { "static_cast<float*>" };
            b.stmt(&format!("kernels::{k}("));
            if gqa {
                b.stmt("    la.q_pre.data(), la.k_pre.data(),");
            } else {
                b.stmt("    q_recur_full, k_recur_full,");
            }
            b.stmt("    la.v_fp32.data(), la.g_log.data(), la.beta.data(),");
            b.stmt(&format!(
                "    {cast}(state_cache.recurrent_state_raw({sl}, /*slot=*/0)),"
            ));
            b.stmt("    slot_ids_d, slot_stride,");
            if gqa {
                b.stmt("    la.core_out.data(), R, K_h, V_h, K_d, V_d, stream);");
            } else {
                b.stmt("    la.core_out.data(), R, V_h, K_d, V_d, stream);");
            }
        }
        k @ ("launch_chunk_gated_delta_prefill_batched_warp_tiled"
        | "launch_chunk_gated_delta_prefill_batched_warp_tiled_state_bf16"
        | "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa"
        | "ssm::chunk_gated_delta_prefill_batched_warp_tiled_gqa_state_bf16"
        | "ssm::chunk_gated_delta_prefill_batched_cached"
        | "ssm::chunk_gated_delta_prefill_batched_cached_state_bf16"
        | "ssm::chunk_gated_delta_prefill_batched"
        | "ssm::chunk_gated_delta_prefill_batched_state_bf16") => {
            let gqa_direct =
                k.contains("_gqa") || (!k.contains("warp_tiled") && !k.contains("cached"));
            let bf16 = k.ends_with("state_bf16");
            let fla = !k.contains("warp_tiled") && !k.contains("cached");
            let cast = if bf16 { "" } else { "static_cast<float*>" };
            b.stmt(&format!("kernels::{k}("));
            if gqa_direct {
                b.stmt("    la.q_pre.data(), la.k_pre.data(),");
            } else {
                b.stmt("    q_recur_full, k_recur_full,");
            }
            b.stmt("    la.v_fp32.data(), la.g_log.data(), la.beta.data(),");
            b.stmt(&format!(
                "    {cast}(state_cache.recurrent_state_raw({sl}, /*slot=*/0)),"
            ));
            b.stmt("    slot_ids_d, qo_indptr,");
            b.stmt("    slot_stride, la.core_out.data(),");
            if k.contains("_gqa") || fla {
                if fla {
                    b.stmt("    R, K_h, V_h, K_d, V_d, stream, write_state,");
                    b.stmt("    /*commit_lens=*/nullptr);");
                } else {
                    b.stmt("    R, K_h, V_h, K_d, V_d, stream, write_state);");
                }
            } else {
                b.stmt("    R, V_h, K_d, V_d, stream, write_state);");
            }
        }
        "ssm::repeat_interleave_heads_fp32" => {
            // The declaration states the pair q-then-k; emission binds
            // them in that order, statically.
            let (src, dst) = if *repeat_next_is_k {
                ("la.k_pre.data()", "la.k_norm.data()")
            } else {
                ("la.q_pre.data()", "la.q_norm.data()")
            };
            *repeat_next_is_k = !*repeat_next_is_k;
            b.stmt("kernels::ssm::repeat_interleave_heads_fp32(");
            b.stmt(&format!("    {src}, {dst}, N, K_h, V_h, K_d, stream);"));
        }
        "attn::dispatch_attention_flashinfer_decode" => {
            b.stmt("if (decode_plan == nullptr) {");
            b.stmt("    throw std::runtime_error(");
            b.stmt("        \"generated qwen35: no decode plan\");");
            b.stmt("}");
            kv_view_pre(b);
            b.stmt("  kernels::attn::dispatch_attention_flashinfer_decode(");
            b.stmt("      *decode_plan,");
            b.stmt("      ws.q.data(), kv_view, ws.attn_out.data(),");
            b.stmt("      kv_page_indices, kv_page_indptr, kv_last_page_lens,");
            b.stmt("      attn_ws.view(), stream); }");
        }
        "attn::dispatch_attention_flashinfer_prefill_bf16" => {
            b.stmt("if (prefill_plan == nullptr) {");
            b.stmt("    throw std::runtime_error(");
            b.stmt("        \"generated qwen35: no prefill plan\");");
            b.stmt("}");
            kv_view_pre(b);
            b.stmt("  kernels::attn::dispatch_attention_flashinfer_prefill_bf16(");
            b.stmt("      *prefill_plan,");
            b.stmt("      ws.q.data(), kv_view.k_bf16_pages, kv_view.v_bf16_pages,");
            b.stmt("      ws.attn_out.data(),");
            b.stmt("      qo_indptr, kv_page_indices, kv_page_indptr,");
            b.stmt("      kv_last_page_lens, attn_ws.view(), stream); }");
        }
        "attn::write_kv_explicit_bf16" => {
            kv_view_pre(b);
            b.stmt("  kernels::attn::write_kv_explicit_bf16(");
            b.stmt("      kv_view, ws.k.data(), ws.v.data(),");
            b.stmt("      w_page_d, w_off_d, N, stream, row_valid_d); }");
        }
        "attn::write_kv_to_pages" => {
            kv_view_pre(b);
            b.stmt("  kernels::attn::write_kv_to_pages(");
            b.stmt("      kv_view, ws.k.data(), ws.v.data(),");
            b.stmt("      qo_indptr, kv_page_indices, kv_page_indptr,");
            b.stmt("      kv_last_page_lens, N, R, stream); }");
        }
        k @ ("qwen35_verify_stash_load" | "qwen35_verify_stash_store") => {
            // The pseudo-symbols: a cudaMemcpyAsync trio against the
            // layer's stash slab, with the COMPACT linear index a
            // compile-time constant here (the interpreter derives it by
            // counting; the emitter counts at emission).
            let load = k.ends_with("load");
            let linear_idx = (0..sl).filter(|&l| !facts.is_full_attn(l)).count();
            b.stmt("if (!state_cache.verify_hidden_stash_enabled()) {");
            b.stmt("    throw std::runtime_error(");
            b.stmt("        \"generated qwen35: stated stash op but the live \"");
            b.stmt("        \"stash is disabled (cross-check drift)\");");
            b.stmt("}");
            b.stmt("{");
            b.stmt(&format!(
                "    auto* stash = static_cast<std::uint16_t*>(\n        state_cache.verify_hidden_stash_layer({linear_idx}));"
            ));
            b.stmt("    const std::size_t stash_stride =");
            b.stmt("        static_cast<std::size_t>(state_cache.verify_stash_max_tokens());");
            b.stmt("    const std::size_t a_off = stash_stride * conv_dim;");
            b.stmt("    const std::size_t b_off =");
            b.stmt("        a_off + stash_stride * static_cast<std::size_t>(V_h);");
            b.stmt("    const std::size_t n_qkv =");
            b.stmt("        static_cast<std::size_t>(N) * conv_dim * sizeof(std::uint16_t);");
            b.stmt("    const std::size_t n_ab =");
            b.stmt("        static_cast<std::size_t>(N) * V_h * sizeof(std::uint16_t);");
            let (d1, s1, d2, s2, d3, s3) = if load {
                (
                    "la.mixed_qkv.data()",
                    "stash",
                    "la.a.data()",
                    "stash + a_off",
                    "la.b.data()",
                    "stash + b_off",
                )
            } else {
                (
                    "stash",
                    "la.mixed_qkv.data()",
                    "stash + a_off",
                    "la.a.data()",
                    "stash + b_off",
                    "la.b.data()",
                )
            };
            for (dst, src, n) in [(d1, s1, "n_qkv"), (d2, s2, "n_ab"), (d3, s3, "n_ab")] {
                b.stmt("    CUDA_CHECK(cudaMemcpyAsync(");
                b.stmt(&format!("        {dst}, {src}, {n},"));
                b.stmt("        cudaMemcpyDeviceToDevice, stream));");
            }
            b.stmt("}");
        }
        // The ROTATION, now that `cuda::rope_partial` names it and
        // carries its width. The semantic arm read that width off the
        // op; this reads it off the statement's params, which is the
        // same fact through the channel that survives a `Launch`.
        "rope::rope_partial_bf16" => {
            let rot = params
                .first()
                .expect("a partial rotation states its rotary width");
            b.stmt("kernels::rope::rope_partial_bf16(");
            b.stmt("    ws.q.data(), ws.k.data(), positions,");
            b.stmt("    N, num_q_heads, num_kv_heads,");
            b.stmt(&format!("    d, {rot}, cfg.rope_theta, stream);"));
        }
        "rope::rope_bf16" => {
            b.stmt("kernels::rope::rope_bf16(");
            b.stmt("    ws.q.data(), ws.k.data(), positions,");
            b.stmt("    N, num_q_heads, num_kv_heads, d,");
            b.stmt("    cfg.rope_theta, stream);");
        }
        // The two GDN SPLITS, told apart by their symbols where the
        // semantic arm compared widths against `conv_dim` / `V_dim`.
        "layout::split_bf16_rows" => {
            b.stmt("kernels::layout::split_bf16_rows(");
            b.stmt("    la.mixed_qkvz.data(), la.mixed_qkv.data(), la.z.data(),");
            b.stmt("    N, conv_dim, V_dim, stream);");
        }
        "layout::split_qwen_gdn_ba_bf16" => {
            b.stmt("kernels::layout::split_qwen_gdn_ba_bf16(");
            b.stmt("    la.ba.data(), la.b.data(), la.a.data(), N, V_h, stream);");
        }
        // The ROW norms, now that `cuda::rmsnorm` states the fold.
        "norm::rmsnorm_bf16" | "norm::rmsnorm_gemma_bf16" => {
            let weight = weights.first().expect("a stated row norm names its weight");
            emit_row_norm_q35(b, weight);
        }
        other => panic!("emitter(q35): stated kernel {other} out of scope"),
    }
}
