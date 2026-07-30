#include "model/llama_like/declared_forward.hpp"

#include <charconv>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>
#include <string_view>

#include <cuda_runtime.h>

#include "kernels/embed.hpp"
#include "kernels/gather_rows.hpp"
#include "kernels/head_dim_pad.hpp"
#include "kernels/kv_paged.hpp"
#include "kernels/rmsnorm.hpp"
#include "kernels/rope.hpp"
#include "kernels/split_packed.hpp"
#include "kernels/swiglu.hpp"
#include "ops/attention_flashinfer.hpp"
#include "ops/attention_xqa.hpp"
#include "ops/gemm.hpp"

namespace pie_cuda_driver::model {

namespace {

using pie_forward::PieForwardNormVariant;
using pie_forward::PieForwardOp;
using pie_forward::PieForwardOpKind;
using pie_forward::PieForwardRopeKind;

// A plan weight name split into its layer index and field: "layer.3.qkv" →
// {3, "qkv"}; prologue/epilogue names ("embed", "final_norm") keep layer -1.
// The vocabulary is `forward/src/family.rs`'s and nothing else; anything the
// parse cannot place throws, loudly, because a name the resolver does not
// know means the trace and this executor have drifted.
struct ParsedWeightName {
    int layer = -1;
    std::string_view field;
};

[[noreturn]] void throw_unknown_weight(std::string_view name) {
    throw std::runtime_error(
        "declared forward: unknown weight name '" + std::string(name) +
        "' (trace vocabulary is forward/src/family.rs's)");
}

ParsedWeightName parse_weight_name(std::string_view name) {
    constexpr std::string_view prefix = "layer.";
    if (name.substr(0, prefix.size()) != prefix) {
        return ParsedWeightName{-1, name};
    }
    const std::size_t dot = name.find('.', prefix.size());
    if (dot == std::string_view::npos) throw_unknown_weight(name);
    int layer = -1;
    const char* first = name.data() + prefix.size();
    const char* last = name.data() + dot;
    const auto [ptr, ec] = std::from_chars(first, last, layer);
    if (ec != std::errc() || ptr != last || layer < 0) {
        throw_unknown_weight(name);
    }
    return ParsedWeightName{layer, name.substr(dot + 1)};
}

const Qwen3LayerWeights& layer_of(
    const Qwen3Weights& w, const ParsedWeightName& nm, std::string_view name)
{
    if (nm.layer < 0 ||
        nm.layer >= static_cast<int>(w.layers.size())) {
        throw_unknown_weight(name);
    }
    return w.layers[nm.layer];
}

const DeviceTensor* require(const DeviceTensor* t, std::string_view name) {
    if (t == nullptr) {
        throw std::runtime_error(
            "declared forward: weight '" + std::string(name) +
            "' is named by the trace but not bound");
    }
    return t;
}

// Cursor advance for the fused decode-QKV peephole: the ONE kernel just
// launched (`launch_qkv_decode_qk_norm_rope_write_kv_bf16`) computed the
// plan ops
//   SplitQkv, RmsnormPerHead(q_norm), RmsnormPerHead(k_norm),
//   Rope(Standard), KvAppend(layer)
// so the walk must not launch them again. family.rs's layer body emits
// exactly this run after Matmul(qkv) whenever fused_qkv && qk_norm — both
// of which the peephole's predicate requires — so the adjacency is checked
// op by op, kinds AND payloads, and a mismatch throws: a trace whose shape
// drifted must fail, not silently half-fuse (the kernel's side effects are
// already in ws.q and the paged cache).
//
// Returns the index of the KvAppend; the loop's `++i` then lands on
// Attention.
std::size_t skip_fused_decode_qkv_ops(
    const pie_forward::ForwardPlan& plan, std::size_t i, int layer)
{
    const auto expect = [&](std::size_t at, PieForwardOpKind kind,
                            const char* what) -> const PieForwardOp& {
        if (at >= plan.op_count() || plan.op(at).kind != kind) {
            throw std::runtime_error(
                std::string("declared forward: fused decode-QKV peephole "
                            "expected ") + what +
                " at op " + std::to_string(at) +
                " after Matmul(qkv); the trace's shape drifted from "
                "family.rs's fused_qkv layer body");
        }
        return plan.op(at);
    };
    expect(i + 1, PieForwardOpKind::SplitQkv, "SplitQkv");
    const PieForwardOp& qn = expect(
        i + 2, PieForwardOpKind::RmsnormPerHead, "RmsnormPerHead(q_norm)");
    const PieForwardOp& kn = expect(
        i + 3, PieForwardOpKind::RmsnormPerHead, "RmsnormPerHead(k_norm)");
    const PieForwardOp& rope = expect(
        i + 4, PieForwardOpKind::Rope, "Rope(Standard)");
    const PieForwardOp& append = expect(
        i + 5, PieForwardOpKind::KvAppend, "KvAppend");
    const ParsedWeightName qn_nm = parse_weight_name(plan.weight_name(qn));
    const ParsedWeightName kn_nm = parse_weight_name(plan.weight_name(kn));
    if (qn_nm.field != "q_norm" || qn_nm.layer != layer ||
        kn_nm.field != "k_norm" || kn_nm.layer != layer ||
        rope.param0 !=
            static_cast<std::uint32_t>(PieForwardRopeKind::Standard) ||
        static_cast<int>(append.param0) != layer) {
        throw std::runtime_error(
            "declared forward: fused decode-QKV peephole matched op kinds "
            "but not their payloads at layer " + std::to_string(layer) +
            "; the trace's shape drifted from family.rs's fused_qkv "
            "layer body");
    }
    return i + 5;
}

}  // namespace

LlamaLikeDeclaredPlan build_llama_like_declared_plan(
    const HfConfig& cfg,
    const LlamaLikeForwardCfg& fwd_cfg,
    const Qwen3Weights& w)
{
    LlamaLikeDeclaredPlan out;

    // Representability gate: everything the v0 trace has no vocabulary for
    // returns empty and the model keeps the hand-written path. Each line
    // names the hand-written feature it stands in for.
    if (fwd_cfg.rope_kind != RopeKind::Standard) return out;   // YaRN/M-RoPE
    if (fwd_cfg.norm_placement != NormPlacement::Pre) return out;  // OLMo-3
    if (fwd_cfg.use_qkv_bias) return out;                      // Qwen-2 bias
    if (fwd_cfg.tp_size > 1) return out;                       // all-reduces
    // Padded head_dim (Phi-3-mini, 96 → 128) is admitted: the pad/strip
    // launches around KV-write/attention are emitter knowledge (the trace
    // speaks the logical head_dim throughout), handled in the executor
    // exactly as the hand-written `head_dim_padded` branches.
    if (w.layers.empty() ||
        w.layers.size() != static_cast<std::size_t>(cfg.num_hidden_layers)) {
        return out;
    }
    const bool fused_qkv = w.layers[0].qkv_proj_fused != nullptr;
    for (const auto& layer : w.layers) {
        // Quantized projections route through QuantMeta WeightViews the
        // trace does not describe; and a mixed fused/unfused binding would
        // make the single `fused_qkv` fact a lie.
        if (layer.q_proj_quant || layer.k_proj_quant || layer.v_proj_quant ||
            layer.o_proj_quant || layer.gate_proj_quant ||
            layer.up_proj_quant || layer.down_proj_quant) {
            return out;
        }
        if ((layer.qkv_proj_fused != nullptr) != fused_qkv) return out;
    }
    if (fwd_cfg.use_qk_norm) {
        // The trace's RmsnormPerHead is the per-head convention (weight
        // shape [head_dim]); the global-norm convention (OLMo-2 7B+) is a
        // different op the family does not declare yet.
        const DeviceTensor* qn = w.layers[0].q_norm;
        const DeviceTensor* kn = w.layers[0].k_norm;
        const bool per_head =
            qn != nullptr && kn != nullptr &&
            qn->shape().size() == 1 && qn->shape()[0] == cfg.head_dim &&
            kn->shape().size() == 1 && kn->shape()[0] == cfg.head_dim;
        if (!per_head) return out;
    }

    pie_forward::PieForwardLlamaLikeFacts facts{};
    facts.hidden = static_cast<std::uint32_t>(cfg.hidden_size);
    facts.layers = static_cast<std::uint32_t>(cfg.num_hidden_layers);
    facts.q_heads = static_cast<std::uint32_t>(cfg.num_attention_heads);
    facts.kv_heads = static_cast<std::uint32_t>(cfg.num_key_value_heads);
    facts.head_dim = static_cast<std::uint32_t>(cfg.head_dim);
    facts.intermediate = static_cast<std::uint32_t>(cfg.intermediate_size);
    facts.vocab = static_cast<std::uint32_t>(cfg.vocab_size);
    facts.rope = static_cast<std::uint32_t>(PieForwardRopeKind::Standard);
    facts.norm_variant =
        static_cast<std::uint32_t>(PieForwardNormVariant::Plain);
    facts.qk_norm = fwd_cfg.use_qk_norm ? 1 : 0;
    facts.fused_qkv = fused_qkv ? 1 : 0;
    // A binding fact, like fused_qkv: bind_llama_like aliases lm_head to
    // embed when the checkpoint ties them, so pointer equality is the truth.
    facts.tied_embeddings = (w.lm_head == w.embed) ? 1 : 0;

    out.plan = pie_forward::ForwardPlan::trace_llama_like(facts);
    out.fused_qkv = fused_qkv;
    return out;
}

void llama_like_forward_declared(
    const LlamaLikeDeclaredPlan& declared,
    const Qwen3Weights& w,
    const HfConfig& cfg,
    const LlamaLikeForwardCfg& fwd_cfg,
    const LlamaLikePlanState& plan_state,
    Workspace& ws,
    KvCache& cache,
    AttentionWorkspace& attn_ws,
    ops::CublasHandle& cublas,
    const std::int32_t* token_ids,
    const std::int32_t* positions,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    int total_tokens,
    int num_requests,
    bool is_pure_decode,
    const std::int32_t* logit_row_indices_d,
    int num_logit_rows,
    const std::uint32_t* w_page_d,
    const std::uint32_t* w_off_d,
    const std::uint8_t* row_valid_d,
    bool has_write_desc,
    int runtime_window_left)
{
    const pie_forward::ForwardPlan& plan = declared.plan;
    // Parity-harness visibility (PIE_HOOK_PREFIX_TRACE's pattern): without
    // it a silent fallback to the hand-written path would be
    // indistinguishable from a passing A/B run.
    if (std::getenv("PIE_DECLARED_FORWARD_TRACE")) {
        std::fprintf(stderr,
                     "[declared-forward] N=%d R=%d decode=%d ops=%zu\n",
                     total_tokens, num_requests, is_pure_decode ? 1 : 0,
                     plan.op_count());
    }
    const int N = total_tokens;
    const int R = num_requests;
    const int H = cfg.hidden_size;
    const int Hq = cfg.num_attention_heads * cfg.head_dim;
    const int Hk = cfg.num_key_value_heads * cfg.head_dim;
    const int I = cfg.intermediate_size;
    const int V = cfg.vocab_size;
    const int d = cfg.head_dim;
    const int dk = cfg.head_dim_kernel;  // padded HEAD_DIM the attention
                                         // kernel runs at (Phi-3: 96 → 128)
    const bool head_dim_padded = (d != dk);
    const int num_q_heads = cfg.num_attention_heads;
    const int num_kv_heads = cfg.num_key_value_heads;
    const float eps = cfg.rms_norm_eps;
    // Inherit cublas's stream so every launch lands on the captured graph,
    // for the reason llama_like_forward_paged states at its stream setup.
    cudaStream_t stream = cublas.stream();
    // With padding the attention kernel runs at `dk` but the softmax must
    // stay scaled to the real head dim — `1/sqrt(d)`, the hand-written
    // override. Unpadded, -1 lets the dispatch pick `1/sqrt(dk)` (== d).
    const float sm_scale_override = head_dim_padded
        ? (1.0f / std::sqrt(static_cast<float>(d)))
        : -1.f;
    // Padded Q/K/V staging (the hand-written `attn_q`/... indirection):
    // GEMM in/out buffers stay at `d`; the KV write and attention consume
    // the zero-padded `dk` copies, and the o_proj reads the stripped
    // output. All identity when the model's head_dim is a dispatch value.
    void* const attn_q = head_dim_padded ? ws.q_padded.data() : ws.q.data();
    void* const attn_k = head_dim_padded ? ws.k_padded.data() : ws.k.data();
    void* const attn_v = head_dim_padded ? ws.v_padded.data() : ws.v.data();
    void* const attn_out_buf =
        head_dim_padded ? ws.attn_out_padded.data() : ws.attn_out.data();

    // Attention path choice: the same booleans the hand-written body derives
    // from the plan_state the (unchanged) prepare hook filled. Custom-mask
    // and hook branches are absent because the caller's gate excluded them.
    const bool use_xqa_decode_path =
        is_pure_decode && plan_state.use_xqa_decode;
    const bool use_decode_path =
        is_pure_decode &&
        (!fwd_cfg.force_prefill_path || use_xqa_decode_path);
    const bool use_prefill_decode_path =
        use_decode_path && !use_xqa_decode_path &&
        plan_state.use_prefill_decode_plan;
    const ops::DecodePlanCache* decode_plan =
        plan_state.decode_plan ? plan_state.decode_plan.get() : nullptr;
    const ops::PrefillPlanCache* prefill_decode_plan =
        plan_state.prefill_decode_plan ? plan_state.prefill_decode_plan.get()
                                       : nullptr;
    const ops::PrefillPlanCache* prefill_plan =
        plan_state.prefill_plan ? plan_state.prefill_plan.get() : nullptr;

    if (use_xqa_decode_path) {
        ops::prepare_attention_xqa_decode_bf16(
            kv_page_indices, kv_page_indptr, kv_last_page_lens,
            R, cache.page_size(), plan_state.xqa_max_pages_per_seq,
            attn_ws, stream);
    }

    // Whether the gate_up Matmul took the fused binding; decides which
    // swiglu kernel the following Swiglu op launches (the hand-written
    // `use_fused_gu` pairing).
    bool gate_up_used_fused = false;

    // Rope-table state for the fused decode-QKV peephole: built once per
    // fire on the first layer whose peephole fires (the hand-written
    // `rope_table_ready` latch), reused by every later layer. Stays null
    // when the workspace carries no table — the fused kernel then derives
    // cos/sin from theta itself, exactly as the hand-written branch does.
    bool rope_table_ready = false;
    const float* rope_table = nullptr;

    const std::size_t op_count = plan.op_count();
    for (std::size_t i = 0; i < op_count; ++i) {
        const PieForwardOp& op = plan.op(i);
        switch (op.kind) {
        case PieForwardOpKind::Embed: {
            const std::string_view name = plan.weight_name(op);
            if (name != "embed") throw_unknown_weight(name);
            kernels::launch_embed_bf16(
                token_ids, require(w.embed, name)->data(), ws.y.data(),
                N, H, V, stream);
            break;
        }
        case PieForwardOpKind::Rmsnorm: {
            if (op.param0 !=
                static_cast<std::uint32_t>(PieForwardNormVariant::Plain)) {
                throw std::runtime_error(
                    "declared forward: only the Plain rmsnorm variant is "
                    "emitted (Gemma folding is a different arithmetic)");
            }
            const std::string_view name = plan.weight_name(op);
            const ParsedWeightName nm = parse_weight_name(name);
            if (nm.field == "attn_norm") {
                const auto& layer = layer_of(w, nm, name);
                kernels::launch_rmsnorm_bf16(
                    ws.y.data(), require(layer.attn_norm, name)->data(),
                    ws.norm_x.data(), N, H, eps, stream);
            } else if (nm.field == "mlp_norm") {
                const auto& layer = layer_of(w, nm, name);
                kernels::launch_rmsnorm_bf16(
                    ws.y.data(), require(layer.mlp_norm, name)->data(),
                    ws.norm_y.data(), N, H, eps, stream);
            } else if (nm.layer < 0 && nm.field == "final_norm") {
                // Deferred to LmHead: the hand-written epilogue interleaves
                // the final norm with the logit-row gather (norm is row-wise,
                // so gather-then-norm equals norm-then-gather), and copying
                // that block whole is what keeps the two paths bit-identical.
                require(w.final_norm, name);
            } else {
                throw_unknown_weight(name);
            }
            break;
        }
        case PieForwardOpKind::Matmul: {
            const std::string_view name = plan.weight_name(op);
            const ParsedWeightName nm = parse_weight_name(name);
            const auto& layer = layer_of(w, nm, name);
            const float beta = op.param0 != 0 ? 1.f : 0.f;
            if (nm.field == "qkv") {
                // Fused decode-QKV peephole. The trace deliberately stays
                // unfused: fusion is the EMITTER's decision, never the
                // author's or the trace's (pie-application-plan.md §5.1 —
                // "a fused edge cannot be a merge point", so fusion and
                // merging must be chosen together, by the planner; and
                // stage1-notes.md, where `fused_decode_qkv_post` became a
                // row-count gate for exactly that reason). This is the
                // knowledge the trace does not carry, so the emitter
                // re-derives it here: when the adjacency Matmul(qkv) +
                // SplitQkv + RmsnormPerHead x2 + Rope + KvAppend holds AND
                // the hand-written predicate holds, launch the ONE fused
                // kernel the hand-written path would.
                //
                // The predicate is the hand-written `fused_decode_qkv_post`
                // (llama_like.cpp), term for term. Two of its terms are
                // resolved by this path's caller gate rather than dropped:
                //   * `fast_rows > 0` — hooks are null here (gate), so
                //     Stage 1's hook-free prefix is every row: fast_rows ==
                //     R, the unfused tail is empty, and the all-fused case
                //     is the only one this executor needs.
                //   * `!has_custom_mask` — the gate excludes custom masks.
                const bool q_norm_is_per_head =
                    layer.q_norm && layer.q_norm->shape().size() == 1 &&
                    layer.q_norm->shape()[0] == d;
                const bool k_norm_is_per_head =
                    layer.k_norm && layer.k_norm->shape().size() == 1 &&
                    layer.k_norm->shape()[0] == d;
                const bool use_fused_qkv =
                    layer.qkv_proj_fused != nullptr && !ws.qkv_fused.empty();
                const bool fused_decode_qkv_post =
                    use_fused_qkv &&
                    R > 0 &&  // fast_rows > 0, with fast_rows == R (above)
                    decode_fused_post_enabled() &&
                    is_pure_decode &&
                    (!has_write_desc ||
                     (w_page_d != nullptr && w_off_d != nullptr)) &&
                    cache.format().is_native_bf16() &&
                    cfg.head_dim == cfg.head_dim_kernel &&  // !padded
                    !fwd_cfg.use_qkv_bias &&
                    fwd_cfg.use_qk_norm &&
                    q_norm_is_per_head && k_norm_is_per_head &&
                    fwd_cfg.rope_kind == RopeKind::Standard;
                ops::gemm_act_x_w(cublas.handle(),
                    ws.norm_x.data(),
                    ops::WeightView(*require(layer.qkv_proj_fused, name)),
                    ws.qkv_fused.data(), N, Hq + 2 * Hk, H);
                if (fused_decode_qkv_post) {
                    if (!rope_table_ready && !ws.rope_table.empty()) {
                        kernels::launch_rope_standard_table(
                            positions,
                            static_cast<float*>(ws.rope_table.data()),
                            N, d, cfg.rope_theta, stream);
                        rope_table =
                            static_cast<const float*>(ws.rope_table.data());
                        rope_table_ready = true;
                    }
                    kernels::launch_qkv_decode_qk_norm_rope_write_kv_bf16(
                        ws.qkv_fused.data(),
                        ws.q.data(),
                        cache.k(nm.layer), cache.v(nm.layer),
                        layer.q_norm->data(), layer.k_norm->data(),
                        positions,
                        rope_table,
                        kv_page_indices, kv_page_indptr, kv_last_page_lens,
                        has_write_desc ? w_page_d : nullptr,
                        has_write_desc ? w_off_d : nullptr,
                        row_valid_d,
                        R, num_q_heads, num_kv_heads, d,
                        cache.page_size(), cache.hnd_layout(),
                        cfg.rope_theta, eps, stream);
                    // The kernel owns everything through the KV write;
                    // advance past those plan ops (validated, or throw).
                    i = skip_fused_decode_qkv_ops(plan, i, nm.layer);
                }
            } else if (nm.field == "q_proj") {
                ops::gemm_act_x_w(cublas.handle(),
                    ws.norm_x.data(),
                    make_weight_view(require(layer.q_proj, name),
                                     layer.q_proj_quant),
                    ws.q.data(), N, Hq, H);
            } else if (nm.field == "k_proj") {
                ops::gemm_act_x_w(cublas.handle(),
                    ws.norm_x.data(),
                    make_weight_view(require(layer.k_proj, name),
                                     layer.k_proj_quant),
                    ws.k.data(), N, Hk, H);
            } else if (nm.field == "v_proj") {
                ops::gemm_act_x_w(cublas.handle(),
                    ws.norm_x.data(),
                    make_weight_view(require(layer.v_proj, name),
                                     layer.v_proj_quant),
                    ws.v.data(), N, Hk, H);
            } else if (nm.field == "o_proj") {
                // Residual accumulate folded into the GEMM (beta from the
                // trace's beta_one), exactly the hand-written T==1 branch.
                ops::gemm_act_x_w(cublas.handle(),
                    ws.attn_out.data(),
                    make_weight_view(require(layer.o_proj, name),
                                     layer.o_proj_quant),
                    ws.y.data(), N, H, Hq, beta);
            } else if (nm.field == "gate_up") {
                // The trace declares one packed matmul either way; whether
                // the binding materialised it fused is this emitter's call,
                // the same dispatch the hand-written `use_fused_gu` makes.
                gate_up_used_fused =
                    layer.gate_up_proj_fused != nullptr &&
                    !ws.gate_up_fused.empty();
                if (gate_up_used_fused) {
                    ops::gemm_act_x_w(cublas.handle(),
                        ws.norm_y.data(),
                        ops::WeightView(*layer.gate_up_proj_fused),
                        ws.gate_up_fused.data(), N, 2 * I, H);
                } else {
                    ops::gemm_act_x_w(cublas.handle(),
                        ws.norm_y.data(),
                        make_weight_view(require(layer.gate_proj, name),
                                         layer.gate_proj_quant),
                        ws.gate.data(), N, I, H);
                    ops::gemm_act_x_w(cublas.handle(),
                        ws.norm_y.data(),
                        make_weight_view(require(layer.up_proj, name),
                                         layer.up_proj_quant),
                        ws.up.data(), N, I, H);
                }
            } else if (nm.field == "down") {
                ops::gemm_act_x_w(cublas.handle(),
                    ws.gate.data(),
                    make_weight_view(require(layer.down_proj, name),
                                     layer.down_proj_quant),
                    ws.y.data(), N, H, I, beta);
            } else {
                throw_unknown_weight(name);
            }
            break;
        }
        case PieForwardOpKind::SplitQkv: {
            kernels::launch_split_qkv_bf16(
                ws.qkv_fused.data(),
                ws.q.data(), ws.k.data(), ws.v.data(),
                N, Hq, Hk, stream);
            break;
        }
        case PieForwardOpKind::RmsnormPerHead: {
            // Peephole: RmsnormPerHead(q) + RmsnormPerHead(k) + Rope
            // adjacency is what the hand-written `fuse_qk_norm_rope` branch
            // computes with ONE kernel, and bf16 rounding differs between
            // the fused kernel and a norm+rope pair — parity requires the
            // same launch, not just the same math.
            const bool fuse =
                i + 2 < op_count &&
                plan.op(i + 1).kind == PieForwardOpKind::RmsnormPerHead &&
                plan.op(i + 2).kind == PieForwardOpKind::Rope &&
                plan.op(i + 2).param0 ==
                    static_cast<std::uint32_t>(PieForwardRopeKind::Standard);
            if (fuse) {
                const std::string_view q_name = plan.weight_name(op);
                const std::string_view k_name =
                    plan.weight_name(plan.op(i + 1));
                const ParsedWeightName q_nm = parse_weight_name(q_name);
                const ParsedWeightName k_nm = parse_weight_name(k_name);
                if (q_nm.field != "q_norm" || k_nm.field != "k_norm") {
                    throw_unknown_weight(q_nm.field != "q_norm" ? q_name
                                                                : k_name);
                }
                const auto& layer = layer_of(w, q_nm, q_name);
                kernels::launch_qk_rmsnorm_rope_bf16(
                    ws.q.data(), ws.k.data(),
                    require(layer.q_norm, q_name)->data(),
                    require(layer.k_norm, k_name)->data(),
                    positions, N, num_q_heads, num_kv_heads, d,
                    cfg.rope_theta, eps, stream);
                i += 2;  // consumed the second RmsnormPerHead and the Rope
                break;
            }
            // Standalone per-head norm: in place, one row per head — the
            // hand-written `rmsnorm_qk` per-head branch.
            const std::string_view name = plan.weight_name(op);
            const ParsedWeightName nm = parse_weight_name(name);
            const auto& layer = layer_of(w, nm, name);
            if (nm.field == "q_norm") {
                kernels::launch_rmsnorm_bf16(
                    ws.q.data(), require(layer.q_norm, name)->data(),
                    ws.q.data(), N * num_q_heads, d, eps, stream);
            } else if (nm.field == "k_norm") {
                kernels::launch_rmsnorm_bf16(
                    ws.k.data(), require(layer.k_norm, name)->data(),
                    ws.k.data(), N * num_kv_heads, d, eps, stream);
            } else {
                throw_unknown_weight(name);
            }
            break;
        }
        case PieForwardOpKind::Rope: {
            // Reached only when neither peephole consumed it (no qk-norm
            // in the trace, so the fused decode-QKV predicate — which
            // requires qk-norm — cannot hold either). Positions go straight
            // to the kernel, as the hand-written `apply_rope` does for
            // RopeKind::Standard; the rope_table serves only the fused
            // decode postprocess above.
            if (op.param0 !=
                static_cast<std::uint32_t>(PieForwardRopeKind::Standard)) {
                throw std::runtime_error(
                    "declared forward: only standard rope is emitted "
                    "(build gate admits nothing else)");
            }
            kernels::launch_rope_bf16(
                ws.q.data(), ws.k.data(), positions,
                N, num_q_heads, num_kv_heads, d,
                cfg.rope_theta, stream);
            break;
        }
        case PieForwardOpKind::KvAppend: {
            // Padded head_dim: the cache stores `dk`-wide cells, so Q/K/V
            // are zero-padded to the `*_padded` staging buffers first —
            // the hand-written pad block, same launch order (q, k, v),
            // placed exactly where it sits there: after rope, before the
            // write. Identity when unpadded (`attn_*` alias ws.q/k/v).
            if (head_dim_padded) {
                kernels::launch_pad_head_dim_bf16(
                    ws.q.data(), attn_q, N, num_q_heads, d, dk, stream);
                kernels::launch_pad_head_dim_bf16(
                    ws.k.data(), attn_k, N, num_kv_heads, d, dk, stream);
                kernels::launch_pad_head_dim_bf16(
                    ws.v.data(), attn_v, N, num_kv_heads, d, dk, stream);
            }
            auto kv_view = cache.layer_view(static_cast<int>(op.param0));
            if (has_write_desc) {
                // Explicit-descriptor write, N cells (one per query token) —
                // the hand-written `has_write_desc` branch, including why N
                // and not R (llama_like.cpp's B2 comment). Graph-replayed
                // decode fires always take this branch: their captures
                // record the w_page/w_off path so padded rows stay steerable
                // at replay time.
                kernels::launch_write_kv_explicit_bf16(
                    kv_view,
                    attn_k, attn_v,
                    w_page_d, w_off_d, N, stream, row_valid_d);
            } else {
                // Page-derived append (the hand-written non-write-desc
                // branch): position re-derived from the page table.
                kernels::launch_write_kv_to_pages(
                    kv_view,
                    attn_k, attn_v,
                    qo_indptr, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens,
                    N, R, stream, row_valid_d);
            }
            break;
        }
        case PieForwardOpKind::Attention: {
            const int L = static_cast<int>(op.param0);
            auto kv_view = cache.layer_view(L);
            // Same per-layer window resolution as the hand-written body;
            // runtime_window_left is -2 on this path (gate) so the
            // config-driven values decide.
            const int layer_window_left = runtime_window_left >= -1
                ? runtime_window_left
                : (!fwd_cfg.per_layer_window_left.empty() &&
                   L < static_cast<int>(fwd_cfg.per_layer_window_left.size()))
                    ? fwd_cfg.per_layer_window_left[L]
                    : fwd_cfg.sliding_window;
            if (use_xqa_decode_path) {
                ops::launch_attention_xqa_decode_bf16_prepared(
                    attn_q, kv_view.k_bf16_pages, kv_view.v_bf16_pages,
                    attn_out_buf,
                    R, num_q_heads, num_kv_heads, dk,
                    cache.page_size(), plan_state.xqa_max_pages_per_seq,
                    attn_ws, stream, sm_scale_override);
            } else if (use_prefill_decode_path) {
                if (prefill_decode_plan == nullptr) {
                    throw std::runtime_error(
                        "declared forward: prefill-decode path has no plan");
                }
                const int num_pages_in_batch = kv_page_indptr_h[R];
                kernels::launch_dequant_kv_cache_layer_to_bf16_active(
                    kv_view, kv_page_indices, num_pages_in_batch, stream);
                ops::dispatch_attention_flashinfer_prefill_bf16(
                    *prefill_decode_plan,
                    attn_q, kv_view.k_bf16_pages, kv_view.v_bf16_pages,
                    attn_out_buf,
                    qo_indptr, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens,
                    attn_ws, stream, /*logits_soft_cap=*/0.f,
                    sm_scale_override);
            } else if (use_decode_path) {
                if (decode_plan == nullptr) {
                    throw std::runtime_error(
                        "declared forward: decode path has no plan");
                }
                ops::dispatch_attention_flashinfer_decode(
                    *decode_plan,
                    attn_q, kv_view, attn_out_buf,
                    kv_page_indices, kv_page_indptr, kv_last_page_lens,
                    attn_ws, stream, layer_window_left,
                    /*logits_soft_cap=*/0.f, sm_scale_override);
            } else if (plan_state.use_prefill_plan && prefill_plan != nullptr) {
                const int num_pages_in_batch = kv_page_indptr_h[R];
                kernels::launch_dequant_kv_cache_layer_to_bf16_active(
                    kv_view, kv_page_indices, num_pages_in_batch, stream);
                ops::dispatch_attention_flashinfer_prefill_bf16(
                    *prefill_plan,
                    attn_q, kv_view.k_bf16_pages, kv_view.v_bf16_pages,
                    attn_out_buf,
                    qo_indptr, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens,
                    attn_ws, stream, /*logits_soft_cap=*/0.f,
                    sm_scale_override);
            } else {
                ops::launch_attention_flashinfer_prefill(
                    attn_q, kv_view, attn_out_buf,
                    qo_indptr, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens,
                    qo_indptr_h, kv_page_indptr_h,
                    N, R, num_q_heads, attn_ws, stream, layer_window_left,
                    /*logits_soft_cap=*/0.f, sm_scale_override);
            }
            // Strip the trailing pad cols before the o_proj GEMM reads
            // `[N, num_q*d]` — the hand-written post-attention strip.
            if (head_dim_padded) {
                kernels::launch_strip_head_dim_bf16(
                    attn_out_buf, ws.attn_out.data(),
                    N, num_q_heads, d, dk, stream);
            }
            break;
        }
        case PieForwardOpKind::Swiglu: {
            if (gate_up_used_fused) {
                kernels::launch_chunked_swiglu_bf16(
                    ws.gate_up_fused.data(), ws.gate.data(), N, I, stream);
            } else {
                kernels::launch_swiglu_bf16(
                    ws.gate.data(), ws.up.data(), ws.gate.data(),
                    N * I, stream);
            }
            break;
        }
        case PieForwardOpKind::LmHead: {
            if (!fwd_cfg.emit_logits) break;
            const std::string_view name = plan.weight_name(op);
            // Tied embeddings trace the lm head as "embed"; either way the
            // binding already aliased `w.lm_head` accordingly.
            const DeviceTensor* lm_head =
                name == "embed" ? require(w.embed, name)
                : name == "lm_head" ? require(w.lm_head, name)
                : nullptr;
            if (lm_head == nullptr) throw_unknown_weight(name);
            // The hand-written epilogue, copied whole (T==1, no fused-AR
            // final norm on this path): compact-logit fires gather the
            // sampled rows first, then final-norm just those; full emits
            // recompute the final norm from `ws.y` for the reason
            // llama_like.cpp's comment gives (§6.2 staleness).
            const bool compact_logits =
                logit_row_indices_d != nullptr && num_logit_rows > 0 &&
                num_logit_rows < N;
            const void* lm_head_input = nullptr;
            int lm_head_rows = N;
            if (compact_logits) {
                kernels::launch_gather_bf16_rows(
                    static_cast<const std::uint16_t*>(ws.y.data()),
                    logit_row_indices_d,
                    static_cast<std::uint16_t*>(ws.norm_x.data()),
                    num_logit_rows, H, stream);
                kernels::launch_rmsnorm_bf16(
                    ws.norm_x.data(), w.final_norm->data(),
                    ws.norm_y.data(), num_logit_rows, H, eps, stream);
                lm_head_input = ws.norm_y.data();
                lm_head_rows = num_logit_rows;
            } else {
                kernels::launch_rmsnorm_bf16(
                    ws.y.data(), w.final_norm->data(), ws.norm_y.data(),
                    N, H, eps, stream);
                lm_head_input = ws.norm_y.data();
            }
            ops::gemm_act_x_w(cublas.handle(),
                lm_head_input, ops::WeightView(*lm_head),
                ws.logits.data(), lm_head_rows, V, H);
            break;
        }
        default:
            throw std::runtime_error(
                "declared forward: op kind " +
                std::to_string(static_cast<std::uint32_t>(op.kind)) +
                " has no emission rule");
        }
    }
}

}  // namespace pie_cuda_driver::model
