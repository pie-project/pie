#include "model/gemma4/declared_forward.hpp"

#include <cmath>
#include <stdexcept>
#include <vector>

#include "kernels/gather_rows.hpp"
#include "kernels/rmsnorm.hpp"
#include "kernels/rope.hpp"
#include "kernels/scalar_mul.hpp"
#include "kernels/softcap.hpp"
#include "kernels/split_packed.hpp"
#include "kernels/swiglu.hpp"
#include "kernels/embed.hpp"
#include "kernels/kv_paged.hpp"
#include "ops/attention_flashinfer.hpp"
#include "ops/gemm.hpp"
#include <string>
#include <string_view>

namespace pie_cuda_driver::model {

namespace {

// The launcher registry — every kernel a gemma-4 class trace may STATE,
// one enum value per symbol. Deliberately EXHAUSTIVE against the traced
// decode plan: `gemma4_validate_stated_kernels` walks the plan at load
// and a symbol outside this list is a model-load failure, so this list
// and `family::gemma4_cuda` are two spellings of one vocabulary.
enum class G4Kernel {
    QkvPackedPost,
    QkRmsnormRopeRounded,
    RopeQOnly,
    RopeQOnlyPartial,
    RmsnormNoScale,
    WriteKvToPages,
    AttnFlashinferDecode,
    AttnFlashinferPrefill,
    GegluTanh,
    ChunkedGegluTanh,
    NormResidualScaleNorm,
    NormResidualAdd,
    ScalarMul,
    TransposeNldToLnd,
    LogitSoftcap,
    ResidualAdd,
};

G4Kernel resolve_g4_kernel(std::string_view k) {
    if (k == "launch_qkv_packed_qk_norm_rope_vnorm_write_kv_bf16")
        return G4Kernel::QkvPackedPost;
    if (k == "launch_qk_rmsnorm_rope_bf16_rounded")
        return G4Kernel::QkRmsnormRopeRounded;
    if (k == "launch_rope_bf16") return G4Kernel::RopeQOnly;
    if (k == "launch_rope_partial_bf16") return G4Kernel::RopeQOnlyPartial;
    if (k == "launch_rmsnorm_no_scale_bf16") return G4Kernel::RmsnormNoScale;
    if (k == "launch_write_kv_to_pages") return G4Kernel::WriteKvToPages;
    if (k == "dispatch_attention_flashinfer_decode")
        return G4Kernel::AttnFlashinferDecode;
    if (k == "dispatch_attention_flashinfer_prefill_bf16")
        return G4Kernel::AttnFlashinferPrefill;
    if (k == "launch_geglu_tanh_bf16") return G4Kernel::GegluTanh;
    if (k == "launch_chunked_geglu_tanh_bf16") return G4Kernel::ChunkedGegluTanh;
    if (k == "launch_rmsnorm_residual_add_scale_rmsnorm_bf16")
        return G4Kernel::NormResidualScaleNorm;
    if (k == "launch_rmsnorm_residual_add_bf16") return G4Kernel::NormResidualAdd;
    if (k == "launch_scalar_mul_bf16") return G4Kernel::ScalarMul;
    if (k == "launch_transpose_bf16_nld_to_lnd")
        return G4Kernel::TransposeNldToLnd;
    if (k == "launch_logit_softcap_bf16") return G4Kernel::LogitSoftcap;
    if (k == "launch_residual_add_bf16") return G4Kernel::ResidualAdd;
    throw std::runtime_error(
        "declared gemma4: stated kernel '" + std::string(k) +
        "' is not in this executor's registry (the trace and the driver "
        "drifted)");
}

}  // namespace

void gemma4_validate_stated_kernels(const pie_forward::ForwardPlan& plan) {
    const std::size_t n = plan.op_count();
    for (std::size_t i = 0; i < n; ++i) {
        const auto& op = plan.op(i);
        if (op.kind != pie_forward::PieForwardOpKind::Launch) continue;
        (void)resolve_g4_kernel(plan.weight_name(op));
    }
}

}  // namespace pie_cuda_driver::model

namespace pie_cuda_driver::model {

namespace {

using pie_forward::PieForwardOp;
using pie_forward::PieForwardOpKind;

[[noreturn]] void throw_drift(const std::string& what) {
    throw std::runtime_error("declared gemma4: " + what +
                             " (the trace and the driver drifted)");
}

// A plan weight name split into layer and field — llama_like's parse.
struct ParsedName {
    int layer = -1;
    std::string_view field;
};

ParsedName parse_name(std::string_view name) {
    constexpr std::string_view prefix = "layer.";
    if (name.substr(0, prefix.size()) != prefix) return {-1, name};
    const std::size_t dot = name.find('.', prefix.size());
    if (dot == std::string_view::npos) throw_drift("weight name '" +
                                                   std::string(name) + "'");
    int layer = 0;
    for (std::size_t i = prefix.size(); i < dot; ++i) {
        if (name[i] < '0' || name[i] > '9') {
            throw_drift("weight name '" + std::string(name) + "'");
        }
        layer = layer * 10 + (name[i] - '0');
    }
    return {layer, name.substr(dot + 1)};
}

// The binder. gemma-4's trace names its weights after the driver's own
// fields, so this is a map and not a translation.
const DeviceTensor* bind(const Gemma4Weights& w, std::string_view name) {
    const ParsedName nm = parse_name(name);
    if (nm.layer < 0) {
        if (nm.field == "embed") return w.embed;
        if (nm.field == "embed_per_layer") return w.embed_per_layer;
        if (nm.field == "ple_model_proj") return w.ple_model_proj;
        if (nm.field == "ple_model_norm") return w.ple_model_norm;
        if (nm.field == "final_norm") return w.final_norm;
        if (nm.field == "lm_head") return w.lm_head;
        throw_drift("unknown model weight '" + std::string(name) + "'");
    }
    if (nm.layer >= static_cast<int>(w.layers.size())) {
        throw_drift("weight names layer " + std::to_string(nm.layer));
    }
    const Gemma4LayerWeights& l = w.layers[static_cast<std::size_t>(nm.layer)];
    if (nm.field == "attn_norm") return l.attn_norm_pre;
    if (nm.field == "post_attn_norm") return l.attn_norm_post;
    if (nm.field == "pre_ffw_norm") return l.mlp_norm_pre;
    if (nm.field == "post_ffw_norm") return l.mlp_norm_post;
    if (nm.field == "qkv") return l.qkv_proj_fused;
    if (nm.field == "q_proj") return l.q_proj;
    if (nm.field == "k_proj") return l.k_proj;
    if (nm.field == "v_proj") return l.v_proj;
    if (nm.field == "o_proj") return l.o_proj;
    if (nm.field == "q_norm") return l.q_norm;
    if (nm.field == "k_norm") return l.k_norm;
    if (nm.field == "gate_up") return l.gate_up_proj_fused;
    if (nm.field == "down") return l.down_proj;
    if (nm.field == "ple_gate") return l.ple_input_gate;
    if (nm.field == "ple_proj") return l.ple_projection;
    if (nm.field == "ple_norm") return l.ple_norm;
    throw_drift("unknown layer weight '" + std::string(name) + "'");
}

const DeviceTensor& require(const Gemma4Weights& w, std::string_view name) {
    const DeviceTensor* t = bind(w, name);
    if (t == nullptr) {
        throw std::runtime_error("declared gemma4: weight '" +
                                 std::string(name) +
                                 "' is named by the trace but not bound");
    }
    return *t;
}

}  // namespace

bool gemma4_forward_declared(
    const Gemma4DeclaredPlan& declared,
    const Gemma4Weights& w,
    const HfConfig& cfg,
    const Gemma4ForwardCfg& fwd_cfg,
    Workspace& ws,
    Gemma4MoeMlpWorkspace& moe_ws,
    KvCache& cache,
    AttentionWorkspace& attn_ws,
    ops::CublasHandle& cublas,
    const std::int32_t* token_ids,
    const std::int32_t* positions,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    const std::uint32_t* kv_page_indptr_h,
    int total_tokens,
    int num_requests,
    const std::uint8_t* row_valid_d,
    const std::int32_t* logit_row_indices_d,
    int num_logit_rows)
{
    if (!declared.usable) return false;
    const pie_forward::ForwardPlan& plan = declared.decode;

    const int N = total_tokens;
    const int R = num_requests;
    const int H = cfg.hidden_size;
    const int I = cfg.intermediate_size;
    const int V = cfg.vocab_size;
    const int L = cfg.num_hidden_layers;
    const int ple_dim = cfg.gemma_hidden_size_per_layer_input;
    const float eps = cfg.rms_norm_eps;
    cudaStream_t stream = cublas.stream();

    // The PLE relay's two buffers. Without them the prologue has nowhere
    // to land, and the hand-written pass allocates on the fly — a shape
    // this drive does not reproduce, so it declines instead.
    if (moe_ws.ple_token.empty() || moe_ws.ple_proj.empty()) return false;
    if (!cache.format().is_native_bf16()) return false;

    // The value the previous statement produced, by slot. gemma-4's
    // buffers are the hand-written pass's, so the drive threads them the
    // way that pass does rather than allocating an arena: `ws.y` is the
    // residual stream, `ws.norm_x` the block scratch.
    void* per_layer_token = moe_ws.ple_token.data();

    const auto execute_op = [&](const PieForwardOp& op) {
        switch (op.kind) {
        case PieForwardOpKind::Embed: {
            const std::string_view name = plan.weight_name(op);
            if (name == "embed") {
                kernels::launch_embed_bf16(token_ids, require(w, name).data(),
                                           ws.y.data(), N, H, V, stream);
            } else {
                kernels::launch_embed_bf16(
                    token_ids, require(w, name).data(), per_layer_token,
                    N, L * ple_dim, V, stream);
            }
            break;
        }
        default:
            throw_drift("op kind " +
                        std::to_string(static_cast<std::uint32_t>(op.kind)) +
                        " has no emission rule");
        }
    };

    (void)execute_op;
    (void)fwd_cfg;
    (void)attn_ws;
    (void)cublas;
    (void)positions;
    (void)kv_page_indices;
    (void)kv_page_indptr;
    (void)kv_last_page_lens;
    (void)kv_page_indptr_h;
    (void)row_valid_d;
    (void)logit_row_indices_d;
    (void)num_logit_rows;
    (void)R;
    (void)I;
    (void)eps;
    (void)moe_ws;
    // The arms below this line are the remaining work; until they exist
    // the drive declines rather than half-executing a fire.
    return false;
}

}  // namespace pie_cuda_driver::model
