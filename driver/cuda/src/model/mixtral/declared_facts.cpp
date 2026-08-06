#include "model/mixtral/declared_forward.hpp"

#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

namespace pie_cuda_driver::model {

namespace {

using pie_forward::ForwardPlan;
using pie_forward::PieForwardGptOssCudaFacts;
using pie_forward::PieForwardGptOssFacts;

}  // namespace

bool gpt_oss_declared_forward_enabled() {
    // Tracing and VALIDATING follows the tree-wide polarity: default ON,
    // `=0` disarms. It costs one trace at load and is what catches a
    // drift between the text and this registry loudly.
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_DECLARED_FORWARD");
        return v == nullptr || v[0] == '\0' || v[0] != '0';
    }();
    return enabled;
}

bool gpt_oss_declared_drive_enabled() {
    // EXECUTING it is opt-in while the arms are new.
    static const bool on = [] {
        const char* v = std::getenv("PIE_DECLARED_FORWARD_GPT_OSS");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return on;
}

GptOssDeclaredPlan build_gpt_oss_declared_plan(
    const HfConfig& cfg,
    const MixtralWeights& w,
    const LlamaLikeForwardCfg& fwd_cfg,
    int num_experts,
    int top_k) {
    GptOssDeclaredPlan out;
    const auto refuse = [&](const std::string& why) {
        std::fprintf(stderr, "[declared-gptoss] declined: %s\n", why.c_str());
        return false;
    };
#define GO_REFUSE(why)                    \
    do {                                  \
        (void)refuse(why);                \
        return out;                       \
    } while (0)

    if (fwd_cfg.tp_size > 1) GO_REFUSE("tp>1");
    if (w.layers.empty() ||
        static_cast<int>(w.layers.size()) != cfg.num_hidden_layers) {
        GO_REFUSE("bound layer count != config");
    }
    if (w.embed == nullptr || w.final_norm == nullptr || w.lm_head == nullptr) {
        GO_REFUSE("a model-level tensor is unbound");
    }
    if (num_experts <= 0 || top_k <= 0) GO_REFUSE("no MoE routing");
    if (cfg.swiglu_limit <= 0.f) {
        GO_REFUSE("no swiglu limit (the unclamped GLU is not stated)");
    }

    // The three per-layer answers the text turned into facts, asked of
    // EVERY layer rather than of layer 0. A mixed stack would make one
    // fact a lie for the layers it does not describe, and the trace bakes
    // these in — so a disagreement has to refuse here, not be discovered
    // by a wrong number later.
    for (const auto& l : w.layers) {
        if (l.attn_sinks == nullptr) {
            GO_REFUSE("a layer carries no attention sink");
        }
        if (l.q_bias == nullptr || l.k_bias == nullptr ||
            l.v_bias == nullptr || l.o_bias == nullptr ||
            l.router_bias == nullptr) {
            GO_REFUSE("a layer is missing one of the biases the text folds");
        }
        // The fused MXFP4 leg indexes experts through these arrays. An
        // empty one is the runtime's own signal that the leg is
        // unavailable, and the alternative is the host-routed walk this
        // declaration refuses by name.
        if (l.expert_gate_up_packed_ptrs.empty() ||
            l.expert_gate_up_scale_ptrs.empty() ||
            l.expert_down_packed_ptrs.empty() ||
            l.expert_down_scale_ptrs.empty()) {
            GO_REFUSE("a layer has no per-expert pointer bank "
                      "(the host-routed walk is not stated)");
        }
        if (l.expert_cache != nullptr) {
            GO_REFUSE("streamed experts (the page-in round trip "
                      "is not stated)");
        }
    }
    // `mixtral_forward_paged` reads these two straight from the config
    // for its GEMV admission; both are the fused kernel's own alignment
    // requirement, so a checkpoint that fails them never reaches the leg.
    const int H = cfg.hidden_size;
    const int I = cfg.intermediate_size;
    if (H % 32 != 0 || I % 32 != 0) {
        GO_REFUSE("hidden/intermediate not 32-aligned (the fused GEMV's own bar)");
    }

    PieForwardGptOssFacts facts{};
    facts.hidden = static_cast<std::uint32_t>(H);
    facts.layers = static_cast<std::uint32_t>(cfg.num_hidden_layers);
    facts.q_heads = static_cast<std::uint32_t>(cfg.num_attention_heads);
    facts.kv_heads = static_cast<std::uint32_t>(cfg.num_key_value_heads);
    facts.head_dim = static_cast<std::uint32_t>(cfg.head_dim);
    facts.intermediate = static_cast<std::uint32_t>(I);
    facts.experts = static_cast<std::uint32_t>(num_experts);
    facts.top_k = static_cast<std::uint32_t>(top_k);
    facts.vocab = static_cast<std::uint32_t>(cfg.vocab_size);
    facts.tied_embeddings = (w.lm_head == w.embed) ? 1 : 0;
    facts.attention_bias = 1;
    facts.attn_sinks = 1;
    facts.swiglu_limit = cfg.swiglu_limit;

    PieForwardGptOssCudaFacts cuda{};
    cuda.mxfp4_decode_gemv = 1;
    cuda.streamed_experts = 0;
    // The hand pass's own default when the field is unset. Carried on the
    // plan so the drive asks the same admission question rather than
    // mirroring the constant.
    int max_routes = w.mxfp4_decode_max_routes;
    if (max_routes == 0) max_routes = 32 * num_experts;
    cuda.mxfp4_decode_max_routes = static_cast<std::uint32_t>(max_routes);
    out.max_routes = max_routes;

    try {
        out.decode = ForwardPlan::trace_gpt_oss_cuda(
            facts, cuda, pie_forward::PieForwardFireClass::Decode);
        gpt_oss_validate_stated_kernels(out.decode);
    } catch (const std::exception& e) {
        GO_REFUSE(std::string("trace failed: ") + e.what());
    }

    out.facts_digest =
        "gptoss/l" + std::to_string(cfg.num_hidden_layers) +
        "/h" + std::to_string(H) + "/i" + std::to_string(I) +
        "/hq" + std::to_string(cfg.num_attention_heads) +
        "/hk" + std::to_string(cfg.num_key_value_heads) +
        "/d" + std::to_string(cfg.head_dim) +
        "/e" + std::to_string(num_experts) + "/k" + std::to_string(top_k) +
        "/mr" + std::to_string(max_routes);
#undef GO_REFUSE
    out.usable = true;
    std::fprintf(stderr,
                 "[declared-gptoss] traced ops=%zu layers=%d experts=%d "
                 "top_k=%d max_routes=%d validation=OK\n",
                 out.decode.op_count(), cfg.num_hidden_layers, num_experts,
                 top_k, max_routes);
    return out;
}

}  // namespace pie_cuda_driver::model
