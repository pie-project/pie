#include "model/gemma4/declared_forward.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <exception>
#include <string>

namespace pie_cuda_driver::model {

namespace {

using pie_forward::ForwardPlan;
using pie_forward::PieForwardGemma4CudaFacts;
using pie_forward::PieForwardGemma4Facts;

// The layer-kind schedule reduced to an interval, and CHECKED against
// every layer rather than only the full ones — qwen3_5's reduction, and
// the same strictness for the same reason: an irregular stack that
// happens to contain the formula's positions would be mis-scheduled
// silently.
int reduce_interval(const HfConfig& cfg, std::string& reason) {
    const int L = cfg.num_hidden_layers;
    if (cfg.layer_types.empty() ||
        static_cast<int>(cfg.layer_types.size()) != L) {
        reason = "layer_types missing or wrong length";
        return -1;
    }
    int first_full = -1;
    for (int l = 0; l < L; ++l) {
        const std::string& t = cfg.layer_types[l];
        if (t != "sliding_attention" && t != "full_attention") {
            reason = "unexpected layer_type '" + t + "'";
            return -1;
        }
        if (first_full < 0 && t == "full_attention") first_full = l;
    }
    if (first_full < 0) {
        reason = "no full_attention layer";
        return -1;
    }
    const int interval = first_full + 1;
    for (int l = 0; l < L; ++l) {
        const bool full = interval <= 1 || (l % interval) == (interval - 1);
        if (full != (cfg.layer_types[l] == "full_attention")) {
            reason = "irregular layer_types schedule";
            return -1;
        }
    }
    return interval;
}

bool uniform(const std::vector<int>& v, int& value) {
    if (v.empty()) return false;
    value = v[0];
    return std::all_of(v.begin(), v.end(), [&](int x) { return x == value; });
}

}  // namespace

bool gemma4_declared_forward_enabled() {
    // Tracing and VALIDATING the declaration follows llama_like's
    // polarity — default ON, `=0` disarms — because it costs one trace
    // at load and is what catches a drift loudly.
    static const bool enabled = [] {
        const char* v = std::getenv("PIE_DECLARED_FORWARD");
        return v == nullptr || v[0] == '\0' || v[0] != '0';
    }();
    return enabled;
}

bool gemma4_declared_drive_enabled() {
    // EXECUTING it is opt-in, and stays opt-in until the arms are right.
    // They are written and they build; on a live E4B decode they fault
    // (an illegal access that surfaces at the next device copy), so the
    // buffer threading in at least one arm is wrong. Default-on would
    // make that everyone's problem for the sake of a rung that is not
    // finished.
    static const bool on = [] {
        const char* v = std::getenv("PIE_DECLARED_FORWARD_GEMMA4");
        return v != nullptr && v[0] != '\0' && v[0] != '0';
    }();
    return on;
}

Gemma4DeclaredPlan build_gemma4_declared_plan(
    const HfConfig& cfg, const Gemma4Weights& w, int tp_size) {
    Gemma4DeclaredPlan out;
    // Refusal is a FALLBACK, not an error: say why, once, and hand back
    // an unusable plan. `ForwardPlan` is move-only, so the lambda cannot
    // return `out` by value — it marks and the caller returns.
    const auto refuse = [&](const std::string& why) {
        std::fprintf(stderr, "[declared-gemma4] declined: %s\n", why.c_str());
        return false;
    };
#define G4_REFUSE(why)                    \
    do {                                  \
        (void)refuse(why);                \
        return out;                       \
    } while (0)

    const int L = cfg.num_hidden_layers;
    if (tp_size > 1) G4_REFUSE("tp>1");
    if (static_cast<int>(w.layers.size()) != L) {
        G4_REFUSE("bound layer count != config");
    }
    std::string reason;
    const int interval = reduce_interval(cfg, reason);
    if (interval < 0) G4_REFUSE(reason);

    // The MoE-on-top variant (26B-A4B) runs a sparse block in PARALLEL
    // with the dense MLP. That is a whole second MLP the declaration does
    // not state, and its router is a module of its own.
    for (const auto& lw : w.layers) {
        if (lw.router_proj != nullptr || lw.moe_gate_up_proj != nullptr) {
            G4_REFUSE("the MoE-on-top block is not stated");
        }
    }

    // The per-layer geometry the driver already resolved. The trace's
    // axis is TWO head dims, so anything with more than two distinct
    // values is outside it. The MLP WIDTH is derived below, once the
    // KV-shared count is known — the double-wide variant keys on it.
    int intermediate = 0;
    bool double_wide = false;
    int kv_heads = 0;
    if (!uniform(w.per_layer_num_kv_heads, kv_heads)) {
        G4_REFUSE("per-layer kv head counts differ");
    }
    if (static_cast<int>(w.per_layer_head_dim.size()) != L) {
        G4_REFUSE("per-layer head dims unset");
    }
    int sliding_d = -1;
    int full_d = -1;
    for (int l = 0; l < L; ++l) {
        const bool full = interval <= 1 || (l % interval) == (interval - 1);
        int& slot = full ? full_d : sliding_d;
        if (slot < 0) {
            slot = w.per_layer_head_dim[l];
        } else if (slot != w.per_layer_head_dim[l]) {
            G4_REFUSE("head dim varies within a layer KIND");
        }
    }
    if (sliding_d <= 0 || full_d <= 0) G4_REFUSE("a layer kind is absent");

    // KV sharing: the trailing run of layers whose source is not
    // themselves. The declaration states a COUNT, so a checkpoint whose
    // sharing is not a trailing run is outside it.
    int shared = 0;
    if (static_cast<int>(w.kv_source_layer.size()) == L) {
        for (int l = L - 1; l >= 0 && w.kv_source_layer[l] != l; --l) ++shared;
        for (int l = 0; l < L - shared; ++l) {
            if (w.kv_source_layer[l] != l) {
                G4_REFUSE("KV sharing is not a trailing run");
            }
        }
    }

    // The MLP width, now that `shared` is known. One extra shape is
    // admitted: the double-wide variant (E2B), where the KV-SHARED
    // layers carry `2 * intermediate`. Admitted by DERIVING the narrow
    // width and checking every layer against the predicate the trace
    // will use, so a checkpoint that widens some other set of layers
    // still refuses rather than being folded into a fact that does not
    // describe it.
    if (!uniform(w.per_layer_intermediate, intermediate)) {
        if (shared <= 0 ||
            static_cast<int>(w.per_layer_intermediate.size()) != L) {
            G4_REFUSE("per-layer intermediate widths differ");
        }
        const int first_shared = L - shared;
        intermediate = w.per_layer_intermediate[0];
        for (int l = 0; l < L; ++l) {
            const int want = (l >= first_shared) ? 2 * intermediate : intermediate;
            if (w.per_layer_intermediate[static_cast<std::size_t>(l)] != want) {
                G4_REFUSE("per-layer intermediate widths differ in a shape "
                          "other than double-wide-on-shared");
            }
        }
        double_wide = true;
    }

    // Partial rope on the FULL layers, the driver's own derivation.
    int rotary = full_d;
    if (static_cast<int>(w.per_layer_partial_rotary_factor.size()) == L) {
        const float f = w.per_layer_partial_rotary_factor[interval - 1];
        rotary = std::max(2, 2 * static_cast<int>(0.5f * f * full_d));
    }

    PieForwardGemma4Facts facts{};
    facts.hidden = static_cast<std::uint32_t>(cfg.hidden_size);
    facts.layers = static_cast<std::uint32_t>(L);
    facts.full_attn_interval = static_cast<std::uint32_t>(interval);
    facts.q_heads = static_cast<std::uint32_t>(cfg.num_attention_heads);
    facts.kv_heads = static_cast<std::uint32_t>(kv_heads);
    facts.head_dim = static_cast<std::uint32_t>(sliding_d);
    facts.global_head_dim = static_cast<std::uint32_t>(full_d);
    facts.global_rotary_dim = static_cast<std::uint32_t>(rotary);
    facts.intermediate = static_cast<std::uint32_t>(intermediate);
    facts.vocab = static_cast<std::uint32_t>(cfg.vocab_size);
    facts.tied_embeddings = (w.lm_head == w.embed) ? 1 : 0;
    facts.kv_shared_layers = static_cast<std::uint32_t>(shared);
    facts.ple_dim = static_cast<std::uint32_t>(cfg.gemma_hidden_size_per_layer_input);
    facts.double_wide_shared = double_wide ? 1 : 0;
    facts.logit_softcap = cfg.gemma_final_logit_softcap;

    if (facts.ple_dim == 0 || w.embed_per_layer == nullptr) {
        G4_REFUSE("no PLE table (the declaration states one)");
    }

    // The bindings. Layer 0 speaks for the deployment: the loader's join
    // contract accepts or declines a GROUP uniformly, and the executor
    // cross-checks per launch rather than trusting this line.
    PieForwardGemma4CudaFacts cuda{};
    cuda.fused_qkv = w.layers[0].qkv_proj_fused != nullptr ? 1 : 0;
    cuda.gate_up_fused = w.layers[0].gate_up_proj_fused != nullptr ? 1 : 0;
    // The cache format is not knowable here (the cache is engine-owned
    // and built after the model), so the deployment DEFAULT stands in and
    // the executor's fused-post arm refuses if the live view disagrees —
    // qwen3_5's `state_bf16` precedent exactly.
    cuda.kv_native_bf16 = 1;

    try {
        out.decode = ForwardPlan::trace_gemma4_cuda(
            facts, cuda, pie_forward::PieForwardFireClass::Decode);
        gemma4_validate_stated_kernels(out.decode);
        out.prefill = ForwardPlan::trace_gemma4_cuda(
            facts, cuda, pie_forward::PieForwardFireClass::Prefill);
        gemma4_validate_stated_kernels(out.prefill);
    } catch (const std::exception& e) {
        G4_REFUSE(std::string("trace failed: ") + e.what());
    }

    out.facts_digest =
        "gemma4/l" + std::to_string(L) + "/int" + std::to_string(interval) +
        "/h" + std::to_string(cfg.hidden_size) +
        "/d" + std::to_string(sliding_d) + "," + std::to_string(full_d) +
        "/rot" + std::to_string(rotary) +
        "/i" + std::to_string(intermediate) +
        "/ple" + std::to_string(facts.ple_dim) +
        "/kvs" + std::to_string(shared) +
        "/dw" + std::to_string(double_wide ? 1 : 0) +
        "/fq" + std::to_string(cuda.fused_qkv) +
        "/gu" + std::to_string(cuda.gate_up_fused);
#undef G4_REFUSE
    out.usable = true;
    std::fprintf(stderr,
                 "[declared-gemma4] traced ops=%zu/%zu layers=%d interval=%d "
                 "shared=%d d=%d/%d validation=OK\n",
                 out.decode.op_count(), out.prefill.op_count(),
                 L, interval, shared, sliding_d, full_d);
    return out;
}

}  // namespace pie_cuda_driver::model
