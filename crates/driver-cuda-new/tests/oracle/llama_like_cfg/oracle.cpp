// The llama_like config/plan-state oracle — slice A of gate-plan-state.
//
// Compiles the REAL model/llama_like/llama_like.cpp (all 3.2k lines) and
// drives its host-pure surface:
//
//   * `LlamaLikeForwardCfg` and `LlamaLikePlanState` defaults, field by
//     field, in declaration order;
//   * `rope_kind_from_hf_config` over every `RopeScaling` value;
//   * `apply_rope_config` over a grid of HF rope parameter blocks;
//   * `decode_fused_post_enabled` (driven per-process by run.sh, because
//     the C++ caches its env read in a function-local static);
//   * `llama_like_decode_graph_layout`, `llama_like_supergraph_graph_layout`
//     and `llama_like_prefill_graph_capturable` over a 1,152-point grid of
//     plan-state shapes.
//
// The only replaced implementations are the flashinfer plan-cache entry
// points — `DecodePlanCache`/`PrefillPlanCache` are opaque in the real
// header precisely so their definition can live GPU-side, which is what
// makes a host-side recorder definition legitimate here. Each recorder
// returns the layout value the driver stored into it, so the transcript is
// about the DRIVER's branch structure and mixing, never about flashinfer.
//
// Everything else in the TU (the forward body, the prepare hook, lora
// staging) is compiled but discarded: the build uses -ffunction-sections
// and links with --gc-sections, so any function the driven surface actually
// reaches must be defined — the linker enforces the stub inventory.
//
// Floats are printed as their IEEE bit patterns. A `%g` on one side and a
// `{}` on the other would make the golden a claim about formatting.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>

#include "model/llama_like/llama_like.hpp"
#include "attn/attention_xqa.hpp"

namespace attn = pie_cuda_driver::kernels::attn;
using pie_cuda_driver::HfConfig;
using pie_cuda_driver::model::LlamaLikeForwardCfg;
using pie_cuda_driver::model::LlamaLikePlanState;
using pie_cuda_driver::model::NormPlacement;
using pie_cuda_driver::model::RopeKind;

// ── the plan-cache recorders ────────────────────────────────────────────────
//
// The real caches hold flashinfer's DecodePlanInfo; the layout functions
// digest them to a uint32. Here the digest IS the stored value, so the
// oracle controls exactly what each slot contributes to the mix.

namespace pie_cuda_driver::kernels::attn {

struct DecodePlanCache {
    std::uint32_t layout;
};
struct PrefillPlanCache {
    std::uint32_t layout;
    bool capturable;
};

void DecodePlanCacheDeleter::operator()(DecodePlanCache* p) const noexcept {
    delete p;
}
void PrefillPlanCacheDeleter::operator()(PrefillPlanCache* p) const noexcept {
    delete p;
}
DecodePlanCachePtr make_decode_plan() {
    return DecodePlanCachePtr(new DecodePlanCache{0});
}
PrefillPlanCachePtr make_prefill_plan() {
    return PrefillPlanCachePtr(new PrefillPlanCache{0, false});
}
std::uint32_t decode_plan_graph_layout(const DecodePlanCache& c) {
    return c.layout;
}
std::uint32_t prefill_plan_graph_layout(const PrefillPlanCache& c) {
    return c.layout;
}
bool prefill_plan_graph_capturable(const PrefillPlanCache& c) {
    return c.capturable;
}
// Pure of an int in the real code too; the mapping here just has to be one
// the Rust parity test's mock reproduces bit for bit.
std::uint8_t xqa_decode_graph_layout(int max_pages_per_seq) {
    return static_cast<std::uint8_t>(0xE0u ^ (max_pages_per_seq * 7));
}

}  // namespace pie_cuda_driver::kernels::attn

// ── transcript helpers ──────────────────────────────────────────────────────

constexpr char SEP = '\x1f';

static std::uint32_t fbits(float f) {
    std::uint32_t u;
    std::memcpy(&u, &f, sizeof u);
    return u;
}

static void row_kv(const char* section, const char* field, const std::string& v) {
    std::printf("%s%c%s%c%s\n", section, SEP, field, SEP, v.c_str());
}
static void kv_u(const char* s, const char* f, std::uint64_t v) {
    row_kv(s, f, std::to_string(v));
}
static void kv_i(const char* s, const char* f, std::int64_t v) {
    row_kv(s, f, std::to_string(v));
}
static void kv_f(const char* s, const char* f, float v) {
    row_kv(s, f, std::to_string(fbits(v)));
}

// ── S1: the defaults, field by field, declaration order ─────────────────────

static void sweep_cfg_defaults() {
    const LlamaLikeForwardCfg c;
    const char* S = "cfg-default";
    kv_u(S, "use_qk_norm", c.use_qk_norm);
    kv_u(S, "use_qkv_bias", c.use_qkv_bias);
    kv_i(S, "norm_placement", static_cast<int>(c.norm_placement));
    kv_i(S, "rope_kind", static_cast<int>(c.rope_kind));
    kv_f(S, "yarn_factor", c.yarn_factor);
    kv_f(S, "yarn_low_freq_factor", c.yarn_low_freq_factor);
    kv_f(S, "yarn_high_freq_factor", c.yarn_high_freq_factor);
    kv_i(S, "yarn_original_max_position", c.yarn_original_max_position);
    kv_f(S, "yarn_beta_fast", c.yarn_beta_fast);
    kv_f(S, "yarn_beta_slow", c.yarn_beta_slow);
    kv_f(S, "yarn_attention_factor", c.yarn_attention_factor);
    kv_i(S, "sliding_window", c.sliding_window);
    kv_u(S, "per_layer_window_left.len", c.per_layer_window_left.size());
    kv_u(S, "force_prefill_path", c.force_prefill_path);
    kv_u(S, "use_xqa_decode", c.use_xqa_decode);
    kv_u(S, "decode_plan_cuda_graph", c.decode_plan_cuda_graph);
    kv_u(S, "use_prefill_decode_plan", c.use_prefill_decode_plan);
    kv_i(S, "prefill_decode_full_attention_min_requests",
         c.prefill_decode_full_attention_min_requests);
    kv_i(S, "prefill_decode_full_attention_min_kv_pages",
         c.prefill_decode_full_attention_min_kv_pages);
    kv_i(S, "prefill_decode_min_kv_pages", c.prefill_decode_min_kv_pages);
    kv_i(S, "tp_size", c.tp_size);
    kv_u(S, "tp_comm_null", c.tp_comm == nullptr);
    kv_u(S, "emit_logits", c.emit_logits);
    kv_i(S, "logits_argmax_chunk_tokens", c.logits_argmax_chunk_tokens);
    kv_i(S, "mrope_section_t", c.mrope_section_t);
    kv_i(S, "mrope_section_h", c.mrope_section_h);
    kv_i(S, "mrope_section_w", c.mrope_section_w);
}

static void sweep_state_defaults() {
    const LlamaLikePlanState s;
    const char* S = "state-default";
    kv_u(S, "decode_plan_null", s.decode_plan == nullptr);
    kv_u(S, "prefill_plan_null", s.prefill_plan == nullptr);
    kv_u(S, "prefill_decode_plan_null", s.prefill_decode_plan == nullptr);
    kv_u(S, "mask_decode_plan_null", s.mask_decode_plan == nullptr);
    kv_u(S, "depth_prefix_decode_plan_null",
         s.depth_prefix_decode_plan == nullptr);
    kv_u(S, "depth_band_plans.len", s.depth_band_plans.size());
    for (std::size_t i = 0; i < s.depth_band_plans.size(); ++i) {
        kv_u(S, ("depth_band_plans_null." + std::to_string(i)).c_str(),
             s.depth_band_plans[i] == nullptr);
    }
    kv_u(S, "depth_band_prefill_plans.len", s.depth_band_prefill_plans.size());
    for (std::size_t i = 0; i < s.depth_band_prefill_plans.size(); ++i) {
        kv_u(S, ("depth_band_prefill_plans_null." + std::to_string(i)).c_str(),
             s.depth_band_prefill_plans[i] == nullptr);
    }
    for (std::size_t i = 0; i < s.depth_band_k.size(); ++i) {
        kv_u(S, ("depth_band_k." + std::to_string(i)).c_str(),
             s.depth_band_k[i]);
    }
    for (std::size_t i = 0; i < s.depth_band_rows.size(); ++i) {
        kv_u(S, ("depth_band_rows." + std::to_string(i)).c_str(),
             s.depth_band_rows[i]);
    }
    kv_u(S, "depth_band_count", s.depth_band_count);
    kv_u(S, "mixed_mid_decode_plan_null", s.mixed_mid_decode_plan == nullptr);
    kv_i(S, "mixed_mid_start", s.mixed_mid_start);
    kv_i(S, "spatial_mask_split", s.spatial_mask_split);
    kv_i(S, "spatial_mask_row_split", s.spatial_mask_row_split);
    kv_u(S, "use_prefill_plan", s.use_prefill_plan);
    kv_u(S, "use_prefill_decode_plan", s.use_prefill_decode_plan);
    kv_u(S, "use_mask_decode_plan", s.use_mask_decode_plan);
    kv_u(S, "prefill_score_window", s.prefill_score_window);
    kv_u(S, "lora_staged_null", s.lora_staged == nullptr);
    kv_u(S, "lora_staged_table_null", s.lora_staged_table == nullptr);
    kv_u(S, "use_xqa_decode", s.use_xqa_decode);
    kv_i(S, "xqa_max_pages_per_seq", s.xqa_max_pages_per_seq);
    kv_u(S, "prefill_decode_qo_indptr_h.len",
         s.prefill_decode_qo_indptr_h.size());
}

// ── S2/S3: the rope mapping ─────────────────────────────────────────────────

static void sweep_rope() {
    const HfConfig::RopeScaling kinds[] = {
        HfConfig::RopeScaling::None,
        HfConfig::RopeScaling::Llama3,
        HfConfig::RopeScaling::OriginalYaRN,
    };
    for (auto k : kinds) {
        HfConfig hf{};
        hf.rope_scaling_kind = k;
        std::printf("rope-kind%c%d%c%d\n", SEP, static_cast<int>(k), SEP,
                    static_cast<int>(
                        pie_cuda_driver::model::rope_kind_from_hf_config(hf)));
    }

    // Distinct, sign-and-magnitude-varied values per grid point so a field
    // routed into the wrong slot cannot alias a field routed correctly.
    struct Block {
        HfConfig::RopeScaling kind;
        float factor, low, high;
        int omp;
        float bfast, bslow, afactor;
    };
    const Block grid[] = {
        {HfConfig::RopeScaling::None, 1.0f, 1.0f, 4.0f, 8192, 32.0f, 1.0f,
         1.0f},
        {HfConfig::RopeScaling::Llama3, 32.0f, 0.001953125f, 0.0078125f, 16,
         16.0f, 2.0f, 0.75f},
        {HfConfig::RopeScaling::OriginalYaRN, 2.5f, -1.5f, 3.25f, 4096,
         48.0f, 0.5f, 1.25f},
        {HfConfig::RopeScaling::Llama3, 8.0f, 1.0f, 4.0f, 32768, 32.0f, 1.0f,
         -2.0f},
    };
    int i = 0;
    for (const Block& b : grid) {
        HfConfig hf{};
        hf.rope_scaling_kind = b.kind;
        hf.rope_factor = b.factor;
        hf.rope_low_freq_factor = b.low;
        hf.rope_high_freq_factor = b.high;
        hf.rope_original_max_position = b.omp;
        hf.rope_beta_fast = b.bfast;
        hf.rope_beta_slow = b.bslow;
        hf.rope_attention_factor = b.afactor;

        LlamaLikeForwardCfg cfg;
        pie_cuda_driver::model::apply_rope_config(cfg, hf);
        std::printf("apply-rope%c%d%c%d%c%u%c%u%c%u%c%d%c%u%c%u%c%u\n", SEP,
                    i++, SEP, static_cast<int>(cfg.rope_kind), SEP,
                    fbits(cfg.yarn_factor), SEP, fbits(cfg.yarn_low_freq_factor),
                    SEP, fbits(cfg.yarn_high_freq_factor), SEP,
                    cfg.yarn_original_max_position, SEP,
                    fbits(cfg.yarn_beta_fast), SEP, fbits(cfg.yarn_beta_slow),
                    SEP, fbits(cfg.yarn_attention_factor));
    }
}

// ── S5: the graph-layout functions over a plan-state grid ───────────────────

static void sweep_layouts() {
    int i = 0;
    const int spatials[] = {-1, 0, 2};
    for (int sm : spatials)
    for (int use_mask = 0; use_mask <= 1; ++use_mask)
    for (int mask_present = 0; mask_present <= 1; ++mask_present)
    for (int decode_present = 0; decode_present <= 1; ++decode_present)
    for (int xqa = 0; xqa <= 1; ++xqa)
    for (int pd = 0; pd < 3; ++pd)          // off / on-but-absent / on
    for (int pf = 0; pf < 4; ++pf) {        // off / on-but-absent /
                                            // on-noncapturable / on-capturable
        LlamaLikePlanState s;
        s.spatial_mask_split = sm;
        s.use_mask_decode_plan = use_mask != 0;
        if (mask_present) {
            s.mask_decode_plan = attn::PrefillPlanCachePtr(
                new attn::PrefillPlanCache{0x70u + (static_cast<std::uint32_t>(i) % 7u), false});
        }
        if (decode_present) {
            s.decode_plan = attn::DecodePlanCachePtr(
                new attn::DecodePlanCache{0x10u + (static_cast<std::uint32_t>(i) % 5u)});
        }
        s.use_xqa_decode = xqa != 0;
        s.xqa_max_pages_per_seq = xqa != 0 ? 3 + (i % 4) : 0;
        s.use_prefill_decode_plan = pd != 0;
        if (pd == 2) {
            s.prefill_decode_plan = attn::PrefillPlanCachePtr(
                new attn::PrefillPlanCache{0x50u + (static_cast<std::uint32_t>(i) % 3u), false});
        }
        s.use_prefill_plan = pf != 0;
        if (pf >= 2) {
            s.prefill_plan = attn::PrefillPlanCachePtr(
                new attn::PrefillPlanCache{0x30u + (static_cast<std::uint32_t>(i) % 9u), pf == 3});
        }

        std::printf(
            "layout%c%d%c%d%c%d%c%d%c%d%c%d%c%d%c%d%c%d%c%u%c%u%c%d\n", SEP,
            i, SEP, sm, SEP, use_mask, SEP, mask_present, SEP, decode_present,
            SEP, xqa, SEP, s.xqa_max_pages_per_seq, SEP, pd, SEP, pf, SEP,
            pie_cuda_driver::model::llama_like_decode_graph_layout(s), SEP,
            pie_cuda_driver::model::llama_like_supergraph_graph_layout(s), SEP,
            static_cast<int>(
                pie_cuda_driver::model::llama_like_prefill_graph_capturable(s)));
        ++i;
    }
}

int main(int argc, char** argv) {
    if (argc >= 3 && std::strcmp(argv[1], "fused_post") == 0) {
        // One row per PROCESS: the C++ caches the env read in a static.
        std::printf("fused_post%c%s%c%d\n", SEP, argv[2], SEP,
                    static_cast<int>(
                        pie_cuda_driver::model::decode_fused_post_enabled()));
        return 0;
    }
    sweep_cfg_defaults();
    sweep_state_defaults();
    sweep_rope();
    sweep_layouts();
    return 0;
}
