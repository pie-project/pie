// The prepare-hook oracle — slice B of gate-plan-state.
//
// Compiles the REAL `llama_like.cpp` (as slice A did: -ffunction-sections,
// --gc-sections, the forward body type-checked and discarded) and drives
// `prepare_llama_like_decode_plan` across every deployment branch: plain
// decode, prefill-decode, XQA, real prefill, force-prefill, fire-level
// custom masks, the NS-2 spatial split (with its recursive prefix build),
// the M-2 mixed fire (with the NO-DEMOTION middle), the S-2 depth prefix,
// and the ④ banded-depth stamps on four different deployments.
//
// The replaced surface is the flashinfer PLANNER boundary: the two plan
// entry points become recorders that dump every argument — the CSR arrays
// BY CONTENT, because the rebasing arithmetic (the suffix slice, the
// middle's page-base subtraction, the identity qo) is exactly what this
// gate has to prove; a pointer would prove nothing. Workspaces are real
// (`attention_workspace.cpp` linked over silent CUDA), named by which
// buffer their view carries — the two-plans-one-workspace ISOLATION rule
// is half of what the transcript pins. The KvCache is real too, built by
// the same `kv_cache.cpp` the live gate proved.
//
// `prefill_graph_plan_enabled` is the one driver function stubbed here
// (batch/forward.cpp:200 — its TU is the whole batch engine). The stub
// reproduces its three lines verbatim, and run.sh sweeps the env var
// across processes, so the WIRING `gate && decode_plan_cuda_graph` is
// still proven against the real caller.

#include <cstdio>
#include <cstdlib>
#include <map>
#include <string>
#include <vector>

#include "model/llama_like/llama_like.hpp"
#include "store/kv_cache.hpp"
#include "attn/attention_xqa.hpp"

using pie_cuda_driver::AttentionWorkspace;
using pie_cuda_driver::HfConfig;
using pie_cuda_driver::KvCache;
using pie_cuda_driver::model::LlamaLikeForwardCfg;
using pie_cuda_driver::model::LlamaLikePlanState;

namespace {
constexpr char SEP = '\x1f';
std::string g_case;
std::map<const void*, std::string> g_ws_names;

std::string ws_name(const void* float_buffer) {
    auto it = g_ws_names.find(float_buffer);
    return it == g_ws_names.end() ? "ws?" : it->second;
}

std::string join_u32(const std::uint32_t* p, int n) {
    if (p == nullptr) return "null";
    std::string s = "[";
    for (int i = 0; i < n; ++i) {
        if (i) s += ',';
        s += std::to_string(p[i]);
    }
    return s + "]";
}
}  // namespace

// ── the planner recorders ───────────────────────────────────────────────────

namespace pie_cuda_driver::kernels::attn {

struct DecodePlanCache {
    int id;
};
struct PrefillPlanCache {
    int id;
};

namespace {
int g_next_decode = 0;
int g_next_prefill = 0;
}  // namespace

void reset_plan_ids() {
    g_next_decode = 0;
    g_next_prefill = 0;
}

void DecodePlanCacheDeleter::operator()(DecodePlanCache* p) const noexcept {
    delete p;
}
void PrefillPlanCacheDeleter::operator()(PrefillPlanCache* p) const noexcept {
    delete p;
}
DecodePlanCachePtr make_decode_plan() {
    return DecodePlanCachePtr(new DecodePlanCache{g_next_decode++});
}
PrefillPlanCachePtr make_prefill_plan() {
    return PrefillPlanCachePtr(new PrefillPlanCache{g_next_prefill++});
}

void plan_attention_flashinfer_decode_bf16(
    DecodePlanCache& cache,
    const std::uint32_t* kv_page_indptr_h,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    AttentionWorkspaceView workspace,
    cudaStream_t stream,
    bool enable_cuda_graph,
    bool full_attention_variant,
    bool hnd_layout,
    int window_left)
{
    std::printf(
        "%s%cplan-decode%cdp#%d%ckvpp=%s%cR=%d%cqh=%d%ckvh=%d%chd=%d%cpsz=%d"
        "%c%s%c%s%cgraph=%d%cfav=%d%chnd=%d%cwl=%d\n",
        g_case.c_str(), SEP, SEP, cache.id, SEP,
        join_u32(kv_page_indptr_h, num_requests + 1).c_str(), SEP,
        num_requests, SEP, num_q_heads, SEP, num_kv_heads, SEP, head_dim, SEP,
        page_size, SEP, ws_name(workspace.float_buffer).c_str(), SEP,
        stream == nullptr ? "s0" : "s?", SEP, enable_cuda_graph ? 1 : 0, SEP,
        full_attention_variant ? 1 : 0, SEP, hnd_layout ? 1 : 0, SEP,
        window_left);
}

void plan_attention_flashinfer_prefill_bf16(
    PrefillPlanCache& cache,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    const std::uint32_t* kv_last_page_lens_h,
    int total_tokens,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    AttentionWorkspaceView workspace,
    cudaStream_t stream,
    bool enable_cuda_graph,
    int window_left,
    bool full_attention_variant,
    bool hnd_layout,
    bool causal_mask,
    bool custom_mask,
    bool wants_prefill_score)
{
    std::printf(
        "%s%cplan-prefill%cpp#%d%cqo=%s%ckvpp=%s%clens=%s%cT=%d%cR=%d%cqh=%d"
        "%ckvh=%d%chd=%d%cpsz=%d%c%s%c%s%cgraph=%d%cwl=%d%cfav=%d%chnd=%d"
        "%ccausal=%d%ccustom=%d%cscore=%d\n",
        g_case.c_str(), SEP, SEP, cache.id, SEP,
        join_u32(qo_indptr_h, num_requests + 1).c_str(), SEP,
        join_u32(kv_page_indptr_h, num_requests + 1).c_str(), SEP,
        join_u32(kv_last_page_lens_h, num_requests).c_str(), SEP, total_tokens,
        SEP, num_requests, SEP, num_q_heads, SEP, num_kv_heads, SEP, head_dim,
        SEP, page_size, SEP, ws_name(workspace.float_buffer).c_str(), SEP,
        stream == nullptr ? "s0" : "s?", SEP, enable_cuda_graph ? 1 : 0, SEP,
        window_left, SEP, full_attention_variant ? 1 : 0, SEP,
        hnd_layout ? 1 : 0, SEP, causal_mask ? 1 : 0, SEP, custom_mask ? 1 : 0,
        SEP, wants_prefill_score ? 1 : 0);
}

// Pure of its input in the real code too; the mapping only has to be one
// the Rust mock reproduces exactly, and odd enough that an unbucketed
// pass-through cannot alias it.
int xqa_decode_page_bucket(int max_pages_per_seq) {
    int b = 4;
    while (b < max_pages_per_seq) b *= 2;
    return b;
}

}  // namespace pie_cuda_driver::kernels::attn

// ── the one stubbed driver function ─────────────────────────────────────────
//
// batch/forward.cpp:200, verbatim: its TU is the whole batch engine.
namespace pie_cuda_driver {
bool prefill_graph_plan_enabled() {
    static const bool value = [] {
        const char* const env = std::getenv("PIE_PREFILL_GRAPH_PLAN");
        return env != nullptr && *env != '\0' && env[0] != '0';
    }();
    return value;
}
}  // namespace pie_cuda_driver

// ── the state dump ──────────────────────────────────────────────────────────

namespace {

std::string dp(const pie_cuda_driver::kernels::attn::DecodePlanCachePtr& p) {
    return p ? "dp#" + std::to_string(p->id) : "null";
}
std::string pp(const pie_cuda_driver::kernels::attn::PrefillPlanCachePtr& p) {
    return p ? "pp#" + std::to_string(p->id) : "null";
}

void dump_state(const LlamaLikePlanState& s) {
    std::string bands, band_pf, bk, br;
    for (std::size_t i = 0; i < s.depth_band_plans.size(); ++i) {
        if (i) bands += ',';
        bands += dp(s.depth_band_plans[i]);
    }
    for (std::size_t i = 0; i < s.depth_band_prefill_plans.size(); ++i) {
        if (i) band_pf += ',';
        band_pf += pp(s.depth_band_prefill_plans[i]);
    }
    for (std::size_t i = 0; i < 3; ++i) {
        if (i) {
            bk += ',';
            br += ',';
        }
        bk += std::to_string(s.depth_band_k[i]);
        br += std::to_string(s.depth_band_rows[i]);
    }
    std::string pdqo = "[";
    for (std::size_t i = 0; i < s.prefill_decode_qo_indptr_h.size(); ++i) {
        if (i) pdqo += ',';
        pdqo += std::to_string(s.prefill_decode_qo_indptr_h[i]);
    }
    pdqo += "]";
    std::printf(
        "%s%cstate%cdecode=%s%cprefill=%s%cpd=%s%cmask=%s%cdpfx=%s%cbands=%s"
        "%cband_pf=%s%cband_k=%s%cband_rows=%s%cband_n=%u%cmid=%s%cmid_start=%d"
        "%csm=%d%csm_row=%d%cuse_pf=%d%cuse_pd=%d%cuse_mask=%d%cscore_w=%u"
        "%cxqa=%d%cxqa_max=%d%cpd_qo=%s\n",
        g_case.c_str(), SEP, SEP, dp(s.decode_plan).c_str(), SEP,
        pp(s.prefill_plan).c_str(), SEP, pp(s.prefill_decode_plan).c_str(),
        SEP, pp(s.mask_decode_plan).c_str(), SEP,
        dp(s.depth_prefix_decode_plan).c_str(), SEP, bands.c_str(), SEP,
        band_pf.c_str(), SEP, bk.c_str(), SEP, br.c_str(), SEP,
        s.depth_band_count, SEP, dp(s.mixed_mid_decode_plan).c_str(), SEP,
        s.mixed_mid_start, SEP, s.spatial_mask_split, SEP,
        s.spatial_mask_row_split, SEP, s.use_prefill_plan ? 1 : 0, SEP,
        s.use_prefill_decode_plan ? 1 : 0, SEP, s.use_mask_decode_plan ? 1 : 0,
        SEP, s.prefill_score_window, SEP, s.use_xqa_decode ? 1 : 0, SEP,
        s.xqa_max_pages_per_seq, SEP, pdqo.c_str());
}

// ── the case driver ─────────────────────────────────────────────────────────

struct Fire {
    std::vector<std::uint32_t> qo;    // R+1
    std::vector<std::uint32_t> kvpp;  // R+1
    std::vector<std::uint32_t> lens;  // R
    int total_tokens() const { return static_cast<int>(qo.back()); }
    int requests() const { return static_cast<int>(qo.size()) - 1; }
};

struct Env {
    AttentionWorkspace& attn_ws;
    KvCache& cache;
    const HfConfig& cfg;
};

void run_case(
    const char* name,
    Env& env,
    LlamaLikePlanState& state,
    const LlamaLikeForwardCfg& fwd,
    const Fire& fire,
    bool is_pure_decode,
    bool have_custom_mask,
    std::uint32_t attn_score_window = 0,
    std::uint32_t unmasked_prefix_rows = 0xffffffffu,
    const std::uint32_t* mask_suffix_page_counts = nullptr,
    const std::uint32_t* mask_suffix_last_lens = nullptr,
    std::uint32_t full_depth_rows = 0xffffffffu,
    const std::uint32_t* depth_band_k = nullptr,
    const std::uint32_t* depth_band_rows = nullptr,
    std::uint32_t depth_band_count = 0)
{
    g_case = name;
    std::printf("%s%ccall%cR=%d%cT=%d%cpure=%d%cmask=%d%cscore_w=%u%cprefix=%u"
                "%cfdr=%u%cbands=%u\n",
                g_case.c_str(), SEP, SEP, fire.requests(), SEP,
                fire.total_tokens(), SEP, is_pure_decode ? 1 : 0, SEP,
                have_custom_mask ? 1 : 0, SEP, attn_score_window, SEP,
                unmasked_prefix_rows, SEP, full_depth_rows, SEP,
                depth_band_count);
    pie_cuda_driver::model::prepare_llama_like_decode_plan(
        state, env.attn_ws, env.cache, env.cfg, fwd, fire.qo.data(),
        /*kv_page_indices_d=*/nullptr, fire.kvpp.data(),
        /*kv_page_indptr_d=*/nullptr, fire.lens.data(),
        /*kv_last_page_lens_d=*/nullptr, fire.total_tokens(), fire.requests(),
        is_pure_decode, have_custom_mask, attn_score_window,
        unmasked_prefix_rows, mask_suffix_page_counts, mask_suffix_last_lens,
        full_depth_rows, depth_band_k, depth_band_rows, depth_band_count);
    dump_state(state);
}

}  // namespace

int main(int argc, char** argv) {
    const std::string mode = argc > 1 ? argv[1] : "main";

    HfConfig cfg{};
    cfg.num_attention_heads = 8;
    cfg.num_key_value_heads = 4;
    cfg.head_dim = 64;
    cfg.head_dim_kernel = 64;

    HfConfig cfg_padded = cfg;  // phi3-like: padded kernel head_dim
    cfg_padded.head_dim = 80;
    cfg_padded.head_dim_kernel = 96;

    auto attn_ws = AttentionWorkspace::allocate(1024, 512, 2);
    g_ws_names[attn_ws.view().float_buffer] = "ws-main";
    g_ws_names[pie_cuda_driver::model::spatial_suffix_attn_ws()
                   .view()
                   .float_buffer] = "ws-suffix";
    for (int i = 0; i < 3; ++i) {
        g_ws_names[pie_cuda_driver::model::depth_band_attn_ws_public(i)
                       .view()
                       .float_buffer] = "ws-band" + std::to_string(i);
    }

    auto cache = KvCache::allocate(
        1, 64, 16, 4, 64, pie_cuda_driver::kv_cache_format_from_string("bf16"));
    auto cache_int8 = KvCache::allocate(
        1, 64, 16, 4, 64,
        pie_cuda_driver::kv_cache_format_from_string("int8_per_token_head"));
    Env env{attn_ws, cache, cfg};
    Env env_int8{attn_ws, cache_int8, cfg};
    Env env_padded{attn_ws, cache, cfg_padded};

    LlamaLikeForwardCfg base;

    const Fire decode4{{0, 1, 2, 3, 4}, {0, 2, 5, 6, 10}, {3, 16, 1, 7}};
    const Fire decode3{{0, 1, 2, 3}, {0, 3, 4, 9}, {5, 2, 16}};
    const Fire prefill3{{0, 5, 9, 10}, {0, 2, 5, 6}, {3, 16, 1}};
    // Mixed: two prefill lanes then 1-token rows.
    const Fire mixed5{{0, 5, 9, 10, 11, 12}, {0, 2, 5, 6, 8, 9},
                      {3, 16, 1, 7, 2}};
    // Mixed where request 1 is already width-1 (the NO-DEMOTION middle).
    const Fire mixed_mid{{0, 5, 6, 7, 8, 9}, {0, 2, 4, 5, 7, 8},
                         {3, 1, 16, 7, 2}};

    if (mode == "main") {
        {
            LlamaLikePlanState st;
            run_case("a-plain-decode", env, st, base, decode4, true, false);
            // Same state again: the plan objects must be REUSED, not remade.
            run_case("a2-reuse", env, st, base, decode4, true, false);
        }
        {
            LlamaLikePlanState st;
            LlamaLikeForwardCfg f;
            f.use_prefill_decode_plan = true;
            run_case("b-prefill-decode", env, st, f, decode4, true, false);
            // The full-attention variant arms on the request floor.
            LlamaLikeForwardCfg f2;
            f2.use_prefill_decode_plan = true;
            f2.prefill_decode_full_attention_min_requests = 2;
            run_case("b2-pd-fav", env, st, f2, decode4, true, false);
            // The page floor declines the pd plan entirely.
            LlamaLikeForwardCfg f3;
            f3.use_prefill_decode_plan = true;
            f3.prefill_decode_min_kv_pages = 4;
            run_case("b3-pd-declined", env, st, f3, decode4, true, false);
            // decode4 averages 10 pages over 4 requests: exactly the case
            // where the CEILING (3) clears a floor of 3 and a truncating
            // division (2) would not.
            LlamaLikeForwardCfg f4;
            f4.use_prefill_decode_plan = true;
            f4.prefill_decode_min_kv_pages = 3;
            run_case("b4-pd-ceiling", env, st, f4, decode4, true, false);
        }
        {
            LlamaLikePlanState st;
            LlamaLikeForwardCfg f;
            f.use_xqa_decode = true;
            const std::uint32_t bk[2] = {8, 4};
            const std::uint32_t br[2] = {3, 1};
            run_case("c-xqa", env, st, f, decode4, true, false, 0, 0xffffffffu,
                     nullptr, nullptr, 0xffffffffu, bk, br, 2);
            // A non-native cache blocks XQA; the same fire plans decode.
            LlamaLikePlanState st2;
            run_case("c2-xqa-nonnative", env_int8, st2, f, decode4, true,
                     false);
        }
        {
            LlamaLikePlanState st;
            run_case("d-prefill", env, st, base, prefill3, false, false, 3);
            LlamaLikeForwardCfg f;
            f.sliding_window = 128;
            LlamaLikePlanState st2;
            run_case("d2-prefill-sliding", env, st2, f, prefill3, false, false,
                     3);
            LlamaLikeForwardCfg f2;
            f2.per_layer_window_left = {128, -1, 128};
            LlamaLikePlanState st3;
            run_case("d3-prefill-per-layer", env, st3, f2, prefill3, false,
                     false);
        }
        {
            LlamaLikePlanState st;
            LlamaLikeForwardCfg f;
            f.force_prefill_path = true;
            const std::uint32_t bk[1] = {6};
            const std::uint32_t br[1] = {2};
            run_case("e-force-prefill", env, st, f, decode3, true, false, 0,
                     0xffffffffu, nullptr, nullptr, 0xffffffffu, bk, br, 1);
        }
        {
            LlamaLikePlanState st;
            run_case("f-mask-decode", env, st, base, decode4, true, false);
            // Masked pure decode with NO split lands in the dedicated slot.
            run_case("f2-mask-dedicated", env, st, base, decode4, true, true);
            // Masked prefill-shaped keeps the prefill slot.
            LlamaLikePlanState st2;
            run_case("f3-mask-prefill-shaped", env, st2, base, prefill3, false,
                     true);
        }
        {
            // NS-2: masked pure decode with a planned unmasked prefix.
            LlamaLikePlanState st;
            run_case("g-spatial-split", env, st, base, decode4, true, true, 0,
                     2);
            // split == 0: all-masked composed fire, no prefix plan.
            LlamaLikePlanState st2;
            run_case("g2-spatial-zero", env, st2, base, decode4, true, true, 0,
                     0);
            // Resolver-threaded suffix geometry wins over the host slice.
            const std::uint32_t counts[2] = {3, 2};
            const std::uint32_t lens2[2] = {9, 11};
            LlamaLikePlanState st3;
            run_case("g3-spatial-resolved", env, st3, base, decode4, true,
                     true, 0, 2, counts, lens2);
            // A padded head_dim keeps the fire-level arm.
            LlamaLikePlanState st4;
            run_case("g4-spatial-padded-declined", env_padded, st4, base,
                     decode4, true, true, 0, 2);
            // An XQA deployment keeps the fire-level arm too.
            LlamaLikePlanState st5;
            LlamaLikeForwardCfg fx;
            fx.use_xqa_decode = true;
            run_case("g5-spatial-xqa-declined", env, st5, fx, decode4, true,
                     true, 0, 2);
        }
        {
            // M-2: the mixed fire. No width-1 request before the split, so
            // no middle.
            LlamaLikePlanState st;
            run_case("h-mixed", env, st, base, mixed5, false, true, 0, 3);
            // The NO-DEMOTION middle: P=1, mid=2.
            LlamaLikePlanState st2;
            run_case("h2-mixed-mid", env, st2, base, mixed_mid, false, true, 0,
                     3);
            // A prefix past the request count declines the split.
            LlamaLikePlanState st3;
            run_case("h3-mixed-declined", env, st3, base, mixed5, false, true,
                     0, 7);
        }
        {
            // S-2: the depth union's prefix plan, and its guards.
            LlamaLikePlanState st;
            run_case("i-depth-prefix", env, st, base, decode4, true, false, 0,
                     0xffffffffu, nullptr, nullptr, 2);
            LlamaLikePlanState st2;
            run_case("i2-depth-full-declined", env, st2, base, decode4, true,
                     false, 0, 0xffffffffu, nullptr, nullptr, 4);
        }
        {
            // ④ banded depth on the plain-decode deployment; the zero-row
            // middle band allocates no plan.
            LlamaLikePlanState st;
            const std::uint32_t bk[3] = {8, 4, 2};
            const std::uint32_t br[3] = {2, 0, 1};
            run_case("j-bands-decode", env, st, base, decode4, true, false, 0,
                     0xffffffffu, nullptr, nullptr, 0xffffffffu, bk, br, 3);
            // The SAME state fires unbanded: the stamps must clear, or the
            // walker would band a fire the planner never banded — the
            // exact bug the C++'s re-stamp comment records.
            run_case("j1b-bands-cleared", env, st, base, decode4, true,
                     false);
            // The prefill-decode deployment builds prefill-family bands.
            LlamaLikePlanState st2;
            LlamaLikeForwardCfg f;
            f.use_prefill_decode_plan = true;
            const std::uint32_t bk2[2] = {6, 3};
            const std::uint32_t br2[2] = {3, 1};
            run_case("j2-bands-pd", env, st2, f, decode4, true, false, 0,
                     0xffffffffu, nullptr, nullptr, 0xffffffffu, bk2, br2, 2);
            // The SAME state, unbanded, still on the pd deployment. This
            // is the one path where the top-of-branch re-stamp is the ONLY
            // reset — the pd branch returns before the decode branch's own
            // clear — so it is what separates the two resets. Added by a
            // surviving mutant, which is what mutation testing is for.
            run_case("j2b-bands-pd-cleared", env, st2, f, decode4, true,
                     false);
        }
        {
            // Tensor parallelism divides the head counts.
            LlamaLikePlanState st;
            LlamaLikeForwardCfg f;
            f.tp_size = 2;
            run_case("k-tp2", env, st, f, decode4, true, false);
        }
    } else if (mode == "spatial-off") {
        // PIE_SPATIAL_MASK=0: the g-case input takes the fire-level arm.
        LlamaLikePlanState st;
        run_case("z-spatial-off", env, st, base, decode4, true, true, 0, 2);
    } else if (mode == "mid-off") {
        // PIE_MIXED_MID=0: the h2 input plans no middle.
        LlamaLikePlanState st;
        run_case("z-mid-off", env, st, base, mixed_mid, false, true, 0, 3);
    } else if (mode == "graph-plan-on") {
        // PIE_PREFILL_GRAPH_PLAN=1: the prefill plan goes graph-mode.
        LlamaLikePlanState st;
        run_case("z-graph-plan", env, st, base, prefill3, false, false, 3);
    }
    return 0;
}
