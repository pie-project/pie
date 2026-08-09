// The stage-hooks oracle — gate-stage-hooks.
//
// `stage_hooks.hpp` is header-only: a data struct and the inline
// `invoke_stage_hook` guard chain the generated bodies call 1,916 times.
// The oracle includes the REAL header and drives that chain with a
// recording `execute`:
//
//   * the null-hooks and null-execute no-ops;
//   * the Tier-2 truncation guard (`layer >= hook_rows_k`), at the
//     boundary on both sides;
//   * the sideband's observation defaulting (a call-site null is filled
//     from the hooks' own observation; a call-site value is NOT
//     overwritten);
//   * every argument forwarded, context included;
//   * the struct's defaults, field by field.
//
// The ambient-compat overload (`ScopedStageHooks` + the point-first
// `invoke_stage_hook`) is deliberately NOT driven: the header's own
// comment records that upstream-style qwen3_5 hooks are DORMANT this era
// (re-port pending), and the Rust port carries the same statement instead
// of machinery for a path nothing exercises.

#include <cstdint>
#include <cstdio>
#include <string>

#include "model/stage_hooks.hpp"

using pie_cuda_driver::model::AttentionObservation;
using pie_cuda_driver::model::StageHookPoint;
using pie_cuda_driver::model::StageHooks;
using pie_cuda_driver::model::StageHookSideband;
using pie_cuda_driver::model::invoke_stage_hook;

namespace {

constexpr char SEP = '\x1f';
std::string g_case;

void note(const std::string& body) {
    std::printf("%s%c%s\n", g_case.c_str(), SEP, body.c_str());
}

// Identity registry: the transcript names pointers by role, never value.
const void* g_ctx = reinterpret_cast<const void*>(0x100);
const void* g_query = reinterpret_cast<const void*>(0x200);
const AttentionObservation* g_obs_a =
    reinterpret_cast<const AttentionObservation*>(0x300);
const AttentionObservation* g_obs_b =
    reinterpret_cast<const AttentionObservation*>(0x400);

std::string who(const void* p) {
    if (p == nullptr) return "null";
    if (p == g_ctx) return "ctx";
    if (p == g_query) return "query";
    if (p == g_obs_a) return "obsA";
    if (p == g_obs_b) return "obsB";
    return "unknown";
}

void record_execute(
    void* context,
    StageHookPoint point,
    const void* query_data,
    std::uint32_t query_rows,
    std::uint32_t query_columns,
    std::uint32_t layer,
    cudaStream_t stream,
    bool query_is_f32,
    const StageHookSideband& sideband) {
    note("execute ctx=" + who(context) + " point=" +
         std::to_string(static_cast<int>(point)) + " q=" + who(query_data) +
         " rows=" + std::to_string(query_rows) + " cols=" +
         std::to_string(query_columns) + " layer=" + std::to_string(layer) +
         " stream=" + (stream == nullptr ? "s0" : "s?") + " f32=" +
         std::to_string(query_is_f32 ? 1 : 0) + " obs=" +
         who(sideband.observation) + " scores=" + who(sideband.scores) +
         " sink=" + who(sideband.mask_sink));
}

}  // namespace

int main() {
    // a. Defaults, field by field.
    g_case = "a-defaults";
    {
        const StageHooks h;
        note("context=" + who(h.context));
        note("wants_attn_score=" + std::to_string(h.wants_attn_score ? 1 : 0));
        note("attn_score_window=" + std::to_string(h.attn_score_window));
        note("wants_page_mask=" + std::to_string(h.wants_page_mask ? 1 : 0));
        note("hook_free_prefix_rows=" +
             std::to_string(h.hook_free_prefix_rows));
        note("hook_rows_k=" + std::to_string(h.hook_rows_k));
        note("sideband_arena=" +
             std::string(h.sideband_arena == nullptr ? "null" : "set"));
        note("observation=" + who(h.observation));
        note("execute=" +
             std::string(h.execute == nullptr ? "null" : "set"));
        note("prepare_replay=" +
             std::string(h.prepare_replay == nullptr ? "null" : "set"));
        note("verify_replay_capture=" +
             std::string(h.verify_replay_capture == nullptr ? "null" : "set"));
        const StageHookSideband s;
        note("sideband obs=" + who(s.observation) + " scores=" +
             who(s.scores) + " sink=" + who(s.mask_sink));
    }

    // b. The no-op arms cost one branch and record nothing.
    g_case = "b-noop";
    {
        invoke_stage_hook(nullptr, StageHookPoint::OnAttn, g_query, 4, 64, 0,
                          nullptr);
        StageHooks silent;
        invoke_stage_hook(&silent, StageHookPoint::OnAttn, g_query, 4, 64, 0,
                          nullptr);
        note("done");
    }

    // c. Argument forwarding, sideband defaulting included.
    g_case = "c-forward";
    {
        StageHooks h;
        h.context = const_cast<void*>(g_ctx);
        h.observation = g_obs_a;
        h.execute = record_execute;
        // A call-site null observation is filled from the hooks' own.
        invoke_stage_hook(&h, StageHookPoint::OnAttnProj, g_query, 7, 128, 3,
                          nullptr, false);
        // A call-site observation is NOT overwritten.
        invoke_stage_hook(&h, StageHookPoint::OnAttn, g_query, 1, 64, 9,
                          nullptr, true, StageHookSideband{g_obs_b});
        // The f32 default is false.
        invoke_stage_hook(&h, StageHookPoint::OnAttn, g_query, 2, 32, 0,
                          nullptr);
    }

    // d. The Tier-2 truncation guard, at the boundary on both sides.
    g_case = "d-truncation";
    {
        StageHooks h;
        h.context = const_cast<void*>(g_ctx);
        h.observation = g_obs_a;
        h.execute = record_execute;
        h.hook_rows_k = 5;
        for (std::uint32_t layer : {0u, 4u, 5u, 6u}) {
            invoke_stage_hook(&h, StageHookPoint::OnAttn, g_query, 1, 64,
                              layer, nullptr);
        }
        note("swept");
        h.hook_rows_k = 0;
        invoke_stage_hook(&h, StageHookPoint::OnAttn, g_query, 1, 64, 0,
                          nullptr);
        note("k0-swept");
    }

    return 0;
}
