// Stub `model/stage_hooks.hpp`.
//
// The real header is 221 lines carrying the whole hook-dispatch surface;
// `attn_score.cu` reads three fields off it. Those three are what is here —
// the same discipline as the page-mask oracle's stub, and the same check: a
// fourth field read by a later edit stops this oracle compiling.
#pragma once

namespace pie_cuda_driver::model {

struct AttentionObservation;
class HookSidebandArena;

struct StageHooks {
    /// The launch's own answer to "does any program read `AttnScore`".
    bool wants_attn_score = false;
    const AttentionObservation* observation = nullptr;
    HookSidebandArena* sideband_arena = nullptr;
};

}  // namespace pie_cuda_driver::model
