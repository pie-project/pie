// Stub `model/stage_hooks.hpp`.
//
// The real header is 221 lines carrying the whole hook-dispatch surface;
// `attn_page_mask.cu` reads three fields off it. Those three are what is here.
//
// Keeping the stub this narrow is itself a check: if a later edit to
// `attn_page_mask.cu` starts reading a fourth field, this oracle stops
// compiling and the new dependency has to be looked at rather than absorbed.
#pragma once

namespace pie_cuda_driver::model {

struct AttentionObservation;
class HookSidebandArena;

struct StageHooks {
    /// The launch's own answer to "does any program write the mask sink".
    bool wants_page_mask = false;
    const AttentionObservation* observation = nullptr;
    HookSidebandArena* sideband_arena = nullptr;
};

}  // namespace pie_cuda_driver::model
