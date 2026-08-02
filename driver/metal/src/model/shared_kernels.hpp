#pragma once

/// Host mirrors and launch geometry for kernels that are NOT family-specific.
///
/// A `.metal` buffer layout is a property of the kernel, not of the model that
/// happens to dispatch it. `rms_norm.metal` and `row_gather.metal` are shared
/// sources, so their host mirrors belong here once rather than once per family
/// — three private copies of the same struct drift silently, and a mismatch is
/// not a compile error: the GPU reads whatever bytes sit at the offset.

#include <cstdint>
#include <cstring>
#include <stdexcept>

#include "../mtl4_context.hpp"

namespace pie::metal::shared_kernels {

/// Params structs, replicated EXACTLY from the .metal sources.
struct RmsParams {          // rms_norm.metal:22      (buffer 3)
    float eps;
    std::uint32_t axis_size;  // feature dim
    std::uint32_t w_stride;   // 1 (contiguous)
    std::uint32_t plus_one;   // 1 applies the `(1 + w)` gain, 0 uses `w` raw
};
struct RowGatherParams {    // row_gather.metal       (buffer 3)
    std::uint32_t width;
    std::uint32_t count;
};
// The routing kernels live in `gptoss.metal` for historical reasons -- that is
// the family that first needed them -- but neither is gpt-oss-specific, and
// both are dispatched by the llama family's MoE path too. The mirrors belong
// here for the same reason the rms one does.
struct RouterParams {       // gptoss.metal:11        (buffer 3)
    std::uint32_t n_experts;
    std::uint32_t experts_per_token;
};
struct ExpertCombineParams {  // gptoss.metal         (buffer 3)
    std::uint32_t width;
    std::uint32_t experts_per_token;
};

/// Flat elementwise `dispatchThreads` grid: one thread per element, capped at a
/// 256-wide threadgroup.
inline void elementwise_dispatch(int n, Grid& g, Threadgroup& tg) {
    const int width = n < 256 ? (n > 0 ? n : 1) : 256;
    g = Grid{std::uint32_t(n > 0 ? n : 1), 1, 1};
    tg = Threadgroup{std::uint32_t(width), 1, 1};
}

/// `router_topk`: one threadgroup per token row, one lane per expert.
///
/// Rounded up to a whole simdgroup, because the kernel reduces ACROSS
/// simdgroups and a partial one would leave a reduction slot uninitialised.
/// Capped at the 1024-thread threadgroup limit, which is also the widest expert
/// count this shape can serve. 32 on gpt-oss, 128 on Qwen3-MoE.
inline void router_topk_dispatch(int n_experts, Grid& g, Threadgroup& tg, int rows = 1) {
    std::uint32_t w = std::uint32_t(n_experts < 1 ? 1 : n_experts);
    w = (w + 31u) / 32u * 32u;
    if (w > 1024u) w = 1024u;
    g = Grid{w, std::uint32_t(rows < 1 ? 1 : rows), 1};
    tg = Threadgroup{w, 1, 1};
}

/// `expert_combine`: one thread per output element, one row per `gid.y`. The k
/// slots are summed inside the kernel, so they do not appear in the grid.
inline void expert_combine_dispatch(int hidden, Grid& g, Threadgroup& tg, int rows = 1) {
    const std::uint32_t w = std::uint32_t(hidden > 0 ? hidden : 1);
    g = Grid{w, std::uint32_t(rows < 1 ? 1 : rows), 1};
    tg = Threadgroup{w < 256u ? w : 256u, 1, 1};
}

/// The routed matvec: one simdgroup per output row, `tid.z` selecting the slot.
///
/// The slot axis is the whole difference from the dense matvec, and it is why
/// the expert activations are `experts_per_token` times taller: every slot
/// writes its own full-width result, and only `expert_combine` collapses them.
inline void routed_qmv_dispatch(int N, int experts_per_token, Grid& g, Threadgroup& tg,
                                int rows = 1) {
    const std::uint32_t slots = std::uint32_t(experts_per_token > 0 ? experts_per_token : 1) *
                                std::uint32_t(rows > 0 ? rows : 1);
    g = Grid{32u, std::uint32_t(N > 0 ? N : 1), slots};
    tg = Threadgroup{32, 1, 1};
}

/// Bind a POD constant value into a fresh resident slot at (ordinal, index).
///
/// `who` names the caller in the failure, which is the only thing the families
/// ever varied here.
template <class V>
inline void bind_const(RawMetalContext& ctx, int ord, std::uint8_t idx, const V& val,
                       int* count, const char* who) {
    SlotHandle s = ctx.const_slot(ord, idx, sizeof(V));
    if (!s.valid()) {
        throw std::runtime_error(std::string(who) +
                                 " consts: heap_alloc failed (budget too small)");
    }
    std::memcpy(s.contents(), &val, sizeof(V));
    ctx.arg_bind_ordinal(ord, idx, s);
    if (count != nullptr) ++*count;
}

}  // namespace pie::metal::shared_kernels
