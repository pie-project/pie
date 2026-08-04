#pragma once

/// Colouring a family's activation dataflow onto pool buffers.
///
/// Every family's `color_*_scratch` was the same seventeen lines: widen the
/// family's `Use` into the shared one, call `scratch::color_live_ranges`, and
/// fan the result back out per dispatch. Only two things varied -- the `Use`
/// type, which is a different struct per family with identical fields, and the
/// concurrency runs, which genuinely are the family's own.
///
/// So both vary as parameters and the adapter is written once. A family keeps
/// its own `Use` (it is produced by its own dataflow walk and named in its own
/// header) and supplies its own run ends; nothing else about the call was ever
/// family-specific.

#include <cstdint>
#include <vector>

#include "../batch/scratch_color.hpp"

namespace pie::metal::model {

/// One buffer of one dispatch, and the pool colour it resolved to.
struct ScratchBind {
    std::uint8_t bind_index = 0;
    int color = -1;
};

struct ScratchColoring {
    std::vector<std::vector<ScratchBind>> per_dispatch;
    /// Which colour each VALUE landed in, indexed by value id.
    ///
    /// `per_dispatch` answers what to bind; this answers how big to make it.
    /// Sizing a pool slot needs the widest value sharing it, and a value's
    /// width is a property of the value -- a routed model's expert stack is
    /// `experts_per_token` times taller than the dense tensor beside it, and
    /// there is no way to see that from a bind index.
    std::vector<int> color_of_value;
    int colors_used = 0;
    /// False if a value's live range crosses a dispatch that would clobber it
    /// without an intervening barrier. The encoder refuses to run on false
    /// rather than producing subtly wrong activations.
    bool hazard_free = false;
};

/// `Use` is any struct with `index`, `bind_index`, `value` and `is_write` --
/// which is what each family's dataflow walk already produces.
template <class Use>
ScratchColoring color_family_scratch(std::size_t dag_size, const std::vector<Use>& uses,
                                     const std::vector<int>& run_ends, int value_count,
                                     bool no_recycle = false) {
    std::vector<pie::metal::scratch::Use> widened;
    widened.reserve(uses.size());
    for (const Use& u : uses) {
        widened.push_back({u.index, u.bind_index, u.value, u.is_write});
    }
    const auto colored =
        pie::metal::scratch::color_live_ranges(widened, run_ends, value_count, no_recycle);

    ScratchColoring out;
    out.colors_used = colored.colors_used;
    out.color_of_value = colored.color;
    out.hazard_free = colored.hazard_free;
    out.per_dispatch.resize(dag_size);
    for (const Use& u : uses) {
        out.per_dispatch[std::size_t(u.index)].push_back(
            {u.bind_index, colored.color[std::size_t(u.value)]});
    }
    return out;
}

}  // namespace pie::metal::model
