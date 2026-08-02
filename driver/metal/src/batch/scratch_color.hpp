#pragma once

/// Linear-scan colouring of activation live ranges onto a buffer pool.
///
/// Extracted so a second model family does not arrive with a second copy that
/// drifts. It is pure and knows nothing about any family: it takes the uses a
/// dataflow walk produced, the concurrency runs the encoder will honour, and
/// answers which pool buffer each value lives in.
///
/// The hazard guarantee is the overlap rule. A value's interval runs from its
/// first use to its last, EXTENDED to the end of the concurrency run its last
/// use sits in — because the encoder drops barriers inside such a run, so a
/// buffer last read at ordinal i must not be rewritten by anything still
/// running alongside i. Overlap is inclusive, which is what makes a
/// same-dispatch write-after-read and two concurrent outputs interfere and
/// therefore land in different buffers.

#include <algorithm>
#include <cstdint>
#include <limits>
#include <vector>

namespace pie::metal::scratch {

/// One buffer of one dispatch and the activation value it carries.
struct Use {
    int ordinal = 0;
    std::uint8_t bind_index = 0;
    int value = 0;
    bool is_write = false;
};

struct Coloring {
    /// `color[v]` — which pool buffer value `v` lives in.
    std::vector<int> color;
    int colors_used = 0;
    /// Self-checked: no two values whose intervals overlap share a buffer.
    bool hazard_free = false;
};

/// `run_ends[i]` is the last ordinal of the concurrency run containing `i`, or
/// `i` itself when it runs alone.
inline Coloring color_live_ranges(const std::vector<Use>& uses, const std::vector<int>& run_ends,
                                  int value_count, bool no_recycle,
                                  /// Values that must keep a buffer of their own while
                                  /// everything else still recycles. `no_recycle` preserves
                                  /// every intermediate and is the honest instrument, but it
                                  /// multiplies the pool by the DAG's length -- on a 40-layer
                                  /// mixture that is 18 GiB of scratch and the run dies before
                                  /// it dumps. Pinning ONE layer costs a layer's worth of slots
                                  /// and answers the same question, as long as the question is
                                  /// about that layer.
                                  const std::vector<bool>* pinned = nullptr) {
    Coloring out;
    const int n = value_count;
    std::vector<int> def(static_cast<std::size_t>(n), -1);
    std::vector<int> last(static_cast<std::size_t>(n), -1);
    for (const Use& u : uses) {
        int& d = def[std::size_t(u.value)];
        int& l = last[std::size_t(u.value)];
        if (d < 0 || u.ordinal < d) d = u.ordinal;
        if (u.ordinal > l) l = u.ordinal;
    }
    for (int v = 0; v < n; ++v) {
        const int l = last[std::size_t(v)];
        if (l >= 0 && l < static_cast<int>(run_ends.size())) {
            last[std::size_t(v)] = std::max(l, run_ends[std::size_t(l)]);
        }
    }

    std::vector<int> order(static_cast<std::size_t>(n));
    for (int i = 0; i < n; ++i) order[std::size_t(i)] = i;
    std::sort(order.begin(), order.end(), [&](int a, int b) {
        return def[std::size_t(a)] < def[std::size_t(b)];
    });

    out.color.assign(static_cast<std::size_t>(n), -1);
    std::vector<int> free_at;  // free_at[b]: the ordinal after which buffer b is free
    if (no_recycle) {
        // One buffer per value: preserves every intermediate for dumping, and
        // separates scratch-aliasing races from in-kernel ones.
        for (int v = 0; v < n; ++v) {
            out.color[std::size_t(v)] = v;
            free_at.push_back(last[std::size_t(v)]);
        }
    } else {
        // Pinned values take their buffers first and keep them: `free_at` at
        // INT_MAX never satisfies the `< def[v]` test below, so the linear scan
        // can see these slots and will never choose one.
        if (pinned != nullptr) {
            for (int v = 0; v < n; ++v) {
                if (std::size_t(v) < pinned->size() && (*pinned)[std::size_t(v)]) {
                    out.color[std::size_t(v)] = static_cast<int>(free_at.size());
                    free_at.push_back(std::numeric_limits<int>::max());
                }
            }
        }
        for (int v : order) {
            if (out.color[std::size_t(v)] >= 0) continue;  // pinned above
            int chosen = -1;
            for (std::size_t b = 0; b < free_at.size(); ++b) {
                if (free_at[b] < def[std::size_t(v)]) {  // strictly before: no overlap
                    chosen = static_cast<int>(b);
                    break;
                }
            }
            if (chosen < 0) {
                chosen = static_cast<int>(free_at.size());
                free_at.push_back(-1);
            }
            out.color[std::size_t(v)] = chosen;
            free_at[std::size_t(chosen)] = last[std::size_t(v)];
        }
    }
    out.colors_used = static_cast<int>(free_at.size());

    out.hazard_free = true;
    for (int a = 0; a < n && out.hazard_free; ++a) {
        if (def[std::size_t(a)] < 0) continue;
        for (int b = a + 1; b < n; ++b) {
            if (def[std::size_t(b)] < 0) continue;
            const bool overlap = std::max(def[std::size_t(a)], def[std::size_t(b)]) <=
                                 std::min(last[std::size_t(a)], last[std::size_t(b)]);
            if (overlap && out.color[std::size_t(a)] == out.color[std::size_t(b)]) {
                out.hazard_free = false;
                break;
            }
        }
    }
    return out;
}

}  // namespace pie::metal::scratch
