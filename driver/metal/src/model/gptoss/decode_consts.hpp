#pragma once

/// GPT-OSS's per-dispatch constants.

#include <cstddef>
#include <vector>

#include "../../mtl4_context.hpp"
#include "decode_step.hpp"
#include "geometry.hpp"

namespace pie::metal::gptoss {

/// A matvec's `(in_vec, out_vec)`, or `{0, 0}` when the kind is not one.
struct KN {
    int K = 0;
    int N = 0;
};
KN qmv_kn(Kind k, const GptOssGeometry& g);

/// Bind every constant the DAG's dispatches read. Returns how many were bound.
///
/// `rows` is the token count. `paged` picks which attention ABI the KV kinds are
/// bound against — the same choice gemma4's binder makes, for the same reason:
/// the contiguous and paged shaders put different meanings at the same slots.
/// `head_rows` is how many rows the fire will SAMPLE -- what `Kind::RowGather`
/// compacts, and what the tail after it runs on. 0 means "all of them".
int bind_gptoss_consts(RawMetalContext& ctx, const std::vector<Dispatch>& dag,
                       const GptOssGeometry& g, int rows = 1, bool paged = false,
                       int head_rows = 0);

}  // namespace pie::metal::gptoss
