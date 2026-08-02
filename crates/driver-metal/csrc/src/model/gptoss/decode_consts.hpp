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

/// Point every FP16-staged projection at the shared staging buffer and tell
/// the cast pass how much of it to fill.
///
/// Buffer 12 is the tile loader's source and 13 is the cast's element count.
/// Rebound per fire because both depend on the row count -- and the count is a
/// device buffer rather than a constant because the cast kernel is shared with
/// llama, which binds it the same way. Distinct counts are few (one per K), so
/// they are pooled and `keep` owns them until the next bind replaces them.
int bind_gptoss_fp16_qmm(RawMetalContext& ctx, const std::vector<Dispatch>& dag,
                         const GptOssGeometry& g, int rows, int head_rows,
                         const SlotHandle& staging, std::vector<SlotHandle>& keep);

}  // namespace pie::metal::gptoss
