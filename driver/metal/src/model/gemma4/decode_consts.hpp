#pragma once

/// Gemma 4's per-dispatch constants.
///
/// Unlike qwen3.5's, these are per LAYER rather than per kind: head_dim, the
/// RoPE base, the rotated fraction and the MLP width all depend on the layer's
/// attention type and on whether it sits in the KV-shared range.

#include <vector>

#include "../../mtl4_context.hpp"
#include "decode_step.hpp"
#include "geometry.hpp"

namespace pie::metal::gemma4 {

/// A matvec's `(in_vec, out_vec)`, or `{0, 0}` when the kind is not one.
struct KN {
    int K = 0;
    int N = 0;
};
KN qmv_kn(Kind k, const Gemma4Geometry& g, int layer);

/// Bind every constant the DAG's dispatches read. Returns how many were bound,
/// which is the number a test can pin.
/// `rows` is the token count: a GEMM must be told M, and an elementwise
/// kernel over a contiguous [rows, width] buffer counts rows*width. Defaults to
/// the decode path's single row.
int bind_gemma4_consts(RawMetalContext& ctx, const std::vector<Dispatch>& dag,
                       const Gemma4Geometry& g, int rows = 1);

}  // namespace pie::metal::gemma4
