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

/// Bind the FP16 staging buffer and its element count to every projection whose
/// GEMM reads a staged input. Returns how many were bound.
int bind_gemma4_fp16_qmm(RawMetalContext& ctx, const std::vector<Dispatch>& dag,
                         const Gemma4Geometry& g, int rows, int head_rows,
                         const SlotHandle& staging, std::vector<SlotHandle>& keep);

/// Bind every constant the DAG's dispatches read. Returns how many were bound,
/// which is the number a test can pin.
///
/// `rows` is the token count: a GEMM must be told M, and an elementwise kernel
/// over a contiguous [rows, width] buffer counts rows*width. Defaults to the
/// decode path's single row.
///
/// `paged` picks which attention ABI the two KV kinds are bound against.
/// `Kind::Sdpa` and `Kind::KvAppend` are ONE kind each whose kernel changes with
/// the batch -- contiguous ring at M=1, page table at M>1 -- and the two ABIs
/// put different meanings at the same slot indices. Binding the contiguous
/// constants against the paged shader is not a crash: `bind::Sdpa::KHeadStride`
/// is `bind::SdpaPaged::ReqOfToken`, so it is a stride read as a pointer.
/// `head_rows` is how many rows the fire will SAMPLE -- what `Kind::RowGather`
/// compacts, and what the tail after it runs on. 0 means "all of them", which
/// is what a test that reads every row wants.
int bind_gemma4_consts(RawMetalContext& ctx, const std::vector<Dispatch>& dag,
                       const Gemma4Geometry& g, int rows = 1, bool paged = false,
                       int head_rows = 0);

}  // namespace pie::metal::gemma4
