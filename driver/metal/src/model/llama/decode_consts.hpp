#pragma once

/// The llama families' per-dispatch constants.
///
/// Bound per ORDINAL rather than per kind, like every other family here: the
/// argument table is keyed by ordinal, and a kind appears once per layer.

#include <vector>

#include "../../mtl4_context.hpp"
#include "decode_step.hpp"
#include "geometry.hpp"

namespace pie::metal::llama {

/// A matvec's `(in_vec, out_vec)`, or `{0, 0}` when the kind is not one.
///
/// No `layer` parameter, unlike gemma4's: llama's layers are uniform. Every
/// layer has the same head width, the same MLP width and the same rope base,
/// which is most of why this family is cheap.
struct KN {
    int K = 0;
    int N = 0;
};
KN qmv_kn(Kind k, const LlamaGeometry& g);

/// Bind every constant the DAG's dispatches read. Returns how many were bound,
/// which is the number a test can pin.
///
/// `rows` is the token count: an elementwise kernel over a contiguous
/// [rows, width] buffer counts rows*width. Defaults to the decode path's row.
///
/// `paged` picks which attention ABI the two KV kinds are bound against.
/// `Kind::Sdpa` and `Kind::KvAppend` are ONE kind each whose kernel changes with
/// the batch -- contiguous ring at M=1, page table at M>1 -- and the two ABIs
/// put different meanings at the same slot indices. Binding the contiguous
/// constants against the paged shader is not a crash: `bind::Sdpa::KHeadStride`
/// is `bind::SdpaPaged::ReqOfToken`, so it is a stride read as a pointer.
int bind_llama_consts(RawMetalContext& ctx, const std::vector<Dispatch>& dag,
                      const LlamaGeometry& g, int rows = 1, bool paged = false);

}  // namespace pie::metal::llama
