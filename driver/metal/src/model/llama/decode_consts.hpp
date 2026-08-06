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

/// Bind the split-K side of every dense projection that takes one.
///
/// Three numbers per dispatch, all of which change with `rows`: where the
/// partial slices go (ordinal 8), how long a K partition is (ordinal 9) and how
/// far apart the slices sit (ordinal 10), plus the partition count the reduce
/// sums (ordinal 11). One allocation is divided into concurrency lanes so a
/// reduce can overlap the next independent projection's GEMM.
///
/// `keep` receives the small constant buffers; the caller owns them for as long
/// as the argument tables point at them. Buffers are shared between dispatches
/// that want the same triple, which on this family means one per projection
/// shape rather than one per layer.
///
/// Returns how many dispatches were split, which is what a test can pin.
int bind_llama_splitk(RawMetalContext& ctx, const std::vector<Dispatch>& dag,
                      const LlamaGeometry& g, int rows, const SlotHandle& partials,
                      std::vector<SlotHandle>& keep, int requests = 1);

/// Bind the one-per-source BF16->FP16 staging used by dense g64/b4 QMMs.
int bind_llama_fp16_qmm(RawMetalContext& ctx, const std::vector<Dispatch>& dag,
                        const LlamaGeometry& g, int rows, int head_rows, int requests,
                        const SlotHandle& staging, std::vector<SlotHandle>& keep);

}  // namespace pie::metal::llama
