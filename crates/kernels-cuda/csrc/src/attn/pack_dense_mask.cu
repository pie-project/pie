// The two host launchers, and nothing else. Both `__global__`s live in
// `attn/pack_dense_mask.cuh` -- ONE definition, read by nvcc here and by
// NVRTC from the same text at run time.
//
// Dense per-lane attention mask -> FlashInfer packed-mask adapter (the
// general AttnMask-port lowering; formerly beam-specific).
//
// A program that binds the `AttnMask` descriptor port emits a DENSE per-cell
// validity mask over the LOGICAL grid `[LANES, STRIDE]` bool (index
// `lane*STRIDE + col`). The port contract stays this dense logical-grid bool
// (physical layout is driver-internal). But FlashInfer's custom-mask prefill
// (`attention_flashinfer_prefill_custom_bf16`, `MaskMode::kCustom`) wants
// a BIT-PACKED `[qo_len x kv_len]` bitmap per request (bit `q*kv_len + j`,
// `packed[bit/8] >> (bit%8) & 1`) with a per-request BYTE offset `mask_indptr`
// (mirrors `brle::decode` + `masked_attention_parity.cu`).
//
// For a 1-query-per-lane decode each lane is a 1-query request over its physical
// span `klen[lane] = (np[lane]-1)*PAGE + last_page_len`. This adapter packs, per
// lane, the first `klen[lane]` dense mask cells (page-major over the lane's live
// pages) into the packed bitmap at the lane's byte offset. The physical span is
// contiguous in the logical grid `[q_row*STRIDE .. q_row*STRIDE+klen[lane])`
// because pages are laid out page-major and `klen` counts only the live prefix,
// so the logical index maps 1:1 to the mask column, including mid-page holes in
// non-last pages (a frozen/shared page).
//
// PREFILL generalization: a lane may carry `qo_len[l] = qo_indptr[l+1]-qo_indptr[l]`
// query rows (> 1 for a variable-length prompt prefill), not just one. Each lane's
// request is then a genuine `[qo_len x klen]` custom mask: query row `qi` (global
// row `qo_indptr[l]+qi`) contributes bits `qi*klen + j` for `j in [0,klen)`, read
// from dense cell `(qo_indptr[l]+qi)*STRIDE + j`. The dense mask is `[TOTAL_Q,
// STRIDE]` (one row per QUERY token). Decode is the `qo_len==1` special case:
// query row == lane, bits `0*klen + j`, identical to the old behavior.
//
// # The one thing this file does that a launcher normally does not
//
// `StructuredMaskParams` exists twice: here, in the host `.hpp` every caller
// and the generated `shim.cpp` compile against, and in the `.cuh` as device
// text NVRTC can reach. `families/attn.rs` recorded that duplication as the
// reason this file "was not split at all". It is admissible because it is
// CHECKED rather than asserted: the four `static_assert`s below compare size,
// alignment and every field offset of the two spellings, so a field added,
// reordered or widened on either side fails the ahead-of-time build here,
// naming both types. `quant/mxfp4_marlin.cuh` mirrors an enum for the same
// reason with only a comment to hold it; this is that precedent with the
// grep replaced by the compiler.
#include "attn/pack_dense_mask.cuh"

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "attn/pack_dense_mask.hpp"

namespace pie_cuda_driver::kernels::attn {

// The bridge. `reinterpret_cast` across the two spellings is sound exactly
// when these hold, so they are stated where the cast is and not in a test
// that could be skipped.
static_assert(sizeof(StructuredMaskParams) == sizeof(device::StructuredMaskParams),
              "attn::StructuredMaskParams and its device mirror in "
              "attn/pack_dense_mask.cuh disagree on size");
static_assert(alignof(StructuredMaskParams) == alignof(device::StructuredMaskParams),
              "attn::StructuredMaskParams and its device mirror in "
              "attn/pack_dense_mask.cuh disagree on alignment");
static_assert(offsetof(StructuredMaskParams, kind) ==
                  offsetof(device::StructuredMaskParams, kind),
              "attn::StructuredMaskParams::kind moved away from its device mirror");
static_assert(offsetof(StructuredMaskParams, window) ==
                  offsetof(device::StructuredMaskParams, window),
              "attn::StructuredMaskParams::window moved away from its device mirror");
static_assert(offsetof(StructuredMaskParams, sink) ==
                  offsetof(device::StructuredMaskParams, sink),
              "attn::StructuredMaskParams::sink moved away from its device mirror");

// Pack the dense [TOTAL_Q, STRIDE] mask into the FlashInfer packed bitmap.
// `packed` must be zero-initialised and sized to `mask_indptr[LANES]` bytes.
// `mask_indptr` (device, [LANES+1]) is the per-lane byte-offset CSR = prefix-sum
// of ceil(qo_len[l]*klen[l]/8) — built on the host from the same klen/qo_indptr
// the attention call uses.
void pack_dense_mask(
    const std::uint8_t* kvm_dense,
    const std::uint32_t* klen,
    const std::uint32_t* qo_indptr,
    const std::int32_t* mask_indptr,
    std::uint8_t* packed,
    int B,
    int P_PAGE,
    cudaStream_t stream)
{
    if (B <= 0) return;
    constexpr int BLOCK = 128;
    device::pack_dense_mask<<<B, BLOCK, 0, stream>>>(
        kvm_dense, klen, qo_indptr, mask_indptr, packed, B, P_PAGE);
    CUDA_CHECK(cudaGetLastError());
}

void pack_structured_mask(
    const std::uint32_t* positions,
    const std::uint32_t* klen,
    const std::uint32_t* qo_indptr,
    const std::int32_t* mask_indptr,
    const StructuredMaskParams* masks,
    std::uint8_t* packed,
    int B,
    cudaStream_t stream) {
    if (B <= 0) return;
    constexpr int block = 128;
    device::pack_structured_mask<<<B, block, 0, stream>>>(
        positions, klen, qo_indptr, mask_indptr,
        reinterpret_cast<const device::StructuredMaskParams*>(masks),
        packed, B);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace pie_cuda_driver::kernels::attn
