// Dense per-lane attention mask → FlashInfer packed-mask adapter (the general
// AttnMask-port lowering; formerly beam-specific).
//
// A program that binds the `AttnMask` descriptor port emits a DENSE per-cell
// validity mask over the LOGICAL grid `[LANES, STRIDE]` bool (index
// `lane*STRIDE + col`). The port contract stays this dense logical-grid bool
// (physical layout is driver-internal). But FlashInfer's custom-mask prefill
// (`launch_attention_flashinfer_prefill_custom_bf16`, `MaskMode::kCustom`) wants
// a BIT-PACKED `[qo_len × kv_len]` bitmap per request (bit `q·kv_len + j`,
// `packed[bit/8] >> (bit%8) & 1`) with a per-request BYTE offset `mask_indptr`
// (mirrors `brle::decode` + `masked_attention_parity.cu`).
//
// For a 1-query-per-lane decode each lane is a 1-query request over its physical
// span `klen[lane] = (np[lane]-1)·PAGE + last_page_len`. This adapter packs, per
// lane, the first `klen[lane]` dense mask cells (page-major over the lane's live
// pages) into the packed bitmap at the lane's byte offset. The physical span is
// contiguous in the logical grid `[q_row·STRIDE .. q_row·STRIDE+klen[lane])`
// because pages are laid out page-major and `klen` counts only the live prefix,
// so the logical index maps 1:1 to the mask column, including mid-page holes in
// non-last pages (a frozen/shared page).
//
// PREFILL generalization: a lane may carry `qo_len[l] = qo_indptr[l+1]-qo_indptr[l]`
// query rows (> 1 for a variable-length prompt prefill), not just one. Each lane's
// request is then a genuine `[qo_len × klen]` custom mask: query row `qi` (global
// row `qo_indptr[l]+qi`) contributes bits `qi·klen + j` for `j in [0,klen)`, read
// from dense cell `(qo_indptr[l]+qi)·STRIDE + j`. The dense mask is `[TOTAL_Q,
// STRIDE]` (one row per QUERY token). Decode is the `qo_len==1` special case:
// query row == lane, bits `0·klen + j`, identical to the old behavior.

#include <cstdint>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels/pack_dense_mask.hpp"

namespace pie_cuda_driver::kernels {

// One block per lane. Each thread packs a strided subset of the lane's
// `qo_len·klen` bits. `mask_dense` is [TOTAL_Q, STRIDE] with one byte per cell
// (0/1). `mask_indptr` is the per-lane BYTE offset into `packed` ([LANES+1],
// prefix-summed on the host from ceil(qo_len[l]·klen[l]/8)). `qo_indptr`
// ([LANES+1]) gives each lane's query-row range. `packed` is pre-zeroed.
__global__ void pack_dense_mask_kernel(
    const std::uint8_t* __restrict__ kvm_dense,   // [TOTAL_Q, STRIDE] bytes (0/1)
    const std::uint32_t* __restrict__ klen,       // [LANES] physical span per lane
    const std::uint32_t* __restrict__ qo_indptr,  // [LANES+1] query-row CSR
    const std::int32_t* __restrict__ mask_indptr, // [LANES+1] byte offsets
    std::uint8_t* __restrict__ packed,            // out: bit-packed, pre-zeroed
    int B,
    int P_PAGE)                                   // STRIDE (logical row stride)
{
    const int b = blockIdx.x;
    if (b >= B) return;
    const int kl = static_cast<int>(klen[b]);
    const int qo_lo = static_cast<int>(qo_indptr[b]);
    const int qo_len = static_cast<int>(qo_indptr[b + 1]) - qo_lo;
    if (kl <= 0 || qo_len <= 0) return;
    const long long total_bits =
        static_cast<long long>(qo_len) * static_cast<long long>(kl);
    std::uint8_t* out = packed + mask_indptr[b];
    // Each thread owns whole output BYTES to avoid RMW races on shared bytes.
    const int nbytes = static_cast<int>((total_bits + 7) / 8);
    for (int byte = threadIdx.x; byte < nbytes; byte += blockDim.x) {
        std::uint8_t acc = 0;
        const long long base = static_cast<long long>(byte) * 8;
        #pragma unroll
        for (int bit = 0; bit < 8; ++bit) {
            const long long gbit = base + bit;
            if (gbit < total_bits) {
                const int qi = static_cast<int>(gbit / kl);
                const int col = static_cast<int>(gbit % kl);
                const std::uint8_t* row =
                    kvm_dense + static_cast<long long>(qo_lo + qi) * P_PAGE;
                if (row[col] != 0) acc |= static_cast<std::uint8_t>(1u << bit);
            }
        }
        out[byte] = acc;
    }
}

// Pack the dense [TOTAL_Q, STRIDE] mask into the FlashInfer packed bitmap.
// `packed` must be zero-initialised and sized to `mask_indptr[LANES]` bytes.
// `mask_indptr` (device, [LANES+1]) is the per-lane byte-offset CSR = prefix-sum
// of ceil(qo_len[l]·klen[l]/8) — built on the host from the same klen/qo_indptr
// the attention call uses.
void launch_pack_dense_mask(
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
    pack_dense_mask_kernel<<<B, BLOCK, 0, stream>>>(
        kvm_dense, klen, qo_indptr, mask_indptr, packed, B, P_PAGE);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void pack_structured_mask_kernel(
    const std::uint32_t* __restrict__ positions,
    const std::uint32_t* __restrict__ klen,
    const std::uint32_t* __restrict__ qo_indptr,
    const std::int32_t* __restrict__ mask_indptr,
    const StructuredMaskParams* __restrict__ masks,
    std::uint8_t* __restrict__ packed,
    int B) {
    const int request = blockIdx.x;
    if (request >= B) return;
    const std::uint32_t keys = klen[request];
    const std::uint32_t query_begin = qo_indptr[request];
    const std::uint32_t queries =
        qo_indptr[request + 1] - query_begin;
    const std::uint64_t bits =
        static_cast<std::uint64_t>(queries) * keys;
    const auto descriptor = masks[request];
    std::uint8_t* output = packed + mask_indptr[request];
    const std::uint32_t bytes =
        static_cast<std::uint32_t>((bits + 7) / 8);
    for (std::uint32_t byte = threadIdx.x;
         byte < bytes;
         byte += blockDim.x) {
        std::uint8_t value = 0;
        const std::uint64_t begin =
            static_cast<std::uint64_t>(byte) * 8;
        #pragma unroll
        for (std::uint32_t bit = 0; bit < 8; ++bit) {
            const std::uint64_t index = begin + bit;
            if (index >= bits) break;
            const std::uint32_t query =
                static_cast<std::uint32_t>(index / keys);
            const std::uint32_t key =
                static_cast<std::uint32_t>(index % keys);
            const std::uint32_t position =
                positions[query_begin + query];
            const std::uint32_t key_plus_window =
                descriptor.window > UINT32_MAX - key
                    ? UINT32_MAX
                    : key + descriptor.window;
            const bool causal = key <= position;
            const bool in_window =
                causal && key_plus_window > position;
            const bool allowed = causal &&
                (descriptor.kind == 1 ||
                 (descriptor.kind == 2 && in_window) ||
                 (descriptor.kind == 3 &&
                  (key < descriptor.sink || in_window)));
            if (allowed) {
                value |= static_cast<std::uint8_t>(1u << bit);
            }
        }
        output[byte] = value;
    }
}

void launch_pack_structured_mask(
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
    pack_structured_mask_kernel<<<B, block, 0, stream>>>(
        positions, klen, qo_indptr, mask_indptr, masks, packed, B);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace pie_cuda_driver::kernels
