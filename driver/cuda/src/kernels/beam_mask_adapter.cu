// Beam [B,P] kvm → FlashInfer packed-mask adapter (SEAM 1, charlie's G2).
//
// The beam epilogue emits `kvm` as a DENSE per-cell validity mask over the
// LOGICAL grid `[B, P*PAGE]` bool (index `lane*(P*PAGE) + page*PAGE + off`,
// `kvm[b][j·PAGE+o] = o < lens[b][j]`). The port contract stays this dense
// logical-grid bool (echo-ratified: physical layout is driver-internal). But
// FlashInfer's custom-mask prefill (`launch_attention_flashinfer_prefill_custom_bf16`,
// `MaskMode::kCustom`) wants a BIT-PACKED `[qo_len × kv_len]` bitmap per request
// (bit `q·kv_len + j`, `packed[bit/8] >> (bit%8) & 1`) with a per-request BYTE
// offset `mask_indptr` (mirrors `brle::decode` + `masked_attention_parity.cu`).
//
// For beam DECODE each beam is a 1-query request over its physical span
// `klen[b] = (np[b]-1)·PAGE + last_page_len`. This adapter packs, per beam b, the
// first `klen[b]` dense kvm cells (page-major over the beam's `np[b]` pages) into
// the packed bitmap at the beam's byte offset. The physical span is contiguous in
// the logical grid `[b·(P·PAGE) .. b·(P·PAGE)+klen[b])` because pages are laid out
// page-major and `klen` counts only the live prefix — so the logical index maps
// 1:1 to the mask column, including mid-page holes in non-last pages (the frozen
// beam's shared page).

#include <cstdint>

#include <cuda_runtime.h>

#include "cuda_check.hpp"

namespace pie_cuda_driver::kernels {

// One block per beam. Each thread packs a strided subset of the beam's klen bits.
// `kvm_dense` is [B, P*PAGE] with one byte per cell (0/1). `mask_indptr` is the
// per-beam BYTE offset into `packed` ([B+1], prefix-summed on the host from
// ceil(klen[b]/8)). `packed` is pre-zeroed by the caller.
__global__ void beam_pack_kvm_kernel(
    const std::uint8_t* __restrict__ kvm_dense,   // [B, P*PAGE] bytes (0/1)
    const std::uint32_t* __restrict__ klen,       // [B] physical span per beam
    const std::int32_t* __restrict__ mask_indptr, // [B+1] byte offsets
    std::uint8_t* __restrict__ packed,            // out: bit-packed, pre-zeroed
    int B,
    int P_PAGE)                                   // P*PAGE (logical stride)
{
    const int b = blockIdx.x;
    if (b >= B) return;
    const int kl = static_cast<int>(klen[b]);
    const std::uint8_t* row = kvm_dense + static_cast<long long>(b) * P_PAGE;
    std::uint8_t* out = packed + mask_indptr[b];
    // Each thread owns whole output BYTES to avoid RMW races on shared bytes.
    const int nbytes = (kl + 7) / 8;
    for (int byte = threadIdx.x; byte < nbytes; byte += blockDim.x) {
        std::uint8_t acc = 0;
        const int base = byte * 8;
        #pragma unroll
        for (int bit = 0; bit < 8; ++bit) {
            const int col = base + bit;
            if (col < kl && row[col] != 0) acc |= static_cast<std::uint8_t>(1u << bit);
        }
        out[byte] = acc;
    }
}

// Pack the dense [B,P*PAGE] kvm into the FlashInfer packed bitmap. `packed` must
// be zero-initialised and sized to `mask_indptr[B]` bytes. `mask_indptr` (device,
// [B+1]) is the per-beam byte-offset CSR = prefix-sum of ceil(klen[b]/8) — built
// on the host from the same klen the attention call uses.
void launch_beam_pack_kvm(
    const std::uint8_t* kvm_dense,
    const std::uint32_t* klen,
    const std::int32_t* mask_indptr,
    std::uint8_t* packed,
    int B,
    int P_PAGE,
    cudaStream_t stream)
{
    if (B <= 0) return;
    constexpr int BLOCK = 128;
    beam_pack_kvm_kernel<<<B, BLOCK, 0, stream>>>(
        kvm_dense, klen, mask_indptr, packed, B, P_PAGE);
    CUDA_CHECK(cudaGetLastError());
}

}  // namespace pie_cuda_driver::kernels
