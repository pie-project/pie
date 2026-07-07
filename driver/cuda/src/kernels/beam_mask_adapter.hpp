// Beam [B,P] kvm → FlashInfer packed-mask adapter (SEAM 1, charlie's G2). See
// beam_mask_adapter.cu for the full contract.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

// Pack the beam epilogue's dense [B, P*PAGE] per-cell kvm (one byte/cell, 0/1)
// into FlashInfer's bit-packed custom mask, per beam, over its klen[b] physical
// span. `mask_indptr` ([B+1], device) is the per-beam BYTE-offset CSR
// (prefix-sum of ceil(klen[b]/8)); `packed` must be pre-zeroed and sized to
// `mask_indptr[B]` bytes. Bit layout matches `brle::decode`: bit `q·kv_len + j`
// (q=0 for the single beam query), `packed[bit/8] >> (bit%8) & 1`.
void launch_beam_pack_kvm(
    const std::uint8_t* kvm_dense,        // [B, P*PAGE] bytes (0/1)
    const std::uint32_t* klen,            // [B] physical span per beam
    const std::int32_t* mask_indptr,      // [B+1] byte offsets
    std::uint8_t* packed,                 // out: bit-packed, pre-zeroed
    int B,
    int P_PAGE,                           // P*PAGE logical stride
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
