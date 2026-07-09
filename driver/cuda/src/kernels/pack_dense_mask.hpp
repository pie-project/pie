// Dense per-lane attention mask → FlashInfer packed-mask adapter (the general
// AttnMask-port lowering; formerly beam_mask_adapter). See pack_dense_mask.cu
// for the full contract.
#pragma once

#include <cstdint>

#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels {

// Pack a program's dense `[TOTAL_Q, STRIDE]` per-cell mask (one byte/cell, 0/1,
// from an `AttnMask` descriptor port) into FlashInfer's bit-packed custom mask,
// per lane, over its `qo_len[l]` query rows × `klen[l]` physical span.
// `qo_indptr` ([LANES+1], device) gives each lane's query-row range;
// `mask_indptr` ([LANES+1], device) is the per-lane BYTE-offset CSR (prefix-sum
// of ceil(qo_len[l]·klen[l]/8)); `packed` must be pre-zeroed and sized to
// `mask_indptr[LANES]` bytes. Bit layout matches `brle::decode`: bit
// `q·kv_len + j` (q the within-lane query row), `packed[bit/8] >> (bit%8) & 1`.
// Decode is the `qo_len==1` case (q=0, query row == lane).
void launch_pack_dense_mask(
    const std::uint8_t* kvm_dense,        // [TOTAL_Q, STRIDE] bytes (0/1)
    const std::uint32_t* klen,            // [LANES] physical span per lane
    const std::uint32_t* qo_indptr,       // [LANES+1] query-row CSR
    const std::int32_t* mask_indptr,      // [LANES+1] byte offsets
    std::uint8_t* packed,                 // out: bit-packed, pre-zeroed
    int B,
    int P_PAGE,                           // STRIDE (logical row stride)
    cudaStream_t stream);

}  // namespace pie_cuda_driver::kernels
