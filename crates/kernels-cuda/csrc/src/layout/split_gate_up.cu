//===-- split_gate_up.cu - the packed gate/up launcher ---------------===//
//
// One host launcher and not one `__global__`: the device text is in
// `layout/split_gate_up.cuh`, which this file includes so the archive and the
// JIT header set hold the SAME definition rather than two that drift.
//
//===----------------------------------------------------------------------===//

// The scalar layer and the fixed-width integer names, out of the prelude.
#include "pie_device.cuh"
#include "layout/split_gate_up.hpp"

// The `__global__` this launcher fires. ONE definition of it.
#include "layout/split_gate_up.cuh"

namespace pie_cuda_driver::kernels::layout {

void split_gate_up_bf16(
    const void* packed,
    void* gate_out, void* up_out,
    int n_tokens, int inter,
    cudaStream_t stream)
{
    if (n_tokens == 0) return;
    constexpr int BLOCK = 256;
    const int xblocks = (inter + BLOCK - 1) / BLOCK;
    dim3 grid(xblocks, n_tokens);
    device::split_gate_up<device::bf16><<<grid, BLOCK, 0, stream>>>(
        static_cast<const device::bf16*>(packed),
        static_cast<device::bf16*>(gate_out),
        static_cast<device::bf16*>(up_out),
        inter);
}

}  // namespace pie_cuda_driver::kernels::layout
