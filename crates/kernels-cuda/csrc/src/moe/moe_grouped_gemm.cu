// The grouped GEMM's launcher and its support predicate, and nothing else.
//
// The `__global__` moved to `moe_grouped_gemm.cuh`, which NVRTC compiles from
// a string and nvcc reads through the `#include` below -- one definition, two
// compilers. What stays here is the host's: the shape test that decides
// whether this kernel or cuBLAS runs, and the grid it is fired on.
#include "pie_device.cuh"
#include "moe/moe_grouped_gemm.cuh"
#include "moe/moe_grouped_gemm.hpp"

namespace pie_cuda_driver::kernels::moe {

// Above this K cuBLAS's tuned mainloop beats the early exit, and a HOST
// constant because the decision is the launcher's: the kernel is correct at
// any K and this is the bound at which firing it stops paying.
constexpr int kShortK = 512;

bool moe_grouped_gemm_bf16_supported(int M, int N, int K) {
    // Measured on Qwen3.6-35B-A3B tp2 decode against cuBLAS:
    //   down     K=256   7.94 -> 5.91 ms   taken
    //   gate_up  K=2048  11.08 -> 11.98    left on cuBLAS (see the header)
    return M == device::kFrag && N > 0 && K > 0 && K <= kShortK &&
           (N % device::kNTile) == 0 && (K % device::kFrag) == 0;
}

void moe_grouped_gemm_bf16(
    const void* a,
    const void* weight_base,
    void* c,
    const device::i32* expert_ids,
    int max_blocks,
    int M,
    int N,
    int K,
    cudaStream_t stream)
{
    if (max_blocks <= 0 || !moe_grouped_gemm_bf16_supported(M, N, K)) return;
    // `max_blocks` is a host-side bound on the padded batch, not an extent of
    // any operand -- which is why no launch rule states this grid.
    const dim3 grid(N / device::kNTile, max_blocks);
    device::moe_grouped_gemm<device::bf16><<<grid, device::kGemmWarps * 32, 0, stream>>>(
        static_cast<const device::bf16*>(a),
        static_cast<const device::bf16*>(weight_base),
        static_cast<device::bf16*>(c), expert_ids, N, K);
}

}  // namespace pie_cuda_driver::kernels::moe
