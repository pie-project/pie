// The host launchers, and nothing else. Both `__global__`s live in
// `attn/attn_sink.cuh` -- ONE definition, read by nvcc here and by NVRTC
// from the same text at run time.
//
// `<cuda_bf16.h>` is gone with them: NVRTC answers no CUDA header, so the
// device text speaks the prelude's `device::bf16` and this file speaks the
// same type rather than `__nv_bfloat16`. They are the same two bytes.
#include "attn/attn_sink.cuh"
#include "attn/attn_sink.hpp"

namespace pie_cuda_driver::kernels::attn {

void lse_log2_to_ln(float* lse, int n, cudaStream_t stream) {
    if (n <= 0) return;
    const int block = 256;
    device::lse_log2_to_ln<float><<<(n + block - 1) / block, block, 0, stream>>>(
        lse, static_cast<::pie_cuda_driver::kernels::device::usize>(n));
}

void attention_sink_rescale_bf16(
    void*        o,
    const float* lse,
    const void*  sinks,
    int N,
    int num_q_heads,
    int head_dim,
    cudaStream_t stream)
{
    using bf16 = ::pie_cuda_driver::kernels::device::bf16;
    const dim3 grid(static_cast<unsigned>(N), static_cast<unsigned>(num_q_heads));
    const int block = (head_dim < 32) ? 32 : (head_dim > 128 ? 128 : head_dim);
    device::attn_sink_rescale<bf16><<<grid, block, 0, stream>>>(
        static_cast<bf16*>(o),
        lse,
        static_cast<const bf16*>(sinks),
        N, num_q_heads, head_dim);
}

}  // namespace pie_cuda_driver::kernels::attn
