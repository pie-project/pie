#include "ssm_scan.cuh"

#include <cuda_bf16.h>

// Lifted from driver/cuda/src/kernels/nemotron_h.cu — the
// `mamba_ssm_batched_warp_kernel` device kernel plus the warp-kernel dispatch
// path of `launch_nemotron_mamba_ssm_batched_bf16`. The recurrence body is
// VERBATIM; the only edits are de-branding:
//   * namespace pie_cuda_driver::kernels -> pie_cuda_device::kernels
//   * launcher renamed launch_nemotron_mamba_ssm_batched_bf16 ->
//     ssm_selective_scan_bf16 (the "nemotron"/"mamba" branding dropped; the
//     math is the model-agnostic Mamba-2/SSD scan)
// The prefill-register and shared-memory-atomicAdd variants, plus the env-var
// kernel switches that selected between them, are NOT lifted (see ssm_scan.cuh).
// This entry point unconditionally uses the warp-reduction kernel, which
// produces the same result for both prefill and decode.

namespace pie_cuda_device::kernels {

namespace {

__device__ __forceinline__ float bf16_to_float(const __nv_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float v) {
    return __float2bfloat16(v);
}

__device__ __forceinline__ float softplus_f(float x) {
    // Stable enough for the dt range in Nemotron-H checkpoints.
    return x > 20.f ? x : log1pf(__expf(x));
}

__device__ __forceinline__ float warp_sum(float v) {
    unsigned mask = 0xffffffffu;
    for (int off = 16; off > 0; off >>= 1) {
        v += __shfl_down_sync(mask, v, off);
    }
    return v;
}

// One CUDA block owns one (request, mamba-head) stream. Threads cooperate over
// the head_dim x state_size slab for that head, preserving token order inside
// the request. One warp maps to one head_dim channel and reduces the state axis
// inside the warp (avoids the shared-memory atomicAdd hot path of the generic
// kernel).
__global__ void mamba_ssm_batched_warp_kernel(
    const __nv_bfloat16* __restrict__ conv_out,
    const __nv_bfloat16* __restrict__ dt_in,
    const float* __restrict__ A,
    const float* __restrict__ D,
    const float* __restrict__ dt_bias,
    const float* __restrict__ dt_precomputed,
    const float* __restrict__ dA_precomputed,
    __nv_bfloat16* __restrict__ state_base,
    const std::int32_t* __restrict__ slot_ids,
    const std::uint32_t* __restrict__ qo_indptr,
    __nv_bfloat16* __restrict__ y,
    int num_heads,
    int head_dim,
    int state_size,
    int n_groups,
    int conv_dim,
    int intermediate,
    float time_step_min)
{
    const int r = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int num_warps = blockDim.x >> 5;
    const int t0 = static_cast<int>(qo_indptr[r]);
    const int n_tokens = static_cast<int>(qo_indptr[r + 1]) - t0;
    if (n_tokens <= 0) return;

    const int slot = slot_ids ? slot_ids[r] : 0;
    const long long state_stride =
        static_cast<long long>(num_heads) * head_dim * state_size;
    __nv_bfloat16* state =
        state_base + static_cast<long long>(slot) * state_stride +
        static_cast<long long>(h) * head_dim * state_size;

    const int heads_per_group = num_heads / n_groups;
    const int group = h / heads_per_group;
    const float A_h = A[h];
    const float D_h = D[h];
    const float dt_b = dt_bias[h];
    const int bc_base = intermediate + group * state_size;
    const int c_base = intermediate + n_groups * state_size +
                       group * state_size;

    extern __shared__ float bc_smem[];
    float* b_s = bc_smem;
    float* c_s = bc_smem + state_size;

    for (int local_t = 0; local_t < n_tokens; ++local_t) {
        const int row = t0 + local_t;
        const long long dt_idx = static_cast<long long>(row) * num_heads + h;
        const float dt = dt_precomputed != nullptr
            ? dt_precomputed[dt_idx]
            : fmaxf(softplus_f(bf16_to_float(dt_in[dt_idx]) + dt_b),
                    time_step_min);
        const float dA = dA_precomputed != nullptr
            ? dA_precomputed[dt_idx]
            : __expf(dt * A_h);
        const __nv_bfloat16* row_conv =
            conv_out + static_cast<long long>(row) * conv_dim;

        for (int s = tid; s < state_size; s += blockDim.x) {
            b_s[s] = bf16_to_float(row_conv[bc_base + s]);
            c_s[s] = bf16_to_float(row_conv[c_base + s]);
        }
        __syncthreads();

        for (int dim = warp; dim < head_dim; dim += num_warps) {
            const float x = bf16_to_float(row_conv[h * head_dim + dim]);
            float sum = 0.f;
            for (int s = lane; s < state_size; s += 32) {
                const int idx = dim * state_size + s;
                const float old = bf16_to_float(state[idx]);
                const float next = old * dA + (dt * b_s[s]) * x;
                state[idx] = float_to_bf16(next);
                sum += next * c_s[s];
            }
            sum = warp_sum(sum);
            if (lane == 0) {
                y[static_cast<long long>(row) * intermediate +
                  h * head_dim + dim] = float_to_bf16(sum + D_h * x);
            }
        }
        __syncthreads();
    }
}

}  // namespace

void ssm_selective_scan_bf16(
    const void* conv_out,
    const void* dt,
    const float* A,
    const float* D,
    const float* dt_bias,
    const float* dt_precomputed,
    const float* dA_precomputed,
    void* ssm_state_base,
    const std::int32_t* slot_ids,
    const std::uint32_t* qo_indptr,
    void* y,
    int R,
    int num_heads,
    int head_dim,
    int state_size,
    int n_groups,
    int conv_dim,
    int intermediate,
    float time_step_min,
    cudaStream_t stream)
{
    if (R <= 0 || num_heads <= 0 || head_dim <= 0 || state_size <= 0) return;
    constexpr int BLOCK = 256;
    dim3 grid(R, num_heads);
    const std::size_t shared =
        2ull * static_cast<std::size_t>(state_size) * sizeof(float);
    mamba_ssm_batched_warp_kernel<<<grid, BLOCK, shared, stream>>>(
        static_cast<const __nv_bfloat16*>(conv_out),
        static_cast<const __nv_bfloat16*>(dt),
        A,
        D,
        dt_bias,
        dt_precomputed,
        dA_precomputed,
        static_cast<__nv_bfloat16*>(ssm_state_base),
        slot_ids, qo_indptr,
        static_cast<__nv_bfloat16*>(y),
        num_heads, head_dim, state_size, n_groups,
        conv_dim, intermediate, time_step_min);
}

}  // namespace pie_cuda_device::kernels
