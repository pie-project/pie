#include "mamba_proj.cuh"

#include <cuda_bf16.h>

// Lifted VERBATIM from driver/cuda/src/kernels/nemotron_h.cu — the
// mamba_split_kernel / mamba_split_conv_dt_kernel, prepare_mamba_params_kernel,
// prepare_mamba_dt_da_kernel and zamba_rmsnorm_gated_kernel device kernels plus
// their launcher dispatch logic. The recurrence/index math is unchanged; the
// only edits are de-branding:
//   * namespace pie_cuda_driver::kernels -> pie_cuda_device::kernels
//   * launch_nemotron_mamba_split_bf16        -> mamba_split_bf16
//   * launch_nemotron_prepare_mamba_params    -> prepare_mamba_params_bf16
//   * launch_nemotron_prepare_mamba_dt_da     -> prepare_mamba_dt_da_bf16
//   * launch_zamba_rmsnorm_gated_bf16         -> rmsnorm_gated_bf16

namespace pie_cuda_device::kernels {

namespace {

__device__ __forceinline__ float bf16_to_float(const __nv_bfloat16 v) {
    return __bfloat162float(v);
}

__device__ __forceinline__ float silu_f(float x) {
    return x / (1.f + __expf(-x));
}

__device__ __forceinline__ float softplus_f(float x) {
    // Stable enough for the dt range in Nemotron-H checkpoints.
    return x > 20.f ? x : log1pf(__expf(x));
}

__global__ void mamba_split_kernel(
    const __nv_bfloat16* __restrict__ projected,
    __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ conv_in,
    __nv_bfloat16* __restrict__ dt,
    int projection_dim,
    int intermediate,
    int conv_dim,
    int num_heads,
    int total)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int row = i / projection_dim;
    const int col = i - row * projection_dim;
    const auto v = projected[i];
    if (col < intermediate) {
        gate[static_cast<long long>(row) * intermediate + col] = v;
    } else if (col < intermediate + conv_dim) {
        conv_in[static_cast<long long>(row) * conv_dim +
                (col - intermediate)] = v;
    } else if (col < intermediate + conv_dim + num_heads) {
        dt[static_cast<long long>(row) * num_heads +
           (col - intermediate - conv_dim)] = v;
    }
}

__global__ void mamba_split_conv_dt_kernel(
    const __nv_bfloat16* __restrict__ projected,
    __nv_bfloat16* __restrict__ conv_in,
    __nv_bfloat16* __restrict__ dt,
    int projection_dim,
    int intermediate,
    int conv_dim,
    int num_heads,
    int total)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int row = i / (conv_dim + num_heads);
    const int col = i - row * (conv_dim + num_heads);
    const __nv_bfloat16* src =
        projected + static_cast<long long>(row) * projection_dim + intermediate;
    if (col < conv_dim) {
        conv_in[static_cast<long long>(row) * conv_dim + col] = src[col];
    } else {
        dt[static_cast<long long>(row) * num_heads + (col - conv_dim)] =
            src[col];
    }
}

__global__ void prepare_mamba_params_kernel(
    const __nv_bfloat16* __restrict__ A_log,
    const __nv_bfloat16* __restrict__ D,
    const __nv_bfloat16* __restrict__ dt_bias,
    float* __restrict__ A,
    float* __restrict__ D_f32,
    float* __restrict__ dt_bias_f32,
    int num_heads)
{
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= num_heads) return;
    A[h] = -__expf(bf16_to_float(A_log[h]));
    D_f32[h] = bf16_to_float(D[h]);
    dt_bias_f32[h] = bf16_to_float(dt_bias[h]);
}

__global__ void prepare_mamba_dt_da_kernel(
    const __nv_bfloat16* __restrict__ dt_in,
    const float* __restrict__ A,
    const float* __restrict__ dt_bias,
    float* __restrict__ dt_out,
    float* __restrict__ dA_out,
    int total,
    int num_heads,
    float time_step_min)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int h = i - (i / num_heads) * num_heads;
    const float dt = fmaxf(
        softplus_f(bf16_to_float(dt_in[i]) + dt_bias[h]),
        time_step_min);
    dt_out[i] = dt;
    dA_out[i] = __expf(dt * A[h]);
}

__global__ void zamba_rmsnorm_gated_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ y,
    int hidden,
    int gate_stride,
    int group_size,
    float eps)
{
    const int row = blockIdx.x;
    const int group = blockIdx.y;
    const int tid = threadIdx.x;
    const int base = row * hidden + group * group_size;
    const long long gate_base =
        static_cast<long long>(row) * gate_stride + group * group_size;

    float local = 0.f;
    for (int i = tid; i < group_size; i += blockDim.x) {
        const float xv = bf16_to_float(x[base + i]);
        const float gv = bf16_to_float(gate[gate_base + i]);
        const float v = xv * silu_f(gv);
        local += v * v;
    }

    __shared__ float buf[256];
    buf[tid] = local;
    __syncthreads();
    for (int off = blockDim.x / 2; off > 0; off >>= 1) {
        if (tid < off) buf[tid] += buf[tid + off];
        __syncthreads();
    }
    const float inv_rms = rsqrtf(buf[0] / static_cast<float>(group_size) + eps);

    for (int i = tid; i < group_size; i += blockDim.x) {
        const float xv = bf16_to_float(x[base + i]);
        const float gv = bf16_to_float(gate[gate_base + i]);
        const float v = xv * silu_f(gv) * inv_rms;
        const int h = group * group_size + i;
        y[base + i] = __float2bfloat16(v * bf16_to_float(weight[h]));
    }
}

}  // namespace

void mamba_split_bf16(
    const void* projected,
    void* gate,
    void* conv_in,
    void* dt,
    int N,
    int projection_dim,
    int intermediate,
    int conv_dim,
    int num_heads,
    cudaStream_t stream)
{
    const int total = N * projection_dim;
    if (total <= 0) return;
    constexpr int BLOCK = 256;
    if (gate == nullptr) {
        const int conv_dt_total = N * (conv_dim + num_heads);
        const int conv_dt_grid = (conv_dt_total + BLOCK - 1) / BLOCK;
        mamba_split_conv_dt_kernel<<<conv_dt_grid, BLOCK, 0, stream>>>(
            static_cast<const __nv_bfloat16*>(projected),
            static_cast<__nv_bfloat16*>(conv_in),
            static_cast<__nv_bfloat16*>(dt),
            projection_dim, intermediate, conv_dim, num_heads,
            conv_dt_total);
        return;
    }
    const int grid = (total + BLOCK - 1) / BLOCK;
    mamba_split_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(projected),
        static_cast<__nv_bfloat16*>(gate),
        static_cast<__nv_bfloat16*>(conv_in),
        static_cast<__nv_bfloat16*>(dt),
        projection_dim, intermediate, conv_dim, num_heads, total);
}

void prepare_mamba_params_bf16(
    const void* A_log,
    const void* D,
    const void* dt_bias,
    float* A,
    float* D_f32,
    float* dt_bias_f32,
    int num_heads,
    cudaStream_t stream)
{
    if (num_heads <= 0) return;
    constexpr int BLOCK = 256;
    const int grid = (num_heads + BLOCK - 1) / BLOCK;
    prepare_mamba_params_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(A_log),
        static_cast<const __nv_bfloat16*>(D),
        static_cast<const __nv_bfloat16*>(dt_bias),
        A, D_f32, dt_bias_f32, num_heads);
}

void prepare_mamba_dt_da_bf16(
    const void* dt,
    const float* A,
    const float* dt_bias,
    float* dt_out,
    float* dA_out,
    int N,
    int num_heads,
    float time_step_min,
    cudaStream_t stream)
{
    const int total = N * num_heads;
    if (total <= 0) return;
    constexpr int BLOCK = 256;
    const int grid = (total + BLOCK - 1) / BLOCK;
    prepare_mamba_dt_da_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(dt),
        A, dt_bias, dt_out, dA_out, total, num_heads, time_step_min);
}

void rmsnorm_gated_bf16(
    const void* x,
    const void* gate,
    const void* weight,
    void* y,
    int N,
    int hidden,
    int gate_stride,
    int group_size,
    float eps,
    cudaStream_t stream)
{
    if (N <= 0 || hidden <= 0 || group_size <= 0) return;
    if (gate_stride <= 0) gate_stride = hidden;
    constexpr int BLOCK = 256;
    const int groups = hidden / group_size;
    dim3 grid(N, groups);
    zamba_rmsnorm_gated_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<const __nv_bfloat16*>(gate),
        static_cast<const __nv_bfloat16*>(weight),
        static_cast<__nv_bfloat16*>(y),
        hidden, gate_stride, group_size, eps);
}

}  // namespace pie_cuda_device::kernels
