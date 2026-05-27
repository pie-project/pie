#include "kernels/dsv4_hc.hpp"

#include <cfloat>
#include <cmath>
#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels {

namespace {

constexpr int BLOCK = 256;
constexpr int MAX_HC_MULT = 8;

__global__ void hc_pre_postprocess_kernel(
    const float* __restrict__ mixes,     // [N, mix_hc]
    const float* __restrict__ scale,     // [3]
    const float* __restrict__ base,      // [mix_hc]
    const __nv_bfloat16* __restrict__ residual, // [N, M, H]
    float* __restrict__ post_mix,        // [N, M]
    float* __restrict__ comb_mix,        // [N, M, M]
    __nv_bfloat16* __restrict__ layer_input, // [N, H]
    int M,    // hc_mult
    int H,
    float hc_eps,
    float hc_post_alpha,
    int sinkhorn_iters)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;

    const int mix_hc = M * 2 + M * M;
    const float* row = mixes + static_cast<long long>(n) * mix_hc;

    __shared__ float pre[MAX_HC_MULT];
    __shared__ float post[MAX_HC_MULT];
    __shared__ float comb[MAX_HC_MULT * MAX_HC_MULT];

    if (tid < M) {
        // Pre-mix: sigmoid + eps
        const float logit = row[tid] * scale[0] + base[tid];
        pre[tid] = 1.f / (1.f + expf(-logit)) + hc_eps;
    }
    if (tid < M) {
        // Post-mix: sigmoid * alpha
        const float logit = row[M + tid] * scale[1] + base[M + tid];
        post[tid] = 1.f / (1.f + expf(-logit)) * hc_post_alpha;
        post_mix[static_cast<long long>(n) * M + tid] = post[tid];
    }
    __syncthreads();

    // Comb-mix: softmax + sinkhorn
    if (tid < M * M) {
        const int i = tid / M;
        const int j = tid % M;
        const float logit = row[2 * M + tid] * scale[2] + base[2 * M + tid];
        comb[tid] = logit;
    }
    __syncthreads();

    // Softmax per row (rows = M, cols = M)
    if (tid < M) {
        float max_v = -FLT_MAX;
        for (int j = 0; j < M; ++j)
            max_v = fmaxf(max_v, comb[tid * M + j]);
        float sum = 0.f;
        for (int j = 0; j < M; ++j) {
            comb[tid * M + j] = expf(comb[tid * M + j] - max_v);
            sum += comb[tid * M + j];
        }
        for (int j = 0; j < M; ++j)
            comb[tid * M + j] = comb[tid * M + j] / sum + hc_eps;
    }
    __syncthreads();

    // Sinkhorn iterations
    for (int iter = 0; iter < sinkhorn_iters - 1; ++iter) {
        // Normalize columns
        if (tid < M) {
            float col_sum = hc_eps;
            for (int i = 0; i < M; ++i) col_sum += comb[i * M + tid];
            for (int i = 0; i < M; ++i)
                comb[i * M + tid] = comb[i * M + tid] / col_sum;
        }
        __syncthreads();
        // Normalize rows
        if (tid < M) {
            float row_sum = hc_eps;
            for (int j = 0; j < M; ++j) row_sum += comb[tid * M + j];
            for (int j = 0; j < M; ++j)
                comb[tid * M + j] = comb[tid * M + j] / row_sum;
        }
        __syncthreads();
    }

    // Write comb_mix
    if (tid < M * M) {
        comb_mix[static_cast<long long>(n) * M * M + tid] = comb[tid];
    }
    __syncthreads();

    // Compute layer_input = sum_i(pre_i * residual[n, i, :])
    const __nv_bfloat16* res_n =
        residual + static_cast<long long>(n) * M * H;
    __nv_bfloat16* out = layer_input + static_cast<long long>(n) * H;

    for (int h = tid; h < H; h += blockDim.x) {
        float acc = 0.f;
        for (int i = 0; i < M; ++i) {
            acc += pre[i] * __bfloat162float(res_n[i * H + h]);
        }
        out[h] = __float2bfloat16(acc);
    }
}

__global__ void hc_post_kernel(
    const __nv_bfloat16* __restrict__ x,        // [N, H]
    const __nv_bfloat16* __restrict__ residual,  // [N, M, H]
    const float* __restrict__ post_mix,          // [N, M]
    const float* __restrict__ comb_mix,          // [N, M, M]
    __nv_bfloat16* __restrict__ out,             // [N, M, H]
    int M,
    int H)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = gridDim.x * blockDim.x;

    // Each thread handles one (n, j, h) element
    const int n_j_h = idx;
    if (n_j_h >= total) return;

    const int h = n_j_h % H;
    const int j = (n_j_h / H) % M;
    const int n = n_j_h / (H * M);

    const float* comb_n = comb_mix + static_cast<long long>(n) * M * M;
    const float post_j = post_mix[static_cast<long long>(n) * M + j];
    const float x_h = __bfloat162float(x[static_cast<long long>(n) * H + h]);

    float acc = post_j * x_h;
    const __nv_bfloat16* res_n = residual + static_cast<long long>(n) * M * H;
    for (int i = 0; i < M; ++i) {
        acc += comb_n[j * M + i] * __bfloat162float(res_n[i * H + h]);
    }
    out[static_cast<long long>(n) * M * H + j * H + h] = __float2bfloat16(acc);
}

__global__ void hc_head_postprocess_kernel(
    const float* __restrict__ mixes,     // [N, M] after GEMM
    const float* __restrict__ scale,     // [1]
    const float* __restrict__ base,      // [M]
    const __nv_bfloat16* __restrict__ residual, // [N, M, H]
    __nv_bfloat16* __restrict__ out,     // [N, H]
    int M,
    int H)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ float gates[MAX_HC_MULT];

    if (tid < M) {
        const float logit = mixes[static_cast<long long>(n) * M + tid] * scale[0] + base[tid];
        gates[tid] = 1.f / (1.f + expf(-logit));
    }
    __syncthreads();

    // Normalize gates
    if (tid == 0) {
        float sum = 0.f;
        for (int i = 0; i < M; ++i) sum += gates[i];
        if (sum > 0.f) {
            for (int i = 0; i < M; ++i) gates[i] /= sum;
        }
    }
    __syncthreads();

    const __nv_bfloat16* res_n = residual + static_cast<long long>(n) * M * H;
    __nv_bfloat16* out_n = out + static_cast<long long>(n) * H;

    for (int h = tid; h < H; h += blockDim.x) {
        float acc = 0.f;
        for (int i = 0; i < M; ++i) {
            acc += gates[i] * __bfloat162float(res_n[i * H + h]);
        }
        out_n[h] = __float2bfloat16(acc);
    }
}

__global__ void hc_expand_kernel(
    const __nv_bfloat16* __restrict__ input,  // [N, H]
    __nv_bfloat16* __restrict__ output,       // [N, M, H]
    int M,
    int H)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int n = idx / H;
    const int h = idx % H;
    if (h >= H) return;

    const __nv_bfloat16 val = input[static_cast<long long>(n) * H + h];
    for (int m = 0; m < M; ++m) {
        output[static_cast<long long>(n) * M * H + m * H + h] = val;
    }
}

__global__ void hc_rmsnorm_to_f32_kernel(
    const __nv_bfloat16* __restrict__ input,  // [N, dim]
    float* __restrict__ output,               // [N, dim]
    int dim,
    float eps)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const __nv_bfloat16* row = input + static_cast<long long>(n) * dim;
    float* out = output + static_cast<long long>(n) * dim;

    // Two-pass: first compute sum-of-squares, then scale
    __shared__ float shared_sum;
    if (tid == 0) shared_sum = 0.f;
    __syncthreads();
    float local_sum = 0.f;
    for (int d = tid; d < dim; d += blockDim.x) {
        float v = __bfloat162float(row[d]);
        local_sum += v * v;
    }
    // Warp reduce
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);
    if (tid % 32 == 0)
        atomicAdd(&shared_sum, local_sum);
    __syncthreads();

    if (tid == 0) shared_sum = rsqrtf(shared_sum / dim + eps);
    __syncthreads();

    const float scale = shared_sum;
    for (int d = tid; d < dim; d += blockDim.x) {
        out[d] = __bfloat162float(row[d]) * scale;
    }
}

}  // namespace

void launch_hc_pre_postprocess_bf16(
    const float* mixes,
    const float* scale,
    const float* base,
    const void* residual,
    float* post_mix,
    float* comb_mix,
    void* layer_input,
    int N,
    int hc_mult,
    int hidden_size,
    float hc_eps,
    float hc_post_alpha,
    int sinkhorn_iters,
    cudaStream_t stream)
{
    if (N <= 0) return;
    hc_pre_postprocess_kernel<<<N, BLOCK, 0, stream>>>(
        mixes, scale, base,
        static_cast<const __nv_bfloat16*>(residual),
        post_mix, comb_mix,
        static_cast<__nv_bfloat16*>(layer_input),
        hc_mult, hidden_size, hc_eps, hc_post_alpha, sinkhorn_iters);
}

void launch_hc_post_bf16(
    const void* x,
    const void* residual,
    const float* post_mix,
    const float* comb_mix,
    void* out_residual,
    int N,
    int hc_mult,
    int hidden_size,
    cudaStream_t stream)
{
    const long long total = static_cast<long long>(N) * hc_mult * hidden_size;
    if (total <= 0) return;
    const int grid = static_cast<int>((total + BLOCK - 1) / BLOCK);
    hc_post_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(x),
        static_cast<const __nv_bfloat16*>(residual),
        post_mix, comb_mix,
        static_cast<__nv_bfloat16*>(out_residual),
        hc_mult, hidden_size);
}

void launch_hc_head_postprocess_bf16(
    const float* mixes,
    const float* scale,
    const float* base,
    const void* residual,
    void* out,
    int N,
    int hc_mult,
    int hidden_size,
    cudaStream_t stream)
{
    if (N <= 0) return;
    hc_head_postprocess_kernel<<<N, BLOCK, 0, stream>>>(
        mixes, scale, base,
        static_cast<const __nv_bfloat16*>(residual),
        static_cast<__nv_bfloat16*>(out),
        hc_mult, hidden_size);
}

void launch_hc_expand_bf16(
    const void* input,
    void* output,
    int N,
    int hc_mult,
    int hidden_size,
    cudaStream_t stream)
{
    const long long total = static_cast<long long>(N) * hidden_size;
    if (total <= 0) return;
    const int grid = static_cast<int>((total + BLOCK - 1) / BLOCK);
    hc_expand_kernel<<<grid, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        static_cast<__nv_bfloat16*>(output),
        hc_mult, hidden_size);
}

void launch_hc_rmsnorm_to_f32(
    const void* input,
    float* output,
    int N,
    int dim,
    float eps,
    cudaStream_t stream)
{
    if (N <= 0) return;
    // Initialize shared sum to 0
    hc_rmsnorm_to_f32_kernel<<<N, BLOCK, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(input),
        output, dim, eps);
}

}  // namespace pie_cuda_driver::kernels
