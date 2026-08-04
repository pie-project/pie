#include "kernels/attn_res.hpp"

#include <cstdint>
#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels {

namespace {

constexpr int kMaxBlocks = 32;  // K3 opens ceil(93 / 12) = 8; 32 is slack.
constexpr int kThreads = 256;

__device__ __forceinline__ float block_reduce_sum(float x, float* scratch) {
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        x += __shfl_down_sync(0xffffffffu, x, offset);
    }
    const int lane = threadIdx.x & (warpSize - 1);
    const int warp = threadIdx.x / warpSize;
    if (lane == 0) scratch[warp] = x;
    __syncthreads();
    const int warps = blockDim.x / warpSize;
    float total = 0.f;
    if (threadIdx.x == 0) {
        for (int w = 0; w < warps; ++w) total += scratch[w];
        scratch[0] = total;
    }
    __syncthreads();
    total = scratch[0];
    __syncthreads();
    return total;
}

// One block per token. `rows + 1` passes over H for the scores, one more for
// the blend.
__global__ void attn_res_blend_kernel(
    const __nv_bfloat16* __restrict__ prefix,
    const __nv_bfloat16* __restrict__ blocks,
    const __nv_bfloat16* __restrict__ norm_weight,
    const __nv_bfloat16* __restrict__ proj_weight,
    __nv_bfloat16* __restrict__ out,
    int T, int B, int H, int block_rows, float eps)
{
    const int t = blockIdx.x;
    if (t >= T) return;

    __shared__ float scratch[kThreads / 32];
    __shared__ float prob_s[kMaxBlocks + 1];

    const long long token_off = (long long)t * H;
    const int rows = B + 1;

    // `row(j)` is block j for j < B, and the running prefix for j == B.
    auto row_ptr = [&](int j) -> const __nv_bfloat16* {
        return (j < B) ? blocks + ((long long)j * block_rows + t) * H
                       : prefix + token_off;
    };

    for (int j = 0; j < rows; ++j) {
        const __nv_bfloat16* v = row_ptr(j);
        float ss = 0.f;
        for (int h = threadIdx.x; h < H; h += blockDim.x) {
            const float x = __bfloat162float(v[h]);
            ss += x * x;
        }
        ss = block_reduce_sum(ss, scratch);
        const float scale = rsqrtf(ss / static_cast<float>(H) + eps);

        float dot = 0.f;
        for (int h = threadIdx.x; h < H; h += blockDim.x) {
            dot += __bfloat162float(v[h]) * scale *
                   __bfloat162float(norm_weight[h]) *
                   __bfloat162float(proj_weight[h]);
        }
        dot = block_reduce_sum(dot, scratch);
        if (threadIdx.x == 0) {
            prob_s[j] = dot;
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        float m = prob_s[0];
        for (int j = 1; j < rows; ++j) m = fmaxf(m, prob_s[j]);
        float sum = 0.f;
        for (int j = 0; j < rows; ++j) {
            prob_s[j] = __expf(prob_s[j] - m);
            sum += prob_s[j];
        }
        const float inv = 1.f / sum;
        for (int j = 0; j < rows; ++j) prob_s[j] *= inv;
    }
    __syncthreads();

    for (int h = threadIdx.x; h < H; h += blockDim.x) {
        float acc = 0.f;
        for (int j = 0; j < rows; ++j) {
            acc += prob_s[j] * __bfloat162float(row_ptr(j)[h]);
        }
        out[token_off + h] = __float2bfloat16(acc);
    }
}

}  // namespace

void launch_attn_res_blend_bf16(
    const void* prefix, const void* blocks, const void* norm_weight,
    const void* proj_weight, void* out, int T, int B, int H, int block_rows,
    float eps, cudaStream_t stream)
{
    if (T <= 0 || H <= 0) return;
    attn_res_blend_kernel<<<static_cast<unsigned>(T), kThreads, 0, stream>>>(
        static_cast<const __nv_bfloat16*>(prefix),
        static_cast<const __nv_bfloat16*>(blocks),
        static_cast<const __nv_bfloat16*>(norm_weight),
        static_cast<const __nv_bfloat16*>(proj_weight),
        static_cast<__nv_bfloat16*>(out), T, B, H,
        block_rows > 0 ? block_rows : T, eps);
}

}  // namespace pie_cuda_driver::kernels
