#include "kernels/gemv.hpp"

#include <cuda_bf16.h>

#include <cstdint>
#include <cstdlib>

namespace pie_cuda_driver::kernels {

namespace {

// One warp per output row. Each lane walks the row in float4 strides (8
// bf16 = 16 B, so a warp step covers 512 B — four full cache lines) and
// accumulates in fp32; a single shuffle tree finishes the dot product.
//
// The walk is unrolled by `kUnroll` and the loads are hoisted above the math
// that consumes them. That is the whole reason for the unroll: written the
// obvious way each lane has exactly ONE load in flight, because the FMA on
// `w4[i]` is the next instruction after it, and a warp that is waiting on a
// single HBM round trip cannot cover the latency no matter how many warps the
// SM holds. Measured on an H100 at gpt-oss's o_proj shape (N=2880, K=4096,
// 23.6 MB per layer) the one-at-a-time version sustained about 963 GB/s.
template <int kWarps>
__global__ void gemv_bf16_kernel(
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ act,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out,
    int N, int K, float beta)
{
    const int row = blockIdx.x * kWarps + threadIdx.y;
    if (row >= N) return;
    const float4* w4 =
        reinterpret_cast<const float4*>(weight + (long long)row * K);
    const float4* x4 = reinterpret_cast<const float4*>(act);
    const int vectors = K / 8;
    constexpr int kUnroll = 4;
    float acc = 0.f;
    int i = threadIdx.x;
    for (; i + 32 * (kUnroll - 1) < vectors; i += 32 * kUnroll) {
        float4 wv[kUnroll];
        float4 xv[kUnroll];
        #pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            wv[u] = w4[i + 32 * u];
            xv[u] = x4[i + 32 * u];
        }
        #pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            const __nv_bfloat16* wb =
                reinterpret_cast<const __nv_bfloat16*>(&wv[u]);
            const __nv_bfloat16* xb =
                reinterpret_cast<const __nv_bfloat16*>(&xv[u]);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc += __bfloat162float(wb[j]) * __bfloat162float(xb[j]);
            }
        }
    }
    for (; i < vectors; i += 32) {
        float4 wv = w4[i];
        float4 xv = x4[i];
        const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv);
        const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(&xv);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc += __bfloat162float(wb[j]) * __bfloat162float(xb[j]);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (threadIdx.x == 0) {
        if (beta != 0.f) acc += beta * __bfloat162float(out[row]);
        // Round to bf16 *before* adding the bias, then round again. That is
        // a redundant-looking double rounding, and it is deliberate: it is
        // exactly what the separate `add_bias_bf16_kernel` did when it read
        // this kernel's bf16 output back. Folding the two launches into one
        // is a launch-count optimization, not an arithmetic change, so it
        // has to stay bit-identical or it stops being free to validate.
        __nv_bfloat16 v = __float2bfloat16(acc);
        if (bias != nullptr) {
            v = __float2bfloat16(__bfloat162float(v) +
                                 __bfloat162float(bias[row]));
        }
        out[row] = v;
    }
}

// One BLOCK per output row, its warps splitting K between them and reducing
// through shared memory.
//
// The kernel above gives one warp to each row, so the grid is N/4 blocks and
// the machine is only busy if N is large. gpt-oss decodes through three shapes
// where it is not: k_proj and v_proj are N=512, which is 128 blocks on 132 SMs,
// and the MoE router is N=32, which is EIGHT. vLLM runs the same projections
// through cuBLAS's `splitK` nvjet kernels at 2.3-2.5 TB/s for exactly this
// reason. Splitting inside the block keeps it to one launch and needs no
// scratch, which the reduce-across-blocks form would.
template <int kWarps>
__global__ void gemv_splitk_bf16_kernel(
    const __nv_bfloat16* __restrict__ weight,
    const __nv_bfloat16* __restrict__ act,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out,
    int N, int K, float beta)
{
    const int row = blockIdx.x;
    if (row >= N) return;
    const int warp = threadIdx.y;
    const float4* w4 =
        reinterpret_cast<const float4*>(weight + (long long)row * K);
    const float4* x4 = reinterpret_cast<const float4*>(act);
    const int vectors = K / 8;

    float acc = 0.f;
    // Warp `warp` walks a strided share of the row, so the whole block still
    // reads it in the same coalesced order the one-warp kernel does.
    for (int i = warp * 32 + threadIdx.x; i < vectors; i += kWarps * 32) {
        float4 wv = w4[i];
        float4 xv = x4[i];
        const __nv_bfloat16* wb = reinterpret_cast<const __nv_bfloat16*>(&wv);
        const __nv_bfloat16* xb = reinterpret_cast<const __nv_bfloat16*>(&xv);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc += __bfloat162float(wb[j]) * __bfloat162float(xb[j]);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }

    __shared__ float partial[kWarps];
    if (threadIdx.x == 0) partial[warp] = acc;
    __syncthreads();
    if (warp != 0 || threadIdx.x != 0) return;
    float total = 0.f;
    #pragma unroll
    for (int w = 0; w < kWarps; ++w) total += partial[w];
    if (beta != 0.f) total += beta * __bfloat162float(out[row]);
    // Same double rounding as the kernel above, for the same reason: it is
    // what the separate bias kernel used to do, so the fold stays bit-exact.
    __nv_bfloat16 v = __float2bfloat16(total);
    if (bias != nullptr) {
        v = __float2bfloat16(__bfloat162float(v) + __bfloat162float(bias[row]));
    }
    out[row] = v;
}

bool aligned16(const void* p) {
    return (reinterpret_cast<std::uintptr_t>(p) & 15u) == 0;
}

}  // namespace

bool launch_gemv_bf16(
    const void* weight,
    const void* act,
    const void* bias,
    void*       out,
    int N, int K,
    cudaStream_t stream,
    float beta)
{
    // The float4 loads need each row to start 16-byte aligned: that holds
    // iff the base is aligned and the row stride is a multiple of 8 bf16.
    if (N <= 0 || K <= 0 || (K % 8) != 0) return false;
    if (weight == nullptr || act == nullptr || out == nullptr) return false;
    if (!aligned16(weight) || !aligned16(act)) return false;
    constexpr int kWarps = 4;
    // Below this the row-per-warp grid cannot fill the device, and splitting K
    // inside the block is strictly more parallel for the same traffic. 2048
    // rows is 512 blocks, about four per SM, which is where the row-per-warp
    // form stops being the constraint.
    static const int kSplitKMaxRows = [] {
        const char* v = std::getenv("PIE_GEMV_SPLITK_MAX_ROWS");
        const int n = (v != nullptr) ? std::atoi(v) : 4096;
        return (n >= 0) ? n : 4096;
    }();
    if (N <= kSplitKMaxRows) {
        constexpr int kSplitWarps = 8;
        gemv_splitk_bf16_kernel<kSplitWarps>
            <<<dim3(static_cast<unsigned>(N)), dim3(32, kSplitWarps), 0,
               stream>>>(
                static_cast<const __nv_bfloat16*>(weight),
                static_cast<const __nv_bfloat16*>(act),
                static_cast<const __nv_bfloat16*>(bias),
                static_cast<__nv_bfloat16*>(out),
                N, K, beta);
        return true;
    }
    const long long blocks = (N + kWarps - 1) / kWarps;
    if (blocks > 2147483647LL) return false;
    // Everything below is unconditional, so the caller never has to
    // reason about a half-enqueued launch. In particular this must not
    // poll `cudaGetLastError`: that would consume an unrelated pending
    // error the driver's own checks are waiting to report.
    gemv_bf16_kernel<kWarps>
        <<<dim3(static_cast<unsigned>(blocks)), dim3(32, kWarps), 0, stream>>>(
            static_cast<const __nv_bfloat16*>(weight),
            static_cast<const __nv_bfloat16*>(act),
            static_cast<const __nv_bfloat16*>(bias),
            static_cast<__nv_bfloat16*>(out),
            N, K, beta);
    return true;
}

}  // namespace pie_cuda_driver::kernels
