#include "kernels/gemv.hpp"

#include <cuda_bf16.h>

#include <cstdint>
#include <cstdlib>
#include <type_traits>

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
template <int kWarps, int kUnrollP = 4>
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
    constexpr int kUnroll = kUnrollP;
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

// How deep to unroll the row walk. Blackwell wants a SHALLOWER unroll than
// Hopper, which is the opposite of what the comment above predicts, so it is
// selected from the device rather than fixed.
//
// The unroll exists to keep several loads in flight; on an H100 one-at-a-time
// sustained only ~963 GB/s, and 4 was the tuned depth there. On B200 four is
// past the point of diminishing returns and costs more than it buys -- two is
// enough to cover the latency and leaves registers for occupancy. Measured
// cold, 8 rotating buffers, bf16, M=1 (driver/cuda/bench/gemv_bench.cu sweep):
//
//   shape                     unroll=4          unroll=2
//   qwen27 gate/up      34.2us 5.22TB/s   30.7us 5.80TB/s   -10.2%
//   qwen27 down         35.3us 5.06TB/s   32.2us 5.54TB/s    -8.8%
//   gemma31 down        43.3us 5.34TB/s   39.1us 5.91TB/s    -9.7%
//   gptoss lm_head     227.4us 5.09TB/s  194.5us 5.95TB/s   -14.5%
//
// This kernel is 78% of Qwen3.6-27B's decode step and 77% of gemma-4-31B's,
// so the depth is worth taking from a measurement. Hopper keeps 4: it was
// tuned there and nothing here re-measured it on that part.
inline int gemv_unroll_depth() {
    static const int depth = [] {
        // PIE_GEMV_B200_TUNING=0 reverts to the Hopper-tuned constants (this
        // predicate gates both the row-per-warp unroll and the split-K
        // warps/unroll), so a corruption can be bisected against them without
        // a rebuild.
        const char* off = std::getenv("PIE_GEMV_B200_TUNING");
        if (off != nullptr && off[0] == '0') return 4;
        int dev = 0, major = 0;
        if (cudaGetDevice(&dev) != cudaSuccess) return 4;
        if (cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor,
                                   dev) != cudaSuccess) {
            return 4;
        }
        return major >= 10 ? 2 : 4;
    }();
    return depth;
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

// MEASURED DEAD END, recorded because the reasoning looks sound and someone
// will try it again: staging the activation in shared memory before the dot
// product is SLOWER. Every block reads the same activation vector, so the
// loop below issues two global loads per iteration -- one weight, one
// activation -- and repeats the activation read for all N blocks, which looks
// like an obvious thing to hoist. It is not: those re-reads are already
// served by L2, and paying a __syncthreads plus a shared-memory round trip to
// avoid them costs more than they do. Measured cold on B200 (bf16, one token):
// gpt-oss q_proj 10.3 -> 13.7 us, o_proj 8.3 -> 11.5, gemma-4-31B kv_proj
// 12.5 -> 16.4. Results were bit-identical, so this is purely a loss.
template <int kWarps, int kUnrollP = 1>
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
    // `kUnrollP` hoists loads above the math for the same reason the
    // row-per-warp kernel does it: written flat, each lane has ONE load in
    // flight. Default 1 keeps the shipping code byte-identical.
    constexpr int kU = kUnrollP;
    const int stride = kWarps * 32;
    int i = warp * 32 + threadIdx.x;
    for (; i + stride * (kU - 1) < vectors; i += stride * kU) {
        float4 wv[kU];
        float4 xv[kU];
        #pragma unroll
        for (int u = 0; u < kU; ++u) {
            wv[u] = w4[i + stride * u];
            xv[u] = x4[i + stride * u];
        }
        #pragma unroll
        for (int u = 0; u < kU; ++u) {
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
    for (; i < vectors; i += stride) {
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

// Three projections, one launch, one grid.
//
// q/k/v share an activation and differ only in their weight and row count, and
// running them separately leaves the two small ones alone on the device. At
// gpt-oss's decode shape the tuner measures q (N=4096) at 11.2 us and 2.1 TB/s
// but k and v (N=512) at 6.0 us each for an eighth of the bytes -- 0.5 TB/s,
// because 512 rows is not enough grid to cover the memory latency whatever the
// kernel does with them. Concatenating the grids gives the small two the same
// occupancy the large one already had, for exactly the same traffic. vLLM
// arrives at the same place from the other direction, by fusing the weights at
// load and issuing one GEMM.
template <int kWarps, int kUnrollP = 1>
__global__ void gemv3_bf16_kernel(
    const __nv_bfloat16* __restrict__ w0,
    const __nv_bfloat16* __restrict__ w1,
    const __nv_bfloat16* __restrict__ w2,
    const __nv_bfloat16* __restrict__ b0,
    const __nv_bfloat16* __restrict__ b1,
    const __nv_bfloat16* __restrict__ b2,
    __nv_bfloat16* __restrict__ o0,
    __nv_bfloat16* __restrict__ o1,
    __nv_bfloat16* __restrict__ o2,
    const __nv_bfloat16* __restrict__ act,
    int n0, int n1, int n2, int K)
{
    int row = blockIdx.x;
    const __nv_bfloat16* weight;
    const __nv_bfloat16* bias;
    __nv_bfloat16* out;
    if (row < n0)            { weight = w0; bias = b0; out = o0; }
    else if (row < n0 + n1)  { row -= n0;      weight = w1; bias = b1; out = o1; }
    else                     { row -= n0 + n1; weight = w2; bias = b2; out = o2; }

    const int warp = threadIdx.y;
    const float4* w4 =
        reinterpret_cast<const float4*>(weight + (long long)row * K);
    const float4* x4 = reinterpret_cast<const float4*>(act);
    const int vectors = K / 8;

    // Same shape as gemv_splitk_bf16_kernel, and the same fix: written flat
    // each lane keeps ONE load in flight. Default kUnrollP = 1 keeps the
    // shipping code byte-identical.
    float acc = 0.f;
    constexpr int kU = kUnrollP;
    const int stride = kWarps * 32;
    int i = warp * 32 + threadIdx.x;
    for (; i + stride * (kU - 1) < vectors; i += stride * kU) {
        float4 wv[kU];
        float4 xv[kU];
        #pragma unroll
        for (int u = 0; u < kU; ++u) {
            wv[u] = w4[i + stride * u];
            xv[u] = x4[i + stride * u];
        }
        #pragma unroll
        for (int u = 0; u < kU; ++u) {
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
    for (; i < vectors; i += stride) {
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
        // Warps per block and unroll depth, measured under GRAPH REPLAY at the
        // shapes these five models actually decode through.
        //
        // Two earlier attempts got this wrong the same way: both swept EAGER,
        // where the launch floor on this box is ~4.1 us. Most of these shapes
        // run under that, so the sweep compared launch overhead rather than
        // kernels. The first shipped a blanket warps=2 and cost gemma-4-26B
        // 3.4% and Qwen3.6-27B 1.9%; the second papered over it with a size
        // threshold. Timed the way pie actually decodes -- inside a captured
        // graph -- w=8,u=1 is the WORST config on 11 of 12 shapes:
        //
        //   shape                MB   w8u1   w4u2   per-shape best
        //   qwen35 q_proj      16.8   5.45   4.00   3.79 (w2u1)
        //   qwen35 o_proj      16.8   4.32   3.45   3.45
        //   qwen35 lin qk       8.4   3.52   2.65   2.65
        //   qwen27 gdn qk      21.0   5.22   4.17   3.91 (w2u2)
        //   gptoss o_proj      23.6   6.25   4.86   4.80 (w2u1)
        //   gemma31 kv_proj    44.0  11.35   9.25   9.25
        //   gemma26 q_proj     23.1   7.34   5.48   4.85 (w2u2)
        //
        // Summed over all twelve, w=4,u=2 is 20% faster than w=8,u=1 and
        // within 3% of picking the best config per shape -- one config, no
        // threshold. Its only loss is gpt-oss's 0.2 MB router, 1.6% of 1.9 us.
        // Hopper keeps w=8,u=1, which is where it was tuned.
        if (gemv_unroll_depth() == 2) {
            constexpr int kSplitWarpsB = 4;
            gemv_splitk_bf16_kernel<kSplitWarpsB, /*kUnrollP=*/2>
                <<<dim3(static_cast<unsigned>(N)), dim3(32, kSplitWarpsB), 0,
                   stream>>>(
                    static_cast<const __nv_bfloat16*>(weight),
                    static_cast<const __nv_bfloat16*>(act),
                    static_cast<const __nv_bfloat16*>(bias),
                    static_cast<__nv_bfloat16*>(out),
                    N, K, beta);
            return true;
        }
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
    if (gemv_unroll_depth() == 2) {
        gemv_bf16_kernel<kWarps, 2>
            <<<dim3(static_cast<unsigned>(blocks)), dim3(32, kWarps), 0,
               stream>>>(
                static_cast<const __nv_bfloat16*>(weight),
                static_cast<const __nv_bfloat16*>(act),
                static_cast<const __nv_bfloat16*>(bias),
                static_cast<__nv_bfloat16*>(out),
                N, K, beta);
        return true;
    }
    gemv_bf16_kernel<kWarps, 4>
        <<<dim3(static_cast<unsigned>(blocks)), dim3(32, kWarps), 0, stream>>>(
            static_cast<const __nv_bfloat16*>(weight),
            static_cast<const __nv_bfloat16*>(act),
            static_cast<const __nv_bfloat16*>(bias),
            static_cast<__nv_bfloat16*>(out),
            N, K, beta);
    return true;
}

// Sweep entry point for the fused QKV GEMV. Shapes must come from a model.
bool launch_gemv3_bf16_tuned(
    const void* w0, const void* w1, const void* w2,
    const void* act, void* o0, void* o1, void* o2,
    int n0, int n1, int n2, int K, int warps, int unroll,
    cudaStream_t stream)
{
    if (K <= 0 || (K % 8) != 0) return false;
    const auto* a = static_cast<const __nv_bfloat16*>(act);
    auto go = [&](auto W, auto U) {
        constexpr int kW = decltype(W)::value, kU = decltype(U)::value;
        gemv3_bf16_kernel<kW, kU>
            <<<dim3(static_cast<unsigned>(n0 + n1 + n2)), dim3(32, kW), 0,
               stream>>>(
                static_cast<const __nv_bfloat16*>(w0),
                static_cast<const __nv_bfloat16*>(w1),
                static_cast<const __nv_bfloat16*>(w2),
                nullptr, nullptr, nullptr,
                static_cast<__nv_bfloat16*>(o0),
                static_cast<__nv_bfloat16*>(o1),
                static_cast<__nv_bfloat16*>(o2), a, n0, n1, n2, K);
    };
#define PIE_G3_CASE(W, U) \
    if (warps == (W) && unroll == (U)) {                                   \
        go(std::integral_constant<int, W>{},                               \
           std::integral_constant<int, U>{});                              \
        return true;                                                       \
    }
    PIE_G3_CASE(2,1) PIE_G3_CASE(4,1) PIE_G3_CASE(8,1) PIE_G3_CASE(16,1)
    PIE_G3_CASE(2,2) PIE_G3_CASE(4,2) PIE_G3_CASE(8,2) PIE_G3_CASE(4,4)
#undef PIE_G3_CASE
    return false;
}

bool launch_gemv3_bf16(
    const void* w0, const void* w1, const void* w2,
    const void* b0, const void* b1, const void* b2,
    void* o0, void* o1, void* o2,
    const void* act,
    int n0, int n1, int n2, int K,
    cudaStream_t stream)
{
    if (n0 <= 0 || n1 <= 0 || n2 <= 0 || K <= 0 || (K % 8) != 0) return false;
    if (w0 == nullptr || w1 == nullptr || w2 == nullptr || act == nullptr) {
        return false;
    }
    if (o0 == nullptr || o1 == nullptr || o2 == nullptr) return false;
    if (!aligned16(w0) || !aligned16(w1) || !aligned16(w2) ||
        !aligned16(act)) {
        return false;
    }
    // Warps per block and unroll depth. Same structure as the split-K kernel
    // -- one block per output row, warps dividing K -- and the same answer:
    // eight warps is the worst of the configurations measured. Timed under
    // GRAPH REPLAY at the QKV shapes these models decode through:
    //
    //   shape             MB    w8u1   best        gain
    //   gptoss QKV      29.5    8.59   5.66 w2u1   34%
    //   gemma26 QKV     46.1   12.61   7.18 w2u2   43%
    //   qwen35 QKV      21.0    7.22   4.59 w2u2   36%
    //   gemma31 QKV    176.2   45.02  29.42 w2u2   35%
    //
    // Two warps with a 2-deep unroll: best or within a few percent on all
    // four. Hopper keeps w=8,u=1, where it was tuned.
    if (gemv_unroll_depth() == 2) {   // same sm_100+ predicate
        constexpr int kW = 2;
        gemv3_bf16_kernel<kW, /*kUnrollP=*/2>
            <<<dim3(static_cast<unsigned>(n0 + n1 + n2)), dim3(32, kW), 0,
               stream>>>(
                static_cast<const __nv_bfloat16*>(w0),
                static_cast<const __nv_bfloat16*>(w1),
                static_cast<const __nv_bfloat16*>(w2),
                static_cast<const __nv_bfloat16*>(b0),
                static_cast<const __nv_bfloat16*>(b1),
                static_cast<const __nv_bfloat16*>(b2),
                static_cast<__nv_bfloat16*>(o0),
                static_cast<__nv_bfloat16*>(o1),
                static_cast<__nv_bfloat16*>(o2),
                static_cast<const __nv_bfloat16*>(act), n0, n1, n2, K);
        return true;
    }
    constexpr int kSplitWarps = 8;
    gemv3_bf16_kernel<kSplitWarps>
        <<<dim3(static_cast<unsigned>(n0 + n1 + n2)), dim3(32, kSplitWarps), 0,
           stream>>>(
            static_cast<const __nv_bfloat16*>(w0),
            static_cast<const __nv_bfloat16*>(w1),
            static_cast<const __nv_bfloat16*>(w2),
            static_cast<const __nv_bfloat16*>(b0),
            static_cast<const __nv_bfloat16*>(b1),
            static_cast<const __nv_bfloat16*>(b2),
            static_cast<__nv_bfloat16*>(o0),
            static_cast<__nv_bfloat16*>(o1),
            static_cast<__nv_bfloat16*>(o2),
            static_cast<const __nv_bfloat16*>(act),
            n0, n1, n2, K);
    return true;
}


// Sweep entry point for driver/cuda/bench/gemv_bench.cu. The row-per-warp
// GEMV's rows-per-block and unroll depth are compile-time constants chosen on
// an H100; this exposes the combinations so B200 can be measured rather than
// assumed. `warps`/`unroll` outside the instantiated set fall back to the
// shipping <4,4>.
bool launch_gemv_bf16_tuned(
    const void* weight, const void* act, const void* bias, void* out,
    int N, int K, int warps, int unroll, cudaStream_t stream)
{
    if (N <= 0 || K <= 0 || (K % 8) != 0) return false;
    const auto* w = static_cast<const __nv_bfloat16*>(weight);
    const auto* a = static_cast<const __nv_bfloat16*>(act);
    const auto* b = static_cast<const __nv_bfloat16*>(bias);
    auto* o = static_cast<__nv_bfloat16*>(out);
    auto go = [&](auto W, auto U) {
        constexpr int kW = decltype(W)::value, kU = decltype(U)::value;
        const long long blocks = (N + kW - 1) / kW;
        gemv_bf16_kernel<kW, kU>
            <<<dim3(static_cast<unsigned>(blocks)), dim3(32, kW), 0, stream>>>(
                w, a, b, o, N, K, 0.f);
    };
#define PIE_GEMV_CASE(W, U) \
    if (warps == (W) && unroll == (U)) {                                  \
        go(std::integral_constant<int, W>{},                              \
           std::integral_constant<int, U>{});                             \
        return true;                                                      \
    }
    PIE_GEMV_CASE(1, 4) PIE_GEMV_CASE(2, 4) PIE_GEMV_CASE(4, 4)
    PIE_GEMV_CASE(8, 4) PIE_GEMV_CASE(4, 2) PIE_GEMV_CASE(4, 8)
    PIE_GEMV_CASE(2, 8) PIE_GEMV_CASE(8, 8) PIE_GEMV_CASE(1, 8)
    PIE_GEMV_CASE(2, 2) PIE_GEMV_CASE(8, 2) PIE_GEMV_CASE(16, 4)
#undef PIE_GEMV_CASE
    return false;
}

// Sweep entry point for the split-K form's warps-per-block. `kSplitWarps = 8`
// is another constant fixed on an H100, and this kernel is 18% of
// Qwen3.6-35B-A3B's decode step (270 calls at 4.12 us). The row-per-warp
// kernel's unroll depth turned out to want a different value on Blackwell, so
// this one is worth measuring rather than inheriting too.
bool launch_gemv_splitk_tuned(
    const void* weight, const void* act, const void* bias, void* out,
    int N, int K, int warps, int unroll, cudaStream_t stream)
{
    if (N <= 0 || K <= 0 || (K % 8) != 0) return false;
    const auto* w = static_cast<const __nv_bfloat16*>(weight);
    const auto* a = static_cast<const __nv_bfloat16*>(act);
    const auto* b = static_cast<const __nv_bfloat16*>(bias);
    auto* o = static_cast<__nv_bfloat16*>(out);
    auto go = [&](auto W, auto U) {
        constexpr int kW = decltype(W)::value, kU = decltype(U)::value;
        gemv_splitk_bf16_kernel<kW, kU>
            <<<dim3(static_cast<unsigned>(N)), dim3(32, kW), 0, stream>>>(
                w, a, b, o, N, K, 0.f);
    };
#define PIE_SPLITK_CASE(W, U) \
    if (warps == (W) && unroll == (U)) {                                   \
        go(std::integral_constant<int, W>{},                               \
           std::integral_constant<int, U>{});                              \
        return true;                                                       \
    }
    PIE_SPLITK_CASE(2,1) PIE_SPLITK_CASE(4,1) PIE_SPLITK_CASE(8,1)
    PIE_SPLITK_CASE(16,1) PIE_SPLITK_CASE(2,2) PIE_SPLITK_CASE(4,2)
    PIE_SPLITK_CASE(8,2) PIE_SPLITK_CASE(16,2) PIE_SPLITK_CASE(4,4)
    PIE_SPLITK_CASE(8,4) PIE_SPLITK_CASE(2,4)
#undef PIE_SPLITK_CASE
    return false;
}

}  // namespace pie_cuda_driver::kernels