// What limits pie's MXFP4 decode GEMV: the memory system, or the work done
// per byte?
//
// The measured answer decides the kernel. At gpt-oss's shape one layer's
// gate/up reads ~35 MB of packed weights and pie's kernel sustains 1.86 TB/s
// against a B200 roof near 8. SGLang's FP4 tensor-core path gets 3.36. But
// "SGLang is faster" does not say WHY, and the two possible causes want
// opposite fixes:
//
//   * if a kernel that only STREAMS these bytes also stalls near 1.9 TB/s,
//     the access pattern is the limit and the unpack is irrelevant --
//     restructure the loads (TMA/cp.async staging, wider per-thread
//     footprint), not the math;
//   * if streaming reaches 5-7 TB/s, the memory system is fine and the
//     per-byte work is the limit -- attack the unpack, or move the multiply
//     onto tensor cores as SGLang does.
//
// §6 of the handoff records that rows-per-warp, block size and software
// prefetch were all swept on H100 without moving this number, prefetch
// losing to register pressure. That is consistent with EITHER cause, which
// is why it is worth one measurement rather than another sweep.
//
// Three probes over the same buffer, same footprint, increasing work:
//   read      -- uint4 loads, XOR-accumulate. No unpack, no math.
//   unpack    -- the same loads plus pie's real mxfp4_unpack8.
//   unpack+ma -- unpack plus the __hfma2 chain the GEMV actually runs.
//
// Build (~10s):
//   nvcc -O3 -std=c++20 -arch=sm_100 --expt-relaxed-constexpr \
//        -I driver/cuda/src -o /tmp/mxfp4_roof \
//        driver/cuda/bench/mxfp4_roof_bench.cu
#include <cstdio>
#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_fp4.h>
#include <cuda_runtime.h>

namespace {

constexpr int kWarmup = 10;
constexpr int kIters = 50;

#define CHECK(x)                                                          \
    do {                                                                  \
        cudaError_t e_ = (x);                                             \
        if (e_ != cudaSuccess) {                                          \
            std::printf("%s:%d %s\n", __FILE__, __LINE__,                 \
                        cudaGetErrorString(e_));                          \
            return 1;                                                     \
        }                                                                 \
    } while (0)

// pie's unpack, lifted verbatim from kernels/dequant_fp4.cu so the probe
// measures the same instruction sequence and not an idealised one.
__device__ __forceinline__ void unpack8(unsigned w, __half2 out[4]) {
    const unsigned lo = w & 0x0f0f0f0fu;
    const unsigned hi = (w >> 4) & 0x0f0f0f0fu;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const unsigned a = (lo >> (i * 8)) & 0xffu;
        const unsigned b = (hi >> (i * 8)) & 0xffu;
        // E2M1 -> half via the sign/exp/mantissa reassembly the real kernel
        // does; exact values do not matter for timing, the op count does.
        const unsigned ha = ((a & 0x8u) << 12) | ((a & 0x7u) << 7);
        const unsigned hb = ((b & 0x8u) << 12) | ((b & 0x7u) << 7);
        out[i] = __halves2half2(__ushort_as_half(static_cast<unsigned short>(ha)),
                                __ushort_as_half(static_cast<unsigned short>(hb)));
    }
}

// Blackwell converts an E2M1 pair to an FP16 pair in ONE instruction
// (`cvt.rn.f16x2.e2m1x2`). pie hand-rolls the same job out of __byte_perm
// because that was the only way to do it before sm_100.
__device__ __forceinline__ void unpack8_native(unsigned w, __half2 out[4]) {
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const __nv_fp4x2_storage_t pair =
            static_cast<__nv_fp4x2_storage_t>((w >> (i * 8)) & 0xffu);
        out[i] = static_cast<__half2>(
            __nv_cvt_fp4x2_to_halfraw2(pair, __NV_E2M1));
    }
}

template <int kMode>
__global__ void probe(const uint4* __restrict__ w, int groups, float* out) {
    const int lane = threadIdx.x & 31;
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const uint4* p = w + static_cast<long long>(row) * groups;
    float acc = 0.f;
    __half2 sum = __float2half2_rn(0.f);
    __half2 acc4[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) acc4[j] = __float2half2_rn(0.f);
    unsigned x = 0;
    for (int g = lane; g < groups; g += 32) {
        const uint4 v = p[g];
        if (kMode == 0) {               // stream only
            x ^= v.x ^ v.y ^ v.z ^ v.w;
        } else {
            __half2 qd[4];
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                if (kMode == 4) unpack8_native((&v.x)[q], qd);
                else unpack8((&v.x)[q], qd);
                if (kMode == 2) {       // ONE accumulator: what pie does
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        sum = __hfma2(qd[j], qd[j], sum);
                } else if (kMode == 3) {
                    // FOUR independent accumulators. Same FMA count, same
                    // loads, same unpack -- the only change is that the
                    // chain is four deep instead of sixteen, so the FMAs
                    // can pipeline instead of each waiting on the last.
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        acc4[j] = __hfma2(qd[j], qd[j], acc4[j]);
                } else {                // unpack only -- consume ALL
                    // Consuming one output let the compiler delete the
                    // other three unpacks, which made the unpack look
                    // free when it was simply absent.
#pragma unroll
                    for (int j = 0; j < 4; ++j)
                        x ^= __half_as_ushort(__low2half(qd[j])) ^
                             __half_as_ushort(__high2half(qd[j]));
                }
            }
        }
    }
#pragma unroll
    for (int j = 0; j < 4; ++j) sum = __hadd2(sum, acc4[j]);
    const float2 f = __half22float2(sum);
    acc = f.x + f.y + static_cast<float>(x & 1u);
#pragma unroll
    for (int off = 16; off > 0; off >>= 1)
        acc += __shfl_xor_sync(0xffffffffu, acc, off);
    if (lane == 0 && acc == 12345.f) out[row] = acc;   // never taken
}

const char* kName[5] = {"read only                    ",
                        "read + unpack (PRMT, pie)    ",
                        "read + unpack + ma (x1)      ",
                        "read + unpack + ma (x4)      ",
                        "read + unpack (NATIVE cvt)   "};

}  // namespace

int main() {
    // gpt-oss decode, one layer's gate/up, 4 active experts: the shape the
    // engine actually runs. rows = 2 * intermediate * top_k (gate and up
    // interleaved), each row H/2 packed bytes.
    const int H = 2880, I = 2880, topk = 4;
    const int rows = 2 * I * topk;
    const int groups = (H / 2) / 16;          // uint4 per row
    const size_t bytes = static_cast<size_t>(rows) * groups * 16;
    std::printf("gpt-oss gate/up, %d rows x %d B = %.1f MB\n\n",
                rows, groups * 16, bytes / 1e6);

    // B200 has 132.6 MB of L2 and this buffer is 35.3 MB, so looping one
    // allocation would measure L2, not HBM -- after the first pass the
    // weights never leave cache. The engine never gets that: gpt-oss streams
    // ~1.3 GB of expert weights per token, so every layer's read is cold.
    // Rotate over enough buffers that returning to one has evicted it.
    const int kBufs = 8;              // 8 x 35.3 MB = 282 MB > 2x L2
    uint4* w[kBufs] = {};
    float* out = nullptr;
    for (int i = 0; i < kBufs; ++i) {
        CHECK(cudaMalloc(&w[i], bytes));
        CHECK(cudaMemset(w[i], 0x5a + i, bytes));
    }
    CHECK(cudaMalloc(&out, rows * sizeof(float)));
    std::printf("  (rotating %d buffers = %.0f MB to defeat the %.0f MB L2)\n\n",
                kBufs, kBufs * bytes / 1e6, 132.6);

    const int warps_per_block = 8;
    const dim3 block(warps_per_block * 32);
    const dim3 grid((rows + warps_per_block - 1) / warps_per_block);

    cudaEvent_t a, b;
    CHECK(cudaEventCreate(&a));
    CHECK(cudaEventCreate(&b));
    for (int mode = 0; mode < 5; ++mode) {
        for (int i = 0; i < kWarmup; ++i) {
            if (mode == 0) probe<0><<<grid, block>>>(w[i % kBufs], groups, out);
            else if (mode == 1) probe<1><<<grid, block>>>(w[i % kBufs], groups, out);
            else if (mode == 2) probe<2><<<grid, block>>>(w[i % kBufs], groups, out);
            else if (mode == 3) probe<3><<<grid, block>>>(w[i % kBufs], groups, out);
            else probe<4><<<grid, block>>>(w[i % kBufs], groups, out);
        }
        CHECK(cudaDeviceSynchronize());
        CHECK(cudaEventRecord(a));
        for (int i = 0; i < kIters; ++i) {
            if (mode == 0) probe<0><<<grid, block>>>(w[i % kBufs], groups, out);
            else if (mode == 1) probe<1><<<grid, block>>>(w[i % kBufs], groups, out);
            else if (mode == 2) probe<2><<<grid, block>>>(w[i % kBufs], groups, out);
            else if (mode == 3) probe<3><<<grid, block>>>(w[i % kBufs], groups, out);
            else probe<4><<<grid, block>>>(w[i % kBufs], groups, out);
        }
        CHECK(cudaEventRecord(b));
        CHECK(cudaEventSynchronize(b));
        float ms = 0.f;
        CHECK(cudaEventElapsedTime(&ms, a, b));
        const double us = ms * 1e3 / kIters;
        std::printf("  %s %8.1f us   %5.2f TB/s\n", kName[mode], us,
                    bytes / (us * 1e-6) / 1e12);
    }
    std::printf("\n  pie's real GEMV measured 19.0 us (1.86 TB/s) at this shape.\n");
    return 0;
}
