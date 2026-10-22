#include "linear/codec.cuh"
#include "prim/device.cuh"
#include "prim/mma_sm80.cuh"
#include <cuda_bf16.h>
#include <cuda_pipeline.h>

namespace pie::linear {

// A warp's dot of weight row `row` (K wide) with the activation `act`,
// folded to lane 0.
template <int kUnroll>
__device__ __forceinline__ float gemv_bf16_row(
    const bf16* __restrict__ weight, const bf16* __restrict__ act, int row, int K)
{
    const float4* w4 =
        reinterpret_cast<const float4*>(weight + (long long)row * K);
    const float4* x4 = reinterpret_cast<const float4*>(act);
    const int vectors = K / 8;
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
            const bf16* wb = reinterpret_cast<const bf16*>(&wv[u]);
            const bf16* xb = reinterpret_cast<const bf16*>(&xv[u]);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc += bf16_to_f32(wb[j]) * bf16_to_f32(xb[j]);
            }
        }
    }
    for (; i < vectors; i += 32) {
        float4 wv = w4[i];
        float4 xv = x4[i];
        const bf16* wb = reinterpret_cast<const bf16*>(&wv);
        const bf16* xb = reinterpret_cast<const bf16*>(&xv);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc += bf16_to_f32(wb[j]) * bf16_to_f32(xb[j]);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    return acc;
}

template <int kWarps, int kUnrollP = 4>
__global__ void gemv_bf16_kernel(
    const bf16* __restrict__ weight,
    const bf16* __restrict__ act,
    const bf16* __restrict__ bias,
    bf16* __restrict__ out,
    int N, int K, float beta)
{
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    const int row = blockIdx.x * kWarps + threadIdx.y;
    if (row >= N) return;
    float acc = gemv_bf16_row<kUnrollP>(weight, act, row, K);
    if (threadIdx.x == 0) {
        if (beta != 0.f) acc += beta * bf16_to_f32(out[row]);

        bf16 v = f32_to_bf16(acc);
        if (bias != nullptr) {
            v = f32_to_bf16(bf16_to_f32(v) + bf16_to_f32(bias[row]));
        }
        out[row] = v;
    }
}

// `gemv_bf16_kernel` over a grouped plane at one row: weight row `row` is
// group `row / group_n`'s, against that group's K-wide slice of the row.
template <int kWarps>
__global__ void gemv_bf16_grouped_kernel(
    const bf16* __restrict__ weight,
    const bf16* __restrict__ act,
    bf16* __restrict__ out,
    int N, int K, int group_n)
{
    const int row = blockIdx.x * kWarps + threadIdx.y;
    if (row >= N) return;
    const float acc = gemv_bf16_row<4>(weight, act + (long long)(row / group_n) * K, row, K);
    if (threadIdx.x == 0) out[row] = f32_to_bf16(acc);
}

// `out[row][n] = act[row] . weight[n]` for a product a few columns wide
// (`N` <= 8): a block a row, its threads striding the contraction eight
// elements at a time with every column's sum in registers, folded across
// the block. The bias, if any, lands as the gemv lands it.
template <int N>
__global__ void __launch_bounds__(256) gemm_narrow_bf16_kernel(
    const bf16* __restrict__ weight,
    const bf16* __restrict__ act,
    const bf16* __restrict__ bias,
    bf16* __restrict__ out,
    int k)
{
    constexpr int kThreads = 256;
    const int row = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const bf16* x = act + static_cast<long long>(row) * k;
    float acc[N];
#pragma unroll
    for (int n = 0; n < N; ++n) acc[n] = 0.f;
    for (int i = tid * 8; i < k; i += kThreads * 8) {
        const uint4 xv = *reinterpret_cast<const uint4*>(x + i);
        const bf16* x8 = reinterpret_cast<const bf16*>(&xv);
        float xf[8];
#pragma unroll
        for (int e = 0; e < 8; ++e) xf[e] = bf16_to_f32(x8[e]);
#pragma unroll
        for (int n = 0; n < N; ++n) {
            const uint4 wv = *reinterpret_cast<const uint4*>(weight + static_cast<long long>(n) * k + i);
            const bf16* w8 = reinterpret_cast<const bf16*>(&wv);
#pragma unroll
            for (int e = 0; e < 8; ++e) acc[n] += xf[e] * bf16_to_f32(w8[e]);
        }
    }
    __shared__ float part[kThreads / 32][N];
#pragma unroll
    for (int n = 0; n < N; ++n) {
        float v = acc[n];
#pragma unroll
        for (int sh = 16; sh > 0; sh >>= 1) v += __shfl_xor_sync(0xffffffffu, v, sh);
        if ((tid & 31) == 0) part[tid >> 5][n] = v;
    }
    __syncthreads();
    if (tid < N) {
        float v = 0.f;
#pragma unroll
        for (int w = 0; w < kThreads / 32; ++w) v += part[w][tid];
        bf16 o = f32_to_bf16(v);
        if (bias != nullptr) o = f32_to_bf16(bf16_to_f32(o) + bf16_to_f32(bias[tid]));
        out[static_cast<long long>(row) * N + tid] = o;
    }
}

template <int kWarps, int kUnrollP = 1>
__global__ void gemv_splitk_bf16_kernel(
    const bf16* __restrict__ weight,
    const bf16* __restrict__ act,
    const bf16* __restrict__ bias,
    bf16* __restrict__ out,
    int N, int K, float beta)
{
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    const int row = blockIdx.x;
    if (row >= N) return;
    const int warp = threadIdx.y;
    const float4* w4 =
        reinterpret_cast<const float4*>(weight + (long long)row * K);
    const float4* x4 = reinterpret_cast<const float4*>(act);
    const int vectors = K / 8;

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
            const bf16* wb = reinterpret_cast<const bf16*>(&wv[u]);
            const bf16* xb = reinterpret_cast<const bf16*>(&xv[u]);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc += bf16_to_f32(wb[j]) * bf16_to_f32(xb[j]);
            }
        }
    }
    for (; i < vectors; i += stride) {
        float4 wv = w4[i];
        float4 xv = x4[i];
        const bf16* wb = reinterpret_cast<const bf16*>(&wv);
        const bf16* xb = reinterpret_cast<const bf16*>(&xv);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            acc += bf16_to_f32(wb[j]) * bf16_to_f32(xb[j]);
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
    if (beta != 0.f) total += beta * bf16_to_f32(out[row]);

    bf16 v = f32_to_bf16(total);
    if (bias != nullptr) {
        v = f32_to_bf16(bf16_to_f32(v) + bf16_to_f32(bias[row]));
    }
    out[row] = v;
}

// `out[r][n] = act[r] . weight[n]` for a few rows (`kM` <= 8, a decode's):
// each of a block's eight warps streams four weight rows' slice of K, a
// weight vector of each against every activation row (read through L1, so
// each activation load serves four weight loads), `kSplit` warps the rows'
// slices; each warp's sums fold across its lanes, and a row's slices add in
// slice order, so a launch lands the same bits each time. One launch, every
// weight byte read once: a split GEMM's reduce launch saved. Over a cluster
// of `kCl` blocks (a few columns, too few blocks to keep the loads in
// flight), each block takes a slice of K and the first adds the blocks' sums
// in rank order.
template <int kM, int kSplit, int kCl = 1>
__global__ void __launch_bounds__(256) gemm_rows_bf16_kernel(
    const bf16* __restrict__ weight,
    const bf16* __restrict__ act,
    const bf16* __restrict__ bias,
    bf16* __restrict__ out,
    int N, int K)
{
    static_assert(kM >= 1 && kM <= 8, "a few rows");
    // A programmatic dependent (`kernel.pdl`) reads nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    static_assert(8 % kSplit == 0, "a block's warps split evenly over its rows");
    constexpr int kR = 2;                     // weight rows a warp
    constexpr int kRows = 8 / kSplit * kR;    // weight rows a block
    __shared__ float red[8][kR][kM];
    __shared__ float sum[kRows * kM];
    const int warp = static_cast<int>(threadIdx.x) / 32;
    const int lane = static_cast<int>(threadIdx.x) % 32;
    const int block = static_cast<int>(blockIdx.x) / kCl;
    const int rank = static_cast<int>(blockIdx.x) % kCl;
    const int row0 = block * kRows + warp / kSplit * kR;
    const int part = rank * kSplit + warp % kSplit;
    const int vectors = K / 8;
    const int per = vectors / (kSplit * kCl);
    float acc[kR][kM];
#pragma unroll
    for (int q = 0; q < kR; ++q) {
#pragma unroll
        for (int r = 0; r < kM; ++r) acc[q][r] = 0.f;
    }
    const uint4* x4 = reinterpret_cast<const uint4*>(act) + part * per;
    const uint4* w4[kR];
#pragma unroll
    for (int q = 0; q < kR; ++q) {
        // A row past N reads row N - 1 again and lands nothing.
        const int row = min(row0 + q, N - 1);
        w4[q] = reinterpret_cast<const uint4*>(weight + static_cast<long long>(row) * K) + part * per;
    }
#pragma unroll 4
    for (int i = lane; i < per; i += 32) {
        uint4 wv[kR];
#pragma unroll
        for (int q = 0; q < kR; ++q) wv[q] = __ldcs(w4[q] + i);
#pragma unroll
        for (int r = 0; r < kM; ++r) {
            const uint4 xv = __ldg(x4 + static_cast<long long>(r) * vectors + i);
            const unsigned xw[4] = {xv.x, xv.y, xv.z, xv.w};
#pragma unroll
            for (int q = 0; q < kR; ++q) {
                const unsigned ww[4] = {wv[q].x, wv[q].y, wv[q].z, wv[q].w};
#pragma unroll
                for (int j = 0; j < 4; ++j) {
                    acc[q][r] = fmaf(__uint_as_float(ww[j] << 16), __uint_as_float(xw[j] << 16),
                                     acc[q][r]);
                    acc[q][r] = fmaf(__uint_as_float(ww[j] & 0xffff0000u),
                                     __uint_as_float(xw[j] & 0xffff0000u), acc[q][r]);
                }
            }
        }
    }
#pragma unroll
    for (int q = 0; q < kR; ++q) {
#pragma unroll
        for (int r = 0; r < kM; ++r) {
#pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                acc[q][r] += __shfl_xor_sync(0xffffffffu, acc[q][r], off);
            }
        }
    }
    if (lane == 0) {
#pragma unroll
        for (int q = 0; q < kR; ++q) {
#pragma unroll
            for (int r = 0; r < kM; ++r) red[warp][q][r] = acc[q][r];
        }
    }
    __syncthreads();
    // Thread t sums block row t / kM's column for activation row t % kM.
    for (int t = static_cast<int>(threadIdx.x); t < kRows * kM; t += 256) {
        const int local = t / kM;
        const int r = t % kM;
        const int group = local / kR;   // the warp group holding the row
        const int q = local % kR;
        float s = red[group * kSplit][q][r];
#pragma unroll
        for (int p = 1; p < kSplit; ++p) s += red[group * kSplit + p][q][r];
        sum[t] = s;
    }
    if constexpr (kCl > 1) {
        asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    } else {
        __syncthreads();
    }
    if (rank == 0) {
        for (int t = static_cast<int>(threadIdx.x); t < kRows * kM; t += 256) {
            const int n = block * kRows + t / kM;
            float s = sum[t];
#pragma unroll
            for (int c = 1; c < kCl; ++c) {
                const unsigned local = static_cast<unsigned>(__cvta_generic_to_shared(sum + t));
                unsigned remote;
                float v;
                asm("mapa.shared::cluster.u32 %0, %1, %2;" : "=r"(remote) : "r"(local), "r"(c));
                asm("ld.shared::cluster.f32 %0, [%1];" : "=f"(v) : "r"(remote));
                s += v;
            }
            if (n < N) {
                bf16 v = f32_to_bf16(s);
                if (bias != nullptr) {
                    v = f32_to_bf16(bf16_to_f32(v) + bf16_to_f32(bias[n]));
                }
                out[static_cast<long long>(t % kM) * N + n] = v;
            }
        }
    }
    if constexpr (kCl > 1) {
        // No block retires while the first may read its sums.
        asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    }
}

template <class TA, class TW, class TY>
__global__ void lane_gemm(
    const TA* __restrict__ act,
    const TW* __restrict__ w,
    TY* __restrict__ y,
    int rows,
    int n,
    int k)
{
    const int warp = static_cast<int>(threadIdx.x) >> 5;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int c = static_cast<int>(blockIdx.x) * (static_cast<int>(blockDim.x) >> 5) + warp;
    const int r = static_cast<int>(blockIdx.y);
    if (c >= n || r >= rows) return;
    const TA* a = act + static_cast<long long>(r) * k;
    const TW* ww = w + static_cast<long long>(c) * k;
    float acc = 0.f;
    for (int i = lane; i < k; i += 32) {
        acc = fmaf(Elem<TA>::to_f32(a[i]), Elem<TW>::to_f32(ww[i]), acc);
    }
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) {
        y[static_cast<long long>(r) * n + c] = Elem<TY>::from_f32(acc);
    }
}

constexpr int kSkinnyM = 64;

template <int kBK>
__host__ __device__ constexpr int skinny_ld() {
    return kBK + 8;
}

__device__ __forceinline__ void skinny_ldmatrix_x4(unsigned (&reg)[4], const bf16* at) {
    const unsigned addr = static_cast<unsigned>(__cvta_generic_to_shared(at));
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(reg[0]), "=r"(reg[1]), "=r"(reg[2]), "=r"(reg[3])
                 : "r"(addr));
}

__device__ __forceinline__ unsigned long long skinny_policy(bool keep) {
    unsigned long long policy;
    if (keep) {
        asm volatile("createpolicy.fractional.L2::evict_last.b64 %0, 1.0;\n" : "=l"(policy));
    } else {
        asm volatile("createpolicy.fractional.L2::evict_first.b64 %0, 1.0;\n" : "=l"(policy));
    }
    return policy;
}

__device__ __forceinline__ void skinny_cp_async16(void* dst, const void* src, unsigned long long policy) {
    const unsigned d = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global.L2::cache_hint [%0], [%1], 16, %2;\n"
                 :
                 : "r"(d), "l"(src), "l"(policy)
                 : "memory");
}

__device__ __forceinline__ float skinny_geglu(float g, float u) {
    constexpr float kAlpha = 0.7978845608028654f;
    constexpr float kBeta = 0.044715f;
    const float inner = kAlpha * (g + kBeta * g * g * g);
    const float gelu = 0.5f * g * (1.f + tanhf(inner));
    return gelu * u;
}

template <int kWarps, int kStages, int kBK, int kEpilogue>
__global__ __launch_bounds__(kWarps * 32) void skinny_bf16_kernel(
    const bf16* __restrict__ w,
    const bf16* __restrict__ act,
    bf16* __restrict__ out,
    int m,
    int n,
    int k,
    float cap)
{
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    constexpr bool kGeglu = kEpilogue == 2;
    constexpr int kThreads = kWarps * 32;
    constexpr int kSkinnyLd = skinny_ld<kBK>();
    static_assert(kBK % 64 == 0, "a step is whole 128-byte row pieces");
    static_assert(!kGeglu || kWarps % 2 == 0, "geglu halves the block's rows between gate and up");

    constexpr int kN = kWarps * 16;

    constexpr int kCols = kGeglu ? kN / 2 : kN;

    constexpr int kNSubs = kSkinnyM / 8;

    constexpr int kQuad = kBK / 16;
    constexpr int kWStage = kN * kSkinnyLd;
    constexpr int kAStage = kSkinnyM * kSkinnyLd;
    constexpr int kStageElems = kWStage + kAStage;
    constexpr int kChunksPerRow = kBK / 8;
    constexpr int kWChunks = kN * kChunksPerRow;
    constexpr int kAChunks = kSkinnyM * kChunksPerRow;

    constexpr int kLdC = kN + 8;
    static_assert(kStages >= 2, "one stage in flight while one is multiplied, at least");
    static_assert(kWChunks % kThreads == 0 && kAChunks % kThreads == 0,
                  "the staging map is a whole number of 16-byte chunks per thread");
    static_assert(kSkinnyM * kLdC <= kStageElems, "the epilogue tile fits in one stage of the ring");
    static_assert(kCols % 8 == 0, "the drain stores whole 16-byte chunks");

    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;

    const int n0 = static_cast<int>(blockIdx.x) * kCols;

    extern __shared__ __align__(16) unsigned char skinny_smem[];
    bf16* ring = reinterpret_cast<bf16*>(skinny_smem);

    if (m < kSkinnyM) {
        u32* z = reinterpret_cast<u32*>(skinny_smem);
        constexpr int kWords = kStages * kStageElems / 2;
        for (int i = tid; i < kWords; i += kThreads) z[i] = 0u;
        __syncthreads();
    }

    auto weight_row = [&](int r) -> int {
        if constexpr (kGeglu) {
            return r < kCols ? n0 + r : n + n0 + (r - kCols);
        } else {
            return n0 + r;
        }
    };

    const unsigned long long stream_policy = skinny_policy(false);
    const unsigned long long keep_policy = skinny_policy(true);

    auto stage = [&](int buf, int k0) {
        bf16* wdst = ring + buf * kStageElems;
        bf16* adst = wdst + kWStage;
#pragma unroll
        for (int i = 0; i < kWChunks / kThreads; ++i) {
            const int chunk = tid + i * kThreads;
            const int r = chunk / kChunksPerRow;
            const int c = (chunk % kChunksPerRow) * 8;
            skinny_cp_async16(
                wdst + r * kSkinnyLd + c,
                w + static_cast<long long>(weight_row(r)) * k + k0 + c,
                stream_policy);
        }
#pragma unroll
        for (int i = 0; i < kAChunks / kThreads; ++i) {
            const int chunk = tid + i * kThreads;
            const int r = chunk / kChunksPerRow;
            const int c = (chunk % kChunksPerRow) * 8;
            if (r < m) {
                skinny_cp_async16(
                    adst + r * kSkinnyLd + c,
                    act + static_cast<long long>(r) * k + k0 + c,
                    keep_policy);
            }
        }
        __pipeline_commit();
    };

    const int steps = k / kBK;

    float acc[kNSubs][4];
#pragma unroll
    for (int s = 0; s < kNSubs; ++s) {
#pragma unroll
        for (int i = 0; i < 4; ++i) acc[s][i] = 0.f;
    }

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        if (s < steps) {
            stage(s, s * kBK);
        } else {
            __pipeline_commit();
        }
    }

    const int a_row = lane & 15;
    const int a_col = (lane >> 4) << 3;

    for (int step = 0; step < steps; ++step) {

        const int fetch = step + kStages - 1;
        if (fetch < steps) {
            stage(fetch % kStages, fetch * kBK);
        } else {
            __pipeline_commit();
        }
        __pipeline_wait_prior(kStages - 1);
        __syncthreads();

        const bf16* wsrc = ring + (step % kStages) * kStageElems + warp * 16 * kSkinnyLd;
        const bf16* asrc = ring + (step % kStages) * kStageElems + kWStage;
#pragma unroll
        for (int kk = 0; kk < kQuad; ++kk) {
            unsigned a[4];
            skinny_ldmatrix_x4(a, wsrc + a_row * kSkinnyLd + kk * 16 + a_col);
#pragma unroll
            for (int pair = 0; pair < kNSubs / 2; ++pair) {
                unsigned b[4];
                skinny_ldmatrix_x4(b, asrc + (pair * 16 + a_row) * kSkinnyLd + kk * 16 + a_col);
                const unsigned b_lo[2] = {b[0], b[2]};
                const unsigned b_hi[2] = {b[1], b[3]};
                ::nvcuda::wmma::detail::mma_m16n8k16(acc[2 * pair], a, b_lo, acc[2 * pair]);
                ::nvcuda::wmma::detail::mma_m16n8k16(acc[2 * pair + 1], a, b_hi, acc[2 * pair + 1]);
            }
        }
        __syncthreads();
    }

    __pipeline_wait_prior(0);
    __syncthreads();
    bf16* c_tile = ring;
    {
        const int g = lane >> 2;
        const int t = lane & 3;
#pragma unroll
        for (int s = 0; s < kNSubs; ++s) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int col = warp * 16 + g + 8 * half;
                const int row = s * 8 + 2 * t;
                c_tile[row * kLdC + col] = Elem<bf16>::from_f32(acc[s][2 * half]);
                c_tile[(row + 1) * kLdC + col] = Elem<bf16>::from_f32(acc[s][2 * half + 1]);
            }
        }
    }
    __syncthreads();

    constexpr int kChunksOfRow = kCols / 8;
    constexpr int kDrain = kSkinnyM * kChunksOfRow;
#pragma unroll
    for (int i = 0; i < (kDrain + kThreads - 1) / kThreads; ++i) {
        const int chunk = tid + i * kThreads;
        if (chunk < kDrain) {
            const int r = chunk / kChunksOfRow;
            const int c = (chunk % kChunksOfRow) * 8;
            if (r < m) {
                bf16* dst = out + static_cast<long long>(r) * n + n0 + c;
                const bf16* src = c_tile + r * kLdC + c;
                if constexpr (kEpilogue == 0) {
                    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src);
                } else {
                    __align__(16) bf16 piece[8];
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        float v;
                        if constexpr (kEpilogue == 1) {
                            v = cap * tanhf(Elem<bf16>::to_f32(src[e]) * (1.f / cap));
                        } else {
                            v = skinny_geglu(
                                Elem<bf16>::to_f32(src[e]),
                                Elem<bf16>::to_f32(src[kCols + e]));
                        }
                        piece[e] = Elem<bf16>::from_f32(v);
                    }
                    *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(piece);
                }
            }
        }
    }
}

}
