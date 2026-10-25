#pragma once

// The contraction frames: `y = act . w^T` in bf16 over a decode's few rows,
// each a hand-written parallel skeleton that computes the products and hands
// them, a few elements of one row of the product tile at a time, to an
// epilogue body through the hooks of `TileHook`. The body is a kernels-ir
// region over the product tile, printed by kernels-cuda (`linear::frame`);
// the plain store is the body that stores each product as it is.
//
// - `gemv_frame`, `gemv_splitk_frame`: one row, a warp (or a block split
//   over K) a column.
// - `rows_frame`: a few rows (up to 16), a block's eight warps streaming
//   weight rows.
// - `skinny_frame`: up to 64 rows on mma.sync, the weight streamed once past
//   the activation's rows in shared memory.
//
// A body's hooks are the row frame's (`elemwise/frame.cuh`), so one printer
// serves both: `F::N` elements a call, `f.ld(k, e)` (a product where `k` is
// below the parts, else an input's element), `f.one(k)` (an input's one
// value a row), `f.st(k, e, v)`, `f.valid(e)`, `f.round_bf16(v)`,
// `f.col(e)`, `f.sin(v)`, `f.cos(v)` and `f.flush<k>()`. An epilogue reduces nothing (kernels-ir refuses one that
// does), so a frame offers no reduction.

#include "prim/device.cuh"
#include "prim/mma_sm80.cuh"
#include "prim/smem.cuh"
#include <cuda_bf16.h>
#include <cuda_pipeline.h>

namespace pie::linear {

// One of an epilogue's inputs: its element `(row, col)` of the product tile
// at `base[row * row_stride + col * col_stride]`. A tile is `(ld, 1)`, a
// vector a column `(0, 1)`, a vector a row `(1, 0)`, one value `(0, 0)`.
// `kernels_cuda::linear::frame::In` mirrors it.
struct EpiIn {
    const bf16* base;
    long long row_stride;
    long long col_stride;
};

// An epilogue's IN inputs and OUT results, each result an `m x n` bf16 plane
// `ld` apart a row. One input past IN, unread, so a body of none is a struct
// still.
template <int IN, int OUT>
struct Epi {
    EpiIn in[IN + 1];
    bf16* out[OUT];
    long long ld;
};

// `kN` consecutive elements of one row of the product tile, from column
// `col`: each of the `P` parts' products, set by the frame (rounded to bf16
// first where the contraction declares it, `ROUND`), and the body's results
// held until it flushes them. Every `k` and `e` a body passes is constant
// once its loops unroll, so the arrays stay in registers.
template <int P, int kN, int IN, int OUT, bool ROUND>
struct TileHook {
    static constexpr int N = kN;

    const Epi<IN, OUT>& epi;
    long long row;
    int col_;
    int width;
    float acc[P][kN];
    __align__(16) bf16 res[OUT][kN];

    __device__ __forceinline__ TileHook(const Epi<IN, OUT>& epi, long long row, int col, int width)
        : epi(epi), row(row), col_(col), width(width) {}

    // The frame's: part `p`'s product at element `e`.
    __device__ __forceinline__ void set(int p, int e, float v) {
        acc[p][e] = ROUND ? round_bf16(v) : v;
    }

    __device__ __forceinline__ bool valid(int e) const { return col_ + e < width; }

    // The product tile's column of element `e`.
    __device__ __forceinline__ unsigned col(int e) const { return static_cast<unsigned>(col_ + e); }

    // An angle's sine and cosine as the row frame turns one.
    __device__ __forceinline__ float sin(float v) const { return __sinf(v); }
    __device__ __forceinline__ float cos(float v) const { return __cosf(v); }

    __device__ __forceinline__ float ld(int k, int e) const {
        if (k < P) return acc[k < P ? k : 0][e];
        const EpiIn& in = epi.in[k - P];
        return bf16_to_f32(in.base[row * in.row_stride + static_cast<long long>(col_ + e) * in.col_stride]);
    }

    __device__ __forceinline__ float one(int k) const {
        const EpiIn& in = epi.in[k - P];
        return bf16_to_f32(in.base[row * in.row_stride]);
    }

    // `v` as result `k` holds it, rounded to its bf16 plane.
    __device__ __forceinline__ float st(int k, int e, float v) {
        res[k][e] = f32_to_bf16(v);
        return bf16_to_f32(res[k][e]);
    }

    __device__ __forceinline__ float round_bf16(float v) const { return bf16_to_f32(f32_to_bf16(v)); }

    // Result `K`'s elements to memory: whole vectors of 8 where the frame
    // holds them (its columns and `ld` whole eights, its planes 16-byte
    // aligned), else one by one.
    template <int K>
    __device__ __forceinline__ void flush() {
        bf16* dst = epi.out[K] + row * epi.ld + col_;
        if constexpr (kN % 8 == 0) {
#pragma unroll
            for (int v = 0; v < kN / 8; ++v) {
                if (col_ + v * 8 < width) {
                    *reinterpret_cast<uint4*>(dst + v * 8) = *reinterpret_cast<const uint4*>(&res[K][v * 8]);
                }
            }
        } else {
#pragma unroll
            for (int e = 0; e < kN; ++e) {
                if (valid(e)) dst[e] = res[K][e];
            }
        }
    }
};

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

// One row: a warp a column, `kWarps` columns a block, each part's weight
// row `col + p * N` in turn. Lane 0 hands the column's products to the body.
template <class Body, int kWarps, int kUnroll, int P, int IN, int OUT, bool ROUND>
__global__ void gemv_frame(
    const bf16* __restrict__ weight,
    const bf16* __restrict__ act,
    const Epi<IN, OUT> epi,
    const Body body,
    int N, int K)
{
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    const int col = blockIdx.x * kWarps + threadIdx.y;
    if (col >= N) return;
    float acc[P];
#pragma unroll
    for (int p = 0; p < P; ++p) acc[p] = gemv_bf16_row<kUnroll>(weight, act, col + p * N, K);
    if (threadIdx.x == 0) {
        TileHook<P, 1, IN, OUT, ROUND> f(epi, 0, col, N);
#pragma unroll
        for (int p = 0; p < P; ++p) f.set(p, 0, acc[p]);
        body(f);
    }
}

// One row, a block a column split over K across its `kWarps` warps: each
// warp's slice folds across its lanes, and the slices add in warp order.
template <class Body, int kWarps, int kUnrollP, int P, int IN, int OUT, bool ROUND>
__global__ void gemv_splitk_frame(
    const bf16* __restrict__ weight,
    const bf16* __restrict__ act,
    const Epi<IN, OUT> epi,
    const Body body,
    int N, int K)
{
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    const int col = blockIdx.x;
    if (col >= N) return;
    const int warp = threadIdx.y;
    const float4* x4 = reinterpret_cast<const float4*>(act);
    const int vectors = K / 8;
    constexpr int kU = kUnrollP;
    const int stride = kWarps * 32;
    __shared__ float partial[P][kWarps];

#pragma unroll
    for (int p = 0; p < P; ++p) {
        const float4* w4 =
            reinterpret_cast<const float4*>(weight + (long long)(col + p * N) * K);
        float acc = 0.f;
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
        if (threadIdx.x == 0) partial[p][warp] = acc;
    }
    __syncthreads();
    if (warp != 0 || threadIdx.x != 0) return;
    TileHook<P, 1, IN, OUT, ROUND> f(epi, 0, col, N);
#pragma unroll
    for (int p = 0; p < P; ++p) {
        float total = 0.f;
        #pragma unroll
        for (int w = 0; w < kWarps; ++w) total += partial[p][w];
        f.set(p, 0, total);
    }
    body(f);
}

// A few rows (`kM` <= 16, a decode's): each of a block's eight warps
// streams two weight rows' slice of K, a weight vector of each against
// every activation row (read through L1, so each activation load serves two
// weight loads), `kSplit` warps the rows' slices; each warp's sums fold
// across its lanes, and a row's slices add in slice order, so a launch
// lands the same bits each time. One launch, every weight byte read once: a
// split GEMM's reduce launch saved. Over a cluster of `kCl` blocks (a few
// columns, too few blocks to keep the loads in flight), each block takes a
// slice of K and the first adds the blocks' sums in rank order. Each
// product goes to the body alone: one dense bank.
template <class Body, int kM, int kSplit, int kCl, int P, int IN, int OUT, bool ROUND>
__global__ void __launch_bounds__(256) rows_frame(
    const bf16* __restrict__ weight,
    const bf16* __restrict__ act,
    const Epi<IN, OUT> epi,
    const Body body,
    int N, int K)
{
    static_assert(kM >= 1 && kM <= 16, "a few rows");
    static_assert(P == 1, "one dense bank");
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
            acc[q][r] = ::pie::warp_sum(acc[q][r]);
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
                TileHook<1, 1, IN, OUT, ROUND> f(epi, t % kM, n, N);
                f.set(0, 0, s);
                body(f);
            }
        }
    }
    if constexpr (kCl > 1) {
        // No block retires while the first may read its sums.
        asm volatile("barrier.cluster.arrive.release.aligned;\n\tbarrier.cluster.wait.acquire.aligned;" ::: "memory");
    }
}

constexpr int kSkinnyM = 64;

template <int kBK>
__host__ __device__ constexpr int skinny_ld() {
    return kBK + 8;
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

// The product tile as the skinny frame holds it: bf16 where the contraction
// rounds its products to bf16 (the same values), f32 where it reads them
// unrounded.
template <bool ROUND> struct SkinnyTile { using T = float; };
template <> struct SkinnyTile<true> { using T = bf16; };

// Up to 64 rows: a block's `kWarps` warps take 16 weight rows each, the
// block's `kWarps * 16` rows `P` parts of `kWarps * 16 / P` columns (part
// `p`'s column `c` weight row `p * n + c`), against every activation row
// staged in shared memory, `kStages` steps of `kBK` in flight. The weight
// streams through L2 once (evict-first), the activation stays (evict-last).
// The products land in a tile over the ring; each thread then hands the body
// 8 columns of one row, every part's product at each.
template <class Body, int kWarps, int kStages, int kBK, int P, int IN, int OUT, bool ROUND>
__global__ __launch_bounds__(kWarps * 32) void skinny_frame(
    const bf16* __restrict__ w,
    const bf16* __restrict__ act,
    const Epi<IN, OUT> epi,
    const Body body,
    int m,
    int n,
    int k)
{
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    using T = typename SkinnyTile<ROUND>::T;
    constexpr int kThreads = kWarps * 32;
    constexpr int kSkinnyLd = skinny_ld<kBK>();
    static_assert(kBK % 64 == 0, "a step is whole 128-byte row pieces");

    constexpr int kN = kWarps * 16;
    static_assert(kN % (P * 8) == 0, "each part is whole vectors of 8 columns of the block's rows");
    constexpr int kCols = kN / P;

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
    static_assert(kSkinnyM * kLdC * sizeof(T) <= kStages * kStageElems * sizeof(bf16),
                  "the product tile fits over the ring");
    static_assert(kCols % 8 == 0, "the drain hands whole vectors of 8");

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

    auto weight_row = [&](int r) -> int { return (r / kCols) * n + n0 + r % kCols; };

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
            ldmatrix_x4(a, wsrc + a_row * kSkinnyLd + kk * 16 + a_col);
#pragma unroll
            for (int pair = 0; pair < kNSubs / 2; ++pair) {
                unsigned b[4];
                ldmatrix_x4(b, asrc + (pair * 16 + a_row) * kSkinnyLd + kk * 16 + a_col);
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
    T* c_tile = reinterpret_cast<T*>(skinny_smem);
    {
        const int g = lane >> 2;
        const int t = lane & 3;
#pragma unroll
        for (int s = 0; s < kNSubs; ++s) {
#pragma unroll
            for (int half = 0; half < 2; ++half) {
                const int col = warp * 16 + g + 8 * half;
                const int row = s * 8 + 2 * t;
                c_tile[row * kLdC + col] = static_cast<T>(acc[s][2 * half]);
                c_tile[(row + 1) * kLdC + col] = static_cast<T>(acc[s][2 * half + 1]);
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
                TileHook<P, 8, IN, OUT, ROUND> f(epi, r, n0 + c, n);
#pragma unroll
                for (int p = 0; p < P; ++p) {
#pragma unroll
                    for (int e = 0; e < 8; ++e) {
                        f.set(p, e, static_cast<float>(c_tile[r * kLdC + p * kCols + c + e]));
                    }
                }
                body(f);
            }
        }
    }
}

// The plain store as a contraction frame's body, written by hand: each
// product stored as it is. The frames' own reference, as `elemwise/frame.cu`
// is the row frame's; a generated epilogue (`linear::frame`) fires through
// the same frames.
struct Store {
    template <class F>
    __device__ __forceinline__ void operator()(F& f) const {
#pragma unroll
        for (int e = 0; e < F::N; ++e) f.st(0, e, f.ld(0, e));
        f.template flush<0>();
    }
};

}
