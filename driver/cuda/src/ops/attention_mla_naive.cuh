// Naive paged MLA kernel (Blackwell / sm100 fallback).
//
// Lives in a header so the standalone microbenchmark
// (`driver/cuda/bench/mla_bench.cu`) compiles the *same* source the driver
// runs. Iterating on this kernel through a full engine rebuild costs minutes;
// the microbenchmark costs seconds.
#pragma once

#include <cstdint>
#include <cstdlib>
#include <mutex>
#include <stdexcept>
#include <string>

#include <cuda_bf16.h>
#include <cuda_pipeline.h>
#include <cuda_runtime.h>
#include <math_constants.h>

#ifndef PIE_MLA_NAIVE_CHECK
#define PIE_MLA_NAIVE_CHECK(expr)                                             \
    do {                                                                      \
        const cudaError_t pie_mla_err_ = (expr);                              \
        if (pie_mla_err_ != cudaSuccess) {                                    \
            throw std::runtime_error(std::string("naive MLA: ") +             \
                                     cudaGetErrorString(pie_mla_err_));       \
        }                                                                     \
    } while (0)
#endif

namespace pie_cuda_driver::ops::mla_naive {

inline bool mla_mma_supported(int kv_lora_rank, int qk_rope_head_dim, int num_heads);
inline void launch_mla_mma_paged_raw(
    const void* q_nope, const void* q_pe,
    const void* ckv_pages, const void* kpe_pages, int page_size, void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    int total_tokens, int num_requests, int num_heads,
    float sm_scale, bool causal, cudaStream_t stream,
    const std::uint8_t* index_mask, int index_mask_stride);

constexpr int kMlaNaiveBlock = 256;
constexpr int kMlaNaiveWarps = kMlaNaiveBlock / 32;
constexpr int kMlaNaiveMaxPer = 16;  // kv_lora_rank <= 512 with 32 lanes
constexpr int kMlaNaiveMaxPePer = 4; // qk_rope_head_dim <= 128 with 32 lanes

// Flash-style online softmax over the paged ckv/kpe cache.
//
// The 16 warps of a block are split into `G` head slots x `K = 16 / G` key
// slices, and the block covers heads `[blockIdx.y * G, +G)` of token
// `blockIdx.x`. MLA decode is MQA-shaped — every query head reads the *same*
// latent KV — so grouping heads into one block is what keeps the KV traffic
// down: the G warps that share a key slice walk the identical `j` sequence and
// hit in L1 instead of pulling the same cache lines G times from L2/HBM. With
// `G == 1` this degenerates to the pure key-parallel layout, which is what
// small batches want (there the grid is the only source of parallelism).
//
// Scores are reduced inside a warp, not across the block: a block-wide tree
// reduction per key costs seven `__syncthreads()` for every KV entry, which at
// decode dwarfs the arithmetic. Each warp keeps its own running max/sum/
// accumulator in registers and the partial softmax states are merged once at
// the end — the structure flash-decoding uses.
__global__ void mla_naive_paged_kernel(
    const __nv_bfloat16* __restrict__ q_nope,   // [N, H, CKV]
    const __nv_bfloat16* __restrict__ q_pe,     // [N, H, KPE]
    const __nv_bfloat16* __restrict__ ckv_pages,// [pages, page_size, CKV]
    const __nv_bfloat16* __restrict__ kpe_pages,// [pages, page_size, KPE]
    const std::uint32_t* __restrict__ qo_indptr,
    const std::uint32_t* __restrict__ kv_page_indices,
    const std::uint32_t* __restrict__ kv_page_indptr,
    const std::uint32_t* __restrict__ kv_last_page_lens,
    __nv_bfloat16* __restrict__ o,              // [N, H, CKV]
    const std::uint8_t* __restrict__ index_mask, int index_mask_stride,
    int R, int H, int CKV, int KPE, int page_size, float sm_scale, bool causal,
    int G)
{
    const int t = blockIdx.x;   // query token
    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    const int K = kMlaNaiveWarps / G;  // key slices per head
    const int g = warp / K;            // head slot inside the group
    const int s = warp % K;            // key slice
    const int h = blockIdx.y * G + g;  // head
    const int per = CKV / 32;          // latent dims per lane
    const int pper = KPE / 32;
    // DSA mask row for this query (in-batch keys only).
    const std::uint8_t* mrow =
        (index_mask != nullptr) ? index_mask + static_cast<long long>(t) * index_mask_stride
                                : nullptr;

    // Resolve request, kv length, and this query's absolute position.
    int lo = 0, hi = R - 1;
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (t < static_cast<int>(qo_indptr[mid + 1])) hi = mid; else lo = mid + 1;
    }
    const int r = lo;
    const int qo_lo = static_cast<int>(qo_indptr[r]);
    const int new_tokens = static_cast<int>(qo_indptr[r + 1]) - qo_lo;
    const int pages_first = static_cast<int>(kv_page_indptr[r]);
    const int num_pages = static_cast<int>(kv_page_indptr[r + 1]) - pages_first;
    const int kv_len =
        (num_pages - 1) * page_size + static_cast<int>(kv_last_page_lens[r]);
    const int pre_kv = kv_len - new_tokens;
    const int abs_q = pre_kv + (t - qo_lo);
    const int j_end = causal ? (abs_q + 1) : kv_len;

    extern __shared__ float smem[];
    float* wacc  = smem;                        // kMlaNaiveWarps * CKV
    float* wm    = wacc + kMlaNaiveWarps * CKV; // kMlaNaiveWarps
    float* wl    = wm + kMlaNaiveWarps;         // kMlaNaiveWarps

    // The query stays in registers: each lane owns the dims it needs for its
    // slice of the dot product, so there is no per-block staging or barrier.
    const __nv_bfloat16* qn =
        q_nope + (static_cast<long long>(t) * H + h) * CKV;
    const __nv_bfloat16* qp =
        q_pe + (static_cast<long long>(t) * H + h) * KPE;
    float qn_r[kMlaNaiveMaxPer];
    float qp_r[kMlaNaiveMaxPePer];
    for (int i = 0; i < per; ++i) qn_r[i] = __bfloat162float(qn[lane + i * 32]);
    for (int i = 0; i < pper; ++i) qp_r[i] = __bfloat162float(qp[lane + i * 32]);

    float acc[kMlaNaiveMaxPer];
    for (int i = 0; i < per; ++i) acc[i] = 0.f;
    float m = -CUDART_INF_F, lsum = 0.f;

    for (int j = s; j < j_end; j += K) {
        // DSA: skip keys not selected by the lightning indexer (in-batch only).
        if (mrow != nullptr && j < index_mask_stride && mrow[j] == 0) continue;
        const int page =
            static_cast<int>(kv_page_indices[pages_first + j / page_size]);
        const int off = j % page_size;
        const __nv_bfloat16* ckv_j =
            ckv_pages + (static_cast<long long>(page) * page_size + off) * CKV;
        const __nv_bfloat16* kpe_j =
            kpe_pages + (static_cast<long long>(page) * page_size + off) * KPE;
        // The latent row is read once and kept in registers: it is needed both
        // for the score and, right after, as the value vector. Lanes take a
        // strided slice (`lane + i*32`): a warp still covers 64 contiguous
        // bytes per instruction, and unpacking wider vector loads costs more
        // registers than the extra instructions save.
        float kv[kMlaNaiveMaxPer];
        float pd = 0.f;
        for (int i = 0; i < per; ++i) {
            kv[i] = __bfloat162float(ckv_j[lane + i * 32]);
            pd += qn_r[i] * kv[i];
        }
        for (int i = 0; i < pper; ++i) {
            pd += qp_r[i] * __bfloat162float(kpe_j[lane + i * 32]);
        }
        #pragma unroll
        for (int sh = 16; sh > 0; sh >>= 1) {
            pd += __shfl_xor_sync(0xffffffffu, pd, sh);
        }
        const float score = pd * sm_scale;
        const float m_new = fmaxf(m, score);
        const float corr = __expf(m - m_new);
        const float p = __expf(score - m_new);
        lsum = lsum * corr + p;
        for (int i = 0; i < per; ++i) {
            acc[i] = acc[i] * corr + p * kv[i];
        }
        m = m_new;
    }

    // Merge the per-warp softmax states (per head slot, across its K slices).
    for (int i = 0; i < per; ++i) {
        wacc[warp * CKV + lane + i * 32] = acc[i];
    }
    if (lane == 0) { wm[warp] = m; wl[warp] = lsum; }
    __syncthreads();

    const int total_out = G * CKV;
    for (int idx = tid; idx < total_out; idx += kMlaNaiveBlock) {
        const int gg = idx / CKV;
        const int d = idx % CKV;
        const int w0 = gg * K;
        float m_all = -CUDART_INF_F;
        for (int w = w0; w < w0 + K; ++w) m_all = fmaxf(m_all, wm[w]);
        float l_all = 0.f, v = 0.f;
        for (int w = w0; w < w0 + K; ++w) {
            if (wm[w] > -CUDART_INF_F) {
                const float e = __expf(wm[w] - m_all);
                l_all += wl[w] * e;
                v += wacc[w * CKV + d] * e;
            }
        }
        const float inv = (l_all > 0.f) ? (1.f / l_all) : 0.f;
        o[(static_cast<long long>(t) * H + blockIdx.y * G + gg) * CKV + d] =
            __float2bfloat16(v * inv);
    }
}

inline void launch_mla_naive_paged_raw(
    const void* q_nope, const void* q_pe,
    const void* ckv_pages, const void* kpe_pages,
    int kv_lora_rank, int qk_rope_head_dim, int page_size, void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    int total_tokens, int num_requests, int num_heads,
    float sm_scale, bool causal, cudaStream_t stream,
    const std::uint8_t* index_mask, int index_mask_stride)
{
    if (total_tokens <= 0) return;
    if (qo_indptr_d == nullptr || kv_page_indptr_d == nullptr ||
        kv_last_page_lens_d == nullptr) {
        throw std::runtime_error(
            "naive MLA: missing device indptr/lens (qo/kv_page_indptr/"
            "kv_last_page_lens)");
    }
    if (mla_mma_supported(kv_lora_rank, qk_rope_head_dim, num_heads)) {
        launch_mla_mma_paged_raw(
            q_nope, q_pe, ckv_pages, kpe_pages, page_size, o, qo_indptr_d,
            kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d,
            total_tokens, num_requests, num_heads, sm_scale, causal, stream,
            index_mask, index_mask_stride);
        return;
    }
    const int CKV = kv_lora_rank;
    const int KPE = qk_rope_head_dim;
    if (CKV % 32 != 0 || CKV / 32 > kMlaNaiveMaxPer) {
        throw std::runtime_error("naive MLA: unsupported kv_lora_rank");
    }
    if (KPE % 32 != 0 || KPE / 32 > kMlaNaiveMaxPePer) {
        throw std::runtime_error("naive MLA: unsupported qk_rope_head_dim");
    }
    // Pick the largest head group that still fills the machine. Every head in a
    // block walks the same keys, so a bigger group means the latent KV is read
    // from L1 instead of L2/HBM — but the grid is (tokens x head-groups), so
    // shrinking it too far starves the SMs. Two waves is the target.
    constexpr int kForcedGroup = 0;
    const int kMlaWaveTarget = 296;
    int G = kMlaNaiveWarps;
    if (kForcedGroup > 0) {
        G = kForcedGroup;
        while (G > 1 && (num_heads % G != 0 || kMlaNaiveWarps % G != 0)) G >>= 1;
    } else {
        while (G > 1 &&
               (num_heads % G != 0 ||
                static_cast<long long>(total_tokens) * (num_heads / G) <
                    kMlaWaveTarget)) {
            G >>= 1;
        }
    }
    const std::size_t smem =
        (static_cast<std::size_t>(kMlaNaiveWarps) * CKV +
         2 * kMlaNaiveWarps) * sizeof(float);
    // Wide blocks are what make this kernel fast at decode: the grid is only
    // (tokens x head-groups), so with a narrow block the SMs sit at single-digit
    // occupancy and every key's load latency is exposed. The partial-softmax
    // scratch that buys the extra warps can exceed the 48 KB default.
    static std::once_flag smem_optin;
    std::call_once(smem_optin, [&] {
        cudaFuncSetAttribute(
            reinterpret_cast<const void*>(mla_naive_paged_kernel),
            cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
    });
    dim3 grid(total_tokens, num_heads / G);
    mla_naive_paged_kernel<<<grid, kMlaNaiveBlock, smem, stream>>>(
        static_cast<const __nv_bfloat16*>(q_nope),
        static_cast<const __nv_bfloat16*>(q_pe),
        static_cast<const __nv_bfloat16*>(ckv_pages),
        static_cast<const __nv_bfloat16*>(kpe_pages),
        qo_indptr_d, kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d,
        static_cast<__nv_bfloat16*>(o),
        index_mask, index_mask_stride,
        num_requests, num_heads, CKV, KPE, page_size, sm_scale, causal, G);
    PIE_MLA_NAIVE_CHECK(cudaGetLastError());
}


// ── Tensor-core paged MLA ───────────────────────────────────────────────
// The scalar kernel above tops out around 3 TFLOP/s on every shape, which is
// ~4% of the FP32 CUDA-core roofline and ~0.2% of the BF16 tensor-core one.
// MLA is MQA-shaped -- all H query heads of a token read the *same* latent KV
// row -- so the H heads of one token form the M dimension of a real GEMM:
//
//     S = Q[BM, 576] x K[BK, 576]^T        (BM = 16 heads, BK = 32 keys)
//     O = P[BM, BK]  x V[BK, 512]          (V is K's first 512 columns)
//
// Both operands live in one shared-memory K tile: `sK[key][dim]` is already
// the `.col` B-layout for QK^T (n = key, k = dim) and the `.row` B-layout for
// PV (k = key, n = dim), so neither GEMM needs a transpose.
//
// Warps split the *output* dimension, not the rows: with a 512-wide V a single
// warp owning all of it would need 256 accumulator registers. Four warps own
// 128 dims each (64 accumulator floats), and phase 1 gives each warp one n8
// column strip of S.
//
// Requires kv_lora_rank == 512, qk_rope_head_dim == 64 and num_heads % 16 == 0
// (true for GLM-5.2, Kimi K2.6 and both DeepSeek-V4 variants); anything else
// falls back to the scalar kernel.
namespace mma_detail {

#ifndef PIE_MLA_MMA_BK
#define PIE_MLA_MMA_BK 64
#endif
#ifndef PIE_MLA_MMA_WARPS
#define PIE_MLA_MMA_WARPS 8
#endif
// KV tiles are staged through shared memory with cp.async instead of a
// global -> register -> shared round trip. Extra stages buy cross-tile overlap
// but cost a full sK copy (73 KB) of shared memory, and on B200 that drops the
// block occupancy from 2/SM to 1/SM, which loses more than the overlap wins:
// at 128x64 heads, kv=1024, stages=1 is 0.275 ms and stages=2 is 0.428 ms.
// Keep this at 1 unless sK gets small enough that two stages still fit twice.
#ifndef PIE_MLA_MMA_STAGES
#define PIE_MLA_MMA_STAGES 1
#endif
// Occupancy hint. Shared memory (see `smem_bytes()`) already caps residency,
// so asking for more blocks than that only tightens the register budget and
// buys spills.
#ifndef PIE_MLA_MMA_MINBLK
#define PIE_MLA_MMA_MINBLK 2
#endif

constexpr int kBM = 16;                    // query rows per block == heads
constexpr int kBK = PIE_MLA_MMA_BK;        // keys per tile
constexpr int kWarps = PIE_MLA_MMA_WARPS;
constexpr int kStages = PIE_MLA_MMA_STAGES;
static_assert(kStages >= 1, "kStages must be at least 1");
constexpr int kThreads = kWarps * 32;
constexpr int kCkv = 512;
constexpr int kKpe = 64;
constexpr int kD = kCkv + kKpe;   // 576
constexpr int kLdD = kD + 8;      // 584 halves = 1168 B, conflict-free
constexpr int kLdP = kBK + 8;     // 40
constexpr int kDimsPerWarp = kCkv / kWarps;
constexpr int kNTiles = kDimsPerWarp / 8;
constexpr int kSNTiles = kBK / 8 / kWarps;    // score n8 tiles per warp
static_assert(kSNTiles >= 1, "kBK must be at least 8 * kWarps");
static_assert(kBK % 16 == 0, "kBK must be a multiple of the mma k step");

__device__ __forceinline__ std::uint32_t pack2(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    std::uint32_t l, h;
    std::memcpy(&l, &lo, 2);
    std::memcpy(&h, &hi, 2);
    return (l & 0xffffu) | ((h & 0xffffu) << 16);
}

// mma.m16n8k16 fragment layouts (PTX ISA "Matrix Fragments for mma.m16n8k16"):
//   A[16][16] row-major : a0/a1 -> rows g, g+8 at cols 2p, 2p+1
//                         a2/a3 -> the same rows at cols 2p+8, 2p+9
//   B[16][8]  col-major : b0    -> col g at rows 2p, 2p+1
//                         b1    -> col g at rows 2p+8, 2p+9
//   C[16][8]  f32       : c0/c1 -> row g   at cols 2p, 2p+1
//                         c2/c3 -> row g+8 at cols 2p, 2p+1
// with g = lane >> 2 and p = lane & 3.
__device__ __forceinline__ void mma_m16n8k16(float (&d)[4], const std::uint32_t (&a)[4],
                                             const std::uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

// A fragment from a row-major [16][ld] tile.
// A fragment (16x16) for both S = Q.K^T and O = P.V. One ldmatrix.x4 replaces
// four 32-bit shared loads; the mma issue rate, not memory, is what limits this
// kernel, so the instruction count per mma is the thing worth cutting.
// ldmatrix.x4 wants lane l to address row (l & 15) of the k0 half for l < 16 and
// of the k0+8 half for l >= 16, which is exactly the four 8x8 tiles the mma A
// operand is made of.
__device__ __forceinline__ void ld_a(std::uint32_t (&a)[4], const __nv_bfloat16* base,
                                     int ld, int lane, int k0) {
    const __nv_bfloat16* r =
        base + static_cast<long long>(lane & 15) * ld + k0 + ((lane & 16) ? 8 : 0);
    const std::uint32_t addr =
        static_cast<std::uint32_t>(__cvta_generic_to_shared(r));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3])
        : "r"(addr));
}

// B fragment for S = Q.K^T: n = key (row of sK), k = dim (contiguous).
// Keys are rows of sK with the head dim contiguous, so the mma B operand is a
// plain (untransposed) 8x8 pair: lanes 0-7 address the k0 half, lanes 8-15 the
// k0+8 half. Two 32-bit shared loads collapse into one ldmatrix.x2.
__device__ __forceinline__ void ld_b_k(std::uint32_t (&b)[2], const __nv_bfloat16* base,
                                       int ld, int lane, int k0) {
    const __nv_bfloat16* q =
        base + static_cast<long long>(lane & 7) * ld + k0 + ((lane & 8) ? 8 : 0);
    const std::uint32_t addr =
        static_cast<std::uint32_t>(__cvta_generic_to_shared(q));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(b[0]), "=r"(b[1])
        : "r"(addr));
}

// B fragment for O = P.V: k = key (row of sK), n = dim (column of sK).
// V is the transpose of what sK holds, so this is a transposed 8x8 fetch --
// exactly `ldmatrix.trans`. Doing it by hand costs four strided 2-byte shared
// loads per fragment (32 fragments per warp per KV tile) and hits bank
// conflicts; the hardware path is one instruction and, with kLdD = kD + 8,
// conflict-free: the eight row addresses land on banks 4*row mod 32.
__device__ __forceinline__ void ld_b_v(std::uint32_t (&b)[2], const __nv_bfloat16* sK,
                                       int ld, int lane, int k0, int nbase) {
    // Lanes 0-7 address matrix 0 (keys k0..k0+7), lanes 8-15 matrix 1
    // (k0+8..k0+15); lanes 16-31 are ignored by ldmatrix.x2.
    const __nv_bfloat16* q =
        sK + static_cast<long long>(k0 + (lane & 15)) * ld + nbase;
    const std::uint32_t addr =
        static_cast<std::uint32_t>(__cvta_generic_to_shared(q));
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0, %1}, [%2];\n"
        : "=r"(b[0]), "=r"(b[1])
        : "r"(addr));
}

__global__ __launch_bounds__(kThreads, PIE_MLA_MMA_MINBLK) void mla_mma_paged_kernel(
    const __nv_bfloat16* __restrict__ q_nope,
    const __nv_bfloat16* __restrict__ q_pe,
    const __nv_bfloat16* __restrict__ ckv_pages,
    const __nv_bfloat16* __restrict__ kpe_pages,
    const std::uint32_t* __restrict__ qo_indptr,
    const std::uint32_t* __restrict__ kv_page_indices,
    const std::uint32_t* __restrict__ kv_page_indptr,
    const std::uint32_t* __restrict__ kv_last_page_lens,
    __nv_bfloat16* __restrict__ o,
    const std::uint8_t* __restrict__ index_mask, int index_mask_stride,
    int R, int H, int page_size, float sm_scale, bool causal)
{
    extern __shared__ __align__(16) char smem_raw[];
    __nv_bfloat16* sQ = reinterpret_cast<__nv_bfloat16*>(smem_raw);
    __nv_bfloat16* sKbuf = sQ + kBM * kLdD;
    __nv_bfloat16* sP = sKbuf + kStages * kBK * kLdD;
    float* sS = reinterpret_cast<float*>(sP + kBM * kLdP);
    float* sM = sS + kBM * kBK;
    float* sL = sM + kBM;
    float* sCorr = sL + kBM;

    const int tid = threadIdx.x;
    const int lane = tid & 31;
    const int warp = tid >> 5;
    // Head groups of the same token are adjacent in the grid so the blocks
    // that share a KV sequence are co-resident: the sequence is then fetched
    // from HBM once and the other groups hit in L2. With the token on x the
    // co-resident blocks are all *different* requests, and at any real batch
    // the union of their KV overflows L2.
    const int t = blockIdx.y;
    const int h0 = blockIdx.x * kBM;

    __shared__ int s_req;
    if (tid == 0) s_req = 0;
    if (tid < kBM) { sM[tid] = -CUDART_INF_F; sL[tid] = 0.f; }
    __syncthreads();
    // One coalesced pass beats a binary search here: the search is ~log2(R)
    // *dependent* global loads on the critical path of every block, and blocks
    // are the unit this kernel has millions of.
    for (int i = tid; i < R; i += kThreads) {
        if (t >= static_cast<int>(qo_indptr[i]) &&
            t < static_cast<int>(qo_indptr[i + 1])) {
            s_req = i;
        }
    }
    __syncthreads();
    const int r = s_req;

    const int qo_lo = static_cast<int>(qo_indptr[r]);
    const int new_tokens = static_cast<int>(qo_indptr[r + 1]) - qo_lo;
    const int pages_first = static_cast<int>(kv_page_indptr[r]);
    const int num_pages = static_cast<int>(kv_page_indptr[r + 1]) - pages_first;
    const int kv_len =
        (num_pages - 1) * page_size + static_cast<int>(kv_last_page_lens[r]);
    const int abs_q = kv_len - new_tokens + (t - qo_lo);
    const int j_end = causal ? (abs_q + 1) : kv_len;
    const std::uint8_t* mrow =
        (index_mask != nullptr)
            ? index_mask + static_cast<long long>(t) * index_mask_stride
            : nullptr;

    constexpr int kChunksPerRow = kD / 8;
    for (int c = tid; c < kBM * kChunksPerRow; c += kThreads) {
        const int row = c / kChunksPerRow;
        const int d = (c % kChunksPerRow) * 8;
        const long long qh = (static_cast<long long>(t) * H + h0 + row);
        const int4 v = (d < kCkv)
            ? *reinterpret_cast<const int4*>(q_nope + qh * kCkv + d)
            : *reinterpret_cast<const int4*>(q_pe + qh * kKpe + (d - kCkv));
        *reinterpret_cast<int4*>(sQ + row * kLdD + d) = v;
    }

    float oacc[kNTiles][4];
    #pragma unroll
    for (int n = 0; n < kNTiles; ++n) {
        #pragma unroll
        for (int i = 0; i < 4; ++i) oacc[n][i] = 0.f;
    }
    const int g = lane >> 2, p = (lane & 3) << 1;
    const int dbase = warp * kDimsPerWarp;

    // Stream KV tiles through `kStages` shared buffers: the cp.async group for
    // tile s+kStages-1 is in flight while the mma pipeline chews on tile s.
    // Resolving a key's page costs an integer divide, an integer modulo and a
    // dependent global load. Done per 16-byte chunk that is 72 times per key,
    // and GPUs have no integer divide, so it dominated the whole kernel: ~90% of
    // the runtime was here, not in the mma pipeline or the KV bandwidth. One
    // lookup per key per tile instead, published through shared memory.
    __shared__ long long s_slot[kBK];
    auto issue_tile = [&](int j0, int stage) {
        for (int rr = tid; rr < kBK; rr += kThreads) {
            const int j = j0 + rr;
            s_slot[rr] =
                (j < j_end)
                    ? (static_cast<long long>(
                           kv_page_indices[pages_first + j / page_size]) *
                           page_size +
                       (j % page_size))
                    : -1;
        }
        __syncthreads();
        __nv_bfloat16* dst_base = sKbuf + stage * (kBK * kLdD);
        for (int c = tid; c < kBK * kChunksPerRow; c += kThreads) {
            const int row = c / kChunksPerRow;
            const int d = (c % kChunksPerRow) * 8;
            const long long slot = s_slot[row];
            __nv_bfloat16* dst = dst_base + row * kLdD + d;
            if (slot >= 0) {
                const void* src =
                    (d < kCkv)
                        ? static_cast<const void*>(ckv_pages + slot * kCkv + d)
                        : static_cast<const void*>(kpe_pages + slot * kKpe +
                                                   (d - kCkv));
                __pipeline_memcpy_async(dst, src, 16);
            } else {
                *reinterpret_cast<int4*>(dst) = make_int4(0, 0, 0, 0);
            }
        }
        __pipeline_commit();
    };

    #pragma unroll
    for (int s = 0; s < kStages - 1; ++s) {
        const int j0 = s * kBK;
        if (j0 < j_end) issue_tile(j0, s);
    }

    int stage = 0;
    for (int j0 = 0; j0 < j_end; j0 += kBK, stage = (stage + 1) % kStages) {
        // Everyone is done reading the buffer the prefetch below reuses.
        __syncthreads();
        const int prefetch_j0 = j0 + (kStages - 1) * kBK;
        if (prefetch_j0 < j_end) {
            issue_tile(prefetch_j0, (stage + kStages - 1) % kStages);
        } else {
            __pipeline_commit();
        }
        __pipeline_wait_prior(kStages - 1);
        __syncthreads();
        const __nv_bfloat16* sK = sKbuf + stage * (kBK * kLdD);

        float sacc[kSNTiles][4];
        #pragma unroll
        for (int n = 0; n < kSNTiles; ++n) {
            #pragma unroll
            for (int i = 0; i < 4; ++i) sacc[n][i] = 0.f;
        }
        {
            // The Q fragment is the expensive operand here (four 32-bit shared
            // loads per k step), so it is loaded once and reused across this
            // warp's key strips.
            const __nv_bfloat16* kbase =
                sK + static_cast<long long>(warp * 8 * kSNTiles) * kLdD;
            #pragma unroll 2
            for (int k0 = 0; k0 < kD; k0 += 16) {
                std::uint32_t a[4];
                ld_a(a, sQ, kLdD, lane, k0);
                #pragma unroll
                for (int n = 0; n < kSNTiles; ++n) {
                    std::uint32_t b[2];
                    ld_b_k(b, kbase + static_cast<long long>(n * 8) * kLdD,
                           kLdD, lane, k0);
                    mma_m16n8k16(sacc[n], a, b);
                }
            }
        }
        #pragma unroll
        for (int n = 0; n < kSNTiles; ++n) {
            const int ncol = (warp * kSNTiles + n) * 8 + p;
            sS[g * kBK + ncol] = sacc[n][0];
            sS[g * kBK + ncol + 1] = sacc[n][1];
            sS[(g + 8) * kBK + ncol] = sacc[n][2];
            sS[(g + 8) * kBK + ncol + 1] = sacc[n][3];
        }
        __syncthreads();

        {
            constexpr int kSubs = kThreads / kBM;
            constexpr int kColsPerSub = kBK / kSubs;
            const int row = tid / kSubs;
            const int sub = tid % kSubs;
            float v[kColsPerSub];
            float local = -CUDART_INF_F;
            #pragma unroll
            for (int i = 0; i < kColsPerSub; ++i) {
                const int jj = sub * kColsPerSub + i;
                const int j = j0 + jj;
                float s = -CUDART_INF_F;
                if (j < j_end &&
                    !(mrow != nullptr && j < index_mask_stride && mrow[j] == 0)) {
                    s = sS[row * kBK + jj] * sm_scale;
                }
                v[i] = s;
                local = fmaxf(local, s);
            }
            #pragma unroll
            for (int m = 1; m < kSubs; m <<= 1) {
                local = fmaxf(local, __shfl_xor_sync(0xffffffffu, local, m));
            }
            const float m_prev = sM[row];
            const float m_new = fmaxf(m_prev, local);
            const bool empty = (m_new == -CUDART_INF_F);
            const float corr =
                empty ? 1.f : ((m_prev == -CUDART_INF_F) ? 0.f : __expf(m_prev - m_new));
            float lsum = 0.f;
            #pragma unroll
            for (int i = 0; i < kColsPerSub; ++i) {
                v[i] = empty ? 0.f : __expf(v[i] - m_new);
                lsum += v[i];
            }
            #pragma unroll
            for (int m = 1; m < kSubs; m <<= 1) {
                lsum += __shfl_xor_sync(0xffffffffu, lsum, m);
            }
            if (sub == 0) {
                sM[row] = m_new;
                sL[row] = sL[row] * corr + lsum;
                sCorr[row] = corr;
            }
            #pragma unroll
            for (int i = 0; i < kColsPerSub; ++i) {
                sP[row * kLdP + sub * kColsPerSub + i] = __float2bfloat16(v[i]);
            }
        }
        __syncthreads();

        const float c0 = sCorr[g], c1 = sCorr[g + 8];
        #pragma unroll
        for (int n = 0; n < kNTiles; ++n) {
            oacc[n][0] *= c0; oacc[n][1] *= c0;
            oacc[n][2] *= c1; oacc[n][3] *= c1;
        }
        #pragma unroll
        for (int k0 = 0; k0 < kBK; k0 += 16) {
            std::uint32_t a[4];
            ld_a(a, sP, kLdP, lane, k0);
            #pragma unroll
            for (int n = 0; n < kNTiles; ++n) {
                std::uint32_t b[2];
                ld_b_v(b, sK, kLdD, lane, k0, dbase + n * 8);
                mma_m16n8k16(oacc[n], a, b);
            }
        }
    }

    __syncthreads();
    const float l0 = sL[g], l1 = sL[g + 8];
    const float i0 = (l0 > 0.f) ? (1.f / l0) : 0.f;
    const float i1 = (l1 > 0.f) ? (1.f / l1) : 0.f;
    __nv_bfloat16* o0 = o + (static_cast<long long>(t) * H + h0 + g) * kCkv;
    __nv_bfloat16* o1 = o + (static_cast<long long>(t) * H + h0 + g + 8) * kCkv;
    #pragma unroll
    for (int n = 0; n < kNTiles; ++n) {
        const int d = dbase + n * 8 + p;
        o0[d] = __float2bfloat16(oacc[n][0] * i0);
        o0[d + 1] = __float2bfloat16(oacc[n][1] * i0);
        o1[d] = __float2bfloat16(oacc[n][2] * i1);
        o1[d + 1] = __float2bfloat16(oacc[n][3] * i1);
    }
}

inline std::size_t smem_bytes() {
    return static_cast<std::size_t>(kBM * kLdD + kStages * kBK * kLdD +
                                    kBM * kLdP) *
               sizeof(__nv_bfloat16) +
           static_cast<std::size_t>(kBM * kBK + 3 * kBM) * sizeof(float);
}

}  // namespace mma_detail

inline bool mla_mma_supported(int kv_lora_rank, int qk_rope_head_dim, int num_heads) {
    static const int forced = [] {
        return 0;
        if (false) return -1;
        if (false) return 1;
        return 0;
    }();
    if (forced < 0) return false;
    return kv_lora_rank == mma_detail::kCkv &&
           qk_rope_head_dim == mma_detail::kKpe &&
           num_heads % mma_detail::kBM == 0;
}

inline void launch_mla_mma_paged_raw(
    const void* q_nope, const void* q_pe,
    const void* ckv_pages, const void* kpe_pages, int page_size, void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    int total_tokens, int num_requests, int num_heads,
    float sm_scale, bool causal, cudaStream_t stream,
    const std::uint8_t* index_mask, int index_mask_stride)
{
    using namespace mma_detail;
    const std::size_t smem = smem_bytes();
    static std::once_flag opt_in;
    std::call_once(opt_in, [&] {
        PIE_MLA_NAIVE_CHECK(cudaFuncSetAttribute(
            reinterpret_cast<const void*>(mla_mma_paged_kernel),
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem)));
    });
    dim3 grid(num_heads / kBM, total_tokens);
    mla_mma_paged_kernel<<<grid, kThreads, smem, stream>>>(
        static_cast<const __nv_bfloat16*>(q_nope),
        static_cast<const __nv_bfloat16*>(q_pe),
        static_cast<const __nv_bfloat16*>(ckv_pages),
        static_cast<const __nv_bfloat16*>(kpe_pages),
        qo_indptr_d, kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d,
        static_cast<__nv_bfloat16*>(o), index_mask, index_mask_stride,
        num_requests, num_heads, page_size, sm_scale, causal);
    PIE_MLA_NAIVE_CHECK(cudaGetLastError());
}

}  // namespace pie_cuda_driver::ops::mla_naive
