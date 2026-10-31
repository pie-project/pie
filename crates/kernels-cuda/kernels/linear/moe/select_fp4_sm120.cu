#include "linear/codec.cuh"
#include "linear/act_quant.cuh"
#include "linear/moe_w4.cuh"

namespace pie::linear {

/// `x`'s rows `[rows, k]` as e4m3 in 1x`kGran` blocks (`E4m3Block`): codes
/// row-major into `codes`, each block's ue8m0 exponent a byte into `exps`
/// once a 32 (`[rows, k / 32]`), as the block-scaled MMA reads it. `kGran /
/// 8` lanes a block; 128 threads, `ceil(rows k / 1024)` blocks.
template <int kGran>
__global__ void __launch_bounds__(128) quantize_rows_mxfp8(
    const bf16* __restrict__ x, u8* __restrict__ codes, u8* __restrict__ exps, int rows, int k)
{
    using Block = E4m3Block<kGran>;
    const int blocks = k / kGran;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int slot = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) / Block::kLanes);
    // Every lane joins the block's shuffles; a slot past the rows reads row 0.
    const bool live = slot < rows * blocks;
    const int sl = live ? slot : 0;
    const long long at = static_cast<long long>(sl / blocks) * k + (sl % blocks) * kGran;
    Block q;
    q.load(x + at, lane);
    if (!live) return;
    *reinterpret_cast<unsigned long long*>(codes + at + (lane % Block::kLanes) * 8) = q.codes();
    const int part = lane % Block::kLanes;
    if (part % 4 == 0) exps[at / 32 + part / 4] = static_cast<u8>(q.e + 127);
}

namespace w4a8 {

// A packed run of eight e2m1 codes (low nibble first) as the two words of
// 8-bit containers the f8f6f4 MMA reads, each code at bits 2..5 of its byte.
__device__ __forceinline__ void spread(unsigned w, unsigned& lo, unsigned& hi) {
    const unsigned even = (w << 2) & 0x3C3C3C3Cu;
    const unsigned odd = (w >> 2) & 0x3C3C3C3Cu;
    lo = __byte_perm(even, odd, 0x5140);
    hi = __byte_perm(even, odd, 0x7362);
}

// d += (A 2^(sa - 127)) . (B 2^(sb - 127)), A 16x32 e2m1, B 32x8 e4m3, one
// ue8m0 scale a row of A and a column of B.
__device__ __forceinline__ void mma(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1,
                                    unsigned sa, unsigned sb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3."
        "f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, "
        "{%13}, {%14, %15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa),
          "h"(static_cast<unsigned short>(0)), "h"(static_cast<unsigned short>(0)), "r"(sb),
          "h"(static_cast<unsigned short>(0)), "h"(static_cast<unsigned short>(0)));
}

__device__ __forceinline__ void cp16(void* dst, const void* from, int bytes) {
    const unsigned saddr = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(saddr), "l"(from), "r"(bytes));
}

__device__ __forceinline__ void cp4(void* dst, const void* from, int bytes) {
    const unsigned saddr = static_cast<unsigned>(__cvta_generic_to_shared(dst));
    asm volatile("cp.async.ca.shared.global [%0], [%1], 4, %2;\n" ::"r"(saddr), "l"(from), "r"(bytes));
}

}  // namespace w4a8

/// `moe_select_mxfp4_mma_pipe` W4A8 on sm_120a's block-scaled mma.sync
/// (`kind::mxf8f6f4`): the routes' rows quantized as MXFP8 by
/// `quantize_rows_mxfp8` (`xq` codes, `xe` a ue8m0 a 32), the bank's e2m1
/// codes against them under both sides' e8m0 scales, as the FlashInfer
/// SM120 CUTLASS MoE runs MXFP4 experts. Eight warps a CTA (256 rows, or 128
/// GLU columns) a 64 routes, `kGroups` of them a CTA over a slice of 64
/// `kGroups` routes, `k` whole 128s: each 128-code chunk's codes, routes and
/// scales staged through one `kStages` cp.async ring.
template <int kGlu, int kStages, int kGroups>
__global__ void __launch_bounds__(256 * kGroups, 1) moe_select_mxfp4_w4a8(
    const u8* __restrict__ xq,
    const u8* __restrict__ xe,
    const i32* __restrict__ order,
    const i32* __restrict__ offsets,
    const i32* __restrict__ work,
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    const bf16* __restrict__ bias,
    bf16* __restrict__ out,
    int act_div,
    int n,
    int k,
    int num_experts,
    int inter,
    float limit,
    float galpha)
{
    constexpr int kNb = 8;
    constexpr int kTiles = 2;
    constexpr int kWarps = 8;
    constexpr int kRowsWarp = kGlu != 0 ? 16 : kTiles * 16;
    constexpr int kRowsCta = kWarps * kRowsWarp;
    constexpr int kSlice = 8 * kNb * kGroups;
    // A stage: the slice's 128 codes a route, each 16-byte run at its index
    // xor the route's (so a warp's 8-byte reads of eight routes fall on
    // distinct banks), a route's four scales, then the CTA's 256 rows' 64
    // code bytes, each run at its index xor half the row's (the same for the
    // 4-byte reads). The rows' scales go to registers a chunk ahead.
    constexpr int kXStride = 128;
    constexpr int kWStride = 64;
    constexpr int kX = kSlice * kXStride;
    constexpr int kXs = kSlice * 4;
    constexpr int kW = kWarps * 32 * kWStride;
    constexpr int kStage = kX + kXs + kW;

    const int at = static_cast<int>(act_div == 1 ? blockIdx.y : blockIdx.x);
    if (at >= offsets[num_experts + 1]) return;
    const int item = work[at];
    const int expert = item >> 16;
    const int slice = item & 0xFFFF;
    const int begin = offsets[expert] + slice * kSlice;
    const int count = min(offsets[expert + 1] - begin, kSlice);
    if (count <= 0) return;

    const int tid = static_cast<int>(threadIdx.x);
    const int warp = (tid >> 5) % kWarps;
    const int group = tid / 256;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int c = lane & 3;
    const int cta_row = static_cast<int>(act_div == 1 ? blockIdx.x : blockIdx.y) * kRowsCta;
    const int row_base = cta_row + warp * kRowsWarp;
    const int span = kGlu != 0 ? inter : n;
    const bool rows_live = row_base < span;

    const int chunks = k >> 7;
    const long long row_bytes = k >> 1;
    const int sf_row = k >> 5;
    const u8* w_e = codes + static_cast<long long>(expert) * n * row_bytes;
    const u8* s_e = scales + static_cast<long long>(expert) * n * sf_row;

    // The stage row this thread copies: warp tid / 32's tile (tid / 16) % 2,
    // half (tid / 8) % 2, row tid % 8, as the warp's lanes read it.
    const int own = tid % 256;
    const int own_half = tid / 256;
    const int own_base = cta_row + (own >> 5) * kRowsWarp;
    const int own_t = (own >> 4) & 1;
    const int own_tile = kGlu != 0 ? own_base + own_t * inter : own_base + 16 * own_t;
    const int own_row = min(max(own_tile, 0) + (own & 15), n - 1);
    // The slice routes this thread copies a quarter of: route tid / 4 (and
    // past them, tid / 4 + 64 kGroups, with fewer threads than 4 a route).
    const int xr = tid >> 2;
    const int xq4 = tid & 3;
    const int xsrc = xr < count ? order[begin + xr] / act_div : -1;
    // Each tile's row whose scale this lane hands the MMA (g at even c, g + 8
    // at odd), its four a chunk.
    const u8* srow[kTiles];
#pragma unroll
    for (int t = 0; t < kTiles; ++t) {
        const int tile = kGlu != 0 ? row_base + t * inter : row_base + 16 * t;
        srow[t] = s_e + static_cast<long long>(min(max(tile, 0) + 8 * (c & 1) + g, n - 1)) * sf_row;
    }
    auto load_s = [&](int ch, unsigned (&s)[kTiles]) {
#pragma unroll
        for (int t = 0; t < kTiles; ++t) s[t] = __ldg(reinterpret_cast<const unsigned*>(srow[t] + ch * 4));
    };

    extern __shared__ __align__(16) unsigned char w4a8_smem[];
    auto stage = [&](int ch) {
        if (ch < chunks) {
            unsigned char* st = w4a8_smem + (ch % kStages) * kStage;
            const bool live = xsrc >= 0;
            const u8* xf = xq + static_cast<long long>(live ? xsrc : 0) * k + ch * 128 + xq4 * 16;
            w4a8::cp16(st + xr * kXStride + ((xq4 ^ (xr & 7)) * 16), xf, live ? 16 : 0);
            w4a8::cp16(st + xr * kXStride + (((4 + xq4) ^ (xr & 7)) * 16), xf + 64, live ? 16 : 0);
            if (xq4 == 0) {
                w4a8::cp4(st + kX + xr * 4, xe + static_cast<long long>(live ? xsrc : 0) * sf_row + ch * 4,
                          live ? 4 : 0);
            }
            unsigned char* ws = st + kX + kXs;
            const u8* wf = w_e + own_row * row_bytes + ch * 64;
#pragma unroll
            for (int q = own_half * 4 / kGroups; q < (own_half + 1) * 4 / kGroups; ++q)
                w4a8::cp16(ws + own * kWStride + ((q ^ ((own >> 1) & 3)) * 16), wf + q * 16, 16);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };

    float acc[kTiles][kNb][4];
#pragma unroll
    for (int t = 0; t < kTiles; ++t)
#pragma unroll
        for (int b = 0; b < kNb; ++b)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[t][b][e] = 0.f;

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) stage(s);

    unsigned cs[kTiles], ns[kTiles];
    load_s(0, cs);
    for (int ch = 0; ch < chunks; ++ch) {
        if (ch + 1 < chunks) load_s(ch + 1, ns);
        asm volatile("cp.async.wait_group %0;\n" ::"n"(kStages - 2));
        __syncthreads();
        // The ring's oldest stage, every warp past its MMAs.
        stage(ch + kStages - 1);
        if (rows_live) {
            const unsigned char* st = w4a8_smem + (ch % kStages) * kStage;
            const unsigned* xs = reinterpret_cast<const unsigned*>(st + kX);
            const unsigned char* ws = st + kX + kXs;
            // A row's scale from the lanes of its parity (row g at even c,
            // g + 8 at odd), a route's from its group's lanes.
            unsigned sa[kTiles], sb[kNb];
#pragma unroll
            for (int t = 0; t < kTiles; ++t) sa[t] = cs[t];
#pragma unroll
            for (int b = 0; b < kNb; ++b) sb[b] = xs[64 * group + 8 * b + g];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                unsigned a[kTiles][4];
#pragma unroll
                for (int t = 0; t < kTiles; ++t)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int r = warp * 32 + t * 16 + h * 8 + g;
                        const unsigned w = *reinterpret_cast<const unsigned*>(
                            ws + r * kWStride + ((j ^ ((r >> 1) & 3)) * 16) + c * 4);
                        w4a8::spread(w, a[t][h], a[t][2 + h]);
                    }
#pragma unroll
                for (int b = 0; b < kNb; ++b) {
                    const int xrow = 64 * group + 8 * b + g;
                    const uint2 bv = *reinterpret_cast<const uint2*>(
                        st + xrow * kXStride + (((2 * j + (c >> 1)) ^ (xrow & 7)) * 16) + 8 * (c & 1));
                    const unsigned sbj = (sb[b] >> (8 * j)) & 0xFFu;
#pragma unroll
                    for (int t = 0; t < kTiles; ++t)
                        w4a8::mma(acc[t][b], a[t], bv.x, bv.y, (sa[t] >> (8 * j)) & 0xFFu, sbj);
                }
            }
        }
#pragma unroll
        for (int t = 0; t < kTiles; ++t) cs[t] = ns[t];
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
    if (!rows_live || count <= 64 * group) return;

    const bf16* b_e = bias != nullptr ? bias + static_cast<long long>(expert) * n : nullptr;
    auto factor = [](int) { return 1.f; };
    w4mma::w4_land<kGlu, kTiles, kNb>(acc, factor, b_e, order, begin + 64 * group, count - 64 * group, out, n, inter,
                                      row_base, g, c, limit, galpha);
}

}  // namespace pie::linear

namespace pie::linear {

/// `quantize_rows_mxfp8` into route order: position `p` of `order` (each
/// expert's routes contiguous) takes row `order[p] / act_div` of `x`, so an
/// expert's slice is one box of rows for TMA. Positions past the fire's
/// routes (`offsets[num_experts]`) are left. `kSplit`: a 32's codes 8c..8c+3
/// at 4c and 8c+4..8c+7 at 16 + 4c, as `moe_select_mxfp4_w4a8_ws`'s
/// ldmatrix hands lane c its B words, and the exponents a 128-chunk's word
/// a route, chunk-major (`[k / 128][routes]`), an 8-route block's one sector.
template <int kGran, bool kSplit = false>
__global__ void __launch_bounds__(128) quantize_rows_mxfp8_routed(
    const bf16* __restrict__ x, const i32* __restrict__ order, const i32* __restrict__ offsets,
    u8* __restrict__ codes, u8* __restrict__ exps, int rows, int k, int act_div, int num_experts)
{
    using Block = E4m3Block<kGran>;
    const int blocks = k / kGran;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int slot = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) / Block::kLanes);
    const int total = min(offsets[num_experts], rows);
    const bool live = slot < total * blocks;
    const int sl = live ? slot : 0;
    const int p = sl / blocks;
    const int src = live ? order[p] / act_div : 0;
    Block q;
    q.load(x + static_cast<long long>(src) * k + (sl % blocks) * kGran, lane);
    if (!live) return;
    const long long at = static_cast<long long>(p) * k + (sl % blocks) * kGran;
    if constexpr (kSplit) {
        const int q8 = (lane % Block::kLanes) * 8;
        const unsigned long long w = q.codes();
        u8* to = codes + at + (q8 & ~31) + ((q8 >> 3) & 3) * 4;
        *reinterpret_cast<unsigned*>(to) = static_cast<unsigned>(w);
        *reinterpret_cast<unsigned*>(to + 16) = static_cast<unsigned>(w >> 32);
    } else {
        *reinterpret_cast<unsigned long long*>(codes + at + (lane % Block::kLanes) * 8) = q.codes();
    }
    const int part = lane % Block::kLanes;
    if (part % 4 != 0) return;
    if constexpr (kSplit) {
        const int b32 = (sl % blocks) * (kGran / 32) + part / 4;
        exps[(static_cast<long long>(b32 >> 2) * total + p) * 4 + (b32 & 3)] = static_cast<u8>(q.e + 127);
    } else {
        exps[at / 32 + part / 4] = static_cast<u8>(q.e + 127);
    }
}

namespace w4a8 {

// CUtensorMap: 128 opaque bytes the host encodes, passed __grid_constant__.
struct alignas(64) TmaMap {
    unsigned long long raw[16];
};

__device__ __forceinline__ unsigned saddr(const void* p) {
    return static_cast<unsigned>(__cvta_generic_to_shared(p));
}

__device__ __forceinline__ void bar_init(unsigned long long* bar, unsigned count) {
    asm volatile("mbarrier.init.shared::cta.b64 [%0], %1;\n" ::"r"(saddr(bar)), "r"(count));
}

__device__ __forceinline__ void bar_expect(unsigned long long* bar, unsigned bytes) {
    asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;\n" ::"r"(saddr(bar)), "r"(bytes)
                 : "memory");
}

__device__ __forceinline__ void bar_wait(unsigned long long* bar, unsigned parity) {
    unsigned done = 0;
    do {
        asm volatile(
            "{\n .reg .pred p;\n mbarrier.try_wait.parity.shared::cta.b64 p, [%1], %2;\n"
            " selp.u32 %0, 1, 0, p;\n}\n"
            : "=r"(done)
            : "r"(saddr(bar)), "r"(parity)
            : "memory");
    } while (done == 0);
}

// A 2D box at (x = inner, y = outer) into shared memory, completing `bar`.
__device__ __forceinline__ void tma_2d(void* dst, unsigned long long* bar, const TmaMap* map, int x, int y) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cluster.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4}], [%2];\n" ::"r"(saddr(dst)),
        "l"(reinterpret_cast<unsigned long long>(map)), "r"(saddr(bar)), "r"(x), "r"(y)
        : "memory");
}

}  // namespace w4a8

/// `moe_select_mxfp4_w4a8` with each stage landed by TMA: the slice's routes
/// as one 128-row box of `map_x` (rows quantized in route order by
/// `quantize_rows_mxfp8_routed`, 128B-swizzled), the CTA's 256 rows (or a
/// GLU's 128 gate and 128 up) as two 128-row boxes of `map_w` (the bank's
/// codes, 64B-swizzled), one thread issuing them a stage ahead onto that
/// stage's mbarrier; the reads are `moe_select_mxfp4_w4a8`'s.
template <int kGlu, int kStages>
__global__ void __launch_bounds__(512, 1) moe_select_mxfp4_w4a8_tma(
    const __grid_constant__ w4a8::TmaMap map_x,
    const __grid_constant__ w4a8::TmaMap map_w,
    const u8* __restrict__ xe,
    const i32* __restrict__ order,
    const i32* __restrict__ offsets,
    const i32* __restrict__ work,
    const u8* __restrict__ scales,
    const bf16* __restrict__ bias,
    bf16* __restrict__ out,
    int act_div,
    int n,
    int k,
    int num_experts,
    int inter,
    float limit,
    float galpha)
{
    constexpr int kNb = 8;
    constexpr int kTiles = 2;
    constexpr int kWarps = 8;
    constexpr int kGroups = 2;
    constexpr int kRowsWarp = kGlu != 0 ? 16 : kTiles * 16;
    constexpr int kRowsCta = kWarps * kRowsWarp;
    constexpr int kSlice = 8 * kNb * kGroups;
    constexpr int kX = kSlice * 128;
    constexpr int kW = 256 * 64;
    constexpr int kStage = kX + kW;

    // Row tiles fastest: a slice's rows, and an expert's codes, shared in L2.
    const int at = static_cast<int>(blockIdx.y);
    if (at >= offsets[num_experts + 1]) return;
    const int item = work[at];
    const int expert = item >> 16;
    const int slice = item & 0xFFFF;
    const int begin = offsets[expert] + slice * kSlice;
    const int count = min(offsets[expert + 1] - begin, kSlice);
    if (count <= 0) return;

    const int tid = static_cast<int>(threadIdx.x);
    const int warp = (tid >> 5) % kWarps;
    const int group = tid / 256;
    const int lane = tid & 31;
    const int g = lane >> 2;
    const int c = lane & 3;
    const int cta_row = static_cast<int>(blockIdx.x) * kRowsCta;
    const int row_base = cta_row + warp * kRowsWarp;
    const int span = kGlu != 0 ? inter : n;
    const bool rows_live = row_base < span;

    const int chunks = k >> 7;
    const int sf_row = k >> 5;
    const u8* s_e = scales + static_cast<long long>(expert) * n * sf_row;
    // The bank rows of the two boxes: 256 in order, or the GLU's gate and up.
    const int w_y0 = expert * n + cta_row;
    const int w_y1 = kGlu != 0 ? expert * n + inter + cta_row : w_y0 + 128;

    // Stages 1024-aligned for the 128B swizzle; the barriers past them.
    extern __shared__ __align__(1024) unsigned char w4a8_tma_smem[];
    unsigned char* ring = reinterpret_cast<unsigned char*>(
        (reinterpret_cast<unsigned long long>(w4a8_tma_smem) + 1023) & ~1023ull);
    unsigned long long* full = reinterpret_cast<unsigned long long*>(ring + kStages * kStage);
    unsigned long long* empty = full + kStages;
    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < kStages; ++s) {
            w4a8::bar_init(full + s, 1);
            w4a8::bar_init(empty + s, kWarps * kGroups);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();
    auto stage = [&](int ch) {
        if (tid == 0 && ch < chunks) {
            const int s = ch % kStages;
            unsigned char* st = ring + s * kStage;
            asm volatile("fence.proxy.async.shared::cta;\n" ::: "memory");
            w4a8::bar_expect(full + s, kStage);
            w4a8::tma_2d(st, full + s, &map_x, ch * 128, begin);
            w4a8::tma_2d(st + kX, full + s, &map_w, ch * 64, w_y0);
            w4a8::tma_2d(st + kX + kW / 2, full + s, &map_w, ch * 64, w_y1);
        }
    };

    // Each tile's row whose scale this lane hands the MMA (g at even c, g + 8
    // at odd), and each 8-route block's route of this lane: four a chunk.
    const u8* srow[kTiles];
#pragma unroll
    for (int t = 0; t < kTiles; ++t) {
        const int tile = kGlu != 0 ? row_base + t * inter : row_base + 16 * t;
        srow[t] = s_e + static_cast<long long>(min(max(tile, 0) + 8 * (c & 1) + g, n - 1)) * sf_row;
    }
    const int last = offsets[num_experts] - 1;
    const u8* erow[kNb];
#pragma unroll
    for (int b = 0; b < kNb; ++b)
        erow[b] = xe + static_cast<long long>(min(begin + 8 * kNb * group + 8 * b + g, last)) * sf_row;
    auto load_s = [&](int ch, unsigned (&sa)[kTiles], unsigned (&sb)[kNb]) {
#pragma unroll
        for (int t = 0; t < kTiles; ++t) sa[t] = __ldg(reinterpret_cast<const unsigned*>(srow[t] + ch * 4));
#pragma unroll
        for (int b = 0; b < kNb; ++b) sb[b] = __ldg(reinterpret_cast<const unsigned*>(erow[b] + ch * 4));
    };
    // A lane's stage rows: tile t's half h, in the boxes' order.
    auto local_row = [&](int t, int h) {
        return kGlu != 0 ? t * 128 + warp * 16 + h * 8 + g : warp * 32 + t * 16 + h * 8 + g;
    };

    float acc[kTiles][kNb][4];
#pragma unroll
    for (int t = 0; t < kTiles; ++t)
#pragma unroll
        for (int b = 0; b < kNb; ++b)
#pragma unroll
            for (int e = 0; e < 4; ++e) acc[t][b][e] = 0.f;

#pragma unroll
    for (int s = 0; s < kStages - 1; ++s) stage(s);

    unsigned sa[kTiles], sb[kNb], na[kTiles], nb[kNb];
    load_s(0, sa, sb);
    for (int ch = 0; ch < chunks; ++ch) {
        if (ch + 1 < chunks) load_s(ch + 1, na, nb);
        // Every warp past the chunk before: its stage takes the next fill.
        if (tid == 0 && ch > 0 && ch + kStages - 1 < chunks)
            w4a8::bar_wait(empty + (ch - 1) % kStages, ((ch - 1) / kStages) & 1);
        stage(ch + kStages - 1);
        w4a8::bar_wait(full + ch % kStages, (ch / kStages) & 1);
        if (rows_live) {
            const unsigned char* st = ring + (ch % kStages) * kStage;
            const unsigned char* ws = st + kX;
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                unsigned a[kTiles][4];
#pragma unroll
                for (int t = 0; t < kTiles; ++t)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const int r = local_row(t, h);
                        const unsigned w = *reinterpret_cast<const unsigned*>(
                            ws + r * 64 + ((j ^ ((r >> 1) & 3)) * 16) + c * 4);
                        w4a8::spread(w, a[t][h], a[t][2 + h]);
                    }
#pragma unroll
                for (int b = 0; b < kNb; ++b) {
                    const int xrow = 8 * kNb * group + 8 * b + g;
                    const uint2 bv = *reinterpret_cast<const uint2*>(
                        st + xrow * 128 + (((2 * j + (c >> 1)) ^ (xrow & 7)) * 16) + 8 * (c & 1));
                    const unsigned sbj = (sb[b] >> (8 * j)) & 0xFFu;
#pragma unroll
                    for (int t = 0; t < kTiles; ++t)
                        w4a8::mma(acc[t][b], a[t], bv.x, bv.y, (sa[t] >> (8 * j)) & 0xFFu, sbj);
                }
            }
        }
        __syncwarp();
        if (lane == 0) asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(w4a8::saddr(empty + ch % kStages)) : "memory");
#pragma unroll
        for (int t = 0; t < kTiles; ++t) sa[t] = na[t];
#pragma unroll
        for (int b = 0; b < kNb; ++b) sb[b] = nb[b];
    }
    if (!rows_live || count <= 8 * kNb * group) return;

    const bf16* b_e = bias != nullptr ? bias + static_cast<long long>(expert) * n : nullptr;
    auto factor = [](int) { return 1.f; };
    w4mma::w4_land<kGlu, kTiles, kNb>(acc, factor, b_e, order, begin + 8 * kNb * group,
                                      count - 8 * kNb * group, out, n, inter, row_base, g, c, limit, galpha);
}

}  // namespace pie::linear

namespace pie::linear {

namespace w4a8 {

// Four 8x8 b16 matrices of shared memory: lane l's address the row l % 8 of
// matrix l / 8, a lane's word i its quad's row of matrix i at 4 (lane % 4).
__device__ __forceinline__ void ldsm4(unsigned (&r)[4], unsigned addr) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(addr));
}

// `mma` under byte `kJ` of each side's scale word.
template <int kJ>
__device__ __forceinline__ void mma_at(float (&d)[4], const unsigned (&a)[4], unsigned b0, unsigned b1,
                                       unsigned sa, unsigned sb) {
    asm volatile(
        "mma.sync.aligned.kind::mxf8f6f4.block_scale.scale_vec::1X.m16n8k32.row.col.f32.e2m1.e4m3."
        "f32.ue8m0 {%0, %1, %2, %3}, {%4, %5, %6, %7}, {%8, %9}, {%0, %1, %2, %3}, {%10}, {%11, %12}, "
        "{%13}, {%14, %15};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1), "r"(sa), "n"(kJ), "n"(0), "r"(sb),
          "n"(kJ), "n"(0));
}

// `tma_2d` into this CTA's shared memory (no cluster-address path).
__device__ __forceinline__ void tma_2d_cta(void* dst, unsigned long long* bar, const TmaMap* map, int x, int y) {
    asm volatile(
        "cp.async.bulk.tensor.2d.shared::cta.global.mbarrier::complete_tx::bytes"
        " [%0], [%1, {%3, %4}], [%2];\n" ::"r"(saddr(dst)),
        "l"(reinterpret_cast<unsigned long long>(map)), "r"(saddr(bar)), "r"(x), "r"(y)
        : "memory");
}

template <int kV>
struct Const {
    static constexpr int value = kV;
};

}  // namespace w4a8

/// `moe_select_mxfp4_w4a8_tma` warp-specialized and persistent: warp 8's
/// lane (warpgroup 2, 40 registers a thread) issues each stage's boxes once the eight MMA warps have released it,
/// a CTA a SM walking the (row tile, slice) jobs, row tiles fastest. An MMA
/// warp holds 64 rows x 64 routes, both read by ldmatrix: the routes'
/// codes as `quantize_rows_mxfp8_routed<kGran, true>` lays them (a 32's
/// codes 8c..8c+3 at 4c, 8c+4..8c+7 at 16 + 4c), each 8-route block's B
/// words one matrix row; the scales a byte of a chunk's word to each MMA.
template <int kStages>
__global__ void __launch_bounds__(384, 1) moe_select_mxfp4_w4a8_ws(
    const __grid_constant__ w4a8::TmaMap map_x,
    const __grid_constant__ w4a8::TmaMap map_w,
    const u8* __restrict__ xe,
    const i32* __restrict__ order,
    const i32* __restrict__ offsets,
    const i32* __restrict__ work,
    const u8* __restrict__ scales,
    const bf16* __restrict__ bias,
    bf16* __restrict__ out,
    int act_div,
    int n,
    int k,
    int num_experts,
    int inter,
    float limit,
    float galpha)
{
    constexpr int kNb = 8;
    constexpr int kTiles = 4;
    constexpr int kX = 128 * 128;
    constexpr int kW = 256 * 64;
    constexpr int kStage = kX + kW;

    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
    const int lane = tid & 31;
    const int tiles = (n + 255) / 256;
    const int jobs = tiles * offsets[num_experts + 1];
    const int chunks = k >> 7;

    extern __shared__ __align__(1024) unsigned char w4a8_ws_smem[];
    unsigned char* ring = reinterpret_cast<unsigned char*>(
        (reinterpret_cast<unsigned long long>(w4a8_ws_smem) + 1023) & ~1023ull);
    unsigned long long* full = reinterpret_cast<unsigned long long*>(ring + kStages * kStage);
    unsigned long long* empty = full + kStages;
    if (tid == 0) {
#pragma unroll
        for (int s = 0; s < kStages; ++s) {
            w4a8::bar_init(full + s, 1);
            w4a8::bar_init(empty + s, 8);
        }
        asm volatile("fence.mbarrier_init.release.cluster;\n" ::: "memory");
    }
    __syncthreads();

    // Warpgroup 2 the producer, its registers to the MMA warpgroups.
    if (warp >= 8) {
        asm volatile("setmaxnreg.dec.sync.aligned.u32 40;\n" ::: "memory");
        if (warp != 8 || lane != 0) return;
        int s = 0;
        unsigned ph = 0;
        for (int job = static_cast<int>(blockIdx.x); job < jobs; job += static_cast<int>(gridDim.x)) {
            const int item = work[job / tiles];
            const int expert = item >> 16;
            const int begin = offsets[expert] + (item & 0xFFFF) * 128;
            if (offsets[expert + 1] - begin <= 0) continue;
            const int w_y = expert * n + (job % tiles) * 256;
            for (int ch = 0; ch < chunks; ++ch) {
                w4a8::bar_wait(empty + s, ph ^ 1);
                unsigned char* st = ring + s * kStage;
                w4a8::bar_expect(full + s, kStage);
                w4a8::tma_2d_cta(st, full + s, &map_x, ch * 128, begin);
                w4a8::tma_2d_cta(st + kX, full + s, &map_w, ch * 64, w_y);
                w4a8::tma_2d_cta(st + kX + kW / 2, full + s, &map_w, ch * 64, w_y + 128);
                if (++s == kStages) {
                    s = 0;
                    ph ^= 1;
                }
            }
        }
        return;
    }

    asm volatile("setmaxnreg.inc.sync.aligned.u32 232;\n" ::: "memory");
    // Warp (wr, wc): rows 64 wr, routes 64 wc of the CTA's 256 x 128.
    const int wr = warp & 3;
    const int wc = warp >> 2;
    const int g = lane >> 2;
    const int c = lane & 3;
    const int sf_row = k >> 5;
    const int last = offsets[num_experts] - 1;
    // ldmatrix rows: A matrix m is tile m / 2's rows 8 (m % 2).., B matrix m
    // block m / 2's half m % 2 (the 64B and 128B swizzles off the rows).
    const int m = lane >> 3;
    const unsigned ring_s = w4a8::saddr(ring);
    const unsigned a_at = kX + (wr * 64 + 16 * (m >> 1) + 8 * (m & 1) + (lane & 7)) * 64;
    const int a_sw = (lane >> 1) & 3;
    const unsigned b_at = (wc * 64 + 8 * (m >> 1) + (lane & 7)) * 128;
    const int b_sw = lane & 7;
    const int b_h = m & 1;

    float acc[kTiles][kNb][4];
    unsigned sa[kTiles], sb[kNb], nb[kNb];
    uint2 qa[kTiles], qn[kTiles];
    int s = 0;
    unsigned ph = 0;
    for (int job = static_cast<int>(blockIdx.x); job < jobs; job += static_cast<int>(gridDim.x)) {
        const int item = work[job / tiles];
        const int expert = item >> 16;
        const int begin = offsets[expert] + (item & 0xFFFF) * 128;
        const int count = min(offsets[expert + 1] - begin, 128);
        if (count <= 0) continue;
        const int cta_row = (job % tiles) * 256;
        const int row_base = cta_row + wr * 64;
        const int live = row_base < n ? min(max(count - wc * 64, 0), 64) : 0;
        const u8* s_e = scales + static_cast<long long>(expert) * n * sf_row;
        // The rows' scales four chunks a load: lane c's row (g, or g + 8 at
        // odd c) at chunks 4q + 2 (c / 2).., a chunk's word shuffled from
        // its lane; the routes' a chunk (chunk-major, a block one sector).
        auto load_a = [&](int q, uint2 (&ta)[kTiles]) {
            const int from = 4 * q + 2 * (c >> 1);
#pragma unroll
            for (int t = 0; t < kTiles; ++t) {
                const int row = min(row_base + 16 * t + 8 * (c & 1) + g, n - 1);
                ta[t] = from < chunks ? __ldg(reinterpret_cast<const uint2*>(
                                            s_e + static_cast<long long>(row) * sf_row + from * 4))
                                      : make_uint2(0u, 0u);
            }
        };
        auto load_b = [&](int ch, unsigned (&tb)[kNb]) {
#pragma unroll
            for (int b = 0; b < kNb; ++b) {
                const int route = min(begin + wc * 64 + 8 * b + g, last);
                tb[b] = __ldg(reinterpret_cast<const unsigned*>(xe) + static_cast<long long>(ch) * (last + 1) + route);
            }
        };
#pragma unroll
        for (int t = 0; t < kTiles; ++t)
#pragma unroll
            for (int b = 0; b < kNb; ++b)
#pragma unroll
                for (int e = 0; e < 4; ++e) acc[t][b][e] = 0.f;
        // The warp's 64 routes' rows for the landing, lane l's 2l and 2l + 1.
        const int own0 = 2 * lane < live ? order[begin + wc * 64 + 2 * lane] : 0;
        const int own1 = 2 * lane + 1 < live ? order[begin + wc * 64 + 2 * lane + 1] : 0;
        if (live > 0) {
            load_a(0, qa);
            load_b(0, sb);
        }
        // A 32 of the chunk: each tile pair's A words and block pair's B
        // words one ldmatrix; blocks past the warp's routes left.
        auto step = [&](unsigned st, auto partial, auto jc) {
            constexpr bool kPart = decltype(partial)::value != 0;
            constexpr int j = decltype(jc)::value;
            unsigned a[kTiles][4];
#pragma unroll
            for (int tp = 0; tp < 2; ++tp) {
                unsigned r[4];
                w4a8::ldsm4(r, st + a_at + tp * 2048 + ((j ^ a_sw) << 4));
                w4a8::spread(r[0], a[2 * tp][0], a[2 * tp][2]);
                w4a8::spread(r[1], a[2 * tp][1], a[2 * tp][3]);
                w4a8::spread(r[2], a[2 * tp + 1][0], a[2 * tp + 1][2]);
                w4a8::spread(r[3], a[2 * tp + 1][1], a[2 * tp + 1][3]);
            }
#pragma unroll
            for (int bp = 0; bp < kNb / 2; ++bp) {
                if (kPart && 16 * bp >= live) continue;
                unsigned r[4];
                w4a8::ldsm4(r, st + b_at + bp * 2048 + (((2 * j + b_h) ^ b_sw) << 4));
#pragma unroll
                for (int t = 0; t < kTiles; ++t)
                    w4a8::mma_at<j>(acc[t][2 * bp], a[t], r[0], r[1], sa[t], sb[2 * bp]);
                if (kPart && 16 * bp + 8 >= live) continue;
#pragma unroll
                for (int t = 0; t < kTiles; ++t)
                    w4a8::mma_at<j>(acc[t][2 * bp + 1], a[t], r[2], r[3], sa[t], sb[2 * bp + 1]);
            }
        };
        auto chunk = [&](unsigned st, auto partial) {
            step(st, partial, w4a8::Const<0>{});
            step(st, partial, w4a8::Const<1>{});
            step(st, partial, w4a8::Const<2>{});
            step(st, partial, w4a8::Const<3>{});
        };
        for (int ch = 0; ch < chunks; ++ch) {
            if (live > 0) {
                if (ch + 1 < chunks) load_b(ch + 1, nb);
                if ((ch & 3) == 0 && ch + 4 < chunks) load_a((ch >> 2) + 1, qn);
                const int i = ch & 3;
                const int src = (lane & ~3) | (c & 1) | (i & 2);
#pragma unroll
                for (int t = 0; t < kTiles; ++t) sa[t] = __shfl_sync(0xffffffffu, (i & 1) ? qa[t].y : qa[t].x, src);
            }
            w4a8::bar_wait(full + s, ph);
            const unsigned st = ring_s + s * kStage;
            if (live == 64)
                chunk(st, w4a8::Const<0>{});
            else if (live > 0)
                chunk(st, w4a8::Const<1>{});
            __syncwarp();
            if (lane == 0)
                asm volatile("mbarrier.arrive.shared::cta.b64 _, [%0];\n" ::"r"(w4a8::saddr(empty + s)) : "memory");
            if (++s == kStages) {
                s = 0;
                ph ^= 1;
            }
#pragma unroll
            for (int b = 0; b < kNb; ++b) sb[b] = nb[b];
            if ((ch & 3) == 3)
#pragma unroll
                for (int t = 0; t < kTiles; ++t) qa[t] = qn[t];
        }
        if (live <= 0) continue;
        // Land each 8-route block's routes 2c, 2c + 1 (a bf16 pair a row
        // 8m + g) transposed over the quad's eight g: lane g then holds
        // rows 8g..8g+7 of both, a 16-byte store each.
        const bf16* b_e = bias != nullptr ? bias + static_cast<long long>(expert) * n : nullptr;
        const int rows8 = row_base + 8 * g;
#pragma unroll
        for (int b = 0; b < kNb; ++b) {
            if (8 * b >= live) continue;
            unsigned w[8];
#pragma unroll
            for (int mm = 0; mm < 8; ++mm) {
                const float add =
                    b_e != nullptr ? bf16_to_f32(b_e[min(row_base + 8 * mm + g, n - 1)]) : 0.f;
                w[mm] = pack_bf16x2(acc[mm >> 1][b][2 * (mm & 1)] + add, acc[mm >> 1][b][2 * (mm & 1) + 1] + add);
            }
#pragma unroll
            for (int kb = 4; kb > 0; kb >>= 1) {
                const bool up = (g & kb) != 0;
#pragma unroll
                for (int mm = 0; mm < 8; ++mm) {
                    if ((mm & kb) != 0) continue;
                    const unsigned x = __shfl_xor_sync(0xffffffffu, up ? w[mm] : w[mm | kb], kb << 2);
                    if (up)
                        w[mm] = x;
                    else
                        w[mm | kb] = x;
                }
            }
            const int tok = 8 * b + 2 * c;
            const long long route0 = __shfl_sync(0xffffffffu, own0, 4 * b + c);
            const long long route1 = __shfl_sync(0xffffffffu, own1, 4 * b + c);
            if (rows8 >= n) continue;
            if (tok < live) {
                *reinterpret_cast<uint4*>(out + route0 * n + rows8) =
                    make_uint4(__byte_perm(w[0], w[1], 0x5410), __byte_perm(w[2], w[3], 0x5410),
                               __byte_perm(w[4], w[5], 0x5410), __byte_perm(w[6], w[7], 0x5410));
            }
            if (tok + 1 < live) {
                *reinterpret_cast<uint4*>(out + route1 * n + rows8) =
                    make_uint4(__byte_perm(w[0], w[1], 0x7632), __byte_perm(w[2], w[3], 0x7632),
                               __byte_perm(w[4], w[5], 0x7632), __byte_perm(w[6], w[7], 0x7632));
            }
        }
    }
}

}  // namespace pie::linear

namespace pie::linear {

/// Each live route's position in `order` (`slot[order[p]] = p`); routes
/// past the live ones keep what the scratch held.
__global__ void __launch_bounds__(256) moe_route_slots(
    const i32* __restrict__ order, const i32* __restrict__ offsets, i32* __restrict__ slot, int num_experts)
{
    const int p = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (p < offsets[num_experts + 2]) slot[order[p]] = p;
}

/// `quantize_rows_mxfp8_routed<kGran, true>` a token at a time: each of
/// `x`'s `rows` tokens quantized once, its codes and exponents landed at
/// the positions of its `top` routes (`slot`, kept only where `order`
/// agrees and the route is selected).
template <int kGran>
__global__ void __launch_bounds__(128) quantize_rows_mxfp8_scatter(
    const bf16* __restrict__ x, const i32* __restrict__ order, const i32* __restrict__ offsets,
    const i32* __restrict__ slot, u8* __restrict__ codes, u8* __restrict__ exps, int rows, int k, int top,
    int num_experts)
{
    using Block = E4m3Block<kGran>;
    const int blocks = k / kGran;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int at = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) / Block::kLanes);
    const bool live = at < rows * blocks;
    const int sl = live ? at : 0;
    const int t = sl / blocks;
    Block q;
    q.load(x + static_cast<long long>(t) * k + (sl % blocks) * kGran, lane);
    if (!live) return;
    const int total = offsets[num_experts];
    const unsigned long long w = q.codes();
    const int part = lane % Block::kLanes;
    const int q8 = part * 8;
    const int in_row = (sl % blocks) * kGran + (q8 & ~31) + ((q8 >> 3) & 3) * 4;
    const int b32 = (sl % blocks) * (kGran / 32) + part / 4;
    for (int s = 0; s < top; ++s) {
        const int r = t * top + s;
        const int p = slot[r];
        if (p < 0 || p >= total || order[p] != r) continue;
        u8* to = codes + static_cast<long long>(p) * k + in_row;
        *reinterpret_cast<unsigned*>(to) = static_cast<unsigned>(w);
        *reinterpret_cast<unsigned*>(to + 16) = static_cast<unsigned>(w >> 32);
        if (part % 4 == 0)
            exps[(static_cast<long long>(b32 >> 2) * total + p) * 4 + (b32 & 3)] = static_cast<u8>(q.e + 127);
    }
}

}  // namespace pie::linear
