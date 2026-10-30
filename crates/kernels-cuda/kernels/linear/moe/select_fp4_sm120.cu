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
/// routes (`offsets[num_experts]`) are left.
template <int kGran>
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
    *reinterpret_cast<unsigned long long*>(codes + at + (lane % Block::kLanes) * 8) = q.codes();
    const int part = lane % Block::kLanes;
    if (part % 4 == 0) exps[at / 32 + part / 4] = static_cast<u8>(q.e + 127);
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
