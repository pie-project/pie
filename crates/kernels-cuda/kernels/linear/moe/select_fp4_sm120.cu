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
/// GLU columns), a slice of 64 routes, `k` whole 128s: each 128-code chunk's
/// codes, routes and scales staged through one `kStages` cp.async ring.
template <int kGlu, int kStages>
__global__ void __launch_bounds__(256, 1) moe_select_mxfp4_w4a8(
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
    constexpr int kSlice = 8 * kNb;
    // A stage: the slice's 128 codes a route (rows 144 apart, so a warp's
    // 8-byte reads of eight routes fall on distinct banks), a route's four
    // scales, the CTA's 256 rows' 64 code bytes (80 apart, for its 4-byte
    // reads), a row's four scales.
    constexpr int kXStride = 144;
    constexpr int kWStride = 80;
    constexpr int kX = kSlice * kXStride;
    constexpr int kXs = kSlice * 4;
    constexpr int kW = kWarps * 32 * kWStride;
    constexpr int kWs = kWarps * 32 * 4;
    constexpr int kStage = kX + kXs + kW + kWs;

    const int at = static_cast<int>(act_div == 1 ? blockIdx.y : blockIdx.x);
    if (at >= offsets[num_experts + 1]) return;
    const int item = work[at];
    const int expert = item >> 16;
    const int slice = item & 0xFFFF;
    const int begin = offsets[expert] + slice * kSlice;
    const int count = min(offsets[expert + 1] - begin, kSlice);
    if (count <= 0) return;

    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid >> 5;
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
    const int own_base = cta_row + (tid >> 5) * kRowsWarp;
    const int own_t = (tid >> 4) & 1;
    const int own_tile = kGlu != 0 ? own_base + own_t * inter : own_base + 16 * own_t;
    const int own_row = min(max(own_tile, 0) + (tid & 15), n - 1);
    // The slice route this thread copies a quarter of: route tid / 4.
    const int xr = tid >> 2;
    const int xq4 = tid & 3;
    const int xsrc = xr < count ? order[begin + xr] / act_div : -1;

    extern __shared__ __align__(16) unsigned char w4a8_smem[];
    auto stage = [&](int ch) {
        if (ch < chunks) {
            unsigned char* st = w4a8_smem + (ch % kStages) * kStage;
            const bool live = xsrc >= 0;
            const u8* xf = xq + static_cast<long long>(live ? xsrc : 0) * k + ch * 128 + xq4 * 16;
            w4a8::cp16(st + xr * kXStride + xq4 * 16, xf, live ? 16 : 0);
            w4a8::cp16(st + xr * kXStride + 64 + xq4 * 16, xf + 64, live ? 16 : 0);
            if (xq4 == 0) {
                w4a8::cp4(st + kX + xr * 4, xe + static_cast<long long>(live ? xsrc : 0) * sf_row + ch * 4,
                          live ? 4 : 0);
            }
            unsigned char* ws = st + kX + kXs;
            const u8* wf = w_e + own_row * row_bytes + ch * 64;
#pragma unroll
            for (int q = 0; q < 4; ++q) w4a8::cp16(ws + tid * kWStride + q * 16, wf + q * 16, 16);
            w4a8::cp4(ws + kW + tid * 4, s_e + static_cast<long long>(own_row) * sf_row + ch * 4, 4);
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

    for (int ch = 0; ch < chunks; ++ch) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(kStages - 2));
        __syncthreads();
        // The ring's oldest stage, every warp past its MMAs.
        stage(ch + kStages - 1);
        if (rows_live) {
            const unsigned char* st = w4a8_smem + (ch % kStages) * kStage;
            const unsigned* xs = reinterpret_cast<const unsigned*>(st + kX);
            const unsigned char* ws = st + kX + kXs;
            const unsigned* wsc = reinterpret_cast<const unsigned*>(ws + kW);
            // A row's scale from the lanes of its parity (row g at even c,
            // g + 8 at odd), a route's from its group's lanes.
            unsigned sa[kTiles], sb[kNb];
#pragma unroll
            for (int t = 0; t < kTiles; ++t) sa[t] = wsc[warp * 32 + t * 16 + (c & 1) * 8 + g];
#pragma unroll
            for (int b = 0; b < kNb; ++b) sb[b] = xs[8 * b + g];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                unsigned a[kTiles][4];
#pragma unroll
                for (int t = 0; t < kTiles; ++t)
#pragma unroll
                    for (int h = 0; h < 2; ++h) {
                        const unsigned w = *reinterpret_cast<const unsigned*>(
                            ws + (warp * 32 + t * 16 + h * 8 + g) * kWStride + j * 16 + c * 4);
                        w4a8::spread(w, a[t][h], a[t][2 + h]);
                    }
#pragma unroll
                for (int b = 0; b < kNb; ++b) {
                    const uint2 bv =
                        *reinterpret_cast<const uint2*>(st + (8 * b + g) * kXStride + j * 32 + 8 * c);
                    const unsigned sbj = (sb[b] >> (8 * j)) & 0xFFu;
#pragma unroll
                    for (int t = 0; t < kTiles; ++t)
                        w4a8::mma(acc[t][b], a[t], bv.x, bv.y, (sa[t] >> (8 * j)) & 0xFFu, sbj);
                }
            }
        }
    }
    asm volatile("cp.async.wait_group 0;\n" ::);
    if (!rows_live) return;

    const bf16* b_e = bias != nullptr ? bias + static_cast<long long>(expert) * n : nullptr;
    auto factor = [](int) { return 1.f; };
    w4mma::w4_land<kGlu, kTiles, kNb>(acc, factor, b_e, order, begin, count, out, n, inter,
                                      row_base, g, c, limit, galpha);
}

}  // namespace pie::linear
