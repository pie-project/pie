#include "prim/kv_rows.cuh"

// DeepSeek-V4's fp8 key/value rows, as FlashMLA's `fp8_ds_mla` (V4) lays
// them: a page of `ps` positions is `ps` 576-byte rows (448 e4m3 codes of
// the unrotated dims, then the 64 rotated dims in bf16) followed by `ps`
// 8-byte scale rows (one ue8m0 exponent a 64 codes, seven, and a zero pad).
// A code reads as `code * 2^(exponent - 127)`. The window's shared key/value
// plane and the compressed rows are written alike.

namespace pie::attn::dsv4 {

// 2^ceil(log2(amax / 448)) as its biased exponent, from the quotient's bits
// (the reference's `fast_round_scale`), amax at least 1e-4.
__device__ __forceinline__ int ue8m0_ceil(float amax) {
    const unsigned bits = __float_as_uint(fmaxf(amax, 1e-4f) * (1.f / kFp8Max));
    return static_cast<int>((bits >> 23) & 0xffu) + ((bits & 0x7fffffu) != 0u ? 1 : 0);
}

// One row of 512 into `row`, by one warp: lane l quantizes dims 16l..16l+15.
template <int F>
__device__ __forceinline__ void rows_store_warp(const bf16* __restrict__ src, DsRow row, int lane) {
    if constexpr (F == kV4) {
        ds_store_warp(src, row, lane);
    } else {
        const int d0 = lane * 16;
        float x[16];
        {
            const uint4 a = *reinterpret_cast<const uint4*>(src + d0);
            const uint4 b = *reinterpret_cast<const uint4*>(src + d0 + 8);
            const bf16* pa = reinterpret_cast<const bf16*>(&a);
            const bf16* pb = reinterpret_cast<const bf16*>(&b);
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                x[i] = Elem<bf16>::to_f32(pa[i]);
                x[8 + i] = Elem<bf16>::to_f32(pb[i]);
            }
        }
        float amax = 0.f;
#pragma unroll
        for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(x[i]));
        if constexpr (F == kMx8) {
            // A 32-wide block is two lanes' runs.
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
            const int e = ue8m0_ceil(amax);
            const float inv = __uint_as_float(static_cast<unsigned>(254 - e) << 23);
            std::uint8_t codes[16];
#pragma unroll
            for (int i = 0; i < 16; ++i) {
                const float v = fminf(fmaxf(x[i] * inv, -kFp8Max), kFp8Max);
                codes[i] = static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
            }
            *reinterpret_cast<uint4*>(row.data + d0) = *reinterpret_cast<const uint4*>(codes);
            if ((lane & 1) == 0) row.scales[lane >> 1] = static_cast<std::uint8_t>(e);
        } else {
            // The reference's training floor keeps an all-zero group's scale
            // nonzero; the scale rounds to e4m3, the codes to e2m1 under it.
            const std::uint8_t sb = static_cast<std::uint8_t>(
                __nv_cvt_float_to_fp8(fmaxf(amax, 6.f * 0.001953125f) / 6.f, __NV_SATFINITE, __NV_E4M3));
            const float sv = e4m3_float(sb);
            std::uint8_t packed[8];
#pragma unroll
            for (int i = 0; i < 8; ++i) {
                const unsigned lo = e2m1_code(fminf(fmaxf(__fdiv_rn(x[2 * i], sv), -6.f), 6.f));
                const unsigned hi = e2m1_code(fminf(fmaxf(__fdiv_rn(x[2 * i + 1], sv), -6.f), 6.f));
                packed[i] = static_cast<std::uint8_t>(lo | (hi << 4));
            }
            *reinterpret_cast<uint2*>(row.data + d0 / 2) = *reinterpret_cast<const uint2*>(packed);
            row.scales[lane] = sb;
        }
    }
}

// The window's plane in format F: row b of `plane_rows` at its page and
// offset (`kv_append_shared`). A warp a row.
template <int F>
__global__ __launch_bounds__(128) void rows_append(
    const bf16* __restrict__ plane_rows,
    std::uint8_t* __restrict__ pages,
    const u32* __restrict__ w_page,
    const u32* __restrict__ w_off,
    const u8* __restrict__ row_valid,
    int B,
    int page_size,
    const u32* __restrict__ win)
{
    const int b = static_cast<int>(blockIdx.x) * 4 + static_cast<int>(threadIdx.x >> 5);
    const int lane = static_cast<int>(threadIdx.x & 31);
    if (b >= B) return;
    if (win != nullptr && b >= static_cast<int>(win[0])) return;
    const int b_row = win != nullptr ? b + static_cast<int>(win[1]) : b;
    if (row_valid != nullptr && row_valid[b_row] == 0) return;
    const int page = static_cast<int>(w_page[b_row]);
    const int off = static_cast<int>(w_off[b_row]);
    if (off < 0 || off >= page_size) return;
    rows_store_warp<F>(plane_rows + static_cast<long long>(b_row) * kDsDim,
                       rows_at<F>(pages, page, off, page_size), lane);
}

// The compressed rows in format F: entry c at its boundary's position in its
// request's pages (`pool_kv_append`). A warp an entry.
template <int F>
__global__ __launch_bounds__(128) void rows_store_entries(
    const bf16* __restrict__ entries,
    std::uint8_t* __restrict__ pages,
    const i32* __restrict__ boundary_pos,
    const i32* __restrict__ boundary_req,
    const u32* __restrict__ page_indices,
    const u32* __restrict__ page_indptr,
    int n,
    int page_size,
    int div)
{
    const int c = static_cast<int>(blockIdx.x) * 4 + static_cast<int>(threadIdx.x >> 5);
    const int lane = static_cast<int>(threadIdx.x & 31);
    if (c >= n) return;
    const int pos = boundary_pos[c];
    if (pos < 0) return;
    const int cell = pos / div, rows = page_size / div;
    const u32 page = page_indices[page_indptr[boundary_req[c]] + cell / rows];
    rows_store_warp<F>(entries + static_cast<long long>(c) * kDsDim, rows_at<F>(pages, page, cell % rows, rows),
                       lane);
}

}
