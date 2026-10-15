#pragma once

// DeepSeek-V4's fp8 key/value rows, as FlashMLA's `fp8_ds_mla` (V4) lays
// them: a page of `ps` positions is `ps` 576-byte rows (448 e4m3 codes of
// the unrotated dims, then the 64 rotated dims in bf16) followed by `ps`
// 8-byte scale rows (one ue8m0 exponent a 64 codes, seven, and a zero pad).
// A code reads as `code * 2^(exponent - 127)`. The window's shared key/value
// plane and the compressed rows are written alike.

#include <cstdint>

#include <cuda_bf16.h>
#include <cuda_fp8.h>

#include "prelude/device.cuh"

namespace pie::attn::dsv4 {

constexpr int kDsDim = 512;
constexpr int kDsNope = 448;
constexpr int kDsTile = 64;
constexpr int kDsScales = kDsNope / kDsTile;  // 7
constexpr int kDsData = kDsNope + (kDsDim - kDsNope) * 2;  // 576
constexpr int kDsScaleBytes = 8;
constexpr int kDsRow = kDsData + kDsScaleBytes;  // 584
constexpr float kFp8Max = 448.0f;

// A position's row: its 576 data bytes and its 8 scale bytes.
struct DsRow {
    std::uint8_t* data;
    std::uint8_t* scales;
};

__device__ __forceinline__ DsRow ds_row(std::uint8_t* plane, long long page, int off, int ps) {
    std::uint8_t* base = plane + page * static_cast<long long>(ps) * kDsRow;
    return {base + static_cast<long long>(off) * kDsData,
            base + static_cast<long long>(ps) * kDsData + static_cast<long long>(off) * kDsScaleBytes};
}

__device__ __forceinline__ float ue8m0(std::uint8_t e) {
    return __uint_as_float(static_cast<unsigned>(e) << 23);
}

// One row of 512 into `row`, by one warp: lane l holds dims 16l..16l+15,
// four lanes a 64-wide scale tile; lanes 28..31 hold the rotated dims, kept
// in bf16. As vLLM's insert: the exponent ceil(log2(amax / 448)), amax at
// least 1e-4, the codes saturated.
__device__ __forceinline__ void ds_store_warp(const bf16* __restrict__ src, DsRow row, int lane) {
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
    if (d0 >= kDsNope) {
        // The rotated dims, as they are.
        std::uint8_t* dst = row.data + kDsNope + (d0 - kDsNope) * 2;
        *reinterpret_cast<uint4*>(dst) = *reinterpret_cast<const uint4*>(src + d0);
        *reinterpret_cast<uint4*>(dst + 16) = *reinterpret_cast<const uint4*>(src + d0 + 8);
    }
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(x[i]));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    if (d0 >= kDsNope) return;
    amax = fmaxf(amax, 1e-4f);
    const float e = ceilf(log2f(amax / kFp8Max));
    const float inv = exp2f(-e);
    std::uint8_t codes[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const float v = fminf(fmaxf(x[i] * inv, -kFp8Max), kFp8Max);
        codes[i] = static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3));
    }
    *reinterpret_cast<uint4*>(row.data + d0) = *reinterpret_cast<const uint4*>(codes);
    if ((lane & 3) == 0) {
        row.scales[lane >> 2] = static_cast<std::uint8_t>(fmaxf(fminf(e + 127.f, 255.f), 0.f));
        if (lane == 0) row.scales[kDsScales] = 0;
    }
}

// The window's plane: row b of `plane_rows` at its page and offset (the
// `kv_append_shared` of an fp8 plane). A warp a row.
__global__ __launch_bounds__(128) void ds_append(
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
    ds_store_warp(plane_rows + static_cast<long long>(b_row) * kDsDim, ds_row(pages, page, off, page_size),
                  lane);
}

// The compressed rows: entry c at its boundary's position in its request's
// pages (the `pool_kv_append` of an fp8 plane). A warp an entry.
__global__ __launch_bounds__(128) void ds_store_entries(
    const bf16* __restrict__ entries,
    std::uint8_t* __restrict__ pages,
    const i32* __restrict__ boundary_pos,
    const i32* __restrict__ boundary_req,
    const u32* __restrict__ page_indices,
    const u32* __restrict__ page_indptr,
    int n,
    int page_size)
{
    const int c = static_cast<int>(blockIdx.x) * 4 + static_cast<int>(threadIdx.x >> 5);
    const int lane = static_cast<int>(threadIdx.x & 31);
    if (c >= n) return;
    const int pos = boundary_pos[c];
    if (pos < 0) return;
    const u32 page = page_indices[page_indptr[boundary_req[c]] + pos / page_size];
    ds_store_warp(entries + static_cast<long long>(c) * kDsDim,
                  ds_row(pages, page, pos % page_size, page_size), lane);
}

// 8 dims of a row from its data and scale bytes: `j` the first dim (a
// multiple of 8).
__device__ __forceinline__ uint4 ds_load8(const std::uint8_t* data, const std::uint8_t* scales, int j) {
    if (j >= kDsNope) return *reinterpret_cast<const uint4*>(data + kDsNope + (j - kDsNope) * 2);
    const float s = ue8m0(scales[j / kDsTile]);
    const uint2 codes = *reinterpret_cast<const uint2*>(data + j);
    const std::uint16_t* c2 = reinterpret_cast<const std::uint16_t*>(&codes);
    uint4 out;
    std::uint32_t* o32 = reinterpret_cast<std::uint32_t*>(&out);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        float lo, hi;
        asm("{\n\t.reg .b32 h2;\n\t.reg .b16 l, h;\n\t"
            "cvt.rn.f16x2.e4m3x2 h2, %2;\n\t"
            "mov.b32 {l, h}, h2;\n\t"
            "cvt.f32.f16 %0, l;\n\t"
            "cvt.f32.f16 %1, h;\n\t}"
            : "=f"(lo), "=f"(hi)
            : "h"(c2[i]));
        const __nv_bfloat162 b = __floats2bfloat162_rn(lo * s, hi * s);
        o32[i] = *reinterpret_cast<const std::uint32_t*>(&b);
    }
    return out;
}

}  // namespace pie::attn::dsv4
