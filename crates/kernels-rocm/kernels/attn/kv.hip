#pragma once

// The paged KV cache's writes. A page holds `page_size` slots of `h_kv`
// heads × `d` bf16, NHD (slot → head → d) or HND (head → slot → d). A
// quantized pool keeps NHD code pages (fp8 per tensor, fp8 / int8 with an
// f32 scale per (slot, head), or e2m1 two to a byte with a scale per
// `block_size`) and a bf16 shadow the attention reads, which the
// `dequant_*_active` kernels refresh. The fp8 codes are OCP e4m3fn / e5m2,
// converted in software: gfx942's converters speak fnuz.

#include "prelude/device.hip"

namespace pie::attn {

// The element offset of (page, slot, head, j) in either layout.
template <bool HND>
__device__ __forceinline__ long long kv_at(long long page, int slot, int head, int j, int page_size,
                                           int h_kv, int d) {
    return HND ? ((page * h_kv + head) * page_size + slot) * d + j
               : ((page * page_size + slot) * h_kv + head) * d + j;
}

// Token row b's k and v (h_kv·d each) into page w_page[b], slot w_off[b].
// A row whose slot is out of range, or that `row_valid` marks dead, is
// skipped. One block per token row; 8 elements a thread when d allows.
template <bool HND>
__global__ void kv_append_explicit(const bf16* __restrict__ k, const bf16* __restrict__ v,
                                   bf16* __restrict__ k_pages, bf16* __restrict__ v_pages,
                                   const u32* __restrict__ w_page, const u32* __restrict__ w_off,
                                   const u8* __restrict__ row_valid, int B, int page_size,
                                   int h_kv, int d, const u32* __restrict__ win) {
    const Window window{win};
    if (static_cast<int>(blockIdx.x) >= B || !window.live(blockIdx.x)) return;
    const long long b = window.plane(blockIdx.x);
    if (row_valid != nullptr && row_valid[b] == 0) return;
    const long long page = w_page[b];
    const int slot = static_cast<int>(w_off[b]);
    if (slot < 0 || slot >= page_size) return;
    const int row = h_kv * d;
    const bf16* ks = k + b * row;
    const bf16* vs = v + b * row;
    if (d % 8 == 0 && aligned16(ks) && aligned16(vs) && aligned16(k_pages) && aligned16(v_pages)) {
        for (int i = threadIdx.x * 8; i < row; i += blockDim.x * 8) {
            const long long dst = kv_at<HND>(page, slot, i / d, i % d, page_size, h_kv, d);
            *reinterpret_cast<b128*>(k_pages + dst) = *reinterpret_cast<const b128*>(ks + i);
            *reinterpret_cast<b128*>(v_pages + dst) = *reinterpret_cast<const b128*>(vs + i);
        }
        return;
    }
    for (int i = threadIdx.x; i < row; i += blockDim.x) {
        const long long dst = kv_at<HND>(page, slot, i / d, i % d, page_size, h_kv, d);
        k_pages[dst] = ks[i];
        v_pages[dst] = vs[i];
    }
}

// ------------------------------------------------------------ quantized --

// The fp8 interpretations the host names: 0 e4m3, 1 e5m2.
constexpr u32 kFp8E5M2 = 1;

constexpr float pow2(int e) {
    float v = 1.f;
    for (; e > 0; --e) v *= 2.f;
    for (; e < 0; ++e) v *= 0.5f;
    return v;
}

// f32 → an OCP fp8 code of M mantissa bits and exponent bias BIAS: round to
// nearest even, saturate to the largest finite code `TOP`, NaN to 0x7f (the
// rounding and saturation of CUDA's `__NV_SATFINITE`).
template <int M, int BIAS, u32 TOP>
__device__ __forceinline__ u8 fp8_encode(float x) {
    const u32 sign = (__float_as_uint(x) >> 24) & 0x80u;
    const float a = fabsf(x);
    if (!(a == a)) return static_cast<u8>(sign | 0x7fu);
    // Below the least normal the code is a count of the least subnormal
    // step; a count of 2^M is the least normal's own code.
    if (a < pow2(1 - BIAS)) return static_cast<u8>(sign | static_cast<u32>(rintf(a * pow2(BIAS - 1 + M))));
    u32 u = __float_as_uint(a);
    u += (1u << (22 - M)) - 1u + ((u >> (23 - M)) & 1u);
    const u32 code = (((u >> 23) - 127u + BIAS) << M) | ((u >> (23 - M)) & ((1u << M) - 1u));
    return static_cast<u8>(sign | min(code, TOP));
}

template <int M, int BIAS>
__device__ __forceinline__ float fp8_decode(u32 c) {
    const u32 e = (c & 0x7fu) >> M, m = c & ((1u << M) - 1u);
    float a = e == 0 ? static_cast<float>(m) * pow2(1 - BIAS - M)
                     : __uint_as_float(((e + 127u - BIAS) << 23) | (m << (23 - M)));
    if constexpr (M == 3) {
        if ((c & 0x7fu) == 0x7fu) a = __builtin_nanf("");
    } else {
        if (e == 31u) a = m == 0 ? __builtin_huge_valf() : __builtin_nanf("");
    }
    return (c & 0x80u) ? -a : a;
}

__device__ __forceinline__ u8 e4m3_encode(float x) { return fp8_encode<3, 7, 0x7eu>(x); }
__device__ __forceinline__ float e4m3_decode(u32 c) { return fp8_decode<3, 7>(c & 0xffu); }

__device__ __forceinline__ u8 fp8_encode(float x, u32 kind) {
    return kind == kFp8E5M2 ? fp8_encode<2, 15, 0x7bu>(x) : e4m3_encode(x);
}
__device__ __forceinline__ float fp8_decode(u32 c, u32 kind) {
    return kind == kFp8E5M2 ? fp8_decode<2, 15>(c & 0xffu) : e4m3_decode(c);
}

// e2m1, sign over two exponent bits and one mantissa bit: the magnitudes
// 0, 0.5, 1, 1.5, 2, 3, 4, 6.
__device__ __forceinline__ float e2m1_decode(u32 code) {
    const u32 mag = code & 7u;
    const float v = mag < 4u ? 0.5f * static_cast<float>(mag)
                             : static_cast<float>(2u << ((mag - 4u) >> 1)) * (1.f + 0.5f * static_cast<float>(mag & 1u));
    return (code & 8u) ? -v : v;
}

// The nearest e2m1 level to x, a tie going to the smaller magnitude: the
// count of level midpoints |x| is past.
__device__ __forceinline__ u32 e2m1_encode(float x) {
    const float a = fabsf(x);
    const u32 mag = (a > 0.25f) + (a > 0.75f) + (a > 1.25f) + (a > 1.75f) + (a > 2.5f) + (a > 3.5f) + (a > 5.f);
    return (x < 0.f ? 8u : 0u) | mag;
}

// The page and slot token row `token` of the launch lands in: the
// request's new rows are the last of its kv.
__device__ __forceinline__ void resolve_dst(const u32* __restrict__ qo_indptr,
                                            const u32* __restrict__ page_indices,
                                            const u32* __restrict__ page_indptr,
                                            const u32* __restrict__ last_page_lens, int R,
                                            int page_size, int token, long long& page, int& slot) {
    int r = R - 1;
    for (int i = 0; i < R; ++i)
        if (token < static_cast<int>(qo_indptr[i + 1])) {
            r = i;
            break;
        }
    const int fresh = static_cast<int>(qo_indptr[r + 1] - qo_indptr[r]);
    const int pages = static_cast<int>(page_indptr[r + 1] - page_indptr[r]);
    const int kv_after = (pages - 1) * page_size + static_cast<int>(last_page_lens[r]);
    const int at = kv_after - fresh + (token - static_cast<int>(qo_indptr[r]));
    page = page_indices[page_indptr[r] + at / page_size];
    slot = at % page_size;
}

// Token row t's k and v as fp8 codes, no scale. One block per token row.
// A launch row resolves its slot against the indptr as launched and reads
// its plane row; a window caps the requests at its lane count.
__global__ void kv_append_fp8_per_tensor(const bf16* __restrict__ k, const bf16* __restrict__ v,
                                         u8* __restrict__ k_pages, u8* __restrict__ v_pages,
                                         const u32* __restrict__ qo_indptr,
                                         const u32* __restrict__ page_indices,
                                         const u32* __restrict__ page_indptr,
                                         const u32* __restrict__ last_page_lens, int R,
                                         int page_size, int h_kv, int d, u32 fp8_kind,
                                         const u32* __restrict__ win) {
    const Window window{win};
    const int t = static_cast<int>(blockIdx.x);
    if (!window.live(t)) return;
    const long long t_row = window.plane(t);
    long long page;
    int slot;
    resolve_dst(qo_indptr, page_indices, page_indptr, last_page_lens, window.lanes(R), page_size, t,
                page, slot);

    const int row = h_kv * d;
    const bf16* ks = k + t_row * row;
    const bf16* vs = v + t_row * row;
    const long long dst = (page * page_size + slot) * row;
    if (row % 8 == 0 && aligned16(ks) && aligned16(vs)) {
        for (int i = threadIdx.x * 8; i < row; i += blockDim.x * 8) {
            const Pack8<bf16> kp = *reinterpret_cast<const Pack8<bf16>*>(ks + i);
            const Pack8<bf16> vp = *reinterpret_cast<const Pack8<bf16>*>(vs + i);
            u64 kc = 0, vc = 0;
#pragma unroll
            for (int e = 0; e < 8; ++e) {
                kc |= static_cast<u64>(fp8_encode(to_f32(kp.v[e]), fp8_kind)) << (8 * e);
                vc |= static_cast<u64>(fp8_encode(to_f32(vp.v[e]), fp8_kind)) << (8 * e);
            }
            *reinterpret_cast<u64*>(k_pages + dst + i) = kc;
            *reinterpret_cast<u64*>(v_pages + dst + i) = vc;
        }
        return;
    }
    for (int i = threadIdx.x; i < row; i += blockDim.x) {
        k_pages[dst + i] = fp8_encode(to_f32(ks[i]), fp8_kind);
        v_pages[dst + i] = fp8_encode(to_f32(vs[i]), fp8_kind);
    }
}

// Token row t, head h, as e4m3 (FP8) or i8 codes with one f32 scale each for
// k and v: amax / qmax, or 1 for an all-zero head. One block per (token
// row, head).
template <bool FP8>
__global__ void kv_append_per_token_head(const bf16* __restrict__ k, const bf16* __restrict__ v,
                                         void* __restrict__ k_pages_raw,
                                         void* __restrict__ v_pages_raw,
                                         float* __restrict__ k_scales,
                                         float* __restrict__ v_scales,
                                         const u32* __restrict__ qo_indptr,
                                         const u32* __restrict__ page_indices,
                                         const u32* __restrict__ page_indptr,
                                         const u32* __restrict__ last_page_lens, int R,
                                         int page_size, int h_kv, int d,
                                         const u32* __restrict__ win) {
    __shared__ float red[1024 / kWave];
    const Window window{win};
    const int t = static_cast<int>(blockIdx.x);
    if (!window.live(t)) return;
    const long long t_row = window.plane(t);
    const int h = static_cast<int>(blockIdx.y);

    const long long src = (t_row * h_kv + h) * d;
    float k_abs = 0.f, v_abs = 0.f;
    for (int j = threadIdx.x; j < d; j += blockDim.x) {
        k_abs = fmaxf(k_abs, fabsf(to_f32(k[src + j])));
        v_abs = fmaxf(v_abs, fabsf(to_f32(v[src + j])));
    }
    k_abs = block_max(k_abs, red);
    v_abs = block_max(v_abs, red);

    long long page;
    int slot;
    resolve_dst(qo_indptr, page_indices, page_indptr, last_page_lens, window.lanes(R), page_size, t,
                page, slot);
    const long long cell = (page * page_size + slot) * h_kv + h;
    const long long dst = cell * d;

    const float qmax = FP8 ? 448.f : 127.f;
    const float k_scale = k_abs > 0.f ? k_abs / qmax : 1.f;
    const float v_scale = v_abs > 0.f ? v_abs / qmax : 1.f;
    if (threadIdx.x == 0) {
        k_scales[cell] = k_scale;
        v_scales[cell] = v_scale;
    }
    const float k_inv = 1.f / k_scale;
    const float v_inv = 1.f / v_scale;

    if constexpr (FP8) {
        u8* kp = static_cast<u8*>(k_pages_raw);
        u8* vp = static_cast<u8*>(v_pages_raw);
        for (int j = threadIdx.x; j < d; j += blockDim.x) {
            kp[dst + j] = e4m3_encode(to_f32(k[src + j]) * k_inv);
            vp[dst + j] = e4m3_encode(to_f32(v[src + j]) * v_inv);
        }
    } else {
        i8* kp = static_cast<i8*>(k_pages_raw);
        i8* vp = static_cast<i8*>(v_pages_raw);
        for (int j = threadIdx.x; j < d; j += blockDim.x) {
            kp[dst + j] = static_cast<i8>(fminf(fmaxf(rintf(to_f32(k[src + j]) * k_inv), -128.f), 127.f));
            vp[dst + j] = static_cast<i8>(fminf(fmaxf(rintf(to_f32(v[src + j]) * v_inv), -128.f), 127.f));
        }
    }
}

// Token row t, head h, block b (elements b·block_size onward) as e2m1 codes
// with one f32 scale each for k and v: amax / 6, or 1 for an all-zero
// block. A head packs ⌈d/2⌉ bytes, element 2i in byte i's low nibble. One
// wavefront per (token row, head, block); block_size is even, so no byte
// straddles two blocks.
__global__ void kv_append_fp4_block(const bf16* __restrict__ k, const bf16* __restrict__ v,
                                    u8* __restrict__ k_pages, u8* __restrict__ v_pages,
                                    float* __restrict__ k_scales, float* __restrict__ v_scales,
                                    const u32* __restrict__ qo_indptr,
                                    const u32* __restrict__ page_indices,
                                    const u32* __restrict__ page_indptr,
                                    const u32* __restrict__ last_page_lens, int R, int page_size,
                                    int h_kv, int d, int block_size, const u32* __restrict__ win) {
    const Window window{win};
    const int t = static_cast<int>(blockIdx.x);
    if (!window.live(t)) return;
    const long long t_row = window.plane(t);
    const int h = static_cast<int>(blockIdx.y);
    const int b = static_cast<int>(blockIdx.z);
    const int start = b * block_size;
    const int end = min(start + block_size, d);

    const long long src = (t_row * h_kv + h) * d;
    float k_abs = 0.f, v_abs = 0.f;
    for (int j = start + threadIdx.x; j < end; j += blockDim.x) {
        k_abs = fmaxf(k_abs, fabsf(to_f32(k[src + j])));
        v_abs = fmaxf(v_abs, fabsf(to_f32(v[src + j])));
    }
    k_abs = wave_max(k_abs);
    v_abs = wave_max(v_abs);
    const float k_scale = k_abs > 0.f ? k_abs / 6.f : 1.f;
    const float v_scale = v_abs > 0.f ? v_abs / 6.f : 1.f;

    long long page;
    int slot;
    resolve_dst(qo_indptr, page_indices, page_indptr, last_page_lens, window.lanes(R), page_size, t,
                page, slot);
    const int packed_d = (d + 1) / 2;
    const int blocks_per_head = (d + block_size - 1) / block_size;
    const long long cell = (page * page_size + slot) * h_kv + h;
    if (threadIdx.x == 0) {
        k_scales[cell * blocks_per_head + b] = k_scale;
        v_scales[cell * blocks_per_head + b] = v_scale;
    }

    const float k_inv = 1.f / k_scale;
    const float v_inv = 1.f / v_scale;
    u8* kp = k_pages + cell * packed_d;
    u8* vp = v_pages + cell * packed_d;
    for (int byte = start / 2 + threadIdx.x; byte <= (end - 1) / 2; byte += blockDim.x) {
        const int j0 = 2 * byte, j1 = j0 + 1;
        u32 kc = e2m1_encode(to_f32(k[src + j0]) * k_inv);
        u32 vc = e2m1_encode(to_f32(v[src + j0]) * v_inv);
        if (j1 < d) {
            kc |= e2m1_encode(to_f32(k[src + j1]) * k_inv) << 4;
            vc |= e2m1_encode(to_f32(v[src + j1]) * v_inv) << 4;
        }
        kp[byte] = static_cast<u8>(kc);
        vp[byte] = static_cast<u8>(vc);
    }
}

// The dequant kernels walk the batch's pages as one logical run of
// `page_indices`-ordered pages, eight elements a thread: element i of the
// run is element i % page_elems of page page_indices[i / page_elems], and
// lands at that same element of the bf16 shadow. When a head's width is a
// multiple of eight, a thread's eight share one page and one scale and move
// as one wide access.
constexpr int kDequantRun = 8;

__device__ __forceinline__ long long dequant_first() {
    return (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) * kDequantRun;
}

__global__ void dequant_fp8_pages_active(const u8* __restrict__ k_pages,
                                         const u8* __restrict__ v_pages, bf16* __restrict__ k_out,
                                         bf16* __restrict__ v_out,
                                         const u32* __restrict__ page_indices, long long n,
                                         int page_elems, u32 fp8_kind) {
    const long long first = dequant_first();
    if (first >= n) return;
    if (page_elems % kDequantRun == 0) {
        const long long elem =
            static_cast<long long>(page_indices[first / page_elems]) * page_elems + first % page_elems;
        const u64 kc = *reinterpret_cast<const u64*>(k_pages + elem);
        const u64 vc = *reinterpret_cast<const u64*>(v_pages + elem);
        Pack8<bf16> ko, vo;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            ko.v[e] = from_f32<bf16>(fp8_decode(static_cast<u32>(kc >> (8 * e)), fp8_kind));
            vo.v[e] = from_f32<bf16>(fp8_decode(static_cast<u32>(vc >> (8 * e)), fp8_kind));
        }
        *reinterpret_cast<Pack8<bf16>*>(k_out + elem) = ko;
        *reinterpret_cast<Pack8<bf16>*>(v_out + elem) = vo;
        return;
    }
    for (long long i = first; i < min(first + kDequantRun, n); ++i) {
        const long long elem =
            static_cast<long long>(page_indices[i / page_elems]) * page_elems + i % page_elems;
        k_out[elem] = from_f32<bf16>(fp8_decode(k_pages[elem], fp8_kind));
        v_out[elem] = from_f32<bf16>(fp8_decode(v_pages[elem], fp8_kind));
    }
}

// Codes scaled per (slot, head), e4m3 (FP8) or i8: code · scale.
template <bool FP8, class T, class C>
__device__ __forceinline__ void dequant_per_token_head(const C* __restrict__ k_pages,
                                                       const C* __restrict__ v_pages,
                                                       const float* __restrict__ k_scales,
                                                       const float* __restrict__ v_scales,
                                                       T* __restrict__ k_out, T* __restrict__ v_out,
                                                       const u32* __restrict__ page_indices,
                                                       long long n, int page_size, int h_kv, int d) {
    const long long first = dequant_first();
    if (first >= n) return;
    const int page_elems = page_size * h_kv * d;
    auto value = [](u32 c) {
        return FP8 ? e4m3_decode(c) : static_cast<float>(static_cast<i8>(c & 0xffu));
    };
    if (d % kDequantRun == 0) {
        const long long page = page_indices[first / page_elems];
        const int local = static_cast<int>(first % page_elems);
        const long long elem = page * page_elems + local;
        const long long cell = page * page_size * h_kv + local / d;
        const float ks = k_scales[cell], vs = v_scales[cell];
        const u64 kc = *reinterpret_cast<const u64*>(k_pages + elem);
        const u64 vc = *reinterpret_cast<const u64*>(v_pages + elem);
        Pack8<T> ko, vo;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            ko.v[e] = from_f32<T>(value(static_cast<u32>(kc >> (8 * e))) * ks);
            vo.v[e] = from_f32<T>(value(static_cast<u32>(vc >> (8 * e))) * vs);
        }
        *reinterpret_cast<Pack8<T>*>(k_out + elem) = ko;
        *reinterpret_cast<Pack8<T>*>(v_out + elem) = vo;
        return;
    }
    for (long long i = first; i < min(first + kDequantRun, n); ++i) {
        const long long page = page_indices[i / page_elems];
        const int local = static_cast<int>(i % page_elems);
        const long long elem = page * page_elems + local;
        const long long cell = page * page_size * h_kv + local / d;
        k_out[elem] = from_f32<T>(value(static_cast<u8>(k_pages[elem])) * k_scales[cell]);
        v_out[elem] = from_f32<T>(value(static_cast<u8>(v_pages[elem])) * v_scales[cell]);
    }
}

__global__ void dequant_fp8_per_token_head_pages_active(
    const u8* __restrict__ k_pages, const u8* __restrict__ v_pages,
    const float* __restrict__ k_scales, const float* __restrict__ v_scales, bf16* __restrict__ k_out,
    bf16* __restrict__ v_out, const u32* __restrict__ page_indices, long long n, int page_size,
    int h_kv, int d) {
    dequant_per_token_head<true>(k_pages, v_pages, k_scales, v_scales, k_out, v_out, page_indices,
                                 n, page_size, h_kv, d);
}

__global__ void dequant_int8_per_token_head_pages_active(
    const i8* __restrict__ k_pages, const i8* __restrict__ v_pages,
    const float* __restrict__ k_scales, const float* __restrict__ v_scales, bf16* __restrict__ k_out,
    bf16* __restrict__ v_out, const u32* __restrict__ page_indices, long long n, int page_size,
    int h_kv, int d) {
    dequant_per_token_head<false>(k_pages, v_pages, k_scales, v_scales, k_out, v_out, page_indices,
                                  n, page_size, h_kv, d);
}

// e2m1 codes with a scale per block: the shadow's element (page, row, j),
// row over (slot, head), is code(row, j) · scale(row, j / block_size).
__global__ void dequant_fp4_pages_active(const u8* __restrict__ k_pages,
                                         const u8* __restrict__ v_pages,
                                         const float* __restrict__ k_scales,
                                         const float* __restrict__ v_scales, bf16* __restrict__ k_out,
                                         bf16* __restrict__ v_out,
                                         const u32* __restrict__ page_indices, long long n,
                                         int page_size, int h_kv, int d, int block_size) {
    const long long first = dequant_first();
    if (first >= n) return;
    const int page_elems = page_size * h_kv * d;
    const int packed_d = (d + 1) / 2;
    const int blocks_per_head = (d + block_size - 1) / block_size;
    if (d % kDequantRun == 0 && block_size % kDequantRun == 0) {
        const long long page = page_indices[first / page_elems];
        const int local = static_cast<int>(first % page_elems);
        const long long row = page * page_size * h_kv + local / d;
        const int j = local % d;
        const float ks = k_scales[row * blocks_per_head + j / block_size];
        const float vs = v_scales[row * blocks_per_head + j / block_size];
        const u32 kc = *reinterpret_cast<const u32*>(k_pages + row * packed_d + j / 2);
        const u32 vc = *reinterpret_cast<const u32*>(v_pages + row * packed_d + j / 2);
        Pack8<bf16> ko, vo;
#pragma unroll
        for (int e = 0; e < 8; ++e) {
            ko.v[e] = from_f32<bf16>(e2m1_decode(kc >> (4 * e)) * ks);
            vo.v[e] = from_f32<bf16>(e2m1_decode(vc >> (4 * e)) * vs);
        }
        *reinterpret_cast<Pack8<bf16>*>(k_out + page * page_elems + local) = ko;
        *reinterpret_cast<Pack8<bf16>*>(v_out + page * page_elems + local) = vo;
        return;
    }
    for (long long i = first; i < min(first + kDequantRun, n); ++i) {
        const long long page = page_indices[i / page_elems];
        const int local = static_cast<int>(i % page_elems);
        const long long row = page * page_size * h_kv + local / d;
        const int j = local % d;
        const long long cell = row * blocks_per_head + j / block_size;
        const int shift = (j & 1) ? 4 : 0;
        k_out[page * page_elems + local] = from_f32<bf16>(e2m1_decode(k_pages[row * packed_d + j / 2] >> shift) * k_scales[cell]);
        v_out[page * page_elems + local] = from_f32<bf16>(e2m1_decode(v_pages[row * packed_d + j / 2] >> shift) * v_scales[cell]);
    }
}

}  // namespace pie::attn
