#pragma once

// Force-included ahead of every unit this crate compiles (jit/hiprtc.rs puts
// it on the first line), so it is in scope before any carried header. It
// does three things, in this order:
//
// 1. Says which compiler this is: `__CUDACC_RTC__` for the units that branch
//    on it, `PIE_ROCM` for the ones that branch on us, and the portable fp8 /
//    bf16 paths for the shims that would otherwise reach for PTX.
// 2. Renames the fp16/bf16 names `hiprtc_runtime.h` already declares for
//    HIP's own `__half` / `__hip_bfloat16`. The carried shims declare the
//    same names for `::pie::f16` / `::pie::bf16`, and a C++ overload cannot
//    differ only in its return type. The rename reaches every use in the
//    carried text, so the units read exactly as they did under NVRTC.
// 3. Answers the CUDA warp intrinsics. A CUDA warp is 32 lanes; a CDNA
//    wavefront is 64. The carried units compute `lane = tid & 31` and
//    `warp = tid >> 5` and shuffle with 32-lane offsets, so every warp
//    intrinsic here acts on the 32-lane half of the wavefront the calling
//    lane sits in: a CUDA warp becomes one segment of a wavefront, and the
//    arithmetic the units do on lane and warp ids stays true.

#define __CUDACC_RTC__ 1
#define PIE_ROCM 1
#define PIE_HALFTYPE_FORCE_PORTABLE 1

// ---- 2. names hiprtc already declares -------------------------------------

#define __half pie_cuda_half
#define half pie_cuda_half_t
#define __half2 pie_cuda_half2
#define half2 pie_cuda_half2_t
#define __half_raw pie_cuda_half_raw
#define __half2_raw pie_cuda_half2_raw
#define __half2float pie_cuda_half2float
#define __float2half pie_cuda_float2half
#define __float2half_rn pie_cuda_float2half_rn
#define __float2half2_rn pie_cuda_float2half2_rn
#define __floats2half2_rn pie_cuda_floats2half2_rn
#define __float22half2_rn pie_cuda_float22half2_rn
#define __halves2half2 pie_cuda_halves2half2
#define __half22float2 pie_cuda_half22float2
#define make_half2 pie_cuda_make_half2
#define __half_as_ushort pie_cuda_half_as_ushort
#define __ushort_as_half pie_cuda_ushort_as_half
#define __hmul pie_cuda_hmul
#define __hsub pie_cuda_hsub
#define __hadd pie_cuda_hadd
#define __hmax pie_cuda_hmax
#define __hmul2 pie_cuda_hmul2
#define __hsub2 pie_cuda_hsub2
#define __hfma2 pie_cuda_hfma2
#define __hmax2 pie_cuda_hmax2
#define __half2half2 pie_cuda_half2half2
#define __habs pie_cuda_habs
#define __habs2 pie_cuda_habs2
#define __hadd2_rn pie_cuda_hadd2_rn
#define __bfloat162float pie_cuda_bfloat162float
#define __float2bfloat16 pie_cuda_float2bfloat16
#define __float2bfloat16_rn pie_cuda_float2bfloat16_rn
#define __float2bfloat162_rn pie_cuda_float2bfloat162_rn
#define __floats2bfloat162_rn pie_cuda_floats2bfloat162_rn
#define __float22bfloat162_rn pie_cuda_float22bfloat162_rn
#define __bfloat1622float2 pie_cuda_bfloat1622float2
#define make_bfloat162 pie_cuda_make_bfloat162
#define __bfloat162bfloat162 pie_cuda_bfloat162bfloat162
#define __bfloat16_as_ushort pie_cuda_bfloat16_as_ushort

// ---- 3. the CUDA warp, as one 32-lane segment of the wavefront -------------

#ifdef warpSize
#undef warpSize
#endif
#define warpSize 32

namespace pie_hip {

constexpr int kWarp = 32;

// The calling lane's position in its wavefront (0..63) and the segment the
// CUDA warp it stands for occupies (0 or 1 on a 64-lane wavefront).
__device__ __forceinline__ unsigned wave_lane() { return __builtin_amdgcn_mbcnt_hi(~0u, __builtin_amdgcn_mbcnt_lo(~0u, 0u)); }
__device__ __forceinline__ unsigned segment_shift() { return wave_lane() & ~31u; }

__device__ __forceinline__ int clamp_width(int width) { return width > kWarp ? kWarp : width; }

// A shuffle moves 32-bit words; wider and narrower values travel as words.
template <class T>
__device__ __forceinline__ T shuffle_words(T v, int (*move)(int, int, int), int arg, int width) {
    static_assert(sizeof(T) % 4 == 0 || sizeof(T) < 4, "shuffled values are whole words or narrower");
    if constexpr (sizeof(T) < 4) {
        int word = 0;
        __builtin_memcpy(&word, &v, sizeof(T));
        word = move(word, arg, width);
        T out;
        __builtin_memcpy(&out, &word, sizeof(T));
        return out;
    } else {
        constexpr int kWords = sizeof(T) / 4;
        int words[kWords];
        __builtin_memcpy(words, &v, sizeof(T));
#pragma unroll
        for (int i = 0; i < kWords; ++i) words[i] = move(words[i], arg, width);
        T out;
        __builtin_memcpy(&out, words, sizeof(T));
        return out;
    }
}

__device__ __forceinline__ int move_idx(int v, int src, int width) { return __shfl(v, src, width); }
__device__ __forceinline__ int move_xor(int v, int mask, int width) { return __shfl_xor(v, mask, width); }
__device__ __forceinline__ int move_up(int v, int delta, int width) { return __shfl_up(v, static_cast<unsigned>(delta), width); }
__device__ __forceinline__ int move_down(int v, int delta, int width) { return __shfl_down(v, static_cast<unsigned>(delta), width); }

// The ballot of the calling lane's 32-lane segment, as CUDA's 32-bit mask.
__device__ __forceinline__ unsigned segment_ballot(int predicate) {
    const unsigned long long all = __ballot(predicate);
    return static_cast<unsigned>(all >> segment_shift());
}

}  // namespace pie_hip

template <class T>
__device__ __forceinline__ T pie_shfl_sync(unsigned long long, T v, int src, int width = 32) {
    return pie_hip::shuffle_words(v, pie_hip::move_idx, src, pie_hip::clamp_width(width));
}
template <class T>
__device__ __forceinline__ T pie_shfl_xor_sync(unsigned long long, T v, int mask, int width = 32) {
    return pie_hip::shuffle_words(v, pie_hip::move_xor, mask, pie_hip::clamp_width(width));
}
template <class T>
__device__ __forceinline__ T pie_shfl_up_sync(unsigned long long, T v, unsigned delta, int width = 32) {
    return pie_hip::shuffle_words(v, pie_hip::move_up, static_cast<int>(delta), pie_hip::clamp_width(width));
}
template <class T>
__device__ __forceinline__ T pie_shfl_down_sync(unsigned long long, T v, unsigned delta, int width = 32) {
    return pie_hip::shuffle_words(v, pie_hip::move_down, static_cast<int>(delta), pie_hip::clamp_width(width));
}
__device__ __forceinline__ unsigned pie_ballot_sync(unsigned long long mask, int predicate) {
    return pie_hip::segment_ballot(predicate) & static_cast<unsigned>(mask);
}
__device__ __forceinline__ int pie_any_sync(unsigned long long mask, int predicate) {
    return (pie_hip::segment_ballot(predicate) & static_cast<unsigned>(mask)) != 0u;
}
__device__ __forceinline__ int pie_all_sync(unsigned long long mask, int predicate) {
    const unsigned active = pie_hip::segment_ballot(1) & static_cast<unsigned>(mask);
    return (pie_hip::segment_ballot(predicate) & active) == active;
}
__device__ __forceinline__ unsigned pie_activemask() { return pie_hip::segment_ballot(1); }
__device__ __forceinline__ void pie_syncwarp(unsigned long long = ~0ull) {
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
}

#define __shfl_sync pie_shfl_sync
#define __shfl_xor_sync pie_shfl_xor_sync
#define __shfl_up_sync pie_shfl_up_sync
#define __shfl_down_sync pie_shfl_down_sync
#define __ballot_sync pie_ballot_sync
#define __any_sync pie_any_sync
#define __all_sync pie_all_sync
#define __activemask pie_activemask
#define __syncwarp pie_syncwarp

// ---- the rest of the CUDA device surface the units touch ------------------

// CUDA's `__byte_perm(x, y, s)`: byte i of the result is byte s[4i..4i+3] of
// the eight bytes {y, x} (x low).
__device__ __forceinline__ unsigned pie_byte_perm(unsigned x, unsigned y, unsigned s) {
    return __builtin_amdgcn_perm(y, x, ((s & 0x7u) | ((s >> 4 & 0x7u) << 8) | ((s >> 8 & 0x7u) << 16) |
                                        ((s >> 12 & 0x7u) << 24)));
}
#define __byte_perm pie_byte_perm

// A generic pointer into LDS is the LDS aperture base plus the LDS offset,
// so the offset is its low word, as CUDA's shared window is.
__device__ __forceinline__ unsigned pie_cvta_generic_to_shared(const void* p) {
    return static_cast<unsigned>(reinterpret_cast<unsigned long long>(p));
}
#define __cvta_generic_to_shared pie_cvta_generic_to_shared

// ---- the PTX the carried units wrapped in helpers -------------------------
//
// Each answers what the instruction computes for the 32-lane CUDA warp the
// caller stands in. These are the portable forms, built on the warp
// shuffles above; the MFMA kernels that replace the hot callers come after.

namespace pie_hip {

// `ldmatrix.sync.aligned.m8n8.x4.shared.b16`: lanes 8i..8i+7 name the eight
// 16-byte rows of matrix i, and lane l receives, from each matrix i, the two
// b16 at row l/4, columns 2(l%4) and 2(l%4)+1.
__device__ __forceinline__ void ldmatrix_x4(unsigned (&r)[4], const void* row) {
    const int lane = static_cast<int>(threadIdx.x & 31u);
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const char* src = pie_shfl_sync(~0ull, static_cast<const char*>(row), 8 * i + (lane >> 2));
        r[i] = *reinterpret_cast<const unsigned*>(src + 4 * (lane & 3));
    }
}

__device__ __forceinline__ float bf16_lo(unsigned w) { return __uint_as_float(w << 16); }
__device__ __forceinline__ float bf16_hi(unsigned w) { return __uint_as_float(w & 0xffff0000u); }

// `mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32`, d = a·b + c, with
// the PTX fragment map (g = lane/4, t = lane%4):
//   a[0] = A[g][2t..2t+1]    a[1] = A[g+8][2t..]   a[2] = A[g][2t+8..]   a[3] = A[g+8][2t+8..]
//   b[0] = B[2t..2t+1][g]    b[1] = B[2t+8..2t+9][g]
//   d[0..1] = D[g][2t..2t+1] d[2..3] = D[g+8][2t..2t+1]
// Lane (g, t) gathers row g's and g+8's A words from lanes 4g..4g+3 and the
// B words of columns 2t, 2t+1 from lanes 8t..8t+3 and 8t+4..8t+7.
__device__ __forceinline__ void mma_m16n8k16_bf16(float (&d)[4], const unsigned (&a)[4],
                                                  const unsigned (&b)[2], const float (&c)[4]) {
    const int lane = static_cast<int>(threadIdx.x & 31u);
    const int g = lane >> 2;
    const int t = lane & 3;
    float acc[4] = {c[0], c[1], c[2], c[3]};
#pragma unroll
    for (int src = 0; src < 4; ++src) {
        // k = 2src, 2src+1 (words 0/1) and 2src+8, 2src+9 (words 2/3).
        const unsigned a_top_lo = pie_shfl_sync(~0ull, a[0], 4 * g + src);
        const unsigned a_bot_lo = pie_shfl_sync(~0ull, a[1], 4 * g + src);
        const unsigned a_top_hi = pie_shfl_sync(~0ull, a[2], 4 * g + src);
        const unsigned a_bot_hi = pie_shfl_sync(~0ull, a[3], 4 * g + src);
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const int col_lane = 4 * (2 * t + j) + src;
            const unsigned b_lo = pie_shfl_sync(~0ull, b[0], col_lane);
            const unsigned b_hi = pie_shfl_sync(~0ull, b[1], col_lane);
            float top = acc[j];
            top = fmaf(bf16_lo(a_top_lo), bf16_lo(b_lo), top);
            top = fmaf(bf16_hi(a_top_lo), bf16_hi(b_lo), top);
            top = fmaf(bf16_lo(a_top_hi), bf16_lo(b_hi), top);
            top = fmaf(bf16_hi(a_top_hi), bf16_hi(b_hi), top);
            acc[j] = top;
            float bot = acc[2 + j];
            bot = fmaf(bf16_lo(a_bot_lo), bf16_lo(b_lo), bot);
            bot = fmaf(bf16_hi(a_bot_lo), bf16_hi(b_lo), bot);
            bot = fmaf(bf16_lo(a_bot_hi), bf16_lo(b_hi), bot);
            bot = fmaf(bf16_hi(a_bot_hi), bf16_hi(b_hi), bot);
            acc[2 + j] = bot;
        }
    }
    d[0] = acc[0];
    d[1] = acc[1];
    d[2] = acc[2];
    d[3] = acc[3];
}

// `cp.async.cg.shared.global [dst], [src], 16, bytes`: `bytes` from `src`,
// the rest of the 16 zero. Done before the call returns (see cuda_pipeline.h).
__device__ __forceinline__ void cp_async16(void* dst, const void* src, int bytes) {
    if (bytes >= 16) {
        *static_cast<uint4*>(dst) = *static_cast<const uint4*>(src);
        return;
    }
    unsigned char* out = static_cast<unsigned char*>(dst);
    const unsigned char* in = static_cast<const unsigned char*>(src);
#pragma unroll
    for (int i = 0; i < 16; ++i) out[i] = i < bytes ? in[i] : 0;
}

// `lop3.b32 d, a, b, c, lut`: bit i of `lut` is F at (a, b, c) = (i&4, i&2, i&1).
template <int lut>
__device__ __forceinline__ int lop3(int a, int b, int c) {
    int r = 0;
#pragma unroll
    for (int i = 0; i < 8; ++i) {
        if ((lut >> i) & 1) r |= ((i & 4) ? a : ~a) & ((i & 2) ? b : ~b) & ((i & 1) ? c : ~c);
    }
    return r;
}

}  // namespace pie_hip
