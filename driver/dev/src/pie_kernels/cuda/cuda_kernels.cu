// Batched random-matmul / generate kernels.
//
// Computes (without materializing W in the matmul path):
//   y[b, o]    = sum_i x[b, i] * S[i, o] * Phi(seed=seeds[b], i, o_global)
//   W[b, i, o] = S[i, o] * Phi(seed=seeds[b], i, o_global)
// where o_global = o + col_offset and Phi is a seeded standard-normal RNG.
//
// RNG scheme (shared by matmul + generate so they agree element-wise):
//   Pack 4 consecutive i-rows per Philox-4x32 call.
//     counter = i_pack * global_cols + o_global
//     key     = seed
//   The 4 uint32 outputs produce 4 normals via the chosen transform; element
//   i = 4*i_pack + r uses the r-th normal.
//
// Compile-time knobs (see cuda_impl.py for the variants we build):
//   -DRNG_METHOD=0   Box-Muller  (default)
//   -DRNG_METHOD=1   Probit via normcdfinvf
//   -DRNG_METHOD=2   Ziggurat (Marsaglia-Tsang, shmem-resident LUT)
//   -DPHILOX_ROUNDS  Philox round count (default 10; 7 passes BigCrush)
//
// See stats.py for the statistical validation harness.

#include <torch/extension.h>
#include <ATen/cuda/CUDAContext.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>
#include <cstdint>

#ifndef RNG_METHOD
#define RNG_METHOD 0
#endif

#ifndef PHILOX_ROUNDS
#define PHILOX_ROUNDS 10
#endif

namespace {

template <typename T> struct type_tag { using type = T; };

constexpr uint32_t PHILOX_M0 = 0xD2511F53U;
constexpr uint32_t PHILOX_M1 = 0xCD9E8D57U;
constexpr uint32_t PHILOX_W0 = 0x9E3779B9U;
constexpr uint32_t PHILOX_W1 = 0xBB67AE85U;

struct Normals4 { float n0, n1, n2, n3; };

// atomicAdd that accepts fp32 rhs, dispatched on output dtype.
template<typename T>
__device__ __forceinline__ void atomic_add_out(T* addr, float val);

template<>
__device__ __forceinline__ void atomic_add_out<float>(float* addr, float val) {
    atomicAdd(addr, val);
}

template<>
__device__ __forceinline__ void atomic_add_out<at::Half>(at::Half* addr, float val) {
    atomicAdd(reinterpret_cast<__half*>(addr), __float2half(val));
}

template<>
__device__ __forceinline__ void atomic_add_out<at::BFloat16>(at::BFloat16* addr, float val) {
    atomicAdd(reinterpret_cast<__nv_bfloat16*>(addr), __float2bfloat16(val));
}

// Philox-4x32 rounds, templated on round count. Returns 4 uint32s.
template<int Rounds>
__device__ __forceinline__ void philox_rounds(
    uint32_t offset, uint32_t seed,
    uint32_t& c0, uint32_t& c1, uint32_t& c2, uint32_t& c3
) {
    c0 = offset; c1 = 0u; c2 = 0u; c3 = 0u;
    uint32_t k0 = seed, k1 = 0u;
    #pragma unroll
    for (int r = 0; r < Rounds; r++) {
        uint32_t lo0 = PHILOX_M0 * c0;
        uint32_t hi0 = __umulhi(PHILOX_M0, c0);
        uint32_t lo1 = PHILOX_M1 * c2;
        uint32_t hi1 = __umulhi(PHILOX_M1, c2);
        uint32_t n0 = hi1 ^ c1 ^ k0;
        uint32_t n1 = lo1;
        uint32_t n2 = hi0 ^ c3 ^ k1;
        uint32_t n3 = lo0;
        c0 = n0; c1 = n1; c2 = n2; c3 = n3;
        k0 += PHILOX_W0;
        k1 += PHILOX_W1;
    }
}

#if RNG_METHOD == 2
#include "ziggurat_tables.h"

// All Ziggurat device functions take LUT pointers; each kernel declares
// block-scope __shared__ arrays and passes them in. Dodges the Ampere
// constant-cache per-distinct-address serialization when `iz = u & 127`
// varies across lanes of a warp.
struct ZigLUT {
    const uint32_t* __restrict__ kn;
    const float*    __restrict__ wn;
    const float*    __restrict__ fn;
};

// Call in kernel prologue. The caller must have declared __shared__ arrays.
__device__ __forceinline__ void load_zig_lut_to(uint32_t* kn_sh, float* wn_sh, float* fn_sh,
                                                int block_threads) {
    int tid = threadIdx.x + threadIdx.y * blockDim.x;
    for (int i = tid; i < 128; i += block_threads) {
        kn_sh[i] = zig_kn[i];
        wn_sh[i] = zig_wn[i];
        fn_sh[i] = zig_fn[i];
    }
    __syncthreads();
}

// Ziggurat nfix slow path. ~3% of samples. A fresh Philox call under a
// disjoint counter namespace keeps the uniforms un-correlated with the
// fast-path draws.
__device__ __forceinline__ float ziggurat_nfix(
    int32_t hz, uint32_t iz, uint32_t seed, uint32_t base_offset, ZigLUT L
) {
    const float r = 3.442619855899f;
    const float inv_r = 0.29047645161474317f;  // 1/r
    uint32_t nx_ctr = base_offset ^ 0x80000000u;

    #pragma unroll 1
    for (int iter = 0; iter < 4; iter++) {
        uint32_t e0, e1, e2, e3;
        philox_rounds<PHILOX_ROUNDS>(nx_ctr + (uint32_t)iter, seed, e0, e1, e2, e3);
        constexpr float INV_2_31 = 4.656612873077393e-10f;

        if (iz == 0) {
            float u1 = fmaxf((float)(e0 & 0x7FFFFFFFu) * INV_2_31, 1.0e-7f);
            float u2 = fmaxf((float)(e1 & 0x7FFFFFFFu) * INV_2_31, 1.0e-7f);
            float x = -__logf(u1) * inv_r;
            float y = -__logf(u2);
            if (y + y >= x * x) {
                return (hz > 0) ? (r + x) : -(r + x);
            }
            continue;
        }
        float x = (float)hz * L.wn[iz];
        float u = (float)(e0 & 0x7FFFFFFFu) * INV_2_31;
        float y = L.fn[iz] + u * (L.fn[iz - 1] - L.fn[iz]);
        if (y < __expf(-0.5f * x * x)) {
            return x;
        }
        hz = (int32_t)e1;
        iz = e1 & 127u;
        uint32_t abs_hz = (uint32_t)(hz < 0 ? -hz : hz);
        if (abs_hz < L.kn[iz]) {
            return (float)hz * L.wn[iz];
        }
    }
    // <1e-6 residual: fall back to Box-Muller.
    uint32_t e0, e1, e2, e3;
    philox_rounds<PHILOX_ROUNDS>(base_offset ^ 0xC0000000u, seed, e0, e1, e2, e3);
    constexpr float INV_2_31 = 4.656612873077393e-10f;
    float u1 = fmaxf((float)(e0 & 0x7FFFFFFFu) * INV_2_31, 1.0e-7f);
    float u2 = (float)(e1 & 0x7FFFFFFFu) * INV_2_31;
    float rr = sqrtf(-2.0f * __logf(u1));
    float c, s;
    __sincosf(6.283185307179586f * u2, &s, &c);
    return rr * c;
}

__device__ __forceinline__ float zig_from_uint(uint32_t u, uint32_t seed,
                                               uint32_t base_offset, ZigLUT L) {
    int32_t hz = (int32_t)u;
    uint32_t iz = u & 127u;
    uint32_t abs_hz = (uint32_t)(hz < 0 ? -hz : hz);
    if (abs_hz < L.kn[iz]) {
        return (float)hz * L.wn[iz];
    }
    return ziggurat_nfix(hz, iz, seed, base_offset, L);
}
#endif  // RNG_METHOD == 2

#if RNG_METHOD == 2
    #define ZIG_LUT_PARAM , ZigLUT zig_lut
    #define ZIG_LUT_ARG   , zig_lut
#else
    #define ZIG_LUT_PARAM
    #define ZIG_LUT_ARG
#endif

// Declare block-scope shared LUTs + load them. Put at the top of each kernel.
#if RNG_METHOD == 2
    #define ZIG_KERNEL_PROLOGUE(BLOCK_THREADS) \
        __shared__ uint32_t _zig_kn_sh[128]; \
        __shared__ float    _zig_wn_sh[128]; \
        __shared__ float    _zig_fn_sh[128]; \
        load_zig_lut_to(_zig_kn_sh, _zig_wn_sh, _zig_fn_sh, (BLOCK_THREADS)); \
        ZigLUT zig_lut{_zig_kn_sh, _zig_wn_sh, _zig_fn_sh};
#else
    #define ZIG_KERNEL_PROLOGUE(BLOCK_THREADS)
#endif

__device__ __forceinline__ Normals4 philox_4_normals(uint32_t seed, uint32_t offset ZIG_LUT_PARAM) {
    uint32_t c0, c1, c2, c3;
    philox_rounds<PHILOX_ROUNDS>(offset, seed, c0, c1, c2, c3);

#if RNG_METHOD == 0
    // Box-Muller: two B-M pairs from 4 uniforms → 4 normals.
    constexpr float INV_2_31 = 4.656612873077393e-10f;
    float u0 = (float)(c0 & 0x7FFFFFFFU) * INV_2_31;
    float u1 = (float)(c1 & 0x7FFFFFFFU) * INV_2_31;
    float u2 = (float)(c2 & 0x7FFFFFFFU) * INV_2_31;
    float u3 = (float)(c3 & 0x7FFFFFFFU) * INV_2_31;
    u0 = fmaxf(u0, 1.0e-7f);
    u2 = fmaxf(u2, 1.0e-7f);
    float r_a = sqrtf(-2.0f * __logf(u0));
    float r_b = sqrtf(-2.0f * __logf(u2));
    float s_a, c_a, s_b, c_b;
    __sincosf(6.283185307179586f * u1, &s_a, &c_a);
    __sincosf(6.283185307179586f * u3, &s_b, &c_b);
    Normals4 out{ r_a * c_a, r_a * s_a, r_b * c_b, r_b * s_b };
    return out;
#elif RNG_METHOD == 1
    // Probit: normcdfinvf on each uniform. 23-bit mantissa trick keeps u
    // strictly in (2^-24, 1-2^-24) with no branches and no Inf.
    constexpr float INV_2_23 = 1.1920928955078125e-07f;
    float u0 = ((float)(c0 >> 9) + 0.5f) * INV_2_23;
    float u1 = ((float)(c1 >> 9) + 0.5f) * INV_2_23;
    float u2 = ((float)(c2 >> 9) + 0.5f) * INV_2_23;
    float u3 = ((float)(c3 >> 9) + 0.5f) * INV_2_23;
    Normals4 out{ normcdfinvf(u0), normcdfinvf(u1), normcdfinvf(u2), normcdfinvf(u3) };
    return out;
#elif RNG_METHOD == 2
    Normals4 out{
        zig_from_uint(c0, seed, offset, zig_lut),
        zig_from_uint(c1, seed, offset, zig_lut),
        zig_from_uint(c2, seed, offset, zig_lut),
        zig_from_uint(c3, seed, offset, zig_lut)
    };
    return out;
#else
    #error "unknown RNG_METHOD"
#endif
}

// ============================================================================
//  Matmul kernel.
// ============================================================================
//
// Grid:   (cdiv(O, BN), B, k_split)
// Block:  BN threads, each thread → one output column o = n_tile*BN + t.
//         Thread iterates its K slice in steps of 4 (one Philox call = 4 normals).

template<typename XT, typename ST, typename OT, int BN, bool HAS_MEAN>
__global__ void __launch_bounds__(BN, 4) batched_randn_matmul_kernel(
    const XT* __restrict__ x,
    const int64_t* __restrict__ seeds,
    const ST* __restrict__ S,
    const ST* __restrict__ W_mean,     // optional; iff HAS_MEAN
    int stride_Wi, int stride_Wo,      // strides for W_mean
    OT* __restrict__ y,
    int B, int I, int O,
    int stride_xb, int stride_xi,
    int stride_Si, int stride_So,
    int stride_yb, int stride_yo,
    int col_offset, int global_cols,
    int k_split, int k_pack_chunk,
    int64_t seed_offset,
    float alpha, float beta
) {
    ZIG_KERNEL_PROLOGUE(BN);

    int b       = blockIdx.y;
    int n_tile  = blockIdx.x;
    int k_tile  = blockIdx.z;
    int t       = threadIdx.x;
    int o_local = n_tile * BN + t;

    int64_t seed_val = seeds[b];
    if (seed_val == 0) {
        if (k_split == 1 && k_tile == 0 && o_local < O) {
            float y_old = (beta != 0.0f) ? (float)y[b * stride_yb + o_local * stride_yo] : 0.0f;
            y[b * stride_yb + o_local * stride_yo] = (OT)(beta * y_old);
        }
        return;
    }
    uint32_t seed = (uint32_t)(seed_val + seed_offset);

    // Each k_tile spans [kp_start*4, kp_end*4) rows.
    const int I_packs = (I + 3) / 4;
    const int kp_start = k_tile * k_pack_chunk;
    const int kp_end_unclamped = kp_start + k_pack_chunk;
    const int kp_end = kp_end_unclamped < I_packs ? kp_end_unclamped : I_packs;

    const int o_abs = o_local + col_offset;

    // Shared memory for x row chunks. One chunk of 4 floats per thread0 load.
    constexpr int X_PACK = 32;  // packs of 4 → 128 x-values held in shmem
    constexpr int X_VALS = X_PACK * 4;
    __shared__ float x_sh[X_VALS];

    float acc = 0.0f;
    const int x_base = b * stride_xb;
    const bool active = (o_local < O);

    // ALL threads participate in the cooperative x-load and __syncthreads,
    // even threads with o_local >= O. Only accumulation is gated on `active`.
    int kp = kp_start;
    while (kp < kp_end) {
        int chunk = (kp_end - kp) < X_PACK ? (kp_end - kp) : X_PACK;
        int vals_to_load = chunk * 4;

        __syncthreads();
        for (int j = t; j < vals_to_load; j += BN) {
            int row = kp * 4 + j;
            float xv = 0.0f;
            if (row < I) xv = (float)x[x_base + row * stride_xi];
            x_sh[j] = xv;
        }
        __syncthreads();

        if (active) {
            #pragma unroll 4
            for (int pp = 0; pp < chunk; pp++) {
                int this_kp = kp + pp;
                int row0 = this_kp * 4;

                uint32_t offset = (uint32_t)this_kp * (uint32_t)global_cols + (uint32_t)o_abs;
                Normals4 w = philox_4_normals(seed, offset ZIG_LUT_ARG);

                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                if (row0     < I) s0 = (float)S[(row0    ) * stride_Si + o_local * stride_So];
                if (row0 + 1 < I) s1 = (float)S[(row0 + 1) * stride_Si + o_local * stride_So];
                if (row0 + 2 < I) s2 = (float)S[(row0 + 2) * stride_Si + o_local * stride_So];
                if (row0 + 3 < I) s3 = (float)S[(row0 + 3) * stride_Si + o_local * stride_So];

                int base = pp * 4;
                if constexpr (HAS_MEAN) {
                    float m0 = 0.0f, m1 = 0.0f, m2 = 0.0f, m3 = 0.0f;
                    if (row0     < I) m0 = (float)W_mean[(row0    ) * stride_Wi + o_local * stride_Wo];
                    if (row0 + 1 < I) m1 = (float)W_mean[(row0 + 1) * stride_Wi + o_local * stride_Wo];
                    if (row0 + 2 < I) m2 = (float)W_mean[(row0 + 2) * stride_Wi + o_local * stride_Wo];
                    if (row0 + 3 < I) m3 = (float)W_mean[(row0 + 3) * stride_Wi + o_local * stride_Wo];
                    acc += x_sh[base + 0] * (m0 + s0 * w.n0);
                    acc += x_sh[base + 1] * (m1 + s1 * w.n1);
                    acc += x_sh[base + 2] * (m2 + s2 * w.n2);
                    acc += x_sh[base + 3] * (m3 + s3 * w.n3);
                } else {
                    acc += x_sh[base + 0] * s0 * w.n0;
                    acc += x_sh[base + 1] * s1 * w.n1;
                    acc += x_sh[base + 2] * s2 * w.n2;
                    acc += x_sh[base + 3] * s3 * w.n3;
                }
            }
        }
        kp += chunk;
    }

    if (active) {
        if (k_split == 1) {
            float y_old = (beta != 0.0f) ? (float)y[b * stride_yb + o_local * stride_yo] : 0.0f;
            y[b * stride_yb + o_local * stride_yo] = (OT)(alpha * acc + beta * y_old);
        } else {
            atomic_add_out<OT>(&y[b * stride_yb + o_local * stride_yo], alpha * acc);
        }
    }
}

// ============================================================================
//  Matmul kernel specialized for small O (wide I).
// ============================================================================
//
// One-warp block (32 lanes). BN output columns × BK K-reduction lanes fit
// in a single warp (BN * BK == 32).
//
// Lane layout: lane = tx + ty * BN, where tx ∈ [0, BN), ty ∈ [0, BK).
//
// Grid:   (cdiv(O, BN), B, k_split)

template<typename XT, typename ST, typename OT, int BN, int BK, bool HAS_MEAN>
__global__ void __launch_bounds__(32, 8) batched_randn_matmul_wide_kernel(
    const XT* __restrict__ x,
    const int64_t* __restrict__ seeds,
    const ST* __restrict__ S,
    const ST* __restrict__ W_mean,
    int stride_Wi, int stride_Wo,
    OT* __restrict__ y,
    int B, int I, int O,
    int stride_xb, int stride_xi,
    int stride_Si, int stride_So,
    int stride_yb, int stride_yo,
    int col_offset, int global_cols,
    int k_split, int kp_chunk,
    int64_t seed_offset,
    float alpha, float beta
) {
    static_assert(BN * BK == 32, "BN*BK must equal 32 (one warp)");
    ZIG_KERNEL_PROLOGUE(32);

    const int b        = blockIdx.y;
    const int n_tile   = blockIdx.x;
    const int k_tile   = blockIdx.z;
    const int lane     = threadIdx.x;
    const int tx       = lane % BN;
    const int ty       = lane / BN;
    const int o_local  = n_tile * BN + tx;

    int64_t seed_val = seeds[b];
    if (seed_val == 0) {
        if (k_split == 1 && k_tile == 0 && ty == 0 && o_local < O) {
            float y_old = (beta != 0.0f) ? (float)y[b * stride_yb + o_local * stride_yo] : 0.0f;
            y[b * stride_yb + o_local * stride_yo] = (OT)(beta * y_old);
        }
        return;
    }
    uint32_t seed = (uint32_t)(seed_val + seed_offset);

    const int I_packs = (I + 3) / 4;
    const int kp_start = k_tile * kp_chunk;
    const int kp_end_un = kp_start + kp_chunk;
    const int kp_end = kp_end_un < I_packs ? kp_end_un : I_packs;

    const int o_abs = o_local + col_offset;
    const int x_base = b * stride_xb;

    float acc = 0.0f;
    const bool active = (o_local < O);

    if (active) {
        for (int kp = kp_start + ty; kp < kp_end; kp += BK) {
            int row0 = kp * 4;

            uint32_t offset = (uint32_t)kp * (uint32_t)global_cols + (uint32_t)o_abs;
            Normals4 w = philox_4_normals(seed, offset ZIG_LUT_ARG);

            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            float x0 = 0.0f, x1 = 0.0f, x2 = 0.0f, x3 = 0.0f;
            if (row0     < I) { s0 = (float)S[(row0    ) * stride_Si + o_local * stride_So];
                                x0 = (float)x[x_base + (row0    ) * stride_xi]; }
            if (row0 + 1 < I) { s1 = (float)S[(row0 + 1) * stride_Si + o_local * stride_So];
                                x1 = (float)x[x_base + (row0 + 1) * stride_xi]; }
            if (row0 + 2 < I) { s2 = (float)S[(row0 + 2) * stride_Si + o_local * stride_So];
                                x2 = (float)x[x_base + (row0 + 2) * stride_xi]; }
            if (row0 + 3 < I) { s3 = (float)S[(row0 + 3) * stride_Si + o_local * stride_So];
                                x3 = (float)x[x_base + (row0 + 3) * stride_xi]; }

            if constexpr (HAS_MEAN) {
                float m0 = 0.0f, m1 = 0.0f, m2 = 0.0f, m3 = 0.0f;
                if (row0     < I) m0 = (float)W_mean[(row0    ) * stride_Wi + o_local * stride_Wo];
                if (row0 + 1 < I) m1 = (float)W_mean[(row0 + 1) * stride_Wi + o_local * stride_Wo];
                if (row0 + 2 < I) m2 = (float)W_mean[(row0 + 2) * stride_Wi + o_local * stride_Wo];
                if (row0 + 3 < I) m3 = (float)W_mean[(row0 + 3) * stride_Wi + o_local * stride_Wo];
                acc = fmaf(x0, m0 + s0 * w.n0, acc);
                acc = fmaf(x1, m1 + s1 * w.n1, acc);
                acc = fmaf(x2, m2 + s2 * w.n2, acc);
                acc = fmaf(x3, m3 + s3 * w.n3, acc);
            } else {
                acc = fmaf(x0 * s0, w.n0, acc);
                acc = fmaf(x1 * s1, w.n1, acc);
                acc = fmaf(x2 * s2, w.n2, acc);
                acc = fmaf(x3 * s3, w.n3, acc);
            }
        }
    }

    // Reduce across ty dimension (stride BN in lane space).
    #pragma unroll
    for (int step = 1; step < BK; step <<= 1) {
        acc += __shfl_xor_sync(0xFFFFFFFFu, acc, step * BN);
    }

    if (ty == 0 && active) {
        if (k_split == 1) {
            float y_old = (beta != 0.0f) ? (float)y[b * stride_yb + o_local * stride_yo] : 0.0f;
            y[b * stride_yb + o_local * stride_yo] = (OT)(alpha * acc + beta * y_old);
        } else {
            atomic_add_out<OT>(&y[b * stride_yb + o_local * stride_yo], alpha * acc);
        }
    }
}

// ============================================================================
//  Matmul kernel, sectioned (Q/K/V-fused) for LoRA DOWN projection.
// ============================================================================
//
// Same as batched_randn_matmul_kernel but each output column o is assigned to
// a section based on its position, and each section has its own seed offset.
// This fuses the 3 Q/K/V noise calls in LoRA DOWN into a single kernel launch:
//
//   S:       (I, section_widths[0] + section_widths[1] + section_widths[2])
//   output:  (B, same O)
//
// For N sections (N ≤ 4), section k covers columns [Σ_{j<k} w_j, Σ_{j≤k} w_j).

struct Sections {
    int widths[4];      // unused trailing entries: 0
    int64_t offsets[4]; // per-section seed offset added to seeds[b]
    int n;              // number of active sections (1..4)
};

template<typename XT, typename ST, typename OT, int BN, bool HAS_MEAN>
__global__ void __launch_bounds__(BN, 4) batched_randn_matmul_sectioned_kernel(
    const XT* __restrict__ x,
    const int64_t* __restrict__ seeds,
    const ST* __restrict__ S,
    const ST* __restrict__ W_mean,
    int stride_Wi, int stride_Wo,
    OT* __restrict__ y,
    int B, int I, int O,
    int stride_xb, int stride_xi,
    int stride_Si, int stride_So,
    int stride_yb, int stride_yo,
    int col_offset, int global_cols,
    int k_split, int k_pack_chunk,
    Sections sec,
    float alpha, float beta
) {
    ZIG_KERNEL_PROLOGUE(BN);

    int b       = blockIdx.y;
    int n_tile  = blockIdx.x;
    int k_tile  = blockIdx.z;
    int t       = threadIdx.x;
    int o_local = n_tile * BN + t;

    // Find section for this thread's output column. Each section's Philox
    // counter matches a standalone call of batched_randn_matmul with that
    // section's S slice — i.e. counter = i_pack * section_width + o_in_section.
    int sec_idx = 0;
    int boundary = 0;
    int sec_width = sec.widths[0];
    #pragma unroll
    for (int s = 0; s < 4; s++) {
        if (s < sec.n && o_local >= boundary + sec.widths[s]) {
            boundary += sec.widths[s];
            sec_idx = s + 1;
            sec_width = (s + 1 < sec.n) ? sec.widths[s + 1] : 0;
        }
    }
    int64_t seed_offset_for_o = (sec_idx < sec.n) ? sec.offsets[sec_idx] : 0;
    int o_in_section = o_local - boundary;

    int64_t seed_val = seeds[b];
    if (seed_val == 0) {
        if (k_split == 1 && k_tile == 0 && o_local < O) {
            float y_old = (beta != 0.0f) ? (float)y[b * stride_yb + o_local * stride_yo] : 0.0f;
            y[b * stride_yb + o_local * stride_yo] = (OT)(beta * y_old);
        }
        return;
    }
    uint32_t seed = (uint32_t)(seed_val + seed_offset_for_o);

    const int I_packs = (I + 3) / 4;
    const int kp_start = k_tile * k_pack_chunk;
    const int kp_end_unclamped = kp_start + k_pack_chunk;
    const int kp_end = kp_end_unclamped < I_packs ? kp_end_unclamped : I_packs;

    // col_offset is applied within-section to match the standalone-call semantics.
    const int o_abs = o_in_section + col_offset;
    const uint32_t section_global_cols = (uint32_t)sec_width;

    constexpr int X_PACK = 32;
    constexpr int X_VALS = X_PACK * 4;
    __shared__ float x_sh[X_VALS];

    float acc = 0.0f;
    const int x_base = b * stride_xb;
    const bool active = (o_local < O);

    int kp = kp_start;
    while (kp < kp_end) {
        int chunk = (kp_end - kp) < X_PACK ? (kp_end - kp) : X_PACK;
        int vals_to_load = chunk * 4;

        __syncthreads();
        for (int j = t; j < vals_to_load; j += BN) {
            int row = kp * 4 + j;
            float xv = 0.0f;
            if (row < I) xv = (float)x[x_base + row * stride_xi];
            x_sh[j] = xv;
        }
        __syncthreads();

        if (active) {
            #pragma unroll 4
            for (int pp = 0; pp < chunk; pp++) {
                int this_kp = kp + pp;
                int row0 = this_kp * 4;

                uint32_t offset = (uint32_t)this_kp * section_global_cols + (uint32_t)o_abs;
                Normals4 w = philox_4_normals(seed, offset ZIG_LUT_ARG);

                float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
                if (row0     < I) s0 = (float)S[(row0    ) * stride_Si + o_local * stride_So];
                if (row0 + 1 < I) s1 = (float)S[(row0 + 1) * stride_Si + o_local * stride_So];
                if (row0 + 2 < I) s2 = (float)S[(row0 + 2) * stride_Si + o_local * stride_So];
                if (row0 + 3 < I) s3 = (float)S[(row0 + 3) * stride_Si + o_local * stride_So];

                int base = pp * 4;
                if constexpr (HAS_MEAN) {
                    float m0 = 0.0f, m1 = 0.0f, m2 = 0.0f, m3 = 0.0f;
                    if (row0     < I) m0 = (float)W_mean[(row0    ) * stride_Wi + o_local * stride_Wo];
                    if (row0 + 1 < I) m1 = (float)W_mean[(row0 + 1) * stride_Wi + o_local * stride_Wo];
                    if (row0 + 2 < I) m2 = (float)W_mean[(row0 + 2) * stride_Wi + o_local * stride_Wo];
                    if (row0 + 3 < I) m3 = (float)W_mean[(row0 + 3) * stride_Wi + o_local * stride_Wo];
                    acc += x_sh[base + 0] * (m0 + s0 * w.n0);
                    acc += x_sh[base + 1] * (m1 + s1 * w.n1);
                    acc += x_sh[base + 2] * (m2 + s2 * w.n2);
                    acc += x_sh[base + 3] * (m3 + s3 * w.n3);
                } else {
                    acc += x_sh[base + 0] * s0 * w.n0;
                    acc += x_sh[base + 1] * s1 * w.n1;
                    acc += x_sh[base + 2] * s2 * w.n2;
                    acc += x_sh[base + 3] * s3 * w.n3;
                }
            }
        }
        kp += chunk;
    }

    if (active) {
        if (k_split == 1) {
            float y_old = (beta != 0.0f) ? (float)y[b * stride_yb + o_local * stride_yo] : 0.0f;
            y[b * stride_yb + o_local * stride_yo] = (OT)(alpha * acc + beta * y_old);
        } else {
            atomic_add_out<OT>(&y[b * stride_yb + o_local * stride_yo], alpha * acc);
        }
    }
}

// ============================================================================
//  Multi-input sectioned matmul kernel.
//
//  Like batched_randn_matmul_sectioned, but each output section reads from a
//  DIFFERENT slice of the input. The reduction dim is shared (`x_width`)
//  across sections; each section has its own `x_start` column offset into
//  the packed input.
//
//  This collapses the 3 UP matmul calls in the LoRA adapter into a single
//  launch: x_packed = [d_q | d_k | d_v] (B, 3·rank), output sections
//  Q/K/V each pull from their respective rank-wide input slice.
//
//  The full x row fits comfortably in shared memory for typical LoRA
//  configurations (rank · num_sections ≤ 64), so all sections share one
//  cooperative load.
// ============================================================================

struct MultiInputSections {
    int x_starts[4];     // input col offset per section
    int widths[4];       // output col width per section (used as noise's section_global_cols)
    int64_t offsets[4];  // seed offset per section
    int n;
};

template<typename XT, typename ST, typename OT, int BN, bool HAS_MEAN>
__global__ void __launch_bounds__(BN, 4)
batched_randn_matmul_multi_input_sectioned_kernel(
    const XT* __restrict__ x,         // (B, sum_in)
    const int64_t* __restrict__ seeds,
    const ST* __restrict__ S,         // (x_width, sum_out)
    const ST* __restrict__ W_mean,    // optional, same shape as S
    int stride_Wi, int stride_Wo,
    OT* __restrict__ y,               // (B, sum_out)
    int B, int sum_in, int x_width, int O,
    int stride_xb, int stride_xi,
    int stride_Si, int stride_So,
    int stride_yb, int stride_yo,
    MultiInputSections sec,
    float alpha, float beta
) {
    ZIG_KERNEL_PROLOGUE(BN);

    int b       = blockIdx.y;
    int n_tile  = blockIdx.x;
    int t       = threadIdx.x;
    int o_local = n_tile * BN + t;

    // Find which output section this thread is in. The loop walks sections
    // in order; once we find one whose end exceeds o_local, we stop.
    // (`done` flag is necessary: with unequal section widths, the original
    // accumulator-style check `o_local >= boundary + widths[s]` would
    // erroneously match later sections when boundary hasn't yet advanced.)
    int sec_idx = 0;
    int boundary = 0;
    int sec_width = sec.widths[0];
    int sec_x_start = sec.x_starts[0];
    int64_t sec_seed_offset = sec.offsets[0];
    bool done = false;
    #pragma unroll
    for (int s = 0; s < 4; s++) {
        if (s < sec.n && !done) {
            int next_boundary = boundary + sec.widths[s];
            if (o_local < next_boundary) {
                done = true;
            } else {
                boundary = next_boundary;
                sec_idx = s + 1;
                sec_width = (s + 1 < sec.n) ? sec.widths[s + 1] : 0;
                sec_x_start = (s + 1 < sec.n) ? sec.x_starts[s + 1] : 0;
                sec_seed_offset = (s + 1 < sec.n) ? sec.offsets[s + 1] : 0;
            }
        }
    }
    int o_in_section = o_local - boundary;

    int64_t seed_val = seeds[b];
    if (seed_val == 0) {
        if (o_local < O) {
            float y_old = (beta != 0.0f) ? (float)y[b * stride_yb + o_local * stride_yo] : 0.0f;
            y[b * stride_yb + o_local * stride_yo] = (OT)(beta * y_old);
        }
        return;
    }
    uint32_t seed = (uint32_t)(seed_val + sec_seed_offset);

    const uint32_t section_global_cols = (uint32_t)sec_width;
    const int o_abs = o_in_section;  // col_offset=0 (multi-input is single-GPU only)

    // Cooperative load: all of x[b, 0:sum_in] into shared memory. Capped at
    // 128 floats which covers up to rank=32, num_sections=4.
    constexpr int X_MAX = 128;
    __shared__ float x_sh[X_MAX];
    const int x_base = b * stride_xb;
    for (int j = t; j < sum_in; j += BN) {
        x_sh[j] = (float)x[x_base + j * stride_xi];
    }
    __syncthreads();

    const bool active = (o_local < O);
    const int packs = (x_width + 3) / 4;

    float acc = 0.0f;
    if (active) {
        for (int kp = 0; kp < packs; kp++) {
            int row0 = kp * 4;

            uint32_t offset = (uint32_t)kp * section_global_cols + (uint32_t)o_abs;
            Normals4 w = philox_4_normals(seed, offset ZIG_LUT_ARG);

            float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
            if (row0     < x_width) s0 = (float)S[(row0    ) * stride_Si + o_local * stride_So];
            if (row0 + 1 < x_width) s1 = (float)S[(row0 + 1) * stride_Si + o_local * stride_So];
            if (row0 + 2 < x_width) s2 = (float)S[(row0 + 2) * stride_Si + o_local * stride_So];
            if (row0 + 3 < x_width) s3 = (float)S[(row0 + 3) * stride_Si + o_local * stride_So];

            float x0 = 0.0f, x1 = 0.0f, x2 = 0.0f, x3 = 0.0f;
            if (row0     < x_width) x0 = x_sh[sec_x_start + row0    ];
            if (row0 + 1 < x_width) x1 = x_sh[sec_x_start + row0 + 1];
            if (row0 + 2 < x_width) x2 = x_sh[sec_x_start + row0 + 2];
            if (row0 + 3 < x_width) x3 = x_sh[sec_x_start + row0 + 3];

            if constexpr (HAS_MEAN) {
                float m0 = 0.0f, m1 = 0.0f, m2 = 0.0f, m3 = 0.0f;
                if (row0     < x_width) m0 = (float)W_mean[(row0    ) * stride_Wi + o_local * stride_Wo];
                if (row0 + 1 < x_width) m1 = (float)W_mean[(row0 + 1) * stride_Wi + o_local * stride_Wo];
                if (row0 + 2 < x_width) m2 = (float)W_mean[(row0 + 2) * stride_Wi + o_local * stride_Wo];
                if (row0 + 3 < x_width) m3 = (float)W_mean[(row0 + 3) * stride_Wi + o_local * stride_Wo];
                acc = fmaf(x0, m0 + s0 * w.n0, acc);
                acc = fmaf(x1, m1 + s1 * w.n1, acc);
                acc = fmaf(x2, m2 + s2 * w.n2, acc);
                acc = fmaf(x3, m3 + s3 * w.n3, acc);
            } else {
                acc = fmaf(x0 * s0, w.n0, acc);
                acc = fmaf(x1 * s1, w.n1, acc);
                acc = fmaf(x2 * s2, w.n2, acc);
                acc = fmaf(x3 * s3, w.n3, acc);
            }
        }
    }

    if (active) {
        float y_old = (beta != 0.0f) ? (float)y[b * stride_yb + o_local * stride_yo] : 0.0f;
        y[b * stride_yb + o_local * stride_yo] = (OT)(alpha * acc + beta * y_old);
    }
}

// ============================================================================
//  Generate kernel.
// ============================================================================
//
// Grid:   (cdiv(O, BN), cdiv(I_packs, BM), B)
// Block:  (BN, BM). Each thread handles one (i_pack, o) tuple and writes
//         4 elements along the I axis.

template<typename ST, int BN, int BM>
__global__ void batched_randn_generate_kernel(
    const int64_t* __restrict__ seeds,
    const ST* __restrict__ S,
    float* __restrict__ y,
    int B, int I, int O,
    int stride_Si, int stride_So,
    int stride_yb, int stride_yi, int stride_yo,
    int col_offset, int global_cols
) {
    ZIG_KERNEL_PROLOGUE(BN * BM);

    int b = blockIdx.z;
    int ipk = blockIdx.y * BM + threadIdx.y;
    int o = blockIdx.x * BN + threadIdx.x;

    const int I_packs = (I + 3) / 4;
    if (ipk >= I_packs || o >= O) return;

    int64_t seed_val = seeds[b];

    int row0 = ipk * 4;
    float n0 = 0.0f, n1 = 0.0f, n2 = 0.0f, n3 = 0.0f;
    if (seed_val != 0) {
        uint32_t seed = (uint32_t)seed_val;
        uint32_t offset = (uint32_t)ipk * (uint32_t)global_cols + (uint32_t)(o + col_offset);
        Normals4 w = philox_4_normals(seed, offset ZIG_LUT_ARG);
        n0 = w.n0; n1 = w.n1; n2 = w.n2; n3 = w.n3;
    }

    float s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;
    if (seed_val != 0) {
        if (row0     < I) s0 = (float)S[(row0    ) * stride_Si + o * stride_So];
        if (row0 + 1 < I) s1 = (float)S[(row0 + 1) * stride_Si + o * stride_So];
        if (row0 + 2 < I) s2 = (float)S[(row0 + 2) * stride_Si + o * stride_So];
        if (row0 + 3 < I) s3 = (float)S[(row0 + 3) * stride_Si + o * stride_So];
    }

    float* y_col = y + b * stride_yb + o * stride_yo;
    if (row0     < I) y_col[(row0    ) * stride_yi] = s0 * n0;
    if (row0 + 1 < I) y_col[(row0 + 1) * stride_yi] = s1 * n1;
    if (row0 + 2 < I) y_col[(row0 + 2) * stride_yi] = s2 * n2;
    if (row0 + 3 < I) y_col[(row0 + 3) * stride_yi] = s3 * n3;
}

// ============================================================================
//  Launchers
// ============================================================================

template<typename XT, typename ST, typename OT, int BN>
void launch_matmul(
    const torch::Tensor& x, const torch::Tensor& seeds, const torch::Tensor& S,
    torch::Tensor& y, int col_offset, int global_cols,
    int64_t seed_offset, float alpha, float beta,
    const torch::Tensor* W_mean_opt
) {
    const int B = (int)x.size(0);
    const int I = (int)x.size(1);
    const int O = (int)S.size(1);

    const int num_sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int tiles_n = (O + BN - 1) / BN;
    const int I_packs = (I + 3) / 4;

    const int target_blocks = 4 * num_sms;
    const int base_blocks = B * tiles_n;
    int k_split = 1;
    if (base_blocks < target_blocks && I_packs > 16) {
        k_split = (target_blocks + base_blocks - 1) / base_blocks;
        int max_split = (I_packs + 15) / 16;
        if (k_split > max_split) k_split = max_split;
        if (k_split < 1) k_split = 1;
    }
    int k_pack_chunk = (I_packs + k_split - 1) / k_split;
    k_split = (I_packs + k_pack_chunk - 1) / k_pack_chunk;

    // k_split>1 path atomic-adds partials; pre-scale y by beta (or zero).
    if (k_split > 1) {
        if (beta == 0.0f) { y.zero_(); }
        else               { y.mul_(beta); }
    }
    // k_split==1 kernel applies beta directly (read-modify-write).
    float kernel_beta = (k_split > 1) ? 0.0f : beta;

    dim3 grid(tiles_n, B, k_split);
    dim3 block(BN);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (W_mean_opt != nullptr) {
        const auto& W = *W_mean_opt;
        batched_randn_matmul_kernel<XT, ST, OT, BN, true><<<grid, block, 0, stream>>>(
            x.data_ptr<XT>(),
            seeds.data_ptr<int64_t>(),
            S.data_ptr<ST>(),
            W.data_ptr<ST>(),
            (int)W.stride(0), (int)W.stride(1),
            y.data_ptr<OT>(),
            B, I, O,
            (int)x.stride(0), (int)x.stride(1),
            (int)S.stride(0), (int)S.stride(1),
            (int)y.stride(0), (int)y.stride(1),
            col_offset, global_cols,
            k_split, k_pack_chunk,
            seed_offset, alpha, kernel_beta
        );
    } else {
        batched_randn_matmul_kernel<XT, ST, OT, BN, false><<<grid, block, 0, stream>>>(
            x.data_ptr<XT>(),
            seeds.data_ptr<int64_t>(),
            S.data_ptr<ST>(),
            nullptr, 0, 0,
            y.data_ptr<OT>(),
            B, I, O,
            (int)x.stride(0), (int)x.stride(1),
            (int)S.stride(0), (int)S.stride(1),
            (int)y.stride(0), (int)y.stride(1),
            col_offset, global_cols,
            k_split, k_pack_chunk,
            seed_offset, alpha, kernel_beta
        );
    }
}

template<typename XT, typename ST, typename OT, int BN>
void launch_matmul_sectioned(
    const torch::Tensor& x, const torch::Tensor& seeds, const torch::Tensor& S,
    torch::Tensor& y, int col_offset, int global_cols,
    Sections sec, float alpha, float beta,
    const torch::Tensor* W_mean_opt
) {
    const int B = (int)x.size(0);
    const int I = (int)x.size(1);
    const int O = (int)S.size(1);

    const int num_sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int tiles_n = (O + BN - 1) / BN;
    const int I_packs = (I + 3) / 4;

    const int target_blocks = 4 * num_sms;
    const int base_blocks = B * tiles_n;
    int k_split = 1;
    if (base_blocks < target_blocks && I_packs > 16) {
        k_split = (target_blocks + base_blocks - 1) / base_blocks;
        int max_split = (I_packs + 15) / 16;
        if (k_split > max_split) k_split = max_split;
        if (k_split < 1) k_split = 1;
    }
    int k_pack_chunk = (I_packs + k_split - 1) / k_split;
    k_split = (I_packs + k_pack_chunk - 1) / k_pack_chunk;

    if (k_split > 1) {
        if      (beta == 0.0f) y.zero_();
        else if (beta == 1.0f) {}
        else                   y.mul_(beta);
    }
    float kernel_beta = (k_split > 1) ? 0.0f : beta;

    dim3 grid(tiles_n, B, k_split);
    dim3 block(BN);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (W_mean_opt != nullptr) {
        const auto& W = *W_mean_opt;
        batched_randn_matmul_sectioned_kernel<XT, ST, OT, BN, true><<<grid, block, 0, stream>>>(
            x.data_ptr<XT>(),
            seeds.data_ptr<int64_t>(),
            S.data_ptr<ST>(),
            W.data_ptr<ST>(),
            (int)W.stride(0), (int)W.stride(1),
            y.data_ptr<OT>(),
            B, I, O,
            (int)x.stride(0), (int)x.stride(1),
            (int)S.stride(0), (int)S.stride(1),
            (int)y.stride(0), (int)y.stride(1),
            col_offset, global_cols,
            k_split, k_pack_chunk,
            sec, alpha, kernel_beta
        );
    } else {
        batched_randn_matmul_sectioned_kernel<XT, ST, OT, BN, false><<<grid, block, 0, stream>>>(
            x.data_ptr<XT>(),
            seeds.data_ptr<int64_t>(),
            S.data_ptr<ST>(),
            nullptr, 0, 0,
            y.data_ptr<OT>(),
            B, I, O,
            (int)x.stride(0), (int)x.stride(1),
            (int)S.stride(0), (int)S.stride(1),
            (int)y.stride(0), (int)y.stride(1),
            col_offset, global_cols,
            k_split, k_pack_chunk,
            sec, alpha, kernel_beta
        );
    }
}

template<typename XT, typename ST, typename OT, int BN, int BK>
void launch_matmul_wide(
    const torch::Tensor& x, const torch::Tensor& seeds, const torch::Tensor& S,
    torch::Tensor& y, int col_offset, int global_cols,
    int64_t seed_offset, float alpha, float beta,
    const torch::Tensor* W_mean_opt
) {
    static_assert(BN * BK == 32, "wide kernel: BN*BK must be 32");
    const int B = (int)x.size(0);
    const int I = (int)x.size(1);
    const int O = (int)S.size(1);

    const int num_sms = at::cuda::getCurrentDeviceProperties()->multiProcessorCount;
    const int tiles_n = (O + BN - 1) / BN;
    const int I_packs = (I + 3) / 4;

    const int target_blocks = 4 * num_sms;
    const int base_blocks = B * tiles_n;
    int k_split = 1;
    if (base_blocks < target_blocks && I_packs > BK * 2) {
        int max_split = I_packs / (BK * 2);
        if (max_split < 1) max_split = 1;
        k_split = (target_blocks + base_blocks - 1) / base_blocks;
        if (k_split > max_split) k_split = max_split;
        if (k_split < 1) k_split = 1;
    }
    int kp_chunk = (I_packs + k_split - 1) / k_split;
    k_split = (I_packs + kp_chunk - 1) / kp_chunk;

    // k_split>1 needs atomic accumulation. Pre-condition y appropriately:
    //   beta == 0: overwrite (zero first, kernel atomic-adds)
    //   beta == 1: accumulate (leave y alone, kernel atomic-adds)
    //   other  : scale then accumulate (mul_, kernel atomic-adds)
    if (k_split > 1) {
        if      (beta == 0.0f) y.zero_();
        else if (beta == 1.0f) {}  // fast path: no pre-scale kernel launch
        else                   y.mul_(beta);
    }
    float kernel_beta = (k_split > 1) ? 0.0f : beta;

    dim3 grid(tiles_n, B, k_split);
    dim3 block(32);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (W_mean_opt != nullptr) {
        const auto& W = *W_mean_opt;
        batched_randn_matmul_wide_kernel<XT, ST, OT, BN, BK, true><<<grid, block, 0, stream>>>(
            x.data_ptr<XT>(),
            seeds.data_ptr<int64_t>(),
            S.data_ptr<ST>(),
            W.data_ptr<ST>(),
            (int)W.stride(0), (int)W.stride(1),
            y.data_ptr<OT>(),
            B, I, O,
            (int)x.stride(0), (int)x.stride(1),
            (int)S.stride(0), (int)S.stride(1),
            (int)y.stride(0), (int)y.stride(1),
            col_offset, global_cols,
            k_split, kp_chunk,
            seed_offset, alpha, kernel_beta
        );
    } else {
        batched_randn_matmul_wide_kernel<XT, ST, OT, BN, BK, false><<<grid, block, 0, stream>>>(
            x.data_ptr<XT>(),
            seeds.data_ptr<int64_t>(),
            S.data_ptr<ST>(),
            nullptr, 0, 0,
            y.data_ptr<OT>(),
            B, I, O,
            (int)x.stride(0), (int)x.stride(1),
            (int)S.stride(0), (int)S.stride(1),
            (int)y.stride(0), (int)y.stride(1),
            col_offset, global_cols,
            k_split, kp_chunk,
            seed_offset, alpha, kernel_beta
        );
    }
}

template<typename XT, typename ST, typename OT, int BN>
void launch_matmul_multi_input_sectioned(
    const torch::Tensor& x, const torch::Tensor& seeds, const torch::Tensor& S,
    torch::Tensor& y,
    MultiInputSections sec, float alpha, float beta,
    const torch::Tensor* W_mean_opt
) {
    const int B = (int)x.size(0);
    const int sum_in = (int)x.size(1);
    const int x_width = (int)S.size(0);
    const int O = (int)S.size(1);

    const int tiles_n = (O + BN - 1) / BN;
    dim3 grid(tiles_n, B);
    dim3 block(BN);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    if (W_mean_opt != nullptr) {
        const auto& W = *W_mean_opt;
        batched_randn_matmul_multi_input_sectioned_kernel<XT, ST, OT, BN, true><<<grid, block, 0, stream>>>(
            x.data_ptr<XT>(),
            seeds.data_ptr<int64_t>(),
            S.data_ptr<ST>(),
            W.data_ptr<ST>(),
            (int)W.stride(0), (int)W.stride(1),
            y.data_ptr<OT>(),
            B, sum_in, x_width, O,
            (int)x.stride(0), (int)x.stride(1),
            (int)S.stride(0), (int)S.stride(1),
            (int)y.stride(0), (int)y.stride(1),
            sec, alpha, beta
        );
    } else {
        batched_randn_matmul_multi_input_sectioned_kernel<XT, ST, OT, BN, false><<<grid, block, 0, stream>>>(
            x.data_ptr<XT>(),
            seeds.data_ptr<int64_t>(),
            S.data_ptr<ST>(),
            nullptr, 0, 0,
            y.data_ptr<OT>(),
            B, sum_in, x_width, O,
            (int)x.stride(0), (int)x.stride(1),
            (int)S.stride(0), (int)S.stride(1),
            (int)y.stride(0), (int)y.stride(1),
            sec, alpha, beta
        );
    }
}

template<typename ST, int BN, int BM>
void launch_generate(
    const torch::Tensor& seeds, const torch::Tensor& S, torch::Tensor& y,
    int col_offset, int global_cols
) {
    const int B = (int)seeds.size(0);
    const int I = (int)S.size(0);
    const int O = (int)S.size(1);
    const int I_packs = (I + 3) / 4;

    dim3 grid((O + BN - 1) / BN, (I_packs + BM - 1) / BM, B);
    dim3 block(BN, BM);

    auto stream = at::cuda::getCurrentCUDAStream().stream();
    batched_randn_generate_kernel<ST, BN, BM><<<grid, block, 0, stream>>>(
        seeds.data_ptr<int64_t>(),
        S.data_ptr<ST>(),
        y.data_ptr<float>(),
        B, I, O,
        (int)S.stride(0), (int)S.stride(1),
        (int)y.stride(0), (int)y.stride(1), (int)y.stride(2),
        col_offset, global_cols
    );
}

}  // anonymous namespace

// ============================================================================
//  Python bindings
// ============================================================================

torch::Tensor batched_randn_matmul_cuda(
    torch::Tensor x, torch::Tensor seeds, torch::Tensor S,
    int64_t col_offset, int64_t global_cols,
    int64_t seed_offset,
    c10::optional<torch::Tensor> out_opt,
    double alpha,
    double beta,
    c10::optional<torch::Tensor> W_mean_opt
) {
    TORCH_CHECK(x.is_cuda() && S.is_cuda(), "x and S must be CUDA");
    TORCH_CHECK(x.dim() == 2 && S.dim() == 2);
    const int64_t B = x.size(0);
    const int64_t O = S.size(1);

    torch::Tensor y;
    float kernel_alpha = (float)alpha;
    float kernel_beta = (float)beta;
    if (out_opt.has_value()) {
        y = *out_opt;
        TORCH_CHECK(y.size(0) == B && y.size(1) == O, "out shape mismatch");
        TORCH_CHECK(
            y.scalar_type() == torch::kFloat32
            || y.scalar_type() == torch::kFloat16
            || y.scalar_type() == torch::kBFloat16,
            "out must be float32, float16, or bfloat16");
    } else {
        y = torch::empty({B, O}, x.options().dtype(torch::kFloat32));
        kernel_beta = 0.0f;
    }

    const torch::Tensor* W_mean_ptr = nullptr;
    if (W_mean_opt.has_value()) {
        const auto& W = *W_mean_opt;
        TORCH_CHECK(W.is_cuda(), "W_mean must be CUDA");
        TORCH_CHECK(W.scalar_type() == S.scalar_type(),
                    "W_mean dtype must match S dtype");
        TORCH_CHECK(W.dim() == 2 && W.size(0) == S.size(0) && W.size(1) == S.size(1),
                    "W_mean shape must equal S shape");
        W_mean_ptr = &W;
    }

    auto dispatch_bn = [&](auto xt_tag, auto st_tag, auto ot_tag) {
        using XT = typename decltype(xt_tag)::type;
        using ST = typename decltype(st_tag)::type;
        using OT = typename decltype(ot_tag)::type;
        if (O <= 4) {
            launch_matmul_wide<XT, ST, OT, 4, 8>(x, seeds, S, y, col_offset, global_cols,
                                                 seed_offset, kernel_alpha, kernel_beta, W_mean_ptr);
        } else if (O <= 8) {
            launch_matmul_wide<XT, ST, OT, 8, 4>(x, seeds, S, y, col_offset, global_cols,
                                                 seed_offset, kernel_alpha, kernel_beta, W_mean_ptr);
        } else if (O <= 16) {
            launch_matmul_wide<XT, ST, OT, 16, 2>(x, seeds, S, y, col_offset, global_cols,
                                                  seed_offset, kernel_alpha, kernel_beta, W_mean_ptr);
        } else if (O <= 32) {
            launch_matmul<XT, ST, OT, 32>(x, seeds, S, y, col_offset, global_cols,
                                          seed_offset, kernel_alpha, kernel_beta, W_mean_ptr);
        } else if (O <= 64) {
            launch_matmul<XT, ST, OT, 64>(x, seeds, S, y, col_offset, global_cols,
                                          seed_offset, kernel_alpha, kernel_beta, W_mean_ptr);
        } else {
            launch_matmul<XT, ST, OT, 128>(x, seeds, S, y, col_offset, global_cols,
                                           seed_offset, kernel_alpha, kernel_beta, W_mean_ptr);
        }
    };

    auto xdt = x.scalar_type();
    auto sdt = S.scalar_type();
    auto odt = y.scalar_type();

    auto ot_dispatch = [&](auto xt_tag, auto st_tag) {
        using XT = typename decltype(xt_tag)::type;
        using ST = typename decltype(st_tag)::type;
        if (odt == torch::kFloat32)       dispatch_bn(type_tag<XT>{}, type_tag<ST>{}, type_tag<float>{});
        else if (odt == torch::kFloat16)  dispatch_bn(type_tag<XT>{}, type_tag<ST>{}, type_tag<at::Half>{});
        else if (odt == torch::kBFloat16) dispatch_bn(type_tag<XT>{}, type_tag<ST>{}, type_tag<at::BFloat16>{});
        else TORCH_CHECK(false, "unsupported output dtype");
    };

    // x ∈ {fp16, fp32, bf16}  ×  S ∈ {fp16, fp32, bf16}
    if      (xdt == torch::kFloat16   && sdt == torch::kFloat32)   ot_dispatch(type_tag<at::Half>{},     type_tag<float>{});
    else if (xdt == torch::kFloat16   && sdt == torch::kFloat16)   ot_dispatch(type_tag<at::Half>{},     type_tag<at::Half>{});
    else if (xdt == torch::kFloat16   && sdt == torch::kBFloat16)  ot_dispatch(type_tag<at::Half>{},     type_tag<at::BFloat16>{});
    else if (xdt == torch::kFloat32   && sdt == torch::kFloat32)   ot_dispatch(type_tag<float>{},        type_tag<float>{});
    else if (xdt == torch::kFloat32   && sdt == torch::kFloat16)   ot_dispatch(type_tag<float>{},        type_tag<at::Half>{});
    else if (xdt == torch::kFloat32   && sdt == torch::kBFloat16)  ot_dispatch(type_tag<float>{},        type_tag<at::BFloat16>{});
    else if (xdt == torch::kBFloat16  && sdt == torch::kFloat32)   ot_dispatch(type_tag<at::BFloat16>{}, type_tag<float>{});
    else if (xdt == torch::kBFloat16  && sdt == torch::kFloat16)   ot_dispatch(type_tag<at::BFloat16>{}, type_tag<at::Half>{});
    else if (xdt == torch::kBFloat16  && sdt == torch::kBFloat16)  ot_dispatch(type_tag<at::BFloat16>{}, type_tag<at::BFloat16>{});
    else TORCH_CHECK(false, "unsupported dtype combination");
    return y;
}

torch::Tensor batched_randn_matmul_sectioned_cuda(
    torch::Tensor x, torch::Tensor seeds, torch::Tensor S,
    std::vector<int64_t> section_widths,
    std::vector<int64_t> section_offsets,
    int64_t col_offset, int64_t global_cols,
    c10::optional<torch::Tensor> out_opt,
    double alpha,
    double beta,
    c10::optional<torch::Tensor> W_mean_opt
) {
    TORCH_CHECK(x.is_cuda() && S.is_cuda(), "x and S must be CUDA");
    TORCH_CHECK(x.dim() == 2 && S.dim() == 2);
    TORCH_CHECK(section_widths.size() == section_offsets.size(), "section widths/offsets length mismatch");
    TORCH_CHECK(section_widths.size() >= 1 && section_widths.size() <= 4, "up to 4 sections supported");
    const int64_t B = x.size(0);
    const int64_t O = S.size(1);

    int64_t total = 0;
    for (auto w : section_widths) total += w;
    TORCH_CHECK(total == O, "sum(section_widths) must equal S.shape[1]");

    Sections sec{};
    sec.n = (int)section_widths.size();
    for (int i = 0; i < sec.n; i++) {
        sec.widths[i]  = (int)section_widths[i];
        sec.offsets[i] = section_offsets[i];
    }

    torch::Tensor y;
    float kernel_alpha = (float)alpha;
    float kernel_beta = (float)beta;
    if (out_opt.has_value()) {
        y = *out_opt;
        TORCH_CHECK(y.size(0) == B && y.size(1) == O, "out shape mismatch");
        TORCH_CHECK(
            y.scalar_type() == torch::kFloat32
            || y.scalar_type() == torch::kFloat16
            || y.scalar_type() == torch::kBFloat16,
            "out must be float32, float16, or bfloat16");
    } else {
        y = torch::empty({B, O}, x.options().dtype(torch::kFloat32));
        kernel_beta = 0.0f;
    }

    const torch::Tensor* W_mean_ptr = nullptr;
    if (W_mean_opt.has_value()) {
        const auto& W = *W_mean_opt;
        TORCH_CHECK(W.is_cuda(), "W_mean must be CUDA");
        TORCH_CHECK(W.scalar_type() == S.scalar_type(), "W_mean dtype must match S dtype");
        TORCH_CHECK(W.dim() == 2 && W.size(0) == S.size(0) && W.size(1) == S.size(1),
                    "W_mean shape must equal S shape");
        W_mean_ptr = &W;
    }

    auto dispatch_bn = [&](auto xt_tag, auto st_tag, auto ot_tag) {
        using XT = typename decltype(xt_tag)::type;
        using ST = typename decltype(st_tag)::type;
        using OT = typename decltype(ot_tag)::type;
        if (O <= 32)       launch_matmul_sectioned<XT, ST, OT, 32>(x, seeds, S, y, col_offset, global_cols, sec, kernel_alpha, kernel_beta, W_mean_ptr);
        else if (O <= 64)  launch_matmul_sectioned<XT, ST, OT, 64>(x, seeds, S, y, col_offset, global_cols, sec, kernel_alpha, kernel_beta, W_mean_ptr);
        else                launch_matmul_sectioned<XT, ST, OT, 128>(x, seeds, S, y, col_offset, global_cols, sec, kernel_alpha, kernel_beta, W_mean_ptr);
    };

    auto xdt = x.scalar_type();
    auto sdt = S.scalar_type();
    auto odt = y.scalar_type();
    auto ot_dispatch = [&](auto xt_tag, auto st_tag) {
        using XT = typename decltype(xt_tag)::type;
        using ST = typename decltype(st_tag)::type;
        if (odt == torch::kFloat32)       dispatch_bn(type_tag<XT>{}, type_tag<ST>{}, type_tag<float>{});
        else if (odt == torch::kFloat16)  dispatch_bn(type_tag<XT>{}, type_tag<ST>{}, type_tag<at::Half>{});
        else if (odt == torch::kBFloat16) dispatch_bn(type_tag<XT>{}, type_tag<ST>{}, type_tag<at::BFloat16>{});
        else TORCH_CHECK(false, "unsupported output dtype");
    };
    if      (xdt == torch::kFloat16   && sdt == torch::kFloat32)  ot_dispatch(type_tag<at::Half>{},     type_tag<float>{});
    else if (xdt == torch::kFloat16   && sdt == torch::kFloat16)  ot_dispatch(type_tag<at::Half>{},     type_tag<at::Half>{});
    else if (xdt == torch::kFloat16   && sdt == torch::kBFloat16) ot_dispatch(type_tag<at::Half>{},     type_tag<at::BFloat16>{});
    else if (xdt == torch::kFloat32   && sdt == torch::kFloat32)  ot_dispatch(type_tag<float>{},        type_tag<float>{});
    else if (xdt == torch::kFloat32   && sdt == torch::kFloat16)  ot_dispatch(type_tag<float>{},        type_tag<at::Half>{});
    else if (xdt == torch::kFloat32   && sdt == torch::kBFloat16) ot_dispatch(type_tag<float>{},        type_tag<at::BFloat16>{});
    else if (xdt == torch::kBFloat16  && sdt == torch::kFloat32)  ot_dispatch(type_tag<at::BFloat16>{}, type_tag<float>{});
    else if (xdt == torch::kBFloat16  && sdt == torch::kFloat16)  ot_dispatch(type_tag<at::BFloat16>{}, type_tag<at::Half>{});
    else if (xdt == torch::kBFloat16  && sdt == torch::kBFloat16) ot_dispatch(type_tag<at::BFloat16>{}, type_tag<at::BFloat16>{});
    else TORCH_CHECK(false, "unsupported dtype combination");

    return y;
}

torch::Tensor batched_randn_matmul_multi_input_sectioned_cuda(
    torch::Tensor x, torch::Tensor seeds, torch::Tensor S,
    std::vector<int64_t> x_starts,
    std::vector<int64_t> section_widths,
    std::vector<int64_t> section_offsets,
    c10::optional<torch::Tensor> out_opt,
    double alpha,
    double beta,
    c10::optional<torch::Tensor> W_mean_opt
) {
    TORCH_CHECK(x.is_cuda() && S.is_cuda(), "x and S must be CUDA");
    TORCH_CHECK(x.dim() == 2 && S.dim() == 2);
    TORCH_CHECK(x_starts.size() == section_widths.size() && x_starts.size() == section_offsets.size(),
                "x_starts/section_widths/section_offsets length mismatch");
    TORCH_CHECK(section_widths.size() >= 1 && section_widths.size() <= 4, "up to 4 sections supported");
    const int64_t B = x.size(0);
    const int64_t sum_in = x.size(1);
    const int64_t x_width = S.size(0);
    const int64_t O = S.size(1);
    TORCH_CHECK(sum_in <= 128, "multi-input sectioned: total packed input width must be <= 128");

    int64_t total_out = 0;
    for (auto w : section_widths) total_out += w;
    TORCH_CHECK(total_out == O, "sum(section_widths) must equal S.shape[1]");

    MultiInputSections sec{};
    sec.n = (int)section_widths.size();
    for (int i = 0; i < sec.n; i++) {
        sec.widths[i]   = (int)section_widths[i];
        sec.offsets[i]  = section_offsets[i];
        sec.x_starts[i] = (int)x_starts[i];
        TORCH_CHECK(sec.x_starts[i] + (int)x_width <= (int)sum_in,
                    "x_starts[i] + x_width must fit in sum_in");
    }

    torch::Tensor y;
    float kernel_alpha = (float)alpha;
    float kernel_beta = (float)beta;
    if (out_opt.has_value()) {
        y = *out_opt;
        TORCH_CHECK(y.size(0) == B && y.size(1) == O, "out shape mismatch");
        TORCH_CHECK(
            y.scalar_type() == torch::kFloat32
            || y.scalar_type() == torch::kFloat16
            || y.scalar_type() == torch::kBFloat16,
            "out must be float32, float16, or bfloat16");
    } else {
        y = torch::empty({B, O}, x.options().dtype(torch::kFloat32));
        kernel_beta = 0.0f;
    }

    const torch::Tensor* W_mean_ptr = nullptr;
    if (W_mean_opt.has_value()) {
        const auto& W = *W_mean_opt;
        TORCH_CHECK(W.is_cuda(), "W_mean must be CUDA");
        TORCH_CHECK(W.scalar_type() == S.scalar_type(), "W_mean dtype must match S dtype");
        TORCH_CHECK(W.dim() == 2 && W.size(0) == S.size(0) && W.size(1) == S.size(1),
                    "W_mean shape must equal S shape");
        W_mean_ptr = &W;
    }

    auto dispatch_bn = [&](auto xt_tag, auto st_tag, auto ot_tag) {
        using XT = typename decltype(xt_tag)::type;
        using ST = typename decltype(st_tag)::type;
        using OT = typename decltype(ot_tag)::type;
        if (O <= 32)       launch_matmul_multi_input_sectioned<XT, ST, OT, 32>(x, seeds, S, y, sec, kernel_alpha, kernel_beta, W_mean_ptr);
        else if (O <= 64)  launch_matmul_multi_input_sectioned<XT, ST, OT, 64>(x, seeds, S, y, sec, kernel_alpha, kernel_beta, W_mean_ptr);
        else                launch_matmul_multi_input_sectioned<XT, ST, OT, 128>(x, seeds, S, y, sec, kernel_alpha, kernel_beta, W_mean_ptr);
    };

    auto xdt = x.scalar_type();
    auto sdt = S.scalar_type();
    auto odt = y.scalar_type();
    auto ot_dispatch = [&](auto xt_tag, auto st_tag) {
        using XT = typename decltype(xt_tag)::type;
        using ST = typename decltype(st_tag)::type;
        if (odt == torch::kFloat32)       dispatch_bn(type_tag<XT>{}, type_tag<ST>{}, type_tag<float>{});
        else if (odt == torch::kFloat16)  dispatch_bn(type_tag<XT>{}, type_tag<ST>{}, type_tag<at::Half>{});
        else if (odt == torch::kBFloat16) dispatch_bn(type_tag<XT>{}, type_tag<ST>{}, type_tag<at::BFloat16>{});
        else TORCH_CHECK(false, "unsupported output dtype");
    };
    if      (xdt == torch::kFloat16   && sdt == torch::kFloat32)  ot_dispatch(type_tag<at::Half>{},     type_tag<float>{});
    else if (xdt == torch::kFloat16   && sdt == torch::kFloat16)  ot_dispatch(type_tag<at::Half>{},     type_tag<at::Half>{});
    else if (xdt == torch::kFloat16   && sdt == torch::kBFloat16) ot_dispatch(type_tag<at::Half>{},     type_tag<at::BFloat16>{});
    else if (xdt == torch::kFloat32   && sdt == torch::kFloat32)  ot_dispatch(type_tag<float>{},        type_tag<float>{});
    else if (xdt == torch::kFloat32   && sdt == torch::kFloat16)  ot_dispatch(type_tag<float>{},        type_tag<at::Half>{});
    else if (xdt == torch::kFloat32   && sdt == torch::kBFloat16) ot_dispatch(type_tag<float>{},        type_tag<at::BFloat16>{});
    else if (xdt == torch::kBFloat16  && sdt == torch::kFloat32)  ot_dispatch(type_tag<at::BFloat16>{}, type_tag<float>{});
    else if (xdt == torch::kBFloat16  && sdt == torch::kFloat16)  ot_dispatch(type_tag<at::BFloat16>{}, type_tag<at::Half>{});
    else if (xdt == torch::kBFloat16  && sdt == torch::kBFloat16) ot_dispatch(type_tag<at::BFloat16>{}, type_tag<at::BFloat16>{});
    else TORCH_CHECK(false, "unsupported dtype combination");

    return y;
}


torch::Tensor batched_randn_generate_cuda(
    torch::Tensor seeds, torch::Tensor S,
    int64_t col_offset, int64_t global_cols
) {
    TORCH_CHECK(S.is_cuda(), "S must be CUDA");
    TORCH_CHECK(S.dim() == 2);
    const int64_t B = seeds.size(0);
    const int64_t I = S.size(0);
    const int64_t O = S.size(1);

    auto y = torch::empty({B, I, O}, S.options().dtype(torch::kFloat32));

    auto dispatch_bn = [&](auto st_tag) {
        using ST = typename decltype(st_tag)::type;
        if (O >= 128)      launch_generate<ST, 32, 8>(seeds, S, y, col_offset, global_cols);
        else if (O >= 32)  launch_generate<ST, 16, 16>(seeds, S, y, col_offset, global_cols);
        else                launch_generate<ST, 8, 16>(seeds, S, y, col_offset, global_cols);
    };

    auto sdt = S.scalar_type();
    if (sdt == torch::kFloat32)      dispatch_bn(type_tag<float>{});
    else if (sdt == torch::kFloat16) dispatch_bn(type_tag<at::Half>{});
    else TORCH_CHECK(false, "unsupported S dtype");

    return y;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("batched_randn_matmul", &batched_randn_matmul_cuda,
          "batched_randn_matmul (CUDA)",
          py::arg("x"), py::arg("seeds"), py::arg("S"),
          py::arg("col_offset") = 0, py::arg("global_cols") = 0,
          py::arg("seed_offset") = 0,
          py::arg("out") = c10::nullopt,
          py::arg("alpha") = 1.0,
          py::arg("beta") = 0.0,
          py::arg("W_mean") = c10::nullopt);
    m.def("batched_randn_matmul_sectioned", &batched_randn_matmul_sectioned_cuda,
          "batched_randn_matmul with per-output-section seed offsets (CUDA)",
          py::arg("x"), py::arg("seeds"), py::arg("S"),
          py::arg("section_widths"), py::arg("section_offsets"),
          py::arg("col_offset") = 0, py::arg("global_cols") = 0,
          py::arg("out") = c10::nullopt,
          py::arg("alpha") = 1.0,
          py::arg("beta") = 0.0,
          py::arg("W_mean") = c10::nullopt);
    m.def("batched_randn_matmul_multi_input_sectioned",
          &batched_randn_matmul_multi_input_sectioned_cuda,
          "batched_randn_matmul where each output section pulls from a "
          "different slice of the packed input (CUDA, single-GPU only)",
          py::arg("x"), py::arg("seeds"), py::arg("S"),
          py::arg("x_starts"),
          py::arg("section_widths"),
          py::arg("section_offsets"),
          py::arg("out") = c10::nullopt,
          py::arg("alpha") = 1.0,
          py::arg("beta") = 0.0,
          py::arg("W_mean") = c10::nullopt);
    m.def("batched_randn_generate", &batched_randn_generate_cuda,
          "batched_randn_generate (CUDA)",
          py::arg("seeds"), py::arg("S"),
          py::arg("col_offset") = 0, py::arg("global_cols") = 0);
}
