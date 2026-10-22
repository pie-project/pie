#include "linear/codec.cuh"
#include "linear/act_quant.cuh"
#include "linear/grouped_fp8.cuh"
#include "linear/moe_w4.cuh"
#include "linear/quant.cuh"

namespace pie::linear {

constexpr int kQuantCodes = 0;

__device__ __forceinline__ float global_of(const float* __restrict__ input_scales,
                                           const i32* __restrict__ routes, int r, bool* dropped) {
    int e = 0;
    if (routes != nullptr) {
        e = routes[r];
        if (e < 0) {
            *dropped = true;
            return 0.f;
        }
    }
    return __fdiv_rn(1.f, input_scales[e]);
}

// One thread a 16-wide block. Under `kFan` (codes only) row `r` is a token
// read once, its block's codes landed at each of its `div` routes' rows
// (`pos[r * div + j]`) under that route's expert's global, as each route's
// own row would land them.
template <int kMode, bool kFan = false>
__global__ void act_quant_nvfp4(
    const bf16* __restrict__ x,
    const i32* __restrict__ routes,
    const float* __restrict__ input_scales,
    const i32* __restrict__ pos,
    int m_max,
    u8* __restrict__ codes,
    u8* __restrict__ atoms,
    bf16* __restrict__ values,
    int rows,
    int k,
    int div)
{
    const int blocks = k / 16;
    const long long at = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at >= static_cast<long long>(rows) * blocks) return;
    const int r = static_cast<int>(at / blocks);
    const int b = static_cast<int>(at % blocks);
    if constexpr (kFan) {
        float v[16];
        const float amax = load_block16(x + static_cast<long long>(r) * k + b * 16, v);
        // The routes' globals are one (load lays the experts' largest), so
        // the block's codes are quantized once a global (its divide once a
        // scale) and land at each route. Where a warp holds whole rows (and
        // a row's routes fit it), lane j reads route j's row and scale once
        // for the warp.
        const int lane = threadIdx.x & 31;
        const bool shared_routes = blocks % 32 == 0 && div <= 32;
        int my_o = -1;
        float my_scale = 0.f;
        if (shared_routes && lane < div) {
            my_o = pos[r * div + lane];
            const int e = routes != nullptr ? routes[r * div + lane] : 0;
            if (e < 0) my_o = -1;
            if (my_o >= 0) my_scale = input_scales[e];
        }
        float last_scale = -1.f;
        uint2 w = make_uint2(0u, 0u);
        u8 sf_code = 0;
        for (int j = 0; j < div; ++j) {
            int o;
            float scale;
            if (shared_routes) {
                o = __shfl_sync(0xffffffffu, my_o, j);
                scale = __shfl_sync(0xffffffffu, my_scale, j);
                if (o < 0) continue;
            } else {
                const int route = r * div + j;
                o = pos[route];
                if (o < 0) continue;
                const int e = routes != nullptr ? routes[route] : 0;
                if (e < 0) continue;
                scale = input_scales[e];
            }
            if (scale != last_scale) {
                last_scale = scale;
                sf_code = nvfp4_block16(v, amax, __fdiv_rn(1.f, scale), w);
            }
            *reinterpret_cast<uint2*>(codes + static_cast<long long>(o) * (k / 2) + b * 8) = w;
            atoms[atom_offset(atom_row(o, m_max), b, blocks)] = sf_code;
        }
        return;
    }
    // A route that lands nowhere -- dropped, or past a captured fire's live
    // window, whose route ids this fire never wrote -- is not read: its id
    // is no expert's, so it indexes no input scale.
    const int o = pos != nullptr ? pos[r] : r;
    if (kMode == kQuantCodes && o < 0) return;
    bool dropped = false;
    const float gs = global_of(input_scales, routes, r, &dropped);
    const bf16* src = x + static_cast<long long>(r / div) * k + b * 16;
    float v[16];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        v[i] = dropped ? 0.f : bf16_to_f32(src[i]);
        amax = fmaxf(amax, fabsf(v[i]));
    }
    const float sf = gs * (amax * rcp_approx_ftz(6.f));
    const u8 sf_code = __nv_cvt_float_to_fp8(sf, __NV_SATFINITE, __NV_E4M3);
    const float sf_value = e4m3_value(sf_code);
    const float out_scale = sf_value != 0.f ? rcp_approx_ftz(sf_value * rcp_approx_ftz(gs)) : 0.f;
    if constexpr (kMode == kQuantCodes) {
        u8* dst = codes + static_cast<long long>(o) * (k / 2) + b * 8;
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            dst[i] = static_cast<u8>(e2m1_pair(v[2 * i] * out_scale, v[2 * i + 1] * out_scale));
        }
        atoms[atom_offset(atom_row(o, m_max), b, blocks)] = sf_code;
    } else {
        bf16* dst = values + static_cast<long long>(r) * k + b * 16;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            dst[i] = f32_to_bf16(kFp4Lut[e2m1_code(v[i] * out_scale)] * sf_value);
        }
    }
}

// One warp a `kGran`-wide block, each lane `kGran / 32` elements. Under
// `kFan` (codes only) row `r` is a token read once and landed at each of its
// `div` routes' rows (`pos[r * div + j]`, none where negative), as each
// route's own row would land it.
template <int kMode, int kGran, bool kFan = false>
__global__ void act_quant_e4m3(
    const bf16* __restrict__ x,
    const i32* __restrict__ pos,
    int m_max,
    u8* __restrict__ codes,
    u8* __restrict__ atoms,
    bf16* __restrict__ values,
    int rows,
    int k,
    int div)
{
    constexpr int kEach = kGran / 32;
    const int blocks = k / kGran;
    const long long warp = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (warp >= static_cast<long long>(rows) * blocks) return;
    const int r = static_cast<int>(warp / blocks);
    const int b = static_cast<int>(warp % blocks);
    const int src_row = kFan ? r : r / div;
    const bf16* src = x + static_cast<long long>(src_row) * k + b * kGran + lane * kEach;
    float v[kEach];
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < kEach; ++i) {
        v[i] = bf16_to_f32(src[i]);
        amax = fmaxf(amax, fabsf(v[i]));
    }
    amax = ::pie::warp_max(amax);
    const int e = e4m3_block_exp(amax);
    const float s = __uint_as_float(static_cast<unsigned>(e + 127) << 23);
    u8 q[kEach];
#pragma unroll
    for (int i = 0; i < kEach; ++i) {
        q[i] = __nv_cvt_float_to_fp8(fminf(fmaxf(v[i] / s, -448.f), 448.f),
                                     __NV_SATFINITE, __NV_E4M3);
    }
    if constexpr (kMode == kQuantCodes) {
        for (int j = 0; j < (kFan ? div : 1); ++j) {
            const int o = kFan ? pos[static_cast<long long>(r) * div + j] : (pos != nullptr ? pos[r] : r);
            if (o < 0) continue;
            u8* dst = codes + static_cast<long long>(o) * k + b * kGran + lane * kEach;
#pragma unroll
            for (int i = 0; i < kEach; ++i) dst[i] = q[i];
            if (lane == 0) atoms[atom_offset(atom_row(o, m_max), b, blocks)] = static_cast<u8>(e + 127);
        }
    } else {
        bf16* dst = values + static_cast<long long>(r) * k + b * kGran + lane * kEach;
#pragma unroll
        for (int i = 0; i < kEach; ++i) dst[i] = f32_to_bf16(e4m3_value(q[i]) * s);
    }
}

// `act_quant_e4m3` eight elements a lane (`kFan` as there).
template <int kMode, int kGran, bool kFan = false>
__global__ void act_quant_e4m3_v8(
    const bf16* __restrict__ x,
    const i32* __restrict__ pos,
    int m_max,
    u8* __restrict__ codes,
    u8* __restrict__ atoms,
    bf16* __restrict__ values,
    int rows,
    int k,
    int div)
{
    // A programmatic dependent (`kernel.pdl`) reads nothing until its
    // producer is done; its consumer (a tcgen05 GEMM) may launch from here.
    pdl_wait();
    pdl_launch();
    using Block = E4m3Block<kGran>;
    const int blocks = k / kGran;
    const long long at = (static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x);
    const long long slot = at / Block::kLanes;
    const int lane = static_cast<int>(threadIdx.x % 32);
    // A slot past the last block keeps its lanes in the shuffle, landing
    // nothing.
    const bool live = slot < static_cast<long long>(rows) * blocks;
    const long long sl = live ? slot : 0;
    const int r = static_cast<int>(sl / blocks);
    const int b = static_cast<int>(sl % blocks);
    Block q;
    q.load(x + static_cast<long long>(kFan ? r : r / div) * k + b * kGran, lane);
    if (!live) return;
    if constexpr (kMode == kQuantCodes) {
        for (int j = 0; j < (kFan ? div : 1); ++j) {
            const int o = kFan ? pos[static_cast<long long>(r) * div + j] : (pos != nullptr ? pos[r] : r);
            if (o < 0) continue;
            q.store(codes, atoms, o, b, k, m_max, lane);
        }
    } else {
        const int sub = lane % Block::kLanes;
        const float s = __uint_as_float(static_cast<unsigned>(q.e + 127) << 23);
        const unsigned long long c = q.codes();
        unsigned w[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float lo = e4m3_value(static_cast<u8>(c >> (16 * i))) * s;
            const float hi = e4m3_value(static_cast<u8>(c >> (16 * i + 8))) * s;
            w[i] = pack_bf16x2(lo, hi);
        }
        *reinterpret_cast<uint4*>(values + static_cast<long long>(r) * k + b * kGran + sub * 8) =
            make_uint4(w[0], w[1], w[2], w[3]);
    }
}

// A gate-up's bf16 rows (`[rows][gate inter | up inter]`) through SwiGLU
// (rounded to bf16, as a GLU landing them would) into e4m3 codes in 1x128
// blocks at the same rows, eight a lane, `m_max` a masked layout's.
__global__ void swiglu_quant_e4m3_v8(
    const bf16* __restrict__ h, int rows, int inter, int m_max, u8* __restrict__ codes,
    u8* __restrict__ atoms)
{
    pdl_wait();
    pdl_launch();
    using Block = E4m3Block<128>;
    const int blocks = inter / 128;
    const long long at = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    const long long slot = at / Block::kLanes;
    const int lane = static_cast<int>(threadIdx.x % 32);
    const bool live = slot < static_cast<long long>(rows) * blocks;
    const long long sl = live ? slot : 0;
    const int r = static_cast<int>(sl / blocks);
    const int b = static_cast<int>(sl % blocks);
    const bf16* g8 = h + static_cast<long long>(r) * 2 * inter + b * 128 + (lane % Block::kLanes) * 8;
    const uint4 gw = *reinterpret_cast<const uint4*>(g8);
    const uint4 uw = *reinterpret_cast<const uint4*>(g8 + inter);
    const unsigned gs[4] = {gw.x, gw.y, gw.z, gw.w};
    const unsigned us[4] = {uw.x, uw.y, uw.z, uw.w};
    float y[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int j = 0; j < 2; ++j) {
            const float g = __uint_as_float(j ? gs[i] & 0xFFFF0000u : gs[i] << 16);
            const float u = __uint_as_float(j ? us[i] & 0xFFFF0000u : us[i] << 16);
            y[2 * i + j] = g / (1.f + __expf(-g)) * u;
        }
    }
    Block q;
    q.hold(y);
    if (!live) return;
    q.store(codes, atoms, r, b, inter, m_max, lane);
}

// A 128x128-block FP8 plane under f32 block factors (DeepSeek-V3's and
// GLM-5.3's release: `w = code * factor`) requantized in place to ue8m0
// exponents, as vLLM's DeepGEMM path does on Blackwell: each block's factor
// rounded up to its power of two `2^e`, each code re-rounded to
// `e4m3(code * factor / 2^e)` (never past e4m3's range, the ratio at most
// one), the exponent `e + 127` landing in `exps` (`[rows / 128][cols / 128]`,
// the factors' shape). A block a CTA; a factor not above zero leaves its
// codes and lands exponent 0.
__global__ void fp8_tile_requant(
    u8* __restrict__ codes,
    const float* __restrict__ factors,
    u8* __restrict__ exps,
    int rows,
    int cols,
    int per)
{
    const int tiles_k = (cols + 127) / 128;
    const int rb = blockIdx.y;
    const int cb = blockIdx.x;
    const int r1 = min(rows, (rb + 1) * 128);
    const int c0 = cb * 128;
    const int c1 = min(cols, c0 + 128);
    // Each row its factor's (`per` rows a factor: 128 or 1).
    for (int r = rb * 128 + static_cast<int>(threadIdx.y); r < r1; r += blockDim.y) {
        const long long at = static_cast<long long>(r / per) * tiles_k + cb;
        const float f = factors[at];
        int e = 0;
        float ratio = 1.f;
        if (f > 0.f) {
            int ex;
            const float m = frexpf(f, &ex);  // f = m * 2^ex, m in [0.5, 1)
            e = m == 0.5f ? ex - 1 : ex;     // 2^e the least power of two >= f
            ratio = f / ldexpf(1.f, e);
        }
        if (threadIdx.x == 0 && r % per == 0) {
            exps[at] = f > 0.f ? static_cast<u8>(min(max(e + 127, 0), 254)) : static_cast<u8>(0);
        }
        if (!(f > 0.f) || ratio == 1.f) continue;
        u8* row = codes + static_cast<long long>(r) * cols;
        for (int c = c0 + static_cast<int>(threadIdx.x); c < c1; c += blockDim.x) {
            const float v = e4m3_value(row[c]) * ratio;
            row[c] = __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
        }
    }
}

// A row-major plane of one-byte scales, `cols` a row, into the atom layout;
// atom row `r` reads plane row `r / div` (a 128-row block's one scale
// spread to each of its rows).
__global__ void scales_to_atoms(
    const u8* __restrict__ plane,
    u8* __restrict__ atoms,
    int rows,
    int cols,
    int div)
{
    const long long at = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (at >= static_cast<long long>(rows) * cols) return;
    const int r = static_cast<int>(at / cols);
    const int c = static_cast<int>(at % cols);
    atoms[atom_offset(r, c, cols)] = plane[static_cast<long long>(r / div) * cols + c];
}

/// A bf16 plane `[rows, k]` (whole 128x128 blocks) as fp8b128: each block's
/// ue8m0 exponent `e` the least with `amax <= 448 * 2^e` (DeepSeek's rule,
/// found exactly rather than through a rounded quotient), spread to each
/// of its rows in tcgen05's atoms (`atoms`, zeroed by the caller), and each
/// value's code `e4m3(v * 2^-e)`. A plane whose values are already some
/// fp8b128 plane's (DeepSeek-V4's `wo_a`, dequantized at import) lands
/// codes of the same values: `e` is that plane's exponent or below it, and
/// each code its code times `2^(s - e)`, which e4m3 holds. Grid `[k / 128,
/// rows / 128]`, 256 threads.
__global__ void __launch_bounds__(256) quantize_fp8b128(const bf16* __restrict__ src,
                                                        u8* __restrict__ codes,
                                                        u8* __restrict__ atoms, int rows,
                                                        int k) {
    __shared__ float part[8];
    __shared__ int block_e;
    const int kb = blockIdx.x;
    const int rb = blockIdx.y;
    const int blocks = k / 128;
    const int tid = static_cast<int>(threadIdx.x);
    // A thread's 64 values: half of one row of the block.
    const int r = rb * 128 + tid / 2;
    const long long at = static_cast<long long>(r) * k + kb * 128 + (tid % 2) * 64;
    float v[64];
#pragma unroll
    for (int q = 0; q < 8; ++q) {
        const uint4 w = *reinterpret_cast<const uint4*>(src + at + q * 8);
        const unsigned ws[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            v[q * 8 + 2 * i] = __uint_as_float(ws[i] << 16);
            v[q * 8 + 2 * i + 1] = __uint_as_float(ws[i] & 0xFFFF0000u);
        }
    }
    float amax = 0.f;
#pragma unroll
    for (int i = 0; i < 64; ++i) amax = fmaxf(amax, fabsf(v[i]));
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) amax = fmaxf(amax, __shfl_xor_sync(0xFFFFFFFFu, amax, o));
    if ((tid & 31) == 0) part[tid >> 5] = amax;
    __syncthreads();
    if (tid == 0) {
        float a = 0.f;
#pragma unroll
        for (int w = 0; w < 8; ++w) a = fmaxf(a, part[w]);
        // 448 * 2^e, exact for e in -126 .. 119.
        auto bound = [](int e) { return 448.f * __uint_as_float(static_cast<unsigned>(e + 127) << 23); };
        int e = -126;
        if (a > 0.f) {
            const unsigned bits = __float_as_uint(a / 448.f);
            e = static_cast<int>((bits >> 23) & 0xFFu) - 127;
            e = max(min(e, 119), -126);
            while (e < 119 && a > bound(e)) ++e;
            while (e > -126 && a <= bound(e - 1)) --e;
        }
        block_e = e;
    }
    __syncthreads();
    const int e = block_e;
    const float inv = __uint_as_float(static_cast<unsigned>(127 - e) << 23);
    unsigned packed[16];
#pragma unroll
    for (int i = 0; i < 16; ++i) {
        const unsigned short lo = gfp8::e4m3x2_of(v[4 * i + 1] * inv, v[4 * i] * inv);
        const unsigned short hi = gfp8::e4m3x2_of(v[4 * i + 3] * inv, v[4 * i + 2] * inv);
        packed[i] = static_cast<unsigned>(lo) | (static_cast<unsigned>(hi) << 16);
    }
#pragma unroll
    for (int q = 0; q < 4; ++q)
        *reinterpret_cast<uint4*>(codes + at + q * 16) =
            make_uint4(packed[4 * q], packed[4 * q + 1], packed[4 * q + 2], packed[4 * q + 3]);
    if (tid % 2 == 0) atoms[atom_offset(r, kb, blocks)] = static_cast<u8>(e + 127);
}

/// Rows `[rows, k]` of bf16 as e4m3 in 1x128 blocks, as the tcgen05 path
/// quantizes them (`E4m3Block`): codes row-major into `codes`, each block's
/// biased exponent a byte into `exps` (`[rows, k / 128]`). 16 lanes a
/// block; 128 threads, `ceil(rows k / 1024)` blocks.
__global__ void __launch_bounds__(128) quantize_rows_e4m3b128(
    const bf16* __restrict__ x, u8* __restrict__ codes, u8* __restrict__ exps, int rows, int k)
{
    using Block = E4m3Block<128>;
    const int blocks = k / 128;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int slot = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) / Block::kLanes);
    // Every lane joins the block's shuffles; a slot past the rows reads row 0.
    const bool live = slot < rows * blocks;
    const int sl = live ? slot : 0;
    const long long at = static_cast<long long>(sl / blocks) * k + (sl % blocks) * 128;
    Block q;
    q.load(x + at, lane);
    if (!live) return;
    *reinterpret_cast<unsigned long long*>(codes + at + (lane % Block::kLanes) * 8) = q.codes();
    if (lane % Block::kLanes == 0) exps[sl] = static_cast<u8>(q.e + 127);
}

/// Rows `[rows, k]` of bf16 as e4m3 in 1x128 blocks for cuBLASLt's
/// block-scaled FP8 GEMM: `quantize_rows_e4m3b128`'s codes (rows past `rows`
/// up to `mpad` zero), each block's factor an f32 at `scales[kb * mpad + r]`
/// (cuBLASLt's `VEC128_32F` B scales). 16 lanes a block, `mpad k / 1024`
/// blocks of 128 threads.
__global__ void __launch_bounds__(128) quantize_rows_e4m3b128_lt(
    const bf16* __restrict__ x, u8* __restrict__ codes, float* __restrict__ scales, int rows,
    int mpad, int k)
{
    using Block = E4m3Block<128>;
    const int blocks = k / 128;
    const int lane = static_cast<int>(threadIdx.x) & 31;
    const int slot = static_cast<int>((blockIdx.x * blockDim.x + threadIdx.x) / Block::kLanes);
    // Every lane joins the block's shuffles; a slot past the rows reads row 0.
    const bool live = slot < mpad * blocks;
    const int sl = live ? slot : 0;
    const int r = sl / blocks, kb = sl % blocks;
    const long long at = static_cast<long long>(r) * k + kb * 128;
    Block q;
    q.load(x + (r < rows ? at : kb * 128), lane);
    if (!live) return;
    u8* dst = codes + at + (lane % Block::kLanes) * 8;
    if (r < rows) {
        *reinterpret_cast<unsigned long long*>(dst) = q.codes();
        if (lane % Block::kLanes == 0) scales[kb * mpad + r] = gfp8::exp2_biased(q.e + 127);
    } else {
        *reinterpret_cast<unsigned long long*>(dst) = 0ull;
        if (lane % Block::kLanes == 0) scales[kb * mpad + r] = 0.f;
    }
}

/// A 128x128-block FP8 plane's exponents (in tcgen05's atoms, a weight row an
/// atom row) as cuBLASLt's `BLK128x128_32F` A scales: an f32 a block at
/// `out[nb * kbp + kb]`, `kbp` the K blocks rounded up to four (the padding
/// zero). One thread a slot of `n / 128` x `kbp`.
__global__ void __launch_bounds__(256) fp8b128_atoms_to_lt(
    const u8* __restrict__ atoms, float* __restrict__ out, int n, int k)
{
    const int blocks = k / 128;
    const int kbp = (blocks + 3) / 4 * 4;
    const int i = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (i >= (n / 128) * kbp) return;
    const int nb = i / kbp, kb = i % kbp;
    out[i] = kb < blocks ? gfp8::exp2_biased(atoms[atom_offset(nb * 128, kb, blocks)]) : 0.f;
}

struct fp8_e4m3 {
    using store = u8;

    static __host__ __device__ __forceinline__ float max_abs() { return 448.f; }
    static __device__ __forceinline__ store narrow(float v) {
        return __nv_cvt_float_to_fp8(v, __NV_SATFINITE, __NV_E4M3);
    }
};

__global__ void quant_act_fp8_per_group(
    const bf16* __restrict__ act,
    u8* __restrict__ out,
    float* __restrict__ scale_out,
    i32 m,
    i32 k,
    i32 gs,
    i32 n_groups) {
    const i32 row = blockIdx.y;
    const i32 g = blockIdx.x;
    if (row >= m || g >= n_groups) return;

    const i32 base = g * gs;
    const i32 remaining = k - base;
    const i32 count = (gs < remaining) ? gs : remaining;
    const usize off = static_cast<usize>(row) * k + base;

    float amax = 0.f;
    for (i32 i = threadIdx.x; i < count; i += blockDim.x) {
        amax = fmaxf(amax, fabsf(bf16_to_f32(act[off + i])));
    }
    __shared__ float warp_max[128 / 32];
    const unsigned lane = threadIdx.x & 31;
    const unsigned warp = threadIdx.x / 32;
    for (int o = 16; o > 0; o >>= 1) {
        amax = fmaxf(amax, __shfl_down_sync(0xffffffffu, amax, o));
    }
    if (lane == 0) warp_max[warp] = amax;
    __syncthreads();
    if (threadIdx.x == 0) {
        float v = warp_max[0];
        for (unsigned w = 1; w < blockDim.x / 32; ++w) v = fmaxf(v, warp_max[w]);
        warp_max[0] = v;
    }
    __syncthreads();
    amax = warp_max[0];

    const float scale = (amax > 0.f) ? (amax / fp8_e4m3::max_abs()) : 1.f;
    const float scale_rcp = (amax > 0.f) ? (fp8_e4m3::max_abs() / amax) : 0.f;
    if (threadIdx.x == 0) {
        scale_out[static_cast<usize>(row) * n_groups + g] = scale;
    }
    for (i32 i = threadIdx.x; i < count; i += blockDim.x) {
        out[off + i] = fp8_e4m3::narrow(bf16_to_f32(act[off + i]) * scale_rcp);
    }
}

// A prefill's RMS norm of a 4096-wide row that also lands its e4m3 1x128
// operand for the next GEMM (`Run::hold_e4m3`), in one pass.
template <class T>
__global__ void __launch_bounds__(256) rmsnorm_act_quant_e4m3(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T* __restrict__ y,
    int hidden,
    float eps,
    u8* __restrict__ codes,
    u8* __restrict__ atoms,
    const u32* __restrict__ win)
{
    static_assert(sizeof(T) == 2, "16 values are two 16-byte runs");
    const int n = static_cast<int>(blockIdx.x);
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const int tid = static_cast<int>(threadIdx.x);
    const int lane = tid & 31;
    const int e0 = tid * 16;
    const bool live = e0 < hidden;
    const T* xr = x + static_cast<long long>(row) * hidden;
    float v[16];
    float ss = 0.f;
    if (live) {
        const uint4 a = *reinterpret_cast<const uint4*>(xr + e0);
        const uint4 b = *reinterpret_cast<const uint4*>(xr + e0 + 8);
        const T* pa = reinterpret_cast<const T*>(&a);
        const T* pb = reinterpret_cast<const T*>(&b);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            v[i] = Elem<T>::to_f32(pa[i]);
            v[8 + i] = Elem<T>::to_f32(pb[i]);
        }
#pragma unroll
        for (int i = 0; i < 16; ++i) ss = fmaf(v[i], v[i], ss);
    }
    ss = ::pie::warp_sum(ss);
    __shared__ float warp_ss[8];
    if (lane == 0) warp_ss[tid >> 5] = ss;
    __syncthreads();
    float total = 0.f;
#pragma unroll
    for (int w = 0; w < 8; ++w) total += warp_ss[w];
    const float inv = rsqrtf(total / static_cast<float>(hidden) + eps);
    // The normed row, rounded to T as the plane holds it.
    float q[16];
    float amax = 0.f;
    if (live) {
        const uint4 wa = *reinterpret_cast<const uint4*>(weight + e0);
        const uint4 wb = *reinterpret_cast<const uint4*>(weight + e0 + 8);
        const T* pa = reinterpret_cast<const T*>(&wa);
        const T* pb = reinterpret_cast<const T*>(&wb);
        uint4 oa, ob;
        T* qa = reinterpret_cast<T*>(&oa);
        T* qb = reinterpret_cast<T*>(&ob);
#pragma unroll
        for (int i = 0; i < 8; ++i) {
            qa[i] = Elem<T>::from_f32(v[i] * inv * Elem<T>::to_f32(pa[i]));
            qb[i] = Elem<T>::from_f32(v[8 + i] * inv * Elem<T>::to_f32(pb[i]));
            q[i] = Elem<T>::to_f32(qa[i]);
            q[8 + i] = Elem<T>::to_f32(qb[i]);
        }
        T* yr = y + static_cast<long long>(row) * hidden;
        *reinterpret_cast<uint4*>(yr + e0) = oa;
        *reinterpret_cast<uint4*>(yr + e0 + 8) = ob;
#pragma unroll
        for (int i = 0; i < 16; ++i) amax = fmaxf(amax, fabsf(q[i]));
    }
    // A 128-wide block's eight threads (aligned in the warp) share its amax.
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
    amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
    if (!live) return;
    const int e = pie::linear::e4m3_block_exp(amax);
    const float s = __uint_as_float(static_cast<unsigned>(e + 127) << 23);
    u32 w4[4];
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        u32 word = 0;
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const u32 c = __nv_cvt_float_to_fp8(fminf(fmaxf(q[4 * j + i] / s, -448.f), 448.f), __NV_SATFINITE, __NV_E4M3);
            word |= c << (8 * i);
        }
        w4[j] = word;
    }
    *reinterpret_cast<uint4*>(codes + static_cast<long long>(row) * hidden + e0) = make_uint4(w4[0], w4[1], w4[2], w4[3]);
    if ((tid & 7) == 0) {
        atoms[pie::linear::atom_offset(row, e0 / 128, hidden / 128)] = static_cast<u8>(e + 127);
    }
}

}
