#include "prim/device.cuh"
#include "prim/paged_kv.cuh"
#include "prim/rope.cuh"
#include "prim/kv_rows.cuh"

namespace pie::elemwise {

// A head's RMS norm (times `weight` where given) and the rotation of its
// last `rotary_dim` dims, a warp a (row, head) unit: lane `l` holds the
// head's runs of RUN elements `l`, `l + 32`, ... (RUN 4 for a 128-wide head,
// else 8), rounds its normed values to bf16 as the norm's own plane would,
// and turns the interleaved pairs of the tail it holds by the angles
// `rope_partial_last_spread` works out, a lane working one pair's and
// handing it on. `head_dim = 32 * PER`; the tail starts on a run and is at
// most 128 wide. APPEND (one 512-wide head a row, DeepSeek-V4's kv plane)
// also writes the row as DeepSeek's fp8 row at its page and offset, as
// `rows_append<kV4>` writes it: a 64-wide tile's exponent over its eight lanes'
// runs, the rotated tile kept in bf16.
// A unit's head as its lane holds it: which plane row (-1 past the stage's
// live rows) and its runs.
template <int PER>
struct NormRuns {
    static constexpr int RUN = PER == 4 ? 4 : 8;
    static constexpr int RUNS = PER / RUN;
    int plane_row;
    bf16 v[RUNS][RUN];
};

// Loads unit `unit`'s head (its plane row's, where the stage's window keeps it).
template <int PER>
__device__ __forceinline__ void norm_rope_fetch(NormRuns<PER>& in, int unit, const bf16* __restrict__ x,
                                                int heads, const u32* __restrict__ win)
{
    static_assert(PER == 4 || PER == 8 || PER == 16, "a lane holds one to two runs");
    constexpr int RUN = NormRuns<PER>::RUN;
    in.plane_row = -1;
    if (win != nullptr && unit / heads >= static_cast<int>(win[0])) return;
    in.plane_row = win != nullptr ? unit + static_cast<int>(win[1]) * heads : unit;
    const int lane = threadIdx.x & 31;
    const long long at = static_cast<long long>(in.plane_row) * (32 * PER);
#pragma unroll
    for (int r = 0; r < NormRuns<PER>::RUNS; ++r) {
        const int e = (lane + 32 * r) * RUN;
        if constexpr (RUN == 4) {
            *reinterpret_cast<uint2*>(in.v[r]) = *reinterpret_cast<const uint2*>(x + at + e);
        } else {
            *reinterpret_cast<uint4*>(in.v[r]) = *reinterpret_cast<const uint4*>(x + at + e);
        }
    }
}

// One (row, head) unit of `norm_rope_last` from its fetched head, its
// block's frequencies in `freq_s`.
template <int PER, bool WEIGHTED, bool APPEND>
__device__ __forceinline__ void norm_rope_unit(
    const float* freq_s,
    const NormRuns<PER>& in,
    const bf16* __restrict__ weight,
    bf16* __restrict__ y,
    const i32* __restrict__ positions,
    int heads,
    float eps,
    int rotary_dim,
    bool inverse,
    std::uint8_t* __restrict__ pages,
    const u32* __restrict__ w_page,
    const u32* __restrict__ w_off,
    const u8* __restrict__ row_valid,
    int page_size)
{
    static_assert(!APPEND || PER == 16, "an fp8 row is 512 wide");
    constexpr int head_dim = 32 * PER;
    constexpr int RUN = NormRuns<PER>::RUN;
    constexpr int RUNS = NormRuns<PER>::RUNS;
    if (in.plane_row < 0) return;
    const int plane_row = in.plane_row;
    const int row = plane_row / heads;
    const int lane = threadIdx.x & 31;
    const long long at = static_cast<long long>(plane_row) * head_dim;

    bf16 v[RUNS][RUN];
#pragma unroll
    for (int r = 0; r < RUNS; ++r)
#pragma unroll
        for (int u = 0; u < RUN; ++u) v[r][u] = in.v[r][u];
    // The tail's angles a lane each (pairs `lane` and `lane + 32`), handed
    // to the lanes that hold their pairs; worked out while the head loads.
    const int offset = head_dim - rotary_dim;
    const int rope_half = rotary_dim / 2;
    const int pos = positions[row];
    float2 cs[2];
#pragma unroll
    for (int k = 0; k < 2; ++k) {
        const int d = lane + 32 * k;
        cs[k] = make_float2(1.f, 0.f);
        if (d < rope_half) {
            const float ang = (inverse ? -1.f : 1.f) * static_cast<float>(pos) * freq_s[d];
            __sincosf(ang, &cs[k].y, &cs[k].x);
        }
    }
    float local = 0.f;
#pragma unroll
    for (int r = 0; r < RUNS; ++r)
#pragma unroll
        for (int u = 0; u < RUN; ++u) {
            const float f = bf16_to_f32(v[r][u]);
            local = fmaf(f, f, local);
        }
    local = ::pie::warp_sum(local);
    const float inv_rms = rsqrtf(local / static_cast<float>(head_dim) + eps);
#pragma unroll
    for (int r = 0; r < RUNS; ++r)
#pragma unroll
        for (int u = 0; u < RUN; ++u) {
            float n = bf16_to_f32(v[r][u]) * inv_rms;
            if constexpr (WEIGHTED) n *= bf16_to_f32(weight[(lane + 32 * r) * RUN + u]);
            v[r][u] = f32_to_bf16(n);
        }

#pragma unroll
    for (int r = 0; r < RUNS; ++r) {
        const int e = (lane + 32 * r) * RUN;
        const bool tail = e >= offset;
        const int first = tail ? (e - offset) / 2 : 0;
        // Whether any of the warp's runs here is the tail's (alike across it).
        const bool any_tail = (32 * r + 31) * RUN >= offset;
#pragma unroll
        for (int u = 0; u < RUN / 2; ++u) {
            if (!any_tail) break;
            const int d = first + u;
            const float c0 = __shfl_sync(0xffffffffu, cs[0].x, d & 31);
            const float s0 = __shfl_sync(0xffffffffu, cs[0].y, d & 31);
            const float c1 = __shfl_sync(0xffffffffu, cs[1].x, d & 31);
            const float s1 = __shfl_sync(0xffffffffu, cs[1].y, d & 31);
            if (tail) {
                const float cos_v = d < 32 ? c0 : c1;
                const float sin_v = d < 32 ? s0 : s1;
                const float a = bf16_to_f32(v[r][2 * u]);
                const float b = bf16_to_f32(v[r][2 * u + 1]);
                v[r][2 * u] = f32_to_bf16(a * cos_v - b * sin_v);
                v[r][2 * u + 1] = f32_to_bf16(b * cos_v + a * sin_v);
            }
        }
        if constexpr (RUN == 4) {
            *reinterpret_cast<uint2*>(y + at + e) = *reinterpret_cast<const uint2*>(v[r]);
        } else {
            *reinterpret_cast<uint4*>(y + at + e) = *reinterpret_cast<const uint4*>(v[r]);
        }
    }
    if constexpr (APPEND) {
        namespace ds = ::pie::kv_rows;
        if (row_valid != nullptr && row_valid[plane_row] == 0) return;
        const int page = static_cast<int>(w_page[plane_row]);
        const int off = static_cast<int>(w_off[plane_row]);
        if (off < 0 || off >= page_size) return;
        const ds::DsRow dst = ds::ds_row(pages, page, off, page_size);
#pragma unroll
        for (int r = 0; r < RUNS; ++r) {
            const int e = (lane + 32 * r) * RUN;
            float f[RUN];
            float amax = 0.f;
#pragma unroll
            for (int u = 0; u < RUN; ++u) {
                f[u] = bf16_to_f32(v[r][u]);
                amax = fmaxf(amax, fabsf(f[u]));
            }
            // A tile is eight lanes' runs (alike across the warp: r says
            // whether this run is the rotated tile's).
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 1));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 2));
            amax = fmaxf(amax, __shfl_xor_sync(0xffffffffu, amax, 4));
            if (e >= ds::kDsNope) {
                *reinterpret_cast<uint4*>(dst.data + ds::kDsNope + (e - ds::kDsNope) * 2) =
                    *reinterpret_cast<const uint4*>(v[r]);
                continue;
            }
            amax = fmaxf(amax, 1e-4f);
            const float ex = ceilf(log2f(amax / ds::kFp8Max));
            const float inv = exp2f(-ex);
            std::uint8_t codes[RUN];
#pragma unroll
            for (int u = 0; u < RUN; ++u) {
                const float c = fminf(fmaxf(f[u] * inv, -ds::kFp8Max), ds::kFp8Max);
                codes[u] = static_cast<std::uint8_t>(__nv_cvt_float_to_fp8(c, __NV_SATFINITE, __NV_E4M3));
            }
            *reinterpret_cast<uint2*>(dst.data + e) = *reinterpret_cast<const uint2*>(codes);
            if ((lane & 7) == 0) {
                dst.scales[e / ds::kDsTile] = static_cast<std::uint8_t>(fmaxf(fminf(ex + 127.f, 255.f), 0.f));
            }
        }
        if (lane == 0) dst.scales[ds::kDsScales] = 0;
    }
}


template <int PER, bool WEIGHTED, bool APPEND = false>
__global__ void __launch_bounds__(256) norm_rope_last(
    const bf16* __restrict__ x,
    const bf16* __restrict__ weight,
    bf16* __restrict__ y,
    const i32* __restrict__ positions,
    int units,
    int heads,
    float eps,
    int rotary_dim,
    float theta,
    bool inverse,
    float yarn_factor,
    float yarn_low_dim,
    float yarn_high_dim,
    std::uint8_t* __restrict__ pages,
    const u32* __restrict__ w_page,
    const u32* __restrict__ w_off,
    const u8* __restrict__ row_valid,
    int page_size,
    const u32* __restrict__ win)
{
    // The tail's frequencies once a block, as the spread form works them.
    __shared__ float freq_s[64];
    if (threadIdx.x < rotary_dim / 2) {
        const int d = static_cast<int>(threadIdx.x);
        float freq = powf(theta, -2.f * static_cast<float>(d) / static_cast<float>(rotary_dim));
        if (yarn_factor > 1.f) {
            freq = yarn_original_freq(freq, yarn_factor, yarn_low_dim, yarn_high_dim, d);
        }
        freq_s[d] = freq;
    }
    __syncthreads();
    // A warp a unit, the grid striding over them where it is smaller; the
    // next unit's head loads while this one's is worked.
    const int stride = static_cast<int>(gridDim.x) * 8;
    int unit = static_cast<int>(blockIdx.x) * 8 + static_cast<int>(threadIdx.x >> 5);
    if (unit >= units) return;
    NormRuns<PER> cur;
    norm_rope_fetch<PER>(cur, unit, x, heads, win);
    for (; unit < units; unit += stride) {
        NormRuns<PER> next;
        next.plane_row = -1;
        if (unit + stride < units) norm_rope_fetch<PER>(next, unit + stride, x, heads, win);
        norm_rope_unit<PER, WEIGHTED, APPEND>(freq_s, cur, weight, y, positions, heads, eps, rotary_dim,
                                              inverse, pages, w_page, w_off, row_valid, page_size);
        cur = next;
    }
}

// DeepSeek-V4's q (scale-free, QPER a lane) and its kv plane (scaled, one
// 512-wide head, written to the cache where APPEND) normed and rotated in
// one launch: warps [0, q_units) take q's (row, head) units, the rest the
// plane's rows, the frequencies shared.
template <int QPER, bool APPEND>
__global__ void __launch_bounds__(256) norm_rope_pair(
    const bf16* __restrict__ qx,
    bf16* __restrict__ qy,
    int q_units,
    int q_heads,
    float q_eps,
    const bf16* __restrict__ kx,
    const bf16* __restrict__ kw,
    bf16* __restrict__ ky,
    int kv_units,
    float kv_eps,
    const i32* __restrict__ positions,
    int rotary_dim,
    float theta,
    float yarn_factor,
    float yarn_low_dim,
    float yarn_high_dim,
    std::uint8_t* __restrict__ pages,
    const u32* __restrict__ w_page,
    const u32* __restrict__ w_off,
    const u8* __restrict__ row_valid,
    int page_size,
    const u32* __restrict__ win)
{
    __shared__ float freq_s[64];
    if (threadIdx.x < rotary_dim / 2) {
        const int d = static_cast<int>(threadIdx.x);
        float freq = powf(theta, -2.f * static_cast<float>(d) / static_cast<float>(rotary_dim));
        if (yarn_factor > 1.f) {
            freq = yarn_original_freq(freq, yarn_factor, yarn_low_dim, yarn_high_dim, d);
        }
        freq_s[d] = freq;
    }
    __syncthreads();
    // A warp a unit, the grid striding over them where it is smaller; the
    // next q unit's head loads while this one's is worked.
    const int stride = static_cast<int>(gridDim.x) * 8;
    int unit = static_cast<int>(blockIdx.x) * 8 + static_cast<int>(threadIdx.x >> 5);
    NormRuns<QPER> cur;
    if (unit < q_units) norm_rope_fetch<QPER>(cur, unit, qx, q_heads, win);
    for (; unit < q_units; unit += stride) {
        NormRuns<QPER> next;
        next.plane_row = -1;
        if (unit + stride < q_units) norm_rope_fetch<QPER>(next, unit + stride, qx, q_heads, win);
        norm_rope_unit<QPER, false, false>(freq_s, cur, nullptr, qy, positions, q_heads, q_eps, rotary_dim,
                                           false, nullptr, nullptr, nullptr, nullptr, 0);
        cur = next;
    }
    for (; unit < q_units + kv_units; unit += stride) {
        NormRuns<16> kv;
        norm_rope_fetch<16>(kv, unit - q_units, kx, 1, win);
        norm_rope_unit<16, true, APPEND>(freq_s, kv, kw, ky, positions, 1, kv_eps, rotary_dim, false, pages,
                                         w_page, w_off, row_valid, page_size);
    }
}

}
