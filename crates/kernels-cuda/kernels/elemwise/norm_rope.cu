#include "prim/device.cuh"
#include "prim/paged_kv.cuh"
#include "prim/rope.cuh"
#include "prim/kv_rows.cuh"

namespace pie::elemwise {

template <int PER>
__global__ __launch_bounds__(256) void qg_qknorm_rope_append(
    const bf16* __restrict__ qg,
    const bf16* __restrict__ k,
    const bf16* __restrict__ v,
    const bf16* __restrict__ q_norm,
    const bf16* __restrict__ k_norm,
    const i32* __restrict__ positions,
    bf16* __restrict__ q_out,
    bf16* __restrict__ gate_out,
    bf16* __restrict__ k_pages,
    bf16* __restrict__ v_pages,
    const u32* __restrict__ w_page,
    const u32* __restrict__ w_off,
    const u8* __restrict__ row_valid,
    int page_size,
    int Hq,
    int Hk,
    int units,
    int rotary_dim,
    int span,
    float theta,
    float q_eps,
    float k_eps,
    const u32* __restrict__ win)
{
    static_assert(PER == 4 || PER == 8, "a lane's run is one 8- or 16-byte load");
    using Run = typename QgRun<PER>::type;
    constexpr int D = 32 * PER;
    __shared__ float freqs[D / 2];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    // The pairs' frequencies once a block, as rope_partial takes them.
    for (int pr = threadIdx.x; pr < rotary_dim / 2; pr += blockDim.x) {
        freqs[pr] = powf(theta, -static_cast<float>(pr) / static_cast<float>(span));
    }
    __syncthreads();
    const int unit = static_cast<int>(blockIdx.x) * 8 + warp;
    if (unit >= units) return;
    const int heads = Hq + Hk;
    const int n = unit / heads;
    const int head = unit - n * heads;
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const bool is_q = head < Hq;
    const int kvh = head - Hq;

    KvSlot slot{};
    if (!is_q) {
        slot.page = static_cast<int>(w_page[row]);
        slot.offset_in_page = static_cast<int>(w_off[row]);
        const bool write_this_row = (row_valid == nullptr || row_valid[row] != 0) &&
                                    slot.offset_in_page >= 0 && slot.offset_in_page < page_size;
        if (!write_this_row) return;
    }
    const bf16* x = is_q ? qg + (static_cast<long long>(row) * Hq + head) * 2 * D
                         : k + (static_cast<long long>(row) * Hk + kvh) * D;
    const bf16* w = is_q ? q_norm : k_norm;
    const float eps = is_q ? q_eps : k_eps;
    const int e0 = lane * PER;

    bf16 xv[PER], wv[PER];
    *reinterpret_cast<Run*>(xv) = *reinterpret_cast<const Run*>(x + e0);
    *reinterpret_cast<Run*>(wv) = *reinterpret_cast<const Run*>(w + e0);
    if (is_q) {
        *reinterpret_cast<Run*>(gate_out + (static_cast<long long>(row) * Hq + head) * D + e0) =
            *reinterpret_cast<const Run*>(x + D + e0);
    }
    float local = 0.f;
#pragma unroll
    for (int u = 0; u < PER; ++u) {
        const float t = bf16_to_f32(xv[u]);
        local += t * t;
    }
    local = ::pie::warp_sum(local);
    const float inv_rms = rsqrtf(local / static_cast<float>(D) + eps);
    float y[PER];
#pragma unroll
    for (int u = 0; u < PER; ++u) {
        y[u] = bf16_to_f32(f32_to_bf16(bf16_to_f32(xv[u]) * inv_rms * (bf16_to_f32(wv[u]) + 1.f)));
    }
    // A pair's mate is `span` on, `span / PER` lanes away at the same slot
    // (the caller holds span to a whole number of lanes' runs).
    const int hop = span / PER;
    float up[PER], down[PER];
#pragma unroll
    for (int u = 0; u < PER; ++u) {
        up[u] = __shfl_sync(0xffffffffu, y[u], (lane + hop) & 31);
        down[u] = __shfl_sync(0xffffffffu, y[u], (lane - hop) & 31);
    }

    const int pos = positions[row];
    const int pairs = rotary_dim / 2;
    bf16 out[PER];
#pragma unroll
    for (int u = 0; u < PER; ++u) {
        const int i = e0 + u;
        float o = y[u];
        int pair = -1;
        if (i < pairs) pair = i;
        else if (i >= span && i < span + pairs) pair = i - span;
        if (pair >= 0) {
            const float ang = static_cast<float>(pos) * freqs[pair];
            float cos_v, sin_v;
            __sincosf(ang, &sin_v, &cos_v);
            const float a = i == pair ? y[u] : down[u];
            const float b = i == pair ? up[u] : y[u];
            o = i == pair ? a * cos_v - b * sin_v : b * cos_v + a * sin_v;
        }
        out[u] = f32_to_bf16(o);
    }
    if (is_q) {
        *reinterpret_cast<Run*>(q_out + (static_cast<long long>(row) * Hq + head) * D + e0) =
            *reinterpret_cast<const Run*>(out);
    } else {
        const long long d = kv_dst_index<false>(slot, kvh * D + e0, page_size, Hk, D);
        *reinterpret_cast<Run*>(k_pages + d) = *reinterpret_cast<const Run*>(out);
        *reinterpret_cast<Run*>(v_pages + d) =
            *reinterpret_cast<const Run*>(v + (static_cast<long long>(row) * Hk + kvh) * D + e0);
    }
}

template <int BLOCK>
__global__ __launch_bounds__(BLOCK) void q_rmsnorm_rope_partial(
    const bf16* __restrict__ x,
    const bf16* __restrict__ weight,
    const i32* __restrict__ positions,
    bf16* __restrict__ y,
    int head_dim,
    int heads,
    int rotary_dim,
    int span,
    float theta,
    float eps,
    const u32* __restrict__ win)
{
    static_assert(BLOCK % 32 == 0 && BLOCK <= 1024, "whole warps");
    constexpr int kWarps = BLOCK / 32;
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const int head = blockIdx.y;
    const long long at = (static_cast<long long>(row) * heads + head) * head_dim;
    const bf16* xr = x + at;
    bf16* yr = y + at;

    float local = 0.f;
    for (int i = threadIdx.x; i < head_dim; i += BLOCK) {
        const float v = bf16_to_f32(xr[i]);
        local += v * v;
    }
    local = ::pie::warp_sum(local);
    __shared__ float partial[kWarps + 1];
    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    if (lane == 0) partial[warp] = local;
    __syncthreads();
    if (warp == 0) {
        float v = threadIdx.x < kWarps ? partial[threadIdx.x] : 0.f;
        v = ::pie::warp_sum(v);
        if (lane == 0) partial[kWarps] = v;
    }
    __syncthreads();
    const float inv_rms = rsqrtf(partial[kWarps] / static_cast<float>(head_dim) + eps);

    // the norm covers the whole head; the first rotary_dim/2 pairs turn, each
    // pair `span` apart at theta^(-i/span): span = rotary_dim/2 for a plain
    // partial rotary (HF, mlx-lm, llama.cpp), head_dim/2 for a proportional
    // one (Gemma 4's global layers).
    const int pos = positions[row];
    for (int i = threadIdx.x; i < head_dim; i += BLOCK) {
        yr[i] = f32_to_bf16(bf16_to_f32(xr[i]) * inv_rms * bf16_to_f32(weight[i]));
    }
    __syncthreads();
    for (int dim_pair = threadIdx.x; dim_pair < rotary_dim / 2; dim_pair += BLOCK) {
        const int mate = dim_pair + span;
        const float a = bf16_to_f32(xr[dim_pair]) * inv_rms * bf16_to_f32(weight[dim_pair]);
        const float b = bf16_to_f32(xr[mate]) * inv_rms * bf16_to_f32(weight[mate]);
        const float freq = powf(theta,
            -static_cast<float>(dim_pair) / static_cast<float>(span));
        const float ang = static_cast<float>(pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);
        yr[dim_pair] = f32_to_bf16(a * cos_v - b * sin_v);
        yr[mate] = f32_to_bf16(b * cos_v + a * sin_v);
    }
}

// `q_rmsnorm_rope_partial` a warp a (row, head): the head lands in shared
// memory 16 bytes a lane, the warp folds its square sum, and each lane writes
// its own 8-element runs, normed, scaled and (where a run holds a rotary pair's
// half) turned against its mate read back from the staged head. The pairs'
// frequencies are tabled once a block, each warp striding over the (row, head)
// items. The same arithmetic as the block-a-head form, element for element; a
// head is read whole before any of it is written, so y may be x. Heads are a
// whole number of 8-element runs and at most 8 * 32 * RUNS wide.
template <int WARPS, int RUNS>
__global__ __launch_bounds__(WARPS * 32) void q_rmsnorm_rope_partial_warp(
    const bf16* x,
    const bf16* __restrict__ weight,
    const i32* __restrict__ positions,
    bf16* y,
    int rows,
    int head_dim,
    int heads,
    int rotary_dim,
    int span,
    float theta,
    float eps,
    const u32* __restrict__ win)
{
    __shared__ __align__(16) uint4 staged[WARPS][32 * RUNS];
    __shared__ float freqs[128 * RUNS];
    const int pairs = rotary_dim / 2;
    for (int p = threadIdx.x; p < pairs; p += WARPS * 32) {
        freqs[p] = powf(theta, -static_cast<float>(p) / static_cast<float>(span));
    }
    __syncthreads();

    const int warp = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int live = win != nullptr ? min(rows, static_cast<int>(win[0])) : rows;
    const int base = win != nullptr ? static_cast<int>(win[1]) : 0;
    const long long items = static_cast<long long>(live) * heads;
    const uint4* wr = reinterpret_cast<const uint4*>(weight);
    uint4* mine = staged[warp];
    const bf16* head_x = reinterpret_cast<const bf16*>(mine);
    const int runs = head_dim >> 3;

    for (long long item = static_cast<long long>(blockIdx.x) * WARPS + warp; item < items;
         item += static_cast<long long>(gridDim.x) * WARPS) {
        const int n = static_cast<int>(item / heads);
        const int head = static_cast<int>(item - static_cast<long long>(n) * heads);
        const int row = n + base;
        const long long at = (static_cast<long long>(row) * heads + head) * head_dim;
        const uint4* xr = reinterpret_cast<const uint4*>(x + at);
        uint4* yr = reinterpret_cast<uint4*>(y + at);

        float local = 0.f;
#pragma unroll
        for (int k = 0; k < RUNS; ++k) {
            const int c = lane + 32 * k;
            if (c < runs) {
                const uint4 v = xr[c];
                mine[c] = v;
                const bf16* e = reinterpret_cast<const bf16*>(&v);
#pragma unroll
                for (int u = 0; u < 8; ++u) {
                    const float f = bf16_to_f32(e[u]);
                    local += f * f;
                }
            }
        }
        local = ::pie::warp_sum(local);
        __syncwarp();
        const float inv_rms = rsqrtf(local / static_cast<float>(head_dim) + eps);
        const float pos = static_cast<float>(positions[row]);

#pragma unroll
        for (int k = 0; k < RUNS; ++k) {
            const int c = lane + 32 * k;
            if (c >= runs) continue;
            const uint4 v = mine[c];
            const uint4 wv = wr[c];
            const bf16* e = reinterpret_cast<const bf16*>(&v);
            const bf16* we = reinterpret_cast<const bf16*>(&wv);
            uint4 packed;
            bf16* out = reinterpret_cast<bf16*>(&packed);
#pragma unroll
            for (int u = 0; u < 8; ++u) {
                const int i = c * 8 + u;
                const float self = bf16_to_f32(e[u]) * inv_rms * bf16_to_f32(we[u]);
                float o = self;
                int pair = -1;
                int mate = 0;
                if (i < pairs) {
                    pair = i;
                    mate = i + span;
                } else if (i >= span && i < span + pairs) {
                    pair = i - span;
                    mate = pair;
                }
                if (pair >= 0) {
                    const float other =
                        bf16_to_f32(head_x[mate]) * inv_rms * bf16_to_f32(weight[mate]);
                    float cos_v, sin_v;
                    __sincosf(pos * freqs[pair], &sin_v, &cos_v);
                    o = i == pair ? self * cos_v - other * sin_v : self * cos_v + other * sin_v;
                }
                out[u] = f32_to_bf16(o);
            }
            yr[c] = packed;
        }
        // the next item restages this warp's head
        __syncwarp();
    }
}

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
