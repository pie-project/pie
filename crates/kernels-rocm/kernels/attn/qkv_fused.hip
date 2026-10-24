#pragma once

// The decode step's q/k/v epilogue in one pass over the packed qkv row: q,
// k ← RMSNorm·w (w absent: scaleless; `qk_norm` off: unnormed) then NeoX
// RoPE on the first rotary_dim / 2 pairs, `span` apart; v ← unweighted
// RMSNorm when `v_norm`, times `v_scale`. q lands in its plane, k and v in
// the row's kv slot ((w_page, w_off), else request r's last slot);
// `row_valid` drops dead rows.

#include "prim/device.hip"
#include "attn/kv.hip"
#include "elemwise/rope.hip"

namespace pie::custom {

__device__ __forceinline__ float norm_weight(const bf16* __restrict__ w, int i) {
    return w != nullptr ? to_f32(w[i]) : 1.f;
}

// Where head `kv_head` of row `row` lands in the pages; `r` is the launch's
// own row, which the page tables are indexed by.
__device__ __forceinline__ long long kv_slot_row(const u32* __restrict__ page_indices,
                                                 const u32* __restrict__ page_indptr,
                                                 const u32* __restrict__ last_page_lens,
                                                 const u32* __restrict__ w_page,
                                                 const u32* __restrict__ w_off, long long row, int r,
                                                 int kv_head, int num_kv_heads, int head_dim,
                                                 int page_size, bool hnd) {
    long long page;
    int slot;
    if (w_page != nullptr && w_off != nullptr) {
        page = w_page[row];
        slot = static_cast<int>(w_off[row]);
    } else {
        const int first = static_cast<int>(page_indptr[r]);
        const int pages = static_cast<int>(page_indptr[r + 1]) - first;
        const int at = (pages - 1) * page_size + static_cast<int>(last_page_lens[r]) - 1;
        page = page_indices[first + at / page_size];
        slot = at % page_size;
    }
    return hnd ? ((page * num_kv_heads + kv_head) * page_size + slot) * head_dim
               : (page * page_size + slot) * num_kv_heads * head_dim + kv_head * head_dim;
}

// The rotation of pair i at `pos`: identity past the rotated width.
__device__ __forceinline__ void rope_angle(int i, int head_dim, int rotary_dim, int pos, float theta,
                                           float& c, float& s) {
    if (i >= rotary_dim / 2) {
        c = 1.f;
        s = 0.f;
        return;
    }
    const float freq = powf(theta, -2.f * static_cast<float>(i) / static_cast<float>(head_dim));
    sincosf(static_cast<float>(pos) * freq, &s, &c);
}

// Any head width: one block of BLOCK threads per (row, head), heads ordered
// q then kv.
template <int BLOCK, bool USE_ROPE_TABLE>
__global__ __launch_bounds__(BLOCK) void qkv_decode_qk_norm_rope_vnorm_write_kv(
    const bf16* __restrict__ packed, bf16* __restrict__ q_out, bf16* __restrict__ k_pages,
    bf16* __restrict__ v_pages, const bf16* __restrict__ q_weight, const bf16* __restrict__ k_weight,
    const i32* __restrict__ positions, const float* __restrict__ rope_table,
    const u32* __restrict__ page_indices, const u32* __restrict__ page_indptr,
    const u32* __restrict__ last_page_lens, const u32* __restrict__ w_page,
    const u32* __restrict__ w_off, const u8* __restrict__ row_valid, const u32* __restrict__ win,
    int num_q_heads, int num_kv_heads, int head_dim, int page_size, bool hnd_layout, float theta,
    float eps, bool v_norm, int rotary_dim) {
    __shared__ float red[BLOCK / kWave];
    const Window window{win};
    const int r = static_cast<int>(blockIdx.x);
    if (!window.live(r)) return;
    const long long row = window.plane(r);
    const int head = static_cast<int>(blockIdx.y);
    const bool is_q = head < num_q_heads;
    if (!is_q && row_valid != nullptr && row_valid[row] == 0) return;
    const int local = is_q ? head : head - num_q_heads;
    const int q_dim = num_q_heads * head_dim;
    const int kv_dim = num_kv_heads * head_dim;
    const bf16* src_row = packed + row * (q_dim + 2 * kv_dim);
    const bf16* src = src_row + (is_q ? 0 : q_dim) + local * head_dim;
    const bf16* v_src = src_row + q_dim + kv_dim + local * head_dim;
    const bf16* weight = is_q ? q_weight : k_weight;

    float ss = 0.f, ss_v = 0.f;
    for (int i = threadIdx.x; i < head_dim; i += BLOCK) {
        const float x = to_f32(src[i]);
        ss += x * x;
        if (!is_q) {
            const float y = to_f32(v_src[i]);
            ss_v += y * y;
        }
    }
    ss = block_sum(ss, red);
    ss_v = block_sum(ss_v, red);

    bf16* dst;
    if (is_q) {
        dst = q_out + (row * num_q_heads + local) * head_dim;
    } else {
        const long long at = kv_slot_row(page_indices, page_indptr, last_page_lens, w_page, w_off, row, r,
                                         local, num_kv_heads, head_dim, page_size, hnd_layout);
        dst = k_pages + at;
        const float inv_v = v_norm ? rsqrtf(ss_v / static_cast<float>(head_dim) + eps) : 1.f;
        for (int i = threadIdx.x; i < head_dim; i += BLOCK) v_pages[at + i] = from_f32<bf16>(to_f32(v_src[i]) * inv_v);
    }

    const float inv_rms = rsqrtf(ss / static_cast<float>(head_dim) + eps);
    const int half = head_dim / 2;
    const float* rope_row = USE_ROPE_TABLE ? rope_table + row * head_dim : nullptr;
    const int pos = USE_ROPE_TABLE || positions == nullptr ? 0 : positions[row];
    for (int i = threadIdx.x; i < half; i += BLOCK) {
        const float a = to_f32(src[i]) * inv_rms * norm_weight(weight, i);
        const float b = to_f32(src[i + half]) * inv_rms * norm_weight(weight, i + half);
        float c, s;
        if constexpr (USE_ROPE_TABLE) {
            c = i < rotary_dim / 2 ? rope_row[i] : 1.f;
            s = i < rotary_dim / 2 ? rope_row[i + half] : 0.f;
        } else {
            rope_angle(i, head_dim, rotary_dim, pos, theta, c, s);
        }
        dst[i] = from_f32<bf16>(a * c - b * s);
        dst[i + half] = from_f32<bf16>(b * c + a * s);
    }
}

// A 64-, 128-, 256- or 512-wide head: HEAD_DIM / E lanes per head (a
// wavefront at E = HEAD_DIM / 64, a 16-byte run a lane at E = 8), a block
// over consecutive heads of one row (grid: rows × head groups), heads
// ordered q then kv. The row's rotation (cos, sin of each pair) is computed
// once a block into LDS rather than once a lane for every head. Lane l of a
// head holds elements E·l .. E·l + E − 1, read and written as one 2E-byte
// access; a pair's partner `span` away is the lane span / E across, one
// exchange. The packed heads sit at their own widths (q and k `qk_width`,
// v `v_width`), landing HEAD_DIM wide, zero past their widths.
template <int E>
struct alignas(2 * E) BfRun {
    bf16 v[E];
};

// A run's e4m3 codes (an e4m3 pool's pages), each from the bf16 the
// unfused append would have read.
template <int E>
__device__ __forceinline__ void put_e4m3(bf16* pages, long long at, const BfRun<E>& x) {
    float f[E < 4 ? 4 : E] = {};
#pragma unroll
    for (int i = 0; i < E; ++i) f[i] = to_f32(x.v[i]);
    u64 c = 0;
#pragma unroll
    for (int i = 0; i < E; i += 4) c |= u64{attn::e4m3x4(f[i], f[i + 1], f[i + 2], f[i + 3])} << (8 * i);
    using U = typename conditional<E == 8, u64, typename conditional<E == 4, u32, typename conditional<E == 2, u16, u8>::type>::type>::type;
    *reinterpret_cast<U*>(reinterpret_cast<u8*>(pages) + at) = static_cast<U>(c);
}

template <int HEAD_DIM, bool USE_ROPE_TABLE, int E, bool FP8>
__global__ void qkv_decode_qk_norm_rope_vnorm_write_kv_warp(
    const bf16* __restrict__ packed, bf16* __restrict__ q_out, bf16* __restrict__ k_pages,
    bf16* __restrict__ v_pages, const bf16* __restrict__ q_weight, const bf16* __restrict__ k_weight,
    const i32* __restrict__ positions, const float* __restrict__ rope_table,
    const u32* __restrict__ page_indices, const u32* __restrict__ page_indptr,
    const u32* __restrict__ last_page_lens, const u32* __restrict__ w_page,
    const u32* __restrict__ w_off, const u8* __restrict__ row_valid, const u32* __restrict__ win,
    int num_q_heads, int num_kv_heads, int page_size, bool hnd_layout, float theta, float eps,
    bool v_norm, int rotary_dim, int span, bool qk_norm, float v_scale, int qk_width, int v_width) {
    constexpr int LPH = HEAD_DIM / E;
    static_assert(HEAD_DIM % E == 0 && LPH <= kWave && kWave % LPH == 0, "a head's lanes share a wavefront");
    constexpr int kPairs = HEAD_DIM / 2;
    using Run = BfRun<E>;
    __shared__ float rot_c[kPairs], rot_s[kPairs];

    const int r = static_cast<int>(blockIdx.x);
    const Window window{win};
    if (!window.live(r)) return;
    const long long row = window.plane(r);
    const int ln = static_cast<int>(threadIdx.x) % LPH;
    const int heads = num_q_heads + num_kv_heads;
    const int head = static_cast<int>(blockIdx.y * (blockDim.x / LPH) + threadIdx.x / LPH);
    const bool is_q = head < num_q_heads;
    const bool live = head < heads && (is_q || row_valid == nullptr || row_valid[row] != 0);
    const int local = is_q ? head : head - num_q_heads;
    const int q_src = num_q_heads * qk_width;
    const int k_src = q_src + num_kv_heads * qk_width;
    const bf16* src_row = packed + row * (k_src + num_kv_heads * v_width);
    const bf16* weight = is_q ? q_weight : k_weight;

    // The head's reads first, so they fly while the rotation is computed.
    Run xr{}, yr{}, wr{};
    if (live) {
        if (ln * E < qk_width) xr = reinterpret_cast<const Run*>(src_row + (is_q ? 0 : q_src) + local * qk_width)[ln];
        if (!is_q && ln * E < v_width) yr = reinterpret_cast<const Run*>(src_row + k_src + local * v_width)[ln];
        if (weight != nullptr) wr = reinterpret_cast<const Run*>(weight)[ln];
    }

    if constexpr (USE_ROPE_TABLE) {
        const float* rope_row = rope_table + row * HEAD_DIM;
        for (int p = threadIdx.x; p < kPairs; p += blockDim.x) {
            rot_c[p] = p < rotary_dim / 2 ? rope_row[p] : 1.f;
            rot_s[p] = p < rotary_dim / 2 ? rope_row[p + kPairs] : 0.f;
        }
    } else {
        const int pos = positions != nullptr ? positions[row] : 0;
        for (int p = threadIdx.x; p < span; p += blockDim.x) {
            float c, s;
            rope_angle(p, 2 * span, rotary_dim, pos, theta, c, s);
            rot_c[p] = c;
            rot_s[p] = s;
        }
    }
    __syncthreads();
    if (!live) return;

    float x[E], y[E];
    float ss = 0.f, ss_v = 0.f;
#pragma unroll
    for (int i = 0; i < E; ++i) {
        x[i] = to_f32(xr.v[i]);
        ss += x[i] * x[i];
        y[i] = to_f32(yr.v[i]);
        ss_v += y[i] * y[i];
    }
    if constexpr (LPH == kWave) {
        ss = wave_sum(ss);
        ss_v = wave_sum(ss_v);
    } else {
#pragma unroll
        for (int off = LPH / 2; off > 0; off >>= 1) {
            ss += __shfl_xor(ss, off, kWave);
            ss_v += __shfl_xor(ss_v, off, kWave);
        }
    }

    if (qk_norm) {
        const float inv_rms = rsqrtf(ss / static_cast<float>(HEAD_DIM) + eps);
#pragma unroll
        for (int i = 0; i < E; ++i) x[i] *= inv_rms * (weight != nullptr ? to_f32(wr.v[i]) : 1.f);
    }

    Run out;
#pragma unroll
    for (int i = 0; i < E; ++i) {
        const int dim = ln * E + i;
        const float pair = __shfl_xor(x[i], span / E, kWave);
        const int in_two = dim & (2 * span - 1);
        const int p = in_two & (span - 1);
        out.v[i] = from_f32<bf16>(dim < 2 * span ? x[i] * rot_c[p] + (in_two < span ? -pair : pair) * rot_s[p] : x[i]);
    }

    bf16* dst;
    if (is_q) {
        dst = q_out + (row * num_q_heads + local) * HEAD_DIM;
    } else {
        const long long at = kv_slot_row(page_indices, page_indptr, last_page_lens, w_page, w_off, row, r,
                                         local, num_kv_heads, HEAD_DIM, page_size, hnd_layout);
        dst = k_pages + at;
        const float inv_v = v_norm ? rsqrtf(ss_v / static_cast<float>(HEAD_DIM) + eps) : 1.f;
        Run vq;
#pragma unroll
        for (int i = 0; i < E; ++i) vq.v[i] = from_f32<bf16>(y[i] * inv_v * v_scale);
        if constexpr (FP8) {
            put_e4m3<E>(k_pages, at + ln * E, out);
            put_e4m3<E>(v_pages, at + ln * E, vq);
            return;
        }
        reinterpret_cast<Run*>(v_pages + at)[ln] = vq;
    }
    reinterpret_cast<Run*>(dst)[ln] = out;
}

// Qwen's gated attention prologue in one pass: the gates split from q, q
// and k ← RMSNorm·(w + 1) per head then the partial NeoX rotation, k and v
// to the row's kv slot unless dead. Each step rounds where `split_q_gate`,
// `rmsnorm_per_head_plus_one`, `rope_partial` and `kv_append` store. A
// wavefront a head, WAVES heads a block; grid (rows, ⌈heads / WAVES⌉).
template <int HEAD_DIM, int WAVES, bool HND>
__global__ __launch_bounds__(WAVES* kWave) void q_gate_k_norm_rope_append(
    const bf16* __restrict__ qg, const bf16* __restrict__ k, const bf16* __restrict__ v,
    const bf16* __restrict__ q_weight, const bf16* __restrict__ k_weight, const i32* __restrict__ positions,
    bf16* __restrict__ q_out, bf16* __restrict__ gate_out, bf16* __restrict__ q_split, bf16* __restrict__ k_pages,
    bf16* __restrict__ v_pages, const u32* __restrict__ w_page, const u32* __restrict__ w_off,
    const u8* __restrict__ row_valid, const u32* __restrict__ win, int num_q_heads, int num_kv_heads,
    int page_size, float q_eps, float k_eps, int rotary_dim, int span, float theta) {
    static_assert(HEAD_DIM % 8 == 0 && HEAD_DIM / 8 <= kWave, "a lane holds one pack of a head");
    constexpr int PACKS = HEAD_DIM / 8;
    using P8 = Pack8<bf16>;
    __shared__ float normed[WAVES][HEAD_DIM];
    __shared__ float rot_c[HEAD_DIM / 2], rot_s[HEAD_DIM / 2];
    const Window window{win};
    const int r = static_cast<int>(blockIdx.x);
    if (!window.live(r)) return;
    const long long row = window.plane(r);
    // The row's rotation, once a block rather than once a lane a head.
    const int pos = positions[row];
    const int pairs = rotary_dim / 2;
    for (int p = threadIdx.x; p < pairs; p += blockDim.x) elemwise::rotation(pos, p, span, theta, rot_c[p], rot_s[p]);
    __syncthreads();
    const int wave = threadIdx.x / kWave, ln = lane();
    const int head = static_cast<int>(blockIdx.y) * WAVES + wave;
    if (head >= num_q_heads + num_kv_heads) return;
    const bool is_q = head < num_q_heads;
    const int local = is_q ? head : head - num_q_heads;
    const bool lands = is_q || row_valid == nullptr || row_valid[row] != 0;
    long long page = 0;
    int slot = 0;
    bool in_page = false;
    if (!is_q) {
        page = w_page[row];
        slot = static_cast<int>(w_off[row]);
        in_page = slot >= 0 && slot < page_size;
    }
    const bf16* src = is_q ? qg + (row * num_q_heads + local) * 2 * HEAD_DIM
                           : k + (row * num_kv_heads + local) * HEAD_DIM;
    const bf16* w = is_q ? q_weight : k_weight;
    float* nr = normed[wave];
    P8 x{}, wv{};
    if (ln < PACKS) {
        x = reinterpret_cast<const P8*>(src)[ln];
        wv = reinterpret_cast<const P8*>(w)[ln];
    }
    if (is_q && ln < PACKS) {
        const long long at = (row * num_q_heads + local) * HEAD_DIM;
        reinterpret_cast<P8*>(gate_out + at)[ln] = reinterpret_cast<const P8*>(src + HEAD_DIM)[ln];
        reinterpret_cast<P8*>(q_split + at)[ln] = x;
    }
    if (!is_q && lands && in_page && ln < PACKS) {
        const long long dst = attn::kv_at<HND>(page, slot, local, 8 * ln, page_size, num_kv_heads, HEAD_DIM);
        *reinterpret_cast<P8*>(v_pages + dst) = reinterpret_cast<const P8*>(v + (row * num_kv_heads + local) * HEAD_DIM)[ln];
    }
    float local_ss = 0.f;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const float f = to_f32(x.v[j]);
        local_ss += f * f;
    }
    // The norm kernel's block sum: its first wavefront holds every pack, the
    // rest add zeros.
    const float total = 0.f + wave_sum(ln < PACKS ? local_ss : 0.f);
    const float inv = rsqrtf(total / static_cast<float>(HEAD_DIM) + (is_q ? q_eps : k_eps));
    if (ln < PACKS) {
#pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float sc = to_f32(wv.v[j]) + 1.f;
            nr[8 * ln + j] = to_f32(from_f32<bf16>(to_f32(x.v[j]) * inv * sc));
        }
    }
    __builtin_amdgcn_fence(__ATOMIC_RELEASE, "wavefront");
    __builtin_amdgcn_wave_barrier();
    __builtin_amdgcn_fence(__ATOMIC_ACQUIRE, "wavefront");
    P8 o;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int e = 8 * ln + j;
        float val = e < HEAD_DIM ? nr[e] : 0.f;
        const int i = e < span ? e : e - span;
        const bool lo = e < span;
        if (e < HEAD_DIM && i < pairs && (lo || e - span < pairs)) {
            const float c = rot_c[i], s = rot_s[i];
            const float a = nr[i], b = nr[i + span];
            val = lo ? a * c - b * s : b * c + a * s;
        }
        o.v[j] = from_f32<bf16>(val);
    }
    if (ln >= PACKS) return;
    if (is_q) {
        reinterpret_cast<P8*>(q_out + (row * num_q_heads + local) * HEAD_DIM)[ln] = o;
    } else if (lands && in_page) {
        const long long dst = attn::kv_at<HND>(page, slot, local, 8 * ln, page_size, num_kv_heads, HEAD_DIM);
        *reinterpret_cast<P8*>(k_pages + dst) = o;
    }
}

}  // namespace pie::custom
