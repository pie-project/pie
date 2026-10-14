#pragma once

// The decode step's q/k/v epilogue in one pass over the packed qkv row
// [q heads | k heads | v heads]:
//
//   q, k  ← RMSNorm(x)·w, then NeoX RoPE on the first rotary_dim / 2 pairs
//           (i, i + D/2) at the row's position, θ^(−2i/D)
//   v     ← RMSNorm(x) without a weight
//
// q lands in its own plane; k and v land in the row's kv page slot, named
// by (w_page, w_off) when the engine staged them and else the last slot of
// request r's pages. A row `row_valid` marks dead writes no kv.

#include "prelude/device.hip"
#include "attn/kv.hip"
#include "elemwise/rope.hip"

namespace pie::custom {

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
    float eps, int rotary_dim) {
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
        const float inv_v = rsqrtf(ss_v / static_cast<float>(head_dim) + eps);
        for (int i = threadIdx.x; i < head_dim; i += BLOCK) v_pages[at + i] = from_f32<bf16>(to_f32(v_src[i]) * inv_v);
    }

    const float inv_rms = rsqrtf(ss / static_cast<float>(head_dim) + eps);
    const int half = head_dim / 2;
    const float* rope_row = USE_ROPE_TABLE ? rope_table + row * head_dim : nullptr;
    const int pos = USE_ROPE_TABLE ? 0 : positions[row];
    for (int i = threadIdx.x; i < half; i += BLOCK) {
        const float a = to_f32(src[i]) * inv_rms * to_f32(weight[i]);
        const float b = to_f32(src[i + half]) * inv_rms * to_f32(weight[i + half]);
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

// A 64-, 128-, 256- or 512-wide head: one wavefront per head, a block of
// blockDim / 64 wavefronts over consecutive heads of one row (grid: rows ×
// head groups), heads ordered q then kv. The row's rotation (cos, sin of
// each pair) is computed once a block into LDS rather than once a lane for
// every head. Lane l holds elements E·l .. E·l + E − 1, read and written as
// one 2E-byte access; a pair's partner D/2 away is the lane 32 across, one
// exchange.
template <int E>
struct alignas(2 * E) BfRun {
    bf16 v[E];
};

template <int HEAD_DIM, bool USE_ROPE_TABLE>
__global__ void qkv_decode_qk_norm_rope_vnorm_write_kv_warp(
    const bf16* __restrict__ packed, bf16* __restrict__ q_out, bf16* __restrict__ k_pages,
    bf16* __restrict__ v_pages, const bf16* __restrict__ q_weight, const bf16* __restrict__ k_weight,
    const i32* __restrict__ positions, const float* __restrict__ rope_table,
    const u32* __restrict__ page_indices, const u32* __restrict__ page_indptr,
    const u32* __restrict__ last_page_lens, const u32* __restrict__ w_page,
    const u32* __restrict__ w_off, const u8* __restrict__ row_valid, const u32* __restrict__ win,
    int num_q_heads, int num_kv_heads, int page_size, bool hnd_layout, float theta, float eps,
    int rotary_dim) {
    static_assert(kWave == 64 && HEAD_DIM % 64 == 0, "a lane holds whole elements of each half");
    constexpr int E = HEAD_DIM / kWave;
    constexpr int kHalf = kWave / 2;
    constexpr int kPairs = HEAD_DIM / 2;
    using Run = BfRun<E>;
    __shared__ float rot_c[kPairs], rot_s[kPairs];

    const int r = static_cast<int>(blockIdx.x);
    const Window window{win};
    if (!window.live(r)) return;
    const long long row = window.plane(r);
    const int ln = lane();
    const int heads = num_q_heads + num_kv_heads;
    const int head = static_cast<int>(blockIdx.y * (blockDim.x / kWave) + threadIdx.x / kWave);
    const bool is_q = head < num_q_heads;
    const bool live = head < heads && (is_q || row_valid == nullptr || row_valid[row] != 0);
    const int local = is_q ? head : head - num_q_heads;
    const int q_dim = num_q_heads * HEAD_DIM;
    const int kv_dim = num_kv_heads * HEAD_DIM;
    const bf16* src_row = packed + row * (q_dim + 2 * kv_dim);

    // The head's reads first, so they fly while the rotation is computed.
    Run xr{}, yr{}, wr{};
    if (live) {
        xr = reinterpret_cast<const Run*>(src_row + (is_q ? 0 : q_dim) + local * HEAD_DIM)[ln];
        if (!is_q) yr = reinterpret_cast<const Run*>(src_row + q_dim + kv_dim + local * HEAD_DIM)[ln];
        wr = reinterpret_cast<const Run*>(is_q ? q_weight : k_weight)[ln];
    }

    if constexpr (USE_ROPE_TABLE) {
        const float* rope_row = rope_table + row * HEAD_DIM;
        for (int p = threadIdx.x; p < kPairs; p += blockDim.x) {
            rot_c[p] = p < rotary_dim / 2 ? rope_row[p] : 1.f;
            rot_s[p] = p < rotary_dim / 2 ? rope_row[p + kPairs] : 0.f;
        }
    } else {
        const int pos = positions[row];
        for (int p = threadIdx.x; p < kPairs; p += blockDim.x) {
            float c, s;
            rope_angle(p, HEAD_DIM, rotary_dim, pos, theta, c, s);
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
    ss = wave_sum(ss);
    ss_v = wave_sum(ss_v);

    const float inv_rms = rsqrtf(ss / static_cast<float>(HEAD_DIM) + eps);
#pragma unroll
    for (int i = 0; i < E; ++i) x[i] *= inv_rms * to_f32(wr.v[i]);

    Run out;
#pragma unroll
    for (int i = 0; i < E; ++i) {
        const float pair = __shfl_xor(x[i], kHalf, kWave);
        const int p = (ln * E + i) % kPairs;
        out.v[i] = from_f32<bf16>(x[i] * rot_c[p] + (ln < kHalf ? -pair : pair) * rot_s[p]);
    }

    bf16* dst;
    if (is_q) {
        dst = q_out + (row * num_q_heads + local) * HEAD_DIM;
    } else {
        const long long at = kv_slot_row(page_indices, page_indptr, last_page_lens, w_page, w_off, row, r,
                                         local, num_kv_heads, HEAD_DIM, page_size, hnd_layout);
        dst = k_pages + at;
        const float inv_v = rsqrtf(ss_v / static_cast<float>(HEAD_DIM) + eps);
        Run vq;
#pragma unroll
        for (int i = 0; i < E; ++i) vq.v[i] = from_f32<bf16>(y[i] * inv_v);
        reinterpret_cast<Run*>(v_pages + at)[ln] = vq;
    }
    reinterpret_cast<Run*>(dst)[ln] = out;
}

// Qwen's gated attention prologue in one pass, from the three projections'
// rows: q heads interleaved with their gates ([q_h | gate_h] a head), k
// heads and v heads.
//
//   gate  ← the gate halves (and q's, as the split lands them)
//   q, k  ← RMSNorm(x)·(w + 1) per head (rounded), then the partial NeoX
//           rotation of pairs (i, i + span), i < rotary_dim / 2, at θ^(−i/span)
//   k, v  → the row's kv page slot (w_page, w_off), unless `row_valid` marks
//           the row dead or the slot is out of range
//
// Each step rounds where `split_q_gate`, `rmsnorm_per_head_plus_one`,
// `rope_partial` and `kv_append` store, and the norm's sum walks theirs (a
// pack of eight a lane, the wavefront's butterfly). A wavefront a head, a
// block of WAVES heads of one row (q heads then k heads); grid (rows,
// ⌈heads / WAVES⌉).
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
    const Window window{win};
    const int r = static_cast<int>(blockIdx.x);
    if (!window.live(r)) return;
    const long long row = window.plane(r);
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
    const int pos = positions[row];
    const int pairs = rotary_dim / 2;
    P8 o;
#pragma unroll
    for (int j = 0; j < 8; ++j) {
        const int e = 8 * ln + j;
        float val = e < HEAD_DIM ? nr[e] : 0.f;
        const int i = e < span ? e : e - span;
        const bool lo = e < span;
        if (e < HEAD_DIM && i < pairs && (lo || e - span < pairs)) {
            float c, s;
            elemwise::rotation(pos, i, span, theta, c, s);
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
