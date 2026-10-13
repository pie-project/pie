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

// A 64-, 128- or 256-wide head: one wavefront per (row, head), WAVES to a
// block, units ordered row-major over [rows][q heads + kv heads]. Lane l
// holds elements E·l .. E·l + E − 1; a pair's partner D/2 away is the lane
// 32 across, one exchange.
template <int HEAD_DIM, bool USE_ROPE_TABLE>
__global__ void qkv_decode_qk_norm_rope_vnorm_write_kv_warp(
    const bf16* __restrict__ packed, bf16* __restrict__ q_out, bf16* __restrict__ k_pages,
    bf16* __restrict__ v_pages, const bf16* __restrict__ q_weight, const bf16* __restrict__ k_weight,
    const i32* __restrict__ positions, const float* __restrict__ rope_table,
    const u32* __restrict__ page_indices, const u32* __restrict__ page_indptr,
    const u32* __restrict__ last_page_lens, const u32* __restrict__ w_page,
    const u32* __restrict__ w_off, const u8* __restrict__ row_valid, const u32* __restrict__ win,
    int num_rows, int num_q_heads, int num_kv_heads, int page_size, bool hnd_layout, float theta,
    float eps, int rotary_dim) {
    static_assert(kWave == 64 && HEAD_DIM % 64 == 0, "a lane holds whole elements of each half");
    constexpr int E = HEAD_DIM / kWave;
    constexpr int kHalf = kWave / 2;

    const int ln = lane();
    const int heads = num_q_heads + num_kv_heads;
    const int unit = static_cast<int>(blockIdx.x * (blockDim.x / kWave) + threadIdx.x / kWave);
    if (unit >= num_rows * heads) return;
    const int r = unit / heads;
    const Window window{win};
    if (!window.live(r)) return;
    const long long row = window.plane(r);
    const int head = unit - r * heads;
    const bool is_q = head < num_q_heads;
    if (!is_q && row_valid != nullptr && row_valid[row] == 0) return;
    const int local = is_q ? head : head - num_q_heads;
    const int q_dim = num_q_heads * HEAD_DIM;
    const int kv_dim = num_kv_heads * HEAD_DIM;
    const bf16* src_row = packed + row * (q_dim + 2 * kv_dim);
    const bf16* src = src_row + (is_q ? 0 : q_dim) + local * HEAD_DIM + ln * E;
    const bf16* v_src = src_row + q_dim + kv_dim + local * HEAD_DIM + ln * E;
    const bf16* weight = (is_q ? q_weight : k_weight) + ln * E;

    float x[E], y[E];
    float ss = 0.f, ss_v = 0.f;
#pragma unroll
    for (int i = 0; i < E; ++i) {
        x[i] = to_f32(src[i]);
        ss += x[i] * x[i];
        y[i] = is_q ? 0.f : to_f32(v_src[i]);
        ss_v += y[i] * y[i];
    }
    ss = wave_sum(ss);
    ss_v = wave_sum(ss_v);

    const float inv_rms = rsqrtf(ss / static_cast<float>(HEAD_DIM) + eps);
#pragma unroll
    for (int i = 0; i < E; ++i) x[i] *= inv_rms * to_f32(weight[i]);

    const float* rope_row = USE_ROPE_TABLE ? rope_table + row * HEAD_DIM : nullptr;
    const int pos = USE_ROPE_TABLE ? 0 : positions[row];
#pragma unroll
    for (int i = 0; i < E; ++i) {
        const float pair = __shfl_xor(x[i], kHalf, kWave);
        const int p = (ln * E + i) % (HEAD_DIM / 2);
        float c, s;
        if constexpr (USE_ROPE_TABLE) {
            c = p < rotary_dim / 2 ? rope_row[p] : 1.f;
            s = p < rotary_dim / 2 ? rope_row[p + HEAD_DIM / 2] : 0.f;
        } else {
            rope_angle(p, HEAD_DIM, rotary_dim, pos, theta, c, s);
        }
        x[i] = x[i] * c + (ln < kHalf ? -pair : pair) * s;
    }

    bf16* dst;
    if (is_q) {
        dst = q_out + (row * num_q_heads + local) * HEAD_DIM;
    } else {
        const long long at = kv_slot_row(page_indices, page_indptr, last_page_lens, w_page, w_off, row, r,
                                         local, num_kv_heads, HEAD_DIM, page_size, hnd_layout);
        dst = k_pages + at;
        const float inv_v = rsqrtf(ss_v / static_cast<float>(HEAD_DIM) + eps);
#pragma unroll
        for (int i = 0; i < E; ++i) v_pages[at + ln * E + i] = from_f32<bf16>(y[i] * inv_v);
    }
#pragma unroll
    for (int i = 0; i < E; ++i) dst[ln * E + i] = from_f32<bf16>(x[i]);
}

}  // namespace pie::custom
