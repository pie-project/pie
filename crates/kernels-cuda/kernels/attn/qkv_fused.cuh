#pragma once

#include "prelude/device.cuh"

namespace pie::custom {

template <int BLOCK, bool USE_ROPE_TABLE>
__global__ void qkv_decode_qk_norm_rope_vnorm_write_kv(
    const bf16* __restrict__ packed,
    bf16* __restrict__ q_out,
    bf16* __restrict__ k_pages,
    bf16* __restrict__ v_pages,
    const bf16* __restrict__ q_weight,
    const bf16* __restrict__ k_weight,
    const i32* __restrict__ positions,
    const float* __restrict__ rope_table,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    const u32* __restrict__ w_page,
    const u32* __restrict__ w_off,
    const u8* __restrict__ row_valid,
    const u32* __restrict__ win,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    bool hnd_layout,
    float theta,
    float eps)
{
    const int r = blockIdx.x;

    if (win != nullptr && r >= static_cast<int>(win[0])) return;
    // And `win[1]` is where those live rows START: `packed`, `q_out`,
    // `positions`, `rope_table`, `row_valid` and the staged `w_page` / `w_off`
    // write tables are all row planes handed at their base, and move with it.
    // The `kv_page_indptr` / `kv_last_page_lens` FALLBACK below stays on the
    // raw block index: those are per-REQUEST prefix sums, and a window that
    // starts anywhere but row zero is admissible through the staged write
    // tables only.
    const int row = win != nullptr ? r + static_cast<int>(win[1]) : r;
    const int head_idx = blockIdx.y;
    const bool is_q = head_idx < num_q_heads;
    if (!is_q && row_valid != nullptr && row_valid[row] == 0) return;
    const int local_head = is_q ? head_idx : (head_idx - num_q_heads);
    const int q_dim = num_q_heads * head_dim;
    const int kv_dim = num_kv_heads * head_dim;
    const int packed_stride = q_dim + 2 * kv_dim;
    const bf16* src_row =
        packed + static_cast<long long>(row) * packed_stride;
    const bf16* src = is_q
        ? src_row + local_head * head_dim
        : src_row + q_dim + local_head * head_dim;
    const bf16* weight = is_q ? q_weight : k_weight;

    const bf16* v_src =
        is_q ? nullptr : src_row + q_dim + kv_dim + local_head * head_dim;
    float local = 0.f;
    float local_v = 0.f;
    for (int i = threadIdx.x; i < head_dim; i += BLOCK) {
        const float v = bf16_to_f32(src[i]);
        local += v * v;
        if (!is_q) {
            const float vv = bf16_to_f32(v_src[i]);
            local_v += vv * vv;
        }
    }

    __shared__ float buf[BLOCK];
    __shared__ float buf_v[BLOCK];
    buf[threadIdx.x] = local;
    buf_v[threadIdx.x] = local_v;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            buf[threadIdx.x] += buf[threadIdx.x + off];
            buf_v[threadIdx.x] += buf_v[threadIdx.x + off];
        }
        __syncthreads();
    }

    bf16* dst = nullptr;
    bf16* v_dst = nullptr;
    if (is_q) {
        dst = q_out + (static_cast<long long>(row) * num_q_heads + local_head) *
                      head_dim;
    } else {
        int actual_page;
        int offset_in_page;
        if (w_page != nullptr && w_off != nullptr) {
            actual_page = static_cast<int>(w_page[row]);
            offset_in_page = static_cast<int>(w_off[row]);
        } else {
            const int pages_first = kv_page_indptr[r];
            const int pages_last = kv_page_indptr[r + 1];
            const int num_pages_r = pages_last - pages_first;
            const int abs_kv_pos =
                (num_pages_r - 1) * page_size +
                static_cast<int>(kv_last_page_lens[r]) - 1;
            const int page_in_req = abs_kv_pos / page_size;
            offset_in_page = abs_kv_pos % page_size;
            actual_page = static_cast<int>(
                kv_page_indices[pages_first + page_in_req]);
        }
        if (hnd_layout) {
            const long long page_row =
                ((static_cast<long long>(actual_page) * num_kv_heads +
                  local_head) * page_size + offset_in_page) * head_dim;
            dst = k_pages + page_row;
            v_dst = v_pages + page_row;
        } else {
            const long long page_row =
                ((static_cast<long long>(actual_page) * page_size) +
                 offset_in_page) * kv_dim;
            dst = k_pages + page_row + local_head * head_dim;
            v_dst = v_pages + page_row + local_head * head_dim;
        }
    }

    if (!is_q) {
        const float inv_v =
            rsqrtf(buf_v[0] / static_cast<float>(head_dim) + eps);
        for (int i = threadIdx.x; i < head_dim; i += BLOCK) {
            v_dst[i] = f32_to_bf16(bf16_to_f32(v_src[i]) * inv_v);
        }
    }

    const float inv_rms = rsqrtf(buf[0] / static_cast<float>(head_dim) + eps);
    const int half = head_dim / 2;
    const float* rope_row = nullptr;
    int pos = 0;
    if constexpr (USE_ROPE_TABLE) {
        rope_row = rope_table + static_cast<long long>(row) * head_dim;
    } else {
        pos = positions[row];
    }
    for (int dim_pair = threadIdx.x; dim_pair < half; dim_pair += BLOCK) {
        const float a = bf16_to_f32(src[dim_pair]) *
            inv_rms * bf16_to_f32(weight[dim_pair]);
        const float b = bf16_to_f32(src[dim_pair + half]) *
            inv_rms * bf16_to_f32(weight[dim_pair + half]);
        float cos_v, sin_v;
        if constexpr (USE_ROPE_TABLE) {
            cos_v = rope_row[dim_pair];
            sin_v = rope_row[dim_pair + half];
        } else {
            const float freq = powf(
                theta,
                -2.f * static_cast<float>(dim_pair) /
                    static_cast<float>(head_dim));
            const float ang = static_cast<float>(pos) * freq;
            __sincosf(ang, &sin_v, &cos_v);
        }
        dst[dim_pair] = f32_to_bf16(a * cos_v - b * sin_v);
        dst[dim_pair + half] = f32_to_bf16(b * cos_v + a * sin_v);
    }
}

template <int HEAD_DIM, bool USE_ROPE_TABLE>
__global__ void qkv_decode_qk_norm_rope_vnorm_write_kv_warp(
    const bf16* __restrict__ packed,
    bf16* __restrict__ q_out,
    bf16* __restrict__ k_pages,
    bf16* __restrict__ v_pages,
    const bf16* __restrict__ q_weight,
    const bf16* __restrict__ k_weight,
    const i32* __restrict__ positions,
    const float* __restrict__ rope_table,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    const u32* __restrict__ w_page,
    const u32* __restrict__ w_off,
    const u8* __restrict__ row_valid,
    const u32* __restrict__ win,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int page_size,
    bool hnd_layout,
    float theta,
    float eps)
{
    constexpr unsigned FULL_MASK = 0xffffffffu;
    constexpr int ELEMS_PER_THREAD = HEAD_DIM / 32;
    static_assert(HEAD_DIM % 64 == 0);

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;
    const int total_qk_heads = num_q_heads + num_kv_heads;
    const int unit = blockIdx.x * warps_per_block + warp_id;
    if (unit >= num_requests * total_qk_heads) return;

    const int r = unit / total_qk_heads;

    if (win != nullptr && r >= static_cast<int>(win[0])) return;
    // And `win[1]` is where those live rows START: `packed`, `q_out`,
    // `positions`, `rope_table`, `row_valid` and the staged `w_page` / `w_off`
    // write tables are all row planes handed at their base, and move with it.
    // The `kv_page_indptr` / `kv_last_page_lens` FALLBACK below stays on the
    // raw block index: those are per-REQUEST prefix sums, and a window that
    // starts anywhere but row zero is admissible through the staged write
    // tables only.
    const int row = win != nullptr ? r + static_cast<int>(win[1]) : r;
    // `head_idx` un-flattens the LAUNCH unit and keeps the raw `r`.
    const int head_idx = unit - r * total_qk_heads;
    const bool is_q = head_idx < num_q_heads;
    if (!is_q && row_valid != nullptr && row_valid[row] == 0) return;
    const int local_head = is_q ? head_idx : (head_idx - num_q_heads);
    const int q_dim = num_q_heads * HEAD_DIM;
    const int kv_dim = num_kv_heads * HEAD_DIM;
    const int packed_stride = q_dim + 2 * kv_dim;
    const bf16* src_row =
        packed + static_cast<long long>(row) * packed_stride;
    const bf16* src = is_q
        ? src_row + local_head * HEAD_DIM
        : src_row + q_dim + local_head * HEAD_DIM;
    const bf16* weight = is_q ? q_weight : k_weight;

    const bf16* v_src =
        is_q ? nullptr : src_row + q_dim + kv_dim + local_head * HEAD_DIM;
    float vals[ELEMS_PER_THREAD];
    float v_vals[ELEMS_PER_THREAD];
    float sum = 0.f;
    float v_sum = 0.f;
#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        const int dim = lane * ELEMS_PER_THREAD + i;
        const float v = bf16_to_f32(src[dim]);
        vals[i] = v;
        sum += v * v;
        v_vals[i] = is_q ? 0.f : bf16_to_f32(v_src[dim]);
        v_sum += v_vals[i] * v_vals[i];
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(FULL_MASK, sum, offset, 32);
        v_sum += __shfl_xor_sync(FULL_MASK, v_sum, offset, 32);
    }

    const float inv_rms =
        rsqrtf(sum / static_cast<float>(HEAD_DIM) + eps);
#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        const int dim = lane * ELEMS_PER_THREAD + i;
        vals[i] *= inv_rms * bf16_to_f32(weight[dim]);
    }

    const int pair_offset = (HEAD_DIM / 2) / ELEMS_PER_THREAD;
    const float* rope_row = nullptr;
    int pos = 0;
    if constexpr (USE_ROPE_TABLE) {
        rope_row = rope_table + static_cast<long long>(row) * HEAD_DIM;
    } else {
        pos = positions[row];
    }
#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        const int dim = lane * ELEMS_PER_THREAD + i;
        const float pair = __shfl_xor_sync(FULL_MASK, vals[i], pair_offset, 32);
        const float signed_pair = (lane < pair_offset) ? -pair : pair;
        const int dim_pair = (dim * 2) % HEAD_DIM / 2;
        float cos_v, sin_v;
        if constexpr (USE_ROPE_TABLE) {
            cos_v = rope_row[dim_pair];
            sin_v = rope_row[dim_pair + HEAD_DIM / 2];
        } else {
            const float freq = powf(
                theta,
                -2.f * static_cast<float>(dim_pair) /
                    static_cast<float>(HEAD_DIM));
            const float ang = static_cast<float>(pos) * freq;
            __sincosf(ang, &sin_v, &cos_v);
        }
        vals[i] = vals[i] * cos_v + signed_pair * sin_v;
    }

    bf16* dst = nullptr;
    bf16* v_dst = nullptr;
    if (is_q) {
        dst = q_out + (static_cast<long long>(row) * num_q_heads + local_head) *
                      HEAD_DIM;
    } else {
        int actual_page;
        int offset_in_page;
        if (w_page != nullptr && w_off != nullptr) {
            actual_page = static_cast<int>(w_page[row]);
            offset_in_page = static_cast<int>(w_off[row]);
        } else {
            const int pages_first = kv_page_indptr[r];
            const int pages_last = kv_page_indptr[r + 1];
            const int num_pages_r = pages_last - pages_first;
            const int abs_kv_pos =
                (num_pages_r - 1) * page_size +
                static_cast<int>(kv_last_page_lens[r]) - 1;
            const int page_in_req = abs_kv_pos / page_size;
            offset_in_page = abs_kv_pos % page_size;
            actual_page = static_cast<int>(
                kv_page_indices[pages_first + page_in_req]);
        }
        if (hnd_layout) {
            const long long page_row =
                ((static_cast<long long>(actual_page) * num_kv_heads +
                  local_head) * page_size + offset_in_page) * HEAD_DIM;
            dst = k_pages + page_row;
            v_dst = v_pages + page_row;
        } else {
            const long long page_row =
                ((static_cast<long long>(actual_page) * page_size) +
                 offset_in_page) * kv_dim;
            dst = k_pages + page_row + local_head * HEAD_DIM;
            v_dst = v_pages + page_row + local_head * HEAD_DIM;
        }
    }

#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        const int dim = lane * ELEMS_PER_THREAD + i;
        dst[dim] = f32_to_bf16(vals[i]);
    }
    if (!is_q) {
        const float inv_v =
            rsqrtf(v_sum / static_cast<float>(HEAD_DIM) + eps);
#pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            const int dim = lane * ELEMS_PER_THREAD + i;
            v_dst[dim] = f32_to_bf16(v_vals[i] * inv_v);
        }
    }
}

template <int BLOCK>
__global__ void qkv_packed_qk_norm_rope_vnorm_write_kv(
    const bf16* __restrict__ packed,
    bf16* __restrict__ q_out,
    bf16* __restrict__ k_pages,
    bf16* __restrict__ v_pages,
    const bf16* __restrict__ q_weight,
    const bf16* __restrict__ k_weight,
    const i32* __restrict__ positions,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    const u8* __restrict__ row_valid,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    bool hnd_layout,
    float theta,
    float eps)
{
    const int row = blockIdx.x;
    const int head_idx = blockIdx.y;
    const bool is_q = head_idx < num_q_heads;
    if (!is_q && row_valid != nullptr && row_valid[row] == 0) return;
    const int local_head = is_q ? head_idx : (head_idx - num_q_heads);
    const int q_dim = num_q_heads * head_dim;
    const int kv_dim = num_kv_heads * head_dim;
    const int packed_stride = q_dim + 2 * kv_dim;
    const bf16* src_row =
        packed + static_cast<long long>(row) * packed_stride;
    const bf16* src = is_q
        ? src_row + local_head * head_dim
        : src_row + q_dim + local_head * head_dim;
    const bf16* weight = is_q ? q_weight : k_weight;

    float local = 0.f;
    float local_v = 0.f;
    const bf16* v_src = nullptr;
    if (!is_q) {
        v_src = src_row + q_dim + kv_dim + local_head * head_dim;
    }
    for (int i = threadIdx.x; i < head_dim; i += BLOCK) {
        const float v = bf16_to_f32(src[i]);
        local += v * v;
        if (!is_q) {
            const float vv = bf16_to_f32(v_src[i]);
            local_v += vv * vv;
        }
    }

    __shared__ float buf[BLOCK];
    __shared__ float buf_v[BLOCK];
    buf[threadIdx.x] = local;
    buf_v[threadIdx.x] = local_v;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            buf[threadIdx.x] += buf[threadIdx.x + off];
            buf_v[threadIdx.x] += buf_v[threadIdx.x + off];
        }
        __syncthreads();
    }

    bf16* dst = nullptr;
    bf16* v_dst = nullptr;
    if (is_q) {
        dst = q_out + (static_cast<long long>(row) * num_q_heads + local_head) *
                      head_dim;
    } else {
        const int pages_first = kv_page_indptr[row];
        const int pages_last = kv_page_indptr[row + 1];
        const int num_pages_r = pages_last - pages_first;
        const int abs_kv_pos =
            (num_pages_r - 1) * page_size +
            static_cast<int>(kv_last_page_lens[row]) - 1;
        const int page_in_req = abs_kv_pos / page_size;
        const int offset_in_page = abs_kv_pos % page_size;
        const int actual_page =
            static_cast<int>(kv_page_indices[pages_first + page_in_req]);
        if (hnd_layout) {
            const long long page_row =
                ((static_cast<long long>(actual_page) * num_kv_heads +
                  local_head) * page_size + offset_in_page) * head_dim;
            dst = k_pages + page_row;
            v_dst = v_pages + page_row;
        } else {
            const long long page_row =
                ((static_cast<long long>(actual_page) * page_size) +
                 offset_in_page) * kv_dim;
            dst = k_pages + page_row + local_head * head_dim;
            v_dst = v_pages + page_row + local_head * head_dim;
        }
    }

    const float inv_rms =
        rsqrtf(buf[0] / static_cast<float>(head_dim) + eps);
    const int half = head_dim / 2;
    const int pos = positions[row];
    for (int dim_pair = threadIdx.x; dim_pair < half; dim_pair += BLOCK) {
        const bf16 norm_a = f32_to_bf16(
            bf16_to_f32(src[dim_pair]) *
            inv_rms * bf16_to_f32(weight[dim_pair]));
        const bf16 norm_b = f32_to_bf16(
            bf16_to_f32(src[dim_pair + half]) *
            inv_rms * bf16_to_f32(weight[dim_pair + half]));
        const float a = bf16_to_f32(norm_a);
        const float b = bf16_to_f32(norm_b);
        const float freq = powf(
            theta,
            -2.f * static_cast<float>(dim_pair) /
                static_cast<float>(head_dim));
        const float ang = static_cast<float>(pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);
        dst[dim_pair] = f32_to_bf16(a * cos_v - b * sin_v);
        dst[dim_pair + half] = f32_to_bf16(b * cos_v + a * sin_v);
    }

    if (!is_q) {
        const float inv_v =
            rsqrtf(buf_v[0] / static_cast<float>(head_dim) + eps);
        for (int i = threadIdx.x; i < head_dim; i += BLOCK) {
            v_dst[i] = f32_to_bf16(bf16_to_f32(v_src[i]) * inv_v);
        }
    }
}

}
