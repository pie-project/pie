#pragma once

#include "prelude/device.cuh"

namespace pie::attn {

template <class T>
__global__ void average_pool(
    const T* __restrict__ input,
    T* __restrict__ output,
    int N,
    int dim,
    int ratio)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_tokens = N / ratio;
    if (idx >= out_tokens * dim) return;

    const int d = idx % dim;
    const int out_tok = idx / dim;
    const int in_start = out_tok * ratio;

    float sum = 0.f;
    const int end = min(in_start + ratio, N);
    for (int t = in_start; t < end; ++t) {
        sum += Elem<T>::to_f32(input[static_cast<long long>(t) * dim + d]);
    }
    output[static_cast<long long>(out_tok) * dim + d] =
        Elem<T>::from_f32(sum / static_cast<float>(end - in_start));
}

template <class T>
__global__ void add_ape(
    T* __restrict__ data,
    const float* __restrict__ ape,
    int N_compressed,
    int dim,
    int ratio)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_compressed * dim) return;

    const int d = idx % dim;
    const int tok = idx / dim;
    const int pos_in_window = tok % ratio;

    const float val = Elem<T>::to_f32(data[idx]) +
                      ape[pos_in_window * dim + d];
    data[idx] = Elem<T>::from_f32(val);
}

template <class T>
__global__ void gated_softmax_pool(
    const T* __restrict__ kv,
    const T* __restrict__ score,
    T* __restrict__ output,
    int N,
    int dim,
    int ratio)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_tokens = N / ratio;
    if (idx >= out_tokens * dim) return;

    const int d = idx % dim;
    const int g = idx / dim;
    const int base = g * ratio;

    float max_s = neg_inf();
    for (int i = 0; i < ratio && (base + i) < N; ++i) {
        const float s = Elem<T>::to_f32(score[static_cast<long long>(base + i) * dim + d]);
        max_s = fmaxf(max_s, s);
    }

    float sum_exp = 0.f;
    float weighted_sum = 0.f;
    for (int i = 0; i < ratio && (base + i) < N; ++i) {
        const long long pos = static_cast<long long>(base + i) * dim + d;
        const float s = Elem<T>::to_f32(score[pos]);
        const float v = Elem<T>::to_f32(kv[pos]);
        const float e = expf(s - max_s);
        sum_exp += e;
        weighted_sum += v * e;
    }

    output[static_cast<long long>(g) * dim + d] =
        Elem<T>::from_f32(sum_exp > 0.f ? weighted_sum / sum_exp : 0.f);
}

constexpr int ATTN_BLOCK = 128;

struct CompressedAttnParams {
    int qo_lo;
    int qo_hi;
    int comp_offset;
    int comp_len;
    int comp_ratio;
};

__global__ void compressed_attn(
    const bf16* __restrict__ q,
    const bf16* __restrict__ comp_kv,
    bf16* __restrict__ o,
    float* __restrict__ lse_out,
    const CompressedAttnParams* __restrict__ params,
    int num_q_heads,
    int head_dim,
    float scale)
{
    const int r       = blockIdx.x;
    const int qo_off  = blockIdx.y;
    const int q_head  = blockIdx.z;
    const int tid     = threadIdx.x;

    const auto& p = params[r];
    const int qo_lo = p.qo_lo;
    const int qo_hi = p.qo_hi;
    const int comp_off = p.comp_offset;
    const int comp_len = p.comp_len;
    const int ratio = p.comp_ratio;

    if (qo_lo + qo_off >= qo_hi) return;
    const int qi = qo_lo + qo_off;

    const int num_visible = min((qo_off + 1) / ratio, comp_len);

    extern __shared__ float smem[];
    float* q_smem = smem;
    float* reduce = smem + head_dim;

    const bf16* q_row =
        q + (static_cast<long long>(qi) * num_q_heads + q_head) * head_dim;
    for (int d = tid; d < head_dim; d += ATTN_BLOCK) {
        q_smem[d] = bf16_to_f32(q_row[d]);
    }
    __syncthreads();

    bf16* o_row =
        o + (static_cast<long long>(qi) * num_q_heads + q_head) * head_dim;

    if (num_visible <= 0) {

        for (int d = tid; d < head_dim; d += ATTN_BLOCK) {
            o_row[d] = f32_to_bf16(0.f);
        }
        if (lse_out != nullptr && tid == 0) {
            lse_out[qi * num_q_heads + q_head] = neg_inf();
        }
        return;
    }

    float local_max = neg_inf();
    for (int c = tid; c < num_visible; c += ATTN_BLOCK) {
        const bf16* k_row =
            comp_kv + static_cast<long long>(comp_off + c) * head_dim;
        float dot = 0.f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q_smem[d] * bf16_to_f32(k_row[d]);
        }
        local_max = fmaxf(local_max, dot * scale);
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (int off = ATTN_BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) reduce[tid] = fmaxf(reduce[tid], reduce[tid + off]);
        __syncthreads();
    }
    const float row_max = reduce[0];

    const int dims_per_thread = (head_dim + ATTN_BLOCK - 1) / ATTN_BLOCK;
    float acc[8] = {};
    float local_z = 0.f;

    for (int c = 0; c < num_visible; ++c) {

        const bf16* k_row =
            comp_kv + static_cast<long long>(comp_off + c) * head_dim;
        float dot = 0.f;
        for (int d = tid; d < head_dim; d += ATTN_BLOCK) {
            dot += q_smem[d] * bf16_to_f32(k_row[d]);
        }
        reduce[tid] = dot;
        __syncthreads();
        for (int off = ATTN_BLOCK / 2; off > 0; off >>= 1) {
            if (tid < off) reduce[tid] += reduce[tid + off];
            __syncthreads();
        }
        const float w = expf(reduce[0] * scale - row_max);
        if (tid == 0) local_z += w;
        __syncthreads();

        for (int i = 0; i < dims_per_thread; ++i) {
            const int d = tid + i * ATTN_BLOCK;
            if (d < head_dim) {
                acc[i] += w * bf16_to_f32(k_row[d]);
            }
        }
    }

    __shared__ float z_shared;
    if (tid == 0) z_shared = local_z;
    __syncthreads();
    const float inv_z = z_shared > 0.f ? 1.0f / z_shared : 0.f;

    if (lse_out != nullptr && tid == 0) {
        constexpr float kLog2e = 1.44269504088896340736f;
        lse_out[qi * num_q_heads + q_head] =
            z_shared > 0.f ? ((logf(z_shared) + row_max) * kLog2e) : neg_inf();
    }

    for (int i = 0; i < dims_per_thread; ++i) {
        const int d = tid + i * ATTN_BLOCK;
        if (d < head_dim) {
            o_row[d] = f32_to_bf16(acc[i] * inv_z);
        }
    }
}

template <class T>
__global__ void pool_gather(
    const T* __restrict__ kv_proj,
    const T* __restrict__ score_proj,
    const float* __restrict__ ape,
    const i32* __restrict__ boundary_tok,
    const i32* __restrict__ boundary_pos,
    const i32* __restrict__ window_lo,
    T* __restrict__ out,
    int head_dim,
    int ratio,
    int coff) {
    const int c = blockIdx.x;
    const int window = coff * ratio;
    const int proj_dim = coff * head_dim;
    const int btok = boundary_tok[c];
    const int bpos = boundary_pos[c];
    const int lo = window_lo[c];

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float max_s = neg_inf();
        for (int i = 0; i < window; ++i) {
            const int rel = i - (window - 1);
            const int tok = btok + rel;
            const int pos = bpos + rel;
            if (tok < lo || pos < 0) continue;
            const int col = ((i >= ratio) ? head_dim : 0) + d;
            float s = Elem<T>::to_f32(score_proj[static_cast<long long>(tok) * proj_dim + col]);
            if (ape != nullptr) {
                s += ape[static_cast<long long>(pos % ratio) * proj_dim + col];
            }
            max_s = fmaxf(max_s, s);
        }
        if (!isfinite(max_s)) {
            out[static_cast<long long>(c) * head_dim + d] = Elem<T>::from_f32(0.0f);
            continue;
        }
        float sum_e = 0.0f;
        float acc = 0.0f;
        for (int i = 0; i < window; ++i) {
            const int rel = i - (window - 1);
            const int tok = btok + rel;
            const int pos = bpos + rel;
            if (tok < lo || pos < 0) continue;
            const int col = ((i >= ratio) ? head_dim : 0) + d;
            const long long idx = static_cast<long long>(tok) * proj_dim + col;
            float s = Elem<T>::to_f32(score_proj[idx]);
            if (ape != nullptr) {
                s += ape[static_cast<long long>(pos % ratio) * proj_dim + col];
            }
            const float e = __expf(s - max_s);
            sum_e += e;
            acc += e * Elem<T>::to_f32(kv_proj[idx]);
        }
        out[static_cast<long long>(c) * head_dim + d] =
            Elem<T>::from_f32(sum_e > 0.0f ? acc / sum_e : 0.0f);
    }
}

__device__ __forceinline__ long long paged_slot(
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    int req, int pos, int page_size) {
    const u32 page =
        kv_page_indices[kv_page_indptr[req] + pos / page_size];
    return static_cast<long long>(page) * page_size + (pos % page_size);
}

template <class T = i32>
__global__ void pool_boundary_decode(
    const i32* __restrict__ positions,
    i32* __restrict__ out_pos,
    i32* __restrict__ out_req,
    i32* __restrict__ out_rope,
    int n,
    int ratio,
    const u8* __restrict__ row_valid) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const int p = positions[t];

    const bool valid = (row_valid == nullptr) || (row_valid[t] != 0);
    const bool is_boundary = valid && (((p + 1) % ratio) == 0);
    out_pos[t] = is_boundary ? p : -1;
    out_req[t] = t;
    out_rope[t] = is_boundary ? (p / ratio) * ratio : 0;
}

template <class T = i32>
__global__ void pool_boundary_prefill(
    const i32* __restrict__ positions,
    const u32* __restrict__ qo_indptr,
    i32* __restrict__ out_pos,
    i32* __restrict__ out_req,
    i32* __restrict__ out_rope,
    int n,
    int num_requests,
    int ratio,
    const u8* __restrict__ row_valid) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const int p = positions[t];
    const bool valid = (row_valid == nullptr) || (row_valid[t] != 0);
    const bool is_boundary = valid && (((p + 1) % ratio) == 0);
    out_pos[t] = is_boundary ? p : -1;

    int lo = 0;
    int hi = num_requests;
    while (lo + 1 < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (static_cast<int>(qo_indptr[mid]) <= t) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    out_req[t] = lo;
    out_rope[t] = is_boundary ? (p / ratio) * ratio : 0;
}

// Scatters this fire's compressor projections into the rolling state
// `pool_gather_paged` below pools out of: `state_kv[slot] = wkv·x` and
// `state_score[slot] = wgate·x`, at `slot = w_page[i] * page_size + w_off[i]`
// — the SOURCE cache's own cell for token row `i`, the cell the latent
// appender writes in the same fire.
//
// **THE STATE IS ADDRESSED BY THE CACHE AND NOT BY THE FIRE**, which is why
// this is a scatter and not a rectangle: a pooling window closing at this
// fire's boundary reaches back `coff * ratio` positions and most of those
// tokens were written earlier. `paged_slot` in the gather and this `slot` are
// the same arithmetic said two ways.
//
// One block per row, one thread per column.
template <class T>
__global__ void pool_state_write(
    const T* __restrict__ kv,
    const T* __restrict__ score,
    T* __restrict__ state_kv,
    T* __restrict__ state_score,
    const u32* __restrict__ w_page,
    const u32* __restrict__ w_off,
    int width,
    int page_size,
    int state_pitch) {
    const int i = blockIdx.x;
    // A carved fire pads rows past the live ones with no cell to write.
    const int page = static_cast<int>(w_page[i]);
    const int off = static_cast<int>(w_off[i]);
    if (page < 0 || off < 0 || off >= page_size) return;
    const long long slot = static_cast<long long>(page) * page_size + off;
    const long long dst = slot * static_cast<long long>(state_pitch);
    const long long src = static_cast<long long>(i) * width;
    for (int d = threadIdx.x; d < width; d += blockDim.x) {
        state_kv[dst + d] = kv[src + d];
        state_score[dst + d] = score[src + d];
    }
}

template <class T>
__global__ void pool_gather_paged(
    const T* __restrict__ state_kv,
    const T* __restrict__ state_score,
    const float* __restrict__ ape,
    const i32* __restrict__ boundary_pos,
    const i32* __restrict__ boundary_req,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    T* __restrict__ out,
    int head_dim,
    int ratio,
    int coff,
    int page_size,
    // The ROW PITCH the two state slabs are laid out at, which is not always
    // `coff * head_dim`: one artifact can hold pooled layers at two ratios
    // (dsv4-flash carries ratio 4 and ratio 128), a reservation may lay one
    // plane at the widest of them, and a narrower gather must still stride by
    // the plane's row and read its own `coff * head_dim` columns inside it.
    // `attn/pool.metal`'s twin took this argument first; the two shaders read
    // the state at one arithmetic.
    int state_pitch) {
    const int c = blockIdx.x;
    const int window = coff * ratio;
    const int width = coff * head_dim;
    const long long pitch = static_cast<long long>(state_pitch);
    const int bpos = boundary_pos[c];
    const int req = boundary_req[c];

    if (bpos < 0) {
        T* z = out + static_cast<long long>(c) * head_dim;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            z[d] = Elem<T>::from_f32(0.f);
        }
        return;
    }

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float max_s = neg_inf();
        for (int i = 0; i < window; ++i) {
            const int pos = bpos + i - (window - 1);
            if (pos < 0) continue;
            const int col = ((i >= ratio) ? head_dim : 0) + d;
            const long long slot =
                paged_slot(kv_page_indices, kv_page_indptr, req, pos, page_size);
            float sc = Elem<T>::to_f32(state_score[slot * pitch + col]);
            if (ape != nullptr) {
                sc += ape[static_cast<long long>(pos % ratio) * width + col];
            }
            max_s = fmaxf(max_s, sc);
        }
        if (!isfinite(max_s)) {
            out[static_cast<long long>(c) * head_dim + d] = Elem<T>::from_f32(0.0f);
            continue;
        }
        float sum_e = 0.0f;
        float acc = 0.0f;
        for (int i = 0; i < window; ++i) {
            const int pos = bpos + i - (window - 1);
            if (pos < 0) continue;
            const int col = ((i >= ratio) ? head_dim : 0) + d;
            const long long slot =
                paged_slot(kv_page_indices, kv_page_indptr, req, pos, page_size);
            float sc = Elem<T>::to_f32(state_score[slot * pitch + col]);
            if (ape != nullptr) {
                sc += ape[static_cast<long long>(pos % ratio) * width + col];
            }
            const float e = __expf(sc - max_s);
            sum_e += e;
            acc += e * Elem<T>::to_f32(state_kv[slot * pitch + col]);
        }
        out[static_cast<long long>(c) * head_dim + d] =
            Elem<T>::from_f32(sum_e > 0.0f ? acc / sum_e : 0.0f);
    }
}

template <class T>
__global__ void pool_store_entries(
    const T* __restrict__ entries,
    T* __restrict__ comp_kv_pages,
    const i32* __restrict__ boundary_pos,
    const i32* __restrict__ boundary_req,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    int head_dim,
    int page_size) {
    const int c = blockIdx.x;
    if (boundary_pos[c] < 0) return;
    const long long slot = paged_slot(kv_page_indices, kv_page_indptr,
                                      boundary_req[c], boundary_pos[c], page_size);
    const T* src = entries + static_cast<long long>(c) * head_dim;
    T* dst = comp_kv_pages + slot * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) dst[d] = src[d];
}

__global__ void pool_lse_paged(
    const bf16* __restrict__ q,
    const bf16* __restrict__ comp_kv_pages,
    bf16* __restrict__ o,
    float* __restrict__ lse_out,
    const i32* __restrict__ positions,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const i32* __restrict__ req_of_token,
    int num_q_heads,
    int head_dim,
    int ratio,
    int page_size,
    float scale,
    const u32* __restrict__ win) {
    const int qi = blockIdx.x;
    // The staged-geometry seat (qkv_fused.cuh's idiom): a replay whose grid
    // was carved at a bucket retires its padded rows here, off a word the
    // fire staged, not a parameter the recording baked.
    if (win != nullptr && qi >= static_cast<int>(win[0])) return;
    // And WHERE those rows begin: an armed seat's pointers are plane bases,
    // so `win[1]` is the plane row this launch's first block owns. Every
    // per-token plane below is read there — the lane a row names and the
    // position that dates it as much as the query itself.
    const int qi_row = win != nullptr ? qi + static_cast<int>(win[1]) : qi;
    const int q_head = blockIdx.y;
    const int tid = threadIdx.x;

    const int req = req_of_token[qi_row];
    const int qpos = positions[qi_row];

    const int num_visible = (qpos + 1) / ratio;

    extern __shared__ float smem[];
    float* q_smem = smem;
    float* reduce = smem + head_dim;

    const bf16* q_row =
        q + (static_cast<long long>(qi_row) * num_q_heads + q_head) * head_dim;
    for (int d = tid; d < head_dim; d += ATTN_BLOCK) {
        q_smem[d] = bf16_to_f32(q_row[d]);
    }
    __syncthreads();

    bf16* o_row =
        o + (static_cast<long long>(qi_row) * num_q_heads + q_head) * head_dim;

    if (num_visible <= 0) {
        for (int d = tid; d < head_dim; d += ATTN_BLOCK) {
            o_row[d] = f32_to_bf16(0.f);
        }
        if (lse_out != nullptr && tid == 0) {
            lse_out[qi_row * num_q_heads + q_head] = neg_inf();
        }
        return;
    }

    float local_max = neg_inf();
    for (int c = tid; c < num_visible; c += ATTN_BLOCK) {
        const long long slot = paged_slot(kv_page_indices, kv_page_indptr, req,
                                          (c + 1) * ratio - 1, page_size);
        const bf16* k_row = comp_kv_pages + slot * head_dim;
        float dot = 0.f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q_smem[d] * bf16_to_f32(k_row[d]);
        }
        local_max = fmaxf(local_max, dot * scale);
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (int off = ATTN_BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) reduce[tid] = fmaxf(reduce[tid], reduce[tid + off]);
        __syncthreads();
    }
    const float row_max = reduce[0];

    const int dims_per_thread = (head_dim + ATTN_BLOCK - 1) / ATTN_BLOCK;
    float acc[8] = {};
    float local_z = 0.f;

    for (int c = 0; c < num_visible; ++c) {
        const long long slot = paged_slot(kv_page_indices, kv_page_indptr, req,
                                          (c + 1) * ratio - 1, page_size);
        const bf16* k_row = comp_kv_pages + slot * head_dim;
        float dot = 0.f;
        for (int d = tid; d < head_dim; d += ATTN_BLOCK) {
            dot += q_smem[d] * bf16_to_f32(k_row[d]);
        }
        reduce[tid] = dot;
        __syncthreads();
        for (int off = ATTN_BLOCK / 2; off > 0; off >>= 1) {
            if (tid < off) reduce[tid] += reduce[tid + off];
            __syncthreads();
        }
        const float w = expf(reduce[0] * scale - row_max);
        if (tid == 0) local_z += w;
        __syncthreads();

        for (int i = 0; i < dims_per_thread; ++i) {
            const int d = tid + i * ATTN_BLOCK;
            if (d < head_dim) acc[i] += w * bf16_to_f32(k_row[d]);
        }
    }

    __shared__ float z_shared;
    if (tid == 0) z_shared = local_z;
    __syncthreads();
    const float inv_z = z_shared > 0.f ? 1.0f / z_shared : 0.f;

    if (lse_out != nullptr && tid == 0) {
        constexpr float kLog2e = 1.44269504088896340736f;
        lse_out[qi_row * num_q_heads + q_head] =
            z_shared > 0.f ? ((logf(z_shared) + row_max) * kLog2e) : neg_inf();
    }
    for (int i = 0; i < dims_per_thread; ++i) {
        const int d = tid + i * ATTN_BLOCK;
        if (d < head_dim) o_row[d] = f32_to_bf16(acc[i] * inv_z);
    }
}

}
