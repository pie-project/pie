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

// The positions request `req`'s pages hold: a padded lane owns none, and no
// position at or past this is read through `paged_slot`.
__device__ __forceinline__ int owned_positions(
    const u32* __restrict__ kv_page_indptr, int req, int page_size) {
    const int pages = static_cast<int>(kv_page_indptr[req + 1]) -
                      static_cast<int>(kv_page_indptr[req]);
    return max(pages, 0) * page_size;
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
    // A fire's indptr seats more lanes than it fills, and the unfilled ones
    // read 0: the live lanes end where the indptr first falls. The search
    // below walks only those, and a row past the last live request closes
    // nothing (it would otherwise land on a lane that owns no pages).
    __shared__ int s_live;
    if (threadIdx.x == 0) s_live = num_requests;
    __syncthreads();
    for (int r = 1 + static_cast<int>(threadIdx.x); r <= num_requests; r += blockDim.x) {
        if (qo_indptr[r] < qo_indptr[r - 1]) atomicMin(&s_live, r - 1);
    }
    __syncthreads();
    const int live = s_live;

    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const int p = positions[t];
    const bool owned = live > 0 && t < static_cast<int>(qo_indptr[live]);
    const bool valid = owned && ((row_valid == nullptr) || (row_valid[t] != 0));
    const bool is_boundary = valid && (((p + 1) % ratio) == 0);
    out_pos[t] = is_boundary ? p : -1;

    int lo = 0;
    int hi = max(live, 1);
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
__global__ void index_block_mean_paged(
    const T* __restrict__ key_pages,
    T* __restrict__ out,
    const i32* __restrict__ boundary_pos,
    const i32* __restrict__ boundary_req,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    int head_dim,
    int ratio,
    int page_size,
    const u32* __restrict__ win) {
    const int cb = blockIdx.x;
    if (win != nullptr && cb >= static_cast<int>(win[0])) return;
    const int c = win != nullptr ? cb + static_cast<int>(win[1]) : cb;
    const int bpos = boundary_pos[c];
    const int req = boundary_req[c];
    T* orow = out + static_cast<long long>(c) * head_dim;

    if (bpos < 0 || bpos >= owned_positions(kv_page_indptr, req, page_size)) {
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            orow[d] = Elem<T>::from_f32(0.f);
        }
        return;
    }

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.f;
        for (int i = 0; i < ratio; ++i) {
            const int pos = bpos + i - (ratio - 1);
            if (pos < 0) continue;
            const long long slot =
                paged_slot(kv_page_indices, kv_page_indptr, req, pos, page_size);
            acc += Elem<T>::to_f32(key_pages[slot * head_dim + d]);
        }
        orow[d] = Elem<T>::from_f32(acc / static_cast<float>(ratio));
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
    int state_pitch,
    const u32* __restrict__ win) {
    // A staged fire seats more rows than it fills, and the boundary rows
    // past the live ones were never written this fire: only the window's
    // rows are read.
    const int cb = blockIdx.x;
    if (win != nullptr && cb >= static_cast<int>(win[0])) return;
    const int c = win != nullptr ? cb + static_cast<int>(win[1]) : cb;
    const int window = coff * ratio;
    const int width = coff * head_dim;
    const long long pitch = static_cast<long long>(state_pitch);
    const int bpos = boundary_pos[c];
    const int req = boundary_req[c];

    if (bpos < 0 || bpos >= owned_positions(kv_page_indptr, req, page_size)) {
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

// `pool_gather_paged` in one pass over 16-byte runs: a run of `head_dim / 8`
// threads covers an entry's width eight columns a thread, the block's runs
// split the window between them (each folding its share as an online
// softmax), and the runs' partials merge through shared memory. The window's
// slots are looked up once a block. 256 threads; `head_dim % 8 == 0`,
// `head_dim / 8` divides 256.
template <class T>
__global__ void __launch_bounds__(256) pool_gather_vec(
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
    int state_pitch,
    const u32* __restrict__ win) {
    const int cb = blockIdx.x;
    if (win != nullptr && cb >= static_cast<int>(win[0])) return;
    const int c = win != nullptr ? cb + static_cast<int>(win[1]) : cb;
    const int window = coff * ratio;
    const int width = coff * head_dim;
    const long long pitch = static_cast<long long>(state_pitch);
    const int bpos = boundary_pos[c];
    const int req = boundary_req[c];
    const int run = head_dim / 8;
    const int lane = static_cast<int>(threadIdx.x) % run;
    const int group = static_cast<int>(threadIdx.x) / run;
    const int groups = static_cast<int>(blockDim.x) / run;
    T* orow = out + static_cast<long long>(c) * head_dim;

    if (bpos < 0 || bpos >= owned_positions(kv_page_indptr, req, page_size)) {
        if (group == 0) *reinterpret_cast<uint4*>(orow + lane * 8) = make_uint4(0u, 0u, 0u, 0u);
        return;
    }

    extern __shared__ __align__(16) unsigned char pg_smem[];
    long long* slots = reinterpret_cast<long long*>(pg_smem);
    float* part = reinterpret_cast<float*>(pg_smem + ((window * 8 + 15) & ~15));
    for (int i = threadIdx.x; i < window; i += blockDim.x) {
        const int pos = bpos + i - (window - 1);
        slots[i] = pos < 0 ? -1 : paged_slot(kv_page_indices, kv_page_indptr, req, pos, page_size);
    }
    __syncthreads();

    float m[8], sum[8], acc[8];
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        m[u] = neg_inf();
        sum[u] = 0.f;
        acc[u] = 0.f;
    }
    for (int i = group; i < window; i += groups) {
        const long long slot = slots[i];
        if (slot < 0) continue;
        const int pos = bpos + i - (window - 1);
        const int col = ((i >= ratio) ? head_dim : 0) + lane * 8;
        T sc[8], kv[8];
        *reinterpret_cast<uint4*>(sc) = *reinterpret_cast<const uint4*>(state_score + slot * pitch + col);
        *reinterpret_cast<uint4*>(kv) = *reinterpret_cast<const uint4*>(state_kv + slot * pitch + col);
        float bias[8];
        if (ape != nullptr) {
            const float4* a = reinterpret_cast<const float4*>(ape + static_cast<long long>(pos % ratio) * width + col);
            const float4 a0 = a[0], a1 = a[1];
            bias[0] = a0.x; bias[1] = a0.y; bias[2] = a0.z; bias[3] = a0.w;
            bias[4] = a1.x; bias[5] = a1.y; bias[6] = a1.z; bias[7] = a1.w;
        } else {
#pragma unroll
            for (int u = 0; u < 8; ++u) bias[u] = 0.f;
        }
#pragma unroll
        for (int u = 0; u < 8; ++u) {
            const float s = Elem<T>::to_f32(sc[u]) + bias[u];
            const float v = Elem<T>::to_f32(kv[u]);
            if (s > m[u]) {
                const float r = __expf(m[u] - s);
                sum[u] = sum[u] * r + 1.f;
                acc[u] = acc[u] * r + v;
                m[u] = s;
            } else if (s > neg_inf()) {
                const float e = __expf(s - m[u]);
                sum[u] += e;
                acc[u] = fmaf(e, v, acc[u]);
            }
        }
    }
    // Partials laid out [group][value][lane][8].
    float* mine = part + static_cast<long long>(group) * 3 * head_dim + lane * 8;
#pragma unroll
    for (int u = 0; u < 8; ++u) {
        mine[u] = m[u];
        mine[head_dim + u] = sum[u];
        mine[2 * head_dim + u] = acc[u];
    }
    __syncthreads();
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float mx = neg_inf();
        for (int g = 0; g < groups; ++g) mx = fmaxf(mx, part[g * 3 * head_dim + d]);
        float o = 0.f;
        if (isfinite(mx)) {
            float se = 0.f, a = 0.f;
            for (int g = 0; g < groups; ++g) {
                const float mg = part[g * 3 * head_dim + d];
                if (!(mg > neg_inf())) continue;
                const float r = __expf(mg - mx);
                se = fmaf(part[g * 3 * head_dim + head_dim + d], r, se);
                a = fmaf(part[g * 3 * head_dim + 2 * head_dim + d], r, a);
            }
            o = se > 0.f ? a / se : 0.f;
        }
        orow[d] = Elem<T>::from_f32(o);
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
    int page_size,
    const u32* __restrict__ win) {
    const int cb = blockIdx.x;
    if (win != nullptr && cb >= static_cast<int>(win[0])) return;
    const int c = win != nullptr ? cb + static_cast<int>(win[1]) : cb;
    if (boundary_pos[c] < 0 ||
        boundary_pos[c] >= owned_positions(kv_page_indptr, boundary_req[c], page_size))
        return;
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

    if (win != nullptr && qi >= static_cast<int>(win[0])) return;

    const int qi_row = win != nullptr ? qi + static_cast<int>(win[1]) : qi;
    const int q_head = blockIdx.y;
    const int tid = threadIdx.x;

    const int req = req_of_token[qi_row];
    const int qpos = positions[qi_row];

    const int num_visible =
        min((qpos + 1) / ratio, owned_positions(kv_page_indptr, req, page_size) / ratio);

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

// `pool_lse_paged` over the compressed rows `attention.index_topk` selected
// for each query: a row's `top_k` ids, ended early by a negative one, and an
// id not yet closed at the query's position read as nothing.
__global__ void pool_lse_selected_paged(
    const bf16* __restrict__ q,
    const bf16* __restrict__ comp_kv_pages,
    const i32* __restrict__ selection,
    bf16* __restrict__ o,
    float* __restrict__ lse_out,
    const i32* __restrict__ positions,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const i32* __restrict__ req_of_token,
    int num_q_heads,
    int head_dim,
    int ratio,
    int top_k,
    int page_size,
    float scale,
    const u32* __restrict__ win) {
    const int qi = blockIdx.x;

    if (win != nullptr && qi >= static_cast<int>(win[0])) return;

    const int qi_row = win != nullptr ? qi + static_cast<int>(win[1]) : qi;
    const int q_head = blockIdx.y;
    const int tid = threadIdx.x;

    const int req = req_of_token[qi_row];
    const int qpos = positions[qi_row];
    const int num_visible =
        min((qpos + 1) / ratio, owned_positions(kv_page_indptr, req, page_size) / ratio);
    const i32* srow = selection + static_cast<long long>(qi_row) * top_k;

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

    float local_max = neg_inf();
    if (num_visible > 0) {
        for (int n = tid; n < top_k; n += ATTN_BLOCK) {
            const int c = srow[n];
            if (c < 0) break;
            if (c >= num_visible) continue;
            const long long slot = paged_slot(kv_page_indices, kv_page_indptr, req,
                                              (c + 1) * ratio - 1, page_size);
            const bf16* k_row = comp_kv_pages + slot * head_dim;
            float dot = 0.f;
            for (int d = 0; d < head_dim; ++d) {
                dot += q_smem[d] * bf16_to_f32(k_row[d]);
            }
            local_max = fmaxf(local_max, dot * scale);
        }
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (int off = ATTN_BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) reduce[tid] = fmaxf(reduce[tid], reduce[tid + off]);
        __syncthreads();
    }
    const float row_max = reduce[0];
    __syncthreads();

    if (!isfinite(row_max)) {
        for (int d = tid; d < head_dim; d += ATTN_BLOCK) {
            o_row[d] = f32_to_bf16(0.f);
        }
        if (lse_out != nullptr && tid == 0) {
            lse_out[qi_row * num_q_heads + q_head] = neg_inf();
        }
        return;
    }

    const int dims_per_thread = (head_dim + ATTN_BLOCK - 1) / ATTN_BLOCK;
    float acc[8] = {};
    float local_z = 0.f;

    for (int n = 0; n < top_k; ++n) {
        const int c = srow[n];
        if (c < 0) break;
        if (c >= num_visible) continue;
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
        __syncthreads();
        local_z += w;

        for (int i = 0; i < dims_per_thread; ++i) {
            const int d = tid + i * ATTN_BLOCK;
            if (d < head_dim) acc[i] += w * bf16_to_f32(k_row[d]);
        }
    }

    const float inv_z = local_z > 0.f ? 1.0f / local_z : 0.f;
    if (lse_out != nullptr && tid == 0) {
        constexpr float kLog2e = 1.44269504088896340736f;
        lse_out[qi_row * num_q_heads + q_head] =
            local_z > 0.f ? ((logf(local_z) + row_max) * kLog2e) : neg_inf();
    }
    for (int i = 0; i < dims_per_thread; ++i) {
        const int d = tid + i * ATTN_BLOCK;
        if (d < head_dim) o_row[d] = f32_to_bf16(acc[i] * inv_z);
    }
}

}
