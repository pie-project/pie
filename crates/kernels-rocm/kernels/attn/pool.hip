#pragma once

// Compressed KV pools (DeepSeek-V4's CSA/HCA, GLM's pooled index keys): a
// pool keeps one entry per `ratio` tokens, at the cell of the block's last
// token (position (c + 1)·ratio − 1 for entry c). The step finds its block
// boundaries, folds the block's per-token state into the entry (a gated
// softmax pool, or a plain mean of index keys), stores it, and attends over
// the pool: every entry, or the top-k an indexer selected.

#include "prelude/device.hip"
#include "attn/mla.hip"

namespace pie::attn {

__device__ __forceinline__ long long pool_slot(const i32* __restrict__ page_indices,
                                               const i32* __restrict__ page_indptr, int req, int pos,
                                               int page_size) {
    return static_cast<long long>(page_indices[page_indptr[req] + pos / page_size]) * page_size + pos % page_size;
}

// --------------------------------------------------------- boundaries --

// Token t closes a block when (p + 1) % ratio == 0: its boundary position
// is p (−1 otherwise), its request t (one token a request in decode), its
// entry's rope position the block's first token.
template <class T = i32>
__global__ void pool_boundary_decode(const i32* __restrict__ positions, i32* __restrict__ out_pos,
                                     i32* __restrict__ out_req, i32* __restrict__ out_rope, int n, int ratio,
                                     const u8* __restrict__ row_valid) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const int p = positions[t];
    const bool boundary = (row_valid == nullptr || row_valid[t] != 0) && (p + 1) % ratio == 0;
    out_pos[t] = boundary ? p : -1;
    out_req[t] = t;
    out_rope[t] = boundary ? (p / ratio) * ratio : 0;
}

template <class T = i32>
__global__ void pool_boundary_prefill(const i32* __restrict__ positions, const i32* __restrict__ qo_indptr,
                                      i32* __restrict__ out_pos, i32* __restrict__ out_req,
                                      i32* __restrict__ out_rope, int n, int num_requests, int ratio,
                                      const u8* __restrict__ row_valid) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const int p = positions[t];
    const bool boundary = (row_valid == nullptr || row_valid[t] != 0) && (p + 1) % ratio == 0;
    out_pos[t] = boundary ? p : -1;
    out_req[t] = request_of(qo_indptr, num_requests, t);
    out_rope[t] = boundary ? (p / ratio) * ratio : 0;
}

// ------------------------------------------------------------- state --

// Row i's compressor projections (kv and score, `width` each) into the
// state slot its write page and offset name. One block per row.
template <class T>
__global__ void pool_state_write(const T* __restrict__ kv, const T* __restrict__ score, T* __restrict__ state_kv,
                                 T* __restrict__ state_score, const i32* __restrict__ w_page,
                                 const i32* __restrict__ w_off, int width, int page_size, int state_pitch) {
    const int i = blockIdx.x;
    const int page = w_page[i], off = w_off[i];
    if (page < 0 || off < 0 || off >= page_size) return;
    const long long dst = (static_cast<long long>(page) * page_size + off) * state_pitch;
    const long long src = static_cast<long long>(i) * width;
    for (int d = threadIdx.x; d < width; d += blockDim.x) {
        state_kv[dst + d] = kv[src + d];
        state_score[dst + d] = score[src + d];
    }
}

// A boundary's entry from the state of its window: coff·ratio tokens ending
// at the boundary (two overlapping blocks when coff is 2, the earlier one
// in the upper half of the projections), softmax-gated per column by the
// scores (plus the absolute position embedding, when given). Positions
// before 0 are skipped; a row that is no boundary gets zeros.
template <class T>
__global__ void pool_gather_paged(const T* __restrict__ state_kv, const T* __restrict__ state_score,
                                  const float* __restrict__ ape, const i32* __restrict__ boundary_pos,
                                  const i32* __restrict__ boundary_req, const i32* __restrict__ page_indices,
                                  const i32* __restrict__ page_indptr, T* __restrict__ out, int head_dim, int ratio,
                                  int coff, int page_size, int state_pitch) {
    const int c = blockIdx.x;
    const int window = coff * ratio, width = coff * head_dim;
    const long long pitch = state_pitch;
    const int bpos = boundary_pos[c], req = boundary_req[c];
    T* orow = out + static_cast<long long>(c) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        if (bpos < 0) {
            orow[d] = from_f32<T>(0.f);
            continue;
        }
        float m = neg_inf(), sum = 0.f, acc = 0.f;
        for (int i = 0; i < window; ++i) {
            const int pos = bpos + i - (window - 1);
            if (pos < 0) continue;
            const int col = (i >= ratio ? head_dim : 0) + d;
            const long long at = pool_slot(page_indices, page_indptr, req, pos, page_size) * pitch + col;
            float s = to_f32(state_score[at]);
            if (ape != nullptr) s += ape[static_cast<long long>(pos % ratio) * width + col];
            const float v = to_f32(state_kv[at]);
            const float m_new = fmaxf(m, s);
            const float a = expf(m - m_new), e = expf(s - m_new);
            sum = sum * a + e;
            acc = acc * a + e * v;
            m = m_new;
        }
        orow[d] = from_f32<T>(sum > 0.f ? acc / sum : 0.f);
    }
}

// The mean of the index keys of a boundary's block (ratio cells ending at
// it; cells before position 0 count as zero). A row that is no boundary
// gets zeros. One block per row.
template <class T>
__global__ void index_block_mean_paged(const T* __restrict__ key_pages, T* __restrict__ out,
                                       const i32* __restrict__ boundary_pos, const i32* __restrict__ boundary_req,
                                       const i32* __restrict__ page_indices, const i32* __restrict__ page_indptr,
                                       int head_dim, int ratio, int page_size, const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long c = window.plane(blockIdx.x);
    const int bpos = boundary_pos[c], req = boundary_req[c];
    T* orow = out + c * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float acc = 0.f;
        if (bpos >= 0) {
            for (int i = 0; i < ratio; ++i) {
                const int pos = bpos + i - (ratio - 1);
                if (pos < 0) continue;
                acc += to_f32(key_pages[pool_slot(page_indices, page_indptr, req, pos, page_size) * head_dim + d]);
            }
        }
        orow[d] = from_f32<T>(bpos >= 0 ? acc / static_cast<float>(ratio) : 0.f);
    }
}

// A boundary row's entry into the pool cell of its position.
template <class T>
__global__ void pool_store_entries(const T* __restrict__ entries, T* __restrict__ pages,
                                   const i32* __restrict__ boundary_pos, const i32* __restrict__ boundary_req,
                                   const i32* __restrict__ page_indices, const i32* __restrict__ page_indptr,
                                   int head_dim, int page_size) {
    const int c = blockIdx.x;
    if (boundary_pos[c] < 0) return;
    const long long slot = pool_slot(page_indices, page_indptr, boundary_req[c], boundary_pos[c], page_size);
    copy_row(entries + static_cast<long long>(c) * head_dim, pages + slot * head_dim, head_dim);
}

// --------------------------------------------------------- attention --
//
// A token at position q, request r, sees the (q + 1) / ratio entries of
// r's pool before it, each one row that is both key and value (MQA over the
// pool). The 512-wide entries take the latent walk of attn/mla.hip
// (`latent_gather` with positions); this is the walk for any other width,
// a block per (token, G heads), with or without a selection of entries.
__global__ __launch_bounds__(mla_naive::kBlock) void pool_lse_paged(
    const bf16* __restrict__ q, const bf16* __restrict__ entries, bf16* __restrict__ o, float* __restrict__ lse,
    const i32* __restrict__ positions, const i32* __restrict__ page_indices, const i32* __restrict__ page_indptr,
    const i32* __restrict__ req_of_token, const i32* __restrict__ selection, int top_k, int heads, int head_dim,
    int ratio, int page_size, float sm_scale, int G, const u32* __restrict__ win) {
    extern __shared__ float pool_smem[];
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long t = window.plane(blockIdx.x);
    const int K = mla_naive::kWaves / G;
    const int h = blockIdx.y * G + (threadIdx.x / kWave) / K;
    const int req = req_of_token[t];
    const int visible = (positions[t] + 1) / ratio;
    const LatentKeys keys{selection != nullptr ? selection + t * top_k : nullptr,
                          page_indices + page_indptr[req],
                          page_size,
                          ratio,
                          0,
                          selection != nullptr ? top_k : visible,
                          visible - 1};
    const long long head0 = t * heads + static_cast<long long>(blockIdx.y) * G;
    mla_naive::attend(
        q + (t * heads + h) * head_dim, nullptr, entries, nullptr, head_dim, 0, keys,
        sm_scale * 1.4426950408889634f, G, pool_smem,
        [&](int g, int d, float v) { o[(head0 + g) * head_dim + d] = from_f32<bf16>(v); },
        [&](int g, float v) {
            if (lse != nullptr) lse[head0 + g] = v;
        });
}

}  // namespace pie::attn
