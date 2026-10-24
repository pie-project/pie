#pragma once

// Row gathers, splits and copies: pure data movement, 16 bytes at a time
// where the row allows it.

#include "prim/device.hip"

namespace pie::layout {

// y[n] = table[ids[n]] (an id outside the vocabulary reads row 0). One
// thread per element, or per 8 elements when VEC.
template <bool VEC>
__global__ void embed(const i32* __restrict__ ids, const bf16* __restrict__ table,
                      bf16* __restrict__ y, int hidden, int vocab, int rows, int per_row,
                      const u32* __restrict__ win) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<long long>(rows) * per_row) return;
    const Window window{win};
    const long long n = idx / per_row;
    if (!window.live(n)) return;
    const long long row = window.plane(n);
    const int h = static_cast<int>(idx % per_row);
    const i32 id = ids[row];
    const long long src = static_cast<long long>(id >= 0 && id < vocab ? id : 0) * hidden;
    if constexpr (VEC) {
        reinterpret_cast<b128*>(y + row * hidden)[h] = reinterpret_cast<const b128*>(table + src)[h];
    } else {
        y[row * hidden + h] = table[src + h];
    }
}

// The token's embedding row, scaled and added into the residual stream:
//   e = table[id], ê = e·a, y += ê, ŷ = y·b
// with a and b rounded to bf16 first and every intermediate rounded where
// it lands, as the model computes it in bf16. One thread per element.
__device__ __forceinline__ float bf16_round(float v) { return to_f32(from_f32<bf16>(v)); }

__global__ void embed_scale_add(const i32* __restrict__ ids, const bf16* __restrict__ table,
                                bf16* __restrict__ e, float a, bf16* __restrict__ e_scaled,
                                bf16* __restrict__ y, float b, bf16* __restrict__ y_scaled, int hidden,
                                int vocab, int rows, const u32* __restrict__ win) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<long long>(rows) * hidden) return;
    const Window window{win};
    const long long n = idx / hidden;
    if (!window.live(n)) return;
    const long long row = window.plane(n);
    const int h = static_cast<int>(idx % hidden);
    const i32 id = ids[row];
    const long long at = row * hidden + h;
    const bf16 ev = table[static_cast<long long>(id >= 0 && id < vocab ? id : 0) * hidden + h];
    e[at] = ev;
    const float es = bf16_round(to_f32(ev) * bf16_round(a));
    e_scaled[at] = from_f32<bf16>(es);
    const float yv = bf16_round(to_f32(y[at]) + es);
    y[at] = from_f32<bf16>(yv);
    y_scaled[at] = from_f32<bf16>(yv * bf16_round(b));
}

// embed_scale_add whose residual is one layer's slice of a stacked row,
// [.. | layer col..col+hidden | ..], landing in its own plane.
__global__ void embed_scale_add_select(const i32* __restrict__ ids, const bf16* __restrict__ table,
                                       bf16* __restrict__ e, float a, bf16* __restrict__ e_scaled,
                                       const bf16* __restrict__ stacked, int stacked_width, int col,
                                       bf16* __restrict__ y_out, float b, bf16* __restrict__ y_scaled,
                                       int hidden, int vocab, int rows, const u32* __restrict__ win) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<long long>(rows) * hidden) return;
    const Window window{win};
    const long long n = idx / hidden;
    if (!window.live(n)) return;
    const long long row = window.plane(n);
    const int h = static_cast<int>(idx % hidden);
    const i32 id = ids[row];
    const long long at = row * hidden + h;
    const bf16 ev = table[static_cast<long long>(id >= 0 && id < vocab ? id : 0) * hidden + h];
    e[at] = ev;
    const float es = bf16_round(to_f32(ev) * bf16_round(a));
    e_scaled[at] = from_f32<bf16>(es);
    const float yv = bf16_round(to_f32(stacked[row * stacked_width + col + h]) + es);
    y_out[at] = from_f32<bf16>(yv);
    y_scaled[at] = from_f32<bf16>(yv * bf16_round(b));
}

// One rank's band of a vocabulary-sharded table: rows [offset, offset +
// local) gather, every other id lands zeros (the all-reduce after sums the
// bands). One block per row.
__global__ void embed_vocab_shard(const i32* __restrict__ ids, const bf16* __restrict__ table,
                                  bf16* __restrict__ y, int hidden, int local, int offset,
                                  const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    const int id = ids[row] - offset;
    const bool mine = id >= 0 && id < local;
    const bf16* src = table + static_cast<long long>(mine ? id : 0) * hidden;
    bf16* dst = y + row * hidden;
    for (int h = threadIdx.x; h < hidden; h += blockDim.x) dst[h] = mine ? src[h] : from_f32<bf16>(0.f);
}

// An all-gather's rank-major landing, `world` planes of [rows, width], laid
// side by side: dst[r] = [src₀[r] | src₁[r] | ...]. One thread per element.
__global__ void gather_width_concat(const bf16* __restrict__ src, bf16* __restrict__ dst, int rows, int width,
                                    int world) {
    const long long wide = static_cast<long long>(width) * world;
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<long long>(rows) * wide) return;
    const long long r = idx / wide;
    const long long c = idx - r * wide;
    const long long k = c / width;
    dst[idx] = src[(k * rows + r) * width + (c - k * width)];
}

// A fused projection row [q | k | v] into its three planes. Grid: (rows,
// blocks striding the wider of q and kv).
__global__ void split_qkv(const bf16* __restrict__ src, bf16* __restrict__ q, bf16* __restrict__ k,
                          bf16* __restrict__ v, i32 q_dim, i32 kv_dim, const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    const bf16* s = src + row * (q_dim + 2 * kv_dim);
    const int first = blockIdx.y * blockDim.x + threadIdx.x;
    const int stride = blockDim.x * gridDim.y;
    for (int j = first; j < q_dim; j += stride) q[row * q_dim + j] = s[j];
    for (int j = first; j < kv_dim; j += stride) {
        k[row * kv_dim + j] = s[q_dim + j];
        v[row * kv_dim + j] = s[q_dim + kv_dim + j];
    }
}

// A packed row of q, k and v heads (q and k `qk` wide, v `vw`) into heads
// `d` wide, zero past their widths, 16 bytes a thread.
__global__ void split_qkv_padded(const bf16* __restrict__ src, bf16* __restrict__ q_out,
                                 bf16* __restrict__ k_out, bf16* __restrict__ v_out, i32 q_heads,
                                 i32 kv_heads, i32 qk, i32 vw, i32 d, const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    const int d8 = d / 8;
    const uint4* s = reinterpret_cast<const uint4*>(src) + row * ((q_heads * qk + kv_heads * (qk + vw)) / 8);
    const int out8 = (q_heads + 2 * kv_heads) * d8;
    const int stride = blockDim.x * gridDim.y;
    for (int j = blockIdx.y * blockDim.x + threadIdx.x; j < out8; j += stride) {
        int h = j / d8;
        const int c = j - h * d8;
        uint4* dst;
        int base8, width;
        if (h < q_heads) {
            dst = reinterpret_cast<uint4*>(q_out) + row * q_heads * d8;
            base8 = h * qk / 8;
            width = qk;
        } else if (h < q_heads + kv_heads) {
            h -= q_heads;
            dst = reinterpret_cast<uint4*>(k_out) + row * kv_heads * d8;
            base8 = (q_heads * qk + h * qk) / 8;
            width = qk;
        } else {
            h -= q_heads + kv_heads;
            dst = reinterpret_cast<uint4*>(v_out) + row * kv_heads * d8;
            base8 = (q_heads * qk + kv_heads * qk + h * vw) / 8;
            width = vw;
        }
        dst[h * d8 + c] = c * 8 < width ? s[base8 + c] : make_uint4(0, 0, 0, 0);
    }
}

// out[n] = table[n][offset .. offset + width]: one layer's slice of a
// stacked row. One block per row.
template <class T>
__global__ void select(const T* __restrict__ table, T* __restrict__ out, int stride, int offset, int width,
                       const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    const T* src = table + row * stride + offset;
    T* dst = out + row * width;
    for (int i = threadIdx.x; i < width; i += blockDim.x) dst[i] = src[i];
}

// A packed row of heads [q₀ g₀ | q₁ g₁ | ...] (each head_dim wide) into q
// and gate. Grid: (rows, heads).
__global__ void split_q_gate(const bf16* __restrict__ packed, bf16* __restrict__ q, bf16* __restrict__ gate,
                             int rows, int heads, int head_dim, const u32* __restrict__ win) {
    const Window window{win};
    if (static_cast<int>(blockIdx.x) >= rows || !window.live(blockIdx.x)) return;
    const long long head = window.plane(blockIdx.x) * heads + blockIdx.y;
    const bf16* src = packed + head * 2 * head_dim;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        q[head * head_dim + i] = src[i];
        gate[head * head_dim + i] = src[head_dim + i];
    }
}

// A row [left | right] into its two halves. Grid: (rows, ⌈width / 8·block⌉);
// each thread moves 8 elements, as one 16-byte access when both halves and
// all three planes allow it.
template <class T>
__global__ void split_rows(const T* __restrict__ src, T* __restrict__ left, T* __restrict__ right,
                           int left_dim, int right_dim, const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    const int total = left_dim + right_dim;
    const int i = (blockIdx.y * blockDim.x + threadIdx.x) * 8;
    if (i >= total) return;
    const T* s = src + row * total;
    T* l = left + row * left_dim;
    T* r = right + row * right_dim;
    const bool packed = sizeof(T) == 2 && left_dim % 8 == 0 && right_dim % 8 == 0 &&
                        aligned16(s) && aligned16(l) && aligned16(r);
    if (packed) {
        const b128 v = *reinterpret_cast<const b128*>(s + i);
        *reinterpret_cast<b128*>(i < left_dim ? l + i : r + (i - left_dim)) = v;
        return;
    }
    for (int j = i; j < min(i + 8, total); ++j) {
        if (j < left_dim) {
            l[j] = s[j];
        } else {
            r[j - left_dim] = s[j];
        }
    }
}

// tight[n] = wide[index[n]], `units` U per row. One block per row.
template <class U>
__global__ void gather_rows(const U* __restrict__ wide, U* __restrict__ tight,
                            const i32* __restrict__ index, int units) {
    const U* src = wide + static_cast<long long>(index[blockIdx.x]) * units;
    U* dst = tight + static_cast<long long>(blockIdx.x) * units;
    for (int i = threadIdx.x; i < units; i += blockDim.x) dst[i] = src[i];
}

// wide[index[n]] = tight[n], `units` U per row. One block per row.
template <class U>
__global__ void scatter_rows(const U* __restrict__ tight, U* __restrict__ wide,
                             const i32* __restrict__ index, int units) {
    const U* src = tight + static_cast<long long>(blockIdx.x) * units;
    U* dst = wide + static_cast<long long>(index[blockIdx.x]) * units;
    for (int i = threadIdx.x; i < units; i += blockDim.x) dst[i] = src[i];
}

// o[n] = x[perm[n]], `units` U per row. One block per row.
template <class U>
__global__ void pack_rows(const U* __restrict__ x, const i32* __restrict__ perm, U* __restrict__ o,
                          int units, const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    const U* src = x + static_cast<long long>(perm[row]) * units;
    U* dst = o + row * units;
    for (int i = threadIdx.x; i < units; i += blockDim.x) dst[i] = src[i];
}

// o[perm[n]] = x[n], `units` U per row: pack_rows undone. One block per row.
template <class U>
__global__ void unpack_rows(const U* __restrict__ x, const i32* __restrict__ perm, U* __restrict__ o,
                            int units, const u32* __restrict__ win) {
    const Window window{win};
    if (!window.live(blockIdx.x)) return;
    const long long row = window.plane(blockIdx.x);
    const U* src = x + row * units;
    U* dst = o + static_cast<long long>(perm[row]) * units;
    for (int i = threadIdx.x; i < units; i += blockDim.x) dst[i] = src[i];
}

}  // namespace pie::layout
