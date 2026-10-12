#pragma once

// Row gathers, splits and copies: pure data movement, 16 bytes at a time
// where the row allows it.

#include "prelude/device.hip"

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

// A packed row of heads [q₀ g₀ | q₁ g₁ | ...] (each head_dim wide) into q
// and gate. Grid: (rows, heads).
template <class T>
__global__ void split_q_gate(const T* __restrict__ packed, T* __restrict__ q, T* __restrict__ gate,
                             int rows, int heads, int head_dim, const u32* __restrict__ win) {
    const Window window{win};
    if (static_cast<int>(blockIdx.x) >= rows || !window.live(blockIdx.x)) return;
    const long long head = window.plane(blockIdx.x) * heads + blockIdx.y;
    const T* src = packed + head * 2 * head_dim;
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

}  // namespace pie::layout
