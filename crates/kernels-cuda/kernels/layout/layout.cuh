#pragma once

#include "prelude/device.cuh"

namespace pie::layout {

template <class T>
using Elem = ::pie::Elem<T>;

template <bool VEC>
__global__ void embed(
    const i32* __restrict__ token_ids,
    const bf16* __restrict__ weight,
    bf16* __restrict__ y,
    int hidden, int vocab, int num_tokens, int per_row,
    const u32* __restrict__ win)
{
    const int idx = static_cast<int>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= num_tokens * per_row) return;
    const int n = idx / per_row;
    const int h = idx % per_row;
    // The staged-geometry seat (qkv_fused.cuh's idiom): a replay whose grid
    // was carved at a bucket retires its padded rows here, off a word the
    // fire staged, not a parameter the recording baked.
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    // And `win[1]` is where those live rows start. `token_ids` and `y` are row
    // planes handed at their base and move with it; `weight` is the VOCAB
    // bank, whose row axis is the id the vector yields, and never moves. The
    // `idx >= num_tokens * per_row` bound above is the LAUNCH's, and stays.
    const int plane_row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const i32 tid_raw = token_ids[plane_row];
    const int tid = (tid_raw >= 0 && tid_raw < vocab) ? tid_raw : 0;
    const bf16* row = weight + static_cast<long long>(tid) * hidden;
    bf16* out = y + static_cast<long long>(plane_row) * hidden;

    if constexpr (VEC) {
        reinterpret_cast<float4*>(out)[h] =
            reinterpret_cast<const float4*>(row)[h];
    } else {
        out[h] = row[h];
    }
}

template <class T>
__global__ void embed_vocab_shard(
    const i32* __restrict__ token_ids,
    const T* __restrict__ weight,
    T* __restrict__ y,
    int hidden, int local_vocab, int vocab_offset)
{
    const int n = blockIdx.x;
    const i32 tid_raw = token_ids[n];
    const int local_tid = tid_raw - vocab_offset;
    const bool in_shard = local_tid >= 0 && local_tid < local_vocab;
    const T* row =
        weight + static_cast<long long>(in_shard ? local_tid : 0) * hidden;
    T* out = y + static_cast<long long>(n) * hidden;

    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        out[h] = in_shard ? row[h] : Elem<T>::from_f32(0.0f);
    }
}

template <class T>
__global__ void deinterleave_rows(
    const T* __restrict__ fused,
    T* __restrict__       gate_out,
    T* __restrict__       up_out,
    int H)
{
    const int row = blockIdx.x;
    const T* gate_src = fused + (2 * row    ) * H;
    const T* up_src   = fused + (2 * row + 1) * H;
    T* gate_dst = gate_out + row * H;
    T* up_dst   = up_out   + row * H;
    for (int j = threadIdx.x; j < H; j += blockDim.x) {
        gate_dst[j] = gate_src[j];
        up_dst[j]   = up_src[j];
    }
}

template <class T>
__global__ void deinterleave_vec(
    const T* __restrict__ fused,
    T* __restrict__       gate_out,
    T* __restrict__       up_out,
    int I)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= I) return;
    gate_out[i] = fused[2 * i];
    up_out[i]   = fused[2 * i + 1];
}

template <class T>
__global__ void split_q_gate(
    const T* __restrict__ packed,
    T* __restrict__ q_out,
    T* __restrict__ gate_out,
    int N, int num_heads, int head_dim,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;
    const int h = blockIdx.y;
    if (n >= N || h >= num_heads) return;
    // The staged-geometry seat (qkv_fused.cuh's idiom): a replay whose grid
    // was carved at a bucket retires its padded rows here, off a word the
    // fire staged, not a parameter the recording baked.
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    // And `win[1]` is where those live rows start: `packed`, `q_out` and
    // `gate_out` are row planes handed at their base and move together. The
    // `n >= N` bound above is the LAUNCH's, and stays on the raw index.
    const int plane_row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const int twod = 2 * head_dim;
    const T* row = packed + ((long long)plane_row * num_heads + h) * twod;
    T* q_row     = q_out   + ((long long)plane_row * num_heads + h) * head_dim;
    T* gate_row  = gate_out + ((long long)plane_row * num_heads + h) * head_dim;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        q_row[i]    = row[i];
        gate_row[i] = row[head_dim + i];
    }
}

template <class T>
__global__ void concat_rows(
    const T* __restrict__ left,
    const T* __restrict__ right,
    T* __restrict__ out,
    int left_dim, int right_dim)
{
    const int n = blockIdx.x;
    const int total_dim = left_dim + right_dim;
    const T* l = left + (long long)n * left_dim;
    const T* r = right + (long long)n * right_dim;
    T* o = out + (long long)n * total_dim;
    for (int i = threadIdx.x; i < total_dim; i += blockDim.x) {
        o[i] = (i < left_dim) ? l[i] : r[i - left_dim];
    }
}

template <class T>
__global__ void split_gdn_ba(
    const T* __restrict__ ba,
    T* __restrict__ b_out,
    T* __restrict__ a_out,
    int v_h)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const T* ba_row = ba + (long long)n * (2 * v_h);
    T* b_row = b_out + (long long)n * v_h;
    T* a_row = a_out + (long long)n * v_h;
    for (int i = tid; i < v_h; i += blockDim.x) {
        b_row[i] = ba_row[i];
        a_row[i] = ba_row[v_h + i];
    }
}

template <class T>
__global__ void split_rows(
    const T* __restrict__ src,
    T* __restrict__ left,
    T* __restrict__ right,
    int left_dim, int right_dim,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;
    // The staged-geometry seat (qkv_fused.cuh's idiom): a replay whose grid
    // was carved at a bucket retires its padded rows here, off a word the
    // fire staged, not a parameter the recording baked.
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    // And `win[1]` is where those live rows start: `src` and the two halves
    // are row planes handed at their base and move together.
    const int plane_row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const int total = left_dim + right_dim;
    const T* row = src + (long long)plane_row * total;
    T* l = left + (long long)plane_row * left_dim;
    T* r = right + (long long)plane_row * right_dim;
    for (int i = threadIdx.x; i < total; i += blockDim.x) {
        if (i < left_dim) {
            l[i] = row[i];
        } else {
            r[i - left_dim] = row[i];
        }
    }
}

// Eight elements per thread, many blocks per row: the cut falls on a vector
// boundary, so each 16-byte vector lands whole on one side.
template <class T>
__global__ void split_rows_vec8(
    const T* __restrict__ src,
    T* __restrict__ left,
    T* __restrict__ right,
    int left_dim, int right_dim,
    const u32* __restrict__ win)
{
    constexpr int VEC = 8;
    const int n = blockIdx.y;
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int plane_row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const int total = left_dim + right_dim;
    const int i = (blockIdx.x * blockDim.x + threadIdx.x) * VEC;
    if (i >= total) return;
    const uint4 v = *reinterpret_cast<const uint4*>(src + (long long)plane_row * total + i);
    if (i < left_dim) {
        *reinterpret_cast<uint4*>(left + (long long)plane_row * left_dim + i) = v;
    } else {
        *reinterpret_cast<uint4*>(right + (long long)plane_row * right_dim + (i - left_dim)) = v;
    }
}

template <class T>
__global__ void select(
    const T* __restrict__ table,
    T* __restrict__ out,
    int stride,
    int offset,
    int width,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;
    // The staged-geometry seat (qkv_fused.cuh's idiom): a replay whose grid
    // was carved at a bucket retires its padded rows here, off a word the
    // fire staged, not a parameter the recording baked.
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    // And `win[1]` is where those live rows start: `table` is stacked slices
    // PER ROW, so its row axis is this launch's, and `out` shares it. The
    // `offset` picks a column and is untouched by the shift.
    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const T* src = table + (long long)row * stride + offset;
    T* dst = out + (long long)row * width;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        dst[i] = src[i];
    }
}

template <class T>
__global__ void repeat_interleave_heads(
    const T* __restrict__ in,
    T* __restrict__ out,
    int N, int kv_heads, int q_heads, int head_dim)
{
    const int n = blockIdx.x;
    const int qh = blockIdx.y;
    if (n >= N || qh >= q_heads) return;
    const int repeat = q_heads / kv_heads;
    const int kh = qh / repeat;
    const T* src =
        in + ((long long)n * kv_heads + kh) * head_dim;
    T* dst =
        out + ((long long)n * q_heads + qh) * head_dim;
    for (int i = threadIdx.x; i < head_dim; i += blockDim.x) {
        dst[i] = src[i];
    }
}

template <class T>
__global__ void split_qkv(
    const T* __restrict__ src,
    T* __restrict__ q_out,
    T* __restrict__ k_out,
    T* __restrict__ v_out,
    i32 q_dim, i32 kv_dim,
    const u32* __restrict__ win)
{
    const int n = static_cast<int>(blockIdx.y);
    // The staged-geometry seat (qkv_fused.cuh's idiom): the rows ride
    // `blockIdx.y` here, and a replay carved at a bucket retires its padded
    // ones off the LIVE-ROWS word the fire staged — one word, not
    // `split_qkv_devwin`'s `(start, count)` pair below.
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    // And `win[1]` is where those live rows START: `src` and the three
    // destinations are row planes handed at their base and move together.
    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const int stride = q_dim + 2 * kv_dim;
    const T* src_row = src + static_cast<long long>(row) * stride;

    for (int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); j < q_dim;
         j += static_cast<int>(blockDim.x * gridDim.x)) {
        q_out[static_cast<long long>(row) * q_dim + j] = src_row[j];
    }

    for (int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); j < kv_dim;
         j += static_cast<int>(blockDim.x * gridDim.x)) {
        k_out[static_cast<long long>(row) * kv_dim + j] = src_row[q_dim + j];
    }

    for (int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); j < kv_dim;
         j += static_cast<int>(blockDim.x * gridDim.x)) {
        v_out[static_cast<long long>(row) * kv_dim + j] = src_row[q_dim + kv_dim + j];
    }
}

template <class T>
__global__ void split_qkv_devwin(
    const T* __restrict__ src,
    T* __restrict__ q_out,
    T* __restrict__ k_out,
    T* __restrict__ v_out,
    const u32* __restrict__ devwin,
    i32 q_dim, i32 kv_dim)
{
    const int n = static_cast<int>(blockIdx.y);
    // `devwin` is the pre-staged device window pair, `(start, count)`:
    // word 0 is a START. The staged-geometry seat's `win` is `(count,
    // start)` — same pointer shape, opposite word order, and the rename
    // is what keeps one from ever arming the other.
    const int w0 = static_cast<int>(devwin[0]);
    const int w1 = static_cast<int>(devwin[1]);
    if (n < w0 || n >= w0 + w1) return;
    const int stride = q_dim + 2 * kv_dim;
    const T* src_row = src + static_cast<long long>(n) * stride;
    for (int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); j < q_dim;
         j += static_cast<int>(blockDim.x * gridDim.x)) {
        q_out[static_cast<long long>(n) * q_dim + j] = src_row[j];
    }
    for (int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); j < kv_dim;
         j += static_cast<int>(blockDim.x * gridDim.x)) {
        k_out[static_cast<long long>(n) * kv_dim + j] = src_row[q_dim + j];
    }
    for (int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); j < kv_dim;
         j += static_cast<int>(blockDim.x * gridDim.x)) {
        v_out[static_cast<long long>(n) * kv_dim + j] =
            src_row[q_dim + kv_dim + j];
    }
}

/// **THE ROW MOVEMENT `Fallback::Copy` IS MADE OF** (palo design §3).
///
/// A windowed consumer P4 could not seat stands over several row intervals of
/// the fire. `gather_rows` reads those rows out of the fire-wide rectangle
/// and lays them down as one; `scatter_rows` puts the answers back where they
/// came from. `index[i]` is the FIRE row the `i`-th compacted row stands at,
/// and the two kernels are the same map read in the two directions — which is
/// what makes the pair a permutation of bytes and not an arithmetic step.
///
/// `U` is a COPY UNIT and never a number: the caller picks the widest one the
/// row's byte width and both addresses admit, so a bf16 activation and an f32
/// log-sum-exp move through the same kernel and neither is rounded, promoted
/// or canonicalised on the way.
template <class U>
__global__ void gather_rows(
    const U* __restrict__ wide,
    U* __restrict__ tight,
    const i32* __restrict__ index,
    int units)
{
    const int n = static_cast<int>(blockIdx.x);
    const U* src = wide + static_cast<long long>(index[n]) * units;
    U* dst = tight + static_cast<long long>(n) * units;
    for (int i = static_cast<int>(threadIdx.x); i < units;
         i += static_cast<int>(blockDim.x)) {
        dst[i] = src[i];
    }
}

template <class U>
__global__ void scatter_rows(
    const U* __restrict__ tight,
    U* __restrict__ wide,
    const i32* __restrict__ index,
    int units)
{
    const int n = static_cast<int>(blockIdx.x);
    const U* src = tight + static_cast<long long>(n) * units;
    U* dst = wide + static_cast<long long>(index[n]) * units;
    for (int i = static_cast<int>(threadIdx.x); i < units;
         i += static_cast<int>(blockDim.x)) {
        dst[i] = src[i];
    }
}

}
