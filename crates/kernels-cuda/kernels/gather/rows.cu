#include <cstdint>
#include "prim/device.cuh"

namespace pie::gather {

template <class T>
using Elem = ::pie::Elem<T>;

template <class T>
__global__ void gather_width_concat(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int rows, int width, int world)
{
    const long long wide = static_cast<long long>(width) * world;
    const long long total = static_cast<long long>(rows) * wide;
    const long long idx =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    const long long r = idx / wide;
    const long long c = idx - r * wide;
    const long long k = c / width;
    const long long w = c - k * width;
    dst[idx] = src[(k * rows + r) * width + w];
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

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

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

// `split_q_gate` 16 bytes a lane: a thread moves one eight-element run of
// q and the matching run of the gate; `runs` is `head_dim / 8`.
template <class T>
__global__ void split_q_gate_vec8(
    const T* __restrict__ packed,
    T* __restrict__ q_out,
    T* __restrict__ gate_out,
    long long total, int num_heads, int runs,
    const u32* __restrict__ win)
{
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const long long unit = i / runs;
    const int c = static_cast<int>(i - unit * runs);
    if (win != nullptr && unit / num_heads >= static_cast<long long>(win[0])) return;
    const long long u = win != nullptr
        ? unit + static_cast<long long>(win[1]) * num_heads
        : unit;
    const long long head_dim = static_cast<long long>(runs) * 8;
    const T* row = packed + u * 2 * head_dim + c * 8;
    *reinterpret_cast<uint4*>(q_out + u * head_dim + c * 8) = *reinterpret_cast<const uint4*>(row);
    *reinterpret_cast<uint4*>(gate_out + u * head_dim + c * 8) =
        *reinterpret_cast<const uint4*>(row + head_dim);
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

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

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

template <class T>
__global__ void split_rows_vec8(
    const T* __restrict__ src,
    T* __restrict__ left,
    T* __restrict__ right,
    int left_dim, int right_dim,
    const u32* __restrict__ win)
{
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    constexpr int VEC = 8;

    const int n = blockIdx.x;
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int plane_row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const int total = left_dim + right_dim;
    const int i = (blockIdx.y * blockDim.x + threadIdx.x) * VEC;
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

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const T* src = table + (long long)row * stride + offset;
    T* dst = out + (long long)row * width;
    for (int i = threadIdx.x; i < width; i += blockDim.x) {
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

    const int n = static_cast<int>(blockIdx.x);

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const int stride = q_dim + 2 * kv_dim;
    const T* src_row = src + static_cast<long long>(row) * stride;

    for (int j = static_cast<int>(blockIdx.y * blockDim.x + threadIdx.x); j < q_dim;
         j += static_cast<int>(blockDim.x * gridDim.y)) {
        q_out[static_cast<long long>(row) * q_dim + j] = src_row[j];
    }

    for (int j = static_cast<int>(blockIdx.y * blockDim.x + threadIdx.x); j < kv_dim;
         j += static_cast<int>(blockDim.x * gridDim.y)) {
        k_out[static_cast<long long>(row) * kv_dim + j] = src_row[q_dim + j];
    }

    for (int j = static_cast<int>(blockIdx.y * blockDim.x + threadIdx.x); j < kv_dim;
         j += static_cast<int>(blockDim.x * gridDim.y)) {
        v_out[static_cast<long long>(row) * kv_dim + j] = src_row[q_dim + kv_dim + j];
    }
}

// split_qkv eight columns a thread: q_dim and kv_dim are multiples of 8 and
// every plane 16-byte aligned, so each run is one 16-byte load and store.
template <class T>
__global__ void split_qkv_vec8(
    const T* __restrict__ src,
    T* __restrict__ q_out,
    T* __restrict__ k_out,
    T* __restrict__ v_out,
    i32 q_dim, i32 kv_dim,
    const u32* __restrict__ win)
{
    static_assert(sizeof(T) == 2, "eight columns are one 16-byte run");
    const int n = static_cast<int>(blockIdx.x);
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const int q8 = q_dim / 8;
    const int kv8 = kv_dim / 8;
    const uint4* src_row =
        reinterpret_cast<const uint4*>(src) + static_cast<long long>(row) * (q8 + 2 * kv8);
    uint4* q_row = reinterpret_cast<uint4*>(q_out) + static_cast<long long>(row) * q8;
    uint4* k_row = reinterpret_cast<uint4*>(k_out) + static_cast<long long>(row) * kv8;
    uint4* v_row = reinterpret_cast<uint4*>(v_out) + static_cast<long long>(row) * kv8;
    const int step = static_cast<int>(blockDim.x * gridDim.y);
    const int first = static_cast<int>(blockIdx.y * blockDim.x + threadIdx.x);
    for (int j = first; j < q8; j += step) q_row[j] = src_row[j];
    for (int j = first; j < kv8; j += step) k_row[j] = src_row[q8 + j];
    for (int j = first; j < kv8; j += step) v_row[j] = src_row[q8 + kv8 + j];
}

// `split_qkv_vec8` from heads narrower than the cut's: q and k heads `qk`
// columns wide in the packed row, v heads `vw`, each landing `d` wide and
// zero past its width (all whole 16-byte runs).
__global__ void split_qkv_padded_vec8(
    const bf16* __restrict__ src,
    bf16* __restrict__ q_out,
    bf16* __restrict__ k_out,
    bf16* __restrict__ v_out,
    i32 q_heads, i32 kv_heads, i32 qk, i32 vw, i32 d,
    const u32* __restrict__ win)
{
    const int n = static_cast<int>(blockIdx.x);
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const int d8 = d / 8;
    const int stride8 = (q_heads * qk + kv_heads * (qk + vw)) / 8;
    const uint4* src_row = reinterpret_cast<const uint4*>(src) + static_cast<long long>(row) * stride8;
    const int out8 = (q_heads + 2 * kv_heads) * d8;
    const int step = static_cast<int>(blockDim.x * gridDim.y);
    for (int j = static_cast<int>(blockIdx.y * blockDim.x + threadIdx.x); j < out8; j += step) {
        int h = j / d8;
        const int c = j - h * d8;
        uint4* dst;
        int base8, width;
        if (h < q_heads) {
            dst = reinterpret_cast<uint4*>(q_out) + static_cast<long long>(row) * q_heads * d8;
            base8 = h * qk / 8;
            width = qk;
        } else if (h < q_heads + kv_heads) {
            h -= q_heads;
            dst = reinterpret_cast<uint4*>(k_out) + static_cast<long long>(row) * kv_heads * d8;
            base8 = (q_heads * qk + h * qk) / 8;
            width = qk;
        } else {
            h -= q_heads + kv_heads;
            dst = reinterpret_cast<uint4*>(v_out) + static_cast<long long>(row) * kv_heads * d8;
            base8 = (q_heads * qk + kv_heads * qk + h * vw) / 8;
            width = vw;
        }
        dst[h * d8 + c] = c * 8 < width ? src_row[base8 + c] : make_uint4(0, 0, 0, 0);
    }
}

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


template <class U>
__global__ void pack_rows(
    const U* __restrict__ x,
    const i32* __restrict__ perm,
    U* __restrict__ o,
    int units,
    const u32* __restrict__ win)
{
    const int n = static_cast<int>(blockIdx.x);
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const U* src = x + static_cast<long long>(perm[row]) * units;
    U* dst = o + static_cast<long long>(row) * units;
    for (int i = static_cast<int>(threadIdx.x); i < units;
         i += static_cast<int>(blockDim.x)) {
        dst[i] = src[i];
    }
}

template <class U>
__global__ void unpack_rows(
    const U* __restrict__ x,
    const i32* __restrict__ perm,
    U* __restrict__ o,
    int units,
    const u32* __restrict__ win)
{
    const int n = static_cast<int>(blockIdx.x);
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const U* src = x + static_cast<long long>(row) * units;
    U* dst = o + static_cast<long long>(perm[row]) * units;
    for (int i = static_cast<int>(threadIdx.x); i < units;
         i += static_cast<int>(blockDim.x)) {
        dst[i] = src[i];
    }
}

// One input of a column concat: its rows' `width` bytes into `out`'s rows
// at byte `at` of each (`stride` bytes a row), 16 bytes a thread where the
// widths allow, else a byte.
__global__ void concat_cols(const unsigned char* __restrict__ x, unsigned char* __restrict__ out,
                            int rows, int width, int stride, int at, int wide) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (wide) {
        const int per = width / 16;
        if (i >= static_cast<long long>(rows) * per) return;
        const int r = static_cast<int>(i / per), c = static_cast<int>(i % per) * 16;
        *reinterpret_cast<uint4*>(out + static_cast<long long>(r) * stride + at + c) =
            *reinterpret_cast<const uint4*>(x + static_cast<long long>(r) * width + c);
    } else {
        if (i >= static_cast<long long>(rows) * width) return;
        const int r = static_cast<int>(i / width), c = static_cast<int>(i % width);
        out[static_cast<long long>(r) * stride + at + c] = x[static_cast<long long>(r) * width + c];
    }
}

template <class U>
__global__ void scatter_live_rows(
    const U* __restrict__ tight,
    U* __restrict__ wide,
    const i32* __restrict__ index,
    int units,
    const u32* __restrict__ win)
{
    const int n = static_cast<int>(blockIdx.x);

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const i32 at = index[row];
    if (at < 0) return;

    const U* src = tight + static_cast<long long>(row) * units;
    U* dst = wide + static_cast<long long>(at) * units;
    for (int i = static_cast<int>(threadIdx.x); i < units;
         i += static_cast<int>(blockDim.x)) {
        dst[i] = src[i];
    }
}

// tight[row] = wide[index[row]] for the live rows whose index is not
// negative: `scatter_live_rows` read the other way. One block per row.
template <class U>
__global__ void gather_live_rows(
    const U* __restrict__ wide,
    U* __restrict__ tight,
    const i32* __restrict__ index,
    int units,
    const u32* __restrict__ win)
{
    const int n = static_cast<int>(blockIdx.x);

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const i32 at = index[row];
    if (at < 0) return;

    const U* src = wide + static_cast<long long>(at) * units;
    U* dst = tight + static_cast<long long>(row) * units;
    for (int i = static_cast<int>(threadIdx.x); i < units;
         i += static_cast<int>(blockDim.x)) {
        dst[i] = src[i];
    }
}

// Token `g` of a paged plane (its page slot times the page's tokens, plus
// its token within the page) stands `slot * per_slot + plane_at + in_page *
// units` units past the planes' base.
__device__ __forceinline__ long long page_unit(
    i32 g, int page_tokens, long long per_slot, long long plane_at, int units)
{
    const long long slot = g / page_tokens;
    const long long in_page = g % page_tokens;
    return slot * per_slot + plane_at + in_page * units;
}

// pages[tokens[row]] = rows[row] for the live rows whose token is not
// negative: rows into the paged plane's tokens they land at. One block per
// row.
template <class U>
__global__ void page_scatter_rows(
    const U* __restrict__ rows,
    U* __restrict__ pages,
    const i32* __restrict__ tokens,
    int units,
    long long per_slot,
    long long plane_at,
    int page_tokens,
    const u32* __restrict__ win)
{
    const int n = static_cast<int>(blockIdx.x);

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const i32 g = tokens[row];
    if (g < 0) return;

    const U* src = rows + static_cast<long long>(row) * units;
    U* dst = pages + page_unit(g, page_tokens, per_slot, plane_at, units);
    for (int i = static_cast<int>(threadIdx.x); i < units;
         i += static_cast<int>(blockDim.x)) {
        dst[i] = src[i];
    }
}

// rows[e] = pages[tokens[e]] for every row whose token is not negative: a
// paged plane's tokens out into rows. One block per row.
template <class U>
__global__ void page_gather_rows(
    const U* __restrict__ pages,
    U* __restrict__ rows,
    const i32* __restrict__ tokens,
    int units,
    long long per_slot,
    long long plane_at,
    int page_tokens)
{
    const int e = static_cast<int>(blockIdx.x);

    const i32 g = tokens[e];
    if (g < 0) return;

    const U* src = pages + page_unit(g, page_tokens, per_slot, plane_at, units);
    U* dst = rows + static_cast<long long>(e) * units;
    for (int i = static_cast<int>(threadIdx.x); i < units;
         i += static_cast<int>(blockDim.x)) {
        dst[i] = src[i];
    }
}

template <class T>
__global__ void pool_rows(
    const T* __restrict__ x,
    T* __restrict__ y,
    int width,
    int block)
{
    const int out = blockIdx.x;
    const long long base = static_cast<long long>(out) * block * width;

    for (int i = threadIdx.x; i < width; i += blockDim.x) {
        float acc = 0.f;
        for (int r = 0; r < block; ++r) {
            acc += Elem<T>::to_f32(x[base + static_cast<long long>(r) * width + i]);
        }
        y[static_cast<long long>(out) * width + i] =
            Elem<T>::from_f32(acc / static_cast<float>(block));
    }
}


template <class T>
__global__ void merge_rows(
    const T* __restrict__ x,
    T* __restrict__ y,
    int width,
    int block)
{
    const int out = blockIdx.x;
    const long long units = static_cast<long long>(block) * width;
    const T* src = x + static_cast<long long>(out) * units;
    T* dst = y + static_cast<long long>(out) * units;

    for (long long i = threadIdx.x; i < units; i += blockDim.x) {
        dst[i] = src[i];
    }
}

}
