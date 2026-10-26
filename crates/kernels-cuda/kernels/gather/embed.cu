#include "prim/device.cuh"

namespace pie::gather {

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

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

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
    int hidden, int local_vocab, int vocab_offset,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int plane_row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const i32 tid_raw = token_ids[plane_row];
    const int local_tid = tid_raw - vocab_offset;
    const bool in_shard = local_tid >= 0 && local_tid < local_vocab;
    const T* row =
        weight + static_cast<long long>(in_shard ? local_tid : 0) * hidden;
    T* out = y + static_cast<long long>(plane_row) * hidden;

    for (int h = threadIdx.x; h < hidden; h += blockDim.x) {
        out[h] = in_shard ? row[h] : Elem<T>::from_f32(0.0f);
    }
}

template <class T>
__global__ void embed_concat(
    const int* __restrict__ ids,
    const T* __restrict__ table,
    T* __restrict__ y,
    int rows,
    int heads,
    int width,
    int vocab,
    int held,
    int base,
    const u32* __restrict__ win)
{
    const long long i =
        (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)rows * heads * width;
    if (i >= total) return;

    const int w = (int)(i % width);
    const long long rh = i / width;
    const int h = (int)(rh % heads);
    const long long r = rh / heads;

    if (win != nullptr && r >= static_cast<long long>(win[0])) return;

    const long long row = win != nullptr ? r + static_cast<long long>(win[1]) : r;
    const long long at = (row * heads + h) * (long long)width + w;

    // a tensor-parallel rank holds rows [base, base + held) of the table, zeros elsewhere
    const int id = ids[row * heads + h];
    const int local = id - base;
    y[at] = (id < 0 || id >= vocab || local < 0 || local >= held)
        ? Elem<T>::from_f32(0.f)
        : table[(long long)local * width + w];
}

struct alignas(16) EmbedTableBases {
    const u8* codes;
    const u8* scales;
    const u8* biases;
    const u8* pad;
};

template <class T>
__global__ void embed_concat_mlxu4(
    const int* __restrict__ ids,
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    const u8* __restrict__ biases,
    T* __restrict__ y,
    int rows,
    int heads,
    int width,
    int group,
    int vocab,
    int offset,
    const EmbedTableBases* __restrict__ bases,
    unsigned int* __restrict__ hits,
    const u32* __restrict__ win)
{
    const long long i =
        (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)rows * heads * width;
    if (i >= total) return;

    if (hits != nullptr && i == 0) atomicAdd(hits, 1u);

    const u8* codes_at = codes;
    const u8* scales_at = scales;
    const u8* biases_at = biases;
    if (bases != nullptr) {
        const EmbedTableBases seat = *bases;
        codes_at = seat.codes;
        scales_at = seat.scales;
        biases_at = seat.biases;
    }

    const int w = (int)(i % width);
    const long long rh = i / width;
    const int h = (int)(rh % heads);
    const long long r = rh / heads;

    if (win != nullptr && r >= static_cast<long long>(win[0])) return;

    const long long row = win != nullptr ? r + static_cast<long long>(win[1]) : r;
    const long long at = (row * heads + h) * (long long)width + w;

    const int id = ids[row * heads + h] - offset;
    if (id < 0 || id >= vocab) {
        y[at] = Elem<T>::from_f32(0.f);
        return;
    }

    const u8 byte = codes_at[(long long)id * (width / 2) + w / 2];
    const float code = (float)((w & 1) ? (byte >> 4) : (byte & 0xF));
    const long long fx = (long long)id * (width / group) + w / group;
    const bf16* s16 = reinterpret_cast<const bf16*>(scales_at);
    const bf16* b16 = reinterpret_cast<const bf16*>(biases_at);
    const float v = code * Elem<bf16>::to_f32(s16[fx]) + Elem<bf16>::to_f32(b16[fx]);
    y[at] = Elem<T>::from_f32(v);
}

template <class T>
__global__ void embed_concat_mlxu8(
    const int* __restrict__ ids,
    const u8* __restrict__ codes,
    const u8* __restrict__ scales,
    const u8* __restrict__ biases,
    T* __restrict__ y,
    int rows,
    int heads,
    int width,
    int group,
    int vocab,
    int offset,
    const EmbedTableBases* __restrict__ bases,
    unsigned int* __restrict__ hits,
    const u32* __restrict__ win)
{
    const long long i =
        (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)rows * heads * width;
    if (i >= total) return;

    if (hits != nullptr && i == 0) atomicAdd(hits, 1u);

    const u8* codes_at = codes;
    const u8* scales_at = scales;
    const u8* biases_at = biases;
    if (bases != nullptr) {
        const EmbedTableBases seat = *bases;
        codes_at = seat.codes;
        scales_at = seat.scales;
        biases_at = seat.biases;
    }

    const int w = (int)(i % width);
    const long long rh = i / width;
    const int h = (int)(rh % heads);
    const long long r = rh / heads;

    if (win != nullptr && r >= static_cast<long long>(win[0])) return;

    const long long row = win != nullptr ? r + static_cast<long long>(win[1]) : r;
    const long long at = (row * heads + h) * (long long)width + w;

    const int id = ids[row * heads + h] - offset;
    if (id < 0 || id >= vocab) {
        y[at] = Elem<T>::from_f32(0.f);
        return;
    }

    const float code = (float)codes_at[(long long)id * width + w];
    const long long fx = (long long)id * (width / group) + w / group;
    const bf16* s16 = reinterpret_cast<const bf16*>(scales_at);
    const bf16* b16 = reinterpret_cast<const bf16*>(biases_at);
    const float v = code * Elem<bf16>::to_f32(s16[fx]) + Elem<bf16>::to_f32(b16[fx]);
    y[at] = Elem<T>::from_f32(v);
}

// A hashed table stored e4m3 under one f32 scale for the whole of it, read
// only at the rows its ids gather: each code widened and scaled, then
// rounded to the output dtype once.
__device__ __forceinline__ float e4m3_value(u8 b) {
    const unsigned e = (b >> 3) & 0xFu;
    const unsigned m = b & 0x7u;
    const float v = e != 0u ? __int_as_float(static_cast<int>(((e + 120u) << 23) | (m << 20)))
                            : static_cast<float>(m) * 0.001953125f;
    return (b & 0x80u) != 0u ? -v : v;
}

// `ids` `[rows, heads]` rows of a `[vocab, width]` table; `y` `[rows, heads *
// width]`, each head's row side by side. An id outside the table lands zeros.
template <class T>
__global__ void embed_concat_e4m3(
    const int* __restrict__ ids,
    const u8* __restrict__ codes,
    const float* __restrict__ scale,
    T* __restrict__ y,
    int rows,
    int heads,
    int width,
    int vocab,
    const u32* __restrict__ win)
{
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)rows * heads * width;
    if (i >= total) return;

    const int w = (int)(i % width);
    const long long rh = i / width;
    const int h = (int)(rh % heads);
    const long long r = rh / heads;

    if (win != nullptr && r >= static_cast<long long>(win[0])) return;

    const long long row = win != nullptr ? r + static_cast<long long>(win[1]) : r;
    const long long at = (row * heads + h) * (long long)width + w;

    const int id = ids[row * heads + h];
    y[at] = (id < 0 || id >= vocab)
        ? Elem<T>::from_f32(0.f)
        : Elem<T>::from_f32(e4m3_value(codes[(long long)id * width + w]) * scale[0]);
}

// embed_concat_e4m3 sixteen codes a thread: `width` a multiple of 16 and the
// table 16-byte aligned, so each thread reads one 16-byte run of a row (one
// request, where the table sits in mapped host memory). Under `kMx` the
// table is MXFP8 (DeepSeek-V4.1's Engram): `scale` is its `[vocab, width /
// 32]` ue8m0 exponents, a value `code * 2^(e - 127)`; else one f32.
template <class T, bool kMx>
__global__ void embed_concat_e4m3_vec16(
    const int* __restrict__ ids,
    const u8* __restrict__ codes,
    const void* __restrict__ scale,
    T* __restrict__ y,
    int rows,
    int heads,
    int width,
    int vocab,
    const u32* __restrict__ win)
{
    const int runs = width / 16;
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    const long long total = (long long)rows * heads * runs;
    if (i >= total) return;

    const int c = (int)(i % runs);
    const long long rh = i / runs;
    const int h = (int)(rh % heads);
    const long long r = rh / heads;

    if (win != nullptr && r >= static_cast<long long>(win[0])) return;

    const long long row = win != nullptr ? r + static_cast<long long>(win[1]) : r;
    T* out = y + (row * heads + h) * (long long)width + c * 16;

    const int id = ids[row * heads + h];
    if (id < 0 || id >= vocab) {
#pragma unroll
        for (int k = 0; k < 16; ++k) out[k] = Elem<T>::from_f32(0.f);
        return;
    }
    const uint4 run = *reinterpret_cast<const uint4*>(codes + (long long)id * width + c * 16);
    const u8* b = reinterpret_cast<const u8*>(&run);
    float s;
    if constexpr (kMx) {
        const unsigned e = static_cast<const u8*>(scale)[(long long)id * (width / 32) + c / 2];
        // 2^(e - 127) as a float's bits; e 0 (2^-127) is a subnormal's.
        s = e != 0u ? __uint_as_float(e << 23) : __uint_as_float(0x00400000u);
    } else {
        s = static_cast<const float*>(scale)[0];
    }
#pragma unroll
    for (int k = 0; k < 16; ++k) out[k] = Elem<T>::from_f32(e4m3_value(b[k]) * s);
}

template <class T>
__global__ void embed_weighted(
    const i32* __restrict__ ids,
    const float* __restrict__ weights,
    const T* __restrict__ table,
    T* __restrict__ y,
    int hidden,
    int vocab,
    int taps,
    const u32* __restrict__ win)
{
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const i32* row_ids = ids + static_cast<long long>(row) * taps;
    const float* row_w = weights + static_cast<long long>(row) * taps;
    T* out = y + static_cast<long long>(row) * hidden;

    for (int i = threadIdx.x; i < hidden; i += blockDim.x) {
        float acc = 0.f;
        for (int t = 0; t < taps; ++t) {
            const i32 raw = row_ids[t];
            const int at = (raw >= 0 && raw < vocab) ? raw : 0;
            acc += row_w[t] *
                   Elem<T>::to_f32(table[static_cast<long long>(at) * hidden + i]);
        }
        out[i] = Elem<T>::from_f32(acc);
    }
}

}
