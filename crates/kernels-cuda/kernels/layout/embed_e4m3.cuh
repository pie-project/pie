#pragma once

// A hashed table stored e4m3 under one f32 scale for the whole of it, read
// only at the rows its ids gather: each code widened and scaled, then
// rounded to the output dtype once.

#include "prelude/device.cuh"

namespace pie::layout {

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
// request, where the table sits in mapped host memory) and widens each code
// as the element walk does.
template <class T>
__global__ void embed_concat_e4m3_vec16(
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
    const float s = scale[0];
#pragma unroll
    for (int k = 0; k < 16; ++k) out[k] = Elem<T>::from_f32(e4m3_value(b[k]) * s);
}

// An MXFP8 table (DeepSeek-V4.1's Engram): `codes` `[vocab, width]` e4m3
// under `exps` `[vocab, width / 32]` ue8m0, one a 32 of each row. Sixteen
// codes a thread, the run's exponent read once; each value `code ·
// 2^(e - 127)`, exact before its one rounding to the output dtype.
template <class T>
__global__ void embed_concat_mxfp8_vec16(
    const int* __restrict__ ids,
    const u8* __restrict__ codes,
    const u8* __restrict__ exps,
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
    const unsigned e = exps[(long long)id * (width / 32) + c / 2];
    // 2^(e - 127) as a float's bits; e 0 (2^-127) is a subnormal's.
    const float s = e != 0u ? __uint_as_float(e << 23) : __uint_as_float(0x00400000u);
#pragma unroll
    for (int k = 0; k < 16; ++k) out[k] = Elem<T>::from_f32(e4m3_value(b[k]) * s);
}

}  // namespace pie::layout
