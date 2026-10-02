#pragma once

#include "prelude/device.cuh"

namespace pie::layout {

template <class T>
__global__ void embed_concat(
    const int* __restrict__ ids,
    const T* __restrict__ table,
    T* __restrict__ y,
    int rows,
    int heads,
    int width,
    int vocab,
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

    const int id = ids[row * heads + h];
    y[at] = (id < 0 || id >= vocab)
        ? Elem<T>::from_f32(0.f)
        : table[(long long)id * width + w];
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

}
