#pragma once

// Hashed n-gram embeddings: each row's `heads` ids gather one table row
// each, landing side by side ([rows, heads·width]); an id outside the table
// lands zeros. One thread per element.

#include "prelude/device.hip"
#include "linear/act_quant.hip"

namespace pie::layout {

// The element a flat thread lands: (row, head, column).
struct ConcatAt {
    long long row;
    int head, w;
};

// Where this thread lands, or false past the launch or the live window.
__device__ __forceinline__ bool concat_at(int rows, int heads, int width, const u32* win, ConcatAt& at) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<long long>(rows) * heads * width) return false;
    const Window window{win};
    at.w = static_cast<int>(i % width);
    const long long rh = i / width;
    at.head = static_cast<int>(rh % heads);
    const long long r = rh / heads;
    if (!window.live(r)) return false;
    at.row = window.plane(r);
    return true;
}

template <class T>
__global__ void embed_concat(const i32* __restrict__ ids, const T* __restrict__ table, T* __restrict__ y,
                             int rows, int heads, int width, int vocab, const u32* __restrict__ win) {
    ConcatAt at;
    if (!concat_at(rows, heads, width, win, at)) return;
    const long long slot = at.row * heads + at.head;
    const int id = ids[slot];
    y[slot * width + at.w] =
        id < 0 || id >= vocab ? from_f32<T>(0.f) : table[static_cast<long long>(id) * width + at.w];
}

// The three planes of an MLX affine table a streamed group seats: where a
// seat is named, its bases replace the launch's own.
struct alignas(16) EmbedTableBases {
    const u8* codes;
    const u8* scales;
    const u8* biases;
    const u8* pad;
};

// An MLX affine table, BITS (4 or 8) per code and one bf16 scale and bias
// per `group` columns: v = code·scale + bias. `offset` moves the ids into a
// vocab band's own rows (`vocab` of them); `hits` counts the launches that
// ran.
template <class T, int BITS>
__device__ __forceinline__ void embed_affine(const i32* __restrict__ ids, const u8* codes, const u8* scales,
                                             const u8* biases, T* __restrict__ y, int rows, int heads,
                                             int width, int group, int vocab, int offset,
                                             const EmbedTableBases* __restrict__ bases,
                                             u32* __restrict__ hits, const u32* __restrict__ win) {
    if (hits != nullptr && blockIdx.x == 0 && threadIdx.x == 0) atomicAdd(hits, 1u);
    ConcatAt at;
    if (!concat_at(rows, heads, width, win, at)) return;
    if (bases != nullptr) {
        const EmbedTableBases seat = *bases;
        codes = seat.codes;
        scales = seat.scales;
        biases = seat.biases;
    }
    const long long slot = at.row * heads + at.head;
    const int id = ids[slot] - offset;
    T* dst = y + slot * width + at.w;
    if (id < 0 || id >= vocab) {
        *dst = from_f32<T>(0.f);
        return;
    }
    float code;
    if constexpr (BITS == 4) {
        const u8 byte = codes[static_cast<long long>(id) * (width / 2) + at.w / 2];
        code = static_cast<float>((at.w & 1) ? (byte >> 4) : (byte & 0xF));
    } else {
        code = static_cast<float>(codes[static_cast<long long>(id) * width + at.w]);
    }
    const long long fx = static_cast<long long>(id) * (width / group) + at.w / group;
    const float scale = to_f32(reinterpret_cast<const bf16*>(scales)[fx]);
    const float bias = to_f32(reinterpret_cast<const bf16*>(biases)[fx]);
    *dst = from_f32<T>(code * scale + bias);
}

template <class T>
__global__ void embed_concat_mlxu4(const i32* __restrict__ ids, const u8* __restrict__ codes,
                                   const u8* __restrict__ scales, const u8* __restrict__ biases,
                                   T* __restrict__ y, int rows, int heads, int width, int group, int vocab,
                                   int offset, const EmbedTableBases* __restrict__ bases,
                                   u32* __restrict__ hits, const u32* __restrict__ win) {
    embed_affine<T, 4>(ids, codes, scales, biases, y, rows, heads, width, group, vocab, offset, bases, hits,
                       win);
}

template <class T>
__global__ void embed_concat_mlxu8(const i32* __restrict__ ids, const u8* __restrict__ codes,
                                   const u8* __restrict__ scales, const u8* __restrict__ biases,
                                   T* __restrict__ y, int rows, int heads, int width, int group, int vocab,
                                   int offset, const EmbedTableBases* __restrict__ bases,
                                   u32* __restrict__ hits, const u32* __restrict__ win) {
    embed_affine<T, 8>(ids, codes, scales, biases, y, rows, heads, width, group, vocab, offset, bases, hits,
                       win);
}

// An MXFP8 table: e4m3 codes under one ue8m0 exponent a 1x32 block, v =
// code · 2^(e - 127), exact in bf16.
template <class T>
__global__ void embed_concat_mxfp8(const i32* __restrict__ ids, const u8* __restrict__ codes,
                                   const u8* __restrict__ scales, T* __restrict__ y, int rows, int heads, int width,
                                   int vocab, const u32* __restrict__ win) {
    ConcatAt at;
    if (!concat_at(rows, heads, width, win, at)) return;
    const long long slot = at.row * heads + at.head;
    const int id = ids[slot];
    T* dst = y + slot * width + at.w;
    if (id < 0 || id >= vocab) {
        *dst = from_f32<T>(0.f);
        return;
    }
    const long long r = static_cast<long long>(id);
    const float v = pie::linear::e4m3fn_value(codes[r * width + at.w]) *
                    pie::linear::e8m0_value(scales[r * (width / 32) + at.w / 32]);
    *dst = from_f32<T>(v);
}

}  // namespace pie::layout
