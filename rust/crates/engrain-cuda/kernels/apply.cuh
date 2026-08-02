// Apply a mask to logits, in place.
//
// The one piece the surface was missing. A caller was handed packed words and
// left to unpack them: `((mask.unsqueeze(-1) >> bits) & 1)` materialises a
// bool tensor the width of the vocabulary times the batch - 78 MB at batch 512
// on a 151,669-token vocabulary - to decide a store the mask already decides.
// XGrammar ships this kernel; not shipping it meant every integration wrote it
// again, and the tutorial for this library had to invent a helper that did not
// exist.
//
// Bandwidth-bound on the logits, so the mask read is free: one word covers 32
// tokens and is loaded once per warp-width of them.

#pragma once

#include <cstdint>
#include <cuda_fp16.h>

namespace en {

template <typename T>
__device__ __forceinline__ void apply_row(const int32_t* words, T* row,
                                          int32_t vocabulary, T floor) {
    for (int32_t token = blockIdx.y * blockDim.x + threadIdx.x; token < vocabulary;
         token += gridDim.y * blockDim.x) {
        if (((words[token >> 5] >> (token & 31)) & 1) == 0) {
            row[token] = floor;
        }
    }
}

}  // namespace en

/// `logits[row, token] = -inf` wherever the row's mask does not hold `token`.
///
/// `stride` is the logits row stride rather than the vocabulary, because a
/// model's output is padded and the grammar's is not: vLLM's spans the padded
/// vocabulary and ours the tokenizer's, and assuming they matched once left
/// the tail of every row as whatever it held.
extern "C" __global__ void en_apply_f32(const int32_t* mask, float* logits,
                                        int32_t mask_words, int32_t vocabulary,
                                        int32_t stride) {
    en::apply_row<float>(mask + (int64_t)blockIdx.x * mask_words,
                         logits + (int64_t)blockIdx.x * stride, vocabulary,
                         -INFINITY);
}

extern "C" __global__ void en_apply_f16(const int32_t* mask, __half* logits,
                                        int32_t mask_words, int32_t vocabulary,
                                        int32_t stride) {
    // Negative infinity, as float32 uses, rather than fp16's most negative
    // finite value: a caller that checks `isfinite` to see what was forbidden
    // would otherwise get a different answer from the two dtypes.
    en::apply_row<__half>(mask + (int64_t)blockIdx.x * mask_words,
                          logits + (int64_t)blockIdx.x * stride, vocabulary,
                          __ushort_as_half((unsigned short)0xFC00));
}
