// Turning a mask back into the set it stands for.
//
// A vocabulary-wide bitmask is what every constrained decoder hands back, and
// it is the wrong shape for the thing that reads it. At a JSON number position
// a grammar allows a few hundred of a hundred and fifty thousand tokens, and
// the sampler that follows sorts, scans and normalises all of them anyway.
// Sampling over the allowed set instead is cheaper than sampling with *no*
// constraint at all - which nobody exploits, because nobody has the set on the
// device to begin with.
//
// So this is not an optimisation of the mask. It is the reason for keeping the
// parse on the GPU stated in one kernel: the set never has to become a mask,
// and the mask never has to leave.

#pragma once

#include <stdint.h>

namespace gg {

/// An exclusive prefix sum of one value per thread, and the block's total.
///
/// Warp shuffles and one serial pass over at most eight warp totals, rather
/// than a shared-memory tree: the tree costs `log2(blockDim)` barriers per
/// tile and this kernel runs a tile every thirty-two words.
__device__ __forceinline__ int32_t block_scan(int32_t value, int32_t* room,
                                              int32_t* total) {
    int32_t lane = threadIdx.x & 31;
    int32_t warp = threadIdx.x >> 5;
    int32_t warps = blockDim.x >> 5;

    int32_t running = value;
    for (int32_t step = 1; step < 32; step <<= 1) {
        int32_t other = __shfl_up_sync(0xffffffffu, running, step);
        if (lane >= step) {
            running += other;
        }
    }
    if (lane == 31) {
        room[warp] = running;
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        int32_t sum = 0;
        for (int32_t at = 0; at < warps; ++at) {
            int32_t held = room[at];
            room[at] = sum;
            sum += held;
        }
        *total = sum;
    }
    __syncthreads();
    return room[warp] + running - value;
}

}  // namespace gg

/// Every token a sequence admits, as sorted ids rather than as a bitmask.
///
/// `counts` is written whatever `capacity` is, so a caller always learns the
/// true size of the set even when the buffer could not hold it. A sequence
/// whose count came out larger than the capacity has a truncated list and the
/// caller has to fall back to the mask for it - which is the honest failure,
/// and the one the dense regime needs: a JSON string body admits a hundred and
/// forty-seven thousand tokens, and gathering those is not what this is for.
///
/// Sorted, and deterministically so, because a sampler seeded the same way
/// must draw the same token twice. That is what the prefix sum buys; an
/// atomic counter per sequence would have been shorter and would have made the
/// order depend on which warp arrived first.
extern "C" __global__ void gg_compact(
    const int32_t* mask,
    int32_t* allowed,
    int32_t* counts,
    int32_t mask_words,
    int32_t capacity,
    int32_t vocabulary) {
    int32_t sequence = blockIdx.x;
    const int32_t* row = mask + (int64_t)sequence * mask_words;

    __shared__ int32_t room[32];
    __shared__ int32_t total;
    int32_t written = 0;

    for (int32_t tile = 0; tile < mask_words; tile += blockDim.x) {
        int32_t at = tile + threadIdx.x;
        int32_t word = 0;
        if (at < mask_words) {
            word = row[at];
            // The last word of a row runs past the vocabulary; the bits above
            // it are nobody's token and must not be handed to a sampler.
            int32_t over = vocabulary - at * 32;
            if (over < 32) {
                word &= over <= 0 ? 0 : (int32_t)((1u << over) - 1u);
            }
        }
        int32_t mine = __popc(word);
        __syncthreads();
        int32_t start = written + gg::block_scan(mine, room, &total);
        while (word != 0) {
            int32_t bit = __ffs(word) - 1;
            if (start < capacity) {
                allowed[(int64_t)sequence * capacity + start] = at * 32 + bit;
            }
            ++start;
            word &= word - 1;
        }
        written += total;
    }

    if (threadIdx.x == 0) {
        counts[sequence] = written;
    }
}
