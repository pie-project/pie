#pragma once

#include "prelude/device.cuh"

namespace pie::layout {

/// **THE GATHER THAT INTERPOLATES** (`.wiki/alto/multimodal.md` §9.2).
///
/// `y[r] = sum over t of weights[r, t] * table[ids[r, t]]`, one block per
/// output row, threads striding the width.
///
/// The vision towers store one learned position grid at `num_grid_per_side^2`
/// and resample it to each image's own grid; upstream writes that as
/// `(pos_embed(interp_indices) * interp_weights[:, :, None]).sum(1)` over
/// `[patches, taps]` indices and weights, which is this expression with the
/// sum moved inside. Four taps for bilinear, sixteen for bicubic — read off
/// the operand rather than stated, because the operand carries it.
///
/// **THE ACCUMULATION IS f32 AND THE WEIGHTS ARRIVE f32.** Upstream multiplies
/// a float weight into a float-promoted embedding and sums; a bf16 running sum
/// over four taps would round four times, and a bf16 WEIGHT would move the
/// resample by more than the gather it feeds. Only the write is in the model
/// element.
///
/// **OUT-OF-RANGE IDS CLAMP TO ROW ZERO**, which is `embed`'s own rule one
/// file over: a gather with an index it cannot honour reads a defined row
/// rather than an address, and the vector was checked host-side before the
/// launch.
template <class T>
__global__ void embed_weighted(
    const i32* __restrict__ ids,
    const float* __restrict__ weights,
    const T* __restrict__ table,
    T* __restrict__ y,
    int hidden,
    int vocab,
    int taps)
{
    const int n = blockIdx.x;
    const i32* row_ids = ids + static_cast<long long>(n) * taps;
    const float* row_w = weights + static_cast<long long>(n) * taps;
    T* out = y + static_cast<long long>(n) * hidden;

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
