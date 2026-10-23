#pragma once

// Position encodings computed on the device: a timestep's sinusoid row, and
// the T5 relative-position bucket bias table.

#include "prim/device.hip"

namespace pie::elemwise {

// One block a row: o[row, i] = sin(scale · t[row] · f_i) and o[row, i + dim/2]
// = cos(...) (swapped under `flip_sin_cos`), f_i = max_period^(-i / (dim/2)),
// f32; an odd width's last element is zero.
__global__ void sinusoid(const float* __restrict__ t, float* __restrict__ o, int dim, float max_period,
                         int flip_sin_cos, float scale, const u32* __restrict__ win) {
    const Window window{win};
    const int n = blockIdx.x;
    if (!window.live(n)) return;
    const long long row = window.plane(n);

    const int half = dim / 2;
    const float tv = t[row];
    const float log_period = logf(max_period);
    float* orow = o + row * dim;

    for (int i = threadIdx.x; i < half; i += blockDim.x) {
        const float freq = expf(-log_period * static_cast<float>(i) / static_cast<float>(half));
        float sin_v, cos_v;
        sincosf(scale * (tv * freq), &sin_v, &cos_v);
        orow[i] = flip_sin_cos ? cos_v : sin_v;
        orow[i + half] = flip_sin_cos ? sin_v : cos_v;
    }

    if ((dim & 1) != 0 && threadIdx.x == 0) orow[dim - 1] = 0.f;
}

// T5's bucket of a relative position `d`: exact below half the (directional)
// buckets, logarithmic to `max_distance` past them, the last bucket beyond.
__device__ __forceinline__ int relative_position_bucket(int d, bool bidirectional, int num_buckets,
                                                        float log_ratio) {
    int bucket = 0;
    int n;
    if (bidirectional) {
        num_buckets /= 2;
        if (d > 0) bucket += num_buckets;
        n = d < 0 ? -d : d;
    } else {
        n = d < 0 ? -d : 0;
    }
    const int max_exact = num_buckets / 2;
    if (n < max_exact) return bucket + n;
    const float x = logf(static_cast<float>(n) / static_cast<float>(max_exact));
    const float scaled = x / log_ratio * static_cast<float>(num_buckets - max_exact);
    int large = max_exact + static_cast<int>(scaled);
    if (large > num_buckets - 1) large = num_buckets - 1;
    return bucket + large;
}

// One thread a cell: y[h, c] = embedding[bucket(c - (max_len - 1)), h], f32,
// a row of 2 · max_len - 1 offsets a head.
template <class T>
__global__ void relative_bucket_bias(const T* __restrict__ embedding, float* __restrict__ y, int heads, int span,
                                     int max_len, int stride, int num_buckets, int bidirectional,
                                     float log_ratio) {
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<long long>(heads) * span) return;
    const int h = static_cast<int>(i / span);
    const int c = static_cast<int>(i % span);
    const int d = c - (max_len - 1);
    const int b = relative_position_bucket(d, bidirectional != 0, num_buckets, log_ratio);
    y[i] = to_f32(embedding[static_cast<long long>(b) * stride + h]);
}

}  // namespace pie::elemwise
