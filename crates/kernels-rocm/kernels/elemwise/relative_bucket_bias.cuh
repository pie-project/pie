#pragma once

#include "prelude/device.cuh"

namespace pie::elemwise {

__device__ __forceinline__ int relative_position_bucket(
    int d, bool bidirectional, int num_buckets, float log_ratio)
{
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

template <class T>
__global__ void relative_bucket_bias(
    const T* __restrict__ embedding,
    float* __restrict__ y,
    int heads,
    int span,
    int max_len,
    int stride,
    int num_buckets,
    int bidirectional,
    float log_ratio)
{
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= static_cast<long long>(heads) * span) return;
    const int h = static_cast<int>(i / span);
    const int c = static_cast<int>(i % span);
    const int d = c - (max_len - 1);
    const int b = relative_position_bucket(d, bidirectional != 0, num_buckets, log_ratio);
    y[i] = Elem<T>::to_f32(embedding[static_cast<long long>(b) * stride + h]);
}

}
