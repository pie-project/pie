#pragma once

// out[n] = act · weight[n] (+ beta·out[n]) (+ bias[n]): the one-row
// projection a decode step runs. Weight rows are K bf16, K a multiple of 8,
// loaded 16 bytes per lane; f32 accumulation.

#include "prelude/device.hip"

namespace pie::linear {

constexpr int kGemvUnroll = 4;

// The dot product of `vectors` 8-element packs starting at lane `first`,
// stepping `stride` packs: this lane's share of one row.
__device__ __forceinline__ float dot_packs(const Pack8<bf16>* __restrict__ w,
                                           const Pack8<bf16>* __restrict__ x, int vectors,
                                           int first, int stride) {
    float acc = 0.f;
    int i = first;
    for (; i + stride * (kGemvUnroll - 1) < vectors; i += stride * kGemvUnroll) {
        Pack8<bf16> wv[kGemvUnroll], xv[kGemvUnroll];
#pragma unroll
        for (int u = 0; u < kGemvUnroll; ++u) {
            wv[u] = w[i + stride * u];
            xv[u] = x[i + stride * u];
        }
#pragma unroll
        for (int u = 0; u < kGemvUnroll; ++u)
#pragma unroll
            for (int j = 0; j < 8; ++j) acc += to_f32(wv[u].v[j]) * to_f32(xv[u].v[j]);
    }
    for (; i < vectors; i += stride) {
        const Pack8<bf16> wv = w[i], xv = x[i];
#pragma unroll
        for (int j = 0; j < 8; ++j) acc += to_f32(wv.v[j]) * to_f32(xv.v[j]);
    }
    return acc;
}

__device__ __forceinline__ void gemv_store(bf16* out, const bf16* bias, int row, float acc, float beta) {
    if (beta != 0.f) acc += beta * to_f32(out[row]);
    float v = to_f32(from_f32<bf16>(acc));
    if (bias != nullptr) v += to_f32(bias[row]);
    out[row] = from_f32<bf16>(v);
}

// Block (kWave, WAVES). SPLIT = false: each wavefront owns a row, WAVES rows
// a block. SPLIT = true: the block owns a row and its WAVES wavefronts split
// K, for a short N that would not fill the device a wavefront per row.
template <int WAVES, bool SPLIT>
__global__ __launch_bounds__(kWave* WAVES) void gemv_bf16(const bf16* __restrict__ weight,
                                                          const bf16* __restrict__ act,
                                                          const bf16* __restrict__ bias,
                                                          bf16* __restrict__ out, int N, int K,
                                                          float beta) {
    const int wave = threadIdx.y;
    const int row = SPLIT ? blockIdx.x : blockIdx.x * WAVES + wave;
    if (row >= N) return;
    const auto* w = reinterpret_cast<const Pack8<bf16>*>(weight + static_cast<long long>(row) * K);
    const auto* x = reinterpret_cast<const Pack8<bf16>*>(act);
    const int vectors = K / 8;
    if constexpr (SPLIT) {
        __shared__ float partial[WAVES];
        const float acc = wave_sum(dot_packs(w, x, vectors, wave * kWave + lane(), WAVES * kWave));
        if (lane() == 0) partial[wave] = acc;
        __syncthreads();
        if (wave == 0 && lane() == 0) {
            float total = 0.f;
#pragma unroll
            for (int i = 0; i < WAVES; ++i) total += partial[i];
            gemv_store(out, bias, row, total, beta);
        }
    } else {
        const float acc = wave_sum(dot_packs(w, x, vectors, lane(), kWave));
        if (lane() == 0) gemv_store(out, bias, row, acc, beta);
    }
}

}  // namespace pie::linear
