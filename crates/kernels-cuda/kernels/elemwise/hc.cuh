#pragma once

#include "prim/device.cuh"

namespace pie::elemwise {

constexpr int MAX_HC_MULT = 8;

// The combine's Sinkhorn walk in one warp, lane `i * M + j` holding entry
// (i, j) from its logit `c`: the row softmax plus the eps, then column,
// then (row, column) `iters - 1` times, every sum by shuffles.
__device__ __forceinline__ float hc_sinkhorn(float c, int tid, int M, float hc_eps, int iters) {
    if (M == 4) {
        // Four streams: a row is four adjacent lanes and a column lanes four
        // apart, each sum two butterfly steps.
        auto rows = [](float v) {
            v += __shfl_xor_sync(0xffffffffu, v, 1);
            return v + __shfl_xor_sync(0xffffffffu, v, 2);
        };
        auto cols = [](float v) {
            v += __shfl_xor_sync(0xffffffffu, v, 4);
            return v + __shfl_xor_sync(0xffffffffu, v, 8);
        };
        float max_v = fmaxf(c, __shfl_xor_sync(0xffffffffu, c, 1));
        max_v = fmaxf(max_v, __shfl_xor_sync(0xffffffffu, max_v, 2));
        c = tid < 16 ? expf(c - max_v) : 1.f;
        c = c / rows(c) + hc_eps;
        c = c / (cols(c) + hc_eps);
        for (int iter = 0; iter < iters - 1; ++iter) {
            c = c / (rows(c) + hc_eps);
            c = c / (cols(c) + hc_eps);
        }
        return c;
    }
    const int MM = M * M, i = tid / M, j = tid % M;
    auto at = [&](int lane) { return __shfl_sync(0xffffffffu, c, lane < MM ? lane : 0); };
    float max_v = -flt_max();
    for (int k = 0; k < M; ++k) max_v = fmaxf(max_v, at(i * M + k));
    c = expf(c - max_v);
    float sum = 0.f;
    for (int k = 0; k < M; ++k) sum += at(i * M + k);
    c = c / sum + hc_eps;
    auto col = [&]() {
        float t = 0.f;
        for (int k = 0; k < M; ++k) t += at(k * M + j);
        c = c / (t + hc_eps);
    };
    auto rowf = [&]() {
        float t = 0.f;
        for (int k = 0; k < M; ++k) t += at(i * M + k);
        c = c / (t + hc_eps);
    };
    col();
    for (int iter = 0; iter < iters - 1; ++iter) {
        rowf();
        col();
    }
    return c;
}

}
