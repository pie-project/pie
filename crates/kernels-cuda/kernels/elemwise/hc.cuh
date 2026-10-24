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

// `hc_sinkhorn` at four streams on one thread's registers, the whole 4x4
// combine `c` (row-major) walked in place: each row and column sum the
// shuffles' tree, ((0 + 1) + (2 + 3)), with its reciprocal taken once and
// multiplied in (the hardware's approximate reciprocal, within an ulp).
// No shuffle and no division on the dependent path, so a walk is a few
// hundred cycles where the warp's took thousands; each entry within a few
// ulps of the divided walk's.
__device__ __forceinline__ void hc_sinkhorn4_regs(float (&c)[16], float hc_eps, int iters) {
    // A sum here is at least the eps: never subnormal, so the bare
    // approximate reciprocal (one MUFU, no range fix-up) is exact enough.
    auto rcp = [](float x) {
        float r;
        asm("rcp.approx.ftz.f32 %0, %1;" : "=f"(r) : "f"(x));
        return r;
    };
    auto rows = [&](float eps, bool first) {
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            const float s = (c[4 * i] + c[4 * i + 1]) + (c[4 * i + 2] + c[4 * i + 3]);
            const float r = first ? __frcp_rn(s) : rcp(s + eps);
#pragma unroll
            for (int j = 0; j < 4; ++j) c[4 * i + j] = first ? c[4 * i + j] * r + hc_eps : c[4 * i + j] * r;
        }
    };
    auto cols = [&]() {
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float s = (c[j] + c[4 + j]) + (c[8 + j] + c[12 + j]);
            const float r = rcp(s + hc_eps);
#pragma unroll
            for (int i = 0; i < 4; ++i) c[4 * i + j] *= r;
        }
    };
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float m = fmaxf(fmaxf(c[4 * i], c[4 * i + 1]), fmaxf(c[4 * i + 2], c[4 * i + 3]));
#pragma unroll
        for (int j = 0; j < 4; ++j) c[4 * i + j] = expf(c[4 * i + j] - m);
    }
    rows(0.f, true);
    cols();
    for (int iter = 0; iter < iters - 1; ++iter) {
        rows(hc_eps, false);
        cols();
    }
}

}
