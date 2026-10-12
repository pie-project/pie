#pragma once

#include "prelude/device.cuh"

namespace pie::linear {

__device__ __forceinline__ float hash_sqrt_softplus(float x) {
    const float sp = x > 20.f ? x : log1pf(expf(x));
    return sqrtf(fmaxf(sp, 0.f));
}

__global__ void hash_route_gather(
    const i32* __restrict__ token_ids,
    const i64* __restrict__ tid2eid,
    const bf16* __restrict__ logits,
    i32* __restrict__ routes,
    float* __restrict__ weights,
    int tokens,
    int vocab,
    int n_experts,
    int top_k,
    int renormalize,
    float scaling,
    const u32* __restrict__ win)
{
    const int n = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                + static_cast<int>(threadIdx.x);
    if (n >= tokens) return;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int plane_row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    const i32 raw = token_ids[plane_row];
    const int tid = (raw >= 0 && raw < vocab) ? raw : 0;
    const i64* picks = tid2eid + static_cast<long long>(tid) * top_k;
    const bf16* score = logits + static_cast<long long>(plane_row) * n_experts;
    i32* ids = routes + static_cast<long long>(plane_row) * top_k;
    float* ws = weights + static_cast<long long>(plane_row) * top_k;
    float sum = 0.f;
    for (int r = 0; r < top_k; ++r) {
        const i64 e = picks[r];
        const float w = (e >= 0 && e < static_cast<i64>(n_experts))
            ? hash_sqrt_softplus(bf16_to_f32(score[static_cast<int>(e)])) : 0.f;
        ids[r] = static_cast<i32>(e);
        ws[r] = w;
        sum += w;
    }
    const float scale = (renormalize != 0 && sum > 0.f) ? scaling / sum : scaling;
    for (int r = 0; r < top_k; ++r) ws[r] *= scale;
}

__global__ void group_routes(
    i32* __restrict__ routes,
    int tokens,
    int groups,
    const u32* __restrict__ win)
{
    const int idx = static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x)
                  + static_cast<int>(threadIdx.x);
    if (idx >= tokens * groups) return;
    const int n = idx / groups;
    const int slot = idx - n * groups;
    if (win != nullptr && n >= static_cast<int>(win[0])) return;
    const int plane_row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    routes[static_cast<long long>(plane_row) * groups + slot] = slot;
}

}
