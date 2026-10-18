#pragma once

// Routes that need no top-k: a hash table that names each token id's
// experts, and the identity fan-out of a grouped projection. One thread per
// token row (per route slot for the grouping).

#include "prelude/device.hip"

namespace pie::linear {

__device__ __forceinline__ float hash_sqrt_softplus(float x) {
    const float sp = x > 20.f ? x : log1pf(expf(x));
    return sqrtf(fmaxf(sp, 0.f));
}

// routes[n] = tid2eid[ids[n]]; weights[n] = sqrt(softplus(logit)) of each
// named expert, renormalized to `scaling` when asked. An id past the
// vocabulary reads row 0; an expert past the logits weighs nothing.
__global__ void hash_route_gather(const i32* __restrict__ token_ids, const i64* __restrict__ tid2eid,
                                  const bf16* __restrict__ logits, i32* __restrict__ routes,
                                  float* __restrict__ weights, int tokens, int vocab, int n_experts,
                                  int top_k, int renormalize, float scaling, const u32* __restrict__ win) {
    const int n = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    const Window window{win};
    if (n >= tokens || !window.live(n)) return;
    const long long row = window.plane(n);

    const i32 raw = token_ids[row];
    const int tid = raw >= 0 && raw < vocab ? raw : 0;
    const i64* picks = tid2eid + static_cast<long long>(tid) * top_k;
    const bf16* score = logits + row * n_experts;
    i32* ids = routes + row * top_k;
    float* ws = weights + row * top_k;
    float sum = 0.f;
    for (int r = 0; r < top_k; ++r) {
        const i64 e = picks[r];
        const float w = e >= 0 && e < n_experts ? hash_sqrt_softplus(to_f32(score[e])) : 0.f;
        ids[r] = static_cast<i32>(e);
        ws[r] = w;
        sum += w;
    }
    const float scale = renormalize != 0 && sum > 0.f ? scaling / sum : scaling;
    for (int r = 0; r < top_k; ++r) ws[r] *= scale;
}

// routes[n][slot] = slot: every token visits every group once.
__global__ void group_routes(i32* __restrict__ routes, int tokens, int groups, const u32* __restrict__ win) {
    const long long idx = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<long long>(tokens) * groups) return;
    const long long n = idx / groups;
    const int slot = static_cast<int>(idx - n * groups);
    const Window window{win};
    if (!window.live(n)) return;
    routes[window.plane(n) * groups + slot] = slot;
}

}  // namespace pie::linear
