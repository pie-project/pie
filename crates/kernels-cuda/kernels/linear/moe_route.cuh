#pragma once

#include "prelude/device.cuh"

namespace pie::linear {

/// **THE LOOKUP ROUTER'S GATHER**: `linear.moe_hash_route`, the CUDA twin of
/// `kernels-metal`'s `linear/moe_route.metal::hash_route_gather`.
///
/// DeepSeek-V4-Flash's first `num_hash_layers` layers do not SCORE a gate.
/// `tid2eid [vocab, top_k]` (I64) names, for every token id, the `top_k`
/// experts that id routes to; this reads the row the token id selects and
/// lays down UNIFORM weights `1/top_k`. What it lands is the same pair the
/// four ranked routers in `moe.cuh` land — `i32` ids and `f32` weights, row
/// major at `top_k` per token row — so `moe_matmul_select` and
/// `moe_weighted_sum` behind it cannot tell a lookup from a gate.
///
/// **THIS FILE IS NOT `moe.cuh` AND THAT IS THE METAL PRECEDENT, NOT A
/// CONFLICT MAP.** The router that reads no logits shares no scratch, no
/// `kRouterMaxExperts` staging and no shuffle reduction with the four that
/// do; on the metal plane the same split is `moe_route.metal` beside
/// `moe.metal`, and the launcher for both still sits with its siblings.
/// (`moe.cuh` once carried a `hash_route_lookup` that predated this: it read
/// router logits it was not owed and weighted the slots by sqrt-softplus — a
/// DIFFERENT answer from the reference — and nothing ever launched it. It was
/// deleted when this file landed; the name survives only in this sentence.)
///
/// **ONE THREAD PER (TOKEN ROW, SLOT), FLATTENED** — `embed`'s grid, because
/// the work per lane is one 64-bit load and two 32-bit stores and a block per
/// row of six would be a warp spent on a quarter of a load. The row is
/// `idx / top_k` and the slot is what is left, which is what lets the seat's
/// guard below read a TOKEN row off a grid that counts pairs.
///
/// **THE TABLE IS I64 AND THE ROUTES ARE I32**, and the narrowing is this
/// gather's to make. `tid2eid` is a lookup, not a weight representation the
/// trace can intern, and everything downstream — the sorted select, the
/// weighted fold — already reads an expert id as `int`. An expert count never
/// approaches 2^31. The id is read at 64 bits where the table spells it and
/// written at 32 where the path consumes it, in the one place the two planes
/// meet, and no clamp against the expert count happens here: the reference
/// does not clamp, and the consumers already refuse a route outside `[0, E)`
/// by name.
///
/// **AN OUT-OF-RANGE TOKEN ID FALLS TO ROW 0**, exactly as `layout.embed`'s
/// gather does, so a shell hands this and the embedding the SAME id stream
/// and both read the table rather than off the end of it. The reference reads
/// the id as `uint` and this reads it as `i32`; for the ids a fire actually
/// carries they are the same bits, and a negative one fails `raw >= 0` here
/// where it fails `raw < vocab` there — one row, either way. A table row that
/// names the same expert twice is copied as it stands: the hash may repeat,
/// and a uniform fold weights every slot alike.
__device__ __forceinline__ float hash_sqrt_softplus(float x) {
    const float sp = x > 20.f ? x : log1pf(expf(x));
    return sqrtf(fmaxf(sp, 0.f));
}

// One thread per TOKEN ROW: the row's `top_k` weights normalize together,
// exactly as `moe_topk_sqrt_softplus`'s normalization does over its picks.
// The weights are the GATE'S — `sqrt(softplus(logits))` at the table's
// experts, renormalized and scaled (the official `Gate.forward` scores every
// layer and a hash layer only replaces the CHOICE) — not uniform, which is
// what this kernel laid down before the reference was run against it.
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
    // The staged-geometry seat: the guard counts token rows, and `win[1]` is
    // where the live rows start. `tid2eid` is the VOCAB bank and never moves.
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

/// **THE STATIC ROUTES OF A GROUPED PROJECTION** (`linear.group_routes`):
/// slot `g` of every token row names group `g`. One thread per (row, slot).
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
