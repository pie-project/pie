#include "prim/bf16.slang"
#include "prim/params.slang"
#if !defined(PIE_WGSL)
#include "prim/slot16.slang"
#endif

// Routes sorted into expert tiles for the routed tiled matmuls, and the rows
// moved in and out. `PIE_ROUTE_SORT`: one workgroup counts each expert's
// routes, lays the experts out in whole `tile_rows` tiles (`tile_expert` a
// tile, -1 past the last), and lands each route's sorted place (`perm` a
// place: the route, -1 for padding; `inv` a route: its place).
// `PIE_ROUTE_GATHER` copies each place's activation row in (zero for
// padding), `PIE_ROUTE_SCATTER` each route's result row out. SPIR-V moves a
// value a 16-bit slot an invocation; WGSL moves a word, so its host refuses
// an odd width.

#define ROUTER_MAX_EXPERTS 1024u
// Invocations the sort runs: 1024, or 256 where the device allows fewer.
#if !defined(PIE_SORT_LANES)
#define PIE_SORT_LANES 1024u
#endif

struct Params {
    uint n;
    uint n_experts;
    uint experts_per_token;
    uint tile_rows;
    uint padded;
    uint width;
    uint x_pitch;
};

#if defined(PIE_ROUTE_SORT)

[[vk::binding(0)]] StructuredBuffer<int> expert_ids;
[[vk::binding(1)]] RWStructuredBuffer<int> perm;
[[vk::binding(2)]] RWStructuredBuffer<int> row_expert;
[[vk::binding(3)]] RWStructuredBuffer<int> tile_expert;
[[vk::binding(4)]] RWStructuredBuffer<int> inv;
PIE_PARAMS(5, Params, params);

groupshared Atomic<uint> counts[ROUTER_MAX_EXPERTS];
groupshared uint base[ROUTER_MAX_EXPERTS];

[numthreads(PIE_SORT_LANES, 1, 1)]
void main(uint3 local : SV_GroupThreadID) {
    uint lid = local.x;
    uint nthreads = PIE_SORT_LANES;
    uint E = min(params.n_experts, ROUTER_MAX_EXPERTS);
    uint tile = params.tile_rows < 1u ? 1u : params.tile_rows;
    uint tiles = params.padded / tile;

    for (uint e = lid; e < E; e += nthreads) counts[e].store(0u);
    for (uint i = lid; i < params.padded; i += nthreads) { perm[i] = -1; row_expert[i] = 0; }
    for (uint t = lid; t < tiles; t += nthreads) tile_expert[t] = -1;
    for (uint i = lid; i < params.n; i += nthreads) inv[i] = -1;
    GroupMemoryBarrierWithGroupSync();

    for (uint i = lid; i < params.n; i += nthreads) {
        int e = expert_ids[i];
        if (e >= 0 && uint(e) < E) {
            counts[uint(e)].add(1u);
        }
    }
    GroupMemoryBarrierWithGroupSync();

    if (lid == 0u) {
        uint at = 0u;
        for (uint e = 0u; e < E; e++) {
            uint c = counts[e].load();
            uint span = c > 0u ? ((c + tile - 1u) / tile) * tile : 0u;
            base[e] = at;
            for (uint t = at / tile; t < (at + span) / tile && t < tiles; t++) tile_expert[t] = int(e);
            counts[e].store(0u);
            at += span;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint i = lid; i < params.n; i += nthreads) {
        int e = expert_ids[i];
        if (e < 0 || uint(e) >= E) continue;
        uint previous = counts[uint(e)].add(1u);
        uint at = base[uint(e)] + previous;
        if (at < params.padded) {
            perm[at] = int(i);
            row_expert[at] = e;
            inv[i] = int(at);
        }
    }
}

#elif defined(PIE_ROUTE_GATHER)

#if defined(PIE_WGSL)
[[vk::binding(0)]] StructuredBuffer<uint> x;
[[vk::binding(1)]] RWStructuredBuffer<uint> out_;
#else
PIE_SLOTS_RO(0, x);
PIE_SLOTS_RW(1, out_);
#endif
[[vk::binding(2)]] StructuredBuffer<int> perm;
PIE_PARAMS(3, Params, params);

#if defined(PIE_WGSL)
// A word of the place's row an invocation.
[numthreads(256, 1, 1)]
void main(uint3 thread : SV_DispatchThreadID) {
    uint word = thread.x;
    uint half_width = params.width >> 1;
    if (half_width == 0u) return;
    uint r = word / half_width;
    if (r >= params.padded) return;
    uint c = (word - r * half_width) << 1;
    int sel = perm[r];
    if (sel < 0) {
        out_[word] = 0u;
        return;
    }
    uint k = max(params.experts_per_token, 1u);
    uint pitch = params.x_pitch != 0u ? params.x_pitch : params.width;
    uint src = (uint(sel) / k) * pitch + c;
    out_[word] = f32_to_bf16(bf16_at(x, src)) | (f32_to_bf16(bf16_at(x, src + 1u)) << 16);
}
#else
[numthreads(256, 1, 1)]
void main(uint3 thread : SV_DispatchThreadID) {
    uint c = thread.x;
    uint r = thread.y;
    if (c >= params.width || r >= params.padded) return;
    int sel = perm[r];
    uint k = params.experts_per_token < 1u ? 1u : params.experts_per_token;
    uint pitch = params.x_pitch != 0u ? params.x_pitch : params.width;
    out_[r * params.width + c] = sel < 0 ? uint16_t(0u) : x[(uint(sel) / k) * pitch + c];
}
#endif

#else

struct Scatter {
    uint n;
    uint width;
    uint zero;
};

#if defined(PIE_WGSL)
[[vk::binding(0)]] StructuredBuffer<uint> sorted;
[[vk::binding(1)]] RWStructuredBuffer<uint> y;
#else
PIE_SLOTS_RO(0, sorted);
PIE_SLOTS_RW(1, y);
#endif
[[vk::binding(2)]] StructuredBuffer<int> inv;
PIE_PARAMS(3, Scatter, params);

#if defined(PIE_WGSL)
// A word of the route's row an invocation.
[numthreads(256, 1, 1)]
void main(uint3 thread : SV_DispatchThreadID) {
    uint word = thread.x;
    uint half_width = params.width >> 1;
    if (half_width == 0u) return;
    uint r = word / half_width;
    if (r >= params.n) return;
    uint c2 = word - r * half_width;
    int at = inv[r];
    y[word] = at < 0 ? 0u : sorted[uint(at) * half_width + c2];
}
#else
[numthreads(256, 1, 1)]
void main(uint3 thread : SV_DispatchThreadID) {
    uint c = thread.x;
    uint r = thread.y;
    if (c >= params.width || r >= params.n) return;
    int at = inv[r];
    y[r * params.width + c] = at < 0 ? uint16_t(0u) : sorted[uint(at) * params.width + c];
}
#endif

#endif

// pie:instantiate linear_moe_route_sort PIE_ROUTE_SORT=1 PIE_TARGET=spirv
// pie:instantiate linear_moe_route_sort_256 PIE_ROUTE_SORT=1 PIE_SORT_LANES=256u
// pie:instantiate linear_moe_route_sort_gather PIE_ROUTE_GATHER=1
// pie:instantiate linear_moe_route_sort_scatter PIE_ROUTE_SCATTER=1
