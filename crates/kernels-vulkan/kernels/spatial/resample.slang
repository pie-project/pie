#include "common/bf16.slang"
#include "spatial/grid.slang"


#if defined(PIE_CACHE_GATHER)
PIE_BUFFER_RO(0, slab);
[[vk::binding(1)]] StructuredBuffer<int> slot_ids;
[[vk::binding(2)]] StructuredBuffer<int> grid;
PIE_BUFFER_RW(3, cache);
struct Push { uint clips; uint frames; uint channels; uint slot_stride; uint rows; uint row_span; };
#elif defined(PIE_CACHE_STORE)
PIE_BUFFER_RO(0, x);
PIE_BUFFER_RO(1, cache);
[[vk::binding(2)]] StructuredBuffer<int> slot_ids;
[[vk::binding(3)]] StructuredBuffer<int> grid;
PIE_BUFFER_RW(4, slab);
struct Push { uint clips; uint frames; uint channels; uint slot_stride; uint rows; uint row_span; };
#else
PIE_BUFFER_RO(0, x);
[[vk::binding(1)]] StructuredBuffer<int> grid;
PIE_BUFFER_RW(2, y);
[[vk::binding(3)]] StructuredBuffer<int> o_grid;
// p3 is keep_first (upsample), trim_t (shuffle), the averaged run (avg-down).
#ifdef PIE_UNSHUFFLE
struct Push { uint channels; uint clips; uint r1; uint r2; uint r3; uint rows; uint row_span; };
#else
struct Push { uint channels; uint clips; uint r1; uint r2; uint r3; uint p3; uint rows; uint row_span; };
#endif
#endif
[[vk::push_constant]] ConstantBuffer<Push> pc;

#if defined(PIE_CACHE_GATHER) || defined(PIE_CACHE_STORE)
int cache_clip(uint row, out int base, out SpatialClip found) {
    base = 0;
    found = clip_at(grid, 0);
    int at = 0;
    for (uint l = 0u; l < pc.clips; ++l) {
        SpatialClip c = clip_at(grid, int(l));
        int rows = int(pc.frames) * clip_plane(c);
        if (int(row) < at + rows) {
            base = at;
            found = c;
            return int(l);
        }
        at += rows;
    }
    return -1;
}
#endif

[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID, uint3 group : SV_GroupID) {
    uint col = gid.x;
    uint row = spatial_row(group, pc.row_span);
    if (row >= pc.rows) return;
#if defined(PIE_CACHE_GATHER)
    if (col >= pc.channels) return;
    uint at_out = row * pc.channels + col;
    int base;
    SpatialClip c;
    int found = cache_clip(row, base, c);
    if (found < 0) {
        cache[at_out] = PIE_STORE(0.0);
        return;
    }
    uint local = (row - uint(base)) * pc.channels + col;
    if (local >= pc.slot_stride) {
        cache[at_out] = PIE_STORE(0.0);
        return;
    }
    cache[at_out] = slab[uint(slot_ids[found]) * pc.slot_stride + local];
#elif defined(PIE_CACHE_STORE)
    if (col >= pc.channels) return;
    int base;
    SpatialClip g;
    int found = cache_clip(row, base, g);
    if (found < 0) return;
    int local_row = int(row) - base;
    uint local = uint(local_row) * pc.channels + col;
    if (local >= pc.slot_stride) return;
    int plane = clip_plane(g);
    int f = local_row / plane;
    int hw = local_row - f * plane;
    int src_t = g.t - int(pc.frames) + f;
    PIE_ACT value;
    if (src_t >= 0) {
        value = x[(uint(g.off) + uint(src_t) * uint(plane) + uint(hw)) * pc.channels + col];
    } else {
        value = cache[(uint(base) + uint(f + g.t) * uint(plane) + uint(hw)) * pc.channels + col];
    }
    slab[uint(slot_ids[found]) * pc.slot_stride + local] = value;
#else
#if defined(PIE_UNSHUFFLE)
    uint r = pc.r1 * pc.r2 * pc.r3;
    uint width = pc.channels * r;
#elif defined(PIE_AVG_DOWN)
    uint r = pc.r1 * pc.r2 * pc.r3;
    uint width = pc.channels * r / pc.p3;
#else
    uint width = pc.channels;
#endif
    if (col >= width) return;
    uint e = row * width + col;
    SpatialClip og;
    int l = clip_of(o_grid, int(pc.clips), int(row), og);
    if (l < 0) {
        y[e] = PIE_STORE(0.0);
        return;
    }
    SpatialClip ig = clip_at(grid, l);
    SpatialVoxel o = unravel(og, int(row) - og.off);
#if defined(PIE_UPSAMPLE)
    int ti = pc.p3 != 0u
        ? (o.t == 0 ? 0 : (o.t - 1) / int(pc.r1) + 1)
        : o.t / int(pc.r1);
    int src = ravel(ig, ti, o.h / int(pc.r2), o.w / int(pc.r3));
    y[e] = x[uint(src) * pc.channels + col];
#elif defined(PIE_SHUFFLE)
    int r1 = int(pc.r1), r2 = int(pc.r2), r3 = int(pc.r3);
    int ot = o.t + int(pc.p3);
    int src = ravel(ig, ot / r1, o.h / r2, o.w / r3);
    int block = ((ot % r1) * r2 + (o.h % r2)) * r3 + (o.w % r3);
    uint r = pc.r1 * pc.r2 * pc.r3;
    uint c_in = pc.channels * r;
    y[e] = x[uint(src) * c_in + col * r + uint(block)];
#elif defined(PIE_UNSHUFFLE)
    uint cin = col / r;
    uint block = col - cin * r;
    uint i1 = block / (pc.r2 * pc.r3);
    uint i2 = (block / pc.r3) % pc.r2;
    uint i3 = block % pc.r3;
    int src = ravel(ig, o.t * int(pc.r1) + int(i1), o.h * int(pc.r2) + int(i2),
                    o.w * int(pc.r3) + int(i3));
    y[e] = x[uint(src) * pc.channels + cin];
#elif defined(PIE_AVG_DOWN)
    int pad_t = (int(pc.r1) - ig.t % int(pc.r1)) % int(pc.r1);
    float acc = 0.0;
    for (uint j = 0u; j < pc.p3; ++j) {
        uint q = col * pc.p3 + j;
        uint cin = q / r;
        uint block = q - cin * r;
        uint i1 = block / (pc.r2 * pc.r3);
        uint i2 = (block / pc.r3) % pc.r2;
        uint i3 = block % pc.r3;
        int ti = o.t * int(pc.r1) + int(i1) - pad_t;
        if (ti < 0) continue;
        int src = ravel(ig, ti, o.h * int(pc.r2) + int(i2), o.w * int(pc.r3) + int(i3));
        acc += PIE_LOAD(x[uint(src) * pc.channels + cin]);
    }
    y[e] = PIE_STORE(acc / float(pc.p3));
#endif
#endif
}

// pie:instantiate spatial_upsample_nearest_bf16 PIE_UPSAMPLE=1
// pie:instantiate spatial_upsample_nearest_f32 PIE_UPSAMPLE=1 PIE_ACT_F32=1
// pie:instantiate spatial_pixel_shuffle_bf16 PIE_SHUFFLE=1
// pie:instantiate spatial_pixel_shuffle_f32 PIE_SHUFFLE=1 PIE_ACT_F32=1
// pie:instantiate spatial_pixel_unshuffle_bf16 PIE_UNSHUFFLE=1
// pie:instantiate spatial_pixel_unshuffle_f32 PIE_UNSHUFFLE=1 PIE_ACT_F32=1
// pie:instantiate spatial_avg_down_bf16 PIE_AVG_DOWN=1
// pie:instantiate spatial_cache_gather_bf16 PIE_CACHE_GATHER=1
// pie:instantiate spatial_cache_store_bf16 PIE_CACHE_STORE=1
