#include "prim/bf16.slang"
#include "prim/clip.slang"
#include "prim/params.slang"

// Attention over a clip's voxels, one head the whole channel row: a
// workgroup a query row, four 32-lane strips walking the keys of its
// segment (the clip, or its run of `seg_frames` frames) four at a time, each
// strip with its own running max and sum, folded at the end. A lane holds
// `PIE_C_MAX / 32` channels of its strip's accumulator; the answer lands a
// pair of channels (one bf16 word) a lane, so `channels` is even and at most
// `PIE_C_MAX`. A row no clip holds lands zero.

#define STRIPS 4
#define LANES 32
#define THREADS (STRIPS * LANES)
#define VPT (PIE_C_MAX / LANES)

[[vk::binding(0)]] StructuredBuffer<uint> q;
[[vk::binding(1)]] StructuredBuffer<uint> k;
[[vk::binding(2)]] StructuredBuffer<uint> v;
[[vk::binding(3)]] RWStructuredBuffer<uint> y;
[[vk::binding(4)]] StructuredBuffer<int> grid;

struct Params {
    uint channels;
    uint clips;
    int seg_frames;
    float sm_scale;
    uint rows;
    uint row_span;
    // The rectangles' row 0 in `grid`'s fire-wide rows (a segmented fire's window starts later).
    int base;
};
PIE_PARAMS(5, Params, params);

groupshared float voxel_q[PIE_C_MAX];
groupshared float voxel_fold[STRIPS][LANES];
groupshared float voxel_acc[STRIPS * PIE_C_MAX];
groupshared float voxel_m[STRIPS];
groupshared float voxel_l[STRIPS];
groupshared int voxel_begin;
groupshared int voxel_end;

[shader("compute")]
[numthreads(THREADS, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    int row = int(clip_row(group, params.row_span));
    if (uint(row) >= params.rows) {
        return;
    }
    int tid = int(local.x);
    int strip = tid / LANES;
    int lane = tid % LANES;
    uint c = params.channels;

    if (tid == 0) {
        Clip g;
        int l = clip_of(grid, int(params.clips), row + params.base, g);
        int begin = 0;
        int end = 0;
        if (l >= 0) {
            g.off -= params.base;
            segment_of(g, row - g.off, params.seg_frames, begin, end);
        }
        voxel_begin = begin;
        voxel_end = end;
    }
    GroupMemoryBarrierWithGroupSync();
    // Read uniformly: WGSL lets a barrier follow only workgroup-uniform
    // control flow, and the segment bounds the key loop.
    int begin = workgroupUniformLoad(voxel_begin);
    int end = workgroupUniformLoad(voxel_end);

    uint out_row = uint(row) * c;
    if (end <= begin) {
        for (uint i = 2u * uint(tid); i < c; i += 2u * THREADS) {
            y[(out_row + i) >> 1] = 0u;
        }
        return;
    }
    for (uint i = uint(tid); i < c; i += THREADS) {
        voxel_q[i] = bf16_at(q, out_row + i);
    }
    GroupMemoryBarrierWithGroupSync();

    float acc[VPT];
    for (int u = 0; u < VPT; u++) {
        acc[u] = 0.0;
    }
    float running_max = -3.0e38;
    float running_sum = 0.0;

    int steps = (end - begin + STRIPS - 1) / STRIPS;
    for (int n = 0; n < steps; n++) {
        int j = begin + n * STRIPS + strip;
        bool live = j < end;
        uint kv_row = live ? uint(j) * c : 0u;
        float dot = 0.0;
        if (live) {
            for (int u = 0; u < VPT; u++) {
                uint i = uint(lane + u * LANES);
                if (i < c) {
                    dot += voxel_q[i] * bf16_at(k, kv_row + i);
                }
            }
        }
        voxel_fold[strip][lane] = dot;
        GroupMemoryBarrierWithGroupSync();
        for (int half_ = LANES >> 1; half_ > 0; half_ >>= 1) {
            if (lane < half_) {
                voxel_fold[strip][lane] += voxel_fold[strip][lane + half_];
            }
            GroupMemoryBarrierWithGroupSync();
        }
        dot = voxel_fold[strip][0];
        GroupMemoryBarrierWithGroupSync();
        if (live) {
            float score = dot * params.sm_scale;
            float widened = max(running_max, score);
            float rescale = exp(running_max - widened);
            float weight = exp(score - widened);
            for (int u = 0; u < VPT; u++) {
                uint i = uint(lane + u * LANES);
                if (i < c) {
                    acc[u] = acc[u] * rescale + weight * bf16_at(v, kv_row + i);
                }
            }
            running_sum = running_sum * rescale + weight;
            running_max = widened;
        }
    }

    if (lane == 0) {
        voxel_m[strip] = running_max;
        voxel_l[strip] = running_sum;
    }
    for (int u = 0; u < VPT; u++) {
        uint i = uint(lane + u * LANES);
        if (i < c) {
            voxel_acc[strip * PIE_C_MAX + int(i)] = acc[u];
        }
    }
    GroupMemoryBarrierWithGroupSync();

    float folded_max = -3.0e38;
    for (int s = 0; s < STRIPS; s++) {
        folded_max = max(folded_max, voxel_m[s]);
    }
    float denominator = 0.0;
    for (int s = 0; s < STRIPS; s++) {
        denominator += voxel_l[s] * exp(voxel_m[s] - folded_max);
    }
    float inv = denominator > 0.0 ? 1.0 / denominator : 0.0;
    for (uint i = 2u * uint(tid); i < c; i += 2u * THREADS) {
        float2 sum = float2(0.0);
        for (int s = 0; s < STRIPS; s++) {
            float scale = exp(voxel_m[s] - folded_max);
            sum += float2(voxel_acc[s * PIE_C_MAX + int(i)], voxel_acc[s * PIE_C_MAX + int(i) + 1]) * scale;
        }
        y[(out_row + i) >> 1] = pack_bf16(sum.x * inv, sum.y * inv);
    }
}

// pie:instantiate attn_voxel_bf16_c_256 PIE_C_MAX=256
// pie:instantiate attn_voxel_bf16_c_512 PIE_C_MAX=512
// WGSL's workgroup memory (16 KiB by default) holds four 1024-wide strips
// of accumulators but not the query beside them.
// pie:instantiate attn_voxel_bf16_c_1024 PIE_C_MAX=1024 PIE_TARGET=spirv
