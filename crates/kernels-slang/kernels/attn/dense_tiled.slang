#include "prim/bf16.slang"
#include "prim/params.slang"

// Ragged dense attention, tiled: 32 query rows of one head a workgroup, four
// lanes a row each holding a quarter of the head, keys streamed four at a
// time through workgroup memory. Query row `r` of segment `s` attends keys
// `kv_segments[s]..kv_segments[s + 1]`; with `PIE_CUSTOM`, only where bit
// `qo * kv_len + kv` (each within the segment) of `mask_bits`, from byte
// `mask_indptr[s]`, is set. bf16 rows; a lane's quarter is an even number of
// columns, so it lands whole words.

#ifndef PIE_D
#error "attn/dense_tiled.slang is compiled at a head width: -DPIE_D"
#endif
#define PIE_BQ 32
#define PIE_PARTS 4
#define PIE_BKEY 4
#define PIE_THREADS (PIE_BQ * PIE_PARTS)
#define PIE_DPT (PIE_D / PIE_PARTS)

[[vk::binding(0)]] StructuredBuffer<uint> q;
[[vk::binding(1)]] StructuredBuffer<uint> k;
[[vk::binding(2)]] StructuredBuffer<uint> v;
[[vk::binding(3)]] RWStructuredBuffer<uint> o;
[[vk::binding(4)]] StructuredBuffer<int> segments;
[[vk::binding(5)]] StructuredBuffer<int> kv_segments;
#ifdef PIE_CUSTOM
[[vk::binding(6)]] StructuredBuffer<uint> mask_bits;
[[vk::binding(7)]] StructuredBuffer<int> mask_indptr;
#define PIE_PARAMS_AT 8
#else
#define PIE_PARAMS_AT 6
#endif

struct Params {
    int num_segments;
    int num_q_heads;
    int num_kv_heads;
    int head_dim;
    float sm_scale;
};
PIE_PARAMS(PIE_PARAMS_AT, Params, params);

groupshared float kt[PIE_BKEY * PIE_D];
groupshared float vt[PIE_BKEY * PIE_D];
groupshared float partial[PIE_BQ * PIE_BKEY * PIE_PARTS];
groupshared int2 spans[PIE_BQ];
groupshared int segs[PIE_BQ];
groupshared int tile_lo;
groupshared int tile_hi;

// The segment row `row` stands in, or -1.
int segment_of(int row) {
    if (row < segments[0] || row >= segments[uint(params.num_segments)]) {
        return -1;
    }
    int lo = 0;
    int hi = params.num_segments - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (segments[uint(mid)] <= row) {
            lo = mid;
        } else {
            hi = mid - 1;
        }
    }
    return lo;
}

#ifdef PIE_CUSTOM
// Whether row `row` of segment `seg` keeps key `j`.
bool mask_keeps(int seg, int row, int j) {
    uint qo = uint(row - segments[uint(seg)]);
    uint kv = uint(j - kv_segments[uint(seg)]);
    uint kv_len = uint(kv_segments[uint(seg) + 1u] - kv_segments[uint(seg)]);
    uint bit = uint(mask_indptr[uint(seg)]) * 8u + qo * kv_len + kv;
    return ((mask_bits[bit >> 5u] >> (bit & 31u)) & 1u) != 0u;
}
#endif

[shader("compute")]
[numthreads(PIE_THREADS, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint head = group.x;
    uint tid = local.x;
    uint qi = tid / uint(PIE_PARTS);
    uint part = tid % uint(PIE_PARTS);
    int row = int(group.y) * PIE_BQ + int(qi);
    int total = segments[uint(params.num_segments)];
    uint kv_head = head / uint(params.num_q_heads / params.num_kv_heads);
    uint d0 = part * uint(PIE_DPT);

    if (part == 0u) {
        int seg = segment_of(row);
        segs[qi] = seg;
        spans[qi] = seg < 0 ? int2(0, 0)
                            : int2(kv_segments[uint(seg)], kv_segments[uint(seg + 1)]);
    }
    GroupMemoryBarrierWithGroupSync();
    // The keys any row of the tile reads, folded once and read uniformly:
    // they bound the loop of barriers below.
    if (tid == 0u) {
        int lo = 0x7fffffff;
        int hi = 0;
        for (uint i = 0u; i < uint(PIE_BQ); i++) {
            int2 s = spans[i];
            if (s.y > s.x) {
                lo = min(lo, s.x);
                hi = max(hi, s.y);
            }
        }
        tile_lo = lo;
        tile_hi = hi;
    }
    GroupMemoryBarrierWithGroupSync();
    int2 span = spans[qi];
#ifdef PIE_CUSTOM
    int seg = segs[qi];
#endif
    int lo = workgroupUniformLoad(tile_lo);
    int hi = workgroupUniformLoad(tile_hi);

    bool live = row < total && span.y > span.x;
    uint q_row = (uint(max(row, 0)) * uint(params.num_q_heads) + head) * uint(PIE_D);
    float qv[PIE_DPT];
    float acc[PIE_DPT];
    [ForceUnroll] for (uint d = 0u; d < uint(PIE_DPT); d++) {
        qv[d] = live ? bf16_at(q, q_row + d0 + d) * params.sm_scale : 0.0;
        acc[d] = 0.0;
    }
    float m = -3.0e38;
    float l = 0.0;

    for (int t = lo; t < hi; t += PIE_BKEY) {
        GroupMemoryBarrierWithGroupSync();
        for (uint e = tid; e < uint(PIE_BKEY * PIE_D); e += uint(PIE_THREADS)) {
            int j = t + int(e / uint(PIE_D));
            uint d = e % uint(PIE_D);
            bool in_ = j < hi;
            uint at = (uint(in_ ? j : 0) * uint(params.num_kv_heads) + kv_head) * uint(PIE_D) + d;
            kt[e] = in_ ? bf16_at(k, at) : 0.0;
            vt[e] = in_ ? bf16_at(v, at) : 0.0;
        }
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll] for (uint jj = 0u; jj < uint(PIE_BKEY); jj++) {
            float dot = 0.0;
            [ForceUnroll] for (uint d = 0u; d < uint(PIE_DPT); d++) {
                dot += qv[d] * kt[jj * uint(PIE_D) + d0 + d];
            }
            partial[(qi * uint(PIE_BKEY) + jj) * uint(PIE_PARTS) + part] = dot;
        }
        GroupMemoryBarrierWithGroupSync();
        float s[PIE_BKEY];
        float top = m;
        [ForceUnroll] for (uint jj = 0u; jj < uint(PIE_BKEY); jj++) {
            int j = t + int(jj);
            uint at = (qi * uint(PIE_BKEY) + jj) * uint(PIE_PARTS);
            float sum = partial[at] + partial[at + 1u] + partial[at + 2u] + partial[at + 3u];
            bool in_ = live && j >= span.x && j < span.y;
#ifdef PIE_CUSTOM
            if (in_) {
                in_ = mask_keeps(seg, row, j);
            }
#endif
            s[jj] = in_ ? sum : -3.0e38;
            top = max(top, s[jj]);
        }
        // A row with no live key in this block skips it without leaving
        // the loop body: every lane reaches the next block's barriers.
        if (top > -3.0e38) {
            float rescale = exp(m - top);
            l *= rescale;
            [ForceUnroll] for (uint d = 0u; d < uint(PIE_DPT); d++) {
                acc[d] *= rescale;
            }
            [ForceUnroll] for (uint jj = 0u; jj < uint(PIE_BKEY); jj++) {
                if (s[jj] > -3.0e38) {
                    float p = exp(s[jj] - top);
                    l += p;
                    [ForceUnroll] for (uint d = 0u; d < uint(PIE_DPT); d++) {
                        acc[d] += p * vt[jj * uint(PIE_D) + d0 + d];
                    }
                }
            }
            m = top;
        }
    }

    if (row >= total || row < 0) {
        return;
    }
    float inv = l > 0.0 ? 1.0 / l : 0.0;
    [ForceUnroll] for (uint d = 0u; d < uint(PIE_DPT); d += 2u) {
        o[(q_row + d0 + d) >> 1] = pack_bf16(acc[d] * inv, acc[d + 1u] * inv);
    }
}

// pie:instantiate attn_dense_tiled_ragged_bf16_d_64 PIE_D=64
// pie:instantiate attn_dense_tiled_ragged_bf16_d_128 PIE_D=128
// pie:instantiate attn_dense_tiled_ragged_custom_bf16_d_64 PIE_D=64 PIE_CUSTOM=1
// pie:instantiate attn_dense_tiled_ragged_custom_bf16_d_128 PIE_D=128 PIE_CUSTOM=1
