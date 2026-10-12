#include "common/bf16.slang"

// 32 query rows of one head a workgroup, 4 invocations a row; keys stream 4 at a time.

#ifndef PIE_D
#error "tiled attention is compiled at a head width: -DPIE_D=128"
#endif
#define PIE_BQ 32
#define PIE_PARTS 4
#define PIE_BKEY 4
#define PIE_THREADS (PIE_BQ * PIE_PARTS)
#define PIE_DPT (PIE_D / PIE_PARTS)

PIE_BUFFER_RO(0, q);
PIE_BUFFER_RO(1, k);
PIE_BUFFER_RO(2, v);
PIE_BUFFER_RW(3, o);
[[vk::binding(4)]] StructuredBuffer<int> segments;
[[vk::binding(5)]] StructuredBuffer<int> kv_segments;
struct Push { int num_segments; int num_q_heads; int num_kv_heads; int head_dim; float sm_scale; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

groupshared float kt[PIE_BKEY * PIE_D];
groupshared float vt[PIE_BKEY * PIE_D];
groupshared float partial[PIE_BQ * PIE_BKEY * PIE_PARTS];
groupshared int2 spans[PIE_BQ];

int2 span_of(int row) {
    int first = segments[0];
    int total = segments[uint(pc.num_segments)];
    if (row < first || row >= total) return int2(0, 0);
    int lo = 0;
    int hi = pc.num_segments - 1;
    while (lo < hi) {
        int mid = (lo + hi + 1) >> 1;
        if (segments[uint(mid)] <= row) lo = mid; else hi = mid - 1;
    }
    return int2(kv_segments[uint(lo)], kv_segments[uint(lo + 1)]);
}

[numthreads(PIE_THREADS, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint head = group.x;
    uint tid = local.x;
    uint qi = tid / uint(PIE_PARTS);
    uint part = tid % uint(PIE_PARTS);
    int row = int(group.y) * PIE_BQ + int(qi);
    int total = segments[uint(pc.num_segments)];
    uint kv_head = head / uint(pc.num_q_heads / pc.num_kv_heads);
    uint d0 = part * uint(PIE_DPT);

    if (part == 0u) spans[qi] = span_of(row);
    GroupMemoryBarrierWithGroupSync();
    int2 span = spans[qi];
    int lo = 0x7fffffff;
    int hi = 0;
    for (uint i = 0u; i < uint(PIE_BQ); i++) {
        int2 s = spans[i];
        if (s.y > s.x) {
            lo = min(lo, s.x);
            hi = max(hi, s.y);
        }
    }

    bool live = row < total && span.y > span.x;
    uint q_row = (uint(max(row, 0)) * uint(pc.num_q_heads) + head) * uint(PIE_D);
    float qv[PIE_DPT];
    float acc[PIE_DPT];
    [ForceUnroll] for (uint d = 0u; d < uint(PIE_DPT); d++) {
        qv[d] = live ? PIE_LOAD(q[q_row + d0 + d]) * pc.sm_scale : 0.0;
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
            uint at = (uint(in_ ? j : 0) * uint(pc.num_kv_heads) + kv_head) * uint(PIE_D) + d;
            kt[e] = in_ ? PIE_LOAD(k[at]) : 0.0;
            vt[e] = in_ ? PIE_LOAD(v[at]) : 0.0;
        }
        GroupMemoryBarrierWithGroupSync();
        [ForceUnroll] for (uint jj = 0u; jj < uint(PIE_BKEY); jj++) {
            float dot = 0.0;
            [ForceUnroll] for (uint d = 0u; d < uint(PIE_DPT); d++) dot += qv[d] * kt[jj * uint(PIE_D) + d0 + d];
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
            s[jj] = in_ ? sum : -3.0e38;
            top = max(top, s[jj]);
        }
        if (top <= -3.0e38) continue;
        float rescale = exp(m - top);
        l *= rescale;
        [ForceUnroll] for (uint d = 0u; d < uint(PIE_DPT); d++) acc[d] *= rescale;
        [ForceUnroll] for (uint jj = 0u; jj < uint(PIE_BKEY); jj++) {
            if (s[jj] <= -3.0e38) continue;
            float p = exp(s[jj] - top);
            l += p;
            [ForceUnroll] for (uint d = 0u; d < uint(PIE_DPT); d++) acc[d] += p * vt[jj * uint(PIE_D) + d0 + d];
        }
        m = top;
    }

    if (row >= total || row < 0) return;
    float inv = l > 0.0 ? 1.0 / l : 0.0;
    [ForceUnroll] for (uint d = 0u; d < uint(PIE_DPT); d++) o[q_row + d0 + d] = PIE_STORE(acc[d] * inv);
}

// pie:instantiate ragged_tiled_bf16_d_64 PIE_D=64
// pie:instantiate ragged_tiled_bf16_d_128 PIE_D=128
