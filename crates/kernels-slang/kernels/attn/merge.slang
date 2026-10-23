#include "prim/bf16.slang"
#include "prim/params.slang"

// What follows an attention read, on bf16 outputs a word (two head
// columns) a lane. `PIE_MERGE`: two readings of the same rows folded by
// their base-2 log-sum-exps into one, with the merged LSE; a reading whose
// LSE is not finite gives way to the other. `PIE_SINK`: each head rescaled
// in place by `sigmoid(lse ln2 - sink)`, its learned sink's share.
// `PIE_SOFTCAP`: logits capped in place, `cap * tanh(x / cap)`, over a flat
// run folded onto two grid axes. `PIE_<KERNEL>` picks the kernel.

float2 word_pair(uint w) {
    return float2(bf16_to_f32(w & 0xffffu), bf16_to_f32(w >> 16));
}

bool finite(float v) {
    return (asuint(v) & 0x7f800000u) != 0x7f800000u;
}

#if defined(PIE_MERGE)

[[vk::binding(0)]] StructuredBuffer<uint> o1;
[[vk::binding(1)]] StructuredBuffer<float> lse1;
[[vk::binding(2)]] StructuredBuffer<uint> o2;
[[vk::binding(3)]] StructuredBuffer<float> lse2;
[[vk::binding(4)]] RWStructuredBuffer<uint> o;
[[vk::binding(5)]] RWStructuredBuffer<float> lse;

struct Params {
    int head_dim;
    int heads;
};
PIE_PARAMS(6, Params, params);

// The grid is (head words, heads, rows).
[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint d = 2u * gid.x;
    if (d >= uint(params.head_dim)) {
        return;
    }
    uint col = gid.z * uint(params.heads) + gid.y;
    uint i = (col * uint(params.head_dim) + d) >> 1;
    float l1 = lse1[col];
    float l2 = lse2[col];
    if (!finite(l2) || !finite(l1)) {
        bool first = !finite(l2);
        o[i] = first ? o1[i] : o2[i];
        if (d == 0u) {
            lse[col] = first ? l1 : l2;
        }
        return;
    }
    float top = max(l1, l2);
    float w1 = exp2(l1 - top);
    float w2 = exp2(l2 - top);
    float2 v = (word_pair(o1[i]) * w1 + word_pair(o2[i]) * w2) / (w1 + w2);
    o[i] = pack_bf16(v.x, v.y);
    if (d == 0u) {
        lse[col] = top + log2(w1 + w2);
    }
}

#elif defined(PIE_SINK)

[[vk::binding(0)]] RWStructuredBuffer<uint> o;
[[vk::binding(1)]] StructuredBuffer<float> lse;
[[vk::binding(2)]] StructuredBuffer<uint> sinks;

struct Params {
    int head_dim;
    int heads;
};
PIE_PARAMS(3, Params, params);

// The grid is (head words, heads, rows).
[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint d = 2u * gid.x;
    if (d >= uint(params.head_dim)) {
        return;
    }
    uint h = gid.y;
    uint th = gid.z * uint(params.heads) + h;
    float r = 1.0;
    float l = lse[th];
    if (finite(l)) {
        r = 1.0 / (1.0 + exp(-(l * 0.69314718055994530942 - bf16_at(sinks, h))));
    }
    uint w = (th * uint(params.head_dim) + d) >> 1;
    float2 v = word_pair(o[w]) * r;
    o[w] = pack_bf16(v.x, v.y);
}

#elif defined(PIE_SOFTCAP)

[[vk::binding(0)]] RWStructuredBuffer<uint> logits;

struct Params {
    float cap;
    int n;
    uint row_groups;
};
PIE_PARAMS(1, Params, params);

float softcap(float v) {
    float t = 1.0 - 2.0 / (exp(2.0 * abs(v / params.cap)) + 1.0);
    return params.cap * (v < 0.0 ? -t : t);
}

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint n = uint(max(params.n, 0));
    uint w = (group.y * params.row_groups + group.x) * uint(PIE_GROUP_X) + local.x;
    uint lo = 2u * w;
    if (lo >= n) {
        return;
    }
    uint word = logits[w];
    float a = softcap(bf16_to_f32(word & 0xffffu));
    if (lo + 1u < n) {
        logits[w] = pack_bf16(a, softcap(bf16_to_f32(word >> 16)));
    } else {
        logits[w] = (word & 0xffff0000u) | f32_to_bf16(a);
    }
}

#else
#error "attn/merge.slang: no kernel selected"
#endif

// pie:instantiate attn_merge_lse_bf16 PIE_MERGE=1 PIE_GROUP_X=256
// pie:instantiate attn_sink_bf16 PIE_SINK=1 PIE_GROUP_X=64
// pie:instantiate attn_softcap_bf16 PIE_SOFTCAP=1 PIE_GROUP_X=256
