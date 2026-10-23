#include "prim/bf16.slang"
#include "prim/params.slang"

// Mixture-of-experts routing: each token row's experts and their weights.
// One workgroup a row stages the row's scores in workgroup memory and its
// first lane picks the top-k. `PIE_SOFTMAX`: a softmax over the selected
// logits (`PIE_SCALED`: each weight times its expert's gain).
// `PIE_SIGMOID`: sigmoid scores, ranked by score plus a correction bias
// (`PIE_BIASED`) or `sqrt(softplus)` scores so (`PIE_SQRT_SOFTPLUS`), the
// weights renormalized and scaled. `PIE_HASH`: experts read from a token-id
// hash table (one i64 an entry), weighted by `sqrt(softplus)` of their logit,
// a row a lane. `PIE_GROUP_ROUTES`: each row's routes are its group slots.

#define MAX_TOPK 16u
#define MAX_EXPERTS 1024u
static const float NEG = -3.0e38;

#if defined(PIE_SOFTMAX)

[[vk::binding(0)]] StructuredBuffer<uint> logits;
[[vk::binding(1)]] RWStructuredBuffer<int> expert_ids;
[[vk::binding(2)]] RWStructuredBuffer<float> expert_weights;
#if defined(PIE_SCALED)
[[vk::binding(3)]] StructuredBuffer<uint> per_expert_scale;
#define PIE_PARAMS_AT 4
#else
#define PIE_PARAMS_AT 3
#endif

struct Params {
    uint n_experts;
    uint experts_per_token;
    uint softmax_over_all;
    uint logits_pitch;
};
PIE_PARAMS(PIE_PARAMS_AT, Params, params);

groupshared float s_logits[MAX_EXPERTS];
groupshared float chosen[MAX_TOPK];
groupshared uint chosen_i[MAX_TOPK];

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint lid = local.x;
    uint row = group.y;
    uint n = min(params.n_experts, MAX_EXPERTS);
    uint k = min(params.experts_per_token, MAX_TOPK);
    uint pitch = params.logits_pitch != 0u ? params.logits_pitch : n;
    for (uint e = lid; e < MAX_EXPERTS; e += uint(PIE_GROUP_X)) {
        s_logits[e] = e < n ? bf16_at(logits, row * pitch + e) : NEG;
    }
    GroupMemoryBarrierWithGroupSync();
    if (lid != 0u) {
        return;
    }
    float all_max = NEG;
    for (uint e = 0u; e < n; e++) {
        all_max = max(all_max, s_logits[e]);
    }
    float all_sum = 0.0;
    for (uint e = 0u; e < n; e++) {
        all_sum += exp(s_logits[e] - all_max);
    }
    for (uint r = 0u; r < k; r++) {
        float best = NEG;
        uint best_i = 0xffffffffu;
        for (uint e = 0u; e < n; e++) {
            if (s_logits[e] > best) {
                best = s_logits[e];
                best_i = e;
            }
        }
        chosen[r] = best;
        chosen_i[r] = best_i;
        if (best_i < n) {
            s_logits[best_i] = NEG;
        }
    }
    float mx = all_max;
    float sum = all_sum;
    if (params.softmax_over_all == 0u) {
        mx = NEG;
        for (uint r = 0u; r < k; r++) {
            mx = max(mx, chosen[r]);
        }
        sum = 0.0;
        for (uint r = 0u; r < k; r++) {
            sum += exp(chosen[r] - mx);
        }
    }
    for (uint r = 0u; r < k; r++) {
        uint e = chosen_i[r];
        float w = exp(chosen[r] - mx) / sum;
#if defined(PIE_SCALED)
        w *= bf16_at(per_expert_scale, e);
#endif
        expert_ids[row * k + r] = int(e);
        expert_weights[row * k + r] = w;
    }
}

#elif defined(PIE_SIGMOID)

[[vk::binding(0)]] StructuredBuffer<uint> logits;
#if defined(PIE_BIASED)
[[vk::binding(1)]] StructuredBuffer<float> correction;
[[vk::binding(2)]] RWStructuredBuffer<int> expert_ids;
[[vk::binding(3)]] RWStructuredBuffer<float> expert_weights;
#define PIE_PARAMS_AT 4
#else
[[vk::binding(1)]] RWStructuredBuffer<int> expert_ids;
[[vk::binding(2)]] RWStructuredBuffer<float> expert_weights;
#define PIE_PARAMS_AT 3
#endif

struct Params {
    uint n_experts;
    uint experts_per_token;
    uint renormalize;
    float scaling;
};
PIE_PARAMS(PIE_PARAMS_AT, Params, params);

groupshared float s_score[MAX_EXPERTS];
groupshared float s_rank[MAX_EXPERTS];

float score_of(float x) {
#if defined(PIE_SQRT_SOFTPLUS)
    return sqrt(max(x <= 20.0 ? log(1.0 + exp(x)) : x, 0.0));
#else
    return 1.0 / (1.0 + exp(-x));
#endif
}

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint lid = local.x;
    uint row = group.y;
    uint n = min(params.n_experts, MAX_EXPERTS);
    uint k = min(params.experts_per_token, MAX_TOPK);
    uint picks = min(k, n);
    for (uint e = lid; e < MAX_EXPERTS; e += uint(PIE_GROUP_X)) {
        float s = NEG;
        float rank = NEG;
        if (e < n) {
            s = score_of(bf16_at(logits, row * n + e));
#if defined(PIE_BIASED)
            rank = s + correction[e];
#else
            rank = s;
#endif
        }
        s_score[e] = s;
        s_rank[e] = rank;
    }
    GroupMemoryBarrierWithGroupSync();
    if (lid != 0u) {
        return;
    }
    float sum = 0.0;
    for (uint r = 0u; r < picks; r++) {
        float best = NEG;
        uint best_i = 0u;
        for (uint e = 0u; e < n; e++) {
            if (s_rank[e] > best) {
                best = s_rank[e];
                best_i = e;
            }
        }
        s_rank[best_i] = NEG;
        expert_ids[row * k + r] = int(best_i);
        expert_weights[row * k + r] = s_score[best_i];
        sum += s_score[best_i];
    }
    for (uint r = picks; r < k; r++) {
        expert_ids[row * k + r] = 0;
        expert_weights[row * k + r] = 0.0;
    }
    float scale = (params.renormalize != 0u && sum > 0.0) ? params.scaling / sum : params.scaling;
    for (uint r = 0u; r < k; r++) {
        expert_weights[row * k + r] *= scale;
    }
}

#elif defined(PIE_HASH)

[[vk::binding(0)]] StructuredBuffer<uint> token_ids;
[[vk::binding(1)]] StructuredBuffer<uint> tid2eid;
[[vk::binding(2)]] StructuredBuffer<uint> logits;
[[vk::binding(3)]] RWStructuredBuffer<int> expert_ids;
[[vk::binding(4)]] RWStructuredBuffer<float> expert_weights;

struct Params {
    uint vocab;
    uint n_experts;
    uint experts_per_token;
    uint renormalize;
    float scaling;
    uint rows;
};
PIE_PARAMS(5, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint row = gid.x;
    if (row >= params.rows) {
        return;
    }
    uint k = min(params.experts_per_token, MAX_TOPK);
    uint raw = token_ids[row];
    uint tid = (params.vocab > 0u && raw < params.vocab) ? raw : 0u;
    uint picks = tid * params.experts_per_token;
    uint base = row * params.experts_per_token;
    float sum = 0.0;
    for (uint r = 0u; r < k; r++) {
        uint lo = tid2eid[2u * (picks + r)];
        uint hi = tid2eid[2u * (picks + r) + 1u];
        float w = 0.0;
        // An i64 expert id: one past u32 or negative names no expert.
        if (hi == 0u && lo < params.n_experts) {
            float x = bf16_at(logits, row * params.n_experts + lo);
            w = sqrt(x <= 20.0 ? log(1.0 + exp(x)) : x);
        }
        expert_ids[base + r] = int(lo);
        expert_weights[base + r] = w;
        sum += w;
    }
    float scale = (params.renormalize != 0u && sum > 0.0) ? params.scaling / sum : params.scaling;
    for (uint r = 0u; r < k; r++) {
        expert_weights[base + r] *= scale;
    }
}

#elif defined(PIE_GROUP_ROUTES)

[[vk::binding(0)]] RWStructuredBuffer<int> routes;

struct Params {
    uint groups;
};
PIE_PARAMS(1, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    if (gid.x < params.groups) {
        routes[gid.y * params.groups + gid.x] = int(gid.x);
    }
}

#else
#error "linear/moe/route.slang: no kernel selected"
#endif

// pie:instantiate linear_moe_route_topk_softmax_bf16 PIE_SOFTMAX=1 PIE_GROUP_X=256
// pie:instantiate linear_moe_route_topk_softmax_scaled_bf16 PIE_SOFTMAX=1 PIE_SCALED=1 PIE_GROUP_X=256
// pie:instantiate linear_moe_route_topk_sigmoid_bf16 PIE_SIGMOID=1 PIE_GROUP_X=256
// pie:instantiate linear_moe_route_topk_sigmoid_biased_bf16 PIE_SIGMOID=1 PIE_BIASED=1 PIE_GROUP_X=256
// pie:instantiate linear_moe_route_topk_sqrt_softplus_bf16 PIE_SIGMOID=1 PIE_BIASED=1 PIE_SQRT_SOFTPLUS=1 PIE_GROUP_X=256
// pie:instantiate linear_moe_route_hash_bf16 PIE_HASH=1 PIE_GROUP_X=256
// pie:instantiate linear_moe_route_group_routes PIE_GROUP_ROUTES=1 PIE_GROUP_X=256
