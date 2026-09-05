#include "common/bf16.slang"

#define ROUTER_MAX_TOPK 16u
#define ROUTER_MAX_EXPERTS 1024u

#ifndef PIE_GROUP_X
#define PIE_GROUP_X 256
#endif

#if defined(PIE_ROUTER_TOPK)

PIE_BUFFER_RO(0, logits);
[[vk::binding(1)]] RWStructuredBuffer<int> expert_ids;
[[vk::binding(2)]] RWStructuredBuffer<float> expert_weights;
PIE_BUFFER_RO(3, per_expert_scale);

struct Push {
    uint n_experts;
    uint experts_per_token;
    uint softmax_over_all;
    uint logits_pitch;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

groupshared float s_logits[ROUTER_MAX_EXPERTS];
groupshared float chosen[ROUTER_MAX_TOPK];
groupshared uint chosen_i[ROUTER_MAX_TOPK];

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group: SV_GroupID, uint3 local: SV_GroupThreadID) {
    uint lid = local.x;
    uint row = group.y;
    uint n = min(pc.n_experts, ROUTER_MAX_EXPERTS);
    uint k = min(pc.experts_per_token, ROUTER_MAX_TOPK);
    uint pitch = pc.logits_pitch != 0u ? pc.logits_pitch : n;
    float neg = -3.0e38;

    for (uint e = lid; e < ROUTER_MAX_EXPERTS; e += PIE_GROUP_X) {
        s_logits[e] = e < n ? PIE_LOAD(logits[row * pitch + e]) : neg;
    }
    GroupMemoryBarrierWithGroupSync();

    if (lid == 0u) {
        float all_max = neg;
        for (uint e = 0u; e < n; e++) all_max = max(all_max, s_logits[e]);
        float all_sum = 0.0;
        for (uint e = 0u; e < n; e++) all_sum += exp(s_logits[e] - all_max);

        for (uint r = 0u; r < k; r++) {
            float best = neg;
            uint best_i = 0xffffffffu;
            for (uint e = 0u; e < n; e++) {
                float v = s_logits[e];
                if (v > best) { best = v; best_i = e; }
            }
            chosen[r] = best;
            chosen_i[r] = best_i;
            if (best_i < n) s_logits[best_i] = neg;
        }

        float mx = all_max;
        float sum = all_sum;
        if (pc.softmax_over_all == 0u) {
            mx = neg;
            for (uint r = 0u; r < k; r++) mx = max(mx, chosen[r]);
            sum = 0.0;
            for (uint r = 0u; r < k; r++) sum += exp(chosen[r] - mx);
        }
        for (uint r = 0u; r < k; r++) {
            uint e = chosen_i[r];
            float w = exp(chosen[r] - mx) / sum;
#if defined(PIE_SCALED)
            w *= PIE_LOAD(per_expert_scale[e]);
#endif
            expert_ids[row * k + r] = int(e);
            expert_weights[row * k + r] = w;
        }
    }
}

#elif defined(PIE_ROUTER_SIGMOID)

PIE_BUFFER_RO(0, logits);
#if defined(PIE_BIASED)
[[vk::binding(1)]] StructuredBuffer<float> correction;
[[vk::binding(2)]] RWStructuredBuffer<int> expert_ids;
[[vk::binding(3)]] RWStructuredBuffer<float> expert_weights;
#else
[[vk::binding(1)]] RWStructuredBuffer<int> expert_ids;
[[vk::binding(2)]] RWStructuredBuffer<float> expert_weights;
#endif

struct Push {
    uint n_experts;
    uint experts_per_token;
    uint renormalize;
    float scaling;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

groupshared float s_score[ROUTER_MAX_EXPERTS];
groupshared float s_rank[ROUTER_MAX_EXPERTS];

float score_of(float x) {
#if defined(PIE_SQRT_SOFTPLUS)
    float sp = x > 20.0 ? x : log(1.0 + exp(x));
    return sqrt(max(sp, 0.0));
#else
    return 1.0 / (1.0 + exp(-x));
#endif
}

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group: SV_GroupID, uint3 local: SV_GroupThreadID) {
    uint lid = local.x;
    uint row = group.y;
    uint n = min(pc.n_experts, ROUTER_MAX_EXPERTS);
    uint k = min(pc.experts_per_token, ROUTER_MAX_TOPK);
    uint picks = min(k, n);
    float neg = -3.0e38;

    for (uint e = lid; e < ROUTER_MAX_EXPERTS; e += PIE_GROUP_X) {
        float s = e < n ? score_of(PIE_LOAD(logits[row * n + e])) : neg;
        s_score[e] = s;
#if defined(PIE_BIASED)
        s_rank[e] = e < n ? s + correction[e] : neg;
#else
        s_rank[e] = s;
#endif
    }
    GroupMemoryBarrierWithGroupSync();

    if (lid == 0u) {
        float sum = 0.0;
        for (uint r = 0u; r < picks; r++) {
            float best = neg;
            uint best_i = 0u;
            for (uint e = 0u; e < n; e++) {
                float v = s_rank[e];
                if (v > best) { best = v; best_i = e; }
            }
            s_rank[best_i] = neg;
            float w = s_score[best_i];
            expert_ids[row * k + r] = int(best_i);
            expert_weights[row * k + r] = w;
            sum += w;
        }
        for (uint r = picks; r < k; r++) {
            expert_ids[row * k + r] = 0;
            expert_weights[row * k + r] = 0.0;
        }
        float scale = (pc.renormalize != 0u && sum > 0.0) ? pc.scaling / sum : pc.scaling;
        for (uint r = 0u; r < k; r++) expert_weights[row * k + r] *= scale;
    }
}

#elif defined(PIE_EXPERT_COMBINE)

PIE_BUFFER_RO(0, routed);
[[vk::binding(1)]] StructuredBuffer<float> expert_weights;
PIE_BUFFER_RW(2, out_);

struct Push {
    uint width;
    uint experts_per_token;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 thread: SV_DispatchThreadID) {
    uint c = thread.x;
    uint row = thread.y;
    if (c >= pc.width) return;
    uint base = row * pc.experts_per_token;
    float acc = 0.0;
    for (uint e = 0u; e < pc.experts_per_token; e++) {
        acc += expert_weights[base + e] * PIE_LOAD(routed[(base + e) * pc.width + c]);
    }
    out_[row * pc.width + c] = PIE_STORE(acc);
}

#elif defined(PIE_EXPERT_BIAS_COMBINE)

PIE_BUFFER_RO(0, x);
PIE_BUFFER_RO(1, bias);
[[vk::binding(2)]] StructuredBuffer<int> expert_ids;
[[vk::binding(3)]] StructuredBuffer<float> expert_weights;
PIE_BUFFER_RW(4, out_);

struct Push {
    uint width;
    uint experts_per_token;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 thread: SV_DispatchThreadID) {
    uint c = thread.x;
    uint row = thread.y;
    if (c >= pc.width) return;
    uint at = row * pc.width + c;
    uint base = row * pc.experts_per_token;
    float acc = PIE_LOAD(x[at]);
    for (uint e = 0u; e < pc.experts_per_token; e++) {
        int expert = expert_ids[base + e];
        if (expert < 0) continue;
        acc += expert_weights[base + e] * PIE_LOAD(bias[uint(expert) * pc.width + c]);
    }
    out_[at] = PIE_STORE(acc);
}

#elif defined(PIE_HASH_ROUTE)

[[vk::binding(0)]] StructuredBuffer<uint> token_ids;
[[vk::binding(1)]] StructuredBuffer<int64_t> tid2eid;
PIE_BUFFER_RO(2, logits);
[[vk::binding(3)]] RWStructuredBuffer<int> expert_ids;
[[vk::binding(4)]] RWStructuredBuffer<float> expert_weights;

struct Push {
    uint vocab;
    uint n_experts;
    uint experts_per_token;
    uint renormalize;
    float scaling;
    uint rows;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

float hash_sqrt_softplus(float x) {
    float sp = x > 20.0 ? x : log(1.0 + exp(x));
    return sqrt(sp);
}

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 thread: SV_DispatchThreadID) {
    uint row = thread.x;
    if (row >= pc.rows) return;
    uint k = min(pc.experts_per_token, ROUTER_MAX_TOPK);
    uint raw = token_ids[row];
    uint tid = (pc.vocab > 0u && raw < pc.vocab) ? raw : 0u;
    uint picks = tid * pc.experts_per_token;
    uint score = row * pc.n_experts;
    uint base = row * pc.experts_per_token;
    float sum = 0.0;
    for (uint r = 0u; r < k; r++) {
        int64_t e = tid2eid[picks + r];
        float w = (e >= 0 && e < int64_t(pc.n_experts)) ? hash_sqrt_softplus(PIE_LOAD(logits[score + uint(e)])) : 0.0;
        expert_ids[base + r] = int(e);
        expert_weights[base + r] = w;
        sum += w;
    }
    float scale = (pc.renormalize != 0u && sum > 0.0) ? pc.scaling / sum : pc.scaling;
    for (uint r = 0u; r < k; r++) expert_weights[base + r] *= scale;
}

#elif defined(PIE_GROUP_ROUTES)

[[vk::binding(0)]] RWStructuredBuffer<int> routes;

struct Push {
    uint groups;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 thread: SV_DispatchThreadID) {
    uint slot = thread.x;
    uint row = thread.y;
    if (slot >= pc.groups) return;
    routes[row * pc.groups + slot] = int(slot);
}

#else

PIE_BUFFER_RO(0, routed);
PIE_BUFFER_RO(1, shared_);
PIE_BUFFER_RO(2, gate);
PIE_BUFFER_RW(3, out_);

struct Push {
    uint width;
};
[[vk::push_constant]] ConstantBuffer<Push> pc;

[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 thread: SV_DispatchThreadID) {
    uint c = thread.x;
    uint row = thread.y;
    if (c >= pc.width) return;
    float g = 1.0 / (1.0 + exp(-PIE_LOAD(gate[row])));
    uint at = row * pc.width + c;
    out_[at] = PIE_STORE(PIE_LOAD(routed[at]) + g * PIE_LOAD(shared_[at]));
}
#endif

// pie:instantiate router_topk_f32w_bfloat16 PIE_ROUTER_TOPK=1 PIE_GROUP_X=256
// pie:instantiate router_topk_scaled_f32w_bfloat16 PIE_ROUTER_TOPK=1 PIE_SCALED=1 PIE_GROUP_X=256
// pie:instantiate router_topk_sigmoid PIE_ROUTER_SIGMOID=1 PIE_GROUP_X=256
// pie:instantiate router_topk_sigmoid_biased PIE_ROUTER_SIGMOID=1 PIE_BIASED=1 PIE_GROUP_X=256
// pie:instantiate router_topk_sqrt_softplus PIE_ROUTER_SIGMOID=1 PIE_BIASED=1 PIE_SQRT_SOFTPLUS=1 PIE_GROUP_X=256
// pie:instantiate expert_combine PIE_EXPERT_COMBINE=1 PIE_GROUP_X=256
// pie:instantiate expert_bias_combine PIE_EXPERT_BIAS_COMBINE=1 PIE_GROUP_X=256
// pie:instantiate shared_expert_combine PIE_GROUP_X=256
// pie:instantiate hash_route_gather PIE_HASH_ROUTE=1 PIE_GROUP_X=256
// pie:instantiate group_routes PIE_GROUP_ROUTES=1 PIE_GROUP_X=256
