#include "prim/bf16.slang"
#include "prim/params.slang"
#include "prim/reduce.slang"

// Hyper-connections: a row carried as `m` bf16 streams of `h` each, side by
// side. Expand copies a row into every stream; the per-row RMS norm widens a
// stream row to f32; gates turn a row's mix logits into pre, post and
// Sinkhorn-balanced combination weights and land the layer input; fold mixes
// the layer output back into the streams; collapse sums the streams under
// sigmoid gates; mix, inject and the PLE gate are the trunk's per-stream
// gates. Streams land two values a word, so `h` is even; `m` is at most
// `HC_MAX_MULT`. `PIE_<KERNEL>` picks the kernel.

#define HC_MAX_MULT 8u

float sigmoid(float v) {
    return 1.0 / (1.0 + exp(-v));
}

float2 word_at(StructuredBuffer<uint> buf, uint i) {
    uint w = buf[i >> 1];
    return float2(bf16_to_f32(w & 0xffffu), bf16_to_f32(w >> 16));
}

#if defined(PIE_EXPAND)

[[vk::binding(0)]] StructuredBuffer<uint> x;
[[vk::binding(1)]] RWStructuredBuffer<uint> y;

struct Params {
    int n_rows;
    int m;
    int h;
};
PIE_PARAMS(2, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint words = uint(max(params.h, 0)) / 2u;
    uint w = gid.x;
    uint n = gid.y;
    if (w >= words || n >= uint(max(params.n_rows, 0))) {
        return;
    }
    uint m = uint(max(params.m, 0));
    uint v = x[n * words + w];
    for (uint s = 0u; s < m; s++) {
        y[(n * m + s) * words + w] = v;
    }
}

#elif defined(PIE_RMSNORM)

[[vk::binding(0)]] StructuredBuffer<uint> x;
[[vk::binding(1)]] RWStructuredBuffer<float> y;

struct Params {
    int dim;
    float eps;
};
PIE_PARAMS(2, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint dim = uint(max(params.dim, 0));
    uint base = group.x * dim;
    uint lid = local.x;
    float acc = 0.0;
    for (uint d = lid; d < dim; d += uint(PIE_GROUP_X)) {
        float v = bf16_at(x, base + d);
        acc += v * v;
    }
    float inv = pie_inv_rms(acc, lid, dim, params.eps);
    for (uint d = lid; d < dim; d += uint(PIE_GROUP_X)) {
        y[base + d] = bf16_at(x, base + d) * inv;
    }
}

#elif defined(PIE_GATES)

[[vk::binding(0)]] StructuredBuffer<float> mixes;
[[vk::binding(1)]] StructuredBuffer<float> scale;
[[vk::binding(2)]] StructuredBuffer<float> base;
[[vk::binding(3)]] StructuredBuffer<uint> residual;
[[vk::binding(4)]] RWStructuredBuffer<float> post_mix;
[[vk::binding(5)]] RWStructuredBuffer<float> comb_mix;
[[vk::binding(6)]] RWStructuredBuffer<uint> layer_input;

struct Params {
    int m;
    int h;
    float hc_eps;
    float alpha;
    int sinkhorn;
};
PIE_PARAMS(7, Params, params);

groupshared float pre[HC_MAX_MULT];
groupshared float comb[HC_MAX_MULT * HC_MAX_MULT];

// Each of the first `m` lanes divides its row (`by_row`) or column of `comb`
// by that row's or column's sum plus `eps`.
void normalize(uint tid, uint m, bool by_row) {
    if (tid < m) {
        float sum = params.hc_eps;
        for (uint k = 0u; k < m; k++) {
            sum += by_row ? comb[tid * m + k] : comb[k * m + tid];
        }
        for (uint k = 0u; k < m; k++) {
            uint at = by_row ? tid * m + k : k * m + tid;
            comb[at] = comb[at] / sum;
        }
    }
    GroupMemoryBarrierWithGroupSync();
}

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint n = group.x;
    uint tid = local.x;
    uint m = uint(max(params.m, 0));
    uint h = uint(max(params.h, 0));
    uint row = n * (2u * m + m * m);

    if (tid < m) {
        pre[tid] = sigmoid(mixes[row + tid] * scale[0] + base[tid]) + params.hc_eps;
        post_mix[n * m + tid] = sigmoid(mixes[row + m + tid] * scale[1] + base[m + tid]) * params.alpha;
    }
    if (tid < m * m) {
        comb[tid] = mixes[row + 2u * m + tid] * scale[2] + base[2u * m + tid];
    }
    GroupMemoryBarrierWithGroupSync();

    // A softmax along each row, then Sinkhorn: columns, then rows and
    // columns `sinkhorn - 1` more times.
    if (tid < m) {
        float top = -1e30;
        for (uint j = 0u; j < m; j++) {
            top = max(top, comb[tid * m + j]);
        }
        float sum = 0.0;
        for (uint j = 0u; j < m; j++) {
            float e = exp(comb[tid * m + j] - top);
            comb[tid * m + j] = e;
            sum += e;
        }
        for (uint j = 0u; j < m; j++) {
            comb[tid * m + j] = comb[tid * m + j] / sum + params.hc_eps;
        }
    }
    GroupMemoryBarrierWithGroupSync();
    normalize(tid, m, false);
    for (int iter = 0; iter < params.sinkhorn - 1; iter++) {
        normalize(tid, m, true);
        normalize(tid, m, false);
    }

    if (tid < m * m) {
        comb_mix[n * m * m + tid] = comb[tid];
    }

    uint res = n * m * h;
    for (uint k = 2u * tid; k < h; k += 2u * uint(PIE_GROUP_X)) {
        float2 acc = float2(0.0, 0.0);
        for (uint i = 0u; i < m; i++) {
            acc += pre[i] * word_at(residual, res + i * h + k);
        }
        layer_input[(n * h + k) >> 1] = pack_bf16(acc.x, acc.y);
    }
}

#elif defined(PIE_FOLD)

[[vk::binding(0)]] StructuredBuffer<uint> x;
[[vk::binding(1)]] StructuredBuffer<uint> residual;
[[vk::binding(2)]] StructuredBuffer<float> post_mix;
[[vk::binding(3)]] StructuredBuffer<float> comb_mix;
[[vk::binding(4)]] RWStructuredBuffer<uint> y;

struct Params {
    int n_rows;
    int m;
    int h;
};
PIE_PARAMS(5, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint m = uint(max(params.m, 0));
    uint h = uint(max(params.h, 0));
    uint k = 2u * gid.x;
    uint n = gid.y;
    if (m > HC_MAX_MULT || k >= h || n >= uint(max(params.n_rows, 0))) {
        return;
    }
    float2 xv = word_at(x, n * h + k);
    uint res = n * m * h + k;
    float2 r[HC_MAX_MULT];
    for (uint i = 0u; i < m; i++) {
        r[i] = word_at(residual, res + i * h);
    }
    for (uint j = 0u; j < m; j++) {
        float2 acc = post_mix[n * m + j] * xv;
        for (uint i = 0u; i < m; i++) {
            acc += comb_mix[n * m * m + i * m + j] * r[i];
        }
        y[(res + j * h) >> 1] = pack_bf16(acc.x, acc.y);
    }
}

#elif defined(PIE_COLLAPSE)

[[vk::binding(0)]] StructuredBuffer<float> mixes;
[[vk::binding(1)]] StructuredBuffer<float> scale;
[[vk::binding(2)]] StructuredBuffer<float> base;
[[vk::binding(3)]] StructuredBuffer<uint> residual;
[[vk::binding(4)]] RWStructuredBuffer<uint> y;

struct Params {
    int m;
    int h;
    float hc_eps;
};
PIE_PARAMS(5, Params, params);

groupshared float gates[HC_MAX_MULT];

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint n = group.x;
    uint tid = local.x;
    uint m = uint(max(params.m, 0));
    uint h = uint(max(params.h, 0));
    if (tid < m) {
        gates[tid] = sigmoid(mixes[n * m + tid] * scale[0] + base[tid]) + params.hc_eps;
    }
    GroupMemoryBarrierWithGroupSync();
    uint res = n * m * h;
    for (uint k = 2u * tid; k < h; k += 2u * uint(PIE_GROUP_X)) {
        float2 acc = float2(0.0, 0.0);
        for (uint i = 0u; i < m; i++) {
            acc += gates[i] * word_at(residual, res + i * h + k);
        }
        y[(n * h + k) >> 1] = pack_bf16(acc.x, acc.y);
    }
}

#elif defined(PIE_MIX)

[[vk::binding(0)]] StructuredBuffer<uint> gates;
[[vk::binding(1)]] StructuredBuffer<uint> normed;
[[vk::binding(2)]] RWStructuredBuffer<uint> y;

struct Params {
    int n_rows;
    int m;
    int h;
};
PIE_PARAMS(3, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint m = uint(max(params.m, 0));
    uint h = uint(max(params.h, 0));
    uint k = 2u * gid.x;
    uint n = gid.y;
    if (k >= h || n >= uint(max(params.n_rows, 0))) {
        return;
    }
    uint at = n * m * h + k;
    float2 acc = float2(0.0, 0.0);
    for (uint s = 0u; s < m; s++) {
        float2 g = word_at(gates, at + s * h);
        acc += word_at(normed, at + s * h) * float2(sigmoid(g.x), sigmoid(g.y));
    }
    acc /= float(max(m, 1u));
    y[(n * h + k) >> 1] = pack_bf16(acc.x, acc.y);
}

#elif defined(PIE_INJECT)

[[vk::binding(0)]] StructuredBuffer<uint> o;
[[vk::binding(1)]] StructuredBuffer<uint> gates;
[[vk::binding(2)]] RWStructuredBuffer<uint> hyper;

struct Params {
    int n_rows;
    int m;
    int h;
};
PIE_PARAMS(3, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint m = uint(max(params.m, 0));
    uint h = uint(max(params.h, 0));
    uint k = 2u * gid.x;
    uint n = gid.y;
    if (m > HC_MAX_MULT || k >= h || n >= uint(max(params.n_rows, 0))) {
        return;
    }
    float2 ov = word_at(o, n * h + k);
    for (uint s = 0u; s < m; s++) {
        float g = 2.0 * sigmoid(bf16_at(gates, n * m + s) / float(m));
        uint at = ((n * m + s) * h + k) >> 1;
        uint word = hyper[at];
        hyper[at] = pack_bf16(bf16_to_f32(word & 0xffffu) + g * ov.x, bf16_to_f32(word >> 16) + g * ov.y);
    }
}

#elif defined(PIE_PLE_GATE)

[[vk::binding(0)]] StructuredBuffer<uint> key;
[[vk::binding(1)]] StructuredBuffer<uint> query;
[[vk::binding(2)]] StructuredBuffer<uint> value;
[[vk::binding(3)]] RWStructuredBuffer<uint> y;

struct Params {
    int m;
    int h;
};
PIE_PARAMS(4, Params, params);

// One workgroup a (row, stream): the stream scales the row's value by
// `sigmoid(sign(d) * sqrt(|d|))`, `d` its key-query dot over `sqrt(h)`.
[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint m = uint(max(params.m, 1));
    uint h = uint(max(params.h, 0));
    uint n = group.x / m;
    uint stream = group.x * h;
    uint lid = local.x;
    float acc = 0.0;
    for (uint i = lid; i < h; i += uint(PIE_GROUP_X)) {
        acc += bf16_at(key, stream + i) * bf16_at(query, stream + i);
    }
    float d = pie_workgroup_sum(acc, lid) * rsqrt(float(max(h, 1u)));
    float magnitude = sqrt(max(abs(d), 1e-6));
    float gate = sigmoid(d > 0.0 ? magnitude : (d < 0.0 ? -magnitude : 0.0));
    for (uint i = 2u * lid; i < h; i += 2u * uint(PIE_GROUP_X)) {
        float2 v = gate * word_at(value, n * h + i);
        y[(stream + i) >> 1] = pack_bf16(v.x, v.y);
    }
}

#else
#error "elemwise/hc.slang: no kernel selected"
#endif

// pie:instantiate elemwise_hc_expand_bf16 PIE_EXPAND=1 PIE_GROUP_X=256
// pie:instantiate elemwise_hc_rmsnorm_f32_bf16 PIE_RMSNORM=1 PIE_GROUP_X=256
// pie:instantiate elemwise_hc_gates_bf16 PIE_GATES=1 PIE_GROUP_X=256
// pie:instantiate elemwise_hc_fold_bf16 PIE_FOLD=1 PIE_GROUP_X=256
// pie:instantiate elemwise_hc_collapse_bf16 PIE_COLLAPSE=1 PIE_GROUP_X=256
// pie:instantiate elemwise_hc_mix_bf16 PIE_MIX=1 PIE_GROUP_X=256
// pie:instantiate elemwise_hc_inject_bf16 PIE_INJECT=1 PIE_GROUP_X=256
// pie:instantiate elemwise_ple_gate_bf16 PIE_PLE_GATE=1 PIE_GROUP_X=256
