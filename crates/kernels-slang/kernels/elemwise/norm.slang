#include "prim/bf16.slang"
#include "prim/pair.slang"
#include "prim/params.slang"
#include "prim/reduce.slang"

// The norms, one workgroup a normed run: RMS (with a weight, a grouped
// weight, or none), gated RMS, layer norm, the residual blend's RMS-normed
// logits, and the per-column standardize. `PIE_<KERNEL>` picks the kernel.

#if defined(PIE_RMS)

// RMS norm over a row's `axis_size`-wide runs, one workgroup a run:
// `y = gain * x / rms(x) * (plus_one ? 1 + w : w)`, the weight read every
// `w_stride` (1: a bank as wide as the run; 0: one scalar). Grouped, each run
// reads its own slice of a bank as wide as the row; unscaled
// (`PIE_UNSCALED`), there is no weight at all. Rows land two values a word,
// so a run is an even number wide.

#ifndef N_READS
#define N_READS 4
#endif

[[vk::binding(0)]] StructuredBuffer<uint> x;
#ifdef PIE_UNSCALED
[[vk::binding(1)]] RWStructuredBuffer<uint> y;

struct Params {
    float eps;
    int axis_size;
};
PIE_PARAMS(2, Params, params);

float gain_at(uint i, uint plane) {
    return 1.0;
}
#else
[[vk::binding(1)]] StructuredBuffer<uint> w;
[[vk::binding(2)]] RWStructuredBuffer<uint> y;

struct Params {
    float eps;
    int axis_size;
    uint w_stride;
    uint plus_one;
    float gain;
#ifdef PIE_GROUPED
    uint groups;
#endif
};
PIE_PARAMS(3, Params, params);

float gain_at(uint i, uint plane) {
    float wv = bf16_at(w, plane + params.w_stride * i);
    return params.gain * (params.plus_one != 0u ? 1.0 + wv : wv);
}
#endif

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint axis_size = uint(max(params.axis_size, 0));
    uint base = group.x * axis_size;
#ifdef PIE_GROUPED
    uint plane = (group.x % max(params.groups, 1u)) * axis_size * params.w_stride;
#else
    uint plane = 0u;
#endif
    uint span = uint(PIE_GROUP_X) * uint(N_READS);
    uint lid = local.x;

    float acc = 0.0;
    for (uint start = lid * uint(N_READS); start < axis_size; start += span) {
        for (uint i = 0u; i < uint(N_READS); i++) {
            if (start + i < axis_size) {
                float xi = bf16_at(x, base + start + i);
                acc += xi * xi;
            }
        }
    }
    float inv = pie_inv_rms(acc, lid, axis_size, params.eps);

    for (uint start = lid * uint(N_READS); start < axis_size; start += span) {
        for (uint i = 0u; i < uint(N_READS); i += 2u) {
            uint at = start + i;
            if (at < axis_size) {
                uint word = x[(base + at) >> 1];
                float lo = gain_at(at, plane) * (bf16_to_f32(word & 0xffffu) * inv);
                float hi = gain_at(at + 1u, plane) * (bf16_to_f32(word >> 16) * inv);
                y[(base + at) >> 1] = pack_bf16(lo, hi);
            }
        }
    }
}

#elif defined(PIE_GATED_RMS)

// A gated RMS norm over f32 rows, one workgroup a `vd`-wide head:
// `y = x / rms(x) * w * gate(z)`, the gate `z * sigmoid(z)` (SiLU) or, with
// `PIE_SIGMOID`, `sigmoid(z)`. The rows and the weight are f32, the gate and
// the landing bf16 two a word, so a head is an even number wide.

[[vk::binding(0)]] StructuredBuffer<float> x;
[[vk::binding(1)]] StructuredBuffer<uint> z;
[[vk::binding(2)]] StructuredBuffer<float> w;
[[vk::binding(3)]] RWStructuredBuffer<uint> y;

struct Params {
    float eps;
    int vd;
};
PIE_PARAMS(4, Params, params);

float gate(float zr) {
    float e = 1.0 / (1.0 + exp(-abs(zr)));
    float sig = zr < 0.0 ? 1.0 - e : e;
#ifdef PIE_SIGMOID
    return sig;
#else
    return zr * sig;
#endif
}

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint vd = uint(max(params.vd, 0));
    uint base = group.x * vd;
    uint lid = local.x;
    float acc = 0.0;
    for (uint i = lid; i < vd; i += uint(PIE_GROUP_X)) {
        float xi = x[base + i];
        acc += xi * xi;
    }
    float inv = pie_inv_rms(acc, lid, vd, params.eps);
    for (uint i = 2u * lid; i < vd; i += 2u * uint(PIE_GROUP_X)) {
        uint word = z[(base + i) >> 1];
        float lo = (x[base + i] * inv) * w[i] * gate(bf16_to_f32(word & 0xffffu));
        float hi = (x[base + i + 1u] * inv) * w[i + 1u] * gate(bf16_to_f32(word >> 16));
        y[(base + i) >> 1] = pack_bf16(lo, hi);
    }
}

#elif defined(PIE_LAYERNORM)

// Layer norm over whole rows, one workgroup a row and a pair of elements a
// lane: `y = (x - mean) / sqrt(var + eps) * w + b`, or with `PIE_UNSCALED`
// no weight and no bias. The weight and the bias ride the row's element.

[[vk::binding(0)]] StructuredBuffer<PIE_ACT> x;
#ifdef PIE_UNSCALED
[[vk::binding(1)]] RWStructuredBuffer<PIE_ACT> y;
#define PIE_PARAMS_AT 2
#else
[[vk::binding(1)]] StructuredBuffer<PIE_ACT> w;
[[vk::binding(2)]] StructuredBuffer<PIE_ACT> b;
[[vk::binding(3)]] RWStructuredBuffer<PIE_ACT> y;
#define PIE_PARAMS_AT 4
#endif

struct Params {
    float eps;
    int axis_size;
};
PIE_PARAMS(PIE_PARAMS_AT, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint n = uint(max(params.axis_size, 0));
    uint base = group.x * n;
    uint lid = local.x;
    uint stride = 2u * uint(PIE_GROUP_X);

    float acc = 0.0;
    for (uint c = 2u * lid; c < n; c += stride) {
        float2 v = pair_at(x, base + c, c + 1u < n);
        acc += v.x + v.y;
    }
    float mean = pie_workgroup_sum(acc, lid) / float(n);

    float spread = 0.0;
    for (uint c = 2u * lid; c < n; c += stride) {
        bool both = c + 1u < n;
        float2 d = pair_at(x, base + c, both) - mean;
        spread += d.x * d.x + (both ? d.y * d.y : 0.0);
    }
    float inv = rsqrt(pie_workgroup_sum(spread, lid) / float(n) + params.eps);

    for (uint c = 2u * lid; c < n; c += stride) {
        bool both = c + 1u < n;
        float2 v = (pair_at(x, base + c, both) - mean) * inv;
#ifndef PIE_UNSCALED
        v = v * pair_at(w, c, both) + pair_at(b, c, both);
#endif
        pair_put(y, base + c, both, v);
    }
}

#elif defined(PIE_RES_BLEND)

// A residual blend, one workgroup a row and a pair of elements a lane: each
// candidate (the stacked blocks, then the prefix) is RMS-normed, weighted and
// dotted with a projection to a logit; the row is the softmax-weighted sum of
// the raw candidates. The blocks sit as stacked planes,
// `blocks[(j * rows + row) * hidden]`; rows are bf16 and an even width.

#define PIE_MAX_BLOCKS 32

[[vk::binding(0)]] StructuredBuffer<uint> prefix;
[[vk::binding(1)]] StructuredBuffer<uint> blocks;
[[vk::binding(2)]] StructuredBuffer<uint> norm_w;
[[vk::binding(3)]] StructuredBuffer<uint> proj_w;
[[vk::binding(4)]] RWStructuredBuffer<uint> y;

struct Params {
    int blocks_n;
    int hidden;
    int rows;
    float eps;
};
PIE_PARAMS(5, Params, params);

groupshared float pie_prob[PIE_MAX_BLOCKS + 1];

// Candidate `j`'s pair at column `c` of `row`.
float2 candidate(uint j, uint n_blocks, uint row, uint c) {
    uint hidden = uint(max(params.hidden, 0));
    if (j < n_blocks) {
        return pair_at(blocks, (j * uint(max(params.rows, 0)) + row) * hidden + c, true);
    }
    return pair_at(prefix, row * hidden + c, true);
}

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 group : SV_GroupID, uint3 local : SV_GroupThreadID) {
    uint row = group.x;
    uint hidden = uint(max(params.hidden, 0));
    uint n_blocks = uint(max(params.blocks_n, 0));
    uint cands = n_blocks + 1u;
    uint lid = local.x;
    uint stride = 2u * uint(PIE_GROUP_X);

    for (uint j = 0u; j < cands; j++) {
        float ss = 0.0;
        for (uint c = 2u * lid; c < hidden; c += stride) {
            float2 v = candidate(j, n_blocks, row, c);
            ss += dot(v, v);
        }
        float inv = pie_inv_rms(ss, lid, hidden, params.eps);

        float logit = 0.0;
        for (uint c = 2u * lid; c < hidden; c += stride) {
            float2 v = candidate(j, n_blocks, row, c) * inv;
            logit += dot(v * pair_at(norm_w, c, true), pair_at(proj_w, c, true));
        }
        logit = pie_workgroup_sum(logit, lid);
        if (lid == 0u) {
            pie_prob[j] = logit;
        }
    }

    if (lid == 0u) {
        float m = pie_prob[0];
        for (uint j = 1u; j < cands; j++) {
            m = max(m, pie_prob[j]);
        }
        float sum = 0.0;
        for (uint j = 0u; j < cands; j++) {
            pie_prob[j] = exp(pie_prob[j] - m);
            sum += pie_prob[j];
        }
        for (uint j = 0u; j < cands; j++) {
            pie_prob[j] /= sum;
        }
    }
    GroupMemoryBarrierWithGroupSync();

    for (uint c = 2u * lid; c < hidden; c += stride) {
        float2 acc = float2(0.0, 0.0);
        for (uint j = 0u; j < cands; j++) {
            acc += pie_prob[j] * candidate(j, n_blocks, row, c);
        }
        pair_put(y, row * hidden + c, true, acc);
    }
}

#elif defined(PIE_STANDARDIZE)

// `out[r, c] = (out[r, c] - bias[c]) * scale[c]` in place, a pair of columns
// a lane and a row a grid row. The planes ride the row's element.

[[vk::binding(0)]] RWStructuredBuffer<PIE_ACT> out_;
[[vk::binding(1)]] StructuredBuffer<PIE_ACT> bias;
[[vk::binding(2)]] StructuredBuffer<PIE_ACT> scale;

struct Params {
    int width;
    int rows;
};
PIE_PARAMS(3, Params, params);

[shader("compute")]
[numthreads(PIE_GROUP_X, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    uint width = uint(max(params.width, 0));
    uint c = 2u * gid.x;
    if (c >= width || gid.y >= uint(max(params.rows, 0))) {
        return;
    }
    bool both = c + 1u < width;
    uint i = gid.y * width + c;
    float2 v = (pair_at(out_, i, both) - pair_at(bias, c, both)) * pair_at(scale, c, both);
    pair_put(out_, i, both, v);
}

#else
#error "elemwise/norm.slang: no kernel selected"
#endif

// pie:instantiate elemwise_norm_rms_bf16 PIE_RMS=1 PIE_GROUP_X=256 N_READS=4
// pie:instantiate elemwise_norm_rms_grouped_bf16 PIE_RMS=1 PIE_GROUP_X=256 N_READS=4 PIE_GROUPED=1
// pie:instantiate elemwise_norm_rms_unscaled_bf16 PIE_RMS=1 PIE_GROUP_X=256 N_READS=4 PIE_UNSCALED=1
// pie:instantiate elemwise_norm_gated_rms_bf16 PIE_GATED_RMS=1 PIE_GROUP_X=256
// pie:instantiate elemwise_norm_gated_rms_sigmoid_bf16 PIE_GATED_RMS=1 PIE_GROUP_X=256 PIE_SIGMOID=1
// pie:instantiate elemwise_norm_layernorm_bf16 PIE_LAYERNORM=1 PIE_GROUP_X=256
// pie:instantiate elemwise_norm_layernorm_unscaled_bf16 PIE_LAYERNORM=1 PIE_GROUP_X=256 PIE_UNSCALED=1
// pie:instantiate elemwise_norm_layernorm_unscaled_f32 PIE_LAYERNORM=1 PIE_GROUP_X=256 PIE_UNSCALED=1 PIE_ACT_F32=1
// pie:instantiate elemwise_norm_res_blend_bf16 PIE_RES_BLEND=1 PIE_GROUP_X=256
// pie:instantiate elemwise_norm_standardize_bf16 PIE_STANDARDIZE=1 PIE_GROUP_X=256
