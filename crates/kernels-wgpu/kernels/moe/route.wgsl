//#include "common/bf16.inc.wgsl"

const ROUTER_MAX_TOPK = 16u;
const ROUTER_MAX_EXPERTS = 1024u;

//#if defined(PIE_ROUTER_TOPK)

@group(0) @binding(0) var<storage, read> logits: array<u32>;
@group(0) @binding(1) var<storage, read_write> expert_ids: array<i32>;
@group(0) @binding(2) var<storage, read_write> expert_weights: array<f32>;

@group(0) @binding(3) var<storage, read> per_expert_scale: array<u32>;

struct Params {
    n_experts: u32,
    experts_per_token: u32,
    softmax_over_all: u32,
    logits_pitch: u32,
}
@group(0) @binding(4) var<uniform> params: Params;

var<workgroup> s_logits: array<f32, ROUTER_MAX_EXPERTS>;
var<workgroup> chosen: array<f32, ROUTER_MAX_TOPK>;
var<workgroup> chosen_i: array<u32, ROUTER_MAX_TOPK>;

@compute @workgroup_size(PIE_GROUP_X)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let lid = local.x;
    let row = group.y;
    let n = min(params.n_experts, ROUTER_MAX_EXPERTS);
    let k = min(params.experts_per_token, ROUTER_MAX_TOPK);
    let pitch = select(n, params.logits_pitch, params.logits_pitch != 0u);
    let neg = -3.0e38;

    for (var e = lid; e < ROUTER_MAX_EXPERTS; e = e + u32(PIE_GROUP_X)) {
        var v = neg;
        if (e < n) {
            let at = row * pitch + e;
            v = pie_bf16_at(logits[at >> 1u], at);
        }
        s_logits[e] = v;
    }
    workgroupBarrier();

    if (lid == 0u) {
        var all_max = neg;
        for (var e = 0u; e < n; e = e + 1u) {
            all_max = max(all_max, s_logits[e]);
        }
        var all_sum = 0.0;
        for (var e = 0u; e < n; e = e + 1u) {
            all_sum = all_sum + exp(s_logits[e] - all_max);
        }

        for (var r = 0u; r < k; r = r + 1u) {
            var best = neg;
            var best_i = 0xffffffffu;
            for (var e = 0u; e < n; e = e + 1u) {
                let v = s_logits[e];
                if (v > best) {
                    best = v;
                    best_i = e;
                }
            }
            chosen[r] = best;
            chosen_i[r] = best_i;
            if (best_i < n) {
                s_logits[best_i] = neg;
            }
        }

        var mx = all_max;
        var sum = all_sum;
        if (params.softmax_over_all == 0u) {
            mx = neg;
            for (var r = 0u; r < k; r = r + 1u) {
                mx = max(mx, chosen[r]);
            }
            sum = 0.0;
            for (var r = 0u; r < k; r = r + 1u) {
                sum = sum + exp(chosen[r] - mx);
            }
        }
        for (var r = 0u; r < k; r = r + 1u) {
            let e = chosen_i[r];
            var w = exp(chosen[r] - mx) / sum;
//#if defined(PIE_SCALED)
            w = w * pie_bf16_at(per_expert_scale[e >> 1u], e);
//#endif
            expert_ids[row * k + r] = i32(e);
            expert_weights[row * k + r] = w;
        }
    }
}

//#elif defined(PIE_ROUTER_SIGMOID)

@group(0) @binding(0) var<storage, read> logits: array<u32>;
//#if defined(PIE_BIASED)
@group(0) @binding(1) var<storage, read> correction: array<f32>;
@group(0) @binding(2) var<storage, read_write> expert_ids: array<i32>;
@group(0) @binding(3) var<storage, read_write> expert_weights: array<f32>;

struct Params {
    n_experts: u32,
    experts_per_token: u32,
    renormalize: u32,
    scaling: f32,
}
@group(0) @binding(4) var<uniform> params: Params;
//#else
@group(0) @binding(1) var<storage, read_write> expert_ids: array<i32>;
@group(0) @binding(2) var<storage, read_write> expert_weights: array<f32>;

struct Params {
    n_experts: u32,
    experts_per_token: u32,
    renormalize: u32,
    scaling: f32,
}
@group(0) @binding(3) var<uniform> params: Params;
//#endif

var<workgroup> s_score: array<f32, ROUTER_MAX_EXPERTS>;
var<workgroup> s_rank: array<f32, ROUTER_MAX_EXPERTS>;

fn score_of(x: f32) -> f32 {
//#if defined(PIE_SQRT_SOFTPLUS)
    var sp = x;
    if (x <= 20.0) {
        sp = log(1.0 + exp(x));
    }
    return sqrt(max(sp, 0.0));
//#else
    return 1.0 / (1.0 + exp(-x));
//#endif
}

@compute @workgroup_size(PIE_GROUP_X)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let lid = local.x;
    let row = group.y;
    let n = min(params.n_experts, ROUTER_MAX_EXPERTS);
    let k = min(params.experts_per_token, ROUTER_MAX_TOPK);
    let picks = min(k, n);
    let neg = -3.0e38;

    for (var e = lid; e < ROUTER_MAX_EXPERTS; e = e + u32(PIE_GROUP_X)) {
        var s = neg;
        var rank = neg;
        if (e < n) {
            let at = row * n + e;
            s = score_of(pie_bf16_at(logits[at >> 1u], at));
//#if defined(PIE_BIASED)
            rank = s + correction[e];
//#else
            rank = s;
//#endif
        }
        s_score[e] = s;
        s_rank[e] = rank;
    }
    workgroupBarrier();

    if (lid == 0u) {
        var sum = 0.0;
        for (var r = 0u; r < picks; r = r + 1u) {
            var best = neg;
            var best_i = 0u;
            for (var e = 0u; e < n; e = e + 1u) {
                let v = s_rank[e];
                if (v > best) {
                    best = v;
                    best_i = e;
                }
            }
            s_rank[best_i] = neg;
            let w = s_score[best_i];
            expert_ids[row * k + r] = i32(best_i);
            expert_weights[row * k + r] = w;
            sum = sum + w;
        }
        for (var r = picks; r < k; r = r + 1u) {
            expert_ids[row * k + r] = 0;
            expert_weights[row * k + r] = 0.0;
        }
        var scale = params.scaling;
        if (params.renormalize != 0u && sum > 0.0) {
            scale = params.scaling / sum;
        }
        for (var r = 0u; r < k; r = r + 1u) {
            expert_weights[row * k + r] = expert_weights[row * k + r] * scale;
        }
    }
}

//#elif defined(PIE_EXPERT_COMBINE)

@group(0) @binding(0) var<storage, read> routed: array<u32>;
@group(0) @binding(1) var<storage, read> expert_weights: array<f32>;
@group(0) @binding(2) var<storage, read_write> out_: array<u32>;

struct Params {
    width: i32,
    rows: i32,
    experts_per_token: i32,
}
@group(0) @binding(3) var<uniform> params: Params;

fn combined(i: u32) -> f32 {
    let width = u32(params.width);
    let top_k = u32(params.experts_per_token);
    let row = i / width;
    let c = i - row * width;
    let base = row * top_k;
    var acc = 0.0;
    for (var e = 0u; e < top_k; e = e + 1u) {
        let at = (base + e) * width + c;
        acc = acc + expert_weights[base + e] * pie_bf16_at(routed[at >> 1u], at);
    }
    return acc;
}

@compute @workgroup_size(PIE_GROUP_X)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = u32(params.width) * u32(params.rows);
    let lo = gid.x * 2u;
    if (lo >= n) {
        return;
    }
    let v0 = combined(lo);
    if (lo + 1u < n) {
        out_[gid.x] = pie_pack_bf16(v0, combined(lo + 1u));
    } else {
        out_[gid.x] = pie_bf16_into(out_[gid.x], lo, v0);
    }
}

//#elif defined(PIE_EXPERT_BIAS_COMBINE)

@group(0) @binding(0) var<storage, read> x: array<u32>;
@group(0) @binding(1) var<storage, read> bias: array<u32>;
@group(0) @binding(2) var<storage, read> expert_ids: array<i32>;
@group(0) @binding(3) var<storage, read> expert_weights: array<f32>;
@group(0) @binding(4) var<storage, read_write> out_: array<u32>;

struct Params {
    width: i32,
    rows: i32,
    experts_per_token: i32,
}
@group(0) @binding(5) var<uniform> params: Params;

fn combined(i: u32) -> f32 {
    let width = u32(params.width);
    let top_k = u32(params.experts_per_token);
    let row = i / width;
    let c = i - row * width;
    let base = row * top_k;
    var acc = pie_bf16_at(x[i >> 1u], i);
    for (var e = 0u; e < top_k; e = e + 1u) {
        let expert = expert_ids[base + e];
        if (expert >= 0) {
            let at = u32(expert) * width + c;
            acc = acc + expert_weights[base + e] * pie_bf16_at(bias[at >> 1u], at);
        }
    }
    return acc;
}

@compute @workgroup_size(PIE_GROUP_X)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = u32(params.width) * u32(params.rows);
    let lo = gid.x * 2u;
    if (lo >= n) {
        return;
    }
    let v0 = combined(lo);
    if (lo + 1u < n) {
        out_[gid.x] = pie_pack_bf16(v0, combined(lo + 1u));
    } else {
        out_[gid.x] = pie_bf16_into(out_[gid.x], lo, v0);
    }
}

//#elif defined(PIE_HASH_ROUTE)

@group(0) @binding(0) var<storage, read> token_ids: array<u32>;
@group(0) @binding(1) var<storage, read> tid2eid: array<u32>;
@group(0) @binding(2) var<storage, read> logits: array<u32>;
@group(0) @binding(3) var<storage, read_write> expert_ids: array<i32>;
@group(0) @binding(4) var<storage, read_write> expert_weights: array<f32>;

struct Params {
    vocab: u32,
    n_experts: u32,
    experts_per_token: u32,
    renormalize: u32,
    scaling: f32,
    rows: u32,
}
@group(0) @binding(5) var<uniform> params: Params;

fn hash_sqrt_softplus(x: f32) -> f32 {
    var sp = x;
    if (x <= 20.0) {
        sp = log(1.0 + exp(x));
    }
    return sqrt(sp);
}

@compute @workgroup_size(PIE_GROUP_X)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let row = gid.x;
    if (row >= params.rows) {
        return;
    }
    let k = min(params.experts_per_token, ROUTER_MAX_TOPK);
    let raw = token_ids[row];
    var tid = 0u;
    if (params.vocab > 0u && raw < params.vocab) {
        tid = raw;
    }
    let picks = tid * params.experts_per_token;
    let score = row * params.n_experts;
    let base = row * params.experts_per_token;
    var sum = 0.0;
    for (var r = 0u; r < k; r = r + 1u) {
        let lo = tid2eid[2u * (picks + r)];
        let hi = tid2eid[2u * (picks + r) + 1u];
        var w = 0.0;

        if (hi == 0u && lo < params.n_experts) {
            let at = score + lo;
            w = hash_sqrt_softplus(pie_bf16_at(logits[at >> 1u], at));
        }
        expert_ids[base + r] = bitcast<i32>(lo);
        expert_weights[base + r] = w;
        sum = sum + w;
    }
    var scale = params.scaling;
    if (params.renormalize != 0u && sum > 0.0) {
        scale = params.scaling / sum;
    }
    for (var r = 0u; r < k; r = r + 1u) {
        expert_weights[base + r] = expert_weights[base + r] * scale;
    }
}

//#elif defined(PIE_GROUP_ROUTES)

@group(0) @binding(0) var<storage, read_write> routes: array<i32>;

struct Params {
    groups: u32,
}
@group(0) @binding(1) var<uniform> params: Params;

@compute @workgroup_size(PIE_GROUP_X)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let slot = gid.x;
    let row = gid.y;
    if (slot >= params.groups) {
        return;
    }
    routes[row * params.groups + slot] = i32(slot);
}

//#else

@group(0) @binding(0) var<storage, read> routed: array<u32>;
@group(0) @binding(1) var<storage, read> shared_: array<u32>;
@group(0) @binding(2) var<storage, read> gate: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_: array<u32>;

struct Params {
    width: i32,
    rows: i32,
}
@group(0) @binding(4) var<uniform> params: Params;

fn combined(i: u32) -> f32 {
    let row = i / u32(params.width);
    let g = 1.0 / (1.0 + exp(-pie_bf16_at(gate[row >> 1u], row)));
    return pie_bf16_at(routed[i >> 1u], i) + g * pie_bf16_at(shared_[i >> 1u], i);
}

@compute @workgroup_size(PIE_GROUP_X)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let n = u32(params.width) * u32(params.rows);
    let lo = gid.x * 2u;
    if (lo >= n) {
        return;
    }
    let v0 = combined(lo);
    if (lo + 1u < n) {
        out_[gid.x] = pie_pack_bf16(v0, combined(lo + 1u));
    } else {
        out_[gid.x] = pie_bf16_into(out_[gid.x], lo, v0);
    }
}
//#endif

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
