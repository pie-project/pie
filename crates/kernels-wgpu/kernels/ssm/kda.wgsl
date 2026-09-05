//#include "common/bf16.inc.wgsl"
//#include "common/reduce.inc.wgsl"

const PIE_DMAX = 256u;

//#if defined(PIE_COMMITTED)
@group(0) @binding(0) var<storage, read> mixed: array<u32>;
@group(0) @binding(1) var<storage, read> indptr: array<i32>;
@group(0) @binding(2) var<storage, read> replay: array<i32>;
@group(0) @binding(3) var<storage, read> commit: array<i32>;
@group(0) @binding(4) var<storage, read> slots: array<i32>;
@group(0) @binding(5) var<storage, read> f: array<u32>;
@group(0) @binding(6) var<storage, read> b: array<u32>;
@group(0) @binding(7) var<storage, read> dt_bias: array<f32>;
@group(0) @binding(8) var<storage, read> a_log: array<f32>;
@group(0) @binding(9) var<storage, read_write> rstate: array<f32>;
@group(0) @binding(10) var<storage, read_write> work: array<f32>;
@group(0) @binding(11) var<storage, read_write> y: array<f32>;

struct Params {
    lane0: i32,
    heads: i32,
    head_dim: i32,
    norm_eps: f32,
    gate_floor: f32,
}
@group(0) @binding(12) var<uniform> params: Params;

fn st_get(i: u32) -> f32 {
    return work[i];
}
fn st_set(i: u32, v: f32) {
    work[i] = v;
}
//#elif defined(PIE_CHUNKED)
@group(0) @binding(0) var<storage, read> mixed: array<u32>;
@group(0) @binding(1) var<storage, read> indptr: array<i32>;
@group(0) @binding(2) var<storage, read> f: array<u32>;
@group(0) @binding(3) var<storage, read> b: array<u32>;
@group(0) @binding(4) var<storage, read> dt_bias: array<f32>;
@group(0) @binding(5) var<storage, read> a_log: array<f32>;
@group(0) @binding(6) var<storage, read_write> rstate: array<f32>;
@group(0) @binding(7) var<storage, read> slots: array<u32>;
@group(0) @binding(8) var<storage, read_write> y: array<f32>;

struct Params {
    heads: i32,
    head_dim: i32,
    norm_eps: f32,
    gate_floor: f32,
}
@group(0) @binding(9) var<uniform> params: Params;
//#else
@group(0) @binding(0) var<storage, read> mixed: array<u32>;
@group(0) @binding(1) var<storage, read> f: array<u32>;
@group(0) @binding(2) var<storage, read> b: array<u32>;
@group(0) @binding(3) var<storage, read> dt_bias: array<f32>;
@group(0) @binding(4) var<storage, read> a_log: array<f32>;
@group(0) @binding(5) var<storage, read_write> rstate: array<f32>;
@group(0) @binding(6) var<storage, read> slots: array<u32>;
@group(0) @binding(7) var<storage, read_write> y: array<f32>;

struct Params {
    heads: i32,
    head_dim: i32,
    norm_eps: f32,
    gate_floor: f32,
}
@group(0) @binding(8) var<uniform> params: Params;
//#endif
//#if !defined(PIE_COMMITTED)
fn st_get(i: u32) -> f32 {
    return rstate[i];
}
fn st_set(i: u32, v: f32) {
    rstate[i] = v;
}
//#endif

var<workgroup> sq: array<f32, PIE_DMAX>;
var<workgroup> sk: array<f32, PIE_DMAX>;
var<workgroup> sg: array<f32, PIE_DMAX>;

fn load_mixed(i: u32) -> f32 {
    return pie_bf16_at(mixed[i >> 1u], i);
}
fn load_f(i: u32) -> f32 {
    return pie_bf16_at(f[i >> 1u], i);
}
fn load_b(i: u32) -> f32 {
    return pie_bf16_at(b[i >> 1u], i);
}

fn token(tid: u32, t: u32, h: u32, state_base: u32) {
    let d = u32(params.head_dim);
    let wide = u32(params.heads) * d;
    let row = t * 3u * wide;
    let head = h * d;

    var qsum = 0.0;
    var ksum = 0.0;
    for (var i = tid; i < d; i = i + u32(PIE_GROUP_X)) {
        let qv = load_mixed(row + head + i);
        let kv = load_mixed(row + wide + head + i);
        qsum = qsum + qv * qv;
        ksum = ksum + kv * kv;
    }
    let qtot = pie_workgroup_sum(tid, u32(PIE_GROUP_X), qsum);
    let ktot = pie_workgroup_sum(tid, u32(PIE_GROUP_X), ksum);
    let qinv = inverseSqrt(qtot + params.norm_eps) * inverseSqrt(f32(d));
    let kinv = inverseSqrt(ktot + params.norm_eps);

    let alpha = exp(a_log[h]);
    for (var i = tid; i < d; i = i + u32(PIE_GROUP_X)) {
        let at = head + i;
        sq[i] = load_mixed(row + at) * qinv;
        sk[i] = load_mixed(row + wide + at) * kinv;
        let z = load_f(t * wide + at) + dt_bias[at];
        if (params.gate_floor != 0.0) {
            sg[i] = exp(params.gate_floor / (1.0 + exp(-alpha * z)));
        } else {
            let sp = select(log(1.0 + exp(z)), z, z > 20.0);
            sg[i] = exp(-alpha * sp);
        }
    }
    workgroupBarrier();

    let beta = 1.0 / (1.0 + exp(-load_b(t * u32(params.heads) + h)));
    let out = t * wide + head;
    let vbase = row + 2u * wide + head;
    for (var vi = tid; vi < d; vi = vi + u32(PIE_GROUP_X)) {
        let cell = state_base + vi * d;
        var mem = 0.0;
        for (var i = 0u; i < d; i = i + 1u) {
            let s = st_get(cell + i) * sg[i];
            st_set(cell + i, s);
            mem = mem + s * sk[i];
        }
        let delta = (load_mixed(vbase + vi) - mem) * beta;
        var acc = 0.0;
        for (var i = 0u; i < d; i = i + 1u) {
            let s = st_get(cell + i) + sk[i] * delta;
            st_set(cell + i, s);
            acc = acc + s * sq[i];
        }
        y[out + vi] = acc;
    }
    workgroupBarrier();
}

@compute @workgroup_size(PIE_GROUP_X)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let tid = local.x;
    let h = group.y;
    let d = u32(params.head_dim);
    let cells = d * d;
    let heads = u32(params.heads);
//#if defined(PIE_COMMITTED)
    let r = group.z;
    let lane0 = u32(params.lane0);
    var begin = indptr[r];
    for (var j = 0u; j < r; j = j + 1u) {
        begin = begin + replay[lane0 + j];
    }
    let span = (indptr[r + 1u] - indptr[r]) + replay[lane0 + r];
    if (span <= 0) {
        return;
    }
    let slot = slots[lane0 + r];
    if (slot < 0) {
        return;
    }
    let keep = min(commit[lane0 + r], span);
    let bank = (u32(slot) * heads + h) * cells;
    let state_base = ((lane0 + r) * heads + h) * cells;
    for (var i = tid; i < cells; i = i + u32(PIE_GROUP_X)) {
        work[state_base + i] = rstate[bank + i];
    }
    workgroupBarrier();
    for (var t = 0; t < span; t = t + 1) {
        token(tid, u32(begin + t), h, state_base);
        if (t + 1 == keep) {
            for (var i = tid; i < cells; i = i + u32(PIE_GROUP_X)) {
                rstate[bank + i] = work[state_base + i];
            }
            workgroupBarrier();
        }
    }
//#elif defined(PIE_CHUNKED)
    let r = group.z;
    let begin = indptr[r];
    let end = indptr[r + 1u];
    if (end <= begin) {
        return;
    }
    let state_base = (slots[u32(begin)] * heads + h) * cells;
    for (var t = begin; t < end; t = t + 1) {
        token(tid, u32(t), h, state_base);
    }
//#else
    let n = group.z;
    let state_base = (slots[n] * heads + h) * cells;
    token(tid, n, h, state_base);
//#endif
}

// pie:instantiate kda_step_bf16 PIE_GROUP_X=128
// pie:instantiate kda_chunked_bf16 PIE_GROUP_X=128 PIE_CHUNKED=1
// pie:instantiate kda_committed_bf16 PIE_GROUP_X=128 PIE_COMMITTED=1
