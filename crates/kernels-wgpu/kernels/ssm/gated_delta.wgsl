//#include "common/bf16.inc.wgsl"
//#include "common/reduce.inc.wgsl"

const PIE_KMAX = 256u;

//#if defined(PIE_COMMITTED)
@group(0) @binding(0) var<storage, read> qkv: array<u32>;
@group(0) @binding(1) var<storage, read> indptr: array<i32>;
@group(0) @binding(2) var<storage, read> replay: array<i32>;
@group(0) @binding(3) var<storage, read> commit: array<i32>;
@group(0) @binding(4) var<storage, read> slots: array<i32>;
@group(0) @binding(5) var<storage, read> gates: array<f32>;
@group(0) @binding(6) var<storage, read_write> rstate: array<f32>;
@group(0) @binding(7) var<storage, read_write> work: array<f32>;
@group(0) @binding(8) var<storage, read_write> y: array<f32>;

struct Params {
    lane0: i32,
    k_heads: i32,
    v_heads: i32,
    k_dim: i32,
    v_dim: i32,
}
@group(0) @binding(9) var<uniform> params: Params;

fn st_get(i: u32) -> f32 {
    return work[i];
}
fn st_set(i: u32, v: f32) {
    work[i] = v;
}
//#elif defined(PIE_CHUNKED)
@group(0) @binding(0) var<storage, read> qkv: array<u32>;
@group(0) @binding(1) var<storage, read> indptr: array<i32>;
@group(0) @binding(2) var<storage, read> gates: array<f32>;
@group(0) @binding(3) var<storage, read_write> rstate: array<f32>;
@group(0) @binding(4) var<storage, read> slots: array<u32>;
@group(0) @binding(5) var<storage, read_write> y: array<f32>;

struct Params {
    k_heads: i32,
    v_heads: i32,
    k_dim: i32,
    v_dim: i32,
}
@group(0) @binding(6) var<uniform> params: Params;
//#else
@group(0) @binding(0) var<storage, read> qkv: array<u32>;
@group(0) @binding(1) var<storage, read> gates: array<f32>;
@group(0) @binding(2) var<storage, read_write> rstate: array<f32>;
@group(0) @binding(3) var<storage, read> slots: array<u32>;
@group(0) @binding(4) var<storage, read_write> y: array<f32>;

struct Params {
    k_heads: i32,
    v_heads: i32,
    k_dim: i32,
    v_dim: i32,
}
@group(0) @binding(5) var<uniform> params: Params;
//#endif
//#if !defined(PIE_COMMITTED)
fn st_get(i: u32) -> f32 {
    return rstate[i];
}
fn st_set(i: u32, v: f32) {
    rstate[i] = v;
}
//#endif

var<workgroup> sq: array<f32, PIE_KMAX>;
var<workgroup> sk: array<f32, PIE_KMAX>;

//#if defined(PIE_DK_REG)

var<private> st_reg: array<f32, PIE_DK_REG>;
//#endif

fn load_qkv(i: u32) -> f32 {
    return pie_bf16_at(qkv[i >> 1u], i);
}

fn token(tid: u32, t: u32, hv: u32, hk: u32, state_base: u32) {
    let dk = u32(params.k_dim);
    let dv = u32(params.v_dim);
    let v_heads = u32(params.v_heads);
    let keys = u32(params.k_heads) * dk;
    let pitch = 2u * keys + v_heads * dv;
    let row = t * pitch;
    let qbase = row + hk * dk;
    let kbase = qbase + keys;
    let vbase = row + 2u * keys + hv * dv;

    var qsum = 0.0;
    var ksum = 0.0;
    for (var i = tid; i < dk; i = i + u32(PIE_GROUP_X)) {
        let qv = load_qkv(qbase + i);
        let kv = load_qkv(kbase + i);
        sq[i] = qv;
        sk[i] = kv;
        qsum = qsum + qv * qv;
        ksum = ksum + kv * kv;
    }
    let qtot = pie_workgroup_sum(tid, u32(PIE_GROUP_X), qsum);
    let ktot = pie_workgroup_sum(tid, u32(PIE_GROUP_X), ksum);
    let scale = 1.0 / sqrt(f32(dk));
    let qinv = inverseSqrt(qtot + 1e-6) * scale;
    let kinv = inverseSqrt(ktot + 1e-6);
    for (var i = tid; i < dk; i = i + u32(PIE_GROUP_X)) {
        sq[i] = sq[i] * qinv;
        sk[i] = sk[i] * kinv;
    }
    workgroupBarrier();

    let fused = t * 2u * v_heads + hv;
    let decay = exp(gates[fused]);
    let beta = gates[fused + v_heads];
    let out = (t * v_heads + hv) * dv;
//#if defined(PIE_DK_REG)
    if (tid < dv) {
        var kv_mem = 0.0;
        for (var i = 0u; i < dk; i = i + 1u) {
            let s = st_reg[i] * decay;
            st_reg[i] = s;
            kv_mem = kv_mem + s * sk[i];
        }
        let delta = (load_qkv(vbase + tid) - kv_mem) * beta;
        var acc = 0.0;
        for (var i = 0u; i < dk; i = i + 1u) {
            let s = st_reg[i] + sk[i] * delta;
            st_reg[i] = s;
            acc = acc + s * sq[i];
        }
        y[out + tid] = acc;
    }
//#else
    for (var c = tid; c < dv; c = c + u32(PIE_GROUP_X)) {
        let cell = state_base + c * dk;
        var kv_mem = 0.0;
        for (var i = 0u; i < dk; i = i + 1u) {
            let s = st_get(cell + i) * decay;
            st_set(cell + i, s);
            kv_mem = kv_mem + s * sk[i];
        }
        let delta = (load_qkv(vbase + c) - kv_mem) * beta;
        var acc = 0.0;
        for (var i = 0u; i < dk; i = i + 1u) {
            let s = st_get(cell + i) + sk[i] * delta;
            st_set(cell + i, s);
            acc = acc + s * sq[i];
        }
        y[out + c] = acc;
    }
//#endif
    workgroupBarrier();
}

@compute @workgroup_size(PIE_GROUP_X)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let tid = local.x;
    let hv = group.y;
    let v_heads = u32(params.v_heads);
    let hk = hv / (v_heads / u32(params.k_heads));
    let head = u32(params.v_dim) * u32(params.k_dim);
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
    let dk = u32(params.k_dim);
    let dv = u32(params.v_dim);
    let bank = (u32(slot) * v_heads + hv) * head;
    let state_base = ((lane0 + r) * v_heads + hv) * head;
//#if defined(PIE_DK_REG)

    if (tid < dv) {
        for (var i = 0u; i < dk; i = i + 1u) {
            st_reg[i] = rstate[bank + tid * dk + i];
        }
    }
    workgroupBarrier();
    for (var t = 0; t < span; t = t + 1) {
        token(tid, u32(begin + t), hv, hk, state_base);
        if (t + 1 == keep && tid < dv) {
            for (var i = 0u; i < dk; i = i + 1u) {
                rstate[bank + tid * dk + i] = st_reg[i];
            }
        }
    }
//#else

    for (var c = tid; c < dv; c = c + u32(PIE_GROUP_X)) {
        for (var i = 0u; i < dk; i = i + 1u) {
            work[state_base + c * dk + i] = rstate[bank + c * dk + i];
        }
    }
    workgroupBarrier();
    for (var t = 0; t < span; t = t + 1) {
        token(tid, u32(begin + t), hv, hk, state_base);
        if (t + 1 == keep) {
            for (var c = tid; c < dv; c = c + u32(PIE_GROUP_X)) {
                for (var i = 0u; i < dk; i = i + 1u) {
                    rstate[bank + c * dk + i] = work[state_base + c * dk + i];
                }
            }
        }
    }
//#endif
//#elif defined(PIE_CHUNKED)
    let r = group.z;
    let begin = indptr[r];
    let end = indptr[r + 1u];
    if (end <= begin) {
        return;
    }
    let state_base = (slots[u32(begin)] * v_heads + hv) * head;
//#if defined(PIE_DK_REG)
    let dvc = u32(params.v_dim);
    let dkc = u32(params.k_dim);
    if (tid < dvc) {
        for (var i = 0u; i < dkc; i = i + 1u) {
            st_reg[i] = rstate[state_base + tid * dkc + i];
        }
    }
    workgroupBarrier();
//#endif
    for (var t = begin; t < end; t = t + 1) {
        token(tid, u32(t), hv, hk, state_base);
    }
//#if defined(PIE_DK_REG)
    if (tid < dvc) {
        for (var i = 0u; i < dkc; i = i + 1u) {
            rstate[state_base + tid * dkc + i] = st_reg[i];
        }
    }
//#endif
//#else
    let n = group.z;
    let state_base = (slots[n] * v_heads + hv) * head;
//#if defined(PIE_DK_REG)
    let dvs = u32(params.v_dim);
    let dks = u32(params.k_dim);
    if (tid < dvs) {
        for (var i = 0u; i < dks; i = i + 1u) {
            st_reg[i] = rstate[state_base + tid * dks + i];
        }
    }
    workgroupBarrier();
//#endif
    token(tid, n, hv, hk, state_base);
//#if defined(PIE_DK_REG)
    if (tid < dvs) {
        for (var i = 0u; i < dks; i = i + 1u) {
            rstate[state_base + tid * dks + i] = st_reg[i];
        }
    }
//#endif
//#endif
}

// pie:instantiate gated_delta_bf16 PIE_GROUP_X=128
// pie:instantiate gated_delta_chunked_bf16 PIE_GROUP_X=128 PIE_CHUNKED=1
// pie:instantiate gated_delta_committed_bf16 PIE_GROUP_X=128 PIE_COMMITTED=1
// pie:instantiate gated_delta_r128_bf16 PIE_GROUP_X=128 PIE_DK_REG=128
// pie:instantiate gated_delta_chunked_r128_bf16 PIE_GROUP_X=128 PIE_CHUNKED=1 PIE_DK_REG=128
// pie:instantiate gated_delta_committed_r128_bf16 PIE_GROUP_X=128 PIE_COMMITTED=1 PIE_DK_REG=128
