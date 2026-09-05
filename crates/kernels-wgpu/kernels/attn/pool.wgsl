//#include "common/bf16.inc.wgsl"
//#include "common/reduce.inc.wgsl"

fn pie_neg_inf() -> f32 {
    return bitcast<f32>(0xff800000u);
}

fn byte_at(word: u32, i: u32) -> u32 {
    return (word >> ((i & 3u) * 8u)) & 0xffu;
}

//#if defined(PIE_POOL_BOUNDARY)

@group(0) @binding(0) var<storage, read> positions: array<i32>;
//#if defined(PIE_PREFILL)
@group(0) @binding(1) var<storage, read> qo_indptr: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_pos: array<i32>;
@group(0) @binding(3) var<storage, read_write> out_req: array<i32>;
@group(0) @binding(4) var<storage, read_write> out_rope: array<i32>;

@group(0) @binding(5) var<storage, read> row_valid: array<u32>;
struct Params {
    n: i32,
    num_requests: i32,
    ratio: i32,
}
@group(0) @binding(6) var<uniform> params: Params;
//#else
@group(0) @binding(1) var<storage, read_write> out_pos: array<i32>;
@group(0) @binding(2) var<storage, read_write> out_req: array<i32>;
@group(0) @binding(3) var<storage, read_write> out_rope: array<i32>;
@group(0) @binding(4) var<storage, read> row_valid: array<u32>;
struct Params {
    n: i32,
    ratio: i32,
}
@group(0) @binding(5) var<uniform> params: Params;
//#endif

@compute @workgroup_size(PIE_GROUP_X, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let t = i32(gid.x);
    if (t >= params.n) {
        return;
    }
    let p = positions[u32(t)];
    let valid = byte_at(row_valid[u32(t) >> 2u], u32(t)) != 0u;
    let is_boundary = valid && (((p + 1) % params.ratio) == 0);
    out_pos[u32(t)] = select(-1, p, is_boundary);
    out_rope[u32(t)] = select(0, (p / params.ratio) * params.ratio, is_boundary);
//#if defined(PIE_PREFILL)
    var lo = 0;
    var hi = params.num_requests;
    loop {
        if (lo + 1 >= hi) {
            break;
        }
        let mid = lo + (hi - lo) / 2;
        if (i32(qo_indptr[u32(mid)]) <= t) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    out_req[u32(t)] = lo;
//#else
    out_req[u32(t)] = t;
//#endif
}

//#elif defined(PIE_POOL_STATE_WRITE)

@group(0) @binding(0) var<storage, read> kv: array<u32>;
@group(0) @binding(1) var<storage, read> score: array<u32>;
@group(0) @binding(2) var<storage, read_write> state_kv: array<u32>;
@group(0) @binding(3) var<storage, read_write> state_score: array<u32>;
@group(0) @binding(4) var<storage, read> w_page: array<u32>;
@group(0) @binding(5) var<storage, read> w_off: array<u32>;
struct Params {
    width: i32,
    page_size: i32,
    state_pitch: i32,
    rows: i32,
}
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(PIE_GROUP_X, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = i32(gid.x) * 2;
    let i = gid.y;
    if (d >= params.width || i >= u32(params.rows)) {
        return;
    }
    let page = i32(w_page[i]);
    let off = i32(w_off[i]);
    if (page < 0 || off < 0 || off >= params.page_size) {
        return;
    }
    let slot = u32(page) * u32(params.page_size) + u32(off);
    let dst = (slot * u32(params.state_pitch) + u32(d)) >> 1u;
    let src = (i * u32(params.width) + u32(d)) >> 1u;
    state_kv[dst] = kv[src];
    state_score[dst] = score[src];
}

//#elif defined(PIE_POOL_GATHER)

@group(0) @binding(0) var<storage, read> state_kv: array<u32>;
@group(0) @binding(1) var<storage, read> state_score: array<u32>;
@group(0) @binding(2) var<storage, read> ape: array<f32>;
@group(0) @binding(3) var<storage, read> boundary_pos: array<i32>;
@group(0) @binding(4) var<storage, read> boundary_req: array<i32>;
@group(0) @binding(5) var<storage, read> page_indices: array<u32>;
@group(0) @binding(6) var<storage, read> page_indptr: array<u32>;
@group(0) @binding(7) var<storage, read_write> out_: array<u32>;
struct Params {
    head_dim: i32,
    ratio: i32,
    coff: i32,
    page_size: i32,
    has_ape: i32,
    state_pitch: i32,
    rows: i32,
}
@group(0) @binding(8) var<uniform> params: Params;

fn pool_slot(req: i32, pos: i32) -> u32 {
    let page = page_indices[page_indptr[u32(req)] + u32(pos / params.page_size)];
    return page * u32(params.page_size) + u32(pos % params.page_size);
}

fn pooled(d: i32, bpos: i32, req: i32) -> f32 {
    let window = params.coff * params.ratio;
    let width = u32(params.coff * params.head_dim);
    let pitch = u32(params.state_pitch);
    var max_s = pie_neg_inf();
    for (var i = 0; i < window; i = i + 1) {
        let pos = bpos + i - (window - 1);
        if (pos < 0) {
            continue;
        }
        let col = u32(select(0, params.head_dim, i >= params.ratio) + d);
        let slot = pool_slot(req, pos);
        let e = slot * pitch + col;
        var sc = pie_bf16_at(state_score[e >> 1u], e);
        if (params.has_ape != 0) {
            sc = sc + ape[u32(pos % params.ratio) * width + col];
        }
        max_s = max(max_s, sc);
    }
    if (!(max_s > -3.0e38)) {
        return 0.0;
    }
    var sum_e = 0.0;
    var acc = 0.0;
    for (var i = 0; i < window; i = i + 1) {
        let pos = bpos + i - (window - 1);
        if (pos < 0) {
            continue;
        }
        let col = u32(select(0, params.head_dim, i >= params.ratio) + d);
        let slot = pool_slot(req, pos);
        let e = slot * pitch + col;
        var sc = pie_bf16_at(state_score[e >> 1u], e);
        if (params.has_ape != 0) {
            sc = sc + ape[u32(pos % params.ratio) * width + col];
        }
        let ex = exp(sc - max_s);
        sum_e = sum_e + ex;
        acc = acc + ex * pie_bf16_at(state_kv[e >> 1u], e);
    }
    if (sum_e > 0.0) {
        return acc / sum_e;
    }
    return 0.0;
}

@compute @workgroup_size(PIE_GROUP_X, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = i32(gid.x) * 2;
    let c = gid.y;
    if (d >= params.head_dim || c >= u32(params.rows)) {
        return;
    }
    let bpos = boundary_pos[c];
    let req = boundary_req[c];
    let at = (c * u32(params.head_dim) + u32(d)) >> 1u;
    if (bpos < 0) {
        out_[at] = 0u;
        return;
    }
    out_[at] = pie_pack_bf16(pooled(d, bpos, req), pooled(d + 1, bpos, req));
}

//#elif defined(PIE_POOL_STORE)

@group(0) @binding(0) var<storage, read> entries: array<u32>;
@group(0) @binding(1) var<storage, read_write> comp_kv_pages: array<u32>;
@group(0) @binding(2) var<storage, read> boundary_pos: array<i32>;
@group(0) @binding(3) var<storage, read> boundary_req: array<i32>;
@group(0) @binding(4) var<storage, read> page_indices: array<u32>;
@group(0) @binding(5) var<storage, read> page_indptr: array<u32>;
struct Params {
    head_dim: i32,
    page_size: i32,
    rows: i32,
}
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(PIE_GROUP_X, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = i32(gid.x) * 2;
    let c = gid.y;
    if (d >= params.head_dim || c >= u32(params.rows)) {
        return;
    }
    let bpos = boundary_pos[c];
    if (bpos < 0) {
        return;
    }
    let req = boundary_req[c];
    let page = page_indices[page_indptr[u32(req)] + u32(bpos / params.page_size)];
    let slot = page * u32(params.page_size) + u32(bpos % params.page_size);
    comp_kv_pages[(slot * u32(params.head_dim) + u32(d)) >> 1u] = entries[(c * u32(params.head_dim) + u32(d)) >> 1u];
}

//#else

const PIE_POOL_HEAD_MAX = 512u;
const PIE_PAIRS_PER_THREAD = PIE_POOL_HEAD_MAX / (2u * PIE_GROUP_X);
@group(0) @binding(0) var<storage, read> q: array<u32>;
@group(0) @binding(1) var<storage, read> comp_kv_pages: array<u32>;
//#if defined(PIE_SELECTED)
@group(0) @binding(2) var<storage, read> selection: array<i32>;
@group(0) @binding(3) var<storage, read_write> out_: array<u32>;
@group(0) @binding(4) var<storage, read_write> lse_out: array<f32>;
@group(0) @binding(5) var<storage, read> positions: array<i32>;
@group(0) @binding(6) var<storage, read> page_indices: array<u32>;
@group(0) @binding(7) var<storage, read> page_indptr: array<u32>;
@group(0) @binding(8) var<storage, read> req_of_token: array<i32>;
struct Params {
    num_q_heads: i32,
    head_dim: i32,
    ratio: i32,
    top_k: i32,
    page_size: i32,
    scale: f32,
}
@group(0) @binding(9) var<uniform> params: Params;
//#else
@group(0) @binding(2) var<storage, read_write> out_: array<u32>;
@group(0) @binding(3) var<storage, read_write> lse_out: array<f32>;
@group(0) @binding(4) var<storage, read> positions: array<i32>;
@group(0) @binding(5) var<storage, read> page_indices: array<u32>;
@group(0) @binding(6) var<storage, read> page_indptr: array<u32>;
@group(0) @binding(7) var<storage, read> req_of_token: array<i32>;
struct Params {
    num_q_heads: i32,
    head_dim: i32,
    ratio: i32,
    page_size: i32,
    scale: f32,
}
@group(0) @binding(8) var<uniform> params: Params;
//#endif

var<workgroup> pie_pool_q: array<f32, PIE_POOL_HEAD_MAX>;
var<workgroup> pie_pool_visible: i32;

fn pool_slot(req: i32, pos: i32) -> u32 {
    let page = page_indices[page_indptr[u32(req)] + u32(pos / params.page_size)];
    return page * u32(params.page_size) + u32(pos % params.page_size);
}

fn key_of(n: i32, qi: u32, num_visible: i32) -> i32 {
//#if defined(PIE_SELECTED)
    let c = selection[qi * u32(params.top_k) + u32(n)];
    if (c < 0 || c >= num_visible) {
        return -1;
    }
    return c;
//#else
    return n;
//#endif
}

@compute @workgroup_size(PIE_GROUP_X, 1, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let qi = group.y;
    let q_head = group.z;
    let tid = i32(local.x);
    let lanes = u32(PIE_GROUP_X);
    let head_dim = u32(params.head_dim);
    let req = req_of_token[qi];
    if (tid == 0) {
        pie_pool_visible = (positions[qi] + 1) / params.ratio;
    }

    let q_row = (qi * u32(params.num_q_heads) + q_head) * head_dim;
    for (var d = u32(tid) * 2u; d < head_dim; d = d + lanes * 2u) {
        let word = q[(q_row + d) >> 1u];
        pie_pool_q[d] = pie_bf16_to_f32(word & 0xffffu);
        pie_pool_q[d + 1u] = pie_bf16_to_f32(word >> 16u);
    }
    workgroupBarrier();
    let num_visible = workgroupUniformLoad(&pie_pool_visible);

//#if defined(PIE_SELECTED)
    var steps = params.top_k;
//#else
    var steps = num_visible;
//#endif
    if (num_visible <= 0) {
        steps = 0;
    }

    var local_max = pie_neg_inf();
    for (var n = tid; n < steps; n = n + PIE_GROUP_X) {
        let c = key_of(n, qi, num_visible);
        if (c < 0) {
            continue;
        }
        let k_row = pool_slot(req, (c + 1) * params.ratio - 1) * head_dim;
        var dot = 0.0;
        for (var d = 0u; d < head_dim; d = d + 2u) {
            let word = comp_kv_pages[(k_row + d) >> 1u];
            dot = dot + pie_pool_q[d] * pie_bf16_to_f32(word & 0xffffu) + pie_pool_q[d + 1u] * pie_bf16_to_f32(word >> 16u);
        }
        local_max = max(local_max, dot * params.scale);
    }
    let row_max = pie_workgroup_max(u32(tid), lanes, local_max);
    let dead = !(row_max > -3.0e38);

    var acc: array<f32, 4>;
    for (var i = 0u; i < 4u; i = i + 1u) {
        acc[i] = 0.0;
    }
    var local_z = 0.0;
    for (var n = 0; n < steps; n = n + 1) {
        let c = key_of(n, qi, num_visible);
        let live = c >= 0;
        var k_row = 0u;
        var dot = 0.0;
        if (live) {
            k_row = pool_slot(req, (c + 1) * params.ratio - 1) * head_dim;
            for (var d = u32(tid) * 2u; d < head_dim; d = d + lanes * 2u) {
                let word = comp_kv_pages[(k_row + d) >> 1u];
                dot = dot + pie_pool_q[d] * pie_bf16_to_f32(word & 0xffffu) + pie_pool_q[d + 1u] * pie_bf16_to_f32(word >> 16u);
            }
        }
        let s = pie_workgroup_sum(u32(tid), lanes, dot);
        if (live) {
            let w = exp(s * params.scale - row_max);
            local_z = local_z + w;
            for (var i = 0u; i < PIE_PAIRS_PER_THREAD; i = i + 1u) {
                let d = (u32(tid) + i * lanes) * 2u;
                if (d < head_dim) {
                    let word = comp_kv_pages[(k_row + d) >> 1u];
                    acc[2u * i] = acc[2u * i] + w * pie_bf16_to_f32(word & 0xffffu);
                    acc[2u * i + 1u] = acc[2u * i + 1u] + w * pie_bf16_to_f32(word >> 16u);
                }
            }
        }
    }
    let lse_at = qi * u32(params.num_q_heads) + q_head;
    var inv_z = 0.0;
    if (local_z > 0.0) {
        inv_z = 1.0 / local_z;
    }
    if (dead || num_visible <= 0) {
        inv_z = 0.0;
    }
    if (tid == 0) {
        if (local_z > 0.0 && !dead && num_visible > 0) {
            lse_out[lse_at] = (log(local_z) + row_max) * 1.44269504088896340736;
        } else {
            lse_out[lse_at] = pie_neg_inf();
        }
    }
    for (var i = 0u; i < PIE_PAIRS_PER_THREAD; i = i + 1u) {
        let d = (u32(tid) + i * lanes) * 2u;
        if (d < head_dim) {
            out_[(q_row + d) >> 1u] = pie_pack_bf16(acc[2u * i] * inv_z, acc[2u * i + 1u] * inv_z);
        }
    }
}
//#endif

// pie:instantiate pool_boundary_decode PIE_POOL_BOUNDARY=1 PIE_GROUP_X=128
// pie:instantiate pool_boundary_prefill PIE_POOL_BOUNDARY=1 PIE_PREFILL=1 PIE_GROUP_X=128
// pie:instantiate pool_state_write_bf16 PIE_POOL_STATE_WRITE=1 PIE_GROUP_X=256
// pie:instantiate pool_gather_paged_bf16 PIE_POOL_GATHER=1 PIE_GROUP_X=256
// pie:instantiate pool_store_entries_bf16 PIE_POOL_STORE=1 PIE_GROUP_X=256
// pie:instantiate pool_lse_paged PIE_GROUP_X=128
// pie:instantiate pool_lse_selected_paged PIE_SELECTED=1 PIE_GROUP_X=128
