//#include "common/bf16.inc.wgsl"
//#include "common/reduce.inc.wgsl"

fn index_rope_word(word: u32, i: i32, rope_dim: i32, pos: i32, theta: f32) -> u32 {
    let freq = pow(theta, -2.0 * f32(i) / f32(rope_dim));
    let ang = f32(pos) * freq;
    let c = cos(ang);
    let s = sin(ang);
    let a = pie_bf16_to_f32(word & 0xffffu);
    let b = pie_bf16_to_f32(word >> 16u);
    return pie_pack_bf16(a * c - b * s, b * c + a * s);
}

//#if defined(PIE_INDEX_KNORM)

@group(0) @binding(0) var<storage, read_write> idx_k: array<u32>;
@group(0) @binding(1) var<storage, read> w: array<u32>;
@group(0) @binding(2) var<storage, read> b: array<u32>;
@group(0) @binding(3) var<storage, read> positions: array<i32>;
struct Params {
    head_dim: i32,
    rope_dim: i32,
    theta: f32,
    eps: f32,
}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(PIE_GROUP_X, 1, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let n = group.x;
    let tid = local.x;
    let head_dim = u32(params.head_dim);
    let base = n * head_dim;
    let span = u32(PIE_GROUP_X) * 2u;

    var s = 0.0;
    for (var d = tid * 2u; d < head_dim; d = d + span) {
        let word = idx_k[(base + d) >> 1u];
        s = s + pie_bf16_to_f32(word & 0xffffu) + pie_bf16_to_f32(word >> 16u);
    }
    let mean = pie_workgroup_sum(tid, u32(PIE_GROUP_X), s) / f32(head_dim);

    var vv = 0.0;
    for (var d = tid * 2u; d < head_dim; d = d + span) {
        let word = idx_k[(base + d) >> 1u];
        let x0 = pie_bf16_to_f32(word & 0xffffu) - mean;
        let x1 = pie_bf16_to_f32(word >> 16u) - mean;
        vv = vv + x0 * x0 + x1 * x1;
    }
    let inv = inverseSqrt(pie_workgroup_sum(tid, u32(PIE_GROUP_X), vv) / f32(head_dim) + params.eps);

    for (var d = tid * 2u; d < head_dim; d = d + span) {
        let at = (base + d) >> 1u;
        let word = idx_k[at];
        let ww = w[d >> 1u];
        let bb = b[d >> 1u];
        let x0 = (pie_bf16_to_f32(word & 0xffffu) - mean) * inv;
        let x1 = (pie_bf16_to_f32(word >> 16u) - mean) * inv;
        idx_k[at] = pie_pack_bf16(
            x0 * pie_bf16_to_f32(ww & 0xffffu) + pie_bf16_to_f32(bb & 0xffffu),
            x1 * pie_bf16_to_f32(ww >> 16u) + pie_bf16_to_f32(bb >> 16u),
        );
    }

    storageBarrier();
    workgroupBarrier();

    let pos = positions[n];
    let pairs = params.rope_dim / 2;
    for (var i = i32(tid); i < pairs; i = i + PIE_GROUP_X) {
        let at = (base >> 1u) + u32(i);
        idx_k[at] = index_rope_word(idx_k[at], i, params.rope_dim, pos, params.theta);
    }
}

//#elif defined(PIE_INDEX_Q_ROPE)

@group(0) @binding(0) var<storage, read_write> idx_q: array<u32>;
@group(0) @binding(1) var<storage, read> positions: array<i32>;
struct Params {
    n_heads: i32,
    head_dim: i32,
    rope_dim: i32,
    theta: f32,
}
@group(0) @binding(2) var<uniform> params: Params;

@compute @workgroup_size(PIE_GROUP_X, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let h = i32(gid.x);
    if (h >= params.n_heads) {
        return;
    }
    let n = gid.y;
    let base = ((n * u32(params.n_heads) + u32(h)) * u32(params.head_dim)) >> 1u;
    let pos = positions[n];
    let pairs = params.rope_dim / 2;
    for (var i = 0; i < pairs; i = i + 1) {
        let at = base + u32(i);
        idx_q[at] = index_rope_word(idx_q[at], i, params.rope_dim, pos, params.theta);
    }
}

//#else

const PIE_LANES_PER_KEY = 64;
const PIE_KEYS_PER_PASS = PIE_GROUP_X / PIE_LANES_PER_KEY;
@group(0) @binding(0) var<storage, read> idx_q: array<u32>;
@group(0) @binding(1) var<storage, read> idx_w: array<u32>;
@group(0) @binding(2) var<storage, read> key_pages: array<u32>;
@group(0) @binding(3) var<storage, read> positions: array<i32>;
@group(0) @binding(4) var<storage, read> req_of_token: array<i32>;
@group(0) @binding(5) var<storage, read> kv_page_indices: array<u32>;
@group(0) @binding(6) var<storage, read> kv_page_indptr: array<u32>;
@group(0) @binding(7) var<storage, read_write> scores: array<f32>;
@group(0) @binding(8) var<storage, read_write> selection: array<i32>;
struct Params {
    H: i32,
    D: i32,
    page_size: i32,
    score_stride: i32,
    topk: i32,
    ratio: i32,
}
@group(0) @binding(9) var<uniform> params: Params;

var<workgroup> pie_key_acc: array<f32, PIE_GROUP_X>;
var<workgroup> pie_total: i32;

@compute @workgroup_size(PIE_GROUP_X, 1, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let t = group.x;
    let tid = i32(local.x);
    let srow = t * u32(params.topk);
    let topk = params.topk;

    let r = req_of_token[t];
    let pages_first = kv_page_indptr[u32(r)];
    var stride = 1;
    if (params.ratio > 0) {
        stride = params.ratio;
    }

    if (tid == 0) {
        pie_total = positions[t] + 1;
    }
    workgroupBarrier();
    let total = workgroupUniformLoad(&pie_total);
    var npools = 0;
    var tail = 0;
    if (total > 0) {
        npools = total / stride;
        tail = total - npools * stride;
    }
    if (tail > topk) {
        tail = topk;
    }
    let pool_budget = (topk - tail) / stride;
    if (npools > params.score_stride) {
        npools = params.score_stride;
    }

    let frow = t * u32(params.score_stride);
    let H = u32(params.H);
    let D = u32(params.D);
    let qi = t * H * D;
    let wi = t * H;

    let slot = tid / PIE_LANES_PER_KEY;
    let hh = tid % PIE_LANES_PER_KEY;
    for (var j0 = 0; j0 < npools; j0 = j0 + PIE_KEYS_PER_PASS) {
        let j = j0 + slot;
        var acc = 0.0;
        if (j < npools) {
            let cell = (j + 1) * stride - 1;
            let page = kv_page_indices[pages_first + u32(cell / params.page_size)];
            let kj = (page * u32(params.page_size) + u32(cell % params.page_size)) * D;
            for (var h = u32(hh); h < H; h = h + u32(PIE_LANES_PER_KEY)) {
                let qh = qi + h * D;
                var dot = 0.0;
                for (var d = 0u; d < D; d = d + 2u) {
                    let qw = idx_q[(qh + d) >> 1u];
                    let kw = key_pages[(kj + d) >> 1u];
                    dot = dot + pie_bf16_to_f32(qw & 0xffffu) * pie_bf16_to_f32(kw & 0xffffu)
                        + pie_bf16_to_f32(qw >> 16u) * pie_bf16_to_f32(kw >> 16u);
                }
                acc = acc + max(dot, 0.0) * pie_bf16_at(idx_w[(wi + h) >> 1u], wi + h);
            }
        }
        pie_key_acc[tid] = acc;
        workgroupBarrier();
        if (tid < PIE_KEYS_PER_PASS && j0 + tid < npools) {
            var sum = 0.0;
            for (var g = 0; g < PIE_LANES_PER_KEY; g = g + 1) {
                sum = sum + pie_key_acc[tid * PIE_LANES_PER_KEY + g];
            }
            scores[frow + u32(j0 + tid)] = sum;
        }
        workgroupBarrier();
    }

    storageBarrier();
    workgroupBarrier();

    for (var i = tid; i < tail; i = i + PIE_GROUP_X) {
        selection[srow + u32(i)] = npools * stride + i;
    }

    if (npools <= pool_budget) {
        for (var n = tid; n < topk - tail; n = n + PIE_GROUP_X) {
            let j = n / stride;
            var id = -1;
            if (j < npools) {
                id = j * stride + (n % stride);
            }
            selection[srow + u32(tail + n)] = id;
        }
        return;
    }

    var lo_l = 3.0e38;
    var hi_l = -3.0e38;
    for (var j = tid; j < npools; j = j + PIE_GROUP_X) {
        let s = scores[frow + u32(j)];
        lo_l = min(lo_l, s);
        hi_l = max(hi_l, s);
    }
    var lo = -pie_workgroup_max(u32(tid), u32(PIE_GROUP_X), -lo_l);
    var hi = pie_workgroup_max(u32(tid), u32(PIE_GROUP_X), hi_l);

    var thr = hi;
    for (var it = 0; it < 40; it = it + 1) {
        let mid = 0.5 * (lo + hi);
        var c = 0.0;
        for (var j = tid; j < npools; j = j + PIE_GROUP_X) {
            if (scores[frow + u32(j)] >= mid) {
                c = c + 1.0;
            }
        }
        let cnt = i32(pie_workgroup_sum(u32(tid), u32(PIE_GROUP_X), c));
        if (cnt > pool_budget) {
            lo = mid;
        } else {
            hi = mid;
        }
        thr = hi;
    }

    if (tid == 0) {
        var n = tail;
        for (var j = 0; j < npools && n + stride <= topk; j = j + 1) {
            if (scores[frow + u32(j)] >= thr) {
                for (var i = 0; i < stride; i = i + 1) {
                    selection[srow + u32(n)] = j * stride + i;
                    n = n + 1;
                }
            }
        }
        for (; n < topk; n = n + 1) {
            selection[srow + u32(n)] = -1;
        }
    }
}
//#endif

// pie:instantiate index_knorm_rope_bf16 PIE_INDEX_KNORM=1 PIE_GROUP_X=256
// pie:instantiate index_q_rope_bf16 PIE_INDEX_Q_ROPE=1 PIE_GROUP_X=32
// pie:instantiate index_topk_paged_bf16 PIE_GROUP_X=256
