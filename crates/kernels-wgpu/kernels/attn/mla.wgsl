//#include "common/bf16.inc.wgsl"
//#include "common/reduce.inc.wgsl"

//#if defined(PIE_MLA_LATENTS)

@group(0) @binding(0) var<storage, read> kv_a: array<u32>;
@group(0) @binding(1) var<storage, read> norm_weight: array<u32>;
@group(0) @binding(2) var<storage, read_write> kv_c: array<u32>;
@group(0) @binding(3) var<storage, read_write> k_pe: array<u32>;
struct Params {
    kv_lora: i32,
    rope: i32,
    src_row_stride: i32,
    eps: f32,
}
@group(0) @binding(4) var<uniform> params: Params;

@compute @workgroup_size(PIE_GROUP_X, 1, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let row = group.x;
    let lid = local.x;
    let base = row * u32(params.src_row_stride);
    let kv_lora = u32(params.kv_lora);
    let rope = u32(params.rope);
    let span = u32(PIE_GROUP_X) * 2u;
    for (var d = lid * 2u; d < rope; d = d + span) {
        k_pe[(row * rope + d) >> 1u] = kv_a[(base + kv_lora + d) >> 1u];
    }
    var sq = 0.0;
    for (var d = lid * 2u; d < kv_lora; d = d + span) {
        let word = kv_a[(base + d) >> 1u];
        let a = pie_bf16_to_f32(word & 0xffffu);
        let b = pie_bf16_to_f32(word >> 16u);
        sq = sq + a * a + b * b;
    }
    let inv = pie_inv_rms(lid, u32(PIE_GROUP_X), sq, kv_lora, params.eps);
    for (var d = lid * 2u; d < kv_lora; d = d + span) {
        let word = kv_a[(base + d) >> 1u];
        let w = norm_weight[d >> 1u];
        let a = pie_bf16_to_f32(word & 0xffffu) * inv * pie_bf16_to_f32(w & 0xffffu);
        let b = pie_bf16_to_f32(word >> 16u) * inv * pie_bf16_to_f32(w >> 16u);
        kv_c[(row * kv_lora + d) >> 1u] = pie_pack_bf16(a, b);
    }
}

//#elif defined(PIE_MLA_SPLIT_Q)

@group(0) @binding(0) var<storage, read> q_b: array<u32>;
@group(0) @binding(1) var<storage, read_write> q_nope: array<u32>;
@group(0) @binding(2) var<storage, read_write> q_pe: array<u32>;
struct Params {
    total: i32,
    heads: i32,
    nope: i32,
    rope: i32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(PIE_GROUP_X, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = i32(gid.x) * 2;
    if (i >= params.total) {
        return;
    }
    let per = params.nope + params.rope;
    let d = i % per;
    let h = (i / per) % params.heads;
    let n = i / (params.heads * per);
    let word = q_b[u32(i) >> 1u];
    if (d < params.nope) {
        q_nope[u32((n * params.heads + h) * params.nope + d) >> 1u] = word;
    } else {
        q_pe[u32((n * params.heads + h) * params.rope + (d - params.nope)) >> 1u] = word;
    }
}

//#elif defined(PIE_MLA_KV_APPEND)

@group(0) @binding(0) var<storage, read> kv_c: array<u32>;
@group(0) @binding(1) var<storage, read> k_pe: array<u32>;
@group(0) @binding(2) var<storage, read_write> ckv_pages: array<u32>;
@group(0) @binding(3) var<storage, read_write> kpe_pages: array<u32>;
@group(0) @binding(4) var<storage, read> w_page: array<u32>;
@group(0) @binding(5) var<storage, read> w_off: array<u32>;
struct Params {
    page_size: i32,
    kv_lora: i32,
    rope: i32,
    rows: i32,
}
@group(0) @binding(6) var<uniform> params: Params;

@compute @workgroup_size(PIE_GROUP_X, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let d = i32(gid.x) * 2;
    let row = gid.y;
    if (row >= u32(params.rows)) {
        return;
    }

    let page = i32(w_page[row]);
    let off = i32(w_off[row]);
    if (page < 0 || off < 0 || off >= params.page_size) {
        return;
    }
    let slot = u32(page) * u32(params.page_size) + u32(off);
    if (d < params.kv_lora) {
        ckv_pages[(slot * u32(params.kv_lora) + u32(d)) >> 1u] = kv_c[(row * u32(params.kv_lora) + u32(d)) >> 1u];
    }
    if (d < params.rope) {
        kpe_pages[(slot * u32(params.rope) + u32(d)) >> 1u] = k_pe[(row * u32(params.rope) + u32(d)) >> 1u];
    }
}

//#elif defined(PIE_MLA_ABSORB_Q)

@group(0) @binding(0) var<storage, read> q_nope: array<u32>;
@group(0) @binding(1) var<storage, read> kv_b: array<u32>;
@group(0) @binding(2) var<storage, read_write> q_latent: array<u32>;
struct Params {
    heads: i32,
    rank: i32,
    nope: i32,
    v_dim: i32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(PIE_GROUP_X, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let i = i32(gid.x) * 2;
    let h = i32(gid.y);
    let t = i32(gid.z);
    if (i >= params.rank) {
        return;
    }
    let rank = u32(params.rank);
    let qn_base = u32((t * params.heads + h) * params.nope);
    let kb_base = u32(h) * u32(params.nope + params.v_dim) * rank;
    var acc0 = 0.0;
    var acc1 = 0.0;
    for (var j = 0u; j < u32(params.nope); j = j + 2u) {
        let qw = q_nope[(qn_base + j) >> 1u];
        let q0 = pie_bf16_to_f32(qw & 0xffffu);
        let q1 = pie_bf16_to_f32(qw >> 16u);
        let w0 = kv_b[(kb_base + j * rank + u32(i)) >> 1u];
        let w1 = kv_b[(kb_base + (j + 1u) * rank + u32(i)) >> 1u];
        acc0 = acc0 + q0 * pie_bf16_to_f32(w0 & 0xffffu) + q1 * pie_bf16_to_f32(w1 & 0xffffu);
        acc1 = acc1 + q0 * pie_bf16_to_f32(w0 >> 16u) + q1 * pie_bf16_to_f32(w1 >> 16u);
    }
    q_latent[u32((t * params.heads + h) * params.rank + i) >> 1u] = pie_pack_bf16(acc0, acc1);
}

//#elif defined(PIE_MLA_ABSORB_OUT)

@group(0) @binding(0) var<storage, read> latent: array<u32>;
@group(0) @binding(1) var<storage, read> kv_b: array<u32>;
@group(0) @binding(2) var<storage, read_write> out_: array<u32>;
struct Params {
    heads: i32,
    rank: i32,
    v_dim: i32,
    nope: i32,
}
@group(0) @binding(3) var<uniform> params: Params;

@compute @workgroup_size(PIE_GROUP_X, 1, 1)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let j = i32(gid.x) * 2;
    let h = i32(gid.y);
    let t = i32(gid.z);
    if (j >= params.v_dim) {
        return;
    }
    let rank = u32(params.rank);
    let lat_base = u32((t * params.heads + h) * params.rank);
    let wv_base = u32(h) * u32(params.nope + params.v_dim) * rank + u32(params.nope) * rank;
    let row0 = wv_base + u32(j) * rank;
    let row1 = row0 + rank;
    var acc0 = 0.0;
    var acc1 = 0.0;
    for (var i = 0u; i < rank; i = i + 2u) {
        let lw = latent[(lat_base + i) >> 1u];
        let l0 = pie_bf16_to_f32(lw & 0xffffu);
        let l1 = pie_bf16_to_f32(lw >> 16u);
        let w0 = kv_b[(row0 + i) >> 1u];
        let w1 = kv_b[(row1 + i) >> 1u];
        acc0 = acc0 + l0 * pie_bf16_to_f32(w0 & 0xffffu) + l1 * pie_bf16_to_f32(w0 >> 16u);
        acc1 = acc1 + l0 * pie_bf16_to_f32(w1 & 0xffffu) + l1 * pie_bf16_to_f32(w1 >> 16u);
    }
    out_[u32((t * params.heads + h) * params.v_dim + j) >> 1u] = pie_pack_bf16(acc0, acc1);
}

//#else

const PIE_MLA_LANES = 32u;
const PIE_MAX_CKV_PAIRS = 8u;
const PIE_MAX_KPE_PAIRS = 2u;

@group(0) @binding(0) var<storage, read> q_nope: array<u32>;
@group(0) @binding(1) var<storage, read> q_pe: array<u32>;
@group(0) @binding(2) var<storage, read> ckv_pages: array<u32>;
@group(0) @binding(3) var<storage, read> kpe_pages: array<u32>;
@group(0) @binding(4) var<storage, read_write> out_: array<u32>;
@group(0) @binding(5) var<storage, read> position_ids: array<i32>;
@group(0) @binding(6) var<storage, read> req_of_token: array<i32>;
@group(0) @binding(7) var<storage, read> kv_page_indices: array<u32>;
@group(0) @binding(8) var<storage, read> kv_page_indptr: array<u32>;
//#if defined(PIE_SELECTED)
@group(0) @binding(9) var<storage, read> selection: array<i32>;
//#endif
struct Params {
    page_size: i32,
    heads: i32,
    ckv: i32,
    kpe: i32,
    sm_scale: f32,
//#if defined(PIE_SELECTED)
    top_k: i32,
//#endif
}
//#if defined(PIE_SELECTED)
@group(0) @binding(10) var<uniform> params: Params;
//#else
@group(0) @binding(9) var<uniform> params: Params;
//#endif

var<workgroup> pie_mla_fold: array<f32, PIE_MLA_LANES>;
var<workgroup> pie_mla_steps: i32;

fn mla_lane_sum(v: f32, lane: u32) -> f32 {
    pie_mla_fold[lane] = v;
    workgroupBarrier();
    var half_ = PIE_MLA_LANES >> 1u;
    loop {
        if (half_ == 0u) {
            break;
        }
        if (lane < half_) {
            pie_mla_fold[lane] = pie_mla_fold[lane] + pie_mla_fold[lane + half_];
        }
        workgroupBarrier();
        half_ = half_ >> 1u;
    }
    let total = pie_mla_fold[0];
    workgroupBarrier();
    return total;
}

@compute @workgroup_size(PIE_MLA_LANES, 1, 1)
fn main(@builtin(workgroup_id) group: vec3<u32>, @builtin(local_invocation_id) local: vec3<u32>) {
    let h = group.x;
    let row = group.y;
    let lane = local.x;
    let ckv = u32(params.ckv);
    let kpe = u32(params.kpe);
    let pairs = ckv / (2u * PIE_MLA_LANES);
    let ppairs = kpe / (2u * PIE_MLA_LANES);
    let heads = u32(params.heads);

    let r = req_of_token[row];
    let q_pos = position_ids[row];
    let j_end = q_pos + 1;
    let page_base = kv_page_indptr[u32(r)];

    let qn_base = ((row * heads + h) * ckv) >> 1u;
    let qp_base = ((row * heads + h) * kpe) >> 1u;
    var qn_r: array<f32, 16>;
    var qp_r: array<f32, 4>;
    for (var p = 0u; p < PIE_MAX_CKV_PAIRS; p = p + 1u) {
        var w = 0u;
        if (p < pairs) {
            w = q_nope[qn_base + lane + p * PIE_MLA_LANES];
        }
        qn_r[2u * p] = pie_bf16_to_f32(w & 0xffffu);
        qn_r[2u * p + 1u] = pie_bf16_to_f32(w >> 16u);
    }
    for (var p = 0u; p < PIE_MAX_KPE_PAIRS; p = p + 1u) {
        var w = 0u;
        if (p < ppairs) {
            w = q_pe[qp_base + lane + p * PIE_MLA_LANES];
        }
        qp_r[2u * p] = pie_bf16_to_f32(w & 0xffffu);
        qp_r[2u * p + 1u] = pie_bf16_to_f32(w >> 16u);
    }

    var acc: array<f32, 16>;
    for (var i = 0u; i < 16u; i = i + 1u) {
        acc[i] = 0.0;
    }
    var m = -3.0e38;
    var lsum = 0.0;

    if (lane == 0u) {
//#if defined(PIE_SELECTED)
        pie_mla_steps = params.top_k;
//#else
        pie_mla_steps = j_end;
//#endif
    }
    workgroupBarrier();
    let steps = workgroupUniformLoad(&pie_mla_steps);
//#if defined(PIE_SELECTED)
    let srow = row * u32(params.top_k);
//#endif
    for (var n = 0; n < steps; n = n + 1) {
        var j = n;
        var live = true;
//#if defined(PIE_SELECTED)
        j = selection[srow + u32(n)];
        live = j >= 0 && j < j_end;
//#endif
        var kv: array<f32, 16>;
        var pd = 0.0;
        if (live) {
            let page = kv_page_indices[page_base + u32(j / params.page_size)];
            let slot = page * u32(params.page_size) + u32(j % params.page_size);
            let ckv_j = (slot * ckv) >> 1u;
            let kpe_j = (slot * kpe) >> 1u;
            for (var p = 0u; p < PIE_MAX_CKV_PAIRS; p = p + 1u) {
                var w = 0u;
                if (p < pairs) {
                    w = ckv_pages[ckv_j + lane + p * PIE_MLA_LANES];
                }
                kv[2u * p] = pie_bf16_to_f32(w & 0xffffu);
                kv[2u * p + 1u] = pie_bf16_to_f32(w >> 16u);
                pd = pd + qn_r[2u * p] * kv[2u * p] + qn_r[2u * p + 1u] * kv[2u * p + 1u];
            }
            for (var p = 0u; p < PIE_MAX_KPE_PAIRS; p = p + 1u) {
                if (p < ppairs) {
                    let w = kpe_pages[kpe_j + lane + p * PIE_MLA_LANES];
                    pd = pd + qp_r[2u * p] * pie_bf16_to_f32(w & 0xffffu) + qp_r[2u * p + 1u] * pie_bf16_to_f32(w >> 16u);
                }
            }
        } else {
            for (var i = 0u; i < 16u; i = i + 1u) {
                kv[i] = 0.0;
            }
        }
        pd = mla_lane_sum(pd, lane);
        if (live) {
            let score = pd * params.sm_scale;
            let m_new = max(m, score);
            let corr = exp(m - m_new);
            let p = exp(score - m_new);
            lsum = lsum * corr + p;
            for (var i = 0u; i < 16u; i = i + 1u) {
                acc[i] = acc[i] * corr + p * kv[i];
            }
            m = m_new;
        }
    }

    var inv = 0.0;
    if (lsum > 0.0) {
        inv = 1.0 / lsum;
    }
    let o_base = ((row * heads + h) * ckv) >> 1u;
    for (var p = 0u; p < pairs; p = p + 1u) {
        out_[o_base + lane + p * PIE_MLA_LANES] = pie_pack_bf16(acc[2u * p] * inv, acc[2u * p + 1u] * inv);
    }
}
//#endif

// pie:instantiate mla_latents_bf16 PIE_MLA_LATENTS=1 PIE_GROUP_X=256
// pie:instantiate mla_split_q_b_bf16 PIE_MLA_SPLIT_Q=1 PIE_GROUP_X=256
// pie:instantiate mla_kv_append_bf16 PIE_MLA_KV_APPEND=1 PIE_GROUP_X=256
// pie:instantiate mla_absorb_q_bf16 PIE_MLA_ABSORB_Q=1 PIE_GROUP_X=64
// pie:instantiate mla_absorb_out_bf16 PIE_MLA_ABSORB_OUT=1 PIE_GROUP_X=64
// pie:instantiate mla_naive_paged_bf16
// pie:instantiate mla_naive_paged_selected_bf16 PIE_SELECTED=1
