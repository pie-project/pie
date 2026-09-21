//#include "common/bf16.inc.wgsl"
//#include "attn/sdpa_online.inc.wgsl"
//#if defined(PIE_SUBGROUP)
//#include "common/subgroup.inc.wgsl"
//#endif

const PIE_PAIRS = PIE_HEAD_DIM / (2 * PIE_LANES);
const PIE_ELEMS = 2 * PIE_PAIRS;

fn pie_word_at(base: u32, lane: u32, i: u32) -> u32 {
    return base + lane + i * PIE_LANES;
}

@group(0) @binding(0) var<storage, read> queries: array<u32>;
@group(0) @binding(1) var<storage, read> k_pages: array<u32>;
@group(0) @binding(2) var<storage, read> v_pages: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_: array<u32>;
@group(0) @binding(4) var<storage, read> position_ids: array<i32>;
@group(0) @binding(5) var<storage, read> req_of_token: array<i32>;
@group(0) @binding(6) var<storage, read> kv_page_indices: array<u32>;
@group(0) @binding(7) var<storage, read> kv_page_indptr: array<u32>;

@group(0) @binding(8) var<storage, read> attention_mask: array<u32>;
@group(0) @binding(9) var<storage, read> attention_mask_enabled: array<u32>;

//#if defined(PIE_LSE)

@group(0) @binding(11) var<storage, read_write> lse_out: array<f32>;
//#endif

struct Params {
    gqa_factor: i32,
    page_size: i32,
    n_kv_heads: i32,
    scale: f32,
    attention_mask_stride: u32,
    window: i32,
//#if defined(PIE_TILED)
    n_rows: i32,
//#endif
}
//#if defined(PIE_LSE)
@group(0) @binding(12) var<uniform> params: Params;
//#else
@group(0) @binding(10) var<uniform> params: Params;
//#endif

var<workgroup> pie_part: array<array<f32, PIE_LANES>, PIE_ROWS>;
var<workgroup> pie_pos: array<i32, PIE_ROWS>;
var<workgroup> pie_steps: i32;
var<workgroup> pie_max: array<f32, PIE_ROWS>;
var<workgroup> pie_sum: array<f32, PIE_ROWS>;

fn page_slot(req: i32, kp: i32) -> u32 {
    let page_ix = u32(kp / params.page_size);
    let page_off = u32(kp % params.page_size);
    let phys = kv_page_indices[kv_page_indptr[u32(req)] + page_ix];
    return phys * u32(params.page_size) + page_off;
}

fn byte_at(word: u32, i: u32) -> u32 {
    return (word >> ((i & 3u) * 8u)) & 0xffu;
}

fn keeps(row: u32, kp: i32, q_pos: i32, start: i32) -> bool {
    if (kp > q_pos || kp < start) {
        return false;
    }
    if (byte_at(attention_mask_enabled[row >> 2u], row) != 0u) {
        if (u32(kp) >= params.attention_mask_stride) {
            return false;
        }
        let at = row * params.attention_mask_stride + u32(kp);
        if (byte_at(attention_mask[at >> 2u], at) == 0u) {
            return false;
        }
    }
    return true;
}

fn window_start(q_pos: i32) -> i32 {
    if (params.window > 0 && q_pos >= params.window) {
        return q_pos - params.window + 1;
    }
    return 0;
}

@compute @workgroup_size(PIE_LANES, PIE_ROWS, 1)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let lane = local.x;
    let slot = local.y;
    let q_head = group.x;
    let n_q_heads = groups.x;
    let kv_head = u32(i32(q_head) / params.gqa_factor);

//#if defined(PIE_TILED)
    let row = group.y * PIE_ROWS + slot;
    let live = row < u32(params.n_rows);
    var req = 0;
    var q_pos = -1;
    if (live) {
        req = req_of_token[row];
        q_pos = position_ids[row];
    }
//#else
    let row = group.y;
    let live = true;
    let req = req_of_token[row];
    let q_pos = position_ids[row];
//#endif
    let start = window_start(q_pos);

//#if defined(PIE_SPLIT)

    let split = group.z;
    let n_splits = groups.z;
    let n_rows = groups.y;
    let key_count = max(q_pos - start + 1, 0);
    let per_split = (key_count + i32(n_splits) - 1) / i32(n_splits);
    let lo = start + i32(split) * per_split;
    let hi = min(q_pos, lo + per_split - 1);
//#else
    let lo = start;
    let hi = q_pos;
//#endif

    if (lane == 0u) {
        pie_pos[slot] = q_pos;
    }
    workgroupBarrier();
    if (lane == 0u && slot == 0u) {
        var last = -1;
        for (var s = 0u; s < PIE_ROWS; s++) {
            last = max(last, pie_pos[s]);
        }
//#if defined(PIE_TILED)
        pie_steps = last + 1;
//#else

        let count = max(min(last, hi) - lo + 1, 0);
        pie_steps = (count + PIE_ROWS - 1) / PIE_ROWS;
//#endif
    }
    let steps = workgroupUniformLoad(&pie_steps);

    var q_base = 0u;
    if (live) {
        q_base = ((row * n_q_heads + q_head) * PIE_HEAD_DIM) >> 1u;
    }
//#if defined(PIE_SPLIT)
    let o_col = (split * n_rows + row) * n_q_heads + q_head;
    let o_base = (o_col * PIE_HEAD_DIM) >> 1u;
//#else
    let o_col = row * n_q_heads + q_head;
    let o_base = q_base;
//#endif
    var qv: array<f32, PIE_ELEMS>;
    var acc: array<f32, PIE_ELEMS>;
    for (var i = 0u; i < PIE_PAIRS; i++) {
        var w = 0u;
        if (live) {
            w = queries[pie_word_at(q_base, lane, i)];
        }
        qv[2u * i] = params.scale * pie_bf16_to_f32(w & 0xffffu);
        qv[2u * i + 1u] = params.scale * pie_bf16_to_f32(w >> 16u);
        acc[2u * i] = 0.0;
        acc[2u * i + 1u] = 0.0;
    }

    var max_score = PIE_SDPA_NEG_INF;
    var sum_exp = 0.0;
    for (var j = 0; j < steps; j++) {
//#if defined(PIE_TILED)
        let kp = j;
//#else
        let kp = lo + i32(slot) + j * PIE_ROWS;
//#endif
        let take = live && kp <= hi && keeps(row, kp, q_pos, start);
        var base = 0u;
        var partial = 0.0;
        if (take) {
            let s = page_slot(req, kp);
            base = ((s * u32(params.n_kv_heads) + kv_head) * PIE_HEAD_DIM) >> 1u;
            for (var i = 0u; i < PIE_PAIRS; i++) {
                let w = k_pages[pie_word_at(base, lane, i)];
                partial += qv[2u * i] * pie_bf16_to_f32(w & 0xffffu)
                    + qv[2u * i + 1u] * pie_bf16_to_f32(w >> 16u);
            }
        }
//#if defined(PIE_SUBGROUP)

        let score = pie_subgroup_sum16(partial);
//#else
        pie_part[slot][lane] = partial;
        workgroupBarrier();
        var score = 0.0;
        for (var l = 0u; l < PIE_LANES; l++) {
            score += pie_part[slot][l];
        }

        workgroupBarrier();
//#endif
        if (take) {
            let sc = sdpa_online_scales(score, max_score);
            max_score = max(max_score, score);
            sum_exp = sum_exp * sc.x + sc.y;
            for (var i = 0u; i < PIE_PAIRS; i++) {
                let w = v_pages[pie_word_at(base, lane, i)];
                acc[2u * i] = acc[2u * i] * sc.x + sc.y * pie_bf16_to_f32(w & 0xffffu);
                acc[2u * i + 1u] = acc[2u * i + 1u] * sc.x + sc.y * pie_bf16_to_f32(w >> 16u);
            }
        }
    }

//#if defined(PIE_TILED)
    if (live) {
//#if defined(PIE_LSE)
        if (lane == 0u) {
            lse_out[o_col] = sdpa_lse_base2(max_score, sum_exp);
        }
//#endif
        let inv = select(1.0 / sum_exp, 1.0, sum_exp == 0.0);
        for (var i = 0u; i < PIE_PAIRS; i++) {
            out_[pie_word_at(o_base, lane, i)] =
                pie_pack_bf16(acc[2u * i] * inv, acc[2u * i + 1u] * inv);
        }
    }
//#else

    if (lane == 0u) {
        pie_max[slot] = max_score;
        pie_sum[slot] = sum_exp;
    }
    workgroupBarrier();
    var merged_max = PIE_SDPA_NEG_INF;
    for (var s = 0u; s < PIE_ROWS; s++) {
        merged_max = max(merged_max, pie_max[s]);
    }
    var merged_sum = 0.0;
    for (var s = 0u; s < PIE_ROWS; s++) {
        merged_sum += pie_sum[s] * exp(pie_max[s] - merged_max);
    }
    let w_slot = exp(max_score - merged_max);
    let inv = select(1.0 / merged_sum, 1.0, merged_sum == 0.0);
    for (var i = 0u; i < PIE_PAIRS; i++) {
        pie_part[slot][lane] = acc[2u * i] * w_slot;
        workgroupBarrier();
        var lo_v = 0.0;
        if (slot == 0u) {
            for (var s = 0u; s < PIE_ROWS; s++) {
                lo_v += pie_part[s][lane];
            }
        }
        workgroupBarrier();
        pie_part[slot][lane] = acc[2u * i + 1u] * w_slot;
        workgroupBarrier();
        if (slot == 0u) {
            var hi_v = 0.0;
            for (var s = 0u; s < PIE_ROWS; s++) {
                hi_v += pie_part[s][lane];
            }
            out_[pie_word_at(o_base, lane, i)] = pie_pack_bf16(lo_v * inv, hi_v * inv);
        }
        workgroupBarrier();
    }
//#if defined(PIE_LSE)
    if (lane == 0u && slot == 0u) {
        lse_out[o_col] = sdpa_lse_base2(merged_max, merged_sum);
    }
//#endif
//#endif
}

// pie:instantiate sdpa_paged_decode_bfloat16_d_64 PIE_HEAD_DIM=64 PIE_LANES=16 PIE_ROWS=16
// pie:instantiate sdpa_paged_decode_bfloat16_d_128 PIE_HEAD_DIM=128 PIE_LANES=16 PIE_ROWS=16
// pie:instantiate sdpa_paged_decode_bfloat16_d_256 PIE_HEAD_DIM=256 PIE_LANES=16 PIE_ROWS=16
// pie:instantiate sdpa_paged_decode_bfloat16_d_512 PIE_HEAD_DIM=512 PIE_LANES=16 PIE_ROWS=16
// pie:instantiate sdpa_paged_decode_lse_bfloat16_d_64 PIE_HEAD_DIM=64 PIE_LANES=16 PIE_ROWS=16 PIE_LSE=1
// pie:instantiate sdpa_paged_decode_lse_bfloat16_d_128 PIE_HEAD_DIM=128 PIE_LANES=16 PIE_ROWS=16 PIE_LSE=1
// pie:instantiate sdpa_paged_decode_lse_bfloat16_d_256 PIE_HEAD_DIM=256 PIE_LANES=16 PIE_ROWS=16 PIE_LSE=1
// pie:instantiate sdpa_paged_decode_lse_bfloat16_d_512 PIE_HEAD_DIM=512 PIE_LANES=16 PIE_ROWS=16 PIE_LSE=1
// pie:instantiate sdpa_paged_split_bfloat16_d_64 PIE_HEAD_DIM=64 PIE_LANES=16 PIE_ROWS=16 PIE_LSE=1 PIE_SPLIT=1
// pie:instantiate sdpa_paged_split_bfloat16_d_128 PIE_HEAD_DIM=128 PIE_LANES=16 PIE_ROWS=16 PIE_LSE=1 PIE_SPLIT=1
// pie:instantiate sdpa_paged_split_bfloat16_d_256 PIE_HEAD_DIM=256 PIE_LANES=16 PIE_ROWS=16 PIE_LSE=1 PIE_SPLIT=1
// pie:instantiate sdpa_paged_split_bfloat16_d_512 PIE_HEAD_DIM=512 PIE_LANES=16 PIE_ROWS=16 PIE_LSE=1 PIE_SPLIT=1
// pie:instantiate sdpa_paged_tiled_bfloat16_d_64 PIE_HEAD_DIM=64 PIE_LANES=16 PIE_ROWS=16 PIE_TILED=1
// pie:instantiate sdpa_paged_tiled_bfloat16_d_128 PIE_HEAD_DIM=128 PIE_LANES=16 PIE_ROWS=16 PIE_TILED=1
// pie:instantiate sdpa_paged_tiled_bfloat16_d_256 PIE_HEAD_DIM=256 PIE_LANES=16 PIE_ROWS=16 PIE_TILED=1
// pie:instantiate sdpa_paged_tiled_bfloat16_d_512 PIE_HEAD_DIM=512 PIE_LANES=16 PIE_ROWS=16 PIE_TILED=1
// pie:instantiate sdpa_paged_tiled_lse_bfloat16_d_64 PIE_HEAD_DIM=64 PIE_LANES=16 PIE_ROWS=16 PIE_TILED=1 PIE_LSE=1
// pie:instantiate sdpa_paged_tiled_lse_bfloat16_d_128 PIE_HEAD_DIM=128 PIE_LANES=16 PIE_ROWS=16 PIE_TILED=1 PIE_LSE=1
// pie:instantiate sdpa_paged_tiled_lse_bfloat16_d_256 PIE_HEAD_DIM=256 PIE_LANES=16 PIE_ROWS=16 PIE_TILED=1 PIE_LSE=1
// pie:instantiate sdpa_paged_tiled_lse_bfloat16_d_512 PIE_HEAD_DIM=512 PIE_LANES=16 PIE_ROWS=16 PIE_TILED=1 PIE_LSE=1
// pie:instantiate sdpa_paged_decode_bfloat16_d_64 @subgroup PIE_HEAD_DIM=64 PIE_LANES=16 PIE_ROWS=64
// pie:instantiate sdpa_paged_decode_bfloat16_d_128 @subgroup PIE_HEAD_DIM=128 PIE_LANES=16 PIE_ROWS=64
// pie:instantiate sdpa_paged_decode_bfloat16_d_256 @subgroup PIE_HEAD_DIM=256 PIE_LANES=16 PIE_ROWS=64
// pie:instantiate sdpa_paged_decode_bfloat16_d_512 @subgroup PIE_HEAD_DIM=512 PIE_LANES=16 PIE_ROWS=64
// pie:instantiate sdpa_paged_decode_lse_bfloat16_d_64 @subgroup PIE_HEAD_DIM=64 PIE_LANES=16 PIE_ROWS=64 PIE_LSE=1
// pie:instantiate sdpa_paged_decode_lse_bfloat16_d_128 @subgroup PIE_HEAD_DIM=128 PIE_LANES=16 PIE_ROWS=64 PIE_LSE=1
// pie:instantiate sdpa_paged_decode_lse_bfloat16_d_256 @subgroup PIE_HEAD_DIM=256 PIE_LANES=16 PIE_ROWS=64 PIE_LSE=1
// pie:instantiate sdpa_paged_decode_lse_bfloat16_d_512 @subgroup PIE_HEAD_DIM=512 PIE_LANES=16 PIE_ROWS=64 PIE_LSE=1
// pie:instantiate sdpa_paged_split_bfloat16_d_64 @subgroup PIE_HEAD_DIM=64 PIE_LANES=16 PIE_ROWS=64 PIE_LSE=1 PIE_SPLIT=1
// pie:instantiate sdpa_paged_split_bfloat16_d_128 @subgroup PIE_HEAD_DIM=128 PIE_LANES=16 PIE_ROWS=64 PIE_LSE=1 PIE_SPLIT=1
// pie:instantiate sdpa_paged_split_bfloat16_d_256 @subgroup PIE_HEAD_DIM=256 PIE_LANES=16 PIE_ROWS=64 PIE_LSE=1 PIE_SPLIT=1
// pie:instantiate sdpa_paged_split_bfloat16_d_512 @subgroup PIE_HEAD_DIM=512 PIE_LANES=16 PIE_ROWS=64 PIE_LSE=1 PIE_SPLIT=1
// pie:instantiate sdpa_paged_tiled_bfloat16_d_64 @subgroup PIE_HEAD_DIM=64 PIE_LANES=16 PIE_ROWS=16 PIE_TILED=1
// pie:instantiate sdpa_paged_tiled_bfloat16_d_128 @subgroup PIE_HEAD_DIM=128 PIE_LANES=16 PIE_ROWS=16 PIE_TILED=1
// pie:instantiate sdpa_paged_tiled_bfloat16_d_256 @subgroup PIE_HEAD_DIM=256 PIE_LANES=16 PIE_ROWS=16 PIE_TILED=1
// pie:instantiate sdpa_paged_tiled_bfloat16_d_512 @subgroup PIE_HEAD_DIM=512 PIE_LANES=16 PIE_ROWS=16 PIE_TILED=1
// pie:instantiate sdpa_paged_tiled_lse_bfloat16_d_64 @subgroup PIE_HEAD_DIM=64 PIE_LANES=16 PIE_ROWS=16 PIE_TILED=1 PIE_LSE=1
// pie:instantiate sdpa_paged_tiled_lse_bfloat16_d_128 @subgroup PIE_HEAD_DIM=128 PIE_LANES=16 PIE_ROWS=16 PIE_TILED=1 PIE_LSE=1
// pie:instantiate sdpa_paged_tiled_lse_bfloat16_d_256 @subgroup PIE_HEAD_DIM=256 PIE_LANES=16 PIE_ROWS=16 PIE_TILED=1 PIE_LSE=1
// pie:instantiate sdpa_paged_tiled_lse_bfloat16_d_512 @subgroup PIE_HEAD_DIM=512 PIE_LANES=16 PIE_ROWS=16 PIE_TILED=1 PIE_LSE=1
