//#include "common/bf16.inc.wgsl"
//#include "common/reduce.inc.wgsl"

const PIE_MAX_BLOCKS = 32u;

@group(0) @binding(0) var<storage, read> prefix: array<u32>;
@group(0) @binding(1) var<storage, read> blocks: array<u32>;
@group(0) @binding(2) var<storage, read> norm_w: array<u32>;
@group(0) @binding(3) var<storage, read> proj_w: array<u32>;
@group(0) @binding(4) var<storage, read_write> out_: array<u32>;

struct Params {
    blocks_n: u32,
    hidden: u32,
    rows: u32,
    eps: f32,
}
@group(0) @binding(5) var<uniform> params: Params;

var<workgroup> pie_prob: array<f32, PIE_MAX_BLOCKS + 1u>;

fn cand_word_base(j: u32, row: u32, n_blocks: u32, hidden: u32, rows: u32) -> u32 {
    if (j < n_blocks) {
        return ((j * rows + row) * hidden) / 2u;
    }
    return (row * hidden) / 2u;
}

fn cand_word(j: u32, n_blocks: u32, at: u32) -> u32 {
    if (j < n_blocks) {
        return blocks[at];
    }
    return prefix[at];
}

@compute @workgroup_size(PIE_GROUP_X)
fn main(
    @builtin(workgroup_id) group: vec3<u32>,
    @builtin(local_invocation_id) local: vec3<u32>,
) {
    let row = group.x;
    let hidden = params.hidden;
    let n_blocks = params.blocks_n;
    let cands = n_blocks + 1u;
    let lid = local.x;
    let lanes = u32(PIE_GROUP_X);
    let words = hidden / 2u;
    let out_base = (row * hidden) / 2u;

    for (var j = 0u; j < cands; j = j + 1u) {
        let base = cand_word_base(j, row, n_blocks, hidden, params.rows);
        var ss = 0.0;
        for (var w = lid; w < words; w = w + lanes) {
            let word = cand_word(j, n_blocks, base + w);
            let a = pie_bf16_at(word, 0u);
            let b = pie_bf16_at(word, 1u);
            ss = ss + a * a + b * b;
        }
        let inv = pie_inv_rms(lid, lanes, ss, hidden, params.eps);

        var dot = 0.0;
        for (var w = lid; w < words; w = w + lanes) {
            let word = cand_word(j, n_blocks, base + w);
            let nw = norm_w[w];
            let pw = proj_w[w];
            dot = dot
                + pie_bf16_at(word, 0u) * inv * pie_bf16_at(nw, 0u) * pie_bf16_at(pw, 0u)
                + pie_bf16_at(word, 1u) * inv * pie_bf16_at(nw, 1u) * pie_bf16_at(pw, 1u);
        }
        let total = pie_workgroup_sum(lid, lanes, dot);

        if (lid == 0u) {
            pie_prob[j] = total;
        }
    }

    if (lid == 0u) {
        var m = pie_prob[0];
        for (var j = 1u; j < cands; j = j + 1u) {
            m = max(m, pie_prob[j]);
        }
        var sum = 0.0;
        for (var j = 0u; j < cands; j = j + 1u) {
            let e = exp(pie_prob[j] - m);
            pie_prob[j] = e;
            sum = sum + e;
        }
        let inv_sum = 1.0 / sum;
        for (var j = 0u; j < cands; j = j + 1u) {
            pie_prob[j] = pie_prob[j] * inv_sum;
        }
    }
    workgroupBarrier();

    for (var w = lid; w < words; w = w + lanes) {
        var lo = 0.0;
        var hi = 0.0;
        for (var j = 0u; j < cands; j = j + 1u) {
            let base = cand_word_base(j, row, n_blocks, hidden, params.rows);
            let word = cand_word(j, n_blocks, base + w);
            let p = pie_prob[j];
            lo = lo + p * pie_bf16_at(word, 0u);
            hi = hi + p * pie_bf16_at(word, 1u);
        }
        out_[out_base + w] = pie_pack_bf16(lo, hi);
    }
}

// pie:instantiate res_blend_bf16 PIE_GROUP_X=256
