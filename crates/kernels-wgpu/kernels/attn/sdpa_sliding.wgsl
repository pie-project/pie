// Dense decode SDPA over a SLIDING window, plus the learned-sink variant.
//
// The vector body with two changes. The causal end moves to `n - (M-1-row)`,
// so a batch of M query rows against one cache each stops at its own position
// rather than at the last one; and a window, when the row's `window` operand is
// positive, moves the START forward. The window is an OPERAND and not a flag --
// this port's rule that a per-fire choice the C++ made at encode time becomes
// data on the dispatch.
//
// The sink variant folds a per-head learned logit into the denominator with no
// value behind it. Its row is unstated, so `sinks` sits at the binding the
// buffer run gives it -- 4, after `out` -- which is what
// `kernels-vulkan/kernels/attn/sdpa_sliding.comp` also declares.
//
// One lane owns a channel PAIR, for the reason `sdpa_vector.wgsl` gives: a
// bf16 word holds two values, WGSL has no sub-word atomic, and a 512-wide
// workgroup would exceed WebGPU's guaranteed 256 invocations. At `d_512` this
// body runs exactly 256 lanes.

//#include "common/bf16.inc.wgsl"
//#include "attn/sdpa_online.inc.wgsl"

@group(0) @binding(0) var<storage, read_write> queries: array<u32>;
@group(0) @binding(1) var<storage, read_write> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> values: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_: array<u32>;
//#if defined(PIE_WITH_SINK)
@group(0) @binding(4) var<storage, read_write> sinks: array<u32>;
//#endif

// `scale` at 40 and `window` at 44, because the four strides ahead of them are
// `vec2<u32>` and align to eight. Derived by
// `kernels_wgpu::uniform_layout` from the row's scalar order, not counted; the
// deleted `dump_layout` example only printed that layout.
struct Params {
    gqa_factor: i32,
    n: i32,
    k_head_stride: vec2<u32>,
    k_seq_stride: vec2<u32>,
    v_head_stride: vec2<u32>,
    v_seq_stride: vec2<u32>,
    scale: f32,
    window: i32,
    q_row_stride: i32,
    o_row_stride: i32,
}
@group(1) @binding(0) var<uniform> params: Params;

// One lane per output word.
const PIE_PAIRS: u32 = PIE_HEAD_DIM / 2;

// The bf16 half-index unpack, per buffer. `pie_load_bf16(&queries, i)` is the
// shared answer and cannot be CALLED: its `ptr<storage, array<u32>, read>`
// parameter is WGSL's `unrestricted_pointer_parameters`, which naga does not
// implement, so a module that calls it parses and then fails
// `create_shader_module`. The CONVERSION keeps one definition in
// `common/bf16.inc.wgsl`; only the address arithmetic is restated.
fn q_at(i: u32) -> f32 {
    let word = queries[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}

fn k_at(i: u32) -> f32 {
    let word = keys[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}

fn v_at(i: u32) -> f32 {
    let word = values[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}

//#if defined(PIE_WITH_SINK)
fn sink_at(i: u32) -> f32 {
    let word = sinks[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}
//#endif

fn dot_qk(q_base: u32, k_base: u32) -> f32 {
    var acc = 0.0;
    for (var d = 0u; d < PIE_HEAD_DIM; d = d + 1u) {
        // Scale per term, where the sibling backends put it: a parity walk
        // compares numbers, and hoisting it out changes the rounding.
        acc = acc + params.scale * q_at(q_base + d) * k_at(k_base + d);
    }
    return acc;
}

@compute @workgroup_size(PIE_PAIRS)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let q_head = wg.x;
    let row = wg.y;
    let d_out = lid.x * 2u;
    let n_rows = i32(groups.y);

    // This row's own causal end. The grid's y extent is the batch, and the last
    // row is the one at `n`; every earlier row stops that many keys short.
    let n_row = params.n - (n_rows - 1 - i32(row));
    var kv_start = 0;
    if (params.window > 0 && n_row > params.window) { kv_start = n_row - params.window; }

    let kv_head = q_head / u32(params.gqa_factor);
    // gemma reads its query out of a wider buffer than it writes, so the two
    // pitches are separate and either may be absent -- a non-positive pitch
    // means "packed", which is the grid's own x extent times the head width.
    var q_row = row * groups.x * PIE_HEAD_DIM;
    if (params.q_row_stride > 0) { q_row = row * u32(params.q_row_stride); }
    var o_row = row * groups.x * PIE_HEAD_DIM;
    if (params.o_row_stride > 0) { o_row = row * u32(params.o_row_stride); }
    let q_base = q_row + q_head * PIE_HEAD_DIM;
    let o_base = o_row + q_head * PIE_HEAD_DIM;

    var max_score = PIE_SDPA_NEG_INF;
    var sum_exp = 0.0;
    var acc = vec2<f32>(0.0, 0.0);
    // The window bound is a LOOP bound and not an early return. Nothing in this
    // body barriers, so a return would be safe here -- but the same shape one
    // file over is inside a workgroup that does, and an early return in front
    // of a barrier is a hang rather than a wrong number.
    for (var i = kv_start; i < n_row; i = i + 1) {
        // The low word of each stride, and the high word cannot matter: every
        // term is unsigned, no product exceeds the sum it belongs to, and the
        // sum is an element index into a storage range that is itself 32-bit.
        let k_base = kv_head * params.k_head_stride.x + u32(i) * params.k_seq_stride.x;
        let step = pie_sdpa_online_update(dot_qk(q_base, k_base), max_score, sum_exp);
        max_score = step.max_score;
        sum_exp = step.sum_exp;
        let v_index = kv_head * params.v_head_stride.x + u32(i) * params.v_seq_stride.x + d_out;
        acc = acc * step.history_scale
            + step.score_scale * vec2<f32>(v_at(v_index), v_at(v_index + 1u));
    }

//#if defined(PIE_WITH_SINK)
    // The sink joins the softmax after the last key: it moves the running
    // maximum and the denominator, and contributes nothing to the numerator.
    let merged = pie_sdpa_merge_sink(sink_at(q_head), max_score, sum_exp);
    acc = acc * merged.output_scale;
    sum_exp = merged.sum_exp;
//#endif

    var norm = acc;
    // An empty window: zero over zero is NaN where the reference gives zero.
    if (sum_exp != 0.0) { norm = acc / sum_exp; }
    let at = (o_base + d_out) >> 1u;
    if (at < arrayLength(&out_)) { out_[at] = pie_pack_bf16(norm.x, norm.y); }
}

// pie:instantiate sdpa_vector_decode_swa_bfloat16_d_256 PIE_HEAD_DIM=256
// pie:instantiate sdpa_vector_decode_swa_bfloat16_d_512 PIE_HEAD_DIM=512
// pie:instantiate sdpa_vector_decode_sink_bfloat16_d_64 PIE_HEAD_DIM=64 PIE_WITH_SINK=1
