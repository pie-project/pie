// Dense decode SDPA: one query row against a contiguous KV cache.
//
// The Metal path splits the keys across simdgroups and merges with subgroup
// reductions. This body keeps the same online softmax and assigns one output
// WORD to one invocation instead, which is deliberately subgroup-width agnostic
// -- WebGPU does not guarantee `subgroupAdd` at all, and a body that needed it
// would be a body that runs on some adapters.
//
// ## Why a lane owns two channels and not one
//
// bf16 crosses as `array<u32>`, two values to a word, so a lane that owned one
// channel would read-modify-write a word its neighbour writes at the same
// moment: WGSL has no sub-word atomic and the store would be lost. A lane owns
// the PAIR, accumulates it as a `vec2<f32>`, and writes a whole word. That also
// halves the redundant Q.K work, and it keeps the workgroup at or under 256
// invocations for every point of the head-dim axis -- WebGPU's guaranteed
// ceiling, which a 512-wide `@workgroup_size` would exceed on hardware that
// would otherwise have run it.

//#include "common/bf16.inc.wgsl"
//#include "attn/sdpa_online.inc.wgsl"

@group(0) @binding(0) var<storage, read_write> queries: array<u32>;
@group(0) @binding(1) var<storage, read_write> keys: array<u32>;
@group(0) @binding(2) var<storage, read_write> values: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_: array<u32>;

// Two `i32`s, then four 64-bit strides, then a float: 0, 4, 8, 16, 24, 32, 40.
// A `vec2<u32>` aligns to eight, so `k_head_stride` starts at 8 and not at 8
// by accident -- `n` at 4 is followed by no padding only because 8 is already
// the next multiple. Derived by `kernels_wgpu::uniform_layout` from the row's
// scalar order, not counted; the deleted `dump_layout` example only printed
// that layout.
struct Params {
    gqa_factor: i32,
    n: i32,
    k_head_stride: vec2<u32>,
    k_seq_stride: vec2<u32>,
    v_head_stride: vec2<u32>,
    v_seq_stride: vec2<u32>,
    scale: f32,
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

// The whole Q.K^T for one key, recomputed by every lane of the workgroup.
//
// This used to say it was "the Vulkan baseline's shape and kept on purpose",
// and both halves have expired. `kernels-vulkan` reduces cooperatively here
// now, and it measured the rationale false rather than arguing with it: 67.7
// ms to 42.5 ms at d_128 with 16 heads, timing n=4096 against n=256 so the
// difference is the key loop and not the buffer setup.
//
// It is NOT fixed here, and that is a decision with a reason rather than an
// omission. `sdpa_paged`'s decode arm took the fix in the same commit that
// wrote this comment, because that is the arm every fire in the curated suite
// takes: `model-compiler`'s `sdpa` lowers to this kernel only when `paged` is
// false, and the model text this backend runs is paged throughout.
// `driver-wgpu` binds and dispatches this module if a plan names it -- see
// `dispatch.rs` -- so it is reachable and unexercised, which is the worst
// place to make an unmeasurable change. `driver-vulkan` says the same of its
// twin: "the dense decode, off this model's path".
//
// If a dense entrypoint ever arrives, the fix is `sdpa_paged`'s `decode_row`,
// which is uniform for the same reason this kernel would be -- one workgroup
// is one row and one head.
fn dot_qk(q_base: u32, k_base: u32) -> f32 {
    var acc = 0.0;
    for (var d = 0u; d < PIE_HEAD_DIM; d = d + 1u) {
        // The scale rides inside the loop rather than multiplying the sum,
        // because that is where `kernels-metal` and `kernels-vulkan` put it and
        // the two roundings differ. A parity walk compares numbers.
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
    // The low channel of the pair this lane owns.
    let d_out = lid.x * 2u;
    let n_rows = groups.y;
    let kv_head = q_head / u32(params.gqa_factor);
    // `[head][row][channel]`, which is the dense decode layout the sibling
    // backends address and not the `[row][head][channel]` the paged one does.
    let q_offset = (q_head * n_rows + row) * PIE_HEAD_DIM;

    var max_score = PIE_SDPA_NEG_INF;
    var sum_exp = 0.0;
    var acc = vec2<f32>(0.0, 0.0);
    for (var i = 0; i < params.n; i = i + 1) {
        // Only the LOW word of each stride is used. The high word cannot
        // matter: every term here is unsigned, so no product exceeds the sum it
        // belongs to, and the sum is an element index into a bound storage
        // range -- itself a 32-bit quantity. The ABI carries 64 bits because it
        // is shared with `kernels-metal`, where a >4 GiB buffer makes it real.
        let k_base = kv_head * params.k_head_stride.x + u32(i) * params.k_seq_stride.x;
        let step = pie_sdpa_online_update(dot_qk(q_offset, k_base), max_score, sum_exp);
        max_score = step.max_score;
        sum_exp = step.sum_exp;
        let v_index = kv_head * params.v_head_stride.x + u32(i) * params.v_seq_stride.x + d_out;
        acc = acc * step.history_scale
            + step.score_scale * vec2<f32>(v_at(v_index), v_at(v_index + 1u));
    }

    var norm = acc;
    // A row with no keys at all: the denominator is exactly zero and the
    // numerator is too, so the quotient would be NaN rather than the zero the
    // reference produces.
    if (sum_exp != 0.0) { norm = acc / sum_exp; }
    let at = (q_offset + d_out) >> 1u;
    if (at < arrayLength(&out_)) { out_[at] = pie_pack_bf16(norm.x, norm.y); }
}

// pie:instantiate sdpa_vector_decode_bfloat16_d_64 PIE_HEAD_DIM=64
// pie:instantiate sdpa_vector_decode_bfloat16_d_128 PIE_HEAD_DIM=128
// pie:instantiate sdpa_vector_decode_bfloat16_d_256 PIE_HEAD_DIM=256
