// Paged attention, read side: decode and tiled prefill from one body.
//
// The page-table arithmetic is Metal's, unchanged: request -> CSR page list ->
// physical page + in-page offset, then NHD element addressing. Eleven storage
// buffers, which is over WebGPU's guaranteed floor of eight -- that is the
// HOST's problem (it must request the adapter's real limits rather than
// `downlevel_defaults`, see `over_downlevel_storage_limit`) and not a reason to
// bind fewer. Six of the eleven are the fire's own tables and the ROW is the
// only place they are written down:
//
//     cargo run -p kernels-wgpu --example dump_layout -- sdpa_paged_decode
//
// The numbers below come from that. They are NOT Metal's, because this backend
// sends the row's five interleaved scalars to a uniform block and the buffer
// run closes up around them: `attention_mask` is the row's thirteenth operand
// and this file's binding 8.
//
// `_p32` and `_sg8` are ABI points inherited from Metal's table, not claims
// about hardware. `_p32` compiles the page arithmetic against a page size of
// 32 (a shift instead of a division); `_sg8` names a subgroup width this body
// does not read, because nothing here is subgroup-shaped.
//
// One lane owns a channel PAIR: bf16 crosses as `array<u32>`, two values to a
// word, and a lane owning one channel would read-modify-write a word its
// neighbour writes at the same instant with no sub-word atomic to arbitrate.
// It also keeps every workgroup at or under WebGPU's guaranteed 256
// invocations -- at `d_512` this body runs exactly 256 lanes where a
// channel-per-lane body would ask for 512 and fail to create a pipeline.

//#include "common/bf16.inc.wgsl"
//#include "attn/sdpa_online.inc.wgsl"

@group(0) @binding(0) var<storage, read> queries: array<u32>;
@group(0) @binding(1) var<storage, read> k_pages: array<u32>;
@group(0) @binding(2) var<storage, read> v_pages: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_: array<u32>;
@group(0) @binding(4) var<storage, read> position_ids: array<i32>;
@group(0) @binding(5) var<storage, read> req_of_token: array<i32>;
@group(0) @binding(6) var<storage, read> kv_page_indices: array<u32>;
@group(0) @binding(7) var<storage, read> kv_page_indptr: array<u32>;
// `U8s` in the row, and WGSL has no eight-bit storage element any more than it
// has a sixteen-bit one -- the smallest is a `u32`. Both mask buffers are
// therefore four bytes to a word and a byte is a shift, the same divergence
// bf16 makes and for the same reason.
@group(0) @binding(8) var<storage, read> attention_mask: array<u32>;
@group(0) @binding(9) var<storage, read> attention_mask_enabled: array<u32>;
@group(0) @binding(10) var<storage, read> sinks: array<u32>;

// The row's six scalars in ROW order, at 0, 4, 8, 12, 16 and 20. All four
// bytes wide, so this block is the one place in the family where the naive sum
// of widths happens to be the right answer -- `kv_write`'s is not, and the
// difference is `Usize`.
struct Params {
    gqa_factor: i32,
    page_size: i32,
    n_kv_heads: i32,
    scale: f32,
    attention_mask_stride: u32,
    window: i32,
//#if defined(PIE_TILED)
    // The tiled rows are UNSTATED -- the table names no operands for
    // `sdpa_paged_tiled` -- so these three are not derivable from a row, and
    // they are appended here in the order `kernels-vulkan`'s push block states
    // them. A driver launching an unstated row follows the lowered plan's own
    // argument order, and this is that order.
    n_rows: i32,
//#if defined(PIE_STRIDED)
    q_row_pitch: i32,
    o_row_pitch: i32,
//#endif
//#endif
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
    let word = k_pages[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}

fn v_at(i: u32) -> f32 {
    let word = v_pages[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}

fn sink_at(i: u32) -> f32 {
    let word = sinks[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}

fn page_slot(req: i32, kp: i32) -> u32 {
//#if defined(PIE_PAGE_SIZE) && PIE_PAGE_SIZE == 32
    // The `_p32` points: a 32-entry page is a shift and a mask, and the
    // division this replaces is the inner loop's only integer divide.
    let page_ix = u32(kp) >> 5u;
    let page_off = u32(kp) & 31u;
    let phys = kv_page_indices[kv_page_indptr[u32(req)] + page_ix];
    return phys * 32u + page_off;
//#else
    let page_ix = u32(kp / params.page_size);
    let page_off = u32(kp % params.page_size);
    let phys = kv_page_indices[kv_page_indptr[u32(req)] + page_ix];
    return phys * u32(params.page_size) + page_off;
//#endif
}

// One byte out of a `u32` array. Module-scope storage is addressable from any
// function here, so these take the row index rather than a pointer -- a
// `ptr<storage, ...>` parameter is a WGSL language extension `naga` does not
// implement.
fn mask_enabled(row: u32) -> bool {
    let word = attention_mask_enabled[row >> 2u];
    return ((word >> ((row & 3u) * 8u)) & 0xffu) != 0u;
}

fn mask_allows(at: u32) -> bool {
    let word = attention_mask[at >> 2u];
    return ((word >> ((at & 3u) * 8u)) & 0xffu) != 0u;
}

fn keeps(row: u32, kp: i32, q_pos: i32, start: i32) -> bool {
    if (kp > q_pos || kp < start) { return false; }
    if (mask_enabled(row)) {
        // The mask's own stride bounds it: a request whose history is longer
        // than the mask the fire supplied has no entry to read, and reading
        // past the row would pick up the NEXT row's mask.
        if (u32(kp) >= params.attention_mask_stride) { return false; }
        if (!mask_allows(row * params.attention_mask_stride + u32(kp))) { return false; }
    }
    return true;
}

fn q_base_for(row: u32, q_head: u32, n_q_heads: u32) -> u32 {
//#if defined(PIE_TILED) && defined(PIE_STRIDED)
    var base = row * n_q_heads * PIE_HEAD_DIM;
    if (params.q_row_pitch > 0) { base = row * u32(params.q_row_pitch); }
    return base + q_head * PIE_HEAD_DIM;
//#else
    return (row * n_q_heads + q_head) * PIE_HEAD_DIM;
//#endif
}

fn o_base_for(row: u32, q_head: u32, n_q_heads: u32) -> u32 {
//#if defined(PIE_TILED) && defined(PIE_STRIDED)
    var base = row * n_q_heads * PIE_HEAD_DIM;
    if (params.o_row_pitch > 0) { base = row * u32(params.o_row_pitch); }
    return base + q_head * PIE_HEAD_DIM;
//#else
    return (row * n_q_heads + q_head) * PIE_HEAD_DIM;
//#endif
}

fn dot_page(q_base: u32, req: i32, kv_head: i32, kp: i32) -> f32 {
    let slot = page_slot(req, kp);
    let k_base = (slot * u32(params.n_kv_heads) + u32(kv_head)) * PIE_HEAD_DIM;
    var acc = 0.0;
    for (var d = 0u; d < PIE_HEAD_DIM; d = d + 1u) {
        // Scale per term, where `kernels-metal` and `kernels-vulkan` put it.
        // Hoisting it out of the loop is a different rounding, and a parity
        // walk between backends compares numbers.
        acc = acc + params.scale * q_at(q_base + d) * k_at(k_base + d);
    }
    return acc;
}

// One output word: the pair `(d_out, d_out + 1)` of one (row, head).
fn compute_one(row: u32, q_head: u32, d_out: u32, n_q_heads: u32) {
    let req = req_of_token[row];
    let q_pos = position_ids[row];
//#if defined(PIE_FAST_FULL)
    // The `_p32` points attend over the whole history by construction, so the
    // window arithmetic is compiled out rather than evaluated to zero.
    var start = 0;
//#else
    var start = 0;
    if (params.window > 0 && q_pos >= params.window) { start = q_pos - params.window + 1; }
//#endif
    let q_base = q_base_for(row, q_head, n_q_heads);
    let o_base = o_base_for(row, q_head, n_q_heads);
    let kv_head = i32(q_head) / params.gqa_factor;

    var max_score = PIE_SDPA_NEG_INF;
    var sum_exp = 0.0;
    var acc = vec2<f32>(0.0, 0.0);
    for (var kp = start; kp <= q_pos; kp = kp + 1) {
        if (!keeps(row, kp, q_pos, start)) { continue; }
        let step = pie_sdpa_online_update(dot_page(q_base, req, kv_head, kp), max_score, sum_exp);
        max_score = step.max_score;
        sum_exp = step.sum_exp;
        let slot = page_slot(req, kp);
        let v_index = (slot * u32(params.n_kv_heads) + u32(kv_head)) * PIE_HEAD_DIM + d_out;
        acc = acc * step.history_scale
            + step.score_scale * vec2<f32>(v_at(v_index), v_at(v_index + 1u));
    }
//#if defined(PIE_WITH_SINK)
    // gpt-oss's per-head learned logit: it joins the softmax with no value
    // behind it, moving the denominator and nothing else.
    let merged = pie_sdpa_merge_sink(sink_at(q_head), max_score, sum_exp);
    acc = acc * merged.output_scale;
    sum_exp = merged.sum_exp;
//#endif

    var norm = acc;
    // A masked-out row keeps a zero denominator, and zero over zero is NaN
    // where the reference gives zero.
    if (sum_exp != 0.0) { norm = acc / sum_exp; }
    let at = (o_base + d_out) >> 1u;
    if (at < arrayLength(&out_)) { out_[at] = pie_pack_bf16(norm.x, norm.y); }
}

//#if defined(PIE_TILED)

// 32 x 8 and not 32 x 32: WebGPU guarantees only 256 invocations per workgroup,
// so the y lanes sweep the group's 32 rows four at a time instead of one lane
// per row. The GROUP still covers 32 rows, which keeps the host's grid
// arithmetic -- `ceil(n_rows / 32)` in y -- exactly as `kernels-vulkan` states
// it. Nothing in this arm barriers, so the `continue` below is a skip and not a
// hang.
@compute @workgroup_size(32, 8)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let q_head = wg.x;
    for (var rr = lid.y; rr < 32u; rr = rr + 8u) {
        let row = wg.y * 32u + rr;
        if (row >= u32(params.n_rows)) { continue; }
        for (var pair = lid.x; pair < PIE_PAIRS; pair = pair + 32u) {
            compute_one(row, q_head, pair * 2u, groups.x);
        }
    }
}

//#else

@compute @workgroup_size(PIE_PAIRS)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    compute_one(wg.y, wg.x, lid.x * 2u, groups.x);
}

//#endif

// pie:instantiate sdpa_paged_decode_bfloat16_d_64 PIE_HEAD_DIM=64
// pie:instantiate sdpa_paged_decode_bfloat16_d_128 PIE_HEAD_DIM=128
// pie:instantiate sdpa_paged_decode_bfloat16_d_256 PIE_HEAD_DIM=256
// pie:instantiate sdpa_paged_decode_bfloat16_d_512 PIE_HEAD_DIM=512
// pie:instantiate sdpa_paged_decode_bfloat16_d_64_p32 PIE_HEAD_DIM=64 PIE_PAGE_SIZE=32 PIE_FAST_FULL=1
// pie:instantiate sdpa_paged_decode_bfloat16_d_128_p32 PIE_HEAD_DIM=128 PIE_PAGE_SIZE=32 PIE_FAST_FULL=1
// pie:instantiate sdpa_paged_decode_bfloat16_d_64_p32_sg8 PIE_HEAD_DIM=64 PIE_PAGE_SIZE=32 PIE_FAST_FULL=1 PIE_SHORT_GROUP=8
// pie:instantiate sdpa_paged_decode_sink_bfloat16_d_64 PIE_HEAD_DIM=64 PIE_WITH_SINK=1
// pie:instantiate sdpa_paged_tiled_bfloat16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1
// pie:instantiate sdpa_paged_tiled_bfloat16_d_128 PIE_HEAD_DIM=128 PIE_TILED=1
// pie:instantiate sdpa_paged_tiled_bfloat16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1
// pie:instantiate sdpa_paged_tiled_bfloat16_d_512 PIE_HEAD_DIM=512 PIE_TILED=1
// pie:instantiate sdpa_paged_tiled_sink_bfloat16_d_64 PIE_HEAD_DIM=64 PIE_TILED=1 PIE_WITH_SINK=1
// pie:instantiate sdpa_paged_tiled_strided_bfloat16_d_256 PIE_HEAD_DIM=256 PIE_TILED=1 PIE_STRIDED=1
