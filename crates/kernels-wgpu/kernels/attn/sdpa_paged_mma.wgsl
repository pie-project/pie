// Paged prefill SDPA at the Metal MMA entrypoint names.
//
// ## The name is an ABI point, not a hardware claim
//
// `kernels-metal` runs this with the matrix unit and `kernels-vulkan`'s
// equivalent is `GL_KHR_cooperative_matrix`, which it compiles as a `@coopmat`
// TIER over a scalar baseline. WebGPU's equivalent is
// `wgpu::Features::SUBGROUP_MATRIX`, and this crate has no tier for it:
// `capability.rs` offers Baseline, Fp16 and Subgroup, where Subgroup is
// `subgroupAdd`-class reductions and nothing else, deliberately -- a tier is a
// promise about a BODY, and two bodies must not share one.
//
// So this file is the baseline and only the baseline: a tiled scalar path with
// shared K/V tiles, f32 accumulation and the same online softmax the rest of
// the family runs. `_mma` here reads exactly like `_p32` and `_sg8` do one file
// over -- a name the table inherited, kept so the coverage is row-for-row, with
// a body that claims nothing about the device.
//
// ## What the tiling buys, since it is not a matrix unit
//
// The K and V rows of a 16-key tile are read from the paged cache ONCE per
// workgroup and shared, instead of once per lane. The scalar body in
// `sdpa_paged.wgsl` re-reads them per lane, which is what makes this the
// prefill body: a prefill dispatch has 32 rows of work per group to amortise
// the staging over.
//
// The tiles are staged as `f32` rather than as packed bf16 words. Workgroup
// memory has no addressing problem to solve -- there is no 4 GiB range and no
// two-values-per-word constraint once the data is off the buffer -- and two
// 16x64 f32 tiles are 8 KiB of the 16 KiB WebGPU guarantees. Storing them
// packed would save 4 KiB and cost every read a shift and a mask.

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
// `U8s` in the row: WGSL's smallest storage element is four bytes, so both mask
// buffers cross as `array<u32>` and a byte is a shift.
@group(0) @binding(8) var<storage, read> attention_mask: array<u32>;
@group(0) @binding(9) var<storage, read> attention_mask_enabled: array<u32>;
@group(0) @binding(10) var<storage, read> sinks: array<u32>;

// Both rows here are UNSTATED, so no `bindings()` answer covers them; this is
// the order `kernels-vulkan/kernels/attn/sdpa_paged_mma.comp` states, which is
// the lowered plan's own order a driver falls back to.
struct Params {
    gqa_factor: i32,
    page_size: i32,
    n_kv_heads: i32,
    scale: f32,
    attention_mask_stride: u32,
    window: i32,
    n_rows: i32,
}
@group(1) @binding(0) var<uniform> params: Params;

// Keys per tile.
const PIE_KT = 16;
// Output WORDS per row: a lane owns a bf16 PAIR, because a half-word store is a
// read-modify-write and WGSL has no sub-word atomic.
const PIE_PAIRS: u32 = PIE_HEAD_DIM / 2;
// How many of those one x-lane owns, over 32 x-lanes.
const PIE_SLOTS: u32 = (PIE_PAIRS + 31u) / 32u;

var<workgroup> k_tile: array<f32, PIE_KT * PIE_HEAD_DIM>;
var<workgroup> v_tile: array<f32, PIE_KT * PIE_HEAD_DIM>;

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
    let page_ix = u32(kp / params.page_size);
    let page_off = u32(kp % params.page_size);
    let phys = kv_page_indices[kv_page_indptr[u32(req)] + page_ix];
    return phys * u32(params.page_size) + page_off;
}

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
        // Past the mask's own stride there is no entry to read, and reading on
        // would pick up the next row's mask.
        if (u32(kp) >= params.attention_mask_stride) { return false; }
        if (!mask_allows(row * params.attention_mask_stride + u32(kp))) { return false; }
    }
    return true;
}

// 32 x 8, not 32 x 32: WebGPU guarantees 256 invocations per workgroup, and the
// 1024 a 32x32 group asks for fails pipeline creation on hardware that would
// otherwise run this. The group still owns 32 rows -- the y lanes sweep them
// four at a time -- so the host's `ceil(n_rows / 32)` y grid is unchanged.
@compute @workgroup_size(32, 8)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
    @builtin(num_workgroups) groups: vec3<u32>,
) {
    let q_head = wg.x;
    let row_lo = wg.y * 32u;
    let lx = lid.x;
    let ly = lid.y;
    let flat = ly * 32u + lx;
    let n_q_heads = groups.x;
    let kv_head = i32(q_head) / params.gqa_factor;

    for (var rr = 0u; rr < 32u; rr = rr + 1u) {
        let row = row_lo + rr;
        let row_valid = row < u32(params.n_rows);
        // Who may WRITE this row's output -- not who computes. Every lane runs
        // the sweep and every lane reaches every barrier; the row bound is a
        // guard on the STORE and never an early return, because
        // `workgroupBarrier()` must sit in control flow uniform across the
        // workgroup and a return in front of one is a hang, not a wrong number.
        let mine = ly == (rr & 7u) && row_valid;

        var req = 0;
        var q_pos = -1;
        if (row_valid) {
            req = req_of_token[row];
            q_pos = position_ids[row];
        }
        var start = 0;
        if (params.window > 0 && q_pos >= params.window) { start = q_pos - params.window + 1; }
        let q_base = (row * n_q_heads + q_head) * PIE_HEAD_DIM;

        var max_score = PIE_SDPA_NEG_INF;
        var sum_exp = 0.0;
        var acc: array<vec2<f32>, PIE_SLOTS>;
        for (var s = 0u; s < PIE_SLOTS; s = s + 1u) { acc[s] = vec2<f32>(0.0, 0.0); }

        // `q_pos` and `start` come from read-only storage and the uniform block
        // and are the same for every lane at this `rr`, so this loop's trip
        // count is workgroup-uniform and the barriers inside it are reached by
        // everyone the same number of times. An invalid row leaves `q_pos` at
        // -1 and runs it zero times, uniformly.
        for (var base = start; base <= q_pos; base = base + PIE_KT) {
            let cnt = min(PIE_KT, q_pos + 1 - base);
            for (var e = flat; e < u32(PIE_KT * PIE_HEAD_DIM); e = e + 256u) {
                let kk = e / PIE_HEAD_DIM;
                let d = e - kk * PIE_HEAD_DIM;
                var k_v = 0.0;
                var v_v = 0.0;
                if (i32(kk) < cnt) {
                    let slot = page_slot(req, base + i32(kk));
                    let off = (slot * u32(params.n_kv_heads) + u32(kv_head)) * PIE_HEAD_DIM + d;
                    k_v = k_at(off);
                    v_v = v_at(off);
                }
                // The tail of the last tile is ZEROED rather than left stale: a
                // key beyond `q_pos` is skipped by `keeps`, but `v_tile` is
                // read for every slot a lane owns and a stale value would
                // survive into the next tile's accumulation.
                k_tile[e] = k_v;
                v_tile[e] = v_v;
            }
            // The fill must be complete before anybody reads it.
            workgroupBarrier();

            if (mine) {
                for (var kk = 0; kk < cnt; kk = kk + 1) {
                    let kp = base + kk;
                    if (!keeps(row, kp, q_pos, start)) { continue; }
                    var score = 0.0;
                    for (var d = 0u; d < PIE_HEAD_DIM; d = d + 1u) {
                        // Scale per term, where the sibling backends put it: a
                        // parity walk compares numbers and hoisting it changes
                        // the rounding.
                        score = score + params.scale
                            * q_at(q_base + d)
                            * k_tile[u32(kk) * PIE_HEAD_DIM + d];
                    }
                    let step = pie_sdpa_online_update(score, max_score, sum_exp);
                    max_score = step.max_score;
                    sum_exp = step.sum_exp;
                    // Every output slot this lane owns is rescaled by the same
                    // history factor and takes the same key's weight, which is
                    // why the running max and sum are per LANE and not per
                    // slot: the key loop is shared and the accumulator is not.
                    for (var s = 0u; s < PIE_SLOTS; s = s + 1u) {
                        let d_out = (lx + s * 32u) * 2u;
                        if (d_out < PIE_HEAD_DIM) {
                            let at = u32(kk) * PIE_HEAD_DIM + d_out;
                            acc[s] = acc[s] * step.history_scale
                                + step.score_scale * vec2<f32>(v_tile[at], v_tile[at + 1u]);
                        }
                    }
                }
            }
            // The load-bearing one: without it a lane that has reached the next
            // tile's fill overwrites a tile another lane is still reading. It
            // is the barrier that looks removable and is not.
            workgroupBarrier();
        }

        if (mine) {
//#if defined(PIE_WITH_SINK)
            // A per-head learned logit with no value behind it: it moves the
            // denominator and rescales the numerator, once, after the last key.
            let merged = pie_sdpa_merge_sink(sink_at(q_head), max_score, sum_exp);
            sum_exp = merged.sum_exp;
            for (var s = 0u; s < PIE_SLOTS; s = s + 1u) { acc[s] = acc[s] * merged.output_scale; }
//#endif
            for (var s = 0u; s < PIE_SLOTS; s = s + 1u) {
                let d_out = (lx + s * 32u) * 2u;
                if (d_out < PIE_HEAD_DIM) {
                    var norm = acc[s];
                    // A fully masked row keeps a zero denominator, and zero
                    // over zero is NaN where the reference gives zero.
                    if (sum_exp != 0.0) { norm = acc[s] / sum_exp; }
                    let at = (q_base + d_out) >> 1u;
                    if (at < arrayLength(&out_)) { out_[at] = pie_pack_bf16(norm.x, norm.y); }
                }
            }
        }
    }
}

// pie:instantiate sdpa_paged_mma_bfloat16_d_64 PIE_HEAD_DIM=64
// pie:instantiate sdpa_paged_mma_sink_bfloat16_d_64 PIE_HEAD_DIM=64 PIE_WITH_SINK=1
