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

@group(0) @binding(0) var<storage, read_write> queries: array<u32>;
@group(0) @binding(1) var<storage, read_write> k_pages: array<u32>;
@group(0) @binding(2) var<storage, read_write> v_pages: array<u32>;
@group(0) @binding(3) var<storage, read_write> out_: array<u32>;
@group(0) @binding(4) var<storage, read_write> position_ids: array<i32>;
@group(0) @binding(5) var<storage, read_write> req_of_token: array<i32>;
@group(0) @binding(6) var<storage, read_write> kv_page_indices: array<u32>;
@group(0) @binding(7) var<storage, read_write> kv_page_indptr: array<u32>;
// `U8s` in the row: WGSL's smallest storage element is four bytes, so both mask
// buffers cross as `array<u32>` and a byte is a shift.
@group(0) @binding(8) var<storage, read_write> attention_mask: array<u32>;
@group(0) @binding(9) var<storage, read_write> attention_mask_enabled: array<u32>;
@group(0) @binding(10) var<storage, read_write> sinks: array<u32>;

// Both rows STATE these eighteen operands now -- `kernels-metal` stated them
// and the three tables are compared row for row -- so `bindings()` answers for
// this file and a driver binds from the row rather than from a fallback. The
// order is unchanged: it is the one
// `kernels-vulkan/kernels/attn/sdpa_paged_mma.comp` states, which was the
// lowered plan's own order back when nothing could be derived.
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
// Head dimension in vec4 and vec2 units. The tiles are held vectorised: the
// score loop reads a key's whole row and the accumulate reads a value PAIR, so
// four words and two words are the natural grains, and every index below is
// workgroup-uniform or depends on `ly` alone -- broadcast reads, not strided
// ones.
const PIE_DIM4: u32 = PIE_HEAD_DIM / 4u;
const PIE_DIM2: u32 = PIE_HEAD_DIM / 2u;

var<workgroup> k_tile: array<vec4<f32>, PIE_KT * PIE_DIM4>;
var<workgroup> v_tile: array<vec2<f32>, PIE_KT * PIE_DIM2>;
// The segment's queries, staged once.
//
// The score is a 64-term inner product that every x-lane of a row needs whole,
// since each of them owns an output pair rather than a piece of the score.
// Read from global that was 64 bf16 unpacks per lane per key, which for a
// 512-token prefill is where this kernel spent most of its remaining time.
// Staged, it is a workgroup broadcast.
var<workgroup> q_tile: array<vec4<f32>, 8u * PIE_DIM4>;
// THE TILE'S SCORES, ONE PER (ROW, KEY), COMPUTED ONCE.
//
// Each x-lane needing the whole score is not the same thing as each x-lane
// COMPUTING it, and this kernel used to do the second: all 32 of a row's
// lanes ran the same 64-term product over the same two staged tiles, so the
// workgroup retired 32 scores' worth of arithmetic per score. Against the
// value accumulation -- which really is per lane, one output pair each -- the
// query-key product was thirty-two thirty-thirds of the inner loop's work and
// one thirty-third of its result.
//
// A segment is at most 8 rows and a tile is PIE_KT keys, so the whole score
// rectangle is 128 numbers: fewer than the workgroup has lanes, computed by
// one flat pass and read by everyone after a barrier. The arithmetic inside
// the pass is the SAME four statements in the same order the per-lane loop
// used, because the parity walk against `kernels-metal` and `kernels-vulkan`
// compares the numbers and not the schedule.
var<workgroup> s_tile: array<f32, 8 * PIE_KT>;

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
// otherwise run this. The group still owns 32 rows -- the y lanes take eight of
// them at a time -- so the host's `ceil(n_rows / 32)` y grid is unchanged.
//
// ## Eight rows per sweep, not one
//
// The tiling above is only worth staging if something reuses the tile. This
// body used to sweep the whole key range ONCE PER ROW -- the `base` loop sat
// inside a `for rr in 0..32` -- so a 16-key tile was read from the paged cache
// thirty-two times, and during the multiply only the eighth of the lanes that
// owned that row (`ly == rr & 7`) did anything while the other 224 sat at the
// barrier. Staging bought nothing it did not immediately throw away.
//
// Measured on an M4, Llama-3.2-1B, one 512-token prefill: 6.9 SECONDS in this
// entrypoint, 64% of the whole fire, for 2.1 GFLOP of arithmetic per layer.
//
// So the key sweep is the outer loop and the eight y-lanes carry eight
// DIFFERENT rows through it. The tile is read once for all eight, and every
// lane multiplies. The row's own `q_pos` still bounds its keys -- `keeps`
// already masked per row, and a segment sweeps to the widest position in it,
// so a shorter row simply contributes nothing past its own.
//
// ## Why a segment and not always eight
//
// A shared tile is a shared REQUEST: `page_slot` resolves through that
// request's page table, and eight rows of one workgroup need not come from one
// request. So the width is measured rather than assumed -- eight rows if they
// agree, one if they do not, which is exactly the old behaviour for the at most
// one straddling group a batch boundary produces. Segment width is read from
// storage at workgroup-uniform indices, so the `while` it drives, and the
// barriers inside it, stay in uniform control flow.
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

    var rg = 0u;
    while (rg < 32u) {
        // How many of the next rows share one request, up to the eight y-lanes.
        // A row past `n_rows` constrains nothing: it is skipped here and its
        // lane is idle below.
        var seg_req = -1;
        var width = 8u;
        for (var j = 0u; j < 8u; j = j + 1u) {
            if (rg + j >= 32u) { break; }
            let row = row_lo + rg + j;
            if (row >= u32(params.n_rows)) { continue; }
            let r = req_of_token[row];
            if (seg_req < 0) {
                seg_req = r;
            } else if (r != seg_req) {
                width = 1u;
                break;
            }
        }
        width = min(width, 32u - rg);

        // The sweep's bounds are the SEGMENT's: the widest position any row in
        // it attends to, and the earliest any row's window opens.
        var q_max = -1;
        var g_start = 0x7fffffff;
        for (var j = 0u; j < width; j = j + 1u) {
            let row = row_lo + rg + j;
            if (row >= u32(params.n_rows)) { continue; }
            let p = position_ids[row];
            q_max = max(q_max, p);
            var s = 0;
            if (params.window > 0 && p >= params.window) { s = p - params.window + 1; }
            g_start = min(g_start, s);
        }
        // A segment with no valid row sweeps nothing; the start would otherwise
        // be a sentinel that overflows the moment `PIE_KT` is added to it.
        if (q_max < 0) { g_start = 0; }

        // This lane's row. Non-uniform, and therefore never a barrier's
        // condition -- every `if (mine)` below is barrier-free.
        let row = row_lo + rg + ly;
        let mine = ly < width && (rg + ly) < 32u && row < u32(params.n_rows);
        var q_pos = -1;
        var start = 0;
        if (mine) {
            q_pos = position_ids[row];
            if (params.window > 0 && q_pos >= params.window) { start = q_pos - params.window + 1; }
        }
        let q_base = (row * n_q_heads + q_head) * PIE_HEAD_DIM;

        // Stage the segment's queries. The barrier in front guards against a
        // lane still reading the previous segment's tiles; the one behind
        // publishes this segment's.
        workgroupBarrier();
        for (var e = flat; e < 8u * PIE_DIM4; e = e + 256u) {
            let j = e / PIE_DIM4;
            let d4 = (e - j * PIE_DIM4) * 4u;
            let rr = row_lo + rg + j;
            var qv = vec4<f32>(0.0, 0.0, 0.0, 0.0);
            if (j < width && (rg + j) < 32u && rr < u32(params.n_rows)) {
                let at = (rr * n_q_heads + q_head) * PIE_HEAD_DIM + d4;
                qv = vec4<f32>(q_at(at), q_at(at + 1u), q_at(at + 2u), q_at(at + 3u));
            }
            q_tile[e] = qv;
        }
        workgroupBarrier();

        var max_score = PIE_SDPA_NEG_INF;
        var sum_exp = 0.0;
        var acc: array<vec2<f32>, PIE_SLOTS>;
        for (var s = 0u; s < PIE_SLOTS; s = s + 1u) { acc[s] = vec2<f32>(0.0, 0.0); }

        // `q_max` and `g_start` are read from storage at indices built out of
        // the workgroup id and the uniform loop counters, so this loop's trip
        // count is workgroup-uniform and the barriers inside it are reached by
        // everyone the same number of times.
        for (var base = g_start; base <= q_max; base = base + PIE_KT) {
            let cnt = min(PIE_KT, q_max + 1 - base);
            for (var e = flat; e < u32(PIE_KT) * PIE_DIM4; e = e + 256u) {
                let kk = e / PIE_DIM4;
                let d4 = (e - kk * PIE_DIM4) * 4u;
                var kv = vec4<f32>(0.0, 0.0, 0.0, 0.0);
                if (i32(kk) < cnt && seg_req >= 0) {
                    let slot = page_slot(seg_req, base + i32(kk));
                    let off = (slot * u32(params.n_kv_heads) + u32(kv_head)) * PIE_HEAD_DIM + d4;
                    kv = vec4<f32>(k_at(off), k_at(off + 1u), k_at(off + 2u), k_at(off + 3u));
                }
                // The tail of the last tile is ZEROED rather than left stale: a
                // key beyond `q_pos` is skipped by `keeps`, but `v_tile` is
                // read for every slot a lane owns and a stale value would
                // survive into the next tile's accumulation.
                k_tile[e] = kv;
            }
            for (var e = flat; e < u32(PIE_KT) * PIE_DIM2; e = e + 256u) {
                let kk = e / PIE_DIM2;
                let d2 = (e - kk * PIE_DIM2) * 2u;
                var vv = vec2<f32>(0.0, 0.0);
                if (i32(kk) < cnt && seg_req >= 0) {
                    let slot = page_slot(seg_req, base + i32(kk));
                    let off = (slot * u32(params.n_kv_heads) + u32(kv_head)) * PIE_HEAD_DIM + d2;
                    vv = vec2<f32>(v_at(off), v_at(off + 1u));
                }
                v_tile[e] = vv;
            }
            // The fill must be complete before anybody reads it.
            workgroupBarrier();

            // One lane per (row, key) of the tile, and 128 of the 256 have one.
            // Keys past `cnt` are staged as zero and their scores are never
            // read, so this pass carries no bound of its own and stays
            // workgroup-uniform -- which is what lets the barrier behind it be
            // reached by everybody.
            for (var e = flat; e < 8u * u32(PIE_KT); e = e + 256u) {
                let j = e / u32(PIE_KT);
                let kk = e - j * u32(PIE_KT);
                var score = 0.0;
                for (var d4 = 0u; d4 < PIE_DIM4; d4 = d4 + 1u) {
                    // Scale per term, where the sibling backends put it: a
                    // parity walk compares numbers and hoisting it changes
                    // the rounding. The four terms stay separate statements
                    // for the same reason -- this is the scalar loop with
                    // its loads batched, not a dot product.
                    let qv = q_tile[j * PIE_DIM4 + d4];
                    let kv = k_tile[kk * PIE_DIM4 + d4];
                    score = score + params.scale * qv.x * kv.x;
                    score = score + params.scale * qv.y * kv.y;
                    score = score + params.scale * qv.z * kv.z;
                    score = score + params.scale * qv.w * kv.w;
                }
                s_tile[e] = score;
            }
            workgroupBarrier();

            if (mine) {
                for (var kk = 0; kk < cnt; kk = kk + 1) {
                    let kp = base + kk;
                    if (!keeps(row, kp, q_pos, start)) { continue; }
                    let score = s_tile[ly * u32(PIE_KT) + u32(kk)];
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
                            let at = u32(kk) * PIE_DIM2 + (d_out >> 1u);
                            acc[s] = acc[s] * step.history_scale
                                + step.score_scale * v_tile[at];
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

        rg = rg + width;
    }
}

// pie:instantiate sdpa_paged_mma_bfloat16_d_64 PIE_HEAD_DIM=64
// pie:instantiate sdpa_paged_mma_sink_bfloat16_d_64 PIE_HEAD_DIM=64 PIE_WITH_SINK=1
