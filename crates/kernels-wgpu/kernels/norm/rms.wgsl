// RMSNorm: `out = w * (x / rms(x))`, one workgroup per row.
//
// The WGSL port of `kernels-vulkan/kernels/norm/rms.comp`, which is the Vulkan
// port of `kernels-metal/kernels/norm/rms.metal`, which is MLX's
// `rms_single_row` scoped to this driver. Five entrypoints off one body: the
// plain row, the strided row a prefill wants, the two-level strided head row
// the per-head q/k norms want, and the two residual-folding forms gemma4 wants.
//
// ## What moved, and what did not
//
// The arithmetic did not. The accumulate is fp32, the row is walked in chunks
// of `workgroup * N_READS` so a 5120-wide hidden does not ask for a 1280-lane
// workgroup no implementation allows, and the gain is applied in float before
// the single bf16 round on the way out.
//
// Three things moved.
//
// **Where the scalars ride.** Metal binds five `constant` references from
// buffer 3 up and Vulkan pushes the same five words; WebGPU has no push
// constants at all, so every one of them is a FIELD of the one `@group(1)
// @binding(0)` uniform block declared below. `src/lib.rs` states the rule:
// the ROUTINE's signature picks the storage and uniform runs -- buffers in the
// order the body asks for them, scalars as the fields of the `@group(1)` block
// `driver-wgpu::lowering::routine::bind` packs. The strided pair's buffers
// follow the sibling kernels' order.
//
// **Every bf16 index.** WGSL has no 16-bit storage type, so `x`, `w`, `out`
// and the residual cross as `array<u32>` with TWO values per word
// (`common/bf16.inc.wgsl`). A read is a word load and a half select, which the
// `*_at` helpers below do so the `>> 1u` appears once per buffer instead of
// once per use; a write goes a WHOLE WORD at a time, because a half-word store
// is a read-modify-write and two invocations can share a word. The one edge
// where they provably do -- an odd row width or pitch -- goes through
// `store_half`, and that is why `out` is an `array<atomic<u32>>`.
//
// **The reduction's signature.** WGSL has no `gl_SubgroupID` and no
// `gl_NumSubgroups`, so `pie_inv_rms` is told the lane and the workgroup width
// rather than reading them. They must be `local_invocation_id.x` and the
// `@workgroup_size` below, which is why both come from `PIE_LANES`.
//
// ## One block, where there were two
//
// `eps`, `axis_size`, `w_stride`, `plus_one` and `gain` were a `RmsParams`
// STRUCT on `@group(0) @binding(3)` -- MLX's Metal layout, ported through
// `norm/rms_params.h`, staged from the statement's params run whole and read
// by FIELD -- while the strided pair's `row_pitch` was a number the BODY
// passed and therefore a `@group(1)` uniform of its own. The two roads carried
// scalars to one shader at once, and `driver-wgpu::lowering::routine::bind`
// has a branch for exactly that pair: the body's scalars to the uniform, the
// statement's run to the storage pointer, because appending them to the run
// would leave the uniform empty and `Device::check_bindable` refuses that by
// name.
//
// All five are marks now -- `norm::rms_single_row` and its four siblings state
// every one of them, where they used to state two and read three out of a
// struct nothing in Rust could name -- and a mark is the SAME word of the SAME
// run, reached by index instead of by field. Nothing about which numbers
// arrive has changed. What changes is that they arrive the way `row_pitch`
// already did, so this module declares no storage block at all and takes that
// branch no longer.
//
// ## The one kernel in the sweep whose bindings had to be renumbered
//
// `params` was not the LAST binding here. It sat at 3 with the residual arms'
// `r` and `s` declared after it, so deleting it left a hole at 3 -- and a hole
// is not free on this backend: `driver-wgpu` builds a bind group from the
// module's DECLARATIONS, so every declared slot is one the dispatch has to
// fill and a gap is a layout the dispatch cannot satisfy. `r` is binding 3 now
// and `s` is 4, and `norm::rms_residual`/`rms_residual_scaled` bind them
// there.
//
// ## The field order is the body's argument order
//
// `driver-wgpu::lowering::routine::bind` packs the block by walking the fire's
// argument list and appending each scalar's own width, so the struct below is
// what the five bodies pass, in the order they pass it. That is the STRUCT's
// old order, field for field, because that order was the statement's: word 0
// is the epsilon, word 4 is the gain, and a body that reordered them would
// read a gain as an epsilon with no error anywhere.
//
// `row_pitch` keeps the END of the strided arms' block for the same reason a
// stride keeps the end of every list in this family: folded in ahead of the
// five, it would renumber the fields all five entrypoints share.

//#include "common/bf16.inc.wgsl"
//#include "common/reduce.inc.wgsl"

// The workgroup width, named once so the attribute and the reduction cannot
// drift apart: `pie_inv_rms` folds `lanes` partials, and a body that declared
// 256 and reduced over 128 would silently norm by half a row.
const PIE_LANES = 256u;

@group(0) @binding(0) var<storage, read_write> x: array<u32>;
@group(0) @binding(1) var<storage, read_write> w: array<u32>;
// Atomic ONLY because of the odd-width edge `store_half` handles; see there.
// The element type does not change what the host binds -- it is still a
// read_write storage buffer of 4-byte words -- so the row's ABI is untouched.
@group(0) @binding(2) var<storage, read_write> out_: array<atomic<u32>>;

// BINDING 3, WHICH WAS 4, AND 4, WHICH WAS 5. The `RmsParams` storage block
// held 3 and these two were declared after it; with the block gone they close
// up rather than leave a hole, because a declared-and-unfilled slot is a bind
// group `driver-wgpu` cannot build. The conditional bindings still come after
// the unconditional ones, so folding the residual in does not renumber the
// three every form shares.
//#if defined(PIE_RESIDUAL)
@group(0) @binding(3) var<storage, read_write> r: array<u32>;
//#if defined(PIE_SCALED)
@group(0) @binding(4) var<storage, read_write> s: array<u32>;
//#endif
//#endif

// The shared prefix is written out once per arm rather than shared between
// them, because a WGSL struct cannot be extended: the strided arms' pitch is a
// SIXTH FIELD and not a second block, for the reason the header gives at
// length. `PIE_HEAD_ROWS` is only ever built with `PIE_STRIDED` beside it --
// see the instantiation lines at the foot of this file -- so the two-level
// form takes the six-field arm as well.
//#if defined(PIE_STRIDED)
struct Params {
    eps: f32,
    axis_size: u32,
    w_stride: u32,
    plus_one: u32,
    gain: f32,
    row_pitch: i32,
}
//#else
struct Params {
    eps: f32,
    axis_size: u32,
    w_stride: u32,
    plus_one: u32,
    gain: f32,
}
//#endif
@group(1) @binding(0) var<uniform> params: Params;

// The half-index split, one helper per buffer.
//
// `pie_bf16_at` takes a WORD and not a pointer: core WGSL allows a pointer
// parameter only in the `function`, `private` and `workgroup` address spaces,
// so a shared `load(&buffer, i)` cannot exist -- `ptr<storage, ...>` needs the
// `unrestricted_pointer_parameters` extension, which naga rejects while the
// PARSE succeeds. Hence a helper per binding, and the `>> 1u` visible in each.
fn x_at(i: u32) -> f32 {
    return pie_bf16_at(x[i >> 1u], i);
}

fn w_at(i: u32) -> f32 {
    return pie_bf16_at(w[i >> 1u], i);
}

//#if defined(PIE_RESIDUAL)
fn r_at(i: u32) -> f32 {
    return pie_bf16_at(r[i >> 1u], i);
}
//#endif

// One bf16 of a word this invocation does NOT own outright.
//
// Reachable only when a row's base or its end is odd -- an odd `axis_size`, or
// an odd `row_pitch` -- which puts the tail of one row and the head of the next
// in ONE WORD. Those two halves are then written by two different WORKGROUPS,
// and a read-modify-write of a plain `u32` keeps whichever landed second and
// drops the other: one element per row boundary, lost silently.
//
// So the store goes through a device-scoped compare-exchange, which is the
// scope of the race -- the two writers are not in one workgroup, so no barrier
// reaches them. `...Weak` is allowed to fail spuriously; that is what the loop
// is for, and `old_value` is why the new word is recomputed inside it rather
// than once outside. `kernels/quant/qmm_t.wgsl` carries the same pattern for a
// race that is unconditional there rather than an odd-width edge.
//
// The whole-word path does NOT come here: a word whose halves are both inside
// one row has exactly one writer, and its `atomicStore` is a plain relaxed
// store on every backend this targets.
fn store_half(i: u32, value: f32) {
    let at = i >> 1u;
    var old = atomicLoad(&out_[at]);
    loop {
        let res = atomicCompareExchangeWeak(&out_[at], old, pie_bf16_into(old, i, value));
        if (res.exchanged) { break; }
        old = res.old_value;
    }
}

// The row this workgroup owns, as an ELEMENT offset.
//
// Three shapes, and the third is why the head form cannot be the strided one
// with a different pitch: a token holds `n_rows` per-head norms packed
// `axis_size` apart, and the next token is a uniform `row_pitch` away, so the
// base is two-level and one grid axis cannot carry both terms. The launch gives
// it two -- `workgroup_id.y` is the head, `.z` is the token -- exactly as
// Metal's does.
fn row_base(wg: vec3<u32>) -> u32 {
//#if defined(PIE_HEAD_ROWS)
    return wg.z * u32(params.row_pitch) + wg.y * params.axis_size;
//#elif defined(PIE_STRIDED)
    return wg.x * u32(params.row_pitch);
//#else
    return wg.x * params.axis_size;
//#endif
}

// The gain at element `i` of the row, in float.
//
// `plus_one` is the gemma convention -- every RMSNorm weight is stored as `w`
// and applied as `1 + w` -- and it is folded in FLOAT, before the bf16 round,
// because MLX materialises `add(weight, 1.0f)` in float and a parity walk
// against it has to make the same choice.
fn gain_at(i: u32) -> f32 {
    let wv = w_at(params.w_stride * i);
    return params.gain * select(wv, 1.0 + wv, params.plus_one != 0u);
}

// One output element: `at` is the row-relative index, `abs` the absolute one.
//
// Two indices and not one because they do not agree: `x` and the residual are
// addressed from the row's base, while the norm weight is addressed from the
// row's START -- it is one vector shared by every row, so a strided launch that
// fed it the absolute index would read the next row's gains.
fn normed(abs: u32, at: u32, inv: f32, post: f32) -> f32 {
    let value = gain_at(at) * (x_at(abs) * inv);
//#if defined(PIE_RESIDUAL)
    return (value + r_at(abs)) * post;
//#else
    // `post` is the scaled form's per-layer gain and this form has none. It
    // stays a parameter so the store loop below is one loop rather than two.
    return value;
//#endif
}

@compute @workgroup_size(PIE_LANES)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let lane = lid.x;
    let base = row_base(wg);
    let axis = params.axis_size;
    let span = PIE_LANES * u32(N_READS);

    var acc = 0.0;
    for (var start = lane * u32(N_READS); start < axis; start = start + span) {
        for (var i = 0u; i < u32(N_READS); i = i + 1u) {
            // The last chunk of a row whose width is not a multiple of `span`
            // is ragged -- 5120 is a multiple at 256 lanes and 4 reads, 3072 is
            // not -- so the tail is tested per element rather than assumed.
            if (start + i < axis) {
                let xi = x_at(base + start + i);
                acc = acc + xi * xi;
            }
        }
    }
    // Every lane arrives here, including one whose whole chunk was past the end
    // of the row. `pie_inv_rms` barriers inside, and a lane that had returned
    // early would hang the ones that had not -- a hang, not a wrong number.
    let inv = pie_inv_rms(lane, PIE_LANES, acc, axis, params.eps);

//#if defined(PIE_RESIDUAL) && defined(PIE_SCALED)
    // One number for the whole row, read by every lane: a broadcast load, which
    // every implementation coalesces, rather than a shared slot that would want
    // a barrier this loop does not otherwise need.
    let post = pie_bf16_at(s[0], 0u);
//#else
    let post = 1.0;
//#endif

    // The store walks WORDS and not `N_READS` chunks. A word is the smallest
    // thing WGSL can write, `N_READS` is a READ width (MLX's name for it), and
    // the two are different questions: one lane per word is what makes the
    // interior of a row a plain whole-word write, leaving only the two edges to
    // the compare-exchange.
    //
    // The bounds are absolute word indices, so a row that does not begin on a
    // word boundary is still addressed correctly.
    let first = base >> 1u;
    let end = (base + axis + 1u) >> 1u;
    for (var word = first + lane; word < end; word = word + PIE_LANES) {
        let lo = word * 2u;
        let hi = lo + 1u;
        let has_lo = lo >= base && lo < base + axis;
        let has_hi = hi < base + axis;
        if (has_lo && has_hi) {
            // Both halves are this row's, so this invocation owns the word and
            // the store races nobody: the whole word goes out in one write.
            atomicStore(&out_[word], pie_pack_bf16(
                normed(lo, lo - base, inv, post),
                normed(hi, hi - base, inv, post),
            ));
        } else if (has_hi) {
            // The row starts in this word's UPPER half, which only an odd
            // `axis_size` or `row_pitch` can produce. The lower half is then
            // the PREVIOUS row's last element, written by a different
            // workgroup at the same time, so this half goes through the
            // compare-exchange rather than a read-modify-write that would drop
            // one of the two.
            store_half(hi, normed(hi, hi - base, inv, post));
        } else if (has_lo) {
            // The mirror case: the row ends in this word's LOWER half and the
            // next row owns the upper one. Tested rather than assumed as the
            // remaining case, so that a word range computed wrong writes
            // nothing instead of writing a neighbour.
            store_half(lo, normed(lo, lo - base, inv, post));
        }
    }
}

// pie:instantiate rms_single_row_bfloat16 N_READS=4
// pie:instantiate rms_strided_row_bfloat16 N_READS=4 PIE_STRIDED=1
// pie:instantiate rms_strided_head_row_bfloat16 N_READS=4 PIE_STRIDED=1 PIE_HEAD_ROWS=1
// pie:instantiate rms_residual_bfloat16 N_READS=4 PIE_RESIDUAL=1
// pie:instantiate rms_residual_scaled_bfloat16 N_READS=4 PIE_RESIDUAL=1 PIE_SCALED=1
