// The per-head RMS norm and the NEOX rotation, in one dispatch.
//
// `kernels-vulkan/kernels/norm/rms_rope.slang`, in WGSL, with one structural
// change this backend forces and one numerical choice it forces with it. Read
// the sibling first: it states why the merge is free by construction (the two
// stages are the same shape and the rotation reads what the norm already holds)
// and what it removes (the rope's own read and write of the head).
//
// # What it is worth HERE, which is a different number
//
// Vulkan fuses to remove a BARRIER: its decode is 73% ordering and the memo
// there prices the merge at 0.099 ms. This backend has no barrier to remove --
// `driver-wgpu/src/device.rs` records that wgpu-hal's Metal `transition_buffers`
// is an empty function -- so what a fusion removes here is a DISPATCH, and a
// dispatch has a measured price:
//
//   * `serving.rs`'s marginal census fires the per-head norm twice and reads
//     +0.334 ms for 56 extra launches: **6.0 us each**.
//   * The decode's 480 launches include 56 per-head qk-norms, so removing them
//     is 0.34 ms of a 9.53 ms token, **3.5%**.
//
// That is the whole of it, and it is worth stating plainly because this file's
// sibling reads as though the fusion were free money. It is not; it is one of
// four rows on `driver-wgpu/src/turns.rs`'s price list, the list totals 1.18 ms,
// and this is the best-plumbed row rather than a large one.
//
// # The structural change: an invocation owns WORDS, not elements
//
// The sibling writes `x[i1]` and `x[i2]` as single bf16 elements. WGSL's
// smallest storage element is four bytes and two bf16 share one, so an
// invocation that wrote one half would be a read-modify-write racing the
// neighbour that owns the other half. `rope/neox.wgsl` already solved this and
// this file copies the solution exactly: invocation `t` owns the WORD holding
// channels `2t, 2t+1` and the WORD holding their partners `2t+dist, 2t+dist+1`,
// rotates both pairs, and writes both words whole.
//
// The consequence is that the rotation's active lane count is `pairs/2` rather
// than `pairs` -- 32 of 256 at a 128-wide head. That is not a regression: the
// same 32 lanes are the only ones the REDUCTION has work for either, because
// `rms.wgsl` reads `N_READS = 4` elements a lane and a 128-element axis is
// 32 lanes' worth. The width is 256 to match `rms.wgsl` exactly, for the
// reason the next section gives.
//
// # The numerical choice: this kernel reproduces the two dispatches BIT FOR BIT
//
// `kernels-metal` declares `rms_rope` in `ELSEWHERE` with no `.metal` text, so
// there is no in-tree numeric reference for this family to be walked against.
// What this kernel must therefore agree with is what THIS BACKEND already
// produces from `rms_single_row` followed by `neox_mb`. Two things make that
// exact rather than approximate:
//
//   * **The reduction is `rms.wgsl`'s, unchanged.** 256 lanes, `N_READS = 4`,
//     `pie_inv_rms(lane, 256, acc, axis, eps)`. A 32-lane tree over the same
//     partials would give the same f32 -- the 224 zero slots fold exactly, and
//     `x + 0.0` is exact -- but writing it that way would be a claim that has to
//     be re-argued every time `N_READS` moves. Matching the shape costs 224 idle
//     lanes and settles it.
//
//   * **The normed value is ROUNDED TO BF16 before it is rotated.** In the
//     unfused pair, `rms_single_row` stores bf16 and `neox_mb` loads it back, so
//     the rotation's input has been through a bf16 round. Keeping f32 all the
//     way -- which is what the sibling does, and what "fused" usually means --
//     would be strictly MORE accurate and would change the answers. This
//     backend has no reference that could say the new answers are the better
//     ones, and a serving suite that pins tokens would read the improvement as
//     a regression. So the round is reproduced, in `pie_round_bf16`, and it is
//     one pack and one unpack.
//
// Dropping that round is the obvious follow-up and it is deliberately NOT taken
// here: it would make this change a numerical one as well as a structural one,
// and the two want separate measurements.
//
// # What is NOT here
//
// The `_freqs`, `_prop` and `_decode` points the sibling instantiates. There is
// one entrypoint because `model-dsl`'s `rms_rope` emits one name --
// `rms_rope_bfloat16` -- and the Vulkan routine fires that name unconditionally
// too, so the sibling's other five points are as unreachable as this file's
// absent ones. The rescaled-ladder case is excluded upstream by
// `forward/mod.rs`'s `!metal.rope_freq_table`; the proportional case is
// arithmetically identical to this one whenever the rotary is the whole head,
// which `rope/neox.wgsl`'s own header states, and differs otherwise. See the
// host routine for the refusal that keeps the differing case away from here.

//#include "common/bf16.inc.wgsl"
//#include "common/reduce.inc.wgsl"

@group(0) @binding(0) var<storage, read_write> x: array<u32>;
@group(0) @binding(1) var<storage, read_write> w: array<u32>;
@group(0) @binding(2) var<storage, read_write> position: array<i32>;

// `RmsRopeParams`, field for field, in the order `model-dsl::metal::rms_rope`
// writes them. The host forwards them one at a time -- this backend derives its
// uniform block from the fire's argument list rather than binding a params
// buffer -- so the order here is the order there and nothing checks it but the
// answers.
//
// Every field is `i32`, including the four that are floats: `Ctx::param` reads
// a params slot as `i32` and the family that fills the slot wrote
// `f32::to_bits`, so the bits travel as an integer and are bitcast here. A
// field declared `f32` would receive the same four bytes and read them as a
// float twice over.
struct Params {
    eps: i32,
    axis_size: i32,
    w_stride: i32,
    plus_one: i32,
    gain: i32,
    row_pitch: i32,
    rotary: i32,
    scale: i32,
    base: i32,
}
@group(1) @binding(0) var<uniform> params: Params;

// Must match `kernels_wgpu::norm::RMS_ROPE_LANES`, and `rms.wgsl`'s own width,
// for the parity argument in the header.
const PIE_LANES = 256u;
const N_READS = 4u;

fn x_at(i: u32) -> f32 {
    return pie_bf16_at(x[i >> 1u], i & 1u);
}

fn w_at(i: u32) -> f32 {
    return pie_bf16_at(w[i >> 1u], i & 1u);
}

// `rms.wgsl`'s `gain_at`, unchanged: gemma stores every RMSNorm weight as `w`
// and applies it as `1 + w`, folded in FLOAT before the bf16 round because MLX
// materialises `add(weight, 1.0f)` in float.
fn gain_at(i: u32) -> f32 {
    let wv = w_at(u32(params.w_stride) * i);
    return bitcast<f32>(params.gain) * select(wv, 1.0 + wv, params.plus_one != 0);
}

// A value put through the bf16 round the unfused pair's store-then-load
// performs. See the header: this is a fidelity choice, not an accident.
fn pie_round_bf16(v: f32) -> f32 {
    return pie_bf16_at(pie_pack_bf16(v, 0.0), 0u);
}

// The normed channel `at` of the head based at `base`, rounded as the separate
// norm would have stored it.
fn normed(base: u32, at: u32, inv: f32) -> f32 {
    return pie_round_bf16(gain_at(at) * (x_at(base + at) * inv));
}

@compute @workgroup_size(PIE_LANES)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid: vec3<u32>,
) {
    let lane = lid.x;
    // y is the head and z is the row, which is the sibling's `per_head_row` and
    // Metal's launch. x would do for the head; it is not used for it because
    // the whole family already answers this the same way.
    let head = wg.y;
    let row = wg.z;
    // Two-level, exactly as `rms.wgsl`'s `PIE_HEAD_ROWS` base is: a token holds
    // `n_head` per-head rows packed `axis_size` apart, and the next token is a
    // uniform `row_pitch` away. One grid axis cannot carry both terms.
    let axis = u32(params.axis_size);
    let base = row * u32(params.row_pitch) + head * axis;

    // `rms.wgsl`'s reduction, element for element and in its order.
    let span = PIE_LANES * N_READS;
    var acc = 0.0;
    for (var start = lane * N_READS; start < axis; start = start + span) {
        for (var i = 0u; i < N_READS; i = i + 1u) {
            if (start + i < axis) {
                let xi = x_at(base + start + i);
                acc = acc + xi * xi;
            }
        }
    }
    // Every lane arrives here, including one whose chunk was past the end of the
    // head. `pie_inv_rms` barriers inside, and a lane that had returned early
    // would hang the ones that had not -- a hang, not a wrong number.
    let inv = pie_inv_rms(lane, PIE_LANES, acc, axis, bitcast<f32>(params.eps));

    let pairs = u32(params.rotary) >> 1u;
    let pos = f32(position[row]);
    let scale = bitcast<f32>(params.scale);
    let theta_base = bitcast<f32>(params.base);

    // The rotation, one invocation per WORD-PAIR. Channels `2t, 2t+1` and their
    // partners `2t+pairs, 2t+pairs+1`: four channels, two whole words, and no
    // two invocations touch the same word. The reduction above read the head
    // before any of this wrote it and `pie_inv_rms` ends in a barrier, so this
    // loop needs no ordering of its own.
    for (var t = lane; t * 2u < pairs; t = t + PIE_LANES) {
        let i0 = 2u * t;
        let lo_at = (base + i0) >> 1u;
        let hi_at = (base + i0 + pairs) >> 1u;

        let a0 = normed(base, i0, inv);
        let a1 = normed(base, i0 + 1u, inv);
        let b0 = normed(base, i0 + pairs, inv);
        let b1 = normed(base, i0 + pairs + 1u, inv);

        let th0 = scale * pos * exp2(-(f32(i0) / f32(pairs)) * theta_base);
        let c0 = cos(th0);
        let s0 = sin(th0);
        let r0 = vec2<f32>(a0 * c0 - b0 * s0, a0 * s0 + b0 * c0);

        // The odd tail of a partial rotary: channel `i0+1` is past the rotated
        // range, so it keeps its NORMED value. Reachable only when `pairs` is
        // odd, which no checkpoint in this tree produces -- every rotary and
        // head width is a multiple of four -- and kept because `neox.wgsl`
        // keeps it and the two bodies are read side by side.
        var r1 = vec2<f32>(a1, b1);
        if (i0 + 1u < pairs) {
            let th1 = scale * pos * exp2(-(f32(i0 + 1u) / f32(pairs)) * theta_base);
            let c1 = cos(th1);
            let s1 = sin(th1);
            r1 = vec2<f32>(a1 * c1 - b1 * s1, a1 * s1 + b1 * c1);
        }

        x[lo_at] = pie_pack_bf16(r0.x, r1.x);
        x[hi_at] = pie_pack_bf16(r0.y, r1.y);
    }

    // The tail the rotation does not reach, normed and stored where it lies.
    // Empty whenever `rotary == axis_size`, which is every model this backend
    // fuses. A partial rotation still norms the WHOLE head: dropping these
    // would be wrong and storing them unnormed would be worse.
    //
    // Walked in WORDS for the same reason the loop above is, and the host
    // refuses a launch whose `rotary`, `axis_size` or `row_pitch` would put a
    // head's edge inside a word.
    let rot = u32(params.rotary);
    // Both bounds hoisted: `word < (base + axis) >> 1u` parses as the start of a
    // TEMPLATE argument list in WGSL, not as a comparison against a shift.
    let tail_from = ((base + rot) >> 1u) + lane;
    let tail_to = (base + axis) >> 1u;
    for (var word = tail_from; word < tail_to; word = word + PIE_LANES) {
        let at = word * 2u - base;
        x[word] = pie_pack_bf16(normed(base, at, inv), normed(base, at + 1u, inv));
    }
}

// pie:instantiate rms_rope_bfloat16
