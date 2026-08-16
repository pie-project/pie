// Device argmax plus EOS compare.
//
//   next_token[r] = argmax_i logits[r, i]      (lowest index on a tie)
//   eos_flag[r]   = (next_token[r] in eos_ids) ? 1 : 0
//
// Bit-exact to the host scan it replaces: bf16 widens to f32 exactly, the scan
// is ascending with a strict `>`, and every reduction step keeps the LOWER
// index on a value tie. That is what makes the GPU-resident loop possible --
// `next_token` is bound directly as the next step's token id, with no readback,
// and a host argmax is what forces the per-token drain.
//
// The row is UNSTATED in the table, so the bindings are the sibling backends'
// and not a derivation: logits 0, next_token 1, params 2, eos_flag 3. Params
// and eos_flag are append-only extensions to the original two-buffer bind, and
// the two scalars ride INSIDE the params buffer, so there is no `@group(1)`.
//
// ## Why this reduces its own pair rather than calling `common/reduce.inc.wgsl`
//
// That fragment reduces a VALUE -- `pie_workgroup_max` answers what the largest
// logit is, and an argmax that has lost the index has answered the wrong
// question. Recovering it needs a second reduction over the indices that attain
// the maximum, which is two trees and a float round-trip of an integer where
// one tree over the PAIR is exact and obvious. The recurrence below is the same
// one that fragment states -- one slot per lane, a power-of-two tree, a barrier
// per level -- with `max` replaced by the lexicographic compare that keeps the
// lower index.
//
// ## 256 lanes, not Metal's or Vulkan's 1024
//
// WebGPU's guaranteed `maxComputeInvocationsPerWorkgroup` is **256**. A 1024-
// wide workgroup is a pipeline that fails to create on a conformant device, and
// the tree is the same recurrence at either width -- it just strides the vocab
// four times as often. The one lane per slot sizing is `reduce.inc.wgsl`'s, for
// the reason it gives: a `var<workgroup>` array is sized by a const-expression
// and a subgroup width is a runtime value.
//
// One workgroup owns one row (`workgroup_id.y`). A shell that over-dispatches
// on x is harmless without a guard: every workgroup strides the WHOLE vocabulary
// from its own lane offset, so each computes the same argmax and writes the
// same two words.

//#include "common/bf16.inc.wgsl"

@group(0) @binding(0) var<storage, read_write> logits: array<u32>;
@group(0) @binding(1) var<storage, read_write> next_token: array<u32>;
// `vocab` and `n_eos` are fields, not uniform-block scalars: the row states no
// operands, and both siblings pack them here with the stop-token list.
struct ArgmaxParams { vocab: u32, n_eos: u32, eos_ids: array<u32, 8> }
@group(0) @binding(2) var<storage, read_write> params: ArgmaxParams;
@group(0) @binding(3) var<storage, read_write> eos_flag: array<u32>;

const LANES = 256u;

var<workgroup> sh_v: array<f32, 256>;
var<workgroup> sh_i: array<u32, 256>;

@compute @workgroup_size(256)
fn main(
    @builtin(workgroup_id) wg: vec3<u32>,
    @builtin(local_invocation_id) lid3: vec3<u32>,
) {
    let row = wg.y;
    let lid = lid3.x;
    let vocab = params.vocab;
    let base = row * vocab;

    // The identity for a max is not zero and not the lowest finite float: a
    // masked logit really is -inf, and a lane whose stripe is empty must lose
    // to every lane that has one. WGSL has no infinity literal and a const
    // division by zero is a compile error, so the bit pattern is the way to
    // name it.
    let neg_inf = bitcast<f32>(0xff800000u);
    var best_v = neg_inf;
    var best_i = 0u;
    // Ascending within a lane, so a strict `>` already keeps this lane's lowest
    // index among its own ties.
    for (var i = lid; i < vocab; i = i + LANES) {
        let at = base + i;
        // `pie_load_bf16` says this in one call and cannot be used: it takes a
        // `ptr<storage, ...>`, and naga's validator allows a pointer argument
        // only in the `private` and `function` address spaces
        // (`unrestricted_pointer_parameters` is unimplemented, gfx-rs/wgpu#5158).
        // The row may start on either half, so the parity is `at`'s and not
        // `i`'s.
        let word = logits[at >> 1u];
        let v = pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (at & 1u) == 1u));
        if (v > best_v) {
            best_v = v;
            best_i = i;
        }
    }

    sh_v[lid] = best_v;
    sh_i[lid] = best_i;
    workgroupBarrier();

    // Every invocation reaches every barrier: the trip count is a constant and
    // the guard is on the STORE, not on entry. An early return in front of a
    // barrier is a hang rather than a wrong number.
    var stride = LANES >> 1u;
    loop {
        if (stride == 0u) { break; }
        if (lid < stride) {
            let ov = sh_v[lid + stride];
            let oi = sh_i[lid + stride];
            // Higher value wins; on a value tie the LOWER index wins, which is
            // what makes this an argmax the host scan agrees with rather than
            // whichever index the schedule happened to reduce last.
            if (ov > sh_v[lid] || (ov == sh_v[lid] && oi < sh_i[lid])) {
                sh_v[lid] = ov;
                sh_i[lid] = oi;
            }
        }
        workgroupBarrier();
        stride = stride >> 1u;
    }

    if (lid == 0u) {
        let tok = sh_i[0];
        next_token[row] = tok;
        var flag = 0u;
        // `min` because the list is eight slots and `n_eos` arrives from a
        // buffer: a WGSL index past a fixed-size array is clamped, so a wrong
        // count would silently compare against `eos_ids[7]` over and over
        // rather than fail.
        let n = min(params.n_eos, 8u);
        for (var e = 0u; e < n; e = e + 1u) {
            if (tok == params.eos_ids[e]) {
                flag = 1u;
                break;
            }
        }
        eos_flag[row] = flag;
    }
}

// pie:instantiate argmax_logits_bfloat16
