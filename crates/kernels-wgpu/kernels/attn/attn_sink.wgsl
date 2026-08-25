// `attention.sink` — the rescale, and the rebase inside it.
//
// gpt-oss learns a per-head sink scalar and extends the softmax denominator
// with `exp(sink)`. This is the POST-PASS reading of that, and it is the one
// the floor declares: `attention.decode_lse` writes an output and the
// log-sum-exp of the denominator that made it, and this statement multiplies
// the output by `sigmoid(lse - sink)`, which is exactly the factor the extra
// denominator term would have applied.
//
//   o[t, h, :] *= sigmoid(lse[t, h] * ln(2) - sink[h])
//
// Equivalent to a virtual KV slot with logit `sink[h]` and value zero: the
// denominator grows by `exp(sink)` and the numerator does not, so every
// component of the head shrinks by one factor. `sigmoid(l - s) = e^l/(e^l +
// e^s)`, and `e^l` IS the denominator, so this is `D/(D + e^s)`.
//
// # This file is not `sdpa_paged.wgsl`'s sink arm, and the difference is the point
//
// `attn/sdpa_paged.wgsl` has a `PIE_WITH_SINK` arm that folds the same scalar
// into the online softmax's denominator BEFORE the division, and its
// `sdpa_paged_{decode,tiled}_sink_bfloat16_d_64` entry points are that reading.
// Numerically the two agree — `driver-wgpu`'s `device_sink` measures it, over
// identical bytes, to within a bf16 store. They are still not the same
// statement:
//
//   * a folded arm publishes no lse, and `attention.decode_lse` DECLARES one
//     (`lse: Out<Self::Tensor<f32>>`, shaped `[q.rows, q.width / head_dim]`).
//     A point is a contract about what is written, so an arm that writes only
//     `o` does not answer it however right `o` is;
//   * a folded arm answers `attention.decode` for a family that happens to
//     have sinks. Nothing in this tree states that — gpt-oss's text says
//     `decode_lse` then `sink`, two statements — so on this plane the folded
//     arms stay dark and the sdpa suites are what keep them honest.
//
// # THE ONE PLACE TWO BASES MEET
//
// The `kLn2` below is not a conversion somebody remembered to schedule in
// front of this kernel; it is the point itself. `attention.decode_lse` states
// BASE TWO — flashinfer's, the base every attention kernel on the cuda plane
// has for free because its host folds `log2(e)` into `sm_scale` — and this
// plane's own softmax accumulates on WGSL's `exp`, which is natural log, and
// rebases on the way out (`pie_sdpa_lse_base2`, in `attn/sdpa_online.inc.wgsl`).
// The sink beside it is a CHECKPOINT WEIGHT: gpt-oss's `self_attn.sinks`,
// BF16 [64], values like 2.515625 and 0.55859375, in the natural-log
// formulation HF wrote them in. So the sigmoid argument has one operand in each
// base and this multiply is what makes them comparable.
//
// Without it the argument is off by a factor of 0.693. That matched HF's top-1
// on most prompts by accident and then drifted — greedy decoding degenerated
// after a few steps on some inputs — which is the history recorded in
// `kernels-cuda/kernels/attn/attn_sink.cuh`, whose `pie::attn::attn_sink_rescale`
// this kernel is the twin of and is checked against on the same card. The
// constant is spelled to full fp32 precision in both, because a rebased LSE and
// a rebasing rescale must agree on the same last bit or the two paths disagree
// on which token wins.
//
// # Why the sink is bf16 and the lse is f32
//
// A sink is a learned weight and rides the checkpoint's element. An lse is
// accumulator state, produced and consumed inside one fire, and stays f32 —
// which is also what `Attention::sink` states: `sink: Const<Tensor<T>>` beside
// `lse: In<Tensor<f32>>`.

//#include "common/bf16.inc.wgsl"

// Two buffers for one plane. `Attention::sink` states `o` as `InOut`, and this
// plane cuts an in-place mark into a read half and a write half carrying the
// same handle — the shape `attn/logit_softcap.wgsl` already has. So this IS in
// place, and the kernel does not have to know it.
//
// BOTH ARE `read_write` AND THAT IS THE TREE'S DECISION, not an oversight:
// `driver-wgpu`'s `no_shader_declares_a_read_only_storage_binding` holds every
// shader here to it. A `var<storage, read>` half would be honest about this
// body and would cost a real copy — `Device::run_all` shadows a read-only range
// that shares a buffer with a writable one, which on a decode measured 451
// copies for 452 launches — while two writable bindings into one buffer are
// legal WebGPU, which `two_read_write_bindings_into_one_buffer_are_legal`
// asserts on a device rather than assuming. Nothing here relies on seeing
// another invocation's write: one lane owns one word and reads exactly the word
// it writes.
@group(0) @binding(0) var<storage, read_write> o_in: array<u32>;
@group(0) @binding(1) var<storage, read_write> o_out: array<u32>;
// Base two, and one f32 per query head per row.
@group(0) @binding(2) var<storage, read_write> lse: array<f32>;
// The learned per-head logit, in the checkpoint's own natural log.
@group(0) @binding(3) var<storage, read_write> sinks: array<u32>;

// BOTH EXTENTS ARE STATED, where `attn_sink.metal` reads them off
// `threads_per_grid`. WGSL has no such builtin — `num_workgroups` counts
// GROUPS, and this kernel's x extent is rounded up to one — so a shader that
// tried to recover the head width from the launch would recover 256. They are
// the statement's own numbers either way: `Attention::sink` states `head_dim`
// and the head count is `o.width / head_dim`, which the claim body divides once
// rather than every lane re-deriving.
struct Params { head_dim: i32, heads: i32 }
@group(1) @binding(0) var<uniform> params: Params;

// `isfinite`, which WGSL does not have, by the definition IEEE gives it: an
// exponent field of all ones is an infinity or a NaN and nothing else is. A
// comparison against a large literal would answer the same for infinities and
// the WRONG thing for NaN, since every comparison with a NaN is false — and NaN
// is one of the two values an empty row can carry here.
fn pie_finite(v: f32) -> bool {
    return (bitcast<u32>(v) & 0x7f800000u) != 0x7f800000u;
}

fn sink_at(i: u32) -> f32 {
    let word = sinks[i >> 1u];
    return pie_bf16_to_f32(select(word & 0xffffu, word >> 16u, (i & 1u) == 1u));
}

// One lane owns a channel PAIR and writes a whole word, for the reason
// `attn/kv_write.wgsl` gives at length: bf16 crosses as `array<u32>`, WGSL has
// no sub-word atomic, and a lane owning one channel would read-modify-write a
// word its neighbour writes at the same instant. The host launches a full head
// on the x axis (`head_grid`), so the upper half exits at the guard — the
// harmless direction.
//
// `y` is the query head and `z` the token row, so every lane of a group shares
// one `(row, head)` and therefore one factor. Nothing here barriers and nothing
// is shared, so the redundant `sigmoid` per lane is cheaper than any way of
// avoiding it.
@compute @workgroup_size(256)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    const kLn2 = 0.69314718055994530942;

    let hd = u32(params.head_dim);
    let heads = u32(params.heads);
    let d = gid.x * 2u;
    if (d >= hd) { return; }
    let h = gid.y;
    let t = gid.z;
    if (h >= heads) { return; }

    let lse_at = t * heads + h;
    if (lse_at >= arrayLength(&lse)) { return; }

    var r = 1.0;
    let lse_val = lse[lse_at];
    if (pie_finite(lse_val)) {
        // THE REBASE, and it is the operand and not the kernel that needed it:
        // `lse_val` is base two because `pie_sdpa_lse_base2` published it that
        // way, and `sink_at(h)` is natural log because a checkpoint wrote it.
        let diff = lse_val * kLn2 - sink_at(h);
        r = 1.0 / (1.0 + exp(-diff));
    }
    // `lse = -inf` on a row that kept no key: causally masked out, or a window
    // with nothing in it. `o` is already zero there, so the factor is
    // don't-care and 1 is the cheapest don't-care. It also covers the NaN a
    // zero-length row can produce.

    // `heads * hd` and `hd` are both even, so this index is even and the pair
    // is exactly one word.
    let at = ((t * heads + h) * hd + d) >> 1u;
    if (at < arrayLength(&o_out)) {
        let word = o_in[at];
        o_out[at] = pie_pack_bf16(
            pie_bf16_to_f32(word & 0xffffu) * r,
            pie_bf16_to_f32(word >> 16u) * r,
        );
    }
}

// pie:instantiate attn_sink_rescale_bfloat16
