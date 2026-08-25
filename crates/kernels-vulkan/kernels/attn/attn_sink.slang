// `attention.sink` -- the rescale, and the rebase inside it.
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
// # This file is not `sdpa_paged.slang`'s sink arm, and the difference is the point
//
// `attn/sdpa_paged.slang` has a `PIE_WITH_SINK` arm that folds the same scalar
// into the online softmax's denominator BEFORE the division, and its
// `sdpa_paged_{decode,tiled,decode_combine}_sink_bfloat16_d_64` entry points
// are that reading. Numerically the two agree -- `driver-vulkan`'s
// `device_sink` measures it, over identical bytes, to within a bf16 store.
// They are still not the same statement:
//
//   * a folded arm publishes no lse, and `attention.decode_lse` DECLARES one
//     (`lse: Out<Self::Tensor<f32>>`, shaped `[q.rows, q.width / head_dim]`).
//     A point is a contract about what is written, so an arm that writes only
//     `o` does not answer it however right `o` is;
//   * a folded arm answers `attention.decode` for a family that happens to
//     have sinks. Nothing in this tree states that -- gpt-oss's text says
//     `decode_lse` then `sink`, two statements -- so on this plane the folded
//     arms stay dark, and `device_sink` has to hand-state their operand run to
//     fire one at all, which is the finding rather than a shortcut.
//
// # THE ONE PLACE TWO BASES MEET
//
// The `kLn2` below is not a conversion somebody remembered to schedule in
// front of this kernel; it is the point itself. `attention.decode_lse` states
// BASE TWO -- flashinfer's, the base every attention kernel on the cuda plane
// has for free because its host folds `log2(e)` into `sm_scale` -- and this
// plane's own softmax accumulates on Slang's `exp`, which compiles to
// GLSL.std.450's `Exp` and is therefore natural log, and rebases on the way out
// (`sdpa_lse_base2`, in `attn/sdpa_online.slang`, where the measurement is
// recorded). The sink beside it is a CHECKPOINT WEIGHT: gpt-oss's
// `self_attn.sinks`, BF16 [64], values like 2.515625 and 0.55859375, in the
// natural-log formulation HF wrote them in. So the sigmoid argument has one
// operand in each base and this multiply is what makes them comparable.
//
// Without it the argument is off by a factor of 0.693. That matched HF's top-1
// on most prompts by accident and then drifted -- greedy decoding degenerated
// after a few steps on some inputs -- which is the history recorded in
// `kernels-cuda/kernels/attn/attn_sink.cuh`, whose `pie::attn::attn_sink_rescale`
// this kernel is the twin of and is checked against on the same card. The
// constant is spelled to full fp32 precision in both, because a rebased LSE and
// a rebasing rescale must agree on the same last bit or the two paths disagree
// on which token wins.
//
// # Why the sink is bf16 and the lse is f32
//
// A sink is a learned weight and rides the checkpoint's element. An lse is
// accumulator state, produced and consumed inside one fire, and stays f32 --
// which is also what `Attention::sink` states: `sink: Const<Tensor<T>>` beside
// `lse: In<Tensor<f32>>`.

#include "common/bf16.slang"

// Two buffers for one plane. `Attention::sink` states `o` as `InOut`, and this
// plane cuts an in-place mark into a read half and a write half carrying the
// same handle -- the shape `attn/logit_softcap.slang` already has, where the
// claim binds `x.ptr.arg(), x.arg()`. So this IS in place, and the kernel does
// not have to know it.
//
// The read half is a `StructuredBuffer` and the write half is not, which on
// this plane is a real distinction rather than a naming one: Slang decorates
// the read half `NonWritable`. Nothing here relies on seeing another
// invocation's write -- one lane owns one element and reads exactly the element
// it writes -- so the alias needs no shadow copy.
PIE_BUFFER_RO(0, o_in);
PIE_BUFFER_RW(1, out_);
// Base two, and one f32 per query head per row.
[[vk::binding(2)]] StructuredBuffer<float> lse;
// The learned per-head logit, in the checkpoint's own natural log.
PIE_BUFFER_RO(3, sinks);

// BOTH EXTENTS ARE PUSHED, where `attn_sink.metal` reads them off
// `threads_per_grid`. Slang's `gl_NumWorkGroups` counts GROUPS, and this
// kernel's x extent is rounded up to one at gpt-oss's width -- so a shader that
// tried to recover the head width from the launch would recover 256. They are
// the statement's own numbers either way: `Attention::sink` states `head_dim`
// and the head count is `o.width / head_dim`, which the claim body divides once
// rather than every lane re-deriving. `attn/kv_write.slang` pushes its head
// geometry for the same reason and takes the same guard.
struct Push { int head_dim; int heads; };
[[vk::push_constant]] ConstantBuffer<Push> pc;

// `isfinite` by the definition IEEE gives it: an exponent field of all ones is
// an infinity or a NaN and nothing else is.
//
// Slang HAS `isfinite`, and this is here anyway, because what the point states
// is a BIT PATTERN. `sdpa_lse_base2` publishes `asfloat(0xff800000u)` on a row
// that kept no key, and testing the exponent field is that same statement read
// back. It also answers correctly for the NaN a zero-length row can produce,
// where a comparison against a large literal would answer the WRONG thing --
// every comparison with a NaN is false.
bool pie_finite(float v) {
    return (asuint(v) & 0x7f800000u) != 0x7f800000u;
}

// `x` is the channel within the head, `y` the query head, `z` the token row,
// and the launch is `head_grid(head_dim, heads, rows)` threads -- so every lane
// of a group shares one `(row, head)` and therefore one factor. Nothing here
// barriers and nothing is shared, so the redundant `sigmoid` per lane is
// cheaper than any way of avoiding it.
//
// ONE LANE OWNS ONE CHANNEL, where `attn/attn_sink.wgsl` next door has to own a
// PAIR. That is the whole of what bf16-as-`uint16_t` buys this plane: WGSL has
// no 16-bit storage type, so bf16 crosses as `array<u32>` there and a lane
// owning one channel would read-modify-write a word its neighbour writes at the
// same instant. `PIE_ACT` is `uint16_t` here and the hazard does not exist.
//
// Only the x axis overshoots -- `head_grid` states `heads` and `rows` exactly
// -- so only x is guarded, which is `attn/kv_write.slang`'s guard exactly.
[numthreads(256, 1, 1)]
void main(uint3 gid : SV_DispatchThreadID) {
    static const float kLn2 = 0.69314718055994530942;

    uint d = gid.x;
    uint h = gid.y;
    uint t = gid.z;
    if (d >= uint(pc.head_dim)) return;

    float r = 1.0;
    float lse_val = lse[t * uint(pc.heads) + h];
    if (pie_finite(lse_val)) {
        // THE REBASE, and it is the operand and not the kernel that needed it:
        // `lse_val` is base two because `sdpa_lse_base2` published it that way,
        // and `sinks[h]` is natural log because a checkpoint wrote it.
        float diff = lse_val * kLn2 - PIE_LOAD(sinks[h]);
        r = 1.0 / (1.0 + exp(-diff));
    }
    // `lse = -inf` on a row that kept no key: causally masked out, or a window
    // with nothing in it. `o` is already zero there, so the factor is
    // don't-care and 1 is the cheapest don't-care. It also covers the NaN a
    // zero-length row can produce.

    uint i = (t * uint(pc.heads) + h) * uint(pc.head_dim) + d;
    out_[i] = PIE_STORE(PIE_LOAD(o_in[i]) * r);
}

// pie:instantiate attn_sink_rescale_bfloat16
