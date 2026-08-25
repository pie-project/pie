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
// # This file is not `sdpa_merge_sink`, and the difference is the point
//
// `attn/sdpa_paged.metal` has a `WITH_SINK` template parameter that folds the
// same scalar into the online softmax's denominator BEFORE the division, and
// its `_sink` entry points are that reading. Numerically the two agree. They
// are not the same statement:
//
//   * a folded arm publishes no lse, and `attention.decode_lse` DECLARES one
//     (`lse: Out<Self::Tensor<f32>>`, shaped `[q.rows, q.width / head_dim]`).
//     A point is a contract about what is written, so an arm that writes only
//     `o` does not answer it however right `o` is;
//   * a folded arm answers `attention.decode` for a family that happens to
//     have sinks. Nothing in this tree states that -- gpt-oss's text says
//     `decode_lse` then `sink`, two statements -- so on this plane the folded
//     arms are dark, and `device_attention` is what keeps them honest.
//
// # THE ONE PLACE TWO BASES MEET
//
// The `kLn2` below is not a conversion somebody remembered to schedule in
// front of this kernel; it is the point itself. `attention.decode_lse` states
// BASE TWO -- flashinfer's, the base every attention kernel on the cuda plane
// has for free because its host folds `log2(e)` into `sm_scale` -- and this
// plane's own softmax accumulates in natural log and rebases on the way out
// (`sdpa_lse_base2`, in `sdpa_online.h`). The sink beside it is a CHECKPOINT
// WEIGHT: `gpt-oss`'s `self_attn.sinks`, BF16 [64], values like 2.515625 and
// 0.55859375, in the natural-log formulation HF wrote them in. So the sigmoid
// argument has one operand in each base and the multiply is what makes them
// comparable.
//
// Without it the argument is off by a factor of 0.693. That matched HF's top-1
// on most prompts by accident and then drifted -- greedy decoding degenerated
// after a few steps on some inputs -- which is the history recorded in
// `kernels-cuda/kernels/attn/attn_sink.cuh`, the kernel this one is the twin
// of. The constant is spelled to full fp32 precision in both, because a
// rebased LSE and a rebasing rescale must agree on the same last bit or the
// two paths disagree on which token wins.
//
// # Why the sink is `T` and the lse is `float`
//
// A sink is a learned weight and rides the checkpoint's element. An lse is
// accumulator state, produced and consumed inside one fire, and stays fp32 --
// which is also what `Attention::sink` states: `sink: Const<Tensor<T>>`
// beside `lse: In<Tensor<f32>>`.

#include <metal_stdlib>
using namespace metal;

// THE GRID IS THE EXTENT, as it is in `logit_softcap.metal` and for the same
// reason: an operand that restates a dispatch dimension is a second place for
// it to be wrong. `x` is the channel within the head, `y` the query head, `z`
// the token row, and the launch is `[head_dim, heads, rows]` threads with a
// threadgroup of one head's channels -- so every lane of a group shares one
// `(row, head)` and therefore one factor.
//
// Two buffers for one plane. `Attention::sink` states `o` as `InOut`, and this
// plane cuts an in-place mark into a read half and a write half that carry the
// same address (`kernels_metal::points::{read_half, write_half}`) -- so this
// IS in place, and the kernel does not have to know that.
template <typename T>
[[kernel]] void attn_sink_rescale(
    const device T* o_in     [[buffer(0)]],   // [rows, heads, head_dim]
    device T* o_out          [[buffer(1)]],
    const device float* lse  [[buffer(2)]],   // [rows, heads], base two
    const device T* sinks    [[buffer(3)]],   // [heads]
    uint3 tid  [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  constexpr float kLn2 = 0.69314718055994530942f;

  const uint head_dim = grid.x;
  const uint heads = grid.y;
  const uint d = tid.x;
  const uint h = tid.y;
  const uint t = tid.z;

  const float lse_val = lse[size_t(t) * size_t(heads) + size_t(h)];
  float r;
  if (!isfinite(lse_val)) {
    // `lse = -inf` on a row that kept no key: causally masked out, or a window
    // with nothing in it. `o` is already zero there, so the factor is
    // don't-care and 1 is the cheapest don't-care. It also covers the NaN a
    // zero-length row can produce.
    r = 1.0f;
  } else {
    const float diff = lse_val * kLn2 - static_cast<float>(sinks[h]);
    r = 1.0f / (1.0f + precise::exp(-diff));
  }

  const size_t i =
      (size_t(t) * size_t(heads) + size_t(h)) * size_t(head_dim) + size_t(d);
  o_out[i] = static_cast<T>(static_cast<float>(o_in[i]) * r);
}

#define instantiate_attn_sink_rescale(name, itype)                      \
  template [[host_name("attn_sink_rescale_" #name)]]                    \
  [[kernel]] void attn_sink_rescale<itype>(                             \
      const device itype*, device itype*, const device float*,          \
      const device itype*, uint3, uint3);

instantiate_attn_sink_rescale(bfloat16, bfloat)
