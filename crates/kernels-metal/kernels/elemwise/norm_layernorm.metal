#include <metal_stdlib>
using namespace metal;

/// **THE CENTRING, WHICH IS THE HALF OF A `LayerNorm` NO IMPORT CAN FOLD** —
/// the Metal mirror of `kernels-cuda`'s `elemwise/layernorm.cuh`
/// (`.wiki/alto/multimodal.md` §6.1, §9.1, next.md B5).
///
/// `y = (x - mean(x)) * rsqrt(var(x) + eps) * w + b`, whole rows, both planes
/// `[width]`. Every qwen vision block is an `nn.LayerNorm` — the checkpoints
/// publish `blocks.{l}.norm1.bias` beside `.weight`, and an RMSNorm has no
/// bias — and the tower says it twenty-five times per qwen35 fire
/// (`norm1`/`norm2` on twelve blocks, plus `merger.norm`).
///
/// **TWO REDUCTIONS AND NOT ONE.** `var = E[x^2] - E[x]^2` would halve the
/// barriers and is the reason this kernel would be subtly wrong: a tower row
/// whose mean is large against its spread cancels catastrophically in f32,
/// and the failure shows up as a slightly wrong norm rather than as a NaN.
/// The mean is reduced, then the centred squares are reduced against it, the
/// way `torch.nn.LayerNorm` computes it.
///
/// `eps` sits INSIDE the root beside the variance, which is where LayerNorm
/// puts it and where `norm_rms.metal` next door puts it too.
///
/// **AND IT IS THE IDEAL ARITHMETIC, NOT THE COMPOSITION'S.** The three-op
/// spelling this replaces — `add_bias(b, rmsnorm(layernorm_no_scale(x, eps),
/// w, eps))` — rounds the centred row to `T` before the `rmsnorm` reads it,
/// then reduces THOSE rounded values and multiplies by their reciprocal rms,
/// a uniform per-row factor of `1 +/- 1.4e-4` that is the composition's
/// artifact and no part of LayerNorm. This kernel has no such intermediate:
/// the centred row stays f32 to the single rounding at the store, which lands
/// strictly nearer the f32 reference than the form it replaces.
///
/// `fma` and not a multiply-then-add: the scale and the bias are one
/// operation, which is the one place the fusion buys accuracy as well as
/// launches.
///
/// **THE REDUCTION IS THIS PLANE'S OWN IDIOM** — `simd_sum` into a
/// `partials[32]` scratch and one more `simd_sum` over that — which is
/// `norm_rms.metal`'s `rms_inv_from_lane_sum` read twice. The twin's
/// `block_reduce_sum_exact` is a CUDA shuffle ladder and the same sum.
METAL_FUNC float layernorm_group_sum(
    float acc, threadgroup float* out, threadgroup float* partials,
    uint simd_lane, uint simd_group) {
  acc = simd_sum(acc);
  if (simd_group == 0) partials[simd_lane] = 0.0f;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_lane == 0) partials[simd_group] = acc;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (simd_group == 0) {
    acc = simd_sum(partials[simd_lane]);
    if (simd_lane == 0) out[0] = acc;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return out[0];
}

template <typename T>
[[kernel]] void layernorm(
    const device T* x              [[buffer(0)]],
    const device T* w              [[buffer(1)]],
    const device T* b              [[buffer(2)]],
    device T* out                  [[buffer(3)]],
    const constant float& eps      [[buffer(4)]],
    const constant uint& axis_size [[buffer(5)]],
    uint gid                       [[threadgroup_position_in_grid]],
    uint lid                       [[thread_position_in_threadgroup]],
    uint simd_lane                 [[thread_index_in_simdgroup]],
    uint simd_group                [[simdgroup_index_in_threadgroup]],
    uint tg_size                   [[threads_per_threadgroup]]) {
  threadgroup float reduced[1], partials[32];

  const size_t row = size_t(gid) * size_t(axis_size);
  const device T* xr = x + row;
  device T* outr = out + row;

  // Pass one: the mean.
  float acc = 0.0f;
  for (uint i = lid; i < axis_size; i += tg_size) {
    acc += float(xr[i]);
  }
  const float mean =
      layernorm_group_sum(acc, reduced, partials, simd_lane, simd_group) /
      float(axis_size);
  threadgroup_barrier(mem_flags::mem_threadgroup);

  // Pass two: the centred squares, against that mean and never against a
  // second moment.
  float spread = 0.0f;
  for (uint i = lid; i < axis_size; i += tg_size) {
    const float c = float(xr[i]) - mean;
    spread += c * c;
  }
  const float inv = precise::rsqrt(
      layernorm_group_sum(spread, reduced, partials, simd_lane, simd_group) /
          float(axis_size) +
      eps);

  for (uint i = lid; i < axis_size; i += tg_size) {
    const float c = (float(xr[i]) - mean) * inv;
    outr[i] = static_cast<T>(fma(c, float(w[i]), float(b[i])));
  }
}

#define instantiate_layernorm(name, itype)                            \
  template [[host_name("layernorm_" #name)]]                          \
  [[kernel]] void layernorm<itype>(                                   \
      const device itype*, const device itype*, const device itype*,  \
      device itype*, const constant float&, const constant uint&,     \
      uint, uint, uint, uint, uint);

instantiate_layernorm(bfloat16, bfloat)
