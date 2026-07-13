#include <metal_stdlib>
#include "ptir_rng.generated.metal"
using namespace metal;

struct PtirRngCase {
  uint key;
  uint counter;
  uint index;
  uint reserved;
};

struct PtirReductionResult {
  uint sum_bits;
  uint max_bits;
  uint min_bits;
  uint argmax;
};

kernel void ptir_m0_rng_vectors(
    const device PtirRngCase* cases [[buffer(0)]],
    device ulong* seed_hashes [[buffer(1)]],
    device uint* uniform_bits [[buffer(2)]],
    uint gid [[thread_position_in_grid]]) {
  const PtirRngCase test = cases[gid];
  const ulong seed = ptir_rng_keyed_seed(test.key, test.counter);
  seed_hashes[gid] = seed;
  uniform_bits[gid] =
      as_type<uint>(ptir_rng_hash_uniform(seed, test.index));
}

inline float ptir_canonical_max(float left, float right) {
  const bool left_nan = isnan(left);
  const bool right_nan = isnan(right);
  if (left_nan && right_nan) return -INFINITY;
  if (left_nan) return right;
  if (right_nan) return left;
  if (left == 0.0f && right == 0.0f)
    return signbit(left) && signbit(right) ? -0.0f : 0.0f;
  return max(left, right);
}

inline float ptir_canonical_min(float left, float right) {
  const bool left_nan = isnan(left);
  const bool right_nan = isnan(right);
  if (left_nan && right_nan) return INFINITY;
  if (left_nan) return right;
  if (right_nan) return left;
  if (left == 0.0f && right == 0.0f)
    return signbit(left) || signbit(right) ? -0.0f : 0.0f;
  return min(left, right);
}

kernel void ptir_m0_reduce32(
    const device float* rows [[buffer(0)]],
    const device uint* lengths [[buffer(1)]],
    device PtirReductionResult* results [[buffer(2)]],
    uint3 group [[threadgroup_position_in_grid]],
    uint lane [[thread_index_in_simdgroup]]) {
  const uint row = group.y;
  const uint length = min(lengths[row], 32u);
  const bool active = lane < length;
  const float input = active ? rows[ulong(row) * 32ul + lane] : 0.0f;

  float sum_value = active ? input : 0.0f;
  float max_value = active ? input : -INFINITY;
  float min_value = active ? input : INFINITY;
  uint arg_index = lane;
  uint arg_have = active && !isnan(input) ? 1u : 0u;
  float arg_value = input;

  for (uint offset = 16; offset > 0; offset >>= 1) {
    const float other_sum = simd_shuffle_down(sum_value, offset);
    const float other_max = simd_shuffle_down(max_value, offset);
    const float other_min = simd_shuffle_down(min_value, offset);
    const float other_arg_value = simd_shuffle_down(arg_value, offset);
    const uint other_arg_index = simd_shuffle_down(arg_index, offset);
    const uint other_arg_have = simd_shuffle_down(arg_have, offset);
    if (lane < offset) {
      sum_value += other_sum;
      max_value = ptir_canonical_max(max_value, other_max);
      min_value = ptir_canonical_min(min_value, other_min);
      if (other_arg_have != 0 &&
          (arg_have == 0 || other_arg_value > arg_value ||
           (other_arg_value == arg_value &&
            other_arg_index < arg_index))) {
        arg_value = other_arg_value;
        arg_index = other_arg_index;
        arg_have = 1;
      }
    }
  }

  if (lane == 0) {
    results[row] = {
        as_type<uint>(sum_value),
        as_type<uint>(max_value),
        as_type<uint>(min_value),
        arg_have != 0 ? arg_index : 0u,
    };
  }
}

// M0 storage proof: a generated-kernel-shaped self-loop consumes sequence 0,
// writes sequence 1, then publishes head/tail. The host interpreter observes
// the same cells and words after the command buffer completes.
kernel void ptir_m0_channel_self_loop(
    device uint* cells [[buffer(0)]],
    device ulong* words [[buffer(1)]],
    uint gid [[thread_position_in_grid]]) {
  if (gid != 0) return;
  cells[1] = cells[0] + 1u;
  words[0] = 1ul;
  words[1] = 2ul;
}
