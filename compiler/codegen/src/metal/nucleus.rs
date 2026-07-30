//! `emit_grouped_nucleus_msl` — the grouped nucleus-sampling library kernel.
//!
//! One threadgroup per (lane, row): a radix ordering of the row's logits, a
//! stable prefix sum over the softmax, and the cutoff draw — the whole
//! `nucleus_sample` library op as a single dispatch. The body is one long MSL
//! literal lifted verbatim from the C++ oracle; only the four value slots and
//! the kernel name are interpolated.

use crate::error::{EmitError, RegionForm};
use alloc::string::String;
use core::fmt::Write as _;

use pie_plan::{CompiledStage, LibraryOp, Region};

use super::preamble::{RUNTIME_TEMPLATE, grouped_preamble};
use super::validate::{library_op_byte, library_region_valid};

const PROLOGUE: &str = r#"
inline uint m3_nucleus_order_digit(float value, uint pass) {
  if (pass < 8u) {
    if (isnan(value)) return 0u;
    if (value == 0.0f) value = 0.0f;
    const uint bits = as_type<uint>(value);
    const uint ascending =
        (bits & 0x80000000u) != 0u ? ~bits : (bits ^ 0x80000000u);
    const uint descending = ~ascending;
    return (descending >> (pass * 4u)) & 15u;
  }
  return isnan(value) ? 1u : 0u;
}

kernel void "#;

const SIGNATURE: &str = r#"(
    const device uchar* lane_bytes [[buffer(0)]],
    const device M1ValueDesc* all_descriptors [[buffer(1)]],
    const device M1OpParams* params [[buffer(2)]],
    const device uint* offsets [[buffer(3)]],
    device uchar* all_scratch [[buffer(4)]],
    const device M3GroupLayout* layout [[buffer(5)]],
    const device uint* channel_bindings [[buffer(6)]],
    device uchar* pending_flags [[buffer(7)]],
    const device uint* lane_indices [[buffer(8)]],
    const device M3RowMeta* all_row_meta [[buffer(9)]],
    const device uint* row_indices [[buffer(10)]],
    uint thread_index [[thread_index_in_threadgroup]],
    uint threads [[threads_per_threadgroup]],
    uint group_position [[threadgroup_position_in_grid]]) {
  (void)params;
  (void)channel_bindings;
  (void)pending_flags;
  (void)all_row_meta;
  (void)row_indices;
  if (threads != 256u || layout->reserved1 == 0u) return;
  const uint dispatch_lane = group_position / layout->reserved1;
  const uint row = group_position % layout->reserved1;
  if (dispatch_lane >= layout->lane_count) return;
  const uint lane_index = lane_indices[dispatch_lane];
  const device M3LaneHeader* header =
      reinterpret_cast<const device M3LaneHeader*>(lane_bytes);
  const device M3LaneRecord* lanes =
      reinterpret_cast<const device M3LaneRecord*>(
          lane_bytes + sizeof(M3LaneHeader));
  const M3LaneRecord lane = lanes[lane_index];
  device M1Status* status =
      reinterpret_cast<device M1Status*>(lane.commit_slot);
  if (status->state != 1u) return;
  const device M1ValueDesc* descriptors =
      all_descriptors + dispatch_lane * layout->value_count;
  device uchar* scratch =
      all_scratch + dispatch_lane * layout->scratch_stride;
"#;

const BODY: &str = r#"
  const M1ValueDesc logits_descriptor = descriptors[kLogits];
  if (row >= logits_descriptor.rows) return;
  const uint len = logits_descriptor.last;
  if (len == 0u) {
    if (thread_index == 0u) {
      reinterpret_cast<device int*>(
          scratch + offsets[kOutput])[row] = 0;
    }
    return;
  }
  const device float* logits =
      reinterpret_cast<const device float*>(scratch + offsets[kLogits]) +
      ulong(row) * len;
  const device float* top_p_values =
      reinterpret_cast<const device float*>(scratch + offsets[kTopP]);
  const device uint* state =
      reinterpret_cast<const device uint*>(scratch + offsets[kState]);
  device int* sampled =
      reinterpret_cast<device int*>(scratch + offsets[kOutput]);
  device uchar* workspace_bytes =
      scratch + layout->temporary_offset + ulong(row) * len * 16ul;
  device uint* order_a = reinterpret_cast<device uint*>(workspace_bytes);
  device uint* order_b = order_a + len;
  device float* probabilities =
      reinterpret_cast<device float*>(order_b + len);
  device float* reduction_a = probabilities + len;
  device float* reduction_b =
      reinterpret_cast<device float*>(order_a);

  threadgroup uint digit_offsets[256 * 16];
  threadgroup uint selected_count;
  threadgroup float candidate_values[256];
  threadgroup uint candidate_indices[256];
  threadgroup uchar candidate_have[256];

  for (uint index = thread_index; index < len; index += threads) {
    reduction_a[index] = logits[index];
  }
  threadgroup_barrier(mem_flags::mem_device);
  device float* reduction_input = reduction_a;
  device float* reduction_output = reduction_b;
  uint count = len;
  while (count > 1u) {
    const uint chunks = (count + 31u) / 32u;
    for (uint chunk = thread_index; chunk < chunks; chunk += threads) {
      float values[32];
      for (uint lane_in_chunk = 0; lane_in_chunk < 32u; ++lane_in_chunk) {
        const uint index = chunk * 32u + lane_in_chunk;
        values[lane_in_chunk] =
            index < count ? reduction_input[index] : -INFINITY;
      }
      for (uint offset = 16u; offset > 0u; offset >>= 1u)
        for (uint lane_in_chunk = 0; lane_in_chunk < offset;
             ++lane_in_chunk)
          values[lane_in_chunk] = m1_canonical_max(
              values[lane_in_chunk],
              values[lane_in_chunk + offset]);
      reduction_output[chunk] = values[0];
    }
    threadgroup_barrier(mem_flags::mem_device);
    device float* swap = reduction_input;
    reduction_input = reduction_output;
    reduction_output = swap;
    count = chunks;
  }
  const float maximum_value = reduction_input[0];
  threadgroup_barrier(mem_flags::mem_device);
  for (uint index = thread_index; index < len; index += threads) {
    const float value = precise::exp(logits[index] - maximum_value);
    probabilities[index] = value;
    reduction_a[index] = value;
  }
  threadgroup_barrier(mem_flags::mem_device);
  reduction_input = reduction_a;
  reduction_output = reduction_b;
  count = len;
  while (count > 1u) {
    const uint chunks = (count + 31u) / 32u;
    for (uint chunk = thread_index; chunk < chunks; chunk += threads) {
      float values[32];
      for (uint lane_in_chunk = 0; lane_in_chunk < 32u; ++lane_in_chunk) {
        const uint index = chunk * 32u + lane_in_chunk;
        values[lane_in_chunk] =
            index < count ? reduction_input[index] : 0.0f;
      }
      for (uint offset = 16u; offset > 0u; offset >>= 1u)
        for (uint lane_in_chunk = 0; lane_in_chunk < offset;
             ++lane_in_chunk)
          values[lane_in_chunk] += values[lane_in_chunk + offset];
      reduction_output[chunk] = values[0];
    }
    threadgroup_barrier(mem_flags::mem_device);
    device float* swap = reduction_input;
    reduction_input = reduction_output;
    reduction_output = swap;
    count = chunks;
  }
  const float probability_sum = reduction_input[0];
  threadgroup_barrier(mem_flags::mem_device);
  for (uint index = thread_index; index < len; index += threads) {
    probabilities[index] /= probability_sum;
    order_a[index] = index;
  }
  threadgroup_barrier(mem_flags::mem_device);

  device uint* input_order = order_a;
  device uint* output_order = order_b;
  const uint chunk_begin =
      uint((ulong(len) * thread_index) / threads);
  const uint chunk_end =
      uint((ulong(len) * (thread_index + 1u)) / threads);
  for (uint pass = 0u; pass < 9u; ++pass) {
    uint digit_counts[16];
    uint digit_written[16];
    for (uint digit = 0u; digit < 16u; ++digit) {
      digit_counts[digit] = 0u;
      digit_written[digit] = 0u;
    }
    for (uint position = chunk_begin; position < chunk_end; ++position) {
      const uint index = input_order[position];
      ++digit_counts[
          m3_nucleus_order_digit(probabilities[index], pass)];
    }
    for (uint digit = 0u; digit < 16u; ++digit)
      digit_offsets[thread_index * 16u + digit] =
          digit_counts[digit];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0u) {
      uint base = 0u;
      for (uint digit = 0u; digit < 16u; ++digit) {
        uint running = base;
        for (uint worker = 0u; worker < threads; ++worker) {
          const uint offset = worker * 16u + digit;
          const uint count_for_worker = digit_offsets[offset];
          digit_offsets[offset] = running;
          running += count_for_worker;
        }
        base = running;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint position = chunk_begin; position < chunk_end; ++position) {
      const uint index = input_order[position];
      const uint digit =
          m3_nucleus_order_digit(probabilities[index], pass);
      output_order[
          digit_offsets[thread_index * 16u + digit] +
          digit_written[digit]++] = index;
    }
    threadgroup_barrier(mem_flags::mem_device);
    device uint* swap = input_order;
    input_order = output_order;
    output_order = swap;
  }

  if (thread_index == 0u) {
    const M1ValueDesc top_p_descriptor = descriptors[kTopP];
    const float threshold =
        top_p_values[top_p_descriptor.len <= 1u ? 0u : row];
    float exclusive = 0.0f;
    uint selected = 0u;
    for (uint position = 0u; position < len; ++position) {
      if (!(exclusive < threshold)) break;
      ++selected;
      exclusive += probabilities[input_order[position]];
    }
    selected_count = selected;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);

  const ulong seed = ptir_rng_keyed_seed(state[0], state[1]);
  float best_value = -INFINITY;
  uint best_index = 0u;
  bool have = false;
  for (uint position = thread_index; position < len; position += threads) {
    const uint index = input_order[position];
    const float uniform =
        ptir_rng_hash_uniform(seed, uint(ulong(row) * len + index));
    const float noise =
        -precise::log(-precise::log(uniform));
    const float score =
        (position < selected_count ? logits[index] : -INFINITY) + noise;
    if (!isnan(score) &&
        (!have || score > best_value ||
         (score == best_value && index < best_index))) {
      best_value = score;
      best_index = index;
      have = true;
    }
  }
  candidate_values[thread_index] = best_value;
  candidate_indices[thread_index] = best_index;
  candidate_have[thread_index] = have ? 1u : 0u;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint offset = 128u; offset > 0u; offset >>= 1u) {
    if (thread_index < offset) {
      const uint other = thread_index + offset;
      if (candidate_have[other] != 0u &&
          (candidate_have[thread_index] == 0u ||
           candidate_values[other] > candidate_values[thread_index] ||
           (candidate_values[other] == candidate_values[thread_index] &&
            candidate_indices[other] < candidate_indices[thread_index]))) {
        candidate_values[thread_index] = candidate_values[other];
        candidate_indices[thread_index] = candidate_indices[other];
        candidate_have[thread_index] = 1u;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  if (thread_index == 0u)
    sampled[row] =
        int(candidate_have[0] != 0u ? candidate_indices[0] : 0u);
}
"#;

/// `emit_grouped_nucleus_msl`.
///
/// The C++ guard is `library_op != PTIR_LIBRARY_NUCLEUS_SAMPLE ||
/// !library_region_valid(...)`, which is *not* the same as
/// `nucleus_library_region_valid`: a generated region carries a `library_op`
/// byte of 0 — which is `PTIR_LIBRARY_NUCLEUS_SAMPLE` — and
/// `library_region_valid` waves every non-library region straight through. So
/// a generated region reaches the body and its inputs/outputs are indexed
/// unchecked (the TopK sibling has the `!region.library` test this one is
/// missing). The port keeps the guard as written, so a generated region with a
/// nucleus-shaped operand list still emits the kernel, but adds the arity test
/// the C++ omits rather than reproducing the out-of-bounds read.
pub fn emit_grouped_nucleus(
    function_name: &str,
    stage: &CompiledStage,
    region: &Region,
) -> Result<String, EmitError> {
    let scaled = region.inputs.len() == 5;
    if library_op_byte(region) != LibraryOp::NucleusSample as u8
        || !library_region_valid(stage, region)
        || !(region.inputs.len() == 3 || scaled)
        || region.outputs.is_empty()
    {
        return Err(EmitError::LibraryRegionAbiInvalid(
            RegionForm::GroupedNucleus,
        ));
    }
    // The scaled arity carries the pre-division logits and the divisor ahead of
    // the operands this kernel reads. The scaled logits at index 2 are a real
    // materialized value -- `compile.rs` refuses the match if any library input
    // is produced inside the region, so the Div ran as its own region and left
    // the result in scratch -- so the body below is identical for both forms.
    let logits_value = region.inputs[if scaled { 2 } else { 0 }];
    let top_p_value = region.inputs[if scaled { 3 } else { 1 }];
    let state_value = region.inputs[if scaled { 4 } else { 2 }];
    let output_value = region.outputs[0];

    let mut source = String::new();
    source.push_str(RUNTIME_TEMPLATE);
    source.push('\n');
    source.push_str(grouped_preamble());
    source.push_str(PROLOGUE);
    source.push_str(function_name);
    source.push_str(SIGNATURE);
    let _ = writeln!(source, "  constexpr uint kLogits = {logits_value}u;");
    let _ = writeln!(source, "  constexpr uint kTopP = {top_p_value}u;");
    let _ = writeln!(source, "  constexpr uint kState = {state_value}u;");
    let _ = writeln!(source, "  constexpr uint kOutput = {output_value}u;");
    source.push_str(BODY);
    Ok(source)
}
