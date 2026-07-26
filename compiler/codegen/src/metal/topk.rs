//! `emit_grouped_topk_msl` — the grouped top-k library kernel.
//!
//! Same shape as the nucleus kernel: one threadgroup per (lane, row), a radix
//! ordering of the row, then the first `k` entries written to the value and
//! index results. The `top_k` op defines two results, so the emitter needs the
//! node's result base as well as its argument.

use alloc::string::{String, ToString};
use core::fmt::Write as _;

use pie_plan::{CompiledStage, LibraryOp, Region};

use super::op_view::{OpView, result_bases};
use super::preamble::{RUNTIME_TEMPLATE, grouped_preamble};
use super::validate::{is_library, library_op_byte, library_region_valid};

const OP_TOP_K: u8 = 0x51;

const PROLOGUE: &str = r#"
inline uint m3_topk_order_digit(float value, uint pass) {
  if (pass < 8u) {
    if (isnan(value)) return 0u;
    if (value == 0.0f) value = 0.0f;
    const uint bits = as_type<uint>(value);
    const uint ascending =
        (bits & 0x80000000u) != 0u ? ~bits : (bits ^ 0x80000000u);
    return ((~ascending) >> (pass * 4u)) & 15u;
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
  device M1Status* status =
      reinterpret_cast<device M1Status*>(lanes[lane_index].commit_slot);
  if (status->state != 1u) return;
  const device M1ValueDesc* descriptors =
      all_descriptors + dispatch_lane * layout->value_count;
  const device M1OpParams* lane_params =
      params + dispatch_lane * layout->reserved2;
  device uchar* scratch =
      all_scratch + dispatch_lane * layout->scratch_stride;
  device uchar* temporary = scratch + layout->temporary_offset;
"#;

const BODY: &str = r#"
  const M1ValueDesc input_desc = descriptors[kInput];
  if (row >= input_desc.rows) return;
  const uint len = input_desc.last;
  const device float* input =
      reinterpret_cast<const device float*>(scratch + offsets[kInput]) +
      ulong(row) * len;
  device float* top_values =
      reinterpret_cast<device float*>(scratch + offsets[kValues]);
  device uint* top_indices =
      reinterpret_cast<device uint*>(scratch + offsets[kIndices]);
  device uint* order_a =
      reinterpret_cast<device uint*>(
          temporary + ulong(row) * len * 8ul);
  device uint* order_b = order_a + len;
  threadgroup uint digit_offsets[256 * 16];

  for (uint index = thread_index; index < len; index += threads)
    order_a[index] = index;
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
      ++digit_counts[m3_topk_order_digit(input[index], pass)];
    }
    for (uint digit = 0u; digit < 16u; ++digit)
      digit_offsets[thread_index * 16u + digit] = digit_counts[digit];
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
      const uint digit = m3_topk_order_digit(input[index], pass);
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
    const uint count = min(k, len);
    for (uint position = 0u; position < count; ++position) {
      const uint index = input_order[position];
      top_values[ulong(row) * k + position] = input[index];
      top_indices[ulong(row) * k + position] = index;
    }
  }
  threadgroup_barrier(mem_flags::mem_device);
"#;

/// `emit_grouped_topk_msl`.
pub fn emit_grouped_topk(
    function_name: &str,
    stage: &CompiledStage,
    region: &Region,
) -> Result<String, String> {
    if !is_library(region)
        || library_op_byte(region) != LibraryOp::TopK as u8
        || !library_region_valid(stage, region)
    {
        return Err("invalid grouped TopK library region".to_string());
    }
    let ops: alloc::vec::Vec<OpView> = OpView::of_all(&stage.normalized.ops);
    let bases = result_bases(&ops);
    let topk_node = region.nodes[0] as usize;
    if topk_node >= ops.len() {
        return Err("TopK library node is out of range".to_string());
    }
    let topk = &ops[topk_node];
    if topk.tag != OP_TOP_K
        || topk.args.len() != 1
        || topk.results != 2
        || bases[topk_node] as usize + 1 >= stage.normalized.value_types.len()
    {
        return Err("TopK library node is invalid".to_string());
    }

    let mut source = String::new();
    source.push_str(RUNTIME_TEMPLATE);
    source.push('\n');
    source.push_str(grouped_preamble());
    source.push_str(PROLOGUE);
    source.push_str(function_name);
    source.push_str(SIGNATURE);
    let _ = writeln!(source, "  constexpr uint kInput = {}u;", topk.args[0]);
    let _ = writeln!(source, "  constexpr uint kValues = {}u;", bases[topk_node]);
    let _ = writeln!(
        source,
        "  constexpr uint kIndices = {}u;",
        bases[topk_node] + 1
    );
    let _ = writeln!(source, "  constexpr uint k = {}u;", topk.imm);
    source.push_str(BODY);
    source.push_str("}\n");
    Ok(source)
}
