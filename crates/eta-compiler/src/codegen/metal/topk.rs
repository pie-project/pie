use crate::codegen::error::{EmitError, RegionForm};
use alloc::string::String;
use core::fmt::Write as _;
use eta_ir::op::tags;

use crate::plan::{CompiledStage, LibraryOp, Region};

use super::preamble::{RUNTIME_TEMPLATE, grouped_preamble};
use super::validate::{is_library, library_op_byte, library_region_valid};
use crate::codegen::op_view::{OpView, result_bases};

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

pub const SIGNATURE: &str = r#"(
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
      all_scratch + ulong(dispatch_lane) * ulong(layout->scratch_stride);
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

pub const SELECT_MAX_K: u32 = 1024;

const SELECT_PROLOGUE: &str = r#"
inline uint m3_topk_key(float value, thread uint& flag) {
  if (isnan(value)) { flag = 1u; return 0u; }
  flag = 0u;
  if (value == 0.0f) value = 0.0f;
  const uint bits = as_type<uint>(value);
  const uint ascending = (bits & 0x80000000u) != 0u ? ~bits : (bits ^ 0x80000000u);
  return ~ascending;
}
"#;

const SELECT_BODY: &str = r#"
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
  const uint count = min(k, len);
  if (count == 0u) return;
  const uint simd_lane_id = thread_index & 31u;
  threadgroup atomic_uint tg_hist[512];
  threadgroup uint tg_scan_lt[256];
  threadgroup uint tg_scan_eq[256];
  threadgroup ulong tg_key[kSelectMax];
  threadgroup uint tg_idx[kSelectMax];
  threadgroup uint tg_pick[4];
  threadgroup atomic_uint tg_fill[2];

  uint sel_flag = 0u, prefix = 0u, remaining = count;
  for (uint pass = 0u; pass < 4u; ++pass) {
    for (uint b = thread_index; b < 512u; b += threads)
      atomic_store_explicit(&tg_hist[b], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint shift = 24u - 8u * pass;
    for (uint i0 = thread_index; i0 < len; i0 += 8u * threads) {
      float held[8];
      for (uint u = 0u; u < 8u; ++u) {
        const uint i = i0 + u * threads;
        held[u] = i < len ? input[i] : 0.0f;
      }
      for (uint u = 0u; u < 8u; ++u) {
        const uint i = i0 + u * threads;
        const bool live = i < len;
        uint flag = 0u;
        const uint key = live ? m3_topk_key(held[u], flag) : 0u;
        if (pass == 0u) {
          uint pending = live ? ((flag << 8u) | (key >> 24u)) : ~0u;
          while (true) {
            const simd_vote vote = simd_ballot(pending != ~0u);
            const uint mask = uint(simd_vote::vote_t(vote));
            if (mask == 0u) break;
            const uint leader = ctz(mask);
            const uint bin = simd_broadcast(pending, ushort(leader));
            const bool same = pending == bin;
            const uint total = simd_sum(same ? 1u : 0u);
            if (simd_lane_id == leader)
              atomic_fetch_add_explicit(&tg_hist[bin], total, memory_order_relaxed);
            if (same) pending = ~0u;
          }
        } else if (live && flag == sel_flag && (key >> (shift + 8u)) == (prefix >> (shift + 8u))) {
          atomic_fetch_add_explicit(&tg_hist[(key >> shift) & 255u], 1u, memory_order_relaxed);
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0u) {
      const uint bins = pass == 0u ? 512u : 256u;
      uint before = 0u;
      uint digit = 0u;
      for (; digit + 1u < bins; ++digit) {
        const uint here = atomic_load_explicit(&tg_hist[digit], memory_order_relaxed);
        if (before + here >= remaining) break;
        before += here;
      }
      if (pass == 0u) {
        tg_pick[0] = digit >> 8u;
        tg_pick[1] = (digit & 255u) << 24u;
      } else {
        tg_pick[0] = sel_flag;
        tg_pick[1] = prefix | (digit << shift);
      }
      tg_pick[2] = remaining - before;
      atomic_store_explicit(&tg_fill[0], 0u, memory_order_relaxed);
      atomic_store_explicit(&tg_fill[1], 0u, memory_order_relaxed);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    sel_flag = tg_pick[0];
    prefix = tg_pick[1];
    remaining = tg_pick[2];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const ulong pivot = (ulong(sel_flag) << 32) | ulong(prefix);
  const uint total_lt = count - remaining;

  const uint eq_room = kSelectMax - total_lt;
  for (uint i0 = thread_index; i0 < len; i0 += 8u * threads) {
    float held[8];
    for (uint u = 0u; u < 8u; ++u) {
      const uint i = i0 + u * threads;
      held[u] = i < len ? input[i] : 0.0f;
    }
    for (uint u = 0u; u < 8u; ++u) {
      const uint i = i0 + u * threads;
      if (i >= len) break;
      uint flag;
      const uint key = m3_topk_key(held[u], flag);
      const ulong full = (ulong(flag) << 32) | ulong(key);
      if (full < pivot) {
        const uint slot = atomic_fetch_add_explicit(&tg_fill[0], 1u, memory_order_relaxed);
        tg_key[slot] = full;
        tg_idx[slot] = i;
      } else if (full == pivot) {
        const uint slot = atomic_fetch_add_explicit(&tg_fill[1], 1u, memory_order_relaxed);
        if (slot < eq_room) {
          tg_key[total_lt + slot] = full;
          tg_idx[total_lt + slot] = i;
        }
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  const uint n_eq = atomic_load_explicit(&tg_fill[1], memory_order_relaxed);
  uint held = total_lt + min(n_eq, eq_room);
  if (n_eq > eq_room) {
    const uint chunk_begin = uint((ulong(len) * thread_index) / threads);
    const uint chunk_end = uint((ulong(len) * (thread_index + 1u)) / threads);
    uint n_mine = 0u;
    for (uint i = chunk_begin; i < chunk_end; ++i) {
      uint flag;
      const uint key = m3_topk_key(input[i], flag);
      n_mine += (((ulong(flag) << 32) | ulong(key)) == pivot) ? 1u : 0u;
    }
    tg_scan_eq[thread_index] = n_mine;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (thread_index == 0u) {
      uint run = 0u;
      for (uint t = 0u; t < threads; ++t) {
        const uint here = tg_scan_eq[t];
        tg_scan_eq[t] = run;
        run += here;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint at = tg_scan_eq[thread_index];
    for (uint i = chunk_begin; i < chunk_end; ++i) {
      uint flag;
      const uint key = m3_topk_key(input[i], flag);
      if ((((ulong(flag) << 32) | ulong(key)) == pivot)) {
        if (at < remaining) {
          tg_key[total_lt + at] = pivot;
          tg_idx[total_lt + at] = i;
        }
        ++at;
      }
    }
    held = count;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  (void)tg_scan_lt;

  uint n = 1u;
  while (n < held) n <<= 1u;
  for (uint p = held + thread_index; p < n; p += threads) {
    tg_key[p] = ~ulong(0);
    tg_idx[p] = ~0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint size = 2u; size <= n; size <<= 1u) {
    for (uint stride = size >> 1u; stride > 0u; stride >>= 1u) {
      for (uint t = thread_index; t < n / 2u; t += threads) {
        const uint lo = 2u * stride * (t / stride) + (t % stride);
        const uint hi = lo + stride;
        const bool ascending = ((lo & size) == 0u);
        const ulong klo = tg_key[lo], khi = tg_key[hi];
        const uint ilo = tg_idx[lo], ihi = tg_idx[hi];
        const bool lo_greater = klo > khi || (klo == khi && ilo > ihi);
        if (lo_greater == ascending) {
          tg_key[lo] = khi; tg_key[hi] = klo;
          tg_idx[lo] = ihi; tg_idx[hi] = ilo;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
  }
  for (uint p = thread_index; p < count; p += threads) {
    const uint index = tg_idx[p];
    top_values[ulong(row) * k + p] = input[index];
    top_indices[ulong(row) * k + p] = index;
  }
  threadgroup_barrier(mem_flags::mem_device);
"#;

pub fn emit_grouped_topk(
    function_name: &str,
    stage: &CompiledStage,
    region: &Region,
) -> Result<String, EmitError> {
    if !is_library(region)
        || library_op_byte(region) != LibraryOp::TopK as u8
        || !library_region_valid(stage, region)
    {
        return Err(EmitError::LibraryRegionAbiInvalid(RegionForm::GroupedTopK));
    }
    let ops: alloc::vec::Vec<OpView> = OpView::of_all(&stage.normalized.ops);
    let bases = result_bases(&ops);
    let topk_node = region.nodes[0].index();
    if topk_node >= ops.len() {
        return Err(EmitError::RegionNodeOutOfRange(RegionForm::GroupedTopK));
    }
    let topk = &ops[topk_node];
    if topk.tag != tags::TOP_K
        || topk.args.len() != 1
        || topk.results != 2
        || bases[topk_node] as usize + 1 >= stage.normalized.value_types.len()
    {
        return Err(EmitError::RegionNodesUnordered(RegionForm::GroupedTopK));
    }

    let mut source = String::new();
    source.push_str(RUNTIME_TEMPLATE);
    source.push('\n');
    source.push_str(grouped_preamble());
    if topk.imm <= SELECT_MAX_K {
        source.push_str(SELECT_PROLOGUE);
    }
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
    if topk.imm <= SELECT_MAX_K {
        let _ = writeln!(source, "  constexpr uint kSelectMax = {SELECT_MAX_K}u;");
        source.push_str(SELECT_BODY);
    } else {
        source.push_str(BODY);
    }
    source.push_str("}\n");
    Ok(source)
}
