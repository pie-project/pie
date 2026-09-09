use crate::codegen::error::{EmitError, RegionForm};
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write as _;
use eta_ir::op::tags;

use crate::plan::{CompiledStage, LibraryOp, Region};

use super::streamed::{StepKind, kernel_head, streamed_step};
use super::topk::SELECT_MAX_K;
use super::validate::{is_library, library_op_byte, library_region_valid, used_channel_slots};
use crate::codegen::op_view::{OpView, result_bases};

const HELPERS: &str = r#"
inline uint m4_topk_key(float value, thread uint& flag) {
  if (isnan(value)) { flag = 1u; return 0u; }
  flag = 0u;
  if (value == 0.0f) value = 0.0f;
  const uint bits = as_type<uint>(value);
  const uint ascending = (bits & 0x80000000u) != 0u ? ~bits : (bits ^ 0x80000000u);
  return ~ascending;
}
"#;

const BODY: &str = r#"
  const M1ValueDesc in_desc = descriptors[kInput];
  const uint rows = in_desc.rows;
  const uint last = in_desc.last;
  const uint total = rows * last;
  const device float* input = reinterpret_cast<const device float*>(scratch + offsets[kInput]);
  device float* top_values = reinterpret_cast<device float*>(scratch + offsets[kValues]);
  device uint* top_indices = reinterpret_cast<device uint*>(scratch + offsets[kIndices]);
  device atomic_uint* hist = reinterpret_cast<device atomic_uint*>(temporary);
  device uint* pick = reinterpret_cast<device uint*>(temporary) + rows * 512u;      // [row][4]: flag, prefix, remaining, total_lt
  device atomic_uint* fill = reinterpret_cast<device atomic_uint*>(pick + rows * 4u);  // [row][2]: lt, eq
  const uint cand_at = ((rows * 512u + rows * 4u + rows * 2u) * 4u + 15u) & ~15u;
  device ulong* cand_key = reinterpret_cast<device ulong*>(temporary + cand_at);
  device uint* cand_idx = reinterpret_cast<device uint*>(temporary + cand_at + ulong(rows) * kCap * 8u);
  const uint count_k = min(kK, last);
  threadgroup atomic_uint tg_hist[512];
  threadgroup ulong tg_key[kCap];
  threadgroup uint tg_idx[kCap];
  threadgroup uint tg_scan[1024];
  const uint step_index = step.index;

  if (step_index == 0u) {
    if (m4_group.x != 0u) return;
    for (uint b = m3_tid; b < rows * 512u; b += m3_threads) atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
    for (uint b = m3_tid; b < rows * 2u; b += m3_threads) atomic_store_explicit(&fill[b], 0u, memory_order_relaxed);
    for (uint r = m3_tid; r < rows; r += m3_threads) {
      pick[r * 4u + 0u] = 0u;
      pick[r * 4u + 1u] = 0u;
      pick[r * 4u + 2u] = count_k;
      pick[r * 4u + 3u] = 0u;
    }
    return;
  }
  if (step_index == 1u || step_index == 3u || step_index == 5u || step_index == 7u) {
    const uint pass = (step_index - 1u) / 2u;
    const uint shift = 24u - 8u * pass;
    const bool one_row = rows == 1u;
    if (one_row) {
      for (uint b = m3_tid; b < 512u; b += m3_threads) atomic_store_explicit(&tg_hist[b], 0u, memory_order_relaxed);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i0 = m4_gtid; i0 < total; i0 += 8u * m4_gthreads) {
      float held[8];
      for (uint u = 0u; u < 8u; ++u) {
        const uint i = i0 + u * m4_gthreads;
        held[u] = i < total ? input[i] : 0.0f;
      }
      for (uint u = 0u; u < 8u; ++u) {
        const uint i = i0 + u * m4_gthreads;
        if (i >= total) break;
        const uint row = one_row ? 0u : i / last;
        uint flag;
        const uint key = m4_topk_key(held[u], flag);
        uint bin = ~0u;
        if (pass == 0u) {
          bin = (flag << 8u) | (key >> 24u);
        } else {
          const uint sel_flag = pick[row * 4u + 0u];
          const uint prefix = pick[row * 4u + 1u];
          if (flag == sel_flag && (key >> (shift + 8u)) == (prefix >> (shift + 8u))) bin = (key >> shift) & 255u;
        }
        if (bin != ~0u) {
          if (one_row) atomic_fetch_add_explicit(&tg_hist[bin], 1u, memory_order_relaxed);
          else atomic_fetch_add_explicit(&hist[row * 512u + bin], 1u, memory_order_relaxed);
        }
      }
    }
    if (one_row) {
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint b = m3_tid; b < 512u; b += m3_threads) {
        const uint c = atomic_load_explicit(&tg_hist[b], memory_order_relaxed);
        if (c != 0u) atomic_fetch_add_explicit(&hist[b], c, memory_order_relaxed);
      }
    }
    return;
  }
  if (step_index == 2u || step_index == 4u || step_index == 6u || step_index == 8u) {
    if (m4_group.x != 0u) return;
    const uint pass = (step_index - 2u) / 2u;
    const uint shift = 24u - 8u * pass;
    const uint bins = pass == 0u ? 512u : 256u;
    for (uint r = 0u; r < rows; ++r) {
      if (m3_tid == 0u) {
        const uint remaining = pick[r * 4u + 2u];
        uint before = 0u;
        uint digit = 0u;
        for (; digit + 1u < bins; ++digit) {
          const uint here = atomic_load_explicit(&hist[r * 512u + digit], memory_order_relaxed);
          if (before + here >= remaining) break;
          before += here;
        }
        if (pass == 0u) {
          pick[r * 4u + 0u] = digit >> 8u;
          pick[r * 4u + 1u] = (digit & 255u) << 24u;
        } else {
          pick[r * 4u + 1u] = pick[r * 4u + 1u] | (digit << shift);
        }
        pick[r * 4u + 2u] = remaining - before;
        pick[r * 4u + 3u] = count_k - (remaining - before);
      }
      threadgroup_barrier(mem_flags::mem_device);
      for (uint b = m3_tid; b < 512u; b += m3_threads) atomic_store_explicit(&hist[r * 512u + b], 0u, memory_order_relaxed);
      threadgroup_barrier(mem_flags::mem_device);
    }
    return;
  }
  if (step_index == 9u) {
    for (uint i0 = m4_gtid; i0 < total; i0 += 8u * m4_gthreads) {
      float held[8];
      for (uint u = 0u; u < 8u; ++u) {
        const uint i = i0 + u * m4_gthreads;
        held[u] = i < total ? input[i] : 0.0f;
      }
      for (uint u = 0u; u < 8u; ++u) {
        const uint i = i0 + u * m4_gthreads;
        if (i >= total) break;
        const uint row = rows == 1u ? 0u : i / last;
        const uint col = i - row * last;
        uint flag;
        const uint key = m4_topk_key(held[u], flag);
        const ulong full = (ulong(flag) << 32) | ulong(key);
        const ulong pivot = (ulong(pick[row * 4u + 0u]) << 32) | ulong(pick[row * 4u + 1u]);
        const uint total_lt = pick[row * 4u + 3u];
        if (full < pivot) {
          const uint slot = atomic_fetch_add_explicit(&fill[row * 2u], 1u, memory_order_relaxed);
          cand_key[row * kCap + slot] = full;
          cand_idx[row * kCap + slot] = col;
        } else if (full == pivot) {
          const uint slot = atomic_fetch_add_explicit(&fill[row * 2u + 1u], 1u, memory_order_relaxed);
          if (total_lt + slot < kCap) {
            cand_key[row * kCap + total_lt + slot] = full;
            cand_idx[row * kCap + total_lt + slot] = col;
          }
        }
      }
    }
    return;
  }
  if (step_index == 10u) {
    if (m4_group.x != 0u) return;
    for (uint r = 0u; r < rows; ++r) {
      const device float* row_in = input + ulong(r) * last;
      const ulong pivot = (ulong(pick[r * 4u + 0u]) << 32) | ulong(pick[r * 4u + 1u]);
      const uint remaining = pick[r * 4u + 2u];
      const uint total_lt = pick[r * 4u + 3u];
      const uint n_eq = atomic_load_explicit(&fill[r * 2u + 1u], memory_order_relaxed);
      const uint eq_room = kCap - total_lt;
      uint held_n = total_lt + min(n_eq, eq_room);
      for (uint p = m3_tid; p < held_n; p += m3_threads) {
        tg_key[p] = cand_key[r * kCap + p];
        tg_idx[p] = cand_idx[r * kCap + p];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (n_eq > eq_room) {
        const uint chunk_begin = uint((ulong(last) * m3_tid) / m3_threads);
        const uint chunk_end = uint((ulong(last) * (m3_tid + 1u)) / m3_threads);
        uint mine = 0u;
        for (uint i = chunk_begin; i < chunk_end; ++i) {
          uint flag;
          const uint key = m4_topk_key(row_in[i], flag);
          mine += (((ulong(flag) << 32) | ulong(key)) == pivot) ? 1u : 0u;
        }
        tg_scan[m3_tid] = mine;
        threadgroup_barrier(mem_flags::mem_threadgroup);
        if (m3_tid == 0u) {
          uint run = 0u;
          for (uint t = 0u; t < m3_threads; ++t) {
            const uint here = tg_scan[t];
            tg_scan[t] = run;
            run += here;
          }
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
        uint at = tg_scan[m3_tid];
        for (uint i = chunk_begin; i < chunk_end; ++i) {
          uint flag;
          const uint key = m4_topk_key(row_in[i], flag);
          if ((((ulong(flag) << 32) | ulong(key)) == pivot)) {
            if (at < remaining) {
              tg_key[total_lt + at] = pivot;
              tg_idx[total_lt + at] = i;
            }
            ++at;
          }
        }
        held_n = count_k;
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      uint n = 1u;
      while (n < held_n) n <<= 1u;
      for (uint p = held_n + m3_tid; p < n; p += m3_threads) {
        tg_key[p] = ~ulong(0);
        tg_idx[p] = ~0u;
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint size = 2u; size <= n; size <<= 1u) {
        for (uint stride = size >> 1u; stride > 0u; stride >>= 1u) {
          for (uint t = m3_tid; t < n / 2u; t += m3_threads) {
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
      for (uint p = m3_tid; p < count_k; p += m3_threads) {
        const uint index = tg_idx[p];
        top_values[ulong(r) * kK + p] = row_in[index];
        top_indices[ulong(r) * kK + p] = index;
      }
      threadgroup_barrier(mem_flags::mem_device);
    }
    return;
  }
"#;

pub fn emit_streamed_topk(
    function_name: &str,
    stage: &CompiledStage,
    region: &Region,
) -> Result<(String, Vec<u32>), EmitError> {
    if !is_library(region)
        || library_op_byte(region) != LibraryOp::TopK as u8
        || !library_region_valid(stage, region)
    {
        return Err(EmitError::LibraryRegionAbiInvalid(RegionForm::GroupedTopK));
    }
    let ops: Vec<OpView> = OpView::of_all(&stage.normalized.ops);
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
    if topk.imm > SELECT_MAX_K {
        return Err(EmitError::LibraryRegionAbiInvalid(RegionForm::GroupedTopK));
    }
    let input = topk.args[0];
    let mut source = kernel_head(function_name, used_channel_slots(&ops), "");
    let _ = writeln!(source, "  constexpr uint kInput = {input}u;");
    let _ = writeln!(source, "  constexpr uint kValues = {}u;", bases[topk_node]);
    let _ = writeln!(
        source,
        "  constexpr uint kIndices = {}u;",
        bases[topk_node] + 1
    );
    let _ = writeln!(source, "  constexpr uint kK = {}u;", topk.imm);
    let _ = writeln!(source, "  constexpr uint kCap = {SELECT_MAX_K}u;");
    source.push_str(BODY);
    source.push_str("}\n");
    let at = source
        .find("kernel void ")
        .ok_or(EmitError::LibraryRegionAbiInvalid(RegionForm::GroupedTopK))?;
    source.insert_str(at, HELPERS);
    let steps = alloc::vec![
        streamed_step(input, StepKind::Single),
        streamed_step(input, StepKind::Wide),
        streamed_step(input, StepKind::Single),
        streamed_step(input, StepKind::Wide),
        streamed_step(input, StepKind::Single),
        streamed_step(input, StepKind::Wide),
        streamed_step(input, StepKind::Single),
        streamed_step(input, StepKind::Wide),
        streamed_step(input, StepKind::Single),
        streamed_step(input, StepKind::Wide),
        streamed_step(input, StepKind::Single),
    ];
    Ok((source, steps))
}
