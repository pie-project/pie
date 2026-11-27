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

/// A row's bytes of the streamed top-k's temporaries, as [`BODY`] lays
/// them: a 512-bin histogram, a 4-word pick and a 2-word fill a row
/// (`cand_at`, at most 15 bytes of alignment above them, which
/// [`streamed_topk_temporary_need`] keeps fixed), then [`SELECT_MAX_K`]
/// candidates of a key and an index.
pub const TOPK_ROW_BYTES: u64 = (512 + 4 + 2) * 4 + SELECT_MAX_K as u64 * (8 + 4);

/// What the streamed form of `plan`'s region `region` takes of a lane's
/// temporary area where it is a top-k library region (the one
/// [`emit_streamed_topk`] emits): [`TOPK_ROW_BYTES`] a row of its input,
/// from the area's bottom. `None` for any other region.
#[must_use]
pub fn streamed_topk_temporary_need(
    plan: &crate::codegen::launch::LaunchStagePlan,
    region: usize,
) -> Option<crate::codegen::temporary::TemporaryNeed> {
    let region = plan.fused.get(region)?;
    if region.kind != crate::plan::RegionKind::Library(LibraryOp::TopK) {
        return None;
    }
    let op = plan.ops.get(*region.nodes.first()? as usize)?;
    if op.tag != tags::TOP_K {
        return None;
    }
    Some(crate::codegen::temporary::TemporaryNeed {
        fixed: 15,
        per_row: Some((*op.args.first()?, TOPK_ROW_BYTES)),
        above_whole: false,
    })
}

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

const SMALL_HELPERS: &str = r#"
// The small-k form's candidates: (key << 32) | index, the key 32 bits with
// NaN at 0xFFFFFFFF (no number's key reaches it), so one compare orders by
// key and then index, as the radix form's (flag, key, index) does.
inline ulong m4_sel_entry(float value, uint index) {
  uint flag;
  const uint key = m4_topk_key(value, flag);
  return (ulong(flag != 0u ? 0xFFFFFFFFu : key) << 32) | ulong(index);
}
// `e` into the ascending `top`, dropping its greatest: a min / max chain
// with no branch, so the list stays in registers.
template <uint K>
inline void m4_sel_keep(thread ulong (&top)[K], ulong e) {
  for (uint j = 0u; j < K; ++j) {
    const ulong t = top[j];
    top[j] = t < e ? t : e;
    e = t < e ? e : t;
  }
}
inline ulong m4_sel_from(ulong v, ushort mask) {
  const uint2 halves = as_type<uint2>(v);
  return as_type<ulong>(uint2(simd_shuffle_xor(halves.x, mask), simd_shuffle_xor(halves.y, mask)));
}
// Every lane of the simdgroup ends holding its K least, ascending: at each
// step a lane's list meets its partner's, min(a[j], b[K-1-j]) (a bitonic
// run holding the pair's K least) sorted back in order.
template <uint K>
inline void m4_sel_simd(thread ulong (&top)[K]) {
  for (ushort off = 16u; off > 0u; off >>= 1u) {
    ulong other[K];
    for (uint j = 0u; j < K; ++j) other[j] = m4_sel_from(top[j], off);
    for (uint j = 0u; j < K; ++j) {
      const ulong x = top[j];
      const ulong y = other[K - 1u - j];
      top[j] = x < y ? x : y;
    }
    for (uint stride = K / 2u; stride > 0u; stride >>= 1u) {
      for (uint j = 0u; j < K; ++j) {
        if ((j & stride) == 0u) {
          const ulong x = top[j];
          const ulong y = top[j + stride];
          top[j] = x < y ? x : y;
          top[j + stride] = x < y ? y : x;
        }
      }
    }
  }
}
"#;

/// The small-k form ([`SMALL_K`]): two steps. Each group's threads keep
/// their least candidates of the row in registers on one read of it, the
/// simdgroups merge their lanes' by shuffles and the group's first
/// simdgroup the simdgroups', written as the group's candidates; then one
/// group merges every group's. The pick and its order are the radix
/// form's (the least `(key, index)`).
const SMALL_BODY: &str = r#"
  const M1ValueDesc in_desc = descriptors[kInput];
  const uint rows = in_desc.rows;
  const uint last = in_desc.last;
  // The row is read at its dtype and the values stored at theirs: a bf16
  // element widens exactly and stores back exactly.
  const device uchar* input = scratch + offsets[kInput];
  const uint in_dtype = in_desc.dtype;
  device uchar* top_values = scratch + offsets[kValues];
  const uint top_dtype = descriptors[kValues].dtype;
  device uint* top_indices = reinterpret_cast<device uint*>(scratch + offsets[kIndices]);
  device uint* sel_pick = reinterpret_cast<device uint*>(temporary) + rows * 512u;
  const uint sel_at = ((rows * 512u + rows * 4u + rows * 2u) * 4u + 15u) & ~15u;
  device ulong* sel_cand = reinterpret_cast<device ulong*>(temporary + sel_at);
  const uint count_k = min(kK, last);
  threadgroup ulong sel_lists[32u * kKP];
  const uint sel_lane = m3_tid & 31u;
  const uint sel_simd = m3_tid >> 5u;
  const uint sel_simds = (m3_threads + 31u) >> 5u;
  ulong top[kKP];

  // The group's least kKP of what its threads hold, into simdgroup 0's.
  #define M4_SEL_GROUP()                                                          \
    m4_sel_simd<kKP>(top);                                                        \
    if (sel_lane == 0u) for (uint j = 0u; j < kKP; ++j) sel_lists[sel_simd * kKP + j] = top[j]; \
    threadgroup_barrier(mem_flags::mem_threadgroup);                              \
    if (sel_simd == 0u) {                                                         \
      for (uint j = 0u; j < kKP; ++j) top[j] = sel_lane < sel_simds ? sel_lists[sel_lane * kKP + j] : ~ulong(0); \
      m4_sel_simd<kKP>(top);                                                      \
    }

  if (step.index == 0u) {
    // The groups whose candidates fit a row's candidate area.
    const uint groups = min(m4_groups.x, kCap / kK);
    if (m4_group.x == 0u && m3_tid == 0u) sel_pick[0] = groups;
    if (m4_group.x >= groups) return;
    const uint stride = groups * m3_threads;
    for (uint r = 0u; r < rows; ++r) {
      const uint row_at = r * last;
      for (uint j = 0u; j < kKP; ++j) top[j] = ~ulong(0);
      for (uint i = m4_group.x * m3_threads + m3_tid; i < last; i += stride) {
        m4_sel_keep<kKP>(top, m4_sel_entry(m1_load_f(input, row_at + i, in_dtype), i));
      }
      M4_SEL_GROUP()
      if (sel_simd == 0u && sel_lane == 0u) {
        for (uint j = 0u; j < kK; ++j) sel_cand[ulong(r) * kCap + m4_group.x * kK + j] = top[j];
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return;
  }
  if (step.index == 1u) {
    if (m4_group.x != 0u) return;
    const uint held = sel_pick[0] * kK;
    for (uint r = 0u; r < rows; ++r) {
      const uint row_at = r * last;
      for (uint j = 0u; j < kKP; ++j) top[j] = ~ulong(0);
      for (uint e = m3_tid; e < held; e += m3_threads) m4_sel_keep<kKP>(top, sel_cand[ulong(r) * kCap + e]);
      M4_SEL_GROUP()
      if (sel_simd == 0u && sel_lane == 0u) {
        for (uint j = 0u; j < count_k; ++j) {
          const uint index = uint(top[j] & 0xFFFFFFFFul);
          m1_store_fd(top_values, r * kK + j, m1_load_f(input, row_at + index, in_dtype), top_dtype);
          top_indices[ulong(r) * kK + j] = index;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    return;
  }
"#;

const BODY: &str = r#"
  const M1ValueDesc in_desc = descriptors[kInput];
  const uint rows = in_desc.rows;
  const uint last = in_desc.last;
  const uint total = rows * last;
  // The row is read at its dtype and the values stored at theirs: a bf16
  // element widens exactly and stores back exactly.
  const device uchar* input = scratch + offsets[kInput];
  const uint in_dtype = in_desc.dtype;
  device uchar* top_values = scratch + offsets[kValues];
  const uint top_dtype = descriptors[kValues].dtype;
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
        held[u] = i < total ? m1_load_f(input, i, in_dtype) : 0.0f;
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
        held[u] = i < total ? m1_load_f(input, i, in_dtype) : 0.0f;
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
      const uint row_at = r * last;
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
          const uint key = m4_topk_key(m1_load_f(input, row_at + i, in_dtype), flag);
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
          const uint key = m4_topk_key(m1_load_f(input, row_at + i, in_dtype), flag);
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
        m1_store_fd(top_values, r * kK + p, m1_load_f(input, row_at + index, in_dtype), top_dtype);
        top_indices[ulong(r) * kK + p] = index;
      }
      threadgroup_barrier(mem_flags::mem_device);
    }
    return;
  }
"#;

/// The largest k the two-step small-k form takes ([`SMALL_BODY`]): its
/// lists live in registers, and every group's k fit a row's candidate
/// area (64 groups at most).
pub const SMALL_K: u32 = 16;

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
    let mut source = kernel_head(function_name, stage, used_channel_slots(&ops), "");
    let _ = writeln!(source, "  constexpr uint kInput = {input}u;");
    let _ = writeln!(source, "  constexpr uint kValues = {}u;", bases[topk_node]);
    let _ = writeln!(
        source,
        "  constexpr uint kIndices = {}u;",
        bases[topk_node] + 1
    );
    let _ = writeln!(source, "  constexpr uint kK = {}u;", topk.imm);
    let _ = writeln!(source, "  constexpr uint kCap = {SELECT_MAX_K}u;");
    let small = (1..=SMALL_K).contains(&topk.imm);
    if small {
        // The register lists' length: k rounded up to a power of two (the
        // shuffle merge's bitonic runs).
        let _ = writeln!(
            source,
            "  constexpr uint kKP = {}u;",
            topk.imm.next_power_of_two()
        );
    }
    source.push_str(if small { SMALL_BODY } else { BODY });
    source.push_str("}\n");
    let at = source
        .find("kernel void ")
        .ok_or(EmitError::LibraryRegionAbiInvalid(RegionForm::GroupedTopK))?;
    source.insert_str(at, HELPERS);
    if small {
        let at = source
            .find("kernel void ")
            .ok_or(EmitError::LibraryRegionAbiInvalid(RegionForm::GroupedTopK))?;
        source.insert_str(at, SMALL_HELPERS);
        let steps = alloc::vec![
            streamed_step(input, StepKind::Wide),
            streamed_step(input, StepKind::Single),
        ];
        return Ok((source, steps));
    }
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
