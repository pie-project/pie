//! `emit_order_region_cuda`: the `Order` library regions, generated.
//!
//! Two ops, one kernel: [`Op::TopK`] and [`Op::SortDesc`]. Both are schedule
//! barriers (a region of their own, [`RegionKind::Library`]) since they
//! read a whole row before writing their first result. `sort_desc` is
//! `top_k` at `k = n`: both define the same two results (values F32, then
//! indices U32) from the same `sort_desc_order`; `kWidth` carries how much
//! of the order is written, with [`ORDER_FULL_ROW`] the sentinel for
//! `sort_desc`, whose width the plan may only know symbolically.
//!
//! # The kernel
//!
//! One block per lane, the fused ABI verbatim. Per row it runs a stable LSD
//! radix sort of the whole row on [`m1_desc_key`] (eight passes of four
//! bits) and writes the first `kWidth` entries of the resulting order to
//! the two results.
//!
//! `m1_desc_key` is the runtime's monotone `float -> u32` map that reverses
//! value order, sends NaN to `0xFFFFFFFF` (sorting NaNs last as a group),
//! and normalises `-0.0` to `+0.0`. Sorting ascending on it walks values
//! descending, and since the sort is stable, equal values come out in
//! ascending index order -- `top_k`'s "ties -> lower index".
//!
//! # Why a full sort and not a selection
//!
//! Cost is `O(8*len)` regardless of `k`. `k` is not small in practice
//! (`locally-typical-sampling` and `tail-free-sampling` pass a caller-set
//! `k_max` up to the vocabulary), so a selection loop's `O(k*vocab)` shape
//! would need a `k` ceiling refused by name to stay honest; a sort needs
//! none.
//!
//! # What it costs in scratch
//!
//! Two `u32` order arrays over one row, taken from the fire's `temporary`
//! arena. `engine::program::scratch::layout` sizes that arena at
//! `4 * sizeof(u32)` per element of the widest value in the stage, so
//! `2 * sizeof(u32) * last` fits with a factor of two to spare.
//!
//! [`RegionKind::Library`]: crate::plan::RegionKind::Library
//! [`Op::TopK`]: eta_ir::op::Op::TopK
//! [`Op::SortDesc`]: eta_ir::op::Op::SortDesc
//! [`m1_desc_key`]: crate::codegen::cuda::runtime

use crate::codegen::error::{EmitError, EmitterKind, RegionForm, ValueLayoutSite};
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write as _;
use eta_ir::op::tags;

use crate::codegen::op_view::{OpView, result_bases};
use crate::codegen::wellformed::{ops_valid, region_ranges_valid, value_types_valid};
use crate::plan::{CompiledStage, LANE_TABLE_ABI_VERSION, LibraryOp, Region, RegionKind};

use super::fused::{PREAMBLE, PROLOGUE, SIGNATURE};
use super::runtime::singleton_runtime_source;
use super::singleton::valid_identifier;

/// `kCudaOrderWorkerCap`: how many threads take part in the counting sort.
/// The per-worker digit table is `workers x 16` `u32`s of shared memory, so
/// every thread being a worker could ask for more than a kernel gets
/// without an opt-in; capping costs occupancy only.
const ORDER_WORKER_CAP: u32 = 256;

/// `kWidth`'s sentinel for "as wide as the row is at run time". Zero cannot
/// collide with a legitimate `top_k` width, since `top_k` refuses `k < 1`.
const ORDER_FULL_ROW: u32 = 0;

/// A `top_k` up to this width takes the SELECTION kernel ([`BODY_SELECT`])
/// rather than the full sort: a radix select of the k-th key in four
/// 8-bit passes over the row, one coalesced pass compacting the winners
/// (keys below the k-th, then the first of the keys equal to it in index
/// order), and a bitonic sort of those in shared memory. It needs no
/// temporary arena, so the engine gives it one block per ROW
/// (`engine-cuda::program::launch::region_rows`). Wider `top_k`, and
/// `sort_desc`, walk the radix sort with its two order arrays.
pub const TOP_K_SELECT_MAX: u32 = 1024;

/// The kernel body, after the six `constexpr`s the emitter writes (worker
/// cap, input value id, two result value ids, result width, full-row
/// sentinel). Every `__syncthreads()` is reached by the whole block: loop
/// bounds are uniform and their guards never contain one.
const BODY: &str = r#"
  // Bound to the fused ABI, and this kernel reads only part of it.
  (void)channels;
  (void)params;
  (void)pending_flags;
  (void)intrinsic_bases;
  (void)intrinsic_modes;
  (void)intrinsic_widths;
  (void)intrinsic_strides;
  (void)intrinsic_offsets;
  (void)lane_active;
  (void)status;

  constexpr m1_u32 kOrderDigits = 16u;
  constexpr m1_u32 kOrderPasses = 8u;
  __shared__ m1_u32 order_offsets[kOrderWorkerCap * kOrderDigits];
  __shared__ m1_u32 order_digit_total[kOrderDigits];
  __shared__ m1_u32 order_digit_base[kOrderDigits];

  const M1ValueDesc order_input_desc = descriptors[kInput];
  const m1_u8* order_input = scratch + offsets[kInput];
  m1_u8* order_values = scratch + offsets[kValues];
  m1_u8* order_indices = scratch + offsets[kIndices];
  const m1_u32 order_len = order_input_desc.last;
  const m1_u32 order_rows = order_input_desc.rows;
  // `sort_desc` writes the whole row and the plan may only know that row's
  // width symbolically, so it asks the descriptor rather than a literal; a
  // `top_k`'s `k` is a trace-known immediate and is spliced in as one.
  const m1_u32 order_width = kWidth == kOrderFullRow ? order_len : kWidth;
  // `k <= last` is `top_k`'s own contract (`infer.rs` refuses the trace
  // otherwise), so the clamp only ever matters if a launch package disagreed
  // with the plan it came from; it truncates rather than reading off the row.
  const m1_u32 order_count = order_width < order_len ? order_width : order_len;
  m1_u32* order_a = reinterpret_cast<m1_u32*>(temporary);
  m1_u32* order_b = order_a + order_len;

  // Contiguous chunks, so worker order IS position order and the counting sort
  // comes out stable. A strided assignment would sort the same multiset into a
  // different tie order.
  const m1_u32 order_workers =
      blockDim.x < kOrderWorkerCap ? blockDim.x : kOrderWorkerCap;
  const bool order_worker = threadIdx.x < order_workers;
  const m1_u32 order_begin =
      order_worker
          ? (m1_u32)(((m1_u64)order_len * threadIdx.x) / order_workers)
          : 0u;
  const m1_u32 order_end =
      order_worker
          ? (m1_u32)(((m1_u64)order_len * (threadIdx.x + 1u)) / order_workers)
          : 0u;

  if (order_len != 0u) {
    // The lane's blocks share its rows: block `lane_row` takes every
    // `lane_blocks`-th row, on its own slice of the temporary arena.
    for (m1_u32 order_row = lane_row; order_row < order_rows; order_row += lane_blocks) {
      const m1_u32 order_base = order_row * order_len;
      for (m1_u32 i = threadIdx.x; i < order_len; i += blockDim.x)
        order_a[i] = i;
      __syncthreads();
      m1_u32* order_in = order_a;
      m1_u32* order_out = order_b;
      for (m1_u32 order_pass = 0u; order_pass < kOrderPasses; ++order_pass) {
        const m1_u32 order_shift = order_pass * 4u;
        m1_u32 order_counts[kOrderDigits];
        m1_u32 order_written[kOrderDigits];
        for (m1_u32 digit = 0u; digit < kOrderDigits; ++digit) {
          order_counts[digit] = 0u;
          order_written[digit] = 0u;
        }
        if (order_worker) {
          for (m1_u32 at = order_begin; at < order_end; ++at) {
            const m1_u32 key = m1_desc_key(m1_load_f(
                order_input, order_base + order_in[at],
                order_input_desc.dtype));
            ++order_counts[(key >> order_shift) & 15u];
          }
          for (m1_u32 digit = 0u; digit < kOrderDigits; ++digit)
            order_offsets[threadIdx.x * kOrderDigits + digit] =
                order_counts[digit];
        }
        __syncthreads();
        // The exclusive scan over (digit, worker) in digit-major order, split
        // sixteen ways rather than run by thread 0 alone: at the cap that is
        // 4096 shared-memory steps per pass on one lane, which dominated the
        // kernel at a real vocabulary.
        if (threadIdx.x < kOrderDigits) {
          m1_u32 total = 0u;
          for (m1_u32 worker = 0u; worker < order_workers; ++worker)
            total += order_offsets[worker * kOrderDigits + threadIdx.x];
          order_digit_total[threadIdx.x] = total;
        }
        __syncthreads();
        if (threadIdx.x == 0u) {
          m1_u32 running = 0u;
          for (m1_u32 digit = 0u; digit < kOrderDigits; ++digit) {
            order_digit_base[digit] = running;
            running += order_digit_total[digit];
          }
        }
        __syncthreads();
        if (threadIdx.x < kOrderDigits) {
          m1_u32 running = order_digit_base[threadIdx.x];
          for (m1_u32 worker = 0u; worker < order_workers; ++worker) {
            const m1_u32 slot = worker * kOrderDigits + threadIdx.x;
            const m1_u32 held = order_offsets[slot];
            order_offsets[slot] = running;
            running += held;
          }
        }
        __syncthreads();
        if (order_worker) {
          for (m1_u32 at = order_begin; at < order_end; ++at) {
            const m1_u32 index = order_in[at];
            const m1_u32 key = m1_desc_key(m1_load_f(
                order_input, order_base + index, order_input_desc.dtype));
            const m1_u32 digit = (key >> order_shift) & 15u;
            order_out[order_offsets[threadIdx.x * kOrderDigits + digit] +
                      order_written[digit]] = index;
            ++order_written[digit];
          }
        }
        __syncthreads();
        m1_u32* order_swap = order_in;
        order_in = order_out;
        order_out = order_swap;
      }
      // The result rows are `order_width` wide, not `order_count` wide: the
      // width is the shape the plan laid out, and a clamped count leaves the
      // tail as the zeros the fire wrote rather than shifting the next row up
      // onto it.
      for (m1_u32 at = threadIdx.x; at < order_count; at += blockDim.x) {
        const m1_u32 index = order_in[at];
        m1_store_f(
            order_values,
            order_row * order_width + at,
            m1_load_f(order_input, order_base + index, order_input_desc.dtype));
        m1_store_u(order_indices, order_row * order_width + at, index);
      }
      __syncthreads();
    }
  }
"#;

/// The selection kernel's body, after the emitter's `constexpr`s (the same
/// six as [`BODY`] plus `kSelectCap`, the power of two at or above
/// `kWidth`). Ordering is [`BODY`]'s exactly — keys ascend on
/// `m1_desc_key`, ties by ascending index — so a `top_k` answers the same
/// rows either way. Every `__syncthreads()` is reached by the whole block:
/// the row loop and the window loop have uniform bounds.
const BODY_SELECT: &str = r#"
  (void)channels;
  (void)params;
  (void)pending_flags;
  (void)intrinsic_bases;
  (void)intrinsic_modes;
  (void)intrinsic_widths;
  (void)intrinsic_strides;
  (void)intrinsic_offsets;
  (void)temporary;
  (void)kOrderWorkerCap;
  (void)kOrderFullRow;
  const M1ValueDesc order_input_desc = descriptors[kInput];
  const m1_u8* order_input = scratch + offsets[kInput];
  m1_u8* order_values = scratch + offsets[kValues];
  m1_u8* order_indices = scratch + offsets[kIndices];
  const m1_u32 order_len = order_input_desc.last;
  const m1_u32 order_rows = order_input_desc.rows;
  const m1_u32 order_width = kWidth;
  const m1_u32 order_count = order_width < order_len ? order_width : order_len;
  __shared__ m1_u32 sel_hist[256];
  __shared__ m1_u32 sel_warp[32];
  __shared__ m1_u64 sel_cand[kSelectCap];
  __shared__ m1_u32 sel_fill;
  __shared__ m1_u32 sel_digit;
  __shared__ m1_u32 sel_below;
  __shared__ m1_u32 sel_equal_seen;
  const m1_u32 sel_lane = threadIdx.x & 31u;
  const m1_u32 sel_warp_id = threadIdx.x >> 5u;
  const m1_u32 sel_warps = (blockDim.x + 31u) >> 5u;
  if (order_count != 0u) {
    for (m1_u32 order_row = lane_row; order_row < order_rows; order_row += lane_blocks) {
      const m1_u32 order_base = order_row * order_len;
      // 1. The `order_count`-th smallest key, one 8-bit digit per pass.
      m1_u32 prefix = 0u;
      m1_u32 want = order_count;
      for (m1_u32 pass = 0u; pass < 4u; ++pass) {
        const m1_u32 shift = 24u - pass * 8u;
        const m1_u32 fixed = pass == 0u ? 0u : (0xFFFFFFFFu << (shift + 8u));
        for (m1_u32 b = threadIdx.x; b < 256u; b += blockDim.x) sel_hist[b] = 0u;
        __syncthreads();
        for (m1_u32 i = threadIdx.x; i < order_len; i += blockDim.x) {
          const m1_u32 key = m1_desc_key(m1_load_f(
              order_input, order_base + i, order_input_desc.dtype));
          if ((key & fixed) == prefix) atomicAdd(&sel_hist[(key >> shift) & 255u], 1u);
        }
        __syncthreads();
        if (threadIdx.x == 0u) {
          m1_u32 below = 0u;
          m1_u32 digit = 0u;
          for (; digit < 255u; ++digit) {
            const m1_u32 c = sel_hist[digit];
            if (below + c >= want) break;
            below += c;
          }
          sel_digit = digit;
          sel_below = below;
        }
        __syncthreads();
        prefix |= sel_digit << shift;
        want -= sel_below;
        __syncthreads();
      }
      const m1_u32 threshold = prefix;
      const m1_u32 less_count = order_count - want;
      const m1_u32 equal_take = want;
      // 2. Compaction, one coalesced window of the row per step: keys below
      // the threshold in any order, keys equal to it by ascending index.
      if (threadIdx.x == 0u) {
        sel_fill = 0u;
        sel_equal_seen = 0u;
      }
      __syncthreads();
      for (m1_u32 window = 0u; window < order_len; window += blockDim.x) {
        const m1_u32 i = window + threadIdx.x;
        const bool valid = i < order_len;
        const m1_u32 key = valid
            ? m1_desc_key(m1_load_f(order_input, order_base + i, order_input_desc.dtype))
            : 0xFFFFFFFFu;
        if (valid && key < threshold) {
          const m1_u32 at = atomicAdd(&sel_fill, 1u);
          if (at < kSelectCap) sel_cand[at] = ((m1_u64)key << 32) | (m1_u64)i;
        }
        const bool equal = valid && key == threshold;
        const unsigned ballot = __ballot_sync(0xffffffffu, equal);
        if (sel_lane == 0u) sel_warp[sel_warp_id] = __popc(ballot);
        __syncthreads();
        if (threadIdx.x == 0u) {
          m1_u32 run = sel_equal_seen;
          for (m1_u32 w = 0u; w < sel_warps; ++w) {
            const m1_u32 c = sel_warp[w];
            sel_warp[w] = run;
            run += c;
          }
          sel_equal_seen = run;
        }
        __syncthreads();
        if (equal) {
          const m1_u32 rank = sel_warp[sel_warp_id] + __popc(ballot & ((1u << sel_lane) - 1u));
          if (rank < equal_take) sel_cand[less_count + rank] = ((m1_u64)key << 32) | (m1_u64)i;
        }
        __syncthreads();
      }
      for (m1_u32 at = order_count + threadIdx.x; at < kSelectCap; at += blockDim.x)
        sel_cand[at] = 0xFFFFFFFFFFFFFFFFull;
      __syncthreads();
      // 3. Bitonic sort of the winners, ascending on (key, index).
      for (m1_u32 size = 2u; size <= kSelectCap; size <<= 1u) {
        for (m1_u32 stride = size >> 1u; stride > 0u; stride >>= 1u) {
          for (m1_u32 i = threadIdx.x; i < kSelectCap; i += blockDim.x) {
            const m1_u32 partner = i ^ stride;
            if (partner > i) {
              const m1_u64 a = sel_cand[i];
              const m1_u64 b = sel_cand[partner];
              const bool ascending = (i & size) == 0u;
              if ((a > b) == ascending) {
                sel_cand[i] = b;
                sel_cand[partner] = a;
              }
            }
          }
          __syncthreads();
        }
      }
      // 4. The result rows are `order_width` wide (see `BODY`).
      for (m1_u32 at = threadIdx.x; at < order_count; at += blockDim.x) {
        const m1_u32 index = (m1_u32)(sel_cand[at] & 0xFFFFFFFFull);
        m1_store_f(
            order_values,
            order_row * order_width + at,
            m1_load_f(order_input, order_base + index, order_input_desc.dtype));
        m1_store_u(order_indices, order_row * order_width + at, index);
      }
      __syncthreads();
    }
  }
"#;

/// Whether `region` is one of the single-node `Order` library regions this
/// emitter serves. The tag is re-read off the node rather than trusted from
/// `region.kind`, so a mislabelled region can't be emitted as a ranking of
/// the wrong operand.
pub fn is_order_region(stage: &CompiledStage, region: &Region) -> bool {
    let expected = match region.kind {
        RegionKind::Library(LibraryOp::TopK) => tags::TOP_K,
        RegionKind::Library(LibraryOp::Sort) => tags::SORT_DESC,
        _ => return false,
    };
    if region.nodes.len() != 1 {
        return false;
    }
    let Some(op) = stage.normalized.ops.get(region.nodes[0].index()) else {
        return false;
    };
    let view = OpView::of(op);
    view.tag == expected && view.args.len() == 1 && view.results == 2
}

/// `emit_order_region_cuda`.
///
/// # Errors
///
/// [`EmitError::EntryNameNotCIdentifier`]: entry not spellable in C.
/// [`EmitError::LibraryRegionAbiInvalid`]: not a single-node `Order` lift
/// this emitter serves. Plus `validate_generated_region`'s
/// plan-well-formedness refusals.
pub fn emit_order_region(
    entry_name: &str,
    stage: &CompiledStage,
    region: &Region,
) -> Result<String, EmitError> {
    if !valid_identifier(entry_name) {
        return Err(EmitError::EntryNameNotCIdentifier(EmitterKind::CudaOrder));
    }
    if !is_order_region(stage, region) {
        return Err(EmitError::LibraryRegionAbiInvalid(RegionForm::CudaOrder));
    }
    value_types_valid(stage)?;
    ops_valid(stage, ValueLayoutSite::CudaFusedStage)?;
    region_ranges_valid(stage, region, RegionForm::CudaOrder)?;

    let ops: Vec<OpView> = OpView::of_all(&stage.normalized.ops);
    let bases = result_bases(&ops);
    let node = region.nodes[0].index();
    let order = &ops[node];
    // The indices value is the id after the values one.
    if bases[node] as usize + 1 >= stage.normalized.value_types.len() {
        return Err(EmitError::RegionNodeOutOfRange(RegionForm::CudaOrder));
    }
    let width = if order.tag == tags::SORT_DESC {
        ORDER_FULL_ROW
    } else {
        order.imm
    };

    let mut source = singleton_runtime_source();
    source.push_str(PROLOGUE);
    source.push_str(entry_name);
    source.push_str(SIGNATURE);
    let _ = write!(source, "{LANE_TABLE_ABI_VERSION}");
    source.push_str(PREAMBLE);
    let _ = writeln!(
        source,
        "  constexpr m1_u32 kOrderWorkerCap = {ORDER_WORKER_CAP}u;"
    );
    let _ = writeln!(
        source,
        "  constexpr m1_u32 kOrderFullRow = {ORDER_FULL_ROW}u;"
    );
    let _ = writeln!(source, "  constexpr m1_u32 kInput = {}u;", order.args[0]);
    let _ = writeln!(source, "  constexpr m1_u32 kValues = {}u;", bases[node]);
    let _ = writeln!(
        source,
        "  constexpr m1_u32 kIndices = {}u;",
        bases[node] + 1
    );
    let _ = writeln!(source, "  constexpr m1_u32 kWidth = {width}u;");
    if order.tag == tags::TOP_K && (1..=TOP_K_SELECT_MAX).contains(&width) {
        let cap = width.next_power_of_two().max(2);
        let _ = writeln!(source, "  constexpr m1_u32 kSelectCap = {cap}u;");
        source.push_str(BODY_SELECT);
    } else {
        source.push_str(BODY);
    }
    source.push_str("}\n");
    Ok(source)
}
