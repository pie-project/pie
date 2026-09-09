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

const ORDER_WORKER_CAP: u32 = 256;

const ORDER_FULL_ROW: u32 = 0;

pub const TOP_K_SELECT_MAX: u32 = 1024;

pub const TOP_K_SELECT_POOL: u32 = 4096;

const BODY: &str = r#"
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
  const m1_u32 order_width = kWidth == kOrderFullRow ? order_len : kWidth;
  const m1_u32 order_count = order_width < order_len ? order_width : order_len;
  m1_u32* order_a = reinterpret_cast<m1_u32*>(temporary);
  m1_u32* order_b = order_a + order_len;

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
  const bool direct = kDirectIntrinsic != 0xFFFFFFFFu;
  const m1_u32 direct_index = dispatch_lane * kIntrinsicSlots + (direct ? kDirectIntrinsic : 0u);
  const m1_u8* direct_base =
      direct ? reinterpret_cast<const m1_u8*>(intrinsic_bases[direct_index]) : nullptr;
  const m1_u32 direct_mode = direct ? intrinsic_modes[direct_index] : 0u;
  const m1_u32 direct_stride_raw = direct ? intrinsic_strides[direct_index] : 0u;
  const m1_u32 direct_stride = direct_stride_raw == 0u ? order_len : direct_stride_raw;
  const m1_u64 direct_row0 =
      direct ? (m1_u64)intrinsic_offsets[direct_index] + (m1_u64)params[kDirectNode].imm2 : 0u;
  const float direct_divisor = (direct && kDirectDivisor != 0xFFFFFFFFu)
      ? m1_load_f(scratch + offsets[kDirectDivisor], 0u, descriptors[kDirectDivisor].dtype)
      : 1.0f;
  auto order_value = [&](m1_u32 order_row, m1_u32 column) -> float {
    if (direct)
      return m1_intrinsic_row_load(direct_base, direct_row0 + order_row, column, direct_stride, direct_mode) / direct_divisor;
    return m1_load_f(order_input, order_row * order_len + column, order_input_desc.dtype);
  };
  __shared__ m1_u32 sel_hist[256];
  __shared__ m1_u32 sel_warp[32];
  __shared__ m1_u64 sel_cand[kSelectCap];
  __shared__ m1_u64 sel_pool[kSelectPool];
  __shared__ m1_u32 sel_pool_fill;
  __shared__ m1_u32 sel_in_bin;
  __shared__ m1_u32 sel_fill;
  __shared__ m1_u32 sel_digit;
  __shared__ m1_u32 sel_below;
  __shared__ m1_u32 sel_equal_seen;
  const m1_u32 sel_lane = threadIdx.x & 31u;
  const m1_u32 sel_warp_id = threadIdx.x >> 5u;
  const m1_u32 sel_warps = (blockDim.x + 31u) >> 5u;
  if (order_count != 0u) {
    for (m1_u32 order_row = lane_row; order_row < order_rows; order_row += lane_blocks) {
      m1_u32 prefix = 0u;
      m1_u32 want = order_count;
      m1_u32 pooled = 0xFFFFFFFFu;
      if (threadIdx.x == 0u) {
        sel_fill = 0u;
        sel_pool_fill = 0u;
      }
      __syncthreads();
      for (m1_u32 pass = 0u; pass < 4u; ++pass) {
        const m1_u32 shift = 24u - pass * 8u;
        const m1_u32 fixed = pass == 0u ? 0u : (0xFFFFFFFFu << (shift + 8u));
        for (m1_u32 b = threadIdx.x; b < 256u; b += blockDim.x) sel_hist[b] = 0u;
        __syncthreads();
        if (pooled == 0xFFFFFFFFu) {
          for (m1_u32 i = threadIdx.x; i < order_len; i += blockDim.x) {
            const m1_u32 key = m1_desc_key(order_value(order_row, i));
            if ((key & fixed) == prefix) atomicAdd(&sel_hist[(key >> shift) & 255u], 1u);
          }
        } else {
          for (m1_u32 i = threadIdx.x; i < pooled; i += blockDim.x) {
            const m1_u32 key = (m1_u32)(sel_pool[i] >> 32);
            if ((key & fixed) == prefix) atomicAdd(&sel_hist[(key >> shift) & 255u], 1u);
          }
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
          sel_in_bin = sel_hist[digit];
        }
        __syncthreads();
        prefix |= sel_digit << shift;
        want -= sel_below;
        const m1_u32 in_bin = sel_in_bin;
        __syncthreads();
        if (pass == 1u && in_bin <= kSelectPool) {
          for (m1_u32 i = threadIdx.x; i < order_len; i += blockDim.x) {
            const m1_u32 key = m1_desc_key(order_value(order_row, i));
            const m1_u32 high = key & 0xFFFF0000u;
            if (high < prefix) {
              const m1_u32 at = atomicAdd(&sel_fill, 1u);
              if (at < kSelectCap) sel_cand[at] = ((m1_u64)key << 32) | (m1_u64)i;
            } else if (high == prefix) {
              const m1_u32 at = atomicAdd(&sel_pool_fill, 1u);
              if (at < kSelectPool) sel_pool[at] = ((m1_u64)key << 32) | (m1_u64)i;
            }
          }
          __syncthreads();
          pooled = sel_pool_fill;
          __syncthreads();
        }
      }
      const m1_u32 threshold = prefix;
      const m1_u32 less_count = order_count - want;
      const m1_u32 equal_take = want;
      if (threadIdx.x == 0u) {
        sel_equal_seen = 0u;
      }
      __syncthreads();
      if (pooled == 0xFFFFFFFFu) {
        for (m1_u32 window = 0u; window < order_len; window += blockDim.x) {
          const m1_u32 i = window + threadIdx.x;
          const bool valid = i < order_len;
          const m1_u32 key = valid
              ? m1_desc_key(order_value(order_row, i))
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
      } else {
        for (m1_u32 at = threadIdx.x; at < pooled; at += blockDim.x) {
          const m1_u64 entry = sel_pool[at];
          const m1_u32 key = (m1_u32)(entry >> 32);
          const m1_u32 index = (m1_u32)entry;
          if (key < threshold) {
            const m1_u32 slot = atomicAdd(&sel_fill, 1u);
            if (slot < kSelectCap) sel_cand[slot] = entry;
          } else if (key == threshold) {
            m1_u32 rank = 0u;
            for (m1_u32 j = 0u; j < pooled; ++j) {
              const m1_u64 other = sel_pool[j];
              rank += ((m1_u32)(other >> 32) == threshold && (m1_u32)other < index) ? 1u : 0u;
            }
            if (rank < equal_take) sel_cand[less_count + rank] = entry;
          }
        }
        __syncthreads();
      }
      for (m1_u32 at = order_count + threadIdx.x; at < kSelectCap; at += blockDim.x)
        sel_cand[at] = 0xFFFFFFFFFFFFFFFFull;
      __syncthreads();
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
      for (m1_u32 at = threadIdx.x; at < order_count; at += blockDim.x) {
        const m1_u32 index = (m1_u32)(sel_cand[at] & 0xFFFFFFFFull);
        m1_store_f(
            order_values,
            order_row * order_width + at,
            order_value(order_row, index));
        m1_store_u(order_indices, order_row * order_width + at, index);
      }
      __syncthreads();
    }
  }
"#;

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
        let _ = writeln!(
            source,
            "  constexpr m1_u32 kSelectPool = {TOP_K_SELECT_POOL}u;"
        );
        let direct = super::fused::analyze_direct_topk(stage)[node];
        let _ = writeln!(
            source,
            "  constexpr m1_u32 kDirectIntrinsic = {}u;",
            direct.map_or(u32::MAX, |d| u32::from(d.intrinsic))
        );
        let _ = writeln!(
            source,
            "  constexpr m1_u32 kDirectNode = {}u;",
            direct.map_or(0, |d| d.node)
        );
        let _ = writeln!(
            source,
            "  constexpr m1_u32 kDirectDivisor = {}u;",
            direct.and_then(|d| d.divisor).unwrap_or(u32::MAX)
        );
        let _ = writeln!(
            source,
            "  constexpr m1_u32 kIntrinsicSlots = {}u;",
            super::fused::PTIR_INTRINSIC_SLOTS
        );
        source.push_str(BODY_SELECT);
    } else {
        source.push_str(BODY);
    }
    source.push_str("}\n");
    Ok(source)
}
