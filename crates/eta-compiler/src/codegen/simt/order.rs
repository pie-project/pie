use crate::codegen::error::{EmitError, EmitterKind, RegionForm, ValueLayoutSite};
use alloc::string::String;
use alloc::vec::Vec;
use core::fmt::Write as _;
use eta_ir::op::tags;

use crate::codegen::op_view::{OpView, result_bases};
use crate::codegen::wellformed::{ops_valid, region_ranges_valid, value_types_valid};
use crate::plan::{CompiledStage, LANE_TABLE_ABI_VERSION, LibraryOp, Region, RegionKind};

use super::fused::{PREAMBLE, PROLOGUE, SIGNATURE};
use super::runtime::{Dialect, runtime_source};
use super::singleton::valid_identifier;

const ORDER_WORKER_CAP: u32 = 256;

const ORDER_FULL_ROW: u32 = 0;

pub const TOP_K_SELECT_MAX: u32 = 1024;

pub const TOP_K_SELECT_POOL: u32 = 4096;

/// The widest select cap whose per-thread list a block keeps in registers
/// (`BODY_SELECT_REGISTERS`): a thread holds its `cap` least candidates and
/// the block merges the threads' lists. Sixteen 64-bit entries are the most
/// a CDNA thread holds with no spill at a block of 1,024 threads (77 VGPRs;
/// at 32 it spills); past that the radix select (`BODY_SELECT`) runs.
pub const TOP_K_SELECT_REGISTERS: u32 = 16;

/// The widest select a lane's one row may split across blocks: at least two
/// blocks' candidates, padded to the select's power of two, merge in the
/// pool.
pub const TOP_K_SPLIT_MAX: u32 = TOP_K_SELECT_POOL / 2;

/// The most blocks a select of `width` over a lane's one row splits across:
/// as many as the merge pool holds their candidates (each block's land in
/// the lane's first arena, which holds the pool's).
#[must_use]
pub const fn top_k_split_blocks(width: u32) -> u32 {
    TOP_K_SELECT_POOL / select_cap(width)
}

/// Whether a select of `width` over a lane's one row runs split across the
/// lane's blocks (`BODY_SELECT`): each block selects over its stretch of
/// the row into its lane arena, and the last block to finish merges them.
#[must_use]
pub const fn top_k_splits(width: u32) -> bool {
    width >= 1 && select_cap(width) <= TOP_K_SPLIT_MAX
}

/// A select's candidate count: `width` to a power of two, at least 2.
const fn select_cap(width: u32) -> u32 {
    let cap = width.next_power_of_two();
    if cap < 2 { 2 } else { cap }
}

const BODY: &str = r#"
  (void)channels;
  (void)params;
  (void)pending_flags;
  (void)intrinsic_bases;
  (void)intrinsic_modes;
  (void)intrinsic_widths;
  (void)intrinsic_strides;
  (void)intrinsic_offsets;
  (void)intrinsic_rows;
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
  // The values are copies of the row's elements, at its dtype: a bf16
  // row's are exact.
  const m1_u32 order_values_dtype = descriptors[kValues].dtype;
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
        m1_store_fd(
            order_values,
            order_row * order_width + at,
            m1_load_f(order_input, order_base + index, order_input_desc.dtype),
            order_values_dtype);
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
  (void)intrinsic_rows;
  (void)temporary;
  (void)kOrderWorkerCap;
  (void)kOrderFullRow;
  const M1ValueDesc order_input_desc = descriptors[kInput];
  const m1_u8* order_input = scratch + offsets[kInput];
  m1_u8* order_values = scratch + offsets[kValues];
  // The values are copies of the row's elements, at its dtype: a bf16
  // row's are exact.
  const m1_u32 order_values_dtype = descriptors[kValues].dtype;
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
  const m1_u32 direct_rows = direct ? intrinsic_rows[direct_index] : 0u;
  const float direct_divisor = (direct && kDirectDivisor != 0xFFFFFFFFu)
      ? m1_load_f(scratch + offsets[kDirectDivisor], 0u, descriptors[kDirectDivisor].dtype)
      : 1.0f;
  auto order_value = [&](m1_u32 order_row, m1_u32 column) -> float {
    if (direct)
      return kDirectScales
          ? m1_intrinsic_row_load(direct_base, direct_row0 + order_row, column, direct_stride, direct_mode, direct_rows) * direct_divisor
          : m1_intrinsic_row_load(direct_base, direct_row0 + order_row, column, direct_stride, direct_mode, direct_rows) / direct_divisor;
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
  // A lane's one row on several blocks: each selects over its stretch, and
  // the last to finish merges every block's candidates (`sel_merge`).
  const bool sel_split = kSplit != 0u && lane_blocks > 1u && order_rows == 1u;
  const m1_u32 sel_lo =
      sel_split ? (m1_u32)(((m1_u64)order_len * lane_row) / lane_blocks) : 0u;
  const m1_u32 sel_hi =
      sel_split ? (m1_u32)(((m1_u64)order_len * (lane_row + 1u)) / lane_blocks) : order_len;
  const m1_u32 sel_n = sel_hi - sel_lo;
  const m1_u32 sel_count = order_count < sel_n ? order_count : sel_n;
  if (order_count != 0u) {
    for (m1_u32 order_row = sel_split ? 0u : lane_row; order_row < order_rows;
         order_row += sel_split ? order_rows : lane_blocks) {
      m1_u32 prefix = 0u;
      m1_u32 want = sel_count;
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
          for (m1_u32 i = threadIdx.x; i < sel_n; i += blockDim.x) {
            const m1_u32 key = m1_desc_key(order_value(order_row, sel_lo + i));
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
          for (m1_u32 i = threadIdx.x; i < sel_n; i += blockDim.x) {
            const m1_u32 key = m1_desc_key(order_value(order_row, sel_lo + i));
            const m1_u32 high = key & 0xFFFF0000u;
            if (high < prefix) {
              const m1_u32 at = atomicAdd(&sel_fill, 1u);
              if (at < kSelectCap) sel_cand[at] = ((m1_u64)key << 32) | (m1_u64)(sel_lo + i);
            } else if (high == prefix) {
              const m1_u32 at = atomicAdd(&sel_pool_fill, 1u);
              if (at < kSelectPool) sel_pool[at] = ((m1_u64)key << 32) | (m1_u64)(sel_lo + i);
            }
          }
          __syncthreads();
          pooled = sel_pool_fill;
          __syncthreads();
        }
      }
      const m1_u32 threshold = prefix;
      const m1_u32 less_count = sel_count - want;
      const m1_u32 equal_take = want;
      if (threadIdx.x == 0u) {
        sel_equal_seen = 0u;
      }
      __syncthreads();
      if (pooled == 0xFFFFFFFFu) {
        for (m1_u32 window = 0u; window < sel_n; window += blockDim.x) {
          const m1_u32 i = window + threadIdx.x;
          const bool valid = i < sel_n;
          const m1_u32 key = valid
              ? m1_desc_key(order_value(order_row, sel_lo + i))
              : 0xFFFFFFFFu;
          if (valid && key < threshold) {
            const m1_u32 at = atomicAdd(&sel_fill, 1u);
            if (at < kSelectCap) sel_cand[at] = ((m1_u64)key << 32) | (m1_u64)(sel_lo + i);
          }
          const bool equal = valid && key == threshold;
          const unsigned ballot = m1_chunk_ballot(equal);
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
            if (rank < equal_take)
              sel_cand[less_count + rank] = ((m1_u64)key << 32) | (m1_u64)(sel_lo + i);
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
      for (m1_u32 at = sel_count + threadIdx.x; at < kSelectCap; at += blockDim.x)
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
      const m1_u64* sel_sorted = sel_cand;
      bool sel_lands = true;
      if (sel_split) {
        // Each block's candidates, sorted, into its arena past its line;
        // the last block to count itself in merges them all, in the same
        // (key, index) order a one-block select keeps.
        m1_u8* sel_span = lane_arenas + (m1_u64)dispatch_lane * M1_LANE_SPAN(1u);
        m1_u64* sel_out = reinterpret_cast<m1_u64*>(sel_span + M1_LANE_HEAD) +
            (m1_u64)lane_row * kSelectCap;
        for (m1_u32 at = threadIdx.x; at < kSelectCap; at += blockDim.x)
          M1_LANE_PUT64(sel_out + at, sel_cand[at]);
        M1_LANE_RELEASE();
        __syncthreads();
        if (threadIdx.x == 0u) {
          m1_u32* sel_counter = reinterpret_cast<m1_u32*>(sel_span);
          sel_digit = atomicAdd(sel_counter, 1u) + 1u == lane_blocks ? 1u : 0u;
        }
        __syncthreads();
        sel_lands = sel_digit != 0u;
        if (sel_lands) {
          M1_LANE_ACQUIRE();
          const m1_u32 merged = lane_blocks * kSelectCap;
          for (m1_u32 at = threadIdx.x; at < kSelectPool; at += blockDim.x)
            sel_pool[at] = at < merged
                ? M1_LANE_GET64(reinterpret_cast<m1_u64*>(sel_span + M1_LANE_HEAD) + at)
                : 0xFFFFFFFFFFFFFFFFull;
          __syncthreads();
          if (merged <= blockDim.x) {
            // A thread an entry: its rank is how many entries come before
            // it (no two are equal, their indices differ), and the first
            // `order_count` ranks are the select's, in order.
            if (threadIdx.x < merged) {
              const m1_u64 entry = sel_pool[threadIdx.x];
              m1_u32 rank = 0u;
              for (m1_u32 j = 0u; j < merged; ++j) rank += sel_pool[j] < entry ? 1u : 0u;
              if (rank < order_count) sel_cand[rank] = entry;
            }
            __syncthreads();
            sel_sorted = sel_cand;
          } else {
          m1_u32 sel_merge = 2u;
          while (sel_merge < merged) sel_merge <<= 1u;
          for (m1_u32 size = 2u; size <= sel_merge; size <<= 1u) {
            for (m1_u32 stride = size >> 1u; stride > 0u; stride >>= 1u) {
              for (m1_u32 i = threadIdx.x; i < sel_merge; i += blockDim.x) {
                const m1_u32 partner = i ^ stride;
                if (partner > i) {
                  const m1_u64 a = sel_pool[i];
                  const m1_u64 b = sel_pool[partner];
                  const bool ascending = (i & size) == 0u;
                  if ((a > b) == ascending) {
                    sel_pool[i] = b;
                    sel_pool[partner] = a;
                  }
                }
              }
              __syncthreads();
            }
          }
          sel_sorted = sel_pool;
          }
        }
      }
      if (sel_lands) {
        for (m1_u32 at = threadIdx.x; at < order_count; at += blockDim.x) {
          const m1_u32 index = (m1_u32)(sel_sorted[at] & 0xFFFFFFFFull);
          m1_store_fd(
              order_values,
              order_row * order_width + at,
              order_value(order_row, index),
              order_values_dtype);
          m1_store_u(order_indices, order_row * order_width + at, index);
        }
      }
      __syncthreads();
    }
  }
"#;

/// The register select's helpers: a thread's ascending list of its least
/// candidates (`(key << 32) | index`, all distinct), and the merges of
/// lists across a 32-lane chunk and a block.
const SELECT_REGISTER_HELPERS: &str = r#"
// `e` into the ascending `top`, dropping its greatest: a min/max chain with no
// branch, so the list stays in registers.
template <m1_u32 K>
__device__ __forceinline__ void m1_select_keep(m1_u64 (&top)[K], m1_u64 e) {
#pragma unroll
  for (m1_u32 j = 0u; j < K; ++j) {
    const m1_u64 t = top[j];
    top[j] = t < e ? t : e;
    e = t < e ? e : t;
  }
}
// Every lane of the 32-lane chunk ends holding the chunk's K least, ascending:
// at each step a lane's list meets its partner's, and min(a[j], b[K-1-j]) (a
// bitonic run holding the pair's K least) is merged in order.
template <m1_u32 K>
__device__ __forceinline__ void m1_select_chunk(m1_u64 (&top)[K]) {
  const m1_u32 lane = threadIdx.x & 31u;
#pragma unroll
  for (m1_u32 off = 16u; off > 0u; off >>= 1u) {
    m1_u64 other[K];
#pragma unroll
    for (m1_u32 j = 0u; j < K; ++j) other[j] = m1_chunk_from(top[j], lane ^ off);
#pragma unroll
    for (m1_u32 j = 0u; j < K; ++j) {
      const m1_u64 x = top[j];
      const m1_u64 y = other[K - 1u - j];
      top[j] = x < y ? x : y;
    }
#pragma unroll
    for (m1_u32 stride = K / 2u; stride > 0u; stride >>= 1u) {
#pragma unroll
      for (m1_u32 j = 0u; j < K; ++j) {
        if ((j & stride) == 0u) {
          const m1_u64 x = top[j];
          const m1_u64 y = top[j + stride];
          top[j] = x < y ? x : y;
          top[j + stride] = x < y ? y : x;
        }
      }
    }
  }
}
// The block's `want` least into out[0, want), the rest of out[0, K) past
// every entry: each chunk merges its lanes', the first chunk the chunks'.
template <m1_u32 K>
__device__ __forceinline__ void m1_select_block(
    m1_u64 (&top)[K], m1_u32 want, m1_u64* out, m1_u64* lists) {
  const m1_u32 lane = threadIdx.x & 31u;
  const m1_u32 chunk = threadIdx.x >> 5u;
  const m1_u32 chunks = (blockDim.x + 31u) >> 5u;
  m1_select_chunk<K>(top);
  if (lane == 0u) {
#pragma unroll
    for (m1_u32 j = 0u; j < K; ++j) lists[chunk * K + j] = top[j];
  }
  __syncthreads();
  if (chunk == 0u) {
#pragma unroll
    for (m1_u32 j = 0u; j < K; ++j)
      top[j] = lane < chunks ? lists[lane * K + j] : 0xFFFFFFFFFFFFFFFFull;
    m1_select_chunk<K>(top);
    if (lane == 0u) {
#pragma unroll
      for (m1_u32 j = 0u; j < K; ++j) out[j] = j < want ? top[j] : 0xFFFFFFFFFFFFFFFFull;
    }
  }
  __syncthreads();
}
"#;

/// A select whose cap is at most `TOP_K_SELECT_REGISTERS`: each thread keeps
/// its least candidates of the row in registers as it reads them, and the
/// block merges the lists: one read of the row, no histogram passes. The
/// pick and its order are the radix select's (the least `(key, index)`).
const BODY_SELECT_REGISTERS: &str = r#"
  (void)channels;
  (void)params;
  (void)pending_flags;
  (void)intrinsic_bases;
  (void)intrinsic_modes;
  (void)intrinsic_widths;
  (void)intrinsic_strides;
  (void)intrinsic_offsets;
  (void)intrinsic_rows;
  (void)temporary;
  (void)kOrderWorkerCap;
  (void)kOrderFullRow;
  const M1ValueDesc order_input_desc = descriptors[kInput];
  const m1_u8* order_input = scratch + offsets[kInput];
  m1_u8* order_values = scratch + offsets[kValues];
  // The values are copies of the row's elements, at its dtype: a bf16
  // row's are exact.
  const m1_u32 order_values_dtype = descriptors[kValues].dtype;
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
  const m1_u32 direct_rows = direct ? intrinsic_rows[direct_index] : 0u;
  const float direct_divisor = (direct && kDirectDivisor != 0xFFFFFFFFu)
      ? m1_load_f(scratch + offsets[kDirectDivisor], 0u, descriptors[kDirectDivisor].dtype)
      : 1.0f;
  auto order_value = [&](m1_u32 order_row, m1_u32 column) -> float {
    if (direct)
      return kDirectScales
          ? m1_intrinsic_row_load(direct_base, direct_row0 + order_row, column, direct_stride, direct_mode, direct_rows) * direct_divisor
          : m1_intrinsic_row_load(direct_base, direct_row0 + order_row, column, direct_stride, direct_mode, direct_rows) / direct_divisor;
    return m1_load_f(order_input, order_row * order_len + column, order_input_desc.dtype);
  };
  __shared__ m1_u64 sel_lists[32u * kSelectCap];
  __shared__ m1_u64 sel_cand[kSelectCap];
  __shared__ m1_u64 sel_pool[kSelectPool];
  __shared__ m1_u32 sel_digit;
  // A lane's one row on several blocks: each selects over its stretch, and
  // the last to finish merges every block's candidates.
  const bool sel_split = kSplit != 0u && lane_blocks > 1u && order_rows == 1u;
  const m1_u32 sel_lo =
      sel_split ? (m1_u32)(((m1_u64)order_len * lane_row) / lane_blocks) : 0u;
  const m1_u32 sel_hi =
      sel_split ? (m1_u32)(((m1_u64)order_len * (lane_row + 1u)) / lane_blocks) : order_len;
  const m1_u32 sel_n = sel_hi - sel_lo;
  const m1_u32 sel_count = order_count < sel_n ? order_count : sel_n;
  if (order_count != 0u) {
    for (m1_u32 order_row = sel_split ? 0u : lane_row; order_row < order_rows;
         order_row += sel_split ? order_rows : lane_blocks) {
      m1_u64 sel_top[kSelectCap];
#pragma unroll
      for (m1_u32 j = 0u; j < kSelectCap; ++j) sel_top[j] = 0xFFFFFFFFFFFFFFFFull;
      for (m1_u32 i = sel_lo + threadIdx.x; i < sel_hi; i += blockDim.x)
        m1_select_keep<kSelectCap>(
            sel_top, ((m1_u64)m1_desc_key(order_value(order_row, i)) << 32) | (m1_u64)i);
      m1_select_block<kSelectCap>(sel_top, sel_count, sel_cand, sel_lists);
      const m1_u64* sel_sorted = sel_cand;
      bool sel_lands = true;
      if (sel_split) {
        // Each block's candidates, sorted, into its arena past its line;
        // the last block to count itself in merges them all, in the same
        // (key, index) order a one-block select keeps.
        m1_u8* sel_span = lane_arenas + (m1_u64)dispatch_lane * M1_LANE_SPAN(1u);
        m1_u64* sel_out = reinterpret_cast<m1_u64*>(sel_span + M1_LANE_HEAD) +
            (m1_u64)lane_row * kSelectCap;
        for (m1_u32 at = threadIdx.x; at < kSelectCap; at += blockDim.x)
          M1_LANE_PUT64(sel_out + at, sel_cand[at]);
        M1_LANE_RELEASE();
        __syncthreads();
        if (threadIdx.x == 0u) {
          m1_u32* sel_counter = reinterpret_cast<m1_u32*>(sel_span);
          sel_digit = atomicAdd(sel_counter, 1u) + 1u == lane_blocks ? 1u : 0u;
        }
        __syncthreads();
        sel_lands = sel_digit != 0u;
        if (sel_lands) {
          M1_LANE_ACQUIRE();
          const m1_u32 merged = lane_blocks * kSelectCap;
          for (m1_u32 at = threadIdx.x; at < kSelectPool; at += blockDim.x)
            sel_pool[at] = at < merged
                ? M1_LANE_GET64(reinterpret_cast<m1_u64*>(sel_span + M1_LANE_HEAD) + at)
                : 0xFFFFFFFFFFFFFFFFull;
          __syncthreads();
          if (merged <= blockDim.x) {
            // A thread an entry: its rank is how many entries come before
            // it (no two are equal, their indices differ), and the first
            // `order_count` ranks are the select's, in order.
            if (threadIdx.x < merged) {
              const m1_u64 entry = sel_pool[threadIdx.x];
              m1_u32 rank = 0u;
              for (m1_u32 j = 0u; j < merged; ++j) rank += sel_pool[j] < entry ? 1u : 0u;
              if (rank < order_count) sel_cand[rank] = entry;
            }
            __syncthreads();
            sel_sorted = sel_cand;
          } else {
          m1_u32 sel_merge = 2u;
          while (sel_merge < merged) sel_merge <<= 1u;
          for (m1_u32 size = 2u; size <= sel_merge; size <<= 1u) {
            for (m1_u32 stride = size >> 1u; stride > 0u; stride >>= 1u) {
              for (m1_u32 i = threadIdx.x; i < sel_merge; i += blockDim.x) {
                const m1_u32 partner = i ^ stride;
                if (partner > i) {
                  const m1_u64 a = sel_pool[i];
                  const m1_u64 b = sel_pool[partner];
                  const bool ascending = (i & size) == 0u;
                  if ((a > b) == ascending) {
                    sel_pool[i] = b;
                    sel_pool[partner] = a;
                  }
                }
              }
              __syncthreads();
            }
          }
          sel_sorted = sel_pool;
          }
        }
      }
      if (sel_lands) {
        for (m1_u32 at = threadIdx.x; at < order_count; at += blockDim.x) {
          const m1_u32 index = (m1_u32)(sel_sorted[at] & 0xFFFFFFFFull);
          m1_store_fd(
              order_values,
              order_row * order_width + at,
              order_value(order_row, index),
              order_values_dtype);
          m1_store_u(order_indices, order_row * order_width + at, index);
        }
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
    dialect: Dialect,
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

    let registers = order.tag == tags::TOP_K
        && (1..=TOP_K_SELECT_MAX).contains(&width)
        && width.next_power_of_two().max(2) <= TOP_K_SELECT_REGISTERS;
    let mut source = runtime_source(dialect);
    // A select may split its row across the lane's blocks, meeting in the
    // lane arenas (`BODY_SELECT`, `BODY_SELECT_REGISTERS`).
    if order.tag == tags::TOP_K && (1..=TOP_K_SELECT_MAX).contains(&width) {
        source.push_str(dialect.lane);
        source.push_str(super::sampler::LANE_ARENA);
    }
    if registers {
        source.push_str(SELECT_REGISTER_HELPERS);
    }
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
        let cap = select_cap(width);
        let _ = writeln!(source, "  constexpr m1_u32 kSelectCap = {cap}u;");
        let _ = writeln!(
            source,
            "  constexpr m1_u32 kSplit = {}u;",
            u32::from(top_k_splits(width))
        );
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
            "  constexpr bool kDirectScales = {};",
            direct.is_some_and(|d| d.scales)
        );
        let _ = writeln!(
            source,
            "  constexpr m1_u32 kIntrinsicSlots = {}u;",
            super::fused::output_pitch(stage)
        );
        source.push_str(if registers {
            BODY_SELECT_REGISTERS
        } else {
            BODY_SELECT
        });
    } else {
        source.push_str(BODY);
    }
    source.push_str("}\n");
    Ok(source)
}
