

struct PtirLaneTableHeader {
  m1_u32 abi_version;
  m1_u32 lane_count;
  m1_u32 channel_slots_per_lane;
  m1_u32 flags;
};

struct PtirLaneRecord {
  m1_u64 logits_base;
  m1_u32 logits_row_offset;
  m1_u32 logits_row_count;
  m1_u32 kv_len;
  m1_u32 page_count;
  m1_u32 row_count;
  m1_u32 token_count;
  m1_u32 sampled_rows;
  m1_u32 query_len;
  m1_u32 key_len;
  m1_u32 channel_slot_offset;
  m1_u64 rng_state;
  m1_u64 commit_slot;
  m1_u64 active_row_mask;
  m1_u64 sample_output_channel_mask;
  m1_u64 row_valid;
  m1_u32 row_valid_offset;
  m1_u32 reserved0;
};

struct PtirLaneChannelSlot {
  m1_u64 committed_cell;
  m1_u64 pending_cell;
  m1_u64 expected_head;
  m1_u64 expected_tail;
};

static_assert(sizeof(PtirLaneTableHeader) == 16, "lane header ABI");
static_assert(sizeof(PtirLaneRecord) == 96, "lane record ABI");
static_assert(sizeof(PtirLaneChannelSlot) == 32, "lane channel ABI");

__device__ __forceinline__ void ptir_parallel_copy(
    const m1_u8* input,
    m1_u8* output,
    m1_u32 len,
    m1_u32 dtype) {
  const m1_u32 bytes = dtype == 3u ? len : len * 4u;
  for (m1_u32 index = threadIdx.x; index < bytes; index += blockDim.x)
    output[index] = input[index];
}

__device__ __forceinline__ void ptir_parallel_intrinsic(
    const m1_u8* input,
    m1_u8* output,
    const M1ValueDesc output_desc,
    const M1OpParams p) {
  const m1_u32 width = output_desc.last == 0u ? p.imm : output_desc.last;
  const m1_u32 stride =
      p.intrinsic_row_stride == 0u ? width : p.intrinsic_row_stride;
  const m1_u64 first_row =
      (m1_u64)p.intrinsic_row_offset + (m1_u64)p.imm2;
  for (m1_u32 index = threadIdx.x;
       index < output_desc.len;
       index += blockDim.x) {
    const m1_u32 row = index / width;
    const m1_u32 column = index % width;
    m1_store_f(
        output,
        index,
        m1_intrinsic_row_load(
            input,
            first_row + row,
            column,
            stride,
            p.intrinsic_dtype));
  }
}

__device__ __forceinline__ void ptir_parallel_elementwise(
    m1_u32 tag,
    M1Status* status,
    const M1ValueDesc* descriptors,
    const M1OpParams p,
    const m1_u8* a0,
    const m1_u8* a1,
    const m1_u8* a2,
    m1_u8* o0) {
  const M1ValueDesc d0 = descriptors[p.a0];
  const M1ValueDesc d1 = descriptors[p.a1];
  const M1ValueDesc d2 = descriptors[p.a2];
  const M1ValueDesc out = descriptors[p.o0];
  for (m1_u32 i = threadIdx.x; i < out.len; i += blockDim.x) {
    const m1_u32 xindex = m1_pick(d0.len, i);
    const m1_u32 yindex = m1_pick(d1.len, i);
    if (tag == 0x01u || tag == 0x02u || tag == 0x04u) {
      const float value = m1_load_f(a0, xindex, d0.dtype);
      m1_store_f(
          o0,
          i,
          tag == 0x01u
              ? expf(value)
              : (tag == 0x02u ? logf(value) : 1.0f / value));
      continue;
    }
    if (tag == 0x03u || tag == 0x05u || tag == 0x06u) {
      if (d0.dtype == 0u) {
        const float value = m1_load_f(a0, xindex, d0.dtype);
        const float result =
            tag == 0x03u
                ? -value
                : (tag == 0x05u
                       ? fabsf(value)
                       : (value > 0.0f
                              ? 1.0f
                              : (value < 0.0f ? -1.0f : 0.0f)));
        m1_store_f(o0, i, result);
      } else if (d0.dtype == 1u) {
        const int value = m1_load_i(a0, xindex, d0.dtype);
        int result = value;
        if (tag == 0x03u) result = (int)(0u - (m1_u32)value);
        else if (tag == 0x05u)
          result = (m1_u32)value == 0x80000000u
              ? value
              : (value < 0 ? -value : value);
        else result = value > 0 ? 1 : (value < 0 ? -1 : 0);
        m1_store_i(o0, i, result);
      } else if (d0.dtype == 2u) {
        const m1_u32 value = m1_load_u(a0, xindex, d0.dtype);
        m1_store_u(
            o0,
            i,
            tag == 0x03u
                ? 0u - value
                : (tag == 0x06u ? (value != 0u ? 1u : 0u) : value));
      } else {
        if (threadIdx.x == 0) m1_fault(status, tag);
        return;
      }
      continue;
    }
    if (tag == 0x07u) {
      if (out.dtype == 0u)
        m1_store_f(o0, i, m1_load_f(a0, xindex, d0.dtype));
      else if (out.dtype == 1u)
        m1_store_i(o0, i, m1_load_i(a0, xindex, d0.dtype));
      else if (out.dtype == 2u)
        m1_store_u(o0, i, m1_load_u(a0, xindex, d0.dtype));
      else
        m1_store_b(o0, i, m1_load_b(a0, xindex, d0.dtype));
      continue;
    }
    if ((tag >= 0x10u && tag <= 0x1du) || tag == 0x1fu) {
      if (tag >= 0x16u && tag <= 0x1du) {
        bool result = false;
        if (tag == 0x1cu || tag == 0x1du) {
          const bool left = m1_load_b(a0, xindex, d0.dtype);
          const bool right = m1_load_b(a1, yindex, d1.dtype);
          result = tag == 0x1cu ? left && right : left || right;
        } else if (d0.dtype == 0u) {
          const float left = m1_load_f(a0, xindex, d0.dtype);
          const float right = m1_load_f(a1, yindex, d1.dtype);
          if (tag == 0x16u) result = left > right;
          else if (tag == 0x17u) result = left >= right;
          else if (tag == 0x18u) result = left == right;
          else if (tag == 0x19u) result = left != right;
          else if (tag == 0x1au) result = left < right;
          else result = left <= right;
        } else if (d0.dtype == 1u) {
          const int left = m1_load_i(a0, xindex, d0.dtype);
          const int right = m1_load_i(a1, yindex, d1.dtype);
          if (tag == 0x16u) result = left > right;
          else if (tag == 0x17u) result = left >= right;
          else if (tag == 0x18u) result = left == right;
          else if (tag == 0x19u) result = left != right;
          else if (tag == 0x1au) result = left < right;
          else result = left <= right;
        } else {
          const m1_u32 left = m1_load_u(a0, xindex, d0.dtype);
          const m1_u32 right = m1_load_u(a1, yindex, d1.dtype);
          if (tag == 0x16u) result = left > right;
          else if (tag == 0x17u) result = left >= right;
          else if (tag == 0x18u) result = left == right;
          else if (tag == 0x19u) result = left != right;
          else if (tag == 0x1au) result = left < right;
          else result = left <= right;
        }
        m1_store_b(o0, i, result);
      } else if (d0.dtype == 0u) {
        const float left = m1_load_f(a0, xindex, d0.dtype);
        const float right = m1_load_f(a1, yindex, d1.dtype);
        float result = 0.0f;
        if (tag == 0x10u) result = left + right;
        else if (tag == 0x11u) result = left - right;
        else if (tag == 0x12u) result = left * right;
        else if (tag == 0x13u) result = left / right;
        else if (tag == 0x14u) result = m1_element_max(left, right);
        else if (tag == 0x15u) result = m1_element_min(left, right);
        else result = fmodf(left, right);
        m1_store_f(o0, i, result);
      } else if (d0.dtype == 1u) {
        const int left = m1_load_i(a0, xindex, d0.dtype);
        const int right = m1_load_i(a1, yindex, d1.dtype);
        int result = 0;
        if (tag == 0x10u) result = (int)((m1_u32)left + (m1_u32)right);
        else if (tag == 0x11u)
          result = (int)((m1_u32)left - (m1_u32)right);
        else if (tag == 0x12u)
          result = (int)((m1_u32)left * (m1_u32)right);
        else if (tag == 0x13u) result = m1_i32_div(left, right);
        else if (tag == 0x14u) result = left > right ? left : right;
        else if (tag == 0x15u) result = left < right ? left : right;
        else result = m1_i32_rem(left, right);
        m1_store_i(o0, i, result);
      } else {
        const m1_u32 left = m1_load_u(a0, xindex, d0.dtype);
        const m1_u32 right = m1_load_u(a1, yindex, d1.dtype);
        m1_u32 result = 0u;
        if (tag == 0x10u) result = left + right;
        else if (tag == 0x11u) result = left - right;
        else if (tag == 0x12u) result = left * right;
        else if (tag == 0x13u) result = right == 0u ? 0u : left / right;
        else if (tag == 0x14u) result = left > right ? left : right;
        else if (tag == 0x15u) result = left < right ? left : right;
        else result = right == 0u ? 0u : left % right;
        m1_store_u(o0, i, result);
      }
      continue;
    }
    if (tag == 0x1eu) {
      m1_store_b(o0, i, !m1_load_b(a0, xindex, d0.dtype));
      continue;
    }
    if (tag == 0x20u) {
      const bool condition = m1_load_b(a0, xindex, d0.dtype);
      const m1_u32 left_index = m1_pick(d1.len, i);
      const m1_u32 right_index = m1_pick(d2.len, i);
      if (out.dtype == 0u)
        m1_store_f(
            o0,
            i,
            condition
                ? m1_load_f(a1, left_index, d1.dtype)
                : m1_load_f(a2, right_index, d2.dtype));
      else if (out.dtype == 1u)
        m1_store_i(
            o0,
            i,
            condition
                ? m1_load_i(a1, left_index, d1.dtype)
                : m1_load_i(a2, right_index, d2.dtype));
      else if (out.dtype == 2u)
        m1_store_u(
            o0,
            i,
            condition
                ? m1_load_u(a1, left_index, d1.dtype)
                : m1_load_u(a2, right_index, d2.dtype));
      else
        m1_store_b(
            o0,
            i,
            condition
                ? m1_load_b(a1, left_index, d1.dtype)
                : m1_load_b(a2, right_index, d2.dtype));
      continue;
    }
    if (tag == 0x64u) {
      m1_store_u(o0, i, i);
      continue;
    }
    if (tag == 0x65u) {
      const m1_u32 width = d0.rank == 0u ? 1u : d0.dims[d0.rank - 1u];
      const m1_u32 column = i % width;
      const m1_u32 word = column >> 5;
      const m1_u32 mask =
          word < d1.len ? m1_load_u(a1, word, d1.dtype) : 0u;
      m1_store_f(
          o0,
          i,
          ((mask >> (column & 31u)) & 1u) != 0u
              ? m1_load_f(a0, i, d0.dtype)
              : m1_neg_inf());
      continue;
    }
    if (tag == 0x66u || tag == 0x67u || tag == 0x68u) {
      const m1_u32 key_count = p.imm;
      const m1_u32 position_index =
          key_count == 0u ? 0u : i / key_count;
      const m1_u32 key = key_count == 0u ? 0u : i % key_count;
      const m1_u32 position =
          m1_load_u(a0, position_index, d0.dtype);
      bool allowed = key_count != 0u && key <= position;
      if (allowed && tag != 0x66u) {
        const m1_u32 window = tag == 0x67u ? p.imm2 : p.imm3;
        const m1_u32 reach =
            key > 0xffffffffu - window ? 0xffffffffu : key + window;
        const bool recent = reach > position;
        allowed = tag == 0x67u ? recent : (key < p.imm2 || recent);
      }
      m1_store_b(o0, i, allowed);
      continue;
    }
    if (tag == 0x70u || tag == 0x71u) {
      m1_u64 seed;
      if (tag == 0x70u) {
        seed = ptir_rng_seed_eff_stream((m1_u32)p.rng_seed, p.imm);
      } else {
        const m1_u32 key = m1_load_u(a0, 0u, d0.dtype);
        const m1_u32 counter =
            d0.len > 1u ? m1_load_u(a0, 1u, d0.dtype) : 0u;
        seed = ptir_rng_keyed_seed(key, counter);
      }
      const float uniform = ptir_rng_hash_uniform(seed, i);
      m1_store_f(
          o0,
          i,
          p.kind == 0u ? uniform : -logf(-logf(uniform)));
      continue;
    }
  }
}

__device__ __forceinline__ void ptir_copy_element(
    const m1_u8* input,
    m1_u8* output,
    m1_u32 source,
    m1_u32 destination,
    m1_u32 dtype) {
  if (dtype == 0u)
    m1_store_f(output, destination, m1_load_f(input, source, dtype));
  else if (dtype == 1u)
    m1_store_i(output, destination, m1_load_i(input, source, dtype));
  else if (dtype == 2u)
    m1_store_u(output, destination, m1_load_u(input, source, dtype));
  else
    m1_store_b(output, destination, m1_load_b(input, source, dtype));
}

__device__ __forceinline__ void ptir_zero_element(
    m1_u8* output,
    m1_u32 destination,
    m1_u32 dtype) {
  if (dtype == 0u)
    m1_store_f(output, destination, 0.0f);
  else if (dtype == 1u)
    m1_store_i(output, destination, 0);
  else if (dtype == 2u)
    m1_store_u(output, destination, 0u);
  else
    m1_store_b(output, destination, false);
}

__device__ __forceinline__ void ptir_parallel_broadcast(
    const m1_u8* input,
    m1_u8* output,
    const M1ValueDesc input_desc,
    const M1ValueDesc output_desc) {
  for (m1_u32 linear = threadIdx.x;
       linear < output_desc.len;
       linear += blockDim.x) {
    m1_u32 rem = linear;
    m1_u32 source_index = 0;
    m1_u32 source_stride[4] = {1, 1, 1, 1};
    for (int dim = (int)output_desc.rank - 2; dim >= 0; --dim) {
      source_stride[dim] =
          source_stride[dim + 1] *
          ((m1_u32)(dim + 1) < input_desc.rank
               ? input_desc.dims[dim + 1]
               : 1u);
    }
    for (m1_u32 dim = 0; dim < output_desc.rank; ++dim) {
      m1_u32 stride = 1;
      for (m1_u32 next = dim + 1; next < output_desc.rank; ++next)
        stride *= output_desc.dims[next];
      if (stride == 0u) stride = 1u;
      const m1_u32 coordinate = rem / stride;
      rem %= stride;
      const m1_u32 source_dim =
          dim < input_desc.rank ? input_desc.dims[dim] : 1u;
      if (source_dim != 1u)
        source_index += coordinate * source_stride[dim];
    }
    ptir_copy_element(
        input, output, source_index, linear, output_desc.dtype);
  }
}

__device__ __forceinline__ void ptir_parallel_transpose(
    M1Status* status,
    const m1_u8* input,
    m1_u8* output,
    const M1ValueDesc input_desc,
    const M1ValueDesc output_desc) {
  if (input_desc.rank != 2u) {
    if (threadIdx.x == 0u) m1_fault(status, 0x3au);
    return;
  }
  const m1_u32 rows = input_desc.dims[0];
  const m1_u32 columns = input_desc.dims[1];
  for (m1_u32 index = threadIdx.x;
       index < rows * columns;
       index += blockDim.x) {
    const m1_u32 source =
        (index % rows) * columns + index / rows;
    ptir_copy_element(
        input, output, source, index, output_desc.dtype);
  }
}

__device__ __forceinline__ void ptir_parallel_gather(
    m1_u32 tag,
    const m1_u8* input,
    const m1_u8* indices,
    m1_u8* output,
    const M1ValueDesc input_desc,
    const M1ValueDesc index_desc,
    const M1ValueDesc output_desc) {
  if (tag == 0x61u) {
    const m1_u32 rows = input_desc.dims[0];
    const m1_u32 columns = input_desc.dims[1];
    for (m1_u32 row = threadIdx.x; row < rows; row += blockDim.x) {
      const long long column =
          m1_load_index(indices, row, index_desc.dtype);
      const bool valid =
          column >= 0 && (m1_u64)column < columns;
      if (valid)
        ptir_copy_element(
            input,
            output,
            row * columns + (m1_u32)column,
            row,
            output_desc.dtype);
      else
        ptir_zero_element(output, row, output_desc.dtype);
    }
    return;
  }
  m1_u32 rest = 1u;
  m1_u32 rows = 1u;
  if (input_desc.rank != 0u) {
    rows = input_desc.dims[0];
    rest = rows == 0u ? 1u : input_desc.len / rows;
  }
  const m1_u32 total = index_desc.len * rest;
  for (m1_u32 output_index = threadIdx.x;
       output_index < total;
       output_index += blockDim.x) {
    const m1_u32 k = output_index / rest;
    const m1_u32 r = output_index % rest;
    const long long row =
        m1_load_index(indices, k, index_desc.dtype);
    const bool valid = row >= 0 && (m1_u64)row < rows;
    if (valid)
      ptir_copy_element(
          input,
          output,
          (m1_u32)row * rest + r,
          output_index,
          output_desc.dtype);
    else
      ptir_zero_element(output, output_index, output_desc.dtype);
  }
}

// cummass_le (top-p / nucleus): keep the DESCENDING prefix whose EXCLUSIVE
// cumulative mass stays below `p` (interp.rs `Predicate::CummassLe`: sort the
// row descending, then `k[i] = excl < p; excl += row[i]`). The input is NOT
// sorted, so this is a block-cooperative selection loop -- one block-wide
// "next largest still-unpicked element" pick per iteration, carrying the
// previous pick as a total-order threshold instead of a visited set (the same
// technique as tier0's `k_pivot_cummassle`). It stops as soon as the running
// mass clears `p`, so a peaked LM row costs a handful of passes.
//
// This replaces the single-threaded M1 reference, which was O(len^3) on
// thread 0 alone (a selection sort with a linear "already picked" rescan per
// candidate) and therefore never returned at a 151936-token vocabulary --
// every hand-written top-p sampler that failed to match
// `LibraryOp::NucleusSample` wedged the GPU here.
__device__ __forceinline__ void ptir_parallel_pivot_cummass(
    const m1_u8* input,
    const m1_u8* threshold,
    m1_u8* output,
    const M1ValueDesc input_desc,
    const M1ValueDesc threshold_desc) {
  __shared__ float pivot_share_value[32];
  __shared__ m1_u32 pivot_share_index[32];
  __shared__ float pivot_previous_value;
  __shared__ m1_u32 pivot_previous_index;
  __shared__ float pivot_exclusive;
  __shared__ m1_u32 pivot_stop;
  const m1_u32 none = 0xFFFFFFFFu;
  const m1_u32 len = input_desc.last;
  const m1_u32 rows = input_desc.rows;
  for (m1_u32 flat = threadIdx.x;
       flat < input_desc.len;
       flat += blockDim.x)
    m1_store_b(output, flat, false);
  __syncthreads();
  if (len == 0u) return;
  for (m1_u32 row = 0; row < rows; ++row) {
    const m1_u32 base = row * len;
    const float cutoff = m1_load_f(
        threshold,
        m1_pick(threshold_desc.len, row),
        threshold_desc.dtype);
    if (threadIdx.x == 0u) {
      // Sentinel sorts before every real element, so the first pick is free.
      pivot_previous_value = m1_pos_inf();
      pivot_previous_index = 0u;
      pivot_exclusive = 0.0f;
      pivot_stop = 0u;
    }
    __syncthreads();
    for (m1_u32 pick = 0; pick < len; ++pick) {
      if (pivot_stop != 0u) break;
      const float previous_value = pivot_previous_value;
      const m1_u32 previous_index = pivot_previous_index;
      float best_value = 0.0f;
      m1_u32 best_index = none;
      for (m1_u32 i = threadIdx.x; i < len; i += blockDim.x) {
        const float value = m1_load_f(input, base + i, input_desc.dtype);
        if (!m1_sort_better(previous_value, previous_index, value, i))
          continue;
        if (best_index == none ||
            m1_sort_better(value, i, best_value, best_index)) {
          best_value = value;
          best_index = i;
        }
      }
      for (m1_u32 offset = 16u; offset > 0u; offset >>= 1) {
        const float other_value =
            __shfl_down_sync(0xFFFFFFFFu, best_value, offset);
        const m1_u32 other_index =
            __shfl_down_sync(0xFFFFFFFFu, best_index, offset);
        if (other_index != none &&
            (best_index == none ||
             m1_sort_better(
                 other_value, other_index, best_value, best_index))) {
          best_value = other_value;
          best_index = other_index;
        }
      }
      if ((threadIdx.x & 31u) == 0u) {
        pivot_share_value[threadIdx.x >> 5] = best_value;
        pivot_share_index[threadIdx.x >> 5] = best_index;
      }
      __syncthreads();
      if (threadIdx.x == 0u) {
        const m1_u32 warps = (blockDim.x + 31u) >> 5;
        float value = pivot_share_value[0];
        m1_u32 index = pivot_share_index[0];
        for (m1_u32 warp = 1u; warp < warps; ++warp) {
          if (pivot_share_index[warp] != none &&
              (index == none ||
               m1_sort_better(
                   pivot_share_value[warp],
                   pivot_share_index[warp],
                   value,
                   index))) {
            value = pivot_share_value[warp];
            index = pivot_share_index[warp];
          }
        }
        // Descending order ⇒ once the mass condition fails it fails for every
        // remaining element, so the zero-initialised tail is already correct.
        if (index == none || !(pivot_exclusive < cutoff)) {
          pivot_stop = 1u;
        } else {
          m1_store_b(output, base + index, true);
          pivot_exclusive += value;
          pivot_previous_value = value;
          pivot_previous_index = index;
        }
      }
      __syncthreads();
    }
    __syncthreads();
  }
}

// rank_le (top-k by rank): keep the elements whose count of strictly-greater
// values is below `k` (interp.rs `Predicate::RankLe`).
//
// This replaces a literal per-element rank pass, which re-scanned the whole row
// for every element and so cost O(len^2) unconditionally -- ~2.3e10 element
// visits per row at a 151936-token vocabulary, which is what made
// `mirostat-v2-sampling` take minutes per request. Instead: a block-cooperative
// 4-pass 8-bit MSB radix select on `m1_desc_key`, O(5*len) regardless of `k`.
//
// Equivalence: `greater(i)` equals the count of strictly smaller keys, which is
// monotone in the key, so `greater(i) < k` holds exactly when `key(i) <= K_k`
// for `K_k` the k-th smallest key counting multiplicity. Ties therefore all
// survive or all fall together, which is what the reference does (it can keep
// more than `k` elements when the boundary value repeats). NaN keys sort last
// so they never displace a real element, and the marking pass excludes them.
__device__ __forceinline__ void ptir_parallel_pivot_rank(
    const m1_u8* input,
    const m1_u8* threshold,
    m1_u8* output,
    const M1ValueDesc input_desc,
    const M1ValueDesc threshold_desc) {
  __shared__ m1_u32 rank_hist[256];
  __shared__ m1_u32 rank_prefix;
  __shared__ m1_u32 rank_target;
  const m1_u32 len = input_desc.last;
  const m1_u32 rows = input_desc.rows;
  if (len == 0u) return;
  for (m1_u32 row = 0u; row < rows; ++row) {
    const m1_u32 base = row * len;
    m1_u32 k;
    if (threshold_desc.dtype == 1u) {
      const int signed_k = m1_load_i(
          threshold, m1_pick(threshold_desc.len, row), threshold_desc.dtype);
      k = signed_k <= 0 ? 0u : (m1_u32)signed_k;
    } else {
      k = m1_load_u(
          threshold, m1_pick(threshold_desc.len, row), threshold_desc.dtype);
    }
    if (k > len) k = len;
    if (k == 0u) {
      for (m1_u32 i = threadIdx.x; i < len; i += blockDim.x)
        m1_store_b(output, base + i, false);
      __syncthreads();
      continue;
    }
    if (threadIdx.x == 0u) {
      rank_prefix = 0u;
      rank_target = k;
    }
    __syncthreads();
    for (int pass = 0; pass < 4; ++pass) {
      const int shift = 24 - 8 * pass;
      // Bits fixed by earlier passes; `pass == 0` is special-cased because
      // shifting a 32-bit value by 32 is undefined, not zero.
      const m1_u32 high_mask =
          (pass == 0) ? 0u : (0xFFFFFFFFu << (shift + 8));
      for (m1_u32 bucket = threadIdx.x; bucket < 256u; bucket += blockDim.x)
        rank_hist[bucket] = 0u;
      __syncthreads();
      const m1_u32 prefix = rank_prefix;
      for (m1_u32 i = threadIdx.x; i < len; i += blockDim.x) {
        const m1_u32 key =
            m1_desc_key(m1_load_f(input, base + i, input_desc.dtype));
        if ((key & high_mask) == (prefix & high_mask))
          atomicAdd(&rank_hist[(key >> shift) & 0xFFu], 1u);
      }
      __syncthreads();
      if (threadIdx.x == 0u) {
        const m1_u32 target = rank_target;
        m1_u32 run = 0u;
        m1_u32 chosen = 255u;
        for (m1_u32 bucket = 0u; bucket < 256u; ++bucket) {
          if (run + rank_hist[bucket] >= target) { chosen = bucket; break; }
          run += rank_hist[bucket];
        }
        rank_target = target - run;
        rank_prefix = prefix | (chosen << shift);
      }
      __syncthreads();
    }
    const m1_u32 cutoff_key = rank_prefix;
    for (m1_u32 i = threadIdx.x; i < len; i += blockDim.x) {
      const float value = m1_load_f(input, base + i, input_desc.dtype);
      m1_store_b(output, base + i,
                 !m1_isnan(value) && m1_desc_key(value) <= cutoff_key);
    }
    __syncthreads();
  }
}

__device__ __forceinline__ void ptir_parallel_pivot(
    const m1_u8* input,
    const m1_u8* threshold,
    m1_u8* output,
    const M1ValueDesc input_desc,
    const M1ValueDesc threshold_desc,
    const M1OpParams p) {
  if (p.pred_tag == 0u) {
    ptir_parallel_pivot_rank(
        input, threshold, output, input_desc, threshold_desc);
    return;
  }
  for (m1_u32 flat = threadIdx.x;
       flat < input_desc.len;
       flat += blockDim.x) {
    const m1_u32 row =
        input_desc.last == 0u ? 0u : flat / input_desc.last;
    const float value = m1_load_f(input, flat, input_desc.dtype);
    const float cutoff = m1_load_f(
        threshold,
        m1_pick(threshold_desc.len, row),
        threshold_desc.dtype);
    m1_store_b(output, flat, value >= cutoff);
  }
}

__device__ __forceinline__ void ptir_scatter_updates(
    m1_u32 tag,
    const m1_u8* indices,
    const m1_u8* updates,
    m1_u8* output,
    const M1ValueDesc base_desc,
    const M1ValueDesc index_desc,
    const M1ValueDesc update_desc) {
  m1_u32 rest = 1u;
  m1_u32 rows = 1u;
  if (base_desc.rank != 0u) {
    rows = base_desc.dims[0];
    rest = rows == 0u ? 1u : base_desc.len / rows;
  }
  const bool scalar =
      update_desc.len == 1u && index_desc.len * rest != 1u;
  for (m1_u32 k = 0; k < index_desc.len; ++k) {
    const long long row =
        m1_load_index(indices, k, index_desc.dtype);
    if (row < 0 || (m1_u64)row >= rows) continue;
    for (m1_u32 r = 0; r < rest; ++r) {
      const m1_u32 destination = (m1_u32)row * rest + r;
      const m1_u32 source = scalar ? 0u : k * rest + r;
      if (base_desc.dtype == 0u) {
        const float value =
            m1_load_f(updates, source, update_desc.dtype);
        m1_store_f(
            output,
            destination,
            tag == 0x62u
                ? m1_load_f(output, destination, 0u) + value
                : value);
      } else if (base_desc.dtype == 1u) {
        const int value =
            m1_load_i(updates, source, update_desc.dtype);
        m1_store_i(
            output,
            destination,
            tag == 0x62u
                ? m1_bits_i32(
                      (m1_u32)m1_load_i(output, destination, 1u) +
                      (m1_u32)value)
                : value);
      } else if (base_desc.dtype == 2u) {
        const m1_u32 value =
            m1_load_u(updates, source, update_desc.dtype);
        m1_store_u(
            output,
            destination,
            tag == 0x62u
                ? m1_load_u(output, destination, 2u) + value
                : value);
      } else {
        m1_store_b(
            output,
            destination,
            m1_load_b(updates, source, update_desc.dtype));
      }
    }
  }
}

__device__ __forceinline__ void ptir_parallel_reduce_f32(
    m1_u32 tag,
    const m1_u8* input,
    m1_u8* output,
    m1_u8* temporary,
    const M1ValueDesc input_desc) {
  float* work_a = reinterpret_cast<float*>(temporary);
  const m1_u32 first_chunks = (input_desc.last + 31u) / 32u;
  float* work_b = work_a + first_chunks;
  const float* values = reinterpret_cast<const float*>(input);
  float* result = reinterpret_cast<float*>(output);
  const m1_u32 lane = threadIdx.x & 31u;
  const m1_u32 warp = threadIdx.x >> 5u;
  const m1_u32 warps = blockDim.x >> 5u;
  const unsigned mask = 0xffffffffu;
  for (m1_u32 row = 0; row < input_desc.rows; ++row) {
    const m1_u32 base = row * input_desc.last;
    if (input_desc.last == 0u) {
      if (threadIdx.x == 0u)
        result[row] =
            tag == 0x30u
                ? 0.0f
                : (tag == 0x31u ? m1_neg_inf() : m1_pos_inf());
      __syncthreads();
      continue;
    }
    for (m1_u32 chunk = warp;
         chunk < first_chunks;
         chunk += warps) {
      const m1_u32 index = chunk * 32u + lane;
      const float identity =
          tag == 0x30u
              ? 0.0f
              : (tag == 0x31u ? m1_neg_inf() : m1_pos_inf());
      float value =
          index < input_desc.last ? values[base + index] : identity;
      for (m1_u32 offset = 16u; offset > 0u; offset >>= 1u) {
        const float other = __shfl_down_sync(mask, value, offset);
        if (lane < offset) {
          value =
              tag == 0x30u
                  ? value + other
                  : (tag == 0x31u
                         ? m1_canonical_max(value, other)
                         : m1_canonical_min(value, other));
        }
      }
      if (lane == 0u) work_a[chunk] = value;
    }
    __syncthreads();
    m1_u32 count = first_chunks;
    float* reduction_input = work_a;
    float* reduction_output = work_b;
    while (count > 1u) {
      const m1_u32 chunks = (count + 31u) / 32u;
      for (m1_u32 chunk = warp;
           chunk < chunks;
           chunk += warps) {
        const m1_u32 index = chunk * 32u + lane;
        const float identity =
            tag == 0x30u
                ? 0.0f
                : (tag == 0x31u ? m1_neg_inf() : m1_pos_inf());
        float value =
            index < count ? reduction_input[index] : identity;
        for (m1_u32 offset = 16u; offset > 0u; offset >>= 1u) {
          const float other = __shfl_down_sync(mask, value, offset);
          if (lane < offset) {
            value =
                tag == 0x30u
                    ? value + other
                    : (tag == 0x31u
                           ? m1_canonical_max(
                                 value, other)
                           : m1_canonical_min(
                                 value, other));
          }
        }
        if (lane == 0u) reduction_output[chunk] = value;
      }
      __syncthreads();
      float* swap = reduction_input;
      reduction_input = reduction_output;
      reduction_output = swap;
      count = chunks;
    }
    if (threadIdx.x == 0u) result[row] = reduction_input[0];
    __syncthreads();
  }
}

__device__ __forceinline__ void ptir_parallel_argmax(
    const m1_u8* input,
    m1_u8* output,
    m1_u8* temporary,
    const M1ValueDesc input_desc) {
  int* result = reinterpret_cast<int*>(output);
  const m1_u32 lane = threadIdx.x & 31u;
  const m1_u32 warp = threadIdx.x >> 5u;
  const m1_u32 warps = blockDim.x >> 5u;
  const unsigned mask = 0xffffffffu;
  const m1_u32 first_chunks = (input_desc.last + 31u) / 32u;
  if (input_desc.dtype == 0u) {
    M1ArgmaxCandidate* work_a =
        reinterpret_cast<M1ArgmaxCandidate*>(temporary);
    M1ArgmaxCandidate* work_b = work_a + first_chunks;
    const float* values = reinterpret_cast<const float*>(input);
    for (m1_u32 row = 0; row < input_desc.rows; ++row) {
      const m1_u32 base = row * input_desc.last;
      if (input_desc.last == 0u) {
        if (threadIdx.x == 0u) result[row] = 0;
        __syncthreads();
        continue;
      }
      for (m1_u32 chunk = warp;
           chunk < first_chunks;
           chunk += warps) {
        const m1_u32 index = chunk * 32u + lane;
        M1ArgmaxCandidate candidate =
            index < input_desc.last
                ? M1ArgmaxCandidate{
                      values[base + index],
                      index,
                      m1_isnan(values[base + index]) ? 0u : 1u,
                      0u}
                : M1ArgmaxCandidate{m1_neg_inf(), 0u, 0u, 0u};
        for (m1_u32 offset = 16u; offset > 0u; offset >>= 1u) {
          M1ArgmaxCandidate other{
              __shfl_down_sync(mask, candidate.value, offset),
              __shfl_down_sync(mask, candidate.index, offset),
              __shfl_down_sync(mask, candidate.have, offset),
              0u};
          if (lane < offset)
            candidate = m1_argmax_combine(candidate, other);
        }
        if (lane == 0u) work_a[chunk] = candidate;
      }
      __syncthreads();
      m1_u32 count = first_chunks;
      M1ArgmaxCandidate* reduction_input = work_a;
      M1ArgmaxCandidate* reduction_output = work_b;
      while (count > 1u) {
        const m1_u32 chunks = (count + 31u) / 32u;
        for (m1_u32 chunk = warp;
             chunk < chunks;
             chunk += warps) {
          const m1_u32 index = chunk * 32u + lane;
          M1ArgmaxCandidate candidate =
              index < count
                  ? reduction_input[index]
                  : M1ArgmaxCandidate{m1_neg_inf(), 0u, 0u, 0u};
          for (m1_u32 offset = 16u; offset > 0u; offset >>= 1u) {
            M1ArgmaxCandidate other{
                __shfl_down_sync(mask, candidate.value, offset),
                __shfl_down_sync(mask, candidate.index, offset),
                __shfl_down_sync(mask, candidate.have, offset),
                0u};
            if (lane < offset)
              candidate = m1_argmax_combine(candidate, other);
          }
          if (lane == 0u) reduction_output[chunk] = candidate;
        }
        __syncthreads();
        M1ArgmaxCandidate* swap = reduction_input;
        reduction_input = reduction_output;
        reduction_output = swap;
        count = chunks;
      }
      if (threadIdx.x == 0u)
        result[row] = (int)reduction_input[0].index;
      __syncthreads();
    }
    return;
  }
  M1IntArgmaxCandidate* work_a =
      reinterpret_cast<M1IntArgmaxCandidate*>(temporary);
  M1IntArgmaxCandidate* work_b = work_a + first_chunks;
  for (m1_u32 row = 0; row < input_desc.rows; ++row) {
    const m1_u32 base = row * input_desc.last;
    if (input_desc.last == 0u) {
      if (threadIdx.x == 0u) result[row] = 0;
      __syncthreads();
      continue;
    }
    for (m1_u32 chunk = warp;
         chunk < first_chunks;
         chunk += warps) {
      const m1_u32 index = chunk * 32u + lane;
      M1IntArgmaxCandidate candidate =
          index < input_desc.last
              ? M1IntArgmaxCandidate{
                    m1_load_index(
                        input, base + index, input_desc.dtype),
                    index,
                    1u}
              : M1IntArgmaxCandidate{0ll, 0u, 0u};
      for (m1_u32 offset = 16u; offset > 0u; offset >>= 1u) {
        M1IntArgmaxCandidate other{
            __shfl_down_sync(mask, candidate.value, offset),
            __shfl_down_sync(mask, candidate.index, offset),
            __shfl_down_sync(mask, candidate.have, offset)};
        if (lane < offset)
          candidate = m1_int_argmax_combine(candidate, other);
      }
      if (lane == 0u) work_a[chunk] = candidate;
    }
    __syncthreads();
    m1_u32 count = first_chunks;
    M1IntArgmaxCandidate* reduction_input = work_a;
    M1IntArgmaxCandidate* reduction_output = work_b;
    while (count > 1u) {
      const m1_u32 chunks = (count + 31u) / 32u;
      for (m1_u32 chunk = warp;
           chunk < chunks;
           chunk += warps) {
        const m1_u32 index = chunk * 32u + lane;
        M1IntArgmaxCandidate candidate =
            index < count
                ? reduction_input[index]
                : M1IntArgmaxCandidate{0ll, 0u, 0u};
        for (m1_u32 offset = 16u; offset > 0u; offset >>= 1u) {
          M1IntArgmaxCandidate other{
              __shfl_down_sync(mask, candidate.value, offset),
              __shfl_down_sync(mask, candidate.index, offset),
              __shfl_down_sync(mask, candidate.have, offset)};
          if (lane < offset)
            candidate = m1_int_argmax_combine(candidate, other);
        }
        if (lane == 0u) reduction_output[chunk] = candidate;
      }
      __syncthreads();
      M1IntArgmaxCandidate* swap = reduction_input;
      reduction_input = reduction_output;
      reduction_output = swap;
      count = chunks;
    }
    if (threadIdx.x == 0u)
      result[row] = (int)reduction_input[0].index;
    __syncthreads();
  }
}

__device__ __forceinline__ M1ArgmaxCandidate m1_argmax_warp_reduce(
    M1ArgmaxCandidate candidate, m1_u32 lane) {
  const unsigned mask = 0xffffffffu;
  for (m1_u32 offset = 16u; offset > 0u; offset >>= 1u) {
    M1ArgmaxCandidate other{
        __shfl_down_sync(mask, candidate.value, offset),
        __shfl_down_sync(mask, candidate.index, offset),
        __shfl_down_sync(mask, candidate.have, offset),
        0u};
    if (lane < offset)
      candidate = m1_argmax_combine(candidate, other);
  }
  return candidate;
}

__device__ __forceinline__ M1IntArgmaxCandidate m1_int_argmax_warp_reduce(
    M1IntArgmaxCandidate candidate, m1_u32 lane) {
  const unsigned mask = 0xffffffffu;
  for (m1_u32 offset = 16u; offset > 0u; offset >>= 1u) {
    M1IntArgmaxCandidate other{
        __shfl_down_sync(mask, candidate.value, offset),
        __shfl_down_sync(mask, candidate.index, offset),
        __shfl_down_sync(mask, candidate.have, offset)};
    if (lane < offset)
      candidate = m1_int_argmax_combine(candidate, other);
  }
  return candidate;
}

__device__ __forceinline__ void ptir_fast_argmax(
    const m1_u8* input,
    m1_u8* output,
    const M1ValueDesc input_desc) {
  __shared__ M1ArgmaxCandidate float_candidates[32];
  __shared__ M1IntArgmaxCandidate int_candidates[32];
  int* result = reinterpret_cast<int*>(output);
  const m1_u32 lane = threadIdx.x & 31u;
  const m1_u32 warp = threadIdx.x >> 5u;
  const m1_u32 warps = blockDim.x >> 5u;
  for (m1_u32 row = 0; row < input_desc.rows; ++row) {
    const m1_u32 base = row * input_desc.last;
    if (input_desc.dtype == 0u) {
      M1ArgmaxCandidate candidate{
          m1_neg_inf(), 0u, 0u, 0u};
      for (m1_u32 index = threadIdx.x;
           index < input_desc.last;
           index += blockDim.x) {
        const float value =
            reinterpret_cast<const float*>(input)[base + index];
        const M1ArgmaxCandidate next{
            value, index, m1_isnan(value) ? 0u : 1u, 0u};
        candidate = m1_argmax_combine(candidate, next);
      }
      candidate = m1_argmax_warp_reduce(candidate, lane);
      if (lane == 0u) float_candidates[warp] = candidate;
      __syncthreads();
      if (warp == 0u) {
        candidate =
            lane < warps
                ? float_candidates[lane]
                : M1ArgmaxCandidate{m1_neg_inf(), 0u, 0u, 0u};
        candidate = m1_argmax_warp_reduce(candidate, lane);
        if (lane == 0u) result[row] = (int)candidate.index;
      }
      __syncthreads();
      continue;
    }
    M1IntArgmaxCandidate candidate{0ll, 0u, 0u};
    for (m1_u32 index = threadIdx.x;
         index < input_desc.last;
         index += blockDim.x) {
      const M1IntArgmaxCandidate next{
          m1_load_index(input, base + index, input_desc.dtype),
          index,
          1u};
      candidate = m1_int_argmax_combine(candidate, next);
    }
    candidate = m1_int_argmax_warp_reduce(candidate, lane);
    if (lane == 0u) int_candidates[warp] = candidate;
    __syncthreads();
    if (warp == 0u) {
      candidate =
          lane < warps
              ? int_candidates[lane]
              : M1IntArgmaxCandidate{0ll, 0u, 0u};
      candidate = m1_int_argmax_warp_reduce(candidate, lane);
      if (lane == 0u) result[row] = (int)candidate.index;
    }
    __syncthreads();
  }
}

__device__ __forceinline__ void ptir_fast_argmax_intrinsic(
    const m1_u8* input,
    m1_u8* output,
    const M1ValueDesc input_desc,
    m1_u32 mode,
    m1_u32 stride,
    m1_u32 row_offset) {
  __shared__ M1ArgmaxCandidate candidates[32];
  int* result = reinterpret_cast<int*>(output);
  const m1_u32 lane = threadIdx.x & 31u;
  const m1_u32 warp = threadIdx.x >> 5u;
  const m1_u32 warps = blockDim.x >> 5u;
  // Mode 3: the reduction already happened. The driver interleaves the LM head
  // GEMM with the argmax so the logits never reach HBM, and hands us the
  // finished token ids instead of a vocab to scan. Like mode 2 this is a table
  // of row pointers, because a lane's sampled rows are not contiguous -- each
  // entry addresses one i32 rather than one vocabulary row. The epilogue still
  // runs and still performs every side effect it declared; only where this one
  // value comes from changed. `mode` is block-uniform, so the early return is
  // too.
  //
  // Safe only because the driver proves every `logits` reader in the stage is
  // one of these reductions; a stage that also reads the raw values would find
  // token ids behind the same intrinsic slot.
  if (mode == 3u) {
    const m1_u64* rows = reinterpret_cast<const m1_u64*>(input);
    for (m1_u32 row = threadIdx.x; row < input_desc.rows; row += blockDim.x) {
      result[row] = *reinterpret_cast<const int*>(
          rows[(m1_u64)row_offset + row]);
    }
    __syncthreads();
    return;
  }
  for (m1_u32 row = 0; row < input_desc.rows; ++row) {
    const m1_u8* row_base = m1_intrinsic_row_base(
        input, (m1_u64)row_offset + row, stride, mode);
    const m1_u32 last = input_desc.last;
    M1ArgmaxCandidate candidate{
        m1_neg_inf(), 0u, 0u, 0u};
    // `m1_argmax_combine` takes the max and breaks ties toward the lower
    // index, so it is commutative and associative: widening each thread's scan
    // to a 16-byte load changes how many instructions the scan costs, not what
    // it answers.
    const m1_u32 vectors =
        (mode != 0u &&
         ((m1_u64)row_base & 15ull) == 0ull)
            ? (last >> 3)
            : 0u;
    for (m1_u32 v = threadIdx.x; v < vectors; v += blockDim.x) {
      const M1U32x4 packed = reinterpret_cast<const M1U32x4*>(row_base)[v];
      const m1_u32 words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
      for (m1_u32 k = 0; k < 8u; ++k) {
        const float value =
            __uint_as_float((words[k >> 1] >> ((k & 1u) * 16u)) << 16);
        const M1ArgmaxCandidate next{
            value, v * 8u + k, m1_isnan(value) ? 0u : 1u, 0u};
        candidate = m1_argmax_combine(candidate, next);
      }
    }
    for (m1_u32 index = (vectors << 3) + threadIdx.x;
         index < last;
         index += blockDim.x) {
      const float value = m1_intrinsic_column_load(row_base, index, mode);
      const M1ArgmaxCandidate next{
          value, index, m1_isnan(value) ? 0u : 1u, 0u};
      candidate = m1_argmax_combine(candidate, next);
    }
    candidate = m1_argmax_warp_reduce(candidate, lane);
    if (lane == 0u) candidates[warp] = candidate;
    __syncthreads();
    if (warp == 0u) {
      candidate =
          lane < warps
              ? candidates[lane]
              : M1ArgmaxCandidate{m1_neg_inf(), 0u, 0u, 0u};
      candidate = m1_argmax_warp_reduce(candidate, lane);
      if (lane == 0u) result[row] = (int)candidate.index;
    }
    __syncthreads();
  }
}

extern "C" __global__ void 