

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
  m1_u64 attn_score_base;
  m1_u32 attn_score_row_stride;
  m1_u32 reserved1;
  m1_u64 mtp_drafts_base;
  m1_u32 mtp_drafts_depth;
  m1_u32 reserved2;
};

struct PtirLaneChannelSlot {
  m1_u64 committed_cell;
  m1_u64 pending_cell;
  m1_u64 expected_head;
  m1_u64 expected_tail;
};

static_assert(sizeof(PtirLaneTableHeader) == 16, "lane header ABI");
static_assert(sizeof(PtirLaneRecord) == 120, "lane record ABI");
static_assert(sizeof(PtirLaneChannelSlot) == 32, "lane channel ABI");

__device__ __forceinline__ void ptir_parallel_copy(
    const m1_u8* input,
    m1_u8* output,
    m1_u32 len,
    m1_u32 dtype) {
  const m1_u32 bytes = dtype == 3u ? len : len * 4u;
  // Sixteen bytes a thread where both ends allow it (a vocabulary-wide
  // value moved a byte a thread is a block's whole pass), the tail bytewise.
  m1_u32 head = 0u;
  if (((reinterpret_cast<m1_u64>(input) | reinterpret_cast<m1_u64>(output)) & 15ull) == 0ull) {
    const m1_u32 quads = bytes >> 4;
    const uint4* in4 = reinterpret_cast<const uint4*>(input);
    uint4* out4 = reinterpret_cast<uint4*>(output);
    for (m1_u32 q = threadIdx.x; q < quads; q += blockDim.x) out4[q] = in4[q];
    head = quads << 4;
  }
  for (m1_u32 index = head + threadIdx.x; index < bytes; index += blockDim.x)
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
  if (output_desc.len <= width) {
    // One row: its columns, no division an element.
    for (m1_u32 index = threadIdx.x; index < output_desc.len; index += blockDim.x)
      m1_store_f(output, index,
                 m1_intrinsic_row_load(input, first_row, index, stride, p.intrinsic_dtype));
    return;
  }
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

// A keyed draw's seed from its state, a whole value `[key, counter]`: a
// row-parallel region reads it through the state's whole descriptor and
// storage, never through a row's narrowed view of it.
__device__ __forceinline__ m1_u64 ptir_rng_state_seed(
    const m1_u8* state,
    const M1ValueDesc state_desc) {
  const m1_u32 key = m1_load_u(state, 0u, state_desc.dtype);
  const m1_u32 counter =
      state_desc.len > 1u ? m1_load_u(state, 1u, state_desc.dtype) : 0u;
  return ptir_rng_keyed_seed(key, counter);
}

__device__ __forceinline__ float ptir_rng_draw(
    m1_u64 seed,
    m1_u32 index,
    m1_u32 kind) {
  if (kind == 2u) return ptir_rng_hash_normal(seed, index);
  const float uniform = ptir_rng_hash_uniform(seed, index);
  return kind == 0u ? uniform : -logf(-logf(uniform));
}

__device__ __forceinline__ void ptir_parallel_rng_keyed(
    const m1_u8* state,
    const M1ValueDesc state_desc,
    m1_u8* output,
    const M1ValueDesc out,
    const M1OpParams p) {
  const m1_u64 seed = ptir_rng_state_seed(state, state_desc);
  for (m1_u32 i = threadIdx.x; i < out.len; i += blockDim.x)
    m1_store_f(output, i, ptir_rng_draw(seed, i + p.imm3, p.kind));
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
    if (tag >= 0x08u && tag <= 0x0bu) {
      const float value = m1_load_f(a0, xindex, d0.dtype);
      m1_store_f(
          o0,
          i,
          tag == 0x08u
              ? sinf(value)
              : (tag == 0x09u
                     ? cosf(value)
                     : (tag == 0x0au ? sqrtf(value) : 1.0f / sqrtf(value))));
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
      const m1_u64 seed =
          tag == 0x70u
              ? ptir_rng_seed_eff_stream((m1_u32)p.rng_seed, p.imm)
              : ptir_rng_state_seed(a0, d0);
      m1_store_f(o0, i, ptir_rng_draw(seed, i + p.imm3, p.kind));
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
  if (input_desc.len == 1u) {
    // A scalar's broadcast: its one element everywhere, no index walk.
    if (output_desc.dtype == 3u) {
      const m1_u8 v = input[0] != 0u ? 1u : 0u;
      for (m1_u32 i = threadIdx.x; i < output_desc.len; i += blockDim.x) output[i] = v;
    } else {
      const m1_u32 v = reinterpret_cast<const m1_u32*>(input)[0];
      m1_u32* out = reinterpret_cast<m1_u32*>(output);
      m1_u32 head = 0u;
      if ((reinterpret_cast<m1_u64>(output) & 15ull) == 0ull) {
        const uint4 v4 = make_uint4(v, v, v, v);
        const m1_u32 quads = output_desc.len >> 2;
        for (m1_u32 q = threadIdx.x; q < quads; q += blockDim.x) reinterpret_cast<uint4*>(out)[q] = v4;
        head = quads << 2;
      }
      for (m1_u32 i = head + threadIdx.x; i < output_desc.len; i += blockDim.x) out[i] = v;
    }
    return;
  }
  // A row-parallel region narrows the output to one row and leaves an input
  // every row shares whole: its leading dims, all 1, are the ones the output
  // dropped, so they are skipped.
  const m1_u32 skip = input_desc.rank > output_desc.rank
      ? input_desc.rank - output_desc.rank
      : 0u;
  for (m1_u32 linear = threadIdx.x;
       linear < output_desc.len;
       linear += blockDim.x) {
    m1_u32 rem = linear;
    m1_u32 source_index = 0;
    m1_u32 source_stride[4] = {1, 1, 1, 1};
    for (int dim = (int)output_desc.rank - 2; dim >= 0; --dim) {
      source_stride[dim] =
          source_stride[dim + 1] *
          ((m1_u32)(dim + 1) + skip < input_desc.rank
               ? input_desc.dims[dim + 1 + skip]
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
          dim + skip < input_desc.rank ? input_desc.dims[dim + skip] : 1u;
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

    const m1_u32 rows = input_desc.rank <= 1u ? 1u : input_desc.dims[0];
    const m1_u32 columns = input_desc.rank <= 1u ? input_desc.len : input_desc.dims[1];
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

// The top-p cut, `PivotThreshold(CummassLe)`: a row is walked in descending
// order (ties by index, NaNs last), and an element is kept while the mass
// summed before it, sequentially in f32, is below the cutoff. The device
// lands on the walk's own boundary, rounding for rounding, without the walk.
//
// The head of the row — its best M1_CUT_CHUNK elements, picked by a radix
// select on the descending key — is sorted in LDS and walked by one thread,
// as the reference walks it. Past the head the running sum S is far larger
// than any element it adds, and every element's rounded contribution is a
// whole number of S's ulp u: while S stays in one binade, fl(S + y) =
// S + d(y) u, d(y) being y / u rounded to nearest — whatever S is, except
// for an element exactly half a unit off a whole, which rounds to even and
// so depends on S's last bit. So within a binade the sum is an integer sum,
// order-free but for those half-unit ties, and a radix select over the key
// with each element weighing d(y) finds the element at which S reaches the
// cutoff or leaves the binade; that element's own add is then done in f32,
// and the next binade starts there. The ties are gathered by value: an
// equal-valued run of them adds a closed form, and the sum between two tie
// values is again order-free, so a phase with ties walks the gaps between
// them. Rows outside that contract — a negative value, an infinity, more
// tie values than fit — are walked chunk by chunk from the head on. As in
// the reference, the first NaN (by index) is kept only when every other
// element is and the row's whole mass is below the cutoff.

#define M1_CUT_CHUNK 2048u
#define M1_CUT_TIE_SLOTS 512u
#define M1_CUT_TIES 256u

#define M1_CUT_RUN 0u
#define M1_CUT_STOPPED 1u
#define M1_CUT_ALL 2u
#define M1_CUT_BAIL 3u

#define M1_CUT_COUNT (-100000)

struct alignas(16) M1CutShared {
  m1_u64 chunk[M1_CUT_CHUNK];
  float before[M1_CUT_CHUNK];
  m1_u64 weight[256];
  m1_u32 least[256];
  m1_u32 tie_slot[M1_CUT_TIE_SLOTS];
  m1_u32 tie_key[M1_CUT_TIES];
  m1_u32 tie_count[M1_CUT_TIES];
  m1_u64 gap[M1_CUT_TIES + 1u];
  m1_u64 cursor;
  m1_u64 pivot;
  m1_u64 select_need;
  m1_u64 select_before;
  m1_u64 segment_low;
  m1_u64 segment_high;
  m1_u64 segment_units;
  m1_u64 crossing_units;
  m1_u32 select_prefix;
  m1_u32 select_least;
  m1_u32 select_short;
  m1_u32 count;
  m1_u32 ties;
  m1_u32 overflow;
  m1_u32 first_nan;
  m1_u32 negative;
  m1_u32 key_least;
  m1_u32 key_most;
  m1_u32 key_skip;
  m1_u32 wide;
  m1_u32 stage;
  m1_u32 nan_kept;
  m1_u32 crossing;
  m1_u32 crossing_key;
  m1_u32 crossing_rank;
  float mass;
};

// Where a function's cut state lives: the launch's dynamic shared memory
// where the dialect says so (CUDA's), else a static array.
#if defined(M1_CUT_DYNAMIC)
static_assert(sizeof(M1CutShared) <= M1_CUT_STATE_BYTES, "the cut's state fits what the launcher gives it");
#define M1_CUT_STATE(name)                                              \
  extern __shared__ __align__(16) unsigned char m1_cut_state[];         \
  M1CutShared& name = *reinterpret_cast<M1CutShared*>(m1_cut_state)
#else
#define M1_CUT_STATE(name) __shared__ M1CutShared name
#endif

__device__ __forceinline__ float m1_cut_value(m1_u32 key) {
  const m1_u32 ascending = ~key;
  return m1_bits_f32((ascending & 0x80000000u) ? (ascending & 0x7FFFFFFFu)
                                               : ~ascending);
}

__device__ __forceinline__ m1_u64 m1_cut_composite(m1_u32 key, m1_u32 index) {
  return ((m1_u64)key << 32) | (m1_u64)index;
}

// Every element of the row in [lo, hi), eight loads in flight a thread
// before any is looked at: a block walks its stretch of the row, so a pass is
// bound by how many loads it keeps outstanding.
template <class Body>
__device__ __forceinline__ void m1_cut_scan_range(
    const m1_u8* input, m1_u32 base, m1_u32 lo, m1_u32 hi, m1_u32 dtype, Body body) {
  const m1_u32 stride = blockDim.x;
  const m1_u32 len = hi > lo ? hi - lo : 0u;
  const float* row = reinterpret_cast<const float*>(input) + base + lo;
  if (dtype == 0u && (reinterpret_cast<m1_u64>(row) & 15ull) == 0ull) {
    // An f32 row on a 16-byte boundary: two float4 loads in flight a thread.
    const float4* quads = reinterpret_cast<const float4*>(row);
    const m1_u32 whole = len >> 2;
    for (m1_u32 at = threadIdx.x; at < whole; at += 2u * stride) {
      const bool second = at + stride < whole;
      const float4 a = quads[at];
      const float4 b = second ? quads[at + stride] : a;
      const m1_u32 i = lo + (at << 2);
      body(i, a.x);
      body(i + 1u, a.y);
      body(i + 2u, a.z);
      body(i + 3u, a.w);
      if (second) {
        const m1_u32 j = lo + ((at + stride) << 2);
        body(j, b.x);
        body(j + 1u, b.y);
        body(j + 2u, b.z);
        body(j + 3u, b.w);
      }
    }
    for (m1_u32 i = (whole << 2) + threadIdx.x; i < len; i += stride)
      body(lo + i, row[i]);
    return;
  }
  for (m1_u32 at = threadIdx.x; at < len; at += 8u * stride) {
    float batch[8];
#pragma unroll
    for (m1_u32 u = 0u; u < 8u; ++u) {
      const m1_u32 i = at + u * stride;
      batch[u] = i < len ? m1_load_f(input, base + lo + i, dtype) : 0.0f;
    }
#pragma unroll
    for (m1_u32 u = 0u; u < 8u; ++u) {
      const m1_u32 i = at + u * stride;
      if (i < len) body(lo + i, batch[u]);
    }
  }
}

template <class Body>
__device__ __forceinline__ void m1_cut_scan(
    const m1_u8* input, m1_u32 base, m1_u32 len, m1_u32 dtype, Body body) {
  m1_cut_scan_range(input, base, 0u, len, dtype, body);
}

// The blocks a row's cut runs on. A cut walks the row in passes; each pass
// leaves sums, least indices, gathered elements or tie values in the block's
// LDS, and every decision after a pass is a function of those alone. One
// block (`M1Solo`) walks the whole row and its hooks do nothing; a group of
// blocks (`M1Lane`, sample_lane.cuh) each walks its stretch [lo, hi) and its
// hooks merge every block's pass results into every block's LDS, in block
// order, so the blocks decide alike.
struct M1Solo {
  m1_u32 lo;
  m1_u32 hi;
  __device__ __forceinline__ void row(m1_u32 len) {
    lo = 0u;
    hi = len;
  }
  __device__ __forceinline__ void entry(M1CutShared&) {}
  __device__ __forceinline__ void select(M1CutShared&, bool, bool) {}
  __device__ __forceinline__ void weights(M1CutShared&) {}
  __device__ __forceinline__ void chunk(M1CutShared&) {}
  __device__ __forceinline__ void gaps(M1CutShared&, m1_u32) {}
  __device__ __forceinline__ void sum_u32(m1_u32*, m1_u32) {}
  __device__ __forceinline__ void fast(const m1_u8*, m1_u32, m1_u32, m1_u32, float, M1CutShared&) {}
};

// What a nonnegative `y` (no larger than the running sum) adds to a sum that
// is a whole number of units u = 2^eu: `whole` units, and where the rest sits
// against half a unit — 0 below, 1 exactly half (a tie), 2 above.
__device__ __forceinline__ m1_u32 m1_cut_units(float y, int eu, m1_u32* whole) {
  const m1_u32 bits = __float_as_uint(y) & 0x7FFFFFFFu;
  if (bits == 0u) {
    *whole = 0u;
    return 0u;
  }
  const m1_u32 biased = bits >> 23;
  const m1_u32 mantissa =
      biased == 0u ? (bits & 0x7FFFFFu) : ((bits & 0x7FFFFFu) | 0x800000u);
  const int ey = biased == 0u ? -149 : (int)biased - 150;
  const int shift = eu - ey;
  if (shift <= 0) {
    *whole = mantissa << (-shift);
    return 0u;
  }
  if (shift > 24) {
    *whole = 0u;
    return 0u;
  }
  const m1_u32 rest = mantissa & ((1u << shift) - 1u);
  const m1_u32 half = 1u << (shift - 1);
  *whole = mantissa >> shift;
  return rest < half ? 0u : (rest == half ? 1u : 2u);
}

__device__ __forceinline__ m1_u64 m1_cut_weight(float y, int eu) {
  if (eu == M1_CUT_COUNT) return 1ull;
  m1_u32 whole;
  const m1_u32 rest = m1_cut_units(y, eu, &whole);
  return (m1_u64)whole + (rest == 2u ? 1ull : 0ull);
}

// Among the row's live elements (not NaN, low < composite < high) in
// descending order, each weighing 1 (eu == M1_CUT_COUNT) or its units at
// 2^eu, the key whose run holds the element at which the running weight
// first reaches `need`: `select_prefix` is that key, `select_before` the
// weight before its run, `select_need` the weight still wanted inside it,
// `select_least` its run's least live index; `select_short` if the live
// weight never reaches `need`.
// Gathers a tie value (once, however many elements hold it) into the hash
// set; past M1_CUT_TIES values the phase gives up on its gaps.
__device__ __forceinline__ void m1_cut_tie(m1_u32 key, M1CutShared& cut) {
  if (*(volatile m1_u32*)&cut.overflow != 0u) return;
  m1_u32 h = (key * 2654435761u) >> 23;
  m1_u32 probe = 0u;
  for (; probe < M1_CUT_TIE_SLOTS; ++probe) {
    const m1_u32 previous = atomicCAS(&cut.tie_slot[h], 0u, key);
    if (previous == 0u) {
      if (atomicAdd(&cut.ties, 1u) >= M1_CUT_TIES) cut.overflow = 1u;
      return;
    }
    if (previous == key) return;
    h = (h + 1u) & (M1_CUT_TIE_SLOTS - 1u);
  }
  cut.overflow = 1u;
}

// With `gather_ties`, the first pass also gathers the elements exactly half
// a unit off a whole (weighing none of them) and the select stops there if
// it found any: their weight is not order-free (see m1_cut_phase).
// The first of 256 buckets at which the running weight reaches `want` (the
// weight before it plus its own): its index into `found` (256 if the whole
// is short of it) and the weight before it into `before`, for every thread.
// Thirty-two threads sum eight buckets each, the one whose stretch holds the
// crossing walks its eight.
__device__ __forceinline__ void m1_bucket_find(
    const m1_u64* weight, m1_u64 want, m1_u32& found, m1_u64& before) {
  __shared__ m1_u64 sums[32];
  __shared__ m1_u32 found_s;
  __shared__ m1_u64 before_s;
  const m1_u32 t = threadIdx.x;
  if (t < 32u) {
    m1_u64 sum = 0ull;
#pragma unroll
    for (m1_u32 b = 0u; b < 8u; ++b) sum += weight[t * 8u + b];
    sums[t] = sum;
  }
  __syncthreads();
  if (t < 32u) {
    m1_u64 prior = 0ull;
    for (m1_u32 j = 0u; j < t; ++j) prior += sums[j];
    const m1_u64 mine = sums[t];
    if ((t == 0u || prior < want) && prior + mine >= want) {
      m1_u64 run = prior;
      for (m1_u32 b = t * 8u; b < t * 8u + 8u; ++b) {
        if (run + weight[b] >= want) {
          found_s = b;
          before_s = run;
          break;
        }
        run += weight[b];
      }
    } else if (t == 31u && prior + mine < want) {
      found_s = 256u;
      before_s = prior + mine;
    }
  }
  __syncthreads();
  found = found_s;
  before = before_s;
}

template <class G>
__device__ void m1_cut_select_key(
    const m1_u8* input, m1_u32 base, m1_u32 len, m1_u32 dtype,
    m1_u64 low, m1_u64 high, m1_u64 need, int eu, M1CutShared& cut, G& g,
    bool gather_ties = false) {
  __syncthreads();
  // Leading key bytes every element in the window shares are not passes:
  // the window is the row's key range cut to (low, high).
  const m1_u32 low_key = (m1_u32)(low >> 32);
  const m1_u32 high_key =
      high == ~0ull ? 0xFFFFFFFFu : (m1_u32)((high - 1ull) >> 32);
  const m1_u32 floor_key = low_key > cut.key_least ? low_key : cut.key_least;
  const m1_u32 ceiling_key = high_key < cut.key_most ? high_key : cut.key_most;
  const m1_u32 differ =
      floor_key <= ceiling_key ? (floor_key ^ ceiling_key) : 0xFFFFFFFFu;
  int skip = 0;
  while (skip < 3 && (differ >> (24 - 8 * skip)) == 0u) ++skip;
  if (threadIdx.x == 0u) {
    cut.select_prefix =
        skip == 0 ? 0u : (floor_key & (0xFFFFFFFFu << (32 - 8 * skip)));
    cut.select_need = need;
    cut.select_before = 0ull;
    cut.select_short = 0u;
  }
  __syncthreads();
  for (int pass = skip; pass < 4; ++pass) {
    const int shift = 24 - 8 * pass;
    const m1_u32 high_mask = pass == 0 ? 0u : (0xFFFFFFFFu << (shift + 8));
    for (m1_u32 bucket = threadIdx.x; bucket < 256u; bucket += blockDim.x) {
      cut.weight[bucket] = 0ull;
      cut.least[bucket] = 0xFFFFFFFFu;
    }
    __syncthreads();
    const m1_u32 prefix = cut.select_prefix & high_mask;
    m1_u32 held_bucket = 256u;
    m1_u64 held = 0ull;
    m1_cut_scan_range(input, base, g.lo, g.hi, dtype, [&](m1_u32 i, float y) {
      if (m1_isnan(y)) return;
      const m1_u32 key = m1_desc_key(y);
      if ((key & high_mask) != prefix) return;
      const m1_u64 composite = m1_cut_composite(key, i);
      if (composite <= low || composite >= high) return;
      const m1_u32 bucket = (key >> shift) & 0xFFu;
      if (gather_ties && pass == skip) {
        m1_u32 whole;
        if (m1_cut_units(y, eu, &whole) == 1u) {
          m1_cut_tie(key, cut);
          return;
        }
      }
      if (pass == 3) atomicMin(&cut.least[bucket], i);
      if (bucket != held_bucket) {
        if (held != 0ull) atomicAdd(&cut.weight[held_bucket], held);
        held_bucket = bucket;
        held = 0ull;
      }
      held += m1_cut_weight(y, eu);
    });
    if (held != 0ull) atomicAdd(&cut.weight[held_bucket], held);
    __syncthreads();
    g.select(cut, pass == 3, gather_ties && pass == skip);
    const m1_u64 want = cut.select_need;
    m1_u32 chosen;
    m1_u64 run;
    m1_bucket_find(cut.weight, want, chosen, run);
    if (threadIdx.x == 0u) {
      if (chosen == 256u) {
        cut.select_short = 1u;
      } else {
        cut.select_need = want - run;
        cut.select_before += run;
        cut.select_prefix = prefix | (chosen << shift);
        cut.select_least = cut.least[chosen];
      }
    }
    __syncthreads();
    if (gather_ties && pass == skip && (cut.ties | cut.overflow) != 0u) return;
    if (cut.select_short != 0u) return;
  }
}

// The `rank`-th (from 0) least index among the live elements of key `key`
// past `low`.
template <class G>
__device__ m1_u32 m1_cut_select_index(
    const m1_u8* input, m1_u32 base, m1_u32 len, m1_u32 dtype,
    m1_u32 key, m1_u64 low, m1_u64 rank, M1CutShared& cut, G& g) {
  int bits = 0;
  while (bits < 32 && ((len - 1u) >> bits) != 0u) ++bits;
  const int passes = (bits + 7) / 8;
  __syncthreads();
  if (threadIdx.x == 0u) {
    cut.select_prefix = 0u;
    cut.select_need = rank + 1ull;
  }
  __syncthreads();
  for (int pass = 0; pass < passes; ++pass) {
    const int shift = 8 * (passes - 1 - pass);
    const m1_u32 high_mask =
        pass == 0 ? 0u : (0xFFFFFFFFu << (shift + 8));
    for (m1_u32 bucket = threadIdx.x; bucket < 256u; bucket += blockDim.x)
      cut.weight[bucket] = 0ull;
    __syncthreads();
    const m1_u32 prefix = cut.select_prefix & high_mask;
    for (m1_u32 i = g.lo + threadIdx.x; i < g.hi; i += blockDim.x) {
      if ((i & high_mask) != prefix) continue;
      const float y = m1_load_f(input, base + i, dtype);
      if (m1_isnan(y) || m1_desc_key(y) != key) continue;
      if (m1_cut_composite(key, i) <= low) continue;
      atomicAdd(&cut.weight[(i >> shift) & 0xFFu], 1ull);
    }
    __syncthreads();
    g.weights(cut);
    const m1_u64 want = cut.select_need;
    m1_u32 chosen;
    m1_u64 run;
    m1_bucket_find(cut.weight, want, chosen, run);
    if (chosen == 256u) chosen = 255u;
    if (threadIdx.x == 0u) {
      cut.select_need = want - run;
      cut.select_prefix = prefix | (chosen << shift);
    }
    __syncthreads();
  }
  const m1_u32 index = cut.select_prefix;
  __syncthreads();
  return index;
}

__device__ void m1_cut_sort(m1_u64* items, m1_u32 count) {
  m1_u32 n = 1u;
  while (n < count) n <<= 1;
  for (m1_u32 i = count + threadIdx.x; i < n; i += blockDim.x)
    items[i] = ~0ull;
  __syncthreads();
  for (m1_u32 k = 2u; k <= n; k <<= 1) {
    for (m1_u32 j = k >> 1; j > 0u; j >>= 1) {
      for (m1_u32 t = threadIdx.x; t < (n >> 1); t += blockDim.x) {
        const m1_u32 i = ((t & ~(j - 1u)) << 1) | (t & (j - 1u));
        const m1_u32 l = i + j;
        const bool up = (i & k) == 0u;
        const m1_u64 x = items[i];
        const m1_u64 y = items[l];
        if ((x > y) == up) {
          items[i] = y;
          items[l] = x;
        }
      }
      __syncthreads();
    }
  }
}

// The next M1_CUT_CHUNK live elements past the cursor, sorted and walked in
// f32 as the reference walks them. A monotone row (no negative value) stops
// at the first element whose exclusive mass reaches the cutoff; any other
// row is walked to its end, each element's verdict stored as it is reached.
template <class G>
__device__ void m1_cut_chunk(
    const m1_u8* input, m1_u32 base, m1_u32 len, m1_u32 dtype,
    float cutoff, bool monotone, m1_u8* output, M1CutShared& cut, G& g) {
  const m1_u64 low = cut.cursor;
  m1_cut_select_key(
      input, base, len, dtype, low, ~0ull, M1_CUT_CHUNK + 1u, M1_CUT_COUNT, cut, g);
  m1_u64 high = ~0ull;
  if (cut.select_short == 0u) {
    const m1_u32 key = cut.select_prefix;
    if (cut.select_before >= M1_CUT_CHUNK / 2u) {
      high = m1_cut_composite(key, 0u);
    } else {
      const m1_u64 rank = cut.select_need - 1ull;
      const m1_u32 least = cut.select_least;
      const m1_u32 index =
          rank == 0ull
              ? least
              : m1_cut_select_index(input, base, len, dtype, key, low, rank, cut, g);
      high = m1_cut_composite(key, index);
    }
  }
  if (threadIdx.x == 0u) cut.count = 0u;
  __syncthreads();
  m1_cut_scan_range(input, base, g.lo, g.hi, dtype, [&](m1_u32 i, float y) {
    if (m1_isnan(y)) return;
    const m1_u64 composite = m1_cut_composite(m1_desc_key(y), i);
    if (composite <= low || composite >= high) return;
    const m1_u32 slot = atomicAdd(&cut.count, 1u);
    if (slot < M1_CUT_CHUNK) cut.chunk[slot] = composite;
  });
  __syncthreads();
  g.chunk(cut);
  const m1_u32 count = cut.count < M1_CUT_CHUNK ? cut.count : M1_CUT_CHUNK;
  if (count == 0u) {
    if (threadIdx.x == 0u) {
      cut.stage = M1_CUT_ALL;
      cut.nan_kept = cut.mass < cutoff ? 1u : 0u;
    }
    __syncthreads();
    return;
  }
  m1_cut_sort(cut.chunk, count);
  for (m1_u32 n = threadIdx.x; n < count; n += blockDim.x)
    cut.before[n] = m1_cut_value((m1_u32)(cut.chunk[n] >> 32));
  __syncthreads();
  // The walk itself is the one sequential piece: one thread, an f32 add an
  // element. A monotone row only looks for the stop, eight values read ahead
  // of the adds; any other row leaves each slot holding the mass before its
  // element.
  if (threadIdx.x == 0u) {
    float mass = cut.mass;
    m1_u32 walked = count;
    m1_u32 n = 0u;
    if (monotone) {
      const float4* quads = reinterpret_cast<const float4*>(cut.before);
      bool stopped = false;
      for (; n + 8u <= count && !stopped; n += 8u) {
        const float4 a = quads[n >> 2];
        const float4 b = quads[(n >> 2) + 1u];
        const float v[8] = {a.x, a.y, a.z, a.w, b.x, b.y, b.z, b.w};
#pragma unroll
        for (m1_u32 u = 0u; u < 8u; ++u) {
          if (!stopped) {
            if (!(mass < cutoff)) {
              walked = n + u;
              stopped = true;
            } else {
              mass += v[u];
            }
          }
        }
      }
      if (!stopped) {
        for (; n < count; ++n) {
          if (!(mass < cutoff)) {
            walked = n;
            break;
          }
          mass += cut.before[n];
        }
      }
    } else {
      for (; n < count; ++n) {
        const float v = cut.before[n];
        cut.before[n] = mass;
        mass += v;
      }
    }
    cut.mass = mass;
    cut.count = walked;
    const m1_u64 previous = walked == 0u ? low : cut.chunk[walked - 1u];
    cut.cursor = previous;
    if (walked < count) {
      cut.stage = M1_CUT_STOPPED;
      cut.pivot = previous;
    }
  }
  __syncthreads();
  if (!monotone) {
    for (m1_u32 n = threadIdx.x; n < count; n += blockDim.x)
      m1_store_b(output, base + (m1_u32)cut.chunk[n], cut.before[n] < cutoff);
    __syncthreads();
  }
}

// One binade of the running sum past the head (see the top of this section):
// finds the element at which the sum reaches the cutoff or its next binade,
// adds it in f32, and either stops there or moves the cursor past it.
template <class G>
__device__ void m1_cut_phase(
    const m1_u8* input, m1_u32 base, m1_u32 len, m1_u32 dtype,
    float cutoff, M1CutShared& cut, G& g) {
  const float mass = cut.mass;
  const m1_u32 mass_bits = __float_as_uint(mass);
  const m1_u32 biased = mass_bits >> 23;
  const int eu = biased == 0u ? -149 : (int)biased - 150;
  const m1_u64 start = biased == 0u
                           ? (m1_u64)(mass_bits & 0x7FFFFFu)
                           : (m1_u64)((mass_bits & 0x7FFFFFu) | 0x800000u);
  const m1_u64 top = 1ull << 24;
  // The least whole number of units at or past the cutoff, held to the
  // binade's top: ceil(cutoff / 2^eu) from the cutoff's own bits.
  m1_u64 limit = top;
  {
    const m1_u32 bits = __float_as_uint(cutoff);
    const m1_u32 cb = bits >> 23;
    if (cb < 255u) {
      const m1_u64 cm = cb == 0u ? (m1_u64)(bits & 0x7FFFFFu)
                                 : (m1_u64)((bits & 0x7FFFFFu) | 0x800000u);
      const int ec = cb == 0u ? -149 : (int)cb - 150;
      const int up = ec - eu;
      if (up >= 0) {
        if (up < 26) limit = cm << up;
      } else if (-up < 40) {
        const int down = -up;
        limit = (cm + (1ull << down) - 1ull) >> down;
      } else {
        limit = 1ull;
      }
      if (limit > top) limit = top;
    }
  }
  const m1_u64 low = cut.cursor;

  for (m1_u32 slot = threadIdx.x; slot < M1_CUT_TIE_SLOTS; slot += blockDim.x)
    cut.tie_slot[slot] = 0u;
  if (threadIdx.x == 0u) {
    cut.ties = 0u;
    cut.overflow = 0u;
    cut.crossing = 0u;
  }
  __syncthreads();
  // The first select pass gathers the phase's ties; without any it is the
  // select itself.
  m1_cut_select_key(
      input, base, len, dtype, low, ~0ull, limit - start, eu, cut, g, true);
  if (cut.overflow != 0u) {
    __syncthreads();
    if (threadIdx.x == 0u) cut.stage = M1_CUT_BAIL;
    __syncthreads();
    return;
  }
  const m1_u32 ties = cut.ties;
  // 1: the crossing lies in a stretch free of ties; 2: in a run of a tie
  // value; 0: nowhere — every live element is kept.
  if (ties == 0u) {
    if (threadIdx.x == 0u) {
      if (cut.select_short == 0u) {
        cut.crossing = 1u;
        cut.segment_low = low;
        cut.segment_high = ~0ull;
        cut.segment_units = start;
      }
    }
    __syncthreads();
  } else {
    // The tie values in key order: gathered out of the hash slots (into
    // tie_count, free until the gap pass), then each placed by its rank.
    if (threadIdx.x == 0u) cut.count = 0u;
    __syncthreads();
    for (m1_u32 slot = threadIdx.x; slot < M1_CUT_TIE_SLOTS; slot += blockDim.x) {
      const m1_u32 key = cut.tie_slot[slot];
      if (key != 0u) cut.tie_count[atomicAdd(&cut.count, 1u)] = key;
    }
    __syncthreads();
    for (m1_u32 t = threadIdx.x; t < ties; t += blockDim.x) {
      const m1_u32 key = cut.tie_count[t];
      m1_u32 rank = 0u;
      for (m1_u32 other = 0u; other < ties; ++other)
        rank += cut.tie_count[other] < key ? 1u : 0u;
      cut.tie_key[rank] = key;
    }
    __syncthreads();
    for (m1_u32 g = threadIdx.x; g <= ties; g += blockDim.x) {
      cut.gap[g] = 0ull;
      if (g < ties) cut.tie_count[g] = 0u;
    }
    // Which gap an element falls in, by a table over the row's key range
    // (the hash slots, spent): the ties below each of its stretches, from
    // which a step or two lands on the element's own gap.
    const m1_u32 floor_key = cut.key_least;
    const m1_u32 span = cut.key_most - floor_key;
    int stretch = 0;
    while (stretch < 32 && (span >> stretch) >= M1_CUT_TIE_SLOTS) ++stretch;
    for (m1_u32 b = threadIdx.x; b < M1_CUT_TIE_SLOTS; b += blockDim.x) {
      const m1_u64 from = (m1_u64)floor_key + ((m1_u64)b << stretch);
      m1_u32 lo = 0u;
      m1_u32 hi = ties;
      while (lo < hi) {
        const m1_u32 mid = (lo + hi) >> 1;
        if ((m1_u64)cut.tie_key[mid] < from) lo = mid + 1u;
        else hi = mid;
      }
      cut.tie_slot[b] = lo;
    }
    __syncthreads();
    m1_cut_scan_range(input, base, g.lo, g.hi, dtype, [&](m1_u32 i, float y) {
      if (m1_isnan(y)) return;
      const m1_u32 key = m1_desc_key(y);
      if (m1_cut_composite(key, i) <= low) return;
      m1_u32 lo = cut.tie_slot[(key - floor_key) >> stretch];
      while (lo < ties && cut.tie_key[lo] < key) ++lo;
      m1_u32 whole;
      const m1_u32 rest = m1_cut_units(y, eu, &whole);
      if (rest == 1u)
        atomicAdd(&cut.tie_count[lo], 1u);
      else
        atomicAdd(&cut.gap[lo], (m1_u64)whole + (rest == 2u ? 1ull : 0ull));
    });
    __syncthreads();
    g.gaps(cut, ties);
    if (threadIdx.x == 0u) {
      m1_u64 units = start;
      for (m1_u32 g = 0u; g <= ties; ++g) {
        if (units + cut.gap[g] >= limit) {
          cut.crossing = 1u;
          const m1_u64 floor =
              g == 0u ? 0ull : m1_cut_composite(cut.tie_key[g - 1u], 0xFFFFFFFFu);
          cut.segment_low = floor > low ? floor : low;
          cut.segment_high =
              g == ties ? ~0ull : m1_cut_composite(cut.tie_key[g], 0u);
          cut.segment_units = units;
          break;
        }
        units += cut.gap[g];
        if (g == ties) break;
        const m1_u64 run = cut.tie_count[g];
        m1_u32 whole;
        m1_cut_units(m1_cut_value(cut.tie_key[g]), eu, &whole);
        const m1_u64 q = whole;
        const m1_u64 first = q + ((units + q) & 1ull);
        if (units + first >= limit) {
          cut.crossing = 2u;
          cut.crossing_key = cut.tie_key[g];
          cut.crossing_rank = 0u;
          cut.crossing_units = units;
          break;
        }
        units += first;
        const m1_u64 next = q + (q & 1ull);
        if (run > 1ull && next != 0ull && units + (run - 1ull) * next >= limit) {
          const m1_u64 steps = (limit - units + next - 1ull) / next;
          cut.crossing = 2u;
          cut.crossing_key = cut.tie_key[g];
          cut.crossing_rank = (m1_u32)steps;
          cut.crossing_units = units + (steps - 1ull) * next;
          break;
        }
        units += (run - 1ull) * next;
      }
    }
    __syncthreads();
  }
  const m1_u32 crossing = cut.crossing;
  if (crossing == 0u) {
    __syncthreads();
    if (threadIdx.x == 0u) {
      cut.stage = M1_CUT_ALL;
      cut.nan_kept = 1u;
    }
    __syncthreads();
    return;
  }
  m1_u32 key;
  m1_u32 index;
  m1_u64 units;
  if (crossing == 1u) {
    const m1_u64 segment_low = cut.segment_low;
    const m1_u64 segment_units = cut.segment_units;
    if (ties != 0u)
      m1_cut_select_key(
          input, base, len, dtype, segment_low, cut.segment_high,
          limit - segment_units, eu, cut, g);
    if (cut.select_short != 0u) {
      __syncthreads();
      if (threadIdx.x == 0u) cut.stage = M1_CUT_BAIL;
      __syncthreads();
      return;
    }
    key = cut.select_prefix;
    const m1_u64 before = cut.select_before;
    const m1_u64 rest = cut.select_need;
    const m1_u32 least = cut.select_least;
    const m1_u64 each = m1_cut_weight(m1_cut_value(key), eu);
    const m1_u64 rank = (rest + each - 1ull) / each - 1ull;
    index = rank == 0ull
                ? least
                : m1_cut_select_index(
                      input, base, len, dtype, key, segment_low, rank, cut, g);
    units = segment_units + before + rank * each;
  } else {
    key = cut.crossing_key;
    units = cut.crossing_units;
    index = m1_cut_select_index(
        input, base, len, dtype, key, low, cut.crossing_rank, cut, g);
  }
  __syncthreads();
  if (threadIdx.x == 0u) {
    const float before = ldexpf((float)units, eu);
    const float after = before + m1_cut_value(key);
    const m1_u64 composite = m1_cut_composite(key, index);
    if (!(after < cutoff)) {
      cut.stage = M1_CUT_STOPPED;
      cut.pivot = composite;
    } else {
      cut.cursor = composite;
      cut.mass = after;
    }
  }
  __syncthreads();
}

// The cut of one row, in steps a caller can take apart: the cut's state
// reset, each element of the group's stretch noted (NaNs, signs, the key
// range), the notes merged over the group, the walk, the verdicts marked.
__device__ __forceinline__ void m1_cut_reset(M1CutShared& cut) {
  if (threadIdx.x == 0u) {
    cut.first_nan = 0xFFFFFFFFu;
    cut.negative = 0u;
    cut.wide = 0u;
    cut.cursor = 0ull;
    cut.pivot = 0ull;
    cut.mass = 0.0f;
    cut.stage = M1_CUT_RUN;
    cut.nan_kept = 0u;
    cut.key_least = 0xFFFFFFFFu;
    cut.key_most = 0u;
    cut.key_skip = 0u;
  }
}

__device__ __forceinline__ void m1_cut_note(
    M1CutShared& cut, m1_u32 i, float y, m1_u32& least, m1_u32& most) {
  if (m1_isnan(y)) {
    atomicMin(&cut.first_nan, i);
  } else {
    if (y < 0.0f) cut.negative = 1u;
    if (!(y < 1.8446744e19f)) cut.wide = 1u;
    const m1_u32 key = m1_desc_key(y);
    least = key < least ? key : least;
    most = key > most ? key : most;
  }
}

template <class G>
__device__ void m1_cut_enter(M1CutShared& cut, G& g, m1_u32 least, m1_u32 most) {
  if (least <= most) {
    atomicMin(&cut.key_least, least);
    atomicMax(&cut.key_most, most);
  }
  __syncthreads();
  g.entry(cut);
  if (threadIdx.x == 0u && cut.key_least <= cut.key_most) {
    const m1_u32 differ = cut.key_least ^ cut.key_most;
    m1_u32 skip = 0u;
    while (skip < 3u && (differ >> (24u - 8u * skip)) == 0u) ++skip;
    cut.key_skip = skip;
  }
  __syncthreads();
}

// The walk, after the notes: leaves the stage, the pivot and the NaN's
// verdict in `cut` (a row with a negative value has its verdicts stored as
// it goes).
template <class G>
__device__ __forceinline__ void m1_cut_walk(
    const m1_u8* input, m1_u32 base, m1_u32 len, m1_u32 dtype, float cutoff,
    m1_u8* output, M1CutShared& cut, G& g) {
  const bool monotone = cut.negative == 0u;
  bool binades = monotone && cut.wide == 0u;
  // A group that can walk the row from a histogram first does (its cursor
  // and mass are where the passes below pick up, if it stops short).
  if (binades && cutoff > 0.0f) g.fast(input, base, len, dtype, cutoff, cut);
  if (cut.stage == M1_CUT_RUN)
    m1_cut_chunk(input, base, len, dtype, cutoff, monotone, output, cut, g);
  for (m1_u32 step = 0u; cut.stage == M1_CUT_RUN; ++step) {
    if (binades && step < 512u) {
      m1_cut_phase(input, base, len, dtype, cutoff, cut, g);
      if (cut.stage == M1_CUT_BAIL) {
        binades = false;
        __syncthreads();
        if (threadIdx.x == 0u) cut.stage = M1_CUT_RUN;
        __syncthreads();
      }
    } else {
      m1_cut_chunk(input, base, len, dtype, cutoff, monotone, output, cut, g);
    }
  }
}

// Whether the element `i` of value `y` is kept, by the walk's outcome (a
// row without a negative value).
__device__ __forceinline__ bool m1_cut_kept(const M1CutShared& cut, m1_u32 i, float y) {
  if (m1_isnan(y)) return cut.nan_kept != 0u && i == cut.first_nan;
  return cut.stage == M1_CUT_ALL || m1_cut_composite(m1_desc_key(y), i) <= cut.pivot;
}

template <class G>
__device__ void m1_cut_mark(
    const m1_u8* input, m1_u32 base, m1_u32 dtype, m1_u8* output, M1CutShared& cut, G& g) {
  const bool monotone = cut.negative == 0u;
  m1_cut_scan_range(input, base, g.lo, g.hi, dtype, [&](m1_u32 i, float y) {
    if (m1_isnan(y) || monotone) m1_store_b(output, base + i, m1_cut_kept(cut, i, y));
  });
  __syncthreads();
}

// The cut of one row (at `base`, `len` wide) on the group `g`: the verdicts
// of the group's stretch of the row land in `output`.
template <class G>
__device__ __forceinline__ void m1_cut_row(
    const m1_u8* input, m1_u32 base, m1_u32 len, m1_u32 dtype, float cutoff,
    m1_u8* output, M1CutShared& cut, G& g) {
  m1_cut_reset(cut);
  __syncthreads();
  m1_u32 least = 0xFFFFFFFFu;
  m1_u32 most = 0u;
  m1_cut_scan_range(input, base, g.lo, g.hi, dtype, [&](m1_u32 i, float y) {
    m1_cut_note(cut, i, y, least, most);
  });
  m1_cut_enter(cut, g, least, most);
  m1_cut_walk(input, base, len, dtype, cutoff, output, cut, g);
  m1_cut_mark(input, base, dtype, output, cut, g);
}

__device__ __forceinline__ void ptir_parallel_pivot_cummass(
    const m1_u8* input,
    const m1_u8* threshold,
    m1_u8* output,
    const M1ValueDesc input_desc,
    const M1ValueDesc threshold_desc) {
  M1_CUT_STATE(cut);
  const m1_u32 len = input_desc.last;
  const m1_u32 rows = input_desc.rows;
  const m1_u32 dtype = input_desc.dtype;
  if (len == 0u) return;
  M1Solo g;
  g.row(len);
  for (m1_u32 row = 0; row < rows; ++row) {
    const m1_u32 base = row * len;
    const float cutoff = m1_load_f(
        threshold,
        m1_pick(threshold_desc.len, row),
        threshold_desc.dtype);
    m1_cut_row(input, base, len, dtype, cutoff, output, cut, g);
  }
}

// The top-k cut of one row (`k` already clamped to 1..len) on the group
// `grp`: the verdicts of the group's stretch land in `output`.
template <class G>
__device__ m1_u32 m1_rank_row(
    const m1_u8* input, m1_u32 base, m1_u32 len, m1_u32 dtype, m1_u32 k,
    m1_u8* output, G& grp, bool mark = true) {
  __shared__ m1_u32 rank_hist[2048];
  __shared__ m1_u32 rank_group[64];
  __shared__ m1_u32 rank_prefix;
  __shared__ m1_u32 rank_target;
  (void)len;
  if (threadIdx.x == 0u) {
    rank_prefix = 0u;
    rank_target = k;
  }
  __syncthreads();
  for (int pass = 0; pass < 3; ++pass) {
    const int width = pass == 2 ? 10 : 11;
    const int shift = pass == 0 ? 21 : (pass == 1 ? 10 : 0);
    const m1_u32 bins = 1u << width;
    const m1_u32 high_mask =
        pass == 0 ? 0u : (0xFFFFFFFFu << (shift + width));
    for (m1_u32 bin = threadIdx.x; bin < bins; bin += blockDim.x)
      rank_hist[bin] = 0u;
    __syncthreads();
    const m1_u32 prefix = rank_prefix & high_mask;
    m1_u32 held_bin = 0xFFFFFFFFu;
    m1_u32 held = 0u;
    m1_cut_scan_range(input, base, grp.lo, grp.hi, dtype, [&](m1_u32 i, float y) {
      (void)i;
      const m1_u32 key = m1_desc_key(y);
      if ((key & high_mask) != prefix) return;
      const m1_u32 bin = (key >> shift) & (bins - 1u);
      if (bin != held_bin) {
        if (held != 0u) atomicAdd(&rank_hist[held_bin], held);
        held_bin = bin;
        held = 0u;
      }
      ++held;
    });
    if (held != 0u) atomicAdd(&rank_hist[held_bin], held);
    __syncthreads();
    grp.sum_u32(rank_hist, bins);
    // The bin holding the target-th element: group sums of 32 bins, then
    // one thread walks the groups and the chosen group's bins.
    const m1_u32 groups = bins >> 5;
    for (m1_u32 g = threadIdx.x; g < groups; g += blockDim.x) {
      m1_u32 sum = 0u;
      for (m1_u32 b = 0u; b < 32u; ++b) sum += rank_hist[(g << 5) + b];
      rank_group[g] = sum;
    }
    __syncthreads();
    if (threadIdx.x == 0u) {
      const m1_u32 target = rank_target;
      m1_u32 run = 0u;
      m1_u32 g = 0u;
      while (g + 1u < groups && run + rank_group[g] < target) {
        run += rank_group[g];
        ++g;
      }
      m1_u32 bin = g << 5;
      const m1_u32 last = bin + 31u;
      while (bin < last && run + rank_hist[bin] < target) {
        run += rank_hist[bin];
        ++bin;
      }
      rank_target = target - run;
      rank_prefix = prefix | (bin << shift);
    }
    __syncthreads();
  }
  const m1_u32 cutoff_key = rank_prefix;
  if (mark) {
    m1_cut_scan_range(input, base, grp.lo, grp.hi, dtype, [&](m1_u32 i, float y) {
      m1_store_b(output, base + i, !m1_isnan(y) && m1_desc_key(y) <= cutoff_key);
    });
  }
  __syncthreads();
  return cutoff_key;
}

// The top-k cut, `PivotThreshold(RankLe)`, of each row on one block.
__device__ __forceinline__ void ptir_parallel_pivot_rank(
    const m1_u8* input,
    const m1_u8* threshold,
    m1_u8* output,
    const M1ValueDesc input_desc,
    const M1ValueDesc threshold_desc) {
  const m1_u32 len = input_desc.last;
  const m1_u32 rows = input_desc.rows;
  const m1_u32 dtype = input_desc.dtype;
  if (len == 0u) return;
  M1Solo grp;
  grp.row(len);
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
    m1_rank_row(input, base, len, dtype, k, output, grp);
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

__device__ __forceinline__ float m1_reduce_identity(m1_u32 tag) {
  return tag == 0x30u ? 0.0f : (tag == 0x31u ? m1_neg_inf() : m1_pos_inf());
}

__device__ __forceinline__ float m1_reduce_combine(
    m1_u32 tag, float left, float right) {
  return tag == 0x30u
             ? left + right
             : (tag == 0x31u ? m1_canonical_max(left, right)
                             : m1_canonical_min(left, right));
}

// One chunk's fold, the interpreter's warp tree: lane `l` takes in lane
// `l + offset` for `offset` 16, 8, 4, 2, 1; lane 0 holds the chunk's value.
// Every lane of the chunk calls it.
__device__ __forceinline__ float m1_reduce_chunk(m1_u32 tag, float value) {
  const m1_u32 lane = threadIdx.x & 31u;
  for (m1_u32 offset = 16u; offset > 0u; offset >>= 1u) {
    const float other = m1_chunk_down(value, offset);
    if (lane < offset) value = m1_reduce_combine(tag, value, other);
  }
  return value;
}

// The canonical reduction of `count` floats at `level` — the interpreter's
// `canonical_reduce`: while more than one value is left, each run of 32 folds
// into one, the last run padded with the identity, into `out` and then
// `spare` by turns. One value is the answer as it stands, never folded with
// the identity (`-0` stays `-0`). The whole block calls it, and every thread
// returns the answer.
__device__ __forceinline__ float ptir_canonical_reduce(
    m1_u32 tag,
    const float* level,
    m1_u32 count,
    float* out,
    float* spare) {
  const m1_u32 lane = threadIdx.x & 31u;
  const m1_u32 warp = threadIdx.x >> 5u;
  const m1_u32 warps = blockDim.x >> 5u;
  if (count == 0u) return m1_reduce_identity(tag);
  while (count > 1u) {
    const m1_u32 chunks = (count + 31u) / 32u;
    for (m1_u32 chunk = warp; chunk < chunks; chunk += warps) {
      const m1_u32 index = chunk * 32u + lane;
      const float value = m1_reduce_chunk(
          tag, index < count ? level[index] : m1_reduce_identity(tag));
      if (lane == 0u) out[chunk] = value;
    }
    __syncthreads();
    level = out;
    float* const next = spare;
    spare = out;
    out = next;
    count = chunks;
  }
  const float answer = level[0];
  __syncthreads();
  return answer;
}

__device__ __forceinline__ void ptir_parallel_reduce_f32(
    m1_u32 tag,
    const m1_u8* input,
    m1_u8* output,
    m1_u8* temporary,
    const M1ValueDesc input_desc) {
  float* work_a = reinterpret_cast<float*>(temporary);
  float* work_b = work_a + (input_desc.last + 31u) / 32u;
  const float* values = reinterpret_cast<const float*>(input);
  float* result = reinterpret_cast<float*>(output);
  for (m1_u32 row = 0; row < input_desc.rows; ++row) {
    const float answer = ptir_canonical_reduce(
        tag, values + row * input_desc.last, input_desc.last, work_a, work_b);
    if (threadIdx.x == 0u) result[row] = answer;
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
              m1_chunk_down(candidate.value, offset),
              m1_chunk_down(candidate.index, offset),
              m1_chunk_down(candidate.have, offset),
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
                m1_chunk_down(candidate.value, offset),
                m1_chunk_down(candidate.index, offset),
                m1_chunk_down(candidate.have, offset),
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
            m1_chunk_down(candidate.value, offset),
            m1_chunk_down(candidate.index, offset),
            m1_chunk_down(candidate.have, offset)};
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
              m1_chunk_down(candidate.value, offset),
              m1_chunk_down(candidate.index, offset),
              m1_chunk_down(candidate.have, offset)};
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
  for (m1_u32 offset = 16u; offset > 0u; offset >>= 1u) {
    M1ArgmaxCandidate other{
        m1_chunk_down(candidate.value, offset),
        m1_chunk_down(candidate.index, offset),
        m1_chunk_down(candidate.have, offset),
        0u};
    if (lane < offset)
      candidate = m1_argmax_combine(candidate, other);
  }
  return candidate;
}

__device__ __forceinline__ M1IntArgmaxCandidate m1_int_argmax_warp_reduce(
    M1IntArgmaxCandidate candidate, m1_u32 lane) {
  for (m1_u32 offset = 16u; offset > 0u; offset >>= 1u) {
    M1IntArgmaxCandidate other{
        m1_chunk_down(candidate.value, offset),
        m1_chunk_down(candidate.index, offset),
        m1_chunk_down(candidate.have, offset)};
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

    // 16-byte vectors (eight bf16 or four f32 values), FLIGHT of them in
    // flight a thread before any is folded: one block scans the whole row,
    // so its pace is how many loads it keeps outstanding. The pick is the
    // same in any order (the greatest value, the least index among equals,
    // NaN never), so the scan's order is free.
    constexpr m1_u32 FLIGHT = 4u;
    const m1_u32 per = mode == 0u ? 4u : 8u;
    const m1_u32 vectors =
        ((m1_u64)row_base & 15ull) == 0ull ? last / per : 0u;
    const M1U32x4* packed_row = reinterpret_cast<const M1U32x4*>(row_base);
    for (m1_u32 v0 = threadIdx.x; v0 < vectors; v0 += blockDim.x * FLIGHT) {
      M1U32x4 packed[FLIGHT];
#pragma unroll
      for (m1_u32 f = 0; f < FLIGHT; ++f) {
        const m1_u32 v = v0 + f * blockDim.x;
        if (v < vectors) packed[f] = packed_row[v];
      }
#pragma unroll
      for (m1_u32 f = 0; f < FLIGHT; ++f) {
        const m1_u32 v = v0 + f * blockDim.x;
        if (v >= vectors) continue;
        const m1_u32 words[4] = {packed[f].x, packed[f].y, packed[f].z, packed[f].w};
        if (mode == 0u) {
#pragma unroll
          for (m1_u32 k = 0; k < 4u; ++k) {
            const float value = __uint_as_float(words[k]);
            const M1ArgmaxCandidate next{
                value, v * 4u + k, m1_isnan(value) ? 0u : 1u, 0u};
            candidate = m1_argmax_combine(candidate, next);
          }
        } else {
#pragma unroll
          for (m1_u32 k = 0; k < 8u; ++k) {
            const float value =
                __uint_as_float((words[k >> 1] >> ((k & 1u) * 16u)) << 16);
            const M1ArgmaxCandidate next{
                value, v * 8u + k, m1_isnan(value) ? 0u : 1u, 0u};
            candidate = m1_argmax_combine(candidate, next);
          }
        }
      }
    }
    for (m1_u32 index = vectors * per + threadIdx.x;
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

__device__ __forceinline__ void ptir_fast_gumbel_argmax_intrinsic(
    const m1_u8* input,
    m1_u8* output,
    const M1ValueDesc input_desc,
    m1_u32 mode,
    m1_u32 stride,
    m1_u32 row_offset,
    const m1_u8* state,
    const M1ValueDesc state_desc,
    const m1_u8* divisor,
    const M1ValueDesc divisor_desc,
    m1_u32 noise_base) {
  __shared__ M1ArgmaxCandidate candidates[32];
  int* result = reinterpret_cast<int*>(output);
  const m1_u32 lane = threadIdx.x & 31u;
  const m1_u32 warp = threadIdx.x >> 5u;
  const m1_u32 warps = blockDim.x >> 5u;
  const m1_u64 seed = ptir_rng_state_seed(state, state_desc);
  const bool scaled = divisor != nullptr;
  const float divide_by =
      scaled ? m1_load_f(divisor, 0u, divisor_desc.dtype) : 1.0f;
  for (m1_u32 row = 0; row < input_desc.rows; ++row) {
    const m1_u8* row_base = m1_intrinsic_row_base(
        input, (m1_u64)row_offset + row, stride, mode);
    const m1_u32 last = input_desc.last;
    const m1_u32 element_base = noise_base + row * last;
    M1ArgmaxCandidate candidate{m1_neg_inf(), 0u, 0u, 0u};
    const m1_u32 vectors =
        (mode != 0u && ((m1_u64)row_base & 15ull) == 0ull) ? (last >> 3) : 0u;
    for (m1_u32 v = threadIdx.x; v < vectors; v += blockDim.x) {
      const M1U32x4 packed = reinterpret_cast<const M1U32x4*>(row_base)[v];
      const m1_u32 words[4] = {packed.x, packed.y, packed.z, packed.w};
#pragma unroll
      for (m1_u32 k = 0; k < 8u; ++k) {
        const m1_u32 index = v * 8u + k;
        float value = __uint_as_float((words[k >> 1] >> ((k & 1u) * 16u)) << 16);
        if (scaled) value = value / divide_by;
        const float uniform = ptir_rng_hash_uniform(seed, element_base + index);
        value = value + (-logf(-logf(uniform)));
        const M1ArgmaxCandidate next{value, index, m1_isnan(value) ? 0u : 1u, 0u};
        candidate = m1_argmax_combine(candidate, next);
      }
    }
    for (m1_u32 index = (vectors << 3) + threadIdx.x;
         index < last;
         index += blockDim.x) {
      float value = m1_intrinsic_column_load(row_base, index, mode);
      if (scaled) value = value / divide_by;
      const float uniform = ptir_rng_hash_uniform(seed, element_base + index);
      value = value + (-logf(-logf(uniform)));
      const M1ArgmaxCandidate next{value, index, m1_isnan(value) ? 0u : 1u, 0u};
      candidate = m1_argmax_combine(candidate, next);
    }
    candidate = m1_argmax_warp_reduce(candidate, lane);
    if (lane == 0u) candidates[warp] = candidate;
    __syncthreads();
    if (warp == 0u) {
      candidate =
          lane < warps ? candidates[lane] : M1ArgmaxCandidate{m1_neg_inf(), 0u, 0u, 0u};
      candidate = m1_argmax_warp_reduce(candidate, lane);
      if (lane == 0u) result[row] = (int)candidate.index;
    }
    __syncthreads();
  }
}

extern "C" __global__ void 