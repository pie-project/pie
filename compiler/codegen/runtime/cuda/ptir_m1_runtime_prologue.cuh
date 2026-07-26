
typedef unsigned char m1_u8;
typedef unsigned short m1_u16;
typedef unsigned int m1_u32;
typedef unsigned long long m1_u64;

// A 16-byte load, spelled without depending on NVRTC providing CUDA's built-in
// vector types: these headers are compiled with no include path at all.
struct alignas(16) M1U32x4 {
  unsigned int x, y, z, w;
};

struct M1Status {
  m1_u32 state;
  m1_u32 fault;
  m1_u32 reserved0;
  m1_u32 reserved1;
};

struct M1ValueDesc {
  m1_u32 len;
  m1_u32 rows;
  m1_u32 last;
  m1_u32 rank;
  m1_u32 dtype;
  m1_u32 dims[4];
};

struct M1OpParams {
  m1_u32 tag;
  m1_u32 a0;
  m1_u32 a1;
  m1_u32 a2;
  m1_u32 o0;
  m1_u32 o1;
  m1_u32 imm;
  m1_u32 imm2;
  m1_u32 imm3;
  m1_u32 kind;
  m1_u32 pred_tag;
  m1_u32 lit_dtype;
  m1_u32 lit_bits;
  m1_u32 channel_slot;
  m1_u32 intr;
  m1_u32 sink_bytes;
  m1_u32 intrinsic_dtype;
  m1_u32 bool_storage;
  m1_u32 intrinsic_row_stride;
  m1_u32 intrinsic_row_offset;
  m1_u64 rng_seed;
};

static_assert(sizeof(M1Status) == 16, "M1Status ABI");
static_assert(sizeof(M1ValueDesc) == 36, "M1ValueDesc ABI");
static_assert(sizeof(M1OpParams) == 88, "M1OpParams ABI");

struct M1ArgmaxCandidate {
  float value;
  m1_u32 index;
  m1_u32 have;
  m1_u32 reserved;
};

struct M1IntArgmaxCandidate {
  long long value;
  m1_u32 index;
  m1_u32 have;
};

__device__ __forceinline__ float m1_pos_inf() {
  return __int_as_float(0x7f800000);
}

__device__ __forceinline__ float m1_neg_inf() {
  return __int_as_float((int)0xff800000u);
}

__device__ __forceinline__ float m1_nan() {
  return __int_as_float(0x7fc00000);
}

__device__ __forceinline__ bool m1_isnan(float value) {
  return value != value;
}

__device__ __forceinline__ bool m1_signbit(float value) {
  return (__float_as_uint(value) >> 31) != 0;
}

__device__ __forceinline__ int m1_bits_i32(m1_u32 value) {
  union Bits {
    m1_u32 u;
    int i;
  } bits;
  bits.u = value;
  return bits.i;
}

__device__ __forceinline__ float m1_bits_f32(m1_u32 value) {
  return __uint_as_float(value);
}

// Monotone map float -> m1_u32 that REVERSES value order: a larger float yields
// a smaller key, so a plain unsigned radix select over the keys walks the values
// in descending order. NaN maps to the maximum key (sorts last), and no finite
// float can collide with that sentinel.
__device__ __forceinline__ m1_u32 m1_desc_key(float value) {
  if (m1_isnan(value)) return 0xFFFFFFFFu;
  if (value == 0.0f) value = 0.0f;   // -0.0 compares equal to +0.0
  const m1_u32 u = __float_as_uint(value);
  const m1_u32 ascending = (u & 0x80000000u) ? ~u : (u | 0x80000000u);
  return ~ascending;
}

__device__ __forceinline__ int m1_float_to_i32(float value) {
  if (m1_isnan(value)) return 0;
  if (value >= 2147483647.0f) return 2147483647;
  if (value <= -2147483648.0f) return m1_bits_i32(0x80000000u);
  return (int)value;
}

__device__ __forceinline__ m1_u32 m1_float_to_u32(float value) {
  if (m1_isnan(value) || value <= 0.0f) return 0u;
  if (value >= 4294967295.0f) return 0xffffffffu;
  return (m1_u32)value;
}

__device__ __forceinline__ float m1_load_f(
    const m1_u8* data, m1_u32 index, m1_u32 dtype) {
  if (dtype == 0) return reinterpret_cast<const float*>(data)[index];
  if (dtype == 1) return float(reinterpret_cast<const int*>(data)[index]);
  if (dtype == 2) return float(reinterpret_cast<const m1_u32*>(data)[index]);
  return data[index] != 0 ? 1.0f : 0.0f;
}

__device__ __forceinline__ int m1_load_i(
    const m1_u8* data, m1_u32 index, m1_u32 dtype) {
  if (dtype == 0)
    return m1_float_to_i32(reinterpret_cast<const float*>(data)[index]);
  if (dtype == 1) return reinterpret_cast<const int*>(data)[index];
  if (dtype == 2)
    return m1_bits_i32(reinterpret_cast<const m1_u32*>(data)[index]);
  return data[index] != 0 ? 1 : 0;
}

__device__ __forceinline__ m1_u32 m1_load_u(
    const m1_u8* data, m1_u32 index, m1_u32 dtype) {
  if (dtype == 0)
    return m1_float_to_u32(reinterpret_cast<const float*>(data)[index]);
  if (dtype == 1)
    return (m1_u32)reinterpret_cast<const int*>(data)[index];
  if (dtype == 2) return reinterpret_cast<const m1_u32*>(data)[index];
  return data[index] != 0 ? 1u : 0u;
}

__device__ __forceinline__ bool m1_load_b(
    const m1_u8* data, m1_u32 index, m1_u32 dtype) {
  if (dtype == 0)
    return reinterpret_cast<const float*>(data)[index] != 0.0f;
  if (dtype == 1)
    return reinterpret_cast<const int*>(data)[index] != 0;
  if (dtype == 2)
    return reinterpret_cast<const m1_u32*>(data)[index] != 0u;
  return data[index] != 0;
}

__device__ __forceinline__ void m1_store_f(
    m1_u8* data, m1_u32 index, float value) {
  reinterpret_cast<float*>(data)[index] = value;
}

__device__ __forceinline__ void m1_store_i(
    m1_u8* data, m1_u32 index, int value) {
  reinterpret_cast<int*>(data)[index] = value;
}

__device__ __forceinline__ void m1_store_u(
    m1_u8* data, m1_u32 index, m1_u32 value) {
  reinterpret_cast<m1_u32*>(data)[index] = value;
}

__device__ __forceinline__ void m1_store_b(
    m1_u8* data, m1_u32 index, bool value) {
  data[index] = value ? 1 : 0;
}

__device__ __forceinline__ float m1_canonical_max(
    float left, float right) {
  const bool ln = m1_isnan(left);
  const bool rn = m1_isnan(right);
  if (ln && rn) return m1_neg_inf();
  if (ln) return right;
  if (rn) return left;
  if (left == 0.0f && right == 0.0f)
    return m1_signbit(left) && m1_signbit(right) ? -0.0f : 0.0f;
  return left > right ? left : right;
}

__device__ __forceinline__ float m1_canonical_min(
    float left, float right) {
  const bool ln = m1_isnan(left);
  const bool rn = m1_isnan(right);
  if (ln && rn) return m1_pos_inf();
  if (ln) return right;
  if (rn) return left;
  if (left == 0.0f && right == 0.0f)
    return m1_signbit(left) || m1_signbit(right) ? -0.0f : 0.0f;
  return left < right ? left : right;
}

__device__ __forceinline__ float m1_element_max(
    float left, float right) {
  const bool ln = m1_isnan(left);
  const bool rn = m1_isnan(right);
  if (ln && rn) return left;
  if (ln) return right;
  if (rn) return left;
  if (left == 0.0f && right == 0.0f)
    return m1_signbit(left) && m1_signbit(right) ? -0.0f : 0.0f;
  return left > right ? left : right;
}

__device__ __forceinline__ float m1_element_min(
    float left, float right) {
  const bool ln = m1_isnan(left);
  const bool rn = m1_isnan(right);
  if (ln && rn) return left;
  if (ln) return right;
  if (rn) return left;
  if (left == 0.0f && right == 0.0f)
    return m1_signbit(left) || m1_signbit(right) ? -0.0f : 0.0f;
  return left < right ? left : right;
}

__device__ __forceinline__ long long m1_load_index(
    const m1_u8* data, m1_u32 index, m1_u32 dtype) {
  if (dtype == 1)
    return (long long)reinterpret_cast<const int*>(data)[index];
  if (dtype == 2)
    return (long long)reinterpret_cast<const m1_u32*>(data)[index];
  if (dtype == 3) return data[index] != 0 ? 1ll : 0ll;
  return (long long)m1_float_to_i32(
      reinterpret_cast<const float*>(data)[index]);
}

__device__ __forceinline__ M1ArgmaxCandidate m1_argmax_combine(
    M1ArgmaxCandidate left, M1ArgmaxCandidate right) {
  if (right.have == 0) return left;
  if (left.have == 0 || right.value > left.value ||
      (right.value == left.value && right.index < left.index)) {
    return right;
  }
  return left;
}

__device__ __forceinline__ M1IntArgmaxCandidate m1_int_argmax_combine(
    M1IntArgmaxCandidate left, M1IntArgmaxCandidate right) {
  if (right.have == 0) return left;
  if (left.have == 0 || right.value > left.value ||
      (right.value == left.value && right.index < left.index)) {
    return right;
  }
  return left;
}

__device__ __forceinline__ bool m1_sort_better(
    float value, m1_u32 index, float best, m1_u32 best_index) {
  const bool value_nan = m1_isnan(value);
  const bool best_nan = m1_isnan(best);
  if (value_nan != best_nan) return best_nan;
  if (value_nan) return index < best_index;
  if (value != best) return value > best;
  return index < best_index;
}

__device__ __forceinline__ m1_u32 m1_pick(
    m1_u32 len, m1_u32 index) {
  return len == 1 ? 0u : index;
}

__device__ __forceinline__ void m1_fault(
    M1Status* status, m1_u32 code) {
  status->fault = code;
  status->state = 3;
}

__device__ __forceinline__ void m1_copy_typed(
    const m1_u8* input, m1_u8* output, m1_u32 len, m1_u32 dtype) {
  const m1_u32 bytes = dtype == 3 ? len : len * 4u;
  for (m1_u32 i = 0; i < bytes; ++i) output[i] = input[i];
}

__device__ __forceinline__ int m1_i32_div(int left, int right) {
  if (right == 0) return 0;
  if ((m1_u32)left == 0x80000000u && right == -1) return left;
  return left / right;
}

__device__ __forceinline__ int m1_i32_rem(int left, int right) {
  if (right == 0 || ((m1_u32)left == 0x80000000u && right == -1))
    return 0;
  return left % right;
}

__device__ __forceinline__ float m1_intrinsic_load(
    const m1_u8* input, m1_u64 index, m1_u32 mode) {
  if (mode == 0)
    return reinterpret_cast<const float*>(input)[index];
  const m1_u32 bits =
      (m1_u32)reinterpret_cast<const m1_u16*>(input)[index] << 16;
  return __uint_as_float(bits);
}

__device__ __forceinline__ float m1_intrinsic_row_load(
    const m1_u8* input,
    m1_u64 row,
    m1_u32 column,
    m1_u32 stride,
    m1_u32 mode) {
  if (mode != 2u)
    return m1_intrinsic_load(
        input, row * (m1_u64)stride + column, mode);
  const m1_u64 row_address =
      reinterpret_cast<const m1_u64*>(input)[row];
  const m1_u16 value =
      reinterpret_cast<const m1_u16*>(row_address)[column];
  return __uint_as_float((m1_u32)value << 16);
}

// Resolve a row once, then read columns off it.
//
// `mode == 2` keeps a table of row pointers, so calling `m1_intrinsic_row_load`
// per column put a dependent global load in front of every element: the greedy
// argmax over a 154k vocabulary read its logits at 37 GB/s. The row is loop
// invariant and the dtype branch is block-uniform, so both belong outside the
// column loop.
//
// `mode == 3` (pre-reduced, see `ptir_fast_argmax_intrinsic`) never reaches
// here: there are no columns to address, so its only reader returns before
// resolving a row.
__device__ __forceinline__ const m1_u8* m1_intrinsic_row_base(
    const m1_u8* input, m1_u64 row, m1_u32 stride, m1_u32 mode) {
  if (mode == 2u) {
    return reinterpret_cast<const m1_u8*>(
        reinterpret_cast<const m1_u64*>(input)[row]);
  }
  const m1_u64 element = mode == 0u ? 4u : 2u;
  return input + row * (m1_u64)stride * element;
}

__device__ __forceinline__ float m1_intrinsic_column_load(
    const m1_u8* row_base, m1_u32 column, m1_u32 mode) {
  if (mode == 0u) return reinterpret_cast<const float*>(row_base)[column];
  const m1_u32 bits =
      (m1_u32)reinterpret_cast<const m1_u16*>(row_base)[column] << 16;
  return __uint_as_float(bits);
}

__device__ __forceinline__ void m1_reduce_float(
    m1_u32 tag,
    const m1_u8* input,
    m1_u8* output,
    m1_u8* temporary,
    M1ValueDesc in_desc) {
  float* work = reinterpret_cast<float*>(temporary);
  const float* values = reinterpret_cast<const float*>(input);
  float* result = reinterpret_cast<float*>(output);
  for (m1_u32 row = 0; row < in_desc.rows; ++row) {
    const m1_u32 base = row * in_desc.last;
    for (m1_u32 i = 0; i < in_desc.last; ++i)
      work[i] = values[base + i];
    m1_u32 count = in_desc.last;
    if (count == 0) {
      result[row] =
          tag == 0x30 ? 0.0f
                      : (tag == 0x31 ? m1_neg_inf() : m1_pos_inf());
      continue;
    }
    while (count > 1) {
      const m1_u32 chunks = (count + 31u) / 32u;
      for (m1_u32 chunk = 0; chunk < chunks; ++chunk) {
        float lanes[32];
        const float identity =
            tag == 0x30 ? 0.0f
                        : (tag == 0x31 ? m1_neg_inf() : m1_pos_inf());
        for (m1_u32 lane = 0; lane < 32; ++lane) {
          const m1_u32 index = chunk * 32u + lane;
          lanes[lane] = index < count ? work[index] : identity;
        }
        for (m1_u32 offset = 16; offset > 0; offset >>= 1) {
          for (m1_u32 lane = 0; lane < offset; ++lane) {
            if (tag == 0x30)
              lanes[lane] += lanes[lane + offset];
            else if (tag == 0x31)
              lanes[lane] =
                  m1_canonical_max(lanes[lane], lanes[lane + offset]);
            else
              lanes[lane] =
                  m1_canonical_min(lanes[lane], lanes[lane + offset]);
          }
        }
        work[chunk] = lanes[0];
      }
      count = chunks;
    }
    result[row] = work[0];
  }
}

__device__ __forceinline__ void m1_reduce_integer(
    m1_u32 tag,
    const m1_u8* input,
    m1_u8* output,
    m1_u8* temporary,
    M1ValueDesc in_desc) {
  m1_u32* work = reinterpret_cast<m1_u32*>(temporary);
  for (m1_u32 row = 0; row < in_desc.rows; ++row) {
    const m1_u32 base = row * in_desc.last;
    for (m1_u32 i = 0; i < in_desc.last; ++i) {
      work[i] =
          in_desc.dtype == 1
              ? (m1_u32)reinterpret_cast<const int*>(input)[base + i]
              : reinterpret_cast<const m1_u32*>(input)[base + i];
    }
    m1_u32 count = in_desc.last;
    if (count == 0) {
      if (in_desc.dtype == 1) {
        reinterpret_cast<int*>(output)[row] =
            tag == 0x30
                ? 0
                : (tag == 0x31 ? m1_bits_i32(0x80000000u)
                               : 2147483647);
      } else {
        reinterpret_cast<m1_u32*>(output)[row] =
            tag == 0x32 ? 0xffffffffu : 0u;
      }
      continue;
    }
    while (count > 1) {
      const m1_u32 chunks = (count + 31u) / 32u;
      for (m1_u32 chunk = 0; chunk < chunks; ++chunk) {
        m1_u32 lanes[32];
        for (m1_u32 lane = 0; lane < 32; ++lane) {
          const m1_u32 index = chunk * 32u + lane;
          if (index < count)
            lanes[lane] = work[index];
          else if (tag == 0x30)
            lanes[lane] = 0u;
          else if (in_desc.dtype == 1)
            lanes[lane] =
                tag == 0x31 ? 0x80000000u : 0x7fffffffu;
          else
            lanes[lane] = tag == 0x31 ? 0u : 0xffffffffu;
        }
        for (m1_u32 offset = 16; offset > 0; offset >>= 1) {
          for (m1_u32 lane = 0; lane < offset; ++lane) {
            if (tag == 0x30) {
              lanes[lane] += lanes[lane + offset];
            } else if (in_desc.dtype == 1) {
              const int left = m1_bits_i32(lanes[lane]);
              const int right = m1_bits_i32(lanes[lane + offset]);
              lanes[lane] =
                  (m1_u32)(tag == 0x31
                               ? (left > right ? left : right)
                               : (left < right ? left : right));
            } else {
              const m1_u32 left = lanes[lane];
              const m1_u32 right = lanes[lane + offset];
              lanes[lane] =
                  tag == 0x31
                      ? (left > right ? left : right)
                      : (left < right ? left : right);
            }
          }
        }
        work[chunk] = lanes[0];
      }
      count = chunks;
    }
    if (in_desc.dtype == 1)
      reinterpret_cast<int*>(output)[row] = m1_bits_i32(work[0]);
    else
      reinterpret_cast<m1_u32*>(output)[row] = work[0];
  }
}

__device__ __forceinline__ void m1_reduce_argmax(
    const m1_u8* input,
    m1_u8* output,
    m1_u8* temporary,
    M1ValueDesc in_desc) {
  int* result = reinterpret_cast<int*>(output);
  if (in_desc.dtype != 0) {
    M1IntArgmaxCandidate* work =
        reinterpret_cast<M1IntArgmaxCandidate*>(temporary);
    for (m1_u32 row = 0; row < in_desc.rows; ++row) {
      const m1_u32 base = row * in_desc.last;
      for (m1_u32 i = 0; i < in_desc.last; ++i)
        work[i] = {m1_load_index(input, base + i, in_desc.dtype), i, 1u};
      m1_u32 count = in_desc.last;
      if (count == 0) {
        result[row] = 0;
        continue;
      }
      while (count > 1) {
        const m1_u32 chunks = (count + 31u) / 32u;
        for (m1_u32 chunk = 0; chunk < chunks; ++chunk) {
          M1IntArgmaxCandidate lanes[32];
          for (m1_u32 lane = 0; lane < 32; ++lane) {
            const m1_u32 index = chunk * 32u + lane;
            lanes[lane] =
                index < count
                    ? work[index]
                    : M1IntArgmaxCandidate{0ll, 0u, 0u};
          }
          for (m1_u32 offset = 16; offset > 0; offset >>= 1)
            for (m1_u32 lane = 0; lane < offset; ++lane)
              lanes[lane] =
                  m1_int_argmax_combine(lanes[lane], lanes[lane + offset]);
          work[chunk] = lanes[0];
        }
        count = chunks;
      }
      result[row] = (int)work[0].index;
    }
    return;
  }
  const float* values = reinterpret_cast<const float*>(input);
  M1ArgmaxCandidate* work =
      reinterpret_cast<M1ArgmaxCandidate*>(temporary);
  for (m1_u32 row = 0; row < in_desc.rows; ++row) {
    const m1_u32 base = row * in_desc.last;
    for (m1_u32 i = 0; i < in_desc.last; ++i) {
      const float value = values[base + i];
      work[i] = {value, i, m1_isnan(value) ? 0u : 1u, 0u};
    }
    m1_u32 count = in_desc.last;
    if (count == 0) {
      result[row] = 0;
      continue;
    }
    while (count > 1) {
      const m1_u32 chunks = (count + 31u) / 32u;
      for (m1_u32 chunk = 0; chunk < chunks; ++chunk) {
        M1ArgmaxCandidate lanes[32];
        for (m1_u32 lane = 0; lane < 32; ++lane) {
          const m1_u32 index = chunk * 32u + lane;
          lanes[lane] =
              index < count
                  ? work[index]
                  : M1ArgmaxCandidate{m1_neg_inf(), 0u, 0u, 0u};
        }
        for (m1_u32 offset = 16; offset > 0; offset >>= 1)
          for (m1_u32 lane = 0; lane < offset; ++lane)
            lanes[lane] =
                m1_argmax_combine(lanes[lane], lanes[lane + offset]);
        work[chunk] = lanes[0];
      }
      count = chunks;
    }
    result[row] = (int)work[0].index;
  }
}
