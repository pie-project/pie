
typedef unsigned char m1_u8;
typedef unsigned short m1_u16;
typedef unsigned int m1_u32;
typedef unsigned long long m1_u64;

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

// A descriptor's extent at a runtime index, read by selects: an array read
// at a runtime index is kept in private memory, which on AMDGPU is scratch
// the HIP runtime allocates at the first dispatch that needs it. Past the
// rank a descriptor has no extent; 1 reads as one.
__device__ __forceinline__ m1_u32 m1_dim(const M1ValueDesc& desc, m1_u32 index) {
  return index == 0u ? desc.dims[0]
       : index == 1u ? desc.dims[1]
       : index == 2u ? desc.dims[2]
       : index == 3u ? desc.dims[3]
       : 1u;
}

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
  // The row an intrinsic's reads stop short of: rows past the lane's own
  // are another lane's, or past the plane.
  m1_u32 intrinsic_rows;
  m1_u32 reserved;
  m1_u64 rng_seed;
};

static_assert(sizeof(M1Status) == 16, "M1Status ABI");
static_assert(sizeof(M1ValueDesc) == 36, "M1ValueDesc ABI");
static_assert(sizeof(M1OpParams) == 96, "M1OpParams ABI");

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

__device__ __forceinline__ m1_u32 m1_desc_key(float value) {
  if (m1_isnan(value)) return 0xFFFFFFFFu;
  if (value == 0.0f) value = 0.0f;
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

// bf16 (eta_ir::types::WIRE_BF16, dtype 5): two bytes an element, the upper
// half of an f32's bits. Every op computes in f32: a bf16 operand is widened
// exactly, and a bf16 result is rounded to nearest even once, on its store
// (`m1_store_h`, eta_ir::types::bf16_bits). Its store is one 16-bit store of
// the element's own two bytes (st.global.b16 / global_store_short), never a
// read-modify-write of the word around it: global memory is byte-addressed,
// so two threads storing the two halves of a word store two locations, as
// two threads storing two bytes of a bool value already do. Every op hands
// each element to one thread, so no two threads store one element.
#define M1_DT_BF16 5u

__device__ __forceinline__ bool m1_is_float(m1_u32 dtype) {
  return dtype == 0u || dtype == M1_DT_BF16;
}

// The bytes an element of `dtype` takes in scratch and in a channel cell.
__device__ __forceinline__ m1_u32 m1_elem_bytes(m1_u32 dtype) {
  return dtype == 3u ? 1u : (dtype == M1_DT_BF16 ? 2u : 4u);
}

__device__ __forceinline__ float m1_bf16_widen(m1_u32 bits) {
  return __uint_as_float(bits << 16);
}

// `x` rounded to bf16, round-to-nearest-even, as its 16 bits; a NaN stays a
// quiet NaN of its sign (eta_ir::types::bf16_bits).
__device__ __forceinline__ m1_u32 m1_bf16_bits(float x) {
  const m1_u32 b = __float_as_uint(x);
  if ((b & 0x7fffffffu) > 0x7f800000u) return ((b >> 16) | 0x40u) & 0xffffu;
  return (b + 0x7fffu + ((b >> 16) & 1u)) >> 16;
}

// `x` as a bf16 store keeps it, widened back: what a stream holds in a
// register for a bf16 result, so its readers see the stored value.
__device__ __forceinline__ float m1_round_bf16(float x) {
  return m1_bf16_widen(m1_bf16_bits(x));
}

__device__ __forceinline__ float m1_load_h(const m1_u8* data, m1_u32 index) {
  return m1_bf16_widen(reinterpret_cast<const m1_u16*>(data)[index]);
}

__device__ __forceinline__ float m1_load_f(
    const m1_u8* data, m1_u32 index, m1_u32 dtype) {
  if (dtype == 0) return reinterpret_cast<const float*>(data)[index];
  if (dtype == M1_DT_BF16) return m1_load_h(data, index);
  if (dtype == 1) return float(reinterpret_cast<const int*>(data)[index]);
  if (dtype == 2) return float(reinterpret_cast<const m1_u32*>(data)[index]);
  return data[index] != 0 ? 1.0f : 0.0f;
}

__device__ __forceinline__ int m1_load_i(
    const m1_u8* data, m1_u32 index, m1_u32 dtype) {
  if (dtype == 0)
    return m1_float_to_i32(reinterpret_cast<const float*>(data)[index]);
  if (dtype == M1_DT_BF16) return m1_float_to_i32(m1_load_h(data, index));
  if (dtype == 1) return reinterpret_cast<const int*>(data)[index];
  if (dtype == 2)
    return m1_bits_i32(reinterpret_cast<const m1_u32*>(data)[index]);
  return data[index] != 0 ? 1 : 0;
}

__device__ __forceinline__ m1_u32 m1_load_u(
    const m1_u8* data, m1_u32 index, m1_u32 dtype) {
  if (dtype == 0)
    return m1_float_to_u32(reinterpret_cast<const float*>(data)[index]);
  if (dtype == M1_DT_BF16) return m1_float_to_u32(m1_load_h(data, index));
  if (dtype == 1)
    return (m1_u32)reinterpret_cast<const int*>(data)[index];
  if (dtype == 2) return reinterpret_cast<const m1_u32*>(data)[index];
  return data[index] != 0 ? 1u : 0u;
}

__device__ __forceinline__ bool m1_load_b(
    const m1_u8* data, m1_u32 index, m1_u32 dtype) {
  if (dtype == 0)
    return reinterpret_cast<const float*>(data)[index] != 0.0f;
  if (dtype == M1_DT_BF16) return m1_load_h(data, index) != 0.0f;
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

// The one bf16 store: `value` rounded once, the element's two bytes alone.
__device__ __forceinline__ void m1_store_h(
    m1_u8* data, m1_u32 index, float value) {
  reinterpret_cast<m1_u16*>(data)[index] = (m1_u16)m1_bf16_bits(value);
}

// A float result stored at its value's float dtype: f32 as it is, bf16
// rounded once.
__device__ __forceinline__ void m1_store_fd(
    m1_u8* data, m1_u32 index, float value, m1_u32 dtype) {
  if (dtype == M1_DT_BF16)
    m1_store_h(data, index, value);
  else
    m1_store_f(data, index, value);
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
  if (dtype == M1_DT_BF16) return (long long)m1_float_to_i32(m1_load_h(data, index));
  return (long long)m1_float_to_i32(
      reinterpret_cast<const float*>(data)[index]);
}

// A scatter into a bf16 base (`output`, already holding the base): each
// cell an update names is summed (or set) over every update to it in f32,
// in update order, and rounded once. One thread runs it.
__device__ __forceinline__ void m1_scatter_bf16(
    bool is_add,
    const m1_u8* indices,
    const m1_u8* updates,
    m1_u8* output,
    m1_u32 index_count,
    m1_u32 index_dtype,
    m1_u32 update_dtype,
    m1_u32 rows,
    m1_u32 rest,
    bool scalar) {
  for (m1_u32 k = 0; k < index_count; ++k) {
    const long long row = m1_load_index(indices, k, index_dtype);
    if (row < 0 || (m1_u64)row >= rows) continue;
    bool seen = false;
    for (m1_u32 prior = 0; prior < k && !seen; ++prior)
      seen = m1_load_index(indices, prior, index_dtype) == row;
    if (seen) continue;
    for (m1_u32 r = 0; r < rest; ++r) {
      const m1_u32 destination = (m1_u32)row * rest + r;
      float acc = m1_load_h(output, destination);
      for (m1_u32 later = k; later < index_count; ++later) {
        if (m1_load_index(indices, later, index_dtype) != row) continue;
        const float value =
            m1_load_f(updates, scalar ? 0u : later * rest + r, update_dtype);
        acc = is_add ? acc + value : value;
      }
      m1_store_h(output, destination, acc);
    }
  }
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
  const m1_u32 bytes = len * m1_elem_bytes(dtype);
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

// An intrinsic's storage modes: 0 f32 rows `stride` apart, 1 bf16 rows, 2 a
// table of pointers to bf16 rows, 3 i32 rows, 4 a table of pointers to f32
// rows (a row a lane, wherever each lies).
__device__ __forceinline__ bool m1_intrinsic_row_table(m1_u32 mode) {
  return mode == 2u || mode == 4u;
}

// A row at `rows` or past it is not the lane's: it reads as zero, and
// nothing behind it is touched.
__device__ __forceinline__ float m1_intrinsic_row_load(
    const m1_u8* input,
    m1_u64 row,
    m1_u32 column,
    m1_u32 stride,
    m1_u32 mode,
    m1_u32 rows) {
  if (row >= (m1_u64)rows) return 0.0f;
  if (!m1_intrinsic_row_table(mode))
    return m1_intrinsic_load(
        input, row * (m1_u64)stride + column, mode);
  const m1_u64 row_address =
      reinterpret_cast<const m1_u64*>(input)[row];
  if (mode == 4u) return reinterpret_cast<const float*>(row_address)[column];
  const m1_u16 value =
      reinterpret_cast<const m1_u16*>(row_address)[column];
  return __uint_as_float((m1_u32)value << 16);
}

__device__ __forceinline__ bool m1_aligned16(const void* p) {
  return (reinterpret_cast<m1_u64>(p) & 15u) == 0u;
}

// Four bf16 elements from `index4 * 4`, eight bytes on an eight-byte
// boundary, widened.
__device__ __forceinline__ float4 m1_load4_h(const m1_u8* data, m1_u32 index4) {
  const uint2 raw = reinterpret_cast<const uint2*>(data)[index4];
  return make_float4(__uint_as_float(raw.x << 16), __uint_as_float(raw.x & 0xffff0000u),
                     __uint_as_float(raw.y << 16), __uint_as_float(raw.y & 0xffff0000u));
}

__device__ __forceinline__ float4 m1_load4_f(
    const m1_u8* data, m1_u32 index4, m1_u32 dtype) {
  if (dtype == 0) return reinterpret_cast<const float4*>(data)[index4];
  if (dtype == M1_DT_BF16) return m1_load4_h(data, index4);
  if (dtype == 1) {
    const int4 r = reinterpret_cast<const int4*>(data)[index4];
    return make_float4(float(r.x), float(r.y), float(r.z), float(r.w));
  }
  if (dtype == 2) {
    const uint4 r = reinterpret_cast<const uint4*>(data)[index4];
    return make_float4(float(r.x), float(r.y), float(r.z), float(r.w));
  }
  const uchar4 r = reinterpret_cast<const uchar4*>(data)[index4];
  return make_float4(r.x != 0 ? 1.0f : 0.0f, r.y != 0 ? 1.0f : 0.0f,
                     r.z != 0 ? 1.0f : 0.0f, r.w != 0 ? 1.0f : 0.0f);
}

__device__ __forceinline__ int4 m1_load4_i(
    const m1_u8* data, m1_u32 index4, m1_u32 dtype) {
  if (dtype == 0) {
    const float4 r = reinterpret_cast<const float4*>(data)[index4];
    return make_int4(m1_float_to_i32(r.x), m1_float_to_i32(r.y),
                     m1_float_to_i32(r.z), m1_float_to_i32(r.w));
  }
  if (dtype == M1_DT_BF16) {
    const float4 r = m1_load4_h(data, index4);
    return make_int4(m1_float_to_i32(r.x), m1_float_to_i32(r.y),
                     m1_float_to_i32(r.z), m1_float_to_i32(r.w));
  }
  if (dtype == 1) return reinterpret_cast<const int4*>(data)[index4];
  if (dtype == 2) {
    const uint4 r = reinterpret_cast<const uint4*>(data)[index4];
    return make_int4(m1_bits_i32(r.x), m1_bits_i32(r.y), m1_bits_i32(r.z), m1_bits_i32(r.w));
  }
  const uchar4 r = reinterpret_cast<const uchar4*>(data)[index4];
  return make_int4(r.x != 0 ? 1 : 0, r.y != 0 ? 1 : 0, r.z != 0 ? 1 : 0, r.w != 0 ? 1 : 0);
}

__device__ __forceinline__ uint4 m1_load4_u(
    const m1_u8* data, m1_u32 index4, m1_u32 dtype) {
  if (dtype == 0) {
    const float4 r = reinterpret_cast<const float4*>(data)[index4];
    return make_uint4(m1_float_to_u32(r.x), m1_float_to_u32(r.y),
                      m1_float_to_u32(r.z), m1_float_to_u32(r.w));
  }
  if (dtype == M1_DT_BF16) {
    const float4 r = m1_load4_h(data, index4);
    return make_uint4(m1_float_to_u32(r.x), m1_float_to_u32(r.y),
                      m1_float_to_u32(r.z), m1_float_to_u32(r.w));
  }
  if (dtype == 1) {
    const int4 r = reinterpret_cast<const int4*>(data)[index4];
    return make_uint4((m1_u32)r.x, (m1_u32)r.y, (m1_u32)r.z, (m1_u32)r.w);
  }
  if (dtype == 2) return reinterpret_cast<const uint4*>(data)[index4];
  const uchar4 r = reinterpret_cast<const uchar4*>(data)[index4];
  return make_uint4(r.x != 0 ? 1u : 0u, r.y != 0 ? 1u : 0u, r.z != 0 ? 1u : 0u, r.w != 0 ? 1u : 0u);
}

__device__ __forceinline__ uchar4 m1_load4_b(
    const m1_u8* data, m1_u32 index4, m1_u32 dtype) {
  if (dtype == 0) {
    const float4 r = reinterpret_cast<const float4*>(data)[index4];
    return make_uchar4(r.x != 0.0f, r.y != 0.0f, r.z != 0.0f, r.w != 0.0f);
  }
  if (dtype == M1_DT_BF16) {
    const float4 r = m1_load4_h(data, index4);
    return make_uchar4(r.x != 0.0f, r.y != 0.0f, r.z != 0.0f, r.w != 0.0f);
  }
  if (dtype == 1) {
    const int4 r = reinterpret_cast<const int4*>(data)[index4];
    return make_uchar4(r.x != 0, r.y != 0, r.z != 0, r.w != 0);
  }
  if (dtype == 2) {
    const uint4 r = reinterpret_cast<const uint4*>(data)[index4];
    return make_uchar4(r.x != 0u, r.y != 0u, r.z != 0u, r.w != 0u);
  }
  return reinterpret_cast<const uchar4*>(data)[index4];
}

__device__ __forceinline__ float m1_pick4_f(float4 v, m1_u32 j) {
  return j == 0u ? v.x : j == 1u ? v.y : j == 2u ? v.z : v.w;
}
__device__ __forceinline__ int m1_pick4_i(int4 v, m1_u32 j) {
  return j == 0u ? v.x : j == 1u ? v.y : j == 2u ? v.z : v.w;
}
__device__ __forceinline__ m1_u32 m1_pick4_u(uint4 v, m1_u32 j) {
  return j == 0u ? v.x : j == 1u ? v.y : j == 2u ? v.z : v.w;
}
__device__ __forceinline__ bool m1_pick4_b(uchar4 v, m1_u32 j) {
  return (j == 0u ? v.x : j == 1u ? v.y : j == 2u ? v.z : v.w) != 0;
}
__device__ __forceinline__ void m1_set4_f(float4& v, m1_u32 j, float x) {
  if (j == 0u) v.x = x; else if (j == 1u) v.y = x; else if (j == 2u) v.z = x; else v.w = x;
}
__device__ __forceinline__ void m1_set4_i(int4& v, m1_u32 j, int x) {
  if (j == 0u) v.x = x; else if (j == 1u) v.y = x; else if (j == 2u) v.z = x; else v.w = x;
}
__device__ __forceinline__ void m1_set4_u(uint4& v, m1_u32 j, m1_u32 x) {
  if (j == 0u) v.x = x; else if (j == 1u) v.y = x; else if (j == 2u) v.z = x; else v.w = x;
}
__device__ __forceinline__ void m1_set4_b(uchar4& v, m1_u32 j, bool x) {
  const unsigned char b = x ? 1 : 0;
  if (j == 0u) v.x = b; else if (j == 1u) v.y = b; else if (j == 2u) v.z = b; else v.w = b;
}
__device__ __forceinline__ void m1_store4_f(m1_u8* data, m1_u32 index4, float4 v) {
  reinterpret_cast<float4*>(data)[index4] = v;
}
// Four bf16 elements from `index4 * 4`, each rounded once: eight bytes, the
// two whole words they fill, in one store.
__device__ __forceinline__ void m1_store4_h(m1_u8* data, m1_u32 index4, float4 v) {
  uint2 raw;
  raw.x = m1_bf16_bits(v.x) | (m1_bf16_bits(v.y) << 16);
  raw.y = m1_bf16_bits(v.z) | (m1_bf16_bits(v.w) << 16);
  reinterpret_cast<uint2*>(data)[index4] = raw;
}
__device__ __forceinline__ void m1_store4_i(m1_u8* data, m1_u32 index4, int4 v) {
  reinterpret_cast<int4*>(data)[index4] = v;
}
__device__ __forceinline__ void m1_store4_u(m1_u8* data, m1_u32 index4, uint4 v) {
  reinterpret_cast<uint4*>(data)[index4] = v;
}
__device__ __forceinline__ void m1_store4_b(m1_u8* data, m1_u32 index4, uchar4 v) {
  reinterpret_cast<uchar4*>(data)[index4] = v;
}

__device__ __forceinline__ float4 m1_intrinsic_row_load4(
    const m1_u8* input, m1_u64 row, m1_u32 column, m1_u32 stride, m1_u32 mode,
    m1_u32 rows) {
  if (row >= (m1_u64)rows) return make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  if (mode == 0u) {
    const m1_u64 index = row * (m1_u64)stride + column;
    return reinterpret_cast<const float4*>(input + index * 4u)[0];
  }
  if (mode == 4u) {
    const m1_u8* row_base = reinterpret_cast<const m1_u8*>(reinterpret_cast<const m1_u64*>(input)[row]);
    return reinterpret_cast<const float4*>(row_base + (m1_u64)column * 4u)[0];
  }
  const m1_u8* base = input;
  m1_u64 index = row * (m1_u64)stride + column;
  if (mode == 2u) {
    base = reinterpret_cast<const m1_u8*>(reinterpret_cast<const m1_u64*>(input)[row]);
    index = column;
  }
  const uint2 raw = reinterpret_cast<const uint2*>(base + index * 2u)[0];
  return make_float4(__uint_as_float(raw.x << 16), __uint_as_float(raw.x & 0xffff0000u),
                     __uint_as_float(raw.y << 16), __uint_as_float(raw.y & 0xffff0000u));
}

__device__ __forceinline__ bool m1_intrinsic_row_vectorable(
    const m1_u8* input, m1_u64 row, m1_u32 stride, m1_u32 mode, m1_u32 rows) {
  if ((stride & 3u) != 0u || row >= (m1_u64)rows) return false;
  if (m1_intrinsic_row_table(mode)) {
    const m1_u64 address = reinterpret_cast<const m1_u64*>(input)[row];
    return (address & (mode == 4u ? 15u : 7u)) == 0u;
  }
  const m1_u64 element = mode == 0u ? 4u : 2u;
  const m1_u64 address = reinterpret_cast<m1_u64>(input) + row * (m1_u64)stride * element;
  return (address & (mode == 0u ? 15u : 7u)) == 0u;
}

__device__ __forceinline__ const m1_u8* m1_intrinsic_row_base(
    const m1_u8* input, m1_u64 row, m1_u32 stride, m1_u32 mode) {
  if (m1_intrinsic_row_table(mode)) {
    return reinterpret_cast<const m1_u8*>(
        reinterpret_cast<const m1_u64*>(input)[row]);
  }
  const m1_u64 element = mode == 0u ? 4u : 2u;
  return input + row * (m1_u64)stride * element;
}

__device__ __forceinline__ float m1_intrinsic_column_load(
    const m1_u8* row_base, m1_u32 column, m1_u32 mode) {
  if (mode == 0u || mode == 4u) return reinterpret_cast<const float*>(row_base)[column];
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
  float* result = reinterpret_cast<float*>(output);
  for (m1_u32 row = 0; row < in_desc.rows; ++row) {
    const m1_u32 base = row * in_desc.last;
    for (m1_u32 i = 0; i < in_desc.last; ++i)
      work[i] = m1_load_f(input, base + i, in_desc.dtype);
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
  if (!m1_is_float(in_desc.dtype)) {
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
  M1ArgmaxCandidate* work =
      reinterpret_cast<M1ArgmaxCandidate*>(temporary);
  for (m1_u32 row = 0; row < in_desc.rows; ++row) {
    const m1_u32 base = row * in_desc.last;
    for (m1_u32 i = 0; i < in_desc.last; ++i) {
      const float value = m1_load_f(input, base + i, in_desc.dtype);
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
