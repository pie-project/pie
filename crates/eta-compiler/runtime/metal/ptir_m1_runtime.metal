#include <metal_stdlib>
#include "ptir_rng.generated.metal"
using namespace metal;

struct M1Status {
  uint state;
  uint fault;
  uint reserved0;
  uint reserved1;
};

struct M1ValueDesc {
  uint len;
  uint rows;
  uint last;
  uint rank;
  uint dtype;
  uint dims[4];
};

struct M1OpParams {
  uint tag;
  uint a0;
  uint a1;
  uint a2;
  uint o0;
  uint o1;
  uint imm;
  uint imm2;
  uint imm3;
  uint kind;
  uint pred_tag;
  uint lit_dtype;
  uint lit_bits;
  uint channel_slot;
  uint intr;
  uint sink_bytes;
};

// An output's slot: its lane's first element, how it is stored
// (0 f32, 1 bf16, 2 i32; 3 bf16 and 4 f32 rows through a table of row
// addresses at `at`, a lane-0 read's rows seated at the instance's lanes)
// its row pitch in elements (0: its width) and the rows from `at` that are
// its lane's own (a read past them is another request's rows: refused).
struct M1OutputSeat {
  ulong at;
  uint storage;
  uint stride;
  uint rows;
  uint reserved;
};

struct M1ArgmaxCandidate {
  float value;
  uint index;
  uint have;
  uint reserved;
};

struct M1IntArgmaxCandidate {
  long value;
  uint index;
  uint have;
};

// A bf16 element (wire tag 5): two bytes, the upper half of an f32's bits,
// widened exactly.
inline float m1_load_bf16(const device uchar* data, uint index) {
  return as_type<float>(uint(reinterpret_cast<const device ushort*>(data)[index]) << 16);
}

// `value` rounded to bf16, round-to-nearest-even, as its 16 bits: eta-ir's
// `bf16_bits`, bit for bit (a NaN stays a quiet NaN of its sign; past the
// largest bf16 rounds to infinity).
inline ushort m1_bf16_bits(float value) {
  const uint bits = as_type<uint>(value);
  if ((bits & 0x7fffffffu) > 0x7f800000u) return ushort((bits >> 16) | 0x0040u);
  return ushort((bits + 0x7fffu + ((bits >> 16) & 1u)) >> 16);
}

inline float m1_load_f(const device uchar* data, uint index, uint dtype) {
  if (dtype == 5) return m1_load_bf16(data, index);
  if (dtype == 0) return reinterpret_cast<const device float*>(data)[index];
  if (dtype == 1) return float(reinterpret_cast<const device int*>(data)[index]);
  if (dtype == 2) return float(reinterpret_cast<const device uint*>(data)[index]);
  return data[index] != 0 ? 1.0f : 0.0f;
}

// A 32-bit element's bits, already loaded, as m1_load_f reads it.
inline float m1_bits_f(uint bits, uint dtype) {
  if (dtype == 0) return as_type<float>(bits);
  if (dtype == 1) return float(as_type<int>(bits));
  if (dtype == 2) return float(bits);
  return bits != 0 ? 1.0f : 0.0f;
}

inline int m1_load_i(const device uchar* data, uint index, uint dtype) {
  if (dtype == 5) return int(m1_load_bf16(data, index));
  if (dtype == 0) return int(reinterpret_cast<const device float*>(data)[index]);
  if (dtype == 1) return reinterpret_cast<const device int*>(data)[index];
  if (dtype == 2) return int(reinterpret_cast<const device uint*>(data)[index]);
  return data[index] != 0 ? 1 : 0;
}

inline uint m1_load_u(const device uchar* data, uint index, uint dtype) {
  if (dtype == 5) return uint(m1_load_bf16(data, index));
  if (dtype == 0) return uint(reinterpret_cast<const device float*>(data)[index]);
  if (dtype == 1) return uint(reinterpret_cast<const device int*>(data)[index]);
  if (dtype == 2) return reinterpret_cast<const device uint*>(data)[index];
  return data[index] != 0 ? 1u : 0u;
}

inline bool m1_load_b(const device uchar* data, uint index, uint dtype) {
  if (dtype == 5) return m1_load_bf16(data, index) != 0.0f;
  if (dtype == 0) return reinterpret_cast<const device float*>(data)[index] != 0.0f;
  if (dtype == 1) return reinterpret_cast<const device int*>(data)[index] != 0;
  if (dtype == 2) return reinterpret_cast<const device uint*>(data)[index] != 0u;
  return data[index] != 0;
}

inline void m1_store_f(device uchar* data, uint index, float value) {
  reinterpret_cast<device float*>(data)[index] = value;
}
// A float result stored at its value's dtype: a bf16 value rounds once.
inline void m1_store_fd(device uchar* data, uint index, float value, uint dtype) {
  if (dtype == 5) {
    reinterpret_cast<device ushort*>(data)[index] = m1_bf16_bits(value);
  } else {
    reinterpret_cast<device float*>(data)[index] = value;
  }
}

// `value` as a bf16 store keeps it, widened back: what a register holding
// a bf16 value holds.
inline float m1_round_bf16(float value) {
  return as_type<float>(uint(m1_bf16_bits(value)) << 16);
}

inline void m1_store_h(device uchar* data, uint index, float value) {
  reinterpret_cast<device ushort*>(data)[index] = m1_bf16_bits(value);
}

// Whether a value of `dtype` is a float (f32, or bf16 computed as one).
inline bool m1_is_float(uint dtype) {
  return dtype == 0 || dtype == 5;
}

// A value's bytes: a bool's bits packed, a bf16's two bytes an element,
// every other dtype's four.
inline uint m1_value_bytes(uint len, uint dtype) {
  if (dtype == 3) return (len + 7u) / 8u;
  return len * (dtype == 5 ? 2u : 4u);
}

inline void m1_store_i(device uchar* data, uint index, int value) {
  reinterpret_cast<device int*>(data)[index] = value;
}
inline void m1_store_u(device uchar* data, uint index, uint value) {
  reinterpret_cast<device uint*>(data)[index] = value;
}
inline void m1_store_b(device uchar* data, uint index, bool value) {
  data[index] = value ? 1 : 0;
}

inline float m1_canonical_max(float left, float right) {
  const bool ln = isnan(left), rn = isnan(right);
  if (ln && rn) return -INFINITY;
  if (ln) return right;
  if (rn) return left;
  if (left == 0.0f && right == 0.0f)
    return signbit(left) && signbit(right) ? -0.0f : 0.0f;
  return max(left, right);
}

inline float m1_canonical_min(float left, float right) {
  const bool ln = isnan(left), rn = isnan(right);
  if (ln && rn) return INFINITY;
  if (ln) return right;
  if (rn) return left;
  if (left == 0.0f && right == 0.0f)
    return signbit(left) || signbit(right) ? -0.0f : 0.0f;
  return min(left, right);
}

inline float m1_element_max(float left, float right) {
  const bool ln = isnan(left), rn = isnan(right);
  if (ln && rn) return left;
  if (ln) return right;
  if (rn) return left;
  if (left == 0.0f && right == 0.0f)
    return signbit(left) && signbit(right) ? -0.0f : 0.0f;
  return max(left, right);
}

inline float m1_element_min(float left, float right) {
  const bool ln = isnan(left), rn = isnan(right);
  if (ln && rn) return left;
  if (ln) return right;
  if (rn) return left;
  if (left == 0.0f && right == 0.0f)
    return signbit(left) || signbit(right) ? -0.0f : 0.0f;
  return min(left, right);
}

inline long m1_load_index(const device uchar* data, uint index, uint dtype) {
  if (dtype == 1)
    return long(reinterpret_cast<const device int*>(data)[index]);
  if (dtype == 2)
    return long(reinterpret_cast<const device uint*>(data)[index]);
  if (dtype == 3) return data[index] != 0 ? 1l : 0l;
  return long(reinterpret_cast<const device float*>(data)[index]);
}

inline M1ArgmaxCandidate m1_argmax_combine(
    M1ArgmaxCandidate left,
    M1ArgmaxCandidate right) {
  if (right.have == 0) return left;
  if (left.have == 0 || right.value > left.value ||
      (right.value == left.value && right.index < left.index)) {
    return right;
  }
  return left;
}

inline M1IntArgmaxCandidate m1_int_argmax_combine(
    M1IntArgmaxCandidate left,
    M1IntArgmaxCandidate right) {
  if (right.have == 0) return left;
  if (left.have == 0 || right.value > left.value ||
      (right.value == left.value && right.index < left.index)) {
    return right;
  }
  return left;
}

inline bool m1_sort_better(float value, uint index, float best, uint best_index) {
  const bool value_nan = isnan(value), best_nan = isnan(best);
  if (value_nan != best_nan) return best_nan;
  if (value_nan) return index < best_index;
  if (value != best) return value > best;
  return index < best_index;
}

inline uint m1_desc_key(float value) {
  if (isnan(value)) return 0xFFFFFFFFu;
  if (value == 0.0f) value = 0.0f;
  const uint bits = as_type<uint>(value);
  const uint ascending = (bits & 0x80000000u) != 0u ? ~bits : (bits | 0x80000000u);
  return ~ascending;
}

// A probability's fixed-point mass for the top-p cut: floor(p * 2^32).
#define M4_SEL_SCALE 4294967296.0f
inline ulong m4_sel_fixed(float p) {
  if (!(p > 0.0f)) return 0ul;
  return ulong(p * M4_SEL_SCALE);
}

inline ulong m4_sel_pair(uint lo, uint hi) { return (ulong(hi) << 32) | ulong(lo); }

// The value a descending key stands for (the inverse of m1_desc_key).
inline float m4_sel_value(uint key) {
  const uint ascending = ~key;
  const uint bits = (ascending & 0x80000000u) != 0u ? (ascending & 0x7FFFFFFFu) : ~ascending;
  return as_type<float>(bits);
}
// The fixed-point mass of the cut key's value, low and high words.
inline uint m4_sel_fixed_lo(uint digit, uint key) {
  (void)digit;
  return uint(m4_sel_fixed(m4_sel_value(key)));
}
inline uint m4_sel_fixed_hi(uint key) { return uint(m4_sel_fixed(m4_sel_value(key)) >> 32); }

// Two-word (64-bit) mass arithmetic in 32-bit halves. The pick's walk keeps
// its sums this way: the AGX compiler on Apple7 fails building a loop that
// adds and compares `ulong`s with an early exit.
struct M4Mass {
  uint lo;
  uint hi;
};

inline M4Mass m4_mass(uint lo, uint hi) { return M4Mass{lo, hi}; }

inline M4Mass m4_mass_add(M4Mass a, M4Mass b) {
  const uint lo = a.lo + b.lo;
  return M4Mass{lo, a.hi + b.hi + (lo < a.lo ? 1u : 0u)};
}

inline M4Mass m4_mass_sub(M4Mass a, M4Mass b) {
  return M4Mass{a.lo - b.lo, a.hi - b.hi - (a.lo < b.lo ? 1u : 0u)};
}

inline bool m4_mass_ge(M4Mass a, M4Mass b) {
  return a.hi > b.hi || (a.hi == b.hi && a.lo >= b.lo);
}

// `v * j` for a token's mass `v` and a tie count `j`.
inline M4Mass m4_mass_mul(M4Mass v, uint j) {
  const uint lo = v.lo * j;
  return M4Mass{lo, mulhi(v.lo, j) + v.hi * j};
}

inline uint m1_pick(uint len, uint index) {
  return len == 1 ? 0u : index;
}

// One row's top-k (mode 0) / top-p (mode 1) cut on one thread: four radix
// passes over the descending key, by count and by fixed-point mass, then the
// mask, the cut key's ties in index order.
inline void m1_cut_select_row(
    const device uchar* a0, const device uchar* a1, device uchar* o0, const M1ValueDesc d0,
    const M1ValueDesc d1, uint mode, uint row) {
  const uint n = d0.last;
  const uint base = row * n;
  uint k = 0u;
  ulong target64 = 0ul;
  if (mode == 0u) {
    const int signed_k = m1_load_i(a1, m1_pick(d1.len, row), d1.dtype);
    k = signed_k <= 0 ? 0u : uint(signed_k);
  } else {
    target64 = m4_sel_fixed(m1_load_f(a1, m1_pick(d1.len, row), d1.dtype));
  }
  const M4Mass target = m4_mass(uint(target64), uint(target64 >> 32));
  bool all = mode == 0u && k >= n;
  bool none = !all && (n == 0u || (mode == 0u ? k == 0u : target64 == 0ul));
  uint prefix = 0u, before = 0u, ties = 0u, take = 0u;
  M4Mass mass = m4_mass(0u, 0u);
  for (uint pass = 0u; pass < 4u && !all && !none; ++pass) {
    const uint shift = 24u - 8u * pass;
    uint count[256];
    M4Mass bucket[256];
    for (uint b = 0u; b < 256u; ++b) { count[b] = 0u; bucket[b] = m4_mass(0u, 0u); }
    for (uint i = 0u; i < n; ++i) {
      const float value = m1_load_f(a0, base + i, d0.dtype);
      const uint key = m1_desc_key(value);
      if (pass != 0u && (key >> (shift + 8u)) != (prefix >> (shift + 8u))) continue;
      const uint digit = (key >> shift) & 255u;
      ++count[digit];
      const ulong v = m4_sel_fixed(value);
      bucket[digit] = m4_mass_add(bucket[digit], m4_mass(uint(v), uint(v >> 32)));
    }
    uint digit = 256u;
    for (uint d = 0u; d < 256u; ++d) {
      const bool here = count[d] != 0u &&
          (mode == 0u ? (before + count[d] >= k) : m4_mass_ge(m4_mass_add(mass, bucket[d]), target));
      if (digit == 256u && here) digit = d;
      if (digit == 256u) {
        before += count[d];
        mass = m4_mass_add(mass, bucket[d]);
      }
    }
    if (digit == 256u) {
      all = true;
      break;
    }
    prefix |= digit << shift;
    if (pass == 3u) {
      ties = count[digit];
      take = ties;
      if (mode == 0u) {
        take = k - before;
      } else {
        const M4Mass v = m4_mass(m4_sel_fixed_lo(digit, prefix), m4_sel_fixed_hi(prefix));
        if (v.lo != 0u || v.hi != 0u) {
          const M4Mass room = m4_mass_sub(target, mass);
          uint low = 0u, high = ties;
          while (low < high) {
            const uint mid = low + (high - low) / 2u;
            if (m4_mass_ge(m4_mass_mul(v, mid), room)) high = mid; else low = mid + 1u;
          }
          take = low;
        }
      }
      take = min(take, ties);
    }
  }
  uint tied = 0u;
  for (uint i = 0u; i < n; ++i) {
    bool taken = all;
    if (!all && !none) {
      const uint key = m1_desc_key(m1_load_f(a0, base + i, d0.dtype));
      taken = key < prefix;
      if (key == prefix) {
        taken = tied < take;
        ++tied;
      }
    }
    m1_store_b(o0, base + i, taken);
  }
}

inline void m1_fault(device M1Status* status, uint code) {
  status->fault = code;
  status->state = 3;
}

inline void m1_fault_op(device M1Status* status, uint site, M1OpParams p) {
  status->reserved0 = p.intr;
  status->reserved1 = (site << 24) | (p.imm & 0x00ffffffu);
  status->fault = p.tag;
  status->state = 3;
}

inline void m1_copy_typed_range(
    const device uchar* input,
    device uchar* output,
    uint len,
    uint dtype,
    uint begin,
    uint step) {
  if (dtype == 3) {
    for (uint i = begin; i < len; i += step) output[i] = input[i];
    return;
  }
  if (dtype == 5) {
    const device ushort* src = reinterpret_cast<const device ushort*>(input);
    device ushort* dst = reinterpret_cast<device ushort*>(output);
    for (uint i = begin; i < len; i += step) dst[i] = src[i];
    return;
  }
  const bool aligned =
      ((ulong(input) | ulong(output)) & 3ul) == 0ul;
  if (aligned) {
    const device uint* src = reinterpret_cast<const device uint*>(input);
    device uint* dst = reinterpret_cast<device uint*>(output);
    for (uint i = begin; i < len; i += step) dst[i] = src[i];
  } else {
    for (uint i = begin; i < len * 4u; i += step) output[i] = input[i];
  }
}

inline void m1_copy_typed(
    const device uchar* input,
    device uchar* output,
    uint len,
    uint dtype) {
  m1_copy_typed_range(input, output, len, dtype, 0u, 1u);
}

inline void m1_reduce_float(
    uint tag,
    const device uchar* input,
    device uchar* output,
    device uchar* temporary,
    const M1ValueDesc in_desc) {
  device float* work = reinterpret_cast<device float*>(temporary);
  device float* result = reinterpret_cast<device float*>(output);
  for (uint row = 0; row < in_desc.rows; ++row) {
    const uint base = row * in_desc.last;
    for (uint i = 0; i < in_desc.last; ++i) work[i] = m1_load_f(input, base + i, in_desc.dtype);
    uint count = in_desc.last;
    if (count == 0) {
      result[row] = tag == 0x30 ? 0.0f : (tag == 0x31 ? -INFINITY : INFINITY);
      continue;
    }
    while (count > 1) {
      const uint chunks = (count + 31) / 32;
      for (uint chunk = 0; chunk < chunks; ++chunk) {
        float lanes[32];
        const float identity =
            tag == 0x30 ? 0.0f : (tag == 0x31 ? -INFINITY : INFINITY);
        for (uint lane = 0; lane < 32; ++lane) {
          const uint index = chunk * 32 + lane;
          lanes[lane] = index < count ? work[index] : identity;
        }
        for (uint offset = 16; offset > 0; offset >>= 1) {
          for (uint lane = 0; lane < offset; ++lane) {
            if (tag == 0x30) lanes[lane] += lanes[lane + offset];
            else if (tag == 0x31)
              lanes[lane] = m1_canonical_max(lanes[lane], lanes[lane + offset]);
            else
              lanes[lane] = m1_canonical_min(lanes[lane], lanes[lane + offset]);
          }
        }
        work[chunk] = lanes[0];
      }
      count = chunks;
    }
    result[row] = work[0];
  }
}

inline void m1_reduce_integer(
    uint tag,
    const device uchar* input,
    device uchar* output,
    device uchar* temporary,
    const M1ValueDesc in_desc) {
  device uint* work = reinterpret_cast<device uint*>(temporary);
  for (uint row = 0; row < in_desc.rows; ++row) {
    const uint base = row * in_desc.last;
    for (uint i = 0; i < in_desc.last; ++i) {
      work[i] = in_desc.dtype == 1
                    ? uint(reinterpret_cast<const device int*>(input)[base + i])
                    : reinterpret_cast<const device uint*>(input)[base + i];
    }
    uint count = in_desc.last;
    if (count == 0) {
      if (in_desc.dtype == 1) {
        reinterpret_cast<device int*>(output)[row] =
            tag == 0x30 ? 0 : (tag == 0x31 ? INT_MIN : INT_MAX);
      } else {
        reinterpret_cast<device uint*>(output)[row] =
            tag == 0x32 ? UINT_MAX : 0u;
      }
      continue;
    }
    while (count > 1) {
      const uint chunks = (count + 31) / 32;
      for (uint chunk = 0; chunk < chunks; ++chunk) {
        uint lanes[32];
        for (uint lane = 0; lane < 32; ++lane) {
          const uint index = chunk * 32 + lane;
          if (index < count) lanes[lane] = work[index];
          else if (tag == 0x30) lanes[lane] = 0u;
          else if (in_desc.dtype == 1)
            lanes[lane] = tag == 0x31 ? uint(INT_MIN) : uint(INT_MAX);
          else
            lanes[lane] = tag == 0x31 ? 0u : UINT_MAX;
        }
        for (uint offset = 16; offset > 0; offset >>= 1) {
          for (uint lane = 0; lane < offset; ++lane) {
            if (tag == 0x30) lanes[lane] += lanes[lane + offset];
            else if (in_desc.dtype == 1) {
              const int left = int(lanes[lane]), right = int(lanes[lane + offset]);
              lanes[lane] = uint(tag == 0x31 ? max(left, right) : min(left, right));
            } else {
              lanes[lane] = tag == 0x31
                                ? max(lanes[lane], lanes[lane + offset])
                                : min(lanes[lane], lanes[lane + offset]);
            }
          }
        }
        work[chunk] = lanes[0];
      }
      count = chunks;
    }
    if (in_desc.dtype == 1)
      reinterpret_cast<device int*>(output)[row] = int(work[0]);
    else
      reinterpret_cast<device uint*>(output)[row] = work[0];
  }
}

inline void m1_reduce_argmax(
    const device uchar* input,
    device uchar* output,
    device uchar* temporary,
    const M1ValueDesc in_desc) {
  (void)temporary;
  device int* result = reinterpret_cast<device int*>(output);
  if (!m1_is_float(in_desc.dtype)) {
    for (uint row = 0; row < in_desc.rows; ++row) {
      const uint base = row * in_desc.last;
      M1IntArgmaxCandidate best = {0l, 0u, 0u};
      for (uint i = 0; i < in_desc.last; ++i) {
        best = m1_int_argmax_combine(
            best,
            M1IntArgmaxCandidate{
                m1_load_index(input, base + i, in_desc.dtype), i, 1u});
      }
      result[row] = int(best.index);
    }
    return;
  }
  for (uint row = 0; row < in_desc.rows; ++row) {
    const uint base = row * in_desc.last;

    M1ArgmaxCandidate best[4] = {
        {-INFINITY, 0u, 0u, 0u}, {-INFINITY, 0u, 0u, 0u},
        {-INFINITY, 0u, 0u, 0u}, {-INFINITY, 0u, 0u, 0u}};
    uint i = 0;
    for (; i + 4u <= in_desc.last; i += 4u) {
      const float v0 = m1_load_f(input, base + i + 0, in_desc.dtype), v1 = m1_load_f(input, base + i + 1, in_desc.dtype);
      const float v2 = m1_load_f(input, base + i + 2, in_desc.dtype), v3 = m1_load_f(input, base + i + 3, in_desc.dtype);
      best[0] = m1_argmax_combine(
          best[0], M1ArgmaxCandidate{v0, i + 0u, isnan(v0) ? 0u : 1u, 0u});
      best[1] = m1_argmax_combine(
          best[1], M1ArgmaxCandidate{v1, i + 1u, isnan(v1) ? 0u : 1u, 0u});
      best[2] = m1_argmax_combine(
          best[2], M1ArgmaxCandidate{v2, i + 2u, isnan(v2) ? 0u : 1u, 0u});
      best[3] = m1_argmax_combine(
          best[3], M1ArgmaxCandidate{v3, i + 3u, isnan(v3) ? 0u : 1u, 0u});
    }
    for (; i < in_desc.last; ++i) {
      const float value = m1_load_f(input, base + i, in_desc.dtype);
      best[0] = m1_argmax_combine(
          best[0], M1ArgmaxCandidate{value, i, isnan(value) ? 0u : 1u, 0u});
    }
    M1ArgmaxCandidate folded = m1_argmax_combine(
        m1_argmax_combine(best[0], best[1]),
        m1_argmax_combine(best[2], best[3]));
    result[row] = int(folded.index);
  }
}

inline void m1_reduce_argmax_mt(
    const device uchar* input,
    device uchar* output,
    device uchar* temporary,
    const M1ValueDesc in_desc,
    uint tid,
    uint nthreads,
    threadgroup M1ArgmaxCandidate* tgbuf) {
  if (in_desc.dtype != 0 && in_desc.dtype != 5) {

    if (tid == 0) m1_reduce_argmax(input, output, temporary, in_desc);
    return;
  }
  device int* result = reinterpret_cast<device int*>(output);
  for (uint row = 0; row < in_desc.rows; ++row) {
    const uint base = row * in_desc.last;
    M1ArgmaxCandidate best = {-INFINITY, 0u, 0u, 0u};
    for (uint i = tid; i < in_desc.last; i += nthreads) {
      const float value = m1_load_f(input, base + i, in_desc.dtype);
      best = m1_argmax_combine(
          best, M1ArgmaxCandidate{value, i, isnan(value) ? 0u : 1u, 0u});
    }
    tgbuf[tid] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1u; stride < nthreads; stride <<= 1) {
      if ((tid % (2u * stride)) == 0u && tid + stride < nthreads) {
        tgbuf[tid] = m1_argmax_combine(tgbuf[tid], tgbuf[tid + stride]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) result[row] = int(tgbuf[0].index);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

inline void m1_reduce_float_part(
    uint tag,
    const device uchar* input,
    device uchar* output,
    device uchar* temporary,
    const M1ValueDesc in_desc,
    uint tid,
    uint nthreads) {
  if (nthreads <= 1u) {
    m1_reduce_float(tag, input, output, temporary, in_desc);
    return;
  }
  const float identity =
      tag == 0x30 ? 0.0f : (tag == 0x31 ? -INFINITY : INFINITY);
  device float* plane_a = reinterpret_cast<device float*>(temporary);
  device float* plane_b = plane_a + in_desc.last;
  device float* result = reinterpret_cast<device float*>(output);
  for (uint row = 0; row < in_desc.rows; ++row) {
    const uint base = row * in_desc.last;
    uint count = in_desc.last;
    if (count == 0) {
      if (tid == 0) result[row] = identity;
      continue;
    }

    // The first round reads the row at its dtype, every later one a plane.
    bool first = true;
    const device float* src = plane_a;
    device float* dst = plane_a;
    while (count > 1) {
      const uint chunks = (count + 31) / 32;
      for (uint chunk = tid; chunk < chunks; chunk += nthreads) {
        float lanes[32];
        for (uint lane = 0; lane < 32; ++lane) {
          const uint index = chunk * 32 + lane;
          lanes[lane] = index >= count ? identity
              : (first ? m1_load_f(input, base + index, in_desc.dtype) : src[index]);
        }
        for (uint offset = 16; offset > 0; offset >>= 1) {
          for (uint lane = 0; lane < offset; ++lane) {
            if (tag == 0x30) lanes[lane] += lanes[lane + offset];
            else if (tag == 0x31)
              lanes[lane] = m1_canonical_max(lanes[lane], lanes[lane + offset]);
            else
              lanes[lane] = m1_canonical_min(lanes[lane], lanes[lane + offset]);
          }
        }
        dst[chunk] = lanes[0];
      }
      threadgroup_barrier(mem_flags::mem_device);
      count = chunks;
      first = false;
      src = dst;
      dst = (dst == plane_a) ? plane_b : plane_a;
    }
    if (tid == 0) result[row] = first ? m1_load_f(input, base, in_desc.dtype) : src[0];
    threadgroup_barrier(mem_flags::mem_device);
  }
}

inline void m1_reduce_integer_part(
    uint tag,
    const device uchar* input,
    device uchar* output,
    device uchar* temporary,
    const M1ValueDesc in_desc,
    uint tid,
    uint nthreads) {
  if (nthreads <= 1u) {
    m1_reduce_integer(tag, input, output, temporary, in_desc);
    return;
  }
  const bool is_signed = in_desc.dtype == 1;
  uint identity;
  if (tag == 0x30) identity = 0u;
  else if (is_signed) identity = tag == 0x31 ? uint(INT_MIN) : uint(INT_MAX);
  else identity = tag == 0x31 ? 0u : UINT_MAX;
  device uint* plane_a = reinterpret_cast<device uint*>(temporary);
  device uint* plane_b = plane_a + in_desc.last;
  const device uint* values = reinterpret_cast<const device uint*>(input);
  for (uint row = 0; row < in_desc.rows; ++row) {
    const uint base = row * in_desc.last;
    uint count = in_desc.last;
    if (count == 0) {
      if (tid == 0) {
        if (is_signed) {
          reinterpret_cast<device int*>(output)[row] =
              tag == 0x30 ? 0 : (tag == 0x31 ? INT_MIN : INT_MAX);
        } else {
          reinterpret_cast<device uint*>(output)[row] =
              tag == 0x32 ? UINT_MAX : 0u;
        }
      }
      continue;
    }

    const device uint* src = values + base;
    device uint* dst = plane_a;
    while (count > 1) {
      const uint chunks = (count + 31) / 32;
      for (uint chunk = tid; chunk < chunks; chunk += nthreads) {
        uint lanes[32];
        for (uint lane = 0; lane < 32; ++lane) {
          const uint index = chunk * 32 + lane;
          lanes[lane] = index < count ? src[index] : identity;
        }
        for (uint offset = 16; offset > 0; offset >>= 1) {
          for (uint lane = 0; lane < offset; ++lane) {
            if (tag == 0x30) lanes[lane] += lanes[lane + offset];
            else if (is_signed) {
              const int left = int(lanes[lane]), right = int(lanes[lane + offset]);
              lanes[lane] = uint(tag == 0x31 ? max(left, right) : min(left, right));
            } else {
              lanes[lane] = tag == 0x31
                                ? max(lanes[lane], lanes[lane + offset])
                                : min(lanes[lane], lanes[lane + offset]);
            }
          }
        }
        dst[chunk] = lanes[0];
      }
      threadgroup_barrier(mem_flags::mem_device);
      count = chunks;
      src = dst;
      dst = (dst == plane_a) ? plane_b : plane_a;
    }
    if (tid == 0) {
      if (is_signed)
        reinterpret_cast<device int*>(output)[row] = int(src[0]);
      else
        reinterpret_cast<device uint*>(output)[row] = src[0];
    }
    threadgroup_barrier(mem_flags::mem_device);
  }
}

inline void m1_scatter_rmw(
    uint tag,
    const device uchar* a1,
    const device uchar* a2,
    device uchar* o0,
    const M1ValueDesc d0,
    const M1ValueDesc d1,
    const M1ValueDesc d2) {
  const uint rest = d0.rank == 0 ? 1u : d0.len / max(d0.dims[0], 1u);
  const uint n0 = d0.rank == 0 ? 1u : d0.dims[0];
  const bool scalar = d2.len == 1 && d1.len * rest != 1;
  for (uint k = 0; k < d1.len; ++k) {
    const long index = m1_load_index(a1, k, d1.dtype);
    if (index < 0 || uint(index) >= n0) continue;
    for (uint r = 0; r < rest; ++r) {
      const uint dst = uint(index) * rest + r;
      const uint src = scalar ? 0u : k * rest + r;
      if (d0.dtype == 0 || d0.dtype == 5) {
        const float value = m1_load_f(a2, src, d2.dtype);
        m1_store_fd(o0, dst, tag == 0x62 ? m1_load_f(o0, dst, d0.dtype) + value : value, d0.dtype);
      } else if (d0.dtype == 1) {
        const int value = m1_load_i(a2, src, d2.dtype);
        m1_store_i(o0, dst, tag == 0x62 ? int(uint(m1_load_i(o0, dst, 1)) + uint(value)) : value);
      } else if (d0.dtype == 2) {
        const uint value = m1_load_u(a2, src, d2.dtype);
        m1_store_u(o0, dst, tag == 0x62 ? m1_load_u(o0, dst, 2) + value : value);
      } else {
        const bool value = m1_load_b(a2, src, d2.dtype);
        m1_store_b(o0, dst, value);
      }
    }
  }
}

inline void ptir_m1_execute_part(
    uint generated_tag,
    device M1Status* status,
    const device M1ValueDesc* descriptors,
    const device M1OpParams* params,
    const device uchar* a0,
    const device uchar* a1,
    const device uchar* a2,
    device uchar* o0,
    device uchar* o1,
    device uchar* temporary,
    uint tid,
    uint nthreads) {
  if (status->state != 1) return;
  M1OpParams p = params[0];
  p.tag = generated_tag;
  const M1ValueDesc d0 = descriptors[p.a0];
  const M1ValueDesc d1 = descriptors[p.a1];
  const M1ValueDesc d2 = descriptors[p.a2];
  const M1ValueDesc out0 = descriptors[p.o0];

  if (p.tag == 0x81) {
    for (uint i = tid; i < out0.len; i += nthreads) {
      if (p.lit_dtype == 0) m1_store_fd(o0, i, as_type<float>(p.lit_bits), out0.dtype);
      else if (p.lit_dtype == 1) m1_store_i(o0, i, int(p.lit_bits));
      else if (p.lit_dtype == 2) m1_store_u(o0, i, p.lit_bits);
      else m1_store_b(o0, i, p.lit_bits != 0);
    }
    return;
  }
  if (p.tag == 0x90 || p.tag == 0x91) {
    if (out0.dtype == 3) {
      for (uint i = tid; i < out0.len; i += nthreads)
        o0[i] = (a0[i >> 3] >> (i & 7)) & 1u;
    } else {
      m1_copy_typed_range(a0, o0, out0.len, out0.dtype, tid, nthreads);
    }
    return;
  }
  if (p.tag == 0x92) {
    const uint logical_bytes = m1_value_bytes(d0.len, d0.dtype);
    if (logical_bytes > p.sink_bytes) {
      if (tid == 0) m1_fault_op(status, 1u, p);
      return;
    }
    if (d0.dtype == 3) {
      // One packed byte a thread, from its eight bools: no thread writes a
      // byte another touches, so the pack runs across the whole grid (a
      // single thread setting bits one read-modify-write at a time cost a
      // beam step's mask ~0.1 ms).
      for (uint b = tid; b < p.sink_bytes; b += nthreads) {
        uchar packed = 0;
        for (uint j = 0; j < 8u; ++j) {
          const uint i = b * 8u + j;
          if (i < d0.len && a0[i] != 0) packed |= uchar(1u << j);
        }
        o0[b] = packed;
      }
      return;
    }
    m1_copy_typed_range(a0, o0, d0.len, d0.dtype, tid, nthreads);
    for (uint i = logical_bytes + tid; i < p.sink_bytes; i += nthreads) o0[i] = 0;
    return;
  }
  if (p.tag == 0xA0) {
    const device bfloat* logits = reinterpret_cast<const device bfloat*>(a0);
    device float* out_f = reinterpret_cast<device float*>(o0);
    if (nthreads > 1u) {
      for (uint i = tid; i < out0.len; i += nthreads) out_f[i] = float(logits[i]);
      return;
    }

    uint i = 0;
    for (; i + 8u <= out0.len; i += 8u) {
      const float v0 = float(logits[i + 0]), v1 = float(logits[i + 1]);
      const float v2 = float(logits[i + 2]), v3 = float(logits[i + 3]);
      const float v4 = float(logits[i + 4]), v5 = float(logits[i + 5]);
      const float v6 = float(logits[i + 6]), v7 = float(logits[i + 7]);
      out_f[i + 0] = v0; out_f[i + 1] = v1; out_f[i + 2] = v2; out_f[i + 3] = v3;
      out_f[i + 4] = v4; out_f[i + 5] = v5; out_f[i + 6] = v6; out_f[i + 7] = v7;
    }
    for (; i < out0.len; ++i) out_f[i] = float(logits[i]);
    return;
  }
  if (p.tag == 0xA1) {
    m1_copy_typed_range(a0, o0, out0.len, out0.dtype, tid, nthreads);
    return;
  }
  if (p.tag == 0xA2) {
    return;
  }

  if (p.tag == 0x01 || p.tag == 0x02 || p.tag == 0x04) {
    for (uint i = tid; i < out0.len; i += nthreads) {
      const float value = m1_load_f(a0, m1_pick(d0.len, i), d0.dtype);
      if (p.tag == 0x01) m1_store_fd(o0, i, precise::exp(value), out0.dtype);
      else if (p.tag == 0x02) m1_store_fd(o0, i, precise::log(value), out0.dtype);
      else m1_store_fd(o0, i, 1.0f / value, out0.dtype);
    }
    return;
  }
  if (p.tag >= 0x08 && p.tag <= 0x0B) {
    for (uint i = tid; i < out0.len; i += nthreads) {
      const float value = m1_load_f(a0, m1_pick(d0.len, i), d0.dtype);
      if (p.tag == 0x08) m1_store_fd(o0, i, precise::sin(value), out0.dtype);
      else if (p.tag == 0x09) m1_store_fd(o0, i, precise::cos(value), out0.dtype);
      else if (p.tag == 0x0A) m1_store_fd(o0, i, sqrt(value), out0.dtype);
      else m1_store_fd(o0, i, 1.0f / sqrt(value), out0.dtype);
    }
    return;
  }
  if (p.tag == 0x03 || p.tag == 0x05 || p.tag == 0x06) {
    if (d0.dtype == 3) {
      if (tid == 0 && out0.len != 0) m1_fault(status, p.tag);
      return;
    }
    for (uint i = tid; i < out0.len; i += nthreads) {
      const uint source = m1_pick(d0.len, i);
      if (m1_is_float(d0.dtype)) {
        const float value = m1_load_f(a0, source, d0.dtype);
        const float result =
            p.tag == 0x03 ? -value
                          : (p.tag == 0x05
                                 ? abs(value)
                                 : (value > 0 ? 1.0f : (value < 0 ? -1.0f : 0.0f)));
        m1_store_fd(o0, i, result, out0.dtype);
      } else if (d0.dtype == 1) {
        const int value = m1_load_i(a0, source, d0.dtype);
        int result = value;
        if (p.tag == 0x03) result = int(0u - uint(value));
        else if (p.tag == 0x05) result = value == INT_MIN ? value : abs(value);
        else result = value > 0 ? 1 : (value < 0 ? -1 : 0);
        m1_store_i(o0, i, result);
      } else {
        const uint value = m1_load_u(a0, source, d0.dtype);
        m1_store_u(
            o0, i,
            p.tag == 0x03 ? 0u - value
                          : (p.tag == 0x06 ? (value != 0 ? 1u : 0u) : value));
      }
    }
    return;
  }
  if (p.tag == 0x07) {
    for (uint i = tid; i < out0.len; i += nthreads) {
      const uint source = m1_pick(d0.len, i);
      if (out0.dtype == 0 || out0.dtype == 5) m1_store_fd(o0, i, m1_load_f(a0, source, d0.dtype), out0.dtype);
      else if (out0.dtype == 1) m1_store_i(o0, i, m1_load_i(a0, source, d0.dtype));
      else if (out0.dtype == 2) m1_store_u(o0, i, m1_load_u(a0, source, d0.dtype));
      else m1_store_b(o0, i, m1_load_b(a0, source, d0.dtype));
    }
    return;
  }

  if ((p.tag >= 0x10 && p.tag <= 0x1D) || p.tag == 0x1F) {
    for (uint i = tid; i < out0.len; i += nthreads) {
      const uint xindex = m1_pick(d0.len, i), yindex = m1_pick(d1.len, i);
      if (p.tag >= 0x16 && p.tag <= 0x1D) {
        bool result = false;
        if (p.tag == 0x1C || p.tag == 0x1D) {
          const bool x = m1_load_b(a0, xindex, d0.dtype);
          const bool y = m1_load_b(a1, yindex, d1.dtype);
          result = p.tag == 0x1C ? x && y : x || y;
        } else if (m1_is_float(d0.dtype)) {
          const float x = m1_load_f(a0, xindex, d0.dtype);
          const float y = m1_load_f(a1, yindex, d1.dtype);
          if (p.tag == 0x16) result = x > y;
          else if (p.tag == 0x17) result = x >= y;
          else if (p.tag == 0x18) result = x == y;
          else if (p.tag == 0x19) result = x != y;
          else if (p.tag == 0x1A) result = x < y;
          else result = x <= y;
        } else if (d0.dtype == 1) {
          const int x = m1_load_i(a0, xindex, d0.dtype);
          const int y = m1_load_i(a1, yindex, d1.dtype);
          if (p.tag == 0x16) result = x > y;
          else if (p.tag == 0x17) result = x >= y;
          else if (p.tag == 0x18) result = x == y;
          else if (p.tag == 0x19) result = x != y;
          else if (p.tag == 0x1A) result = x < y;
          else result = x <= y;
        } else {
          const uint x = m1_load_u(a0, xindex, d0.dtype);
          const uint y = m1_load_u(a1, yindex, d1.dtype);
          if (p.tag == 0x16) result = x > y;
          else if (p.tag == 0x17) result = x >= y;
          else if (p.tag == 0x18) result = x == y;
          else if (p.tag == 0x19) result = x != y;
          else if (p.tag == 0x1A) result = x < y;
          else result = x <= y;
        }
        m1_store_b(o0, i, result);
      } else if (m1_is_float(d0.dtype)) {
        const float x = m1_load_f(a0, xindex, d0.dtype);
        const float y = m1_load_f(a1, yindex, d1.dtype);
        float result = 0;
        if (p.tag == 0x10) result = x + y;
        else if (p.tag == 0x11) result = x - y;
        else if (p.tag == 0x12) result = x * y;
        else if (p.tag == 0x13) result = x / y;
        else if (p.tag == 0x14) result = m1_element_max(x, y);
        else if (p.tag == 0x15) result = m1_element_min(x, y);
        else result = fmod(x, y);
        m1_store_fd(o0, i, result, out0.dtype);
      } else if (d0.dtype == 1) {
        const int x = m1_load_i(a0, xindex, d0.dtype);
        const int y = m1_load_i(a1, yindex, d1.dtype);
        int result = 0;
        if (p.tag == 0x10) result = int(uint(x) + uint(y));
        else if (p.tag == 0x11) result = int(uint(x) - uint(y));
        else if (p.tag == 0x12) result = int(uint(x) * uint(y));
        else if (p.tag == 0x13) result = y == 0 ? 0 : x / y;
        else if (p.tag == 0x14) result = max(x, y);
        else if (p.tag == 0x15) result = min(x, y);
        else result = y == 0 ? 0 : x % y;
        m1_store_i(o0, i, result);
      } else {
        const uint x = m1_load_u(a0, xindex, d0.dtype);
        const uint y = m1_load_u(a1, yindex, d1.dtype);
        uint result = 0;
        if (p.tag == 0x10) result = x + y;
        else if (p.tag == 0x11) result = x - y;
        else if (p.tag == 0x12) result = x * y;
        else if (p.tag == 0x13) result = y == 0 ? 0 : x / y;
        else if (p.tag == 0x14) result = max(x, y);
        else if (p.tag == 0x15) result = min(x, y);
        else result = y == 0 ? 0 : x % y;
        m1_store_u(o0, i, result);
      }
    }
    return;
  }
  if (p.tag == 0x1E) {
    for (uint i = tid; i < out0.len; i += nthreads)
      m1_store_b(o0, i, !m1_load_b(a0, m1_pick(d0.len, i), d0.dtype));
    return;
  }
  if (p.tag == 0x20) {
    for (uint i = tid; i < out0.len; i += nthreads) {
      const bool select = m1_load_b(a0, m1_pick(d0.len, i), d0.dtype);
      const uint xi = m1_pick(d1.len, i), yi = m1_pick(d2.len, i);
      if (m1_is_float(out0.dtype))
        m1_store_fd(o0, i, select ? m1_load_f(a1, xi, d1.dtype)
                                 : m1_load_f(a2, yi, d2.dtype), out0.dtype);
      else if (out0.dtype == 1)
        m1_store_i(o0, i, select ? m1_load_i(a1, xi, d1.dtype)
                                 : m1_load_i(a2, yi, d2.dtype));
      else if (out0.dtype == 2)
        m1_store_u(o0, i, select ? m1_load_u(a1, xi, d1.dtype)
                                 : m1_load_u(a2, yi, d2.dtype));
      else
        m1_store_b(o0, i, select ? m1_load_b(a1, xi, d1.dtype)
                                 : m1_load_b(a2, yi, d2.dtype));
    }
    return;
  }

  if (p.tag >= 0x30 && p.tag <= 0x32) {
    if (m1_is_float(d0.dtype))
      m1_reduce_float_part(p.tag, a0, o0, temporary, d0, tid, nthreads);
    else
      m1_reduce_integer_part(p.tag, a0, o0, temporary, d0, tid, nthreads);
    return;
  }
  if (p.tag == 0x33) {

    if (tid == 0) m1_reduce_argmax(a0, o0, temporary, d0);
    return;
  }
  if (p.tag == 0x38) {
    uint source_stride[4] = {1, 1, 1, 1};
    for (int dim = int(out0.rank) - 2; dim >= 0; --dim)
      source_stride[dim] =
          source_stride[dim + 1] * (uint(dim + 1) < d0.rank ? d0.dims[dim + 1] : 1u);
    for (uint linear = tid; linear < out0.len; linear += nthreads) {
      uint rem = linear, source_index = 0;
      for (uint dim = 0; dim < out0.rank; ++dim) {
        uint stride = 1;
        for (uint next = dim + 1; next < out0.rank; ++next)
          stride *= out0.dims[next];
        const uint coordinate = rem / max(stride, 1u);
        rem %= max(stride, 1u);
        const uint source_dim = dim < d0.rank ? d0.dims[dim] : 1u;
        if (source_dim != 1) source_index += coordinate * source_stride[dim];
      }
      if (out0.dtype == 0 || out0.dtype == 5) m1_store_fd(o0, linear, m1_load_f(a0, source_index, d0.dtype), out0.dtype);
      else if (out0.dtype == 1) m1_store_i(o0, linear, m1_load_i(a0, source_index, d0.dtype));
      else if (out0.dtype == 2) m1_store_u(o0, linear, m1_load_u(a0, source_index, d0.dtype));
      else m1_store_b(o0, linear, m1_load_b(a0, source_index, d0.dtype));
    }
    return;
  }
  if (p.tag == 0x39) {
    m1_copy_typed_range(a0, o0, out0.len, out0.dtype, tid, nthreads);
    return;
  }
  if (p.tag == 0x3A) {
    if (d0.rank != 2) {
      if (tid == 0) m1_fault(status, p.tag);
      return;
    }
    const uint m = d0.dims[0], n = d0.dims[1];
    for (uint index = tid; index < m * n; index += nthreads) {
      const uint source = (index % m) * n + index / m;
      if (out0.dtype == 0 || out0.dtype == 5) m1_store_fd(o0, index, m1_load_f(a0, source, d0.dtype), out0.dtype);
      else if (out0.dtype == 1) m1_store_i(o0, index, m1_load_i(a0, source, d0.dtype));
      else if (out0.dtype == 2) m1_store_u(o0, index, m1_load_u(a0, source, d0.dtype));
      else m1_store_b(o0, index, m1_load_b(a0, source, d0.dtype));
    }
    return;
  }
  if (p.tag == 0x40 || p.tag == 0x41) {

    const bool is_sum = p.tag == 0x40;
    for (uint row = tid; row < d0.rows; row += nthreads) {
      float accumulated_f = is_sum ? 0.0f : 1.0f;
      uint accumulated_u = is_sum ? 0u : 1u;
      int accumulated_i = is_sum ? 0 : 1;
      for (uint column = 0; column < d0.last; ++column) {
        const uint index = row * d0.last + column;
        if (d0.dtype == 1) {
          const int value = m1_load_i(a0, index, d0.dtype);
          accumulated_i = is_sum ? int((uint)accumulated_i + (uint)value)
                                 : int((uint)accumulated_i * (uint)value);
          m1_store_i(o0, index, accumulated_i);
        } else if (d0.dtype == 2) {
          const uint value = m1_load_u(a0, index, d0.dtype);
          accumulated_u = is_sum ? accumulated_u + value : accumulated_u * value;
          m1_store_u(o0, index, accumulated_u);
        } else {
          const float value = m1_load_f(a0, index, d0.dtype);
          accumulated_f = is_sum ? accumulated_f + value : accumulated_f * value;
          m1_store_fd(o0, index, accumulated_f, out0.dtype);
        }
      }
    }
    return;
  }
  if (p.tag == 0x50) {
    if (tid != 0) return;
    for (uint position = 0; position < d0.len; ++position) {
      uint best_index = 0;
      float best_value = NAN;
      bool found = false;
      for (uint candidate = 0; candidate < d0.len; ++candidate) {
        bool used = false;
        for (uint prior = 0; prior < position; ++prior)
          if (reinterpret_cast<device uint*>(o1)[prior] == candidate) used = true;
        if (used) continue;
        const float value = m1_load_f(a0, candidate, d0.dtype);
        if (!found || m1_sort_better(value, candidate, best_value, best_index)) {
          found = true;
          best_value = value;
          best_index = candidate;
        }
      }
      m1_store_fd(o0, position, best_value, out0.dtype);
      m1_store_u(o1, position, best_index);
    }
    return;
  }
  if (p.tag == 0x51) {

    const uint count = min(p.imm, d0.last);
    for (uint row = tid; row < d0.rows; row += nthreads) {
      for (uint position = 0; position < count; ++position) {
        uint best_index = 0;
        float best_value = NAN;
        bool found = false;
        for (uint candidate = 0; candidate < d0.last; ++candidate) {
          bool used = false;
          for (uint prior = 0; prior < position; ++prior)
            if (reinterpret_cast<device uint*>(o1)[row * count + prior] == candidate)
              used = true;
          if (used) continue;
          const float value = m1_load_f(a0, row * d0.last + candidate, d0.dtype);
          if (!found || m1_sort_better(value, candidate, best_value, best_index)) {
            found = true;
            best_value = value;
            best_index = candidate;
          }
        }
        m1_store_fd(o0, row * count + position, best_value, out0.dtype);
        m1_store_u(o1, row * count + position, best_index);
      }
    }
    return;
  }
  if (p.tag == 0x55) {
    if (d0.rank != 2 || d1.rank != 2) {
      if (tid == 0) m1_fault(status, p.tag);
      return;
    }

    const uint m = d0.dims[0], inner = d0.dims[1], n = d1.dims[1];
    for (uint row = tid; row < m; row += nthreads) {
      for (uint column = 0; column < n; ++column)
        m1_store_fd(o0, row * n + column, 0.0f, out0.dtype);
      for (uint k = 0; k < inner; ++k) {
        const float left = m1_load_f(a0, row * inner + k, d0.dtype);
        if (left == 0.0f) continue;
        for (uint column = 0; column < n; ++column) {
          const uint index = row * n + column;
          const float old = m1_load_f(o0, index, 0);
          m1_store_fd(o0, index, old + left * m1_load_f(a1, k * n + column, d1.dtype), out0.dtype);
        }
      }
    }
    return;
  }
  if (p.tag == 0x58) {
    if (p.pred_tag == 2) {

      for (uint i = tid; i < d0.rows * d0.last; i += nthreads) {
        const uint row = d0.last == 0u ? 0u : i / d0.last;
        const float threshold = m1_load_f(a1, m1_pick(d1.len, row), d1.dtype);
        m1_store_b(o0, i, m1_load_f(a0, i, d0.dtype) >= threshold);
      }
      return;
    }

    // Top-k (pred 0) / top-p (pred 1), a thread a row: the rule and the
    // radix select of the cut the threadgroup forms use (m1_cut_select_mt),
    // with the histograms in the thread's own arrays.
    for (uint row = tid; row < d0.rows; row += nthreads) {
      m1_cut_select_row(a0, a1, o0, d0, d1, p.pred_tag == 0 ? 0u : 1u, row);
    }
    return;
  }
  if (p.tag == 0x60) {
    const uint rest = d0.rank == 0 ? 1u : d0.len / max(d0.dims[0], 1u);
    const uint n0 = d0.rank == 0 ? 1u : d0.dims[0];
    for (uint output_index = tid; output_index < d1.len * rest; output_index += nthreads) {
      const uint k = output_index / rest;
      const uint r = output_index - k * rest;
      const long index = m1_load_index(a1, k, d1.dtype);
      const bool valid = index >= 0 && uint(index) < n0;
      const uint source = valid ? uint(index) * rest + r : 0;
      if (out0.dtype == 0 || out0.dtype == 5) m1_store_fd(o0, output_index, valid ? m1_load_f(a0, source, d0.dtype) : 0.0f, out0.dtype);
      else if (out0.dtype == 1) m1_store_i(o0, output_index, valid ? m1_load_i(a0, source, d0.dtype) : 0);
      else if (out0.dtype == 2) m1_store_u(o0, output_index, valid ? m1_load_u(a0, source, d0.dtype) : 0u);
      else m1_store_b(o0, output_index, valid && m1_load_b(a0, source, d0.dtype));
    }
    return;
  }
  if (p.tag == 0x61) {
    const uint rows = d0.dims[0], columns = d0.dims[1];
    for (uint row = tid; row < rows; row += nthreads) {
      const long column = m1_load_index(a1, row, d1.dtype);
      const bool valid = column >= 0 && uint(column) < columns;
      const uint source = valid ? row * columns + uint(column) : 0;
      if (out0.dtype == 0 || out0.dtype == 5) m1_store_fd(o0, row, valid ? m1_load_f(a0, source, d0.dtype) : 0.0f, out0.dtype);
      else if (out0.dtype == 1) m1_store_i(o0, row, valid ? m1_load_i(a0, source, d0.dtype) : 0);
      else if (out0.dtype == 2) m1_store_u(o0, row, valid ? m1_load_u(a0, source, d0.dtype) : 0u);
      else m1_store_b(o0, row, valid && m1_load_b(a0, source, d0.dtype));
    }
    return;
  }
  if (p.tag == 0x62 || p.tag == 0x63) {

    m1_copy_typed_range(a0, o0, d0.len, d0.dtype, tid, nthreads);
    if (nthreads > 1u) threadgroup_barrier(mem_flags::mem_device);
    if (tid != 0) return;
    m1_scatter_rmw(p.tag, a1, a2, o0, d0, d1, d2);
    return;
  }
  if (p.tag == 0x64) {
    for (uint i = tid; i < out0.len; i += nthreads) m1_store_u(o0, i, i);
    return;
  }
  if (p.tag == 0x65) {
    const uint mask_width =
        d0.rank == 0 ? 1u : d0.dims[d0.rank - 1];
    for (uint i = tid; i < d0.len; i += nthreads) {
      const uint column = i % mask_width;
      const uint word = column >> 5;
      const uint mask = word < d1.len ? m1_load_u(a1, word, d1.dtype) : 0u;
      m1_store_fd(
          o0, i,
          ((mask >> (column & 31)) & 1u) != 0
              ? m1_load_f(a0, i, d0.dtype)
              : -INFINITY,
          out0.dtype);
    }
    return;
  }
  if (p.tag == 0x66 || p.tag == 0x67 || p.tag == 0x68) {
    const uint key_count = p.imm;
    const uint window = p.tag == 0x67 ? p.imm2 : p.imm3;
    for (uint index = tid; index < out0.len; index += nthreads) {
      const uint position_index =
          key_count == 0u ? 0u : index / key_count;
      const uint key = key_count == 0u ? 0u : index % key_count;
      const uint position =
          m1_load_u(a0, position_index, d0.dtype);
      bool allowed = key_count != 0u && key <= position;
      if (allowed && p.tag != 0x66) {
        const uint reach =
            key > UINT_MAX - window ? UINT_MAX : key + window;
        const bool recent = reach > position;
        allowed =
            p.tag == 0x67
                ? recent
                : (key < p.imm2 || recent);
      }
      m1_store_b(o0, index, allowed);
    }
    return;
  }
  if (p.tag == 0x70 || p.tag == 0x71) {
    ulong seed;
    if (p.tag == 0x70) {
      seed = ptir_rng_seed_eff_stream(0u, p.imm);
    } else {
      const ulong key = ulong(m1_load_u(a0, 0, d0.dtype));
      const ulong counter =
          ulong(d0.len > 1 ? m1_load_u(a0, 1, d0.dtype) : 0u);
      seed = ptir_rng_keyed_seed(uint(key), uint(counter));
    }
    for (uint i = tid; i < out0.len; i += nthreads) {
      if (p.kind == 2) {
        m1_store_fd(o0, i, ptir_rng_hash_normal(seed, i), out0.dtype);
        continue;
      }
      const float uniform = ptir_rng_hash_uniform(seed, i);
      m1_store_fd(
          o0, i,
          p.kind == 0 ? uniform : -precise::log(-precise::log(uniform)),
          out0.dtype);
    }
    return;
  }

  if (tid == 0) m1_fault_op(status, 3u, p);
}

inline void ptir_m1_execute(
    uint generated_tag,
    device M1Status* status,
    const device M1ValueDesc* descriptors,
    const device M1OpParams* params,
    const device uchar* a0,
    const device uchar* a1,
    const device uchar* a2,
    device uchar* o0,
    device uchar* o1,
    device uchar* temporary) {
  ptir_m1_execute_part(generated_tag, status, descriptors, params, a0, a1, a2, o0,
                       o1, temporary, 0u, 1u);
}

inline M1ArgmaxCandidate m1_sort_pick(M1ArgmaxCandidate left, M1ArgmaxCandidate right) {
  if (right.have == 0u) return left;
  if (left.have == 0u) return right;
  return m1_sort_better(right.value, right.index, left.value, left.index) ? right : left;
}

inline uint m4_reduce_count(uint last, uint level) {
  uint count = last;
  for (uint l = 0; l < level; ++l) count = (count + 31u) / 32u;
  return count;
}

// Element `i` of a reduction's input as the bits its fold reads: a bf16
// element widened to its f32's bits, any other a word.
inline uint m4_element_bits(const device uchar* input, uint i, uint dtype) {
  if (dtype == 5) return uint(reinterpret_cast<const device ushort*>(input)[i]) << 16;
  return reinterpret_cast<const device uint*>(input)[i];
}

inline uint m4_reduce_identity(uint tag, uint dtype) {
  if (m1_is_float(dtype)) return as_type<uint>(tag == 0x30 ? 0.0f : (tag == 0x31 ? -INFINITY : INFINITY));
  if (tag == 0x30) return 0u;
  if (dtype == 1) return tag == 0x31 ? uint(INT_MIN) : uint(INT_MAX);
  return tag == 0x31 ? 0u : UINT_MAX;
}

inline uint m4_fold_chunk(uint tag, uint dtype, uint bits) {
  if (m1_is_float(dtype)) {
    float v = as_type<float>(bits);
    for (uint offset = 16u; offset > 0u; offset >>= 1) {
      const float other = simd_shuffle_down(v, offset);
      if (tag == 0x30) v = v + other;
      else if (tag == 0x31) v = m1_canonical_max(v, other);
      else v = m1_canonical_min(v, other);
    }
    return as_type<uint>(v);
  }
  uint v = bits;
  for (uint offset = 16u; offset > 0u; offset >>= 1) {
    const uint other = simd_shuffle_down(v, offset);
    if (tag == 0x30) v = v + other;
    else if (dtype == 1) {
      const int left = int(v), right = int(other);
      v = uint(tag == 0x31 ? max(left, right) : min(left, right));
    } else {
      v = tag == 0x31 ? max(v, other) : min(v, other);
    }
  }
  return v;
}

#define M4_REDUCE_CHUNKS_PER_GROUP 4u

inline void m4_reduce_two_levels(
    uint tag,
    const device uchar* input,
    device uchar* output,
    device uchar* temporary,
    const M1ValueDesc d,
    uint level,
    uint group,
    uint tid,
    uint threads,
    threadgroup M1ArgmaxCandidate* tgbuf,
    uint simd_lane,
    uint simd_id,
    // Where stated, level 0 reads exp(x - center[row]) for x, both read
    // through their dtypes: a log-softmax's sum of exponentials from its
    // row itself.
    const device uchar* center = nullptr,
    uint center_dtype = 0u) {
  const uint count = m4_reduce_count(d.last, level);
  const uint chunks = (count + 31u) / 32u;
  const uint pitch = (d.last + 31u) / 32u;
  const uint next_chunks = (chunks + 31u) / 32u;
  const uint identity = m4_reduce_identity(tag, d.dtype);
  const device uint* plane = reinterpret_cast<const device uint*>(temporary);
  device uint* planes = reinterpret_cast<device uint*>(temporary);
  device uint* result = reinterpret_cast<device uint*>(output);
  if (count <= 1u) {

    if (group != 0u) return;
    for (uint row = tid; row < d.rows; row += threads) {
      if (count == 0u) {
        result[row] = identity;
      } else {
        result[row] = level == 0u
            ? (center != nullptr
                   ? as_type<uint>(precise::exp(m1_load_f(input, row * d.last, d.dtype) - m1_load_f(center, row, center_dtype)))
                   : m4_element_bits(input, row * d.last, d.dtype))
            : plane[((level - 1u) & 1u) * d.rows * pitch + row * pitch];
      }
    }
    return;
  }
  const uint span = 32u * M4_REDUCE_CHUNKS_PER_GROUP;
  // A row's groups side by side, the rows one after another: the host
  // launches per_row groups a row (`StepKind::Reduce`), so the rows fold
  // at once, not one after another in every group.
  const uint per_row = max((next_chunks + M4_REDUCE_CHUNKS_PER_GROUP - 1u) / M4_REDUCE_CHUNKS_PER_GROUP, 1u);
  const uint row = group / per_row;
  group = group % per_row;
  if (row >= d.rows || group * M4_REDUCE_CHUNKS_PER_GROUP >= next_chunks) return;
  threadgroup uint* tg = reinterpret_cast<threadgroup uint*>(tgbuf);
  const uint simds = threads / 32u;
  const uint rounds = span / simds;
  {
    const device uint* src = level == 0u
        ? reinterpret_cast<const device uint*>(input) + row * d.last
        : plane + ((level - 1u) & 1u) * d.rows * pitch + row * pitch;
    const float centre = (level == 0u && center != nullptr) ? m1_load_f(center, row, center_dtype) : 0.0f;

    uint held[8];
    for (uint k = 0; k < 8u; ++k) {
      const uint chunk = group * span + simd_id + k * simds;
      const uint at = chunk * 32u + simd_lane;
      held[k] = (k < rounds && chunk < chunks && at < count)
          ? (level == 0u ? m4_element_bits(input, row * d.last + at, d.dtype) : src[at])
          : identity;
      // A bf16 element is read widened to its f32's bits.
      if (level == 0u && center != nullptr && k < rounds && chunk < chunks && at < count) {
        held[k] = as_type<uint>(precise::exp(m1_bits_f(held[k], d.dtype == 5u ? 0u : d.dtype) - centre));
      }
    }
    for (uint k = 0; k < 8u; ++k) {
      if (k < rounds) {
        const uint folded = m4_fold_chunk(tag, d.dtype, held[k]);
        if (simd_lane == 0u) tg[simd_id + k * simds] = folded;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (simd_id < M4_REDUCE_CHUNKS_PER_GROUP) {
      const uint next_chunk = group * M4_REDUCE_CHUNKS_PER_GROUP + simd_id;
      const uint folded = m4_fold_chunk(tag, d.dtype, tg[simd_id * 32u + simd_lane]);
      if (simd_lane == 0u && next_chunk < next_chunks) {
        if (next_chunks == 1u) result[row] = folded;
        else planes[((level + 1u) & 1u) * d.rows * pitch + row * pitch + next_chunk] = folded;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}


inline void m4_reduce_partial(
    uint tag, uint dtype, const device uchar* input, device uint* plane, uint n,
    uint gtid, uint gthreads, uint simd_lane) {
  const uint identity = m4_reduce_identity(tag, dtype);

  uint i0 = gtid - simd_lane;
  for (; i0 < n; i0 += 8u * gthreads) {
    uint held[8];
    for (uint k = 0; k < 8u; ++k) {
      const uint i = i0 + k * gthreads + simd_lane;
      held[k] = i < n ? m4_element_bits(input, i, dtype) : identity;
    }
    for (uint k = 0; k < 8u; ++k) {
      const uint chunk0 = i0 + k * gthreads;
      if (chunk0 < n) {
        const uint folded = m4_fold_chunk(tag, dtype, held[k]);
        if (simd_lane == 0u) plane[chunk0 >> 5] = folded;
      }
    }
  }
}

inline uint m4_reduce_final(
    uint tag, uint dtype, const device uchar* input, uint n, const device uint* plane,
    device M1Status* status, uint threads, uint simd_lane, uint simd_id,
    threadgroup M1ArgmaxCandidate* tgbuf) {
  const uint identity = m4_reduce_identity(tag, dtype);
  if (n == 0u) return identity;
  if (n == 1u) return m4_element_bits(input, 0u, dtype);
  uint count = (n + 31u) / 32u;
  if (count == 1u) return plane[0];
  if (count > 32768u) {
    if (simd_lane == 0u && simd_id == 0u) m1_fault(status, 0xB4u);
    return identity;
  }
  threadgroup uint* tg = reinterpret_cast<threadgroup uint*>(tgbuf);
  const uint simds = threads / 32u;
  uint next = (count + 31u) / 32u;

  for (uint c0 = simd_id; c0 < next; c0 += 16u * simds) {
    uint held[16];
    for (uint k = 0; k < 16u; ++k) {
      const uint at = (c0 + k * simds) * 32u + simd_lane;
      held[k] = (c0 + k * simds < next && at < count) ? plane[at] : identity;
    }
    for (uint k = 0; k < 16u; ++k) {
      const uint c = c0 + k * simds;
      if (c < next) {
        const uint folded = m4_fold_chunk(tag, dtype, held[k]);
        if (simd_lane == 0u) tg[c] = folded;
      }
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  uint src_at = 0u, dst_at = 1024u;
  count = next;
  while (count > 1u) {
    next = (count + 31u) / 32u;
    for (uint c = simd_id; c < next; c += simds) {
      const uint at = c * 32u + simd_lane;
      const uint folded = m4_fold_chunk(tag, dtype, at < count ? tg[src_at + at] : identity);
      if (simd_lane == 0u) tg[dst_at + c] = folded;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    const uint swap = src_at;
    src_at = dst_at;
    dst_at = swap;
    count = next;
  }
  const uint result = tg[src_at];
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return result;
}


// The streamed top-k / top-p select (`pivot_threshold` under RANK_LE or
// CUMMASS_LE): which tokens of each row a sample may draw, as a mask.
//
// Tokens are taken in descending value, ties in ascending index. Top-k takes
// the first k. Top-p takes every token whose exclusive mass (the mass of the
// tokens before it) is below p: the boundary is the first token whose
// exclusive mass is not below p, as before.
//
// The cut is found by a radix select on the descending key, 8 bits a pass,
// four passes over the row whatever the nucleus's size: each pass histograms
// the tokens still in play by count and by mass, and one thread walks the 256
// buckets to the one the cut lies in. The mass is exact and order-independent:
// a probability p is the 64-bit integer floor(p * 2^32) (a token below 2^-32
// weighs nothing), summed as two 32-bit words with carry, so no summation
// order moves the cut. A row's total is at most V * 2^32 for probabilities
// (2^50 at V = 2^18), far inside the two words. The threshold is
// floor(p * 2^32) the same way.
//
// Each row keeps its state and histograms at `base + row * M4_SEL_ROW_BYTES`,
// for up to M4_SEL_ROWS rows; a select of more rows is refused by name.

#define M4_SEL_ROWS 16u
#define M4_SEL_ROW_BYTES 3200u
#define M4_SEL_BYTES (M4_SEL_ROWS * M4_SEL_ROW_BYTES)

struct M4SelState {
  uint done;        // the cut is known (or the row takes nothing / all)
  uint all;         // every token of the row is taken
  uint prefix;      // the cut key's digits picked so far
  uint before;      // tokens strictly before the bucket in play
  uint before_lo;   // their mass, low word
  uint before_hi;   // their mass, high word
  uint target_lo;   // top-p: the threshold's mass; top-k: k in target_lo
  uint target_hi;
  uint take;        // tokens of the cut key taken (in index order)
  uint ties;        // tokens of the cut key in the row
  uint pad[6];
};

inline device M4SelState* m4_sel_state(device uchar* base, uint row) {
  return reinterpret_cast<device M4SelState*>(base + row * M4_SEL_ROW_BYTES);
}
// Per row: counts at +64, mass low words at +1088, high words at +2112.
inline device atomic_uint* m4_sel_count(device uchar* base, uint row) {
  return reinterpret_cast<device atomic_uint*>(base + row * M4_SEL_ROW_BYTES + 64u);
}
inline device atomic_uint* m4_sel_lo(device uchar* base, uint row) {
  return reinterpret_cast<device atomic_uint*>(base + row * M4_SEL_ROW_BYTES + 1088u);
}
inline device atomic_uint* m4_sel_hi(device uchar* base, uint row) {
  return reinterpret_cast<device atomic_uint*>(base + row * M4_SEL_ROW_BYTES + 2112u);
}

// Adds `v` to the two-word counter (lo, hi), carrying out of the low word.
inline void m4_sel_add(threadgroup atomic_uint* lo, threadgroup atomic_uint* hi, ulong v) {
  const uint low = uint(v);
  const uint old = atomic_fetch_add_explicit(lo, low, memory_order_relaxed);
  const uint carry = (old + low < old) ? 1u : 0u;
  const uint high = uint(v >> 32) + carry;
  if (high != 0u) atomic_fetch_add_explicit(hi, high, memory_order_relaxed);
}
inline void m4_sel_add(device atomic_uint* lo, device atomic_uint* hi, ulong v) {
  const uint low = uint(v);
  const uint old = atomic_fetch_add_explicit(lo, low, memory_order_relaxed);
  const uint carry = (old + low < old) ? 1u : 0u;
  const uint high = uint(v >> 32) + carry;
  if (high != 0u) atomic_fetch_add_explicit(hi, high, memory_order_relaxed);
}

inline void m4_sel_init(
    device uchar* base, device uchar* o0, const M1ValueDesc d0, const M1ValueDesc d1,
    const device uchar* a1, uint mode, uint gtid, uint gthreads, device M1Status* status,
    M1OpParams p) {
  const uint rows = d0.rows;
  if (rows > M4_SEL_ROWS) {
    // A guard the host names (site 4): a select of more rows than its state holds.
    if (gtid == 0u) m1_fault_op(status, 4u, p);
    return;
  }
  for (uint row = 0u; row < rows; ++row) {
    device atomic_uint* count = m4_sel_count(base, row);
    device atomic_uint* lo = m4_sel_lo(base, row);
    device atomic_uint* hi = m4_sel_hi(base, row);
    for (uint b = gtid; b < 256u; b += gthreads) {
      atomic_store_explicit(&count[b], 0u, memory_order_relaxed);
      atomic_store_explicit(&lo[b], 0u, memory_order_relaxed);
      atomic_store_explicit(&hi[b], 0u, memory_order_relaxed);
    }
  }
  if (gtid == 0u) {
    for (uint row = 0u; row < rows; ++row) {
      device M4SelState* st = m4_sel_state(base, row);
      st->prefix = 0u;
      st->before = 0u;
      st->before_lo = 0u;
      st->before_hi = 0u;
      st->take = 0u;
      st->ties = 0u;
      st->all = 0u;
      uint done = d0.last == 0u ? 1u : 0u;
      if (mode == 0u) {
        const int signed_k = m1_load_i(a1, 0u, d1.dtype);
        const uint k = signed_k <= 0 ? 0u : uint(signed_k);
        st->target_lo = k;
        st->target_hi = 0u;
        if (k == 0u) done = 1u;
        if (k >= d0.last) { st->all = 1u; done = 1u; }
      } else {
        const ulong target = m4_sel_fixed(m1_load_f(a1, 0u, d1.dtype));
        st->target_lo = uint(target);
        st->target_hi = uint(target >> 32);
        // A threshold of no mass takes nothing: the first token's exclusive
        // mass, zero, is not below it.
        if (target == 0ul) done = 1u;
      }
      st->done = done;
    }
  }
}

inline void m4_sel_hist_pass(
    device uchar* base, const device uchar* a0, const M1ValueDesc d0, uint pass,
    uint gtid, uint gthreads, uint tid, uint threads, threadgroup atomic_uint* tg) {
  if (d0.rows > M4_SEL_ROWS) return;
  const uint n = d0.last;
  const uint shift = 24u - 8u * pass;
  for (uint row = 0u; row < d0.rows; ++row) {
    const device M4SelState* st = m4_sel_state(base, row);
    if (st->done != 0u) continue;
    const uint prefix = st->prefix;
    for (uint b = tid; b < 768u; b += threads) atomic_store_explicit(&tg[b], 0u, memory_order_relaxed);
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint i = gtid; i < n; i += gthreads) {
      const float value = m1_load_f(a0, row * n + i, d0.dtype);
      const uint key = m1_desc_key(value);
      if (pass != 0u && (key >> (shift + 8u)) != (prefix >> (shift + 8u))) continue;
      const uint digit = (key >> shift) & 255u;
      atomic_fetch_add_explicit(&tg[digit], 1u, memory_order_relaxed);
      m4_sel_add(&tg[256u + digit], &tg[512u + digit], m4_sel_fixed(value));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    device atomic_uint* count = m4_sel_count(base, row);
    device atomic_uint* lo = m4_sel_lo(base, row);
    device atomic_uint* hi = m4_sel_hi(base, row);
    for (uint b = tid; b < 256u; b += threads) {
      const uint c = atomic_load_explicit(&tg[b], memory_order_relaxed);
      if (c == 0u) continue;
      atomic_fetch_add_explicit(&count[b], c, memory_order_relaxed);
      const ulong m = m4_sel_pair(atomic_load_explicit(&tg[256u + b], memory_order_relaxed),
                                  atomic_load_explicit(&tg[512u + b], memory_order_relaxed));
      m4_sel_add(&lo[b], &hi[b], m);
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// One thread walks a row's 256 buckets, in descending key, to the one the cut
// lies in, then clears the histograms for the next pass.
inline void m4_sel_pick(device uchar* base, const M1ValueDesc d0, uint mode, uint pass, uint tid, uint threads) {
  if (d0.rows > M4_SEL_ROWS) return;
  for (uint row = 0u; row < d0.rows; ++row) {
    device M4SelState* st = m4_sel_state(base, row);
    device atomic_uint* count = m4_sel_count(base, row);
    device atomic_uint* lo = m4_sel_lo(base, row);
    device atomic_uint* hi = m4_sel_hi(base, row);
    if (tid == 0u && st->done == 0u) {
      const uint shift = 24u - 8u * pass;
      uint before = st->before;
      M4Mass mass = m4_mass(st->before_lo, st->before_hi);
      const M4Mass target = m4_mass(st->target_lo, st->target_hi);
      uint digit = 256u;
      for (uint d = 0u; d < 256u; ++d) {
        const uint c = atomic_load_explicit(&count[d], memory_order_relaxed);
        const M4Mass m = m4_mass(atomic_load_explicit(&lo[d], memory_order_relaxed),
                                 atomic_load_explicit(&hi[d], memory_order_relaxed));
        // The cut lies in this bucket where its tokens reach the target:
        // top-k by count, top-p by mass (the bucket's last token's exclusive
        // mass stays below the target until its inclusive mass reaches it).
        const bool here = c != 0u &&
            (mode == 0u ? (before + c >= st->target_lo) : m4_mass_ge(m4_mass_add(mass, m), target));
        if (digit == 256u && here) digit = d;
        if (digit == 256u) {
          before += c;
          mass = m4_mass_add(mass, m);
        }
      }
      if (digit == 256u) {
        // Only at the first pass: the row's whole mass (or count) does not
        // reach the cut, so every token is taken. (Past it, the bucket in
        // play crossed at the pass before, so one of its digits does.)
        st->all = 1u;
        st->done = 1u;
      } else {
        st->prefix = (pass == 0u ? 0u : st->prefix) | (digit << shift);
        st->before = before;
        st->before_lo = mass.lo;
        st->before_hi = mass.hi;
        if (pass == 3u) {
          const uint ties = atomic_load_explicit(&count[digit], memory_order_relaxed);
          uint take = ties;
          if (mode == 0u) {
            take = st->target_lo - before;
          } else {
            // Every tie weighs the same: the mass of the cut key's value.
            const M4Mass v = m4_mass(m4_sel_fixed_lo(digit, st->prefix), m4_sel_fixed_hi(st->prefix));
            if (v.lo != 0u || v.hi != 0u) {
              // The j-th tie (from 0) is taken while mass + j * v is below
              // the target: the least j where it is not, by bisection.
              const M4Mass room = m4_mass_sub(target, mass);
              uint low = 0u, high = ties;
              while (low < high) {
                const uint mid = low + (high - low) / 2u;
                if (m4_mass_ge(m4_mass_mul(v, mid), room)) high = mid; else low = mid + 1u;
              }
              take = low;
            }
          }
          st->ties = ties;
          st->take = min(take, ties);
          st->done = 1u;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_device);
    for (uint b = tid; b < 256u; b += threads) {
      atomic_store_explicit(&count[b], 0u, memory_order_relaxed);
      atomic_store_explicit(&lo[b], 0u, memory_order_relaxed);
      atomic_store_explicit(&hi[b], 0u, memory_order_relaxed);
    }
  }
}

// Writes every token's mask: taken where its key comes before the cut key,
// or the cut key itself where all its ties are taken.
inline void m4_sel_mark(
    device uchar* base, const device uchar* a0, device uchar* o0, const M1ValueDesc d0,
    uint gtid, uint gthreads) {
  if (d0.rows > M4_SEL_ROWS) return;
  const uint n = d0.last;
  for (uint row = 0u; row < d0.rows; ++row) {
    const device M4SelState* st = m4_sel_state(base, row);
    const bool all = st->all != 0u;
    const bool none = !all && st->ties == 0u;
    const uint cut = st->prefix;
    const bool every_tie = st->take == st->ties;
    for (uint i = gtid; i < n; i += gthreads) {
      bool taken = all;
      if (!all && !none) {
        const uint key = m1_desc_key(m1_load_f(a0, row * n + i, d0.dtype));
        taken = key < cut || (key == cut && every_tie);
      }
      m1_store_b(o0, row * n + i, taken);
    }
  }
}

// Where only some of the cut key's ties are taken, takes the first `take` of
// them in index order: one threadgroup, each thread a contiguous chunk.
inline void m4_sel_ties(
    device uchar* base, const device uchar* a0, device uchar* o0, const M1ValueDesc d0,
    uint tid, uint threads, threadgroup uint* scan) {
  if (d0.rows > M4_SEL_ROWS) return;
  const uint n = d0.last;
  for (uint row = 0u; row < d0.rows; ++row) {
    const device M4SelState* st = m4_sel_state(base, row);
    if (st->all != 0u || st->ties == 0u || st->take == st->ties) continue;
    const uint cut = st->prefix;
    const uint take = st->take;
    const uint begin = uint((ulong(n) * tid) / threads);
    const uint end = uint((ulong(n) * (tid + 1u)) / threads);
    uint mine = 0u;
    for (uint i = begin; i < end; ++i)
      mine += m1_desc_key(m1_load_f(a0, row * n + i, d0.dtype)) == cut ? 1u : 0u;
    scan[tid] = mine;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      uint run = 0u;
      for (uint t = 0u; t < threads; ++t) {
        const uint here = scan[t];
        scan[t] = run;
        run += here;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint at = scan[tid];
    for (uint i = begin; i < end && at < take; ++i) {
      if (m1_desc_key(m1_load_f(a0, row * n + i, d0.dtype)) == cut) {
        m1_store_b(o0, row * n + i, true);
        ++at;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// The single-threadgroup top-k (mode 0) / top-p (mode 1) cut, the fused and
// grouped forms' select: the same rule and the same radix select as the
// streamed one (m4_sel_*), four passes over each row, ties in index order,
// its histograms and state in `tgbuf` (at least 5184 bytes: 768 histogram
// words, a scan word a thread, 16 state words).
inline void m1_cut_select_mt(
    const device uchar* a0,
    const device uchar* a1,
    device uchar* o0,
    const M1ValueDesc d0,
    const M1ValueDesc d1,
    uint mode,
    uint tid,
    uint nthreads,
    threadgroup M1ArgmaxCandidate* tgbuf) {
  threadgroup atomic_uint* hist = reinterpret_cast<threadgroup atomic_uint*>(tgbuf);
  threadgroup uint* scan = reinterpret_cast<threadgroup uint*>(tgbuf) + 768u;
  threadgroup uint* st = scan + nthreads;  // prefix, before, mass lo, mass hi, ties, take, all, done
  const uint n = d0.last;
  for (uint row = 0; row < d0.rows; ++row) {
    const uint base = row * n;
    uint k = 0u;
    ulong target64 = 0ul;
    if (mode == 0u) {
      const int signed_k = m1_load_i(a1, m1_pick(d1.len, row), d1.dtype);
      k = signed_k <= 0 ? 0u : uint(signed_k);
    } else {
      target64 = m4_sel_fixed(m1_load_f(a1, m1_pick(d1.len, row), d1.dtype));
    }
    const M4Mass target = m4_mass(uint(target64), uint(target64 >> 32));
    if (tid == 0u) {
      st[0] = 0u; st[1] = 0u; st[2] = 0u; st[3] = 0u; st[4] = 0u; st[5] = 0u;
      const bool every = mode == 0u && k >= n;
      st[6] = every ? 1u : 0u;
      st[7] = (n == 0u || every || (mode == 0u ? k == 0u : target64 == 0ul)) ? 1u : 0u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint pass = 0u; pass < 4u; ++pass) {
      if (st[7] != 0u) break;
      const uint shift = 24u - 8u * pass;
      const uint prefix = st[0];
      for (uint b = tid; b < 768u; b += nthreads) atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint i = tid; i < n; i += nthreads) {
        const float value = m1_load_f(a0, base + i, d0.dtype);
        const uint key = m1_desc_key(value);
        if (pass != 0u && (key >> (shift + 8u)) != (prefix >> (shift + 8u))) continue;
        const uint digit = (key >> shift) & 255u;
        atomic_fetch_add_explicit(&hist[digit], 1u, memory_order_relaxed);
        m4_sel_add(&hist[256u + digit], &hist[512u + digit], m4_sel_fixed(value));
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid == 0u) {
        uint before = st[1];
        M4Mass mass = m4_mass(st[2], st[3]);
        uint digit = 256u;
        for (uint d = 0u; d < 256u; ++d) {
          const uint c = atomic_load_explicit(&hist[d], memory_order_relaxed);
          const M4Mass m = m4_mass(atomic_load_explicit(&hist[256u + d], memory_order_relaxed),
                                   atomic_load_explicit(&hist[512u + d], memory_order_relaxed));
          const bool here = c != 0u &&
              (mode == 0u ? (before + c >= k) : m4_mass_ge(m4_mass_add(mass, m), target));
          if (digit == 256u && here) digit = d;
          if (digit == 256u) {
            before += c;
            mass = m4_mass_add(mass, m);
          }
        }
        if (digit == 256u) {
          st[6] = 1u;
          st[7] = 1u;
        } else {
          st[0] = (pass == 0u ? 0u : st[0]) | (digit << shift);
          st[1] = before;
          st[2] = mass.lo;
          st[3] = mass.hi;
          if (pass == 3u) {
            const uint ties = atomic_load_explicit(&hist[digit], memory_order_relaxed);
            uint take = ties;
            const M4Mass v = m4_mass(m4_sel_fixed_lo(digit, st[0]), m4_sel_fixed_hi(st[0]));
            if (mode == 0u) {
              take = k - before;
            } else if (v.lo != 0u || v.hi != 0u) {
              const M4Mass room = m4_mass_sub(target, mass);
              uint low = 0u, high = ties;
              while (low < high) {
                const uint mid = low + (high - low) / 2u;
                if (m4_mass_ge(m4_mass_mul(v, mid), room)) high = mid; else low = mid + 1u;
              }
              take = low;
            }
            st[4] = ties;
            st[5] = min(take, ties);
            st[7] = 1u;
          }
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const bool all = st[6] != 0u;
    const uint ties = st[4];
    const uint take = st[5];
    const uint cut = st[0];
    const bool none = !all && ties == 0u;
    const bool every_tie = take == ties;
    for (uint i = tid; i < n; i += nthreads) {
      bool taken = all;
      if (!all && !none) {
        const uint key = m1_desc_key(m1_load_f(a0, base + i, d0.dtype));
        taken = key < cut || (key == cut && every_tie);
      }
      m1_store_b(o0, base + i, taken);
    }
    threadgroup_barrier(mem_flags::mem_device);
    if (!all && ties != 0u && !every_tie) {
      const uint begin = uint((ulong(n) * tid) / nthreads);
      const uint end = uint((ulong(n) * (tid + 1u)) / nthreads);
      uint mine = 0u;
      for (uint i = begin; i < end; ++i)
        mine += m1_desc_key(m1_load_f(a0, base + i, d0.dtype)) == cut ? 1u : 0u;
      scan[tid] = mine;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (tid == 0u) {
        uint run = 0u;
        for (uint t = 0u; t < nthreads; ++t) {
          const uint here = scan[t];
          scan[t] = run;
          run += here;
        }
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      uint at = scan[tid];
      for (uint i = begin; i < end && at < take; ++i) {
        if (m1_desc_key(m1_load_f(a0, base + i, d0.dtype)) == cut) {
          m1_store_b(o0, base + i, true);
          ++at;
        }
      }
    }
    threadgroup_barrier(mem_flags::mem_device);
  }
}

inline void m1_nucleus_select_mt(
    const device uchar* a0, const device uchar* a1, device uchar* o0, const M1ValueDesc d0,
    const M1ValueDesc d1, uint tid, uint nthreads, threadgroup M1ArgmaxCandidate* tgbuf) {
  m1_cut_select_mt(a0, a1, o0, d0, d1, 1u, tid, nthreads, tgbuf);
}

inline void m1_topk_select_mt(
    const device uchar* a0, const device uchar* a1, device uchar* o0, const M1ValueDesc d0,
    const M1ValueDesc d1, uint tid, uint nthreads, threadgroup M1ArgmaxCandidate* tgbuf) {
  m1_cut_select_mt(a0, a1, o0, d0, d1, 0u, tid, nthreads, tgbuf);
}

inline void m4_argmax_partial(
    const device uchar* input,
    device uchar* temporary,
    const M1ValueDesc d,
    uint group,
    uint groups,
    uint tid,
    uint nthreads,
    threadgroup M1ArgmaxCandidate* tgbuf) {
  device M1ArgmaxCandidate* partials = reinterpret_cast<device M1ArgmaxCandidate*>(temporary);
  const uint span = groups * nthreads;
  const uint begin = group * nthreads + tid;
  for (uint row = 0; row < d.rows; ++row) {
    const uint base = row * d.last;
    M1ArgmaxCandidate best = {-INFINITY, 0u, 0u, 0u};
    for (uint i = begin; i < d.last; i += span) {
      const float value = m1_load_f(input, base + i, d.dtype);
      best = m1_argmax_combine(best, M1ArgmaxCandidate{value, i, isnan(value) ? 0u : 1u, 0u});
    }
    tgbuf[tid] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1u; stride < nthreads; stride <<= 1) {
      if ((tid % (2u * stride)) == 0u && tid + stride < nthreads)
        tgbuf[tid] = m1_argmax_combine(tgbuf[tid], tgbuf[tid + stride]);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) partials[row * groups + group] = tgbuf[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

inline void m4_argmax_final(
    const device uchar* temporary,
    device uchar* output,
    const M1ValueDesc d,
    uint groups,
    uint tid,
    uint nthreads,
    threadgroup M1ArgmaxCandidate* tgbuf) {
  const device M1ArgmaxCandidate* partials =
      reinterpret_cast<const device M1ArgmaxCandidate*>(temporary);
  device int* result = reinterpret_cast<device int*>(output);
  for (uint row = 0; row < d.rows; ++row) {
    M1ArgmaxCandidate best = {-INFINITY, 0u, 0u, 0u};
    for (uint j = tid; j < groups; j += nthreads)
      best = m1_argmax_combine(best, partials[row * groups + j]);
    tgbuf[tid] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1u; stride < nthreads; stride <<= 1) {
      if ((tid % (2u * stride)) == 0u && tid + stride < nthreads)
        tgbuf[tid] = m1_argmax_combine(tgbuf[tid], tgbuf[tid + stride]);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) result[row] = int(tgbuf[0].index);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// `gumbel_argmax`'s first level across the grid: each group's best lane of
// each row, every lane made as `m1_gumbel_argmax_mt` makes it; the second
// level is `m4_argmax_final`.
inline void m4_gumbel_partial(
    const device uchar* logits,
    const device uchar* keep,
    const device uchar* divisor,
    const device uchar* state,
    device uchar* temporary,
    const M1ValueDesc dl,
    const M1ValueDesc dk,
    const M1ValueDesc dd,
    const M1ValueDesc ds,
    uint group,
    uint groups,
    uint tid,
    uint nthreads,
    threadgroup M1ArgmaxCandidate* tgbuf) {
  #pragma clang fp contract(off)
  device M1ArgmaxCandidate* partials = reinterpret_cast<device M1ArgmaxCandidate*>(temporary);
  const ulong key = ulong(m1_load_u(state, 0, ds.dtype));
  const ulong counter = ulong(ds.len > 1 ? m1_load_u(state, 1, ds.dtype) : 0u);
  const ulong seed = ptir_rng_keyed_seed(uint(key), uint(counter));
  // The draw multiplies by its scale (the host's rounded 1 / T), never
  // divides: a multiply is correctly rounded on every backend. The product
  // is its own statement and nothing contracts it, so the lane is the
  // interpreter's two roundings, `logit * scale` then `+ noise`.
  const float scale = m1_load_f(divisor, 0, dd.dtype);
  const uint span = groups * nthreads;
  const uint begin = group * nthreads + tid;
  for (uint row = 0; row < dl.rows; ++row) {
    const uint base = row * dl.last;
    M1ArgmaxCandidate best = {-INFINITY, 0u, 0u, 0u};
    for (uint i = begin; i < dl.last; i += span) {
      const uint flat = base + i;
      float value = -INFINITY;
      if (m1_load_b(keep, m1_pick(dk.len, flat), dk.dtype)) {
        const float uniform = ptir_rng_hash_uniform(seed, flat);
        const float noise = -precise::log(-precise::log(uniform));
        const float scaled = m1_load_f(logits, flat, dl.dtype) * scale;
        value = scaled + noise;
      }
      best = m1_argmax_combine(best, M1ArgmaxCandidate{value, i, isnan(value) ? 0u : 1u, 0u});
    }
    tgbuf[tid] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1u; stride < nthreads; stride <<= 1) {
      if ((tid % (2u * stride)) == 0u && tid + stride < nthreads)
        tgbuf[tid] = m1_argmax_combine(tgbuf[tid], tgbuf[tid + stride]);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) partials[row * groups + group] = tgbuf[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

// `gumbel_argmax`: each row's index of its largest `logits[i] * scale +
// gumbel(state, i)`, a lane whose `keep` is false `-inf`. Every lane is
// made in f32 from its operands and never stored: the noise is lane `i`
// (the flat index) of `rng_keyed(state, logits' shape, Gumbel)`, made with
// the same operations as the keyed rng's Gumbel lane, so over f32 logits
// the draw is bit for bit `reduce_argmax(select(keep, logits * scale,
// -inf) + rng_keyed(..))`. Ties and NaN as `reduce_argmax`'s.
inline void m1_gumbel_argmax_mt(
    const device uchar* logits,
    const device uchar* keep,
    const device uchar* divisor,
    const device uchar* state,
    device uchar* output,
    const M1ValueDesc dl,
    const M1ValueDesc dk,
    const M1ValueDesc dd,
    const M1ValueDesc ds,
    uint tid,
    uint nthreads,
    threadgroup M1ArgmaxCandidate* tgbuf) {
  #pragma clang fp contract(off)
  const ulong key = ulong(m1_load_u(state, 0, ds.dtype));
  const ulong counter = ulong(ds.len > 1 ? m1_load_u(state, 1, ds.dtype) : 0u);
  const ulong seed = ptir_rng_keyed_seed(uint(key), uint(counter));
  // The draw multiplies by its scale (the host's rounded 1 / T), never
  // divides: a multiply is correctly rounded on every backend. The product
  // is its own statement and nothing contracts it, so the lane is the
  // interpreter's two roundings, `logit * scale` then `+ noise`.
  const float scale = m1_load_f(divisor, 0, dd.dtype);
  device int* result = reinterpret_cast<device int*>(output);
  for (uint row = 0; row < dl.rows; ++row) {
    const uint base = row * dl.last;
    M1ArgmaxCandidate best = {-INFINITY, 0u, 0u, 0u};
    for (uint i = tid; i < dl.last; i += nthreads) {
      const uint flat = base + i;
      float value = -INFINITY;
      if (m1_load_b(keep, m1_pick(dk.len, flat), dk.dtype)) {
        const float uniform = ptir_rng_hash_uniform(seed, flat);
        const float noise = -precise::log(-precise::log(uniform));
        const float scaled = m1_load_f(logits, flat, dl.dtype) * scale;
        value = scaled + noise;
      }
      best = m1_argmax_combine(
          best, M1ArgmaxCandidate{value, i, isnan(value) ? 0u : 1u, 0u});
    }
    tgbuf[tid] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1u; stride < nthreads; stride <<= 1) {
      if ((tid % (2u * stride)) == 0u && tid + stride < nthreads) {
        tgbuf[tid] = m1_argmax_combine(tgbuf[tid], tgbuf[tid + stride]);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    if (tid == 0) result[row] = int(tgbuf[0].index);
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
}

inline void ptir_m1_execute_mt(
    uint generated_tag,
    device M1Status* status,
    const device M1ValueDesc* descriptors,
    const device M1OpParams* params,
    const device uchar* a0,
    const device uchar* a1,
    const device uchar* a2,
    device uchar* o0,
    device uchar* o1,
    device uchar* temporary,
    uint tid,
    uint nthreads,
    threadgroup M1ArgmaxCandidate* tgbuf) {
  if (status->state != 1) return;
  M1OpParams p = params[0];
  p.tag = generated_tag;
  const M1ValueDesc d0 = descriptors[p.a0];

  if (p.tag == 0x33) {
    m1_reduce_argmax_mt(a0, o0, temporary, d0, tid, nthreads, tgbuf);
    return;
  }
  if (p.tag == 0x34) {
    // The rng state, the fourth operand, is named by `imm` and its bytes
    // ride in `o1` (the op has one result).
    m1_gumbel_argmax_mt(a0, a1, a2, o1, o0, d0, descriptors[p.a1], descriptors[p.a2],
                        descriptors[p.imm], tid, nthreads, tgbuf);
    return;
  }
  if (p.tag == 0x58 && nthreads > 1u && p.pred_tag != 2) {

    const M1ValueDesc d1 = descriptors[p.a1];
    if (p.pred_tag == 0) m1_topk_select_mt(a0, a1, o0, d0, d1, tid, nthreads, tgbuf);
    else m1_nucleus_select_mt(a0, a1, o0, d0, d1, tid, nthreads, tgbuf);
    return;
  }
  ptir_m1_execute_part(generated_tag, status, descriptors, params, a0, a1, a2, o0,
                       o1, temporary, tid, nthreads);
}
