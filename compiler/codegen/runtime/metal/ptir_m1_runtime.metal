#include <metal_stdlib>
#include "ptir_rng.generated.metal"
using namespace metal;

struct M1Status {
  uint state;  // 0 unset, 1 ready/running, 2 retry, 3 fault, 4 committed
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

inline float m1_load_f(const device uchar* data, uint index, uint dtype) {
  if (dtype == 0) return reinterpret_cast<const device float*>(data)[index];
  if (dtype == 1) return float(reinterpret_cast<const device int*>(data)[index]);
  if (dtype == 2) return float(reinterpret_cast<const device uint*>(data)[index]);
  return data[index] != 0 ? 1.0f : 0.0f;
}

inline int m1_load_i(const device uchar* data, uint index, uint dtype) {
  if (dtype == 0) return int(reinterpret_cast<const device float*>(data)[index]);
  if (dtype == 1) return reinterpret_cast<const device int*>(data)[index];
  if (dtype == 2) return int(reinterpret_cast<const device uint*>(data)[index]);
  return data[index] != 0 ? 1 : 0;
}

inline uint m1_load_u(const device uchar* data, uint index, uint dtype) {
  if (dtype == 0) return uint(reinterpret_cast<const device float*>(data)[index]);
  if (dtype == 1) return uint(reinterpret_cast<const device int*>(data)[index]);
  if (dtype == 2) return reinterpret_cast<const device uint*>(data)[index];
  return data[index] != 0 ? 1u : 0u;
}

inline bool m1_load_b(const device uchar* data, uint index, uint dtype) {
  if (dtype == 0) return reinterpret_cast<const device float*>(data)[index] != 0.0f;
  if (dtype == 1) return reinterpret_cast<const device int*>(data)[index] != 0;
  if (dtype == 2) return reinterpret_cast<const device uint*>(data)[index] != 0u;
  return data[index] != 0;
}

inline void m1_store_f(device uchar* data, uint index, float value) {
  reinterpret_cast<device float*>(data)[index] = value;
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

inline uint m1_pick(uint len, uint index) {
  return len == 1 ? 0u : index;
}

inline void m1_fault(device M1Status* status, uint code) {
  status->fault = code;
  status->state = 3;
}

inline void m1_copy_typed(
    const device uchar* input,
    device uchar* output,
    uint len,
    uint dtype) {
  if (dtype == 3) {
    for (uint i = 0; i < len; ++i) output[i] = input[i];
  } else {
    for (uint i = 0; i < len * 4; ++i) output[i] = input[i];
  }
}

inline void m1_reduce_float(
    uint tag,
    const device uchar* input,
    device uchar* output,
    device uchar* temporary,
    const M1ValueDesc in_desc) {
  device float* work = reinterpret_cast<device float*>(temporary);
  const device float* values = reinterpret_cast<const device float*>(input);
  device float* result = reinterpret_cast<device float*>(output);
  for (uint row = 0; row < in_desc.rows; ++row) {
    const uint base = row * in_desc.last;
    for (uint i = 0; i < in_desc.last; ++i) work[i] = values[base + i];
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
  device int* result = reinterpret_cast<device int*>(output);
  if (in_desc.dtype != 0) {
    device M1IntArgmaxCandidate* work =
        reinterpret_cast<device M1IntArgmaxCandidate*>(temporary);
    for (uint row = 0; row < in_desc.rows; ++row) {
      const uint base = row * in_desc.last;
      for (uint i = 0; i < in_desc.last; ++i) {
        work[i] = {
            m1_load_index(input, base + i, in_desc.dtype), i, 1u};
      }
      uint count = in_desc.last;
      if (count == 0) {
        result[row] = 0;
        continue;
      }
      while (count > 1) {
        const uint chunks = (count + 31) / 32;
        for (uint chunk = 0; chunk < chunks; ++chunk) {
          M1IntArgmaxCandidate lanes[32];
          for (uint lane = 0; lane < 32; ++lane) {
            const uint index = chunk * 32 + lane;
            lanes[lane] =
                index < count
                    ? work[index]
                    : M1IntArgmaxCandidate{0l, 0u, 0u};
          }
          for (uint offset = 16; offset > 0; offset >>= 1)
            for (uint lane = 0; lane < offset; ++lane)
              lanes[lane] = m1_int_argmax_combine(
                  lanes[lane], lanes[lane + offset]);
          work[chunk] = lanes[0];
        }
        count = chunks;
      }
      result[row] = int(work[0].index);
    }
    return;
  }

  const device float* values =
      reinterpret_cast<const device float*>(input);
  device M1ArgmaxCandidate* work =
      reinterpret_cast<device M1ArgmaxCandidate*>(temporary);
  for (uint row = 0; row < in_desc.rows; ++row) {
    const uint base = row * in_desc.last;
    for (uint i = 0; i < in_desc.last; ++i) {
      const float value = values[base + i];
      work[i] = {value, i, isnan(value) ? 0u : 1u, 0u};
    }
    uint count = in_desc.last;
    if (count == 0) {
      result[row] = 0;
      continue;
    }
    while (count > 1) {
      const uint chunks = (count + 31) / 32;
      for (uint chunk = 0; chunk < chunks; ++chunk) {
        M1ArgmaxCandidate lanes[32];
        for (uint lane = 0; lane < 32; ++lane) {
          const uint index = chunk * 32 + lane;
          lanes[lane] = index < count
                            ? work[index]
                            : M1ArgmaxCandidate{-INFINITY, 0u, 0u, 0u};
        }
        for (uint offset = 16; offset > 0; offset >>= 1)
          for (uint lane = 0; lane < offset; ++lane)
            lanes[lane] =
                m1_argmax_combine(lanes[lane], lanes[lane + offset]);
        work[chunk] = lanes[0];
      }
      count = chunks;
    }
    result[row] = int(work[0].index);
  }
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
  if (status->state != 1) return;
  M1OpParams p = params[0];
  p.tag = generated_tag;
  const M1ValueDesc d0 = descriptors[p.a0];
  const M1ValueDesc d1 = descriptors[p.a1];
  const M1ValueDesc d2 = descriptors[p.a2];
  const M1ValueDesc out0 = descriptors[p.o0];

  if (p.tag == 0x81) {  // const
    for (uint i = 0; i < out0.len; ++i) {
      if (p.lit_dtype == 0) m1_store_f(o0, i, as_type<float>(p.lit_bits));
      else if (p.lit_dtype == 1) m1_store_i(o0, i, int(p.lit_bits));
      else if (p.lit_dtype == 2) m1_store_u(o0, i, p.lit_bits);
      else m1_store_b(o0, i, p.lit_bits != 0);
    }
    return;
  }
  if (p.tag == 0x90 || p.tag == 0x91) {  // channel root
    if (out0.dtype == 3) {
      for (uint i = 0; i < out0.len; ++i)
        o0[i] = (a0[i >> 3] >> (i & 7)) & 1u;
    } else {
      m1_copy_typed(a0, o0, out0.len, out0.dtype);
    }
    return;
  }
  if (p.tag == 0x92) {  // direct channel sink
    const uint logical_bytes =
        d0.dtype == 3 ? (d0.len + 7u) / 8u : d0.len * 4u;
    if (logical_bytes > p.sink_bytes) {
      m1_fault(status, p.tag);
      return;
    }
    if (d0.dtype == 3) {
      for (uint i = 0; i < logical_bytes; ++i) o0[i] = 0;
      for (uint i = 0; i < d0.len; ++i)
        if (a0[i] != 0) o0[i >> 3] |= uchar(1u << (i & 7));
    } else {
      m1_copy_typed(a0, o0, d0.len, d0.dtype);
    }
    for (uint i = logical_bytes; i < p.sink_bytes; ++i) o0[i] = 0;
    return;
  }
  if (p.tag == 0xA0) {  // intrinsic logits staging is bf16
    const device bfloat* logits =
        reinterpret_cast<const device bfloat*>(a0) +
        ulong(p.imm2) * p.imm;
    if (p.intr == 6u) {  // MtpDrafts: bounded argmax of the bound MTP rows
      if (p.imm == 0u) {
        m1_fault(status, p.tag);
        return;
      }
      for (uint row = 0; row < out0.len; ++row) {
        float best_value = -INFINITY;
        uint best_index = 0u;
        bool have = false;
        for (uint column = 0; column < p.imm; ++column) {
          const float value = float(logits[ulong(row) * p.imm + column]);
          if (!isnan(value) &&
              (!have || value > best_value ||
               (value == best_value && column < best_index))) {
            best_value = value;
            best_index = column;
            have = true;
          }
        }
        m1_store_i(o0, row, int(have ? best_index : 0u));
      }
      return;
    }
    for (uint i = 0; i < out0.len; ++i) m1_store_f(o0, i, float(logits[i]));
    return;
  }
  if (p.tag == 0xA1) {  // explicit Metal semantic boundary: identity
    m1_copy_typed(a0, o0, out0.len, out0.dtype);
    return;
  }
  if (p.tag == 0xA2) {  // explicit Metal semantic boundary: discard sink
    return;
  }

  if (p.tag == 0x01 || p.tag == 0x02 || p.tag == 0x04) {
    for (uint i = 0; i < out0.len; ++i) {
      const float value = m1_load_f(a0, m1_pick(d0.len, i), d0.dtype);
      if (p.tag == 0x01) m1_store_f(o0, i, precise::exp(value));
      else if (p.tag == 0x02) m1_store_f(o0, i, precise::log(value));
      else m1_store_f(o0, i, 1.0f / value);
    }
    return;
  }
  if (p.tag == 0x03 || p.tag == 0x05 || p.tag == 0x06) {
    for (uint i = 0; i < out0.len; ++i) {
      const uint source = m1_pick(d0.len, i);
      if (d0.dtype == 0) {
        const float value = m1_load_f(a0, source, d0.dtype);
        const float result =
            p.tag == 0x03 ? -value
                          : (p.tag == 0x05
                                 ? abs(value)
                                 : (value > 0 ? 1.0f : (value < 0 ? -1.0f : 0.0f)));
        m1_store_f(o0, i, result);
      } else if (d0.dtype == 1) {
        const int value = m1_load_i(a0, source, d0.dtype);
        int result = value;
        if (p.tag == 0x03) result = int(0u - uint(value));
        else if (p.tag == 0x05) result = value == INT_MIN ? value : abs(value);
        else result = value > 0 ? 1 : (value < 0 ? -1 : 0);
        m1_store_i(o0, i, result);
      } else if (d0.dtype == 2) {
        const uint value = m1_load_u(a0, source, d0.dtype);
        m1_store_u(
            o0, i,
            p.tag == 0x03 ? 0u - value
                          : (p.tag == 0x06 ? (value != 0 ? 1u : 0u) : value));
      } else {
        m1_fault(status, p.tag);
        return;
      }
    }
    return;
  }
  if (p.tag == 0x07) {  // cast
    for (uint i = 0; i < out0.len; ++i) {
      const uint source = m1_pick(d0.len, i);
      if (out0.dtype == 0) m1_store_f(o0, i, m1_load_f(a0, source, d0.dtype));
      else if (out0.dtype == 1) m1_store_i(o0, i, m1_load_i(a0, source, d0.dtype));
      else if (out0.dtype == 2) m1_store_u(o0, i, m1_load_u(a0, source, d0.dtype));
      else m1_store_b(o0, i, m1_load_b(a0, source, d0.dtype));
    }
    return;
  }

  if ((p.tag >= 0x10 && p.tag <= 0x1D) || p.tag == 0x1F) {
    for (uint i = 0; i < out0.len; ++i) {
      const uint xindex = m1_pick(d0.len, i), yindex = m1_pick(d1.len, i);
      if (p.tag >= 0x16 && p.tag <= 0x1D) {
        bool result = false;
        if (p.tag == 0x1C || p.tag == 0x1D) {
          const bool x = m1_load_b(a0, xindex, d0.dtype);
          const bool y = m1_load_b(a1, yindex, d1.dtype);
          result = p.tag == 0x1C ? x && y : x || y;
        } else if (d0.dtype == 0) {
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
      } else if (d0.dtype == 0) {
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
        m1_store_f(o0, i, result);
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
    for (uint i = 0; i < out0.len; ++i)
      m1_store_b(o0, i, !m1_load_b(a0, m1_pick(d0.len, i), d0.dtype));
    return;
  }
  if (p.tag == 0x20) {
    for (uint i = 0; i < out0.len; ++i) {
      const bool select = m1_load_b(a0, m1_pick(d0.len, i), d0.dtype);
      const uint xi = m1_pick(d1.len, i), yi = m1_pick(d2.len, i);
      if (out0.dtype == 0)
        m1_store_f(o0, i, select ? m1_load_f(a1, xi, d1.dtype)
                                 : m1_load_f(a2, yi, d2.dtype));
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
    if (d0.dtype == 0) m1_reduce_float(p.tag, a0, o0, temporary, d0);
    else m1_reduce_integer(p.tag, a0, o0, temporary, d0);
    return;
  }
  if (p.tag == 0x33) {
    m1_reduce_argmax(a0, o0, temporary, d0);
    return;
  }
  if (p.tag == 0x38) {  // left-aligned broadcast
    for (uint linear = 0; linear < out0.len; ++linear) {
      uint rem = linear, source_index = 0;
      uint source_stride[4] = {1, 1, 1, 1};
      for (int dim = int(out0.rank) - 2; dim >= 0; --dim)
        source_stride[dim] =
            source_stride[dim + 1] * (uint(dim + 1) < d0.rank ? d0.dims[dim + 1] : 1u);
      for (uint dim = 0; dim < out0.rank; ++dim) {
        uint stride = 1;
        for (uint next = dim + 1; next < out0.rank; ++next)
          stride *= out0.dims[next];
        const uint coordinate = rem / max(stride, 1u);
        rem %= max(stride, 1u);
        const uint source_dim = dim < d0.rank ? d0.dims[dim] : 1u;
        if (source_dim != 1) source_index += coordinate * source_stride[dim];
      }
      if (out0.dtype == 0) m1_store_f(o0, linear, m1_load_f(a0, source_index, d0.dtype));
      else if (out0.dtype == 1) m1_store_i(o0, linear, m1_load_i(a0, source_index, d0.dtype));
      else if (out0.dtype == 2) m1_store_u(o0, linear, m1_load_u(a0, source_index, d0.dtype));
      else m1_store_b(o0, linear, m1_load_b(a0, source_index, d0.dtype));
    }
    return;
  }
  if (p.tag == 0x39) {
    m1_copy_typed(a0, o0, out0.len, out0.dtype);
    return;
  }
  if (p.tag == 0x3A) {
    if (d0.rank != 2) {
      m1_fault(status, p.tag);
      return;
    }
    const uint m = d0.dims[0], n = d0.dims[1];
    for (uint index = 0; index < m * n; ++index) {
      const uint source = (index % m) * n + index / m;
      if (out0.dtype == 0) m1_store_f(o0, index, m1_load_f(a0, source, d0.dtype));
      else if (out0.dtype == 1) m1_store_i(o0, index, m1_load_i(a0, source, d0.dtype));
      else if (out0.dtype == 2) m1_store_u(o0, index, m1_load_u(a0, source, d0.dtype));
      else m1_store_b(o0, index, m1_load_b(a0, source, d0.dtype));
    }
    return;
  }
  if (p.tag == 0x40 || p.tag == 0x41) {
    for (uint row = 0; row < d0.rows; ++row) {
      float accumulated = p.tag == 0x40 ? 0.0f : 1.0f;
      for (uint column = 0; column < d0.last; ++column) {
        const uint index = row * d0.last + column;
        const float value = m1_load_f(a0, index, d0.dtype);
        accumulated = p.tag == 0x40 ? accumulated + value : accumulated * value;
        m1_store_f(o0, index, accumulated);
      }
    }
    return;
  }
  if (p.tag == 0x50) {
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
      m1_store_f(o0, position, best_value);
      m1_store_u(o1, position, best_index);
    }
    return;
  }
  if (p.tag == 0x51) {
    const uint count = min(p.imm, d0.last);
    for (uint row = 0; row < d0.rows; ++row) {
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
        m1_store_f(o0, row * count + position, best_value);
        m1_store_u(o1, row * count + position, best_index);
      }
    }
    return;
  }
  if (p.tag == 0x55) {
    if (d0.rank != 2 || d1.rank != 2) {
      m1_fault(status, p.tag);
      return;
    }
    const uint m = d0.dims[0], inner = d0.dims[1], n = d1.dims[1];
    for (uint row = 0; row < m; ++row)
      for (uint column = 0; column < n; ++column)
        m1_store_f(o0, row * n + column, 0.0f);
    for (uint row = 0; row < m; ++row)
      for (uint k = 0; k < inner; ++k) {
        const float left = m1_load_f(a0, row * inner + k, d0.dtype);
        if (left == 0.0f) continue;
        for (uint column = 0; column < n; ++column) {
          const uint index = row * n + column;
          const float old = m1_load_f(o0, index, 0);
          m1_store_f(o0, index, old + left * m1_load_f(a1, k * n + column, d1.dtype));
        }
      }
    return;
  }
  if (p.tag == 0x58) {
    for (uint row = 0; row < d0.rows; ++row) {
      const uint base = row * d0.last;
      if (p.pred_tag == 0) {
        int k = m1_load_i(a1, m1_pick(d1.len, row), d1.dtype);
        k = clamp(k, 0, int(d0.last));
        for (uint i = 0; i < d0.last; ++i) {
          const float value = m1_load_f(a0, base + i, d0.dtype);
          int greater = 0;
          if (!isnan(value))
            for (uint j = 0; j < d0.last; ++j) {
              const float other = m1_load_f(a0, base + j, d0.dtype);
              if (!isnan(other) && other > value) ++greater;
            }
          m1_store_b(o0, base + i, !isnan(value) && greater < k);
        }
      } else if (p.pred_tag == 1) {
        const float threshold = m1_load_f(a1, m1_pick(d1.len, row), d1.dtype);
        float exclusive = 0.0f;
        for (uint position = 0; position < d0.last; ++position) {
          uint best_index = 0;
          float best_value = NAN;
          bool found = false;
          for (uint candidate = 0; candidate < d0.last; ++candidate) {
            bool used = false;
            for (uint prior = 0; prior < position; ++prior)
              if (reinterpret_cast<device uint*>(temporary)[prior] == candidate) used = true;
            if (used) continue;
            const float value = m1_load_f(a0, base + candidate, d0.dtype);
            if (!found || m1_sort_better(value, candidate, best_value, best_index)) {
              found = true;
              best_value = value;
              best_index = candidate;
            }
          }
          reinterpret_cast<device uint*>(temporary)[position] = best_index;
          m1_store_b(o0, base + best_index, exclusive < threshold);
          exclusive += best_value;
        }
      } else {
        const float threshold = m1_load_f(a1, m1_pick(d1.len, row), d1.dtype);
        for (uint i = 0; i < d0.last; ++i)
          m1_store_b(
              o0, base + i,
              m1_load_f(a0, base + i, d0.dtype) >= threshold);
      }
    }
    return;
  }
  if (p.tag == 0x60) {
    const uint rest = d0.rank == 0 ? 1u : d0.len / max(d0.dims[0], 1u);
    const uint n0 = d0.rank == 0 ? 1u : d0.dims[0];
    for (uint k = 0; k < d1.len; ++k) {
      const long index = m1_load_index(a1, k, d1.dtype);
      for (uint r = 0; r < rest; ++r) {
        const uint output_index = k * rest + r;
        const bool valid = index >= 0 && uint(index) < n0;
        const uint source = valid ? uint(index) * rest + r : 0;
        if (out0.dtype == 0) m1_store_f(o0, output_index, valid ? m1_load_f(a0, source, d0.dtype) : 0.0f);
        else if (out0.dtype == 1) m1_store_i(o0, output_index, valid ? m1_load_i(a0, source, d0.dtype) : 0);
        else if (out0.dtype == 2) m1_store_u(o0, output_index, valid ? m1_load_u(a0, source, d0.dtype) : 0u);
        else m1_store_b(o0, output_index, valid && m1_load_b(a0, source, d0.dtype));
      }
    }
    return;
  }
  if (p.tag == 0x61) {
    const uint rows = d0.dims[0], columns = d0.dims[1];
    for (uint row = 0; row < rows; ++row) {
      const long column = m1_load_index(a1, row, d1.dtype);
      const bool valid = column >= 0 && uint(column) < columns;
      const uint source = valid ? row * columns + uint(column) : 0;
      if (out0.dtype == 0) m1_store_f(o0, row, valid ? m1_load_f(a0, source, d0.dtype) : 0.0f);
      else if (out0.dtype == 1) m1_store_i(o0, row, valid ? m1_load_i(a0, source, d0.dtype) : 0);
      else if (out0.dtype == 2) m1_store_u(o0, row, valid ? m1_load_u(a0, source, d0.dtype) : 0u);
      else m1_store_b(o0, row, valid && m1_load_b(a0, source, d0.dtype));
    }
    return;
  }
  if (p.tag == 0x62 || p.tag == 0x63) {
    m1_copy_typed(a0, o0, d0.len, d0.dtype);
    const uint rest = d0.rank == 0 ? 1u : d0.len / max(d0.dims[0], 1u);
    const uint n0 = d0.rank == 0 ? 1u : d0.dims[0];
    const bool scalar = d2.len == 1 && d1.len * rest != 1;
    for (uint k = 0; k < d1.len; ++k) {
      const long index = m1_load_index(a1, k, d1.dtype);
      if (index < 0 || uint(index) >= n0) continue;
      for (uint r = 0; r < rest; ++r) {
        const uint dst = uint(index) * rest + r;
        const uint src = scalar ? 0u : k * rest + r;
        if (d0.dtype == 0) {
          const float value = m1_load_f(a2, src, d2.dtype);
          m1_store_f(o0, dst, p.tag == 0x62 ? m1_load_f(o0, dst, 0) + value : value);
        } else if (d0.dtype == 1) {
          const int value = m1_load_i(a2, src, d2.dtype);
          m1_store_i(o0, dst, p.tag == 0x62 ? int(uint(m1_load_i(o0, dst, 1)) + uint(value)) : value);
        } else if (d0.dtype == 2) {
          const uint value = m1_load_u(a2, src, d2.dtype);
          m1_store_u(o0, dst, p.tag == 0x62 ? m1_load_u(o0, dst, 2) + value : value);
        } else {
          const bool value = m1_load_b(a2, src, d2.dtype);
          m1_store_b(o0, dst, value);
        }
      }
    }
    return;
  }
  if (p.tag == 0x64) {
    for (uint i = 0; i < out0.len; ++i) m1_store_u(o0, i, i);
    return;
  }
  if (p.tag == 0x65) {
    const uint mask_width =
        d0.rank == 0 ? 1u : d0.dims[d0.rank - 1];
    for (uint i = 0; i < d0.len; ++i) {
      const uint column = i % mask_width;
      const uint word = column >> 5;
      const uint mask = word < d1.len ? m1_load_u(a1, word, d1.dtype) : 0u;
      m1_store_f(
          o0, i,
          ((mask >> (column & 31)) & 1u) != 0
              ? m1_load_f(a0, i, d0.dtype)
              : -INFINITY);
    }
    return;
  }
  if (p.tag == 0x66 || p.tag == 0x67 || p.tag == 0x68) {
    const uint key_count = p.imm;
    const uint window = p.tag == 0x67 ? p.imm2 : p.imm3;
    for (uint index = 0; index < out0.len; ++index) {
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
    for (uint i = 0; i < out0.len; ++i) {
      const float uniform = ptir_rng_hash_uniform(seed, i);
      m1_store_f(
          o0, i,
          p.kind == 0 ? uniform : -precise::log(-precise::log(uniform)));
    }
    return;
  }

  m1_fault(status, p.tag);
}
