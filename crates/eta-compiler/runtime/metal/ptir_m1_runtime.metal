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

#define M1_CUMMASS_OFF_SLACK 1e-5f

inline bool m1_cummass_keeps_whole_row(float sum, float smallest, float threshold) {
  const float slack = threshold >= 1.0f ? M1_CUMMASS_OFF_SLACK : 0.0f;
  return (sum - smallest) < threshold + slack;
}

inline bool m1_cummass_keeps_row_serial(
    const device uchar* a0, uint base, uint last, uint dtype, float threshold) {
  if (last == 0u) return false;
  float sum = 0.0f;
  float smallest = INFINITY;
  for (uint i = 0; i < last; ++i) {
    const float value = m1_load_f(a0, base + i, dtype);
    sum += value;
    smallest = min(smallest, value);
  }
  return m1_cummass_keeps_whole_row(sum, smallest, threshold);
}

inline bool m1_cummass_keeps_row_mt(
    const device uchar* a0, uint base, uint last, uint dtype, float threshold,
    uint tid, uint nthreads, threadgroup M1ArgmaxCandidate* tgbuf) {
  if (last == 0u) return false;
  float sum = 0.0f;
  float smallest = INFINITY;
  for (uint i = tid; i < last; i += nthreads) {
    const float value = m1_load_f(a0, base + i, dtype);
    sum += value;
    smallest = min(smallest, value);
  }
  tgbuf[tid] = M1ArgmaxCandidate{sum, as_type<uint>(smallest), 1u, 0u};
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint stride = 1u; stride < nthreads; stride <<= 1) {
    if ((tid % (2u * stride)) == 0u && tid + stride < nthreads) {
      const M1ArgmaxCandidate other = tgbuf[tid + stride];
      tgbuf[tid].value += other.value;
      tgbuf[tid].index = as_type<uint>(min(as_type<float>(tgbuf[tid].index), as_type<float>(other.index)));
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }
  const bool keeps =
      m1_cummass_keeps_whole_row(tgbuf[0].value, as_type<float>(tgbuf[0].index), threshold);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  return keeps;
}

inline uint m1_desc_key(float value) {
  if (isnan(value)) return 0xFFFFFFFFu;
  if (value == 0.0f) value = 0.0f;
  const uint bits = as_type<uint>(value);
  const uint ascending = (bits & 0x80000000u) != 0u ? ~bits : (bits | 0x80000000u);
  return ~ascending;
}

inline uint m1_pick(uint len, uint index) {
  return len == 1 ? 0u : index;
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
  (void)temporary;
  device int* result = reinterpret_cast<device int*>(output);
  if (in_desc.dtype != 0) {
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

  const device float* values =
      reinterpret_cast<const device float*>(input);
  for (uint row = 0; row < in_desc.rows; ++row) {
    const uint base = row * in_desc.last;

    M1ArgmaxCandidate best[4] = {
        {-INFINITY, 0u, 0u, 0u}, {-INFINITY, 0u, 0u, 0u},
        {-INFINITY, 0u, 0u, 0u}, {-INFINITY, 0u, 0u, 0u}};
    uint i = 0;
    for (; i + 4u <= in_desc.last; i += 4u) {
      const float v0 = values[base + i + 0], v1 = values[base + i + 1];
      const float v2 = values[base + i + 2], v3 = values[base + i + 3];
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
      const float value = values[base + i];
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
  if (in_desc.dtype != 0) {

    if (tid == 0) m1_reduce_argmax(input, output, temporary, in_desc);
    return;
  }
  device int* result = reinterpret_cast<device int*>(output);
  const device float* values = reinterpret_cast<const device float*>(input);
  for (uint row = 0; row < in_desc.rows; ++row) {
    const uint base = row * in_desc.last;
    M1ArgmaxCandidate best = {-INFINITY, 0u, 0u, 0u};
    for (uint i = tid; i < in_desc.last; i += nthreads) {
      const float value = values[base + i];
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
  const device float* values = reinterpret_cast<const device float*>(input);
  device float* result = reinterpret_cast<device float*>(output);
  for (uint row = 0; row < in_desc.rows; ++row) {
    const uint base = row * in_desc.last;
    uint count = in_desc.last;
    if (count == 0) {
      if (tid == 0) result[row] = identity;
      continue;
    }

    const device float* src = values + base;
    device float* dst = plane_a;
    while (count > 1) {
      const uint chunks = (count + 31) / 32;
      for (uint chunk = tid; chunk < chunks; chunk += nthreads) {
        float lanes[32];
        for (uint lane = 0; lane < 32; ++lane) {
          const uint index = chunk * 32 + lane;
          lanes[lane] = index < count ? src[index] : identity;
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
      src = dst;
      dst = (dst == plane_a) ? plane_b : plane_a;
    }
    if (tid == 0) result[row] = src[0];
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
      if (d0.dtype == 0) {
        const float value = m1_load_f(a2, src, d2.dtype);
        m1_store_f(o0, dst, tag == 0x62 ? m1_load_f(o0, dst, 0) + value : value);
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
      if (p.lit_dtype == 0) m1_store_f(o0, i, as_type<float>(p.lit_bits));
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
    const uint logical_bytes =
        d0.dtype == 3 ? (d0.len + 7u) / 8u : d0.len * 4u;
    if (logical_bytes > p.sink_bytes) {
      if (tid == 0) m1_fault_op(status, 1u, p);
      return;
    }
    if (d0.dtype == 3) {

      if (tid != 0) return;
      for (uint i = 0; i < logical_bytes; ++i) o0[i] = 0;
      for (uint i = 0; i < d0.len; ++i)
        if (a0[i] != 0) o0[i >> 3] |= uchar(1u << (i & 7));
      for (uint i = logical_bytes; i < p.sink_bytes; ++i) o0[i] = 0;
      return;
    }
    m1_copy_typed_range(a0, o0, d0.len, d0.dtype, tid, nthreads);
    for (uint i = logical_bytes + tid; i < p.sink_bytes; i += nthreads) o0[i] = 0;
    return;
  }
  if (p.tag == 0xA0) {

    if (p.intr == 7u) {

      const device float* planes = reinterpret_cast<const device float*>(a0);
      device float* score_out = reinterpret_cast<device float*>(o0);
      for (uint i = tid; i < out0.len; i += nthreads) score_out[i] = planes[i];
      return;
    }
    const device bfloat* logits =
        reinterpret_cast<const device bfloat*>(a0) +
        ulong(p.imm2) * p.imm;
    if (p.intr == 6u) {
      if (p.imm == 0u) {
        if (tid == 0) m1_fault_op(status, 2u, p);
        return;
      }
      for (uint row = tid; row < out0.len; row += nthreads) {
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
      if (p.tag == 0x01) m1_store_f(o0, i, precise::exp(value));
      else if (p.tag == 0x02) m1_store_f(o0, i, precise::log(value));
      else m1_store_f(o0, i, 1.0f / value);
    }
    return;
  }
  if (p.tag >= 0x08 && p.tag <= 0x0B) {
    for (uint i = tid; i < out0.len; i += nthreads) {
      const float value = m1_load_f(a0, m1_pick(d0.len, i), d0.dtype);
      if (p.tag == 0x08) m1_store_f(o0, i, precise::sin(value));
      else if (p.tag == 0x09) m1_store_f(o0, i, precise::cos(value));
      else if (p.tag == 0x0A) m1_store_f(o0, i, sqrt(value));
      else m1_store_f(o0, i, 1.0f / sqrt(value));
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
      if (out0.dtype == 0) m1_store_f(o0, i, m1_load_f(a0, source, d0.dtype));
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
    for (uint i = tid; i < out0.len; i += nthreads)
      m1_store_b(o0, i, !m1_load_b(a0, m1_pick(d0.len, i), d0.dtype));
    return;
  }
  if (p.tag == 0x20) {
    for (uint i = tid; i < out0.len; i += nthreads) {
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
    if (d0.dtype == 0)
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
      if (out0.dtype == 0) m1_store_f(o0, linear, m1_load_f(a0, source_index, d0.dtype));
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
      if (out0.dtype == 0) m1_store_f(o0, index, m1_load_f(a0, source, d0.dtype));
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
          m1_store_f(o0, index, accumulated_f);
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
      m1_store_f(o0, position, best_value);
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
        m1_store_f(o0, row * count + position, best_value);
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
        m1_store_f(o0, row * n + column, 0.0f);
      for (uint k = 0; k < inner; ++k) {
        const float left = m1_load_f(a0, row * inner + k, d0.dtype);
        if (left == 0.0f) continue;
        for (uint column = 0; column < n; ++column) {
          const uint index = row * n + column;
          const float old = m1_load_f(o0, index, 0);
          m1_store_f(o0, index, old + left * m1_load_f(a1, k * n + column, d1.dtype));
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

    for (uint row = tid; row < d0.rows; row += nthreads) {
      const uint base = row * d0.last;
      if (p.pred_tag == 0) {
        int signed_k = m1_load_i(a1, m1_pick(d1.len, row), d1.dtype);
        uint k = signed_k <= 0 ? 0u : uint(signed_k);
        if (k > d0.last) k = d0.last;
        if (k == 0u) {
          for (uint i = 0; i < d0.last; ++i) m1_store_b(o0, base + i, false);
        } else {

          uint histogram[256];
          uint prefix = 0u;
          uint target = k;
          for (int pass = 0; pass < 4; ++pass) {
            const int shift = 24 - 8 * pass;

            const uint high_mask = (pass == 0) ? 0u : (0xFFFFFFFFu << (shift + 8));
            for (uint bucket = 0u; bucket < 256u; ++bucket) histogram[bucket] = 0u;
            for (uint j = 0; j < d0.last; ++j) {
              const uint key = m1_desc_key(m1_load_f(a0, base + j, d0.dtype));
              if ((key & high_mask) == (prefix & high_mask))
                ++histogram[(key >> shift) & 0xFFu];
            }
            uint run = 0u;
            uint chosen = 255u;
            for (uint bucket = 0u; bucket < 256u; ++bucket) {
              if (run + histogram[bucket] >= target) { chosen = bucket; break; }
              run += histogram[bucket];
            }
            target -= run;
            prefix |= chosen << shift;
          }
          for (uint i = 0; i < d0.last; ++i) {
            const float value = m1_load_f(a0, base + i, d0.dtype);
            m1_store_b(o0, base + i, !isnan(value) && m1_desc_key(value) <= prefix);
          }
        }
      } else {

        const float threshold = m1_load_f(a1, m1_pick(d1.len, row), d1.dtype);
        if (m1_cummass_keeps_row_serial(a0, base, d0.last, d0.dtype, threshold)) {
          for (uint i = 0; i < d0.last; ++i)
            m1_store_b(o0, base + i, !isnan(m1_load_f(a0, base + i, d0.dtype)));
          continue;
        }
        for (uint i = 0; i < d0.last; ++i) m1_store_b(o0, base + i, false);
        float exclusive = 0.0f;
        float prev_value = 0.0f;
        uint prev_index = 0;
        bool have_prev = false;
        for (uint position = 0; position < d0.last && exclusive < threshold; ++position) {
          uint best_index = 0;
          float best_value = 0.0f;
          bool found = false;
          for (uint candidate = 0; candidate < d0.last; ++candidate) {
            const float value = m1_load_f(a0, base + candidate, d0.dtype);
            if (have_prev && !m1_sort_better(prev_value, prev_index, value, candidate))
              continue;
            if (!found || m1_sort_better(value, candidate, best_value, best_index)) {
              found = true;
              best_value = value;
              best_index = candidate;
            }
          }
          if (!found) break;
          m1_store_b(o0, base + best_index, exclusive < threshold);
          exclusive += best_value;
          prev_value = best_value;
          prev_index = best_index;
          have_prev = true;
        }
      }
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
      if (out0.dtype == 0) m1_store_f(o0, output_index, valid ? m1_load_f(a0, source, d0.dtype) : 0.0f);
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
      if (out0.dtype == 0) m1_store_f(o0, row, valid ? m1_load_f(a0, source, d0.dtype) : 0.0f);
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
        m1_store_f(o0, i, ptir_rng_hash_normal(seed, i));
        continue;
      }
      const float uniform = ptir_rng_hash_uniform(seed, i);
      m1_store_f(
          o0, i,
          p.kind == 0 ? uniform : -precise::log(-precise::log(uniform)));
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

inline void m1_nucleus_select_mt(
    const device uchar* a0,
    const device uchar* a1,
    device uchar* o0,
    const M1ValueDesc d0,
    const M1ValueDesc d1,
    uint tid,
    uint nthreads,
    threadgroup M1ArgmaxCandidate* tgbuf) {
  for (uint row = 0; row < d0.rows; ++row) {
    const uint base = row * d0.last;
    const float threshold = m1_load_f(a1, m1_pick(d1.len, row), d1.dtype);
    if (m1_cummass_keeps_row_mt(a0, base, d0.last, d0.dtype, threshold, tid, nthreads, tgbuf)) {
      for (uint i = tid; i < d0.last; i += nthreads)
        m1_store_b(o0, base + i, !isnan(m1_load_f(a0, base + i, d0.dtype)));
      threadgroup_barrier(mem_flags::mem_device);
      continue;
    }
    for (uint i = tid; i < d0.last; i += nthreads) m1_store_b(o0, base + i, false);
    threadgroup_barrier(mem_flags::mem_device);
    float exclusive = 0.0f;
    float prev_value = 0.0f;
    uint prev_index = 0;
    bool have_prev = false;
    for (uint position = 0; position < d0.last && exclusive < threshold; ++position) {
      M1ArgmaxCandidate best = {0.0f, 0u, 0u, 0u};
      for (uint candidate = tid; candidate < d0.last; candidate += nthreads) {
        const float value = m1_load_f(a0, base + candidate, d0.dtype);
        if (have_prev && !m1_sort_better(prev_value, prev_index, value, candidate)) continue;
        best = m1_sort_pick(best, M1ArgmaxCandidate{value, candidate, 1u, 0u});
      }
      tgbuf[tid] = best;
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint stride = 1u; stride < nthreads; stride <<= 1) {
        if ((tid % (2u * stride)) == 0u && tid + stride < nthreads) {
          tgbuf[tid] = m1_sort_pick(tgbuf[tid], tgbuf[tid + stride]);
        }
        threadgroup_barrier(mem_flags::mem_threadgroup);
      }
      const M1ArgmaxCandidate found = tgbuf[0];
      threadgroup_barrier(mem_flags::mem_threadgroup);
      if (found.have == 0u) break;
      if (tid == 0) m1_store_b(o0, base + found.index, exclusive < threshold);
      exclusive += found.value;
      prev_value = found.value;
      prev_index = found.index;
      have_prev = true;
    }
    threadgroup_barrier(mem_flags::mem_device);
  }
}

inline void m1_topk_select_mt(
    const device uchar* a0,
    const device uchar* a1,
    device uchar* o0,
    const M1ValueDesc d0,
    const M1ValueDesc d1,
    uint tid,
    uint nthreads,
    threadgroup M1ArgmaxCandidate* tgbuf) {
  threadgroup atomic_uint* histogram = reinterpret_cast<threadgroup atomic_uint*>(tgbuf);
  for (uint row = 0; row < d0.rows; ++row) {
    const uint base = row * d0.last;
    const int signed_k = m1_load_i(a1, m1_pick(d1.len, row), d1.dtype);
    uint k = signed_k <= 0 ? 0u : uint(signed_k);
    if (k > d0.last) k = d0.last;
    if (k == 0u) {
      for (uint i = tid; i < d0.last; i += nthreads) m1_store_b(o0, base + i, false);
      continue;
    }
    uint prefix = 0u;
    uint target = k;
    for (int pass = 0; pass < 4; ++pass) {
      const int shift = 24 - 8 * pass;
      const uint high_mask = (pass == 0) ? 0u : (0xFFFFFFFFu << (shift + 8));
      for (uint bucket = tid; bucket < 256u; bucket += nthreads)
        atomic_store_explicit(histogram + bucket, 0u, memory_order_relaxed);
      threadgroup_barrier(mem_flags::mem_threadgroup);
      for (uint j = tid; j < d0.last; j += nthreads) {
        const uint key = m1_desc_key(m1_load_f(a0, base + j, d0.dtype));
        if ((key & high_mask) == (prefix & high_mask))
          atomic_fetch_add_explicit(histogram + ((key >> shift) & 0xFFu), 1u,
                                    memory_order_relaxed);
      }
      threadgroup_barrier(mem_flags::mem_threadgroup);
      uint run = 0u;
      uint chosen = 255u;
      for (uint bucket = 0u; bucket < 256u; ++bucket) {
        const uint count = atomic_load_explicit(histogram + bucket, memory_order_relaxed);
        if (run + count >= target) { chosen = bucket; break; }
        run += count;
      }
      target -= run;
      prefix |= chosen << shift;
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    for (uint i = tid; i < d0.last; i += nthreads) {
      const float value = m1_load_f(a0, base + i, d0.dtype);
      m1_store_b(o0, base + i, !isnan(value) && m1_desc_key(value) <= prefix);
    }
  }
}


inline uint m4_reduce_count(uint last, uint level) {
  uint count = last;
  for (uint l = 0; l < level; ++l) count = (count + 31u) / 32u;
  return count;
}

inline uint m4_reduce_identity(uint tag, uint dtype) {
  if (dtype == 0) return as_type<uint>(tag == 0x30 ? 0.0f : (tag == 0x31 ? -INFINITY : INFINITY));
  if (tag == 0x30) return 0u;
  if (dtype == 1) return tag == 0x31 ? uint(INT_MIN) : uint(INT_MAX);
  return tag == 0x31 ? 0u : UINT_MAX;
}

inline uint m4_fold_chunk(uint tag, uint dtype, uint bits) {
  if (dtype == 0) {
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
    uint simd_id) {
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
            ? reinterpret_cast<const device uint*>(input)[row * d.last]
            : plane[((level - 1u) & 1u) * d.rows * pitch + row * pitch];
      }
    }
    return;
  }
  const uint span = 32u * M4_REDUCE_CHUNKS_PER_GROUP;
  if (group * M4_REDUCE_CHUNKS_PER_GROUP >= next_chunks) return;
  threadgroup uint* tg = reinterpret_cast<threadgroup uint*>(tgbuf);
  const uint simds = threads / 32u;
  const uint rounds = span / simds;
  for (uint row = 0; row < d.rows; ++row) {
    const device uint* src = level == 0u
        ? reinterpret_cast<const device uint*>(input) + row * d.last
        : plane + ((level - 1u) & 1u) * d.rows * pitch + row * pitch;

    uint held[8];
    for (uint k = 0; k < 8u; ++k) {
      const uint chunk = group * span + simd_id + k * simds;
      const uint at = chunk * 32u + simd_lane;
      held[k] = (k < rounds && chunk < chunks && at < count) ? src[at] : identity;
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
  const device uint* src = reinterpret_cast<const device uint*>(input);

  uint i0 = gtid - simd_lane;
  for (; i0 < n; i0 += 8u * gthreads) {
    uint held[8];
    for (uint k = 0; k < 8u; ++k) {
      const uint i = i0 + k * gthreads + simd_lane;
      held[k] = i < n ? src[i] : identity;
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
  if (n == 1u) return reinterpret_cast<const device uint*>(input)[0];
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


#define M4_SEL_CAP 1024u
#define M4_SEL_BYTES 16384u

struct M4SelState {
  uint fast;
  uint done;
  uint have_bound;
  uint bound_key;
  uint bound_idx;
  float exclusive;
  uint taken;
  uint want;
  uint sel_prefix;
  uint sel_remaining;
  uint total_lt;
  uint held;
  uint pad[4];
};

inline device M4SelState* m4_sel_state(device uchar* base) {
  return reinterpret_cast<device M4SelState*>(base);
}
inline device atomic_uint* m4_sel_hist(device uchar* base) {
  return reinterpret_cast<device atomic_uint*>(base + 64u);
}
inline device atomic_uint* m4_sel_fill(device uchar* base) {
  return reinterpret_cast<device atomic_uint*>(base + 64u + 2048u);
}
inline device uint* m4_sel_keys(device uchar* base) {
  return reinterpret_cast<device uint*>(base + 64u + 2048u + 16u);
}
inline device uint* m4_sel_idx(device uchar* base) {
  return reinterpret_cast<device uint*>(base + 64u + 2048u + 16u + 4096u);
}

inline bool m4_sel_after(const device M4SelState* st, uint key, uint idx) {
  if (st->have_bound == 0u) return true;
  return key > st->bound_key || (key == st->bound_key && idx > st->bound_idx);
}

inline void m4_sel_init(
    device uchar* base, device uchar* o0, const M1ValueDesc d0, const M1ValueDesc d1,
    const device uchar* a1, uint mode, uint gtid, uint gthreads) {
  const uint n = d0.rows * d0.last;
  for (uint i = gtid; i < n; i += gthreads) m1_store_b(o0, i, false);
  device atomic_uint* hist = m4_sel_hist(base);
  for (uint b = gtid; b < 512u; b += gthreads) atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
  if (gtid == 0u) {
    device M4SelState* st = m4_sel_state(base);
    st->fast = d0.rows == 1u ? 1u : 0u;
    st->have_bound = 0u;
    st->bound_key = 0u;
    st->bound_idx = 0u;
    st->exclusive = 0.0f;
    st->taken = 0u;
    st->sel_prefix = 0u;
    st->sel_remaining = 0u;
    st->total_lt = 0u;
    st->held = 0u;
    uint want = M4_SEL_CAP;
    bool done = d0.last == 0u;
    if (mode == 0u) {
      const int signed_k = m1_load_i(a1, 0u, d1.dtype);
      const uint k = signed_k <= 0 ? 0u : uint(signed_k);
      want = min(k, M4_SEL_CAP);
      if (k == 0u) done = true;
    }
    st->want = want;
    st->done = done ? 1u : 0u;
    atomic_store_explicit(&m4_sel_fill(base)[0], 0u, memory_order_relaxed);
    atomic_store_explicit(&m4_sel_fill(base)[1], 0u, memory_order_relaxed);
  }
}

inline void m4_sel_hist_pass(
    device uchar* base, const device uchar* a0, const M1ValueDesc d0, uint pass,
    uint gtid, uint gthreads, uint tid, uint threads, threadgroup atomic_uint* tg_hist) {
  const device M4SelState* st = m4_sel_state(base);
  if (st->done != 0u || st->fast == 0u) return;
  const uint n = d0.last;
  const uint shift = 24u - 8u * pass;
  const uint prefix = st->sel_prefix;
  for (uint b = tid; b < 256u; b += threads) atomic_store_explicit(&tg_hist[b], 0u, memory_order_relaxed);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint i0 = gtid; i0 < n; i0 += 8u * gthreads) {
    float held[8];
    for (uint u = 0u; u < 8u; ++u) {
      const uint i = i0 + u * gthreads;
      held[u] = i < n ? m1_load_f(a0, i, d0.dtype) : 0.0f;
    }
    for (uint u = 0u; u < 8u; ++u) {
      const uint i = i0 + u * gthreads;
      if (i >= n) break;
      const uint key = m1_desc_key(held[u]);
      if (!m4_sel_after(st, key, i)) continue;
      if (pass == 0u || (key >> (shift + 8u)) == (prefix >> (shift + 8u)))
        atomic_fetch_add_explicit(&tg_hist[(key >> shift) & 255u], 1u, memory_order_relaxed);
    }
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  device atomic_uint* hist = m4_sel_hist(base);
  for (uint b = tid; b < 256u; b += threads) {
    const uint c = atomic_load_explicit(&tg_hist[b], memory_order_relaxed);
    if (c != 0u) atomic_fetch_add_explicit(&hist[b], c, memory_order_relaxed);
  }
}

inline void m4_sel_pick(device uchar* base, uint pass, uint tid, uint threads) {
  device M4SelState* st = m4_sel_state(base);
  if (st->done != 0u || st->fast == 0u) return;
  device atomic_uint* hist = m4_sel_hist(base);
  if (tid == 0u) {
    const uint shift = 24u - 8u * pass;
    const uint remaining = pass == 0u ? st->want : st->sel_remaining;
    uint before = 0u;
    uint digit = 0u;
    for (; digit + 1u < 256u; ++digit) {
      const uint here = atomic_load_explicit(&hist[digit], memory_order_relaxed);
      if (before + here >= remaining) break;
      before += here;
    }
    st->sel_prefix = (pass == 0u ? 0u : st->sel_prefix) | (digit << shift);
    st->sel_remaining = remaining - before;
    if (pass == 3u) st->total_lt = st->want - (remaining - before);
  }
  threadgroup_barrier(mem_flags::mem_device);
  for (uint b = tid; b < 512u; b += threads) atomic_store_explicit(&hist[b], 0u, memory_order_relaxed);
  if (tid == 0u) {
    atomic_store_explicit(&m4_sel_fill(base)[0], 0u, memory_order_relaxed);
    atomic_store_explicit(&m4_sel_fill(base)[1], 0u, memory_order_relaxed);
  }
}

inline void m4_sel_compact(
    device uchar* base, const device uchar* a0, const M1ValueDesc d0, uint gtid, uint gthreads) {
  const device M4SelState* st = m4_sel_state(base);
  if (st->done != 0u || st->fast == 0u) return;
  const uint n = d0.last;
  const uint pivot = st->sel_prefix;
  const uint total_lt = st->total_lt;
  device atomic_uint* fill = m4_sel_fill(base);
  device uint* keys = m4_sel_keys(base);
  device uint* idx = m4_sel_idx(base);
  for (uint i0 = gtid; i0 < n; i0 += 8u * gthreads) {
    float held[8];
    for (uint u = 0u; u < 8u; ++u) {
      const uint i = i0 + u * gthreads;
      held[u] = i < n ? m1_load_f(a0, i, d0.dtype) : 0.0f;
    }
    for (uint u = 0u; u < 8u; ++u) {
      const uint i = i0 + u * gthreads;
      if (i >= n) break;
      const uint key = m1_desc_key(held[u]);
      if (!m4_sel_after(st, key, i)) continue;
      if (key < pivot) {
        const uint slot = atomic_fetch_add_explicit(&fill[0], 1u, memory_order_relaxed);
        if (slot < M4_SEL_CAP) { keys[slot] = key; idx[slot] = i; }
      } else if (key == pivot) {
        const uint slot = atomic_fetch_add_explicit(&fill[1], 1u, memory_order_relaxed);
        if (total_lt + slot < M4_SEL_CAP) { keys[total_lt + slot] = key; idx[total_lt + slot] = i; }
      }
    }
  }
}

inline void m4_sel_finish(
    device uchar* base, const device uchar* a0, const device uchar* a1, device uchar* o0,
    const M1ValueDesc d0, const M1ValueDesc d1, uint mode,
    uint tid, uint threads, threadgroup uint* tg_key, threadgroup uint* tg_idx, threadgroup uint* tg_scan) {
  device M4SelState* st = m4_sel_state(base);
  if (st->done != 0u || st->fast == 0u) return;
  const uint n = d0.last;
  const uint pivot = st->sel_prefix;
  const uint total_lt = st->total_lt;
  const uint want = st->want;
  const uint remaining = st->sel_remaining;
  const uint n_eq = atomic_load_explicit(&m4_sel_fill(base)[1], memory_order_relaxed);
  const uint eq_room = M4_SEL_CAP - total_lt;
  device uint* keys = m4_sel_keys(base);
  device uint* idx = m4_sel_idx(base);
  uint held = total_lt + min(n_eq, eq_room);
  for (uint p = tid; p < held; p += threads) {
    tg_key[p] = keys[p];
    tg_idx[p] = idx[p];
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if (n_eq > eq_room) {
    const uint chunk_begin = uint((ulong(n) * tid) / threads);
    const uint chunk_end = uint((ulong(n) * (tid + 1u)) / threads);
    uint mine = 0u;
    for (uint i = chunk_begin; i < chunk_end; ++i) {
      const uint key = m1_desc_key(m1_load_f(a0, i, d0.dtype));
      mine += (key == pivot && m4_sel_after(st, key, i)) ? 1u : 0u;
    }
    tg_scan[tid] = mine;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (tid == 0u) {
      uint run = 0u;
      for (uint t = 0u; t < threads; ++t) {
        const uint here = tg_scan[t];
        tg_scan[t] = run;
        run += here;
      }
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint at = tg_scan[tid];
    for (uint i = chunk_begin; i < chunk_end; ++i) {
      const uint key = m1_desc_key(m1_load_f(a0, i, d0.dtype));
      if (key == pivot && m4_sel_after(st, key, i)) {
        if (at < remaining) {
          tg_key[total_lt + at] = pivot;
          tg_idx[total_lt + at] = i;
        }
        ++at;
      }
    }
    held = want;
    threadgroup_barrier(mem_flags::mem_threadgroup);
  }

  uint n_all = 1u;
  while (n_all < held) n_all <<= 1u;
  for (uint p = held + tid; p < n_all; p += threads) {
    tg_key[p] = ~0u;
    tg_idx[p] = ~0u;
  }
  threadgroup_barrier(mem_flags::mem_threadgroup);
  for (uint size = 2u; size <= n_all; size <<= 1u) {
    for (uint stride = size >> 1u; stride > 0u; stride >>= 1u) {
      for (uint t = tid; t < n_all / 2u; t += threads) {
        const uint lo = 2u * stride * (t / stride) + (t % stride);
        const uint hi = lo + stride;
        const bool ascending = ((lo & size) == 0u);
        const uint klo = tg_key[lo], khi = tg_key[hi];
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
  if (tid == 0u) {

    float exclusive = st->exclusive;
    uint taken = st->taken;
    bool done = false;
    const float threshold = mode == 1u ? m1_load_f(a1, 0u, d1.dtype) : 0.0f;
    uint k = 0u;
    if (mode == 0u) {
      const int signed_k = m1_load_i(a1, 0u, d1.dtype);
      k = signed_k <= 0 ? 0u : uint(signed_k);
    }
    uint p = 0u;
    for (; p < held; ++p) {
      const uint index = tg_idx[p];
      if (mode == 1u) {
        if (!(exclusive < threshold)) { done = true; break; }
        m1_store_b(o0, index, true);
        exclusive += m1_load_f(a0, index, d0.dtype);
      } else {
        if (taken >= k) { done = true; break; }
        m1_store_b(o0, index, true);
        ++taken;
      }
    }
    if (!done && (held < want || held == 0u)) done = true;
    if (!done && mode == 0u && taken >= k) done = true;
    if (!done && mode == 1u && !(exclusive < threshold)) done = true;
    if (!done && held > 0u) {
      st->have_bound = 1u;
      st->bound_key = tg_key[held - 1u];
      st->bound_idx = tg_idx[held - 1u];
    }
    st->exclusive = exclusive;
    st->taken = taken;
    st->want = mode == 0u ? min(k - taken, M4_SEL_CAP) : M4_SEL_CAP;
    st->held = held;
    st->done = done ? 1u : 0u;
  }
}

inline void m4_sel_fallback(
    device uchar* base, const device uchar* a0, const device uchar* a1, device uchar* o0,
    const M1ValueDesc d0, const M1ValueDesc d1, uint mode,
    uint tid, uint threads, threadgroup M1ArgmaxCandidate* tgbuf) {
  device M4SelState* st = m4_sel_state(base);
  if (st->done != 0u) return;
  if (st->fast == 0u) {
    if (mode == 1u) m1_nucleus_select_mt(a0, a1, o0, d0, d1, tid, threads, tgbuf);
    else m1_topk_select_mt(a0, a1, o0, d0, d1, tid, threads, tgbuf);
    return;
  }
  const uint n = d0.last;
  const float threshold = mode == 1u ? m1_load_f(a1, 0u, d1.dtype) : 0.0f;
  uint k = 0u;
  if (mode == 0u) {
    const int signed_k = m1_load_i(a1, 0u, d1.dtype);
    k = signed_k <= 0 ? 0u : uint(signed_k);
  }
  if (mode == 1u &&
      m1_cummass_keeps_row_mt(a0, 0u, n, d0.dtype, threshold, tid, threads, tgbuf)) {
    for (uint i = tid; i < n; i += threads)
      m1_store_b(o0, i, !isnan(m1_load_f(a0, i, d0.dtype)));
    threadgroup_barrier(mem_flags::mem_device);
    if (tid == 0u) st->done = 1u;
    return;
  }
  float exclusive = st->exclusive;
  uint taken = st->taken;
  bool have_prev = st->have_bound != 0u;

  float prev_value = have_prev ? m1_load_f(a0, st->bound_idx, d0.dtype) : 0.0f;
  uint prev_index = st->bound_idx;
  for (uint position = 0; position < n; ++position) {
    if (mode == 1u ? !(exclusive < threshold) : taken >= k) break;
    M1ArgmaxCandidate best = {0.0f, 0u, 0u, 0u};
    for (uint candidate = tid; candidate < n; candidate += threads) {
      const float value = m1_load_f(a0, candidate, d0.dtype);
      if (have_prev && !m1_sort_better(prev_value, prev_index, value, candidate)) continue;
      best = m1_sort_pick(best, M1ArgmaxCandidate{value, candidate, 1u, 0u});
    }
    tgbuf[tid] = best;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    for (uint stride = 1u; stride < threads; stride <<= 1) {
      if ((tid % (2u * stride)) == 0u && tid + stride < threads)
        tgbuf[tid] = m1_sort_pick(tgbuf[tid], tgbuf[tid + stride]);
      threadgroup_barrier(mem_flags::mem_threadgroup);
    }
    const M1ArgmaxCandidate found = tgbuf[0];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (found.have == 0u) break;
    if (tid == 0) m1_store_b(o0, found.index, true);
    exclusive += found.value;
    ++taken;
    prev_value = found.value;
    prev_index = found.index;
    have_prev = true;
  }
  if (tid == 0u) st->done = 1u;
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
  const device float* values = reinterpret_cast<const device float*>(input);
  const uint span = groups * nthreads;
  const uint begin = group * nthreads + tid;
  for (uint row = 0; row < d.rows; ++row) {
    const uint base = row * d.last;
    M1ArgmaxCandidate best = {-INFINITY, 0u, 0u, 0u};
    for (uint i = begin; i < d.last; i += span) {
      const float value = values[base + i];
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
  if (p.tag == 0x58 && nthreads > 1u && p.pred_tag != 2) {

    const M1ValueDesc d1 = descriptors[p.a1];
    if (p.pred_tag == 0) m1_topk_select_mt(a0, a1, o0, d0, d1, tid, nthreads, tgbuf);
    else m1_nucleus_select_mt(a0, a1, o0, d0, d1, tid, nthreads, tgbuf);
    return;
  }
  ptir_m1_execute_part(generated_tag, status, descriptors, params, a0, a1, a2, o0,
                       o1, temporary, tid, nthreads);
}
