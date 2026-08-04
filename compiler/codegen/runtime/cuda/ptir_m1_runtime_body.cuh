
__device__ __forceinline__ void ptir_m1_execute(
    m1_u32 generated_tag,
    M1Status* status,
    const M1ValueDesc* descriptors,
    const M1OpParams* params,
    const m1_u8* a0,
    const m1_u8* a1,
    const m1_u8* a2,
    m1_u8* o0,
    m1_u8* o1,
    m1_u8* temporary) {
  if (status->state != 1) return;
  M1OpParams p = params[0];
  p.tag = generated_tag;
  const M1ValueDesc d0 = descriptors[p.a0];
  const M1ValueDesc d1 = descriptors[p.a1];
  const M1ValueDesc d2 = descriptors[p.a2];
  const M1ValueDesc out0 = descriptors[p.o0];

  if (p.tag == 0x81) {
    for (m1_u32 i = 0; i < out0.len; ++i) {
      if (p.lit_dtype == 0)
        m1_store_f(o0, i, m1_bits_f32(p.lit_bits));
      else if (p.lit_dtype == 1)
        m1_store_i(o0, i, m1_bits_i32(p.lit_bits));
      else if (p.lit_dtype == 2)
        m1_store_u(o0, i, p.lit_bits);
      else
        m1_store_b(o0, i, p.lit_bits != 0);
    }
    return;
  }
  if (p.tag == 0x90 || p.tag == 0x91) {
    if (out0.dtype == 3) {
      if (p.bool_storage > 1u) {
        m1_fault(status, p.tag);
        return;
      }
      if (p.bool_storage == 1u) {
        for (m1_u32 i = 0; i < out0.len; ++i)
          o0[i] = (a0[i >> 3] >> (i & 7)) & 1u;
      } else {
        for (m1_u32 i = 0; i < out0.len; ++i)
          o0[i] = a0[i] != 0 ? 1 : 0;
      }
    } else {
      m1_copy_typed(a0, o0, out0.len, out0.dtype);
    }
    return;
  }
  if (p.tag == 0x92) {
    const m1_u32 logical_bytes =
        d0.dtype == 3
            ? (p.bool_storage == 1u ? (d0.len + 7u) / 8u : d0.len)
            : d0.len * 4u;
    if (d0.dtype == 3 && p.bool_storage > 1u) {
      m1_fault(status, p.tag);
      return;
    }
    if (logical_bytes > p.sink_bytes) {
      m1_fault(status, p.tag);
      return;
    }
    if (d0.dtype == 3) {
      if (p.bool_storage == 1u) {
        for (m1_u32 i = 0; i < logical_bytes; ++i) o0[i] = 0;
        for (m1_u32 i = 0; i < d0.len; ++i)
          if (a0[i] != 0)
            o0[i >> 3] |= (m1_u8)(1u << (i & 7));
      } else {
        for (m1_u32 i = 0; i < d0.len; ++i)
          o0[i] = a0[i] != 0 ? 1 : 0;
      }
    } else {
      m1_copy_typed(a0, o0, d0.len, d0.dtype);
    }
    for (m1_u32 i = logical_bytes; i < p.sink_bytes; ++i) o0[i] = 0;
    return;
  }
  if (p.tag == 0xA0) {
    if (p.intr == 5u) {
      if (out0.dtype != 2u || out0.len != 1u || a0 == nullptr) {
        m1_fault(status, p.tag);
        return;
      }
      m1_store_u(o0, 0u, reinterpret_cast<const m1_u32*>(a0)[0]);
      return;
    }
    if (p.intr == 6u) {
      if (out0.dtype != 1u || a0 == nullptr) {
        m1_fault(status, p.tag);
        return;
      }
      for (m1_u32 index = 0; index < out0.len; ++index)
        m1_store_i(
            o0, index, reinterpret_cast<const int*>(a0)[index]);
      return;
    }
    if (p.imm == 0u || p.intrinsic_dtype > 2u || a0 == nullptr) {
      m1_fault(status, p.tag);
      return;
    }
    const m1_u32 stride =
        p.intrinsic_row_stride == 0u ? p.imm : p.intrinsic_row_stride;
    if (stride < p.imm) {
      m1_fault(status, p.tag);
      return;
    }
    const m1_u64 first_row =
        (m1_u64)p.intrinsic_row_offset + (m1_u64)p.imm2;
    if (out0.dtype != 0u) {
      m1_fault(status, p.tag);
      return;
    }
    const m1_u32 logical_width =
        out0.last == 0u ? p.imm : out0.last;
    if (logical_width == 0u || stride < logical_width) {
      m1_fault(status, p.tag);
      return;
    }
    for (m1_u32 i = 0; i < out0.len; ++i) {
      const m1_u32 row = i / logical_width;
      const m1_u32 column = i % logical_width;
      m1_store_f(
          o0,
          i,
          m1_intrinsic_row_load(
              a0,
              first_row + (m1_u64)row,
              column,
              stride,
              p.intrinsic_dtype));
    }
    return;
  }
  if (p.tag == 0xA1) {
    m1_copy_typed(a0, o0, out0.len, out0.dtype);
    return;
  }
  if (p.tag == 0xA2) return;

  if (p.tag == 0x01 || p.tag == 0x02 || p.tag == 0x04) {
    for (m1_u32 i = 0; i < out0.len; ++i) {
      const float value = m1_load_f(a0, m1_pick(d0.len, i), d0.dtype);
      if (p.tag == 0x01)
        m1_store_f(o0, i, expf(value));
      else if (p.tag == 0x02)
        m1_store_f(o0, i, logf(value));
      else
        m1_store_f(o0, i, 1.0f / value);
    }
    return;
  }
  if (p.tag == 0x03 || p.tag == 0x05 || p.tag == 0x06) {
    for (m1_u32 i = 0; i < out0.len; ++i) {
      const m1_u32 source_index = m1_pick(d0.len, i);
      if (d0.dtype == 0) {
        const float value = m1_load_f(a0, source_index, d0.dtype);
        const float result =
            p.tag == 0x03
                ? -value
                : (p.tag == 0x05
                       ? fabsf(value)
                       : (value > 0
                              ? 1.0f
                              : (value < 0 ? -1.0f : 0.0f)));
        m1_store_f(o0, i, result);
      } else if (d0.dtype == 1) {
        const int value = m1_load_i(a0, source_index, d0.dtype);
        int result = value;
        if (p.tag == 0x03)
          result = m1_bits_i32(0u - (m1_u32)value);
        else if (p.tag == 0x05)
          result =
              (m1_u32)value == 0x80000000u
                  ? value
                  : (value < 0 ? -value : value);
        else
          result = value > 0 ? 1 : (value < 0 ? -1 : 0);
        m1_store_i(o0, i, result);
      } else if (d0.dtype == 2) {
        const m1_u32 value = m1_load_u(a0, source_index, d0.dtype);
        m1_store_u(
            o0,
            i,
            p.tag == 0x03
                ? 0u - value
                : (p.tag == 0x06 ? (value != 0 ? 1u : 0u) : value));
      } else {
        m1_fault(status, p.tag);
        return;
      }
    }
    return;
  }
  if (p.tag == 0x07) {
    for (m1_u32 i = 0; i < out0.len; ++i) {
      const m1_u32 source_index = m1_pick(d0.len, i);
      if (out0.dtype == 0)
        m1_store_f(o0, i, m1_load_f(a0, source_index, d0.dtype));
      else if (out0.dtype == 1)
        m1_store_i(o0, i, m1_load_i(a0, source_index, d0.dtype));
      else if (out0.dtype == 2)
        m1_store_u(o0, i, m1_load_u(a0, source_index, d0.dtype));
      else
        m1_store_b(o0, i, m1_load_b(a0, source_index, d0.dtype));
    }
    return;
  }

  if ((p.tag >= 0x10 && p.tag <= 0x1D) || p.tag == 0x1F) {
    for (m1_u32 i = 0; i < out0.len; ++i) {
      const m1_u32 xindex = m1_pick(d0.len, i);
      const m1_u32 yindex = m1_pick(d1.len, i);
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
          const m1_u32 x = m1_load_u(a0, xindex, d0.dtype);
          const m1_u32 y = m1_load_u(a1, yindex, d1.dtype);
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
        float result = 0.0f;
        if (p.tag == 0x10) result = x + y;
        else if (p.tag == 0x11) result = x - y;
        else if (p.tag == 0x12) result = x * y;
        else if (p.tag == 0x13) result = x / y;
        else if (p.tag == 0x14) result = m1_element_max(x, y);
        else if (p.tag == 0x15) result = m1_element_min(x, y);
        else result = fmodf(x, y);
        m1_store_f(o0, i, result);
      } else if (d0.dtype == 1) {
        const int x = m1_load_i(a0, xindex, d0.dtype);
        const int y = m1_load_i(a1, yindex, d1.dtype);
        int result = 0;
        if (p.tag == 0x10)
          result = m1_bits_i32((m1_u32)x + (m1_u32)y);
        else if (p.tag == 0x11)
          result = m1_bits_i32((m1_u32)x - (m1_u32)y);
        else if (p.tag == 0x12)
          result = m1_bits_i32((m1_u32)x * (m1_u32)y);
        else if (p.tag == 0x13)
          result = m1_i32_div(x, y);
        else if (p.tag == 0x14)
          result = x > y ? x : y;
        else if (p.tag == 0x15)
          result = x < y ? x : y;
        else
          result = m1_i32_rem(x, y);
        m1_store_i(o0, i, result);
      } else {
        const m1_u32 x = m1_load_u(a0, xindex, d0.dtype);
        const m1_u32 y = m1_load_u(a1, yindex, d1.dtype);
        m1_u32 result = 0;
        if (p.tag == 0x10) result = x + y;
        else if (p.tag == 0x11) result = x - y;
        else if (p.tag == 0x12) result = x * y;
        else if (p.tag == 0x13) result = y == 0 ? 0 : x / y;
        else if (p.tag == 0x14) result = x > y ? x : y;
        else if (p.tag == 0x15) result = x < y ? x : y;
        else result = y == 0 ? 0 : x % y;
        m1_store_u(o0, i, result);
      }
    }
    return;
  }
  if (p.tag == 0x1E) {
    for (m1_u32 i = 0; i < out0.len; ++i)
      m1_store_b(
          o0, i, !m1_load_b(a0, m1_pick(d0.len, i), d0.dtype));
    return;
  }
  if (p.tag == 0x20) {
    for (m1_u32 i = 0; i < out0.len; ++i) {
      const bool select =
          m1_load_b(a0, m1_pick(d0.len, i), d0.dtype);
      const m1_u32 xi = m1_pick(d1.len, i);
      const m1_u32 yi = m1_pick(d2.len, i);
      if (out0.dtype == 0)
        m1_store_f(
            o0,
            i,
            select ? m1_load_f(a1, xi, d1.dtype)
                   : m1_load_f(a2, yi, d2.dtype));
      else if (out0.dtype == 1)
        m1_store_i(
            o0,
            i,
            select ? m1_load_i(a1, xi, d1.dtype)
                   : m1_load_i(a2, yi, d2.dtype));
      else if (out0.dtype == 2)
        m1_store_u(
            o0,
            i,
            select ? m1_load_u(a1, xi, d1.dtype)
                   : m1_load_u(a2, yi, d2.dtype));
      else
        m1_store_b(
            o0,
            i,
            select ? m1_load_b(a1, xi, d1.dtype)
                   : m1_load_b(a2, yi, d2.dtype));
    }
    return;
  }

  if (p.tag >= 0x30 && p.tag <= 0x32) {
    if (d0.dtype == 0)
      m1_reduce_float(p.tag, a0, o0, temporary, d0);
    else
      m1_reduce_integer(p.tag, a0, o0, temporary, d0);
    return;
  }
  if (p.tag == 0x33) {
    m1_reduce_argmax(a0, o0, temporary, d0);
    return;
  }
  if (p.tag == 0x38) {
    for (m1_u32 linear = 0; linear < out0.len; ++linear) {
      m1_u32 rem = linear;
      m1_u32 source_index = 0;
      m1_u32 source_stride[4] = {1, 1, 1, 1};
      for (int dim = (int)out0.rank - 2; dim >= 0; --dim) {
        source_stride[dim] =
            source_stride[dim + 1] *
            ((m1_u32)(dim + 1) < d0.rank ? d0.dims[dim + 1] : 1u);
      }
      for (m1_u32 dim = 0; dim < out0.rank; ++dim) {
        m1_u32 stride = 1;
        for (m1_u32 next = dim + 1; next < out0.rank; ++next)
          stride *= out0.dims[next];
        if (stride == 0) stride = 1;
        const m1_u32 coordinate = rem / stride;
        rem %= stride;
        const m1_u32 source_dim =
            dim < d0.rank ? d0.dims[dim] : 1u;
        if (source_dim != 1)
          source_index += coordinate * source_stride[dim];
      }
      if (out0.dtype == 0)
        m1_store_f(o0, linear, m1_load_f(a0, source_index, d0.dtype));
      else if (out0.dtype == 1)
        m1_store_i(o0, linear, m1_load_i(a0, source_index, d0.dtype));
      else if (out0.dtype == 2)
        m1_store_u(o0, linear, m1_load_u(a0, source_index, d0.dtype));
      else
        m1_store_b(o0, linear, m1_load_b(a0, source_index, d0.dtype));
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
    const m1_u32 m = d0.dims[0];
    const m1_u32 n = d0.dims[1];
    for (m1_u32 index = 0; index < m * n; ++index) {
      const m1_u32 source_index = (index % m) * n + index / m;
      if (out0.dtype == 0)
        m1_store_f(
            o0, index, m1_load_f(a0, source_index, d0.dtype));
      else if (out0.dtype == 1)
        m1_store_i(
            o0, index, m1_load_i(a0, source_index, d0.dtype));
      else if (out0.dtype == 2)
        m1_store_u(
            o0, index, m1_load_u(a0, source_index, d0.dtype));
      else
        m1_store_b(
            o0, index, m1_load_b(a0, source_index, d0.dtype));
    }
    return;
  }
  if (p.tag == 0x40 || p.tag == 0x41) {
    // Scanned in the operand's own dtype. A u32 offset scan is exactly what
    // ragged row offsets are built from, and accumulating one through float
    // is exact only below 2^24 -- past that it rounds, silently.
    const bool is_sum = p.tag == 0x40;
    for (m1_u32 row = 0; row < d0.rows; ++row) {
      float accumulated_f = is_sum ? 0.0f : 1.0f;
      m1_u32 accumulated_u = is_sum ? 0u : 1u;
      int accumulated_i = is_sum ? 0 : 1;
      for (m1_u32 column = 0; column < d0.last; ++column) {
        const m1_u32 index = row * d0.last + column;
        if (d0.dtype == 1) {
          const int value = m1_load_i(a0, index, d0.dtype);
          accumulated_i = is_sum ? (int)((m1_u32)accumulated_i + (m1_u32)value)
                                 : (int)((m1_u32)accumulated_i * (m1_u32)value);
          m1_store_i(o0, index, accumulated_i);
        } else if (d0.dtype == 2) {
          const m1_u32 value = m1_load_u(a0, index, d0.dtype);
          accumulated_u = is_sum ? accumulated_u + value
                                 : accumulated_u * value;
          m1_store_u(o0, index, accumulated_u);
        } else {
          const float value = m1_load_f(a0, index, d0.dtype);
          accumulated_f = is_sum ? accumulated_f + value
                                 : accumulated_f * value;
          m1_store_f(o0, index, accumulated_f);
        }
      }
    }
    return;
  }
  if (p.tag == 0x50) {
    for (m1_u32 position = 0; position < d0.len; ++position) {
      m1_u32 best_index = 0;
      float best_value = m1_nan();
      bool found = false;
      for (m1_u32 candidate = 0; candidate < d0.len; ++candidate) {
        bool used = false;
        for (m1_u32 prior = 0; prior < position; ++prior)
          if (reinterpret_cast<m1_u32*>(o1)[prior] == candidate)
            used = true;
        if (used) continue;
        const float value = m1_load_f(a0, candidate, d0.dtype);
        if (!found ||
            m1_sort_better(value, candidate, best_value, best_index)) {
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
    const m1_u32 count = p.imm < d0.last ? p.imm : d0.last;
    for (m1_u32 row = 0; row < d0.rows; ++row) {
      for (m1_u32 position = 0; position < count; ++position) {
        m1_u32 best_index = 0;
        float best_value = m1_nan();
        bool found = false;
        for (m1_u32 candidate = 0; candidate < d0.last; ++candidate) {
          bool used = false;
          for (m1_u32 prior = 0; prior < position; ++prior)
            if (reinterpret_cast<m1_u32*>(o1)[row * count + prior] ==
                candidate)
              used = true;
          if (used) continue;
          const float value =
              m1_load_f(a0, row * d0.last + candidate, d0.dtype);
          if (!found ||
              m1_sort_better(value, candidate, best_value, best_index)) {
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
    const m1_u32 m = d0.dims[0];
    const m1_u32 inner = d0.dims[1];
    const m1_u32 n = d1.dims[1];
    for (m1_u32 row = 0; row < m; ++row)
      for (m1_u32 column = 0; column < n; ++column)
        m1_store_f(o0, row * n + column, 0.0f);
    for (m1_u32 row = 0; row < m; ++row)
      for (m1_u32 k = 0; k < inner; ++k) {
        const float left =
            m1_load_f(a0, row * inner + k, d0.dtype);
        if (left == 0.0f) continue;
        for (m1_u32 column = 0; column < n; ++column) {
          const m1_u32 index = row * n + column;
          const float old = m1_load_f(o0, index, 0);
          m1_store_f(
              o0,
              index,
              old + left *
                  m1_load_f(a1, k * n + column, d1.dtype));
        }
      }
    return;
  }
  if (p.tag == 0x58) {
    for (m1_u32 row = 0; row < d0.rows; ++row) {
      const m1_u32 base = row * d0.last;
      if (p.pred_tag == 0) {
        m1_u32 k;
        if (d1.dtype == 1u) {
          const int signed_k =
              m1_load_i(a1, m1_pick(d1.len, row), d1.dtype);
          k = signed_k <= 0 ? 0u : (m1_u32)signed_k;
        } else {
          k = m1_load_u(a1, m1_pick(d1.len, row), d1.dtype);
        }
        if (k > d0.last) k = d0.last;
        if (k == 0u) {
          for (m1_u32 i = 0; i < d0.last; ++i) m1_store_b(o0, base + i, false);
        } else {
          // 4-pass 8-bit MSB radix select on `m1_desc_key`. The earlier form
          // rescanned the whole row per element, i.e. O(len^2) on a single
          // thread -- ~2.3e10 visits at a 151936-token vocabulary, which never
          // returns. `greater(i) < k` holds exactly when `key(i) <= K_k` for
          // `K_k` the k-th smallest key counting multiplicity, so ties survive
          // or fall together exactly as the reference has them.
          m1_u32 histogram[256];
          m1_u32 prefix = 0u;
          m1_u32 target = k;
          for (int pass = 0; pass < 4; ++pass) {
            const int shift = 24 - 8 * pass;
            const m1_u32 high_mask =
                (pass == 0) ? 0u : (0xFFFFFFFFu << (shift + 8));
            for (m1_u32 bucket = 0u; bucket < 256u; ++bucket)
              histogram[bucket] = 0u;
            for (m1_u32 j = 0; j < d0.last; ++j) {
              const m1_u32 key =
                  m1_desc_key(m1_load_f(a0, base + j, d0.dtype));
              if ((key & high_mask) == (prefix & high_mask))
                ++histogram[(key >> shift) & 0xFFu];
            }
            m1_u32 run = 0u;
            m1_u32 chosen = 255u;
            for (m1_u32 bucket = 0u; bucket < 256u; ++bucket) {
              if (run + histogram[bucket] >= target) { chosen = bucket; break; }
              run += histogram[bucket];
            }
            target -= run;
            prefix |= chosen << shift;
          }
          for (m1_u32 i = 0; i < d0.last; ++i) {
            const float value = m1_load_f(a0, base + i, d0.dtype);
            m1_store_b(o0, base + i,
                       !m1_isnan(value) && m1_desc_key(value) <= prefix);
          }
        }
      } else if (p.pred_tag == 1) {
        // Descending selection with the LAST PICK's total-order key as the
        // availability threshold (the k_pivot_cummassle technique) instead
        // of an already-picked rescan: the rescan made this O(len^3) on ONE
        // thread — a de-facto hang at LM vocab sizes (>10^15 steps at
        // 151,936). Bit-identical picks and keep bits: m1_sort_better is a
        // strict total order, so "strictly after the previous pick" visits
        // the same elements in the same order, and once `exclusive` clears
        // the threshold (or goes NaN) every later keep is false — they are
        // pre-stored and the loop stops early.
        const float threshold =
            m1_load_f(a1, m1_pick(d1.len, row), d1.dtype);
        for (m1_u32 i = 0; i < d0.last; ++i)
          m1_store_b(o0, base + i, false);
        float exclusive = 0.0f;
        float prev_value = 0.0f;
        m1_u32 prev_index = 0;
        bool have_prev = false;
        for (m1_u32 position = 0;
             position < d0.last && exclusive < threshold;
             ++position) {
          m1_u32 best_index = 0;
          float best_value = 0.0f;
          bool found = false;
          for (m1_u32 candidate = 0;
               candidate < d0.last;
               ++candidate) {
            const float value =
                m1_load_f(a0, base + candidate, d0.dtype);
            if (have_prev &&
                !m1_sort_better(
                    prev_value, prev_index, value, candidate))
              continue;
            if (!found ||
                m1_sort_better(
                    value, candidate, best_value, best_index)) {
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
      } else {
        const float threshold =
            m1_load_f(a1, m1_pick(d1.len, row), d1.dtype);
        for (m1_u32 i = 0; i < d0.last; ++i)
          m1_store_b(
              o0,
              base + i,
              m1_load_f(a0, base + i, d0.dtype) >= threshold);
      }
    }
    return;
  }
  if (p.tag == 0x60) {
    m1_u32 rest = 1u;
    m1_u32 n0 = 1u;
    if (d0.rank != 0) {
      n0 = d0.dims[0];
      rest = n0 == 0 ? 1u : d0.len / n0;
    }
    for (m1_u32 k = 0; k < d1.len; ++k) {
      const long long index = m1_load_index(a1, k, d1.dtype);
      for (m1_u32 r = 0; r < rest; ++r) {
        const m1_u32 output_index = k * rest + r;
        const bool valid = index >= 0 && (m1_u64)index < n0;
        const m1_u32 source_index =
            valid ? (m1_u32)index * rest + r : 0;
        if (out0.dtype == 0)
          m1_store_f(
              o0,
              output_index,
              valid ? m1_load_f(a0, source_index, d0.dtype) : 0.0f);
        else if (out0.dtype == 1)
          m1_store_i(
              o0,
              output_index,
              valid ? m1_load_i(a0, source_index, d0.dtype) : 0);
        else if (out0.dtype == 2)
          m1_store_u(
              o0,
              output_index,
              valid ? m1_load_u(a0, source_index, d0.dtype) : 0u);
        else
          m1_store_b(
              o0,
              output_index,
              valid && m1_load_b(a0, source_index, d0.dtype));
      }
    }
    return;
  }
  if (p.tag == 0x61) {
    const m1_u32 rows = d0.dims[0];
    const m1_u32 columns = d0.dims[1];
    for (m1_u32 row = 0; row < rows; ++row) {
      const long long column = m1_load_index(a1, row, d1.dtype);
      const bool valid = column >= 0 && (m1_u64)column < columns;
      const m1_u32 source_index =
          valid ? row * columns + (m1_u32)column : 0;
      if (out0.dtype == 0)
        m1_store_f(
            o0,
            row,
            valid ? m1_load_f(a0, source_index, d0.dtype) : 0.0f);
      else if (out0.dtype == 1)
        m1_store_i(
            o0,
            row,
            valid ? m1_load_i(a0, source_index, d0.dtype) : 0);
      else if (out0.dtype == 2)
        m1_store_u(
            o0,
            row,
            valid ? m1_load_u(a0, source_index, d0.dtype) : 0u);
      else
        m1_store_b(
            o0,
            row,
            valid && m1_load_b(a0, source_index, d0.dtype));
    }
    return;
  }
  if (p.tag == 0x62 || p.tag == 0x63) {
    m1_copy_typed(a0, o0, d0.len, d0.dtype);
    m1_u32 rest = 1u;
    m1_u32 n0 = 1u;
    if (d0.rank != 0) {
      n0 = d0.dims[0];
      rest = n0 == 0 ? 1u : d0.len / n0;
    }
    const bool scalar = d2.len == 1 && d1.len * rest != 1;
    for (m1_u32 k = 0; k < d1.len; ++k) {
      const long long index = m1_load_index(a1, k, d1.dtype);
      if (index < 0 || (m1_u64)index >= n0) continue;
      for (m1_u32 r = 0; r < rest; ++r) {
        const m1_u32 dst = (m1_u32)index * rest + r;
        const m1_u32 src = scalar ? 0u : k * rest + r;
        if (d0.dtype == 0) {
          const float value = m1_load_f(a2, src, d2.dtype);
          m1_store_f(
              o0,
              dst,
              p.tag == 0x62 ? m1_load_f(o0, dst, 0) + value : value);
        } else if (d0.dtype == 1) {
          const int value = m1_load_i(a2, src, d2.dtype);
          m1_store_i(
              o0,
              dst,
              p.tag == 0x62
                  ? m1_bits_i32(
                        (m1_u32)m1_load_i(o0, dst, 1) +
                        (m1_u32)value)
                  : value);
        } else if (d0.dtype == 2) {
          const m1_u32 value = m1_load_u(a2, src, d2.dtype);
          m1_store_u(
              o0,
              dst,
              p.tag == 0x62 ? m1_load_u(o0, dst, 2) + value : value);
        } else {
          m1_store_b(o0, dst, m1_load_b(a2, src, d2.dtype));
        }
      }
    }
    return;
  }
  if (p.tag == 0x64) {
    for (m1_u32 i = 0; i < out0.len; ++i) m1_store_u(o0, i, i);
    return;
  }
  if (p.tag == 0x65) {
    const m1_u32 mask_width =
        d0.rank == 0 ? 1u : d0.dims[d0.rank - 1];
    for (m1_u32 i = 0; i < d0.len; ++i) {
      const m1_u32 column = i % mask_width;
      const m1_u32 word = column >> 5;
      const m1_u32 mask =
          word < d1.len ? m1_load_u(a1, word, d1.dtype) : 0u;
      m1_store_f(
          o0,
          i,
          ((mask >> (column & 31)) & 1u) != 0
              ? m1_load_f(a0, i, d0.dtype)
              : m1_neg_inf());
    }
    return;
  }
  if (p.tag == 0x66 || p.tag == 0x67 || p.tag == 0x68) {
    const m1_u32 key_count = p.imm;
    const m1_u32 window = p.tag == 0x67 ? p.imm2 : p.imm3;
    for (m1_u32 index = 0; index < out0.len; ++index) {
      const m1_u32 position_index =
          key_count == 0u ? 0u : index / key_count;
      const m1_u32 key =
          key_count == 0u ? 0u : index % key_count;
      const m1_u32 position =
          m1_load_u(a0, position_index, d0.dtype);
      bool allowed = key_count != 0u && key <= position;
      if (allowed && p.tag != 0x66) {
        const m1_u32 reach =
            key > 0xffffffffu - window ? 0xffffffffu : key + window;
        const bool recent = reach > position;
        allowed =
            p.tag == 0x67 ? recent : (key < p.imm2 || recent);
      }
      m1_store_b(o0, index, allowed);
    }
    return;
  }
  if (p.tag == 0x70 || p.tag == 0x71) {
    if (p.tag == 0x70) {
      const m1_u64 seed =
          ptir_rng_seed_eff_stream((m1_u32)p.rng_seed, p.imm);
      for (m1_u32 i = 0; i < out0.len; ++i) {
        const float uniform = ptir_rng_hash_uniform(seed, i);
        m1_store_f(
            o0,
            i,
            p.kind == 0 ? uniform : -logf(-logf(uniform)));
      }
    } else {
      const m1_u64 key = (m1_u64)m1_load_u(a0, 0, d0.dtype);
      const m1_u64 counter =
          (m1_u64)(d0.len > 1 ? m1_load_u(a0, 1, d0.dtype) : 0u);
      const m1_u64 seed =
          ptir_rng_keyed_seed((m1_u32)key, (m1_u32)counter);
      for (m1_u32 i = 0; i < out0.len; ++i) {
        const float uniform = ptir_rng_hash_uniform(seed, i);
        m1_store_f(
            o0,
            i,
            p.kind == 0 ? uniform : -logf(-logf(uniform)));
      }
    }
    return;
  }
  m1_fault(status, p.tag);
}
