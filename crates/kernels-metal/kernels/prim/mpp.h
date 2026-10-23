constexpr ushort mpp_first_bit(ulong bits) {
  ushort i = 0;
  while ((bits & 1) == 0) {
    bits >>= 1;
    ++i;
  }
  return i;
}
template <ulong Bits, class T, class F>
__attribute__((always_inline)) void mpp_masked_elements(const thread T &values,
                                                        const thread F &body) {
  if constexpr (Bits != 0) {
    constexpr ushort index = mpp_first_bit(Bits);
    body(index);
    mpp_masked_elements<Bits &(Bits - 1)>(values, body);
  }
}

template <class T, class F>
__attribute__((always_inline)) void mpp_for_each(const thread T &values, ulong mask,
                                                 const thread F &body) {
  if (values.get_capacity() <= 64 && mask == (~0ul >> (64 - values.get_capacity()))) {
#pragma unroll
    for (ushort i = 0; i < values.get_capacity(); ++i)
      body(i);
    return;
  }
  if (values.get_capacity() <= 64) {
    switch (mask) {
    case 0xful:
      mpp_masked_elements<0xful>(values, body);
      return;
    case 0xf0ful:
      mpp_masked_elements<0xf0ful>(values, body);
      return;
    case 0xf0f0f0ful:
      mpp_masked_elements<0xf0f0f0ful>(values, body);
      return;
    case 0xf0ffffful:
      mpp_masked_elements<0xf0ffffful>(values, body);
      return;
    case 0xfffful:
      mpp_masked_elements<0xfffful>(values, body);
      return;
    case 0xfffffffful:
      mpp_masked_elements<0xfffffffful>(values, body);
      return;
    }
  }
#pragma unroll
  for (ushort i = 0; i < values.get_capacity(); ++i)
    if (values.is_valid_element(i))
      body(i);
}

template <class T> __attribute__((always_inline)) ulong mpp_valid_mask(const thread T &values) {
  ulong mask = 0;
#pragma unroll
  for (ushort i = 0; i < min(ushort(64), ushort(values.get_capacity())); ++i)
    mask |= ulong(values.is_valid_element(i)) << i;
  return mask;
}
