#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void merge_lse_combine(
    const device T* o1        [[buffer(0)]],
    const device float* lse1  [[buffer(1)]],
    const device T* o2        [[buffer(2)]],
    const device float* lse2  [[buffer(3)]],
    device T* o_out           [[buffer(4)]],
    device float* lse_out     [[buffer(5)]],
    uint3 tid  [[thread_position_in_grid]],
    uint3 grid [[threads_per_grid]]) {
  const uint head_dim = grid.x;
  const uint heads = grid.y;
  const uint d = tid.x;
  const uint h = tid.y;
  const uint t = tid.z;

  const size_t col = size_t(t) * size_t(heads) + size_t(h);
  const size_t i = col * size_t(head_dim) + size_t(d);

  const float l1 = lse1[col];
  const float l2 = lse2[col];

  if (!isfinite(l2)) {
    o_out[i] = o1[i];
    if (d == 0) lse_out[col] = l1;
    return;
  }
  if (!isfinite(l1)) {
    o_out[i] = o2[i];
    if (d == 0) lse_out[col] = l2;
    return;
  }

  const float merged_max = max(l1, l2);
  const float w1 = exp2(l1 - merged_max);
  const float w2 = exp2(l2 - merged_max);
  const float total = w1 + w2;

  const float v1 = static_cast<float>(o1[i]);
  const float v2 = static_cast<float>(o2[i]);
  o_out[i] = static_cast<T>((v1 * w1 + v2 * w2) / total);

  if (d == 0) {
    lse_out[col] = merged_max + log2(total);
  }
}

#define instantiate_merge_lse_combine(name, itype)                      \
  template [[host_name("merge_lse_combine_" #name)]]                    \
  [[kernel]] void merge_lse_combine<itype>(                             \
      const device itype*, const device float*, const device itype*,    \
      const device float*, device itype*, device float*, uint3, uint3);

instantiate_merge_lse_combine(bfloat16, bfloat)
