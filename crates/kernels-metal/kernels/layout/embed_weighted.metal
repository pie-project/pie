#include <metal_stdlib>
using namespace metal;

/// **THE GATHER THAT INTERPOLATES** — the Metal mirror of `kernels-cuda`'s
/// `::pie::layout::embed_weighted` (`.wiki/alto/multimodal.md` §9.2).
///
/// `y[r] = sum over t of weights[r, t] * table[ids[r, t]]`, one thread per
/// output element: `tid.x` the column of the gathered row, `tid.y` the row.
///
/// The vision towers store one learned position grid at `num_grid_per_side^2`
/// and resample it to each image's own grid; upstream writes that as
/// `(pos_embed(interp_indices) * interp_weights[:, :, None]).sum(1)` over
/// `[patches, taps]` indices and weights, which is this expression with the
/// sum moved inside. Four taps for bilinear, sixteen for bicubic — read off
/// the operand rather than stated, because the operand carries it. gemma's
/// separable table is read at TWO taps with weights of one, which is the same
/// expression at `taps == 2`.
///
/// **THE ACCUMULATION IS f32 AND THE WEIGHTS ARRIVE f32.** Upstream
/// multiplies a float weight into a float-promoted embedding and sums; a bf16
/// running sum over four taps would round four times, and a bf16 WEIGHT would
/// move the resample by more than the gather it feeds. Only the write is in
/// the model element.
///
/// **OUT-OF-RANGE IDS CLAMP TO ROW ZERO**, which is `embed.metal`'s own rule
/// one file over: a gather with an index it cannot honour reads a defined row
/// rather than an address, and the vector was checked host-side before the
/// launch.
template <typename T>
[[kernel]] void embed_weighted(
    const device int* ids       [[buffer(0)]],
    const device float* weights [[buffer(1)]],
    const device T* table       [[buffer(2)]],
    device T* y                 [[buffer(3)]],
    const constant int& hidden  [[buffer(4)]],
    const constant int& vocab   [[buffer(5)]],
    const constant int& taps    [[buffer(6)]],
    uint2 tid [[thread_position_in_grid]]) {
  const int c = int(tid.x);
  if (c >= hidden) {
    return;
  }
  const size_t n = size_t(tid.y);
  const device int* row_ids = ids + n * size_t(taps);
  const device float* row_w = weights + n * size_t(taps);

  float acc = 0.0f;
  for (int t = 0; t < taps; ++t) {
    const int raw = row_ids[t];
    const int at = (raw >= 0 && raw < vocab) ? raw : 0;
    acc += row_w[t] * float(table[size_t(at) * size_t(hidden) + size_t(c)]);
  }
  y[n * size_t(hidden) + size_t(c)] = T(acc);
}

#define instantiate_embed_weighted(name, itype)                            \
  template [[host_name("embed_weighted_" #name)]]                          \
  [[kernel]] void embed_weighted<itype>(                                   \
      const device int*, const device float*, const device itype*,         \
      device itype*, const constant int&, const constant int&,             \
      const constant int&, uint2);

instantiate_embed_weighted(bfloat16, bfloat)
