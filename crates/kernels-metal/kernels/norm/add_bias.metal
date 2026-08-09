// Add a per-column bias to every row, in place: `out[r][c] += bias[c]`.
//
// The Qwen-2 family's attention biases. Until this file, no Metal kernel added
// a bias, so the shared Metal text could not state `AddBias` at all and every
// Qwen2 served through it computed its q/k/v projections without them --
// fluent, wrong text, which nothing downstream can detect.
//
// Written to match `kernels-cuda-new`'s `norm::add_bias_bf16` and
// `kernels-vulkan`'s `norm/add_bias.comp` operand for operand, so one
// statement means one thing on all three backends: IN PLACE over the value it
// biases (`out` is both operand and result, and the trace hands the same
// allocation for each), the bias off the statement's named weight, and the row
// width as a scalar.
//
// Launch: `LaunchRule::RouteRows`, grid = (width, rows, 1). The column is
// `tid.x` because a bias is BROADCAST -- one vector of `width` re-read by
// every row -- so unlike `residual_add`, a flat index over `rows * width`
// would not do: the column has to be recoverable from the invocation.
//
// `dispatchThreads` launches exactly the grid asked for, so there is no
// round-up tail to guard here; `width` is a scalar because the kernel needs
// the row PITCH, which the grid alone does not carry.

#include <metal_stdlib>
using namespace metal;

template <typename T>
[[kernel]] void add_bias(
    device T* out            [[buffer(0)]],   // [rows, width], read and written
    const device T* bias     [[buffer(1)]],   // [width]
    const constant int& width [[buffer(2)]],
    uint2 tid [[thread_position_in_grid]]) {
  const size_t i = size_t(tid.y) * size_t(width) + size_t(tid.x);
  out[i] = T(float(out[i]) + float(bias[tid.x]));
}

#define instantiate_add_bias(name, itype)                          \
  template [[host_name("add_bias_" #name)]]                        \
  [[kernel]] void add_bias<itype>(                                 \
      device itype*, const device itype*, const constant int&, uint2);

instantiate_add_bias(bfloat16, bfloat)
