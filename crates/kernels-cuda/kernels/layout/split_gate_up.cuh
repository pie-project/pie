//===-- split_gate_up.cuh - the halves split of a packed gate/up bank -===//
//
// One `__global__` template. `split_gate_up.cu` included this file and held
// `split_gate_up_bf16`; §43 deleted the file whole -- the launcher had no row,
// so no shim entry, and no C++ caller. Exactly one definition still exists in
// the tree and it is the one below. From §43 until
// `driver_internal::split_gate_up_bf16` there was no host launcher anywhere,
// which is the state `every_carried_file_is_reachable` reported. The refusal
// recorded here still stands, because it was always about the KERNEL's
// indexing and not about the launcher that had gone.
//
// # The rule it was said to want, and the rule it actually states
//
// This file used to record a refusal: the launcher states
// `dim3(ceil(inter/256), n_tokens)` with the CHANNEL axis on `grid.x`, while
// `LaunchRule::ElementwiseRows` is `[rows, ceil(width/256), 1]` -- the same
// rectangle transposed -- so a row under that rule would read `blockIdx.y` as
// a channel chunk and `blockIdx.x` as a token, and every token past the first
// would copy from the wrong row.
//
// The refusal named the wrong rule. `LaunchRule::SplitPacked` is
// `[ceil(in_width / 256), rows, 1]` at 256 threads, which is this launcher's
// axes in this launcher's order, and it is the rule `attn/split_packed.cu`
// was ported from -- one packed buffer taken apart into several. Its `grid.x`
// is sized on the PACKED input's `2 * inter` where the launcher sizes on
// `inter`, so the rule hands over twice the blocks; both loops below stride
// `j += blockDim.x * gridDim.x` and bound themselves on `inter`, so the
// surplus blocks contribute a shorter loop and nothing else. That is the
// licence `attn/split_packed.cuh` wrote down for the same rule over the same
// arithmetic, and no body here changed to collect it.
//
// # Why it is a template when the launcher is bf16
//
// Because a row cannot name a kernel that is not one.
// `DeviceKernel::instantiation()` emits `path<Elem>` -- exactly one type
// argument -- so a plain `__global__` is unnameable whatever its geometry,
// and this kernel's geometry was the only thing anyone had checked. `T` is
// the element type and nothing else: this is a pure copy, so nothing widens
// to float and a `T` two bytes wide moves the same bytes the bf16 form moved.
// The deleted ahead-of-time launcher instantiated it at `bf16`, and
// `layout::split_gate_up::inst` writes that same argument out as the one
// template-id anything in the tree asks NVRTC for.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "prelude/device.cuh"

namespace pie::layout {

/// `[N, 2*inter] -> [N, inter] x 2`, halves rather than parity.
///
/// No token count and no guard on `blockIdx.y`: `LaunchRule::SplitPacked` puts
/// the row on `grid.y` and covers the rows exactly, so a bound check would
/// test what the rule already promised. The launcher never passed one either.
template <class T>
__global__ void split_gate_up(
    const T* __restrict__ src,
    T* __restrict__ gate_out,
    T* __restrict__ up_out,
    int inter)
{
    const int n = blockIdx.y;
    const int stride = 2 * inter;
    const T* src_row = src + static_cast<long long>(n) * stride;

    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < inter;
         j += blockDim.x * gridDim.x) {
        gate_out[static_cast<long long>(n) * inter + j] = src_row[j];
    }
    for (int j = blockIdx.x * blockDim.x + threadIdx.x; j < inter;
         j += blockDim.x * gridDim.x) {
        up_out[static_cast<long long>(n) * inter + j] = src_row[inter + j];
    }
}

}  // namespace pie::layout
