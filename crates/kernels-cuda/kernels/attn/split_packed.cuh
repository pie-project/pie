//===-- split_packed.cuh - the fused QKV output, taken apart -------------===//
//
// Two `__global__` templates, no host code.
//
// The fused QKV matmul writes one row-major `[N, q_dim + 2 * kv_dim]` tensor.
// Everything downstream -- rope, the paged-KV write, the attention kernel --
// addresses each of Q, K and V as its own packed `[N, dim]` buffer, because
// their widths differ under GQA and a single stride cannot describe all
// three. So the split exists to turn one contiguous product into three
// contiguous operands. One pass over packed memory, pure copy, no compute.
//
// The fused alternative -- normalise, rotate and write the cache without ever
// materialising the three buffers -- is `attn/qkv_fused`, and it is why this
// file is short: the three kernels that did that work left for it.
//
// # NO ROW STATES EITHER KERNEL
//
// Both were `<<<dim3(ceil(max(q_dim, kv_dim) / 256), n), 256>>>`.
// `LaunchRule::SplitPacked` is the rule with that shape -- *"pointwise over
// the launch's INPUT width with the row on its own axis ... the QKV split is
// the case: three outputs, and the grid has to cover their sum"* -- and it is
// expressible in this backend's `Dims`, which carries `in_width` for exactly
// this. It is simply NOT PORTED: `runtime::launch::eval` evaluates four
// rules and answers `Ungeometric::Unported` for the rest, and stating a rule
// this backend cannot evaluate fails `every_stated_rule_is_ported` rather
// than failing at a fire. That is the correct order of events, and it is why
// there is no row here rather than a row that would break the gate.
//
// Worth recording for whoever ports it: `SplitPacked`'s grid over the INPUT
// width (`q_dim + 2 * kv_dim`) is WIDER than the launcher's over
// `max(q_dim, kv_dim)`, and the outputs are identical either way -- every
// loop below strides by `blockDim.x * gridDim.x` and bounds itself on its own
// output width, so extra blocks contribute nothing but a shorter loop. The
// port does not have to reproduce `max(q_dim, kv_dim)` to be correct.
//
// # Why the device-window variant is a second kernel and not a flag
//
// `split_qkv_devwin` takes its row window from DEVICE memory: the grid spans
// the full lane count and rows outside `[win[0], win[0] + win[1])` return
// before touching anything. That is what lets a captured graph replay across
// different row splits without re-recording -- the window changes in a
// buffer, not in a launch. The host-window form windows by caller offsets
// instead, and the two cannot be one kernel with a null check, because the
// pointers they are handed mean different things: BASE pointers here,
// already-offset pointers there.
//
// # What the vectorisation comment used to promise
//
// The original said it vectorised copies as `ushort4` -- eight bf16 values
// per transaction -- with a scalar tail. It does not: the loops below are
// scalar, one element per iteration, and they always were. The comment
// described an intention. It is removed rather than corrected, because a
// comment that describes what the code would ideally do is the kind a reader
// trusts and then debugs against.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "prelude/device.cuh"

namespace pie::attn {


/// Q, K and V out of one packed row, into three packed buffers.
///
/// Three independent strided loops rather than one over `q_dim + 2 * kv_dim`
/// with two divisions inside: the loops write DIFFERENT buffers at different
/// strides, and fusing them would put a pair of branches on every element to
/// recover which.
template <class T>
__global__ void split_qkv(
    const T* __restrict__ src,
    T* __restrict__ q_out,
    T* __restrict__ k_out,
    T* __restrict__ v_out,
    i32 q_dim, i32 kv_dim)
{
    const int n = static_cast<int>(blockIdx.y);
    const int stride = q_dim + 2 * kv_dim;
    const T* src_row = src + static_cast<long long>(n) * stride;

    // Q block: cols [0, q_dim)
    for (int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); j < q_dim;
         j += static_cast<int>(blockDim.x * gridDim.x)) {
        q_out[static_cast<long long>(n) * q_dim + j] = src_row[j];
    }
    // K block: cols [q_dim, q_dim + kv_dim)
    for (int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); j < kv_dim;
         j += static_cast<int>(blockDim.x * gridDim.x)) {
        k_out[static_cast<long long>(n) * kv_dim + j] = src_row[q_dim + j];
    }
    // V block: cols [q_dim + kv_dim, q_dim + 2*kv_dim)
    for (int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); j < kv_dim;
         j += static_cast<int>(blockDim.x * gridDim.x)) {
        v_out[static_cast<long long>(n) * kv_dim + j] = src_row[q_dim + kv_dim + j];
    }
}

/// The same split, windowed from device memory.
///
/// The peel device-window form (`north-star-dsl.md`, the device-window
/// campaign): the grid spans every lane the buffers can hold and
/// out-of-window rows early-out, so one captured launch replays across row
/// splits. Buffers are BASE pointers -- the host-window form is handed
/// pointers the caller has already offset, and passing those here would
/// window twice.
template <class T>
__global__ void split_qkv_devwin(
    const T* __restrict__ src,
    T* __restrict__ q_out,
    T* __restrict__ k_out,
    T* __restrict__ v_out,
    const u32* __restrict__ win,
    i32 q_dim, i32 kv_dim)
{
    const int n = static_cast<int>(blockIdx.y);
    const int w0 = static_cast<int>(win[0]);
    const int w1 = static_cast<int>(win[1]);
    if (n < w0 || n >= w0 + w1) return;
    const int stride = q_dim + 2 * kv_dim;
    const T* src_row = src + static_cast<long long>(n) * stride;
    for (int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); j < q_dim;
         j += static_cast<int>(blockDim.x * gridDim.x)) {
        q_out[static_cast<long long>(n) * q_dim + j] = src_row[j];
    }
    for (int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); j < kv_dim;
         j += static_cast<int>(blockDim.x * gridDim.x)) {
        k_out[static_cast<long long>(n) * kv_dim + j] = src_row[q_dim + j];
    }
    for (int j = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x); j < kv_dim;
         j += static_cast<int>(blockDim.x * gridDim.x)) {
        v_out[static_cast<long long>(n) * kv_dim + j] =
            src_row[q_dim + kv_dim + j];
    }
}

}  // namespace pie::attn
