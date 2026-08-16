//===-- head_dim_pad.cuh - reaching a supported head_dim -----------------===//
//
// Two `__global__` templates, no host code.
//
// flashinfer compiles its attention kernels for a fixed set of head widths --
// 64, 128, 256, 512 -- and a model whose `head_dim` is none of them cannot
// use them at all. Phi-3-mini ships 96. These two kernels buy that model the
// dense path: pad every head out to the next supported width on the way in,
// strip the padding on the way out.
//
// The zero pad is not arbitrary. `q_e . k_e = q[:d] . k[:d]` when both are
// zero-extended, so the score is UNCHANGED; and a zero V channel contributes
// nothing to the weighted sum, so the output's padding columns are zero and
// the strip is exact. Any other filler changes the attention.
//
// # NO ROW STATES EITHER KERNEL
//
// Both were `<<<dim3(num_heads, num_tokens), 128>>>` -- one block per (token,
// head), the head on the grid's x axis. That is `LaunchRule::PerHead`, and
// this backend's `runtime::launch` does not evaluate it: `Dims` carries
// `rows`, `width` and `in_width`, and a per-head grid needs a head COUNT and
// a head WIDTH, which are two numbers `Dims` has no fields for. `KernelSig`
// already anticipates the shape -- `head_param` exists precisely because
// gemma-4's full-attention layers carry four KV heads of 512 channels where
// its sliding layers carry sixteen of 256 -- but the CUDA rule evaluator has
// not been given them yet.
//
// Approximating it with `ElementwiseRows` was considered and refused. The
// grids differ: `ElementwiseRows` opens `[rows, ceil(width / 256)]`, so the
// head index would have to be recovered inside the kernel from a flat
// channel, and `head_dim_padded != head_dim` means the input and output
// channel maps are DIFFERENT functions of that flat index. A rule that
// almost fits is worse than one that does not, because the kernel that
// bridges the gap is where the next reader looks last.
//
// So the device text is here, once, NVRTC-clean, and the `.cu` keeps the
// `<<<>>>`. When `PerHead` lands the diff is two rows.
//
// # Why templates for a pure copy
//
// The strip is a copy and the pad is a copy plus zeros -- neither reads a
// value it does not write. Templating on `Elem<T>` is still what the
// row table wants (`DeviceKernel::instantiation()` spells exactly one type
// argument), and it costs nothing: `Elem<T>::from_f32(0.f)` is the only
// arithmetic either kernel does, and it is a compile-time constant per
// format.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "prelude/device.cuh"

namespace pie::attn {


/// The block width both launchers open. 128 rather than 256 because the loop
/// bound is `head_dim_padded` -- 128 or 256 in practice -- and a wider block
/// would leave most of it idle on the common case.
///
/// `[[maybe_unused]]` because this unit states no rows: a compile that
/// instantiates neither template never parses a use of it.
[[maybe_unused]] constexpr int kPadBlock = 128;

/// Copy `head_dim` values per (token, head) and zero the trailing columns.
///
/// Threads stride over the PADDED extent so every thread executes exactly one
/// store -- a copy or a zero -- rather than one branch executing and the
/// other stalling. Same instruction count either side of the boundary.
template <class T>
__global__ void pad_head_dim(
    const T* __restrict__ packed,
    T* __restrict__ padded,
    i32 num_heads, i32 head_dim, i32 head_dim_padded)
{
    const int n = static_cast<int>(blockIdx.y);
    const int h = static_cast<int>(blockIdx.x);
    const T* in =
        packed + (static_cast<long long>(n) * num_heads + h) * head_dim;
    T* out =
        padded + (static_cast<long long>(n) * num_heads + h) * head_dim_padded;
    for (int d = static_cast<int>(threadIdx.x); d < head_dim_padded;
         d += kPadBlock) {
        out[d] = (d < head_dim) ? in[d] : Elem<T>::from_f32(0.f);
    }
}

/// The inverse: copy `head_dim` values back, dropping the padding columns.
template <class T>
__global__ void strip_head_dim(
    const T* __restrict__ padded,
    T* __restrict__ packed,
    i32 num_heads, i32 head_dim, i32 head_dim_padded)
{
    const int n = static_cast<int>(blockIdx.y);
    const int h = static_cast<int>(blockIdx.x);
    const T* in =
        padded + (static_cast<long long>(n) * num_heads + h) * head_dim_padded;
    T* out =
        packed + (static_cast<long long>(n) * num_heads + h) * head_dim;
    for (int d = static_cast<int>(threadIdx.x); d < head_dim;
         d += kPadBlock) {
        out[d] = in[d];
    }
}

}  // namespace pie::attn
