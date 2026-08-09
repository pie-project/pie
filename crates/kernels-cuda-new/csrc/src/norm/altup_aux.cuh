//===-- altup_aux.cuh - AltUp's aux kernels: the whole of the C++ ------===//
//
// Six `__global__` templates and one include. There is no host function here,
// no `<<<>>>` and no entry point -- this file is the ONLY C++ these kernels
// have, and everything else about them is a row in
// `kernels_cuda::norm_device`.
//
// # Where the include resolves, and why that is the whole point
//
// `driver-cuda/src/program/compile.rs` states the rule for every source NVRTC
// sees in this tree: *"a `#include` appearing in an emitted source is a bug in
// the emitter, and the right place to find that out is a compile error rather
// than a search path that silently resolves it against whatever CUDA toolkit
// is installed."* That rule was written for the PTIR emitter and it still
// holds there -- an EMITTED source has no business including anything.
//
// An AUTHORED kernel header is a different category, and it took a stage to
// find that out. This file was first written with no include at all, which
// cost it `device::bf16` (replaced by a `unsigned short` wrapper and two
// conversions written out) and `cooperative_groups` (replaced by
// `__shfl_down_sync`). That was the right price for ONE file and the wrong
// price for a family of them: the second family to widen a bf16 would have
// restated the same arithmetic, and a fifth would have restated it wrong.
//
// So the rule for authored sources is narrower than "no includes":
//
//   No include path on disk. Includes resolve against a header set carried in
//   the binary, or they do not resolve at all.
//
// `driver-cuda/src/bind/headers.rs` is that set, handed to NVRTC as
// `headers[]`/`includeNames[]` -- an in-memory virtual filesystem. So the
// include below is resolved by NAME out of the Rust binary, never from a
// path, and a machine with no CUDA toolkit still compiles this file. That is
// what keeps a toolkit-free RUN, which is worth as much as the toolkit-free
// build and is much easier to lose.
//
// nvcc resolves the same line the ordinary way, through the `-I csrc/src`
// that `abi::emit_device_typecheck`'s TU already compiles with. Two
// resolvers, one spelling, and the offline typecheck keeps working -- which
// is what makes a drifted row a build error rather than a failed fire.
//
// # No entry points
//
// The first version of this pilot wrapped each template in an `extern "C"
// __global__` entry so the driver could find a symbol by name. That file is
// gone. `nvrtcAddNameExpression` takes an instantiation as a STRING --
// `pie_cuda_driver::kernels::norm::device::compute_rms<...::bf16>` -- and
// `nvrtcGetLoweredName` answers with the mangled name to look up. So the
// instantiation set is stated by the ROWS, in Rust, and there is nothing left
// in C++ for a human to keep in step with them.
//
// The offline path states the same set the same way: taking the address of
// an instantiation is what forces it, and the generated typecheck TU
// (`abi::emit_device_typecheck`) does exactly that -- so the file that PROVES
// the rows is also the file that instantiates them, and there is still only
// one list.
//
// # What `T` means here
//
// The element type, and nothing else. Every extent that was a template or a
// runtime argument for the LAUNCH's sake is gone: four of the five `T`s these
// kernels used to take were token counts read only to compute a grid or to
// guard a row index the grid covered exactly. `mean_streams` keeps its `t`
// because there it is a STRIDE -- `streams` is `[K, T, H]`, so the k-th plane
// begins at `k * T * H` -- and a stride is layout, which is the kernel's,
// where an extent is geometry, which is the row's.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "pie_device.cuh"

namespace pie_cuda_driver::kernels::norm::device {

// The scalar layer is the PRELUDE's, not this family's. Named here so the
// kernels below read as they always did, and so a row may keep spelling its
// element type `device::bf16` -- which is where it lives.
using ::pie_cuda_driver::kernels::device::bf16;
using ::pie_cuda_driver::kernels::device::block_sum;
using ::pie_cuda_driver::kernels::device::f16;
template <class T>
using Elem = ::pie_cuda_driver::kernels::device::Elem<T>;

/// Per-row RMS of `ref` ([rows, H]) into `out` ([rows]).
///
/// One block per row. There is no `rows` argument and no bound check on the
/// row axis, and that is the launch rule's promise rather than this kernel's
/// assumption: `LaunchRule::Rms` IS "one block per row", so a grid that
/// covered a different number of rows would be a different rule.
template <class T>
__global__ void compute_rms(const T* __restrict__ ref,
                            float* __restrict__ out, int H, float eps) {
    const int t = blockIdx.x;
    const int tid = static_cast<int>(threadIdx.x);
    const T* row = ref + static_cast<long long>(t) * H;

    extern __shared__ float smem[];
    float local = 0.f;
    for (int h = tid; h < H; h += static_cast<int>(blockDim.x)) {
        const float v = Elem<T>::to_f32(row[h]);
        local += v * v;
    }
    const float total = block_sum(local, smem);
    if (tid == 0) {
        out[t] = sqrtf(fmaxf(total / static_cast<float>(H), eps));
    }
}

/// Rescale `x` ([rows, H]) in place so each row's RMS matches `target_rms`.
template <class T>
__global__ void magnitude_rescale(T* __restrict__ x,
                                  const float* __restrict__ target_rms, int H, float eps) {
    const int t = blockIdx.x;
    const int tid = static_cast<int>(threadIdx.x);
    T* row = x + static_cast<long long>(t) * H;

    extern __shared__ float smem[];
    float local = 0.f;
    for (int h = tid; h < H; h += static_cast<int>(blockDim.x)) {
        const float v = Elem<T>::to_f32(row[h]);
        local += v * v;
    }
    const float total = block_sum(local, smem);
    __shared__ float scale;
    if (tid == 0) {
        const float new_rms = sqrtf(fmaxf(total / static_cast<float>(H), eps));
        scale = target_rms[t] / new_rms;
    }
    __syncthreads();

    for (int h = tid; h < H; h += static_cast<int>(blockDim.x)) {
        row[h] = Elem<T>::from_f32(Elem<T>::to_f32(row[h]) * scale);
    }
}

/// Mean across the K-stream axis: `out[t, h] = (1/K) sum_k streams[k, t, h]`.
///
/// `t` is the plane STRIDE, not the extent -- see the header comment. The
/// channel guard stays because the channel axis is rounded up to a block.
template <class T>
__global__ void mean_streams(const T* __restrict__ streams,
                             T* __restrict__ out, int K, int t_stride, int H) {
    const int t = blockIdx.x;
    const int h = static_cast<int>(blockIdx.y * blockDim.x + threadIdx.x);
    if (h >= H) return;

    const long long plane = static_cast<long long>(t_stride) * H;
    float sum = 0.f;
    for (int k = 0; k < K; ++k) {
        sum += Elem<T>::to_f32(streams[k * plane + static_cast<long long>(t) * H + h]);
    }
    out[static_cast<long long>(t) * H + h] = Elem<T>::from_f32(sum / static_cast<float>(K));
}

/// `out[t, j, k] = in[t, k * K + j]` -- HF's `permute(last two)`, widened to
/// fp32.
///
/// Strided over the row rather than one thread per element, so the block
/// width is a launch decision instead of part of the contract. Before the
/// loop, a block narrower than `K * K` computed a PREFIX of the row and
/// reported nothing.
template <class T>
__global__ void unpack_predict_coefs(const T* __restrict__ in,
                                     float* __restrict__ out, int K) {
    const int t = blockIdx.x;
    for (int kk = static_cast<int>(threadIdx.x); kk < K * K;
         kk += static_cast<int>(blockDim.x)) {
        const int k = kk / K;
        const int j = kk % K;
        const float v = Elem<T>::to_f32(in[static_cast<long long>(t) * K * K +
                                     static_cast<long long>(k) * K + j]);
        out[static_cast<long long>(t) * K * K + static_cast<long long>(j) * K + k] = v;
    }
}

/// `out[t, k] = in[t, k] + 1.0` -- HF's `+ 1.0`, widened to fp32.
template <class T>
__global__ void unpack_correct_coefs(const T* __restrict__ in,
                                     float* __restrict__ out, int K) {
    const int t = blockIdx.x;
    for (int k = static_cast<int>(threadIdx.x); k < K; k += static_cast<int>(blockDim.x)) {
        const float v = Elem<T>::to_f32(in[static_cast<long long>(t) * K + k]);
        out[static_cast<long long>(t) * K + k] = v + 1.0f;
    }
}

/// Element-wise tanh, in place, over a flat extent.
template <class T>
__global__ void tanh_inplace(T* __restrict__ x, int n) {
    const int idx = static_cast<int>(blockIdx.x * blockDim.x + threadIdx.x);
    if (idx >= n) return;
    x[idx] = Elem<T>::from_f32(tanhf(Elem<T>::to_f32(x[idx])));
}

}  // namespace pie_cuda_driver::kernels::norm::device
