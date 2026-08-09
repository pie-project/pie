//===-- altup_aux_device.cuh - AltUp's aux kernels, device side only ---===//
//
// The same six kernels `altup_aux.cu` holds, with everything that is not
// device code removed. There is no `<<<>>>` here, no `dim3`, no
// `cudaStream_t`, and no host function of any kind: a grid is a decision
// about a fire and this file is what runs once the decision is made.
//
// That is the whole of the experiment `.wiki/new-driver/` calls Tier A. The
// launch policy each of these carried -- `BLOCK = 256`, `(H + BLOCK - 1) /
// BLOCK`, the `T <= 0` guard, the `static_cast` off a `void*` -- is not
// kernel knowledge that happens to be written in C++. It is the same fact
// `KernelSig::launch` already states for Metal, written a second time in a
// language the table cannot read. So it is stated once, on the row, and
// `driver-cuda`'s `bind::launch` evaluates it.
//
// # Templated, and what that buys
//
// Every kernel below takes its element type as a parameter, where
// `altup_aux.cu` hard-coded `__nv_bfloat16` and took its buffers as `void*`.
// The `void*` was not a choice either: a host launcher is the only thing
// that crossed the ABI, so the ABI's type had to be the widest one. A
// device entry point names the type it reads, and the SET of entry points
// is what an instantiation manifest states -- `kernels.def`'s job, moved to
// where the rest of the row already lives.
//
// # The stride loops
//
// `unpack_predict_coefs` and `unpack_correct_coefs` were written with one
// thread per element and no loop, which made `blockDim.x` part of their
// contract rather than a launch decision: a block narrower than the row
// silently computed a prefix of it. They stride now, so any width is
// correct and the widest one is merely fastest. Nothing else changed --
// both are pure maps, so the result is unchanged element for element.
//
// # Which extents survived, and why that is the interesting number
//
// Five of the six launchers took a `T` -- the token count -- and four of
// them took it ONLY to compute a grid or to guard a row index the grid
// already covered exactly. Those four are gone from the parameter lists
// below: `LaunchRule::Rms` and `LaunchRule::ElementwiseRows` both put one
// block on each row, so `blockIdx.x < T` holds by construction and a
// kernel that re-checked it was checking the launcher's arithmetic.
//
// `mean_streams` kept its `T`, and the exception is the point: there it is
// a STRIDE (`streams` is `[K, T, H]`, so the k-th plane starts at `k * T *
// H`), not an extent. An extent is launch geometry and belongs to the
// rule; a stride is layout and belongs to the kernel. The old signatures
// spelled both the same way, which is why the distinction was invisible.
//
//===----------------------------------------------------------------------===//
#pragma once

#include <cooperative_groups.h>
#include <cuda_bf16.h>

namespace pie_cuda_driver::kernels::norm::device {

namespace cg = cooperative_groups;

// Block-wide reduction of `local` (one float per thread) to thread 0.
// Uses warp shfl + shared-mem combine. Returns the reduced value;
// only thread 0 has it. `smem` must be sized for `blockDim.x / 32` floats.
__device__ __forceinline__ float block_sum(float local, float* smem) {
    auto tile = cg::tiled_partition<32>(cg::this_thread_block());
    for (int off = 16; off > 0; off >>= 1) {
        local += tile.shfl_down(local, off);
    }
    if (tile.thread_rank() == 0) smem[tile.meta_group_rank()] = local;
    __syncthreads();
    if (tile.meta_group_rank() == 0) {
        float v = (threadIdx.x < tile.meta_group_size()) ? smem[threadIdx.x] : 0.f;
        for (int off = 16; off > 0; off >>= 1) {
            v += tile.shfl_down(v, off);
        }
        if (tile.thread_rank() == 0) smem[0] = v;
    }
    __syncthreads();
    return smem[0];
}

/// `T -> float`, so the kernels below can be written once. Specialized
/// rather than overloaded because `__nv_bfloat16` and `__half` both convert
/// from `float` implicitly and an overload set would pick by accident.
template <class T>
struct Elem;

template <>
struct Elem<__nv_bfloat16> {
    static __device__ __forceinline__ float to_f32(__nv_bfloat16 v) {
        return __bfloat162float(v);
    }
    static __device__ __forceinline__ __nv_bfloat16 from_f32(float v) {
        return __float2bfloat16(v);
    }
};

template <>
struct Elem<__half> {
    static __device__ __forceinline__ float to_f32(__half v) { return __half2float(v); }
    static __device__ __forceinline__ __half from_f32(float v) { return __float2half(v); }
};

/// Per-row RMS of `ref` ([T, H]) into `out` ([T]).
///
/// One block per row. The row axis is the grid's and carries no bound
/// check, which is the launch rule's promise rather than this kernel's
/// assumption -- `LaunchRule::Rms` is "one block per row" and a grid that
/// covered fewer rows would be a different rule.
template <class T>
__device__ __forceinline__ void compute_rms(const T* __restrict__ ref, float* __restrict__ out, int H, float eps) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    const T* row = ref + (long long)t * H;

    extern __shared__ float smem[];
    float local = 0.f;
    for (int h = tid; h < H; h += blockDim.x) {
        const float v = Elem<T>::to_f32(row[h]);
        local += v * v;
    }
    const float total = block_sum(local, smem);
    if (tid == 0) {
        const float mean_sq = total / static_cast<float>(H);
        out[t] = sqrtf(fmaxf(mean_sq, eps));
    }
}

/// Rescale `x` ([T, H]) in place so each row's RMS matches `target_rms[t]`.
template <class T>
__device__ __forceinline__ void magnitude_rescale(T* __restrict__ x, const float* __restrict__ target_rms, int H,
                                  float eps) {
    const int t = blockIdx.x;
    const int tid = threadIdx.x;
    T* row = x + (long long)t * H;

    extern __shared__ float smem[];
    float local = 0.f;
    for (int h = tid; h < H; h += blockDim.x) {
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

    for (int h = tid; h < H; h += blockDim.x) {
        const float v = Elem<T>::to_f32(row[h]) * scale;
        row[h] = Elem<T>::from_f32(v);
    }
}

/// Mean across the K-stream axis: `out[t, h] = (1/K) sum_k streams[k, t, h]`.
///
/// The row is the grid's first axis and the channel its second, which is
/// `LaunchRule::ElementwiseRows` -- a pointwise pass whose rows are not
/// contiguous, so they do not stack flat.
template <class T>
__device__ __forceinline__ void mean_streams(const T* __restrict__ streams, T* __restrict__ out, int K, int T_,
                             int H) {
    const int t = blockIdx.x;
    const int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= T_ || h >= H) return;

    const long long stride = (long long)T_ * H;
    float sum = 0.f;
    for (int k = 0; k < K; ++k) {
        sum += Elem<T>::to_f32(streams[(long long)k * stride + (long long)t * H + h]);
    }
    out[(long long)t * H + h] = Elem<T>::from_f32(sum / static_cast<float>(K));
}

/// `out[t, j, k] = in[t, k * K + j]` -- HF's `permute(last two)`, widened
/// to fp32.
template <class T>
__device__ __forceinline__ void unpack_predict_coefs(const T* __restrict__ in,
                                                     float* __restrict__ out, int K) {
    const int t = blockIdx.x;
    for (int kk = threadIdx.x; kk < K * K; kk += blockDim.x) {
        const int k = kk / K;
        const int j = kk % K;
        const float v = Elem<T>::to_f32(in[(long long)t * K * K + (long long)k * K + j]);
        out[(long long)t * K * K + (long long)j * K + k] = v;
    }
}

/// `out[t, k] = in[t, k] + 1.0` -- HF's `+ 1.0`, widened to fp32.
template <class T>
__device__ __forceinline__ void unpack_correct_coefs(const T* __restrict__ in,
                                                     float* __restrict__ out, int K) {
    const int t = blockIdx.x;
    for (int k = threadIdx.x; k < K; k += blockDim.x) {
        const float v = Elem<T>::to_f32(in[(long long)t * K + k]);
        out[(long long)t * K + k] = v + 1.0f;
    }
}

/// Element-wise tanh, in place, over a flat extent.
template <class T>
__device__ __forceinline__ void tanh_inplace(T* __restrict__ x, int n) {
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    const float v = Elem<T>::to_f32(x[idx]);
    x[idx] = Elem<T>::from_f32(tanhf(v));
}

}  // namespace pie_cuda_driver::kernels::norm::device
