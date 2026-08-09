//===-- vision/tower_naive_kernels.cuh - the towers' shared naive text ---===//
//
// The six `__global__`s more than one VISION TOWER launches, as templates over
// the storage format, in a namespace NVRTC can name.
//
// # Why this exists
//
// `k_matmul` and `k_rms` were byte-identical in `model/csm/` and
// `model/gemma4/` -- fingerprinted, not read. Every tower wrote its kernels in
// an anonymous namespace, so no author could see the copy next door; that
// invisibility was the mechanism, and it produced the same scalar matmul three
// times and the same RMSNorm twice. Collapsing them into one header was the
// first half of the fix and landed long ago.
//
// This file is the second half, and it is what the JIT needed. The kernels
// were `__global__ void k_matmul(const bf*, ...)` inside
// `namespace pie_cuda_driver::model { namespace { ... } }`: unnamed at file
// scope, so `nvrtcAddNameExpression` has nothing to hand NVRTC and the
// runtime cannot get a `CUfunction` for any of them; and non-template, so a
// header included by three translation units emits three strong definitions
// and the ahead-of-time link fails on the duplicates the moment the anonymous
// namespace goes. Both problems have the same answer -- a NAMED template --
// which is why the two changes are one change.
//
// # What is unchanged, and what moved
//
// The arithmetic. Every line below is the original's, in the original's fold
// ORDER, with three substitutions and nothing else:
//
//   * `bf` became `T`, and `F`/`Bf` became `Elem<T>::to_f32`/`from_f32`
//     through the two helpers below. `bf16_to_f32` is a sixteen-bit shift and
//     `f32_to_bf16` is round-to-nearest-even, which is what
//     `__bfloat162float` and `__float2bfloat16` are; `norm/elementwise.cuh`
//     made the same swap first and `residual_add`'s parity harness saw no
//     difference. Nothing here is instantiated at `f16` yet -- the towers are
//     bf16 end to end -- but a second format is now a row rather than a build.
//   * The flat element counts became `usize` from `long`. Both are 64 bits on
//     this target, the counts are element totals and so never negative, and
//     `usize` is the one `Ty` a row can state for them: `Ty::I64` is
//     `long long`, which mangles differently from `long` and would put a lie
//     in `emit_device_typecheck`'s function-pointer initialiser.
//   * `CUDART_INF_F` became `device::pos_inf()`. Same bit pattern, and
//     `math_constants.h` is one of the 31 headers NVRTC answered 0 of.
//
// The fold in `k_rms` and `k_layernorm` is deliberately NOT `device::block_sum`.
// `block_sum` reduces the per-warp partials with a second warp shuffle; these
// sum them serially in thread 0. The two orders agree to within a rounding of
// the last bit and the towers were parity-checked against HF-bf16 dumps at
// the order below, so the order below is what stays.
//
// # What this is still a way station for
//
// The answer is to stop having these at all: `gemm::act_x_wt_bf16` and
// `norm::rmsnorm_bf16` already exist and are what these should call. That swap
// changes the arithmetic -- a naive scalar loop is not cuBLAS -- so it needs
// the tower parity harnesses (`gemma4_vision_full_parity_bf16`,
// `csm_backbone_parity`) run against reference dumps. Templating identical
// bodies does not, which is why this can land and that cannot.
//
// Reading the seven that looked different settled each one, and the findings
// are kept because they are why this header holds six kernels and not thirteen:
//
//   k_add          SAME. Three copies whose only difference was parameter
//                  NAMES (a/b vs h/x) and a line break. Here.
//   k_f32_to_bf16  SAME. Not two definitions at all -- qwen3_vl forward-
//                  declares it and defines it later in the same file, and the
//                  fingerprint had swallowed the next kernel. Here.
//   k_layernorm    SAME ALGORITHM, wider contract. qwen3_vl's takes gamma/beta
//                  as optional (`g ? F(g[d]) : 1.f`) where mimi's dereferences
//                  them; mimi always passes non-null, so the general one is
//                  bit-identical for both. The general one is here.
//   k_gelu         DIFFERENT FUNCTION. mimi computes the exact erf form
//                  (`transformers` ACT2FN["gelu"]); qwen3_vl the tanh
//                  approximation. Merging them would have been a silent
//                  numerics change. Both stay put, and a shared `mlp::gelu`
//                  will have to offer both.
//   k_addpos       DIFFERENT OP. gemma4 indexes a 2-D grid position table
//                  twice per token; qwen3_vl adds a precomputed vector.
//   k_rope         DIFFERENT OP. csm is 1-D with YaRN scaling, gemma4 is
//                  2-D axial. See .wiki/kernel-refactor.md §2.2.
//   k_attn         SAME SHAPE, different axes -- one takes a q-offset into a
//                  KV cache, the other a sliding window. One parameterised
//                  kernel could cover both, which makes it a numerics change
//                  and so a job for the parity harnesses.
//
//===---------------------------------------------------------------------===//
#pragma once

#include "pie_device.cuh"

namespace pie_cuda_driver::kernels::vision::device {

using ::pie_cuda_driver::kernels::device::Elem;
using ::pie_cuda_driver::kernels::device::bf16;
using ::pie_cuda_driver::kernels::device::f16;
using ::pie_cuda_driver::kernels::device::usize;

/// The originals' `F` and `Bf`, kept under their original names so the split
/// reads as the move it is.
///
/// `F` deduces; `Bf` cannot -- nothing in `Bf(0.f)` says which format to
/// narrow to -- so every call site spells `Bf<T>`. That one character is the
/// whole cost of the diff being mechanical.
template <class T>
__device__ __forceinline__ float F(T x) { return Elem<T>::to_f32(x); }
template <class T>
__device__ __forceinline__ T Bf(float x) { return Elem<T>::from_f32(x); }

/// Scalar row-major `y = x * W^T`, one thread per output element.
///
/// UNROWED, and the launcher says why:
/// `k_matmul<<<G2(O,N), B2, 0, S>>>` with `B2 = dim3(16,16)`. A 2-D block is
/// outside the ported `LaunchRule` vocabulary -- every rule there states
/// `block = [n,1,1]` -- and `new-horizon.md` §17.9 is explicit that a rule
/// stretched to fit is worse than a kernel left unrowed.
template <class T>
__global__ void k_matmul(const T* x, const T* W, T* y, int N, int K, int O) {
    int n = blockIdx.y * blockDim.y + threadIdx.y, o = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N || o >= O) return;
    const T* xr = x + (long)n * K;
    const T* wr = W + (long)o * K;
    float a = 0;
    for (int k = 0; k < K; k++) a += F(xr[k]) * F(wr[k]);
    y[(long)n * O + o] = Bf<T>(a);
}

/// RMSNorm with an OPTIONAL weight (`w == nullptr` means unit gain).
///
/// `Rule::PerRow`: `k_rms<<<R, 256, 0, S>>>` at every one of its eight audio
/// call sites and its two tower ones, and `per_row` evaluates to
/// `grid[rows,1,1] block[256,1,1] smem 0`. The shared memory here is STATIC
/// (`__shared__ float warp[32], ss;`), so the launcher's zero dynamic bytes is
/// the whole of the contract and `Rule::Rms`'s 32 dynamic bytes would be an
/// allocation nothing reads.
template <class T>
__global__ void k_rms(const T* x, const T* w, T* o, int R, int D, float eps) {
    int r = blockIdx.x;
    if (r >= R) return;
    const T* xr = x + (long)r * D;
    T* orow = o + (long)r * D;
    float loc = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) { float v = F(xr[d]); loc += v * v; }
    for (int s = warpSize / 2; s > 0; s >>= 1) loc += __shfl_down_sync(0xffffffff, loc, s);
    __shared__ float warp[32], ss;
    if ((threadIdx.x & 31) == 0) warp[threadIdx.x >> 5] = loc;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0;
        int nw = (blockDim.x + 31) / 32;
        for (int i = 0; i < nw; i++) t += warp[i];
        ss = rsqrtf(t / D + eps);
    }
    __syncthreads();
    float inv = ss;
    for (int d = threadIdx.x; d < D; d += blockDim.x) orow[d] = Bf<T>(F(xr[d]) * inv * (w ? F(w[d]) : 1.f));
}

/// In-place `a[i] += b[i]`.
///
/// `Rule::Elementwise`: `k_add<<<(n+255)/256, 256, 0, S>>>` at both audio call
/// sites, and `elementwise` evaluates `rows * width` to the same
/// `ceil(n/256)` blocks of 256.
///
/// The same four lines as `norm::device::residual_add`, and they stay separate
/// on purpose: this one's operands are `[n]`-flat with no notion of rows, and
/// merging them would put a vision tower's launch on a row whose `Dims` the
/// planner fills for a decode step.
template <class T>
__global__ void k_add(T* a, const T* b, usize n) {
    usize i = blockIdx.x * (usize)blockDim.x + threadIdx.x;
    if (i < n) a[i] = Bf<T>(F(a[i]) + F(b[i]));
}

/// Narrow a float buffer into the storage format.
///
/// `Rule::Elementwise`: `k_f32_to_bf16<<<(n+255)/256, 256, 0, S>>>` at all
/// three call sites (two gemma4, one qwen3-vl).
///
/// The name kept `bf16` after the template because the ROW is what a caller
/// reads and the row names the format; renaming it to `k_f32_to_elem` would
/// have broken the one property the split is supposed to preserve, which is
/// that every kernel here is findable by the name it had.
template <class T>
__global__ void k_f32_to_bf16(const float* a, T* o, usize n) {
    usize i = blockIdx.x * (usize)blockDim.x + threadIdx.x;
    if (i < n) o[i] = Bf<T>(a[i]);
}

/// Exact GELU: `0.5*x*(1+erf(x/sqrt(2)))` -- transformers' ACT2FN["gelu"], and
/// `nn.GELU(approximate='none')`.
///
/// `Rule::Elementwise`: `k_gelu_erf<<<(n+255)/256, 256, 0, S>>>` at its one
/// qwen3-vl call site.
///
/// The name carries `_erf` because the OTHER form exists in this tree:
/// qwen3_vl's patch tower uses the tanh approximation, and for a while both
/// were called `k_gelu` in different files. Merging those two by name would
/// have changed numerics silently; keeping the form in the name is what makes
/// the duplicate a real duplicate.
///
/// mimi and qwen3_vl each had this, spelled differently (`if(i>=n)return` vs a
/// guarded block) with the sqrt(2) reciprocal written to different lengths --
/// 0.70710678118f and 0.70710678118654752f both round to the same float, so
/// the two were bit-identical all along.
template <class T>
__global__ void k_gelu_erf(const T* x, T* o, usize n) {
    usize i = blockIdx.x * (usize)blockDim.x + threadIdx.x;
    if (i >= n) return;
    float v = F(x[i]);
    o[i] = Bf<T>(0.5f * v * (1.f + erff(v * 0.70710678118654752f)));
}

/// LayerNorm with OPTIONAL gamma and beta.
///
/// `Rule::PerRow`: `k_layernorm<<<R, 256, 0, S>>>` at all four qwen3-vl call
/// sites, and `per_row` evaluates to `grid[rows,1,1] block[256,1,1] smem 0`.
/// Shared memory is static here for the same reason as `k_rms`.
///
/// Two passes and two folds, in the original's order: the mean is reduced,
/// broadcast, and only then does the variance pass start. Fusing them into one
/// Welford pass is a different sum and so a numerics change.
template <class T>
__global__ void k_layernorm(const T* x, const T* g, const T* bta, T* o, int R, int D, float eps) {
    int r = blockIdx.x;
    if (r >= R) return;
    const T* xr = x + (long)r * D;
    T* orow = o + (long)r * D;
    float sum = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) sum += F(xr[d]);
    for (int s = warpSize / 2; s > 0; s >>= 1) sum += __shfl_down_sync(0xffffffff, sum, s);
    __shared__ float warp[32], smean, svar;
    if ((threadIdx.x & 31) == 0) warp[threadIdx.x >> 5] = sum;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0;
        int nw = (blockDim.x + 31) / 32;
        for (int i = 0; i < nw; i++) t += warp[i];
        smean = t / D;
    }
    __syncthreads();
    float mean = smean, v = 0;
    for (int d = threadIdx.x; d < D; d += blockDim.x) { float dx = F(xr[d]) - mean; v += dx * dx; }
    for (int s = warpSize / 2; s > 0; s >>= 1) v += __shfl_down_sync(0xffffffff, v, s);
    if ((threadIdx.x & 31) == 0) warp[threadIdx.x >> 5] = v;
    __syncthreads();
    if (threadIdx.x == 0) {
        float t = 0;
        int nw = (blockDim.x + 31) / 32;
        for (int i = 0; i < nw; i++) t += warp[i];
        svar = rsqrtf(t / D + eps);
    }
    __syncthreads();
    float inv = svar;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        float nrm = (F(xr[d]) - mean) * inv;
        orow[d] = Bf<T>(nrm * (g ? F(g[d]) : 1.f) + (bta ? F(bta[d]) : 0.f));
    }
}

}  // namespace pie_cuda_driver::kernels::vision::device
