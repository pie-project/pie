//===-- rmsnorm.cuh - the norm family's twelve RMSNorm kernels -----------===//
//
// Twelve `__global__` templates and one `__device__` reduction, and nothing
// else: no host function, no `<<<>>>`, no entry point. Four of them are named
// by rows in `kernels_cuda::families::norm`; the rest are reachable only
// through the ahead-of-time launchers in `rmsnorm.cu`, and this header says
// which and why.
//
// # What a launcher was doing, and where it went
//
// The scalar launchers were all one shape:
//
//     dim3 grid(num_rows);
//     kernel<BLOCK><<<grid, 256, 0, stream>>>(..., hidden_size, eps);
//
// -- one block per row, 256 threads, a block-wide reduction between two
// passes over the row. That is `LaunchRule::Rms` exactly, so `num_rows` left
// the kernel signatures with the rule that recovers it and the rows say what
// the launchers said. The rule also hands the launch 32 bytes of dynamic
// shared memory for `block_sum`'s warp scratch; these kernels reduce
// through `block_reduce_sum_exact` on a STATIC `__shared__ float[BLOCK]`
// instead and never read it. That is deliberate -- see the reduction's own
// note -- and the unread 32 bytes cost nothing.
//
// # `BLOCK`'s default is a claim about rows
//
// `BLOCK` defaults to 256 -- the width `LaunchRule::Rms` fixes -- on every
// kernel whose scalar launcher used 256, so that a row's one-argument
// instantiation `name<T>` IS the launch the rule produces. Where a launcher
// chose another width the parameter has no default and the caller must spell
// it. A defaulted `BLOCK` is therefore a statement that the rule and the
// kernel agree; an undefaulted one is a statement that they do not.
//
// # The eight no row names, and why
//
// * **The four vectorised kernels** (`rmsnorm_vec8`,
//   `residual_add_rmsnorm_vec8`, `rmsnorm_rasr_vec8`) are chosen AT RUN TIME
//   by `rmsnorm_vec8_ok`, which inspects pointer alignment and row pitch --
//   a fact about the buffers a fire was handed, not about its shape. No
//   `LaunchRule` carries a precondition on an operand's ADDRESS, and they
//   launch 512 threads where `Rms` fixes 256. The rows fire the scalar twin,
//   which is what the vectorised forms were measured against: bit-identical
//   at hidden 2048/2816/5376 for the `rasr` pair, and reassociated only in
//   the sum order for the others. So a row is slower, never wrong.
//
// * **`rmsnorm_residual_add_scale_rmsnorm`** has no row because its SCALAR
//   launcher is 512 threads wide, not 256. `Rms` would re-associate the two
//   reductions -- 8 shared-memory levels instead of 9 -- and the fused
//   four-statement landing is the one kernel in this file whose output feeds
//   both the residual stream and the next block's norm, so a moved last bit
//   compounds twice per layer.
//
// * **`rmsnorm`, `rmsnorm_gemma`, `rmsnorm_no_scale` and `rmsnorm_gated`**
//   are each named by a symbol with TWO readings. `OpKind::RmsnormPerHead`
//   lowers to the same symbols as `OpKind::Rmsnorm` and norms
//   `rows · (width / head_dim)` rows of `head_dim` where the plain kind norms
//   `rows` of `width`; the ahead-of-time rows spell that with
//   `Source::IfPresent(&Source::PerHeadDim, ...)`. `LaunchRule::Rms` reads
//   `dims.rows` and nothing else, so a JIT row would launch one block per
//   TOKEN and each block would norm a whole q projection as a single row --
//   gemma-4's heads silently averaged together. The kernels are here and
//   templated; what is missing is a rule that can see a per-head divisor.
//
// * **`rmsnorm_gated_f32_in`** is stated by no table row at all: it exists
//   for qwen3.5's recurrent step, which reaches it through a hand arm. A row
//   would have to invent its sources rather than derive them from a twin.
//
// # Why they are templates over `T` when the originals were not
//
// The originals were `_bf16` and only `_bf16` because an AOT build has to
// choose its instantiations. The scalar bodies are written over `T` through
// `Elem<T>`, so a second numeric format costs a row instead of a
// translation unit. The four vectorised bodies are NOT: they read rows as
// `float4` and unpack them through `bf16x2`, and there is no packed
// layer on `Elem<T>` to write that against. Templating them would mean
// inventing one for a kernel no row can name yet.
//
// The gated kernels' weights stay fp32 in every instantiation -- qwen3.5
// ships RMSNormGated weights in fp32 alongside bf16 activations, and that is
// the checkpoint's fact, not the element type's.
//
// # There is a CuTile ALTERNATIVE beside this file, and it is FASTER
//
// `rmsnorm_tile.cuh` is the twin: about forty lines against this file's 747,
// no block reduction, no `__shared__`, no warp shuffle, no launcher.
// Numerically IDENTICAL -- worst relative error 0.0000, not "within
// tolerance" -- and faster:
//
//                        CuTile     this file     rmsnorm_vec8
//     H=4096, 1 row      1.94 us      2.93 us
//     H=7168, 1 row      2.42        3.84            2.38
//     H=4096, 2048 rows  9.66       12.37
//
// L40S sm_89, bf16, empty-grid floor 0.65 us. At H=7168 it ties
// `rmsnorm_vec8` -- the hand-vectorised path that needs 16-byte-aligned rows
// and the `rmsnorm_vec8_ok` pointer check, which is exactly the check no
// `LaunchRule` can carry and the reason four kernels in this file have no
// row. A JIT states alignment per instantiation instead of branching per
// fire.
//
// **An earlier version of this note said the opposite** -- 3.84 us against
// 2.93, "a code-size argument, not a speed one." That measured a CuTile
// kernel written in a naive dialect: `partition_view` over a 1-D row,
// `dynamic_extent` for the hidden size, a run-time trip count, no hints.
// Written the way NVIDIA writes it -- `ct::iota` plus a `ct::load` gather,
// `latency=1` on each load, the hidden size a template parameter,
// `assume_aligned<16>`, `ct::sum<0>` down to a plain float -- half the
// runtime came back. The twin's header has the list; `new-horizon.md` §23.20
// has the before-and-after for every kernel in that spike, all of which
// moved.
//
// **The twin is an ALTERNATIVE and this file is not going anywhere.** It
// carries `rmsnorm_tile_preferred`, which answers `true` at every shape
// measured, but a preference is only a preference where the alternative can
// be compiled at all: the twin needs NVRTC 13.3, 13.3 runtime headers and
// `tileiras`, and this crate loads NVRTC 13.0.88. So these twelve kernels
// are the fallback on every machine today, and would remain the fallback on
// any machine whose toolchain is older -- which is what makes the twin an
// addition rather than a removal. `moe/moe_grouped_gemm_tile.cuh` states
// the toolchain floor in full.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "prelude/device.cuh"

namespace pie::norm {


/// Block reduction that reproduces the shared-memory tree
/// `for (off = BLOCK/2; off; off >>= 1) buf[t] += buf[t+off]` **exactly**, but
/// runs the last five levels on warp shuffles. Those levels only ever touch
/// threads 0..31, so the pairing is identical and the fp32 rounding is
/// unchanged -- what goes away is five block-wide barriers, which at decode
/// (one block per token, eight blocks on 148 SMs) were a third of the kernel.
///
/// `__device__`, not a bare function: NVRTC does not forgive an unannotated
/// helper the way nvcc does inside a `.cu`.
template <int BLOCK>
__device__ __forceinline__ float block_reduce_sum_exact(float local, float* buf)
{
    static_assert(BLOCK >= 32 && (BLOCK & (BLOCK - 1)) == 0,
                  "block_reduce_sum_exact needs a power-of-two BLOCK >= 32");
    const int tid = threadIdx.x;
    buf[tid] = local;
    __syncthreads();
#pragma unroll
    for (int off = BLOCK / 2; off >= 32; off >>= 1) {
        if (tid < off) buf[tid] += buf[tid + off];
        __syncthreads();
    }
    if (tid < 32) {
        float v = buf[tid];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, off);
        }
        if (tid == 0) buf[0] = v;
    }
    __syncthreads();
    return buf[0];
}

/// One row of `y = w · x · rsqrt(mean(x²) + eps)`, strided on both sides.
///
/// `WEIGHT_PLUS_ONE` selects Gemma's `(1 + w)` convention. It is a template
/// parameter and not an argument because the two conventions are two kernels
/// -- see `rmsnorm` and `rmsnorm_gemma` below, which are the only two
/// spellings of it and share this body so that a fix to one is a fix to both.
///
/// `BLOCK` threads cooperate on the L2-norm reduction; each thread handles
/// `hidden / BLOCK` elements, which stays small even at hidden 8192.
template <class T, int BLOCK, bool WEIGHT_PLUS_ONE>
__device__ __forceinline__ void rmsnorm_row(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T* __restrict__ y,
    int hidden,
    int x_row_stride,
    int y_row_stride,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const T* xr = x + static_cast<long long>(row) * x_row_stride;
    T* yr = y + static_cast<long long>(row) * y_row_stride;

    float local = 0.f;
    for (int i = tid; i < hidden; i += BLOCK) {
        const float v = Elem<T>::to_f32(xr[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_exact<BLOCK>(local, buf);

    const float inv_rms = rsqrtf(buf_sum / static_cast<float>(hidden) + eps);

    for (int i = tid; i < hidden; i += BLOCK) {
        const float xv = Elem<T>::to_f32(xr[i]);
        float wv = Elem<T>::to_f32(weight[i]);
        if constexpr (WEIGHT_PLUS_ONE) wv += 1.f;
        yr[i] = Elem<T>::from_f32(xv * inv_rms * wv);
    }
}

/// `y = w · x · rsqrt(mean(x²) + eps)`, one block per row.
template <class T, int BLOCK = 256>
__global__ void rmsnorm(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T* __restrict__ y,
    int hidden,
    int x_row_stride,
    int y_row_stride,
    float eps)
{
    rmsnorm_row<T, BLOCK, /*WEIGHT_PLUS_ONE=*/false>(
        x, weight, y, hidden, x_row_stride, y_row_stride, eps);
}

/// Gemma folds `(1 + w)` instead of `w` -- different arithmetic, same
/// signature, same row space.
template <class T, int BLOCK = 256>
__global__ void rmsnorm_gemma(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T* __restrict__ y,
    int hidden,
    int x_row_stride,
    int y_row_stride,
    float eps)
{
    rmsnorm_row<T, BLOCK, /*WEIGHT_PLUS_ONE=*/true>(
        x, weight, y, hidden, x_row_stride, y_row_stride, eps);
}

/// Same math as `rmsnorm`, but each thread owns 8 contiguous bf16 (one
/// 16-byte load) instead of one. At decode `num_rows` is 1, so the kernel is a
/// single block on a 148-SM GPU and its cost is entirely the length of the
/// per-thread dependent load chain: at hidden=7168 the scalar form walked 28
/// loads per thread, twice. Vectorized it is 4 (BLOCK=512), and measured
/// device time dropped ~7x (3.48 -> 2.38 us against a 2.20 us empty-launch
/// floor).
///
/// Requires `hidden % 8 == 0` and 16-byte-aligned rows; the launcher checks
/// with `rmsnorm_vec8_ok` and falls back to the scalar kernel otherwise. That
/// check is why no row names this kernel: alignment is a fact about the
/// pointers a fire was handed, and no `LaunchRule` can see it.
template <int BLOCK, bool WEIGHT_PLUS_ONE, bool EMIT_FP16 = false>
__global__ void rmsnorm_vec8(
    const bf16* __restrict__ x,
    const bf16* __restrict__ weight,
    bf16* __restrict__ y,
    // Optional fp16 copy of the same result. The MXFP4 decode GEMV reads fp16,
    // and the only thing between it and this store was a kernel that read the
    // bf16 back and wrote it again -- a few tens of KB, so essentially all
    // launch. Compile-time flag, so the bf16-only instantiation is unchanged.
    f16* __restrict__ y_fp16,
    int hidden,
    int x_row_stride,
    int y_row_stride,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nvec = hidden / 8;

    const float4* xr =
        reinterpret_cast<const float4*>(x + static_cast<long long>(row) * x_row_stride);
    float4* yr =
        reinterpret_cast<float4*>(y + static_cast<long long>(row) * y_row_stride);
    const float4* wr = reinterpret_cast<const float4*>(weight);

    float local = 0.f;
    for (int i = tid; i < nvec; i += BLOCK) {
        float4 v = xr[i];
        const bf16x2* h = reinterpret_cast<const bf16x2*>(&v);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = bf16x2_to_f32(h[j]);
            local += f.x * f.x + f.y * f.y;
        }
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_exact<BLOCK>(local, buf);
    const float inv_rms = rsqrtf(buf_sum / static_cast<float>(hidden) + eps);

    for (int i = tid; i < nvec; i += BLOCK) {
        float4 v = xr[i];
        float4 g = wr[i];
        float4 o;
        const bf16x2* hv = reinterpret_cast<const bf16x2*>(&v);
        const bf16x2* hg = reinterpret_cast<const bf16x2*>(&g);
        bf16x2* ho = reinterpret_cast<bf16x2*>(&o);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 a = bf16x2_to_f32(hv[j]);
            float2 b = bf16x2_to_f32(hg[j]);
            if constexpr (WEIGHT_PLUS_ONE) { b.x += 1.f; b.y += 1.f; }
            ho[j] = f32_to_bf16x2(a.x * inv_rms * b.x,
                                  a.y * inv_rms * b.y);
        }
        yr[i] = o;
        if constexpr (EMIT_FP16) {
            // Rounded from the bf16 that was just stored, not from the fp32
            // behind it, so this is exactly what the cast kernel produced.
            const bf16* ob = reinterpret_cast<const bf16*>(&o);
            #pragma unroll
            for (int j = 0; j < 8; ++j) {
                y_fp16[i * 8 + j] = f32_to_f16(bf16_to_f32(ob[j]));
            }
        }
    }
}

/// float4 form of the fused residual-add + pre-norm. Deliberately a copy of
/// `rmsnorm_vec8`'s structure -- same VBLOCK, same per-vector accumulation
/// order, same `block_reduce_sum_exact` -- with the residual add folded into
/// the first pass, rounded to bf16 exactly where `residual_add` rounds it.
/// Anything else and the sum associates differently and the pair stops being
/// bit-exact.
template <int BLOCK>
__global__ void residual_add_rmsnorm_vec8(
    bf16* __restrict__ hidden,
    const bf16* __restrict__ residual,
    const bf16* __restrict__ weight,
    bf16* __restrict__ norm_out,
    int hidden_size,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int nvec = hidden_size / 8;
    const long long base = static_cast<long long>(row) * hidden_size;

    float4* hr = reinterpret_cast<float4*>(hidden + base);
    const float4* rr = reinterpret_cast<const float4*>(residual + base);
    float4* nr = reinterpret_cast<float4*>(norm_out + base);
    const float4* wr = reinterpret_cast<const float4*>(weight);

    float local = 0.f;
    for (int i = tid; i < nvec; i += BLOCK) {
        float4 hv = hr[i];
        float4 rv = rr[i];
        bf16x2* hh = reinterpret_cast<bf16x2*>(&hv);
        const bf16x2* rh = reinterpret_cast<const bf16x2*>(&rv);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 a = bf16x2_to_f32(hh[j]);
            const float2 b = bf16x2_to_f32(rh[j]);
            hh[j] = f32_to_bf16x2(a.x + b.x, a.y + b.y);
            const float2 f = bf16x2_to_f32(hh[j]);
            local += f.x * f.x + f.y * f.y;
        }
        hr[i] = hv;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_exact<BLOCK>(local, buf);
    const float inv_rms =
        rsqrtf(buf_sum / static_cast<float>(hidden_size) + eps);

    for (int i = tid; i < nvec; i += BLOCK) {
        float4 v = hr[i];
        float4 g = wr[i];
        float4 o;
        const bf16x2* hv = reinterpret_cast<const bf16x2*>(&v);
        const bf16x2* hg = reinterpret_cast<const bf16x2*>(&g);
        bf16x2* ho = reinterpret_cast<bf16x2*>(&o);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 a = bf16x2_to_f32(hv[j]);
            const float2 b = bf16x2_to_f32(hg[j]);
            ho[j] = f32_to_bf16x2(a.x * inv_rms * b.x,
                                  a.y * inv_rms * b.y);
        }
        nr[i] = o;
    }
}

/// Residual add + the next block's pre-norm, fused, and the row's kernel.
///
/// Numerically the two-kernel sequence: the add rounds to `T` exactly where
/// `elementwise.cuh`'s `residual_add` rounds it, and only then is the sum
/// squared. That is what makes the fusion a BINDING a declaration may state
/// rather than a different computation.
template <class T, int BLOCK = 256>
__global__ void residual_add_rmsnorm(
    T* __restrict__ hidden,
    const T* __restrict__ residual,
    const T* __restrict__ weight,
    T* __restrict__ norm_out,
    int hidden_size,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    T* hr = hidden + static_cast<long long>(row) * hidden_size;
    const T* rr = residual + static_cast<long long>(row) * hidden_size;
    T* nr = norm_out + static_cast<long long>(row) * hidden_size;

    float local = 0.f;
    for (int i = tid; i < hidden_size; i += BLOCK) {
        const float sum = Elem<T>::to_f32(hr[i]) + Elem<T>::to_f32(rr[i]);
        const T rounded = Elem<T>::from_f32(sum);
        hr[i] = rounded;
        const float v = Elem<T>::to_f32(rounded);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_exact<BLOCK>(local, buf);

    const float inv_rms =
        rsqrtf(buf_sum / static_cast<float>(hidden_size) + eps);

    for (int i = tid; i < hidden_size; i += BLOCK) {
        const float xv = Elem<T>::to_f32(hr[i]);
        const float wv = Elem<T>::to_f32(weight[i]);
        nr[i] = Elem<T>::from_f32(xv * inv_rms * wv);
    }
}

/// gemma-4's end-of-layer shape: the scale sits BETWEEN the add and the norm,
/// which is why this is not `residual_add_rmsnorm` with a multiply somewhere.
///
/// The scale is rounded to `T` before it is used, so the product is
/// `T(sum) · T(scale)` evaluated in fp32 -- the same rule
/// `elementwise.cuh`'s `scalar_mul` documents, for the same reason.
template <class T, int BLOCK = 256>
__global__ void residual_add_scale_rmsnorm(
    T* __restrict__ hidden,
    const T* __restrict__ residual,
    float scale,
    const T* __restrict__ weight,
    T* __restrict__ norm_out,
    int hidden_size,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    T* hr = hidden + static_cast<long long>(row) * hidden_size;
    const T* rr = residual + static_cast<long long>(row) * hidden_size;
    T* nr = norm_out + static_cast<long long>(row) * hidden_size;
    const float scale_rounded = Elem<T>::to_f32(Elem<T>::from_f32(scale));

    float local = 0.f;
    for (int i = tid; i < hidden_size; i += BLOCK) {
        const float sum = Elem<T>::to_f32(hr[i]) + Elem<T>::to_f32(rr[i]);
        const T rounded_sum = Elem<T>::from_f32(sum);
        const T scaled =
            Elem<T>::from_f32(Elem<T>::to_f32(rounded_sum) * scale_rounded);
        hr[i] = scaled;
        const float v = Elem<T>::to_f32(scaled);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_exact<BLOCK>(local, buf);

    const float inv_rms =
        rsqrtf(buf_sum / static_cast<float>(hidden_size) + eps);

    for (int i = tid; i < hidden_size; i += BLOCK) {
        const float xv = Elem<T>::to_f32(hr[i]);
        const float wv = Elem<T>::to_f32(weight[i]);
        nr[i] = Elem<T>::from_f32(xv * inv_rms * wv);
    }
}

/// Norm, then land it on the residual stream: `hidden += w · x · inv_rms`.
///
/// The normed value is rounded to `T` BEFORE the add, which is what makes
/// this the fusion of a norm and a `residual_add` rather than a fused
/// multiply-add with one rounding.
template <class T, int BLOCK = 256>
__global__ void rmsnorm_residual_add(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T* __restrict__ hidden,
    int hidden_size,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const T* xr = x + static_cast<long long>(row) * hidden_size;
    T* hr = hidden + static_cast<long long>(row) * hidden_size;

    float local = 0.f;
    for (int i = tid; i < hidden_size; i += BLOCK) {
        const float v = Elem<T>::to_f32(xr[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_exact<BLOCK>(local, buf);

    const float inv_rms =
        rsqrtf(buf_sum / static_cast<float>(hidden_size) + eps);
    for (int i = tid; i < hidden_size; i += BLOCK) {
        const T norm = Elem<T>::from_f32(
            Elem<T>::to_f32(xr[i]) * inv_rms * Elem<T>::to_f32(weight[i]));
        hr[i] = Elem<T>::from_f32(
            Elem<T>::to_f32(hr[i]) + Elem<T>::to_f32(norm));
    }
}

/// Vectorized twin of `rmsnorm_residual_add_scale_rmsnorm`: float4 loads (8
/// bf16) instead of scalars. The scalar form walks the row THREE times, once
/// per pass, so at hidden 2816 with BLOCK=512 each thread does ~6 dependent
/// loads per pass; vectorized it is under one. The plain norm's own note
/// records ~7x from exactly this change, and the scalar form measured 10.79
/// us/call in gemma-4-26B's decode -- 8% of the step -- against 2.51 for the
/// vectorized plain norm.
///
/// There is also a CuTile ALTERNATIVE to the scalar form beside this file,
/// `rmsnorm_rasr_tile.cuh`, measured at 2.41 us against 4.33 at one row and
/// 12.86 against 24.71 at 2,048 -- 1.8x and 1.9x. It reads each operand once
/// into a tile and does all three passes in registers, which is the direct
/// fix for the dependent-load chain this note is about. It is bit-identical
/// at 1 and 128 rows and reassociates the SECOND sum only, which is the same
/// trade the paragraph below describes. Neither this kernel nor the scalar
/// form goes anywhere: the alternative needs a toolchain this crate does not
/// load.
///
/// Per-element arithmetic and the bf16 rounding points are identical to the
/// scalar form; only the ORDER of the two sum reductions differs (8 values
/// accumulate per thread before the block reduce), which is the same trade
/// `rmsnorm_vec8` already makes, and at hidden 2048/2816/5376 it measured
/// BIT-identical. Requires `hidden % 8 == 0` and 16-byte-aligned rows -- the
/// launcher checks, which is why no row names this kernel.
template <int BLOCK>
__global__ void rmsnorm_rasr_vec8(
    const bf16* __restrict__ x,
    const bf16* __restrict__ weight,
    bf16* __restrict__ hidden,
    float scale,
    const bf16* __restrict__ next_weight,
    bf16* __restrict__ norm_out,
    int hidden_size,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const int vecs = hidden_size / 8;
    const float4* xr = reinterpret_cast<const float4*>(x + (long long)row * hidden_size);
    const float4* wv = reinterpret_cast<const float4*>(weight);
    const float4* nwv = reinterpret_cast<const float4*>(next_weight);
    float4* hr = reinterpret_cast<float4*>(hidden + (long long)row * hidden_size);
    float4* nr = reinterpret_cast<float4*>(norm_out + (long long)row * hidden_size);

    float local = 0.f;
    for (int i = tid; i < vecs; i += BLOCK) {
        const float4 v = xr[i];
        const bf16* b = reinterpret_cast<const bf16*>(&v);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const float f = bf16_to_f32(b[j]);
            local += f * f;
        }
    }
    __shared__ float buf[BLOCK];
    const float s0 = block_reduce_sum_exact<BLOCK>(local, buf);
    const float inv_rms = rsqrtf(s0 / static_cast<float>(hidden_size) + eps);
    const float scale_rounded = bf16_to_f32(f32_to_bf16(scale));

    float local_next = 0.f;
    for (int i = tid; i < vecs; i += BLOCK) {
        const float4 xv4 = xr[i];
        const float4 wv4 = wv[i];
        float4 hv4 = hr[i];
        const bf16* xb = reinterpret_cast<const bf16*>(&xv4);
        const bf16* wb = reinterpret_cast<const bf16*>(&wv4);
        bf16* hb = reinterpret_cast<bf16*>(&hv4);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            const bf16 norm = f32_to_bf16(
                bf16_to_f32(xb[j]) * inv_rms * bf16_to_f32(wb[j]));
            const float sum = bf16_to_f32(hb[j]) + bf16_to_f32(norm);
            const bf16 rounded = f32_to_bf16(sum);
            const bf16 scaled = f32_to_bf16(bf16_to_f32(rounded) * scale_rounded);
            hb[j] = scaled;
            const float f = bf16_to_f32(scaled);
            local_next += f * f;
        }
        hr[i] = hv4;
    }
    __shared__ float buf2[BLOCK];
    const float s1 = block_reduce_sum_exact<BLOCK>(local_next, buf2);
    const float inv_next = rsqrtf(s1 / static_cast<float>(hidden_size) + eps);

    for (int i = tid; i < vecs; i += BLOCK) {
        const float4 hv4 = hr[i];
        const float4 nw4 = nwv[i];
        const bf16* hb = reinterpret_cast<const bf16*>(&hv4);
        const bf16* nb = reinterpret_cast<const bf16*>(&nw4);
        float4 out4;
        bf16* ob = reinterpret_cast<bf16*>(&out4);
        #pragma unroll
        for (int j = 0; j < 8; ++j) {
            ob[j] = f32_to_bf16(bf16_to_f32(hb[j]) * inv_next * bf16_to_f32(nb[j]));
        }
        nr[i] = out4;
    }
}

/// Four statements in one launch, and two: gemma-4 fuses the next block's
/// input norm into the previous block's landing, which is why its layer body
/// appears to be missing one.
///
/// `BLOCK` has NO default. The scalar launcher runs this 512 threads wide and
/// `LaunchRule::Rms` fixes 256, so there is no width at which a row and this
/// kernel agree -- the missing default is that disagreement, spelled where a
/// future row author will trip over it.
template <class T, int BLOCK>
__global__ void rmsnorm_residual_add_scale_rmsnorm(
    const T* __restrict__ x,
    const T* __restrict__ weight,
    T* __restrict__ hidden,
    float scale,
    const T* __restrict__ next_weight,
    T* __restrict__ norm_out,
    int hidden_size,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;
    const T* xr = x + static_cast<long long>(row) * hidden_size;
    T* hr = hidden + static_cast<long long>(row) * hidden_size;
    T* nr = norm_out + static_cast<long long>(row) * hidden_size;

    float local = 0.f;
    for (int i = tid; i < hidden_size; i += BLOCK) {
        const float v = Elem<T>::to_f32(xr[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_exact<BLOCK>(local, buf);

    const float inv_rms =
        rsqrtf(buf_sum / static_cast<float>(hidden_size) + eps);
    const float scale_rounded = Elem<T>::to_f32(Elem<T>::from_f32(scale));
    float local_next = 0.f;
    for (int i = tid; i < hidden_size; i += BLOCK) {
        const T norm = Elem<T>::from_f32(
            Elem<T>::to_f32(xr[i]) * inv_rms * Elem<T>::to_f32(weight[i]));
        const float sum = Elem<T>::to_f32(hr[i]) + Elem<T>::to_f32(norm);
        const T rounded_sum = Elem<T>::from_f32(sum);
        const T scaled =
            Elem<T>::from_f32(Elem<T>::to_f32(rounded_sum) * scale_rounded);
        hr[i] = scaled;
        const float v = Elem<T>::to_f32(scaled);
        local_next += v * v;
    }

    __shared__ float buf_next[BLOCK];
    const float buf_next_sum = block_reduce_sum_exact<BLOCK>(local_next, buf_next);

    const float inv_next =
        rsqrtf(buf_next_sum / static_cast<float>(hidden_size) + eps);
    for (int i = tid; i < hidden_size; i += BLOCK) {
        nr[i] = Elem<T>::from_f32(
            Elem<T>::to_f32(hr[i]) * inv_next * Elem<T>::to_f32(next_weight[i]));
    }
}

/// No-weight variant -- the V-norm. Mirrors `rmsnorm` but skips the gamma
/// multiplication entirely: `y = x · rsqrt(var + eps)`.
template <class T, int BLOCK = 256>
__global__ void rmsnorm_no_scale(
    const T* __restrict__ x,
    T* __restrict__ y,
    int hidden,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const T* xr = x + static_cast<long long>(row) * hidden;
    T* yr = y + static_cast<long long>(row) * hidden;

    float local = 0.f;
    for (int i = tid; i < hidden; i += BLOCK) {
        const float v = Elem<T>::to_f32(xr[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_exact<BLOCK>(local, buf);
    const float inv_rms = rsqrtf(buf_sum / static_cast<float>(hidden) + eps);

    for (int i = tid; i < hidden; i += BLOCK) {
        yr[i] = Elem<T>::from_f32(Elem<T>::to_f32(xr[i]) * inv_rms);
    }
}

/// `y = weight · (x · rsqrt(mean(x²) + eps)) · silu(gate)`. One block per row.
///
/// `weight` is fp32 in every instantiation -- qwen3.5 ships RMSNormGated
/// weights in fp32 alongside bf16 activations.
template <class T, int BLOCK = 256>
__global__ void rmsnorm_gated(
    const T* __restrict__ x,
    const T* __restrict__ gate,
    const float* __restrict__ weight,
    T* __restrict__ y,
    int hidden,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const T* xr = x + static_cast<long long>(row) * hidden;
    const T* gr = gate + static_cast<long long>(row) * hidden;
    T* yr = y + static_cast<long long>(row) * hidden;

    float local = 0.f;
    for (int i = tid; i < hidden; i += BLOCK) {
        const float v = Elem<T>::to_f32(xr[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_exact<BLOCK>(local, buf);
    const float inv_rms = rsqrtf(buf_sum / static_cast<float>(hidden) + eps);

    for (int i = tid; i < hidden; i += BLOCK) {
        const float xv = Elem<T>::to_f32(xr[i]) * inv_rms;
        const float wv = weight[i];
        const float gv = Elem<T>::to_f32(gr[i]);
        // silu(z) = z / (1 + exp(-z)) = z * sigmoid(z).
        const float sg = gv / (1.f + __expf(-gv));
        yr[i] = Elem<T>::from_f32(wv * xv * sg);
    }
}

/// Same as `rmsnorm_gated` but `x` is fp32 -- it came straight from the GDN
/// recurrent step, which outputs fp32.
///
/// Fuses the separate fp32->bf16 conversion that qwen3.5's forward used to
/// emit as its own kernel launch before calling the gated norm. Per-row HBM
/// traffic dropped from 12·hidden bytes (4-byte x read + 2-byte x write +
/// 4-byte x read + 2-byte gate read + 2-byte y write) to 8·hidden (4-byte x
/// read + 2-byte gate read + 2-byte y write) -- one full pass over the
/// intermediate buffer, gone.
template <class T, int BLOCK = 256>
__global__ void rmsnorm_gated_f32_in(
    const float* __restrict__ x,
    const T* __restrict__ gate,
    const float* __restrict__ weight,
    T* __restrict__ y,
    int hidden,
    float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const float* xr = x + static_cast<long long>(row) * hidden;
    const T* gr = gate + static_cast<long long>(row) * hidden;
    T* yr = y + static_cast<long long>(row) * hidden;

    float local = 0.f;
    for (int i = tid; i < hidden; i += BLOCK) {
        const float v = xr[i];
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    const float buf_sum = block_reduce_sum_exact<BLOCK>(local, buf);
    const float inv_rms = rsqrtf(buf_sum / static_cast<float>(hidden) + eps);

    for (int i = tid; i < hidden; i += BLOCK) {
        const float xv = xr[i] * inv_rms;
        const float wv = weight[i];
        const float gv = Elem<T>::to_f32(gr[i]);
        const float sg = gv / (1.f + __expf(-gv));
        yr[i] = Elem<T>::from_f32(wv * xv * sg);
    }
}

}  // namespace pie::norm
