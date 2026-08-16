//===-- dsv4_hc.cuh - DeepSeek-V4's hyper-connection kernels -------------===//
//
// Seven `__global__` templates and nothing else: no host function, no
// `<<<>>>`, no entry point. Three of them are named by rows in
// `kernels_cuda::families::norm`; four are not, and this header says
// which and why, because a kernel nobody can fire is worth less than a kernel
// nobody can fire *for a reason on record*.
//
// # Hyper-connections, in one paragraph
//
// A hyper-connected layer keeps `M` residual streams instead of one. Before
// the layer, `hc_pre_postprocess` reads a `[N, 2M + M²]` mix matrix, splits it
// into a pre-gate, a post-gate and a combination matrix, Sinkhorn-normalises
// the last of those into a doubly-stochastic mixer, and collapses the `M`
// streams into the single `[N, H]` input the layer actually runs on. After the
// layer, `hc_post` scatters the layer's output back across the `M` streams
// through that mixer. `hc_head_postprocess` is the same collapse without the
// Sinkhorn -- a plain gated sum -- and `hc_expand` is the degenerate mixer
// that broadcasts one stream into `M`.
//
// # What the launchers were doing, and where it went
//
// `hc_pre_postprocess`, `hc_head_postprocess` and `hc_rmsnorm_to_f32` all
// launched `<<<N, 256>>>`: one block per token, the block striding the row,
// with a shared-memory reduction or a shared gate vector across it. That is
// `LaunchRule::Rms` exactly -- *"one block per row, 256 threads, a warp's
// worth of scratch"* -- so `N` left the kernel signatures with the rule that
// recovers it, the `N <= 0` guard became `Ungeometric::Empty`, and the rows
// say what the launchers said.
//
// # The two that no rule states
//
// * **`hc_post` and `hc_expand`** launched `ceil(N·H / 256)` blocks, where `H`
//   is the width of the INPUT and their output is `[N, M, H]`. Every flat
//   rule THEN PORTED sized its grid on the OUTPUT rectangle, so `Elementwise`
//   would launch `M` times too many blocks and lean on the `idx >= N·H` guard
//   to throw them away -- correct, but `M`-fold wasteful, and a rule that has
//   to be wrong to be right is not the rule. **That sentence is what
//   `LaunchRule::ElementwiseIn` was built to answer**, and
//   `runtime::launch::elementwise_in` quotes these three lines as the
//   launcher it reproduces. Both kernels are rowed; the four below are two.
//
// * **`attn_sink_correction` and `per_head_rmsnorm`** launch `dim3(N, heads)`
//   -- a block per (token, head). `per_head_rmsnorm` reads `gridDim.y` for its
//   head count, so it does not merely tolerate that grid, it depends on it.
//   This is `LaunchRule::PerHead`, which `runtime::launch` has not ported; no
//   ported rule produces a `gridDim.y` at all.
//
// Both keep their ahead-of-time launchers. When the missing rule lands, the
// diff is a row -- which is what `hc_post` and `hc_expand` just demonstrated.
//
// # `MAX_HC_MULT` is a precondition no row can state
//
// `hc_post` keeps its `M` residual values in registers -- `float r[MAX_HC_MULT]`
// -- because it runs IN PLACE (`residual == out`) and a thread must read every
// value it owns before it writes any of them. That fixes a compile-time bound
// on `M`, and the launcher refused `hc_mult > MAX_HC_MULT` rather than
// silently indexing off the end of an array. No `LaunchRule` carries a
// precondition on an operand's VALUE and no `Source` computes one, so when
// `hc_post` was rowed the check moved INTO THE KERNEL -- the first statement
// of the body. The launcher's host-side return stays where it is, so the
// archive's behaviour is unchanged to the instruction; the guard is only
// reachable from a fire, which has no host in front of it.
//
// # Why they are templates when the originals were not
//
// The originals were `_bf16` and only `_bf16` because an AOT build has to
// choose its instantiations. The bodies are written over `T` through
// `Elem<T>`, so a second numeric format costs a row instead of a
// translation unit. The mixes, gates and Sinkhorn iterations stay fp32
// throughout: they are a small dense matrix whose row and column sums are
// driven to one, and rounding them to `T` would leave the mixer measurably
// un-stochastic.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "prelude/device.cuh"

namespace pie::norm {


/// The largest `M` (`hc_mult`) `hc_post` can hold in registers, and therefore
/// the largest one its launcher will accept. Also the shared-array bound in
/// `hc_pre_postprocess` and `hc_head_postprocess`, which size `[M]` and `[M²]`
/// scratch at compile time.
constexpr int MAX_HC_MULT = 8;

/// Split the mix matrix, Sinkhorn-normalise the combiner, and collapse `M`
/// residual streams into the layer's `[N, H]` input.
///
/// One block per token. `pre`, `post` and `comb` live in shared memory for
/// the whole kernel: the Sinkhorn iteration alternates row and column
/// normalisation, and a column pass needs every row the previous pass wrote.
///
/// The first `M` (or `M²`) threads do the mixing and the whole block strides
/// `H` for the collapse, so a block is useful at both widths.
template <class T, int BLOCK = 256>
__global__ void hc_pre_postprocess(
    const float* __restrict__ mixes,      // [N, 2M + M*M]
    const float* __restrict__ scale,      // [3]
    const float* __restrict__ base,       // [2M + M*M]
    const T* __restrict__ residual,       // [N, M, H]
    float* __restrict__ post_mix,         // [N, M]
    float* __restrict__ comb_mix,         // [N, M, M]
    T* __restrict__ layer_input,          // [N, H]
    int M,
    int H,
    float hc_eps,
    float hc_post_alpha,
    int sinkhorn_iters)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;

    const int mix_hc = M * 2 + M * M;
    const float* row = mixes + static_cast<long long>(n) * mix_hc;

    __shared__ float pre[MAX_HC_MULT];
    __shared__ float post[MAX_HC_MULT];
    __shared__ float comb[MAX_HC_MULT * MAX_HC_MULT];

    if (tid < M) {
        // Pre-mix: sigmoid + eps
        const float logit = row[tid] * scale[0] + base[tid];
        pre[tid] = 1.f / (1.f + expf(-logit)) + hc_eps;
    }
    if (tid < M) {
        // Post-mix: sigmoid * alpha
        const float logit = row[M + tid] * scale[1] + base[M + tid];
        post[tid] = 1.f / (1.f + expf(-logit)) * hc_post_alpha;
        post_mix[static_cast<long long>(n) * M + tid] = post[tid];
    }
    __syncthreads();

    // Comb-mix: softmax + sinkhorn
    if (tid < M * M) {
        const float logit = row[2 * M + tid] * scale[2] + base[2 * M + tid];
        comb[tid] = logit;
    }
    __syncthreads();

    // Softmax per row + eps  (reference: comb = comb.softmax(-1) + eps)
    if (tid < M) {
        float max_v = -flt_max();
        for (int j = 0; j < M; ++j)
            max_v = fmaxf(max_v, comb[tid * M + j]);
        float sum = 0.f;
        for (int j = 0; j < M; ++j) {
            comb[tid * M + j] = expf(comb[tid * M + j] - max_v);
            sum += comb[tid * M + j];
        }
        for (int j = 0; j < M; ++j)
            comb[tid * M + j] = comb[tid * M + j] / sum + hc_eps;
    }
    __syncthreads();

    // Initial col normalization (reference: comb = comb / (comb.sum(-2) + eps))
    if (tid < M) {
        float col_sum = 0.f;
        for (int i = 0; i < M; ++i) col_sum += comb[i * M + tid];
        col_sum += hc_eps;
        for (int i = 0; i < M; ++i)
            comb[i * M + tid] = comb[i * M + tid] / col_sum;
    }
    __syncthreads();

    // Sinkhorn iterations: (row, col) pairs
    for (int iter = 0; iter < sinkhorn_iters - 1; ++iter) {
        // Normalize rows
        if (tid < M) {
            float row_sum = 0.f;
            for (int j = 0; j < M; ++j) row_sum += comb[tid * M + j];
            row_sum += hc_eps;
            for (int j = 0; j < M; ++j)
                comb[tid * M + j] = comb[tid * M + j] / row_sum;
        }
        __syncthreads();
        // Normalize columns
        if (tid < M) {
            float col_sum = 0.f;
            for (int i = 0; i < M; ++i) col_sum += comb[i * M + tid];
            col_sum += hc_eps;
            for (int i = 0; i < M; ++i)
                comb[i * M + tid] = comb[i * M + tid] / col_sum;
        }
        __syncthreads();
    }

    // Write comb_mix
    if (tid < M * M) {
        comb_mix[static_cast<long long>(n) * M * M + tid] = comb[tid];
    }
    __syncthreads();

    // Compute layer_input = sum_i(pre_i * residual[n, i, :])
    const T* res_n = residual + static_cast<long long>(n) * M * H;
    T* out = layer_input + static_cast<long long>(n) * H;

    for (int h = tid; h < H; h += blockDim.x) {
        float acc = 0.f;
        for (int i = 0; i < M; ++i) {
            acc += pre[i] * Elem<T>::to_f32(res_n[i * H + h]);
        }
        out[h] = Elem<T>::from_f32(acc);
    }
}

/// Scatter the layer's `[N, H]` output back across the `M` residual streams
/// through the Sinkhorn mixer.
///
/// One thread owns a whole `(n, h)` column across all `M` streams. The caller
/// runs this in place (`residual == out`), so a thread must load every
/// residual value it needs before writing any of them; splitting the `M`
/// outputs across threads would let one block overwrite a value another block
/// has not read yet. That is why `r` is a register array and why `M` is
/// bounded by `MAX_HC_MULT`.
///
/// **No row names this kernel**: its grid covers `N·H`, the width of the
/// input, and every ported rule sizes on the output rectangle `[N, M·H]`.
/// -- was true until `LaunchRule::ElementwiseIn` landed, ported FROM this
/// launcher. `norm::hc_post_bf16` is a row now.
///
/// The `M > MAX_HC_MULT` refusal moved here from the launcher, and it had to.
/// `r` is a register array of `MAX_HC_MULT` floats -- the kernel runs in
/// place, so a thread must read every value it owns before writing any of
/// them -- and the ahead-of-time launcher answered a too-wide `M` with a
/// host-side early return. A fire is a `void**` and a grid with no host in
/// front of it, no `LaunchRule` carries a precondition on an operand's VALUE,
/// and no `Source` computes one; the check therefore belongs where both paths
/// reach it. It changes nothing the archive does: `dsv4_hc.cu` still returns
/// before the launch, so this guard is only ever reached by a fire that would
/// otherwise have walked off the end of `r`.
template <class T>
__global__ void hc_post(
    const T* __restrict__ x,             // [N, H]
    const T* residual,                    // [N, M, H]
    const float* __restrict__ post_mix,   // [N, M]
    const float* __restrict__ comb_mix,   // [N, M, M]
    T* out,                               // [N, M, H], may alias residual
    int N,
    int M,
    int H)
{
    if (M > MAX_HC_MULT) return;
    const long long idx =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<long long>(N) * H) return;

    const int h = static_cast<int>(idx % H);
    const int n = static_cast<int>(idx / H);

    const float* comb_n = comb_mix + static_cast<long long>(n) * M * M;
    const float* post_n = post_mix + static_cast<long long>(n) * M;
    const float x_h = Elem<T>::to_f32(x[static_cast<long long>(n) * H + h]);
    const T* res_n = residual + static_cast<long long>(n) * M * H;

    float r[MAX_HC_MULT];
    for (int i = 0; i < M; ++i) {
        r[i] = Elem<T>::to_f32(res_n[static_cast<long long>(i) * H + h]);
    }

    T* out_n = out + static_cast<long long>(n) * M * H;
    // Reference: y[c=j, d=h] = post[c]*x[d] + sum_r comb[r, c] * residual[r, d]
    // comb is stored as [row, col] with row-major layout: comb[r*M + c].
    for (int j = 0; j < M; ++j) {
        float acc = post_n[j] * x_h;
        for (int i = 0; i < M; ++i) {
            acc += comb_n[i * M + j] * r[i];
        }
        out_n[static_cast<long long>(j) * H + h] = Elem<T>::from_f32(acc);
    }
}

/// The head variant of the collapse: a gated sum with NO normalisation.
///
/// `pre = sigmoid(mixes · scale + base) + eps`, then `out = Σ_i pre_i · residual_i`.
/// The Sinkhorn pass is deliberately absent -- the head mixes are not a
/// transport plan, they are `M` independent gates.
template <class T, int BLOCK = 256>
__global__ void hc_head_postprocess(
    const float* __restrict__ mixes,     // [N, M] after GEMM
    const float* __restrict__ scale,     // [1]
    const float* __restrict__ base,      // [M]
    const T* __restrict__ residual,      // [N, M, H]
    T* __restrict__ out,                 // [N, H]
    int M,
    int H,
    float hc_eps)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;

    __shared__ float gates[MAX_HC_MULT];

    if (tid < M) {
        const float logit = mixes[static_cast<long long>(n) * M + tid] * scale[0] + base[tid];
        gates[tid] = 1.f / (1.f + expf(-logit)) + hc_eps;
    }
    __syncthreads();

    const T* res_n = residual + static_cast<long long>(n) * M * H;
    T* out_n = out + static_cast<long long>(n) * H;

    for (int h = tid; h < H; h += blockDim.x) {
        float acc = 0.f;
        for (int i = 0; i < M; ++i) {
            acc += gates[i] * Elem<T>::to_f32(res_n[i * H + h]);
        }
        out_n[h] = Elem<T>::from_f32(acc);
    }
}

/// Broadcast one `[N, H]` stream into `M` of them -- the mixer a layer starts
/// from before any hyper-connection has been learned.
///
/// **No row names this kernel**, for the same reason as `hc_post`: the grid
/// covers the input's `N·H`, not the output's `N·M·H`. -- and for the same
/// reason it has one now. `LaunchRule::ElementwiseIn` sizes on exactly that
/// `N·H`, `norm::hc_expand_bf16` states it, and the launcher this header
/// quoted as unrepresentable is the launcher the rule was ported from.
template <class T>
__global__ void hc_expand(
    const T* __restrict__ input,   // [N, H]
    T* __restrict__ output,        // [N, M, H]
    int N,
    int M,
    int H)
{
    const long long idx =
        static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= static_cast<long long>(N) * H) return;
    const int n = static_cast<int>(idx / H);
    const int h = static_cast<int>(idx % H);

    const T val = input[static_cast<long long>(n) * H + h];
    for (int m = 0; m < M; ++m) {
        output[static_cast<long long>(n) * M * H + m * H + h] = val;
    }
}

/// RMS-normalise a `[N, dim]` rectangle and widen the result to fp32.
///
/// The fp32 output is the point: the mix GEMM that consumes it runs in fp32,
/// and the hyper-connection coefficients derived from it are sensitive enough
/// that a bf16 round-trip through this buffer moves tokens.
///
/// The reduction is a fixed-order tree. An `atomicAdd` across warps would make
/// the sum depend on warp scheduling, and the mixes derived from it amplify
/// that into run-to-run token differences. `BLOCK` is a template parameter
/// only so `warp_sums` can be sized at compile time; `LaunchRule::Rms` fixes
/// the launch at 256, and every instantiation a row names takes the default.
template <class T, int BLOCK = 256>
__global__ void hc_rmsnorm_to_f32(
    const T* __restrict__ input,   // [N, dim]
    float* __restrict__ output,    // [N, dim]
    int dim,
    float eps)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const T* row = input + static_cast<long long>(n) * dim;
    float* out = output + static_cast<long long>(n) * dim;

    float local_sum = 0.f;
    for (int d = tid; d < dim; d += blockDim.x) {
        float v = Elem<T>::to_f32(row[d]);
        local_sum += v * v;
    }
    for (int offset = 16; offset > 0; offset >>= 1)
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, offset);

    __shared__ float warp_sums[BLOCK / 32];
    if ((tid & 31) == 0) warp_sums[tid >> 5] = local_sum;
    __syncthreads();

    __shared__ float scale;
    if (tid == 0) {
        float total = 0.f;
        const int nwarps = (blockDim.x + 31) / 32;
        for (int w = 0; w < nwarps; ++w) total += warp_sums[w];
        scale = rsqrtf(total / dim + eps);
    }
    __syncthreads();

    const float s = scale;
    for (int d = tid; d < dim; d += blockDim.x) {
        out[d] = Elem<T>::to_f32(row[d]) * s;
    }
}

/// Fold an attention sink into an already-computed attention output.
///
/// A sink is a learned logit that competes with every key; scaling the output
/// by `1 / (1 + exp(sink - lse))` is the same as having softmaxed with it in
/// the denominator. Doing it after the fact is what lets FlashAttention stay
/// unmodified.
///
/// **No row names this kernel**: the launcher's grid is `dim3(N, heads)`, and
/// no ported rule produces a `gridDim.y`.
template <class T>
__global__ void attn_sink_correction(
    T* __restrict__ out,
    const float* __restrict__ lse,
    const float* __restrict__ sink,
    int num_heads,
    int head_dim)
{
    const int n = blockIdx.x;
    const int h = blockIdx.y;
    const float s = 1.0f / (1.0f + expf(sink[h] - lse[n * num_heads + h]));
    T* row = out + (static_cast<long long>(n) * num_heads + h) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        row[d] = Elem<T>::from_f32(Elem<T>::to_f32(row[d]) * s);
    }
}

/// RMS-normalise each attention head of a `[N, heads, head_dim]` tensor in
/// place, with no learned weight.
///
/// **No row names this kernel**, and it is the strongest case of the four:
/// it reads `gridDim.y` for its head count. A rule that launched one block per
/// row would not merely waste blocks, it would tell the kernel there is one
/// head. That is `LaunchRule::PerHead`, which `runtime::launch` has not
/// ported.
template <class T>
__global__ void per_head_rmsnorm(
    T* __restrict__ q,
    int head_dim,
    float eps)
{
    // grid: (N, num_heads). Each block handles one head.
    const int n = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int num_heads = gridDim.y;

    T* row = q + (static_cast<long long>(n) * num_heads + h) * head_dim;

    float local_sum = 0.f;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        const float v = Elem<T>::to_f32(row[d]);
        local_sum += v * v;
    }
    for (int off = 16; off > 0; off >>= 1)
        local_sum += __shfl_down_sync(0xFFFFFFFF, local_sum, off);

    __shared__ float scale;
    __shared__ float reduce_buf[32];
    if ((tid & 31) == 0) reduce_buf[tid >> 5] = local_sum;
    __syncthreads();
    if (tid < 32) {
        float v = (tid < (blockDim.x + 31) / 32) ? reduce_buf[tid] : 0.f;
        for (int off = 16; off > 0; off >>= 1)
            v += __shfl_down_sync(0xFFFFFFFF, v, off);
        if (tid == 0) scale = rsqrtf(v / static_cast<float>(head_dim) + eps);
    }
    __syncthreads();

    const float s = scale;
    for (int d = tid; d < head_dim; d += blockDim.x) {
        row[d] = Elem<T>::from_f32(Elem<T>::to_f32(row[d]) * s);
    }
}

}  // namespace pie::norm
