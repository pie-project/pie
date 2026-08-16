//===-- gated_delta_net_prep.cuh - GDN's pre-recurrence kernels ---------===//
//
// The seven `__global__` templates that run BEFORE Gated Delta Net's
// recurrence: two dtype casts, a GQA head broadcast, an L2 norm, the
// gate/beta activation, and Qwen3.5's fused post-conv prep pair. No host
// function, no `<<<>>>`. `gated_delta_net.cu` includes this file and keeps
// only its launchers, so each kernel is defined ONCE in the tree — the two
// definitions `norm/altup_aux` shipped for a release are why
// `tests/device_sources.rs` refuses a second one.
//
// # Why this is a second header rather than the whole file
//
// `gated_delta_net.cu` was 2523 lines and 21 `__global__`s across two
// anonymous namespaces with a hundred lines of launcher between them, and the
// two halves are not the same kind of code. This half is pointwise, on the
// prelude, and portable. The other half — the fourteen recurrence kernels in
// `gated_delta_net.cuh` — still reaches for `__nv_bfloat162` and
// `__floats2bfloat162_rn`, which is why §10.5 records the prelude conversion
// there as REVERTED rather than done.
//
// A single header would tie the two together: NVRTC compiles a unit whole, so
// one unresolved packed-half intrinsic in the recurrence would take these
// seven rows down with it. Splitting on the seam the file already had costs a
// second `#include` and buys the prep half a unit that stands on its own.
//
// # The rows, and the three refusals
//
// Six rows. Five are `LaunchRule::Elementwise` or `ElementwiseRows`:
//
//   • `widen` and `narrow` — `if (n == 0) return;` then `ceil(n / 256)` blocks
//     of 256 over a guarded map. That guard is now the RULE's: `eval` refuses
//     a zero extent as `Ungeometric::Empty` before a grid is computed, which
//     is where the launcher's early return went. Both get an f16 twin the
//     ahead-of-time build never had — not because it was hard, but because a
//     second instantiation cost a translation unit's worth of `cicc` for
//     something nobody had asked for. Under the JIT it costs a row.
//   • `g_beta` — `ElementwiseRows`. The launcher tiles `dim3(N, ceil(V_h/64))`
//     with 64-wide blocks; the rule tiles `dim3(N, ceil(V_h/256))` with
//     256-wide ones. Both cover exactly the channels `h < V_h` and the kernel
//     is a pure map behind `h >= V_h`, so the surplus threads return and the
//     answer is bit-identical. That is the coverage argument
//     `elementwise_rows_covers_every_channel` already measured for
//     `mean_streams`; it would NOT hold for a reduction, where the fold order
//     is part of the contract.
//
// The sixth is `repeat_interleave_heads_fp32`, and what unblocked it was the
// head axis arriving: `LaunchRule::GatedRms` is `[rows, kv_heads]` at 256,
// and `Dims::kv_heads` documents the GDN recurrence's value heads as the
// quantity it carries. What remained after that was arity — a plain
// `__global__` cannot be rowed whatever its geometry — so it is a template
// over `T` with exactly one instantiation, `f32`, which the launcher spells.
// A template with one instantiation is normally a rename; here it is the
// difference between a kernel a row can name and one it cannot.
//
// The other three state geometry no rule produces:
//
//   • `l2norm_scale` — `dim3(N)` by `BLOCK = 128`, which LOOKS like
//     `LaunchRule::Rms`. It is not: `Rms` launches 256 threads, and this
//     kernel's `__shared__ float buf[BLOCK]` is sized by the template
//     parameter. Firing it at 256 wide would have 128 threads write past the
//     array. Sizing the shared array by `blockDim.x` instead would change a
//     `__global__` body, which §8 says needs its own parity evidence — and
//     compiling is not measuring arithmetic.
//   • `qwen_gdn_qk_norm`, `qwen_gdn_v_g_beta` — grid `(rows, K_h)` and
//     `(rows, V_h)`, which `GatedRms` does produce; both are block-wide
//     reductions over `__shared__ float buf[BLOCK]` whose fold walks
//     `BLOCK / 2`, so the rule's 256 against the launchers' 64 and 128 is the
//     `l2norm_scale` overrun again, and the fold order is part of the answer
//     rather than an implementation detail a surplus lane can sit out.
//
// Those three are here because the split is per FILE, and unrowed because
// inventing a rule to fit them would put a guess where a refusal belongs.
//
// # Why they are templates when the launchers are bf16
//
// The ahead-of-time build had to choose its instantiations and chose bf16 —
// a fact about nvcc's compile time, not about the arithmetic, which widens to
// fp32, accumulates there and narrows once in every one of these. `T` and the
// prelude's `Elem<T>` say the same thing and let a row pick. The four unrowed
// kernels carry `T` alongside their existing `int BLOCK`, which no row could
// spell anyway — `DeviceKernel::instantiation()` emits exactly one type
// argument — so their templating is for the reader and the day a rule exists.
//
// `__bfloat162float` and `__float2bfloat16` became `Elem<T>::to_f32` and
// `Elem<T>::from_f32`, which the prelude documents as the same bits: the
// widening is an exponent-preserving zero-extend and the narrowing is
// round-to-nearest-even, both spelled to match the intrinsic.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "prelude/device.cuh"

namespace pie::ssm {

// The scalar layer is the PRELUDE's. Named here so a row may spell its
// element type `bf16`, and so the launchers in `gated_delta_net.cu` —
// which sit in `kernels::ssm`, where this namespace shadows `kernels::device`
// — still resolve `bf16` and `usize` to the prelude's names.

// `Elem<float>` does not exist and this file does not add it. The prelude's
// trait is the widen/narrow pair, and fp32 has nothing to widen — a
// specialisation would be an identity written to satisfy a name. Declaring it
// HERE would be worse than useless: a leaf header's specialisation of a
// prelude template is visible to every unit that includes the leaf and to no
// other, so the next leaf that wants one gets a redefinition in the units
// that see both and silently disagrees in the units that see one. The alias
// is local, four other leaves already carry it — `attn/attn_sink.cuh`,
// `moe/topk_softmax.cuh`, `quant/dtype_cast.cuh`, `sample/argmax.cuh` — and
// an identical typedef repeated across headers is legal C++ however many
// times a unit sees it.
using f32 = float;

template <class T>
__global__ void widen(
    const T* __restrict__ x, float* __restrict__ y, usize n)
{
    const usize i = blockIdx.x * (usize)blockDim.x + threadIdx.x;
    if (i < n) y[i] = Elem<T>::to_f32(x[i]);
}

template <class T>
__global__ void narrow(
    const float* __restrict__ x, T* __restrict__ y, usize n)
{
    const usize i = blockIdx.x * (usize)blockDim.x + threadIdx.x;
    if (i < n) y[i] = Elem<T>::from_f32(x[i]);
}

/// GQA's head broadcast for the recurrence's fp32 operands: `[N, K_h, D]`
/// read, `[N, V_h, D]` written, head `h_v` taking key head `h_v / repeat`.
///
/// `LaunchRule::GatedRms` is its grid — `[rows, kv_heads]`, and
/// [`Dims::kv_heads`] is the field whose own documentation names the GDN
/// recurrence's value heads as the axis it carries. The rule's block is 256
/// where this launcher picks 64 or 128 on `D`, and that is safe HERE for
/// reasons that are properties of this body and not of the rule: the kernel
/// is a pure copy behind `d >= D`, holds no shared memory, folds nothing and
/// calls no `__syncthreads`, so a surplus lane returns before it addresses
/// anything and the strided tail loop it skipped runs zero times. The same
/// substitution under `l2norm_scale` below would write past a
/// `__shared__ float[BLOCK]`.
///
/// Templated over `T` for one reason and one only: a row cannot name a plain
/// `__global__`, because `DeviceKernel::instantiation()` emits `path<Elem>`
/// and stops. The body is a copy, so no `Elem<T>` appears in it and no
/// arithmetic changed — `T = f32` is the only instantiation, the launcher
/// spells it, and the ahead-of-time build emits what it emitted before.
template <class T>
__global__ void repeat_interleave_heads_fp32(
    const T* __restrict__ in, T* __restrict__ out,
    int K_h, int V_h, int D, int repeat)
{
    const int n   = blockIdx.x;
    const int h_v = blockIdx.y;
    const int d   = threadIdx.x;
    if (h_v >= V_h || d >= D) return;
    const int h_k = h_v / repeat;
    const long long src = ((long long)n * K_h + h_k) * D + d;
    const long long dst = ((long long)n * V_h + h_v) * D + d;
    if (d < D) out[dst] = in[src];
    // Iterate if D > blockDim.x.
    for (int dd = d + blockDim.x; dd < D; dd += blockDim.x) {
        out[((long long)n * V_h + h_v) * D + dd] =
            in[((long long)n * K_h + h_k) * D + dd];
    }
}

template <class T, int BLOCK>
__global__ void l2norm_scale(
    const T* __restrict__ x,
    float*               __restrict__ y,
    int hidden, float scale, float eps)
{
    const int row = blockIdx.x;
    const int tid = threadIdx.x;

    const T* xr = x + (long long)row * hidden;
    float*               yr = y + (long long)row * hidden;

    float local = 0.f;
    for (int i = tid; i < hidden; i += BLOCK) {
        const float v = Elem<T>::to_f32(xr[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    buf[tid] = local;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) buf[tid] += buf[tid + off];
        __syncthreads();
    }
    const float inv = rsqrtf(buf[0] + eps);

    for (int i = tid; i < hidden; i += BLOCK) {
        yr[i] = Elem<T>::to_f32(xr[i]) * inv * scale;
    }
}

// g_log[t, h] = -exp(A_log[h]) * softplus(a[t, h] + dt_bias[h])
// beta[t, h]  = sigmoid(b[t, h])
//
// HF Qwen3.5 stores `A_log` and the RMSNormGated weight in fp32 (matches
// the FLA fast-path expectation), even when the rest of the model is
// bf16. dt_bias stays bf16.
template <class T>
__global__ void g_beta(
    const T* __restrict__ a,
    const T* __restrict__ b,
    const float* __restrict__ A_log,
    const T* __restrict__ dt_bias,
    float*               __restrict__ g_log_out,
    float*               __restrict__ beta_out,
    int N, int V_h)
{
    const int t = blockIdx.x;
    const int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (t >= N || h >= V_h) return;

    const float av  = Elem<T>::to_f32(a[(long long)t * V_h + h]);
    const float bv  = Elem<T>::to_f32(b[(long long)t * V_h + h]);
    const float Alh = A_log[h];
    const float dtb = Elem<T>::to_f32(dt_bias[h]);

    // softplus(z) = log1p(exp(z)). Numerically stable variant.
    const float z = av + dtb;
    const float sp = (z > 20.f) ? z : log1pf(__expf(z));

    g_log_out[(long long)t * V_h + h] = -__expf(Alh) * sp;
    beta_out[(long long)t * V_h + h]  = 1.f / (1.f + __expf(-bv));
}

template <class T, int BLOCK>
__global__ void qwen_gdn_qk_norm(
    const T* __restrict__ qkv_post,
    float* __restrict__ q_out,
    float* __restrict__ k_out,
    int K_h, int K_d, int conv_dim,
    float q_scale)
{
    const int n = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int K_dim = K_h * K_d;
    const T* q_base =
        qkv_post + (long long)n * conv_dim + (long long)h * K_d;
    const T* k_base =
        qkv_post + (long long)n * conv_dim + K_dim + (long long)h * K_d;

    float q_sum = 0.f;
    float k_sum = 0.f;
    for (int i = tid; i < K_d; i += BLOCK) {
        const float qv = Elem<T>::to_f32(q_base[i]);
        const float kv = Elem<T>::to_f32(k_base[i]);
        q_sum += qv * qv;
        k_sum += kv * kv;
    }

    __shared__ float q_buf[BLOCK];
    __shared__ float k_buf[BLOCK];
    q_buf[tid] = q_sum;
    k_buf[tid] = k_sum;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) {
            q_buf[tid] += q_buf[tid + off];
            k_buf[tid] += k_buf[tid + off];
        }
        __syncthreads();
    }

    const float q_inv = rsqrtf(q_buf[0] + 1e-6f) * q_scale;
    const float k_inv = rsqrtf(k_buf[0] + 1e-6f);
    float* q_dst = q_out + ((long long)n * K_h + h) * K_d;
    float* k_dst = k_out + ((long long)n * K_h + h) * K_d;
    for (int i = tid; i < K_d; i += BLOCK) {
        q_dst[i] = Elem<T>::to_f32(q_base[i]) * q_inv;
        k_dst[i] = Elem<T>::to_f32(k_base[i]) * k_inv;
    }
}

template <class T, int BLOCK>
__global__ void qwen_gdn_v_g_beta(
    const T* __restrict__ qkv_post,
    const T* __restrict__ a,
    const T* __restrict__ b,
    const float* __restrict__ A_log,
    const T* __restrict__ dt_bias,
    float* __restrict__ v_out,
    float* __restrict__ g_log_out,
    float* __restrict__ beta_out,
    int K_h, int V_h, int K_d, int V_d, int conv_dim)
{
    const int n = blockIdx.x;
    const int h = blockIdx.y;
    const int tid = threadIdx.x;
    const int K_dim = K_h * K_d;
    const T* v_base =
        qkv_post + (long long)n * conv_dim + 2 * K_dim + (long long)h * V_d;
    float* v_dst = v_out + ((long long)n * V_h + h) * V_d;
    for (int i = tid; i < V_d; i += BLOCK) {
        v_dst[i] = Elem<T>::to_f32(v_base[i]);
    }

    if (tid == 0) {
        const long long gh = (long long)n * V_h + h;
        const float av = Elem<T>::to_f32(a[gh]);
        const float bv = Elem<T>::to_f32(b[gh]);
        const float z = av + Elem<T>::to_f32(dt_bias[h]);
        const float sp = (z > 20.f) ? z : log1pf(__expf(z));
        g_log_out[gh] = -__expf(A_log[h]) * sp;
        beta_out[gh] = 1.f / (1.f + __expf(-bv));
    }
}

}  // namespace pie::ssm
