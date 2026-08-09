//===-- kda.cuh - Kimi Delta Attention's four kernels, as device text ---===//
//
// Four `__global__`s and one `__device__` helper, with no host function and
// no `<<<>>>` anywhere. `kda.cu` includes this file and keeps only its
// launchers, so each kernel is defined ONCE in the tree — the archive nvcc
// builds and any cubin NVRTC builds come from the same characters.
// `kernels-cuda/tests/sources.rs` refuses a second definition of a
// namespace-qualified `__global__` precisely because `norm/altup_aux` shipped
// two for a release and every test passed on whichever half it exercised.
//
// # Two rows of four, and the two refusals
//
// `kda_gate_beta` and `kda_o_norm_gated` are the rows. Both launch
// `dim3(T, H)` with `min(D, 256)` threads, which is
// `LaunchRule::PerHeadElementwise` — `[rows, q_heads]` at
// `clamp(head_dim, 32, 128)` — and the head axis the earlier draft of this
// note said did not exist is exactly the thing that arrived.
//
// The other two, `kda_recurrent_step_batched` and `kda_prefill_batched`,
// state the same `dim3(R, H)` grid and are still refused, on two grounds
// that have nothing to do with the axis:
//
//   • DYNAMIC SHARED MEMORY. Both size it `3 * D * sizeof(float)` on the
//     host and address it through `extern __shared__ float smem[]`. `Launch`
//     carries an `smem` field, but only `Rms` fills it and only with a fixed
//     32 bytes; every head-shaped rule returns zero. A rule-sized launch
//     would hand these two a zero-length allocation they immediately write
//     three arrays into — which the hardware does not report.
//   • AN `i64` OPERAND. Both take `slot_stride_elems` as a `long long`,
//     deliberately, because it is an ELEMENT count into a multi-gigabyte
//     arena. `Args::bind` accepts pointers, `I32`, `U32`, `F32` and `Usize`,
//     and nothing else. A row for either would not be fireable even with the
//     grid right.
//
// Neither is templated, and that is deliberate rather than pending: a
// template is the price of admission for a row and buys nothing else, so a
// template with one instantiation and no row is a rename. Inventing a rule
// to fit them would put a guess where a refusal belongs.
//
// # Why they are templates when the launchers are `_bf16`
//
// The ahead-of-time build had to choose its instantiations and chose bf16,
// which is a fact about nvcc's compile time rather than about the
// arithmetic — every one of these accumulates in fp32 and narrows once. The
// element type is spelled `ElemT` here — `T` is already the token count in
// `kda_gate_beta`'s parameter list — and resolved through the prelude's
// `Elem<>`, so the f16 twin costs a row rather than a translation unit. The
// two pure-fp32 recurrences take no element parameter: there is nothing for
// it to name.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "pie_device.cuh"

namespace pie_cuda_driver::kernels::ssm::device {

// The scalar layer is the PRELUDE's. Named here so the kernels read as they
// did, so a row may spell its element type `device::bf16`, and so the
// launchers in `kda.cu` — which sit in `kernels::ssm` and would otherwise
// find this namespace shadowing `kernels::device` — still resolve
// `device::bf16` and `device::i32` to the prelude's types.
using ::pie_cuda_driver::kernels::device::Elem;
using ::pie_cuda_driver::kernels::device::bf16;
using ::pie_cuda_driver::kernels::device::bf16_to_f32;
using ::pie_cuda_driver::kernels::device::f16;
using ::pie_cuda_driver::kernels::device::f32_to_bf16;
using ::pie_cuda_driver::kernels::device::i32;
using ::pie_cuda_driver::kernels::device::u32;
using ::pie_cuda_driver::kernels::device::u8;
using ::pie_cuda_driver::kernels::device::usize;

__device__ __forceinline__ float sigmoidf(float x) {
    return 1.f / (1.f + __expf(-x));
}

// ── Gate + beta ────────────────────────────────────────────────────

template <class ElemT>
__global__ void kda_gate_beta(
    const ElemT* __restrict__ raw_g,
    const ElemT* __restrict__ raw_beta,
    const float* __restrict__ A_log,
    const float* __restrict__ dt_bias,
    float* __restrict__ gate_out,
    float* __restrict__ beta_out,
    int T, int H, int D,
    float lower_bound)
{
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    if (t >= T || h >= H) return;

    const float a = __expf(A_log[h]);
    const long long base = ((long long)t * H + h) * D;

    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float g = Elem<ElemT>::to_f32(raw_g[base + d]) + dt_bias[(long long)h * D + d];
        float gate;
        if (lower_bound < 0.f) {
            gate = lower_bound * sigmoidf(a * g);
        } else {
            // softplus, guarded the way the reference kernel guards it: past
            // 20 the exp overflows and softplus(x) == x to fp32 precision.
            const float sp = (g > 20.f) ? g : __logf(1.f + __expf(g));
            gate = -a * sp;
        }
        gate_out[base + d] = gate;
    }

    if (threadIdx.x == 0) {
        beta_out[(long long)t * H + h] =
            sigmoidf(Elem<ElemT>::to_f32(raw_beta[(long long)t * H + h]));
    }
}

// ── Recurrence ─────────────────────────────────────────────────────

// One block per (request, head), one **warp** per `v` row.
//
// The obvious mapping -- a thread per `v`, looping over `k` -- reads
// `state[v][k]` with 32 threads at 32 different rows, so every warp load
// touches 32 cache lines and returns four useful bytes from each. Giving a
// warp the row instead makes the lanes walk `k` contiguously, which is one
// line per load, and turns the two reductions into shuffles.
__global__ void kda_recurrent_step_batched(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ state_base,
    const device::i32* __restrict__ slot_ids,
    long long slot_stride_elems,
    float* __restrict__ out,
    int H, int D)
{
    const int r = blockIdx.x;
    const int h = blockIdx.y;

    const long long rh = (long long)r * H + h;
    const float* q_h = q_norm + rh * D;
    const float* k_h = k_norm + rh * D;
    const float* v_h = v      + rh * D;
    const float* g_h = gate   + rh * D;
    const float beta_h = beta[rh];

    float* st = state_base + (long long)slot_ids[r] * slot_stride_elems +
                (long long)h * D * D;
    float* out_h = out + rh * D;

    // q, k and the decay are read once per row, so stage them in shared
    // memory rather than re-reading them D times from L2.
    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + D;
    float* sg = smem + 2 * D;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        sq[i] = q_h[i];
        sk[i] = k_h[i];
        sg[i] = __expf(g_h[i]);
    }
    __syncthreads();

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;

    for (int vi = warp; vi < D; vi += warps) {
        float* row = st + (long long)vi * D;
        float mem = 0.f;
        for (int ki = lane; ki < D; ki += 32) {
            const float sv = row[ki] * sg[ki];
            row[ki] = sv;
            mem += sv * sk[ki];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            mem += __shfl_down_sync(0xffffffffu, mem, off);
        }
        const float delta = __shfl_sync(0xffffffffu, (v_h[vi] - mem) * beta_h, 0);

        float acc = 0.f;
        for (int ki = lane; ki < D; ki += 32) {
            const float sv = row[ki] + sk[ki] * delta;
            row[ki] = sv;
            acc += sv * sq[ki];
        }
        #pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            acc += __shfl_down_sync(0xffffffffu, acc, off);
        }
        if (lane == 0) out_h[vi] = acc;
    }
}

// One block per (request, head); the block walks its whole window because the
// recurrence has a strict per-token state dependency. Same warp-per-`v` row
// mapping as the decode step, for the same coalescing reason.
__global__ void kda_prefill_batched(
    const float* __restrict__ q_norm,
    const float* __restrict__ k_norm,
    const float* __restrict__ v,
    const float* __restrict__ gate,
    const float* __restrict__ beta,
    float* __restrict__ state_base,
    const device::i32* __restrict__ slot_ids,
    const device::u32* __restrict__ qo_indptr,
    long long slot_stride_elems,
    float* __restrict__ out,
    int H, int D)
{
    const int r = blockIdx.x;
    const int h = blockIdx.y;

    const long long begin = qo_indptr[r];
    const long long end = qo_indptr[r + 1];
    if (end <= begin) return;

    float* st = state_base + (long long)slot_ids[r] * slot_stride_elems +
                (long long)h * D * D;

    extern __shared__ float smem[];
    float* sq = smem;
    float* sk = smem + D;
    float* sg = smem + 2 * D;

    const int lane = threadIdx.x & 31;
    const int warp = threadIdx.x >> 5;
    const int warps = blockDim.x >> 5;

    for (long long t = begin; t < end; ++t) {
        const long long th = t * H + h;
        for (int i = threadIdx.x; i < D; i += blockDim.x) {
            sq[i] = q_norm[th * D + i];
            sk[i] = k_norm[th * D + i];
            sg[i] = __expf(gate[th * D + i]);
        }
        __syncthreads();

        const float beta_h = beta[th];
        for (int vi = warp; vi < D; vi += warps) {
            float* row = st + (long long)vi * D;
            float mem = 0.f;
            for (int ki = lane; ki < D; ki += 32) {
                const float sv = row[ki] * sg[ki];
                row[ki] = sv;
                mem += sv * sk[ki];
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                mem += __shfl_down_sync(0xffffffffu, mem, off);
            }
            const float delta =
                __shfl_sync(0xffffffffu, (v[th * D + vi] - mem) * beta_h, 0);

            float acc = 0.f;
            for (int ki = lane; ki < D; ki += 32) {
                const float sv = row[ki] + sk[ki] * delta;
                row[ki] = sv;
                acc += sv * sq[ki];
            }
            #pragma unroll
            for (int off = 16; off > 0; off >>= 1) {
                acc += __shfl_down_sync(0xffffffffu, acc, off);
            }
            if (lane == 0) out[th * D + vi] = acc;
        }
        // The next token reads the state this one just wrote, and reloads
        // shared memory over the values this one is still reading.
        __syncthreads();
    }
}

// ── Gated output norm ──────────────────────────────────────────────

// One block per (token, head). RMS over the head's D channels, then scale by
// `weight` and the sigmoid of the gate.
template <class ElemT>
__global__ void kda_o_norm_gated(
    const float* __restrict__ o,
    const ElemT* __restrict__ g,
    const float* __restrict__ weight,
    ElemT* __restrict__ out,
    int H, int D, float eps)
{
    const int t = blockIdx.x;
    const int h = blockIdx.y;
    const long long base = ((long long)t * H + h) * D;

    float acc = 0.f;
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float x = o[base + d];
        acc += x * x;
    }
    __shared__ float ssum;
    // Block-wide reduction through shared memory; D is 128 on K3, so one
    // warp-level reduction plus a single accumulator is enough.
    for (int offset = warpSize / 2; offset > 0; offset >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, offset);
    }
    if (threadIdx.x == 0) ssum = 0.f;
    __syncthreads();
    if ((threadIdx.x & (warpSize - 1)) == 0) atomicAdd(&ssum, acc);
    __syncthreads();

    const float scale = rsqrtf(ssum / static_cast<float>(D) + eps);
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        const float gate = Elem<ElemT>::to_f32(g[base + d]);
        const float y = o[base + d] * scale * weight[d] * sigmoidf(gate);
        out[base + d] = Elem<ElemT>::from_f32(y);
    }
}

}  // namespace pie_cuda_driver::kernels::ssm::device
