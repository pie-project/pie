#pragma once

#include "prelude/device.cuh"

namespace pie::linear {

template <class T>
__global__ void mlp_swiglu_split(
    const T* __restrict__ gate,
    const T* __restrict__ up,
    T* __restrict__ y,
    i32 n)
{
    const i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const float g = Elem<T>::to_f32(gate[idx]);
    const float u = Elem<T>::to_f32(up[idx]);
    const float silu = g / (1.f + expf(-g));
    y[idx] = Elem<T>::from_f32(silu * u);
}

template <class T>
__global__ void mlp_swiglu_clamp_alpha_split(
    const T* __restrict__ gate,
    const T* __restrict__ up,
    T* __restrict__ y,
    f16* __restrict__ y_fp16,
    i32 n,
    float limit,
    float alpha)
{
    const i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = Elem<T>::to_f32(gate[idx]);
    float u = Elem<T>::to_f32(up[idx]);
    g = fminf(g, limit);
    u = fminf(fmaxf(u, -limit), limit);
    const float glu = g / (1.f + expf(-alpha * g));
    const float out = (u + 1.f) * glu;
    y[idx] = Elem<T>::from_f32(out);
    if (y_fp16 != nullptr) y_fp16[idx] = f32_to_f16(out);
}

template <class T>
__global__ void mlp_swiglu_clamp_split(
    const T* __restrict__ gate,
    const T* __restrict__ up,
    T* __restrict__ y,
    i32 n,
    float limit)
{
    const i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    float g = Elem<T>::to_f32(gate[idx]);
    float u = Elem<T>::to_f32(up[idx]);
    g = fminf(g, limit);
    u = fminf(fmaxf(u, -limit), limit);
    y[idx] = Elem<T>::from_f32((g / (1.f + expf(-g))) * u);
}

template <class T>
__global__ void mlp_situ_split(
    const T* __restrict__ gate,
    const T* __restrict__ up,
    T* __restrict__ y,
    i32 n,
    float beta,
    float linear_beta)
{
    const i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    const float g = Elem<T>::to_f32(gate[idx]);
    float u = Elem<T>::to_f32(up[idx]);
    const float s = beta * tanhf(g / beta) / (1.f + expf(-g));
    if (linear_beta > 0.f) {
        u = linear_beta * tanhf(u / linear_beta);
    }
    y[idx] = Elem<T>::from_f32(s * u);
}

template <class T>
__global__ void mlp_geglu_tanh(
    const T* __restrict__ gate,
    const T* __restrict__ up,
    T* __restrict__ y,
    i32 n)
{
    const i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    constexpr float c = 0.7978845608028654f;
    const float g = Elem<T>::to_f32(gate[idx]);
    const float u = Elem<T>::to_f32(up[idx]);
    const float gelu = 0.5f * g * (1.f + tanhf(c * (g + 0.044715f * g * g * g)));
    y[idx] = Elem<T>::from_f32(gelu * u);
}

/// **THE UNGATED GELU** (`.wiki/alto/multimodal.md` §6.2).
///
/// `y = gelu_tanh(x)`, one thread per element, no `up` half to multiply.
/// `Qwen3_5VisionMLP` is `linear_fc2(act(linear_fc1(x)))` with
/// `hidden_act: gelu_pytorch_tanh` and the merger is the same shape — NOT
/// gated, which every other gelu arm on this plane assumes.
///
/// **WHAT NOT HAVING THIS COSTS, said so the arm's existence is a number.**
/// It is bakeable without a kernel: declare `gate_up` at `[2*inter, hidden]`
/// with the `up` half zero and the `up` half of the bias one, and
/// `mlp_geglu_tanh_packed` computes `gelu_tanh(fc1(x)) * 1`. That pays the
/// GEMM and the bank twice over — on qwen36's 27 blocks at 1152 -> 4304 it is
/// 268 M parameters, 0.5 GiB of bf16, written and multiplied to produce ones.
/// The tanh polynomial here is `mlp_geglu_tanh`'s, transcribed, so the two
/// spellings answer the same number.
template <class T>
__global__ void mlp_gelu_tanh(
    const T* __restrict__ x,
    T* __restrict__ y,
    i32 n)
{
    const i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    constexpr float c = 0.7978845608028654f;
    const float g = Elem<T>::to_f32(x[idx]);
    y[idx] = Elem<T>::from_f32(
        0.5f * g * (1.f + tanhf(c * (g + 0.044715f * g * g * g))));
}

template <class T>
__global__ void relu2(
    const T* __restrict__ x,
    T* __restrict__ y,
    i32 n)
{
    const i32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = fmaxf(Elem<T>::to_f32(x[i]), 0.f);
    y[i] = Elem<T>::from_f32(v * v);
}

template <class T>
__global__ void gate_sigmoid_mul(
    T* __restrict__ x,
    const T* __restrict__ gate,
    i32 n)
{
    const i32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float xv = Elem<T>::to_f32(x[i]);
    const float gv = Elem<T>::to_f32(gate[i]);
    const float s = 1.f / (1.f + __expf(-gv));
    x[i] = Elem<T>::from_f32(xv * s);
}

template <class T>
__global__ void mlp_swiglu_clamp_alpha_split_strided(
    const T* __restrict__ gate,
    const T* __restrict__ up,
    T* __restrict__ y,
    i32 cols, i32 in_stride, i32 out_stride, float limit, float alpha)
{
    const i32 row = blockIdx.x;
    const i32 col = blockIdx.y * blockDim.x + threadIdx.x;
    if (col >= cols) return;
    const long long i = static_cast<long long>(row) * in_stride + col;
    float g = Elem<T>::to_f32(gate[i]);
    float u = Elem<T>::to_f32(up[i]);
    if (limit > 0.f) {
        g = fminf(g, limit);
        u = fmaxf(fminf(u, limit), -limit);
    }
    const float glu = g / (1.f + __expf(-alpha * g));
    y[static_cast<long long>(row) * out_stride + col] =
        Elem<T>::from_f32((u + 1.f) * glu);
}

template <bool GateSecond>
__device__ __forceinline__ i32 gate_offset(i32 i, i32 I) {
    return GateSecond ? I + i : i;
}

template <bool GateSecond>
__device__ __forceinline__ i32 up_offset(i32 i, i32 I) {
    return GateSecond ? i : I + i;
}

template <class T, bool GateSecond>
__device__ __forceinline__ void mlp_swiglu_body(
    const T* __restrict__ packed,
    T* __restrict__ y,
    i32 I)
{
    const i32 n = blockIdx.x;
    const i32 i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= I) return;

    const long long row = static_cast<long long>(n) * I;
    const long long packed_row = row * 2;
    const float g = Elem<T>::to_f32(packed[packed_row + gate_offset<GateSecond>(i, I)]);
    const float u = Elem<T>::to_f32(packed[packed_row + up_offset<GateSecond>(i, I)]);
    const float silu = g / (1.f + __expf(-g));
    y[row + i] = Elem<T>::from_f32(silu * u);
}

template <class T>
__global__ void mlp_swiglu(const T* __restrict__ packed, T* __restrict__ y, i32 I) {
    mlp_swiglu_body<T, false>(packed, y, I);
}

template <class T>
__global__ void mlp_swiglu_gate_second(
    const T* __restrict__ packed, T* __restrict__ y, i32 I)
{
    mlp_swiglu_body<T, true>(packed, y, I);
}

template <class T, bool GateSecond>
__device__ __forceinline__ void mlp_situ_body(
    const T* __restrict__ packed,
    T* __restrict__ y,
    i32 I, float beta, float linear_beta)
{
    const i32 n = blockIdx.x;
    const i32 i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= I) return;

    const long long row = static_cast<long long>(n) * I;
    const long long packed_row = row * 2;
    const float g = Elem<T>::to_f32(packed[packed_row + gate_offset<GateSecond>(i, I)]);
    float u = Elem<T>::to_f32(packed[packed_row + up_offset<GateSecond>(i, I)]);
    const float s = beta * tanhf(g / beta) / (1.f + __expf(-g));
    if (linear_beta > 0.f) {
        u = linear_beta * tanhf(u / linear_beta);
    }
    y[row + i] = Elem<T>::from_f32(s * u);
}

template <class T>
__global__ void mlp_situ(
    const T* __restrict__ packed, T* __restrict__ y,
    i32 I, float beta, float linear_beta)
{
    mlp_situ_body<T, false>(packed, y, I, beta, linear_beta);
}

template <class T>
__global__ void mlp_situ_gate_second(
    const T* __restrict__ packed, T* __restrict__ y,
    i32 I, float beta, float linear_beta)
{
    mlp_situ_body<T, true>(packed, y, I, beta, linear_beta);
}

template <class T, bool GateSecond>
__device__ __forceinline__ void mlp_geglu_tanh_packed_body(
    const T* __restrict__ packed,
    T* __restrict__ y,
    i32 I)
{
    const i32 n = blockIdx.x;
    const i32 i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= I) return;
    const long long packed_row = static_cast<long long>(n) * 2 * I;
    const float g = Elem<T>::to_f32(packed[packed_row + gate_offset<GateSecond>(i, I)]);
    const float u = Elem<T>::to_f32(packed[packed_row + up_offset<GateSecond>(i, I)]);
    constexpr float kAlpha = 0.7978845608028654f;
    constexpr float kBeta = 0.044715f;
    const float inner = kAlpha * (g + kBeta * g * g * g);
    const float gelu = 0.5f * g * (1.f + tanhf(inner));
    y[static_cast<long long>(n) * I + i] = Elem<T>::from_f32(gelu * u);
}

template <class T>
__global__ void mlp_geglu_tanh_packed(
    const T* __restrict__ packed, T* __restrict__ y, i32 I)
{
    mlp_geglu_tanh_packed_body<T, false>(packed, y, I);
}

template <class T>
__global__ void mlp_geglu_tanh_packed_gate_second(
    const T* __restrict__ packed, T* __restrict__ y, i32 I)
{
    mlp_geglu_tanh_packed_body<T, true>(packed, y, I);
}

template <class T>
__global__ void mlp_swiglu_clamp(
    const T* __restrict__ packed,
    T* __restrict__ y,
    i32 I, float limit)
{
    const i32 n = blockIdx.x;
    const i32 i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= I) return;

    const long long row = static_cast<long long>(n) * I;
    const long long packed_row = row * 2;
    float g = Elem<T>::to_f32(packed[packed_row + i]);
    float u = Elem<T>::to_f32(packed[packed_row + I + i]);
    g = fminf(g, limit);
    u = fminf(fmaxf(u, -limit), limit);
    y[row + i] = Elem<T>::from_f32((g / (1.f + expf(-g))) * u);
}

template <class T>
__global__ void mlp_swiglu_clamp_alpha(
    const T* __restrict__ packed,
    T* __restrict__ y,
    i32 I, float limit, float alpha)
{
    const i32 n = blockIdx.x;
    const i32 i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= I) return;

    const long long row = static_cast<long long>(n) * I;
    const long long packed_row = row * 2;
    float g = Elem<T>::to_f32(packed[packed_row + i]);
    float u = Elem<T>::to_f32(packed[packed_row + I + i]);
    g = fminf(g, limit);
    u = fminf(fmaxf(u, -limit), limit);
    const float glu = g / (1.f + expf(-alpha * g));
    y[row + i] = Elem<T>::from_f32((u + 1.f) * glu);
}

template <class T>
__global__ void mlp_swiglu_strided(
    const T* __restrict__ packed,
    T* __restrict__ y,
    i32 I, i32 row_stride)
{
    const i32 n = blockIdx.x;
    const i32 i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= I) return;

    const long long row = static_cast<long long>(n) * I;
    const long long packed_row = static_cast<long long>(n) * row_stride;
    const float g = Elem<T>::to_f32(packed[packed_row + i]);
    const float u = Elem<T>::to_f32(packed[packed_row + I + i]);
    const float silu = g / (1.f + __expf(-g));
    y[row + i] = Elem<T>::from_f32(silu * u);
}

template <class T, bool GateSecond>
__device__ __forceinline__ void mlp_swiglu_vec2_body(
    const T* __restrict__ packed,
    T* __restrict__ y,
    i32 N, i32 I)
{
    static_assert(is_same<T, bf16>::value,
                  "the packed path is `bf16x2` arithmetic: a bf16 PAIR, not a "
                  "generic one. A second format needs its own pair type first.");
    const i32 n = blockIdx.x;
    const i32 i = (blockIdx.y * blockDim.x + threadIdx.x) * 2;
    if (n >= N || i >= I) return;

    const long long row = static_cast<long long>(n) * I;
    const long long packed_row = row * 2;
    if (((I & 1) == 0) && i + 1 < I) {
        const auto gate2 = *reinterpret_cast<const bf16x2*>(
            packed + packed_row + gate_offset<GateSecond>(i, I));
        const auto up2 = *reinterpret_cast<const bf16x2*>(
            packed + packed_row + up_offset<GateSecond>(i, I));
        const float2 g = bf16x2_to_f32(gate2);
        const float2 u = bf16x2_to_f32(up2);
        const float y0 = (g.x / (1.f + __expf(-g.x))) * u.x;
        const float y1 = (g.y / (1.f + __expf(-g.y))) * u.y;
        *reinterpret_cast<bf16x2*>(y + row + i) = f32_to_bf16x2(y0, y1);
        return;
    }

    const float g = bf16_to_f32(packed[packed_row + gate_offset<GateSecond>(i, I)]);
    const float u = bf16_to_f32(packed[packed_row + up_offset<GateSecond>(i, I)]);
    const float silu = g / (1.f + __expf(-g));
    y[row + i] = f32_to_bf16(silu * u);
}

template <class T>
__global__ void mlp_swiglu_vec2(
    const T* __restrict__ packed, T* __restrict__ y, i32 N, i32 I)
{
    mlp_swiglu_vec2_body<T, false>(packed, y, N, I);
}

template <class T>
__global__ void mlp_swiglu_vec2_gate_second(
    const T* __restrict__ packed, T* __restrict__ y, i32 N, i32 I)
{
    mlp_swiglu_vec2_body<T, true>(packed, y, N, I);
}

template <class T>
__global__ void mlp_swiglu_strided_vec2(
    const T* __restrict__ packed,
    T* __restrict__ y,
    i32 N, i32 I, i32 row_stride)
{
    static_assert(is_same<T, bf16>::value,
                  "the packed path is `bf16x2` arithmetic: a bf16 PAIR, not a "
                  "generic one. A second format needs its own pair type first.");
    const i32 n = blockIdx.x;
    const i32 i = (blockIdx.y * blockDim.x + threadIdx.x) * 2;
    if (n >= N || i >= I) return;

    const long long row = static_cast<long long>(n) * I;
    const long long packed_row = static_cast<long long>(n) * row_stride;
    if (((row_stride & 1) == 0) && ((I & 1) == 0) && i + 1 < I) {
        const auto gate2 = *reinterpret_cast<const bf16x2*>(packed + packed_row + i);
        const auto up2 = *reinterpret_cast<const bf16x2*>(packed + packed_row + I + i);
        const float2 g = bf16x2_to_f32(gate2);
        const float2 u = bf16x2_to_f32(up2);
        const float y0 = (g.x / (1.f + __expf(-g.x))) * u.x;
        const float y1 = (g.y / (1.f + __expf(-g.y))) * u.y;
        *reinterpret_cast<bf16x2*>(y + row + i) = f32_to_bf16x2(y0, y1);
        return;
    }

    const float g = bf16_to_f32(packed[packed_row + i]);
    const float u = bf16_to_f32(packed[packed_row + I + i]);
    const float silu = g / (1.f + __expf(-g));
    y[row + i] = f32_to_bf16(silu * u);
}

template <class T>
__global__ void moe_sigmoid_gate_add(
    T* __restrict__ out,
    const T* __restrict__ sum,
    const T* __restrict__ x,
    const T* __restrict__ scalar_gate,
    i32 H, i32 stride)
{
    const i32 n = blockIdx.x;
    const i32 h = blockIdx.y * blockDim.x + threadIdx.x;
    if (h >= H) return;
    const float gv = Elem<T>::to_f32(scalar_gate[static_cast<long long>(n) * stride]);
    const float s = 1.f / (1.f + __expf(-gv));
    const long long i = static_cast<long long>(n) * H + h;
    const float ov = Elem<T>::to_f32(sum[i]);
    const float xv = Elem<T>::to_f32(x[i]);
    out[i] = Elem<T>::from_f32(ov + xv * s);
}

template <class T>
__global__ void moe_sigmoid_gate_add_dot(
    const T* __restrict__ x,
    const T* __restrict__ gate_w,
    T* __restrict__ out,
    const T* __restrict__ y,
    i32 H)
{
    const i32 n = blockIdx.x;
    const i32 tid = threadIdx.x;
    const i32 lane = tid & 31;
    const i32 warp = tid >> 5;
    const i32 num_warps = static_cast<i32>(blockDim.x) >> 5;
    extern __shared__ float smem[];

    const T* x_row = x + static_cast<long long>(n) * H;
    T* out_row = out + static_cast<long long>(n) * H;
    const T* y_row = y + static_cast<long long>(n) * H;

    const bool vec = (H & 7) == 0 &&
        ((reinterpret_cast<usize>(x_row) |
          reinterpret_cast<usize>(gate_w) |
          reinterpret_cast<usize>(out_row) |
          reinterpret_cast<usize>(y_row)) & 15) == 0;

    float acc = 0.f;
    const i32 Hv = H >> 3;
    if (vec) {
        const uint4* xv = reinterpret_cast<const uint4*>(x_row);
        const uint4* gv = reinterpret_cast<const uint4*>(gate_w);
        for (i32 i = tid; i < Hv; i += blockDim.x) {
            const uint4 a = xv[i];
            const uint4 b = gv[i];
            const auto* ah = reinterpret_cast<const T*>(&a);
            const auto* bh = reinterpret_cast<const T*>(&b);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                acc += Elem<T>::to_f32(ah[j]) * Elem<T>::to_f32(bh[j]);
            }
        }
    } else {
        for (i32 h = tid; h < H; h += blockDim.x) {
            acc += Elem<T>::to_f32(x_row[h]) * Elem<T>::to_f32(gate_w[h]);
        }
    }

#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) smem[warp] = acc;
    __syncthreads();
    if (tid == 0) {
        float total = 0.f;
        for (i32 w = 0; w < num_warps; ++w) total += smem[w];
        smem[0] = 1.f / (1.f + __expf(-total));
    }
    __syncthreads();
    const float s = smem[0];

    if (vec) {
        uint4* ov = reinterpret_cast<uint4*>(out_row);
        const uint4* yv = reinterpret_cast<const uint4*>(y_row);
        for (i32 i = tid; i < Hv; i += blockDim.x) {
            uint4 o = ov[i];
            const uint4 yy = yv[i];
            auto* oh = reinterpret_cast<T*>(&o);
            const auto* yh = reinterpret_cast<const T*>(&yy);
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                oh[j] = Elem<T>::from_f32(
                    Elem<T>::to_f32(oh[j]) + Elem<T>::to_f32(yh[j]) * s);
            }
            ov[i] = o;
        }
    } else {
        for (i32 h = tid; h < H; h += blockDim.x) {
            const float ov = Elem<T>::to_f32(out_row[h]);
            const float yv = Elem<T>::to_f32(y_row[h]);
            out_row[h] = Elem<T>::from_f32(ov + yv * s);
        }
    }
}

}
