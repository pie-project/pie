// The sigmoid gates: `x *= sigmoid(gate)` elementwise over a rectangle (with
// a 16-byte-a-lane form), and per head with a scale.

#include "prim/device.cuh"

namespace pie::elemwise {

template <class T>
__global__ void gate_sigmoid_mul(
    T* __restrict__ x,
    const T* __restrict__ gate,
    i32 n,
    i32 width,
    i32 fan,
    const u32* __restrict__ win)
{
    const i32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (win != nullptr &&
        static_cast<long long>(i) >=
            static_cast<long long>(win[0]) * fan * width) return;
    const long long at = win != nullptr
        ? i + static_cast<long long>(win[1]) * fan * width
        : i;

    const float xv = Elem<T>::to_f32(x[at]);
    const float gv = Elem<T>::to_f32(gate[at]);
    const float s = 1.f / (1.f + __expf(-gv));
    x[at] = Elem<T>::from_f32(xv * s);
}

// `gate_sigmoid_mul` 16 bytes a lane: `n8` counts eight-element runs.
__global__ void gate_sigmoid_mul_vec8(
    bf16* __restrict__ x,
    const bf16* __restrict__ gate,
    i32 n8,
    i32 width,
    i32 fan,
    const u32* __restrict__ win)
{
    const i32 i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n8) return;
    if (win != nullptr &&
        static_cast<long long>(i) * 8 >=
            static_cast<long long>(win[0]) * fan * width) return;
    const long long at = win != nullptr
        ? static_cast<long long>(i) * 8 + static_cast<long long>(win[1]) * fan * width
        : static_cast<long long>(i) * 8;

    uint4 xv = *reinterpret_cast<const uint4*>(x + at);
    const uint4 gv = *reinterpret_cast<const uint4*>(gate + at);
    bf16x2* xh = reinterpret_cast<bf16x2*>(&xv);
    const bf16x2* gh = reinterpret_cast<const bf16x2*>(&gv);
    #pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 a = bf16x2_to_f32(xh[j]);
        const float2 g = bf16x2_to_f32(gh[j]);
        xh[j] = f32_to_bf16x2(a.x * (1.f / (1.f + __expf(-g.x))), a.y * (1.f / (1.f + __expf(-g.y))));
    }
    *reinterpret_cast<uint4*>(x + at) = xv;
}

template <class T>
__global__ void gate_sigmoid_mul_heads(
    T* __restrict__ x,
    const T* __restrict__ gate,
    i32 heads,
    i32 head_dim,
    float scale,
    const u32* __restrict__ win)
{
    const i32 n = blockIdx.x;
    if (win != nullptr && n >= static_cast<i32>(win[0])) return;
    const i32 row = win != nullptr ? n + static_cast<i32>(win[1]) : n;

    const i32 width = heads * head_dim;
    T* xr = x + static_cast<long long>(row) * width;
    const T* gr = gate + static_cast<long long>(row) * heads;
    for (i32 idx = threadIdx.x; idx < width; idx += blockDim.x) {
        const float g = Elem<T>::to_f32(gr[idx / head_dim]);
        const float s = scale / (1.f + __expf(-g));
        xr[idx] = Elem<T>::from_f32(Elem<T>::to_f32(xr[idx]) * s);
    }
}

}  // namespace pie::elemwise
