#include "prim/device.cuh"

namespace pie::elemwise {

template <class T>
__global__ void residual_add(T* __restrict__ y, const T* __restrict__ x, usize n,
                             int width, const u32* __restrict__ win) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (win != nullptr &&
        i >= static_cast<usize>(win[0]) * static_cast<usize>(width)) return;
    const usize at = win != nullptr
        ? i + static_cast<usize>(win[1]) * static_cast<usize>(width)
        : i;

    const float a = Elem<T>::to_f32(y[at]);
    const float b = Elem<T>::to_f32(x[at]);
    y[at] = Elem<T>::from_f32(a + b);
}

// Eight bf16 lanes a thread, rows of whole 16-byte groups (`width8` a row).
__global__ void residual_add_vec8(bf16* __restrict__ y, const bf16* __restrict__ x, usize n8,
                                  int width8, const u32* __restrict__ win) {
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n8) return;
    if (win != nullptr && i >= static_cast<usize>(win[0]) * static_cast<usize>(width8)) return;
    const usize at = win != nullptr ? i + static_cast<usize>(win[1]) * static_cast<usize>(width8) : i;
    float4 a = reinterpret_cast<const float4*>(y)[at];
    const float4 b = reinterpret_cast<const float4*>(x)[at];
    bf16x2* ah = reinterpret_cast<bf16x2*>(&a);
    const bf16x2* bh = reinterpret_cast<const bf16x2*>(&b);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const float2 fa = bf16x2_to_f32(ah[j]);
        const float2 fb = bf16x2_to_f32(bh[j]);
        ah[j] = f32_to_bf16x2(fa.x + fb.x, fa.y + fb.y);
    }
    reinterpret_cast<float4*>(y)[at] = a;
}

template <class T>
__global__ void mul_scalar(T* __restrict__ x, float s, usize n,
                           int width, const u32* __restrict__ win) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (win != nullptr &&
        i >= static_cast<usize>(win[0]) * static_cast<usize>(width)) return;
    const usize at = win != nullptr
        ? i + static_cast<usize>(win[1]) * static_cast<usize>(width)
        : i;

    const float s_rounded = Elem<T>::to_f32(Elem<T>::from_f32(s));
    x[at] = Elem<T>::from_f32(Elem<T>::to_f32(x[at]) * s_rounded);
}

template <class T>
__global__ void silu_scaled(T* __restrict__ x, float s, usize n,
                            int width, const u32* __restrict__ win) {
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (win != nullptr &&
        i >= static_cast<usize>(win[0]) * static_cast<usize>(width)) return;
    const usize at = win != nullptr
        ? i + static_cast<usize>(win[1]) * static_cast<usize>(width)
        : i;

    const float v = Elem<T>::to_f32(x[at]) * s;
    x[at] = Elem<T>::from_f32(v / (1.f + __expf(-v)));
}

template <class T>
__global__ void scale(T* __restrict__ x, const T* __restrict__ s, usize n,
                      int width, const u32* __restrict__ win) {
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (win != nullptr &&
        i >= static_cast<usize>(win[0]) * static_cast<usize>(width)) return;
    const usize at = win != nullptr
        ? i + static_cast<usize>(win[1]) * static_cast<usize>(width)
        : i;

    const float f = Elem<T>::to_f32(s[0]);
    x[at] = Elem<T>::from_f32(Elem<T>::to_f32(x[at]) * f);
}

template <class T, class TB = T>
__device__ __forceinline__ void add_bias_row(
    T* __restrict__ row,
    const TB* __restrict__ bias,
    int dim)
{
    for (int d = threadIdx.x; d < dim; d += blockDim.x) {
        const float v = Elem<T>::to_f32(row[d]) + Elem<TB>::to_f32(bias[d]);
        row[d] = Elem<T>::from_f32(v);
    }
}

template <class T, class TB = T>
__global__ void add_bias(
    T* __restrict__ out,
    const TB* __restrict__ bias,
    int dim,
    const u32* __restrict__ win)
{
    const int n = static_cast<int>(blockIdx.x);

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? n + static_cast<int>(win[1]) : n;

    add_bias_row<T, TB>(out + static_cast<long long>(row) * dim, bias, dim);
}

constexpr int kBinAdd = 0;
constexpr int kBinMul = 1;
constexpr int kBinDiv = 2;
constexpr int kBinMod = 3;

template <class T, int OP>
__global__ void binary(
    const T* __restrict__ x,
    const T* __restrict__ y,
    T* __restrict__ o,
    i32 n,
    i32 width,
    const u32* __restrict__ win)
{
    const i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (win != nullptr &&
        static_cast<long long>(idx) >= static_cast<long long>(win[0]) * width) return;
    const long long at =
        win != nullptr ? idx + static_cast<long long>(win[1]) * width : idx;

    const float a = Elem<T>::to_f32(x[at]);
    const float b = Elem<T>::to_f32(y[at]);
    float r;
    if (OP == kBinAdd) r = a + b;
    else if (OP == kBinMul) r = a * b;
    else if (OP == kBinDiv) r = a / b;
    else r = a - b * floorf(a / b);
    o[at] = Elem<T>::from_f32(r);
}

constexpr int kActSilu = 0;
constexpr int kActTanh = 1;
constexpr int kActSqrt = 2;

template <class T, int ACT>
__global__ void activation(
    const T* __restrict__ x,
    T* __restrict__ o,
    i32 n,
    i32 width,
    const u32* __restrict__ win)
{
    const i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    if (win != nullptr &&
        static_cast<long long>(idx) >= static_cast<long long>(win[0]) * width) return;
    const long long at =
        win != nullptr ? idx + static_cast<long long>(win[1]) * width : idx;

    const float v = Elem<T>::to_f32(x[at]);

    o[at] = Elem<T>::from_f32(ACT == kActSilu ? v / (1.f + __expf(-v))
                              : ACT == kActTanh ? tanhf(v) : sqrtf(v));
}

template <class T>
__global__ void clamp(T* __restrict__ x, float lo, float hi, usize n,
                      int width, const u32* __restrict__ win)
{
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (win != nullptr &&
        i >= static_cast<usize>(win[0]) * static_cast<usize>(width)) return;
    const usize at = win != nullptr
        ? i + static_cast<usize>(win[1]) * static_cast<usize>(width)
        : i;

    const float lo_r = Elem<T>::to_f32(Elem<T>::from_f32(lo));
    const float hi_r = Elem<T>::to_f32(Elem<T>::from_f32(hi));
    const float v = Elem<T>::to_f32(x[at]);
    x[at] = Elem<T>::from_f32(fminf(fmaxf(v, lo_r), hi_r));
}


template <class T>
__global__ void clamp_learned(
    T* __restrict__ x,
    const T* __restrict__ lo,
    const T* __restrict__ hi,
    usize n,
    int width,
    const u32* __restrict__ win)
{
    const usize i = static_cast<usize>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;

    if (win != nullptr &&
        i >= static_cast<usize>(win[0]) * static_cast<usize>(width)) return;
    const usize at = win != nullptr
        ? i + static_cast<usize>(win[1]) * static_cast<usize>(width)
        : i;

    const float lo_v = Elem<T>::to_f32(lo[0]);
    const float hi_v = Elem<T>::to_f32(hi[0]);
    const float v = Elem<T>::to_f32(x[at]);
    x[at] = Elem<T>::from_f32(fminf(fmaxf(v, lo_v), hi_v));
}

}
