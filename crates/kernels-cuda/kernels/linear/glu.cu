#include "linear/codec.cuh"
#include "linear/act_quant.cuh"
#include "prim/device.cuh"

namespace pie::linear {

// `linear.glu` landing its rows as the e4m3 read-back
// (`act_quant_e4m3_v8`'s values mode) of the bf16 rows it would land: eight
// columns a lane, a 128-wide block's 16 lanes adjacent. A select reading
// them back again lands the same values (each block's values sit on its
// scale's grid, or on the grid one below where the rounding lowered its
// amax), so it may read them as they are. `width` whole blocks.
__global__ void swiglu_clamp_routed_e4m3(
    const bf16* __restrict__ packed,
    const float* __restrict__ weights,
    bf16* __restrict__ y,
    i32 lanes,
    i32 width,
    float limit,
    i32 fan,
    const u32* __restrict__ win)
{
    using Block = E4m3Block<128>;
    const i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int lane = static_cast<int>(threadIdx.x % 32);
    const long long per_row = width / 8;
    long long live_lanes = lanes;
    long long first = 0;
    if (win != nullptr) {
        live_lanes = static_cast<long long>(win[0]) * fan * per_row;
        first = static_cast<long long>(win[1]) * fan * per_row;
    }
    // A lane past the last block keeps its place in the shuffle.
    const bool live = idx < lanes && idx < live_lanes;
    const long long at = (live ? idx : 0) + first;
    const long long row = at / per_row;
    const int col = static_cast<int>(at - row * per_row) * 8;
    const bf16* src = packed + row * 2 * width + col;
    const uint4 gw = *reinterpret_cast<const uint4*>(src);
    const uint4 uw = *reinterpret_cast<const uint4*>(src + width);
    const unsigned gs[4] = {gw.x, gw.y, gw.z, gw.w};
    const unsigned us[4] = {uw.x, uw.y, uw.z, uw.w};
    const float wt = weights[row];
    float v[8];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
#pragma unroll
        for (int h = 0; h < 2; ++h) {
            const unsigned gb = h == 0 ? gs[i] << 16 : gs[i] & 0xFFFF0000u;
            const unsigned ub = h == 0 ? us[i] << 16 : us[i] & 0xFFFF0000u;
            const float g = fminf(__uint_as_float(gb), limit);
            const float u = fminf(fmaxf(__uint_as_float(ub), -limit), limit);
            v[2 * i + h] = (g / (1.f + expf(-g))) * u * wt;
        }
    }
    Block q;
    q.hold(v);
    if (!live) return;
    const float s = __uint_as_float(static_cast<unsigned>(q.e + 127) << 23);
    const unsigned long long c = q.codes();
    unsigned w[4];
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const float lo = e4m3_value(static_cast<u8>(c >> (16 * i))) * s;
        const float hi = e4m3_value(static_cast<u8>(c >> (16 * i + 8))) * s;
        w[i] = pack_bf16x2(lo, hi);
    }
    *reinterpret_cast<uint4*>(y + row * width + col) = make_uint4(w[0], w[1], w[2], w[3]);
}

template <class T>
__global__ void geglu_tanh(
    const T* __restrict__ gate,
    const T* __restrict__ up,
    T* __restrict__ y,
    i32 n,
    i32 width,
    i32 fan,
    const u32* __restrict__ win)
{
    const i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (win != nullptr &&
        static_cast<long long>(idx) >=
            static_cast<long long>(win[0]) * fan * width) return;
    const long long at = win != nullptr
        ? idx + static_cast<long long>(win[1]) * fan * width
        : idx;

    constexpr float c = 0.7978845608028654f;
    const float g = Elem<T>::to_f32(gate[at]);
    const float u = Elem<T>::to_f32(up[at]);
    const float gelu = 0.5f * g * (1.f + tanhf(c * (g + 0.044715f * g * g * g)));
    y[at] = Elem<T>::from_f32(gelu * u);
}

// `gelu_tanh(x)` (`RELU2` false) or `relu(x)^2`, element for element.
template <class T, bool RELU2>
__global__ void act(
    const T* __restrict__ x,
    T* __restrict__ y,
    i32 n,
    i32 width,
    i32 fan,
    const u32* __restrict__ win)
{
    const i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (win != nullptr &&
        static_cast<long long>(idx) >=
            static_cast<long long>(win[0]) * fan * width) return;
    const long long at = win != nullptr
        ? idx + static_cast<long long>(win[1]) * fan * width
        : idx;

    constexpr float c = 0.7978845608028654f;
    const float g = Elem<T>::to_f32(x[at]);
    if constexpr (RELU2) {
        const float r = fmaxf(g, 0.f);
        y[at] = Elem<T>::from_f32(r * r);
    } else {
        y[at] = Elem<T>::from_f32(
            0.5f * g * (1.f + tanhf(c * (g + 0.044715f * g * g * g))));
    }
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
__device__ __forceinline__ void swiglu_body(
    const T* __restrict__ packed,
    T* __restrict__ y,
    i32 I,
    i32 fan,
    const u32* __restrict__ win)
{
    const i32 n = blockIdx.x;

    if (win != nullptr && n >= static_cast<i32>(win[0]) * fan) return;

    const i32 plane_row = win != nullptr ? n + static_cast<i32>(win[1]) * fan : n;
    const i32 i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= I) return;

    const long long row = static_cast<long long>(plane_row) * I;
    const long long packed_row = row * 2;
    const float g = Elem<T>::to_f32(packed[packed_row + gate_offset<GateSecond>(i, I)]);
    const float u = Elem<T>::to_f32(packed[packed_row + up_offset<GateSecond>(i, I)]);
    const float silu = g / (1.f + __expf(-g));
    y[row + i] = Elem<T>::from_f32(silu * u);
}

template <class T>
__global__ void swiglu(const T* __restrict__ packed, T* __restrict__ y, i32 I,
                           i32 fan, const u32* __restrict__ win) {
    swiglu_body<T, false>(packed, y, I, fan, win);
}

// `swiglu` eight columns a thread, read and written 16 bytes at a time:
// the same value for every element.
template <class T>
__global__ void swiglu_vec8(const T* __restrict__ packed, T* __restrict__ y, i32 I,
                                i32 fan, const u32* __restrict__ win) {
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    const i32 n = blockIdx.x;
    if (win != nullptr && n >= static_cast<i32>(win[0]) * fan) return;
    const i32 plane_row = win != nullptr ? n + static_cast<i32>(win[1]) * fan : n;
    const i32 i0 = (blockIdx.y * blockDim.x + threadIdx.x) * 8;
    if (i0 >= I) return;
    const long long row = static_cast<long long>(plane_row) * I;
    const long long packed_row = row * 2;
    const uint4 gw = *reinterpret_cast<const uint4*>(packed + packed_row + i0);
    const uint4 uw = *reinterpret_cast<const uint4*>(packed + packed_row + I + i0);
    const T* g8 = reinterpret_cast<const T*>(&gw);
    const T* u8 = reinterpret_cast<const T*>(&uw);
    uint4 out;
    T* o8 = reinterpret_cast<T*>(&out);
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const float g = Elem<T>::to_f32(g8[k]);
        const float u = Elem<T>::to_f32(u8[k]);
        const float silu = g / (1.f + __expf(-g));
        o8[k] = Elem<T>::from_f32(silu * u);
    }
    *reinterpret_cast<uint4*>(y + row + i0) = out;
}

template <class T, bool GateSecond>
__device__ __forceinline__ void situ_body(
    const T* __restrict__ packed,
    T* __restrict__ y,
    i32 I, float beta, float linear_beta,
    i32 fan,
    const u32* __restrict__ win)
{
    const i32 n = blockIdx.x;

    if (win != nullptr && n >= static_cast<i32>(win[0]) * fan) return;

    const i32 plane_row = win != nullptr ? n + static_cast<i32>(win[1]) * fan : n;
    const i32 i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= I) return;

    const long long row = static_cast<long long>(plane_row) * I;
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
__global__ void situ(
    const T* __restrict__ packed, T* __restrict__ y,
    i32 I, float beta, float linear_beta,
    i32 fan, const u32* __restrict__ win)
{
    situ_body<T, false>(packed, y, I, beta, linear_beta, fan, win);
}

template <class T, bool GateSecond>
__device__ __forceinline__ void geglu_tanh_packed_body(
    const T* __restrict__ packed,
    T* __restrict__ y,
    i32 I,
    i32 fan,
    const u32* __restrict__ win)
{
    const i32 n = blockIdx.x;

    if (win != nullptr && n >= static_cast<i32>(win[0]) * fan) return;

    const i32 plane_row = win != nullptr ? n + static_cast<i32>(win[1]) * fan : n;
    const i32 i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= I) return;

    const long long packed_row = static_cast<long long>(plane_row) * 2 * I;
    const float g = Elem<T>::to_f32(packed[packed_row + gate_offset<GateSecond>(i, I)]);
    const float u = Elem<T>::to_f32(packed[packed_row + up_offset<GateSecond>(i, I)]);
    constexpr float kAlpha = 0.7978845608028654f;
    constexpr float kBeta = 0.044715f;
    const float inner = kAlpha * (g + kBeta * g * g * g);
    const float gelu = 0.5f * g * (1.f + tanhf(inner));
    y[static_cast<long long>(plane_row) * I + i] = Elem<T>::from_f32(gelu * u);
}

template <class T>
__global__ void geglu_tanh_packed(
    const T* __restrict__ packed, T* __restrict__ y, i32 I,
    i32 fan, const u32* __restrict__ win)
{
    geglu_tanh_packed_body<T, false>(packed, y, I, fan, win);
}

// `geglu_tanh_packed` eight columns a thread, read and written 16 bytes
// at a time: the same value for every element.
template <class T>
__global__ void geglu_tanh_packed_vec8(
    const T* __restrict__ packed, T* __restrict__ y, i32 I,
    i32 fan, const u32* __restrict__ win)
{
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    const i32 n = blockIdx.x;
    if (win != nullptr && n >= static_cast<i32>(win[0]) * fan) return;
    const i32 plane_row = win != nullptr ? n + static_cast<i32>(win[1]) * fan : n;
    const i32 i0 = (blockIdx.y * blockDim.x + threadIdx.x) * 8;
    if (i0 >= I) return;
    const long long packed_row = static_cast<long long>(plane_row) * 2 * I;
    const uint4 gw = *reinterpret_cast<const uint4*>(packed + packed_row + gate_offset<false>(i0, I));
    const uint4 uw = *reinterpret_cast<const uint4*>(packed + packed_row + up_offset<false>(i0, I));
    const T* g8 = reinterpret_cast<const T*>(&gw);
    const T* u8 = reinterpret_cast<const T*>(&uw);
    uint4 out;
    T* o8 = reinterpret_cast<T*>(&out);
    constexpr float kAlpha = 0.7978845608028654f;
    constexpr float kBeta = 0.044715f;
#pragma unroll
    for (int k = 0; k < 8; ++k) {
        const float g = Elem<T>::to_f32(g8[k]);
        const float u = Elem<T>::to_f32(u8[k]);
        const float inner = kAlpha * (g + kBeta * g * g * g);
        const float gelu = 0.5f * g * (1.f + tanhf(inner));
        o8[k] = Elem<T>::from_f32(gelu * u);
    }
    *reinterpret_cast<uint4*>(y + static_cast<long long>(plane_row) * I + i0) = out;
}

template <class T>
__global__ void swiglu_clamp(
    const T* __restrict__ packed,
    T* __restrict__ y,
    i32 I, float limit,
    i32 fan,
    const u32* __restrict__ win)
{
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    const i32 n = blockIdx.x;

    if (win != nullptr && n >= static_cast<i32>(win[0]) * fan) return;

    const i32 plane_row = win != nullptr ? n + static_cast<i32>(win[1]) * fan : n;
    const i32 i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= I) return;

    const long long row = static_cast<long long>(plane_row) * I;
    const long long packed_row = row * 2;
    float g = Elem<T>::to_f32(packed[packed_row + i]);
    float u = Elem<T>::to_f32(packed[packed_row + I + i]);
    g = fminf(g, limit);
    u = fminf(fmaxf(u, -limit), limit);
    y[row + i] = Elem<T>::from_f32((g / (1.f + expf(-g))) * u);
}

// `swiglu_clamp` element `at` of a gate and an up plane held apart, or
// of packed rows (`PACKED`: row r's `width` gate columns at 2 r width, its up
// columns after them); under `ROUTED` each row times its route's weight
// before it rounds.
template <class T, bool PACKED, bool ROUTED>
__global__ void swiglu_clamp_lanes(
    const T* __restrict__ gate,
    const T* __restrict__ up,
    const float* __restrict__ weights,
    T* __restrict__ y,
    i32 n,
    i32 width,
    float limit,
    i32 fan,
    const u32* __restrict__ win)
{
    const i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;

    if (win != nullptr &&
        static_cast<long long>(idx) >=
            static_cast<long long>(win[0]) * fan * width) return;
    const long long at = win != nullptr
        ? idx + static_cast<long long>(win[1]) * fan * width
        : idx;
    const long long row = at / width;
    long long g_at = at;
    const T* u_plane = up;
    if constexpr (PACKED) {
        g_at = row * 2 * width + (at - row * width);
        u_plane = gate + width;
    }
    const float g = fminf(Elem<T>::to_f32(gate[g_at]), limit);
    const float u = fminf(fmaxf(Elem<T>::to_f32(u_plane[g_at]), -limit), limit);
    if constexpr (ROUTED) {
        y[at] = Elem<T>::from_f32((g / (1.f + expf(-g))) * u * weights[row]);
    } else {
        y[at] = Elem<T>::from_f32((g / (1.f + expf(-g))) * u);
    }
}

template <class T>
__global__ void swiglu_clamp_alpha(
    const T* __restrict__ packed,
    T* __restrict__ y,
    i32 I, float limit, float alpha,
    i32 fan,
    const u32* __restrict__ win)
{
    const i32 n = blockIdx.x;

    if (win != nullptr && n >= static_cast<i32>(win[0]) * fan) return;

    const i32 plane_row = win != nullptr ? n + static_cast<i32>(win[1]) * fan : n;
    const i32 i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= I) return;

    const long long row = static_cast<long long>(plane_row) * I;
    const long long packed_row = row * 2;
    float g = Elem<T>::to_f32(packed[packed_row + i]);
    float u = Elem<T>::to_f32(packed[packed_row + I + i]);
    g = fminf(g, limit);
    u = fminf(fmaxf(u, -limit), limit);
    const float glu = g / (1.f + expf(-alpha * g));
    y[row + i] = Elem<T>::from_f32((u + 1.f) * glu);
}

}
