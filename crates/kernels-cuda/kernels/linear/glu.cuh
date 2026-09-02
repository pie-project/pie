#pragma once

#include "prelude/device.cuh"

namespace pie::linear {

template <class T>
__global__ void mlp_geglu_tanh(
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
    // The staged-geometry seat, in the ELEMENT form this flat launch needs: a
    // lane is not a row here, so the live-rows word bounds `win[0] * width`
    // elements, and `win[1] * width` is where they begin. Armed, the planes
    // arrive at their base and this lane owns element `at`; null, they arrived
    // pre-shifted and `idx` is the element already.
    if (win != nullptr &&
        static_cast<long long>(idx) >=
            static_cast<long long>(win[0]) * fan * width) return;
    const long long at = win != nullptr
        ? idx + static_cast<long long>(win[1]) * fan * width
        : idx;
    // `gate`, `up` and `y` are one row per token on the axis the guard counts,
    // so all three read the shifted element.
    constexpr float c = 0.7978845608028654f;
    const float g = Elem<T>::to_f32(gate[at]);
    const float u = Elem<T>::to_f32(up[at]);
    const float gelu = 0.5f * g * (1.f + tanhf(c * (g + 0.044715f * g * g * g)));
    y[at] = Elem<T>::from_f32(gelu * u);
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
    i32 n,
    i32 width,
    i32 fan,
    const u32* __restrict__ win)
{
    const i32 idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    // The staged-geometry seat, in the ELEMENT form this flat launch needs: a
    // lane is not a row here, so the live-rows word bounds `win[0] * width`
    // elements, and `win[1] * width` is where they begin. Armed, the planes
    // arrive at their base and this lane owns element `at`; null, they arrived
    // pre-shifted and `idx` is the element already.
    if (win != nullptr &&
        static_cast<long long>(idx) >=
            static_cast<long long>(win[0]) * fan * width) return;
    const long long at = win != nullptr
        ? idx + static_cast<long long>(win[1]) * fan * width
        : idx;
    // `x` and `y` are one row per token on the axis the guard counts, so both
    // read the shifted element.
    constexpr float c = 0.7978845608028654f;
    const float g = Elem<T>::to_f32(x[at]);
    y[at] = Elem<T>::from_f32(
        0.5f * g * (1.f + tanhf(c * (g + 0.044715f * g * g * g))));
}

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
    // The staged-geometry seat, in the ELEMENT form this flat launch needs: a
    // lane is not a row here, so the live-rows word bounds `win[0] * width`
    // elements, and `win[1] * width` is where they begin. Armed, the planes
    // arrive at their base and this lane owns element `at`; null, they arrived
    // pre-shifted and `i` is the element already.
    if (win != nullptr &&
        static_cast<long long>(i) >=
            static_cast<long long>(win[0]) * fan * width) return;
    const long long at = win != nullptr
        ? i + static_cast<long long>(win[1]) * fan * width
        : i;
    // `x` and its gate are one row per token on the axis the guard counts, so
    // both read the shifted element.
    const float xv = Elem<T>::to_f32(x[at]);
    const float gv = Elem<T>::to_f32(gate[at]);
    const float s = 1.f / (1.f + __expf(-gv));
    x[at] = Elem<T>::from_f32(xv * s);
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
    i32 I,
    i32 fan,
    const u32* __restrict__ win)
{
    const i32 n = blockIdx.x;
    // **THE SEAT COUNTS TOKENS AND THIS PLANE COUNTS `fan` ROWS PER TOKEN.**
    // The staged-geometry seat (qkv_fused.cuh's idiom) retires the rows a
    // replay's bucket-carved grid added, off a word the fire staged and not a
    // parameter the recording baked — and `win[0]` counts them in TOKENS. A
    // dense MLP's packed rectangle is one row per token and `fan` is 1; a
    // ROUTED one is a `select`'s output, one row per ROUTE, so a window of
    // `win[0]` token rows starting at `win[1]` is a run of `win[0] * fan`
    // rows starting at row `win[1] * fan`. `moe.cuh`'s select states the
    // identical rule at its own guard, and multiplies once for it.
    //
    // Comparing a route index against a token count is what this multiply
    // closes: at eight tokens and a fan-out of eight it computed the first
    // eight of sixty-four routes and returned from the rest, so every route
    // past the first token's kept the bytes of the fire before this one — a
    // one-fire lag on a routed SKU, invisible on a dense one, and absent
    // whenever the seat is null because then nothing is retired at all.
    if (win != nullptr && n >= static_cast<i32>(win[0]) * fan) return;
    // The seat's second word says WHERE those rows are, on this plane's own
    // axis. Armed, the pointers are the plane's own base and this block owns
    // plane row `win[1] * fan + n`; null, they arrived pre-shifted and `n` is
    // the row already.
    const i32 plane_row = win != nullptr ? n + static_cast<i32>(win[1]) * fan : n;
    const i32 i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= I) return;

    // `packed` and `y` are both one row per PLANE ROW — the axis the guard
    // counts once it is scaled by `fan` — so both read the shifted row.
    const long long row = static_cast<long long>(plane_row) * I;
    const long long packed_row = row * 2;
    const float g = Elem<T>::to_f32(packed[packed_row + gate_offset<GateSecond>(i, I)]);
    const float u = Elem<T>::to_f32(packed[packed_row + up_offset<GateSecond>(i, I)]);
    const float silu = g / (1.f + __expf(-g));
    y[row + i] = Elem<T>::from_f32(silu * u);
}

template <class T>
__global__ void mlp_swiglu(const T* __restrict__ packed, T* __restrict__ y, i32 I,
                           i32 fan, const u32* __restrict__ win) {
    mlp_swiglu_body<T, false>(packed, y, I, fan, win);
}

template <class T, bool GateSecond>
__device__ __forceinline__ void mlp_situ_body(
    const T* __restrict__ packed,
    T* __restrict__ y,
    i32 I, float beta, float linear_beta,
    i32 fan,
    const u32* __restrict__ win)
{
    const i32 n = blockIdx.x;
    // **THE SEAT COUNTS TOKENS AND THIS PLANE COUNTS `fan` ROWS PER TOKEN.**
    // The staged-geometry seat (qkv_fused.cuh's idiom) retires the rows a
    // replay's bucket-carved grid added, off a word the fire staged and not a
    // parameter the recording baked — and `win[0]` counts them in TOKENS. A
    // dense MLP's packed rectangle is one row per token and `fan` is 1; a
    // ROUTED one is a `select`'s output, one row per ROUTE, so a window of
    // `win[0]` token rows starting at `win[1]` is a run of `win[0] * fan`
    // rows starting at row `win[1] * fan`. `moe.cuh`'s select states the
    // identical rule at its own guard, and multiplies once for it.
    //
    // Comparing a route index against a token count is what this multiply
    // closes: at eight tokens and a fan-out of eight it computed the first
    // eight of sixty-four routes and returned from the rest, so every route
    // past the first token's kept the bytes of the fire before this one — a
    // one-fire lag on a routed SKU, invisible on a dense one, and absent
    // whenever the seat is null because then nothing is retired at all.
    if (win != nullptr && n >= static_cast<i32>(win[0]) * fan) return;
    // The seat's second word says WHERE those rows are, on this plane's own
    // axis. Armed, the pointers are the plane's own base and this block owns
    // plane row `win[1] * fan + n`; null, they arrived pre-shifted and `n` is
    // the row already.
    const i32 plane_row = win != nullptr ? n + static_cast<i32>(win[1]) * fan : n;
    const i32 i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= I) return;

    // `packed` and `y` are both one row per PLANE ROW — the axis the guard
    // counts once it is scaled by `fan` — so both read the shifted row.
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
__global__ void mlp_situ(
    const T* __restrict__ packed, T* __restrict__ y,
    i32 I, float beta, float linear_beta,
    i32 fan, const u32* __restrict__ win)
{
    mlp_situ_body<T, false>(packed, y, I, beta, linear_beta, fan, win);
}

template <class T, bool GateSecond>
__device__ __forceinline__ void mlp_geglu_tanh_packed_body(
    const T* __restrict__ packed,
    T* __restrict__ y,
    i32 I,
    i32 fan,
    const u32* __restrict__ win)
{
    const i32 n = blockIdx.x;
    // **THE SEAT COUNTS TOKENS AND THIS PLANE COUNTS `fan` ROWS PER TOKEN.**
    // The staged-geometry seat (qkv_fused.cuh's idiom) retires the rows a
    // replay's bucket-carved grid added, off a word the fire staged and not a
    // parameter the recording baked — and `win[0]` counts them in TOKENS. A
    // dense MLP's packed rectangle is one row per token and `fan` is 1; a
    // ROUTED one is a `select`'s output, one row per ROUTE, so a window of
    // `win[0]` token rows starting at `win[1]` is a run of `win[0] * fan`
    // rows starting at row `win[1] * fan`. `moe.cuh`'s select states the
    // identical rule at its own guard, and multiplies once for it.
    //
    // Comparing a route index against a token count is what this multiply
    // closes: at eight tokens and a fan-out of eight it computed the first
    // eight of sixty-four routes and returned from the rest, so every route
    // past the first token's kept the bytes of the fire before this one — a
    // one-fire lag on a routed SKU, invisible on a dense one, and absent
    // whenever the seat is null because then nothing is retired at all.
    if (win != nullptr && n >= static_cast<i32>(win[0]) * fan) return;
    // The seat's second word says WHERE those rows are, on this plane's own
    // axis. Armed, the pointers are the plane's own base and this block owns
    // plane row `win[1] * fan + n`; null, they arrived pre-shifted and `n` is
    // the row already.
    const i32 plane_row = win != nullptr ? n + static_cast<i32>(win[1]) * fan : n;
    const i32 i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= I) return;
    // `packed` and `y` are both one row per PLANE ROW — the axis the guard
    // counts once it is scaled by `fan` — so both read the shifted row.
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
__global__ void mlp_geglu_tanh_packed(
    const T* __restrict__ packed, T* __restrict__ y, i32 I,
    i32 fan, const u32* __restrict__ win)
{
    mlp_geglu_tanh_packed_body<T, false>(packed, y, I, fan, win);
}

template <class T>
__global__ void mlp_swiglu_clamp(
    const T* __restrict__ packed,
    T* __restrict__ y,
    i32 I, float limit,
    i32 fan,
    const u32* __restrict__ win)
{
    const i32 n = blockIdx.x;
    // **THE SEAT COUNTS TOKENS AND THIS PLANE COUNTS `fan` ROWS PER TOKEN.**
    // The staged-geometry seat (qkv_fused.cuh's idiom) retires the rows a
    // replay's bucket-carved grid added, off a word the fire staged and not a
    // parameter the recording baked — and `win[0]` counts them in TOKENS. A
    // dense MLP's packed rectangle is one row per token and `fan` is 1; a
    // ROUTED one is a `select`'s output, one row per ROUTE, so a window of
    // `win[0]` token rows starting at `win[1]` is a run of `win[0] * fan`
    // rows starting at row `win[1] * fan`. `moe.cuh`'s select states the
    // identical rule at its own guard, and multiplies once for it.
    //
    // Comparing a route index against a token count is what this multiply
    // closes: at eight tokens and a fan-out of eight it computed the first
    // eight of sixty-four routes and returned from the rest, so every route
    // past the first token's kept the bytes of the fire before this one — a
    // one-fire lag on a routed SKU, invisible on a dense one, and absent
    // whenever the seat is null because then nothing is retired at all.
    if (win != nullptr && n >= static_cast<i32>(win[0]) * fan) return;
    // The seat's second word says WHERE those rows are, on this plane's own
    // axis. Armed, the pointers are the plane's own base and this block owns
    // plane row `win[1] * fan + n`; null, they arrived pre-shifted and `n` is
    // the row already.
    const i32 plane_row = win != nullptr ? n + static_cast<i32>(win[1]) * fan : n;
    const i32 i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= I) return;

    // `packed` and `y` are both one row per PLANE ROW — the axis the guard
    // counts once it is scaled by `fan` — so both read the shifted row.
    const long long row = static_cast<long long>(plane_row) * I;
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
    i32 I, float limit, float alpha,
    i32 fan,
    const u32* __restrict__ win)
{
    const i32 n = blockIdx.x;
    // **THE SEAT COUNTS TOKENS AND THIS PLANE COUNTS `fan` ROWS PER TOKEN.**
    // The staged-geometry seat (qkv_fused.cuh's idiom) retires the rows a
    // replay's bucket-carved grid added, off a word the fire staged and not a
    // parameter the recording baked — and `win[0]` counts them in TOKENS. A
    // dense MLP's packed rectangle is one row per token and `fan` is 1; a
    // ROUTED one is a `select`'s output, one row per ROUTE, so a window of
    // `win[0]` token rows starting at `win[1]` is a run of `win[0] * fan`
    // rows starting at row `win[1] * fan`. `moe.cuh`'s select states the
    // identical rule at its own guard, and multiplies once for it.
    //
    // Comparing a route index against a token count is what this multiply
    // closes: at eight tokens and a fan-out of eight it computed the first
    // eight of sixty-four routes and returned from the rest, so every route
    // past the first token's kept the bytes of the fire before this one — a
    // one-fire lag on a routed SKU, invisible on a dense one, and absent
    // whenever the seat is null because then nothing is retired at all.
    if (win != nullptr && n >= static_cast<i32>(win[0]) * fan) return;
    // The seat's second word says WHERE those rows are, on this plane's own
    // axis. Armed, the pointers are the plane's own base and this block owns
    // plane row `win[1] * fan + n`; null, they arrived pre-shifted and `n` is
    // the row already.
    const i32 plane_row = win != nullptr ? n + static_cast<i32>(win[1]) * fan : n;
    const i32 i = blockIdx.y * blockDim.x + threadIdx.x;
    if (i >= I) return;

    // `packed` and `y` are both one row per PLANE ROW — the axis the guard
    // counts once it is scaled by `fan` — so both read the shifted row.
    const long long row = static_cast<long long>(plane_row) * I;
    const long long packed_row = row * 2;
    float g = Elem<T>::to_f32(packed[packed_row + i]);
    float u = Elem<T>::to_f32(packed[packed_row + I + i]);
    g = fminf(g, limit);
    u = fminf(fmaxf(u, -limit), limit);
    const float glu = g / (1.f + expf(-alpha * g));
    y[row + i] = Elem<T>::from_f32((u + 1.f) * glu);
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
__global__ void moe_sigmoid_gate_add(
    T* __restrict__ out,
    const T* __restrict__ sum,
    const T* __restrict__ x,
    const T* __restrict__ scalar_gate,
    i32 H, i32 stride,
    const u32* __restrict__ win)
{
    const i32 n = blockIdx.x;
    // The staged-geometry seat (qkv_fused.cuh's idiom): a replay whose grid
    // was carved at a bucket retires its padded rows here, off a word the
    // fire staged, not a parameter the recording baked.
    if (win != nullptr && n >= static_cast<i32>(win[0])) return;
    // The seat's second word says WHERE those rows are. Armed, the pointers
    // are the plane's own base and this block owns plane row `win[1] + n`;
    // null, they arrived pre-shifted and `n` is the row already.
    const i32 plane_row = win != nullptr ? n + static_cast<i32>(win[1]) : n;
    const i32 h = blockIdx.y * blockDim.x + threadIdx.x;
    if (h >= H) return;
    // The gate is one scalar per token row at its row's head, so it is a
    // plane of the guarded axis like `out`, `sum` and `x`: all four shift
    // together, or the row would be gated by another row's scalar.
    const float gv =
        Elem<T>::to_f32(scalar_gate[static_cast<long long>(plane_row) * stride]);
    const float s = 1.f / (1.f + __expf(-gv));
    const long long i = static_cast<long long>(plane_row) * H + h;
    const float ov = Elem<T>::to_f32(sum[i]);
    const float xv = Elem<T>::to_f32(x[i]);
    out[i] = Elem<T>::from_f32(ov + xv * s);
}

}
