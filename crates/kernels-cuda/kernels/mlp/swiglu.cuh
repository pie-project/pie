//===-- swiglu.cuh - the MLP activations, as `__global__` templates ----===//
//
// Twenty-two `__global__` templates and one include. No host function, no
// `<<<>>>`, no stream -- everything this file used to know about geometry is
// a `LaunchRule` on a row in `kernels_cuda::families::mlp`, and
// everything it used to know about which element type to compile is the `T`
// a row names.
//
// # Why this file exists
//
// `swiglu.cu` held both halves: seventeen kernels and twenty-one launches,
// interleaved, each kernel wrapped in an anonymous namespace so the launcher
// three lines below was the only thing that could ever name it. NVRTC cannot
// compile that file -- it includes `<cstdint>`, it declares host functions
// taking a `cudaStream_t`, and its kernels have no external names to ask for.
// So the device half moved HERE, and `swiglu.cu` now includes this header and
// keeps only the launchers.
//
// SPLIT, never copied. Two definitions of one kernel is a half-finished
// migration where the archive gets one and the JIT gets the other, and they
// agree until the day someone fixes a bug in the copy their tests exercise;
// `norm/altup_aux` shipped exactly that for a release.
// `tests/device_sources.rs::no_global_is_defined_twice` is the check that
// keeps it from happening twice, and it compares namespace-qualified names
// across every `.cuh` under `kernels/`.
//
// # Why the kernels are templates now
//
// Every one of them was `_bf16` in its name and `bf16` in its
// signature, because an ahead-of-time build must NAME its instantiations and
// each one costs a translation unit's worth of `cicc`. That is a fact about
// nvcc, not about the arithmetic: all seventeen widen to fp32, compute, and
// narrow back. `Elem<T>` is the widen/narrow pair, specialised on the
// STORAGE type, so `T` is the only thing that changes and a second numeric
// format costs a ROW rather than a file. `norm/elementwise.cuh` documents the
// trick; this file is the first family to take it at scale.
//
// # Why `GateSecond` became a second template NAME
//
// Four kernels were `template <bool GateSecond>` -- the packed activation
// puts the gate half first or second depending on which framework exported
// the checkpoint, and a runtime branch on it inside the loop would cost a
// register and a predicate per element. Under the JIT a row spells ONE
// template argument: `DeviceKernel::instantiation()` formats exactly
// `path<elem>`, so `chunked_swiglu<bf16, true>` is not a string any row can
// say. The fix keeps the compile-time constant and moves the choice into the
// name: one `__device__` body per activation, two `__global__` templates that
// forward to it with the flag fixed, two rows. The dispatcher picks a SYMBOL,
// which is what it was already doing with a `bool` argument.
//
// # What the rules recover, and what stays
//
// A launcher's `(n + 255) / 256` is `LaunchRule::Elementwise`; a `dim3(N,
// ceil(I / BLOCK))` is `LaunchRule::ElementwiseRows`; a `<<<N, 256, (256 /
// 32) * sizeof(float)>>>` is `LaunchRule::Rms`, dynamic shared memory
// included. So the row count `N` is gone from every chunked kernel -- the
// grid covers `blockIdx.x < N` exactly and the guard could never fire.
//
// `I`, `H`, `cols` and the strides all STAYED, and the distinction is the one
// `altup_aux` had to learn: an extent the rule computes is geometry and
// belongs to the row, while a bound the kernel tests (`i >= I`) or an address
// it computes (`row * row_stride`) is layout and belongs to the kernel. The
// flat kernels keep `n` for the same reason -- `Elementwise` rounds the grid
// UP to a multiple of 256, so the last block runs threads the buffer does not
// have and the guard is what stops them.
//
// # The two transposes, and the one thing this file changed
//
// `gpt_oss_glu_strided` launched `dim3((cols + 255) / 256, rows)` -- column
// tiles on x, rows on y. `ElementwiseRows` is the other way round, and it is
// the other way round because every OTHER strided kernel in this family
// already was. So the kernel's two index lines swapped axes. Coverage is
// identical (the same threads exist, with x and y exchanged) and coalescing
// inside a block is unchanged, since a block still spans 256 contiguous
// columns of one row; only the order blocks are scheduled in differs.
//
// # Three kernels carry no row
//
// `chunked_swiglu_vec2`, its `_gate_second` twin and
// `chunked_swiglu_strided_vec2` move two elements per thread, so their
// launcher asks for `ceil(((I + 1) / 2) / BLOCK)` tiles -- a HALF-WIDTH grid.
// No `LaunchRule` states that, and inventing one to fit three kernels would
// put a geometry in the vocabulary that only these three mean. They stay here
// because the device half of a file belongs in the device half of a file.
//
// NOTHING LAUNCHES THEM ANY MORE. `swiglu.cu`'s launcher picked between them
// and their scalar twins at run time on `I > 10000` and on the parity of
// `row_stride` -- run-time predicates over an operand's VALUE, which is the
// other half of why no row can name them -- and that launcher is deleted
// (§54), along with the whole of the archive crate's
// `kernels-cuda/kernels/mlp/`. It had already
// stopped being CALLED before it was deleted: every `table::mlp` row is in
// `JIT_DISPATCHED`, so the fire resolves to the scalar
// `chunked_swiglu` template on `LaunchRule::ElementwiseRows` at every `I`.
// These three are kept, uninstantiated, because the arithmetic is the record
// of what the vectorised path DID; see `families::mlp`'s header for what
// re-landing it would have to measure first.
//
// They are templates all the same, with a `static_assert` naming the single
// instantiation they have: `bf16x2` is a bf16 PAIR, and the packed path is
// bf16 arithmetic rather than generic arithmetic. Written as a template so
// that a unit nobody asked for does not compile them into every cubin --
// an uninstantiated template emits nothing.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "prelude/device.cuh"

namespace pie::mlp {

// The scalar layer is the PRELUDE's, not this family's. Named here so the
// kernels read as they always did, so a row may keep spelling its element
// type `bf16`, and so `swiglu.cu`'s launchers -- which sit in
// `kernels::mlp` and say `bf16` on every cast -- resolve the same
// spelling to the same type through this namespace.

// ---------------------------------------------------------------------------
// The flat activations: one element per thread, `Elementwise`.
// ---------------------------------------------------------------------------

/// `y = silu(gate) * up`.
template <class T>
__global__ void swiglu(
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

/// gpt-oss's GLU: an asymmetric clamp on the gate and a symmetric one on the
/// up half, then QuickGELU with `alpha`. Matches HF's
/// `GptOssExperts._apply_gate`.
///
/// `y_fp16` is the SECOND output and it is fp16 whatever `T` is, because the
/// MXFP4 down-projection GEMV reads fp16 and the only thing standing between
/// it and this value was a whole kernel that read the bf16 back and wrote it
/// again. The activation is a few tens of KB, so that launch was essentially
/// all overhead. Rounded from the same fp32 the `T` comes from, not from the
/// `T`, so it is strictly no worse.
template <class T>
__global__ void gpt_oss_glu(
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

/// SwiGLU with both halves clamped to `limit`.
template <class T>
__global__ void swiglu_clamp(
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

/// SiTU: `beta * tanh(g / beta) * sigmoid(g)`, with an optional tanh soft-cap
/// on the up half. Not a SwiGLU variant -- the tanh saturates far enough out
/// that a bf16 intermediate loses the distinction the gate exists to make.
template <class T>
__global__ void situ(
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

/// GeLU(tanh) gate. `c = sqrt(2/pi)`; the cubic coefficient is the canonical
/// 0.044715 that `torch.nn.functional.gelu(approximate="tanh")` uses, which
/// is HF's `gelu_pytorch_tanh`.
template <class T>
__global__ void geglu_tanh(
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

/// `y = max(x, 0)^2`.
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

/// `x *= sigmoid(gate)`, in place on `x`. The gate is read-only, which is
/// what lets the row state `in_place` on operand 0 alone.
template <class T>
__global__ void sigmoid_gate_inplace(
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

// ---------------------------------------------------------------------------
// The row-major forms: one block row per token, `ElementwiseRows`.
// ---------------------------------------------------------------------------

/// Strided `gpt_oss_glu`. Marlin writes gate/up at the PADDED intermediate
/// width, because the packed expert weights are aligned to 128, while the
/// activation the down projection consumes is the unpadded one -- so this
/// reads at one stride and writes at another instead of forcing a separate
/// compaction pass.
///
/// `limit <= 0` means no clamp, which is how a caller asks for the plain GLU
/// through the strided entry point.
template <class T>
__global__ void gpt_oss_glu_strided(
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

/// The packed row's two halves, in the order the checkpoint wrote them.
///
/// `[gate | up]` or `[up | gate]`, `I` wide each. Every chunked activation
/// below reads its pair through this, so the layout question is answered once
/// and the four bodies say what they compute.
template <bool GateSecond>
__device__ __forceinline__ i32 gate_offset(i32 i, i32 I) {
    return GateSecond ? I + i : i;
}

template <bool GateSecond>
__device__ __forceinline__ i32 up_offset(i32 i, i32 I) {
    return GateSecond ? i : I + i;
}

template <class T, bool GateSecond>
__device__ __forceinline__ void chunked_swiglu_body(
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

/// SwiGLU over a packed `[N, 2I]` activation, gate half FIRST.
template <class T>
__global__ void chunked_swiglu(const T* __restrict__ packed, T* __restrict__ y, i32 I) {
    chunked_swiglu_body<T, false>(packed, y, I);
}

/// SwiGLU over a packed `[N, 2I]` activation, gate half SECOND.
template <class T>
__global__ void chunked_swiglu_gate_second(
    const T* __restrict__ packed, T* __restrict__ y, i32 I)
{
    chunked_swiglu_body<T, true>(packed, y, I);
}

template <class T, bool GateSecond>
__device__ __forceinline__ void chunked_situ_body(
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

/// SiTU over a packed `[N, 2I]` activation, gate half FIRST.
template <class T>
__global__ void chunked_situ(
    const T* __restrict__ packed, T* __restrict__ y,
    i32 I, float beta, float linear_beta)
{
    chunked_situ_body<T, false>(packed, y, I, beta, linear_beta);
}

/// SiTU over a packed `[N, 2I]` activation, gate half SECOND.
template <class T>
__global__ void chunked_situ_gate_second(
    const T* __restrict__ packed, T* __restrict__ y,
    i32 I, float beta, float linear_beta)
{
    chunked_situ_body<T, true>(packed, y, I, beta, linear_beta);
}

/// Same chunk layout, the GELU-tanh activation Gemma-4 uses on its dense MLP
/// and (for 26B-A4B) on its routed-expert block.
template <class T, bool GateSecond>
__device__ __forceinline__ void chunked_geglu_tanh_body(
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
    constexpr float kAlpha = 0.7978845608028654f;  // sqrt(2/pi)
    constexpr float kBeta = 0.044715f;
    const float inner = kAlpha * (g + kBeta * g * g * g);
    const float gelu = 0.5f * g * (1.f + tanhf(inner));
    y[static_cast<long long>(n) * I + i] = Elem<T>::from_f32(gelu * u);
}

/// GeGLU-tanh over a packed `[N, 2I]` activation, gate half FIRST.
template <class T>
__global__ void chunked_geglu_tanh(
    const T* __restrict__ packed, T* __restrict__ y, i32 I)
{
    chunked_geglu_tanh_body<T, false>(packed, y, I);
}

/// GeGLU-tanh over a packed `[N, 2I]` activation, gate half SECOND.
template <class T>
__global__ void chunked_geglu_tanh_gate_second(
    const T* __restrict__ packed, T* __restrict__ y, i32 I)
{
    chunked_geglu_tanh_body<T, true>(packed, y, I);
}

/// Clamped SwiGLU over a packed `[N, 2I]` activation. Gate half first only:
/// the one caller exports it that way, and a second name with no caller is
/// vocabulary that reads like a choice.
template <class T>
__global__ void chunked_swiglu_clamp(
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

/// gpt-oss's GLU over a packed `[N, 2I]` activation. Gate half first only,
/// for `chunked_swiglu_clamp`'s reason: the one caller exports it that way,
/// and a second name with no caller is vocabulary that reads like a choice.
///
/// THE SAME ARITHMETIC AS `gpt_oss_glu`, SPELLED THE SAME WAY. The flat form
/// above takes gate and up as two planes; this one takes the single row the
/// text states and cuts it in half, which is the whole difference between
/// them. Every line is copied rather than tidied -- `expf` and not `__expf`,
/// `(u + 1.f) * glu` and not `glu * (u + 1.f)` -- because the point of a
/// second entry into one activation is that the two agree BIT FOR BIT and
/// `tests/swiglu_clamp_alpha.rs` says so.
///
/// The transcription is the discipline and not the test's reach: at bf16 the
/// intrinsic and the libm call round to the same eight mantissa bits, so
/// swapping them is invisible and the test would pass. What it does catch is
/// every difference that survives the narrowing -- a symmetric clamp on the
/// gate, a dropped `alpha`, a swapped half. Keeping the spelling identical
/// is what makes those the only ways the two can drift.
///
/// NO `y_fp16`. The flat form's second output feeds the MXFP4 down-projection
/// GEMV; a packed statement declares ONE result, so there is no slot for it
/// and no caller that wants one.
template <class T>
__global__ void chunked_gpt_oss_glu(
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

/// SwiGLU over a packed activation whose ROW is wider than `2I` -- the GEMM
/// wrote it at a padded stride and only the leading `2I` of each row is the
/// projection. `row_stride` is layout, not geometry, which is why it survived
/// the move and `N` did not.
template <class T>
__global__ void chunked_swiglu_strided(
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

// ---------------------------------------------------------------------------
// The vectorised pair: two elements per thread, and NO ROW.
//
// A half-width grid is a geometry no `LaunchRule` states, and the choice
// between these and their scalar twins is a run-time test on `I` and on the
// parity of `row_stride` -- a predicate over an operand's VALUE, which a
// `Source` cannot express either. They stay device text in the device header
// and stay launched from `swiglu.cu`. See this file's header.
// ---------------------------------------------------------------------------

template <class T, bool GateSecond>
__device__ __forceinline__ void chunked_swiglu_vec2_body(
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
__global__ void chunked_swiglu_vec2(
    const T* __restrict__ packed, T* __restrict__ y, i32 N, i32 I)
{
    chunked_swiglu_vec2_body<T, false>(packed, y, N, I);
}

template <class T>
__global__ void chunked_swiglu_vec2_gate_second(
    const T* __restrict__ packed, T* __restrict__ y, i32 N, i32 I)
{
    chunked_swiglu_vec2_body<T, true>(packed, y, N, I);
}

template <class T>
__global__ void chunked_swiglu_strided_vec2(
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

// ---------------------------------------------------------------------------
// The gated residual adds.
// ---------------------------------------------------------------------------

/// `out = sum + x * sigmoid(scalar_gate[n * stride])` -- one gate value per
/// row, broadcast across the row. `moe.sigmoid_gate_add`'s device text.
///
/// OUT OF PLACE, because the point is. It read `out` as its own left addend
/// while it had no caller at all; the declaration states `routed`, `shared`
/// and a separate `y`, which is what the three shader planes' combine takes,
/// and the sum a text hands over is generally live somewhere else.
///
/// `stride` is how far apart two rows' gate values are: 1 when the gate is a
/// dense `[N]` column, and something else when it is one column of a wider
/// tensor. The unfused `sigmoid_scalar_gate_bf16` that used to sit beside
/// this one is gone -- the add always followed the gate, and a launcher with
/// no caller is vocabulary that reads like a choice.
template <class T>
__global__ void sigmoid_scalar_gate_add(
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

/// `out += y * sigmoid(x . gate_w)` -- the shared expert's landing, with the
/// gate computed here rather than in a launch of its own.
///
/// One `T` per thread makes every warp issue a 64-byte access, half a cache
/// line, and this kernel is pure bandwidth: it streams `x`, `y` and `out`.
/// Moving a `uint4` per thread, and reducing the dot product through warp
/// shuffles rather than a `__syncthreads` tree, is the whole optimisation --
/// which is why the alignment test below is in the kernel and not a host-side
/// choice between two kernels. `LaunchRule::Rms` is this launcher exactly:
/// one block per row, 256 threads, and `(256 / 32) * sizeof(float)` bytes of
/// dynamic shared memory for the cross-warp fold.
template <class T>
__global__ void sigmoid_dot_scalar_gate_add(
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

    // `usize`, not `std::uintptr_t`: NVRTC ships no `<cstdint>`, and the
    // prelude's pointer-width integer is the same width by construction.
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
    // The fold order is the original's, warp by warp. `block_sum` folds to
    // thread 0 and this needs the sigmoid of the total broadcast back, so the
    // second level stays written out.
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

}  // namespace pie::mlp
