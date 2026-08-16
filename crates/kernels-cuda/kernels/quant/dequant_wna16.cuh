//===-- dequant_wna16.cuh - the W4A16 decoders, as device text -----------===//
//
// Four `__global__`s, the PRMT unpacker they share and the activation
// shuffle that pairs a `float4` load into the nibble order INT4B8 stores in.
// `dequant_wna16.cu` is the four host launchers; this file is what the GPU
// runs. There is exactly one definition of each, because two copies that
// agree today drift tomorrow and `norm/altup_aux` shipped a release proving
// it.
//
// # One row of four, and exactly what blocks the other three
//
// `dequant_wna16_int4b8` is rowed -- its axes were transposed so that
// `LaunchRule::ElementwiseRows` states the grid exactly, which is the note
// above the kernel. The other three are device text the JIT can compile and
// nothing it can fire, because no launch geometry any of them states is a
// rule this tree has:
//
//   * `wna16_gate_up_decode` and `wna16_down_decode` launch a 2-D grid over
//     (route, output-row slab) with one WARP per output row. `RouteRows` is
//     one block per row and picks the block width from the row width, which
//     is a different decomposition, not a different spelling of this one.
//     Both are also plain `__global__`s and stay that way: the activation is
//     fp16 while the output is bf16 and the scale pointers are `const void*
//     const*`, so there is no ONE type a row could name -- the two formats
//     are the kernel's whole reason for existing (see the `__hfma2` note
//     below) and a `T` over either would be a parameter half the signature
//     ignores.
//   * `bf16_to_fp16` is a capped grid-stride over VECTOR UNITS --
//     `min(ceil((n / 8) / 256), 1024)` blocks -- and `Elementwise` opens
//     `ceil(n / 256)` over elements, which is eight times the grid and no
//     cap. Both loops below are guarded, so the wrong grid would still
//     answer; that is what makes stating it a bad row rather than a broken
//     one. It is the same shape `quant_bf16_to_fp8.cuh`'s `absmax_bf16` has
//     and the same reason no rule states it. **The naming half was never a
//     blocker at all, and finding that out is the point of the note on
//     `Narrow2<f16>` below:** `kernels-cuda`'s NVRTC shim aliases
//     `__half` to `f16`, so under the compiler the JIT actually uses,
//     `bf16_to_narrow<__half>` already IS an instantiation a table can spell.
//     Templating it was never the hard half, and here it was not a half.
//
// `new-horizon.md` §10 forbids inventing a rule to fit a kernel, so the
// kernels are stated here and the rows are not. The split is still worth
// having: the device text has exactly one home, the file compiles under
// NVRTC, and the day a slab rule exists the rows are a table edit.
//
// # The packed-half arithmetic, and why the include is guarded
//
// `__hfma2` and `__hsub2` are hardware instructions, not arithmetic
// `pie_device.cuh` could restate. §10.5 lists them as the reason this file
// was BLOCKED and §8 refused to rewrite the bodies for a migration; §15
// closed it from the other side. `pie_half2.cuh` is the honest name for the
// packed-half shim -- measured bit-identical to nvcc over 32,945,058
// comparisons, including the `__hfma2` fallback that a 2Sum and a
// round-to-odd had to be written for -- so the intrinsics stay spelled the
// way they were and only the include is chosen per compiler. The
// ahead-of-time build reaches this directory with `-iquote` rather than an
// include directory, which answers `#include "..."` and is never searched
// for `#include <...>`, so the shim beside this file cannot shadow NVIDIA's
// real `cuda_fp16.h` for every translation unit in the tree.
//
// # What is NOT here
//
// `<cuda_bf16.h>` is gone, and with it `__nv_bfloat162`, `__bfloat1622float2`
// and `__bfloat162float`. The prelude already carries that trio under its own
// names -- `bf16x2` is four bytes aligned to four with members `x` then `y`,
// deliberately the same SHAPE, and `bf16x2_to_f32` is the same pair of exact
// widenings -- so the staging kernel reads through the prelude and the whole
// bf16 header stops being a dependency. That is a rename, not a rewrite:
// widening bf16 to fp32 drops no bits in either spelling.
//
// `<algorithm>` is gone too. It was there for `std::min` and `std::max` in
// the launcher, which is host code and stayed in the `.cu`; NVRTC ships no
// C++ standard library -- 0 of 31 standard headers answered when it was
// measured -- so a device header naming one compiles nowhere.
//
// # Every `__global__` here is a template, and that is a linkage requirement
//
// nvcc 13.0 gives a non-template `__global__` defined in a header EXTERNAL
// linkage, for both the device function and its generated
// `__device_stub__`, in every translation unit that includes it. A second
// includer is therefore a hard *"multiple definition"* at link **even when it
// never launches the kernel** -- measured on this file with two TUs that do
// nothing but `#include` it: four collisions, the function and the stub for
// each of the two decoders. That made this header single-includer property,
// not by design but by accident, and it is what stopped `norm/rmsnorm.cu`
// from including it to fire `bf16_to_narrow` directly.
//
// `wna16_gate_up_decode` and `wna16_down_decode` therefore carry a defaulted
// non-type parameter `Tu`. It changes no instruction and no call site: the
// instantiation drops to INTERNAL linkage -- `nm` reports `t`, not a weak
// `W` -- so each includer gets a private copy, and an includer that does not
// launch emits no copy at all. Keep it that way. A `__global__` added here
// without a template parameter re-imposes the one-includer limit on the whole
// file, and the error it produces names the linker, not this decision.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "prelude/device.cuh"

#ifdef __CUDACC_RTC__
#include "prelude/half2.cuh"
#else
#include <cuda_fp16.h>
#endif

namespace pie::quant {


// The block width these kernels are launched at is NOT here. Every one of
// them reads `blockDim.x` or a warp lane, so 256 is the launcher's decision
// and it stays in `dequant_wna16.cu` beside the `<<<>>>` that spends it. A
// copy here would be a second place for the same number, which is the whole
// failure this split exists to remove -- and NVRTC said so directly:
// `variable "DECODE_BLOCK" was declared but never referenced`.

// The ROW is `blockIdx.x` and the packed-word column is `blockIdx.y`, which
// is the transpose of what this kernel launched at before the JIT. The old
// order put columns on x and rode a `for (row = blockIdx.y; row < out_dim;
// row += gridDim.y)` stride loop, because gridDim.y caps at 65535 and an
// expert-stacked weight easily has more rows than that. Swapping the axes
// retires the cap outright -- gridDim.x is 2^31-1, so a row per block is
// exact -- and it is what makes `LaunchRule::ElementwiseRows` (grid
// `[rows, ceil(width / 256)]`, block 256) the rule this geometry already is.
//
// The y extent the rule hands over is `ceil(in_dim / 256)` blocks of 256,
// which covers `in_dim` thread-columns where only `in_dim / 8` WORD-columns
// exist: eight times oversubscribed, and the guard below discards the excess.
// Coverage is what matters and coverage holds; the waste is one predicated
// exit on a kernel that runs once per weight at load.
template <class T>
__global__ void dequant_wna16_int4b8(
    const i32* __restrict__ packed,
    const T* __restrict__ scale,
    T* __restrict__ out,
    int in_dim,
    int group_size)
{
    const int row = blockIdx.x;
    const int word_col = blockIdx.y * blockDim.x + threadIdx.x;
    const int words_per_row = in_dim / 8;
    if (word_col >= words_per_row) return;

    const int word = packed[static_cast<long long>(row) * words_per_row + word_col];
    const int k_base = word_col * 8;
    T* row_out = out + static_cast<long long>(row) * in_dim;
    const T* row_scale =
        scale + static_cast<long long>(row) * (in_dim / group_size);

#pragma unroll
    for (int lane = 0; lane < 8; ++lane) {
        const int k = k_base + lane;
        const int nibble = (word >> (lane * 4)) & 0xF;
        const float q = static_cast<float>(nibble - 8);
        const float s = Elem<T>::to_f32(row_scale[k / group_size]);
        row_out[k] = Elem<T>::from_f32(q * s);
    }
}

__device__ __forceinline__ float wna16_load_int4b8(
    const i32* __restrict__ packed,
    const bf16* __restrict__ scale,
    int row,
    int col,
    int in_dim,
    int group_size)
{
    const int words_per_row = in_dim / 8;
    const int word =
        packed[static_cast<long long>(row) * words_per_row + col / 8];
    const int nibble = (word >> ((col & 7) * 4)) & 0xF;
    const float q = static_cast<float>(nibble - 8);
    const float s = bf16_to_f32(
        scale[static_cast<long long>(row) * (in_dim / group_size) +
              col / group_size]);
    return q * s;
}

// Unpacks eight INT4 weights out of one packed word into four fp16 pairs.
//
// The obvious way to turn a nibble into a float is `float(nibble - 8)`: mask,
// convert, subtract. That is three instructions for one weight, and these
// decode GEMVs are ALU-bound by a factor of four -- a loads-only version of
// `wna16_gate_up_decode` at Kimi K2.6's shapes runs at 5935 GB/s while
// the real kernel manages 1350 GB/s, so nothing matters here except the
// instruction count.
//
// So do it in the exponent instead. `0x6400` is fp16 1024.0, whose ten
// mantissa bits are zero; OR-ing a nibble into them yields fp16 `1024 + q`
// exactly, because 1024..1039 all share that exponent. The mask, shift and OR
// fold into a single LOP3, and because the layout puts nibble j and nibble j+4
// in the two halves of a 32-bit lane, one `__hsub2` against 1032.0 (= 1024 + 8)
// finishes two weights at once. Three instructions per pair, against ten.
//
// fp16 rather than bf16 for one reason: the products are then accumulated with
// `__hfma2`, two MACs per instruction and no format conversion at all, and
// fp16's 11 mantissa bits keep that accumulation honest. The bf16 version of
// exactly this kernel measures rel_l2 7.2e-3 against an fp32-accumulate
// reference -- about twice the bf16 *output's* own rounding, i.e. a real
// precision loss. The fp16 version measures 1.6e-3 for gate/up and is
// bit-exact for down, at the same speed.
__device__ __forceinline__ void wna16_unpack8_int4b8(unsigned word,
                                                     __half2 out[4]) {
    constexpr unsigned kMagic = 0x64006400u;  // fp16 1024.0, twice
    const __half2 kBias = __float2half2_rn(1032.0f);
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        unsigned t = ((word >> (j * 4)) & 0x000f000fu) | kMagic;
        out[j] = __hsub2(*reinterpret_cast<__half2*>(&t), kBias);
    }
}

// Rearranges the eight activations a packed word consumes into the four
// (k, k+4) pairs the nibble trick wants, from a single 16-byte load.
//
// The eight fp16 values arrive as four 32-bit lanes {x1x0, x3x2, x5x4, x7x6};
// the pairs needed are (x0,x4), (x1,x5), (x2,x6), (x3,x7). Each is one PRMT
// picking two bytes from each of two source lanes. The alternative -- eight
// scalar loads and four `__halves2half2` -- is what made an earlier `__hfma2`
// attempt *lose* to the fp32 version despite doing half the arithmetic.
__device__ __forceinline__ void wna16_act_pairs(const float4& xv,
                                                __half2 xp[4]) {
    const unsigned* xu = reinterpret_cast<const unsigned*>(&xv);
    unsigned p0 = __byte_perm(xu[0], xu[2], 0x5410);
    unsigned p1 = __byte_perm(xu[0], xu[2], 0x7632);
    unsigned p2 = __byte_perm(xu[1], xu[3], 0x5410);
    unsigned p3 = __byte_perm(xu[1], xu[3], 0x7632);
    xp[0] = *reinterpret_cast<__half2*>(&p0);
    xp[1] = *reinterpret_cast<__half2*>(&p1);
    xp[2] = *reinterpret_cast<__half2*>(&p2);
    xp[3] = *reinterpret_cast<__half2*>(&p3);
}

// One warp per output row for both halves of the fused gate/up projection.
//
// A block-per-row version spends eight `__syncthreads()` on the tree
// reduction for a handful of iterations of actual work; a warp reduces in
// five shuffles with no barrier at all.
//
// The activation arrives as fp16 (staged once per MoE layer by
// `bf16_to_fp16`, which costs one ~2 us launch against the ~15 us per
// layer this saves) so that the whole inner loop is `__hfma2`: two MACs per
// instruction, no format conversion anywhere. The group scale multiplies the
// word's partial sum instead of every product, which is exact for the same
// reason it is cheaper -- the scale is constant across the group -- and the
// promotion to fp32 happens once per word, so error cannot accumulate across
// the row. Measured against an fp32-accumulate reference at Kimi's decode
// shapes: gate/up 85.3 -> 69.0 us at rel_l2 1.6e-3, down 49.5 -> 38.3 us
// bit-exact.
/// **`Tu` buys LINKAGE, not a numeric axis, and it is deliberately a non-type
/// parameter so that nothing can mistake it for one.** A non-template
/// `__global__` defined in a header can be included by exactly ONE translation
/// unit: nvcc 13.0 emits the device function AND its `__device_stub__` with
/// external linkage into every includer, so a second one is
/// *"multiple definition"* at link — measured on this header with two TUs that
/// merely `#include` it and launch nothing. Making the kernel a template with
/// a defaulted parameter drops the instantiation to internal linkage (`nm`
/// says `t`, not even a weak `W`), so each includer gets a private copy and
/// there is nothing to collide. The default is what makes it free: `Tu` is not
/// deducible from any parameter, so every existing `<<<>>>` selects `Tu = 0`
/// un-edited. **Deduction would beat the default** — had the axis been spelled
/// as a `T` in the parameter list, a call site passing `uint16_t*` would
/// silently instantiate `T = unsigned short` — which is the second reason this
/// is an `int`.
///
/// It is NOT an element type and this kernel does not have one. The activation
/// is fp16 and the output is bf16, and the inner loop is `__hfma2` against the
/// measurements above; retyping either end is a body change with its own
/// parity burden, not a retype, so it is not done here.
template <int Tu = 0>
__global__ void wna16_gate_up_decode(
    const __half* __restrict__ act,
    const i32* __restrict__ topk_idx,
    const i32* const* __restrict__ gate_packed_ptrs,
    const void* const* __restrict__ gate_scale_ptrs,
    const i32* const* __restrict__ up_packed_ptrs,
    const void* const* __restrict__ up_scale_ptrs,
    bf16* __restrict__ gate_out,
    bf16* __restrict__ up_out,
    int top_k,
    int hidden,
    int intermediate,
    int group_size)
{
    const int route = blockIdx.x;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int row = blockIdx.y * (blockDim.x >> 5) + warp_in_block;
    if (row >= intermediate) return;
    const int token = route / top_k;
    const int expert = topk_idx[route];

    const auto* gate_packed = gate_packed_ptrs[expert];
    const auto* gate_scale =
        static_cast<const bf16*>(gate_scale_ptrs[expert]);
    const auto* up_packed = up_packed_ptrs[expert];
    const auto* up_scale =
        static_cast<const bf16*>(up_scale_ptrs[expert]);

    float gate_acc = 0.f;
    float up_acc = 0.f;
    const int words_per_row = hidden / 8;
    const int words_per_group = group_size / 8;
    const long long row_base = static_cast<long long>(row) * words_per_row;
    const long long scale_base =
        static_cast<long long>(row) * (hidden / group_size);
    const float4* x4 = reinterpret_cast<const float4*>(
        act + static_cast<long long>(token) * hidden);
    for (int word_col = lane_id; word_col < words_per_row; word_col += 32) {
        const auto gate_word = static_cast<unsigned>(gate_packed[row_base + word_col]);
        const auto up_word = static_cast<unsigned>(up_packed[row_base + word_col]);
        __half2 xp[4], gate_q[4], up_q[4];
        wna16_act_pairs(x4[word_col], xp);
        wna16_unpack8_int4b8(gate_word, gate_q);
        wna16_unpack8_int4b8(up_word, up_q);
        __half2 gate_word_sum = __float2half2_rn(0.f);
        __half2 up_word_sum = gate_word_sum;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            gate_word_sum = __hfma2(gate_q[j], xp[j], gate_word_sum);
            up_word_sum = __hfma2(up_q[j], xp[j], up_word_sum);
        }
        const int group = word_col / words_per_group;
        const float2 gf = __half22float2(gate_word_sum);
        const float2 uf = __half22float2(up_word_sum);
        gate_acc = fmaf(gf.x + gf.y,
                        bf16_to_f32(gate_scale[scale_base + group]),
                        gate_acc);
        up_acc = fmaf(uf.x + uf.y,
                      bf16_to_f32(up_scale[scale_base + group]), up_acc);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        gate_acc += __shfl_xor_sync(0xffffffffu, gate_acc, off);
        up_acc += __shfl_xor_sync(0xffffffffu, up_acc, off);
    }
    if (lane_id == 0) {
        const long long out_idx =
            static_cast<long long>(route) * intermediate + row;
        gate_out[out_idx] = f32_to_bf16(gate_acc);
        up_out[out_idx] = f32_to_bf16(up_acc);
    }
}

/// `Tu` is the linkage parameter documented on `wna16_gate_up_decode` — one
/// private copy per includer, `Tu = 0` selected by default at every un-edited
/// call site, and an `int` rather than a type so it cannot be read as an
/// element axis this kernel does not have.
template <int Tu = 0>
__global__ void wna16_down_decode(
    const __half* __restrict__ act,
    const i32* __restrict__ topk_idx,
    const i32* const* __restrict__ down_packed_ptrs,
    const void* const* __restrict__ down_scale_ptrs,
    bf16* __restrict__ out,
    int top_k,
    int hidden,
    int intermediate,
    int group_size)
{
    const int route = blockIdx.y;
    const int warp_in_block = threadIdx.x >> 5;
    const int lane_id = threadIdx.x & 31;
    const int h = blockIdx.x * (blockDim.x >> 5) + warp_in_block;
    if (h >= hidden) return;
    const int expert = topk_idx[route];
    const auto* down_packed = down_packed_ptrs[expert];
    const auto* down_scale =
        static_cast<const bf16*>(down_scale_ptrs[expert]);

    float acc = 0.f;
    const int words_per_row = intermediate / 8;
    const int words_per_group = group_size / 8;
    const long long row_base = static_cast<long long>(h) * words_per_row;
    const long long scale_base =
        static_cast<long long>(h) * (intermediate / group_size);
    const float4* x4 = reinterpret_cast<const float4*>(
        act + static_cast<long long>(route) * intermediate);
    for (int word_col = lane_id; word_col < words_per_row; word_col += 32) {
        const auto word = static_cast<unsigned>(down_packed[row_base + word_col]);
        __half2 xp[4], q[4];
        wna16_act_pairs(x4[word_col], xp);
        wna16_unpack8_int4b8(word, q);
        __half2 word_sum = __float2half2_rn(0.f);
#pragma unroll
        for (int j = 0; j < 4; ++j) word_sum = __hfma2(q[j], xp[j], word_sum);
        const float2 wf = __half22float2(word_sum);
        acc = fmaf(wf.x + wf.y,
                   bf16_to_f32(
                       down_scale[scale_base + word_col / words_per_group]),
                   acc);
    }
#pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_xor_sync(0xffffffffu, acc, off);
    }
    if (lane_id == 0) {
        out[static_cast<long long>(route) * hidden + h] = f32_to_bf16(acc);
    }
}



// Stages a bf16 activation as fp16 for the W4A16 decode GEMVs.
//
// Those kernels want fp16 so their whole inner loop can be `__hfma2`. Doing
// the conversion inside them instead would cost four instructions per pair on
// every one of the ~2000 rows that read the same vector; doing it once here
// costs one pass over a few kilobytes plus a ~2 us launch, against ~15 us per
// layer saved. `float4`-wide because the conversion is pure bandwidth.
//
// Templated on the DESTINATION so a row can name it -- `DeviceKernel::
// instantiation` spells `path<elem>` and has no way to state a plain
// `__global__`. `Narrow2<T>` is what keeps the vector path across that
// change: the float4 body packs two elements at a time, and the packing
// instruction is per-type, so a naive template over `Elem<T>` would have
// silently demoted this to the scalar loop it exists to avoid.
//
// `n_vec8` is computed here rather than passed. The launcher's `n / 8` was
// the same integer division, and an operand a rule cannot recover is an
// operand a row must state -- so the one that was pure arithmetic on `n`
// moved into the kernel and the row's operand list became the ahead-of-time
// table's, minus the stream.
//
// # Two hooks, because the kernel has two loops
//
// `Narrow2<T>` states both halves of a destination format: `pack` for the
// `float4` body and `narrow` for the ragged tail. The tail called
// `Elem<T>::from_f32` before this commit, and that DID NOT COMPILE at the
// only instantiation the ahead-of-time build asks for -- `Elem` is
// specialised on the prelude's `bf16` and `f16` structs and `__half` is
// neither, so nvcc reported *"incomplete type Elem<__half> is not allowed"*
// and `dequant_wna16.cu` was the one translation unit in the tree that no
// longer built. Routing the tail through `Narrow2` too is what keeps a
// destination format described in ONE place: a format `Narrow2` knows is a
// format the whole kernel knows.
template <class T>
struct Narrow2;

template <>
struct Narrow2<__half> {
    using pair = __half2;
    static __device__ __forceinline__ pair pack(float lo, float hi) {
        return __floats2half2_rn(lo, hi);
    }
    static __device__ __forceinline__ __half narrow(float v) {
        return __float2half(v);
    }
};

template <>
struct Narrow2<bf16> {
    using pair = bf16x2;
    static __device__ __forceinline__ pair pack(float lo, float hi) {
        return ::pie::f32_to_bf16x2(lo, hi);
    }
    static __device__ __forceinline__ bf16 narrow(float v) {
        return ::pie::f32_to_bf16(v);
    }
};

/// The fp16 destination for the compiler that has a REAL `__half`, and the KEY
/// TYPE is the finding.
///
/// Under NVRTC there is nothing to add: `kernels-cuda/shim/cuda_fp16.h`
/// opens with `using __half = ::pie::f16;`, so
/// `Narrow2<__half>` above IS `Narrow2<f16>` — one type, one
/// specialisation, and a second one is a REDEFINITION that fails to compile.
/// The `units` test caught exactly that: *"class
/// `Narrow2<__half>` has already been defined"*. The naming blocker this
/// specialisation was written to remove had therefore already been removed by
/// the shim, and the row `quant::bf16_to_fp16` still lacks is blocked by its
/// capped grid-stride launcher and by nothing else.
///
/// Under nvcc `__half` is NVIDIA's type and `f16` is the prelude's struct, and
/// they are distinct. `dequant_wna16.cu` instantiates only `__half`, so this
/// twin is never emitted — it exists so the SAME TEXT is instantiable at the
/// same names under both compilers. A header where an instantiation compiles
/// under one and not the other is a header whose JIT and ahead-of-time halves
/// have quietly diverged, which is the whole failure this split exists to make
/// impossible.
///
/// **It forwards to the same two instructions the `__half` specialisation
/// does, deliberately.** The obvious body — `Elem<f16>::from_f32` and a
/// software pack — would be a DIFFERENT ANSWER, not a slower one:
/// `pie_device.cuh`'s `f32_to_f16` flushes fp16 subnormals to zero, which
/// `__float2half` does not, so every bf16 input below 2^-14 would narrow to
/// zero here and to a subnormal in the specialisation above. That is a
/// divergence confined to one exponent — the shape of defect the prelude's own
/// `f16_to_f32` header records being caught only by an exhaustive sweep — so
/// this does not go near it. `pair` is `__half2` and not an `f16` pair for the
/// same reason: it is a four-byte staging type inside a `float4`, never the
/// store's type, so the instruction that fills it is free to be the hardware
/// one.
///
/// **The preprocessor cannot answer this and two attempts proved it.** The
/// question is whether `__half` is ALREADY a name for `f16`; asking
/// `__CUDACC_RTC__` instead asks which COMPILER is running, a proxy that holds
/// in both configurations the build ships — NVRTC always reads the shim, nvcc
/// ahead-of-time never does — and breaks in the one that CHECKS them. An
/// offline operand typecheck compiled this header under nvcc with `-I` aimed
/// at `kernels-cuda/csrc/src`, where the shim lives: line 84 spells
/// `#include <cuda_fp16.h>` in ANGLE brackets, the shim answered it, `__half`
/// became `f16`, `__CUDACC_RTC__` stayed undefined, and this twin landed on
/// top of the one at 405 — *"class `Narrow2<__half>` has already been
/// defined"*. Testing the shim's own `PIE_FP16_HAS_SM80` was tried next and
/// fails too: `cuda_fp16.h:745-746` `#undef`s it and its SM53 sibling on the
/// last two lines, deliberately, so that nothing leaks into the translation
/// unit. **No macro survives the include to be tested.**
///
/// So the key is a TYPE. `PickF16<f16, __half>` selects its partial
/// specialisation only when both arguments are one type, and that is the exact
/// question — decided by the compiler's own type identity rather than by a
/// marker some other file has to remember to keep defining. Where they are
/// distinct this specialises `Narrow2<f16>` as before; where the shim has
/// merged them it specialises `Narrow2<Inert>`, a private type nothing names,
/// nothing instantiates and no `T` can reach. `Inert` carries `f16`'s single
/// `unsigned short` so that `narrow`'s aggregate initialiser is well-formed in
/// that build too: an explicit specialisation is not a template, so its member
/// bodies are compiled whether or not anything ever calls them.
namespace half_key {

/// Stands in for `f16` in the builds where `Narrow2<f16>` would be a
/// redefinition. Mirrors `f16`'s layout only so the body below compiles.
struct Inert {
    unsigned short raw;
};

template <class A, class B>
struct PickF16 {
    using type = f16;
};

template <class A>
struct PickF16<A, A> {
    using type = Inert;
};

}  // namespace half_key

/// `f16` where nvcc's `__half` is a type of its own; `half_key::Inert` where
/// the shim has made `__half` another spelling of `f16`.
using f16_or_inert = typename half_key::PickF16<f16, __half>::type;

template <>
struct Narrow2<f16_or_inert> {
    using pair = __half2;
    static __device__ __forceinline__ pair pack(float lo, float hi) {
        return __floats2half2_rn(lo, hi);
    }
    static __device__ __forceinline__ f16_or_inert narrow(float v) {
        return f16_or_inert{__half_as_ushort(__float2half(v))};
    }
};

template <class T>
__global__ void bf16_to_narrow(const bf16* __restrict__ in,
                               T* __restrict__ out,
                               long long n) {
    using pair = typename Narrow2<T>::pair;
    const long long n_vec8 = n / 8;
    const long long stride = (long long)gridDim.x * blockDim.x;
    for (long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
         i < n_vec8; i += stride) {
        const float4 v = reinterpret_cast<const float4*>(in)[i];
        const bf16x2* src = reinterpret_cast<const bf16x2*>(&v);
        float4 o;
        pair* dst = reinterpret_cast<pair*>(&o);
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const float2 f = bf16x2_to_f32(src[j]);
            dst[j] = Narrow2<T>::pack(f.x, f.y);
        }
        reinterpret_cast<float4*>(out)[i] = o;
    }
    for (long long i = n_vec8 * 8 + (long long)blockIdx.x * blockDim.x +
                       threadIdx.x;
         i < n; i += stride) {
        out[i] = Narrow2<T>::narrow(bf16_to_f32(in[i]));
    }
}

}  // namespace pie::quant
