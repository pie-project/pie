//===-- rope.cuh - the rotary family's device text ---------------------===//
//
// Ten `__global__`s and one `__device__` helper: every line of `rope` that
// runs on the GPU, and nothing that runs on the host. `rope.cu` includes this
// file and keeps only its launchers, so there is exactly ONE definition of
// each kernel in the tree -- which is the point. `norm/altup_aux` shipped a
// release with two, a `.cu` copy and a `.cuh` copy that agreed on the day
// they were written; each was correct for whichever half of the tests
// exercised it, and neither was correct after the other was edited.
//
// # Two compilers, one text
//
// nvcc reaches this file through the `-I csrc/src` the CMake build already
// has. NVRTC has no include path at all: it resolves `#include` against a
// header set carried in the Rust binary (`kernels-cuda`'s `build.rs`
// walks this tree and `source::LIBRARY` is the result), so the three
// directives below resolve BY NAME out of the binary and a machine with no
// CUDA toolkit still compiles this file. `norm/altup_aux.cuh` states the rule
// this follows: no include path on disk, includes resolve against a carried
// set or they do not resolve at all.
//
// The practical consequence is that nothing here may reach for the C++
// standard library. NVRTC ships none -- the `stdlib_probe` measured 0 of 31
// standard headers answering -- so `<cstdint>`, `<cmath>` and
// `<cuda_bf16.h>` are all `pie_device.cuh`'s job, and `rope.cu` keeps the
// host-side `<cuda_runtime.h>` to itself.
//
// # Why the kernels left the anonymous namespace
//
// They had to be nameable. `nvrtcAddNameExpression` takes an instantiation as
// a STRING -- `::pie::rope::rotate_partial<...>` -- and internal linkage has
// no such name. So the device text lives in `pie::rope`, and the prelude's
// scalar names reach it unqualified because `pie` encloses `pie::rope`:
// `bf16` here is `pie::bf16` with nothing written to make it so.
//
// # Which kernels are templates, and why only those
//
// Two: `standard_table` over its POSITION type and `rotate_partial` over its
// ELEMENT type. Both are templates because a row instantiates them, and a
// second numeric format then costs a row rather than a translation unit --
// `norm_device.rs` records that measurement for `residual_add_f16`.
//
// The other eight stay bf16. Not an oversight and not laziness: six of them
// rotate through `rope_device.cuh`'s `rotate_pair`, which takes `bf16*` and
// is a SHARED header this migration may not edit, so a `T`-templated caller
// could not be instantiated at anything else. Templating them would produce
// eight templates nothing instantiates, which is text that has never been
// compiled pretending to be a capability. When `rotate_pair` grows an element
// parameter, the templates and the rows follow in one commit.
//
// # What is NOT here
//
// The `<<<>>>`s, the `if (n <= 0) return;` guards and the host-side YaRN ramp
// arithmetic -- all in `rope.cu`, because a launch rule is the ROW's
// statement and a guard is the rule's (`LaunchRule::eval` refuses an empty
// extent, so `Elementwise` over zero rows declines rather than launches).
//
// EIGHT OF THESE TEN KERNELS ONCE HAD NO ROW FOR EXACTLY THAT REASON, and
// that sentence is now wrong in the good direction: all ten are named by a
// row, and `LaunchRule::Rope` -- ported from `rope.cu:82-95`, two-dimensional
// `(token, head)` grid and dynamic shared allocation together -- is what
// states the three that looked unstateable. One row is `LaunchRule::Unstated`
// and says so (`qk_rmsnorm_rotate_rounded`, whose grid is the FIRE's head
// count and not the STATEMENT's), and six of the twelve rows are unsourced,
// which is a different lack: they name the kernel and cannot yet fill its
// arguments. A row is not a routing; `kernels-cuda/src/families/rope.rs`
// carries the table of which is which, and every launcher in `rope.cu` is
// still called.
//
// The host arithmetic above the `<<<>>>` is what most of the unsourcing costs:
// `heads_per_block`, `cache_pairs` and the two YaRN ramps are computed by the
// launcher from `head_dim` and the config, and arrive as ARGUMENTS of these
// kernels. Inventing a `Source` for them would be a value nothing checks, in
// the same way inventing a grid would be.
//
//===----------------------------------------------------------------------===//
#pragma once

// The scalar layer and the fixed-width integer names. What used to be
// `<cuda_bf16.h>` and `<cstdint>`.
#include "prelude/device.cuh"

// `kv_slot_for_token` / `kv_dst_index`, shared with the KV-append family so
// the fused write lands where an unfused one would.
#include "prelude/kv_paged_addr.cuh"

// `rotate_pair`, `rotate_pair_interleaved` and `yarn_original_freq`, shared
// with the fused MLA-prepare kernel so both rotate through byte-identical
// code.
#include "prelude/rope.cuh"

namespace pie::rope {

// The scalar layer is the PRELUDE's, not this family's. Named here so the
// kernels below read as they always did, so a row may keep spelling its
// element type `bf16`, and so `rope.cu`'s launchers -- which sit in
// the enclosing `rope` and write `bf16` meaning the prelude's -- go
// on resolving to the same type through these declarations.
template <class T>
using Elem = ::pie::Elem<T>;

// `positions` is the template parameter and not `i32`, because a row
// instantiates this and a row names ONE type. Today that type is `i32`;
// `graph_pad` writes its pad lanes' positions as `u32`, and the day a caller
// wants the table built straight off those, it costs a row.

// One block per token; threads cover the full QK head_dim grid:
// (head, dim_pair_idx). For Qwen the convention pairs index `i` with
// `i + head_dim/2`, with frequency theta^(-2*i / head_dim).
// `rotate_pair` / `rotate_pair_interleaved` live in rope_device.cuh so the
// fused MLA-prepare kernel rotates through byte-identical code.

template <class P>
__global__ void standard_table(
    const P* __restrict__ positions,
    float* __restrict__ table,
    int head_dim,
    float theta)
{
    const int n = blockIdx.x;
    const int half = head_dim / 2;
    const int pos = static_cast<int>(positions[n]);
    float* row = table + static_cast<long long>(n) * head_dim;
    for (int dim_pair = threadIdx.x; dim_pair < half; dim_pair += blockDim.x) {
        const float freq = powf(theta,
            -2.f * static_cast<float>(dim_pair) / static_cast<float>(head_dim));
        const float ang = static_cast<float>(pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);
        row[dim_pair] = cos_v;
        row[dim_pair + half] = sin_v;
    }
}

// `kWriteKv` lands the rotated K straight in the paged cache, and copies V
// alongside it, instead of leaving both to a following write_kv launch.
//
// The pair is worth fusing because write_kv is one block per current-step
// token: at decode that is a SINGLE block on 148 SMs writing 2 KB, which
// measured 3.69 us -- nearly all of it launch plus one dependent round trip.
// RoPE has already split the heads across blockIdx.y for exactly that reason,
// and its kv-head blocks hold the rotated K in registers at the moment
// write_kv would re-read it from memory. Folding the store in costs those
// blocks a page-index lookup and gives back a launch, a store and a load of K.
//
// V is not rotated; the same threads copy it because they are already
// resolved to the right (page, offset) and each covers dim_pair and its
// partner, which together span head_dim.
template <bool kWriteKv, bool kHnd>
__global__ void rotate(
    bf16* __restrict__ q,
    bf16* __restrict__ k,
    const i32* __restrict__ positions,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    bool interleaved,
    int cache_pairs,
    int heads_per_block,
    const bf16* __restrict__ v,
    bf16* __restrict__ k_pages,
    bf16* __restrict__ v_pages,
    const u32* __restrict__ qo_indptr,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    const u8* __restrict__ row_valid,
    int R,
    int page_size)
{
    const int n = blockIdx.x;
    const int total_heads = num_q_heads + num_kv_heads;

    const int half = head_dim / 2;
    const int pos = positions[n];

    // Every thread in this block shares token n, so the destination row is
    // resolved once here rather than per element.
    KvSlot slot{};
    bool write_this_row = false;
    if constexpr (kWriteKv) {
        write_this_row = (row_valid == nullptr) || (row_valid[n] != 0);
        if (write_this_row) {
            slot = kv_slot_for_token(qo_indptr, kv_page_indices, kv_page_indptr,
                                     kv_last_page_lens, n, R, page_size);
        }
    }

    // # `__sincosf` and the context length, measured
//
// The angle below is `pos * theta^(-2 * dp / head_dim)`. At `dp = 0` that
// exponent is 1, so **the largest angle this kernel evaluates IS the context
// position** -- 4,096 radians at position 4,096. `__sincosf` is the fast
// intrinsic and its accuracy degrades through argument reduction as the
// angle grows, measured against fp64 on an L40S:
//
//     position       __sincosf err     sinf err
//          1,024        4.06e-05        6.08e-10
//          4,096        1.33e-04        4.10e-09
//         16,384        5.48e-04        2.47e-08
//         65,536        1.91e-03        2.35e-08
//        262,144        1.05e-02        1.35e-09
//      8,388,608        2.75e-01        2.69e-09
//
// bf16 resolution is 2^-8 = 3.9e-03, so **this is below the storage noise
// floor up to about 64K and above it beyond**. `sinf`/`cosf` hold ~1e-08 at
// every position.
//
// The fix is a two-word edit and it is not free. Both forms of this family,
// same kernels otherwise:
//
//     tokens     rotate (hoisted)     rotate_partial (not hoisted)
//          1     2.01 -> 2.12  +5.2%    6.63 -> 7.46  +12.5%
//      1,024     3.20 -> 3.38  +5.6%   18.38 -> 21.60 +17.5%
//      8,192     9.88 -> 11.12 +12.5% 126.28 -> 152.07 +20.4%
//
// The hoisting below is what makes the difference between those columns:
// with the cache, a block evaluates `cache_pairs` transcendentals instead of
// `total_heads * cache_pairs`, so making each one 3x dearer costs a third as
// much.
//
// **This is a trade, not a bug, and it is stated rather than taken.** Below
// 64K context the error does not survive bf16 storage and `__sincosf` is
// free accuracy; above it the error is real and grows. Whoever owns the
// numerics decides. `rope/rope_tile.cuh` WAS a CuTile twin that used the
// accurate form -- 1.2-1.4x FASTER than this kernel while being three orders
// of magnitude closer to fp64 -- and was declined for exactly this reason;
// it is deleted now, with the other eleven tile headers, because nothing
// read the `Root` that named it and its NVRTC floor is above this crate's.
//
// The rotation angle depends only on (pos, dim_pair): every head of this
    // token shares it. Computing it inside the element loop ran a full-precision
    // `powf` plus a `__sincosf` once per (head, pair) -- for GLM's 65 QK heads
    // that is 65 evaluations of the same 32 transcendentals, and it made this
    // kernel cost more than the attention it feeds. Hoisting them into shared
    // memory keeps the arithmetic identical, so the outputs are bit-for-bit
    // what the per-element form produced.
    extern __shared__ float rope_cs[];
    const int cached = cache_pairs;
    for (int dp = threadIdx.x; dp < cached; dp += blockDim.x) {
        const float freq = powf(theta, -2.f * static_cast<float>(dp) /
                                       static_cast<float>(head_dim));
        const float ang = static_cast<float>(pos) * freq;
        float c, s;
        __sincosf(ang, &s, &c);
        rope_cs[dp] = c;
        rope_cs[cached + dp] = s;
    }
    if (cached > 0) __syncthreads();

    // Each thread handles one (head, dim_pair_idx).
    const int head_base = blockIdx.y * heads_per_block;
    const int heads_here = min(heads_per_block, total_heads - head_base);
    for (int t = threadIdx.x; t < heads_here * half; t += blockDim.x) {
        const int head_idx = head_base + t / half;
        const int dim_pair = t % half;

        float cos_v, sin_v;
        if (dim_pair < cached) {
            cos_v = rope_cs[dim_pair];
            sin_v = rope_cs[cached + dim_pair];
        } else {
            const float freq = powf(theta, -2.f * static_cast<float>(dim_pair) /
                                           static_cast<float>(head_dim));
            const float ang = static_cast<float>(pos) * freq;
            __sincosf(ang, &sin_v, &cos_v);
        }

        if (head_idx < num_q_heads) {
            bf16* qp = q + (static_cast<long long>(n) * num_q_heads +
                                     head_idx) * head_dim;
            if (interleaved) rotate_pair_interleaved(qp, dim_pair, cos_v, sin_v);
            else rotate_pair(qp, half, dim_pair, cos_v, sin_v);
            continue;
        }
        {
            const int kv_h = head_idx - num_q_heads;
            bf16* kp = k + (static_cast<long long>(n) * num_kv_heads +
                                     kv_h) * head_dim;
            if (interleaved) rotate_pair_interleaved(kp, dim_pair, cos_v, sin_v);
            else rotate_pair(kp, half, dim_pair, cos_v, sin_v);
            if constexpr (kWriteKv) {
                if (write_this_row) {
                    // K is still written to `k` as well: code past this point
                    // may read the contiguous copy, and at decode it is 1 KB.
                    const int j0 = interleaved ? dim_pair * 2 : dim_pair;
                    const int j1 = interleaved ? dim_pair * 2 + 1
                                               : dim_pair + half;
                    const bf16* vp =
                        v + (static_cast<long long>(n) * num_kv_heads + kv_h) *
                                head_dim;
                    const int base = kv_h * head_dim;
                    const long long d0 = kv_dst_index<kHnd>(
                        slot, base + j0, page_size, num_kv_heads, head_dim);
                    const long long d1 = kv_dst_index<kHnd>(
                        slot, base + j1, page_size, num_kv_heads, head_dim);
                    k_pages[d0] = kp[j0];
                    k_pages[d1] = kp[j1];
                    v_pages[d0] = vp[j0];
                    v_pages[d1] = vp[j1];
                }
            }
        }
    }
}

template <int BLOCK>
__global__ void qk_rmsnorm_rotate(
    bf16* __restrict__ q,
    bf16* __restrict__ k,
    const bf16* __restrict__ q_weight,
    const bf16* __restrict__ k_weight,
    const i32* __restrict__ positions,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    float eps)
{
    const int n = blockIdx.x;
    const int head_idx = blockIdx.y;
    const bool is_q = head_idx < num_q_heads;
    const int local_head = is_q ? head_idx : (head_idx - num_q_heads);
    bf16* row = is_q
        ? q + (static_cast<long long>(n) * num_q_heads + local_head) * head_dim
        : k + (static_cast<long long>(n) * num_kv_heads + local_head) * head_dim;
    const bf16* weight = is_q ? q_weight : k_weight;

    float local = 0.f;
    for (int i = threadIdx.x; i < head_dim; i += BLOCK) {
        const float v = bf16_to_f32(row[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    buf[threadIdx.x] = local;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) buf[threadIdx.x] += buf[threadIdx.x + off];
        __syncthreads();
    }

    const float inv_rms = rsqrtf(buf[0] / static_cast<float>(head_dim) + eps);
    const int half = head_dim / 2;
    const int pos = positions[n];
    for (int dim_pair = threadIdx.x; dim_pair < half; dim_pair += BLOCK) {
        const float a = bf16_to_f32(row[dim_pair]) *
            inv_rms * bf16_to_f32(weight[dim_pair]);
        const float b = bf16_to_f32(row[dim_pair + half]) *
            inv_rms * bf16_to_f32(weight[dim_pair + half]);
        const float freq = powf(theta,
            -2.f * static_cast<float>(dim_pair) / static_cast<float>(head_dim));
        const float ang = static_cast<float>(pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);
        row[dim_pair] = f32_to_bf16(a * cos_v - b * sin_v);
        row[dim_pair + half] = f32_to_bf16(b * cos_v + a * sin_v);
    }
}

template <int BLOCK>
__global__ void qk_rmsnorm_rotate_rounded(
    bf16* __restrict__ q,
    bf16* __restrict__ k,
    const bf16* __restrict__ q_weight,
    const bf16* __restrict__ k_weight,
    const i32* __restrict__ positions,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    float eps)
{
    const int n = blockIdx.x;
    const int head_idx = blockIdx.y;
    const bool is_q = head_idx < num_q_heads;
    const int local_head = is_q ? head_idx : (head_idx - num_q_heads);
    bf16* row = is_q
        ? q + (static_cast<long long>(n) * num_q_heads + local_head) * head_dim
        : k + (static_cast<long long>(n) * num_kv_heads + local_head) * head_dim;
    const bf16* weight = is_q ? q_weight : k_weight;

    float local = 0.f;
    for (int i = threadIdx.x; i < head_dim; i += BLOCK) {
        const float v = bf16_to_f32(row[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    buf[threadIdx.x] = local;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) buf[threadIdx.x] += buf[threadIdx.x + off];
        __syncthreads();
    }

    const float inv_rms = rsqrtf(buf[0] / static_cast<float>(head_dim) + eps);
    const int half = head_dim / 2;
    const int pos = positions[n];
    for (int dim_pair = threadIdx.x; dim_pair < half; dim_pair += BLOCK) {
        const bf16 norm_a = f32_to_bf16(
            bf16_to_f32(row[dim_pair]) *
            inv_rms * bf16_to_f32(weight[dim_pair]));
        const bf16 norm_b = f32_to_bf16(
            bf16_to_f32(row[dim_pair + half]) *
            inv_rms * bf16_to_f32(weight[dim_pair + half]));
        const float a = bf16_to_f32(norm_a);
        const float b = bf16_to_f32(norm_b);
        const float freq = powf(theta,
            -2.f * static_cast<float>(dim_pair) / static_cast<float>(head_dim));
        const float ang = static_cast<float>(pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);
        row[dim_pair] = f32_to_bf16(a * cos_v - b * sin_v);
        row[dim_pair + half] = f32_to_bf16(b * cos_v + a * sin_v);
    }
}

// Fused per-head Q/K RMSNorm + interleaved M-RoPE (Qwen3-VL text tower).
// Reads three position components per token (`positions[3n+axis]`, axis
// 0=t,1=h,2=w) and selects the rotary axis per frequency index using the
// interleaved layout (HF `apply_interleaved_mrope`):
//   freqs_t = freqs[T]; H overwrites idx slice(1, 3*s1, 3); W slice(2, 3*s2, 3)
// i.e. for dim_pair j: axis = H if (j%3==1 && j < 3*s1); W if (j%3==2 && j<3*s2);
// otherwise T. The rotation itself is the standard half/half rotate_half pairing
// (j, j+head_dim/2) with frequency theta^(-2j/head_dim). Preserves the
// bf16(rmsnorm(x)) materialization point (parity-sensitive, like Gemma-4).
template <int BLOCK>
__global__ void qk_rmsnorm_rotate_mrope(
    bf16* __restrict__ q,
    bf16* __restrict__ k,
    const bf16* __restrict__ q_weight,
    const bf16* __restrict__ k_weight,
    const i32* __restrict__ positions,  // [num_tokens, 3] (t,h,w)
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    float eps,
    int s0, int s1, int s2)  // mrope_section (t,h,w)
{
    const int n = blockIdx.x;
    const int head_idx = blockIdx.y;
    const bool is_q = head_idx < num_q_heads;
    const int local_head = is_q ? head_idx : (head_idx - num_q_heads);
    bf16* row = is_q
        ? q + (static_cast<long long>(n) * num_q_heads + local_head) * head_dim
        : k + (static_cast<long long>(n) * num_kv_heads + local_head) * head_dim;
    const bf16* weight = is_q ? q_weight : k_weight;

    float local = 0.f;
    for (int i = threadIdx.x; i < head_dim; i += BLOCK) {
        const float v = bf16_to_f32(row[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    buf[threadIdx.x] = local;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) buf[threadIdx.x] += buf[threadIdx.x + off];
        __syncthreads();
    }

    const float inv_rms = rsqrtf(buf[0] / static_cast<float>(head_dim) + eps);
    const int half = head_dim / 2;
    const int pos_t = positions[3 * n + 0];
    const int pos_h = positions[3 * n + 1];
    const int pos_w = positions[3 * n + 2];
    (void)s0;
    for (int dim_pair = threadIdx.x; dim_pair < half; dim_pair += BLOCK) {
        const bf16 norm_a = f32_to_bf16(
            bf16_to_f32(row[dim_pair]) *
            inv_rms * bf16_to_f32(weight[dim_pair]));
        const bf16 norm_b = f32_to_bf16(
            bf16_to_f32(row[dim_pair + half]) *
            inv_rms * bf16_to_f32(weight[dim_pair + half]));
        const float a = bf16_to_f32(norm_a);
        const float b = bf16_to_f32(norm_b);

        // Interleaved axis selection.
        int axis_pos;
        const int m = dim_pair % 3;
        if (m == 1 && dim_pair < 3 * s1)      axis_pos = pos_h;
        else if (m == 2 && dim_pair < 3 * s2) axis_pos = pos_w;
        else                                  axis_pos = pos_t;

        const float freq = powf(theta,
            -2.f * static_cast<float>(dim_pair) / static_cast<float>(head_dim));
        const float ang = static_cast<float>(axis_pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);
        row[dim_pair] = f32_to_bf16(a * cos_v - b * sin_v);
        row[dim_pair + half] = f32_to_bf16(b * cos_v + a * sin_v);
    }
}

// ── The fused norm+rope forms ───────────────────────────────────────────────
//
// All four reduce across a block per (token, head) and rotate in place. They
// are `bf16` and not `T` because no row asks for a second numeric format --
// all four ARE named by rows now, each at bf16, and a template nothing
// instantiates is text that has never been compiled pretending to be a
// capability, which is the argument the opening paragraph makes for the other
// six. The `dim3(num_tokens, total_heads)` grid three of them launch is
// spelled after all, by `LaunchRule::RowsPackedHeadsNarrow`; the fourth,
// `qk_rmsnorm_rotate_rounded`, is `LaunchRule::Unstated` for a reason that is
// not the grid's SHAPE but whose `num_kv_heads` its second axis reads --
// `families/rope.rs` states it at the row.

// Peel device-window variant (the device-window campaign): the row
// window rides in device memory; the grid spans the full lane count and
// out-of-window rows early-out (uniform per block — blockIdx.x is the
// row — so the shared-memory reduction below never diverges). Buffers
// and positions are BASE pointers.
template <int BLOCK>
__global__ void qk_rmsnorm_rotate_devwin(
    bf16* __restrict__ q,
    bf16* __restrict__ k,
    const bf16* __restrict__ q_weight,
    const bf16* __restrict__ k_weight,
    const i32* __restrict__ positions,
    const u32* __restrict__ win,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    float theta,
    float eps)
{
    const int n = blockIdx.x;
    {
        const int w0 = static_cast<int>(win[0]);
        const int w1 = static_cast<int>(win[1]);
        if (n < w0 || n >= w0 + w1) return;
    }
    const int head_idx = blockIdx.y;
    const bool is_q = head_idx < num_q_heads;
    const int local_head = is_q ? head_idx : (head_idx - num_q_heads);
    bf16* row = is_q
        ? q + (static_cast<long long>(n) * num_q_heads + local_head) * head_dim
        : k + (static_cast<long long>(n) * num_kv_heads + local_head) * head_dim;
    const bf16* weight = is_q ? q_weight : k_weight;

    float local = 0.f;
    for (int i = threadIdx.x; i < head_dim; i += BLOCK) {
        const float v = bf16_to_f32(row[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    buf[threadIdx.x] = local;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) buf[threadIdx.x] += buf[threadIdx.x + off];
        __syncthreads();
    }

    const float inv_rms = rsqrtf(buf[0] / static_cast<float>(head_dim) + eps);
    const int half = head_dim / 2;
    const int pos = positions[n];
    for (int dim_pair = threadIdx.x; dim_pair < half; dim_pair += BLOCK) {
        const float a = bf16_to_f32(row[dim_pair]) *
            inv_rms * bf16_to_f32(weight[dim_pair]);
        const float b = bf16_to_f32(row[dim_pair + half]) *
            inv_rms * bf16_to_f32(weight[dim_pair + half]);
        const float freq = powf(theta,
            -2.f * static_cast<float>(dim_pair) / static_cast<float>(head_dim));
        const float ang = static_cast<float>(pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);
        row[dim_pair] = f32_to_bf16(a * cos_v - b * sin_v);
        row[dim_pair + half] = f32_to_bf16(b * cos_v + a * sin_v);
    }
}

// ── YaRN ────────────────────────────────────────────────────────────────────

// Piecewise-linear interp between full-scale (high-freq pairs, kept
// untouched) and `factor`-scaled (low-freq pairs); smooth band uses
// `(orig_max_pos / wavelen - low_freq_factor) / (high - low)` blended.
__device__ __forceinline__ float yarn_freq(
    float base_freq, float factor,
    float low_freq_factor, float high_freq_factor,
    float orig_max_pos)
{
    constexpr float TWO_PI = 6.2831853071795864769f;
    const float wavelen   = TWO_PI / base_freq;
    const float low_wave  = orig_max_pos / low_freq_factor;
    const float high_wave = orig_max_pos / high_freq_factor;
    if (wavelen < high_wave) return base_freq;            // high-freq: no scale
    if (wavelen > low_wave)  return base_freq / factor;   // low-freq: full scale
    const float smooth = (orig_max_pos / wavelen - low_freq_factor) /
                         (high_freq_factor - low_freq_factor);
    return (1.f - smooth) * (base_freq / factor) + smooth * base_freq;
}

__global__ void rotate_yarn(
    bf16* __restrict__ q,
    bf16* __restrict__ k,
    const i32* __restrict__ positions,
    int num_q_heads, int num_kv_heads, int head_dim,
    float theta, float factor,
    float low_freq_factor, float high_freq_factor,
    float orig_max_pos,
    int heads_per_block)
{
    const int n = blockIdx.x;
    const int total_heads = num_q_heads + num_kv_heads;
    const int half = head_dim / 2;
    const int pos = positions[n];

    const int head_base = blockIdx.y * heads_per_block;
    const int heads_here = min(heads_per_block, total_heads - head_base);
    for (int t = threadIdx.x; t < heads_here * half; t += blockDim.x) {
        const int head_idx = head_base + t / half;
        const int dim_pair = t % half;

        const float base_freq = powf(theta,
            -2.f * static_cast<float>(dim_pair) / static_cast<float>(head_dim));
        const float freq = yarn_freq(base_freq, factor,
                                     low_freq_factor, high_freq_factor,
                                     orig_max_pos);
        const float ang = static_cast<float>(pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);

        if (head_idx < num_q_heads) {
            bf16* qp = q + (static_cast<long long>(n) * num_q_heads +
                                     head_idx) * head_dim;
            rotate_pair(qp, half, dim_pair, cos_v, sin_v);
        } else {
            const int kv_h = head_idx - num_q_heads;
            bf16* kp = k + (static_cast<long long>(n) * num_kv_heads +
                                     kv_h) * head_dim;
            rotate_pair(kp, half, dim_pair, cos_v, sin_v);
        }
    }
}

// `yarn_original_freq` lives in rope_device.cuh, shared with the fused
// MLA-prepare kernel.

__global__ void rotate_yarn_original(
    bf16* __restrict__ q,
    bf16* __restrict__ k,
    const i32* __restrict__ positions,
    int num_q_heads, int num_kv_heads, int head_dim,
    float theta, float factor,
    float low_dim, float high_dim,
    float mscale,
    bool interleaved,
    int heads_per_block,
    int cache_pairs)
{
    extern __shared__ float2 yarn_cs[];
    const int n = blockIdx.x;
    const int total_heads = num_q_heads + num_kv_heads;
    const int half = head_dim / 2;
    const int pos = positions[n];

    // The rotation angle depends only on `dim_pair` and the token's position,
    // not on the head -- so a block that covers many heads was recomputing the
    // same `powf` and `__sincosf` once per head. Do it `half` times instead of
    // `heads_per_block * half` times and share the result.
    auto angle = [&](int d) -> float2 {
        const float base_freq = powf(theta,
            -2.f * static_cast<float>(d) / static_cast<float>(head_dim));
        const float freq = yarn_original_freq(base_freq, factor,
                                              low_dim, high_dim, d);
        float cos_v, sin_v;
        __sincosf(static_cast<float>(pos) * freq, &sin_v, &cos_v);
        return make_float2(cos_v * mscale, sin_v * mscale);
    };
    for (int d = threadIdx.x; d < cache_pairs; d += blockDim.x) {
        yarn_cs[d] = angle(d);
    }
    if (cache_pairs > 0) __syncthreads();

    const int head_base = blockIdx.y * heads_per_block;
    const int heads_here = min(heads_per_block, total_heads - head_base);
    for (int t = threadIdx.x; t < heads_here * half; t += blockDim.x) {
        const int head_idx = head_base + t / half;
        const int dim_pair = t % half;
        const float2 cs = dim_pair < cache_pairs ? yarn_cs[dim_pair]
                                                 : angle(dim_pair);

        if (head_idx < num_q_heads) {
            bf16* qp = q + (static_cast<long long>(n) * num_q_heads +
                                     head_idx) * head_dim;
            if (interleaved) rotate_pair_interleaved(qp, dim_pair, cs.x, cs.y);
            else             rotate_pair(qp, half, dim_pair, cs.x, cs.y);
        } else {
            const int kv_h = head_idx - num_q_heads;
            bf16* kp = k + (static_cast<long long>(n) * num_kv_heads +
                                     kv_h) * head_dim;
            if (interleaved) rotate_pair_interleaved(kp, dim_pair, cs.x, cs.y);
            else             rotate_pair(kp, half, dim_pair, cs.x, cs.y);
        }
    }
}

// ── Partial rotary ──────────────────────────────────────────────────────────

// Proportional RoPE (Gemma-4 full-attention layers, HF reference).
//
// HF builds the frequency table as `freq[k] = 1 / theta^(2k/head_dim)`
// for k ∈ [0, rotary_dim/2), then pads the rest of the head's lower-
// half dim entries with `cos=1 / sin=0` (identity). The pair offset
// is the *full* `head_dim/2`, NOT `rotary_dim/2` — every dim in the
// lower half rotates with its mate in the upper half, but the
// rotation angle is zero for k ≥ rotary_dim/2 (so those pairs pass
// through unchanged).
//
// Two ways the previous draft of this kernel got it wrong:
//   1. used `rotary_dim` as the frequency denominator instead of
//      `head_dim` — wrong angle progression.
//   2. used `rotary_dim/2` as the pair offset instead of
//      `head_dim/2` — paired the wrong dims with each other.
template <class T>
__global__ void rotate_partial(
    T* __restrict__ q,
    T* __restrict__ k,
    const i32* __restrict__ positions,
    int position_delta,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    float theta)
{
    const int n = blockIdx.x;
    const int total_heads = num_q_heads + num_kv_heads;
    const int half = head_dim / 2;
    const int rope_angles = rotary_dim / 2;
    const int pos = positions[n] + position_delta;

    for (int t = threadIdx.x; t < total_heads * half; t += blockDim.x) {
        const int head_idx = t / half;
        const int dim_pair = t % half;

        float cos_v = 1.f, sin_v = 0.f;
        if (dim_pair < rope_angles) {
            const float freq = powf(theta,
                -2.f * static_cast<float>(dim_pair) /
                       static_cast<float>(head_dim));
            const float ang = static_cast<float>(pos) * freq;
            __sincosf(ang, &sin_v, &cos_v);
        }
        // Skip identity rotations entirely — `dim_pair ≥ rope_angles`
        // multiplies the pair by [[1,0],[0,1]] which is a no-op.
        if (dim_pair >= rope_angles) continue;

        if (head_idx < num_q_heads) {
            T* qp = q +
                (static_cast<long long>(n) * num_q_heads + head_idx) * head_dim;
            const float a = Elem<T>::to_f32(qp[dim_pair]);
            const float b = Elem<T>::to_f32(qp[dim_pair + half]);
            qp[dim_pair]        = Elem<T>::from_f32(a * cos_v - b * sin_v);
            qp[dim_pair + half] = Elem<T>::from_f32(b * cos_v + a * sin_v);
        } else {
            const int kv_h = head_idx - num_q_heads;
            T* kp = k +
                (static_cast<long long>(n) * num_kv_heads + kv_h) * head_dim;
            const float a = Elem<T>::to_f32(kp[dim_pair]);
            const float b = Elem<T>::to_f32(kp[dim_pair + half]);
            kp[dim_pair]        = Elem<T>::from_f32(a * cos_v - b * sin_v);
            kp[dim_pair + half] = Elem<T>::from_f32(b * cos_v + a * sin_v);
        }
    }
}

// Rotates the LAST `rotary_dim` channels rather than the first, and its
// YaRN ramp bounds are computed ON THE HOST -- which is why its row is
// UNSOURCED. `low_dim`/`high_dim` arrive as arguments because the arithmetic
// that derives them from `beta_fast`/`beta_slow` over `rotary_dim` is not a
// `Source` any row could state. The row itself is `ROPE_SIGS[10]` at
// `LaunchRule::RouteRows`, the one block per token `rope.cu:382-385`
// launches: a grid this kernel HAS a rule for, and operands it does not.
__global__ void rotate_partial_last(
    bf16* __restrict__ q,
    bf16* __restrict__ k,
    const i32* __restrict__ positions,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int rotary_dim,
    float theta,
    bool inverse,
    bool interleaved,
    float yarn_factor,
    float yarn_low_dim,
    float yarn_high_dim)
{
    const int n = blockIdx.x;
    const int total_heads = num_q_heads + num_kv_heads;
    const int rope_half = rotary_dim / 2;
    const int offset = head_dim - rotary_dim;
    const int pos = positions[n];

    for (int t = threadIdx.x; t < total_heads * rope_half; t += blockDim.x) {
        const int head_idx = t / rope_half;
        const int dim_pair = t % rope_half;

        float freq = powf(theta,
            -2.f * static_cast<float>(dim_pair) /
                   static_cast<float>(rotary_dim));
        if (yarn_factor > 1.f) {
            freq = yarn_original_freq(freq, yarn_factor,
                                      yarn_low_dim, yarn_high_dim, dim_pair);
        }
        const float ang = (inverse ? -1.f : 1.f) * static_cast<float>(pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);

        const bool is_q = (head_idx < num_q_heads);
        bf16* base = is_q
            ? q + static_cast<long long>(n * num_q_heads + head_idx) * head_dim
            : k + static_cast<long long>(n * num_kv_heads + (head_idx - num_q_heads)) * head_dim;

        // GPT-J pairing (adjacent dims) for DeepSeek-V4 (`is_neox_style=False`
        // in vLLM `build_deepseek_v4_rope`); NeoX half/half otherwise.
        const int i = interleaved ? offset + 2 * dim_pair : offset + dim_pair;
        const int j = interleaved ? offset + 2 * dim_pair + 1
                                  : offset + dim_pair + rope_half;
        const float a = bf16_to_f32(base[i]);
        const float b = bf16_to_f32(base[j]);
        base[i] = f32_to_bf16(a * cos_v - b * sin_v);
        base[j] = f32_to_bf16(b * cos_v + a * sin_v);
    }
}
}  // namespace pie::rope
