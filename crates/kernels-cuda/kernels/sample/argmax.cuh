//===-- argmax.cuh - the sampling kernels, as `__global__` templates ---===//
//
// Twelve `__global__` templates and one include, for the thirteen kernels
// `argmax.cu` used to hold: `argmax_bf16` and `argmax_fp32` differed only in
// a load, so they are one template with two rows. No host function, no
// `<<<>>>`, no stream. Three rows in `kernels_cuda::families::sample`
// name two of the twelve templates; the other ten are here because the device
// half of a file belongs in the device half of a file, and their launchers
// stay in `argmax.cu` because nothing in the row vocabulary can say what
// those launchers do.
//
// # Why this file exists
//
// `argmax.cu` cannot be handed to NVRTC. It includes `<cstddef>` and
// `<cstdlib>`, it reaches `std::uintptr_t` and `std::min`, it queries the
// device for an SM count, it owns a `static cudaMalloc`'d scratch buffer, and
// its kernels sat in an anonymous namespace where no name expression could
// reach them. The device half moved here, and the two halves now share ONE
// definition -- the property `tests/sources.rs::no_global_is_defined_twice`
// exists to hold after `norm/altup_aux` shipped two copies of a kernel for a
// release.
//
// # `-INFINITY` is a `<math.h>` macro
//
// Eight kernels opened their reduction with it. NVRTC answered 0 of 31
// standard headers when this tree measured it, so the macro is not there and
// the prelude's `neg_inf()` is: `__int_as_float(0xff800000u)`, the same bits
// by construction rather than by whatever the C library's macro expands to.
//
// # What has a row, and what does not
//
// `argmax` and `argmax_compact_scatter` are `LaunchRule::Rms` exactly -- one
// block per row, 256 threads. (The rule also asks for 32 bytes of dynamic
// shared memory, which these kernels do not declare and therefore never
// touch; an unread allocation is not a behaviour.) Both kernels stride the
// vocab by the COMPILE-TIME `BLOCK` and size their shared arrays with it, so
// a rule that launched any other block width would read past the reduction
// buffer. `Rms` gives exactly 256, which is why it fits.
//
// The other ten are blocked, and each for a reason worth writing down:
//
//   * `argmax_vec2` and `argmax_compact_scatter_vec2` are chosen by
//     `argmax_vec2_usable(logits, vocab)` -- a run-time test on an operand's
//     ADDRESS and on the parity of the vocab. A `Source` states where a value
//     comes from, not a predicate over one, and firing the vec2 form on an
//     odd vocab puts every second row on a 2-byte boundary and faults.
//   * `masked_embedding_argmax`, `topk_centroids` and
//     `masked_embedding_tile_argmax_pairs` have launchers that CLAMP
//     `centroid_top_k` to `MAX_MASKED_TOP_K` before passing it. The kernels
//     index `__shared__` arrays of exactly that size with it, so the clamp is
//     load-bearing, and no `Source` expresses `min(k, 64)`. A row would turn
//     a truncation into a shared-memory overrun on any config that asked for
//     more. The tile form is doubly blocked: its grid is `dim3(rows, tiles)`,
//     where `tiles` is a caller's count and not a width any rule divides.
//   * `select_lm_head_argmax_pairs` has an `Elementwise` grid, but its
//     `num_tiles` operand is the block count the launcher computed from
//     `cudaDevAttrMultiProcessorCount`. The grid a rule can state; a device
//     query it cannot.
//   * `lm_head_gemv_argmax_int8` and `lm_head_gemv_argmax_bf16` are the
//     textbook case: grid.x is `min(num_sms * blocks_per_sm, ceil(vocab /
//     8))` -- an occupancy query -- the grid is 2-D over (blocks, rows), the
//     dynamic shared memory is `hidden * sizeof(float)` rather than a
//     constant, the launcher owns a `static cudaMalloc`'d scratch buffer that
//     grows, and it fires TWO kernels. Five separate things no row says.
//   * `argmax_accumulate` launches 1024 threads and carries one accumulator
//     slot per warp, a width fixed by a `static_assert` against
//     `kArgmaxAccumSlots`; `argmax_finalize` launches 32. No rule states
//     either width, and `argmax_accumulate` additionally takes two `bool`s
//     that the operand binder refuses.
//
// Reporting that is worth more than a row that launches the wrong extent.
//
// # What `T` means, and why even the blocked kernels are templates
//
// The element type of the LOGITS, and nothing else. `argmax_fp32` was a
// separate kernel for one reason -- fp32 has no `Elem` specialisation in the
// prelude, because there fp32 is the compute type rather than a storage
// format. `Logit<T>` below is that one specialisation, delegating to `Elem`
// for everything else, and the two kernels became one template with two rows.
//
// The blocked kernels are templates too, and that is deliberate: an
// uninstantiated template emits nothing, so a unit compiles only what its
// rows name. A plain `__global__` would land in every cubin this file
// produces, carrying `char4` and `ldg` intrinsics into a compile that has no
// use for them.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "prelude/device.cuh"

namespace pie::sample {

// The scalar layer is the PRELUDE's. Named here so the kernels read as they
// always did, and so `argmax.cu`'s launchers -- which sit in
// `kernels::sample` and spell every cast `bf16` -- resolve that
// spelling through this namespace to the same type.

/// fp32 logits, as a storage format.
///
/// `Elem<T>` has no `float` specialisation and should not grow one: there
/// fp32 is what a kernel COMPUTES in, and a specialisation would make
/// `Elem<float>::from_f32` an identity that reads like a conversion. Here
/// fp32 is a format a caller's buffer is written in -- `argmax_fp32` was a
/// whole second kernel for that one difference -- so the widening lives at
/// the one call site that needs it and delegates everywhere else.
template <class T>
struct Logit {
    static __device__ __forceinline__ float to_f32(T v) { return Elem<T>::to_f32(v); }
};

using f32 = float;

template <>
struct Logit<f32> {
    static __device__ __forceinline__ float to_f32(f32 v) { return v; }
};

/// The block width these kernels are COMPILED for, not merely launched with.
///
/// Every argmax below strides the vocab by this constant and sizes its
/// `__shared__` reduction by it. `LaunchRule::Rms` names the same 256, which
/// is what makes the two agree; a launch of any other width would read past
/// the buffer.
constexpr i32 BLOCK = 256;
/// The masked-embedding kernels hold their per-row centroid shortlist in
/// shared memory, so the shortlist has a maximum. The launcher clamps to it,
/// which is why those kernels have no row -- see this file's header.
constexpr i32 MAX_MASKED_TOP_K = 64;
/// Tokens scored per tile block, one warp each.
constexpr i32 MASKED_TILE_TOKENS = 8;
/// The vocab-streaming accumulator's block, and the accumulator slots that
/// follow from it: one per warp. `argmax.cu` static_asserts this against the
/// `kArgmaxAccumSlots` its header publishes to callers.
constexpr i32 kAccumThreads = 1024;
constexpr i32 kAccumWarps = kAccumThreads / 32;
/// The fused GEMV's warps per block, one vocab row each.
constexpr i32 GEMV_WARPS = 8;
constexpr i32 GEMV_BLOCK_DIM = GEMV_WARPS * 32;

/// `(value, token)` in one 64-bit word, value in the high half.
///
/// The pair is compared as a value first and a token second, which is what
/// the tie-break needs, and packing it lets a partial result cross a kernel
/// boundary as one store instead of two.
__device__ __forceinline__ u64 pack_argmax_pair(float value, i32 token) {
    const u32 value_bits = __float_as_uint(value);
    return (static_cast<u64>(value_bits) << 32) | static_cast<u32>(token);
}

__device__ __forceinline__ float unpack_argmax_value(u64 pair) {
    const u32 bits = static_cast<u32>(pair >> 32);
    return __uint_as_float(bits);
}

__device__ __forceinline__ i32 unpack_argmax_token(u64 pair) {
    return static_cast<i32>(static_cast<u32>(pair));
}

/// The tie-break, in one place: lowest index wins, which is what
/// `torch.argmax` and `numpy.argmax` both answer.
///
/// A total order on `(value, -index)`, which is why a chunked argmax over
/// slabs answers the same token as one pass over the concatenation: the
/// result cannot depend on scan order.
__device__ __forceinline__ void update_argmax(
    float v, i32 idx, float& best_val, i32& best_idx)
{
    if (v > best_val || (v == best_val && idx < best_idx)) {
        best_val = v;
        best_idx = idx;
    }
}

/// The shared-memory tree the row-per-block argmaxes end with.
///
/// `BLOCK`-wide, halving, `__syncthreads()` between levels -- the original's
/// order, kept because the reduction's order is what decides which of two
/// equal values wins and `driver-pipeline`'s tolerance contract holds argmax
/// indices to zero.
__device__ __forceinline__ i32 block_argmax(
    float best_val, i32 best_idx, float* vals, i32* idxs)
{
    const i32 tid = threadIdx.x;
    vals[tid] = best_val;
    idxs[tid] = best_idx;
    __syncthreads();

    for (i32 off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) {
            update_argmax(vals[tid + off], idxs[tid + off], vals[tid], idxs[tid]);
        }
        __syncthreads();
    }
    return idxs[0];
}

// ---------------------------------------------------------------------------
// The three with rows.
// ---------------------------------------------------------------------------

/// One block per row, threads striding the vocab.
template <class T>
__global__ void argmax(
    const T* __restrict__ logits,
    i32* __restrict__ out,
    i32 vocab)
{
    const i32 row = blockIdx.x;
    const i32 tid = threadIdx.x;
    const T* row_ptr = logits + static_cast<long long>(row) * vocab;

    float best_val = neg_inf();
    i32 best_idx = 0;

    for (i32 i = tid; i < vocab; i += BLOCK) {
        update_argmax(Logit<T>::to_f32(row_ptr[i]), i, best_val, best_idx);
    }

    __shared__ float vals[BLOCK];
    __shared__ i32 idxs[BLOCK];
    const i32 winner = block_argmax(best_val, best_idx, vals, idxs);
    if (tid == 0) out[row] = winner;
}

/// The same argmax over a COMPACT slab, scattered back to the rows the slab
/// was gathered from.
///
/// The logits are indexed by the compact row and the output by
/// `row_indices[compact_row]`, so a fire that dropped rows writes its answers
/// where the un-dropped batch expects them.
template <class T>
__global__ void argmax_compact_scatter(
    const T* __restrict__ logits,
    const i32* __restrict__ row_indices,
    i32* __restrict__ out,
    i32 vocab)
{
    const i32 compact_row = blockIdx.x;
    const i32 original_row = row_indices[compact_row];
    const i32 tid = threadIdx.x;
    const T* row_ptr = logits + static_cast<long long>(compact_row) * vocab;

    float best_val = neg_inf();
    i32 best_idx = 0;

    for (i32 i = tid; i < vocab; i += BLOCK) {
        update_argmax(Logit<T>::to_f32(row_ptr[i]), i, best_val, best_idx);
    }

    __shared__ float vals[BLOCK];
    __shared__ i32 idxs[BLOCK];
    const i32 winner = block_argmax(best_val, best_idx, vals, idxs);
    if (tid == 0) out[original_row] = winner;
}

// ---------------------------------------------------------------------------
// The vectorised pair: chosen by an ADDRESS, so no row. See the header.
// ---------------------------------------------------------------------------

template <class T>
__global__ void argmax_vec2(
    const T* __restrict__ logits,
    i32* __restrict__ out,
    i32 vocab)
{
    static_assert(is_same<T, bf16>::value,
                  "the packed path loads through `bf16x2`, a bf16 PAIR");
    const i32 row = blockIdx.x;
    const i32 tid = threadIdx.x;
    const T* row_ptr = logits + static_cast<long long>(row) * vocab;
    const auto* row2 = reinterpret_cast<const bf16x2*>(row_ptr);

    float best_val = neg_inf();
    i32 best_idx = 0;

    const i32 even_end = vocab & ~1;
    for (i32 j = tid; j < even_end / 2; j += BLOCK) {
        const float2 pair = bf16x2_to_f32(row2[j]);
        const i32 i = j * 2;
        update_argmax(pair.x, i, best_val, best_idx);
        update_argmax(pair.y, i + 1, best_val, best_idx);
    }
    if ((vocab & 1) && tid == 0) {
        update_argmax(bf16_to_f32(row_ptr[vocab - 1]), vocab - 1, best_val, best_idx);
    }

    __shared__ float vals[BLOCK];
    __shared__ i32 idxs[BLOCK];
    const i32 winner = block_argmax(best_val, best_idx, vals, idxs);
    if (tid == 0) out[row] = winner;
}

template <class T>
__global__ void argmax_compact_scatter_vec2(
    const T* __restrict__ logits,
    const i32* __restrict__ row_indices,
    i32* __restrict__ out,
    i32 vocab)
{
    static_assert(is_same<T, bf16>::value,
                  "the packed path loads through `bf16x2`, a bf16 PAIR");
    const i32 compact_row = blockIdx.x;
    const i32 original_row = row_indices[compact_row];
    const i32 tid = threadIdx.x;
    const T* row_ptr = logits + static_cast<long long>(compact_row) * vocab;
    const auto* row2 = reinterpret_cast<const bf16x2*>(row_ptr);

    float best_val = neg_inf();
    i32 best_idx = 0;

    const i32 even_end = vocab & ~1;
    for (i32 j = tid; j < even_end / 2; j += BLOCK) {
        const float2 pair = bf16x2_to_f32(row2[j]);
        const i32 i = j * 2;
        update_argmax(pair.x, i, best_val, best_idx);
        update_argmax(pair.y, i + 1, best_val, best_idx);
    }
    if ((vocab & 1) && tid == 0) {
        update_argmax(bf16_to_f32(row_ptr[vocab - 1]), vocab - 1, best_val, best_idx);
    }

    __shared__ float vals[BLOCK];
    __shared__ i32 idxs[BLOCK];
    const i32 winner = block_argmax(best_val, best_idx, vals, idxs);
    if (tid == 0) out[original_row] = winner;
}

// ---------------------------------------------------------------------------
// Masked embedding: score only the tokens under the top centroids.
//
// No rows -- the launchers clamp `centroid_top_k` and the kernels index
// shared memory with it. See the header.
// ---------------------------------------------------------------------------

template <class T>
__global__ void masked_embedding_argmax(
    const T* __restrict__ centroid_logits,
    const T* __restrict__ hidden_states,
    const T* __restrict__ lm_head_weight,
    const i64* __restrict__ token_ordering,
    i32* __restrict__ out,
    i32 hidden,
    i32 num_centroids,
    i32 centroid_top_k,
    i32 vocab_per_centroid)
{
    const i32 row = blockIdx.x;
    const i32 tid = threadIdx.x;
    const auto* centroid_row =
        centroid_logits + static_cast<long long>(row) * num_centroids;
    const auto* hidden_row = hidden_states + static_cast<long long>(row) * hidden;

    __shared__ i32 top_centroids[MAX_MASKED_TOP_K];
    __shared__ float top_values[MAX_MASKED_TOP_K];
    __shared__ float vals[BLOCK];
    __shared__ i32 idxs[BLOCK];

    if (tid == 0) {
        for (i32 k = 0; k < centroid_top_k; ++k) {
            top_centroids[k] = 0;
            top_values[k] = neg_inf();
        }
        for (i32 c = 0; c < num_centroids; ++c) {
            const float v = Elem<T>::to_f32(centroid_row[c]);
            i32 insert = centroid_top_k;
            for (i32 k = 0; k < centroid_top_k; ++k) {
                if (v > top_values[k] || (v == top_values[k] && c < top_centroids[k])) {
                    insert = k;
                    break;
                }
            }
            if (insert < centroid_top_k) {
                for (i32 k = centroid_top_k - 1; k > insert; --k) {
                    top_values[k] = top_values[k - 1];
                    top_centroids[k] = top_centroids[k - 1];
                }
                top_values[insert] = v;
                top_centroids[insert] = c;
            }
        }
    }
    __syncthreads();

    float best_val = neg_inf();
    i32 best_tok = 0;
    const i32 selected = centroid_top_k * vocab_per_centroid;
    for (i32 s = tid; s < selected; s += BLOCK) {
        const i32 centroid_rank = s / vocab_per_centroid;
        const i32 centroid_off = s - centroid_rank * vocab_per_centroid;
        const i32 centroid = top_centroids[centroid_rank];
        const i64 tok64 =
            token_ordering[static_cast<long long>(centroid) * vocab_per_centroid +
                           centroid_off];
        const i32 tok = static_cast<i32>(tok64);
        const auto* wrow = lm_head_weight + static_cast<long long>(tok) * hidden;
        float dot = 0.f;
        for (i32 h = 0; h < hidden; ++h) {
            dot += Elem<T>::to_f32(hidden_row[h]) * Elem<T>::to_f32(wrow[h]);
        }
        update_argmax(dot, tok, best_val, best_tok);
    }

    const i32 winner = block_argmax(best_val, best_tok, vals, idxs);
    if (tid == 0) out[row] = winner;
}

template <class T>
__global__ void topk_centroids(
    const T* __restrict__ centroid_logits,
    i32* __restrict__ top_centroids,
    i32 num_centroids,
    i32 centroid_top_k)
{
    const i32 row = blockIdx.x;
    const i32 tid = threadIdx.x;
    const auto* centroid_row =
        centroid_logits + static_cast<long long>(row) * num_centroids;

    __shared__ i32 top_idx[MAX_MASKED_TOP_K];
    __shared__ float top_val[MAX_MASKED_TOP_K];

    if (tid == 0) {
        for (i32 k = 0; k < centroid_top_k; ++k) {
            top_idx[k] = 0;
            top_val[k] = neg_inf();
        }
        for (i32 c = 0; c < num_centroids; ++c) {
            const float v = Elem<T>::to_f32(centroid_row[c]);
            i32 insert = centroid_top_k;
            for (i32 k = 0; k < centroid_top_k; ++k) {
                if (v > top_val[k] || (v == top_val[k] && c < top_idx[k])) {
                    insert = k;
                    break;
                }
            }
            if (insert < centroid_top_k) {
                for (i32 k = centroid_top_k - 1; k > insert; --k) {
                    top_val[k] = top_val[k - 1];
                    top_idx[k] = top_idx[k - 1];
                }
                top_val[insert] = v;
                top_idx[insert] = c;
            }
        }
        for (i32 k = 0; k < centroid_top_k; ++k) {
            top_centroids[static_cast<long long>(row) * centroid_top_k + k] = top_idx[k];
        }
    }
}

__device__ __forceinline__ float warp_sum(float v) {
    const unsigned mask = 0xffffffffu;
    v += __shfl_down_sync(mask, v, 16);
    v += __shfl_down_sync(mask, v, 8);
    v += __shfl_down_sync(mask, v, 4);
    v += __shfl_down_sync(mask, v, 2);
    v += __shfl_down_sync(mask, v, 1);
    return v;
}

template <class T>
__global__ void masked_embedding_tile_argmax_pairs(
    const i32* __restrict__ top_centroids,
    const T* __restrict__ hidden_states,
    const T* __restrict__ lm_head_weight,
    const i64* __restrict__ token_ordering,
    u64* __restrict__ partial_pairs,
    i32 hidden,
    i32 centroid_top_k,
    i32 vocab_per_centroid,
    i32 selected,
    i32 num_tiles)
{
    const i32 row = blockIdx.x;
    const i32 tile = blockIdx.y;
    const i32 lane = threadIdx.x & 31;
    const i32 warp = threadIdx.x >> 5;
    const i32 selected_idx = tile * MASKED_TILE_TOKENS + warp;
    const auto* hidden_row = hidden_states + static_cast<long long>(row) * hidden;

    __shared__ float vals[MASKED_TILE_TOKENS];
    __shared__ i32 toks[MASKED_TILE_TOKENS];

    float dot = neg_inf();
    i32 tok = 0;
    if (warp < MASKED_TILE_TOKENS && selected_idx < selected) {
        const i32 centroid_rank = selected_idx / vocab_per_centroid;
        const i32 centroid_off = selected_idx - centroid_rank * vocab_per_centroid;
        if (centroid_rank < centroid_top_k) {
            const i32 centroid =
                top_centroids[static_cast<long long>(row) * centroid_top_k +
                              centroid_rank];
            const i64 tok64 =
                token_ordering[static_cast<long long>(centroid) * vocab_per_centroid +
                               centroid_off];
            tok = static_cast<i32>(tok64);
            const auto* wrow = lm_head_weight + static_cast<long long>(tok) * hidden;
            float sum = 0.f;
            for (i32 h = lane; h < hidden; h += 32) {
                sum += Elem<T>::to_f32(hidden_row[h]) * Elem<T>::to_f32(wrow[h]);
            }
            dot = warp_sum(sum);
        }
    }

    if (lane == 0) {
        vals[warp] = dot;
        toks[warp] = tok;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float best_val = vals[0];
        i32 best_tok = toks[0];
        for (i32 i = 1; i < MASKED_TILE_TOKENS; ++i) {
            update_argmax(vals[i], toks[i], best_val, best_tok);
        }
        partial_pairs[static_cast<usize>(tile) * gridDim.x + row] =
            pack_argmax_pair(best_val, best_tok);
    }
    (void)num_tiles;
}

/// The second half of every fused GEMV: fold one partial pair per tile into
/// one token per row.
///
/// Not a template -- there is no element type in it, only packed pairs -- and
/// no row: `num_tiles` is the block count its caller derived from
/// `cudaDevAttrMultiProcessorCount`, and a device query is not a `Source`.
__global__ void select_lm_head_argmax_pairs(
    const u64* __restrict__ partial_pairs,
    i32* __restrict__ out_tokens,
    i32 num_rows,
    i32 num_tiles)
{
    const i32 row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= num_rows) return;

    const u64 first = partial_pairs[row];
    float best_val = unpack_argmax_value(first);
    i32 best_tok = unpack_argmax_token(first);
    for (i32 tile = 1; tile < num_tiles; ++tile) {
        const u64 pair = partial_pairs[static_cast<usize>(tile) * num_rows + row];
        const float val = unpack_argmax_value(pair);
        const i32 tok = unpack_argmax_token(pair);
        if (val > best_val || (val == best_val && tok < best_tok)) {
            best_val = val;
            best_tok = tok;
        }
    }
    out_tokens[row] = best_tok;
}

// ---------------------------------------------------------------------------
// Vocab-streaming argmax: one slab at a time, `kAccumThreads` wide.
//
// The state carried between slabs is one pair per WARP rather than one per
// thread -- carrying per-thread state round-trips megabytes of scratch per
// slab and measured SLOWER than not chunking at all (§20.36).
// ---------------------------------------------------------------------------

__device__ __forceinline__ void warp_reduce_argmax(float& v, i32& i) {
    #pragma unroll
    for (i32 off = 16; off > 0; off >>= 1) {
        const float ov = __shfl_down_sync(0xffffffffu, v, off);
        const i32 oi = __shfl_down_sync(0xffffffffu, i, off);
        update_argmax(ov, oi, v, i);
    }
}

template <class T>
__global__ void argmax_accumulate(
    const T* __restrict__ slab,
    i32 width,
    i32 row_stride,
    i32 vocab_base,
    float* __restrict__ acc_val,
    i32* __restrict__ acc_idx,
    bool init,
    bool vectorised)
{
    const i32 row = blockIdx.x;
    const T* row_ptr = slab + static_cast<usize>(row) * row_stride;

    float best_val = neg_inf();
    i32 best_idx = 0;

    if (vectorised) {
        // 8 elements per 16-byte load. `vectorised` is decided host-side from
        // the row stride, so every row starts 16-byte aligned.
        const i32 vec_count = width >> 3;
        const uint4* vec_ptr = reinterpret_cast<const uint4*>(row_ptr);
        for (i32 v = threadIdx.x; v < vec_count; v += kAccumThreads) {
            const uint4 packed = vec_ptr[v];
            const auto* lane = reinterpret_cast<const T*>(&packed);
            #pragma unroll
            for (i32 t = 0; t < 8; ++t) {
                update_argmax(Elem<T>::to_f32(lane[t]),
                              vocab_base + (v << 3) + t, best_val, best_idx);
            }
        }
        for (i32 i = (vec_count << 3) + threadIdx.x; i < width; i += kAccumThreads) {
            update_argmax(Elem<T>::to_f32(row_ptr[i]), vocab_base + i,
                          best_val, best_idx);
        }
    } else {
        for (i32 i = threadIdx.x; i < width; i += kAccumThreads) {
            update_argmax(Elem<T>::to_f32(row_ptr[i]), vocab_base + i,
                          best_val, best_idx);
        }
    }

    warp_reduce_argmax(best_val, best_idx);
    if ((threadIdx.x & 31) != 0) return;

    const usize slot = static_cast<usize>(row) * kAccumWarps + (threadIdx.x >> 5);
    if (!init) {
        update_argmax(acc_val[slot], acc_idx[slot], best_val, best_idx);
    }
    acc_val[slot] = best_val;
    acc_idx[slot] = best_idx;
}

/// One warp, folding the per-warp slots of one row to a token.
__global__ void argmax_finalize(
    const float* __restrict__ acc_val,
    const i32* __restrict__ acc_idx,
    i32* __restrict__ token_ids)
{
    const i32 row = blockIdx.x;
    const usize slot = static_cast<usize>(row) * kAccumWarps + threadIdx.x;
    float best_val = acc_val[slot];
    i32 best_idx = acc_idx[slot];
    warp_reduce_argmax(best_val, best_idx);
    if (threadIdx.x == 0) token_ids[row] = best_idx;
}

// ---------------------------------------------------------------------------
// Fused GEMV + argmax: persistent blocks, eight warps, one vocab row each.
//
// Blocks walk the vocab with a grid stride and the hidden vector is staged in
// shared memory once. No rows: the grid comes from an occupancy query. See
// the header.
// ---------------------------------------------------------------------------

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (i32 offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_xor_sync(0xffffffffu, val, offset);
    }
    return val;
}

template <class T>
__global__ void lm_head_gemv_argmax_int8(
    const T* __restrict__ hidden_states,
    const i8* __restrict__ lm_head_weight,
    const float* __restrict__ scale_inv,
    u64* __restrict__ partial_pairs,
    i32 num_rows,
    i32 hidden,
    i32 vocab,
    i32 num_blocks_x)
{
    const i32 row = blockIdx.y;
    const i32 warp = threadIdx.x >> 5;
    const i32 lane = threadIdx.x & 31;

    extern __shared__ char shmem_raw[];
    float* sh_hidden = reinterpret_cast<float*>(shmem_raw);

    const auto* h_row = hidden_states + static_cast<long long>(row) * hidden;
    for (i32 i = threadIdx.x; i < hidden; i += GEMV_BLOCK_DIM) {
        sh_hidden[i] = Elem<T>::to_f32(h_row[i]);
    }
    __syncthreads();

    float best_val = neg_inf();
    i32 best_tok = -1;

    for (i32 v = blockIdx.x * GEMV_WARPS + warp; v < vocab;
         v += num_blocks_x * GEMV_WARPS) {
        const auto* w_row = lm_head_weight + static_cast<long long>(v) * hidden;
        float sum = 0.f;
        const i32 h_vec4 = hidden / 4;
        const auto* w_row_v4 = reinterpret_cast<const char4*>(w_row);
        for (i32 i = lane; i < h_vec4; i += 32) {
            const i32 h = i * 4;
            const char4 w4 = ldg(&w_row_v4[i]);
            sum += sh_hidden[h] * static_cast<float>(w4.x)
                 + sh_hidden[h + 1] * static_cast<float>(w4.y)
                 + sh_hidden[h + 2] * static_cast<float>(w4.z)
                 + sh_hidden[h + 3] * static_cast<float>(w4.w);
        }
        const float dot = warp_reduce_sum(sum) * scale_inv[v];

        if (lane == 0) {
            if (dot > best_val || (dot == best_val && v < best_tok)) {
                best_val = dot;
                best_tok = v;
            }
        }
    }

    __shared__ float sh_vals[GEMV_WARPS];
    __shared__ i32 sh_toks[GEMV_WARPS];
    if (lane == 0) {
        sh_vals[warp] = best_val;
        sh_toks[warp] = best_tok;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float bv = sh_vals[0];
        i32 bt = sh_toks[0];
        for (i32 w = 1; w < GEMV_WARPS; ++w) {
            update_argmax(sh_vals[w], sh_toks[w], bv, bt);
        }
        partial_pairs[static_cast<usize>(blockIdx.x) * num_rows + row] =
            pack_argmax_pair(bv, bt);
    }
}

template <class T>
__global__ void lm_head_gemv_argmax(
    const T* __restrict__ hidden_states,
    const T* __restrict__ lm_head_weight,
    u64* __restrict__ partial_pairs,
    i32 num_rows,
    i32 hidden,
    i32 vocab,
    i32 num_blocks_x)
{
    static_assert(is_same<T, bf16>::value,
                  "the weight row is read through `bf16x2`, a bf16 PAIR");
    const i32 row = blockIdx.y;
    const i32 warp = threadIdx.x >> 5;
    const i32 lane = threadIdx.x & 31;

    extern __shared__ char shmem_raw2[];
    float* sh_hidden = reinterpret_cast<float*>(shmem_raw2);

    const auto* h_row = hidden_states + static_cast<long long>(row) * hidden;
    for (i32 i = threadIdx.x; i < hidden; i += GEMV_BLOCK_DIM) {
        sh_hidden[i] = Elem<T>::to_f32(h_row[i]);
    }
    __syncthreads();

    float best_val = neg_inf();
    i32 best_tok = -1;

    for (i32 v = blockIdx.x * GEMV_WARPS + warp; v < vocab;
         v += num_blocks_x * GEMV_WARPS) {
        const auto* w_row = lm_head_weight + static_cast<long long>(v) * hidden;
        float sum = 0.f;
        const i32 h_vec2 = hidden / 2;
        const auto* w_row_v2 = reinterpret_cast<const bf16x2*>(w_row);
        for (i32 i = lane; i < h_vec2; i += 32) {
            const i32 h = i * 2;
            const bf16x2 w2 = ldg(&w_row_v2[i]);
            sum += sh_hidden[h] * bf16_to_f32(w2.x)
                 + sh_hidden[h + 1] * bf16_to_f32(w2.y);
        }
        const float dot = warp_reduce_sum(sum);

        if (lane == 0) {
            if (dot > best_val || (dot == best_val && v < best_tok)) {
                best_val = dot;
                best_tok = v;
            }
        }
    }

    __shared__ float sh_vals2[GEMV_WARPS];
    __shared__ i32 sh_toks2[GEMV_WARPS];
    if (lane == 0) {
        sh_vals2[warp] = best_val;
        sh_toks2[warp] = best_tok;
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        float bv = sh_vals2[0];
        i32 bt = sh_toks2[0];
        for (i32 w = 1; w < GEMV_WARPS; ++w) {
            update_argmax(sh_vals2[w], sh_toks2[w], bv, bt);
        }
        partial_pairs[static_cast<usize>(blockIdx.x) * num_rows + row] =
            pack_argmax_pair(bv, bt);
    }
}

}  // namespace pie::sample
