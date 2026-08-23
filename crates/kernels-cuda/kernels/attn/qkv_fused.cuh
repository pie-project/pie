//===-- qkv_fused.cuh - the three fused QKV epilogues ---------------------===//
//
// Three `__global__`s, all of them templates, and no host code. `qkv_fused.cu`
// includes this and keeps all FIVE `<<<>>>`, so the ahead-of-time build and
// NVRTC compile ONE text. §21.7 is the record of what two texts cost: fourteen
// kernels duplicated for a week with every gate green, because the split
// renamed them and the duplicate-name gate compares names.
//
// # What these three are
//
// Everything that happens between the QKV projection GEMM and attention, in
// one launch: unpack the fused `[q | k | v]` row, RMSNorm q and k per head,
// RoPE q and k, write k and v into the paged cache. The models here run this
// once per layer per step, so at decode shapes the standalone chain was
// almost entirely launch latency.
//
//  * `qkv_decode_qk_norm_rope_write_kv<BLOCK, USE_ROPE_TABLE>` -- one block
//    per (request, head). The general form: any head dim, `BLOCK` threads
//    striding it.
//  * `qkv_decode_qk_norm_rope_write_kv_warp<HEAD_DIM, USE_ROPE_TABLE>` -- one
//    WARP per (request, head), `HEAD_DIM` known at compile time so
//    `ELEMS_PER_THREAD` is a constant and the norm reduction is
//    `__shfl_xor_sync` with no shared memory and no `__syncthreads`. Chosen
//    for head_dim 64/128/256; the block form is the fallback.
//  * `qkv_packed_qk_norm_rope_vnorm_write_kv<BLOCK>` -- the prefill/packed
//    form, one block per (row, head), and it additionally RMSNorms v.
//
// # `USE_ROPE_TABLE` is a real arm, not a decoration
//
// `false` computes the angle with `powf` and `__sincosf`; `true` reads a
// precomputed `[max_pos, head_dim]` table. Those are different numbers --
// close, not equal -- so the parameter selects between two answers and §18's
// measurement applies directly: *a wrong specialisation arm was 99.83% of the
// right answer, 7 of 4,095 values moved and 0 of the 4,088 actually written.*
// A row that names the wrong arm would pass any tolerance loose enough to
// admit reassociation. The host picks it on `rope_table != nullptr` and both
// values are instantiated and reachable.
//
// `BLOCK` and `HEAD_DIM` are equally real: `BLOCK` sizes `__shared__ float
// buf[BLOCK]` and fixes the halving reduction over it; `HEAD_DIM` fixes
// `ELEMS_PER_THREAD = HEAD_DIM / 32` and every `#pragma unroll` under it.
//
// # Which launcher becomes a row, and which does not
//
// **All five do now.** This section opened *"None of the five, and for one
// reason each"*, and closed *"All five are stated below for whoever writes
// them, with the line they were read from, because `runtime::launch` says a
// rule with no cited launcher is a guess."* Someone wrote them, from those
// lines. Both reasons named were geometric and both were answered:
//
//  * The two warp-form launches (`<<<warp_grid, 256>>>`) size the grid as
//    `ceil(num_requests * (num_q_heads + num_kv_heads) / (256/32))` -- units
//    of WARPS, not blocks, not rows, not heads. No rule divided a product of
//    two head counts by a warps-per-block. `LaunchRule::WarpPackedHeads`
//    does, cited at `qkv_fused.cu:51-53, :57-58, :70-71`.
//  * The two block-form launches and the packed one use `dim3 grid(rows,
//    num_q_heads + num_kv_heads)` -- a second axis that is the SUM of two head
//    counts. `LaunchRule::RowsPackedHeads` (256-wide, `qkv_fused.cu:245-248`)
//    and `RowsPackedHeadsNarrow` (128-wide, `:98-102`, `:126-127`) state it.
//    The two differ only in block width, and that is not a tuning difference:
//    the halving reduction is `__shfl_xor_sync` at one width and `__shared__`
//    at the other.
//
// # What is still NOT reproduced here, and it is not the grid
//
// Two selectors on these kernels remain unspellable, and both are recorded
// where the rows are rather than here:
//
//  * `USE_ROPE_TABLE` is `rope_table != nullptr`, a pointer-null test.
//    `Term::Aligned` **holds of address 0**, so an alignment clause picks the
//    table arm for a fire that has no table -- measured. `Term::Present`
//    exists for this and reads `Fact::Address`.
//  * `HEAD_DIM` is chosen by `if (head_dim == 64 / 128 / 256)` at
//    `qkv_fused.cu:81, :85, :89`. `Term::Multiple { of: 64 }` holds of 192
//    too, so an ordered arm list would send a 192-wide head to the 64
//    expansion, where `ELEMS_PER_THREAD` is 2 and 6 is needed -- a wrong
//    answer, not a fault. **The decode rows therefore PIN 128** and are not
//    dispatchable until a term can say "exactly this value".
//
// # Linkage
//
// **Template-only.** All three are templates, so no host stub takes external
// linkage until something instantiates one, and this header has NO
// single-includer constraint -- unlike the `write_mla` half of
// `mla_paged.cuh`, which holds a non-template `__global__` and may be included
// by exactly one translation unit (§21.6). Nothing was templated here to get
// that; all three arrived as templates.
//
// This list used to open with `pack_dense_mask.cuh`, which held two more
// non-template `__global__`s under the same constraint. It is gone: the driver
// packs the element bitmap on the HOST (`driver-cuda/src/fire/page_mask.rs`,
// `mask[base + (index >> 3)] |= 1 << (index & 7)`, with tests that read it
// back the way the kernels index it), so the device packers were never fired
// and nothing included them.
//
// # NVRTC
//
// The only include is the prelude. `__sincosf`, `powf`, `rsqrtf` and
// `__shfl_xor_sync` are builtins NVRTC accepts without a header -- measured,
// same as `dsa_indexer.cuh` records. `<cstdint>` was the one external include
// and it is gone: the fixed-width names come from `pie`.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "prelude/device.cuh"

namespace pie::attn {

template <int BLOCK, bool USE_ROPE_TABLE>
__global__ void qkv_decode_qk_norm_rope_write_kv(
    const bf16* __restrict__ packed,
    bf16* __restrict__ q_out,
    bf16* __restrict__ k_pages,
    bf16* __restrict__ v_pages,
    const bf16* __restrict__ q_weight,
    const bf16* __restrict__ k_weight,
    const i32* __restrict__ positions,
    const float* __restrict__ rope_table,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    const u32* __restrict__ w_page,
    const u32* __restrict__ w_off,
    const u8* __restrict__ row_valid,
    const u32* __restrict__ win,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    bool hnd_layout,
    float theta,
    float eps)
{
    const int r = blockIdx.x;
    // Peel device window (prefix form): this kernel owns rows
    // [0, win[0]) — the hook-free prefix — and the grid spans the full
    // lane count so a captured launch replays across row splits. The
    // early-out is uniform per block (r is blockIdx.x) and sits before
    // any __syncthreads, so the shared reduction never diverges.
    if (win != nullptr && r >= static_cast<int>(win[0])) return;
    const int head_idx = blockIdx.y;
    const bool is_q = head_idx < num_q_heads;
    if (!is_q && row_valid != nullptr && row_valid[r] == 0) return;
    const int local_head = is_q ? head_idx : (head_idx - num_q_heads);
    const int q_dim = num_q_heads * head_dim;
    const int kv_dim = num_kv_heads * head_dim;
    const int packed_stride = q_dim + 2 * kv_dim;
    const bf16* src_row =
        packed + static_cast<long long>(r) * packed_stride;
    const bf16* src = is_q
        ? src_row + local_head * head_dim
        : src_row + q_dim + local_head * head_dim;
    const bf16* weight = is_q ? q_weight : k_weight;

    float local = 0.f;
    for (int i = threadIdx.x; i < head_dim; i += BLOCK) {
        const float v = bf16_to_f32(src[i]);
        local += v * v;
    }

    __shared__ float buf[BLOCK];
    buf[threadIdx.x] = local;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) buf[threadIdx.x] += buf[threadIdx.x + off];
        __syncthreads();
    }

    bf16* dst = nullptr;
    bf16* v_dst = nullptr;
    if (is_q) {
        dst = q_out + (static_cast<long long>(r) * num_q_heads + local_head) *
                      head_dim;
    } else {
        int actual_page;
        int offset_in_page;
        if (w_page != nullptr && w_off != nullptr) {
            actual_page = static_cast<int>(w_page[r]);
            offset_in_page = static_cast<int>(w_off[r]);
        } else {
            const int pages_first = kv_page_indptr[r];
            const int pages_last = kv_page_indptr[r + 1];
            const int num_pages_r = pages_last - pages_first;
            const int abs_kv_pos =
                (num_pages_r - 1) * page_size +
                static_cast<int>(kv_last_page_lens[r]) - 1;
            const int page_in_req = abs_kv_pos / page_size;
            offset_in_page = abs_kv_pos % page_size;
            actual_page = static_cast<int>(
                kv_page_indices[pages_first + page_in_req]);
        }
        if (hnd_layout) {
            const long long page_row =
                ((static_cast<long long>(actual_page) * num_kv_heads +
                  local_head) * page_size + offset_in_page) * head_dim;
            dst = k_pages + page_row;
            v_dst = v_pages + page_row;
        } else {
            const long long page_row =
                ((static_cast<long long>(actual_page) * page_size) +
                 offset_in_page) * kv_dim;
            dst = k_pages + page_row + local_head * head_dim;
            v_dst = v_pages + page_row + local_head * head_dim;
        }
    }

    if (!is_q) {
        const bf16* v_src =
            src_row + q_dim + kv_dim + local_head * head_dim;
        for (int i = threadIdx.x; i < head_dim; i += BLOCK) {
            v_dst[i] = v_src[i];
        }
    }

    const float inv_rms = rsqrtf(buf[0] / static_cast<float>(head_dim) + eps);
    const int half = head_dim / 2;
    const float* rope_row = nullptr;
    int pos = 0;
    if constexpr (USE_ROPE_TABLE) {
        rope_row = rope_table + static_cast<long long>(r) * head_dim;
    } else {
        pos = positions[r];
    }
    for (int dim_pair = threadIdx.x; dim_pair < half; dim_pair += BLOCK) {
        const float a = bf16_to_f32(src[dim_pair]) *
            inv_rms * bf16_to_f32(weight[dim_pair]);
        const float b = bf16_to_f32(src[dim_pair + half]) *
            inv_rms * bf16_to_f32(weight[dim_pair + half]);
        float cos_v, sin_v;
        if constexpr (USE_ROPE_TABLE) {
            cos_v = rope_row[dim_pair];
            sin_v = rope_row[dim_pair + half];
        } else {
            const float freq = powf(
                theta,
                -2.f * static_cast<float>(dim_pair) /
                    static_cast<float>(head_dim));
            const float ang = static_cast<float>(pos) * freq;
            __sincosf(ang, &sin_v, &cos_v);
        }
        dst[dim_pair] = f32_to_bf16(a * cos_v - b * sin_v);
        dst[dim_pair + half] = f32_to_bf16(b * cos_v + a * sin_v);
    }
}

template <int HEAD_DIM, bool USE_ROPE_TABLE>
__global__ void qkv_decode_qk_norm_rope_write_kv_warp(
    const bf16* __restrict__ packed,
    bf16* __restrict__ q_out,
    bf16* __restrict__ k_pages,
    bf16* __restrict__ v_pages,
    const bf16* __restrict__ q_weight,
    const bf16* __restrict__ k_weight,
    const i32* __restrict__ positions,
    const float* __restrict__ rope_table,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    const u32* __restrict__ w_page,
    const u32* __restrict__ w_off,
    const u8* __restrict__ row_valid,
    const u32* __restrict__ win,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int page_size,
    bool hnd_layout,
    float theta,
    float eps)
{
    constexpr unsigned FULL_MASK = 0xffffffffu;
    constexpr int ELEMS_PER_THREAD = HEAD_DIM / 32;
    static_assert(HEAD_DIM % 64 == 0);

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;
    const int total_qk_heads = num_q_heads + num_kv_heads;
    const int unit = blockIdx.x * warps_per_block + warp_id;
    if (unit >= num_requests * total_qk_heads) return;

    const int r = unit / total_qk_heads;
    // Peel device window (prefix form): rows [0, win[0]) only. One warp
    // is one (row, head) unit, so the early-out is warp-uniform and the
    // FULL_MASK shuffles below never see a partial warp.
    if (win != nullptr && r >= static_cast<int>(win[0])) return;
    const int head_idx = unit - r * total_qk_heads;
    const bool is_q = head_idx < num_q_heads;
    if (!is_q && row_valid != nullptr && row_valid[r] == 0) return;
    const int local_head = is_q ? head_idx : (head_idx - num_q_heads);
    const int q_dim = num_q_heads * HEAD_DIM;
    const int kv_dim = num_kv_heads * HEAD_DIM;
    const int packed_stride = q_dim + 2 * kv_dim;
    const bf16* src_row =
        packed + static_cast<long long>(r) * packed_stride;
    const bf16* src = is_q
        ? src_row + local_head * HEAD_DIM
        : src_row + q_dim + local_head * HEAD_DIM;
    const bf16* weight = is_q ? q_weight : k_weight;

    float vals[ELEMS_PER_THREAD];
    float sum = 0.f;
#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        const int dim = lane * ELEMS_PER_THREAD + i;
        const float v = bf16_to_f32(src[dim]);
        vals[i] = v;
        sum += v * v;
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(FULL_MASK, sum, offset, 32);
    }

    const float inv_rms =
        rsqrtf(sum / static_cast<float>(HEAD_DIM) + eps);
#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        const int dim = lane * ELEMS_PER_THREAD + i;
        vals[i] *= inv_rms * bf16_to_f32(weight[dim]);
    }

    const int pair_offset = (HEAD_DIM / 2) / ELEMS_PER_THREAD;
    const float* rope_row = nullptr;
    int pos = 0;
    if constexpr (USE_ROPE_TABLE) {
        rope_row = rope_table + static_cast<long long>(r) * HEAD_DIM;
    } else {
        pos = positions[r];
    }
#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        const int dim = lane * ELEMS_PER_THREAD + i;
        const float pair = __shfl_xor_sync(FULL_MASK, vals[i], pair_offset, 32);
        const float signed_pair = (lane < pair_offset) ? -pair : pair;
        const int dim_pair = (dim * 2) % HEAD_DIM / 2;
        float cos_v, sin_v;
        if constexpr (USE_ROPE_TABLE) {
            cos_v = rope_row[dim_pair];
            sin_v = rope_row[dim_pair + HEAD_DIM / 2];
        } else {
            const float freq = powf(
                theta,
                -2.f * static_cast<float>(dim_pair) /
                    static_cast<float>(HEAD_DIM));
            const float ang = static_cast<float>(pos) * freq;
            __sincosf(ang, &sin_v, &cos_v);
        }
        vals[i] = vals[i] * cos_v + signed_pair * sin_v;
    }

    bf16* dst = nullptr;
    bf16* v_dst = nullptr;
    if (is_q) {
        dst = q_out + (static_cast<long long>(r) * num_q_heads + local_head) *
                      HEAD_DIM;
    } else {
        int actual_page;
        int offset_in_page;
        if (w_page != nullptr && w_off != nullptr) {
            actual_page = static_cast<int>(w_page[r]);
            offset_in_page = static_cast<int>(w_off[r]);
        } else {
            const int pages_first = kv_page_indptr[r];
            const int pages_last = kv_page_indptr[r + 1];
            const int num_pages_r = pages_last - pages_first;
            const int abs_kv_pos =
                (num_pages_r - 1) * page_size +
                static_cast<int>(kv_last_page_lens[r]) - 1;
            const int page_in_req = abs_kv_pos / page_size;
            offset_in_page = abs_kv_pos % page_size;
            actual_page = static_cast<int>(
                kv_page_indices[pages_first + page_in_req]);
        }
        if (hnd_layout) {
            const long long page_row =
                ((static_cast<long long>(actual_page) * num_kv_heads +
                  local_head) * page_size + offset_in_page) * HEAD_DIM;
            dst = k_pages + page_row;
            v_dst = v_pages + page_row;
        } else {
            const long long page_row =
                ((static_cast<long long>(actual_page) * page_size) +
                 offset_in_page) * kv_dim;
            dst = k_pages + page_row + local_head * HEAD_DIM;
            v_dst = v_pages + page_row + local_head * HEAD_DIM;
        }
    }

#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        const int dim = lane * ELEMS_PER_THREAD + i;
        dst[dim] = f32_to_bf16(vals[i]);
    }
    if (!is_q) {
        const bf16* v_src =
            src_row + q_dim + kv_dim + local_head * HEAD_DIM;
#pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            const int dim = lane * ELEMS_PER_THREAD + i;
            v_dst[dim] = v_src[dim];
        }
    }
}

template <int BLOCK>
__global__ void qkv_packed_qk_norm_rope_vnorm_write_kv(
    const bf16* __restrict__ packed,
    bf16* __restrict__ q_out,
    bf16* __restrict__ k_pages,
    bf16* __restrict__ v_pages,
    const bf16* __restrict__ q_weight,
    const bf16* __restrict__ k_weight,
    const i32* __restrict__ positions,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    const u8* __restrict__ row_valid,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    bool hnd_layout,
    float theta,
    float eps)
{
    const int row = blockIdx.x;
    const int head_idx = blockIdx.y;
    const bool is_q = head_idx < num_q_heads;
    if (!is_q && row_valid != nullptr && row_valid[row] == 0) return;
    const int local_head = is_q ? head_idx : (head_idx - num_q_heads);
    const int q_dim = num_q_heads * head_dim;
    const int kv_dim = num_kv_heads * head_dim;
    const int packed_stride = q_dim + 2 * kv_dim;
    const bf16* src_row =
        packed + static_cast<long long>(row) * packed_stride;
    const bf16* src = is_q
        ? src_row + local_head * head_dim
        : src_row + q_dim + local_head * head_dim;
    const bf16* weight = is_q ? q_weight : k_weight;

    float local = 0.f;
    float local_v = 0.f;
    const bf16* v_src = nullptr;
    if (!is_q) {
        v_src = src_row + q_dim + kv_dim + local_head * head_dim;
    }
    for (int i = threadIdx.x; i < head_dim; i += BLOCK) {
        const float v = bf16_to_f32(src[i]);
        local += v * v;
        if (!is_q) {
            const float vv = bf16_to_f32(v_src[i]);
            local_v += vv * vv;
        }
    }

    __shared__ float buf[BLOCK];
    __shared__ float buf_v[BLOCK];
    buf[threadIdx.x] = local;
    buf_v[threadIdx.x] = local_v;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) {
            buf[threadIdx.x] += buf[threadIdx.x + off];
            buf_v[threadIdx.x] += buf_v[threadIdx.x + off];
        }
        __syncthreads();
    }

    bf16* dst = nullptr;
    bf16* v_dst = nullptr;
    if (is_q) {
        dst = q_out + (static_cast<long long>(row) * num_q_heads + local_head) *
                      head_dim;
    } else {
        const int pages_first = kv_page_indptr[row];
        const int pages_last = kv_page_indptr[row + 1];
        const int num_pages_r = pages_last - pages_first;
        const int abs_kv_pos =
            (num_pages_r - 1) * page_size +
            static_cast<int>(kv_last_page_lens[row]) - 1;
        const int page_in_req = abs_kv_pos / page_size;
        const int offset_in_page = abs_kv_pos % page_size;
        const int actual_page =
            static_cast<int>(kv_page_indices[pages_first + page_in_req]);
        if (hnd_layout) {
            const long long page_row =
                ((static_cast<long long>(actual_page) * num_kv_heads +
                  local_head) * page_size + offset_in_page) * head_dim;
            dst = k_pages + page_row;
            v_dst = v_pages + page_row;
        } else {
            const long long page_row =
                ((static_cast<long long>(actual_page) * page_size) +
                 offset_in_page) * kv_dim;
            dst = k_pages + page_row + local_head * head_dim;
            v_dst = v_pages + page_row + local_head * head_dim;
        }
    }

    const float inv_rms =
        rsqrtf(buf[0] / static_cast<float>(head_dim) + eps);
    const int half = head_dim / 2;
    const int pos = positions[row];
    for (int dim_pair = threadIdx.x; dim_pair < half; dim_pair += BLOCK) {
        const bf16 norm_a = f32_to_bf16(
            bf16_to_f32(src[dim_pair]) *
            inv_rms * bf16_to_f32(weight[dim_pair]));
        const bf16 norm_b = f32_to_bf16(
            bf16_to_f32(src[dim_pair + half]) *
            inv_rms * bf16_to_f32(weight[dim_pair + half]));
        const float a = bf16_to_f32(norm_a);
        const float b = bf16_to_f32(norm_b);
        const float freq = powf(
            theta,
            -2.f * static_cast<float>(dim_pair) /
                static_cast<float>(head_dim));
        const float ang = static_cast<float>(pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);
        dst[dim_pair] = f32_to_bf16(a * cos_v - b * sin_v);
        dst[dim_pair + half] = f32_to_bf16(b * cos_v + a * sin_v);
    }

    if (!is_q) {
        const float inv_v =
            rsqrtf(buf_v[0] / static_cast<float>(head_dim) + eps);
        for (int i = threadIdx.x; i < head_dim; i += BLOCK) {
            v_dst[i] = f32_to_bf16(bf16_to_f32(v_src[i]) * inv_v);
        }
    }
}

}  // namespace pie::attn
