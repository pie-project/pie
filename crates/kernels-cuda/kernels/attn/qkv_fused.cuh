#pragma once

#include <cuda_fp8.h>

#include "prelude/device.cuh"

namespace pie::custom {

// A head norm's weight at `i`, or 1 for a scaleless norm.
__device__ __forceinline__ float norm_weight(const bf16* __restrict__ w, int i) {
    return w != nullptr ? bf16_to_f32(w[i]) : 1.f;
}

template <int E>
__device__ __forceinline__ void store_bf16_run(bf16* __restrict__ at, const float (&in)[E]);

// A head's landing row: the query output, or a key or value page row as the
// pool holds it (bf16, or e4m3 under a unit scale).
template <class KV>
struct HeadDst {
    bf16* q = nullptr;
    KV* kv = nullptr;

    __device__ __forceinline__ void put(int i, float x) const {
        if (q != nullptr) {
            q[i] = f32_to_bf16(x);
        } else if constexpr (is_same<KV, bf16>::value) {
            kv[i] = f32_to_bf16(x);
        } else {
            kv[i] = KV(x);
        }
    }

    template <int E>
    __device__ __forceinline__ void run(int at, const float (&in)[E]) const {
        if (q != nullptr) {
            store_bf16_run<E>(q + at, in);
        } else if constexpr (is_same<KV, bf16>::value) {
            store_bf16_run<E>(kv + at, in);
        } else {
#pragma unroll
            for (int i = 0; i < E; ++i) kv[at + i] = KV(in[i]);
        }
    }
};

template <int BLOCK, bool USE_ROPE_TABLE, class KV = bf16>
__global__ void qkv_decode_qk_norm_rope_vnorm_write_kv(
    const bf16* __restrict__ packed,
    bf16* __restrict__ q_out,
    KV* __restrict__ k_pages,
    KV* __restrict__ v_pages,
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
    float eps,
    bool v_norm,
    int rotary_dim)
{
    const int r = blockIdx.x;

    if (win != nullptr && r >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? r + static_cast<int>(win[1]) : r;
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

    const bf16* v_src =
        is_q ? nullptr : src_row + q_dim + kv_dim + local_head * head_dim;
    float local = 0.f;
    float local_v = 0.f;
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

    HeadDst<KV> dst;
    HeadDst<KV> v_dst;
    if (is_q) {
        dst.q = q_out + (static_cast<long long>(row) * num_q_heads + local_head) *
                      head_dim;
    } else {
        int actual_page;
        int offset_in_page;
        if (w_page != nullptr && w_off != nullptr) {
            actual_page = static_cast<int>(w_page[row]);
            offset_in_page = static_cast<int>(w_off[row]);
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
            dst.kv = k_pages + page_row;
            v_dst.kv = v_pages + page_row;
        } else {
            const long long page_row =
                ((static_cast<long long>(actual_page) * page_size) +
                 offset_in_page) * kv_dim;
            dst.kv = k_pages + page_row + local_head * head_dim;
            v_dst.kv = v_pages + page_row + local_head * head_dim;
        }
    }

    if (!is_q) {
        const float inv_v = v_norm ? rsqrtf(buf_v[0] / static_cast<float>(head_dim) + eps) : 1.f;
        for (int i = threadIdx.x; i < head_dim; i += BLOCK) {
            v_dst.put(i, bf16_to_f32(v_src[i]) * inv_v);
        }
    }

    const float inv_rms = rsqrtf(buf[0] / static_cast<float>(head_dim) + eps);
    const int half = head_dim / 2;
    const float* rope_row = nullptr;
    int pos = 0;
    if constexpr (USE_ROPE_TABLE) {
        rope_row = rope_table + static_cast<long long>(row) * head_dim;
    } else {
        pos = positions != nullptr ? positions[row] : 0;
    }
    for (int dim_pair = threadIdx.x; dim_pair < half; dim_pair += BLOCK) {
        const float a = bf16_to_f32(src[dim_pair]) *
            inv_rms * norm_weight(weight, dim_pair);
        const float b = bf16_to_f32(src[dim_pair + half]) *
            inv_rms * norm_weight(weight, dim_pair + half);
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

        if (dim_pair >= rotary_dim / 2) {
            cos_v = 1.f;
            sin_v = 0.f;
        }
        dst.put(dim_pair, a * cos_v - b * sin_v);
        dst.put(dim_pair + half, b * cos_v + a * sin_v);
    }
}

template <int HEAD_DIM, bool USE_ROPE_TABLE, class KV = bf16>
__global__ void qkv_decode_qk_norm_rope_vnorm_write_kv_warp(
    const bf16* __restrict__ packed,
    bf16* __restrict__ q_out,
    KV* __restrict__ k_pages,
    KV* __restrict__ v_pages,
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
    float eps,
    bool v_norm,
    int rotary_dim,
    int span,
    bool qk_norm,
    float v_scale,
    int qk_width,
    int v_width,
    int rest_width,
    bf16* __restrict__ rest)
{
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    constexpr unsigned FULL_MASK = 0xffffffffu;
    constexpr int ELEMS_PER_THREAD = HEAD_DIM / 32;
    static_assert(HEAD_DIM % 64 == 0);
    // The decode attention after it (`kernel.pdl`) reads its plan meanwhile
    // and waits for these rows.
    pdl_launch();

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;
    const int total_qk_heads = num_q_heads + num_kv_heads;
    const int unit = blockIdx.x * warps_per_block + warp_id;
    if (unit >= num_requests * total_qk_heads) return;

    const int r = unit / total_qk_heads;

    if (win != nullptr && r >= static_cast<int>(win[0])) return;

    const int row = win != nullptr ? r + static_cast<int>(win[1]) : r;

    const int head_idx = unit - r * total_qk_heads;
    const bool is_q = head_idx < num_q_heads;
    if (!is_q && row_valid != nullptr && row_valid[row] == 0) return;
    const int local_head = is_q ? head_idx : (head_idx - num_q_heads);
    // The packed row's heads at their own widths (`qk_width`, `v_width`),
    // each landing HEAD_DIM wide, zero past its width.
    const int kv_dim = num_kv_heads * HEAD_DIM;
    const int qkv_width = num_q_heads * qk_width + num_kv_heads * (qk_width + v_width);
    const int packed_stride = qkv_width + rest_width;
    const bf16* src_row =
        packed + static_cast<long long>(row) * packed_stride;
    // The row's columns past its qkv (`rest`, a whole number of 16-byte
    // words), a q head's share each.
    if (is_q && rest_width > 0) {
        const int per = ((rest_width + num_q_heads - 1) / num_q_heads + 7) / 8 * 8;
        const int end = min((local_head + 1) * per, rest_width);
        for (int c = local_head * per + lane * 8; c < end; c += 256) {
            *reinterpret_cast<uint4*>(rest + static_cast<long long>(row) * rest_width + c) =
                *reinterpret_cast<const uint4*>(src_row + qkv_width + c);
        }
    }
    const bf16* src = is_q
        ? src_row + local_head * qk_width
        : src_row + num_q_heads * qk_width + local_head * qk_width;
    const bf16* weight = is_q ? q_weight : k_weight;

    const bf16* v_src = is_q
        ? nullptr
        : src_row + (num_q_heads + num_kv_heads) * qk_width + local_head * v_width;
    float vals[ELEMS_PER_THREAD];
    float v_vals[ELEMS_PER_THREAD];
    float sum = 0.f;
    float v_sum = 0.f;
#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        const int dim = lane * ELEMS_PER_THREAD + i;
        const float v = dim < qk_width ? bf16_to_f32(src[dim]) : 0.f;
        vals[i] = v;
        sum += v * v;
        v_vals[i] = is_q || dim >= v_width ? 0.f : bf16_to_f32(v_src[dim]);
        v_sum += v_vals[i] * v_vals[i];
    }
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        sum += __shfl_xor_sync(FULL_MASK, sum, offset, 32);
        v_sum += __shfl_xor_sync(FULL_MASK, v_sum, offset, 32);
    }

    const float inv_rms =
        rsqrtf(sum / static_cast<float>(HEAD_DIM) + eps);
#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        const int dim = lane * ELEMS_PER_THREAD + i;
        if (qk_norm) vals[i] *= inv_rms * norm_weight(weight, dim);
    }

    // Pairs `span` apart turn at theta^(-i/span), the first rotary_dim/2 of
    // them; entries past the first 2 * span stay.
    const int pair_offset = span / ELEMS_PER_THREAD;
    const float* rope_row = nullptr;
    int pos = 0;
    if constexpr (USE_ROPE_TABLE) {
        rope_row = rope_table + static_cast<long long>(row) * HEAD_DIM;
    } else {
        pos = positions != nullptr ? positions[row] : 0;
    }
#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        const int dim = lane * ELEMS_PER_THREAD + i;
        const float pair = __shfl_xor_sync(FULL_MASK, vals[i], pair_offset, 32);
        const int in_two = dim & (2 * span - 1);
        const float signed_pair = in_two < span ? -pair : pair;
        const int dim_pair = in_two & (span - 1);
        float cos_v, sin_v;
        if constexpr (USE_ROPE_TABLE) {
            cos_v = rope_row[dim_pair];
            sin_v = rope_row[dim_pair + HEAD_DIM / 2];
        } else {
            const float freq = powf(
                theta,
                -static_cast<float>(dim_pair) / static_cast<float>(span));
            const float ang = static_cast<float>(pos) * freq;
            __sincosf(ang, &sin_v, &cos_v);
        }

        if (dim >= 2 * span || dim_pair >= rotary_dim / 2) {
            cos_v = 1.f;
            sin_v = 0.f;
        }
        vals[i] = vals[i] * cos_v + signed_pair * sin_v;
    }

    HeadDst<KV> dst;
    HeadDst<KV> v_dst;
    if (is_q) {
        dst.q = q_out + (static_cast<long long>(row) * num_q_heads + local_head) *
                      HEAD_DIM;
    } else {
        int actual_page;
        int offset_in_page;
        if (w_page != nullptr && w_off != nullptr) {
            actual_page = static_cast<int>(w_page[row]);
            offset_in_page = static_cast<int>(w_off[row]);
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
            dst.kv = k_pages + page_row;
            v_dst.kv = v_pages + page_row;
        } else {
            const long long page_row =
                ((static_cast<long long>(actual_page) * page_size) +
                 offset_in_page) * kv_dim;
            dst.kv = k_pages + page_row + local_head * HEAD_DIM;
            v_dst.kv = v_pages + page_row + local_head * HEAD_DIM;
        }
    }

#pragma unroll
    for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
        const int dim = lane * ELEMS_PER_THREAD + i;
        dst.put(dim, vals[i]);
    }
    if (!is_q) {
        const float inv_v = v_norm ? rsqrtf(v_sum / static_cast<float>(HEAD_DIM) + eps) : 1.f;
#pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            const int dim = lane * ELEMS_PER_THREAD + i;
            v_dst.put(dim, v_vals[i] * inv_v * v_scale);
        }
    }
}

template <int E>
__device__ __forceinline__ void load_bf16_run(const bf16* __restrict__ at, float (&out)[E]) {
    if constexpr (E % 8 == 0) {
#pragma unroll
        for (int c = 0; c < E / 8; ++c) {
            const uint4 v = reinterpret_cast<const uint4*>(at)[c];
            const unsigned int w[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                out[c * 8 + 2 * k] = bf16_to_f32(bf16{static_cast<unsigned short>(w[k] & 0xffffu)});
                out[c * 8 + 2 * k + 1] = bf16_to_f32(bf16{static_cast<unsigned short>(w[k] >> 16)});
            }
        }
    } else if constexpr (E == 4) {
        const uint2 v = *reinterpret_cast<const uint2*>(at);
        const unsigned int w[2] = {v.x, v.y};
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            out[2 * k] = bf16_to_f32(bf16{static_cast<unsigned short>(w[k] & 0xffffu)});
            out[2 * k + 1] = bf16_to_f32(bf16{static_cast<unsigned short>(w[k] >> 16)});
        }
    } else {
#pragma unroll
        for (int i = 0; i < E; ++i) {
            out[i] = bf16_to_f32(at[i]);
        }
    }
}

template <int E>
__device__ __forceinline__ void store_bf16_run(bf16* __restrict__ at, const float (&in)[E]) {
    if constexpr (E % 8 == 0) {
#pragma unroll
        for (int c = 0; c < E / 8; ++c) {
            unsigned int w[4];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                w[k] = static_cast<unsigned int>(f32_to_bf16(in[c * 8 + 2 * k]).raw)
                    | (static_cast<unsigned int>(f32_to_bf16(in[c * 8 + 2 * k + 1]).raw) << 16);
            }
            reinterpret_cast<uint4*>(at)[c] = make_uint4(w[0], w[1], w[2], w[3]);
        }
    } else if constexpr (E == 4) {
        unsigned int w[2];
#pragma unroll
        for (int k = 0; k < 2; ++k) {
            w[k] = static_cast<unsigned int>(f32_to_bf16(in[2 * k]).raw)
                | (static_cast<unsigned int>(f32_to_bf16(in[2 * k + 1]).raw) << 16);
        }
        *reinterpret_cast<uint2*>(at) = make_uint2(w[0], w[1]);
    } else {
#pragma unroll
        for (int i = 0; i < E; ++i) {
            at[i] = f32_to_bf16(in[i]);
        }
    }
}

// The warp kernel above, a block a row: the row's rotation is computed once
// into shared memory rather than once a head, and each head's run moves in
// whole 16-byte words. With E = 8 entries a lane (a 64- or 128-wide head's
// word) a head takes part of the warp and the warp several heads at once,
// its sums adding in another order than the warp kernel's. With STAGE
// (sm_90 on) the whole packed row lands in shared memory by one bulk copy
// and the rest leaves by another, so a head's loads wait no round trip
// each.
template <int HEAD_DIM, int E = HEAD_DIM / 32, class KV = bf16, bool STAGE = false>
__global__ void qkv_qk_norm_rope_vnorm_write_kv_rows(
    const bf16* __restrict__ packed,
    bf16* __restrict__ q_out,
    KV* __restrict__ k_pages,
    KV* __restrict__ v_pages,
    const bf16* __restrict__ q_weight,
    const bf16* __restrict__ k_weight,
    const i32* __restrict__ positions,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    const u32* __restrict__ w_page,
    const u32* __restrict__ w_off,
    const u8* __restrict__ row_valid,
    const u32* __restrict__ win,
    int num_q_heads,
    int num_kv_heads,
    int page_size,
    bool hnd_layout,
    float theta,
    float eps,
    bool v_norm,
    int rotary_dim,
    int span,
    bool qk_norm,
    float v_scale,
    int qk_width,
    int v_width,
    int rest_width,
    bf16* __restrict__ rest)
{
    constexpr unsigned FULL_MASK = 0xffffffffu;
    constexpr int ELEMS_PER_THREAD = E;
    constexpr int L = HEAD_DIM / E;  // lanes a head
    constexpr int G = 32 / L;        // heads a warp
    static_assert(L * E == HEAD_DIM && G * L == 32);
    constexpr int HALF = HEAD_DIM / 2;
    static_assert(HEAD_DIM % 64 == 0);

    __shared__ float rotation[HEAD_DIM];

    const int r = blockIdx.x;
    if (win != nullptr && r >= static_cast<int>(win[0])) return;
    const int row = win != nullptr ? r + static_cast<int>(win[1]) : r;
    extern __shared__ __align__(16) unsigned char qkv_stage[];
    const int row_bytes =
        (num_q_heads * qk_width + num_kv_heads * (qk_width + v_width) + rest_width) * 2;
    unsigned long long* staged = reinterpret_cast<unsigned long long*>(qkv_stage + row_bytes);
    const unsigned staged_at = static_cast<unsigned>(__cvta_generic_to_shared(staged));
    const unsigned stage_at = static_cast<unsigned>(__cvta_generic_to_shared(qkv_stage));
    if (STAGE && threadIdx.x == 0) {
#if __CUDA_ARCH__ >= 900
        asm volatile("mbarrier.init.shared::cta.b64 [%0], 1;" ::"r"(staged_at));
        asm volatile("fence.mbarrier_init.release.cluster;" ::: "memory");
        asm volatile("mbarrier.arrive.expect_tx.shared::cta.b64 _, [%0], %1;" ::"r"(staged_at), "r"(row_bytes)
                     : "memory");
        asm volatile(
            "cp.async.bulk.shared::cluster.global.mbarrier::complete_tx::bytes [%0], [%1], %2, [%3];" ::"r"(
                stage_at),
            "l"(packed + static_cast<long long>(row) * (row_bytes / 2)), "r"(row_bytes), "r"(staged_at)
            : "memory");
#endif
    } else if (threadIdx.x == 0) {
#if __CUDA_ARCH__ >= 900
        // The packed row on its way into L2 under the rotation's setup (the
        // bulk prefetch is sm_90's; below it the row is read as it comes).
        const int width = num_q_heads * qk_width + num_kv_heads * (qk_width + v_width) + rest_width;
        asm volatile("cp.async.bulk.prefetch.L2.global [%0], %1;" ::"l"(
                         packed + static_cast<long long>(row) * width),
                     "r"(static_cast<unsigned>(width) * 2u)
                     : "memory");
#endif
    }
    const int pos = positions != nullptr ? positions[row] : 0;
    for (int dim_pair = threadIdx.x; dim_pair < span; dim_pair += blockDim.x) {
        const float freq = powf(theta, -static_cast<float>(dim_pair) / static_cast<float>(span));
        const float ang = static_cast<float>(pos) * freq;
        float cos_v, sin_v;
        __sincosf(ang, &sin_v, &cos_v);
        if (dim_pair >= rotary_dim / 2) {
            cos_v = 1.f;
            sin_v = 0.f;
        }
        rotation[dim_pair] = cos_v;
        rotation[HALF + dim_pair] = sin_v;
    }
    __syncthreads();

    const int warp_id = threadIdx.x >> 5;
    const int lane = threadIdx.x & 31;
    const int warps_per_block = blockDim.x >> 5;
    const int total_qk_heads = num_q_heads + num_kv_heads;
    // The packed row's heads at their own widths, landing HEAD_DIM wide.
    const int kv_dim = num_kv_heads * HEAD_DIM;
    const int q_src = num_q_heads * qk_width;
    const int qkv_width = q_src + num_kv_heads * (qk_width + v_width);
    const int packed_stride = qkv_width + rest_width;
    const bf16* src_row = STAGE ? reinterpret_cast<const bf16*>(qkv_stage)
                                : packed + static_cast<long long>(row) * packed_stride;
    if constexpr (STAGE) {
#if __CUDA_ARCH__ >= 900
        unsigned done = 0;
        while (!done) {
            asm volatile(
                "{\n\t.reg .pred p;\n\t"
                "mbarrier.try_wait.parity.shared::cta.b64 p, [%1], 0;\n\t"
                "selp.u32 %0, 1, 0, p;\n\t}"
                : "=r"(done)
                : "r"(staged_at)
                : "memory");
        }
        if (threadIdx.x == 0 && rest_width > 0) {
            asm volatile("cp.async.bulk.global.shared::cta.bulk_group [%0], [%1], %2;" ::"l"(
                             rest + static_cast<long long>(row) * rest_width),
                         "r"(stage_at + qkv_width * 2), "r"(rest_width * 2)
                         : "memory");
            asm volatile("cp.async.bulk.commit_group;" ::: "memory");
        }
#endif
    } else {
        // The row's columns past its qkv (`rest`, a whole number of 16-byte
        // words).
        for (int c = threadIdx.x * 8; c < rest_width; c += blockDim.x * 8) {
            *reinterpret_cast<uint4*>(rest + static_cast<long long>(row) * rest_width + c) =
                *reinterpret_cast<const uint4*>(src_row + qkv_width + c);
        }
    }
    const bool kv_valid = row_valid == nullptr || row_valid[row] != 0;

    int actual_page = 0;
    int offset_in_page = 0;
    if (kv_valid) {
        if (w_page != nullptr && w_off != nullptr) {
            actual_page = static_cast<int>(w_page[row]);
            offset_in_page = static_cast<int>(w_off[row]);
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
    }

    const int pair_offset = span / ELEMS_PER_THREAD;
    const int sub = lane / L;
    const int part = lane % L;
    // A whole-width rotation (pairs half a head apart) over 16-byte words: a
    // lane takes four dims of each half instead, its pairs its own, the
    // rotation's eight values and both weights' held across the heads.
    bool walked = false;
    if constexpr (STAGE && ELEMS_PER_THREAD == 8 && is_same<KV, bf16>::value) {
        if (2 * span == HEAD_DIM && rotary_dim == HEAD_DIM && qk_width == HEAD_DIM && v_width == HEAD_DIM) {
            walked = true;
            const int lo = part * 4;
            auto ld8 = [&](const bf16* at, float (&o)[8]) {
                const uint2 x = *reinterpret_cast<const uint2*>(at + lo);
                const uint2 y = *reinterpret_cast<const uint2*>(at + HALF + lo);
                const unsigned int w4[4] = {x.x, x.y, y.x, y.y};
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    o[2 * k] = __uint_as_float(w4[k] << 16);
                    o[2 * k + 1] = __uint_as_float(w4[k] & 0xffff0000u);
                }
            };
            auto st8 = [&](bf16* at, const float (&o)[8]) {
                *reinterpret_cast<uint2*>(at + lo) = make_uint2(pack_bf16x2(o[0], o[1]), pack_bf16x2(o[2], o[3]));
                *reinterpret_cast<uint2*>(at + HALF + lo) =
                    make_uint2(pack_bf16x2(o[4], o[5]), pack_bf16x2(o[6], o[7]));
            };
            float cs[4], sn[4], wq[8], wk[8];
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                cs[k] = rotation[lo + k];
                sn[k] = rotation[HALF + lo + k];
            }
#pragma unroll
            for (int k = 0; k < 8; ++k) wq[k] = wk[k] = 1.f;
            if (q_weight != nullptr) ld8(q_weight, wq);
            if (k_weight != nullptr) ld8(k_weight, wk);
            for (int head0 = warp_id * G; head0 < total_qk_heads; head0 += warps_per_block * G) {
                const int head_idx = head0 + sub;
                const bool is_q = head_idx < num_q_heads;
                const bool on = head_idx < total_qk_heads && (is_q || kv_valid);
                const int local_head = is_q ? head_idx : (head_idx - num_q_heads);
                float x[8], y[8];
#pragma unroll
                for (int k = 0; k < 8; ++k) x[k] = y[k] = 0.f;
                if (on) ld8(src_row + (is_q ? 0 : q_src) + local_head * HEAD_DIM, x);
                if (on && !is_q) ld8(src_row + q_src + num_kv_heads * HEAD_DIM + local_head * HEAD_DIM, y);
                float sum = 0.f, v_sum = 0.f;
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    sum += x[k] * x[k];
                    v_sum += y[k] * y[k];
                }
#pragma unroll
                for (int offset = L / 2; offset > 0; offset >>= 1) {
                    sum += __shfl_xor_sync(FULL_MASK, sum, offset, L);
                    v_sum += __shfl_xor_sync(FULL_MASK, v_sum, offset, L);
                }
                if (!on) continue;
                if (qk_norm) {
                    const float inv_rms = rsqrtf(sum / static_cast<float>(HEAD_DIM) + eps);
#pragma unroll
                    for (int k = 0; k < 8; ++k) x[k] *= inv_rms * (is_q ? wq[k] : wk[k]);
                }
                // x[k] dim lo + k's (first half), x[4 + k] its pair's.
#pragma unroll
                for (int k = 0; k < 4; ++k) {
                    const float u = x[k], v = x[4 + k];
                    x[k] = u * cs[k] - v * sn[k];
                    x[4 + k] = v * cs[k] + u * sn[k];
                }
                if (is_q) {
                    st8(q_out + (static_cast<long long>(row) * num_q_heads + local_head) * HEAD_DIM, x);
                    continue;
                }
                const long long page_row = hnd_layout
                    ? ((static_cast<long long>(actual_page) * num_kv_heads + local_head) * page_size +
                       offset_in_page) * HEAD_DIM
                    : ((static_cast<long long>(actual_page) * page_size) + offset_in_page) * kv_dim +
                          local_head * HEAD_DIM;
                st8(k_pages + page_row, x);
                const float inv_v = (v_norm ? rsqrtf(v_sum / static_cast<float>(HEAD_DIM) + eps) : 1.f) * v_scale;
#pragma unroll
                for (int k = 0; k < 8; ++k) y[k] *= inv_v;
                st8(v_pages + page_row, y);
            }
        }
    }
    // Otherwise over 16-byte words, each lane's rotation (its cos, its
    // signed sin, the pair `span` away a shuffle off) and both weights' runs
    // held across the heads.
    if constexpr (STAGE && ELEMS_PER_THREAD == 8 && is_same<KV, bf16>::value) {
        if (!walked) {
            walked = true;
            const int run = part * ELEMS_PER_THREAD;
            float cs[8], sn[8], wq[8], wk[8];
#pragma unroll
            for (int k = 0; k < 8; ++k) {
                const int dim = run + k;
                const int in_two = dim & (2 * span - 1);
                const int dim_pair = in_two & (span - 1);
                const bool turns = dim < 2 * span;
                cs[k] = turns ? rotation[dim_pair] : 1.f;
                sn[k] = turns ? (in_two < span ? -rotation[HALF + dim_pair] : rotation[HALF + dim_pair]) : 0.f;
                wq[k] = wk[k] = 1.f;
            }
            if (q_weight != nullptr && run < qk_width) load_bf16_run<8>(q_weight + run, wq);
            if (k_weight != nullptr && run < qk_width) load_bf16_run<8>(k_weight + run, wk);
            for (int head0 = warp_id * G; head0 < total_qk_heads; head0 += warps_per_block * G) {
                const int head_idx = head0 + sub;
                const bool is_q = head_idx < num_q_heads;
                const bool on = head_idx < total_qk_heads && (is_q || kv_valid);
                const int local_head = is_q ? head_idx : (head_idx - num_q_heads);
                float x[8], y[8];
#pragma unroll
                for (int k = 0; k < 8; ++k) x[k] = y[k] = 0.f;
                if (on && run < qk_width) {
                    load_bf16_run<8>(src_row + (is_q ? 0 : q_src) + local_head * qk_width + run, x);
                }
                if (on && !is_q && run < v_width) {
                    load_bf16_run<8>(src_row + q_src + num_kv_heads * qk_width + local_head * v_width + run, y);
                }
                float sum = 0.f, v_sum = 0.f;
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    sum += x[k] * x[k];
                    v_sum += y[k] * y[k];
                }
#pragma unroll
                for (int offset = L / 2; offset > 0; offset >>= 1) {
                    sum += __shfl_xor_sync(FULL_MASK, sum, offset, L);
                    v_sum += __shfl_xor_sync(FULL_MASK, v_sum, offset, L);
                }
                if (qk_norm) {
                    const float inv_rms = rsqrtf(sum / static_cast<float>(HEAD_DIM) + eps);
#pragma unroll
                    for (int k = 0; k < 8; ++k) x[k] *= inv_rms * (is_q ? wq[k] : wk[k]);
                }
#pragma unroll
                for (int k = 0; k < 8; ++k) {
                    const float pair = __shfl_xor_sync(FULL_MASK, x[k], pair_offset, L);
                    x[k] = x[k] * cs[k] + pair * sn[k];
                }
                if (!on) continue;
                HeadDst<KV> dst;
                HeadDst<KV> v_dst;
                if (is_q) {
                    dst.q = q_out + (static_cast<long long>(row) * num_q_heads + local_head) * HEAD_DIM;
                } else {
                    const long long page_row = hnd_layout
                        ? ((static_cast<long long>(actual_page) * num_kv_heads + local_head) * page_size +
                           offset_in_page) * HEAD_DIM
                        : ((static_cast<long long>(actual_page) * page_size) + offset_in_page) * kv_dim +
                              local_head * HEAD_DIM;
                    dst.kv = k_pages + page_row;
                    v_dst.kv = v_pages + page_row;
                }
                dst.template run<8>(run, x);
                if (!is_q) {
                    const float inv_v = (v_norm ? rsqrtf(v_sum / static_cast<float>(HEAD_DIM) + eps) : 1.f) * v_scale;
#pragma unroll
                    for (int k = 0; k < 8; ++k) y[k] *= inv_v;
                    v_dst.template run<8>(run, y);
                }
            }
        }
    }
    for (int head0 = walked ? total_qk_heads : warp_id * G; head0 < total_qk_heads; head0 += warps_per_block * G) {
        // A part of the warp a head: every lane joins the shuffles, a lane
        // past the heads (or on an invalid row's kv) loading and storing nothing.
        const int head_idx = head0 + sub;
        const bool is_q = head_idx < num_q_heads;
        const bool on = head_idx < total_qk_heads && (is_q || kv_valid);
        const int local_head = is_q ? head_idx : (head_idx - num_q_heads);
        const bf16* src = src_row + (is_q ? 0 : q_src) + local_head * qk_width;
        const int run = part * ELEMS_PER_THREAD;
        const bf16* weight = is_q ? q_weight : k_weight;

        float vals[ELEMS_PER_THREAD];
        float v_vals[ELEMS_PER_THREAD];
        float w[ELEMS_PER_THREAD];
        if (on && run < qk_width) {
            load_bf16_run<ELEMS_PER_THREAD>(src + run, vals);
        } else {
#pragma unroll
            for (int i = 0; i < ELEMS_PER_THREAD; ++i) vals[i] = 0.f;
        }
        if (weight != nullptr) {
            load_bf16_run<ELEMS_PER_THREAD>(weight + run, w);
        } else {
#pragma unroll
            for (int i = 0; i < ELEMS_PER_THREAD; ++i) w[i] = 1.f;
        }
        if (!on || is_q || run >= v_width) {
#pragma unroll
            for (int i = 0; i < ELEMS_PER_THREAD; ++i) v_vals[i] = 0.f;
        } else {
            load_bf16_run<ELEMS_PER_THREAD>(
                src_row + q_src + num_kv_heads * qk_width + local_head * v_width + run, v_vals);
        }
        float sum = 0.f;
        float v_sum = 0.f;
#pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            sum += vals[i] * vals[i];
            v_sum += v_vals[i] * v_vals[i];
        }
#pragma unroll
        for (int offset = L / 2; offset > 0; offset >>= 1) {
            sum += __shfl_xor_sync(FULL_MASK, sum, offset, L);
            v_sum += __shfl_xor_sync(FULL_MASK, v_sum, offset, L);
        }

        const float inv_rms = qk_norm ? rsqrtf(sum / static_cast<float>(HEAD_DIM) + eps) : 1.f;
#pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            vals[i] *= qk_norm ? inv_rms * w[i] : 1.f;
        }
#pragma unroll
        for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
            const int dim = run + i;
            const float pair = __shfl_xor_sync(FULL_MASK, vals[i], pair_offset, L);
            const int in_two = dim & (2 * span - 1);
            const float signed_pair = in_two < span ? -pair : pair;
            const int dim_pair = in_two & (span - 1);
            if (dim < 2 * span) {
                vals[i] = vals[i] * rotation[dim_pair] + signed_pair * rotation[HALF + dim_pair];
            }
        }

        if (!on) continue;
        HeadDst<KV> dst;
        HeadDst<KV> v_dst;
        if (is_q) {
            dst.q = q_out + (static_cast<long long>(row) * num_q_heads + local_head) * HEAD_DIM;
        } else if (hnd_layout) {
            const long long page_row =
                ((static_cast<long long>(actual_page) * num_kv_heads +
                  local_head) * page_size + offset_in_page) * HEAD_DIM;
            dst.kv = k_pages + page_row;
            v_dst.kv = v_pages + page_row;
        } else {
            const long long page_row =
                ((static_cast<long long>(actual_page) * page_size) +
                 offset_in_page) * kv_dim;
            dst.kv = k_pages + page_row + local_head * HEAD_DIM;
            v_dst.kv = v_pages + page_row + local_head * HEAD_DIM;
        }
        dst.template run<ELEMS_PER_THREAD>(run, vals);
        if (!is_q) {
            const float inv_v = v_norm ? rsqrtf(v_sum / static_cast<float>(HEAD_DIM) + eps) : 1.f;
#pragma unroll
            for (int i = 0; i < ELEMS_PER_THREAD; ++i) {
                v_vals[i] = v_vals[i] * inv_v * v_scale;
            }
            v_dst.template run<ELEMS_PER_THREAD>(run, v_vals);
        }
    }
    if constexpr (STAGE) {
#if __CUDA_ARCH__ >= 900
        if (threadIdx.x == 0 && rest_width > 0) asm volatile("cp.async.bulk.wait_group.read 0;" ::: "memory");
#endif
    }
}

}
