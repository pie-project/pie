#include <cstdint>
#include <cstring>
#include <cuda_bf16.h>
#include <cuda_fp8.h>
#include <cuda_pipeline.h>
#include <math_constants.h>
#include "prim/device.cuh"
#include "prim/rope.cuh"

namespace pie::attn::mla {

template <class T>
__global__ void mla_split_q_b(
    const T* __restrict__ q_b,
    T* __restrict__ q_nope,
    T* __restrict__ q_pe,
    int total,
    int heads,
    int nope,
    int rope,
    const u32* __restrict__ win)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    const int per = nope + rope;
    const int d = i % per;
    const int h = (i / per) % heads;
    const int n = i / (heads * per);

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int n_row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const T v = q_b[(static_cast<long long>(n_row) * heads + h) * per + d];
    if (d < nope) {
        q_nope[(static_cast<long long>(n_row) * heads + h) * nope + d] = v;
    } else {
        q_pe[(static_cast<long long>(n_row) * heads + h) * rope + (d - nope)] = v;
    }
}

// `mla_split_q_b` eight elements a thread (16-byte runs), for a cut whose
// halves are whole runs: `runs` the 8-element runs of every row's heads.
template <class T>
__global__ void mla_split_q_b_vec8(
    const T* __restrict__ q_b,
    T* __restrict__ q_nope,
    T* __restrict__ q_pe,
    long long runs,
    int heads,
    int nope,
    int rope,
    const u32* __restrict__ win)
{
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    const long long i = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= runs) return;
    const int per = (nope + rope) / 8;
    const int d = static_cast<int>(i % per) * 8;
    const long long hn = i / per;
    const int h = static_cast<int>(hn % heads);
    const long long n = hn / heads;
    if (win != nullptr && n >= static_cast<long long>(win[0])) return;
    const long long n_row = win != nullptr ? n + static_cast<long long>(win[1]) : n;
    const long long head = n_row * heads + h;
    const uint4 v = *reinterpret_cast<const uint4*>(q_b + head * (nope + rope) + d);
    if (d < nope) {
        *reinterpret_cast<uint4*>(q_nope + head * nope + d) = v;
    } else {
        *reinterpret_cast<uint4*>(q_pe + head * rope + (d - nope)) = v;
    }
}

template <class T, int BLOCK_DIM = 256>
__global__ void mla_latents(
    const T* __restrict__ kv_a,
    const T* __restrict__ norm_weight,
    T* __restrict__ kv_c,
    T* __restrict__ k_pe,
    int kv_lora,
    int rope,
    int src_row_stride,
    float eps,
    const u32* __restrict__ win)
{
    // A programmatic dependent (`kernel.pdl`) touches nothing until its
    // producer is done; its own dependents may launch from here.
    pdl_wait();
    pdl_launch();
    const int n = blockIdx.x;

    if (win != nullptr && n >= static_cast<int>(win[0])) return;

    const int n_row = win != nullptr ? n + static_cast<int>(win[1]) : n;
    const int tid = threadIdx.x;
    const T* row = kv_a + static_cast<long long>(n_row) * src_row_stride;

    for (int d = tid; d < rope; d += BLOCK_DIM) {
        k_pe[static_cast<long long>(n_row) * rope + d] = row[kv_lora + d];
    }

    float local = 0.f;
    for (int d = tid; d < kv_lora; d += BLOCK_DIM) {
        const float v = Elem<T>::to_f32(row[d]);
        local += v * v;
    }
    __shared__ float buf[BLOCK_DIM];
    buf[tid] = local;
    __syncthreads();
    for (int off = BLOCK_DIM / 2; off > 0; off >>= 1) {
        if (tid < off) buf[tid] += buf[tid + off];
        __syncthreads();
    }
    const float inv_rms = rsqrtf(buf[0] / static_cast<float>(kv_lora) + eps);
    for (int d = tid; d < kv_lora; d += BLOCK_DIM) {
        const float v = Elem<T>::to_f32(row[d]);
        const float w = Elem<T>::to_f32(norm_weight[d]);
        kv_c[static_cast<long long>(n_row) * kv_lora + d] = Elem<T>::from_f32(v * inv_rms * w);
    }
}

__device__ __forceinline__ int mla_find_request(const u32* qo_indptr,
                                                int R,
                                                int token_idx) {
    // The first request whose rows end past the token, as a walk finds it
    // (the indptr is not ordered past the live requests, so no search may
    // halve it): 32 requests a round, a lane each, their loads in flight
    // together, the lowest lane past the token taken. Every caller's warps
    // are whole and alike.
    const int lane = static_cast<int>(threadIdx.x % 32);
    for (int b = 0; b < R; b += 32) {
        const int r = b + lane;
        const bool past = r < R && token_idx < static_cast<int>(__ldg(&qo_indptr[r + 1]));
        const unsigned m = __ballot_sync(0xffffffffu, past);
        if (m != 0u) return b + __ffs(m) - 1;
    }
    return R - 1;
}

__device__ __forceinline__ void mla_resolve_dst(
    const u32* __restrict__ qo_indptr,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    int R,
    int page_size,
    int token_idx,
    int& actual_page,
    int& offset_in_page)
{
    const int r = mla_find_request(qo_indptr, R, token_idx);
    const int qo_lo = qo_indptr[r];
    const int qo_hi = qo_indptr[r + 1];
    const int new_tokens_r = qo_hi - qo_lo;
    const int offset_in_new = token_idx - qo_lo;
    const int pages_first = kv_page_indptr[r];
    const int pages_last = kv_page_indptr[r + 1];
    const int num_pages_r = pages_last - pages_first;
    const int total_kv_after =
        (num_pages_r - 1) * page_size + kv_last_page_lens[r];
    const int pre_kv_len = total_kv_after - new_tokens_r;
    const int abs_kv_pos = pre_kv_len + offset_in_new;
    const int page_in_req = abs_kv_pos / page_size;
    offset_in_page = abs_kv_pos % page_size;
    actual_page = static_cast<int>(kv_page_indices[pages_first + page_in_req]);
}

// Under a staged window `win` a block walks only the window's live rows: its
// source and validity rows sit `win[1]` into the plane, and its requests are
// the window's first `win[2]` lanes.
__global__ void mla_kv_append(
    const bf16* __restrict__ ckv_curr,
    const bf16* __restrict__ kpe_curr,
    bf16* __restrict__ ckv_pages,
    bf16* __restrict__ kpe_pages,
    const u32* __restrict__ qo_indptr,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    const u8* __restrict__ row_valid,
    int R,
    int page_size,
    int kv_lora_rank,
    int qk_rope_head_dim,
    const u32* __restrict__ win)
{
    const int t = blockIdx.x;

    if (win != nullptr && t >= static_cast<int>(win[0])) return;

    const int t_row = win != nullptr ? t + static_cast<int>(win[1]) : t;

    if (win != nullptr && static_cast<int>(win[2]) < R) R = static_cast<int>(win[2]);
    if (row_valid != nullptr && row_valid[t_row] == 0) return;
    int actual_page = 0;
    int offset_in_page = 0;
    mla_resolve_dst(qo_indptr, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens, R, page_size, t, actual_page,
                    offset_in_page);

    const long long ckv_src = static_cast<long long>(t_row) * kv_lora_rank;
    const long long ckv_dst =
        (static_cast<long long>(actual_page) * page_size + offset_in_page) *
        kv_lora_rank;
    for (int i = threadIdx.x; i < kv_lora_rank; i += blockDim.x) {
        ckv_pages[ckv_dst + i] = ckv_curr[ckv_src + i];
    }

    const long long kpe_src = static_cast<long long>(t_row) * qk_rope_head_dim;
    const long long kpe_dst =
        (static_cast<long long>(actual_page) * page_size + offset_in_page) *
        qk_rope_head_dim;
    for (int i = threadIdx.x; i < qk_rope_head_dim; i += blockDim.x) {
        kpe_pages[kpe_dst + i] = kpe_curr[kpe_src + i];
    }
}

// The latent rows and their rotated dims as vLLM's standard fp8 rows: 512
// latent codes then `rope` (64 or 0) rotated ones, e4m3 under a scale of 1
// (saturating). A block a token, a thread four values (five whole warps, as
// the request search takes them).

__global__ __launch_bounds__(160) void mla_kv_append_fp8(
    const bf16* __restrict__ ckv_curr,
    const bf16* __restrict__ kpe_curr,
    u8* __restrict__ rows,
    const u32* __restrict__ qo_indptr,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    const u8* __restrict__ row_valid,
    int R,
    int page_size,
    int rope,
    const u32* __restrict__ win)
{
    const int t = blockIdx.x;

    if (win != nullptr && t >= static_cast<int>(win[0])) return;

    const int t_row = win != nullptr ? t + static_cast<int>(win[1]) : t;

    if (win != nullptr && static_cast<int>(win[2]) < R) R = static_cast<int>(win[2]);
    if (row_valid != nullptr && row_valid[t_row] == 0) return;
    int actual_page = 0;
    int offset_in_page = 0;
    mla_resolve_dst(qo_indptr, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens, R, page_size, t, actual_page,
                    offset_in_page);
    const int width = 512 + rope;
    u8* row = rows + (static_cast<long long>(actual_page) * page_size + offset_in_page) * width;
    const int i = static_cast<int>(threadIdx.x);
    if (i >= width / 4) return;
    const bf16* src = i < 128 ? ckv_curr + static_cast<long long>(t_row) * 512 + 4 * i
                              : kpe_curr + static_cast<long long>(t_row) * rope + 4 * (i - 128);
    std::uint32_t packed = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const std::uint32_t b = __nv_cvt_float_to_fp8(__bfloat162float(src[k]), __NV_SATFINITE, __NV_E4M3);
        packed |= (b & 0xffu) << (8 * k);
    }
    *reinterpret_cast<std::uint32_t*>(row + 4 * i) = packed;
}

}
