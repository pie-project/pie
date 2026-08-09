//===-- attention_naive.cuh - the reference attention, and MTP's plumbing -===//
//
// Five `__global__` templates and the two `__device__` helpers they call. No
// host code.
//
// # What lives here and why it is worth keeping
//
// Three of these are REFERENCE attention -- four passes over the KV range,
// fp32 accumulators, a shared-memory softmax, no tiling and no tensor cores.
// They exist so that a parity test has something to compare flashinfer
// against on a machine and a shape flashinfer does not cover, and the header
// on the original said so: *"sized for parity tests (<= 1024 tokens);
// production paths will use flashinfer."* Migrating them to the JIT is not
// about speed -- it is about there being one definition of the reference, in
// a file the JIT can also read, so the reference cannot drift from the thing
// it is the reference for.
//
// The other two are MTP's hidden-state plumbing: the speculative decoder
// feeds each drafted position the PREVIOUS position's hidden state, and the
// first token of a request takes it from a per-slot pending buffer instead.
// One kernel does the shift, the other refreshes the pending buffer at the
// end of a step.
//
// # One row out of five
//
// `mtp_shift_hidden` was `<<<total_tokens, 256>>>` -- one block per row,
// 256 wide, a stride loop over `hidden_size`. That is `LaunchRule::Rms`, and
// it has a row. `Rms` also hands it 32 bytes of dynamic shared memory that a
// pure copy does not touch; a rule per unused allocation is not a trade worth
// making.
//
// The four that do not:
//
// * `attn_naive`, `attn_mtp_history` and `attn_mtp_paged_history` are
//   `<<<dim3(num_q_heads, num_tokens), 256, (extent + 256) * 4>>>`. Two
//   obstacles, either one fatal: the grid's x axis is a HEAD COUNT, which
//   this backend's `Dims` -- `rows`, `width`, `in_width` -- has no field for,
//   and the dynamic shared memory is sized on a KV extent no ported rule
//   derives. A launch that under-sizes shared memory does not fail; it reads
//   the neighbouring block's scores.
// * `attn_mtp_paged_history` has a third obstacle that is not about geometry
//   at all: its launcher CHOOSES BETWEEN TWO KERNELS. With no global cache,
//   or with `max_global_tokens + history_steps > 8192`, it calls
//   `attention_mtp_history_bf16` instead -- a shared-memory budget check that
//   falls back rather than failing the launch. A `LaunchRule` selects a
//   rectangle, not a kernel, so that decision has no place to go in a row and
//   stays in the `.cu`.
// * `mtp_update_pending_hidden` is `<<<num_requests, 256>>>` -- one block per
//   REQUEST, which no rule opened when this was written. `LaunchRule::PerRequest`
//   does, and `families/attn.rs`'s `ATTENTION_NAIVE_SIGS[2]` is the row: this
//   is now FOUR refusals, not five.
//
// All five are here anyway, once, NVRTC-clean. When a rule arrives for one,
// the diff is a row.
//
// # What NVRTC forced
//
// `<cuda_bf16.h>`, `<cmath>` and `<cstdint>` are gone: NVRTC answered 0 of 31
// standard headers when it was measured, and it ships no CUDA headers either.
// `__nv_bfloat16` became the prelude's `device::bf16` (the same two bytes,
// through a struct that makes `bf16` and `f16` distinguishable in a row),
// `std::uint32_t`/`std::int32_t` became `device::u32`/`device::i32`, and
// `-INFINITY` became `device::neg_inf()` -- NVRTC does not define the
// `INFINITY` macro, measured, and `__int_as_float(0xff800000)` is the same
// bit pattern with no macro at all.
//
// `expf`, `fmaxf` and `sqrtf` it does define, and they are the originals'.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "pie_device.cuh"

namespace pie_cuda_driver::kernels::attn::device {

using ::pie_cuda_driver::kernels::device::Elem;
using ::pie_cuda_driver::kernels::device::bf16;
using ::pie_cuda_driver::kernels::device::i32;
using ::pie_cuda_driver::kernels::device::neg_inf;
using ::pie_cuda_driver::kernels::device::u32;

/// The block width every kernel here is launched at, and the width
/// `reduce_buf` is sized on in the shared-memory budget the launchers
/// compute. Changing one without the other is a silent overlap.
///
/// `[[maybe_unused]]` because a compile that instantiates only
/// `mtp_shift_hidden` -- which the JIT's one row does -- never parses a use
/// of it, and a warning on a constant the `.cu` shares is noise a reader
/// learns to skip past.
[[maybe_unused]] constexpr int BLOCK = 256;

/// One block per (query head, query position). Threads cover `head_dim` and
/// the key range cooperatively.
///
///   1. `scores[j] = (q . k_j) / sqrt(d)` for `j in [0, query_pos]`
///   2. reduce-max, for numerical stability
///   3. `exp(scores[j] - max)`, and sum
///   4. `out[d] = sum_j (exp_score[j] / total) * v_j[d]`
///
/// Shared memory is `scores[num_tokens]` then `reduce_buf[BLOCK]`, and the
/// launcher sizes it -- which is one of the two reasons no row states this
/// kernel. See the header.
template <class T>
__global__ void attn_naive(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    T* __restrict__ o,
    i32 num_tokens,
    i32 num_q_heads,
    i32 num_kv_heads,
    i32 head_dim,
    float scale)
{
    extern __shared__ float smem[];
    float* scores = smem;                         // size: num_tokens
    float* reduce_buf = smem + num_tokens;        // size: BLOCK

    const int head      = static_cast<int>(blockIdx.x);
    const int query_pos = static_cast<int>(blockIdx.y);
    const int tid       = static_cast<int>(threadIdx.x);
    const int gqa_ratio = num_q_heads / num_kv_heads;
    const int kv_head   = head / gqa_ratio;

    const T* q_vec =
        q + (static_cast<long long>(query_pos) * num_q_heads + head) * head_dim;

    // Pass 1: scores
    for (int j = tid; j <= query_pos; j += BLOCK) {
        const T* k_vec =
            k + (static_cast<long long>(j) * num_kv_heads + kv_head) * head_dim;

        float dot = 0.f;
        for (int i = 0; i < head_dim; ++i) {
            dot += Elem<T>::to_f32(q_vec[i]) * Elem<T>::to_f32(k_vec[i]);
        }
        scores[j] = dot * scale;
    }
    __syncthreads();

    // Pass 2: max
    float m = neg_inf();
    for (int j = tid; j <= query_pos; j += BLOCK) {
        m = fmaxf(m, scores[j]);
    }
    reduce_buf[tid] = m;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + off]);
        __syncthreads();
    }
    const float max_score = reduce_buf[0];

    // Pass 3: exp & sum
    float local_sum = 0.f;
    for (int j = tid; j <= query_pos; j += BLOCK) {
        const float e = expf(scores[j] - max_score);
        scores[j] = e;
        local_sum += e;
    }
    reduce_buf[tid] = local_sum;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) reduce_buf[tid] += reduce_buf[tid + off];
        __syncthreads();
    }
    const float inv_total = 1.0f / reduce_buf[0];

    // Pass 4: weighted sum of V
    T* o_vec =
        o + (static_cast<long long>(query_pos) * num_q_heads + head) * head_dim;

    for (int i = tid; i < head_dim; i += BLOCK) {
        float acc = 0.f;
        for (int j = 0; j <= query_pos; ++j) {
            const T* v_vec =
                v + (static_cast<long long>(j) * num_kv_heads + kv_head) * head_dim;
            acc += scores[j] * inv_total * Elem<T>::to_f32(v_vec[i]);
        }
        o_vec[i] = Elem<T>::from_f32(acc);
    }
}

/// The same four passes over MTP's DRAFT history rather than a KV cache.
///
/// The history is `[history_steps, history_stride, heads, head_dim]` and a
/// row reads down the step axis at its own offset -- `hist_row = j *
/// history_stride + row`. `history_stride` is a stride that happens to equal
/// an extent on most fires and not on all, which is exactly the pair a
/// launcher's `<<<>>>` could not tell apart.
///
/// NO ROW STATES THIS KERNEL: per-head grid, extent-sized shared memory.
template <class T>
__global__ void attn_mtp_history(
    const T* __restrict__ q,
    const T* __restrict__ k,
    const T* __restrict__ v,
    T* __restrict__ o,
    i32 num_tokens,
    i32 history_steps,
    i32 history_stride,
    i32 num_q_heads,
    i32 num_kv_heads,
    i32 head_dim,
    float scale)
{
    extern __shared__ float smem[];
    float* scores = smem;                         // size: history_steps
    float* reduce_buf = smem + history_steps;     // size: BLOCK

    const int head = static_cast<int>(blockIdx.x);
    const int row = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    const int gqa_ratio = num_q_heads / num_kv_heads;
    const int kv_head = head / gqa_ratio;

    if (row >= num_tokens) return;

    const T* q_vec =
        q + (static_cast<long long>(row) * num_q_heads + head) * head_dim;

    for (int j = tid; j < history_steps; j += BLOCK) {
        const int hist_row = j * history_stride + row;
        const T* k_vec =
            k + (static_cast<long long>(hist_row) * num_kv_heads + kv_head) * head_dim;
        float dot = 0.f;
        for (int i = 0; i < head_dim; ++i) {
            dot += Elem<T>::to_f32(q_vec[i]) * Elem<T>::to_f32(k_vec[i]);
        }
        scores[j] = dot * scale;
    }
    __syncthreads();

    float m = neg_inf();
    for (int j = tid; j < history_steps; j += BLOCK) {
        m = fmaxf(m, scores[j]);
    }
    reduce_buf[tid] = m;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + off]);
        __syncthreads();
    }
    const float max_score = reduce_buf[0];

    float local_sum = 0.f;
    for (int j = tid; j < history_steps; j += BLOCK) {
        const float e = expf(scores[j] - max_score);
        scores[j] = e;
        local_sum += e;
    }
    reduce_buf[tid] = local_sum;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) reduce_buf[tid] += reduce_buf[tid + off];
        __syncthreads();
    }
    const float inv_total = 1.0f / reduce_buf[0];

    T* o_vec =
        o + (static_cast<long long>(row) * num_q_heads + head) * head_dim;
    for (int i = tid; i < head_dim; i += BLOCK) {
        float acc = 0.f;
        for (int j = 0; j < history_steps; ++j) {
            const int hist_row = j * history_stride + row;
            const T* v_vec =
                v + (static_cast<long long>(hist_row) * num_kv_heads + kv_head) * head_dim;
            acc += scores[j] * inv_total * Elem<T>::to_f32(v_vec[i]);
        }
        o_vec[i] = Elem<T>::from_f32(acc);
    }
}

/// Which request token `token_idx` belongs to, by scanning `qo_indptr`.
///
/// A linear scan and not a binary search: `R` is the batch size, tens at
/// most, and the scan is warp-uniform so every thread in the block walks the
/// same short prefix. Falls back to the last request rather than returning
/// -1, because a token past the final offset is a lowering bug and reading
/// the last request's slot is the failure that shows up in output rather than
/// as an out-of-bounds load.
__device__ __forceinline__ int find_request_u32(
    const u32* __restrict__ qo_indptr,
    int R,
    int token_idx) {
    for (int r = 0; r < R; ++r) {
        if (token_idx < static_cast<int>(qo_indptr[r + 1])) return r;
    }
    return R - 1;
}

/// MTP's input shift: each drafted position reads the PREVIOUS position's
/// hidden state, and the first token of a request reads its slot's pending
/// buffer.
///
/// One block per token; the `t >= total_tokens` guard the launcher's grid
/// made redundant is `LaunchRule::Rms`'s promise now, so `total_tokens` is
/// gone from the signature -- the same trade `norm/altup_aux.cuh` documents.
/// `num_requests` stays, because `find_request_u32` bounds its scan with it
/// and that is a LENGTH the rule does not know.
///
/// `slot_ids` is nullable and the null branch means slot zero: a
/// single-request fire does not carry a slot table.
template <class T>
__global__ void mtp_shift_hidden(
    const T* __restrict__ target_hidden,
    const T* __restrict__ pending_hidden,
    const u32* __restrict__ qo_indptr,
    const i32* __restrict__ slot_ids,
    T* __restrict__ out,
    i32 num_requests,
    i32 hidden_size)
{
    const int t = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    const int r = find_request_u32(qo_indptr, num_requests, t);
    const bool first_in_request = t == static_cast<int>(qo_indptr[r]);
    const int slot = slot_ids != nullptr ? slot_ids[r] : 0;
    const T* src = first_in_request
        ? pending_hidden + static_cast<long long>(slot) * hidden_size
        : target_hidden + static_cast<long long>(t - 1) * hidden_size;
    T* dst = out + static_cast<long long>(t) * hidden_size;
    for (int i = tid; i < hidden_size; i += static_cast<int>(blockDim.x)) {
        dst[i] = src[i];
    }
}

/// The end-of-step refresh: each request's LAST target hidden state becomes
/// its slot's pending state for the next step.
///
/// NO ROW STATED THIS KERNEL for as long as every ported rule opened its grid
/// over rows: a fire of eight requests and ninety-three tokens would open
/// ninety-three blocks -- eighty-five of them writing a slot that is not
/// theirs. `LaunchRule::PerRequest` opens it over `Dims::requests` instead,
/// and `families/attn.rs`'s `ATTENTION_NAIVE_SIGS[2]` is the row.
template <class T>
__global__ void mtp_update_pending_hidden(
    const T* __restrict__ target_hidden,
    T* __restrict__ pending_hidden,
    const u32* __restrict__ qo_indptr,
    const i32* __restrict__ slot_ids,
    i32 num_requests,
    i32 hidden_size)
{
    const int r = static_cast<int>(blockIdx.x);
    const int tid = static_cast<int>(threadIdx.x);
    if (r >= num_requests) return;
    const int lo = static_cast<int>(qo_indptr[r]);
    const int hi = static_cast<int>(qo_indptr[r + 1]);
    if (hi <= lo) return;
    const int slot = slot_ids != nullptr ? slot_ids[r] : 0;
    const T* src = target_hidden + static_cast<long long>(hi - 1) * hidden_size;
    T* dst = pending_hidden + static_cast<long long>(slot) * hidden_size;
    for (int i = tid; i < hidden_size; i += static_cast<int>(blockDim.x)) {
        dst[i] = src[i];
    }
}

/// The address of one paged KV vector, in either layout.
///
/// A local helper and not `kv_paged_addr.cuh`'s: this path indexes by
/// POSITION WITHIN A REQUEST (`pos / page_size` walks the request's own page
/// list) rather than by a global slot, and it takes the layout as a runtime
/// `bool` rather than a template parameter, because the launcher does not
/// know it until the cache is opened.
template <class T>
__device__ __forceinline__ const T* mtp_paged_vec(
    const T* __restrict__ pages,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    int request,
    int pos,
    int page_size,
    int num_kv_heads,
    int head_dim,
    int kv_head,
    bool hnd_layout)
{
    const int page_in_req = pos / page_size;
    const int off = pos - page_in_req * page_size;
    const int actual_page = static_cast<int>(
        kv_page_indices[kv_page_indptr[request] + page_in_req]);
    if (hnd_layout) {
        return pages +
            (((static_cast<long long>(actual_page) * num_kv_heads + kv_head) *
              page_size + off) * head_dim);
    }
    return pages +
        (((static_cast<long long>(actual_page) * page_size + off) *
          num_kv_heads + kv_head) * head_dim);
}

/// Reference attention over the paged global cache AND the draft history, in
/// one softmax.
///
/// The two key sources are concatenated along `j`: `[0, global_len)` reads
/// pages, `[global_len, total_steps)` reads history. One softmax over the
/// concatenation is the whole point -- normalising them separately would give
/// the draft history its own denominator and change the distribution.
///
/// `prefix_global` says whether `position_ids` counts the drafted tokens: a
/// prefix-positioned cache has already absorbed `history_steps - 1` of them,
/// so the global length is the position minus that. Getting it backwards
/// double-counts the draft window and attends to keys the draft is about to
/// overwrite.
///
/// NO ROW STATES THIS KERNEL: per-head grid, extent-sized shared memory, AND
/// a launcher that chooses between this kernel and `attn_mtp_history`. See
/// the header.
template <class T>
__global__ void attn_mtp_paged_history(
    const T* __restrict__ q,
    const T* __restrict__ k_pages,
    const T* __restrict__ v_pages,
    const T* __restrict__ k_history,
    const T* __restrict__ v_history,
    T* __restrict__ o,
    const i32* __restrict__ position_ids,
    const i32* __restrict__ request_ids,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const u32* __restrict__ kv_last_page_lens,
    i32 num_tokens,
    i32 history_steps,
    i32 history_stride,
    i32 max_global_tokens,
    i32 page_size,
    i32 num_q_heads,
    i32 num_kv_heads,
    i32 head_dim,
    bool hnd_layout,
    float scale,
    bool prefix_global)
{
    extern __shared__ float smem[];
    float* scores = smem;
    float* reduce_buf = smem + max_global_tokens + history_steps;

    const int head = static_cast<int>(blockIdx.x);
    const int row = static_cast<int>(blockIdx.y);
    const int tid = static_cast<int>(threadIdx.x);
    if (row >= num_tokens) return;

    const int gqa_ratio = num_q_heads / num_kv_heads;
    const int kv_head = head / gqa_ratio;
    const int request = request_ids[row];
    const int page_lo = static_cast<int>(kv_page_indptr[request]);
    const int page_hi = static_cast<int>(kv_page_indptr[request + 1]);
    const int pages = page_hi - page_lo;
    const int max_kv_len = pages <= 0
        ? 0
        : (pages - 1) * page_size + static_cast<int>(kv_last_page_lens[request]);
    int global_len = position_ids[row] -
                     (prefix_global ? (history_steps - 1) : 0);
    if (global_len < 0) global_len = 0;
    if (global_len > max_kv_len) global_len = max_kv_len;
    if (global_len > max_global_tokens) global_len = max_global_tokens;
    const int total_steps = global_len + history_steps;

    const T* q_vec =
        q + (static_cast<long long>(row) * num_q_heads + head) * head_dim;

    for (int j = tid; j < total_steps; j += BLOCK) {
        const T* k_vec = nullptr;
        if (j < global_len) {
            k_vec = mtp_paged_vec<T>(k_pages, kv_page_indices, kv_page_indptr,
                request, j, page_size, num_kv_heads, head_dim, kv_head,
                hnd_layout);
        } else {
            const int hist_step = j - global_len;
            const int hist_row = hist_step * history_stride + row;
            k_vec = k_history +
                (static_cast<long long>(hist_row) * num_kv_heads + kv_head) *
                    head_dim;
        }
        float dot = 0.f;
        for (int i = 0; i < head_dim; ++i) {
            dot += Elem<T>::to_f32(q_vec[i]) * Elem<T>::to_f32(k_vec[i]);
        }
        scores[j] = dot * scale;
    }
    __syncthreads();

    float m = neg_inf();
    for (int j = tid; j < total_steps; j += BLOCK) {
        m = fmaxf(m, scores[j]);
    }
    reduce_buf[tid] = m;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) reduce_buf[tid] = fmaxf(reduce_buf[tid], reduce_buf[tid + off]);
        __syncthreads();
    }
    const float max_score = reduce_buf[0];

    float local_sum = 0.f;
    for (int j = tid; j < total_steps; j += BLOCK) {
        const float e = expf(scores[j] - max_score);
        scores[j] = e;
        local_sum += e;
    }
    reduce_buf[tid] = local_sum;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) reduce_buf[tid] += reduce_buf[tid + off];
        __syncthreads();
    }
    const float inv_total = 1.0f / reduce_buf[0];

    T* o_vec =
        o + (static_cast<long long>(row) * num_q_heads + head) * head_dim;
    for (int i = tid; i < head_dim; i += BLOCK) {
        float acc = 0.f;
        for (int j = 0; j < total_steps; ++j) {
            const T* v_vec = nullptr;
            if (j < global_len) {
                v_vec = mtp_paged_vec<T>(v_pages, kv_page_indices, kv_page_indptr,
                    request, j, page_size, num_kv_heads, head_dim, kv_head,
                    hnd_layout);
            } else {
                const int hist_step = j - global_len;
                const int hist_row = hist_step * history_stride + row;
                v_vec = v_history +
                    (static_cast<long long>(hist_row) * num_kv_heads + kv_head) *
                        head_dim;
            }
            acc += scores[j] * inv_total * Elem<T>::to_f32(v_vec[i]);
        }
        o_vec[i] = Elem<T>::from_f32(acc);
    }
}

}  // namespace pie_cuda_driver::kernels::attn::device
