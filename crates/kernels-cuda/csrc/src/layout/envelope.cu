#include "layout/envelope.hpp"
#include "layout/envelope_device.cuh"

#include <cstdint>

#include <cuda_bf16.h>
#include <math_constants.h>

namespace pie_cuda_driver::kernels::layout {

namespace {

// The per-(page, kv_head) min/max reduction, shared by full recompute and the
// page-list update so both paths are literally the same numerics (the full
// recompute is what `test_envelope_dot` parity-checks).
__device__ inline void envelope_reduce_page(
    const __nv_bfloat16* __restrict__ k_pages,
    int page,
    int kh,
    int live,
    int page_size,
    int num_kv_heads,
    int head_dim,
    __nv_bfloat16* __restrict__ env_min,
    __nv_bfloat16* __restrict__ env_max)
{
    const long token_stride = static_cast<long>(num_kv_heads) * head_dim;
    const long page_base = static_cast<long>(page) * page_size * token_stride +
                           static_cast<long>(kh) * head_dim;
    const long env_base =
        (static_cast<long>(page) * num_kv_heads + kh) * head_dim;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float mn = CUDART_INF_F;
        float mx = -CUDART_INF_F;
        for (int t = 0; t < live; ++t) {
            const float v = __bfloat162float(
                k_pages[page_base + static_cast<long>(t) * token_stride + d]);
            mn = fminf(mn, v);
            mx = fmaxf(mx, v);
        }
        env_min[env_base + d] = __float2bfloat16_rd(mn);
        env_max[env_base + d] = __float2bfloat16_ru(mx);
    }
}

// One block per (page, kv_head); threads stride over head_dim, each reducing its
// dims' min/max across the page's live tokens. Streaming reads of the NHD layout.
__global__ void envelope_recompute_kernel(
    const __nv_bfloat16* __restrict__ k_pages,
    const std::int32_t* __restrict__ page_live_lens,
    __nv_bfloat16* __restrict__ env_min,
    __nv_bfloat16* __restrict__ env_max,
    int page_size,
    int num_kv_heads,
    int head_dim)
{
    const int page = blockIdx.x;
    const int kh = blockIdx.y;
    envelope_reduce_page(k_pages, page, kh, page_live_lens[page], page_size,
                         num_kv_heads, head_dim, env_min, env_max);
}

// Refresh exactly the pages this fire appended to, deriving that set on-device
// from the same CSR arithmetic `write_kv_kernel` uses. A request's post-append
// length is `(pages-1)*page_size + last_page_len`, so its new tokens occupy
// `[total_after - qo_len, total_after)` and the touched pages are that span
// divided by `page_size`. Rescanning the whole page list instead would cost a
// full KV read per layer -- as much as attention itself.
//
// One block per (touched slot, kv_head), where the grid's x extent is the host's
// worst-case bound `ceil(total_tokens/page_size) + num_requests`; blocks past
// the true count exit. Pages are append-only, so recomputing a touched page in
// full gives the same answer an incremental merge would.
__global__ void envelope_update_appended_kernel(
    const __nv_bfloat16* __restrict__ k_pages,
    const std::uint32_t* __restrict__ qo_indptr,
    const std::uint32_t* __restrict__ kv_page_indices,
    const std::uint32_t* __restrict__ kv_page_indptr,
    const std::uint32_t* __restrict__ kv_last_page_lens,
    __nv_bfloat16* __restrict__ env_min,
    __nv_bfloat16* __restrict__ env_max,
    int num_requests,
    int page_size,
    int num_kv_heads,
    int head_dim)
{
    const int slot = blockIdx.x;
    const int kh = blockIdx.y;

    // Walk requests accumulating their touched-page counts until `slot` lands
    // inside one. R is the batch size, so a linear scan beats the divergence a
    // binary search would add.
    int seen = 0;
    for (int r = 0; r < num_requests; ++r) {
        const int pages_first = static_cast<int>(kv_page_indptr[r]);
        const int pages_last = static_cast<int>(kv_page_indptr[r + 1]);
        const int num_pages_r = pages_last - pages_first;
        if (num_pages_r <= 0) continue;

        const int qo_len =
            static_cast<int>(qo_indptr[r + 1]) - static_cast<int>(qo_indptr[r]);
        if (qo_len <= 0) continue;

        const int total_after =
            (num_pages_r - 1) * page_size + static_cast<int>(kv_last_page_lens[r]);
        const int pre_len = total_after - qo_len;
        if (total_after <= 0) continue;

        const int first_page = pre_len / page_size;
        const int last_page = (total_after - 1) / page_size;
        const int touched = last_page - first_page + 1;

        if (slot < seen + touched) {
            const int page_in_req = first_page + (slot - seen);
            if (page_in_req >= num_pages_r) return;
            const int live = (page_in_req == last_page)
                ? static_cast<int>(kv_last_page_lens[r])
                : page_size;
            if (live <= 0) return;
            envelope_reduce_page(
                k_pages,
                static_cast<int>(kv_page_indices[pages_first + page_in_req]),
                kh, live, page_size, num_kv_heads, head_dim, env_min, env_max);
            return;
        }
        seen += touched;
    }
}

// One block per (kv_head, page); threads reduce over the group·head_dim terms of
// `Σ max(q·min, q·max)`. Pages beyond `live_pages` are `-inf`.
template <int BLOCK>
__global__ void envelope_dot_kernel(
    const float* __restrict__ q,
    const __nv_bfloat16* __restrict__ env_min,
    const __nv_bfloat16* __restrict__ env_max,
    float* __restrict__ score,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int p_max,
    int live_pages)
{
    const int kh = blockIdx.y;
    const int p = blockIdx.x;
    float* out = &score[static_cast<long>(kh) * p_max + p];

    if (p >= live_pages) {
        if (threadIdx.x == 0) *out = -CUDART_INF_F;
        return;
    }

    const int group = num_q_heads / num_kv_heads;
    const long env_base =
        (static_cast<long>(p) * num_kv_heads + kh) * head_dim;

    const float local = envelope_dot_thread_partial(
        q, env_min, env_max, env_base, kh * group, group, head_dim,
        static_cast<int>(threadIdx.x), BLOCK);

    __shared__ float buf[BLOCK];
    buf[threadIdx.x] = local;
    __syncthreads();
    for (int off = BLOCK / 2; off > 0; off >>= 1) {
        if (threadIdx.x < off) buf[threadIdx.x] += buf[threadIdx.x + off];
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = buf[0];
}

// One thread per (page, kv_head, dim) triple of the empty envelope.
__global__ void envelope_seed_empty_kernel(
    __nv_bfloat16* __restrict__ env_min,
    __nv_bfloat16* __restrict__ env_max,
    std::size_t n)
{
    const std::size_t i =
        static_cast<std::size_t>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (i >= n) return;
    env_min[i] = __float2bfloat16(INFINITY);
    env_max[i] = __float2bfloat16(-INFINITY);
}

// Single-launch form of the reset+merge pair below, for fires narrow enough
// that the launch overhead of the two-kernel version dominates -- which is
// every decode fire, and therefore the overwhelming majority of calls. Two
// launches cost ~4.6 us per layer on an L40S regardless of how few tokens they
// carry, so at 28 layers the pair is ~129 us of pure dispatch on the critical
// path of every step; this halves that.
//
// It removes the atomics too. Blocks elect ONE writer per (page, kv_head) --
// the first valid token naming that page -- so that block owns the page's
// envelope outright and can gather its own fire's keys into registers and
// store once. Sole ownership is also what makes the reset safe to fold in:
// the race the two-kernel split exists to avoid (a reset erasing a key merged
// by another token of the same fire) cannot arise when the same block does
// both, in order, for every token on the page.
//
// Thread 0 would serialise the O(num_tokens) scans, which shows up as soon as
// a fire carries more than a handful of tokens, so the scan is strided across
// the block and the gathered list is built with shared-memory atomics. Order
// within the list is irrelevant -- it feeds a min/max.
constexpr int kEnvelopeFuseMaxTokens = 128;

__global__ void envelope_merge_written_fused_kernel(
    const __nv_bfloat16* __restrict__ k_curr,
    const std::uint32_t* __restrict__ w_page,
    const std::uint32_t* __restrict__ w_off,
    const std::uint8_t* __restrict__ row_valid,
    __nv_bfloat16* __restrict__ env_min,
    __nv_bfloat16* __restrict__ env_max,
    int num_tokens,
    int num_kv_heads,
    int head_dim)
{
    __shared__ int s_mine[kEnvelopeFuseMaxTokens];
    __shared__ int s_count;
    __shared__ int s_started;
    __shared__ int s_taken;

    const int token = blockIdx.x;
    const int kh = blockIdx.y;
    if (token >= num_tokens) return;

    if (threadIdx.x == 0) {
        s_count = 0;
        s_started = 0;
        s_taken = 0;
    }
    __syncthreads();

    // Uniform across the block: it depends only on blockIdx.
    if (row_valid != nullptr && row_valid[token] == 0) return;
    const std::uint32_t page = w_page[token];

    for (int t = threadIdx.x; t < num_tokens; t += blockDim.x) {
        if (row_valid != nullptr && row_valid[t] == 0) continue;
        if (w_page[t] != page) continue;
        if (t < token) {
            atomicOr(&s_taken, 1);  // an earlier token owns this page
            continue;
        }
        s_mine[atomicAdd(&s_count, 1)] = t;
        if (w_off[t] == 0u) atomicOr(&s_started, 1);
    }
    __syncthreads();
    if (s_taken != 0) return;

    const long env_base =
        (static_cast<long>(page) * num_kv_heads + kh) * head_dim;
    const int count = s_count;
    const bool started = s_started != 0;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float lo = CUDART_INF_F;
        float hi = -CUDART_INF_F;
        for (int i = 0; i < count; ++i) {
            const long src =
                (static_cast<long>(s_mine[i]) * num_kv_heads + kh) * head_dim;
            const float v = __bfloat162float(k_curr[src + d]);
            lo = fminf(lo, v);
            hi = fmaxf(hi, v);
        }
        // Started in this fire => the page was recycled, so the previous
        // tenant's bound is dead and must be replaced, not widened.
        if (!started) {
            lo = fminf(lo, __bfloat162float(env_min[env_base + d]));
            hi = fmaxf(hi, __bfloat162float(env_max[env_base + d]));
        }
        env_min[env_base + d] = __float2bfloat16_rd(lo);
        env_max[env_base + d] = __float2bfloat16_ru(hi);
    }
}

// bf16 min/max via CAS on the bit pattern. bf16 IS the top 16 bits of an
// IEEE-754 float, so the ordering argument is the float one: the values in play
// are real keys plus the `+inf`/`-inf` seed, the ordinary comparison inside the
// loop is what decides, and the CAS only serialises the update.
//
// The stored value is rounded AWAY from the incoming key -- down for the
// minimum, up for the maximum -- so a widening can never round back inside the
// true range. The keys are themselves bf16, so in practice both roundings are
// exact and this costs nothing; it is here so that the envelope stays a valid
// bound if a caller ever merges a value that is not already a bf16.
__device__ inline void envelope_atomic_min(__nv_bfloat16* addr, float value) {
    unsigned short* as_u16 = reinterpret_cast<unsigned short*>(addr);
    const unsigned short want = __bfloat16_as_ushort(__float2bfloat16_rd(value));
    unsigned short old = *as_u16;
    unsigned short assumed;
    do {
        if (__bfloat162float(__ushort_as_bfloat16(old)) <= value) return;
        assumed = old;
        old = atomicCAS(as_u16, assumed, want);
    } while (assumed != old);
}

__device__ inline void envelope_atomic_max(__nv_bfloat16* addr, float value) {
    unsigned short* as_u16 = reinterpret_cast<unsigned short*>(addr);
    const unsigned short want = __bfloat16_as_ushort(__float2bfloat16_ru(value));
    unsigned short old = *as_u16;
    unsigned short assumed;
    do {
        if (__bfloat162float(__ushort_as_bfloat16(old)) >= value) return;
        assumed = old;
        old = atomicCAS(as_u16, assumed, want);
    } while (assumed != old);
}

// Companion to the merge below: a page whose FIRST cell is being written in
// this fire is being started from scratch, so whatever its envelope holds
// belongs to content that is now dead (the page was recycled through the
// pool). Reset it to empty before the merge widens it, or the bound would
// accumulate every request that ever used the page and converge on "keep
// everything". Pages entered at a non-zero offset are being continued and must
// keep what they have.
//
// A separate launch, not a branch inside the merge: two tokens of the same
// fire can land on one page, and a reset racing a merge would erase a key.
__global__ void envelope_reset_started_pages_kernel(
    const std::uint32_t* __restrict__ w_page,
    const std::uint32_t* __restrict__ w_off,
    const std::uint8_t* __restrict__ row_valid,
    __nv_bfloat16* __restrict__ env_min,
    __nv_bfloat16* __restrict__ env_max,
    int num_tokens,
    int num_kv_heads,
    int head_dim)
{
    const int token = blockIdx.x;
    const int kh = blockIdx.y;
    if (token >= num_tokens) return;
    if (row_valid != nullptr && row_valid[token] == 0) return;
    if (w_off[token] != 0u) return;

    const long env_base =
        (static_cast<long>(w_page[token]) * num_kv_heads + kh) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        env_min[env_base + d] = __float2bfloat16(CUDART_INF_F);
        env_max[env_base + d] = __float2bfloat16(-CUDART_INF_F);
    }
}

// Envelope maintenance for the EXPLICIT-descriptor KV write, where the program
// names the (physical page, offset) for every token instead of letting the
// kernel derive it from the CSR. There is no page list to walk and no live
// length to recompute from, so this MERGES each written key into the page's
// existing envelope rather than recomputing the page.
//
// Merging is the right operation here, not a shortcut:
//   * for the append-only case it is exactly equal to a full recompute,
//     because the seed is the empty envelope (+inf, -inf), the reset pass
//     above restores that seed whenever a page is started, and every key that
//     has ever been written to the page since then has passed through here;
//   * for the beam fork/freeze case -- the reason the explicit path exists at
//     all -- a cell can be REWRITTEN mid-page, and a recompute keyed on the
//     descriptor's offset would shrink the envelope to a prefix and could drop
//     a page that still holds a live key. Merging only ever widens, so the
//     bound stays an upper bound, which is the direction Quest must fail in.
//
// One block per (token, kv_head); threads stride over head_dim.
__global__ void envelope_merge_written_kernel(
    const __nv_bfloat16* __restrict__ k_curr,
    const std::uint32_t* __restrict__ w_page,
    const std::uint8_t* __restrict__ row_valid,
    __nv_bfloat16* __restrict__ env_min,
    __nv_bfloat16* __restrict__ env_max,
    int num_tokens,
    int num_kv_heads,
    int head_dim)
{
    const int token = blockIdx.x;
    const int kh = blockIdx.y;
    if (token >= num_tokens) return;
    if (row_valid != nullptr && row_valid[token] == 0) return;

    const long src_base =
        (static_cast<long>(token) * num_kv_heads + kh) * head_dim;
    const long env_base =
        (static_cast<long>(w_page[token]) * num_kv_heads + kh) * head_dim;

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float v = __bfloat162float(k_curr[src_base + d]);
        envelope_atomic_min(&env_min[env_base + d], v);
        envelope_atomic_max(&env_max[env_base + d], v);
    }
}

}  // namespace

void launch_envelope_merge_written_bf16(
    const std::uint16_t* k_curr,
    const std::uint32_t* w_page,
    const std::uint32_t* w_off,
    const std::uint8_t* row_valid,
    std::uint16_t* env_min,
    std::uint16_t* env_max,
    int num_tokens,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream)
{
    if (num_tokens <= 0 || num_kv_heads <= 0 || head_dim <= 0) return;
    const dim3 grid(static_cast<unsigned>(num_tokens),
                    static_cast<unsigned>(num_kv_heads));
    const int threads = head_dim < 256 ? head_dim : 256;
    if (num_tokens <= kEnvelopeFuseMaxTokens) {
        envelope_merge_written_fused_kernel<<<grid, threads, 0, stream>>>(
            reinterpret_cast<const __nv_bfloat16*>(k_curr),
            w_page, w_off, row_valid,
            reinterpret_cast<__nv_bfloat16*>(env_min),
        reinterpret_cast<__nv_bfloat16*>(env_max),
            num_tokens, num_kv_heads, head_dim);
        return;
    }
    envelope_reset_started_pages_kernel<<<grid, threads, 0, stream>>>(
        w_page, w_off, row_valid,
        reinterpret_cast<__nv_bfloat16*>(env_min),
        reinterpret_cast<__nv_bfloat16*>(env_max),
        num_tokens, num_kv_heads, head_dim);
    envelope_merge_written_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(k_curr),
        w_page, row_valid,
        reinterpret_cast<__nv_bfloat16*>(env_min),
        reinterpret_cast<__nv_bfloat16*>(env_max),
        num_tokens, num_kv_heads, head_dim);
}

void launch_envelope_seed_empty_bf16(
    std::uint16_t* env_min,
    std::uint16_t* env_max,
    int num_pages,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream)
{
    if (num_pages <= 0 || num_kv_heads <= 0 || head_dim <= 0) return;
    const std::size_t n = static_cast<std::size_t>(num_pages) *
                          static_cast<std::size_t>(num_kv_heads) *
                          static_cast<std::size_t>(head_dim);
    const int threads = 256;
    const std::size_t blocks = (n + threads - 1) / threads;
    envelope_seed_empty_kernel<<<static_cast<unsigned>(blocks), threads, 0,
                                 stream>>>(
        reinterpret_cast<__nv_bfloat16*>(env_min),
        reinterpret_cast<__nv_bfloat16*>(env_max), n);
}

void launch_envelope_recompute_bf16(
    const std::uint16_t* k_pages,
    const std::int32_t* page_live_lens,
    std::uint16_t* env_min,
    std::uint16_t* env_max,
    int num_pages,
    int page_size,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream)
{
    if (num_pages <= 0 || num_kv_heads <= 0 || head_dim <= 0) return;
    const dim3 grid(static_cast<unsigned>(num_pages),
                    static_cast<unsigned>(num_kv_heads));
    const int threads = head_dim < 256 ? head_dim : 256;
    envelope_recompute_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(k_pages),
        page_live_lens,
        reinterpret_cast<__nv_bfloat16*>(env_min),
        reinterpret_cast<__nv_bfloat16*>(env_max),
        page_size, num_kv_heads, head_dim);
}

void launch_envelope_dot_f32(
    const float* q,
    const std::uint16_t* env_min,
    const std::uint16_t* env_max,
    float* score,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int p_max,
    int live_pages,
    cudaStream_t stream)
{
    if (p_max <= 0 || num_kv_heads <= 0) return;
    constexpr int BLOCK = 128;
    const dim3 grid(static_cast<unsigned>(p_max),
                    static_cast<unsigned>(num_kv_heads));
    envelope_dot_kernel<BLOCK><<<grid, BLOCK, 0, stream>>>(
        q,
        reinterpret_cast<const __nv_bfloat16*>(env_min),
        reinterpret_cast<const __nv_bfloat16*>(env_max),
        score,
        num_q_heads, num_kv_heads, head_dim, p_max, live_pages);
}

void launch_envelope_update_appended_bf16(
    const std::uint16_t* k_pages,
    const std::uint32_t* qo_indptr,
    const std::uint32_t* kv_page_indices,
    const std::uint32_t* kv_page_indptr,
    const std::uint32_t* kv_last_page_lens,
    std::uint16_t* env_min,
    std::uint16_t* env_max,
    int num_requests,
    int max_touched,
    int page_size,
    int num_kv_heads,
    int head_dim,
    cudaStream_t stream)
{
    if (num_requests <= 0 || max_touched <= 0 || num_kv_heads <= 0 ||
        head_dim <= 0 || page_size <= 0) {
        return;
    }
    const dim3 grid(static_cast<unsigned>(max_touched),
                    static_cast<unsigned>(num_kv_heads));
    const int threads = head_dim < 256 ? head_dim : 256;
    envelope_update_appended_kernel<<<grid, threads, 0, stream>>>(
        reinterpret_cast<const __nv_bfloat16*>(k_pages),
        qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
        reinterpret_cast<__nv_bfloat16*>(env_min),
        reinterpret_cast<__nv_bfloat16*>(env_max),
        num_requests, page_size, num_kv_heads, head_dim);
}

}  // namespace pie_cuda_driver::kernels::layout
