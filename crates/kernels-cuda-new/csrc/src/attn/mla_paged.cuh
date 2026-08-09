//===-- mla_paged.cuh - the MLA prologue and the MLA page write -----------===//
//
// Two `__global__`s, the two `__device__` helpers they share, and the shared
// bound the rope cache is sized on. No host code: `mla_paged.cu` includes this
// and keeps both `<<<>>>`, so the ahead-of-time build and NVRTC compile ONE
// text -- which is the whole point of the split, because two copies that agree
// today are two kernels that drift, each right for whichever half of the tree
// its tests exercise. `norm/altup_aux` shipped exactly that for a release with
// every test green.
//
// # What lives here
//
// `write_mla` is the MLA cache's append: one block per current-step token,
// copying the compressed latent and the rotated `k_pe` into the page slot the
// CSR resolves to.
//
// `mla_prepare<BLOCK_DIM>` is the fused prologue -- `split_kv_a` + RMSNorm,
// `split_q_b`, RoPE and that same page write, in one launch. The four kernels
// it replaces are all per-token with no cross-token dependency, and at decode
// shapes each cost about what an empty kernel costs: 70 us/step of which
// roughly 62 us was launch latency and inter-kernel gap, against ~8 us of
// arithmetic. **Nothing here is faster than it was; there is simply one launch
// instead of four.**
//
// `blockIdx.x` is the token. `blockIdx.y == 0` is the KV lane (norm, `k_pe`
// rotation, page write); `blockIdx.y >= 1` splits the query heads. The KV lane
// must own both the rotation and the write, because the write consumes the
// rotated value and a cross-block dependency inside one kernel would need a
// grid sync.
//
// # Why the helpers carry an `mla_` prefix
//
// `attn/kv_paged.cuh` already defines `attn::device::find_request` with the
// same body, and `attn/attention_naive.cuh` defines `find_request_u32`. Two
// identical `__device__ __forceinline__` definitions of one name in one
// namespace is a redefinition, not a merge, so any translation unit that
// carried both headers would stop compiling. The names are therefore local to
// this file's subject -- the same convention `kv_paged_addr.cuh`'s
// `kv_find_request` follows.
//
// They are NOT merged into one shared helper for the reason `kv_paged.cuh`
// gives about its own addressing: *"merging them would make one kernel's
// index bug the other's."* The MLA cache and the KV cache resolve a page from
// the same CSR shape and are two caches; `smallop_bench` keeps the copies
// honest against each other.
//
// # `rope_device.cuh` is included, not copied
//
// `rope_cos_sin`, `rope_cos_sin_yarn_original`, `rotate_pair_to` and
// `rotate_pair_interleaved_to` live in `pie_cuda_driver::kernels`, one
// namespace out, and this header reaches them the same way the `.cu` did.
// That file is already in the JIT crate's tree and already NVRTC-clean, so
// the include costs nothing and copying four rotations would be four more
// texts to drift.
//
// # Which launcher becomes a row, and which does not
//
// Neither, and the two refusals are different.
//
//  * `write_mla` launches `<<<total_tokens, 256>>>`: one block per token, a
//    fixed 256 threads, a stride loop over `kv_lora_rank` and then over
//    `qk_rope_head_dim`. That is the same shape `kv_paged.cuh`'s six write
//    forms have and the same refusal applies verbatim -- there is no rule for
//    it. `Rms` is one block per row at 256 but asks for 32 bytes of shared
//    memory and means a REDUCTION; `RouteRows` sizes the block from the row
//    width and would launch a different geometry. Naming either would be
//    inventing a rule under an existing name.
//  * `mla_prepare` launches `dim3 grid(total_tokens, 1 + q_blocks)` -- a
//    second grid axis that is neither heads nor rows but `1 + ceil(heads /
//    heads_per_block)`, where `heads_per_block` is itself derived on the host
//    from `half >= BS ? 1 : BS / half`. No ported rule computes a grid axis
//    from a head dimension and a block width, and the leading `1` is the KV
//    lane, which is not a head at all. Whoever states it has to state that
//    arithmetic, and `runtime::launch`'s own header says a rule with no cited
//    launcher is a guess.
//
// `mla_prepare`'s `BLOCK_DIM` is a value the kernel is compiled AGAINST, not
// a decoration: it sizes `__shared__ float buf[BLOCK_DIM]` and fixes the
// halving reduction over it, so a row that picks the wrong one reduces
// through shared memory the launch never wrote. §17.6's rule applies --
// take it from the ahead-of-time launcher and cite it. `mla_paged.cu` opens
// `constexpr int BS = 256` and launches `<BS>` at `BS` threads; **256 is the
// only value instantiated anywhere.**
//
// # Linkage
//
// `write_mla` is a non-template `__global__`, so §21.6's measurement applies:
// the host stub and the function both take external linkage and **this header
// may be included by exactly one translation unit**, which is `mla_paged.cu`.
// A second includer is a hard `multiple definition` at link even when it never
// launches it. It stays a non-template because it has no honest parameter --
// every buffer is `bf16` and the block width reaches it as `blockDim.x` --
// and `mxfp4_marlin.cuh`'s refusal is the precedent: *"a width parameter
// would be a lie that compiles."* `mla_prepare<BLOCK_DIM>` carries no such
// constraint, being a template already.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "pie_device.cuh"
#include "rope_device.cuh"

namespace pie_cuda_driver::kernels::attn::device {

// Pulled in by name rather than by `using namespace`, so that `device::` here
// and in `mla_paged.cu` means the same thing: inside `attn::device` the
// qualifier resolves to THIS namespace, and a prelude name not re-exported
// here would stop resolving in the `.cu` the moment it includes this header.
using ::pie_cuda_driver::kernels::device::bf16;
using ::pie_cuda_driver::kernels::device::bf16_to_f32;
using ::pie_cuda_driver::kernels::device::f32_to_bf16;
using ::pie_cuda_driver::kernels::device::i32;
using ::pie_cuda_driver::kernels::device::u32;
using ::pie_cuda_driver::kernels::device::u8;

/// 32 KB of shared would cap this far higher; every MLA model here uses 64.
///
/// Named in the header rather than in the `.cu` because `mla_prepare` sizes
/// `__shared__ float cs[2 * kMaxRopePairs]` on it. `mla_paged.cu`'s
/// `mla_prepare_supported` reads it as `device::kMaxRopePairs` to refuse a
/// head dim the cache cannot hold, so the predicate and the array are one
/// constant and not two.
constexpr int kMaxRopePairs = 128;

/// Which request token `token_idx` belongs to.
///
/// Linear scan -- `R` is bounded by the batch size, a few hundred at most.
/// Prefixed `mla_` because `kv_paged.cuh` defines the same body as
/// `find_request` in this namespace; see the header comment.
__device__ __forceinline__ int mla_find_request(const u32* qo_indptr,
                                                int R,
                                                int token_idx) {
    for (int r = 0; r < R; ++r) {
        if (token_idx < static_cast<int>(qo_indptr[r + 1])) return r;
    }
    return R - 1;
}

/// The page and the slot within it that `token_idx` writes.
///
///     pre_kv_len   = total_kv_after - new_tokens_r
///     abs_kv_pos   = pre_kv_len + offset_in_new
///     actual_page  = kv_page_indices[kv_page_indptr[r] + abs_kv_pos / page_size]
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

/// One block per current-step token: the compressed latent and the rotated
/// `k_pe`, into the page slot the CSR resolves to.
__global__ void write_mla(
    const device::bf16* __restrict__ ckv_curr,
    const device::bf16* __restrict__ kpe_curr,
    device::bf16* __restrict__ ckv_pages,
    device::bf16* __restrict__ kpe_pages,
    const device::u32* __restrict__ qo_indptr,
    const device::u32* __restrict__ kv_page_indices,
    const device::u32* __restrict__ kv_page_indptr,
    const device::u32* __restrict__ kv_last_page_lens,
    const device::u8* __restrict__ row_valid,
    int R,
    int page_size,
    int kv_lora_rank,
    int qk_rope_head_dim)
{
    const int t = blockIdx.x;
    if (row_valid != nullptr && row_valid[t] == 0) return;
    int actual_page = 0;
    int offset_in_page = 0;
    mla_resolve_dst(qo_indptr, kv_page_indices, kv_page_indptr,
                    kv_last_page_lens, R, page_size, t, actual_page,
                    offset_in_page);

    const long long ckv_src = static_cast<long long>(t) * kv_lora_rank;
    const long long ckv_dst =
        (static_cast<long long>(actual_page) * page_size + offset_in_page) *
        kv_lora_rank;
    for (int i = threadIdx.x; i < kv_lora_rank; i += blockDim.x) {
        ckv_pages[ckv_dst + i] = ckv_curr[ckv_src + i];
    }

    const long long kpe_src = static_cast<long long>(t) * qk_rope_head_dim;
    const long long kpe_dst =
        (static_cast<long long>(actual_page) * page_size + offset_in_page) *
        qk_rope_head_dim;
    for (int i = threadIdx.x; i < qk_rope_head_dim; i += blockDim.x) {
        kpe_pages[kpe_dst + i] = kpe_curr[kpe_src + i];
    }
}

/// Fused MLA prologue: split_kv_a+rmsnorm, split_q_b, RoPE and the paged-cache
/// write in one launch.
///
/// Layout: blockIdx.x is the token. blockIdx.y == 0 handles the KV lane (norm,
/// k_pe rotation, page write); blockIdx.y >= 1 splits the query heads. The KV
/// lane must own both the k_pe rotation and the page write, because the write
/// consumes the rotated value and a cross-block dependency inside one kernel
/// would need a grid sync.
template <int BLOCK_DIM>
__global__ void mla_prepare(
    const device::bf16* __restrict__ kv_a,
    const device::bf16* __restrict__ kv_a_norm_w,
    const device::bf16* __restrict__ q_b,
    device::bf16* __restrict__ kv_c,
    device::bf16* __restrict__ k_pe,
    device::bf16* __restrict__ q_nope,
    device::bf16* __restrict__ q_pe,
    device::bf16* __restrict__ ckv_pages,
    device::bf16* __restrict__ kpe_pages,
    const device::i32* __restrict__ positions,
    const device::u32* __restrict__ qo_indptr,
    const device::u32* __restrict__ kv_page_indices,
    const device::u32* __restrict__ kv_page_indptr,
    const device::u32* __restrict__ kv_last_page_lens,
    const device::u8* __restrict__ row_valid,
    int R,
    int page_size,
    int heads,
    int kv_lora,
    int nope,
    int rope,
    int src_row_stride,
    float eps,
    float theta,
    bool interleaved,
    int heads_per_block,
    // Original-YaRN scaling (Kimi/DeepSeek). factor <= 0 selects plain RoPE.
    float yarn_factor,
    float yarn_low_dim,
    float yarn_high_dim,
    float yarn_mscale)
{
    const int n = blockIdx.x;
    const int tid = threadIdx.x;
    const int half = rope / 2;
    const int pos = positions[n];

    // The angle depends only on (pos, dim_pair), so every head of this token
    // shares it. rope is 64 for every MLA model here, so 32 pairs; a static
    // array avoids the "two extern __shared__ of different types in one TU"
    // problem and costs 512 B.
    __shared__ float cs[2 * kMaxRopePairs];
    const int cached = half <= kMaxRopePairs ? half : 0;
    const bool yarn = yarn_factor > 0.f;
    auto angle = [&](int dp, float& c, float& s_) {
        if (yarn) {
            rope_cos_sin_yarn_original(theta, dp, rope, pos, yarn_factor,
                                       yarn_low_dim, yarn_high_dim,
                                       yarn_mscale, c, s_);
        } else {
            rope_cos_sin(theta, dp, rope, pos, c, s_);
        }
    };
    for (int dp = tid; dp < cached; dp += BLOCK_DIM) {
        angle(dp, cs[dp], cs[cached + dp]);
    }
    if (cached > 0) __syncthreads();

    if (blockIdx.y == 0) {
        const device::bf16* row =
            kv_a + static_cast<long long>(n) * src_row_stride;
        device::bf16* kpe_out = k_pe + static_cast<long long>(n) * rope;

        // k_pe: copy out of the projection and rotate in one pass. The
        // standalone pair wrote the unrotated value to memory and read it back.
        for (int dp = tid; dp < half; dp += BLOCK_DIM) {
            float cos_v, sin_v;
            if (dp < cached) { cos_v = cs[dp]; sin_v = cs[cached + dp]; }
            else angle(dp, cos_v, sin_v);
            if (interleaved) {
                rotate_pair_interleaved_to(row + kv_lora, kpe_out, dp, cos_v, sin_v);
            } else {
                rotate_pair_to(row + kv_lora, kpe_out, half, dp, cos_v, sin_v);
            }
        }

        // RMSNorm over the latent half. Same reduction tree and same block
        // width as `split_kv_a_norm_kernel`, so the sum is bit-identical.
        float local = 0.f;
        for (int d = tid; d < kv_lora; d += BLOCK_DIM) {
            const float v = device::bf16_to_f32(row[d]);
            local += v * v;
        }
        __shared__ float buf[BLOCK_DIM];
        buf[tid] = local;
        __syncthreads();
        for (int off = BLOCK_DIM / 2; off > 0; off >>= 1) {
            if (tid < off) buf[tid] += buf[tid + off];
            __syncthreads();
        }
        const float inv_rms =
            rsqrtf(buf[0] / static_cast<float>(kv_lora) + eps);

        device::bf16* kvc_out = kv_c + static_cast<long long>(n) * kv_lora;
        for (int d = tid; d < kv_lora; d += BLOCK_DIM) {
            const float v = device::bf16_to_f32(row[d]);
            const float w = device::bf16_to_f32(kv_a_norm_w[d]);
            kvc_out[d] = device::f32_to_bf16(v * inv_rms * w);
        }

        if (row_valid != nullptr && row_valid[n] == 0) return;
        __syncthreads();

        int actual_page = 0;
        int offset_in_page = 0;
        mla_resolve_dst(qo_indptr, kv_page_indices, kv_page_indptr,
                        kv_last_page_lens, R, page_size, n,
                        actual_page, offset_in_page);
        const long long slot =
            static_cast<long long>(actual_page) * page_size + offset_in_page;
        device::bf16* ckv_dst = ckv_pages + slot * kv_lora;
        for (int d = tid; d < kv_lora; d += BLOCK_DIM) ckv_dst[d] = kvc_out[d];
        device::bf16* kpe_dst = kpe_pages + slot * rope;
        for (int d = tid; d < rope; d += BLOCK_DIM) kpe_dst[d] = kpe_out[d];
        return;
    }

    // Query lane: split q_b into (q_nope, q_pe) and rotate q_pe.
    const int per = nope + rope;
    const int head_base = (blockIdx.y - 1) * heads_per_block;
    const int heads_here = min(heads_per_block, heads - head_base);
    if (heads_here <= 0) return;

    const device::bf16* qb_row =
        q_b + (static_cast<long long>(n) * heads + head_base) * per;
    device::bf16* qn_row =
        q_nope + (static_cast<long long>(n) * heads + head_base) * nope;
    for (int i = tid; i < heads_here * nope; i += BLOCK_DIM) {
        const int h = i / nope;
        qn_row[i] = qb_row[static_cast<long long>(h) * per + (i - h * nope)];
    }

    device::bf16* qp_row =
        q_pe + (static_cast<long long>(n) * heads + head_base) * rope;
    for (int i = tid; i < heads_here * half; i += BLOCK_DIM) {
        const int h = i / half;
        const int dp = i - h * half;
        float cos_v, sin_v;
        if (dp < cached) { cos_v = cs[dp]; sin_v = cs[cached + dp]; }
        else angle(dp, cos_v, sin_v);
        const device::bf16* src =
            qb_row + static_cast<long long>(h) * per + nope;
        device::bf16* dst = qp_row + static_cast<long long>(h) * rope;
        if (interleaved) rotate_pair_interleaved_to(src, dst, dp, cos_v, sin_v);
        else rotate_pair_to(src, dst, half, dp, cos_v, sin_v);
    }
}

}  // namespace pie_cuda_driver::kernels::attn::device
