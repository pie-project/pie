//===-- dsv4_compress.cuh - deepseek_v4's compressed-KV device text ------===//
//
// Eleven `__global__`s and the two `__device__` helpers they call, and no
// host code at all. The `<<<>>>`s stay in `dsv4_compress.cu`, which includes
// this header rather than defining what it launches -- so nvcc and NVRTC
// compile ONE text. Two copies that agree today are two kernels that drift,
// and each stays right for whichever half of the tree its tests exercise:
// `norm/altup_aux` shipped exactly that for a release with every test green.
//
// # What this family computes
//
// deepseek_v4 attends through a SECOND KV cache beside the fine-grained one,
// holding one compressed entry per `ratio` tokens. Every query attends both
// and the two outputs are merged by their log-sum-exps -- exact algebra, not
// an approximation. The kernels here build that second cache (`average_pool`,
// `gated_softmax_pool`, `add_ape`, the two gathers, the store), attend it
// (`compressed_attn`, `compressed_attn_paged`), merge the halves
// (`combine_attn_outputs`), and emit the boundary metadata that says which
// tokens close a compression window (`dsv4_boundary_meta_*`).
//
// # Which launchers became rows, and which did not
//
// Six. `average_pool`, `add_ape` and `gated_softmax_pool` were each
// `total = out_tokens * dim`, a guard, and `<<<(total + 255) / 256, 256>>>`
// -- `LaunchRule::Elementwise` to the digit. The three gathers were
// `<<<num_entries, head_dim < 256 ? round32(head_dim) : 256>>>`, which is
// `LaunchRule::RouteRows` exactly: one block per entry, the row's width
// rounded to a warp and clamped at the rule's 1024. Both rules already return
// `Ungeometric::Empty` for a zero extent, which is what the `if (total <= 0)`
// and `if (num_entries <= 0)` guards were.
//
// Five did not, and they are named here so the absence is a record rather
// than an omission:
//
// * `combine_attn_outputs` launches `grid(N, num_heads)` with a block width
//   clamped into `[32, 256]`. The grid is `PerHeadElementwise` to the digit
//   -- token on `grid.x`, head on `grid.y` -- and `runtime::launch` ports
//   that rule now, so the geometry blocker this file recorded before the port
//   is gone. The BLOCK is not: the rule clamps into `[32, 128]`, and a row
//   that states a rule returning a different number than the launcher it was
//   checked against is a row nothing can falsify -- every loop in the kernel
//   strides `d += blockDim.x`, so the narrower block is slower and never
//   wrong, which means the disagreement would never surface as a failure.
//   The `__global__` IS a template as of this commit, so the row costs one
//   line the day the clamp is reconciled; `dsv4_compress.cu` carries the
//   finding beside the `<<<>>>` that states it.
// * `compressed_attn` needs a `cudaMallocAsync`'d `CompressedAttnParams[R]`
//   built on the host and copied in before the launch. A `LaunchRule`
//   computes a geometry; it cannot allocate, fill and free a side buffer.
// * `compressed_attn_paged` launches `grid(total_tokens, num_q_heads)` with
//   `(head_dim + 128) * sizeof(float)` of dynamic shared memory. No ported
//   rule computes a shared-memory size from an operand width.
// * `dsv4_boundary_meta_decode` and `dsv4_boundary_meta_paged` are blocked
//   TWICE and both times structurally. They read and write `int` and `u8`
//   arrays and have no element type at all, while
//   `DeviceKernel::instantiation()` spells `path<elem>` unconditionally: a
//   table that can only name a one-type template cannot name a kernel that
//   has no type, and giving them an unread `T` would put a parameter in the
//   source to satisfy the table's grammar, which is the opposite of what the
//   table is for. Their launchers are also `<<<ceil(n / 128), 128>>>` and
//   `LaunchRule::Elementwise` is 256 -- the same guard-bounded flat index at
//   half the block, which is a different launch even though it computes the
//   same bytes. An earlier revision of this header called them
//   "`Elementwise`"; they are that SHAPE and not that RULE.
//
// # Why seven are templates when the originals were not
//
// The originals were `_bf16` and only `_bf16`, because an ahead-of-time build
// has to choose its instantiations and nobody spends a translation unit on a
// format nobody asked for. Under a JIT the element type is the row's, so a
// second format costs a line in a table -- `norm/elementwise` measured that.
// The arithmetic is unchanged: widen to fp32, compute, narrow back, which is
// what the bf16 tolerance contract was measured against.
//
// The four that stay concrete stay concrete on purpose, and the reason is not
// the same as "no row can state it". `compressed_attn` and
// `compressed_attn_paged` are blocked by their HOST half, which a type
// parameter does not touch; the two boundary-meta kernels have no element
// type to parameterise. `combine_attn_outputs` is the one that was concrete
// only because nobody had needed it otherwise, and it is a template now.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "prelude/device.cuh"

namespace pie::attn {

template <class T>
__global__ void average_pool(
    const T* __restrict__ input,  // [N, dim]
    T* __restrict__ output,       // [N/ratio, dim]
    int N,
    int dim,
    int ratio)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_tokens = N / ratio;
    if (idx >= out_tokens * dim) return;

    const int d = idx % dim;
    const int out_tok = idx / dim;
    const int in_start = out_tok * ratio;

    float sum = 0.f;
    const int end = min(in_start + ratio, N);
    for (int t = in_start; t < end; ++t) {
        sum += Elem<T>::to_f32(input[static_cast<long long>(t) * dim + d]);
    }
    output[static_cast<long long>(out_tok) * dim + d] =
        Elem<T>::from_f32(sum / static_cast<float>(end - in_start));
}

template <class T>
__global__ void add_ape(
    T* __restrict__ data,    // [N_compressed, dim]
    const float* __restrict__ ape,       // [ratio, dim]
    int N_compressed,
    int dim,
    int ratio)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N_compressed * dim) return;

    const int d = idx % dim;
    const int tok = idx / dim;
    const int pos_in_window = tok % ratio;

    const float val = Elem<T>::to_f32(data[idx]) +
                      ape[pos_in_window * dim + d];
    data[idx] = Elem<T>::from_f32(val);
}

// ── Gated softmax pooling kernel ─────────────────────────────────────
// One thread per output element (group_idx, dim_idx).
// For each element: compute softmax over ratio consecutive scores, then
// weighted sum of kv values.
template <class T>
__global__ void gated_softmax_pool(
    const T* __restrict__ kv,      // [N, dim]
    const T* __restrict__ score,   // [N, dim]
    T* __restrict__ output,        // [N/ratio, dim]
    int N,
    int dim,
    int ratio)
{
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int out_tokens = N / ratio;
    if (idx >= out_tokens * dim) return;

    const int d = idx % dim;
    const int g = idx / dim;
    const int base = g * ratio;

    // Numerically-stable softmax over ratio elements at dimension d
    float max_s = neg_inf();
    for (int i = 0; i < ratio && (base + i) < N; ++i) {
        const float s = Elem<T>::to_f32(score[static_cast<long long>(base + i) * dim + d]);
        max_s = fmaxf(max_s, s);
    }

    float sum_exp = 0.f;
    float weighted_sum = 0.f;
    for (int i = 0; i < ratio && (base + i) < N; ++i) {
        const long long pos = static_cast<long long>(base + i) * dim + d;
        const float s = Elem<T>::to_f32(score[pos]);
        const float v = Elem<T>::to_f32(kv[pos]);
        const float e = expf(s - max_s);
        sum_exp += e;
        weighted_sum += v * e;
    }

    output[static_cast<long long>(g) * dim + d] =
        Elem<T>::from_f32(sum_exp > 0.f ? weighted_sum / sum_exp : 0.f);
}

/// Two attention halves and their log-sum-exps, merged into one — exact
/// algebra, not an approximation: `o = (w1·o1 + w2·o2) / (w1 + w2)` with the
/// weights taken relative to `max(lse1, lse2)` so neither exponential
/// overflows.
///
/// BASE TWO, because `attention.merge_lse` states base two, because an lse
/// is what an attention kernel HAS at the end and an attention kernel folds
/// `log2(e)` into its scale and runs on `exp2`. The weights are ratios, so
/// this fold is the same number in any base; what is NOT base-free is the
/// `lse_out` it leaves, which a sink correction downstream reads against a
/// checkpoint's natural-log logit. `exp2f`/`log2f` here and `expf`/`logf`
/// there would be the same bug in two places at once.
///
/// One block per (token, head). Threads stride along `head_dim`, and every
/// loop below is `for (d = threadIdx.x; d < head_dim; d += blockDim.x)` —
/// which is what makes the block width a performance choice rather than a
/// correctness one, and is the whole of the finding recorded on the launcher
/// side of this file.
///
/// A `-inf` on either side is a half that had no entries: the other half is
/// copied through untouched, in `T` and never via fp32, so a bf16 that
/// survives a fire with no compressed entries is bit-identical to the
/// attention output that produced it.
///
/// Templated as of this commit, and the retype is mechanical: `bf16_to_f32`
/// became `Elem<T>::to_f32`, `f32_to_bf16` became `Elem<T>::from_f32`, and
/// the two copy loops move `T` where they moved `bf16`. Widen to fp32,
/// compute, narrow back — which is what the original did and what the bf16
/// tolerance contract was measured against. It is a template because
/// `DeviceKernel::instantiation` spells `path<elem>` and cannot name a plain
/// `__global__` at all; the row that would use it is still blocked, and
/// `attn/dsv4_compress.cu` says by what.
template <class T>
__global__ void combine_attn_outputs(
    const T* __restrict__ o1,
    const float* __restrict__ lse1,
    const T* __restrict__ o2,
    const float* __restrict__ lse2,
    T* __restrict__ o_out,
    float* __restrict__ lse_out,
    int num_heads,
    int head_dim)
{
    const int n = blockIdx.x;  // token index
    const int h = blockIdx.y;  // head index

    const float l1 = lse1[n * num_heads + h];
    const float l2 = lse2[n * num_heads + h];

    // If lse2 is -inf, compressed attention had no entries — keep o1 unchanged.
    if (!isfinite(l2)) {
        // Copy o1 to o_out if they differ
        if (o1 != o_out) {
            const long long off = (static_cast<long long>(n) * num_heads + h) * head_dim;
            for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
                o_out[off + d] = o1[off + d];
            }
        }
        if (lse_out != nullptr && lse_out != lse1) {
            if (threadIdx.x == 0) lse_out[n * num_heads + h] = l1;
        }
        return;
    }

    // If lse1 is -inf (SWA had no entries — shouldn't happen but handle), use o2
    if (!isfinite(l1)) {
        const long long off = (static_cast<long long>(n) * num_heads + h) * head_dim;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            o_out[off + d] = o2[off + d];
        }
        if (lse_out != nullptr) {
            if (threadIdx.x == 0) lse_out[n * num_heads + h] = l2;
        }
        return;
    }

    const float lse_max = fmaxf(l1, l2);
    const float w1 = exp2f(l1 - lse_max);
    const float w2 = exp2f(l2 - lse_max);
    const float inv_total = 1.0f / (w1 + w2);

    const long long off = (static_cast<long long>(n) * num_heads + h) * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        const float v1 = Elem<T>::to_f32(o1[off + d]);
        const float v2 = Elem<T>::to_f32(o2[off + d]);
        o_out[off + d] = Elem<T>::from_f32((v1 * w1 + v2 * w2) * inv_total);
    }

    if (lse_out != nullptr && threadIdx.x == 0) {
        lse_out[n * num_heads + h] = lse_max + log2f(w1 + w2);
    }
}

// ── Dense attention over compressed KV (per-request causal) ──────────
// One block per (request, query_offset, head).
// Each block computes attention for one query row against the compressed KV
// entries of its request, with causal masking: query at local offset t can
// see compressed entry c only if c < (t+1) / ratio.
constexpr int ATTN_BLOCK = 128;

// Packed parameters uploaded to device memory before kernel launch.
struct CompressedAttnParams {
    int qo_lo;
    int qo_hi;
    int comp_offset;
    int comp_len;
    int comp_ratio;
};

__global__ void compressed_attn(
    const bf16* __restrict__ q,       // [total_tokens, num_q_heads, head_dim]
    const bf16* __restrict__ comp_kv, // [total_comp, head_dim]
    bf16* __restrict__ o,             // [total_tokens, num_q_heads, head_dim]
    float* __restrict__ lse_out,               // [total_tokens, num_q_heads] or nullptr
    const CompressedAttnParams* __restrict__ params, // [R]
    int num_q_heads,
    int head_dim,
    float scale)
{
    const int r       = blockIdx.x;
    const int qo_off  = blockIdx.y;
    const int q_head  = blockIdx.z;
    const int tid     = threadIdx.x;

    const auto& p = params[r];
    const int qo_lo = p.qo_lo;
    const int qo_hi = p.qo_hi;
    const int comp_off = p.comp_offset;
    const int comp_len = p.comp_len;
    const int ratio = p.comp_ratio;

    if (qo_lo + qo_off >= qo_hi) return;
    const int qi = qo_lo + qo_off;  // absolute query index in the batch

    // How many compressed entries can this query see?
    // Query at local position t (0-indexed within the request) can see
    // compressed entries [0, (t+1)/ratio). The local position is qo_off.
    const int num_visible = min((qo_off + 1) / ratio, comp_len);

    extern __shared__ float smem[];
    float* q_smem = smem;                  // [head_dim]
    float* reduce = smem + head_dim;       // [ATTN_BLOCK]

    // Load query vector into shared memory
    const bf16* q_row =
        q + (static_cast<long long>(qi) * num_q_heads + q_head) * head_dim;
    for (int d = tid; d < head_dim; d += ATTN_BLOCK) {
        q_smem[d] = bf16_to_f32(q_row[d]);
    }
    __syncthreads();

    // Output row
    bf16* o_row =
        o + (static_cast<long long>(qi) * num_q_heads + q_head) * head_dim;

    if (num_visible <= 0) {
        // No compressed entries visible — zero output, lse = -inf
        for (int d = tid; d < head_dim; d += ATTN_BLOCK) {
            o_row[d] = f32_to_bf16(0.f);
        }
        if (lse_out != nullptr && tid == 0) {
            lse_out[qi * num_q_heads + q_head] = neg_inf();
        }
        return;
    }

    // Two-pass attention: find max score, then compute exp-weighted sum
    // Pass 1: find max score
    float local_max = neg_inf();
    for (int c = tid; c < num_visible; c += ATTN_BLOCK) {
        const bf16* k_row =
            comp_kv + static_cast<long long>(comp_off + c) * head_dim;
        float dot = 0.f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q_smem[d] * bf16_to_f32(k_row[d]);
        }
        local_max = fmaxf(local_max, dot * scale);
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (int off = ATTN_BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) reduce[tid] = fmaxf(reduce[tid], reduce[tid + off]);
        __syncthreads();
    }
    const float row_max = reduce[0];

    // Pass 2: compute exp-weighted sum of V
    const int dims_per_thread = (head_dim + ATTN_BLOCK - 1) / ATTN_BLOCK;
    float acc[8] = {};  // dims_per_thread <= 8 (head_dim <= 1024)
    float local_z = 0.f;

    for (int c = 0; c < num_visible; ++c) {
        // Compute score for this compressed entry
        const bf16* k_row =
            comp_kv + static_cast<long long>(comp_off + c) * head_dim;
        float dot = 0.f;
        for (int d = tid; d < head_dim; d += ATTN_BLOCK) {
            dot += q_smem[d] * bf16_to_f32(k_row[d]);
        }
        reduce[tid] = dot;
        __syncthreads();
        for (int off = ATTN_BLOCK / 2; off > 0; off >>= 1) {
            if (tid < off) reduce[tid] += reduce[tid + off];
            __syncthreads();
        }
        const float w = expf(reduce[0] * scale - row_max);
        if (tid == 0) local_z += w;
        __syncthreads();

        // Accumulate V (compressed KV serves as both K and V — MLA style)
        for (int i = 0; i < dims_per_thread; ++i) {
            const int d = tid + i * ATTN_BLOCK;
            if (d < head_dim) {
                acc[i] += w * bf16_to_f32(k_row[d]);
            }
        }
    }

    __shared__ float z_shared;
    if (tid == 0) z_shared = local_z;
    __syncthreads();
    const float inv_z = z_shared > 0.f ? 1.0f / z_shared : 0.f;

    // `pool.attention_lse` states base two ONE MULTIPLY AWAY: this row
    // accumulated on `expf` against a natural-log `row_max`, so the lse it
    // has is `ln`, and the slot it writes is the one a merge against a
    // flashinfer reading folds. Rebasing here costs a `fmul` on one thread
    // per (token, head); rebasing the OTHER side would cost a launch over
    // the whole rectangle.
    if (lse_out != nullptr && tid == 0) {
        constexpr float kLog2e = 1.44269504088896340736f;
        lse_out[qi * num_q_heads + q_head] =
            z_shared > 0.f ? ((logf(z_shared) + row_max) * kLog2e) : neg_inf();
    }

    for (int i = 0; i < dims_per_thread; ++i) {
        const int d = tid + i * ATTN_BLOCK;
        if (d < head_dim) {
            o_row[d] = f32_to_bf16(acc[i] * inv_z);
        }
    }
}


template <class T>
__global__ void dsv4_compress_gather(
    const T* __restrict__ kv_proj,
    const T* __restrict__ score_proj,
    const float* __restrict__ ape,
    const i32* __restrict__ boundary_tok,
    const i32* __restrict__ boundary_pos,
    const i32* __restrict__ window_lo,
    T* __restrict__ out,
    int head_dim,
    int ratio,
    int coff) {
    const int c = blockIdx.x;
    const int window = coff * ratio;
    const int proj_dim = coff * head_dim;
    const int btok = boundary_tok[c];
    const int bpos = boundary_pos[c];
    const int lo = window_lo[c];

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float max_s = neg_inf();
        for (int i = 0; i < window; ++i) {
            const int rel = i - (window - 1);
            const int tok = btok + rel;
            const int pos = bpos + rel;
            if (tok < lo || pos < 0) continue;
            const int col = ((i >= ratio) ? head_dim : 0) + d;
            float s = Elem<T>::to_f32(score_proj[static_cast<long long>(tok) * proj_dim + col]);
            if (ape != nullptr) {
                s += ape[static_cast<long long>(pos % ratio) * proj_dim + col];
            }
            max_s = fmaxf(max_s, s);
        }
        if (!isfinite(max_s)) {
            out[static_cast<long long>(c) * head_dim + d] = Elem<T>::from_f32(0.0f);
            continue;
        }
        float sum_e = 0.0f;
        float acc = 0.0f;
        for (int i = 0; i < window; ++i) {
            const int rel = i - (window - 1);
            const int tok = btok + rel;
            const int pos = bpos + rel;
            if (tok < lo || pos < 0) continue;
            const int col = ((i >= ratio) ? head_dim : 0) + d;
            const long long idx = static_cast<long long>(tok) * proj_dim + col;
            float s = Elem<T>::to_f32(score_proj[idx]);
            if (ape != nullptr) {
                s += ape[static_cast<long long>(pos % ratio) * proj_dim + col];
            }
            const float e = __expf(s - max_s);
            sum_e += e;
            acc += e * Elem<T>::to_f32(kv_proj[idx]);
        }
        out[static_cast<long long>(c) * head_dim + d] =
            Elem<T>::from_f32(sum_e > 0.0f ? acc / sum_e : 0.0f);
    }
}


__device__ __forceinline__ long long paged_slot(
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    int req, int pos, int page_size) {
    const u32 page =
        kv_page_indices[kv_page_indptr[req] + pos / page_size];
    return static_cast<long long>(page) * page_size + (pos % page_size);
}

// Device-side boundary metadata for the pure-decode path.
//
// The host path scans `positions` on the CPU and emits a compacted boundary
// list, which needs a D2H copy plus a stream sync and so makes the whole
// forward ineligible for CUDA graph capture. In pure decode every request
// contributes exactly one token, so instead we emit one *fixed* slot per
// token and mark non-boundaries with `pos = -1`; the consumers skip those.
// Nothing is read back to the host, so the layer is capturable.
//
// **`T` is unread, and it is here so a row can NAME this kernel.**
// `DeviceKernel::instantiation()` always emits `path<...>`, so a plain
// `__global__` is unspellable from the table however simple its launch is.
// The parameter is DEFAULTED, which is the whole of the trick: measured under
// nvcc 13 on this tree's `sm_89`, an un-edited call site writing
// `dsv4_boundary_meta_decode<<<blocks, threads, 0, stream>>>(...)`
// still compiles, so `attn/dsv4_compress.cu` needed no change and the archive
// emits exactly the instructions it emitted before. Templating a `__global__`
// and editing its call site is what `combine_attn_outputs` had to do; a
// default argument does the same job across a file boundary a migration may
// not cross.
template <class T = i32>
__global__ void dsv4_boundary_meta_decode(
    const i32* __restrict__ positions,
    i32* __restrict__ out_pos,
    i32* __restrict__ out_req,
    i32* __restrict__ out_rope,
    int n,
    int ratio,
    const u8* __restrict__ row_valid) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const int p = positions[t];
    // CUDA-graph padding rows must never emit a compressed entry.
    const bool valid = (row_valid == nullptr) || (row_valid[t] != 0);
    const bool is_boundary = valid && (((p + 1) % ratio) == 0);
    out_pos[t] = is_boundary ? p : -1;
    out_req[t] = t;                              // one token per request
    out_rope[t] = is_boundary ? (p / ratio) * ratio : 0;
}

// The same metadata for a fire that brings MANY rows per request.
//
// Everything here is per TOKEN and identical to the decode form: whether a
// position closes a compression window is a fact about that position, and so
// is the rope base of the window it closes. One line differs, and it is the
// one the decode kernel could shortcut — `out_req[t] = t` holds only when
// each request contributes exactly one row.
//
// The request a token belongs to is the CSR row its index falls in, which is
// what `qo_indptr` states: request `r` owns `[qo_indptr[r], qo_indptr[r+1])`.
// Binary search rather than a linear scan because a prefill fire can carry
// hundreds of requests and every token pays this.
//
// `T` is unread and defaulted, for the reason the decode form states.
template <class T = i32>
__global__ void dsv4_boundary_meta_paged(
    const i32* __restrict__ positions,
    const u32* __restrict__ qo_indptr,
    i32* __restrict__ out_pos,
    i32* __restrict__ out_req,
    i32* __restrict__ out_rope,
    int n,
    int num_requests,
    int ratio,
    const u8* __restrict__ row_valid) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n) return;
    const int p = positions[t];
    const bool valid = (row_valid == nullptr) || (row_valid[t] != 0);
    const bool is_boundary = valid && (((p + 1) % ratio) == 0);
    out_pos[t] = is_boundary ? p : -1;
    // The last `r` with `qo_indptr[r] <= t`. `qo_indptr` is non-decreasing and
    // has `num_requests + 1` entries, so this lands in `[0, num_requests)` for
    // every `t < qo_indptr[num_requests]`.
    int lo = 0;
    int hi = num_requests;
    while (lo + 1 < hi) {
        const int mid = lo + (hi - lo) / 2;
        if (static_cast<int>(qo_indptr[mid]) <= t) {
            lo = mid;
        } else {
            hi = mid;
        }
    }
    out_req[t] = lo;
    out_rope[t] = is_boundary ? (p / ratio) * ratio : 0;
}

template <class T>
__global__ void dsv4_compress_gather_paged(
    const T* __restrict__ state_kv,
    const T* __restrict__ state_score,
    const float* __restrict__ ape,
    const i32* __restrict__ boundary_pos,
    const i32* __restrict__ boundary_req,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    T* __restrict__ out,
    int head_dim,
    int ratio,
    int coff,
    int page_size) {
    const int c = blockIdx.x;
    const int window = coff * ratio;
    const int width = coff * head_dim;
    const int bpos = boundary_pos[c];
    const int req = boundary_req[c];
    // `bpos < 0` marks a padding row: the CUDA-graph-safe decode path emits a
    // fixed-length boundary list (one slot per token) instead of a host-
    // compacted one, so slots for tokens that are not window boundaries have
    // to fall through as zeros rather than shrink the launch.
    if (bpos < 0) {
        T* z = out + static_cast<long long>(c) * head_dim;
        for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
            z[d] = Elem<T>::from_f32(0.f);
        }
        return;
    }

    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) {
        float max_s = neg_inf();
        for (int i = 0; i < window; ++i) {
            const int pos = bpos + i - (window - 1);
            if (pos < 0) continue;
            const int col = ((i >= ratio) ? head_dim : 0) + d;
            const long long slot =
                paged_slot(kv_page_indices, kv_page_indptr, req, pos, page_size);
            float sc = Elem<T>::to_f32(state_score[slot * width + col]);
            if (ape != nullptr) {
                sc += ape[static_cast<long long>(pos % ratio) * width + col];
            }
            max_s = fmaxf(max_s, sc);
        }
        if (!isfinite(max_s)) {
            out[static_cast<long long>(c) * head_dim + d] = Elem<T>::from_f32(0.0f);
            continue;
        }
        float sum_e = 0.0f;
        float acc = 0.0f;
        for (int i = 0; i < window; ++i) {
            const int pos = bpos + i - (window - 1);
            if (pos < 0) continue;
            const int col = ((i >= ratio) ? head_dim : 0) + d;
            const long long slot =
                paged_slot(kv_page_indices, kv_page_indptr, req, pos, page_size);
            float sc = Elem<T>::to_f32(state_score[slot * width + col]);
            if (ape != nullptr) {
                sc += ape[static_cast<long long>(pos % ratio) * width + col];
            }
            const float e = __expf(sc - max_s);
            sum_e += e;
            acc += e * Elem<T>::to_f32(state_kv[slot * width + col]);
        }
        out[static_cast<long long>(c) * head_dim + d] =
            Elem<T>::from_f32(sum_e > 0.0f ? acc / sum_e : 0.0f);
    }
}

template <class T>
__global__ void dsv4_store_comp_entries(
    const T* __restrict__ entries,
    T* __restrict__ comp_kv_pages,
    const i32* __restrict__ boundary_pos,
    const i32* __restrict__ boundary_req,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    int head_dim,
    int page_size) {
    const int c = blockIdx.x;
    if (boundary_pos[c] < 0) return;   // padding row (see gather kernel)
    const long long slot = paged_slot(kv_page_indices, kv_page_indptr,
                                      boundary_req[c], boundary_pos[c], page_size);
    const T* src = entries + static_cast<long long>(c) * head_dim;
    T* dst = comp_kv_pages + slot * head_dim;
    for (int d = threadIdx.x; d < head_dim; d += blockDim.x) dst[d] = src[d];
}

__global__ void compressed_attn_paged(
    const bf16* __restrict__ q,
    const bf16* __restrict__ comp_kv_pages,
    bf16* __restrict__ o,
    float* __restrict__ lse_out,
    const i32* __restrict__ positions,
    const u32* __restrict__ kv_page_indices,
    const u32* __restrict__ kv_page_indptr,
    const i32* __restrict__ req_of_token,
    int num_q_heads,
    int head_dim,
    int ratio,
    int page_size,
    float scale) {
    const int qi = blockIdx.x;
    const int q_head = blockIdx.y;
    const int tid = threadIdx.x;

    const int req = req_of_token[qi];
    const int qpos = positions[qi];
    // Entry c ends at absolute position (c + 1) * ratio - 1.
    const int num_visible = (qpos + 1) / ratio;

    extern __shared__ float smem[];
    float* q_smem = smem;                  // [head_dim]
    float* reduce = smem + head_dim;       // [ATTN_BLOCK]

    const bf16* q_row =
        q + (static_cast<long long>(qi) * num_q_heads + q_head) * head_dim;
    for (int d = tid; d < head_dim; d += ATTN_BLOCK) {
        q_smem[d] = bf16_to_f32(q_row[d]);
    }
    __syncthreads();

    bf16* o_row =
        o + (static_cast<long long>(qi) * num_q_heads + q_head) * head_dim;

    if (num_visible <= 0) {
        for (int d = tid; d < head_dim; d += ATTN_BLOCK) {
            o_row[d] = f32_to_bf16(0.f);
        }
        if (lse_out != nullptr && tid == 0) {
            lse_out[qi * num_q_heads + q_head] = neg_inf();
        }
        return;
    }

    float local_max = neg_inf();
    for (int c = tid; c < num_visible; c += ATTN_BLOCK) {
        const long long slot = paged_slot(kv_page_indices, kv_page_indptr, req,
                                          (c + 1) * ratio - 1, page_size);
        const bf16* k_row = comp_kv_pages + slot * head_dim;
        float dot = 0.f;
        for (int d = 0; d < head_dim; ++d) {
            dot += q_smem[d] * bf16_to_f32(k_row[d]);
        }
        local_max = fmaxf(local_max, dot * scale);
    }
    reduce[tid] = local_max;
    __syncthreads();
    for (int off = ATTN_BLOCK / 2; off > 0; off >>= 1) {
        if (tid < off) reduce[tid] = fmaxf(reduce[tid], reduce[tid + off]);
        __syncthreads();
    }
    const float row_max = reduce[0];

    const int dims_per_thread = (head_dim + ATTN_BLOCK - 1) / ATTN_BLOCK;
    float acc[8] = {};
    float local_z = 0.f;

    for (int c = 0; c < num_visible; ++c) {
        const long long slot = paged_slot(kv_page_indices, kv_page_indptr, req,
                                          (c + 1) * ratio - 1, page_size);
        const bf16* k_row = comp_kv_pages + slot * head_dim;
        float dot = 0.f;
        for (int d = tid; d < head_dim; d += ATTN_BLOCK) {
            dot += q_smem[d] * bf16_to_f32(k_row[d]);
        }
        reduce[tid] = dot;
        __syncthreads();
        for (int off = ATTN_BLOCK / 2; off > 0; off >>= 1) {
            if (tid < off) reduce[tid] += reduce[tid + off];
            __syncthreads();
        }
        const float w = expf(reduce[0] * scale - row_max);
        if (tid == 0) local_z += w;
        __syncthreads();

        for (int i = 0; i < dims_per_thread; ++i) {
            const int d = tid + i * ATTN_BLOCK;
            if (d < head_dim) acc[i] += w * bf16_to_f32(k_row[d]);
        }
    }

    __shared__ float z_shared;
    if (tid == 0) z_shared = local_z;
    __syncthreads();
    const float inv_z = z_shared > 0.f ? 1.0f / z_shared : 0.f;

    // `pool.attention_lse` states base two ONE MULTIPLY AWAY: this row
    // accumulated on `expf` against a natural-log `row_max`, so the lse it
    // has is `ln`, and the slot it writes is the one a merge against a
    // flashinfer reading folds. Rebasing here costs a `fmul` on one thread
    // per (token, head); rebasing the OTHER side would cost a launch over
    // the whole rectangle.
    if (lse_out != nullptr && tid == 0) {
        constexpr float kLog2e = 1.44269504088896340736f;
        lse_out[qi * num_q_heads + q_head] =
            z_shared > 0.f ? ((logf(z_shared) + row_max) * kLog2e) : neg_inf();
    }
    for (int i = 0; i < dims_per_thread; ++i) {
        const int d = tid + i * ATTN_BLOCK;
        if (d < head_dim) o_row[d] = f32_to_bf16(acc[i] * inv_z);
    }
}

}  // namespace pie::attn
