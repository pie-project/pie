//===-- moe_dispatch.cuh - the sparse-MoE dispatch kernels ----------------===//
//
// Twenty-two `__global__`s' worth of device text, as templates. No launcher,
// and now no second compiler either: `moe_dispatch.cu` used to include this
// file and keep every `<<<>>>`, so each kernel had exactly ONE definition
// that nvcc and NVRTC both read. That `.cu` is DELETED and all eight of its
// launchers are `driver-cuda/src/fire/moe_dispatch.rs`, so NVRTC is the only
// reader left and the two halves can no longer disagree. This is
// the largest file in the family and the one the split matters most for —
// twenty-two kernels copied instead of moved is twenty-two chances for the
// archive and the JIT to disagree, each of them right for whichever half its
// test exercises.
//
// Eight of the twenty-two are fired by name from that Rust, and every one of
// those names carries a `_dev` suffix in `families::moe` — see the mapping
// block there. `execution`'s `a_walk_is_only_a_walk` is why: the ABI symbol a
// model trace records is walked, and a walked symbol may not be unit-hosted.
//
// # What this file is for
//
// A sparse MoE decode step is a permutation problem wrapped around two
// GEMMs. Routes are `(token, expert)` pairs; the weights are per-expert, so
// the rows have to be bucketed by expert before a batched GEMM can touch
// them, and unbucketed afterwards. Everything here is one of:
//
//   * a SORT — `moe_align_decode`, `moe_bucket_exact`;
//   * a PERMUTE — `gather_moe_aligned_inputs`, `reorder_moe_aligned_output`;
//   * a POINTER BUILD — the batched-GEMM argument arrays, built on device so
//     the host never sees the routing;
//   * a GEMV — the decode fast path, where M is 1 and a GEMM buys nothing;
//   * a COMBINE — the weighted sum of a token's `top_k` expert outputs.
//
// # The wmma call site, and why this family was blocked
//
// `moe_decode_wmma_*` is one of the two `wmma` users in the tree.
// `pie_mma.cuh`'s header names both. NVRTC 13.0 refused `mma.h` outright, so
// until that shim existed and was proved bit-identical against
// `nvcuda::wmma` on an L40S, no unit containing this file could compile at
// all. The shape asked for here — 16×16×16, bf16×bf16→f32, A `row_major`,
// B `col_major`, store `mem_row_major` — is exactly the one it implements.
//
// The include is conditional because the two compilers get two different
// implementations on purpose: NVRTC gets the shim beside this file, nvcc
// keeps the toolkit's `<mma.h>`. See `moe_grouped_gemm.cuh`, which says the
// same thing at more length.
//
// # Vector and scalar are two kernels because the CHOICE is the host's
//
// Four pairs here (`token_batched_weighted_sum`, its `_add`, the aligned
// gather, the aligned reorder) exist twice: once a `uint4` per thread, once
// a scalar. Which one runs depends on `hidden % 8` and on the runtime
// ALIGNMENT of three pointers — facts about an allocation, not about a
// shape. A `Source` states where a value comes from, never a predicate over
// one, so only the SCALAR forms are rowed: they are always correct, and
// firing a vectorised form on an odd hidden size puts every second row on a
// 2-byte boundary and faults. The launcher keeps the choice.
//
// # Six rows, and the seventeen kernels that have none
//
// `scalar_weighted_add` states `LaunchRule::Elementwise` — `ceil(n / 256)`
// blocks of 256 over a flat rectangle, which is the launcher exactly.
// `add_moe_route_bias` states `LaunchRule::Rms` — one block per row, 256
// threads, the block striding the row — because its output IS the
// route-major staging, so the grid's routes and the rectangle's rows are the
// same number. `token_batched_weighted_sum`, its `_add`,
// `token_batched_weighted_sum_aligned` and `gather_moe_aligned_inputs` state
// `LaunchRule::ElementwiseRows`.
//
// Those four moved an axis to get there. The ahead-of-time launchers fired
// `dim3(ceil(width / 256), rows)` — the row on `y` — and `ElementwiseRows`
// is the same rectangle with the row on `x`. The two index lines in each
// kernel moved with the rule and the `dim3` in `moe_dispatch.cu` moved with
// them, so the ahead-of-time path launches the transposed grid too and every
// thread computes the element it computed before; the guard is `h >= hidden`
// either way. `mlp::gpt_oss_glu_strided_bf16` made the same move for the
// same reason.
//
// `gather_moe_aligned_inputs` is rowed and `reorder_moe_aligned_output` is
// NOT, which looks arbitrary and is not: `ElementwiseRows` opens its grid
// over the OUTPUT's rows. The gather WRITES the aligned rectangle, so its
// grid rows are its output's rows. The reorder READS it and writes route
// rows, so its grid rows are its INPUT's — the padded block-major count,
// which is larger. A row there would launch `routes` blocks over an
// `aligned_rows`-deep permutation and drop the tail.
//
// The other seventeen are migrated as TEXT and unmigrated as ROWS, in five
// groups:
//
//   * **A grid over routes or padded blocks.**
//     `build_moe_ptrs_decode_batched` (`ceil(routes / 256)`),
//     `build_moe_ptrs_aligned` (one thread per padded block),
//     `moe_decode_gemv_*` and `moe_decode_wmma_*` (a 2D grid of output tiles
//     × routes). `Dims` carries the fire's rectangle; the route count is
//     `tokens · top_k` and the padded block count is a host bound, and
//     neither is a rectangle's extent.
//   * **A grid of ONE.** `build_dual_gemm_ptrs` is `<<<1, 1>>>` and
//     `build_moe_ptrs_decode` is `<<<1, top_k>>>`. Every rule opens its grid
//     over rows.
//   * **Dynamic shared memory sized from an operand VALUE.**
//     `moe_align_decode` and `moe_bucket_exact` are single-block counting
//     sorts whose shared slab is `(3·num_experts + …)·4` bytes. `Launch`
//     does carry an `smem` field — `Rms` sets it to 32 — but no rule
//     computes it from anything, and `num_experts` is a parameter rather
//     than an extent in the first place.
//   * **A 2D or 3D block.** `moe_decode_gemv_*` wants `dim3(32, kWarps)` and
//     `transpose_expert_scales` wants `dim3(32, 8)` on a 3D grid. Every
//     ported rule produces `[BLOCK, 1, 1]`.
//   * **A single-row reduction.** `batched_weighted_sum` collapses `batch`
//     rows into one and launches `ceil(hidden / 256)`. `Elementwise` would
//     multiply that by the fire's rows and rely on the `h >= hidden` guard
//     to throw the surplus away — which is the shape `norm/dsv4_hc` refused
//     for `hc_post`, and refusing it once and not twice is not a standard.
//
// **`moe_align_decode` and `moe_bucket_exact` are also the two rows a rule
// could never state for a second reason**: they are launched `<<<1, 1024>>>`
// whatever the routing, because the counting sort's exclusive scan is
// block-wide. Giving either a row axis — which is what `RouterSort` reads
// like from its name — launches N copies of the same sort, each clearing the
// counters the others are reading.
//
//===----------------------------------------------------------------------===//
#pragma once

#include "pie_device.cuh"

#ifdef __CUDACC_RTC__
// NVRTC: the shim, which resolves and typedefs `__nv_bfloat16` to the
// prelude's `device::bf16`.
#include "pie_mma.cuh"
#else
// nvcc: the vendor's headers, which are on the ahead-of-time include path.
#include <cuda_bf16.h>
#include <mma.h>
#endif

namespace pie_cuda_driver::kernels::moe::device {

// The scalar layer is the PRELUDE's, named here so `device::i32` keeps its
// meaning inside `kernels::moe` once this nested namespace shadows it.
using ::pie_cuda_driver::kernels::device::Elem;
using ::pie_cuda_driver::kernels::device::bf16;
using ::pie_cuda_driver::kernels::device::f16;
using ::pie_cuda_driver::kernels::device::i32;
using ::pie_cuda_driver::kernels::device::is_same;
using ::pie_cuda_driver::kernels::device::u8;

namespace wmma = ::nvcuda::wmma;

/// Threads per block for every kernel here whose launcher fixed one.
///
/// `[[maybe_unused]]` — and the same on the two constants below — because a
/// NVRTC unit instantiates only the templates its rows name, and this file
/// rows six of its twenty-five entry points. A constant read solely by an
/// un-instantiated template is "declared but never referenced" to the front
/// end. Saying so per symbol beats a `--diag-suppress` that would also hide
/// the warning about a constant nothing reads at all.
[[maybe_unused]] constexpr int kDispatchBlock = 256;

/// bf16 per `uint4`, and the width the vectorised forms below step by.
///
/// One bf16 per thread makes every warp issue a 64-byte access, half a cache
/// line. Eight turns that into a full 512-byte contiguous transaction.
[[maybe_unused]] constexpr int kMoeVecWidth = 8;

/// The most routes a token can take, and the bound the `#pragma unroll`ed
/// combine loops are written against.
///
/// The loops `break` at `top_k`, so this is an unroll bound and not a limit:
/// a `top_k` above it would be computed correctly by a loop the compiler did
/// not fully unroll. 16 is twice Qwen3.6's 8.
constexpr int kMaxTopK = 16;

/// `out[i] += weight · src[i]`, flat.
///
/// The decode fast path: with one token every per-expert contribution lands
/// on the same destination row, so an indexed scatter would degenerate to
/// an fma over a row — which saves an expert-index D2H copy and the gather.
template <class T>
__global__ void scalar_weighted_add(
    T* __restrict__ out,
    const T* __restrict__ src,
    float weight, int n)
{
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float ov = Elem<T>::to_f32(out[i]);
    const float sv = Elem<T>::to_f32(src[i]);
    out[i] = Elem<T>::from_f32(ov + weight * sv);
}

/// Two rows of a batched GEMM's pointer arrays, written by one thread.
///
/// `<<<1, 1>>>`, and it has to be on the device: the pointers are into
/// device allocations and the array they go in is read by cuBLAS on the same
/// stream, so writing them from the host would need a synchronisation that
/// this exists to avoid.
template <class T>
__global__ void build_dual_gemm_ptrs(
    const T* act,
    const T* w0,
    const T* w1,
    T* out0,
    T* out1,
    const T** act_ptrs,
    const T** w_ptrs,
    T** out_ptrs)
{
    act_ptrs[0] = act;
    act_ptrs[1] = act;
    w_ptrs[0] = w0;
    w_ptrs[1] = w1;
    out_ptrs[0] = out0;
    out_ptrs[1] = out1;
}

/// `out[h] = Σ_k weights[k] · src[k, h]` for one token's `batch` routes.
///
/// The N=1 combine: `batch` is `top_k` and small, `hidden` is the model's.
/// One launch replaces `top_k` scalar-weighted adds.
template <class T>
__global__ void batched_weighted_sum(
    T* __restrict__ out,
    const T* __restrict__ src,      // [batch, hidden]
    const float* __restrict__ weights,  // [batch]
    int batch, int hidden)
{
    const int h = blockIdx.x * blockDim.x + threadIdx.x;
    if (h >= hidden) return;
    float acc = 0.f;
    #pragma unroll
    for (int k = 0; k < kMaxTopK; ++k) {  // unroll up to top_k=16; loop bounded
        if (k >= batch) break;
        const float v = Elem<T>::to_f32(src[(long long)k * hidden + h]);
        acc += weights[k] * v;
    }
    out[h] = Elem<T>::from_f32(acc);
}

/// The same combine for a whole rectangle of tokens: `out[n, h] = Σ_k
/// weights[n, k] · src[n, k, h]`.
///
/// One grid axis per dimension, the TOKEN ON `x` and the channel tile on
/// `y` — which is `LaunchRule::ElementwiseRows` exactly. The ahead-of-time
/// launcher fired the transpose of this; the two index lines moved with the
/// rule and its `dim3` moved with them, and the coverage is identical
/// because the guard is `h >= hidden` either way.
template <class T>
__global__ void token_batched_weighted_sum(
    T* __restrict__ out,
    const T* __restrict__ src,          // [num_tokens, top_k, hidden]
    const float* __restrict__ weights,  // [num_tokens, top_k]
    int top_k, int hidden)
{
    const int n = blockIdx.x;
    const int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (h >= hidden) return;
    const long long base = static_cast<long long>(n) * top_k;
    float acc = 0.f;
    #pragma unroll
    for (int k = 0; k < kMaxTopK; ++k) {
        if (k >= top_k) break;
        const long long r = base + k;
        const float v = Elem<T>::to_f32(src[r * hidden + h]);
        acc += weights[r] * v;
    }
    out[static_cast<long long>(n) * hidden + h] = Elem<T>::from_f32(acc);
}

/// `token_batched_weighted_sum`, accumulating onto what is already there.
///
/// A separate kernel and not a flag: the shared-expert leg adds its result
/// onto the routed one, and a read-modify-write that a branch skips still
/// costs the read.
template <class T>
__global__ void token_batched_weighted_sum_add(
    T* __restrict__ out,
    const T* __restrict__ src,          // [num_tokens, top_k, hidden]
    const float* __restrict__ weights,  // [num_tokens, top_k]
    int top_k, int hidden)
{
    const int n = blockIdx.x;
    const int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (h >= hidden) return;
    const long long base = static_cast<long long>(n) * top_k;
    float acc = 0.f;
    #pragma unroll
    for (int k = 0; k < kMaxTopK; ++k) {
        if (k >= top_k) break;
        const long long r = base + k;
        const float v = Elem<T>::to_f32(src[r * hidden + h]);
        acc += weights[r] * v;
    }
    const long long out_idx = static_cast<long long>(n) * hidden + h;
    out[out_idx] = Elem<T>::from_f32(Elem<T>::to_f32(out[out_idx]) + acc);
}

/// `token_batched_weighted_sum` at eight elements per thread.
///
/// These two stream `top_k · hidden` elements per token, so the access width
/// is the whole cost. `hidden_vec` is `hidden / kMoeVecWidth`, and the
/// launcher fires this form only when `hidden` divides and both pointers are
/// 16-byte aligned — a fact about an allocation, which is why the CHOICE
/// cannot be a rule.
template <class T>
__global__ void token_batched_weighted_sum_vec(
    T* __restrict__ out,
    const T* __restrict__ src,
    const float* __restrict__ weights,
    int top_k, int hidden_vec)
{
    static_assert(sizeof(T) == 2, "kMoeVecWidth elements are one uint4");
    const int n = blockIdx.x;
    const int hv = blockIdx.y * blockDim.x + threadIdx.x;
    if (hv >= hidden_vec) return;
    const long long base = static_cast<long long>(n) * top_k;
    float acc[kMoeVecWidth];
#pragma unroll
    for (int j = 0; j < kMoeVecWidth; ++j) acc[j] = 0.f;
    const uint4* sv = reinterpret_cast<const uint4*>(src);
    for (int k = 0; k < top_k; ++k) {
        const long long r = base + k;
        const uint4 v = sv[r * hidden_vec + hv];
        const auto* vh = reinterpret_cast<const T*>(&v);
        const float w = weights[r];
#pragma unroll
        for (int j = 0; j < kMoeVecWidth; ++j) acc[j] += w * Elem<T>::to_f32(vh[j]);
    }
    uint4 o;
    auto* oh = reinterpret_cast<T*>(&o);
#pragma unroll
    for (int j = 0; j < kMoeVecWidth; ++j) oh[j] = Elem<T>::from_f32(acc[j]);
    reinterpret_cast<uint4*>(out)[
        static_cast<long long>(n) * hidden_vec + hv] = o;
}

/// `token_batched_weighted_sum_add` at eight elements per thread.
template <class T>
__global__ void token_batched_weighted_sum_add_vec(
    T* __restrict__ out,
    const T* __restrict__ src,
    const float* __restrict__ weights,
    int top_k, int hidden_vec)
{
    static_assert(sizeof(T) == 2, "kMoeVecWidth elements are one uint4");
    const int n = blockIdx.x;
    const int hv = blockIdx.y * blockDim.x + threadIdx.x;
    if (hv >= hidden_vec) return;
    const long long base = static_cast<long long>(n) * top_k;
    float acc[kMoeVecWidth];
#pragma unroll
    for (int j = 0; j < kMoeVecWidth; ++j) acc[j] = 0.f;
    const uint4* sv = reinterpret_cast<const uint4*>(src);
    for (int k = 0; k < top_k; ++k) {
        const long long r = base + k;
        const uint4 v = sv[r * hidden_vec + hv];
        const auto* vh = reinterpret_cast<const T*>(&v);
        const float w = weights[r];
#pragma unroll
        for (int j = 0; j < kMoeVecWidth; ++j) acc[j] += w * Elem<T>::to_f32(vh[j]);
    }
    const long long oi = static_cast<long long>(n) * hidden_vec + hv;
    uint4 o = reinterpret_cast<uint4*>(out)[oi];
    auto* oh = reinterpret_cast<T*>(&o);
#pragma unroll
    for (int j = 0; j < kMoeVecWidth; ++j) {
        oh[j] = Elem<T>::from_f32(Elem<T>::to_f32(oh[j]) + acc[j]);
    }
    reinterpret_cast<uint4*>(out)[oi] = o;
}

/// The combine for the ALIGNED path, where a route's output row is wherever
/// the counting sort put it.
///
/// `route_to_aligned_row` is that sort's inverse map. Reading it here is what
/// lets the permutation stay undone until the very end — the alternative is a
/// separate unpermute pass over the same bytes.
template <class T>
__global__ void token_batched_weighted_sum_aligned(
    T* __restrict__ out,
    const T* __restrict__ aligned_out,
    const float* __restrict__ weights,
    const i32* __restrict__ route_to_aligned_row,
    int top_k,
    int hidden)
{
    const int n = blockIdx.x;
    const int h = blockIdx.y * blockDim.x + threadIdx.x;
    if (h >= hidden) return;
    const long long base = static_cast<long long>(n) * top_k;
    float acc = 0.f;
#pragma unroll
    for (int k = 0; k < kMaxTopK; ++k) {
        if (k >= top_k) break;
        const long long route = base + k;
        const int row = route_to_aligned_row[route];
        const float v = Elem<T>::to_f32(
            aligned_out[static_cast<long long>(row) * hidden + h]);
        acc += weights[route] * v;
    }
    out[static_cast<long long>(n) * hidden + h] = Elem<T>::from_f32(acc);
}

/// One block, `top_k` threads: each thread owns one of the active experts and
/// emits its row of the cuBLAS batched-GEMM pointer arrays.
template <class T>
__global__ void build_moe_ptrs_decode(
    const i32* topk_idx,
    const float* topk_w,
    const T* gate_up_base,
    const T* down_base,
    const T* norm_x,
    T* expert_gate_up,
    T* expert_act,
    T* expert_out,
    const T** a_gu_ptrs,
    const T** b_gu_ptrs,
    T**       c_gu_ptrs,
    const T** a_dn_ptrs,
    const T** b_dn_ptrs,
    T**       c_dn_ptrs,
    float*    weights_out,
    int top_k, int H, int I_moe)
{
    const int k = threadIdx.x;
    if (k >= top_k) return;
    const long long stride_gu = 2LL * I_moe * H;
    const long long stride_dn = (long long)H * I_moe;
    const int e = topk_idx[k];

    a_gu_ptrs[k] = gate_up_base + e * stride_gu;
    b_gu_ptrs[k] = norm_x;
    c_gu_ptrs[k] = expert_gate_up + (long long)k * 2 * I_moe;

    a_dn_ptrs[k] = down_base + e * stride_dn;
    b_dn_ptrs[k] = expert_act + (long long)k * I_moe;
    c_dn_ptrs[k] = expert_out + (long long)k * H;

    weights_out[k] = topk_w[k];
}

/// The same pointer build for a rectangle of tokens: one thread per route.
///
/// Flat over `num_tokens · top_k`, which is `LaunchRule::Elementwise` over a
/// `[tokens, top_k]` rectangle exactly — the one pointer builder a rule
/// states.
template <class T>
__global__ void build_moe_ptrs_decode_batched(
    const i32* topk_idx,
    const float* topk_w,
    const T* gate_up_base,
    const T* down_base,
    const T* norm_x,
    T* expert_gate_up,
    T* expert_act,
    T* expert_out,
    const T** a_gu_ptrs,
    const T** b_gu_ptrs,
    T**       c_gu_ptrs,
    const T** a_dn_ptrs,
    const T** b_dn_ptrs,
    T**       c_dn_ptrs,
    float*    weights_out,
    int num_tokens, int top_k, int H, int I_moe)
{
    const int r = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = num_tokens * top_k;
    if (r >= total) return;
    const long long stride_gu = 2LL * I_moe * H;
    const long long stride_dn = static_cast<long long>(H) * I_moe;
    const int token = r / top_k;
    const int e = topk_idx[r];

    a_gu_ptrs[r] = gate_up_base + static_cast<long long>(e) * stride_gu;
    b_gu_ptrs[r] = norm_x + static_cast<long long>(token) * H;
    c_gu_ptrs[r] = expert_gate_up + static_cast<long long>(r) * 2 * I_moe;

    a_dn_ptrs[r] = down_base + static_cast<long long>(e) * stride_dn;
    b_dn_ptrs[r] = expert_act + static_cast<long long>(r) * I_moe;
    c_dn_ptrs[r] = expert_out + static_cast<long long>(r) * H;

    weights_out[r] = topk_w[r];
}

/// The decode GEMM on tensor cores, one 64-wide output tile per block.
///
/// `ActByToken` selects the input row: the gate/up projection reads the
/// TOKEN's hidden state, shared by that token's `top_k` routes; the down
/// projection reads the ROUTE's own activation.
///
/// A `__device__` body because a row supplies one template argument and this
/// needs two — the two entry points below are what the rows name.
///
/// The `__CUDA_ARCH__` guard is the vendor's requirement, not a preference:
/// bf16 `wmma` fragments start at sm_80. It stays because an architecture
/// below that must still COMPILE this translation unit; what it must not do
/// is fire the kernel, and the launchers are guarded by the same fact.
template <class T, bool ActByToken>
__device__ __forceinline__ void moe_decode_wmma_body(
    const i32* __restrict__ topk_idx,
    const T* __restrict__ act,
    const T* __restrict__ weight_base,
    T* __restrict__ out,
    int top_k,
    int K,
    int N,
    long long expert_stride)
{
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
    static_assert(is_same<T, bf16>::value,
                  "pie_mma.cuh implements bf16 fragments only -- see its "
                  "static_assert, and do not extend it without a parity run");
    constexpr int N_TILE = 64;
    const int n0 = blockIdx.x * N_TILE;
    const int route = blockIdx.y;
    const int expert = topk_idx[route];
    if (expert < 0 || n0 >= N) return;

    extern __shared__ __align__(16) unsigned char wmma_smem[];
    auto* a_tile = reinterpret_cast<T*>(wmma_smem);
    auto* c_tile = reinterpret_cast<float*>(a_tile + 16 * 16);
    const int warp_id = threadIdx.x / 32;
    const int lane = threadIdx.x & 31;
    const int n_warp = n0 + warp_id * 16;

    const int token = route / top_k;
    const T* act_row = act + static_cast<long long>(ActByToken ? token : route) * K;
    const T* weight = weight_base + static_cast<long long>(expert) * expert_stride;

    wmma::fragment<wmma::matrix_a, 16, 16, 16, __nv_bfloat16, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, 16, 16, 16, __nv_bfloat16, wmma::col_major> b_frag;
    wmma::fragment<wmma::accumulator, 16, 16, 16, float> acc_frag;
    wmma::fill_fragment(acc_frag, 0.0f);

    for (int k0 = 0; k0 < K; k0 += 16) {
        for (int i = threadIdx.x; i < 16 * 16; i += blockDim.x) {
            a_tile[i] = Elem<T>::from_f32(0.0f);
        }
        if (threadIdx.x < 16) {
            a_tile[threadIdx.x] = act_row[k0 + threadIdx.x];
        }
        __syncthreads();

        wmma::load_matrix_sync(
            a_frag, reinterpret_cast<const __nv_bfloat16*>(a_tile), 16);
        wmma::load_matrix_sync(
            b_frag,
            reinterpret_cast<const __nv_bfloat16*>(
                weight + static_cast<long long>(n_warp) * K + k0),
            K);
        wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
        __syncthreads();
    }

    wmma::store_matrix_sync(
        c_tile + warp_id * 16 * 16, acc_frag, 16, wmma::mem_row_major);
    __syncthreads();

    if (lane < 16) {
        const long long out_base = static_cast<long long>(route) * N + n0;
        out[out_base + warp_id * 16 + lane] =
            Elem<T>::from_f32(c_tile[warp_id * 16 * 16 + lane]);
    }
#endif
}

/// The gate/up form: the activation row is the token's.
template <class T>
__global__ void moe_decode_wmma_by_token(
    const i32* __restrict__ topk_idx,
    const T* __restrict__ act,
    const T* __restrict__ weight_base,
    T* __restrict__ out,
    int top_k, int K, int N, long long expert_stride)
{
    moe_decode_wmma_body<T, true>(topk_idx, act, weight_base, out,
                                  top_k, K, N, expert_stride);
}

/// The down form: the activation row is the route's own.
template <class T>
__global__ void moe_decode_wmma_by_route(
    const i32* __restrict__ topk_idx,
    const T* __restrict__ act,
    const T* __restrict__ weight_base,
    T* __restrict__ out,
    int top_k, int K, int N, long long expert_stride)
{
    moe_decode_wmma_body<T, false>(topk_idx, act, weight_base, out,
                                   top_k, K, N, expert_stride);
}

/// Bandwidth-optimal decode GEMV for the sparse MoE hot path.
///
/// At decode the routed GEMMs are M=1: 8 experts x [1, H] x [H, N]. There
/// is no reuse of the weight to exploit, so the only thing that matters is
/// reading it at full HBM bandwidth. Tensor cores buy nothing at M=1 (the
/// WMMA path measured 45% SLOWER end-to-end), and `cublasGemmBatchedEx`
/// leaves bandwidth on the table because its tiling is chosen for shapes
/// with an M to fill. One warp per output row, float4 loads, fp32
/// accumulate, one shuffle reduction: 1624 GB/s vs cuBLAS's 1282 on A100
/// for Qwen3.6's gate/up shape (routes=8, N=1024, K=2048).
///
/// `kUnroll` hoists several loads above the math that consumes them, as
/// gemv.cuh's row-per-warp kernel does. MEASURED AND NOT WORTH IT HERE, which
/// is why the entry points below fix it at 1 (identical code to before it was
/// templated).
///
/// The reasoning that motivated it was sound and still wrong: with no unroll
/// each lane keeps one load in flight, the condition gemv.cuh records as
/// having cost it ~963 GB/s. But this kernel does not have that problem -- it already
/// runs near roofline. Swept at gemma-4-26B-A4B's real shapes (from its config
/// and cross-checked against its decode trace, 60 calls/step at 11.50 us):
///
///   gate/up N=1408 K=2816 (63 MB): w4,u1 10.91us 5.81TB/s  <- shipping
///                                  w4,u4 10.87us 5.84TB/s  (noise)
///   down    N=2816 K=704  (32 MB): w4,u1 10.29us 3.08TB/s  <- shipping
///                                  w8,u2  9.40us 3.38TB/s  (-8.6%)
///
/// 5.81 TB/s against a ~6.7 TB/s machine roof leaves nothing to win on gate/up.
/// The `down` shape is latency-bound (K=704 is 2.75 float4 per lane) and does
/// improve, but it is 30 calls of a 7599 us step: -0.35% end to end, well under
/// the run-to-run spread. The sweep entry point is kept so the next person can
/// re-check rather than re-derive.
template <class T, bool ActByToken, int kWarps, int kUnroll>
__device__ __forceinline__ void moe_decode_gemv_body(
    const i32* __restrict__ topk_idx,
    const T* __restrict__ act,
    const T* __restrict__ weight_base,
    T* __restrict__ out,
    int top_k, int K, int N, long long expert_stride)
{
    const int route = blockIdx.y;
    const int row = blockIdx.x * kWarps + threadIdx.y;
    if (row >= N) return;
    const int expert = topk_idx[route];
    const T* w = weight_base + expert * expert_stride + (long long)row * K;
    const T* x = act + (long long)(ActByToken ? route / top_k : route) * K;

    const int lane = threadIdx.x;
    float acc = 0.f;
    const int vec = K / kMoeVecWidth;
    const float4* w4 = reinterpret_cast<const float4*>(w);
    const float4* x4 = reinterpret_cast<const float4*>(x);
    int i = lane;
    for (; i + 32 * (kUnroll - 1) < vec; i += 32 * kUnroll) {
        float4 wv[kUnroll];
        float4 xv[kUnroll];
        #pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            wv[u] = w4[i + 32 * u];
            xv[u] = x4[i + 32 * u];
        }
        #pragma unroll
        for (int u = 0; u < kUnroll; ++u) {
            const T* wb = reinterpret_cast<const T*>(&wv[u]);
            const T* xb = reinterpret_cast<const T*>(&xv[u]);
            #pragma unroll
            for (int j = 0; j < kMoeVecWidth; ++j) {
                acc += Elem<T>::to_f32(wb[j]) * Elem<T>::to_f32(xb[j]);
            }
        }
    }
    for (; i < vec; i += 32) {
        float4 wv = w4[i];
        float4 xv = x4[i];
        const T* wb = reinterpret_cast<const T*>(&wv);
        const T* xb = reinterpret_cast<const T*>(&xv);
        #pragma unroll
        for (int j = 0; j < kMoeVecWidth; ++j) {
            acc += Elem<T>::to_f32(wb[j]) * Elem<T>::to_f32(xb[j]);
        }
    }
    #pragma unroll
    for (int off = 16; off > 0; off >>= 1) {
        acc += __shfl_down_sync(0xffffffffu, acc, off);
    }
    if (lane == 0) out[(long long)route * N + row] = Elem<T>::from_f32(acc);
}

/// The sweep's entry point: warps per block and unroll depth chosen
/// explicitly. Host-only, and deliberately unrowed — a microbenchmark is not
/// a statement any trace makes.
template <class T, bool ActByToken, int kWarps, int kUnroll = 1>
__global__ void moe_decode_gemv(
    const i32* __restrict__ topk_idx,
    const T* __restrict__ act,
    const T* __restrict__ weight_base,
    T* __restrict__ out,
    int top_k, int K, int N, long long expert_stride)
{
    moe_decode_gemv_body<T, ActByToken, kWarps, kUnroll>(
        topk_idx, act, weight_base, out, top_k, K, N, expert_stride);
}

/// Warps per block for the two shipping GEMV forms — the `w4` of the sweep
/// table above.
constexpr int kGemvWarps = 4;

/// The gate/up form at the shipping geometry.
template <class T>
__global__ void moe_decode_gemv_by_token(
    const i32* __restrict__ topk_idx,
    const T* __restrict__ act,
    const T* __restrict__ weight_base,
    T* __restrict__ out,
    int top_k, int K, int N, long long expert_stride)
{
    moe_decode_gemv_body<T, true, kGemvWarps, 1>(
        topk_idx, act, weight_base, out, top_k, K, N, expert_stride);
}

/// The down form at the shipping geometry.
template <class T>
__global__ void moe_decode_gemv_by_route(
    const i32* __restrict__ topk_idx,
    const T* __restrict__ act,
    const T* __restrict__ weight_base,
    T* __restrict__ out,
    int top_k, int K, int N, long long expert_stride)
{
    moe_decode_gemv_body<T, false, kGemvWarps, 1>(
        topk_idx, act, weight_base, out, top_k, K, N, expert_stride);
}

/// The counting sort that buckets routes by expert and pads each bucket to a
/// block boundary, so one batched GEMM covers every expert.
///
/// ONE BLOCK, whatever the routing: the exclusive scan over per-expert
/// padded counts is block-wide, and the counters are in shared memory. A row
/// axis would launch N copies of this sort, each clearing what the others are
/// reading — which is the failure a rule named for a "router sort" invites.
///
/// The shared slab is dynamic and sized from `num_experts`, a PARAMETER: it
/// holds `counts`, `offsets` (+1), `fill`, 32 warp partials and one running
/// base. No launch rule can compute that, which is the second reason this row
/// states none.
///
/// `T` is the index type every buffer here is, and the row states it: this
/// kernel has no element type, and a signature format that supplies exactly
/// one template argument still has to be handed something true.
template <class T>
__global__ void moe_align_decode(
    const T* __restrict__ topk_idx,
    T* __restrict__ sorted_route_ids,
    T* __restrict__ expert_ids,
    T* __restrict__ route_to_aligned_row,
    int num_routes,
    int num_experts,
    int block_size,
    int max_blocks,
    T* __restrict__ num_tokens_past_padded)
{
    static_assert(is_same<T, i32>::value, "the routing indices are i32");
    extern __shared__ i32 align_smem[];
    i32* counts = align_smem;
    i32* offsets = counts + num_experts;
    i32* fill = offsets + num_experts + 1;
    i32* warp_totals = fill + num_experts;
    i32* block_base = warp_totals + 32;

    const int aligned_rows = max_blocks * block_size;
    if (threadIdx.x == 0) *block_base = 0;
    for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
        counts[i] = 0;
        fill[i] = 0;
    }
    for (int i = threadIdx.x; i < aligned_rows; i += blockDim.x) {
        sorted_route_ids[i] = num_routes;
    }
    for (int i = threadIdx.x; i < max_blocks; i += blockDim.x) {
        expert_ids[i] = -1;
    }
    __syncthreads();

    for (int r = threadIdx.x; r < num_routes; r += blockDim.x) {
        const int e = topk_idx[r];
        if (0 <= e && e < num_experts) {
            atomicAdd(counts + e, 1);
        }
    }
    __syncthreads();

    // Block-wide exclusive scan of the per-expert padded block counts.
    // Running this serially on thread 0 cost ~15 us per layer -- 256
    // dependent shared-memory reads and integer divides, x40 layers.
    {
        const int lane = threadIdx.x & 31;
        const int warp = static_cast<int>(threadIdx.x) >> 5;
        const int num_warps = static_cast<int>(blockDim.x) >> 5;
        for (int base = 0; base < num_experts; base += static_cast<int>(blockDim.x)) {
            const int e = base + static_cast<int>(threadIdx.x);
            int padded = 0;
            if (e < num_experts) {
                const int c = counts[e];
                padded = ((c + block_size - 1) / block_size) * block_size;
            }
            int value = padded;
            for (int off = 1; off < 32; off <<= 1) {
                const int n = __shfl_up_sync(0xffffffffu, value, off);
                if (lane >= off) value += n;
            }
            if (lane == 31) warp_totals[warp] = value;
            __syncthreads();
            if (warp == 0) {
                int t = (lane < num_warps) ? warp_totals[lane] : 0;
                for (int off = 1; off < 32; off <<= 1) {
                    const int n = __shfl_up_sync(0xffffffffu, t, off);
                    if (lane >= off) t += n;
                }
                if (lane < num_warps) warp_totals[lane] = t;
            }
            __syncthreads();
            const int warp_prefix = (warp == 0) ? 0 : warp_totals[warp - 1];
            if (e < num_experts) {
                // Exclusive within this chunk, plus everything before it.
                offsets[e] = *block_base + warp_prefix + value - padded;
            }
            __syncthreads();
            if (threadIdx.x == 0) {
                *block_base += warp_totals[num_warps - 1];
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            offsets[num_experts] = *block_base;
#if defined(PIE_MOE_ALIGN_REPORT)
            // How many blocks the routing actually needs, against the
            // worst-case `max_blocks` the batched GEMM always launches.
            printf("[moe-align] used=%d max=%d routes=%d experts=%d\n",
                   *block_base / block_size, max_blocks, num_routes,
                   num_experts);
#endif
        }
    }
    __syncthreads();
    for (int e = threadIdx.x; e < num_experts; e += blockDim.x) {
        const int begin = offsets[e];
        const int end = offsets[e + 1];
        for (int row = begin; row < end; row += block_size) {
            const int b = row / block_size;
            if (b < max_blocks) expert_ids[b] = e;
        }
    }
    __syncthreads();

    for (int r = threadIdx.x; r < num_routes; r += blockDim.x) {
        const int e = topk_idx[r];
        if (0 <= e && e < num_experts) {
            const int pos = atomicAdd(fill + e, 1);
            const int out = offsets[e] + pos;
            if (out < aligned_rows) {
                sorted_route_ids[out] = r;
                if (route_to_aligned_row != nullptr) {
                    route_to_aligned_row[r] = out;
                }
            }
        }
    }
    // Marlin/Triton-style grouped GEMMs iterate M-blocks up to this and index
    // `expert_ids` inside it; the entries past it are the -1 padding above, so
    // publishing the true total is what keeps them from being read.
    __syncthreads();
    if (num_tokens_past_padded != nullptr && threadIdx.x == 0) {
        *num_tokens_past_padded = *block_base;
    }
}

/// The UNPADDED counting sort: exact per-expert counts, for the host to build
/// cuBLAS grouped shapes from.
///
/// One block for the same reason as `moe_align_decode`, and a dynamic shared
/// slab sized from `num_experts` for the same reason. The scan here is
/// serial on thread 0 because there is no padding division in it — the
/// measurement that justified the parallel scan above was of the padded form.
template <class T>
__global__ void moe_bucket_exact(
    const T* __restrict__ topk_idx,
    T* __restrict__ sorted_route_ids,
    T* __restrict__ route_to_sorted_row,
    T* __restrict__ counts_out,
    int num_routes,
    int num_experts)
{
    static_assert(is_same<T, i32>::value, "the routing indices are i32");
    extern __shared__ i32 bucket_smem[];
    i32* counts = bucket_smem;
    i32* offsets = counts + num_experts;
    i32* fill = offsets + num_experts + 1;

    for (int i = threadIdx.x; i < num_experts; i += blockDim.x) {
        counts[i] = 0;
        fill[i] = 0;
    }
    __syncthreads();

    for (int r = threadIdx.x; r < num_routes; r += blockDim.x) {
        const int e = topk_idx[r];
        if (0 <= e && e < num_experts) {
            atomicAdd(counts + e, 1);
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        int running = 0;
        for (int e = 0; e < num_experts; ++e) {
            offsets[e] = running;
            const int c = counts[e];
            counts_out[e] = c;
            running += c;
        }
        offsets[num_experts] = running;
    }
    __syncthreads();

    for (int r = threadIdx.x; r < num_routes; r += blockDim.x) {
        const int e = topk_idx[r];
        if (0 <= e && e < num_experts) {
            const int pos = atomicAdd(fill + e, 1);
            const int out = offsets[e] + pos;
            sorted_route_ids[out] = r;
            route_to_sorted_row[r] = out;
        }
    }
}

/// Stage each aligned row's input, eight elements per thread.
///
/// A padding row reads nothing and writes zeros — `token` stays -1 — which is
/// what makes the padded batched GEMM's extra entries harmless rather than
/// garbage. `shared_row_begin >= 0` marks the tail rows that belong to the
/// shared expert, which reads the token's own hidden state rather than a
/// routed one.
template <class T>
__global__ void gather_moe_aligned_inputs_vec(
    const T* __restrict__ norm_x,
    const i32* __restrict__ sorted_route_ids,
    T* __restrict__ aligned_in,
    int num_routes,
    int aligned_rows,
    int top_k,
    int hidden_vec,
    int shared_row_begin,
    int num_tokens)
{
    static_assert(sizeof(T) == 2, "kMoeVecWidth elements are one uint4");
    const int hv = blockIdx.y * blockDim.x + threadIdx.x;
    const int row = blockIdx.x;
    if (hv >= hidden_vec || row >= aligned_rows) return;
    int token = -1;
    if (shared_row_begin >= 0 && row >= shared_row_begin) {
        const int t = row - shared_row_begin;
        if (t < num_tokens) token = t;
    } else {
        const int route = sorted_route_ids[row];
        if (route < num_routes) token = route / top_k;
    }
    uint4 v = make_uint4(0u, 0u, 0u, 0u);
    if (token >= 0) {
        v = reinterpret_cast<const uint4*>(norm_x)[
            static_cast<long long>(token) * hidden_vec + hv];
    }
    reinterpret_cast<uint4*>(aligned_in)[
        static_cast<long long>(row) * hidden_vec + hv] = v;
}

/// The scalar gather, for a hidden size or an allocation the vector form
/// cannot take.
template <class T>
__global__ void gather_moe_aligned_inputs(
    const T* __restrict__ norm_x,
    const i32* __restrict__ sorted_route_ids,
    T* __restrict__ aligned_in,
    int num_routes,
    int aligned_rows,
    int top_k,
    int hidden,
    int shared_row_begin,
    int num_tokens)
{
    const int h = blockIdx.y * blockDim.x + threadIdx.x;
    const int row = blockIdx.x;
    if (h >= hidden || row >= aligned_rows) return;
    int token = -1;
    if (shared_row_begin >= 0 && row >= shared_row_begin) {
        const int t = row - shared_row_begin;
        if (t < num_tokens) token = t;
    } else {
        const int route = sorted_route_ids[row];
        if (route < num_routes) token = route / top_k;
    }
    T v = Elem<T>::from_f32(0.0f);
    if (token >= 0) {
        v = norm_x[static_cast<long long>(token) * hidden + h];
    }
    aligned_in[static_cast<long long>(row) * hidden + h] = v;
}

/// One thread per padded block: the aligned path's batched-GEMM pointers.
///
/// The grid is over `max_blocks`, a host-computed worst case rather than an
/// extent of anything — which is why this row states no rule. Blocks past
/// `routed_blocks` are the shared expert's and take its weights; an
/// `expert_ids` entry of -1 is padding and is clamped to expert 0, whose
/// result the combine then multiplies by a zero weight.
template <class T>
__global__ void build_moe_ptrs_aligned(
    const i32* __restrict__ expert_ids,
    const T* __restrict__ gate_up_base,
    const T* __restrict__ down_base,
    const T* __restrict__ aligned_in,
    T* __restrict__ aligned_gate_up,
    T* __restrict__ aligned_act,
    T* __restrict__ aligned_out,
    const T** __restrict__ a_gu_ptrs,
    const T** __restrict__ b_gu_ptrs,
    T** __restrict__ c_gu_ptrs,
    const T** __restrict__ a_dn_ptrs,
    const T** __restrict__ b_dn_ptrs,
    T** __restrict__ c_dn_ptrs,
    int max_blocks,
    int block_size,
    int H,
    int I_moe,
    int routed_blocks,
    const T* __restrict__ shared_gate_up_base,
    const T* __restrict__ shared_down_base)
{
    const int b = blockIdx.x * blockDim.x + threadIdx.x;
    if (b >= max_blocks) return;
    const bool is_shared = (b >= routed_blocks);
    int e = is_shared ? 0 : expert_ids[b];
    if (e < 0) e = 0;
    const long long row = static_cast<long long>(b) * block_size;
    const long long stride_gu = 2LL * I_moe * H;
    const long long stride_dn = static_cast<long long>(H) * I_moe;

    a_gu_ptrs[b] = is_shared
        ? shared_gate_up_base
        : gate_up_base + static_cast<long long>(e) * stride_gu;
    b_gu_ptrs[b] = aligned_in + row * H;
    c_gu_ptrs[b] = aligned_gate_up + row * (2LL * I_moe);

    a_dn_ptrs[b] = is_shared
        ? shared_down_base
        : down_base + static_cast<long long>(e) * stride_dn;
    b_dn_ptrs[b] = aligned_act + row * I_moe;
    c_dn_ptrs[b] = aligned_out + row * H;
}

/// Undo the permutation, eight elements per thread.
///
/// The gather's other half: an aligned row goes back to the route it came
/// from, or to the shared expert's own output when it is a tail row.
template <class T>
__global__ void reorder_moe_aligned_output_vec(
    const T* __restrict__ aligned_out,
    const i32* __restrict__ sorted_route_ids,
    T* __restrict__ route_out,
    int num_routes,
    int aligned_rows,
    int hidden_vec,
    int shared_row_begin,
    int num_tokens,
    T* __restrict__ shared_out)
{
    static_assert(sizeof(T) == 2, "kMoeVecWidth elements are one uint4");
    const int hv = blockIdx.y * blockDim.x + threadIdx.x;
    const int row = blockIdx.x;
    if (hv >= hidden_vec || row >= aligned_rows) return;
    const uint4 v = reinterpret_cast<const uint4*>(aligned_out)[
        static_cast<long long>(row) * hidden_vec + hv];
    if (shared_row_begin >= 0 && row >= shared_row_begin) {
        const int t = row - shared_row_begin;
        if (t < num_tokens) {
            reinterpret_cast<uint4*>(shared_out)[
                static_cast<long long>(t) * hidden_vec + hv] = v;
        }
        return;
    }
    const int route = sorted_route_ids[row];
    if (route >= num_routes) return;
    reinterpret_cast<uint4*>(route_out)[
        static_cast<long long>(route) * hidden_vec + hv] = v;
}

/// The scalar unpermute.
template <class T>
__global__ void reorder_moe_aligned_output(
    const T* __restrict__ aligned_out,
    const i32* __restrict__ sorted_route_ids,
    T* __restrict__ route_out,
    int num_routes,
    int aligned_rows,
    int hidden,
    int shared_row_begin,
    int num_tokens,
    T* __restrict__ shared_out)
{
    const int h = blockIdx.y * blockDim.x + threadIdx.x;
    const int row = blockIdx.x;
    if (h >= hidden || row >= aligned_rows) return;
    const T v = aligned_out[static_cast<long long>(row) * hidden + h];
    if (shared_row_begin >= 0 && row >= shared_row_begin) {
        const int t = row - shared_row_begin;
        if (t < num_tokens) {
            shared_out[static_cast<long long>(t) * hidden + h] = v;
        }
        return;
    }
    const int route = sorted_route_ids[row];
    if (route >= num_routes) return;
    route_out[static_cast<long long>(route) * hidden + h] = v;
}

/// Add each route's expert bias onto the route's row, in place.
///
/// One block per route, striding by `blockDim.x` — so this one is
/// width-agnostic, and its grid IS
/// `LaunchRule::Rms`: the value it writes is the route-major staging, one
/// row per route, so the rectangle's rows and the launcher's routes are the
/// same number. The statement is still `whole`, because `topk_idx` is
/// route-global and a window over rows would read another window's experts.
template <class T>
__global__ void add_moe_route_bias(
    T* __restrict__ out,
    const T* __restrict__ bias,
    const i32* __restrict__ topk_idx,
    int num_routes, int cols, int out_stride)
{
    const int route = blockIdx.x;
    if (route >= num_routes) return;
    const int e = topk_idx[route];
    if (e < 0) return;
    const T* b = bias + static_cast<long long>(e) * cols;
    T* o = out + static_cast<long long>(route) * out_stride;
    for (int i = threadIdx.x; i < cols; i += blockDim.x) {
        o[i] = Elem<T>::from_f32(Elem<T>::to_f32(o[i]) + Elem<T>::to_f32(b[i]));
    }
}

/// `[experts, n, k/32] -> [experts, k/32, n]`, one byte per group scale.
///
/// A quantised MoE's group scales arrive per output row and are read per
/// k-group. `T` is the byte: this kernel converts nothing and its row states
/// `u8`, which is what its buffers are.
///
/// A 3D grid with a 2D block — the expert on `z`, and 32×8 threads so that a
/// warp reads 32 contiguous k-groups. Nothing in the rule vocabulary
/// produces either, which is why the row states none.
template <class T>
__global__ void transpose_expert_scales(
    const T* __restrict__ src,
    T* __restrict__ dst,
    int n, int kg)
{
    const int e = blockIdx.z;
    const int j = blockIdx.x * blockDim.x + threadIdx.x;   // k-group
    const int i = blockIdx.y * blockDim.y + threadIdx.y;   // n
    if (i >= n || j >= kg) return;
    const long long base = static_cast<long long>(e) * n * kg;
    dst[base + static_cast<long long>(j) * n + i] =
        src[base + static_cast<long long>(i) * kg + j];
}

}  // namespace pie_cuda_driver::kernels::moe::device
