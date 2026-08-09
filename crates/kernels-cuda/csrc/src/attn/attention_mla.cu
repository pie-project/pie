#include "attn/attention_mla.hpp"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <string>

#include <cuda_bf16.h>
#include <math_constants.h>

#include <flashinfer/attention/mla.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/fastdiv.cuh>

#include "cuda_check.hpp"
#include "attn/attention_mla_naive.cuh"

// ── THE TWO NAIVE-MLA LAUNCHERS, MOVED HERE FROM THE DEVICE HEADER ──────
//
// `kernels-cuda-new/csrc/src/attn/attention_mla_naive.cuh` held these four
// host functions and the two triple-angle launches beside its two
// `__global__`s. It no longer does: it is device text and nothing else, so
// that it can be a JIT unit root, which it could not be while it opened
// `<mutex>` and `<stdexcept>`.
//
// **They moved DOWN into this file rather than being deleted, and the
// direction is the finding.** `kernels-cuda/tests/sources.rs` counts launches
// over `.cu` and `.cpp`, which was correct for the whole life of this tree —
// a `.cuh` under `kernels-cuda-new/csrc` is device text carried into NVRTC,
// and device text does not launch — and stopped being correct the moment one
// `.cuh` grew a launcher. The whole-tree census therefore read ZERO while
// these two were live (`new-horizon.md` §63.3). Widening the scan was
// available and is the wrong repair: it would leave a launch in a file whose
// extension says it cannot have one. Putting the launches in the `.cu` makes
// the census TRUE rather than merely LARGER, and this file dies with the
// archive and takes them with it.
//
// So the count in this file goes 0 -> 2 on purpose, and `sources.rs` states
// both at `EXPECTED` with this file named.
//
// THEIR RUST FORM IS WRITTEN: `driver-cuda/src/fire/mla_naive.rs`, with every
// grid, block and shared-memory figure cited to a line. It cannot replace
// these yet and the reason is not size — see that module's header, and the
// note on `dispatch_attention_mla_bf16` in `csrc/CMakeLists.txt`.

#ifndef PIE_MLA_NAIVE_CHECK
#define PIE_MLA_NAIVE_CHECK(expr)                                             \
    do {                                                                      \
        const cudaError_t pie_mla_err_ = (expr);                              \
        if (pie_mla_err_ != cudaSuccess) {                                    \
            throw std::runtime_error(std::string("naive MLA: ") +             \
                                     cudaGetErrorString(pie_mla_err_));       \
        }                                                                     \
    } while (0)
#endif

namespace pie_cuda_driver::kernels::attn::mla_naive {

// Declared before its callers rather than forward-declared: the header needed
// two forward declarations because `launch_mla_naive_paged_raw` sat above the
// tensor-core half it falls through to. Here the order is definition order and
// the declarations are gone.
namespace mma_detail {

inline std::size_t smem_bytes() {
    return static_cast<std::size_t>(kBM * kLdD + kStages * kBK * kLdD +
                                    kBM * kLdP) *
               sizeof(__nv_bfloat16) +
           static_cast<std::size_t>(kBM * kBK + 3 * kBM) * sizeof(float);
}

}  // namespace mma_detail

inline bool mla_mma_supported(int kv_lora_rank, int qk_rope_head_dim, int num_heads) {
    static const int forced = [] {
        return 0;
        if (false) return -1;
        if (false) return 1;
        return 0;
    }();
    if (forced < 0) return false;
    return kv_lora_rank == mma_detail::kCkv &&
           qk_rope_head_dim == mma_detail::kKpe &&
           num_heads % mma_detail::kBM == 0;
}

inline void launch_mla_mma_paged_raw(
    const void* q_nope, const void* q_pe,
    const void* ckv_pages, const void* kpe_pages, int page_size, void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    int total_tokens, int num_requests, int num_heads,
    float sm_scale, bool causal, cudaStream_t stream,
    const std::uint8_t* index_mask, int index_mask_stride)
{
    using namespace mma_detail;
    const std::size_t smem = smem_bytes();
    static std::once_flag opt_in;
    std::call_once(opt_in, [&] {
        PIE_MLA_NAIVE_CHECK(cudaFuncSetAttribute(
            reinterpret_cast<const void*>(mla_mma_paged_kernel),
            cudaFuncAttributeMaxDynamicSharedMemorySize,
            static_cast<int>(smem)));
    });
    dim3 grid(num_heads / kBM, total_tokens);
    mla_mma_paged_kernel<<<grid, kThreads, smem, stream>>>(
        static_cast<const __nv_bfloat16*>(q_nope),
        static_cast<const __nv_bfloat16*>(q_pe),
        static_cast<const __nv_bfloat16*>(ckv_pages),
        static_cast<const __nv_bfloat16*>(kpe_pages),
        qo_indptr_d, kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d,
        static_cast<__nv_bfloat16*>(o), index_mask, index_mask_stride,
        num_requests, num_heads, page_size, sm_scale, causal);
    PIE_MLA_NAIVE_CHECK(cudaGetLastError());
}

inline void launch_mla_naive_paged_raw(
    const void* q_nope, const void* q_pe,
    const void* ckv_pages, const void* kpe_pages,
    int kv_lora_rank, int qk_rope_head_dim, int page_size, void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    int total_tokens, int num_requests, int num_heads,
    float sm_scale, bool causal, cudaStream_t stream,
    const std::uint8_t* index_mask, int index_mask_stride)
{
    if (total_tokens <= 0) return;
    if (qo_indptr_d == nullptr || kv_page_indptr_d == nullptr ||
        kv_last_page_lens_d == nullptr) {
        throw std::runtime_error(
            "naive MLA: missing device indptr/lens (qo/kv_page_indptr/"
            "kv_last_page_lens)");
    }
    if (mla_mma_supported(kv_lora_rank, qk_rope_head_dim, num_heads)) {
        launch_mla_mma_paged_raw(
            q_nope, q_pe, ckv_pages, kpe_pages, page_size, o, qo_indptr_d,
            kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d,
            total_tokens, num_requests, num_heads, sm_scale, causal, stream,
            index_mask, index_mask_stride);
        return;
    }
    const int CKV = kv_lora_rank;
    const int KPE = qk_rope_head_dim;
    if (CKV % 32 != 0 || CKV / 32 > kMlaNaiveMaxPer) {
        throw std::runtime_error("naive MLA: unsupported kv_lora_rank");
    }
    if (KPE % 32 != 0 || KPE / 32 > kMlaNaiveMaxPePer) {
        throw std::runtime_error("naive MLA: unsupported qk_rope_head_dim");
    }
    // Pick the largest head group that still fills the machine. Every head in a
    // block walks the same keys, so a bigger group means the latent KV is read
    // from L1 instead of L2/HBM — but the grid is (tokens x head-groups), so
    // shrinking it too far starves the SMs. Two waves is the target.
    constexpr int kForcedGroup = 0;
    const int kMlaWaveTarget = 296;
    int G = kMlaNaiveWarps;
    if (kForcedGroup > 0) {
        G = kForcedGroup;
        while (G > 1 && (num_heads % G != 0 || kMlaNaiveWarps % G != 0)) G >>= 1;
    } else {
        while (G > 1 &&
               (num_heads % G != 0 ||
                static_cast<long long>(total_tokens) * (num_heads / G) <
                    kMlaWaveTarget)) {
            G >>= 1;
        }
    }
    const std::size_t smem =
        (static_cast<std::size_t>(kMlaNaiveWarps) * CKV +
         2 * kMlaNaiveWarps) * sizeof(float);
    // Wide blocks are what make this kernel fast at decode: the grid is only
    // (tokens x head-groups), so with a narrow block the SMs sit at single-digit
    // occupancy and every key's load latency is exposed. The partial-softmax
    // scratch that buys the extra warps can exceed the 48 KB default.
    static std::once_flag smem_optin;
    std::call_once(smem_optin, [&] {
        cudaFuncSetAttribute(
            reinterpret_cast<const void*>(mla_naive_paged_kernel),
            cudaFuncAttributeMaxDynamicSharedMemorySize, 200 * 1024);
    });
    dim3 grid(total_tokens, num_heads / G);
    mla_naive_paged_kernel<<<grid, kMlaNaiveBlock, smem, stream>>>(
        static_cast<const __nv_bfloat16*>(q_nope),
        static_cast<const __nv_bfloat16*>(q_pe),
        static_cast<const __nv_bfloat16*>(ckv_pages),
        static_cast<const __nv_bfloat16*>(kpe_pages),
        qo_indptr_d, kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d,
        static_cast<__nv_bfloat16*>(o),
        index_mask, index_mask_stride,
        num_requests, num_heads, CKV, KPE, page_size, sm_scale, causal, G);
    PIE_MLA_NAIVE_CHECK(cudaGetLastError());
}

}  // namespace pie_cuda_driver::kernels::attn::mla_naive

namespace pie_cuda_driver::kernels::attn {
namespace {

using DTypeQ = __nv_bfloat16;
using DTypeKV = __nv_bfloat16;
using DTypeO = __nv_bfloat16;
using IdType = int32_t;

template <typename T>
inline T* offset_ptr(void* base, std::int64_t off) {
    return reinterpret_cast<T*>(reinterpret_cast<std::uint8_t*>(base) + off);
}

}  // namespace

struct MlaPlanCache {
    ::flashinfer::MLAPlanInfo plan_info;
    int total_tokens = 0;
    int num_requests = 0;
    int num_heads = 0;
    int kv_lora_rank = 0;
    int qk_rope_head_dim = 0;
    int page_size = 0;
    bool causal = false;
    float sm_scale = 1.f;
    bool valid = false;
    std::vector<IdType> qo_h_buf;
    std::vector<IdType> kv_h_buf;
    std::vector<IdType> kv_len_h_buf;
};

void MlaPlanCacheDeleter::operator()(MlaPlanCache* p) const noexcept {
    delete p;
}
namespace {

bool mla_use_naive_backend() {
    static const int choice = [] {
        int dev = 0;
        cudaGetDevice(&dev);
        int major = 0;
        cudaDeviceGetAttribute(&major, cudaDevAttrComputeCapabilityMajor, dev);
        return major >= 10 ? 1 : 0;
    }();
    return choice == 1;
}

}  // namespace
namespace {

template <::flashinfer::MaskMode MASK>
void dispatch_mla_512_64(
    const MlaPlanCache& cache,
    const void* q_nope,
    const void* q_pe,
    MlaCacheLayerView layer,
    void* o,
    const std::uint32_t* kv_page_indices_d,
    AttentionWorkspaceView workspace,
    cudaStream_t stream,
    float* lse_out)
{
    using Params = ::flashinfer::MLAParams<DTypeQ, DTypeKV, DTypeO, IdType>;
    Params params;
    void* int_buf = workspace.int_buffer;
    void* float_buf = workspace.float_buffer;
    const auto& p = cache.plan_info;

    params.q_nope = const_cast<DTypeQ*>(static_cast<const DTypeQ*>(q_nope));
    params.q_pe = const_cast<DTypeQ*>(static_cast<const DTypeQ*>(q_pe));
    params.ckv = static_cast<DTypeKV*>(layer.ckv_pages);
    params.kpe = static_cast<DTypeKV*>(layer.kpe_pages);
    params.final_o = static_cast<DTypeO*>(o);
    params.final_lse = lse_out;
    params.partial_o = offset_ptr<DTypeO>(float_buf, p.partial_o_offset);
    params.partial_lse = offset_ptr<float>(float_buf, p.partial_lse_offset);

    params.q_indptr = offset_ptr<IdType>(int_buf, p.q_indptr_offset);
    params.kv_indptr = offset_ptr<IdType>(int_buf, p.kv_indptr_offset);
    params.partial_indptr = offset_ptr<IdType>(int_buf, p.partial_indptr_offset);
    params.merge_packed_offset_start =
        offset_ptr<IdType>(int_buf, p.merge_packed_offset_start_offset);
    params.merge_packed_offset_end =
        offset_ptr<IdType>(int_buf, p.merge_packed_offset_end_offset);
    params.merge_partial_packed_offset_start =
        offset_ptr<IdType>(int_buf, p.merge_partial_packed_offset_start_offset);
    params.merge_partial_packed_offset_end =
        offset_ptr<IdType>(int_buf, p.merge_partial_packed_offset_end_offset);
    params.merge_partial_stride =
        offset_ptr<IdType>(int_buf, p.merge_partial_stride_offset);
    params.kv_indices =
        const_cast<IdType*>(reinterpret_cast<const IdType*>(kv_page_indices_d));
    params.q_len = offset_ptr<IdType>(int_buf, p.q_len_offset);
    params.kv_len = offset_ptr<IdType>(int_buf, p.kv_len_offset);
    params.q_start = offset_ptr<IdType>(int_buf, p.q_start_offset);
    params.kv_start = offset_ptr<IdType>(int_buf, p.kv_start_offset);
    params.kv_end = offset_ptr<IdType>(int_buf, p.kv_end_offset);
    params.work_indptr = offset_ptr<IdType>(int_buf, p.work_indptr_offset);

    params.block_size = ::flashinfer::uint_fastdiv(
        static_cast<std::uint32_t>(cache.page_size));
    params.num_heads = ::flashinfer::uint_fastdiv(
        static_cast<std::uint32_t>(cache.num_heads));

    params.q_nope_stride_n =
        static_cast<std::uint32_t>(cache.num_heads * cache.kv_lora_rank);
    params.q_nope_stride_h = static_cast<std::uint32_t>(cache.kv_lora_rank);
    params.q_pe_stride_n =
        static_cast<std::uint32_t>(cache.num_heads * cache.qk_rope_head_dim);
    params.q_pe_stride_h = static_cast<std::uint32_t>(cache.qk_rope_head_dim);
    params.ckv_stride_page =
        static_cast<std::uint32_t>(cache.page_size * cache.kv_lora_rank);
    params.ckv_stride_n = static_cast<std::uint32_t>(cache.kv_lora_rank);
    params.kpe_stride_page =
        static_cast<std::uint32_t>(cache.page_size * cache.qk_rope_head_dim);
    params.kpe_stride_n = static_cast<std::uint32_t>(cache.qk_rope_head_dim);
    params.o_stride_n =
        static_cast<std::uint32_t>(cache.num_heads * cache.kv_lora_rank);
    params.o_stride_h = static_cast<std::uint32_t>(cache.kv_lora_rank);
    params.sm_scale = cache.sm_scale;
    params.return_lse_base_on_e = true;

    CUDA_CHECK((::flashinfer::mla::BatchMLAPagedAttention<MASK, 512, 64>(
        params,
        static_cast<std::uint32_t>(p.num_blks_x),
        static_cast<std::uint32_t>(p.num_blks_y),
        stream)));
}

}  // namespace

// ── Naive paged MLA (Blackwell / sm100 fallback) ────────────────────────
// FlashInfer's FA2 BatchMLAPagedAttention (a cooperative kernel) produces
// zero output on sm_100; the ecosystem (sglang/vllm) routes Blackwell MLA to
// trtllm/cutlass/ragged kernels instead. This is a correctness-first,
// arch-agnostic latent-space MLA: one block per (token, head), flash-style
// online softmax over the paged ckv/kpe cache. Output is in the kv_lora
// latent space (same as the FA2 path), so the rest of the MLA forward
// (latent_to_v, o_proj) is unchanged.
namespace {

// The kernel + launcher live in a header so `crates/driver-cuda/csrc/bench/mla_bench.cu`
// can compile the identical source standalone (seconds per iteration instead
// of a full engine rebuild).
using mla_naive::launch_mla_naive_paged_raw;

inline void launch_mla_naive_paged(
    const void* q_nope, const void* q_pe,
    const MlaCacheLayerView& layer, void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    int total_tokens, int num_requests, int num_heads,
    float sm_scale, bool causal, cudaStream_t stream,
    const std::uint8_t* index_mask, int index_mask_stride)
{
    launch_mla_naive_paged_raw(
        q_nope, q_pe, layer.ckv_pages, layer.kpe_pages,
        layer.kv_lora_rank, layer.qk_rope_head_dim, layer.page_size, o,
        qo_indptr_d, kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d,
        total_tokens, num_requests, num_heads, sm_scale, causal, stream,
        index_mask, index_mask_stride);
}

// Returns true if the naive MLA path should be used. Defaults to the device
// compute capability (Blackwell sm_100+ -> naive, since FlashInfer's FA2 MLA
// zero-outputs there); overridable via PIE_MLA_BACKEND=naive|fa2.
}  // namespace

void dispatch_attention_mla_bf16(
    const MlaPlanCache& cache,
    const void* q_nope,
    const void* q_pe,
    MlaCacheLayerView layer,
    void* o,
    const std::uint32_t* kv_page_indices_d,
    AttentionWorkspaceView workspace,
    cudaStream_t stream,
    float* lse_out,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    const std::uint8_t* index_mask,
    int index_mask_stride)
{
    if (!cache.valid) {
        throw std::runtime_error(
            "dispatch_attention_mla_bf16: cache is empty; call plan first");
    }
    if (layer.kv_lora_rank != cache.kv_lora_rank ||
        layer.qk_rope_head_dim != cache.qk_rope_head_dim ||
        layer.page_size != cache.page_size) {
        throw std::runtime_error("flashinfer MLA: layer/cache shape mismatch");
    }
    if (mla_use_naive_backend()) {
        launch_mla_naive_paged(
            q_nope, q_pe, layer, o,
            qo_indptr_d, kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d,
            cache.total_tokens, cache.num_requests, cache.num_heads,
            cache.sm_scale, cache.causal, stream,
            index_mask, index_mask_stride);
        return;
    }
    if (cache.causal) {
        dispatch_mla_512_64<::flashinfer::MaskMode::kCausal>(
            cache, q_nope, q_pe, layer, o, kv_page_indices_d,
            workspace, stream, lse_out);
    } else {
        dispatch_mla_512_64<::flashinfer::MaskMode::kNone>(
            cache, q_nope, q_pe, layer, o, kv_page_indices_d,
            workspace, stream, lse_out);
    }
}

}  // namespace pie_cuda_driver::kernels::attn
