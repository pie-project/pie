// FlashInfer-backed paged attention for ggml's CUDA backend.
//
// Registers an out-of-tree implementation against the
// `GGML_OP_PAGED_ATTN_EXT` dispatch hook in ggml-cuda
// (`ggml_cuda_register_paged_attn_ext`, added by
// `patches/ggml-pie-paged-attn-ext.patch`). The hook stays null until
// this translation unit's static initializer fires; while null,
// supports_op returns false and Pie's graph builders take the
// materialize + flash_attn_ext fallback.
//
// Compiled only when `GGML_CUDA=ON`; conditional in the parent
// CMakeLists. Other backends (Metal / Vulkan / CPU) never see this
// code.
//
// Why FlashInfer: the canonical decode kernel for paged KV (used by
// vLLM and SGLang). Our pool layout `[n_kv_heads*head_dim,
// total_pages*page_size]` matches FlashInfer's `kNHD` byte-for-byte
// (offset formula coincides), so K/V can be wrapped without copies.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <mutex>
#include <stdexcept>
#include <vector>

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cuda_bf16.h>

#include <ggml.h>
#include <ggml-backend.h>

#include <flashinfer/attention/decode.cuh>
#include <flashinfer/attention/default_decode_params.cuh>
#include <flashinfer/attention/scheduler.cuh>
#include <flashinfer/attention/variants.cuh>
#include <flashinfer/layout.cuh>
#include <flashinfer/page.cuh>

namespace {

// Pie's KV cache is F16; Pie casts Q to BF16 in the graph builder
// before the paged op. FlashInfer's mixed-precision support requires
// the K/V types to match each other. We use F16 for both K and V to
// match Pie's pool dtype, and BF16 for Q (matching driver/cuda's
// production setup).
using DTypeQ  = __nv_bfloat16;
using DTypeKV = half;
using DTypeO  = __nv_bfloat16;
using IdType  = int32_t;

constexpr auto POS_ENC = ::flashinfer::PosEncodingMode::kNone;

// Two attention variants — plain and softcapped — mirroring driver/cuda.
// `use_sliding_window=true` lets us pass `window_left` per-call;
// setting `window_left = -1` makes the predicate trivially true (full
// causal), so this single variant covers both Pie code paths.
using AttnVariant = ::flashinfer::DefaultAttention<
    /*use_custom_mask=*/false,
    /*use_sliding_window=*/true,
    /*use_logits_soft_cap=*/false,
    /*use_alibi=*/false>;

using AttnVariantSoftcap = ::flashinfer::DefaultAttention<
    /*use_custom_mask=*/false,
    /*use_sliding_window=*/true,
    /*use_logits_soft_cap=*/true,
    /*use_alibi=*/false>;

using DecodeParams = ::flashinfer::BatchDecodeParams<DTypeQ, DTypeKV, DTypeO, IdType>;

// Workspace for FlashInfer's planner + kernel. Allocated lazily on
// first call, sized for the largest realistic batch (n_req=512,
// max_pages_per_req=512). Held for the lifetime of the process.
class Workspace {
public:
    void ensure_size(std::size_t int_bytes, std::size_t float_bytes) {
        std::lock_guard<std::mutex> lk(mu_);
        if (int_bytes > int_bytes_) {
            cudaFree(int_buf_);
            cudaMalloc(&int_buf_, int_bytes);
            int_bytes_ = int_bytes;
        }
        if (float_bytes > float_bytes_) {
            cudaFree(float_buf_);
            cudaMalloc(&float_buf_, float_bytes);
            float_bytes_ = float_bytes;
        }
        if (page_locked_int_bytes_ < int_bytes) {
            cudaFreeHost(page_locked_int_);
            cudaMallocHost(&page_locked_int_, int_bytes);
            page_locked_int_bytes_ = int_bytes;
        }
    }
    void* int_buf()           const { return int_buf_; }
    void* float_buf()         const { return float_buf_; }
    void* page_locked_int()   const { return page_locked_int_; }
    std::size_t int_bytes()   const { return int_bytes_; }
    std::size_t float_bytes() const { return float_bytes_; }

private:
    std::mutex  mu_;
    void*       int_buf_           = nullptr;
    void*       float_buf_         = nullptr;
    void*       page_locked_int_   = nullptr;
    std::size_t int_bytes_           = 0;
    std::size_t float_bytes_         = 0;
    std::size_t page_locked_int_bytes_ = 0;
};

Workspace& workspace() {
    static Workspace w;
    return w;
}

// Side channel for the host-side `kv_page_indptr` array. FlashInfer's
// DecodePlan needs it on the host to compute a partition-kv schedule.
// Pie already has the array on the host before upload_graph_inputs runs;
// we stash it here so the kernel doesn't have to round-trip through a
// device→host copy. Set by `pie_paged_attn_set_host_indptr()`, called
// from `upload_graph_inputs` immediately before the graph compute.
//
// thread_local because Pie may have multiple ForwardEngines (one per
// device) running on separate threads; each gets its own snapshot.
thread_local std::vector<IdType> g_host_indptr;

// Same for sliding_window: FlashInfer's `window_left` parameter is set
// per-call; the value lives in op_params already, but stashing here
// keeps the planner+launcher symmetric with the indptr cache and
// lets the planner specialize on full-vs-sliding when we ever want it.
thread_local int g_sliding_window = -1;

// Plan cache. FlashInfer's DecodePlan is the expensive step (~2 ms per
// call on Blackwell): it allocates scratch, computes a partition-kv
// schedule, and host-syncs. The plan is a function of (n_req,
// num_q_heads, num_kv_heads, page_size, head_dim, indptr-content), all
// of which are stable across the layers of one forward pass. Cache the
// last plan and reuse when those keys match.
struct PlanCacheKey {
    uint32_t  num_requests   = 0;
    uint32_t  num_q_heads    = 0;
    uint32_t  num_kv_heads   = 0;
    uint32_t  page_size      = 0;
    uint32_t  head_dim       = 0;
    uint32_t  group_size     = 0;
    bool      use_softcap    = false;
    std::size_t indptr_hash    = 0;
    bool operator==(const PlanCacheKey& o) const {
        return num_requests == o.num_requests &&
               num_q_heads  == o.num_q_heads  &&
               num_kv_heads == o.num_kv_heads &&
               page_size    == o.page_size    &&
               head_dim     == o.head_dim     &&
               group_size   == o.group_size   &&
               use_softcap  == o.use_softcap  &&
               indptr_hash  == o.indptr_hash;
    }
};

struct PlanCacheEntry {
    PlanCacheKey                 key;
    ::flashinfer::DecodePlanInfo plan_info;
    bool                         valid = false;
};

thread_local PlanCacheEntry g_plan_cache;

std::size_t hash_indptr(const IdType* p, std::size_t n) {
    // Lightweight FNV-1a; the indptr is small (n_req+1 entries) and we
    // only need to detect changes across forward passes, not adversarial
    // collisions.
    std::size_t h = 1469598103934665603ull;
    for (std::size_t i = 0; i < n; ++i) {
        h = (h ^ static_cast<std::size_t>(p[i])) * 1099511628211ull;
    }
    return h;
}

// Per-(head_dim, group_size) work estimator for the planner.
//
// We force `split_kv = false` after FlashInfer's estimator runs. The
// adaptive choice would flip to two-stage (split-K + reduction) at long
// contexts, which changes the actual kernel launches FlashInfer makes.
// ggml-cuda's CUDA graph capture keys on observed launches; a mid-bench
// flip invalidates the captured graph and triggers a recapture stall
// that has been observed to hang at 256+ tokens. Single-stage trades
// some perf at very long contexts (>~4K KV/req) for deterministic
// launches and stable graph capture across all decode steps.
template <uint32_t HEAD_DIM, uint32_t GROUP_SIZE>
struct DecodeWorkEstimator {
    cudaError_t operator()(bool& split_kv, uint32_t& max_grid_size,
                           uint32_t& max_num_pages_per_batch, uint32_t& new_batch_size,
                           uint32_t& gdy, uint32_t batch_size, IdType* kv_indptr_h,
                           uint32_t num_qo_heads, uint32_t page_size, bool enable_cuda_graph,
                           cudaStream_t stream) {
        cudaError_t rc = ::flashinfer::BatchDecodeWithPagedKVCacheWorkEstimationDispatched<
            GROUP_SIZE, HEAD_DIM, POS_ENC, AttnVariant, DecodeParams>(
            split_kv, max_grid_size, max_num_pages_per_batch, new_batch_size, gdy,
            batch_size, kv_indptr_h, num_qo_heads, page_size, enable_cuda_graph, stream);
        split_kv       = false;
        new_batch_size = batch_size;
        return rc;
    }
};

// Plan + dispatch for a single (HEAD_DIM, gqa_group_size, Variant) tuple.
template <uint32_t HEAD_DIM, uint32_t GROUP_SIZE, class Variant>
cudaError_t plan_and_dispatch(
        cudaStream_t stream,
        uint32_t num_requests,
        uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t page_size,
        DTypeQ*  q_ptr,
        DTypeKV* k_pool_ptr,
        DTypeKV* v_pool_ptr,
        IdType*  page_indices_d,
        IdType*  page_indptr_d,
        IdType*  last_page_lens_d,
        DTypeO*  out_ptr,
        int      window_left,
        float    sm_scale,
        float    logits_soft_cap)
{
    Workspace& ws = workspace();

    // Build cache key from this call's shape + indptr content.
    PlanCacheKey key;
    key.num_requests = num_requests;
    key.num_q_heads  = num_q_heads;
    key.num_kv_heads = num_kv_heads;
    key.page_size    = page_size;
    key.head_dim     = HEAD_DIM;
    key.group_size   = GROUP_SIZE;
    key.use_softcap  = std::is_same<Variant, AttnVariantSoftcap>::value;
    key.indptr_hash  = hash_indptr(
        g_host_indptr.data(), static_cast<std::size_t>(num_requests + 1));

    if (!g_plan_cache.valid || !(g_plan_cache.key == key)) {
        DecodeWorkEstimator<HEAD_DIM, GROUP_SIZE> estimator;
        cudaError_t st = ::flashinfer::DecodePlan<HEAD_DIM, POS_ENC, AttnVariant, DecodeParams>(
            ws.float_buf(), ws.float_bytes(),
            ws.int_buf(),   ws.page_locked_int(), ws.int_bytes(),
            g_plan_cache.plan_info,
            g_host_indptr.data(),
            num_requests, num_q_heads, page_size,
            /*enable_cuda_graph=*/true,
            stream, estimator);
        if (st != cudaSuccess) return st;
        g_plan_cache.key   = key;
        g_plan_cache.valid = true;
    }
    const ::flashinfer::DecodePlanInfo& plan_info = g_plan_cache.plan_info;

    ::flashinfer::paged_kv_t<DTypeKV, IdType> paged_kv(
        num_kv_heads, page_size, HEAD_DIM, num_requests,
        ::flashinfer::QKVLayout::kNHD,
        k_pool_ptr, v_pool_ptr,
        page_indices_d, page_indptr_d, last_page_lens_d);

    DecodeParams params;
    params.q                  = q_ptr;
    params.q_rope_offset      = nullptr;
    params.paged_kv           = paged_kv;
    params.o                  = out_ptr;
    params.lse                = nullptr;
    params.maybe_alibi_slopes = nullptr;
    params.num_qo_heads       = num_q_heads;
    params.q_stride_n         = static_cast<IdType>(num_q_heads * HEAD_DIM);
    params.q_stride_h         = static_cast<IdType>(HEAD_DIM);
    params.window_left        = window_left;
    params.logits_soft_cap    = logits_soft_cap;
    params.sm_scale           = sm_scale;
    params.rope_rcp_scale     = 1.0f;
    params.rope_rcp_theta     = 1.0f;

    auto offset_int = [&](std::int64_t off) {
        return reinterpret_cast<IdType*>(
            reinterpret_cast<std::uint8_t*>(ws.int_buf()) + off);
    };
    params.request_indices    = offset_int(plan_info.request_indices_offset);
    params.kv_tile_indices    = offset_int(plan_info.kv_tile_indices_offset);
    params.o_indptr           = offset_int(plan_info.o_indptr_offset);
    params.kv_chunk_size_ptr  = offset_int(plan_info.kv_chunk_size_ptr_offset);
    params.padded_batch_size  = static_cast<uint32_t>(plan_info.padded_batch_size);
    params.partition_kv       = plan_info.split_kv;

    DTypeO* tmp_v = nullptr;
    float*  tmp_s = nullptr;
    if (plan_info.split_kv) {
        tmp_v = reinterpret_cast<DTypeO*>(
            reinterpret_cast<std::uint8_t*>(ws.float_buf()) + plan_info.v_offset);
        tmp_s = reinterpret_cast<float*>(
            reinterpret_cast<std::uint8_t*>(ws.float_buf()) + plan_info.s_offset);
    }

    return ::flashinfer::BatchDecodeWithPagedKVCacheDispatched<
        HEAD_DIM, POS_ENC, Variant, DecodeParams>(
        params, tmp_v, tmp_s, /*enable_pdl=*/false, stream);
}

template <uint32_t HEAD_DIM, class Variant>
cudaError_t dispatch_for_gqa(
        cudaStream_t stream,
        uint32_t num_requests,
        uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t page_size,
        DTypeQ* q, DTypeKV* k, DTypeKV* v,
        IdType* pi, IdType* pip, IdType* lpl,
        DTypeO* o,
        int window_left, float scale, float softcap)
{
    const uint32_t group = num_q_heads / num_kv_heads;
    switch (group) {
        case 1: return plan_and_dispatch<HEAD_DIM, 1, Variant>(
            stream, num_requests, num_q_heads, num_kv_heads, page_size,
            q, k, v, pi, pip, lpl, o, window_left, scale, softcap);
        case 2: return plan_and_dispatch<HEAD_DIM, 2, Variant>(
            stream, num_requests, num_q_heads, num_kv_heads, page_size,
            q, k, v, pi, pip, lpl, o, window_left, scale, softcap);
        case 4: return plan_and_dispatch<HEAD_DIM, 4, Variant>(
            stream, num_requests, num_q_heads, num_kv_heads, page_size,
            q, k, v, pi, pip, lpl, o, window_left, scale, softcap);
        case 8: return plan_and_dispatch<HEAD_DIM, 8, Variant>(
            stream, num_requests, num_q_heads, num_kv_heads, page_size,
            q, k, v, pi, pip, lpl, o, window_left, scale, softcap);
    }
    return cudaErrorInvalidValue;
}

template <class Variant>
cudaError_t dispatch_for_head_dim(
        uint32_t head_dim,
        cudaStream_t stream,
        uint32_t num_requests,
        uint32_t num_q_heads, uint32_t num_kv_heads, uint32_t page_size,
        DTypeQ* q, DTypeKV* k, DTypeKV* v,
        IdType* pi, IdType* pip, IdType* lpl,
        DTypeO* o,
        int window_left, float scale, float softcap)
{
    switch (head_dim) {
        case 64:  return dispatch_for_gqa< 64, Variant>(
            stream, num_requests, num_q_heads, num_kv_heads, page_size,
            q, k, v, pi, pip, lpl, o, window_left, scale, softcap);
        case 128: return dispatch_for_gqa<128, Variant>(
            stream, num_requests, num_q_heads, num_kv_heads, page_size,
            q, k, v, pi, pip, lpl, o, window_left, scale, softcap);
        case 256: return dispatch_for_gqa<256, Variant>(
            stream, num_requests, num_q_heads, num_kv_heads, page_size,
            q, k, v, pi, pip, lpl, o, window_left, scale, softcap);
    }
    return cudaErrorInvalidValue;
}

// ── ggml-cuda hook implementations ────────────────────────────────────────

void pie_cuda_paged_attn_dispatch(cudaStream_t stream, ggml_tensor* dst) {
    const ggml_tensor* Q              = dst->src[0];
    const ggml_tensor* K_pool         = dst->src[1];
    const ggml_tensor* V_pool         = dst->src[2];
    const ggml_tensor* page_indices   = dst->src[3];
    const ggml_tensor* page_indptr    = dst->src[4];
    const ggml_tensor* last_page_lens = dst->src[5];

    const float* params_f32 = (const float*) dst->op_params;
    const int*   params_i32 = (const int*)   dst->op_params;
    const float scale          = params_f32[0];
    const float softcap        = params_f32[1];
    const int   page_size      = params_i32[2];
    const int   head_dim       = params_i32[3];
    const int   n_kv_heads     = params_i32[4];
    const int   sliding_window = params_i32[5];

    const int n_q_heads = Q->ne[2];
    const int n_req     = Q->ne[3];

    // Lazy workspace allocation. FlashInfer's planner + kernel read/write
    // a few hundred KB of scratch; size for the worst case we expect.
    workspace().ensure_size(/*int_bytes=*/  16ull * 1024 * 1024,
                            /*float_bytes=*/64ull * 1024 * 1024);

    // The host-side indptr stash must have been set by upload_graph_inputs.
    if ((int) g_host_indptr.size() < n_req + 1) {
        std::fprintf(stderr,
            "[paged_attn_cuda] host indptr not set or too small "
            "(have %zu, need %d) — set via pie_paged_attn_set_host_indptr() "
            "in upload_graph_inputs.\n",
            g_host_indptr.size(), n_req + 1);
        std::abort();
    }

    auto* q_ptr   = (DTypeQ*)  Q->data;
    auto* k_ptr   = (DTypeKV*) K_pool->data;
    auto* v_ptr   = (DTypeKV*) V_pool->data;
    auto* pi_ptr  = (IdType*)  page_indices->data;
    auto* pip_ptr = (IdType*)  page_indptr->data;
    auto* lpl_ptr = (IdType*)  last_page_lens->data;
    auto* o_ptr   = (DTypeO*)  dst->data;

    cudaError_t st;
    if (softcap > 0.0f) {
        st = dispatch_for_head_dim<AttnVariantSoftcap>(
            head_dim, stream, (uint32_t) n_req,
            (uint32_t) n_q_heads, (uint32_t) n_kv_heads, (uint32_t) page_size,
            q_ptr, k_ptr, v_ptr, pi_ptr, pip_ptr, lpl_ptr, o_ptr,
            sliding_window, scale, softcap);
    } else {
        st = dispatch_for_head_dim<AttnVariant>(
            head_dim, stream, (uint32_t) n_req,
            (uint32_t) n_q_heads, (uint32_t) n_kv_heads, (uint32_t) page_size,
            q_ptr, k_ptr, v_ptr, pi_ptr, pip_ptr, lpl_ptr, o_ptr,
            sliding_window, scale, softcap);
    }

    if (st != cudaSuccess) {
        std::fprintf(stderr, "[paged_attn_cuda] dispatch failed: %s\n",
                     cudaGetErrorString(st));
        std::abort();
    }
}

bool pie_cuda_paged_attn_supported(int /*device*/, const ggml_tensor* dst) {
    const ggml_tensor* Q      = dst->src[0];
    const ggml_tensor* K_pool = dst->src[1];
    const ggml_tensor* V_pool = dst->src[2];

    if (Q->type != GGML_TYPE_BF16) return false;
    if (K_pool->type != GGML_TYPE_F16) return false;
    if (V_pool->type != GGML_TYPE_F16) return false;
    if (dst->type   != GGML_TYPE_BF16) return false;

    const int* params_i32 = (const int*) dst->op_params;
    const int head_dim    = params_i32[3];
    const int n_kv_heads  = params_i32[4];
    if (head_dim != 64 && head_dim != 128 && head_dim != 256) return false;

    const int n_q_heads = Q->ne[2];
    const int gqa_ratio = n_q_heads / n_kv_heads;
    if (gqa_ratio != 1 && gqa_ratio != 2 && gqa_ratio != 4 && gqa_ratio != 8) {
        return false;
    }
    if (Q->ne[1] != 1) return false;   // decode-only

    return true;
}

}  // namespace

// ── Public API consumed by Pie ───────────────────────────────────────────

extern "C" void pie_paged_attn_set_host_indptr(
        const std::int32_t* indptr, std::size_t n_req_plus_one) {
    g_host_indptr.assign(indptr, indptr + n_req_plus_one);
}

extern "C" void pie_paged_attn_set_sliding_window(int window_left) {
    g_sliding_window = window_left;
}

// ── Static registration with ggml-cuda ───────────────────────────────────

extern "C" void ggml_cuda_register_paged_attn_ext(
        void (*dispatch)(cudaStream_t, ggml_tensor*),
        bool (*supported)(int, const ggml_tensor*));

namespace {
struct PagedAttnRegistration {
    PagedAttnRegistration() {
        ggml_cuda_register_paged_attn_ext(
            pie_cuda_paged_attn_dispatch,
            pie_cuda_paged_attn_supported);
    }
};
PagedAttnRegistration _registration;
}  // namespace
