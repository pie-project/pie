#include "ops/attention_xqa.hpp"

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <stdexcept>

#include <cuda_bf16.h>

#include "cuda_check.hpp"

// Build the GQA=5 FlashInfer XQA specialization into the native driver. Other
// ratios live in separate translation units because the XQA csrc exposes
// non-templated launch entry points.
#define NDEBUG 1
#define BEAM_WIDTH 1
#define USE_INPUT_KV 0
#define USE_CUSTOM_BARRIER 1
#define INPUT_FP16 0
#define DTYPE __nv_bfloat16
#define CACHE_ELEM_ENUM 0
#define TOKENS_PER_PAGE 32
#define HEAD_ELEMS 128
#define HEAD_GRP_SIZE 5
#define SLIDING_WINDOW 0
#define LOW_PREC_OUTPUT 0
#define SPEC_DEC 0
#define MLA_WRAPPER 0
#define USE_SM90_MHA 0
#define launchMHA launchMHA_xqa_gqa5_bf16_p32_h128
#define launchMHAFlashInfer launchMHAFlashInfer_xqa_gqa5_bf16_p32_h128

#include <xqa/mha.cu>

#undef launchMHA
#undef launchMHAFlashInfer

namespace pie_cuda_driver::ops {

namespace {

constexpr int kXqaPageSize = TOKENS_PER_PAGE;
constexpr int kXqaHeadDim = HEAD_ELEMS;
constexpr int kXqaHeadGroupRatio = HEAD_GRP_SIZE;
constexpr std::size_t kSemaphoreAlignment = 256;

std::uintptr_t align_up_ptr(std::uintptr_t p, std::size_t a) {
    return (p + a - 1) / a * a;
}

__global__ void build_xqa_metadata_kernel(
    const std::uint32_t* __restrict__ kv_page_indices,
    const std::uint32_t* __restrict__ kv_page_indptr,
    const std::uint32_t* __restrict__ kv_last_page_lens,
    std::int32_t* __restrict__ page_table,
    std::uint32_t* __restrict__ seq_lens,
    int num_requests,
    int max_pages_per_seq,
    int page_size)
{
    const int r = blockIdx.x;
    if (r >= num_requests) return;
    const int page_lo = static_cast<int>(kv_page_indptr[r]);
    const int page_hi = static_cast<int>(kv_page_indptr[r + 1]);
    const int pages = page_hi > page_lo ? (page_hi - page_lo) : 0;
    for (int p = threadIdx.x; p < max_pages_per_seq; p += blockDim.x) {
        const std::uint32_t src =
            (p < pages) ? kv_page_indices[page_lo + p] : 0u;
        page_table[static_cast<std::size_t>(r) * max_pages_per_seq + p] =
            static_cast<std::int32_t>(src);
    }
    if (threadIdx.x == 0) {
        const std::uint32_t last =
            pages > 0 ? kv_last_page_lens[r] : 0u;
        seq_lens[r] = pages > 0
            ? static_cast<std::uint32_t>((pages - 1) * page_size) + last
            : 0u;
    }
}

int current_device_major() {
    thread_local int cached_device = -1;
    thread_local int cached_major = 0;
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    if (dev != cached_device) {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        cached_device = dev;
        cached_major = prop.major;
    }
    return cached_major;
}

int current_device_sm_count() {
    thread_local int cached_device = -1;
    thread_local int cached_sms = 0;
    int dev = 0;
    CUDA_CHECK(cudaGetDevice(&dev));
    if (dev != cached_device) {
        cudaDeviceProp prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&prop, dev));
        cached_device = dev;
        cached_sms = prop.multiProcessorCount;
    }
    return cached_sms;
}

bool xqa_ratio_supported(int ratio) {
    return ratio == 5 || ratio == 8;
}

}  // namespace

namespace detail {

void launch_attention_xqa_decode_bf16_gqa8(
    const void* q,
    void* k_pages,
    void* v_pages,
    void* o,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    int max_pages_per_seq,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    float sm_scale);

void launch_attention_xqa_decode_bf16_gqa8_prepared(
    const void* q,
    void* k_pages,
    void* v_pages,
    void* o,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    int max_pages_per_seq,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    float sm_scale);

}  // namespace detail

bool xqa_decode_bf16_supported(int num_q_heads,
                               int num_kv_heads,
                               int head_dim,
                               int page_size,
                               int window_left,
                               float logits_soft_cap,
                               float sm_scale)
{
    if (num_kv_heads <= 0 || num_q_heads % num_kv_heads != 0) return false;
    if (!xqa_ratio_supported(num_q_heads / num_kv_heads)) return false;
    if (head_dim != kXqaHeadDim || page_size != kXqaPageSize) return false;
    if (window_left >= 0 || logits_soft_cap > 0.f) return false;
    const float default_scale = 1.0f / std::sqrt(static_cast<float>(head_dim));
    if (sm_scale > 0.f && std::abs(sm_scale - default_scale) > 1.0e-6f) {
        return false;
    }
    return current_device_major() >= 8;
}

void xqa_decode_bf16_gqa5_warmup_current_device() {
    // Re-issue the smem-symbol read + per-kernel attribute set that
    // FlashInfer's xqa csrc does in a static initializer. The static
    // covers only one device; multi-GPU TP needs every rank's device to
    // see the attribute or the captured kernel launch trips
    // cudaErrorInvalidValue.
    std::uint32_t size = 0;
    CUDA_CHECK(cudaMemcpyFromSymbol(&size, smemSize, sizeof(smemSize)));
    CUDA_CHECK(cudaFuncSetAttribute(
        kernel_mha,
        cudaFuncAttributeMaxDynamicSharedMemorySize,
        static_cast<int>(size)));
}

int xqa_decode_page_bucket(int max_pages_per_seq) {
    int bucket = 1;
    const int pages = std::max(1, max_pages_per_seq);
    while (bucket < pages && bucket < 4096) bucket <<= 1;
    return bucket;
}

std::uint8_t xqa_decode_graph_layout(int max_pages_per_seq) {
    int bucket = xqa_decode_page_bucket(max_pages_per_seq);
    int log2_bucket = 0;
    while (bucket > 1) {
        bucket >>= 1;
        ++log2_bucket;
    }
    return static_cast<std::uint8_t>(48 + std::min(log2_bucket, 15));
}

void launch_attention_xqa_decode_bf16(
    const void* q,
    void* k_pages,
    void* v_pages,
    void* o,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    int max_pages_per_seq,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    float sm_scale)
{
    if (!xqa_decode_bf16_supported(
            num_q_heads, num_kv_heads, head_dim, page_size,
            /*window_left=*/-1, /*logits_soft_cap=*/0.f, sm_scale)) {
        throw std::runtime_error("xqa decode: unsupported shape");
    }
    if (num_requests <= 0) return;

    prepare_attention_xqa_decode_bf16(
        kv_page_indices_d,
        kv_page_indptr_d,
        kv_last_page_lens_d,
        num_requests,
        page_size,
        max_pages_per_seq,
        workspace,
        stream);
    launch_attention_xqa_decode_bf16_prepared(
        q,
        k_pages,
        v_pages,
        o,
        num_requests,
        num_q_heads,
        num_kv_heads,
        head_dim,
        page_size,
        max_pages_per_seq,
        workspace,
        stream,
        sm_scale);
}

void prepare_attention_xqa_decode_bf16(
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    int num_requests,
    int page_size,
    int max_pages_per_seq,
    AttentionWorkspace& workspace,
    cudaStream_t stream)
{
    if (num_requests <= 0) return;
    const int page_bucket = xqa_decode_page_bucket(max_pages_per_seq);
    const std::size_t page_table_bytes =
        static_cast<std::size_t>(num_requests) * page_bucket *
        sizeof(std::int32_t);
    const std::size_t seq_lens_bytes =
        static_cast<std::size_t>(num_requests) * sizeof(std::uint32_t);
    std::uintptr_t base =
        reinterpret_cast<std::uintptr_t>(workspace.float_buffer());
    std::uintptr_t p_page_table = align_up_ptr(base, alignof(std::int32_t));
    std::uintptr_t p_seq_lens =
        align_up_ptr(p_page_table + page_table_bytes, alignof(std::uint32_t));
    std::uintptr_t p_scratch =
        align_up_ptr(p_seq_lens + seq_lens_bytes, kSemaphoreAlignment);
    const std::uintptr_t end =
        reinterpret_cast<std::uintptr_t>(workspace.float_buffer()) +
        workspace.float_bytes();
    if (p_scratch >= end) {
        throw std::runtime_error("xqa decode: attention workspace too small");
    }

    auto* page_table = reinterpret_cast<std::int32_t*>(p_page_table);
    auto* seq_lens = reinterpret_cast<std::uint32_t*>(p_seq_lens);
    build_xqa_metadata_kernel<<<num_requests, 128, 0, stream>>>(
        kv_page_indices_d,
        kv_page_indptr_d,
        kv_last_page_lens_d,
        page_table,
        seq_lens,
        num_requests,
        page_bucket,
        page_size);
    CUDA_CHECK(cudaPeekAtLastError());
}

void launch_attention_xqa_decode_bf16_prepared(
    const void* q,
    void* k_pages,
    void* v_pages,
    void* o,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    int max_pages_per_seq,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    float sm_scale)
{
    if (!xqa_decode_bf16_supported(
            num_q_heads, num_kv_heads, head_dim, page_size,
            /*window_left=*/-1, /*logits_soft_cap=*/0.f, sm_scale)) {
        throw std::runtime_error("xqa decode: unsupported shape");
    }
    if (num_requests <= 0) return;

    const int head_group_ratio = num_q_heads / num_kv_heads;
    if (head_group_ratio == 8) {
        detail::launch_attention_xqa_decode_bf16_gqa8_prepared(
            q,
            k_pages,
            v_pages,
            o,
            num_requests,
            num_q_heads,
            num_kv_heads,
            head_dim,
            page_size,
            max_pages_per_seq,
            workspace,
            stream,
            sm_scale);
        return;
    }
    if (head_group_ratio != kXqaHeadGroupRatio) {
        throw std::runtime_error("xqa decode: unsupported GQA ratio");
    }

    const int page_bucket = xqa_decode_page_bucket(max_pages_per_seq);
    const std::size_t page_table_bytes =
        static_cast<std::size_t>(num_requests) * page_bucket *
        sizeof(std::int32_t);
    const std::size_t seq_lens_bytes =
        static_cast<std::size_t>(num_requests) * sizeof(std::uint32_t);
    std::uintptr_t base =
        reinterpret_cast<std::uintptr_t>(workspace.float_buffer());
    std::uintptr_t p_page_table = align_up_ptr(base, alignof(std::int32_t));
    std::uintptr_t p_seq_lens =
        align_up_ptr(p_page_table + page_table_bytes, alignof(std::uint32_t));
    std::uintptr_t p_scratch =
        align_up_ptr(p_seq_lens + seq_lens_bytes, kSemaphoreAlignment);
    const std::uintptr_t end =
        reinterpret_cast<std::uintptr_t>(workspace.float_buffer()) +
        workspace.float_bytes();
    if (p_scratch >= end) {
        throw std::runtime_error("xqa decode: attention workspace too small");
    }

    auto* page_table = reinterpret_cast<std::int32_t*>(p_page_table);
    auto* seq_lens = reinterpret_cast<std::uint32_t*>(p_seq_lens);
    void* scratch = reinterpret_cast<void*>(p_scratch);

    const int semaphore_count = num_requests * num_kv_heads;
    if (static_cast<std::size_t>(semaphore_count) * sizeof(std::uint32_t) >
        workspace.int_bytes()) {
        throw std::runtime_error("xqa decode: semaphore workspace too small");
    }
    auto* semaphores =
        reinterpret_cast<std::uint32_t*>(workspace.int_buffer());
    CUDA_CHECK(cudaMemsetAsync(
        semaphores, 0,
        static_cast<std::size_t>(semaphore_count) * sizeof(std::uint32_t),
        stream));

    const float q_scale = 1.0f;
    const float kv_scale = 1.0f;
    const std::uint64_t kv_stride_head =
        static_cast<std::uint64_t>(head_dim);
    const std::uint64_t kv_stride_token =
        static_cast<std::uint64_t>(num_kv_heads) * head_dim;
    const std::uint64_t kv_stride_page =
        static_cast<std::uint64_t>(page_size) * num_kv_heads * head_dim;

    launchMHAFlashInfer_xqa_gqa5_bf16_p32_h128(
        static_cast<std::uint32_t>(current_device_sm_count()),
        static_cast<std::uint32_t>(num_kv_heads),
        /*slidingWinSize=*/0,
        q_scale,
        /*qScalePtr=*/nullptr,
        reinterpret_cast<OutputHead*>(o),
        reinterpret_cast<InputHead const*>(q),
        /*attentionSinks=*/nullptr,
        reinterpret_cast<GMemCacheHead*>(k_pages),
        reinterpret_cast<GMemCacheHead*>(v_pages),
        reinterpret_cast<KVCachePageIndex const*>(page_table),
        static_cast<std::uint32_t>(page_bucket * page_size),
        seq_lens,
        static_cast<std::uint32_t>(num_requests),
        kv_scale,
        /*kvScalePtr=*/nullptr,
        semaphores,
        scratch,
        current_device_major() >= 9,
        kv_stride_page,
        kv_stride_token,
        kv_stride_head,
        stream);
}

}  // namespace pie_cuda_driver::ops
