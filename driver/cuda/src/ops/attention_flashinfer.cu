//===-- attention_flashinfer.cu ---------------------------------*- CUDA -*-===//
//
// Public entry points of the FA2 attention driver. Everything templated on
// HEAD_DIM lives in attention_flashinfer_common.cuh and is instantiated by the
// per-head_dim translation units; this file only holds the plan-cache
// plumbing and the head_dim dispatch, both driven by src/kernels.def.
//
//===----------------------------------------------------------------------===//
#include "ops/attention_flashinfer_common.cuh"

namespace pie_cuda_driver::ops {

void DecodePlanCacheDeleter::operator()(DecodePlanCache* p) const noexcept {
    delete p;
}

void PrefillPlanCacheDeleter::operator()(PrefillPlanCache* p) const noexcept {
    delete p;
}

DecodePlanCachePtr make_decode_plan() {
    return DecodePlanCachePtr(new DecodePlanCache{});
}

PrefillPlanCachePtr make_prefill_plan() {
    return PrefillPlanCachePtr(new PrefillPlanCache{});
}

std::uint32_t decode_plan_graph_layout(const DecodePlanCache& cache) {
    if (!cache.valid) return 0;
    return static_cast<std::uint32_t>(
        (cache.plan_info.split_kv ? 2u : 1u) |
        (cache.full_attention_variant ? 4u : 0u) |
        (cache.hnd_layout ? 8u : 0u));
}

bool decode_plan_is_page_count_independent(const DecodePlanCache& cache) {
    return cache.valid && cache.page_count_independent;
}

std::uint32_t prefill_plan_graph_layout(const PrefillPlanCache& cache) {    if (!cache.valid) return 0;
    if (cache.use_sm90) {
        return 0x00800000u |
               static_cast<std::uint32_t>(
                   hopper_prefill_graph_layout(cache.sm90_plan));
    }
    std::uint32_t tile_class = 0;
    switch (cache.plan_info.cta_tile_q) {
        case 16:  tile_class = 1; break;
        case 32:  tile_class = 2; break;
        case 64:  tile_class = 3; break;
        case 128: tile_class = 4; break;
        default:  tile_class = 0; break;
    }
    const std::uint32_t variant_class =
        cache.full_attention_variant ? 1u : 0u;
    const auto padded_batch_size = static_cast<std::uint32_t>(
        std::min<std::int64_t>(cache.plan_info.padded_batch_size,
                               0x000fffff));
    return static_cast<std::uint32_t>(
        0x100u |
        (cache.plan_info.split_kv ? 1u : 0u) |
        (tile_class << 1) |
        (variant_class << 4) |
        (cache.hnd_layout ? 32u : 0u) |
        (cache.causal_mask ? 64u : 0u) |
        (padded_batch_size << 8));
}

namespace {

bool can_use_static_nonsplit_decode_plan(uint32_t num_requests) {
    // DecodeWorkEstimator above already overrides FlashInfer's split-kv choice
    // to false for the TP1 latency shapes we care about. In that case the
    // schedule is independent of KV lengths, so avoid rerunning the full
    // FlashInfer planner for every decode batch.
    return static_nonsplit_decode_plan_enabled() &&
           current_device_major() >= 8 &&
           num_requests > 0 &&
           num_requests <= 512 &&
           !force_split_kv_small_enabled();
}

void refresh_static_nonsplit_decode_vectors(
    DecodePlanCache& cache,
    int num_requests)
{
    if (cache.static_nonsplit_num_requests == num_requests) {
        return;
    }
    cache.static_nonsplit_num_requests = num_requests;
    cache.static_request_indices.resize(num_requests);
    cache.static_kv_tile_indices.assign(num_requests, IdType{0});
    cache.static_o_indptr.resize(num_requests + 1);
    for (int r = 0; r < num_requests; ++r) {
        cache.static_request_indices[r] = static_cast<IdType>(r);
        cache.static_o_indptr[r] = static_cast<IdType>(r);
    }
    cache.static_o_indptr[num_requests] = static_cast<IdType>(num_requests);
}

void plan_static_nonsplit_decode(
    DecodePlanCache& cache,
    const std::uint32_t* kv_page_indptr_h,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    bool enable_cuda_graph,
    bool full_attention_variant,
    bool hnd_layout)
{
    refresh_static_nonsplit_decode_vectors(cache, num_requests);

    std::size_t cursor = 0;
    auto alloc = [&](std::size_t bytes, std::size_t alignment) {
        cursor = align_up_bytes(cursor, alignment);
        const std::size_t offset = cursor;
        cursor += bytes;
        return static_cast<std::int64_t>(offset);
    };

    auto& plan = cache.plan_info;
    plan.padded_batch_size = num_requests;
    plan.v_offset = 0;
    plan.s_offset = 0;
    plan.request_indices_offset =
        alloc(sizeof(IdType) * static_cast<std::size_t>(num_requests), 16);
    plan.kv_tile_indices_offset =
        alloc(sizeof(IdType) * static_cast<std::size_t>(num_requests), 16);
    plan.o_indptr_offset =
        alloc(sizeof(IdType) * static_cast<std::size_t>(num_requests + 1), 16);
    plan.kv_chunk_size_ptr_offset = alloc(sizeof(IdType), 1);
    plan.block_valid_mask_offset = 0;
    plan.enable_cuda_graph = enable_cuda_graph;
    plan.split_kv = false;

    if (cursor + cache.int_base_bytes > workspace.int_bytes()) {
        throw std::runtime_error(
            "flashinfer decode static plan: attention int workspace too small");
    }

    auto* host = static_cast<std::uint8_t*>(workspace.page_locked_int()) +
                 cache.int_base_bytes;
    std::memcpy(host + plan.request_indices_offset,
                cache.static_request_indices.data(),
                sizeof(IdType) * static_cast<std::size_t>(num_requests));
    std::memcpy(host + plan.kv_tile_indices_offset,
                cache.static_kv_tile_indices.data(),
                sizeof(IdType) * static_cast<std::size_t>(num_requests));
    std::memcpy(host + plan.o_indptr_offset,
                cache.static_o_indptr.data(),
                sizeof(IdType) * static_cast<std::size_t>(num_requests + 1));
    *reinterpret_cast<IdType*>(host + plan.kv_chunk_size_ptr_offset) =
        static_cast<IdType>(page_size);

    CUDA_CHECK(cudaMemcpyAsync(
        static_cast<std::uint8_t*>(workspace.int_buffer()) +
            cache.int_base_bytes,
        static_cast<std::uint8_t*>(workspace.page_locked_int()) +
            cache.int_base_bytes,
        cursor, cudaMemcpyHostToDevice, stream));

    cache.num_requests = num_requests;
    cache.num_q_heads  = num_q_heads;
    cache.num_kv_heads = num_kv_heads;
    cache.head_dim     = head_dim;
    cache.page_size    = page_size;
    cache.num_pages_in_batch = kv_page_indptr_h[num_requests];
    cache.enable_pdl   = current_device_supports_pdl();
    cache.full_attention_variant = full_attention_variant;
    cache.hnd_layout   = hnd_layout;
    cache.valid        = true;
}

}  // namespace

void set_decode_plan_int_base(DecodePlanCache& cache, std::size_t bytes) {
    cache.int_base_bytes = bytes;
}

void plan_attention_flashinfer_decode_bf16(
    DecodePlanCache& cache,
    const std::uint32_t* kv_page_indptr_h,
    int num_requests,
    int num_q_heads, int num_kv_heads, int head_dim, int page_size,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    bool enable_cuda_graph,
    bool full_attention_variant,
    bool hnd_layout)
{
    const int gqa_group_size = num_q_heads / num_kv_heads;
    // Checked up front, not just in the switch below: the static non-split
    // plan short-circuits past the dispatch entirely, so an unsupported
    // head_dim would otherwise be reported as a valid plan and only fail
    // later inside the kernel launch.
    if (!attn_head_dim_instantiated(head_dim)) {
        throw_unsupported_head_dim("flashinfer decode", head_dim);
    }

    if (can_use_static_nonsplit_decode_plan(
            static_cast<uint32_t>(num_requests))) {
        plan_static_nonsplit_decode(
            cache, kv_page_indptr_h, num_requests, num_q_heads, num_kv_heads,
            head_dim, page_size, workspace, stream, enable_cuda_graph,
            full_attention_variant, hnd_layout);
        cache.page_count_independent = true;
        return;
    }
    cache.page_count_independent = false;

    cache.indptr_h_buf.resize(num_requests + 1);
    for (int r = 0; r <= num_requests; ++r) {
        cache.indptr_h_buf[r] = static_cast<IdType>(kv_page_indptr_h[r]);
    }

    cudaError_t status;
    switch (head_dim) {
#define PIE_ATTN_HEAD_DIM(HD)                                                \
        case HD:                                                             \
            status = AttnHd<HD>::plan_decode(                                \
                cache, cache.indptr_h_buf,                                   \
                num_requests, num_q_heads, page_size, gqa_group_size,        \
                workspace, stream, enable_cuda_graph, full_attention_variant);\
            break;
#include "kernels.def"
        default:
            throw_unsupported_head_dim("flashinfer decode", head_dim);
    }
    CUDA_CHECK(status);

    cache.num_requests = num_requests;
    cache.num_q_heads  = num_q_heads;
    cache.num_kv_heads = num_kv_heads;
    cache.head_dim     = head_dim;
    cache.page_size    = page_size;
    cache.num_pages_in_batch = kv_page_indptr_h[num_requests];
    cache.enable_pdl   = current_device_supports_pdl();
    cache.full_attention_variant = full_attention_variant;
    cache.hnd_layout   = hnd_layout;
    cache.valid        = true;
}

void plan_attention_flashinfer_prefill_bf16(
    PrefillPlanCache& cache,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    const std::uint32_t* kv_last_page_lens_h,
    int total_tokens,
    int num_requests,
    int num_q_heads,
    int num_kv_heads,
    int head_dim,
    int page_size,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    bool enable_cuda_graph,
    int window_left,
    bool full_attention_variant,
    bool hnd_layout,
    bool causal_mask,
    bool custom_mask,
    bool wants_prefill_score)  // NOLINT(readability-non-const-parameter)
{
    if (!attn_head_dim_instantiated(head_dim)) {
        throw_unsupported_head_dim("flashinfer prefill plan", head_dim);
    }
    if (wants_prefill_score) {
        if (window_left >= 0) {
            throw std::invalid_argument(
                "prefill score capture does not support sliding-window "
                "attention: LogitsMask runs after LogitsTransform, so the "
                "captured row would include positions the softmax discards");
        }
        // `AttnVariant` differs from `AttnVariantFull` only by a runtime
        // window predicate that is trivially true at `window_left < 0` (see
        // the alias comments in attention_flashinfer_common.cuh), and the
        // prefill plan itself does not depend on the variant -- only on
        // `window_left` and geometry. Promoting here therefore changes no
        // numerics; it just spares the capture kernel a second instantiation
        // over a template flag that cannot fire.
        full_attention_variant = true;
    }
    cache.use_sm90 = false;
    cache.sm90_plan.valid = false;
    cache.graph_capturable = false;
    if (!custom_mask && !hnd_layout && !wants_prefill_score &&
        kv_last_page_lens_h != nullptr &&
        hopper_prefill_supported(
            head_dim, window_left, total_tokens, num_requests)) {
        plan_attention_flashinfer_prefill_sm90_bf16(
            cache.sm90_plan,
            qo_indptr_h,
            kv_page_indptr_h,
            kv_last_page_lens_h,
            total_tokens,
            num_requests,
            num_q_heads,
            num_kv_heads,
            head_dim,
            page_size,
            workspace,
            stream,
            enable_cuda_graph,
            causal_mask,
            window_left);
        cache.total_tokens = total_tokens;
        cache.num_requests = num_requests;
        cache.num_q_heads = num_q_heads;
        cache.num_kv_heads = num_kv_heads;
        cache.head_dim = head_dim;
        cache.page_size = page_size;
        cache.window_left = window_left;
        cache.full_attention_variant = full_attention_variant;
        cache.causal_mask = causal_mask;
        cache.hnd_layout = hnd_layout;
        cache.enable_pdl = current_device_supports_pdl();
        cache.use_sm90 = true;
        cache.valid = true;
        return;
    }

    cache.qo_h_buf.resize(num_requests + 1);
    cache.kv_h_buf.resize(num_requests + 1);
    for (int r = 0; r <= num_requests; ++r) {
        cache.qo_h_buf[r] = static_cast<IdType>(qo_indptr_h[r]);
        cache.kv_h_buf[r] = static_cast<IdType>(kv_page_indptr_h[r]);
    }

    const bool head_dim_supports_split =
        head_dim_supports_cascade_merge(static_cast<uint32_t>(head_dim));
    const bool disable_split_kv =
        !head_dim_supports_split;

    // Graph-mode planning fixes the launch geometry as a pure function of
    // (total_tokens, num_requests) but in exchange always splits KV, so the
    // plan always carves its float partials — sized by the padded (not
    // actual) work-item count. Demote to a content-shaped plan when the
    // carve would overflow the float workspace grant: the wave then runs
    // eager (uncapturable) instead of failing the plan. Graph mode with
    // split disabled is not demote-exempt either — flashinfer only writes
    // `block_valid_mask` on the split path, so an unsplit padded grid would
    // read uninitialized work assignments.
    if (enable_cuda_graph && !disable_split_kv) {
        const std::uint64_t gqa_group =
            static_cast<std::uint64_t>(num_q_heads) /
            std::max(1, num_kv_heads);
        const std::uint64_t max_qo_len =
            static_cast<std::uint64_t>(
                std::max(1, total_tokens - num_requests + 1)) *
            std::max<std::uint64_t>(1, gqa_group);
        const std::uint64_t cta_tile_q = ::flashinfer::FA2DetermineCtaTileQ(
            static_cast<std::int64_t>(max_qo_len),
            static_cast<std::uint32_t>(head_dim));
        int num_sm = 0;
        int dev_id = 0;
        CUDA_CHECK(cudaGetDevice(&dev_id));
        CUDA_CHECK(cudaDeviceGetAttribute(
            &num_sm, cudaDevAttrMultiProcessorCount, dev_id));
        const std::uint64_t max_batch_size_if_split =
            static_cast<std::uint64_t>(2 * num_sm) /
            std::max(1, num_kv_heads);
        const std::uint64_t total_tiles =
            (static_cast<std::uint64_t>(total_tokens) *
                 std::max<std::uint64_t>(1, gqa_group) +
             cta_tile_q - 1) /
                cta_tile_q +
            static_cast<std::uint64_t>(std::max(0, num_requests - 1));
        const std::uint64_t padded_batch =
            std::max(max_batch_size_if_split, total_tiles);
        const std::uint64_t carve_bytes =
            static_cast<std::uint64_t>(num_q_heads) * padded_batch *
                cta_tile_q * (static_cast<std::uint64_t>(head_dim) + 1) *
                sizeof(float) +
            2 * 16;  // two 16-byte-aligned allocations
        if (carve_bytes > workspace.float_bytes()) {
            enable_cuda_graph = false;
        }
    } else {
        enable_cuda_graph = enable_cuda_graph && !disable_split_kv;
    }

    auto status = ::flashinfer::PrefillPlan<IdType>(
        workspace.float_buffer(), workspace.float_bytes(),
        workspace.int_buffer(), workspace.page_locked_int(),
        workspace.int_bytes(),
        cache.plan_info,
        cache.qo_h_buf.data(), cache.kv_h_buf.data(),
        static_cast<uint32_t>(total_tokens),
        static_cast<uint32_t>(num_requests),
        static_cast<uint32_t>(num_q_heads),
        static_cast<uint32_t>(num_kv_heads),
        static_cast<uint32_t>(head_dim),
        static_cast<uint32_t>(head_dim),
        static_cast<uint32_t>(page_size),
        enable_cuda_graph,
        sizeof(DTypeO),
        window_left,
        /*fixed_split_size=*/-1,
        disable_split_kv,
        /*num_colocated_ctas=*/0,
        stream);
    CUDA_CHECK(status);

    cache.total_tokens = total_tokens;
    cache.num_requests = num_requests;
    cache.num_q_heads = num_q_heads;
    cache.num_kv_heads = num_kv_heads;
    cache.head_dim = head_dim;
    cache.page_size = page_size;
    cache.window_left = window_left;
    cache.full_attention_variant = full_attention_variant;
    cache.causal_mask = causal_mask;
    cache.hnd_layout = hnd_layout;
    // Only the causal FA2 prefill dispatch is captured (Phase 1): the
    // custom-mask variant stays eager, and the decode-shaped plans are
    // admitted through the pure-decode rules instead.
    cache.graph_capturable = enable_cuda_graph && causal_mask && !custom_mask;
    cache.enable_pdl = current_device_supports_pdl();
    cache.valid = true;
}

void dispatch_attention_flashinfer_decode_bf16(
    const DecodePlanCache& cache,
    const void* q, void* k_pages, void* v_pages, void* o,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    int window_left,
    float logits_soft_cap,
    float sm_scale,
    float* lse_out,
    bool broadcast_q)
{
    if (!cache.valid) {
        throw std::runtime_error(
            "dispatch_attention_flashinfer_decode_bf16: cache is empty; "
            "call plan_attention_flashinfer_decode_bf16 first");
    }
    cudaError_t status;
    switch (cache.head_dim) {
#define PIE_ATTN_HEAD_DIM(HD)                                                \
        case HD:                                                             \
            status = AttnHd<HD>::dispatch_decode(                            \
                cache, q, k_pages, v_pages, o, kv_page_indices_d,            \
                kv_page_indptr_d, kv_last_page_lens_d, workspace, stream,    \
                window_left, logits_soft_cap, sm_scale, lse_out, broadcast_q); \
            break;
#include "kernels.def"
        default:
            throw_unsupported_head_dim("flashinfer decode dispatch", cache.head_dim);
    }
    CUDA_CHECK(status);
}

// ── Score-observing decode ─────────────────────────────────────────────────

// Turn the captured scaled logits into attention probabilities, in place.
//
// This is a plain row-wise softmax and NOT an approximation: at decode
// `qo_len == 1`, so the row the variant captured is the complete set of
// logits the kernel's own online softmax consumed. Recomputing the
// denominator here is therefore exact, and it means the decode path does not
// have to allocate or plumb an LSE buffer it otherwise never needs.
//
// `kv_len` is derived from the page CSR rather than passed in. That is
// deliberate: the CSR is the single source of truth for sequence length in
// this driver (`kernels/geometry.cu`), and a second, independently-computed
// length is exactly how a silent mis-attribution bug gets in.
__global__ void k_attn_score_normalize(
    float* __restrict__ scores,
    const std::int32_t* __restrict__ score_indptr,
    const std::uint32_t* __restrict__ kv_page_indptr,
    const std::uint32_t* __restrict__ kv_last_page_lens,
    int page_size)
{
    constexpr int kThreads = 256;
    __shared__ float shared[kThreads];

    const int request = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int pages = static_cast<int>(kv_page_indptr[request + 1]) -
                      static_cast<int>(kv_page_indptr[request]);
    if (pages <= 0) return;
    const int kv_len =
        (pages - 1) * page_size + static_cast<int>(kv_last_page_lens[request]);
    if (kv_len <= 0) return;

    float* row = scores + static_cast<std::size_t>(score_indptr[request]) +
                 static_cast<std::size_t>(head) * static_cast<std::size_t>(kv_len);

    float local = -INFINITY;
    for (int i = threadIdx.x; i < kv_len; i += kThreads) {
        local = fmaxf(local, row[i]);
    }
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            shared[threadIdx.x] =
                fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float row_max = shared[0];
    __syncthreads();

    float total = 0.f;
    for (int i = threadIdx.x; i < kv_len; i += kThreads) {
        const float e = __expf(row[i] - row_max);
        row[i] = e;
        total += e;
    }
    shared[threadIdx.x] = total;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float denom = shared[0];
    if (denom <= 0.f) return;
    const float inv = 1.f / denom;
    for (int i = threadIdx.x; i < kv_len; i += kThreads) {
        row[i] *= inv;
    }
}

// Prefill counterpart. Two things make this not just the decode kernel with an
// extra grid dimension:
//
//  1. **Every window row has a different causal support.** The hook fires
//     before the kernel's mask (`LogitsMask` runs after `LogitsTransform`), so
//     a captured row contains real dot products at positions the softmax is
//     about to discard. Normalising over the full `kv_len` would spread mass
//     onto the future. Window row `w` belongs to the query at absolute
//     position `kv_len - rows + w`, so it may attend to `kv_len - rows + w + 1`
//     keys and no more. Everything past that is zeroed here, which is also
//     what makes the folded row a distribution over the prefix.
//
//  2. **`rows` is `min(window, qo_len)`.** A prompt shorter than the
//     observation window contributes fewer rows; the rest of its slot is never
//     written by the kernel and must already be zero.
__global__ void k_attn_prefill_score_normalize(
    float* __restrict__ scores,
    const std::int32_t* __restrict__ score_indptr,
    const std::uint32_t* __restrict__ qo_indptr,
    const std::uint32_t* __restrict__ kv_page_indptr,
    const std::uint32_t* __restrict__ kv_last_page_lens,
    int page_size,
    int window)
{
    constexpr int kThreads = 256;
    __shared__ float shared[kThreads];

    const int request = static_cast<int>(blockIdx.x);
    const int head = static_cast<int>(blockIdx.y);
    const int w = static_cast<int>(blockIdx.z);

    const int pages = static_cast<int>(kv_page_indptr[request + 1]) -
                      static_cast<int>(kv_page_indptr[request]);
    if (pages <= 0) return;
    const int kv_len =
        (pages - 1) * page_size + static_cast<int>(kv_last_page_lens[request]);
    if (kv_len <= 0) return;
    const int qo_len = static_cast<int>(qo_indptr[request + 1]) -
                       static_cast<int>(qo_indptr[request]);
    const int rows = min(window, qo_len);
    if (w >= rows) return;

    const int limit = min(kv_len - rows + w + 1, kv_len);
    if (limit <= 0) return;

    float* row = scores + static_cast<std::size_t>(score_indptr[request]) +
                 (static_cast<std::size_t>(head) * static_cast<std::size_t>(window) +
                  static_cast<std::size_t>(w)) *
                     static_cast<std::size_t>(kv_len);

    float local = -INFINITY;
    for (int i = threadIdx.x; i < limit; i += kThreads) {
        local = fmaxf(local, row[i]);
    }
    shared[threadIdx.x] = local;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            shared[threadIdx.x] =
                fmaxf(shared[threadIdx.x], shared[threadIdx.x + stride]);
        }
        __syncthreads();
    }
    const float row_max = shared[0];
    __syncthreads();

    // Accumulate the denominator WITHOUT storing the exponentials. Storing them
    // and rescaling in a second pass costs one more full write and one more full
    // read of a `heads * window * kv_len` buffer, which at 8K context is 16 MB
    // per layer; recomputing `__expf` in the final pass is a handful of SFU
    // cycles against a kernel that is entirely bandwidth-bound.
    float total = 0.f;
    for (int i = threadIdx.x; i < limit; i += kThreads) {
        total += __expf(row[i] - row_max);
    }
    shared[threadIdx.x] = total;
    __syncthreads();
    for (int stride = kThreads / 2; stride > 0; stride >>= 1) {
        if (static_cast<int>(threadIdx.x) < stride) {
            shared[threadIdx.x] += shared[threadIdx.x + stride];
        }
        __syncthreads();
    }
    const float denom = shared[0];
    // `denom >= 1` in exact arithmetic -- the argmax element contributes
    // `exp(0)` -- so this is unreachable. Zeroing rather than returning is still
    // the right failure: an early return would leave raw LOGITS in a buffer the
    // fold is about to average, and negative logits would read as negative
    // attention mass.
    const float inv = denom > 0.f ? 1.f / denom : 0.f;

    // One pass over the WHOLE row: positions at or past the causal limit were
    // computed by the kernel but never attended to (`LogitsMask` runs after
    // `LogitsTransform`), so they are zeroed here rather than in a separate
    // sweep. This is what makes the folded row a distribution over the prefix.
    for (int i = threadIdx.x; i < kv_len; i += kThreads) {
        row[i] = i < limit ? __expf(row[i] - row_max) * inv : 0.f;
    }
}

// Fold `[head, window_row, kv]` down to one row per request.
//
// Averaging rather than summing, for the same reason the decode fold averages:
// every contributing row is a distribution over the prefix, so the mean is one
// too, and a policy can threshold it in absolute terms without knowing how many
// heads or window rows went into it. The divisor is `heads * rows` with
// `rows = min(window, qo_len)` -- rows that do not exist contribute nothing and
// must not be counted, or a short prompt's mass would be scaled down.
//
// The folded row lands at `score_indptr[r] / (heads * window)`, which is the
// same derivation trick the decode fold uses: the ragged offset divided by the
// per-request multiplier is exactly the folded offset.
__global__ void k_attn_prefill_score_fold(
    const float* __restrict__ scores,
    float* __restrict__ folded,
    const std::int32_t* __restrict__ score_indptr,
    const std::uint32_t* __restrict__ qo_indptr,
    const std::uint32_t* __restrict__ kv_page_indptr,
    const std::uint32_t* __restrict__ kv_last_page_lens,
    int page_size,
    int num_q_heads,
    int window)
{
    const int request = static_cast<int>(blockIdx.x);
    const int pages = static_cast<int>(kv_page_indptr[request + 1]) -
                      static_cast<int>(kv_page_indptr[request]);
    if (pages <= 0) return;
    const int kv_len =
        (pages - 1) * page_size + static_cast<int>(kv_last_page_lens[request]);
    if (kv_len <= 0) return;
    const int qo_len = static_cast<int>(qo_indptr[request + 1]) -
                       static_cast<int>(qo_indptr[request]);
    const int rows = min(window, qo_len);
    if (rows <= 0) return;

    const std::size_t base = static_cast<std::size_t>(score_indptr[request]);
    const std::size_t out_base =
        base / (static_cast<std::size_t>(num_q_heads) *
                static_cast<std::size_t>(window));
    const float inv = 1.f / static_cast<float>(num_q_heads * rows);

    for (int k = static_cast<int>(threadIdx.x) +
                 static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.x);
         k < kv_len;
         k += static_cast<int>(blockDim.x) * static_cast<int>(gridDim.y)) {
        float acc = 0.f;
        for (int h = 0; h < num_q_heads; ++h) {
            for (int w = 0; w < rows; ++w) {
                acc += scores[base +
                              (static_cast<std::size_t>(h) *
                                   static_cast<std::size_t>(window) +
                               static_cast<std::size_t>(w)) *
                                  static_cast<std::size_t>(kv_len) +
                              static_cast<std::size_t>(k)];
            }
        }
        folded[out_base + static_cast<std::size_t>(k)] = acc * inv;
    }
}

void dispatch_attention_flashinfer_decode_capture_bf16(
    const DecodePlanCache& cache,
    const void* q, void* k_pages, void* v_pages, void* o,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    float* score_out,
    const std::int32_t* score_indptr_d,
    int window_left,
    float logits_soft_cap,
    float sm_scale,
    float* lse_out)
{
    if (!cache.valid) {
        throw std::runtime_error(
            "dispatch_attention_flashinfer_decode_capture_bf16: cache is "
            "empty; call plan_attention_flashinfer_decode_bf16 first");
    }
    if (score_out == nullptr || score_indptr_d == nullptr) {
        throw std::invalid_argument(
            "dispatch_attention_flashinfer_decode_capture_bf16: score sink is "
            "null");
    }
    if (logits_soft_cap > 0.f) {
        throw std::invalid_argument(
            "attention score capture does not support logits_soft_cap: the "
            "hook would record cap*tanh(s/cap), which is not the pre-softmax "
            "score H2O/TOVA define their eviction policy over");
    }
    if (window_left >= 0) {
        throw std::invalid_argument(
            "attention score capture does not support sliding-window "
            "attention: LogitsMask runs after LogitsTransform, so the "
            "captured row would include positions the softmax discards");
    }

    const fa2::DecodeScoreSink sink{
        score_out, reinterpret_cast<const fa2::IdType*>(score_indptr_d)};

    cudaError_t status;
    switch (cache.head_dim) {
#define PIE_ATTN_HEAD_DIM(HD)                                                \
        case HD:                                                             \
            status = AttnHd<HD>::dispatch_decode_capture(                    \
                cache, q, k_pages, v_pages, o, kv_page_indices_d,            \
                kv_page_indptr_d, kv_last_page_lens_d, workspace, stream,    \
                window_left, logits_soft_cap, sm_scale, lse_out, sink);      \
            break;
#include "kernels.def"
        default:
            throw_unsupported_head_dim(
                "flashinfer decode capture dispatch", cache.head_dim);
    }
    CUDA_CHECK(status);

    const dim3 grid(static_cast<unsigned>(cache.num_requests),
                    static_cast<unsigned>(cache.num_q_heads));
    k_attn_score_normalize<<<grid, 256, 0, stream>>>(
        score_out, score_indptr_d, kv_page_indptr_d, kv_last_page_lens_d,
        cache.page_size);
    CUDA_CHECK(cudaGetLastError());
}

// Fold the per-head probability rows into one row per request.
//
// Eviction here is necessarily a per-REQUEST decision: the paged KV layout
// carries a single page list per request, so a per-head keep-set has nowhere to
// live. Quest already makes (and documents) the same collapse. Averaging rather
// than summing keeps the folded row a probability distribution -- it sums to 1
// over the live prefix -- so a policy can threshold it in absolute terms.
//
// The folded CSR is not a second array: `score_indptr[r]` counts
// `num_q_heads * kv_len(r')` elements for every earlier request, so dividing it
// by `num_q_heads` is exactly the folded offset. Deriving it removes the chance
// of two CSRs disagreeing.
__global__ void k_attn_score_fold_heads(
    const float* __restrict__ scores,
    const std::int32_t* __restrict__ score_indptr,
    const std::uint32_t* __restrict__ kv_page_indptr,
    const std::uint32_t* __restrict__ kv_last_page_lens,
    int page_size,
    int num_q_heads,
    float* __restrict__ folded)
{
    const int request = static_cast<int>(blockIdx.x);
    const int pages = static_cast<int>(kv_page_indptr[request + 1]) -
                      static_cast<int>(kv_page_indptr[request]);
    if (pages <= 0 || num_q_heads <= 0) return;
    const int kv_len =
        (pages - 1) * page_size + static_cast<int>(kv_last_page_lens[request]);
    if (kv_len <= 0) return;

    const std::size_t base = static_cast<std::size_t>(score_indptr[request]);
    const float* rows = scores + base;
    float* out = folded + base / static_cast<std::size_t>(num_q_heads);
    const float inv_heads = 1.f / static_cast<float>(num_q_heads);

    for (int i = static_cast<int>(threadIdx.x) +
                 static_cast<int>(blockIdx.y) * static_cast<int>(blockDim.x);
         i < kv_len;
         i += static_cast<int>(blockDim.x) * static_cast<int>(gridDim.y)) {
        float total = 0.f;
        for (int h = 0; h < num_q_heads; ++h) {
            total += rows[static_cast<std::size_t>(h) *
                              static_cast<std::size_t>(kv_len) +
                          static_cast<std::size_t>(i)];
        }
        out[i] = total * inv_heads;
    }
}

void launch_attn_score_fold_heads(
    const float* scores,
    const std::int32_t* score_indptr_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    int page_size,
    int num_requests,
    int num_q_heads,
    float* folded,
    cudaStream_t stream)
{
    if (num_requests <= 0) return;
    if (scores == nullptr || folded == nullptr || score_indptr_d == nullptr) {
        throw std::invalid_argument(
            "launch_attn_score_fold_heads: null score buffer");
    }
    const dim3 grid(static_cast<unsigned>(num_requests), 64u);
    k_attn_score_fold_heads<<<grid, 256, 0, stream>>>(
        scores, score_indptr_d, kv_page_indptr_d, kv_last_page_lens_d,
        page_size, num_q_heads, folded);
    CUDA_CHECK(cudaGetLastError());
}

void dispatch_attention_flashinfer_decode_capture(
    const DecodePlanCache& cache,
    const void* q,
    KvCacheLayerView kv_layer,
    void* o,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    float* score_out,
    const std::int32_t* score_indptr_d,
    int window_left,
    float logits_soft_cap,
    float sm_scale,
    float* lse_out)
{
    kernels::launch_dequant_kv_cache_layer_to_bf16_active(
        kv_layer, kv_page_indices_d, cache.num_pages_in_batch, stream);
    dispatch_attention_flashinfer_decode_capture_bf16(
        cache, q,
        kv_layer.k_bf16_pages,
        kv_layer.v_bf16_pages,
        o,
        kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d,
        workspace, stream, score_out, score_indptr_d,
        window_left, logits_soft_cap, sm_scale, lse_out);
}

void dispatch_attention_flashinfer_decode(
    const DecodePlanCache& cache,
    const void* q,
    KvCacheLayerView kv_layer,
    void* o,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    int window_left,
    float logits_soft_cap,
    float sm_scale,
    float* lse_out)
{
    kernels::launch_dequant_kv_cache_layer_to_bf16_active(
        kv_layer, kv_page_indices_d, cache.num_pages_in_batch, stream);
    dispatch_attention_flashinfer_decode_bf16(
        cache, q,
        kv_layer.k_bf16_pages,
        kv_layer.v_bf16_pages,
        o,
        kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d,
        workspace, stream, window_left, logits_soft_cap, sm_scale, lse_out);
}

// ── Prefill ────────────────────────────────────────────────────────────────

// The FA2 prefill params block, shared by the plain and score-capturing
// dispatches. Factored out because the two must agree exactly: if the capture
// path built its params even slightly differently it would compute a different
// attention, and the whole premise of an observation hook is that it observes
// the attention that actually ran.
static PrefillParams make_prefill_params(
    const PrefillPlanCache& cache,
    const void* q, void* k_pages, void* v_pages, void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    AttentionWorkspace& workspace,
    float logits_soft_cap,
    float sm_scale,
    float* lse_out,
    DTypeO*& tmp_v,
    float*& tmp_s)
{
    ::flashinfer::paged_kv_t<DTypeKV, IdType> paged_kv(
        static_cast<uint32_t>(cache.num_kv_heads),
        static_cast<uint32_t>(cache.page_size),
        static_cast<uint32_t>(cache.head_dim),
        static_cast<uint32_t>(cache.num_requests),
        kv_layout(cache.hnd_layout),
        static_cast<DTypeKV*>(k_pages),
        static_cast<DTypeKV*>(v_pages),
        const_cast<IdType*>(reinterpret_cast<const IdType*>(kv_page_indices_d)),
        const_cast<IdType*>(reinterpret_cast<const IdType*>(kv_page_indptr_d)),
        const_cast<IdType*>(reinterpret_cast<const IdType*>(kv_last_page_lens_d)));

    PrefillParams params;
    params.q = const_cast<DTypeQ*>(static_cast<const DTypeQ*>(q));
    params.paged_kv = paged_kv;
    params.maybe_custom_mask = nullptr;
    params.q_indptr = const_cast<IdType*>(reinterpret_cast<const IdType*>(qo_indptr_d));
    params.maybe_mask_indptr = nullptr;
    params.maybe_q_rope_offset = nullptr;
    params.o = static_cast<DTypeO*>(o);
    params.lse = lse_out;
    params.maybe_alibi_slopes = nullptr;
    params.group_size = ::flashinfer::uint_fastdiv(
        static_cast<uint32_t>(cache.num_q_heads / cache.num_kv_heads));
    params.num_qo_heads = static_cast<uint32_t>(cache.num_q_heads);
    params.q_stride_n = static_cast<IdType>(cache.num_q_heads * cache.head_dim);
    params.q_stride_h = static_cast<IdType>(cache.head_dim);
    params.window_left = cache.window_left;
    params.logits_soft_cap = logits_soft_cap;
    params.sm_scale = (sm_scale > 0.f)
        ? sm_scale
        : (1.0f / std::sqrt(static_cast<float>(cache.head_dim)));
    params.rope_rcp_scale = 1.0f;
    params.rope_rcp_theta = 1.0f;

    void* int_buf = workspace.int_buffer();
    void* float_buf = workspace.float_buffer();
    const auto& plan_info = cache.plan_info;
    params.request_indices = offset_ptr<IdType>(int_buf, plan_info.request_indices_offset);
    params.qo_tile_indices = offset_ptr<IdType>(int_buf, plan_info.qo_tile_indices_offset);
    params.kv_tile_indices = offset_ptr<IdType>(int_buf, plan_info.kv_tile_indices_offset);
    params.o_indptr = offset_ptr<IdType>(int_buf, plan_info.o_indptr_offset);
    params.kv_chunk_size_ptr = offset_ptr<IdType>(int_buf, plan_info.kv_chunk_size_ptr_offset);
    params.padded_batch_size = static_cast<uint32_t>(plan_info.padded_batch_size);
    params.partition_kv = plan_info.split_kv;
    params.max_total_num_rows = static_cast<uint32_t>(plan_info.total_num_rows);
    params.merge_indptr = nullptr;
    params.block_valid_mask = nullptr;
    params.total_num_rows = nullptr;
    params.maybe_prefix_len_ptr = nullptr;
    params.maybe_token_pos_in_items_ptr = nullptr;
    params.token_pos_in_items_len = 0;
    params.maybe_max_item_len_ptr = nullptr;

    tmp_v = nullptr;
    tmp_s = nullptr;
    if (plan_info.split_kv) {
        params.merge_indptr = offset_ptr<IdType>(int_buf, plan_info.merge_indptr_offset);
        tmp_v = offset_ptr<DTypeO>(float_buf, plan_info.v_offset);
        tmp_s = offset_ptr<float>(float_buf, plan_info.s_offset);
        if (plan_info.enable_cuda_graph) {
            params.block_valid_mask =
                offset_ptr<bool>(int_buf, plan_info.block_valid_mask_offset);
        }
    }

    return params;
}

void dispatch_attention_flashinfer_prefill_bf16(
    const PrefillPlanCache& cache,
    const void* q,
    void* k_pages, void* v_pages, void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    float logits_soft_cap,
    float sm_scale,
    float* lse_out)
{
    if (!cache.valid) {
        throw std::runtime_error(
            "dispatch_attention_flashinfer_prefill_bf16: cache is empty; "
            "call plan_attention_flashinfer_prefill_bf16 first");
    }
    if (cache.use_sm90) {
        dispatch_attention_flashinfer_prefill_sm90_bf16(
            cache.sm90_plan,
            q,
            k_pages,
            v_pages,
            o,
            kv_page_indices_d,
            workspace,
            stream,
            logits_soft_cap,
            sm_scale,
            lse_out);
        return;
    }

    DTypeO* tmp_v = nullptr;
    float* tmp_s = nullptr;
    const auto& plan_info = cache.plan_info;
    PrefillParams params = make_prefill_params(
        cache, q, k_pages, v_pages, o, qo_indptr_d, kv_page_indices_d,
        kv_page_indptr_d, kv_last_page_lens_d, workspace, logits_soft_cap,
        sm_scale, lse_out, tmp_v, tmp_s);

    // Mask mode and attention variant are runtime-policy axes, so they stay
    // inside AttnHd; only head_dim selects a translation unit.
    cudaError_t status;
    switch (cache.head_dim) {
#define PIE_ATTN_HEAD_DIM(HD)                                                \
        case HD:                                                             \
            status = AttnHd<HD>::prefill(                                    \
                params, plan_info, tmp_v, tmp_s, cache.enable_pdl, stream,   \
                cache.full_attention_variant, cache.causal_mask,             \
                logits_soft_cap);                                            \
            break;
#include "kernels.def"
        default:
            throw_unsupported_head_dim("flashinfer prefill dispatch", cache.head_dim);
    }
    CUDA_CHECK(status);
}

void dispatch_attention_flashinfer_prefill_capture_bf16(
    const PrefillPlanCache& cache,
    const void* q,
    void* k_pages, void* v_pages, void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    float* score_out,
    float* folded_out,
    const std::int32_t* score_indptr_d,
    int window,
    float logits_soft_cap,
    float sm_scale,
    float* lse_out)
{
    if (!cache.valid) {
        throw std::runtime_error(
            "dispatch_attention_flashinfer_prefill_capture_bf16: cache is "
            "empty; call plan_attention_flashinfer_prefill_bf16 first");
    }
    if (score_out == nullptr || score_indptr_d == nullptr ||
        folded_out == nullptr) {
        throw std::invalid_argument(
            "dispatch_attention_flashinfer_prefill_capture_bf16: score sink is "
            "null");
    }
    if (window <= 0) {
        throw std::invalid_argument(
            "prefill score capture needs a positive observation window");
    }
    if (cache.use_sm90) {
        // The Hopper kernel takes a different variant API -- a constructor over
        // a block coordinate rather than (params, batch_idx, smem), and no
        // inherited qo_len -- so it needs its own capture struct. Refusing here
        // is the honest behaviour: a silent fallthrough would run attention
        // with no capture and hand the policy an all-zero row, which reads as
        // "nothing was attended to" and evicts the whole prefix.
        throw std::runtime_error(
            "prefill score capture is not implemented for the SM90 kernel; "
            "plan without it (the planner honours wants_prefill_score)");
    }
    if (logits_soft_cap > 0.f) {
        throw std::invalid_argument(
            "prefill score capture does not support logits_soft_cap: the hook "
            "would record cap*tanh(s/cap), which is not the pre-softmax score "
            "SnapKV defines its selection over");
    }
    if (cache.window_left >= 0) {
        throw std::invalid_argument(
            "prefill score capture does not support sliding-window attention: "
            "LogitsMask runs after LogitsTransform, so the captured row would "
            "include positions the softmax discards");
    }
    if (!cache.full_attention_variant) {
        throw std::invalid_argument(
            "prefill score capture requires the full-attention variant");
    }

    DTypeO* tmp_v = nullptr;
    float* tmp_s = nullptr;
    const auto& plan_info = cache.plan_info;
    PrefillParams params = make_prefill_params(
        cache, q, k_pages, v_pages, o, qo_indptr_d, kv_page_indices_d,
        kv_page_indptr_d, kv_last_page_lens_d, workspace, logits_soft_cap,
        sm_scale, lse_out, tmp_v, tmp_s);

    const fa2::PrefillScoreSink sink{
        score_out, reinterpret_cast<const fa2::IdType*>(score_indptr_d),
        static_cast<std::uint32_t>(window)};

    cudaError_t status;
    switch (cache.head_dim) {
#define PIE_ATTN_HEAD_DIM(HD)                                                \
        case HD:                                                             \
            status = AttnHd<HD>::prefill_capture(                            \
                params, plan_info, tmp_v, tmp_s, cache.enable_pdl, stream,   \
                cache.causal_mask, logits_soft_cap, cache.window_left,       \
                sink);                                                       \
            break;
#include "kernels.def"
        default:
            throw_unsupported_head_dim(
                "flashinfer prefill capture dispatch", cache.head_dim);
    }
    CUDA_CHECK(status);

    const dim3 norm_grid(static_cast<unsigned>(cache.num_requests),
                         static_cast<unsigned>(cache.num_q_heads),
                         static_cast<unsigned>(window));
    k_attn_prefill_score_normalize<<<norm_grid, 256, 0, stream>>>(
        score_out, score_indptr_d, qo_indptr_d, kv_page_indptr_d,
        kv_last_page_lens_d, cache.page_size, window);
    CUDA_CHECK(cudaGetLastError());

    const dim3 fold_grid(static_cast<unsigned>(cache.num_requests), 32u);
    k_attn_prefill_score_fold<<<fold_grid, 256, 0, stream>>>(
        score_out, folded_out, score_indptr_d, qo_indptr_d, kv_page_indptr_d,
        kv_last_page_lens_d, cache.page_size, cache.num_q_heads, window);
    CUDA_CHECK(cudaGetLastError());
}

void launch_attention_flashinfer_prefill_bf16(
    const void* q, void* k_pages, void* v_pages, void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    int total_tokens,
    int num_requests,
    int num_q_heads, int num_kv_heads, int head_dim, int page_size,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    int window_left,
    float logits_soft_cap,
    float sm_scale,
    float* lse_out,
    bool hnd_layout)
{
    if (!attn_head_dim_instantiated(head_dim)) {
        throw_unsupported_head_dim("flashinfer prefill", head_dim);
    }

    // 1. paged_kv_t — same construction as decode.
    ::flashinfer::paged_kv_t<DTypeKV, IdType> paged_kv(
        static_cast<uint32_t>(num_kv_heads),
        static_cast<uint32_t>(page_size),
        static_cast<uint32_t>(head_dim),
        static_cast<uint32_t>(num_requests),
        kv_layout(hnd_layout),
        static_cast<DTypeKV*>(k_pages),
        static_cast<DTypeKV*>(v_pages),
        const_cast<IdType*>(reinterpret_cast<const IdType*>(kv_page_indices_d)),
        const_cast<IdType*>(reinterpret_cast<const IdType*>(kv_page_indptr_d)),
        const_cast<IdType*>(reinterpret_cast<const IdType*>(kv_last_page_lens_d)));

    // 2. Plan.
    ::flashinfer::PrefillPlanInfo plan_info;
    std::vector<IdType> qo_h(num_requests + 1);
    std::vector<IdType> kv_h(num_requests + 1);
    for (int r = 0; r <= num_requests; ++r) {
        qo_h[r] = static_cast<IdType>(qo_indptr_h[r]);
        kv_h[r] = static_cast<IdType>(kv_page_indptr_h[r]);
    }

    // The cascade-merge kernel `VariableLengthMergeStates` only instantiates
    // head_dim ∈ {64, 128, 256, 512}. For other head dims (e.g. Phi-3-mini at
    // 96), force the prefill to a single-CTA-per-request schedule by disabling
    // split-KV — that path skips the cascade merge entirely.
    const bool head_dim_supports_split =
        head_dim_supports_cascade_merge(static_cast<uint32_t>(head_dim));

    auto status = ::flashinfer::PrefillPlan<IdType>(
        workspace.float_buffer(), workspace.float_bytes(),
        workspace.int_buffer(), workspace.page_locked_int(),
        workspace.int_bytes(),
        plan_info,
        qo_h.data(), kv_h.data(),
        /*total_num_rows=*/static_cast<uint32_t>(total_tokens),
        /*batch_size=*/static_cast<uint32_t>(num_requests),
        /*num_qo_heads=*/static_cast<uint32_t>(num_q_heads),
        /*num_kv_heads=*/static_cast<uint32_t>(num_kv_heads),
        /*head_dim_qk=*/static_cast<uint32_t>(head_dim),
        /*head_dim_vo=*/static_cast<uint32_t>(head_dim),
        /*page_size=*/static_cast<uint32_t>(page_size),
        /*enable_cuda_graph=*/false,
        /*sizeof_dtype_o=*/sizeof(DTypeO),
        /*window_left=*/window_left,
        /*fixed_split_size=*/-1,
        /*disable_split_kv=*/!head_dim_supports_split,
        /*num_colocated_ctas=*/0,
        stream);
    CUDA_CHECK(status);

    // 3. Build params.
    PrefillParams params;
    params.q = const_cast<DTypeQ*>(static_cast<const DTypeQ*>(q));
    params.paged_kv = paged_kv;
    params.maybe_custom_mask = nullptr;
    params.q_indptr = const_cast<IdType*>(reinterpret_cast<const IdType*>(qo_indptr_d));
    params.maybe_mask_indptr = nullptr;
    params.maybe_q_rope_offset = nullptr;
    params.o = static_cast<DTypeO*>(o);
    params.lse = lse_out;
    params.maybe_alibi_slopes = nullptr;
    params.group_size = ::flashinfer::uint_fastdiv(
        static_cast<uint32_t>(num_q_heads / num_kv_heads));
    params.num_qo_heads = static_cast<uint32_t>(num_q_heads);
    params.q_stride_n = static_cast<IdType>(num_q_heads * head_dim);
    params.q_stride_h = static_cast<IdType>(head_dim);
    params.window_left = window_left;
    params.logits_soft_cap = logits_soft_cap;
    params.sm_scale = (sm_scale > 0.f)
        ? sm_scale
        : (1.0f / std::sqrt(static_cast<float>(head_dim)));
    params.rope_rcp_scale = 1.0f;
    params.rope_rcp_theta = 1.0f;

    void* int_buf   = workspace.int_buffer();
    void* float_buf = workspace.float_buffer();
    params.request_indices   = offset_ptr<IdType>(int_buf, plan_info.request_indices_offset);
    params.qo_tile_indices   = offset_ptr<IdType>(int_buf, plan_info.qo_tile_indices_offset);
    params.kv_tile_indices   = offset_ptr<IdType>(int_buf, plan_info.kv_tile_indices_offset);
    params.o_indptr          = offset_ptr<IdType>(int_buf, plan_info.o_indptr_offset);
    params.kv_chunk_size_ptr = offset_ptr<IdType>(int_buf, plan_info.kv_chunk_size_ptr_offset);
    params.padded_batch_size = static_cast<uint32_t>(plan_info.padded_batch_size);
    params.partition_kv      = plan_info.split_kv;
    params.max_total_num_rows = static_cast<uint32_t>(plan_info.total_num_rows);
    params.merge_indptr      = nullptr;
    params.block_valid_mask  = nullptr;
    params.total_num_rows    = nullptr;
    params.maybe_prefix_len_ptr = nullptr;
    params.maybe_token_pos_in_items_ptr = nullptr;
    params.token_pos_in_items_len = 0;
    params.maybe_max_item_len_ptr = nullptr;

    DTypeO* tmp_v = nullptr;
    float*  tmp_s = nullptr;
    if (plan_info.split_kv) {
        params.merge_indptr = offset_ptr<IdType>(int_buf, plan_info.merge_indptr_offset);
        tmp_v = offset_ptr<DTypeO>(float_buf, plan_info.v_offset);
        tmp_s = offset_ptr<float>(float_buf, plan_info.s_offset);
    }

    // 4. Dispatch on head_dim; AttnHd::prefill picks the soft-cap variant.
    //    This path is always causal and never the full-attention variant.
    switch (head_dim) {
#define PIE_ATTN_HEAD_DIM(HD)                                                \
        case HD:                                                             \
            status = AttnHd<HD>::prefill(                                    \
                params, plan_info, tmp_v, tmp_s,                             \
                current_device_supports_pdl(), stream,                       \
                /*full_attention_variant=*/false, /*causal_mask=*/true,      \
                logits_soft_cap);                                            \
            break;
#include "kernels.def"
        default:
            throw_unsupported_head_dim("flashinfer prefill", head_dim);
    }
    CUDA_CHECK(status);
}

void launch_attention_flashinfer_prefill(
    const void* q,
    KvCacheLayerView kv_layer,
    void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    int total_tokens,
    int num_requests,
    int num_q_heads,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    int window_left,
    float logits_soft_cap,
    float sm_scale,
    float* lse_out)
{
    const int num_pages_in_batch = kv_page_indptr_h[num_requests];
    kernels::launch_dequant_kv_cache_layer_to_bf16_active(
        kv_layer, kv_page_indices_d, num_pages_in_batch, stream);
    launch_attention_flashinfer_prefill_bf16(
        q,
        kv_layer.k_bf16_pages,
        kv_layer.v_bf16_pages,
        o,
        qo_indptr_d, kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d,
        qo_indptr_h, kv_page_indptr_h,
        total_tokens, num_requests, num_q_heads, kv_layer.num_kv_heads,
        kv_layer.head_dim, kv_layer.page_size, workspace, stream,
        window_left, logits_soft_cap, sm_scale,
        lse_out, kv_layer.hnd_layout);
}

// ── Prefill with custom mask ───────────────────────────────────────────────

void dispatch_attention_flashinfer_prefill_custom_bf16(
    const PrefillPlanCache& cache,
    const void* q, void* k_pages, void* v_pages, void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    const std::uint8_t* mask_d,
    const std::int32_t* mask_indptr_d,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    float logits_soft_cap,
    float sm_scale,
    float* lse_out)
{
    if (!cache.valid || cache.use_sm90) {
        throw std::runtime_error(
            "custom prefill dispatch requires a prepared non-SM90 plan");
    }
    ::flashinfer::paged_kv_t<DTypeKV, IdType> paged_kv(
        static_cast<uint32_t>(cache.num_kv_heads),
        static_cast<uint32_t>(cache.page_size),
        static_cast<uint32_t>(cache.head_dim),
        static_cast<uint32_t>(cache.num_requests),
        kv_layout(cache.hnd_layout),
        static_cast<DTypeKV*>(k_pages),
        static_cast<DTypeKV*>(v_pages),
        const_cast<IdType*>(reinterpret_cast<const IdType*>(kv_page_indices_d)),
        const_cast<IdType*>(reinterpret_cast<const IdType*>(kv_page_indptr_d)),
        const_cast<IdType*>(
            reinterpret_cast<const IdType*>(kv_last_page_lens_d)));

    PrefillParams params;
    params.q = const_cast<DTypeQ*>(static_cast<const DTypeQ*>(q));
    params.paged_kv = paged_kv;
    params.maybe_custom_mask = const_cast<std::uint8_t*>(mask_d);
    params.q_indptr =
        const_cast<IdType*>(reinterpret_cast<const IdType*>(qo_indptr_d));
    params.maybe_mask_indptr = const_cast<IdType*>(mask_indptr_d);
    params.maybe_q_rope_offset = nullptr;
    params.o = static_cast<DTypeO*>(o);
    params.lse = lse_out;
    params.maybe_alibi_slopes = nullptr;
    params.group_size = ::flashinfer::uint_fastdiv(
        static_cast<uint32_t>(cache.num_q_heads / cache.num_kv_heads));
    params.num_qo_heads = static_cast<uint32_t>(cache.num_q_heads);
    params.q_stride_n =
        static_cast<IdType>(cache.num_q_heads * cache.head_dim);
    params.q_stride_h = static_cast<IdType>(cache.head_dim);
    params.window_left = -1;
    params.logits_soft_cap = logits_soft_cap;
    params.sm_scale = sm_scale > 0.f
        ? sm_scale
        : 1.0f / std::sqrt(static_cast<float>(cache.head_dim));
    params.rope_rcp_scale = 1.0f;
    params.rope_rcp_theta = 1.0f;

    void* int_buf = workspace.int_buffer();
    void* float_buf = workspace.float_buffer();
    const auto& plan_info = cache.plan_info;
    params.request_indices =
        offset_ptr<IdType>(int_buf, plan_info.request_indices_offset);
    params.qo_tile_indices =
        offset_ptr<IdType>(int_buf, plan_info.qo_tile_indices_offset);
    params.kv_tile_indices =
        offset_ptr<IdType>(int_buf, plan_info.kv_tile_indices_offset);
    params.o_indptr = offset_ptr<IdType>(int_buf, plan_info.o_indptr_offset);
    params.kv_chunk_size_ptr =
        offset_ptr<IdType>(int_buf, plan_info.kv_chunk_size_ptr_offset);
    params.padded_batch_size =
        static_cast<uint32_t>(plan_info.padded_batch_size);
    params.partition_kv = plan_info.split_kv;
    params.max_total_num_rows =
        static_cast<uint32_t>(plan_info.total_num_rows);
    params.merge_indptr = nullptr;
    params.block_valid_mask = nullptr;
    params.total_num_rows = nullptr;
    params.maybe_prefix_len_ptr = nullptr;
    params.maybe_token_pos_in_items_ptr = nullptr;
    params.token_pos_in_items_len = 0;
    params.maybe_max_item_len_ptr = nullptr;

    DTypeO* tmp_v = nullptr;
    float* tmp_s = nullptr;
    if (plan_info.split_kv) {
        params.merge_indptr =
            offset_ptr<IdType>(int_buf, plan_info.merge_indptr_offset);
        tmp_v = offset_ptr<DTypeO>(float_buf, plan_info.v_offset);
        tmp_s = offset_ptr<float>(float_buf, plan_info.s_offset);
        if (plan_info.enable_cuda_graph) {
            params.block_valid_mask =
                offset_ptr<bool>(int_buf, plan_info.block_valid_mask_offset);
        }
    }

    cudaError_t status;
    switch (cache.head_dim) {
#define PIE_ATTN_HEAD_DIM(HD)                                                \
        case HD:                                                             \
            status = AttnHd<HD>::prefill_custom(                             \
                params, plan_info, tmp_v, tmp_s, cache.enable_pdl, stream,   \
                logits_soft_cap);                                            \
            break;
#include "kernels.def"
        default:
            throw_unsupported_head_dim("flashinfer custom prefill dispatch", cache.head_dim);
    }
    CUDA_CHECK(status);
}

void dispatch_attention_flashinfer_prefill_custom(
    const PrefillPlanCache& cache,
    const void* q,
    KvCacheLayerView kv_layer,
    void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    const std::uint8_t* mask_d,
    const std::int32_t* mask_indptr_d,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    float logits_soft_cap,
    float sm_scale,
    float* lse_out)
{
    const int num_pages_in_batch =
        cache.num_requests > 0 ? cache.kv_h_buf[cache.num_requests] : 0;
    kernels::launch_dequant_kv_cache_layer_to_bf16_active(
        kv_layer, kv_page_indices_d, num_pages_in_batch, stream);
    dispatch_attention_flashinfer_prefill_custom_bf16(
        cache, q, kv_layer.k_bf16_pages, kv_layer.v_bf16_pages, o,
        qo_indptr_d, kv_page_indices_d, kv_page_indptr_d,
        kv_last_page_lens_d, mask_d, mask_indptr_d, workspace, stream,
        logits_soft_cap, sm_scale, lse_out);
}

void launch_attention_flashinfer_prefill_custom_bf16(
    const void* q, void* k_pages, void* v_pages, void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    const std::uint8_t*  mask_d,
    const std::int32_t*  mask_indptr_d,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    int total_tokens,
    int num_requests,
    int num_q_heads, int num_kv_heads, int head_dim, int page_size,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    int /* window_left */,  // ignored — kCustom owns the mask
    float logits_soft_cap,
    float sm_scale,
    float* lse_out,
    bool hnd_layout)
{
    if (!attn_head_dim_instantiated(head_dim)) {
        throw_unsupported_head_dim("flashinfer prefill (custom mask)", head_dim);
    }

    // 1. paged_kv_t (same as kCausal path).
    ::flashinfer::paged_kv_t<DTypeKV, IdType> paged_kv(
        static_cast<uint32_t>(num_kv_heads),
        static_cast<uint32_t>(page_size),
        static_cast<uint32_t>(head_dim),
        static_cast<uint32_t>(num_requests),
        kv_layout(hnd_layout),
        static_cast<DTypeKV*>(k_pages),
        static_cast<DTypeKV*>(v_pages),
        const_cast<IdType*>(reinterpret_cast<const IdType*>(kv_page_indices_d)),
        const_cast<IdType*>(reinterpret_cast<const IdType*>(kv_page_indptr_d)),
        const_cast<IdType*>(reinterpret_cast<const IdType*>(kv_last_page_lens_d)));

    // 2. Plan (same as kCausal — the planner doesn't care about mask mode).
    ::flashinfer::PrefillPlanInfo plan_info;
    std::vector<IdType> qo_h(num_requests + 1);
    std::vector<IdType> kv_h(num_requests + 1);
    for (int r = 0; r <= num_requests; ++r) {
        qo_h[r] = static_cast<IdType>(qo_indptr_h[r]);
        kv_h[r] = static_cast<IdType>(kv_page_indptr_h[r]);
    }

    // See note above re: head_dims unsupported by `VariableLengthMergeStates`.
    const bool head_dim_supports_split =
        head_dim_supports_cascade_merge(static_cast<uint32_t>(head_dim));

    auto status = ::flashinfer::PrefillPlan<IdType>(
        workspace.float_buffer(), workspace.float_bytes(),
        workspace.int_buffer(), workspace.page_locked_int(),
        workspace.int_bytes(),
        plan_info,
        qo_h.data(), kv_h.data(),
        static_cast<uint32_t>(total_tokens),
        static_cast<uint32_t>(num_requests),
        static_cast<uint32_t>(num_q_heads),
        static_cast<uint32_t>(num_kv_heads),
        static_cast<uint32_t>(head_dim), static_cast<uint32_t>(head_dim),
        static_cast<uint32_t>(page_size),
        /*enable_cuda_graph=*/false,
        sizeof(DTypeO),
        /*window_left=*/-1,
        /*fixed_split_size=*/-1,
        /*disable_split_kv=*/!head_dim_supports_split,
        /*num_colocated_ctas=*/0,
        stream);
    CUDA_CHECK(status);

    // 3. Build params, including custom mask pointers.
    PrefillParams params;
    params.q = const_cast<DTypeQ*>(static_cast<const DTypeQ*>(q));
    params.paged_kv = paged_kv;
    params.maybe_custom_mask = const_cast<std::uint8_t*>(mask_d);
    params.q_indptr = const_cast<IdType*>(reinterpret_cast<const IdType*>(qo_indptr_d));
    params.maybe_mask_indptr = const_cast<IdType*>(mask_indptr_d);
    params.maybe_q_rope_offset = nullptr;
    params.o = static_cast<DTypeO*>(o);
    params.lse = lse_out;
    params.maybe_alibi_slopes = nullptr;
    params.group_size = ::flashinfer::uint_fastdiv(
        static_cast<uint32_t>(num_q_heads / num_kv_heads));
    params.num_qo_heads = static_cast<uint32_t>(num_q_heads);
    params.q_stride_n = static_cast<IdType>(num_q_heads * head_dim);
    params.q_stride_h = static_cast<IdType>(head_dim);
    params.window_left = -1;  // kCustom — caller-supplied bitmap is the source of truth
    params.logits_soft_cap = logits_soft_cap;
    params.sm_scale = (sm_scale > 0.f)
        ? sm_scale
        : 1.0f / std::sqrt(static_cast<float>(head_dim));
    params.rope_rcp_scale = 1.0f;
    params.rope_rcp_theta = 1.0f;

    void* int_buf   = workspace.int_buffer();
    void* float_buf = workspace.float_buffer();
    params.request_indices   = offset_ptr<IdType>(int_buf, plan_info.request_indices_offset);
    params.qo_tile_indices   = offset_ptr<IdType>(int_buf, plan_info.qo_tile_indices_offset);
    params.kv_tile_indices   = offset_ptr<IdType>(int_buf, plan_info.kv_tile_indices_offset);
    params.o_indptr          = offset_ptr<IdType>(int_buf, plan_info.o_indptr_offset);
    params.kv_chunk_size_ptr = offset_ptr<IdType>(int_buf, plan_info.kv_chunk_size_ptr_offset);
    params.padded_batch_size = static_cast<uint32_t>(plan_info.padded_batch_size);
    params.partition_kv      = plan_info.split_kv;
    params.max_total_num_rows = static_cast<uint32_t>(plan_info.total_num_rows);
    params.merge_indptr      = nullptr;
    params.block_valid_mask  = nullptr;
    params.total_num_rows    = nullptr;
    params.maybe_prefix_len_ptr = nullptr;
    params.maybe_token_pos_in_items_ptr = nullptr;
    params.token_pos_in_items_len = 0;
    params.maybe_max_item_len_ptr = nullptr;

    DTypeO* tmp_v = nullptr;
    float*  tmp_s = nullptr;
    if (plan_info.split_kv) {
        params.merge_indptr = offset_ptr<IdType>(int_buf, plan_info.merge_indptr_offset);
        tmp_v = offset_ptr<DTypeO>(float_buf, plan_info.v_offset);
        tmp_s = offset_ptr<float>(float_buf, plan_info.s_offset);
    }

    // 4. Dispatch on head_dim; AttnHd::prefill_custom picks the soft-cap
    //    variant, mirroring the kCausal path.
    switch (head_dim) {
#define PIE_ATTN_HEAD_DIM(HD)                                                \
        case HD:                                                             \
            status = AttnHd<HD>::prefill_custom(                             \
                params, plan_info, tmp_v, tmp_s,                             \
                current_device_supports_pdl(), stream, logits_soft_cap);     \
            break;
#include "kernels.def"
        default:
            throw_unsupported_head_dim("flashinfer prefill (custom mask)", head_dim);
    }
    CUDA_CHECK(status);
}

void launch_attention_flashinfer_prefill_custom(
    const void* q,
    KvCacheLayerView kv_layer,
    void* o,
    const std::uint32_t* qo_indptr_d,
    const std::uint32_t* kv_page_indices_d,
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    const std::uint8_t*  mask_d,
    const std::int32_t*  mask_indptr_d,
    const std::uint32_t* qo_indptr_h,
    const std::uint32_t* kv_page_indptr_h,
    int total_tokens,
    int num_requests,
    int num_q_heads,
    AttentionWorkspace& workspace,
    cudaStream_t stream,
    int window_left,
    float logits_soft_cap,
    float sm_scale,
    float* lse_out)
{
    const int num_pages_in_batch = kv_page_indptr_h[num_requests];
    kernels::launch_dequant_kv_cache_layer_to_bf16_active(
        kv_layer, kv_page_indices_d, num_pages_in_batch, stream);
    launch_attention_flashinfer_prefill_custom_bf16(
        q,
        kv_layer.k_bf16_pages,
        kv_layer.v_bf16_pages,
        o,
        qo_indptr_d, kv_page_indices_d, kv_page_indptr_d, kv_last_page_lens_d,
        mask_d, mask_indptr_d, qo_indptr_h, kv_page_indptr_h,
        total_tokens, num_requests, num_q_heads, kv_layer.num_kv_heads,
        kv_layer.head_dim, kv_layer.page_size, workspace, stream,
        window_left, logits_soft_cap, sm_scale,
        lse_out, kv_layer.hnd_layout);
}

}  // namespace pie_cuda_driver::ops
