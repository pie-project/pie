#include "model/attn_score.hpp"

#include <cstdio>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <vector>

#include "model/attn_observation.hpp"
#include "model/stage_hooks.hpp"
#include "ops/attention_flashinfer.hpp"
#include "store/kv_cache.hpp"

namespace pie_cuda_driver {
namespace model {

namespace {

thread_local const AttentionScores* g_attention_scores = nullptr;

// Per-fire scratch for the ragged CSR. A `thread_local` vector rather than a
// fresh allocation per layer: the geometry is identical for every layer of a
// fire, and the host CSR has to outlive the hook that reads it.
thread_local std::vector<std::uint32_t> g_raw_offsets;
thread_local std::vector<std::uint32_t> g_folded_offsets;
thread_local std::vector<std::int32_t> g_raw_offsets_i32;
thread_local int g_capture_depth = 0;

// A capture whose raw rows would cost more than this is refused rather than
// served. The prefill row set is `heads * window * kv_len` floats, which is
// quadratic in nothing but still grows with the context: at 16 heads, a
// 32-token window and a 128K context it is 268 MB. Refusing is the honest
// outcome -- the PTIR side then fails loudly on the missing `AttnScore` rather
// than evicting on a row that could not be produced.
constexpr std::uint64_t kMaxScoreBytes = 1ull << 30;  // 1 GiB

}  // namespace

namespace detail {

bool ScoreBuffers::allocate(
    std::uint64_t raw_elems,
    std::uint64_t folded_elems,
    const std::int32_t* indptr_h,
    std::uint32_t num_requests,
    cudaStream_t stream) noexcept {
    // `cudaMallocAsync` (not a pool) because the extent is a function of the
    // live context length and changes every fire; the sizes are small relative
    // to the KV read the same layer already does.
    if (cudaMallocAsync(
            reinterpret_cast<void**>(&raw),
            static_cast<std::size_t>(raw_elems) * sizeof(float),
            stream) != cudaSuccess) {
        raw = nullptr;
        return false;
    }
    if (cudaMallocAsync(
            reinterpret_cast<void**>(&folded),
            static_cast<std::size_t>(folded_elems) * sizeof(float),
            stream) != cudaSuccess) {
        folded = nullptr;
        release(stream);
        return false;
    }
    // The host CSR that sized these is an UPPER BOUND, not the exact geometry.
    // The frame layer is free to hand the body a conservative host-side page
    // CSR (graph lattice padding, the decode-envelope KV bound in `frame.cpp`),
    // while the DEVICE CSR the attention kernel reads is exact. So the capture
    // and fold kernels -- which derive their widths from the device CSR, as
    // everything in this driver must -- write only the true `kv_len` of each
    // request, and the slack up to the bound would otherwise be whatever the
    // stream-ordered allocator last left there.
    //
    // Zeroing it makes that slack read as `0.0`, which is not a papering-over
    // but the intrinsic's defined value: those positions do not exist, and a
    // position that does not exist received no attention. Without this the
    // padding is live garbage, which is the difference between "never evict a
    // position that isn't there" and "evict a real one instead".
    if (cudaMemsetAsync(
            folded, 0,
            static_cast<std::size_t>(folded_elems) * sizeof(float),
            stream) != cudaSuccess) {
        release(stream);
        return false;
    }
    const std::size_t indptr_bytes =
        (static_cast<std::size_t>(num_requests) + 1) * sizeof(std::int32_t);
    if (cudaMallocAsync(
            reinterpret_cast<void**>(&indptr_d), indptr_bytes, stream) !=
        cudaSuccess) {
        indptr_d = nullptr;
        release(stream);
        return false;
    }
    if (cudaMemcpyAsync(
            indptr_d, indptr_h, indptr_bytes, cudaMemcpyHostToDevice,
            stream) != cudaSuccess) {
        release(stream);
        return false;
    }
    return true;
}

void ScoreBuffers::release(cudaStream_t stream) noexcept {
    if (raw != nullptr) cudaFreeAsync(raw, stream);
    if (folded != nullptr) cudaFreeAsync(folded, stream);
    if (indptr_d != nullptr) cudaFreeAsync(indptr_d, stream);
    raw = nullptr;
    folded = nullptr;
    indptr_d = nullptr;
}

}  // namespace detail

std::uint32_t default_attn_score_window() noexcept {
    static const std::uint32_t value = [] {
        const char* env = std::getenv("PIE_ATTN_SCORE_WINDOW");
        if (env == nullptr) return 32u;
        const long parsed = std::strtol(env, nullptr, 10);
        if (parsed <= 0 || parsed > 4096) {
            std::fprintf(stderr,
                         "[pie-driver-cuda] PIE_ATTN_SCORE_WINDOW=%s out of "
                         "range (1..4096); using 32\n", env);
            return 32u;
        }
        return static_cast<std::uint32_t>(parsed);
    }();
    return value;
}

const AttentionScores* active_attention_scores() noexcept {
    return g_attention_scores;
}

ScopedAttentionScores::ScopedAttentionScores(
    const AttentionScores* scores) noexcept
    : previous_(g_attention_scores) {
    g_attention_scores = scores;
}

ScopedAttentionScores::~ScopedAttentionScores() noexcept {
    g_attention_scores = previous_;
}

LayerScoreCapture::LayerScoreCapture(
    std::uint32_t layer,
    std::uint32_t num_q_heads,
    bool capturable,
    cudaStream_t stream) noexcept
    : stream_(stream), layer_(layer), num_q_heads_(num_q_heads) {
    const StageHooks* hooks = active_stage_hooks;
    if (hooks == nullptr || !hooks->wants_attn_score || !capturable ||
        num_q_heads == 0) {
        return;
    }
    // The host CSR below lives in thread-local scratch, so exactly one capture
    // may be live at a time. Layers run in sequence, so that holds -- but a
    // future nested use would silently hand the outer capture the inner one's
    // offsets, which is the same shape of bug as a stale score row.
    if (g_capture_depth != 0) {
        std::fprintf(
            stderr,
            "[pie-driver-cuda] nested attention score capture is not "
            "supported; the inner capture is disabled\n");
        return;
    }
    const AttentionObservation* obs = active_attention_observation();
    if (obs == nullptr || !obs->usable()) {
        return;
    }

    // The ragged layout is derived from the page CSR, which is this driver's
    // single source of truth for sequence length (`kernels/geometry.cu`).
    // Deriving it here rather than taking a second length argument is what
    // keeps the score row attributable to the right positions.
    const int requests = obs->num_requests;
    const int page_size = obs->kv->page_size();
    g_raw_offsets.assign(static_cast<std::size_t>(requests) + 1, 0u);
    g_folded_offsets.assign(static_cast<std::size_t>(requests) + 1, 0u);
    std::uint64_t raw_total = 0;
    std::uint64_t folded_total = 0;
    for (int r = 0; r < requests; ++r) {
        const std::uint32_t pages =
            obs->kv_page_indptr_h[r + 1] - obs->kv_page_indptr_h[r];
        const std::uint32_t kv_len =
            pages == 0
                ? 0u
                : (pages - 1u) * static_cast<std::uint32_t>(page_size) +
                      obs->kv_last_page_lens_h[r];
        g_raw_offsets[static_cast<std::size_t>(r)] =
            static_cast<std::uint32_t>(raw_total);
        g_folded_offsets[static_cast<std::size_t>(r)] =
            static_cast<std::uint32_t>(folded_total);
        raw_total += static_cast<std::uint64_t>(kv_len) * num_q_heads;
        folded_total += kv_len;
    }
    g_raw_offsets[static_cast<std::size_t>(requests)] =
        static_cast<std::uint32_t>(raw_total);
    g_folded_offsets[static_cast<std::size_t>(requests)] =
        static_cast<std::uint32_t>(folded_total);
    if (raw_total == 0 || raw_total > 0xffffffffull) {
        return;
    }

    g_raw_offsets_i32.assign(g_raw_offsets.begin(), g_raw_offsets.end());

    if (!buf_.allocate(raw_total, folded_total, g_raw_offsets_i32.data(),
                       static_cast<std::uint32_t>(requests), stream)) {
        return;
    }

    folded_offsets_h_ = g_folded_offsets.data();
    active_ = true;
    ++g_capture_depth;
}

void LayerScoreCapture::publish(
    const std::uint32_t* kv_page_indptr_d,
    const std::uint32_t* kv_last_page_lens_d,
    int page_size) {
    if (!active_ || published_) return;
    const AttentionObservation* obs = active_attention_observation();
    if (obs == nullptr || !obs->usable()) {
        throw std::runtime_error(
            "attention score capture lost its fire geometry mid-layer");
    }
    ops::launch_attn_score_fold_heads(
        buf_.raw, buf_.indptr_d, kv_page_indptr_d, kv_last_page_lens_d,
        page_size, obs->num_requests, static_cast<int>(num_q_heads_),
        buf_.folded, stream_);

    payload_ = AttentionScores{
        .values = buf_.folded,
        .offsets_h = folded_offsets_h_,
        .num_requests = static_cast<std::uint32_t>(obs->num_requests),
        .layer = layer_,
    };
    binding_ = new (std::nothrow) ScopedAttentionScores(&payload_);
    published_ = binding_ != nullptr;
    if (!published_) {
        throw std::runtime_error(
            "attention score capture could not publish its payload");
    }
}

void LayerScoreCapture::release() noexcept {
    if (active_) --g_capture_depth;
    buf_.release(stream_);
    active_ = false;
}

LayerScoreCapture::~LayerScoreCapture() {
    delete binding_;
    binding_ = nullptr;
    published_ = false;
    release();
}


// ── Prefill capture (SnapKV) ───────────────────────────────────────────────

namespace {

// Separate scratch from the decode capture's. The two are mutually exclusive
// within a layer (the attention branch in a model body is an if/else chain),
// but keeping the vectors apart means a future body that constructs both --
// even with one inactive -- cannot have the inactive one clobber the live
// one's host CSR, which would survive as a silently misattributed score row.
thread_local std::vector<std::uint32_t> g_pf_folded_offsets;
thread_local std::vector<std::int32_t> g_pf_raw_offsets_i32;
thread_local int g_pf_capture_depth = 0;

}  // namespace

LayerPrefillScoreCapture::LayerPrefillScoreCapture(
    std::uint32_t layer,
    std::uint32_t num_q_heads,
    std::uint32_t window,
    bool capturable,
    cudaStream_t stream) noexcept
    : stream_(stream), layer_(layer), num_q_heads_(num_q_heads),
      window_(window) {
    const StageHooks* hooks = active_stage_hooks;
    if (hooks == nullptr || !hooks->wants_attn_score || !capturable ||
        num_q_heads == 0 || window == 0) {
        return;
    }
    if (g_pf_capture_depth != 0) {
        std::fprintf(
            stderr,
            "[pie-driver-cuda] nested prefill score capture is not "
            "supported; the inner capture is disabled\n");
        return;
    }
    const AttentionObservation* obs = active_attention_observation();
    if (obs == nullptr || !obs->usable()) {
        return;
    }

    const int requests = obs->num_requests;
    const int page_size = obs->kv->page_size();
    g_pf_folded_offsets.assign(static_cast<std::size_t>(requests) + 1, 0u);
    g_pf_raw_offsets_i32.assign(static_cast<std::size_t>(requests) + 1, 0);
    std::uint64_t raw_total = 0;
    std::uint64_t folded_total = 0;
    for (int r = 0; r < requests; ++r) {
        const std::uint32_t pages =
            obs->kv_page_indptr_h[r + 1] - obs->kv_page_indptr_h[r];
        const std::uint32_t kv_len =
            pages == 0
                ? 0u
                : (pages - 1u) * static_cast<std::uint32_t>(page_size) +
                      obs->kv_last_page_lens_h[r];
        g_pf_raw_offsets_i32[static_cast<std::size_t>(r)] =
            static_cast<std::int32_t>(raw_total);
        g_pf_folded_offsets[static_cast<std::size_t>(r)] =
            static_cast<std::uint32_t>(folded_total);
        raw_total += static_cast<std::uint64_t>(kv_len) * num_q_heads * window;
        folded_total += kv_len;
    }
    g_pf_raw_offsets_i32[static_cast<std::size_t>(requests)] =
        static_cast<std::int32_t>(raw_total);
    g_pf_folded_offsets[static_cast<std::size_t>(requests)] =
        static_cast<std::uint32_t>(folded_total);
    // The int32 CSR is what the kernels index with, so the total has to fit a
    // signed 32-bit element offset -- and the byte ceiling bites first anyway.
    if (raw_total == 0 || raw_total > 0x7fffffffull ||
        raw_total * sizeof(float) > kMaxScoreBytes) {
        if (raw_total != 0) {
            std::fprintf(
                stderr,
                "[pie-driver-cuda] prefill score capture needs %llu MiB "
                "(%u heads x %u window rows); refusing\n",
                static_cast<unsigned long long>(raw_total * sizeof(float) >> 20),
                num_q_heads, window);
        }
        return;
    }

    if (!buf_.allocate(raw_total, folded_total, g_pf_raw_offsets_i32.data(),
                       static_cast<std::uint32_t>(requests), stream)) {
        return;
    }

    folded_offsets_h_ = g_pf_folded_offsets.data();
    num_requests_ = static_cast<std::uint32_t>(requests);
    active_ = true;
    ++g_pf_capture_depth;
}

void LayerPrefillScoreCapture::publish() {
    if (!active_ || published_) return;
    payload_ = AttentionScores{
        .values = buf_.folded,
        .offsets_h = folded_offsets_h_,
        .num_requests = num_requests_,
        .layer = layer_,
    };
    binding_ = new (std::nothrow) ScopedAttentionScores(&payload_);
    published_ = binding_ != nullptr;
    if (!published_) {
        throw std::runtime_error(
            "prefill score capture could not publish its payload");
    }
}

void LayerPrefillScoreCapture::release() noexcept {
    if (active_) --g_pf_capture_depth;
    buf_.release(stream_);
    active_ = false;
}

LayerPrefillScoreCapture::~LayerPrefillScoreCapture() {
    delete binding_;
    binding_ = nullptr;
    published_ = false;
    release();
}

}  // namespace model
}  // namespace pie_cuda_driver
