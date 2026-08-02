#include "model/attn_score.hpp"

#include <cstdio>
#include <cstdlib>
#include <new>
#include <stdexcept>
#include <vector>

#include "model/attn_observation.hpp"
#include "model/hook_sideband_arena.hpp"
#include "model/stage_hooks.hpp"
#include "ops/attention_flashinfer.hpp"
#include "store/kv_cache.hpp"

namespace pie_cuda_driver {
namespace model {

namespace {

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

// Sub-buffer alignment inside an arena slot. 256 covers every dtype the
// sidebands carve (and matches cudaMalloc's own guarantee).
constexpr std::size_t kSidebandAlign = 256;

constexpr std::size_t align_up(std::size_t n) noexcept {
    return (n + kSidebandAlign - 1) & ~(kSidebandAlign - 1);
}

// The score slot's internal carve — raw, then folded, then the CSR, each
// aligned. ONE definition shared by `ScoreBuffers::acquire` and the
// hook-graph prepare helper (`prepare_decode_score_capture`): the helper
// derives arena-stable addresses a captured graph bakes, so the two going
// out of step would be a silent replay miscompute, not a crash.
struct ScoreSlotLayout {
    std::size_t total = 0;
    std::size_t folded_offset = 0;
    std::size_t indptr_offset = 0;
};

constexpr ScoreSlotLayout score_slot_layout(
    std::size_t raw_bytes,
    std::size_t folded_bytes,
    std::size_t indptr_bytes) noexcept {
    return ScoreSlotLayout{
        align_up(raw_bytes) + align_up(folded_bytes) + align_up(indptr_bytes),
        align_up(raw_bytes),
        align_up(raw_bytes) + align_up(folded_bytes),
    };
}

}  // namespace

namespace {

struct DecodeScoreCsrTotals {
    std::uint64_t raw_total = 0;
    std::uint64_t folded_total = 0;
};

// Fill the decode capture's thread-local CSRs from the fire's KV geometry.
// ONE definition shared by `LayerScoreCapture`'s constructor (which runs at
// capture time, inside the recorded body) and `prepare_decode_score_capture`
// (which runs before every replay): the captured upload node reads
// `g_raw_offsets_i32`'s storage at replay time, so both sites must compute
// byte-identical contents into the same vectors or a replayed fire scores
// against a stale channel view of its KV lengths.
DecodeScoreCsrTotals compute_decode_score_csr(
    const AttentionObservation& obs,
    std::uint32_t num_q_heads) {
    const int requests = obs.num_requests;
    const int page_size = obs.kv->page_size();
    g_raw_offsets.assign(static_cast<std::size_t>(requests) + 1, 0u);
    g_folded_offsets.assign(static_cast<std::size_t>(requests) + 1, 0u);
    DecodeScoreCsrTotals totals;
    for (int r = 0; r < requests; ++r) {
        const std::uint32_t pages =
            obs.kv_page_indptr_h[r + 1] - obs.kv_page_indptr_h[r];
        const std::uint32_t kv_len =
            pages == 0
                ? 0u
                : (pages - 1u) * static_cast<std::uint32_t>(page_size) +
                      obs.kv_last_page_lens_h[r];
        g_raw_offsets[static_cast<std::size_t>(r)] =
            static_cast<std::uint32_t>(totals.raw_total);
        g_folded_offsets[static_cast<std::size_t>(r)] =
            static_cast<std::uint32_t>(totals.folded_total);
        totals.raw_total +=
            static_cast<std::uint64_t>(kv_len) * num_q_heads;
        totals.folded_total += kv_len;
    }
    g_raw_offsets[static_cast<std::size_t>(requests)] =
        static_cast<std::uint32_t>(totals.raw_total);
    g_folded_offsets[static_cast<std::size_t>(requests)] =
        static_cast<std::uint32_t>(totals.folded_total);
    return totals;
}

}  // namespace

DecodeScoreCapturePlan prepare_decode_score_capture(
    HookSidebandArena* arena,
    const AttentionObservation& observation,
    std::uint32_t num_q_heads,
    cudaStream_t stream) {
    DecodeScoreCapturePlan plan;
    if (arena == nullptr || !observation.usable() || num_q_heads == 0) {
        return plan;
    }
    const DecodeScoreCsrTotals totals =
        compute_decode_score_csr(observation, num_q_heads);
    // Same validity bounds as the constructor: a fire the constructor would
    // refuse must not be declared replayable.
    if (totals.raw_total == 0 || totals.raw_total > 0xffffffffull) {
        return plan;
    }
    g_raw_offsets_i32.assign(g_raw_offsets.begin(), g_raw_offsets.end());

    const std::size_t raw_bytes =
        static_cast<std::size_t>(totals.raw_total) * sizeof(float);
    const std::size_t folded_bytes =
        static_cast<std::size_t>(totals.folded_total) * sizeof(float);
    const std::size_t indptr_bytes =
        (static_cast<std::size_t>(observation.num_requests) + 1) *
        sizeof(std::int32_t);
    const ScoreSlotLayout layout =
        score_slot_layout(raw_bytes, folded_bytes, indptr_bytes);
    // Acquire-and-release: growth (stream-synced free+realloc) is pulled to
    // HERE, outside any captured region; the capture-time constructor then
    // finds sufficient capacity and its acquire is a host-side pointer
    // return. The slot is not held — the busy flag guards overlapping
    // captures, and this pass holds nothing across the fire.
    auto* base = static_cast<std::uint8_t*>(arena->acquire(
        HookSidebandArena::Region::Score, layout.total, stream));
    if (base == nullptr) {
        return plan;
    }
    arena->release(HookSidebandArena::Region::Score);
    plan.ok = true;
    plan.folded = reinterpret_cast<const float*>(base + layout.folded_offset);
    plan.indptr_d = reinterpret_cast<const std::int32_t*>(
        base + layout.indptr_offset);
    plan.indptr_h_data = g_raw_offsets_i32.data();
    plan.folded_offsets_h = g_folded_offsets.data();
    plan.num_requests =
        static_cast<std::uint32_t>(observation.num_requests);
    return plan;
}

namespace detail {

bool ScoreBuffers::acquire(
    HookSidebandArena* arena,
    std::uint64_t raw_elems,
    std::uint64_t folded_elems,
    const std::int32_t* indptr_h,
    std::uint32_t num_requests,
    cudaStream_t stream) noexcept {
    if (arena == nullptr) {
        // The launch path always wires the arena onto the fire's hooks
        // (batch/frame.cpp); reaching here means a new call site forgot.
        // Refusing takes the same path as an allocation failure: the capture
        // stands down and the PTIR side fails loudly at the hook.
        std::fprintf(
            stderr,
            "[pie-driver-cuda] score capture has no hook sideband arena; "
            "refusing the capture\n");
        return false;
    }
    // One slot for all three buffers: raw, then folded, then the CSR, each
    // aligned. The extent is a function of the live context length and
    // changes every fire, but the arena only ever GROWS — steady state is a
    // capacity check, not an allocation.
    const std::size_t raw_bytes =
        static_cast<std::size_t>(raw_elems) * sizeof(float);
    const std::size_t folded_bytes =
        static_cast<std::size_t>(folded_elems) * sizeof(float);
    const std::size_t indptr_bytes =
        (static_cast<std::size_t>(num_requests) + 1) * sizeof(std::int32_t);
    const ScoreSlotLayout layout =
        score_slot_layout(raw_bytes, folded_bytes, indptr_bytes);
    auto* base = static_cast<std::uint8_t*>(arena->acquire(
        HookSidebandArena::Region::Score, layout.total, stream));
    if (base == nullptr) {
        return false;
    }
    arena_ = arena;
    raw = reinterpret_cast<float*>(base);
    folded = reinterpret_cast<float*>(base + layout.folded_offset);
    indptr_d = reinterpret_cast<std::int32_t*>(
        base + layout.indptr_offset);
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
    // position that isn't there" and "evict a real one instead". With a
    // reused slot the garbage would be the PREVIOUS layer's folded row — a
    // plausible attention distribution, the worst kind of wrong — so the
    // memset happens on every acquire, never just on growth.
    if (cudaMemsetAsync(folded, 0, folded_bytes, stream) != cudaSuccess) {
        release();
        return false;
    }
    if (cudaMemcpyAsync(
            indptr_d, indptr_h, indptr_bytes, cudaMemcpyHostToDevice,
            stream) != cudaSuccess) {
        release();
        return false;
    }
    return true;
}

void ScoreBuffers::release() noexcept {
    if (arena_ != nullptr) {
        arena_->release(HookSidebandArena::Region::Score);
    }
    arena_ = nullptr;
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

LayerScoreCapture::LayerScoreCapture(
    const StageHooks* hooks,
    std::uint32_t layer,
    std::uint32_t num_q_heads,
    bool capturable,
    cudaStream_t stream) noexcept
    : stream_(stream), layer_(layer), num_q_heads_(num_q_heads),
      hooks_(hooks) {
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
    const AttentionObservation* obs = hooks->observation;
    if (obs == nullptr || !obs->usable()) {
        return;
    }

    // The ragged layout is derived from the page CSR, which is this driver's
    // single source of truth for sequence length (`kernels/geometry.cu`).
    // Deriving it here rather than taking a second length argument is what
    // keeps the score row attributable to the right positions. The
    // computation is shared with `prepare_decode_score_capture` — under a
    // captured hook body (stage 6 increment 4) THAT is what refreshes these
    // thread-locals before every replay, so the two must be one function.
    const int requests = obs->num_requests;
    const DecodeScoreCsrTotals totals =
        compute_decode_score_csr(*obs, num_q_heads);
    const std::uint64_t raw_total = totals.raw_total;
    const std::uint64_t folded_total = totals.folded_total;
    if (raw_total == 0 || raw_total > 0xffffffffull) {
        return;
    }

    g_raw_offsets_i32.assign(g_raw_offsets.begin(), g_raw_offsets.end());

    if (!buf_.acquire(hooks->sideband_arena, raw_total, folded_total,
                      g_raw_offsets_i32.data(),
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
    const AttentionObservation* obs =
        hooks_ != nullptr ? hooks_->observation : nullptr;
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
    published_ = true;
}

void LayerScoreCapture::release() noexcept {
    if (active_) --g_capture_depth;
    buf_.release();
    active_ = false;
}

LayerScoreCapture::~LayerScoreCapture() {
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
    const StageHooks* hooks,
    std::uint32_t layer,
    std::uint32_t num_q_heads,
    std::uint32_t window,
    bool capturable,
    cudaStream_t stream) noexcept
    : stream_(stream), layer_(layer), num_q_heads_(num_q_heads),
      window_(window) {
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
    const AttentionObservation* obs = hooks->observation;
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

    if (!buf_.acquire(hooks->sideband_arena, raw_total, folded_total,
                      g_pf_raw_offsets_i32.data(),
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
    published_ = true;
}

void LayerPrefillScoreCapture::release() noexcept {
    if (active_) --g_pf_capture_depth;
    buf_.release();
    active_ = false;
}

LayerPrefillScoreCapture::~LayerPrefillScoreCapture() {
    published_ = false;
    release();
}

}  // namespace model
}  // namespace pie_cuda_driver
