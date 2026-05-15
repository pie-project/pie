#include "request_handler.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <condition_variable>
#include <iostream>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "attention_workspace.hpp"
#include "brle.hpp"
#include "cuda_check.hpp"
#include "device_buffer.hpp"
#include "distributed.hpp"
#include "engine.hpp"
#include "kv_cache.hpp"
#include "response_subpass.hpp"
#include "model/qwen3.hpp"
#include "model/qwen3_forward.hpp"
#include "ops/gemm.hpp"
#include "sampler_type.hpp"
#include "sampling_dispatch.hpp"
#include "spec_expansion.hpp"

namespace pie_cuda_driver {

namespace {

struct TpCpuGate {
    std::mutex mu;
    std::condition_variable cv;
    std::uint64_t seq = 0;
};

std::mutex g_tp_cpu_gates_mu;
std::unordered_map<std::string, std::shared_ptr<TpCpuGate>> g_tp_cpu_gates;

std::shared_ptr<TpCpuGate> tp_cpu_gate_for(const std::string& key) {
    std::lock_guard<std::mutex> lk(g_tp_cpu_gates_mu);
    auto& gate = g_tp_cpu_gates[key];
    if (!gate) gate = std::make_shared<TpCpuGate>();
    return gate;
}

void tp_cpu_gate_notify(const std::string& key) {
    if (key.empty()) return;
    auto gate = tp_cpu_gate_for(key);
    {
        std::lock_guard<std::mutex> lk(gate->mu);
        ++gate->seq;
    }
    gate->cv.notify_all();
}

void tp_cpu_gate_wait(const std::string& key,
                      std::uint64_t& seen,
                      std::atomic<bool>& stop) {
    if (key.empty()) return;
    auto gate = tp_cpu_gate_for(key);
    std::unique_lock<std::mutex> lk(gate->mu);
    gate->cv.wait(lk, [&] {
        return stop.load() || gate->seq != seen;
    });
    seen = gate->seq;
}

// Broadcast header sent from rank 0 → followers before each fire's
// per-fire payload. Followers parse it to size the subsequent broadcasts
// + the forward call. Two magic values:
//
//   * TP_FIRE_MAGIC: a regular fire is incoming; payload broadcasts follow.
//   * TP_STOP_MAGIC: shutdown sentinel; follower exits its serve loop.
//
// Sized at exactly 8 i32 so we can broadcast it as `8 * sizeof(int32_t)`
// bytes without alignment surprises across compilers.
struct TpFireHeader {
    std::int32_t magic;
    std::int32_t total_tokens;
    std::int32_t num_requests;
    std::int32_t is_pure_decode;
    std::int32_t kv_indices_count;
    std::int32_t mask_bytes;
    std::int32_t mask_indptr_count;
    // 1 = slot_ids[R] (int32) and is_fresh[R] (uint8) follow the
    // existing payload broadcasts. Inert (0) for archs that don't use
    // a state cache — followers skip those broadcasts.
    std::int32_t has_slot_ids;
};
static_assert(sizeof(TpFireHeader) == 8 * sizeof(std::int32_t),
              "TpFireHeader must pack into exactly 8 ints");
constexpr std::int32_t TP_FIRE_MAGIC = 0x55504954;  // 'TPIU' tag
constexpr std::int32_t TP_STOP_MAGIC = 0x504F5453;  // 'STOP' tag

// Lazily-allocated 32-byte device buffer holding the broadcast header.
// Both rank 0 and followers reuse it across fires; no need to plumb it
// through ForwardContext.
std::int32_t* tp_hdr_dev_buf() {
    thread_local std::int32_t* buf = nullptr;
    if (buf == nullptr) {
        CUDA_CHECK(cudaMalloc(&buf, sizeof(TpFireHeader)));
    }
    return buf;
}

// Issue every per-fire broadcast in dependency order. Caller has already
// refilled `pi.*` with the current fire's data; this just fans them out.
// All ops run on the default stream so they sequence correctly with the
// kernels that follow inside `forward_fn.body`.
void tp_broadcast_inputs(NcclComm& comm, PersistentInputs& pi,
                         int N, int R, bool is_pure_decode,
                         int kv_indices_count,
                         int mask_bytes, int mask_indptr_count,
                         bool has_slot_ids,
                         cudaStream_t stream)
{
    auto* d_hdr = tp_hdr_dev_buf();
    TpFireHeader hdr{
        TP_FIRE_MAGIC, N, R, is_pure_decode ? 1 : 0,
        kv_indices_count, mask_bytes, mask_indptr_count,
        has_slot_ids ? 1 : 0,
    };
    // Header goes first (synchronous from the followers' POV — they need
    // to parse sizes before posting matching payload broadcasts).
    CUDA_CHECK(cudaMemcpyAsync(d_hdr, &hdr, sizeof(hdr),
                               cudaMemcpyHostToDevice, stream));
    NCCL_CHECK(ncclBroadcast(d_hdr, d_hdr, sizeof(hdr), ncclChar, 0,
                             comm.comm(), stream));
    // Group the payload broadcasts so NCCL submits them as a single batch
    // — tens of microseconds of host-side launch overhead saved per fire,
    // most visible at small batch sizes (decode where each broadcast is
    // sub-KB but the fixed per-op cost dominates).
    NCCL_CHECK(ncclGroupStart());
    NCCL_CHECK(ncclBroadcast(pi.tokens.data(), pi.tokens.data(),
                             static_cast<std::size_t>(N) * 4, ncclChar, 0,
                             comm.comm(), stream));
    NCCL_CHECK(ncclBroadcast(pi.positions.data(), pi.positions.data(),
                             static_cast<std::size_t>(N) * 4, ncclChar, 0,
                             comm.comm(), stream));
    NCCL_CHECK(ncclBroadcast(pi.qo_indptr.data(), pi.qo_indptr.data(),
                             static_cast<std::size_t>(R + 1) * 4, ncclChar, 0,
                             comm.comm(), stream));
    NCCL_CHECK(ncclBroadcast(pi.kv_page_indptr.data(), pi.kv_page_indptr.data(),
                             static_cast<std::size_t>(R + 1) * 4, ncclChar, 0,
                             comm.comm(), stream));
    if (R > 0) {
        NCCL_CHECK(ncclBroadcast(pi.kv_last_page_lens.data(),
                                 pi.kv_last_page_lens.data(),
                                 static_cast<std::size_t>(R) * 4, ncclChar, 0,
                                 comm.comm(), stream));
    }
    if (kv_indices_count > 0) {
        NCCL_CHECK(ncclBroadcast(pi.kv_page_indices.data(),
                                 pi.kv_page_indices.data(),
                                 static_cast<std::size_t>(kv_indices_count) * 4,
                                 ncclChar, 0, comm.comm(), stream));
    }
    if (mask_bytes > 0) {
        NCCL_CHECK(ncclBroadcast(pi.custom_mask.data(),
                                 pi.custom_mask.data(),
                                 static_cast<std::size_t>(mask_bytes), ncclChar, 0,
                                 comm.comm(), stream));
        NCCL_CHECK(ncclBroadcast(pi.custom_mask_indptr.data(),
                                 pi.custom_mask_indptr.data(),
                                 static_cast<std::size_t>(mask_indptr_count) * 4,
                                 ncclChar, 0, comm.comm(), stream));
    }
    if (has_slot_ids && R > 0) {
        NCCL_CHECK(ncclBroadcast(pi.slot_ids.data(), pi.slot_ids.data(),
                                 static_cast<std::size_t>(R) * 4, ncclChar, 0,
                                 comm.comm(), stream));
        NCCL_CHECK(ncclBroadcast(pi.is_fresh.data(), pi.is_fresh.data(),
                                 static_cast<std::size_t>(R), ncclChar, 0,
                                 comm.comm(), stream));
    }
    NCCL_CHECK(ncclGroupEnd());
}

}  // namespace

// Per-stage timing instrumentation. Logged every ~50 fires so the
// throughput bench's per-fire timeline shows where the host overhead
// goes between flashinfer's plan, the GPU sync, and msgpack
// encode/decode. Toggle by setting PIE_TIMING=1 in the env.
struct FireTiming {
    std::uint64_t t_start_us;
    std::uint64_t t_after_parse_us;
    std::uint64_t t_after_views_us;
    std::uint64_t t_after_uploads_us;
    std::uint64_t t_after_prepare_us;
    std::uint64_t t_after_launch_us;
    std::uint64_t t_after_sync_us;
    std::uint64_t t_after_response_us;
    int R;
    int N;
};

inline std::uint64_t fire_now_us() noexcept {
    using namespace std::chrono;
    return duration_cast<microseconds>(
               steady_clock::now().time_since_epoch()).count();
}

inline bool fire_timing_enabled() {
    static const bool enabled = std::getenv("PIE_TIMING") != nullptr;
    return enabled;
}

void handle_fire_batch(
    std::uint32_t req_id,
    const pie_driver::PieForwardRequestView& view,
    pie_driver::PieForwardResponseView& out_resp,
    ForwardContext& ctx,
    std::uint64_t handled)
{
    FireTiming tm{};
    const bool tm_on = fire_timing_enabled();
    if (tm_on) tm.t_start_us = fire_now_us();

    // Local references so the (lifted) body uses the same names it had
    // when it lived as a `[&]`-capturing lambda in main.cpp. Avoids a
    // mechanical rename across ~900 lines.
    auto& engine               = ctx.engine;
    auto& ws                   = ctx.ws;
    auto& kv_cache             = ctx.kv_cache;
    auto& attn_ws              = ctx.attn_ws;
    auto& cublas               = ctx.cublas;
    auto& pi                   = ctx.inputs;  // persistent input slabs
    auto& forward_fn           = ctx.forward_fn;
    const int max_workspace_tokens = ctx.max_workspace_tokens;

    // Track whether the custom-mask path was populated this fire so the
    // forward kernel knows whether to consume `pi.custom_mask`. Sizes are
    // stashed alongside so the TP broadcast knows how many bytes to fan
    // out to followers.
    bool have_custom_mask = false;
    int mask_bytes = 0;
    int mask_indptr_count = 0;

    try {
        const auto tok_view_orig   = view.token_ids.as<std::uint32_t>();
        const auto pos_view_orig   = view.position_ids.as<std::uint32_t>();
        const auto qo_view_orig    = view.qo_indptr.as<std::uint32_t>();
        const auto kvpi_view = view.kv_page_indices.as<std::uint32_t>();
        const auto kvpp_view = view.kv_page_indptr.as<std::uint32_t>();
        const auto kvlpl_view_orig = view.kv_last_page_lens.as<std::uint32_t>();

        // Wire-trace, enabled via `PIE_CUDA_TRACE_KV=1`. Used during the
        // 2026-05-14 diagnosis that proved cuda's multi-token prefill is
        // broken (produces token-0 argmax) while the bridge wire is
        // identical to portable's. Kept in tree as a low-cost diagnostic.
        if (std::getenv("PIE_CUDA_TRACE_KV")) {
            const auto sidx_trace = view.sampling_indices.as<std::uint32_t>();
            const auto sptr_trace = view.sampling_indptr.as<std::uint32_t>();
            const auto types_trace = view.sampler_types.as<std::uint32_t>();
            const auto temp_trace = view.sampler_temperatures.as<float>();
            std::cerr << "[cuda-trace] req: tokens=[";
            for (auto t : tok_view_orig) std::cerr << t << ",";
            std::cerr << "] positions=[";
            for (auto p : pos_view_orig) std::cerr << p << ",";
            std::cerr << "] qo_indptr=[";
            for (auto q : qo_view_orig) std::cerr << q << ",";
            std::cerr << "] sampling_indices=[";
            for (auto s : sidx_trace) std::cerr << s << ",";
            std::cerr << "] sampling_indptr=[";
            for (auto s : sptr_trace) std::cerr << s << ",";
            std::cerr << "] sampler_types=[";
            for (auto s : types_trace) std::cerr << s << ",";
            std::cerr << "] sampler_temps=[";
            for (auto s : temp_trace) std::cerr << s << ",";
            std::cerr << "] kv_pages=[";
            for (auto k : kvpi_view) std::cerr << k << ",";
            std::cerr << "] kv_indptr=[";
            for (auto k : kvpp_view) std::cerr << k << ",";
            std::cerr << "] kv_last_lens=[";
            for (auto k : kvlpl_view_orig) std::cerr << k << ",";
            std::cerr << "]\n";
        }
        const auto sidx_view_orig  = view.sampling_indices.as<std::uint32_t>();
        const auto sptr_view_orig  = view.sampling_indptr.as<std::uint32_t>();
        // Per-request stable context ids — used to drive the linear-attn
        // state-cache slot allocator below.
        const auto ctx_id_view     = view.context_ids.as<std::uint64_t>();

        // Sampler params (per-sampler arrays). Read here (rather than
        // further down) so the spec expansion below can append cloned
        // entries for the verification block.
        const auto temp_view_orig  = view.sampler_temperatures.as<float>();
        const auto top_k_view_orig = view.sampler_top_k.as<std::uint32_t>();
        const auto top_p_view_orig = view.sampler_top_p.as<float>();
        const auto min_p_view_orig = view.sampler_min_p.as<float>();
        const auto types_view_orig = view.sampler_types.as<std::uint32_t>();
        const auto seed_view_orig  = view.sampler_seeds.as<std::uint32_t>();
        const auto rns_view_orig   = view.request_num_samplers.as<std::uint32_t>();

        // Spec-decoding fields. When non-empty for some request, splice
        // drafts into the forward and append a verification block to
        // the sampling layout (one extra sample per draft + one bonus).
        // Mirrors pie_driver's `get_spec_expanded_*`.
        const auto spec_tok_view  = view.spec_token_ids.as<std::uint32_t>();
        const auto spec_pos_view  = view.spec_position_ids.as<std::uint32_t>();
        const auto spec_iptr_view = view.spec_indptr.as<std::uint32_t>();
        const bool has_spec_drafts = !spec_tok_view.empty();

        const int R = static_cast<int>(qo_view_orig.size()) - 1;

        // Spec-decoding batch expansion. When `has_spec_drafts` is false
        // the result has empty vectors and `verify_slot_start[r] == -1`
        // for every r; the active spans below fall through to the
        // original wire views.
        const SpecExpansion spec = expand_spec_batch(
            SpecExpansionInputs{
                tok_view_orig, pos_view_orig, qo_view_orig, kvlpl_view_orig,
                sidx_view_orig, sptr_view_orig, rns_view_orig,
                types_view_orig, top_k_view_orig, seed_view_orig,
                temp_view_orig, top_p_view_orig, min_p_view_orig,
                spec_tok_view, spec_pos_view, spec_iptr_view,
                kv_cache.page_size(),
            },
            R);
        const std::vector<int>& verify_slot_start = spec.verify_slot_start;
        const std::vector<int>& verify_n_drafts   = spec.verify_n_drafts;

        // Active views: spec-expanded if drafts present, else direct
        // wire. The rest of the function uses these.
        const std::span<const std::uint32_t> tok_view   = spec.has_drafts ? std::span<const std::uint32_t>(spec.tokens)               : tok_view_orig;
        const std::span<const std::uint32_t> pos_view   = spec.has_drafts ? std::span<const std::uint32_t>(spec.positions)            : pos_view_orig;
        const std::span<const std::uint32_t> qo_view    = spec.has_drafts ? std::span<const std::uint32_t>(spec.qo_indptr)            : qo_view_orig;
        const std::span<const std::uint32_t> kvlpl_view = spec.has_drafts ? std::span<const std::uint32_t>(spec.kv_last_page_lens)    : kvlpl_view_orig;
        const std::span<const std::uint32_t> sidx_view  = spec.has_drafts ? std::span<const std::uint32_t>(spec.sampling_indices)     : sidx_view_orig;
        const std::span<const std::uint32_t> sptr_view  = spec.has_drafts ? std::span<const std::uint32_t>(spec.sampling_indptr)      : sptr_view_orig;
        const std::span<const std::uint32_t> rns_view   = spec.has_drafts ? std::span<const std::uint32_t>(spec.request_num_samplers) : rns_view_orig;
        const std::span<const std::uint32_t> types_view = spec.has_drafts ? std::span<const std::uint32_t>(spec.sampler_types)        : types_view_orig;
        const std::span<const std::uint32_t> top_k_view = spec.has_drafts ? std::span<const std::uint32_t>(spec.sampler_top_k)        : top_k_view_orig;
        const std::span<const std::uint32_t> seed_view  = spec.has_drafts ? std::span<const std::uint32_t>(spec.sampler_seeds)        : seed_view_orig;
        const std::span<const float>         temp_view  = spec.has_drafts ? std::span<const float>        (spec.sampler_temperatures) : temp_view_orig;
        const std::span<const float>         top_p_view = spec.has_drafts ? std::span<const float>        (spec.sampler_top_p)        : top_p_view_orig;
        const std::span<const float>         min_p_view = spec.has_drafts ? std::span<const float>        (spec.sampler_min_p)        : min_p_view_orig;

        const int N = static_cast<int>(tok_view.size());
        const int num_sampling = static_cast<int>(sidx_view.size());

        if (N == 0 || R <= 0) {
            // Empty batch — emit a zero-request response view.
            std::vector<pie_driver::PerRequestOutput> empty(std::max(R, 0));
            ctx.response_builder.build(empty, out_resp);
            return;
        }
        if (N > max_workspace_tokens) {
            std::cerr << "[pie-driver-cuda] batch tokens=" << N
                      << " exceeds workspace=" << max_workspace_tokens << "\n";
            out_resp = pie_driver::PieForwardResponseView{};
            return;
        }

        // Compute max KV length across requests for shmem sizing.
        // Also detect "pure decode" (every request has qo_len == 1) so
        // we can dispatch flashinfer's decode kernel on the hot path.
        const int page_size = kv_cache.page_size();
        int max_kv_len = 0;
        const std::uint32_t* h_kvpp  = kvpp_view.data();
        const std::uint32_t* h_kvlpl = kvlpl_view.data();
        const std::uint32_t* h_qo    = qo_view.data();
        bool is_pure_decode = (R > 0);
        for (int r = 0; r < R; ++r) {
            const int num_pages_r = static_cast<int>(h_kvpp[r + 1] - h_kvpp[r]);
            if (num_pages_r <= 0) continue;
            const int kv_len_r = (num_pages_r - 1) * page_size +
                                 static_cast<int>(h_kvlpl[r]);
            if (kv_len_r > max_kv_len) max_kv_len = kv_len_r;
            if (h_qo[r + 1] - h_qo[r] != 1u) is_pure_decode = false;
        }

        if (tm_on) tm.t_after_parse_us = fire_now_us();
        // Refill persistent device buffers with this fire's wire inputs.
        // Same device addresses every fire — required for graph-replay
        // safety; cheap (single async memcpy each) on its own.
        pi.tokens.copy_from_host(tok_view);
        pi.positions.copy_from_host(pos_view);
        pi.qo_indptr.copy_from_host(qo_view);
        pi.kv_page_indices.copy_from_host(kvpi_view);
        pi.kv_page_indptr.copy_from_host(kvpp_view);
        pi.kv_last_page_lens.copy_from_host(kvlpl_view);

        // BRLE attention masks. For prefill batches that aren't pure
        // causal, decode + upload a packed bitmap and route through the
        // flashinfer kCustom path. For decode-only batches the kernel
        // doesn't support custom masks; we proceed without one (a
        // limitation we'd have to fix by routing decode through the
        // prefill kernel for custom-mask inferlets).
        const auto fmask_view  = view.flattened_masks.as<std::uint32_t>();
        const auto mskptr_view = view.mask_indptr.as<std::uint32_t>();
        if (has_spec_drafts) {
            // Spec mode: synthesize a causal mask for the expanded
            // sequence directly (bypasses BRLE decode). This mirrors
            // pie_driver's spec-expanded `attention_masks_ext`: every
            // token attends to keys at positions [0, pos]. Bit packing
            // matches `brle::decode`: LSB-first within each byte.
            std::vector<std::int32_t> ind(R + 1, 0);
            for (int r = 0; r < R; ++r) {
                const int num_pages_r = static_cast<int>(kvpp_view[r + 1] - kvpp_view[r]);
                const int kv_len_r = (num_pages_r > 0)
                    ? (num_pages_r - 1) * page_size + static_cast<int>(kvlpl_view[r])
                    : 0;
                const int qo_len_r = static_cast<int>(qo_view[r + 1] - qo_view[r]);
                const std::int64_t bits = static_cast<std::int64_t>(qo_len_r) * kv_len_r;
                ind[r + 1] = ind[r] + static_cast<std::int32_t>((bits + 7) / 8);
            }
            std::vector<std::uint8_t> packed(ind.back(), 0);
            for (int r = 0; r < R; ++r) {
                const int num_pages_r = static_cast<int>(kvpp_view[r + 1] - kvpp_view[r]);
                const int kv_len_r = (num_pages_r > 0)
                    ? (num_pages_r - 1) * page_size + static_cast<int>(kvlpl_view[r])
                    : 0;
                const int qo_lo = static_cast<int>(qo_view[r]);
                const int qo_hi = static_cast<int>(qo_view[r + 1]);
                std::uint8_t* base = packed.data() + ind[r];
                for (int q = 0; q < (qo_hi - qo_lo); ++q) {
                    const int abs_pos = static_cast<int>(pos_view[qo_lo + q]);
                    const int valid = std::min(abs_pos + 1, kv_len_r);
                    const std::int64_t row_off =
                        static_cast<std::int64_t>(q) * kv_len_r;
                    for (int k = 0; k < valid; ++k) {
                        const std::int64_t bit = row_off + k;
                        base[bit / 8] |= static_cast<std::uint8_t>(1u << (bit % 8));
                    }
                }
            }
            pi.custom_mask.copy_from_host(std::span<const std::uint8_t>(packed));
            pi.custom_mask_indptr.copy_from_host(std::span<const std::int32_t>(ind));
            mask_bytes = static_cast<int>(packed.size());
            mask_indptr_count = static_cast<int>(ind.size());
            have_custom_mask = true;
        } else if (!is_pure_decode && !fmask_view.empty()) {
            const auto qo_span =
                std::span<const std::uint32_t>(qo_view.data(), qo_view.size());
            const auto kvpp_span =
                std::span<const std::uint32_t>(kvpp_view.data(), kvpp_view.size());
            const auto kvlpl_span =
                std::span<const std::uint32_t>(kvlpl_view.data(), kvlpl_view.size());
            if (!pie_cuda_driver::brle::is_pure_causal(
                    fmask_view, mskptr_view,
                    qo_span, kvpp_span, kvlpl_span,
                    kv_cache.page_size())) {
                auto decoded = pie_cuda_driver::brle::decode(
                    fmask_view, mskptr_view,
                    qo_span, kvpp_span, kvlpl_span,
                    kv_cache.page_size());
                pi.custom_mask.copy_from_host(
                    std::span<const std::uint8_t>(decoded.packed));
                pi.custom_mask_indptr.copy_from_host(
                    std::span<const std::int32_t>(decoded.mask_indptr));
                mask_bytes = static_cast<int>(decoded.packed.size());
                mask_indptr_count = static_cast<int>(decoded.mask_indptr.size());
                have_custom_mask = true;
            }
        }

        // Linear-attention slot allocation. Active only when the
        // allocator was sized at engine init (max_slots > 0 — i.e.
        // qwen3.5 / qwen3.6-MoE with any linear-attn layers); a
        // zero-capacity allocator returns an empty span and the
        // forward simply ignores the slot args.
        std::vector<std::int32_t> slot_ids_h;
        std::vector<std::uint8_t> is_fresh_h;
        const bool use_slots =
            ctx.slot_alloc.max_slots() > 0 && R > 0 &&
            ctx_id_view.size() == static_cast<std::size_t>(R);
        if (use_slots) {
            slot_ids_h.resize(R);
            is_fresh_h.resize(R);
            for (int r = 0; r < R; ++r) {
                const auto acq = ctx.slot_alloc.acquire(ctx_id_view[r]);
                slot_ids_h[r]  = acq.slot;
                is_fresh_h[r]  = acq.is_fresh ? 1u : 0u;
            }
            ctx.slot_alloc.end_of_fire();
            pi.slot_ids.copy_from_host(std::span<const std::int32_t>(slot_ids_h));
            pi.is_fresh.copy_from_host(std::span<const std::uint8_t>(is_fresh_h));
        }

        // TP fan-out. Rank 0 broadcasts the per-fire payload (header +
        // refilled persistent_inputs) to every follower so they can run
        // the same forward kernels against an identical view of inputs.
        // The all-reduces inside `forward_fn.body` then synchronise the
        // ranks layer-by-layer.
        if (ctx.tp_comm != nullptr) {
            tp_cpu_gate_notify(ctx.tp_cpu_gate_key);
            tp_broadcast_inputs(*ctx.tp_comm, pi,
                                N, R, is_pure_decode,
                                static_cast<int>(kvpi_view.size()),
                                mask_bytes, mask_indptr_count,
                                /*has_slot_ids=*/use_slots,
                                /*stream=*/nullptr);
        }

        // ── Sample-plan construction (hoisted) ──────────────────
        // The body call below is wrapped in a CUDA-graph capture region
        // when `forward_fn.graph_safe`. For the captured graph to
        // include the sampling kernel + scatter (so we don't pay
        // per-fire kernel-launch overhead on the sampling step either),
        // the per-row sampler params must be uploaded to `pi.sample_*`
        // BEFORE we begin capture. So: build the host-side plan here,
        // then upload via cudaMemcpyAsync on the default stream — same
        // pattern as `pi.tokens.copy_from_host` for the wire payload.
        // Legacy-default-stream semantics keep the captured kernel
        // (on `cstream`) ordered after the upload.
        const auto outspec_view = view.output_spec_flags.as<std::uint8_t>();
        bool need_msgpack = false;
        for (auto t : types_view) {
            if (pie_cuda_driver::is_msgpack_only(t)) { need_msgpack = true; break; }
        }
        if (!need_msgpack) {
            for (auto f : outspec_view) {
                if (f) { need_msgpack = true; break; }
            }
        }
        if (has_spec_drafts) need_msgpack = true;

        std::vector<float> h_per_temp(N, 0.f);
        std::vector<float> h_per_min_p(N, 0.f);
        std::vector<float> h_per_top_p(N, 1.f);
        std::vector<std::int32_t> h_per_top_k(N, 0);
        std::vector<std::uint32_t> h_per_seed(N, 0u);

        const std::uint32_t* h_sptr  = sptr_view.data();
        const std::uint32_t* h_sidx  = sidx_view.data();
        const std::uint32_t* h_rns   = rns_view.data();
        const float*         h_temp  = temp_view.data();
        const std::uint32_t* h_top_k = top_k_view.data();
        const float*         h_top_p = top_p_view.data();
        const float*         h_min_p = min_p_view.data();
        const std::uint32_t* h_seed  = seed_view.data();

        std::vector<std::uint32_t> per_slot_type(num_sampling, 1u);
        std::vector<float>         per_slot_temp (num_sampling, 0.f);
        std::vector<float>         per_slot_top_p(num_sampling, 1.f);
        std::vector<float>         per_slot_min_p(num_sampling, 0.f);
        std::vector<std::int32_t>  per_slot_top_k(num_sampling, 0);
        std::vector<std::uint32_t> per_slot_seed (num_sampling, 0u);

        bool any_topk_topp = false;
        {
            std::uint32_t sampler_off = 0;
            for (int r = 0; r < R; ++r) {
                const std::uint32_t ns =
                    (rns_view.size() > static_cast<std::size_t>(r)) ? h_rns[r] : 0u;
                const std::uint32_t lo = h_sptr[r];
                const std::uint32_t hi = h_sptr[r + 1];
                const std::uint32_t qo_lo = h_qo[r];
                for (std::uint32_t k = lo; k < hi; ++k) {
                    const std::uint32_t s_idx = sampler_off + (k - lo);
                    const std::uint32_t type =
                        (s_idx < types_view.size()) ? types_view[s_idx] : 1u;
                    per_slot_type[k] = type;
                    const float T = (s_idx < temp_view.size()) ? h_temp[s_idx] : 1.f;
                    const float Tp = (s_idx < top_p_view.size()) ? h_top_p[s_idx] : 1.f;
                    const float Mp = (s_idx < min_p_view.size()) ? h_min_p[s_idx] : 0.f;
                    const std::int32_t Tk_raw = (s_idx < top_k_view.size())
                        ? static_cast<std::int32_t>(h_top_k[s_idx]) : 0;
                    const std::int32_t Tk =
                        (Tk_raw == 0) ? engine.hf_config().vocab_size : Tk_raw;
                    const std::uint32_t s = (s_idx < seed_view.size()) ? h_seed[s_idx] : 0u;
                    per_slot_temp[k] = T;
                    per_slot_top_p[k] = Tp;
                    per_slot_min_p[k] = Mp;
                    per_slot_top_k[k] = Tk;
                    per_slot_seed[k] = s;
                    const bool is_token = pie_cuda_driver::is_token_sampler(type);
                    if (is_token) {
                        if ((Tk_raw > 0 || Tp < 1.f) && T > 0.f) any_topk_topp = true;
                        const std::uint32_t row = qo_lo + h_sidx[k];
                        if (row < static_cast<std::uint32_t>(N)) {
                            h_per_temp[row]  = T;
                            h_per_top_k[row] = Tk;
                            h_per_top_p[row] = Tp;
                            h_per_min_p[row] = Mp;
                            h_per_seed[row]  = s;
                        }
                    }
                }
                sampler_off += ns;
            }
        }

        // Per-slot → row mapping for the topk+top-p scatter. Built
        // unconditionally so the sampling-dispatch entry has a uniform
        // input shape.
        std::vector<std::int32_t> h_sample_idx(num_sampling, 0);
        {
            int k_g = 0;
            for (int r = 0; r < R; ++r) {
                const std::uint32_t qo_lo = h_qo[r];
                for (std::uint32_t k = h_sptr[r]; k < h_sptr[r + 1]; ++k, ++k_g) {
                    h_sample_idx[k_g] =
                        static_cast<std::int32_t>(qo_lo + h_sidx[k]);
                }
            }
        }

        const SamplingPlan sample_plan{
            any_topk_topp,
            std::span<const float>(h_per_temp),
            std::span<const float>(h_per_top_p),
            std::span<const float>(h_per_min_p),
            std::span<const std::int32_t>(h_per_top_k),
            std::span<const std::uint32_t>(h_per_seed),
            std::span<const std::int32_t>(h_sample_idx),
        };

        // ── prepare hook ────────────────────────────────────────
        // Always run the per-arch prepare phase first (when present).
        // For graph-capable archs this updates pinned host / device
        // plan state for the captured body to read. Lives outside any
        // capture region so the host work re-runs every fire.
        if (tm_on) tm.t_after_views_us = fire_now_us();
        if (forward_fn.prepare) {
            forward_fn.prepare(attn_ws, h_kvpp, R, is_pure_decode);
        }
        if (tm_on) tm.t_after_prepare_us = fire_now_us();

        // ── Upload sampling inputs (must precede capture) ──────
        // Per-row sampler params land in `pi.sample_*`. We do this on
        // the default stream so the captured body's sampling kernel —
        // launched on `cstream` — sees the fresh values via legacy
        // default-stream ordering. Doing the upload INSIDE capture
        // would dangling-reference the host-side `h_per_*` vectors on
        // replay (they're per-fire stack locals).
        upload_sampling_inputs(pi, sample_plan, N, /*stream=*/nullptr);
        if (tm_on) tm.t_after_uploads_us = fire_now_us();

        // ── Forward pass ────────────────────────────────────────
        // Graph-capture path activates only when the arch declares
        // itself graph-safe. The flag is the per-arch flip set in
        // main.cpp once the dispatch upgrades to graph-stable kernel
        // args (currently none — see ForwardFn::graph_safe).
        const bool try_graphs =
            ctx.graph_cache != nullptr && is_pure_decode && !have_custom_mask
            && forward_fn.graph_safe;
        if (try_graphs) {
            const ForwardGraphKey key{R};
            cudaGraphExec_t exec = ctx.graph_cache->get(key);
            if (exec == nullptr) {
                // First fire of this shape: capture. Body writes its
                // output to `ws` workspace buffers + `cache.k/v` pages.
                // Persistent inputs (pi.*) provide stable kernel-arg
                // pointers; the next replay reads new contents from the
                // same addresses, refreshed by `prepare` above.
                //
                // Bind cuBLAS to `cstream` for the duration of capture
                // so its kernel launches are recorded onto the captured
                // graph rather than slipping onto the default stream.
                // Relaxed mode allows cross-stream operations during
                // capture but only operations on the captured stream
                // (or streams joined to it) make it into the graph.
                cudaStream_t cstream = nullptr;
                CUDA_CHECK(cudaStreamCreateWithFlags(&cstream, cudaStreamNonBlocking));
                // Upload sampling params on `cstream` so the cudaMemcpyAsync
                // calls inside `dispatch_sampling` are recorded into the
                // graph rather than slipping onto the default stream.
                // For replay, the host-side handler re-uploads via the
                // same dispatch_sampling call on the same stream — the
                // captured memcpy nodes copy from the same host pointers
                // each time, which CUDA Graph handles via pinned-staging.
                cublas.set_stream(cstream);
                CUDA_CHECK(cudaStreamBeginCapture(cstream, cudaStreamCaptureModeRelaxed));
                forward_fn.body(
                    ws, kv_cache, attn_ws, cublas,
                    reinterpret_cast<const std::int32_t*>(pi.tokens.data()),
                    reinterpret_cast<const std::int32_t*>(pi.positions.data()),
                    pi.qo_indptr.data(), pi.kv_page_indices.data(),
                    pi.kv_page_indptr.data(), pi.kv_last_page_lens.data(),
                    h_qo, h_kvpp,
                    N, R, is_pure_decode, nullptr, nullptr,
                    use_slots ? slot_ids_h.data() : nullptr,
                    use_slots ? is_fresh_h.data() : nullptr,
                    use_slots ? pi.slot_ids.data() : nullptr);
                // Sampling kernel + on-device scatter live inside the
                // captured graph. `pi.sample_*` were uploaded on the
                // default stream just before this block; legacy-default-
                // stream semantics order them before the captured
                // kernel even though the upload itself is not captured.
                launch_sampling_kernel(
                    ws, pi.sampled.data(), pi, sample_plan,
                    N, num_sampling, engine.hf_config().vocab_size,
                    /*prng_offset=*/static_cast<std::uint64_t>(handled),
                    /*stream=*/cstream);
                cudaGraph_t graph = nullptr;
                CUDA_CHECK(cudaStreamEndCapture(cstream, &graph));
                cublas.set_stream(nullptr);
                CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
                cudaGraphDestroy(graph);
                cudaStreamDestroy(cstream);
                ctx.graph_cache->put(key, exec);
                const auto sz = ctx.graph_cache->size();
                if (ctx.verbose && (sz <= 4 || sz % 16 == 0)) {
                    std::cerr << "[pie-driver-cuda] graph captured: R=" << R
                              << " (cache size=" << sz << ")\n";
                }
            } else {
                CUDA_CHECK(cudaGraphLaunch(exec, /*stream=*/nullptr));
            }
        } else {
            forward_fn.body(
                ws, kv_cache, attn_ws, cublas,
                reinterpret_cast<const std::int32_t*>(pi.tokens.data()),
                reinterpret_cast<const std::int32_t*>(pi.positions.data()),
                pi.qo_indptr.data(), pi.kv_page_indices.data(),
                pi.kv_page_indptr.data(), pi.kv_last_page_lens.data(),
                /*qo_indptr_h=*/h_qo,
                /*kv_page_indptr_h=*/h_kvpp,
                N, R, is_pure_decode,
                have_custom_mask ? pi.custom_mask.data()        : nullptr,
                have_custom_mask ? pi.custom_mask_indptr.data() : nullptr,
                use_slots ? slot_ids_h.data() : nullptr,
                use_slots ? is_fresh_h.data() : nullptr,
                use_slots ? pi.slot_ids.data() : nullptr);
            launch_sampling_kernel(
                ws, pi.sampled.data(), pi, sample_plan,
                N, num_sampling, engine.hf_config().vocab_size,
                /*prng_offset=*/static_cast<std::uint64_t>(handled),
                /*stream=*/cublas.stream());
        }

        // Sample plan was built above the prepare hook (hoisted so the
        // sampling kernel can run inside the captured graph). The host
        // variables (`need_msgpack`, `per_slot_*`, `any_topk_topp`,
        // `h_per_*`, `h_sample_idx`) are still in scope here for the
        // response builder.

        // Only copy the first N entries — `pi.sampled` is sized for
        // max_workspace_tokens, but only [0, N) are valid this fire.
        // Async on the same stream the sampler ran on so it slots into
        // the stream's FIFO; we sync immediately after because the
        // response payload depends on these tokens. (Future work moves
        // the sync past the host-side response-prep so the host
        // and GPU can overlap.)
        if (tm_on) tm.t_after_launch_us = fire_now_us();
        std::vector<std::int32_t> all_sampled(N);
        CUDA_CHECK(cudaMemcpyAsync(all_sampled.data(), pi.sampled.data(),
                                   sizeof(std::int32_t) * N,
                                   cudaMemcpyDeviceToHost,
                                   cublas.stream()));
        CUDA_CHECK(cudaStreamSynchronize(cublas.stream()));
        if (tm_on) tm.t_after_sync_us = fire_now_us();

        // Flat-path arrays: token sampler is the only slot type allowed
        // here (need_msgpack would have flipped otherwise), so counts
        // align 1:1 with sampling slots.
        std::vector<std::uint32_t> per_request_counts(R);
        std::vector<std::uint32_t> sampled_tokens;
        sampled_tokens.reserve(num_sampling);
        for (int r = 0; r < R; ++r) {
            const std::uint32_t lo = h_sptr[r];
            const std::uint32_t hi = h_sptr[r + 1];
            const std::uint32_t qo_lo = h_qo[r];
            per_request_counts[r] = hi - lo;
            for (std::uint32_t k = lo; k < hi; ++k) {
                const std::uint32_t row = qo_lo + h_sidx[k];
                sampled_tokens.push_back(
                    static_cast<std::uint32_t>(all_sampled[row]));
            }
        }
        if (const char* dbg = std::getenv("PIE_DEBUG_SAMPLED");
            dbg && std::string(dbg) == "1") {
            std::cerr << "[pie-driver-cuda] sampled tokens for handled=" << handled
                      << ":";
            for (auto t : sampled_tokens) std::cerr << ' ' << t;
            std::cerr << '\n';
        }

        // Single structured response. The fast path (need_msgpack ==
        // false) populates only `tokens`; rich paths additionally fill
        // dists/logits/logprobs/entropies via the per-sampler sub-passes.
        std::vector<pie_driver::PerRequestOutput> per_req(R);

        if (need_msgpack) {
            const ResponseSubpassContext sub_ctx{
                ws,
                R, num_sampling, engine.hf_config().vocab_size,
                std::span<const std::uint32_t>(per_slot_type),
                std::span<const float>(per_slot_temp),
                std::span<const std::int32_t>(per_slot_top_k),
                qo_view, sptr_view, sidx_view, rns_view,
            };
            gather_raw_logits(sub_ctx, per_req);
            compute_entropy_slots(sub_ctx, per_req);
            compute_logprob_slots(sub_ctx, view, per_req);
            compute_dist_slots(sub_ctx, per_req);

            // Per-request token list. For non-spec requests this is the
            // token-typed slots' samples. For spec requests we walk the
            // verification block (cloned token samplers at the bonus +
            // each draft position) and produce the accepted prefix; the
            // inferlet's own samples for that request are discarded.
            for (int r = 0; r < R; ++r) {
                const std::uint32_t qo_lo = h_qo[r];
                auto& bucket = per_req[r].tokens;

                if (has_spec_drafts && verify_slot_start[r] >= 0) {
                    const int vs = verify_slot_start[r];
                    const int n_d = verify_n_drafts[r];
                    const int spec_lo = (r < static_cast<int>(spec_iptr_view.size()))
                        ? static_cast<int>(spec_iptr_view[r]) : 0;
                    std::vector<std::uint32_t> block(n_d + 1);
                    for (int j = 0; j <= n_d; ++j) {
                        const std::uint32_t row = qo_lo + h_sidx[vs + j];
                        block[j] = static_cast<std::uint32_t>(all_sampled[row]);
                    }
                    int match = 0;
                    for (int k = 0; k < n_d; ++k) {
                        if (block[k] == spec_tok_view[spec_lo + k]) match++;
                        else break;
                    }
                    bucket.assign(block.begin(), block.begin() + match + 1);
                } else {
                    const std::uint32_t lo = h_sptr[r];
                    const std::uint32_t hi = h_sptr[r + 1];
                    bucket.reserve(hi - lo);
                    for (std::uint32_t k = lo; k < hi; ++k) {
                        const std::uint32_t type = per_slot_type[k];
                        if (!pie_cuda_driver::is_token_sampler(type)) continue;
                        const std::uint32_t row = qo_lo + h_sidx[k];
                        bucket.push_back(static_cast<std::uint32_t>(all_sampled[row]));
                    }
                }
            }
        } else {
            // Token-only fast path: split sampled_tokens by per_request_counts.
            std::size_t cursor = 0;
            for (int r = 0; r < R; ++r) {
                const std::uint32_t n = per_request_counts[r];
                per_req[r].tokens.assign(
                    sampled_tokens.begin() + cursor,
                    sampled_tokens.begin() + cursor + n);
                cursor += n;
            }
        }

        ctx.response_builder.build(per_req, out_resp);

        if (tm_on) {
            tm.t_after_response_us = fire_now_us();
            tm.R = R;
            tm.N = N;
            // Per-fire timeline. Print every 50 fires after warmup to
            // avoid spam — first dozens are unrepresentative (kernel
            // JIT, plan first-fire-only costs).
            if (handled >= 50 && handled % 50 == 0) {
                const auto us = [&](std::uint64_t a, std::uint64_t b){
                    return (b > a) ? static_cast<double>(b - a) : 0.0;
                };
                std::cerr
                    << "[fire " << handled << " R=" << R << " N=" << N << "] "
                    << "parse=" << us(tm.t_start_us, tm.t_after_parse_us) << "us "
                    << "views=" << us(tm.t_after_parse_us, tm.t_after_views_us) << "us "
                    << "prep="  << us(tm.t_after_views_us, tm.t_after_prepare_us) << "us "
                    << "upld="  << us(tm.t_after_prepare_us, tm.t_after_uploads_us) << "us "
                    << "launch=" << us(tm.t_after_uploads_us, tm.t_after_launch_us) << "us "
                    << "sync="  << us(tm.t_after_launch_us, tm.t_after_sync_us) << "us "
                    << "resp="  << us(tm.t_after_sync_us, tm.t_after_response_us) << "us "
                    << "TOTAL=" << us(tm.t_start_us, tm.t_after_response_us) << "us\n";
            }
        }

        if (ctx.verbose && (handled <= 4 || handled % 100 == 0)) {
            std::cerr << "[pie-driver-cuda] req_id=" << req_id
                      << " R=" << R << " N=" << N
                      << " sampled=" << num_sampling
                      << " max_kv=" << max_kv_len << "\n";
        }
        return;

    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-cuda] fire_batch failed for req_id="
                  << req_id << ": " << e.what() << "\n";
        out_resp = pie_driver::PieForwardResponseView{};
        return;
    }
}

// ============================================================================
// TP follower service loop
// ============================================================================
//
// Symmetric counterpart of `handle_fire_batch` for ranks > 0:
//
//   * No shmem decode — the inputs arrive via NCCL broadcast from rank 0.
//   * No sampling — only rank 0 owns the response buffer + sampler RNG.
//   * No graph capture — the broadcast issues an h2d memcpy (`d_hdr`)
//     and the body's all-reduces aren't graph-captured today; staying
//     out of the graph path keeps semantics simple.
//
// The loop blocks on `ncclBroadcast` for the header. NCCL serialises ops
// per-comm, so a follower naturally idles until rank 0 issues the
// matching broadcast in `tp_broadcast_inputs`.
void tp_follower_serve(ForwardContext& ctx, std::atomic<bool>& stop) {
    if (ctx.tp_comm == nullptr) {
        std::cerr << "[pie-driver-cuda] tp_follower_serve: no tp_comm\n";
        return;
    }
    auto& pi      = ctx.inputs;
    auto& comm    = *ctx.tp_comm;
    auto* d_hdr   = tp_hdr_dev_buf();
    cudaStream_t stream = nullptr;
    std::uint64_t cpu_gate_seq = 0;

    // Sized lazily; R is at most max_workspace_tokens (one request per token).
    std::vector<std::uint32_t> h_qo, h_kvpp;

    while (!stop.load()) {
        tp_cpu_gate_wait(ctx.tp_cpu_gate_key, cpu_gate_seq, stop);
        // 1. Receive header.
        NCCL_CHECK(ncclBroadcast(d_hdr, d_hdr, sizeof(TpFireHeader),
                                 ncclChar, 0, comm.comm(), stream));
        TpFireHeader hdr{};
        CUDA_CHECK(cudaMemcpyAsync(&hdr, d_hdr, sizeof(hdr),
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaStreamSynchronize(stream));

        if (hdr.magic == TP_STOP_MAGIC) break;
        if (hdr.magic != TP_FIRE_MAGIC) {
            std::cerr << "[pie-driver-cuda] tp follower: unexpected header "
                      << "magic 0x" << std::hex << hdr.magic << std::dec
                      << "; aborting\n";
            break;
        }

        const int N = hdr.total_tokens;
        const int R = hdr.num_requests;
        const bool is_pure_decode = (hdr.is_pure_decode != 0);

        // 2. Receive payloads. Mirror order in `tp_broadcast_inputs`,
        //    grouped so NCCL submits the batch as a single op.
        const bool have_custom_mask = (hdr.mask_bytes > 0);
        NCCL_CHECK(ncclGroupStart());
        NCCL_CHECK(ncclBroadcast(pi.tokens.data(), pi.tokens.data(),
                                 static_cast<std::size_t>(N) * 4,
                                 ncclChar, 0, comm.comm(), stream));
        NCCL_CHECK(ncclBroadcast(pi.positions.data(), pi.positions.data(),
                                 static_cast<std::size_t>(N) * 4,
                                 ncclChar, 0, comm.comm(), stream));
        NCCL_CHECK(ncclBroadcast(pi.qo_indptr.data(), pi.qo_indptr.data(),
                                 static_cast<std::size_t>(R + 1) * 4,
                                 ncclChar, 0, comm.comm(), stream));
        NCCL_CHECK(ncclBroadcast(pi.kv_page_indptr.data(),
                                 pi.kv_page_indptr.data(),
                                 static_cast<std::size_t>(R + 1) * 4,
                                 ncclChar, 0, comm.comm(), stream));
        if (R > 0) {
            NCCL_CHECK(ncclBroadcast(pi.kv_last_page_lens.data(),
                                     pi.kv_last_page_lens.data(),
                                     static_cast<std::size_t>(R) * 4,
                                     ncclChar, 0, comm.comm(), stream));
        }
        if (hdr.kv_indices_count > 0) {
            NCCL_CHECK(ncclBroadcast(pi.kv_page_indices.data(),
                                     pi.kv_page_indices.data(),
                                     static_cast<std::size_t>(hdr.kv_indices_count) * 4,
                                     ncclChar, 0, comm.comm(), stream));
        }
        if (have_custom_mask) {
            NCCL_CHECK(ncclBroadcast(pi.custom_mask.data(),
                                     pi.custom_mask.data(),
                                     static_cast<std::size_t>(hdr.mask_bytes),
                                     ncclChar, 0, comm.comm(), stream));
            NCCL_CHECK(ncclBroadcast(pi.custom_mask_indptr.data(),
                                     pi.custom_mask_indptr.data(),
                                     static_cast<std::size_t>(hdr.mask_indptr_count) * 4,
                                     ncclChar, 0, comm.comm(), stream));
        }
        const bool have_slot_ids = (hdr.has_slot_ids != 0) && R > 0;
        if (have_slot_ids) {
            NCCL_CHECK(ncclBroadcast(pi.slot_ids.data(), pi.slot_ids.data(),
                                     static_cast<std::size_t>(R) * 4,
                                     ncclChar, 0, comm.comm(), stream));
            NCCL_CHECK(ncclBroadcast(pi.is_fresh.data(), pi.is_fresh.data(),
                                     static_cast<std::size_t>(R),
                                     ncclChar, 0, comm.comm(), stream));
        }
        NCCL_CHECK(ncclGroupEnd());

        // 3. Pull the host views of qo/kv_page indptrs for the per-arch
        // attention planner (lives outside the captured kernel sequence).
        h_qo.resize(R + 1);
        h_kvpp.resize(R + 1);
        CUDA_CHECK(cudaMemcpyAsync(h_qo.data(), pi.qo_indptr.data(),
                                   static_cast<std::size_t>(R + 1) * 4,
                                   cudaMemcpyDeviceToHost, stream));
        CUDA_CHECK(cudaMemcpyAsync(h_kvpp.data(), pi.kv_page_indptr.data(),
                                   static_cast<std::size_t>(R + 1) * 4,
                                   cudaMemcpyDeviceToHost, stream));
        std::vector<std::int32_t> h_slot_ids;
        std::vector<std::uint8_t> h_is_fresh;
        if (have_slot_ids) {
            h_slot_ids.resize(R);
            h_is_fresh.resize(R);
            CUDA_CHECK(cudaMemcpyAsync(h_slot_ids.data(), pi.slot_ids.data(),
                                       static_cast<std::size_t>(R) * 4,
                                       cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaMemcpyAsync(h_is_fresh.data(), pi.is_fresh.data(),
                                       static_cast<std::size_t>(R),
                                       cudaMemcpyDeviceToHost, stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));

        // 4. Run the same forward function as rank 0. The all-reduces
        // inside synchronise both ranks; we don't sample or write a
        // response — that's rank-0-only.
        if (ctx.forward_fn.prepare) {
            ctx.forward_fn.prepare(ctx.attn_ws, h_kvpp.data(), R, is_pure_decode);
        }
        // Mirror rank 0's graph capture/replay decision so NCCL ops
        // inside the body record on both ranks simultaneously (otherwise
        // rank 0 would record while rank 1 executes, deadlocking the
        // first capture). The same `(R)` shape key keeps the per-rank
        // graph caches in lockstep; the captured graph on rank 1 has no
        // sampling / response work, just the forward kernels + NCCL.
        const bool try_graphs =
            ctx.graph_cache != nullptr && is_pure_decode && !have_custom_mask
            && ctx.forward_fn.graph_safe;
        if (try_graphs) {
            const ForwardGraphKey key{R};
            cudaGraphExec_t exec = ctx.graph_cache->get(key);
            if (exec == nullptr) {
                cudaStream_t cstream = nullptr;
                CUDA_CHECK(cudaStreamCreateWithFlags(&cstream, cudaStreamNonBlocking));
                CUDA_CHECK(cudaStreamBeginCapture(cstream, cudaStreamCaptureModeRelaxed));
                ctx.forward_fn.body(
                    ctx.ws, ctx.kv_cache, ctx.attn_ws, ctx.cublas,
                    reinterpret_cast<const std::int32_t*>(pi.tokens.data()),
                    reinterpret_cast<const std::int32_t*>(pi.positions.data()),
                    pi.qo_indptr.data(), pi.kv_page_indices.data(),
                    pi.kv_page_indptr.data(), pi.kv_last_page_lens.data(),
                    h_qo.data(), h_kvpp.data(),
                    N, R, is_pure_decode, nullptr, nullptr,
                    have_slot_ids ? h_slot_ids.data() : nullptr,
                    have_slot_ids ? h_is_fresh.data() : nullptr,
                    have_slot_ids ? pi.slot_ids.data() : nullptr);
                cudaGraph_t graph = nullptr;
                CUDA_CHECK(cudaStreamEndCapture(cstream, &graph));
                CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
                cudaGraphDestroy(graph);
                cudaStreamDestroy(cstream);
                ctx.graph_cache->put(key, exec);
            } else {
                CUDA_CHECK(cudaGraphLaunch(exec, /*stream=*/nullptr));
            }
        } else {
            ctx.forward_fn.body(
                ctx.ws, ctx.kv_cache, ctx.attn_ws, ctx.cublas,
                reinterpret_cast<const std::int32_t*>(pi.tokens.data()),
                reinterpret_cast<const std::int32_t*>(pi.positions.data()),
                pi.qo_indptr.data(), pi.kv_page_indices.data(),
                pi.kv_page_indptr.data(), pi.kv_last_page_lens.data(),
                h_qo.data(), h_kvpp.data(),
                N, R, is_pure_decode,
                have_custom_mask ? pi.custom_mask.data()        : nullptr,
                have_custom_mask ? pi.custom_mask_indptr.data() : nullptr,
                have_slot_ids ? h_slot_ids.data() : nullptr,
                have_slot_ids ? h_is_fresh.data() : nullptr,
                have_slot_ids ? pi.slot_ids.data() : nullptr);
        }
    }
}

void tp_send_shutdown(NcclComm& comm, const std::string& cpu_gate_key) {
    tp_cpu_gate_notify(cpu_gate_key);
    auto* d_hdr = tp_hdr_dev_buf();
    cudaStream_t stream = nullptr;
    TpFireHeader hdr{TP_STOP_MAGIC, 0, 0, 0, 0, 0, 0, 0};
    CUDA_CHECK(cudaMemcpyAsync(d_hdr, &hdr, sizeof(hdr),
                               cudaMemcpyHostToDevice, stream));
    NCCL_CHECK(ncclBroadcast(d_hdr, d_hdr, sizeof(hdr), ncclChar, 0,
                             comm.comm(), stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
}

}  // namespace pie_cuda_driver
