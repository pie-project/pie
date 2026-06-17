#include "executor/executor.hpp"

#include <algorithm>
#include <array>
#include <chrono>
#include <limits>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <iomanip>
#include <iostream>
#include <map>
#include <sstream>
#include <memory>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <ggml.h>
#include <ggml-alloc.h>
#include <ggml-backend.h>

#include "arch_spec.hpp"
#include "graph_common.hpp"
#include "graph_gemma4.hpp"
#include "graph_qwen3.hpp"
#include "graph_phi3small.hpp"
#include "graph_phi3_5moe.hpp"
#include "graph_qwen3_5.hpp"
#include "kv_cache_quant.hpp"
#include "plan.hpp"
#include "sampler.hpp"

namespace pie_portable_driver {

// Graph cache (P7 + P7b). One slot, keyed by graph topology signature.
// Pure-decode batches: max_n_kv is rounded up to a page boundary in
// plan_() so consecutive decode steps land on the same shape; packed
// gather + mask sizes follow that bucketed max_n_kv. Slow-path graphs
// (prefill, custom attention masks, mixed n_tokens) cache too — useful
// when consecutive prefill batches share shape (e.g. replicate tests
// or identical-prompt re-runs).
struct Executor::GraphCache {
    // Signature fields — every value the graph topology depends on.
    bool          valid              = false;
    PieArch       arch{};
    bool          pure_decode        = false;
    std::int32_t  n_request          = 0;
    std::int32_t  total_n_tokens     = 0;
    std::int32_t  max_n_kv           = 0;
    bool          all_greedy         = false;
    bool          uniform_top_sample = false;
    std::int32_t  uniform_top_k      = 0;
    const Adapter* adapter           = nullptr;
    // Slow-path only: per-request mask + gather shapes. Empty on pure-
    // decode (where the packed tensors are fully determined by
    // n_request + max_n_kv).
    std::vector<std::int32_t> n_tokens_pad_per_req;
    std::vector<std::int32_t> n_kv_per_req;
    // Pure-decode paged-attn only: total KV pages across the batch. The
    // `page_indices` tensor is allocated to this exact size, but unlike
    // page_indptr/last_page_lens (sized purely from n_request) it can
    // change between batches that otherwise share the same signature
    // (e.g. a request's KV usage crosses a page boundary without moving
    // max_n_kv). Must invalidate the cache on change, else upload_graph_inputs
    // writes a larger plan.page_indices_i32 into the smaller cached tensor.
    std::int32_t total_pages_in_batch = 0;
    // State-bearing archs (Qwen 3.5) bake the per-request state_slot
    // offset into the graph view. The cache must invalidate when slots
    // change, even if everything else matches.
    std::vector<std::int32_t> state_slot_per_req;

    ggml_context* ctx                = nullptr;
    GraphResult   result;

    bool matches(PieArch a,
                 const Executor::BatchPlan& plan) const noexcept {
        if (!valid) return false;
        if (arch != a) return false;
        if (pure_decode != plan.pure_decode) return false;
        if (n_request != static_cast<std::int32_t>(plan.reqs.size())) return false;
        if (total_n_tokens != plan.total_n_tokens) return false;
        if (max_n_kv != plan.max_n_kv) return false;
        if (all_greedy != plan.all_greedy) return false;
        if (uniform_top_sample != plan.uniform_top_sample) return false;
        if (uniform_top_k != plan.uniform_top_k) return false;
        if (adapter != plan.active_adapter) return false;
        if (plan.pure_decode) {
            if (total_pages_in_batch !=
                static_cast<std::int32_t>(plan.page_indices_i32.size())) return false;
        }
        if (!plan.pure_decode) {
            // Slow path: per-request shapes must match exactly.
            if (n_tokens_pad_per_req.size() != plan.reqs.size()) return false;
            for (std::size_t r = 0; r < plan.reqs.size(); ++r) {
                if (n_tokens_pad_per_req[r] != plan.reqs[r].n_tokens_pad) return false;
                if (n_kv_per_req[r]         != plan.reqs[r].n_kv)         return false;
            }
        }
        if (!state_slot_per_req.empty()) {
            if (state_slot_per_req.size() != plan.reqs.size()) return false;
            for (std::size_t r = 0; r < plan.reqs.size(); ++r) {
                if (state_slot_per_req[r] != plan.reqs[r].state_slot) return false;
            }
        }
        return true;
    }

    void store_key(PieArch a,
                   const Executor::BatchPlan& plan) {
        valid              = true;
        arch               = a;
        pure_decode        = plan.pure_decode;
        n_request          = static_cast<std::int32_t>(plan.reqs.size());
        total_n_tokens     = plan.total_n_tokens;
        max_n_kv           = plan.max_n_kv;
        all_greedy         = plan.all_greedy;
        uniform_top_sample = plan.uniform_top_sample;
        uniform_top_k      = plan.uniform_top_k;
        adapter            = plan.active_adapter;
        if (plan.pure_decode) {
            n_tokens_pad_per_req.clear();
            n_kv_per_req.clear();
            total_pages_in_batch = static_cast<std::int32_t>(plan.page_indices_i32.size());
        } else {
            n_tokens_pad_per_req.resize(plan.reqs.size());
            n_kv_per_req.resize(plan.reqs.size());
            for (std::size_t r = 0; r < plan.reqs.size(); ++r) {
                n_tokens_pad_per_req[r] = plan.reqs[r].n_tokens_pad;
                n_kv_per_req[r]         = plan.reqs[r].n_kv;
            }
        }
        // Stash slot ids only when any request actually carries one.
        bool any_slot = false;
        for (const auto& rp : plan.reqs) if (rp.state_slot >= 0) { any_slot = true; break; }
        if (any_slot) {
            state_slot_per_req.resize(plan.reqs.size());
            for (std::size_t r = 0; r < plan.reqs.size(); ++r) {
                state_slot_per_req[r] = plan.reqs[r].state_slot;
            }
        } else {
            state_slot_per_req.clear();
        }
    }

    void release() {
        if (ctx) {
            ggml_free(ctx);
            ctx = nullptr;
        }
        result = GraphResult{};
        valid = false;
    }

    ~GraphCache() { release(); }
};

namespace {

// Returns true if any slot produced a special-sampler payload
// (Distribution / RawLogits / Logprob / Logprobs / Entropy).
inline bool any_slot_special(const std::vector<SlotOutput>& slots) {
    for (const auto& s : slots) {
        if (s.has_dist || !s.raw_logits.empty()
            || !s.logprobs.empty() || s.has_entropy) {
            return true;
        }
    }
    return false;
}

// Sample every slot for one request (after BRLE logit-mask application).
std::vector<SlotOutput> sample_request_slots(const Executor::ReqPlan& rp,
                                             float* slots_logits_base,
                                             std::int32_t n_slots,
                                             std::int32_t vocab_size) {
    std::vector<SlotOutput> out(n_slots);
    // Each slot can have its own sampler kind (an inferlet may attach
    // Argmax + RawLogits + Distribution + ... at the same position in
    // one fire). Fall back to `rp.sampler` when the per-slot array
    // doesn't cover an index — that only happens for spec-decode bonus
    // slots, which all share the request's primary sampler.
    for (std::int32_t s = 0; s < n_slots; ++s) {
        float* row = slots_logits_base + static_cast<std::size_t>(s) * vocab_size;
        if (!rp.logit_mask_runs.empty()) {
            apply_brle_logit_mask(row, vocab_size,
                                  rp.logit_mask_runs.data(),
                                  rp.logit_mask_runs.size());
        }
        // Per-token additive logit bias: applied after the mask, before
        // sampling. Out-of-range token ids are ignored.
        for (std::size_t b = 0; b < rp.logit_bias_tokens.size(); ++b) {
            const std::uint32_t tok = rp.logit_bias_tokens[b];
            if (tok < static_cast<std::uint32_t>(vocab_size)) {
                row[tok] += rp.logit_bias_values[b];
            }
        }
        const SamplerParams& sp =
            s < static_cast<std::int32_t>(rp.samplers.size())
                ? rp.samplers[static_cast<std::size_t>(s)]
                : rp.sampler;
        sample_slot(row, vocab_size, sp, out[s]);
    }
    return out;
}

// Resolve per-request sampled slots into a final SamplerOutput. Three
// modes:
//   - special: hand through per-slot special payloads
//   - spec decode: walk drafts vs predictions, accept matching prefix +
//     1 bonus token
//   - plain: emit the single sampled token
void resolve_request_output(const Executor::ReqPlan& rp,
                            std::vector<SlotOutput>&& slot_out,
                            SamplerOutput& dst) {
    // Prefill-only requests carry no sampler slots — emit nothing.
    if (slot_out.empty()) {
        return;
    }
    if (any_slot_special(slot_out)) {
        dst.special_slots = std::move(slot_out);
        return;
    }
    const std::int32_t n_drafts =
        static_cast<std::int32_t>(rp.draft_tokens.size());
    if (n_drafts == 0) {
        // No driver-side drafts. Emit every sampled slot. The SDK uses
        // multi-position sampling for speculative-decoding verify
        // passes that route drafts through `input_tokens` rather than
        // the `spec_token_ids` channel, and expects one token per slot.
        dst.tokens.reserve(slot_out.size());
        for (auto& s : slot_out) {
            dst.tokens.push_back(s.token);
        }
        return;
    }
    // Spec verifier walk. Slot k predicts the token that should follow
    // draft k (slot 0 predicts the first draft itself). Accept the
    // matching prefix + 1 bonus.
    dst.tokens.reserve(n_drafts + 1);
    bool all_match = true;
    for (std::int32_t k = 0; k < n_drafts; ++k) {
        if (slot_out[k].token == rp.draft_tokens[k]) {
            dst.tokens.push_back(rp.draft_tokens[k]);
        } else {
            dst.tokens.push_back(slot_out[k].token);  // bonus replacing rejected
            all_match = false;
            break;
        }
    }
    if (all_match) {
        // Every draft accepted — append the bonus from the last slot.
        dst.tokens.push_back(slot_out[n_drafts].token);
    }
}

// Top-level batch sampler: walk the per-request slot output starting at
// `all_logits` (laid out flat as [n_slots, vocab_size]).
std::vector<SamplerOutput> sample_batch(const Executor::BatchPlan& plan,
                                        float* all_logits,
                                        std::int32_t vocab_size) {
    const std::int32_t n_req = static_cast<std::int32_t>(plan.reqs.size());
    std::vector<SamplerOutput> sampled(n_req);
    std::int32_t slot_off = 0;
    for (std::int32_t r = 0; r < n_req; ++r) {
        const auto& rp = plan.reqs[r];
        const std::int32_t n_slots =
            static_cast<std::int32_t>(rp.sampling_positions.size());
        float* base = all_logits + static_cast<std::size_t>(slot_off) * vocab_size;
        auto slot_out = sample_request_slots(rp, base, n_slots, vocab_size);
        resolve_request_output(rp, std::move(slot_out), sampled[r]);
        slot_off += n_slots;
    }
    return sampled;
}

// GPU uniform-top-sample fast path: per-slot list of top-K (prob, idx)
// pairs already sorted descending and temperature-softmaxed by the
// graph. Walk per-slot, dispatch to sampler::sample_token_from_topk,
// and stitch through resolve_request_output (which handles spec-decode).
std::vector<SamplerOutput> sample_batch_uniform_top(
        const Executor::BatchPlan& plan,
        const std::int32_t* top_idx_flat,   // [K, n_slots]
        const float*        top_prob_flat,  // [K, n_slots]
        std::int32_t        K) {
    const std::int32_t n_req = static_cast<std::int32_t>(plan.reqs.size());
    std::vector<SamplerOutput> sampled(n_req);
    std::int32_t slot_off = 0;
    for (std::int32_t r = 0; r < n_req; ++r) {
        const auto& rp = plan.reqs[r];
        const std::int32_t n_slots =
            static_cast<std::int32_t>(rp.sampling_positions.size());
        std::vector<SlotOutput> slot_out(n_slots);
        for (std::int32_t s = 0; s < n_slots; ++s) {
            const std::int32_t global_slot = slot_off + s;
            const std::size_t base = static_cast<std::size_t>(global_slot) * K;
            slot_out[s].token = sample_token_from_topk(
                top_idx_flat + base, top_prob_flat + base, K,
                rp.sampler, static_cast<std::uint64_t>(global_slot));
        }
        resolve_request_output(rp, std::move(slot_out), sampled[r]);
        slot_off += n_slots;
    }
    return sampled;
}

// Greedy fast path: every slot's token came from a GPU argmax; we just
// stitch the i32 ids back into per-request SamplerOutput, running the
// spec-decode verifier walk where applicable. Skips the per-slot logit
// mask + softmax + sort the host-side sampler does.
std::vector<SamplerOutput> sample_batch_greedy(const Executor::BatchPlan& plan,
                                               const std::int32_t* slot_tokens) {
    const std::int32_t n_req = static_cast<std::int32_t>(plan.reqs.size());
    std::vector<SamplerOutput> sampled(n_req);
    std::int32_t slot_off = 0;
    for (std::int32_t r = 0; r < n_req; ++r) {
        const auto& rp = plan.reqs[r];
        const std::int32_t n_slots =
            static_cast<std::int32_t>(rp.sampling_positions.size());
        std::vector<SlotOutput> slot_out(n_slots);
        for (std::int32_t s = 0; s < n_slots; ++s) {
            slot_out[s].token = static_cast<std::uint32_t>(slot_tokens[slot_off + s]);
        }
        // resolve_request_output handles the plain (single-token) and
        // spec-decode (verifier walk) cases identically to the slow path.
        resolve_request_output(rp, std::move(slot_out), sampled[r]);
        slot_off += n_slots;
    }
    return sampled;
}

KvCachePaged build_kv_for_(Model& model,
                            std::int32_t total_pages,
                            std::int32_t page_size,
                            const std::string& kv_cache_dtype) {
    const auto& h = model.hparams();
    auto quant_format = kv_cache_quant_format_from_string(kv_cache_dtype);
    if (h.arch == PieArch::Gemma4) {
        // Per-layer head_dim AND kv_heads. Sliding layers carry
        // [num_key_value_heads, head_dim]; full layers carry
        // [head_dim_global] and (for Gemma 4 31B / 26B-A4B alt-attention)
        // num_global_key_value_heads instead of num_key_value_heads.
        std::vector<std::int32_t> per_layer_dim(h.num_hidden_layers, h.head_dim);
        std::vector<std::int32_t> per_layer_kvh(h.num_hidden_layers,
                                                h.num_key_value_heads);
        for (std::int32_t i = 0; i < h.num_hidden_layers; ++i) {
            const bool is_full = !h.layer_types.empty()
                && h.layer_types[i] == 'g';
            per_layer_dim[i] = is_full ? h.gemma4_head_dim_global : h.head_dim;
            if (is_full && h.num_global_key_value_heads > 0) {
                per_layer_kvh[i] = h.num_global_key_value_heads;
            }
        }
        return KvCachePaged(model.backend(),
                             std::move(per_layer_kvh),
                             std::move(per_layer_dim),
                             total_pages, page_size,
                             GGML_TYPE_F16,
                             std::move(quant_format));
    }
    return KvCachePaged(model.backend(),
                        h.num_hidden_layers,
                        h.num_key_value_heads,
                        h.head_dim,
                        total_pages, page_size,
                        GGML_TYPE_F16,
                        std::move(quant_format));
}

}  // namespace

// =============================================================================
// Executor
// =============================================================================

Executor::Executor(Model& model,
                   std::int32_t total_pages,
                   std::int32_t page_size,
                   std::string kv_cache_dtype)
    : model_(model),
      kv_(build_kv_for_(model, total_pages, page_size, kv_cache_dtype)),
      cache_(std::make_unique<GraphCache>()) {
    // Qwen 3.5 / 3.6 needs a recurrent-state cache for its linear-
    // attention layers. Allocate one slot per concurrent context the
    // batching config admits.
    const auto& h = model.hparams();
    if (h.arch == PieArch::Qwen3_5) {
        std::vector<std::int32_t> linear_layers;
        for (std::int32_t i = 0; i < h.num_hidden_layers; ++i) {
            if (i < static_cast<std::int32_t>(h.layer_types.size()) &&
                h.layer_types[i] == 'l') {
                linear_layers.push_back(i);
            }
        }
        const std::int32_t conv_dim =
            2 * h.qwen35_linear_num_k_heads * h.qwen35_linear_k_head_dim
            +     h.qwen35_linear_num_v_heads * h.qwen35_linear_v_head_dim;
        // Keep recurrent-state slots bounded so we don't burn arbitrary GPU RAM
        // on idle slots.
        const std::int32_t n_slots = 64;
        state_ = std::make_unique<StateCache>(
            model.backend(), n_slots, linear_layers,
            h.qwen35_linear_num_v_heads,
            h.qwen35_linear_k_head_dim,
            h.qwen35_linear_v_head_dim,
            conv_dim,
            h.qwen35_linear_conv_kernel);
    }
    // Build the backend list for the multi-backend scheduler. Order
    // matters: sched picks the FIRST backend in the list that supports
    // each op, so the primary GPU backend gets first dibs and CPU is
    // the fallback only for ops the primary lacks (e.g. ggml-vulkan's
    // missing CPY pipelines). When the primary is already CPU, the
    // sched is single-backend and acts as a direct executor.
    std::vector<ggml_backend_t> backends;
    backends.push_back(model.backend());
    if (auto* cpu_fb = model.cpu_fallback()) {
        backends.push_back(cpu_fb);
    }
    // op_offload=false: sched only splits graphs when an op is genuinely
    // unsupported by the primary backend. With op_offload=true, sched
    // may proactively offload "small" ops to a faster backend in the
    // list, which on Vulkan triggers extra GPU↔CPU transfers per layer
    // and was observed to 4x decode latency on Qwen3-0.6B (3 ms → 12 ms).
    // Since CPU is strictly the fallback here (never preferred over
    // Vulkan/CUDA), proactive offload is the wrong default for us.
    sched_ = ggml_backend_sched_new(
        backends.data(), /*bufts=*/nullptr,
        static_cast<int>(backends.size()),
        GRAPH_MAX_NODES,
        /*parallel=*/false,
        /*op_offload=*/false);
    if (!sched_) {
        throw std::runtime_error("forward: ggml_backend_sched_new failed");
    }
}

Executor::~Executor() {
    // Release cached graph context BEFORE the scheduler: sched owns
    // the backend buffers that back the cached graph's tensors.
    if (cache_) cache_->release();
    if (sched_) ggml_backend_sched_free(sched_);
}

void Executor::log_timings(const char* label) const {
    if (timings_.n_calls == 0) return;
    const double n = static_cast<double>(timings_.n_calls);
    auto pct = [&](std::uint64_t b) {
        return timings_.total_us
            ? 100.0 * static_cast<double>(b) / static_cast<double>(timings_.total_us)
            : 0.0;
    };
    auto avg = [&](std::uint64_t b) {
        return static_cast<double>(b) / n / 1000.0;  // ms
    };

    std::ostream& o = std::cerr;
    o << std::fixed << std::setprecision(3);
    o << "[forward.timings] " << label << " — "
      << timings_.n_calls << " call(s), avg "
      << avg(timings_.total_us) << " ms / call\n";
    o << "  plan      : " << avg(timings_.plan_us)        << " ms ("
                          << pct(timings_.plan_us)        << "%)\n";
    o << "  graph_bld : " << avg(timings_.graph_build_us) << " ms ("
                          << pct(timings_.graph_build_us) << "%)\n";
    o << "  graph_alc : " << avg(timings_.graph_alloc_us) << " ms ("
                          << pct(timings_.graph_alloc_us) << "%)\n";
    o << "  upload    : " << avg(timings_.upload_us)      << " ms ("
                          << pct(timings_.upload_us)      << "%)\n";
    o << "  compute   : " << avg(timings_.compute_us)     << " ms ("
                          << pct(timings_.compute_us)     << "%)\n";
    o << "  logits_dl : " << avg(timings_.logits_dl_us)   << " ms ("
                          << pct(timings_.logits_dl_us)   << "%)\n";
    o << "  sample    : " << avg(timings_.sample_us)      << " ms ("
                          << pct(timings_.sample_us)      << "%)\n";
    o << "  resp_pack : " << avg(timings_.response_pack_us) << " ms ("
                          << pct(timings_.response_pack_us) << "%)\n";
    o.unsetf(std::ios_base::floatfield);
}

// -----------------------------------------------------------------------------
// Plan: wire → BatchPlan (real page-table)
// -----------------------------------------------------------------------------

Executor::BatchPlan Executor::plan_(const pie_driver::PieForwardRequestView& req) {
    const auto& hpar = model_.hparams();
    const ArchSpec spec = arch_spec_for(hpar.arch, hpar);
    // Slow-only path: skip the in-graph `ggml_top_k` (which would emit
    // `ggml_argsort` over the full vocab) and instead download raw
    // logits to host for sampling. Triggered for arches whose graph
    // builder demands it AND when the primary backend can't run a
    // vocab-sized argsort.
    //
    //   * Qwen 3.5 still runs per-request (state-bearing linear layers
    //     require per-request slot views) and emits only `logits`.
    //   * Gemma 4 used to live here; the M11 packed-decode path now
    //     lets sliding-attn layers go through flash_attn_ext (with the
    //     SWA-clipped mask) and full-attn layers (head_dim=512, beyond
    //     flash_attn_ext) through a batched manual SDPA, so it no
    //     longer forces slow-only.
    //   * `!supports_in_graph_topk()` catches ggml-vulkan, whose
    //     argsort kernel caps at 1024 cols — without this, sched
    //     would round-trip the sort through CPU per token (~9 ms of
    //     wasted GPU↔CPU latency).
    const bool slow_only = hpar.arch == PieArch::Qwen3_5 ||
                           !model_.supports_in_graph_topk();

    const PlanArrays arrays = extract_plan_arrays(req);
    validate_plan_top_level(arrays);
    if (state_ && arrays.n_request > 0 &&
        arrays.rs_slot_ids.size() != static_cast<std::size_t>(arrays.n_request)) {
        throw std::runtime_error(
            "plan: rs_cache forward missing runtime-assigned slot ids");
    }

    BatchPlan plan;
    plan.total_n_tokens = arrays.total_n_tokens;
    plan.tokens_i32.resize(plan.total_n_tokens);
    plan.positions_i32.resize(plan.total_n_tokens);
    plan.kv_idxs_i64.resize(plan.total_n_tokens);
    // sampling_pos_i32 is FLAT across all requests' slots; reserve a
    // generous upper bound (handles spec decode's 1 + n_drafts case).
    plan.sampling_pos_i32.reserve(arrays.n_request * 4);
    plan.reqs.reserve(arrays.n_request);

    // M9 LoRA: at most one adapter per fire_batch in v1.
    const std::int64_t active_adapter_id = resolve_active_adapter_id(arrays);

    const std::int32_t page_size   = kv_.page_size();
    const std::int32_t total_pages = kv_.total_pages();
    for (std::int32_t r = 0; r < arrays.n_request; ++r) {
        plan_single_request(arrays, r, page_size, total_pages, spec, plan);
    }
    if (state_ && arrays.rs_slot_flags.size() == static_cast<std::size_t>(arrays.n_request)) {
        for (std::int32_t r = 0; r < arrays.n_request; ++r) {
            if ((arrays.rs_slot_flags[r] & 1u) != 0 && plan.reqs[r].state_slot >= 0) {
                state_->zero_slot(plan.reqs[r].state_slot);
            }
        }
    }

    // M11 packed-decode fast path: all-decode (n_tokens == 1) batches with
    // no custom masks fuse into one attn call per layer.
    plan.pure_decode = !plan.reqs.empty() && !arrays.batch_has_attn_masks;
    plan.max_n_kv = 0;
    for (const auto& rp : plan.reqs) {
        plan.max_n_kv = std::max(plan.max_n_kv, rp.n_kv);
        if (rp.n_tokens != 1) plan.pure_decode = false;
    }
    if (slow_only) plan.pure_decode = false;
    if (plan.pure_decode) {
        // Bucket max_n_kv up to a kv-page boundary so consecutive decode
        // steps within the same page land on identical graph topology
        // and hit the cache in compute_(). Padding rows are masked
        // (-INF) and gather entries default to 0 — masked-out entries
        // contribute exp(-INF)=0 to the softmax, so output is unchanged.
        const std::int32_t bucket = kv_.page_size();
        plan.max_n_kv = ((plan.max_n_kv + bucket - 1) / bucket) * bucket;
        // Gemma 4 has mixed sliding+full-attention layers, so the
        // packing builder must emit BOTH a sliding-clipped mask (for
        // 's' layers) and a full-context mask (for 'g' layers).
        const bool need_full_mask = hpar.arch == PieArch::Gemma4;
        build_pure_decode_packing(plan, arrays.n_request, kv_.page_size(),
                                  spec.sliding_window, need_full_mask);
    }

    // GPU-greedy detection: EVERY slot (across all requests) is sampled
    // by argmax (temperature ≤ ε) with a token-producing sampler kind,
    // and no request applies a logit mask. A request with a special
    // sampler (Distribution / RawLogits / Logprob / Logprobs / Entropy)
    // on any slot disqualifies the fast path even if other slots are
    // greedy — the graph emits one logits output per slot, but the
    // GPU-greedy graph replaces logits with argmax(token id) and
    // special samplers need raw logits to compute their payloads.
    auto sampler_is_greedy_token = [](const SamplerParams& s) {
        const bool greedy_temp = s.temperature <= 1e-5f;
        const bool token_producing =
               s.type == SamplerType::Multinomial
            || s.type == SamplerType::TopK
            || s.type == SamplerType::TopP
            || s.type == SamplerType::MinP
            || s.type == SamplerType::TopKTopP;
        return greedy_temp && token_producing;
    };
    plan.all_greedy = !plan.reqs.empty() && !slow_only;
    for (const auto& rp : plan.reqs) {
        if (!rp.logit_mask_runs.empty()) { plan.all_greedy = false; break; }
        bool all_slots_greedy = !rp.samplers.empty();
        for (const auto& s : rp.samplers) {
            if (!sampler_is_greedy_token(s)) { all_slots_greedy = false; break; }
        }
        if (!all_slots_greedy) { plan.all_greedy = false; break; }
    }

    // GPU uniform-top-sample detection (non-greedy fast path). All slots
    // must use the same temperature, none can have a logit mask, and
    // none can be Multinomial (which needs the full vocab distribution)
    // or any special sampler kind. Per-slot top_k / top_p / min_p
    // remain heterogeneous — they're applied host-side on the downloaded
    // top-K list.
    if (!plan.all_greedy && !plan.reqs.empty() && !slow_only) {
        // Anchor temperature on the first slot of the first request that
        // has samplers — prefill-only requests have empty `samplers`.
        const SamplerParams* anchor = nullptr;
        for (const auto& rp : plan.reqs) {
            if (!rp.samplers.empty()) { anchor = &rp.samplers[0]; break; }
        }
        bool ok = anchor != nullptr;
        std::int32_t k_max = 0;
        for (const auto& rp : plan.reqs) {
            if (!rp.logit_mask_runs.empty()) { ok = false; break; }
            for (const auto& s : rp.samplers) {
                if (s.temperature <= 1e-5f
                    || s.temperature != anchor->temperature
                    || s.type == SamplerType::Multinomial
                    || s.type == SamplerType::Distribution
                    || s.type == SamplerType::RawLogits
                    || s.type == SamplerType::Logprob
                    || s.type == SamplerType::Logprobs
                    || s.type == SamplerType::Entropy) {
                    ok = false;
                    break;
                }
                const std::int32_t k = s.top_k > 0
                    ? static_cast<std::int32_t>(s.top_k) : 0;
                if (k > k_max) k_max = k;
            }
            if (!ok) break;
        }
        if (ok) {
            // Default K of 256 covers >99% of nucleus mass for typical
            // top-p≥0.9 traffic on transformer models. Bump if any slot
            // explicitly asked for more.
            constexpr std::int32_t kDefaultK = 256;
            const std::int32_t v = hpar.vocab_size;
            std::int32_t k = std::max(k_max, kDefaultK);
            if (k > v) k = v;
            plan.uniform_top_sample = true;
            plan.uniform_top_k      = k;
        }
    }

    // Resolve the active adapter via the pool. If lookup fails, the
    // adapter wasn't registered yet — fall through to base-model behavior.
    if (active_adapter_id >= 0 && adapters_) {
        plan.active_adapter = adapters_->get(
            static_cast<std::uint64_t>(active_adapter_id));
        if (!plan.active_adapter) {
            std::cerr << "[forward] adapter id " << active_adapter_id
                      << " not in pool — running without adapter\n";
        }
    }

    // Multimodal: compute the vision encoder side-inputs (no-op for text).
    plan_vision_(req, plan);
    // Multimodal: compute the Gemma-4 audio encoder side-inputs (no-op for text).
    plan_audio_gemma4_(req, plan);
    // Qwen3-VL M-RoPE: per-token [t,h,w] positions (no-op for non-mrope models).
    build_mrope_positions_(req, plan);

    return plan;
}

void Executor::build_mrope_positions_(const pie_driver::PieForwardRequestView& req,
                                      BatchPlan& plan) {
    const auto& h = model_.hparams();
    if (!h.use_mrope) return;
    const int N = plan.total_n_tokens;
    if (N <= 0) return;
    // Text default: every row gets (p,p,p) from the 1-D positions.
    plan.mrope_positions_i32.assign(static_cast<std::size_t>(N) * 3, 0);
    for (int t = 0; t < N; ++t) {
        const std::int32_t p =
            t < static_cast<int>(plan.positions_i32.size()) ? plan.positions_i32[t] : 0;
        plan.mrope_positions_i32[static_cast<std::size_t>(3 * t) + 0] = p;
        plan.mrope_positions_i32[static_cast<std::size_t>(3 * t) + 1] = p;
        plan.mrope_positions_i32[static_cast<std::size_t>(3 * t) + 2] = p;
    }
    // Overwrite image/video soft-token rows with their staged [t,h,w] positions
    // (mirrors the CUDA driver's merge). image_mrope_indptr is a per-image CSR in
    // element units (3 per token); image i's rows start at batch row anchors[i].
    const int num_img = static_cast<int>(req.num_images());
    if (num_img == 0 || req.image_mrope_positions.empty() ||
        req.image_mrope_indptr.size() < static_cast<std::size_t>(num_img) + 1 ||
        req.image_anchor_rows.size() < static_cast<std::size_t>(num_img)) {
        return;
    }
    const std::uint32_t* mpos    = req.image_mrope_positions.data();
    const std::uint32_t* mindptr = req.image_mrope_indptr.data();
    const std::uint32_t* anchors = req.image_anchor_rows.data();
    const std::size_t mpos_len   = req.image_mrope_positions.size();
    for (int im = 0; im < num_img; ++im) {
        const std::uint32_t anchor_row = anchors[im];
        const std::uint32_t lo = mindptr[im];
        const std::uint32_t hi = mindptr[im + 1];
        if (hi < lo || hi > mpos_len) continue;
        const std::uint32_t n_tok = (hi - lo) / 3u;
        for (std::uint32_t j = 0; j < n_tok; ++j) {
            const int row = static_cast<int>(anchor_row + j);
            if (row < 0 || row >= N) continue;
            plan.mrope_positions_i32[static_cast<std::size_t>(3 * row) + 0] =
                static_cast<std::int32_t>(mpos[lo + 3 * j + 0]);
            plan.mrope_positions_i32[static_cast<std::size_t>(3 * row) + 1] =
                static_cast<std::int32_t>(mpos[lo + 3 * j + 1]);
            plan.mrope_positions_i32[static_cast<std::size_t>(3 * row) + 2] =
                static_cast<std::int32_t>(mpos[lo + 3 * j + 2]);
        }
    }
}

void Executor::plan_vision_(const pie_driver::PieForwardRequestView& req,
                            BatchPlan& plan) {
    const auto& h = model_.hparams();
    if (req.num_images() == 0) return;
    if (h.arch == PieArch::Gemma4 && model_.weights().gemma4_vision.present) {
        plan_vision_gemma4_(req, plan);
        return;
    }
    if (h.arch != PieArch::Qwen3VL) return;
    const auto& V = model_.weights().vision;
    if (!V.present || !V.pos_embed) return;
    const std::int32_t num_img = static_cast<std::int32_t>(req.num_images());

    const std::int32_t merge   = h.vision_spatial_merge_size;
    const std::int32_t hidden  = h.vision_hidden_size;
    const std::int32_t heads   = h.vision_num_heads;
    const std::int32_t head_dim = h.vision_head_dim > 0 ? h.vision_head_dim
                                                        : hidden / heads;
    const std::int32_t patch_dim = h.vision_in_channels *
        h.vision_temporal_patch_size * h.vision_patch_size * h.vision_patch_size;

    // Learned abs pos-embed table -> f32 (read once, shared across images).
    const std::int32_t num_pos = static_cast<std::int32_t>(V.pos_embed->ne[1]);
    const std::int32_t side =
        static_cast<std::int32_t>(0.5 + std::sqrt(static_cast<double>(num_pos)));
    std::vector<float> table(static_cast<std::size_t>(num_pos) * hidden);
    {
        const std::size_t n = table.size();
        if (V.pos_embed->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(V.pos_embed, table.data(), 0, n * sizeof(float));
        } else if (V.pos_embed->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> tmp(n);
            ggml_backend_tensor_get(V.pos_embed, tmp.data(), 0, n * sizeof(ggml_fp16_t));
            for (std::size_t i = 0; i < n; ++i) table[i] = ggml_fp16_to_fp32(tmp[i]);
        } else {  // BF16
            std::vector<ggml_bf16_t> tmp(n);
            ggml_backend_tensor_get(V.pos_embed, tmp.data(), 0, n * sizeof(ggml_bf16_t));
            for (std::size_t i = 0; i < n; ++i) table[i] = ggml_bf16_to_fp32(tmp[i]);
        }
    }

    const std::uint32_t* grids = req.image_grids.data();
    const std::uint8_t* px_bytes = req.image_pixels.data();
    const std::uint32_t* pix_indptr = req.image_pixel_indptr.size() >=
        static_cast<std::size_t>(num_img) + 1 ? req.image_pixel_indptr.data() : nullptr;
    const std::uint32_t* anchors = req.image_anchor_rows.data();

    // First pass: per-image n_patch / n_token from the grid.
    struct ImgGeom { std::int32_t gt, gh, gw, n_patch, n_token, patch_off; std::int64_t anchor; };
    std::vector<ImgGeom> imgs;
    std::int32_t total_patch = 0, total_token = 0, patch_cursor = 0;
    for (std::int32_t im = 0; im < num_img; ++im) {
        const std::int32_t gt = static_cast<std::int32_t>(grids[3 * im + 0]);
        const std::int32_t gh = static_cast<std::int32_t>(grids[3 * im + 1]);
        const std::int32_t gw = static_cast<std::int32_t>(grids[3 * im + 2]);
        const std::int32_t np = gt * gh * gw;
        if (np <= 0 || (np % (merge * merge)) != 0) return;
        imgs.push_back({gt, gh, gw, np, np / (merge * merge), patch_cursor,
                        static_cast<std::int64_t>(anchors[im])});
        total_patch += np; total_token += np / (merge * merge); patch_cursor += np;
    }

    plan.vis_pixels.resize(static_cast<std::size_t>(total_patch) * patch_dim);
    plan.vis_pos_embed.resize(static_cast<std::size_t>(total_patch) * hidden);
    plan.vis_rope_cos.resize(static_cast<std::size_t>(total_patch) * head_dim);
    plan.vis_rope_sin.resize(static_cast<std::size_t>(total_patch) * head_dim);
    plan.vis_img_rows.resize(static_cast<std::size_t>(total_token));

    // Second pass: fill combined buffers (concatenated per image).
    std::int32_t tok_off = 0;
    for (std::int32_t im = 0; im < num_img; ++im) {
        const ImgGeom& g = imgs[static_cast<std::size_t>(im)];
        const std::int32_t np = g.n_patch, po = g.patch_off;
        // pixels.
        const std::size_t need = static_cast<std::size_t>(np) * patch_dim;
        const std::uint8_t* src = pix_indptr ? px_bytes + pix_indptr[im] : px_bytes;
        std::memcpy(plan.vis_pixels.data() + static_cast<std::size_t>(po) * patch_dim,
                    src, need * sizeof(float));
        // positions -> pos-embed (bilinear) + 2D-RoPE.
        std::vector<std::int32_t> pos = qwen3vl_vision_positions(g.gt, g.gh, g.gw, merge);
        std::vector<float> pe = qwen3vl_pos_embed_interp(table.data(), side, hidden,
                                                         pos.data(), np, g.gh, g.gw);
        std::copy(pe.begin(), pe.end(),
                  plan.vis_pos_embed.begin() + static_cast<std::ptrdiff_t>(po) * hidden);
        std::vector<float> rc, rs;
        qwen3vl_rope_cos_sin(pos.data(), np, head_dim, h.vision_rope_theta, rc, rs);
        std::copy(rc.begin(), rc.end(),
                  plan.vis_rope_cos.begin() + static_cast<std::ptrdiff_t>(po) * head_dim);
        std::copy(rs.begin(), rs.end(),
                  plan.vis_rope_sin.begin() + static_cast<std::ptrdiff_t>(po) * head_dim);
        // scatter rows.
        for (std::int32_t i = 0; i < g.n_token; ++i)
            plan.vis_img_rows[static_cast<std::size_t>(tok_off + i)] = g.anchor + i;
        tok_off += g.n_token;
    }

    // Block-diagonal attention mask (only when >1 image; single -> null).
    if (num_img > 1) {
        const float ninf = -std::numeric_limits<float>::infinity();
        plan.vis_attn_mask.assign(
            static_cast<std::size_t>(total_patch) * total_patch, 0.0f);
        std::vector<std::int32_t> img_of(static_cast<std::size_t>(total_patch));
        for (std::int32_t im = 0; im < num_img; ++im)
            for (std::int32_t i = 0; i < imgs[static_cast<std::size_t>(im)].n_patch; ++i)
                img_of[static_cast<std::size_t>(imgs[static_cast<std::size_t>(im)].patch_off + i)] = im;
        for (std::int32_t qi = 0; qi < total_patch; ++qi)
            for (std::int32_t kj = 0; kj < total_patch; ++kj)
                if (img_of[static_cast<std::size_t>(qi)] != img_of[static_cast<std::size_t>(kj)])
                    plan.vis_attn_mask[static_cast<std::size_t>(qi) * total_patch + kj] = ninf;
    }

    plan.vis_patch_dim = patch_dim;
    plan.vis_n_patch   = total_patch;
    plan.vis_n_token   = total_token;
    plan.has_images    = true;
}

void Executor::plan_vision_gemma4_(const pie_driver::PieForwardRequestView& req,
                                   BatchPlan& plan) {
    const auto& h = model_.hparams();
    const auto& V = model_.weights().gemma4_vision;
    if (!V.present || !V.patch_w || !V.pos_table) return;
    const std::int32_t num_img = static_cast<std::int32_t>(req.num_images());
    if (num_img == 0) return;
    const std::int32_t hidden  = h.vision_hidden_size;
    const std::int32_t heads   = h.vision_num_heads;
    const std::int32_t head_dim = h.vision_head_dim > 0 ? h.vision_head_dim
                                                        : hidden / heads;
    const std::int32_t pool_k = h.vision_pool_kernel > 0 ? h.vision_pool_kernel : 1;
    const std::int32_t patch_dim = static_cast<std::int32_t>(V.patch_w->ne[0]);

    // Factored pos-embed table [2,P,hidden] -> f32 (read once, shared).
    const std::int32_t P = static_cast<std::int32_t>(V.pos_table->ne[1]);
    std::vector<float> table(static_cast<std::size_t>(2) * P * hidden);
    {
        const std::size_t n = table.size();
        if (V.pos_table->type == GGML_TYPE_F32) {
            ggml_backend_tensor_get(V.pos_table, table.data(), 0, n * sizeof(float));
        } else if (V.pos_table->type == GGML_TYPE_F16) {
            std::vector<ggml_fp16_t> tmp(n);
            ggml_backend_tensor_get(V.pos_table, tmp.data(), 0, n * sizeof(ggml_fp16_t));
            for (std::size_t i = 0; i < n; ++i) table[i] = ggml_fp16_to_fp32(tmp[i]);
        } else {
            std::vector<ggml_bf16_t> tmp(n);
            ggml_backend_tensor_get(V.pos_table, tmp.data(), 0, n * sizeof(ggml_bf16_t));
            for (std::size_t i = 0; i < n; ++i) table[i] = ggml_bf16_to_fp32(tmp[i]);
        }
    }

    // Per-image pixel byte ranges. image_pixel_indptr is [0, end0, end1, ...]
    // (num_img+1, bytes); fall back to "all pixels = image 0" if absent.
    const std::uint32_t* pix_indptr = req.image_pixel_indptr.size() >=
        static_cast<std::size_t>(num_img) + 1 ? req.image_pixel_indptr.data() : nullptr;
    const float* px_all = reinterpret_cast<const float*>(req.image_pixels.data());
    const std::uint32_t* pp_all = req.image_patch_positions.data();
    const std::uint32_t* anchors = req.image_anchor_rows.data();

    // First pass: per-image geometry (n_patch / grid / n_token).
    struct ImgGeom { std::int32_t n_patch, n_token, gw, gh, patch_off; std::int64_t anchor; };
    std::vector<ImgGeom> imgs;
    imgs.reserve(static_cast<std::size_t>(num_img));
    std::int32_t total_patch = 0, total_token = 0, patch_cursor = 0;
    for (std::int32_t im = 0; im < num_img; ++im) {
        std::int32_t n_patch_im;
        if (pix_indptr) {
            const std::size_t b0 = pix_indptr[im], b1 = pix_indptr[im + 1];
            n_patch_im = static_cast<std::int32_t>((b1 - b0) / sizeof(float) / patch_dim);
        } else {
            n_patch_im = static_cast<std::int32_t>(req.image_patch_positions.size() / 2);
        }
        if (n_patch_im <= 0) return;
        std::int32_t maxx = 0, maxy = 0;
        for (std::int32_t i = 0; i < n_patch_im; ++i) {
            const std::int32_t x = static_cast<std::int32_t>(pp_all[2 * (patch_cursor + i)]);
            const std::int32_t y = static_cast<std::int32_t>(pp_all[2 * (patch_cursor + i) + 1]);
            maxx = std::max(maxx, x); maxy = std::max(maxy, y);
        }
        const std::int32_t gw = maxx + 1, gh = maxy + 1;
        const std::int32_t n_token_im = (gw / pool_k) * (gh / pool_k);
        imgs.push_back({n_patch_im, n_token_im, gw, gh, patch_cursor,
                        static_cast<std::int64_t>(anchors[im])});
        total_patch += n_patch_im; total_token += n_token_im;
        patch_cursor += n_patch_im;
    }

    // Allocate combined buffers.
    plan.vis_pixels.resize(static_cast<std::size_t>(total_patch) * patch_dim);
    plan.vis_pos_embed.resize(static_cast<std::size_t>(total_patch) * hidden);
    plan.vis_rope_cos.resize(static_cast<std::size_t>(total_patch) * head_dim);
    plan.vis_rope_sin.resize(static_cast<std::size_t>(total_patch) * head_dim);
    plan.vis_pool_matrix.assign(
        static_cast<std::size_t>(total_token) * total_patch, 0.0f);
    plan.vis_img_rows.resize(static_cast<std::size_t>(total_token));

    // Second pass: fill per-image data into the combined buffers (block-diagonal
    // pool), and stage pixels (pre-scaled 2x-1).
    std::int32_t tok_off = 0;
    for (std::int32_t im = 0; im < num_img; ++im) {
        const ImgGeom& g = imgs[static_cast<std::size_t>(im)];
        const std::int32_t np = g.n_patch, po = g.patch_off;
        std::vector<std::int32_t> pos(static_cast<std::size_t>(np) * 2);
        for (std::int32_t i = 0; i < np; ++i) {
            pos[2 * i]     = static_cast<std::int32_t>(pp_all[2 * (po + i)]);
            pos[2 * i + 1] = static_cast<std::int32_t>(pp_all[2 * (po + i) + 1]);
        }
        // pixels (2x-1).
        const float* src = pix_indptr ? px_all + pix_indptr[im] / sizeof(float) : px_all;
        const std::size_t pbytes = static_cast<std::size_t>(np) * patch_dim;
        for (std::size_t i = 0; i < pbytes; ++i)
            plan.vis_pixels[static_cast<std::size_t>(po) * patch_dim + i] =
                2.0f * src[i] - 1.0f;
        // pos-embed.
        std::vector<float> pe = gemma4_vision_pos_embed(table.data(), P, hidden,
                                                        pos.data(), np);
        std::copy(pe.begin(), pe.end(),
                  plan.vis_pos_embed.begin() + static_cast<std::ptrdiff_t>(po) * hidden);
        // rope.
        std::vector<float> rc, rs;
        gemma4_vision_rope_cos_sin(pos.data(), np, head_dim, h.vision_rope_theta, rc, rs);
        std::copy(rc.begin(), rc.end(),
                  plan.vis_rope_cos.begin() + static_cast<std::ptrdiff_t>(po) * head_dim);
        std::copy(rs.begin(), rs.end(),
                  plan.vis_rope_sin.begin() + static_cast<std::ptrdiff_t>(po) * head_dim);
        // pool block (token-major [n_token_im, np]) -> combined block-diagonal.
        std::int32_t nt = 0;
        std::vector<float> pm = gemma4_vision_pool_matrix(pos.data(), np, g.gw, g.gh,
                                                          pool_k, hidden, nt);
        for (std::int32_t t = 0; t < nt; ++t)
            for (std::int32_t p = 0; p < np; ++p)
                plan.vis_pool_matrix[static_cast<std::size_t>(tok_off + t) * total_patch +
                                     (po + p)] = pm[static_cast<std::size_t>(t) * np + p];
        // scatter rows.
        for (std::int32_t t = 0; t < nt; ++t)
            plan.vis_img_rows[static_cast<std::size_t>(tok_off + t)] = g.anchor + t;
        tok_off += nt;
    }

    // Block-diagonal attention mask (only when >1 image; single image -> null).
    if (num_img > 1) {
        const float ninf = -std::numeric_limits<float>::infinity();
        plan.vis_attn_mask.assign(
            static_cast<std::size_t>(total_patch) * total_patch, 0.0f);
        std::vector<std::int32_t> img_of(static_cast<std::size_t>(total_patch));
        for (std::int32_t im = 0; im < num_img; ++im)
            for (std::int32_t i = 0; i < imgs[static_cast<std::size_t>(im)].n_patch; ++i)
                img_of[static_cast<std::size_t>(imgs[static_cast<std::size_t>(im)].patch_off + i)] = im;
        for (std::int32_t qi = 0; qi < total_patch; ++qi)
            for (std::int32_t kj = 0; kj < total_patch; ++kj)
                if (img_of[static_cast<std::size_t>(qi)] != img_of[static_cast<std::size_t>(kj)])
                    plan.vis_attn_mask[static_cast<std::size_t>(qi) * total_patch + kj] = ninf;
    }

    if (std::getenv("PIE_PORTABLE_DUMP_TOKENS")) {
        std::cerr << "[g4vis] num_img=" << num_img << " total_patch=" << total_patch
                  << " total_token=" << total_token << " patch_dim=" << patch_dim << "\n";
    }

    plan.vis_patch_dim = patch_dim;
    plan.vis_n_patch   = total_patch;
    plan.vis_n_token   = total_token;
    plan.has_images    = true;
}

void Executor::plan_audio_gemma4_(const pie_driver::PieForwardRequestView& req,
                                  BatchPlan& plan) {
    const auto& h = model_.hparams();
    const auto& A = model_.weights().gemma4_audio;
    if (!A.present || h.audio_hidden_size <= 0) return;
    const std::int32_t num_clip = static_cast<std::int32_t>(req.num_clips());
    if (num_clip == 0) return;
    const std::int32_t hidden = h.audio_hidden_size;
    const std::int32_t n_mel  = h.audio_feature_size;
    const std::int32_t P      = h.audio_context_left;
    const std::int32_t Dwin   = P - 1;

    plan.aud_pe = gemma4_audio_rel_pos_enc(P, hidden);
    plan.aud_n_mel = n_mel;
    plan.aud_dwin  = Dwin;

    // Each clip is encoded independently (N-separate-encodes; the SSCP/depthwise
    // convs mix across time, so clips can't be stacked like vision images).
    const std::uint32_t* fp = req.audio_feature_indptr.data();
    const std::uint32_t* anchors = req.audio_anchor_rows.data();
    for (std::int32_t ci = 0; ci < num_clip; ++ci) {
        const std::size_t blo = fp[ci], bhi = fp[ci + 1];
        if (bhi <= blo) return;
        const std::int32_t n_frame =
            static_cast<std::int32_t>((bhi - blo) / sizeof(float)) / n_mel;
        if (n_frame <= 0) return;
        const std::int32_t n_token = gemma4_audio_subsampled_len(n_frame);
        if (n_token <= 0) return;

        BatchPlan::AudClip clip;
        clip.n_frame = n_frame;
        clip.n_token = n_token;
        const float* src =
            reinterpret_cast<const float*>(req.audio_features.data() + blo);
        clip.features.assign(src, src + static_cast<std::size_t>(n_frame) * n_mel);
        clip.win_mask = gemma4_audio_window_mask(Dwin, n_token);
        const std::int64_t anchor = static_cast<std::int64_t>(anchors[ci]);
        clip.rows.resize(static_cast<std::size_t>(n_token));
        for (std::int32_t i = 0; i < n_token; ++i)
            clip.rows[static_cast<std::size_t>(i)] = anchor + i;
        plan.aud_clips.push_back(std::move(clip));

        if (std::getenv("PIE_PORTABLE_DUMP_TOKENS")) {
            std::cerr << "[g4aud] clip " << ci << " n_frame=" << n_frame
                      << " n_token=" << n_token << " dwin=" << Dwin << "\n";
        }
    }
    plan.has_audio = true;
}


// -----------------------------------------------------------------------------
// Test harness plan: simulate Pie's page allocator with contiguous pages
// starting at `page_offset`.
// -----------------------------------------------------------------------------

Executor::BatchPlan Executor::plan_test_simple_(
        std::span<const std::uint32_t> token_ids,
        std::span<const std::uint32_t> position_ids,
        std::int32_t sampling_pos,
        std::int32_t page_offset) {
    if (token_ids.size() != position_ids.size() || token_ids.empty()) {
        throw std::runtime_error("plan_test: token/position size mismatch or empty");
    }
    const std::int32_t n_tok = static_cast<std::int32_t>(token_ids.size());
    if (sampling_pos < 0 || sampling_pos >= n_tok) {
        throw std::runtime_error("plan_test: sampling_pos out of range");
    }

    std::int32_t max_pos = 0;
    for (auto p : position_ids) {
        max_pos = std::max(max_pos, static_cast<std::int32_t>(p));
    }
    const std::int32_t seq_len   = max_pos + 1;
    const std::int32_t page_size = kv_.page_size();
    const std::int32_t pages_n   = (seq_len + page_size - 1) / page_size;
    if (page_offset + pages_n > kv_.total_pages()) {
        throw std::runtime_error(
            "plan_test: page allocation exceeds total_pages");
    }

    BatchPlan plan;
    plan.total_n_tokens = n_tok;
    plan.tokens_i32.resize(n_tok);
    plan.positions_i32.resize(n_tok);
    plan.kv_idxs_i64.resize(n_tok);
    // Filled below alongside ReqPlan::sampling_positions.

    auto pos_to_phys = [&](std::int32_t p) -> std::int64_t {
        const std::int32_t page = page_offset + p / page_size;
        return static_cast<std::int64_t>(page) * page_size + (p % page_size);
    };

    for (std::int32_t i = 0; i < n_tok; ++i) {
        const std::int32_t p = static_cast<std::int32_t>(position_ids[i]);
        plan.tokens_i32[i]    = static_cast<std::int32_t>(token_ids[i]);
        plan.positions_i32[i] = p;
        plan.kv_idxs_i64[i]   = pos_to_phys(p);
    }

    ReqPlan rp;
    rp.qo_start     = 0;
    rp.n_tokens     = n_tok;
    rp.n_tokens_pad = ((n_tok + MASK_PAD - 1) / MASK_PAD) * MASK_PAD;
    // Bucket n_kv to a kv-page boundary for archs that hit the slow path
    // (Qwen 3.5 — recurrent state forces per-request views). Pure-decode
    // M11 archs use rp.n_kv directly to size masks, so leave it raw.
    const bool slow_only_arch = model_.hparams().arch == PieArch::Qwen3_5;
    const std::int32_t kv_bucket = kv_.page_size();
    const std::int32_t n_kv_eff = slow_only_arch
        ? ((seq_len + kv_bucket - 1) / kv_bucket) * kv_bucket
        : seq_len;
    rp.n_kv = n_kv_eff;
    rp.sampling_positions.push_back(sampling_pos);
    rp.gather_idxs.assign(n_kv_eff, 0);
    for (std::int32_t k = 0; k < seq_len; ++k) {
        rp.gather_idxs[k] = static_cast<std::int32_t>(pos_to_phys(k));
    }
    build_causal_mask_f16(rp.mask_f16, n_kv_eff, n_tok, rp.n_tokens_pad,
                          plan.positions_i32.data());
    rp.sampler = SamplerParams{};
    rp.sampler.temperature = 0.0f; // greedy for offline test mode
    rp.samplers.assign(1, rp.sampler);
    rp.state_slot = 0;             // single-context test harness
    plan.sampling_pos_i32.push_back(sampling_pos);
    plan.reqs.push_back(std::move(rp));
    // Qwen 3.5 has GPU-greedy but no M11 packed-decode (recurrent state
    // requires per-request slot views). Gemma 4 supports both via the
    // packed-decode path with two mask variants (sliding + full).
    const bool slow_only = model_.hparams().arch == PieArch::Qwen3_5;
    plan.all_greedy = !slow_only;

    // Detect pure-decode (single token) for the M11 fast path.
    if (n_tok == 1 && !slow_only) {
        plan.pure_decode = true;
        const std::int32_t bucket = kv_.page_size();
        plan.max_n_kv = ((seq_len + bucket - 1) / bucket) * bucket;
        const auto& h = model_.hparams();
        const std::int32_t W = h.sliding_window.value_or(0);
        const bool need_full_mask = h.arch == PieArch::Gemma4 && W > 0;
        build_pure_decode_packing(plan, /*n_request=*/1, kv_.page_size(), W,
                                  need_full_mask);
    }
    return plan;
}

// -----------------------------------------------------------------------------
// Compute
// -----------------------------------------------------------------------------

std::vector<SamplerOutput> Executor::compute_(const BatchPlan& plan) {
    using clock = std::chrono::steady_clock;
    const auto t_compute_start = clock::now();
    auto stage_start = t_compute_start;
    auto take_us = [&](std::uint64_t& bucket) {
        const auto now = clock::now();
        bucket += std::chrono::duration_cast<std::chrono::microseconds>(
                      now - stage_start).count();
        stage_start = now;
    };

    const PieArch arch = model_.hparams().arch;
    // Cache pure-decode and slow-path graphs alike. Pure-decode hits
    // dominate steady-state serving (max_n_kv bucketed to page boundary
    // in plan_); slow-path hits help when consecutive batches share
    // shape (e.g. replicate prefill, identical-prompt batched prefills).
    const bool hit = cache_->matches(arch, plan);

    if (!hit) {
        cache_->release();  // free any previous cached ctx

        const std::size_t mem_size =
            ggml_tensor_overhead() * (1ull << 20) +
            ggml_graph_overhead_custom(GRAPH_MAX_NODES, false);
        ggml_init_params ip{
            /*.mem_size   =*/ mem_size,
            /*.mem_buffer =*/ nullptr,
            /*.no_alloc   =*/ true,
        };
        cache_->ctx = ggml_init(ip);
        if (!cache_->ctx) throw std::runtime_error("compute: ggml_init failed");

        try {
            if (arch == PieArch::Qwen3_5) {
                cache_->result = build_qwen3_5_graph(
                    cache_->ctx, model_, kv_, *state_, plan);
            } else if (arch == PieArch::Gemma4) {
                cache_->result = build_gemma4_graph(
                    cache_->ctx, model_, kv_, plan);
            } else {
                if (model_.hparams().arch == PieArch::Phi3Small) {
                    cache_->result = build_phi3small_graph(cache_->ctx, model_, kv_, plan);
                } else if (model_.hparams().arch == PieArch::Phi3_5Moe) {
                    cache_->result = build_phi3_5moe_graph(cache_->ctx, model_, kv_, plan);
                } else {
                    cache_->result = build_qwen3_graph(
                    cache_->ctx, model_, kv_, plan);
                }
            }
            take_us(timings_.graph_build_us);

            // On cache miss the graph topology has changed (or this is
            // the first call), so wipe sched's previous assignments
            // and re-allocate buffers for the new graph. After this
            // point `sched_` is configured for `cache_->result.gf` and
            // subsequent cache HITs can skip both the reset and the
            // alloc — the scheduler's split assignments and per-backend
            // buffers remain valid for the same graph topology, which
            // keeps the steady-state decode path at zero per-call
            // sched overhead.
            ggml_backend_sched_reset(sched_);
            if (!ggml_backend_sched_alloc_graph(sched_, cache_->result.gf)) {
                throw std::runtime_error("compute: sched_alloc_graph failed");
            }
            take_us(timings_.graph_alloc_us);

            // No silent CPU fallback: after the scheduler assigns backends, flag
            // any compute node placed on the CPU fallback instead of the primary
            // (Metal/GPU) backend. A missing Metal kernel silently routing a hot
            // op to CPU is a performance trap (it can peg a core); surface it so
            // it's never silent. Runs only on cache-miss (new graph topology).
            // PIE_PORTABLE_STRICT_METAL=1 turns the warning into a hard error.
            if (ggml_backend_t cpu_fb = model_.cpu_fallback()) {
                ggml_cgraph* g = cache_->result.gf;
                std::map<std::string, int> cpu_ops;
                int cpu_nodes = 0;
                const int nn = ggml_graph_n_nodes(g);
                for (int i = 0; i < nn; ++i) {
                    ggml_tensor* node = ggml_graph_node(g, i);
                    // No-op metadata ops (view/reshape/permute/transpose) carry no
                    // kernel and compute nothing — the scheduler may place them on
                    // any backend at zero cost, so they are not real fallbacks.
                    switch (node->op) {
                        case GGML_OP_NONE:
                        case GGML_OP_VIEW:
                        case GGML_OP_RESHAPE:
                        case GGML_OP_PERMUTE:
                        case GGML_OP_TRANSPOSE:
                            continue;
                        default: break;
                    }
                    if (ggml_backend_sched_get_tensor_backend(sched_, node) == cpu_fb) {
                        ++cpu_nodes;
                        cpu_ops[ggml_op_name(node->op)]++;
                    }
                }
                if (cpu_nodes > 0) {
                    std::ostringstream msg;
                    msg << cpu_nodes << " graph node(s) fell back to CPU (no Metal "
                           "kernel):";
                    for (const auto& [op, n] : cpu_ops) msg << " " << op << "x" << n;
                    if (std::getenv("PIE_PORTABLE_STRICT_METAL")) {
                        throw std::runtime_error("compute: CPU fallback forbidden: " +
                                                 msg.str());
                    }
                    std::cerr << "[pie-driver-portable] WARNING: " << msg.str()
                              << " (set PIE_PORTABLE_STRICT_METAL=1 to forbid)\n";
                }
            }
        } catch (...) {
            cache_->release();
            throw;
        }

        cache_->store_key(arch, plan);
    }
    // On cache hit we attribute zero microseconds to graph_build/alloc.

    const GraphResult& g = cache_->result;

    upload_graph_inputs(g, plan);
    take_us(timings_.upload_us);

    const auto status = ggml_backend_sched_graph_compute(sched_, g.gf);
    if (status != GGML_STATUS_SUCCESS) {
        throw std::runtime_error("compute: sched_graph_compute status=" +
                                 std::to_string(static_cast<int>(status)));
    }
    take_us(timings_.compute_us);

    const std::int32_t n_slots =
        static_cast<std::int32_t>(plan.sampling_pos_i32.size());

    std::vector<SamplerOutput> out;
    if (plan.all_greedy) {
        // GPU-greedy fast path: download only the int32 token ids.
        std::vector<std::int32_t> tokens(n_slots);
        ggml_backend_tensor_get(g.tokens_out, tokens.data(), 0,
                                tokens.size() * sizeof(std::int32_t));
        take_us(timings_.logits_dl_us);
        out = sample_batch_greedy(plan, tokens.data());
        take_us(timings_.sample_us);
    } else if (plan.uniform_top_sample) {
        // GPU non-greedy fast path: download top-K probs + indices and
        // finalize per-slot host-side. K * n_slots * 8 bytes total.
        const std::int32_t K = plan.uniform_top_k;
        std::vector<std::int32_t> top_idx(static_cast<std::size_t>(K) * n_slots);
        std::vector<float>        top_prob(static_cast<std::size_t>(K) * n_slots);
        ggml_backend_tensor_get(g.top_k_idx, top_idx.data(), 0,
                                top_idx.size() * sizeof(std::int32_t));
        ggml_backend_tensor_get(g.top_k_probs, top_prob.data(), 0,
                                top_prob.size() * sizeof(float));
        take_us(timings_.logits_dl_us);
        out = sample_batch_uniform_top(plan, top_idx.data(), top_prob.data(), K);
        take_us(timings_.sample_us);
    } else {
        const std::int32_t vocab_size = model_.hparams().vocab_size;
        std::vector<float> all_logits(
            static_cast<std::size_t>(vocab_size) * n_slots);
        ggml_backend_tensor_get(g.logits, all_logits.data(), 0,
                                all_logits.size() * sizeof(float));
        take_us(timings_.logits_dl_us);
        out = sample_batch(plan, all_logits.data(), vocab_size);
        take_us(timings_.sample_us);
    }
    timings_.total_us += std::chrono::duration_cast<std::chrono::microseconds>(
                            clock::now() - t_compute_start).count();
    ++timings_.n_calls;
    return out;
}

// -----------------------------------------------------------------------------
// Public entry points
// -----------------------------------------------------------------------------

void Executor::run(const pie_driver::PieForwardRequestView& req,
                        pie_driver::ResponseBuilder& builder,
                        pie_driver::PieForwardResponseView& out) {
    using clock = std::chrono::steady_clock;
    auto t_stage = clock::now();
    auto take_us = [&](std::uint64_t& bucket) {
        const auto now = clock::now();
        bucket += std::chrono::duration_cast<std::chrono::microseconds>(
                      now - t_stage).count();
        t_stage = now;
    };

    // Empty-batch fast path. The runtime drains its queue with zero-token
    // fires (token_ids empty, qo_indptr like [0,0,...,0]) — e.g. when a
    // ForwardPass is built and submitted with no `input_tokens` calls, or
    // during scheduler idle ticks. Emit one empty PerRequestOutput per
    // request slot so the response count matches what fire_batch expects.
    // Matches the driver/cuda/src/executor/executor.cpp:321 short-circuit.
    {
        const auto tok_view = req.token_ids.as<std::uint32_t>();
        const auto qo_view  = req.qo_indptr.as<std::uint32_t>();
        const int R = qo_view.empty()
            ? 0
            : static_cast<int>(qo_view.size()) - 1;
        if (tok_view.empty() || R <= 0) {
            std::vector<pie_driver::PerRequestOutput> empty(
                static_cast<std::size_t>(std::max(R, 0)));
            builder.build(empty, out);
            return;
        }
    }

    BatchPlan plan;
    try {
        plan = plan_(req);
    } catch (const std::exception& e) {
        std::cerr << "[forward] plan failed: " << e.what() << "\n";
        out = pie_driver::PieForwardResponseView{};
        return;
    }
    take_us(timings_.plan_us);

    std::vector<SamplerOutput> sampled;
    try {
        sampled = compute_(plan);
    } catch (const std::exception& e) {
        std::cerr << "[forward] compute failed: " << e.what() << "\n";
        out = pie_driver::PieForwardResponseView{};
        return;
    }
    // compute_() credits its own sub-buckets + total + n_calls.
    t_stage = clock::now();

    // Translate per-request SamplerOutput → pie_driver::PerRequestOutput.
    //
    // Two cases:
    //   * All slots are token-producing samplers. `resolve_request_output`
    //     filled `s.tokens` directly; `special_slots` is empty.
    //   * Any slot is a special sampler. `resolve_request_output` moved
    //     the whole `slot_out` list into `special_slots` (so it could
    //     keep the per-slot payloads next to each other), which means
    //     `s.tokens` is empty even though token-producing slots may be
    //     present. Walk `rp.samplers` in lock-step with `special_slots`
    //     to route each slot's payload into the right per_req array.
    std::vector<pie_driver::PerRequestOutput> per_req;
    per_req.reserve(sampled.size());
    for (std::size_t r = 0; r < sampled.size(); ++r) {
        auto& s = sampled[r];
        const auto& rp = plan.reqs[r];
        pie_driver::PerRequestOutput pr;
        pr.tokens = std::move(s.tokens);
        for (std::size_t i = 0; i < s.special_slots.size(); ++i) {
            auto& slot = s.special_slots[i];
            const SamplerParams& sampler = i < rp.samplers.size()
                ? rp.samplers[i] : rp.sampler;
            switch (sampler.type) {
                case SamplerType::Multinomial:
                case SamplerType::TopK:
                case SamplerType::TopP:
                case SamplerType::MinP:
                case SamplerType::TopKTopP:
                    pr.tokens.push_back(slot.token);
                    break;
                case SamplerType::Distribution:
                    pr.dists.emplace_back(std::move(slot.dist_ids),
                                          std::move(slot.dist_vals));
                    break;
                case SamplerType::RawLogits:
                    pr.logits.push_back(std::move(slot.raw_logits));
                    break;
                case SamplerType::Logprob:
                case SamplerType::Logprobs:
                    pr.logprobs.push_back(std::move(slot.logprobs));
                    break;
                case SamplerType::Entropy:
                    pr.entropies.push_back(slot.entropy);
                    break;
                default:
                    break;
            }
        }
        per_req.push_back(std::move(pr));
    }

    builder.build(per_req, out);
    take_us(timings_.response_pack_us);
}

std::vector<std::uint32_t> Executor::generate(
        std::span<const std::uint32_t> prompt_tokens,
        std::int32_t max_new_tokens,
        std::uint64_t /*context_id*/,
        std::int32_t page_offset) {
    if (prompt_tokens.empty()) {
        throw std::runtime_error("generate: empty prompt");
    }
    std::vector<std::uint32_t> out;
    out.reserve(static_cast<std::size_t>(max_new_tokens));

    const std::int32_t prompt_n = static_cast<std::int32_t>(prompt_tokens.size());

    std::vector<std::uint32_t> positions(prompt_tokens.size());
    for (std::size_t i = 0; i < prompt_tokens.size(); ++i) {
        positions[i] = static_cast<std::uint32_t>(i);
    }

    // Fresh-context start: ensure slot 0's recurrent state is zero.
    if (state_) state_->zero_slot(0);
    auto plan = plan_test_simple_(prompt_tokens,
                                  std::span<const std::uint32_t>(positions),
                                  prompt_n - 1, page_offset);
    auto sampled = compute_(plan);
    out.push_back(sampled[0].tokens.front());

    for (std::int32_t i = 1; i < max_new_tokens; ++i) {
        const std::uint32_t pos = static_cast<std::uint32_t>(prompt_n + i - 1);
        const std::array<std::uint32_t, 1> tok{out.back()};
        const std::array<std::uint32_t, 1> p{pos};
        auto plan_step = plan_test_simple_(std::span<const std::uint32_t>(tok),
                                           std::span<const std::uint32_t>(p),
                                           /*sampling_pos=*/ 0, page_offset);
        auto step_out = compute_(plan_step);
        out.push_back(step_out[0].tokens.front());
    }
    return out;
}

std::vector<std::vector<std::uint32_t>> Executor::generate_multi(
        std::vector<std::vector<std::uint32_t>>& prompts,
        std::int32_t max_new_tokens,
        std::vector<std::uint64_t> /*context_ids*/) {
    if (prompts.empty()) {
        throw std::runtime_error("generate_multi: no prompts");
    }
    const std::size_t n_req = prompts.size();
    std::vector<std::vector<std::uint32_t>> out(n_req);
    std::vector<std::int32_t> prompt_lens(n_req);
    std::vector<std::int32_t> page_offsets(n_req);

    // Allocate non-overlapping page ranges per context.
    const std::int32_t page_size = kv_.page_size();
    std::int32_t cursor = 0;
    for (std::size_t r = 0; r < n_req; ++r) {
        const std::int32_t need = static_cast<std::int32_t>(prompts[r].size()) +
                                  max_new_tokens;
        const std::int32_t pages = (need + page_size - 1) / page_size;
        page_offsets[r] = cursor;
        cursor += pages;
        if (cursor > kv_.total_pages()) {
            throw std::runtime_error(
                "generate_multi: not enough pages (need " +
                std::to_string(cursor) + ", have " +
                std::to_string(kv_.total_pages()) + ")");
        }
    }

    // ---- Prefill each context separately (single-request plans). -----------
    for (std::size_t r = 0; r < n_req; ++r) {
        const auto& p = prompts[r];
        if (p.empty()) {
            throw std::runtime_error("generate_multi: empty prompt at " +
                                     std::to_string(r));
        }
        prompt_lens[r] = static_cast<std::int32_t>(p.size());

        std::vector<std::uint32_t> positions(p.size());
        for (std::size_t i = 0; i < p.size(); ++i) {
            positions[i] = static_cast<std::uint32_t>(i);
        }
        auto plan = plan_test_simple_(std::span<const std::uint32_t>(p),
                                      std::span<const std::uint32_t>(positions),
                                      prompt_lens[r] - 1, page_offsets[r]);
        // Each context gets its own state slot; plan_test_simple_ always
        // returns slot 0, so override here.
        plan.reqs[0].state_slot = static_cast<std::int32_t>(r);
        if (state_) state_->zero_slot(plan.reqs[0].state_slot);
        auto sampled = compute_(plan);
        out[r].push_back(sampled[0].tokens.front());
    }

    // ---- Decode loop: ONE multi-request plan per step. ---------------------
    for (std::int32_t step = 1; step < max_new_tokens; ++step) {
        BatchPlan plan;
        plan.total_n_tokens = static_cast<std::int32_t>(n_req);
        plan.tokens_i32.resize(n_req);
        plan.positions_i32.resize(n_req);
        plan.kv_idxs_i64.resize(n_req);
        plan.sampling_pos_i32.resize(n_req);
        plan.reqs.reserve(n_req);

        for (std::size_t r = 0; r < n_req; ++r) {
            const std::int32_t pos = prompt_lens[r] + step - 1;
            const std::int32_t qo_start = static_cast<std::int32_t>(r);
            const std::int32_t page_offset = page_offsets[r];

            auto pos_to_phys = [&](std::int32_t p) -> std::int64_t {
                const std::int32_t page = page_offset + p / page_size;
                return static_cast<std::int64_t>(page) * page_size + (p % page_size);
            };

            plan.tokens_i32[r]    = static_cast<std::int32_t>(out[r].back());
            plan.positions_i32[r] = pos;
            plan.kv_idxs_i64[r]   = pos_to_phys(pos);
            plan.sampling_pos_i32[r] = qo_start;

            const std::int32_t seq_len = pos + 1;
            // Bucket n_kv to a page boundary for slow-path archs only —
            // pure-decode M11 archs use rp.n_kv directly and would treat a
            // padded value as "actual KV count". See plan_test_simple_.
            const bool slow_only_arch =
                model_.hparams().arch == PieArch::Qwen3_5;
            const std::int32_t kv_bucket = kv_.page_size();
            const std::int32_t n_kv_eff = slow_only_arch
                ? ((seq_len + kv_bucket - 1) / kv_bucket) * kv_bucket
                : seq_len;
            ReqPlan rp;
            rp.qo_start     = qo_start;
            rp.n_tokens     = 1;
            rp.n_tokens_pad = MASK_PAD;
            rp.n_kv         = n_kv_eff;
            rp.sampling_positions.push_back(qo_start);
            rp.gather_idxs.assign(n_kv_eff, 0);
            for (std::int32_t k = 0; k < seq_len; ++k) {
                rp.gather_idxs[k] = static_cast<std::int32_t>(pos_to_phys(k));
            }
            const std::int32_t one_pos[1] = {pos};
            build_causal_mask_f16(rp.mask_f16, n_kv_eff, 1, MASK_PAD, one_pos);
            rp.sampler.temperature = 0.0f;  // greedy for the test harness
            rp.samplers.assign(1, rp.sampler);
            plan.reqs.push_back(std::move(rp));
        }

        // Qwen3-VL (use_mrope) widens pos_input to 4×; keep the invariant that
        // mrope_positions_i32 is populated. No images in this decode harness, so
        // every axis is the 1-D position.
        if (model_.hparams().use_mrope) {
            const int Nt = plan.total_n_tokens;
            plan.mrope_positions_i32.resize(static_cast<std::size_t>(Nt) * 3);
            for (int t = 0; t < Nt; ++t) {
                const std::int32_t p = plan.positions_i32[static_cast<std::size_t>(t)];
                plan.mrope_positions_i32[static_cast<std::size_t>(3 * t) + 0] = p;
                plan.mrope_positions_i32[static_cast<std::size_t>(3 * t) + 1] = p;
                plan.mrope_positions_i32[static_cast<std::size_t>(3 * t) + 2] = p;
            }
        }

        // Activate the M11 packed-decode fast path for this multi-context
        // step (every request has n_tokens=1, no custom attention masks).
        // Qwen 3.5 stays on slow per-request for attention (recurrent
        // state requires per-request slot views); everything else,
        // including Gemma 4 with mixed sliding+full layers, batches.
        const bool qwen35 = model_.hparams().arch == PieArch::Qwen3_5;
        plan.pure_decode = !qwen35;
        plan.all_greedy  = true;
        // Stamp the per-request state slot so Qwen 3.5's StateCache
        // assigns each context a stable slot for the run. Other archs
        // ignore this.
        for (std::size_t r = 0; r < plan.reqs.size(); ++r) {
            plan.reqs[r].state_slot = static_cast<std::int32_t>(r);
        }
        plan.max_n_kv = 0;
        for (const auto& rp : plan.reqs) plan.max_n_kv = std::max(plan.max_n_kv, rp.n_kv);
        if (plan.pure_decode) {
            // Bucket to a kv-page boundary so the cache hits across
            // consecutive decode steps.
            const std::int32_t bucket = kv_.page_size();
            plan.max_n_kv = ((plan.max_n_kv + bucket - 1) / bucket) * bucket;
            // Gemma 4 needs both sliding-clipped and full-context masks;
            // pick the SWA value from hparams (other archs pass 0).
            const auto& h = model_.hparams();
            const bool gemma4 = h.arch == PieArch::Gemma4;
            const std::int32_t W = gemma4 ? h.sliding_window.value_or(0) : 0;
            build_pure_decode_packing(plan,
                                      static_cast<std::int32_t>(n_req),
                                      kv_.page_size(),
                                      W,
                                      /*also_build_no_swa_mask=*/gemma4);
        }

        auto sampled = compute_(plan);
        for (std::size_t r = 0; r < n_req; ++r) {
            out[r].push_back(sampled[r].tokens.front());
        }
    }
    return out;
}

}  // namespace pie_portable_driver
