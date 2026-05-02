#include "plan.hpp"

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include <ggml.h>

#include "sampler.hpp"

namespace pie_portable_driver {

void build_attn_mask_f16(std::vector<std::uint16_t>& dst,
                         std::int32_t n_kv,
                         std::int32_t n_tokens,
                         std::int32_t n_tokens_pad,
                         const std::int32_t* positions,
                         const PerTokenMaskRuns* per_token_runs,
                         std::int32_t sliding_window) {
    dst.assign(static_cast<std::size_t>(n_kv) * n_tokens_pad,
               ggml_fp32_to_fp16(-INFINITY));
    const auto zero = ggml_fp32_to_fp16(0.0f);
    for (std::int32_t i = 0; i < n_tokens; ++i) {
        const std::int32_t p_i = positions[i];
        std::uint16_t* row = dst.data() + static_cast<std::size_t>(i) * n_kv;
        const std::int32_t hi = std::min(n_kv - 1, p_i);
        // Sliding window cuts off the past below `p_i - W + 1`.
        const std::int32_t lo =
            (sliding_window > 0) ? std::max(0, p_i - sliding_window + 1) : 0;

        if (!per_token_runs || per_token_runs[i].n_runs == 0) {
            for (std::int32_t j = lo; j <= hi; ++j) row[j] = zero;
            continue;
        }

        // Custom BRLE over [0, p_i]. Anything past p_i stays -INF. SWA
        // also clips below `lo`.
        const auto* runs   = per_token_runs[i].runs;
        const auto  n_runs = per_token_runs[i].n_runs;
        bool is_true = false;
        std::int32_t pos = 0;
        for (std::size_t k = 0; k < n_runs && pos <= hi; ++k) {
            const std::int32_t len = static_cast<std::int32_t>(runs[k]);
            const std::int32_t end = std::min(pos + len, hi + 1);
            if (is_true) {
                const std::int32_t a = std::max(pos, lo);
                for (std::int32_t j = a; j < end; ++j) row[j] = zero;
            }
            pos = end;
            is_true = !is_true;
        }
    }
}

void build_causal_mask_f16(std::vector<std::uint16_t>& dst,
                           std::int32_t n_kv,
                           std::int32_t n_tokens,
                           std::int32_t n_tokens_pad,
                           const std::int32_t* positions) {
    build_attn_mask_f16(dst, n_kv, n_tokens, n_tokens_pad, positions, nullptr, 0);
}

PlanArrays extract_plan_arrays(const schema::DecodedRequest& req) {
    PlanArrays a;
    a.context_ids          = req.as<std::uint64_t>(schema::A_CONTEXT_IDS);
    a.token_ids            = req.as<std::uint32_t>(schema::A_TOKEN_IDS);
    a.position_ids         = req.as<std::uint32_t>(schema::A_POSITION_IDS);
    a.qo_indptr            = req.as<std::uint32_t>(schema::A_QO_INDPTR);
    a.sampling_idx         = req.as<std::uint32_t>(schema::A_SAMPLING_INDICES);
    a.sampling_indptr      = req.as<std::uint32_t>(schema::A_SAMPLING_INDPTR);
    a.request_num_samplers = req.as<std::uint32_t>(schema::A_REQUEST_NUM_SAMPLERS);
    a.kv_page_indices      = req.as<std::uint32_t>(schema::A_KV_PAGE_INDICES);
    a.kv_page_indptr       = req.as<std::uint32_t>(schema::A_KV_PAGE_INDPTR);
    a.kv_last_lens         = req.as<std::uint32_t>(schema::A_KV_LAST_PAGE_LENS);
    a.sampler_types        = req.as<std::uint32_t>(schema::A_SAMPLER_TYPES);
    a.sampler_temps        = req.as<float>(schema::A_SAMPLER_TEMPERATURES);
    a.sampler_top_k        = req.as<std::uint32_t>(schema::A_SAMPLER_TOP_K);
    a.sampler_top_p        = req.as<float>(schema::A_SAMPLER_TOP_P);
    a.sampler_min_p        = req.as<float>(schema::A_SAMPLER_MIN_P);
    a.sampler_seeds        = req.as<std::uint32_t>(schema::A_SAMPLER_SEEDS);
    a.sampler_label_ids    = req.as<std::uint32_t>(schema::A_SAMPLER_LABEL_IDS);
    a.sampler_label_indptr = req.as<std::uint32_t>(schema::A_SAMPLER_LABEL_INDPTR);
    a.logit_masks          = req.as<std::uint32_t>(schema::A_LOGIT_MASKS);
    a.logit_mask_indptr    = req.as<std::uint32_t>(schema::A_LOGIT_MASK_INDPTR);
    a.flat_attn_masks      = req.as<std::uint32_t>(schema::A_FLATTENED_MASKS);
    a.attn_mask_indptr     = req.as<std::uint32_t>(schema::A_MASK_INDPTR);
    a.adapter_indices      = req.as<std::int64_t>(schema::A_ADAPTER_INDICES);
    a.spec_token_ids       = req.as<std::uint32_t>(schema::A_SPEC_TOKEN_IDS);
    a.spec_position_ids    = req.as<std::uint32_t>(schema::A_SPEC_POSITION_IDS);
    a.spec_indptr          = req.as<std::uint32_t>(schema::A_SPEC_INDPTR);

    a.n_request      = static_cast<std::int32_t>(a.context_ids.size());
    a.total_n_tokens = static_cast<std::int32_t>(a.token_ids.size());
    a.batch_has_drafts =
        a.spec_indptr.size() == static_cast<std::size_t>(a.n_request) + 1;
    // Custom (non-causal) masks are present iff the indptr fully covers
    // every token's row. Otherwise the M11 packed-decode fast path is safe.
    a.batch_has_attn_masks =
        !a.flat_attn_masks.empty() &&
        a.attn_mask_indptr.size() ==
            static_cast<std::size_t>(a.total_n_tokens) + 1;
    return a;
}

void validate_plan_top_level(const PlanArrays& a) {
    if (a.n_request == 0) {
        throw std::runtime_error("plan: no requests in batch");
    }
    if (a.token_ids.size() != a.position_ids.size()) {
        throw std::runtime_error("plan: token_ids/position_ids size mismatch");
    }
    const auto n_plus_1 = static_cast<std::size_t>(a.n_request) + 1;
    if (a.qo_indptr.size() != n_plus_1) {
        throw std::runtime_error("plan: qo_indptr length must be num_requests+1");
    }
    if (a.kv_page_indptr.size() != n_plus_1) {
        throw std::runtime_error("plan: kv_page_indptr length must be num_requests+1");
    }
    if (a.kv_last_lens.size() != static_cast<std::size_t>(a.n_request)) {
        throw std::runtime_error("plan: kv_last_page_lens length must equal num_requests");
    }
    if (a.request_num_samplers.size() != static_cast<std::size_t>(a.n_request)) {
        throw std::runtime_error("plan: request_num_samplers length mismatch");
    }
    if (a.sampling_indptr.size() != n_plus_1) {
        throw std::runtime_error("plan: sampling_indptr length must be num_requests+1");
    }
}

std::int64_t resolve_active_adapter_id(const PlanArrays& a) {
    std::int64_t active = -1;
    bool any = false;
    for (auto x : a.adapter_indices) {
        if (x < 0) continue;
        if (!any) { active = x; any = true; }
        else if (x != active) {
            throw std::runtime_error(
                "plan: mixed adapters in one batch — v1 supports a single "
                "adapter per fire_batch (got " + std::to_string(active) +
                " and " + std::to_string(x) + ")");
        }
    }
    return active;
}

void plan_single_request(const PlanArrays& a,
                         std::int32_t r,
                         std::int32_t page_size,
                         std::int32_t total_pages,
                         const ArchSpec& spec,
                         ForwardEngine::BatchPlan& plan) {
    // M8 spec decode allows multi-slot per request. v1 expects ≥1 slot.
    if (a.request_num_samplers[r] < 1) {
        throw std::runtime_error(
            "plan: request " + std::to_string(r) + " has 0 sampler slots");
    }
    const std::int32_t qo_start = static_cast<std::int32_t>(a.qo_indptr[r]);
    const std::int32_t qo_end   = static_cast<std::int32_t>(a.qo_indptr[r + 1]);
    const std::int32_t n_tok    = qo_end - qo_start;
    if (n_tok <= 0) {
        throw std::runtime_error(
            "plan: request " + std::to_string(r) + " has zero tokens");
    }

    const std::int32_t pages_off   = static_cast<std::int32_t>(a.kv_page_indptr[r]);
    const std::int32_t num_pages_r =
        static_cast<std::int32_t>(a.kv_page_indptr[r + 1]) - pages_off;
    const std::int32_t last_len    = static_cast<std::int32_t>(a.kv_last_lens[r]);
    const std::int32_t seq_len     = num_pages_r > 0
        ? (num_pages_r - 1) * page_size + last_len
        : last_len;
    if (seq_len <= 0) {
        throw std::runtime_error(
            "plan: request " + std::to_string(r) + " has empty KV state");
    }

    // Sanity-check that page IDs are in-bounds for the configured pool.
    for (std::int32_t pi = 0; pi < num_pages_r; ++pi) {
        const auto p = a.kv_page_indices[pages_off + pi];
        if (p >= static_cast<std::uint32_t>(total_pages)) {
            throw std::runtime_error(
                "plan: page id " + std::to_string(p) +
                " out of bounds (total_pages=" +
                std::to_string(total_pages) + ")");
        }
    }

    // BPIQ provides 1 slot per request (the last-pending position that
    // predicts the first draft / next token). Spec decode grows this to
    // 1 + n_drafts.
    const std::int32_t s_start = static_cast<std::int32_t>(a.sampling_indptr[r]);
    const std::int32_t s_end   = static_cast<std::int32_t>(a.sampling_indptr[r + 1]);
    const std::int32_t n_bpiq_slots = s_end - s_start;
    if (n_bpiq_slots < 1) {
        throw std::runtime_error(
            "plan: request " + std::to_string(r) +
            " has 0 BPIQ sampling indices; expected ≥1");
    }
    // Runtime emits sampling_indices as per-request relative offsets
    // (0 = first token of that request's qo range, n_tok-1 = last).
    // We carry global flat-array positions through the plan, so add
    // qo_start to translate.
    const std::int32_t rel_primary =
        static_cast<std::int32_t>(a.sampling_idx[s_start]);
    const std::int32_t primary_idx = qo_start + rel_primary;
    if (rel_primary < 0 || rel_primary >= n_tok) {
        throw std::runtime_error(
            "plan: request " + std::to_string(r) +
            " sampling_index " + std::to_string(rel_primary) +
            " out of relative [0," + std::to_string(n_tok) + ")");
    }

    // Per-token: tokens, positions, and write idxs.
    for (std::int32_t i = qo_start; i < qo_end; ++i) {
        const std::int32_t pos_i = static_cast<std::int32_t>(a.position_ids[i]);
        if (pos_i >= seq_len) {
            throw std::runtime_error(
                "plan: request " + std::to_string(r) +
                " token at position " + std::to_string(pos_i) +
                " exceeds seq_len " + std::to_string(seq_len));
        }
        plan.tokens_i32[i]    = static_cast<std::int32_t>(a.token_ids[i]);
        plan.positions_i32[i] = pos_i;
        plan.kv_idxs_i64[i] = physical_idx(a.kv_page_indices.data(),
                                           pages_off, page_size, pos_i);
    }

    ForwardEngine::ReqPlan rp;
    rp.qo_start     = qo_start;
    rp.n_tokens     = n_tok;
    rp.n_tokens_pad = ((n_tok + MASK_PAD - 1) / MASK_PAD) * MASK_PAD;
    rp.n_kv         = seq_len;

    // Sampling slots: primary first, then one per draft.
    rp.sampling_positions.push_back(primary_idx);
    plan.sampling_pos_i32.push_back(primary_idx);

    if (a.batch_has_drafts) {
        const std::int32_t d_start = static_cast<std::int32_t>(a.spec_indptr[r]);
        const std::int32_t d_end   = static_cast<std::int32_t>(a.spec_indptr[r + 1]);
        const std::int32_t n_drafts = d_end - d_start;
        if (n_drafts > 0) {
            rp.draft_tokens.assign(a.spec_token_ids.data() + d_start,
                                   a.spec_token_ids.data() + d_end);
            // Each draft sits at the unique batch index i where
            // positions_i32[i] == draft_pos.
            for (std::int32_t k = 0; k < n_drafts; ++k) {
                const std::int32_t draft_pos =
                    static_cast<std::int32_t>(a.spec_position_ids[d_start + k]);
                std::int32_t idx = -1;
                for (std::int32_t i = qo_start; i < qo_end; ++i) {
                    if (static_cast<std::int32_t>(a.position_ids[i]) == draft_pos) {
                        idx = i; break;
                    }
                }
                if (idx < 0) {
                    throw std::runtime_error(
                        "plan: spec draft at position " + std::to_string(draft_pos) +
                        " not found in request " + std::to_string(r) + "'s qo range");
                }
                rp.sampling_positions.push_back(idx);
                plan.sampling_pos_i32.push_back(idx);
            }
        }
    }

    rp.gather_idxs.resize(seq_len);
    for (std::int32_t k = 0; k < seq_len; ++k) {
        rp.gather_idxs[k] = static_cast<std::int32_t>(
            physical_idx(a.kv_page_indices.data(), pages_off, page_size, k));
    }

    // Custom per-token attention masks (M6) — optional.
    std::vector<PerTokenMaskRuns> per_token_runs;
    if (a.batch_has_attn_masks) {
        per_token_runs.resize(n_tok);
        for (std::int32_t i = 0; i < n_tok; ++i) {
            const std::int32_t global_i = qo_start + i;
            const std::int32_t lo = static_cast<std::int32_t>(a.attn_mask_indptr[global_i]);
            const std::int32_t hi = static_cast<std::int32_t>(a.attn_mask_indptr[global_i + 1]);
            per_token_runs[i].runs   = a.flat_attn_masks.data() + lo;
            per_token_runs[i].n_runs = static_cast<std::size_t>(hi - lo);
        }
    }
    // Note: this builds ONE mask per request applied to all layers, using
    // the arch's single sliding_window value. For archs with mixed
    // sliding/full layer patterns (gemma2/3, gemma4, gpt-oss, olmo3), the
    // mask is therefore an approximation: sliding-window clipping applies
    // uniformly even on full-attention layers. With seq_len ≤ window
    // (most smoke tests, most short contexts) this has no effect because
    // the clip never bites. With long contexts on mixed-pattern archs the
    // full-attention layers will be incorrectly clipped to the sliding
    // window. FUTURE: build two masks per request (one full, one with
    // sliding clip) and select per-layer in the graph.
    build_attn_mask_f16(rp.mask_f16, seq_len, n_tok, rp.n_tokens_pad,
                        plan.positions_i32.data() + qo_start,
                        a.batch_has_attn_masks ? per_token_runs.data() : nullptr,
                        spec.sliding_window);

    // Sampler params (slot index = s_start in v1, where each request has 1 slot).
    if (s_start >= static_cast<std::int32_t>(a.sampler_types.size())) {
        throw std::runtime_error(
            "plan: sampler_types too short for request " + std::to_string(r));
    }
    rp.sampler.type = static_cast<SamplerType>(a.sampler_types[s_start]);
    auto opt_at = [&](auto span, auto fallback) {
        return s_start < static_cast<std::int32_t>(span.size())
            ? span[s_start] : fallback;
    };
    rp.sampler.temperature = opt_at(a.sampler_temps, 1.0f);
    rp.sampler.top_k       = opt_at(a.sampler_top_k, 0u);
    rp.sampler.top_p       = opt_at(a.sampler_top_p, 1.0f);
    rp.sampler.min_p       = opt_at(a.sampler_min_p, 0.0f);
    rp.sampler.seed        = opt_at(a.sampler_seeds, 0u);

    // Per-slot Logprob/Logprobs labels (sampler-slot-keyed indptr).
    if (a.sampler_label_indptr.size() > static_cast<std::size_t>(s_start) + 1) {
        const std::int32_t la = static_cast<std::int32_t>(a.sampler_label_indptr[s_start]);
        const std::int32_t lb = static_cast<std::int32_t>(a.sampler_label_indptr[s_start + 1]);
        if (lb > la) {
            rp.sampler.labels.assign(
                a.sampler_label_ids.data() + la,
                a.sampler_label_ids.data() + lb);
        }
    }

    // Per-request BRLE logit mask. Optional — empty = no constraint.
    if (a.logit_mask_indptr.size() == static_cast<std::size_t>(a.n_request) + 1) {
        const std::int32_t lm_start = static_cast<std::int32_t>(a.logit_mask_indptr[r]);
        const std::int32_t lm_end   = static_cast<std::int32_t>(a.logit_mask_indptr[r + 1]);
        if (lm_end > lm_start) {
            rp.logit_mask_runs.assign(
                a.logit_masks.data() + lm_start,
                a.logit_masks.data() + lm_end);
        }
    }

    plan.reqs.push_back(std::move(rp));
}

void build_pure_decode_packing(ForwardEngine::BatchPlan& plan,
                               std::int32_t n_request,
                               std::int32_t sliding_window,
                               bool also_build_no_swa_mask) {
    const std::int32_t M = plan.max_n_kv;
    const std::int32_t N = n_request;
    plan.packed_gather_idxs.assign(static_cast<std::size_t>(M) * N, 0);
    plan.packed_mask_f16.assign(
        static_cast<std::size_t>(M) * MASK_PAD * N,
        ggml_fp32_to_fp16(-INFINITY));
    const bool build_full = also_build_no_swa_mask && sliding_window > 0;
    if (build_full) {
        plan.packed_mask_full_f16.assign(
            static_cast<std::size_t>(M) * MASK_PAD * N,
            ggml_fp32_to_fp16(-INFINITY));
    } else {
        plan.packed_mask_full_f16.clear();
    }
    const auto zero = ggml_fp32_to_fp16(0.0f);
    const std::int32_t W = sliding_window;
    for (std::int32_t r = 0; r < N; ++r) {
        const auto& rp = plan.reqs[r];
        for (std::int32_t k = 0; k < rp.n_kv; ++k) {
            plan.packed_gather_idxs[
                static_cast<std::size_t>(r) * M + k] = rp.gather_idxs[k];
        }
        // SWA-clipped row.
        std::uint16_t* row = plan.packed_mask_f16.data()
            + static_cast<std::size_t>(r) * M * MASK_PAD;
        const std::int32_t lo = (W > 0) ? std::max(0, rp.n_kv - W) : 0;
        for (std::int32_t k = lo; k < rp.n_kv; ++k) {
            row[k] = zero;
        }
        // Full (no SWA) row — only when caller asked for it.
        if (build_full) {
            std::uint16_t* row_full = plan.packed_mask_full_f16.data()
                + static_cast<std::size_t>(r) * M * MASK_PAD;
            for (std::int32_t k = 0; k < rp.n_kv; ++k) {
                row_full[k] = zero;
            }
        }
    }
}

}  // namespace pie_portable_driver
