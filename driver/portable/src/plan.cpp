#include "plan.hpp"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>

#include <ggml.h>

#include "sampler.hpp"

namespace pie_portable_driver {

// Phi-3-small blocksparse mask: like build_attn_mask_f16 (causal +
// optional BRLE override) but additionally enforces the blocksparse
// pattern. A position kv_pos is allowed iff:
//   (kv_pos in local window: q_pos - kv_pos < num_local_blocks * block_size)
// OR
//   (kv_pos in vertical-stride block: (kv_pos / block_size) % vert_stride == 0)
// AND causal still applies (kv_pos <= q_pos). For seq_len within the
// local window AND no stride blocks beyond it, this is identical to
// the causal mask — short prompts are not affected.
void build_phi3small_blocksparse_mask_f16(
        std::vector<std::uint16_t>& dst,
        std::int32_t n_kv, std::int32_t n_tokens, std::int32_t n_tokens_pad,
        const std::int32_t* positions,
        std::int32_t block_size, std::int32_t num_local_blocks,
        std::int32_t vert_stride) {
    dst.assign(static_cast<std::size_t>(n_kv) * n_tokens_pad,
               ggml_fp32_to_fp16(-INFINITY));
    const auto zero = ggml_fp32_to_fp16(0.0f);
    const std::int32_t W = num_local_blocks * block_size;
    for (std::int32_t i = 0; i < n_tokens; ++i) {
        const std::int32_t p_i = positions[i];
        std::uint16_t* row = dst.data() + static_cast<std::size_t>(i) * n_kv;
        const std::int32_t hi = std::min(n_kv - 1, p_i);
        const std::int32_t local_lo = std::max(0, p_i - W + 1);
        // Local window (causal-clipped).
        for (std::int32_t j = local_lo; j <= hi; ++j) row[j] = zero;
        // Vertical-stride blocks: every vert_stride-th block from start
        // is "always attended" (subject to causal). Iterate kv blocks
        // outside the local window.
        const std::int32_t local_lo_block = local_lo / block_size;
        for (std::int32_t b = 0; b < local_lo_block; b += vert_stride) {
            const std::int32_t a = b * block_size;
            const std::int32_t e = std::min(a + block_size, hi + 1);
            for (std::int32_t j = a; j < e; ++j) row[j] = zero;
        }
    }
}

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

PlanArrays extract_plan_arrays(const pie_driver::PieForwardRequestView& req) {
    PlanArrays a;
    a.context_ids          = req.context_ids.as<std::uint64_t>();
    a.token_ids            = req.token_ids.as<std::uint32_t>();
    a.position_ids         = req.position_ids.as<std::uint32_t>();
    a.qo_indptr            = req.qo_indptr.as<std::uint32_t>();
    a.sampling_idx         = req.sampling_indices.as<std::uint32_t>();
    a.sampling_indptr      = req.sampling_indptr.as<std::uint32_t>();
    a.request_num_samplers = req.request_num_samplers.as<std::uint32_t>();
    a.kv_page_indices      = req.kv_page_indices.as<std::uint32_t>();
    a.kv_page_indptr       = req.kv_page_indptr.as<std::uint32_t>();
    a.kv_last_lens         = req.kv_last_page_lens.as<std::uint32_t>();
    a.sampler_types        = req.sampler_types.as<std::uint32_t>();
    a.sampler_temps        = req.sampler_temperatures.as<float>();
    a.sampler_top_k        = req.sampler_top_k.as<std::uint32_t>();
    a.sampler_top_p        = req.sampler_top_p.as<float>();
    a.sampler_min_p        = req.sampler_min_p.as<float>();
    a.sampler_seeds        = req.sampler_seeds.as<std::uint32_t>();
    a.sampler_label_ids    = req.sampler_label_ids.as<std::uint32_t>();
    a.sampler_label_indptr = req.sampler_label_indptr.as<std::uint32_t>();
    a.logit_masks          = req.logit_masks.as<std::uint32_t>();
    a.logit_mask_indptr    = req.logit_mask_indptr.as<std::uint32_t>();
    a.flat_attn_masks      = req.flattened_masks.as<std::uint32_t>();
    a.attn_mask_indptr     = req.mask_indptr.as<std::uint32_t>();
    a.adapter_indices      = req.adapter_indices.as<std::int64_t>();
    a.spec_token_ids       = req.spec_token_ids.as<std::uint32_t>();
    a.spec_position_ids    = req.spec_position_ids.as<std::uint32_t>();
    a.spec_indptr          = req.spec_indptr.as<std::uint32_t>();

    a.n_request      = static_cast<std::int32_t>(a.context_ids.size());
    a.total_n_tokens = static_cast<std::int32_t>(a.token_ids.size());
    a.batch_has_drafts =
        a.spec_indptr.size() == static_cast<std::size_t>(a.n_request) + 1;
    // Detect "no user mask" so we can take the M11 packed-decode fast
    // path. The runtime always sends an attention mask (synthesizing
    // Brle::all_true(pos+1) per row when the user supplied none, see
    // runtime/src/api/inference.rs:341-352), so .empty() is never true.
    // Instead we structurally recognize the synthesized causal pattern:
    // every BRLE row is exactly [0, pos+1] (2 u32 elements).
    a.batch_has_attn_masks = false;
    if (!a.flat_attn_masks.empty() &&
        a.attn_mask_indptr.size() ==
            static_cast<std::size_t>(a.total_n_tokens) + 1) {
        // First-pass cheap reject: any row whose byte step != 2 u32s
        // is necessarily a user mask.
        bool maybe_synthesized = true;
        for (std::int32_t i = 0; i < a.total_n_tokens; ++i) {
            const std::uint32_t lo = a.attn_mask_indptr[i];
            const std::uint32_t hi = a.attn_mask_indptr[i + 1];
            if (hi - lo != 2u) { maybe_synthesized = false; break; }
        }
        if (maybe_synthesized) {
            // Second pass: confirm values match [0, position+1]. Position
            // ids are u32 here; pos+1 cannot overflow within u32 for any
            // realistic context length.
            for (std::int32_t i = 0; i < a.total_n_tokens; ++i) {
                const std::uint32_t off = a.attn_mask_indptr[i];
                const std::uint32_t v0  = a.flat_attn_masks[off + 0];
                const std::uint32_t v1  = a.flat_attn_masks[off + 1];
                const std::uint32_t expected = a.position_ids[i] + 1u;
                if (v0 != 0u || v1 != expected) {
                    maybe_synthesized = false;
                    break;
                }
            }
        }
        a.batch_has_attn_masks = !maybe_synthesized;
    }
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
                         Executor::BatchPlan& plan) {
    // 0 sampler slots = prefill-only (e.g. `Context::flush`): write KV for
    // the supplied tokens, no logit sampling. Decode is 1 slot, M8
    // spec-decode is 1 + n_drafts.
    const bool prefill_only = (a.request_num_samplers[r] == 0);

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

    // wire slot count must agree with the prefill/decode mode.
    const std::int32_t s_start = static_cast<std::int32_t>(a.sampling_indptr[r]);
    const std::int32_t s_end   = static_cast<std::int32_t>(a.sampling_indptr[r + 1]);
    const std::int32_t n_bpiq_slots = s_end - s_start;
    const std::int32_t expected_min = prefill_only ? 0 : 1;
    if (n_bpiq_slots < expected_min || (prefill_only && n_bpiq_slots != 0)) {
        throw std::runtime_error(
            "plan: request " + std::to_string(r) + " has " +
            std::to_string(a.request_num_samplers[r]) +
            " sampler slot(s) but " + std::to_string(n_bpiq_slots) +
            " wire sampling indices");
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

    Executor::ReqPlan rp;
    rp.qo_start     = qo_start;
    rp.n_tokens     = n_tok;
    rp.n_tokens_pad = ((n_tok + MASK_PAD - 1) / MASK_PAD) * MASK_PAD;
    rp.n_kv         = seq_len;

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

    // Phi-3-small: also build the blocksparse mask. Diverges from
    // mask_f16 only when seq_len exceeds num_local_blocks * block_size
    // (= 1024 with default params), so short-prompt smoke tests are
    // bit-identical regardless of dispatch. block_size > 0 in the spec
    // signals the arch is Phi-3-small.
    if (spec.phi3small_block_size > 0) {
        build_phi3small_blocksparse_mask_f16(
            rp.mask_blocksparse_f16, seq_len, n_tok, rp.n_tokens_pad,
            plan.positions_i32.data() + qo_start,
            spec.phi3small_block_size,
            spec.phi3small_num_local_blocks,
            spec.phi3small_vert_stride);
    }

    if (prefill_only) {
        // No sampling slots; sampler/labels/logit_mask stay default and the
        // host-side sample loop emits an empty SamplerOutput for this request.
        plan.reqs.push_back(std::move(rp));
        return;
    }

    // Decode-only: sampling slots, sampler params, optional logit mask.

    // Runtime emits sampling_indices as per-request relative offsets
    // (0 = first token of that request's qo range, n_tok-1 = last). We
    // carry global flat-array positions through the plan, so add qo_start.
    // A request can carry multiple samplers (the SDK uses this for
    // speculative-decoding verify passes that route drafts through the
    // regular `input_tokens` channel rather than `spec_token_ids`), so
    // honor every entry in [s_start, s_end).
    for (std::int32_t k = s_start; k < s_end; ++k) {
        const std::int32_t rel = static_cast<std::int32_t>(a.sampling_idx[k]);
        if (rel < 0 || rel >= n_tok) {
            throw std::runtime_error(
                "plan: request " + std::to_string(r) +
                " sampling_index " + std::to_string(rel) +
                " out of relative [0," + std::to_string(n_tok) + ")");
        }
        const std::int32_t idx = qo_start + rel;
        rp.sampling_positions.push_back(idx);
        plan.sampling_pos_i32.push_back(idx);
    }

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

    // Per-slot sampler params. The inferlet SDK can attach multiple
    // sampler kinds to one position in a single forward pass (e.g.
    // Argmax + RawLogits + Distribution + Logprob + Logprobs + Entropy
    // all at the same slot — see tests/inferlets/test_sampler_suite.py),
    // and the wire format pairs each entry in `sampling_indices` with
    // the same-indexed entries in the per-sampler SoA arrays. Build one
    // `SamplerParams` per `[s_start, s_end)` slot and store them on the
    // request plan in slot order; `rp.sampler` (singular) tracks the
    // first one for fast-path detection + graph-side reads.
    const std::int32_t total_slots =
        rp.sampling_positions.empty() ? 0
        : static_cast<std::int32_t>(rp.sampling_positions.size());
    if (total_slots > 0 &&
        s_start + total_slots > static_cast<std::int32_t>(a.sampler_types.size())) {
        throw std::runtime_error(
            "plan: sampler_types too short for request " + std::to_string(r));
    }
    rp.samplers.resize(static_cast<std::size_t>(total_slots));
    for (std::int32_t k = 0; k < total_slots; ++k) {
        const std::int32_t si = s_start + k;
        auto opt_at = [&](auto span, auto fallback) {
            return si < static_cast<std::int32_t>(span.size())
                ? span[si] : fallback;
        };
        SamplerParams& sp = rp.samplers[static_cast<std::size_t>(k)];
        sp.type        = static_cast<SamplerType>(a.sampler_types[si]);
        sp.temperature = opt_at(a.sampler_temps, 1.0f);
        sp.top_k       = opt_at(a.sampler_top_k, 0u);
        sp.top_p       = opt_at(a.sampler_top_p, 1.0f);
        sp.min_p       = opt_at(a.sampler_min_p, 0.0f);
        sp.seed        = opt_at(a.sampler_seeds, 0u);

        // Per-slot Logprob/Logprobs labels (sampler-slot-keyed indptr).
        if (a.sampler_label_indptr.size() > static_cast<std::size_t>(si) + 1) {
            const std::int32_t la = static_cast<std::int32_t>(a.sampler_label_indptr[si]);
            const std::int32_t lb = static_cast<std::int32_t>(a.sampler_label_indptr[si + 1]);
            if (lb > la) {
                sp.labels.assign(
                    a.sampler_label_ids.data() + la,
                    a.sampler_label_ids.data() + lb);
            }
        }
    }
    // `rp.sampler` is the legacy single-handle used by graph builders
    // and the GPU fast-path detection. Populate from slot 0 when present;
    // leave as default for prefill-only requests.
    if (!rp.samplers.empty()) {
        rp.sampler = rp.samplers[0];
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

void build_pure_decode_packing(Executor::BatchPlan& plan,
                               std::int32_t n_request,
                               std::int32_t page_size,
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
    // Paged-attn inputs (FlashInfer / vLLM shape). Derived from
    // rp.gather_idxs: for block b, page id = gather_idxs[b*page_size] /
    // page_size (since gather_idxs[k] = page_id[k/page_size]*page_size +
    // (k % page_size)). Variable-length flat layout via prefix sums.
    plan.page_indptr_i32.assign(static_cast<std::size_t>(N + 1), 0);
    plan.last_page_lens_i32.assign(static_cast<std::size_t>(N), 0);
    plan.page_indices_i32.clear();
    plan.page_indices_i32.reserve(static_cast<std::size_t>(M / page_size) * N);
    const auto zero = ggml_fp32_to_fp16(0.0f);
    const std::int32_t W = sliding_window;
    for (std::int32_t r = 0; r < N; ++r) {
        const auto& rp = plan.reqs[r];
        for (std::int32_t k = 0; k < rp.n_kv; ++k) {
            plan.packed_gather_idxs[
                static_cast<std::size_t>(r) * M + k] = rp.gather_idxs[k];
        }
        const std::int32_t num_pages_r =
            (rp.n_kv + page_size - 1) / page_size;
        for (std::int32_t b = 0; b < num_pages_r; ++b) {
            plan.page_indices_i32.push_back(
                rp.gather_idxs[b * page_size] / page_size);
        }
        plan.page_indptr_i32[r + 1] =
            plan.page_indptr_i32[r] + num_pages_r;
        // Last-page slot count: 1..page_size.
        const std::int32_t tail = rp.n_kv - (num_pages_r - 1) * page_size;
        plan.last_page_lens_i32[r] = tail;
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
