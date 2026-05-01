#include "request_handler.hpp"

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <span>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include "attention_workspace.hpp"
#include "brle.hpp"
#include "cuda_check.hpp"
#include "device_buffer.hpp"
#include "engine.hpp"
#include "kv_cache.hpp"
#include "msgpack_subpass.hpp"
#include "model/qwen3.hpp"
#include "model/qwen3_forward.hpp"
#include "ops/gemm.hpp"
#include "response_writer.hpp"
#include "sampler_type.hpp"
#include "sampling_dispatch.hpp"
#include "shmem_ipc.hpp"
#include "shmem_schema.hpp"
#include "spec_expansion.hpp"

namespace pie_cuda_driver {

std::size_t handle_fire_batch(
    const SlotRequest& req,
    std::span<std::uint8_t> response,
    ForwardContext& ctx,
    std::uint64_t handled)
{
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
    // forward kernel knows whether to consume `pi.custom_mask`.
    bool have_custom_mask = false;

    try {
        namespace S = pie_cuda_driver::schema;
        const auto dec = S::decode_request(req.payload);

        const auto tok_view_orig   = dec.as<std::uint32_t>(S::A_TOKEN_IDS);
        const auto pos_view_orig   = dec.as<std::uint32_t>(S::A_POSITION_IDS);
        const auto qo_view_orig    = dec.as<std::uint32_t>(S::A_QO_INDPTR);
        const auto kvpi_view = dec.as<std::uint32_t>(S::A_KV_PAGE_INDICES);
        const auto kvpp_view = dec.as<std::uint32_t>(S::A_KV_PAGE_INDPTR);
        const auto kvlpl_view_orig = dec.as<std::uint32_t>(S::A_KV_LAST_PAGE_LENS);
        const auto sidx_view_orig  = dec.as<std::uint32_t>(S::A_SAMPLING_INDICES);
        const auto sptr_view_orig  = dec.as<std::uint32_t>(S::A_SAMPLING_INDPTR);

        // Sampler params (per-sampler arrays in BPIQ wire). Decoded here
        // (rather than further down) so the spec expansion below can
        // append cloned entries for the verification block.
        const auto temp_view_orig  = dec.as<float>(S::A_SAMPLER_TEMPERATURES);
        const auto top_k_view_orig = dec.as<std::uint32_t>(S::A_SAMPLER_TOP_K);
        const auto top_p_view_orig = dec.as<float>(S::A_SAMPLER_TOP_P);
        const auto min_p_view_orig = dec.as<float>(S::A_SAMPLER_MIN_P);
        const auto types_view_orig = dec.as<std::uint32_t>(S::A_SAMPLER_TYPES);
        const auto seed_view_orig  = dec.as<std::uint32_t>(S::A_SAMPLER_SEEDS);
        const auto rns_view_orig   = dec.as<std::uint32_t>(S::A_REQUEST_NUM_SAMPLERS);

        // Spec-decoding wire fields. When non-empty for some request,
        // splice drafts into the forward and append a verification
        // block to the sampling layout (one extra sample per draft +
        // one bonus). Mirrors pie_driver's `get_spec_expanded_*`.
        const auto spec_tok_view  = dec.as<std::uint32_t>(S::A_SPEC_TOKEN_IDS);
        const auto spec_pos_view  = dec.as<std::uint32_t>(S::A_SPEC_POSITION_IDS);
        const auto spec_iptr_view = dec.as<std::uint32_t>(S::A_SPEC_INDPTR);
        const bool has_spec_drafts = !spec_tok_view.empty();

        const int R = static_cast<int>(qo_view_orig.size()) - 1;

        // Spec-decoding batch expansion. When `has_spec_drafts` is false
        // the result has empty vectors and `verify_slot_start[r] == -1`
        // for every r; the active spans below fall through to the
        // original BPIQ views.
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
        // BPIQ wire. The rest of the function uses these.
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
            // Empty batch — emit a zero-token flat response.
            std::vector<std::uint32_t> counts(std::max(R, 0), 0u);
            return pie_cuda_driver::response::write_flat_response(
                response, counts, {});
        }
        if (N > max_workspace_tokens) {
            std::cerr << "[pie-driver-cuda] batch tokens=" << N
                      << " exceeds workspace=" << max_workspace_tokens << "\n";
            return 0;
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

        // Refill persistent device buffers with this fire's BPIQ inputs.
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
        const auto fmask_view  = dec.as<std::uint32_t>(S::A_FLATTENED_MASKS);
        const auto mskptr_view = dec.as<std::uint32_t>(S::A_MASK_INDPTR);
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
                have_custom_mask = true;
            }
        }

        // Forward pass. The graph-capture path activates only when the
        // request handler decided this fire is pure-decode AND the
        // caller passed a graph cache (`--cuda-graphs`). All other
        // fires take the direct dispatch path that just calls the
        // forward function.
        const bool try_graphs =
            ctx.graph_cache != nullptr && is_pure_decode && !have_custom_mask;
        if (try_graphs) {
            const ForwardGraphKey key{R};
            cudaGraphExec_t exec = ctx.graph_cache->get(key);
            if (exec == nullptr) {
                // First fire of this shape: capture. Forward writes its
                // output to `ws` workspace buffers + `cache.k/v` pages.
                // Persistent inputs (pi.*) provide stable kernel-arg
                // pointers; the next replay reads new contents from the
                // same addresses.
                cudaStream_t cstream = nullptr;
                CUDA_CHECK(cudaStreamCreateWithFlags(&cstream, cudaStreamNonBlocking));
                CUDA_CHECK(cudaStreamBeginCapture(cstream, cudaStreamCaptureModeRelaxed));
                forward_fn(
                    ws, kv_cache, attn_ws, cublas,
                    reinterpret_cast<const std::int32_t*>(pi.tokens.data()),
                    reinterpret_cast<const std::int32_t*>(pi.positions.data()),
                    pi.qo_indptr.data(), pi.kv_page_indices.data(),
                    pi.kv_page_indptr.data(), pi.kv_last_page_lens.data(),
                    h_qo, h_kvpp,
                    N, R, is_pure_decode, nullptr, nullptr);
                cudaGraph_t graph = nullptr;
                CUDA_CHECK(cudaStreamEndCapture(cstream, &graph));
                CUDA_CHECK(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
                cudaGraphDestroy(graph);
                cudaStreamDestroy(cstream);
                ctx.graph_cache->put(key, exec);
                // Throttle the capture log: first 4 captures, then every
                // 16th. At saturation R can swing across many distinct
                // values, and one line per shape blows up the log.
                const auto sz = ctx.graph_cache->size();
                if (sz <= 4 || sz % 16 == 0) {
                    std::cerr << "[pie-driver-cuda] graph captured: R=" << R
                              << " (cache size=" << sz << ")\n";
                }
            } else {
                CUDA_CHECK(cudaGraphLaunch(exec, /*stream=*/nullptr));
            }
        } else {
            forward_fn(
                ws, kv_cache, attn_ws, cublas,
                reinterpret_cast<const std::int32_t*>(pi.tokens.data()),
                reinterpret_cast<const std::int32_t*>(pi.positions.data()),
                pi.qo_indptr.data(), pi.kv_page_indices.data(),
                pi.kv_page_indptr.data(), pi.kv_last_page_lens.data(),
                /*qo_indptr_h=*/h_qo,
                /*kv_page_indptr_h=*/h_kvpp,
                N, R, is_pure_decode,
                have_custom_mask ? pi.custom_mask.data()        : nullptr,
                have_custom_mask ? pi.custom_mask_indptr.data() : nullptr);
        }

        // Sampler param views (temp_view, top_k_view, …, rns_view) were
        // hoisted to the top of the request handler to support the spec
        // expansion. Only `outspec_view` is decoded here.
        const auto outspec_view = dec.as<std::uint8_t>(S::A_OUTPUT_SPEC_FLAGS);

        // Detect msgpack slow path. Sampler types {0, 7, 8, 9, 10} cover
        // Logprob / Logprobs / RawLogits / Entropy / dist-output. Output
        // spec flags trigger the slow path too. Mirrors Python's
        // `_SPECIAL_SAMPLERS` predicate in shmem_schema.py.
        bool need_msgpack = false;
        for (auto t : types_view) {
            if (pie_cuda_driver::is_msgpack_only(t)) { need_msgpack = true; break; }
        }
        if (!need_msgpack) {
            for (auto f : outspec_view) {
                if (f) { need_msgpack = true; break; }
            }
        }
        // Spec verification produces variable-length per-request token
        // lists (the accepted prefix). The runtime detects this via
        // tokens.len() != expected_token_slots, but the flat path bakes
        // per_request_counts from the sampling layout — wrong for spec.
        // Force msgpack so we can write per_req[r].tokens precisely.
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

        // Per-slot sampler type. Each (sampling_index, sampler) pair is
        // one slot, and `request_num_samplers[r]` == sampling_indptr step
        // for r — so global slot k maps to global sampler index
        // `sampler_off(r) + (k - h_sptr[r])`.
        std::vector<std::uint32_t> per_slot_type(num_sampling, 1u);
        // Per-slot sampler params (used by token samplers; ignored
        // for special types but kept aligned for indexing simplicity).
        std::vector<float>         per_slot_temp (num_sampling, 0.f);
        std::vector<float>         per_slot_top_p(num_sampling, 1.f);
        std::vector<float>         per_slot_min_p(num_sampling, 0.f);
        std::vector<std::int32_t>  per_slot_top_k(num_sampling, 0);
        std::vector<std::uint32_t> per_slot_seed (num_sampling, 0u);

        bool any_topk_topp = false;
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
                // BPIQ uses 0 to mean "no top-k filter"; flashinfer
                // interprets 0 as "keep zero tokens" (always returns 0).
                // Map to vocab so the filter is a no-op.
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

                // Token-sampler types: 1=Multinomial, 2=TopK, 3=TopP,
                // 4=MinP, 5=TopKTopP. Only these consume the row-indexed
                // sampling kernel output; non-token slots get their data
                // from the msgpack-only sub-passes below.
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

        // d_sampled lives in `pi.sampled` (capacity = max_workspace_tokens).
        // Only the first N rows are written/read this fire.

        // Map each sampling slot to its global logit-row index. Only the
        // topk+top-p path uses this; we build it unconditionally so the
        // sampling-dispatch helper has a uniform input shape.
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

        dispatch_sampling(
            ws, pi.sampled.data(),
            SamplingPlan{
                any_topk_topp,
                std::span<const float>(h_per_temp),
                std::span<const float>(h_per_top_p),
                std::span<const float>(h_per_min_p),
                std::span<const std::int32_t>(h_per_top_k),
                std::span<const std::uint32_t>(h_per_seed),
                std::span<const std::int32_t>(h_sample_idx),
            },
            N, num_sampling, engine.hf_config().vocab_size,
            /*prng_offset=*/static_cast<std::uint64_t>(handled));

        // Only copy the first N entries — `pi.sampled` is sized for
        // max_workspace_tokens, but only [0, N) are valid this fire.
        std::vector<std::int32_t> all_sampled(N);
        CUDA_CHECK(cudaMemcpy(all_sampled.data(), pi.sampled.data(),
                              sizeof(std::int32_t) * N,
                              cudaMemcpyDeviceToHost));

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

        std::size_t resp_bytes;
        if (need_msgpack) {
            std::vector<pie_cuda_driver::response::PerRequestMsgpack> per_req(R);

            const MsgpackSubpassContext sub_ctx{
                ws,
                R, num_sampling, engine.hf_config().vocab_size,
                std::span<const std::uint32_t>(per_slot_type),
                std::span<const float>(per_slot_temp),
                std::span<const std::int32_t>(per_slot_top_k),
                qo_view, sptr_view, sidx_view, rns_view,
            };
            gather_raw_logits(sub_ctx, per_req);
            compute_entropy_slots(sub_ctx, per_req);
            compute_logprob_slots(sub_ctx, dec, per_req);
            compute_dist_slots(sub_ctx, per_req);

            // Per-request token list. For non-spec requests this is the
            // token-typed slots' samples (regular path). For spec
            // requests we walk the verification block (cloned token
            // samplers at the bonus + each draft position) and produce
            // the accepted prefix; the inferlet's own samples for that
            // request are discarded.
            std::vector<std::vector<std::uint32_t>> per_req_tokens(R);
            for (int r = 0; r < R; ++r) {
                const std::uint32_t qo_lo = h_qo[r];
                auto& bucket = per_req_tokens[r];

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
                per_req[r].tokens = std::span<const std::uint32_t>(
                    bucket.data(), bucket.size());
            }
            resp_bytes = pie_cuda_driver::response::write_msgpack_response(
                response, per_req);
        } else {
            resp_bytes = pie_cuda_driver::response::write_flat_response(
                response, per_request_counts, sampled_tokens);
        }

        if (handled <= 4 || handled % 100 == 0) {
            std::cerr << "[pie-driver-cuda] req_id=" << req.req_id
                      << " R=" << R << " N=" << N
                      << " sampled=" << num_sampling
                      << " max_kv=" << max_kv_len
                      << " resp=" << resp_bytes << "B\n";
        }
        return resp_bytes;

    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-cuda] fire_batch failed for req_id="
                  << req.req_id << ": " << e.what() << "\n";
        return 0;
    }
}

}  // namespace pie_cuda_driver
