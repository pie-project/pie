// pie_driver_abi/response_builder.hpp — driver-side per-fire response view.
//
// Each backend builds a `std::vector<PerRequestOutput>` from its sampler
// pipeline (one entry per request), then calls
// `ResponseBuilder::build(per_req, view)`. The builder concatenates the
// per-request arrays into reused scratch vectors and writes `PieSlice`s
// into `view` that point at that scratch. The view stays valid until
// the next `build()` call on the same builder (or the builder's
// destruction) — long enough for the `send_response` call that
// immediately follows, which is the contract `InProcServer` enforces.
//
// `PerRequestOutput` is the C++ side's owned per-request response
// buffer; the builder gathers these and stamps the indptr arrays in
// the wire-shape `pie_driver_abi::ForwardResponse`. Empty fields produce
// zero-length slots in the view — fine to leave unpopulated when the
// request didn't ask for that sampler type.
//
// Header-only: shared between the cuda and metal backends.

#pragma once

#include <cstddef>
#include <cstdint>
#include <span>
#include <utility>
#include <vector>

#include "view.hpp"

namespace pie_driver {

struct PerRequestOutput {
    // Accepted tokens (length 1 for plain sampling; 1..n_drafts+1 for
    // spec decode).
    std::vector<std::uint32_t> tokens;
    // One entry per `Sampler::Dist` slot. Both members of each pair
    // share length (top-K).
    std::vector<std::pair<std::vector<std::uint32_t>, std::vector<float>>> dists;
    // One byte-blob per `Sampler::RawLogits` slot. Each blob is
    // `vocab_size * sizeof(float)` bytes of native-endian f32.
    std::vector<std::vector<std::uint8_t>> logits;
    // One f32 vec per `Sampler::Logprob` (length 1) / `Sampler::Logprobs`
    // (length K) slot.
    std::vector<std::vector<float>> logprobs;
    // One f32 per `Sampler::Entropy` slot.
    std::vector<float> entropies;
    // Next-iteration system-speculation drafts.
    std::vector<std::uint32_t> spec_tokens;
    std::vector<std::uint32_t> spec_positions;
};

class ResponseBuilder {
public:
    ResponseBuilder() = default;
    ResponseBuilder(const ResponseBuilder&) = delete;
    ResponseBuilder& operator=(const ResponseBuilder&) = delete;

    // Concatenate per-request arrays into scratch, point `out`'s slices
    // at the scratch buffers. `out` is fully (re)written; any prior
    // contents are discarded.
    inline void build(std::span<const PerRequestOutput> per_request,
                      PieForwardResponseView& out) {
        const std::uint32_t R = static_cast<std::uint32_t>(per_request.size());

        tokens_indptr_.clear();
        tokens_.clear();
        dists_req_indptr_.clear();
        dists_kv_indptr_.clear();
        dists_ids_.clear();
        dists_probs_.clear();
        logits_req_indptr_.clear();
        logits_byte_indptr_.clear();
        logits_bytes_.clear();
        logprobs_req_indptr_.clear();
        logprobs_val_indptr_.clear();
        logprobs_values_.clear();
        entropies_indptr_.clear();
        entropies_.clear();
        spec_indptr_.clear();
        spec_tokens_.clear();
        spec_positions_.clear();

        tokens_indptr_.reserve(R + 1);
        dists_req_indptr_.reserve(R + 1);
        logits_req_indptr_.reserve(R + 1);
        logprobs_req_indptr_.reserve(R + 1);
        entropies_indptr_.reserve(R + 1);
        spec_indptr_.reserve(R + 1);

        // Two-level indptrs start with a leading 0 for the kv/byte/val side.
        dists_kv_indptr_.push_back(0);
        logits_byte_indptr_.push_back(0);
        logprobs_val_indptr_.push_back(0);

        for (std::uint32_t r = 0; r < R; ++r) {
            const auto& pr = per_request[r];

            tokens_indptr_.push_back(static_cast<std::uint32_t>(tokens_.size()));
            dists_req_indptr_.push_back(
                static_cast<std::uint32_t>(dists_kv_indptr_.size() - 1));
            logits_req_indptr_.push_back(
                static_cast<std::uint32_t>(logits_byte_indptr_.size() - 1));
            logprobs_req_indptr_.push_back(
                static_cast<std::uint32_t>(logprobs_val_indptr_.size() - 1));
            entropies_indptr_.push_back(static_cast<std::uint32_t>(entropies_.size()));
            spec_indptr_.push_back(static_cast<std::uint32_t>(spec_tokens_.size()));

            tokens_.insert(tokens_.end(), pr.tokens.begin(), pr.tokens.end());
            entropies_.insert(entropies_.end(), pr.entropies.begin(), pr.entropies.end());
            spec_tokens_.insert(spec_tokens_.end(),
                                pr.spec_tokens.begin(), pr.spec_tokens.end());
            spec_positions_.insert(spec_positions_.end(),
                                   pr.spec_positions.begin(), pr.spec_positions.end());

            for (const auto& [ids, probs] : pr.dists) {
                dists_ids_.insert(dists_ids_.end(), ids.begin(), ids.end());
                dists_probs_.insert(dists_probs_.end(), probs.begin(), probs.end());
                dists_kv_indptr_.push_back(static_cast<std::uint32_t>(dists_ids_.size()));
            }

            for (const auto& bytes : pr.logits) {
                logits_bytes_.insert(logits_bytes_.end(), bytes.begin(), bytes.end());
                logits_byte_indptr_.push_back(
                    static_cast<std::uint32_t>(logits_bytes_.size()));
            }

            for (const auto& values : pr.logprobs) {
                logprobs_values_.insert(logprobs_values_.end(),
                                         values.begin(), values.end());
                logprobs_val_indptr_.push_back(
                    static_cast<std::uint32_t>(logprobs_values_.size()));
            }
        }

        // Trailing indptr entries.
        tokens_indptr_.push_back(static_cast<std::uint32_t>(tokens_.size()));
        dists_req_indptr_.push_back(
            static_cast<std::uint32_t>(dists_kv_indptr_.size() - 1));
        logits_req_indptr_.push_back(
            static_cast<std::uint32_t>(logits_byte_indptr_.size() - 1));
        logprobs_req_indptr_.push_back(
            static_cast<std::uint32_t>(logprobs_val_indptr_.size() - 1));
        entropies_indptr_.push_back(static_cast<std::uint32_t>(entropies_.size()));
        spec_indptr_.push_back(static_cast<std::uint32_t>(spec_tokens_.size()));

        out = PieForwardResponseView{};
        out.num_requests = R;
        out.tokens_indptr        = slice_from(tokens_indptr_.data(), tokens_indptr_.size());
        out.tokens               = slice_from(tokens_.data(), tokens_.size());
        out.dists_req_indptr     = slice_from(dists_req_indptr_.data(), dists_req_indptr_.size());
        out.dists_kv_indptr      = slice_from(dists_kv_indptr_.data(), dists_kv_indptr_.size());
        out.dists_ids            = slice_from(dists_ids_.data(), dists_ids_.size());
        out.dists_probs          = slice_from(dists_probs_.data(), dists_probs_.size());
        out.logits_req_indptr    = slice_from(logits_req_indptr_.data(), logits_req_indptr_.size());
        out.logits_byte_indptr   = slice_from(logits_byte_indptr_.data(), logits_byte_indptr_.size());
        out.logits_bytes         = slice_from(logits_bytes_.data(), logits_bytes_.size());
        out.logprobs_req_indptr  = slice_from(logprobs_req_indptr_.data(), logprobs_req_indptr_.size());
        out.logprobs_val_indptr  = slice_from(logprobs_val_indptr_.data(), logprobs_val_indptr_.size());
        out.logprobs_values      = slice_from(logprobs_values_.data(), logprobs_values_.size());
        out.entropies_indptr     = slice_from(entropies_indptr_.data(), entropies_indptr_.size());
        out.entropies            = slice_from(entropies_.data(), entropies_.size());
        out.spec_indptr          = slice_from(spec_indptr_.data(), spec_indptr_.size());
        out.spec_tokens          = slice_from(spec_tokens_.data(), spec_tokens_.size());
        out.spec_positions       = slice_from(spec_positions_.data(), spec_positions_.size());
    }

    inline void build_token_only(std::span<const std::uint32_t> per_request_counts,
                                 std::span<const std::uint32_t> tokens,
                                 PieForwardResponseView& out) {
        reset_for_build(static_cast<std::uint32_t>(per_request_counts.size()));

        tokens_.assign(tokens.begin(), tokens.end());
        std::uint32_t cursor = 0;
        tokens_indptr_.push_back(0);
        for (std::uint32_t n : per_request_counts) {
            cursor += n;
            tokens_indptr_.push_back(cursor);
        }

        finish_view(static_cast<std::uint32_t>(per_request_counts.size()), out);
    }

    inline void build_token_only_dense(std::span<const std::int32_t> tokens,
                                       PieForwardResponseView& out) {
        const std::uint32_t R = static_cast<std::uint32_t>(tokens.size());
        reset_for_build(R);

        tokens_.resize(tokens.size());
        tokens_indptr_.resize(static_cast<std::size_t>(R) + 1);
        for (std::uint32_t r = 0; r < R; ++r) {
            tokens_[r] = static_cast<std::uint32_t>(tokens[r]);
            tokens_indptr_[r] = r;
        }
        tokens_indptr_[R] = R;

        finish_view(R, out);
    }

private:
    inline void reset_for_build(std::uint32_t R) {
        tokens_indptr_.clear();
        tokens_.clear();
        dists_req_indptr_.clear();
        dists_kv_indptr_.clear();
        dists_ids_.clear();
        dists_probs_.clear();
        logits_req_indptr_.clear();
        logits_byte_indptr_.clear();
        logits_bytes_.clear();
        logprobs_req_indptr_.clear();
        logprobs_val_indptr_.clear();
        logprobs_values_.clear();
        entropies_indptr_.clear();
        entropies_.clear();
        spec_indptr_.clear();
        spec_tokens_.clear();
        spec_positions_.clear();

        tokens_indptr_.reserve(static_cast<std::size_t>(R) + 1);
        dists_req_indptr_.assign(static_cast<std::size_t>(R) + 1, 0);
        logits_req_indptr_.assign(static_cast<std::size_t>(R) + 1, 0);
        logprobs_req_indptr_.assign(static_cast<std::size_t>(R) + 1, 0);
        entropies_indptr_.assign(static_cast<std::size_t>(R) + 1, 0);
        spec_indptr_.assign(static_cast<std::size_t>(R) + 1, 0);
        dists_kv_indptr_.assign(1, 0);
        logits_byte_indptr_.assign(1, 0);
        logprobs_val_indptr_.assign(1, 0);
    }

    inline void finish_view(std::uint32_t R, PieForwardResponseView& out) {
        out = PieForwardResponseView{};
        out.num_requests = R;
        out.tokens_indptr        = slice_from(tokens_indptr_.data(), tokens_indptr_.size());
        out.tokens               = slice_from(tokens_.data(), tokens_.size());
        out.dists_req_indptr     = slice_from(dists_req_indptr_.data(), dists_req_indptr_.size());
        out.dists_kv_indptr      = slice_from(dists_kv_indptr_.data(), dists_kv_indptr_.size());
        out.dists_ids            = slice_from(dists_ids_.data(), dists_ids_.size());
        out.dists_probs          = slice_from(dists_probs_.data(), dists_probs_.size());
        out.logits_req_indptr    = slice_from(logits_req_indptr_.data(), logits_req_indptr_.size());
        out.logits_byte_indptr   = slice_from(logits_byte_indptr_.data(), logits_byte_indptr_.size());
        out.logits_bytes         = slice_from(logits_bytes_.data(), logits_bytes_.size());
        out.logprobs_req_indptr  = slice_from(logprobs_req_indptr_.data(), logprobs_req_indptr_.size());
        out.logprobs_val_indptr  = slice_from(logprobs_val_indptr_.data(), logprobs_val_indptr_.size());
        out.logprobs_values      = slice_from(logprobs_values_.data(), logprobs_values_.size());
        out.entropies_indptr     = slice_from(entropies_indptr_.data(), entropies_indptr_.size());
        out.entropies            = slice_from(entropies_.data(), entropies_.size());
        out.spec_indptr          = slice_from(spec_indptr_.data(), spec_indptr_.size());
        out.spec_tokens          = slice_from(spec_tokens_.data(), spec_tokens_.size());
        out.spec_positions       = slice_from(spec_positions_.data(), spec_positions_.size());
    }

    // Concatenated bodies + R+1 indptrs. Reused fire-to-fire — `build()`
    // clears and refills.
    std::vector<std::uint32_t> tokens_indptr_;
    std::vector<std::uint32_t> tokens_;
    std::vector<std::uint32_t> dists_req_indptr_;
    std::vector<std::uint32_t> dists_kv_indptr_;
    std::vector<std::uint32_t> dists_ids_;
    std::vector<float>         dists_probs_;
    std::vector<std::uint32_t> logits_req_indptr_;
    std::vector<std::uint32_t> logits_byte_indptr_;
    std::vector<std::uint8_t>  logits_bytes_;
    std::vector<std::uint32_t> logprobs_req_indptr_;
    std::vector<std::uint32_t> logprobs_val_indptr_;
    std::vector<float>         logprobs_values_;
    std::vector<std::uint32_t> entropies_indptr_;
    std::vector<float>         entropies_;
    std::vector<std::uint32_t> spec_indptr_;
    std::vector<std::uint32_t> spec_tokens_;
    std::vector<std::uint32_t> spec_positions_;
};

}  // namespace pie_driver
