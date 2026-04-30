#include "sampler.hpp"

#include <algorithm>
#include <atomic>
#include <cmath>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace pie_ggml_driver {

namespace {

// Greedy argmax — used when temperature ≈ 0 or for any final fallback.
std::uint32_t argmax(const float* logits, std::int32_t vocab_size) {
    std::int32_t best = 0;
    float best_v = logits[0];
    for (std::int32_t v = 1; v < vocab_size; ++v) {
        if (logits[v] > best_v) {
            best_v = logits[v];
            best = v;
        }
    }
    return static_cast<std::uint32_t>(best);
}

// PCG32 with a stable two-input init (slot index + user seed) so a process
// running with the same wire-level seeds produces the same tokens.
class Pcg32 {
public:
    explicit Pcg32(std::uint64_t seed, std::uint64_t stream = 0xda3e39cb94b95bdbULL) {
        state_ = 0;
        inc_   = (stream << 1u) | 1u;
        next_();
        state_ += seed;
        next_();
    }
    std::uint32_t next_() {
        const std::uint64_t old = state_;
        state_ = old * 6364136223846793005ULL + inc_;
        const std::uint32_t xorshifted =
            static_cast<std::uint32_t>(((old >> 18u) ^ old) >> 27u);
        const std::uint32_t rot = static_cast<std::uint32_t>(old >> 59u);
        return (xorshifted >> rot) | (xorshifted << ((-static_cast<std::int32_t>(rot)) & 31));
    }
    // Uniform float in [0, 1).
    float next_uniform() {
        constexpr float kInv = 1.0f / static_cast<float>(1ull << 24);
        // Take the top 24 bits — float has 24 bits of mantissa.
        return (next_() >> 8) * kInv;
    }
private:
    std::uint64_t state_;
    std::uint64_t inc_;
};

// Process-global RNG — used when no per-slot seed is provided. Seeded once
// from a CPU clock so re-runs differ; concurrent calls share a single
// counter (atomic increment) so output is still well-defined per call.
std::uint64_t default_rng_seed() {
    static std::atomic<std::uint64_t> ctr{0};
    return 0x9E3779B97F4A7C15ULL ^ (ctr.fetch_add(1, std::memory_order_relaxed));
}

// Apply temperature scaling and softmax in place. After this call,
// `probs[v]` is a probability distribution.
void softmax_with_temperature(float* probs, std::int32_t vocab_size,
                              float temperature) {
    // Numerically stable softmax: subtract max before exp.
    float max_logit = probs[0];
    for (std::int32_t v = 1; v < vocab_size; ++v) {
        if (probs[v] > max_logit) max_logit = probs[v];
    }
    const float inv_t = 1.0f / std::max(temperature, 1e-5f);
    double sum = 0.0;
    for (std::int32_t v = 0; v < vocab_size; ++v) {
        const float e = std::exp((probs[v] - max_logit) * inv_t);
        probs[v] = e;
        sum += e;
    }
    const float inv_sum = static_cast<float>(1.0 / std::max(1e-30, sum));
    for (std::int32_t v = 0; v < vocab_size; ++v) {
        probs[v] *= inv_sum;
    }
}

// Categorical sample by inverse-CDF.
std::uint32_t categorical_sample(const std::vector<std::pair<float, std::uint32_t>>& sorted,
                                 float total, Pcg32& rng) {
    const float u = rng.next_uniform() * total;
    float acc = 0.0f;
    for (const auto& [p, id] : sorted) {
        acc += p;
        if (u < acc) return id;
    }
    return sorted.back().second;  // numerical safety
}

}  // namespace

void apply_brle_logit_mask(float* logits,
                           std::int32_t vocab_size,
                           const std::uint32_t* runs,
                           std::size_t n_runs) {
    if (n_runs == 0) return;  // empty BRLE = no constraint

    // Non-empty BRLE: start all-masked, then unmask `true` runs.
    // BRLE always begins with a (possibly-zero-length) `false` run, then
    // alternates false / true / false / true / ...
    //
    // We can't trivially "start all-masked" without a second pass, so
    // walk the runs once and only write -INF to the false runs. Anything
    // past the last run is implicitly false (mask it).
    bool is_true_run = false;
    std::int32_t pos = 0;
    for (std::size_t i = 0; i < n_runs && pos < vocab_size; ++i) {
        const std::int32_t len = static_cast<std::int32_t>(runs[i]);
        const std::int32_t end = std::min<std::int32_t>(pos + len, vocab_size);
        if (!is_true_run) {
            for (std::int32_t v = pos; v < end; ++v) {
                logits[v] = -std::numeric_limits<float>::infinity();
            }
        }
        pos = end;
        is_true_run = !is_true_run;
    }
    for (std::int32_t v = pos; v < vocab_size; ++v) {
        logits[v] = -std::numeric_limits<float>::infinity();
    }
}

// ============================================================================
// Special samplers (M10)
// ============================================================================

namespace {

// Compute log_softmax in place: out[v] = logits[v] - logsumexp(logits).
// Returns the resulting buffer; numerically stable.
std::vector<float> log_softmax(const float* logits, std::int32_t vocab_size) {
    float max_l = logits[0];
    for (std::int32_t v = 1; v < vocab_size; ++v) {
        if (logits[v] > max_l) max_l = logits[v];
    }
    double sum = 0.0;
    for (std::int32_t v = 0; v < vocab_size; ++v) {
        sum += std::exp(static_cast<double>(logits[v] - max_l));
    }
    const float lse = max_l + static_cast<float>(std::log(std::max(1e-30, sum)));
    std::vector<float> lp(vocab_size);
    for (std::int32_t v = 0; v < vocab_size; ++v) {
        lp[v] = logits[v] - lse;
    }
    return lp;
}

void run_distribution(const float* logits, std::int32_t vocab_size,
                      const SamplerParams& params, SlotOutput& out) {
    // Top-K probs (with default temperature scaling). Pie's reference
    // uses the same temperature/top_k params as token-producing samplers
    // for the top-K size. Default K = 8 if unspecified.
    std::vector<float> probs(logits, logits + vocab_size);
    softmax_with_temperature(probs.data(), vocab_size, params.temperature);

    const std::int32_t k = params.top_k > 0
        ? std::min<std::int32_t>(params.top_k, vocab_size)
        : std::min<std::int32_t>(8, vocab_size);

    std::vector<std::pair<float, std::int32_t>> sorted;
    sorted.reserve(vocab_size);
    for (std::int32_t v = 0; v < vocab_size; ++v) {
        sorted.emplace_back(probs[v], v);
    }
    std::partial_sort(sorted.begin(), sorted.begin() + k, sorted.end(),
                      [](const auto& a, const auto& b) { return a.first > b.first; });

    out.has_dist = true;
    out.dist_ids.resize(k);
    out.dist_vals.resize(k);
    for (std::int32_t i = 0; i < k; ++i) {
        out.dist_ids[i] = static_cast<std::uint32_t>(sorted[i].second);
        out.dist_vals[i] = sorted[i].first;
    }
}

void run_raw_logits(const float* logits, std::int32_t vocab_size,
                    SlotOutput& out) {
    // Native-endian f32 bytes — matches Pie's reference (numpy
    // `.tobytes()` of an f32 array).
    out.raw_logits.resize(static_cast<std::size_t>(vocab_size) * sizeof(float));
    std::memcpy(out.raw_logits.data(), logits,
                static_cast<std::size_t>(vocab_size) * sizeof(float));
}

void run_logprob(const float* logits, std::int32_t vocab_size,
                 const SamplerParams& params, SlotOutput& out) {
    if (params.labels.empty()) {
        throw std::runtime_error("sampler: Logprob slot has no labels");
    }
    const auto lp = log_softmax(logits, vocab_size);
    const std::uint32_t label = params.labels[0];
    if (label >= static_cast<std::uint32_t>(vocab_size)) {
        throw std::runtime_error("sampler: Logprob label id out of vocab");
    }
    out.logprobs.assign({lp[label]});
}

void run_logprobs_many(const float* logits, std::int32_t vocab_size,
                       const SamplerParams& params, SlotOutput& out) {
    if (params.labels.empty()) {
        throw std::runtime_error("sampler: Logprobs slot has no labels");
    }
    const auto lp = log_softmax(logits, vocab_size);
    out.logprobs.resize(params.labels.size());
    for (std::size_t i = 0; i < params.labels.size(); ++i) {
        const std::uint32_t label = params.labels[i];
        if (label >= static_cast<std::uint32_t>(vocab_size)) {
            throw std::runtime_error("sampler: Logprobs label id out of vocab");
        }
        out.logprobs[i] = lp[label];
    }
}

void run_entropy(const float* logits, std::int32_t vocab_size,
                 SlotOutput& out) {
    // H(p) = -sum_v p_v * log p_v   = -sum_v exp(lp_v) * lp_v.
    const auto lp = log_softmax(logits, vocab_size);
    double H = 0.0;
    for (std::int32_t v = 0; v < vocab_size; ++v) {
        const double p = std::exp(static_cast<double>(lp[v]));
        H -= p * lp[v];
    }
    out.has_entropy = true;
    out.entropy = static_cast<float>(H);
}

}  // namespace

void sample_slot(const float* logits,
                 std::int32_t vocab_size,
                 const SamplerParams& params,
                 SlotOutput& out) {
    using T = SamplerType;
    switch (params.type) {
        case T::Multinomial:
        case T::TopK:
        case T::TopP:
        case T::MinP:
        case T::TopKTopP:
            out.token = sample_token(logits, vocab_size, params);
            return;
        case T::Distribution:
            run_distribution(logits, vocab_size, params, out);
            return;
        case T::RawLogits:
            run_raw_logits(logits, vocab_size, out);
            return;
        case T::Logprob:
            run_logprob(logits, vocab_size, params, out);
            return;
        case T::Logprobs:
            run_logprobs_many(logits, vocab_size, params, out);
            return;
        case T::Entropy:
            run_entropy(logits, vocab_size, out);
            return;
    }
    throw std::runtime_error(
        "sampler: unknown type " +
        std::to_string(static_cast<int>(params.type)));
}

std::uint32_t sample_token(const float* logits,
                           std::int32_t vocab_size,
                           const SamplerParams& params) {
    using T = SamplerType;
    switch (params.type) {
        case T::Multinomial:
        case T::TopK:
        case T::TopP:
        case T::MinP:
        case T::TopKTopP:
            break;
        default:
            throw std::runtime_error(
                "sampler: type " +
                std::to_string(static_cast<int>(params.type)) +
                " is a special sampler — call sample() and use msgpack mode");
    }

    // Greedy fast path — no need to softmax / sort.
    if (params.temperature <= 1e-5f) {
        return argmax(logits, vocab_size);
    }

    // Stage 1: temperature + softmax.
    std::vector<float> probs(logits, logits + vocab_size);
    softmax_with_temperature(probs.data(), vocab_size, params.temperature);

    const std::uint64_t seed_u64 = params.seed != 0
        ? static_cast<std::uint64_t>(params.seed)
        : default_rng_seed();
    Pcg32 rng(seed_u64);

    if (params.type == T::Multinomial) {
        // Full categorical without filtering — sample directly.
        const float u = rng.next_uniform();
        float acc = 0.0f;
        for (std::int32_t v = 0; v < vocab_size; ++v) {
            acc += probs[v];
            if (u < acc) return static_cast<std::uint32_t>(v);
        }
        return argmax(probs.data(), vocab_size);
    }

    // Stage 2: build (prob, id) pairs sorted by descending prob. For top-k
    // alone we partial-sort; for top-p / min-p we need full sort because
    // the threshold may fall anywhere.
    std::vector<std::pair<float, std::uint32_t>> sorted;
    sorted.reserve(vocab_size);
    for (std::int32_t v = 0; v < vocab_size; ++v) {
        if (probs[v] > 0.0f) {
            sorted.emplace_back(probs[v], static_cast<std::uint32_t>(v));
        }
    }
    if (sorted.empty()) return 0;

    auto desc_by_prob = [](const auto& a, const auto& b) {
        return a.first > b.first;
    };

    // Stage 3: filter.
    std::size_t keep = sorted.size();

    if (params.type == T::TopK || params.type == T::TopKTopP) {
        if (params.top_k != 0 && params.top_k < keep) {
            // Partial sort — top-k keeps the K largest.
            std::partial_sort(sorted.begin(),
                              sorted.begin() + params.top_k,
                              sorted.end(), desc_by_prob);
            keep = params.top_k;
        } else {
            std::sort(sorted.begin(), sorted.end(), desc_by_prob);
        }
    } else {
        std::sort(sorted.begin(), sorted.end(), desc_by_prob);
    }

    if (params.type == T::TopP || params.type == T::TopKTopP) {
        const float threshold = std::clamp(params.top_p, 0.0f, 1.0f);
        if (threshold < 1.0f) {
            float cum = 0.0f;
            std::size_t cut = 0;
            for (std::size_t i = 0; i < keep; ++i) {
                cum += sorted[i].first;
                if (cum >= threshold) {
                    cut = i + 1;
                    break;
                }
            }
            if (cut > 0 && cut < keep) keep = cut;
        }
    }

    if (params.type == T::MinP) {
        const float thr = params.min_p * sorted[0].first;  // top is sorted[0]
        std::size_t cut = 0;
        for (; cut < keep; ++cut) {
            if (sorted[cut].first < thr) break;
        }
        if (cut > 0) keep = cut;
    }

    sorted.resize(keep);

    // Stage 4: renormalize and sample.
    double total = 0.0;
    for (const auto& [p, _] : sorted) total += p;
    if (total <= 0.0) return sorted[0].second;

    return categorical_sample(sorted, static_cast<float>(total), rng);
}

}  // namespace pie_ggml_driver
