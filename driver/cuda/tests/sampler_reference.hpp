#pragma once

// CPU golden reference for the Sampling-IR parity tests (lane L7 / hotel).
//
// This header is the single source of truth for sampler *math* on the host
// side. It transcribes — byte-for-byte where it matters — the production
// Gumbel-max temperature/min-p sampler in
// `driver/cuda/src/kernels/sample_temp.cu`, plus the standard top-k / top-p
// masking semantics the IR path must reproduce.
//
// Parity strategy:
//   * argmax (T == 0)            → bit-exact token match (no transcendentals).
//   * temperature / min-p (T>0)  → Gumbel-max; the host and device `logf`/`expf`
//                                  can differ by ~1-2 ulp, so the harness treats
//                                  a row as matching when the device token equals
//                                  the reference argmax OR the two candidates'
//                                  scores are within a near-tie tolerance. The
//                                  reference exposes the score margin to make
//                                  that classification possible.
//   * top-k / top-p              → reference-defines-truth (FlashInfer uses a
//                                  different PRNG); validated here by brute force,
//                                  and the IR path is compared against it in W3.
//
// Pure C++ (no CUDA headers) so it is reusable from host-only unit tests. The
// caller is responsible for feeding logits that already passed through the same
// bf16 rounding the device kernel sees (round float→bf16→float once, share both).

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#include <numeric>
#include <vector>

namespace pie_sampler_ref {

// ── RNG: SplitMix64 + (seed × column), matching sample_temp.cu exactly. ──────
//
// hash_uniform(seed_eff, j):
//   x  = seed_eff + 0x9E3779B97F4A7C15 * (j + 1)
//   x ^= x>>27; x *= 0x3C79AC492BA7B653
//   x ^= x>>33; x *= 0x1C69B3F74AC4AE35
//   x ^= x>>27
//   bits = (u32)(x >> 40)                  // high 24 bits
//   u    = (bits + 0.5) * (1/16777216.0)   // ∈ (0, 1)
//
// The caller passes the *raw* per-row u32 seed; we apply the `^ 0xa5a5a5a5`
// salt here, identically to the kernel.

inline std::uint64_t seed_eff_from_row(std::uint32_t row_seed) {
    return static_cast<std::uint64_t>(row_seed) ^ 0xa5a5a5a5ull;
}

inline float hash_uniform(std::uint64_t seed_eff, int j) {
    std::uint64_t x =
        seed_eff + 0x9E3779B97F4A7C15ull * static_cast<std::uint64_t>(j + 1);
    x ^= x >> 27; x *= 0x3C79AC492BA7B653ull;
    x ^= x >> 33; x *= 0x1C69B3F74AC4AE35ull;
    x ^= x >> 27;
    const std::uint32_t bits = static_cast<std::uint32_t>(x >> 40);
    return (static_cast<float>(bits) + 0.5f) * (1.0f / 16777216.0f);
}

inline float gumbel_noise(std::uint64_t seed_eff, int j) {
    const float u = hash_uniform(seed_eff, j);
    return -logf(-logf(u));
}

// ── Result of a token-producing sample, with the info the harness needs to
//    classify near-ties (transcendental ulp noise) vs genuine divergence. ────
struct SampleResult {
    int   token       = -1;
    float best_score  = -std::numeric_limits<float>::infinity();
    float second_score = -std::numeric_limits<float>::infinity();

    // Margin between the winning and runner-up score. A device/host token
    // disagreement is attributable to transcendental ulp noise only when this
    // margin is below the harness tolerance.
    float margin() const { return best_score - second_score; }
};

// Plain argmax over logits with lowest-index tie-break (== sample_temp greedy
// path, T <= 0). No transcendentals → device must match this bit-for-bit.
inline SampleResult argmax(const std::vector<float>& logits) {
    SampleResult r;
    for (int j = 0; j < static_cast<int>(logits.size()); ++j) {
        const float v = logits[j];
        if (v > r.best_score) {
            r.second_score = r.best_score;
            r.best_score = v;
            r.token = j;
        } else if (v > r.second_score) {
            r.second_score = v;
        }
    }
    return r;
}

// Max logit over a row (deterministic; float max is order-independent).
inline float row_max(const std::vector<float>& logits) {
    float m = -std::numeric_limits<float>::infinity();
    for (float v : logits) m = fmaxf(m, v);
    return m;
}

// Gumbel-max temperature + min-p sampler — faithful transcription of
// `sample_temp_kernel`. `temperature <= 0` collapses to plain argmax;
// `min_p > 0 && temperature > 0` masks tokens with logit < max+log(min_p).
inline SampleResult sample_temp(const std::vector<float>& logits,
                                float temperature,
                                float min_p,
                                std::uint32_t row_seed) {
    const int vocab = static_cast<int>(logits.size());
    const bool greedy = !(temperature > 0.f);
    const float inv_T = greedy ? 1.f : (1.f / temperature);
    const bool apply_min_p = (min_p > 0.f) && !greedy;
    const std::uint64_t seed = seed_eff_from_row(row_seed);

    float min_threshold = -std::numeric_limits<float>::infinity();
    if (apply_min_p) {
        min_threshold = row_max(logits) + logf(min_p);
    }

    SampleResult r;
    for (int j = 0; j < vocab; ++j) {
        const float logit = logits[j];
        if (apply_min_p && logit < min_threshold) continue;
        const float score =
            greedy ? logit : (logit * inv_T + gumbel_noise(seed, j));
        if (score > r.best_score) {
            r.second_score = r.best_score;
            r.best_score = score;
            r.token = j;
        } else if (score > r.second_score) {
            r.second_score = score;
        }
    }
    return r;
}

// ── Masking semantics (top-k / top-p / min-p) the IR path must reproduce. ────
//
// Returned masks are over the original vocab order: `true` == kept. These are
// reference-defines-truth (FlashInfer's PRNG differs), validated by the
// brute-force checks in the unit tests and compared against the IR in W3.

// Softmax with temperature, in fp32 (matches sample_flashinfer softmax math:
// subtract row max, exp, normalize). temperature <= 0 → plain softmax (T=1).
inline std::vector<float> softmax(const std::vector<float>& logits,
                                  float temperature = 1.f) {
    const float inv_T = (temperature > 0.f) ? (1.f / temperature) : 1.f;
    const int n = static_cast<int>(logits.size());
    float m = -std::numeric_limits<float>::infinity();
    for (float v : logits) m = fmaxf(m, v * inv_T);
    std::vector<float> p(n);
    float sum = 0.f;
    for (int j = 0; j < n; ++j) {
        p[j] = expf(logits[j] * inv_T - m);
        sum += p[j];
    }
    const float inv = 1.f / sum;
    for (float& v : p) v *= inv;
    return p;
}

// Keep the k highest-logit tokens (lowest index wins ties). k <= 0 → keep all.
inline std::vector<bool> top_k_mask(const std::vector<float>& logits, int k) {
    const int n = static_cast<int>(logits.size());
    std::vector<bool> keep(n, true);
    if (k <= 0 || k >= n) return keep;
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return logits[a] > logits[b]; });
    std::fill(keep.begin(), keep.end(), false);
    for (int i = 0; i < k; ++i) keep[order[i]] = true;
    return keep;
}

// Nucleus (top-p): keep the smallest high-prob prefix whose cumulative mass
// reaches p. Sorted descending by probability (lowest index wins ties); the
// token that crosses the threshold is included. p >= 1 → keep all.
inline std::vector<bool> top_p_mask(const std::vector<float>& logits,
                                    float p,
                                    float temperature = 1.f) {
    const int n = static_cast<int>(logits.size());
    std::vector<bool> keep(n, false);
    if (p >= 1.f) { std::fill(keep.begin(), keep.end(), true); return keep; }
    const std::vector<float> prob = softmax(logits, temperature);
    std::vector<int> order(n);
    std::iota(order.begin(), order.end(), 0);
    std::stable_sort(order.begin(), order.end(),
                     [&](int a, int b) { return prob[a] > prob[b]; });
    float cum = 0.f;
    for (int i = 0; i < n; ++i) {
        keep[order[i]] = true;
        cum += prob[order[i]];
        if (cum >= p) break;
    }
    return keep;
}

// min-p: keep tokens whose prob >= min_p * max_prob, i.e. (in logit space)
// logit >= max_logit + log(min_p). Strict-less-than is dropped, matching the
// kernel. min_p <= 0 → keep all.
inline std::vector<bool> min_p_mask(const std::vector<float>& logits,
                                    float min_p) {
    const int n = static_cast<int>(logits.size());
    std::vector<bool> keep(n, true);
    if (min_p <= 0.f) return keep;
    const float thr = row_max(logits) + logf(min_p);
    for (int j = 0; j < n; ++j) keep[j] = !(logits[j] < thr);
    return keep;
}

// Gumbel-max over a pre-masked logit row (masked-out tokens excluded). This is
// the selection step the IR top-k/top-p/min-p programs use after building their
// mask. temperature scales logits; lowest-index tie-break.
inline SampleResult gumbel_argmax_masked(const std::vector<float>& logits,
                                         const std::vector<bool>& keep,
                                         float temperature,
                                         std::uint32_t row_seed) {
    const int vocab = static_cast<int>(logits.size());
    const bool greedy = !(temperature > 0.f);
    const float inv_T = greedy ? 1.f : (1.f / temperature);
    const std::uint64_t seed = seed_eff_from_row(row_seed);
    SampleResult r;
    for (int j = 0; j < vocab; ++j) {
        if (!keep[j]) continue;
        const float score =
            greedy ? logits[j] : (logits[j] * inv_T + gumbel_noise(seed, j));
        if (score > r.best_score) {
            r.second_score = r.best_score;
            r.best_score = score;
            r.token = j;
        } else if (score > r.second_score) {
            r.second_score = score;
        }
    }
    return r;
}

// ── Probe-style reductions (entropy etc.) used by the rich-output programs. ──

// Shannon entropy of softmax(logits/T), in nats (matches entropy.cu intent).
inline float entropy(const std::vector<float>& logits, float temperature = 1.f) {
    const std::vector<float> p = softmax(logits, temperature);
    float h = 0.f;
    for (float v : p) {
        if (v > 0.f) h -= v * logf(v);
    }
    return h;
}

}  // namespace pie_sampler_ref
