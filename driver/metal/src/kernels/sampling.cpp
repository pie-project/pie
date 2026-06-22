#include "kernels/sampling.hpp"

#include <limits>

#include <mlx/mlx.h>

namespace pie_metal_driver::sampling {

namespace mx = mlx::core;

namespace {

bool is_greedy(const SamplerParams& p) {
    return p.temperature <= kGreedyTempEps;
}

// Sample one token id from a single logits row [vocab] (float32) under the
// given params. Returns a scalar uint32 MLX array (lazy; evaled by caller).
Tensor sample_row(const Tensor& row, const SamplerParams& p,
                  std::uint64_t seed) {
    if (is_greedy(p)) {
        return mx::astype(mx::argmax(row, /*axis=*/0), mx::uint32);
    }

    Tensor key = mx::random::key(seed);
    Tensor l = mx::divide(row, mx::array(p.temperature));

    const bool has_top_k = p.top_k > 0;
    const bool has_top_p = p.top_p < 1.0f;
    const bool has_min_p = p.min_p > 0.0f;

    if (!has_top_k && !has_top_p && !has_min_p) {
        // Plain multinomial over the full (temperature-scaled) distribution.
        return mx::astype(mx::random::categorical(l, /*axis=*/0, key), mx::uint32);
    }

    const int vocab = row.shape(0);
    const float neg_inf = -std::numeric_limits<float>::infinity();

    // Descending sort by logit: argsort(-l) gives indices high→low.
    Tensor order = mx::argsort(mx::negative(l), /*axis=*/0);  // [vocab] int32
    Tensor ls = mx::take(l, order, /*axis=*/0);               // sorted desc
    Tensor ps = mx::softmax(ls, /*axis=*/0);                  // sorted probs

    Tensor keep = mx::full({vocab}, true, mx::bool_);

    if (has_top_k) {
        Tensor pos = mx::arange(0, vocab, mx::int32);
        keep = mx::logical_and(
            keep, mx::less(pos, mx::array(static_cast<int>(p.top_k))));
    }
    if (has_top_p) {
        // Keep the smallest prefix whose cumulative prob first crosses top_p:
        // token i is kept iff (cum[i] - ps[i]) < top_p.
        Tensor cum = mx::cumsum(ps, /*axis=*/0, /*reverse=*/false,
                                /*inclusive=*/true);
        keep = mx::logical_and(
            keep, mx::less(mx::subtract(cum, ps), mx::array(p.top_p)));
    }
    if (has_min_p) {
        Tensor max_p = mx::take(ps, mx::array(0), /*axis=*/0);  // sorted[0] = max
        Tensor thresh = mx::multiply(max_p, mx::array(p.min_p));
        keep = mx::logical_and(keep, mx::greater_equal(ps, thresh));
    }

    Tensor masked = mx::where(keep, ls, mx::array(neg_inf));
    Tensor sorted_sample = mx::random::categorical(masked, /*axis=*/0, key);
    Tensor token = mx::take(order, sorted_sample, /*axis=*/0);
    return mx::astype(token, mx::uint32);
}

}  // namespace

std::vector<std::uint32_t> sample_tokens(
    const Tensor& logits,
    const std::vector<SamplerParams>& params,
    std::uint64_t base_seed) {
    const int n_slots = static_cast<int>(params.size());
    std::vector<std::uint32_t> out(n_slots, 0);
    if (n_slots == 0) return out;

    Tensor lf = mx::astype(logits, mx::float32);

    // Fast path: every slot greedy → one batched argmax over the vocab axis.
    bool all_greedy = true;
    for (const auto& p : params) {
        if (!is_greedy(p)) { all_greedy = false; break; }
    }
    if (all_greedy) {
        Tensor toks = mx::astype(mx::argmax(lf, /*axis=*/1), mx::uint32);
        toks.eval();
        const std::uint32_t* p = toks.data<std::uint32_t>();
        for (int s = 0; s < n_slots; ++s) out[s] = p[s];
        return out;
    }

    // General path: per-slot sampling graphs, evaluated together.
    std::vector<Tensor> toks;
    toks.reserve(n_slots);
    for (int s = 0; s < n_slots; ++s) {
        Tensor row = mx::slice(lf, {s, 0}, {s + 1, lf.shape(1)});
        row = mx::reshape(row, {lf.shape(1)});
        std::uint64_t seed = params[s].seed != 0
            ? static_cast<std::uint64_t>(params[s].seed)
            : (base_seed + static_cast<std::uint64_t>(s));
        toks.push_back(sample_row(row, params[s], seed));
    }
    mx::eval(toks);
    for (int s = 0; s < n_slots; ++s) {
        out[s] = toks[s].item<std::uint32_t>();
    }
    return out;
}

}  // namespace pie_metal_driver::sampling
