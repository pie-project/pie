#include "masked_distribution.hpp"

#include <algorithm>
#include <cmath>
#include <limits>
#include <stdexcept>
#include <utility>

namespace pie_cuda_driver {

MaskedDistribution masked_top_k_distribution(
    std::span<const float> logits,
    std::span<const std::uint32_t> mask_runs,
    float temperature,
    std::size_t top_k)
{
    if (logits.empty() || top_k == 0) {
        throw std::runtime_error(
            "masked distribution requires logits and top_k");
    }

    std::vector<std::uint32_t> allowed_ids;
    allowed_ids.reserve(logits.size());

    auto include = [&](std::size_t index) {
        if (!std::isfinite(logits[index])) {
            throw std::runtime_error(
                "masked distribution contains an invalid logit");
        }
        allowed_ids.push_back(static_cast<std::uint32_t>(index));
    };

    if (mask_runs.empty()) {
        for (std::size_t index = 0; index < logits.size(); ++index) {
            include(index);
        }
    } else {
        bool is_allowed = false;
        std::size_t position = 0;
        for (const std::uint32_t run_length : mask_runs) {
            const std::size_t remaining = logits.size() - position;
            const std::size_t effective = std::min<std::size_t>(run_length, remaining);
            if (is_allowed) {
                for (std::size_t offset = 0; offset < effective; ++offset) {
                    include(position + offset);
                }
            }
            position += effective;
            is_allowed = !is_allowed;
            if (position == logits.size()) break;
        }
    }

    if (allowed_ids.empty()) {
        throw std::runtime_error(
            "masked distribution has no allowed tokens");
    }

    constexpr double GREEDY_TEMPERATURE = 1e-5;
    const double effective_temperature =
        temperature > GREEDY_TEMPERATURE
            ? static_cast<double>(temperature)
            : GREEDY_TEMPERATURE;
    const double inverse_temperature = 1.0 / effective_temperature;

    double maximum = -std::numeric_limits<double>::infinity();
    for (const std::uint32_t token_id : allowed_ids) {
        maximum = std::max(
            maximum,
            static_cast<double>(logits[token_id]) * inverse_temperature);
    }

    std::vector<std::pair<float, std::uint32_t>> allowed;
    allowed.reserve(allowed_ids.size());
    double allowed_mass = 0.0;
    for (const std::uint32_t token_id : allowed_ids) {
        const double probability = std::exp(
            static_cast<double>(logits[token_id]) * inverse_temperature - maximum);
        allowed.emplace_back(static_cast<float>(probability), token_id);
        allowed_mass += probability;
    }
    if (!std::isfinite(allowed_mass) || allowed_mass <= 0.0) {
        throw std::runtime_error(
            "masked distribution has no allowed probability mass");
    }

    const std::size_t count = std::min(top_k, allowed.size());
    std::partial_sort(
        allowed.begin(), allowed.begin() + count, allowed.end(),
        [](const auto& lhs, const auto& rhs) {
            if (lhs.first != rhs.first) return lhs.first > rhs.first;
            return lhs.second < rhs.second;
        });

    MaskedDistribution result;
    result.token_ids.reserve(count);
    result.probabilities.reserve(count);
    for (std::size_t index = 0; index < count; ++index) {
        result.token_ids.push_back(allowed[index].second);
        result.probabilities.push_back(
            static_cast<float>(allowed[index].first / allowed_mass));
    }
    return result;
}

}  // namespace pie_cuda_driver
