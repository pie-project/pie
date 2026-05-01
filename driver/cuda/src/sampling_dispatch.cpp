#include "sampling_dispatch.hpp"

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels/sample_flashinfer.hpp"
#include "kernels/sample_temp.hpp"
#include "model/qwen3_forward.hpp"

namespace pie_cuda_driver {

void dispatch_sampling(
    model::Qwen3Workspace& ws,
    DeviceBuffer<std::int32_t>& d_sampled,
    const SamplingPlan& plan,
    int N,
    int num_sampling,
    int vocab_size,
    std::uint64_t prng_offset)
{
    if (plan.any_topk_topp) {
        // flashinfer's seed input is u64; widen.
        std::vector<std::uint64_t> h_per_seed64(N);
        for (int i = 0; i < N; ++i) {
            h_per_seed64[i] = static_cast<std::uint64_t>(plan.seed[i]);
        }

        auto d_temp_f          = DeviceBuffer<float>::from_host(plan.temp);
        auto d_top_p_f         = DeviceBuffer<float>::from_host(plan.top_p);
        auto d_top_k           = DeviceBuffer<std::int32_t>::from_host(plan.top_k);
        auto d_seed64          = DeviceBuffer<std::uint64_t>::from_host(
            std::span<const std::uint64_t>(h_per_seed64));
        auto d_sample_idx      = DeviceBuffer<std::int32_t>::from_host(plan.sample_idx);
        auto d_per_sample_token = DeviceBuffer<std::int32_t>::alloc(num_sampling);
        // `valid` scratch: kernel writes per-sample, caller doesn't read.
        auto d_valid           = DeviceBuffer<bool>::alloc(num_sampling);

        kernels::launch_sample_topk_topp_bf16(
            ws.logits.data(), ws.probs.data(),
            d_temp_f.data(), d_sample_idx.data(), d_top_k.data(),
            d_top_p_f.data(), d_seed64.data(),
            d_valid.data(), d_per_sample_token.data(),
            N, num_sampling, vocab_size,
            prng_offset, /*stream=*/nullptr);

        // Scatter per-sample tokens back into d_sampled at their global
        // row positions so downstream code can index by logit row.
        const auto h_per_sample_token = d_per_sample_token.to_host();
        std::vector<std::int32_t> h_all_sampled(N, 0);
        for (int k = 0; k < num_sampling; ++k) {
            h_all_sampled[plan.sample_idx[k]] = h_per_sample_token[k];
        }
        CUDA_CHECK(cudaMemcpy(d_sampled.data(), h_all_sampled.data(),
                              sizeof(std::int32_t) * N,
                              cudaMemcpyHostToDevice));
    } else {
        // Greedy / temperature / min-p path. Writes output for every
        // row, including non-sample rows; caller indexes by sample row.
        auto d_temp  = DeviceBuffer<float>::from_host(plan.temp);
        auto d_min_p = DeviceBuffer<float>::from_host(plan.min_p);
        auto d_seed  = DeviceBuffer<std::uint32_t>::from_host(plan.seed);

        kernels::launch_sample_temp_bf16(
            ws.logits.data(),
            d_temp.data(), d_min_p.data(), d_seed.data(),
            d_sampled.data(),
            N, vocab_size, /*stream=*/nullptr);
    }
}

}  // namespace pie_cuda_driver
