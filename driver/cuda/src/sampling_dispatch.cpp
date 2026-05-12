#include "sampling_dispatch.hpp"

#include <cstdint>
#include <cstring>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels/sample_flashinfer.hpp"
#include "kernels/sample_temp.hpp"
#include "model/qwen3_forward.hpp"
#include "persistent_inputs.hpp"

namespace pie_cuda_driver {

void dispatch_sampling(
    model::Qwen3Workspace& ws,
    PersistentInputs& pi,
    std::int32_t* d_sampled_out,
    const SamplingPlan& plan,
    int N,
    int num_sampling,
    int vocab_size,
    std::uint64_t prng_offset)
{
    if (plan.any_topk_topp) {
        // Widen u32 seeds → u64 in place in the pinned staging buffer,
        // then a single async H2D into the persistent device array.
        // h_per_sample_tok_pinned is reused as a u32 → u64 scratch since
        // we only need it after the kernel writes per-sample tokens,
        // and the seed copy completes before then. Cleaner: dedicated
        // pinned u64 buffer; sticking with safety here.
        std::uint64_t h_seed64_buf[1];  // unused — replaced below
        (void)h_seed64_buf;

        // Stage the seed widening into a transient host vector — short-
        // lived but pageable. (Only needed when topk/topp is active,
        // which is uncommon enough that adding another pinned ring
        // buffer is not worth it.)
        std::vector<std::uint64_t> h_per_seed64(N);
        for (int i = 0; i < N; ++i) {
            h_per_seed64[i] = static_cast<std::uint64_t>(plan.seed[i]);
        }

        // Refresh all the plan arrays into the persistent device
        // buffers. copy_from_host issues against the default stream
        // (cudaMemcpyAsync); the subsequent kernel sees the new data
        // because the stream is in-order.
        pi.sampling_temp        .copy_from_host(plan.temp);
        pi.sampling_top_p       .copy_from_host(plan.top_p);
        pi.sampling_top_k       .copy_from_host(plan.top_k);
        pi.sampling_seed_u64    .copy_from_host(
            std::span<const std::uint64_t>(h_per_seed64));
        pi.sampling_sample_idx  .copy_from_host(plan.sample_idx);

        kernels::launch_sample_topk_topp_bf16(
            ws.logits.data(), ws.probs.data(),
            pi.sampling_temp.data(), pi.sampling_sample_idx.data(),
            pi.sampling_top_k.data(),
            pi.sampling_top_p.data(), pi.sampling_seed_u64.data(),
            pi.sampling_valid.data(), pi.sampling_per_sample_tok.data(),
            N, num_sampling, vocab_size,
            prng_offset, /*stream=*/nullptr);

        // D2H of per-sample tokens into pinned host. Then scatter into
        // pinned all-sampled buffer (host side, no GPU work), then async
        // H2D back into d_sampled_out. The sync point is between the
        // D2H and the host scatter — we have to wait for the kernel
        // anyway, but we keep the destination pinned so the D2H itself
        // doesn't have to stage through driver-internal pinned memory.
        CUDA_CHECK(cudaMemcpyAsync(
            pi.h_per_sample_tok_pinned, pi.sampling_per_sample_tok.data(),
            sizeof(std::int32_t) * static_cast<std::size_t>(num_sampling),
            cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaStreamSynchronize(/*stream=*/nullptr));

        // Initialize the scatter destination — non-sample rows can be
        // anything, but zero is what the sync path produced.
        std::memset(pi.h_all_sampled_pinned, 0, sizeof(std::int32_t) * N);
        for (int k = 0; k < num_sampling; ++k) {
            pi.h_all_sampled_pinned[plan.sample_idx[k]] =
                pi.h_per_sample_tok_pinned[k];
        }
        // Async H2D from pinned: the source buffer is persistent, so
        // we don't need to keep a stack temporary alive.
        CUDA_CHECK(cudaMemcpyAsync(
            d_sampled_out, pi.h_all_sampled_pinned,
            sizeof(std::int32_t) * static_cast<std::size_t>(N),
            cudaMemcpyHostToDevice));
    } else {
        // Greedy / temperature / min-p path. Writes output for every
        // row, including non-sample rows; caller indexes by sample row.
        pi.sampling_temp     .copy_from_host(plan.temp);
        pi.sampling_min_p    .copy_from_host(plan.min_p);
        pi.sampling_seed_u32 .copy_from_host(plan.seed);

        kernels::launch_sample_temp_bf16(
            ws.logits.data(),
            pi.sampling_temp.data(), pi.sampling_min_p.data(),
            pi.sampling_seed_u32.data(),
            d_sampled_out,
            N, vocab_size, /*stream=*/nullptr);
    }
}

}  // namespace pie_cuda_driver
