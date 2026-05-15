#include "sampling_dispatch.hpp"

#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "kernels/sample_flashinfer.hpp"
#include "kernels/sample_temp.hpp"
#include "kernels/scatter_int32.hpp"
#include "model/qwen3_forward.hpp"
#include "persistent_inputs.hpp"

namespace pie_cuda_driver {

namespace {

// Async copy a host span to a pre-allocated device buffer on the given
// stream. Length must fit; partial copies are not allowed.
template <typename T>
void upload_async(DeviceBuffer<T>& dst, std::span<const T> src,
                  cudaStream_t stream) {
    if (src.empty()) return;
    if (src.size() > dst.size()) {
        throw std::runtime_error(
            "sampling upload: src size > device buffer capacity");
    }
    CUDA_CHECK(cudaMemcpyAsync(dst.data(), src.data(),
                               src.size() * sizeof(T),
                               cudaMemcpyHostToDevice, stream));
}

// Pinned host scratch for the topk+top-p seed widening. Allocated once
// per process so we can keep upload_sampling_inputs allocation-free.
std::uint64_t* seed64_pinned_buf(std::size_t want_elems) {
    static std::uint64_t* buf = nullptr;
    static std::size_t buf_capacity = 0;
    if (want_elems > buf_capacity) {
        if (buf) cudaFreeHost(buf);
        CUDA_CHECK(cudaMallocHost(&buf, want_elems * sizeof(std::uint64_t)));
        buf_capacity = want_elems;
    }
    return buf;
}

}  // namespace

void upload_sampling_inputs(
    PersistentInputs& pi,
    const SamplingPlan& plan,
    int N,
    cudaStream_t stream)
{
    if (plan.any_topk_topp) {
        // flashinfer's seed is u64; widen via pinned host scratch so the
        // cudaMemcpyAsync stays truly async (and graph-capturable, if a
        // future code path moves it back inside capture).
        auto* h_seed64 = seed64_pinned_buf(static_cast<std::size_t>(N));
        for (int i = 0; i < N; ++i) {
            h_seed64[i] = static_cast<std::uint64_t>(plan.seed[i]);
        }
        upload_async<float>        (pi.sample_temp,    plan.temp,        stream);
        upload_async<float>        (pi.sample_top_p,   plan.top_p,       stream);
        upload_async<std::int32_t> (pi.sample_top_k,   plan.top_k,       stream);
        upload_async<std::uint64_t>(pi.sample_seed64,
            std::span<const std::uint64_t>(h_seed64, static_cast<std::size_t>(N)), stream);
        upload_async<std::int32_t> (pi.sample_idx,     plan.sample_idx,  stream);
    } else {
        upload_async<float>        (pi.sample_temp,  plan.temp,  stream);
        upload_async<float>        (pi.sample_min_p, plan.min_p, stream);
        upload_async<std::uint32_t>(pi.sample_seed,  plan.seed,  stream);
    }
}

void launch_sampling_kernel(
    model::Qwen3Workspace& ws,
    std::int32_t* d_sampled_out,
    PersistentInputs& pi,
    const SamplingPlan& plan,
    int N,
    int num_sampling,
    int vocab_size,
    std::uint64_t prng_offset,
    cudaStream_t stream)
{
    if (plan.any_topk_topp) {
        kernels::launch_sample_topk_topp_bf16(
            ws.logits.data(), ws.probs.data(),
            pi.sample_temp.data(), pi.sample_idx.data(), pi.sample_top_k.data(),
            pi.sample_top_p.data(), pi.sample_seed64.data(),
            pi.sample_valid.data(), pi.sample_per_token.data(),
            N, num_sampling, vocab_size,
            prng_offset, stream);
        // On-device scatter — keeps the whole step graph-safe.
        kernels::launch_scatter_int32(
            d_sampled_out,
            pi.sample_idx.data(),
            pi.sample_per_token.data(),
            num_sampling, N, stream);
    } else {
        kernels::launch_sample_temp_bf16(
            ws.logits.data(),
            pi.sample_temp.data(), pi.sample_min_p.data(), pi.sample_seed.data(),
            d_sampled_out,
            N, vocab_size, stream);
    }
}

void dispatch_sampling(
    model::Qwen3Workspace& ws,
    std::int32_t* d_sampled_out,
    PersistentInputs& pi,
    const SamplingPlan& plan,
    int N,
    int num_sampling,
    int vocab_size,
    std::uint64_t prng_offset,
    cudaStream_t stream)
{
    upload_sampling_inputs(pi, plan, N, stream);
    launch_sampling_kernel(ws, d_sampled_out, pi, plan,
                           N, num_sampling, vocab_size,
                           prng_offset, stream);
}

}  // namespace pie_cuda_driver
