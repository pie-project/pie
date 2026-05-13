#include "persistent_inputs.hpp"

#include <cstddef>
#include <cuda_runtime.h>

#include "cuda_check.hpp"

namespace pie_cuda_driver {

namespace {

template <class T>
T* alloc_pinned(std::size_t count) {
    if (count == 0) return nullptr;
    void* p = nullptr;
    CUDA_CHECK(cudaMallocHost(&p, count * sizeof(T)));
    return static_cast<T*>(p);
}

void free_pinned(void* p) noexcept {
    if (p) cudaFreeHost(p);
}

}  // namespace

PersistentInputs PersistentInputs::allocate(
    int max_workspace_tokens,
    int max_requests,
    int max_kv_pages,
    std::size_t max_custom_mask_bytes)
{
    PersistentInputs p;
    p.tokens             = DeviceBuffer<std::uint32_t>::alloc(max_workspace_tokens);
    p.positions          = DeviceBuffer<std::uint32_t>::alloc(max_workspace_tokens);
    p.sampled            = DeviceBuffer<std::int32_t >::alloc(max_workspace_tokens);
    p.qo_indptr          = DeviceBuffer<std::uint32_t>::alloc(static_cast<std::size_t>(max_requests) + 1);
    p.kv_page_indptr     = DeviceBuffer<std::uint32_t>::alloc(static_cast<std::size_t>(max_requests) + 1);
    p.kv_last_page_lens  = DeviceBuffer<std::uint32_t>::alloc(max_requests);
    p.kv_page_indices    = DeviceBuffer<std::uint32_t>::alloc(max_kv_pages);
    p.custom_mask        = DeviceBuffer<std::uint8_t >::alloc(max_custom_mask_bytes);
    p.custom_mask_indptr = DeviceBuffer<std::int32_t >::alloc(static_cast<std::size_t>(max_requests) + 1);
    p.slot_ids           = DeviceBuffer<std::int32_t >::alloc(max_requests);
    p.is_fresh           = DeviceBuffer<std::uint8_t >::alloc(max_requests);

    // Sampling-dispatch scratch — all capped at max_workspace_tokens.
    const std::size_t S = static_cast<std::size_t>(max_workspace_tokens);
    p.sampling_temp           = DeviceBuffer<float       >::alloc(S);
    p.sampling_min_p          = DeviceBuffer<float       >::alloc(S);
    p.sampling_top_p          = DeviceBuffer<float       >::alloc(S);
    p.sampling_top_k          = DeviceBuffer<std::int32_t>::alloc(S);
    p.sampling_seed_u32       = DeviceBuffer<std::uint32_t>::alloc(S);
    p.sampling_seed_u64       = DeviceBuffer<std::uint64_t>::alloc(S);
    p.sampling_sample_idx     = DeviceBuffer<std::int32_t>::alloc(S);
    p.sampling_per_sample_tok = DeviceBuffer<std::int32_t>::alloc(S);
    p.sampling_valid          = DeviceBuffer<bool        >::alloc(S);

    // Msgpack subpass scratch — same cap.
    p.subpass_rows         = DeviceBuffer<std::int32_t>::alloc(S);
    p.subpass_lp_rows      = DeviceBuffer<std::int32_t>::alloc(S);
    p.subpass_lp_lindptr   = DeviceBuffer<std::int32_t>::alloc(S + 1);
    p.subpass_lp_lids      = DeviceBuffer<std::int32_t>::alloc(S);
    p.subpass_lp_out       = DeviceBuffer<float       >::alloc(S);
    p.subpass_ent_rows     = DeviceBuffer<std::int32_t>::alloc(S);
    p.subpass_ent_out      = DeviceBuffer<float       >::alloc(S);
    p.subpass_dist_rows    = DeviceBuffer<std::int32_t>::alloc(S);
    p.subpass_dist_temps   = DeviceBuffer<float       >::alloc(S);

    // Pinned host staging.
    p.h_per_sample_tok_pinned = alloc_pinned<std::int32_t>(S);
    p.h_all_sampled_pinned    = alloc_pinned<std::int32_t>(S);
    p.h_sampled_pinned        = alloc_pinned<std::int32_t>(S);
    p.h_ent_pinned            = alloc_pinned<float       >(S);
    p.h_lp_pinned             = alloc_pinned<float       >(S);

    return p;
}

PersistentInputs::PersistentInputs(PersistentInputs&& o) noexcept
    : tokens(std::move(o.tokens)),
      positions(std::move(o.positions)),
      sampled(std::move(o.sampled)),
      qo_indptr(std::move(o.qo_indptr)),
      kv_page_indptr(std::move(o.kv_page_indptr)),
      kv_last_page_lens(std::move(o.kv_last_page_lens)),
      kv_page_indices(std::move(o.kv_page_indices)),
      custom_mask(std::move(o.custom_mask)),
      custom_mask_indptr(std::move(o.custom_mask_indptr)),
      slot_ids(std::move(o.slot_ids)),
      is_fresh(std::move(o.is_fresh)),
      sampling_temp(std::move(o.sampling_temp)),
      sampling_min_p(std::move(o.sampling_min_p)),
      sampling_top_p(std::move(o.sampling_top_p)),
      sampling_top_k(std::move(o.sampling_top_k)),
      sampling_seed_u32(std::move(o.sampling_seed_u32)),
      sampling_seed_u64(std::move(o.sampling_seed_u64)),
      sampling_sample_idx(std::move(o.sampling_sample_idx)),
      sampling_per_sample_tok(std::move(o.sampling_per_sample_tok)),
      sampling_valid(std::move(o.sampling_valid)),
      subpass_rows(std::move(o.subpass_rows)),
      subpass_lp_rows(std::move(o.subpass_lp_rows)),
      subpass_lp_lindptr(std::move(o.subpass_lp_lindptr)),
      subpass_lp_lids(std::move(o.subpass_lp_lids)),
      subpass_lp_out(std::move(o.subpass_lp_out)),
      subpass_ent_rows(std::move(o.subpass_ent_rows)),
      subpass_ent_out(std::move(o.subpass_ent_out)),
      subpass_dist_rows(std::move(o.subpass_dist_rows)),
      subpass_dist_temps(std::move(o.subpass_dist_temps)),
      h_per_sample_tok_pinned(o.h_per_sample_tok_pinned),
      h_all_sampled_pinned(o.h_all_sampled_pinned),
      h_sampled_pinned(o.h_sampled_pinned),
      h_ent_pinned(o.h_ent_pinned),
      h_lp_pinned(o.h_lp_pinned)
{
    o.h_per_sample_tok_pinned = nullptr;
    o.h_all_sampled_pinned    = nullptr;
    o.h_sampled_pinned        = nullptr;
    o.h_ent_pinned            = nullptr;
    o.h_lp_pinned             = nullptr;
}

PersistentInputs& PersistentInputs::operator=(PersistentInputs&& o) noexcept {
    if (this != &o) {
        this->~PersistentInputs();
        new (this) PersistentInputs(std::move(o));
    }
    return *this;
}

PersistentInputs::~PersistentInputs() {
    free_pinned(h_per_sample_tok_pinned);
    free_pinned(h_all_sampled_pinned);
    free_pinned(h_sampled_pinned);
    free_pinned(h_ent_pinned);
    free_pinned(h_lp_pinned);
}

}  // namespace pie_cuda_driver
