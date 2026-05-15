#include "executor/persistent_inputs.hpp"

namespace pie_cuda_driver {

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
    // Sampler scratch — sized to the worst case (`max_workspace_tokens`
    // rows). Capacity-wise these are tiny (~10s of KiB total) so we
    // don't try to right-size per arch; the win is eliminating per-fire
    // `DeviceBuffer::from_host` allocations inside `dispatch_sampling`.
    p.sample_temp        = DeviceBuffer<float>::alloc(max_workspace_tokens);
    p.sample_top_p       = DeviceBuffer<float>::alloc(max_workspace_tokens);
    p.sample_min_p       = DeviceBuffer<float>::alloc(max_workspace_tokens);
    p.sample_top_k       = DeviceBuffer<std::int32_t>::alloc(max_workspace_tokens);
    p.sample_seed        = DeviceBuffer<std::uint32_t>::alloc(max_workspace_tokens);
    p.sample_seed64      = DeviceBuffer<std::uint64_t>::alloc(max_workspace_tokens);
    p.sample_idx         = DeviceBuffer<std::int32_t>::alloc(max_workspace_tokens);
    p.sample_per_token   = DeviceBuffer<std::int32_t>::alloc(max_workspace_tokens);
    p.sample_valid       = DeviceBuffer<bool>::alloc(max_workspace_tokens);
    return p;
}

}  // namespace pie_cuda_driver
