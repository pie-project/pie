#include "batch/persistent_inputs.hpp"

#include <algorithm>

namespace pie_cuda_driver {

PersistentInputs PersistentInputs::allocate(
    int max_workspace_tokens,
    int max_requests,
    int max_kv_pages,
    std::size_t max_custom_mask_bytes,
    int max_mtp_draft_rows)
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
    p.structured_mask_klen =
        DeviceBuffer<std::uint32_t>::alloc(max_requests);
    p.structured_masks =
        DeviceBuffer<kernels::StructuredMaskParams>::alloc(max_requests);
    p.w_page             = DeviceBuffer<std::uint32_t>::alloc(max_workspace_tokens);
    p.w_off              = DeviceBuffer<std::uint32_t>::alloc(max_workspace_tokens);
    p.slot_ids           = DeviceBuffer<std::int32_t >::alloc(max_requests);
    p.is_fresh           = DeviceBuffer<std::uint8_t >::alloc(max_requests);
    p.rs_slot_flags      = DeviceBuffer<std::uint8_t >::alloc(max_requests);
    p.rs_fold_lens       = DeviceBuffer<std::uint32_t>::alloc(max_requests);
    const std::size_t rs_buffer_refs =
        static_cast<std::size_t>(std::max(0, max_workspace_tokens)) +
        static_cast<std::size_t>(std::max(0, max_requests));
    p.rs_buffer_slot_ids =
        DeviceBuffer<std::uint32_t>::alloc(rs_buffer_refs);
    p.rs_buffer_slot_indptr =
        DeviceBuffer<std::uint32_t>::alloc(
            static_cast<std::size_t>(max_requests) + 1);
    p.rs_slot_flags_host =
        PinnedHostBuffer<std::uint8_t>::alloc(max_requests);
    p.rs_fold_lens_host =
        PinnedHostBuffer<std::uint32_t>::alloc(max_requests);
    p.rs_buffer_slot_ids_host =
        PinnedHostBuffer<std::uint32_t>::alloc(rs_buffer_refs);
    p.rs_buffer_slot_indptr_host =
        PinnedHostBuffer<std::uint32_t>::alloc(
            static_cast<std::size_t>(max_requests) + 1);
    p.mtp_request_ids    = DeviceBuffer<std::int32_t >::alloc(max_requests);
    const std::size_t mtp_rows =
        static_cast<std::size_t>(std::max(0, max_mtp_draft_rows));
    p.mtp_positions_host =
        PinnedHostBuffer<std::int32_t>::alloc(mtp_rows);
    p.mtp_hidden_rows_host =
        PinnedHostBuffer<std::int32_t>::alloc(mtp_rows);
    p.mtp_request_ids_host =
        PinnedHostBuffer<std::int32_t>::alloc(mtp_rows);
    p.sample_idx         = DeviceBuffer<std::int32_t>::alloc(max_workspace_tokens);
    return p;
}

std::size_t persistent_input_bytes(int N,
                                   int R,
                                   int max_page_refs,
                                   int max_custom_mask_bytes) {
    std::size_t bytes = 0;
    bytes += static_cast<std::size_t>(N) * (4 + 4 + 4 + 4);
    bytes += static_cast<std::size_t>(R + 1) * (4 + 4);
    bytes += static_cast<std::size_t>(R) * (4 + 4 + 1 + 4);
    bytes += static_cast<std::size_t>(R) * (1 + 4);
    bytes += static_cast<std::size_t>(std::max(0, N) + std::max(0, R)) * 4;
    bytes += static_cast<std::size_t>(R + 1) * 4;
    bytes += static_cast<std::size_t>(max_page_refs) * 4;
    bytes += static_cast<std::size_t>(max_custom_mask_bytes);
    bytes += static_cast<std::size_t>(R + 1) * 4;
    bytes += static_cast<std::size_t>(R) *
        (sizeof(std::uint32_t) + sizeof(kernels::StructuredMaskParams));
    bytes += static_cast<std::size_t>(N) * (4 + 4);  // w_page + w_off
    return bytes;
}

}  // namespace pie_cuda_driver
