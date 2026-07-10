#pragma once

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>

#include <pie_driver_abi.h>

namespace pie_native::abi {

inline int validate_version(std::uint32_t version) noexcept {
    return version == PIE_DRIVER_ABI_VERSION
        ? PIE_STATUS_OK
        : PIE_STATUS_BAD_ABI_VERSION;
}

template <typename T>
inline int validate_slice(const T* ptr, std::size_t len) noexcept {
    if (len == 0) return PIE_STATUS_OK;
    if (ptr == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    if (len > std::numeric_limits<std::size_t>::max() / sizeof(T)) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }

    const auto address = reinterpret_cast<std::uintptr_t>(ptr);
    const std::size_t bytes = len * sizeof(T);
    if (address % alignof(T) != 0 ||
        bytes > std::numeric_limits<std::uintptr_t>::max() - address) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    return PIE_STATUS_OK;
}

inline int validate_bytes(PieBytes bytes) noexcept {
    return validate_slice(bytes.ptr, bytes.len);
}

inline int validate_terminal_cell_ptr(PieTerminalCell* cell) noexcept {
    if (cell == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    if ((reinterpret_cast<std::uintptr_t>(cell) % alignof(PieTerminalCell)) != 0) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    if (cell->reserved0 != 0 ||
        std::atomic_ref<std::uint32_t>(cell->outcome).load(
            std::memory_order_acquire) != PIE_TERMINAL_OUTCOME_PENDING) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    return PIE_STATUS_OK;
}

inline int validate_terminal_cells(PieTerminalCellPtrSlice cells) noexcept {
    int status = validate_slice(cells.ptr, cells.len);
    if (status != PIE_STATUS_OK) return status;
    for (std::size_t i = 0; i < cells.len; ++i) {
        status = validate_terminal_cell_ptr(cells.ptr[i]);
        if (status != PIE_STATUS_OK) return status;
        for (std::size_t j = 0; j < i; ++j) {
            if (cells.ptr[i] == cells.ptr[j]) {
                return PIE_STATUS_INVALID_ARGUMENT;
            }
        }
    }
    return PIE_STATUS_OK;
}

inline bool valid_memory_domain(std::uint32_t domain) noexcept {
    return domain == PIE_MEMORY_DOMAIN_HOST_PINNED ||
           domain == PIE_MEMORY_DOMAIN_CUDA_DEVICE ||
           domain == PIE_MEMORY_DOMAIN_ROCM_DEVICE ||
           domain == PIE_MEMORY_DOMAIN_METAL_SHARED ||
           domain == PIE_MEMORY_DOMAIN_METAL_PRIVATE;
}

inline int validate_channel_values(PieChannelValueDescSlice values) noexcept {
    int status = validate_slice(values.ptr, values.len);
    if (status != PIE_STATUS_OK) return status;
    if (values.len > std::numeric_limits<std::uint32_t>::max()) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    for (std::size_t i = 0; i < values.len; ++i) {
        status = validate_bytes(values.ptr[i].bytes);
        if (status != PIE_STATUS_OK ||
            values.ptr[i].bytes.len > std::numeric_limits<std::uint32_t>::max()) {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
    }
    return PIE_STATUS_OK;
}

inline int validate_csr(PieU32Slice indptr,
                        std::size_t values_len,
                        std::size_t outer_count) noexcept {
    int status = validate_slice(indptr.ptr, indptr.len);
    if (status != PIE_STATUS_OK) return status;
    if (indptr.len == 0) {
        return values_len == 0 ? PIE_STATUS_OK : PIE_STATUS_INVALID_ARGUMENT;
    }
    if (outer_count == std::numeric_limits<std::size_t>::max() ||
        indptr.len != outer_count + 1 ||
        values_len > std::numeric_limits<std::uint32_t>::max() ||
        indptr.ptr[0] != 0) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    for (std::size_t i = 0; i < outer_count; ++i) {
        if (indptr.ptr[i] > indptr.ptr[i + 1]) {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
    }
    return indptr.ptr[outer_count] == values_len
        ? PIE_STATUS_OK
        : PIE_STATUS_INVALID_ARGUMENT;
}

inline int validate_optional_rows(std::size_t len,
                                  std::size_t rows) noexcept {
    return (len == 0 || len == rows)
        ? PIE_STATUS_OK
        : PIE_STATUS_INVALID_ARGUMENT;
}

inline int validate_create_desc(const PieDriverCreateDesc* desc,
                                PieDriverCaps* caps) noexcept {
    if (desc == nullptr || caps == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    int status = validate_version(desc->abi_version);
    if (status != PIE_STATUS_OK) return status;
    status = validate_version(desc->runtime.abi_version);
    if (status != PIE_STATUS_OK) return status;
    if (desc->reserved0 != 0 || desc->runtime.reserved0 != 0 ||
        desc->runtime.notify == nullptr) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    status = validate_bytes(desc->config_bytes);
    if (status != PIE_STATUS_OK || desc->config_bytes.len == 0) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    return PIE_STATUS_OK;
}

inline int validate_program_desc(const PieProgramDesc* desc,
                                 std::uint64_t* program_id) noexcept {
    if (desc == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    int status = validate_version(desc->abi_version);
    if (status != PIE_STATUS_OK) return status;
    if (desc->reserved0 != 0 || program_id == nullptr) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    status = validate_bytes(desc->canonical_bytes);
    if (status != PIE_STATUS_OK) return status;
    return validate_bytes(desc->sidecar_bytes);
}

inline int validate_channel_desc(
    const PieChannelDesc* desc,
    const PieChannelEndpointBinding* binding) noexcept {
    if (desc == nullptr || binding == nullptr) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    int status = validate_version(desc->abi_version);
    if (status != PIE_STATUS_OK) return status;
    if (desc->reserved0 != 0 || desc->reserved1 != 0 ||
        desc->channel_id == 0 || desc->capacity == 0 ||
        desc->dtype > PIE_CHANNEL_DTYPE_ACT ||
        desc->host_role > PIE_CHANNEL_HOST_ROLE_READER ||
        desc->seeded > 1 ||
        desc->extern_dir > PIE_CHANNEL_EXTERN_EXPORT ||
        desc->reader_wait_id == 0 || desc->writer_wait_id == 0 ||
        desc->reader_wait_id == desc->writer_wait_id) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    status = validate_slice(desc->shape.ptr, desc->shape.len);
    if (status != PIE_STATUS_OK) return status;
    status = validate_bytes(desc->extern_name);
    if (status != PIE_STATUS_OK) return status;
    for (std::size_t i = 0; i < desc->shape.len; ++i) {
        if (desc->shape.ptr[i] == 0) return PIE_STATUS_INVALID_ARGUMENT;
    }
    if (desc->extern_dir == PIE_CHANNEL_EXTERN_NONE) {
        if (desc->extern_name.len != 0) return PIE_STATUS_INVALID_ARGUMENT;
    } else if (desc->extern_name.len == 0 ||
               desc->host_role != PIE_CHANNEL_HOST_ROLE_NONE ||
               desc->seeded != 0) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    return PIE_STATUS_OK;
}

inline int validate_completion(
    PieCompletion completion,
    bool require_terminal_cell) noexcept {
    if (completion.wait_id == 0 || completion.target_epoch == 0) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    if (require_terminal_cell) {
        return validate_terminal_cell_ptr(completion.terminal_cell);
    }
    return completion.terminal_cell == nullptr
        ? PIE_STATUS_OK
        : PIE_STATUS_INVALID_ARGUMENT;
}

inline int validate_instance_desc(const PieInstanceDesc* desc,
                                  PieInstanceBinding* binding) noexcept {
    if (desc == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    int status = validate_version(desc->abi_version);
    if (status != PIE_STATUS_OK) return status;
    if (desc->reserved0 != 0 || binding == nullptr) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    status = validate_slice(desc->channel_ids.ptr, desc->channel_ids.len);
    if (status != PIE_STATUS_OK) return status;
    status = validate_channel_values(desc->seed_values);
    if (status != PIE_STATUS_OK) return status;
    if (desc->channel_ids.len > std::numeric_limits<std::uint32_t>::max()) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    return PIE_STATUS_OK;
}

inline int validate_masks(const PieMaskWordsDesc& masks,
                          std::size_t request_count) noexcept {
    int status = validate_slice(
        masks.request_indptr.ptr, masks.request_indptr.len);
    if (status != PIE_STATUS_OK) return status;
    status = validate_slice(masks.word_indptr.ptr, masks.word_indptr.len);
    if (status != PIE_STATUS_OK) return status;
    status = validate_slice(masks.words.ptr, masks.words.len);
    if (status != PIE_STATUS_OK) return status;

    if (masks.request_indptr.len == 0) {
        return masks.word_indptr.len == 0 && masks.words.len == 0
            ? PIE_STATUS_OK
            : PIE_STATUS_INVALID_ARGUMENT;
    }
    if (masks.request_indptr.len == 1 &&
        masks.request_indptr.ptr[0] == 0 &&
        masks.word_indptr.len == 1 &&
        masks.word_indptr.ptr[0] == 0 &&
        masks.words.len == 0) {
        return PIE_STATUS_OK;
    }
    if (request_count == std::numeric_limits<std::size_t>::max() ||
        masks.request_indptr.len != request_count + 1 ||
        masks.request_indptr.ptr[0] != 0) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    for (std::size_t i = 0; i < request_count; ++i) {
        if (masks.request_indptr.ptr[i] > masks.request_indptr.ptr[i + 1]) {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
    }
    const std::size_t row_count =
        masks.request_indptr.ptr[request_count];
    return validate_csr(masks.word_indptr, masks.words.len, row_count);
}

inline int validate_launch_desc(const PieLaunchDesc* desc) noexcept {
    if (desc == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    int status = validate_version(desc->abi_version);
    if (status != PIE_STATUS_OK) return status;
    if (desc->reserved0 != 0 || desc->single_token_mode > 1 ||
        desc->has_user_mask > 1) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    for (std::uint8_t value : desc->reserved_flags) {
        if (value != 0) return PIE_STATUS_INVALID_ARGUMENT;
    }

#define PIE_VALIDATE_SLICE(field)                                              \
    do {                                                                       \
        status = validate_slice(desc->field.ptr, desc->field.len);             \
        if (status != PIE_STATUS_OK) return status;                            \
    } while (false)

    PIE_VALIDATE_SLICE(instance_ids);
    status = validate_terminal_cells(desc->terminal_cells);
    if (status != PIE_STATUS_OK) return status;
    PIE_VALIDATE_SLICE(token_ids);
    PIE_VALIDATE_SLICE(position_ids);
    PIE_VALIDATE_SLICE(kv_page_indices);
    PIE_VALIDATE_SLICE(kv_page_indptr);
    PIE_VALIDATE_SLICE(kv_last_page_lens);
    PIE_VALIDATE_SLICE(qo_indptr);
    PIE_VALIDATE_SLICE(rs_slot_ids);
    PIE_VALIDATE_SLICE(rs_slot_flags);
    PIE_VALIDATE_SLICE(rs_fold_lens);
    PIE_VALIDATE_SLICE(rs_buffer_slot_ids);
    PIE_VALIDATE_SLICE(rs_buffer_slot_indptr);
    PIE_VALIDATE_SLICE(sampling_indices);
    PIE_VALIDATE_SLICE(sampling_indptr);
    PIE_VALIDATE_SLICE(context_ids);
    PIE_VALIDATE_SLICE(image_indptr);
    PIE_VALIDATE_SLICE(image_grids);
    PIE_VALIDATE_SLICE(image_anchor_positions);
    PIE_VALIDATE_SLICE(image_pixel_indptr);
    PIE_VALIDATE_SLICE(image_mrope_positions);
    PIE_VALIDATE_SLICE(image_mrope_indptr);
    PIE_VALIDATE_SLICE(image_patch_positions);
    PIE_VALIDATE_SLICE(image_anchor_rows);
    PIE_VALIDATE_SLICE(audio_feature_indptr);
    PIE_VALIDATE_SLICE(audio_anchor_rows);
    PIE_VALIDATE_SLICE(audio_indptr);
    PIE_VALIDATE_SLICE(host_put_indptr);
    PIE_VALIDATE_SLICE(kv_len);
    PIE_VALIDATE_SLICE(kv_len_device);
#undef PIE_VALIDATE_SLICE

    status = validate_bytes(desc->image_pixels);
    if (status != PIE_STATUS_OK) return status;
    status = validate_bytes(desc->audio_features);
    if (status != PIE_STATUS_OK) return status;
    status = validate_channel_values(desc->ptir_host_put_values);
    if (status != PIE_STATUS_OK) return status;
    status = validate_masks(desc->masks, desc->instance_ids.len);
    if (status != PIE_STATUS_OK) return status;

    const std::size_t request_count = desc->instance_ids.len;
    const std::size_t max_int =
        static_cast<std::size_t>(std::numeric_limits<int>::max());
    for (std::size_t len : {
             desc->instance_ids.len,
             desc->token_ids.len,
             desc->position_ids.len,
             desc->kv_page_indices.len,
             desc->kv_page_indptr.len,
             desc->kv_last_page_lens.len,
             desc->qo_indptr.len,
             desc->rs_slot_ids.len,
             desc->rs_slot_flags.len,
             desc->rs_fold_lens.len,
             desc->rs_buffer_slot_ids.len,
             desc->rs_buffer_slot_indptr.len,
             desc->masks.request_indptr.len,
             desc->masks.word_indptr.len,
             desc->masks.words.len,
             desc->sampling_indices.len,
             desc->sampling_indptr.len,
             desc->context_ids.len,
             desc->image_indptr.len,
             desc->image_grids.len,
             desc->image_anchor_positions.len,
             desc->image_pixels.len,
             desc->image_pixel_indptr.len,
             desc->image_mrope_positions.len,
             desc->image_mrope_indptr.len,
             desc->image_patch_positions.len,
             desc->image_anchor_rows.len,
             desc->audio_features.len,
             desc->audio_feature_indptr.len,
             desc->audio_anchor_rows.len,
             desc->audio_indptr.len,
             desc->ptir_host_put_values.len,
             desc->host_put_indptr.len,
             desc->kv_len.len,
             desc->kv_len_device.len,
         }) {
        if (len > max_int) return PIE_STATUS_INVALID_ARGUMENT;
    }
    if (request_count > std::numeric_limits<std::uint32_t>::max() ||
        desc->terminal_cells.len != request_count ||
        desc->position_ids.len != desc->token_ids.len) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    for (std::size_t i = 0; i < request_count; ++i) {
        for (std::size_t j = 0; j < i; ++j) {
            if (desc->instance_ids.ptr[i] == desc->instance_ids.ptr[j]) {
                return PIE_STATUS_INVALID_ARGUMENT;
            }
        }
    }
    for (std::size_t i = 0; i < desc->rs_slot_flags.len; ++i) {
        if ((desc->rs_slot_flags.ptr[i] &
             ~(PIE_RS_FLAG_RESET | PIE_RS_FLAG_FOLD)) != 0) {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
    }
    if (desc->rs_slot_ids.len != desc->rs_slot_flags.len) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    for (std::size_t i = 0; i < desc->rs_slot_ids.len; ++i) {
        if (desc->rs_slot_ids.ptr[i] >
            static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
    }
    for (std::size_t i = 0; i < desc->rs_buffer_slot_ids.len; ++i) {
        if (desc->rs_buffer_slot_ids.ptr[i] >
            static_cast<std::uint32_t>(std::numeric_limits<int>::max())) {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
    }

    for (std::size_t len : {
             desc->kv_last_page_lens.len,
             desc->rs_slot_ids.len,
             desc->rs_fold_lens.len,
             desc->context_ids.len,
             desc->kv_len.len,
         }) {
        status = validate_optional_rows(len, request_count);
        if (status != PIE_STATUS_OK) return status;
    }
    if (desc->kv_len_device.len > 1) return PIE_STATUS_INVALID_ARGUMENT;

    status = validate_csr(
        desc->qo_indptr, desc->token_ids.len, request_count);
    if (status != PIE_STATUS_OK) return status;
    status = validate_csr(
        desc->kv_page_indptr, desc->kv_page_indices.len, request_count);
    if (status != PIE_STATUS_OK) return status;
    status = validate_csr(
        desc->rs_buffer_slot_indptr,
        desc->rs_buffer_slot_ids.len,
        request_count);
    if (status != PIE_STATUS_OK) return status;
    status = validate_csr(
        desc->sampling_indptr, desc->sampling_indices.len, request_count);
    if (status != PIE_STATUS_OK) return status;
    status = validate_csr(
        desc->host_put_indptr,
        desc->ptir_host_put_values.len,
        request_count);
    if (status != PIE_STATUS_OK) return status;
    if (desc->kv_page_indptr.len != 0 &&
        desc->kv_last_page_lens.len != request_count) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    if (desc->token_ids.len != 0 &&
        (desc->kv_page_indptr.len == 0 ||
         desc->kv_last_page_lens.len != request_count ||
         desc->sampling_indptr.len == 0)) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    if (desc->sampling_indices.len != 0 && desc->qo_indptr.len == 0) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    if (desc->sampling_indices.len != 0) {
        for (std::size_t request = 0; request < request_count; ++request) {
            const std::uint32_t query_rows =
                desc->qo_indptr.ptr[request + 1] -
                desc->qo_indptr.ptr[request];
            for (std::uint32_t index =
                     desc->sampling_indptr.ptr[request];
                 index < desc->sampling_indptr.ptr[request + 1];
                 ++index) {
                if (desc->sampling_indices.ptr[index] >= query_rows) {
                    return PIE_STATUS_INVALID_ARGUMENT;
                }
            }
        }
    }

    if (desc->image_grids.len % 3 != 0 ||
        desc->image_patch_positions.len % 2 != 0) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    const std::size_t image_count = desc->image_grids.len / 3;
    if (desc->image_anchor_positions.len != image_count ||
        desc->image_anchor_rows.len != image_count) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    for (std::size_t i = 0; i < image_count; ++i) {
        if (desc->image_anchor_rows.ptr[i] >= desc->token_ids.len) {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
    }
    status = validate_csr(
        desc->image_indptr, image_count, request_count);
    if (status != PIE_STATUS_OK) return status;
    if (image_count != 0 && desc->image_pixel_indptr.len == 0) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    status = validate_csr(
        desc->image_pixel_indptr, desc->image_pixels.len, image_count);
    if (status != PIE_STATUS_OK) return status;
    status = validate_csr(
        desc->image_mrope_indptr,
        desc->image_mrope_positions.len,
        image_count);
    if (status != PIE_STATUS_OK) return status;

    const std::size_t audio_count = desc->audio_anchor_rows.len;
    for (std::size_t i = 0; i < audio_count; ++i) {
        if (desc->audio_anchor_rows.ptr[i] >= desc->token_ids.len) {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
    }
    status = validate_csr(
        desc->audio_indptr, audio_count, request_count);
    if (status != PIE_STATUS_OK) return status;
    if (audio_count != 0 && desc->audio_feature_indptr.len == 0) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    status = validate_csr(
        desc->audio_feature_indptr,
        desc->audio_features.len,
        audio_count);
    if (status != PIE_STATUS_OK) return status;
    return PIE_STATUS_OK;
}

inline int validate_kv_copy_desc(const PieKvCopyDesc* desc) noexcept {
    if (desc == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    int status = validate_version(desc->abi_version);
    if (status != PIE_STATUS_OK) return status;
    if (desc->reserved0 != 0 ||
        !valid_memory_domain(desc->src_domain) ||
        !valid_memory_domain(desc->dst_domain)) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    status = validate_slice(desc->src_page_ids.ptr, desc->src_page_ids.len);
    if (status != PIE_STATUS_OK) return status;
    status = validate_slice(desc->dst_page_ids.ptr, desc->dst_page_ids.len);
    if (status != PIE_STATUS_OK) return status;
    status = validate_slice(desc->cells.ptr, desc->cells.len);
    if (status != PIE_STATUS_OK) return status;
    if (desc->src_page_ids.len != desc->dst_page_ids.len ||
        desc->cells.len >
            static_cast<std::size_t>(std::numeric_limits<int>::max())) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    return PIE_STATUS_OK;
}

inline int validate_state_copy_desc(const PieStateCopyDesc* desc) noexcept {
    if (desc == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    int status = validate_version(desc->abi_version);
    if (status != PIE_STATUS_OK) return status;
    if (desc->reserved0 != 0) return PIE_STATUS_INVALID_ARGUMENT;
    status = validate_slice(desc->slot_ranges.ptr, desc->slot_ranges.len);
    if (status != PIE_STATUS_OK) return status;
    for (std::size_t i = 0; i < desc->slot_ranges.len; ++i) {
        const PieStateCopyRange& range = desc->slot_ranges.ptr[i];
        if (range.src_slot_id >
                static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
            range.dst_slot_id >
                static_cast<std::uint32_t>(std::numeric_limits<int>::max()) ||
            static_cast<std::uint64_t>(range.src_token_offset) +
                    range.token_count >
                std::numeric_limits<std::uint32_t>::max() ||
            static_cast<std::uint64_t>(range.dst_token_offset) +
                    range.token_count >
                std::numeric_limits<std::uint32_t>::max()) {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
    }
    return PIE_STATUS_OK;
}

inline int validate_pool_resize_desc(const PiePoolResizeDesc* desc) noexcept {
    if (desc == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    int status = validate_version(desc->abi_version);
    if (status != PIE_STATUS_OK) return status;
    if (desc->reserved0 != 0 ||
        desc->target_pages >
            static_cast<std::uint64_t>(std::numeric_limits<int>::max())) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    status = validate_slice(desc->map_ranges.ptr, desc->map_ranges.len);
    if (status != PIE_STATUS_OK) return status;
    status = validate_slice(desc->unmap_ranges.ptr, desc->unmap_ranges.len);
    if (status != PIE_STATUS_OK) return status;

    for (PiePoolRangeSlice ranges :
         {desc->map_ranges, desc->unmap_ranges}) {
        for (std::size_t i = 0; i < ranges.len; ++i) {
            if (ranges.ptr[i].page_count >
                std::numeric_limits<std::uint64_t>::max() -
                    ranges.ptr[i].page_index) {
                return PIE_STATUS_INVALID_ARGUMENT;
            }
        }
    }
    return PIE_STATUS_OK;
}

}  // namespace pie_native::abi
