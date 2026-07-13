#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <limits>

#include <pie_driver_abi.h>

namespace pie_cuda_driver::abi {

struct MultimodalLimits {
    int gemma4_pool_kernel = 0;
    int gemma4_position_table = 0;
    int qwen3_vl_patch_dim = 0;
    int qwen3_vl_merge_unit = 0;
    int audio_mel_bins = 0;
};

inline int validate_encode_resources(
    const PieEncodeDesc& encode,
    const MultimodalLimits& multimodal,
    int hidden_size) noexcept {
    if (multimodal.gemma4_pool_kernel <= 0 ||
        multimodal.gemma4_position_table <= 0 || hidden_size <= 0) {
        return PIE_STATUS_UNSUPPORTED;
    }
    const std::uint64_t patch_bytes =
        3u * 16u * 16u * sizeof(float);
    const std::uint64_t pool_area =
        static_cast<std::uint64_t>(multimodal.gemma4_pool_kernel) *
        static_cast<std::uint64_t>(multimodal.gemma4_pool_kernel);
    std::uint64_t patch_count = 0;
    std::uint64_t output_rows = 0;
    for (std::size_t image = 0; image < encode.image_anchor_rows.len; ++image) {
        const std::uint64_t begin = encode.image_pixel_indptr.ptr[image];
        const std::uint64_t end = encode.image_pixel_indptr.ptr[image + 1];
        if (end <= begin || (end - begin) % patch_bytes != 0) {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
        const std::uint64_t patches = (end - begin) / patch_bytes;
        if (patches == 0 || patches % pool_area != 0 ||
            patch_count > std::numeric_limits<std::uint64_t>::max() - patches ||
            output_rows > std::numeric_limits<std::uint64_t>::max() -
                patches / pool_area) {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
        for (std::uint64_t patch = 0; patch < patches; ++patch) {
            const std::uint64_t position = (patch_count + patch) * 2;
            if (position + 1 >= encode.image_patch_positions.len ||
                encode.image_patch_positions.ptr[position] >=
                    static_cast<std::uint32_t>(
                        multimodal.gemma4_position_table) ||
                encode.image_patch_positions.ptr[position + 1] >=
                    static_cast<std::uint32_t>(
                        multimodal.gemma4_position_table)) {
                return PIE_STATUS_INVALID_ARGUMENT;
            }
        }
        patch_count += patches;
        output_rows += patches / pool_area;
    }
    if (patch_count > std::numeric_limits<std::size_t>::max() / 2 ||
        encode.image_patch_positions.len !=
            static_cast<std::size_t>(patch_count) * 2 ||
        output_rows > std::numeric_limits<std::uint32_t>::max()) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    const std::uint64_t required_bytes =
        output_rows * static_cast<std::uint64_t>(hidden_size) *
        sizeof(std::uint16_t);
    if (required_bytes > encode.output_rows.len) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    return PIE_STATUS_OK;
}

inline int validate_launch_resources(const PieLaunchDesc& launch,
                                     int device_pages,
                                     int page_size,
                                     int rs_slots,
                                     int rs_buffer_slots,
                                     const MultimodalLimits& multimodal) noexcept {
    if (device_pages < 0 || page_size <= 0) {
        return PIE_STATUS_DRIVER_ERROR;
    }
    const PieU32Slice physical_pages =
        launch.kv_translation.len != 0
            ? launch.kv_translation
            : launch.kv_page_indices;
    for (std::size_t i = 0; i < physical_pages.len; ++i) {
        if (physical_pages.ptr[i] >=
            static_cast<std::uint32_t>(device_pages)) {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
    }
    const std::size_t wire_rows =
        launch.qo_indptr.len == 0 ? 0 : launch.qo_indptr.len - 1;
    for (std::size_t request = 0;
         request < wire_rows && launch.kv_page_indptr.len != 0;
         ++request) {
        const bool has_pages =
            launch.kv_page_indptr.ptr[request] !=
            launch.kv_page_indptr.ptr[request + 1];
        const std::uint32_t page_count =
            launch.kv_page_indptr.ptr[request + 1] -
            launch.kv_page_indptr.ptr[request];
        const std::uint32_t last_len =
            launch.kv_last_page_lens.ptr[request];
        if ((!has_pages && last_len != 0) ||
            (has_pages &&
             (last_len == 0 ||
              last_len > static_cast<std::uint32_t>(page_size)))) {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
        const std::uint32_t query_rows =
            launch.qo_indptr.ptr[request + 1] -
            launch.qo_indptr.ptr[request];
        if (query_rows != 0) {
            if (!has_pages) return PIE_STATUS_INVALID_ARGUMENT;
            const std::uint64_t kv_len =
                static_cast<std::uint64_t>(page_count - 1) *
                    static_cast<std::uint32_t>(page_size) +
                last_len;
            if (kv_len < query_rows) {
                return PIE_STATUS_INVALID_ARGUMENT;
            }
        }
    }

    if (launch.rs_slot_ids.len != 0) {
        if (rs_slots <= 0) return PIE_STATUS_INVALID_ARGUMENT;
        for (std::size_t i = 0; i < launch.rs_slot_ids.len; ++i) {
            if (launch.rs_slot_ids.ptr[i] >=
                static_cast<std::uint32_t>(rs_slots)) {
                return PIE_STATUS_INVALID_ARGUMENT;
            }
        }
    }
    if (launch.rs_buffer_slot_ids.len != 0) {
        if (rs_buffer_slots <= 0) return PIE_STATUS_INVALID_ARGUMENT;
        for (std::size_t i = 0; i < launch.rs_buffer_slot_ids.len; ++i) {
            if (launch.rs_buffer_slot_ids.ptr[i] >=
                static_cast<std::uint32_t>(rs_buffer_slots)) {
                return PIE_STATUS_INVALID_ARGUMENT;
            }
        }
    }

    const std::size_t image_count = launch.image_anchor_rows.len;
    if (image_count != 0 &&
        multimodal.gemma4_pool_kernel <= 0 &&
        multimodal.qwen3_vl_patch_dim <= 0) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    if (multimodal.qwen3_vl_patch_dim > 0 && image_count != 0) {
        if (multimodal.qwen3_vl_merge_unit <= 0 ||
            launch.image_mrope_indptr.len != image_count + 1) {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
        for (std::size_t image = 0; image < image_count; ++image) {
            const std::uint64_t temporal = launch.image_grids.ptr[3 * image];
            const std::uint64_t height = launch.image_grids.ptr[3 * image + 1];
            const std::uint64_t width = launch.image_grids.ptr[3 * image + 2];
            if (temporal == 0 || height == 0 || width == 0 ||
                temporal > std::numeric_limits<std::uint64_t>::max() / height ||
                temporal * height >
                    std::numeric_limits<std::uint64_t>::max() / width) {
                return PIE_STATUS_INVALID_ARGUMENT;
            }
            const std::uint64_t patches = temporal * height * width;
            const std::uint64_t patch_bytes =
                static_cast<std::uint64_t>(multimodal.qwen3_vl_patch_dim) *
                sizeof(float);
            if (patches >
                std::numeric_limits<std::uint64_t>::max() / patch_bytes) {
                return PIE_STATUS_INVALID_ARGUMENT;
            }
            const std::uint64_t bytes =
                launch.image_pixel_indptr.ptr[image + 1] -
                launch.image_pixel_indptr.ptr[image];
            if (bytes != patches * patch_bytes ||
                patches %
                    static_cast<std::uint64_t>(
                        multimodal.qwen3_vl_merge_unit) != 0) {
                return PIE_STATUS_INVALID_ARGUMENT;
            }
            const std::uint64_t output_rows =
                patches / static_cast<std::uint64_t>(
                              multimodal.qwen3_vl_merge_unit);
            if (output_rows >
                launch.token_ids.len -
                    launch.image_anchor_rows.ptr[image]) {
                return PIE_STATUS_INVALID_ARGUMENT;
            }
            const std::uint64_t mrope_values =
                launch.image_mrope_indptr.ptr[image + 1] -
                launch.image_mrope_indptr.ptr[image];
            if (output_rows >
                    std::numeric_limits<std::uint64_t>::max() / 3 ||
                mrope_values != output_rows * 3) {
                return PIE_STATUS_INVALID_ARGUMENT;
            }
        }
    }
    if (multimodal.gemma4_pool_kernel > 0 && image_count != 0) {
        constexpr std::uint64_t patch_bytes =
            3u * 16u * 16u * sizeof(float);
        const std::uint64_t pool_area =
            static_cast<std::uint64_t>(multimodal.gemma4_pool_kernel) *
            static_cast<std::uint64_t>(multimodal.gemma4_pool_kernel);
        std::uint64_t patch_count = 0;
        for (std::size_t image = 0;
             image < image_count;
             ++image) {
            const std::uint64_t bytes =
                launch.image_pixel_indptr.ptr[image + 1] -
                launch.image_pixel_indptr.ptr[image];
            if (bytes % patch_bytes != 0) {
                return PIE_STATUS_INVALID_ARGUMENT;
            }
            const std::uint64_t image_patches = bytes / patch_bytes;
            if (image_patches % pool_area != 0 ||
                patch_count >
                    std::numeric_limits<std::uint64_t>::max() -
                        image_patches) {
                return PIE_STATUS_INVALID_ARGUMENT;
            }
            patch_count += image_patches;
            const std::uint64_t output_rows = image_patches / pool_area;
            if (output_rows >
                launch.token_ids.len -
                    launch.image_anchor_rows.ptr[image]) {
                return PIE_STATUS_INVALID_ARGUMENT;
            }
            if (patch_count >
                    std::numeric_limits<std::size_t>::max() / 2 ||
                launch.image_patch_positions.len <
                    static_cast<std::size_t>(patch_count) * 2) {
                return PIE_STATUS_INVALID_ARGUMENT;
            }
            std::uint32_t max_x = 0;
            for (std::uint64_t patch = 0; patch < image_patches; ++patch) {
                const std::size_t position =
                    static_cast<std::size_t>((patch_count - image_patches + patch) * 2);
                const std::uint32_t x =
                    launch.image_patch_positions.ptr[position];
                const std::uint32_t y =
                    launch.image_patch_positions.ptr[position + 1];
                if (multimodal.gemma4_position_table <= 0 ||
                    x >= static_cast<std::uint32_t>(
                        multimodal.gemma4_position_table) ||
                    y >= static_cast<std::uint32_t>(
                        multimodal.gemma4_position_table)) {
                    return PIE_STATUS_INVALID_ARGUMENT;
                }
                max_x = std::max(max_x, x);
            }
            const std::uint64_t groups_x =
                (static_cast<std::uint64_t>(max_x) + 1) /
                static_cast<std::uint64_t>(
                    multimodal.gemma4_pool_kernel);
            if (groups_x == 0) return PIE_STATUS_INVALID_ARGUMENT;
            for (std::uint64_t patch = 0; patch < image_patches; ++patch) {
                const std::size_t position =
                    static_cast<std::size_t>((patch_count - image_patches + patch) * 2);
                const std::uint64_t x =
                    launch.image_patch_positions.ptr[position];
                const std::uint64_t y =
                    launch.image_patch_positions.ptr[position + 1];
                const std::uint64_t group =
                    x / multimodal.gemma4_pool_kernel +
                    groups_x * (y / multimodal.gemma4_pool_kernel);
                if (group >= output_rows) {
                    return PIE_STATUS_INVALID_ARGUMENT;
                }
            }
        }
        if (patch_count >
                std::numeric_limits<std::size_t>::max() / 2 ||
            launch.image_patch_positions.len !=
                static_cast<std::size_t>(patch_count) * 2) {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
    }
    const std::size_t audio_count = launch.audio_anchor_rows.len;
    if (audio_count != 0 && multimodal.audio_mel_bins <= 0) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    for (std::size_t clip = 0; clip < audio_count; ++clip) {
        const std::uint64_t bytes =
            launch.audio_feature_indptr.ptr[clip + 1] -
            launch.audio_feature_indptr.ptr[clip];
        const std::uint64_t frame_bytes =
            static_cast<std::uint64_t>(multimodal.audio_mel_bins) *
            sizeof(float);
        if (bytes == 0 || bytes % frame_bytes != 0) {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
        std::uint64_t output_rows = bytes / frame_bytes;
        output_rows = (output_rows + 1) / 2;
        output_rows = (output_rows + 1) / 2;
        if (output_rows >
            launch.token_ids.len -
                launch.audio_anchor_rows.ptr[clip]) {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
    }
    return PIE_STATUS_OK;
}

inline int validate_kv_copy_resources(const PieKvCopyDesc& copy,
                                      std::uint32_t device_ordinal,
                                      int device_pages,
                                      int host_pages,
                                      int page_size,
                                      bool native_bf16) noexcept {
    if (device_pages < 0 || host_pages < 0 || page_size <= 0) {
        return PIE_STATUS_DRIVER_ERROR;
    }
    const bool src_cuda = copy.src_domain == PIE_MEMORY_DOMAIN_CUDA_DEVICE;
    const bool dst_cuda = copy.dst_domain == PIE_MEMORY_DOMAIN_CUDA_DEVICE;
    const bool src_host = copy.src_domain == PIE_MEMORY_DOMAIN_HOST_PINNED;
    const bool dst_host = copy.dst_domain == PIE_MEMORY_DOMAIN_HOST_PINNED;
    if ((!src_cuda && !src_host) || (!dst_cuda && !dst_host)) {
        return PIE_STATUS_UNSUPPORTED;
    }
    if ((src_cuda && copy.src_device_ordinal != device_ordinal) ||
        (dst_cuda && copy.dst_device_ordinal != device_ordinal)) {
        return PIE_STATUS_UNSUPPORTED;
    }
    if (copy.cells.len != 0 && (!src_cuda || !dst_cuda)) {
        return PIE_STATUS_UNSUPPORTED;
    }
    if (copy.cells.len != 0 && !native_bf16) {
        return PIE_STATUS_UNSUPPORTED;
    }

    const std::uint32_t src_pages = static_cast<std::uint32_t>(
        src_cuda ? device_pages : host_pages);
    const std::uint32_t dst_pages = static_cast<std::uint32_t>(
        dst_cuda ? device_pages : host_pages);
    for (std::size_t i = 0; i < copy.src_page_ids.len; ++i) {
        if (copy.src_page_ids.ptr[i] >= src_pages ||
            copy.dst_page_ids.ptr[i] >= dst_pages) {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
    }
    for (std::size_t i = 0; i < copy.cells.len; ++i) {
        const PieKvMoveCell& cell = copy.cells.ptr[i];
        if (cell.src_page_id >= static_cast<std::uint32_t>(device_pages) ||
            cell.dst_page_id >= static_cast<std::uint32_t>(device_pages) ||
            cell.src_token_offset >= static_cast<std::uint32_t>(page_size) ||
            cell.dst_token_offset >= static_cast<std::uint32_t>(page_size)) {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
    }
    return PIE_STATUS_OK;
}

inline int validate_state_copy_resources(const PieStateCopyDesc& copy,
                                         int rs_slots) noexcept {
    if (rs_slots <= 0) return PIE_STATUS_UNSUPPORTED;
    for (std::size_t i = 0; i < copy.slot_ranges.len; ++i) {
        const PieStateCopyRange& range = copy.slot_ranges.ptr[i];
        if (range.src_token_offset != 0 ||
            range.dst_token_offset != 0 ||
            range.token_count != 0) {
            return PIE_STATUS_UNSUPPORTED;
        }
        if (range.src_slot_id >= static_cast<std::uint32_t>(rs_slots) ||
            range.dst_slot_id >= static_cast<std::uint32_t>(rs_slots)) {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
    }
    return PIE_STATUS_OK;
}

}  // namespace pie_cuda_driver::abi
