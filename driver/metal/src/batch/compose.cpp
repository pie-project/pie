#include <pie_native/step_launch.hpp>
#include "batch/compose.hpp"

#include <algorithm>

namespace pie::metal::batch {

namespace {

constexpr std::uint8_t kRsFlagReset = 1;

template <typename T>
std::vector<T> copy_slice(const T* ptr, std::size_t len) {
    if (len == 0) return {};
    return std::vector<T>(ptr, ptr + len);
}

}  // namespace

pie_native::LaunchView build_launch_view(const pie_native::StepLaunch& launch) {
    pie_native::LaunchView view{};
    view.token_ids =
        pie_native::slice_from_u32(launch.token_ids.ptr, launch.token_ids.len);
    view.position_ids =
        pie_native::slice_from_u32(launch.position_ids.ptr, launch.position_ids.len);
    view.kv_page_indices =
        pie_native::slice_from_u32(
            launch.kv_page_indices.ptr,
            launch.kv_page_indices.len);
    view.kv_page_indptr =
        pie_native::slice_from_u32(
            launch.kv_page_indptr.ptr,
            launch.kv_page_indptr.len);
    view.kv_last_page_lens =
        pie_native::slice_from_u32(
            launch.kv_last_page_lens.ptr,
            launch.kv_last_page_lens.len);
    view.qo_indptr =
        pie_native::slice_from_u32(launch.qo_indptr.ptr, launch.qo_indptr.len);
    view.rs_slot_ids =
        pie_native::slice_from_u32(
            launch.rs_slot_ids.ptr,
            launch.rs_slot_ids.len);
    view.rs_slot_flags =
        pie_native::slice_from_u8(
            launch.rs_slot_flags.ptr,
            launch.rs_slot_flags.len);
    view.rs_buffer_slot_ids =
        pie_native::slice_from_u32(
            launch.rs_buffer_slot_ids.ptr,
            launch.rs_buffer_slot_ids.len);
    view.rs_buffer_slot_indptr =
        pie_native::slice_from_u32(
            launch.rs_buffer_slot_indptr.ptr,
            launch.rs_buffer_slot_indptr.len);
    view.sampling_indices =
        pie_native::slice_from_u32(
            launch.sampling_indices.ptr,
            launch.sampling_indices.len);
    view.sampling_indptr =
        pie_native::slice_from_u32(
            launch.sampling_indptr.ptr,
            launch.sampling_indptr.len);
    view.kv_translation =
        pie_native::slice_from_u32(
            launch.kv_translation.ptr,
            launch.kv_translation.len);
    view.kv_translation_indptr =
        pie_native::slice_from_u32(
            launch.kv_translation_indptr.ptr,
            launch.kv_translation_indptr.len);
    view.flattened_masks =
        pie_native::slice_from_u32(
            launch.masks.words.ptr, launch.masks.words.len);
    view.mask_indptr =
        pie_native::slice_from_u32(
            launch.masks.word_indptr.ptr,
            launch.masks.word_indptr.len);
    view.has_user_mask = launch.has_user_mask != 0;
    return view;
}

OwnedLaunchView OwnedLaunchView::capture(const pie_native::StepLaunch& launch) {
    OwnedLaunchView owned;
    owned.token_ids = copy_slice(launch.token_ids.ptr, launch.token_ids.len);
    owned.position_ids = copy_slice(launch.position_ids.ptr, launch.position_ids.len);
    owned.kv_page_indices =
        copy_slice(launch.kv_page_indices.ptr, launch.kv_page_indices.len);
    owned.kv_page_indptr =
        copy_slice(launch.kv_page_indptr.ptr, launch.kv_page_indptr.len);
    owned.kv_last_page_lens =
        copy_slice(launch.kv_last_page_lens.ptr, launch.kv_last_page_lens.len);
    owned.qo_indptr = copy_slice(launch.qo_indptr.ptr, launch.qo_indptr.len);
    owned.rs_slot_ids = copy_slice(launch.rs_slot_ids.ptr, launch.rs_slot_ids.len);
    owned.rs_slot_flags =
        copy_slice(launch.rs_slot_flags.ptr, launch.rs_slot_flags.len);
    owned.rs_buffer_slot_ids =
        copy_slice(
            launch.rs_buffer_slot_ids.ptr,
            launch.rs_buffer_slot_ids.len);
    owned.rs_buffer_slot_indptr =
        copy_slice(
            launch.rs_buffer_slot_indptr.ptr,
            launch.rs_buffer_slot_indptr.len);
    owned.sampling_indices =
        copy_slice(launch.sampling_indices.ptr, launch.sampling_indices.len);
    owned.sampling_indptr =
        copy_slice(launch.sampling_indptr.ptr, launch.sampling_indptr.len);
    owned.kv_translation =
        copy_slice(launch.kv_translation.ptr, launch.kv_translation.len);
    owned.kv_translation_indptr =
        copy_slice(launch.kv_translation_indptr.ptr, launch.kv_translation_indptr.len);
    owned.mask_request_indptr = copy_slice(
        launch.masks.request_indptr.ptr,
        launch.masks.request_indptr.len);
    owned.mask_word_indptr = copy_slice(
        launch.masks.word_indptr.ptr,
        launch.masks.word_indptr.len);
    owned.mask_words =
        copy_slice(launch.masks.words.ptr, launch.masks.words.len);
    owned.required_kv_pages = launch.required_kv_pages;
    owned.has_user_mask = launch.has_user_mask != 0;
    return owned;
}

pie_native::LaunchView OwnedLaunchView::view() const {
    pie_native::LaunchView view{};
    view.token_ids = pie_native::slice_from_u32(token_ids.data(), token_ids.size());
    view.position_ids =
        pie_native::slice_from_u32(position_ids.data(), position_ids.size());
    view.kv_page_indices =
        pie_native::slice_from_u32(kv_page_indices.data(), kv_page_indices.size());
    view.kv_page_indptr =
        pie_native::slice_from_u32(kv_page_indptr.data(), kv_page_indptr.size());
    view.kv_last_page_lens =
        pie_native::slice_from_u32(kv_last_page_lens.data(), kv_last_page_lens.size());
    view.qo_indptr = pie_native::slice_from_u32(qo_indptr.data(), qo_indptr.size());
    view.rs_slot_ids =
        pie_native::slice_from_u32(rs_slot_ids.data(), rs_slot_ids.size());
    view.rs_slot_flags =
        pie_native::slice_from_u8(rs_slot_flags.data(), rs_slot_flags.size());
    view.rs_buffer_slot_ids = pie_native::slice_from_u32(
        rs_buffer_slot_ids.data(), rs_buffer_slot_ids.size());
    view.rs_buffer_slot_indptr = pie_native::slice_from_u32(
        rs_buffer_slot_indptr.data(), rs_buffer_slot_indptr.size());
    view.sampling_indices =
        pie_native::slice_from_u32(sampling_indices.data(), sampling_indices.size());
    view.sampling_indptr =
        pie_native::slice_from_u32(sampling_indptr.data(), sampling_indptr.size());
    view.kv_translation =
        pie_native::slice_from_u32(kv_translation.data(), kv_translation.size());
    view.kv_translation_indptr = pie_native::slice_from_u32(
        kv_translation_indptr.data(), kv_translation_indptr.size());
    view.flattened_masks =
        pie_native::slice_from_u32(mask_words.data(), mask_words.size());
    view.mask_indptr = pie_native::slice_from_u32(
        mask_word_indptr.data(), mask_word_indptr.size());
    view.has_user_mask = has_user_mask;
    return view;
}

bool build_member_forward_desc(
    const pie_native::LaunchView& view,
    std::size_t member,
    std::size_t member_count,
    bool has_linear_attn,
    std::uint32_t page_size,
    const pie_native::ptir::FireGeometry* resolved,
    MemberForwardDesc& desc,
    std::string& error) {
    page_size = std::max<std::uint32_t>(page_size, 1);
    if (resolved != nullptr) {
        desc.token_ids = resolved->token_ids;
        desc.position_ids = resolved->position_ids;
        desc.kv_pages = resolved->kv_page_indices;
        desc.qo_indptr = resolved->qo_indptr;
        desc.kv_page_indptr = resolved->kv_page_indptr;
        desc.kv_last_page_lens =
            resolved->kv_last_page_lens;
        desc.sampling_indptr = resolved->sampling_indptr;
        desc.kv_last_page_len =
            resolved->kv_last_page_lens.size() != 1
                ? 0
                : resolved->kv_last_page_lens[0];
        desc.readout_local_indices = resolved->sampling_indices;
        desc.has_write_desc = resolved->has_write_desc;
        desc.w_page = resolved->w_page;
        desc.w_off = resolved->w_off;
        desc.requires_paged = true;
        desc.has_attention_mask = resolved->has_mask;
        desc.attention_mask = resolved->mask;
        desc.structured_mask = resolved->structured_mask;
        if (desc.has_attention_mask) {
            if (desc.token_ids.empty() ||
                desc.attention_mask.empty() ||
                desc.attention_mask.size() %
                        desc.token_ids.size() !=
                    0) {
                error =
                    "resolved attention mask has an invalid dense shape";
                return false;
            }
            desc.attention_mask_stride =
                static_cast<std::uint32_t>(
                    desc.attention_mask.size() /
                    desc.token_ids.size());
        } else if (desc.structured_mask) {
            error =
                "structured attention mask has no dense fallback; direct "
                "structured Metal attention is not supported";
            return false;
        }
        desc.row_count = resolved->qo_indptr.empty()
                             ? 1
                             : static_cast<std::uint32_t>(
                                   resolved->qo_indptr.size() - 1);
        if (resolved->kv_page_indptr.size() == desc.row_count + 1 &&
            resolved->kv_last_page_lens.size() == desc.row_count) {
            for (std::uint32_t row = 0; row < desc.row_count; ++row) {
                const std::uint32_t pages =
                    resolved->kv_page_indptr[row + 1] -
                    resolved->kv_page_indptr[row];
                const std::uint32_t length =
                    pages == 0
                        ? 0
                        : (pages - 1) * page_size +
                              resolved->kv_last_page_lens[row];
                desc.key_len = std::max(desc.key_len, length);
            }
        }
    } else {
        if (view.qo_indptr.size() != member_count + 1) {
            error = "launch is missing qo_indptr for a forward-needing member";
            return false;
        }
        const std::uint32_t* qo = view.qo_indptr.data();
        const std::uint32_t begin = qo[member];
        const std::uint32_t end = qo[member + 1];
        if (end < begin ||
            end > view.token_ids.size() ||
            end > view.position_ids.size()) {
            error = "malformed qo_indptr/token_ids for this member";
            return false;
        }
        desc.token_ids.assign(
            view.token_ids.data() + begin,
            view.token_ids.data() + end);
        desc.position_ids.assign(
            view.position_ids.data() + begin,
            view.position_ids.data() + end);

        if (!view.kv_page_indptr.empty()) {
            if (view.kv_page_indptr.size() != member_count + 1) {
                error = "malformed kv_page_indptr for this launch";
                return false;
            }
            const std::uint32_t* pages = view.kv_page_indptr.data();
            const std::uint32_t page_begin = pages[member];
            const std::uint32_t page_end = pages[member + 1];
            if (page_end < page_begin ||
                page_end > view.kv_page_indices.size()) {
                error = "malformed kv_page_indices for this member";
                return false;
            }
            desc.kv_pages.assign(
                view.kv_page_indices.data() + page_begin,
                view.kv_page_indices.data() + page_end);
            if (view.kv_last_page_lens.size() == member_count) {
                desc.kv_last_page_len =
                    view.kv_last_page_lens.data()[member];
            }
        }
    }

    const std::size_t request_count =
        resolved != nullptr && resolved->qo_indptr.size() >= 2
            ? resolved->qo_indptr.size() - 1
            : 1;
    if (has_linear_attn) {
        if (resolved != nullptr) {
            if (view.rs_slot_ids.size() != request_count ||
                view.rs_slot_flags.size() != request_count) {
                error =
                    "resolved hybrid geometry requires exactly one folded "
                    "recurrent-state slot and flag per request";
                return false;
            }
            desc.request_rs_slot_ids.assign(
                view.rs_slot_ids.data(),
                view.rs_slot_ids.data() + request_count);
            for (std::size_t request = 0;
                 request < request_count;
                 ++request) {
                const std::uint8_t flag =
                    view.rs_slot_flags.data()[request];
                desc.request_rs_reset.push_back(
                    (flag & kRsFlagReset) != 0);
            }
        } else if (
            view.rs_slot_ids.size() == member_count &&
            view.rs_slot_flags.size() == member_count) {
            desc.request_rs_slot_ids = {
                view.rs_slot_ids.data()[member],
            };
            desc.request_rs_reset = {
                static_cast<std::uint8_t>(
                    (view.rs_slot_flags.data()[member] &
                     kRsFlagReset) != 0),
            };
        } else {
            error =
                "missing folded recurrent-state slot assignment for a "
                "hybrid-attention model";
            return false;
        }
        desc.request_rs_read.resize(request_count);
        desc.request_rs_write.assign(request_count, 1);
        for (std::size_t request = 0; request < request_count;
             ++request) {
            desc.request_rs_read[request] =
                desc.request_rs_reset[request] == 0;
        }
        desc.has_rs_slot = true;
        desc.rs_slot_id = desc.request_rs_slot_ids[0];
        desc.rs_reset = desc.request_rs_reset[0] != 0;
    }

    if (resolved == nullptr && !view.sampling_indptr.empty()) {
        if (view.sampling_indptr.size() != member_count + 1) {
            error = "malformed sampling_indptr for this launch";
            return false;
        }
        const std::uint32_t* sampling = view.sampling_indptr.data();
        const std::uint32_t begin = sampling[member];
        const std::uint32_t end = sampling[member + 1];
        if (end < begin || end > view.sampling_indices.size()) {
            error = "malformed sampling_indices for this member";
            return false;
        }
        desc.readout_local_indices.assign(
            view.sampling_indices.data() + begin,
            view.sampling_indices.data() + end);
    }
    if (desc.qo_indptr.empty()) {
        desc.qo_indptr = {
            0,
            static_cast<std::uint32_t>(
                desc.token_ids.size()),
        };
    }
    if (desc.kv_page_indptr.empty()) {
        desc.kv_page_indptr = {
            0,
            static_cast<std::uint32_t>(
                desc.kv_pages.size()),
        };
    }
    if (desc.kv_last_page_lens.empty()) {
        desc.kv_last_page_lens = {
            desc.kv_last_page_len,
        };
    }
    if (desc.sampling_indptr.empty()) {
        desc.sampling_indptr = {
            0,
            static_cast<std::uint32_t>(
                desc.readout_local_indices.size()),
        };
    }
    desc.sampled_rows =
        static_cast<std::uint32_t>(desc.readout_local_indices.size());
    desc.token_count =
        static_cast<std::uint32_t>(desc.token_ids.size());
    desc.page_count =
        static_cast<std::uint32_t>(desc.kv_pages.size());
    desc.query_len = desc.token_count;
    for (const std::uint32_t position : desc.position_ids) {
        desc.kv_len = std::max(desc.kv_len, position + 1);
    }
    if (desc.key_len == 0) {
        desc.key_len =
            desc.kv_pages.empty()
                ? desc.kv_len
                : static_cast<std::uint32_t>(
                      (desc.kv_pages.size() - 1) * page_size +
                      (desc.kv_last_page_len != 0
                           ? desc.kv_last_page_len
                           : (desc.position_ids.empty()
                                  ? 0
                                  : desc.position_ids.back() % page_size + 1)));
    }
    return true;
}

}  // namespace pie::metal::batch
