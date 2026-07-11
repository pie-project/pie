#include "batch/compose.hpp"

namespace pie::metal::batch {

namespace {

constexpr std::uint8_t kRsFlagReset = 1;

}  // namespace

pie_native::LaunchView build_launch_view(const PieLaunchDesc& launch) {
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
    return view;
}

bool build_member_forward_desc(
    const pie_native::LaunchView& view,
    std::size_t member,
    std::size_t member_count,
    bool has_linear_attn,
    const pie_native::ptir::FireGeometry* resolved,
    MemberForwardDesc& desc,
    std::string& error) {
    if (resolved != nullptr) {
        desc.token_ids = resolved->token_ids;
        desc.position_ids = resolved->position_ids;
        desc.kv_pages = resolved->kv_page_indices;
        desc.kv_last_page_len =
            resolved->kv_last_page_lens.empty()
                ? 0
                : resolved->kv_last_page_lens.back();
        desc.readout_local_indices = resolved->sampling_indices;
        desc.has_write_desc = resolved->has_write_desc;
        desc.w_page = resolved->w_page;
        desc.w_off = resolved->w_off;
        desc.requires_paged = true;
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

    desc.has_rs_slot =
        view.rs_slot_ids.size() == member_count &&
        view.rs_slot_flags.size() == member_count;
    if (has_linear_attn && !desc.has_rs_slot) {
        error =
            "missing recurrent-state slot assignment for a hybrid-attention model";
        return false;
    }
    if (desc.has_rs_slot) {
        desc.rs_slot_id = view.rs_slot_ids.data()[member];
        desc.rs_reset =
            (view.rs_slot_flags.data()[member] & kRsFlagReset) != 0;
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
    return true;
}

}  // namespace pie::metal::batch
