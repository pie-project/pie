#include <pie/driver/fire/step.hpp>
#include "batch/compose.hpp"

#include <stdexcept>
#include <algorithm>

namespace pie::metal::batch {

namespace {

constexpr std::uint8_t kRsFlagFold = 2;
constexpr std::uint8_t kRsFlagBufferWrite = 4;
constexpr std::uint8_t kRsFlagFoldLenDevice = 8;
constexpr std::uint8_t kRsFlagReset = 1;

template <typename T>
std::vector<T> copy_slice(const T* ptr, std::size_t len) {
    if (len == 0) return {};
    return std::vector<T>(ptr, ptr + len);
}

}  // namespace

pie::driver::fire::LaunchView build_launch_view(const pie::driver::fire::StepLaunch& launch) {
    pie::driver::fire::LaunchView view{};
    view.token_ids =
        pie::driver::slice_from_u32(launch.token_ids.ptr, launch.token_ids.len);
    view.position_ids =
        pie::driver::slice_from_u32(launch.position_ids.ptr, launch.position_ids.len);
    view.kv_page_indices =
        pie::driver::slice_from_u32(
            launch.kv_page_indices.ptr,
            launch.kv_page_indices.len);
    view.kv_page_indptr =
        pie::driver::slice_from_u32(
            launch.kv_page_indptr.ptr,
            launch.kv_page_indptr.len);
    view.kv_last_page_lens =
        pie::driver::slice_from_u32(
            launch.kv_last_page_lens.ptr,
            launch.kv_last_page_lens.len);
    view.qo_indptr =
        pie::driver::slice_from_u32(launch.qo_indptr.ptr, launch.qo_indptr.len);
    view.rs_slot_ids =
        pie::driver::slice_from_u32(
            launch.rs_slot_ids.ptr,
            launch.rs_slot_ids.len);
    view.rs_slot_flags =
        pie::driver::slice_from_u8(
            launch.rs_slot_flags.ptr,
            launch.rs_slot_flags.len);
    view.rs_buffer_slot_ids =
        pie::driver::slice_from_u32(
            launch.rs_buffer_slot_ids.ptr,
            launch.rs_buffer_slot_ids.len);
    view.rs_buffer_slot_indptr =
        pie::driver::slice_from_u32(
            launch.rs_buffer_slot_indptr.ptr,
            launch.rs_buffer_slot_indptr.len);
    // The buffer READ path is CUDA-only. Metal has no extended token layout,
    // so it would run the new tokens' recurrence from the FOLDED state and
    // silently ignore what is already buffered -- a wrong answer that then
    // gets folded and cannot be recovered. Refuse instead.
    if (launch.rs_buffer_read_lens.len != 0 &&
        std::any_of(launch.rs_buffer_read_lens.ptr,
                    launch.rs_buffer_read_lens.ptr +
                        launch.rs_buffer_read_lens.len,
                    [](std::uint32_t len) { return len != 0; })) {
        throw std::runtime_error(
            "this driver cannot replay buffered recurrent tokens; fold the "
            "buffer before appending to it");
    }
    // A device-resident fold length is CUDA-only too, and it fails in the
    // quietest way of all: the wire slot carries the host's UPPER BOUND, so
    // this driver would read a perfectly well-formed number and fold the whole
    // buffer instead of the prefix the device actually accepted. There is no
    // descriptor resolution here to substitute the real value.
    if (launch.rs_slot_flags.len != 0 &&
        std::any_of(launch.rs_slot_flags.ptr,
                    launch.rs_slot_flags.ptr + launch.rs_slot_flags.len,
                    [](std::uint8_t f) {
                        return (f & kRsFlagFoldLenDevice) != 0;
                    })) {
        throw std::runtime_error(
            "this driver cannot resolve a device-resident fold length; read "
            "the length back to the host and pass it as a constant");
    }
    // Likewise for a mid-page fold: this driver's buffer gather/scatter treat
    // logical buffer token 0 as physical offset 0, so a non-zero head would
    // read the tokens a fold already absorbed and overwrite the live ones.
    if (launch.rs_buffer_heads.len != 0 &&
        std::any_of(launch.rs_buffer_heads.ptr,
                    launch.rs_buffer_heads.ptr + launch.rs_buffer_heads.len,
                    [](std::uint32_t head) { return head != 0; })) {
        throw std::runtime_error(
            "this driver cannot address a buffer whose first live token is "
            "mid-page; fold whole pages only");
    }
    // A fold boundary landing strictly INSIDE a fire's own new tokens needs
    // the 2R-segment two-call shape: the row is cut at the boundary, the head
    // persists the state there and the tail continues from it without moving
    // it again. This driver issues one call per fire and would fold the whole
    // row, leaving the host believing tokens are still buffered that the
    // device has already absorbed -- a double fold on the next fire.
    // The per-row refusals below are SAFETY checks, so an unexpected shape
    // must fail rather than skip them: silently admitting a fire because its
    // arrays did not line up is exactly the outcome the checks exist to
    // prevent.
    if (launch.rs_fold_lens.len != 0 &&
        (launch.rs_slot_flags.len != launch.rs_fold_lens.len ||
         launch.qo_indptr.len != launch.rs_fold_lens.len + 1)) {
        throw std::runtime_error(
            "malformed RS launch: per-row fold lengths, slot flags and the "
            "token CSR must agree on the row count");
    }
    if (launch.rs_fold_lens.len != 0) {
        for (std::size_t r = 0; r < launch.rs_fold_lens.len; ++r) {
            if ((launch.rs_slot_flags.ptr[r] & kRsFlagBufferWrite) == 0) continue;
            const std::uint32_t n = launch.rs_fold_lens.ptr[r];
            // Reads are refused above, so the extended row IS the fire's own
            // tokens and the boundary is directly comparable to their count.
            const std::uint32_t rows =
                launch.qo_indptr.ptr[r + 1] - launch.qo_indptr.ptr[r];
            if (n != 0 && n < rows) {
                throw std::runtime_error(
                    "this driver cannot land a fold boundary inside a fire's "
                    "own tokens; fold the whole row or none of it");
            }
        }
    }
    // A row spanning NO TOKENS is how a guest says "compute nothing, only
    // move the recurrent boundary". This driver has no replay path at all
    // (the read refusal above), so such a row would run an empty forward and
    // move nothing, leaving the host believing a fold happened.
    if (launch.rs_fold_lens.len != 0) {
        for (std::size_t r = 0; r < launch.rs_fold_lens.len; ++r) {
            if (launch.qo_indptr.ptr[r + 1] == launch.qo_indptr.ptr[r]) {
                throw std::runtime_error(
                    "this driver cannot replay a buffered prefix, so a request "
                    "row carrying no tokens would fold nothing; fold in a fire "
                    "that computes");
            }
        }
    }
    // And likewise for a MIXED fire, where one row folds its recurrence while
    // another only buffers. The two shapes are the same dispatch and differ
    // only in whether the state persists, which CUDA expresses as a per-row
    // mask; this driver has only the pass-level flag, so it would fold every
    // row or none of them. Either way one row's state is wrong and the error
    // is unrecoverable once folded.
    if (launch.rs_slot_flags.len != 0 &&
        launch.rs_buffer_slot_indptr.len != launch.rs_slot_flags.len + 1) {
        throw std::runtime_error(
            "malformed RS launch: the buffer-slot CSR and the per-row slot "
            "flags must agree on the row count");
    }
    if (launch.rs_slot_flags.len != 0) {
        const auto persists = [&](std::size_t r) {
            const bool buffered =
                launch.rs_buffer_slot_indptr.ptr[r + 1] >
                launch.rs_buffer_slot_indptr.ptr[r];
            return !buffered ||
                (launch.rs_slot_flags.ptr[r] & kRsFlagFold) != 0;
        };
        for (std::size_t r = 1; r < launch.rs_slot_flags.len; ++r) {
            if (persists(r) != persists(0)) {
                throw std::runtime_error(
                    "this driver cannot fold one request's recurrent state while "
                    "another only buffers; split the fire");
            }
        }
    }
    view.rs_translation =
        pie::driver::slice_from_u32(
            launch.rs_translation.ptr,
            launch.rs_translation.len);
    view.rs_translation_indptr =
        pie::driver::slice_from_u32(
            launch.rs_translation_indptr.ptr,
            launch.rs_translation_indptr.len);
    view.sampling_indices =
        pie::driver::slice_from_u32(
            launch.sampling_indices.ptr,
            launch.sampling_indices.len);
    view.sampling_indptr =
        pie::driver::slice_from_u32(
            launch.sampling_indptr.ptr,
            launch.sampling_indptr.len);
    view.kv_translation =
        pie::driver::slice_from_u32(
            launch.kv_translation.ptr,
            launch.kv_translation.len);
    view.kv_translation_indptr =
        pie::driver::slice_from_u32(
            launch.kv_translation_indptr.ptr,
            launch.kv_translation_indptr.len);
    view.flattened_masks =
        pie::driver::slice_from_u32(
            launch.masks.words.ptr, launch.masks.words.len);
    view.mask_indptr =
        pie::driver::slice_from_u32(
            launch.masks.word_indptr.ptr,
            launch.masks.word_indptr.len);
    view.has_user_mask = launch.has_user_mask != 0;
    return view;
}

OwnedLaunchView OwnedLaunchView::capture(const pie::driver::fire::StepLaunch& launch) {
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
    owned.rs_translation =
        copy_slice(launch.rs_translation.ptr, launch.rs_translation.len);
    owned.rs_translation_indptr =
        copy_slice(launch.rs_translation_indptr.ptr, launch.rs_translation_indptr.len);
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

pie::driver::fire::LaunchView OwnedLaunchView::view() const {
    pie::driver::fire::LaunchView view{};
    view.token_ids = pie::driver::slice_from_u32(token_ids.data(), token_ids.size());
    view.position_ids =
        pie::driver::slice_from_u32(position_ids.data(), position_ids.size());
    view.kv_page_indices =
        pie::driver::slice_from_u32(kv_page_indices.data(), kv_page_indices.size());
    view.kv_page_indptr =
        pie::driver::slice_from_u32(kv_page_indptr.data(), kv_page_indptr.size());
    view.kv_last_page_lens =
        pie::driver::slice_from_u32(kv_last_page_lens.data(), kv_last_page_lens.size());
    view.qo_indptr = pie::driver::slice_from_u32(qo_indptr.data(), qo_indptr.size());
    view.rs_slot_ids =
        pie::driver::slice_from_u32(rs_slot_ids.data(), rs_slot_ids.size());
    view.rs_slot_flags =
        pie::driver::slice_from_u8(rs_slot_flags.data(), rs_slot_flags.size());
    view.rs_buffer_slot_ids = pie::driver::slice_from_u32(
        rs_buffer_slot_ids.data(), rs_buffer_slot_ids.size());
    view.rs_buffer_slot_indptr = pie::driver::slice_from_u32(
        rs_buffer_slot_indptr.data(), rs_buffer_slot_indptr.size());
    view.rs_translation =
        pie::driver::slice_from_u32(rs_translation.data(), rs_translation.size());
    view.rs_translation_indptr = pie::driver::slice_from_u32(
        rs_translation_indptr.data(), rs_translation_indptr.size());
    view.sampling_indices =
        pie::driver::slice_from_u32(sampling_indices.data(), sampling_indices.size());
    view.sampling_indptr =
        pie::driver::slice_from_u32(sampling_indptr.data(), sampling_indptr.size());
    view.kv_translation =
        pie::driver::slice_from_u32(kv_translation.data(), kv_translation.size());
    view.kv_translation_indptr = pie::driver::slice_from_u32(
        kv_translation_indptr.data(), kv_translation_indptr.size());
    view.flattened_masks =
        pie::driver::slice_from_u32(mask_words.data(), mask_words.size());
    view.mask_indptr = pie::driver::slice_from_u32(
        mask_word_indptr.data(), mask_word_indptr.size());
    view.has_user_mask = has_user_mask;
    return view;
}

bool build_member_forward_desc(
    const pie::driver::fire::LaunchView& view,
    std::size_t member,
    std::size_t member_count,
    bool has_linear_attn,
    std::uint32_t page_size,
    const pie::driver::fire::FireGeometry* resolved,
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
            // A wire fire that names KV pages is paged, exactly as a resolved
            // one is. The sealed M=1 ring path used to claim every wire fire,
            // so a prefill posted on the wire landed in the ring while the
            // decode that continues it -- device-resolved, therefore paged --
            // could not find its history. The two halves of one sequence have
            // to agree on where the KV lives.
            if (!desc.kv_pages.empty()) {
                desc.requires_paged = true;
                desc.qo_indptr = {
                    0u, static_cast<std::uint32_t>(desc.token_ids.size())};
                desc.kv_page_indptr = {
                    0u, static_cast<std::uint32_t>(desc.kv_pages.size())};
                desc.kv_last_page_lens = {desc.kv_last_page_len};
            }
        }
    }

    const std::size_t request_count =
        resolved != nullptr && resolved->qo_indptr.size() >= 2
            ? resolved->qo_indptr.size() - 1
            : 1;
    if (has_linear_attn) {
        if (resolved != nullptr) {
            // The launch's recurrent-state arrays are indexed one of two ways
            // and the difference only shows up once a batch carries more than
            // one member: either they are already scoped to this member's
            // requests, or they are launch-wide with one entry per member --
            // which is what the host branch below has always assumed. Reading
            // the launch-wide form from index 0 gave member 1 member 0's slot
            // and rejected any batch whose member count differed from one
            // member's request count, so two concurrent decodes could never
            // share a forward.
            const std::uint32_t* rs_ids = nullptr;
            const std::uint8_t* rs_flags = nullptr;
            if (view.rs_slot_ids.size() == request_count &&
                view.rs_slot_flags.size() == request_count) {
                rs_ids = view.rs_slot_ids.data();
                rs_flags = view.rs_slot_flags.data();
            } else if (request_count == 1 &&
                       view.rs_slot_ids.size() == member_count &&
                       view.rs_slot_flags.size() == member_count) {
                rs_ids = view.rs_slot_ids.data() + member;
                rs_flags = view.rs_slot_flags.data() + member;
            } else {
                error =
                    "resolved hybrid geometry requires exactly one folded "
                    "recurrent-state slot and flag per request";
                return false;
            }
            desc.request_rs_slot_ids.assign(rs_ids, rs_ids + request_count);
            for (std::size_t request = 0;
                 request < request_count;
                 ++request) {
                desc.request_rs_reset.push_back(
                    (rs_flags[request] & kRsFlagReset) != 0);
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
