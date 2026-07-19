#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <pie_driver_abi.h>

#include "batch/forward.hpp"
#include "pie_native/launch_view.hpp"
#include "pie_native/ptir/fire_geometry.hpp"

namespace pie::metal::batch {

inline constexpr std::uint64_t kNoChannelTicket = ~std::uint64_t{0};

struct ChannelTicket {
    std::uint64_t channel_id = 0;
    std::size_t dense = 0;
    std::uint64_t expected_head = kNoChannelTicket;
    std::uint64_t expected_tail = kNoChannelTicket;
    bool requires_input = false;
};

struct LaunchMember {
    std::uint64_t instance_id = 0;
    std::vector<ChannelTicket> tickets;
    bool needs_forward = false;
    bool requires_m2 = false;
    int mtp_draft_row = -1;
    int fwd_slot = -1;
    std::string build_err;
    PieTerminalCell* terminal_cell = nullptr;
};

struct OwnedLaunchView {
    std::vector<std::uint32_t> token_ids;
    std::vector<std::uint32_t> position_ids;
    std::vector<std::uint32_t> kv_page_indices;
    std::vector<std::uint32_t> kv_page_indptr;
    std::vector<std::uint32_t> kv_last_page_lens;
    std::vector<std::uint32_t> qo_indptr;
    std::vector<std::uint32_t> rs_slot_ids;
    std::vector<std::uint8_t> rs_slot_flags;
    // Preserved as activation-suffix metadata only; never a folded-state
    // fallback for resolved request rows.
    std::vector<std::uint32_t> rs_buffer_slot_ids;
    std::vector<std::uint32_t> rs_buffer_slot_indptr;
    std::vector<std::uint32_t> sampling_indices;
    std::vector<std::uint32_t> sampling_indptr;
    std::vector<std::uint32_t> kv_translation;
    std::vector<std::uint32_t> kv_translation_indptr;
    std::vector<std::uint32_t> mask_request_indptr;
    std::vector<std::uint32_t> mask_word_indptr;
    std::vector<std::uint32_t> mask_words;
    std::uint32_t required_kv_pages = 0;
    bool has_user_mask = false;

    static OwnedLaunchView capture(const PieLaunchDesc& launch);
    pie_native::LaunchView view() const;
};

struct LaunchJobData {
    std::vector<LaunchMember> members;
    OwnedLaunchView launch;
    std::vector<MemberForwardDesc> fwd_descs;
    PieCompletion completion{};
    std::uint64_t lease_id = 0;
};

pie_native::LaunchView build_launch_view(const PieLaunchDesc& launch);

bool build_member_forward_desc(
    const pie_native::LaunchView& view,
    std::size_t member,
    std::size_t member_count,
    bool has_linear_attn,
    std::uint32_t page_size,
    const pie_native::ptir::FireGeometry* resolved,
    MemberForwardDesc& desc,
    std::string& error);

}  // namespace pie::metal::batch
