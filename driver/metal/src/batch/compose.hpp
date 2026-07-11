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

struct WriterConsume {
    std::uint64_t channel_id = 0;
    std::size_t dense = 0;
    std::uint64_t target_head = 0;
};

struct LaunchMember {
    std::uint64_t instance_id = 0;
    std::vector<WriterConsume> consumes;
    bool needs_forward = false;
    int fwd_slot = -1;
    std::string build_err;
    PieTerminalCell* terminal_cell = nullptr;
};

struct LaunchJobData {
    std::vector<LaunchMember> members;
    std::vector<MemberForwardDesc> fwd_descs;
    PieCompletion completion{};
};

pie_native::LaunchView build_launch_view(const PieLaunchDesc& launch);

bool build_member_forward_desc(
    const pie_native::LaunchView& view,
    std::size_t member,
    std::size_t member_count,
    bool has_linear_attn,
    const pie_native::ptir::FireGeometry* resolved,
    MemberForwardDesc& desc,
    std::string& error);

}  // namespace pie::metal::batch
