#pragma once

#include <span>

#include "pie_native/load_plan.hpp"

namespace pie_cuda_driver {

using LoadPlan = pie_load_planner::LoadPlan;

inline LoadPlan deserialize_load_plan(
    std::span<const std::uint8_t> bytes,
    std::uint64_t expected_compiler_version) {
    return LoadPlan::deserialize(bytes, expected_compiler_version);
}

}  // namespace pie_cuda_driver
