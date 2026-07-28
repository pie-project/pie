#pragma once

#include <cstdint>
#include <limits>

#include "pie_native/launch/plan.hpp"

namespace pie_cuda_driver::pipeline {

inline std::uint32_t library_region_launch_node(
    const pie_native::launch::plan::Region& region) noexcept {
    return region.nodes.empty()
        ? std::numeric_limits<std::uint32_t>::max()
        : region.nodes.back();
}

}  // namespace pie_cuda_driver::pipeline
