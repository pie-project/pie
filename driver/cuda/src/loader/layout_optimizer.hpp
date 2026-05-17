#pragma once

#include <string>
#include <vector>

#include "loader/layout_plan.hpp"

namespace pie_cuda_driver {

struct LayoutOptimizerPassStats {
    std::string name;
    std::size_t exprs_before = 0;
    std::size_t exprs_after = 0;
    std::size_t rewrites = 0;
};

struct LayoutOptimizerResult {
    std::vector<LayoutOptimizerPassStats> passes;
};

LayoutOptimizerResult optimize_layout_algebra(LayoutPlan& plan);

}  // namespace pie_cuda_driver
