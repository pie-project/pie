#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "pie_native/ptir/plan.hpp"

namespace pie::metal::pipeline {

inline constexpr std::uint16_t kMetalM1EmitterVersion = 23;
inline constexpr std::size_t kMetalM1MaxChannels = 29;
inline constexpr std::size_t kMetalM2MaxFusedChannels = 12;

struct M1ChannelEffect {
    bool requires_full = false;
    bool requires_empty = false;
    bool take = false;
    bool put = false;
    std::uint32_t capacity = 1;
};

struct M1OpMeta {
    std::uint32_t node = 0;
    std::uint32_t result_base = 0;
    pie_native::ptir::container::COp op;
};

bool validate_singleton_plan(
    const pie_native::ptir::plan::StagePlan& plan,
    std::vector<M1OpMeta>& operations,
    std::string& error);

std::string emit_singleton_region_msl(
    const std::string& runtime_template,
    const std::string& function_name,
    std::uint8_t op_tag);

std::string emit_readiness_msl(
    const std::string& function_name,
    const std::vector<M1ChannelEffect>& channels);

std::string emit_commit_msl(
    const std::string& function_name,
    const std::vector<M1ChannelEffect>& channels);

bool emit_fused_region_msl(
    const std::string& runtime_template,
    const std::string& function_name,
    const pie_native::ptir::plan::StagePlan& stage,
    const pie_native::ptir::plan::Region& region,
    std::string& source,
    std::string& error);

bool emit_grouped_fused_region_msl(
    const std::string& runtime_template,
    const std::string& function_name,
    const pie_native::ptir::plan::StagePlan& stage,
    const pie_native::ptir::plan::Region& region,
    std::string& source,
    std::string& error);

bool emit_grouped_nucleus_msl(
    const std::string& runtime_template,
    const std::string& function_name,
    const pie_native::ptir::plan::StagePlan& stage,
    const pie_native::ptir::plan::Region& region,
    std::string& source,
    std::string& error);

bool emit_grouped_topk_msl(
    const std::string& runtime_template,
    const std::string& function_name,
    const pie_native::ptir::plan::StagePlan& stage,
    const pie_native::ptir::plan::Region& region,
    std::string& source,
    std::string& error);

std::string emit_grouped_readiness_msl(
    const std::string& function_name);

std::string emit_grouped_commit_msl(
    const std::string& function_name);

}  // namespace pie::metal::pipeline
