#pragma once

#include <bit>
#include <cstddef>
#include <cstdint>

#include "pie_native/ptir/plan.hpp"

namespace pie_cuda_driver::pipeline {

inline std::uint8_t program_extent_bucket(std::uint32_t value) noexcept {
    value = value == 0 ? 1 : value;
    return static_cast<std::uint8_t>(std::bit_width(value - 1));
}

inline std::uint64_t compiled_stage_identity(
    const pie_native::ptir::plan::StagePlan& stage) noexcept {
    std::uint64_t hash = 0xcbf29ce484222325ULL;
    const auto add_byte = [&](std::uint8_t byte) {
        hash ^= byte;
        hash *= 0x100000001b3ULL;
    };
    const auto add_u32 = [&](std::uint32_t value) {
        for (std::uint32_t shift = 0; shift < 32; shift += 8) {
            add_byte(static_cast<std::uint8_t>(value >> shift));
        }
    };
    add_byte(stage.stage);
    add_u32(static_cast<std::uint32_t>(stage.signature.size()));
    for (const std::uint8_t byte : stage.signature) add_byte(byte);
    add_byte(stage.fused.kind);
    add_byte(stage.fused.whole_stage_fallback ? 1u : 0u);
    add_u32(static_cast<std::uint32_t>(stage.fused.regions.size()));
    for (const auto& region : stage.fused.regions) {
        add_byte(region.schedule);
        add_byte(region.library ? 1u : 0u);
        add_byte(region.library_op);
        add_u32(static_cast<std::uint32_t>(region.nodes.size()));
        for (const std::uint32_t node : region.nodes) add_u32(node);
        add_u32(static_cast<std::uint32_t>(region.inputs.size()));
        for (const std::uint32_t input : region.inputs) add_u32(input);
        add_u32(static_cast<std::uint32_t>(region.outputs.size()));
        for (const std::uint32_t output : region.outputs) add_u32(output);
        add_u32(static_cast<std::uint32_t>(region.sinks.size()));
        for (const auto& sink : region.sinks) {
            add_u32(sink.channel_slot);
            add_u32(sink.value);
        }
    }
    return hash == 0 ? 1 : hash;
}

inline std::uint64_t avalanche_program_identity(std::uint64_t value) noexcept {
    value ^= value >> 30;
    value *= 0xbf58476d1ce4e5b9ULL;
    value ^= value >> 27;
    value *= 0x94d049bb133111ebULL;
    return value ^ (value >> 31);
}

class ProgramSetIdentityFold {
  public:
    void add(std::uint64_t stage_identity, std::uint8_t row_bucket) noexcept {
        const std::uint64_t item = avalanche_program_identity(
            stage_identity ^
            (static_cast<std::uint64_t>(row_bucket) *
             0x9e3779b97f4a7c15ULL));
        sum_ += item;
        squares_ += item * item;
        product_ *= item | 1u;
        xor_ ^= item;
        ++count_;
    }

    std::uint64_t finish() const noexcept {
        if (count_ == 0) return 0;
        std::uint64_t hash = avalanche_program_identity(
            sum_ ^ std::rotl(squares_, 17) ^
            std::rotl(product_, 31) ^ std::rotl(xor_, 47) ^
            (static_cast<std::uint64_t>(count_) *
             0xd6e8feb86659fd93ULL));
        return hash == 0 ? 1 : hash;
    }

    std::uint32_t count() const noexcept { return count_; }

  private:
    std::uint64_t sum_ = 0;
    std::uint64_t squares_ = 0;
    std::uint64_t product_ = 1;
    std::uint64_t xor_ = 0;
    std::uint32_t count_ = 0;
};

}  // namespace pie_cuda_driver::pipeline
