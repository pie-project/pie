#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <pie_driver_abi.h>

#include "pie_native/ptir_channels.hpp"
#include "pipeline/interp.hpp"

namespace pie::metal::pipeline {

struct ProgramRecord {
    std::uint64_t program_id = 0;
    std::uint64_t program_hash = 0;
    std::vector<pie_native::PtirChannelDecl> channels;
    ExecPlan plan;
};

struct InstanceRecord {
    std::uint64_t instance_id = 0;
    std::uint64_t program_id = 0;
    std::uint64_t program_hash = 0;
    std::vector<std::uint64_t> channel_ids;
    std::uint64_t fire_seq = 0;
    InterpInstance interp;
};

struct ChannelRecord {
    PieChannelDesc desc{};
    std::vector<std::uint32_t> shape;
    std::string extern_name;
    std::vector<std::uint8_t> mirror;
    std::vector<std::uint64_t> words;
    std::unordered_map<std::uint64_t, std::uint8_t> attachments;
    std::uint64_t pulled = 0;
    std::uint64_t reserved_head = 0;
    bool seed_credit = false;
    std::shared_ptr<ChannelState> shared_state;

    std::size_t numel() const;
    DType program_dtype() const;
};

class Registry {
  public:
    int register_program(const PieProgramDesc& program, std::uint64_t* program_id);
    int register_channel(
        const PieChannelDesc& channel,
        PieChannelEndpointBinding* binding);
    int bind_instance(
        const PieInstanceDesc& instance,
        PieInstanceBinding* binding);

    ProgramRecord* find_program(std::uint64_t program_id);
    const ProgramRecord* find_program(std::uint64_t program_id) const;
    InstanceRecord* find_instance(std::uint64_t instance_id);
    const InstanceRecord* find_instance(std::uint64_t instance_id) const;
    ChannelRecord* find_channel(std::uint64_t channel_id);
    const ChannelRecord* find_channel(std::uint64_t channel_id) const;

    int close_instance(std::uint64_t instance_id);
    int close_channel(std::uint64_t channel_id);

  private:
    std::uint64_t next_program_id_ = 1;
    std::uint64_t next_instance_id_ = 1;
    std::unordered_map<std::uint64_t, ProgramRecord> programs_;
    std::unordered_map<std::uint64_t, std::uint64_t> program_ids_by_hash_;
    std::unordered_map<std::uint64_t, InstanceRecord> instances_;
    std::unordered_map<std::uint64_t, ChannelRecord> channels_;
};

}  // namespace pie::metal::pipeline
