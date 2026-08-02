#pragma once

#include <cstddef>
#include <cstdint>
#include <memory>
#include <string>
#include <unordered_map>
#include <vector>

#include <pie_driver_abi.h>

#include "pipeline/interp.hpp"

namespace pie::metal::pipeline {

struct M1ProgramExecutable;

// One host-emitted MSL kernel, or the reason the host chose not to emit
// this one. Owns its byte strings so it outlives the caller's
// `PieEmittedKernel` buffer, which is only guaranteed valid for the
// register-program call.
struct HostEmittedKernel {
    std::uint32_t kind = 0;
    std::uint32_t stage_index = 0;
    std::uint32_t region_index = 0;
    std::string entry_name;
    std::string source;
    std::string error;
};

struct ProgramRecord {
    std::uint64_t program_id = 0;
    std::uint64_t program_hash = 0;
    std::vector<pie::driver::launch::Channel> channels;
    ExecPlan plan;
    std::vector<HostEmittedKernel> emitted_kernels;
    std::shared_ptr<M1ProgramExecutable> m1_executable;
    std::string m1_error;
};

struct InstanceRecord {
    std::uint64_t instance_id = 0;
    std::uint64_t program_id = 0;
    std::uint64_t program_hash = 0;
    /// What the runtime classified this instance as. The driver used to infer
    /// "does this need descriptor resolution?" from the trace's port shapes,
    /// which predates the class travelling on the wire and disagrees with it:
    /// a decode envelope that leaves `pages` constant produces six of the seven
    /// ports `is_loop_carried_explicit_geometry_trace` demands, so the driver
    /// silently used the wire's placeholder geometry.
    std::uint32_t geometry_class = PIE_GEOMETRY_CLASS_HOST;
    std::vector<std::uint64_t> channel_ids;
    std::uint64_t fire_seq = 0;
    InterpInstance interp;
};

struct ChannelRecord {
    PieChannelDesc desc{};
    std::vector<std::uint32_t> shape;
    std::string extern_name;
    std::unordered_map<std::uint64_t, std::uint8_t> attachments;
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
