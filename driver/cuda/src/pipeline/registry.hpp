#pragma once

// pipeline/: guest programs -- PTIR dispatch, tier0 execution, frame carrier.
//
// `pipeline::Registry`: the single owner of PTIR program records, instance
// bindings, and the `Dispatch` object itself (constructed exactly once, here).
// `Context` validates ABI handles and resource limits, then
// delegates every program/instance/channel operation to this class — no
// shadow copies of program bytes or instance maps live at the entry layer.

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <vector>

#include <pie_driver_abi.h>

#include "pie_native/launch_view.hpp"
#include "pipeline/dispatch.hpp"

namespace pie_cuda_driver::pipeline {

struct ProgramRecord {
    std::uint64_t program_id = 0;
    std::uint64_t program_hash = 0;
};

struct InstanceRecord {
    std::uint64_t instance_id = 0;
    std::uint64_t program_id = 0;
    std::uint64_t program_hash = 0;
    // The runtime's ACK'd classification (classify once): launch validation
    // keys its deferred-geometry exemption on this, per instance — never on
    // a batch-level shape sniff (RV-26/C2).
    std::uint32_t geometry_class = PIE_GEOMETRY_CLASS_HOST;
};

class Registry {
  public:
    Registry() = default;
    Registry(const Registry&) = delete;
    Registry& operator=(const Registry&) = delete;

    // Dedups by `program.program_hash`: if already registered, `*program_id`
    // is set to the existing id and PIE_STATUS_OK is returned without
    // touching the dispatcher. Otherwise validates + registers the program
    // bytes with `dispatch_` and allocates a fresh program id.
    int register_program(const PieProgramDesc& program,
                         std::uint64_t* program_id,
                         std::string* err);

    int register_channel(const PieChannelDesc& channel,
                         PieChannelEndpointBinding* binding,
                         std::string* err);

    // Looks up `instance.program_id`, allocates (or takes the requested)
    // instance id, binds it with `dispatch_`, and records the
    // instance -> program mapping. Returns PIE_STATUS_INVALID_ARGUMENT if
    // `instance.program_id` is unknown.
    int bind_instance(const PieInstanceDesc& instance,
                      PieInstanceBinding* binding,
                      std::string* err);

    // Resolves each wire instance id to its bound program record, in order.
    // Returns PIE_STATUS_INVALID_ARGUMENT on the first unknown id (leaving
    // `out` partially filled).
    int resolve_instances(const PieU64Slice& instance_ids,
                         std::vector<InstanceRecord>* out) const;

    int validate_launch(const pie_native::LaunchView& view, std::string* err) {
        return dispatch_.validate_launch(view, err);
    }

    // Returns PIE_STATUS_CLOSED if `instance_id` is unknown (already closed
    // or never bound), else closes it with `dispatch_` and returns
    // PIE_STATUS_OK.
    int close_instance(std::uint64_t instance_id);
    int close_channel(std::uint64_t channel_id, std::string* err);

    // The single Dispatch instance, for the actual fire (`run`,
    // `settle_failed_launch`, `resolve_descriptors`) and for handing a
    // non-owning pointer to the batch executor.
    Dispatch& dispatch() { return dispatch_; }
    const Dispatch& dispatch() const { return dispatch_; }

  private:
    Dispatch dispatch_;
    std::uint64_t next_program_id_ = 1;
    std::uint64_t next_instance_id_ = 1;
    std::unordered_map<std::uint64_t, ProgramRecord> programs_;
    std::unordered_map<std::uint64_t, std::uint64_t> program_ids_by_hash_;
    std::unordered_map<std::uint64_t, InstanceRecord> instances_;
};

}  // namespace pie_cuda_driver::pipeline
