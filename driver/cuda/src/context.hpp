#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <pie_driver_abi.h>

namespace pie::cuda {

// Composition root for the CUDA driver: owns device state, the loaded model,
// stores (KV/MLA/DSA/swap), the pipeline registry, the batch engine
// (BatchEngine), TP comm + follower thread, async completion bookkeeping, and
// reported capabilities. `abi.cpp` maps the opaque `PieDriver*` handle to a
// `Context` and validates ABI-level arguments; `Context` owns everything
// else. See `cpp-refact.md` Phase 3.
class Context {
  public:
    Context();
    ~Context();

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    // Create-time initialization: parse device/group config, select the CUDA
    // context, and initialize NCCL. No checkpoint metadata or payload is read.
    int initialize(const std::string& config_path, const PieRuntimeCallbacks& runtime);

    void fill_device_facts(PieDriverCaps* caps) const;
    int load_model(const PieModelLoadDesc& load, PieDriverCaps* caps);
    int register_program(const PieProgramDesc& program, std::uint64_t* program_id);
    int register_channel(const PieChannelDesc& channel, PieChannelEndpointBinding* binding);
    int bind_instance(const PieInstanceDesc& instance, PieInstanceBinding* binding);
    int launch(const PieLaunchDesc& launch, PieCompletion completion);
    int prepare_launch(
        const PieLaunchDesc& launch,
        PieLaunchPrepareResult* result);
    int launch_prepared(
        const PieLaunchDesc& launch,
        std::uint64_t lease_id,
        PieCompletion completion);
    int release_launch(std::uint64_t lease_id);
    int flush_settlement();
    int encode(const PieEncodeDesc& encode, PieCompletion completion);
    int copy_kv(const PieKvCopyDesc& copy, PieCompletion completion);
    int copy_state(const PieStateCopyDesc& copy, PieCompletion completion);
    int resize_pool(const PiePoolResizeDesc& resize, PieCompletion completion);
    int close_instance(std::uint64_t instance_id);
    int close_channel(std::uint64_t channel_id);

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pie::cuda
