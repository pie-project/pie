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

    // Composition-root initializer: parses `config_path`, loads the model,
    // runs the memory planner, allocates stores, constructs the batch engine
    // and pipeline registry, and bootstraps TP. Returns a `PIE_STATUS_*`
    // code; on any non-OK return the `Context` must not be used and should
    // be destroyed (a partially initialized `Context` never leaks: all
    // intermediate allocations are owned and torn down by the destructor).
    int initialize(const std::string& config_path, const PieRuntimeCallbacks& runtime);

    void fill_caps(PieDriverCaps* caps) const;
    int register_program(const PieProgramDesc& program, std::uint64_t* program_id);
    int register_channel(const PieChannelDesc& channel, PieChannelEndpointBinding* binding);
    int bind_instance(const PieInstanceDesc& instance, PieInstanceBinding* binding);
    int launch(const PieLaunchDesc& launch, PieCompletion completion);
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
