#pragma once

#include <cstdint>
#include <memory>
#include <string>

#include <pie_driver_abi.h>
#include <pie_native/step_launch.hpp>

namespace pie::metal {

class Context {
  public:
    Context();
    ~Context();

    Context(const Context&) = delete;
    Context& operator=(const Context&) = delete;

    int initialize(const std::string& config_path, const PieRuntimeCallbacks& runtime);
    void fill_device_facts(PieDriverCaps* caps) const;
    int load_model(const PieModelLoadDesc& load, PieDriverCaps* caps);
    int register_program(const PieProgramDesc& program, std::uint64_t* program_id);
    int register_channel(const PieChannelDesc& channel, PieChannelEndpointBinding* binding);
    int bind_instance(const PieInstanceDesc& instance, PieInstanceBinding* binding);
    int launch(const PieFrameDesc& frame, PieCompletion completion);
    int copy_kv(const PieKvCopyDesc& copy, PieCompletion completion);
    int copy_state(const PieStateCopyDesc& copy, PieCompletion completion);
    int resize_pool(const PiePoolResizeDesc& resize, PieCompletion completion);
    int close_instance(std::uint64_t instance_id);
    int close_channel(std::uint64_t channel_id);

  private:
    class Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pie::metal
