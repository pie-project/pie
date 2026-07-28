#include <cstdint>
#include <iostream>
#include <vector>

#include "pie_native/fire/descriptor.hpp"

namespace ptir = pie_native::launch;
namespace descriptor = pie_native::launch::descriptor;

int main() {
    launch::Trace trace;
    const auto add_channel = [&](launch::DType dtype, launch::Shape shape) {
        const auto id = static_cast<launch::ChannelId>(trace.channels.size());
        trace.channels.push_back(launch::Channel{
            .id = id,
            .type = launch::TensorType{std::move(shape), dtype},
        });
        return id;
    };
    const auto bind = [&](std::uint8_t port, launch::ChannelId channel) {
        trace.ports.push_back(launch::PortBinding{
            .port = port,
            .channel = channel,
        });
    };

    bind(descriptor::kPortEmbedTokens,
         add_channel(launch::DType::I32, launch::Shape::vec(1)));
    bind(descriptor::kPortEmbedIndptr,
         add_channel(launch::DType::U32, launch::Shape::vec(2)));
    bind(descriptor::kPortPositions,
         add_channel(launch::DType::U32, launch::Shape::vec(1)));
    bind(descriptor::kPortPages,
         add_channel(launch::DType::U32, launch::Shape::vec(8)));
    bind(descriptor::kPortPageIndptr,
         add_channel(launch::DType::U32, launch::Shape::vec(2)));
    bind(descriptor::kPortKvLen,
         add_channel(launch::DType::U32, launch::Shape::vec(1)));
    bind(descriptor::kPortWSlot,
         add_channel(launch::DType::U32, launch::Shape::vec(1)));
    bind(descriptor::kPortWOff,
         add_channel(launch::DType::U32, launch::Shape::vec(1)));
    bind(descriptor::kPortReadout,
         add_channel(launch::DType::U32, launch::Shape::vec(1)));

    if (!descriptor::is_decode_envelope_trace(trace)) {
        std::cerr << "channelized single-lane decode envelope was rejected\n";
        return 1;
    }
    trace.channels[trace.ports[1].channel].type.shape =
        launch::Shape::vec(3);
    if (descriptor::is_decode_envelope_trace(trace)) {
        std::cerr << "invalid EmbedIndptr shape was accepted\n";
        return 2;
    }
    return 0;
}
