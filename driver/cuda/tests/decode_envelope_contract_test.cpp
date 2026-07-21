#include <cstdint>
#include <iostream>
#include <vector>

#include "pie_native/ptir/descriptor.hpp"

namespace ptir = pie_native::ptir;
namespace descriptor = pie_native::ptir::descriptor;

int main() {
    ptir::Trace trace;
    const auto add_channel = [&](ptir::DType dtype, ptir::Shape shape) {
        const auto id = static_cast<ptir::ChannelId>(trace.channels.size());
        trace.channels.push_back(ptir::Channel{
            .id = id,
            .type = ptir::TensorType{std::move(shape), dtype},
        });
        return id;
    };
    const auto bind = [&](std::uint8_t port, ptir::ChannelId channel) {
        trace.ports.push_back(ptir::PortBinding{
            .port = port,
            .channel = channel,
        });
    };

    bind(descriptor::kPortEmbedTokens,
         add_channel(ptir::DType::I32, ptir::Shape::vec(1)));
    bind(descriptor::kPortEmbedIndptr,
         add_channel(ptir::DType::U32, ptir::Shape::vec(2)));
    bind(descriptor::kPortPositions,
         add_channel(ptir::DType::U32, ptir::Shape::vec(1)));
    bind(descriptor::kPortPages,
         add_channel(ptir::DType::U32, ptir::Shape::vec(8)));
    bind(descriptor::kPortPageIndptr,
         add_channel(ptir::DType::U32, ptir::Shape::vec(2)));
    bind(descriptor::kPortKvLen,
         add_channel(ptir::DType::U32, ptir::Shape::vec(1)));
    bind(descriptor::kPortWSlot,
         add_channel(ptir::DType::U32, ptir::Shape::vec(1)));
    bind(descriptor::kPortWOff,
         add_channel(ptir::DType::U32, ptir::Shape::vec(1)));
    bind(descriptor::kPortReadout,
         add_channel(ptir::DType::U32, ptir::Shape::vec(1)));

    if (!descriptor::is_decode_envelope_trace(trace)) {
        std::cerr << "channelized single-lane decode envelope was rejected\n";
        return 1;
    }
    trace.channels[trace.ports[1].channel].type.shape =
        ptir::Shape::vec(3);
    if (descriptor::is_decode_envelope_trace(trace)) {
        std::cerr << "invalid EmbedIndptr shape was accepted\n";
        return 2;
    }
    return 0;
}
