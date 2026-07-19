#pragma once

#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include "pie_native/ptir/trace.hpp"

namespace pie_native::ptir::descriptor {

inline constexpr std::uint8_t kPortEmbedTokens = 0;
inline constexpr std::uint8_t kPortEmbedIndptr = 1;
inline constexpr std::uint8_t kPortPositions = 2;
inline constexpr std::uint8_t kPortPages = 3;
inline constexpr std::uint8_t kPortPageIndptr = 4;
inline constexpr std::uint8_t kPortKvLen = 5;
inline constexpr std::uint8_t kPortWSlot = 6;
inline constexpr std::uint8_t kPortWOff = 7;
inline constexpr std::uint8_t kPortReadout = 8;
inline constexpr std::uint8_t kPortAttnMask = 9;

inline bool is_device_geometry_trace(const Trace& trace) {
    bool has_write_desc = false;
    ChannelId pages_channel = 0;
    bool has_pages = false;
    for (const PortBinding& binding : trace.ports) {
        if (binding.is_const) continue;
        if (binding.port == kPortWSlot || binding.port == kPortWOff) {
            has_write_desc = true;
        } else if (binding.port == kPortPages) {
            pages_channel = binding.channel;
            has_pages = true;
        }
    }
    if (!has_write_desc ||
        !has_pages ||
        pages_channel >= trace.channels.size()) {
        return false;
    }
    const auto& dims = trace.channels[pages_channel].type.shape.dims;
    return dims.size() == 2 && dims[1] > 1;
}

inline bool stage_puts_channel(const Trace& trace, ChannelId channel) {
    for (const Stage& stage : trace.stages) {
        for (const ChannelPut& put : stage.puts) {
            if (put.channel == channel) return true;
        }
    }
    return false;
}

inline bool const_u32_port(
    const PortBinding& binding,
    std::span<const std::uint32_t> expected = {}) {
    if (!binding.is_const ||
        binding.const_type.dtype != DType::U32 ||
        binding.const_type.shape.dims.size() != 1) {
        return false;
    }
    if (expected.empty()) {
        return binding.const_type.shape.dims[0] * sizeof(std::uint32_t) ==
            binding.const_data.size();
    }
    if (binding.const_type.shape.dims[0] != expected.size() ||
        binding.const_data.size() != expected.size_bytes()) {
        return false;
    }
    for (std::size_t index = 0; index < expected.size(); ++index) {
        std::uint32_t actual = 0;
        std::memcpy(
            &actual,
            binding.const_data.data() +
                index * sizeof(std::uint32_t),
            sizeof(actual));
        if (actual != expected[index]) return false;
    }
    return true;
}

// A single-lane decode can own its complete explicit geometry without using
// the beam page-lease protocol. Once every required descriptor channel is
// re-published by the program, later fires must resolve the live/pending cells
// instead of replaying the host-side seed geometry.
inline bool is_loop_carried_explicit_geometry_trace(const Trace& trace) {
    constexpr std::uint8_t required[] = {
        kPortEmbedTokens,
        kPortPositions,
        kPortPages,
        kPortPageIndptr,
        kPortKvLen,
        kPortWSlot,
        kPortWOff,
    };
    for (const std::uint8_t port : required) {
        bool produced = false;
        for (const PortBinding& binding : trace.ports) {
            if (binding.port != port || binding.is_const) continue;
            produced = stage_puts_channel(trace, binding.channel);
            break;
        }
        if (!produced) return false;
    }
    return true;
}

// Bind-time verifier for the DecodeEnvelope class: EXECUTION invariants
// only — the channel bindings, shapes, and dtypes the envelope compose
// kernels dereference. Classification (derivability, seededness, value
// semantics) is the runtime's job, done once; the driver only checks that
// it can run the claimed class safely. The golden trace parity corpus pins
// the two sides together.
inline bool is_decode_envelope_trace(const Trace& trace) {
    const PortBinding* token = nullptr;
    const PortBinding* kv_len = nullptr;
    const PortBinding* positions = nullptr;
    const PortBinding* embed_indptr = nullptr;
    const PortBinding* readout = nullptr;
    const PortBinding* pages = nullptr;
    const PortBinding* page_indptr = nullptr;
    const PortBinding* w_slot = nullptr;
    const PortBinding* w_off = nullptr;
    for (const PortBinding& binding : trace.ports) {
        switch (binding.port) {
            case kPortEmbedTokens:
                if (binding.is_const || token != nullptr) return false;
                token = &binding;
                break;
            case kPortKvLen:
                if (binding.is_const || kv_len != nullptr) return false;
                kv_len = &binding;
                break;
            case kPortEmbedIndptr:
                if (embed_indptr != nullptr ||
                    (binding.is_const && !const_u32_port(binding))) {
                    return false;
                }
                embed_indptr = &binding;
                break;
            case kPortPositions:
                if (positions != nullptr) return false;
                if (binding.is_const) {
                    if (!const_u32_port(binding)) {
                        return false;
                    }
                }
                positions = &binding;
                break;
            case kPortReadout:
                // Read-out shapes sampling on the RUNTIME side; execution
                // never dereferences the port. Only reject duplicates.
                if (readout != nullptr) return false;
                readout = &binding;
                break;
            case kPortPages:
                if (binding.is_const || pages != nullptr) return false;
                pages = &binding;
                break;
            case kPortPageIndptr:
                // A trace-const lane CSR executes from the wire template;
                // channel-fed CSRs resolve device cells.
                if (page_indptr != nullptr ||
                    (binding.is_const && !const_u32_port(binding))) {
                    return false;
                }
                page_indptr = &binding;
                break;
            case kPortWSlot:
                if (binding.is_const || w_slot != nullptr) return false;
                w_slot = &binding;
                break;
            case kPortWOff:
                if (binding.is_const || w_off != nullptr) return false;
                w_off = &binding;
                break;
            default:
                return false;
        }
    }
    if (token == nullptr || kv_len == nullptr ||
        positions == nullptr || pages == nullptr ||
        page_indptr == nullptr || w_slot == nullptr || w_off == nullptr ||
        token->channel >= trace.channels.size() ||
        kv_len->channel >= trace.channels.size()) {
        return false;
    }
    const auto& token_type = trace.channels[token->channel].type;
    const auto& kv_len_type = trace.channels[kv_len->channel].type;
    if ((token_type.dtype != DType::I32 &&
         token_type.dtype != DType::U32) ||
        token_type.shape.dims.size() != 1 ||
        token_type.shape.dims[0] == 0) {
        return false;
    }
    const std::uint32_t token_count = token_type.shape.dims[0];
    std::uint32_t lane_count = 1;
    if (embed_indptr != nullptr && embed_indptr->is_const) {
        const std::size_t count =
            embed_indptr->const_data.size() / sizeof(std::uint32_t);
        if (count < 2 ||
            embed_indptr->const_type.shape.dims.size() != 1 ||
            embed_indptr->const_type.shape.dims[0] != count) {
            return false;
        }
        std::uint32_t prior = 0;
        for (std::size_t index = 0; index < count; ++index) {
            std::uint32_t value = 0;
            std::memcpy(
                &value,
                embed_indptr->const_data.data() +
                    index * sizeof(std::uint32_t),
                sizeof(value));
            if ((index == 0 && value != 0) ||
                (index != 0 && value != prior + 1)) {
                return false;
            }
            prior = value;
        }
        if (prior != token_count) return false;
        lane_count = static_cast<std::uint32_t>(count - 1);
    } else if (embed_indptr != nullptr) {
        if (embed_indptr->channel >= trace.channels.size()) return false;
        const auto& indptr_type =
            trace.channels[embed_indptr->channel].type;
        if (indptr_type.dtype != DType::U32 ||
            indptr_type.shape.dims !=
                std::vector<std::uint32_t>{token_count + 1}) {
            return false;
        }
        lane_count = token_count;
    } else if (token_count != 1) {
        return false;
    }
    if (
        kv_len_type.dtype != DType::U32 ||
        kv_len_type.shape.dims.size() != 1 ||
        kv_len_type.shape.dims[0] != lane_count) {
        return false;
    }
    if (positions != nullptr && !positions->is_const) {
        if (positions->channel >= trace.channels.size()) return false;
        const auto& position_type =
            trace.channels[positions->channel].type;
        if (position_type.dtype != DType::U32 ||
            position_type.shape.dims.size() != 1 ||
            position_type.shape.dims[0] != token_count) {
            return false;
        }
    }
    // Executable geometry shapes — the compose kernels' dereference
    // assumptions — hold for EVERY envelope, independent of how positions
    // are sourced.
    {
        auto channel_type = [&](const PortBinding* binding)
            -> const TensorType* {
            return binding != nullptr && !binding->is_const &&
                    binding->channel < trace.channels.size()
                ? &trace.channels[binding->channel].type
                : nullptr;
        };
        const TensorType* pages_type = channel_type(pages);
        const TensorType* page_indptr_type = channel_type(page_indptr);
        const TensorType* w_slot_type = channel_type(w_slot);
        const TensorType* w_off_type = channel_type(w_off);
        const bool pages_shape_valid =
            pages_type != nullptr &&
            ((lane_count == 1 &&
              pages_type->shape.dims.size() == 1 &&
              pages_type->shape.dims[0] > 0) ||
             (pages_type->shape.dims.size() == 2 &&
              pages_type->shape.dims[0] == lane_count &&
              pages_type->shape.dims[1] > 0));
        if (pages_type == nullptr ||
            w_slot_type == nullptr || w_off_type == nullptr ||
            pages_type->dtype != DType::U32 ||
            !pages_shape_valid ||
            w_slot_type->dtype != DType::U32 ||
            w_slot_type->shape.dims !=
                std::vector<std::uint32_t>{token_count} ||
            w_off_type->dtype != DType::U32 ||
            w_off_type->shape.dims !=
                std::vector<std::uint32_t>{token_count}) {
            return false;
        }
        // page_indptr is either a trace-const CSR (wire template) or a
        // [lanes+1] u32 device channel.
        if (page_indptr_type != nullptr &&
            (page_indptr_type->dtype != DType::U32 ||
             page_indptr_type->shape.dims !=
                 std::vector<std::uint32_t>{lane_count + 1})) {
            return false;
        }
    }
    // Const positions/read-out payloads travel on the wire; execution never
    // dereferences them here — no value checks (runtime classification owns
    // value semantics).
    for (const Channel& channel : trace.channels) {
        if (channel.extern_dir >= 0) return false;
    }
    return true;
}

inline bool requires_descriptor_resolution(const Trace& trace) {
    return is_device_geometry_trace(trace) ||
           is_loop_carried_explicit_geometry_trace(trace);
}

inline std::uint32_t last_page_len(
    std::uint32_t length,
    std::uint32_t page_size) {
    return length == 0 || page_size == 0
               ? 0
               : ((length - 1) % page_size) + 1;
}

}  // namespace pie_native::ptir::descriptor
