#pragma once

#include <cstdint>
#include <cstring>
#include <span>
#include <vector>

#include <ptir_abi.h>

#include "pie/driver/launch/program.hpp"

namespace pie::driver::fire::descriptor {

// Derived from the generated header, not retyped. `ptir_abi.h` is emitted from
// `pie_ir::registry`, which is the only place a port tag is decided; hand-copied
// copies of exactly this kind were how three wrong op tags reached the CUDA and
// Metal emitters (ptir-refactor.md §3.2), and nothing would have caught a wrong
// port tag here — a swapped pair silently fills the wrong geometry field.
inline constexpr std::uint8_t kPortEmbedTokens = PTIR_PORT_EMBED_TOKENS;
inline constexpr std::uint8_t kPortEmbedIndptr = PTIR_PORT_EMBED_INDPTR;
inline constexpr std::uint8_t kPortPositions = PTIR_PORT_POSITIONS;
inline constexpr std::uint8_t kPortPages = PTIR_PORT_PAGES;
inline constexpr std::uint8_t kPortPageIndptr = PTIR_PORT_PAGE_INDPTR;
inline constexpr std::uint8_t kPortKvLen = PTIR_PORT_KV_LEN;
inline constexpr std::uint8_t kPortWSlot = PTIR_PORT_W_SLOT;
inline constexpr std::uint8_t kPortWOff = PTIR_PORT_W_OFF;
inline constexpr std::uint8_t kPortReadout = PTIR_PORT_READOUT;
inline constexpr std::uint8_t kPortAttnMask = PTIR_PORT_ATTN_MASK;
// Recurrent-state buffered-slot family (tags 10-14). The device-resolved
// counterpart of the host-composed `rs_buffer_slot_ids` / `rs_buffer_slot_indptr`
// lowering; resolved by the same pre-forward port->field copier as the KV family.
inline constexpr std::uint8_t kPortRsBufferPages = PTIR_PORT_RS_BUFFER_PAGES;
inline constexpr std::uint8_t kPortRsBufferIndptr = PTIR_PORT_RS_BUFFER_INDPTR;
inline constexpr std::uint8_t kPortRsBufferLen = PTIR_PORT_RS_BUFFER_LEN;
inline constexpr std::uint8_t kPortRsWSlot = PTIR_PORT_RS_W_SLOT;
inline constexpr std::uint8_t kPortRsWOff = PTIR_PORT_RS_W_OFF;
// How far each request's folded boundary advances. The one RS port a guest
// still binds, and the only one whose value the HOST may not know: a
// speculative decode computes its accepted count on device, and this is the
// path that count takes to the recurrence without a host round-trip.
inline constexpr std::uint8_t kPortRsFoldLen = PTIR_PORT_RS_FOLD_LEN;

inline bool is_device_geometry_trace(const launch::Trace& trace) {
    bool has_write_desc = false;
    launch::ChannelId pages_channel = 0;
    bool has_pages = false;
    for (const launch::PortBinding& binding : trace.ports) {
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

inline bool stage_puts_channel(const launch::Trace& trace, launch::ChannelId channel) {
    for (const launch::Stage& stage : trace.stages) {
        for (const launch::ChannelPut& put : stage.puts) {
            if (put.channel == channel) return true;
        }
    }
    return false;
}

inline bool const_u32_port(
    const launch::PortBinding& binding,
    std::span<const std::uint32_t> expected = {}) {
    if (!binding.is_const ||
        binding.const_type.dtype != launch::DType::U32 ||
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
inline bool is_loop_carried_explicit_geometry_trace(const launch::Trace& trace) {
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
        for (const launch::PortBinding& binding : trace.ports) {
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
inline bool is_decode_envelope_trace(const launch::Trace& trace) {
    const launch::PortBinding* token = nullptr;
    const launch::PortBinding* kv_len = nullptr;
    const launch::PortBinding* positions = nullptr;
    const launch::PortBinding* embed_indptr = nullptr;
    const launch::PortBinding* readout = nullptr;
    const launch::PortBinding* pages = nullptr;
    const launch::PortBinding* page_indptr = nullptr;
    const launch::PortBinding* w_slot = nullptr;
    const launch::PortBinding* w_off = nullptr;
    for (const launch::PortBinding& binding : trace.ports) {
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
    if ((token_type.dtype != launch::DType::I32 &&
         token_type.dtype != launch::DType::U32) ||
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
        if (indptr_type.dtype != launch::DType::U32 ||
            indptr_type.shape.dims !=
                std::vector<std::uint32_t>{token_count + 1}) {
            return false;
        }
        lane_count = token_count;
    } else if (token_count != 1) {
        return false;
    }
    if (
        kv_len_type.dtype != launch::DType::U32 ||
        kv_len_type.shape.dims.size() != 1 ||
        kv_len_type.shape.dims[0] != lane_count) {
        return false;
    }
    if (positions != nullptr && !positions->is_const) {
        if (positions->channel >= trace.channels.size()) return false;
        const auto& position_type =
            trace.channels[positions->channel].type;
        if (position_type.dtype != launch::DType::U32 ||
            position_type.shape.dims.size() != 1 ||
            position_type.shape.dims[0] != token_count) {
            return false;
        }
    }
    // Executable geometry shapes — the compose kernels' dereference
    // assumptions — hold for EVERY envelope, independent of how positions
    // are sourced.
    {
        auto channel_type = [&](const launch::PortBinding* binding)
            -> const launch::TensorType* {
            return binding != nullptr && !binding->is_const &&
                    binding->channel < trace.channels.size()
                ? &trace.channels[binding->channel].type
                : nullptr;
        };
        const launch::TensorType* pages_type = channel_type(pages);
        const launch::TensorType* page_indptr_type = channel_type(page_indptr);
        const launch::TensorType* w_slot_type = channel_type(w_slot);
        const launch::TensorType* w_off_type = channel_type(w_off);
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
            pages_type->dtype != launch::DType::U32 ||
            !pages_shape_valid ||
            w_slot_type->dtype != launch::DType::U32 ||
            w_slot_type->shape.dims !=
                std::vector<std::uint32_t>{token_count} ||
            w_off_type->dtype != launch::DType::U32 ||
            w_off_type->shape.dims !=
                std::vector<std::uint32_t>{token_count}) {
            return false;
        }
        // page_indptr is either a trace-const CSR (wire template) or a
        // [lanes+1] u32 device channel.
        if (page_indptr_type != nullptr &&
            (page_indptr_type->dtype != launch::DType::U32 ||
             page_indptr_type->shape.dims !=
                 std::vector<std::uint32_t>{lane_count + 1})) {
            return false;
        }
    }
    // Const positions/read-out payloads travel on the wire; execution never
    // dereferences them here — no value checks (runtime classification owns
    // value semantics).
    for (const launch::Channel& channel : trace.channels) {
        if (channel.extern_dir >= 0) return false;
    }
    return true;
}

inline bool requires_descriptor_resolution(const launch::Trace& trace) {
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

}  // namespace pie::driver::fire::descriptor
