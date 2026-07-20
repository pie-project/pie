#include "pipeline/registry.hpp"

#include <algorithm>
#include <cstring>
#include <iostream>
#include <unordered_set>
#include <utility>

namespace pie::metal::pipeline {

std::size_t ChannelRecord::numel() const {
    std::size_t n = 1;
    for (const std::uint32_t dim : shape) n *= dim;
    return n;
}

DType ChannelRecord::program_dtype() const {
    switch (desc.dtype) {
        case PIE_CHANNEL_DTYPE_I32: return DType::I32;
        case PIE_CHANNEL_DTYPE_U32: return DType::U32;
        case PIE_CHANNEL_DTYPE_BOOL: return DType::Bool;
        default: return DType::F32;
    }
}

int Registry::register_program(
    const PieProgramDesc& program,
    std::uint64_t* program_id) {
    const auto found = program_ids_by_hash_.find(program.program_hash);
    if (found != program_ids_by_hash_.end()) {
        const ProgramRecord& existing = programs_.at(found->second);
        if (program.canonical_bytes.len != 0 &&
            (program.canonical_bytes.len != existing.canonical_bytes.size() ||
             std::memcmp(
                 program.canonical_bytes.ptr,
                 existing.canonical_bytes.data(),
                 existing.canonical_bytes.size()) != 0)) {
            std::cerr
                << "[pie-driver-metal] register_program: program hash collision\n";
            return PIE_STATUS_INVALID_ARGUMENT;
        }
        if (program_id != nullptr) *program_id = found->second;
        return PIE_STATUS_OK;
    }
    if (program.canonical_bytes.len == 0) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }

    ProgramRecord record;
    record.program_id = next_program_id_++;
    record.program_hash = program.program_hash;
    record.canonical_bytes.assign(
        program.canonical_bytes.ptr,
        program.canonical_bytes.ptr + program.canonical_bytes.len);
    std::string decode_error;
    if (!pie_native::decode_ptir_channels(
            program.canonical_bytes.ptr,
            program.canonical_bytes.len,
            record.channels,
            &decode_error)) {
        std::cerr << "[pie-driver-metal] register_program: "
                  << decode_error << "\n";
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    if (!build_exec_plan(
            program.canonical_bytes.ptr,
            program.canonical_bytes.len,
            program.sidecar_bytes.ptr,
            program.sidecar_bytes.len,
            record.plan,
            &decode_error)) {
        std::cerr << "[pie-driver-metal] register_program: "
                  << decode_error << "\n";
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    program_ids_by_hash_[record.program_hash] = record.program_id;
    if (program_id != nullptr) *program_id = record.program_id;
    programs_.emplace(record.program_id, std::move(record));
    return PIE_STATUS_OK;
}

int Registry::register_channel(
    const PieChannelDesc& channel,
    PieChannelEndpointBinding* binding) {
    if (channels_.find(channel.channel_id) != channels_.end()) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    pie_native::PtirChannelDecl geometry;
    geometry.dtype = channel.dtype;
    geometry.dims.assign(channel.shape.ptr, channel.shape.ptr + channel.shape.len);
    const std::uint32_t cell_bytes = geometry.cell_bytes();
    if (cell_bytes == 0) return PIE_STATUS_INVALID_ARGUMENT;

    ChannelRecord record;
    record.desc = channel;
    record.shape.assign(channel.shape.ptr, channel.shape.ptr + channel.shape.len);
    record.desc.shape.ptr = record.shape.data();
    if (channel.extern_name.len != 0) {
        record.extern_name.assign(
            reinterpret_cast<const char*>(channel.extern_name.ptr),
            channel.extern_name.len);
    }
    record.desc.extern_name.ptr =
        reinterpret_cast<const std::uint8_t*>(record.extern_name.data());
    record.shared_state = make_platform_channel_state(
        record.program_dtype(), record.numel(), channel.capacity);
    if (record.shared_state == nullptr) return PIE_STATUS_DRIVER_ERROR;
    auto [it, inserted] =
        channels_.emplace(channel.channel_id, std::move(record));
    if (!inserted) return PIE_STATUS_INVALID_ARGUMENT;

    ChannelRecord& stored = it->second;
    stored.desc.shape.ptr = stored.shape.data();
    stored.desc.extern_name.ptr =
        reinterpret_cast<const std::uint8_t*>(stored.extern_name.data());
    ChannelState& state = *stored.shared_state;
    *binding = PieChannelEndpointBinding{
        .channel_id = channel.channel_id,
        .mirror_base = reinterpret_cast<std::uint64_t>(state.cells.contents),
        .word_base = reinterpret_cast<std::uint64_t>(state.words.contents),
        .mirror_bytes = state.cells.size,
        .word_bytes = state.words.size,
        .cell_bytes = cell_bytes,
        .capacity = channel.capacity,
        .head_word_index = 0,
        .tail_word_index = 1,
        .poison_word_index = 2,
        .closed_word_index = 3,
    };
    return PIE_STATUS_OK;
}

int Registry::bind_instance(
    const PieInstanceDesc& instance,
    PieInstanceBinding* binding) {
    if (instance.geometry_class != PIE_GEOMETRY_CLASS_HOST) {
        return PIE_STATUS_UNSUPPORTED;
    }
    const ProgramRecord* program_ptr = find_program(instance.program_id);
    if (program_ptr == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    const std::uint64_t instance_id =
        instance.requested_instance_id != 0
            ? instance.requested_instance_id
            : next_instance_id_++;
    if (find_instance(instance_id) != nullptr) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    const ProgramRecord& program = *program_ptr;
    if (instance.channel_ids.len != program.channels.size()) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }

    std::unordered_set<std::uint64_t> unique_ids(
        instance.channel_ids.ptr,
        instance.channel_ids.ptr + instance.channel_ids.len);
    if (unique_ids.size() != instance.channel_ids.len) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    std::unordered_set<std::uint64_t> seeded_ids;
    for (std::size_t i = 0; i < instance.seed_values.len; ++i) {
        const PieChannelValueDesc& seed = instance.seed_values.ptr[i];
        const auto id = std::find(
            instance.channel_ids.ptr,
            instance.channel_ids.ptr + instance.channel_ids.len,
            seed.channel_id);
        if (id == instance.channel_ids.ptr + instance.channel_ids.len ||
            !seeded_ids.insert(seed.channel_id).second) {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
        const std::size_t channel =
            static_cast<std::size_t>(id - instance.channel_ids.ptr);
        if (!program.channels[channel].seeded ||
            seed.bytes.len != program.channels[channel].cell_bytes() ||
            seed.bytes.ptr == nullptr) {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
    }

    for (std::size_t i = 0; i < instance.channel_ids.len; ++i) {
        ChannelRecord* endpoint = find_channel(instance.channel_ids.ptr[i]);
        if (endpoint == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
        const auto& decl = program.channels[i];
        const auto& endpoint_desc = endpoint->desc;
        if (endpoint_desc.dtype != decl.dtype ||
            endpoint_desc.capacity != decl.capacity ||
            endpoint_desc.host_role != decl.host_role ||
            endpoint_desc.seeded != static_cast<std::uint8_t>(decl.seeded) ||
            endpoint->shape != decl.dims ||
            (decl.extern_dir == PIE_CHANNEL_EXTERN_NONE
                 ? endpoint_desc.extern_dir != PIE_CHANNEL_EXTERN_NONE ||
                       !endpoint->attachments.empty()
                 : endpoint_desc.extern_dir == PIE_CHANNEL_EXTERN_NONE ||
                       endpoint->extern_name != decl.extern_name ||
                       std::any_of(
                           endpoint->attachments.begin(),
                           endpoint->attachments.end(),
                           [&](const auto& attachment) {
                               return attachment.second == decl.extern_dir;
                           }))) {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
    }

    InstanceRecord record;
    record.instance_id = instance_id;
    record.program_id = instance.program_id;
    record.program_hash = program.program_hash;
    record.channel_ids.assign(
        instance.channel_ids.ptr,
        instance.channel_ids.ptr + instance.channel_ids.len);

    for (std::size_t i = 0; i < instance.seed_values.len; ++i) {
        const PieChannelValueDesc& seed = instance.seed_values.ptr[i];
        ChannelRecord& endpoint = *find_channel(seed.channel_id);
        Value value;
        if (!decode_wire(
                seed.bytes.ptr,
                seed.bytes.len,
                endpoint.program_dtype(),
                endpoint.numel(),
                value)) {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
        if (!endpoint.shared_state->empty() ||
            !endpoint.shared_state->push(std::move(value))) {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
    }

    std::vector<std::shared_ptr<ChannelState>> states;
    states.reserve(record.channel_ids.size());
    for (std::size_t i = 0; i < record.channel_ids.size(); ++i) {
        ChannelRecord& endpoint = *find_channel(record.channel_ids[i]);
        states.push_back(endpoint.shared_state);
    }
    record.interp = make_instance(program.plan, states);

    for (std::size_t i = 0; i < record.channel_ids.size(); ++i) {
        find_channel(record.channel_ids[i])
            ->attachments.emplace(instance_id, program.channels[i].extern_dir);
    }
    if (binding != nullptr) {
        std::memset(binding, 0, sizeof(*binding));
        binding->instance_id = instance_id;
        binding->geometry_class = instance.geometry_class;
    }
    instances_[instance_id] = std::move(record);
    return PIE_STATUS_OK;
}

ProgramRecord* Registry::find_program(std::uint64_t program_id) {
    const auto it = programs_.find(program_id);
    return it == programs_.end() ? nullptr : &it->second;
}

const ProgramRecord* Registry::find_program(std::uint64_t program_id) const {
    const auto it = programs_.find(program_id);
    return it == programs_.end() ? nullptr : &it->second;
}

InstanceRecord* Registry::find_instance(std::uint64_t instance_id) {
    const auto it = instances_.find(instance_id);
    return it == instances_.end() ? nullptr : &it->second;
}

const InstanceRecord* Registry::find_instance(std::uint64_t instance_id) const {
    const auto it = instances_.find(instance_id);
    return it == instances_.end() ? nullptr : &it->second;
}

ChannelRecord* Registry::find_channel(std::uint64_t channel_id) {
    const auto it = channels_.find(channel_id);
    return it == channels_.end() ? nullptr : &it->second;
}

const ChannelRecord* Registry::find_channel(std::uint64_t channel_id) const {
    const auto it = channels_.find(channel_id);
    return it == channels_.end() ? nullptr : &it->second;
}

int Registry::close_instance(std::uint64_t instance_id) {
    const auto it = instances_.find(instance_id);
    if (it == instances_.end()) return PIE_STATUS_CLOSED;
    for (const std::uint64_t channel_id : it->second.channel_ids) {
        if (ChannelRecord* channel = find_channel(channel_id)) {
            channel->attachments.erase(instance_id);
        }
    }
    instances_.erase(it);
    return PIE_STATUS_OK;
}

int Registry::close_channel(std::uint64_t channel_id) {
    const auto it = channels_.find(channel_id);
    if (it == channels_.end()) return PIE_STATUS_CLOSED;
    if (!it->second.attachments.empty()) return PIE_STATUS_INVALID_ARGUMENT;
    it->second.shared_state->store_word(3, 1);
    channels_.erase(it);
    return PIE_STATUS_OK;
}

}  // namespace pie::metal::pipeline
