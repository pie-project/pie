#include "pipeline/registry.hpp"

namespace pie_cuda_driver::pipeline {

int Registry::register_program(
    const PieProgramDesc& program,
    std::uint64_t* program_id,
    std::string* err) {
    auto found = program_ids_by_hash_.find(program.program_hash);
    if (found != program_ids_by_hash_.end()) {
        if (program_id != nullptr) *program_id = found->second;
        return PIE_STATUS_OK;
    }

    if (program.canonical_bytes.len == 0) return PIE_STATUS_INVALID_ARGUMENT;
    const int rc = dispatch_.register_program(
        program.program_hash,
        pie_native::ByteSlice{program.canonical_bytes.ptr, program.canonical_bytes.len},
        pie_native::ByteSlice{program.sidecar_bytes.ptr, program.sidecar_bytes.len},
        err);
    if (rc != PIE_STATUS_OK) return rc;

    ProgramRecord record;
    record.program_id = next_program_id_++;
    record.program_hash = program.program_hash;
    program_ids_by_hash_[record.program_hash] = record.program_id;
    if (program_id != nullptr) *program_id = record.program_id;
    programs_.emplace(record.program_id, std::move(record));
    return PIE_STATUS_OK;
}

int Registry::register_channel(
    const PieChannelDesc& channel,
    PieChannelEndpointBinding* binding,
    std::string* err) {
    return dispatch_.register_channel(channel, binding, err);
}

int Registry::bind_instance(
    const PieInstanceDesc& instance,
    PieInstanceBinding* binding,
    std::string* err) {
    auto pit = programs_.find(instance.program_id);
    if (pit == programs_.end()) return PIE_STATUS_INVALID_ARGUMENT;

    std::vector<std::uint64_t> channel_ids(
        instance.channel_ids.ptr,
        instance.channel_ids.ptr + instance.channel_ids.len);
    std::vector<PieChannelValueDesc> seeds;
    if (instance.seed_values.len > 0) {
        seeds.assign(instance.seed_values.ptr,
                     instance.seed_values.ptr + instance.seed_values.len);
    }

    const std::uint64_t instance_id = instance.requested_instance_id != 0
        ? instance.requested_instance_id
        : next_instance_id_++;
    const int rc = dispatch_.bind_instance(
        instance_id,
        pit->second.program_hash,
        instance.pacing_wait_id,
        channel_ids,
        seeds,
        binding,
        err);
    if (rc != PIE_STATUS_OK) return rc;

    instances_[instance_id] = InstanceRecord{
        instance_id, instance.program_id, pit->second.program_hash};
    return PIE_STATUS_OK;
}

int Registry::resolve_instances(
    const PieU64Slice& instance_ids,
    std::vector<InstanceRecord>* out) const {
    out->clear();
    out->reserve(instance_ids.len);
    for (std::size_t i = 0; i < instance_ids.len; ++i) {
        auto it = instances_.find(instance_ids.ptr[i]);
        if (it == instances_.end()) return PIE_STATUS_INVALID_ARGUMENT;
        out->push_back(it->second);
    }
    return PIE_STATUS_OK;
}

int Registry::close_instance(std::uint64_t instance_id) {
    auto it = instances_.find(instance_id);
    if (it == instances_.end()) return PIE_STATUS_CLOSED;
    dispatch_.close_instance(instance_id);
    instances_.erase(it);
    return PIE_STATUS_OK;
}

int Registry::close_channel(std::uint64_t channel_id, std::string* err) {
    return dispatch_.close_channel(channel_id, err);
}

}  // namespace pie_cuda_driver::pipeline
