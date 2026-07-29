#include "model/expert_pack_build.hpp"

namespace pie_cuda_driver {

void expert_pack_d2h_append(
    ExpertPackWriter& writer,
    const void* device_ptr,
    std::uint64_t nbytes,
    std::vector<std::uint8_t>& host)
{
    // Reuse the caller's bounce buffer so we don't allocate per section.
    host.resize(static_cast<std::size_t>(nbytes));
    CUDA_CHECK(cudaMemcpy(
        host.data(), device_ptr, static_cast<std::size_t>(nbytes),
        cudaMemcpyDeviceToHost));
    writer.append_bytes(host.data(), nbytes);
}

void expert_pack_emit_slot_sections(
    ExpertPackWriter& writer,
    const StreamedExpertTable& table,
    const DeviceBuf* const* sections,
    int num_sections,
    std::vector<std::uint8_t>& host_bounce)
{
    // Lay out sections at the same offsets the stream template expects, then
    // pad to a full slot so every (layer, expert) occupies `slot_bytes`.
    if (table.slot_bytes == 0) {
        throw std::runtime_error(
            "expert pack: refuse to emit with slot_bytes == 0");
    }
    const auto& sb = table.section_bytes;
    const std::uint64_t slot = table.slot_bytes;
    std::uint64_t cursor = 0;
    for (int s = 0; s < num_sections; ++s) {
        const std::uint64_t off =
            table.section_offsets[static_cast<std::size_t>(s)];
        if (off < cursor) {
            throw std::runtime_error(
                "expert pack: section_offsets[" + std::to_string(s) +
                "]=" + std::to_string(off) +
                " is before write cursor " + std::to_string(cursor));
        }
        if (off > cursor) {
            writer.append_zeros(off - cursor);
            cursor = off;
        }
        const std::uint64_t nbytes = sb[static_cast<std::size_t>(s)];
        expert_pack_d2h_append(writer, sections[s]->ptr, nbytes, host_bounce);
        cursor += nbytes;
    }
    if (slot > cursor) {
        writer.append_zeros(slot - cursor);
    }
}

}  // namespace pie_cuda_driver
