#pragma once

// Generic offline expert-pack builder: hit/miss, writer lifecycle, and L×E loop.
// Per-kind GPU transforms live in specialization TUs that supply a Traits type.
// Peak VRAM stays O(one expert): staging buffers are reused across the loop.

#include <cstdint>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_check.hpp"
#include "model/expert_pack_cache.hpp"

namespace pie_cuda_driver {

// Owning CUDA device allocation used as staging for one expert at a time.
struct DeviceBuf {
    void* ptr = nullptr;
    std::uint64_t bytes = 0;
    DeviceBuf() = default;
    explicit DeviceBuf(std::uint64_t n) : bytes(n)
    {
        if (n == 0) return;
        CUDA_CHECK(cudaMalloc(&ptr, static_cast<std::size_t>(n)));
    }
    ~DeviceBuf()
    {
        if (ptr) cudaFree(ptr);
    }
    DeviceBuf(const DeviceBuf&) = delete;
    DeviceBuf& operator=(const DeviceBuf&) = delete;
    DeviceBuf(DeviceBuf&& o) noexcept : ptr(o.ptr), bytes(o.bytes)
    {
        o.ptr = nullptr;
        o.bytes = 0;
    }
    DeviceBuf& operator=(DeviceBuf&& o) noexcept
    {
        if (this == &o) return *this;
        if (ptr) cudaFree(ptr);
        ptr = o.ptr;
        bytes = o.bytes;
        o.ptr = nullptr;
        o.bytes = 0;
        return *this;
    }
};

// Copy `nbytes` from device into `host`, then append that payload to the pack.
void expert_pack_d2h_append(
    ExpertPackWriter& writer,
    const void* device_ptr,
    std::uint64_t nbytes,
    std::vector<std::uint8_t>& host);

// Write one expert slot: each section at `table.section_offsets[i]`, then pad
// the remainder of `table.slot_bytes` with zeros so slots stay fixed-size.
void expert_pack_emit_slot_sections(
    ExpertPackWriter& writer,
    const StreamedExpertTable& table,
    const DeviceBuf* const* sections,
    int num_sections,
    std::vector<std::uint8_t>& host_bounce);

// Build or open `{cache_key}.experts` and remaps `table` extents onto it.
//
// Traits contract:
//   kSections              — expected table.sections_per_expert
//   miss_label()           — verbose string for a cold build
//   require_build_support()— throw if this pack kind cannot be built
//   Context                — geometry + reusable staging buffers
//   prepare(...)           — probe shapes / allocate Context
//   load_expert(...)       — H2D one expert from the checkpoint
//   transform(...)         — GPU convert into pack section buffers
//   emit(...)              — D2H those sections into the writer slot
template <typename Traits>
bool ensure_expert_pack(
    StreamedExpertTable& table,
    const std::string& cache_key,
    SafetensorsCheckpointSource& checkpoint,
    bool verbose)
{
    if (table.sections_per_expert != Traits::kSections) {
        throw std::runtime_error(
            std::string("expert pack: expected ") +
            std::to_string(Traits::kSections) + " sections, got " +
            std::to_string(table.sections_per_expert));
    }
    if (table.slot_bytes == 0) {
        throw std::runtime_error(
            "expert pack: stream table slot_bytes must be > 0");
    }
    // Warm path: pack already on disk for this cache key + layout.
    if (ExpertPackWriter::exists_and_matches(cache_key, table)) {
        if (verbose) {
            std::cerr << "[pie-driver-cuda] expert pack hit "
                      << expert_pack_path(cache_key) << "\n";
        }
        remap_streamed_table_to_expert_pack(
            table, expert_pack_path(cache_key), cache_key);
        return true;
    }

    Traits::require_build_support();

    if (verbose) {
        std::cerr << "[pie-driver-cuda] expert pack miss — "
                  << Traits::miss_label()
                  << " (O(one expert) staging) → "
                  << expert_pack_path(cache_key) << "\n";
    }

    // Cold path: stream every expert through staging into a temp pack file.
    ExpertPackWriter writer(cache_key, table);
    try {
        auto ctx = Traits::prepare(table, checkpoint);
        std::vector<std::uint8_t> host_bounce;
        const int L = table.num_layers;
        const int E = table.num_experts;
        for (int layer = 0; layer < L; ++layer) {
            for (int expert = 0; expert < E; ++expert) {
                Traits::load_expert(ctx, checkpoint, layer, expert);
                Traits::transform(ctx);
                Traits::emit(ctx, writer, table, host_bounce);
            }
        }
        writer.finalize();
        if (verbose) {
            std::cerr << "[pie-driver-cuda] expert pack written "
                      << expert_pack_path(cache_key) << " ("
                      << (expert_pack_body_bytes(table) / (1024 * 1024))
                      << " MiB body)\n";
        }
    } catch (...) {
        writer.abandon();
        throw;
    }

    // Point the stream cache at the finished pack instead of HF shards.
    remap_streamed_table_to_expert_pack(
        table, expert_pack_path(cache_key), cache_key);
    return true;
}

}  // namespace pie_cuda_driver
