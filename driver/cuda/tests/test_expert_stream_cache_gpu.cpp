// GPU integration test for ExpertStreamCache: execute deferred stream-template
// ExtentWrites (pread from a real temp file into pinned staging, async H2D
// into the LRU slab), then D2H readback and byte-compare against the file
// contents. Requires one CUDA device.

#include "expert_stream_cache.hpp"

#include <unistd.h>

#include <array>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <span>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "cuda_check.hpp"

namespace {

using pie_cuda_driver::ExpertSectionPointers;
using pie_cuda_driver::ExpertStreamCache;
using pie_cuda_driver::StreamedExpertTable;

#define CHECK(cond)                                                      \
    do {                                                                 \
        if (!(cond)) {                                                   \
            std::cerr << "FAILED at " << __FILE__ << ":" << __LINE__     \
                      << ": " #cond "\n";                                \
            std::exit(1);                                                \
        }                                                                \
    } while (0)

constexpr int kLayers = 2;
constexpr int kExperts = 4;
// Local section count for this test (not an arch constant from the cache).
constexpr int kSections = 6;
// Deliberately non-uniform, non-aligned section sizes.
constexpr std::array<std::uint64_t, kSections> kSectionBytes = {
    1024, 96, 512, 48, 1000, 100,
};

// Deterministic byte pattern unique per (layer, expert, section, i).
std::uint8_t pattern(int layer, int expert, int section, std::uint64_t i)
{
    return static_cast<std::uint8_t>(
        (layer * 131 + expert * 31 + section * 7 + static_cast<int>(i)) & 0xff);
}

// Write every expert's sections back-to-back into one shard file and
// record the extents.
StreamedExpertTable make_table(const std::filesystem::path& shard)
{
    StreamedExpertTable table;
    table.num_layers = kLayers;
    table.num_experts = kExperts;
    table.sections_per_expert = kSections;
    table.section_bytes.assign(kSectionBytes.begin(), kSectionBytes.end());
    table.shard_paths = {shard};
    table.extents.resize(kLayers * kExperts);

    std::ofstream out(shard, std::ios::binary);
    CHECK(out.good());
    std::uint64_t offset = 0;
    for (int l = 0; l < kLayers; ++l) {
        for (int e = 0; e < kExperts; ++e) {
            auto& entry = table.extents[static_cast<std::size_t>(
                l * kExperts + e)];
            entry.sections.resize(static_cast<std::size_t>(kSections));
            for (int s = 0; s < kSections; ++s) {
                entry.sections[static_cast<std::size_t>(s)] = {
                    .shard = 0,
                    .file_offset = offset,
                };
                std::vector<std::uint8_t> bytes(kSectionBytes[
                    static_cast<std::size_t>(s)]);
                for (std::uint64_t i = 0; i < bytes.size(); ++i) {
                    bytes[i] = pattern(l, e, s, i);
                }
                out.write(reinterpret_cast<const char*>(bytes.data()),
                          static_cast<std::streamsize>(bytes.size()));
                offset += bytes.size();
            }
        }
    }
    out.close();
    return table;
}

// D2H-read one resident expert's sections and compare with the pattern.
void check_expert_bytes(const ExpertSectionPointers& ptrs,
                        int layer,
                        int expert)
{
    CHECK(ptrs.num_sections() == kSections);
    for (int s = 0; s < kSections; ++s) {
        const std::uint64_t n = kSectionBytes[static_cast<std::size_t>(s)];
        std::vector<std::uint8_t> host(n);
        CUDA_CHECK(cudaMemcpy(host.data(),
                              ptrs.at(s),
                              n, cudaMemcpyDeviceToHost));
        for (std::uint64_t i = 0; i < n; ++i) {
            if (host[i] != pattern(layer, expert, s, i)) {
                std::cerr << "byte mismatch layer=" << layer
                          << " expert=" << expert << " section=" << s
                          << " i=" << i << "\n";
                std::exit(1);
            }
        }
    }
}

std::uint64_t payload_per_expert()
{
    std::uint64_t total = 0;
    for (const auto b : kSectionBytes) total += b;
    return total;
}

}  // namespace

int main()
{
    int device_count = 0;
    if (cudaGetDeviceCount(&device_count) != cudaSuccess ||
        device_count == 0) {
        std::cout << "expert_stream_cache_gpu: no CUDA device, skipping\n";
        return 0;
    }

    char tmpl[] = "/tmp/pie_expert_stream_XXXXXX";
    const int tmp_fd = ::mkstemp(tmpl);
    CHECK(tmp_fd >= 0);
    ::close(tmp_fd);
    const std::filesystem::path shard(tmpl);

    const StreamedExpertTable table = make_table(shard);

    // Slot stride = sections at 256-aligned offsets, so a budget of three
    // padded strides yields exactly 3 slots (< the 8 experts in the table).
    std::uint64_t stride = 0;
    {
        const std::uint64_t generous = 16 * payload_per_expert();
        ExpertStreamCache probe(table, generous, /*verbose=*/false);
        CHECK(probe.num_slots() == kLayers * kExperts);  // clamped to expert set
        CHECK(probe.sections_per_expert() == kSections);
        stride = probe.slot_stride_bytes();
        CHECK(stride >= payload_per_expert());
    }

    ExpertStreamCache cache(table, 3 * stride, /*verbose=*/false);
    CHECK(cache.num_slots() == 3);

    cudaStream_t stream = nullptr;
    CUDA_CHECK(cudaStreamCreate(&stream));
    std::vector<ExpertSectionPointers> ptrs;

    // Cold misses: three experts of layer 0 fill the cache; bytes must
    // round-trip exactly.
    {
        const std::array<int, 3> ids = {0, 1, 2};
        cache.ensure_resident(0, std::span<const int>(ids), stream, ptrs);
        CHECK(ptrs.size() == 3);
        for (int i = 0; i < 3; ++i) check_expert_bytes(ptrs[i], 0, ids[i]);
        const auto s = cache.stats();
        CHECK(s.hits == 0 && s.misses == 3 && s.evictions == 0);
        CHECK(s.bytes_read == 3 * payload_per_expert());
    }

    // Full hit: same experts, same slots, no new I/O.
    {
        const std::array<int, 3> ids = {0, 1, 2};
        cache.ensure_resident(0, std::span<const int>(ids), stream, ptrs);
        for (int i = 0; i < 3; ++i) check_expert_bytes(ptrs[i], 0, ids[i]);
        const auto s = cache.stats();
        CHECK(s.hits == 3 && s.misses == 3 && s.evictions == 0);
        CHECK(s.bytes_read == 3 * payload_per_expert());
    }

    // Mixed batch with eviction: expert 0 hits and is pinned; expert 3
    // must evict one of {1, 2} (the LRU), never the pinned 0.
    {
        const std::array<int, 2> ids = {0, 3};
        cache.ensure_resident(0, std::span<const int>(ids), stream, ptrs);
        check_expert_bytes(ptrs[0], 0, 0);
        check_expert_bytes(ptrs[1], 0, 3);
        const auto s = cache.stats();
        CHECK(s.hits == 4 && s.misses == 4 && s.evictions == 1);
    }

    // Cross-layer: layer-1 experts are distinct cache entries.
    {
        const std::array<int, 2> ids = {0, 3};
        cache.ensure_resident(1, std::span<const int>(ids), stream, ptrs);
        check_expert_bytes(ptrs[0], 1, 0);
        check_expert_bytes(ptrs[1], 1, 3);
        const auto s = cache.stats();
        CHECK(s.hits == 4 && s.misses == 6);
    }

    // Batch larger than the cache must be rejected (callers chunk).
    {
        const std::array<int, 4> ids = {0, 1, 2, 3};
        bool threw = false;
        try {
            cache.ensure_resident(0, std::span<const int>(ids), stream, ptrs);
        } catch (const std::exception&) {
            threw = true;
        }
        CHECK(threw);
    }

    CUDA_CHECK(cudaStreamDestroy(stream));
    std::filesystem::remove(shard);
    std::cout << "expert_stream_cache_gpu: all tests passed\n";
    return 0;
}
