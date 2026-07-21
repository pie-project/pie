#include "expert_stream_cache.hpp"

#include <fcntl.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cstring>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <unordered_map>

#include "cuda_check.hpp"
#include "../../weight_loader/include/weight_loader_cpp.hpp"

namespace pie_cuda_driver {

namespace {

constexpr std::uint64_t kSectionAlign = 256;

std::uint64_t align_up(std::uint64_t v, std::uint64_t a)
{
    return (v + a - 1) / a * a;
}

}  // namespace

// ── StreamedExpertTable ─────────────────────────────────────────────

std::uint64_t StreamedExpertTable::payload_bytes_per_expert() const noexcept
{
    std::uint64_t total = 0;
    for (const auto bytes : section_bytes) total += bytes;
    return total;
}

std::uint64_t StreamedExpertTable::total_payload_bytes() const noexcept
{
    return payload_bytes_per_expert() * extents.size();
}

const StreamedExpertExtents& StreamedExpertTable::at(int layer, int expert) const
{
    if (layer < 0 || layer >= num_layers || expert < 0 ||
        expert >= num_experts) {
        throw std::out_of_range(
            "StreamedExpertTable::at(" + std::to_string(layer) + ", " +
            std::to_string(expert) + ") outside " +
            std::to_string(num_layers) + "x" + std::to_string(num_experts));
    }
    return extents[static_cast<std::size_t>(layer) *
                       static_cast<std::size_t>(num_experts) +
                   static_cast<std::size_t>(expert)];
}

StreamedExpertTable streamed_expert_table_from_program(
    const pie_weight_loader::PieLoaderStorageProgramView& program)
{
    const auto& stream = program.stream;
    if (stream.template_.len == 0 || stream.sections_per_expert == 0) {
        throw std::runtime_error(
            "expert streaming: storage program has an empty stream plan "
            "(compile with stream_routed_experts)");
    }
    if (stream.template_.len != stream.sections_per_expert) {
        throw std::runtime_error(
            "expert streaming: stream template length mismatch");
    }
    if (stream.section_bytes.len != stream.sections_per_expert ||
        stream.section_offsets.len != stream.sections_per_expert ||
        stream.section_bytes.ptr == nullptr ||
        stream.section_offsets.ptr == nullptr) {
        throw std::runtime_error(
            "expert streaming: stream plan section layout length mismatch");
    }
    if (stream.num_layers == 0 || stream.num_experts == 0) {
        throw std::runtime_error(
            "expert streaming: stream plan has empty expert grid");
    }

    const int sections = static_cast<int>(stream.sections_per_expert);
    const std::size_t expected_bindings =
        static_cast<std::size_t>(stream.num_layers) *
        static_cast<std::size_t>(stream.num_experts) *
        static_cast<std::size_t>(sections);
    if (stream.bindings.len != expected_bindings ||
        stream.bindings.ptr == nullptr) {
        throw std::runtime_error(
            "expert streaming: stream bindings length mismatch (got " +
            std::to_string(stream.bindings.len) + ", expected " +
            std::to_string(expected_bindings) + ")");
    }

    // Validate template instrs are ExtentWrites with slot-relative dests.
    for (std::size_t i = 0; i < stream.template_.len; ++i) {
        const std::uint32_t id = stream.template_.ptr[i];
        const pie_weight_loader::PieLoaderStorageInstrView* instr = nullptr;
        for (std::size_t j = 0; j < program.instrs.len; ++j) {
            if (program.instrs.ptr[j].id == id) {
                instr = &program.instrs.ptr[j];
                break;
            }
        }
        if (instr == nullptr) {
            throw std::runtime_error(
                "expert streaming: stream template instr id " +
                std::to_string(id) + " missing from program.instrs");
        }
        if (instr->kind !=
            pie_weight_loader::PieLoaderStorageInstrKind::ExtentWrite) {
            throw std::runtime_error(
                "expert streaming: stream template instr must be ExtentWrite");
        }
        if (!instr->has_dest) {
            throw std::runtime_error(
                "expert streaming: stream template ExtentWrite missing dest");
        }
        if (instr->dest.offset != stream.section_offsets.ptr[i]) {
            throw std::runtime_error(
                "expert streaming: template dest offset does not match "
                "stream.section_offsets");
        }
    }

    StreamedExpertTable table;
    table.num_layers = static_cast<int>(stream.num_layers);
    table.num_experts = static_cast<int>(stream.num_experts);
    table.sections_per_expert = sections;
    table.slot_bytes = stream.slot_bytes;
    table.section_bytes.assign(
        stream.section_bytes.ptr,
        stream.section_bytes.ptr + stream.section_bytes.len);
    table.section_offsets.assign(
        stream.section_offsets.ptr,
        stream.section_offsets.ptr + stream.section_offsets.len);

    // Compact unique shard paths; map FileId → shard index in table.
    std::unordered_map<std::uint32_t, std::uint32_t> file_to_shard;
    auto shard_for_file = [&](std::uint32_t file_id) -> std::uint32_t {
        auto it = file_to_shard.find(file_id);
        if (it != file_to_shard.end()) return it->second;
        if (file_id >= stream.files.len || stream.files.ptr == nullptr) {
            throw std::runtime_error(
                "expert streaming: stream binding file_id " +
                std::to_string(file_id) + " out of range");
        }
        const auto path =
            pie_weight_loader::cpp::bytes_to_string(stream.files.ptr[file_id]);
        if (path.empty()) {
            throw std::runtime_error(
                "expert streaming: empty path for file_id " +
                std::to_string(file_id));
        }
        const auto shard =
            static_cast<std::uint32_t>(table.shard_paths.size());
        table.shard_paths.push_back(path);
        file_to_shard.emplace(file_id, shard);
        return shard;
    };

    const std::size_t n_experts =
        static_cast<std::size_t>(table.num_layers) *
        static_cast<std::size_t>(table.num_experts);
    table.extents.resize(n_experts);
    for (int layer = 0; layer < table.num_layers; ++layer) {
        for (int expert = 0; expert < table.num_experts; ++expert) {
            auto& entry =
                table.extents[static_cast<std::size_t>(layer) *
                                  static_cast<std::size_t>(table.num_experts) +
                              static_cast<std::size_t>(expert)];
            entry.sections.resize(static_cast<std::size_t>(sections));
            for (int s = 0; s < sections; ++s) {
                const std::size_t bi =
                    (static_cast<std::size_t>(layer) *
                         static_cast<std::size_t>(table.num_experts) +
                     static_cast<std::size_t>(expert)) *
                        static_cast<std::size_t>(sections) +
                    static_cast<std::size_t>(s);
                const auto& b = stream.bindings.ptr[bi];
                if (b.span_bytes !=
                    table.section_bytes[static_cast<std::size_t>(s)]) {
                    throw std::runtime_error(
                        "expert streaming: binding span does not match "
                        "section_bytes");
                }
                entry.sections[static_cast<std::size_t>(s)] =
                    ExpertSectionExtent{
                        .shard = shard_for_file(b.file_id),
                        .file_offset = b.file_offset,
                    };
            }
        }
    }
    return table;
}

// ── ExpertSlotIndex ─────────────────────────────────────────────────

ExpertSlotIndex::ExpertSlotIndex(int num_layers, int num_experts, int num_slots)
    : num_layers_(num_layers), num_experts_(num_experts)
{
    if (num_layers <= 0 || num_experts <= 0 || num_slots <= 0) {
        throw std::invalid_argument(
            "ExpertSlotIndex: layers/experts/slots must be positive");
    }
    slot_of_.assign(static_cast<std::size_t>(num_layers) *
                        static_cast<std::size_t>(num_experts),
                    -1);
    slots_.resize(static_cast<std::size_t>(num_slots));
}

std::size_t ExpertSlotIndex::key(int layer, int expert) const
{
    if (layer < 0 || layer >= num_layers_ || expert < 0 ||
        expert >= num_experts_) {
        throw std::out_of_range(
            "ExpertSlotIndex: (" + std::to_string(layer) + ", " +
            std::to_string(expert) + ") outside " +
            std::to_string(num_layers_) + "x" + std::to_string(num_experts_));
    }
    return static_cast<std::size_t>(layer) *
               static_cast<std::size_t>(num_experts_) +
           static_cast<std::size_t>(expert);
}

int ExpertSlotIndex::find(int layer, int expert) const
{
    return slot_of_[key(layer, expert)];
}

void ExpertSlotIndex::touch_and_pin(int slot)
{
    auto& s = slots_.at(static_cast<std::size_t>(slot));
    s.age = ++tick_;
    s.pinned = true;
}

ExpertSlotIndex::Acquired ExpertSlotIndex::acquire(int layer, int expert)
{
    const std::size_t k = key(layer, expert);

    int victim = -1;
    std::uint64_t best_age = std::numeric_limits<std::uint64_t>::max();
    for (int i = 0; i < num_slots(); ++i) {
        const auto& s = slots_[static_cast<std::size_t>(i)];
        if (s.pinned) continue;
        if (s.age < best_age) {
            best_age = s.age;
            victim = i;
        }
    }
    if (victim < 0) {
        throw std::runtime_error(
            "ExpertSlotIndex: all " + std::to_string(num_slots()) +
            " slots pinned by the current batch — batch larger than cache");
    }

    auto& s = slots_[static_cast<std::size_t>(victim)];
    Acquired out{.slot = victim, .evicted = s.layer >= 0};
    if (out.evicted) {
        slot_of_[key(s.layer, s.expert)] = -1;
        ++evictions_;
    }
    s.layer = layer;
    s.expert = expert;
    s.age = ++tick_;
    s.pinned = true;
    slot_of_[k] = victim;
    return out;
}

void ExpertSlotIndex::unpin_all()
{
    for (auto& s : slots_) s.pinned = false;
}

// ── ExpertStreamCache ───────────────────────────────────────────────

ExpertStreamCache::ExpertStreamCache(StreamedExpertTable table,
                                     std::uint64_t budget_bytes,
                                     bool verbose)
    : table_(std::move(table)), verbose_(verbose)
{
    if (table_.empty()) {
        throw std::runtime_error("expert streaming: empty extent table");
    }
    if (table_.sections_per_expert <= 0 ||
        static_cast<int>(table_.section_bytes.size()) !=
            table_.sections_per_expert) {
        throw std::runtime_error(
            "expert streaming: table section layout is incomplete");
    }

    if (table_.slot_bytes > 0) {
        if (static_cast<int>(table_.section_offsets.size()) !=
            table_.sections_per_expert) {
            throw std::runtime_error(
                "expert streaming: table section_offsets length mismatch");
        }
        section_offsets_ = table_.section_offsets;
        slot_stride_ = table_.slot_bytes;
    } else {
        section_offsets_.assign(
            static_cast<std::size_t>(table_.sections_per_expert), 0);
        std::uint64_t offset = 0;
        for (int s = 0; s < table_.sections_per_expert; ++s) {
            section_offsets_[static_cast<std::size_t>(s)] = offset;
            offset = align_up(
                offset + table_.section_bytes[static_cast<std::size_t>(s)],
                kSectionAlign);
        }
        slot_stride_ = align_up(offset, kSectionAlign);
    }

    const std::uint64_t total_experts = table_.extents.size();
    const std::uint64_t budget_slots = budget_bytes / slot_stride_;
    if (budget_slots == 0) {
        throw std::runtime_error(
            "expert streaming: cache budget " + std::to_string(budget_bytes) +
            " B cannot fit one expert slot (" + std::to_string(slot_stride_) +
            " B); raise [model].expert_cache_gb");
    }
    const int num_slots = static_cast<int>(
        std::min<std::uint64_t>(budget_slots, total_experts));
    index_ = ExpertSlotIndex(table_.num_layers, table_.num_experts, num_slots);

    CUDA_CHECK(cudaMalloc(
        &slab_, slot_stride_ * static_cast<std::uint64_t>(num_slots)));

    shard_fds_.reserve(table_.shard_paths.size());
    for (const auto& path : table_.shard_paths) {
        const int fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        if (fd < 0) {
            throw std::runtime_error(
                "expert streaming: failed to open shard '" + path.string() +
                "': " + std::strerror(errno));
        }
        shard_fds_.push_back(fd);
    }

    CUDA_CHECK(cudaStreamCreateWithFlags(&upload_stream_,
                                         cudaStreamNonBlocking));
    staging_.assign(2, nullptr);
    staging_done_.assign(2, nullptr);
    for (int b = 0; b < 2; ++b) {
        CUDA_CHECK(cudaMallocHost(
            reinterpret_cast<void**>(&staging_[static_cast<std::size_t>(b)]),
            slot_stride_));
        CUDA_CHECK(cudaEventCreateWithFlags(
            &staging_done_[static_cast<std::size_t>(b)],
            cudaEventDisableTiming));
    }

    if (verbose_) {
        std::cerr << "[pie-driver-cuda] expert stream cache: "
                  << num_slots << "/" << total_experts << " slots x "
                  << (slot_stride_ / (1024 * 1024)) << " MiB (slab "
                  << (slab_bytes() / (1024 * 1024)) << " MiB, payload "
                  << (table_.payload_bytes_per_expert() / (1024 * 1024))
                  << " MiB/expert, " << table_.sections_per_expert
                  << " sections, " << table_.shard_paths.size()
                  << " shards; deferred ExtentWrite template)\n";
    }
}

ExpertStreamCache::~ExpertStreamCache()
{
    for (std::size_t b = 0; b < staging_done_.size(); ++b) {
        if (staging_done_[b] != nullptr) {
            cudaEventDestroy(staging_done_[b]);
        }
        if (b < staging_.size() && staging_[b] != nullptr) {
            cudaFreeHost(staging_[b]);
        }
    }
    if (upload_stream_ != nullptr) {
        cudaStreamDestroy(upload_stream_);
    }
    if (slab_ != nullptr) {
        cudaFree(slab_);
    }
    for (const int fd : shard_fds_) {
        if (fd >= 0) ::close(fd);
    }
}

ExpertSectionPointers ExpertStreamCache::slot_pointers(int slot) const
{
    ExpertSectionPointers out;
    out.section.resize(static_cast<std::size_t>(table_.sections_per_expert));
    const auto* base = static_cast<const std::uint8_t*>(slot_base(slot));
    for (int s = 0; s < table_.sections_per_expert; ++s) {
        out.section[static_cast<std::size_t>(s)] =
            base + section_offsets_[static_cast<std::size_t>(s)];
    }
    return out;
}

void ExpertStreamCache::execute_stream_template(int layer,
                                                int expert,
                                                std::uint8_t* buf)
{
    // Deferred loader execution: each section is an ExtentWrite from the
    // stream plan's binding into the slot-relative dest offset.
    const auto& e = table_.at(layer, expert);
    if (static_cast<int>(e.sections.size()) != table_.sections_per_expert) {
        throw std::runtime_error(
            "expert streaming: extent section count mismatch");
    }
    for (int s = 0; s < table_.sections_per_expert; ++s) {
        const auto& sec = e.sections[static_cast<std::size_t>(s)];
        if (sec.shard >= shard_fds_.size()) {
            throw std::runtime_error(
                "expert streaming: shard index out of range");
        }
        const int fd = shard_fds_[sec.shard];
        std::uint8_t* dst = buf + section_offsets_[static_cast<std::size_t>(s)];
        std::uint64_t remaining =
            table_.section_bytes[static_cast<std::size_t>(s)];
        std::uint64_t off = sec.file_offset;
        while (remaining > 0) {
            const ssize_t n = ::pread(
                fd, dst, static_cast<std::size_t>(remaining),
                static_cast<off_t>(off));
            if (n < 0) {
                throw std::runtime_error(
                    "expert streaming: pread failed for stream-template "
                    "ExtentWrite (layer=" +
                    std::to_string(layer) + " expert=" +
                    std::to_string(expert) + " section=" +
                    std::to_string(s) + "): " + std::strerror(errno));
            }
            if (n == 0) {
                throw std::runtime_error(
                    "expert streaming: short pread for stream-template "
                    "ExtentWrite");
            }
            dst += static_cast<std::uint64_t>(n);
            off += static_cast<std::uint64_t>(n);
            remaining -= static_cast<std::uint64_t>(n);
        }
        stats_.bytes_read += table_.section_bytes[static_cast<std::size_t>(s)];
    }
}

void ExpertStreamCache::ensure_resident(int layer,
                                        std::span<const int> experts,
                                        cudaStream_t compute_stream,
                                        std::vector<ExpertSectionPointers>& out)
{
    out.assign(experts.size(), ExpertSectionPointers{});
    if (experts.empty()) return;
    if (static_cast<int>(experts.size()) > num_slots()) {
        throw std::runtime_error(
            "expert streaming: ensure_resident batch of " +
            std::to_string(experts.size()) + " experts exceeds the cache's " +
            std::to_string(num_slots()) + " slots");
    }

    std::vector<std::size_t> misses;
    misses.reserve(experts.size());
    for (std::size_t i = 0; i < experts.size(); ++i) {
        const int slot = index_.find(layer, experts[i]);
        if (slot >= 0) {
            index_.touch_and_pin(slot);
            out[i] = slot_pointers(slot);
            ++stats_.hits;
        } else {
            misses.push_back(i);
            ++stats_.misses;
        }
    }

    if (!misses.empty()) {
        // Sync compute so no kernel can still be reading a victim slot.
        CUDA_CHECK(cudaStreamSynchronize(compute_stream));
    }

    for (std::size_t mi = 0; mi < misses.size(); ++mi) {
        const std::size_t i = misses[mi];
        const int expert = experts[i];
        const auto acquired = index_.acquire(layer, expert);
        if (acquired.evicted) {
            stats_.evictions = index_.evictions();
        }

        const int buf = static_cast<int>(mi & 1);
        CUDA_CHECK(cudaEventSynchronize(
            staging_done_[static_cast<std::size_t>(buf)]));

        const auto t0 = std::chrono::steady_clock::now();
        execute_stream_template(layer, expert,
                                staging_[static_cast<std::size_t>(buf)]);
        stats_.pread_ms +=
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0)
                .count();

        CUDA_CHECK(cudaMemcpyAsync(
            slot_base(acquired.slot), staging_[static_cast<std::size_t>(buf)],
            slot_stride_, cudaMemcpyHostToDevice, upload_stream_));
        CUDA_CHECK(cudaEventRecord(
            staging_done_[static_cast<std::size_t>(buf)], upload_stream_));
        out[i] = slot_pointers(acquired.slot);
    }

    if (!misses.empty()) {
        const auto t0 = std::chrono::steady_clock::now();
        CUDA_CHECK(cudaStreamSynchronize(upload_stream_));
        stats_.upload_wait_ms +=
            std::chrono::duration<double, std::milli>(
                std::chrono::steady_clock::now() - t0)
                .count();
    }
    index_.unpin_all();
}

ExpertStreamCache::Stats ExpertStreamCache::stats() const { return stats_; }

void ExpertStreamCache::log_stats(const char* tag) const
{
    const auto s = stats();
    std::cerr << "[pie-driver-cuda] expert stream cache (" << tag << "): "
              << s.hits << " hits, " << s.misses << " misses, "
              << s.evictions << " evictions, "
              << (s.bytes_read / (1024 * 1024)) << " MiB read, pread "
              << static_cast<int>(s.pread_ms) << " ms, upload wait "
              << static_cast<int>(s.upload_wait_ms) << " ms\n";
}

}  // namespace pie_cuda_driver
