#pragma once
// checkpoint_source.hpp — zero-copy byte-range reader over the files a LoadPlan
// declares.
//
// This used to be `SafetensorsView`, a name-addressed reader that discovered the
// shards itself, parsed every safetensors header, and handed out tensors by
// name. None of that survives: the loader parses the headers and the plan
// carries the resulting byte offsets, so the remaining job is to map the files
// the plan names and hand out spans of them (`architecture.md` §6).
//
//     pie_loader::CheckpointSource source(plan.view());
//     source.copy_storage_bytes(file_id, offset, bytes, dst, tile);
//
// It ships with the loader rather than with either driver because nothing in it
// is device-specific — it is POSIX and the plan's file table, and it was
// previously written out twice, once per backend. The two copies had already
// drifted in every incidental way (eager versus lazy mapping, `MAP_PRIVATE`
// versus `MAP_SHARED`, three different spellings of the same bounds check)
// while agreeing on everything that mattered.
//
// Three ways to get at a span, because the three callers want different things:
//
//   * `storage_host_ptr`         — the mapped bytes, for a device copy engine
//                                  that will DMA straight out of the page cache.
//   * `read_storage_bytes_to_host` — `pread` into a caller's buffer, for a
//                                  pinned staging lane that wants the copy done
//                                  by the kernel rather than by a page fault.
//   * `copy_storage_bytes`       — `memcpy` in tiles, dropping each tile from
//                                  the page cache behind it, for unified memory
//                                  where the destination *is* host memory and
//                                  the mapping would otherwise double the RSS.
//
// Mappings are read-only and live for the lifetime of the source.

#include <algorithm>
#include <cerrno>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include "pie_loader.h"

namespace pie_loader {

class CheckpointSource {
  public:
    /// Map every file the plan declares, in the plan's own order.
    ///
    /// `file_id` is an index into that table, so no rediscovery happens here —
    /// and none is wanted, since a second enumeration of the snapshot directory
    /// could disagree with the one the plan's offsets were computed against.
    ///
    /// Mapping is eager so that a checkpoint which changed under a cached plan
    /// is reported before the caller has allocated anything. It costs address
    /// space, not I/O.
    explicit CheckpointSource(const PieLoaderPlan& plan) {
        const long page = ::sysconf(_SC_PAGESIZE);
        page_size_ = static_cast<std::uintptr_t>(page > 0 ? page : 1);
        files_.reserve(plan.files.len);
        try {
            for (std::size_t i = 0; i < plan.files.len; ++i) {
                const auto& decl = plan.files.ptr[i];
                if (decl.id != i) {
                    throw std::runtime_error(
                        "load plan file table entry " + std::to_string(i) +
                        " declares id " + std::to_string(decl.id) +
                        "; ids index the table and must equal their position");
                }
                files_.push_back(map_file(
                    std::string(reinterpret_cast<const char*>(decl.path.ptr),
                                decl.path.len),
                    decl.size_bytes));
            }
        } catch (...) {
            close_all();
            throw;
        }
    }

    ~CheckpointSource() { close_all(); }

    CheckpointSource(const CheckpointSource&) = delete;
    CheckpointSource& operator=(const CheckpointSource&) = delete;
    CheckpointSource(CheckpointSource&& other) noexcept
        : files_(std::move(other.files_)), page_size_(other.page_size_) {
        other.files_.clear();
    }
    CheckpointSource& operator=(CheckpointSource&& other) noexcept {
        if (this != &other) {
            close_all();
            files_ = std::move(other.files_);
            page_size_ = other.page_size_;
            other.files_.clear();
        }
        return *this;
    }

    std::size_t file_count() const noexcept { return files_.size(); }

    /// The mapped bytes for a span the plan named.
    ///
    /// This class maps; it does not move bytes to a device. The CUDA reader used
    /// to own two `cudaMemcpy` wrappers as well, which put a device call behind
    /// a name that reads like a file operation. The copy belongs with the code
    /// that owns the stream it runs on.
    const std::uint8_t* storage_host_ptr(
        std::uint32_t file_id,
        std::uint64_t file_offset,
        std::uint64_t span_bytes) const {
        return checked_file(file_id, file_offset, span_bytes).data + file_offset;
    }

    /// The whole mapping for one plan file: its base and its length.
    ///
    /// Every other accessor here hands out a span the plan named, because that
    /// is what a reader wants. A consumer that means to hand the mapping ITSELF
    /// to a device needs the base and the length instead -- Metal's no-copy
    /// buffer takes a page-aligned pointer and a length, and one buffer per
    /// file is the difference between one allocation and one per tensor.
    ///
    /// Returns `{nullptr, 0}` for an unknown or empty file rather than
    /// throwing: "this file cannot be mapped in place" is an answer a caller
    /// acts on, not an error.
    std::pair<const std::uint8_t*, std::size_t> mapped_file(
        std::uint32_t file_id) const noexcept {
        if (file_id >= files_.size()) return {nullptr, 0};
        const File& file = files_[file_id];
        return {file.data, file.mapped_size};
    }

    /// Ask the kernel to start faulting a span in.
    ///
    /// Advisory in both directions: it may do nothing, and a reader that skips
    /// it is still correct, only later. It exists because the one thing the
    /// page cache cannot work out for itself is which span comes next --
    /// streaming reads experts in an order a router picks at runtime, and no
    /// readahead heuristic will guess it. A caller that knows says so.
    void advise_will_need(
        std::uint32_t file_id,
        std::uint64_t file_offset,
        std::uint64_t span_bytes) const noexcept {
        if (span_bytes == 0 || file_id >= files_.size()) return;
        const File& file = files_[file_id];
        if (file.data == nullptr) return;
        if (file_offset > file.mapped_size ||
            span_bytes > file.mapped_size - file_offset) {
            return;
        }
        // Page-align down; madvise wants a page boundary and rounds the length
        // up itself.
        const auto start =
            reinterpret_cast<std::uintptr_t>(file.data + file_offset);
        const std::uintptr_t from = start - start % page_size_;
        (void)::madvise(reinterpret_cast<void*>(from),
                        static_cast<std::size_t>(span_bytes + (start - from)),
                        MADV_WILLNEED);
    }

    /// Read a span into a caller-owned host buffer.
    ///
    /// `pread` rather than `memcpy` from the mapping: the caller is a pinned
    /// staging lane, and letting the kernel write straight into pinned memory
    /// avoids faulting the whole span into the page cache first.
    void read_storage_bytes_to_host(
        std::uint32_t file_id,
        std::uint64_t file_offset,
        std::uint64_t span_bytes,
        void* dst) const {
        if (dst == nullptr && span_bytes != 0) {
            throw std::runtime_error("checkpoint host destination is null");
        }
        const File& file = checked_file(file_id, file_offset, span_bytes);
        auto* output = static_cast<std::uint8_t*>(dst);
        std::uint64_t done = 0;
        while (done < span_bytes) {
            const std::size_t chunk = static_cast<std::size_t>(
                std::min<std::uint64_t>(
                    span_bytes - done, std::numeric_limits<std::size_t>::max()));
            const ssize_t got = ::pread(
                file.fd, output + done, chunk,
                static_cast<off_t>(file_offset + done));
            if (got < 0 && errno == EINTR) continue;
            if (got <= 0) {
                throw std::runtime_error(
                    got == 0 ? "checkpoint short pread"
                             : std::string("checkpoint pread failed: ") +
                                   std::strerror(errno));
            }
            done += static_cast<std::uint64_t>(got);
        }
    }

    /// Hand a span of the mapping to `read`, then release it from the page
    /// cache once.
    ///
    /// `copy_storage_bytes` is the right shape for a whole tensor and the wrong
    /// one for a selection: splitting gpt-oss's fused expert projections copies
    /// millions of 1.4 KB runs, and a `madvise` per run costs a syscall each
    /// while a `memcpy` that size costs nothing. This lets a caller take all of
    /// them from the mapping and pay for the release exactly once — which is
    /// also what makes the runs parallelizable, since nothing in `read` has to
    /// serialize on the kernel.
    template <typename Read>
    void with_mapped_span(
        std::uint32_t file_id,
        std::uint64_t file_offset,
        std::uint64_t span_bytes,
        const Read& read) const {
        const File& file = checked_file(file_id, file_offset, span_bytes);
        const std::uint8_t* base = file.data + file_offset;
        read(base);
        const auto start = reinterpret_cast<std::uintptr_t>(base);
        const auto end = start + span_bytes;
        const std::uintptr_t from = start - start % page_size_;
        const std::uintptr_t to = end + (page_size_ - end % page_size_) % page_size_;
        if (to > from) {
            (void)::madvise(
                reinterpret_cast<void*>(from), static_cast<std::size_t>(to - from),
                MADV_DONTNEED);
        }
    }

    /// Copy a span out of the mapping in `max_tile_bytes` chunks, releasing each
    /// chunk from the page cache behind it.
    ///
    /// The releasing is the point. On unified memory the destination is host
    /// memory too, so without it a load holds both the checkpoint and the model
    /// resident and doubles the peak. `max_tile_bytes == 0` means one tile.
    void copy_storage_bytes(
        std::uint32_t file_id,
        std::uint64_t file_offset,
        std::uint64_t bytes,
        void* destination,
        std::uint64_t max_tile_bytes) const {
        if (bytes != 0 && destination == nullptr) {
            throw std::runtime_error("checkpoint host destination is null");
        }
        const File& file = checked_file(file_id, file_offset, bytes);
        const std::uint64_t tile = max_tile_bytes == 0
            ? std::max<std::uint64_t>(1, bytes)
            : std::max<std::uint64_t>(1, max_tile_bytes);
        const std::uint8_t* source = file.data + file_offset;
        auto* dest = static_cast<std::uint8_t*>(destination);
        for (std::uint64_t copied = 0; copied < bytes;) {
            const std::uint64_t chunk = std::min(tile, bytes - copied);
            std::memcpy(
                dest + copied, source + copied, static_cast<std::size_t>(chunk));
            const std::uint64_t next = copied + chunk;
            // Whole pages only, and the trailing partial page only when this was
            // the last tile — a page still holding bytes the next tile will read
            // must not be dropped.
            const auto start = reinterpret_cast<std::uintptr_t>(source + copied);
            const auto end = reinterpret_cast<std::uintptr_t>(source + next);
            const std::uintptr_t from = start - start % page_size_;
            const std::uintptr_t to = next == bytes
                ? end + (page_size_ - end % page_size_) % page_size_
                : end - end % page_size_;
            if (to > from) {
                (void)::madvise(
                    reinterpret_cast<void*>(from),
                    static_cast<std::size_t>(to - from), MADV_DONTNEED);
            }
            copied = next;
        }
    }

  private:
    struct File {
        std::string path;
        int fd = -1;
        std::size_t mapped_size = 0;
        const std::uint8_t* data = nullptr;
    };

    static File map_file(std::string path, std::uint64_t declared_size) {
        File file;
        file.path = std::move(path);
        file.fd = ::open(file.path.c_str(), O_RDONLY);
        if (file.fd < 0) {
            throw std::runtime_error(
                "open(" + file.path + ") failed: " + std::strerror(errno));
        }
        struct stat st {};
        if (::fstat(file.fd, &st) != 0) {
            ::close(file.fd);
            throw std::runtime_error("fstat(" + file.path + ") failed");
        }
        // Every offset in the plan is a position inside the bytes that were
        // here when the plan was compiled. A file that changed size is a
        // different file, and reading it would silently produce garbage.
        if (static_cast<std::uint64_t>(st.st_size) != declared_size) {
            ::close(file.fd);
            throw std::runtime_error(
                file.path + " is " + std::to_string(st.st_size) +
                " bytes but the load plan was compiled against " +
                std::to_string(declared_size) +
                " bytes; the checkpoint changed after the plan was built");
        }
        file.mapped_size = static_cast<std::size_t>(st.st_size);
        // `MAP_PRIVATE` because the mapping is read-only in both senses: nothing
        // writes through it, and nothing should be able to. The fd stays open
        // for `read_storage_bytes_to_host`.
        void* mapping = file.mapped_size == 0
            ? nullptr
            : ::mmap(nullptr, file.mapped_size, PROT_READ, MAP_PRIVATE, file.fd, 0);
        if (mapping == MAP_FAILED) {
            ::close(file.fd);
            throw std::runtime_error(
                "mmap(" + file.path + ") failed: " + std::strerror(errno));
        }
        if (mapping != nullptr) {
            (void)::madvise(mapping, file.mapped_size, MADV_SEQUENTIAL);
        }
        file.data = static_cast<const std::uint8_t*>(mapping);
        return file;
    }

    const File& checked_file(
        std::uint32_t file_id,
        std::uint64_t file_offset,
        std::uint64_t span_bytes) const {
        if (file_id >= files_.size()) {
            throw std::runtime_error(
                "checkpoint file id " + std::to_string(file_id) +
                " is out of range; the plan declares " +
                std::to_string(files_.size()) + " files");
        }
        const File& file = files_[file_id];
        if (file_offset > file.mapped_size ||
            span_bytes > file.mapped_size - file_offset) {
            throw std::runtime_error(
                "checkpoint byte range exceeds " + file.path);
        }
        return file;
    }

    void close_all() noexcept {
        for (auto& file : files_) {
            if (file.data != nullptr && file.mapped_size != 0) {
                ::munmap(const_cast<std::uint8_t*>(file.data), file.mapped_size);
            }
            if (file.fd >= 0) ::close(file.fd);
        }
        files_.clear();
    }

    std::vector<File> files_;
    std::uintptr_t page_size_ = 1;
};

}  // namespace pie_loader
