#pragma once

/// A page-aligned materialization of part of a load plan, on disk, reusable
/// across runs.
///
/// Streaming a weight means the GPU reads it where it lies rather than from a
/// copy the loader made. What stops that from working directly against the
/// checkpoint is alignment: safetensors packs its tensors back to back, so a
/// tensor almost never begins on a page, and a mapping can only begin on one.
/// The pack is the fix -- write the bytes once, each at a page boundary, and
/// every later run maps that file instead of copying anything.
///
/// This lives in the loader rather than in a driver because nothing about it
/// is a device decision. Which buffers can be packed is a property of the plan
/// (a buffer whose whole content is a verbatim range of a checkpoint file);
/// where they land is arithmetic on the page size; and whether a pack on disk
/// may be trusted is a question about that file alone. What a driver decides
/// is what to do with the mapping once it has one, and that stays the
/// driver's.
///
/// The three things a cache of this kind has to get right, and how:
///
///   * The key names the contents, not the producer. Two runs that pack
///     *different subsets* of the same plan must not collide, so the key
///     covers the layout -- every buffer's name, size and offset -- and not
///     just the plan's cache key. A selection rule that changes is then a key
///     that changes, rather than a stale pack that loads and reads as
///     somebody else's weights.
///
///   * A pack is complete or it is absent. Size alone cannot tell the two
///     apart: the file is created at its full size and filled afterwards, so a
///     run that dies in the middle leaves one that is exactly as long as it
///     should be and mostly zeroes. So the bytes are written to a temporary
///     and published by `rename`, which is atomic -- a reader sees the whole
///     pack or no pack.
///
///   * Two processes booting the same model race. The temporary is
///     process-unique and the rename is atomic, so the loser's work is
///     discarded rather than interleaved with the winner's.

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

namespace pie_loader {

/// One packed buffer: what the driver calls it, and where it lands.
struct StreamPackEntry {
    std::uint32_t id = 0;
    std::string name;
    std::uint64_t bytes = 0;
    std::uint64_t offset = 0;
};

/// Where everything lands, and a fingerprint of that arrangement.
struct StreamPackLayout {
    std::vector<StreamPackEntry> entries;
    std::uint64_t total_bytes = 0;
    /// Over the entries, so a pack whose selection changed gets a new name.
    std::uint64_t digest = 0;

    const StreamPackEntry* find(std::uint32_t id) const {
        for (const StreamPackEntry& e : entries) {
            if (e.id == id) return &e;
        }
        return nullptr;
    }
};

namespace stream_pack_detail {

inline std::uint64_t mix(std::uint64_t h, std::uint64_t v) {
    // FNV-1a's constants over 64-bit words. The digest only has to separate
    // arrangements, not resist anyone, so the cheapest thing that avalanches
    // is the right one.
    h ^= v;
    h *= 0x100000001b3ULL;
    return h;
}

inline std::uint64_t mix_bytes(std::uint64_t h, const void* p, std::size_t n) {
    const auto* b = static_cast<const std::uint8_t*>(p);
    for (std::size_t i = 0; i < n; ++i) h = mix(h, b[i]);
    return h;
}

inline std::uint64_t page_size() {
    const long v = ::sysconf(_SC_PAGESIZE);
    return v > 0 ? static_cast<std::uint64_t>(v) : 4096;
}

}  // namespace stream_pack_detail

/// Lay `entries` out page-aligned, in the order given, and fingerprint it.
///
/// The caller fixes the order -- it is the caller who knows whether buffer id
/// order or first-use order is the one whose locality it wants -- and this
/// only assigns offsets. Callers who want a stable pack across runs must hand
/// over a stable order, or the digest moves and the pack is rebuilt every
/// boot for no reason.
inline StreamPackLayout stream_pack_layout(std::vector<StreamPackEntry> entries) {
    const std::uint64_t page = stream_pack_detail::page_size();
    StreamPackLayout layout;
    layout.entries = std::move(entries);
    std::uint64_t at = 0;
    std::uint64_t digest = 0xcbf29ce484222325ULL;
    for (StreamPackEntry& e : layout.entries) {
        e.offset = at;
        at += ((e.bytes + page - 1) / page) * page;
        digest = stream_pack_detail::mix_bytes(digest, e.name.data(), e.name.size());
        digest = stream_pack_detail::mix(digest, e.bytes);
        digest = stream_pack_detail::mix(digest, e.offset);
    }
    layout.total_bytes = at;
    layout.digest = digest;
    return layout;
}

/// A mapped pack, either freshly created and awaiting its bytes or reopened
/// from a previous run.
///
/// The lifecycle is deliberately two-state and explicit. `needs_fill()` says
/// the caller now owns an empty, writable mapping and must produce every
/// entry's bytes; `commit()` publishes it. A caller that fills and forgets to
/// commit leaves the previous run's pack -- or no pack -- untouched, which is
/// the failure that costs a rebuild rather than the one that reads garbage.
class StreamPack {
  public:
    /// Format version. Bump when the on-disk arrangement changes meaning; an
    /// older pack then fails the header check and is rebuilt rather than
    /// misread.
    static constexpr std::uint32_t kVersion = 1;
    static constexpr std::uint64_t kMagic = 0x4b434150454950ULL;  // "PIEPACK"

    /// Open the pack for `plan_key` under `root`, creating it if what is there
    /// does not match `layout`.
    ///
    /// `root` must exist. It is a parameter rather than a policy of this file
    /// because where a cache belongs is a deployment question -- a temp
    /// directory is right for a machine whose temp is a disk and wrong for one
    /// whose temp is a small tmpfs, and the loader cannot tell which it is on.
    static StreamPack open(const std::string& root, const std::string& plan_key,
                           const StreamPackLayout& layout) {
        StreamPack pack;
        pack.path_ = root + "/" + name_for(plan_key, layout);
        pack.bytes_ = layout.total_bytes;
        if (pack.map_existing(layout)) return pack;
        pack.create_temp(layout);
        return pack;
    }

    ~StreamPack() { reset(); }

    StreamPack(const StreamPack&) = delete;
    StreamPack& operator=(const StreamPack&) = delete;
    StreamPack(StreamPack&& o) noexcept { *this = std::move(o); }
    StreamPack& operator=(StreamPack&& o) noexcept {
        if (this != &o) {
            reset();
            fd_ = o.fd_;
            base_ = o.base_;
            bytes_ = o.bytes_;
            needs_fill_ = o.needs_fill_;
            path_ = std::move(o.path_);
            temp_path_ = std::move(o.temp_path_);
            o.fd_ = -1;
            o.base_ = nullptr;
            o.bytes_ = 0;
            o.needs_fill_ = false;
        }
        return *this;
    }

    /// False when the pack could not be opened or created at all -- no room,
    /// no permission, no such directory. The caller falls back to copying.
    bool valid() const { return base_ != nullptr; }

    /// True while the mapping is writable and empty. The caller must write
    /// every entry and then `commit()`.
    bool needs_fill() const { return needs_fill_; }

    void* base() const { return base_; }
    std::uint64_t bytes() const { return bytes_; }
    const std::string& path() const { return path_; }

    /// Publish a filled pack, atomically. Returns false if publishing failed,
    /// in which case the mapping is still usable for this run but no later run
    /// will find it.
    ///
    /// The header goes down last and the rename after that, so the only file
    /// that ever appears under the real name is one whose bytes are all there.
    bool commit() {
        if (!needs_fill_ || base_ == nullptr) return false;
        Header h{};
        h.magic = kMagic;
        h.version = kVersion;
        h.total_bytes = bytes_;
        std::memcpy(static_cast<std::uint8_t*>(base_) + bytes_, &h, sizeof(h));
        if (::msync(base_, static_cast<std::size_t>(bytes_ + sizeof(Header)), MS_SYNC) != 0) {
            return false;
        }
        if (::rename(temp_path_.c_str(), path_.c_str()) != 0) {
            return false;
        }
        temp_path_.clear();
        needs_fill_ = false;
        return true;
    }

  private:
    /// Trailing, so a buffer's offset is its offset and nothing has to account
    /// for a prologue. Written after the bytes it vouches for.
    struct Header {
        std::uint64_t magic;
        std::uint32_t version;
        std::uint32_t reserved;
        std::uint64_t total_bytes;
    };

    static std::string name_for(const std::string& plan_key,
                                const StreamPackLayout& layout) {
        char suffix[32];
        std::snprintf(suffix, sizeof(suffix), "-%016llx.piepack",
                      static_cast<unsigned long long>(layout.digest));
        return plan_key + suffix;
    }

    bool map_existing(const StreamPackLayout& layout) {
        const std::uint64_t want = layout.total_bytes + sizeof(Header);
        struct stat st {};
        if (::stat(path_.c_str(), &st) != 0) return false;
        if (static_cast<std::uint64_t>(st.st_size) != want) return false;
        const int fd = ::open(path_.c_str(), O_RDONLY);
        if (fd < 0) return false;
        void* p = ::mmap(nullptr, static_cast<std::size_t>(want), PROT_READ, MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) {
            ::close(fd);
            return false;
        }
        Header h{};
        std::memcpy(&h, static_cast<std::uint8_t*>(p) + layout.total_bytes, sizeof(h));
        if (h.magic != kMagic || h.version != kVersion || h.total_bytes != layout.total_bytes) {
            ::munmap(p, static_cast<std::size_t>(want));
            ::close(fd);
            return false;
        }
        fd_ = fd;
        base_ = p;
        needs_fill_ = false;
        return true;
    }

    void create_temp(const StreamPackLayout& layout) {
        const std::uint64_t want = layout.total_bytes + sizeof(Header);
        // Unique per attempt, not just per process. Two packers in one
        // process -- two engines, two models sharing a selection -- would
        // otherwise share a temporary, and the first `rename` would move the
        // second's file out from under it, leaving the second writing into
        // what is now the published pack.
        static std::atomic<std::uint64_t> attempt{0};
        temp_path_ = path_ + ".partial." + std::to_string(::getpid()) + "." +
                     std::to_string(attempt.fetch_add(1, std::memory_order_relaxed));
        const int fd = ::open(temp_path_.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        if (fd < 0) return;
        if (::ftruncate(fd, static_cast<off_t>(want)) != 0) {
            ::close(fd);
            ::unlink(temp_path_.c_str());
            temp_path_.clear();
            return;
        }
        void* p = ::mmap(nullptr, static_cast<std::size_t>(want), PROT_READ | PROT_WRITE,
                         MAP_SHARED, fd, 0);
        if (p == MAP_FAILED) {
            ::close(fd);
            ::unlink(temp_path_.c_str());
            temp_path_.clear();
            return;
        }
        fd_ = fd;
        base_ = p;
        needs_fill_ = true;
    }

    void reset() {
        if (base_ != nullptr) ::munmap(base_, static_cast<std::size_t>(bytes_ + sizeof(Header)));
        if (fd_ >= 0) ::close(fd_);
        // An uncommitted temporary is this process's litter, not a cache.
        if (!temp_path_.empty()) ::unlink(temp_path_.c_str());
        fd_ = -1;
        base_ = nullptr;
        bytes_ = 0;
        needs_fill_ = false;
        temp_path_.clear();
    }

    StreamPack() = default;

    int fd_ = -1;
    void* base_ = nullptr;
    std::uint64_t bytes_ = 0;
    bool needs_fill_ = false;
    std::string path_;
    std::string temp_path_;
};

}  // namespace pie_loader
