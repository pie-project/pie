// What a pack on disk is allowed to be trusted for.
//
// The pack exists so a later run can map bytes instead of copying them, which
// means every one of these checks is standing between a mapping and the wrong
// weights. A stale pack does not crash: it reads as plausible numbers, which
// is the failure mode that costs a week.

#include <pie_loader/stream_pack.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <string>
#include <fcntl.h>
#include <sys/stat.h>
#include <unistd.h>

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

/// A scratch directory that cleans up after itself.
class TempDir {
  public:
    TempDir() {
        const char* base = std::getenv("TMPDIR");
        path_ = std::string(base != nullptr && base[0] != '\0' ? base : "/tmp") +
                "/pie_stream_pack_test." + std::to_string(::getpid());
        ::mkdir(path_.c_str(), 0755);
    }
    ~TempDir() {
        const std::string cmd = "rm -rf '" + path_ + "'";
        if (std::system(cmd.c_str()) != 0) {
            std::cerr << "note: could not clean up " << path_ << "\n";
        }
    }
    const std::string& path() const { return path_; }

  private:
    std::string path_;
};

pie_loader::StreamPackLayout two_buffers() {
    return pie_loader::stream_pack_layout({
        {1, "layers.0.experts.gate_up", 4096, 0},
        {2, "layers.0.experts.down", 100, 0},
    });
}

void fill(pie_loader::StreamPack& pack, const pie_loader::StreamPackLayout& layout,
          std::uint8_t value) {
    for (const auto& e : layout.entries) {
        std::memset(static_cast<std::uint8_t*>(pack.base()) + e.offset, value,
                    static_cast<std::size_t>(e.bytes));
    }
}

bool all_equal(const pie_loader::StreamPack& pack,
               const pie_loader::StreamPackLayout& layout, std::uint8_t value) {
    for (const auto& e : layout.entries) {
        const auto* p = static_cast<const std::uint8_t*>(pack.base()) + e.offset;
        for (std::uint64_t i = 0; i < e.bytes; ++i) {
            if (p[i] != value) return false;
        }
    }
    return true;
}

/// Every entry starts on a page, and the pack is exactly as long as that needs.
void test_the_layout_puts_every_entry_on_a_page() {
    const auto layout = two_buffers();
    const std::uint64_t page = static_cast<std::uint64_t>(::sysconf(_SC_PAGESIZE));
    for (const auto& e : layout.entries) {
        check(e.offset % page == 0, "entry '" + e.name + "' begins on a page");
    }
    check(layout.entries[0].offset == 0, "the first entry begins at zero");
    check(layout.entries[1].offset == 4096 || layout.entries[1].offset == page,
          "the second entry begins after the first, rounded up to a page");
    check(layout.total_bytes >= 4096 + 100, "the pack holds both entries");
    check(layout.total_bytes % page == 0, "the pack is a whole number of pages");
}

/// A pack written by one run is mapped by the next, with its bytes intact.
void test_a_committed_pack_is_reused() {
    TempDir dir;
    const auto layout = two_buffers();

    auto first = pie_loader::StreamPack::open(dir.path(), "planA", layout);
    check(first.valid(), "the pack is created");
    check(first.needs_fill(), "a pack that did not exist wants filling");
    fill(first, layout, 0x5a);
    check(first.commit(), "a filled pack commits");

    auto second = pie_loader::StreamPack::open(dir.path(), "planA", layout);
    check(second.valid(), "the committed pack reopens");
    check(!second.needs_fill(), "a committed pack does not want filling again");
    check(all_equal(second, layout, 0x5a), "the reopened pack holds what was written");
}

/// The failure this replaces. A run that dies after the file is sized but
/// before its bytes are there must not be mistaken for a finished pack --
/// which is exactly what a length check alone does, since the file is created
/// at its full length.
void test_an_uncommitted_pack_is_not_reused() {
    TempDir dir;
    const auto layout = two_buffers();
    {
        auto abandoned = pie_loader::StreamPack::open(dir.path(), "planA", layout);
        check(abandoned.valid(), "the pack is created");
        fill(abandoned, layout, 0x11);
        // No commit: the process "dies" here.
    }
    auto next = pie_loader::StreamPack::open(dir.path(), "planA", layout);
    check(next.valid(), "a pack is available after an abandoned attempt");
    check(next.needs_fill(), "the abandoned attempt is not mistaken for a finished pack");
}

/// A pack truncated or grown behind our back is refused rather than mapped
/// short.
void test_a_resized_pack_is_refused() {
    TempDir dir;
    const auto layout = two_buffers();
    std::string path;
    {
        auto pack = pie_loader::StreamPack::open(dir.path(), "planA", layout);
        fill(pack, layout, 0x22);
        check(pack.commit(), "a filled pack commits");
        path = pack.path();
    }
    check(::truncate(path.c_str(), 64) == 0, "the pack can be truncated for the test");

    auto next = pie_loader::StreamPack::open(dir.path(), "planA", layout);
    check(next.valid(), "a pack is still available");
    check(next.needs_fill(), "a resized pack is rebuilt rather than mapped");
}

/// The key covers the contents. Two runs that pack different subsets of the
/// same plan must not collide -- this is what a plan-key-only name gets wrong,
/// and the symptom is a mapping full of somebody else's weights.
void test_a_different_selection_is_a_different_pack() {
    TempDir dir;
    const auto whole = two_buffers();
    const auto subset = pie_loader::stream_pack_layout({
        {1, "layers.0.experts.gate_up", 4096, 0},
    });
    check(whole.digest != subset.digest, "a different selection digests differently");

    auto first = pie_loader::StreamPack::open(dir.path(), "planA", whole);
    fill(first, whole, 0x33);
    check(first.commit(), "the whole-selection pack commits");

    auto second = pie_loader::StreamPack::open(dir.path(), "planA", subset);
    check(second.valid(), "the subset pack is available");
    check(second.needs_fill(),
          "the subset does not load the whole selection's pack under the same plan key");
    check(second.path() != first.path(), "the two selections name different files");
}

/// The same fact one level down: a tensor that changed size is a different
/// arrangement even when every name is the same.
void test_a_changed_size_is_a_different_pack() {
    const auto a = pie_loader::stream_pack_layout({{1, "w", 4096, 0}});
    const auto b = pie_loader::stream_pack_layout({{1, "w", 8192, 0}});
    check(a.digest != b.digest, "a changed size digests differently");
}

/// Two processes packing the same thing at once. Each writes its own
/// temporary and publishes by rename, so the loser's bytes are discarded
/// rather than interleaved with the winner's.
void test_concurrent_creators_do_not_interleave() {
    TempDir dir;
    const auto layout = two_buffers();

    auto a = pie_loader::StreamPack::open(dir.path(), "planA", layout);
    auto b = pie_loader::StreamPack::open(dir.path(), "planA", layout);
    check(a.valid() && b.valid(), "both creators get a mapping");
    check(a.needs_fill() && b.needs_fill(), "neither sees a finished pack");
    check(a.path() == b.path(), "both intend to publish the same pack");

    fill(a, layout, 0xaa);
    fill(b, layout, 0xbb);
    check(a.commit(), "the first publishes");
    check(b.commit(), "the second publishes over it");

    auto reader = pie_loader::StreamPack::open(dir.path(), "planA", layout);
    check(reader.valid() && !reader.needs_fill(), "a finished pack is there");
    check(all_equal(reader, layout, 0xaa) || all_equal(reader, layout, 0xbb),
          "the pack holds one writer's bytes throughout, not a mixture");
}

/// A root that does not exist is a fallback, not a crash: the caller copies.
void test_an_unusable_root_is_reported() {
    const auto layout = two_buffers();
    auto pack = pie_loader::StreamPack::open("/nonexistent-pie-pack-root", "planA", layout);
    check(!pack.valid(), "a pack under an unusable root is invalid rather than fatal");
}

/// A file of exactly the right length at exactly the right path, whose bytes
/// were never vouched for. This is what a length check alone accepts, and what
/// the trailing header is for: a pack written by an older scheme, a copy that
/// stopped early, or anything else that happens to be that long.
void test_a_file_of_the_right_size_without_a_header_is_refused() {
    TempDir dir;
    const auto layout = two_buffers();
    std::string path;
    {
        auto pack = pie_loader::StreamPack::open(dir.path(), "planA", layout);
        check(pack.valid(), "the pack is created");
        path = pack.path();
        fill(pack, layout, 0x44);
        check(pack.commit(), "a filled pack commits");
    }
    // Same length, no header: zero the trailer the way a direct-write scheme
    // that never had one would have left it.
    {
        const int fd = ::open(path.c_str(), O_WRONLY);
        check(fd >= 0, "the pack can be reopened for the test");
        struct stat st {};
        check(::fstat(fd, &st) == 0, "the pack can be stat'd");
        const off_t trailer = static_cast<off_t>(layout.total_bytes);
        std::uint8_t zeros[24] = {0};
        check(::pwrite(fd, zeros, sizeof(zeros), trailer) == ssize_t(sizeof(zeros)),
              "the header can be cleared for the test");
        ::close(fd);
    }
    auto next = pie_loader::StreamPack::open(dir.path(), "planA", layout);
    check(next.valid(), "a pack is still available");
    check(next.needs_fill(), "a headerless file of the right size is rebuilt, not mapped");
}

/// A pack from an older format version is rebuilt rather than reinterpreted.
void test_a_pack_from_another_version_is_refused() {
    TempDir dir;
    const auto layout = two_buffers();
    std::string path;
    {
        auto pack = pie_loader::StreamPack::open(dir.path(), "planA", layout);
        path = pack.path();
        fill(pack, layout, 0x55);
        check(pack.commit(), "a filled pack commits");
    }
    {
        const int fd = ::open(path.c_str(), O_WRONLY);
        check(fd >= 0, "the pack can be reopened for the test");
        const std::uint32_t older = pie_loader::StreamPack::kVersion + 1;
        const off_t version_at = static_cast<off_t>(layout.total_bytes) + 8;
        check(::pwrite(fd, &older, sizeof(older), version_at) == ssize_t(sizeof(older)),
              "the version can be changed for the test");
        ::close(fd);
    }
    auto next = pie_loader::StreamPack::open(dir.path(), "planA", layout);
    check(next.needs_fill(), "a pack of another version is rebuilt, not reinterpreted");
}

}  // namespace

int main() {
    test_the_layout_puts_every_entry_on_a_page();
    test_a_committed_pack_is_reused();
    test_an_uncommitted_pack_is_not_reused();
    test_a_resized_pack_is_refused();
    test_a_different_selection_is_a_different_pack();
    test_a_changed_size_is_a_different_pack();
    test_concurrent_creators_do_not_interleave();
    test_a_file_of_the_right_size_without_a_header_is_refused();
    test_a_pack_from_another_version_is_refused();
    test_an_unusable_root_is_reported();

    if (failures != 0) {
        std::cerr << failures << " check(s) failed\n";
        return 1;
    }
    std::cout << "stream_pack: all checks passed\n";
    return 0;
}
