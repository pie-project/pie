// The materialized-weight artifact, against the two claims it makes.
//
// The artifact is a snapshot of finished device weights, written after the
// first load so a later boot can skip materializing them again. Nothing here is
// validated by the code that uses it, since a store restored from a corrupt
// file and a store restored from a good one both just look like a store.
//
// Two claims, and they are independent:
//
//   1. Round trip. What comes back is what went in -- every tensor's bytes, its
//      spec, and the view relationships between them. A view is a tensor that
//      points into another's buffer, so a codec that restored views as copies
//      would pass any per-tensor byte check and still double the memory.
//
//   2. Every payload begins on a page. This is what makes the artifact
//      mappable: a payload on a page of its own can be handed to the device as
//      a mapping of the file rather than a copy into device memory, which is
//      what lets a weight be paged in on demand. It has to be its own page --
//      two payloads sharing one means faulting either drags in the other.
//
// The second claim is about arithmetic that no caller can observe, which is
// exactly the kind that rots. It is checked here against the file offsets the
// artifact reports, and against a page size stated here rather than borrowed
// from whatever the writer chose -- so a writer that dropped to 4 KiB would
// fail this test instead of agreeing with itself.

#include <algorithm>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <iostream>
#include <string>
#include <utility>
#include <vector>

#include <unistd.h>

#include <cuda_runtime.h>

#include "loader/staged_h2d.hpp"
#include "loader/weight_store_codec.hpp"
#include "model/weight_store.hpp"
#include "tensor.hpp"

namespace {

using pie_cuda_driver::DeviceTensor;
using pie_cuda_driver::DType;

int failures = 0;

void check(bool ok, const std::string& what)
{
    if (!ok) {
        std::cerr << "FAIL: " << what << "\n";
        ++failures;
    }
}

DeviceTensor filled(DType dtype, std::vector<std::int64_t> shape, std::uint8_t byte)
{
    DeviceTensor t = DeviceTensor::allocate(dtype, shape);
    std::vector<std::uint8_t> host(t.nbytes(), byte);
    cudaMemcpy(t.data(), host.data(), host.size(), cudaMemcpyHostToDevice);
    return t;
}

std::vector<std::uint8_t> read_back(const DeviceTensor& t)
{
    std::vector<std::uint8_t> host(t.nbytes());
    cudaMemcpy(host.data(), t.data(), host.size(), cudaMemcpyDeviceToHost);
    return host;
}

// Sizes that are deliberately not multiples of the alignment, so padding is
// exercised: an artifact whose tensors happened to be page-sized would pass an
// alignment check without the writer doing anything.
//
// `q` is a window into `arena` rather than a buffer of its own, which is the
// relationship the round trip has to preserve.
pie_cuda_driver::WeightStore build_store()
{
    using namespace pie_cuda_driver;
    WeightStore store;
    WeightStoreBuilder b(store);
    b.insert("a", filled(DType::BF16, {3, 101}, 0xA1));
    b.insert("b", filled(DType::BF16, {7, 13}, 0xB2));
    b.insert("c", filled(DType::FP32, {5}, 0xC3));

    DeviceTensor arena = filled(DType::UINT8, {4096}, 0xD4);
    void* window = static_cast<std::uint8_t*>(arena.data()) + 1024;
    TensorDecl arena_spec;
    arena_spec.name = "arena";
    arena_spec.dtype = DType::UINT8;
    arena_spec.shape = {4096};
    arena_spec.ownership = TensorOwnershipKind::Owned;

    TensorDecl view_spec;
    view_spec.name = "q";
    view_spec.dtype = DType::UINT8;
    view_spec.shape = {512};
    view_spec.ownership = TensorOwnershipKind::BorrowedView;
    view_spec.backing_tensor = "arena";

    b.insert("arena", std::move(arena), arena_spec);
    b.insert("q", DeviceTensor::view(window, DType::UINT8, {512}), view_spec);
    b.finalize();
    return store;
}

// What a mapping needs, stated here rather than read from the writer. If this
// asked the codec for its alignment, a change to 64 would leave every assertion
// true and every payload sharing a page with its neighbours, which is the
// failure this file exists to prevent.
constexpr std::uint64_t kPageBytes = 16384;

std::filesystem::path scratch()
{
    return std::filesystem::temp_directory_path() /
           ("pie_weight_codec_test_" + std::to_string(::getpid()));
}

}  // namespace

int main()
{
    using namespace pie_cuda_driver;

    if (cudaSetDevice(0) != cudaSuccess) {
        std::cerr << "no CUDA device; skipping\n";
        return 0;
    }

    const std::filesystem::path dir = scratch();
    std::error_code ec;
    std::filesystem::remove_all(dir, ec);
    std::filesystem::create_directories(dir, ec);
    const auto path = dir / "weights.zt";

    const std::string key = "test-key";
    {
        WeightStore store = build_store();
        check(weight_codec::serialize_weight_store(store, key, path),
              "an ordinary store serializes");
    }
    check(std::filesystem::exists(path), "the artifact was published");

    // --- claim 2: everything a mapping would need to start on a page does ---
    //
    // Asked of the artifact itself through the loader's reader, which is a
    // different path from the one the driver's restore takes.
    {
        pie_loader::PieLoaderWeightStore* opened = nullptr;
        pie_loader::LoadPlanDiagnostics diags;
        const auto status = pie_loader::pie_loader_weight_store_open(
            pie_loader::borrow(path.string()), pie_loader::borrow(key), &opened,
            diags.slot());
        check(status == pie_loader::PieLoaderStatus::Ok,
              "the artifact opens: " + diags.text());
        if (opened != nullptr) {
            check(opened->tensors.len == 4, "four tensors have bytes of their own");
            check(opened->views.len == 1, "the window is recorded as a view");

            bool any_unaligned_size = false;
            // Sorted by where they landed, not by name: the tensors are
            // reported in name order and written in whatever order the store
            // iterated, so "the next one" is a question about the file.
            std::vector<std::pair<std::uint64_t, std::uint64_t>> spans;
            for (std::size_t i = 0; i < opened->tensors.len; ++i) {
                const auto& t = opened->tensors.ptr[i];
                check(t.file_offset % kPageBytes == 0,
                      "payload " + std::to_string(i) +
                          " begins on a page of its own: " +
                          std::to_string(t.file_offset));
                spans.emplace_back(t.file_offset, t.nbytes);
                any_unaligned_size |= (t.nbytes % kPageBytes) != 0;
            }
            std::sort(spans.begin(), spans.end());
            for (std::size_t i = 0; i + 1 < spans.size(); ++i) {
                check(spans[i].first + spans[i].second <= spans[i + 1].first,
                      "the payload at " + std::to_string(spans[i].first) +
                          " runs into the next");
            }
            // The premise: without padding these would not be aligned, so the
            // checks above would be measuring the tensors' sizes rather than
            // the writer.
            check(any_unaligned_size,
                  "the fixture's tensors are not page-sized, so alignment had to "
                  "be done");
            pie_loader::pie_loader_weight_store_close(opened);
        }
    }

    // --- claim 1: what comes back is what went in ---
    {
        WeightStore restored;
        WeightStoreBuilder rb(restored);
        PinnedLanePool pool(2, 8ull << 20);
        const bool ok = weight_codec::restore_weight_store(
            path, key, /*verify=*/true, rb, pool);
        check(ok, "the artifact restores");
        if (ok) {
            for (const auto& [name, byte] : std::vector<std::pair<std::string, std::uint8_t>>{
                     {"a", 0xA1}, {"b", 0xB2}, {"c", 0xC3}, {"arena", 0xD4}}) {
                const auto it = restored.find(name);
                if (it == restored.end()) {
                    check(false, "tensor " + name + " came back");
                    continue;
                }
                const auto host = read_back(it->second.tensor);
                bool same = !host.empty();
                for (const auto b : host) {
                    same &= (b == byte);
                }
                check(same, "tensor " + name + " came back with its own bytes");
            }

            // The view is a window into the arena, not a copy of one: same
            // allocation, same offset, and no memory of its own.
            const auto arena = restored.find("arena");
            const auto window = restored.find("q");
            if (arena == restored.end() || window == restored.end()) {
                check(false, "the arena and its window both came back");
            } else {
                check(!window->second.tensor.owns_memory(),
                      "the window did not come back as a copy");
                check(static_cast<const std::uint8_t*>(window->second.tensor.data()) ==
                          static_cast<const std::uint8_t*>(arena->second.tensor.data()) +
                              1024,
                      "the window came back at its own offset into the arena");
                check(window->second.spec.backing_tensor == "arena",
                      "the window still names what it was cut from");
            }
        }
    }

    // --- a file that is not this artifact is a miss, not a crash ---
    {
        WeightStore restored;
        WeightStoreBuilder rb(restored);
        PinnedLanePool pool(2, 8ull << 20);
        check(!weight_codec::restore_weight_store(path, "another-key", true, rb, pool),
              "an artifact written for another plan is refused");
    }
    {
        WeightStore restored;
        WeightStoreBuilder rb(restored);
        PinnedLanePool pool(2, 8ull << 20);
        check(!weight_codec::restore_weight_store(dir / "absent.zt", key, true, rb, pool),
              "an artifact that is not there is refused");
    }
    {
        // Truncation is the shape a partial write leaves behind, and the one
        // that would otherwise be read as data.
        const auto cut_path = dir / "cut.zt";
        std::string bytes;
        {
            std::vector<char> raw(std::filesystem::file_size(path));
            std::FILE* f = std::fopen(path.string().c_str(), "rb");
            const std::size_t n = std::fread(raw.data(), 1, raw.size(), f);
            std::fclose(f);
            bytes.assign(raw.data(), n / 2);
        }
        std::FILE* out = std::fopen(cut_path.string().c_str(), "wb");
        std::fwrite(bytes.data(), 1, bytes.size(), out);
        std::fclose(out);

        WeightStore restored;
        WeightStoreBuilder rb(restored);
        PinnedLanePool pool(2, 8ull << 20);
        check(!weight_codec::restore_weight_store(cut_path, key, true, rb, pool),
              "a truncated artifact is refused");
    }
    {
        // Corruption inside a payload: the manifest still parses, so nothing
        // but the digest can catch this. The byte to flip is the first payload's
        // own first byte -- flipping a fixed offset would land in padding, which
        // no digest covers and nothing should notice.
        std::uint64_t payload_start = 0;
        {
            pie_loader::PieLoaderWeightStore* opened = nullptr;
            pie_loader::pie_loader_weight_store_open(
                pie_loader::borrow(path.string()), pie_loader::borrow(key), &opened,
                nullptr);
            if (opened != nullptr) {
                payload_start = opened->tensors.ptr[0].file_offset;
                pie_loader::pie_loader_weight_store_close(opened);
            }
        }
        check(payload_start > 0, "the first payload has an offset to corrupt");

        const auto rotten_path = dir / "rotten.zt";
        std::filesystem::copy_file(path, rotten_path,
                                   std::filesystem::copy_options::overwrite_existing, ec);
        std::FILE* f = std::fopen(rotten_path.string().c_str(), "r+b");
        std::fseek(f, static_cast<long>(payload_start), SEEK_SET);
        const unsigned char flip = 0x5a;
        std::fwrite(&flip, 1, 1, f);
        std::fclose(f);

        WeightStore restored;
        WeightStoreBuilder rb(restored);
        PinnedLanePool pool(2, 8ull << 20);
        check(!weight_codec::restore_weight_store(rotten_path, key, true, rb, pool),
              "a corrupt payload is refused rather than loaded");
    }

    std::filesystem::remove_all(dir, ec);
    if (failures == 0) {
        std::cout << "weight_store_codec: all checks passed\n";
    }
    return failures == 0 ? 0 : 1;
}
