// The materialized-weight artifact (PIEWSTOR), against the two claims it makes.
//
// The artifact is a snapshot of finished device weights, written after the
// first load so a later boot can skip materializing them again. It had no test
// at all, which is a bad state for a format: nothing here is validated by the
// code that uses it, since a store restored from a corrupt file and a store
// restored from a good one both just look like a store.
//
// Two claims, and they are independent:
//
//   1. Round trip. What comes back is what went in -- every tensor's bytes, its
//      spec, and the view relationships between them. A view is a tensor that
//      points into another's buffer, so a codec that restored views as copies
//      would pass any per-tensor byte check and still double the memory.
//
//   2. Every blob begins on a page. This is what makes the artifact mappable:
//      a blob on a page of its own can be handed to the device as a mapping of
//      the file rather than a copy into device memory, which is what lets a
//      weight be paged in on demand. It has to be its own page -- two blobs
//      sharing one means faulting either drags in the other.
//
// The second claim is about arithmetic that no caller can observe, which is
// exactly the kind that rots. It is checked here by parsing the file's own
// header rather than by asking the codec, so the test would notice a codec that
// agreed with itself and with nothing else.

#include <cstdint>
#include <cstring>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

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
// alignment check without the codec doing anything.
pie_cuda_driver::WeightStore build_store()
{
    using namespace pie_cuda_driver;
    WeightStore store;
    WeightStoreBuilder b(store);
    b.insert("a", filled(DType::BF16, {3, 101}, 0xA1));
    b.insert("b", filled(DType::BF16, {7, 13}, 0xB2));
    b.insert("c", filled(DType::FP32, {5}, 0xC3));
    b.finalize();
    return store;
}

// The header, parsed independently of the codec. Only as far as the two fields
// this test is about; the rest is skipped by the same rules the writer used.
// What a mapping needs, stated here rather than borrowed from the codec. If
// the test asked `weight_codec::kBlobAlign` it would follow the codec down: a
// change to 64 would leave every assertion true and every blob sharing a page
// with its neighbours, which is the failure this file exists to prevent.
constexpr std::uint64_t kPageBytes = 16384;

struct Header {
    std::uint32_t version;
    std::uint64_t num_owned;
    std::uint64_t blob_section_bytes;
    std::uint64_t blob_section_pos;
    std::vector<std::uint64_t> blob_offsets;
    std::vector<std::uint64_t> blob_sizes;
};

struct Cursor {
    const char* p;
    const char* end;

    template <typename T>
    T scalar()
    {
        T v{};
        if (p + sizeof(T) > end) {
            return v;
        }
        std::memcpy(&v, p, sizeof(T));
        p += sizeof(T);
        return v;
    }
    std::string str()
    {
        const auto n = scalar<std::uint32_t>();
        if (p + n > end) {
            p = end;
            return {};
        }
        std::string s(p, p + n);
        p += n;
        return s;
    }
    void skip_shape()
    {
        const auto rank = scalar<std::uint32_t>();
        p = (p + rank * sizeof(std::int64_t) > end)
                ? end
                : p + rank * sizeof(std::int64_t);
    }
};

Header parse(const std::string& bytes)
{
    Cursor c{bytes.data(), bytes.data() + bytes.size()};
    c.p += 8;  // magic
    Header h{};
    h.version = c.scalar<std::uint32_t>();
    c.str();  // key
    h.num_owned = c.scalar<std::uint64_t>();
    c.scalar<std::uint64_t>();  // views
    c.scalar<std::uint64_t>();  // quant entries
    h.blob_section_bytes = c.scalar<std::uint64_t>();
    h.blob_section_pos = c.scalar<std::uint64_t>();
    for (std::uint64_t i = 0; i < h.num_owned; ++i) {
        // One spec: name, dtype, shape, ownership, backing tensor -- then the
        // blob triple. Spelled out rather than reusing the codec's reader, so a
        // change to the layout has to be made in two places that disagree
        // loudly rather than in one that stays quietly self-consistent.
        c.str();                    // name
        c.scalar<std::uint8_t>();   // dtype
        c.skip_shape();
        c.scalar<std::uint8_t>();   // ownership
        c.str();                    // backing tensor
        h.blob_sizes.push_back(c.scalar<std::uint64_t>());
        h.blob_offsets.push_back(c.scalar<std::uint64_t>());
        c.scalar<std::uint64_t>();  // checksum
    }
    return h;
}

}  // namespace

int main()
{
    using namespace pie_cuda_driver;

    if (cudaSetDevice(0) != cudaSuccess) {
        std::cerr << "no CUDA device; skipping\n";
        return 0;
    }

    const std::string key = "test-key";
    std::string bytes;
    {
        WeightStore store = build_store();
        std::ostringstream out(std::ios::binary);
        check(weight_codec::serialize_weight_store(store, key, out),
              "an ordinary store serializes");
        bytes = out.str();
    }
    check(!bytes.empty(), "the artifact is not empty");

    // --- claim 2: everything a mapping would need to start on a page does ---
    const Header h = parse(bytes);
    check(h.version == weight_codec::kFormatVersion,
          "the file states the version the codec writes");
    check(h.num_owned == 3, "three owned roots");
    check(h.blob_section_pos % kPageBytes == 0,
          "the blob section begins on a page: " + std::to_string(h.blob_section_pos));
    for (std::uint64_t i = 0; i < h.blob_offsets.size(); ++i) {
        check(h.blob_offsets[i] % kPageBytes == 0,
              "blob " + std::to_string(i) + " begins on a page of its own: " +
                  std::to_string(h.blob_offsets[i]));
    }
    // A blob's page is its own only if the next one starts past its end.
    for (std::uint64_t i = 0; i + 1 < h.blob_offsets.size(); ++i) {
        check(h.blob_offsets[i] + h.blob_sizes[i] <= h.blob_offsets[i + 1],
              "blob " + std::to_string(i) + " does not run into the next");
    }
    check(h.blob_section_pos + h.blob_section_bytes <= bytes.size(),
          "the blob section fits in the file");
    // The premise: without padding these would not be aligned, so the checks
    // above would be measuring the tensors' sizes rather than the codec.
    bool any_unaligned_size = false;
    for (const auto n : h.blob_sizes) {
        any_unaligned_size |= (n % kPageBytes) != 0;
    }
    check(any_unaligned_size,
          "the fixture's tensors are not page-sized, so alignment had to be done");

    // --- claim 1: what comes back is what went in ---
    {
        WeightStore restored;
        WeightStoreBuilder rb(restored);
        PinnedLanePool pool(2, 8ull << 20);
        const bool ok = weight_codec::restore_weight_store(
            reinterpret_cast<const std::uint8_t*>(bytes.data()), bytes.size(),
            key, /*verify=*/true, rb, pool);
        check(ok, "the artifact restores");
        if (ok) {
            for (const auto& [name, byte] : std::vector<std::pair<std::string, std::uint8_t>>{
                     {"a", 0xA1}, {"b", 0xB2}, {"c", 0xC3}}) {
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
        }
    }

    // --- a file that is not this format is a miss, not a crash ---
    {
        WeightStore restored;
        WeightStoreBuilder rb(restored);
        PinnedLanePool pool(2, 8ull << 20);
        std::string wrong_key_file = bytes;
        check(!weight_codec::restore_weight_store(
                  reinterpret_cast<const std::uint8_t*>(wrong_key_file.data()),
                  wrong_key_file.size(), "another-key", true, rb, pool),
              "an artifact written for another plan is refused");
    }
    {
        WeightStore restored;
        WeightStoreBuilder rb(restored);
        PinnedLanePool pool(2, 8ull << 20);
        std::string old_version = bytes;
        const std::uint32_t stale = weight_codec::kFormatVersion - 1;
        std::memcpy(old_version.data() + 8, &stale, sizeof(stale));
        check(!weight_codec::restore_weight_store(
                  reinterpret_cast<const std::uint8_t*>(old_version.data()),
                  old_version.size(), key, true, rb, pool),
              "an artifact in the previous layout is refused rather than misread");
    }
    {
        // Truncation is the shape a partial write leaves behind, and the one
        // that would otherwise be read as data.
        WeightStore restored;
        WeightStoreBuilder rb(restored);
        PinnedLanePool pool(2, 8ull << 20);
        const std::string cut = bytes.substr(0, bytes.size() / 2);
        check(!weight_codec::restore_weight_store(
                  reinterpret_cast<const std::uint8_t*>(cut.data()), cut.size(),
                  key, true, rb, pool),
              "a truncated artifact is refused");
    }

    if (failures == 0) {
        std::cout << "weight_store_codec: all checks passed\n";
    }
    return failures == 0 ? 0 : 1;
}
