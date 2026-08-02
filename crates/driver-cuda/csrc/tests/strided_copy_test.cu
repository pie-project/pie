// copy_strided_extent_to_device: the non-compact ExtentWrite path.
//
// Twenty-one percent of the writes in the golden plans are strided, and they
// are the row-parallel tensors -- every layer's `o_proj` and `down_proj` under
// tp > 1 -- so this is a routine path, not an exotic one. It had no test.
//
// It now has two implementations: a cudaMemcpy2D for a pitched rectangle with
// rows wide enough to amortise the per-row DMA descriptor, and a host gather
// for everything else. The point of this test is that the choice is invisible:
// each case below is checked against an independent reference gather, so a
// wrong predicate shows up as wrong bytes rather than as a slow load.

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include <unistd.h>

#include <cuda_runtime.h>

#include "loader/strided_copy.hpp"

namespace lp = pie_loader;

namespace {

int failures = 0;

void check(bool ok, const std::string& what) {
    if (!ok) {
        std::fprintf(stderr, "strided_copy_test: FAIL %s\n", what.c_str());
        ++failures;
    }
}

/// The file the extents below are read out of, written once and mmapped by
/// `CheckpointSource` exactly as a checkpoint would be.
struct ScratchFile {
    std::string path;
    std::vector<std::uint8_t> bytes;

    explicit ScratchFile(std::size_t size) : bytes(size) {
        for (std::size_t i = 0; i < size; ++i) {
            bytes[i] = static_cast<std::uint8_t>((i * 31 + (i >> 8) * 7) & 0xff);
        }
        path = "/tmp/pie_strided_copy_test_" + std::to_string(::getpid()) + ".bin";
        std::FILE* f = std::fopen(path.c_str(), "wb");
        if (f == nullptr) {
            std::fprintf(stderr, "strided_copy_test: cannot open %s\n",
                         path.c_str());
            std::exit(1);
        }
        std::fwrite(bytes.data(), 1, bytes.size(), f);
        std::fclose(f);
    }

    ~ScratchFile() { ::unlink(path.c_str()); }
};

/// What the extent denotes, computed independently of the code under test.
std::vector<std::uint8_t> reference_gather(
    const std::vector<std::uint8_t>& file,
    std::uint64_t base_offset,
    std::uint64_t element_bytes,
    const std::vector<lp::PieLoaderDimSpecView>& dims) {
    std::uint64_t elements = 1;
    for (const auto& d : dims) {
        elements *= static_cast<std::uint64_t>(d.count);
    }
    std::vector<std::uint8_t> out(elements * element_bytes);
    for (std::uint64_t linear = 0; linear < elements; ++linear) {
        std::uint64_t remaining = linear;
        std::uint64_t offset = 0;
        for (std::size_t a = dims.size(); a > 0; --a) {
            const std::uint64_t count = static_cast<std::uint64_t>(dims[a - 1].count);
            offset += (remaining % count) *
                      static_cast<std::uint64_t>(dims[a - 1].src_stride);
            remaining /= count;
        }
        std::memcpy(out.data() + linear * element_bytes,
                    file.data() + base_offset + offset, element_bytes);
    }
    return out;
}

void run_case(lp::CheckpointSource& source,
              const ScratchFile& file,
              const std::string& name,
              std::uint64_t base_offset,
              std::uint32_t element_bytes,
              std::vector<lp::PieLoaderDimSpecView> dims) {
    const std::vector<std::uint8_t> expected =
        reference_gather(file.bytes, base_offset, element_bytes, dims);

    lp::PieLoaderSourceExtentView src{};
    src.file_id = 0;
    src.tensor_id = 0;
    src.file_offset = 0;
    src.span_bytes = expected.size();
    src.stride.base_offset = base_offset;
    src.stride.element_bytes = element_bytes;
    src.stride.dims.ptr = dims.data();
    src.stride.dims.len = dims.size();

    void* dev = nullptr;
    // A zero-byte extent is a legal shape and still must not write anything,
    // but `cudaMalloc(0)` need not hand back a usable pointer, so give it a
    // byte to aim at.
    const std::size_t capacity = expected.empty() ? 1 : expected.size();
    if (cudaMalloc(&dev, capacity) != cudaSuccess) {
        std::fprintf(stderr, "strided_copy_test: cudaMalloc failed\n");
        std::exit(1);
    }
    // Poison the destination so a copy that writes nothing cannot pass.
    cudaMemset(dev, 0xAB, capacity);

    pie_cuda_driver::copy_strided_extent_to_device(source, src, dev,
                                                   expected.size());

    std::vector<std::uint8_t> got(expected.size());
    if (!got.empty()) {
        cudaMemcpy(got.data(), dev, got.size(), cudaMemcpyDeviceToHost);
    }
    cudaFree(dev);

    check(got == expected, name);
}

}  // namespace

int main() {
    ScratchFile file(1u << 20);

    lp::PieLoaderCheckpointFileView decl{};
    decl.id = 0;
    decl.path.ptr = reinterpret_cast<const std::uint8_t*>(file.path.data());
    decl.path.len = file.path.size();
    decl.size_bytes = file.bytes.size();

    lp::PieLoaderPlan plan{};
    plan.files.ptr = &decl;
    plan.files.len = 1;
    lp::CheckpointSource source(plan);

    // Wide rows: the cudaMemcpy2D branch. This is `down_proj` under tp, scaled
    // down -- a column band of a row-major matrix.
    run_case(source, file, "wide rows take the 2D path and land correctly", 0,
             704, {{256, 1408, 704}});

    // Exactly at the measured crossover, and one byte either side of it, so a
    // predicate that is off by one is caught rather than merely made slower.
    run_case(source, file, "a 16-byte row (the crossover) is correct", 64, 16,
             {{512, 48, 16}});
    run_case(source, file, "a 15-byte row (just under) is correct", 64, 15,
             {{512, 48, 15}});
    run_case(source, file, "a 17-byte row (just over) is correct", 64, 17,
             {{512, 48, 17}});

    // Narrow rows: the host-gather branch. This is GPT-OSS's `gate_up_proj`
    // bias, whose contract reads every other bf16 element.
    run_case(source, file, "a 2-byte stride-2 gather is correct", 0, 2,
             {{4096, 4, 2}});

    // Multi-dimensional: no rectangle exists, so the gather must be used
    // whatever the row width. This is the shape the tp GPT-OSS golden carries.
    run_case(source, file, "a two-dimensional extent is correct", 0, 2,
             {{2, 512, 128}, {64, 4, 2}});
    run_case(source, file, "a two-dimensional extent with wide rows is correct",
             0, 64, {{8, 4096, 512}, {8, 256, 64}});

    // A pitch narrower than the row means the rows overlap, which is not a
    // rectangle any 2D copy can express. The gather still denotes something,
    // and that is what must come back.
    run_case(source, file, "overlapping rows fall back to the gather", 0, 64,
             {{128, 32, 64}});

    // A single row, and an empty extent: the degenerate shapes.
    run_case(source, file, "a single wide row is correct", 128, 4096,
             {{1, 8192, 4096}});
    run_case(source, file, "a zero-count extent copies nothing", 0, 64,
             {{0, 128, 64}});

    if (failures != 0) {
        std::fprintf(stderr, "strided_copy_test: %d failure(s)\n", failures);
        return 1;
    }
    std::puts("strided_copy_test: OK");
    return 0;
}
