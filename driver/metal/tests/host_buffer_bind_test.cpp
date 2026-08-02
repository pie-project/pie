// Can a kernel read a buffer this driver does NOT own?
//
// Weight streaming on Apple silicon rests on one primitive: wrap a mapping the
// driver did not allocate — the checkpoint's own pages — in an MTLBuffer and
// bind it. Measured outside this driver, that works: a 5.3 GB file-backed
// mapping wrapped with `newBufferWithBytesNoCopy`, committed to a residency set
// and read by a kernel at scattered offsets, returns the right bytes with the
// process at 20 MB resident. The pages are demand-faulted and clean, so the
// kernel evicts them under pressure rather than the driver paying for every
// weight, resident, forever.
//
// Inside this driver it produced zeros, and the bytes were proven identical to
// the ones the copying path stages — so what is in question is the BINDING, not
// the data. This driver binds by ADDRESS into an MTL4 argument table, which the
// outside measurement never exercised.
//
// So this asks the question directly, with no checkpoint and no model in the
// way: three buffers with the same contents — one from the heap, one wrapped
// over anonymous host memory, one wrapped over a FILE — read by the same kernel
// through the same argument table. Whatever separates them is the answer.

#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "mtl4_context.hpp"

using namespace pie::metal;

namespace {

int failures = 0;

void expect(bool ok, const std::string& what) {
    std::printf("  %s  %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

struct RowGatherParams {
    std::uint32_t width;
    std::uint32_t count;
};

/// Copy `width` bfloat16 elements out of `in` with `row_gather`, and report how
/// many came back right. The kernel is the family-agnostic one gemma4 and
/// gpt-oss both use, chosen because its whole job is to read one buffer and
/// write another — nothing else can be blamed.
int read_back(RawMetalContext& ctx, const Pso& pso, const SlotHandle& in,
              const SlotHandle& out, const SlotHandle& rows, const SlotHandle& params,
              int width, const std::vector<std::uint16_t>& want) {
    std::memset(out.contents(), 0, std::size_t(width) * 2);
    const int ord = 7000;
    ctx.arg_bind_ordinal(ord, 0, in);
    ctx.arg_bind_ordinal(ord, 1, out);
    ctx.arg_bind_ordinal(ord, 2, rows);
    ctx.arg_bind_ordinal(ord, 3, params);
    const StepTiming t = ctx.run_step([&](StepEncoder& se) {
        se.set_pso(pso);
        se.set_argtable_ordinal(ord);
        se.dispatch(Grid{std::uint32_t(width), 1, 1}, Threadgroup{64, 1, 1});
    });
    if (!t.succeeded()) return -1;
    const auto* got = static_cast<const std::uint16_t*>(out.contents());
    int right = 0;
    for (int i = 0; i < width; ++i) right += (got[i] == want[std::size_t(i)]) ? 1 : 0;
    return right;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("host_buffer_bind_test\n");

    auto ctx = RawMetalContext::create(std::size_t(64) << 20);
    if (!ctx) {
        std::printf("  SKIP (no Metal device)\n");
        return 0;
    }
    std::string err;
    const Pso pso = ctx->compile_pso_from_file(
        std::string(PIE_METAL_KERNELS_DIR_FOR_TEST) + "/row_gather.metal",
        "row_gather_bfloat16", &err);
    if (!pso.valid()) {
        std::printf("  FAIL  compiling row_gather: %s\n", err.c_str());
        return 1;
    }

    const int width = 1024;
    const std::size_t bytes = std::size_t(width) * 2;
    const std::size_t page = std::size_t(::getpagesize());
    const std::size_t span = ((bytes + page - 1) / page) * page;

    // The pattern every buffer below holds.
    std::vector<std::uint16_t> want;
    want.resize(std::size_t(width));
    for (int i = 0; i < width; ++i) want[std::size_t(i)] = std::uint16_t(0x3F00 + (i & 0xFF));

    SlotHandle out = ctx->heap_alloc(bytes);
    SlotHandle rows = ctx->heap_alloc(64);
    SlotHandle params = ctx->heap_alloc(sizeof(RowGatherParams));
    *static_cast<std::uint32_t*>(rows.contents()) = 0;
    *static_cast<RowGatherParams*>(params.contents()) = RowGatherParams{std::uint32_t(width), 1};

    // ── 1. the heap, which is what the driver does today ──
    SlotHandle heap_in = ctx->heap_alloc(bytes);
    std::memcpy(heap_in.contents(), want.data(), bytes);

    // ── 2. anonymous host memory the driver did not allocate ──
    void* anon = ::mmap(nullptr, span, PROT_READ | PROT_WRITE,
                        MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
    if (anon == MAP_FAILED) {
        std::printf("  FAIL  anonymous mmap\n");
        return 1;
    }
    std::memcpy(anon, want.data(), bytes);
    SlotHandle anon_in = ctx->wrap_host_memory(anon, span);
    // ── 3. a FILE, which is what a checkpoint is ──
    const std::string path = "/tmp/pie_metal_host_buffer_test.bin";
    {
        FILE* f = std::fopen(path.c_str(), "wb");
        if (f == nullptr) {
            std::printf("  FAIL  creating %s\n", path.c_str());
            return 1;
        }
        std::vector<std::uint8_t> pad(span, 0);
        std::memcpy(pad.data(), want.data(), bytes);
        std::fwrite(pad.data(), 1, span, f);
        std::fclose(f);
    }
    const int fd = ::open(path.c_str(), O_RDONLY);
    void* filemap = fd >= 0 ? ::mmap(nullptr, span, PROT_READ, MAP_SHARED, fd, 0) : MAP_FAILED;
    if (filemap == MAP_FAILED) {
        std::printf("  FAIL  file mmap\n");
        return 1;
    }
    SlotHandle file_in = ctx->wrap_host_memory(filemap, span);

    // Residency comes LAST, which is the order staging uses: every wrap above
    // happened while the context was not yet resident. That ordering is the
    // only thing this test does that the standalone measurement did not.
    ctx->make_resident();

    {
        const int right = read_back(*ctx, pso, heap_in, out, rows, params, width, want);
        std::printf("    heap buffer: %d/%d elements\n", right, width);
        expect(right == width, "a heap buffer reads back");
    }
    expect(anon_in.valid(), "an anonymous mapping can be wrapped");
    if (anon_in.valid()) {
        const int right = read_back(*ctx, pso, anon_in, out, rows, params, width, want);
        std::printf("    anonymous mapping: %d/%d elements (gpu_address=0x%llx)\n", right,
                    width, static_cast<unsigned long long>(anon_in.gpu_address));
        expect(right == width,
               "a wrapped anonymous mapping reads back through the argument table");
    }
    expect(file_in.valid(), "a file mapping can be wrapped");
    if (file_in.valid()) {
        const int right = read_back(*ctx, pso, file_in, out, rows, params, width, want);
        std::printf("    file mapping: %d/%d elements (gpu_address=0x%llx)\n", right, width,
                    static_cast<unsigned long long>(file_in.gpu_address));
        expect(right == width,
               "a wrapped FILE mapping reads back -- which is weight streaming");
    }

    // ── 4. a mapping the size of a real checkpoint ──
    //
    // The sizes above prove the mechanism; this proves it at the size that
    // matters. gpt-oss's expert bank is 10.75 GB, and everything smaller
    // working says nothing about whether a residency set will carry that --
    // which is exactly the gap between a measurement and a feature.
    //
    // Sparse, so it costs no disk: only the page carrying the pattern is
    // written, and it sits near the END so the slice offset is real.
    {
        const std::uint64_t big = std::uint64_t(12) << 30;
        const std::uint64_t at = ((big - (std::uint64_t(1) << 30)) / page) * page;
        const std::string bpath = "/tmp/pie_metal_host_buffer_big.bin";
        const int bfd = ::open(bpath.c_str(), O_RDWR | O_CREAT | O_TRUNC, 0644);
        bool ok = bfd >= 0 && ::ftruncate(bfd, off_t(big)) == 0;
        if (ok) {
            ok = ::pwrite(bfd, want.data(), bytes, off_t(at)) == ssize_t(bytes);
        }
        void* bmap = ok ? ::mmap(nullptr, std::size_t(big), PROT_READ, MAP_SHARED, bfd, 0)
                        : MAP_FAILED;
        if (bmap == MAP_FAILED) {
            std::printf("    (12 GB mapping unavailable, skipped)\n");
        } else {
            SlotHandle whole = ctx->wrap_host_memory(bmap, std::size_t(big));
            expect(whole.valid(), "a 12 GB file mapping can be wrapped");
            if (whole.valid()) {
                SlotHandle in = whole;
                in.contents_ptr = static_cast<std::uint8_t*>(whole.contents()) + at;
                in.gpu_address = whole.gpu_address + at;
                in.size = std::size_t(bytes);
                const int right = read_back(*ctx, pso, in, out, rows, params, width, want);
                std::printf("    12 GB mapping, read at +%.2f GB: %d/%d elements\n",
                            double(at) / 1e9, right, width);
                expect(right == width,
                       "a checkpoint-sized mapping reads back at a real offset");
            }
            ::munmap(bmap, std::size_t(big));
        }
        if (bfd >= 0) ::close(bfd);
        std::remove(bpath.c_str());
    }

    ::munmap(anon, span);
    ::munmap(filemap, span);
    if (fd >= 0) ::close(fd);
    std::remove(path.c_str());

    std::printf("\n==== host_buffer_bind_test: %s ====\n",
                failures == 0 ? "all passed" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
