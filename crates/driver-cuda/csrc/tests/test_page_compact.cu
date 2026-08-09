// PTIR Layer 4 -- page-CSR compaction under a page mask.
//
// `compact_page_csr` is what turns Quest's `attn_page_mask` from an
// observation into an eviction: it gathers a fire's paged-KV table down to the
// pages the mask keeps, and the resulting table is what FlashInfer attends
// over. Every failure mode here is silent at runtime -- a reordered page list
// relabels keys, a dropped last page corrupts the `kv_len` identity, an empty
// list is undefined behaviour inside the attention kernel -- so this asserts
// the three invariants directly against an independent host reference rather
// than against the kernel's own arithmetic.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "attn/page_compact.hpp"

using pie_cuda_driver::kernels::attn::compact_page_csr;

namespace {

int g_fail = 0;

void rt(cudaError_t e, const char* what, int line) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "FATAL %s:%d: %s -> %s\n", __FILE__, line, what,
                     cudaGetErrorString(e));
        std::exit(2);
    }
}
#define RT(e) rt((e), #e, __LINE__)

struct Csr {
    std::vector<std::uint32_t> indices;
    std::vector<std::uint32_t> indptr;
    std::vector<std::uint32_t> last_lens;
};

// Host reference. Deliberately written as the obvious sequential loop: it
// shares no code with the kernel, so agreement is evidence and not a tautology.
Csr host_compact(const Csr& in, const std::vector<std::uint8_t>& keep) {
    const std::size_t R = in.last_lens.size();
    Csr out;
    out.indptr.assign(R + 1, 0);
    out.last_lens = in.last_lens;
    for (std::size_t r = 0; r < R; ++r) {
        const std::uint32_t beg = in.indptr[r];
        const std::uint32_t pages = in.indptr[r + 1] - beg;
        for (std::uint32_t p = 0; p < pages; ++p) {
            const bool last = (p + 1 == pages);
            if (last || keep[beg + p] != 0) {
                out.indices.push_back(in.indices[beg + p]);
            }
        }
        out.indptr[r + 1] = static_cast<std::uint32_t>(out.indices.size());
    }
    return out;
}

struct Device {
    std::uint32_t* indices = nullptr;
    std::uint32_t* indptr = nullptr;
    std::uint32_t* last_lens = nullptr;
    std::uint8_t* keep = nullptr;
    std::uint32_t* out_indices = nullptr;
    std::uint32_t* out_indptr = nullptr;
    std::uint32_t* out_last_lens = nullptr;
    std::uint32_t* counts = nullptr;
};

// The mask the driver hands the kernel is `[R, stride]`, not CSR-laid-out.
// Tests author masks in CSR terms because that is how a policy thinks about
// them ("page p of request r"); this is the translation, and `extra_stride`
// lets a test make the rows wider than any request needs -- exactly what the
// driver does when the host page CSR it sized from is a bound.
std::vector<std::uint8_t> to_strided(
    const Csr& in,
    const std::vector<std::uint8_t>& keep,
    std::uint32_t stride) {
    const int R = static_cast<int>(in.last_lens.size());
    std::vector<std::uint8_t> rows(
        static_cast<std::size_t>(R) * stride, 1);
    for (int r = 0; r < R; ++r) {
        const std::uint32_t beg = in.indptr[r];
        const std::uint32_t pages = in.indptr[r + 1] - beg;
        for (std::uint32_t p = 0; p < pages && p < stride; ++p) {
            rows[static_cast<std::size_t>(r) * stride + p] = keep[beg + p];
        }
    }
    return rows;
}

std::uint32_t widest_request(const Csr& in) {
    const int R = static_cast<int>(in.last_lens.size());
    std::uint32_t stride = 1;
    for (int r = 0; r < R; ++r) {
        stride = std::max(stride, in.indptr[r + 1] - in.indptr[r]);
    }
    return stride;
}

Csr run_device(
    const Csr& in,
    const std::vector<std::uint8_t>& keep,
    std::uint32_t extra_stride = 0) {
    const int R = static_cast<int>(in.last_lens.size());
    const std::size_t total = in.indices.size();
    const std::uint32_t stride = widest_request(in) + extra_stride;
    const std::vector<std::uint8_t> rows = to_strided(in, keep, stride);
    Device d;
    RT(cudaMalloc(&d.indices, std::max<std::size_t>(total, 1) * 4));
    RT(cudaMalloc(&d.indptr, (R + 1) * 4));
    RT(cudaMalloc(&d.last_lens, std::max(R, 1) * 4));
    RT(cudaMalloc(&d.keep, std::max<std::size_t>(rows.size(), 1)));
    RT(cudaMalloc(&d.out_indices, std::max<std::size_t>(total, 1) * 4));
    RT(cudaMalloc(&d.out_indptr, (R + 1) * 4));
    RT(cudaMalloc(&d.out_last_lens, std::max(R, 1) * 4));

    if (total != 0) {
        RT(cudaMemcpy(d.indices, in.indices.data(), total * 4,
                      cudaMemcpyHostToDevice));
    }
    if (!rows.empty()) {
        RT(cudaMemcpy(d.keep, rows.data(), rows.size(),
                      cudaMemcpyHostToDevice));
    }
    RT(cudaMemcpy(d.indptr, in.indptr.data(), (R + 1) * 4,
                  cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d.last_lens, in.last_lens.data(), R * 4,
                  cudaMemcpyHostToDevice));
    // Poison the outputs: a kernel that fails to write a slot must not pass by
    // inheriting a zero from a fresh allocation.
    RT(cudaMemset(d.out_indices, 0xEE, std::max<std::size_t>(total, 1) * 4));
    RT(cudaMemset(d.out_indptr, 0xEE, (R + 1) * 4));
    RT(cudaMemset(d.out_last_lens, 0xEE, std::max(R, 1) * 4));

    RT(cudaMalloc(&d.counts, std::max(R, 1) * 4));
    compact_page_csr(d.indices, d.indptr, d.last_lens, d.keep, d.counts,
                            stride, R, d.out_indices, d.out_indptr,
                            d.out_last_lens, nullptr);
    RT(cudaDeviceSynchronize());

    Csr got;
    got.indptr.assign(R + 1, 0);
    got.last_lens.assign(R, 0);
    RT(cudaMemcpy(got.indptr.data(), d.out_indptr, (R + 1) * 4,
                  cudaMemcpyDeviceToHost));
    RT(cudaMemcpy(got.last_lens.data(), d.out_last_lens, R * 4,
                  cudaMemcpyDeviceToHost));
    got.indices.assign(got.indptr[R], 0);
    if (got.indptr[R] != 0) {
        RT(cudaMemcpy(got.indices.data(), d.out_indices, got.indptr[R] * 4,
                      cudaMemcpyDeviceToHost));
    }

    cudaFree(d.indices);
    cudaFree(d.indptr);
    cudaFree(d.last_lens);
    cudaFree(d.keep);
    cudaFree(d.out_indices);
    cudaFree(d.out_indptr);
    cudaFree(d.out_last_lens);
    cudaFree(d.counts);
    return got;
}

bool same(const Csr& a, const Csr& b) {
    return a.indices == b.indices && a.indptr == b.indptr &&
           a.last_lens == b.last_lens;
}

void report(const char* name, bool ok) {
    std::printf("[%s] %s\n", ok ? " ok " : "FAIL", name);
    if (!ok) ++g_fail;
}

Csr make_csr(const std::vector<std::uint32_t>& pages_per_request,
             std::uint32_t page_size, std::mt19937& rng) {
    Csr in;
    in.indptr.assign(pages_per_request.size() + 1, 0);
    std::uniform_int_distribution<std::uint32_t> page_id(0, 100000);
    std::uniform_int_distribution<std::uint32_t> len(1, page_size);
    for (std::size_t r = 0; r < pages_per_request.size(); ++r) {
        for (std::uint32_t p = 0; p < pages_per_request[r]; ++p) {
            in.indices.push_back(page_id(rng));
        }
        in.indptr[r + 1] = static_cast<std::uint32_t>(in.indices.size());
        in.last_lens.push_back(pages_per_request[r] == 0 ? 0 : len(rng));
    }
    return in;
}

void check_against_reference(const char* name, const Csr& in,
                             const std::vector<std::uint8_t>& keep) {
    report(name, same(run_device(in, keep), host_compact(in, keep)));
}

// Compaction runs once per LAYER per fire, so its cost is not amortised over
// the model the way a one-shot host operation would be. `PIE_PAGE_COMPACT_BENCH`
// times it in isolation with CUDA events -- e2e wall-clock on a shared host is
// worthless for a kernel this small (see the measurement caveat in the design
// doc's section 9.8).
void bench(int requests, int pages_per_request) {
    std::mt19937 rng(0xBEEFu);
    std::vector<std::uint32_t> counts(requests, pages_per_request);
    const Csr in = make_csr(counts, 16, rng);
    const std::uint32_t stride = widest_request(in);
    const std::vector<std::uint8_t> keep(in.indices.size(), 1);
    const std::vector<std::uint8_t> rows = to_strided(in, keep, stride);

    Device d;
    const std::size_t total = in.indices.size();
    RT(cudaMalloc(&d.indices, total * 4));
    RT(cudaMalloc(&d.indptr, (requests + 1) * 4));
    RT(cudaMalloc(&d.last_lens, requests * 4));
    RT(cudaMalloc(&d.keep, rows.size()));
    RT(cudaMalloc(&d.out_indices, total * 4));
    RT(cudaMalloc(&d.out_indptr, (requests + 1) * 4));
    RT(cudaMalloc(&d.out_last_lens, requests * 4));
    RT(cudaMalloc(&d.counts, requests * 4));
    RT(cudaMemcpy(d.indices, in.indices.data(), total * 4, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d.indptr, in.indptr.data(), (requests + 1) * 4, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d.last_lens, in.last_lens.data(), requests * 4, cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d.keep, rows.data(), rows.size(), cudaMemcpyHostToDevice));

    const int iters = 200;
    for (int i = 0; i < 20; ++i) {
        compact_page_csr(d.indices, d.indptr, d.last_lens, d.keep,
                                d.counts, stride, requests, d.out_indices,
                                d.out_indptr, d.out_last_lens, nullptr);
    }
    RT(cudaDeviceSynchronize());

    cudaEvent_t beg, end;
    RT(cudaEventCreate(&beg));
    RT(cudaEventCreate(&end));
    RT(cudaEventRecord(beg, nullptr));
    for (int i = 0; i < iters; ++i) {
        compact_page_csr(d.indices, d.indptr, d.last_lens, d.keep,
                                d.counts, stride, requests, d.out_indices,
                                d.out_indptr, d.out_last_lens, nullptr);
    }
    RT(cudaEventRecord(end, nullptr));
    RT(cudaEventSynchronize(end));
    float ms = 0.f;
    RT(cudaEventElapsedTime(&ms, beg, end));
    std::printf("[bench] R=%-3d pages/req=%-6d total=%-8zu  %8.2f us/call\n",
                requests, pages_per_request, total,
                static_cast<double>(ms) / iters * 1000.0);
    cudaEventDestroy(beg);
    cudaEventDestroy(end);
    cudaFree(d.indices);
    cudaFree(d.indptr);
    cudaFree(d.last_lens);
    cudaFree(d.keep);
    cudaFree(d.out_indices);
    cudaFree(d.out_indptr);
    cudaFree(d.out_last_lens);
    cudaFree(d.counts);
}

}  // namespace

int main() {
    std::mt19937 rng(0xC0FFEEu);

    if (std::getenv("PIE_PAGE_COMPACT_BENCH") != nullptr) {
        // 128 pages = 2K context, 8192 = 128K context at page_size 16.
        for (const int pages : {128, 1024, 8192}) {
            for (const int r : {1, 8, 32}) bench(r, pages);
        }
        return 0;
    }

    // 1. All-ones mask is the identity. This is the safety floor: a fire whose
    //    program writes no useful mask must attend over exactly what it would
    //    have without the feature.
    {
        const Csr in = make_csr({3, 1, 7, 12}, 16, rng);
        std::vector<std::uint8_t> keep(in.indices.size(), 1);
        const Csr got = run_device(in, keep);
        report("all-ones mask is the identity", same(got, in));
    }

    // 2. All-zero mask still yields exactly one page per request -- invariants
    //    2 and 3. An eviction policy that scores everything below its threshold
    //    must degrade to "attend over the current token", never to an empty
    //    page list.
    {
        const Csr in = make_csr({3, 1, 7, 12}, 16, rng);
        std::vector<std::uint8_t> keep(in.indices.size(), 0);
        const Csr got = run_device(in, keep);
        bool ok = true;
        for (std::size_t r = 0; r < in.last_lens.size(); ++r) {
            const std::uint32_t pages = in.indptr[r + 1] - in.indptr[r];
            const std::uint32_t kept = got.indptr[r + 1] - got.indptr[r];
            ok = ok && kept == (pages == 0 ? 0u : 1u);
            if (pages != 0) {
                // ...and it must be the LAST page, the one holding this fire's
                // token, not merely some page.
                ok = ok && got.indices[got.indptr[r]] ==
                               in.indices[in.indptr[r + 1] - 1];
            }
        }
        ok = ok && got.last_lens == in.last_lens;
        report("all-zero mask keeps exactly the last page", ok);
    }

    // 3. Order preservation on an adversarial mask. FlashInfer derives a
    //    position's absolute index from its slot in the page list, so a
    //    survivor emitted out of order relabels every key after it.
    {
        const Csr in = make_csr({40, 33, 5}, 16, rng);
        std::vector<std::uint8_t> keep(in.indices.size(), 0);
        for (std::size_t i = 0; i < keep.size(); ++i) {
            keep[i] = (i % 3 == 0) ? 1 : 0;
        }
        check_against_reference("strided mask preserves order", in, keep);
    }

    // 4. Multi-tile: more pages than the kernel's 256-thread block, so the
    //    cross-tile running aggregate in the scatter is exercised. A tile-local
    //    scan would pass every test above and fail exactly here.
    {
        const Csr in = make_csr({1000, 257, 256, 255, 1}, 16, rng);
        std::vector<std::uint8_t> keep(in.indices.size(), 0);
        std::mt19937 mrng(7);
        std::bernoulli_distribution coin(0.4);
        for (auto& k : keep) k = coin(mrng) ? 1 : 0;
        check_against_reference("multi-tile mask (>256 pages/request)", in,
                                keep);
    }

    // 5. Empty and singleton requests mixed with real ones -- the boundary the
    //    per-request block loop is most likely to get wrong.
    {
        const Csr in = make_csr({0, 1, 0, 2, 1, 0}, 16, rng);
        std::vector<std::uint8_t> keep(in.indices.size(), 0);
        for (std::size_t i = 0; i < keep.size(); ++i) keep[i] = (i % 2);
        check_against_reference("empty and singleton requests", in, keep);
    }

    // 6b. More requests than a block has threads. The scatter kernel derives
    //     its own output base by summing the counts of every earlier request,
    //     and that loop strides by blockDim.x -- so a batch wider than the
    //     block is the only shape that exercises more than one iteration of it.
    //     Get this wrong and every request past the 256th writes to the wrong
    //     offset, which is silent and looks like another request's cache.
    {
        std::uniform_int_distribution<std::uint32_t> npages(0, 9);
        std::vector<std::uint32_t> shape(300);
        for (auto& v : shape) v = npages(rng);
        const Csr in = make_csr(shape, 16, rng);
        std::bernoulli_distribution coin(0.5);
        std::vector<std::uint8_t> keep(in.indices.size(), 0);
        for (auto& k : keep) k = coin(rng) ? 1 : 0;
        check_against_reference("more requests than block threads", in, keep);
    }

    // 6. Randomized soak across shapes and densities.
    {
        bool ok = true;
        std::uniform_int_distribution<std::uint32_t> nreq(1, 24);
        std::uniform_int_distribution<std::uint32_t> npages(0, 600);
        for (int trial = 0; trial < 60 && ok; ++trial) {
            const std::uint32_t R = nreq(rng);
            std::vector<std::uint32_t> shape(R);
            for (auto& s : shape) s = npages(rng);
            const Csr in = make_csr(shape, 16, rng);
            const double density = 0.05 + 0.9 * (trial % 10) / 10.0;
            std::bernoulli_distribution coin(density);
            std::vector<std::uint8_t> keep(in.indices.size(), 0);
            for (auto& k : keep) k = coin(rng) ? 1 : 0;
            ok = same(run_device(in, keep), host_compact(in, keep));
            if (!ok) std::printf("  trial %d disagreed\n", trial);
        }
        report("randomized soak (60 trials)", ok);
    }

    // 7. Rows wider than any request needs. This is the shape the driver
    //    actually produces: the mask is sized from the host page CSR, which on
    //    the decode-envelope path is a conservative BOUND, while the CSR being
    //    compacted is the geometry the device resolved. Slot p of request r
    //    must still mean slot p of request r -- if the kernel ever recovered
    //    the row base from `page_indptr_in` instead of from r, every request
    //    after the first would read another request's mask, and this is the
    //    only test that would notice.
    {
        bool ok = true;
        std::uniform_int_distribution<std::uint32_t> pad(1, 300);
        for (int trial = 0; trial < 40 && ok; ++trial) {
            const Csr in = make_csr({17, 3, 128, 1, 64}, 16, rng);
            std::bernoulli_distribution coin(0.35);
            std::vector<std::uint8_t> keep(in.indices.size(), 0);
            for (auto& k : keep) k = coin(rng) ? 1 : 0;
            ok = same(run_device(in, keep, pad(rng)), host_compact(in, keep));
            if (!ok) std::printf("  padded trial %d disagreed\n", trial);
        }
        report("over-wide rows (host CSR is a bound)", ok);
    }

    // 8. The same, at multi-tile width, so the padding interacts with the
    //    `running` aggregate that carries survivor counts across tiles.
    {
        const Csr in = make_csr({700, 1, 300}, 16, rng);
        std::vector<std::uint8_t> keep(in.indices.size(), 0);
        for (std::size_t i = 0; i < keep.size(); ++i) {
            keep[i] = (i % 5 == 0) ? 1 : 0;
        }
        report("over-wide multi-tile rows",
               same(run_device(in, keep, 257), host_compact(in, keep)));
    }

    std::printf(g_fail ? "\nPAGE_COMPACT FAILED (%d)\n"
                       : "\nALL PASS (0 failures)\n",
                g_fail);
    return g_fail ? 1 : 0;
}
