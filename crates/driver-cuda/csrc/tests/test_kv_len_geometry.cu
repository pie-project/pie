// PTIR M5 — geometry-as-data (C1 FINAL) kv_len handshake.
//
// The FINAL form of C1 has forward geometry produced by a PREVIOUS PASS's KERNEL
// into a device buffer the host never reads (vs the INTERIM host-computed
// columns). This test is the standalone half of the M5 handshake for the length
// column: it asserts the DEVICE-PRODUCED `kv_len` (launch_derive_kv_len) equals
// the HOST reference formula from runtime/src/inference/request.rs BIT FOR BIT
// over randomized page geometries — so a forward binding the device-resident
// `kv_len_device` handle sees byte-identical geometry to the host-fed run.
// (The full forward-binds-device-buffer path rides the executor late-bind
// read-seam; this locks the producer/host equality it depends on.)

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

#include <cuda_runtime.h>

#include "kernels/geometry.hpp"

using pie_cuda_driver::kernels::launch_derive_kv_len;
using pie_cuda_driver::kernels::launch_resolve_slot_to_block;

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

// Host reference — the EXACT formula in request.rs append_request_with_options.
std::uint32_t host_kv_len(std::uint32_t page_count, std::uint32_t page_size,
                          std::uint32_t last_page_len) {
    return page_count == 0u ? 0u
                            : (page_count - 1u) * page_size + last_page_len;
}

// Build a randomized CSR page geometry for `R` requests, derive kv_len on the
// device, and compare bit-for-bit to the host formula.
void check(const char* name, std::uint32_t R, std::uint32_t page_size,
           std::uint32_t max_pages, std::mt19937& rng) {
    std::uniform_int_distribution<std::uint32_t> page_dist(0, max_pages);

    std::vector<std::uint32_t> indptr(R + 1, 0);
    std::vector<std::uint32_t> last_page_lens(R, 0);
    std::vector<std::uint32_t> ref(R, 0);
    for (std::uint32_t r = 0; r < R; ++r) {
        const std::uint32_t page_count = page_dist(rng);
        // last_page_len is 1..=page_size for a live page, 0 for an empty request.
        std::uint32_t last = 0;
        if (page_count != 0) {
            std::uniform_int_distribution<std::uint32_t> len_dist(1, page_size);
            last = len_dist(rng);
        }
        indptr[r + 1] = indptr[r] + page_count;
        last_page_lens[r] = last;
        ref[r] = host_kv_len(page_count, page_size, last);
    }

    std::uint32_t *d_indptr, *d_last, *d_out;
    RT(cudaMalloc(&d_indptr, (R + 1) * sizeof(std::uint32_t)));
    RT(cudaMalloc(&d_last, R * sizeof(std::uint32_t)));
    RT(cudaMalloc(&d_out, R * sizeof(std::uint32_t)));
    RT(cudaMemcpy(d_indptr, indptr.data(), (R + 1) * sizeof(std::uint32_t),
                  cudaMemcpyHostToDevice));
    RT(cudaMemcpy(d_last, last_page_lens.data(), R * sizeof(std::uint32_t),
                  cudaMemcpyHostToDevice));
    // Poison the output so an unwritten slot can never coincidentally match.
    RT(cudaMemset(d_out, 0xAB, R * sizeof(std::uint32_t)));

    launch_derive_kv_len(d_indptr, d_last, page_size, R, d_out, nullptr);
    RT(cudaDeviceSynchronize());

    std::vector<std::uint32_t> got(R, 0);
    RT(cudaMemcpy(got.data(), d_out, R * sizeof(std::uint32_t),
                  cudaMemcpyDeviceToHost));

    bool ok = true;
    std::uint32_t bad_r = 0;
    for (std::uint32_t r = 0; r < R && ok; ++r) {
        if (got[r] != ref[r]) { ok = false; bad_r = r; }
    }
    std::printf("[%s] %s (R=%u, page_size=%u)\n", ok ? " ok " : "FAIL", name, R,
                page_size);
    if (!ok) {
        std::printf("  first mismatch r=%u: device=%u host=%u\n", bad_r,
                    got[bad_r], ref[bad_r]);
        ++g_fail;
    }

    cudaFree(d_indptr); cudaFree(d_last); cudaFree(d_out);
}

}  // namespace

int main() {
    RT(cudaSetDevice(0));
    std::mt19937 rng(0xC1F17A15u);

    // Single-token decode (the common case): 1 page, last_page_len=1.
    {
        std::vector<std::uint32_t> indptr{0, 1};
        std::vector<std::uint32_t> last{1};
        std::uint32_t *d_indptr, *d_last, *d_out;
        RT(cudaMalloc(&d_indptr, 2 * sizeof(std::uint32_t)));
        RT(cudaMalloc(&d_last, sizeof(std::uint32_t)));
        RT(cudaMalloc(&d_out, sizeof(std::uint32_t)));
        RT(cudaMemcpy(d_indptr, indptr.data(), 2 * sizeof(std::uint32_t),
                      cudaMemcpyHostToDevice));
        RT(cudaMemcpy(d_last, last.data(), sizeof(std::uint32_t),
                      cudaMemcpyHostToDevice));
        launch_derive_kv_len(d_indptr, d_last, 16, 1, d_out, nullptr);
        RT(cudaDeviceSynchronize());
        std::uint32_t got = 0;
        RT(cudaMemcpy(&got, d_out, sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
        const bool ok = (got == 1u);
        std::printf("[%s] single-token decode (kv_len=1)\n", ok ? " ok " : "FAIL");
        if (!ok) { std::printf("  device=%u host=1\n", got); ++g_fail; }
        cudaFree(d_indptr); cudaFree(d_last); cudaFree(d_out);
    }

    // Empty request (0 pages ⇒ kv_len 0) mixed into a batch.
    {
        std::vector<std::uint32_t> indptr{0, 0, 3};  // r0 empty, r1 has 3 pages
        std::vector<std::uint32_t> last{0, 5};
        std::uint32_t *d_indptr, *d_last, *d_out;
        RT(cudaMalloc(&d_indptr, 3 * sizeof(std::uint32_t)));
        RT(cudaMalloc(&d_last, 2 * sizeof(std::uint32_t)));
        RT(cudaMalloc(&d_out, 2 * sizeof(std::uint32_t)));
        RT(cudaMemcpy(d_indptr, indptr.data(), 3 * sizeof(std::uint32_t),
                      cudaMemcpyHostToDevice));
        RT(cudaMemcpy(d_last, last.data(), 2 * sizeof(std::uint32_t),
                      cudaMemcpyHostToDevice));
        RT(cudaMemset(d_out, 0xAB, 2 * sizeof(std::uint32_t)));
        launch_derive_kv_len(d_indptr, d_last, 16, 2, d_out, nullptr);
        RT(cudaDeviceSynchronize());
        std::uint32_t got[2] = {0, 0};
        RT(cudaMemcpy(got, d_out, 2 * sizeof(std::uint32_t), cudaMemcpyDeviceToHost));
        const bool ok = (got[0] == 0u) && (got[1] == 2u * 16u + 5u);
        std::printf("[%s] empty + multi-page batch (kv_len=[0,37])\n",
                    ok ? " ok " : "FAIL");
        if (!ok) { std::printf("  device=[%u,%u]\n", got[0], got[1]); ++g_fail; }
        cudaFree(d_indptr); cudaFree(d_last); cudaFree(d_out);
    }

    // Randomized soak — several batch sizes and page caps, page_size=16.
    check("random small batch", 8, 16, 4, rng);
    check("random medium batch", 64, 16, 32, rng);
    check("random large batch", 1024, 16, 256, rng);
    // A larger page_size (block cache) and a wide batch spanning many blocks.
    check("random wide (page_size=256)", 4096, 256, 64, rng);

    // ── M5/C1-FINAL: slot → physical BlockId resolution (beam/§6.1 pages) ──────
    // The device-produced `pages` (working-set slot ids) resolve to physical
    // page-pool BlockIds via a runtime-uploaded slot→block dictionary. Verify
    // device-resolved == host-resolved bit-for-bit, incl. slot-0-valid and the
    // out-of-range loud sentinel.
    {
        // Dictionary: slot s → block (s*7+3), so slot 0 → block 3 (a real block).
        constexpr std::uint32_t NUM_SLOTS = 64;
        std::vector<std::uint32_t> slot_to_block(NUM_SLOTS);
        for (std::uint32_t s = 0; s < NUM_SLOTS; ++s) slot_to_block[s] = s * 7u + 3u;

        // Flattened [B,P] pages incl. slot 0 (valid) and one out-of-range (99).
        std::vector<std::uint32_t> pages{0, 5, 63, 12, 0, 99, 7, 40};
        const std::uint32_t count = static_cast<std::uint32_t>(pages.size());

        std::vector<std::uint32_t> ref(count);
        for (std::uint32_t i = 0; i < count; ++i) {
            ref[i] = pages[i] < NUM_SLOTS ? slot_to_block[pages[i]] : 0xFFFFFFFFu;
        }

        std::uint32_t *d_pages, *d_dict, *d_out;
        RT(cudaMalloc(&d_pages, count * sizeof(std::uint32_t)));
        RT(cudaMalloc(&d_dict, NUM_SLOTS * sizeof(std::uint32_t)));
        RT(cudaMalloc(&d_out, count * sizeof(std::uint32_t)));
        RT(cudaMemcpy(d_pages, pages.data(), count * sizeof(std::uint32_t),
                      cudaMemcpyHostToDevice));
        RT(cudaMemcpy(d_dict, slot_to_block.data(), NUM_SLOTS * sizeof(std::uint32_t),
                      cudaMemcpyHostToDevice));
        RT(cudaMemset(d_out, 0xAB, count * sizeof(std::uint32_t)));
        launch_resolve_slot_to_block(d_pages, d_dict, NUM_SLOTS, count, d_out, nullptr);
        RT(cudaDeviceSynchronize());
        std::vector<std::uint32_t> got(count);
        RT(cudaMemcpy(got.data(), d_out, count * sizeof(std::uint32_t),
                      cudaMemcpyDeviceToHost));
        bool ok = true;
        for (std::uint32_t i = 0; i < count; ++i) ok = ok && (got[i] == ref[i]);
        std::printf("[%s] slot->block resolve (slot0 valid + oob sentinel)\n",
                    ok ? " ok " : "FAIL");
        if (!ok) ++g_fail;
        cudaFree(d_pages); cudaFree(d_dict); cudaFree(d_out);
    }

    // Randomized soak: device-resolved == host-resolved bit-for-bit.
    {
        constexpr std::uint32_t NUM_SLOTS = 4096, COUNT = 8192;
        std::uniform_int_distribution<std::uint32_t> blk(0, 1u << 20);
        std::uniform_int_distribution<std::uint32_t> slot(0, NUM_SLOTS - 1);
        std::vector<std::uint32_t> dict(NUM_SLOTS), pages(COUNT), ref(COUNT);
        for (auto& b : dict) b = blk(rng);
        for (std::uint32_t i = 0; i < COUNT; ++i) {
            pages[i] = slot(rng);
            ref[i] = dict[pages[i]];
        }
        std::uint32_t *d_pages, *d_dict, *d_out;
        RT(cudaMalloc(&d_pages, COUNT * sizeof(std::uint32_t)));
        RT(cudaMalloc(&d_dict, NUM_SLOTS * sizeof(std::uint32_t)));
        RT(cudaMalloc(&d_out, COUNT * sizeof(std::uint32_t)));
        RT(cudaMemcpy(d_pages, pages.data(), COUNT * sizeof(std::uint32_t),
                      cudaMemcpyHostToDevice));
        RT(cudaMemcpy(d_dict, dict.data(), NUM_SLOTS * sizeof(std::uint32_t),
                      cudaMemcpyHostToDevice));
        launch_resolve_slot_to_block(d_pages, d_dict, NUM_SLOTS, COUNT, d_out, nullptr);
        RT(cudaDeviceSynchronize());
        std::vector<std::uint32_t> got(COUNT);
        RT(cudaMemcpy(got.data(), d_out, COUNT * sizeof(std::uint32_t),
                      cudaMemcpyDeviceToHost));
        bool ok = true;
        for (std::uint32_t i = 0; i < COUNT; ++i) ok = ok && (got[i] == ref[i]);
        std::printf("[%s] slot->block resolve soak (random)\n", ok ? " ok " : "FAIL");
        if (!ok) ++g_fail;
        cudaFree(d_pages); cudaFree(d_dict); cudaFree(d_out);
    }

    std::printf(g_fail ? "\nKV_LEN_GEOMETRY FAILED (%d)\n" : "\nALL PASS (0 failures)\n",
                g_fail);
    return g_fail ? 1 : 0;
}
