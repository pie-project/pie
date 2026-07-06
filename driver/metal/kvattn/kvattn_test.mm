// kvattn_test — bit-exact Metal parity for the paged-KV data-movement kernels
// (M3 phase 1: write_kv + gather_rows). Each Metal kernel is compared byte-for-
// byte (u16) against the CPU reference (reference.hpp), which is a port of the
// CUDA index math (kv_paged.cu / gather_rows.cu). Reuses the ptir MetalHarness.
//
// Usage: kvattn_test [kernels_dir]

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>

#include "metal_harness.hpp"  // ../ptir/metal_harness.hpp
#include "reference.hpp"

#ifndef KVATTN_KERNELS_DIR
#define KVATTN_KERNELS_DIR "."
#endif

using namespace ptir_metal;
using namespace kvattn;

namespace {
int g_pass = 0, g_fail = 0;

template <class T>
void report(const std::string& name, const std::vector<T>& got, const std::vector<T>& want) {
    bool ok = got.size() == want.size() &&
              std::memcmp(got.data(), want.data(), got.size() * sizeof(T)) == 0;
    if (ok) { std::printf("  PASS  %s (%zu words byte-exact)\n", name.c_str(), want.size()); ++g_pass; }
    else {
        long d = -1;
        for (std::size_t i = 0; i < got.size() && i < want.size(); ++i)
            if (got[i] != want[i]) { d = (long)i; break; }
        std::printf("  FAIL  %s: first diff @%ld (got %u want %u; sizes %zu/%zu)\n", name.c_str(),
                    d, d >= 0 ? (unsigned)got[d] : 0u, d >= 0 ? (unsigned)want[d] : 0u,
                    got.size(), want.size());
        ++g_fail;
    }
}

// Distinct 16-bit pattern per (token,element) so any mis-scatter is visible.
std::uint16_t pat(int t, int i) { return (std::uint16_t)(((t & 0xff) << 8) | (i & 0xff)); }

void test_write_kv(MetalHarness& h, bool hnd) {
    ref::KvGeometry g;
    g.R = 2; g.page_size = 4; g.h_kv = 2; g.d = 4;
    // req0: 2 new tokens, 2 pages (indices [0,2]); last page holds 3 → 4+3=7 kv total.
    // req1: 3 new tokens, 2 pages (indices [1,3]); last page holds 2 → 4+2=6 kv total.
    g.qo_indptr = {0, 2, 5};
    g.kv_page_indptr = {0, 2, 4};
    g.kv_page_indices = {0, 2, 1, 3};
    g.kv_last_page_lens = {3, 2};
    g.total_pages = 4;  // physical page ids 0..3 used

    const int tt = g.total_tokens(), row = g.row();
    std::vector<std::uint16_t> k_curr(tt * row), v_curr(tt * row);
    for (int t = 0; t < tt; ++t)
        for (int i = 0; i < row; ++i) {
            k_curr[t * row + i] = pat(t, i);
            v_curr[t * row + i] = (std::uint16_t)(pat(t, i) ^ 0x5555);
        }

    std::vector<std::uint16_t> k_ref, v_ref;
    ref::write_kv(g, k_curr, v_curr, hnd, k_ref, v_ref);

    std::vector<std::uint16_t> k_got(g.pages_words(), 0), v_got(g.pages_words(), 0);
    int R = g.R, ps = g.page_size, hkv = g.h_kv, d = g.d, hl = hnd ? 1 : 0;
    std::uint32_t total_elems = (std::uint32_t)(tt * row);
    std::vector<Arg> args = {
        Arg::in(k_curr.data(), k_curr.size() * 2),
        Arg::in(v_curr.data(), v_curr.size() * 2),
        Arg::out(k_got.data(), k_got.size() * 2),
        Arg::out(v_got.data(), v_got.size() * 2),
        Arg::in(g.qo_indptr.data(), g.qo_indptr.size() * 4),
        Arg::in(g.kv_page_indices.data(), g.kv_page_indices.size() * 4),
        Arg::in(g.kv_page_indptr.data(), g.kv_page_indptr.size() * 4),
        Arg::in(g.kv_last_page_lens.data(), g.kv_last_page_lens.size() * 4),
        Arg::in(&R, 4), Arg::in(&ps, 4), Arg::in(&hkv, 4), Arg::in(&d, 4),
        Arg::in(&hl, 4), Arg::in(&total_elems, 4),
    };
    if (!h.run("write_kv", args, total_elems)) {
        std::printf("  FAIL  write_kv(%s): %s\n", hnd ? "HND" : "NHD", h.error().c_str());
        ++g_fail; return;
    }
    report(std::string("write_kv K (") + (hnd ? "HND" : "NHD") + ")", k_got, k_ref);
    report(std::string("write_kv V (") + (hnd ? "HND" : "NHD") + ")", v_got, v_ref);
}

void test_gather_rows(MetalHarness& h) {
    const int n_src = 6, vocab = 16;
    std::vector<std::uint16_t> src(n_src * vocab);
    for (int r = 0; r < n_src; ++r)
        for (int j = 0; j < vocab; ++j) src[r * vocab + j] = (std::uint16_t)((r << 8) | j);
    std::vector<std::int32_t> idx = {5, 0, 3, 3, 1};  // includes a repeat
    auto want = ref::gather_rows(src, idx, vocab);
    std::vector<std::uint16_t> got(idx.size() * vocab, 0);
    int v = vocab; std::uint32_t total = (std::uint32_t)(idx.size() * vocab);
    std::vector<Arg> args = {
        Arg::in(src.data(), src.size() * 2),
        Arg::in(idx.data(), idx.size() * 4),
        Arg::out(got.data(), got.size() * 2),
        Arg::in(&v, 4), Arg::in(&total, 4),
    };
    if (!h.run("gather_rows", args, total)) {
        std::printf("  FAIL  gather_rows: %s\n", h.error().c_str());
        ++g_fail; return;
    }
    report("gather_rows", got, want);
}

}  // namespace

int main(int argc, char** argv) {
    std::string kernels_dir = argc > 1 ? argv[1] : KVATTN_KERNELS_DIR;
    MetalHarness h;
    if (!h.ok()) { std::printf("KVATTN_TEST_FAIL: %s\n", h.error().c_str()); return 2; }
    std::printf("device: %s\n", h.device_name().c_str());
    if (!h.load_library(kernels_dir + "/kv_attn.metal")) {
        std::printf("KVATTN_TEST_FAIL: %s\n", h.error().c_str());
        return 2;
    }
    test_write_kv(h, /*hnd=*/false);
    test_write_kv(h, /*hnd=*/true);
    test_gather_rows(h);
    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    if (g_fail == 0) { std::printf("KVATTN_TEST_OK\n"); return 0; }
    std::printf("KVATTN_TEST_FAIL\n");
    return 1;
}
