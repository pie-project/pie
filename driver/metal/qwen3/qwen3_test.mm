// qwen3_test — Metal parity for the Qwen3-0.6B decoder-layer primitives.
// Each Metal kernel is validated within tolerance against the CPU f32 reference
// (reference.hpp), which ports the CUDA kernel formulas. Reuses the ptir
// MetalHarness. Uses the real Qwen3-0.6B dims (arch.hpp) where relevant.
//
// Usage: qwen3_test [kernels_dir]

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "arch.hpp"
#include "metal_harness.hpp"  // ../ptir/metal_harness.hpp
#include "reference.hpp"

#ifndef QWEN3_KERNELS_DIR
#define QWEN3_KERNELS_DIR "."
#endif

using namespace ptir_metal;
namespace A = qwen3::arch;

namespace {
int g_pass = 0, g_fail = 0;

// Deterministic pseudo-random f32 in [-1,1).
struct Rng {
    std::uint32_t s;
    explicit Rng(std::uint32_t seed) : s(seed) {}
    float next() {
        s = s * 1664525u + 1013904223u;
        return ((float)((s >> 9) & 0xffff) / 32768.0f) - 1.0f;
    }
};
std::vector<float> rand_vec(std::size_t n, std::uint32_t seed) {
    Rng r(seed);
    std::vector<float> v(n);
    for (auto& x : v) x = r.next();
    return v;
}

void check_tol(const std::string& name, const std::vector<float>& got,
               const std::vector<float>& want, double tol) {
    double max_abs = 0.0, max_rel = 0.0;
    for (std::size_t i = 0; i < want.size(); ++i) {
        double a = std::fabs((double)got[i] - (double)want[i]);
        max_abs = a > max_abs ? a : max_abs;
        double rel = a / (std::fabs((double)want[i]) + 1e-6);
        max_rel = rel > max_rel ? rel : max_rel;
    }
    bool ok = max_abs <= tol;
    std::printf("  %s  %s (max_abs=%.3e max_rel=%.3e, %zu elems, tol=%.1e)\n",
                ok ? "PASS" : "FAIL", name.c_str(), max_abs, max_rel, want.size(), tol);
    ok ? ++g_pass : ++g_fail;
}

void test_matmul(MetalHarness& h) {
    // Qwen3 QKV projection shape: x[M, hidden] @ Wᵀ[q_dim, hidden].
    const int M = 4, K = A::HIDDEN, N = A::Q_DIM;
    auto x = rand_vec((std::size_t)M * K, 11);
    auto w = rand_vec((std::size_t)N * K, 22);
    auto want = qwen3::ref::matmul_xwt(x, w, M, N, K);
    std::vector<float> got((std::size_t)M * N, 0.0f);
    int m = M, n = N, k = K; std::uint32_t total = (std::uint32_t)(M * N);
    std::vector<Arg> args = {
        Arg::in(x.data(), x.size() * 4), Arg::in(w.data(), w.size() * 4),
        Arg::out(got.data(), got.size() * 4), Arg::in(&m, 4), Arg::in(&n, 4),
        Arg::in(&k, 4), Arg::in(&total, 4),
    };
    if (!h.run("matmul_xwt", args, total)) { std::printf("  FAIL matmul: %s\n", h.error().c_str()); ++g_fail; return; }
    // K=1024 accumulation; f32 rounding grows ~sqrt(K)*eps*|terms|.
    check_tol("matmul_xwt (QKV proj [4,1024]x[2048,1024])", got, want, 1e-3);
}

void test_rmsnorm(MetalHarness& h) {
    const int rows = 4, dim = A::HIDDEN;
    auto x = rand_vec((std::size_t)rows * dim, 33);
    auto w = rand_vec(dim, 44);
    auto want = qwen3::ref::rmsnorm(x, w, rows, dim, A::RMS_EPS);
    std::vector<float> got((std::size_t)rows * dim, 0.0f);
    int r = rows, d = dim; float eps = A::RMS_EPS;
    std::vector<Arg> args = {
        Arg::in(x.data(), x.size() * 4), Arg::in(w.data(), w.size() * 4),
        Arg::out(got.data(), got.size() * 4), Arg::in(&r, 4), Arg::in(&d, 4), Arg::in(&eps, 4),
    };
    if (!h.run("rmsnorm", args, rows)) { std::printf("  FAIL rmsnorm: %s\n", h.error().c_str()); ++g_fail; return; }
    check_tol("rmsnorm (hidden=1024)", got, want, 1e-5);
}

void test_qk_norm(MetalHarness& h) {
    // Per-head QK-norm: RMSNorm over head_dim, rows = N * n_q_heads.
    const int N = 3, rows = N * A::N_Q_HEADS, dim = A::HEAD_DIM;
    auto x = rand_vec((std::size_t)rows * dim, 55);
    auto w = rand_vec(dim, 66);
    auto want = qwen3::ref::rmsnorm(x, w, rows, dim, A::RMS_EPS);
    std::vector<float> got((std::size_t)rows * dim, 0.0f);
    int r = rows, d = dim; float eps = A::RMS_EPS;
    std::vector<Arg> args = {
        Arg::in(x.data(), x.size() * 4), Arg::in(w.data(), w.size() * 4),
        Arg::out(got.data(), got.size() * 4), Arg::in(&r, 4), Arg::in(&d, 4), Arg::in(&eps, 4),
    };
    if (!h.run("rmsnorm", args, rows)) { std::printf("  FAIL qk_norm: %s\n", h.error().c_str()); ++g_fail; return; }
    check_tol("qk_norm (per-head RMSNorm head_dim=128)", got, want, 1e-5);
}

void test_rope(MetalHarness& h) {
    const int rows = 3, n_heads = A::N_Q_HEADS, hd = A::HEAD_DIM;
    auto h_in = rand_vec((std::size_t)rows * n_heads * hd, 77);
    std::vector<std::int32_t> pos = {0, 7, 40};
    auto want = h_in;
    qwen3::ref::rope(want, pos, rows, n_heads, hd, A::ROPE_THETA);

    std::vector<float> io = h_in;  // rope_qwen rotates in place
    int nh = n_heads, d = hd; float theta = A::ROPE_THETA;
    std::uint32_t total = (std::uint32_t)(rows * n_heads * (hd / 2));
    std::vector<Arg> args = {
        Arg::inout(io.data(), io.size() * 4),
        Arg::in(pos.data(), pos.size() * 4), Arg::in(&nh, 4), Arg::in(&d, 4),
        Arg::in(&theta, 4), Arg::in(&total, 4),
    };
    if (!h.run("rope_qwen", args, total)) { std::printf("  FAIL rope: %s\n", h.error().c_str()); ++g_fail; return; }
    check_tol("rope_qwen (theta=1e6, head_dim=128)", io, want, 1e-5);
}

void test_swiglu(MetalHarness& h) {
    const int n = 4 * A::INTERMEDIATE;
    auto gate = rand_vec(n, 88);
    auto up = rand_vec(n, 99);
    auto want = qwen3::ref::swiglu(gate, up);
    std::vector<float> got(n, 0.0f);
    std::uint32_t nn = (std::uint32_t)n;
    std::vector<Arg> args = {
        Arg::in(gate.data(), gate.size() * 4), Arg::in(up.data(), up.size() * 4),
        Arg::out(got.data(), got.size() * 4), Arg::in(&nn, 4),
    };
    if (!h.run("swiglu", args, nn)) { std::printf("  FAIL swiglu: %s\n", h.error().c_str()); ++g_fail; return; }
    check_tol("swiglu (intermediate=3072)", got, want, 1e-6);
}

}  // namespace

int main(int argc, char** argv) {
    std::string kernels_dir = argc > 1 ? argv[1] : QWEN3_KERNELS_DIR;
    MetalHarness h;
    if (!h.ok()) { std::printf("QWEN3_TEST_FAIL: %s\n", h.error().c_str()); return 2; }
    std::printf("device: %s | Qwen3-0.6B: hidden=%d head_dim=%d n_q=%d n_kv=%d I=%d\n\n",
                h.device_name().c_str(), A::HIDDEN, A::HEAD_DIM, A::N_Q_HEADS, A::N_KV_HEADS, A::INTERMEDIATE);
    if (!h.load_library(kernels_dir + "/layer.metal")) {
        std::printf("QWEN3_TEST_FAIL: %s\n", h.error().c_str());
        return 2;
    }
    test_matmul(h);
    test_rmsnorm(h);
    test_qk_norm(h);
    test_rope(h);
    test_swiglu(h);
    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    if (g_fail == 0) { std::printf("QWEN3_TEST_OK\n"); return 0; }
    std::printf("QWEN3_TEST_FAIL\n");
    return 1;
}
