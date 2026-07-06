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

// ── Metal kernel wrappers (host-chained intermediates for the layer assembly) ─
struct Chain {
    MetalHarness& h;
    bool ok = true;
    std::vector<float> matmul(const std::vector<float>& x, const std::vector<float>& w,
                              int M, int N, int K) {
        std::vector<float> y((std::size_t)M * N, 0.0f);
        int m = M, n = N, k = K; std::uint32_t total = (std::uint32_t)(M * N);
        std::vector<Arg> a = {Arg::in(x.data(), x.size() * 4), Arg::in(w.data(), w.size() * 4),
                              Arg::out(y.data(), y.size() * 4), Arg::in(&m, 4), Arg::in(&n, 4),
                              Arg::in(&k, 4), Arg::in(&total, 4)};
        ok = ok && h.run("matmul_xwt", a, total);
        return y;
    }
    std::vector<float> rmsnorm(const std::vector<float>& x, const std::vector<float>& w,
                               int rows, int dim, float eps) {
        std::vector<float> y((std::size_t)rows * dim, 0.0f);
        int r = rows, d = dim;
        std::vector<Arg> a = {Arg::in(x.data(), x.size() * 4), Arg::in(w.data(), w.size() * 4),
                              Arg::out(y.data(), y.size() * 4), Arg::in(&r, 4), Arg::in(&d, 4),
                              Arg::in(&eps, 4)};
        ok = ok && h.run("rmsnorm", a, rows);
        return y;
    }
    void rope_ip(std::vector<float>& io, const std::vector<std::int32_t>& pos, int rows,
                 int n_heads, int hd, float theta) {
        int nh = n_heads, d = hd; float th = theta;
        std::uint32_t total = (std::uint32_t)(rows * n_heads * (hd / 2));
        std::vector<Arg> a = {Arg::inout(io.data(), io.size() * 4), Arg::in(pos.data(), pos.size() * 4),
                              Arg::in(&nh, 4), Arg::in(&d, 4), Arg::in(&th, 4), Arg::in(&total, 4)};
        ok = ok && h.run("rope_qwen", a, total);
    }
    std::vector<float> attention(const qwen3::ref::AttnConfig& c, const std::vector<float>& q,
                                 const std::vector<float>& k, const std::vector<float>& v) {
        std::vector<float> out((std::size_t)c.N * c.n_q_heads * c.d, 0.0f);
        int nqh = c.n_q_heads, nkv = c.n_kv_heads, d = c.d, ps = c.page_size, gqa = c.gqa();
        float scale = c.scale; std::uint32_t total = (std::uint32_t)(c.N * c.n_q_heads);
        std::vector<Arg> a = {Arg::in(q.data(), q.size() * 4), Arg::in(k.data(), k.size() * 4),
                              Arg::in(v.data(), v.size() * 4), Arg::out(out.data(), out.size() * 4),
                              Arg::in(c.position_ids.data(), c.position_ids.size() * 4),
                              Arg::in(c.req_of_token.data(), c.req_of_token.size() * 4),
                              Arg::in(c.kv_page_indices.data(), c.kv_page_indices.size() * 4),
                              Arg::in(c.kv_page_indptr.data(), c.kv_page_indptr.size() * 4),
                              Arg::in(&nqh, 4), Arg::in(&nkv, 4), Arg::in(&d, 4), Arg::in(&ps, 4),
                              Arg::in(&gqa, 4), Arg::in(&scale, 4), Arg::in(&total, 4)};
        ok = ok && h.run("paged_attention", a, total);
        return out;
    }
    std::vector<float> swiglu(const std::vector<float>& gate, const std::vector<float>& up) {
        std::vector<float> y(gate.size(), 0.0f);
        std::uint32_t n = (std::uint32_t)gate.size();
        std::vector<Arg> a = {Arg::in(gate.data(), gate.size() * 4), Arg::in(up.data(), up.size() * 4),
                              Arg::out(y.data(), y.size() * 4), Arg::in(&n, 4)};
        ok = ok && h.run("swiglu", a, n);
        return y;
    }
    void add_ip(std::vector<float>& a, const std::vector<float>& b) {
        std::uint32_t n = (std::uint32_t)a.size();
        std::vector<Arg> ar = {Arg::inout(a.data(), a.size() * 4), Arg::in(b.data(), b.size() * 4),
                               Arg::in(&n, 4)};
        ok = ok && h.run("add_inplace", ar, n);
    }
};

void test_prefill_attention(MetalHarness& h) {
    // Single request, N prompt tokens at positions 0..N-1 (canonical prefill).
    qwen3::ref::AttnConfig c;
    c.N = 5; c.n_q_heads = A::N_Q_HEADS; c.n_kv_heads = A::N_KV_HEADS; c.d = A::HEAD_DIM;
    c.page_size = c.N; c.scale = 1.0f / std::sqrt((float)c.d);
    c.position_ids = {0, 1, 2, 3, 4};
    c.req_of_token.assign(c.N, 0);
    c.kv_page_indices = {0};
    c.kv_page_indptr = {0, 1};
    auto q = rand_vec((std::size_t)c.N * c.n_q_heads * c.d, 101);
    auto k = rand_vec((std::size_t)c.N * c.n_kv_heads * c.d, 102);
    auto v = rand_vec((std::size_t)c.N * c.n_kv_heads * c.d, 103);
    auto want = qwen3::ref::paged_attention(c, q, k, v);
    Chain ch{h};
    auto got = ch.attention(c, q, k, v);
    if (!ch.ok) { std::printf("  FAIL prefill_attention: %s\n", h.error().c_str()); ++g_fail; return; }
    check_tol("prefill_attention (N=5 causal, head_dim=128, GQA=2)", got, want, 1e-5);
}

void test_decoder_layer(MetalHarness& h) {
    qwen3::ref::LayerDims D;
    D.N = 5; D.hidden = A::HIDDEN; D.n_q_heads = A::N_Q_HEADS; D.n_kv_heads = A::N_KV_HEADS;
    D.head_dim = A::HEAD_DIM; D.intermediate = A::INTERMEDIATE;
    D.rms_eps = A::RMS_EPS; D.rope_theta = A::ROPE_THETA;
    D.attn_scale = 1.0f / std::sqrt((float)A::HEAD_DIM);
    const int H = D.hidden, qd = D.q_dim(), kd = D.kv_dim(), I = D.intermediate, hd = D.head_dim;

    // Small weights (scaled down so activations stay in a sane range for f32).
    auto scaled = [](std::size_t n, std::uint32_t seed, float s) {
        auto v = rand_vec(n, seed);
        for (auto& x : v) x *= s;
        return v;
    };
    qwen3::ref::LayerWeights w;
    w.input_ln = scaled(H, 1, 1.0f);
    w.wq = scaled((std::size_t)qd * H, 2, 0.03f);
    w.wk = scaled((std::size_t)kd * H, 3, 0.03f);
    w.wv = scaled((std::size_t)kd * H, 4, 0.03f);
    w.q_norm = scaled(hd, 5, 1.0f);
    w.k_norm = scaled(hd, 6, 1.0f);
    w.wo = scaled((std::size_t)H * qd, 7, 0.03f);
    w.post_ln = scaled(H, 8, 1.0f);
    w.wgate = scaled((std::size_t)I * H, 9, 0.03f);
    w.wup = scaled((std::size_t)I * H, 10, 0.03f);
    w.wdown = scaled((std::size_t)H * I, 11, 0.03f);
    auto x = scaled((std::size_t)D.N * H, 12, 1.0f);
    std::vector<std::int32_t> positions = {0, 1, 2, 3, 4};

    auto want = qwen3::ref::decoder_layer(x, positions, w, D);

    // Metal assembly — the exact same op sequence, host-chained.
    Chain ch{h};
    auto nx = ch.rmsnorm(x, w.input_ln, D.N, H, D.rms_eps);
    auto q = ch.matmul(nx, w.wq, D.N, qd, H);
    auto k = ch.matmul(nx, w.wk, D.N, kd, H);
    auto v = ch.matmul(nx, w.wv, D.N, kd, H);
    q = ch.rmsnorm(q, w.q_norm, D.N * D.n_q_heads, hd, D.rms_eps);
    k = ch.rmsnorm(k, w.k_norm, D.N * D.n_kv_heads, hd, D.rms_eps);
    ch.rope_ip(q, positions, D.N, D.n_q_heads, hd, D.rope_theta);
    ch.rope_ip(k, positions, D.N, D.n_kv_heads, hd, D.rope_theta);
    qwen3::ref::AttnConfig c;
    c.N = D.N; c.n_q_heads = D.n_q_heads; c.n_kv_heads = D.n_kv_heads; c.d = hd;
    c.page_size = D.N; c.scale = D.attn_scale; c.position_ids = positions;
    c.req_of_token.assign(D.N, 0); c.kv_page_indices = {0}; c.kv_page_indptr = {0, 1};
    auto attn = ch.attention(c, q, k, v);
    auto o = ch.matmul(attn, w.wo, D.N, H, qd);
    std::vector<float> y = x;
    ch.add_ip(y, o);   // y = x + attn·Woᵀ
    auto ny = ch.rmsnorm(y, w.post_ln, D.N, H, D.rms_eps);
    auto gate = ch.matmul(ny, w.wgate, D.N, I, H);
    auto up = ch.matmul(ny, w.wup, D.N, I, H);
    auto g = ch.swiglu(gate, up);
    auto down = ch.matmul(g, w.wdown, D.N, H, I);
    std::vector<float> out = y;
    ch.add_ip(out, down);   // out = y + g·Wdownᵀ

    if (!ch.ok) { std::printf("  FAIL decoder_layer: %s\n", h.error().c_str()); ++g_fail; return; }
    check_tol("decoder_layer (full Qwen3-0.6B layer, N=5)", out, want, 5e-3);
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
    test_prefill_attention(h);
    test_decoder_layer(h);
    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    if (g_fail == 0) { std::printf("QWEN3_TEST_OK\n"); return 0; }
    std::printf("QWEN3_TEST_FAIL\n");
    return 1;
}
