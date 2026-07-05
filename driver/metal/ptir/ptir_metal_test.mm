// ptir_metal_test — bit-exact parity harness for the PTIR sampling-IR
// foundational ops on Metal.
//
// Each op's Metal kernel output is compared byte-for-byte (u32 bit pattern)
// against the CPU reference (reference.hpp, a port of echo's Rust eval.rs).
// The CPU reference is the interim cross-backend oracle; once echo's golden
// vector files are wired in, the same reference is asserted == golden so the
// chain Metal == CPU-ref == golden == CUDA holds.
//
// Usage: ptir_metal_test [kernels_dir]   (default: PTIR_KERNELS_DIR)

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

#include "metal_harness.hpp"
#include "reference.hpp"

#ifndef PTIR_KERNELS_DIR
#define PTIR_KERNELS_DIR "."
#endif

using namespace ptir_metal;

namespace {

int g_pass = 0, g_fail = 0;

void report(const std::string& name, const std::vector<float>& got,
            const std::vector<float>& want) {
    long d = first_bit_diff(got, want);
    if (d < 0) {
        std::printf("  PASS  %s (%zu lanes bit-exact)\n", name.c_str(), want.size());
        ++g_pass;
    } else {
        std::uint32_t bg = 0, bw = 0;
        if (static_cast<std::size_t>(d) < got.size())
            std::memcpy(&bg, &got[d], 4);
        if (static_cast<std::size_t>(d) < want.size())
            std::memcpy(&bw, &want[d], 4);
        std::printf("  FAIL  %s: lane %ld got=0x%08x want=0x%08x\n", name.c_str(), d, bg, bw);
        ++g_fail;
    }
}

// Byte-exact comparison for non-float payloads (I32 tokens, bool bytes).
template <class T>
void report_raw(const std::string& name, const std::vector<T>& got,
                const std::vector<T>& want) {
    if (got.size() == want.size() &&
        std::memcmp(got.data(), want.data(), got.size() * sizeof(T)) == 0) {
        std::printf("  PASS  %s (%zu lanes byte-exact)\n", name.c_str(), want.size());
        ++g_pass;
    } else {
        std::printf("  FAIL  %s: byte mismatch (got %zu, want %zu lanes)\n", name.c_str(),
                    got.size(), want.size());
        ++g_fail;
    }
}

// ── mask_apply_packed (vector) ───────────────────────────────────────────────
void test_mask_apply_packed(MetalHarness& h) {
    const std::uint32_t n = 70;  // spans 3 mask words, tests tail bits
    std::vector<float> logits(n);
    for (std::uint32_t j = 0; j < n; ++j) logits[j] = static_cast<float>(j) - 12.5f;
    // Allow every 3rd token; word-packed.
    std::vector<std::uint32_t> mask((n + 31) / 32, 0);
    for (std::uint32_t j = 0; j < n; ++j)
        if (j % 3 == 0) mask[j >> 5] |= (1u << (j & 31u));

    auto want = ref::mask_apply_packed(logits, mask);
    std::vector<float> got(n, 0.0f);
    std::vector<Arg> args = {
        Arg::in(logits.data(), n * sizeof(float)),
        Arg::in(mask.data(), mask.size() * sizeof(std::uint32_t)),
        Arg::out(got.data(), n * sizeof(float)),
        Arg::in(&n, sizeof(n)),
    };
    if (!h.run("mask_apply_packed", args, n)) {
        std::printf("  FAIL  mask_apply_packed: %s\n", h.error().c_str());
        ++g_fail;
        return;
    }
    report("mask_apply_packed", got, want);
}

// ── mask_apply_packed (matrix, single mask broadcast over rows) ──────────────
void test_mask_apply_packed_matrix(MetalHarness& h) {
    const std::uint32_t rows = 2, vocab = 40;  // matrix_mask_apply_packed golden shape
    const std::uint32_t wpr = (vocab + 31) / 32;
    const std::uint32_t total = rows * vocab;
    std::vector<float> logits(total);
    for (std::uint32_t i = 0; i < total; ++i) logits[i] = static_cast<float>(i) * 0.25f - 3.0f;
    // ONE packed word-row [ceil(vocab/32)] broadcast across all rows; bit=column.
    std::vector<std::uint32_t> mask(wpr, 0);
    for (std::uint32_t c = 0; c < vocab; ++c)
        if (c % 5 == 0) mask[c >> 5] |= (1u << (c & 31u));

    auto want = ref::mask_apply_packed_matrix(logits, mask, rows, vocab);
    std::vector<float> got(total, 0.0f);
    std::vector<Arg> args = {
        Arg::in(logits.data(), total * sizeof(float)),
        Arg::in(mask.data(), mask.size() * sizeof(std::uint32_t)),
        Arg::out(got.data(), total * sizeof(float)),
        Arg::in(&vocab, sizeof(vocab)),
        Arg::in(&total, sizeof(total)),
    };
    if (!h.run("mask_apply_packed_matrix", args, total)) {
        std::printf("  FAIL  mask_apply_packed_matrix: %s\n", h.error().c_str());
        ++g_fail;
        return;
    }
    report("mask_apply_packed_matrix", got, want);
}

// ── dselect (matrix keep-mask apply) ─────────────────────────────────────────
void test_dselect(MetalHarness& h) {
    const std::uint32_t rows = 4, vocab = 32;
    const std::uint32_t n = rows * vocab;
    std::vector<std::uint8_t> cond(n);
    std::vector<float> a(n), b(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        cond[i] = (i * 7u + 1u) % 3u == 0 ? 1 : 0;
        a[i] = static_cast<float>(i) * 0.5f;
        b[i] = ref::kNegInf;  // the neg-inf fill (dselect(keep, scores, neg_inf))
    }
    auto want = ref::dselect_f32(cond, a, b);
    std::vector<float> got(n, 0.0f);
    std::vector<Arg> args = {
        Arg::in(cond.data(), n * sizeof(std::uint8_t)),
        Arg::in(a.data(), n * sizeof(float)),
        Arg::in(b.data(), n * sizeof(float)),
        Arg::out(got.data(), n * sizeof(float)),
        Arg::in(&n, sizeof(n)),
    };
    std::uint32_t lc = n, la = n, lb = n;
    args.push_back(Arg::in(&lc, sizeof(lc)));
    args.push_back(Arg::in(&la, sizeof(la)));
    args.push_back(Arg::in(&lb, sizeof(lb)));
    if (!h.run("dselect_f32", args, n)) {
        std::printf("  FAIL  dselect_f32: %s\n", h.error().c_str());
        ++g_fail;
        return;
    }
    report("dselect_f32", got, want);
}

// ── broadcast_matrix (scalar neg_inf -> [k, vocab]) ──────────────────────────
void test_broadcast_matrix(MetalHarness& h) {
    const std::uint32_t k = 4, vocab = 40;  // echo's broadcast_matrix(neg_inf, k, vocab)
    std::vector<float> src = {ref::kNegInf};
    auto want = ref::broadcast_matrix_f32(src, 1, 1, k, vocab);
    std::vector<float> got(static_cast<std::size_t>(k) * vocab, 0.0f);
    std::uint32_t sr = 1, sc = 1, dr = k, dc = vocab;
    std::vector<Arg> args = {
        Arg::in(src.data(), src.size() * sizeof(float)),
        Arg::out(got.data(), got.size() * sizeof(float)),
        Arg::in(&sr, sizeof(sr)),
        Arg::in(&sc, sizeof(sc)),
        Arg::in(&dr, sizeof(dr)),
        Arg::in(&dc, sizeof(dc)),
    };
    if (!h.run("broadcast_matrix_f32", args, k * vocab)) {
        std::printf("  FAIL  broadcast_matrix_f32: %s\n", h.error().c_str());
        ++g_fail;
        return;
    }
    report("broadcast_matrix_f32", got, want);

    // Also exercise per-row [m]->[m,n] (src viewed as (m,1)).
    std::vector<float> rowsrc = {1.0f, -2.0f, 3.0f, ref::kNegInf};
    auto want2 = ref::broadcast_matrix_f32(rowsrc, 4, 1, 4, vocab);
    std::vector<float> got2(static_cast<std::size_t>(4) * vocab, 0.0f);
    std::uint32_t sr2 = 4, sc2 = 1, dr2 = 4, dc2 = vocab;
    std::vector<Arg> args2 = {
        Arg::in(rowsrc.data(), rowsrc.size() * sizeof(float)),
        Arg::out(got2.data(), got2.size() * sizeof(float)),
        Arg::in(&sr2, sizeof(sr2)),
        Arg::in(&sc2, sizeof(sc2)),
        Arg::in(&dr2, sizeof(dr2)),
        Arg::in(&dc2, sizeof(dc2)),
    };
    if (!h.run("broadcast_matrix_f32", args2, 4 * vocab)) {
        std::printf("  FAIL  broadcast_matrix_f32 (per-row): %s\n", h.error().c_str());
        ++g_fail;
        return;
    }
    report("broadcast_matrix_f32 (per-row)", got2, want2);
}

// ── elementwise + reductions + scans ─────────────────────────────────────────
void test_elementwise(MetalHarness& h) {
    const std::uint32_t n = 33;
    std::vector<float> a(n), b(n);
    for (std::uint32_t i = 0; i < n; ++i) {
        a[i] = static_cast<float>(i) * 0.5f - 4.0f;
        b[i] = static_cast<float>((i * 3) % 7) + 0.25f;  // nonzero for div
    }
    std::uint32_t la = n, lb = n;

    // neg
    {
        auto want = ref::neg_f32(a);
        std::vector<float> got(n, 0.0f);
        std::vector<Arg> args = {Arg::in(a.data(), n * 4), Arg::out(got.data(), n * 4),
                                 Arg::in(&n, 4)};
        if (h.run("neg_f32", args, n)) report("neg_f32", got, want);
        else { std::printf("  FAIL  neg_f32: %s\n", h.error().c_str()); ++g_fail; }
    }

    struct BinOp { const char* fn; std::function<float(float, float)> f; };
    std::vector<BinOp> bins = {
        {"add_f32", [](float x, float y) { return x + y; }},
        {"sub_f32", [](float x, float y) { return x - y; }},
        {"mul_f32", [](float x, float y) { return x * y; }},
        {"div_f32", [](float x, float y) { return x / y; }},
        {"max_elem_f32", [](float x, float y) { return x > y ? x : (y > x ? y : x); }},
        {"min_elem_f32", [](float x, float y) { return x < y ? x : (y < x ? y : x); }},
    };
    for (auto& op : bins) {
        auto want = ref::binary_f32(a, b, op.f);
        std::vector<float> got(n, 0.0f);
        std::vector<Arg> args = {Arg::in(a.data(), n * 4), Arg::in(b.data(), n * 4),
                                 Arg::out(got.data(), n * 4), Arg::in(&n, 4),
                                 Arg::in(&la, 4), Arg::in(&lb, 4)};
        if (h.run(op.fn, args, n)) report(op.fn, got, want);
        else { std::printf("  FAIL  %s: %s\n", op.fn, h.error().c_str()); ++g_fail; }
    }

    // scalar (len-1) broadcast: temperature scale  logits / temp
    {
        std::vector<float> temp = {0.7f};
        std::uint32_t l1 = 1;
        auto want = ref::binary_f32(a, temp, [](float x, float y) { return x / y; });
        std::vector<float> got(n, 0.0f);
        std::vector<Arg> args = {Arg::in(a.data(), n * 4), Arg::in(temp.data(), 4),
                                 Arg::out(got.data(), n * 4), Arg::in(&n, 4),
                                 Arg::in(&la, 4), Arg::in(&l1, 4)};
        if (h.run("div_f32", args, n)) report("div_f32 (scalar temp broadcast)", got, want);
        else { std::printf("  FAIL  div_f32 scalar: %s\n", h.error().c_str()); ++g_fail; }
    }

    // comparison → bool bytes
    {
        auto want = ref::cmp_f32(a, b, [](float x, float y) { return x > y; });
        std::vector<std::uint8_t> got(n, 0);
        std::vector<Arg> args = {Arg::in(a.data(), n * 4), Arg::in(b.data(), n * 4),
                                 Arg::out(got.data(), n), Arg::in(&n, 4),
                                 Arg::in(&la, 4), Arg::in(&lb, 4)};
        if (h.run("gt_f32", args, n)) report_raw("gt_f32", got, want);
        else { std::printf("  FAIL  gt_f32: %s\n", h.error().c_str()); ++g_fail; }
    }
}

void test_reductions(MetalHarness& h) {
    const std::uint32_t rows = 3, len = 11;
    const std::uint32_t total = rows * len;
    std::vector<float> in(total);
    for (std::uint32_t i = 0; i < total; ++i)
        in[i] = static_cast<float>((i * 37) % 13) * 0.5f - 2.0f;

    struct RedF { const char* fn; float init; std::function<float(float, float)> f; };
    std::vector<RedF> reds = {
        {"reduce_sum_rows", 0.0f, [](float a, float b) { return a + b; }},
        {"reduce_max_rows", ref::kNegInf, [](float a, float b) { return a > b ? a : (b > a ? b : a); }},
        {"reduce_min_rows", -ref::kNegInf, [](float a, float b) { return a < b ? a : (b < a ? b : a); }},
    };
    for (auto& op : reds) {
        auto want = ref::reduce_rows(in, rows, len, op.init, op.f);
        std::vector<float> got(rows, 0.0f);
        std::vector<Arg> args = {Arg::in(in.data(), total * 4), Arg::out(got.data(), rows * 4),
                                 Arg::in(&rows, 4), Arg::in(&len, 4)};
        if (h.run(op.fn, args, rows)) report(op.fn, got, want);
        else { std::printf("  FAIL  %s: %s\n", op.fn, h.error().c_str()); ++g_fail; }
    }

    // argmax → I32 tokens (greedy)
    {
        auto want = ref::argmax_rows(in, rows, len);
        std::vector<std::int32_t> got(rows, 0);
        std::vector<Arg> args = {Arg::in(in.data(), total * 4), Arg::out(got.data(), rows * 4),
                                 Arg::in(&rows, 4), Arg::in(&len, 4)};
        if (h.run("reduce_argmax_rows", args, rows)) report_raw("reduce_argmax_rows", got, want);
        else { std::printf("  FAIL  reduce_argmax_rows: %s\n", h.error().c_str()); ++g_fail; }
    }

    // cumsum (top-p prefix)
    {
        auto want = ref::scan_rows(in, rows, len, 0.0f, [](float a, float b) { return a + b; });
        std::vector<float> got(total, 0.0f);
        std::vector<Arg> args = {Arg::in(in.data(), total * 4), Arg::out(got.data(), total * 4),
                                 Arg::in(&rows, 4), Arg::in(&len, 4)};
        if (h.run("cumsum_rows", args, rows)) report("cumsum_rows", got, want);
        else { std::printf("  FAIL  cumsum_rows: %s\n", h.error().c_str()); ++g_fail; }
    }
    {
        auto want = ref::scan_rows(in, rows, len, 1.0f, [](float a, float b) { return a * b; });
        std::vector<float> got(total, 0.0f);
        std::vector<Arg> args = {Arg::in(in.data(), total * 4), Arg::out(got.data(), total * 4),
                                 Arg::in(&rows, 4), Arg::in(&len, 4)};
        if (h.run("cumprod_rows", args, rows)) report("cumprod_rows", got, want);
        else { std::printf("  FAIL  cumprod_rows: %s\n", h.error().c_str()); ++g_fail; }
    }
}

void test_indexing(MetalHarness& h) {
    // gather: out[j] = src[idx[j]] with some invalid indices -> 0
    {
        std::vector<float> src = {10.0f, 11.0f, 12.0f, 13.0f, 14.0f};
        std::uint32_t src_len = static_cast<std::uint32_t>(src.size());
        std::vector<std::int32_t> idx = {0, 4, 2, -1, 5, 1};  // -1 and 5 invalid
        std::uint32_t k = static_cast<std::uint32_t>(idx.size());
        auto want = ref::gather_f32(src, idx);
        std::vector<float> got(k, 99.0f);
        std::vector<Arg> args = {Arg::in(src.data(), src.size() * 4),
                                 Arg::in(idx.data(), idx.size() * 4),
                                 Arg::out(got.data(), k * 4), Arg::in(&src_len, 4), Arg::in(&k, 4)};
        if (h.run("gather_f32", args, k)) report("gather_f32", got, want);
        else { std::printf("  FAIL  gather_f32: %s\n", h.error().c_str()); ++g_fail; }
    }
    // gather_row: accept-ratio p[i, draft[i]] with an invalid column
    {
        const std::uint32_t rows = 4, n = 5;
        std::vector<float> src(rows * n);
        for (std::uint32_t i = 0; i < rows * n; ++i) src[i] = static_cast<float>(i) * 0.1f;
        std::vector<std::int32_t> idx = {0, 3, 4, -1};  // last invalid -> 0
        auto want = ref::gather_row_f32(src, idx, rows, n);
        std::vector<float> got(rows, 99.0f);
        std::vector<Arg> args = {Arg::in(src.data(), src.size() * 4),
                                 Arg::in(idx.data(), idx.size() * 4),
                                 Arg::out(got.data(), rows * 4), Arg::in(&rows, 4), Arg::in(&n, 4)};
        if (h.run("gather_row_f32", args, rows)) report("gather_row_f32 (accept-ratio)", got, want);
        else { std::printf("  FAIL  gather_row_f32: %s\n", h.error().c_str()); ++g_fail; }
    }
}

}  // namespace

int main(int argc, char** argv) {
    std::string kernels_dir = argc > 1 ? argv[1] : PTIR_KERNELS_DIR;
    std::string kernel_src = kernels_dir + "/sampling_ir.metal";

    MetalHarness h;
    if (!h.ok()) {
        std::printf("PTIR_METAL_TEST_FAIL: %s\n", h.error().c_str());
        return 2;
    }
    std::printf("device: %s\n", h.device_name().c_str());
    if (!h.load_library(kernel_src)) {
        std::printf("PTIR_METAL_TEST_FAIL: %s\n", h.error().c_str());
        return 2;
    }

    test_mask_apply_packed(h);
    test_mask_apply_packed_matrix(h);
    test_dselect(h);
    test_broadcast_matrix(h);
    test_elementwise(h);
    test_reductions(h);
    test_indexing(h);

    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    if (g_fail == 0) {
        std::printf("PTIR_METAL_TEST_OK\n");
        return 0;
    }
    std::printf("PTIR_METAL_TEST_FAIL\n");
    return 1;
}
