// ptir_golden_cert — PTIR cross-backend golden certification on Metal.
//
// Proves the Metal tier-0 ops are bit-exact to echo's golden oracle (the same
// golden files gating CUDA, interface/sampling-ir/tests/golden-ptir/*.txt).
// For each golden:
//   1. decode the container with charlie's CUDA-free decoder (container.hpp) and
//      assert container_hash == the golden's identity hash  (SAME program bytes);
//   2. assert the decoded op-tag sequence matches the expected program (so a
//      container change is flagged, not silently mis-run);
//   3. feed the golden's OWN inputs (incl. the 1-byte host-Bool mask — the (B)
//      contract), execute on Metal, and match the expected `take` bytes.
//
// This cross-backend cert is exactly what catches a host-Bool dtype mismatch:
// the mask flows through the real Metal Select with the golden's 1-byte bytes.
//
// Usage: ptir_golden_cert [golden_dir] [kernels_dir]

#include <cstdint>
#include <cstdio>
#include <string>
#include <vector>

#include "ptir/container.hpp"  // charlie's CUDA-free PTIR decoder
#include "golden.hpp"
#include "metal_harness.hpp"
#include "reference.hpp"

#ifndef PTIR_KERNELS_DIR
#define PTIR_KERNELS_DIR "."
#endif
#ifndef PTIR_GOLDEN_DIR
#define PTIR_GOLDEN_DIR "."
#endif

using namespace ptir_metal;
namespace cont = pie_cuda_driver::ptir::container;

namespace {
int g_pass = 0, g_fail = 0;
void expect(bool ok, const std::string& what) {
    if (ok) { ++g_pass; std::printf("  PASS  %s\n", what.c_str()); }
    else { ++g_fail; std::printf("  FAIL  %s\n", what.c_str()); }
}

std::vector<std::uint8_t> flat_ops(const cont::Container& c) {
    std::vector<std::uint8_t> tags;
    for (const auto& st : c.stages)
        for (const auto& op : st.ops) tags.push_back(op.tag);
    return tags;
}

// Decode + assert container identity + op-tag structure. Returns true if the
// golden is the expected program.
bool decode_and_check(const golden::Golden& g, const std::vector<std::uint8_t>& want_tags,
                      cont::Container& c) {
    auto cb = golden::hex_to_bytes(g.container_hex);
    cont::DecodeError e;
    if (!cont::decode(cb.data(), cb.size(), c, &e)) {
        expect(false, g.name + ": container decode (" + e.detail + ")");
        return false;
    }
    expect(c.hash == g.hash, g.name + ": container_hash == golden hash (identity)");
    auto tags = flat_ops(c);
    expect(tags == want_tags, g.name + ": op-tag sequence matches expected program");
    return c.hash == g.hash && tags == want_tags;
}

// ── matrix_select_mask: select(host Bool mask, logits[4,8], -inf) -> argmax ──
void cert_matrix_select_mask(MetalHarness& h, const std::string& dir) {
    std::printf("[matrix_select_mask]\n");
    golden::Golden g;
    if (!golden::load(dir + "/matrix_select_mask.txt", g)) {
        expect(false, "matrix_select_mask: load");
        return;
    }
    // op DAG: intrinsic_val(logits) chan_take(mask) const(-inf) Broadcast Select
    //         ReduceArgmax chan_put
    const std::vector<std::uint8_t> want = {0xa0, 0x90, 0x81, 0x38, 0x20, 0x33, 0x92};
    cont::Container c;
    if (!decode_and_check(g, want, c)) return;

    // Inputs straight from the golden file.
    auto logits = golden::parse_f32(golden::bracket(g.line_with("logits: Some(F32(")));
    auto mask = golden::parse_bool(golden::bracket(g.line_with("host_put chan=0 = Bool(")));
    auto want_take = golden::parse_i32(golden::bracket(g.line_with("take chan=1 = I32(")));
    const std::uint32_t rows = 4, vocab = 8, total = rows * vocab;
    if (logits.size() != total || mask.size() != total || want_take.size() != rows) {
        expect(false, "matrix_select_mask: parsed input shapes");
        return;
    }
    expect(true, "matrix_select_mask: parsed golden inputs (logits[32], host-Bool mask[32] 1-byte, take[4])");

    // 1) neg_inf broadcast scalar -> [4,8]
    std::vector<float> neg_inf_src = {ref::kNegInf};
    std::vector<float> neg_inf(total, 0.0f);
    std::uint32_t sr = 1, sc = 1, dr = rows, dc = vocab;
    std::vector<Arg> a1 = {Arg::in(neg_inf_src.data(), 4), Arg::out(neg_inf.data(), total * 4),
                           Arg::in(&sr, 4), Arg::in(&sc, 4), Arg::in(&dr, 4), Arg::in(&dc, 4)};
    if (!h.run("broadcast_matrix_f32", a1, total)) { expect(false, std::string("broadcast: ") + h.error()); return; }

    // 2) select(mask_bool, logits, neg_inf)  — the (B) 1-byte host-Bool path
    std::vector<float> masked(total, 0.0f);
    std::uint32_t n = total, lc = total, la = total, lb = total;
    std::vector<Arg> a2 = {Arg::in(mask.data(), total), Arg::in(logits.data(), total * 4),
                           Arg::in(neg_inf.data(), total * 4), Arg::out(masked.data(), total * 4),
                           Arg::in(&n, 4), Arg::in(&lc, 4), Arg::in(&la, 4), Arg::in(&lb, 4)};
    if (!h.run("dselect_f32", a2, total)) { expect(false, std::string("dselect: ") + h.error()); return; }

    // 3) per-row argmax -> I32[4]
    std::vector<std::int32_t> got_take(rows, -1);
    std::vector<Arg> a3 = {Arg::in(masked.data(), total * 4), Arg::out(got_take.data(), rows * 4),
                           Arg::in(&rows, 4), Arg::in(&vocab, 4)};
    if (!h.run("reduce_argmax_rows", a3, rows)) { expect(false, std::string("argmax: ") + h.error()); return; }

    bool match = got_take == want_take;
    std::string got = "[";
    for (std::size_t i = 0; i < got_take.size(); ++i) got += std::to_string(got_take[i]) + (i + 1 < got_take.size() ? "," : "");
    got += "]";
    expect(match, "take chan=1 == golden I32([2,3,4,5]) (Metal got " + got + ")");
}

// ── beam_epilogue: decode + identity only (full behavioral exec is the larger
//    16-channel / top_k / log_softmax / geometry runner — staged follow-on). ──
void cert_beam_epilogue_identity(const std::string& dir) {
    std::printf("[beam_epilogue]\n");
    golden::Golden g;
    if (!golden::load(dir + "/beam_epilogue.txt", g)) {
        expect(false, "beam_epilogue: load");
        return;
    }
    cont::Container c;
    auto cb = golden::hex_to_bytes(g.container_hex);
    cont::DecodeError e;
    if (!cont::decode(cb.data(), cb.size(), c, &e)) {
        expect(false, "beam_epilogue: container decode (" + e.detail + ")");
        return;
    }
    expect(c.hash == g.hash, "beam_epilogue: container_hash == golden hash (identity)");
    std::size_t nops = 0;
    for (auto& st : c.stages) nops += st.ops.size();
    std::printf("  NOTE  beam_epilogue decoded: %zu channels, %zu ops — behavioral exec (top_k/"
                "log_softmax/geometry gathers+scatters, 16-channel multi-step) is the staged follow-on.\n",
                c.channels.size(), nops);
}

}  // namespace

int main(int argc, char** argv) {
    std::string golden_dir = argc > 1 ? argv[1] : PTIR_GOLDEN_DIR;
    std::string kernels_dir = argc > 2 ? argv[2] : PTIR_KERNELS_DIR;

    MetalHarness h;
    if (!h.ok()) { std::printf("PTIR_GOLDEN_CERT_FAIL: %s\n", h.error().c_str()); return 2; }
    std::printf("PTIR golden cert — device: %s, goldens: %s\n\n", h.device_name().c_str(), golden_dir.c_str());
    if (!h.load_library(kernels_dir + "/sampling_ir.metal")) {
        std::printf("PTIR_GOLDEN_CERT_FAIL: %s\n", h.error().c_str());
        return 2;
    }

    cert_matrix_select_mask(h, golden_dir);
    cert_beam_epilogue_identity(golden_dir);

    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    if (g_fail == 0) { std::printf("PTIR_GOLDEN_CERT_OK\n"); return 0; }
    std::printf("PTIR_GOLDEN_CERT_FAIL\n");
    return 1;
}
