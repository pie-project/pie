// PTIR cross-backend golden step-exec gate (charlie, thrust-3 P4). The headline
// conformance test: decode echo's ACTUAL golden container bytes + PTIB typed
// sidecar (bound.hpp), translate to an executable Trace, run it through the
// tier-0 stage-runner with echo's canonical inputs/seeds, and match echo's
// step/take results (committed flags + taken token values) byte-for-byte.
//
// Inputs + seeds are transcribed from echo's generator (interface/sampling-ir/
// tests/ptir_golden.rs); the container/sidecar hex + expected hash come straight
// from the vendored golden files, so this pins the same cross-language vectors.
//
//   nvcc -std=c++17 -arch=sm_89 --extended-lambda --expt-relaxed-constexpr \
//        -Isrc tests/ptir_golden_exec_test.cu -o ptir_golden_exec_test
//   ./ptir_golden_exec_test tests/golden-ptir

#include <cstdint>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <sstream>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "ptir/bound.hpp"
#include "ptir/tier0_runner.hpp"

using namespace pie_cuda_driver::ptir;

namespace {
int g_pass = 0, g_fail = 0;
void expect(bool ok, const std::string& what) {
    if (ok) { ++g_pass; std::printf("  PASS  %s\n", what.c_str()); }
    else    { ++g_fail; std::printf("  FAIL  %s\n", what.c_str()); }
}

std::vector<std::uint8_t> hex_to_bytes(const std::string& h) {
    std::vector<std::uint8_t> b;
    for (std::size_t i = 0; i + 1 < h.size(); i += 2)
        b.push_back((std::uint8_t)std::stoul(h.substr(i, 2), nullptr, 16));
    return b;
}
std::string trim(const std::string& s) {
    std::size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    return s.substr(a, s.find_last_not_of(" \t\r\n") - a + 1);
}
// pull "container:" and "sidecar:" hex + "hash:" from a golden file.
bool load_golden(const std::string& path, std::string& container_hex, std::string& sidecar_hex,
                 std::uint64_t& hash) {
    std::ifstream f(path); std::string line; bool got_c = false;
    while (std::getline(f, line)) {
        auto c = line.find(':'); if (c == std::string::npos) continue;
        std::string k = trim(line.substr(0, c)), v = trim(line.substr(c + 1));
        if (k == "container") { container_hex = v; got_c = true; }
        else if (k == "sidecar") sidecar_hex = v;
        else if (k == "hash") hash = std::stoull(v, nullptr, 16);
    }
    return got_c;
}

// Decode + translate a golden into an executable Trace, asserting the identity
// chain (container hash == sidecar's inner hash == file's hash: line).
bool build_trace(const std::string& dir, const std::string& name, Trace& out) {
    std::string chex, shex; std::uint64_t fhash = 0;
    if (!load_golden(dir + "/" + name + ".txt", chex, shex, fhash)) { expect(false, name + ": load"); return false; }
    auto cb = hex_to_bytes(chex), sb = hex_to_bytes(shex);
    container::Container c; container::DecodeError e;
    if (!container::decode(cb.data(), cb.size(), c, &e)) { expect(false, name + ": decode " + e.detail); return false; }
    bound::Bound b; std::string se;
    if (!bound::parse_sidecar(sb.data(), sb.size(), b, &se)) { expect(false, name + ": sidecar " + se); return false; }
    expect(c.hash == fhash && b.container_hash == fhash, name + ": identity chain (container==sidecar==file hash)");
    auto tr = bound::container_to_trace(c, b);
    if (!tr.ok) { expect(false, name + ": translate " + tr.error); return false; }
    out = std::move(tr.trace);
    return true;
}

// ── counter_pingpong: chan0 seeded [10]; ctr := ctr+1; out back-pressures ──
void run_counter(const std::string& dir) {
    std::printf("[counter_pingpong]\n");
    Trace t; if (!build_trace(dir, "counter_pingpong", t)) return;
    Tier0Runner runner(t);
    std::uint32_t seed = 10; runner.arena().seed_cell(0, &seed, sizeof(seed));
    FireInputs in;

    PassResult r0 = runner.run_pass(in);
    expect(r0.ok && r0.committed, "step 0: committed=true");
    PassResult r1 = runner.run_pass(in);
    expect(r1.ok && !r1.committed, "step 1: committed=false (out back-pressure)");
    std::uint32_t v = 0;
    expect(runner.arena().committed_full(1), "take chan=1 ready");
    runner.arena().host_take(1, &v, sizeof(v));
    expect(v == 11, "take chan=1 == 11 (got " + std::to_string(v) + ")");
    PassResult r2 = runner.run_pass(in);
    expect(r2.ok && r2.committed, "step 2: committed=true");
    runner.arena().host_take(1, &v, sizeof(v));
    expect(v == 12, "take chan=1 == 12 (got " + std::to_string(v) + ")");
    expect(!runner.arena().committed_full(1), "take chan=1 == WouldBlock (empty)");
}

// ── greedy_argmax: chan0 seeded token [1] via embed_tokens; argmax(logits) ──
void run_greedy(const std::string& dir) {
    std::printf("[greedy_argmax]\n");
    Trace t; if (!build_trace(dir, "greedy_argmax", t)) return;
    Tier0Runner runner(t);
    std::int32_t seed = 1; runner.arena().seed_cell(0, &seed, sizeof(seed));

    float* d_logits = nullptr; cudaMalloc(&d_logits, 8 * sizeof(float));
    auto step = [&](std::vector<float> logits, std::int32_t want) {
        cudaMemcpy(d_logits, logits.data(), 8 * sizeof(float), cudaMemcpyHostToDevice);
        FireInputs in; in.logits = d_logits; in.vocab = 8;
        PassResult r = runner.run_pass(in);
        std::int32_t v = -1; runner.arena().host_take(1, &v, sizeof(v));
        expect(r.ok && r.committed && v == want,
               "token == " + std::to_string(want) + " (got " + std::to_string(v) + ", committed=" + (r.committed?"T":"F") + ")");
    };
    step({0, 1, 9, 2, 0, 0, 0, 3}, 2);   // echo step 0
    step({7, 1, 0, 2, 0, 0, 0, 3}, 0);   // echo step 1
    cudaFree(d_logits);
}

// ── section3_masked_gumbel: overview §3 — greedy + grammar mask + gumbel, with
//    the late-mask dummy-run + recover (P4 exit criterion). VOCAB=32. ──
void run_section3(const std::string& dir) {
    std::printf("[section3_masked_gumbel]\n");
    Trace t; if (!build_trace(dir, "section3_masked_gumbel", t)) return;
    const std::uint32_t V = 32;
    Tier0Runner runner(t);
    // seeds (echo ptir_golden.rs): chan0 tok=[1] i32, chan3 len=[1] u32, chan4 rng=[1234,0] u32
    std::int32_t s0 = 1;  runner.arena().seed_cell(0, &s0, sizeof(s0));
    std::uint32_t s3 = 1; runner.arena().seed_cell(3, &s3, sizeof(s3));
    std::uint32_t s4[2] = {1234, 0}; runner.arena().seed_cell(4, s4, sizeof(s4));

    // logits = flat_logits(7, 100.0): all 0 except index 7 = 100.
    std::vector<float> logits(V, 0.f); logits[7] = 100.f;
    float* d_logits = nullptr; cudaMalloc(&d_logits, V * sizeof(float));
    cudaMemcpy(d_logits, logits.data(), V * sizeof(float), cudaMemcpyHostToDevice);
    // rng seed buffer unused for rng_keyed (state rides chan4) but bind a dummy.
    std::uint32_t rs = 0; std::uint32_t* d_rs = nullptr; cudaMalloc(&d_rs, sizeof(rs));
    cudaMemcpy(d_rs, &rs, sizeof(rs), cudaMemcpyHostToDevice);
    FireInputs in; in.logits = d_logits; in.vocab = V; in.row_seeds = d_rs;

    auto feed_mask = [&](const std::vector<std::uint8_t>& m) {
        runner.arena().host_feed(2, m.data(), m.size());   // Bool[32] unpacked
    };
    std::vector<std::uint8_t> allow_all(V, 1);
    std::vector<std::uint8_t> allow_only3(V, 0); allow_only3[3] = 1;

    // step 0: mask = allow_all → argmax(logits) = 7.
    feed_mask(allow_all);
    PassResult r0 = runner.run_pass(in);
    std::int32_t v = -1; runner.arena().host_take(1, &v, sizeof(v));
    expect(r0.ok && r0.committed && v == 7, "step 0: token == 7 (got " + std::to_string(v) + ", committed=" + (r0.committed?"T":"F") + ")");

    // step 1: mask channel now empty (consumed) → late-mask MISS, dummy-run.
    PassResult r1 = runner.run_pass(in);
    expect(r1.ok && !r1.committed, "step 1: late-mask dummy-run (committed=false)");
    expect(!runner.arena().committed_full(1), "step 1: out == WouldBlock");

    // step 2: mask = allow_only([3]) → only token 3 finite → argmax = 3.
    feed_mask(allow_only3);
    PassResult r2 = runner.run_pass(in);
    runner.arena().host_take(1, &v, sizeof(v));
    expect(r2.ok && r2.committed && v == 3, "step 2: recover, token == 3 (got " + std::to_string(v) + ", committed=" + (r2.committed?"T":"F") + ")");

    cudaFree(d_logits); cudaFree(d_rs);
}

// ── beam_epilogue: overview §6.2 — 16 chans, geometry gathers/flat-scatters/
//    top_k/log_softmax. step0 misses (no fresh grant), step1 commits. ──
void run_beam(const std::string& dir) {
    std::printf("[beam_epilogue]\n");
    Trace t; if (!build_trace(dir, "beam_epilogue", t)) return;
    const std::uint32_t BB = 2, V = 8, P = 3, PAGE = 4;
    Tier0Runner runner(t);
    auto seedU = [&](ChannelId c, std::vector<std::uint32_t> v) { runner.arena().seed_cell(c, v.data(), v.size() * 4); };
    auto seedI = [&](ChannelId c, std::vector<std::int32_t> v) { runner.arena().seed_cell(c, v.data(), v.size() * 4); };
    auto seedF = [&](ChannelId c, std::vector<float> v) { runner.arena().seed_cell(c, v.data(), v.size() * 4); };
    seedU(0, {5,6,0, 5,6,0});        // pages [2,3]
    seedU(1, {4,2,0, 4,2,0});        // lens  [2,3]
    seedU(2, {6,6});                 // klen  [2]
    std::vector<std::uint8_t> kvm(BB * P * PAGE);   // kvm[lane][j*PAGE+o]=o<lens
    std::uint32_t lens_j[3] = {4,2,0};
    for (std::uint32_t lane = 0; lane < BB; ++lane)
        for (std::uint32_t j = 0; j < P; ++j)
            for (std::uint32_t o = 0; o < PAGE; ++o)
                kvm[lane*(P*PAGE) + j*PAGE + o] = (o < lens_j[j]) ? 1 : 0;
    runner.arena().seed_cell(3, kvm.data(), kvm.size());  // kvm [2,12] bool
    seedU(4, {6,6});                 // pos
    seedU(5, {2,2});                 // np
    seedU(6, {6,6});                 // tslot
    seedU(7, {2,2});                 // tfill
    seedU(8, {6,6});                 // w_slot
    seedU(9, {2,2});                 // w_off
    seedI(10, {1,2});                // toks
    seedF(11, {0.f,0.f});            // scores

    std::vector<float> logits(BB * V, 0.f); logits[3] = 8.f; logits[V + 5] = 7.f;
    float* d_logits = nullptr; cudaMalloc(&d_logits, logits.size() * sizeof(float));
    cudaMemcpy(d_logits, logits.data(), logits.size() * sizeof(float), cudaMemcpyHostToDevice);
    std::uint32_t rs = 0; std::uint32_t* d_rs = nullptr; cudaMalloc(&d_rs, sizeof(rs)); cudaMemcpy(d_rs, &rs, 4, cudaMemcpyHostToDevice);
    FireInputs in; in.logits = d_logits; in.vocab = V; in.row_seeds = d_rs;

    // step 0: no fresh grant (chan12 empty) → miss.
    PassResult r0 = runner.run_pass(in);
    expect(r0.ok && !r0.committed, "step 0: miss (no fresh grant)");
    // host grants fresh slots [7,8], then step 1 commits.
    std::vector<std::uint32_t> fresh{7,8}; runner.arena().host_feed(12, fresh.data(), fresh.size()*4);
    PassResult r1 = runner.run_pass(in);
    expect(r1.ok && r1.committed, "step 1: committed");

    std::int32_t out[2]; runner.arena().host_take(13, out, sizeof(out));
    expect(out[0] == 3 && out[1] == 5, "take chan13 out == [3,5] (got [" + std::to_string(out[0]) + "," + std::to_string(out[1]) + "])");
    std::uint32_t par[2]; runner.arena().host_take(14, par, sizeof(par));
    expect(par[0] == 0 && par[1] == 1, "take chan14 out_par == [0,1] (got [" + std::to_string(par[0]) + "," + std::to_string(par[1]) + "])");
    float scr[2]; runner.arena().host_take(15, scr, sizeof(scr));
    auto near = [](float a, float b) { return std::fabs(a - b) <= 1e-4f + 1e-4f * std::fabs(b); };
    bool sok = near(scr[0], -0.0023454318f) && near(scr[1], -0.006362776f);
    char buf[64]; std::snprintf(buf, sizeof(buf), "[%g, %g]", scr[0], scr[1]);
    expect(sok, std::string("take chan15 out_scr == [-0.00235,-0.00636] (got ") + buf + ")");

    cudaFree(d_logits); cudaFree(d_rs);
}

}  // namespace

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "tests/golden-ptir";
    cudaDeviceProp p{}; cudaGetDeviceProperties(&p, 0);
    std::printf("PTIR cross-backend golden step-exec — device: %s (sm_%d%d), goldens: %s\n\n",
                p.name, p.major, p.minor, dir.c_str());
    run_counter(dir);
    run_greedy(dir);
    run_section3(dir);
    run_beam(dir);
    std::printf("\n==== golden step-exec: %d passed, %d failed ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
