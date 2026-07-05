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

}  // namespace

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "tests/golden-ptir";
    cudaDeviceProp p{}; cudaGetDeviceProperties(&p, 0);
    std::printf("PTIR cross-backend golden step-exec — device: %s (sm_%d%d), goldens: %s\n\n",
                p.name, p.major, p.minor, dir.c_str());
    run_counter(dir);
    run_greedy(dir);
    std::printf("\n==== golden step-exec: %d passed, %d failed ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
