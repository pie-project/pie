// PTIR container reader conformance test.
//
// Decodes the golden `PTIR` v1 containers (interface/sampling-ir/tests/
// golden-ptir, vendored under tests/golden-ptir) and checks two things against
// the golden oracle byte-for-byte:
//   1. container_hash — my FNV-1a64 over the raw bytes == the golden `hash:` line;
//   2. readiness table — my per-channel first-op direction derivation ==
//      the golden `readiness:` lines (C2's producer, the stage-runner's input).
// Negative (validator-ERR) goldens are checked for a clean structural decode +
// hash only — the semantic verdict is the validator's job, not the reader's.
//
// Host-only (no CUDA). Golden dir = argv[1] (default tests/golden-ptir).
//   g++ -std=c++17 -Isrc tests/ptir_container_test.cpp -o ptir_container_test
//   ./ptir_container_test tests/golden-ptir

#include <cstdint>
#include <cstdio>
#include <fstream>
#include <map>
#include <sstream>
#include <string>
#include <vector>

#include "pie_native/ptir/container.hpp"

using namespace pie_native::ptir;

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

struct Golden {
    std::string name, verdict, container_hex;
    std::uint64_t hash = 0;
    bool has_hash = false;
    // chan -> (phase, dir_needs_empty)
    std::map<std::uint32_t, std::pair<std::uint8_t, bool>> readiness;
};

std::string trim(const std::string& s) {
    std::size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    std::size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

Golden parse_golden(const std::string& path) {
    Golden g;
    std::ifstream f(path);
    std::string line;
    while (std::getline(f, line)) {
        auto colon = line.find(':');
        if (colon == std::string::npos) continue;
        std::string key = trim(line.substr(0, colon));
        std::string val = trim(line.substr(colon + 1));
        if (key == "name") g.name = val;
        else if (key == "hash") { g.hash = std::stoull(val, nullptr, 16); g.has_hash = true; }
        else if (key == "container") g.container_hex = val;
        else if (key == "verdict") g.verdict = val;
        else if (key == "readiness") {
            // "chan=N phase=0xNN dir=NeedsFull"
            std::uint32_t chan = 0; std::uint32_t phase = 0; bool empty = false;
            std::istringstream ss(val); std::string tok;
            while (ss >> tok) {
                if (tok.rfind("chan=", 0) == 0) chan = (std::uint32_t)std::stoul(tok.substr(5));
                else if (tok.rfind("phase=", 0) == 0) phase = (std::uint32_t)std::stoul(tok.substr(6), nullptr, 16);
                else if (tok.rfind("dir=", 0) == 0) empty = (tok.substr(4) == "NeedsEmpty");
            }
            g.readiness[chan] = {(std::uint8_t)phase, empty};
        }
    }
    return g;
}

void run_one(const std::string& dir, const std::string& name) {
    Golden g = parse_golden(dir + "/" + name + ".txt");
    if (g.container_hex.empty()) { expect(false, name + ": no container hex"); return; }
    auto bytes = hex_to_bytes(g.container_hex);

    container::Container c;
    container::DecodeError err;
    bool ok = container::decode(bytes.data(), bytes.size(), c, &err);
    expect(ok, name + ": decode (" + (ok ? "ok" : err.detail) + ")");
    if (!ok) return;

    expect(!g.has_hash || c.hash == g.hash, name + ": container_hash");

    bool is_negative = g.verdict.rfind("ERR", 0) == 0;
    if (is_negative) return;   // validator verdict is external; reader only decodes+hashes

    auto rt = container::derive_readiness(c);
    std::map<std::uint32_t, std::pair<std::uint8_t, bool>> got;
    for (auto& e : rt) got[e.chan] = {e.phase, e.dir == container::Direction::NeedsEmpty};
    bool match = (got.size() == g.readiness.size());
    for (auto& kv : g.readiness) {
        auto it = got.find(kv.first);
        if (it == got.end() || it->second != kv.second) { match = false; break; }
    }
    expect(match, name + ": readiness table (" + std::to_string(got.size()) + " chans)");
    if (!match) {
        for (auto& kv : g.readiness) {
            auto it = got.find(kv.first);
            std::printf("        want chan=%u phase=0x%02x %s | got %s\n", kv.first, kv.second.first,
                        kv.second.second ? "NeedsEmpty" : "NeedsFull",
                        it == got.end() ? "(missing)" :
                        (std::string("phase=0x") + (it->second.first < 16 ? "0" : "") ).c_str());
        }
    }
}
}  // namespace

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "tests/golden-ptir";
    std::printf("PTIR container reader conformance vs golden vectors (%s)\n", dir.c_str());
    const char* names[] = {
        "greedy_argmax", "counter_pingpong", "section3_masked_gumbel",
        "beam_epilogue", "staged_dispatch",
        "neg_spsc_second_producer", "neg_sink_at_epilogue", "neg_t10_nonreplayable",
        "neg_intrinsic_wrong_stage", "neg_model_gated_missing", "neg_body_type_error",
    };
    for (const char* n : names) run_one(dir, n);
    std::printf("\n==== container conformance: %d passed, %d failed ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
