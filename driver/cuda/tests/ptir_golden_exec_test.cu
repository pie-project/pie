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
#include "ptir/program_runtime.hpp"
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

// ── program_runtime: drive greedy_argmax through the P2c DRIVER-SIDE runtime
//    (PtirProgramCache hash-decode + PtirInstance seed/fire) — the entry point
//    delta's ~10-line submit-fire calls. Proves: first-fire container+sidecar
//    decode + cache-by-hash, steady-state cache hit on empty bytes, a loud miss
//    on an uncached hash, and a seeded per-instance fire reproducing echo's
//    argmax vectors through the wrapper (not the raw runner). ──
void run_via_runtime(const std::string& dir) {
    std::printf("[program_runtime greedy_argmax]\n");
    std::string chex, shex; std::uint64_t fhash = 0;
    if (!load_golden(dir + "/greedy_argmax.txt", chex, shex, fhash)) { expect(false, "runtime: load"); return; }
    auto cb = hex_to_bytes(chex), sb = hex_to_bytes(shex);

    PtirProgramCache cache;
    std::string err;
    // First fire of the hash ships container + sidecar → decode + cache.
    const Trace* t = cache.get_or_decode(fhash, cb.data(), cb.size(), sb.data(), sb.size(), &err);
    expect(t != nullptr, "runtime: first-fire decode+cache (" + err + ")");
    if (!t) return;
    expect(cache.size() == 1 && cache.contains(fhash), "runtime: cached by hash");
    // Steady state: empty bytes MUST hit the cache + return the SAME Trace.
    const Trace* t2 = cache.get_or_decode(fhash, nullptr, 0, nullptr, 0, &err);
    expect(t2 == t, "runtime: steady-state cache hit (same Trace, empty bytes)");
    // An uncached hash with empty bytes must FAIL loudly, never decode garbage.
    const Trace* miss = cache.get_or_decode(fhash ^ 0x1ull, nullptr, 0, nullptr, 0, &err);
    expect(miss == nullptr, "runtime: uncached hash + empty bytes → loud miss");

    // Instantiate with the D2 seed (chan0 token=[1], i32 LE) + fire echo's steps.
    std::vector<ChannelValue> seeds = {{0, {1, 0, 0, 0}}};
    PtirInstance inst(*t, seeds);
    float* d_logits = nullptr; cudaMalloc(&d_logits, 8 * sizeof(float));
    auto step = [&](std::vector<float> logits, std::int32_t want) {
        cudaMemcpy(d_logits, logits.data(), 8 * sizeof(float), cudaMemcpyHostToDevice);
        FireInputs in; in.logits = d_logits; in.vocab = 8;
        PassResult r = inst.fire({}, in);   // greedy: seed + logits only, no host_puts
        auto outs = inst.harvest_outputs(); // the (channel, wire_bytes) table for delta's response SoA
        bool shape = (outs.size() == 1 && outs[0].first == 1 && outs[0].second.size() == 4);
        std::int32_t v = shape ? *reinterpret_cast<const std::int32_t*>(outs[0].second.data()) : -1;
        expect(r.ok && r.committed && shape && v == want,
               "runtime: harvest_outputs [(1," + std::to_string(want) + ")] (chans=" +
               std::to_string(outs.size()) + " v=" + std::to_string(v) + ")");
    };
    step({0, 1, 9, 2, 0, 0, 0, 3}, 2);   // echo step 0
    step({7, 1, 0, 2, 0, 0, 0, 3}, 0);   // echo step 1
    cudaFree(d_logits);
}

// ── program_runtime STATEFUL: counter_pingpong through PtirInstance proves the
//    per-instance arena PERSISTS across fires (the counter 10→11→12 survives
//    each fire's commit) — the "persistent instance" lifecycle (one instance,
//    many fires, seeds applied ONCE at instantiation). ──
void run_via_runtime_stateful(const std::string& dir) {
    std::printf("[program_runtime counter_pingpong (stateful persistence)]\n");
    std::string chex, shex; std::uint64_t fhash = 0;
    if (!load_golden(dir + "/counter_pingpong.txt", chex, shex, fhash)) { expect(false, "runtime-sf: load"); return; }
    auto cb = hex_to_bytes(chex), sb = hex_to_bytes(shex);
    PtirProgramCache cache; std::string err;
    const Trace* t = cache.get_or_decode(fhash, cb.data(), cb.size(), sb.data(), sb.size(), &err);
    expect(t != nullptr, "runtime-sf: decode+cache (" + err + ")");
    if (!t) return;

    // Seed chan0 = 10 (u32) ONCE at instantiation; the arena persists across fires.
    std::vector<ChannelValue> seeds = {{0, {10, 0, 0, 0}}};
    PtirInstance inst(*t, seeds);
    FireInputs in;

    PassResult r0 = inst.fire({}, in);
    expect(r0.ok && r0.committed, "runtime-sf: fire 0 committed");
    PassResult r1 = inst.fire({}, in);
    expect(r1.ok && !r1.committed, "runtime-sf: fire 1 back-pressure (!committed)");
    std::uint32_t v = 0;
    bool got = inst.take_output(1, &v, sizeof(v));
    expect(got && v == 11, "runtime-sf: take chan1 == 11 (persisted +1; got " + std::to_string(v) + ")");
    PassResult r2 = inst.fire({}, in);
    expect(r2.ok && r2.committed, "runtime-sf: fire 2 committed");
    inst.take_output(1, &v, sizeof(v));
    expect(v == 12, "runtime-sf: take chan1 == 12 (persisted +2; got " + std::to_string(v) + ")");
}

// ── mtp_verify_tail: echo's §6.1 match-verify K=3 cross-backend anchor. Exercises
//    the Stage-2 [K,vocab] MtpLogits MATRIX read — the K draft rows live in
//    ws.logits AFTER the sample rows (mtp_draft_row = 4), read as a [3,8] matrix.
//    logits [4,8] are the verify positions (K+1); a Bool[4,8] mask forces a
//    mid-prefix miss at row 2 (only token 2 finite) so the accept-prefix is
//    [3,5,2,-1]; chan3 = per-row argmax of mtp_logits = [1,4,0]. Injects synthetic
//    draft logits, so it validates the READ semantics with no MTP model. ──
void run_mtp_verify_tail(const std::string& dir) {
    std::printf("[mtp_verify_tail]\n");
    Trace t; if (!build_trace(dir, "mtp_verify_tail", t)) return;
    const std::uint32_t V = 8, NROWS = 4;  // K=3 drafts + 1 bonus verify position
    Tier0Runner runner(t);
    // seed chan0 = draft tokens [3,5,6] (I32) to verify.
    std::int32_t drafts[3] = {3, 5, 6};
    runner.arena().seed_cell(0, drafts, sizeof(drafts));
    // Pack [logits(4x8) | mtp_logits(3x8)] into one ws.logits base; MtpLogits reads
    // the 3 draft rows at mtp_draft_row = NROWS (the model-faithful layout).
    std::vector<float> packed = {
        // logits [4,8] — verify positions
        0,0,0,9,0,0,0,0,  0,0,0,0,0,9,0,0,  0,0,1,0,0,0,9,0,  0,0,0,0,9,0,0,0,
        // mtp_logits [3,8] — the K draft rows
        0,7,0,0,0,0,0,0,  0,0,0,0,7,0,0,0,  7,0,0,0,0,0,0,0,
    };
    float* d_logits = nullptr; cudaMalloc(&d_logits, packed.size() * sizeof(float));
    cudaMemcpy(d_logits, packed.data(), packed.size() * sizeof(float), cudaMemcpyHostToDevice);
    FireInputs in; in.logits = d_logits; in.vocab = V; in.mtp_draft_row = static_cast<int>(NROWS);
    // host_put chan1 = Bool[4,8] mask (unpacked, 1 byte/bool). Row 2 = only token 2
    // finite → forces the accept-prefix miss.
    std::vector<std::uint8_t> mask = {
        1,1,1,1,1,1,1,1,  1,1,1,1,1,1,1,1,  0,0,1,0,0,0,0,0,  1,1,1,1,1,1,1,1,
    };
    runner.arena().host_feed(1, mask.data(), mask.size());

    PassResult r = runner.run_pass(in);
    expect(r.ok && r.committed,
           "step 0: committed=true (ok=" + std::string(r.ok ? "T" : "F") +
               " committed=" + (r.committed ? "T" : "F") + " err=" + r.error + ")");
    // take chan2 = accept-prefix [3,5,2,-1] (I32[4]).
    std::int32_t acc[4] = {0, 0, 0, 0};
    runner.arena().host_take(2, acc, sizeof(acc));
    expect(acc[0] == 3 && acc[1] == 5 && acc[2] == 2 && acc[3] == -1,
           "take chan2 accept-prefix == [3,5,2,-1] (got [" + std::to_string(acc[0]) + "," +
               std::to_string(acc[1]) + "," + std::to_string(acc[2]) + "," +
               std::to_string(acc[3]) + "])");
    // take chan3 = per-row argmax of the [3,8] MtpLogits matrix = [1,4,0].
    std::int32_t mtp[3] = {0, 0, 0};
    runner.arena().host_take(3, mtp, sizeof(mtp));
    expect(mtp[0] == 1 && mtp[1] == 4 && mtp[2] == 0,
           "take chan3 mtp-argmax == [1,4,0] (got [" + std::to_string(mtp[0]) + "," +
               std::to_string(mtp[1]) + "," + std::to_string(mtp[2]) + "])");
    cudaFree(d_logits);
}

// ── dfa_ingraph: capstone pentathlon technique — an IN-GRAPH grammar (DFA) walk.
//    The grammar state (chan2, class=InPlace) persists in-graph across steps; the
//    allow-mask table (chan0 Bool[3*8]) and next-state table (chan1 U32[3*8]) are
//    seeded once and READ every step (not consumed). Each step: gather the current
//    state's allow row → mask logits → argmax = the constrained token → gather the
//    next state from the transition table → update chan2 in place. VOCAB=8, a
//    3-state DFA. This is the CUDA tier-0 analog of echo's cross-backend golden and
//    validates the InPlace channel class + loop-carried grammar state. ──
void run_dfa_ingraph(const std::string& dir) {
    std::printf("[dfa_ingraph]\n");
    Trace t; if (!build_trace(dir, "dfa_ingraph", t)) return;
    const std::uint32_t V = 8;
    Tier0Runner runner(t);
    // seeds (echo ptir_golden.rs): chan0 = allow-mask table Bool[3*8] (state-major),
    // chan1 = next-state table U32[3*8], chan2 = current DFA state U32([0]) (InPlace).
    std::uint8_t allow[24] = {
        0,1,1,0,0,0,0,0,   // state 0: tokens {1,2}
        0,0,0,1,0,0,0,0,   // state 1: token  {3}
        1,0,0,0,0,0,0,0,   // state 2: token  {0}
    };
    std::uint32_t next[24] = {
        0,1,1,0,0,0,0,0,   // state 0: on tok 1|2 → state 1
        0,0,0,2,0,0,0,0,   // state 1: on tok 3   → state 2
        2,0,0,0,0,0,0,0,   // state 2: on tok 0   → state 2
    };
    std::uint32_t s0 = 0;
    runner.arena().seed_cell(0, allow, sizeof(allow));
    runner.arena().seed_cell(1, next, sizeof(next));
    runner.arena().seed_cell(2, &s0, sizeof(s0));

    float* d_logits = nullptr; cudaMalloc(&d_logits, V * sizeof(float));
    auto step = [&](std::vector<float> logits, std::int32_t want) {
        cudaMemcpy(d_logits, logits.data(), V * sizeof(float), cudaMemcpyHostToDevice);
        FireInputs in; in.logits = d_logits; in.vocab = V;
        PassResult r = runner.run_pass(in);
        std::int32_t v = -1; runner.arena().host_take(3, &v, sizeof(v));
        expect(r.ok && r.committed && v == want,
               "token == " + std::to_string(want) + " (got " + std::to_string(v) +
                   ", committed=" + (r.committed ? "T" : "F") + " err=" + r.error + ")");
    };
    step({0, 0, 1, 0, 0, 9, 0, 0}, 2);   // state 0: {1,2} allowed → argmax over allowed = 2
    step({0, 0, 0, 1, 0, 0, 9, 0}, 3);   // state 1: {3}   allowed → 3
    step({1, 0, 0, 0, 0, 0, 0, 9}, 0);   // state 2: {0}   allowed → 0
    cudaFree(d_logits);
}

int main(int argc, char** argv) {
    std::string dir = argc > 1 ? argv[1] : "tests/golden-ptir";
    cudaDeviceProp p{}; cudaGetDeviceProperties(&p, 0);
    std::printf("PTIR cross-backend golden step-exec — device: %s (sm_%d%d), goldens: %s\n\n",
                p.name, p.major, p.minor, dir.c_str());
    run_counter(dir);
    run_greedy(dir);
    run_via_runtime(dir);
    run_via_runtime_stateful(dir);
    run_section3(dir);
    run_beam(dir);
    run_mtp_verify_tail(dir);
    run_dfa_ingraph(dir);
    std::printf("\n==== golden step-exec: %d passed, %d failed ====\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
