// Recognizer (#8) decision-core unit test (lane L4 / echo): param-inference
// `SamplerParams → StandardSamplerKind` + the `kind → DispatchTarget` scorecard
// table. Pure host logic (no CUDA) — the executor's dispatch keystone.
//
// The inference predicates here are the canonical contract golf's explicit
// SDK/WIT canonical-kind tag mirrors 1:1 (so swapping param-inference → tag is
// behaviorally a no-op): T<=0→argmax, T>0&no-filters→temp, min_p>0→min-p,
// k>0→top-k, p<1→top-p.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <fstream>
#include <sstream>
#include <string>

#include "sampling_ir/sampler_dispatch.hpp"

using namespace pie_cuda_driver::sampling_ir;

namespace {
int g_failures = 0;
#define CHECK(cond)                                                       \
    do {                                                                  \
        if (!(cond)) {                                                    \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n", __FILE__, __LINE__, \
                         #cond);                                          \
            ++g_failures;                                                 \
        }                                                                 \
    } while (0)

SamplerParams params(float t, std::uint32_t k, float p, float mp) {
    SamplerParams sp;
    sp.temperature = t;
    sp.top_k = k;
    sp.top_p = p;
    sp.min_p = mp;
    return sp;
}

bool kind_from_string(const std::string& s, StandardSamplerKind* out) {
    if (s == "Argmax")        { *out = StandardSamplerKind::Argmax;      return true; }
    if (s == "Temperature")   { *out = StandardSamplerKind::Temperature; return true; }
    if (s == "MinP")          { *out = StandardSamplerKind::MinP;        return true; }
    if (s == "TopK")          { *out = StandardSamplerKind::TopK;        return true; }
    if (s == "TopP")          { *out = StandardSamplerKind::TopP;        return true; }
    if (s == "TopKTopP")      { *out = StandardSamplerKind::TopKTopP;    return true; }
    return false;
}

std::string trim(const std::string& s) {
    std::size_t a = s.find_first_not_of(" \t\r\n");
    if (a == std::string::npos) return "";
    std::size_t b = s.find_last_not_of(" \t\r\n");
    return s.substr(a, b - a + 1);
}

// Cross-language pin: the recognizer ladder golden vector is the single source
// of truth (golf factors it; both this C++ ctest and golf's Rust drift-guard
// read the same file at test runtime → the recognizers cannot drift). The C++
// host asserts EVERY row (incl. `host-only` combos a custom inferlet can hand
// the production recognizer bypassing the SDK sugar); the SDK test skips
// `host-only`. Format per line: `temperature,top_k,top_p,min_p,kind[,scope]`
// — `#`/blank lines ignored, scope (sdk|host-only) optional and unused here.
void read_and_check_golden(const char* path) {
    std::ifstream f(path);
    if (!f.is_open()) {
        std::fprintf(stderr,
                     "[recognizer-dispatch] NOTE: golden vector not found at %s "
                     "— cross-language pin INACTIVE (expected pre-merge of golf's "
                     "file; active once merged).\n",
                     path);
        return;
    }
    int rows = 0;
    std::string line;
    for (int lineno = 1; std::getline(f, line); ++lineno) {
        std::string t = trim(line);
        if (t.empty() || t[0] == '#') continue;
        // Strip an inline `# ...` comment (golf's host-only rows annotate the
        // kind column with a trailing comment).
        std::size_t hash = t.find('#');
        if (hash != std::string::npos) t = trim(t.substr(0, hash));
        if (t.empty()) continue;
        // Skip a column-header / non-data line (first field non-numeric).
        char c0 = t[0];
        if (!(c0 == '-' || c0 == '+' || c0 == '.' || (c0 >= '0' && c0 <= '9')))
            continue;
        std::stringstream ss(t);
        std::string c_t, c_k, c_p, c_mp, c_kind, c_scope;
        std::getline(ss, c_t, ',');
        std::getline(ss, c_k, ',');
        std::getline(ss, c_p, ',');
        std::getline(ss, c_mp, ',');
        std::getline(ss, c_kind, ',');
        std::getline(ss, c_scope, ',');
        c_kind = trim(c_kind);
        StandardSamplerKind expected;
        if (c_t.empty() || c_k.empty() || c_p.empty() || c_mp.empty() ||
            !kind_from_string(c_kind, &expected)) {
            std::fprintf(stderr, "FAIL: golden:%d: malformed row '%s'\n", lineno,
                         t.c_str());
            ++g_failures;
            continue;
        }
        SamplerParams sp = params(std::strtof(trim(c_t).c_str(), nullptr),
                                  static_cast<std::uint32_t>(std::strtoul(
                                      trim(c_k).c_str(), nullptr, 10)),
                                  std::strtof(trim(c_p).c_str(), nullptr),
                                  std::strtof(trim(c_mp).c_str(), nullptr));
        StandardSamplerKind got = infer_sampler_kind(sp);
        if (got != expected) {
            std::fprintf(stderr,
                         "FAIL: golden:%d: (T=%s k=%s p=%s min_p=%s) expected %s "
                         "got kind=%d\n",
                         lineno, trim(c_t).c_str(), trim(c_k).c_str(),
                         trim(c_p).c_str(), trim(c_mp).c_str(), c_kind.c_str(),
                         static_cast<int>(got));
            ++g_failures;
        }
        ++rows;
    }
    if (rows == 0) {
        std::fprintf(stderr,
                     "FAIL: golden vector at %s parsed 0 rows (malformed/empty)\n",
                     path);
        ++g_failures;
        return;
    }
    std::fprintf(stderr,
                 "[recognizer-dispatch] cross-language golden pin: %d rows from %s "
                 "verified against infer_sampler_kind\n",
                 rows, path);
}
}  // namespace

int main() {
    // ── infer_sampler_kind (param-inference) ────────────────────────────
    // Greedy → argmax regardless of k (T=0 picks the max).
    CHECK(infer_sampler_kind(params(0.0f, 1, 1.0f, 0.0f)) == StandardSamplerKind::Argmax);
    CHECK(infer_sampler_kind(params(0.0f, 0, 1.0f, 0.0f)) == StandardSamplerKind::Argmax);
    CHECK(infer_sampler_kind(params(-1.0f, 5, 0.9f, 0.1f)) == StandardSamplerKind::Argmax);
    // T>0, no filters → temperature.
    CHECK(infer_sampler_kind(params(0.8f, 0, 1.0f, 0.0f)) == StandardSamplerKind::Temperature);
    CHECK(infer_sampler_kind(params(1.0f, 0, 1.0f, 0.0f)) == StandardSamplerKind::Temperature);
    // min_p>0 → min-p.
    CHECK(infer_sampler_kind(params(0.8f, 0, 1.0f, 0.05f)) == StandardSamplerKind::MinP);
    // k>0 → top-k.
    CHECK(infer_sampler_kind(params(0.8f, 40, 1.0f, 0.0f)) == StandardSamplerKind::TopK);
    // p<1 → top-p.
    CHECK(infer_sampler_kind(params(0.8f, 0, 0.9f, 0.0f)) == StandardSamplerKind::TopP);
    // k>0 && p<1 → top-k-top-p.
    CHECK(infer_sampler_kind(params(0.8f, 40, 0.9f, 0.0f)) == StandardSamplerKind::TopKTopP);

    // ── dispatch_target (the scorecard table, manager's §2f ruling) ─────
    // temp → BakedIR (IR ~2× faster, token-exact → the de-hardwiring win).
    CHECK(dispatch_target(StandardSamplerKind::Temperature) == DispatchTarget::BakedIR);
    // min-p → DedicatedKernel: IR ~1.1× slower → keep sample_temp as its target
    // (de-hardwiring unaffected; fixed enum still gone, recognizer still decides).
    CHECK(dispatch_target(StandardSamplerKind::MinP) == DispatchTarget::DedicatedKernel);
    // argmax + top-k/top-p/top-k-top-p → DedicatedKernel (hand-tuned kernel wins).
    CHECK(dispatch_target(StandardSamplerKind::Argmax) == DispatchTarget::DedicatedKernel);
    CHECK(dispatch_target(StandardSamplerKind::TopK) == DispatchTarget::DedicatedKernel);
    CHECK(dispatch_target(StandardSamplerKind::TopP) == DispatchTarget::DedicatedKernel);
    CHECK(dispatch_target(StandardSamplerKind::TopKTopP) == DispatchTarget::DedicatedKernel);

    // ── runtime-leg drift pin: request → pi (per-slot params) → recognizer ──
    // Complements the SDK-sugar pin (foxtrot canonical_kind / golf drift-guard):
    // those pin the spec side; THIS pins the runtime `pi` side. The executor
    // normalizes top_k `0 → vocab_size` ("unset") for the dedicated kernels, so
    // the recognizer MUST treat top_k>=vocab as unset (via params_from_slot) —
    // else every Temperature/MinP/TopP/Argmax request mis-routes to TopK. Each
    // standard kind must survive request→pi→recognizer. (vocab = qwen3 151936.)
    const std::int32_t V = 151936;
    const std::int32_t KUNSET = V;  // executor's "no top-k" normalized value
    // Argmax request: greedy ⇒ T=0; top_k unset (or any) ⇒ still argmax.
    CHECK(infer_sampler_kind(params_from_slot(0.0f, KUNSET, 1.0f, 0.0f, V)) ==
          StandardSamplerKind::Argmax);
    CHECK(infer_sampler_kind(params_from_slot(0.0f, 5, 1.0f, 0.0f, V)) ==
          StandardSamplerKind::Argmax);
    // Temperature request: T>0, all filters unset (top_k normalized to vocab).
    CHECK(infer_sampler_kind(params_from_slot(0.8f, KUNSET, 1.0f, 0.0f, V)) ==
          StandardSamplerKind::Temperature);
    // MinP request: min_p>0, top_k unset ⇒ MinP (NOT TopK).
    CHECK(infer_sampler_kind(params_from_slot(0.8f, KUNSET, 1.0f, 0.05f, V)) ==
          StandardSamplerKind::MinP);
    // TopP request: 0<p<1, top_k unset ⇒ TopP (NOT TopKTopP).
    CHECK(infer_sampler_kind(params_from_slot(0.8f, KUNSET, 0.9f, 0.0f, V)) ==
          StandardSamplerKind::TopP);
    // TopK request: concrete k below vocab survives the normalization.
    CHECK(infer_sampler_kind(params_from_slot(0.8f, 40, 1.0f, 0.0f, V)) ==
          StandardSamplerKind::TopK);
    // TopKTopP request: both filters preserved (joint route, no p-drop).
    CHECK(infer_sampler_kind(params_from_slot(0.8f, 40, 0.9f, 0.0f, V)) ==
          StandardSamplerKind::TopKTopP);
    // Greedy-bit invariant: recognizer Argmax ⟺ legacy greedy detection (T<=0),
    // over a temperature sweep at unset top_k — the exact bit the executor's
    // fast_dense_greedy_argmax (h_temp[r] > 0.f) uses. They cannot diverge.
    for (float T : {-1.0f, -0.001f, 0.0f, 0.0001f, 0.5f, 1.0f, 2.0f}) {
        const bool argmax = infer_sampler_kind(params_from_slot(
                                T, KUNSET, 1.0f, 0.0f, V)) ==
                            StandardSamplerKind::Argmax;
        const bool legacy_greedy = !(T > 0.0f);
        CHECK(argmax == legacy_greedy);
    }

    // ── cross-language golden-vector pin (single source of truth) ───────
    // Asserts infer_sampler_kind against the same data file golf's Rust
    // drift-guard reads → the C++ host + Rust SDK recognizers cannot drift.
#ifdef PIE_RECOGNIZER_GOLDEN_CSV
    read_and_check_golden(PIE_RECOGNIZER_GOLDEN_CSV);
#else
    std::fprintf(stderr, "[recognizer-dispatch] NOTE: PIE_RECOGNIZER_GOLDEN_CSV "
                         "undefined — cross-language pin not wired in this build.\n");
#endif

    std::fprintf(stderr, "[recognizer-dispatch] param-inference + scorecard: %s (%d failures)\n",
                 g_failures == 0 ? "PASS" : "FAIL", g_failures);
    return g_failures == 0 ? 0 : 1;
}
