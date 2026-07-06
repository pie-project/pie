// pentathlon_test — cross-backend cert of echo's `pentathlon_iter` capstone on
// Metal tier-0. One MCTS iteration composing all six techniques (quest per-layer
// envelope_dot→page-mask sink, beam top_k, grammar masks, speculative MTP-verify
// tail, contrastive expert−λ·amateur, MCTS value taps) across two programs
// (pentathlon_expand + pentathlon_rollout). Replays the golden's scripted phases
// (expand A/B → rollout A5/B3) and asserts step / sink / take bit-exact-within-tol
// to echo's eval.rs oracle — proving the whole sophisticated-scheme composition
// runs identically on Metal.
//
// Usage: pentathlon_test [golden_dir] [kernels_dir]

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <memory>
#include <string>
#include <vector>

#include "ptir/bound.hpp"
#include "ptir/container.hpp"
#include "runner.hpp"
#include "tier0_golden.hpp"

#ifndef TIER0_GOLDEN_DIR
#define TIER0_GOLDEN_DIR "."
#endif
#ifndef TIER0_KERNELS_DIR
#define TIER0_KERNELS_DIR "."
#endif

namespace C = pie_cuda_driver::ptir::container;
namespace B = pie_cuda_driver::ptir::bound;
namespace P = pie_cuda_driver::ptir;
namespace G = tier0::golden;

namespace {
int g_pass = 0, g_fail = 0;
void expect(bool ok, const std::string& what) {
    if (ok) { ++g_pass; std::printf("  PASS  %s\n", what.c_str()); }
    else { ++g_fail; std::printf("  FAIL  %s\n", what.c_str()); }
}

bool vals_eq(const tier0::Val& a, const tier0::Val& b) {
    if (a.dt == P::DType::F32 || b.dt == P::DType::F32) {
        auto x = a.to_f32(), y = b.to_f32();
        if (x.size() != y.size()) return false;
        for (std::size_t i = 0; i < x.size(); ++i)
            if (std::fabs(x[i] - y[i]) > 1e-4f) return false;
        return true;
    }
    return a.i == b.i;
}
std::string val_str(const tier0::Val& v) {
    std::string s = "[";
    if (v.dt == P::DType::F32) for (std::size_t k = 0; k < v.f.size(); ++k) s += std::to_string(v.f[k]) + (k + 1 < v.f.size() ? "," : "");
    else for (std::size_t k = 0; k < v.i.size(); ++k) s += std::to_string(v.i[k]) + (k + 1 < v.i.size() ? "," : "");
    return s + "]";
}
const char* stage_str(P::StageKind k) {
    switch (k) {
        case P::StageKind::Prologue: return "Prologue";
        case P::StageKind::OnAttnProj: return "OnAttnProj";
        case P::StageKind::OnAttn: return "OnAttn";
        case P::StageKind::Epilogue: return "Epilogue";
    }
    return "?";
}

// Deterministic Quest kernel host (echo's stub): envelope_dot(q) -> [P_MAX]
// score[p] = |q[0]| + p + calls, calls++ after each call (process-local, shared
// across all phases; call order per pass = layer0 then layer1).
struct QuestHost : tier0::KernelHost {
    std::uint32_t calls = 0;
    tier0::Val call(const std::string& name, const std::vector<tier0::Val>& args,
                    const P::TensorType& rt) override {
        if (name != "envelope_dot") return tier0::Val::make_f32({});
        float q0 = 0.0f;
        if (!args.empty()) { auto q = args[0].to_f32(); if (!q.empty()) q0 = std::fabs(q[0]); }
        std::size_t n = 1;
        for (auto d : rt.shape.dims) n *= d;
        if (n == 0) n = 1;
        float c = (float)calls;
        ++calls;
        std::vector<float> o(n);
        for (std::size_t p = 0; p < n; ++p) o[p] = q0 + (float)p + c;
        return tier0::Val::make_f32(std::move(o));
    }
};

// ── golden-file parse ──────────────────────────────────────────────────────
struct Prog { std::string name, container_hex, sidecar_hex; };

// Parse the `query: [F32([..]), F32([..])]` list into per-layer vectors.
std::vector<std::vector<float>> parse_query(const std::string& line) {
    std::vector<std::vector<float>> out;
    std::size_t q = line.find("query:");
    if (q == std::string::npos) return out;
    std::size_t pos = q;
    while (true) {
        std::size_t f = line.find("F32(", pos);
        if (f == std::string::npos) break;
        // stop if this F32 belongs to a later field (query is the last field here)
        tier0::Val v = G::parse_typed(line.substr(f));
        out.push_back(v.f);
        std::size_t close = line.find(']', f);
        if (close == std::string::npos) break;
        pos = close + 1;
    }
    return out;
}

// Extract "FIELD: Some(TYPE([..]))" typed value from a PassInputs line.
bool parse_field(const std::string& line, const std::string& field, tier0::Val& out) {
    std::size_t p = line.find(field + ": Some(");
    if (p == std::string::npos) return false;
    out = G::parse_typed(line.substr(p));
    return true;
}
}  // namespace

int main(int argc, char** argv) {
    std::string gdir = argc > 1 ? argv[1] : TIER0_GOLDEN_DIR;
    std::string kdir = argc > 2 ? argv[2] : TIER0_KERNELS_DIR;
    ptir_metal::MetalHarness h;
    if (!h.ok()) { std::printf("PENTATHLON_FAIL: %s\n", h.error().c_str()); return 2; }
    tier0::MetalOps ops{h};
    if (!ops.load(kdir)) { std::printf("PENTATHLON_FAIL: %s\n", h.error().c_str()); return 2; }
    std::printf("pentathlon_iter cross-backend cert — device: %s\n\n", h.device_name().c_str());

    std::ifstream f(gdir + "/pentathlon_iter.txt");
    if (!f) { std::printf("PENTATHLON_FAIL: cannot open pentathlon_iter.txt\n"); return 2; }

    // Decode both programs, then replay the scripted phases.
    std::vector<Prog> progs;
    struct Bundle { C::Container ct; B::Bound bd; P::Trace tr; };
    std::vector<Bundle> bundles;  // parallel to progs (kept alive)
    auto& expand_bundle = bundles;  // resolved after decode

    // Phase state
    QuestHost host;
    std::string line, cur_prog;
    bool in_phases = false;

    // Per-phase accumulators
    std::string phase_label;
    int phase_prog = -1;  // 0 expand, 1 rollout
    std::vector<std::pair<std::uint32_t, tier0::Val>> seeds;
    std::vector<std::pair<std::uint32_t, tier0::Val>> hputs;
    tier0::FireInputs fin;
    std::unique_ptr<tier0::Instance> inst;
    tier0::StepReport rep;
    bool stepped = false;
    std::size_t sink_i = 0;

    auto find_prog = [&](const std::string& name) -> int {
        for (std::size_t i = 0; i < progs.size(); ++i)
            if (progs[i].name == name) return (int)i;
        return -1;
    };

    // Lazily decode all programs once we hit the first phase.
    auto decode_all = [&]() {
        bundles.resize(progs.size());
        for (std::size_t i = 0; i < progs.size(); ++i) {
            auto cb = G::hex_to_bytes(progs[i].container_hex);
            auto sb = G::hex_to_bytes(progs[i].sidecar_hex);
            C::DecodeError e;
            if (!C::decode(cb.data(), cb.size(), bundles[i].ct, &e))
                { expect(false, progs[i].name + ": decode " + e.detail); continue; }
            std::string se;
            if (!B::parse_sidecar(sb.data(), sb.size(), bundles[i].bd, &se))
                { expect(false, progs[i].name + ": sidecar " + se); continue; }
            auto tr = B::container_to_trace(bundles[i].ct, bundles[i].bd);
            if (!tr.ok) { expect(false, progs[i].name + ": translate " + tr.error); continue; }
            bundles[i].tr = tr.trace;
            expect(bundles[i].ct.hash == bundles[i].bd.container_hash, progs[i].name + ": hash identity");
        }
    };

    auto start_phase = [&](const std::string& label) {
        phase_label = label;
        phase_prog = label.rfind("expand", 0) == 0 ? 0 : 1;  // "expand …" / "rollout …"
        seeds.clear(); hputs.clear();
        fin = tier0::FireInputs{};
        inst.reset();
        stepped = false;
        sink_i = 0;
    };

    auto ensure_instance = [&]() {
        if (inst) return;
        Bundle& bd = bundles[phase_prog];
        inst = std::make_unique<tier0::Instance>(bd.tr, bd.bd, ops);
        for (auto& s : seeds) inst->seed(s.first, s.second);
        inst->set_num_layers(2);
        for (auto& hp : hputs) inst->host_put(hp.first, hp.second);
    };

    while (std::getline(f, line)) {
        std::string t = G::trim(line);
        if (t.empty()) continue;

        if (t.rfind("== ", 0) == 0) {
            in_phases = true;
            if (bundles.empty()) decode_all();
            start_phase(G::trim(t.substr(3)));
            continue;
        }
        if (!in_phases) {
            auto colon = t.find(':');
            std::string key = colon == std::string::npos ? "" : G::trim(t.substr(0, colon));
            std::string val = colon == std::string::npos ? "" : G::trim(t.substr(colon + 1));
            if (key == "name") progs.push_back(Prog{val, "", ""});
            else if (key == "container" && !progs.empty()) progs.back().container_hex = val;
            else if (key == "sidecar" && !progs.empty()) progs.back().sidecar_hex = val;
            continue;
        }

        // ── phase body ──
        if (t.rfind("seed chan=", 0) == 0) {
            std::uint32_t ch = (std::uint32_t)std::stoul(t.substr(10));
            seeds.push_back({ch, G::parse_typed(t)});
        } else if (t.rfind("host_put chan=", 0) == 0) {
            std::uint32_t ch = (std::uint32_t)std::stoul(t.substr(14));
            hputs.push_back({ch, G::parse_typed(t)});
        } else if (t.rfind("inputs ", 0) == 0) {
            tier0::Val v;
            if (parse_field(t, "logits", v)) { fin.has_logits = true; fin.logits = v.f; }
            if (parse_field(t, "mtp_logits", v)) { fin.has_mtp_logits = true; fin.mtp_logits = v.f; }
            if (parse_field(t, "value_head", v)) fin.value_head = v.f;
            fin.query = parse_query(t);
        } else if (t.rfind("step ", 0) == 0) {
            ensure_instance();
            rep = inst->step(fin, &host);
            stepped = true;
            bool cok = rep.committed == (t.find("committed=true") != std::string::npos);
            std::size_t exp_sinks = 0;
            std::size_t sp = t.find("sinks=");
            if (sp != std::string::npos) exp_sinks = (std::size_t)std::stoul(t.substr(sp + 6));
            expect(rep.ok && cok, phase_label + ": step committed");
            expect(rep.sinks.size() == exp_sinks,
                   phase_label + ": sink count (" + std::to_string(rep.sinks.size()) +
                       " vs " + std::to_string(exp_sinks) + ")");
        } else if (t.rfind("sink ", 0) == 0) {
            // sink NAME: stage=X layer=Y args=[TYPE([..])]
            std::string body = t.substr(5);
            std::string name = G::trim(body.substr(0, body.find(':')));
            std::size_t sp = body.find("stage=");
            std::string stage = body.substr(sp + 6, body.find(' ', sp) - (sp + 6));
            std::size_t lp = body.find("layer=");
            std::uint32_t layer = (std::uint32_t)std::stoul(body.substr(lp + 6));
            // args=[TYPE([..])] — skip the outer '[' so parse_typed sees the
            // inner typed value (not the nested bracket).
            tier0::Val expv = G::parse_typed(body.substr(body.find("args=[") + 6));
            bool ok = sink_i < rep.sinks.size();
            if (ok) {
                const auto& sk = rep.sinks[sink_i];
                std::string got_name = sk.name_idx < bundles[phase_prog].tr.names.size()
                                           ? bundles[phase_prog].tr.names[sk.name_idx] : "";
                ok = got_name == name && stage_str(sk.stage) == stage && sk.layer == layer &&
                     !sk.args.empty() && vals_eq(sk.args[0], expv);
            }
            expect(ok, phase_label + ": sink[" + std::to_string(sink_i) + "] " + name +
                           " L" + std::to_string(layer));
            ++sink_i;
        } else if (t.rfind("take chan=", 0) == 0) {
            std::uint32_t ch = (std::uint32_t)std::stoul(t.substr(10));
            tier0::Val expv = G::parse_typed(t);
            tier0::Val got;
            auto err = inst->host_take(ch, got);
            bool ok = err == tier0::HostErr::Ok && vals_eq(got, expv);
            expect(ok, phase_label + ": take chan=" + std::to_string(ch) + " = " +
                           (err == tier0::HostErr::Ok ? val_str(got) : "ERR"));
        }
        (void)stepped;
        (void)expand_bundle;
    }

    std::printf("\n%d passed, %d failed  (envelope_dot calls=%u)\n", g_pass, g_fail, host.calls);
    if (g_fail == 0) { std::printf("PENTATHLON_OK\n"); return 0; }
    std::printf("PENTATHLON_FAIL\n");
    return 1;
}
