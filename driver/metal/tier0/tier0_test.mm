// tier0_test — PTIR tier-0 interpreter cert on Metal. Decodes echo's golden
// containers (charlie's CUDA-free decoder) into a typed Trace, runs them through
// the host channel-runtime (runner.hpp, ported from interp.rs) with compute ops
// on Metal kernels, and asserts the step/take results match the golden.
//
// Usage: tier0_test [golden_dir] [kernels_dir]

#include <cstdint>
#include <cstdio>
#include <cmath>
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

void run_golden(const std::string& dir, const std::string& name, tier0::MetalOps& ops) {
    std::printf("[%s]\n", name.c_str());
    tier0::golden::Golden g;
    if (!tier0::golden::load(dir + "/" + name + ".txt", g)) { expect(false, name + ": load"); return; }

    auto cb = tier0::golden::hex_to_bytes(g.container_hex);
    auto sb = tier0::golden::hex_to_bytes(g.sidecar_hex);
    C::Container ct; C::DecodeError e;
    if (!C::decode(cb.data(), cb.size(), ct, &e)) { expect(false, name + ": decode " + e.detail); return; }
    B::Bound b; std::string se;
    if (!B::parse_sidecar(sb.data(), sb.size(), b, &se)) { expect(false, name + ": sidecar " + se); return; }
    expect(ct.hash == g.hash && b.container_hash == g.hash, name + ": identity (hash)");
    auto tr = B::container_to_trace(ct, b);
    if (!tr.ok) { expect(false, name + ": translate " + tr.error); return; }

    tier0::Instance inst(tr.trace, b, ops);
    for (auto& s : g.seeds) inst.seed(s.chan, s.val);

    tier0::FireInputs cur;
    int step_no = 0;
    bool all_ok = true;
    for (auto& a : g.actions) {
        switch (a.kind) {
            case tier0::golden::Action::SetInputs:
                cur = tier0::FireInputs{};
                if (a.has_logits) { cur.has_logits = true; cur.logits = a.logits; }
                break;
            case tier0::golden::Action::Step: {
                auto rep = inst.step(cur);
                if (!rep.ok) { expect(false, name + " step " + std::to_string(step_no) + ": " + rep.error); all_ok = false; }
                bool cok = rep.committed == a.exp_committed;
                bool mok = rep.has_miss == a.exp_miss && (!a.exp_miss || rep.miss_chan == a.miss_chan);
                if (!cok || !mok) {
                    std::printf("  (step %d: committed got=%d want=%d, miss got=%d/ch%u want=%d/ch%u)\n",
                                step_no, rep.committed, a.exp_committed, rep.has_miss, rep.miss_chan,
                                a.exp_miss, a.miss_chan);
                    all_ok = false;
                }
                ++step_no;
                break;
            }
            case tier0::golden::Action::Take: {
                tier0::Val got;
                auto err = inst.host_take(a.chan, got);
                if (a.exp_wouldblock) {
                    if (err != tier0::HostErr::WouldBlock) { std::printf("  (take ch%u: expected WouldBlock)\n", a.chan); all_ok = false; }
                } else {
                    if (err != tier0::HostErr::Ok) { std::printf("  (take ch%u: got err, want %s)\n", a.chan, val_str(a.val).c_str()); all_ok = false; }
                    else if (!vals_eq(got, a.val)) {
                        std::printf("  (take ch%u: got %s want %s)\n", a.chan, val_str(got).c_str(), val_str(a.val).c_str());
                        all_ok = false;
                    }
                }
                break;
            }
            case tier0::golden::Action::HostPut:
                inst.host_put(a.chan, a.val);
                break;
        }
    }
    expect(all_ok, name + ": full replay (steps + takes match golden)");
}

}  // namespace

int main(int argc, char** argv) {
    std::string gdir = argc > 1 ? argv[1] : TIER0_GOLDEN_DIR;
    std::string kdir = argc > 2 ? argv[2] : TIER0_KERNELS_DIR;
    ptir_metal::MetalHarness h;
    if (!h.ok()) { std::printf("TIER0_TEST_FAIL: %s\n", h.error().c_str()); return 2; }
    tier0::MetalOps ops{h};
    if (!ops.load(kdir)) { std::printf("TIER0_TEST_FAIL: %s\n", h.error().c_str()); return 2; }
    std::printf("PTIR tier-0 interp cert — device: %s\n\n", h.device_name().c_str());

    run_golden(gdir, "counter_pingpong", ops);
    run_golden(gdir, "greedy_argmax", ops);

    std::printf("\n%d passed, %d failed\n", g_pass, g_fail);
    if (g_fail == 0) { std::printf("TIER0_TEST_OK\n"); return 0; }
    std::printf("TIER0_TEST_FAIL\n");
    return 1;
}
