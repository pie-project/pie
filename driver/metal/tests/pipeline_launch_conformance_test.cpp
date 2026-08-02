// Host-runnable conformance gate for the Metal launch-package interpreter.
//
// `interp.hpp` is live code: `pie::metal::pipeline::step()` is Metal's M0
// execution path, reached from `context.cpp`. After the driver stopped
// decoding PTIR (`ptir-refactor.md` phase 3'), the three tests that used to
// cover it went with the fixture format they were written against -- they fed
// hand-transcribed container + PTIB sidecar bytes to a decoder that no longer
// exists. That left the interpreter with no host-runnable coverage at all, and
// a code review promptly found five bugs in it. This is the replacement, and
// it is deliberately not a port: it drives the *shipped* launch package, which
// is what the driver actually receives.
//
// Pure host. No Metal, no Apple SDK, no GPU -- it links nothing and always
// runs, which is the point.
//
// The fixtures are the same relocatable `PieLaunch*` images the CUDA driver
// tests register (`driver/fixtures/<name>.launch`, written by
// `cargo test -p pie-compiler-tests --test cuda_golden emit_driver_test_kernel`).
// The image is an ABI record dump, so it is backend-neutral: only the
// `.kernels` sibling is CUDA source, and nothing here reads it.
//
// Three halves, each pinning a property that was actually violated:
//
//  1. *Readiness is one direction per channel.* A channel that is taken and
//     then put back -- the linear `take -> put` of a counter, a beam cursor, a
//     DFA state -- is in both effect sets, and a gate derived from the union
//     demands it simultaneously full and empty. No ring satisfies that, so the
//     stage retries forever. The package ships the first-touch direction
//     instead; this asserts the shipped value is single-valued and that such a
//     channel actually fires.
//  2. *Every shipped take is scheduled.* Stage indexes are built by
//     reachability from op results, op args, and puts, so a take whose result
//     is dead is invisible to that walk -- and skipping it is not a no-op,
//     because readiness still requires the channel non-empty while the ring
//     never advances.
//  3. *Names survive adoption.* `Op::name_index` addresses `Trace::names`; if
//     adoption drops it, every intrinsic-boundary check silently reads slot 0.

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <map>
#include <string>
#include <vector>

#include "pie/driver/testing/image.hpp"
#include "pipeline/interp.hpp"

using namespace pie::metal::pipeline;
namespace launch = pie::driver::launch;

namespace {

int g_pass = 0;
int g_fail = 0;

void expect(bool ok, const std::string& what) {
    if (ok) {
        ++g_pass;
        std::printf("  PASS  %s\n", what.c_str());
    } else {
        ++g_fail;
        std::printf("  FAIL  %s\n", what.c_str());
    }
}

// Every program the driver corpus can bind. Named rather than globbed so a
// fixture that stops being emitted fails here instead of shrinking the sweep
// to nothing.
const char* const kPrograms[] = {
    "beam_epilogue",   "counter_pingpong",
    "dfa_ingraph",     "greedy_argmax",
    "mtp_verify_tail", "nucleus_sample",
    "pivot_predicates_multistage", "section3_masked_gumbel",
    "staged_dispatch", "structured_masks",
};

struct Loaded {
    pie::driver::testing::PackageImage image;
    ExecPlan plan;
};

bool load(const std::string& dir, const std::string& name, Loaded& out) {
    std::string error;
    if (!out.image.load(dir + "/" + name + ".launch", &error)) {
        expect(false, name + ": load launch image (" + error + ")");
        return false;
    }
    if (!adopt_launch_package(out.image.package(), out.plan, &error)) {
        expect(false, name + ": adopt launch package (" + error + ")");
        return false;
    }
    return true;
}

// (1) The shipped direction is single-valued, and it agrees with the effect
// sets wherever the sets are unambiguous. Where they disagree -- a channel in
// both `takes` and `puts` -- the shipped value is the only answer there is,
// and asserting it is *not* `Untouched` is the whole gate: `Untouched` is what
// a channel gets when nothing touched it, and an in-place channel is touched
// twice.
void check_readiness_is_one_direction(const std::string& name, const ExecPlan& plan) {
    std::size_t in_place = 0;
    bool consistent = true;
    bool in_place_gated = true;
    for (std::uint32_t c = 0; c < plan.trace.channels.size(); ++c) {
        const launch::Readiness readiness = plan.trace.channels[c].readiness;
        const bool taken = plan.takes_channel(c) || plan.reads_channel(c);
        const bool put = plan.puts_channel(c);
        if (taken && put) {
            ++in_place;
            if (readiness == launch::Readiness::Untouched) in_place_gated = false;
            continue;
        }
        if (taken && readiness != launch::Readiness::NeedsFull) consistent = false;
        if (!taken && put && readiness != launch::Readiness::NeedsEmpty) consistent = false;
        if (!taken && !put && readiness != launch::Readiness::Untouched) consistent = false;
    }
    expect(consistent, name + ": shipped readiness agrees with the unambiguous channels");
    if (in_place > 0) {
        expect(in_place_gated,
               name + ": in-place channels carry a direction (" +
                   std::to_string(in_place) + " of them)");
    }
}

// (1b) The behavioural half, and the one that would have caught the bug. A
// union-derived gate makes an in-place channel permanently unready, so a fire
// that should commit instead returns `committed == false` naming the channel
// it was about to take from. Two passes, because a take that leaves the ring
// where it found it passes once by luck and stalls on the second.
//
// Note what the corpus says about the direction: `staged_dispatch` has a
// channel that is both taken and put and ships `NeedsEmpty`, because its first
// touch in pass order is the put. So the answer is not "take wins" -- it is the
// order, which the effect *sets* have thrown away by the time the driver sees
// them.
void check_in_place_channel_fires(const std::string& name, const ExecPlan& plan) {
    bool has_in_place = false;
    for (std::uint32_t c = 0; c < plan.trace.channels.size(); ++c) {
        if ((plan.takes_channel(c) || plan.reads_channel(c)) && plan.puts_channel(c)) {
            has_in_place = true;
            break;
        }
    }
    if (!has_in_place) return;
    // Non-executable is not a failure: Metal rejects per-layer taps by design,
    // and the reject reason is checked separately.
    if (!plan.executable || plan.needs_forward()) return;

    // The package declares which channels are seeded but not with what: the
    // seed is a compile-time constant the registry materialises. Zeros are
    // enough here -- this asks whether the ring advances, not what it holds.
    std::map<std::uint32_t, Value> seeds;
    for (std::uint32_t c = 0; c < plan.trace.channels.size(); ++c) {
        const launch::Channel& decl = plan.trace.channels[c];
        if (!decl.has_seed) continue;
        seeds.emplace(c, zeros(decl.type.dtype,
                               std::max<std::uint64_t>(decl.type.shape.numel(), 1)));
    }
    InterpInstance inst = make_instance(plan, {}, seeds);

    bool every_pass_committed = true;
    std::string detail;
    for (int pass = 0; pass < 3; ++pass) {
        const StepResult result = step(inst, plan);
        if (!result.ok) {
            expect(false, name + ": in-place fire faults on pass " +
                              std::to_string(pass) + " (" + result.error + ")");
            return;
        }
        if (!result.committed) {
            every_pass_committed = false;
            detail = " (pass " + std::to_string(pass) + " missed channel " +
                std::to_string(result.missed_channel) + ")";
            break;
        }
        // A host-readable output channel is capacity-bounded on purpose: leave
        // it full and the next pass blocks on backpressure, which is correct
        // and has nothing to do with the in-place channels under test. Drain it
        // the way a real host would.
        for (std::uint32_t c = 0; c < plan.trace.channels.size(); ++c) {
            if (!plan.trace.channels[c].host_reader) continue;
            Value harvested;
            (void)host_take(inst, plan, c, harvested);
        }
    }
    expect(every_pass_committed,
           name + ": the in-place channel fires on every pass, not just the first" + detail);
}

// (2) Every channel the stage says it takes has its take value scheduled.
// Reachability alone misses a take whose result nothing reads.
void check_every_take_is_scheduled(const std::string& name, const ExecPlan& plan) {
    bool all_scheduled = true;
    for (std::size_t si = 0; si < plan.trace.stages.size(); ++si) {
        const launch::Stage& stage = plan.trace.stages[si];
        const std::vector<std::uint32_t>& ids = plan.stages[si].value_ids;
        for (std::size_t v = 0; v < plan.trace.values.size(); ++v) {
            const launch::Value& value = plan.trace.values[v];
            const bool taken = value.source == launch::ValueSource::ChannelTake &&
                std::find(stage.takes.begin(), stage.takes.end(), value.channel) !=
                    stage.takes.end();
            const bool peeked = value.source == launch::ValueSource::ChannelRead &&
                std::find(stage.reads.begin(), stage.reads.end(), value.channel) !=
                    stage.reads.end();
            if (!taken && !peeked) continue;
            if (std::find(ids.begin(), ids.end(), static_cast<std::uint32_t>(v)) == ids.end()) {
                all_scheduled = false;
            }
        }
    }
    expect(all_scheduled, name + ": every shipped take/read root is scheduled");
}

// (3) `name_index` addresses `Trace::names`, so it has to survive adoption and
// stay in range. A dropped index is not a crash -- it reads slot 0 and the
// boundary check quietly compares the wrong name.
void check_name_indexes_are_in_range(const std::string& name, const ExecPlan& plan) {
    bool in_range = true;
    bool any_named = false;
    for (const launch::Stage& stage : plan.trace.stages) {
        for (const launch::Op& op : stage.ops) {
            if (op.name_index == 0) continue;
            any_named = true;
            if (op.name_index >= plan.trace.names.size()) in_range = false;
        }
    }
    expect(in_range, name + ": every op name index addresses a real name");
    if (any_named) {
        expect(!plan.trace.names.empty(), name + ": named ops come with a name table");
    }
}

// Adoption is the driver's only validation of a package it did not build, so
// a package that survives it must be internally consistent: every op argument
// and every channel op slot has to address something real. This walks what
// adoption produced rather than hand-building a corrupt package, because the
// C ABI slices borrow and a hand-built one would be testing the test.
void check_structural_bounds(const std::string& name, const ExecPlan& plan) {
    bool args_ok = true;
    bool chans_ok = true;
    for (const launch::Stage& stage : plan.trace.stages) {
        for (const launch::Op& op : stage.ops) {
            for (std::uint32_t arg : op.args) {
                if (arg >= plan.trace.values.size()) args_ok = false;
            }
        }
    }
    // The region plan is the other op stream -- the one the generated kernels
    // are keyed to -- and its `chan` is signed, with -1 for "no channel". An
    // unguarded subscript turns that into SIZE_MAX, so registration validates
    // the slot; this pins that every plan the corpus produces would survive it.
    for (const launch::plan::StagePlan& region_plan : plan.region_plans) {
        for (const launch::plan::NormalizedOp& normalized : region_plan.ops) {
            const std::uint8_t tag = normalized.op.tag;
            const bool is_channel_op = tag == PTIR_OP_CHAN_TAKE ||
                tag == PTIR_OP_CHAN_READ || tag == PTIR_OP_CHAN_PUT;
            if (!is_channel_op) continue;
            if (normalized.op.chan < 0 ||
                static_cast<std::size_t>(normalized.op.chan) >=
                    region_plan.channel_bindings.size()) {
                chans_ok = false;
            }
        }
    }
    expect(args_ok, name + ": every op argument addresses a real value");
    expect(chans_ok, name + ": every channel op names a bound channel");
}

// A program Metal cannot run has to say why. Silent rejection and rejection
// with an empty reason look identical to a caller, and the difference is
// whether the next person can tell an unsupported program from a broken one.
void check_rejection_is_explained(const std::string& name, const ExecPlan& plan) {
    if (plan.executable) {
        expect(plan.reject_reason.empty(), name + ": an executable plan carries no reject reason");
        return;
    }
    expect(!plan.reject_reason.empty(), name + ": a rejected plan says why");
}

}  // namespace

int main(int argc, char** argv) {
    const std::string dir = argc > 1 ? argv[1] : "../fixtures";
    std::printf("metal launch-package interpreter conformance (%s)\n", dir.c_str());

    std::size_t loaded = 0;
    for (const char* name : kPrograms) {
        Loaded fixture;
        if (!load(dir, name, fixture)) continue;
        ++loaded;
        check_readiness_is_one_direction(name, fixture.plan);
        check_in_place_channel_fires(name, fixture.plan);
        check_every_take_is_scheduled(name, fixture.plan);
        check_name_indexes_are_in_range(name, fixture.plan);
        check_rejection_is_explained(name, fixture.plan);
        check_structural_bounds(name, fixture.plan);
    }
    // Zero fixtures is a green run that proves nothing -- the failure mode this
    // whole file exists to prevent.
    expect(loaded == sizeof(kPrograms) / sizeof(kPrograms[0]),
           "every corpus program loaded (" + std::to_string(loaded) + ")");
    if (loaded == 0) {
        // `driver/fixtures/` is gitignored, so the images are absent on a fresh
        // checkout. Say what writes them rather than leaving a wall of
        // "cannot open" as the only clue.
        std::fprintf(stderr,
                     "no launch images in %s -- emit them with:\n"
                     "  cargo test -p pie-compiler-tests --test cuda_golden "
                     "emit_driver_test_kernel\n",
                     dir.c_str());
    }

    std::printf("%d passed, %d failed\n", g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
