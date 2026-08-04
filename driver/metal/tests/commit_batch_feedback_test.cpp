// Metal 4 submit-path gate: batched `commit:count:options:` + commit feedback.
//
// Two Metal 4 capabilities the driver used to leave on the table, both of which
// now sit behind the single `Impl::commit_and_signal` primitive that every
// submit path routes through:
//
//   * every commit went out as `commit:&cb count:1`, so a pass that split into
//     independent chunks paid one submit each. `run_steps` encodes N command
//     buffers against one allocator and submits them in a single call.
//   * GPU timing was inferred from host wall-clock around the event wait, which
//     folds in host wake-up latency, and a GPU-side error surfaced only as a
//     wait timeout. `MTL4CommitOptions/addFeedbackHandler:` reports the real
//     GPUStartTime/GPUEndTime and any error.
//
// Runs on-device: it compiles a trivial kernel from source and dispatches it.

#include <cstdio>
#include <cstdint>
#include <cstdlib>
#include <string>
#include <algorithm>
#include <functional>
#include <chrono>
#include <vector>

#include <unistd.h>

#include "mtl4_context.hpp"

using pie::metal::GpuCommitFeedback;
using pie::metal::Grid;
using pie::metal::RawMetalContext;
using pie::metal::SlotHandle;
using pie::metal::StepEncoder;
using pie::metal::StepTiming;
using pie::metal::Threadgroup;

namespace {
int g_pass = 0, g_fail = 0;
bool expect(bool ok, const std::string& what) {
    if (ok) { ++g_pass; std::printf("  PASS  %s\n", what.c_str()); }
    else    { ++g_fail; std::printf("  FAIL  %s\n", what.c_str()); }
    return ok;
}

// Writes `tag` into out[i] for a fixed-size grid, so each command buffer in a
// batch leaves a distinguishable footprint in its own output slot.
constexpr const char* kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void fill(device uint* out [[buffer(0)]],
                 constant uint& tag [[buffer(1)]],
                 uint i [[thread_position_in_grid]]) {
    out[i] = tag + i;
}
)MSL";

int finish(const char* name) {
    std::printf("\n==== %s: %d passed, %d failed ====\n", name, g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
}  // namespace

int main() {
    std::printf("[Metal 4 submit path: batched commit + commit feedback]\n");

    auto ctx = RawMetalContext::create(16u << 20);
    if (!expect(ctx != nullptr, "RawMetalContext::create succeeds")) {
        return finish("commit_batch_feedback_test");
    }

    std::string err;
    auto pso = ctx->compile_pso(kSrc, "fill", &err);
    if (!expect(pso.valid(), "trivial fill kernel compiles under MSL 4.0 (" + err + ")")) {
        return finish("commit_batch_feedback_test");
    }

    constexpr std::uint32_t kElems = 64;
    constexpr int kBatch = 4;
    std::vector<SlotHandle> outs;
    std::vector<SlotHandle> tags;
    for (int i = 0; i < kBatch; ++i) {
        outs.push_back(ctx->create_standalone_buffer(kElems * sizeof(std::uint32_t)));
        tags.push_back(ctx->create_standalone_buffer(sizeof(std::uint32_t)));
        if (outs.back().contents_ptr == nullptr || tags.back().contents_ptr == nullptr) {
            expect(false, "shared buffers allocated");
            return finish("commit_batch_feedback_test");
        }
        *static_cast<std::uint32_t*>(tags.back().contents_ptr) =
            static_cast<std::uint32_t>(1000 * (i + 1));
        std::fill_n(static_cast<std::uint32_t*>(outs.back().contents_ptr), kElems, 0u);
    }
    ctx->make_resident();

    // Each closure owns ordinal `i` so the batch has no cross-buffer binding
    // hazard -- the precondition run_steps documents.
    std::vector<std::function<void(StepEncoder&)>> encoders;
    for (int i = 0; i < kBatch; ++i) {
        ctx->arg_bind_ordinal(i, 0, outs[i]);
        ctx->arg_bind_ordinal(i, 1, tags[i]);
        encoders.push_back([&, i](StepEncoder& se) {
            se.set_pso(pso);
            se.set_argtable_ordinal(i);
            se.dispatch(Grid{kElems, 1, 1}, Threadgroup{32, 1, 1});
        });
    }

    const StepTiming timing = ctx->run_steps(encoders);
    expect(timing.completed, "run_steps completes");
    expect(!timing.timed_out, "run_steps does not time out");
    expect(!timing.gpu_error, "commit feedback reports no GPU error");

    bool all_correct = true;
    for (int i = 0; i < kBatch; ++i) {
        const auto* got = static_cast<const std::uint32_t*>(outs[i].contents_ptr);
        const std::uint32_t tag = static_cast<std::uint32_t>(1000 * (i + 1));
        for (std::uint32_t e = 0; e < kElems; ++e) {
            if (got[e] != tag + e) { all_correct = false; break; }
        }
    }
    expect(all_correct,
           "all " + std::to_string(kBatch) +
               " command buffers in ONE commit:count:options: ran and wrote their own slot");

    // The handler is asynchronous, so poll briefly rather than assuming it has
    // already landed by the time the event fence released us.
    GpuCommitFeedback fb;
    for (int spin = 0; spin < 2000 && !fb.valid(); ++spin) {
        fb = ctx->last_commit_feedback();
        if (!fb.valid()) usleep(1000);
    }
    expect(fb.valid(), "commit feedback handler fired");
    expect(fb.event_value == ctx->last_event(),
           "feedback is tagged with the batch's queue timeline value");
    expect(!fb.had_error, "feedback carries no error (" + fb.error + ")");
    expect(fb.gpu_end_s >= fb.gpu_start_s, "GPUEndTime >= GPUStartTime");
    expect(fb.gpu_ms > 0.0,
           "GPU-measured duration is positive (" + std::to_string(fb.gpu_ms) + " ms)");
    // The GPU cannot have spent longer executing than the host spent waiting on
    // it; this is what makes the feedback number the tighter of the two.
    expect(fb.gpu_ms <= timing.gpu_exec_ms + 1e-6,
           "GPU-measured duration (" + std::to_string(fb.gpu_ms) +
               " ms) <= host-observed (" + std::to_string(timing.gpu_exec_ms) + " ms)");

    // A single-buffer step must still work and still get feedback: run_step is
    // now just run_steps with one encoder.
    std::fill_n(static_cast<std::uint32_t*>(outs[0].contents_ptr), kElems, 0u);
    const StepTiming single = ctx->run_step([&](StepEncoder& se) {
        se.set_pso(pso);
        se.set_argtable_ordinal(0);
        se.dispatch(Grid{kElems, 1, 1}, Threadgroup{32, 1, 1});
    });
    expect(single.completed && !single.gpu_error, "run_step still completes cleanly");
    expect(static_cast<const std::uint32_t*>(outs[0].contents_ptr)[7] == 1000u + 7u,
           "run_step wrote the expected value");

    // ── Abandoning a wait ────────────────────────────────────────────────────
    //
    // A GPU that never signals used to be invisible. `run_steps` retried the
    // five-second wait in a bare `while` loop, so the driver sat inside
    // `waitUntilSignaledValue` indefinitely and printed nothing -- twenty-two
    // minutes of it, on the checkpoint that sent this here. Meanwhile
    // `StepTiming::timed_out` meant "the FIRST probe expired", which a slow but
    // perfectly healthy step also does, and the one caller that acted on it
    // killed the load for being slow. The reporting existed; the loop made it
    // unreachable.
    //
    // Kept last in this file on purpose: it ends the context. Abandoning a wait
    // does not abandon the command buffer, and `[allocator reset]` requires
    // every buffer drawn from it to have completed -- so a context that stopped
    // waiting can never encode again, and says so rather than resetting an
    // allocator out from under live work.
    std::printf("[abandoning a wait]\n");
    ctx->force_next_wait_timeout_for_test();
    const auto before = std::chrono::steady_clock::now();
    const StepTiming abandoned = ctx->run_step([&](StepEncoder& se) {
        se.set_pso(pso);
        se.set_argtable_ordinal(0);
        se.dispatch(Grid{kElems, 1, 1}, Threadgroup{32, 1, 1});
    });
    expect(!abandoned.succeeded(), "a wait the driver gave up on is not a success");
    expect(abandoned.timed_out, "and it says so: timed_out");
    expect(!abandoned.completed,
           "and does NOT claim completed -- the fence was never observed");

    const StepTiming after = ctx->run_step([&](StepEncoder& se) {
        se.set_pso(pso);
        se.set_argtable_ordinal(0);
        se.dispatch(Grid{kElems, 1, 1}, Threadgroup{32, 1, 1});
    });
    expect(!after.succeeded() && after.timed_out,
           "every later step on that context fails too, rather than resetting an "
           "allocator whose command buffers may still be running");
    const double elapsed_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - before)
            .count();
    // Both calls above are the give-up path. If the second one had gone back to
    // the queue it would have spent the whole budget, which is a minute.
    expect(elapsed_ms < 1000.0,
           "and fails immediately rather than spending the budget again (" +
               std::to_string(int(elapsed_ms)) + " ms)");

    return finish("commit_batch_feedback_test");
}
