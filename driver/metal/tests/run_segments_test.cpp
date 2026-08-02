// `run_segments`: GPU segments in order, with the host running between them.
//
// The driver's two existing runners both commit everything before they wait for
// anything. That is right when the host has nothing to contribute, and it makes
// one thing impossible: a step whose later half depends on a decision only the
// host can make from the earlier half's output. Paging routed experts is that
// step -- the router picks experts on the GPU, and which weights are resident
// is a fact the host owns -- so the ordering and the callback are the feature.
//
// Proving that needs a chain, not a pair. Each segment doubles what it is given;
// the host reads the result and feeds back one more than it. Starting at 1:
//
//     segment 0: 1*2 = 2     host: in = 3
//     segment 1: 3*2 = 6     host: in = 7
//     segment 2: 7*2 = 14
//
// Only 2, 6, 14 satisfies both claims at once. If the segments were committed
// together the way `run_steps` commits them, every one would read the initial 1
// and the outputs would be 2, 2, 2 -- so this distinguishes the new runner from
// the old one rather than merely exercising it. If the callback ran but the
// segments raced, the later outputs would vary between runs instead of landing
// on one wrong answer.
//
// Runs on-device: it compiles a trivial kernel from source and dispatches it.

#include <cstdint>
#include <cstdio>
#include <functional>
#include <string>
#include <vector>

#include "mtl4_context.hpp"

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

constexpr const char* kSrc = R"MSL(
#include <metal_stdlib>
using namespace metal;
kernel void twice(device uint* out [[buffer(0)]],
                  const device uint* in [[buffer(1)]],
                  uint i [[thread_position_in_grid]]) {
    out[i] = in[0] * 2u;
}
)MSL";

int finish(const char* name) {
    std::printf("\n==== %s: %d passed, %d failed ====\n", name, g_pass, g_fail);
    return g_fail == 0 ? 0 : 1;
}
}  // namespace

int main() {
    std::printf("[run_segments: ordered GPU segments with a host callback between]\n");

    auto ctx = RawMetalContext::create(16u << 20);
    if (!expect(ctx != nullptr, "RawMetalContext::create succeeds")) {
        return finish("run_segments_test");
    }

    std::string err;
    auto pso = ctx->compile_pso(kSrc, "twice", &err);
    if (!expect(pso.valid(), "trivial doubling kernel compiles (" + err + ")")) {
        return finish("run_segments_test");
    }

    constexpr int kSegments = 3;
    SlotHandle in = ctx->create_standalone_buffer(sizeof(std::uint32_t));
    std::vector<SlotHandle> outs;
    for (int i = 0; i < kSegments; ++i) {
        outs.push_back(ctx->create_standalone_buffer(sizeof(std::uint32_t)));
        if (outs.back().contents_ptr == nullptr) {
            expect(false, "shared buffers allocated");
            return finish("run_segments_test");
        }
        *static_cast<std::uint32_t*>(outs.back().contents_ptr) = 0u;
    }
    if (in.contents_ptr == nullptr) {
        expect(false, "shared buffers allocated");
        return finish("run_segments_test");
    }
    *static_cast<std::uint32_t*>(in.contents_ptr) = 1u;
    ctx->make_resident();

    std::vector<std::function<void(StepEncoder&)>> segments;
    for (int i = 0; i < kSegments; ++i) {
        ctx->arg_bind_ordinal(i, 0, outs[i]);
        ctx->arg_bind_ordinal(i, 1, in);
        segments.push_back([&, i](StepEncoder& se) {
            se.set_pso(pso);
            se.set_argtable_ordinal(i);
            se.dispatch(Grid{1, 1, 1}, Threadgroup{1, 1, 1});
        });
    }

    int callbacks = 0;
    std::vector<std::uint32_t> seen_by_host;
    const StepTiming timing = ctx->run_segments(segments, [&](std::size_t i) {
        ++callbacks;
        const std::uint32_t produced =
            *static_cast<const std::uint32_t*>(outs[i].contents_ptr);
        seen_by_host.push_back(produced);
        *static_cast<std::uint32_t*>(in.contents_ptr) = produced + 1u;
    });

    expect(timing.completed, "run_segments completes");
    expect(!timing.timed_out, "run_segments does not time out");

    // After every segment, not between them only: a caller that pages weights
    // needs somewhere to give back the pins the last segment took.
    expect(callbacks == kSegments,
           "the callback runs once per segment, including after the last (" +
               std::to_string(callbacks) + ")");

    const std::vector<std::uint32_t> want{2u, 6u, 14u};
    std::vector<std::uint32_t> got;
    for (int i = 0; i < kSegments; ++i) {
        got.push_back(*static_cast<const std::uint32_t*>(outs[i].contents_ptr));
    }
    const auto render = [](const std::vector<std::uint32_t>& v) {
        std::string s;
        for (std::size_t i = 0; i < v.size(); ++i) {
            s += (i != 0 ? " " : "") + std::to_string(v[i]);
        }
        return s;
    };
    expect(got == want,
           "each segment read what the host wrote after the previous one: got " +
               render(got) + ", want " + render(want));

    // The same chain read from the other end. The host saw each segment's
    // output, which it could not have if the callback ran before the GPU
    // finished -- the buffers start at zero.
    expect(seen_by_host == want,
           "the host observed every segment's result before encoding the next: " +
               render(seen_by_host));

    return finish("run_segments_test");
}
