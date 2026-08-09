// The AttentionWorkspace oracle — gate-attn-ws.
//
// Compiles the REAL `attention_workspace.cpp` and drives it through eight
// scripts: allocation shapes, the plan-slot rotation at depths 1/2/3, a
// move-assignment over a pending upload, and four failure landings (the pin
// and the event create, each at boot and at the lazy mid-rotation pin).
//
// Two implementations are replaced. `DeviceTensor::allocate` (the shared
// `tensor_recorder.cpp`) because its real body ends in `cudaMalloc`; and the
// six CUDA entry points (stub/cuda_recorder.cpp) because the class's whole
// observable behaviour is WHICH of them it calls WHEN — the transcript is
// the call sequence, with pins and events named by creation ordinal.
//
// What the golden actually pins, in one sentence each:
//   * slot 0 pins EAGERLY at allocate; other slots pin lazily on first
//     rotation, so a non-rotating workspace holds one slot of pinned memory;
//   * `begin_plan_update` rotates FIRST and syncs a pending upload BEFORE
//     handing the slot out — the fence that makes reuse safe;
//   * the rotation advances even when the lazy pin THROWS, and the machine
//     keeps working afterwards (scripts g/h);
//   * teardown syncs every pending upload before destroying its event and
//     freeing its pin, slot by slot in order;
//   * `allocate`'s catch block frees exactly what was created before the
//     failure — the pin without its event (script f), or nothing (script e).

#include <cstdio>
#include <stdexcept>
#include <string>
#include <map>

#include "attention_workspace.hpp"
#include "cuda_runtime.h"

using pie_cuda_driver::AttentionWorkspace;

// tensor_recorder.cpp's log of DeviceTensor::allocate calls.
namespace pie_cuda_driver {
void reset_alloc_log();
const std::vector<std::string>& alloc_log();
}  // namespace pie_cuda_driver

namespace {

constexpr char SEP = '\x1f';

std::string g_script;
std::size_t g_log_drained = 0;
std::size_t g_alloc_drained = 0;
std::map<const void*, std::string> g_dev;

/// Emit every recorder row that arrived since the last flush, prefixed with
/// the current script's name.
void flush() {
    const auto& rows = oracle_cuda::log();
    for (std::size_t i = g_log_drained; i < rows.size(); ++i) {
        std::printf("%s%c%s\n", g_script.c_str(), SEP, rows[i].c_str());
    }
    g_log_drained = rows.size();
    const auto& allocs = pie_cuda_driver::alloc_log();
    for (std::size_t i = g_alloc_drained; i < allocs.size(); ++i) {
        std::printf("%s%cdev%c%s\n", g_script.c_str(), SEP, SEP,
                    allocs[i].c_str());
    }
    g_alloc_drained = allocs.size();
}

void begin_script(const std::string& name) {
    oracle_cuda::reset_case();
    pie_cuda_driver::reset_alloc_log();
    g_log_drained = 0;
    g_alloc_drained = 0;
    g_dev.clear();
    g_script = name;
    oracle_cuda::note("case-begin");
    flush();
}

void call(const char* what) {
    oracle_cuda::note(std::string("call\x1f") + what);
    flush();
}

std::string dev_name(const void* p) {
    if (p == nullptr) return "null";
    auto it = g_dev.find(p);
    return it == g_dev.end() ? "unknown" : it->second;
}

/// Register the two device buffers under symbolic names, in the order
/// `allocate` created them.
void name_buffers(AttentionWorkspace& ws) {
    g_dev[ws.float_buffer()] = "dev#float";
    g_dev[ws.int_buffer()] = "dev#int";
}

void view_row(AttentionWorkspace& ws) {
    const auto v = ws.view();
    oracle_cuda::note(
        "view\x1f" + dev_name(v.float_buffer) + "\x1f" +
        std::to_string(v.float_bytes) + "\x1f" + dev_name(v.int_buffer) +
        "\x1f" + std::to_string(v.int_bytes) + "\x1f" +
        oracle_cuda::pin_name(v.page_locked_int));
    flush();
}

}  // namespace

int main() {
    cudaStream_t sA = reinterpret_cast<cudaStream_t>(0xA0);
    cudaStream_t sB = reinterpret_cast<cudaStream_t>(0xB0);

    // a. Boot shape: slot 0 pinned eagerly, one event, one view, teardown
    //    with nothing pending.
    begin_script("a-alloc");
    {
        oracle_cuda::name_stream(sA, "sA");
        call("allocate(1024,512,1)");
        auto ws = AttentionWorkspace::allocate(1024, 512, 1);
        name_buffers(ws);
        view_row(ws);
        call("drop");
    }
    flush();

    // b. Zero slots clamps to one; a single slot fences ITSELF on reuse.
    begin_script("b-one-slot");
    {
        oracle_cuda::name_stream(sA, "sA");
        oracle_cuda::name_stream(sB, "sB");
        call("allocate(2048,256,0)");
        auto ws = AttentionWorkspace::allocate(2048, 256, 0);
        name_buffers(ws);
        call("begin");
        ws.begin_plan_update();
        view_row(ws);
        call("end(sA)");
        ws.end_plan_update(sA);
        call("begin");
        ws.begin_plan_update();
        view_row(ws);
        call("end(sB)");
        ws.end_plan_update(sB);
        call("drop");
    }
    flush();

    // c. Depth 3: lazy pins on first rotation, the fence on wraparound, and
    //    a teardown with two uploads still pending.
    begin_script("c-rotate-3");
    {
        oracle_cuda::name_stream(sA, "sA");
        oracle_cuda::name_stream(sB, "sB");
        call("allocate(64,32,3)");
        auto ws = AttentionWorkspace::allocate(64, 32, 3);
        name_buffers(ws);
        call("begin");
        ws.begin_plan_update();
        view_row(ws);
        call("end(sA)");
        ws.end_plan_update(sA);
        call("begin");
        ws.begin_plan_update();
        view_row(ws);
        call("end(sB)");
        ws.end_plan_update(sB);
        call("begin");
        ws.begin_plan_update();
        view_row(ws);
        call("end(sA)");
        ws.end_plan_update(sA);
        call("begin");
        ws.begin_plan_update();
        view_row(ws);
        call("drop");
    }
    flush();

    // d. Move-assignment onto a workspace of its own: the target syncs its
    //    pending upload, tears down its slots, then adopts.
    begin_script("d-move-assign");
    {
        oracle_cuda::name_stream(sA, "sA");
        call("allocate-a(16,16,2)");
        auto a = AttentionWorkspace::allocate(16, 16, 2);
        name_buffers(a);
        call("a.begin");
        a.begin_plan_update();
        call("a.end(sA)");
        a.end_plan_update(sA);
        call("allocate-b(16,16,1)");
        auto b = AttentionWorkspace::allocate(16, 16, 1);
        call("b.begin");
        b.begin_plan_update();
        call("b.end(sA)");
        b.end_plan_update(sA);
        call("b=move(a)");
        b = std::move(a);
        call("b.begin");
        b.begin_plan_update();
        view_row(b);
        call("drop");
    }
    flush();

    // e. The pin fails during allocate: nothing was created, nothing to
    //    free, the throw propagates.
    begin_script("e-alloc-pin-fail");
    {
        call("allocate(8,8,1) [pin will fail]");
        oracle_cuda::fail_next_malloc_host();
        try {
            auto ws = AttentionWorkspace::allocate(8, 8, 1);
            oracle_cuda::note("no-throw");
        } catch (const std::exception&) {
            oracle_cuda::note("threw");
        }
    }
    flush();

    // f. The event create fails during allocate: the pin it followed must
    //    be freed by the catch block.
    begin_script("f-alloc-event-fail");
    {
        call("allocate(8,8,1) [event will fail]");
        oracle_cuda::fail_next_event_create();
        try {
            auto ws = AttentionWorkspace::allocate(8, 8, 1);
            oracle_cuda::note("no-throw");
        } catch (const std::exception&) {
            oracle_cuda::note("threw");
        }
    }
    flush();

    // g. The LAZY pin fails mid-rotation: begin throws, the rotation has
    //    already advanced, and the machine keeps working on the next begin.
    begin_script("g-lazy-pin-fail");
    {
        oracle_cuda::name_stream(sA, "sA");
        oracle_cuda::name_stream(sB, "sB");
        call("allocate(8,8,2)");
        auto ws = AttentionWorkspace::allocate(8, 8, 2);
        name_buffers(ws);
        call("begin");
        ws.begin_plan_update();
        call("end(sA)");
        ws.end_plan_update(sA);
        call("begin [pin will fail]");
        oracle_cuda::fail_next_malloc_host();
        try {
            ws.begin_plan_update();
            oracle_cuda::note("no-throw");
        } catch (const std::exception&) {
            oracle_cuda::note("threw");
        }
        call("begin");
        ws.begin_plan_update();
        view_row(ws);
        call("end(sB)");
        ws.end_plan_update(sB);
        call("drop");
    }
    flush();

    // h. The lazy EVENT create fails after its pin succeeded: the slot is
    //    left half-built, and teardown frees the orphan pin without an
    //    event to destroy.
    begin_script("h-lazy-event-fail");
    {
        oracle_cuda::name_stream(sA, "sA");
        call("allocate(8,8,2)");
        auto ws = AttentionWorkspace::allocate(8, 8, 2);
        name_buffers(ws);
        call("begin");
        ws.begin_plan_update();
        call("end(sA)");
        ws.end_plan_update(sA);
        call("begin [event will fail]");
        oracle_cuda::fail_next_event_create();
        try {
            ws.begin_plan_update();
            oracle_cuda::note("no-throw");
        } catch (const std::exception&) {
            oracle_cuda::note("threw");
        }
        call("begin");
        ws.begin_plan_update();
        view_row(ws);
        call("drop");
    }
    flush();

    return 0;
}
