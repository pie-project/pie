#pragma once

// X2 — CUDA frames/mirrors carrier (Runtime–Driver Boundary B5/B6/B8/B9/B11/B13).
//
// The real-device dual of X1's `MockControlPlane` frame regions. X1 backed the
// B5 address triple (`FrameAddresses{frame_base,mirror_base,word_base}`) with
// plain host allocations; X2 backs it with the real thing:
//   * frame_base   — a DEVICE frame (`cudaMalloc`). Cells live at
//                    `frame_base + channel offset + ring index`. FIXED for the
//                    instance's lifetime (B6) — a dedicated allocation, never
//                    slab-recycled, so wakers + direct reads have a stable base.
//   * mirror_base  — a PINNED host mirror the host reads committed cells from
//                    (B8/B13 — reads are pure loads from here, never through the
//                    driver).
//   * word_base    — PINNED host ring-index words the host waits on (B9 — the
//                    driver advances the word; a waiter resolves when it passes).
//
// The CARRIER is the direct driver<->inferlet frame transport: on a batch commit
// it D2H-mirrors the instance's committed frame cells into the pinned mirror on a
// dedicated non-blocking copy stream, publishes the pinned ring-index word
// (stream-ordered, B11 publish-before-wake), then runs a completion host callback
// — the X0 wake bridge that resolves the parked host future. The value path never
// travels through the driver; the boundary is addresses plus wakes (C5).
//
// This mirrors the `tensor_io.{hpp,cpp}` substrate (dedicated copy stream +
// async copies + `cudaLaunchHostFunc` completion + an `extern "C" pie_*` surface
// the runtime declares directly), and is built + device-verified in isolation
// (the `test_frame_carrier_device` target) before it composes into the executor.
//
// PROVISIONAL (velocity shift): the exact per-channel frame layout is guru's to
// reconcile as he steers the boundary rework. This module builds the *mechanism*
// against X1's `FrameAddresses` triple; the layout is a trace-derived
// {frame,mirror,word} sizing stand-in (identical in shape to X1's `MockProgram`),
// swapped for the real channel-list layout at the reconcile WITHOUT touching the
// carrier. The runtime-side dual is `CudaControlPlane` (control_cuda.rs).

#include <cstddef>
#include <cstdint>
#include <mutex>
#include <vector>

#include <cuda_runtime.h>

namespace pie_cuda_driver::sampling_ir {

// Provisional trace-derived frame layout (B4). Real drivers derive these from the
// trace's channel list + ring sizes; here the sizes come from the trace length so
// bind has concrete regions to allocate — identical in shape to X1's `MockProgram`.
struct FrameLayout {
    std::size_t frame_bytes = 0;   // device frame  (frame_base + chan offset + ring idx)
    std::size_t mirror_bytes = 0;  // pinned mirror  (host reads committed cells here)
    std::size_t word_bytes = 0;    // pinned ring-index words (host waits on these)
};

// One bound instance's frame regions. Addresses are FIXED for the instance's
// lifetime (B6): each is a dedicated backing allocation (never recycled through a
// slab free-list), released only at `close_instance`.
struct FrameInstance {
    std::uint64_t program = 0;
    void*          device_frame = nullptr;  // frame_base  (device)
    void*          host_mirror = nullptr;   // mirror_base (pinned host)
    std::uint64_t* host_words = nullptr;    // word_base   (pinned host ring words)
    std::size_t    frame_bytes = 0;
    std::size_t    mirror_bytes = 0;
    std::size_t    word_bytes = 0;
};

// The completion callback the carrier runs once the D2H mirror + word publish has
// landed (via `cudaLaunchHostFunc`). Runs on a driver-internal thread and MUST NOT
// call CUDA APIs. The X0 wake bridge lives behind this on the runtime side.
using FrameCarryDone = void (*)(void* user_data);

// Owns the dedicated non-blocking copy stream + the instance table. One per
// process for the MVP (the `TensorIoEngine` pattern); made executor-scoped at the
// reconcile if multiple executors ever coexist.
class FrameCarrierEngine {
public:
    static FrameCarrierEngine& instance();

    // B4 — register a trace: compute its (provisional) frame layout and return a
    // stable 1-based program handle. 0 is never a valid handle.
    std::uint64_t register_program(const std::uint8_t* trace, std::size_t trace_len);

    // B4/B5 — bind an instance: allocate its device frame + pinned mirror + pinned
    // words and write the three bases out. Returns the 1-based instance id, or 0 if
    // `program` is unknown (out params left untouched).
    std::uint64_t bind_instance(std::uint64_t program,
                                std::uint64_t* out_frame_base,
                                std::uint64_t* out_mirror_base,
                                std::uint64_t* out_word_base);

    // B6 — release an instance's frame/mirror/word regions. Fail-loud on an unknown
    // or already-closed instance (the tensor_io house style: a loud abort, never a
    // silent trap). The §5.2 grace-period discipline is the caller's.
    void close_instance(std::uint64_t instance);

    // WRITE leg (host inferlet -> device frame): async H2D `n_bytes` of `host_src`
    // into the instance's device frame at `frame_offset`, on the copy stream. The
    // input direction of the transport (bounds-checked <= frame_bytes).
    void carry_in(std::uint64_t instance, const void* host_src, std::size_t n_bytes,
                  std::size_t frame_offset);

    // The CARRIER (driver -> inferlet), the one per-fire commit call: D2H
    // `n_bytes` of the instance's committed frame cells (at `frame_offset`) into
    // its pinned mirror (at `mirror_offset`) on the copy stream, then — stream
    // ordered AFTER the mirror lands — publish `host_words[word_index] = target`
    // (B11 publish-before-wake) and run `done(user_data)` (the X0 wake). `n_bytes
    // == 0` mirrors the whole committed frame (min of the frame/mirror capacities
    // past the offsets) — the layout-agnostic provisional default. `forward_evt`
    // (a `cudaEvent_t` as an opaque handle, or null) is the executor's forward-done
    // event: when non-null the copy stream waits on it before the D2H so the mirror
    // never captures pre-commit cells (cross-stream ordering; the tensor_io pattern).
    void carry(std::uint64_t instance,
               std::size_t frame_offset, std::size_t mirror_offset, std::size_t n_bytes,
               std::size_t word_index, std::uint64_t target,
               void* forward_evt,
               FrameCarryDone done, void* user_data);

    cudaStream_t copy_stream() const { return stream_; }

    // Introspection for the isolation test (asserts bind allocated distinct,
    // non-null bases and close reclaimed them).
    std::size_t live_instances() const;

    FrameCarrierEngine(const FrameCarrierEngine&) = delete;
    FrameCarrierEngine& operator=(const FrameCarrierEngine&) = delete;

private:
    FrameCarrierEngine();
    ~FrameCarrierEngine();

    FrameInstance* lookup(std::uint64_t instance);  // caller holds mu_; nullptr if unknown

    cudaStream_t stream_ = nullptr;  // the dedicated non-blocking copy stream
    mutable std::mutex mu_;
    std::vector<FrameLayout> programs_;             // program id -> layout (1-based)
    std::vector<FrameInstance*> instances_;         // instance id -> regions (1-based)
    std::uint64_t next_program_ = 1;
    std::uint64_t next_instance_ = 1;
};

}  // namespace pie_cuda_driver::sampling_ir

// ── extern "C" FFI surface (host-declared in Rust; cbindgen-shaped like the
//    pie_pinned_* / pie_device_* tensor-io surface). The runtime's
//    `CudaControlPlane` calls these directly into `pie_driver_cuda_lib`,
//    bypassing the IPC/driver channel — the X1 control plane's direct fast path. ──
extern "C" {

// B4 — register a trace; returns a 1-based program handle (0 = rejected).
std::uint64_t pie_frame_register(const std::uint8_t* trace, std::size_t trace_len);

// B4/B5 — bind an instance; writes the frame/mirror/word bases. Returns the 1-based
// instance id (0 = unknown program).
std::uint64_t pie_frame_bind(std::uint64_t program, std::uint64_t* out_frame_base,
                             std::uint64_t* out_mirror_base, std::uint64_t* out_word_base);

// B6 — release an instance (fail-loud on unknown/closed).
void pie_frame_close(std::uint64_t instance);

// WRITE leg — async H2D host cells into the device frame (input direction).
void pie_frame_write(std::uint64_t instance, const void* host_src, std::size_t n_bytes,
                     std::size_t frame_offset);

// The CARRIER — commit: D2H mirror + word publish + X0-wake host callback.
// `done` runs on a driver-internal thread once the copy + publish land; it must
// not call CUDA APIs. `n_bytes == 0` mirrors the whole committed frame.
// `forward_evt` is the executor's forward-done `cudaEvent_t` (opaque, or null):
// non-null ⇒ the copy stream waits on it before the D2H (cross-stream ordering).
// `done == null` ⇒ the once-registered completion (see `pie_frame_set_carry_done`)
// is used, so the (a) BRIDGE executor passes null and never threads the fn-ptr.
void pie_frame_carry(std::uint64_t instance, std::size_t frame_offset,
                     std::size_t mirror_offset, std::size_t n_bytes,
                     std::size_t word_index, std::uint64_t target,
                     void* forward_evt,
                     void (*done)(void*), void* user_data);

// Register the stable completion callback (the runtime's `cuda_carry_done`) ONCE
// at init. A `pie_frame_carry` with `done == null` then fires this — so the (a)
// BRIDGE executor call site carries only `{user_data, word_index}` per request,
// never the fn-ptr (guru's register-once refinement).
void pie_frame_set_carry_done(void (*done)(void*));

}  // extern "C"
