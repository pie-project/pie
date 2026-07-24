#pragma once

#include <cstddef>

namespace pie_cuda_driver {

// The engine scheduler's maximum run-ahead depth in FRAMES
// (`configured_max_in_flight` in runtime/engine/src/scheduler/frame.rs).
// Venus: one frame carries up to `PIE_FRAME_SIZE` steps, and every step is
// one upload+launch on the stream, so the driver's staging pools must size
// to STEPS in flight = frames × k. The constants below are the ONE source
// the pinned staging pools size themselves from; a pool with its own
// literal will silently re-serialize submits when either side moves.
inline constexpr std::size_t kSchedulerMaxInFlight = 3;

// Supported frame sizes without staging re-serialization (steps in flight
// = kSchedulerMaxInFlight × kMaxPipelinedFrameSize).
inline constexpr std::size_t kMaxPipelinedFrameSize = 4;

// Pinned host staging slots per upload arena. Must EXCEED the in-flight
// STEP count, not match it: a slot is held from its H2D enqueue until the
// GPU passes the copy, so with every slot of a depth-equal pool pending
// exactly when the next submit arrives, the acquire blocks in
// cudaEventSynchronize for a full GPU step (~1.6ms measured on the 4090
// c64 decode workload; Σ318ms/run measured at k=2 when the pool sat at
// the old wave-depth 4).
inline constexpr std::size_t kUploadStagingDepth =
    kSchedulerMaxInFlight * kMaxPipelinedFrameSize + 1;

}  // namespace pie_cuda_driver
