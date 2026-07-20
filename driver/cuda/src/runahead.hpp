#pragma once

#include <cstddef>

namespace pie_cuda_driver {

// The engine scheduler's maximum run-ahead depth (`MAX_IN_FLIGHT` in
// runtime/engine/src/scheduler/quorum.rs): one wave computing plus two
// staged. The two constants below are the ONE source the driver's pinned
// staging pools size themselves from; a pool with its own literal will
// silently re-serialize submits when either side moves.
inline constexpr std::size_t kSchedulerMaxInFlight = 3;

// Pinned host staging slots per upload arena. Must EXCEED the run-ahead
// depth, not match it: a slot is held from its H2D enqueue until the GPU
// passes the copy, so with `kSchedulerMaxInFlight` waves in flight every
// slot of a depth-equal pool is pending exactly when the next submit
// arrives, and the acquire blocks in cudaEventSynchronize for a full GPU
// wave (~1.6ms measured on the 4090 c64 decode workload — it was the
// entire epilogue-enqueue cost before the grouped-stage arena moved off
// a depth-equal pool).
inline constexpr std::size_t kUploadStagingDepth = kSchedulerMaxInFlight + 1;

}  // namespace pie_cuda_driver
