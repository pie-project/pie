// The only translation unit.
//
// `nvcc -fatbin` takes one input, so the kernels are gathered here rather than
// compiled separately and loaded as several modules. That is not a workaround
// but the shape this wants: one unit lets nvcc inline a device function into
// every kernel that calls it, which is what makes the shared parser step -
// `_replay_group` in the Triton engine, written out twice there because a
// Triton kernel cannot call another - a single function here.
//
// Order matters only in that a header must precede its users.

#include "arena.cuh"
#include "probe.cuh"
#include "readback.cuh"
#include "locate.cuh"
#include "candidate.cuh"
#include "commit.cuh"
#include "mask.cuh"
#include "memo.cuh"
