#pragma once

// batch/: measure the forward step, so the planner can stop guessing.
//
// `plan_cuda_memory` picks `max_forward_tokens` by scoring a candidate lattice
// with an analytic model. Where that model disagreed with reality the tree grew
// per-(model, GPU) overrides carrying hand-measured constants. The fix is not a
// better guess: it is to let the driver run the measurement itself, on the
// machine it is actually running on, and cache the result for the planner to
// read on the next start (store/planner_profile_cache.hpp).
//
// Calibration is an explicit operator action (`[batching] calibrate_planner` in
// the driver's startup TOML), not something a serving process does behind your
// back — the sweep costs a few seconds of startup and writes to the profile
// cache. A process that never calibrates selects exactly as it did before.
//
// That startup TOML is generated per boot, and stage one of `pie config tune`
// is the only thing that ever sets the key in it. There is no key in
// the pie config: the request is one run of a measurement, not a description of
// a deployment, and written down it needed the operator to perform a three-step
// ritual whose third step — turning it back off — was the one that mattered.
// Left on, this boot serves from the arena the planner sized to be MEASURED.
//
// It was `PIE_CUDA_PLANNER_CALIBRATE=1` until commit b38414903 deleted the env
// var under the flag-deletion rule and left no replacement, which made the
// whole file unreachable. It came back as a pie config key, which was the same
// mistake in a different file — env var and config were treated as the only two
// options — and is now a command.

#include <cstddef>

namespace pie_cuda_driver {

struct BatchEngine;

// True when the operator asked for a calibration sweep. Exposed so the caller
// can materialise the arenas the sweep needs without paying for them when it is
// not going to run.
bool planner_calibration_requested();

// Publish the request. Called once by `load_config`, before anything reads it —
// the same pattern as `mutable_cache_dir()`, and for the same reason: the
// planner and the batch engine both ask, and neither is constructed anywhere
// that sees a `Config`.
void set_planner_calibration_requested(bool requested);

// Sweeps the prefill token budget at the plan's own request width, times the
// real forward body at each rung, and stores the fastest shape in the planner
// profile cache. Returns the number of rungs measured (0 when calibration is
// not requested or the engine cannot be swept).
//
// Must be called with the workspace/attention arenas materialised and the
// graph lattice already captured — i.e. from the same place as
// `capture_forward_graph_lattice`.
std::size_t calibrate_memory_planner(BatchEngine& engine,
                                     int tp_size,
                                     int kv_page_size);

}  // namespace pie_cuda_driver
