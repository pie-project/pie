#pragma once
// scratch_schedule.hpp — beta's WAR/WAW activation-buffer allocator for the decode DAG.
//
// The Scratch region is a small pool of SCRATCH_POOL ping-pong buffers (decode_abi.hpp).
// Every dispatch reads its input activation(s) from scratch and writes its output
// activation to scratch (except the few slots delta owns: weights, GDN state, KV pages,
// IO scalars/logits). beta owns the schedule that assigns each dispatch's scratch X/Out
// to a concrete pool buffer such that no two simultaneously-live activations alias the
// same buffer (the WAR/WAW hazard), then binds them by ordinal.
//
// This header exposes the PURE schedule (no Metal): build_scratch_schedule(dag, g)
// re-derives the activation dataflow from the fixed DAG order and linear-scan colors the
// live ranges onto pool buffers. bind_scratch(ctx, dag, sched, pool) then binds each
// dispatch's scratch bind-indices to pool[buffer_id] by ordinal (the Metal step, in the
// .cpp, behind alpha's RawMetalContext).
//
// Ownership seam (heap_bind.hpp): delta binds weight/state/KV/IO slots by ordinal; beta
// binds the activation X/Out scratch over the SAME ordinals. The Scratch region must be
// sized for `colors_used` buffers each at the max activation footprint (gdn mixed = 6144,
// intermediate = 3584, qg = 4096 elems) — see scratch_schedule_probe for the exact count.

#include <cstdint>
#include <vector>

#include "decode_abi.hpp"
#include "decode_step.hpp"

namespace pie_metal_driver::raw_metal {

// One scratch activation slot a dispatch binds: bind_index <- pool buffer `buffer_id`.
struct ScratchBind {
    uint8_t bind_index;  // bind::<Kind> activation slot (X / Out / Gate / ...)
    int     buffer_id;   // index into the SCRATCH_POOL pool [0, colors_used)
};

// Per-dispatch scratch bindings (parallel to the DAG; index == ordinal).
struct ScratchDispatch {
    std::vector<ScratchBind> binds;  // the activation slots this dispatch reads/writes
};

// The full schedule: a per-dispatch binding list + how many pool buffers it uses.
struct ScratchSchedule {
    std::vector<ScratchDispatch> per_dispatch;  // size == dag.size()
    int  colors_used = 0;                        // distinct pool buffers needed (<= SCRATCH_POOL)
    bool hazard_free = false;                     // self-checked: no overlapping value shares a buffer
};

// Derive the activation dataflow from the DAG order + geometry and linear-scan color the
// live ranges onto pool buffers (hazard-free, honoring the concurrent ‖-pairs). Pure.
//
// `no_recycle` (dump/diagnostic build): give every distinct activation value its OWN buffer
// (colors_used == number of values, zero reuse). This preserves every intermediate to
// end-of-run so all golden taps can be read post-run, AND is the scratch-aliasing-race
// diagnostic: if the non-determinism vanishes under no_recycle, it was a coloring/‖-pair
// aliasing race; if it persists, it is an in-kernel race or uninitialized read.
ScratchSchedule build_scratch_schedule(const std::vector<Dispatch>& dag,
                                       const DecodeGeometry& g,
                                       bool no_recycle = false);

// Bind the schedule's activation slots into the arg table by ordinal: for each dispatch's
// ScratchBind{bind_index, buffer_id}, arg_bind_ordinal(ordinal, bind_index, pool[buffer_id]).
// `pool` is delta's BoundDecode.scratch[] (SCRATCH_POOL slots); pool_n must be >= colors_used.
// Call after stage/bind of delta's slots and BEFORE make_resident(). Metal (see .cpp).
void bind_scratch(RawMetalContext& ctx,
                  const std::vector<Dispatch>& dag,
                  const ScratchSchedule& sched,
                  const SlotHandle* pool,
                  int pool_n);

}  // namespace pie_metal_driver::raw_metal
