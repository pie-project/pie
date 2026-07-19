#pragma once

// Frame pipeline (ABI v14, second landing): the frame — k sealed forward
// steps — is the GPU forward-pass unit, executed as three modules that
// replace the old `handle_fire_batch` per-step monolith:
//
//   * `prepare_step` (FramePrepare)  — ALL of one step's host work, run for
//     every step at frame entry when nothing of the frame is enqueued and
//     there is nothing to wait on: wave admission (begin_host), descriptor
//     resolution, geometry composition, mask decode, RS/sampling planning,
//     graph-pad decision, and the step's parameter block staged into
//     per-step pinned upload slots. Nothing reaches the stream.
//   * `enqueue_step` (StepEnqueue)   — enqueue-only: the wave's stream half
//     (pull-validate + Prologue), the staged parameter-block commits, mask
//     pack / pad / device-composition kernels, the attention-plan hook
//     (host-light; its device commit is inherently enqueue-track), and the
//     forward body. Steps enqueue in order — step i+1's pull-validate reads
//     ring state step i's settlement publishes, carried by stream order.
//   * `settle_step` (FrameSettle)    — the step's settlement enqueue
//     (`Dispatch::finish`). The frame tail carries the frame completion;
//     settle callbacks are stream-ordered, so the tail's notify implies
//     every step's terminals are latched.
//
// This hoists T_prepare once per frame (the structural lever when T_gpu
// shrinks) and reduces the backend step contract to "consume a prepared
// per-step parameter block".

#include <cstdint>
#include <memory>

#include "pie_native/launch_view.hpp"
#include <pie_driver_abi.h>

namespace pie_cuda_driver {

struct BatchEngine;

// One step's host-prepared execution state, owned across the three phase
// calls. Movable, frame-scoped: it must outlive `settle_step` (the
// settlement enqueue reads geometry it owns) and dies with the frame.
class PreparedStep {
  public:
    PreparedStep();
    ~PreparedStep();
    PreparedStep(PreparedStep&&) noexcept;
    PreparedStep& operator=(PreparedStep&&) noexcept;
    PreparedStep(const PreparedStep&) = delete;
    PreparedStep& operator=(const PreparedStep&) = delete;

    struct Impl;
    Impl* impl() const noexcept { return impl_.get(); }

  private:
    std::unique_ptr<Impl> impl_;
};

// FramePrepare for one step. `view` must outlive the PreparedStep (the
// frame driver owns the expanded step views for the whole frame).
// `previous` is the SAME FRAME's preceding prepared step (nullptr for the
// frame head): when every attention-plan input is content-identical to
// it, the step marks its plan skippable — the workspace already holds
// the identical plan (plan-once-per-frame; intra-frame chained decode
// steps share R and plan from frame-constant envelope bounds).
void prepare_step(
    BatchEngine& engine,
    const pie_native::LaunchView& view,
    PreparedStep& step,
    const PreparedStep* previous = nullptr);

// StepEnqueue: the step's device work, kernel-launch/copy-enqueue only.
void enqueue_step(BatchEngine& engine, PreparedStep& step);

// FrameSettle: the step's settlement enqueue. Non-tail steps pass an empty
// completion; the tail step carries the frame completion.
void settle_step(
    BatchEngine& engine,
    const PieRuntimeCallbacks& runtime,
    PieCompletion completion,
    PreparedStep& step);

// Failure unwind for a step whose staged wave never settled (prepare or
// enqueue fault, or a sibling step's fault failing the frame). Safe on any
// phase state, including a never-prepared step.
void abort_step(BatchEngine& engine, PreparedStep& step) noexcept;

}  // namespace pie_cuda_driver
