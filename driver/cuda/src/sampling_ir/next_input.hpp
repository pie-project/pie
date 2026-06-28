#pragma once

// #6 WS8 P2 — device machinery for the inter-pass `next-input` pipeline link.
// SEPARABLE module (the `group.hpp` pattern): built + device-verified in isolation
// before composing into bravo's new-WIT e2e bring-up.
//
// The link removes the host `output().await` + copy between decode passes: pass t
// samples → `pi.sampled[N]` (device-resident, the scatter target every sampler
// kind writes) → pass t+1's input is sourced device-side from it. This module owns:
//
//   inject_next_input — `next_input[dest_pos] = pi.sampled[src_row]` per link
//     (a scatter with a position map; structurally the #10 scatter + src→dst).
//   inject_next_input_after — the same, gated on the producer's sample-done event
//     (the inject stream waits it), so the forward-only graph that follows reads a
//     populated `next_input`. The forward graph itself is captured/replayed by the
//     executor; this gates it.
//
// Sampler-kind-agnostic: the inject only reads `pi.sampled`, regardless of how each
// row was sampled (argmax / temp BakedIR / custom). Single-buffer + serial-on-stream
// (the async win is removing host bubbles, not overlapping dependent kernels), so
// event ordering alone is hazard-free (write strictly precedes read in stream order).
//
// COMMIT-NEUTRALITY INVARIANT (Q3, load-bearing): the inject writes only the token
// *id* into the consumer's transient `next_input` buffer — it touches NO KV and
// calls NO commit. The KV for that token is produced by t+1's forward into
// UNCOMMITTED working pages; whether it persists is governed by the executor's
// withhold-commit/release on `done` (golf's deferred-commit), which this inject is
// upstream of and never forces. So the device inject is commit-neutral by
// construction — the speculative final-token KV stays releasable (no EOS-in-KV).

#include <cstddef>
#include <cstdint>
#include <span>

#include <cuda_runtime.h>

namespace pie_cuda_driver::sampling_ir {

// A `dest_pos` of this value is the `-1` ignore lane (P1's `u32::MAX` sentinel):
// the link is skipped (that consumer position keeps whatever it already held).
constexpr std::uint32_t kIgnorePosition = 0xFFFFFFFFu;

// One inter-pass link: producer sampling row `src_row` → consumer input position
// `dest_pos` (the N→N per-sequence map carried by P1's `PipelineLink{positions}`).
struct NextInputLink {
    std::uint32_t src_row;
    std::uint32_t dest_pos;  // kIgnorePosition = skip
};

// Inject: for each link, `next_input[dest_pos] = sampled[src_row]` (i32 tokens).
// `dest_pos == kIgnorePosition` is skipped. Async on `stream`; the caller orders
// the consumer forward after it. `sampled` = producer `pi.sampled` device base
// (i32 [N_producer]); `next_input` = consumer input-token device base (i32 [*]).
void inject_next_input(const void* sampled, std::span<const NextInputLink> links,
                       void* next_input, cudaStream_t stream);

// Event-ordered inject: wait `producer_done` (recorded after the producer's sample
// launch) on `stream`, then inject — so a forward-only graph replayed after this on
// `stream` reads a populated `next_input` with the producer's tokens. This is the
// device-resident replacement for P1's host `resolve().await` + inject.
void inject_next_input_after(const void* sampled, std::span<const NextInputLink> links,
                             void* next_input, cudaEvent_t producer_done,
                             cudaStream_t stream);

}  // namespace pie_cuda_driver::sampling_ir
