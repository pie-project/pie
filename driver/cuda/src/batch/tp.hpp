#pragma once

// Tensor-parallel fan-out for direct/PTIR fires.
//
// Rank 0 (`batch/frame.cpp`) broadcasts the per-fire persistent-input
// payload to every TP follower via `tp_broadcast_inputs`, gated by
// `tp_cpu_gate_notify` so idle followers aren't spinning on NCCL between
// fires. Followers run `tp_follower_serve`, a loop that blocks on the
// matching broadcasts and then runs the same forward kernels (mirroring
// the frame step pipeline minus PTIR publication, which is rank-0-only) so the
// all-reduces inside `forward_fn.body` complete against rank 0.

#include <atomic>
#include <cstdint>
#include <string>

#include <cuda_runtime.h>

#include "distributed.hpp"
#include "batch/persistent_inputs.hpp"
#include "batch/rs_metadata.hpp"

namespace pie_cuda_driver {

struct BatchEngine;

// Host-side CSR views rank 0 uses for its own attention plan. Handed to the
// follower through the shared mailbox so both ranks plan from identical
// values and the follower needs no device readback.
struct TpFirePlanViews {
    const std::uint32_t* qo_indptr = nullptr;          // R + 1
    const std::uint32_t* kv_page_indptr = nullptr;     // R + 1
    const std::uint32_t* kv_last_page_lens = nullptr;  // R
    const std::uint32_t* kv_page_indices = nullptr;    // kv_page_indices_len
    std::size_t kv_page_indices_len = 0;
    // Host-known recurrent-state metadata (present when the header's
    // has_slot_ids / rs counts say so). Rank 0 stages exactly these bytes to
    // its own device, so carrying them through the mailbox lets the follower
    // upload its copy H2D instead of receiving ~6 sub-KB broadcasts whose
    // cost is pure launch latency — and drop its per-fire D2H readback +
    // stream synchronize entirely.
    const std::int32_t* slot_ids = nullptr;              // R
    const std::uint8_t* is_fresh = nullptr;              // R
    const std::uint8_t* rs_slot_flags = nullptr;         // R
    const std::uint32_t* rs_fold_lens = nullptr;         // rs_fold_lens_len
    std::size_t rs_fold_lens_len = 0;
    const std::uint32_t* rs_buffer_slot_indptr = nullptr;  // R + 1 (buffered)
    const std::uint32_t* rs_buffer_slot_ids = nullptr;     // rs_buffer_ids_len
    std::size_t rs_buffer_ids_len = 0;
};

// Publish the fire's host-known metadata (header + planner CSR views) into the
// process-shared mailbox and wake the follower — WITHOUT touching the device.
//
// Called as early as possible in the enqueue path. Rank 0 pre-enqueues its
// whole step (`pull_validate`, compose, broadcast, forward) in one go, so its
// forward starts ~17 us after the payload broadcast completes. The follower is
// reactive: it cannot enqueue anything until it is notified, and profiling
// showed its forward starting 588 us after the same broadcast, which rank 0
// then absorbed as a 583 us wait inside the step's FIRST all-reduce (73% of
// rank 0's entire all-reduce time). Waking the follower before rank 0's own
// uploads and compose hands it that window as a head start.
void tp_publish_fire(const std::string& cpu_gate_key,
                     const TpFirePlanViews& views,
                     int N, int R, bool is_pure_decode,
                     int kv_indices_count,
                     int required_kv_pages,
                     int mask_bytes, int mask_indptr_count,
                     bool has_slot_ids,
                     bool has_write_desc,
                     int logit_rows,
                     int structured_window_left,
                     RsExecutionMode rs_mode,
                     int rs_fold_lens_count,
                     int rs_buffer_ids_count);

// Issue every per-fire broadcast in dependency order. Caller has already
// refilled `pi.*` with the current fire's data; this just fans them out.
// All ops run on `stream` so they sequence correctly with the kernels that
// follow inside `forward_fn.body`.
//
// Also publishes the fire header and `views` into the process-shared mailbox
// and notifies the CPU gate, in that order: the follower reads both from host
// memory instead of paying a device round trip for them.
void tp_broadcast_inputs(NcclComm& comm, PersistentInputs& pi,
                         const std::string& cpu_gate_key,
                         const TpFirePlanViews& views,
                         int N, int R, bool is_pure_decode,
                         int kv_indices_count,
                         int required_kv_pages,
                         int mask_bytes, int mask_indptr_count,
                         bool has_slot_ids,
                         bool has_write_desc,
                         int logit_rows,
                         int structured_window_left,
                         RsExecutionMode rs_mode,
                         int rs_fold_lens_count,
                         int rs_buffer_ids_count,
                         cudaStream_t stream);

// Publish an MTP draft step's header into the mailbox and wake the follower.
// Every gate epoch must own exactly one published slot or the follower's slot
// index drifts away from rank 0's.
void tp_publish_mtp(const std::string& cpu_gate_key,
                    int rows,
                    int draft_step,
                    int max_global_tokens);

void tp_broadcast_mtp_step(
    NcclComm& comm,
    PersistentInputs& pi,
    const std::string& cpu_gate_key,
    int rows,
    int draft_step,
    int max_global_tokens,
    cudaStream_t stream);

// Notify TP followers waiting on the CPU gate keyed by `key` that rank 0 has
// begun broadcasting a new fire. No-op when `key` is empty (CPU gate off).
void tp_watchdog_mark_phase(int phase);
void tp_watchdog_count_collective(int rank);

// Fine-grained stall localization (PIE_TP_WATCHDOG=1). Each rank stores the
// stage it last entered as a relaxed atomic; the watchdog thread prints both
// on stall. Stores are branch-free and ~free, so they do not perturb the
// cross-rank timing the way per-fire logging does.
enum class TpRank0Stage : int {
    kNone = 0, kPublish, kUploads, kCompose, kGraphPad, kWaitSlot,
    kBroadcast, kBroadcastDone, kAttnPlan, kForward, kRecordSlot, kMtpDrafts,
    kSettle,
};
enum class TpFollowerStage : int {
    kNone = 0, kGateWait, kHeaderRead, kPayloadRecv, kHostViews, kRsPlan,
    kAttnPlan, kGraphLookup, kGraphCapture, kGraphLaunch, kEagerForward,
    kLoopTop,
};
void tp_watchdog_mark_rank0_stage(TpRank0Stage stage);
void tp_watchdog_mark_follower_stage(TpFollowerStage stage);

void tp_cpu_gate_notify(const std::string& key);

// Release every waiter parked on the CPU gate keyed by `key` and mark the gate
// permanently stopped, so followers leave `tp_follower_serve` without posting
// an unmatched collective. Both ranks share the gate (it is keyed by the NCCL
// unique id), which makes teardown independent of Context destruction order.
// Idempotent and safe to call from any rank.
void tp_cpu_gate_request_stop(const std::string& key);

// TP-follower service loop. Called only on TP ranks > 0. Mirrors
// the frame step pipeline minus PTIR publication: the loop blocks on
// `ncclBroadcast(root=0)` for each fire's header + inputs, runs the same
// `forward_fn.body` so the all-reduces inside complete against rank 0, then
// loops. The loop exits when `stop` is set or the CPU gate is stopped; with
// the gate off, rank 0's shutdown header releases a follower parked inside
// `ncclBroadcast`.
void tp_follower_serve(BatchEngine& engine, std::atomic<bool>& stop);

// Send the shutdown sentinel header from rank 0 so followers exit their
// `tp_follower_serve` loop on the next broadcast.
void tp_send_shutdown(NcclComm& comm, const std::string& cpu_gate_key = {});

}  // namespace pie_cuda_driver
