#pragma once

// Tensor-parallel fan-out for direct/PTIR fires.
//
// Rank 0 (`batch/compose.cpp`) broadcasts the per-fire persistent-input
// payload to every TP follower via `tp_broadcast_inputs`, gated by
// `tp_cpu_gate_notify` so idle followers aren't spinning on NCCL between
// fires. Followers run `tp_follower_serve`, a loop that blocks on the
// matching broadcasts and then runs the same forward kernels (mirroring
// `handle_fire_batch` minus PTIR publication, which is rank-0-only) so the
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

// Issue every per-fire broadcast in dependency order. Caller has already
// refilled `pi.*` with the current fire's data; this just fans them out.
// All ops run on `stream` so they sequence correctly with the kernels that
// follow inside `forward_fn.body`.
void tp_broadcast_inputs(NcclComm& comm, PersistentInputs& pi,
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

void tp_broadcast_mtp_step(
    NcclComm& comm,
    PersistentInputs& pi,
    int rows,
    int draft_step,
    int max_global_tokens,
    cudaStream_t stream);

// Notify TP followers waiting on the CPU gate keyed by `key` that rank 0 has
// begun broadcasting a new fire. No-op when `key` is empty (CPU gate off).
void tp_cpu_gate_notify(const std::string& key);

// TP-follower service loop. Called only on TP ranks > 0. Mirrors
// `handle_fire_batch` minus PTIR publication: the loop blocks on
// `ncclBroadcast(root=0)` for each fire's header + inputs, runs the same
// `forward_fn.body` so the all-reduces inside complete against rank 0, then
// loops. `stop` is checked between fires; rank 0 also sends an explicit
// shutdown header before tearing down so a follower waiting on a broadcast
// unblocks cleanly.
void tp_follower_serve(BatchEngine& engine, std::atomic<bool>& stop);

// Send the shutdown sentinel header from rank 0 so followers exit their
// `tp_follower_serve` loop on the next broadcast.
void tp_send_shutdown(NcclComm& comm, const std::string& cpu_gate_key = {});

}  // namespace pie_cuda_driver
