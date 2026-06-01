#pragma once

// Tensor-parallel collective communicator — a thin NCCL wrapper for the
// clean-rewrite device library. De-branded from driver/cuda's
// `pie_cuda_driver::NcclComm` (src/distributed.{hpp,cpp}) into
// `pie_cuda_device::ops`, stripped down to the collectives the new forward
// path actually needs (in-place sum all-reduce) plus the unique-id plumbing
// the Rust control plane drives.
//
// Topology / lifecycle (matches the old driver's model):
//   * Each TP rank is its own actor that joins ONE `ncclComm_t` keyed by a
//     shared `ncclUniqueId`.
//   * Rank 0 mints the id via `tp_comm_get_unique_id` and the Rust control
//     plane broadcasts the 128 raw bytes (NCCL_UNIQUE_ID_BYTES) to all ranks
//     out-of-band (e.g. over the existing rank-0 rendezvous channel). Every
//     rank then calls `tp_comm_init(rank, world_size, &id)`.
//   * world_size == 1 is the single-GPU fast path: the comm still initializes
//     and all_reduce is an identity (sum over one rank).
//
// Build flavors:
//   * When NCCL headers are available the wrapper links real NCCL. The build
//     system defines PIE_CUDA_DEVICE_HAS_NCCL=1 for that path. We probe the
//     header directly with __has_include so this TU also compiles standalone.
//   * When NCCL is absent the wrapper still compiles: init for world_size==1
//     succeeds with a null comm, all_reduce is a no-op identity (correct for a
//     single rank), and any world_size>1 request fails with
//     cudaErrorNotSupported. This keeps the device library buildable on hosts
//     without NCCL and lets the selftest run its single-rank assertions.

#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>

#if !defined(PIE_CUDA_DEVICE_HAS_NCCL)
#  if defined(__has_include)
#    if __has_include(<nccl.h>)
#      define PIE_CUDA_DEVICE_HAS_NCCL 1
#    else
#      define PIE_CUDA_DEVICE_HAS_NCCL 0
#    endif
#  else
#    define PIE_CUDA_DEVICE_HAS_NCCL 0
#  endif
#endif

#if PIE_CUDA_DEVICE_HAS_NCCL
#  include <nccl.h>
#else
// Minimal stand-in so the public signatures (which take an `ncclUniqueId*`)
// type-check when NCCL is unavailable. Layout matches NCCL's definition:
//   typedef struct { char internal[128]; } ncclUniqueId;
// (NCCL_UNIQUE_ID_BYTES == 128). The control plane only ever moves the raw
// bytes, so this is ABI-compatible for the broadcast path.
typedef struct {
  char internal[128];
} ncclUniqueId;
#endif

namespace pie_cuda_device::ops {

// Opaque communicator handle. Holds the NCCL comm plus the rank metadata and
// the default stream collectives run on when the caller passes a null stream.
// Construct via tp_comm_init; tear down via tp_comm_destroy. Not copyable.
struct TpComm {
#if PIE_CUDA_DEVICE_HAS_NCCL
  ncclComm_t comm = nullptr;
#else
  void* comm = nullptr;
#endif
  int rank = 0;
  int world_size = 1;
  // Default stream for collectives when the per-call stream argument is null.
  // A null cudaStream_t IS the CUDA default stream, so this defaults to it.
  cudaStream_t stream = nullptr;
};

// Mint a fresh unique id (rank 0 only). The control plane broadcasts the raw
// NCCL_UNIQUE_ID_BYTES (128) of `*out` to the other ranks, which feed them
// into tp_comm_init. Returns cudaSuccess on success.
//   * Without NCCL: returns cudaErrorNotSupported (no multi-rank id to mint),
//     after zeroing *out so callers don't read uninitialized bytes.
cudaError_t tp_comm_get_unique_id(ncclUniqueId* out);

// Join the TP group as `rank` of `world_size`, using the shared `id`. Returns
// a heap-allocated TpComm (caller owns it; free via tp_comm_destroy) or
// nullptr on failure. `id` may be null only when world_size == 1.
//   * world_size == 1: single-rank fast path. With NCCL we still create a real
//     1-rank comm so the collective entry points are uniform; without NCCL we
//     return a comm with a null handle (all_reduce becomes identity).
//   * world_size  > 1 without NCCL: returns nullptr (unsupported).
TpComm* tp_comm_init(int rank, int world_size, const ncclUniqueId* id);

// In-place sum all-reduce over `n_elems` bf16 elements at `buf` (device ptr).
// Reduces in bf16 (ncclBfloat16). If `stream` is null, uses comm->stream.
// For world_size == 1 this is an identity (buf unchanged). Returns cudaSuccess
// on success; cudaErrorInvalidValue on bad args; cudaErrorUnknown on a NCCL
// failure. The collective is enqueued on the stream and NOT synchronized —
// the caller syncs the stream when it needs the result.
cudaError_t tp_all_reduce_bf16(TpComm* c, void* buf, std::size_t n_elems,
                               cudaStream_t stream);

// fp32 variant of the above (ncclFloat32). Used e.g. for cross-rank absmax
// reductions where bf16 accumulation would lose precision.
cudaError_t tp_all_reduce_fp32(TpComm* c, void* buf, std::size_t n_elems,
                               cudaStream_t stream);

// Destroy the comm and free the TpComm. Safe to call with nullptr. Best-effort
// on the NCCL side (teardown failures are swallowed).
void tp_comm_destroy(TpComm* c);

}  // namespace pie_cuda_device::ops
