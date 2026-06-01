// Tensor-parallel NCCL communicator wrapper. See tp_comm.cuh for the contract.
//
// De-branded from driver/cuda's pie_cuda_driver::NcclComm (src/distributed.cpp).
// Differences from the old driver:
//   * Surface trimmed to the collectives the new path uses: sum all-reduce in
//     bf16/fp32, plus the unique-id mint. (The old driver also had all-gather,
//     broadcast, barrier, and a custom NVLink all-reduce fast path; those are
//     deferred until the new forward/TP path needs them.)
//   * Lifecycle is plain C-style init/destroy returning an opaque TpComm* so
//     the ABI layer can wrap it without C++ types crossing the boundary.
//   * Compiles with or without NCCL (see PIE_CUDA_DEVICE_HAS_NCCL in the .cuh).

#include "ops/tp_comm.cuh"

#include <new>

namespace pie_cuda_device::ops {

#if PIE_CUDA_DEVICE_HAS_NCCL

cudaError_t tp_comm_get_unique_id(ncclUniqueId* out) {
  if (out == nullptr) {
    return cudaErrorInvalidValue;
  }
  if (ncclGetUniqueId(out) != ncclSuccess) {
    return cudaErrorUnknown;
  }
  return cudaSuccess;
}

TpComm* tp_comm_init(int rank, int world_size, const ncclUniqueId* id) {
  if (world_size < 1 || rank < 0 || rank >= world_size) {
    return nullptr;
  }
  if (world_size > 1 && id == nullptr) {
    // Multi-rank requires the shared id (the control plane broadcasts it).
    return nullptr;
  }

  TpComm* c = new (std::nothrow) TpComm();
  if (c == nullptr) {
    return nullptr;
  }
  c->rank = rank;
  c->world_size = world_size;
  c->stream = nullptr;  // CUDA default stream until the caller overrides.

  if (world_size == 1 && id == nullptr) {
    // Single-rank fast path with no shared id: mint a private one so the
    // 1-rank comm still initializes (sum over one rank is the identity, but
    // keeping a live comm makes the collective entry points uniform).
    ncclUniqueId local_id;
    if (ncclGetUniqueId(&local_id) != ncclSuccess ||
        ncclCommInitRank(&c->comm, world_size, local_id, rank) != ncclSuccess) {
      delete c;
      return nullptr;
    }
    return c;
  }

  if (ncclCommInitRank(&c->comm, world_size, *id, rank) != ncclSuccess) {
    delete c;
    return nullptr;
  }
  return c;
}

static cudaError_t all_reduce_sum(TpComm* c, void* buf, std::size_t n_elems,
                                  ncclDataType_t dtype, cudaStream_t stream) {
  if (c == nullptr || (buf == nullptr && n_elems != 0)) {
    return cudaErrorInvalidValue;
  }
  if (c->world_size == 1) {
    return cudaSuccess;  // identity: sum over one rank.
  }
  if (c->comm == nullptr) {
    return cudaErrorNotSupported;
  }
  cudaStream_t s = (stream != nullptr) ? stream : c->stream;
  ncclResult_t r =
      ncclAllReduce(buf, buf, n_elems, dtype, ncclSum, c->comm, s);
  return (r == ncclSuccess) ? cudaSuccess : cudaErrorUnknown;
}

cudaError_t tp_all_reduce_bf16(TpComm* c, void* buf, std::size_t n_elems,
                               cudaStream_t stream) {
  return all_reduce_sum(c, buf, n_elems, ncclBfloat16, stream);
}

cudaError_t tp_all_reduce_fp32(TpComm* c, void* buf, std::size_t n_elems,
                               cudaStream_t stream) {
  return all_reduce_sum(c, buf, n_elems, ncclFloat32, stream);
}

void tp_comm_destroy(TpComm* c) {
  if (c == nullptr) {
    return;
  }
  if (c->comm != nullptr) {
    // Best-effort: teardown failures are non-fatal.
    ncclCommDestroy(c->comm);
    c->comm = nullptr;
  }
  delete c;
}

#else  // !PIE_CUDA_DEVICE_HAS_NCCL — NCCL-less stub build.

cudaError_t tp_comm_get_unique_id(ncclUniqueId* out) {
  if (out == nullptr) {
    return cudaErrorInvalidValue;
  }
  for (auto& b : out->internal) {
    b = 0;
  }
  return cudaErrorNotSupported;
}

TpComm* tp_comm_init(int rank, int world_size, const ncclUniqueId* /*id*/) {
  if (world_size != 1 || rank != 0) {
    // Multi-rank is impossible without NCCL.
    return nullptr;
  }
  TpComm* c = new (std::nothrow) TpComm();
  if (c == nullptr) {
    return nullptr;
  }
  c->comm = nullptr;
  c->rank = 0;
  c->world_size = 1;
  c->stream = nullptr;
  return c;
}

cudaError_t tp_all_reduce_bf16(TpComm* c, void* buf, std::size_t n_elems,
                               cudaStream_t /*stream*/) {
  if (c == nullptr || (buf == nullptr && n_elems != 0)) {
    return cudaErrorInvalidValue;
  }
  // world_size == 1 (the only possibility here): identity.
  return cudaSuccess;
}

cudaError_t tp_all_reduce_fp32(TpComm* c, void* buf, std::size_t n_elems,
                               cudaStream_t /*stream*/) {
  if (c == nullptr || (buf == nullptr && n_elems != 0)) {
    return cudaErrorInvalidValue;
  }
  return cudaSuccess;
}

void tp_comm_destroy(TpComm* c) { delete c; }

#endif  // PIE_CUDA_DEVICE_HAS_NCCL

}  // namespace pie_cuda_device::ops
