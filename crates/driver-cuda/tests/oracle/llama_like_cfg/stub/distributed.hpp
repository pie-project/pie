#pragma once
// Stub csrc/src/distributed.hpp. llama_like.cpp names NcclComm only inside
// the forward body (discarded by --gc-sections), so declarations suffice —
// nothing here is ever linked or run.
#include <cstddef>
#include "cuda_runtime.h"

using ncclRedOp_t = int;
constexpr ncclRedOp_t ncclSum = 0;

namespace pie_cuda_driver {

namespace kernels::comm { class CustomAllReduce; }

class NcclComm {
public:
    kernels::comm::CustomAllReduce* custom_all_reduce();
    void all_reduce_bf16(void* buf, std::size_t n, ncclRedOp_t op,
                         cudaStream_t stream);
    void all_reduce_bf16_out(const void* in, void* out, std::size_t n,
                             ncclRedOp_t op, cudaStream_t stream);
};

}  // namespace pie_cuda_driver
