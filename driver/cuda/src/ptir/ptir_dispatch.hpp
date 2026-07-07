#pragma once

// Driver-side PTIR (thrust-3) stage-program dispatcher — a CUDA-FREE façade over
// the tier-0 runtime (`program_runtime.hpp`). The impl + the `__global__` tier-0
// kernels live in `ptir_dispatch.cu`, so host `.cpp` translation units that
// include `executor.hpp` never pull device code (the tier-0 headers only compile
// under nvcc). This is the driver half of P2c: the executor calls `run()` when a
// forward request carries `ptir_program_*`; the dispatcher decodes (hash-cache),
// instantiates (persistent, by wire instance id), fires, and harvests outputs.

#include <cstdint>
#include <memory>

#include <cuda_runtime.h>

#include <pie_driver_abi/view.hpp>

namespace pie_cuda_driver::ptir {

class PtirDispatch {
  public:
    PtirDispatch();
    ~PtirDispatch();
    PtirDispatch(const PtirDispatch&) = delete;
    PtirDispatch& operator=(const PtirDispatch&) = delete;

    // Decode / instantiate / fire the request's `ptir_program_*` stage programs
    // on `logits` (the Logits-intrinsic base, `[rows, vocab]`) and fill
    // `out_resp.ptir_output_*` — a per-program CSR of the committed READER-channel
    // `(channel, wire_bytes)` outputs. The staging buffers live until the next
    // `run()` (long enough for `send_response`). Returns true iff the request
    // carried `ptir_program_*` (i.e. this dispatcher handled it).
    bool run(const pie_driver::PieForwardRequestView& view,
             pie_driver::PieForwardResponseView& out_resp,
             const void* logits, std::uint32_t vocab, cudaStream_t stream);

  private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace pie_cuda_driver::ptir
