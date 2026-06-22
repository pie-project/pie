#pragma once

#include <cstdint>

namespace pie_driver {
class InProcServer;
}  // namespace pie_driver

namespace pie_metal_driver::service {

// Stub in-process service for the foundation skeleton.
//
// Handles the driver ABI well enough for a clean Health + Forward
// round-trip through `pie_driver::InProcServer`, without depending on the
// MLX compute layer (`src/ops`, `src/executor`) or the model graphs
// (`src/model`) that beta/charlie land on top of this seam:
//
//   * Health         → status 0.
//   * Forward        → dummy tokens / zeroed distributions, shaped exactly
//                      like the per-request sampler stream expects (mirrors
//                      driver/dummy's handler so the runtime stays in sync).
//   * Copy / KV / RS → not-supported status (no KV cache yet — delta wires it).
//   * Adapter        → no-op success.
//
// Once the executor lands, this class grows an `Executor&` member and the
// Forward arm delegates to it (see driver/cuda/src/service for the shape).
class InProcService {
public:
    explicit InProcService(std::uint32_t vocab_size = 1) : vocab_size_(vocab_size) {}

    void serve_forever(pie_driver::InProcServer& server);

    std::uint64_t handled() const noexcept { return handled_; }

private:
    std::uint32_t vocab_size_;
    std::uint64_t handled_ = 0;
};

}  // namespace pie_metal_driver::service
