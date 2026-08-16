#pragma once

#include <cstdint>

#include <pie_bridge/response_builder.hpp>

namespace pie_driver {
class InProcServer;
}

namespace pie_portable_driver {

class AdapterPool;
class Executor;
class HostSwapPool;
class Model;

namespace service {

class InProcService {
public:
    InProcService(Executor& executor,
                  Model& model,
                  HostSwapPool* swap_pool,
                  AdapterPool& adapters,
                  bool verbose);

    void serve_forever(pie_driver::InProcServer& server);

    std::uint64_t handled() const noexcept { return handled_; }

private:
    Executor& executor_;
    Model& model_;
    HostSwapPool* swap_pool_;
    AdapterPool& adapters_;
    bool verbose_;
    std::uint64_t handled_ = 0;
    pie_driver::ResponseBuilder response_builder_;

    // True when the KV backend does not support ggml_backend_tensor_get
    // at non-zero offsets (ggml-Vulkan silently returns zeros for such
    // reads). Set once in the constructor; guards CopyD2D and CopyD2H.
    // See: https://github.com/pie-project/pie/issues/418
    bool needs_full_tensor_staging_ = false;
};

}  // namespace service
}  // namespace pie_portable_driver
