#pragma once

#include <cstdint>

#include <pie_driver_abi/response_builder.hpp>

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
};

}  // namespace service
}  // namespace pie_portable_driver
