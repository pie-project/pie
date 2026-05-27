#pragma once

#include <cstdint>

namespace pie_driver {
class InProcServer;
}  // namespace pie_driver

namespace pie_cuda_driver {

struct Executor;
class KvCache;
class SwapPool;

namespace service {

class InProcService {
public:
    InProcService(Executor& executor, KvCache& kv_cache, SwapPool& swap_pool);

    void serve_forever(pie_driver::InProcServer& server);

    std::uint64_t handled() const noexcept { return handled_; }

private:
    Executor& executor_;
    KvCache& kv_cache_;
    SwapPool& swap_pool_;
    std::uint64_t handled_ = 0;
};

}  // namespace service
}  // namespace pie_cuda_driver
