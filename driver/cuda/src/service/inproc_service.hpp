#pragma once

#include <cstdint>

namespace pie_driver {
class InProcServer;
}  // namespace pie_driver

namespace pie_cuda_driver {

struct Executor;
class KvCache;
class SwapPool;

namespace model {
class CsmModel;
}  // namespace model

namespace service {

class InProcService {
public:
    // `csm_model` is non-null only for the CSM native-audio-output arch; it
    // backs the GenerateAudio method (pie:core/audio-out). nullptr for every
    // text/multimodal arch — the GenerateAudio path then returns an error.
    InProcService(Executor& executor, KvCache& kv_cache, SwapPool& swap_pool,
                  model::CsmModel* csm_model = nullptr);

    void serve_forever(pie_driver::InProcServer& server);

    std::uint64_t handled() const noexcept { return handled_; }

private:
    Executor& executor_;
    KvCache& kv_cache_;
    SwapPool& swap_pool_;
    model::CsmModel* csm_model_ = nullptr;
    std::uint64_t handled_ = 0;
};

}  // namespace service
}  // namespace pie_cuda_driver
