#include "service/inproc_service.hpp"

#include <cstdint>
#include <exception>
#include <iostream>

#include <pie_bridge/inproc_server.hpp>

#include "executor/executor.hpp"
#include "kv_cache.hpp"
#include "swap_pool.hpp"

namespace pie_cuda_driver::service {

InProcService::InProcService(Executor& executor,
                             KvCache& kv_cache,
                             SwapPool& swap_pool)
    : executor_(executor), kv_cache_(kv_cache), swap_pool_(swap_pool) {}

void InProcService::serve_forever(pie_driver::InProcServer& server) {
    server.serve_forever(
        [&](std::uint32_t req_id,
            const pie_driver::PieInProcRequestView& req,
            pie_driver::PieInProcResponseView& out) {
            switch (req.method) {
                case pie_driver::PIE_METHOD_FORWARD: {
                    ++handled_;
                    handle_fire_batch(
                        req_id, req.forward, out.forward, executor_, handled_);
                    out.status = 0;
                    break;
                }
                case pie_driver::PIE_METHOD_COPY_D2H: {
                    try {
                        swap_pool_.copy_d2h(
                            kv_cache_,
                            req.copy_srcs.as<std::uint32_t>(),
                            req.copy_dsts.as<std::uint32_t>());
                        out.status = 0;
                    } catch (const std::exception& e) {
                        std::cerr << "[pie-driver-cuda] copy_d2h: "
                                  << e.what() << "\n";
                        out.status = 5;
                    }
                    break;
                }
                case pie_driver::PIE_METHOD_COPY_H2D: {
                    try {
                        swap_pool_.copy_h2d(
                            kv_cache_,
                            req.copy_srcs.as<std::uint32_t>(),
                            req.copy_dsts.as<std::uint32_t>());
                        out.status = 0;
                    } catch (const std::exception& e) {
                        std::cerr << "[pie-driver-cuda] copy_h2d: "
                                  << e.what() << "\n";
                        out.status = 5;
                    }
                    break;
                }
                case pie_driver::PIE_METHOD_COPY_D2D: {
                    try {
                        swap_pool_.copy_d2d(
                            kv_cache_,
                            req.copy_srcs.as<std::uint32_t>(),
                            req.copy_dsts.as<std::uint32_t>());
                        out.status = 0;
                    } catch (const std::exception& e) {
                        std::cerr << "[pie-driver-cuda] copy_d2d: "
                                  << e.what() << "\n";
                        out.status = 5;
                    }
                    break;
                }
                case pie_driver::PIE_METHOD_COPY_H2H: {
                    try {
                        swap_pool_.copy_h2h(
                            req.copy_srcs.as<std::uint32_t>(),
                            req.copy_dsts.as<std::uint32_t>());
                        out.status = 0;
                    } catch (const std::exception& e) {
                        std::cerr << "[pie-driver-cuda] copy_h2h: "
                                  << e.what() << "\n";
                        out.status = 5;
                    }
                    break;
                }
                case pie_driver::PIE_METHOD_LOAD_ADAPTER:
                case pie_driver::PIE_METHOD_SAVE_ADAPTER:
                case pie_driver::PIE_METHOD_ZO_INITIALIZE_ADAPTER:
                case pie_driver::PIE_METHOD_ZO_UPDATE_ADAPTER:
                    // Cuda has no AdapterPool yet. Adapter methods are
                    // no-op stubs returning success, matching the previous
                    // entry.cpp dispatch behavior.
                    out.status = 0;
                    break;
                default:
                    std::cerr << "[pie-driver-cuda] unknown method "
                              << req.method << "\n";
                    out.status = 2;
                    break;
            }
        });
}

}  // namespace pie_cuda_driver::service
