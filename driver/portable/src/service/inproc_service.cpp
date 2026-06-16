#include "service/inproc_service.hpp"

#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <ggml.h>
#include <ggml-backend.h>
#include <pie_bridge/inproc_server.hpp>

#include "adapter.hpp"
#include "executor/executor.hpp"
#include "host_swap_pool.hpp"
#include "kv_cache.hpp"
#include "model.hpp"

namespace pie_portable_driver::service {

namespace {

std::size_t page_bytes_of(KvCachePaged& kv) {
    return static_cast<std::size_t>(kv.n_embd_gqa()) * kv.page_size()
           * ggml_type_size(kv.k(0)->type);
}

}  // namespace

InProcService::InProcService(Executor& executor,
                             Model& model,
                             HostSwapPool* swap_pool,
                             AdapterPool& adapters,
                             bool verbose)
    : executor_(executor),
      model_(model),
      swap_pool_(swap_pool),
      adapters_(adapters),
      verbose_(verbose) {}

void InProcService::serve_forever(pie_driver::InProcServer& server) {
    server.serve_forever(
        [&](std::uint32_t req_id,
            const pie_driver::PieInProcRequestView& req,
            pie_driver::PieInProcResponseView& out) {
            out.method = req.method;
            switch (req.method) {
                case pie_driver::PIE_METHOD_FORWARD: {
                    ++handled_;
                    const auto& view = req.forward;
                    if (verbose_ && (handled_ <= 4 || handled_ % 100 == 0)) {
                        const auto tokens = view.token_ids.as<std::uint32_t>();
                        const auto context_ids =
                            view.context_ids.as<std::uint64_t>();
                        std::cerr << "[pie-driver-portable] req_id="
                                  << req_id
                                  << " device=" << view.driver_id
                                  << " single_token="
                                  << (view.single_token_mode ? 1 : 0)
                                  << " tokens=" << tokens.size()
                                  << " contexts=" << context_ids.size()
                                  << "\n";
                    }
                    try {
                        executor_.run(view, response_builder_, out.forward);
                        out.status = 0;
                    } catch (const std::exception& e) {
                        std::cerr << "[pie-driver-portable] forward failed for req_id="
                                  << req_id << ": " << e.what() << "\n";
                        out.forward = pie_driver::PieForwardResponseView{};
                        out.status = 5;
                    }
                    break;
                }
                case pie_driver::PIE_METHOD_COPY_D2H:
                case pie_driver::PIE_METHOD_COPY_H2D:
                case pie_driver::PIE_METHOD_COPY_D2D:
                case pie_driver::PIE_METHOD_COPY_H2H: {
                    const auto srcs = req.copy_srcs.as<std::uint32_t>();
                    const auto dsts = req.copy_dsts.as<std::uint32_t>();
                    if (srcs.size() != dsts.size()) {
                        out.status = 5;
                        break;
                    }
                    auto& kv = executor_.kv();
                    const std::size_t per_page = page_bytes_of(kv);
                    const int total_dev = kv.total_pages();
                    const int total_host =
                        swap_pool_ ? swap_pool_->cpu_pages() : 0;
                    bool ok = true;
                    try {
                        for (std::size_t i = 0; i < srcs.size(); ++i) {
                            const std::uint32_t s = srcs[i];
                            const std::uint32_t d = dsts[i];
                            if (req.method == pie_driver::PIE_METHOD_COPY_D2H) {
                                if (!swap_pool_) {
                                    out.status = 4;
                                    ok = false;
                                    break;
                                }
                                if (s >= static_cast<std::uint32_t>(total_dev) ||
                                    d >= static_cast<std::uint32_t>(total_host)) {
                                    out.status = 3;
                                    ok = false;
                                    break;
                                }
                                const std::size_t off =
                                    static_cast<std::size_t>(s) * per_page;
                                for (std::int32_t il = 0; il < kv.n_layers();
                                     ++il) {
                                    ggml_backend_tensor_get(
                                        kv.k(il), swap_pool_->k_slot(il, d),
                                        off, per_page);
                                    ggml_backend_tensor_get(
                                        kv.v(il), swap_pool_->v_slot(il, d),
                                        off, per_page);
                                }
                            } else if (req.method ==
                                       pie_driver::PIE_METHOD_COPY_H2D) {
                                if (!swap_pool_) {
                                    out.status = 4;
                                    ok = false;
                                    break;
                                }
                                if (s >= static_cast<std::uint32_t>(total_host) ||
                                    d >= static_cast<std::uint32_t>(total_dev)) {
                                    out.status = 3;
                                    ok = false;
                                    break;
                                }
                                const std::size_t off =
                                    static_cast<std::size_t>(d) * per_page;
                                for (std::int32_t il = 0; il < kv.n_layers();
                                     ++il) {
                                    ggml_backend_tensor_set(
                                        kv.k(il), swap_pool_->k_slot(il, s),
                                        off, per_page);
                                    ggml_backend_tensor_set(
                                        kv.v(il), swap_pool_->v_slot(il, s),
                                        off, per_page);
                                }
                            } else if (req.method ==
                                       pie_driver::PIE_METHOD_COPY_D2D) {
                                if (s >= static_cast<std::uint32_t>(total_dev) ||
                                    d >= static_cast<std::uint32_t>(total_dev)) {
                                    out.status = 3;
                                    ok = false;
                                    break;
                                }
                                std::vector<std::uint8_t> tmp(per_page);
                                const std::size_t soff =
                                    static_cast<std::size_t>(s) * per_page;
                                const std::size_t doff =
                                    static_cast<std::size_t>(d) * per_page;
                                for (std::int32_t il = 0; il < kv.n_layers();
                                     ++il) {
                                    ggml_backend_tensor_get(
                                        kv.k(il), tmp.data(), soff, per_page);
                                    ggml_backend_tensor_set(
                                        kv.k(il), tmp.data(), doff, per_page);
                                    ggml_backend_tensor_get(
                                        kv.v(il), tmp.data(), soff, per_page);
                                    ggml_backend_tensor_set(
                                        kv.v(il), tmp.data(), doff, per_page);
                                }
                            } else {
                                if (!swap_pool_) {
                                    out.status = 4;
                                    ok = false;
                                    break;
                                }
                                if (s >= static_cast<std::uint32_t>(total_host) ||
                                    d >= static_cast<std::uint32_t>(total_host)) {
                                    out.status = 3;
                                    ok = false;
                                    break;
                                }
                                for (std::int32_t il = 0; il < kv.n_layers();
                                     ++il) {
                                    std::memcpy(
                                        swap_pool_->k_slot(il, d),
                                        swap_pool_->k_slot(il, s), per_page);
                                    std::memcpy(
                                        swap_pool_->v_slot(il, d),
                                        swap_pool_->v_slot(il, s), per_page);
                                }
                            }
                        }
                        if (ok) out.status = 0;
                    } catch (const std::exception& e) {
                        std::cerr << "[pie-driver-portable] copy failed: "
                                  << e.what() << "\n";
                        out.status = 5;
                    }
                    break;
                }
                case pie_driver::PIE_METHOD_RS_COPY_D2D: {
                    const auto srcs = req.copy_srcs.as<std::uint32_t>();
                    const auto dsts = req.copy_dsts.as<std::uint32_t>();
                    if (srcs.size() != dsts.size()) {
                        out.status = 5;
                        break;
                    }
                    auto* state = executor_.state_cache();
                    if (state == nullptr) {
                        out.status = 4;
                        break;
                    }
                    try {
                        for (std::size_t i = 0; i < srcs.size(); ++i) {
                            state->copy_slot(
                                static_cast<std::int32_t>(srcs[i]),
                                static_cast<std::int32_t>(dsts[i]));
                        }
                        out.status = 0;
                    } catch (const std::exception& e) {
                        std::cerr << "[pie-driver-portable] rs copy failed: "
                                  << e.what() << "\n";
                        out.status = 5;
                    }
                    break;
                }
                case pie_driver::PIE_METHOD_RS_COPY_D2H:
                case pie_driver::PIE_METHOD_RS_COPY_H2D:
                case pie_driver::PIE_METHOD_RS_COPY_H2H:
                    out.status = 4;
                    break;
                case pie_driver::PIE_METHOD_LOAD_ADAPTER: {
                    const auto path_bytes = req.adapter_path.as<char>();
                    std::string path(path_bytes.data(), path_bytes.size());
                    try {
                        const auto& hpar = model_.hparams();
                        auto adapter = std::make_unique<Adapter>(
                            model_.backend(),
                            hpar.num_hidden_layers,
                            /*guessed_rank=*/0,
                            /*scale=*/1.0f,
                            std::filesystem::path(path),
                            hpar);
                        adapters_.insert(req.adapter_id, std::move(adapter));
                        out.status = 0;
                    } catch (const std::exception& e) {
                        std::cerr << "[pie-driver-portable] load_adapter: "
                                  << e.what() << "\n";
                        out.status = 5;
                    }
                    break;
                }
                case pie_driver::PIE_METHOD_SAVE_ADAPTER:
                case pie_driver::PIE_METHOD_ZO_INITIALIZE_ADAPTER:
                case pie_driver::PIE_METHOD_ZO_UPDATE_ADAPTER:
                    // No-op stubs: adapter persistence and zeroth-order
                    // training are not implemented in portable.
                    out.status = 0;
                    break;
                default:
                    std::cerr << "[pie-driver-portable] unknown method "
                              << req.method << "\n";
                    out.status = 2;
                    break;
            }
        });
}

}  // namespace pie_portable_driver::service
