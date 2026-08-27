#include "service/inproc_service.hpp"

#include <cstdint>
#include <cstring>
#include <exception>
#include <filesystem>
#include <iostream>
#include <memory>
#include <string>
#include <vector>

#include <fstream>

#include <ggml.h>
#include <ggml-backend.h>
#include <nlohmann/json.hpp>
#include <pie_bridge/inproc_server.hpp>

#include "adapter.hpp"
#include "executor/executor.hpp"
#include "graph_csm_gen.hpp"
#include "host_swap_pool.hpp"
#include "kv_cache.hpp"
#include "model.hpp"

namespace pie_portable_driver::service {

namespace {

std::size_t page_bytes_of(KvCachePaged& kv) {
    return static_cast<std::size_t>(kv.n_embd_gqa()) * kv.page_size()
           * ggml_type_size(kv.k(0)->type);
}

// Returns true if the backend for tensor `t` does not support partial reads
// via ggml_backend_tensor_get at non-zero offsets. On ggml-Vulkan, such reads
// silently return zeros or garbage, corrupting KV pages during ctx.fork().
bool backend_needs_full_tensor_staging(ggml_tensor* t) {
    ggml_backend_buffer_t buf = t->buffer;
    if (!buf) return false;
    ggml_backend_buffer_type_t buft = ggml_backend_buffer_get_type(buf);
    if (!buft) return false;
    const char* name = ggml_backend_buft_name(buft);
    if (!name) return false;
    // ggml-Vulkan buffer type names contain "Vulkan" (e.g. "Vulkan0").
    return std::string(name).find("Vulkan") != std::string::npos;
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
      verbose_(verbose) {
    // Detect backends that don't support ggml_backend_tensor_get at non-zero
    // offsets. ggml-Vulkan silently returns zeros for such reads, corrupting
    // KV pages during ctx.fork() (CopyD2D / CopyD2H). Check layer 0's K
    // tensor — all layers share the same backend.
    auto& kv = executor_.kv();
    needs_full_tensor_staging_ = (kv.n_layers() > 0)
                                 && backend_needs_full_tensor_staging(kv.k(0));
    if (needs_full_tensor_staging_) {
        std::cerr << "[pie-driver-portable] Vulkan backend detected: "
                     "using full-tensor staging for KV page copies "
                     "(workaround for ggml-Vulkan partial-read bug, issue #418)\n";
    }
}

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
                                    if (needs_full_tensor_staging_) {
                                        // ggml-Vulkan silently returns zeros
                                        // for tensor reads at non-zero offsets.
                                        // Read the entire tensor and slice out
                                        // the page we need. (issue #418)
                                        const std::size_t total_k = ggml_nbytes(kv.k(il));
                                        const std::size_t total_v = ggml_nbytes(kv.v(il));
                                        std::vector<std::uint8_t> staging(
                                            std::max(total_k, total_v));
                                        ggml_backend_tensor_get(
                                            kv.k(il), staging.data(), 0, total_k);
                                        std::memcpy(swap_pool_->k_slot(il, d),
                                                    staging.data() + off, per_page);
                                        ggml_backend_tensor_get(
                                            kv.v(il), staging.data(), 0, total_v);
                                        std::memcpy(swap_pool_->v_slot(il, d),
                                                    staging.data() + off, per_page);
                                    } else {
                                        ggml_backend_tensor_get(
                                            kv.k(il), swap_pool_->k_slot(il, d),
                                            off, per_page);
                                        ggml_backend_tensor_get(
                                            kv.v(il), swap_pool_->v_slot(il, d),
                                            off, per_page);
                                    }
                                }
                            } else if (req.method ==
                                       pie_driver::PIE_METHOD_COPY_H2D) {
                                // tensor_set with non-zero offset works
                                // correctly on all backends including Vulkan.
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
                                // Device-to-device KV page copy, used by
                                // ctx.fork(). On ggml-Vulkan,
                                // ggml_backend_tensor_get with a non-zero
                                // offset silently returns zeros (issue #418).
                                // When needs_full_tensor_staging_ is set, read
                                // the entire tensor into a staging buffer first
                                // and slice out the page we need.
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
                                    if (needs_full_tensor_staging_) {
                                        const std::size_t total_k = ggml_nbytes(kv.k(il));
                                        const std::size_t total_v = ggml_nbytes(kv.v(il));
                                        std::vector<std::uint8_t> staging(
                                            std::max(total_k, total_v));
                                        ggml_backend_tensor_get(
                                            kv.k(il), staging.data(), 0, total_k);
                                        std::memcpy(tmp.data(),
                                                    staging.data() + soff, per_page);
                                        ggml_backend_tensor_set(
                                            kv.k(il), tmp.data(), doff, per_page);
                                        ggml_backend_tensor_get(
                                            kv.v(il), staging.data(), 0, total_v);
                                        std::memcpy(tmp.data(),
                                                    staging.data() + soff, per_page);
                                        ggml_backend_tensor_set(
                                            kv.v(il), tmp.data(), doff, per_page);
                                    } else {
                                        ggml_backend_tensor_get(
                                            kv.k(il), tmp.data(), soff, per_page);
                                        ggml_backend_tensor_set(
                                            kv.k(il), tmp.data(), doff, per_page);
                                        ggml_backend_tensor_get(
                                            kv.v(il), tmp.data(), soff, per_page);
                                        ggml_backend_tensor_set(
                                            kv.v(il), tmp.data(), doff, per_page);
                                    }
                                }
                            } else {
                                // CopyH2H
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
                case pie_driver::PIE_METHOD_GENERATE_AUDIO: {
                    try {
                        if (!model_.weights().csm.present) {
                            std::cerr << "[pie-driver-portable] generate_audio: "
                                         "model is not CSM (no audio output)\n";
                            out.status = -1;
                            break;
                        }
                        const auto bytes = req.adapter_path.as<char>();
                        auto j = nlohmann::json::parse(
                            std::string(bytes.data(), bytes.size()));
                        std::vector<std::int32_t> prompt;
                        for (const auto& t : j.at("prompt")) {
                            prompt.push_back(static_cast<std::int32_t>(
                                t.get<std::int64_t>()));
                        }
                        const int max_frames = j.value("max_frames", 256);
                        const std::string out_path =
                            j.at("out_path").get<std::string>();

                        std::vector<float> pcm;
                        const int n_frames = csm_generate_audio(
                            model_, prompt.data(),
                            static_cast<int>(prompt.size()), max_frames, pcm,
                            nullptr);

                        std::ofstream f(out_path,
                                        std::ios::binary | std::ios::trunc);
                        if (!f) {
                            std::cerr << "[pie-driver-portable] generate_audio: "
                                         "cannot open out_path '" << out_path
                                      << "'\n";
                            out.status = -1;
                            break;
                        }
                        f.write(reinterpret_cast<const char*>(pcm.data()),
                                static_cast<std::streamsize>(pcm.size() *
                                                             sizeof(float)));
                        f.close();
                        out.status = n_frames;
                    } catch (const std::exception& e) {
                        std::cerr << "[pie-driver-portable] generate_audio: "
                                  << e.what() << "\n";
                        out.status = -1;
                    }
                    break;
                }
                default:
                    std::cerr << "[pie-driver-portable] unknown method "
                              << req.method << "\n";
                    out.status = 2;
                    break;
            }
        });
}

}  // namespace pie_portable_driver::service
