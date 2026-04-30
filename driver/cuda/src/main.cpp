// pie_driver_cuda — native CUDA backend, sibling to ../../runtime (Rust)
// and ../../pie (Python driver). Consumed by `pie_driver_cuda_native`.
//
// Currently a scaffold: opens the shmem fast path, decodes incoming
// `BatchedForwardPassRequest`s, logs them, and replies with an empty payload.
// Model loading + flashinfer-backed forward pass are the next milestones.

#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <span>
#include <string>

#include <CLI/CLI.hpp>

#include "config.hpp"
#include "shmem_ipc.hpp"
#include "shmem_schema.hpp"

namespace {

std::atomic<pie_cuda_driver::ShmemServer*> g_server{nullptr};

void on_signal(int) {
    if (auto* s = g_server.load()) s->stop();
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"pie_driver_cuda — native CUDA backend for Pie"};
    std::string config_path = "dev.toml";
    app.add_option("-c,--config", config_path, "Path to TOML config")
        ->check(CLI::ExistingFile);
    CLI11_PARSE(app, argc, argv);

    const auto cfg = pie_cuda_driver::load_config(config_path);

    std::cout << "[pie-driver-cuda] config loaded\n"
              << "  shmem.name      = " << cfg.shmem.name << "\n"
              << "  shmem.num_slots = " << cfg.shmem.num_slots << "\n"
              << "  model.hf_repo   = " << cfg.model.hf_repo << "\n"
              << "  model.device    = " << cfg.model.device << "\n"
              << "  model.dtype     = " << cfg.model.dtype << "\n";

    pie_cuda_driver::ShmemServer server(
        cfg.shmem.name,
        cfg.shmem.num_slots,
        cfg.shmem.req_buf,
        cfg.shmem.resp_buf,
        cfg.shmem.spin_us);
    g_server.store(&server);

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    std::cout << "[pie-driver-cuda] serving on shmem " << server.name()
              << " (" << server.num_slots() << " slots, "
              << "req_buf=" << server.req_buf_size() << ", "
              << "resp_buf=" << server.resp_buf_size() << ")\n";

    std::uint64_t handled = 0;

    server.serve_forever([&](const pie_cuda_driver::SlotRequest& req,
                             std::span<std::uint8_t> response) -> std::size_t {
        ++handled;
        if (req.method_tag != pie_cuda_driver::METHOD_TAG_FIRE_BATCH) {
            std::cerr << "[pie-driver-cuda] unsupported method_tag="
                      << req.method_tag << " req_id=" << req.req_id << "\n";
            return 0;
        }

        try {
            const auto decoded = pie_cuda_driver::schema::decode_request(req.payload);
            const auto tokens =
                decoded.as<std::uint32_t>(pie_cuda_driver::schema::A_TOKEN_IDS);
            const auto context_ids =
                decoded.as<std::uint64_t>(pie_cuda_driver::schema::A_CONTEXT_IDS);

            if (handled <= 4 || handled % 100 == 0) {
                std::cout << "[pie-driver-cuda] req_id=" << req.req_id
                          << " device=" << decoded.device_id
                          << " single_token=" << decoded.single_token_mode
                          << " tokens=" << tokens.size()
                          << " contexts=" << context_ids.size() << "\n";
            }
        } catch (const std::exception& e) {
            std::cerr << "[pie-driver-cuda] decode failed for req_id=" << req.req_id
                      << ": " << e.what() << "\n";
            return 0;
        }

        // TODO(pie-cpp): run actual forward pass; emit sampled tokens.
        (void)response;
        return 0;
    });

    g_server.store(nullptr);
    std::cout << "[pie-driver-cuda] shutting down (handled " << handled
              << " requests)\n";
    return 0;
}
