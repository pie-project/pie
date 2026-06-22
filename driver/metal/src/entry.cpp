// pie_driver_metal — MLX/Metal backend library entry point (foundation
// skeleton).
//
// All meaningful logic lives in `run_impl`; the `extern "C"` wrappers at
// the bottom catch any escaping C++ exception so we never propagate across
// the FFI boundary (which would be UB). Mirrors the shape of
// driver/cuda/src/entry.cpp and driver/portable/src/entry.cpp.
//
// This is the seam beta/charlie/delta build on: the stub `InProcService`
// answers Health + Forward (dummy tokens) so the driver compiles, links,
// registers with `pie-worker`, and round-trips before the MLX compute
// layer, model graphs, and weight loader land.

#include "entry.hpp"

#include <cctype>
#include <csignal>
#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <memory>
#include <string>

#include <CLI/CLI.hpp>
#include <nlohmann/json.hpp>
#include <pie_ipc/inproc_server.hpp>

#include "config.hpp"
#include "driver_startup.hpp"
#include "service/inproc_service.hpp"

namespace {

// Model facts the caps handshake needs. Read best-effort from the
// snapshot's config.json; the stub falls back to neutral defaults when no
// snapshot is wired (e.g. the standalone self-check). The weight loader
// (delta) supersedes this with values pinned by the actual checkpoint.
struct ModelFacts {
    std::uint32_t vocab_size = 32000;
    std::uint32_t max_model_len = 8192;
    std::string arch_name = "llama";
};

ModelFacts read_model_facts(const std::string& hf_path) {
    ModelFacts facts;
    if (hf_path.empty()) return facts;
    const std::filesystem::path cfg =
        std::filesystem::path(hf_path) / "config.json";
    std::ifstream f(cfg);
    if (!f) return facts;
    try {
        nlohmann::json j;
        f >> j;
        if (j.contains("vocab_size") && j["vocab_size"].is_number_integer()) {
            facts.vocab_size = j["vocab_size"].get<std::uint32_t>();
        }
        if (j.contains("max_position_embeddings") &&
            j["max_position_embeddings"].is_number_integer()) {
            facts.max_model_len = j["max_position_embeddings"].get<std::uint32_t>();
        }
        if (j.contains("architectures") && j["architectures"].is_array() &&
            !j["architectures"].empty()) {
            std::string a = j["architectures"][0].get<std::string>();
            for (auto& c : a) c = static_cast<char>(std::tolower(c));
            const std::string suffix = "forcausallm";
            if (a.size() > suffix.size() &&
                a.compare(a.size() - suffix.size(), suffix.size(), suffix) == 0) {
                a.erase(a.size() - suffix.size());
            }
            if (!a.empty()) facts.arch_name = a;
        }
    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-metal] warning: failed to parse "
                  << cfg.string() << ": " << e.what() << "\n";
    }
    return facts;
}

std::string build_caps_json(const pie_metal_driver::Config& cfg,
                            const ModelFacts& facts) {
    const std::uint32_t total_pages = cfg.batching.total_pages;
    nlohmann::json caps = {
        {"total_pages", total_pages},
        {"kv_page_size", cfg.batching.kv_page_size},
        {"swap_pool_size", cfg.batching.cpu_pages},
        {"max_forward_tokens", cfg.batching.max_forward_tokens},
        {"max_forward_requests", cfg.batching.max_forward_requests},
        {"max_page_refs", total_pages},
        {"max_logit_rows", UINT32_MAX},
        {"max_prob_rows", UINT32_MAX},
        {"max_custom_mask_bytes", UINT32_MAX},
        {"max_sampler_rows", UINT32_MAX},
        {"max_logprob_labels", UINT32_MAX},
        {"arch_name", facts.arch_name},
        {"vocab_size", facts.vocab_size},
        {"max_model_len", facts.max_model_len},
        {"activation_dtype", "bf16"},
        {"snapshot_dir", cfg.model.hf_path},
    };
    return caps.dump();
}

// `vtable_opt` is non-null for the in-process serve loop; null for the
// standalone self-check (`pie_driver_metal_run`), which emits caps and
// returns without serving.
int run_impl(int argc,
             char** argv,
             int install_signal_handlers,
             pie_driver_metal_ready_cb ready_cb,
             void* ready_ctx,
             const pie_driver::PieInProcVTable* vtable_opt) {
    if (ready_cb == nullptr) {
        std::cerr << "[pie-driver-metal] fatal: ready_cb is null\n";
        return -1;
    }

    CLI::App app{"pie_driver_metal — MLX/Metal backend for Pie"};
    std::string config_path = "dev.toml";
    app.add_option("-c,--config", config_path, "Path to TOML config");
    app.allow_extras();
    try {
        app.parse(argc, argv);
    } catch (const CLI::ParseError& e) {
        return app.exit(e);
    }

    const pie_metal_driver::Config cfg =
        pie_metal_driver::load_config(config_path);
    const ModelFacts facts = read_model_facts(cfg.model.hf_path);

    const std::string caps = build_caps_json(cfg, facts);
    ready_cb(caps.c_str(), ready_ctx);

    if (vtable_opt == nullptr) {
        // Standalone self-check: caps emitted, nothing to serve.
        if (cfg.runtime.verbose) {
            std::cerr << "[pie-driver-metal] standalone self-check complete; "
                         "embed via pie_driver_metal_run_inproc to serve\n";
        }
        return 0;
    }

    auto server = std::make_unique<pie_driver::InProcServer>(*vtable_opt);
    pie_metal_driver::register_server(server.get());

    if (install_signal_handlers) {
        std::signal(SIGINT, pie_metal_driver::on_signal);
        std::signal(SIGTERM, pie_metal_driver::on_signal);
    }

    pie_metal_driver::service::InProcService service(facts.vocab_size);
    std::cerr << "[pie-driver-metal] serving (arch=" << facts.arch_name
              << ", vocab=" << facts.vocab_size << ", stub forward)\n";
    service.serve_forever(*server);

    pie_metal_driver::unregister_server(server.get());
    std::cerr << "[pie-driver-metal] shutting down (handled "
              << service.handled() << " forward requests)\n";
    return 0;
}

}  // namespace

extern "C" int pie_driver_metal_run(int argc,
                                    char** argv,
                                    int install_signal_handlers,
                                    pie_driver_metal_ready_cb ready_cb,
                                    void* ready_ctx) {
    try {
        return run_impl(argc, argv, install_signal_handlers, ready_cb, ready_ctx,
                        /*vtable_opt=*/nullptr);
    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-metal] fatal: " << e.what() << "\n";
        return -1;
    } catch (...) {
        std::cerr << "[pie-driver-metal] fatal: unknown exception\n";
        return -1;
    }
}

extern "C" int pie_driver_metal_run_inproc(int argc,
                                           char** argv,
                                           int install_signal_handlers,
                                           pie_driver_metal_ready_cb ready_cb,
                                           void* ready_ctx,
                                           pie_driver::PieInProcVTable vtable) {
    try {
        return run_impl(argc, argv, install_signal_handlers, ready_cb, ready_ctx,
                        &vtable);
    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-metal] fatal: " << e.what() << "\n";
        return -1;
    } catch (...) {
        std::cerr << "[pie-driver-metal] fatal: unknown exception\n";
        return -1;
    }
}

extern "C" void pie_driver_metal_request_stop(void) {
    pie_metal_driver::stop_servers();
}
