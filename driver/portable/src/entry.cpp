// pie_driver_portable — library entry point implementation.
//
// Body of what used to be `main()`, lifted into a C-callable function so
// it can be invoked from either the standalone executable shim
// (driver/portable/src/main.cpp) or a host process that links
// `pie_driver_portable_lib` directly (e.g. server/standalone).

#include "entry.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <span>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>
#include <ggml.h>
#include <ggml-backend.h>
#include <nlohmann/json.hpp>

#include "adapter.hpp"
#include "host_swap_pool.hpp"
#include "config.hpp"
#include "executor/executor.hpp"
#include "hf_config.hpp"
#include <pie_bridge/inproc_server.hpp>
#include "kv_cache.hpp"
#include "model.hpp"
#include "service/inproc_service.hpp"

namespace {

std::mutex g_servers_mu;
std::vector<pie_driver::InProcServer*> g_servers;
std::atomic<pie_driver::InProcServer*> g_signal_server{nullptr};

void register_server(pie_driver::InProcServer* server) {
    std::lock_guard<std::mutex> lk(g_servers_mu);
    g_servers.push_back(server);
    g_signal_server.store(server);
}

void unregister_server(pie_driver::InProcServer* server) {
    std::lock_guard<std::mutex> lk(g_servers_mu);
    g_servers.erase(
        std::remove(g_servers.begin(), g_servers.end(), server),
        g_servers.end());
    if (g_signal_server.load() == server) {
        g_signal_server.store(g_servers.empty() ? nullptr : g_servers.back());
    }
}

void stop_servers() {
    std::vector<pie_driver::InProcServer*> servers;
    {
        std::lock_guard<std::mutex> lk(g_servers_mu);
        servers = g_servers;
    }
    for (auto* server : servers) {
        if (server != nullptr) server->stop();
    }
}

void on_signal(int) {
    if (auto* server = g_signal_server.load()) server->stop();
}

constexpr std::int32_t kMinTotalPages = 64;
constexpr std::int32_t kDefaultTotalPages = 1024;
constexpr std::int32_t kDefaultMaxForwardTokens = 10240;
constexpr std::int32_t kDefaultMaxForwardRequests = 512;

std::int32_t pages_for_tokens(std::int32_t tokens, std::int32_t page_size) {
    if (tokens <= 0) return 0;
    return (tokens + page_size - 1) / page_size;
}

std::int32_t derive_total_pages(const pie_portable_driver::Hparams& h,
                                std::int32_t page_size) {
    const auto model_pages = pages_for_tokens(h.max_position_embeddings, page_size);
    return std::max(kMinTotalPages, std::max(kDefaultTotalPages, model_pages));
}

std::int32_t derive_max_forward_tokens(std::int32_t total_pages,
                                       std::int32_t page_size) {
    const auto total_tokens = std::max<std::int32_t>(1, total_pages * page_size);
    return std::max<std::int32_t>(
        1, std::min(kDefaultMaxForwardTokens, total_tokens));
}

std::int32_t derive_max_forward_requests(std::int32_t max_forward_tokens) {
    return std::max<std::int32_t>(
        1, std::min(kDefaultMaxForwardRequests, max_forward_tokens));
}

// -----------------------------------------------------------------------------
// Offline-test scaffolding (--test-prompt-tokens and friends)
// -----------------------------------------------------------------------------

std::vector<std::uint32_t> parse_csv_u32(const std::string& s) {
    std::vector<std::uint32_t> v;
    std::string item;
    for (char c : s) {
        if (c == ',') {
            if (!item.empty()) v.push_back(std::stoul(item));
            item.clear();
        } else if (!std::isspace(static_cast<unsigned char>(c))) {
            item.push_back(c);
        }
    }
    if (!item.empty()) v.push_back(std::stoul(item));
    return v;
}

// "a,b|c,d|e" → {{a,b},{c,d},{e}}
std::vector<std::vector<std::uint32_t>> parse_pipe_csv_u32(const std::string& s) {
    std::vector<std::vector<std::uint32_t>> out;
    std::string chunk;
    for (char c : s) {
        if (c == '|') {
            if (!chunk.empty()) out.push_back(parse_csv_u32(chunk));
            chunk.clear();
        } else {
            chunk.push_back(c);
        }
    }
    if (!chunk.empty()) out.push_back(parse_csv_u32(chunk));
    return out;
}

// 1, 2, 3, ... up to count (test-only synthetic context ids).
std::vector<std::uint64_t> seq_ctx_ids(std::size_t count) {
    std::vector<std::uint64_t> v(count);
    for (std::size_t i = 0; i < count; ++i) v[i] = i + 1;
    return v;
}

void quiet_ggml_log(enum ggml_log_level level, const char* text, void*) {
    (void)level;
    (void)text;
}

enum class OfflineTestMode {
    Single,     // one prompt, print all tokens, report per-context tok/s
    Multi,      // N prompts, print all, report wall-time only
    Replicate,  // N copies of one prompt, print [0] only, report aggregate tok/s
};

// Drives generate_multi() with the supplied prompts + ctx_ids and prints
// outputs + timing per the test mode. After the run, the executor's
// per-stage timing breakdown is logged (covers the prefill + decode loop
// across all contexts).
void run_offline_test(pie_portable_driver::Executor& executor,
                      std::vector<std::vector<std::uint32_t>> prompts,
                      std::vector<std::uint64_t> ctx_ids,
                      int max_new,
                      OfflineTestMode mode) {
    using clock = std::chrono::steady_clock;
    const std::size_t n = prompts.size();

    switch (mode) {
        case OfflineTestMode::Single:
            std::cerr << "[pie-driver-portable] self-test: prompt="
                      << prompts[0].size() << " tokens, max_new=" << max_new << "\n";
            break;
        case OfflineTestMode::Multi:
            std::cerr << "[pie-driver-portable] multi-context self-test: "
                      << n << " contexts, max_new=" << max_new << "\n";
            break;
        case OfflineTestMode::Replicate:
            std::cerr << "[pie-driver-portable] throughput test: " << n
                      << " concurrent contexts × " << max_new
                      << " tokens (prompt=" << prompts[0].size()
                      << " tokens each)\n";
            break;
    }

    executor.reset_timings();
    const auto t0 = clock::now();
    auto outs = executor.generate_multi(prompts, max_new, ctx_ids);
    const long dt_ms = static_cast<long>(
        std::chrono::duration_cast<std::chrono::milliseconds>(
            clock::now() - t0).count());

    if (mode == OfflineTestMode::Replicate) {
        std::cout << "GENERATED[0]";
        for (auto t : outs[0]) std::cout << ' ' << t;
        std::cout << "\n";
    } else if (mode == OfflineTestMode::Single) {
        std::cout << "GENERATED";
        for (auto t : outs[0]) std::cout << ' ' << t;
        std::cout << "\n";
    } else {
        for (std::size_t i = 0; i < outs.size(); ++i) {
            std::cout << "GENERATED[" << i << "]";
            for (auto t : outs[i]) std::cout << ' ' << t;
            std::cout << "\n";
        }
    }

    const long denom = std::max<long>(1, dt_ms);
    switch (mode) {
        case OfflineTestMode::Single: {
            std::cerr << "[pie-driver-portable] generated " << outs[0].size()
                      << " tokens in " << dt_ms << " ms ("
                      << (outs[0].size() * 1000.0 / denom) << " tok/s)\n";
            break;
        }
        case OfflineTestMode::Multi: {
            const std::size_t per = outs.empty() ? 0 : outs[0].size();
            std::cerr << "[pie-driver-portable] generated " << per
                      << " tokens × " << outs.size()
                      << " contexts in " << dt_ms << " ms\n";
            break;
        }
        case OfflineTestMode::Replicate: {
            const long total = static_cast<long>(n) * max_new;
            std::cerr << "[pie-driver-portable] generated " << total
                      << " tokens (" << n << " × " << max_new
                      << ") in " << dt_ms << " ms ("
                      << (total * 1000.0 / denom) << " tok/s aggregate)\n";
            break;
        }
    }
    // Per-stage breakdown (covers every compute_() call across prefill +
    // decode). Useful for spotting whether time is going into graph
    // build, the GPU compute itself, the logits download, or sampling.
    executor.log_timings("offline-test");
}

// All meaningful logic lives in `run_impl`; the `extern "C"` wrapper at
// the bottom catches any escaping C++ exception so we never propagate
// across the FFI boundary (which would be UB).
int run_impl(int argc,
             char** argv,
             int install_signal_handlers,
             pie_driver_portable_ready_cb ready_cb,
             void* ready_ctx,
             pie_driver::PieInProcVTable vtable) {
    if (ready_cb == nullptr) {
        std::cerr << "[pie-driver-portable] fatal: ready_cb is null\n";
        return -1;
    }
    CLI::App app{"pie_driver_portable — portable ggml backend for Pie"};
    std::string config_path = "dev.toml";
    std::string test_prompt_tokens;  // comma-separated u32s
    int         test_max_new = 16;
    app.add_option("-c,--config", config_path, "Path to TOML config")
        ->check(CLI::ExistingFile);
    std::string test_multi_prompts;
    int         test_replicate = 0;
    app.add_option("--test-prompt-tokens", test_prompt_tokens,
                   "Offline self-test: comma-separated prompt token IDs. "
                   "When set, skip shmem and just print the greedy generation.");
    app.add_option("--test-multi-prompt-tokens", test_multi_prompts,
                   "Offline self-test: pipe-separated prompts, each comma-"
                   "separated token IDs. Runs each prompt as a concurrent "
                   "context to exercise the multi-stream KV path.");
    app.add_option("--test-replicate", test_replicate,
                   "Offline throughput test: replicate --test-prompt-tokens "
                   "N times as concurrent contexts. Reports tok/s.");
    app.add_option("--test-max-new", test_max_new,
                   "Number of tokens to generate in --test-* modes (default 16)");
    CLI11_PARSE(app, argc, argv);

    const auto cfg = pie_portable_driver::load_config(config_path);
    if (!cfg.runtime.verbose) {
        ggml_log_set(quiet_ggml_log, nullptr);
    }

    // Informational logs go to stderr — stdout is reserved for the READY
    // handshake line consumed by the host process.
    if (cfg.runtime.verbose) {
        std::cerr << "[pie-driver-portable] config loaded\n"
                  << "  model.hf_path     = " << cfg.model.hf_path << "\n";
    }

    // ---- Backend discovery. Registers all backends compiled into ggml
    // (CPU + CUDA / Metal / Vulkan when those `GGML_*=ON` flags were set
    // at build time). After this, `ggml_backend_init_best()` will pick
    // GPU over CPU.
    ggml_backend_load_all();

    // ---- Load the model. -----------------------------------------------------
    if (cfg.runtime.verbose) {
        std::cerr << "[pie-driver-portable] loading model from " << cfg.model.hf_path
                  << " (backend=" << cfg.model.backend << ")\n";
    }
    pie_portable_driver::Model model(
        cfg.model.hf_path, cfg.model.backend, cfg.runtime.verbose);
    const auto& h = model.hparams();
    if (cfg.runtime.verbose) {
        std::cerr << "[pie-driver-portable] loaded "
                  << model.arch_name_pie() << " ("
                  << h.num_hidden_layers << " layers, hidden=" << h.hidden_size
                  << ", heads=" << h.num_attention_heads << "/"
                  << h.num_key_value_heads << " kv, head_dim=" << h.head_dim
                  << ", vocab=" << h.vocab_size
                  << ", dtype=" << model.activation_dtype_str()
                  << ", buf=" << (model.buffer_size() / (1024.0 * 1024.0)) << " MiB"
                  << ", backend=" << model.backend_name() << ")\n";
    }

    // ---- Allocate forward executor + paged KV pool. ---------------------------
    // The runtime owns page allocation; we report total_pages and page_size
    // in the READY handshake and honor the page IDs the runtime sends in
    // every wire request.
    const std::int32_t page_size =
        static_cast<std::int32_t>(pie_portable_driver::kKvPageSize);
    std::int32_t total_pages = derive_total_pages(h, page_size);
    const std::int32_t requested_pages = total_pages;
    std::unique_ptr<pie_portable_driver::Executor> executor_ptr;
    while (total_pages >= kMinTotalPages) {
        try {
            executor_ptr = std::make_unique<pie_portable_driver::Executor>(
                model, total_pages, page_size);
            break;
        } catch (const std::exception& e) {
            if (total_pages == kMinTotalPages) throw;
            const std::int32_t next_pages =
                std::max<std::int32_t>(kMinTotalPages, total_pages / 2);
            std::cerr << "[pie-driver-portable] warning: KV cache allocation "
                      << "failed at " << total_pages << " pages: " << e.what()
                      << "; retrying with " << next_pages << " pages\n";
            total_pages = next_pages;
        }
    }
    if (!executor_ptr) {
        throw std::runtime_error("forward: failed to allocate KV cache");
    }
    auto& executor = *executor_ptr;
    if (cfg.runtime.verbose) {
        std::cerr << "[pie-driver-portable] forward executor ready (total_pages="
                  << total_pages << ", page_size=" << page_size
                  << ", kv_buf=" << (executor.kv_buffer_size() / (1024.0 * 1024.0))
                  << " MiB)\n";
    }
    if (total_pages != requested_pages) {
        std::cerr << "[pie-driver-portable] warning: reduced KV cache from "
                  << requested_pages << " to " << total_pages
                  << " pages to fit the selected backend\n";
    }

    // ---- Optional host swap pool + cold-path registration. ------------------
    std::unique_ptr<pie_portable_driver::HostSwapPool> swap_pool;
    auto adapters = std::make_unique<pie_portable_driver::AdapterPool>();
    executor.set_adapters(adapters.get());
    const std::int32_t cpu_pages = 0;
    if (cpu_pages > 0) {
        const auto& hp = model.hparams();
        swap_pool = std::make_unique<pie_portable_driver::HostSwapPool>(
            hp.num_hidden_layers,
            hp.num_key_value_heads,
            hp.head_dim,
            cpu_pages,
            page_size,
            ggml_type_size(GGML_TYPE_F16));
        if (cfg.runtime.verbose) {
            std::cerr << "[pie-driver-portable] host swap pool: "
                      << cpu_pages << " pages × "
                      << (swap_pool->page_bytes() / 1024.0) << " KiB/layer × "
                      << hp.num_hidden_layers << " layers × 2 (K+V) = "
                      << (cpu_pages * swap_pool->page_bytes() *
                          hp.num_hidden_layers * 2 / (1024.0 * 1024.0))
                      << " MiB\n";
        }
    }
    // ---- Offline self-test paths. ------------------------------------------
    if (!test_multi_prompts.empty()) {
        auto prompts = parse_pipe_csv_u32(test_multi_prompts);
        auto ctx_ids = seq_ctx_ids(prompts.size());
        run_offline_test(executor, std::move(prompts), std::move(ctx_ids),
                         test_max_new, OfflineTestMode::Multi);
        return 0;
    }
    if (test_replicate > 0 && !test_prompt_tokens.empty()) {
        auto prompt  = parse_csv_u32(test_prompt_tokens);
        std::vector<std::vector<std::uint32_t>> prompts(test_replicate, prompt);
        run_offline_test(executor, std::move(prompts),
                         seq_ctx_ids(static_cast<std::size_t>(test_replicate)),
                         test_max_new, OfflineTestMode::Replicate);
        return 0;
    }
    if (!test_prompt_tokens.empty()) {
        std::vector<std::vector<std::uint32_t>> prompts{
            parse_csv_u32(test_prompt_tokens)};
        run_offline_test(executor, std::move(prompts), seq_ctx_ids(1),
                         test_max_new, OfflineTestMode::Single);
        return 0;
    }

    // ---- Open the in-process server. ----------------------------------------
    // The runtime owns the channel; we receive requests via the FFI vtable
    // it handed us.
    pie_driver::InProcServer server(vtable);
    register_server(&server);

    if (install_signal_handlers) {
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);
    }

    // ---- Capability handshake. ----------------------------------------------
    // Advertise the largest single context the KV pool can hold. Model
    // metadata such as max_position_embeddings may describe the training
    // window, but it should not impose a runtime admission cap.
    const std::uint32_t max_model_len =
        static_cast<std::uint32_t>(
            std::max<std::int64_t>(
                0,
                static_cast<std::int64_t>(total_pages) *
                    static_cast<std::int64_t>(page_size)));
    const std::uint32_t max_forward_tokens =
        static_cast<std::uint32_t>(
            derive_max_forward_tokens(total_pages, page_size));
    const std::uint32_t max_forward_requests =
        static_cast<std::uint32_t>(
            derive_max_forward_requests(static_cast<std::int32_t>(max_forward_tokens)));
    nlohmann::json caps = {
        {"total_pages",            total_pages},
        {"kv_page_size",           page_size},
        {"swap_pool_size",         cpu_pages},
        {"max_forward_tokens",     max_forward_tokens},
        {"max_forward_requests",   max_forward_requests},
        {"max_page_refs",          total_pages},
        {"max_logit_rows",         std::numeric_limits<std::uint32_t>::max()},
        {"max_prob_rows",          std::numeric_limits<std::uint32_t>::max()},
        {"max_custom_mask_bytes",  std::numeric_limits<std::uint32_t>::max()},
        {"max_sampler_rows",       std::numeric_limits<std::uint32_t>::max()},
        {"max_logprob_labels",     std::numeric_limits<std::uint32_t>::max()},
        {"arch_name",              model.arch_name_pie()},
        {"vocab_size",             h.vocab_size},
        {"max_model_len",          max_model_len},
        {"activation_dtype",       model.activation_dtype_str()},
        {"snapshot_dir",           cfg.model.hf_path},
    };
    // Hand caps to the host. The standalone executable's default
    // callback writes `READY <json>` to stdout (the Python wrapper greps
    // for `^READY `); library callers route the JSON wherever they want.
    const std::string caps_json = caps.dump();
    ready_cb(caps_json.c_str(), ready_ctx);

    if (cfg.runtime.verbose) {
        std::cerr << "[pie-driver-portable] serving on in-process channel\n";
    }

    pie_portable_driver::service::InProcService service{
        executor, model, swap_pool.get(), *adapters, cfg.runtime.verbose};
    service.serve_forever(server);
    const std::uint64_t handled = service.handled();

    unregister_server(&server);
    if (cfg.runtime.verbose) {
        std::cerr << "[pie-driver-portable] shutting down (handled " << handled
                  << " requests)\n";
    }
    return 0;
}

}  // namespace

extern "C" int pie_driver_portable_run_inproc(int argc,
                                              char** argv,
                                              int install_signal_handlers,
                                              pie_driver_portable_ready_cb ready_cb,
                                              void* ready_ctx,
                                              pie_driver::PieInProcVTable vtable) {
    try {
        return run_impl(argc, argv, install_signal_handlers, ready_cb, ready_ctx, vtable);
    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-portable] fatal: " << e.what() << "\n";
        return -1;
    } catch (...) {
        std::cerr << "[pie-driver-portable] fatal: unknown exception\n";
        return -1;
    }
}

// Reaches into the same server registry the SIGINT/SIGTERM handler uses.
// One host process can embed multiple same-flavor DP replicas, so stop every
// live driver server rather than only the most recently registered one.
extern "C" void pie_driver_portable_request_stop(void) {
    stop_servers();
}
