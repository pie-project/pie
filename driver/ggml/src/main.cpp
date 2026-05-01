// pie_driver_ggml — portable ggml backend, sibling to ../cuda (native CUDA +
// flashinfer) and ../../runtime (Rust). Consumed by the Python wrapper
// `pie_driver_ggml`. Loads HuggingFace safetensors directly (no GGUF).
//
// M1.2: parses config, loads HF safetensors → ggml backend tensors, emits a
// real `READY <json>` capabilities line, opens shmem, decodes incoming
// `BatchedForwardPassRequest`s, and replies with an empty payload.
// Forward pass + sampler land in M1.3.

#include <atomic>
#include <cctype>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <span>
#include <string>
#include <vector>

#include <CLI/CLI.hpp>
#include <ggml-backend.h>
#include <nlohmann/json.hpp>

#include "aux_server.hpp"
#include "config.hpp"
#include "forward.hpp"
#include "hf_config.hpp"
#include "model.hpp"
#include "shmem_ipc.hpp"
#include "shmem_schema.hpp"

namespace {

std::atomic<pie_ggml_driver::ShmemServer*> g_server{nullptr};

void on_signal(int) {
    if (auto* s = g_server.load()) s->stop();
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

enum class OfflineTestMode {
    Single,     // one prompt, print all tokens, report per-context tok/s
    Multi,      // N prompts, print all, report wall-time only
    Replicate,  // N copies of one prompt, print [0] only, report aggregate tok/s
};

// Drives generate_multi() with the supplied prompts + ctx_ids and prints
// outputs + timing per the test mode. After the run, the engine's
// per-stage timing breakdown is logged (covers the prefill + decode loop
// across all contexts).
void run_offline_test(pie_ggml_driver::ForwardEngine& engine,
                      std::vector<std::vector<std::uint32_t>> prompts,
                      std::vector<std::uint64_t> ctx_ids,
                      int max_new,
                      OfflineTestMode mode) {
    using clock = std::chrono::steady_clock;
    const std::size_t n = prompts.size();

    switch (mode) {
        case OfflineTestMode::Single:
            std::cerr << "[pie-driver-ggml] self-test: prompt="
                      << prompts[0].size() << " tokens, max_new=" << max_new << "\n";
            break;
        case OfflineTestMode::Multi:
            std::cerr << "[pie-driver-ggml] multi-context self-test: "
                      << n << " contexts, max_new=" << max_new << "\n";
            break;
        case OfflineTestMode::Replicate:
            std::cerr << "[pie-driver-ggml] throughput test: " << n
                      << " concurrent contexts × " << max_new
                      << " tokens (prompt=" << prompts[0].size()
                      << " tokens each)\n";
            break;
    }

    engine.reset_timings();
    const auto t0 = clock::now();
    auto outs = engine.generate_multi(prompts, max_new, ctx_ids);
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
            std::cerr << "[pie-driver-ggml] generated " << outs[0].size()
                      << " tokens in " << dt_ms << " ms ("
                      << (outs[0].size() * 1000.0 / denom) << " tok/s)\n";
            break;
        }
        case OfflineTestMode::Multi: {
            const std::size_t per = outs.empty() ? 0 : outs[0].size();
            std::cerr << "[pie-driver-ggml] generated " << per
                      << " tokens × " << outs.size()
                      << " contexts in " << dt_ms << " ms\n";
            break;
        }
        case OfflineTestMode::Replicate: {
            const long total = static_cast<long>(n) * max_new;
            std::cerr << "[pie-driver-ggml] generated " << total
                      << " tokens (" << n << " × " << max_new
                      << ") in " << dt_ms << " ms ("
                      << (total * 1000.0 / denom) << " tok/s aggregate)\n";
            break;
        }
    }
    // Per-stage breakdown (covers every compute_() call across prefill +
    // decode). Useful for spotting whether time is going into graph
    // build, the GPU compute itself, the logits download, or sampling.
    engine.log_timings("offline-test");
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"pie_driver_ggml — portable ggml backend for Pie"};
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

    const auto cfg = pie_ggml_driver::load_config(config_path);

    // Informational logs go to stderr — stdout is reserved for the READY
    // handshake line consumed by the Python wrapper.
    std::cerr << "[pie-driver-ggml] config loaded\n"
              << "  shmem.name        = " << cfg.shmem.name << "\n"
              << "  shmem.num_slots   = " << cfg.shmem.num_slots << "\n"
              << "  model.hf_path     = " << cfg.model.hf_path << "\n"
              << "  model.n_gpu_layers= " << cfg.model.n_gpu_layers << "\n"
              << "  model.n_ctx       = " << cfg.model.n_ctx << "\n";

    // ---- Backend discovery. Registers all backends compiled into ggml
    // (CPU + CUDA / Metal / Vulkan when those `GGML_*=ON` flags were set
    // at build time). After this, `ggml_backend_init_best()` will pick
    // GPU over CPU.
    ggml_backend_load_all();

    // ---- Load the model. -----------------------------------------------------
    const bool prefer_gpu = (cfg.model.n_gpu_layers != 0);
    std::cerr << "[pie-driver-ggml] loading model from " << cfg.model.hf_path
              << " (prefer_gpu=" << (prefer_gpu ? "yes" : "no") << ")\n";
    pie_ggml_driver::Model model(cfg.model.hf_path, prefer_gpu);
    const auto& h = model.hparams();
    std::cerr << "[pie-driver-ggml] loaded "
              << model.arch_name_pie() << " ("
              << h.num_hidden_layers << " layers, hidden=" << h.hidden_size
              << ", heads=" << h.num_attention_heads << "/"
              << h.num_key_value_heads << " kv, head_dim=" << h.head_dim
              << ", vocab=" << h.vocab_size
              << ", dtype=" << model.activation_dtype_str()
              << ", buf=" << (model.buffer_size() / (1024.0 * 1024.0)) << " MiB"
              << ", backend=" << model.backend_name() << ")\n";

    // ---- Allocate forward engine + paged KV pool. ---------------------------
    // The runtime owns page allocation; we report total_pages and page_size
    // in the READY handshake and honor the page IDs the runtime sends in
    // every BPIQ request.
    const std::int32_t total_pages =
        static_cast<std::int32_t>(cfg.batching.max_num_kv_pages);
    const std::int32_t page_size =
        static_cast<std::int32_t>(cfg.batching.kv_page_size);
    pie_ggml_driver::ForwardEngine engine(model, total_pages, page_size);
    std::cerr << "[pie-driver-ggml] forward engine ready (total_pages="
              << total_pages << ", page_size=" << page_size
              << ", kv_buf=" << (engine.kv_buffer_size() / (1024.0 * 1024.0))
              << " MiB)\n";

    // ---- Optional host swap pool + aux IPC (M7). ----------------------------
    std::unique_ptr<pie_ggml_driver::HostSwapPool> swap_pool;
    std::unique_ptr<pie_ggml_driver::AuxServer>    aux_server;
    auto adapters = std::make_unique<pie_ggml_driver::AdapterPool>();
    engine.set_adapters(adapters.get());
    const std::int32_t cpu_pages =
        static_cast<std::int32_t>(cfg.batching.cpu_pages);
    if (cpu_pages > 0) {
        const auto& hp = model.hparams();
        swap_pool = std::make_unique<pie_ggml_driver::HostSwapPool>(
            hp.num_hidden_layers,
            hp.num_key_value_heads,
            hp.head_dim,
            cpu_pages,
            page_size,
            ggml_type_size(GGML_TYPE_F16));
        std::cerr << "[pie-driver-ggml] host swap pool: "
                  << cpu_pages << " pages × "
                  << (swap_pool->page_bytes() / 1024.0) << " KiB/layer × "
                  << hp.num_hidden_layers << " layers × 2 (K+V) = "
                  << (cpu_pages * swap_pool->page_bytes() *
                      hp.num_hidden_layers * 2 / (1024.0 * 1024.0))
                  << " MiB\n";
    }
    if (!cfg.aux_ipc.socket_path.empty()) {
        aux_server = std::make_unique<pie_ggml_driver::AuxServer>(
            cfg.aux_ipc.socket_path,
            engine.kv(),
            swap_pool.get(),
            adapters.get(),
            model.backend(),
            &model.hparams());
        std::cerr << "[pie-driver-ggml] aux IPC listening on "
                  << cfg.aux_ipc.socket_path << "\n";
    }

    // ---- Offline self-test paths. ------------------------------------------
    if (!test_multi_prompts.empty()) {
        auto prompts = parse_pipe_csv_u32(test_multi_prompts);
        auto ctx_ids = seq_ctx_ids(prompts.size());
        run_offline_test(engine, std::move(prompts), std::move(ctx_ids),
                         test_max_new, OfflineTestMode::Multi);
        return 0;
    }
    if (test_replicate > 0 && !test_prompt_tokens.empty()) {
        auto prompt  = parse_csv_u32(test_prompt_tokens);
        std::vector<std::vector<std::uint32_t>> prompts(test_replicate, prompt);
        run_offline_test(engine, std::move(prompts),
                         seq_ctx_ids(static_cast<std::size_t>(test_replicate)),
                         test_max_new, OfflineTestMode::Replicate);
        return 0;
    }
    if (!test_prompt_tokens.empty()) {
        std::vector<std::vector<std::uint32_t>> prompts{
            parse_csv_u32(test_prompt_tokens)};
        run_offline_test(engine, std::move(prompts), seq_ctx_ids(1),
                         test_max_new, OfflineTestMode::Single);
        return 0;
    }

    // ---- Open shmem. ---------------------------------------------------------
    pie_ggml_driver::ShmemServer server(
        cfg.shmem.name,
        cfg.shmem.num_slots,
        cfg.shmem.req_buf,
        cfg.shmem.resp_buf,
        cfg.shmem.spin_us);
    g_server.store(&server);

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    // ---- Capability handshake. ----------------------------------------------
    // Pie's runtime reads this JSON line and constructs a `DriverCapabilities`
    // (see `pie/src/pie/capabilities.py`). KV pages and swap pool size stay
    // 0 in M1.2 — the KV manager (M2) computes real values.
    const std::uint32_t max_model_len =
        static_cast<std::uint32_t>(std::min<std::int32_t>(
            h.max_position_embeddings,
            static_cast<std::int32_t>(cfg.model.n_ctx)));
    nlohmann::json caps = {
        {"total_pages",      total_pages},
        {"kv_page_size",     page_size},
        {"swap_pool_size",   cpu_pages},
        {"max_batch_tokens", cfg.batching.max_batch_tokens},
        {"max_batch_size",   cfg.batching.max_batch_size},
        {"arch_name",        model.arch_name_pie()},
        {"vocab_size",       h.vocab_size},
        {"max_model_len",    max_model_len},
        {"activation_dtype", model.activation_dtype_str()},
        {"snapshot_dir",     cfg.model.hf_path},
        {"shmem_name",       cfg.shmem.name},
    };
    // The wrapper greps stdout for `^READY ` to complete the handshake.
    std::cout << "READY " << caps.dump() << std::endl;

    std::cerr << "[pie-driver-ggml] serving on shmem " << server.name()
              << " (" << server.num_slots() << " slots, "
              << "req_buf=" << server.req_buf_size() << ", "
              << "resp_buf=" << server.resp_buf_size() << ")\n";

    std::uint64_t handled = 0;

    server.serve_forever([&](const pie_ggml_driver::SlotRequest& req,
                             std::span<std::uint8_t> response) -> std::size_t {
        ++handled;
        if (req.method_tag != pie_ggml_driver::METHOD_TAG_FIRE_BATCH) {
            std::cerr << "[pie-driver-ggml] unsupported method_tag="
                      << req.method_tag << " req_id=" << req.req_id << "\n";
            return 0;
        }

        pie_ggml_driver::schema::DecodedRequest decoded;
        try {
            decoded = pie_ggml_driver::schema::decode_request(req.payload);
        } catch (const std::exception& e) {
            std::cerr << "[pie-driver-ggml] decode failed for req_id=" << req.req_id
                      << ": " << e.what() << "\n";
            return 0;
        }

        if (handled <= 4 || handled % 100 == 0) {
            const auto tokens =
                decoded.as<std::uint32_t>(pie_ggml_driver::schema::A_TOKEN_IDS);
            const auto context_ids =
                decoded.as<std::uint64_t>(pie_ggml_driver::schema::A_CONTEXT_IDS);
            std::cerr << "[pie-driver-ggml] req_id=" << req.req_id
                      << " device=" << decoded.device_id
                      << " single_token=" << decoded.single_token_mode
                      << " tokens=" << tokens.size()
                      << " contexts=" << context_ids.size() << "\n";
        }

        try {
            return engine.run(decoded, response);
        } catch (const std::exception& e) {
            std::cerr << "[pie-driver-ggml] forward failed for req_id="
                      << req.req_id << ": " << e.what() << "\n";
            return 0;
        }
    });

    g_server.store(nullptr);
    std::cerr << "[pie-driver-ggml] shutting down (handled " << handled
              << " requests)\n";
    // Print per-stage timings on demand (e.g. e2e benchmarks). Default
    // off so production logs stay quiet.
    if (std::getenv("PIE_GGML_LOG_TIMINGS")) {
        engine.log_timings("shmem-loop");
    }
    return 0;
}
