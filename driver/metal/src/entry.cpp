// pie_driver_metal — MLX/Metal backend library entry point (foundation
// skeleton).
//
// All meaningful logic lives in `run_impl`; the `extern "C"` wrappers at
// the bottom catch any escaping C++ exception so we never propagate across
// the FFI boundary (which would be UB). Mirrors the shape of
// driver/cuda/src/entry.cpp.
//
// This is the seam beta/charlie/delta build on: the stub `InProcService`
// answers Health + Forward (dummy tokens) so the driver compiles, links,
// registers with `pie-worker`, and round-trips before the MLX compute
// layer, model graphs, and weight loader land.

#include "entry.hpp"

#include <cctype>
#include <csignal>
#include <cstdint>
#include <cstdlib>
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
#include "raw_metal/raw_metal_executor.hpp"
#include "service/inproc_service.hpp"

#if defined(PIE_METAL_HAS_MLX)
#include <cstdlib>

#include <mlx/mlx.h>

#include "executor/executor.hpp"
#include "kv_cache.hpp"
#include "loader/model_loader.hpp"
#include "model/config.hpp"
#include "model/model_graph.hpp"
#include "model/weights.hpp"
#endif

namespace {

// Model facts the caps handshake needs. Read best-effort from the
// snapshot's config.json; the stub falls back to neutral defaults when no
// snapshot is wired (e.g. the standalone self-check). The weight loader
// (delta) supersedes this with values pinned by the actual checkpoint.
struct ModelFacts {
    std::uint32_t vocab_size = 32000;
    std::uint32_t max_model_len = 8192;
    std::string arch_name = "llama";
    // True for hybrid linear-attention archs (qwen3.6 / qwen3_next): they need a
    // recurrent-state slot pool (rs_cache) advertised so the runtime allocates
    // per-request rs_slot_ids. Detected from config.json (incl. text_config).
    bool has_linear_attn = false;
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
        // Hybrid linear-attention detection (qwen3.6 / qwen3_next). Config keys
        // live under `text_config` on multimodal-style dumps, else at the root.
        const nlohmann::json& tc =
            (j.contains("text_config") && j["text_config"].is_object())
                ? j["text_config"]
                : j;
        if (tc.contains("linear_num_value_heads") &&
            tc["linear_num_value_heads"].is_number_integer() &&
            tc["linear_num_value_heads"].get<int>() > 0) {
            facts.has_linear_attn = true;
        }
        if (tc.contains("layer_types") && tc["layer_types"].is_array()) {
            for (const auto& t : tc["layer_types"]) {
                if (t.is_string() &&
                    t.get<std::string>() == "linear_attention") {
                    facts.has_linear_attn = true;
                    break;
                }
            }
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
    // Hybrid linear-attention archs (qwen3.6) need a recurrent-state slot pool:
    // delta's LinearStateCache is sized to max_forward_requests, so advertise
    // that many rs_cache slots and clamp max_forward_requests to it (mirrors the
    // cuda driver). The runtime then allocates per-request rs_slot_ids in range.
    const bool rs_cache_required = facts.has_linear_attn;
    const std::uint32_t rs_cache_slots =
        rs_cache_required ? cfg.batching.max_forward_requests : 0u;
    const std::uint32_t max_forward_requests =
        rs_cache_required
            ? std::min(cfg.batching.max_forward_requests, rs_cache_slots)
            : cfg.batching.max_forward_requests;
    nlohmann::json caps = {
        {"total_pages", total_pages},
        {"kv_page_size", cfg.batching.kv_page_size},
        {"swap_pool_size", cfg.batching.cpu_pages},
        {"max_forward_tokens", cfg.batching.max_forward_tokens},
        {"max_forward_requests", max_forward_requests},
        {"rs_cache_required", rs_cache_required},
        {"rs_cache_slots", rs_cache_slots},
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

#if defined(PIE_METAL_HAS_MLX)
namespace mx = mlx::core;

// The live forward pipeline: charlie's arch graph + delta's paged KV cache +
// beta's executor that drives them per Forward fire. Owned by the serve loop
// and attached to the InProcService; the Executor borrows graph + kv, so field
// order matters (graph + kv declared before, destroyed after, the executor).
struct ModelRuntime {
    std::unique_ptr<pie_metal_driver::model::ModelGraph> graph;
    std::unique_ptr<pie_metal_driver::PagedKvCache>      kv;
    // Hybrid linear-attention state store (qwen3.6); null for non-hybrid archs.
    // Declared before `executor` so it outlives the borrowing Executor.
    std::unique_ptr<pie_metal_driver::LinearStateCache>  lin_cache;
    std::unique_ptr<pie_metal_driver::Executor>          executor;
};

// Random bf16 weight, scaled small to keep synthetic activations finite.
pie_metal_driver::Tensor synth_w(std::initializer_list<int> shape,
                                 float scale = 0.02f) {
    mx::Shape s(shape);
    return mx::astype(
        mx::multiply(mx::random::normal(s, mx::float32), mx::array(scale)),
        mx::bfloat16);
}
pie_metal_driver::Tensor synth_norm(int n) {
    return mx::astype(mx::ones({n}, mx::float32), mx::bfloat16);
}

// Build a small random Llama-shaped model so the executor path can be driven
// end-to-end on the Metal GPU before delta's weight_loader lands. Gated behind
// PIE_METAL_SYNTHETIC_MODEL=1 — a dev/self-test affordance only. The real path
// (delta's loader → ModelConfig + ModelWeights from the checkpoint) replaces
// this builder; the rest of the wiring (graph + kv + executor) is identical.
std::unique_ptr<ModelRuntime> build_synthetic_runtime(
    const pie_metal_driver::Config& cfg, const ModelFacts& facts) {
    using namespace pie_metal_driver;
    using namespace pie_metal_driver::model;

    model::ModelConfig mc;
    mc.arch                    = PieArch::Llama3;
    mc.hf_model_type           = "llama";
    mc.num_hidden_layers       = 2;
    mc.num_attention_heads     = 4;
    mc.num_key_value_heads     = 2;
    mc.head_dim                = 16;
    mc.hidden_size             = mc.num_attention_heads * mc.head_dim;
    mc.intermediate_size       = 128;
    mc.vocab_size              = static_cast<std::int32_t>(facts.vocab_size);
    mc.max_position_embeddings = static_cast<std::int32_t>(facts.max_model_len);
    mc.rms_norm_eps            = 1e-6f;
    mc.rope_theta              = 10000.0f;
    mc.tie_word_embeddings     = true;

    const int H   = mc.hidden_size;
    const int qd  = mc.num_attention_heads * mc.head_dim;
    const int kvd = mc.num_key_value_heads * mc.head_dim;
    const int I   = mc.intermediate_size;
    const int V   = mc.vocab_size;

    ModelWeights w;
    w.embed      = synth_w({V, H});
    w.final_norm = synth_norm(H);
    w.layers.resize(mc.num_hidden_layers);
    for (auto& L : w.layers) {
        L.attn_norm = synth_norm(H);
        L.ffn_norm  = synth_norm(H);
        L.q_proj    = synth_w({qd,  H});
        L.k_proj    = synth_w({kvd, H});
        L.v_proj    = synth_w({kvd, H});
        L.o_proj    = synth_w({H,   qd});
        L.gate_proj = synth_w({I,   H});
        L.up_proj   = synth_w({I,   H});
        L.down_proj = synth_w({H,   I});
    }

    auto rt = std::make_unique<ModelRuntime>();
    rt->graph = make_model_graph(mc, std::move(w));

    PagedKvGeometry geo;
    geo.n_layers   = mc.num_hidden_layers;
    geo.page_size  = static_cast<int>(cfg.batching.kv_page_size);
    geo.n_pages    = static_cast<int>(cfg.batching.total_pages);
    geo.n_kv_heads = mc.num_key_value_heads;
    geo.head_dim   = mc.head_dim;
    rt->kv = std::make_unique<PagedKvCache>(geo, DType::BF16);

    rt->executor = std::make_unique<Executor>(*rt->graph, *rt->kv);
    return rt;
}

// Construct the forward pipeline for the loaded model, or nullptr to fall back
// to the stub Forward. Loads the real checkpoint at `cfg.model.hf_path` via
// delta's loader (config.json + safetensors → graph + paged-KV); the
// PIE_METAL_SYNTHETIC_MODEL env override swaps in a random dev model for
// pipeline smoke-testing without a checkpoint.
std::unique_ptr<ModelRuntime> build_model_runtime(
    const pie_metal_driver::Config& cfg, const ModelFacts& facts) {
    using namespace pie_metal_driver;

    const char* synth = std::getenv("PIE_METAL_SYNTHETIC_MODEL");
    if (synth != nullptr && synth[0] != '\0' && synth[0] != '0') {
        std::cerr << "[pie-driver-metal] building SYNTHETIC dev model "
                     "(PIE_METAL_SYNTHETIC_MODEL set) — random weights\n";
        return build_synthetic_runtime(cfg, facts);
    }

    if (cfg.model.hf_path.empty()) {
        std::cerr << "[pie-driver-metal] no model.hf_path configured — "
                     "serving stub Forward\n";
        return nullptr;
    }

    try {
        // Real checkpoint: parse config.json + load/bind safetensors + build
        // graph + allocate the paged-KV cache from model geometry (delta).
        loader::LoadedModel lm =
            loader::load_model(cfg.model.hf_path, cfg.batching);
        std::cerr << "[pie-driver-metal] loaded " << lm.caps.arch_name
                  << " (layers=" << lm.caps.num_hidden_layers
                  << ", heads=" << lm.caps.num_attention_heads
                  << ", kv_heads=" << lm.caps.num_key_value_heads
                  << ", head_dim=" << lm.caps.head_dim
                  << ", hidden=" << lm.caps.hidden_size
                  << ", vocab=" << lm.caps.vocab_size
                  << ", act=" << lm.caps.activation_dtype << ")\n";

        auto rt = std::make_unique<ModelRuntime>();
        rt->graph     = std::move(lm.graph);
        rt->kv        = std::move(lm.kv);
        rt->lin_cache = std::move(lm.lin_cache);  // null for non-hybrid archs
        rt->executor  = std::make_unique<Executor>(*rt->graph, *rt->kv);
        rt->executor->set_linear_state_cache(rt->lin_cache.get());
        return rt;
    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-metal] model load failed (" << e.what()
                  << ") — serving stub Forward\n";
        return nullptr;
    }
}
#endif  // PIE_METAL_HAS_MLX

// Resolve the directory holding the raw-Metal .metal kernel sources for runtime
// PSO compilation: PIE_METAL_KERNELS_DIR env override, else the compiled-in
// source-tree default (packaging an installed kernels dir is a follow-up).
std::string resolve_kernels_dir() {
    if (const char* env = std::getenv("PIE_METAL_KERNELS_DIR");
        env != nullptr && env[0] != '\0') {
        return env;
    }
#ifdef PIE_METAL_KERNELS_DIR_DEFAULT
    return PIE_METAL_KERNELS_DIR_DEFAULT;
#else
    return "";
#endif
}

// Build the default (MLX-free) raw-Metal forward executor for the configured
// checkpoint, or nullptr to fall back to the stub Forward.
std::unique_ptr<pie_metal_driver::raw_metal::RawMetalExecutor>
build_raw_executor(const pie_metal_driver::Config& cfg, const ModelFacts& facts) {
    if (cfg.model.hf_path.empty()) {
        std::cerr << "[pie-driver-metal] no model.hf_path configured — "
                     "serving stub Forward\n";
        return nullptr;
    }
    const std::string kernels_dir = resolve_kernels_dir();
    return pie_metal_driver::raw_metal::make_raw_metal_executor(
        cfg.model.hf_path, kernels_dir, facts.vocab_size);
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

    // Default compute backend: the MLX-free raw-Metal pipeline. Held here so it
    // outlives the serve loop. The optional MLX `ModelGraph` executor is built
    // only on explicit opt-in (PIE_METAL_USE_MLX_EXECUTOR) when compiled in.
    std::unique_ptr<pie_metal_driver::raw_metal::RawMetalExecutor> raw_exec;
    bool attached = false;
#if defined(PIE_METAL_HAS_MLX)
    std::unique_ptr<ModelRuntime> runtime;
    const char* use_mlx = std::getenv("PIE_METAL_USE_MLX_EXECUTOR");
    const bool prefer_mlx = use_mlx != nullptr && use_mlx[0] != '\0' &&
                            use_mlx[0] != '0';
    if (prefer_mlx) {
        runtime = build_model_runtime(cfg, facts);
        if (runtime) {
            service.set_executor(runtime->executor.get());
            attached = true;
            std::cerr << "[pie-driver-metal] serving (arch=" << facts.arch_name
                      << ", vocab=" << facts.vocab_size
                      << ", MLX executor on "
                      << (mx::default_device() == mx::Device::gpu ? "Metal GPU"
                                                                  : "CPU")
                      << ")\n";
        }
    }
#endif
    if (!attached) {
        raw_exec = build_raw_executor(cfg, facts);
        if (raw_exec) {
            service.set_executor(raw_exec.get());
            attached = true;
            std::cerr << "[pie-driver-metal] serving (arch=" << facts.arch_name
                      << ", vocab=" << facts.vocab_size
                      << ", raw-Metal executor)\n";
        }
    }
    if (!attached) {
        std::cerr << "[pie-driver-metal] serving (arch=" << facts.arch_name
                  << ", vocab=" << facts.vocab_size
                  << ", stub forward — no model attached)\n";
    }
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
