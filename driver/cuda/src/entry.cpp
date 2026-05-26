#include "model/mistral3.hpp"
// pie_driver_cuda — native CUDA backend library entry point.
//
// All meaningful logic lives in `run_impl`; the `extern "C"` wrapper
// at the bottom catches any escaping C++ exception so we never
// propagate across the FFI boundary (which would be UB). Mirrors
// driver/portable/src/entry.cpp's shape — see that file for the
// invariants.

#include "entry.hpp"

#include <algorithm>
#include <atomic>
#include <barrier>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdlib>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <unordered_map>
#include <utility>

#include <CLI/CLI.hpp>
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>

#include "attention_workspace.hpp"
#include "brle.hpp"
#include "config.hpp"
#include "cuda_memory_planner.hpp"
#include "custom_all_reduce.hpp"
#include "cuda_check.hpp"
#include "driver_startup.hpp"
#include "hf_snapshot.hpp"
#include "parity_harness.hpp"
#include "model/loaded_model.hpp"
#include "kernels/argmax.hpp"
#include "kernels/sample_flashinfer.hpp"
#include "kernels/sample_temp.hpp"
#include "kv_cache.hpp"
#include "model/bound_model.hpp"
#include "model/gemma2.hpp"
#include "model/gemma3n.hpp"
#include "model/gemma2_model.hpp"
#include "model/gemma3n_model.hpp"
#include "model/gemma4.hpp"
#include "model/gemma4_model.hpp"
#include "model/gemma4_mtp.hpp"
#include "model/gpt_oss.hpp"
#include "model/llama_like.hpp"
#include "model/llama_like_model.hpp"
#include "model/mixtral.hpp"
#include "model/mixtral_model.hpp"
#include "model/nemotron_h.hpp"
#include "model/nemotron_h_forward.hpp"
#include "model/nemotron_h_model.hpp"
#include "model/qwen3_5_config.hpp"
#include "model/qwen3_5_model.hpp"
#include "model/qwen3_5_moe_model.hpp"
#include "model/qwen3.hpp"
#include "model/qwen3_5.hpp"
#include "model/qwen3_5_forward.hpp"
#include "model/qwen3_5_moe.hpp"
#include "model/qwen3_5_moe_forward.hpp"
#include "model/qwen3_forward.hpp"
#include "recurrent_state_cache.hpp"
#include "swap_pool.hpp"
#include <thread>
#include <unistd.h>
#include "ops/gemm.hpp"
#include "executor/executor.hpp"
#include "service/inproc_service.hpp"
#include <pie_bridge/inproc_server.hpp>

namespace {

int configured_mtp_num_drafts(const pie_cuda_driver::Config& cfg) {
    static const int forced = [] {
        const char* v = std::getenv("PIE_MTP_DRAFT_TOKENS");
        if (v == nullptr || v[0] == '\0') return -1;
        return std::clamp(std::atoi(v), 0, 32);
    }();
    if (forced >= 0) return forced;
    return cfg.model.mtp_num_drafts;
}

// `vtable_opt` is non-null for the in-process serve loop; null for the
// parity-only standalone entry (`pie_driver_cuda_run`), which exits
// after running the parity test and never enters serve_forever.
int run_impl(int argc,
             char** argv,
             int install_signal_handlers,
             pie_driver_cuda_ready_cb ready_cb,
             void* ready_ctx,
             const pie_driver::PieInProcVTable* vtable_opt) {
    if (ready_cb == nullptr) {
        std::cerr << "[pie-driver-cuda] fatal: ready_cb is null\n";
        return -1;
    }
    CLI::App app{"pie_driver_cuda — native CUDA backend for Pie"};
    std::string config_path = "dev.toml";
    app.add_option("-c,--config", config_path, "Path to TOML config")
        ->check(CLI::ExistingFile);

    std::string parity_tokens, parity_out;
    bool parity_paged = false;
    bool parity_decode_after_prefill = false;
    auto* parity = app.add_option_group("parity", "Numeric-parity test entry");
    parity->add_option("--parity-tokens", parity_tokens,
                       "Path to a binary file of i32 token ids");
    parity->add_option("--parity-out", parity_out,
                       "Where to write the last-token logits as bf16 [vocab]");
    parity->add_flag("--parity-paged", parity_paged,
                     "Run the paged forward path (wire-shaped KV layout)");
    parity->add_flag("--parity-decode-after-prefill", parity_decode_after_prefill,
                     "After prefill on the first N-1 tokens, run a single "
                     "qo_len=1 decode step at position N-1 and dump that "
                     "step's logits. Exercises the decode kernel + KV-cache "
                     "read path in addition to prefill. Requires --parity-paged.");

    // Default-on under llama-like. `enable_cuda_graph=true` on the
    // flashinfer DecodePlan side pins plan_info layout (padded_batch_size,
    // request_indices_offset, …) across fires; per-fire DecodePlan calls
    // only update int_buf content (request_indices, block_valid_mask), and
    // device pointers stay stable. See `forward_fn.graph_safe = true` below.
    bool use_cuda_graphs = true;
    app.add_flag("--cuda-graphs,!--no-cuda-graphs", use_cuda_graphs,
                 "Capture decode forward into CUDA graphs and replay per "
                 "shape bucket. Default on for cuda_native.");

    // Tensor-parallel knobs. Override [distributed] in the TOML when
    // present so the wrapper can launch ad-hoc TP groups without
    // rewriting the config file. Empty unique-id means "fall back to
    // TOML".
    int cli_tp_size = -1, cli_tp_rank = -1;
    std::string cli_nccl_unique_id_hex;
    app.add_option("--tp-size", cli_tp_size,
                   "Tensor-parallel world size (overrides [distributed].tp_size).");
    app.add_option("--tp-rank", cli_tp_rank,
                   "This process's rank in the TP group (0..tp_size).");
    app.add_option("--nccl-unique-id-hex", cli_nccl_unique_id_hex,
                   "Hex-encoded ncclUniqueId shared across all ranks of "
                   "the TP group. Only required when tp_size > 1.");

    CLI11_PARSE(app, argc, argv);

    auto cfg = pie_cuda_driver::load_config(config_path);
    if (cli_tp_size >= 1) cfg.distributed.tp_size = cli_tp_size;
    if (cli_tp_rank >= 0) cfg.distributed.tp_rank = cli_tp_rank;
    if (!cli_nccl_unique_id_hex.empty())
        cfg.distributed.nccl_unique_id_hex = cli_nccl_unique_id_hex;
    const bool verbose = cfg.runtime.verbose;
    if (cfg.distributed.tp_size > 1 &&
        cfg.distributed.tp_rank > 0 &&
        cfg.distributed.nccl_unique_id_hex.empty()) {
        std::cerr << "[pie-driver-cuda] rank " << cfg.distributed.tp_rank
                  << " requires --nccl-unique-id-hex "
                  << "(or [distributed].nccl_unique_id_hex)\n";
        return 1;
    }

    if (!parity_tokens.empty()) {
        // Parity argument validation up-front so we don't go through
        // NCCL bootstrap only to fail on bad CLI. The actual parity
        // dispatch is deferred until after NCCL init so a TP-mode
        // parity test can drive collectives.
        if (parity_out.empty()) {
            std::cerr << "--parity-tokens requires --parity-out\n";
            return 1;
        }
        if (parity_decode_after_prefill && !parity_paged) {
            std::cerr << "--parity-decode-after-prefill requires --parity-paged\n";
            return 1;
        }
    }

    // Informational logs go to stderr — stdout is reserved for the READY
    // handshake line consumed by the host process.
    if (verbose) {
        std::cerr << "[pie-driver-cuda] config loaded\n"
                  << "  model.snap_dir  = " << cfg.model.snapshot_dir << "\n"
	                  << "  model.device    = " << cfg.model.device << "\n"
	                  << "  model.dtype     = " << cfg.model.dtype << "\n"
	                  << "  model.mxfp4_moe = " << cfg.model.mxfp4_moe << "\n"
	                  << "  tp_size         = " << cfg.distributed.tp_size << "\n"
                  << "  tp_rank         = " << cfg.distributed.tp_rank << "\n";
    }

    // Bind the requested CUDA device before NCCL init — ncclCommInitRank
    // captures whatever is current on the calling thread.
    {
        CUDA_CHECK(cudaSetDevice(
            pie_cuda_driver::parse_cuda_device_id(cfg.model.device)));
    }

    pie_cuda_driver::NcclComm tp_comm;
    if (cfg.distributed.tp_size > 1) {
        // ncclGetUniqueId opens a TCP bootstrap listener inside the
        // calling process — it must outlive the rendezvous. Rank 0
        // generates the id (when no id was passed in), prints it on
        // stdout for the wrapper to relay, then proceeds straight into
        // ncclCommInitRank. Followers receive the id from the wrapper
        // via --nccl-unique-id-hex / [distributed].nccl_unique_id_hex.
        ncclUniqueId uid;
        if (cfg.distributed.tp_rank == 0 &&
            cfg.distributed.nccl_unique_id_hex.empty()) {
            NCCL_CHECK(ncclGetUniqueId(&uid));
            const auto hex = pie_cuda_driver::nccl_unique_id_to_hex(uid);
            std::cout << "NCCL_UID " << hex << std::endl;
        } else {
            uid = pie_cuda_driver::nccl_unique_id_from_hex(
                cfg.distributed.nccl_unique_id_hex);
        }
        tp_comm = pie_cuda_driver::NcclComm(
            cfg.distributed.tp_size, cfg.distributed.tp_rank, uid);
        if (verbose) {
            std::cerr << "[pie-driver-cuda] NCCL comm initialised "
                      << "(world=" << tp_comm.world_size()
                      << ", rank=" << tp_comm.rank() << ")\n";
        }
        pie_cuda_driver::tp_startup_cpu_barrier(cfg);

        // Smoke test: every rank contributes (rank+1); sum should be
        // world*(world+1)/2. Catches mis-numbered ranks at startup.
        cudaStream_t s = nullptr;
        CUDA_CHECK(cudaStreamCreate(&s));
        int* d_v = nullptr;
        CUDA_CHECK(cudaMalloc(&d_v, sizeof(int)));
        const int rank1 = cfg.distributed.tp_rank + 1;
        CUDA_CHECK(cudaMemcpyAsync(d_v, &rank1, sizeof(int),
                                   cudaMemcpyHostToDevice, s));
        NCCL_CHECK_ASYNC(ncclAllReduce(d_v, d_v, 1, ncclInt32, ncclSum,
                                       tp_comm.comm(), s),
                         tp_comm.comm());
        int h_v = 0;
        CUDA_CHECK(cudaMemcpyAsync(&h_v, d_v, sizeof(int),
                                   cudaMemcpyDeviceToHost, s));
        CUDA_CHECK(cudaStreamSynchronize(s));
        CUDA_CHECK(cudaFree(d_v));
        CUDA_CHECK(cudaStreamDestroy(s));
        const int W = cfg.distributed.tp_size;
        const int expected = W * (W + 1) / 2;
        if (h_v != expected) {
            std::cerr << "[pie-driver-cuda] NCCL smoke test FAILED: got "
                      << h_v << ", expected " << expected << "\n";
            return 3;
        }
        if (verbose) {
            std::cerr << "[pie-driver-cuda] NCCL smoke test ok ("
                      << h_v << "==" << expected << ")\n";
        }
    }
    pie_cuda_driver::NcclComm* tp_comm_ptr =
        (cfg.distributed.tp_size > 1) ? &tp_comm : nullptr;

    // Parity mode: every rank participates so collectives complete;
    // only rank 0 dumps logits to disk. The harness compares rank 0's
    // output against a single-GPU reference run.
    if (!parity_tokens.empty()) {
        const std::string out_path = (cfg.distributed.tp_rank == 0)
            ? parity_out
            : (parity_out + ".rank" +
               std::to_string(cfg.distributed.tp_rank));
        return pie_cuda_driver::run_parity(cfg, parity_tokens, out_path, parity_paged,
                          parity_decode_after_prefill, tp_comm_ptr);
    }

    auto engine = pie_cuda_driver::LoadedModel::load(cfg, tp_comm_ptr);

    {
        const auto& mt = engine.hf_config().model_type;
        // Llama-like family. Same RMSNorm + RoPE + GQA + SwiGLU graph; the
        // only branch is whether per-head q/k_norm exists (Qwen3 quirk),
        // which is captured in HfConfig.use_qk_norm.
        const bool supported =
            mt == "qwen3"
         || mt == "qwen3_5" || mt == "qwen3_5_text"
         || mt == "qwen3_5_moe" || mt == "qwen3_5_moe_text"
         || mt == "qwen3_moe"
         || mt == "qwen2"
         || mt == "llama" || mt == "llama3"
         || mt == "mistral" || mt == "mistral3" || mt == "ministral3"
         || mt == "mixtral"
         || mt == "gpt_oss"
         || mt == "phi3"
         // OLMo-V1 (`mt == "olmo"`) used LayerNorm, not RMSNorm — its
         // schema is genuinely different and was never wired up. OLMo-2
         // and OLMo-3 share the post-norm + q/k-norm + RMSNorm setup
         // that `bind_olmo3` materialises, so we accept both here.
         || mt == "olmo2" || mt == "olmo3"
         || mt == "gemma2"
         || mt == "gemma3" || mt == "gemma3_text"
         || mt == "gemma4" || mt == "gemma4_text"
         || mt == "gemma3n" || mt == "gemma3n_text"
         || mt == "nemotron_h";
        if (!supported) {
            std::cerr << "[pie-driver-cuda] arch '" << mt
                      << "' not yet supported (Qwen 2/3, Llama-3, "
                      << "Mistral, Mixtral, GPT-OSS, Phi-3, OLMo-3, Gemma-2/3/4)\n";
            return 2;
        }
    }
    // Centralized bound-model selection. The forward setup below keeps local
    // references for now so the rest of the serving path stays unchanged.
    auto bound_model = pie_cuda_driver::model::bind_cuda_model(engine, verbose);
    auto& weights_llama = bound_model.llama;
    auto& weights_gemma = bound_model.gemma;
    auto& weights_gemma4 = bound_model.gemma4;
    auto& weights_gemma3n = bound_model.gemma3n;
    auto& weights_mixtral = bound_model.mixtral;
    auto& weights_qwen3_5 = bound_model.qwen3_5;
    auto& weights_qwen3_5_moe = bound_model.qwen3_5_moe;
    auto& weights_nemotron_h = bound_model.nemotron_h;

    const bool is_gemma_arch = bound_model.is_gemma();
    const bool is_gemma4_arch = bound_model.is_gemma4();
    const bool is_gemma3n_arch = bound_model.is_gemma3n();
    const bool is_mixtral_arch = bound_model.is_mixtral();
    const bool is_qwen3_5_arch = bound_model.is_qwen3_5();
    const bool is_qwen3_5_moe_arch = bound_model.is_qwen3_5_moe();
    const bool is_nemotron_h_arch = bound_model.is_nemotron_h();
    const int native_mtp_num_drafts = configured_mtp_num_drafts(cfg);

    const std::size_t num_layers_bound = bound_model.num_layers();
    if (verbose) {
        std::cerr << "[pie-driver-cuda] schema bound: "
                  << num_layers_bound << " layers ("
                  << engine.hf_config().model_type
                  << (engine.hf_config().use_qk_norm ? ", q/k norm" : "")
                  << ")\n";
    }

    auto gemma4_mtp_discovery =
        pie_cuda_driver::model::discover_and_load_gemma4_mtp(
            cfg, engine, weights_gemma4,
            is_gemma4_arch, native_mtp_num_drafts, verbose);
    auto& gemma4_mtp_weights = gemma4_mtp_discovery.weights;
    auto& gemma4_mtp_runtime = gemma4_mtp_discovery.runtime;

    // Pre-allocate persistent rs_cache state for serving. CUDA-native no longer
    // accepts manual batch/KV sizing from public config; after weights are
    // resident we plan the forward arena, optional linear-attn rs_cache,
    // and remaining KV pages from gpu_mem_utilization + memory_profile.
    // Per-arch worst-case workspace dims. Gemma-4 has both
    // `use_double_wide_mlp` (intermediate doubles on shared layers)
    // and dual head_dim (sliding=256 vs full=512), so ws.q/k/v need
    // the full-attention sizing. Other archs use the single config
    // values.
    const int local_tp_size = std::max(1, cfg.distributed.tp_size);
    const int local_q_heads =
        engine.hf_config().num_attention_heads / local_tp_size;
    const int local_kv_heads =
        engine.hf_config().num_key_value_heads / local_tp_size;
    int max_mlp_intermediate =
        engine.hf_config().intermediate_size / local_tp_size;
    int max_Hq = local_q_heads * engine.hf_config().head_dim;
    int max_Hk = local_kv_heads * engine.hf_config().head_dim;
    if (is_gemma4_arch) {
        for (int v : weights_gemma4.per_layer_intermediate) {
            // Gemma-4 binds this from the already-loaded projection shape,
            // so it is already per-rank under TP.
            const int local_v = v;
            if (local_v > max_mlp_intermediate) max_mlp_intermediate = local_v;
        }
        for (int d : weights_gemma4.per_layer_head_dim) {
            const int Hq = local_q_heads * d;
            const int Hk = local_kv_heads * d;
            if (Hq > max_Hq) max_Hq = Hq;
            if (Hk > max_Hk) max_Hk = Hk;
        }
    } else if (is_gemma3n_arch) {
        // Per-layer intermediate (HF stores it as a list); head_dim is
        // uniform across layers on gemma3n, so KV cache can use the
        // standard allocator.
        for (int v : weights_gemma3n.per_layer_intermediate) {
            const int local_v = v / local_tp_size;
            if (local_v > max_mlp_intermediate) max_mlp_intermediate = local_v;
        }
    } else if (is_nemotron_h_arch) {
        const auto& hf_n = engine.hf_config();
        max_mlp_intermediate = std::max(
            max_mlp_intermediate,
            std::max(hf_n.moe_intermediate_size / local_tp_size,
                     hf_n.shared_expert_intermediate_size / local_tp_size));
    }

    std::vector<bool> qwen3_5_layer_is_linear;
    if (is_qwen3_5_arch || is_qwen3_5_moe_arch) {
        const std::size_t num_layers = is_qwen3_5_arch
            ? weights_qwen3_5.layers.size()
            : weights_qwen3_5_moe.layers.size();
        qwen3_5_layer_is_linear.resize(num_layers);
        for (std::size_t L = 0; L < num_layers; ++L) {
            const bool is_linear = is_qwen3_5_arch
                ? (weights_qwen3_5.layers[L].kind ==
                   pie_cuda_driver::model::Qwen3_5LayerWeights::Kind::LinearAttn)
                : (weights_qwen3_5_moe.layers[L].kind ==
                   pie_cuda_driver::model::Qwen3_5MoeLayerWeights::Kind::LinearAttn);
            qwen3_5_layer_is_linear[L] = is_linear;
        }
    }

    const int qwen3_5_linear_layers = static_cast<int>(std::count(
        qwen3_5_layer_is_linear.begin(), qwen3_5_layer_is_linear.end(), true));

    std::vector<bool> nemotron_h_layer_is_mamba;
    if (is_nemotron_h_arch) {
        nemotron_h_layer_is_mamba.resize(weights_nemotron_h.layers.size());
        for (std::size_t L = 0; L < weights_nemotron_h.layers.size(); ++L) {
            nemotron_h_layer_is_mamba[L] =
                weights_nemotron_h.layers[L].kind ==
                pie_cuda_driver::model::NemotronHLayerWeights::Kind::Mamba;
        }
    }
    const int nemotron_h_mamba_layers = static_cast<int>(std::count(
        nemotron_h_layer_is_mamba.begin(),
        nemotron_h_layer_is_mamba.end(),
        true));
    const int nemotron_h_attention_layer_count = is_nemotron_h_arch
        ? pie_cuda_driver::model::nemotron_h_attention_layers(engine.hf_config())
        : 0;
    const auto kv_format = pie_cuda_driver::kv_cache_format_from_string(
        cfg.batching.kv_cache_dtype, cfg.model.dtype);
    const bool graph_capable_forward =
        use_cuda_graphs && bound_model.is_llama_like() &&
        kv_format.is_native_bf16();
    const auto runtime_quant_scratch_base =
        graph_capable_forward
            ? pie_cuda_driver::runtime_quant_scratch_spec(engine, /*max_tokens=*/0)
            : pie_cuda_driver::ops::RuntimeQuantScratchSpec{};

    const pie_cuda_driver::CudaMemoryPlan mem_plan = pie_cuda_driver::plan_cuda_memory(
        cfg, engine.hf_config(), max_mlp_intermediate, max_Hq, max_Hk,
        is_gemma4_arch, weights_gemma4.per_layer_head_dim,
        weights_gemma4.kv_source_layer, is_qwen3_5_arch,
        is_qwen3_5_moe_arch, qwen3_5_linear_layers,
        is_nemotron_h_arch, nemotron_h_mamba_layers,
        kv_format, runtime_quant_scratch_base, verbose);
    const int max_workspace_tokens = mem_plan.max_workspace_tokens;
    // `mem_plan.kv_pages` is the runtime-visible KV capacity. CUDA graph
    // padding needs one isolated page for synthetic rows when replaying a
    // bucket larger than the real request count; charge that implementation
    // detail to the planner's safety headroom instead of reducing the
    // advertised runtime pool.
    const int runtime_kv_pages = mem_plan.kv_pages;
    const int physical_kv_pages =
        mem_plan.kv_pages > 0 ? mem_plan.kv_pages + 1 : mem_plan.kv_pages;
    const int graph_pad_page =
        mem_plan.kv_pages > 0 ? runtime_kv_pages : -1;
    const bool has_recurrent_state_cache =
        is_qwen3_5_arch || is_qwen3_5_moe_arch || is_nemotron_h_arch;
    const int runtime_state_slots = mem_plan.state_slots;
    const int graph_pad_slot =
        has_recurrent_state_cache && runtime_state_slots > 0 &&
                graph_pad_page >= 0
            ? runtime_state_slots
            : -1;
    const int allocated_state_slots =
        runtime_state_slots + (graph_pad_slot >= 0 ? 1 : 0);

    auto ws = pie_cuda_driver::model::Qwen3Workspace::allocate_full(
        engine.hf_config(), max_workspace_tokens,
        max_mlp_intermediate, max_Hq, max_Hk,
        mem_plan.capacity.max_logit_rows);

    auto kv_cache =
        is_gemma4_arch
            ? pie_cuda_driver::KvCache::allocate_per_layer(
                  engine.hf_config().num_hidden_layers,
                  physical_kv_pages,
                  mem_plan.kv_page_size,
                  local_kv_heads,
                  weights_gemma4.per_layer_head_dim,
                  weights_gemma4.kv_source_layer,
                  weights_gemma4.per_layer_num_kv_heads,
                  kv_format)
            : is_nemotron_h_arch
                ? pie_cuda_driver::KvCache::allocate(
                      nemotron_h_attention_layer_count,
                      physical_kv_pages,
                      mem_plan.kv_page_size,
                      local_kv_heads,
                      engine.hf_config().head_dim_kernel,
                      kv_format)
            : pie_cuda_driver::KvCache::allocate(
                  engine.hf_config().num_hidden_layers,
                  physical_kv_pages,
                  mem_plan.kv_page_size,
                  local_kv_heads,
                  engine.hf_config().head_dim_kernel,
                  kv_format);

    auto attn_ws = pie_cuda_driver::AttentionWorkspace::allocate(
        mem_plan.attn_float_workspace_bytes, 8ull * 1024 * 1024);

    // Plan-state holders used by the prepare/body split for graph-friendly
    // dispatch. Allocated unconditionally — empty on archs that don't use
    // them. `qwen3_5_plan_state` is shared between qwen3_5 and qwen3_5_moe
    // (they share `prepare_qwen3_5_decode_plan`).
    pie_cuda_driver::model::Qwen3_5PlanState qwen3_5_plan_state;

    // Qwen3.5 / Qwen3.6-MoE linear-attention extras: per-layer rs_cache
    // + a per-call workspace. Inert (default-constructed) on every other
    // arch. The MoE arch additionally needs a routed-experts workspace.
    pie_cuda_driver::model::Qwen3_5LinearAttnWorkspace qwen3_5_la_ws;
    pie_cuda_driver::RecurrentStateCache qwen3_5_state_cache;
    pie_cuda_driver::model::Qwen3_5MoeMlpWorkspace qwen3_5_moe_ws;
    pie_cuda_driver::model::NemotronHWorkspace nemotron_h_ws;
    pie_cuda_driver::RecurrentStateCache nemotron_h_state_cache;
    // Constructed inside the is_nemotron_h_arch branch below, after the
    // memory planner has sized nemotron_h_ws and nemotron_h_state_cache.
    std::unique_ptr<pie_cuda_driver::model::NemotronHModel> nemotron_h_model;
    int qwen3_5_runtime_rs_slots = 0;
    int qwen3_5_scratch_rs_slot = -1;
    if (is_qwen3_5_arch || is_qwen3_5_moe_arch) {
        const auto& cfg_q = engine.hf_config();
        const int q35_tp_size = std::max(1, cfg.distributed.tp_size);
        const int local_linear_key_heads =
            cfg_q.linear_num_key_heads / q35_tp_size;
        const int local_linear_value_heads =
            cfg_q.linear_num_value_heads / q35_tp_size;
        const int K_dim = local_linear_key_heads * cfg_q.linear_key_head_dim;
        const int V_dim = local_linear_value_heads * cfg_q.linear_value_head_dim;
        const int conv_dim = 2 * K_dim + V_dim;
        qwen3_5_la_ws = pie_cuda_driver::model::Qwen3_5LinearAttnWorkspace::allocate(
            max_workspace_tokens, conv_dim,
            local_linear_value_heads,
            local_linear_key_heads,
            cfg_q.linear_key_head_dim,
            cfg_q.linear_value_head_dim,
            /*hq=*/(cfg_q.num_attention_heads / q35_tp_size) *
                cfg_q.head_dim);
        // Allocate per-slot state for the linear-attn layers. The memory
        // planner sizes runtime slots before KV pages and clamps max forward
        // requests to the resulting slot count. Keep one unadvertised slot as
        // a rollback scratch for system-spec draft verification, plus a small
        // prefix-snapshot bank so partial MTP rejection can restore accepted
        // recurrent state without replaying the target model.
        const int q35_planned_slots = std::max<int>(1, mem_plan.state_slots);
        qwen3_5_runtime_rs_slots = std::max<int>(1, q35_planned_slots - 1);
        qwen3_5_scratch_rs_slot = qwen3_5_runtime_rs_slots;
        const int q35_spec_snapshot_slots = [] {
            const char* v = std::getenv("PIE_QWEN35_RS_SNAPSHOT_SLOTS");
            if (v == nullptr || v[0] == '\0') return 8;
            return std::clamp(std::atoi(v), 0, 16);
        }();
        const int q35_alloc_slots =
            qwen3_5_runtime_rs_slots + 1 + q35_spec_snapshot_slots;
        qwen3_5_state_cache = pie_cuda_driver::RecurrentStateCache::allocate(
            qwen3_5_layer_is_linear, conv_dim, cfg_q.linear_conv_kernel_dim,
            local_linear_value_heads,
            cfg_q.linear_key_head_dim,
            cfg_q.linear_value_head_dim,
            cfg_q.hidden_size,
            q35_alloc_slots);
        const std::size_t per_slot_recurrent_bytes =
            static_cast<std::size_t>(local_linear_value_heads) *
            cfg_q.linear_key_head_dim *
            cfg_q.linear_value_head_dim * sizeof(float);
        const std::size_t per_slot_conv_bytes =
            static_cast<std::size_t>(cfg_q.linear_conv_kernel_dim) *
            conv_dim * sizeof(std::uint16_t);
        const std::size_t num_linear_layers = qwen3_5_linear_layers;
        const std::size_t total_bytes = num_linear_layers *
            static_cast<std::size_t>(q35_alloc_slots) *
            (per_slot_recurrent_bytes + per_slot_conv_bytes);
        const std::size_t mtp_pending_bytes =
            static_cast<std::size_t>(q35_alloc_slots) *
            static_cast<std::size_t>(cfg_q.hidden_size) *
            sizeof(std::uint16_t);
        if (verbose) {
            std::cerr << "[pie-driver-cuda] qwen3.5 rs_cache: "
                      << num_linear_layers << " linear layers, "
                      << qwen3_5_runtime_rs_slots
                      << " runtime slots + 1 scratch + "
                      << q35_spec_snapshot_slots << " prefix snapshots, "
                      << (per_slot_recurrent_bytes + per_slot_conv_bytes)
                      << " B/slot (recurrent="
                      << per_slot_recurrent_bytes << " conv="
                      << per_slot_conv_bytes << "), mtp_pending="
                      << (mtp_pending_bytes / (1024 * 1024)) << " MiB, total ~"
                      << ((total_bytes + mtp_pending_bytes) / (1024 * 1024))
                      << " MiB\n";
        }

        if (is_qwen3_5_moe_arch) {
            qwen3_5_moe_ws = pie_cuda_driver::model::Qwen3_5MoeMlpWorkspace::allocate(
                max_workspace_tokens,
                cfg_q.hidden_size,
                cfg_q.num_experts,
                cfg_q.num_experts_per_tok,
                cfg_q.moe_intermediate_size / q35_tp_size,
                cfg_q.shared_expert_intermediate_size / q35_tp_size);
        }
    } else if (is_nemotron_h_arch) {
        const auto& cfg_n = engine.hf_config();
        nemotron_h_ws = pie_cuda_driver::model::NemotronHWorkspace::allocate(
            cfg_n, max_workspace_tokens, local_tp_size);
        const bool shard_mamba =
            pie_cuda_driver::model::nemotron_h_tp_mamba_sharding_enabled(
                local_tp_size);
        const int local_mamba_heads = shard_mamba
            ? cfg_n.mamba_num_heads / local_tp_size
            : cfg_n.mamba_num_heads;
        const int local_mamba_groups = shard_mamba
            ? cfg_n.mamba_n_groups / local_tp_size
            : cfg_n.mamba_n_groups;
        const int m_intermediate =
            local_mamba_heads * cfg_n.mamba_head_dim;
        const int conv_dim =
            m_intermediate + 2 * local_mamba_groups * cfg_n.mamba_state_size;
        const int nemotron_max_slots = std::max<int>(1, allocated_state_slots);
        nemotron_h_state_cache =
            pie_cuda_driver::RecurrentStateCache::allocate_bf16_recurrent(
                nemotron_h_layer_is_mamba,
                conv_dim,
                cfg_n.mamba_conv_kernel,
                local_mamba_heads,
                cfg_n.mamba_head_dim,
                cfg_n.mamba_state_size,
                nemotron_max_slots);
        if (verbose) {
            const std::size_t slot_bytes =
                pie_cuda_driver::model::nemotron_h_state_slot_bytes(
                    cfg_n, nemotron_h_mamba_layers, local_tp_size);
            const std::size_t layer_slot_bytes =
                nemotron_h_mamba_layers > 0
                    ? slot_bytes /
                          static_cast<std::size_t>(nemotron_h_mamba_layers)
                    : 0;
            std::cerr << "[pie-driver-cuda] nemotron_h rs_cache: "
                      << nemotron_h_mamba_layers << " mamba layers, "
                      << runtime_state_slots << " runtime slots"
                      << (graph_pad_slot >= 0 ? " (+1 graph pad slot), " : ", ")
                      << layer_slot_bytes
                      << " B/layer-slot, total ~"
                      << (static_cast<std::size_t>(nemotron_max_slots) *
                          slot_bytes / (1024 * 1024))
                      << " MiB\n";
        }
    }

    auto swap_pool = pie_cuda_driver::SwapPool::allocate_for_cache(
        kv_cache, static_cast<int>(cfg.batching.swap_pool_size));

    pie_cuda_driver::ops::CublasHandle cublas;
    auto runtime_quant_scratch = runtime_quant_scratch_base;
    runtime_quant_scratch.max_tokens =
        static_cast<std::size_t>(max_workspace_tokens);
    if (!runtime_quant_scratch.empty()) {
        pie_cuda_driver::ops::reserve_runtime_quant_scratch(
            runtime_quant_scratch,
            /*seal_after_reserve=*/true);
        CUDA_CHECK(cudaDeviceSynchronize());
        if (verbose) {
            std::cerr << "[pie-driver-cuda] runtime quant graph scratch: "
                      << (runtime_quant_scratch.has_fp8 ? "fp8" : "")
                      << (runtime_quant_scratch.has_fp8 &&
                          runtime_quant_scratch.has_int8 ? "+" : "")
                      << (runtime_quant_scratch.has_int8 ? "int8" : "")
                      << " max_tokens=" << runtime_quant_scratch.max_tokens
                      << " max_N=" << runtime_quant_scratch.max_weight_rows
                      << " max_K=" << runtime_quant_scratch.max_weight_cols
                      << " reserved="
                      << (mem_plan.runtime_quant_scratch_bytes /
                          (1024 * 1024))
                      << " MiB (sealed for CUDA graphs)\n";
        }
    }

    // Persistent input buffers, sized for the planned worst case so
    // device pointers stay stable across fires (prereq for graphs).
    auto persistent_inputs = pie_cuda_driver::PersistentInputs::allocate(
        max_workspace_tokens,
        /*max_requests=*/mem_plan.max_requests,
        /*max_kv_pages=*/mem_plan.max_page_refs,
        mem_plan.capacity.max_custom_mask_bytes);

    std::optional<pie_cuda_driver::model::Gemma4MtpWorkspace> gemma4_mtp_ws;
    if (gemma4_mtp_weights) {
        gemma4_mtp_ws.emplace(
            pie_cuda_driver::model::Gemma4MtpWorkspace::allocate(
                *gemma4_mtp_weights,
                mem_plan.max_requests,
                mem_plan.max_page_refs,
                native_mtp_num_drafts));
        if (verbose) {
            std::cerr << "[pie-driver-cuda] Gemma4 MTP system drafter enabled: "
                      << "drafts=" << native_mtp_num_drafts
                      << " max_requests=" << mem_plan.max_requests
                      << " page_refs=" << mem_plan.max_page_refs << "\n";
        }
    }

    pie_cuda_driver::CustomAllReduce custom_ar;
    if (tp_comm_ptr != nullptr && vtable_opt != nullptr &&
        cfg.distributed.tp_size == 2) {
        try {
            custom_ar = pie_cuda_driver::CustomAllReduce(
                *tp_comm_ptr, /*same_process=*/true,
                /*max_bytes=*/pie_cuda_driver::custom_all_reduce_max_bytes(),
                /*rank_data_bytes=*/8 * 1024 * 1024,
                /*fusion_max_tokens=*/mem_plan.max_requests,
                /*fusion_hidden=*/engine.hf_config().hidden_size);
            custom_ar.register_buffer(*tp_comm_ptr, ws.norm_x.data(),
                                      ws.norm_x.nbytes());
            custom_ar.register_buffer(*tp_comm_ptr, ws.norm_y.data(),
                                      ws.norm_y.nbytes());
            if (custom_ar) {
                tp_comm_ptr->set_custom_all_reduce(&custom_ar);
            }
        } catch (const std::exception& e) {
            custom_ar = pie_cuda_driver::CustomAllReduce();
            if (verbose) {
                std::cerr << "[pie-driver-cuda] custom all-reduce unavailable: "
                          << e.what() << "; falling back to NCCL\n";
            }
        }
    }

    if (verbose) {
        std::cerr << "[pie-driver-cuda] kv_cache: "
                  << runtime_kv_pages << " runtime pages";
        if (graph_pad_page >= 0) {
            std::cerr << " (+1 graph pad page)";
        }
        std::cerr << " × "
                  << kv_cache.page_size() << " tokens; "
                  << "format=" << kv_cache.format().name << "; "
                  << "workspace tokens=" << max_workspace_tokens
                  << "; max requests=" << mem_plan.max_requests
                  << "; page_refs=" << mem_plan.max_page_refs
                  << "; arena ~" << (mem_plan.arena_bytes / (1024 * 1024))
                  << " MiB"
                  << "; rq_scratch="
                  << (mem_plan.runtime_quant_scratch_bytes / (1024 * 1024))
                  << " MiB"
                  << "; attn_ws="
                  << (mem_plan.attn_float_workspace_bytes / (1024 * 1024))
                  << " MiB"
                  << "; swap_pool=" << swap_pool.num_pages() << " pages\n";
    }

    // Followers skip the server: rank 0 owns the fast path and broadcasts
    // each fire to followers via NCCL. tp_follower_serve (entered at the
    // end of run_impl) consumes those broadcasts and exits via
    // `tp_send_shutdown` from rank 0 once the next broadcast completes.
    const bool is_tp_follower =
        cfg.distributed.tp_size > 1 && cfg.distributed.tp_rank > 0;
    std::unique_ptr<pie_driver::InProcServer> server_p;
    if (!is_tp_follower && vtable_opt != nullptr) {
        // Response scratch lives in the per-backend `ResponseBuilder`
        // inside Executor — no central byte buffer on this path.
        server_p = std::make_unique<pie_driver::InProcServer>(*vtable_opt);
        pie_cuda_driver::register_server(server_p.get());
    } else if (!is_tp_follower && vtable_opt == nullptr) {
        // Parity-only invocation should have returned by now (the parity
        // branch above exits before reaching here). Falling through means
        // the caller didn't set parity flags — error out instead of
        // hanging without a server.
        std::cerr << "[pie-driver-cuda] standalone binary supports parity "
                     "tests only; embed via pie_driver_cuda_run_inproc\n";
        return 2;
    }

    if (install_signal_handlers) {
        std::signal(SIGINT, pie_cuda_driver::on_signal);
        std::signal(SIGTERM, pie_cuda_driver::on_signal);
    }

    std::uint64_t handled = 0;

    pie_cuda_driver::ForwardGraphCache graph_cache;

    // Per-arch forward knobs from the loaded HF config.
    pie_cuda_driver::model::LlamaLikeForwardCfg fwd_cfg{};
    pie_cuda_driver::model::Gemma2ForwardCfg gemma_fwd_cfg{};
    pie_cuda_driver::model::Gemma4ForwardCfg gemma4_fwd_cfg{};
    {
        const auto& hf = engine.hf_config();
        const std::string& mt = hf.model_type;
        fwd_cfg.use_qk_norm        = hf.use_qk_norm;
        fwd_cfg.use_qkv_bias       = hf.attention_bias;
        // OLMo-2 and OLMo-3 are the post-norm + q/k-norm architectures
        // bind_olmo3 materialises; everything else uses the standard
        // Llama pre-norm placement. q/k norms are forced on regardless
        // of the (sometimes missing) `use_qk_norm` config field.
        const bool is_olmo_post_norm = (mt == "olmo2" || mt == "olmo3");
        fwd_cfg.norm_placement = is_olmo_post_norm
            ? pie_cuda_driver::model::NormPlacement::Post
            : pie_cuda_driver::model::NormPlacement::Pre;
        if (is_olmo_post_norm) {
            fwd_cfg.use_qk_norm = true;
        }
        pie_cuda_driver::model::apply_rope_config(fwd_cfg, hf);
        fwd_cfg.sliding_window            = hf.sliding_window;
        // FlashInfer's decode dispatch set covers {1, 2, 3, 4, 8}. Other
        // GQA ratios use the prefill path for decode-only batches as well.
        const int gqa = hf.num_attention_heads / hf.num_key_value_heads;
        const bool gqa_in_decode_set = pie_cuda_driver::flashinfer_decode_supports_gqa(gqa);
        fwd_cfg.force_prefill_path = !gqa_in_decode_set;
        fwd_cfg.decode_plan_cuda_graph = use_cuda_graphs;
        // Tensor-parallel state. tp_comm == nullptr at tp_size == 1
        // keeps the original single-GPU branches in the forward kernels.
        fwd_cfg.tp_size = cfg.distributed.tp_size;
        fwd_cfg.tp_comm = tp_comm_ptr;
        fwd_cfg.emit_logits = (cfg.distributed.tp_rank == 0);
        {
            const int T = std::max(1, cfg.distributed.tp_size);
            const int local_q_heads = hf.num_attention_heads / T;
            const int local_kv_heads = hf.num_key_value_heads / T;
            fwd_cfg.use_xqa_decode =
                pie_cuda_driver::xqa_decode_enabled_by_env() &&
                pie_cuda_driver::ops::xqa_decode_bf16_supported(
                    local_q_heads, local_kv_heads, hf.head_dim_kernel,
                    mem_plan.kv_page_size, hf.sliding_window,
                    /*logits_soft_cap=*/0.f, /*sm_scale=*/-1.f) &&
                !pie_cuda_driver::has_non_full_attention_layers(hf);
            if (fwd_cfg.use_xqa_decode) {
                fwd_cfg.force_prefill_path = false;
                // Per-rank, per-device init of the selected XQA kernel's
                // smem attribute. FlashInfer sets this in a process-global
                // static initializer, which is not enough once TP ranks bind
                // different current devices.
                if (local_q_heads > 0 && local_kv_heads > 0 &&
                    local_q_heads % local_kv_heads == 0) {
                    pie_cuda_driver::ops::xqa_decode_bf16_warmup_current_device(
                        local_q_heads / local_kv_heads, mem_plan.kv_page_size);
                }
            }
        }

        // Gemma-2 / Gemma-3 forward knobs. `query_pre_attn_scalar` and
        // `final_logit_softcapping` come straight from the HF config —
        // see `loader/hf_config.cpp` for the parsing.
        gemma_fwd_cfg.query_pre_attn_scalar = hf.gemma_query_pre_attn_scalar;
        gemma_fwd_cfg.final_logit_softcap   = hf.gemma_final_logit_softcap;
        gemma_fwd_cfg.attn_logit_softcap    = hf.gemma_attn_logit_softcap;
        gemma_fwd_cfg.use_qk_norm           = (mt == "gemma3" || mt == "gemma3_text");
        gemma_fwd_cfg.force_prefill_path    = !gqa_in_decode_set;
        gemma_fwd_cfg.tp_size = cfg.distributed.tp_size;
        gemma_fwd_cfg.tp_comm = tp_comm_ptr;

        // Build the per-layer attention type → window_left + rope_theta
        // tables. Sliding layers get the configured window; full layers
        // pass -1 (kept for symmetry — flashinfer treats `-1` as "no
        // sliding"). For Gemma-3, sliding layers use the local-base
        // RoPE freq while full layers stick with `rope_theta`.
        const bool homogeneous = !pie_cuda_driver::has_non_full_attention_layers(hf);
        if (!homogeneous) {
            gemma_fwd_cfg.per_layer_window_left.reserve(hf.layer_types.size());
            gemma_fwd_cfg.per_layer_rope_theta.reserve(hf.layer_types.size());
            fwd_cfg.per_layer_window_left.reserve(hf.layer_types.size());
            for (const auto& t : hf.layer_types) {
                const bool is_sliding = (t == "sliding_attention");
                const int window = is_sliding ? hf.sliding_window : -1;
                gemma_fwd_cfg.per_layer_window_left.push_back(window);
                fwd_cfg.per_layer_window_left.push_back(window);
                const float theta =
                    (is_sliding && hf.rope_local_base_freq > 0.f)
                        ? hf.rope_local_base_freq
                        : hf.rope_theta;
                gemma_fwd_cfg.per_layer_rope_theta.push_back(theta);
            }
        }

        cudaDeviceProp serving_prop{};
        int serving_dev = 0;
        CUDA_CHECK(cudaGetDevice(&serving_dev));
        CUDA_CHECK(cudaGetDeviceProperties(&serving_prop, serving_dev));
        const bool prefill_decode_supported_head_dim =
            hf.head_dim_kernel == 64 || hf.head_dim_kernel == 128 ||
            hf.head_dim_kernel == 256 || hf.head_dim_kernel == 512;
        const bool force_prefill_decode_plan = [] {
            const char* v = std::getenv("PIE_CUDA_PREFILL_DECODE_PLAN");
            return v != nullptr && v[0] != '\0' && v[0] != '0';
        }();
        fwd_cfg.use_prefill_decode_plan =
            (serving_prop.major >= 9 || force_prefill_decode_plan) &&
            cfg.distributed.tp_size == 1 &&
            gqa_in_decode_set &&
            !fwd_cfg.force_prefill_path &&
            prefill_decode_supported_head_dim &&
            fwd_cfg.sliding_window < 0 &&
            fwd_cfg.per_layer_window_left.empty();
        if (fwd_cfg.use_prefill_decode_plan) {
            const std::size_t rank_kv_token_bytes =
                pie_cuda_driver::kv_page_bytes_homogeneous(
                    hf, std::max(1, cfg.distributed.tp_size), kv_format);
            const std::size_t global_kv_token_bytes =
                rank_kv_token_bytes *
                static_cast<std::size_t>(std::max(1, cfg.distributed.tp_size));
            const bool kv_heavy_attention =
                global_kv_token_bytes >= 192ull * 1024ull;
            // The dedicated decode kernel is faster for short KV histories.
            // Switch to the prefill-plan path only after the batch has enough
            // average KV pages for split-KV/full-attention work to pay for
            // itself.
            fwd_cfg.prefill_decode_min_kv_pages =
                kv_heavy_attention ? 1 : 7;
            if (const char* v = std::getenv("PIE_CUDA_PREFILL_DECODE_MIN_KV_PAGES")) {
                fwd_cfg.prefill_decode_min_kv_pages =
                    std::max(0, std::atoi(v));
            }
            fwd_cfg.prefill_decode_full_attention_min_requests = 256;
            fwd_cfg.prefill_decode_full_attention_min_kv_pages =
                kv_heavy_attention ? 1 : 7;
            if (const char* v = std::getenv("PIE_CUDA_PREFILL_DECODE_FULL_MIN_KV_PAGES")) {
                fwd_cfg.prefill_decode_full_attention_min_kv_pages =
                    std::max(0, std::atoi(v));
            }
            if (const char* v = std::getenv("PIE_CUDA_PREFILL_DECODE_NOGRAPHS")) {
                if (v[0] != '\0' && v[0] != '0') {
                    fwd_cfg.decode_plan_cuda_graph = false;
                }
            }
        }

        if (verbose) {
            const char* rope_name =
                (fwd_cfg.rope_kind == pie_cuda_driver::model::RopeKind::YaRN)
                    ? "yarn"
                    : (fwd_cfg.rope_kind ==
                       pie_cuda_driver::model::RopeKind::YaRNOriginal)
                          ? "yarn-original"
                          : "standard";
            std::cerr << "[pie-driver-cuda] model_type=" << mt
                      << " use_qk_norm=" << fwd_cfg.use_qk_norm
                      << " use_qkv_bias=" << fwd_cfg.use_qkv_bias
                      << " rope=" << rope_name
                      << " prefill_decode_plan="
                      << (fwd_cfg.use_prefill_decode_plan ? "on" : "off")
                      << " xqa_decode="
                      << (fwd_cfg.use_xqa_decode ? "on" : "off")
                      << " decode_plan_graph="
                      << (fwd_cfg.decode_plan_cuda_graph ? "on" : "off")
                      << " full_attn_min_R="
                      << fwd_cfg.prefill_decode_full_attention_min_requests
                      << "\n";
        }
    }

    if (is_gemma4_arch) {
        const auto& hf = engine.hf_config();
        gemma4_fwd_cfg.final_logit_softcap = hf.gemma_final_logit_softcap;
        const int gqa = hf.num_attention_heads / hf.num_key_value_heads;
        const bool gqa_in_decode_set = pie_cuda_driver::flashinfer_decode_supports_gqa(gqa);
        gemma4_fwd_cfg.force_prefill_path = !gqa_in_decode_set;
        gemma4_fwd_cfg.tp_size = cfg.distributed.tp_size;
        gemma4_fwd_cfg.tp_comm = tp_comm_ptr;
    }

    // Build the type-erased forward closure once. The captures live in
    // `main`'s scope (weights_*, fwd_cfg, gemma_fwd_cfg) and persist for
    // the lifetime of the server.
    pie_cuda_driver::ForwardFn forward_fn;
    pie_cuda_driver::NativeSystemDrafter system_drafter;
    // llama_plan moved into LlamaLikeModel (owned).
    // nemotron_h_plan moved into NemotronHModel (owned).
    std::unique_ptr<pie_cuda_driver::model::Gemma4Model> gemma4_model;
    std::unique_ptr<pie_cuda_driver::model::Gemma2Model> gemma2_model;
    std::unique_ptr<pie_cuda_driver::model::MixtralModel> mixtral_model;
    std::unique_ptr<pie_cuda_driver::model::Gemma3nModel> gemma3n_model;
    std::unique_ptr<pie_cuda_driver::model::Qwen35Model> qwen3_5_model;
    std::unique_ptr<pie_cuda_driver::model::Qwen35MoeModel> qwen3_5_moe_model;
    std::unique_ptr<pie_cuda_driver::model::LlamaLikeModel> llama_like_model;
    // Gemma-4 26B-A4B's MoE block needs a routed-experts workspace
    // alongside the dense forward state. Inert (zero-byte) on dense
    // E2B / E4B / 31B variants.
    pie_cuda_driver::model::Gemma4MoeMlpWorkspace gemma4_moe_ws;
    if (is_gemma4_arch && engine.hf_config().gemma4_enable_moe) {
        const auto& hf_cfg = engine.hf_config();
        gemma4_moe_ws = pie_cuda_driver::model::Gemma4MoeMlpWorkspace::allocate(
            max_workspace_tokens,
            hf_cfg.hidden_size,
            hf_cfg.num_experts,
            hf_cfg.num_experts_per_tok,
            hf_cfg.moe_intermediate_size /
                std::max(1, cfg.distributed.tp_size));
    }
    if (is_gemma4_arch) {
        gemma4_moe_ws.allocate_row_decode(max_workspace_tokens);
    }
    if (is_gemma4_arch &&
        engine.hf_config().gemma_hidden_size_per_layer_input > 0) {
        const auto& hf_cfg = engine.hf_config();
        gemma4_moe_ws.allocate_ple(
            max_workspace_tokens,
            hf_cfg.num_hidden_layers *
                hf_cfg.gemma_hidden_size_per_layer_input);
    }
    if (is_gemma4_arch) {
        gemma4_model = std::make_unique<pie_cuda_driver::model::Gemma4Model>(
            weights_gemma4, engine.hf_config(),
            gemma4_moe_ws, kv_cache,
            gemma4_fwd_cfg, pie_cuda_driver::model::qwen35_small_spec_graph_tokens());
        forward_fn.attach_model(gemma4_model.get());
    } else if (is_gemma3n_arch) {
        pie_cuda_driver::model::Gemma3nForwardCfg gemma3n_fwd_cfg{};
        gemma3n_fwd_cfg.final_logit_softcap = engine.hf_config().gemma_final_logit_softcap;
        gemma3n_fwd_cfg.tp_size = cfg.distributed.tp_size;
        gemma3n_fwd_cfg.tp_comm = tp_comm_ptr;
        gemma3n_model = std::make_unique<pie_cuda_driver::model::Gemma3nModel>(
            weights_gemma3n, engine.hf_config(), gemma3n_fwd_cfg);
        forward_fn.attach_model(gemma3n_model.get());
    } else if (is_gemma_arch) {
        gemma2_model = std::make_unique<pie_cuda_driver::model::Gemma2Model>(
            weights_gemma, engine.hf_config(), gemma_fwd_cfg);
        forward_fn.attach_model(gemma2_model.get());
    } else if (is_mixtral_arch) {
        mixtral_model = std::make_unique<pie_cuda_driver::model::MixtralModel>(
            weights_mixtral, engine.hf_config(), fwd_cfg,
            engine.hf_config().num_experts,
            engine.hf_config().num_experts_per_tok);
        forward_fn.attach_model(mixtral_model.get());
    } else if (is_nemotron_h_arch) {
        nemotron_h_model = std::make_unique<pie_cuda_driver::model::NemotronHModel>(
            weights_nemotron_h, engine.hf_config(),
            nemotron_h_ws, nemotron_h_state_cache, kv_cache,
            fwd_cfg, cfg.distributed.tp_size, tp_comm_ptr);
        forward_fn.attach_model(nemotron_h_model.get());
    } else if (is_qwen3_5_arch) {
        const int q35_tp_size = cfg.distributed.tp_size;
        pie_cuda_driver::NcclComm* q35_tp_comm = tp_comm_ptr;
        const auto& hf_q = engine.hf_config();
        const int gqa_q = hf_q.num_attention_heads /
                          std::max(1, hf_q.num_key_value_heads);
        qwen3_5_model = std::make_unique<pie_cuda_driver::model::Qwen35Model>(
            weights_qwen3_5, hf_q,
            qwen3_5_la_ws, qwen3_5_state_cache, qwen3_5_plan_state,
            kv_cache, q35_tp_size, q35_tp_comm,
            /*force_prefill_path=*/!pie_cuda_driver::flashinfer_decode_supports_gqa(gqa_q),
            /*small_prefill_naive_attention_max_tokens=*/pie_cuda_driver::model::qwen35_small_spec_graph_tokens(),
            /*graph_safe=*/kv_cache.format().is_native_bf16() &&
                !pie_cuda_driver::model::qwen35_forward_profile_enabled(),
            /*supports_small_prefill_graph=*/
                kv_cache.format().is_native_bf16() && !kv_cache.hnd_layout() &&
                pie_cuda_driver::model::qwen35_small_spec_graph_tokens() > 0);
        forward_fn.attach_model(qwen3_5_model.get());
        if (weights_qwen3_5.mtp.has_value() && native_mtp_num_drafts > 0) {
            qwen3_5_model->wire_system_drafter(
                system_drafter, native_mtp_num_drafts,
                pie_cuda_driver::model::qwen35_mtp_draft_position_offset(),
                pie_cuda_driver::model::qwen35_mtp_prefix_global_cache(),
                pie_cuda_driver::model::qwen35_mtp_fused_gemv_enabled());
        }
    } else if (is_qwen3_5_moe_arch) {
        const int q35moe_tp_size = cfg.distributed.tp_size;
        pie_cuda_driver::NcclComm* q35moe_tp_comm = tp_comm_ptr;
        const auto& hf_q = engine.hf_config();
        const int gqa_q_moe = hf_q.num_attention_heads /
                              std::max(1, hf_q.num_key_value_heads);
        qwen3_5_moe_model = std::make_unique<pie_cuda_driver::model::Qwen35MoeModel>(
            weights_qwen3_5_moe, hf_q,
            qwen3_5_la_ws, qwen3_5_moe_ws,
            qwen3_5_state_cache, qwen3_5_plan_state,
            kv_cache, q35moe_tp_size, q35moe_tp_comm,
            /*force_prefill_path=*/!pie_cuda_driver::flashinfer_decode_supports_gqa(gqa_q_moe),
            /*small_prefill_naive_attention_max_tokens=*/pie_cuda_driver::model::qwen35_small_spec_graph_tokens(),
            /*graph_safe=*/[]{
                const char* env = std::getenv("PIE_QWEN35_MOE_PROFILE");
                return !(env != nullptr && env[0] != '\0' && env[0] != '0');
            }(),
            /*supports_small_prefill_graph=*/
                kv_cache.format().is_native_bf16() && !kv_cache.hnd_layout() &&
                pie_cuda_driver::model::qwen35_small_spec_graph_tokens() > 0);
        forward_fn.attach_model(qwen3_5_moe_model.get());
        if (weights_qwen3_5_moe.mtp.has_value() && native_mtp_num_drafts > 0) {
            qwen3_5_moe_model->wire_system_drafter(
                system_drafter, native_mtp_num_drafts,
                pie_cuda_driver::model::qwen35_mtp_draft_position_offset(),
                pie_cuda_driver::model::qwen35_mtp_prefix_global_cache());
        }
    } else {
        // Llama-like fallback: covers Qwen3, Mixtral, Mistral3, GPT-OSS,
        // Gemma2, and any other shape that binds Qwen3Weights and routes
        // through llama_like_forward_paged.
        const bool supports_tp_greedy_argmax =
            cfg.distributed.tp_size > 1 &&
            weights_llama.lm_head_tp_shard != nullptr;
        llama_like_model = std::make_unique<pie_cuda_driver::model::LlamaLikeModel>(
            weights_llama, engine.hf_config(), kv_cache,
            fwd_cfg, supports_tp_greedy_argmax);
        forward_fn.attach_model(llama_like_model.get());
    }

    if (gemma4_mtp_weights && gemma4_mtp_ws) {
        system_drafter.max_drafts = native_mtp_num_drafts;
        system_drafter.draft_next =
            [&weights_gemma4, &mtp_w = *gemma4_mtp_weights,
             &mtp_ws = *gemma4_mtp_ws, gemma4_mtp_runtime](
                const pie_cuda_driver::SystemSpecDraftInputs& in,
                std::span<pie_driver::PerRequestOutput> per_req) {
                pie_cuda_driver::model::gemma4_mtp_draft(
                    mtp_w, weights_gemma4, mtp_ws, gemma4_mtp_runtime,
                    in, per_req);
            };
    }

    pie_cuda_driver::Executor executor{
        engine, ws, kv_cache, attn_ws, cublas,
        max_workspace_tokens,
        mem_plan.max_requests,
        graph_pad_page,
        graph_pad_slot,
        persistent_inputs, verbose, std::move(forward_fn),
        std::move(system_drafter),
        use_cuda_graphs ? &graph_cache : nullptr,
        /*tp_comm=*/tp_comm_ptr,
        /*tp_cpu_gate_key=*/{},
        /*rs_cache=*/((is_qwen3_5_arch || is_qwen3_5_moe_arch)
                          ? &qwen3_5_state_cache
                          : (is_nemotron_h_arch ? &nemotron_h_state_cache
                                                 : nullptr)),
        /*rs_cache_scratch_slot=*/qwen3_5_scratch_rs_slot,
        /*response_builder=*/{},
    };
    executor.tp_cpu_gate_key = cfg.distributed.nccl_unique_id_hex;
    // Pass-level speculation is runtime-owned. `.system_speculation()` is
    // driver-owned when a native drafter is configured.
    if (verbose && use_cuda_graphs) {
        std::cerr << "[pie-driver-cuda] CUDA graphs enabled (experimental)\n";
    }

    // TP ranks run as independent driver instances. Followers can reach the
    // first NCCL receive before rank 0 has finished building its CUDA serving
    // state; posting that idle receive while the leader is still allocating can
    // show as a persistent 100% GPU-util spin and has reproduced startup
    // wedges. Rendezvous on CPU after all persistent allocations are complete,
    // then pre-capture any graph-safe decode lattice before rank 0 publishes
    // readiness and followers enter the NCCL loop.
    pie_cuda_driver::tp_startup_cpu_barrier(cfg);
    if (use_cuda_graphs) {
        pie_cuda_driver::capture_forward_graph_lattice(executor);
    }
    pie_cuda_driver::tp_startup_cpu_barrier(cfg);

    if (is_tp_follower) {
        if (verbose) {
            std::cerr << "[pie-driver-cuda] tp follower rank "
                      << cfg.distributed.tp_rank
                      << " ready (waiting on rank-0 broadcasts"
                      << (executor.tp_cpu_gate_key.empty()
                              ? ", cpu_gate=off"
                              : ", cpu_gate=on")
                      << ")\n";
        }
        // Followers: block on rank-0 broadcasts until shutdown.
        std::atomic<bool> stop{false};
        pie_cuda_driver::tp_follower_serve(executor, stop);
    } else {
        // Capabilities reflect both the loaded HF config and the live
        // KV cache. Only rank 0 reports — the wrapper expects exactly
        // one READY per TP group.
        auto c = engine.capabilities();
        c.total_pages = runtime_kv_pages;
        c.swap_pool_size = swap_pool.num_pages();
        const bool rs_cache_required =
            ((is_qwen3_5_arch || is_qwen3_5_moe_arch) &&
             runtime_state_slots > 0) ||
            (is_nemotron_h_arch && runtime_state_slots > 0);
        const std::uint64_t rs_cache_slots = rs_cache_required
            ? (is_nemotron_h_arch
                   ? static_cast<std::uint64_t>(runtime_state_slots)
                   : static_cast<std::uint64_t>(qwen3_5_runtime_rs_slots))
            : 0;
        const std::uint64_t rs_cache_slot_bytes = rs_cache_required
            ? (is_nemotron_h_arch
                   ? static_cast<std::uint64_t>(
                         pie_cuda_driver::model::nemotron_h_state_slot_bytes(
                             engine.hf_config(), nemotron_h_mamba_layers,
                             local_tp_size))
                   : static_cast<std::uint64_t>(qwen3_5_linear_layers) *
                             (qwen3_5_state_cache.conv_slot_stride_bytes() +
                              qwen3_5_state_cache.recurrent_slot_stride_bytes()) +
                         static_cast<std::uint64_t>(
                             std::max(0, qwen3_5_state_cache.hidden_size())) *
                             sizeof(std::uint16_t))
            : 0;
        const bool rs_cache_spec_rollback =
            rs_cache_required && cfg.distributed.tp_size <= 1 &&
            qwen3_5_scratch_rs_slot >= 0;
        const bool system_speculation_supported =
            static_cast<bool>(executor.system_drafter);
        const auto max_forward_requests_caps = rs_cache_required
            ? std::min<std::uint64_t>(
                  static_cast<std::uint64_t>(mem_plan.capacity.max_forward_requests),
                  rs_cache_slots)
            : static_cast<std::uint64_t>(mem_plan.capacity.max_forward_requests);
        nlohmann::json caps = {
            {"total_pages",            c.total_pages},
            {"kv_page_size",           mem_plan.kv_page_size},
            {"swap_pool_size",         c.swap_pool_size},
            {"rs_cache_required",      rs_cache_required},
            {"rs_cache_slots",         rs_cache_slots},
            {"rs_cache_slot_bytes",    rs_cache_slot_bytes},
            {"rs_cache_spec_rollback", rs_cache_spec_rollback},
            {"system_speculation_supported", system_speculation_supported},
            {"default_system_speculation", system_speculation_supported},
            {"max_forward_tokens",     mem_plan.capacity.max_forward_tokens},
            {"max_forward_requests",   max_forward_requests_caps},
            {"max_page_refs",          mem_plan.capacity.max_page_refs},
            {"max_logit_rows",         mem_plan.capacity.max_logit_rows},
            {"max_prob_rows",          mem_plan.capacity.max_prob_rows},
            {"max_custom_mask_bytes",  mem_plan.capacity.max_custom_mask_bytes},
            {"max_sampler_rows",       mem_plan.capacity.max_sampler_rows},
            {"max_logprob_labels",     mem_plan.capacity.max_logprob_labels},
            {"arch_name",              c.arch_name},
            {"vocab_size",             c.vocab_size},
            {"max_model_len",          c.max_model_len},
            {"activation_dtype",       c.activation_dtype},
            {"snapshot_dir",           c.snapshot_dir},
        };
        if (verbose) {
            std::cerr << "[pie-driver-cuda] forward_limits: "
                      << "tokens=" << mem_plan.capacity.max_forward_tokens
                      << " requests=" << mem_plan.capacity.max_forward_requests
                      << " page_refs=" << mem_plan.capacity.max_page_refs
                      << " logit_rows=" << mem_plan.capacity.max_logit_rows
                      << " prob_rows=" << mem_plan.capacity.max_prob_rows
                      << " custom_mask_bytes="
                      << mem_plan.capacity.max_custom_mask_bytes
                      << " sampler_rows=" << mem_plan.capacity.max_sampler_rows
                      << " logprob_labels="
                      << mem_plan.capacity.max_logprob_labels
                      << "\n";
        }
        const std::string caps_json = caps.dump();
        ready_cb(caps_json.c_str(), ready_ctx);

        if (verbose) {
            std::cerr << "[pie-driver-cuda] serving on in-process channel\n";
        }
        pie_cuda_driver::service::InProcService service{
            executor, kv_cache, swap_pool};
        service.serve_forever(*server_p);
        handled = service.handled();
        // Leader exited serve loop — wake followers so they can tear
        // down cleanly.
        if (cfg.distributed.tp_size > 1) {
            pie_cuda_driver::tp_send_shutdown(
                *tp_comm_ptr, executor.tp_cpu_gate_key);
        }
    }

    if (server_p) {
        pie_cuda_driver::unregister_server(server_p.get());
    }
    if (verbose) {
        std::cerr << "[pie-driver-cuda] shutting down (handled " << handled
                  << " requests)\n";
    }
    return 0;
}

}  // namespace

// Standalone-binary entry. Now parity-test-only — if `--parity-tokens` is
// supplied the engine runs one forward pass and exits; otherwise we
// error out (use `pie_driver_cuda_run_inproc` for serve). The standalone
// `pie_driver_cuda` executable exists solely to host the parity tests
// under `driver/cuda/tests/`.
extern "C" int pie_driver_cuda_run(int argc,
                                   char** argv,
                                   int install_signal_handlers,
                                   pie_driver_cuda_ready_cb ready_cb,
                                   void* ready_ctx) {
    try {
        return run_impl(argc, argv, install_signal_handlers, ready_cb, ready_ctx,
                        /*vtable_opt=*/nullptr);
    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-cuda] fatal: " << e.what() << "\n";
        return -1;
    } catch (...) {
        std::cerr << "[pie-driver-cuda] fatal: unknown exception\n";
        return -1;
    }
}

extern "C" int pie_driver_cuda_run_inproc(int argc,
                                          char** argv,
                                          int install_signal_handlers,
                                          pie_driver_cuda_ready_cb ready_cb,
                                          void* ready_ctx,
                                          pie_driver::PieInProcVTable vtable) {
    try {
        return run_impl(argc, argv, install_signal_handlers, ready_cb, ready_ctx,
                        &vtable);
    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-cuda] fatal: " << e.what() << "\n";
        return -1;
    } catch (...) {
        std::cerr << "[pie-driver-cuda] fatal: unknown exception\n";
        return -1;
    }
}

// Reaches into the same server registry the SIGINT/SIGTERM handler uses.
// One host process can embed multiple same-flavor DP replicas, so stop
// every live driver server (shmem or inproc) rather than only the most
// recently registered one.
extern "C" void pie_driver_cuda_request_stop(void) {
    pie_cuda_driver::stop_servers();
}
