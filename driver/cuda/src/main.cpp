#include "model/mistral3.hpp"
// pie_driver_cuda — native CUDA backend, sibling to ../../runtime (Rust)
// and ../../pie (Python driver). Consumed by `pie_driver_cuda_native`.
//
// Currently a scaffold: opens the shmem fast path, decodes incoming
// `BatchedForwardPassRequest`s, logs them, and replies with an empty payload.
// Model loading + flashinfer-backed forward pass are the next milestones.

#include <algorithm>
#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <span>
#include <string>
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
#include "control_socket.hpp"
#include "cuda_check.hpp"
#include "engine.hpp"
#include "kernels/argmax.hpp"
#include "kernels/sample_flashinfer.hpp"
#include "kernels/sample_temp.hpp"
#include "kv_cache.hpp"
#include "model/gemma2.hpp"
#include "model/gemma4.hpp"
#include "model/llama_like.hpp"
#include "model/qwen3.hpp"
#include "model/qwen3_forward.hpp"
#include "swap_pool.hpp"
#include <thread>
#include <unistd.h>
#include "ops/gemm.hpp"
#include "request_handler.hpp"
#include "shmem_ipc.hpp"

namespace {

std::atomic<pie_cuda_driver::ShmemServer*> g_server{nullptr};

void on_signal(int) {
    if (auto* s = g_server.load()) s->stop();
}

}  // namespace

namespace {

// Run a one-shot forward pass on a binary file of i32 token ids and dump
// the last token's logits (bf16, [vocab]) to `logits_out`. Used by the
// numeric-parity harness; never invoked through the shmem path.
int run_parity(const pie_cuda_driver::Config& cfg,
               const std::string& tokens_in,
               const std::string& logits_out,
               bool paged)
{
    auto engine = pie_cuda_driver::Engine::load(cfg);
    {
        const auto& mt = engine.hf_config().model_type;
        const bool supported =
            mt == "qwen3" || mt == "qwen3_5"
         || mt == "qwen2"
         || mt == "llama" || mt == "llama3"
         || mt == "mistral" || mt == "mistral3";
        if (!supported) {
            std::cerr << "[parity] unsupported model_type: " << mt << "\n";
            return 2;
        }
    }
    const auto weights = pie_cuda_driver::model::bind_llama_like(engine);

    // Read tokens from disk.
    std::vector<std::int32_t> host_tokens;
    {
        std::ifstream in(tokens_in, std::ios::binary);
        if (!in) { std::cerr << "cannot open " << tokens_in << "\n"; return 3; }
        in.seekg(0, std::ios::end);
        const auto bytes = in.tellg();
        in.seekg(0, std::ios::beg);
        if (bytes <= 0 || bytes % 4 != 0) {
            std::cerr << "[parity] " << tokens_in << " is not a multiple of 4 bytes\n";
            return 3;
        }
        host_tokens.resize(static_cast<std::size_t>(bytes) / 4);
        in.read(reinterpret_cast<char*>(host_tokens.data()), bytes);
    }
    const int N = static_cast<int>(host_tokens.size());
    std::cerr << "[parity] running forward on " << N << " tokens\n";

    std::vector<std::int32_t> host_positions(N);
    for (int i = 0; i < N; ++i) host_positions[i] = i;

    // Upload to device.
    std::int32_t* d_tokens = nullptr;
    std::int32_t* d_positions = nullptr;
    CUDA_CHECK(cudaMalloc(&d_tokens, sizeof(std::int32_t) * N));
    CUDA_CHECK(cudaMalloc(&d_positions, sizeof(std::int32_t) * N));
    CUDA_CHECK(cudaMemcpy(d_tokens, host_tokens.data(), sizeof(std::int32_t) * N,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_positions, host_positions.data(), sizeof(std::int32_t) * N,
                          cudaMemcpyHostToDevice));

    auto ws = pie_cuda_driver::model::Qwen3Workspace::allocate(engine.hf_config(), N);
    pie_cuda_driver::ops::CublasHandle cublas;

    if (paged) {
        // Build a single-request paged layout that mirrors what the runtime
        // would send for a fresh request: pages [0..ceil(N/page_size)],
        // last_page_len computed accordingly.
        const int page_size = static_cast<int>(cfg.batching.kv_page_size);
        const int num_pages_needed = (N + page_size - 1) / page_size;

        auto cache = pie_cuda_driver::KvCache::allocate(
            engine.hf_config().num_hidden_layers,
            std::max(num_pages_needed, 1),
            page_size,
            engine.hf_config().num_key_value_heads,
            engine.hf_config().head_dim);

        std::vector<std::uint32_t> h_qo_indptr      = {0u, static_cast<std::uint32_t>(N)};
        std::vector<std::uint32_t> h_kv_page_indptr = {0u, static_cast<std::uint32_t>(num_pages_needed)};
        std::vector<std::uint32_t> h_kv_page_indices(num_pages_needed);
        for (int i = 0; i < num_pages_needed; ++i) h_kv_page_indices[i] = static_cast<std::uint32_t>(i);
        std::vector<std::uint32_t> h_kv_last_page_lens = {
            static_cast<std::uint32_t>(((N - 1) % page_size) + 1)
        };

        std::uint32_t *d_qo, *d_pi, *d_pp, *d_lpl;
        CUDA_CHECK(cudaMalloc(&d_qo,  4 * h_qo_indptr.size()));
        CUDA_CHECK(cudaMalloc(&d_pi,  4 * h_kv_page_indices.size()));
        CUDA_CHECK(cudaMalloc(&d_pp,  4 * h_kv_page_indptr.size()));
        CUDA_CHECK(cudaMalloc(&d_lpl, 4 * h_kv_last_page_lens.size()));
        CUDA_CHECK(cudaMemcpy(d_qo,  h_qo_indptr.data(),         4 * h_qo_indptr.size(),         cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pi,  h_kv_page_indices.data(),   4 * h_kv_page_indices.size(),   cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_pp,  h_kv_page_indptr.data(),    4 * h_kv_page_indptr.size(),    cudaMemcpyHostToDevice));
        CUDA_CHECK(cudaMemcpy(d_lpl, h_kv_last_page_lens.data(), 4 * h_kv_last_page_lens.size(), cudaMemcpyHostToDevice));

        // Parity harness uses the naive paged path (single prefill, qo_len=N).
        // flashinfer's decode kernel only supports qo_len==1; we'll add a
        // separate decode-shaped parity test in Phase 1.4.
        auto parity_attn_ws = pie_cuda_driver::AttentionWorkspace::allocate();
        pie_cuda_driver::model::qwen3_forward_paged(
            weights, engine.hf_config(), ws, cache, parity_attn_ws, cublas,
            d_tokens, d_positions,
            d_qo, d_pi, d_pp, d_lpl,
            /*qo_indptr_h=*/h_qo_indptr.data(),
            /*kv_page_indptr_h=*/h_kv_page_indptr.data(),
            /*total_tokens=*/N, /*num_requests=*/1,
            /*is_pure_decode=*/false);

        cudaFree(d_qo); cudaFree(d_pi); cudaFree(d_pp); cudaFree(d_lpl);
    } else {
        pie_cuda_driver::model::qwen3_forward_prefill(
            weights, engine.hf_config(), ws, cublas, d_tokens, d_positions, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Greedy sample over all rows on the GPU, then echo the last-token
    // id to stderr — the parity harness picks it up and cross-checks
    // against numpy.argmax of the dumped logits.
    {
        const int V = engine.hf_config().vocab_size;
        std::int32_t* d_sampled = nullptr;
        CUDA_CHECK(cudaMalloc(&d_sampled, sizeof(std::int32_t) * N));
        pie_cuda_driver::kernels::launch_argmax_bf16(
            ws.logits.data(), d_sampled, N, V, /*stream=*/nullptr);
        std::vector<std::int32_t> host_sampled(N);
        CUDA_CHECK(cudaMemcpy(host_sampled.data(), d_sampled,
                              sizeof(std::int32_t) * N, cudaMemcpyDeviceToHost));
        cudaFree(d_sampled);
        std::cerr << "[parity] gpu argmax last-token id = "
                  << host_sampled.back() << "\n";
    }

    // Copy last-token logits row out as bf16 (we'll convert in Python).
    const int V = engine.hf_config().vocab_size;
    std::vector<std::uint16_t> host_logits(V);  // bf16 viewed as u16
    const auto* base = static_cast<const std::uint16_t*>(ws.logits.data());
    CUDA_CHECK(cudaMemcpy(host_logits.data(),
                          base + static_cast<std::size_t>(N - 1) * V,
                          V * sizeof(std::uint16_t),
                          cudaMemcpyDeviceToHost));

    {
        std::ofstream out(logits_out, std::ios::binary);
        if (!out) { std::cerr << "cannot open " << logits_out << "\n"; return 4; }
        out.write(reinterpret_cast<const char*>(host_logits.data()),
                  host_logits.size() * 2);
    }
    std::cerr << "[parity] wrote " << V << " bf16 logits to " << logits_out << "\n";

    cudaFree(d_tokens);
    cudaFree(d_positions);
    return 0;
}

}  // namespace

int main(int argc, char** argv) {
    CLI::App app{"pie_driver_cuda — native CUDA backend for Pie"};
    std::string config_path = "dev.toml";
    app.add_option("-c,--config", config_path, "Path to TOML config")
        ->check(CLI::ExistingFile);

    std::string parity_tokens, parity_out;
    bool parity_paged = false;
    auto* parity = app.add_option_group("parity", "Numeric-parity test entry");
    parity->add_option("--parity-tokens", parity_tokens,
                       "Path to a binary file of i32 token ids");
    parity->add_option("--parity-out", parity_out,
                       "Where to write the last-token logits as bf16 [vocab]");
    parity->add_flag("--parity-paged", parity_paged,
                     "Run the paged forward path (BPIQ-shaped KV layout)");

    bool use_cuda_graphs = false;
    app.add_flag("--cuda-graphs", use_cuda_graphs,
                 "Capture decode forward into CUDA graphs and replay per "
                 "shape bucket. Experimental; default off.");

    int control_fd = -1;
    app.add_option("--control-fd", control_fd,
                   "Pre-opened SOCK_SEQPACKET fd for the wrapper control "
                   "channel (copy_d2h / copy_h2d / copy_d2d / copy_h2h).");
    CLI11_PARSE(app, argc, argv);

    const auto cfg = pie_cuda_driver::load_config(config_path);

    if (!parity_tokens.empty()) {
        if (parity_out.empty()) {
            std::cerr << "--parity-tokens requires --parity-out\n";
            return 1;
        }
        return run_parity(cfg, parity_tokens, parity_out, parity_paged);
    }

    // Informational logs go to stderr — stdout is reserved for the READY
    // handshake line consumed by the Python wrapper.
    std::cerr << "[pie-driver-cuda] config loaded\n"
              << "  shmem.name      = " << cfg.shmem.name << "\n"
              << "  shmem.num_slots = " << cfg.shmem.num_slots << "\n"
              << "  model.hf_repo   = " << cfg.model.hf_repo << "\n"
              << "  model.device    = " << cfg.model.device << "\n"
              << "  model.dtype     = " << cfg.model.dtype << "\n";

    auto engine = pie_cuda_driver::Engine::load(cfg);

    {
        const auto& mt = engine.hf_config().model_type;
        // Llama-like family. Same RMSNorm + RoPE + GQA + SwiGLU graph; the
        // only branch is whether per-head q/k_norm exists (Qwen3 quirk),
        // which is captured in HfConfig.use_qk_norm.
        const bool supported =
            mt == "qwen3" || mt == "qwen3_5"
         || mt == "qwen2"
         || mt == "llama" || mt == "llama3"
         || mt == "mistral" || mt == "mistral3"
         || mt == "phi3"
         || mt == "olmo" || mt == "olmo3"
         || mt == "gemma2"
         || mt == "gemma3" || mt == "gemma3_text"
         || mt == "gemma4" || mt == "gemma4_text";
        if (!supported) {
            std::cerr << "[pie-driver-cuda] arch '" << mt
                      << "' not yet supported (Qwen 2/3, Llama-3, "
                      << "Mistral, Phi-3, OLMo-3, Gemma-2/3/4)\n";
            return 2;
        }
    }
    const std::string& mt_for_bind = engine.hf_config().model_type;

    // Per-arch weights live in their own struct shape. We hold one of
    // them on the stack; only the one matching `mt_for_bind` is
    // populated. The dispatch below wraps the right (weights, cfg,
    // forward function) triple into a single `ForwardFn` closure.
    pie_cuda_driver::model::Qwen3Weights  weights_llama;
    pie_cuda_driver::model::Gemma2Weights weights_gemma;
    pie_cuda_driver::model::Gemma4Weights weights_gemma4;
    const bool is_gemma_arch =
        (mt_for_bind == "gemma2" || mt_for_bind == "gemma3" ||
         mt_for_bind == "gemma3_text");
    const bool is_gemma4_arch =
        (mt_for_bind == "gemma4" || mt_for_bind == "gemma4_text");

    if (mt_for_bind == "phi3") {
        weights_llama = pie_cuda_driver::model::bind_phi3(engine);
    } else if (mt_for_bind == "olmo3") {
        weights_llama = pie_cuda_driver::model::bind_olmo3(engine);
    } else if (mt_for_bind == "mistral3") {
        weights_llama = pie_cuda_driver::model::bind_mistral3(engine);
    } else if (mt_for_bind == "gemma2") {
        weights_gemma = pie_cuda_driver::model::bind_gemma2(engine);
    } else if (mt_for_bind == "gemma3" || mt_for_bind == "gemma3_text") {
        weights_gemma = pie_cuda_driver::model::bind_gemma3(engine);
    } else if (is_gemma4_arch) {
        weights_gemma4 = pie_cuda_driver::model::bind_gemma4(engine);
    } else {
        weights_llama = pie_cuda_driver::model::bind_llama_like(engine);
    }
    const std::size_t num_layers_bound =
        is_gemma4_arch ? weights_gemma4.layers.size()
      : is_gemma_arch  ? weights_gemma.layers.size()
                       : weights_llama.layers.size();
    std::cerr << "[pie-driver-cuda] schema bound: "
              << num_layers_bound << " layers ("
              << engine.hf_config().model_type
              << (engine.hf_config().use_qk_norm ? ", q/k norm" : "")
              << ")\n";

    // Pre-allocate persistent state for serving.
    //
    // - Workspace sized for `max_batch_tokens` from config, capped at 8192
    //   to keep the [N, vocab] logits + probs buffers under ~5 GiB combined
    //   (vocab=151936). M2.x will compute logits only at sampling rows so
    //   this can grow.
    // - KV cache sized at `max_num_kv_pages` × `kv_page_size`.
    const int max_workspace_tokens = std::min<int>(cfg.batching.max_batch_tokens, 8192);
    // Per-arch worst-case workspace dims. Gemma-4 has both
    // `use_double_wide_mlp` (intermediate doubles on shared layers)
    // and dual head_dim (sliding=256 vs full=512), so ws.q/k/v need
    // the full-attention sizing. Other archs use the single config
    // values.
    int max_mlp_intermediate = engine.hf_config().intermediate_size;
    int max_Hq = engine.hf_config().num_attention_heads * engine.hf_config().head_dim;
    int max_Hk = engine.hf_config().num_key_value_heads * engine.hf_config().head_dim;
    if (is_gemma4_arch) {
        for (int v : weights_gemma4.per_layer_intermediate) {
            if (v > max_mlp_intermediate) max_mlp_intermediate = v;
        }
        for (int d : weights_gemma4.per_layer_head_dim) {
            const int Hq = engine.hf_config().num_attention_heads * d;
            const int Hk = engine.hf_config().num_key_value_heads * d;
            if (Hq > max_Hq) max_Hq = Hq;
            if (Hk > max_Hk) max_Hk = Hk;
        }
    }
    auto ws = pie_cuda_driver::model::Qwen3Workspace::allocate_full(
        engine.hf_config(), max_workspace_tokens,
        max_mlp_intermediate, max_Hq, max_Hk);

    auto kv_cache =
        is_gemma4_arch
            ? pie_cuda_driver::KvCache::allocate_per_layer(
                  engine.hf_config().num_hidden_layers,
                  static_cast<int>(cfg.batching.max_num_kv_pages),
                  static_cast<int>(cfg.batching.kv_page_size),
                  engine.hf_config().num_key_value_heads,
                  weights_gemma4.per_layer_head_dim,
                  weights_gemma4.kv_source_layer)
            : pie_cuda_driver::KvCache::allocate(
                  engine.hf_config().num_hidden_layers,
                  static_cast<int>(cfg.batching.max_num_kv_pages),
                  static_cast<int>(cfg.batching.kv_page_size),
                  engine.hf_config().num_key_value_heads,
                  engine.hf_config().head_dim_kernel);

    auto attn_ws = pie_cuda_driver::AttentionWorkspace::allocate();

    auto swap_pool = pie_cuda_driver::SwapPool::allocate(
        engine.hf_config().num_hidden_layers,
        static_cast<int>(cfg.batching.swap_pool_size),
        static_cast<int>(cfg.batching.kv_page_size),
        engine.hf_config().num_key_value_heads,
        engine.hf_config().head_dim_kernel);

    pie_cuda_driver::ops::CublasHandle cublas;

    // Persistent input buffers, sized for the configured worst case so
    // device pointers stay stable across fires (prereq for graphs).
    // Worst-case mask is qo_len × kv_len bits per request. Bound:
    //   max_qo == max_workspace_tokens, max_kv == num_pages × page_size.
    const std::size_t max_kv_tokens =
        static_cast<std::size_t>(kv_cache.num_pages()) * kv_cache.page_size();
    const std::size_t max_mask_bits =
        static_cast<std::size_t>(max_workspace_tokens) * max_kv_tokens;
    const std::size_t max_mask_bytes = (max_mask_bits + 7) / 8;
    auto persistent_inputs = pie_cuda_driver::PersistentInputs::allocate(
        max_workspace_tokens,
        /*max_requests=*/max_workspace_tokens,  // each request has ≥1 token
        /*max_kv_pages=*/kv_cache.num_pages(),
        max_mask_bytes);

    std::cerr << "[pie-driver-cuda] kv_cache: "
              << kv_cache.num_pages() << " pages × "
              << kv_cache.page_size() << " tokens; "
              << "workspace tokens=" << max_workspace_tokens
              << "; swap_pool=" << swap_pool.num_pages() << " pages\n";

    // Cold-path control thread. Runtime → wrapper (RPC) → us (socketpair)
    // for KV swap operations. Gated on the wrapper having passed a valid
    // --control-fd; otherwise we just don't service swap requests, which is
    // fine when swap_pool_size==0 (admission control prevents the runtime
    // from issuing them).
    std::thread control_thread;
    if (control_fd >= 0) {
        control_thread = std::thread([&swap_pool, &kv_cache, control_fd] {
            pie_cuda_driver::serve_control_socket(
                control_fd,
                [&swap_pool, &kv_cache](
                    const pie_cuda_driver::CtrlRequest& req) -> std::uint32_t {
                    using namespace pie_cuda_driver;
                    std::span<const std::uint32_t> srcs(
                        req.src_dst_pairs, req.num_pairs);
                    std::span<const std::uint32_t> dsts(
                        req.src_dst_pairs + req.num_pairs, req.num_pairs);
                    // Pairs are laid out as [src_0, src_1, ..., dst_0, dst_1, ...]
                    // so we can pass two contiguous spans without rebuilding.
                    switch (req.method) {
                        case CTRL_METHOD_COPY_D2H:
                            swap_pool.copy_d2h(kv_cache, srcs, dsts);
                            return 0;
                        case CTRL_METHOD_COPY_H2D:
                            swap_pool.copy_h2d(kv_cache, srcs, dsts);
                            return 0;
                        case CTRL_METHOD_COPY_D2D:
                            swap_pool.copy_d2d(kv_cache, srcs, dsts);
                            return 0;
                        case CTRL_METHOD_COPY_H2H:
                            swap_pool.copy_h2h(srcs, dsts);
                            return 0;
                        default:
                            return 3;  // unknown method
                    }
                });
        });
    }

    pie_cuda_driver::ShmemServer server(
        cfg.shmem.name,
        cfg.shmem.num_slots,
        cfg.shmem.req_buf,
        cfg.shmem.resp_buf,
        cfg.shmem.spin_us);
    g_server.store(&server);

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    // Capabilities reflect both the loaded HF config and the live KV cache.
    auto c = engine.capabilities();
    c.total_pages = kv_cache.num_pages();
    c.max_batch_tokens = max_workspace_tokens;
    c.swap_pool_size = swap_pool.num_pages();
    nlohmann::json caps = {
        {"total_pages",      c.total_pages},
        {"kv_page_size",     c.kv_page_size},
        {"swap_pool_size",   c.swap_pool_size},
        {"max_batch_tokens", c.max_batch_tokens},
        {"max_batch_size",   c.max_batch_size},
        {"arch_name",        c.arch_name},
        {"vocab_size",       c.vocab_size},
        {"max_model_len",    c.max_model_len},
        {"activation_dtype", c.activation_dtype},
        {"snapshot_dir",     c.snapshot_dir},
        {"shmem_name",       cfg.shmem.name},
    };
    // The wrapper greps stdout for `^READY ` to complete the handshake.
    std::cout << "READY " << caps.dump() << std::endl;

    std::cerr << "[pie-driver-cuda] serving on shmem " << server.name()
              << " (" << server.num_slots() << " slots, "
              << "req_buf=" << server.req_buf_size() << ", "
              << "resp_buf=" << server.resp_buf_size() << ")\n";

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
        // OLMo-3 is the only currently-supported post-norm architecture;
        // its q/k norms and YaRN scaling are picked up via use_qk_norm /
        // has_rope_scaling on the same HfConfig.
        fwd_cfg.norm_placement = (mt == "olmo3")
            ? pie_cuda_driver::model::NormPlacement::Post
            : pie_cuda_driver::model::NormPlacement::Pre;
        if (mt == "olmo3") {
            fwd_cfg.use_qk_norm = true;  // OLMo-3 always has q/k norms.
        }
        fwd_cfg.rope_kind = hf.has_rope_scaling
            ? pie_cuda_driver::model::RopeKind::YaRN
            : pie_cuda_driver::model::RopeKind::Standard;
        fwd_cfg.yarn_factor               = hf.rope_factor;
        fwd_cfg.yarn_low_freq_factor      = hf.rope_low_freq_factor;
        fwd_cfg.yarn_high_freq_factor     = hf.rope_high_freq_factor;
        fwd_cfg.yarn_original_max_position = hf.rope_original_max_position;
        fwd_cfg.sliding_window            = hf.sliding_window;
        // flashinfer's decode kernel only instantiates GQA group sizes
        // {1, 2, 3, 4, 8}. Models like Qwen2-0.5B (group=7), Qwen2-1.5B
        // (group=6) need the prefill kernel even for decode-only fires.
        const int gqa = hf.num_attention_heads / hf.num_key_value_heads;
        const bool gqa_in_decode_set = (gqa == 1 || gqa == 2 || gqa == 3
                                        || gqa == 4 || gqa == 8);
        fwd_cfg.force_prefill_path = !gqa_in_decode_set;

        // Gemma-2 / Gemma-3 forward knobs. `query_pre_attn_scalar` and
        // `final_logit_softcapping` come straight from the HF config —
        // see `loader/hf_config.cpp` for the parsing.
        gemma_fwd_cfg.query_pre_attn_scalar = hf.gemma_query_pre_attn_scalar;
        gemma_fwd_cfg.final_logit_softcap   = hf.gemma_final_logit_softcap;
        gemma_fwd_cfg.attn_logit_softcap    = hf.gemma_attn_logit_softcap;
        gemma_fwd_cfg.use_qk_norm           = (mt == "gemma3" || mt == "gemma3_text");
        gemma_fwd_cfg.force_prefill_path    = !gqa_in_decode_set;

        // Build the per-layer attention type → window_left + rope_theta
        // tables. Sliding layers get the configured window; full layers
        // pass -1 (kept for symmetry — flashinfer treats `-1` as "no
        // sliding"). For Gemma-3, sliding layers use the local-base
        // RoPE freq while full layers stick with `rope_theta`.
        const bool homogeneous = hf.layer_types.empty();
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

        std::cerr << "[pie-driver-cuda] model_type=" << mt
                  << " use_qk_norm=" << fwd_cfg.use_qk_norm
                  << " use_qkv_bias=" << fwd_cfg.use_qkv_bias
                  << " rope=" << (fwd_cfg.rope_kind ==
                       pie_cuda_driver::model::RopeKind::YaRN ? "yarn" : "standard")
                  << "\n";
    }

    if (is_gemma4_arch) {
        const auto& hf = engine.hf_config();
        gemma4_fwd_cfg.final_logit_softcap = hf.gemma_final_logit_softcap;
        const int gqa = hf.num_attention_heads / hf.num_key_value_heads;
        const bool gqa_in_decode_set = (gqa == 1 || gqa == 2 || gqa == 3
                                        || gqa == 4 || gqa == 8);
        gemma4_fwd_cfg.force_prefill_path = !gqa_in_decode_set;
    }

    // Build the type-erased forward closure once. The captures live in
    // `main`'s scope (weights_*, fwd_cfg, gemma_fwd_cfg) and persist for
    // the lifetime of the server.
    pie_cuda_driver::ForwardFn forward_fn;
    if (is_gemma4_arch) {
        forward_fn = [&engine, &weights_gemma4, gemma4_fwd_cfg](
            pie_cuda_driver::model::Qwen3Workspace& ws,
            pie_cuda_driver::KvCache& cache,
            pie_cuda_driver::AttentionWorkspace& attn_ws,
            pie_cuda_driver::ops::CublasHandle& cublas,
            const std::int32_t* tok, const std::int32_t* pos,
            const std::uint32_t* qo_indptr,
            const std::uint32_t* kv_page_indices,
            const std::uint32_t* kv_page_indptr,
            const std::uint32_t* kv_last_page_lens,
            const std::uint32_t* qo_indptr_h,
            const std::uint32_t* kv_page_indptr_h,
            int N, int R, bool is_pure_decode,
            const std::uint8_t* mask_d, const std::int32_t* mask_indptr_d) {
            pie_cuda_driver::model::gemma4_forward_paged(
                weights_gemma4, engine.hf_config(), gemma4_fwd_cfg,
                ws, cache, attn_ws, cublas,
                tok, pos,
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                qo_indptr_h, kv_page_indptr_h,
                N, R, is_pure_decode, mask_d, mask_indptr_d);
        };
    } else if (is_gemma_arch) {
        forward_fn = [&engine, &weights_gemma, gemma_fwd_cfg](
            pie_cuda_driver::model::Qwen3Workspace& ws,
            pie_cuda_driver::KvCache& cache,
            pie_cuda_driver::AttentionWorkspace& attn_ws,
            pie_cuda_driver::ops::CublasHandle& cublas,
            const std::int32_t* tok, const std::int32_t* pos,
            const std::uint32_t* qo_indptr,
            const std::uint32_t* kv_page_indices,
            const std::uint32_t* kv_page_indptr,
            const std::uint32_t* kv_last_page_lens,
            const std::uint32_t* qo_indptr_h,
            const std::uint32_t* kv_page_indptr_h,
            int N, int R, bool is_pure_decode,
            const std::uint8_t* mask_d, const std::int32_t* mask_indptr_d) {
            pie_cuda_driver::model::gemma2_forward_paged(
                weights_gemma, engine.hf_config(), gemma_fwd_cfg,
                ws, cache, attn_ws, cublas,
                tok, pos,
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                qo_indptr_h, kv_page_indptr_h,
                N, R, is_pure_decode, mask_d, mask_indptr_d);
        };
    } else {
        forward_fn = [&engine, &weights_llama, fwd_cfg](
            pie_cuda_driver::model::Qwen3Workspace& ws,
            pie_cuda_driver::KvCache& cache,
            pie_cuda_driver::AttentionWorkspace& attn_ws,
            pie_cuda_driver::ops::CublasHandle& cublas,
            const std::int32_t* tok, const std::int32_t* pos,
            const std::uint32_t* qo_indptr,
            const std::uint32_t* kv_page_indices,
            const std::uint32_t* kv_page_indptr,
            const std::uint32_t* kv_last_page_lens,
            const std::uint32_t* qo_indptr_h,
            const std::uint32_t* kv_page_indptr_h,
            int N, int R, bool is_pure_decode,
            const std::uint8_t* mask_d, const std::int32_t* mask_indptr_d) {
            pie_cuda_driver::model::llama_like_forward_paged(
                weights_llama, engine.hf_config(), fwd_cfg,
                ws, cache, attn_ws, cublas,
                tok, pos,
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                qo_indptr_h, kv_page_indptr_h,
                N, R, is_pure_decode, mask_d, mask_indptr_d);
        };
    }

    pie_cuda_driver::ForwardContext fwd_ctx{
        engine, ws, kv_cache, attn_ws, cublas,
        max_workspace_tokens, persistent_inputs, std::move(forward_fn),
        use_cuda_graphs ? &graph_cache : nullptr,
    };
    if (use_cuda_graphs) {
        std::cerr << "[pie-driver-cuda] CUDA graphs enabled (experimental)\n";
    }

    server.serve_forever([&](const pie_cuda_driver::SlotRequest& req,
                             std::span<std::uint8_t> response) -> std::size_t {
        ++handled;
        if (req.method_tag != pie_cuda_driver::METHOD_TAG_FIRE_BATCH) {
            std::cerr << "[pie-driver-cuda] unsupported method_tag="
                      << req.method_tag << " req_id=" << req.req_id << "\n";
            return 0;
        }
        return pie_cuda_driver::handle_fire_batch(req, response, fwd_ctx, handled);
    });

    g_server.store(nullptr);
    std::cerr << "[pie-driver-cuda] shutting down (handled " << handled
              << " requests)\n";

    if (control_thread.joinable()) {
        // Closing the wrapper-side fd makes recv() return 0 and the thread
        // exits cleanly. We don't have that fd here, so close ours; SEQPACKET
        // will EOF on either side closing.
        ::close(control_fd);
        control_thread.join();
    }
    return 0;
}
