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
#include <csignal>
#include <cstdint>
#include <cstdio>
#include <iostream>
#include <memory>
#include <mutex>
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
#include "cuda_check.hpp"
#include "model/loaded_model.hpp"
#include "kernels/argmax.hpp"
#include "kernels/sample_flashinfer.hpp"
#include "kernels/sample_temp.hpp"
#include "kv_cache.hpp"
#include "model/gemma2.hpp"
#include "model/gemma3n.hpp"
#include "model/gemma4.hpp"
#include "model/gpt_oss.hpp"
#include "model/llama_like.hpp"
#include "model/mixtral.hpp"
#include "model/qwen3.hpp"
#include "model/qwen3_5.hpp"
#include "model/qwen3_5_forward.hpp"
#include "model/qwen3_5_moe.hpp"
#include "model/qwen3_5_moe_forward.hpp"
#include "model/qwen3_forward.hpp"
#include "qwen3_5_state_cache.hpp"
#include "swap_pool.hpp"
#include <thread>
#include <unistd.h>
#include "ops/gemm.hpp"
#include "executor/executor.hpp"
#include "service/inproc_service.hpp"
#include <pie_bridge/inproc_server.hpp>

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

// All TP ranks in one DP group are threads in the same pie-server
// process. Rendezvous via an in-process `std::barrier` keyed by the
// shared `nccl_unique_id_hex` (which is per-DP-group by construction).
// `nccl_unique_id_hex` doubles as `tp_cpu_gate_key` for the per-fire
// CPU gate downstream (executor/executor.cpp).
void tp_startup_cpu_barrier(const pie_cuda_driver::Config& cfg) {
    if (cfg.distributed.tp_size <= 1) return;

    const std::string& key = cfg.distributed.nccl_unique_id_hex;
    if (key.empty()) return;

    static std::mutex registry_mu;
    static std::unordered_map<std::string, std::shared_ptr<std::barrier<>>>
        registry;

    std::shared_ptr<std::barrier<>> b;
    {
        std::lock_guard<std::mutex> lk(registry_mu);
        auto& entry = registry[key];
        if (!entry) {
            entry = std::make_shared<std::barrier<>>(cfg.distributed.tp_size);
        }
        b = entry;
    }
    b->arrive_and_wait();
}

}  // namespace

namespace {

// Run a one-shot forward pass on a binary file of i32 token ids and dump
// the last token's logits (bf16, [vocab]) to `logits_out`. Used by the
// numeric-parity harness; never invoked through the shmem path.
int run_parity(const pie_cuda_driver::Config& cfg,
               const std::string& tokens_in,
               const std::string& logits_out,
               bool paged,
               bool decode_after_prefill = false,
               pie_cuda_driver::NcclComm* tp_comm = nullptr)
{
    auto engine = pie_cuda_driver::LoadedModel::load(cfg, tp_comm);
    const auto& mt_for_parity = engine.hf_config().model_type;
    const bool is_gpt_oss  = (mt_for_parity == "gpt_oss");
    const bool is_gemma3n  = (mt_for_parity == "gemma3n" || mt_for_parity == "gemma3n_text");
    const bool is_qwen3_5  = (mt_for_parity == "qwen3_5" || mt_for_parity == "qwen3_5_text");
    const bool is_qwen3_5_moe = (mt_for_parity == "qwen3_5_moe" || mt_for_parity == "qwen3_5_moe_text"
                                 || mt_for_parity == "qwen3_moe");
    {
        const bool supported =
            mt_for_parity == "qwen3" || is_qwen3_5 || is_qwen3_5_moe
         || mt_for_parity == "qwen2"
         || mt_for_parity == "llama" || mt_for_parity == "llama3"
         || mt_for_parity == "mistral" || mt_for_parity == "mistral3"
         || is_gpt_oss
         || is_gemma3n;
        if (!supported) {
            std::cerr << "[parity] unsupported model_type: " << mt_for_parity << "\n";
            return 2;
        }
        if ((is_gpt_oss || is_gemma3n || is_qwen3_5 || is_qwen3_5_moe) && !paged) {
            std::cerr << "[parity] " << mt_for_parity << " requires --parity-paged\n";
            return 2;
        }
    }
    pie_cuda_driver::model::Qwen3Weights weights;
    pie_cuda_driver::model::MixtralWeights weights_mixtral;
    pie_cuda_driver::model::Gemma3nWeights weights_gemma3n;
    pie_cuda_driver::model::Qwen3_5Weights weights_qwen3_5;
    pie_cuda_driver::model::Qwen3_5MoeWeights weights_qwen3_5_moe;
    if (is_gpt_oss) {
        weights_mixtral = pie_cuda_driver::model::bind_gpt_oss(engine);
    } else if (is_gemma3n) {
        weights_gemma3n = pie_cuda_driver::model::bind_gemma3n(engine);
    } else if (is_qwen3_5) {
        weights_qwen3_5 = pie_cuda_driver::model::bind_qwen3_5(engine);
    } else if (is_qwen3_5_moe) {
        weights_qwen3_5_moe = pie_cuda_driver::model::bind_qwen3_5_moe(engine);
    } else {
        weights = pie_cuda_driver::model::bind_llama_like(engine);
    }

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

    // For `decode_after_prefill`, the row index in `ws.logits` we want to
    // dump at the end is the LAST row written by the *last* call. Default
    // (single-prefill) is N-1; decode mode overwrites only row 0 in the
    // second call, so the dump position becomes 0.
    int dump_row = N - 1;

    if (paged) {
        // Build a single-request paged layout that mirrors what the runtime
        // would send for a fresh request: pages [0..ceil(N/page_size)],
        // last_page_len computed accordingly.
        //
        // `decode_after_prefill` mode: instead of one prefill of N tokens,
        // do (a) prefill of the first N-1 tokens followed by (b) a single
        // decode-shaped step (qo_len=1) at position N-1. The dumped
        // logits come from step (b) — they should match HF's logits at
        // position N-1, just produced via the decode kernel + cached KV
        // read instead of a fresh prefill. Catches decode-only bugs that
        // multi-step prefill parity can't see.
        if (decode_after_prefill && N < 2) {
            std::cerr << "[parity] --parity-decode-after-prefill requires "
                         "at least 2 tokens; got " << N << "\n";
            return 5;
        }
        const int prefill_N  = decode_after_prefill ? (N - 1) : N;
        const int page_size  = static_cast<int>(cfg.batching.kv_page_size);
        const int total_pages = (N + page_size - 1) / page_size;

        auto cache = pie_cuda_driver::KvCache::allocate(
            engine.hf_config().num_hidden_layers,
            std::max(total_pages, 1),
            page_size,
            engine.hf_config().num_key_value_heads,
            engine.hf_config().head_dim);

        auto parity_attn_ws = pie_cuda_driver::AttentionWorkspace::allocate();

        // qwen3_5 / qwen3_5_moe need their own scratch + state caches.
        pie_cuda_driver::model::Qwen3_5LinearAttnWorkspace q35_la_ws;
        pie_cuda_driver::Qwen3_5StateCache q35_state_cache;
        pie_cuda_driver::model::Qwen3_5MoeMlpWorkspace q35_moe_ws;
        if (is_qwen3_5 || is_qwen3_5_moe) {
            const auto& cfg_q = engine.hf_config();
            const int K_dim = cfg_q.linear_num_key_heads * cfg_q.linear_key_head_dim;
            const int V_dim = cfg_q.linear_num_value_heads * cfg_q.linear_value_head_dim;
            const int conv_dim = 2 * K_dim + V_dim;
            q35_la_ws = pie_cuda_driver::model::Qwen3_5LinearAttnWorkspace::allocate(
                N, conv_dim, cfg_q.linear_num_value_heads,
                cfg_q.linear_num_key_heads,
                cfg_q.linear_key_head_dim, cfg_q.linear_value_head_dim,
                /*hq=*/cfg_q.num_attention_heads * cfg_q.head_dim);
            const std::size_t num_layers = is_qwen3_5
                ? weights_qwen3_5.layers.size()
                : weights_qwen3_5_moe.layers.size();
            std::vector<bool> layer_is_linear(num_layers);
            for (std::size_t L = 0; L < num_layers; ++L) {
                const bool is_linear = is_qwen3_5
                    ? (weights_qwen3_5.layers[L].kind ==
                       pie_cuda_driver::model::Qwen3_5LayerWeights::Kind::LinearAttn)
                    : (weights_qwen3_5_moe.layers[L].kind ==
                       pie_cuda_driver::model::Qwen3_5MoeLayerWeights::Kind::LinearAttn);
                layer_is_linear[L] = is_linear;
            }
            q35_state_cache = pie_cuda_driver::Qwen3_5StateCache::allocate(
                layer_is_linear, conv_dim, cfg_q.linear_conv_kernel_dim,
                cfg_q.linear_num_value_heads,
                cfg_q.linear_key_head_dim, cfg_q.linear_value_head_dim);
            if (is_qwen3_5_moe) {
                q35_moe_ws = pie_cuda_driver::model::Qwen3_5MoeMlpWorkspace::allocate(
                    N, cfg_q.hidden_size,
                    cfg_q.num_experts, cfg_q.num_experts_per_tok,
                    cfg_q.moe_intermediate_size,
                    cfg_q.shared_expert_intermediate_size);
            }
        }

        // Build the per-arch fwd_cfg once; reused across the prefill and
        // (optional) decode calls.
        pie_cuda_driver::model::LlamaLikeForwardCfg fwd_cfg{};
        if (is_gpt_oss) {
            const auto& hf = engine.hf_config();
            fwd_cfg.use_qkv_bias = hf.attention_bias;
            fwd_cfg.rope_kind    = hf.has_rope_scaling
                ? pie_cuda_driver::model::RopeKind::YaRN
                : pie_cuda_driver::model::RopeKind::Standard;
            fwd_cfg.yarn_factor                = hf.rope_factor;
            fwd_cfg.yarn_low_freq_factor       = hf.rope_low_freq_factor;
            fwd_cfg.yarn_high_freq_factor      = hf.rope_high_freq_factor;
            fwd_cfg.yarn_original_max_position = hf.rope_original_max_position;
            fwd_cfg.sliding_window             = hf.sliding_window;
            for (const auto& t : hf.layer_types) {
                fwd_cfg.per_layer_window_left.push_back(
                    (t == "sliding_attention") ? hf.sliding_window : -1);
            }
            // Decode-after-prefill mode exercises both the prefill and
            // decode kernels in sequence. force_prefill_path would defeat
            // the purpose of the test, so leave it false.
            fwd_cfg.force_prefill_path = !decode_after_prefill;
        }

        // Helper to run one paged forward call. `total_n` is qo_len, `kv_n`
        // is the post-write KV length (for the indptr/last_page_len math).
        // `tok_d` / `pos_d` are device pointers to the inputs for this call.
        auto run_call = [&](const std::int32_t* tok_d,
                            const std::int32_t* pos_d,
                            int total_n, int kv_n, bool is_decode) {
            const int n_pages_kv = (kv_n + page_size - 1) / page_size;
            std::vector<std::uint32_t> h_qo  = {0u, (std::uint32_t)total_n};
            std::vector<std::uint32_t> h_pp  = {0u, (std::uint32_t)n_pages_kv};
            std::vector<std::uint32_t> h_pi(n_pages_kv);
            for (int i = 0; i < n_pages_kv; ++i) h_pi[i] = (std::uint32_t)i;
            std::vector<std::uint32_t> h_lpl = {
                (std::uint32_t)(((kv_n - 1) % page_size) + 1)
            };

            std::uint32_t *d_qo, *d_pi, *d_pp, *d_lpl;
            CUDA_CHECK(cudaMalloc(&d_qo,  4 * h_qo.size()));
            CUDA_CHECK(cudaMalloc(&d_pi,  4 * h_pi.size()));
            CUDA_CHECK(cudaMalloc(&d_pp,  4 * h_pp.size()));
            CUDA_CHECK(cudaMalloc(&d_lpl, 4 * h_lpl.size()));
            CUDA_CHECK(cudaMemcpy(d_qo,  h_qo.data(),  4 * h_qo.size(),  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_pi,  h_pi.data(),  4 * h_pi.size(),  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_pp,  h_pp.data(),  4 * h_pp.size(),  cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaMemcpy(d_lpl, h_lpl.data(), 4 * h_lpl.size(), cudaMemcpyHostToDevice));

            if (is_gpt_oss) {
                const auto& hf = engine.hf_config();
                pie_cuda_driver::model::mixtral_forward_paged(
                    weights_mixtral, engine.hf_config(), fwd_cfg,
                    hf.num_experts, hf.num_experts_per_tok,
                    ws, cache, parity_attn_ws, cublas,
                    tok_d, pos_d,
                    d_qo, d_pi, d_pp, d_lpl,
                    /*qo_indptr_h=*/h_qo.data(),
                    /*kv_page_indptr_h=*/h_pp.data(),
                    /*total_tokens=*/total_n, /*num_requests=*/1,
                    /*is_pure_decode=*/is_decode);
            } else if (is_gemma3n) {
                pie_cuda_driver::model::Gemma3nForwardCfg gemma3n_fwd{};
                gemma3n_fwd.final_logit_softcap =
                    engine.hf_config().gemma_final_logit_softcap;
                gemma3n_fwd.force_prefill_path = !decode_after_prefill;
                pie_cuda_driver::model::gemma3n_forward_paged(
                    weights_gemma3n, engine.hf_config(), gemma3n_fwd,
                    ws, cache, parity_attn_ws, cublas,
                    tok_d, pos_d,
                    d_qo, d_pi, d_pp, d_lpl,
                    /*qo_indptr_h=*/h_qo.data(),
                    /*kv_page_indptr_h=*/h_pp.data(),
                    /*total_tokens=*/total_n, /*num_requests=*/1,
                    /*is_pure_decode=*/is_decode);
            } else if (is_qwen3_5) {
                pie_cuda_driver::model::Qwen3_5ForwardCfg q35_fwd{};
                q35_fwd.force_prefill_path = !decode_after_prefill;
                q35_fwd.tp_size = cfg.distributed.tp_size;
                q35_fwd.tp_comm = tp_comm;
                pie_cuda_driver::model::Qwen3_5PlanState q35_plan;
                pie_cuda_driver::model::prepare_qwen3_5_decode_plan(
                    q35_plan, parity_attn_ws, cache, engine.hf_config(),
                    q35_fwd, h_pp.data(), /*num_requests=*/1, is_decode);
                pie_cuda_driver::model::qwen3_5_forward_paged(
                    weights_qwen3_5, engine.hf_config(), q35_fwd, q35_plan,
                    ws, q35_la_ws, cache, q35_state_cache,
                    parity_attn_ws, cublas,
                    tok_d, pos_d,
                    d_qo, d_pi, d_pp, d_lpl,
                    /*qo_indptr_h=*/h_qo.data(),
                    /*kv_page_indptr_h=*/h_pp.data(),
                    /*total_tokens=*/total_n, /*num_requests=*/1,
                    /*is_pure_decode=*/is_decode,
                    /*mask_d=*/nullptr, /*mask_indptr_d=*/nullptr);
            } else if (is_qwen3_5_moe) {
                pie_cuda_driver::model::Qwen3_5ForwardCfg q35_fwd{};
                q35_fwd.force_prefill_path = !decode_after_prefill;
                q35_fwd.tp_size = cfg.distributed.tp_size;
                q35_fwd.tp_comm = tp_comm;
                pie_cuda_driver::model::Qwen3_5PlanState q35_plan;
                pie_cuda_driver::model::prepare_qwen3_5_decode_plan(
                    q35_plan, parity_attn_ws, cache, engine.hf_config(),
                    q35_fwd, h_pp.data(), /*num_requests=*/1, is_decode);
                pie_cuda_driver::model::qwen3_5_moe_forward_paged(
                    weights_qwen3_5_moe, engine.hf_config(), q35_fwd, q35_plan,
                    ws, q35_la_ws, q35_moe_ws,
                    cache, q35_state_cache,
                    parity_attn_ws, cublas,
                    tok_d, pos_d,
                    d_qo, d_pi, d_pp, d_lpl,
                    /*qo_indptr_h=*/h_qo.data(),
                    /*kv_page_indptr_h=*/h_pp.data(),
                    /*total_tokens=*/total_n, /*num_requests=*/1,
                    /*is_pure_decode=*/is_decode,
                    /*mask_d=*/nullptr, /*mask_indptr_d=*/nullptr);
            } else {
                pie_cuda_driver::model::qwen3_forward_paged(
                    weights, engine.hf_config(), ws, cache, parity_attn_ws, cublas,
                    tok_d, pos_d,
                    d_qo, d_pi, d_pp, d_lpl,
                    /*qo_indptr_h=*/h_qo.data(),
                    /*kv_page_indptr_h=*/h_pp.data(),
                    /*total_tokens=*/total_n, /*num_requests=*/1,
                    /*is_pure_decode=*/is_decode);
            }

            cudaFree(d_qo); cudaFree(d_pi); cudaFree(d_pp); cudaFree(d_lpl);
        };

        // Prefill on the first prefill_N tokens.
        run_call(d_tokens, d_positions, prefill_N, prefill_N, /*is_decode=*/false);

        if (decode_after_prefill) {
            // Single decode step at position N-1, reading KV [0, N-1) and
            // appending the K/V for position N-1.
            run_call(d_tokens + (N - 1), d_positions + (N - 1),
                     /*total_n=*/1, /*kv_n=*/N, /*is_decode=*/true);
            // The decode call wrote logits for one token at row 0.
            dump_row = 0;
        }

        // Optional decode-microbench (PIE_PARITY_BENCH_DECODE=K): replay K
        // additional decode steps after the parity step, timing the total
        // wall clock. Logits at the dump_row are unchanged (we use the
        // last step's output).
        if (const char* dbg = std::getenv("PIE_PARITY_BENCH_DECODE")) {
            const int extra = std::max(1, std::atoi(dbg));
            CUDA_CHECK(cudaDeviceSynchronize());
            const auto t0 = std::chrono::steady_clock::now();
            for (int s = 0; s < extra; ++s) {
                run_call(d_tokens + (N - 1), d_positions + (N - 1),
                         /*total_n=*/1, /*kv_n=*/N + s + 1, /*is_decode=*/true);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            const auto t1 = std::chrono::steady_clock::now();
            const double ms =
                std::chrono::duration<double, std::milli>(t1 - t0).count();
            std::cerr << "[bench-decode] " << extra << " steps in "
                      << ms << " ms => " << (extra * 1000.0 / ms)
                      << " tok/s (single-stream, single-request)\n";
        }
    } else {
        pie_cuda_driver::model::qwen3_forward_prefill(
            weights, engine.hf_config(), ws, cublas, d_tokens, d_positions, N);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    // Greedy sample over all rows on the GPU, then echo the last-token
    // id to stderr — the parity harness picks it up and cross-checks
    // against numpy.argmax of the dumped logits.
    // The last call wrote `last_n_rows` rows of logits into ws.logits.
    // For single-prefill: N rows, dump row N-1. For decode-after-prefill:
    // the decode call wrote 1 row at index 0, dump row 0.
    const int last_n_rows = decode_after_prefill ? 1 : N;
    {
        const int V = engine.hf_config().vocab_size;
        std::int32_t* d_sampled = nullptr;
        CUDA_CHECK(cudaMalloc(&d_sampled, sizeof(std::int32_t) * last_n_rows));
        pie_cuda_driver::kernels::launch_argmax_bf16(
            ws.logits.data(), d_sampled, last_n_rows, V, /*stream=*/nullptr);
        std::vector<std::int32_t> host_sampled(last_n_rows);
        CUDA_CHECK(cudaMemcpy(host_sampled.data(), d_sampled,
                              sizeof(std::int32_t) * last_n_rows,
                              cudaMemcpyDeviceToHost));
        cudaFree(d_sampled);
        std::cerr << "[parity] gpu argmax last-token id = "
                  << host_sampled.back() << "\n";
    }

    // Copy last-token logits row out as bf16 (we'll convert in Python).
    const int V = engine.hf_config().vocab_size;
    std::vector<std::uint16_t> host_logits(V);  // bf16 viewed as u16
    const auto* base = static_cast<const std::uint16_t*>(ws.logits.data());
    CUDA_CHECK(cudaMemcpy(host_logits.data(),
                          base + static_cast<std::size_t>(dump_row) * V,
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

namespace {

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
                  << "  tp_size         = " << cfg.distributed.tp_size << "\n"
                  << "  tp_rank         = " << cfg.distributed.tp_rank << "\n";
    }

    // Bind the requested CUDA device before NCCL init — ncclCommInitRank
    // captures whatever is current on the calling thread.
    {
        const std::string& dev = cfg.model.device;
        const auto colon = dev.find(':');
        const int dev_id = (colon == std::string::npos)
            ? 0
            : std::atoi(dev.c_str() + colon + 1);
        CUDA_CHECK(cudaSetDevice(dev_id));
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

        // Smoke test: every rank contributes (rank+1); sum should be
        // world*(world+1)/2. Catches mis-numbered ranks at startup.
        cudaStream_t s = nullptr;
        CUDA_CHECK(cudaStreamCreate(&s));
        int* d_v = nullptr;
        CUDA_CHECK(cudaMalloc(&d_v, sizeof(int)));
        const int rank1 = cfg.distributed.tp_rank + 1;
        CUDA_CHECK(cudaMemcpyAsync(d_v, &rank1, sizeof(int),
                                   cudaMemcpyHostToDevice, s));
        NCCL_CHECK(ncclAllReduce(d_v, d_v, 1, ncclInt32, ncclSum,
                                 tp_comm.comm(), s));
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
        return run_parity(cfg, parity_tokens, out_path, parity_paged,
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
         || mt == "gemma3n" || mt == "gemma3n_text";
        if (!supported) {
            std::cerr << "[pie-driver-cuda] arch '" << mt
                      << "' not yet supported (Qwen 2/3, Llama-3, "
                      << "Mistral, Mixtral, GPT-OSS, Phi-3, OLMo-3, Gemma-2/3/4)\n";
            return 2;
        }
    }
    const std::string& mt_for_bind = engine.hf_config().model_type;

    // Per-arch weights live in their own struct shape. We hold one of
    // them on the stack; only the one matching `mt_for_bind` is
    // populated. The dispatch below wraps the right (weights, cfg,
    // forward function) triple into a single `ForwardFn` closure.
    pie_cuda_driver::model::Qwen3Weights   weights_llama;
    pie_cuda_driver::model::Gemma2Weights  weights_gemma;
    pie_cuda_driver::model::Gemma4Weights  weights_gemma4;
    pie_cuda_driver::model::Gemma3nWeights weights_gemma3n;
    pie_cuda_driver::model::MixtralWeights weights_mixtral;
    pie_cuda_driver::model::Qwen3_5Weights weights_qwen3_5;
    pie_cuda_driver::model::Qwen3_5MoeWeights weights_qwen3_5_moe;
    const bool is_gemma_arch =
        (mt_for_bind == "gemma2" || mt_for_bind == "gemma3" ||
         mt_for_bind == "gemma3_text");
    const bool is_gemma4_arch =
        (mt_for_bind == "gemma4" || mt_for_bind == "gemma4_text");
    const bool is_gemma3n_arch =
        (mt_for_bind == "gemma3n" || mt_for_bind == "gemma3n_text");
    const bool is_gpt_oss_arch = (mt_for_bind == "gpt_oss");
    const bool is_mixtral_arch =
        (mt_for_bind == "mixtral") || is_gpt_oss_arch;  // both use mixtral fwd
    const bool is_qwen3_5_arch =
        (mt_for_bind == "qwen3_5" || mt_for_bind == "qwen3_5_text");
    // Qwen3-MoE (Qwen3-30B-A3B, model_type="qwen3_moe") and Qwen3.5/3.6-MoE
    // (model_type="qwen3_5_moe[_text]") share the bind + forward path —
    // engine.cpp::is_qwen3_5_moe_arch groups them, qwen3_5_moe.cpp branches
    // internally on model_type for the layer_types/shared-expert quirks.
    const bool is_qwen3_5_moe_arch =
        (mt_for_bind == "qwen3_5_moe" || mt_for_bind == "qwen3_5_moe_text"
         || mt_for_bind == "qwen3_moe");

    if (mt_for_bind == "phi3") {
        weights_llama = pie_cuda_driver::model::bind_phi3(engine);
    } else if (mt_for_bind == "olmo2" || mt_for_bind == "olmo3") {
        weights_llama = pie_cuda_driver::model::bind_olmo3(engine);
    } else if (mt_for_bind == "mistral3" || mt_for_bind == "ministral3") {
        weights_llama = pie_cuda_driver::model::bind_mistral3(engine);
    } else if (mt_for_bind == "gemma2") {
        weights_gemma = pie_cuda_driver::model::bind_gemma2(engine);
    } else if (mt_for_bind == "gemma3" || mt_for_bind == "gemma3_text") {
        weights_gemma = pie_cuda_driver::model::bind_gemma3(engine);
    } else if (is_gemma4_arch) {
        weights_gemma4 = pie_cuda_driver::model::bind_gemma4(engine);
    } else if (is_gemma3n_arch) {
        weights_gemma3n = pie_cuda_driver::model::bind_gemma3n(engine);
    } else if (is_gpt_oss_arch) {
        weights_mixtral = pie_cuda_driver::model::bind_gpt_oss(engine);
    } else if (mt_for_bind == "mixtral") {
        weights_mixtral = pie_cuda_driver::model::bind_mixtral(engine);
    } else if (is_qwen3_5_arch) {
        weights_qwen3_5 = pie_cuda_driver::model::bind_qwen3_5(engine);
    } else if (is_qwen3_5_moe_arch) {
        weights_qwen3_5_moe = pie_cuda_driver::model::bind_qwen3_5_moe(engine);
    } else {
        weights_llama = pie_cuda_driver::model::bind_llama_like(engine);
    }
    const std::size_t num_layers_bound =
        is_gemma4_arch    ? weights_gemma4.layers.size()
      : is_gemma3n_arch   ? weights_gemma3n.layers.size()
      : is_gemma_arch     ? weights_gemma.layers.size()
      : is_mixtral_arch   ? weights_mixtral.layers.size()
      : is_qwen3_5_arch   ? weights_qwen3_5.layers.size()
      : is_qwen3_5_moe_arch ? weights_qwen3_5_moe.layers.size()
                            : weights_llama.layers.size();
    if (verbose) {
        std::cerr << "[pie-driver-cuda] schema bound: "
                  << num_layers_bound << " layers ("
                  << engine.hf_config().model_type
                  << (engine.hf_config().use_qk_norm ? ", q/k norm" : "")
                  << ")\n";
    }

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
    } else if (is_gemma3n_arch) {
        // Per-layer intermediate (HF stores it as a list); head_dim is
        // uniform across layers on gemma3n, so KV cache can use the
        // standard allocator.
        for (int v : weights_gemma3n.per_layer_intermediate) {
            if (v > max_mlp_intermediate) max_mlp_intermediate = v;
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

    // Plan-state holders used by the prepare/body split for graph-friendly
    // dispatch. Allocated unconditionally — empty on archs that don't use
    // them. `qwen3_5_plan_state` is shared between qwen3_5 and qwen3_5_moe
    // (they share `prepare_qwen3_5_decode_plan`).
    pie_cuda_driver::model::Qwen3_5PlanState qwen3_5_plan_state;

    // Qwen3.5 / Qwen3.6-MoE linear-attention extras: per-layer state cache
    // + a per-call workspace. Inert (default-constructed) on every other
    // arch. The MoE arch additionally needs a routed-experts workspace.
    pie_cuda_driver::model::Qwen3_5LinearAttnWorkspace qwen3_5_la_ws;
    pie_cuda_driver::Qwen3_5StateCache qwen3_5_state_cache;
    pie_cuda_driver::model::Qwen3_5MoeMlpWorkspace qwen3_5_moe_ws;
    if (is_qwen3_5_arch || is_qwen3_5_moe_arch) {
        const auto& cfg_q = engine.hf_config();
        const int K_dim = cfg_q.linear_num_key_heads * cfg_q.linear_key_head_dim;
        const int V_dim = cfg_q.linear_num_value_heads * cfg_q.linear_value_head_dim;
        const int conv_dim = 2 * K_dim + V_dim;
        qwen3_5_la_ws = pie_cuda_driver::model::Qwen3_5LinearAttnWorkspace::allocate(
            max_workspace_tokens, conv_dim,
            cfg_q.linear_num_value_heads,
            cfg_q.linear_num_key_heads,
            cfg_q.linear_key_head_dim,
            cfg_q.linear_value_head_dim,
            /*hq=*/cfg_q.num_attention_heads * cfg_q.head_dim);
        const std::size_t num_layers = is_qwen3_5_arch
            ? weights_qwen3_5.layers.size()
            : weights_qwen3_5_moe.layers.size();
        std::vector<bool> layer_is_linear(num_layers);
        for (std::size_t L = 0; L < num_layers; ++L) {
            const bool is_linear = is_qwen3_5_arch
                ? (weights_qwen3_5.layers[L].kind ==
                   pie_cuda_driver::model::Qwen3_5LayerWeights::Kind::LinearAttn)
                : (weights_qwen3_5_moe.layers[L].kind ==
                   pie_cuda_driver::model::Qwen3_5MoeLayerWeights::Kind::LinearAttn);
            layer_is_linear[L] = is_linear;
        }
        // Allocate per-slot state for the linear-attn layers.
        //   linear_attn_max_slots = 0 → follow max_batch_size (default,
        //                               fine for small/medium models)
        //   linear_attn_max_slots > 0 → explicit cap to bound VRAM on
        //                               huge MoE × wide max_batch_size
        // Reducing max_slots below max_batch_size doesn't constrain
        // overall serving throughput — the runtime will simply hold
        // more contexts in cooler LRU positions and re-prefill them
        // when their slots are reassigned.
        const int q35_max_slots =
            std::max<int>(1,
                cfg.batching.linear_attn_max_slots > 0
                    ? std::min<std::uint32_t>(
                          cfg.batching.linear_attn_max_slots,
                          cfg.batching.max_batch_size)
                    : cfg.batching.max_batch_size);
        qwen3_5_state_cache = pie_cuda_driver::Qwen3_5StateCache::allocate(
            layer_is_linear, conv_dim, cfg_q.linear_conv_kernel_dim,
            cfg_q.linear_num_value_heads,
            cfg_q.linear_key_head_dim,
            cfg_q.linear_value_head_dim,
            q35_max_slots);
        const std::size_t per_slot_recurrent_bytes =
            static_cast<std::size_t>(cfg_q.linear_num_value_heads) *
            cfg_q.linear_key_head_dim *
            cfg_q.linear_value_head_dim * sizeof(float);
        const std::size_t per_slot_conv_bytes =
            static_cast<std::size_t>(cfg_q.linear_conv_kernel_dim) *
            conv_dim * sizeof(std::uint16_t);
        const std::size_t num_linear_layers = std::count(
            layer_is_linear.begin(), layer_is_linear.end(), true);
        const std::size_t total_bytes = num_linear_layers *
            static_cast<std::size_t>(q35_max_slots) *
            (per_slot_recurrent_bytes + per_slot_conv_bytes);
        if (verbose) {
            std::cerr << "[pie-driver-cuda] qwen3.5 state cache: "
                      << num_linear_layers << " linear layers, "
                      << q35_max_slots << " slots, "
                      << (per_slot_recurrent_bytes + per_slot_conv_bytes)
                      << " B/slot (recurrent="
                      << per_slot_recurrent_bytes << " conv="
                      << per_slot_conv_bytes << "), total ~"
                      << (total_bytes / (1024 * 1024)) << " MiB\n";
        }

        if (is_qwen3_5_moe_arch) {
            qwen3_5_moe_ws = pie_cuda_driver::model::Qwen3_5MoeMlpWorkspace::allocate(
                max_workspace_tokens,
                cfg_q.hidden_size,
                cfg_q.num_experts,
                cfg_q.num_experts_per_tok,
                cfg_q.moe_intermediate_size,
                cfg_q.shared_expert_intermediate_size);
        }
    }

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

    if (verbose) {
        std::cerr << "[pie-driver-cuda] kv_cache: "
                  << kv_cache.num_pages() << " pages × "
                  << kv_cache.page_size() << " tokens; "
                  << "workspace tokens=" << max_workspace_tokens
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
        register_server(server_p.get());
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
        std::signal(SIGINT, on_signal);
        std::signal(SIGTERM, on_signal);
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
        fwd_cfg.rope_kind = hf.has_rope_scaling
            ? pie_cuda_driver::model::RopeKind::YaRN
            : pie_cuda_driver::model::RopeKind::Standard;
        fwd_cfg.yarn_factor               = hf.rope_factor;
        fwd_cfg.yarn_low_freq_factor      = hf.rope_low_freq_factor;
        fwd_cfg.yarn_high_freq_factor     = hf.rope_high_freq_factor;
        fwd_cfg.yarn_original_max_position = hf.rope_original_max_position;
        fwd_cfg.sliding_window            = hf.sliding_window;
        // flashinfer's decode kernel (DISPATCH_GQA_GROUP_SIZE in
        // utils.cuh) instantiates GQA group sizes {1, 2, 3, 4, 8}.
        // Models like Qwen2-0.5B (group=7), Qwen2-1.5B (group=6) need
        // the prefill kernel even for decode-only fires.
        const int gqa = hf.num_attention_heads / hf.num_key_value_heads;
        const bool gqa_in_decode_set = (gqa == 1 || gqa == 2 || gqa == 3
                                        || gqa == 4 || gqa == 8);
        fwd_cfg.force_prefill_path = !gqa_in_decode_set;
        // Tensor-parallel state. tp_comm == nullptr at tp_size == 1
        // keeps the original single-GPU branches in the forward kernels.
        fwd_cfg.tp_size = cfg.distributed.tp_size;
        fwd_cfg.tp_comm = tp_comm_ptr;

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

        if (verbose) {
            std::cerr << "[pie-driver-cuda] model_type=" << mt
                      << " use_qk_norm=" << fwd_cfg.use_qk_norm
                      << " use_qkv_bias=" << fwd_cfg.use_qkv_bias
                      << " rope=" << (fwd_cfg.rope_kind ==
                           pie_cuda_driver::model::RopeKind::YaRN ? "yarn" : "standard")
                      << "\n";
        }
    }

    if (is_gemma4_arch) {
        const auto& hf = engine.hf_config();
        gemma4_fwd_cfg.final_logit_softcap = hf.gemma_final_logit_softcap;
        const int gqa = hf.num_attention_heads / hf.num_key_value_heads;
        const bool gqa_in_decode_set = (gqa == 1 || gqa == 2 || gqa == 3
                                        || gqa == 4 || gqa == 8);
        gemma4_fwd_cfg.force_prefill_path = !gqa_in_decode_set;
        gemma4_fwd_cfg.tp_size = cfg.distributed.tp_size;
        gemma4_fwd_cfg.tp_comm = tp_comm_ptr;
    }

    // Build the type-erased forward closure once. The captures live in
    // `main`'s scope (weights_*, fwd_cfg, gemma_fwd_cfg) and persist for
    // the lifetime of the server.
    pie_cuda_driver::ForwardFn forward_fn;
    pie_cuda_driver::model::LlamaLikePlanState llama_plan;
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
        forward_fn = [&engine, &weights_gemma4, &gemma4_moe_ws, gemma4_fwd_cfg](
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
            const std::uint8_t* mask_d, const std::int32_t* mask_indptr_d,
            const std::int32_t* slot_ids_h, const std::uint8_t* is_fresh_h,
            const std::int32_t* slot_ids_d) {
            pie_cuda_driver::model::gemma4_forward_paged(
                weights_gemma4, engine.hf_config(), gemma4_fwd_cfg,
                ws, gemma4_moe_ws, cache, attn_ws, cublas,
                tok, pos,
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                qo_indptr_h, kv_page_indptr_h,
                N, R, is_pure_decode, mask_d, mask_indptr_d);
        };
    } else if (is_gemma3n_arch) {
        // Loader-only milestone: bind_gemma3n loads every tensor; the
        // forward function (AltUp predict/correct + Laurel + activation
        // sparsity + PLE input gate) is a follow-up. The stub throws
        // with a clear message at the first fire_batch.
        pie_cuda_driver::model::Gemma3nForwardCfg gemma3n_fwd_cfg{};
        gemma3n_fwd_cfg.final_logit_softcap = engine.hf_config().gemma_final_logit_softcap;
        gemma3n_fwd_cfg.tp_size = cfg.distributed.tp_size;
        gemma3n_fwd_cfg.tp_comm = tp_comm_ptr;
        forward_fn = [&engine, &weights_gemma3n, gemma3n_fwd_cfg](
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
            const std::uint8_t* mask_d, const std::int32_t* mask_indptr_d,
            const std::int32_t* slot_ids_h, const std::uint8_t* is_fresh_h,
            const std::int32_t* slot_ids_d) {
            pie_cuda_driver::model::gemma3n_forward_paged(
                weights_gemma3n, engine.hf_config(), gemma3n_fwd_cfg,
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
            const std::uint8_t* mask_d, const std::int32_t* mask_indptr_d,
            const std::int32_t* slot_ids_h, const std::uint8_t* is_fresh_h,
            const std::int32_t* slot_ids_d) {
            pie_cuda_driver::model::gemma2_forward_paged(
                weights_gemma, engine.hf_config(), gemma_fwd_cfg,
                ws, cache, attn_ws, cublas,
                tok, pos,
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                qo_indptr_h, kv_page_indptr_h,
                N, R, is_pure_decode, mask_d, mask_indptr_d);
        };
    } else if (is_mixtral_arch) {
        // Mixtral reuses LlamaLikeForwardCfg for its (identical) attention
        // half. The MoE block reads num_experts / num_experts_per_tok
        // straight from HfConfig.
        const int num_experts = engine.hf_config().num_experts;
        const int top_k       = engine.hf_config().num_experts_per_tok;
        forward_fn = [&engine, &weights_mixtral, fwd_cfg, num_experts, top_k](
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
            const std::uint8_t* mask_d, const std::int32_t* mask_indptr_d,
            const std::int32_t* slot_ids_h, const std::uint8_t* is_fresh_h,
            const std::int32_t* slot_ids_d) {
            pie_cuda_driver::model::mixtral_forward_paged(
                weights_mixtral, engine.hf_config(), fwd_cfg,
                num_experts, top_k,
                ws, cache, attn_ws, cublas,
                tok, pos,
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                qo_indptr_h, kv_page_indptr_h,
                N, R, is_pure_decode, mask_d, mask_indptr_d);
        };
    } else if (is_qwen3_5_arch) {
        const int q35_tp_size = cfg.distributed.tp_size;
        pie_cuda_driver::NcclComm* q35_tp_comm = tp_comm_ptr;
        forward_fn.prepare = [&engine, &kv_cache, &qwen3_5_plan_state,
                              q35_tp_size, q35_tp_comm](
            pie_cuda_driver::AttentionWorkspace& attn_ws,
            const std::uint32_t* kv_page_indptr_h,
            int R, bool is_pure_decode) {
            pie_cuda_driver::model::Qwen3_5ForwardCfg q35_fwd{};
            const auto& hf_q = engine.hf_config();
            const int gqa_q = hf_q.num_attention_heads /
                              std::max(1, hf_q.num_key_value_heads);
            q35_fwd.force_prefill_path =
                !(gqa_q == 1 || gqa_q == 2 || gqa_q == 3 ||
                  gqa_q == 4 || gqa_q == 8);
            q35_fwd.tp_size = q35_tp_size;
            q35_fwd.tp_comm = q35_tp_comm;
            pie_cuda_driver::model::prepare_qwen3_5_decode_plan(
                qwen3_5_plan_state, attn_ws, kv_cache, engine.hf_config(),
                q35_fwd, kv_page_indptr_h, R, is_pure_decode);
        };
        forward_fn.body = [&engine, &weights_qwen3_5, &qwen3_5_la_ws,
                           &qwen3_5_state_cache, &qwen3_5_plan_state,
                           q35_tp_size, q35_tp_comm](
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
            const std::uint8_t* mask_d, const std::int32_t* mask_indptr_d,
            const std::int32_t* slot_ids_h, const std::uint8_t* is_fresh_h,
            const std::int32_t* slot_ids_d) {
            pie_cuda_driver::model::Qwen3_5ForwardCfg q35_fwd{};
            const auto& hf_q = engine.hf_config();
            const int gqa_q = hf_q.num_attention_heads /
                              std::max(1, hf_q.num_key_value_heads);
            q35_fwd.force_prefill_path =
                !(gqa_q == 1 || gqa_q == 2 || gqa_q == 3 ||
                  gqa_q == 4 || gqa_q == 8);
            q35_fwd.tp_size = q35_tp_size;
            q35_fwd.tp_comm = q35_tp_comm;
            pie_cuda_driver::model::qwen3_5_forward_paged(
                weights_qwen3_5, engine.hf_config(), q35_fwd, qwen3_5_plan_state,
                ws, qwen3_5_la_ws, cache, qwen3_5_state_cache,
                attn_ws, cublas,
                tok, pos,
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                qo_indptr_h, kv_page_indptr_h,
                N, R, is_pure_decode, mask_d, mask_indptr_d,
                slot_ids_h, is_fresh_h, slot_ids_d);
        };
    } else if (is_qwen3_5_moe_arch) {
        const int q35moe_tp_size = cfg.distributed.tp_size;
        pie_cuda_driver::NcclComm* q35moe_tp_comm = tp_comm_ptr;
        forward_fn.prepare = [&engine, &kv_cache, &qwen3_5_plan_state,
                              q35moe_tp_size, q35moe_tp_comm](
            pie_cuda_driver::AttentionWorkspace& attn_ws,
            const std::uint32_t* kv_page_indptr_h,
            int R, bool is_pure_decode) {
            pie_cuda_driver::model::Qwen3_5ForwardCfg q35_fwd{};
            const auto& hf_q = engine.hf_config();
            const int gqa_q = hf_q.num_attention_heads /
                              std::max(1, hf_q.num_key_value_heads);
            q35_fwd.force_prefill_path =
                !(gqa_q == 1 || gqa_q == 2 || gqa_q == 3 ||
                  gqa_q == 4 || gqa_q == 8);
            q35_fwd.tp_size = q35moe_tp_size;
            q35_fwd.tp_comm = q35moe_tp_comm;
            pie_cuda_driver::model::prepare_qwen3_5_decode_plan(
                qwen3_5_plan_state, attn_ws, kv_cache, engine.hf_config(),
                q35_fwd, kv_page_indptr_h, R, is_pure_decode);
        };
        forward_fn.body = [&engine, &weights_qwen3_5_moe, &qwen3_5_la_ws,
                           &qwen3_5_moe_ws, &qwen3_5_state_cache,
                           &qwen3_5_plan_state,
                           q35moe_tp_size, q35moe_tp_comm](
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
            const std::uint8_t* mask_d, const std::int32_t* mask_indptr_d,
            const std::int32_t* slot_ids_h, const std::uint8_t* is_fresh_h,
            const std::int32_t* slot_ids_d) {
            pie_cuda_driver::model::Qwen3_5ForwardCfg q35_fwd{};
            const auto& hf_q = engine.hf_config();
            const int gqa_q = hf_q.num_attention_heads /
                              std::max(1, hf_q.num_key_value_heads);
            q35_fwd.force_prefill_path =
                !(gqa_q == 1 || gqa_q == 2 || gqa_q == 3 ||
                  gqa_q == 4 || gqa_q == 8);
            q35_fwd.tp_size = q35moe_tp_size;
            q35_fwd.tp_comm = q35moe_tp_comm;
            pie_cuda_driver::model::qwen3_5_moe_forward_paged(
                weights_qwen3_5_moe, engine.hf_config(), q35_fwd,
                qwen3_5_plan_state,
                ws, qwen3_5_la_ws, qwen3_5_moe_ws,
                cache, qwen3_5_state_cache,
                attn_ws, cublas,
                tok, pos,
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                qo_indptr_h, kv_page_indptr_h,
                N, R, is_pure_decode, mask_d, mask_indptr_d,
                slot_ids_h, is_fresh_h, slot_ids_d);
        };
    } else {
        // Llama-like decode is graph-replay-safe because (a) the body
        // is host-work-free (the prepare hook hoisted DecodePlan out of
        // the capture region); (b) flashinfer's plan_info layout is
        // pinned across fires when `enable_cuda_graph=true` —
        // `padded_batch_size = max_grid_size / gdy` (stable), and the
        // int_buf offsets are deterministic from that; (c) per-fire,
        // DecodePlan only refreshes int_buf content (request_indices,
        // kv_tile_indices, o_indptr, block_valid_mask) at the same
        // device offsets, so the captured kernel reads fresh data through
        // its stable pointer args.
        forward_fn.graph_safe = true;
        forward_fn.prepare = [&engine, &kv_cache, &fwd_cfg, &llama_plan](
            pie_cuda_driver::AttentionWorkspace& attn_ws,
            const std::uint32_t* kv_page_indptr_h,
            int R, bool is_pure_decode) {
            pie_cuda_driver::model::prepare_llama_like_decode_plan(
                llama_plan, attn_ws, kv_cache, engine.hf_config(),
                fwd_cfg, kv_page_indptr_h, R, is_pure_decode);
        };
        forward_fn.body = [&engine, &weights_llama, fwd_cfg, &llama_plan](
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
            const std::uint8_t* mask_d, const std::int32_t* mask_indptr_d,
            const std::int32_t* slot_ids_h, const std::uint8_t* is_fresh_h,
            const std::int32_t* slot_ids_d) {
            pie_cuda_driver::model::llama_like_forward_paged(
                weights_llama, engine.hf_config(), fwd_cfg, llama_plan,
                ws, cache, attn_ws, cublas,
                tok, pos,
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                qo_indptr_h, kv_page_indptr_h,
                N, R, is_pure_decode, mask_d, mask_indptr_d);
        };
    }

    pie_cuda_driver::Executor executor{
        engine, ws, kv_cache, attn_ws, cublas,
        max_workspace_tokens, persistent_inputs, verbose, std::move(forward_fn),
        use_cuda_graphs ? &graph_cache : nullptr,
        /*tp_comm=*/tp_comm_ptr,
        /*tp_cpu_gate_key=*/{},
        /*slot_alloc=*/{},
        /*response_builder=*/{},
    };
    executor.tp_cpu_gate_key = cfg.distributed.nccl_unique_id_hex;
    // Speculation lives entirely in the runtime. The driver runs
    // forward passes; the runtime's `scheduler.speculation_depth`
    // toml knob controls per-ctx chain depth.
    // Size the linear-attn slot allocator only when this arch actually
    // uses a state cache. Default-constructed (max_slots=0) on every
    // other arch — handle_fire_batch's `use_slots` predicate stays false
    // and the body's slot_ids/is_fresh args remain nullptr.
    if ((is_qwen3_5_arch || is_qwen3_5_moe_arch) &&
        qwen3_5_state_cache.max_slots() > 0) {
        executor.slot_alloc.reset(qwen3_5_state_cache.max_slots());
    }
    if (verbose && use_cuda_graphs) {
        std::cerr << "[pie-driver-cuda] CUDA graphs enabled (experimental)\n";
    }

    // TP ranks run as independent driver instances. Followers can reach the
    // first NCCL receive before rank 0 has finished building its CUDA serving
    // state; posting that idle receive while the leader is still allocating can
    // show as a persistent 100% GPU-util spin and has reproduced startup
    // wedges. Rendezvous on CPU after all persistent allocations are complete,
    // then publish rank-0 readiness and let followers enter the NCCL loop.
    tp_startup_cpu_barrier(cfg);

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
        };
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
        unregister_server(server_p.get());
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
    stop_servers();
}
