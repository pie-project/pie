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
#include "custom_all_reduce.hpp"
#include "distributed.hpp"
#include "engine.hpp"
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
#include "request_handler.hpp"
#include "shmem_ipc.hpp"

namespace {

std::atomic<pie_cuda_driver::ShmemServer*> g_server{nullptr};
std::atomic<bool> g_follower_stop{false};

void on_signal(int) {
    if (auto* s = g_server.load()) s->stop();
    g_follower_stop.store(true);
}

// Extract the integer device index from a "cuda:N" / "cuda" string.
int parse_device_index(const std::string& device) {
    const auto colon = device.find(':');
    if (colon == std::string::npos) return 0;
    return std::stoi(device.substr(colon + 1));
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
    auto engine = pie_cuda_driver::Engine::load(cfg, tp_comm);
    const auto& mt_for_parity = engine.hf_config().model_type;
    const bool is_gpt_oss  = (mt_for_parity == "gpt_oss");
    const bool is_gemma3n  = (mt_for_parity == "gemma3n" || mt_for_parity == "gemma3n_text");
    const bool is_qwen3_5  = (mt_for_parity == "qwen3_5" || mt_for_parity == "qwen3_5_text");
    const bool is_qwen3_5_moe = (mt_for_parity == "qwen3_5_moe" || mt_for_parity == "qwen3_5_moe_text"
                                  || mt_for_parity == "qwen3_moe");
    const bool parity_has_linear_attn =
        (mt_for_parity == "qwen3_5" || mt_for_parity == "qwen3_5_text" ||
         mt_for_parity == "qwen3_5_moe" || mt_for_parity == "qwen3_5_moe_text");
    const bool is_gemma2_arch = (mt_for_parity == "gemma2");
    const bool is_gemma3_arch = (mt_for_parity == "gemma3" || mt_for_parity == "gemma3_text");
    const bool is_gemma_arch = is_gemma2_arch || is_gemma3_arch;
    const bool is_mixtral_arch = (mt_for_parity == "mixtral");
    {
        const bool supported =
            mt_for_parity == "qwen3" || is_qwen3_5 || is_qwen3_5_moe
         || mt_for_parity == "qwen2"
         || mt_for_parity == "llama" || mt_for_parity == "llama3"
         || mt_for_parity == "mistral" || mt_for_parity == "mistral3"
         || mt_for_parity == "olmo2" || mt_for_parity == "olmo3"
         || mt_for_parity == "phi3"
         || is_gpt_oss
         || is_gemma3n
         || is_gemma_arch
         || is_mixtral_arch;
        if (!supported) {
            std::cerr << "[parity] unsupported model_type: " << mt_for_parity << "\n";
            return 2;
        }
        if ((is_gpt_oss || is_gemma3n || is_qwen3_5 || is_qwen3_5_moe ||
             is_gemma_arch || is_mixtral_arch) && !paged) {
            std::cerr << "[parity] " << mt_for_parity << " requires --parity-paged\n";
            return 2;
        }
    }
    pie_cuda_driver::model::Qwen3Weights weights;
    pie_cuda_driver::model::MixtralWeights weights_mixtral;
    pie_cuda_driver::model::Gemma3nWeights weights_gemma3n;
    pie_cuda_driver::model::Qwen3_5Weights weights_qwen3_5;
    pie_cuda_driver::model::Qwen3_5MoeWeights weights_qwen3_5_moe;
    pie_cuda_driver::model::Gemma2Weights weights_gemma2;
    if (is_gpt_oss) {
        weights_mixtral = pie_cuda_driver::model::bind_gpt_oss(engine);
    } else if (is_mixtral_arch) {
        weights_mixtral = pie_cuda_driver::model::bind_mixtral(engine);
    } else if (is_gemma3n) {
        weights_gemma3n = pie_cuda_driver::model::bind_gemma3n(engine);
    } else if (is_qwen3_5) {
        weights_qwen3_5 = pie_cuda_driver::model::bind_qwen3_5(engine);
    } else if (is_qwen3_5_moe) {
        weights_qwen3_5_moe = pie_cuda_driver::model::bind_qwen3_5_moe(engine);
    } else if (is_gemma2_arch) {
        weights_gemma2 = pie_cuda_driver::model::bind_gemma2(engine);
    } else if (is_gemma3_arch) {
        weights_gemma2 = pie_cuda_driver::model::bind_gemma3(engine);
    } else if (mt_for_parity == "olmo2" || mt_for_parity == "olmo3") {
        weights = pie_cuda_driver::model::bind_olmo3(engine);
    } else if (mt_for_parity == "phi3") {
        weights = pie_cuda_driver::model::bind_phi3(engine);
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

    // TP-aware workspace: per-rank q/k/v/attn_out/gate/up shrink by tp_size,
    // matching the sharded weights produced by the loader. Replicated
    // buffers (y, norm_x/y, logits) keep their full widths.
    const int parity_T = std::max(1, cfg.distributed.tp_size);
    const int parity_Hq = (engine.hf_config().num_attention_heads *
                           engine.hf_config().head_dim) / parity_T;
    const int parity_Hk = (engine.hf_config().num_key_value_heads *
                           engine.hf_config().head_dim) / parity_T;
    const int parity_I  = engine.hf_config().intermediate_size / parity_T;
    auto ws = pie_cuda_driver::model::Qwen3Workspace::allocate_full(
        engine.hf_config(), N, parity_I, parity_Hq, parity_Hk);
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
        // PIE_PARITY_BENCH_DECODE adds K extra decode steps to the run, so
        // the KV cache must be sized to accommodate N + K tokens (otherwise
        // the bench overflows page-0 once kv_n > page_size and the kernel
        // accesses unallocated pages).
        int bench_extra_decode = 0;
        if (const char* dbg = std::getenv("PIE_PARITY_BENCH_DECODE")) {
            bench_extra_decode = std::max(0, std::atoi(dbg));
        }
        const int total_pages =
            (N + bench_extra_decode + page_size - 1) / page_size;

        auto cache = pie_cuda_driver::KvCache::allocate(
            engine.hf_config().num_hidden_layers,
            std::max(total_pages, 1),
            page_size,
            engine.hf_config().num_key_value_heads / parity_T,
            engine.hf_config().head_dim);

        auto parity_attn_ws = pie_cuda_driver::AttentionWorkspace::allocate();

        // qwen3_5 / qwen3_5_moe need their own scratch + state caches.
        pie_cuda_driver::model::Qwen3_5LinearAttnWorkspace q35_la_ws;
        pie_cuda_driver::Qwen3_5StateCache q35_state_cache;
        pie_cuda_driver::model::Qwen3_5MoeMlpWorkspace q35_moe_ws;
        if (is_qwen3_5 || is_qwen3_5_moe) {
            const auto& cfg_q = engine.hf_config();
            if (parity_has_linear_attn) {
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
            }
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
        } else if (is_mixtral_arch) {
            // Plain Mixtral: no QKV bias, standard RoPE, no sliding window.
            // The MoE block reads `num_experts` / `top_k` directly from
            // the call site below, so the LlamaLikeForwardCfg only carries
            // the attention knobs.
            fwd_cfg.use_qkv_bias    = false;
            fwd_cfg.rope_kind       = pie_cuda_driver::model::RopeKind::Standard;
            fwd_cfg.force_prefill_path = !decode_after_prefill;
        } else if (mt_for_parity == "olmo2" || mt_for_parity == "olmo3") {
            // OLMo-2/3: post-norm + q/k-norm. OLMo-3 also has YaRN +
            // per-layer sliding window.
            const auto& hf = engine.hf_config();
            fwd_cfg.use_qk_norm    = true;
            fwd_cfg.norm_placement = pie_cuda_driver::model::NormPlacement::Post;
            switch (hf.rope_scaling_kind) {
                case pie_cuda_driver::HfConfig::RopeScaling::Llama3:
                    fwd_cfg.rope_kind = pie_cuda_driver::model::RopeKind::YaRN;
                    break;
                case pie_cuda_driver::HfConfig::RopeScaling::OriginalYaRN:
                    fwd_cfg.rope_kind = pie_cuda_driver::model::RopeKind::YaRNOriginal;
                    break;
                default:
                    fwd_cfg.rope_kind = pie_cuda_driver::model::RopeKind::Standard;
                    break;
            }
            fwd_cfg.yarn_factor                = hf.rope_factor;
            fwd_cfg.yarn_low_freq_factor       = hf.rope_low_freq_factor;
            fwd_cfg.yarn_high_freq_factor      = hf.rope_high_freq_factor;
            fwd_cfg.yarn_beta_fast             = hf.rope_beta_fast;
            fwd_cfg.yarn_beta_slow             = hf.rope_beta_slow;
            fwd_cfg.yarn_attention_factor      = hf.rope_attention_factor;
            fwd_cfg.yarn_original_max_position = hf.rope_original_max_position;
            fwd_cfg.sliding_window             = hf.sliding_window;
            for (const auto& t : hf.layer_types) {
                fwd_cfg.per_layer_window_left.push_back(
                    (t == "sliding_attention") ? hf.sliding_window : -1);
            }
            fwd_cfg.force_prefill_path = !decode_after_prefill;
        }
        // TP knobs apply to every llama_like-shaped fwd_cfg user
        // (mixtral / gpt_oss / fall-through llama path).
        fwd_cfg.tp_size = cfg.distributed.tp_size;
        fwd_cfg.tp_comm = tp_comm;

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

            if (is_gpt_oss || is_mixtral_arch) {
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
                gemma3n_fwd.tp_size = cfg.distributed.tp_size;
                gemma3n_fwd.tp_comm = tp_comm;
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
            } else if (is_gemma_arch) {
                // Gemma2 / Gemma3 share the post-norm + extra-norm forward
                // in `gemma2_forward_paged`. We mirror the runtime serve
                // dispatch in main() — TP knobs flow via `Gemma2ForwardCfg`.
                pie_cuda_driver::model::Gemma2ForwardCfg gemma_fwd{};
                const auto& hf = engine.hf_config();
                gemma_fwd.query_pre_attn_scalar = hf.gemma_query_pre_attn_scalar;
                gemma_fwd.final_logit_softcap   = hf.gemma_final_logit_softcap;
                gemma_fwd.attn_logit_softcap    = hf.gemma_attn_logit_softcap;
                gemma_fwd.use_qk_norm = is_gemma3_arch;
                const int gqa = hf.num_attention_heads / hf.num_key_value_heads;
                gemma_fwd.force_prefill_path =
                    !(gqa == 1 || gqa == 2 || gqa == 3 || gqa == 4 || gqa == 8);
                if (!hf.layer_types.empty()) {
                    gemma_fwd.per_layer_window_left.reserve(hf.layer_types.size());
                    gemma_fwd.per_layer_rope_theta.reserve(hf.layer_types.size());
                    for (const auto& t : hf.layer_types) {
                        const bool is_sliding = (t == "sliding_attention");
                        gemma_fwd.per_layer_window_left.push_back(
                            is_sliding ? hf.sliding_window : -1);
                        const float theta = (is_sliding && hf.rope_local_base_freq > 0.f)
                            ? hf.rope_local_base_freq
                            : hf.rope_theta;
                        gemma_fwd.per_layer_rope_theta.push_back(theta);
                    }
                }
                gemma_fwd.tp_size = cfg.distributed.tp_size;
                gemma_fwd.tp_comm = tp_comm;
                pie_cuda_driver::model::gemma2_forward_paged(
                    weights_gemma2, hf, gemma_fwd,
                    ws, cache, parity_attn_ws, cublas,
                    tok_d, pos_d,
                    d_qo, d_pi, d_pp, d_lpl,
                    /*qo_indptr_h=*/h_qo.data(),
                    /*kv_page_indptr_h=*/h_pp.data(),
                    /*total_tokens=*/total_n, /*num_requests=*/1,
                    /*is_pure_decode=*/is_decode);
            } else {
                // Route the parity path through `llama_like_forward_paged`
                // so it shares the TP-aware forward used by the runtime
                // serve path. The outer `fwd_cfg` was already specialised
                // per-arch above (OLMo-2/3 sets post-norm + use_qk_norm +
                // YaRN params; default falls through with attention_bias
                // / use_qk_norm read from HfConfig). For the legacy llama
                // path we just patch in the bits the parity helper used to
                // set itself.
                pie_cuda_driver::model::LlamaLikeForwardCfg parity_fwd = fwd_cfg;
                if (mt_for_parity != "olmo2" && mt_for_parity != "olmo3") {
                    parity_fwd.use_qk_norm  = engine.hf_config().use_qk_norm;
                    parity_fwd.use_qkv_bias = engine.hf_config().attention_bias;
                }
                parity_fwd.tp_size = cfg.distributed.tp_size;
                parity_fwd.tp_comm = tp_comm;
                pie_cuda_driver::model::LlamaLikePlanState parity_plan;
                pie_cuda_driver::model::prepare_llama_like_decode_plan(
                    parity_plan, parity_attn_ws, cache, engine.hf_config(),
                    parity_fwd, h_pp.data(), /*num_requests=*/1, is_decode);
                pie_cuda_driver::model::llama_like_forward_paged(
                    weights, engine.hf_config(), parity_fwd, parity_plan,
                    ws, cache, parity_attn_ws, cublas,
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

int main(int argc, char** argv) {
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
                     "Run the paged forward path (BPIQ-shaped KV layout)");
    parity->add_flag("--parity-decode-after-prefill", parity_decode_after_prefill,
                     "After prefill on the first N-1 tokens, run a single "
                     "qo_len=1 decode step at position N-1 and dump that "
                     "step's logits. Exercises the decode kernel + KV-cache "
                     "read path in addition to prefill. Requires --parity-paged.");

    bool use_cuda_graphs = false;
    app.add_flag("--cuda-graphs", use_cuda_graphs,
                 "Capture decode forward into CUDA graphs and replay per "
                 "shape bucket. Experimental; default off.");

    int control_fd = -1;
    app.add_option("--control-fd", control_fd,
                   "Pre-opened SOCK_SEQPACKET fd for the wrapper control "
                   "channel (copy_d2h / copy_h2d / copy_d2d / copy_h2h).");

    // Tensor-parallel knobs. These override [distributed] in the TOML when
    // present so the wrapper can launch ad-hoc TP groups without rewriting
    // the config file. Empty unique-id means "fall back to TOML".
    int cli_tp_size = -1, cli_tp_rank = -1;
    std::string cli_nccl_unique_id_hex;
    app.add_option("--tp-size", cli_tp_size,
                   "Tensor-parallel world size (overrides [distributed].tp_size).");
    app.add_option("--tp-rank", cli_tp_rank,
                   "This process's rank in the TP group (0..tp_size).");
    app.add_option("--nccl-unique-id-hex", cli_nccl_unique_id_hex,
                   "Hex-encoded ncclUniqueId shared across all ranks of the "
                   "TP group. Only required when tp_size > 1.");

    CLI11_PARSE(app, argc, argv);

    auto cfg = pie_cuda_driver::load_config(config_path);
    if (cli_tp_size >= 1) cfg.distributed.tp_size = cli_tp_size;
    if (cli_tp_rank >= 0) cfg.distributed.tp_rank = cli_tp_rank;
    if (!cli_nccl_unique_id_hex.empty())
        cfg.distributed.nccl_unique_id_hex = cli_nccl_unique_id_hex;
    // Followers (rank > 0) must be told the unique-id explicitly. Rank 0
    // is allowed to auto-generate it; the wrapper relays the resulting
    // `NCCL_UID <hex>` stdout line to the followers.
    if (cfg.distributed.tp_size > 1 &&
        cfg.distributed.tp_rank > 0 &&
        cfg.distributed.nccl_unique_id_hex.empty()) {
        std::cerr << "[pie-driver-cuda] rank " << cfg.distributed.tp_rank
                  << " requires --nccl-unique-id-hex "
                  << "(or [distributed].nccl_unique_id_hex)\n";
        return 1;
    }

    // Parity argument validation runs early so we don't go through NCCL
    // bootstrap only to fail on bad CLI. The actual parity dispatch is
    // deferred until after the NCCL comm is up so a TP-mode parity test
    // can drive collectives.
    if (!parity_tokens.empty()) {
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
    // handshake line consumed by the Python wrapper.
    std::cerr << "[pie-driver-cuda] config loaded\n"
              << "  shmem.name      = " << cfg.shmem.name << "\n"
              << "  shmem.num_slots = " << cfg.shmem.num_slots << "\n"
              << "  model.hf_repo   = " << cfg.model.hf_repo << "\n"
              << "  model.device    = " << cfg.model.device << "\n"
              << "  model.dtype     = " << cfg.model.dtype << "\n"
              << "  tp_size         = " << cfg.distributed.tp_size << "\n"
              << "  tp_rank         = " << cfg.distributed.tp_rank << "\n";

    // Bind to the requested CUDA device *before* NCCL init — ncclCommInitRank
    // captures whatever is current on the calling thread. Engine::load also
    // sets the device, but doing it here lets the smoke test below run on the
    // right GPU and keeps the NCCL comm bound consistently.
    const int dev_id = parse_device_index(cfg.model.device);
    CUDA_CHECK(cudaSetDevice(dev_id));

    pie_cuda_driver::NcclComm tp_comm;
    if (cfg.distributed.tp_size > 1) {
        // ncclGetUniqueId opens a TCP bootstrap listener inside the calling
        // process — it must outlive the rendezvous. Rank 0 therefore
        // generates the id itself (when no id was passed in), prints it on
        // stdout for the wrapper to relay, then proceeds straight into
        // ncclCommInitRank. Followers receive the id from the wrapper via
        // --nccl-unique-id-hex / [distributed].nccl_unique_id_hex.
        ncclUniqueId uid;
        if (cfg.distributed.tp_rank == 0 &&
            cfg.distributed.nccl_unique_id_hex.empty()) {
            NCCL_CHECK(ncclGetUniqueId(&uid));
            const auto hex = pie_cuda_driver::nccl_unique_id_to_hex(uid);
            // Wrapper greps stdout for `^NCCL_UID `; arrival is ordered
            // before any READY line because we init NCCL first.
            std::cout << "NCCL_UID " << hex << std::endl;
        } else {
            if (cfg.distributed.nccl_unique_id_hex.empty()) {
                std::cerr << "[pie-driver-cuda] rank " << cfg.distributed.tp_rank
                          << " requires --nccl-unique-id-hex (only rank 0 may "
                          << "auto-generate)\n";
                return 1;
            }
            uid = pie_cuda_driver::nccl_unique_id_from_hex(
                cfg.distributed.nccl_unique_id_hex);
        }
        tp_comm = pie_cuda_driver::NcclComm(
            cfg.distributed.tp_size, cfg.distributed.tp_rank, uid);
        std::cerr << "[pie-driver-cuda] NCCL comm initialised "
                  << "(world=" << tp_comm.world_size()
                  << ", rank=" << tp_comm.rank() << ")\n";

        // Smoke test: every rank contributes its (rank+1) and we expect the
        // sum to equal world*(world+1)/2. Catches mis-numbered ranks and
        // device-binding bugs at startup rather than mid-fire_batch.
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
        std::cerr << "[pie-driver-cuda] NCCL smoke test ok ("
                  << h_v << "==" << expected << ")\n";
    }

    // Parity mode: every rank participates so collectives complete; only
    // rank 0 dumps logits to disk. The harness compares rank 0's output
    // against a single-GPU reference run.
    if (!parity_tokens.empty()) {
        const std::string out_path = (cfg.distributed.tp_rank == 0)
            ? parity_out
            : (parity_out + ".rank" + std::to_string(cfg.distributed.tp_rank));
        return run_parity(cfg, parity_tokens, out_path, parity_paged,
                          parity_decode_after_prefill,
                          (cfg.distributed.tp_size > 1) ? &tp_comm : nullptr);
    }

    // Both leader (rank 0) and followers (rank > 0) load the model + build
    // their TP-aware workspace. The split happens at the end of `main`:
    // rank 0 enters `server.serve_forever`; followers enter
    // `tp_follower_serve`, which consumes the broadcasts the leader emits
    // from inside `handle_fire_batch`.
    const bool is_tp_follower =
        cfg.distributed.tp_size > 1 && cfg.distributed.tp_rank > 0;

    auto engine = pie_cuda_driver::Engine::load(
        cfg, (cfg.distributed.tp_size > 1) ? &tp_comm : nullptr);

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
    const bool is_qwen3_5_moe_arch =
        (mt_for_bind == "qwen3_5_moe" || mt_for_bind == "qwen3_5_moe_text" ||
         mt_for_bind == "qwen3_moe");
    // Qwen3-MoE is full-attention only; the Qwen3.5/3.6 family adds a
    // hybrid linear-attn schedule and an always-on shared expert. The
    // bind / forward share the same struct, so we only need to gate the
    // linear-attn workspace + state cache and the shared-expert
    // workspace allocations.
    const bool qwen3_5_moe_has_linear_attn =
        (mt_for_bind == "qwen3_5_moe" || mt_for_bind == "qwen3_5_moe_text");

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
    } else if (is_gemma3n_arch) {
        // Per-layer intermediate (HF stores it as a list); head_dim is
        // uniform across layers on gemma3n, so KV cache can use the
        // standard allocator.
        for (int v : weights_gemma3n.per_layer_intermediate) {
            if (v > max_mlp_intermediate) max_mlp_intermediate = v;
        }
    }
    // Tensor-parallel: shrink the per-rank workspace and KV cache by
    // tp_size. Activations on the sharded paths (q/k/v, attn_out, gate,
    // up) are 1/T of the unsharded width; the KV cache stores 1/T of the
    // KV heads. Replicated buffers (y, norm_x/y, logits) stay full.
    const int T_tp = cfg.distributed.tp_size;
    if (T_tp > 1) {
        if (max_mlp_intermediate % T_tp != 0 ||
            max_Hq % T_tp != 0 || max_Hk % T_tp != 0) {
            std::cerr << "[pie-driver-cuda] tp_size=" << T_tp
                      << " does not evenly divide intermediate/Hq/Hk\n";
            return 4;
        }
        max_mlp_intermediate /= T_tp;
        max_Hq /= T_tp;
        max_Hk /= T_tp;
    }
    auto ws = pie_cuda_driver::model::Qwen3Workspace::allocate_full(
        engine.hf_config(), max_workspace_tokens,
        max_mlp_intermediate, max_Hq, max_Hk);

    const int kv_heads_local =
        engine.hf_config().num_key_value_heads / std::max(1, T_tp);
    auto kv_cache =
        is_gemma4_arch
            ? pie_cuda_driver::KvCache::allocate_per_layer(
                  engine.hf_config().num_hidden_layers,
                  static_cast<int>(cfg.batching.max_num_kv_pages),
                  static_cast<int>(cfg.batching.kv_page_size),
                  kv_heads_local,
                  weights_gemma4.per_layer_head_dim,
                  weights_gemma4.kv_source_layer)
            : pie_cuda_driver::KvCache::allocate(
                  engine.hf_config().num_hidden_layers,
                  static_cast<int>(cfg.batching.max_num_kv_pages),
                  static_cast<int>(cfg.batching.kv_page_size),
                  kv_heads_local,
                  engine.hf_config().head_dim_kernel);

    auto attn_ws = pie_cuda_driver::AttentionWorkspace::allocate();

    // Custom NVLink P2P all-reduce. Wraps flashinfer's vllm-style kernel
    // and routes from `NcclComm::all_reduce_bf16` for small bf16 sums
    // (the per-layer attn-O / MLP-down pattern). Built only when TP > 1
    // and registered against the workspace tensors that actually feed
    // the all-reduce (`ws.norm_x`, `ws.norm_y`).
    pie_cuda_driver::CustomAllReduce custom_ar;
    if (cfg.distributed.tp_size > 1) {
        try {
            custom_ar = pie_cuda_driver::CustomAllReduce(tp_comm);
            // Both norm_x and norm_y can be the all-reduce target depending
            // on the arch (llama_like uses norm_x, qwen3_5 uses norm_y for
            // out_proj fused-residual). Register both base addresses.
            custom_ar.register_buffer(tp_comm, ws.norm_x.data(), ws.norm_x.nbytes());
            custom_ar.register_buffer(tp_comm, ws.norm_y.data(), ws.norm_y.nbytes());
            tp_comm.set_custom_all_reduce(&custom_ar);
            std::cerr << "[pie-driver-cuda] custom NVLink all-reduce ready "
                      << "(small-message fast path active)\n";
        } catch (const std::exception& e) {
            std::cerr << "[pie-driver-cuda] custom all-reduce unavailable: "
                      << e.what() << "; falling back to NCCL only\n";
            custom_ar = pie_cuda_driver::CustomAllReduce{};
        }
    }

    // Qwen3.5 / Qwen3.6-MoE linear-attention extras: per-layer state cache
    // + a per-call workspace. Inert (default-constructed) on every other
    // arch. The MoE arch additionally needs a routed-experts workspace.
    pie_cuda_driver::model::Qwen3_5LinearAttnWorkspace qwen3_5_la_ws;
    pie_cuda_driver::Qwen3_5StateCache qwen3_5_state_cache;
    pie_cuda_driver::model::Qwen3_5MoeMlpWorkspace qwen3_5_moe_ws;
    // Decode-plan state, refreshed by the forward.prepare hook on every
    // fire (outside any cudaStream capture region).
    pie_cuda_driver::model::Qwen3_5PlanState qwen3_5_plan_state;
    if (is_qwen3_5_arch || is_qwen3_5_moe_arch) {
        const auto& cfg_q = engine.hf_config();
        const int T_q35 = std::max(1, cfg.distributed.tp_size);
        const bool has_linear_attn =
            is_qwen3_5_arch || qwen3_5_moe_has_linear_attn;
        if (has_linear_attn) {
            // Per-rank head shares: linear-attn K/V heads, full-attn Q/KV
            // heads, and intermediate all divide tp_size — checked at
            // engine load. Workspace dims shrink accordingly so each rank
            // only allocates its slice.
            const int K_h_local = cfg_q.linear_num_key_heads / T_q35;
            const int V_h_local = cfg_q.linear_num_value_heads / T_q35;
            const int K_dim = K_h_local * cfg_q.linear_key_head_dim;
            const int V_dim = V_h_local * cfg_q.linear_value_head_dim;
            const int conv_dim = 2 * K_dim + V_dim;
            const int hq_local =
                (cfg_q.num_attention_heads / T_q35) * cfg_q.head_dim;
            qwen3_5_la_ws = pie_cuda_driver::model::Qwen3_5LinearAttnWorkspace::allocate(
                max_workspace_tokens, conv_dim,
                V_h_local,
                K_h_local,
                cfg_q.linear_key_head_dim,
                cfg_q.linear_value_head_dim,
                /*hq=*/hq_local);
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
            qwen3_5_state_cache = pie_cuda_driver::Qwen3_5StateCache::allocate(
                layer_is_linear, conv_dim, cfg_q.linear_conv_kernel_dim,
                V_h_local,
                cfg_q.linear_key_head_dim,
                cfg_q.linear_value_head_dim);
        }

        if (is_qwen3_5_moe_arch) {
            // Routed experts shard along the intermediate axis under TP —
            // the engine streams per-rank slices straight from the
            // safetensors mmap. Shared-expert width is 0 on Qwen3-MoE
            // (no shared expert); the workspace allocator handles 0-width
            // by skipping those buffers.
            qwen3_5_moe_ws = pie_cuda_driver::model::Qwen3_5MoeMlpWorkspace::allocate(
                max_workspace_tokens,
                cfg_q.hidden_size,
                cfg_q.num_experts,
                cfg_q.num_experts_per_tok,
                cfg_q.moe_intermediate_size / T_q35,
                cfg_q.shared_expert_intermediate_size / T_q35);
        }
    }

    auto swap_pool = pie_cuda_driver::SwapPool::allocate(
        engine.hf_config().num_hidden_layers,
        static_cast<int>(cfg.batching.swap_pool_size),
        static_cast<int>(cfg.batching.kv_page_size),
        kv_heads_local,
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
    // from issuing them). Leader-only: TP followers don't expose any RPC
    // surface to the wrapper (the wrapper only opens the control socket
    // for rank 0 anyway).
    std::thread control_thread;
    if (!is_tp_follower && control_fd >= 0) {
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

    // Leader-only: open the shmem region (followers consume their inputs
    // over NCCL), advertise capabilities to the wrapper, and register
    // signal handlers that stop the server. Followers still register the
    // signal handlers so SIGTERM flips `g_follower_stop`; they break out
    // of `tp_follower_serve` once the next broadcast completes (rank 0
    // emits a `tp_send_shutdown` on its own exit path).
    std::unique_ptr<pie_cuda_driver::ShmemServer> server_p;
    if (!is_tp_follower) {
        server_p = std::make_unique<pie_cuda_driver::ShmemServer>(
            cfg.shmem.name,
            cfg.shmem.num_slots,
            cfg.shmem.req_buf,
            cfg.shmem.resp_buf,
            cfg.shmem.spin_us);
        g_server.store(server_p.get());
    }

    std::signal(SIGINT, on_signal);
    std::signal(SIGTERM, on_signal);

    if (!is_tp_follower) {
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

        std::cerr << "[pie-driver-cuda] serving on shmem " << server_p->name()
                  << " (" << server_p->num_slots() << " slots, "
                  << "req_buf=" << server_p->req_buf_size() << ", "
                  << "resp_buf=" << server_p->resp_buf_size() << ")\n";
    } else {
        std::cerr << "[pie-driver-cuda] tp follower rank "
                  << cfg.distributed.tp_rank
                  << " ready (waiting on rank-0 broadcasts)\n";
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
        switch (hf.rope_scaling_kind) {
            case pie_cuda_driver::HfConfig::RopeScaling::Llama3:
                fwd_cfg.rope_kind = pie_cuda_driver::model::RopeKind::YaRN;
                break;
            case pie_cuda_driver::HfConfig::RopeScaling::OriginalYaRN:
                fwd_cfg.rope_kind = pie_cuda_driver::model::RopeKind::YaRNOriginal;
                break;
            default:
                fwd_cfg.rope_kind = pie_cuda_driver::model::RopeKind::Standard;
                break;
        }
        fwd_cfg.yarn_factor               = hf.rope_factor;
        fwd_cfg.yarn_low_freq_factor      = hf.rope_low_freq_factor;
        fwd_cfg.yarn_high_freq_factor     = hf.rope_high_freq_factor;
        fwd_cfg.yarn_beta_fast            = hf.rope_beta_fast;
        fwd_cfg.yarn_beta_slow            = hf.rope_beta_slow;
        fwd_cfg.yarn_attention_factor     = hf.rope_attention_factor;
        fwd_cfg.yarn_original_max_position = hf.rope_original_max_position;
        fwd_cfg.sliding_window            = hf.sliding_window;
        // flashinfer's decode kernel only instantiates GQA group sizes
        // {1, 2, 3, 4, 8}. Models like Qwen2-0.5B (group=7), Qwen2-1.5B
        // (group=6) need the prefill kernel even for decode-only fires.
        const int gqa = hf.num_attention_heads / hf.num_key_value_heads;
        const bool gqa_in_decode_set = (gqa == 1 || gqa == 2 || gqa == 3
                                        || gqa == 4 || gqa == 8);
        fwd_cfg.force_prefill_path = !gqa_in_decode_set;

        // Tensor-parallel state. tp_comm is null when tp_size == 1; the
        // forward path then takes the single-GPU branches verbatim.
        fwd_cfg.tp_size = cfg.distributed.tp_size;
        fwd_cfg.tp_comm = (cfg.distributed.tp_size > 1) ? &tp_comm : nullptr;

        // Gemma-2 / Gemma-3 forward knobs. `query_pre_attn_scalar` and
        // `final_logit_softcapping` come straight from the HF config —
        // see `loader/hf_config.cpp` for the parsing.
        gemma_fwd_cfg.query_pre_attn_scalar = hf.gemma_query_pre_attn_scalar;
        gemma_fwd_cfg.final_logit_softcap   = hf.gemma_final_logit_softcap;
        gemma_fwd_cfg.attn_logit_softcap    = hf.gemma_attn_logit_softcap;
        gemma_fwd_cfg.use_qk_norm           = (mt == "gemma3" || mt == "gemma3_text");
        gemma_fwd_cfg.force_prefill_path    = !gqa_in_decode_set;
        // Mirror the dense-llama TP plumbing: rank-aware sharded dims +
        // all-reduces in `gemma2_forward_paged`. tp_comm == nullptr at
        // tp_size == 1 keeps the original single-GPU semantics.
        gemma_fwd_cfg.tp_size = cfg.distributed.tp_size;
        gemma_fwd_cfg.tp_comm = (cfg.distributed.tp_size > 1) ? &tp_comm : nullptr;

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
                  << " rope="
                  << (fwd_cfg.rope_kind == pie_cuda_driver::model::RopeKind::YaRN
                          ? "yarn-llama3"
                      : fwd_cfg.rope_kind == pie_cuda_driver::model::RopeKind::YaRNOriginal
                          ? "yarn-original"
                          : "standard")
                  << "\n";
    }

    if (is_gemma4_arch) {
        const auto& hf = engine.hf_config();
        gemma4_fwd_cfg.final_logit_softcap = hf.gemma_final_logit_softcap;
        const int gqa = hf.num_attention_heads / hf.num_key_value_heads;
        const bool gqa_in_decode_set = (gqa == 1 || gqa == 2 || gqa == 3
                                        || gqa == 4 || gqa == 8);
        gemma4_fwd_cfg.force_prefill_path = !gqa_in_decode_set;
        gemma4_fwd_cfg.tp_size = cfg.distributed.tp_size;
        gemma4_fwd_cfg.tp_comm =
            (cfg.distributed.tp_size > 1) ? &tp_comm : nullptr;
    }

    // Build the type-erased forward closure once. The captures live in
    // `main`'s scope (weights_*, fwd_cfg, gemma_fwd_cfg) and persist for
    // the lifetime of the server.
    pie_cuda_driver::ForwardFn forward_fn;
    if (is_gemma4_arch) {
        forward_fn.body = [&engine, &weights_gemma4, gemma4_fwd_cfg](
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
    } else if (is_gemma3n_arch) {
        // Loader-only milestone: bind_gemma3n loads every tensor; the
        // forward function (AltUp predict/correct + Laurel + activation
        // sparsity + PLE input gate) is a follow-up. The stub throws
        // with a clear message at the first fire_batch.
        pie_cuda_driver::model::Gemma3nForwardCfg gemma3n_fwd_cfg{};
        gemma3n_fwd_cfg.final_logit_softcap = engine.hf_config().gemma_final_logit_softcap;
        gemma3n_fwd_cfg.tp_size = cfg.distributed.tp_size;
        gemma3n_fwd_cfg.tp_comm =
            (cfg.distributed.tp_size > 1) ? &tp_comm : nullptr;
        forward_fn.body = [&engine, &weights_gemma3n, gemma3n_fwd_cfg](
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
            pie_cuda_driver::model::gemma3n_forward_paged(
                weights_gemma3n, engine.hf_config(), gemma3n_fwd_cfg,
                ws, cache, attn_ws, cublas,
                tok, pos,
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                qo_indptr_h, kv_page_indptr_h,
                N, R, is_pure_decode, mask_d, mask_indptr_d);
        };
    } else if (is_gemma_arch) {
        forward_fn.body = [&engine, &weights_gemma, gemma_fwd_cfg](
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
    } else if (is_mixtral_arch) {
        // Mixtral reuses LlamaLikeForwardCfg for its (identical) attention
        // half. The MoE block reads num_experts / num_experts_per_tok
        // straight from HfConfig.
        const int num_experts = engine.hf_config().num_experts;
        const int top_k       = engine.hf_config().num_experts_per_tok;
        forward_fn.body = [&engine, &weights_mixtral, fwd_cfg, num_experts, top_k](
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
        // Serving path: graph-friendly decode dispatch when request_handler
        // signals pure-decode. force_prefill_path stays false; the decode
        // plan is refreshed by the prepare hook every fire.
        // forward_fn.graph_safe stays false: graph-correct dispatch
        // requires cuBLAS-on-cstream + null-stream-captured kernels +
        // confirmed plan_info invariance — see notes in graph_safe doc.
        const int q35_tp_size = cfg.distributed.tp_size;
        pie_cuda_driver::NcclComm* q35_tp_comm =
            (cfg.distributed.tp_size > 1) ? &tp_comm : nullptr;
        forward_fn.prepare = [&engine, &kv_cache, &qwen3_5_plan_state,
                              q35_tp_size, q35_tp_comm](
            pie_cuda_driver::AttentionWorkspace& attn_ws,
            const std::uint32_t* kv_page_indptr_h,
            int R, bool is_pure_decode) {
            pie_cuda_driver::model::Qwen3_5ForwardCfg q35_fwd{};
            q35_fwd.force_prefill_path = false;
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
            const std::uint8_t* mask_d, const std::int32_t* mask_indptr_d) {
            pie_cuda_driver::model::Qwen3_5ForwardCfg q35_fwd{};
            q35_fwd.force_prefill_path = false;
            q35_fwd.tp_size = q35_tp_size;
            q35_fwd.tp_comm = q35_tp_comm;
            pie_cuda_driver::model::qwen3_5_forward_paged(
                weights_qwen3_5, engine.hf_config(), q35_fwd, qwen3_5_plan_state,
                ws, qwen3_5_la_ws, cache, qwen3_5_state_cache,
                attn_ws, cublas,
                tok, pos,
                qo_indptr, kv_page_indices, kv_page_indptr, kv_last_page_lens,
                qo_indptr_h, kv_page_indptr_h,
                N, R, is_pure_decode, mask_d, mask_indptr_d);
        };
    } else if (is_qwen3_5_moe_arch) {
        const int q35moe_tp_size = cfg.distributed.tp_size;
        pie_cuda_driver::NcclComm* q35moe_tp_comm =
            (cfg.distributed.tp_size > 1) ? &tp_comm : nullptr;
        forward_fn.prepare = [&engine, &kv_cache, &qwen3_5_plan_state,
                              q35moe_tp_size, q35moe_tp_comm](
            pie_cuda_driver::AttentionWorkspace& attn_ws,
            const std::uint32_t* kv_page_indptr_h,
            int R, bool is_pure_decode) {
            pie_cuda_driver::model::Qwen3_5ForwardCfg q35_fwd{};
            q35_fwd.force_prefill_path = false;
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
            const std::uint8_t* mask_d, const std::int32_t* mask_indptr_d) {
            pie_cuda_driver::model::Qwen3_5ForwardCfg q35_fwd{};
            q35_fwd.force_prefill_path = false;
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
                N, R, is_pure_decode, mask_d, mask_indptr_d);
        };
    } else {
        // llama_like serving path: prepare hook hoists the decode plan
        // out of the body so the body is host-work-free. graph_safe is
        // intentionally LEFT FALSE — flashinfer's DecodePlan bakes
        // KV-size-dependent scalars (`padded_batch_size`, `split_kv`,
        // …) into the per-fire kernel-arg buffer; refreshing it via
        // `prepare` keeps the buffer up-to-date for direct dispatch but
        // a captured graph hardcodes whatever value was live at capture
        // time, so as KV grows past the capture-time bucket the kernels
        // misread the workspace and emit gibberish (validated at
        // 256-token decode). Re-flipping requires either (a) caching
        // multiple graphs per (R, kv-size band), or (b) making the
        // flashinfer decode kernel re-read all plan scalars from a
        // persistent device buffer at replay time.
        static pie_cuda_driver::model::LlamaLikePlanState s_llama_plan;
        forward_fn.graph_safe = false;
        forward_fn.prepare = [&engine, &kv_cache, &fwd_cfg](
            pie_cuda_driver::AttentionWorkspace& attn_ws,
            const std::uint32_t* kv_page_indptr_h,
            int R, bool is_pure_decode) {
            pie_cuda_driver::model::prepare_llama_like_decode_plan(
                s_llama_plan, attn_ws, kv_cache, engine.hf_config(),
                fwd_cfg, kv_page_indptr_h, R, is_pure_decode);
        };
        forward_fn.body = [&engine, &weights_llama, fwd_cfg](
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
                weights_llama, engine.hf_config(), fwd_cfg, s_llama_plan,
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
        (cfg.distributed.tp_size > 1) ? &tp_comm : nullptr,
    };
    if (use_cuda_graphs) {
        std::cerr << "[pie-driver-cuda] CUDA graphs enabled (experimental)\n";
    }

    if (is_tp_follower) {
        // Block on rank-0 broadcasts; each fire runs the same forward path
        // as the leader so the all-reduces inside complete on both sides.
        pie_cuda_driver::tp_follower_serve(fwd_ctx, g_follower_stop);
        std::cerr << "[pie-driver-cuda] tp follower rank "
                  << cfg.distributed.tp_rank << " shutting down\n";
    } else {
        server_p->serve_forever([&](const pie_cuda_driver::SlotRequest& req,
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

        // Tell every TP follower to exit before we tear down our comm.
        // Done from the leader only so followers see exactly one sentinel.
        if (cfg.distributed.tp_size > 1) {
            pie_cuda_driver::tp_send_shutdown(tp_comm);
        }
    }

    if (control_thread.joinable()) {
        // Closing the wrapper-side fd makes recv() return 0 and the thread
        // exits cleanly. We don't have that fd here, so close ours; SEQPACKET
        // will EOF on either side closing.
        ::close(control_fd);
        control_thread.join();
    }
    return 0;
}
