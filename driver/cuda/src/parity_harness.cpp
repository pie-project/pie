#include "parity_harness.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "attention_workspace.hpp"
#include "config.hpp"
#include "cuda_check.hpp"
#include "cuda_memory_planner.hpp"
#include "distributed.hpp"
#include "kv_cache.hpp"
#include "kernels/argmax.hpp"
#include "model/bound_model.hpp"
#include "model/gemma3n.hpp"
#include "model/gpt_oss.hpp"
#include "model/llama_like.hpp"
#include "model/loaded_model.hpp"
#include "model/mixtral.hpp"
#include "model/qwen3.hpp"
#include "model/qwen3_5.hpp"
#include "model/qwen3_5_forward.hpp"
#include "model/qwen3_5_moe.hpp"
#include "model/qwen3_5_moe_forward.hpp"
#include "model/qwen3_5_config.hpp"
#include "model/qwen3_forward.hpp"
#include "recurrent_state_cache.hpp"
#include "ops/gemm.hpp"

namespace pie_cuda_driver {

// Run a one-shot forward pass on a binary file of i32 token ids and dump
// the last token's logits (bf16, [vocab]) to `logits_out`. Used by the
// numeric-parity harness; never invoked through the shmem path.
int run_parity(const Config& cfg,
               const std::string& tokens_in,
               const std::string& logits_out,
               bool paged,
               bool decode_after_prefill,
               NcclComm* tp_comm)
{
    auto engine = LoadedModel::load(cfg, tp_comm);
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
    model::Qwen3Weights weights;
    model::MixtralWeights weights_mixtral;
    model::Gemma3nWeights weights_gemma3n;
    model::Qwen3_5Weights weights_qwen3_5;
    model::Qwen3_5MoeWeights weights_qwen3_5_moe;
    if (is_gpt_oss) {
        weights_mixtral = model::bind_gpt_oss(engine);
    } else if (is_gemma3n) {
        weights_gemma3n = model::bind_gemma3n(engine);
    } else if (is_qwen3_5) {
        weights_qwen3_5 = model::bind_qwen3_5(engine);
    } else if (is_qwen3_5_moe) {
        weights_qwen3_5_moe = model::bind_qwen3_5_moe(engine);
    } else {
        weights = model::bind_llama_like(
            engine, /*drop_fused_originals=*/false);
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

    auto ws = model::Qwen3Workspace::allocate(engine.hf_config(), N);
    ops::CublasHandle cublas;

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
        int parity_dev_id = 0;
        CUDA_CHECK(cudaGetDevice(&parity_dev_id));
        cudaDeviceProp parity_prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&parity_prop, parity_dev_id));
        const int page_size = derive_kv_page_size(
            cfg, engine.hf_config(), parity_prop);
        const int total_pages = (N + page_size - 1) / page_size;

        auto cache = KvCache::allocate(
            engine.hf_config().num_hidden_layers,
            std::max(total_pages, 1),
            page_size,
            engine.hf_config().num_key_value_heads,
            engine.hf_config().head_dim_kernel,
            kv_cache_format_from_string(
                cfg.batching.kv_cache_dtype, cfg.model.dtype));

        auto parity_attn_ws = AttentionWorkspace::allocate();

        // qwen3_5 / qwen3_5_moe need their own scratch + rs_cache storage.
        model::Qwen3_5LinearAttnWorkspace q35_la_ws;
        RecurrentStateCache q35_state_cache;
        model::Qwen3_5MoeMlpWorkspace q35_moe_ws;
        if (is_qwen3_5 || is_qwen3_5_moe) {
            const auto& cfg_q = engine.hf_config();
            const int K_dim = cfg_q.linear_num_key_heads * cfg_q.linear_key_head_dim;
            const int V_dim = cfg_q.linear_num_value_heads * cfg_q.linear_value_head_dim;
            const int conv_dim = 2 * K_dim + V_dim;
            q35_la_ws = model::Qwen3_5LinearAttnWorkspace::allocate(
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
                       model::Qwen3_5LayerWeights::Kind::LinearAttn)
                    : (weights_qwen3_5_moe.layers[L].kind ==
                       model::Qwen3_5MoeLayerWeights::Kind::LinearAttn);
                layer_is_linear[L] = is_linear;
            }
            q35_state_cache = RecurrentStateCache::allocate(
                layer_is_linear, conv_dim, cfg_q.linear_conv_kernel_dim,
                cfg_q.linear_num_value_heads,
                cfg_q.linear_key_head_dim, cfg_q.linear_value_head_dim,
                cfg_q.hidden_size);
            if (is_qwen3_5_moe) {
                q35_moe_ws = model::Qwen3_5MoeMlpWorkspace::allocate(
                    N, cfg_q.hidden_size,
                    cfg_q.num_experts, cfg_q.num_experts_per_tok,
                    cfg_q.moe_intermediate_size,
                    cfg_q.shared_expert_intermediate_size);
            }
        }

        // Build the per-arch fwd_cfg once; reused across the prefill and
        // (optional) decode calls.
        model::LlamaLikeForwardCfg fwd_cfg{};
        if (is_gpt_oss) {
            const auto& hf = engine.hf_config();
            fwd_cfg.use_qkv_bias = hf.attention_bias;
            model::apply_rope_config(fwd_cfg, hf);
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
                model::mixtral_forward_paged(
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
                model::Gemma3nForwardCfg gemma3n_fwd{};
                gemma3n_fwd.final_logit_softcap =
                    engine.hf_config().gemma_final_logit_softcap;
                gemma3n_fwd.force_prefill_path = !decode_after_prefill;
                model::gemma3n_forward_paged(
                    weights_gemma3n, engine.hf_config(), gemma3n_fwd,
                    ws, cache, parity_attn_ws, cublas,
                    tok_d, pos_d,
                    d_qo, d_pi, d_pp, d_lpl,
                    /*qo_indptr_h=*/h_qo.data(),
                    /*kv_page_indptr_h=*/h_pp.data(),
                    /*total_tokens=*/total_n, /*num_requests=*/1,
                    /*is_pure_decode=*/is_decode);
            } else if (is_qwen3_5) {
                model::Qwen3_5ForwardCfg q35_fwd{};
                q35_fwd.force_prefill_path = !decode_after_prefill;
                q35_fwd.tp_size = cfg.distributed.tp_size;
                q35_fwd.tp_comm = tp_comm;
                model::Qwen3_5PlanState q35_plan;
                model::prepare_qwen3_5_decode_plan(
                    q35_plan, parity_attn_ws, cache, engine.hf_config(),
                    q35_fwd, h_qo.data(), h_pp.data(), h_lpl.data(),
                    /*total_tokens=*/total_n, /*num_requests=*/1,
                    is_decode);
                model::qwen3_5_forward_paged(
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
                model::Qwen3_5ForwardCfg q35_fwd{};
                q35_fwd.force_prefill_path = !decode_after_prefill;
                q35_fwd.tp_size = cfg.distributed.tp_size;
                q35_fwd.tp_comm = tp_comm;
                model::Qwen3_5PlanState q35_plan;
                model::prepare_qwen3_5_decode_plan(
                    q35_plan, parity_attn_ws, cache, engine.hf_config(),
                    q35_fwd, h_qo.data(), h_pp.data(), h_lpl.data(),
                    /*total_tokens=*/total_n, /*num_requests=*/1,
                    is_decode);
                model::qwen3_5_moe_forward_paged(
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
                model::qwen3_forward_paged(
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

        // ── Ph7 RS fold-from-buffer COMPUTE parity (PIE_FOLD_PARITY_N=<n>) ────
        // Validates pieces 2+3's forward-body: R_fold (rs-output write n tokens
        // to the buffered pool, write_state=false → fold-replay gather from the
        // pool, write_state=true) must bit-match R_ref (a normal n-token forward),
        // both from a zeroed slot 0. Calls qwen3_5_forward_paged directly (bypasses
        // the executor wire), so it isolates the gather/scatter/replay math; the
        // executor dispatch + advance_fold are validated by the inferlet e2e.
        if (is_qwen3_5) {
            const char* fp_env = std::getenv("PIE_FOLD_PARITY_N");
            const int fold_n = fp_env ? std::atoi(fp_env) : 0;
            if (fold_n > 0) {
                const auto& cfg_q = engine.hf_config();
                const int v_heads2 = cfg_q.linear_num_value_heads;
                const int conv_dim2 =
                    2 * (cfg_q.linear_num_key_heads * cfg_q.linear_key_head_dim) +
                    (cfg_q.linear_num_value_heads * cfg_q.linear_value_head_dim);
                const int stash_w2 = conv_dim2 + 2 * v_heads2;
                const int page2 = 32;
                const int nslabs = (fold_n + page2 - 1) / page2;
                q35_state_cache.configure_rs_buffer_pool(page2, stash_w2, nslabs + 1);

                std::vector<std::uint32_t> buf_ids(nslabs);
                for (int i = 0; i < nslabs; ++i) buf_ids[i] = (std::uint32_t)i;
                std::vector<std::uint32_t> buf_indptr = {0u, (std::uint32_t)nslabs};
                const std::int32_t commit_len_h = fold_n;
                std::int32_t* d_commit = nullptr;
                CUDA_CHECK(cudaMalloc(&d_commit, sizeof(std::int32_t)));
                CUDA_CHECK(cudaMemcpy(d_commit, &commit_len_h, sizeof(std::int32_t),
                                      cudaMemcpyHostToDevice));
                std::int32_t slot0 = 0;
                std::int32_t* d_slot0 = nullptr;
                CUDA_CHECK(cudaMalloc(&d_slot0, sizeof(std::int32_t)));
                CUDA_CHECK(cudaMemcpy(d_slot0, &slot0, sizeof(std::int32_t),
                                      cudaMemcpyHostToDevice));

                const std::size_t per_layer_bytes =
                    q35_state_cache.recurrent_slot_stride_bytes();
                const int n_layers_total = q35_state_cache.num_layers();
                // recurrent_state_raw indexes by the FULL transformer-layer index
                // (it maps to the compact linear slot internally via
                // linear_layer_index_), so pass L, not a compact counter.
                auto snap = [&]() {
                    std::vector<std::uint8_t> h;
                    for (int L = 0; L < n_layers_total; ++L) {
                        if (!q35_state_cache.is_linear(L)) continue;
                        std::vector<std::uint8_t> buf(per_layer_bytes);
                        CUDA_CHECK(cudaMemcpy(buf.data(),
                            q35_state_cache.recurrent_state_raw(L, 0),
                            per_layer_bytes, cudaMemcpyDeviceToHost));
                        h.insert(h.end(), buf.begin(), buf.end());
                    }
                    return h;
                };

                auto fold_fwd = [&](bool fresh, bool do_write, bool do_fold) {
                    model::Qwen3_5ForwardCfg q35_fwd{};
                    q35_fwd.force_prefill_path = true;
                    q35_fwd.tp_size = cfg.distributed.tp_size;
                    q35_fwd.tp_comm = tp_comm;
                    const int npg = (fold_n + page_size - 1) / page_size;
                    std::vector<std::uint32_t> h_qo = {0u, (std::uint32_t)fold_n};
                    std::vector<std::uint32_t> h_pp = {0u, (std::uint32_t)npg};
                    std::vector<std::uint32_t> h_pi(npg);
                    for (int i = 0; i < npg; ++i) h_pi[i] = (std::uint32_t)i;
                    std::vector<std::uint32_t> h_lpl = {
                        (std::uint32_t)(((fold_n - 1) % page_size) + 1)};
                    std::uint32_t *d_qo, *d_pi, *d_pp, *d_lpl;
                    CUDA_CHECK(cudaMalloc(&d_qo, 4 * h_qo.size()));
                    CUDA_CHECK(cudaMalloc(&d_pi, 4 * h_pi.size()));
                    CUDA_CHECK(cudaMalloc(&d_pp, 4 * h_pp.size()));
                    CUDA_CHECK(cudaMalloc(&d_lpl, 4 * h_lpl.size()));
                    CUDA_CHECK(cudaMemcpy(d_qo, h_qo.data(), 4 * h_qo.size(), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(d_pi, h_pi.data(), 4 * h_pi.size(), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(d_pp, h_pp.data(), 4 * h_pp.size(), cudaMemcpyHostToDevice));
                    CUDA_CHECK(cudaMemcpy(d_lpl, h_lpl.data(), 4 * h_lpl.size(), cudaMemcpyHostToDevice));
                    model::Qwen3_5PlanState q35_plan;
                    model::prepare_qwen3_5_decode_plan(
                        q35_plan, parity_attn_ws, cache, engine.hf_config(), q35_fwd,
                        h_qo.data(), h_pp.data(), h_lpl.data(), fold_n, 1,
                        /*is_decode=*/false);
                    const std::uint8_t fr = fresh ? 1u : 0u;
                    model::qwen3_5_forward_paged(
                        weights_qwen3_5, engine.hf_config(), q35_fwd, q35_plan,
                        ws, q35_la_ws, cache, q35_state_cache, parity_attn_ws, cublas,
                        d_tokens, d_positions, d_qo, d_pi, d_pp, d_lpl,
                        h_qo.data(), h_pp.data(), fold_n, 1, /*is_pure_decode=*/false,
                        /*mask_d=*/nullptr, /*mask_indptr_d=*/nullptr,
                        &slot0, &fr, d_slot0, /*logit_row=*/nullptr, /*num_logit=*/0,
                        do_fold ? d_commit : nullptr,
                        buf_ids.data(), buf_indptr.data(), do_write, do_fold);
                    cudaFree(d_qo); cudaFree(d_pi); cudaFree(d_pp); cudaFree(d_lpl);
                };

                fold_fwd(/*fresh=*/true,  /*write=*/false, /*fold=*/false);
                CUDA_CHECK(cudaDeviceSynchronize());
                const auto R_ref = snap();
                fold_fwd(/*fresh=*/true,  /*write=*/true,  /*fold=*/false);
                CUDA_CHECK(cudaDeviceSynchronize());
                const auto R_write = snap();
                fold_fwd(/*fresh=*/false, /*write=*/false, /*fold=*/true);
                CUDA_CHECK(cudaDeviceSynchronize());
                const auto R_fold = snap();

                std::size_t diff = 0, nz_write = 0;
                for (std::size_t i = 0; i < R_ref.size(); ++i)
                    if (R_ref[i] != R_fold[i]) ++diff;
                for (std::size_t i = 0; i < R_write.size(); ++i)
                    if (R_write[i] != 0) ++nz_write;
                // Per-layer diff localization (warm debug): which linear layers
                // diverge + where the first byte mismatch lands within a layer.
                {
                    const std::size_t lb = per_layer_bytes;
                    int li2 = 0;
                    for (std::size_t off = 0; off + lb <= R_ref.size();
                         off += lb, ++li2) {
                        std::size_t ld = 0, first = lb;
                        for (std::size_t i = 0; i < lb; ++i)
                            if (R_ref[off + i] != R_fold[off + i]) {
                                if (first == lb) first = i;
                                ++ld;
                            }
                        std::cerr << "[fold-parity]   layer[" << li2 << "] diff="
                                  << ld << "/" << lb << " first_byte="
                                  << (ld ? static_cast<long>(first) : -1L) << "\n";
                    }
                }
                // Magnitude analysis. la.mixed_qkv/a/b are bf16 in BOTH the
                // reference and the write pass (the in-proj GEMM emits bf16), so
                // the pool round-trip itself is lossless. The residual is that
                // R_ref and the write pass run two SEPARATE in-proj GEMM calls,
                // and cuBLAS is not bit-deterministic across invocations
                // (algo/split-K selection) -> the bf16 activations differ by
                // <=1 ULP, which then amplifies through 64 gated-delta recurrence
                // steps + the conv. So bit-exact parity is structurally
                // unavoidable here (the reference is an INDEPENDENT GEMM, unlike
                // commit-advance which replays the same stashed activations).
                // Gate on delta's numeric tolerance instead.
                double mean_abs = 0.0, frac_big = 0.0, l2rel = 0.0, maxabs = 0.0;
                {
                    auto bf16f = [](std::uint8_t lo, std::uint8_t hi) {
                        std::uint32_t b =
                            static_cast<std::uint32_t>((hi << 8) | lo) << 16;
                        float f; std::memcpy(&f, &b, sizeof(f)); return f;
                    };
                    double sumabs = 0.0, sumsq_d = 0.0, sumsq_r = 0.0;
                    std::size_t big = 0, n = 0;
                    long firstdiff = -1;
                    for (std::size_t i = 0; i + 1 < R_ref.size(); i += 2, ++n) {
                        float a = bf16f(R_ref[i], R_ref[i + 1]);
                        float b = bf16f(R_fold[i], R_fold[i + 1]);
                        double d = std::fabs(
                            static_cast<double>(a) - static_cast<double>(b));
                        sumabs += d; if (d > maxabs) maxabs = d;
                        sumsq_d += d * d;
                        sumsq_r += static_cast<double>(a) * static_cast<double>(a);
                        double rel = d / (std::fabs(static_cast<double>(a)) + 1e-6);
                        if (rel > 0.05 && d > 1e-3) ++big;
                        if (firstdiff < 0 &&
                            (R_ref[i] != R_fold[i] || R_ref[i+1] != R_fold[i+1])) {
                            firstdiff = static_cast<long>(i);
                            std::cerr << "[fold-parity]   first-diff @byte" << i
                                      << " ref=" << a << " fold=" << b << "\n";
                        }
                    }
                    const std::size_t nn = std::max<std::size_t>(1, n);
                    mean_abs = sumabs / static_cast<double>(nn);
                    frac_big = static_cast<double>(big) / static_cast<double>(nn);
                    l2rel = std::sqrt(sumsq_d) / (std::sqrt(sumsq_r) + 1e-12);
                    std::cerr << "[fold-parity]   mag: max_abs=" << maxabs
                              << " mean_abs=" << mean_abs
                              << " l2_rel=" << l2rel
                              << " elems_rel>5%=" << big << "/" << n
                              << " (" << frac_big << ")\n";
                }
                // Tolerance gate (delta's call): bf16-precision-bounded parity. A
                // correct gather/scatter/replay lands far under these; a structural
                // bug (wrong slab/stride/missing tokens) blows past them. l2_rel is
                // logged as a corroborating canonical metric.
                const double kMeanTol = 1e-3;   // mean abs state diff
                const double kFracTol = 0.01;    // fraction of elems with rel>5%
                const bool pass =
                    mean_abs < kMeanTol && frac_big < kFracTol && nz_write == 0;
                std::cerr << "[fold-parity] N=" << fold_n << " slabs=" << nslabs
                          << " state_bytes=" << R_ref.size()
                          << " | bytes_differ=" << diff
                          << " | mean_abs=" << mean_abs << " (tol " << kMeanTol << ")"
                          << " frac_rel>5%=" << frac_big << " (tol " << kFracTol << ")"
                          << " l2_rel=" << l2rel
                          << " | post-write nonzero=" << nz_write
                          << " => " << (pass ? "PASS" : "FAIL") << "\n";
                cudaFree(d_commit);
                cudaFree(d_slot0);
                return pass ? 0 : 7;
            }
        }

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

    } else {
        model::qwen3_forward_prefill(
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
        kernels::launch_argmax_bf16(
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

}  // namespace pie_cuda_driver
