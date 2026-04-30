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
#include <cuda_bf16.h>
#include <cuda_runtime.h>
#include <fstream>
#include <nlohmann/json.hpp>
#include <vector>

#include "attention_workspace.hpp"
#include "config.hpp"
#include "cuda_check.hpp"
#include "engine.hpp"
#include "kernels/argmax.hpp"
#include "kernels/sample_flashinfer.hpp"
#include "kernels/sample_temp.hpp"
#include "kv_cache.hpp"
#include "model/qwen3.hpp"
#include "model/qwen3_forward.hpp"
#include "ops/gemm.hpp"
#include "response_writer.hpp"
#include "shmem_ipc.hpp"
#include "shmem_schema.hpp"

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
    if (engine.hf_config().model_type != "qwen3" &&
        engine.hf_config().model_type != "qwen3_5") {
        std::cerr << "[parity] unsupported model_type: "
                  << engine.hf_config().model_type << "\n";
        return 2;
    }
    const auto weights = pie_cuda_driver::model::bind_qwen3(engine);

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
            /*max_kv_len=*/N,
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

    if (engine.hf_config().model_type != "qwen3" &&
        engine.hf_config().model_type != "qwen3_5") {
        std::cerr << "[pie-driver-cuda] arch '"
                  << engine.hf_config().model_type
                  << "' not yet supported (only qwen3 / qwen3_5 in M1.4)\n";
        return 2;
    }
    const auto weights = pie_cuda_driver::model::bind_qwen3(engine);
    std::cerr << "[pie-driver-cuda] qwen3 schema bound: "
              << weights.layers.size() << " layers\n";

    // Pre-allocate persistent state for serving.
    //
    // - Workspace sized for `max_batch_tokens` from config, capped at 8192
    //   to keep the [N, vocab] logits + probs buffers under ~5 GiB combined
    //   (vocab=151936). M2.x will compute logits only at sampling rows so
    //   this can grow.
    // - KV cache sized at `max_num_kv_pages` × `kv_page_size`.
    const int max_workspace_tokens = std::min<int>(cfg.batching.max_batch_tokens, 8192);
    auto ws = pie_cuda_driver::model::Qwen3Workspace::allocate(
        engine.hf_config(), max_workspace_tokens);

    auto kv_cache = pie_cuda_driver::KvCache::allocate(
        engine.hf_config().num_hidden_layers,
        static_cast<int>(cfg.batching.max_num_kv_pages),
        static_cast<int>(cfg.batching.kv_page_size),
        engine.hf_config().num_key_value_heads,
        engine.hf_config().head_dim);

    auto attn_ws = pie_cuda_driver::AttentionWorkspace::allocate();

    pie_cuda_driver::ops::CublasHandle cublas;
    std::cerr << "[pie-driver-cuda] kv_cache: "
              << kv_cache.num_pages() << " pages × "
              << kv_cache.page_size() << " tokens; "
              << "workspace tokens=" << max_workspace_tokens << "\n";

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

    auto upload_u32 = [](std::span<const std::uint8_t> view) -> std::uint32_t* {
        std::uint32_t* dev = nullptr;
        if (view.empty()) return dev;
        CUDA_CHECK(cudaMalloc(&dev, view.size()));
        CUDA_CHECK(cudaMemcpy(dev, view.data(), view.size(), cudaMemcpyHostToDevice));
        return dev;
    };

    server.serve_forever([&](const pie_cuda_driver::SlotRequest& req,
                             std::span<std::uint8_t> response) -> std::size_t {
        ++handled;
        if (req.method_tag != pie_cuda_driver::METHOD_TAG_FIRE_BATCH) {
            std::cerr << "[pie-driver-cuda] unsupported method_tag="
                      << req.method_tag << " req_id=" << req.req_id << "\n";
            return 0;
        }

        std::uint32_t *d_tokens = nullptr, *d_positions = nullptr;
        std::uint32_t *d_qo = nullptr, *d_kvpi = nullptr,
                      *d_kvpp = nullptr, *d_kvlpl = nullptr;
        std::int32_t* d_sampled = nullptr;

        try {
            namespace S = pie_cuda_driver::schema;
            const auto dec = S::decode_request(req.payload);

            const auto tok_view  = dec.as<std::uint32_t>(S::A_TOKEN_IDS);
            const auto pos_view  = dec.as<std::uint32_t>(S::A_POSITION_IDS);
            const auto qo_view   = dec.as<std::uint32_t>(S::A_QO_INDPTR);
            const auto kvpi_view = dec.as<std::uint32_t>(S::A_KV_PAGE_INDICES);
            const auto kvpp_view = dec.as<std::uint32_t>(S::A_KV_PAGE_INDPTR);
            const auto kvlpl_view = dec.as<std::uint32_t>(S::A_KV_LAST_PAGE_LENS);
            const auto sidx_view = dec.as<std::uint32_t>(S::A_SAMPLING_INDICES);
            const auto sptr_view = dec.as<std::uint32_t>(S::A_SAMPLING_INDPTR);

            const int N = static_cast<int>(tok_view.size());
            const int R = static_cast<int>(qo_view.size()) - 1;
            const int num_sampling = static_cast<int>(sidx_view.size());

            if (N == 0 || R <= 0) {
                // Empty batch — emit a zero-token flat response.
                std::vector<std::uint32_t> counts(std::max(R, 0), 0u);
                return pie_cuda_driver::response::write_flat_response(
                    response, counts, {});
            }
            if (N > max_workspace_tokens) {
                std::cerr << "[pie-driver-cuda] batch tokens=" << N
                          << " exceeds workspace=" << max_workspace_tokens << "\n";
                return 0;
            }

            // Compute max KV length across requests for shmem sizing.
            // Also detect "pure decode" (every request has qo_len == 1) so
            // we can dispatch flashinfer's decode kernel on the hot path.
            const int page_size = kv_cache.page_size();
            int max_kv_len = 0;
            const std::uint32_t* h_kvpp  = kvpp_view.data();
            const std::uint32_t* h_kvlpl = kvlpl_view.data();
            const std::uint32_t* h_qo    = qo_view.data();
            bool is_pure_decode = (R > 0);
            for (int r = 0; r < R; ++r) {
                const int num_pages_r = static_cast<int>(h_kvpp[r + 1] - h_kvpp[r]);
                if (num_pages_r <= 0) continue;
                const int kv_len_r = (num_pages_r - 1) * page_size +
                                     static_cast<int>(h_kvlpl[r]);
                if (kv_len_r > max_kv_len) max_kv_len = kv_len_r;
                if (h_qo[r + 1] - h_qo[r] != 1u) is_pure_decode = false;
            }

            // Upload BPIQ control arrays.
            d_tokens   = upload_u32({reinterpret_cast<const std::uint8_t*>(tok_view.data()),
                                     tok_view.size() * 4});
            d_positions = upload_u32({reinterpret_cast<const std::uint8_t*>(pos_view.data()),
                                      pos_view.size() * 4});
            d_qo       = upload_u32({reinterpret_cast<const std::uint8_t*>(qo_view.data()),
                                     qo_view.size() * 4});
            d_kvpi     = upload_u32({reinterpret_cast<const std::uint8_t*>(kvpi_view.data()),
                                     kvpi_view.size() * 4});
            d_kvpp     = upload_u32({reinterpret_cast<const std::uint8_t*>(kvpp_view.data()),
                                     kvpp_view.size() * 4});
            d_kvlpl    = upload_u32({reinterpret_cast<const std::uint8_t*>(kvlpl_view.data()),
                                     kvlpl_view.size() * 4});

            // Run the paged forward pass. (Token IDs are u32 on the wire but
            // bitwise-identical to i32 for any vocab < 2^31.)
            pie_cuda_driver::model::qwen3_forward_paged(
                weights, engine.hf_config(), ws, kv_cache, attn_ws, cublas,
                reinterpret_cast<const std::int32_t*>(d_tokens),
                reinterpret_cast<const std::int32_t*>(d_positions),
                d_qo, d_kvpi, d_kvpp, d_kvlpl,
                /*qo_indptr_h=*/h_qo,
                /*kv_page_indptr_h=*/h_kvpp,
                N, R, std::max(max_kv_len, 1),
                is_pure_decode);

            // Build per-row temperature / seed arrays. For non-sampling rows
            // we leave T=0 so the kernel collapses to argmax (cheap, output
            // ignored). For sampling rows we look up each request's first
            // sampler and apply it across that request's sample positions —
            // a simplification of the full per-position sampler mapping that
            // M2.5 will implement. Top-k / top-p / min-p still pending.
            const auto temp_view  = dec.as<float>(S::A_SAMPLER_TEMPERATURES);
            const auto top_k_view = dec.as<std::uint32_t>(S::A_SAMPLER_TOP_K);
            const auto top_p_view = dec.as<float>(S::A_SAMPLER_TOP_P);
            const auto min_p_view = dec.as<float>(S::A_SAMPLER_MIN_P);
            const auto seed_view  = dec.as<std::uint32_t>(S::A_SAMPLER_SEEDS);
            const auto rns_view   = dec.as<std::uint32_t>(S::A_REQUEST_NUM_SAMPLERS);

            std::vector<float> h_per_temp(N, 0.f);
            std::vector<float> h_per_min_p(N, 0.f);
            std::vector<float> h_per_top_p(N, 1.f);
            std::vector<std::int32_t> h_per_top_k(N, 0);
            std::vector<std::uint32_t> h_per_seed(N, 0u);

            const std::uint32_t* h_sptr  = sptr_view.data();
            const std::uint32_t* h_sidx  = sidx_view.data();
            const std::uint32_t* h_rns   = rns_view.data();
            const float*         h_temp  = temp_view.data();
            const std::uint32_t* h_top_k = top_k_view.data();
            const float*         h_top_p = top_p_view.data();
            const float*         h_min_p = min_p_view.data();
            const std::uint32_t* h_seed  = seed_view.data();

            bool any_topk_topp = false;
            std::uint32_t sampler_off = 0;
            for (int r = 0; r < R; ++r) {
                const std::uint32_t nsamplers_r =
                    (rns_view.size() > static_cast<std::size_t>(r)) ? h_rns[r] : 0u;
                if (nsamplers_r > 0 && temp_view.size() > sampler_off) {
                    const float T   = h_temp[sampler_off];
                    const float Tp  = (top_p_view.size() > sampler_off) ? h_top_p[sampler_off] : 1.f;
                    const float Mp  = (min_p_view.size() > sampler_off) ? h_min_p[sampler_off] : 0.f;
                    // BPIQ uses 0 to mean "no top-k filter"; flashinfer
                    // interprets 0 as "keep zero tokens" (always returns 0).
                    // Map to vocab so the filter is a no-op.
                    const std::int32_t Tk_raw =
                        (top_k_view.size() > sampler_off)
                            ? static_cast<std::int32_t>(h_top_k[sampler_off]) : 0;
                    const std::int32_t Tk =
                        (Tk_raw == 0) ? engine.hf_config().vocab_size : Tk_raw;
                    const std::uint32_t s = h_seed[sampler_off];

                    if ((Tk_raw > 0 || Tp < 1.f) && T > 0.f) any_topk_topp = true;

                    // sampling_indices is per-request relative — global row =
                    // qo_indptr[r] + sampling_indices[k]. (Matches the
                    // pie_driver Python path; see batching.py.)
                    const std::uint32_t qo_lo = h_qo[r];
                    for (std::uint32_t k = h_sptr[r]; k < h_sptr[r + 1]; ++k) {
                        const std::uint32_t row = qo_lo + h_sidx[k];
                        if (row < static_cast<std::uint32_t>(N)) {
                            h_per_temp[row]  = T;
                            h_per_top_k[row] = Tk;
                            h_per_top_p[row] = Tp;
                            h_per_min_p[row] = Mp;
                            h_per_seed[row]  = s;
                        }
                    }
                }
                sampler_off += nsamplers_r;
            }

            CUDA_CHECK(cudaMalloc(&d_sampled, sizeof(std::int32_t) * N));

            if (any_topk_topp) {
                std::vector<std::uint64_t> h_per_seed64(N);
                for (int i = 0; i < N; ++i) {
                    h_per_seed64[i] = static_cast<std::uint64_t>(h_per_seed[i]);
                }
                std::vector<std::int32_t> h_sample_idx(num_sampling, 0);
                {
                    int k_g = 0;
                    for (int r = 0; r < R; ++r) {
                        const std::uint32_t qo_lo = h_qo[r];
                        for (std::uint32_t k = h_sptr[r]; k < h_sptr[r + 1]; ++k, ++k_g) {
                            h_sample_idx[k_g] =
                                static_cast<std::int32_t>(qo_lo + h_sidx[k]);
                        }
                    }
                }

                float*         d_temp_f = nullptr;
                float*         d_top_p_f = nullptr;
                std::int32_t*  d_top_k = nullptr;
                std::uint64_t* d_seed64 = nullptr;
                std::int32_t*  d_sample_idx = nullptr;
                std::int32_t*  d_per_sample_token = nullptr;
                bool*          d_valid = nullptr;
                CUDA_CHECK(cudaMalloc(&d_temp_f,           sizeof(float)         * N));
                CUDA_CHECK(cudaMalloc(&d_top_p_f,          sizeof(float)         * N));
                CUDA_CHECK(cudaMalloc(&d_top_k,            sizeof(std::int32_t)  * N));
                CUDA_CHECK(cudaMalloc(&d_seed64,           sizeof(std::uint64_t) * N));
                CUDA_CHECK(cudaMalloc(&d_sample_idx,       sizeof(std::int32_t)  * num_sampling));
                CUDA_CHECK(cudaMalloc(&d_per_sample_token, sizeof(std::int32_t)  * num_sampling));
                CUDA_CHECK(cudaMalloc(&d_valid,            sizeof(bool)          * num_sampling));
                CUDA_CHECK(cudaMemcpy(d_temp_f,     h_per_temp.data(),   sizeof(float)         * N,           cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_top_p_f,    h_per_top_p.data(),  sizeof(float)         * N,           cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_top_k,      h_per_top_k.data(),  sizeof(std::int32_t)  * N,           cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_seed64,     h_per_seed64.data(), sizeof(std::uint64_t) * N,           cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_sample_idx, h_sample_idx.data(), sizeof(std::int32_t)  * num_sampling, cudaMemcpyHostToDevice));

                pie_cuda_driver::kernels::launch_sample_topk_topp_bf16(
                    ws.logits.data(), ws.probs.data(),
                    d_temp_f, d_sample_idx, d_top_k, d_top_p_f, d_seed64,
                    d_valid, d_per_sample_token,
                    N, num_sampling, engine.hf_config().vocab_size,
                    /*prng_offset=*/static_cast<std::uint64_t>(handled),
                    /*stream=*/nullptr);

                std::vector<std::int32_t> h_per_sample_token(num_sampling);
                CUDA_CHECK(cudaMemcpy(h_per_sample_token.data(), d_per_sample_token,
                                      sizeof(std::int32_t) * num_sampling,
                                      cudaMemcpyDeviceToHost));
                std::vector<std::int32_t> h_all_sampled(N, 0);
                for (int k = 0; k < num_sampling; ++k) {
                    h_all_sampled[h_sample_idx[k]] = h_per_sample_token[k];
                }
                CUDA_CHECK(cudaMemcpy(d_sampled, h_all_sampled.data(),
                                      sizeof(std::int32_t) * N,
                                      cudaMemcpyHostToDevice));

                cudaFree(d_temp_f); cudaFree(d_top_p_f);
                cudaFree(d_top_k); cudaFree(d_seed64);
                cudaFree(d_sample_idx); cudaFree(d_per_sample_token);
                cudaFree(d_valid);
            } else {
                // No top-k/p configured → existing temperature + min-p path.
                float* d_temp = nullptr;
                float* d_min_p = nullptr;
                std::uint32_t* d_seed = nullptr;
                CUDA_CHECK(cudaMalloc(&d_temp,  sizeof(float) * N));
                CUDA_CHECK(cudaMalloc(&d_min_p, sizeof(float) * N));
                CUDA_CHECK(cudaMalloc(&d_seed,  sizeof(std::uint32_t) * N));
                CUDA_CHECK(cudaMemcpy(d_temp,  h_per_temp.data(),
                                      sizeof(float) * N, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_min_p, h_per_min_p.data(),
                                      sizeof(float) * N, cudaMemcpyHostToDevice));
                CUDA_CHECK(cudaMemcpy(d_seed,  h_per_seed.data(),
                                      sizeof(std::uint32_t) * N, cudaMemcpyHostToDevice));

                pie_cuda_driver::kernels::launch_sample_temp_bf16(
                    ws.logits.data(), d_temp, d_min_p, d_seed, d_sampled,
                    N, engine.hf_config().vocab_size, /*stream=*/nullptr);
                cudaFree(d_temp); cudaFree(d_min_p); cudaFree(d_seed);
            }

            std::vector<std::int32_t> all_sampled(N);
            CUDA_CHECK(cudaMemcpy(all_sampled.data(), d_sampled,
                                  sizeof(std::int32_t) * N, cudaMemcpyDeviceToHost));

            std::vector<std::uint32_t> per_request_counts(R);
            std::vector<std::uint32_t> sampled_tokens;
            sampled_tokens.reserve(num_sampling);
            for (int r = 0; r < R; ++r) {
                const std::uint32_t lo = h_sptr[r];
                const std::uint32_t hi = h_sptr[r + 1];
                const std::uint32_t qo_lo = h_qo[r];
                per_request_counts[r] = hi - lo;
                for (std::uint32_t k = lo; k < hi; ++k) {
                    const std::uint32_t row = qo_lo + h_sidx[k];
                    sampled_tokens.push_back(
                        static_cast<std::uint32_t>(all_sampled[row]));
                }
            }

            const std::size_t resp_bytes =
                pie_cuda_driver::response::write_flat_response(
                    response, per_request_counts, sampled_tokens);

            cudaFree(d_tokens);
            cudaFree(d_positions);
            cudaFree(d_qo); cudaFree(d_kvpi); cudaFree(d_kvpp); cudaFree(d_kvlpl);
            cudaFree(d_sampled);

            if (handled <= 4 || handled % 100 == 0) {
                std::cerr << "[pie-driver-cuda] req_id=" << req.req_id
                          << " R=" << R << " N=" << N
                          << " sampled=" << num_sampling
                          << " max_kv=" << max_kv_len
                          << " resp=" << resp_bytes << "B\n";
            }
            return resp_bytes;

        } catch (const std::exception& e) {
            std::cerr << "[pie-driver-cuda] fire_batch failed for req_id="
                      << req.req_id << ": " << e.what() << "\n";
            cudaFree(d_tokens); cudaFree(d_positions);
            cudaFree(d_qo); cudaFree(d_kvpi); cudaFree(d_kvpp); cudaFree(d_kvlpl);
            cudaFree(d_sampled);
            return 0;
        }
    });

    g_server.store(nullptr);
    std::cerr << "[pie-driver-cuda] shutting down (handled " << handled
              << " requests)\n";
    return 0;
}
