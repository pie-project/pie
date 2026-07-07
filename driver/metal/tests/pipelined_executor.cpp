// pipelined_executor — Metal double-buffered (async) decode executor.
//
// The Metal half of In Gim's vLLM-style pipelining push: submit forward N+1's
// command buffers via `mx::async_eval` BEFORE block-reading forward N's result,
// so the GPU never idles between steps (the readback bubble of N overlaps N+1's
// GPU execution). This is the Metal dual of guru's CUDA stream-ahead enqueue;
// the submit/collect split is the executor pattern the WaitAllPolicy wave drives
// (the wave knows N+1 while N runs). See wiki ptir-mlx-async-scheduling-investigation.
//
// Scope: paged-KV / standard-attention (Qwen3-0.6B via llama_like) — dependency-
// clean (per-step KV write, no GDN recurrent state). VALUE-verified on the Metal
// GPU: the pipelined token sequence is bit-identical to the synchronous path
// (greedy) — the correctness gate. Single-stream latency is GPU-bound (the win
// is bounded by MLX's per-step stream-synchronizing item(); the full "device
// never idles" value is the continuous-batching / wave regime that hides the
// between-batch scheduler bubble — pairs with guru's CUDA async scheduling).
//
// Usage: pipelined_executor <hf_path> [prefill_len=8] [steps=64]

#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <iostream>
#include <string>
#include <vector>

#include <mlx/mlx.h>

#include "loader/model_loader.hpp"
#include "model/model_graph.hpp"
#include "kernels/sampling.hpp"

namespace mx = mlx::core;
using namespace pie_metal_driver;
using Clock = std::chrono::steady_clock;

namespace {

// Single-token decode batch built from a DEVICE token array (no host round-trip
// on the input) — the device-resident-token primitive that lets async_eval hide
// the per-step readback bubble.
model::ForwardBatch make_batch_dev(const mx::array& tok_dev, int pos, int page_size) {
    const int n_pages = pos / page_size + 1;
    std::vector<int> page_idx(n_pages);
    for (int i = 0; i < n_pages; ++i) page_idx[i] = i;
    const int last_page_len = pos % page_size + 1;
    return model::ForwardBatch{
        mx::astype(tok_dev, mx::int32),
        mx::array({pos}, {1}, mx::int32),
        mx::array({0}, {1}, mx::int32),
        mx::array(page_idx.data(), {n_pages}, mx::int32),
        mx::array({0, n_pages}, {2}, mx::int32),
        mx::array({last_page_len}, {1}, mx::int32),
        mx::array({0, 1}, {2}, mx::int32),
        mx::array({pos}, {1}, mx::int32),
        1, 1, 1, /*pure_decode=*/true,
    };
}

// ── The double-buffered executor: submit() enqueues a forward's command buffers
//    (non-blocking); collect() block-reads its sampled token. ────────────────
struct PipelinedExecutor {
    loader::LoadedModel& model;
    int ps;
    std::vector<sampling::SamplerParams> params;  // greedy (temp 0 -> argmax)
    std::uint64_t seed = 0;

    void attach(model::ForwardBatch& b) {
        if (model.lin_cache == nullptr) return;  // no GDN state for std-attn (Qwen3)
        b.lin_cache = model.lin_cache.get();
        b.slot_ids = mx::array({0}, {1}, mx::int32);
        b.qo_indptr_host.assign({0, b.n_total});
    }

    // Submit forward for `tok_dev` at `pos`: forward -> sample_token_device ->
    // async_eval (the N+1-ahead submit). Returns the lazy device token.
    mx::array submit(const mx::array& tok_dev, int pos) {
        auto b = make_batch_dev(tok_dev, pos, ps);
        attach(b);
        mx::array logits = model.graph->forward(b, *model.kv);  // [1, vocab] lazy
        mx::array next = sampling::sample_token_device(logits, params, seed);
        mx::async_eval(std::vector<mx::array>{next});  // non-blocking command-buffer submit
        return next;
    }
    // Block-read a submitted forward's sampled token (the sync point).
    int collect(const mx::array& tok_dev) {
        return static_cast<int>(mx::astype(tok_dev, mx::int32).item<std::int32_t>());
    }
};

// Prefill the prompt and return the first sampled device token + its position.
std::pair<mx::array, int> prefill(loader::LoadedModel& model, PipelinedExecutor& ex,
                                  const std::vector<int>& toks) {
    const int n = (int)toks.size(), ps = ex.ps;
    std::vector<int> pos(n), wi(n);
    for (int i = 0; i < n; ++i) { pos[i] = i; wi[i] = i; }
    const int last = n - 1, n_pages = last / ps + 1;
    std::vector<int> pgi(n_pages);
    for (int i = 0; i < n_pages; ++i) pgi[i] = i;
    model::ForwardBatch b{
        mx::array(toks.data(), {n}, mx::int32), mx::array(pos.data(), {n}, mx::int32),
        mx::array({last}, {1}, mx::int32),
        mx::array(pgi.data(), {n_pages}, mx::int32), mx::array({0, n_pages}, {2}, mx::int32),
        mx::array({last % ps + 1}, {1}, mx::int32), mx::array({0, n}, {2}, mx::int32),
        mx::array(wi.data(), {n}, mx::int32), n, 1, 1, false};
    ex.attach(b);
    mx::array logits = model.graph->forward(b, *model.kv);
    mx::array first = sampling::sample_token_device(logits, ex.params, ex.seed);
    mx::eval(first);
    model.kv->eval();
    return {first, n};  // first token sits at position n (after the prompt)
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: pipelined_executor <hf_path> [prefill=8] [steps=64]\n"; return 2; }
    const std::string hf_path = argv[1];
    const int plen  = argc > 2 ? std::atoi(argv[2]) : 8;
    const int steps = argc > 3 ? std::atoi(argv[3]) : 64;

    BatchingConfig batching;
    loader::LoadedModel model = loader::load_model(hf_path, batching);
    std::printf("[pipe] loaded arch=%s layers=%d page_size=%d, lin_cache=%s\n",
                model.caps.arch_name.c_str(), model.caps.num_hidden_layers,
                model.kv->page_size(), model.lin_cache ? "yes(GDN)" : "no(std-attn)");

    PipelinedExecutor ex{model, model.kv->page_size(),
                         {sampling::SamplerParams{sampling::SamplerType::Multinomial, 0.0f}}, 0};

    // Simulated per-step host-side work (scheduler decision + batch construction
    // + response marshaling) the wave does BETWEEN batches. In the synchronous
    // path it is paid serially (GPU idle); in the pipelined path it overlaps the
    // in-flight forward's GPU execution. Default 0 = pure GPU-bound decode.
    const int host_us = [] {
        const char* e = std::getenv("PIPE_HOST_US");
        return e ? std::atoi(e) : 0;
    }();
    auto host_work = [&]() {
        if (host_us <= 0) return;
        auto until = Clock::now() + std::chrono::microseconds(host_us);
        volatile std::uint64_t x = 0;
        while (Clock::now() < until) x++;  // busy-spin (CPU, like the scheduler)
    };

    std::vector<int> prompt;
    for (int i = 0; i < plen; ++i) prompt.push_back(785 + i * 13);  // fixed valid ids

    // ── run N greedy decode steps SYNCHRONOUSLY (collect N before submit N+1) ──
    auto run_sync = [&]() {
        auto [y0, p0] = prefill(model, ex, prompt);
        std::vector<int> seq;
        mx::array y = y0;
        int p = p0;
        auto t0 = Clock::now();
        for (int t = 0; t < steps; ++t) {
            mx::array s = ex.submit(y, p);
            int tok = ex.collect(s);   // block-read immediately — no overlap
            host_work();               // scheduler/batch-build — GPU idle here
            seq.push_back(tok);
            y = mx::array({tok}, {1}, mx::int32);
            ++p;
        }
        double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        return std::make_pair(seq, ms);
    };

    // ── run N greedy decode steps PIPELINED (submit N+1 before collect N) ──
    auto run_pipe = [&]() {
        auto [y0, p0] = prefill(model, ex, prompt);
        std::vector<int> seq;
        auto t0 = Clock::now();
        mx::array cur = ex.submit(y0, p0);   // submit step 0
        for (int t = 0; t < steps; ++t) {
            bool more = (t + 1 < steps);
            // Submit N+1 (async, non-blocking) BEFORE reading N — the double buffer.
            mx::array next = more ? ex.submit(cur, p0 + t + 1) : cur;
            host_work();                      // scheduler/batch-build — overlaps N+1's GPU exec
            int tok = ex.collect(cur);        // block-read N — overlaps N+1's GPU exec
            seq.push_back(tok);
            cur = next;
        }
        double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
        return std::make_pair(seq, ms);
    };

    run_sync();  // warm up (weights/kernels resident)
    auto [seq_sync, ms_sync] = run_sync();
    auto [seq_pipe, ms_pipe] = run_pipe();

    // ── VALUE-verify on the Metal GPU: token-identity + latency ──
    bool identical = (seq_sync == seq_pipe);
    std::printf("\n[pipe] %d greedy decode steps on the Metal GPU (Qwen3-0.6B std-attn, paged KV), host_us=%d/step:\n", steps, host_us);
    std::printf("  synchronous : %8.2f ms  (%.1f tok/s)\n", ms_sync, steps * 1000.0 / ms_sync);
    std::printf("  pipelined   : %8.2f ms  (%.1f tok/s)\n", ms_pipe, steps * 1000.0 / ms_pipe);
    std::printf("  latency ratio (sync/pipe): %.3fx\n", ms_sync / ms_pipe);
    std::printf("  token-identity (pipelined == synchronous): %s  <== correctness gate\n",
                identical ? "YES" : "NO");
    std::printf("  first 12 tokens:");
    for (int i = 0; i < 12 && i < (int)seq_sync.size(); ++i) std::printf(" %d", seq_sync[i]);
    std::printf("\n");
    if (!identical) {
        std::printf("  MISMATCH at:");
        for (int i = 0; i < (int)seq_sync.size(); ++i)
            if (i >= (int)seq_pipe.size() || seq_sync[i] != seq_pipe[i])
                { std::printf(" [%d: %d vs %d]", i, seq_sync[i],
                              i < (int)seq_pipe.size() ? seq_pipe[i] : -1); break; }
        std::printf("\n");
    }
    std::printf("%s\n", identical ? "PIPELINE_OK" : "PIPELINE_FAIL");
    return identical ? 0 : 1;
}
