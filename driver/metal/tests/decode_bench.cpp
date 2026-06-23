// In-process single-stream decode benchmark — the driver-vs-runtime isolator.
//
// Drives the real load -> graph -> paged-KV -> forward + greedy-sample loop
// DIRECTLY (no server / IPC / scheduler), so its tok/s is the third
// measurement point between:
//   * raw `mlx_lm.generate` (pure MLX, zero pie)         — upper bound
//   * THIS harness (driver graph + sampler, no IPC)       — driver dispatch
//   * pie-worker (full runtime)                           — IPC + scheduler
//
// The (raw - here) gap is the driver's per-token graph-encode/dispatch share;
// the (here - pie-worker) gap is the pie-core IPC/server/scheduler share. On
// Qwen2.5-0.5B (tiny, dispatch-bound) the two are ~equal; on BW/compute-bound
// models (gemma4) this harness ~= pie-worker, i.e. the runtime cost vanishes.
//
// Per step it mirrors exactly what executor.run_forward + the greedy sampler
// fast-path do: graph.forward(decode_batch) -> argmax(axis=1) -> eval -> read.
//
// Usage: decode_bench <hf_path> [prefill_len=16] [decode_steps=256]

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

#include <mlx/mlx.h>
#include <mlx/compile.h>

#include "loader/model_loader.hpp"
#include "model/model_graph.hpp"
#include "executor/executor.hpp"

#include <pie_schema/response_builder.hpp>
#include <pie_schema/view.hpp>

namespace mx = mlx::core;
using namespace pie_metal_driver;

namespace {

// Build a single-request ForwardBatch for `toks` at absolute `pos`, writing to
// contiguous physical KV pages (page i == logical page i for one request).
model::ForwardBatch make_batch(const std::vector<int>& toks,
                               const std::vector<int>& pos, int logit_row,
                               int page_size, bool pure_decode) {
    const int n = static_cast<int>(toks.size());
    const int last_pos = pos.back();
    const int n_pages = last_pos / page_size + 1;
    std::vector<int> page_idx(n_pages);
    for (int i = 0; i < n_pages; ++i) page_idx[i] = i;
    std::vector<int> write_idx(n);
    for (int i = 0; i < n; ++i) write_idx[i] = pos[i];  // contiguous phys slots
    const int last_page_len = last_pos % page_size + 1;

    return model::ForwardBatch{
        mx::array(toks.data(), {n}, mx::int32),
        mx::array(pos.data(), {n}, mx::int32),
        mx::array({logit_row}, {1}, mx::int32),
        mx::array(page_idx.data(), {n_pages}, mx::int32),
        mx::array({0, n_pages}, {2}, mx::int32),
        mx::array({last_page_len}, {1}, mx::int32),
        mx::array({0, n}, {2}, mx::int32),
        mx::array(write_idx.data(), {n}, mx::int32),
        n, 1, 1, pure_decode,
    };
}

// Single-token pure_decode batch whose input token is a DEVICE array (fed back
// from the previous step's argmax without a host readback) — for the pipelined
// path. Position + page metadata are host-deterministic.
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

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: decode_bench <hf_path> [prefill=16] [steps=256]\n";
        return 2;
    }
    const std::string hf_path = argv[1];
    const int prefill = argc > 2 ? std::stoi(argv[2]) : 16;
    const int steps   = argc > 3 ? std::stoi(argv[3]) : 256;

    BatchingConfig batching;
    if (const char* tp = std::getenv("PIE_TOTAL_PAGES")) {
        batching.total_pages = static_cast<std::uint32_t>(std::atoi(tp));
    }
    loader::LoadedModel model = [&] {
        try {
            return loader::load_model(hf_path, batching);
        } catch (const std::exception& e) {
            std::cerr << "[bench] SKIP: load_model failed for this checkpoint ("
                      << e.what() << ")\n";
            std::exit(3);
        }
    }();
    const int ps = model.kv->page_size();
    std::cerr << "[bench] loaded arch=" << model.caps.arch_name
              << " layers=" << model.caps.num_hidden_layers
              << " page_size=" << ps << "\n";

    // Hybrid (qwen3.6 gated-delta-net) staging: the executor normally supplies
    // the linear-attn state seam (lin_cache + per-request slot_ids + host CSR).
    // Mirror it here so the bare graph can run the recurrent layers. slot 0 for
    // the single request; qo_indptr_host matches the batch's token span.
    const bool hybrid = (model.lin_cache != nullptr);
    auto attach_hybrid = [&](model::ForwardBatch& b) {
        if (!hybrid) return;
        b.lin_cache = model.lin_cache.get();
        b.slot_ids  = mx::array({0}, {1}, mx::int32);
        b.qo_indptr_host.assign({0, b.n_total});
    };
    if (hybrid)
        std::cerr << "[bench] hybrid linear-attn cache staged (slot 0)\n";

    // Prefill to populate the KV cache.
    std::vector<int> ptoks(prefill), ppos(prefill);
    for (int i = 0; i < prefill; ++i) { ptoks[i] = 1 + (i % 100); ppos[i] = i; }
    try {
        auto b = make_batch(ptoks, ppos, prefill - 1, ps, /*pure_decode=*/false);
        attach_hybrid(b);
        mx::array lg = model.graph->forward(b, *model.kv);
        mx::eval(lg);
        model.kv->eval();
    } catch (const std::exception& e) {
        // Hybrid (recurrent) archs (e.g. qwen3.6 gated-delta-net) require the
        // executor to stage lin_cache/slot_ids; the bare graph path can't run
        // them. Report clearly rather than crash.
        std::cerr << "[bench] SKIP: graph.forward needs executor state staging "
                     "for this arch (" << e.what() << ")\n";
        return 3;
    }

    std::vector<std::uint32_t>* dump = nullptr;
    std::vector<std::uint32_t> seq;
    auto run_decode = [&](int n_steps) -> double {
        int tok = 5;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int t = 0; t < n_steps; ++t) {
            auto b = make_batch({tok}, {prefill + t}, 0, ps, /*pure_decode=*/true);
            attach_hybrid(b);
            mx::array logits = model.graph->forward(b, *model.kv);  // [1, vocab]
            mx::array next = mx::astype(mx::argmax(logits, 1), mx::uint32);
            next.eval();  // greedy-sampler autoregressive barrier (1 eval/token)
            tok = static_cast<int>(next.item<std::uint32_t>());
            if (dump) dump->push_back(static_cast<std::uint32_t>(tok));
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        return n_steps / std::chrono::duration<double>(t1 - t0).count();
    };

    // ── PROFILE the per-token eval-sync bubble (PIE_PROFILE_DECODE=1) ──
    // Decompose the serial per-token cost into:
    //   build = host-CPU graph construction (make_batch + graph->forward trace +
    //           argmax) — lazy, NO GPU work; this is the portion async_eval
    //           overlaps with the prior token's GPU compute.
    //   eval  = next.eval() = MLX stream-thread command-buffer encode + commit +
    //           GPU execute + waitUntilCompleted (CPU-encode + GPU + wait mixed).
    //   item  = next.item() readback of the 4-byte token (already evaled).
    // Plus a pure-GPU probe: re-eval an already-resident token array (no graph
    // build, no encode) to bound the bare commit+wait latency floor.
    if (std::getenv("PIE_PROFILE_DECODE")) {
        const int warm = 16;
        const int N = steps;
        std::vector<double> tb, te, ti;
        tb.reserve(N); te.reserve(N); ti.reserve(N);
        int tok = 5;
        using clk = std::chrono::high_resolution_clock;
        for (int t = 0; t < N; ++t) {
            auto a0 = clk::now();
            auto b = make_batch({tok}, {prefill + t}, 0, ps, /*pure_decode=*/true);
            attach_hybrid(b);
            mx::array logits = model.graph->forward(b, *model.kv);
            mx::array next = mx::astype(mx::argmax(logits, 1), mx::uint32);
            auto a1 = clk::now();
            next.eval();
            auto a2 = clk::now();
            tok = static_cast<int>(next.item<std::uint32_t>());
            auto a3 = clk::now();
            tb.push_back(std::chrono::duration<double, std::milli>(a1 - a0).count());
            te.push_back(std::chrono::duration<double, std::milli>(a2 - a1).count());
            ti.push_back(std::chrono::duration<double, std::milli>(a3 - a2).count());
        }
        // Bare commit+wait floor: eval a tiny already-resident scalar repeatedly.
        // Measures the Metal command-buffer commit + waitUntilCompleted latency
        // with ~1 trivial op (no transformer encode) — the irreducible per-eval
        // submission cost.
        std::vector<double> tcw;
        {
            mx::array acc = mx::array({0}, {1}, mx::int32);
            mx::eval(acc);
            for (int t = 0; t < 200; ++t) {
                auto c0 = clk::now();
                mx::array x = acc + mx::array({1}, {1}, mx::int32);
                x.eval();
                auto c1 = clk::now();
                tcw.push_back(std::chrono::duration<double, std::milli>(c1 - c0).count());
            }
        }
        auto stats = [&](std::vector<double> v, int skip) {
            std::vector<double> w(v.begin() + std::min<int>(skip, (int)v.size()), v.end());
            std::sort(w.begin(), w.end());
            double sum = 0; for (double x : w) sum += x;
            double mean = w.empty() ? 0 : sum / w.size();
            double med = w.empty() ? 0 : w[w.size() / 2];
            double mn = w.empty() ? 0 : w.front();
            return std::make_tuple(mean, med, mn);
        };
        auto [bm, bd, bn] = stats(tb, warm);
        auto [em, ed, en] = stats(te, warm);
        auto [im, id, in_] = stats(ti, warm);
        auto [cm, cd, cn] = stats(tcw, 16);
        std::fprintf(stderr,
            "\n[PROFILE] per-token decomposition (ms, skip %d warmup, n=%d)\n"
            "  build (host graph trace, lazy/no-GPU) : mean %.3f  median %.3f  min %.3f\n"
            "  eval  (encode+commit+GPU+wait)        : mean %.3f  median %.3f  min %.3f\n"
            "  item  (4B token readback)             : mean %.3f  median %.3f  min %.3f\n"
            "  TOTAL per token                       : mean %.3f  median %.3f\n"
            "  --- bare commit+wait floor (1 trivial op/eval): mean %.4f median %.4f min %.4f ms\n",
            warm, N,
            bm, bd, bn, em, ed, en, im, id, in_,
            bm + em + im, bd + ed + id, cm, cd, cn);
        std::fflush(stderr);
        return 0;
    }

    // ── B2 prototype: whole-step compiled decode (PIE_COMPILE_DECODE=1) ──
    // Wraps the ENTIRE per-token forward (embedding -> layers -> lm_head) in one
    // mx::compile trace so the per-token CPU encode is cached & replayed instead
    // of rebuilt. The paged-KV buffers are threaded as functional inputs/outputs
    // (captured arrays would be baked as constants and never update). Requires
    // PIE_DEVICE_DECODE=1 so the attention op is host-readback-free; otherwise
    // paged_attention's to_host_i32 breaks the trace. Nesting the existing
    // ops::compiled FFN/PLE regions inside this outer compile tests charlie's
    // open question #5 (does mx::compile compose nested compiles?).
    const bool use_compile = [] {
        const char* e = std::getenv("PIE_COMPILE_DECODE");
        return e && e[0] && e[0] != '0';
    }();

    PagedKvCache* kvp = model.kv.get();
    model::ModelGraph* g = model.graph.get();
    auto step_fn = [g, kvp](const std::vector<mx::array>& in) -> std::vector<mx::array> {
        // in: [token, pos, page_idx, last_page_len, write_idx, k0,v0,k1,v1,...]
        std::vector<mx::array> kvbuf(in.begin() + 5, in.end());
        kvp->restore(kvbuf);
        const int n_pages = in[2].shape(0);
        model::ForwardBatch b{
            in[0], in[1], mx::array({0}, {1}, mx::int32),
            in[2], mx::array({0, n_pages}, {2}, mx::int32), in[3],
            mx::array({0, 1}, {2}, mx::int32), in[4],
            1, 1, 1, /*pure_decode=*/true,
        };
        mx::array logits = g->forward(b, *kvp);  // [1, vocab]
        std::vector<mx::array> out;
        out.push_back(logits);
        for (auto& t : kvp->snapshot()) out.push_back(t);
        return out;
    };
    auto compiled_step = mx::compile(step_fn, /*shapeless=*/false);

    auto run_decode_compiled = [&](int n_steps) -> double {
        int tok = 5;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int t = 0; t < n_steps; ++t) {
            const int pos = prefill + t;
            const int n_pages = pos / ps + 1;
            const int lpl = pos % ps + 1;
            std::vector<int> page_idx(n_pages);
            for (int i = 0; i < n_pages; ++i) page_idx[i] = i;
            std::vector<mx::array> in;
            in.push_back(mx::array({tok}, {1}, mx::int32));
            in.push_back(mx::array({pos}, {1}, mx::int32));
            in.push_back(mx::array(page_idx.data(), {n_pages}, mx::int32));
            in.push_back(mx::array({lpl}, {1}, mx::int32));
            in.push_back(mx::array({pos}, {1}, mx::int32));  // write_idx = pos
            for (auto& kb : kvp->snapshot()) in.push_back(kb);
            std::vector<mx::array> outs = compiled_step(in);
            std::vector<mx::array> newkv(outs.begin() + 1, outs.end());
            kvp->restore(newkv);
            mx::array next = mx::astype(mx::argmax(outs[0], 1), mx::uint32);
            std::vector<mx::array> bar = {next};
            for (auto& t2 : newkv) bar.push_back(t2);
            mx::eval(bar);  // one barrier: next token + updated KV
            tok = static_cast<int>(next.item<std::uint32_t>());
            if (dump) dump->push_back(static_cast<std::uint32_t>(tok));
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        return n_steps / std::chrono::duration<double>(t1 - t0).count();
    };
    std::function<double(int)> decode = [&](int n) {
        return use_compile ? run_decode_compiled(n) : run_decode(n);
    };

    // ── Pipelined decode (PIE_PIPELINE_DECODE=1): the actual ceiling lever ──
    // Mirrors mlx_lm's generate_step: the sampled token is kept as a DEVICE
    // array fed straight back as the next input (positions/page metadata are
    // host-deterministic, no readback), and each step's forward is submitted
    // with async_eval while we block-read the PREVIOUS token. This overlaps
    // token N+1's compute (incl. CPU encode) with token N's argmax->item sync,
    // hiding the per-token barrier that mx::compile alone can't recover.
    const bool use_pipeline = [] {
        const char* e = std::getenv("PIE_PIPELINE_DECODE");
        return e && e[0] && e[0] != '0';
    }();
    auto run_decode_pipelined = [&](int n_steps) -> double {
        mx::array y = mx::array({5}, {1}, mx::int32);  // device input token
        mx::array prev = y;
        auto submit = [&](const mx::array& tok_dev, int pos) -> mx::array {
            auto b = make_batch_dev(tok_dev, pos, ps);
            attach_hybrid(b);
            mx::array logits = model.graph->forward(b, *model.kv);  // lazy
            mx::array next = mx::astype(mx::argmax(logits, 1), mx::int32);
            // Only async_eval `next`: its dependency forces this token's KV
            // scatter (touched pages) while preserving in-place donation —
            // snapshotting the full paged buffers would add references and
            // defeat donation (the regression seen with the compiled path).
            mx::async_eval(std::vector<mx::array>{next});  // non-blocking submit
            return next;
        };
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int t = 0; t < n_steps; ++t) {
            mx::array next = submit(y, prefill + t);
            if (t > 0) {  // block-read the PREVIOUS step (overlaps with this GPU work)
                int tok = static_cast<int>(prev.item<std::int32_t>());
                if (dump) dump->push_back(static_cast<std::uint32_t>(tok));
            }
            prev = next;
            y = next;
        }
        int last = static_cast<int>(prev.item<std::int32_t>());  // drain
        if (dump) dump->push_back(static_cast<std::uint32_t>(last));
        auto t1 = std::chrono::high_resolution_clock::now();
        return n_steps / std::chrono::duration<double>(t1 - t0).count();
    };
    if (use_pipeline) decode = [&](int n) { return run_decode_pipelined(n); };

    // ── DECODE_N validation (PIE_DECODE_N=N): exercise the PRODUCTIONIZED
    //    executor multi-token pipelined loop (Executor::run_decode_n) directly
    //    — the real serving path, not the bench lambda. Verifies (a) token
    //    parity vs the per-token run_forward path (diff against BENCH_DUMP) and
    //    (b) that it hits the harness pipeline tok/s through the real executor.
    if (const char* dn = std::getenv("PIE_DECODE_N")) {
        int N = std::atoi(dn); if (N < 1) N = 64;
        Executor exec(*model.graph, *model.kv);
        if (hybrid) exec.set_linear_state_cache(model.lin_cache.get());

        // Single-request decode view: one seed token (5, matching the per-token
        // and pipelined paths) at pos=prefill, identity page table covering
        // [0 .. prefill+N), one greedy sampler.
        const int last_pos = prefill + N - 1;
        const int n_pages  = last_pos / ps + 1;
        std::vector<std::uint32_t> v_tok        = {5};
        std::vector<std::uint32_t> v_pos        = {static_cast<std::uint32_t>(prefill)};
        std::vector<std::uint32_t> v_pageidx(n_pages);
        for (int i = 0; i < n_pages; ++i) v_pageidx[i] = static_cast<std::uint32_t>(i);
        std::vector<std::uint32_t> v_pageindptr = {0, static_cast<std::uint32_t>(n_pages)};
        std::vector<std::uint32_t> v_lastlen    = {static_cast<std::uint32_t>(prefill % ps + 1)};
        std::vector<std::uint32_t> v_qo         = {0, 1};
        std::vector<std::uint32_t> v_sampidx    = {0};
        std::vector<std::uint32_t> v_sampindptr = {0, 1};
        std::vector<std::uint32_t> v_stype      = {1};      // multinomial kind…
        std::vector<float>         v_stemp      = {0.0f};   // …temp 0 → greedy
        std::vector<std::uint32_t> v_stopk      = {0};
        std::vector<float>         v_stopp      = {1.0f};
        std::vector<float>         v_sminp      = {0.0f};
        std::vector<std::uint32_t> v_sseed      = {0};
        std::vector<std::uint32_t> v_rsslot     = {0};

        pie_driver::PieForwardRequestView req{};
        req.token_ids            = {v_tok.data(), v_tok.size()};
        req.position_ids         = {v_pos.data(), v_pos.size()};
        req.kv_page_indices      = {v_pageidx.data(), v_pageidx.size()};
        req.kv_page_indptr       = {v_pageindptr.data(), v_pageindptr.size()};
        req.kv_last_page_lens    = {v_lastlen.data(), v_lastlen.size()};
        req.qo_indptr            = {v_qo.data(), v_qo.size()};
        req.sampling_indices     = {v_sampidx.data(), v_sampidx.size()};
        req.sampling_indptr      = {v_sampindptr.data(), v_sampindptr.size()};
        req.sampler_types        = {v_stype.data(), v_stype.size()};
        req.sampler_temperatures = {v_stemp.data(), v_stemp.size()};
        req.sampler_top_k        = {v_stopk.data(), v_stopk.size()};
        req.sampler_top_p        = {v_stopp.data(), v_stopp.size()};
        req.sampler_min_p        = {v_sminp.data(), v_sminp.size()};
        req.sampler_seeds        = {v_sseed.data(), v_sseed.size()};
        req.rs_slot_ids          = {v_rsslot.data(), v_rsslot.size()};

        pie_driver::ResponseBuilder builder;
        pie_driver::PieForwardResponseView out{};
        auto t0 = std::chrono::high_resolution_clock::now();
        exec.run_decode_n(req, N, builder, out);
        auto t1 = std::chrono::high_resolution_clock::now();
        const double tps = N / std::chrono::duration<double>(t1 - t0).count();

        std::cerr << "[decode_n] tokens:";
        for (std::size_t i = 0; i < out.tokens.size(); ++i)
            std::cerr << " " << out.tokens.data()[i];
        std::cerr << "\n[decode_n] count=" << out.tokens.size()
                  << " tok/s=" << tps << " (N=" << N << ")\n";
        return 0;
    }

    // Accuracy probe: BENCH_DUMP=N prints the first N greedy decode tokens
    // (deterministic argmax) so two builds can be diffed for parity.
    if (const char* d = std::getenv("BENCH_DUMP")) {
        int n = std::atoi(d); if (n < 1) n = 32;
        dump = &seq;
        decode(n);
        std::cerr << "[dump]";
        for (auto v : seq) std::cerr << " " << v;
        std::cerr << "\n";
        return 0;
    }

    decode(16);  // warmup
    std::vector<double> tps;
    for (int r = 0; r < 3; ++r) tps.push_back(decode(steps));
    std::sort(tps.begin(), tps.end());
    std::cerr << "[bench] in-process decode tok/s (median-3): " << tps[1]
              << "  (runs: " << tps[0] << " / " << tps[1] << " / " << tps[2]
              << ")\n";
    std::cerr << "[bench] ms/token: " << 1000.0 / tps[1] << "\n";
    return 0;
}
