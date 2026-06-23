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

#include "loader/model_loader.hpp"
#include "model/model_graph.hpp"

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

    // Prefill to populate the KV cache.
    std::vector<int> ptoks(prefill), ppos(prefill);
    for (int i = 0; i < prefill; ++i) { ptoks[i] = 1 + (i % 100); ppos[i] = i; }
    try {
        auto b = make_batch(ptoks, ppos, prefill - 1, ps, /*pure_decode=*/false);
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

    auto run_decode = [&](int n_steps) -> double {
        int tok = 5;
        auto t0 = std::chrono::high_resolution_clock::now();
        for (int t = 0; t < n_steps; ++t) {
            auto b = make_batch({tok}, {prefill + t}, 0, ps, /*pure_decode=*/true);
            mx::array logits = model.graph->forward(b, *model.kv);  // [1, vocab]
            mx::array next = mx::astype(mx::argmax(logits, 1), mx::uint32);
            next.eval();  // greedy-sampler autoregressive barrier (1 eval/token)
            tok = static_cast<int>(next.item<std::uint32_t>());
        }
        auto t1 = std::chrono::high_resolution_clock::now();
        return n_steps / std::chrono::duration<double>(t1 - t0).count();
    };

    run_decode(16);  // warmup
    std::vector<double> tps;
    for (int r = 0; r < 3; ++r) tps.push_back(run_decode(steps));
    std::sort(tps.begin(), tps.end());
    std::cerr << "[bench] in-process decode tok/s (median-3): " << tps[1]
              << "  (runs: " << tps[0] << " / " << tps[1] << " / " << tps[2]
              << ")\n";
    std::cerr << "[bench] ms/token: " << 1000.0 / tps[1] << "\n";
    return 0;
}
