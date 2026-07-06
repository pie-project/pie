// qwen35_decode_tput — general PTIR decode-path throughput on Metal under high
// concurrency. The counterpart to the MTP throughput study, for ALL decode
// paths: greedy (argmax), temperature sampling, top-k, top-p (nucleus), min-p,
// and grammar-constrained (mask+argmax). Each path is the sampling-IR op applied
// to the backbone logits [B, vocab]; under continuous batching (B concurrent
// streams) we measure aggregate tokens/s = B / (T_forward(B) + T_sample(B)) and
// how each path scales to GPU saturation.
//
// Concurrency = continuous batching (B streams = a B-token backbone forward with
// B logit rows). T_forward is the real Qwen3.5-0.8B GDN backbone; T_sample is the
// decode-path op over a [B, vocab] logits matrix (the cost that differs per path
// — e.g. top-p sorts the 248320-vocab, greedy just reduces it).
//
// Usage: qwen35_decode_tput <hf_path> [maxB=128] [reps=5]

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <iostream>
#include <limits>
#include <string>
#include <vector>

#include <mlx/mlx.h>

#include "loader/model_loader.hpp"
#include "model/model_graph.hpp"

namespace mx = mlx::core;
using namespace pie_metal_driver;
using Clock = std::chrono::steady_clock;

namespace {

double ms_since(Clock::time_point t0) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
}
double median(std::vector<double> v) {
    if (v.empty()) return 0.0;
    std::sort(v.begin(), v.end());
    return v[v.size() / 2];
}
const float NEG = -1e30f;

model::ForwardBatch batch_n(int n, int page_size) {
    std::vector<int> toks(n, 100), pos(n), rows(n), wi(n);
    for (int i = 0; i < n; ++i) { pos[i] = i; rows[i] = i; wi[i] = i; }
    const int last_pos = n - 1, n_pages = last_pos / page_size + 1;
    std::vector<int> pgi(n_pages);
    for (int i = 0; i < n_pages; ++i) pgi[i] = i;
    return model::ForwardBatch{
        mx::array(toks.data(), {n}, mx::int32), mx::array(pos.data(), {n}, mx::int32),
        mx::array(rows.data(), {n}, mx::int32),
        mx::array(pgi.data(), {n_pages}, mx::int32), mx::array({0, n_pages}, {2}, mx::int32),
        mx::array({last_pos % page_size + 1}, {1}, mx::int32), mx::array({0, n}, {2}, mx::int32),
        mx::array(wi.data(), {n}, mx::int32), n, 1, n, false};
}

// ── decode-path samplers: [B, vocab] logits -> [B] token ids ──
mx::array p_greedy(const mx::array& lg) { return mx::argmax(lg, -1); }

mx::array p_temperature(const mx::array& lg) {
    return mx::random::categorical(mx::multiply(lg, mx::array(1.0f / 0.8f)), -1);
}

mx::array p_top_k(const mx::array& lg, int k) {
    mx::array vals = mx::topk(lg, k, -1);                 // [B,k] top values
    mx::array thr = mx::min(vals, -1, true);              // [B,1] k-th largest
    mx::array masked = mx::where(mx::less(lg, thr), mx::array(NEG), lg);
    return mx::random::categorical(masked, -1);
}

mx::array p_top_p(const mx::array& lg, float p) {
    mx::array desc = mx::argsort(mx::negative(lg), -1);   // descending order idx
    mx::array sl = mx::take_along_axis(lg, desc, -1);     // sorted-desc logits
    mx::array sp = mx::softmax(sl, -1);
    mx::array cs = mx::cumsum(sp, -1);                    // inclusive
    mx::array prev = mx::subtract(cs, sp);               // exclusive prefix mass
    mx::array keep = mx::less(prev, mx::array(p));        // nucleus keeps prefix < p
    mx::array masked = mx::where(keep, sl, mx::array(NEG));
    mx::array samp = mx::random::categorical(masked, -1); // index into sorted
    return mx::take_along_axis(desc, mx::reshape(samp, {lg.shape(0), 1}), -1);
}

mx::array p_min_p(const mx::array& lg, float mp) {
    mx::array probs = mx::softmax(lg, -1);
    mx::array maxp = mx::max(probs, -1, true);            // [B,1]
    mx::array thr = mx::multiply(maxp, mx::array(mp));
    mx::array masked = mx::where(mx::less(probs, thr), mx::array(NEG), lg);
    return mx::random::categorical(masked, -1);
}

mx::array p_grammar(const mx::array& lg, const mx::array& allowed) {
    return mx::argmax(mx::where(allowed, lg, mx::array(NEG)), -1);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::cerr << "usage: qwen35_decode_tput <hf_path> [maxB=128] [reps=5]\n"; return 2; }
    const std::string hf_path = argv[1];
    const int maxB = argc > 2 ? std::atoi(argv[2]) : 128;
    const int reps = argc > 3 ? std::atoi(argv[3]) : 5;

    BatchingConfig batching;
    loader::LoadedModel model = loader::load_model(hf_path, batching);
    const int ps = model.kv->page_size();
    const bool hybrid = (model.lin_cache != nullptr);
    auto attach = [&](model::ForwardBatch& b) {
        if (!hybrid) return;
        b.lin_cache = model.lin_cache.get();
        b.slot_ids = mx::array({0}, {1}, mx::int32);
        b.qo_indptr_host.assign({0, b.n_total});
    };
    auto st = mx::load_safetensors(hf_path + "/model.safetensors-00001-of-00001.safetensors");
    const int vocab = st.first.at("model.language_model.embed_tokens.weight").shape(0);

    std::printf("[decode-tput] Qwen3.5-0.8B on Metal — vocab=%d, reps=%d\n", vocab, reps);
    std::printf("[decode-tput] concurrency = continuous batching (B streams = B-token forward)\n\n");

    auto time_fwd = [&](int n) {
        std::vector<double> ts;
        for (int r = 0; r < reps + 1; ++r) {
            auto b = batch_n(n, ps);
            attach(b);
            auto s = Clock::now();
            mx::array lg = model.graph->forward(b, *model.kv);
            mx::eval(lg); model.kv->eval();
            if (r > 0) ts.push_back(ms_since(s));
        }
        return median(ts);
    };

    struct Path { const char* name; std::function<mx::array(const mx::array&)> fn; };

    std::vector<int> Bs;
    for (int b = 1; b <= maxB; b *= 2) Bs.push_back(b);

    // Table 1: sampling-op cost T_sample(B) per path (ms).
    std::printf("── T_sample (decode-path op over [B,%d] logits, ms) ──\n", vocab);
    std::printf("  %4s | %8s | %8s %8s %8s %8s %8s %8s\n",
                "B", "T_fwd", "greedy", "temp", "top_k", "top_p", "min_p", "grammar");
    std::printf("  %s\n", std::string(78, '-').c_str());

    std::vector<double> Tfwd(Bs.size());
    std::vector<std::vector<double>> Tsamp(6, std::vector<double>(Bs.size()));
    const char* names[6] = {"greedy", "temp", "top_k", "top_p", "min_p", "grammar"};

    for (std::size_t bi = 0; bi < Bs.size(); ++bi) {
        int B = Bs[bi];
        Tfwd[bi] = time_fwd(B);
        mx::array lg = mx::random::normal({B, vocab});
        mx::array allowed = mx::greater(mx::random::normal({B, vocab}), mx::array(0.0f));
        mx::eval(lg); mx::eval(allowed);
        std::vector<Path> paths = {
            {"greedy", [](const mx::array& x) { return p_greedy(x); }},
            {"temp", [](const mx::array& x) { return p_temperature(x); }},
            {"top_k", [](const mx::array& x) { return p_top_k(x, 50); }},
            {"top_p", [](const mx::array& x) { return p_top_p(x, 0.9f); }},
            {"min_p", [](const mx::array& x) { return p_min_p(x, 0.05f); }},
            {"grammar", [&](const mx::array& x) { return p_grammar(x, allowed); }},
        };
        for (int pi = 0; pi < 6; ++pi) {
            std::vector<double> ts;
            for (int r = 0; r < reps + 1; ++r) {
                auto s = Clock::now();
                mx::array tok = paths[pi].fn(lg);
                mx::eval(tok);
                if (r > 0) ts.push_back(ms_since(s));
            }
            Tsamp[pi][bi] = median(ts);
        }
        std::printf("  %4d | %8.2f | %8.2f %8.2f %8.2f %8.2f %8.2f %8.2f\n", B, Tfwd[bi],
                    Tsamp[0][bi], Tsamp[1][bi], Tsamp[2][bi], Tsamp[3][bi], Tsamp[4][bi], Tsamp[5][bi]);
    }

    // Table 2: aggregate throughput tokens/s per path = B / (T_fwd + T_sample).
    std::printf("\n── aggregate throughput (tokens/s = B / (T_fwd + T_sample)) ──\n");
    std::printf("  %4s | %8s %8s %8s %8s %8s %8s\n",
                "B", "greedy", "temp", "top_k", "top_p", "min_p", "grammar");
    std::printf("  %s\n", std::string(66, '-').c_str());
    std::vector<double> peak(6, 0.0);
    for (std::size_t bi = 0; bi < Bs.size(); ++bi) {
        std::printf("  %4d |", Bs[bi]);
        for (int pi = 0; pi < 6; ++pi) {
            double tps = Bs[bi] * 1000.0 / (Tfwd[bi] + Tsamp[pi][bi]);
            peak[pi] = std::max(peak[pi], tps);
            std::printf(" %8.1f", tps);
        }
        std::printf("\n");
    }
    std::printf("\n  peak tokens/s:");
    for (int pi = 0; pi < 6; ++pi) std::printf("  %s=%.0f", names[pi], peak[pi]);
    std::printf("\n  (greedy baseline = %.0f tok/s; each path's peak / greedy peak shows its sampling overhead)\n",
                peak[0]);
    std::printf("  overhead vs greedy:");
    for (int pi = 1; pi < 6; ++pi) std::printf("  %s=%.2fx", names[pi], peak[pi] / peak[0]);
    std::printf("\nBENCH_DONE\n");
    return 0;
}
