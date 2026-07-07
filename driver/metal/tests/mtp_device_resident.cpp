// mtp_device_resident — the Metal device-resident MTP spec-decode MACHINERY
// (std-attention, dependency-clean), the Metal dual of charlie's CUDA
// fully-pipelined MTP (bravo's `ptir-fully-pipelined-mtp-sketch`).
//
// Investigation → build: `ptir-metal-mtp-specdecode-investigation`. The three
// device-resident axes composed on Metal:
//   (iii) back-to-back submit  — my async_eval double-buffer (already validated)
//   (i)   on-device accept DAG — spec_verify_greedy as mx:: ops (no .item() on
//                                the hot path): argmax→eq→cumprod→n_acc→commit
//   (ii)  device-resident carrier — the [K+1] window `[seed, drafts]` retained as
//                                an mx::array, device-aliased into the next fire;
//                                seed = take(target, n_acc) (device gather-by-value).
//
// Model = Qwen3-0.6B (std-attention, paged-KV) — dependency-clean (the Qwen3.5
// GDN recurrent-state rewind-on-reject is DEFERRED, gated on bravo's carrier).
// No trained MTP head on Qwen3-0.6B, so drafts come from an alpha-oracle (the
// reference greedy sequence, corrupted to a fixed accept count r per fire)
// standing in for the MTP head's device output — this ISOLATES the device-
// resident verify+carrier+back-to-back machinery. Correctness gate = the
// spec-decode INVARIANT: whatever the draft quality, the committed sequence ==
// plain greedy decode, bit-exact. The A/B quantifies the host-round-trip-removal
// margin as a function of (K, alpha) — the payoff the real MTP head realizes.
//
// Usage: mtp_device_resident <hf_path> [gen=96]   (env PIPE_HOST_US=<us/fire>)

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <tuple>
#include <vector>

#include <mlx/mlx.h>

#include "loader/model_loader.hpp"
#include "model/model_graph.hpp"

namespace mx = mlx::core;
using namespace pie_metal_driver;
using Clock = std::chrono::steady_clock;

namespace {

int g_host_us = 0;
void host_work() {
    if (g_host_us <= 0) return;
    auto until = Clock::now() + std::chrono::microseconds(g_host_us);
    volatile std::uint64_t x = 0;
    while (Clock::now() < until) x++;
}

// Forward batch over `tokens` at absolute positions [base .. base+n-1], KV
// written to those slots (overwrite-after handles rejected-draft KV), logits at
// every row. pure_decode only for the single-token (decode) case.
model::ForwardBatch make_batch(const std::vector<int>& tokens, int base, int ps) {
    const int n = (int)tokens.size();
    std::vector<int> pos(n), rows(n), wr(n);
    for (int i = 0; i < n; ++i) { pos[i] = base + i; rows[i] = i; wr[i] = base + i; }
    const int last = base + n - 1;
    const int n_pages = last / ps + 1;
    std::vector<int> pgi(n_pages);
    for (int i = 0; i < n_pages; ++i) pgi[i] = i;
    return model::ForwardBatch{
        mx::array(tokens.data(), {n}, mx::int32),
        mx::array(pos.data(), {n}, mx::int32),
        mx::array(rows.data(), {n}, mx::int32),
        mx::array(pgi.data(), {n_pages}, mx::int32),
        mx::array({0, n_pages}, {2}, mx::int32),
        mx::array({last % ps + 1}, {1}, mx::int32),
        mx::array({0, n}, {2}, mx::int32),
        mx::array(wr.data(), {n}, mx::int32),
        n, 1, n, /*pure_decode=*/n == 1,
    };
}

struct Model {
    loader::LoadedModel& m;
    int ps;
    mx::array forward(const std::vector<int>& tokens, int base) {
        auto b = make_batch(tokens, base, ps);
        return m.graph->forward(b, *m.kv);
    }
    // Device-token forward: `toks_dev` [n] int32 feeds token_ids DIRECTLY (no
    // host read) — the index/KV plumbing is host-computed from the deterministic
    // cursor. This is what lets the window mx::array flow fire→fire device-resident.
    mx::array forward_dev(const mx::array& toks_dev, int base, int n) {
        std::vector<int> pos(n), rows(n), wr(n);
        for (int i = 0; i < n; ++i) { pos[i] = base + i; rows[i] = i; wr[i] = base + i; }
        const int last = base + n - 1;
        const int n_pages = last / ps + 1;
        std::vector<int> pgi(n_pages);
        for (int i = 0; i < n_pages; ++i) pgi[i] = i;
        model::ForwardBatch b{
            mx::astype(toks_dev, mx::int32),
            mx::array(pos.data(), {n}, mx::int32),
            mx::array(rows.data(), {n}, mx::int32),
            mx::array(pgi.data(), {n_pages}, mx::int32),
            mx::array({0, n_pages}, {2}, mx::int32),
            mx::array({last % ps + 1}, {1}, mx::int32),
            mx::array({0, n}, {2}, mx::int32),
            mx::array(wr.data(), {n}, mx::int32),
            n, 1, n, /*pure_decode=*/n == 1,
        };
        return m.graph->forward(b, *m.kv);
    }
};

mx::array argmax_i32(const mx::array& logits) {  // [n,V] -> [n] int32
    return mx::astype(mx::argmax(logits, /*axis=*/1), mx::int32);
}

// Plain greedy decode of `n` tokens (the reference sequence + baseline tput).
std::pair<std::vector<int>, double> greedy_decode(Model& M, const std::vector<int>& prompt,
                                                  int n) {
    const int L = (int)prompt.size();
    mx::array lg = M.forward(prompt, 0);
    mx::eval(lg);
    M.m.kv->eval();
    mx::array a0 = argmax_i32(mx::slice(lg, {L - 1, 0}, {L, lg.shape(1)}));
    mx::eval(a0);
    int cur = a0.data<std::int32_t>()[0];
    std::vector<int> seq;
    auto t0 = Clock::now();
    for (int i = 0; i < n; ++i) {
        seq.push_back(cur);
        mx::array a = argmax_i32(M.forward({cur}, L + i));
        mx::eval(a);
        host_work();
        cur = a.data<std::int32_t>()[0];
    }
    double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    return {seq, ms};
}

// alpha-oracle drafter (host, shared by both paths — stands in for the MTP head):
// the K drafts for the window anchored at gen-index `gi` are the true greedy
// tokens greedy[gi+1+j], with position `r` corrupted (forces exactly n_acc=r
// accepts). r is a fixed policy ⇒ the cursor is deterministic ⇒ the loop pipelines.
std::vector<int> oracle_drafts(const std::vector<int>& greedy, int gi, int K, int r,
                               int vocab) {
    std::vector<int> d(K);
    for (int j = 0; j < K; ++j) {
        const int truth = greedy[gi + 1 + j];
        d[j] = (j == r) ? (truth + 1) % vocab : truth;
    }
    return d;
}

// ── (A) HOST-round-trip spec-decode: verify via host reads each fire (bubble). ──
std::pair<std::vector<int>, double> spec_host(Model& M, const std::vector<int>& prompt,
                                              const std::vector<int>& greedy, int gen,
                                              int K, int r, int vocab) {
    const int L = (int)prompt.size();
    mx::array lg = M.forward(prompt, 0);
    mx::eval(lg);
    M.m.kv->eval();
    mx::array a0 = argmax_i32(mx::slice(lg, {L - 1, 0}, {L, lg.shape(1)}));
    mx::eval(a0);
    int seed = a0.data<std::int32_t>()[0];  // greedy[0]
    std::vector<int> committed{seed};
    int gi = 0;
    auto t0 = Clock::now();
    while ((int)committed.size() < gen) {
        std::vector<int> drafts = oracle_drafts(greedy, gi, K, r, vocab);
        std::vector<int> window{seed};
        window.insert(window.end(), drafts.begin(), drafts.end());
        mx::array tgt = argmax_i32(M.forward(window, L + gi));  // [K+1]
        mx::eval(tgt);
        const std::int32_t* td = tgt.data<std::int32_t>();
        int n_acc = 0;
        while (n_acc < K && drafts[n_acc] == td[n_acc]) ++n_acc;
        for (int i = 0; i <= n_acc && (int)committed.size() < gen; ++i)
            committed.push_back(td[i]);
        seed = td[n_acc];
        gi += 1 + n_acc;
        host_work();
    }
    double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    committed.resize(gen);
    return {committed, ms};
}

// ── (B) DEVICE-RESIDENT spec-decode: verify DAG + seed-gather + window carrier
//    all as mx:: ops (no .item() on the hot path); the [K+1] window mx::array is
//    retained + device-aliased into the next fire; back-to-back async_eval. ──
std::pair<std::vector<int>, double> spec_device(Model& M, const std::vector<int>& prompt,
                                                const std::vector<int>& greedy, int gen,
                                                int K, int r, int vocab) {
    const int L = (int)prompt.size();
    mx::array lg = M.forward(prompt, 0);
    mx::eval(lg);
    M.m.kv->eval();
    mx::array a0 = argmax_i32(mx::slice(lg, {L - 1, 0}, {L, lg.shape(1)}));
    mx::eval(a0);
    const int seed_host = a0.data<std::int32_t>()[0];  // greedy[0] anchor

    auto build_window = [&](const mx::array& seed_dev, int gi) {
        std::vector<int> drafts = oracle_drafts(greedy, gi, K, r, vocab);
        mx::array d = mx::array(drafts.data(), {K}, mx::int32);
        return mx::concatenate({seed_dev, d}, 0);  // [K+1] device
    };

    std::vector<mx::array> commit_tiles;  // each [K+1] device commit tail (retained)
    mx::array seed_dev = mx::array({seed_host}, {1}, mx::int32);
    int gi = 0;
    const int fires = gen / (r + 1) + 2;

    auto t0 = Clock::now();
    mx::array window = build_window(seed_dev, gi);
    for (int f = 0; f < fires && gi < gen; ++f) {
        // The window mx::array feeds token_ids DIRECTLY (no host read) — verify
        // (target argmax, match, n_acc, seed) stays device-resident; the window
        // flows fire→fire on-device (the carrier).
        mx::array tgt = argmax_i32(M.forward_dev(window, L + gi, K + 1));  // [K+1] device
        mx::array drafts = mx::slice(window, {1}, {K + 1});           // [K] device
        mx::array match = mx::astype(
            mx::equal(drafts, mx::slice(tgt, {0}, {K})), mx::int32);  // [K] device
        mx::array keep = mx::cumprod(match, /*axis=*/0);              // [K] device
        mx::array n_acc = mx::sum(keep);                             // scalar device
        mx::array idx = mx::arange(K + 1, mx::int32);
        mx::array commit = mx::where(mx::less_equal(idx, n_acc), tgt,
                                     mx::full({K + 1}, -1, mx::int32));  // [K+1] device
        commit_tiles.push_back(commit);
        // next seed = tgt[n_acc] — device gather-by-value (the §8.1 seam, clean on MLX)
        mx::array next_seed = mx::take(tgt, mx::reshape(n_acc, {1}), /*axis=*/0);  // [1]
        gi += 1 + r;  // fixed-policy cursor (deterministic ⇒ pipelinable)
        if (gi >= gen) { mx::async_eval(std::vector<mx::array>{commit}); break; }
        window = build_window(next_seed, gi);
        // Back-to-back submit (axis iii): schedule the NEXT window chain + this
        // fire's commit WITHOUT blocking on a host harvest.
        mx::async_eval(std::vector<mx::array>{window, commit});
        host_work();
    }
    // Async harvest the retained commit tails (off the hot path) → greedy sequence.
    std::vector<int> committed{seed_host};
    for (std::size_t f = 0; f < commit_tiles.size() && (int)committed.size() < gen; ++f) {
        mx::eval(commit_tiles[f]);
        const std::int32_t* cd = commit_tiles[f].data<std::int32_t>();
        for (int i = 0; i <= r && (int)committed.size() < gen; ++i) committed.push_back(cd[i]);
    }
    double ms = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    committed.resize(gen);
    return {committed, ms};
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) { std::fprintf(stderr, "usage: mtp_device_resident <hf_path> [gen=96]\n"); return 2; }
    const std::string hf = argv[1];
    const int gen = argc > 2 ? std::atoi(argv[2]) : 96;
    if (const char* e = std::getenv("PIPE_HOST_US")) g_host_us = std::atoi(e);

    BatchingConfig batching;
    loader::LoadedModel lm = loader::load_model(hf, batching);
    Model M{lm, lm.kv->page_size()};
    const int vocab = lm.caps.vocab_size;
    std::printf("[mtp-dr] arch=%s layers=%d page_size=%d lin_cache=%s vocab=%d gen=%d host_us=%d\n",
                lm.caps.arch_name.c_str(), lm.caps.num_hidden_layers, M.ps,
                lm.lin_cache ? "yes(GDN)" : "no(std-attn)", vocab, gen, g_host_us);

    std::vector<int> prompt;
    for (int i = 0; i < 8; ++i) prompt.push_back(785 + i * 13);

    const int margin = 16;
    greedy_decode(M, prompt, gen + margin);  // warm
    auto [greedy, ms_greedy] = greedy_decode(M, prompt, gen + margin);
    const double greedy_tps = (gen + margin) * 1000.0 / ms_greedy;
    std::printf("\n[mtp-dr] greedy baseline: %.1f tok/s (%.2f ms / %d tok)\n",
                greedy_tps, ms_greedy, gen + margin);

    bool all_ok = true;
    std::printf("\n[mtp-dr] device-resident spec-decode MACHINERY (Qwen3-0.6B std-attn, paged-KV):\n");
    std::printf("  correctness gate = committed==greedy (the spec-decode invariant), any draft quality\n");
    for (int K : {1, 2, 4}) {
        for (int r : {K, (K + 1) / 2}) {
            if (r > K || r < 0) continue;
            const double alpha = (double)r / K;
            auto [ch, msh] = spec_host(M, prompt, greedy, gen, K, r, vocab);
            auto [cd, msd] = spec_device(M, prompt, greedy, gen, K, r, vocab);
            bool okh = std::equal(ch.begin(), ch.end(), greedy.begin());
            bool okd = std::equal(cd.begin(), cd.end(), greedy.begin());
            all_ok = all_ok && okh && okd;
            std::printf("  K=%d a=%.2f | host %7.1f tok/s (%.2fx) | device %7.1f tok/s (%.2fx) | host:%s dev:%s\n",
                        K, alpha, gen * 1000.0 / msh, (gen * 1000.0 / msh) / greedy_tps,
                        gen * 1000.0 / msd, (gen * 1000.0 / msd) / greedy_tps,
                        okh ? "OK" : "FAIL", okd ? "OK" : "FAIL");
            if (!okd) for (int i = 0; i < gen; ++i) if (cd[i] != greedy[i]) {
                std::printf("    dev MISMATCH at %d: %d vs greedy %d\n", i, cd[i], greedy[i]); break;
            }
        }
    }
    std::printf("%s\n", all_ok ? "MTP_DEVICE_RESIDENT_OK" : "MTP_DEVICE_RESIDENT_FAIL");
    return all_ok ? 0 : 1;
}
