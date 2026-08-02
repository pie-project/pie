// How fast is the llama family, next to mlx-lm?
//
// Every llama test before this one is a correctness test, and the fastest
// possible driver is the one that computes nothing. This is the other half:
// the same executor `pie serve` calls, driven over a real checkpoint, timed.
//
// It measures the two numbers that are not interchangeable:
//
//   * PREFILL -- a whole prompt in ONE fire. Compute-bound, and the only thing
//     that exercises `affine_qmm_t`, so it is where a tiled GEMM either pays
//     for itself or does not.
//   * DECODE -- one token per fire. Memory-bound: at batch one every weight in
//     the model crosses the bus once per token, so the ceiling is bandwidth
//     divided by the size of the weights, and no amount of arithmetic changes
//     it. Reported as a fraction of that ceiling, because tokens per second on
//     its own says more about the machine than about the driver.
//
// The comparison target is mlx-lm on the SAME checkpoint. That is the fair
// version of the question: an mlx-community 4-bit checkpoint is MLX affine-U4,
// which is the format this driver's kernels already read, so neither side is
// paying a conversion the other avoids.
//
// Skipped (not failed) when the checkpoint is absent, like the forward tests.
//
//   llama_bench [checkpoint_dir] [prompt_tokens] [decode_tokens]

#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "batch/forward.hpp"

using namespace pie::metal;
using namespace pie::metal::batch;

namespace {

bool exists(const std::string& dir) {
    const std::string probe = dir + "/config.json";
    FILE* f = std::fopen(probe.c_str(), "rb");
    if (f == nullptr) return false;
    std::fclose(f);
    return true;
}

double now_s() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

struct Seq {
    std::uint64_t id = 1;
    std::vector<std::uint32_t> tokens;
    std::vector<std::uint32_t> pages;
    std::uint32_t next_position = 0;
};

/// `n` new tokens starting at the sequence's current position, reading only the
/// last row -- the descriptor the runtime builds for a fire.
MemberForwardDesc desc_for(Seq& s, std::uint32_t n, std::uint32_t page_size,
                           std::uint32_t& next_free_page) {
    MemberForwardDesc d;
    d.sequence_id = s.id;
    d.requires_paged = true;
    const std::uint32_t end = s.next_position + n;
    while (std::uint64_t(s.pages.size()) * page_size < end) s.pages.push_back(next_free_page++);
    for (std::uint32_t i = 0; i < n; ++i) {
        d.token_ids.push_back(s.tokens[s.next_position + i]);
        d.position_ids.push_back(s.next_position + i);
    }
    d.qo_indptr = {0u, n};
    d.kv_pages = s.pages;
    d.kv_page_indptr = {0u, std::uint32_t(s.pages.size())};
    d.kv_last_page_lens = {end % page_size == 0 ? page_size : end % page_size};
    d.readout_local_indices = {n - 1};
    d.sampling_indptr = {0u, 1u};
    return d;
}

/// The argmax of a staged logits row, read the way the sampler reads it.
int argmax_of(const LogitsOut& out, std::uint32_t row) {
    const auto* bf = static_cast<const std::uint16_t*>(out.device_contents) +
                     std::size_t(out.device_row_offset + row) * std::size_t(out.vocab);
    const auto f32 = [&](std::uint32_t i) {
        const std::uint32_t bits = std::uint32_t(bf[i]) << 16;
        float f;
        std::memcpy(&f, &bits, 4);
        return f;
    };
    int best = 0;
    float bv = f32(0);
    for (std::uint32_t i = 1; i < out.vocab; ++i) {
        if (const float v = f32(i); v > bv) { bv = v; best = int(i); }
    }
    return best;
}

/// Fire `n` tokens and block until the GPU is done, so the time measured is the
/// work and not the enqueue. Returns the greedy token, or -1 on failure.
int fire(MetalExecutor& exec, Seq& s, std::uint32_t n, std::uint32_t page_size,
         std::uint32_t& next_free_page, bool want_token = false) {
    MemberForwardDesc d = desc_for(s, n, page_size, next_free_page);
    LogitsOut out;
    std::string err;
    if (!exec.forward(d, out, &err)) {
        std::printf("  FAIL  forward: %s\n", err.c_str());
        return -1;
    }
    s.next_position += n;
    return want_token ? argmax_of(out, 0) : 0;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);

    std::string ckpt = argc > 1 ? argv[1] : std::string();
    if (ckpt.empty()) {
        if (const char* env = std::getenv("PIE_BENCH_CKPT"); env != nullptr) ckpt = env;
    }
    const int n_prompt = argc > 2 ? std::atoi(argv[2]) : 128;
    const int n_decode = argc > 3 ? std::atoi(argv[3]) : 64;

    if (ckpt.empty() || !exists(ckpt)) {
        std::printf("llama_bench: SKIP (no checkpoint at '%s')\n", ckpt.c_str());
        return 0;
    }
    std::printf("llama_bench (%s)\n", ckpt.c_str());

    // Qwen3-1.7B's shape. Read off the checkpoint's config.json rather than
    // guessed: this binary is a stopwatch, not a config parser, and a wrong
    // geometry here would be timing a different model.
    SetupConfig cfg;
    cfg.kernels_dir = PIE_METAL_KERNELS_DIR_FOR_TEST;
    cfg.snapshot_dir = ckpt;
    cfg.model_type = "qwen3";
    cfg.vocab_size = 151936;
    cfg.max_forward_tokens = std::uint32_t(std::max(n_prompt, 1));
    cfg.max_forward_requests = 1;
    cfg.kv_page_size = 32;
    cfg.llama.n_layers = 28;
    cfg.llama.hidden = 2048;
    cfg.llama.vocab = 151936;
    cfg.llama.n_q_heads = 16;
    cfg.llama.n_kv_heads = 8;
    cfg.llama.head_dim = 128;
    cfg.llama.intermediate = 6144;
    cfg.llama.eps = 1e-6f;
    cfg.llama.rope_theta = 1.0e6f;
    cfg.llama.qk_norm = true;
    cfg.llama.tied_embeddings = true;

    MetalExecutor exec;
    std::string err;
    const double t_load0 = now_s();
    if (!exec.setup(cfg, &err)) {
        std::printf("  FAIL  setup: %s\n", err.c_str());
        return 1;
    }
    const double load_s = now_s() - t_load0;
    std::printf("  loaded in %.2f s, vocab %u\n", load_s, exec.vocab());

    const std::uint32_t page_size = exec.kv_pool_page_size();

    // An arbitrary but fixed token stream. What is being timed is arithmetic on
    // weights, and that cost does not depend on which tokens they are.
    std::vector<std::uint32_t> prompt;
    for (int i = 0; i < n_prompt; ++i) prompt.push_back(std::uint32_t((i * 137 + 11) % 100000));

    // ── does it compute the right thing at all? ──
    //
    // A benchmark that does not check its own output measures how fast the
    // driver can be wrong. mlx-lm's greedy continuation of this prompt on this
    // checkpoint is the reference; the tokens are hard-coded because the point
    // is to detect a change here, not to re-derive the answer each run.
    {
        const std::vector<std::uint32_t> p{785,  6722, 315,  9625, 374, 12095,
                                           13,   576,  6722, 315,  6323, 374};
        const std::vector<int> want{26194, 13, 576, 6722, 315, 279};  // " Tokyo. The capital of the"
        std::uint32_t page = 0;
        Seq c;
        c.id = 1;
        c.tokens = p;
        c.tokens.resize(p.size() + want.size(), 0u);
        std::vector<int> got;
        int t = fire(exec, c, std::uint32_t(p.size()), page_size, page, true);
        for (std::size_t i = 0; i < want.size() && t >= 0; ++i) {
            got.push_back(t);
            c.tokens[p.size() + i] = std::uint32_t(t);
            t = fire(exec, c, 1, page_size, page, true);
        }
        const bool ok = got == want;
        std::printf("  %s  greedy continuation matches mlx-lm:", ok ? "PASS" : "FAIL");
        for (const int v : got) std::printf(" %d", v);
        std::printf("\n");
        if (!ok) {
            std::printf("        wanted:");
            for (const int v : want) std::printf(" %d", v);
            std::printf("\n");
            return 1;
        }
    }

    // ── warm-up ──
    //
    // The first fire pays for shader specialisation and first-touch of every
    // weight page. Timing it would be timing the loader.
    {
        std::uint32_t page = 0;
        Seq w;
        w.tokens = prompt;
        w.id = 3;
        if (fire(exec, w, std::uint32_t(std::min(n_prompt, 8)), page_size, page) < 0) return 1;
        if (fire(exec, w, 1, page_size, page) < 0) return 1;
    }

    // ── prefill ──
    std::uint32_t next_page = 0;
    Seq s;
    s.id = 4;
    s.tokens = prompt;
    s.tokens.resize(std::size_t(n_prompt + n_decode), 1u);
    const double t0 = now_s();
    if (fire(exec, s, std::uint32_t(n_prompt), page_size, next_page) < 0) return 1;
    const double prefill_s = now_s() - t0;

    // ── decode ──
    //
    // Measured TWICE, because the two numbers answer different questions and
    // only one of them is comparable to mlx-lm.
    //
    // The first feeds each step's own greedy token into the next, which is what
    // a sampler does and what mlx-lm's loop does: the host must see the logits
    // before the next fire can be built, so the GPU cannot run ahead. That is
    // the honest like-for-like figure, and it carries a host-side argmax over
    // the vocabulary that mlx-lm does on the GPU -- a handicap on this side.
    //
    // The second fires a pre-decided token stream, so nothing stops the driver
    // from keeping several command buffers in flight. It is the ceiling the
    // scheduler could reach if sampling were not in the way, and it is reported
    // as such rather than quoted as the decode speed.
    const double t1 = now_s();
    for (int i = 0; i < n_decode; ++i) {
        const int t = fire(exec, s, 1, page_size, next_page, /*want_token=*/true);
        if (t < 0) return 1;
        if (s.next_position < s.tokens.size()) s.tokens[s.next_position] = std::uint32_t(t);
    }
    const double decode_s = now_s() - t1;

    Seq s2;
    s2.id = 5;
    s2.tokens = s.tokens;
    std::uint32_t next_page2 = next_page;
    if (fire(exec, s2, 1, page_size, next_page2) < 0) return 1;
    const double t2 = now_s();
    for (int i = 0; i < n_decode; ++i) {
        if (fire(exec, s2, 1, page_size, next_page2) < 0) return 1;
    }
    const double pipelined_s = now_s() - t2;

    const double prefill_tps = double(n_prompt) / prefill_s;
    const double decode_tps = double(n_decode) / decode_s;
    std::printf("  prefill: %d tok in %.4f s  =  %.1f tok/s\n", n_prompt, prefill_s, prefill_tps);
    std::printf("  decode : %d tok in %.4f s  =  %.1f tok/s  (%.2f ms/tok)  [token fed back]\n",
                n_decode, decode_s, decode_tps, 1000.0 * decode_s / double(n_decode));
    std::printf("  decode : %d tok in %.4f s  =  %.1f tok/s  (%.2f ms/tok)  [no readback: the "
                "scheduler's ceiling, NOT comparable]\n",
                n_decode, pipelined_s, double(n_decode) / pipelined_s,
                1000.0 * pipelined_s / double(n_decode));

    // Decode at batch one reads every weight once per token, so tokens/s times
    // bytes-of-weights is the bandwidth actually achieved. Stating it this way
    // is what makes the number comparable across machines -- and it is the only
    // honest way to say whether a decode kernel is "fast", since the ceiling is
    // the bus and not the ALU.
    const double gib = 1024.0 * 1024.0 * 1024.0;
    double weight_bytes = 0;
    {
        const double h = cfg.llama.hidden, i_ = cfg.llama.intermediate;
        const double q = double(cfg.llama.n_q_heads) * cfg.llama.head_dim;
        const double kv = double(cfg.llama.n_kv_heads) * cfg.llama.head_dim;
        // 4 bits per weight plus a bf16 scale and bias per group of 64.
        const double per = 0.5 + 2.0 * 2.0 / 64.0;
        const double attn = h * q + h * kv * 2 + q * h;
        const double ffn = 3.0 * h * i_;
        weight_bytes = double(cfg.llama.n_layers) * (attn + ffn) * per;
        weight_bytes += double(cfg.llama.vocab) * h * per;  // the tied head
    }
    std::printf("  weights: %.2f GiB  ->  decode moves %.1f GB/s\n", weight_bytes / gib,
                weight_bytes * decode_tps / 1e9);
    return 0;
}
