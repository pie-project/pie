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
#include "model/contract.hpp"
#include "model/llama/encode.hpp"
#include "model/llama/geometry.hpp"
#include "model_facts.hpp"

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
    /// Which recurrent-state slot this sequence's linear-attention history
    /// lives in. A hybrid family carries per-sequence state that is NOT in the
    /// KV pages, and two sequences sharing a slot overwrite each other's.
    std::uint32_t rs_slot = 0;
};

/// `n` new tokens starting at the sequence's current position, reading only the
/// last row -- the descriptor the runtime builds for a fire.
MemberForwardDesc desc_for(Seq& s, std::uint32_t n, std::uint32_t page_size,
                           std::uint32_t& next_free_page, std::uint32_t rs_slots = 0) {
    MemberForwardDesc d;
    d.sequence_id = s.id;
    d.requires_paged = true;
    const std::uint32_t end = s.next_position + n;
    // A page size of zero makes this loop unbounded -- it pushes pages forever,
    // grows the vector until the machine gives out, and says nothing. That is
    // how a driver returning `kv_pool_page_size() == 0` presented itself: eight
    // gigabytes of resident memory and no output at all. An unbounded loop fed
    // by a value the driver chose deserves the check.
    if (page_size == 0) {
        std::printf("  FAIL  the driver's KV pool has no page size, so it is not paged.\n"
                    "        This fire is `requires_paged` and there is nothing to page into.\n");
        std::exit(1);
    }
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
    // A hybrid family carries per-sequence linear-attention state that is NOT
    // in the KV pages. The slot is where that state lives, and a sequence that
    // starts at position zero has none to read yet -- it RESETS, and every
    // continuation of it reads what the fire before wrote. A family with no
    // recurrent slots at all wants none of this said.
    if (rs_slots > 0) {
        d.has_rs_slot = true;
        d.rs_slot_id = s.rs_slot;
        d.rs_reset = s.next_position == 0;
        d.request_rs_slot_ids = {s.rs_slot};
        d.request_rs_reset = {std::uint8_t(d.rs_reset ? 1 : 0)};
        d.request_rs_read = {std::uint8_t(d.rs_reset ? 0 : 1)};
        d.request_rs_write = {std::uint8_t(1)};
    }
    return d;
}

/// Where a staged row's bf16 logits start, at the offset the sampler reads.
const std::uint16_t* bf16_row(const LogitsOut& out, std::uint32_t row) {
    return static_cast<const std::uint16_t*>(out.device_contents) +
           std::size_t(out.device_row_offset + row) * std::size_t(out.vocab);
}

float widen(std::uint16_t h) {
    const std::uint32_t bits = std::uint32_t(h) << 16;
    float f;
    std::memcpy(&f, &bits, 4);
    return f;
}

/// The argmax of a staged logits row, read the way the sampler reads it.
///
/// Deliberately allocation-free: this runs inside the timed decode loop, where
/// the host-side argmax over the vocabulary is already part of what is being
/// measured and a 150k-element copy per step would not be.
int argmax_of(const LogitsOut& out, std::uint32_t row) {
    const std::uint16_t* bf = bf16_row(out, row);
    int best = 0;
    float bv = widen(bf[0]);
    for (std::uint32_t i = 1; i < out.vocab; ++i) {
        if (const float v = widen(bf[i]); v > bv) { bv = v; best = int(i); }
    }
    return best;
}

/// The whole row, widened -- because comparing two fires by their argmax alone
/// cannot tell "a different answer" from "the same answer, one ulp apart".
std::vector<float> row_of(const LogitsOut& out, std::uint32_t row) {
    const std::uint16_t* bf = bf16_row(out, row);
    std::vector<float> v(out.vocab);
    for (std::uint32_t i = 0; i < out.vocab; ++i) v[i] = widen(bf[i]);
    return v;
}

/// Fire `n` tokens and block until the GPU is done, so the time measured is the
/// work and not the enqueue. Returns the greedy token, or -1 on failure.
int fire(MetalExecutor& exec, Seq& s, std::uint32_t n, std::uint32_t page_size,
         std::uint32_t& next_free_page, bool want_token = false,
         LogitsOut* staged = nullptr) {
    MemberForwardDesc d = desc_for(s, n, page_size, next_free_page, exec.rs_slots());
    LogitsOut out;
    std::string err;
    if (!exec.forward(d, out, &err)) {
        std::printf("  FAIL  forward: %s\n", err.c_str());
        return -1;
    }
    s.next_position += n;
    if (staged != nullptr) *staged = out;
    return want_token ? argmax_of(out, 0) : 0;
}

/// "The capital of France is Paris. The capital of Italy is".
///
/// One prompt, used three ways: the greedy gate continues it, the golden-tap
/// dump publishes its fire, and the batched check pairs it with its own
/// eight-token prefix. Writing it out per check invites the day the tap dump
/// and the gate disagree about which fire the taps belong to.
const std::vector<std::uint32_t> kGatePrompt{785, 6722, 315,  9625, 374,  12095,
                                             13,  576,  6722, 315,  6323, 374};

/// The same sentence sixteen times.
///
/// Its only job is to be LONG. A routed prefill switches to the batched
/// mixture at `n_experts * kMoeTileRows / 2` (row, slot) pairs, which on
/// Qwen3-30B-A3B is 1024 pairs -- 128 rows. Every other check in this driver's
/// real-weight path prefills twelve tokens and so runs the mixture as matvecs,
/// which means `affine_qmm_t_routed` -- the kernel a long prompt actually
/// spends its time in -- was benchmarked on this checkpoint and never once
/// checked against it. The synthetic numerics test covers the shape; nothing
/// covered the weights.
///
/// Built by repetition rather than transcribed because 192 token ids in a
/// source file are 192 chances to fix a typo into the expected answer.
std::vector<std::uint32_t> long_gate_prompt() {
    std::vector<std::uint32_t> out;
    for (int i = 0; i < 16; ++i) out.insert(out.end(), kGatePrompt.begin(), kGatePrompt.end());
    return out;
}

/// The shape a gate keys on, whichever family's sub-config holds it.
///
/// `SetupConfig` has one struct per family and a checkpoint fills exactly one of
/// them, so reading `cfg.llama` unconditionally -- which is what this file did
/// -- printed `0 layers, hidden 0` for a qwen3.5 checkpoint and matched it
/// against a table keyed on zeros. Both the printout and the reference lookup
/// go through here so they cannot disagree about which model is running.
struct BenchShape {
    const char* family = "?";
    int n_layers = 0, hidden = 0, n_q_heads = 0, n_kv_heads = 0, head_dim = 0;
    int intermediate = 0, n_experts = 0, experts_per_token = 0, moe_intermediate = 0;
};

BenchShape bench_shape(const SetupConfig& cfg) {
    BenchShape s;
    switch (pie::metal::model::model_family_of(cfg.model_type)) {
    case pie::metal::model::ModelFamily::Qwen35:
        s = {"qwen3.5", cfg.qwen35.n_layers, cfg.qwen35.hidden, cfg.qwen35.n_q_heads,
             cfg.qwen35.n_kv_heads, cfg.qwen35.head_dim, cfg.qwen35.intermediate,
             cfg.qwen35.n_experts, cfg.qwen35.experts_per_token, cfg.qwen35.moe_intermediate};
        break;
    case pie::metal::model::ModelFamily::GptOss:
        s = {"gpt-oss", cfg.gptoss.n_layers, cfg.gptoss.hidden, cfg.gptoss.n_q_heads,
             cfg.gptoss.n_kv_heads, cfg.gptoss.head_dim, cfg.gptoss.intermediate,
             cfg.gptoss.n_experts, cfg.gptoss.experts_per_token, cfg.gptoss.intermediate};
        break;
    case pie::metal::model::ModelFamily::Gemma4:
        s = {"gemma4", cfg.gemma4.n_layers, cfg.gemma4.hidden, cfg.gemma4.n_q_heads,
             cfg.gemma4.n_kv_heads, cfg.gemma4.head_dim, cfg.gemma4.intermediate, 0, 0, 0};
        break;
    case pie::metal::model::ModelFamily::Llama:
    case pie::metal::model::ModelFamily::Unknown:
        s = {"llama", cfg.llama.n_layers, cfg.llama.hidden, cfg.llama.n_q_heads,
             cfg.llama.n_kv_heads, cfg.llama.head_dim, cfg.llama.intermediate,
             cfg.llama.n_experts, cfg.llama.experts_per_token, cfg.llama.moe_intermediate};
        break;
    }
    return s;
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

    // The shape comes from the checkpoint's own config.json, through the SAME
    // parser the runtime uses. Transcribing it here as literals would work for
    // exactly one model, and would silently be timing a different one the day
    // the config changed.
    const ModelFacts facts = read_model_facts(ckpt);
    SetupConfig cfg;
    cfg.kernels_dir = PIE_METAL_KERNELS_DIR_FOR_TEST;
    cfg.snapshot_dir = ckpt;
    cfg.vocab_size = facts.vocab_size;
    // The timing prompt, or the long gate below -- whichever fire is wider.
    cfg.max_forward_tokens =
        std::uint32_t(std::max<std::size_t>(std::size_t(std::max(n_prompt, 32)),
                                            long_gate_prompt().size()));
    // Two: the batch check below fires a pair in ONE pass, and a driver set up
    // for one request would refuse it rather than answer it wrongly.
    cfg.max_forward_requests = 2;
    cfg.kv_page_size = 32;
    // Off by default, so the numbers below stay comparable with every earlier
    // run. Turned on it maps the routed expert bank instead of copying it, and
    // the interesting question -- whether a sparsely read bank costs anything
    // to stream -- is only answerable by running the SAME stopwatch both ways.
    cfg.stream_routed_experts = std::getenv("PIE_METAL_TEST_STREAM_EXPERTS") != nullptr;
    // One sequence, so one sequence's worth of ring. The default is sized for a
    // 64-request fleet and does not scale with the model: at 48 layers it is
    // 13 GiB of KV, which beside a 17 GiB checkpoint does not fit a 32 GiB
    // machine at all. A stopwatch that cannot load the model measures nothing.
    cfg.max_ctx_tokens = std::uint32_t(
        std::max<std::size_t>(std::size_t(std::max(n_prompt + n_decode, 1)),
                              long_gate_prompt().size() + 8) + 64);
    fill_family_geometry(cfg, facts);
    const BenchShape shape = bench_shape(cfg);
    std::printf("  %s [%s]: %d layers, hidden %d, %d/%d heads x %d", cfg.model_type.c_str(),
                shape.family, shape.n_layers, shape.hidden, shape.n_q_heads,
                shape.n_kv_heads, shape.head_dim);
    if (shape.n_experts > 0) {
        std::printf(", %d experts top-%d x %d", shape.n_experts,
                    shape.experts_per_token, shape.moe_intermediate);
    } else {
        std::printf(", ffn %d", shape.intermediate);
    }
    std::printf("\n");

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
        const std::vector<std::uint32_t>& p = kGatePrompt;
        // Keyed by shape rather than by directory name, because the same
        // checkpoint lives under a different path on every machine. A model
        // this table does not know is BENCHED BUT NOT GATED, and says so --
        // the alternative, silently accepting whatever it produced, is how a
        // benchmark starts measuring how fast the driver can be wrong.
        struct Known {
            const char* name;
            int n_layers, n_experts;
            std::vector<int> want;
            /// The continuation of `long_gate_prompt()`, empty if unknown.
            std::vector<int> want_long;
        };
        const std::vector<Known> known{
            // " Tokyo. The capital of the", then "The capital of France is Paris"
            {"Qwen3-1.7B", 28, 0, {26194, 13, 576, 6722, 315, 279},
             {785, 6722, 315, 9625, 374, 12095}},
            // " Tokyo. The capital of Brazil", then the sentence again
            {"Qwen3-30B-A3B", 48, 128, {26194, 13, 576, 6722, 315, 15948},
             {785, 6722, 315, 9625, 374, 12095}},
            // Qwen3.5 tokenizes these ids differently -- its vocabulary is
            // 248320, not Qwen3's 151936 -- so the continuation is a different
            // sentence in the same shape. The gate is on IDS, which is what the
            // driver and mlx-lm both consume, so the disagreement about what
            // they spell is not the gate's business.
            {"Qwen3.5-0.8B", 24, 0, {12095, 13, 576, 6722, 315, 198},
             {785, 6722, 315, 9625, 374, 12095}},
            // Llama-3.2-1B reads these ids as its own vocabulary's text, not
            // the sentence Qwen spells, so the continuation is a different
            // sentence -- and it is the only entry here whose rotation comes
            // from a TABLE (`rope_scaling: llama3`, factor 32) rather than a
            // geometric series, which is the thing this row gates.
            {"Llama-3.2-1B", 16, 0, {12095, 13, 1115, 374, 279, 1890},
             {785, 6722, 315, 9625, 374, 12095}},
        };
        const Known* ref = nullptr;
        for (const Known& k : known) {
            if (k.n_layers == shape.n_layers && k.n_experts == shape.n_experts) {
                ref = &k;
                break;
            }
        }
        const std::vector<int> want = ref != nullptr ? ref->want : std::vector<int>{};
        std::uint32_t page = 0;
        std::uint32_t next_id = 1;

        // Prefill `pr` as one fire, then decode greedily one token at a time,
        // and say whether the result is what mlx-lm produced. Written once and
        // called twice: the short gate and the long one differ in the prompt
        // and in nothing else, and a second copy would be a second place for
        // the comparison to be subtly weaker.
        const auto gate = [&](const std::vector<std::uint32_t>& pr,
                              const std::vector<int>& expect, const char* what) {
            Seq c;
            c.id = next_id++;
            c.tokens = pr;
            c.tokens.resize(pr.size() + expect.size(), 0u);
            std::vector<int> got;
            int t = fire(exec, c, std::uint32_t(pr.size()), page_size, page, true);
            for (std::size_t i = 0; i < expect.size() && t >= 0; ++i) {
                got.push_back(t);
                c.tokens[pr.size() + i] = std::uint32_t(t);
                t = fire(exec, c, 1, page_size, page, true);
            }
            const bool good = ref != nullptr && !expect.empty() && got == expect;
            if (ref == nullptr || expect.empty()) {
                std::printf("  ....  UNGATED (no mlx-lm reference for this shape), produced:");
            } else {
                std::printf("  %s  %s (%s):", good ? "PASS" : "FAIL", what, ref->name);
            }
            for (const int v : got) std::printf(" %d", v);
            std::printf("\n");
            if (ref != nullptr && !expect.empty() && !good) {
                std::printf("        wanted:");
                for (const int v : expect) std::printf(" %d", v);
                std::printf("\n");
                return false;
            }
            return true;
        };

        // A failing gate is exactly when the taps are wanted, so under
        // `PIE_METAL_GOLDEN_DIR` the failure is remembered and the dump below
        // still runs. Without it, stop: the numbers after a wrong answer are
        // the speed of computing the wrong thing.
        const bool dumping = std::getenv("PIE_METAL_GOLDEN_DIR") != nullptr;
        if (!gate(p, want, "greedy continuation matches mlx-lm") && !dumping) return 1;

        // The same check again on a prompt long enough to take the BATCHED
        // mixture. Stated as a precondition rather than assumed: the threshold
        // is arithmetic on the geometry, and the day it moves this check would
        // quietly become a third run of the matvec path it was added to stop
        // covering for.
        if (ref != nullptr && cfg.llama.n_experts > 0) {
            const std::vector<std::uint32_t> lp = long_gate_prompt();
            llama::LlamaGeometry lg;
            std::string ignore;
            const bool have_geo = llama::geometry_from_facts(cfg.llama, lg, &ignore);
            const bool batched =
                have_geo && llama::llama_moe_tile_rows(lg, int(lp.size())) > 1;
            if (!batched) {
                std::printf("  FAIL  the long gate reaches the batched mixture "
                            "(%zu rows is still the matvec path)\n", lp.size());
                return 1;
            }
            std::printf("  ....  long gate: %zu rows, batched mixture\n", lp.size());
            if (!gate(lp, ref->want_long,
                      "batched-mixture continuation matches mlx-lm") && !dumping) {
                return 1;
            }
        }

        // Taps and timings are mutually exclusive, and not as a convenience:
        // `PIE_METAL_GOLDEN_DIR` turns OFF pool recycling so every value keeps
        // its own buffer, which changes the allocation the fire runs against.
        // A number measured under it would be timing a different program. So
        // when taps are asked for, this is the whole run -- and the dump then
        // corresponds to exactly the token list printed here, rather than to
        // whichever synthetic fire happened to go last.
        if (dumping) {
            // One more prefill, on its own pages, so what lands in the dump is
            // the PROMPT's fire rather than the last of the gate's one-row
            // decodes. The gate above has already run; this only re-publishes.
            Seq d;
            d.id = next_id++;
            d.tokens = p;
            std::uint32_t dpage = page;
            fire(exec, d, std::uint32_t(p.size()), page_size, dpage, false);
            std::printf("  taps dumped for:");
            for (const std::uint32_t v : p) std::printf(" %u", v);
            std::printf("\n  (no timings: golden taps disable pool recycling)\n");
            // Both gates above returned non-zero on failure, so reaching here
            // means they passed.
            return 0;
        }
    }


    // ── do two sequences in one fire answer as they do alone? ──
    //
    // Every check above fires ONE sequence, which is the arrangement where
    // ignoring the request axis still gives the right answer. `pie serve` does
    // not do that: it packs several sequences into a pass, each attending its
    // own pages from its own positions. The reference for the pair is the
    // members THEMSELVES run separately -- no new oracle is needed, because
    // batching is supposed to be an optimization and not a computation.
    //
    // The two prompts differ in LENGTH, so a fire that mixed up the row-to-
    // request mapping cannot land on the right answer by symmetry; the shorter
    // is a strict PREFIX of the longer, so a fire that leaked positions or
    // pages between them would answer the longer one twice; and they take pages
    // from a shared allocator in interleaved order, so neither member's page
    // list is contiguous with the other's.
    //
    // The comparison is on the LOGITS ROW, not on the sampled token. Two fires
    // that agree to within a bf16 ulp can still disagree on the argmax, and the
    // first version of this check did exactly that: it called a two-ulp tie
    // between " Paris" and " in" a batching bug. So the row's own top-two
    // margin is measured against the two fires' observed disagreement, and the
    // token is only gated on when the margin is the larger of the two.
    {
        // The gate's prompt, and its eight-token prefix "The capital of
        // France is Paris. The".
        const std::vector<std::uint32_t>& long_p = kGatePrompt;
        const std::vector<std::uint32_t> short_p(long_p.begin(), long_p.begin() + 8);
        // The KV pool here is sized for one sequence, and nothing in this file
        // frees a page, so each independent check starts the allocator over.
        // That is sound because a fire WRITES the pages it names before reading
        // them; what must never repeat is a page WITHIN one fire.
        std::uint64_t next_id = 10;
        const auto alone = [&](const std::vector<std::uint32_t>& p) {
            std::uint32_t page = 0;
            Seq c;
            c.id = next_id++;
            c.tokens = p;
            LogitsOut out;
            if (fire(exec, c, std::uint32_t(p.size()), page_size, page, false, &out) < 0) {
                return std::vector<float>();
            }
            return row_of(out, 0);
        };
        // The argmax, and how far it stands above the runner-up.
        const auto top2 = [](const std::vector<float>& r) {
            int b0 = 0, b1 = -1;
            for (int i = 1; i < int(r.size()); ++i) {
                if (r[i] > r[b0]) { b1 = b0; b0 = i; }
                else if (b1 < 0 || r[i] > r[b1]) { b1 = i; }
            }
            return std::pair<int, float>(b0, r[b0] - r[b1]);
        };
        const auto pair = [&](const std::vector<std::uint32_t>& p0,
                              const std::vector<std::uint32_t>& p1, const char* what) {
            const std::vector<float> want0 = alone(p0);
            const std::vector<float> want1 = alone(p1);
            if (want0.empty() || want1.empty()) return false;
            Seq a, b;
            a.id = next_id++;
            a.tokens = p0;
            a.rs_slot = 0;
            b.id = next_id++;
            b.tokens = p1;
            // Its OWN recurrent slot. A hybrid family's per-sequence linear
            // attention state is not in the KV pages, so two members sharing a
            // slot in one fire compute each other's history.
            b.rs_slot = 1;
            std::uint32_t page = 0;
            std::vector<MemberForwardDesc> descs;
            descs.push_back(desc_for(a, std::uint32_t(p0.size()), page_size, page, exec.rs_slots()));
            descs.push_back(desc_for(b, std::uint32_t(p1.size()), page_size, page, exec.rs_slots()));
            std::vector<LogitsOut> outs(2);
            std::vector<std::uint8_t> ok(2, 0);
            std::vector<std::string> errs(2);
            exec.forward_batch(descs, outs, ok, errs);
            if (ok[0] == 0 || ok[1] == 0) {
                std::printf("  FAIL  one fire, %s: %s / %s\n", what, errs[0].c_str(),
                            errs[1].c_str());
                return false;
            }
            bool good = true;
            const std::vector<float>* want[2] = {&want0, &want1};
            for (int i = 0; i < 2; ++i) {
                const std::vector<float> got = row_of(outs[i], 0);
                const auto [want_tok, margin] = top2(*want[i]);
                const auto [got_tok, _] = top2(got);
                double num = 0.0;
                double den = 0.0;
                float dev = 0.0F;
                for (std::size_t v = 0; v < got.size(); ++v) {
                    const double d0 = double(got[v]) - double((*want[i])[v]);
                    num += d0 * d0;
                    den += double((*want[i])[v]) * double((*want[i])[v]);
                    dev = std::max(dev, std::fabs(got[v] - (*want[i])[v]));
                }
                const double rel = std::sqrt(num / std::max(den, 1e-30));
                // The row is the assertion. Two fires of the same sequence
                // may differ by the arithmetic the batch shape chose -- a head
                // that gathered two rows runs a GEMM where one row ran a matvec
                // -- and that shows up at 1e-3 relative. A row that came from
                // the wrong sequence, or from the wrong position of the right
                // one, shows up near 1: the two live two orders of magnitude
                // apart, with nothing in between to make 0.05 a delicate line.
                const bool same_row = rel < 0.05;
                // Only THEN is the token worth checking, and only when the
                // row's own top-two margin clears what these two fires actually
                // disagreed by. Below that, the argmax is a coin the test has
                // no business calling: " Paris" and " in" after "The capital of
                // France is" sit two bf16 ulps apart, and the first version of
                // this check reported the coin landing differently as a bug.
                const bool decisive = margin > 2.0F * dev;
                const bool agree = got_tok == want_tok;
                const bool bad = !same_row || (decisive && !agree);
                if (bad) good = false;
                std::printf("  %s  one fire, %s member %d: %d vs %d alone "
                            "(margin %.3f, fires differ by %.3f, rel %.5f)%s\n",
                            bad ? "FAIL" : (decisive ? "PASS" : "TIE "), what, i, got_tok,
                            want_tok, double(margin), double(dev), rel,
                            (same_row && !decisive) ? "  [ambiguous: token not gated]" : "");
            }
            return good;
        };
        bool good = pair(long_p, short_p, "long then short");
        good = pair(short_p, long_p, "short then long") && good;
        if (!good) return 1;
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

    // The ceiling has to be measured at the SAME context length as the figure
    // it is the ceiling OF, or the two numbers differ by their attention length
    // as much as by their scheduling. So the second sequence prefills the same
    // prompt -- untimed -- and decodes from the same position.
    //
    // Its pages restart at 0, the way every other independent sequence in this
    // file starts its allocator: nothing here frees a page, `max_ctx_tokens`
    // declares ONE sequence's worth of ring, and a fire writes the pages it
    // names before reading them. Continuing the first sequence's counter ran
    // off the end of the pool at the default argument sizes.
    Seq s2;
    s2.id = 5;
    s2.tokens = s.tokens;
    std::uint32_t next_page2 = 0;
    if (fire(exec, s2, std::uint32_t(n_prompt), page_size, next_page2) < 0) return 1;
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
        const double h = cfg.llama.hidden;
        const double q = double(cfg.llama.n_q_heads) * cfg.llama.head_dim;
        const double kv = double(cfg.llama.n_kv_heads) * cfg.llama.head_dim;
        // 4 bits per weight plus a bf16 scale and bias per group of 64.
        const double per = 0.5 + 2.0 * 2.0 / 64.0;
        const double attn = h * q + h * kv * 2 + q * h;
        // A routed layer reads only the experts it CHOSE, so the width one
        // token pulls through the FFN is `experts_per_token * moe_intermediate`
        // -- not the whole bank, and not `intermediate_size`, which a routed
        // config still carries and which happens to equal the active width on
        // Qwen3-MoE (8 x 768) purely by coincidence.
        const bool routed = cfg.llama.n_experts > 0;
        const double ffn_width = routed ? double(cfg.llama.experts_per_token) *
                                              cfg.llama.moe_intermediate
                                        : double(cfg.llama.intermediate);
        double ffn = 3.0 * h * ffn_width;
        // The router itself is read in full, every token, every layer.
        if (routed) ffn += h * double(cfg.llama.n_experts) * 2.0;
        weight_bytes = double(cfg.llama.n_layers) * (attn + ffn) * per;
        weight_bytes += double(cfg.llama.vocab) * h * per;  // the tied head
    }
    std::printf("  weights: %.2f GiB  ->  decode moves %.1f GB/s\n", weight_bytes / gib,
                weight_bytes * decode_tps / 1e9);
    return 0;
}
