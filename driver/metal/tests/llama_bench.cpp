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
#include <filesystem>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "loader/heap_bind_metal.hpp"
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

/// How many bytes of weights this checkpoint is, off the filesystem.
///
/// The plan's own figure would be better, but it is behind `MetalExecutor` and
/// the point of the probe is to run BEFORE one exists. The weight files are the
/// same number to within their headers, and the probe only needs a capacity
/// that is plainly under the model.
std::uint64_t checkpoint_weight_bytes(const std::string& dir) {
    std::uint64_t total = 0;
    std::error_code ec;
    for (const auto& entry : std::filesystem::directory_iterator(dir, ec)) {
        if (ec || !entry.is_regular_file(ec)) continue;
        const std::string ext = entry.path().extension().string();
        if (ext != ".safetensors" && ext != ".zt" && ext != ".gguf" && ext != ".bin") continue;
        total += entry.file_size(ec);
    }
    return total;
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
                           std::uint32_t& next_free_page, std::uint32_t rs_slots = 0,
                           bool greedy_token_only = false) {
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
    d.greedy_token_only = greedy_token_only;
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
    if (out.device_contents == nullptr) {
        if (out.greedy_contents != nullptr && row < out.rows) {
            return int(out.greedy_contents[out.greedy_row_offset + row]);
        }
        std::printf("  FAIL  no logits or device argmax for row %u\n", row);
        return -1;
    }
    const std::uint16_t* bf = bf16_row(out, row);
    int best = 0;
    float bv = widen(bf[0]);
    for (std::uint32_t i = 1; i < out.vocab; ++i) {
        if (const float v = widen(bf[i]); v > bv) { bv = v; best = int(i); }
    }
    if (out.greedy_contents != nullptr && row < out.rows &&
        out.greedy_contents[out.greedy_row_offset + row] != std::uint32_t(best)) {
        std::printf("  FAIL  device argmax row %u says %u, host scan says %d\n",
                    row, out.greedy_contents[out.greedy_row_offset + row], best);
        return -1;
    }
    if (out.greedy_contents != nullptr && row < out.rows) {
        return int(out.greedy_contents[out.greedy_row_offset + row]);
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
         LogitsOut* staged = nullptr, bool greedy_token_only = false) {
    MemberForwardDesc d =
        desc_for(s, n, page_size, next_free_page, exec.rs_slots(), greedy_token_only);
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
             cfg.gemma4.n_kv_heads, cfg.gemma4.head_dim, cfg.gemma4.intermediate,
             cfg.gemma4.n_experts, cfg.gemma4.experts_per_token,
             cfg.gemma4.moe_intermediate};
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
    // Concurrent sequences for the throughput block at the end. One means the
    // block does not run at all, and -- more importantly -- that none of the
    // sizing below changes, so every number this harness has ever printed
    // stays comparable with every number it prints now. A fleet needs a wider
    // request limit, a wider fire and n times the ring, and all three feed the
    // scratch sizer.
    const int n_seqs = [] {
        const char* e = std::getenv("PIE_BENCH_TPUT");
        const int v = e != nullptr ? std::atoi(e) : 1;
        return v > 1 ? v : 1;
    }();

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
    // The timing prompt, the long gate, or a whole decoding fleet in one fire
    // -- whichever is widest.
    cfg.max_forward_tokens =
        std::uint32_t(std::max<std::size_t>(std::size_t(std::max(n_prompt, 32)),
                                            long_gate_prompt().size()));
    if (std::uint32_t(n_seqs) > cfg.max_forward_tokens) {
        cfg.max_forward_tokens = std::uint32_t(n_seqs);
    }
    // Two: the batch check below fires a pair in ONE pass, and a driver set up
    // for one request would refuse it rather than answer it wrongly.
    cfg.max_forward_requests = std::uint32_t(std::max(2, n_seqs));
    cfg.kv_page_size = 32;
    // Off by default, so the numbers below stay comparable with every earlier
    // run. Turned on it maps the routed expert bank instead of copying it, and
    // the interesting question -- whether a sparsely read bank costs anything
    // to stream -- is only answerable by running the SAME stopwatch both ways.
    cfg.stream_routed_experts = std::getenv("PIE_METAL_TEST_STREAM_EXPERTS") != nullptr;
    // A megabyte budget for the routed experts, which turns mapping OFF and
    // pages them through a slab instead. This is the only setting under which a
    // model can exceed the machine, and the only way to exercise it here: no
    // checkpoint on this box is bigger than its working set, so the bounded
    // budget stands in for the case that cannot be run directly.
    if (const char* mb = std::getenv("PIE_METAL_EXPERT_SLAB_MB")) {
        cfg.expert_slab_bytes = std::uint64_t(std::strtoull(mb, nullptr, 10)) * 1024ull * 1024ull;
    }
    // One sequence, so one sequence's worth of ring. The default is sized for a
    // 64-request fleet and does not scale with the model: at 48 layers it is
    // 13 GiB of KV, which beside a 17 GiB checkpoint does not fit a 32 GiB
    // machine at all. A stopwatch that cannot load the model measures nothing.
    cfg.max_ctx_tokens =
        std::uint32_t(std::max<std::size_t>(std::size_t(std::max(n_prompt + n_decode, 1)),
                                            long_gate_prompt().size() + 8) +
                      64) *
        std::uint32_t(n_seqs);
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

    // Before the real setup: the same setup against a device that has been told
    // it will hold 256 MiB. Every checkpoint this harness is pointed at runs
    // this, so between them it covers BOTH setup paths -- `setup_simple` for
    // llama, qwen3, gemma4 and gpt-oss, and qwen3.5's own -- which is the point.
    // The capacity refusal was written for qwen3.5 and only qwen3.5 performed
    // it; the other path created the heap, copied the weights in, and let the
    // first command buffer come back with an unreadable out-of-memory. A check
    // one caller makes is not a check the driver makes.
    //
    // This costs nothing: the refusal happens before a byte is allocated.
    //
    // Twice, at two capacities, because one of them proves nothing on its own.
    // 256 MiB is under any checkpoint's SCRATCH, so it refuses even a driver
    // that has forgotten the weights entirely -- which is exactly what happened
    // when weights bound where they lie stopped being counted: a 17.17 GB model
    // reported that it needed 0.326 GiB, and this probe still passed. Half the
    // weights is a capacity only the right arithmetic refuses.
    //
    // With a slab budget the SECOND probe inverts, and that inversion is the
    // gate for expert paging as a whole. A budget means the routed bank is read
    // from an mmap the GPU never sees, so it is neither allocated nor resident,
    // and a device holding half the weights must ADMIT the model -- which is
    // the entire capability, stated as the one thing that distinguishes it from
    // every earlier arrangement. Nothing on this machine is bigger than its
    // working set, so this stands in for the case that cannot be run directly.
    const std::uint64_t weight_bytes = checkpoint_weight_bytes(cfg.snapshot_dir);
    const bool paging = cfg.expert_slab_bytes > 0;
    for (const std::uint64_t hold :
         {std::uint64_t(256u << 20), weight_bytes / 2}) {
        if (hold == 0) continue;
        if (paging && hold == weight_bytes / 2) {
            RawMetalContext::set_device_working_set_bytes_for_test(std::size_t(hold));
            MetalExecutor admitted;
            std::string why;
            const bool ok = admitted.setup(cfg, &why);
            RawMetalContext::set_device_working_set_bytes_for_test(0);
            if (!ok) {
                std::printf("  FAIL  paging refused a model it should hold: %s\n", why.c_str());
                return 1;
            }
            std::printf("  PASS  paged the experts onto a device holding %.2f GiB,"
                        " against %.2f GiB of weights\n",
                        double(hold) / (1 << 30), double(weight_bytes) / (1 << 30));
            continue;
        }
        RawMetalContext::set_device_working_set_bytes_for_test(std::size_t(hold));
        MetalExecutor probe;
        std::string too_big;
        const bool set_up = probe.setup(cfg, &too_big);
        RawMetalContext::set_device_working_set_bytes_for_test(0);
        if (set_up) {
            std::printf("  FAIL  setup succeeded on a device that said it would hold %.2f GiB,"
                        " against %.2f GiB of weights\n",
                        double(hold) / (1 << 30), double(weight_bytes) / (1 << 30));
            return 1;
        }
        if (too_big.find("does not fit this GPU") == std::string::npos) {
            std::printf("  FAIL  refused, but not for want of room: %s\n", too_big.c_str());
            return 1;
        }
        std::printf("  PASS  refused before allocating when the model does not fit: %s\n",
                    too_big.c_str());
    }

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
    // The real sentence, repeated to length. This used to be `(i*137+11) %
    // 100000` -- ids that spell nothing -- and that left the logits nearly flat,
    // so the greedy argmax was a coin flip and any comparison of two token
    // streams was unfalsifiable: the fleet check below read "agrees for 1 of 64
    // steps" on a driver that was perfectly correct. These kernels' timing is
    // data-independent (fixed loop bounds, no early exit), so the change costs
    // the comparison against mlx-lm nothing and buys the check its teeth.
    while (int(prompt.size()) < n_prompt)
        prompt.insert(prompt.end(), kGatePrompt.begin(), kGatePrompt.end());
    prompt.resize(std::size_t(n_prompt));

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
            /// Shape alone stopped being a key the day a second quantization of
            /// the SAME model was gated: mlx-community's 8-bit Llama-3.2-1B is
            /// 16 layers and 0 experts, exactly like the 4-bit one, and reads
            /// the prompt as a different sentence because its weights are
            /// different. Keyed on shape alone it inherited the 4-bit row and
            /// failed while being right.
            /// `hidden` joined them when Llama-3.2-3B arrived: 28 layers and
            /// no experts is also Qwen3-1.7B, and the 3B ran that row's
            /// answer and failed while being right.
            int n_layers, n_experts, quant_bits, hidden;
            /// The affine group. Part of the key because it is part of the
            /// answer: the same weights at g32 and at g64 are different
            /// numbers, and a table that could not tell them apart would have
            /// compared a g32 run against a g64 continuation and called a
            /// correct driver wrong.
            int quant_group;
            std::vector<int> want;
            /// The continuation of `long_gate_prompt()`, empty if unknown.
            std::vector<int> want_long;
        };
        const std::vector<Known> known{
            // " Tokyo. The capital of the", then "The capital of France is Paris"
            {"Qwen3-1.7B", 28, 0, 4, 2048, 64, {26194, 13, 576, 6722, 315, 279},
             {785, 6722, 315, 9625, 374, 12095}},
            // " Tokyo. The capital of Brazil", then the sentence again
            {"Qwen3-30B-A3B", 48, 128, 4, 2048, 64, {26194, 13, 576, 6722, 315, 15948},
             {785, 6722, 315, 9625, 374, 12095}},
            // Qwen3.5 tokenizes these ids differently -- its vocabulary is
            // 248320, not Qwen3's 151936 -- so the continuation is a different
            // sentence in the same shape. The gate is on IDS, which is what the
            // driver and mlx-lm both consume, so the disagreement about what
            // they spell is not the gate's business.
            {"Qwen3.5-0.8B", 24, 0, 4, 1024, 64, {12095, 13, 576, 6722, 315, 198},
             {785, 6722, 315, 9625, 374, 12095}},
            // Llama-3.2-1B reads these ids as its own vocabulary's text, not
            // the sentence Qwen spells, so the continuation is a different
            // sentence -- and it is the only entry here whose rotation comes
            // from a TABLE (`rope_scaling: llama3`, factor 32) rather than a
            // geometric series, which is the thing this row gates.
            {"Llama-3.2-1B", 16, 0, 4, 2048, 64, {12095, 13, 1115, 374, 279, 1890},
             {785, 6722, 315, 9625, 374, 12095}},
            // Gemma 4 E2B. Its 262144-entry vocabulary reads these ids as
            // nothing in particular, so the continuation is not a sentence --
            // and it does not need to be. The gate is on IDS, which is what the
            // driver and mlx-lm both consume. What this row covers that no
            // other does: two attention widths in one stack (256 sliding, 512
            // full), a KV-shared tail, per-layer embeddings, and a softcapped
            // logit -- none of which any other entry exercises at all.
            {"Gemma-4-E2B", 35, 0, 4, 1536, 64, {56971, 55353, 374, 56971, 55353, 374},
             {785, 6722, 48352, 6722, 48352, 6722}},
            // Gemma-4-26B-A4B, the mixture member of the same family. What it
            // covers that the E2B does not: 128 experts at top-8 behind a
            // router whose norm carries a gain the checkpoint does not store,
            // a SECOND dense branch summed with the routed one, k-eq-v global
            // attention that ships no v_proj at all, and a checkpoint that
            // quantizes its dense mlp and its router at 8 bits while everything
            // else is 4 -- two full pipeline sets in one stack.
            //
            // This row's answers are the DRIVER's, and that is deliberate. Every
            // other row here is mlx-lm's continuation, because for every other
            // checkpoint mlx-lm HAS one. This one does not: run the same prompt
            // three ways and mlx-lm answers three different things.
            //
            //   prompt re-prefilled each step 246427 246427 246427 246427 246427 246427
            //   through its own kv cache    246427 246427 246427    478    397    389
            //   ties broken by index        785 6722 244674 236764 100 45518 (long)
            //
            // The cause is in the checkpoint, not in either implementation. The
            // router's scores are bf16 over 128 experts packed into a two-logit
            // band, so the eighth and ninth are routinely EQUAL -- exactly equal,
            // same bits -- and which of the two a top-k returns is a choice no
            // arithmetic makes. One flip swaps a tenth of the mixture, and thirty
            // layers of that reach the logits.
            //
            // So the arithmetic was gated instead, against the driver's own
            // prefill. The same thirteen tokens run as one prefill and as a
            // twelve-token prefill plus a decode agree to 1e-5 through layer 5;
            // at layer 6 the eighth and ninth scores are the same bf16 number and
            // the two paths take different ones. Feed the checkpoint's own expert
            // weights the driver's selection and its mixture reconstructs at
            // cos 0.99998 -- the mixture is right, the tie is a coin.
            //
            // The row still discriminates: it is six exact tokens from a
            // 30-layer 128-expert stack, and nothing that broke the mixture, the
            // second dense branch, the 8-bit set or the k-eq-v attention would
            // reproduce them.
            {"Gemma-4-26B-A4B", 30, 128, 4, 2816, 64,
             {246427, 243599, 243599, 243599, 243599, 243599},
             {785, 6722, 244674, 236900, 238270, 237184}},
            // gpt-oss-20b. The only MXFP4 checkpoint here, and the row that
            // says the driver reads openai's format rather than converting it:
            // until the expert bank was bound as the E2M1 nibbles and E8M0
            // block exponents it ships as, the loader dequantized it and
            // re-quantized it affine-U4 -- sixteen levels times a power of two
            // forced through a 15-step grid, so the two runs did not hold the
            // same weights and no row could exist. They do now.
            // Keyed at 32 because that is what its `quantization` block declares --
            // a global mxfp4 g32 that most of its tensors then override back to
            // affine g64. The key is what the config says, not what any one
            // tensor is, which is the same thing every other row's group means.
            {"gpt-oss-20b", 24, 32, 4, 2880, 32, {13, 279, 410, 12038, 410, 25},
             {785, 6722, 315, 9625, 1455, 12095}},
            // The same Llama-3.2-1B at 8 bits. It is here because it is the
            // only row that exercises a width other than 4 across a WHOLE
            // model -- the embedding gather, every projection, the batched
            // prefill matmul and the untied head. gpt-oss's 8-bit router
            // covers one dense matvec and nothing else.
            {"Llama-3.2-1B-8bit", 16, 0, 8, 2048, 64, {12095, 13, 420, 6722, 315, 6323},
             {785, 6722, 315, 9625, 374, 12095}},
            // Llama-3.2-3B. Same family and the same llama3 rope table as the
            // 1B, at 24 query heads over 8 kv heads with head_dim 128 -- the
            // 1B is 32/8 at 64 -- so it is the row that says the geometry is
            // read rather than assumed. It reads these ids as version numbers,
            // and " 1.0.0" is what mlx-lm produces too.
            {"Llama-3.2-3B", 28, 0, 4, 3072, 64, {220, 16, 13, 15, 13, 15},
             {785, 6722, 315, 9625, 374, 12095}},
            // The same Llama-3.2-1B at group 32 and at group 128. They are the
            // only rows whose group is not 64, and 64 was a constant baked into
            // the codec, into every `instantiate_qmm_t`, and into a
            // `"_gs_64_b_"` literal in each of llama's, gemma4's and qwen3.5's
            // PSO builders until these two existed. A g64 pipeline over a g32
            // checkpoint is not a load failure: before this it produced token
            // 3504, six times over, from scales applied to the wrong weights.
            //
            // They are not published -- mlx-community ships this model at g64 --
            // but they are two commands from what is:
            //
            //   mlx_lm.convert --hf-path <the published 4bit> --mlx-path /tmp/bf16 -d
            //   mlx_lm.convert --hf-path /tmp/bf16 --mlx-path ~/.pie-bench/llama-1b-gs32 \
            //                  -q --q-group-size 32 --q-bits 4
            //
            // and the same with 128. Their continuations are mlx-lm's own on
            // those files, which is the only reference that means anything:
            // requantizing changes the weights, so the g64 answer is not the
            // answer here, and both differ from it in the last token.
            {"Llama-3.2-1B-g32", 16, 0, 4, 2048, 32, {12095, 13, 1115, 374, 279, 1193},
             {785, 6722, 315, 9625, 374, 12095}},
            {"Llama-3.2-1B-g128", 16, 0, 4, 2048, 128, {12095, 13, 1115, 374, 279, 1193},
             {785, 6722, 315, 9625, 374, 12095}},
        };
        //
        // What IS checked, and is the evidence for this family, is the tap
        // comparison under the same re-quantization: no tap collapses, every
        // cosine decays smoothly with depth, and the logits agree to 0.9937.
        // That is the signature of quantization drift and not of a defect --
        // the distinction the checkpoint could not make before, because it did
        // not load at all.
        const Known* ref = nullptr;
        for (const Known& k : known) {
            if (k.n_layers == shape.n_layers && k.n_experts == shape.n_experts &&
                k.hidden == shape.hidden &&
                k.quant_bits == (cfg.quant_bits != 0 ? cfg.quant_bits : 4) &&
                k.quant_group == (cfg.quant_group_size != 0 ? cfg.quant_group_size : 64)) {
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
        if (!gate(p, want, "greedy continuation matches the recorded answer") && !dumping) return 1;

        // The same check again on a prompt long enough to take the BATCHED
        // mixture. Stated as a precondition rather than assumed: the threshold
        // is arithmetic on the geometry, and the day it moves this check would
        // quietly become a third run of the matvec path it was added to stop
        // covering for.
        // It used to be gated on `n_experts > 0`, which meant every dense
        // checkpoint here carried a VERIFIED `want_long` that nothing ever
        // read. The batched mixture is not the only thing 192 rows reach: a
        // twelve-token prompt runs the dense projections as matvecs too, so
        // `affine_qmm_t` -- the whole prefill -- was covered by synthetic
        // numerics and by no checkpoint at all. The precondition below is
        // still checked, but only where it means something.
        if (ref != nullptr && !ref->want_long.empty()) {
            const std::vector<std::uint32_t> lp = long_gate_prompt();
            // The precondition can only be CHECKED for the llama family --
            // `llama_moe_tile_rows` reads a `LlamaGeometry`, and gpt-oss's
            // experts live in a different sub-config with a different
            // threshold. So this asks it where it can be asked, and elsewhere
            // says what it is running rather than claiming more.
            if (cfg.llama.n_experts > 0) {
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
            } else {
                std::printf("  ....  long gate: %zu rows%s\n", lp.size(),
                            shape.n_experts > 0 ? ", mixture threshold unchecked here"
                                                : ", batched projections");
            }
            const char* what = shape.n_experts > 0
                                   ? "batched-mixture continuation matches the recorded answer"
                                   : "batched-prefill continuation matches the recorded answer";
            if (!gate(lp, ref->want_long, what) && !dumping) {
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
            // Relative L2 between two logit rows. Used twice per member, which
            // is the point: an absolute threshold has to know how much two
            // fires of the SAME sequence may legitimately differ, and it does
            // not. A batched member runs a GEMM where a lone one ran a matvec,
            // and at a ten-token prefix of this prompt Llama-3.2-1B's rows
            // come out 0.064 apart -- past the 0.05 line this check used to
            // draw, on a fire with nothing wrong with it.
            const auto rel_l2 = [](const std::vector<float>& a2,
                                   const std::vector<float>& b2) {
                double num = 0.0;
                double den = 0.0;
                for (std::size_t v = 0; v < a2.size(); ++v) {
                    const double d0 = double(a2[v]) - double(b2[v]);
                    num += d0 * d0;
                    den += double(b2[v]) * double(b2[v]);
                }
                return std::sqrt(num / std::max(den, 1e-30));
            };
            for (int i = 0; i < 2; ++i) {
                const std::vector<float> got = row_of(outs[i], 0);
                const auto [want_tok, margin] = top2(*want[i]);
                const auto [got_tok, _] = top2(got);
                float dev = 0.0F;
                for (std::size_t v = 0; v < got.size(); ++v) {
                    dev = std::max(dev, std::fabs(got[v] - (*want[i])[v]));
                }
                const double own = rel_l2(got, *want[i]);
                const double other = rel_l2(got, *want[1 - i]);
                // The assertion, with no constant in it: this member's row is
                // nearer its OWN lone fire than the other member's. That is
                // exactly the thing being hunted -- a fire that crossed its
                // rows, or that answered the longer prompt twice because the
                // shorter is its prefix, lands nearer the wrong reference.
                // Arithmetic drift moves `own` and cannot move it past
                // `other`: the two sit two orders of magnitude apart.
                const bool same_row = own < other;
                // Only THEN is the token worth checking, and only when the
                // row's own top-two margin clears what these two fires
                // actually disagreed by. Below that, the argmax is a coin the
                // test has no business calling: " Paris" and " in" after "The
                // capital of France is" sit two bf16 ulps apart, and the first
                // version of this check reported the coin landing differently
                // as a bug. Gemma-4 reads these ids as nothing in particular
                // and is ambiguous at every prefix length, so this stays.
                const bool decisive = margin > 2.0F * dev;
                const bool agree = got_tok == want_tok;
                const bool bad = !same_row || (decisive && !agree);
                if (bad) good = false;
                std::printf("  %s  one fire, %s member %d: %d vs %d alone "
                            "(margin %.3f, fires differ by %.3f, rel %.5f vs "
                            "%.5f to the other member)%s\n",
                            bad ? "FAIL" : (decisive ? "PASS" : "TIE "), what, i, got_tok,
                            want_tok, double(margin), double(dev), own, other,
                            (same_row && !decisive) ? "  [row gated, token ambiguous]" : "");
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
        const int t = fire(exec, s, 1, page_size, next_page, /*want_token=*/true,
                           /*staged=*/nullptr, /*greedy_token_only=*/true);
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

    // ── throughput ──
    //
    // The figures above are LATENCY: one sequence, one token at a time, the
    // host in the loop. A server is not that shape. `PIE_BENCH_TPUT=n` fires n
    // independent sequences in ONE forward and feeds every member its own
    // greedy token, which is what a batched sampler does and the only form
    // comparable to mlx-lm's batch generation.
    //
    // Every member gets its own sequence id and its own pages out of one
    // counter -- the ring is one pool and nothing here frees -- and its own
    // recurrent slot, because a hybrid family's linear-attention state does not
    // live in the KV pages and two members sharing a slot compute each other's
    // history.
    double tput_tps = 0.0;
    if (n_seqs > 1) {
        const std::size_t nf = std::size_t(n_seqs);
        std::vector<Seq> fleet(nf);
        std::uint32_t fpage = 0;
        bool bad = false;
        for (int i = 0; i < n_seqs && !bad; ++i) {
            fleet[std::size_t(i)].id = std::uint32_t(100 + i);
            fleet[std::size_t(i)].tokens = prompt;
            fleet[std::size_t(i)].tokens.resize(std::size_t(n_prompt + n_decode), 1u);
            fleet[std::size_t(i)].rs_slot = exec.rs_slots() > 0 ? i % exec.rs_slots() : 0;
            // Prefilled one at a time and UNTIMED. What is being measured is
            // the decode fleet; folding a prefill into it would report one
            // number for two regimes.
            if (fire(exec, fleet[std::size_t(i)], std::uint32_t(n_prompt), page_size, fpage) < 0) {
                bad = true;
            }
        }
        if (!bad) {
            std::vector<int> fleet_diverged(nf, -1);
            const double t3 = now_s();
            for (int step = 0; step < n_decode && !bad; ++step) {
                std::vector<MemberForwardDesc> descs;
                descs.reserve(nf);
                for (int i = 0; i < n_seqs; ++i) {
                    descs.push_back(desc_for(fleet[std::size_t(i)], 1, page_size, fpage,
                                             exec.rs_slots(), /*greedy_token_only=*/true));
                }
                std::vector<LogitsOut> outs(nf);
                int first_tok = -1;
                std::vector<std::uint8_t> ok(nf, 0);
                std::vector<std::string> errs(nf);
                exec.forward_batch(descs, outs, ok, errs);
                for (int i = 0; i < n_seqs; ++i) {
                    if (ok[std::size_t(i)] == 0) {
                        std::printf("  FAIL  throughput fire: %s\n", errs[std::size_t(i)].c_str());
                        bad = true;
                        break;
                    }
                    Seq& m = fleet[std::size_t(i)];
                    const int t = argmax_of(outs[std::size_t(i)], 0);
                    // Members are identical sequences in ONE fire through ONE
                    // kernel, so they are not merely close -- they are the same
                    // arithmetic and must agree exactly. This catches a fire
                    // that mixes rows up between members, which comparing each
                    // member to a differently-shaped reference cannot.
                    if (i == 0) {
                        first_tok = t;
                    } else if (t != first_tok) {
                        std::printf("  FAIL  members of one fire disagree at step %d: "
                                    "member %d says %d, member 0 says %d\n",
                                    step, i, t, first_tok);
                        bad = true;
                        break;
                    }
                    // Every member was given the SAME prompt, so the fleet is n
                    // copies of the sequence the latency loop above already
                    // decoded alone. That makes the fleet's tokens checkable
                    // for free against a reference this file already holds --
                    // and a throughput number off a wrong forward is not a
                    // throughput number. The batched GEMM is a different kernel
                    // than the batch-1 GEMV so the two are not bit-identical
                    // and a late greedy flip on near-tied logits is fair; a
                    // member that leaves the reference in the first few steps
                    // is not.
                    const std::size_t ref = std::size_t(n_prompt + 1 + step);
                    if (ref < s.tokens.size() && fleet_diverged[std::size_t(i)] < 0 &&
                        std::uint32_t(t) != s.tokens[ref]) {
                        fleet_diverged[std::size_t(i)] = step;
                    }
                    m.next_position += 1;
                    if (m.next_position < m.tokens.size()) m.tokens[m.next_position] = std::uint32_t(t);
                }
            }
            if (!bad) {
                const double tput_s = now_s() - t3;
                tput_tps = double(n_decode) * double(n_seqs) / tput_s;
                int worst = n_decode;
                for (int i = 0; i < n_seqs; ++i) {
                    const int d = fleet_diverged[std::size_t(i)];
                    if (d >= 0 && d < worst) worst = d;
                }
                // A fleet member that never matches the reference is a broken
                // forward, not a rounding difference.
                if (worst == 0) {
                    std::printf("  FAIL  the fleet decodes a different token than the "
                                "same prompt does alone, from its very first step\n");
                    bad = true;
                } else {
                    std::printf("  PASS  fleet agrees with the single-sequence decode for "
                                "%d of %d steps\n", worst, n_decode);
                }
            }
        }
        if (bad) return 1;
    }

    const double prefill_tps = double(n_prompt) / prefill_s;
    const double decode_tps = double(n_decode) / decode_s;
    std::printf("  prefill: %d tok in %.4f s  =  %.1f tok/s\n", n_prompt, prefill_s, prefill_tps);
    std::printf("  decode : %d tok in %.4f s  =  %.1f tok/s  (%.2f ms/tok)  [token fed back]\n",
                n_decode, decode_s, decode_tps, 1000.0 * decode_s / double(n_decode));
    std::printf("  decode : %d tok in %.4f s  =  %.1f tok/s  (%.2f ms/tok)  [no readback: the "
                "scheduler's ceiling, NOT comparable]\n",
                n_decode, pipelined_s, double(n_decode) / pipelined_s,
                1000.0 * pipelined_s / double(n_decode));
    if (tput_tps > 0.0) {
        std::printf("  tput   : %d seqs x %d tok  =  %.1f tok/s  (%.1f tok/s per seq)\n", n_seqs,
                    n_decode, tput_tps, tput_tps / double(n_seqs));
    }

    // Decode at batch one reads (almost) every weight once per token, so
    // tokens/s times bytes-read is the bandwidth actually achieved. Stating it
    // this way is what makes the number comparable across machines -- and it is
    // the only honest way to say whether a decode kernel is "fast", since the
    // ceiling is the bus and not the ALU.
    //
    // Both numbers are MEASURED from the staged tensors. This used to be a
    // formula over `cfg.llama.*`, which is the exact defect `BenchShape` above
    // was written to fix: gemma4 and gpt-oss fill a different sub-config, so it
    // read zeros and printed `weights: 0.00 GiB -> 0.0 GB/s` under a run whose
    // whole purpose is to say whether the decode is bus-bound. Repairing it per
    // family would have been worse than pointless -- qwen3.5's stack is mostly
    // linear attention rather than four projections, and gpt-oss's expert bank
    // is 4 bits under a byte per 32 rather than a bf16 pair per 64 -- and all
    // of it is already a byte count in the heap.
    //
    // Resident and read-per-token are printed separately because for a mixture
    // they are different questions: Qwen3-30B-A3B holds thirty billion weights
    // and touches about three billion of them a token, and one number cannot
    // answer both.
    const double gib = 1024.0 * 1024.0 * 1024.0;
    const auto wb = exec.weight_bytes();
    std::printf("  weights: %.2f GiB resident, %.2f GiB read per token"
                "  ->  decode moves %.1f GB/s\n",
                double(wb.resident) / gib, wb.per_token / gib, wb.per_token * decode_tps / 1e9);
    // A reported bandwidth nobody checks is how this line came to print `0.00
    // GiB` for two families for as long as it did. These are the three things
    // that were violated, stated so a fourth family cannot repeat them:
    if (wb.resident == 0) {
        std::printf("  FAIL  the heap holds no weights, so the bandwidth above is not a "
                    "measurement of anything\n");
        return 1;
    }
    if (wb.per_token > double(wb.resident)) {
        std::printf("  FAIL  a token reads %.2f GiB of a %.2f GiB heap\n", wb.per_token / gib,
                    double(wb.resident) / gib);
        return 1;
    }
    // A mixture must leave out the experts it did not choose, and how many
    // that is comes from the CHECKPOINT's config here, while everything above
    // comes from the heap the driver staged. That independence is the point:
    // the driver agreeing with itself proves nothing, and taking the routing
    // fraction from the wrong family's geometry -- which read as zero -- once
    // claimed 824 GB/s on a bus that does 400.
    if (shape.n_experts > 0 && shape.experts_per_token > 0) {
        // The routed bank is what the discount is TAKEN FROM, so a mixture
        // whose heap counted no routed bytes has not been discounted at all --
        // and the inactive-bytes check below cannot see it, because both of
        // its sides come from the same match. That is exactly how gemma4's
        // 26B came to claim 806 GB/s: its experts are staged as
        // `layers.N.experts.` and the rule matched `mlp.experts.`, so
        // `routed_resident` was zero, `inactive` was zero, and zero unread
        // bytes was duly found to be at least zero. This asks the question the
        // other way round -- the config says there IS a bank, so the heap has
        // to have found one.
        if (shape.experts_per_token < shape.n_experts && wb.routed_resident == 0) {
            std::printf("  FAIL  the config declares %d experts top-%d, but no staged weight "
                        "was counted as routed, so the %.1f GB/s above is the whole heap\n",
                        shape.n_experts, shape.experts_per_token, wb.per_token * decode_tps / 1e9);
            return 1;
        }
        const double inactive =
            double(wb.routed_resident) *
            (1.0 - double(shape.experts_per_token) / double(shape.n_experts));
        if (double(wb.resident) - wb.per_token < inactive * 0.99) {
            std::printf("  FAIL  %d experts top-%d over a %.2f GiB bank leaves %.2f GiB "
                        "unread, but a token was counted as skipping only %.2f GiB\n",
                        shape.n_experts, shape.experts_per_token,
                        double(wb.routed_resident) / gib, inactive / gib,
                        (double(wb.resident) - wb.per_token) / gib);
            return 1;
        }
    }

    return 0;
}
