// Does `MetalExecutor` actually forward gemma4 and gpt-oss?
//
// The forward tests drive the raw path -- `RawMetalContext`, the family's DAG,
// `run_step` -- and prove the MODEL computes. They say nothing about the
// executor, which is what `pie serve` calls: the linear-sequence contract, the
// logits staging, the readout rows. This drives that surface directly, so a
// rejection is a string here rather than a poisoned channel three layers up.
//
// Skipped (not failed) when the checkpoint is absent.

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#include "batch/forward.hpp"
#include "batch/golden_tap.hpp"

using namespace pie::metal;
using namespace pie::metal::batch;

namespace {

int failures = 0;

void expect(bool ok, const std::string& what) {
    std::printf("  %s  %s\n", ok ? "PASS" : "FAIL", what.c_str());
    if (!ok) ++failures;
}

bool exists(const std::string& dir) {
    const std::string probe = dir + "/config.json";
    FILE* f = std::fopen(probe.c_str(), "rb");
    if (f == nullptr) return false;
    std::fclose(f);
    return true;
}

/// The argmax of a staged logits row, read the way the sampler reads it: a
/// bf16 view of PTIR staging, not `LogitsOut::data` (which production leaves
/// empty on purpose).
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
        const float v = f32(i);
        if (v > bv) {
            bv = v;
            best = int(i);
        }
    }
    return best;
}

/// A sequence the harness is driving: its tokens, and the pages it owns.
struct Seq {
    std::uint64_t id = 0;
    std::vector<std::uint32_t> tokens;
    std::vector<std::uint32_t> pages;
    std::uint32_t next_position = 0;
    std::vector<std::uint32_t> sampled;
};

/// The descriptor for this sequence's next fire: `n` new tokens starting at
/// `next_position`, reading only the last row.
///
/// The page list is the whole history's, flashinfer's CSR convention, and it
/// grows as the sequence does -- which is what the runtime does for real.
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

/// One sequence: the whole prompt in ONE fire, then greedy decode.
bool run_family(const std::string& tag, const SetupConfig& cfg,
                const std::vector<std::uint32_t>& prompt, int n_gen,
                std::vector<std::uint32_t>& got) {
    MetalExecutor exec;
    std::string err;
    if (!exec.setup(cfg, &err)) {
        std::printf("  FAIL  %s setup: %s\n", tag.c_str(), err.c_str());
        ++failures;
        return false;
    }
    expect(exec.ready(), tag + ": the executor reports ready");
    expect(exec.vocab() > 0, tag + ": and a vocabulary");
    const std::uint32_t page_size = exec.kv_pool_page_size();
    std::uint32_t next_free_page = 0;

    Seq s;
    s.id = 1;
    s.tokens = prompt;
    for (int step = 0; step <= n_gen; ++step) {
        const std::uint32_t n = std::uint32_t(s.tokens.size()) - s.next_position;
        if (n == 0) break;
        MemberForwardDesc d = desc_for(s, n, page_size, next_free_page);
        LogitsOut out;
        std::string ferr;
        if (!exec.forward(d, out, &ferr)) {
            std::printf("  FAIL  %s forward: %s\n", tag.c_str(), ferr.c_str());
            ++failures;
            return false;
        }
        if (out.rows != 1 || out.device_contents == nullptr) {
            std::printf("  FAIL  %s forward produced no logits row\n", tag.c_str());
            ++failures;
            return false;
        }
        s.next_position += n;
        const int best = argmax_of(out, 0);
        if (step < n_gen) {
            got.push_back(std::uint32_t(best));
            s.tokens.push_back(std::uint32_t(best));
        }
    }
    return true;
}

/// Two sequences, resident at once, sharing every fire.
///
/// This is the property the whole paged path exists for. Sequence B's prompt is
/// a PREFIX of sequence A's, so a fire that leaked pages or positions between
/// them would give one answer twice; they must each get their own.
///
/// Fire 1 prefills both (a 12-row mixed fire). Fire 2 decodes both (a 2-row
/// one). So prefill and decode rows share a command buffer in the first fire
/// and the batch shrinks in the second, which is what a real frame does.
bool run_two_sequences(const SetupConfig& cfg, const std::vector<std::uint32_t>& a_prompt,
                       const std::vector<std::uint32_t>& b_prompt,
                       std::vector<std::uint32_t>& a_out, std::vector<std::uint32_t>& b_out) {
    MetalExecutor exec;
    std::string err;
    if (!exec.setup(cfg, &err)) {
        std::printf("  FAIL  two-sequence setup: %s\n", err.c_str());
        ++failures;
        return false;
    }
    const std::uint32_t page_size = exec.kv_pool_page_size();
    std::uint32_t next_free_page = 0;
    Seq a, b;
    a.id = 11;
    a.tokens = a_prompt;
    b.id = 22;
    b.tokens = b_prompt;

    for (int fire = 0; fire < 2; ++fire) {
        std::vector<MemberForwardDesc> descs;
        for (Seq* s : {&a, &b}) {
            const std::uint32_t n = std::uint32_t(s->tokens.size()) - s->next_position;
            descs.push_back(desc_for(*s, n, page_size, next_free_page));
        }
        std::vector<LogitsOut> outs(descs.size());
        std::vector<std::uint8_t> ok(descs.size(), 0);
        std::vector<std::string> errs(descs.size());
        exec.forward_batch(descs, outs, ok, errs);
        int i = 0;
        for (Seq* s : {&a, &b}) {
            if (ok[i] == 0) {
                std::printf("  FAIL  two-sequence fire %d member %d: %s\n", fire, i,
                            errs[i].c_str());
                ++failures;
                return false;
            }
            s->next_position += std::uint32_t(descs[i].token_ids.size());
            const int best = argmax_of(outs[i], 0);
            s->sampled.push_back(std::uint32_t(best));
            s->tokens.push_back(std::uint32_t(best));
            ++i;
        }
    }
    a_out = a.sampled;
    b_out = b.sampled;
    return true;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("simple_family_executor_test\n");
    const char* home = std::getenv("HOME");
    const std::string root = home != nullptr ? std::string(home) + "/.pie-bench" : ".";
    const std::string kernels = PIE_METAL_KERNELS_DIR_FOR_TEST;
    // A tap dump holds the LAST step's activations, so under one it is worth
    // running exactly ONE step: <bos> at position 0, the case the raw path
    // pins. Everything after it would silently overwrite the dump.
    const bool taps = std::getenv("PIE_METAL_GOLDEN_DIR") != nullptr;
    // The two families' taps share names ("0.attn_norm" is both), so a dump run
    // does ONE of them. `PIE_SFX_TAPS_FAMILY` picks; gemma4 by default.
    const char* taps_family_env = std::getenv("PIE_SFX_TAPS_FAMILY");
    const std::string taps_family = taps_family_env != nullptr ? taps_family_env : "gemma4";
    const bool g4_taps = taps && taps_family == "gemma4";
    const bool go_taps = taps && taps_family == "gptoss";
    // Streaming is a config setting on the driver, not an environment the
    // driver reads -- so the pairing lives here, in the harness, and is passed
    // in like any other operator choice. ctest runs this binary twice, once
    // with this set, and the assertions below are the same assertions: if
    // streaming ever changes an answer, the test that says so is the one that
    // already knows every answer.
    const char* stream_env = std::getenv("PIE_METAL_TEST_STREAM_EXPERTS");
    const bool stream_experts =
        stream_env != nullptr && *stream_env != '\0' && std::string(stream_env) != "0";
    std::printf("  streaming: %s\n", stream_experts ? "on" : "off");

    // ── gemma4 ──
    if (!go_taps) {
        const std::string dir = root + "/gemma4-e2b-pie";
        if (!exists(dir)) {
            std::printf("  gemma4: SKIP (no checkpoint at %s)\n", dir.c_str());
        } else {
            SetupConfig cfg;
            cfg.kernels_dir = kernels;
            cfg.snapshot_dir = dir;
            cfg.stream_routed_experts = stream_experts;
            cfg.model_type = "gemma4";
            cfg.vocab_size = 262144;
            // A fire now carries a whole prompt, and two of them at once.
            cfg.max_forward_tokens = 32;
            cfg.max_forward_requests = 4;
            cfg.kv_page_size = 32;
            cfg.gemma4.n_layers = 35;
            cfg.gemma4.hidden = 1536;
            cfg.gemma4.intermediate = 6144;
            cfg.gemma4.n_q_heads = 8;
            cfg.gemma4.n_kv_heads = 1;
            cfg.gemma4.head_dim = 256;
            cfg.gemma4.global_head_dim = 512;
            cfg.gemma4.sliding_window = 512;
            cfg.gemma4.num_kv_shared_layers = 20;
            cfg.gemma4.per_layer_emb_dim = 256;
            cfg.gemma4.full_attn_interval = 5;
            cfg.gemma4.double_wide_mlp = true;
            cfg.gemma4.final_softcap = 30.0f;
            // The sharpest case first: one <bos> at position 0, whose answer
            // the raw path pins at 236761. It isolates the engine from the
            // replay -- one fire, one row, no KV history.
            std::vector<std::uint32_t> one;
            // Under a tap dump the LAST fire is what lands there, so <bos> is
            // fired ONCE -- continuing would dump the step after the one the
            // reference describes.
            if ((!taps || g4_taps) && run_family("gemma4/bos", cfg, {2}, taps ? 0 : 1, one)) {
                if (taps) {
                    std::printf("    (gemma4 <bos> taps -> %s)\n", golden_tap_dir().c_str());
                } else {
                    std::printf("    gemma4 <bos> -> %u\n", one[0]);
                    expect(one[0] == 236761, "gemma4's <bos> logits match the raw path");
                }
            }
            std::vector<std::uint32_t> got;
            // <bos>, then the eight-token prompt the forward test teacher-forces.
            if (!taps && run_family("gemma4", cfg, {2, 818, 3821, 563, 529, 476, 3625, 506}, 2, got)) {
                std::printf("    gemma4 greedy %u %u\n", got[0], got[1]);
                // mlx-lm's answer for this prompt, which `gemma4_forward_test`
                // pins on the raw path. Getting it through the EXECUTOR is the
                // point: the same numbers, one layer up.
                expect(!got.empty() && got[0] == 3821,
                       "gemma4's first sampled token is mlx-lm's, through the executor");
            }

            // ── Two sequences, resident at once ──
            //
            // The eight-token prompt and its own four-token prefix, prefilled
            // in ONE 12-row fire and then decoded together in a 2-row one.
            // `gemma4_prefill_numerics_test` pins those two answers on the raw
            // path (3821 and 496); getting them from a batch, through the
            // executor, is what "serves more than one sequence" means.
            std::vector<std::uint32_t> ta, tb;
            if (!taps && run_two_sequences(cfg, {2, 818, 3821, 563, 529, 476, 3625, 506},
                                           {2, 818, 3821, 563}, ta, tb)) {
                std::printf("    two sequences: A %u %u   B %u %u\n", ta[0], ta[1], tb[0],
                            tb[1]);
                expect(ta[0] == 3821,
                       "a sequence's answer is unchanged by a sibling sharing its fire");
                expect(tb[0] == 496,
                       "and the shorter one gets its OWN answer, from its own pages and "
                       "positions");
            }
        }
    }

    // ── gpt-oss ──
    if (!taps || go_taps) {
        const std::string dir = root + "/gptoss-20b-pie4";
        if (!exists(dir)) {
            std::printf("  gpt-oss: SKIP (no checkpoint at %s)\n", dir.c_str());
        } else {
            SetupConfig cfg;
            cfg.kernels_dir = kernels;
            cfg.snapshot_dir = dir;
            cfg.stream_routed_experts = stream_experts;
            cfg.model_type = "gpt_oss";
            cfg.vocab_size = 201088;
            cfg.max_forward_tokens = 32;
            cfg.max_forward_requests = 4;
            cfg.kv_page_size = 32;
            cfg.gptoss.n_layers = 24;
            cfg.gptoss.hidden = 2880;
            cfg.gptoss.vocab = 201088;
            cfg.gptoss.n_q_heads = 64;
            cfg.gptoss.n_kv_heads = 8;
            cfg.gptoss.head_dim = 64;
            cfg.gptoss.sliding_window = 128;
            cfg.gptoss.n_experts = 32;
            cfg.gptoss.experts_per_token = 4;
            cfg.gptoss.intermediate = 2880;
            std::vector<std::uint32_t> got;
            // "The capital of France is Paris. The capital of Japan is"
            // Under a tap dump the LAST step is what lands there, so only the
            // prompt runs: one fire, twelve rows, every one of them routed
            // independently.
            if (run_family("gpt-oss", cfg,
                           {976, 9029, 328, 10128, 382, 12650, 13, 623, 9029, 328, 10198, 382},
                           taps ? 0 : 3, got)) {
                if (taps) {
                    std::printf("    (gpt-oss taps -> %s)\n", golden_tap_dir().c_str());
                }
                if (got.size() < 3) return failures == 0 ? 0 : 1;
                std::printf("    gpt-oss greedy %u %u %u\n", got[0], got[1], got[2]);
                // " Tokyo. The" -- mlx-lm's, and the part the prompt forces.
                const std::vector<std::uint32_t> want{40510, 13, 623};
                bool same = got.size() == want.size();
                for (std::size_t i = 0; same && i < want.size(); ++i) same = got[i] == want[i];
                expect(same, "gpt-oss's sampled tokens are mlx-lm's, through the executor");
            }

            // ── Two sequences, resident at once ──
            //
            // gpt-oss now sorts the two members' routed rows into one wider
            // expert-major pass. Paging is what makes their KV histories safe:
            // before it, the cache was one ring and the second sequence
            // overwrote the first.
            //
            // B's prompt is a PREFIX of A's, so a leak between them would give
            // one answer twice. A must still answer 40510 ("Tokyo") beside a
            // sibling.
            std::vector<std::uint32_t> ta, tb;
            if (!taps && !go_taps &&
                run_two_sequences(
                    cfg, {976, 9029, 328, 10128, 382, 12650, 13, 623, 9029, 328, 10198, 382},
                    {976, 9029, 328, 10128, 382}, ta, tb)) {
                std::printf("    two sequences: A %u %u   B %u %u\n", ta[0], ta[1], tb[0],
                            tb[1]);
                expect(ta[0] == 40510,
                       "gpt-oss: a sequence's answer is unchanged by a sibling sharing its "
                       "fire");
                // "The capital of France is" -> " Paris". Pinned rather than
                // merely "different from A": a 17-row fire takes the wide GEMM
                // block, and when that pipeline was mis-selected B's last row
                // was never computed -- which reads as a different token, so a
                // difference test passed on output nothing had written.
                expect(tb[0] == 12650,
                       "gpt-oss: and the shorter one is answered from its OWN pages, not its "
                       "neighbour's");
            }
        }
    }

    std::printf("\n==== simple_family_executor_test: %s ====\n",
                failures == 0 ? "all passed" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
