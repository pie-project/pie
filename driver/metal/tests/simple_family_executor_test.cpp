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

/// One sequence, greedily continued, through the executor's own API.
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

    std::vector<std::uint32_t> tokens = prompt;
    for (int s = 0; s < n_gen; ++s) {
        MemberForwardDesc desc;
        desc.sequence_id = 1;
        desc.token_ids = tokens;
        desc.position_ids.clear();
        for (std::size_t i = 0; i < tokens.size(); ++i) {
            desc.position_ids.push_back(static_cast<std::uint32_t>(i));
        }
        // Only the last row is read: this is a prompt replay, and the sampler
        // wants the token that follows it.
        desc.readout_local_indices = {static_cast<std::uint32_t>(tokens.size() - 1)};

        LogitsOut out;
        std::string ferr;
        if (!exec.forward(desc, out, &ferr)) {
            std::printf("  FAIL  %s forward: %s\n", tag.c_str(), ferr.c_str());
            ++failures;
            return false;
        }
        if (out.rows != 1 || out.device_contents == nullptr) {
            std::printf("  FAIL  %s forward produced no logits row\n", tag.c_str());
            ++failures;
            return false;
        }
        // The production readout: a bf16 view of the staged row, which is what
        // the sampler binds. `LogitsOut::data` is the test-only f32 copy and
        // stays empty here on purpose -- reading it would test a path the engine
        // does not take.
        const auto* bf = static_cast<const std::uint16_t*>(out.device_contents) +
                         std::size_t(out.device_row_offset) * std::size_t(out.vocab);
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
        got.push_back(std::uint32_t(best));
        tokens.push_back(std::uint32_t(best));
    }
    return true;
}

}  // namespace

int main() {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::printf("simple_family_executor_test\n");
    const char* home = std::getenv("HOME");
    const std::string root = home != nullptr ? std::string(home) + "/.pie-bench" : ".";
    const std::string kernels = PIE_METAL_KERNELS_DIR_FOR_TEST;

    // ── gemma4 ──
    {
        const std::string dir = root + "/gemma4-e2b-pie";
        if (!exists(dir)) {
            std::printf("  gemma4: SKIP (no checkpoint at %s)\n", dir.c_str());
        } else {
            SetupConfig cfg;
            cfg.kernels_dir = kernels;
            cfg.snapshot_dir = dir;
            cfg.model_type = "gemma4";
            cfg.vocab_size = 262144;
            cfg.max_forward_tokens = 1;
            cfg.max_forward_requests = 1;
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
            std::vector<std::uint32_t> got;
            // <bos>, then the eight-token prompt the forward test teacher-forces.
            if (run_family("gemma4", cfg, {2, 818, 3821, 563, 529, 476, 3625, 506}, 2, got)) {
                std::printf("    gemma4 greedy %u %u\n", got[0], got[1]);
                // mlx-lm's answer for this prompt, which `gemma4_forward_test`
                // pins on the raw path. Getting it through the EXECUTOR is the
                // point: the same numbers, one layer up.
                expect(!got.empty() && got[0] == 3821,
                       "gemma4's first sampled token is mlx-lm's, through the executor");
            }
        }
    }

    // ── gpt-oss ──
    {
        const std::string dir = root + "/gptoss-20b-pie4";
        if (!exists(dir)) {
            std::printf("  gpt-oss: SKIP (no checkpoint at %s)\n", dir.c_str());
        } else {
            SetupConfig cfg;
            cfg.kernels_dir = kernels;
            cfg.snapshot_dir = dir;
            cfg.model_type = "gpt_oss";
            cfg.vocab_size = 201088;
            cfg.max_forward_tokens = 1;
            cfg.max_forward_requests = 1;
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
            if (run_family("gpt-oss", cfg,
                           {976, 9029, 328, 10128, 382, 12650, 13, 623, 9029, 328, 10198, 382},
                           3, got)) {
                std::printf("    gpt-oss greedy %u %u %u\n", got[0], got[1], got[2]);
                // " Tokyo. The" -- mlx-lm's, and the part the prompt forces.
                const std::vector<std::uint32_t> want{40510, 13, 623};
                bool same = got.size() == want.size();
                for (std::size_t i = 0; same && i < want.size(); ++i) same = got[i] == want[i];
                expect(same, "gpt-oss's sampled tokens are mlx-lm's, through the executor");
            }
        }
    }

    std::printf("\n==== simple_family_executor_test: %s ====\n",
                failures == 0 ? "all passed" : "FAILURES");
    return failures == 0 ? 0 : 1;
}
