#pragma once

// Where a test's checkpoint is, given the two places it can be.
//
// `~/.pie-bench/<name>` is the converted copy these tests were written against.
// On a machine that has never run the conversion it does not exist, and four
// tests then print SKIP and exit 0 -- which ctest reports as a pass. That is
// how `gemma4_forward_test` and `gemma4_prefill_numerics_test`, the only two
// checks this driver has of gemma4's arithmetic against mlx-lm, came to be
// silent on every machine but one.
//
// The same weights are usually already on disk under the Hugging Face cache,
// because that is what `llama_bench`'s gates load. Looking there second costs
// nothing and turns those tests back on.
//
// Only for checkpoints PROVEN interchangeable. gpt-oss deliberately has no
// fallback: `~/.pie-bench/gptoss-20b-pie4` is a different staging of the
// weights, and the mlx-community MXFP4 release throws `unstaged weight
// layers.0.mlp.experts.gate_proj.biases` on the same code path.

#include <cstdlib>
#include <filesystem>
#include <string>

namespace pie::metal::test {

/// The first of the two paths that holds a `config.json`, or empty for neither.
///
/// `hf_repo` is the cache directory's name, which is the repo id with every
/// `/` turned into `--`, e.g. `models--mlx-community--gemma-4-e2b-it-4bit`.
inline std::string find_checkpoint(const std::string& local, const std::string& hf_repo) {
    namespace fs = std::filesystem;
    const char* home = std::getenv("HOME");
    if (home == nullptr) return {};
    std::error_code ec;

    const std::string bench = std::string(home) + "/.pie-bench/" + local;
    if (fs::exists(bench + "/config.json", ec)) return bench;

    // Snapshots are named by commit; any of them is the checkpoint, and there
    // is normally exactly one.
    const fs::path snaps =
        fs::path(home) / ".cache/huggingface/hub" / hf_repo / "snapshots";
    if (!fs::is_directory(snaps, ec)) return {};
    for (const auto& entry : fs::directory_iterator(snaps, ec)) {
        if (fs::exists(entry.path() / "config.json", ec)) return entry.path().string();
    }
    return {};
}

}  // namespace pie::metal::test
