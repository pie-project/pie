#pragma once

// Filesystem helpers for locating HuggingFace cache snapshots. Pulled out
// of entry.cpp so the driver's startup code can stay focused on launching
// the engine rather than path detective work.

#include <filesystem>
#include <optional>

namespace pie_cuda_driver {

// True if `path` looks like an unpacked HF snapshot — i.e. contains a
// config.json at the top level.
bool looks_like_hf_snapshot(const std::filesystem::path& path);

// Walk an HF cache repo directory (the `models--org--name` layout) and
// return its currently-checked-out snapshot if one is uniquely
// resolvable. Prefers refs/main, falls back to the single eligible
// snapshot under `snapshots/`.
std::optional<std::filesystem::path> resolve_hf_cache_snapshot(
    const std::filesystem::path& repo_dir);

// Given a Gemma4 target snapshot, locate the sibling `-assistant`
// snapshot (the MTP head). Returns nullopt when no assistant repo exists.
std::optional<std::filesystem::path> discover_gemma4_mtp_snapshot_dir(
    const std::filesystem::path& target_snapshot_dir);

}  // namespace pie_cuda_driver
