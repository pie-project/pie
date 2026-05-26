#include "hf_snapshot.hpp"

#include <fstream>
#include <string>

namespace pie_cuda_driver {

namespace {

std::string trim_ascii(std::string s) {
    while (!s.empty() &&
           (s.back() == '\n' || s.back() == '\r' || s.back() == ' ' ||
            s.back() == '\t')) {
        s.pop_back();
    }
    std::size_t start = 0;
    while (start < s.size() &&
           (s[start] == ' ' || s[start] == '\t' || s[start] == '\n' ||
            s[start] == '\r')) {
        ++start;
    }
    if (start > 0) s.erase(0, start);
    return s;
}

}  // namespace

bool looks_like_hf_snapshot(const std::filesystem::path& path) {
    return std::filesystem::exists(path / "config.json");
}

std::optional<std::filesystem::path> resolve_hf_cache_snapshot(
    const std::filesystem::path& repo_dir) {
    const auto snapshots_dir = repo_dir / "snapshots";
    if (!std::filesystem::is_directory(snapshots_dir)) return std::nullopt;

    const auto main_ref = repo_dir / "refs" / "main";
    if (std::filesystem::is_regular_file(main_ref)) {
        std::ifstream in(main_ref);
        std::string sha;
        std::getline(in, sha);
        sha = trim_ascii(sha);
        if (!sha.empty()) {
            const auto candidate = snapshots_dir / sha;
            if (looks_like_hf_snapshot(candidate)) return candidate;
        }
    }

    std::optional<std::filesystem::path> only_snapshot;
    int count = 0;
    for (const auto& entry : std::filesystem::directory_iterator(snapshots_dir)) {
        if (!entry.is_directory()) continue;
        if (!looks_like_hf_snapshot(entry.path())) continue;
        only_snapshot = entry.path();
        ++count;
        if (count > 1) return std::nullopt;
    }
    return only_snapshot;
}

std::optional<std::filesystem::path> discover_gemma4_mtp_snapshot_dir(
    const std::filesystem::path& target_snapshot_dir) {
    const auto direct = std::filesystem::path(
        target_snapshot_dir.string() + "-assistant");
    if (looks_like_hf_snapshot(direct)) return direct;

    for (auto cur = target_snapshot_dir;
         !cur.empty() && cur != cur.parent_path();
         cur = cur.parent_path()) {
        const std::string name = cur.filename().string();
        if (name.rfind("models--", 0) != 0) continue;
        const auto assistant_repo =
            cur.parent_path() / (name + "-assistant");
        if (auto snapshot = resolve_hf_cache_snapshot(assistant_repo)) {
            return snapshot;
        }
        break;
    }
    return std::nullopt;
}

}  // namespace pie_cuda_driver
