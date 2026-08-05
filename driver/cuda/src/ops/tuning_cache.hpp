#pragma once

// A tiny on-disk memo for kernel autotuning results.
//
// Picking a kernel by measuring it is only affordable once. A sweep costs a
// few hundred milliseconds per problem shape, and a server meets roughly one
// shape per CUDA-graph bucket, so an untuned start burns seconds and skews the
// first requests it serves. This keeps the answers between runs.
//
// What gets stored is deliberately opaque: two integers, whose meaning is
// entirely up to the caller (they are typically indices into a vendor
// library's candidate list). That makes the file worthless -- actively
// dangerous, even -- if the GPU or the candidate list changes, so the caller
// supplies a signature describing both, it is written as the first line, and a
// file that does not match is discarded rather than replayed.

#include <sys/stat.h>

#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <mutex>
#include <string>
#include <unordered_map>

#include "config.hpp"


namespace pie_cuda_driver::ops {

// Mixes `v` into hash `h`. Callers fold each dimension of a problem shape
// through this to get a cache key.
inline std::uint64_t tuning_hash(std::uint64_t h, std::uint64_t v) {
    h ^= v + 0x9e3779b97f4a7c15ULL + (h << 6) + (h >> 2);
    return h;
}

class TuningCache {
   public:
    // `name` is the file's basename; `signature` describes the hardware and
    // candidate list the entries were measured against. An empty signature
    // disables the cache, which is how callers report that they could not
    // identify the device.
    TuningCache(const char* name, std::string signature)
        : signature_(std::move(signature)), path_(resolve_path(name)) {
        if (!signature_.empty() && !path_.empty()) load();
    }

    bool lookup(std::uint64_t key, int* a, int* b) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto it = entries_.find(key);
        if (it == entries_.end()) return false;
        *a = it->second.first;
        *b = it->second.second;
        return true;
    }

    void store(std::uint64_t key, int a, int b) {
        std::lock_guard<std::mutex> lock(mutex_);
        entries_[key] = {a, b};
        if (path_.empty() || signature_.empty()) return;
        mkdir_parents(path_);
        // Appending rather than rewriting keeps concurrent ranks from
        // truncating each other's entries. A key written twice is harmless:
        // the last line read wins, and both lines named a measured winner.
        FILE* f = std::fopen(path_.c_str(), "a");
        if (f == nullptr) return;
        if (std::ftell(f) == 0) std::fprintf(f, "%s\n", signature_.c_str());
        std::fprintf(f, "%016llx %d %d\n",
                     static_cast<unsigned long long>(key), a, b);
        std::fclose(f);
    }

    std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return entries_.size();
    }

    const std::string& path() const { return path_; }

   private:
    static std::string resolve_path(const char* name) {
        // Convention, not configuration -- see module_cache.hpp.
        const std::string& root = pie_cuda_driver::cache_dir();
        if (!root.empty()) {
            return root + "/" + name;
        }
        const char* xdg = std::getenv("XDG_CACHE_HOME");
        if (xdg != nullptr && xdg[0] != '\0') {
            return std::string(xdg) + "/pie/" + name;
        }
        const char* home = std::getenv("HOME");
        if (home == nullptr || home[0] == '\0') return {};
        return std::string(home) + "/.cache/pie/" + name;
    }

    static void mkdir_parents(const std::string& path) {
        for (std::size_t i = 1; i < path.size(); ++i) {
            if (path[i] != '/') continue;
            ::mkdir(path.substr(0, i).c_str(), 0755);
        }
    }

    void load() {
        FILE* f = std::fopen(path_.c_str(), "r");
        if (f == nullptr) return;

        char line[512];
        bool matches = false;
        if (std::fgets(line, sizeof(line), f) != nullptr) {
            std::string got(line);
            while (!got.empty() && (got.back() == '\n' || got.back() == '\r')) {
                got.pop_back();
            }
            matches = (got == signature_);
        }
        if (matches) {
            unsigned long long key = 0;
            int a = 0;
            int b = 0;
            while (std::fscanf(f, "%llx %d %d", &key, &a, &b) == 3) {
                entries_[static_cast<std::uint64_t>(key)] = {a, b};
            }
        }
        std::fclose(f);
        // Entries measured against a different GPU or candidate list do not
        // name the kernels we would run today, so the file is worse than
        // nothing.
        if (!matches) {
            entries_.clear();
            std::remove(path_.c_str());
        }
    }

    mutable std::mutex mutex_;
    std::string signature_;
    std::string path_;
    std::unordered_map<std::uint64_t, std::pair<int, int>> entries_;
};

}  // namespace pie_cuda_driver::ops
