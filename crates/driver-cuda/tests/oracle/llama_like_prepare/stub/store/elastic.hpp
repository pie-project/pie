#pragma once
// Stub for csrc/src/store/elastic.hpp.
//
// `kv_cache.cpp` uses exactly three things from the elastic arena --
// `ensure_fraction`, `trim_fraction`, `committed_bytes` -- and only to forward
// a page count to it. The arena's own behaviour is proved separately by
// `crate::cuda::vmm`, on a real GPU. What is unproved, and what this stub
// makes visible, is the CLAMP and the ratio `kv_cache.cpp` applies before it
// forwards.
#include <algorithm>
#include <cstddef>
#include <string>
#include <vector>

namespace pie_cuda_driver {

inline std::vector<std::string> g_arena_log;

class CudaArenaAllocator {
public:
    void ensure_fraction(std::size_t used, std::size_t capacity) {
        g_arena_log.push_back("ensure(" + std::to_string(used) + "/" +
                              std::to_string(capacity) + ")");
        committed_ = capacity == 0 ? 0 : used * kBytesPerPage;
    }
    void trim_fraction(std::size_t used, std::size_t capacity) {
        g_arena_log.push_back("trim(" + std::to_string(used) + "/" +
                              std::to_string(capacity) + ")");
        committed_ = capacity == 0 ? 0 : used * kBytesPerPage;
    }
    std::size_t committed_bytes() const noexcept { return committed_; }

private:
    static constexpr std::size_t kBytesPerPage = 4096;
    std::size_t committed_ = 0;
};

}  // namespace pie_cuda_driver
