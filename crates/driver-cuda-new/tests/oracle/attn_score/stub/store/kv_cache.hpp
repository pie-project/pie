// Stub `store/kv_cache.hpp`.
//
// `attn_score.cu` reads exactly one thing off the cache: `page_size()`, to
// turn the page CSR into per-request KV lengths. The live cache is proven by
// its own gate (`tests/oracle/kv_cache_live/`); here it is one settable
// integer, which is also what lets the sweep hand different page sizes to
// different fires without allocating anything.
#pragma once

namespace pie_cuda_driver {

class KvCache {
  public:
    explicit KvCache(int page_size) : page_size_(page_size) {}
    int page_size() const noexcept { return page_size_; }

  private:
    int page_size_ = 0;
};

}  // namespace pie_cuda_driver
