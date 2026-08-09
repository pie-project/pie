// Stub `model/attn_observation.hpp`.
//
// `attn_page_mask.cu` reads exactly two things off an observation —
// `usable()` and the host page CSR (`kv_page_indptr_h`, `num_requests`) — but
// `usable()` gates on seven pointers, so all seven are here. They are kept
// because `usable()` is part of the behaviour under test: a fire that is not
// usable must throw rather than carve, and an oracle that could not construct
// an unusable observation could not check that.
#pragma once

#include <cstdint>

namespace pie_cuda_driver::model {

struct KvCache;

struct AttentionObservation {
    KvCache* kv = nullptr;
    const std::uint32_t* kv_page_indices_d = nullptr;
    const std::uint32_t* kv_page_indptr_d = nullptr;
    const std::uint32_t* kv_last_page_lens_d = nullptr;
    const std::uint32_t* qo_indptr_h = nullptr;
    const std::uint32_t* kv_page_indptr_h = nullptr;
    const std::uint32_t* kv_last_page_lens_h = nullptr;

    int num_requests = 0;
    int total_tokens = 0;

    bool usable() const noexcept {
        return kv != nullptr && kv_page_indices_d != nullptr &&
               kv_page_indptr_d != nullptr && kv_last_page_lens_d != nullptr &&
               qo_indptr_h != nullptr && kv_page_indptr_h != nullptr &&
               kv_last_page_lens_h != nullptr && num_requests > 0;
    }
};

}  // namespace pie_cuda_driver::model
