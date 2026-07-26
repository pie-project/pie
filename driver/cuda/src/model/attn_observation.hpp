#pragma once

#include <cstdint>

namespace pie_cuda_driver {

class KvCache;

namespace model {

// What a PTIR attention-stage program is allowed to observe about the fire it
// is running inside.
//
// `StageHookPoint` already tells a program WHEN it runs and hands it the query;
// this tells it WHAT the query is being scored against. Quest's `envelope_dot`
// needs the KV page table of the fire (which physical pages belong to which
// request, and how much of the last one is live) plus the cache the envelopes
// live in — none of which is reachable from a stage hook, and all of which the
// batch layer already has in `ForwardInputs`.
//
// It is published as a thread-local by `ForwardFn::invoke_body` for exactly the
// duration of one model body, alongside `ScopedStageHooks`. That placement is
// deliberate: it is the single choke point every model family already passes
// through, so no model file has to know this exists. A model that is mid-body
// on another thread has its own binding.
//
// Every pointer is borrowed from the caller's `ForwardInputs` and is only valid
// while the body runs.
struct AttentionObservation {
    KvCache* kv = nullptr;

    // Fire CSR, device side. Page ids are indexed by `kv_page_indptr`.
    const std::uint32_t* kv_page_indices_d = nullptr;

    // Fire CSR, host side. Needed to slice the device page list per request
    // without a device round trip.
    const std::uint32_t* qo_indptr_h = nullptr;
    const std::uint32_t* kv_page_indptr_h = nullptr;
    const std::uint32_t* kv_last_page_lens_h = nullptr;

    int num_requests = 0;
    int total_tokens = 0;

    bool usable() const noexcept {
        return kv != nullptr && kv_page_indices_d != nullptr &&
               qo_indptr_h != nullptr && kv_page_indptr_h != nullptr &&
               kv_last_page_lens_h != nullptr && num_requests > 0;
    }
};

// Null outside a model body, or inside one that was invoked without a fire
// descriptor.
const AttentionObservation* active_attention_observation() noexcept;

class ScopedAttentionObservation {
  public:
    explicit ScopedAttentionObservation(const AttentionObservation* observation) noexcept;
    ~ScopedAttentionObservation() noexcept;

    ScopedAttentionObservation(const ScopedAttentionObservation&) = delete;
    ScopedAttentionObservation& operator=(const ScopedAttentionObservation&) = delete;

  private:
    const AttentionObservation* previous_ = nullptr;
};

}  // namespace model
}  // namespace pie_cuda_driver
