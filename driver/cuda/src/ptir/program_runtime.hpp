#pragma once

// PTIR program runtime — the driver-side stage-runner entry that consumes
// delta's `PtirProgramSubmission` wire contract (P2c, `8a4b20bf`):
//   { hash, bytes?(first-fire container), sidecar?(first-fire PTIB),
//     seeds[](per-instance init D2), host_puts[](per-fire input) }.
//
// Two pieces:
//   * `PtirProgramCache` — the C3 hash-keyed DECODE cache (mirrors
//     `SamplingIrBackend::get_or_compile`): the first fire of a hash ships the
//     container + PTIB sidecar bytes → decode → fold into an executable `Trace`
//     → cache by `hash`; every steady-state fire ships empty bytes and MUST hit
//     the cache. The cached `Trace` is the seed-independent program identity
//     (the hash's payload D1); per-instance state lives in `PtirInstance`.
//   * `PtirInstance` — a per-instance execution context (thrust-3 §5 degenerate
//     depth-0 synchronous loop): the shared `Trace` + its own channel arena,
//     seeded at construction with the instance's D2 seed values. Each fire binds
//     the per-fire host-puts (`host_feed`) + intrinsics (`FireInputs`), runs one
//     tier-0 pass, and harvests host-visible outputs (`host_take`).
//
// This is the driver half of P2c: delta's ~10-line runtime submit-fire binds a
// `PtirProgramSubmission` onto the wire; the executor decodes it here and drives
// the tier-0 runner. Header-only host C++ (arena ops are CUDA memcpys).

#include <cstddef>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ptir/bound.hpp"
#include "ptir/container.hpp"
#include "ptir/tier0_runner.hpp"
#include "ptir/trace.hpp"

namespace pie_cuda_driver::ptir {

// A per-channel host-supplied byte value — mirrors delta's `PtirChannelValue`.
// Seeds are per-instance init (D2, not in the hash); host_puts are per-fire.
struct ChannelValue {
    std::uint32_t channel = 0;
    std::vector<std::uint8_t> bytes;
};

// C3 hash-keyed decoded-program cache. The `Trace` is folded once per identity
// hash from the first fire's container + sidecar; steady-state fires reuse it.
class PtirProgramCache {
  public:
    // Return the cached `Trace` for `hash`, decoding + caching it on a first
    // fire (non-empty container AND sidecar). A steady-state fire (empty bytes)
    // MUST hit the cache. Returns nullptr and sets `*err` on any failure — a
    // decode/version/hash mismatch is a host/bridge bug, surfaced loudly.
    const Trace* get_or_decode(std::uint64_t hash,
                               const std::uint8_t* container_bytes, std::size_t container_len,
                               const std::uint8_t* sidecar_bytes, std::size_t sidecar_len,
                               std::string* err = nullptr) {
        auto it = programs_.find(hash);
        if (it != programs_.end()) return &it->second;  // steady-state cache hit

        if (container_len == 0 || sidecar_len == 0)
            return fail(err, "ptir program hash " + std::to_string(hash) +
                                 " not cached and this fire shipped no first-fire "
                                 "container/sidecar bytes");

        // Structural container (channels/ports/stage op tags) …
        container::Container c;
        container::DecodeError de;
        if (!container::decode(container_bytes, container_len, c, &de))
            return fail(err, "ptir container decode: " + de.detail);

        // … + the PTIB typed sidecar (per-SSA (dtype, shape), channel classes,
        // readiness table — Option-B: the driver does NOT re-infer shapes).
        bound::Bound b;
        std::string se;
        if (!bound::parse_sidecar(sidecar_bytes, sidecar_len, b, &se))
            return fail(err, "ptir sidecar parse: " + se);

        // The identity chain must agree: wire hash == container hash ==
        // sidecar's inner container_hash (else the bytes were mispaired).
        if (c.hash != hash || b.container_hash != hash)
            return fail(err, "ptir hash mismatch: wire=" + std::to_string(hash) +
                                 " container=" + std::to_string(c.hash) +
                                 " sidecar=" + std::to_string(b.container_hash));

        bound::TranslateResult tr = bound::container_to_trace(c, b);
        if (!tr.ok) return fail(err, "ptir container->trace: " + tr.error);

        auto ins = programs_.emplace(hash, std::move(tr.trace));
        return &ins.first->second;
    }

    bool contains(std::uint64_t hash) const { return programs_.find(hash) != programs_.end(); }
    std::size_t size() const { return programs_.size(); }

  private:
    static const Trace* fail(std::string* e, const std::string& m) {
        if (e) *e = m;
        return nullptr;
    }
    std::unordered_map<std::uint64_t, Trace> programs_;
};

// Per-instance execution context: the shared cached `Trace` + its own channel
// arena. Seeded once at construction; fired synchronously (depth-0) thereafter.
class PtirInstance {
  public:
    // Instantiate over a cached `Trace`, applying the instance's D2 seed values
    // to their `seeded` channels' initial cells (`Channel::from`).
    PtirInstance(const Trace& trace, const std::vector<ChannelValue>& seeds)
        : trace_(&trace), runner_(trace) {
        for (const ChannelValue& s : seeds)
            runner_.arena().seed_cell(static_cast<ChannelId>(s.channel),
                                      s.bytes.data(), s.bytes.size());
    }

    // One fire: bind per-fire host-puts (host-fed channels arriving) + the
    // intrinsic/host-tensor `FireInputs`, then run one tier-0 pass. The result's
    // `committed` reflects the end-of-pass predicated commit-bump.
    PassResult fire(const std::vector<ChannelValue>& host_puts, const FireInputs& in) {
        for (const ChannelValue& hp : host_puts)
            runner_.arena().host_feed(static_cast<ChannelId>(hp.channel),
                                      hp.bytes.data(), hp.bytes.size());
        return runner_.run_pass(in);
    }

    // Harvest a host-visible output channel's committed cell (post-commit).
    // Returns false (WouldBlock) if the channel is not currently full.
    bool take_output(ChannelId c, void* out, std::size_t bytes) {
        if (!runner_.arena().committed_full(c)) return false;
        runner_.arena().host_take(c, out, bytes);
        return true;
    }

    // Enumerate the committed host-READER output channels post-fire →
    // `(channel_id, wire_bytes)` pairs — delta's ForwardResponse `ptir_output_*`
    // SoA. ONLY channels that committed THIS fire appear (back-pressure leaves
    // others empty), so the list is a per-fire subset of the declared readers.
    // Consumes (`host_take`) each — one harvest per fire.
    std::vector<std::pair<std::uint32_t, std::vector<std::uint8_t>>> harvest_outputs() {
        std::vector<std::pair<std::uint32_t, std::vector<std::uint8_t>>> outs;
        for (const Channel& ch : trace_->channels) {
            if (!ch.host_reader) continue;
            if (!runner_.arena().committed_full(ch.id)) continue;
            const std::size_t n = runner_.arena().cell_bytes(ch.id);
            std::vector<std::uint8_t> bytes(n);
            runner_.arena().host_take(ch.id, bytes.data(), n);
            outs.emplace_back(static_cast<std::uint32_t>(ch.id), std::move(bytes));
        }
        return outs;
    }

    Tier0Runner& runner() { return runner_; }
    ChannelArena& arena() { return runner_.arena(); }

  private:
    const Trace* trace_;
    Tier0Runner runner_;
};

}  // namespace pie_cuda_driver::ptir
