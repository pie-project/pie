#pragma once

// PTIR program runtime — the driver-side stage-runner entry that consumes
// delta's `PtirProgramSubmission` wire contract (P2c, `8a4b20bf`):
//   { hash, bytes?(first-fire container), sidecar?(first-fire PTIB),
//     seeds[](per-instance init D2) }.
// Host puts do not ride the wire (ABI v2): the runtime writes them into the
// registered channel endpoint's pinned ring and the instance pulls them
// stream-ordered before the consuming pass (`pull_writer_inputs`).
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
//     seeded at construction with the instance's D2 seed values. Each fire
//     pulls the host-writer rings (`pull_writer_inputs`), binds the intrinsic
//     `FireInputs`, runs one tier-0 pass, and harvests host-visible outputs.
//
// This is the driver half of P2c: delta's ~10-line runtime submit-fire binds a
// `PtirProgramSubmission` onto the wire; the executor decodes it here and drives
// the tier-0 runner. Header-only host C++ (arena ops are CUDA memcpys).

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ptir/bound.hpp"
#include "ptir/channel_registry.hpp"
#include "ptir/container.hpp"
#include "ptir/tier0_runner.hpp"
#include "ptir/trace.hpp"

namespace pie_cuda_driver::ptir {

// A per-channel host-supplied byte value — mirrors delta's `PtirChannelValue`.
// Seeds are per-instance init (D2, not in the hash).
// `channel` is the GLOBAL channel id (W0.2 re-key) — the device registry key.
struct ChannelValue {
    std::uint64_t channel = 0;
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

// Per-instance execution context: the shared cached `Trace` + a channel VIEW
// onto the global device channel registry (W0.1). The view maps the trace's
// dense channel indices to shared global slots (`channel_ids` from the wire);
// channels shared across instances/passes resolve one device cell ring. Seeded
// once at construction; fired synchronously (depth-0) thereafter.
class PtirInstance {
  public:
    // Instantiate over a cached `Trace`, binding its dense channels to the
    // global registry via `channel_ids` (dense idx → global id) and applying the
    // D2 seed values (keyed by GLOBAL id). `*err` set + `ok()==false` on a decl
    // conflict / OOM.
    PtirInstance(const Trace& trace, DeviceChannelRegistry* reg,
                 const std::vector<std::uint64_t>& channel_ids,
                 const std::vector<ChannelValue>& seeds, std::string* err)
        : trace_(&trace), reg_(reg), runner_(trace) {
        if (!view_.bind(reg, trace.channels, channel_ids, err)) {
            ok_ = false;
            return;
        }
        runner_.bind_view(&view_);
        if (!validate_values(seeds, true, err)) {
            ok_ = false;
            return;
        }
        for (const ChannelValue& s : seeds) {
            const ChannelId dense = dense_channel(s.channel);
            view_.seed_cell(dense, s.bytes.data(), s.bytes.size());
            if (trace.channels[dense].host_reader) {
                view_.publish_host_seed(
                    dense, s.bytes.data(), s.bytes.size());
            }
        }
        for (const Channel& ch : trace.channels) {
            if (!ch.host_reader) continue;
            bool produced = false;
            for (const Stage& st : trace.stages) {
                for (const ChannelPut& put : st.puts) {
                    if (put.channel == ch.id) {
                        produced = true;
                        break;
                    }
                }
                if (produced) break;
            }
            if (produced) host_reader_output_channels_.push_back(ch.id);
        }
    }

    bool ok() const { return ok_; }

    // §4.3 writer availability: every host-writer channel this trace takes
    // must have a host input (a published ring entry or the one-shot seed
    // credit) before the fire is accepted. A missing put is a guest ordering
    // bug — the launch rejects synchronously, no epoch, no poison.
    bool writer_inputs_available(std::string* err = nullptr) const {
        for (const Channel& ch : trace_->channels) {
            if (!ch.host_visible || ch.host_reader) continue;
            if (!fire_takes_channel(ch.id)) continue;
            if (reg_->writer_available(view_.slot(ch.id)) < 1) {
                if (err) {
                    *err = "ptir channel " +
                        std::to_string(view_.global_id(ch.id)) +
                        " has no host input for this fire "
                        "(put must happen before submit)";
                }
                return false;
            }
        }
        return true;
    }

    // The registry slots one fire of this instance consumes a writer-ring
    // entry from — batch validation aggregates these across members sharing
    // a channel (multi-pass channels).
    std::vector<std::pair<std::uint32_t, std::uint64_t>> writer_take_slots() const {
        std::vector<std::pair<std::uint32_t, std::uint64_t>> out;
        for (const Channel& ch : trace_->channels) {
            if (!ch.host_visible || ch.host_reader) continue;
            if (!fire_takes_channel(ch.id)) continue;
            out.emplace_back(view_.slot(ch.id), view_.global_id(ch.id));
        }
        return out;
    }

    // §4.3 pull: move each host-writer channel's published ring entries into
    // the device cells, stream-ordered before the pass.
    void pull_writer_inputs(
        cudaStream_t stream,
        std::vector<std::vector<std::uint8_t>>& staging) {
        for (const Channel& ch : trace_->channels) {
            if (ch.host_visible && !ch.host_reader) {
                view_.pull_writer_ring(ch.id, stream, staging);
            }
        }
    }

    // Schedule this fire's writer-ring consumes: one per taken host-writer
    // channel. Returns `(slot, target_head)` pairs for the completion
    // callback's head-word publication; a seed-credit spend schedules no
    // ring consume and is omitted.
    std::vector<std::pair<std::uint32_t, std::uint64_t>> schedule_writer_consumes() {
        std::vector<std::pair<std::uint32_t, std::uint64_t>> out;
        for (const Channel& ch : trace_->channels) {
            if (!ch.host_visible || ch.host_reader) continue;
            if (!fire_takes_channel(ch.id)) continue;
            const std::uint32_t slot = view_.slot(ch.id);
            const std::uint64_t target = reg_->schedule_writer_consume(slot);
            if (target != 0) out.emplace_back(slot, target);
        }
        return out;
    }

    // One fire: run one tier-0 pass over the already-pulled channel state.
    // The result's `committed` reflects the end-of-pass predicated bump.
    PassResult fire(const FireInputs& in) {
        return runner_.run_pass(in);
    }

    PassResult fire_async(const FireInputs& in, std::vector<void*>& scratch) {
        return runner_.launch_pass_async(in, scratch);
    }

    // Harvest a host-visible output channel's committed cell (post-commit) by
    // DENSE index. Returns false (WouldBlock) if the channel is not full.
    bool take_output(ChannelId c, void* out, std::size_t bytes) {
        if (!view_.committed_full(c)) return false;
        view_.host_take(c, out, bytes);
        return true;
    }

    // Enumerate the committed host-reader output channels post-fire as
    // `(GLOBAL channel id, wire_bytes)` pairs in the runtime's publication
    // order (re-keyed by global id, W0.2). ONLY channels that
    // committed THIS fire appear (back-pressure leaves others empty). Consumes
    // (`host_take`) each — one harvest per fire.
    std::vector<std::pair<std::uint64_t, std::vector<std::uint8_t>>> harvest_outputs() {
        std::vector<std::pair<std::uint64_t, std::vector<std::uint8_t>>> outs;
        for (const Channel& ch : trace_->channels) {
            if (!ch.host_reader) continue;
            if (!view_.committed_full(ch.id)) continue;
            const std::size_t n = view_.cell_bytes(ch.id);
            std::vector<std::uint8_t> bytes(n);
            view_.host_take(ch.id, bytes.data(), n);
            outs.emplace_back(view_.global_id(ch.id), std::move(bytes));
        }
        return outs;
    }

    // Phase-3 DEVICE value path (C5 — values move by DMA): enumerate committed
    // host-READER channels post-fire → (global id, DEVICE committed-cell ptr, cell
    // bytes, dense id). Does NOT consume — the caller DMAs the device cell straight
    // into the pinned mirror (no host bounce buffer), THEN calls `consume_outputs`
    // to free the device ring slot (after the copy stream has drained the DMA).
    struct DeviceOut {
        std::uint64_t gid;
        void*         device_ptr;
        std::size_t   bytes;
        ChannelId     ch;
        std::uint32_t slot;
    };
    std::vector<DeviceOut> harvest_outputs_device() {
        std::vector<DeviceOut> outs;
        for (const Channel& ch : trace_->channels) {
            if (!ch.host_reader) continue;
            if (!view_.committed_full(ch.id)) continue;
            outs.push_back(DeviceOut{view_.global_id(ch.id), view_.committed_cell(ch.id),
                                     view_.cell_bytes(ch.id), ch.id, view_.slot(ch.id)});
        }
        return outs;
    }
    void consume_outputs(const std::vector<DeviceOut>& outs) {
        for (const DeviceOut& o : outs) view_.host_consume(o.ch);
    }
    std::vector<DeviceOut> predict_outputs_device() {
        std::vector<DeviceOut> outs;
        outs.reserve(host_reader_output_channels_.size());
        for (ChannelId ch : host_reader_output_channels_) {
            outs.push_back(DeviceOut{
                view_.global_id(ch),
                view_.pending_cell(ch),
                view_.cell_bytes(ch),
                ch,
                view_.slot(ch),
            });
        }
        return outs;
    }
    void project_fire_success(const std::vector<DeviceOut>& outs) {
        runner_.apply_host_commit(true);
        if (outs.empty()) return;
        std::vector<std::uint32_t> output_slots;
        output_slots.reserve(outs.size());
        for (const DeviceOut& out : outs) output_slots.push_back(out.slot);
        view_.apply_host_consume(output_slots);
    }
    std::uint32_t* commit_device_flag() const noexcept {
        return runner_.commit_device_flag();
    }
    ChannelView& view() { return view_; }

  private:
    ChannelId dense_channel(std::uint64_t global_id) const {
        for (ChannelId dense = 0; dense < trace_->channels.size(); ++dense) {
            if (view_.global_id(dense) == global_id) return dense;
        }
        return static_cast<ChannelId>(trace_->channels.size());
    }

    // Whether one fire consumes (takes) dense channel `dense` — a stage
    // `chan_take` or a consuming descriptor port. A pass bumps a channel's
    // ring index at most once (register semantics), so this is the per-fire
    // consume count.
    bool fire_takes_channel(ChannelId dense) const {
        for (const Stage& stage : trace_->stages) {
            for (ChannelId taken : stage.takes) {
                if (taken == dense) return true;
            }
        }
        for (const PortBinding& binding : trace_->ports) {
            if (!binding.is_const && binding.channel == dense &&
                port_consumes(binding.port)) {
                return true;
            }
        }
        return false;
    }

    bool validate_values(const std::vector<ChannelValue>& values,
                         bool seeds,
                         std::string* err) const {
        std::unordered_set<std::uint64_t> seen;
        for (const ChannelValue& value : values) {
            const ChannelId dense = dense_channel(value.channel);
            if (dense >= trace_->channels.size()) {
                if (err) *err = "ptir: channel value references an unbound channel";
                return false;
            }
            if (!seen.insert(value.channel).second) {
                if (err) *err = "ptir: duplicate channel value";
                return false;
            }
            const Channel& channel = trace_->channels[dense];
            if ((seeds && !channel.has_seed) ||
                (!seeds &&
                 (!channel.host_visible || channel.host_reader))) {
                if (err) {
                    *err = seeds
                        ? "ptir: seed targets a non-seeded channel"
                        : "ptir: host put targets a non-writer channel";
                }
                return false;
            }
            const std::size_t expected =
                seeds ? view_.cell_bytes(dense) : view_.wire_bytes(dense);
            if (value.bytes.size() != expected) {
                if (err) *err = "ptir: channel value byte length mismatch";
                return false;
            }
        }
        return true;
    }

    const Trace* trace_;
    DeviceChannelRegistry* reg_;
    ChannelView view_;
    Tier0Runner runner_;
    bool ok_ = true;
    std::vector<ChannelId> host_reader_output_channels_;
};

}  // namespace pie_cuda_driver::ptir
