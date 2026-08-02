#pragma once

// Program runtime — the driver-side stage-runner entry.
// Host puts do not ride the wire (ABI v2): the runtime writes them into the
// registered channel endpoint's pinned ring and the instance pulls them
// stream-ordered before the consuming pass (`pull_writer_inputs`).
//
// Two pieces:
//   * `PtirProgramCache` — the hash-keyed program cache. Registration ships a
//     `PieLaunchPackage`, which is adopted into an executable `Trace` + per-
//     stage plans and cached by `hash`; every steady-state fire ships nothing
//     and MUST hit the cache. The cached `Trace` is the seed-independent
//     program identity; per-instance state lives in `PtirInstance`.
//   * `PtirInstance` — a per-instance execution context (§5 degenerate
//     depth-0 synchronous loop): the shared `Trace` + its own channel arena,
//     seeded at construction with the instance's D2 seed values. Each fire
//     pulls the host-writer rings (`pull_writer_inputs`), binds the intrinsic
//     `FireInputs`, runs one tier-0 pass, and harvests host-visible outputs.
//
// This is the driver half of the runtime submission path: the submit-fire call binds a
// `PtirProgramSubmission` onto the wire; the executor decodes it here and drives
// the tier-0 runner. Header-only host C++ (arena ops are CUDA memcpys).

#include <cstddef>
#include <cstdint>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "pie/driver/launch/plan.hpp"
#include "pie/driver/launch/program.hpp"
#include "pipeline/channel_registry.hpp"
#include "pipeline/tier0/tier0_runner.hpp"

namespace pie_cuda_driver::pipeline {

// The driver's execution model (trace/op-table/plan) lives in
// pie::driver::launch (driver/common); bring it into scope so the CUDA-side
// tier-0/1 code below can use it unqualified.
using namespace pie::driver::launch;

// A per-channel host-supplied byte value — mirrors the wire `PtirChannelValue`.
// Seeds are per-instance init (D2, not in the hash).
// `channel` is the GLOBAL channel id (W0.2 re-key) — the device registry key.
struct ChannelValue {
    std::uint64_t channel = 0;
    std::vector<std::uint8_t> bytes;
};

// Hash-keyed program cache. `register_program` ships the launch package once;
// every later call is a lookup.
class PtirProgramCache {
  public:
    // Return the cached `Trace` for `hash`, adopting + caching it on a first
    // registration. A steady-state fire ships no package and MUST hit the
    // cache. Returns nullptr and sets `*err` on a miss.
    //
    // There is nothing to validate here: the package is typed records the host
    // built from its own compile, and `validate_program_desc` already checked
    // every slice. The decoder, the structural validator, and the binding
    // validator this method replaced were all re-deriving what the host knew
    // (`ptir-refactor.md` §2.3).
    const Trace* adopt(std::uint64_t hash,
                       const PieLaunchPackage& package,
                       std::string* err = nullptr) {
        auto it = programs_.find(hash);
        if (it != programs_.end()) return &it->second.trace;

        if (package.stages.len == 0)
            return fail(err, "ptir program hash " + std::to_string(hash) +
                                 " not cached and this fire shipped no launch "
                                 "package");

        AdoptedProgram adopted;
        adopted.trace = pie::driver::launch::adopt(package);
        adopted.plans.reserve(package.plans.len);
        adopted.graph_stage_identities.reserve(package.plans.len);
        for (std::size_t i = 0; i < package.plans.len; ++i) {
            plan::StagePlan stage_plan =
                plan::adopt(package.stages.ptr[i].kind, package.plans.ptr[i]);
            adopted.graph_stage_identities.push_back(stage_plan.identity);
            adopted.plans.push_back(std::move(stage_plan));
        }
        auto ins = programs_.emplace(hash, std::move(adopted));
        return &ins.first->second.trace;
    }


    // Look up an already-registered program. Bind and every steady-state fire
    // go through here: only `register_program` ships a package.
    const Trace* find(std::uint64_t hash, std::string* err = nullptr) const {
        auto it = programs_.find(hash);
        if (it == programs_.end())
            return fail(err, "ptir program hash " + std::to_string(hash) +
                                 " is not registered");
        return &it->second.trace;
    }

    bool contains(std::uint64_t hash) const { return programs_.find(hash) != programs_.end(); }
    std::size_t size() const { return programs_.size(); }
    const std::vector<plan::StagePlan>* plans(std::uint64_t hash) const {
        auto it = programs_.find(hash);
        return it == programs_.end() ? nullptr : &it->second.plans;
    }
    const std::vector<std::uint64_t>* graph_stage_identities(
        std::uint64_t hash) const {
        auto it = programs_.find(hash);
        return it == programs_.end()
            ? nullptr
            : &it->second.graph_stage_identities;
    }

    // The host's per-region analysis (`region_support.hpp`'s former bind gates
    // and `analyze_direct_argmax`) is now the only copy: the driver's was
    // deleted once this counter read `divergent == 0` with `host_supplied != 0`
    // over the vendored corpus and the curated matrix. `derived` and
    // `divergent` are gone with it -- there is nothing left to derive or
    // disagree with -- and what remains is the shape check, which stopped
    // being a diagnostic and became load-bearing the moment the driver started
    // indexing this table instead of rebuilding it.
    struct RegionStats {
        std::uint64_t host_supplied = 0;
    };
    const RegionStats& region_stats() const { return region_stats_; }

    // Validate the host's region table against the program's actual shape.
    // Returns false when it names regions this program does not have, which is
    // an ABI bug: the module cache indexes this table by `(stage, region)` and
    // a short or misaddressed one silently binds the wrong kernel contract.
    bool adopt_host_region_analysis(
        std::uint64_t hash,
        const PieRegionAnalysis* host,
        std::size_t host_len,
        std::string* err) {
        auto it = programs_.find(hash);
        if (it == programs_.end()) return true;
        const auto& plans = it->second.plans;
        std::size_t derived_regions = 0;
        for (const plan::StagePlan& stage : plans) {
            derived_regions += stage.fused.regions.size();
        }
        if (host == nullptr || host_len == 0) {
            // Not an error here: a program with no fused regions needs no
            // table, and the module cache is the one that refuses a fused
            // region with nothing behind it.
            return true;
        }
        if (host_len != derived_regions) {
            if (err) {
                *err = "host supplied " + std::to_string(host_len) +
                    " region analyses for a program with " +
                    std::to_string(derived_regions) + " fused regions";
            }
            return false;
        }
        for (std::size_t entry = 0; entry < host_len; ++entry) {
            const PieRegionAnalysis& supplied = host[entry];
            if (supplied.stage_index >= plans.size()) {
                if (err) {
                    *err = "host region analysis names stage " +
                        std::to_string(supplied.stage_index) +
                        " in a program with " + std::to_string(plans.size()) +
                        " stages";
                }
                return false;
            }
            const plan::StagePlan& stage = plans[supplied.stage_index];
            if (supplied.region_index >= stage.fused.regions.size()) {
                if (err) {
                    *err = "host region analysis names region " +
                        std::to_string(supplied.region_index) +
                        " in a stage with " +
                        std::to_string(stage.fused.regions.size()) +
                        " fused regions";
                }
                return false;
            }
        }
        region_stats_.host_supplied += host_len;
        return true;
    }

  private:
    struct AdoptedProgram {
        Trace trace;
        std::vector<plan::StagePlan> plans;
        // Registration-time compact identities for the forward graph key.
        // Steady-state fires fold these integers without touching plan bytes.
        std::vector<std::uint64_t> graph_stage_identities;
    };
    static const Trace* fail(std::string* e, const std::string& m) {
        if (e) *e = m;
        return nullptr;
    }
    std::unordered_map<std::uint64_t, AdoptedProgram> programs_;
    RegionStats region_stats_;
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
            // Seeds cross the ABI in the WIRE form — the width the endpoint
            // binding reports as `cell_bytes` — and for Bool that is bits,
            // `(numel + 7) / 8`. Every surface below this line speaks the
            // NATIVE cell (one byte per bool: that is what `dtype_size`
            // says, what the device ring is sized to, and what
            // `publish_host_seed` packs FROM), so the unpack belongs here,
            // at the boundary, and nothing downstream changes its contract.
            // `pull_writer_ring` does the same conversion for the host-writer
            // ring; the seed path is the one arrival that never did.
            const void* data = s.bytes.data();
            std::size_t bytes = s.bytes.size();
            if (trace.channels[dense].type.dtype == DType::Bool) {
                // Owned by the instance, not by this loop: the copy is
                // ASYNC on the initialization stream and settles later, so a
                // buffer scoped to the iteration would be freed out from
                // under an in-flight DMA.
                seed_staging_.emplace_back(view_.cell_bytes(dense), 0);
                auto& native = seed_staging_.back();
                const auto* packed = static_cast<const std::uint8_t*>(data);
                for (std::size_t i = 0; i < native.size(); ++i) {
                    native[i] =
                        static_cast<std::uint8_t>((packed[i / 8] >> (i % 8)) & 1u);
                }
                data = native.data();
                bytes = native.size();
            }
            view_.seed_cell_async(dense, data, bytes);
            if (trace.channels[dense].host_reader) {
                view_.publish_host_seed(dense, data, bytes);
            }
        }
        // Seed copies (and the runner's baked-list upload) stay pending on the
        // registry's initialization stream — no host sync here. Fires order
        // after them via `order_after_initialization` (Dispatch::begin and the
        // tier-0 run entries); the old per-bind settle cost a stream sync on
        // every bind of a 1k-bind cohort boundary.
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

    // Device-geometry descriptor resolution still needs host-published
    // geometry before composition. Ordinary fires never call this eager probe;
    // their ticket kernel validates inputs at execution and reports RETRY.
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

    // §4.3 pull: move each host-writer channel's published ring entries into
    // the device cells, stream-ordered before the pass.
    bool pull_writer_inputs(
        cudaStream_t stream,
        std::vector<std::vector<std::uint8_t>>& staging) {
        bool copied = false;
        for (const Channel& ch : trace_->channels) {
            if (ch.host_visible && !ch.host_reader) {
                copied =
                    view_.pull_writer_ring(ch.id, stream, staging) ||
                    copied;
            }
        }
        return copied;
    }

    // One fire: run one tier-0 pass over the already-pulled channel state.
    // The result's `committed` reflects the end-of-pass predicated bump.
    PassResult fire(const FireInputs& in) {
        return runner_.run_pass(in);
    }

    PassResult fire_async(
        const FireInputs& in,
        std::vector<void*>& scratch,
        bool reset_commit = true) {
        return runner_.launch_pass_async(in, scratch, reset_commit);
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
    std::uint32_t* commit_device_flag() const noexcept {
        return runner_.commit_device_flag();
    }
    const std::vector<std::uint32_t>& commit_taken_slots() const noexcept {
        return runner_.commit_taken_slots();
    }
    const std::vector<std::uint32_t>& commit_put_slots() const noexcept {
        return runner_.commit_put_slots();
    }
    const std::uint32_t* commit_taken_device() const noexcept {
        return runner_.commit_taken_device();
    }
    std::uint32_t commit_taken_count() const noexcept {
        return runner_.commit_taken_count();
    }
    const std::uint32_t* commit_put_device() const noexcept {
        return runner_.commit_put_device();
    }
    std::uint32_t commit_put_count() const noexcept {
        return runner_.commit_put_count();
    }
    const Trace& trace() const noexcept { return *trace_; }
    void reset_commit(cudaStream_t stream) { runner_.reset_commit(stream); }
    void finalize_commit(
        cudaStream_t stream,
        const std::uint32_t* commit_override = nullptr) {
        runner_.finalize_commit(stream, commit_override);
    }
    bool takes_channel(ChannelId dense) const { return fire_takes_channel(dense); }
    bool puts_channel(ChannelId dense) const {
        for (const Stage& stage : trace_->stages) {
            for (const ChannelPut& put : stage.puts) {
                if (put.channel == dense) return true;
            }
        }
        return false;
    }
    bool requires_channel_input(ChannelId dense) const {
        const Channel& channel = trace_->channels[dense];
        if (channel.has_seed ||
            (channel.host_visible && !channel.host_reader) ||
            channel.extern_dir == 0) {
            return true;
        }
        // First-touch, as shipped. `fire_takes_channel && !puts_channel` was
        // the same question asked of the effect sets, and it gets an in-place
        // channel (taken, then put back) wrong in the one direction that
        // matters: it reads as "no input needed" for a ring whose first op
        // needs one.
        return channel.readiness == Readiness::NeedsFull;
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
            // Both arrivals are WIRE-form: a seed and a host put cross the
            // same ABI and are sized by the same `cell_bytes` the endpoint
            // binding reported. Seeds asked for the NATIVE width instead,
            // which agreed for every dtype where native == wire and was
            // wrong by exactly eight for the one that packs.
            const std::size_t expected = view_.wire_bytes(dense);
            if (value.bytes.size() != expected) {
                // Say WHICH channel and BY HOW MUCH. The bare sentence this
                // replaces is the whole diagnosis a caller got, and a
                // length mismatch is understood by its ratio: a factor of
                // the dtype width is a dtype disagreement, a whole-cell
                // multiple is a shape one.
                if (err) {
                    *err = "ptir: channel value byte length mismatch on " +
                           std::string(seeds ? "seed" : "host put") +
                           " chan#" + std::to_string(dense) +
                           " (global " + std::to_string(view_.global_id(dense)) +
                           "): got " + std::to_string(value.bytes.size()) +
                           " bytes, expected " + std::to_string(expected) +
                           " (declared numel=" +
                           std::to_string(channel.type.shape.numel()) +
                           " dtype_size=" +
                           std::to_string(dtype_size(channel.type.dtype)) +
                           ")";
                }
                return false;
            }
        }
        return true;
    }

    const Trace* trace_;
    DeviceChannelRegistry* reg_;
    ChannelView view_;
    Tier0Runner runner_;
    // Unpacked Bool seeds, kept alive for the async seed copies (see the
    // seed loop). One mask-sized buffer per bool-seeded channel.
    std::vector<std::vector<std::uint8_t>> seed_staging_;
    bool ok_ = true;
    std::vector<ChannelId> host_reader_output_channels_;
};

}  // namespace pie_cuda_driver::pipeline
