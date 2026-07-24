#pragma once

// PTIR program runtime — the driver-side stage-runner entry that consumes
// the `PtirProgramSubmission` wire contract (`8a4b20bf`):
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

#include "pie_native/ptir/bound.hpp"
#include "pipeline/channel_registry.hpp"
#include "pie_native/ptir/container.hpp"
#include "pie_native/ptir/plan.hpp"
#include "pipeline/program_identity.hpp"
#include "pipeline/tier0/tier0_runner.hpp"
#include "pie_native/ptir/trace.hpp"

namespace pie_cuda_driver::pipeline {

// Shared pure-host PTIR decode model (trace/op-table/container/bound/
// fire-geometry) now lives in pie_native::ptir (driver/common); bring it into
// scope so the CUDA-side tier-0/1 code below can use it unqualified.
using namespace pie_native::ptir;

// A per-channel host-supplied byte value — mirrors the wire `PtirChannelValue`.
// Seeds are per-instance init (D2, not in the hash).
// `channel` is the GLOBAL channel id (W0.2 re-key) — the device registry key.
struct ChannelValue {
    std::uint64_t channel = 0;
    std::vector<std::uint8_t> bytes;
};

inline bool validate_plan_structure(
    const plan::StagePlan& stage,
    std::string* error) {
    auto fail = [&](const char* message) {
        if (error != nullptr) *error = message;
        return false;
    };
    std::uint32_t next = 0;
    for (std::size_t node = 0; node < stage.ops.size(); ++node) {
        const auto& op = stage.ops[node].op;
        const std::uint32_t result_base = next;
        for (const auto argument : op.args) {
            if (argument >= result_base) {
                return fail(
                    "region plan operand is not a prior SSA value");
            }
        }
        if (op.tag == PTIR_OP_PIVOT_THRESHOLD &&
            op.pred_payload >= result_base) {
            return fail(
                "region plan predicate is not a prior SSA value");
        }
        if ((op.tag == PTIR_OP_CHAN_TAKE ||
             op.tag == PTIR_OP_CHAN_READ ||
             op.tag == PTIR_OP_CHAN_PUT) &&
            (op.chan < 0 ||
             static_cast<std::size_t>(op.chan) >=
                 stage.channel_bindings.size())) {
            return fail("region plan channel binding is out of range");
        }
        if (next >
            std::numeric_limits<std::uint32_t>::max() - op.results) {
            return fail("region plan value layout overflows u32");
        }
        next += op.results;
    }
    if (next != stage.value_types.size()) {
        return fail("region plan value layout mismatch");
    }
    auto validate_partition = [&](const plan::Partition& partition) {
        std::vector<std::uint8_t> covered(stage.ops.size(), 0);
        std::uint32_t previous_node = 0;
        bool have_previous = false;
        for (const auto& region : partition.regions) {
            if (region.nodes.empty() ||
                !std::is_sorted(
                    region.nodes.begin(), region.nodes.end())) {
                return false;
            }
            if (have_previous &&
                region.nodes.back() <= previous_node) {
                return false;
            }
            for (const auto node : region.nodes) {
                if (node >= stage.ops.size() || covered[node] != 0) {
                   return false;
                }
                covered[node] = 1;
            }
            previous_node = region.nodes.back();
            have_previous = true;
            if (!region.library) {
                if (region.schedule == PTIR_SCHEDULE_LIBRARY) return false;
                continue;
            }
            if (region.schedule != PTIR_SCHEDULE_LIBRARY) {
                return false;
            }
            if (region.library_op == PTIR_LIBRARY_NUCLEUS_SAMPLE) {
                const bool scaled = region.inputs.size() == 5;
                if (region.nodes.size() != 13 ||
                    (!scaled && region.inputs.size() != 3) ||
                    region.outputs.size() != 1 ||
                    !region.sinks.empty()) {
                    return false;
                }
                const auto logits = region.inputs[0];
                const auto top_p = region.inputs[scaled ? 3 : 1];
                const auto rng_state = region.inputs[scaled ? 4 : 2];
                const auto token = region.outputs[0];
                if (logits >= stage.value_types.size() ||
                    top_p >= stage.value_types.size() ||
                    rng_state >= stage.value_types.size() ||
                    token >= stage.value_types.size() ||
                    stage.value_types[logits].dtype != PTIR_DT_F32 ||
                    stage.value_types[logits].dims.empty() ||
                    (scaled &&
                     stage.value_types[region.inputs[1]].dtype !=
                         PTIR_DT_F32) ||
                    (scaled &&
                     stage.value_types[region.inputs[2]].dtype !=
                         PTIR_DT_F32) ||
                    stage.value_types[top_p].dtype != PTIR_DT_F32 ||
                    stage.value_types[rng_state].dtype != PTIR_DT_U32 ||
                    (stage.value_types[token].dtype != PTIR_DT_I32 &&
                     stage.value_types[token].dtype != PTIR_DT_U32)) {
                    return false;
                }
                continue;
            }
            if (region.nodes.size() != 1) return false;
            const auto& op = stage.ops[region.nodes.front()].op;
            const bool matches =
                (region.library_op == PTIR_LIBRARY_TOP_K &&
                 op.tag == PTIR_OP_TOP_K) ||
                (region.library_op == PTIR_LIBRARY_SORT &&
                 op.tag == PTIR_OP_SORT_DESC) ||
                (region.library_op == PTIR_LIBRARY_SCAN &&
                 (op.tag == PTIR_OP_CUMSUM ||
                 op.tag == PTIR_OP_CUMPROD)) ||
                (region.library_op == PTIR_LIBRARY_MATMUL &&
                 op.tag == PTIR_OP_MATMUL) ||
                (region.library_op == PTIR_LIBRARY_SECOND_PARTY &&
                 (op.tag == PTIR_OP_KERNEL_CALL ||
                  op.tag == PTIR_OP_SINK_CALL));
            if (!matches) return false;
        }
        return std::all_of(
            covered.begin(), covered.end(),
            [](std::uint8_t value) { return value != 0; });
    };
    if (!validate_partition(stage.singleton)) {
        return fail("singleton region plan partition/opcode mismatch");
    }
    if (!validate_partition(stage.fused)) {
        return fail("fused region plan partition/opcode mismatch");
    }
    return true;
}

inline bool validate_plan_bindings(
    const plan::StagePlan& stage,
    const container::CStage& source_stage,
    std::string* error) {
    auto fail = [&](const char* message) {
        if (error != nullptr) *error = message;
        return false;
    };
    for (const auto& normalized : stage.ops) {
        const auto& op = normalized.op;
        const bool channel_op =
            op.tag == PTIR_OP_CHAN_TAKE ||
            op.tag == PTIR_OP_CHAN_READ ||
            op.tag == PTIR_OP_CHAN_PUT;
        if (!channel_op) continue;
        if (op.chan < 0 ||
            static_cast<std::size_t>(op.chan) >=
                stage.channel_bindings.size()) {
            return fail("localized channel is out of range");
        }
        bool matched = false;
        for (const auto source : normalized.source_ops) {
            if (source >= source_stage.ops.size()) {
                return fail("normalized source op is out of range");
            }
            const auto& source_op = source_stage.ops[source];
            if (source_op.tag == op.tag && source_op.chan >= 0) {
                matched = true;
                if (stage.channel_bindings[
                        static_cast<std::size_t>(op.chan)] !=
                    static_cast<std::uint32_t>(source_op.chan)) {
                    return fail("localized channel binding mismatch");
                }
            }
        }
        if (!matched) return fail("channel op has no matching source binding");
    }
    return true;
}

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
        if (it != programs_.end()) {
            if (container_len != 0 || sidecar_len != 0) {
                if (container_len == 0 || sidecar_len == 0 ||
                    container_len != it->second.container_bytes.size() ||
                    sidecar_len != it->second.sidecar_bytes.size() ||
                    std::memcmp(
                        container_bytes,
                        it->second.container_bytes.data(),
                        container_len) != 0 ||
                    std::memcmp(
                        sidecar_bytes,
                        it->second.sidecar_bytes.data(),
                        sidecar_len) != 0) {
                    return fail(
                        err,
                        "ptir program hash collision or payload mismatch");
                }
            }
            return &it->second.trace;
        }

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

        DecodedProgram decoded;
        decoded.container_bytes.assign(
            container_bytes, container_bytes + container_len);
        decoded.sidecar_bytes.assign(
            sidecar_bytes, sidecar_bytes + sidecar_len);
        decoded.trace = std::move(tr.trace);
        if (b.version != PTIB_VERSION ||
            b.plans.size() != c.stages.size()) {
            return fail(err, "ptir sidecar requires compiler v3 PTRP v4 plans");
        }
        decoded.plans.reserve(b.plans.size());
        decoded.graph_stage_identities.reserve(b.plans.size());
        for (std::size_t plan_index = 0;
             plan_index < b.plans.size();
             ++plan_index) {
            const bound::StagePlan& encoded = b.plans[plan_index];
            plan::StagePlan stage_plan;
            std::string plan_error;
            if (!plan::decode(
                    encoded.bytes.data(), encoded.bytes.size(), stage_plan, &plan_error)) {
                return fail(err, "ptir region plan: " + plan_error);
            }
            if (stage_plan.stage != encoded.stage ||
                stage_plan.signature_hash == 0) {
                return fail(err, "ptir region plan identity mismatch");
            }
            if (!validate_plan_structure(stage_plan, &plan_error)) {
                return fail(err, "ptir region plan: " + plan_error);
            }
            if (!validate_plan_bindings(
                    stage_plan,
                    c.stages[plan_index],
                    &plan_error)) {
                return fail(
                    err,
                    "ptir region plan binding: " + plan_error);
            }
            decoded.graph_stage_identities.push_back(
                compiled_stage_identity(stage_plan));
            decoded.plans.push_back(std::move(stage_plan));
        }
        auto ins = programs_.emplace(hash, std::move(decoded));
        return &ins.first->second.trace;
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

  private:
    struct DecodedProgram {
        Trace trace;
        std::vector<std::uint8_t> container_bytes;
        std::vector<std::uint8_t> sidecar_bytes;
        std::vector<plan::StagePlan> plans;
        // Registration-time compact identities for the forward graph key.
        // Steady-state fires fold these integers without touching plan bytes.
        std::vector<std::uint64_t> graph_stage_identities;
    };
    static const Trace* fail(std::string* e, const std::string& m) {
        if (e) *e = m;
        return nullptr;
    }
    std::unordered_map<std::uint64_t, DecodedProgram> programs_;
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
            view_.seed_cell_async(dense, s.bytes.data(), s.bytes.size());
            if (trace.channels[dense].host_reader) {
                view_.publish_host_seed(
                    dense, s.bytes.data(), s.bytes.size());
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
        return fire_takes_channel(dense) && !puts_channel(dense);
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

}  // namespace pie_cuda_driver::pipeline
