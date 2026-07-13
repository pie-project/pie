// PTIR stage-program dispatcher — the nvcc-compiled impl behind the
// CUDA-free `dispatch.hpp` façade. Includes the tier-0 runtime (device
// kernels) here, isolated from the host `.cpp` translation units.

#include "pipeline/dispatch.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <deque>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cuda_bf16.h>

#include "cuda_check.hpp"
#include "pipeline/program_runtime.hpp"
#include "pipeline/grouped_runtime.cuh"
#include "pipeline/generated/module_cache.hpp"
#include "pipeline/generated/fused_runtime.cuh"

#include "pipeline/descriptor_resolve.hpp"
#include "pipeline/frame_carrier.hpp"
#include "pipeline/page_translation.hpp"
#include "batch/rs_metadata.hpp"

namespace pie_cuda_driver::pipeline {

// Shared pure-host PTIR decode model (trace/op-table/container/bound/
// fire-geometry) now lives in pie_native::ptir (driver/common); bring it into
// scope so the CUDA-side tier-0/1 code below can use it unqualified.
using namespace pie_native::ptir;

struct BoundInstance {
    struct CommitSnapshot {
        std::uint32_t* device = nullptr;
        std::uint32_t* host = nullptr;
    };

    std::uint64_t program_hash = 0;
    std::uint64_t pacing_wait_id = 0;
    const Trace* trace = nullptr;
    std::vector<std::uint64_t> channel_ids;
    std::unique_ptr<PtirInstance> instance;
    cudaEvent_t publish_done = nullptr;
    std::deque<CommitSnapshot> commit_snapshots;
};

struct Dispatch::Impl {
    static constexpr std::size_t kSignatureStreamCount = 4;
    PtirProgramCache cache;
    generated::ModuleCache fused_modules;
    DeviceChannelRegistry channels;
    std::unordered_map<std::uint64_t, BoundInstance> instances;
    std::atomic<bool> shutting_down{false};
    std::atomic<std::uint32_t> force_retry_launches_remaining{
        std::getenv("PIE_CUDA_FORCE_RETRY_ONCE") != nullptr ? 1u : 0u
    };
    DispatchStats stats;
    mutable std::mutex stats_mutex;
    cudaStream_t group_streams[2] = {nullptr, nullptr};
    cudaStream_t signature_streams[kSignatureStreamCount] = {};
    bool attention_hook_coverage = false;
    std::uint32_t model_layers = 0;
};

struct StagedLane {
    std::size_t program = 0;
    std::uint64_t instance_id = 0;
    BoundInstance* bound = nullptr;
    BoundInstance::CommitSnapshot* snapshot = nullptr;
    const std::vector<plan::StagePlan>* plans = nullptr;
    const std::vector<std::uint64_t>* plan_identities = nullptr;
    std::shared_ptr<const generated::FusedProgramExecutable>
        generated_program;
    std::array<std::vector<const plan::StagePlan*>, 4> phase_plans;
    std::vector<DeviceHostChannelTicket> tickets;
    DeviceHostChannelTicket* device_tickets = nullptr;
    std::uint32_t device_ticket_count = 0;
    std::unordered_set<std::uint32_t> prior_put_slots;
    std::unordered_set<std::uint32_t> prior_take_slots;
    std::uint32_t row_offset = 0;
    std::uint32_t sampled_rows = 0;
    std::uint32_t token_start = 0;
    std::uint32_t runtime_row_count = kUnavailableGroupedExtent;
    std::uint32_t token_count = kUnavailableGroupedExtent;
    std::uint32_t kv_len = kUnavailableGroupedExtent;
    std::uint32_t page_count = kUnavailableGroupedExtent;
    std::uint32_t query_len = kUnavailableGroupedExtent;
    std::uint32_t key_len = kUnavailableGroupedExtent;
    std::uint32_t logical_vocab = 0;
    std::vector<std::uint64_t> logits_bf16_rows;
    std::vector<std::uint64_t> mtp_logits_bf16_rows;
};

struct StagedLaunch::State {
    Dispatch::Impl* owner = nullptr;
    pie_native::LaunchView view{};
    cudaStream_t stream = nullptr;
    std::vector<std::unique_ptr<StagedLane>> lanes;
    std::vector<std::uint64_t> touched_instances;
    std::uint32_t* device_layer = nullptr;
    cudaEvent_t source_ready = nullptr;
    cudaEvent_t phase_done[2] = {nullptr, nullptr};
    cudaEvent_t signature_ready = nullptr;
    cudaEvent_t signature_done[
        Dispatch::Impl::kSignatureStreamCount] = {};
    std::array<std::uint32_t, 4> phase_invocations{};
    bool active = true;
    bool failed = false;
};

StagedLaunch::StagedLaunch() : state_(std::make_unique<State>()) {}

StagedLaunch::~StagedLaunch() {
    if (!state_) return;
    if (state_->active && state_->stream != nullptr) {
        cudaStreamSynchronize(state_->stream);
    }
    for (auto& lane : state_->lanes) {
        if (lane != nullptr && lane->device_tickets != nullptr) {
            cudaFree(lane->device_tickets);
            lane->device_tickets = nullptr;
        }
    }
    if (state_->device_layer != nullptr) {
        cudaFree(state_->device_layer);
        state_->device_layer = nullptr;
    }
    if (state_->source_ready != nullptr) {
        cudaEventDestroy(state_->source_ready);
        state_->source_ready = nullptr;
    }
    for (cudaEvent_t& event : state_->phase_done) {
        if (event != nullptr) {
            cudaEventDestroy(event);
            event = nullptr;
        }
    }
    if (state_->signature_ready != nullptr) {
        cudaEventDestroy(state_->signature_ready);
        state_->signature_ready = nullptr;
    }
    for (cudaEvent_t& event : state_->signature_done) {
        if (event != nullptr) {
            cudaEventDestroy(event);
            event = nullptr;
        }
    }
}

struct NotifyContext {
    PieRuntimeCallbacks runtime{};
    PieCompletion completion{};
    struct FinalizeEntry {
        struct EndpointUpdate {
            std::uint32_t slot = DeviceChannelRegistry::kBadSlot;
            std::uint64_t target = 0;
            std::uint64_t wait_id = 0;
            // Pinned word block, resolved at enqueue time on the scheduler
            // thread. The completion callback dereferences ONLY this stable
            // pointer (plan §7): registry vectors may be reallocated by a
            // concurrent register_endpoint, but the per-slot pinned block
            // lives until the channel's ordered close.
            std::uint64_t* words = nullptr;
        };

        PieTerminalCell* terminal_cell = nullptr;
        std::uint32_t* commit_host = nullptr;
        bool poison = false;
        std::uint64_t instance_id = 0;
        std::vector<DeviceHostChannelTicket> tickets;
        std::vector<EndpointUpdate> published;
        std::vector<EndpointUpdate> consumed;
        std::vector<EndpointUpdate> poisoned;
    };
    Dispatch::Impl* impl = nullptr;
    std::vector<FinalizeEntry> entries;
    std::vector<CommitBumpLane> commit_lanes;
    std::vector<HostChannelSettlementLane> settlement_lanes;
};

// Word-pointer variants of DeviceChannelRegistry::finalize_host_publish /
// finalize_host_consume for the completion callback: the callback must not
// index registry vectors (a concurrent register_endpoint may reallocate
// them), so it writes through the pinned word pointers precomputed at
// enqueue. Word layout: [0]=head, [1]=tail, [2]=poison, [3]=closed.
void finalize_publish_words(std::uint64_t* words, std::uint64_t target, bool failed) {
    if (words == nullptr) return;
    if (failed) {
        std::atomic_ref<std::uint64_t>(words[2]).store(
            target == 0 ? 1 : target, std::memory_order_release);
        return;
    }
    std::atomic_ref<std::uint64_t>(words[1]).store(target, std::memory_order_release);
    std::atomic_ref<std::uint64_t>(words[2]).store(0, std::memory_order_release);
}

void CUDART_CB notify_runtime_callback(void* userdata) {
    std::unique_ptr<NotifyContext> ctx(static_cast<NotifyContext*>(userdata));
    if (ctx == nullptr) return;
    const bool notify =
        ctx->runtime.notify != nullptr &&
        (ctx->impl == nullptr ||
         !ctx->impl->shutting_down.load(std::memory_order_acquire));
    std::vector<std::pair<std::uint64_t, std::uint64_t>> notifications;
    for (const auto& entry : ctx->entries) {
        const bool committed =
            entry.commit_host != nullptr && *(entry.commit_host) != 0;
        const bool failed = entry.poison;
        const bool retry = !failed && !committed;
        if ((retry || committed) &&
            std::getenv("PIE_CHANNEL_RETRY_TRACE") != nullptr) {
            std::fprintf(
                stderr,
                "[channel-%s] instance=%llu tickets=%zu\n",
                retry ? "retry" : "commit",
                static_cast<unsigned long long>(entry.instance_id),
                entry.tickets.size());
            for (const auto& ticket : entry.tickets) {
                const std::uint64_t head =
                    std::atomic_ref<std::uint64_t>(ticket.words[0]).load(
                        std::memory_order_acquire);
                const std::uint64_t tail =
                    std::atomic_ref<std::uint64_t>(ticket.words[1]).load(
                        std::memory_order_acquire);
                std::fprintf(
                    stderr,
                    "  slot=%u flags=%x head=%llu/%llu tail=%llu/%llu\n",
                    ticket.slot,
                    ticket.flags,
                    static_cast<unsigned long long>(head),
                    static_cast<unsigned long long>(ticket.expected_head),
                    static_cast<unsigned long long>(tail),
                    static_cast<unsigned long long>(ticket.expected_tail));
            }
        }
        if (committed) {
            for (const auto& update : entry.published) {
                const std::uint64_t actual =
                    std::atomic_ref<std::uint64_t>(update.words[1]).load(
                        std::memory_order_acquire);
                notifications.emplace_back(update.wait_id, actual);
            }
            for (const auto& update : entry.consumed) {
                const std::uint64_t actual =
                    std::atomic_ref<std::uint64_t>(update.words[0]).load(
                        std::memory_order_acquire);
                notifications.emplace_back(update.wait_id, actual);
            }
        }
        if (failed) {
            for (const auto& update : entry.poisoned) {
                finalize_publish_words(update.words, update.target, true);
                notifications.emplace_back(update.wait_id, update.target);
            }
        }
        if (entry.terminal_cell != nullptr) {
            entry.terminal_cell->reserved0 = 0;
            std::atomic_ref<std::uint32_t>(entry.terminal_cell->outcome).store(
                failed ? PIE_TERMINAL_OUTCOME_FAILED
                       : (retry ? PIE_TERMINAL_OUTCOME_RETRY
                                : PIE_TERMINAL_OUTCOME_SUCCESS),
                std::memory_order_release);
        }
    }
    if (notify) {
        for (const auto& [wait_id, epoch] : notifications) {
            if (wait_id != 0 && epoch != 0) {
                ctx->runtime.notify(ctx->runtime.ctx, wait_id, epoch);
            }
        }
    }
    // No native instance/channel state is touched after the batch wake: a woken
    // runtime thread may immediately close the instance.
    if (notify && ctx->completion.wait_id != 0) {
        ctx->runtime.notify(
            ctx->runtime.ctx, ctx->completion.wait_id, ctx->completion.target_epoch);
    }
}

namespace {
void close_bound_instance(Dispatch::Impl& s, std::uint64_t instance_id);

// Batch-level channel budget (§4.3 availability + reader capacity): members
// of one batch that share a channel are validated against the AGGREGATE of
// their planned ring consumes and reader publishes. Checked one-by-one, two
// members could both pass on the last available entry/slot and the second
// would die as a device-side poison instead of a synchronous rejection.
std::vector<DeviceHostChannelTicket> build_channel_tickets(
    const pie_native::LaunchView& view,
    std::size_t program,
    BoundInstance& bound,
    DeviceChannelRegistry& channels) {
    const std::size_t count = bound.trace->channels.size();
    const bool supplied =
        view.channel_ticket_indptr.size() ==
            view.ptir_program_instances.size() + 1 &&
        view.channel_expected_head.size() ==
            view.channel_expected_tail.size();
    if (!supplied) {
        throw std::runtime_error(
            "ptir launch requires runtime-assigned channel tickets");
    }
    std::size_t lo = 0;
    std::size_t hi = 0;
    lo = view.channel_ticket_indptr.data()[program];
    hi = view.channel_ticket_indptr.data()[program + 1];
    if (hi < lo || hi - lo != count ||
        hi > view.channel_expected_head.size()) {
        throw std::runtime_error(
            "ptir launch channel ticket segment does not match instance");
    }

    std::vector<DeviceHostChannelTicket> tickets;
    tickets.reserve(count);
    for (ChannelId dense = 0; dense < count; ++dense) {
        const std::uint32_t slot = bound.instance->view().slot(dense);
        const bool consumes = bound.instance->takes_channel(dense);
        const bool publishes = bound.instance->puts_channel(dense);
        std::uint64_t expected_head = kNoChannelTicket;
        std::uint64_t expected_tail = kNoChannelTicket;
        expected_head = view.channel_expected_head.data()[lo + dense];
        expected_tail = view.channel_expected_tail.data()[lo + dense];

        std::uint32_t flags = 0;
        if (consumes && expected_head != kNoChannelTicket) {
            flags |= kTicketConsume;
        }
        if (publishes && expected_tail != kNoChannelTicket) {
            flags |= kTicketPublish;
        }
        if (channels.host_role(slot) == PIE_CHANNEL_HOST_ROLE_WRITER &&
            !(channels.seed_credit(slot) && expected_head == 0)) {
            flags |= kTicketHostWriter;
        }
        if (channels.dtype(slot) == PIE_CHANNEL_DTYPE_BOOL) {
            flags |= kTicketPackedBool;
        }
        if (bound.instance->requires_channel_input(dense)) {
            flags |= kTicketRequireInput;
        }
        channels.apply_sequence_ticket(slot, expected_head, expected_tail);
        if ((flags & (kTicketConsume | kTicketPublish | kTicketRequireInput)) == 0) {
            continue;
        }
        tickets.push_back(DeviceHostChannelTicket{
            .slot = slot,
            .flags = flags,
            .expected_head = expected_head,
            .expected_tail = expected_tail,
            .words = channels.host_words(slot),
            .mirror = static_cast<const std::uint8_t*>(
                channels.host_mirror(slot)),
            .cells = static_cast<std::uint8_t*>(channels.cell_base(slot)),
            .cap1 = channels.capacity(slot) + 1,
            .wire_bytes = static_cast<std::uint32_t>(
                channels.wire_bytes(slot)),
            .native_bytes = static_cast<std::uint32_t>(
                channels.cell_bytes(slot)),
        });
    }
    return tickets;
}

const DeviceHostChannelTicket* find_publish_ticket(
    const std::vector<DeviceHostChannelTicket>& tickets,
    std::uint32_t slot) {
    auto it = std::find_if(
        tickets.begin(), tickets.end(),
        [slot](const DeviceHostChannelTicket& ticket) {
            return ticket.slot == slot &&
                   (ticket.flags & kTicketPublish) != 0;
        });
    return it == tickets.end() ? nullptr : &*it;
}

std::uint32_t stage_mtp_rows(const plan::StagePlan* stage) {
    if (stage == nullptr) return 0;
    std::uint32_t next_value = 0;
    std::uint32_t rows = 0;
    for (const auto& normalized : stage->ops) {
        const auto& op = normalized.op;
        if (op.tag == PTIR_OP_INTRINSIC_VAL &&
            op.intr == PTIR_INTR_MTP_LOGITS) {
            if (next_value >= stage->value_types.size()) {
                throw std::runtime_error(
                    "MtpLogits value is outside the region plan");
            }
            const auto& type = stage->value_types[next_value];
            if (type.dims.size() != 2 || type.dims[0].symbolic ||
                type.dims[0].value == 0) {
                throw std::runtime_error(
                    "MtpLogits requires a static non-empty draft-row extent");
            }
            if (rows != 0 && rows != type.dims[0].value) {
                throw std::runtime_error(
                    "one program declares incompatible MtpLogits row extents");
            }
            rows = type.dims[0].value;
        }
        next_value += op.results;
    }
    return rows;
}

std::uint32_t stage_logits_vocab(
    const plan::StagePlan* stage,
    std::uint32_t fallback) {
    if (stage == nullptr) return fallback;
    std::uint32_t next_value = 0;
    std::uint32_t logical_vocab = 0;
    for (const auto& normalized : stage->ops) {
        const auto& op = normalized.op;
        if (op.tag == PTIR_OP_INTRINSIC_VAL &&
            (op.intr == PTIR_INTR_LOGITS ||
             op.intr == PTIR_INTR_MTP_LOGITS)) {
            if (next_value >= stage->value_types.size() ||
                stage->value_types[next_value].dims.empty()) {
                throw std::runtime_error(
                    "logits intrinsic has no planned vocabulary dimension");
            }
            const auto& dimension =
                stage->value_types[next_value].dims.back();
            if (dimension.symbolic || dimension.value == 0) {
                throw std::runtime_error(
                    "logits vocabulary dimension must be static");
            }
            if (logical_vocab != 0 &&
                logical_vocab != dimension.value) {
                throw std::runtime_error(
                    "program declares incompatible logits vocabularies");
            }
            logical_vocab = dimension.value;
        }
        next_value += op.results;
    }
    if (logical_vocab == 0) return fallback;
    if (logical_vocab > fallback) {
        throw std::runtime_error(
            "PTIR logical vocabulary exceeds the model row stride");
    }
    return logical_vocab;
}

bool stage_uses_intrinsic(
    const plan::StagePlan& stage,
    std::uint16_t intrinsic) {
    return std::any_of(
        stage.ops.begin(), stage.ops.end(),
        [intrinsic](const plan::NormalizedOp& normalized) {
            return normalized.op.tag == PTIR_OP_INTRINSIC_VAL &&
                normalized.op.intr == intrinsic;
        });
}

std::vector<std::uint32_t> channel_alias_topology(
    const GroupedLaneBinding& lane) {
    std::vector<std::uint32_t> topology;
    topology.reserve(lane.plan->channel_bindings.size());
    std::vector<std::uint32_t> slots;
    slots.reserve(lane.plan->channel_bindings.size());
    for (std::uint32_t dense : lane.plan->channel_bindings) {
        const std::uint32_t slot = lane.instance->view().slot(dense);
        auto found = std::find(slots.begin(), slots.end(), slot);
        if (found == slots.end()) {
            topology.push_back(static_cast<std::uint32_t>(slots.size()));
            slots.push_back(slot);
        } else {
            topology.push_back(
                static_cast<std::uint32_t>(found - slots.begin()));
        }
    }
    return topology;
}

void record_stage_channel_effects(
    StagedLane& lane,
    const plan::StagePlan& stage) {
    for (const auto& normalized : stage.ops) {
        const auto& op = normalized.op;
        if (op.chan < 0 ||
            (op.tag != PTIR_OP_CHAN_TAKE &&
             op.tag != PTIR_OP_CHAN_PUT)) {
            continue;
        }
        const std::uint32_t local = static_cast<std::uint32_t>(op.chan);
        if (local >= stage.channel_bindings.size()) continue;
        const std::uint32_t slot = lane.bound->instance->view().slot(
            stage.channel_bindings[local]);
        if (op.tag == PTIR_OP_CHAN_TAKE) {
            lane.prior_take_slots.insert(slot);
        } else {
            lane.prior_put_slots.insert(slot);
        }
    }
}

__global__ void cast_query_bf16_to_f32(
    const __nv_bfloat16* source,
    float* destination,
    std::size_t count) {
    for (std::size_t index =
             blockIdx.x * static_cast<std::size_t>(blockDim.x) + threadIdx.x;
         index < count;
         index += static_cast<std::size_t>(gridDim.x) * blockDim.x) {
        destination[index] = __bfloat162float(source[index]);
    }
}

}  // namespace

Dispatch::Dispatch() : impl_(std::make_unique<Impl>()) {
    for (std::size_t index = 0; index < 2; ++index) {
        CUDA_CHECK(cudaStreamCreateWithFlags(
            &impl_->group_streams[index], cudaStreamNonBlocking));
    }
    for (cudaStream_t& stream : impl_->signature_streams) {
        CUDA_CHECK(cudaStreamCreateWithFlags(
            &stream, cudaStreamNonBlocking));
    }
}
DispatchStats Dispatch::stats() const {
    DispatchStats result;
    {
        std::lock_guard<std::mutex> lock(impl_->stats_mutex);
        result = impl_->stats;
    }
    const auto generated = impl_->fused_modules.stats();
    result.generated_compilations = generated.compilations;
    result.generated_disk_hits = generated.disk_hits;
    result.generated_disk_writes = generated.disk_writes;
    result.generated_disk_errors = generated.disk_errors;
    result.generated_negative_hits = generated.negative_hits;
    result.generated_stage_cache_entries = generated.stage_entries;
    result.generated_program_cache_entries = generated.program_entries;
    result.generated_negative_cache_entries = generated.negative_entries;
    return result;
}

std::vector<std::uint32_t> Dispatch::mtp_draft_rows(
    const pie_native::LaunchView& view) const {
    std::vector<std::uint32_t> rows(view.ptir_program_hashes.size(), 0);
    for (std::size_t program = 0;
         program < view.ptir_program_hashes.size();
         ++program) {
        const auto* plans =
            impl_->cache.plans(view.ptir_program_hashes.data()[program]);
        if (plans == nullptr) {
            throw std::runtime_error(
                "MtpLogits layout requested for an unregistered program");
        }
        for (const auto& stage : *plans) {
            const auto stage_rows = stage_mtp_rows(&stage);
            if (stage_rows == 0) continue;
            if (rows[program] != 0 && rows[program] != stage_rows) {
                throw std::runtime_error(
                    "program stages declare incompatible MtpLogits layouts");
            }
            rows[program] = stage_rows;
        }
    }
    return rows;
}

std::uint64_t Dispatch::compiled_program_set_hash(
    const pie_native::LaunchView& view) const {
    if (view.ptir_program_hashes.empty()) return 0;
    ProgramSetIdentityFold fold;
    for (std::size_t program = 0;
         program < view.ptir_program_hashes.size();
         ++program) {
        const auto* identities = impl_->cache.graph_stage_identities(
            view.ptir_program_hashes.data()[program]);
        if (identities == nullptr) return 0;
        const std::uint32_t rows =
            view.ptir_sample_counts.size() ==
                    view.ptir_program_hashes.size()
                ? view.ptir_sample_counts.data()[program]
                : view.sampling_indptr.size() ==
                    view.ptir_program_hashes.size() + 1
                ? view.sampling_indptr.data()[program + 1] -
                    view.sampling_indptr.data()[program]
                : 0;
        const std::uint8_t row_bucket = program_extent_bucket(rows);
        for (const std::uint64_t identity : *identities) {
            fold.add(identity, row_bucket);
        }
    }
    return fold.finish();
}

Dispatch::~Dispatch() {
    if (!impl_) return;
    impl_->shutting_down.store(true, std::memory_order_release);
    for (cudaStream_t stream : impl_->group_streams) {
        if (stream != nullptr) CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    for (cudaStream_t stream : impl_->signature_streams) {
        if (stream != nullptr) CUDA_CHECK(cudaStreamSynchronize(stream));
    }
    CUDA_CHECK(cudaStreamSynchronize(
        sampling_ir::FrameCarrierEngine::instance().copy_stream()));
    while (!impl_->instances.empty()) {
        close_bound_instance(*impl_, impl_->instances.begin()->first);
    }
    for (std::size_t index = 0; index < 2; ++index) {
        if (impl_->group_streams[index] != nullptr) {
            CUDA_CHECK(cudaStreamDestroy(impl_->group_streams[index]));
        }
    }
    for (cudaStream_t& stream : impl_->signature_streams) {
        if (stream != nullptr) {
            CUDA_CHECK(cudaStreamDestroy(stream));
            stream = nullptr;
        }
    }
}

namespace {

std::vector<ChannelValue> copy_seed_values(
    const std::vector<PieChannelValueDesc>& descs) {
    std::vector<ChannelValue> out;
    out.reserve(descs.size());
    for (const PieChannelValueDesc& desc : descs) {
        ChannelValue value;
        value.channel = desc.channel_id;
        if (desc.bytes.ptr != nullptr && desc.bytes.len > 0) {
            value.bytes.assign(desc.bytes.ptr, desc.bytes.ptr + desc.bytes.len);
        }
        out.push_back(std::move(value));
    }
    return out;
}

void ensure_event(cudaEvent_t* event) {
    if (*event == nullptr) {
        CUDA_CHECK(cudaEventCreateWithFlags(event, cudaEventDisableTiming));
    }
}

BoundInstance::CommitSnapshot& commit_snapshot(
    BoundInstance& bound,
    std::size_t index) {
    while (bound.commit_snapshots.size() <= index) {
        BoundInstance::CommitSnapshot snapshot;
        try {
            CUDA_CHECK(cudaMalloc(
                reinterpret_cast<void**>(&snapshot.device),
                sizeof(std::uint32_t)));
            CUDA_CHECK(cudaMallocHost(
                reinterpret_cast<void**>(&snapshot.host),
                sizeof(std::uint32_t)));
            bound.commit_snapshots.push_back(snapshot);
        } catch (...) {
            if (snapshot.host != nullptr) {
                cudaFreeHost(snapshot.host);
            }
            if (snapshot.device != nullptr) {
                cudaFree(snapshot.device);
            }
            throw;
        }
    }
    return bound.commit_snapshots[index];
}

void close_bound_instance(Dispatch::Impl& s, std::uint64_t instance_id) {
    auto it = s.instances.find(instance_id);
    if (it == s.instances.end()) return;
    if (it->second.publish_done != nullptr) {
        CUDA_CHECK(cudaEventSynchronize(it->second.publish_done));
        CUDA_CHECK(cudaEventDestroy(it->second.publish_done));
    }
    for (const BoundInstance::CommitSnapshot& snapshot :
         it->second.commit_snapshots) {
        if (snapshot.device != nullptr) {
            CUDA_CHECK(cudaFree(snapshot.device));
        }
        if (snapshot.host != nullptr) {
            CUDA_CHECK(cudaFreeHost(snapshot.host));
        }
    }
    s.instances.erase(it);
}

}  // namespace

int Dispatch::register_program(std::uint64_t program_hash,
                                   pie_native::ByteSlice canonical,
                                   pie_native::ByteSlice sidecar,
                                   std::string* err) {
    if (err) err->clear();
    std::string derr;
    const Trace* trace = impl_->cache.get_or_decode(
        program_hash,
        reinterpret_cast<const std::uint8_t*>(canonical.ptr), canonical.size(),
        reinterpret_cast<const std::uint8_t*>(sidecar.ptr), sidecar.size(), &derr);
    if (trace == nullptr) {
        if (err) *err = derr;
        return PIE_STATUS_DRIVER_ERROR;
    }
    for (const Channel& channel : trace->channels) {
        const std::size_t cell_bytes =
            channel.type.shape.numel() * dtype_size(channel.type.dtype);
        if (channel.capacity >= kMaxRing ||
            cell_bytes == 0 ||
            cell_bytes > std::numeric_limits<std::uint32_t>::max()) {
            if (err) *err = "ptir program has an unsupported channel declaration";
            return PIE_STATUS_INVALID_ARGUMENT;
        }
    }
    const auto* plans = impl_->cache.plans(program_hash);
    if (plans == nullptr || plans->empty()) {
        if (err) *err = "ptir program has no compiler region plans";
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    for (const plan::StagePlan& stage : *plans) {
        if ((stage.stage == PTIR_STAGE_ON_ATTN_PROJ ||
             stage.stage == PTIR_STAGE_ON_ATTN) &&
            !impl_->attention_hook_coverage) {
            if (err) {
                *err =
                    "active CUDA model does not implement PTIR attention hooks";
            }
            return PIE_STATUS_UNSUPPORTED;
        }
        for (const plan::NormalizedOp& normalized : stage.ops) {
            const auto& op = normalized.op;
            if (op.tag == PTIR_OP_SINK_CALL) {
                if (err) {
                    *err =
                        "ptir model sinks are not implemented by the active CUDA model";
                }
                return PIE_STATUS_UNSUPPORTED;
            }
            if (!grouped_supported_tag(op.tag)) {
                if (err) {
                    *err =
                        "ptir region plan contains an unsupported generic CUDA op";
                }
                return PIE_STATUS_UNSUPPORTED;
            }
            if (op.tag != PTIR_OP_INTRINSIC_VAL) continue;
            const bool valid =
                (stage.stage == PTIR_STAGE_EPILOGUE &&
                 (op.intr == PTIR_INTR_LOGITS ||
                  op.intr == PTIR_INTR_MTP_LOGITS)) ||
                ((stage.stage == PTIR_STAGE_ON_ATTN_PROJ ||
                  stage.stage == PTIR_STAGE_ON_ATTN) &&
                 (op.intr == PTIR_INTR_QUERY ||
                  op.intr == PTIR_INTR_LAYER));
            if (!valid) {
                if (err) {
                    *err =
                        "ptir intrinsic is unavailable at its declared CUDA phase";
                }
                return PIE_STATUS_UNSUPPORTED;
            }
        }
    }
    generated::CompileFailureKind compile_failure =
        generated::CompileFailureKind::None;
    std::string compile_error;
    const auto compiled_program = impl_->fused_modules.compile_program(
            program_hash,
            *plans,
            compile_failure,
            compile_error);
    if (compiled_program == nullptr) {
        if (err) *err = std::move(compile_error);
        return compile_failure == generated::CompileFailureKind::Deterministic
            ? PIE_STATUS_UNSUPPORTED
            : PIE_STATUS_DRIVER_ERROR;
    }
    if (compiled_program->stages.size() != plans->size()) {
        if (err) *err = "CUDA fused program stage count mismatch";
        return PIE_STATUS_UNSUPPORTED;
    }
    for (std::size_t stage_index = 0;
         stage_index < plans->size();
         ++stage_index) {
        std::string availability_error;
        if (compiled_program->stages[stage_index] == nullptr ||
            !generated::generated_stage_supported(
                *compiled_program->stages[stage_index],
                (*plans)[stage_index],
                &availability_error)) {
            if (err) {
                *err =
                    "CUDA fused registration lacks complete coverage: " +
                    availability_error;
            }
            return PIE_STATUS_UNSUPPORTED;
        }
    }
    return PIE_STATUS_OK;
}

int Dispatch::register_channel(
    const PieChannelDesc& channel,
    PieChannelEndpointBinding* binding,
    std::string* err) {
    if (err) err->clear();
    return impl_->channels.register_endpoint(channel, binding, err)
        ? PIE_STATUS_OK
        : PIE_STATUS_INVALID_ARGUMENT;
}

int Dispatch::bind_instance(std::uint64_t instance_id,
                                std::uint64_t program_hash,
                                std::uint64_t pacing_wait_id,
                                const std::vector<std::uint64_t>& channel_ids,
                                const std::vector<PieChannelValueDesc>& seed_values,
                                PieInstanceBinding* binding,
                                std::string* err) {
    if (err) err->clear();
    std::string derr;
    const Trace* trace = impl_->cache.get_or_decode(
        program_hash, nullptr, 0, nullptr, 0, &derr);
    if (trace == nullptr) {
        if (err) *err = derr;
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    if (channel_ids.size() != trace->channels.size()) {
        if (err) *err = "ptir instance channel count does not match program";
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    if (impl_->instances.find(instance_id) != impl_->instances.end()) {
        if (err) *err = "ptir instance id is already bound";
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    {
        std::unordered_set<std::uint64_t> unique_ids(
            channel_ids.begin(), channel_ids.end());
        if (unique_ids.size() != channel_ids.size()) {
            if (err) *err = "ptir instance channel ids must be unique";
            return PIE_STATUS_INVALID_ARGUMENT;
        }
    }
    std::string ierr;
    auto inst = std::make_unique<PtirInstance>(
        *trace, &impl_->channels, channel_ids, copy_seed_values(seed_values), &ierr);
    if (!inst->ok()) {
        if (err) *err = ierr;
        return PIE_STATUS_INVALID_ARGUMENT;
    }

    BoundInstance bound;
    bound.program_hash = program_hash;
    bound.pacing_wait_id = pacing_wait_id;
    bound.trace = trace;
    bound.channel_ids = channel_ids;
    bound.instance = std::move(inst);
    ensure_event(&bound.publish_done);
    CUDA_CHECK(cudaEventRecord(
        bound.publish_done, sampling_ir::FrameCarrierEngine::instance().copy_stream()));

    if (binding != nullptr) {
        std::memset(binding, 0, sizeof(*binding));
        binding->instance_id = instance_id;
    }
    impl_->instances.emplace(instance_id, std::move(bound));
    return PIE_STATUS_OK;
}

void Dispatch::close_instance(std::uint64_t instance_id) {
    close_bound_instance(*impl_, instance_id);
}

int Dispatch::close_channel(std::uint64_t channel_id, std::string* err) {
    if (err) err->clear();
    return impl_->channels.close_endpoint(channel_id, err)
        ? PIE_STATUS_OK
        : (impl_->channels.contains(channel_id)
               ? PIE_STATUS_INVALID_ARGUMENT
               : PIE_STATUS_CLOSED);
}

int Dispatch::validate_launch(
    const pie_native::LaunchView& view,
    std::string* err) {
    if (err) err->clear();
    const std::size_t count = view.ptir_program_hashes.size();
    if (count == 0 ||
        view.ptir_program_instances.size() != count ||
        view.terminal_cells.size() != count) {
        if (err) *err = "ptir launch has inconsistent program arrays";
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    try {
        for (std::size_t program = 0; program < count; ++program) {
            const std::uint64_t instance_id =
                view.ptir_program_instances.data()[program];
            auto instance = impl_->instances.find(instance_id);
            if (instance == impl_->instances.end() ||
                instance->second.trace == nullptr ||
                instance->second.program_hash !=
                    view.ptir_program_hashes.data()[program]) {
                if (err) *err = "ptir launch references an incompatible instance";
                return PIE_STATUS_INVALID_ARGUMENT;
            }
        }
    } catch (const std::exception& error) {
        if (err) *err = error.what();
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    return PIE_STATUS_OK;
}

void Dispatch::set_attention_hook_coverage(
    bool supported,
    std::uint32_t model_layers) {
    impl_->attention_hook_coverage = supported;
    impl_->model_layers = supported ? model_layers : 0;
}

bool Dispatch::launch_has_attention_stages(
    const pie_native::LaunchView& view) const {
    for (std::size_t program = 0;
         program < view.ptir_program_hashes.size();
         ++program) {
        const auto* plans =
            impl_->cache.plans(view.ptir_program_hashes.data()[program]);
        if (plans == nullptr) continue;
        if (std::any_of(
                plans->begin(), plans->end(),
                [](const plan::StagePlan& stage) {
                    return stage.stage == PTIR_STAGE_ON_ATTN_PROJ ||
                        stage.stage == PTIR_STAGE_ON_ATTN;
                })) {
            return true;
        }
    }
    return false;
}

namespace {

GroupedLaneBinding make_staged_binding(
    StagedLane& lane,
    const plan::StagePlan& stage,
    const float* logits_base,
    std::uint32_t logits_stride,
    const float* query_base,
    std::uint32_t query_columns,
    const std::uint32_t* layer_base) {
    const float* lane_query = nullptr;
    if (query_base != nullptr) {
        lane_query = query_base +
            static_cast<std::size_t>(lane.token_start) * query_columns;
    }
    return GroupedLaneBinding{
        .instance = lane.bound->instance.get(),
        .plan = &stage,
        .plan_identity = lane.plan_identities->at(
            static_cast<std::size_t>(&stage - lane.plans->data())),
        .tickets = &lane.tickets,
        .logits_base = logits_base,
        .query_base = lane_query,
        .layer_base = layer_base,
        .logits_bf16_rows = lane.logits_bf16_rows.empty()
            ? nullptr
            : &lane.logits_bf16_rows,
        .mtp_logits_bf16_rows = lane.mtp_logits_bf16_rows.empty()
            ? nullptr
            : &lane.mtp_logits_bf16_rows,
        .prior_put_slots = &lane.prior_put_slots,
        .prior_take_slots = &lane.prior_take_slots,
        .commit_slot = lane.snapshot->device,
        .logits_row_offset = lane.row_offset,
        .logits_row_count = lane.sampled_rows,
        .row_count = lane.runtime_row_count,
        .token_count = lane.token_count,
        .kv_len = lane.kv_len,
        .page_count = lane.page_count,
        .query_len = lane.query_len,
        .key_len = lane.key_len,
        .vocab = lane.logical_vocab,
        .logits_stride = logits_stride,
        .program_index = static_cast<std::uint32_t>(lane.program),
    };
}

void execute_declared_phase(
    StagedLaunch::State& launch,
    std::uint8_t phase,
    const float* logits_base,
    std::uint32_t logits_stride,
    const float* query_base,
    std::uint32_t query_rows,
    std::uint32_t query_columns,
    std::uint32_t layer,
    cudaStream_t stream) {
    if (!launch.active || launch.failed) {
        throw std::runtime_error("PTIR staged launch is not active");
    }
    if (phase > PTIR_STAGE_EPILOGUE) {
        throw std::runtime_error("invalid PTIR execution phase");
    }
    if ((phase == PTIR_STAGE_ON_ATTN_PROJ ||
         phase == PTIR_STAGE_ON_ATTN) &&
        layer != launch.phase_invocations[phase]) {
        throw std::runtime_error(
            "PTIR model hook layer order is not exact");
    }
    ++launch.phase_invocations[phase];
    launch.stream = stream;
    const cudaStream_t source_stream = stream;
    const std::size_t bridge_index = phase % 2;
    cudaStream_t execution_stream =
        launch.owner->group_streams[bridge_index];
    const bool bridged =
        execution_stream != nullptr && execution_stream != source_stream;
    if (bridged) {
        CUDA_CHECK(cudaEventRecord(
            launch.source_ready, source_stream));
        CUDA_CHECK(cudaStreamWaitEvent(
            execution_stream, launch.source_ready, 0));
        stream = execution_stream;
    }
    struct StreamBridge {
        cudaEvent_t done = nullptr;
        cudaStream_t source = nullptr;
        cudaStream_t execution = nullptr;
        ~StreamBridge() {
            if (done == nullptr) return;
            cudaEventRecord(done, execution);
            cudaStreamWaitEvent(source, done, 0);
        }
    } bridge{
        bridged ? launch.phase_done[bridge_index] : nullptr,
        source_stream,
        execution_stream,
    };
    if (phase == PTIR_STAGE_ON_ATTN_PROJ ||
        phase == PTIR_STAGE_ON_ATTN) {
        CUDA_CHECK(cudaMemcpyAsync(
            launch.device_layer, &layer, sizeof(layer),
            cudaMemcpyHostToDevice, stream));
    }

    std::size_t max_occurrences = 0;
    for (const auto& lane : launch.lanes) {
        max_occurrences = std::max(
            max_occurrences, lane->phase_plans[phase].size());
    }
    for (std::size_t occurrence = 0;
         occurrence < max_occurrences;
         ++occurrence) {
        struct Task {
            StagedLane* lane = nullptr;
            const plan::StagePlan* plan = nullptr;
            const generated::FusedStageExecutable* executable = nullptr;
            GroupedLaneBinding binding;
            bool complete = false;
        };
        std::vector<Task> tasks;
        for (auto& lane_ptr : launch.lanes) {
            StagedLane& lane = *lane_ptr;
            if (occurrence >= lane.phase_plans[phase].size()) continue;
            const plan::StagePlan& stage =
                *lane.phase_plans[phase][occurrence];
            if (stage.ops.empty()) continue;
            const std::size_t stage_index =
                static_cast<std::size_t>(&stage - lane.plans->data());
            if (lane.generated_program == nullptr ||
                stage_index >= lane.generated_program->stages.size()) {
                throw std::runtime_error(
                    "PTIR staged launch has no compiled fused stage");
            }
            if (stage_uses_intrinsic(stage, PTIR_INTR_QUERY)) {
                if (query_base == nullptr || query_columns == 0 ||
                    lane.token_count == kUnavailableGroupedExtent ||
                    lane.token_start > query_rows ||
                    lane.token_count > query_rows - lane.token_start) {
                    throw std::runtime_error(
                        "Query intrinsic is outside the current model query span");
                }
            }
            GroupedLaneBinding binding = make_staged_binding(
                lane, stage, logits_base, logits_stride,
                query_base, query_columns, launch.device_layer);
            std::uint32_t value_base = 0;
            for (const auto& normalized : stage.ops) {
                if (normalized.op.tag == PTIR_OP_INTRINSIC_VAL &&
                    normalized.op.intr == PTIR_INTR_QUERY) {
                    if (value_base >= stage.value_types.size() ||
                        grouped_numel(
                            stage.value_types[value_base], binding) >
                            static_cast<std::uint64_t>(lane.token_count) *
                                query_columns) {
                        throw std::runtime_error(
                            "Query intrinsic shape exceeds the current "
                            "program query tensor");
                    }
                }
                value_base += normalized.op.results;
            }
            tasks.push_back(Task{
                .lane = &lane,
                .plan = &stage,
                .executable =
                    lane.generated_program->stages[stage_index].get(),
                .binding = binding,
            });
        }

        struct ExecutionGroup {
            Task* first = nullptr;
            std::vector<Task*> members;
            std::vector<GroupedLaneBinding> bindings;
        };
        std::vector<ExecutionGroup> groups;
        for (std::size_t first_index = 0;
             first_index < tasks.size();
             ++first_index) {
            if (tasks[first_index].complete) continue;
            Task& first = tasks[first_index];
            std::vector<Task*> members{&first};
            std::vector<GroupedLaneBinding> bindings{first.binding};
            const auto topology = channel_alias_topology(first.binding);
            std::string reason;
            if (!grouped_stage_supported(bindings, &reason)) {
                throw std::runtime_error(
                    "PTRP stage is not executable by the generic CUDA backend: " +
                    reason);
            }
            for (std::size_t candidate = first_index + 1;
                 candidate < tasks.size();
                 ++candidate) {
                Task& next = tasks[candidate];
                if (next.complete ||
                    next.plan->signature_hash !=
                        first.plan->signature_hash ||
                    next.plan->signature != first.plan->signature ||
                    channel_alias_topology(next.binding) != topology) {
                    continue;
                }
                auto proposed = bindings;
                proposed.push_back(next.binding);
                reason.clear();
                if (!grouped_stage_supported(proposed, &reason)) {
                    if (reason.find("shared") != std::string::npos) {
                        std::lock_guard<std::mutex> lock(
                            launch.owner->stats_mutex);
                        ++launch.owner->stats.shared_slot_exclusions;
                        ++launch.owner->stats.ordered_alias_launches;
                    }
                    continue;
                }
                bindings = std::move(proposed);
                members.push_back(&next);
            }
            for (Task* member : members) member->complete = true;
            groups.push_back(ExecutionGroup{
                .first = &first,
                .members = std::move(members),
                .bindings = std::move(bindings),
            });
        }

        const GroupedExecutionOptions execution_options{
            .reset_commits = false,
            .pull_tickets = false,
            .finalize = false,
        };
        auto execute_group = [&](ExecutionGroup& group,
                                 cudaStream_t target_stream) {
            Task& first = *group.first;
            std::string generated_reason;
            if (first.executable == nullptr ||
                !generated::generated_stage_supported(
                    *first.executable,
                    *first.plan,
                    &generated_reason)) {
                throw std::runtime_error(
                    "registered PTIR stage has no generated execution: " +
                    generated_reason);
            }
            GroupedLaunchResult result =
                generated::run_generated_stage(
                    group.bindings,
                    *first.executable,
                    target_stream,
                    execution_options);
            if (result.device_tickets != nullptr) {
                CUDA_CHECK(cudaFreeAsync(
                    result.device_tickets, target_stream));
            }
            const bool direct_bf16 = std::any_of(
                group.bindings.begin(), group.bindings.end(),
                [](const GroupedLaneBinding& binding) {
                    return binding.logits_bf16_rows != nullptr ||
                        binding.mtp_logits_bf16_rows != nullptr;
                });
            {
                std::lock_guard<std::mutex> lock(
                    launch.owner->stats_mutex);
                ++launch.owner->stats.generated_fused_groups;
                launch.owner->stats.generated_fused_body_launches +=
                    result.body_op_launches;
                launch.owner->stats.grouped_lanes +=
                    group.members.size();
                launch.owner->stats.grouped_body_op_launches +=
                    result.body_op_launches;
                if (direct_bf16) {
                    ++launch.owner->stats.direct_bf16_groups;
                }
                if (result.used_nucleus_library) {
                    ++launch.owner->stats.nucleus_library_groups;
                }
                if (result.used_selection_library) {
                    ++launch.owner->stats.selection_library_groups;
                }
                if (result.large_nucleus_scalable) {
                    ++launch.owner->stats.large_nucleus_scalable_groups;
                }
            }
            for (Task* member : group.members) {
                record_stage_channel_effects(
                    *member->lane, *member->plan);
            }
        };

        bool independent = groups.size() > 1;
        std::unordered_set<std::uint32_t> prior_group_slots;
        for (const auto& group : groups) {
            std::unordered_set<std::uint32_t> group_slots;
            for (const auto& binding : group.bindings) {
                group_slots.insert(
                    binding.instance->view().slots().begin(),
                    binding.instance->view().slots().end());
            }
            for (const std::uint32_t slot : group_slots) {
                if (prior_group_slots.contains(slot)) {
                    independent = false;
                }
            }
            prior_group_slots.insert(
                group_slots.begin(), group_slots.end());
        }
        if (!independent) {
            for (auto& group : groups) execute_group(group, stream);
            continue;
        }

        ensure_event(&launch.signature_ready);
        CUDA_CHECK(cudaEventRecord(launch.signature_ready, stream));
        const std::size_t used_streams = std::min(
            groups.size(),
            Dispatch::Impl::kSignatureStreamCount);
        for (std::size_t index = 0; index < used_streams; ++index) {
            ensure_event(&launch.signature_done[index]);
            CUDA_CHECK(cudaStreamWaitEvent(
                launch.owner->signature_streams[index],
                launch.signature_ready,
                0));
        }
        struct SignatureStreamJoin {
            StagedLaunch::State& launch;
            cudaStream_t source;
            std::size_t count;
            ~SignatureStreamJoin() {
                for (std::size_t index = 0; index < count; ++index) {
                    const cudaError_t record_status = cudaEventRecord(
                        launch.signature_done[index],
                        launch.owner->signature_streams[index]);
                    const cudaError_t wait_status =
                        record_status == cudaSuccess
                        ? cudaStreamWaitEvent(
                              source,
                              launch.signature_done[index],
                              0)
                        : record_status;
                    if (wait_status != cudaSuccess) {
                        std::fprintf(
                            stderr,
                            "[pie-driver-cuda] failed to rejoin PTIR "
                            "signature stream: %s\n",
                            cudaGetErrorString(wait_status));
                    }
                }
            }
        } signature_join{launch, stream, used_streams};
        for (std::size_t index = 0; index < groups.size(); ++index) {
            execute_group(
                groups[index],
                launch.owner->signature_streams[
                    index % used_streams]);
        }
        {
            std::lock_guard<std::mutex> lock(
                launch.owner->stats_mutex);
            launch.owner->stats.overlapped_groups += groups.size();
        }
    }
}

}  // namespace

std::unique_ptr<StagedLaunch> Dispatch::begin(
    const pie_native::LaunchView& view,
    cudaStream_t stream) {
    std::string validation_error;
    const int status = validate_launch(view, &validation_error);
    if (status != PIE_STATUS_OK) {
        throw std::runtime_error(
            validation_error.empty()
                ? "invalid PTIR launch"
                : validation_error);
    }
    auto launch = std::unique_ptr<StagedLaunch>(new StagedLaunch());
    StagedLaunch::State& state = *launch->state_;
    state.owner = impl_.get();
    state.view = view;
    state.stream = stream;
    CUDA_CHECK(cudaMallocAsync(
        reinterpret_cast<void**>(&state.device_layer),
        sizeof(std::uint32_t),
        stream));
    CUDA_CHECK(cudaEventCreateWithFlags(
        &state.source_ready, cudaEventDisableTiming));
    for (cudaEvent_t& event : state.phase_done) {
        CUDA_CHECK(cudaEventCreateWithFlags(
            &event, cudaEventDisableTiming));
    }
    const std::size_t count = view.ptir_program_hashes.size();
    std::unordered_map<std::uint64_t, std::size_t> fire_counts;
    state.lanes.reserve(count);
    for (std::size_t program = 0; program < count; ++program) {
        const std::uint64_t instance_id =
            view.ptir_program_instances.data()[program];
        auto found = impl_->instances.find(instance_id);
        if (found == impl_->instances.end()) {
            throw std::runtime_error("PTIR launch references a missing instance");
        }
        BoundInstance& bound = found->second;
        auto lane = std::make_unique<StagedLane>();
        const std::size_t instance_occurrence = fire_counts[instance_id]++;
        lane->program = program;
        lane->instance_id = instance_id;
        lane->bound = &bound;
        lane->snapshot =
            &commit_snapshot(bound, instance_occurrence);
        lane->plans = impl_->cache.plans(bound.program_hash);
        lane->plan_identities =
            impl_->cache.graph_stage_identities(bound.program_hash);
        lane->generated_program =
            impl_->fused_modules.program(bound.program_hash);
        if (lane->plans == nullptr || lane->plan_identities == nullptr ||
            lane->plan_identities->size() != lane->plans->size() ||
            lane->generated_program == nullptr ||
            lane->generated_program->stages.size() !=
                lane->plans->size()) {
            throw std::runtime_error("PTIR launch has no compiler region plans");
        }
        for (const plan::StagePlan& stage : *lane->plans) {
            if (stage.stage > PTIR_STAGE_EPILOGUE) {
                throw std::runtime_error("PTIR plan has an invalid phase");
            }
            lane->phase_plans[stage.stage].push_back(&stage);
        }
        ensure_event(&bound.publish_done);
        CUDA_CHECK(cudaStreamWaitEvent(stream, bound.publish_done, 0));
        const std::uint32_t initial_commit =
            instance_occurrence == 0 ? 1u : 0u;
        CUDA_CHECK(cudaMemcpyAsync(
            lane->snapshot->device, &initial_commit, sizeof(initial_commit),
            cudaMemcpyHostToDevice, stream));
        lane->tickets =
            build_channel_tickets(view, program, bound, impl_->channels);
        if (impl_->force_retry_launches_remaining.exchange(
                0, std::memory_order_relaxed) != 0) {
            bool forced = false;
            for (DeviceHostChannelTicket& ticket : lane->tickets) {
                if ((ticket.flags & kTicketConsume) != 0) {
                    ++ticket.expected_head;
                    forced = true;
                    break;
                }
                if ((ticket.flags & kTicketPublish) != 0) {
                    ++ticket.expected_tail;
                    forced = true;
                    break;
                }
            }
            if (!forced) {
                impl_->force_retry_launches_remaining.store(
                    1, std::memory_order_relaxed);
            }
        }
        lane->device_tickets = launch_pull_validate_host_channels(
            lane->tickets,
            bound.instance->view().d_full(),
            lane->snapshot->device,
            stream);
        lane->device_ticket_count =
            static_cast<std::uint32_t>(lane->tickets.size());
        state.touched_instances.push_back(instance_id);
        state.lanes.push_back(std::move(lane));
    }
    const bool stateful_rs = rs_launch_requires_readiness_settlement(
        view.rs_slot_ids.size(),
        view.rs_fold_lens.size(),
        view.rs_buffer_slot_ids.size(),
        view.rs_buffer_slot_indptr.size());
    auto settle_readiness = [&](const char* phase) {
        for (const auto& lane : state.lanes) {
            CUDA_CHECK(cudaMemcpyAsync(
                lane->snapshot->host,
                lane->snapshot->device,
                sizeof(std::uint32_t),
                cudaMemcpyDeviceToHost,
                stream));
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        for (const auto& lane : state.lanes) {
            if (*lane->snapshot->host == 0) {
                throw RetryableLaunchError(
                    std::string("ptir ") + phase +
                    " readiness did not commit");
            }
        }
    };
    try {
        // Stateful model launches cannot discover a ticket miss after the
        // recurrent-state kernels have already mutated their slots. Settle the
        // host/device ticket pull before Prologue, then settle Prologue's own
        // channel readiness before returning to the model forward.
        if (stateful_rs) settle_readiness("channel ticket");
        execute_declared_phase(
            state, PTIR_STAGE_PROLOGUE,
            nullptr, 0, nullptr, 0, 0, 0, stream);
        if (stateful_rs) settle_readiness("prologue");
    } catch (...) {
        abort(*launch, stream);
        throw;
    }
    return launch;
}

void Dispatch::update_launch_geometry(
    StagedLaunch& launch,
    const pie_native::LaunchView& resolved_view,
    std::span<const std::uint32_t> program_token_starts) {
    StagedLaunch::State& state = *launch.state_;
    if (!state.active ||
        resolved_view.ptir_program_hashes.size() != state.lanes.size() ||
        program_token_starts.size() != state.lanes.size()) {
        throw std::runtime_error("invalid staged PTIR geometry update");
    }
    state.view = resolved_view;
    const std::size_t count = state.lanes.size();
    auto extent = [&](const pie_native::Slice<std::uint32_t>& values,
                      std::size_t program) {
        return values.size() == count
            ? values.data()[program]
            : kUnavailableGroupedExtent;
    };
    for (std::size_t program = 0; program < count; ++program) {
        StagedLane& lane = *state.lanes[program];
        lane.token_start = program_token_starts[program];
        if (resolved_view.ptir_sample_starts.size() == count &&
            resolved_view.ptir_sample_counts.size() == count) {
            lane.row_offset =
                resolved_view.ptir_sample_starts.data()[program];
            lane.sampled_rows =
                resolved_view.ptir_sample_counts.data()[program];
        } else if (resolved_view.sampling_indptr.size() == count + 1) {
            lane.row_offset =
                resolved_view.sampling_indptr.data()[program];
            lane.sampled_rows =
                resolved_view.sampling_indptr.data()[program + 1] -
                lane.row_offset;
        }
        lane.runtime_row_count =
            extent(resolved_view.ptir_row_counts, program);
        lane.token_count =
            extent(resolved_view.ptir_token_counts, program);
        lane.kv_len = extent(resolved_view.ptir_kv_lens, program);
        lane.page_count =
            extent(resolved_view.ptir_page_counts, program);
        lane.query_len =
            extent(resolved_view.ptir_query_lens, program);
        lane.key_len =
            extent(resolved_view.ptir_key_lens, program);
        for (const PortBinding& binding : lane.bound->trace->ports) {
            if (binding.is_const || !port_consumes(binding.port)) continue;
            lane.prior_take_slots.insert(
                lane.bound->instance->view().slot(binding.channel));
        }
    }
}

void Dispatch::execute_attention_phase(
    StagedLaunch& launch,
    std::uint8_t phase,
    const void* query_data,
    std::uint32_t query_rows,
    std::uint32_t query_columns,
    std::uint32_t layer,
    cudaStream_t stream,
    bool query_is_f32) {
    if (phase != PTIR_STAGE_ON_ATTN_PROJ &&
        phase != PTIR_STAGE_ON_ATTN) {
        throw std::runtime_error("model hook invoked a non-attention PTIR phase");
    }
    StagedLaunch::State& state = *launch.state_;
    bool needs_query = false;
    for (const auto& lane : state.lanes) {
        for (const plan::StagePlan* stage : lane->phase_plans[phase]) {
            needs_query =
                needs_query || stage_uses_intrinsic(*stage, PTIR_INTR_QUERY);
        }
    }
    float* query_f32 = nullptr;
    if (needs_query) {
        if (query_data == nullptr || query_rows == 0 || query_columns == 0 ||
            static_cast<std::size_t>(query_rows) >
                std::numeric_limits<std::size_t>::max() / query_columns) {
            throw std::runtime_error("model hook has no valid Query tensor");
        }
        const std::size_t count =
            static_cast<std::size_t>(query_rows) * query_columns;
        if (query_is_f32) {
            query_f32 = const_cast<float*>(
                static_cast<const float*>(query_data));
        } else {
            CUDA_CHECK(cudaMallocAsync(
                reinterpret_cast<void**>(&query_f32),
                count * sizeof(float), stream));
            const std::uint32_t blocks = static_cast<std::uint32_t>(
                std::min<std::size_t>((count + 255) / 256, 65535));
            cast_query_bf16_to_f32<<<blocks, 256, 0, stream>>>(
                static_cast<const __nv_bfloat16*>(query_data),
                query_f32,
                count);
            CUDA_CHECK(cudaGetLastError());
        }
    }
    try {
        execute_declared_phase(
            state, phase, nullptr, 0, query_f32,
            query_rows, query_columns, layer, stream);
    } catch (...) {
        if (query_f32 != nullptr && !query_is_f32) {
            cudaFreeAsync(query_f32, stream);
        }
        state.failed = true;
        throw;
    }
    if (query_f32 != nullptr && !query_is_f32) {
        CUDA_CHECK(cudaFreeAsync(query_f32, stream));
    }
}

bool Dispatch::finish(
    StagedLaunch& launch,
    const pie_native::LaunchView& view,
    const void* logits,
    std::uint32_t vocab,
    cudaStream_t stream,
    const PieRuntimeCallbacks* runtime,
    PieCompletion completion,
    const std::uint16_t* direct_bf16_logits,
    const std::uint32_t* direct_row_indices,
    std::span<const std::uint32_t> mtp_draft_row_starts,
    std::span<const std::uint32_t> mtp_draft_row_counts,
    std::uint32_t direct_bf16_row_capacity) {
    StagedLaunch::State& state = *launch.state_;
    if (!state.active || state.failed ||
        state.lanes.size() != view.ptir_program_hashes.size()) {
        throw std::runtime_error("invalid PTIR staged finish");
    }
    state.view = view;
    state.stream = stream;
    const std::size_t program_count = state.lanes.size();
    for (std::uint8_t phase :
         {std::uint8_t{PTIR_STAGE_ON_ATTN_PROJ},
          std::uint8_t{PTIR_STAGE_ON_ATTN}}) {
        const bool declared = std::any_of(
            state.lanes.begin(), state.lanes.end(),
            [phase](const auto& lane) {
                return !lane->phase_plans[phase].empty();
            });
        if (declared &&
            state.phase_invocations[phase] != impl_->model_layers) {
            throw std::runtime_error(
                "PTIR attention phase did not execute at every model layer");
        }
    }
    for (std::size_t program = 0; program < program_count; ++program) {
        StagedLane& lane = *state.lanes[program];
        std::uint32_t logical_vocab = 0;
        std::uint32_t drafts = 0;
        for (const plan::StagePlan* stage :
             lane.phase_plans[PTIR_STAGE_EPILOGUE]) {
            const std::uint32_t stage_vocab =
                stage_logits_vocab(stage, vocab);
            if (logical_vocab != 0 && stage_vocab != logical_vocab) {
                throw std::runtime_error(
                    "epilogue plans declare incompatible vocabularies");
            }
            logical_vocab = stage_vocab;
            const std::uint32_t stage_drafts = stage_mtp_rows(stage);
            if (stage_drafts != 0 && drafts != 0 &&
                stage_drafts != drafts) {
                throw std::runtime_error(
                    "epilogue plans declare incompatible MtpLogits rows");
            }
            drafts = std::max(drafts, stage_drafts);
        }
        lane.logical_vocab = logical_vocab == 0 ? vocab : logical_vocab;
        lane.logits_bf16_rows.clear();
        lane.mtp_logits_bf16_rows.clear();
        if (direct_bf16_logits != nullptr &&
            direct_row_indices != nullptr) {
            lane.logits_bf16_rows.reserve(lane.sampled_rows);
            for (std::uint32_t row = 0; row < lane.sampled_rows; ++row) {
                const std::uint32_t source =
                    direct_row_indices[lane.row_offset + row];
                if (direct_bf16_row_capacity != 0 &&
                    source >= direct_bf16_row_capacity) {
                    throw std::runtime_error(
                        "direct BF16 sampled row exceeds the logits layout");
                }
                lane.logits_bf16_rows.push_back(
                    reinterpret_cast<std::uint64_t>(
                        direct_bf16_logits +
                        static_cast<std::size_t>(source) * vocab));
            }
        }
        if (drafts != 0) {
            if (mtp_draft_row_starts.size() != program_count ||
                mtp_draft_row_counts.size() != program_count ||
                mtp_draft_row_counts[program] != drafts) {
                throw std::runtime_error(
                    "MtpLogits dedicated rows are unavailable");
            }
            const std::uint32_t start =
                mtp_draft_row_starts[program];
            if (start > direct_bf16_row_capacity ||
                drafts > direct_bf16_row_capacity - start) {
                throw std::runtime_error(
                    "MtpLogits dedicated rows exceed the logits layout");
            }
            if (direct_bf16_logits == nullptr) {
                throw std::runtime_error(
                    "generic staged MtpLogits requires direct BF16 rows");
            }
            lane.mtp_logits_bf16_rows.reserve(drafts);
            for (std::uint32_t row = 0; row < drafts; ++row) {
                lane.mtp_logits_bf16_rows.push_back(
                    reinterpret_cast<std::uint64_t>(
                        direct_bf16_logits +
                        static_cast<std::size_t>(start + row) * vocab));
            }
        }
    }

    try {
        execute_declared_phase(
            state,
            PTIR_STAGE_EPILOGUE,
            static_cast<const float*>(logits),
            vocab,
            nullptr,
            0,
            0,
            0,
            stream);
    } catch (...) {
        state.failed = true;
        throw;
    }

    cudaStream_t callback_stream = stream;
    auto notify = std::make_unique<NotifyContext>();
    if (runtime != nullptr) notify->runtime = *runtime;
    notify->completion = completion;
    notify->impl = impl_.get();
    notify->entries.reserve(program_count);
    const bool batch_settlement =
        std::getenv("PIE_DISABLE_BATCH_SETTLEMENT") == nullptr;
    notify->commit_lanes.reserve(program_count);
    for (auto& lane_ptr : state.lanes) {
        StagedLane& lane = *lane_ptr;
        PtirInstance& instance = *lane.bound->instance;
        if (!batch_settlement) {
            instance.finalize_commit(stream, lane.snapshot->device);
            continue;
        }
        ChannelView& channel_view = instance.view();
        notify->commit_lanes.push_back(CommitBumpLane{
            .full = channel_view.d_full(),
            .head = channel_view.d_head(),
            .tail = channel_view.d_tail(),
            .cap1 = channel_view.d_cap1(),
            .taken = instance.commit_taken_device(),
            .taken_count = instance.commit_taken_count(),
            .put = instance.commit_put_device(),
            .put_count = instance.commit_put_count(),
            .commit = lane.snapshot->device,
        });
    }
    if (batch_settlement) {
        launch_commit_bump_batch(notify->commit_lanes, stream);
    }
    notify->settlement_lanes.reserve(program_count);
    for (auto& lane_ptr : state.lanes) {
        StagedLane& lane = *lane_ptr;
        BoundInstance& bound = *lane.bound;

        auto outputs = bound.instance->predict_outputs_device();
        std::vector<std::uint32_t> output_slots;
        std::vector<NotifyContext::FinalizeEntry::EndpointUpdate> published;
        output_slots.reserve(outputs.size());
        published.reserve(outputs.size());
        for (auto& output : outputs) {
            const DeviceHostChannelTicket* ticket =
                find_publish_ticket(lane.tickets, output.slot);
            if (ticket == nullptr) continue;
            output.device_ptr = ticket->cells +
                static_cast<std::size_t>(
                    ticket->expected_tail % ticket->cap1) *
                    ticket->native_bytes;
            published.push_back({
                .slot = output.slot,
                .target = impl_->channels.schedule_host_publish_at(
                    output.slot,
                    ticket->expected_tail,
                    output.device_ptr,
                    callback_stream),
                .wait_id = impl_->channels.reader_wait_id(output.slot),
                .words = impl_->channels.host_words(output.slot),
            });
            output_slots.push_back(output.slot);
        }
        std::vector<NotifyContext::FinalizeEntry::EndpointUpdate> consumed;
        consumed.reserve(lane.tickets.size());
        for (const DeviceHostChannelTicket& ticket : lane.tickets) {
            if ((ticket.flags & (kTicketConsume | kTicketHostWriter)) !=
                (kTicketConsume | kTicketHostWriter)) {
                continue;
            }
            consumed.push_back({
                .slot = ticket.slot,
                .target = ticket.expected_head + 1,
                .wait_id = impl_->channels.writer_wait_id(ticket.slot),
                .words = ticket.words,
            });
        }
        if (output_slots.size() >
            kMaxConditionalConsumeChannels) {
            throw std::runtime_error(
                "PTIR host output count exceeds settlement capacity");
        }
        HostChannelSettlementLane settlement{
            .full = bound.instance->view().d_full(),
            .head = bound.instance->view().d_head(),
            .cap1 = bound.instance->view().d_cap1(),
            .commit = lane.snapshot->device,
            .tickets = lane.device_tickets,
            .ticket_count = lane.device_ticket_count,
        };
        settlement.consume.n =
            static_cast<std::uint32_t>(output_slots.size());
        for (std::size_t index = 0;
             index < output_slots.size();
             ++index) {
            settlement.consume.slots[index] =
                output_slots[index];
        }
        if (batch_settlement) {
            notify->settlement_lanes.push_back(settlement);
        } else {
            if (!output_slots.empty()) {
                launch_consume_if_committed(
                    bound.instance->view().d_full(),
                    bound.instance->view().d_head(),
                    bound.instance->view().d_cap1(),
                    lane.snapshot->device,
                    output_slots.data(),
                    static_cast<std::uint32_t>(
                        output_slots.size()),
                    callback_stream);
            }
            launch_publish_host_channel_actuals(
                lane.device_tickets,
                lane.device_ticket_count,
                lane.snapshot->device,
                callback_stream);
            if (lane.device_tickets != nullptr) {
                CUDA_CHECK(cudaFreeAsync(
                    lane.device_tickets, callback_stream));
                lane.device_tickets = nullptr;
            }
        }
        CUDA_CHECK(cudaMemcpyAsync(
            lane.snapshot->host,
            lane.snapshot->device,
            sizeof(std::uint32_t),
            cudaMemcpyDeviceToHost,
            callback_stream));
        std::vector<NotifyContext::FinalizeEntry::EndpointUpdate> poisoned;
        poisoned.reserve(bound.trace->channels.size());
        for (std::size_t channel = 0;
             channel < bound.trace->channels.size();
             ++channel) {
            if (!bound.trace->channels[channel].host_visible) continue;
            const std::uint32_t slot =
                impl_->channels.slot_for(bound.channel_ids[channel]);
            if (slot == DeviceChannelRegistry::kBadSlot) continue;
            poisoned.push_back({
                .slot = slot,
                .target = impl_->channels.poison_target(slot),
                .wait_id = impl_->channels.host_wait_id(slot),
                .words = impl_->channels.host_words(slot),
            });
        }
        notify->entries.push_back(NotifyContext::FinalizeEntry{
            .terminal_cell = view.terminal_cells.data()[lane.program],
            .commit_host = lane.snapshot->host,
            .poison = false,
            .instance_id = lane.instance_id,
            .tickets =
                std::getenv("PIE_CHANNEL_RETRY_TRACE") != nullptr
                ? lane.tickets
                : std::vector<DeviceHostChannelTicket>{},
            .published = std::move(published),
            .consumed = std::move(consumed),
            .poisoned = std::move(poisoned),
        });
    }
    if (batch_settlement) {
        launch_settle_host_channels_batch(
            notify->settlement_lanes, callback_stream);
        for (auto& lane_ptr : state.lanes) {
            StagedLane& lane = *lane_ptr;
            if (lane.device_tickets != nullptr) {
                CUDA_CHECK(cudaFreeAsync(
                    lane.device_tickets, callback_stream));
                lane.device_tickets = nullptr;
            }
        }
    }

    std::sort(
        state.touched_instances.begin(),
        state.touched_instances.end());
    state.touched_instances.erase(
        std::unique(
            state.touched_instances.begin(),
            state.touched_instances.end()),
        state.touched_instances.end());
    for (std::uint64_t instance_id : state.touched_instances) {
        auto found = impl_->instances.find(instance_id);
        if (found != impl_->instances.end()) {
            CUDA_CHECK(cudaEventRecord(
                found->second.publish_done, callback_stream));
        }
    }
    CUDA_CHECK(cudaLaunchHostFunc(
        callback_stream, notify_runtime_callback, notify.get()));
    notify.release();
    if (state.device_layer != nullptr) {
        CUDA_CHECK(cudaFreeAsync(state.device_layer, stream));
        state.device_layer = nullptr;
    }
    state.active = false;
    return true;
}

void Dispatch::abort(
    StagedLaunch& launch,
    cudaStream_t stream) noexcept {
    if (!launch.state_ || !launch.state_->active) return;
    StagedLaunch::State& state = *launch.state_;
    const std::uint32_t zero = 0;
    for (auto& lane : state.lanes) {
        if (lane->snapshot != nullptr &&
            lane->snapshot->device != nullptr) {
            cudaMemcpyAsync(
                lane->snapshot->device,
                &zero,
                sizeof(zero),
                cudaMemcpyHostToDevice,
                stream);
        }
        if (lane->device_tickets != nullptr) {
            cudaFreeAsync(lane->device_tickets, stream);
            lane->device_tickets = nullptr;
        }
        if (lane->bound != nullptr &&
            lane->bound->publish_done != nullptr) {
            cudaEventRecord(lane->bound->publish_done, stream);
        }
    }
    if (state.device_layer != nullptr) {
        cudaFreeAsync(state.device_layer, stream);
        state.device_layer = nullptr;
    }
    state.stream = stream;
    state.failed = true;
    state.active = false;
}

bool Dispatch::run(
    const pie_native::LaunchView& view,
    const void* logits,
    std::uint32_t vocab,
    cudaStream_t stream,
    const PieRuntimeCallbacks* runtime,
    PieCompletion completion,
    const std::uint16_t* direct_bf16_logits,
    const std::uint32_t* direct_row_indices,
    std::span<const std::uint32_t> mtp_draft_row_starts,
    std::span<const std::uint32_t> mtp_draft_row_counts,
    std::uint32_t direct_bf16_row_capacity) {
    if (view.ptir_program_hashes.empty()) {
        if (runtime != nullptr && runtime->notify != nullptr &&
            completion.wait_id != 0) {
            runtime->notify(
                runtime->ctx,
                completion.wait_id,
                completion.target_epoch);
        }
        return false;
    }
    auto launch = begin(view, stream);
    try {
        if (launch_has_attention_stages(view)) {
            throw std::runtime_error(
                "PTIR attention stages require launch-scoped model hooks");
        }
        std::vector<std::uint32_t> token_starts(
            view.ptir_program_hashes.size(), 0);
        if (view.ptir_token_counts.size() == token_starts.size()) {
            std::uint32_t cursor = 0;
            for (std::size_t program = 0;
                 program < token_starts.size();
                 ++program) {
                token_starts[program] = cursor;
                const std::uint32_t count =
                    view.ptir_token_counts.data()[program];
                if (count != kUnavailableGroupedExtent) cursor += count;
            }
        }
        update_launch_geometry(*launch, view, token_starts);
        return finish(
            *launch,
            view,
            logits,
            vocab,
            stream,
            runtime,
            completion,
            direct_bf16_logits,
            direct_row_indices,
            mtp_draft_row_starts,
            mtp_draft_row_counts,
            direct_bf16_row_capacity);
    } catch (...) {
        abort(*launch, stream);
        throw;
    }
}

std::vector<std::pair<std::uint64_t, std::uint64_t>>
Dispatch::settle_failed_launch(
    const pie_native::LaunchView& view,
    cudaStream_t execution_stream) {
    if (execution_stream != nullptr) {
        const cudaError_t status = cudaStreamSynchronize(execution_stream);
        if (status != cudaSuccess) {
            std::fprintf(
                stderr,
                "[pie-driver-cuda] failed launch stream synchronization: %s\n",
                cudaGetErrorString(status));
        }
    }
    cudaStream_t callback_stream =
        sampling_ir::FrameCarrierEngine::instance().copy_stream();
    if (callback_stream != nullptr && callback_stream != execution_stream) {
        const cudaError_t status = cudaStreamSynchronize(callback_stream);
        if (status != cudaSuccess) {
            std::fprintf(
                stderr,
                "[pie-driver-cuda] failed launch callback synchronization: %s\n",
                cudaGetErrorString(status));
        }
    }

    Impl& s = *impl_;
    std::vector<std::pair<std::uint64_t, std::uint64_t>> notifications;
    for (std::size_t p = 0; p < view.ptir_program_instances.size(); ++p) {
        const std::uint64_t instance_id =
            view.ptir_program_instances.data()[p];
        auto it = s.instances.find(instance_id);
        if (it == s.instances.end()) continue;
        BoundInstance& bound = it->second;
        for (std::size_t c = 0; c < bound.trace->channels.size(); ++c) {
            if (!bound.trace->channels[c].host_visible) continue;
            const std::uint32_t slot =
                s.channels.slot_for(bound.channel_ids[c]);
            if (slot != DeviceChannelRegistry::kBadSlot) {
                const std::uint64_t poison_epoch =
                    s.channels.poison_target(slot);
                s.channels.finalize_host_publish(slot, poison_epoch, true);
                notifications.emplace_back(
                    s.channels.host_wait_id(slot), poison_epoch);
            }
        }
    }
    return notifications;
}

bool Dispatch::resolve_descriptors(const pie_native::LaunchView& view,
                                   std::uint32_t page_size,
                                   std::uint32_t device_pages,
                                   ResolvedPrograms& out,
                                   std::string* err,
                                   bool allow_structured_masks,
                                   StagedLaunch* launch) {
    if (err) err->clear();
    out = ResolvedPrograms{};
    if (view.ptir_program_hashes.empty()) return false;
    Impl& s = *impl_;
    const std::size_t n_prog = view.ptir_program_hashes.size();
    if (view.ptir_program_instances.size() != n_prog) {
        if (err) *err = "ptir descriptor resolution instance/hash count mismatch";
        return false;
    }
    StagedLaunch::State* staged =
        launch == nullptr ? nullptr : launch->state_.get();
    if (staged != nullptr) {
        if (!staged->active || staged->lanes.size() != n_prog) {
            if (err) *err = "ptir descriptor resolution has no active launch";
            return false;
        }
    }

    for (std::size_t p = 0; p < n_prog; ++p) {
        const std::uint64_t iid = view.ptir_program_instances.data()[p];
        auto it = s.instances.find(iid);
        if (it == s.instances.end()) {
            if (err) *err = "ptir descriptor resolution missing instance " +
                            std::to_string(iid);
            return false;
        }
        if (it->second.program_hash != view.ptir_program_hashes.data()[p]) {
            if (err) *err = "ptir descriptor resolution instance/hash mismatch";
            return false;
        }
        const Trace* trace = it->second.trace;
        if (trace == nullptr) {
            if (err) *err = "ptir descriptor resolution missing trace";
            return false;
        }
    }

    out.per_program.resize(n_prog);
    out.is_device_geometry.assign(n_prog, 0);
    std::vector<detail::PortCellCache> cached_cells(n_prog);
    std::vector<std::vector<std::vector<std::uint8_t>>> writer_staging(
        n_prog);
    bool pulled_writer_input = false;
    for (std::size_t p = 0; p < n_prog; ++p) {
        const std::uint64_t iid =
            view.ptir_program_instances.data()[p];
        auto it = s.instances.find(iid);
        if (!is_device_geometry_trace(*it->second.trace)) continue;
        std::string value_error;
        if (!it->second.instance->writer_inputs_available(&value_error)) {
            throw RetryableLaunchError(value_error);
        }
        pulled_writer_input =
            it->second.instance->pull_writer_inputs(
                nullptr, writer_staging[p]) ||
            pulled_writer_input;
    }
    if (pulled_writer_input) {
        CUDA_CHECK(cudaStreamSynchronize(nullptr));
    }

    struct PortCopy {
        std::size_t program = 0;
        std::uint32_t slot = 0;
        const void* source = nullptr;
        const std::uint8_t* ready_source = nullptr;
    };
    std::vector<PortCopy> port_copies;
    std::vector<std::uint32_t> ready(n_prog, 1);
    bool copied_readiness = false;
    cudaStream_t descriptor_stream =
        staged == nullptr ? nullptr : staged->stream;
    for (std::size_t p = 0; p < n_prog; ++p) {
        const std::uint64_t iid =
            view.ptir_program_instances.data()[p];
        auto it = s.instances.find(iid);
        const Trace* trace = it->second.trace;
        if (!is_device_geometry_trace(*trace)) continue;
        const std::unordered_set<std::uint32_t>* pending_slots =
            staged == nullptr
                ? nullptr
                : &staged->lanes[p]->prior_put_slots;
        for (const PortBinding& binding : trace->ports) {
            if (binding.is_const) continue;
            ChannelView& channel_view = it->second.instance->view();
            const std::uint32_t slot =
                channel_view.slot(binding.channel);
            auto [cell, inserted] =
                cached_cells[p].try_emplace(slot);
            if (!inserted) continue;
            cell->second.bytes.resize(
                channel_view.cell_bytes(binding.channel));
            const bool pending =
                pending_slots != nullptr &&
                pending_slots->contains(slot);
            cell->second.ready = pending ? 1 : 0;
            port_copies.push_back(PortCopy{
                .program = p,
                .slot = slot,
                .source = pending
                    ? channel_view.pending_cell(binding.channel)
                    : channel_view.committed_cell(binding.channel),
                .ready_source = pending
                    ? nullptr
                    : channel_view.d_full() +
                          static_cast<std::size_t>(slot) * kMaxRing +
                          channel_view.registry()->host_head(slot),
            });
        }
        if (staged != nullptr) {
            CUDA_CHECK(cudaMemcpyAsync(
                &ready[p],
                staged->lanes[p]->snapshot->device,
                sizeof(std::uint32_t),
                cudaMemcpyDeviceToHost,
                descriptor_stream));
            copied_readiness = true;
        }
    }
    for (const PortCopy& copy : port_copies) {
        auto& destination = cached_cells[copy.program].at(copy.slot);
        CUDA_CHECK(cudaMemcpyAsync(
            destination.bytes.data(),
            copy.source,
            destination.bytes.size(),
            cudaMemcpyDeviceToHost,
            descriptor_stream));
        if (copy.ready_source != nullptr) {
            CUDA_CHECK(cudaMemcpyAsync(
                &destination.ready,
                copy.ready_source,
                sizeof(destination.ready),
                cudaMemcpyDeviceToHost,
                descriptor_stream));
        }
    }
    if (copied_readiness || !port_copies.empty()) {
        CUDA_CHECK(cudaStreamSynchronize(descriptor_stream));
    }

    for (std::size_t p = 0; p < n_prog; ++p) {
        const std::uint64_t iid = view.ptir_program_instances.data()[p];
        auto it = s.instances.find(iid);
        const Trace* trace = it->second.trace;
        if (!is_device_geometry_trace(*trace)) continue;

        const std::unordered_set<std::uint32_t>* pending_slots = nullptr;
        if (staged != nullptr) {
            const StagedLane& lane = *staged->lanes[p];
            if (ready[p] == 0) {
                throw RetryableLaunchError(
                    "ptir prologue or channel readiness did not commit");
            }
            pending_slots = &lane.prior_put_slots;
        }

        FireGeometry& fg = out.per_program[p];
        if (!resolve_fire_geometry(
                *trace, it->second.instance->view(), page_size, fg, err,
                allow_structured_masks, pending_slots,
                &cached_cells[p])) {
            return false;
        }
        if (fg.structured_mask) {
            std::lock_guard<std::mutex> lock(s.stats_mutex);
            if (fg.has_mask) {
                ++s.stats.structured_mask_dense_fallback;
            } else {
                ++s.stats.structured_mask_direct;
            }
        }

        // WorkingSet page translation (kv_refact.md flattened-table model):
        // channel-resolved `Pages`/`WSlot` values are WorkingSet-RELATIVE
        // indexes — the guest never holds physical ids. Map them through this
        // instance's translation segment (committed mapping overlaid with the
        // fire's prepared write targets, built at prepare). An index past the
        // segment is a reserved-but-unwritten page (a masked-only attention
        // candidate): map it to page 0 — readable garbage the mask discards.
        // An EMPTY segment passes values through (legacy physical geometry).
        if (view.kv_translation_indptr.size() == n_prog + 1) {
            const std::uint32_t lo = view.kv_translation_indptr.data()[p];
            const std::uint32_t hi = view.kv_translation_indptr.data()[p + 1];
            if (hi > lo && hi <= view.kv_translation.size()) {
                const std::uint32_t* tr = view.kv_translation.data() + lo;
                const std::uint32_t tr_len = hi - lo;
                const bool masked_reads =
                    fg.has_mask || static_cast<bool>(fg.structured_mask);
                if (!translate_resolved_page_ids(
                        fg.kv_page_indices,
                        fg.w_page,
                        std::span<const std::uint32_t>(tr, tr_len),
                        masked_reads,
                        err)) {
                    return false;
                }
            }
        }

        if (!validate_fire_geometry(fg, device_pages, page_size, err)) {
            return false;
        }
        out.is_device_geometry[p] = 1;
        ++out.device_count;
    }
    return out.device_count > 0;
}

}  // namespace pie_cuda_driver::pipeline
