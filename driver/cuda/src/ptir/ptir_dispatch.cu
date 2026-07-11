// PTIR (thrust-3) stage-program dispatcher — the nvcc-compiled impl behind the
// CUDA-free `ptir_dispatch.hpp` façade. Includes the tier-0 runtime (device
// kernels) here, isolated from the host `.cpp` translation units.

#include "ptir/ptir_dispatch.hpp"

#include <algorithm>
#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cuda_check.hpp"
#include "ptir/program_runtime.hpp"

#include "ptir/descriptor_resolve.hpp"
#include "sampling_ir/frame_carrier.hpp"

namespace pie_cuda_driver::ptir {

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
    std::uint64_t fire_seq = 0;
    cudaEvent_t fire_ready = nullptr;
    cudaEvent_t publish_done = nullptr;
    std::vector<CommitSnapshot> commit_snapshots;
};

struct PtirDispatch::Impl {
    PtirProgramCache cache;
    DeviceChannelRegistry channels;
    std::unordered_map<std::uint64_t, BoundInstance> instances;
    std::atomic<bool> shutting_down{false};
};

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
        std::vector<EndpointUpdate> published;
        std::vector<EndpointUpdate> consumed;
        std::vector<EndpointUpdate> poisoned;
        // CPU-unpacked bool cells the writer-ring pull copies from; must
        // outlive the async H2D copies, so it rides to the callback.
        std::vector<std::vector<std::uint8_t>> host_staging;
    };
    PtirDispatch::Impl* impl = nullptr;
    std::vector<FinalizeEntry> entries;
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

void finalize_consume_words(std::uint64_t* words, std::uint64_t target, bool failed) {
    if (words == nullptr) return;
    if (failed) {
        std::atomic_ref<std::uint64_t>(words[2]).store(
            target == 0 ? 1 : target, std::memory_order_release);
        return;
    }
    std::atomic_ref<std::uint64_t>(words[0]).store(target, std::memory_order_release);
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
        const bool failed = entry.poison || !committed;
        for (const auto& update : entry.published) {
            finalize_publish_words(update.words, update.target, failed);
            notifications.emplace_back(update.wait_id, update.target);
        }
        for (const auto& update : entry.consumed) {
            finalize_consume_words(update.words, update.target, failed);
            notifications.emplace_back(update.wait_id, update.target);
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
                       : PIE_TERMINAL_OUTCOME_SUCCESS,
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
void close_bound_instance(PtirDispatch::Impl& s, std::uint64_t instance_id);

// Batch-level channel budget (§4.3 availability + reader capacity): members
// of one batch that share a channel are validated against the AGGREGATE of
// their planned ring consumes and reader publishes. Checked one-by-one, two
// members could both pass on the last available entry/slot and the second
// would die as a device-side poison instead of a synchronous rejection.
int validate_channel_budget(PtirDispatch::Impl& s,
                            const pie_native::LaunchView& view,
                            std::string* err) {
    std::unordered_map<std::uint32_t, std::uint64_t> planned_consumes;
    std::unordered_map<std::uint32_t, std::uint64_t> planned_publishes;
    const std::size_t count = view.ptir_program_hashes.size();
    for (std::size_t p = 0; p < count; ++p) {
        const std::uint64_t instance_id = view.ptir_program_instances.data()[p];
        auto it = s.instances.find(instance_id);
        if (it == s.instances.end()) continue;  // caller resolves instances
        for (const auto& [slot, gid] :
             it->second.instance->writer_take_slots()) {
            std::uint64_t& extra = planned_consumes[slot];
            if (s.channels.writer_available(slot) < extra + 1) {
                if (err) {
                    *err = "ptir channel " + std::to_string(gid) +
                        " has no host input for this fire "
                        "(put must happen before submit)";
                }
                return PIE_STATUS_INVALID_ARGUMENT;
            }
            ++extra;
        }
        for (const auto& output :
             it->second.instance->predict_outputs_device()) {
            std::uint64_t& extra = planned_publishes[output.slot];
            if (!s.channels.can_publish_n(output.slot, extra + 1)) {
                if (err) {
                    *err = "ptir channel " + std::to_string(output.gid) +
                        " has no host output capacity";
                }
                return PIE_STATUS_INVALID_ARGUMENT;
            }
            ++extra;
        }
    }
    return PIE_STATUS_OK;
}
}  // namespace

PtirDispatch::PtirDispatch() : impl_(std::make_unique<Impl>()) {}
PtirDispatch::~PtirDispatch() {
    if (!impl_) return;
    impl_->shutting_down.store(true, std::memory_order_release);
    CUDA_CHECK(cudaStreamSynchronize(
        sampling_ir::FrameCarrierEngine::instance().copy_stream()));
    while (!impl_->instances.empty()) {
        close_bound_instance(*impl_, impl_->instances.begin()->first);
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

void close_bound_instance(PtirDispatch::Impl& s, std::uint64_t instance_id) {
    auto it = s.instances.find(instance_id);
    if (it == s.instances.end()) return;
    if (it->second.publish_done != nullptr) {
        CUDA_CHECK(cudaEventSynchronize(it->second.publish_done));
        CUDA_CHECK(cudaEventDestroy(it->second.publish_done));
    }
    if (it->second.fire_ready != nullptr) {
        CUDA_CHECK(cudaEventDestroy(it->second.fire_ready));
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

int PtirDispatch::register_program(std::uint64_t program_hash,
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
    return PIE_STATUS_OK;
}

int PtirDispatch::register_channel(
    const PieChannelDesc& channel,
    PieChannelEndpointBinding* binding,
    std::string* err) {
    if (err) err->clear();
    return impl_->channels.register_endpoint(channel, binding, err)
        ? PIE_STATUS_OK
        : PIE_STATUS_INVALID_ARGUMENT;
}

int PtirDispatch::bind_instance(std::uint64_t instance_id,
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

void PtirDispatch::close_instance(std::uint64_t instance_id) {
    close_bound_instance(*impl_, instance_id);
}

int PtirDispatch::close_channel(std::uint64_t channel_id, std::string* err) {
    if (err) err->clear();
    return impl_->channels.close_endpoint(channel_id, err)
        ? PIE_STATUS_OK
        : (impl_->channels.contains(channel_id)
               ? PIE_STATUS_INVALID_ARGUMENT
               : PIE_STATUS_CLOSED);
}

int PtirDispatch::validate_launch(
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
        // §4.3 availability + reader capacity, aggregated across the batch: a
        // fire that would take from an empty host-writer ring (or publish
        // past the host reader's capacity) is rejected synchronously.
        const int budget = validate_channel_budget(*impl_, view, err);
        if (budget != PIE_STATUS_OK) return budget;
    } catch (const std::exception& error) {
        if (err) *err = error.what();
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    return PIE_STATUS_OK;
}

bool PtirDispatch::run(const pie_native::LaunchView& view,
                       const void* logits, std::uint32_t vocab, cudaStream_t stream,
                       const PieRuntimeCallbacks* runtime,
                       PieCompletion completion) {
    Impl& s = *impl_;
    if (view.ptir_program_hashes.empty()) {
        if (runtime != nullptr && runtime->notify != nullptr && completion.wait_id != 0) {
            runtime->notify(runtime->ctx, completion.wait_id, completion.target_epoch);
        }
        return false;
    }
    const std::size_t n_prog = view.ptir_program_hashes.size();
    if (view.ptir_program_instances.size() != n_prog) {
        throw std::runtime_error("ptir launch instance/hash count mismatch");
    }
    if (view.terminal_cells.size() != n_prog) {
        throw std::runtime_error("ptir launch terminal cell count mismatch");
    }
    for (std::size_t p = 0; p < n_prog; ++p) {
        const std::uint64_t iid = view.ptir_program_instances.data()[p];
        auto it = s.instances.find(iid);
        if (it == s.instances.end()) {
            throw std::runtime_error(
                "ptir launch references missing instance " + std::to_string(iid));
        }
        if (it->second.trace == nullptr ||
            it->second.program_hash != view.ptir_program_hashes.data()[p]) {
            throw std::runtime_error(
                "ptir launch instance/hash mismatch for " + std::to_string(iid));
        }
    }
    {
        std::string budget_error;
        if (validate_channel_budget(s, view, &budget_error) != PIE_STATUS_OK) {
            throw std::runtime_error(budget_error);
        }
    }
    sampling_ir::FrameCarrierEngine& carrier =
        sampling_ir::FrameCarrierEngine::instance();
    cudaStream_t callback_stream = carrier.copy_stream();
    std::vector<NotifyContext::FinalizeEntry> finalize_entries;
    std::vector<std::uint64_t> touched_instances;
    std::unordered_map<std::uint64_t, std::size_t> instance_fire_counts;

    if (std::getenv("PIE_PTIR_TRACE")) {
        std::fprintf(stderr, "[ptir-hook] FIRED: n_prog=%zu vocab=%u logits=%p\n",
                     n_prog, vocab, logits);
    }

    for (std::size_t p = 0; p < n_prog; ++p) {
        const std::uint64_t iid = view.ptir_program_instances.data()[p];
        auto it = s.instances.find(iid);
        BoundInstance& bound = it->second;
        BoundInstance::CommitSnapshot& snapshot =
            commit_snapshot(bound, instance_fire_counts[iid]++);
        ensure_event(&bound.fire_ready);
        ensure_event(&bound.publish_done);
        CUDA_CHECK(cudaStreamWaitEvent(stream, bound.publish_done, 0));

        FireInputs fin;
        // The executor hands this dispatch view PER-PROGRAM offsets into the
        // gathered sampled-logits rows (`n_prog + 1` entries, composed batch
        // order) in `sampling_indptr` — NOT the per-request sampling CSR.
        if (logits != nullptr &&
            view.sampling_indptr.size() == n_prog + 1) {
            const std::size_t row_offset =
                view.sampling_indptr.data()[p];
            fin.logits =
                static_cast<const float*>(logits) + row_offset * vocab;
        } else {
            fin.logits = nullptr;
        }
        fin.vocab = vocab;
        fin.stream = stream;
        std::vector<void*> scratch;
        std::vector<std::vector<std::uint8_t>> host_staging;
        // §4.3: pull the host-writer rings stream-ordered before the pass —
        // the put already lives in the shared pinned ring.
        bound.instance->pull_writer_inputs(stream, host_staging);
        const PassResult pass = bound.instance->fire_async(fin, scratch);
        CUDA_CHECK(cudaMemcpyAsync(
            snapshot.device,
            bound.instance->commit_device_flag(),
            sizeof(std::uint32_t),
            cudaMemcpyDeviceToDevice,
            stream));
        if (!pass.ok && !pass.error.empty()) {
            std::fprintf(stderr, "[pie-driver-cuda] ptir fire failed: %s\n",
                         pass.error.c_str());
        }
        auto outs = bound.instance->predict_outputs_device();
        if (std::getenv("PIE_PTIR_TRACE"))
            std::fprintf(stderr, "[ptir-hook] program %zu: harvested %zu output(s)%s\n",
                         p, outs.size(), outs.empty() ? " (NONE — no committed READER channel)" : "");
        CUDA_CHECK(cudaEventRecord(bound.fire_ready, stream));
        CUDA_CHECK(cudaStreamWaitEvent(callback_stream, bound.fire_ready, 0));

        ++bound.fire_seq;
        std::vector<std::uint32_t> output_slots;
        const bool poison = !pass.ok;
        std::vector<NotifyContext::FinalizeEntry::EndpointUpdate> published;
        published.reserve(outs.size());
        for (const auto& output : outs) {
            published.push_back({
                .slot = output.slot,
                .target = s.channels.schedule_host_publish(
                    output.slot, output.device_ptr, callback_stream),
                .wait_id = s.channels.reader_wait_id(output.slot),
                .words = s.channels.host_words(output.slot),
            });
            output_slots.push_back(output.slot);
        }
        std::vector<NotifyContext::FinalizeEntry::EndpointUpdate> consumed;
        for (const auto& [slot, target] :
             bound.instance->schedule_writer_consumes()) {
            consumed.push_back({
                .slot = slot,
                .target = target,
                .wait_id = s.channels.writer_wait_id(slot),
                .words = s.channels.host_words(slot),
            });
        }
        if (!output_slots.empty()) {
            launch_consume_if_committed(
                bound.instance->view().d_full(),
                bound.instance->view().d_head(),
                bound.instance->view().d_cap1(),
                snapshot.device,
                output_slots.data(),
                static_cast<std::uint32_t>(output_slots.size()),
                callback_stream);
        }
        CUDA_CHECK(cudaMemcpyAsync(snapshot.host,
                                   snapshot.device,
                                   sizeof(std::uint32_t),
                                   cudaMemcpyDeviceToHost,
                                   callback_stream));
        bound.instance->project_fire_success(outs);
        std::vector<NotifyContext::FinalizeEntry::EndpointUpdate> poisoned;
        for (std::size_t c = 0; c < bound.trace->channels.size(); ++c) {
            if (!bound.trace->channels[c].host_visible) continue;
            const std::uint32_t slot =
                s.channels.slot_for(bound.channel_ids[c]);
            if (slot != DeviceChannelRegistry::kBadSlot) {
                poisoned.push_back({
                    .slot = slot,
                    .target = s.channels.poison_target(slot),
                    .wait_id = s.channels.host_wait_id(slot),
                    .words = s.channels.host_words(slot),
                });
            }
        }
        finalize_entries.push_back(NotifyContext::FinalizeEntry{
            .terminal_cell = view.terminal_cells.data()[p],
            .commit_host = snapshot.host,
            .poison = poison,
            .published = std::move(published),
            .consumed = std::move(consumed),
            .poisoned = std::move(poisoned),
            .host_staging = std::move(host_staging),
        });
        touched_instances.push_back(iid);
    }

    if (!finalize_entries.empty()) {
        auto notify = std::make_unique<NotifyContext>();
        if (runtime != nullptr) notify->runtime = *runtime;
        notify->completion = completion;
        notify->impl = &s;
        notify->entries = std::move(finalize_entries);
        std::sort(touched_instances.begin(), touched_instances.end());
        touched_instances.erase(
            std::unique(touched_instances.begin(), touched_instances.end()),
            touched_instances.end());
        for (std::uint64_t iid : touched_instances) {
            auto it = s.instances.find(iid);
            if (it != s.instances.end()) {
                CUDA_CHECK(cudaEventRecord(it->second.publish_done, callback_stream));
            }
        }
        CUDA_CHECK(cudaLaunchHostFunc(
            callback_stream, notify_runtime_callback, notify.get()));
        notify.release();
    }

    return true;
}

std::vector<std::pair<std::uint64_t, std::uint64_t>>
PtirDispatch::settle_failed_launch(
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

bool PtirDispatch::resolve_descriptors(const pie_native::LaunchView& view,
                                       std::uint32_t page_size,
                                       std::uint32_t device_pages,
                                       ResolvedPrograms& out,
                                       std::string* err) {
    if (err) err->clear();
    out = ResolvedPrograms{};
    if (view.ptir_program_hashes.empty()) return false;
    Impl& s = *impl_;
    const std::size_t n_prog = view.ptir_program_hashes.size();
    if (view.ptir_program_instances.size() != n_prog) {
        if (err) *err = "ptir descriptor resolution instance/hash count mismatch";
        return false;
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
        std::string value_error;
        if (!it->second.instance->writer_inputs_available(&value_error)) {
            if (err) *err = value_error;
            return false;
        }
    }

    out.per_program.resize(n_prog);
    out.is_device_geometry.assign(n_prog, 0);
    for (std::size_t p = 0; p < n_prog; ++p) {
        const std::uint64_t iid = view.ptir_program_instances.data()[p];
        auto it = s.instances.find(iid);
        const Trace* trace = it->second.trace;
        if (!is_device_geometry_trace(*trace)) continue;

        // Geometry resolution reads device channel state synchronously, so
        // pull the writer rings on the default stream and drain it first.
        std::vector<std::vector<std::uint8_t>> staging;
        it->second.instance->pull_writer_inputs(nullptr, staging);
        CUDA_CHECK(cudaStreamSynchronize(nullptr));

        FireGeometry& fg = out.per_program[p];
        if (!resolve_fire_geometry(
                *trace, it->second.instance->view(), page_size, fg, err)) {
            return false;
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
                auto translate = [&](std::vector<std::uint32_t>& ids) {
                    for (std::uint32_t& v : ids) v = v < tr_len ? tr[v] : 0u;
                };
                translate(fg.kv_page_indices);
                translate(fg.w_page);
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

}  // namespace pie_cuda_driver::ptir
