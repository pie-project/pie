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

struct InstanceFrame {
    std::uint64_t frame_id = 0;                        // FrameCarrierEngine instance
    std::uint64_t frame_base = 0;
    std::uint64_t mirror_base = 0;
    std::uint64_t word_base = 0;
    std::uint64_t frame_bytes = 0;
    std::uint64_t mirror_bytes = 0;
    std::uint64_t word_bytes = 0;
    std::uint32_t n_reader = 0;
    std::unordered_map<std::uint64_t, std::uint32_t> gid_to_rank;
    std::vector<std::uint64_t> reader_wait_ids;
    std::vector<std::uint32_t> cap1;                   // per-rank mirror ring depth
    std::vector<std::uint32_t> produced;               // per-rank cumulative published
    std::vector<std::uint32_t> head, tail;             // per-rank published ring words
    std::uint64_t fire_seq = 0;                        // pacing[0]
};

struct BoundInstance {
    std::uint64_t program_hash = 0;
    std::uint64_t pacing_wait_id = 0;
    const Trace* trace = nullptr;
    std::vector<std::uint64_t> channel_ids;
    std::vector<PieChannelWait> channel_waits;
    std::unique_ptr<PtirInstance> instance;
    InstanceFrame frame;
    std::vector<PieChannelBinding> channel_bindings;
    cudaEvent_t fire_ready = nullptr;
    cudaEvent_t publish_done = nullptr;
    std::uint32_t* commit_host = nullptr;
};

struct NotifyWait {
    std::uint64_t wait_id = 0;
    std::uint64_t epoch = 0;
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
        std::uint64_t instance_id = 0;
        bool poison = false;
        std::vector<NotifyWait> commit_waits;
        std::vector<PtirInstance::DeviceOut> outs;
    };
    PtirDispatch::Impl* impl = nullptr;
    std::vector<FinalizeEntry> entries;
};

void CUDART_CB notify_runtime_callback(void* userdata) {
    std::unique_ptr<NotifyContext> ctx(static_cast<NotifyContext*>(userdata));
    if (ctx == nullptr) return;
    std::vector<NotifyWait> waits;
    if (ctx->impl != nullptr) {
        for (const auto& entry : ctx->entries) {
            auto it = ctx->impl->instances.find(entry.instance_id);
            if (it == ctx->impl->instances.end()) continue;
            BoundInstance& bound = it->second;
            const bool committed =
                bound.commit_host != nullptr && *(bound.commit_host) != 0;
            bound.instance->apply_fire_result(committed, entry.outs);
            if (entry.poison) {
                auto* words = reinterpret_cast<std::atomic<std::uint64_t>*>(
                    bound.frame.word_base);
                if (words != nullptr) {
                    const std::uint64_t poison_epoch =
                        bound.frame.fire_seq == 0 ? 1 : bound.frame.fire_seq;
                    for (const PieChannelBinding& binding : bound.channel_bindings) {
                        words[binding.poison_word_index].store(
                            poison_epoch, std::memory_order_release);
                    }
                }
            } else if (committed) {
                InstanceFrame& fr = bound.frame;
                for (const auto& out : entry.outs) {
                    const auto rit = fr.gid_to_rank.find(out.gid);
                    if (rit == fr.gid_to_rank.end()) continue;
                    const std::uint32_t r = rit->second;
                    const std::uint32_t prod = ++fr.produced[r];
                    fr.tail[r] = prod;
                }
            }
            waits.insert(
                waits.end(), entry.commit_waits.begin(), entry.commit_waits.end());
        }
    }
    const bool notify =
        ctx->runtime.notify != nullptr &&
        (ctx->impl == nullptr ||
         !ctx->impl->shutting_down.load(std::memory_order_acquire));
    // No native instance/channel state is touched after the first wake: a woken
    // runtime thread may immediately close the instance.
    if (notify) {
        for (const NotifyWait& wait : waits) {
            if (wait.wait_id != 0) {
                ctx->runtime.notify(ctx->runtime.ctx, wait.wait_id, wait.epoch);
            }
        }
    }
    if (notify && ctx->completion.wait_id != 0) {
        ctx->runtime.notify(
            ctx->runtime.ctx, ctx->completion.wait_id, ctx->completion.target_epoch);
    }
}

namespace {
void close_bound_instance(PtirDispatch::Impl& s, std::uint64_t instance_id);
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

// Slice program `p` out of a (channels, blob, lens, indptr) SoA into
// ChannelValue[]. The blob offset for an entry is the running Σ of prior lens
// (the blob is concatenated across all programs' entries in entry order).
// `chans` is now the GLOBAL channel id per entry (W0.2 re-key).
std::vector<ChannelValue> read_channel_values(
    const pie_native::Slice<std::uint64_t>& chans,
    const pie_native::Slice<std::uint8_t>& blob,
    const pie_native::Slice<std::uint32_t>& lens,
    const pie_native::Slice<std::uint32_t>& indptr,
    std::size_t p) {
    std::vector<ChannelValue> out;
    if (indptr.size() < p + 2) return out;
    const std::uint32_t lo = indptr.data()[p];
    const std::uint32_t hi = indptr.data()[p + 1];
    std::size_t off = 0;
    for (std::uint32_t i = 0; i < lo; ++i) off += lens.data()[i];
    for (std::uint32_t e = lo; e < hi; ++e) {
        ChannelValue cv;
        cv.channel = chans.data()[e];
        const std::uint32_t n = lens.data()[e];
        cv.bytes.assign(blob.data() + off, blob.data() + off + n);
        off += n;
        out.push_back(std::move(cv));
    }
    return out;
}

std::vector<std::uint64_t> read_channel_ids(
    const pie_native::Slice<std::uint64_t>& ids,
    const pie_native::Slice<std::uint32_t>& indptr,
    std::size_t p) {
    std::vector<std::uint64_t> out;
    if (indptr.size() < p + 2) return out;
    const std::uint32_t lo = indptr.data()[p];
    const std::uint32_t hi = indptr.data()[p + 1];
    for (std::uint32_t e = lo; e < hi; ++e) out.push_back(ids.data()[e]);
    return out;
}

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
    if (it->second.commit_host != nullptr) {
        CUDA_CHECK(cudaFreeHost(it->second.commit_host));
    }
    if (it->second.frame.frame_id != 0) {
        sampling_ir::FrameCarrierEngine::instance().close_instance(
            it->second.frame.frame_id);
    }
    s.instances.erase(it);
}

bool make_instance_frame(const Trace& trace,
                         const std::vector<std::uint64_t>& channel_ids,
                         const std::vector<PieChannelWait>& channel_waits,
                         std::uint64_t instance_id,
                         InstanceFrame& frame) {
    std::vector<std::uint32_t> cell_bytes;
    std::vector<std::uint32_t> cap1;
    std::uint32_t rank = 0;
    for (std::size_t c = 0; c < trace.channels.size(); ++c) {
        const Channel& ch = trace.channels[c];
        if (!ch.host_reader) continue;
        const std::uint64_t gid = c < channel_ids.size()
            ? channel_ids[c]
            : static_cast<std::uint64_t>(c);
        std::size_t cb = ch.type.shape.numel() * dtype_size(ch.type.dtype);
        if (cb == 0) cb = dtype_size(ch.type.dtype);
        frame.gid_to_rank[gid] = rank;
        cell_bytes.push_back(static_cast<std::uint32_t>(cb));
        cap1.push_back(ch.capacity + 1);
        // Host-visible channel publication advances the consumer-visible tail, so
        // the wake goes to the channel's reader wait slot (not the writer/backpressure
        // slot, which tracks host-writer head movement in the opposite direction).
        frame.reader_wait_ids.push_back(
            c < channel_waits.size() ? channel_waits[c].reader_wait_id : 0);
        ++rank;
    }
    frame.n_reader = rank;
    frame.cap1 = std::move(cap1);
    if (rank == 0) return true;
    frame.produced.assign(rank, 0);
    frame.head.assign(rank, 0);
    frame.tail.assign(rank, 0);
    frame.word_bytes = static_cast<std::uint64_t>(
        sampling_ir::WordLayout::words(rank) * sizeof(std::uint64_t));
    frame.mirror_bytes = 0;
    for (std::uint32_t i = 0; i < rank; ++i) {
        frame.mirror_bytes +=
            static_cast<std::uint64_t>(cell_bytes[i]) * frame.cap1[i];
    }
    sampling_ir::FrameCarrierEngine::instance().bind_channels_keyed(
        instance_id, rank, cell_bytes.data(), frame.cap1.data(),
        &frame.frame_base, &frame.mirror_base, &frame.word_base);
    frame.frame_id = instance_id;
    frame.frame_bytes = 0;
    return true;
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

int PtirDispatch::bind_instance(std::uint64_t instance_id,
                                std::uint64_t program_hash,
                                std::uint64_t pacing_wait_id,
                                const std::vector<std::uint64_t>& channel_ids,
                                const std::vector<PieChannelWait>& channel_waits,
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

    close_bound_instance(*impl_, instance_id);

    BoundInstance bound;
    bound.program_hash = program_hash;
    bound.pacing_wait_id = pacing_wait_id;
    bound.trace = trace;
    bound.channel_ids = channel_ids;
    bound.channel_waits = channel_waits;
    bound.instance = std::move(inst);
    make_instance_frame(*trace, bound.channel_ids, bound.channel_waits, instance_id,
                        bound.frame);
    CUDA_CHECK(cudaMallocHost(reinterpret_cast<void**>(&bound.commit_host),
                              sizeof(std::uint32_t)));
    *bound.commit_host = 0;
    bound.channel_bindings.clear();
    bound.channel_bindings.reserve(bound.frame.n_reader);
    std::uint64_t mirror_offset = 0;
    std::uint32_t rank = 0;
    for (std::size_t c = 0; c < trace->channels.size(); ++c) {
        const Channel& ch = trace->channels[c];
        if (!ch.host_reader) continue;
        std::size_t cell_bytes = ch.type.shape.numel() * dtype_size(ch.type.dtype);
        if (cell_bytes == 0) cell_bytes = dtype_size(ch.type.dtype);
        const std::uint64_t gid =
            c < bound.channel_ids.size() ? bound.channel_ids[c] : static_cast<std::uint64_t>(c);
        bound.channel_bindings.push_back(PieChannelBinding{
            .channel_id = gid,
            .cell_bytes = static_cast<std::uint32_t>(cell_bytes),
            .capacity = ch.capacity,
            .mirror_offset = mirror_offset,
            .head_word_index = sampling_ir::WordLayout::head(rank),
            .tail_word_index = sampling_ir::WordLayout::tail(rank),
            .poison_word_index = sampling_ir::WordLayout::poison(rank),
            .reserved = 0,
        });
        mirror_offset += cell_bytes * static_cast<std::size_t>(ch.capacity + 1);
        ++rank;
    }
    ensure_event(&bound.publish_done);
    CUDA_CHECK(cudaEventRecord(
        bound.publish_done, sampling_ir::FrameCarrierEngine::instance().copy_stream()));

    if (binding != nullptr) {
        std::memset(binding, 0, sizeof(*binding));
        binding->instance_id = instance_id;
        binding->frame_base = bound.frame.frame_base;
        binding->mirror_base = bound.frame.mirror_base;
        binding->word_base = bound.frame.word_base;
        binding->channel_count = bound.frame.n_reader;
        binding->word_count = sampling_ir::WordLayout::words(bound.frame.n_reader);
        binding->frame_bytes = bound.frame.frame_bytes;
        binding->mirror_bytes = bound.frame.mirror_bytes;
        binding->word_bytes = bound.frame.word_bytes;
        binding->channels.ptr = bound.channel_bindings.data();
        binding->channels.len = bound.channel_bindings.size();
    }
    impl_->instances.emplace(instance_id, std::move(bound));
    return PIE_STATUS_OK;
}

void PtirDispatch::close_instance(std::uint64_t instance_id) {
    close_bound_instance(*impl_, instance_id);
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
    std::vector<std::vector<ChannelValue>> host_puts_by_program;
    host_puts_by_program.reserve(n_prog);
    for (std::size_t p = 0; p < n_prog; ++p) {
        auto host_puts = read_channel_values(
            view.ptir_program_host_put_channels,
            view.ptir_program_host_put_blob,
            view.ptir_program_host_put_lens,
            view.ptir_program_host_put_indptr,
            p);
        const std::uint64_t iid = view.ptir_program_instances.data()[p];
        BoundInstance& bound = s.instances.at(iid);
        std::string value_error;
        if (!bound.instance->validate_host_puts(host_puts, &value_error)) {
            throw std::runtime_error(value_error);
        }
        host_puts_by_program.push_back(std::move(host_puts));
    }
    sampling_ir::FrameCarrierEngine& carrier =
        sampling_ir::FrameCarrierEngine::instance();
    cudaStream_t callback_stream = carrier.copy_stream();
    std::vector<NotifyContext::FinalizeEntry> finalize_entries;
    std::vector<std::uint64_t> touched_instances;

    if (std::getenv("PIE_PTIR_TRACE")) {
        std::fprintf(stderr, "[ptir-hook] FIRED: n_prog=%zu vocab=%u logits=%p\n",
                     n_prog, vocab, logits);
    }

    for (std::size_t p = 0; p < n_prog; ++p) {
        const std::uint64_t iid = view.ptir_program_instances.data()[p];
        auto it = s.instances.find(iid);
        BoundInstance& bound = it->second;
        ensure_event(&bound.fire_ready);
        ensure_event(&bound.publish_done);
        CUDA_CHECK(cudaStreamWaitEvent(stream, bound.publish_done, 0));

        const auto& host_puts = host_puts_by_program[p];

        FireInputs fin;
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
        const PassResult pass = bound.instance->fire_async(host_puts, fin, scratch);
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

        InstanceFrame& fr = bound.frame;
        const std::uint64_t fire_seq = ++fr.fire_seq;
        std::vector<NotifyWait> commit_waits;
        std::vector<std::uint32_t> output_slots;
        const bool poison = !pass.ok;
        if (poison) {
            for (std::uint32_t r = 0; r < fr.n_reader; ++r) {
                const std::uint64_t wait_id =
                    r < fr.reader_wait_ids.size() ? fr.reader_wait_ids[r] : 0;
                if (wait_id != 0) {
                    commit_waits.push_back(NotifyWait{wait_id, fr.produced[r] + 1});
                }
            }
        }
        if (fr.n_reader > 0) {
            std::vector<const void*> src(fr.n_reader, nullptr);
            std::vector<std::uint32_t> ring_index(fr.n_reader, 0);
            std::vector<std::uint32_t> tail_after = fr.tail;
            for (auto& o : outs) {
                auto rit = fr.gid_to_rank.find(o.gid);
                if (rit == fr.gid_to_rank.end()) continue;
                const std::uint32_t r = rit->second;
                src[r] = o.device_ptr;
                ring_index[r] = fr.produced[r] % fr.cap1[r];
                tail_after[r] = fr.produced[r] + 1;
                output_slots.push_back(o.slot);
                const std::uint64_t wait_id =
                    r < fr.reader_wait_ids.size() ? fr.reader_wait_ids[r] : 0;
                if (wait_id != 0) {
                    commit_waits.push_back(NotifyWait{wait_id, tail_after[r]});
                }
            }
            carrier.publish_device(
                fr.frame_id, fr.n_reader, src.data(), ring_index.data(),
                fr.head.data(), tail_after.data(),
                bound.instance->commit_device_flag(), fire_seq);
        }
        if (!output_slots.empty()) {
            launch_consume_if_committed(
                bound.instance->view().d_full(),
                bound.instance->view().d_head(),
                bound.instance->view().d_cap1(),
                bound.instance->commit_device_flag(),
                output_slots.data(),
                static_cast<std::uint32_t>(output_slots.size()),
                callback_stream);
        }
        CUDA_CHECK(cudaMemcpyAsync(bound.commit_host,
                                   bound.instance->commit_device_flag(),
                                   sizeof(std::uint32_t),
                                   cudaMemcpyDeviceToHost,
                                   callback_stream));
        if (bound.pacing_wait_id != 0) {
            commit_waits.push_back(NotifyWait{bound.pacing_wait_id, fire_seq});
        }
        finalize_entries.push_back(NotifyContext::FinalizeEntry{
            .instance_id = iid,
            .poison = poison,
            .commit_waits = std::move(commit_waits),
            .outs = std::move(outs),
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
            callback_stream, notify_runtime_callback, notify.release()));
    }

    return true;
}

bool PtirDispatch::resolve_descriptors(const pie_native::LaunchView& view,
                                       std::uint32_t page_size,
                                       std::uint32_t device_pages,
                                       FireGeometry& out,
                                       std::string* err) {
    if (err) err->clear();
    if (view.ptir_program_hashes.empty()) return false;
    Impl& s = *impl_;
    const std::size_t n_prog = view.ptir_program_hashes.size();
    if (view.ptir_program_instances.size() != n_prog) {
        if (err) *err = "ptir descriptor resolution instance/hash count mismatch";
        return false;
    }

    std::vector<std::vector<ChannelValue>> host_puts_by_program;
    host_puts_by_program.reserve(n_prog);
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
        auto host_puts = read_channel_values(
            view.ptir_program_host_put_channels,
            view.ptir_program_host_put_blob,
            view.ptir_program_host_put_lens,
            view.ptir_program_host_put_indptr,
            p);
        std::string value_error;
        if (!it->second.instance->validate_host_puts(host_puts, &value_error)) {
            if (err) *err = value_error;
            return false;
        }
        host_puts_by_program.push_back(std::move(host_puts));
    }

    for (std::size_t p = 0; p < n_prog; ++p) {
        const std::uint64_t iid = view.ptir_program_instances.data()[p];
        auto it = s.instances.find(iid);
        const Trace* trace = it->second.trace;
        bool device_geometry = false;
        for (const PortBinding& pb : trace->ports)
            if (!pb.is_const) { device_geometry = true; break; }
        if (!device_geometry) continue;

        it->second.instance->feed_host_puts(host_puts_by_program[p]);

        if (!resolve_fire_geometry(
                *trace, it->second.instance->view(), page_size, out, err)) {
            return false;
        }
        return validate_fire_geometry(out, device_pages, page_size, err);
    }
    return false;
}

}  // namespace pie_cuda_driver::ptir
