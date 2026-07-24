#include <pie_native/step_launch.hpp>
#include "context.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <fstream>
#include <iostream>
#include <limits>
#include <memory>
#include <mutex>
#include <optional>
#include <span>
#include <string>
#include <thread>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include "pie_native/load_plan.hpp"
#include "batch/workspace.hpp"
#include "config.hpp"
#include "distributed.hpp"
#include "entry_validation.hpp"
#include "cuda_check.hpp"
#include "store/memory_planner.hpp"
#include "device_buffer.hpp"
#include "batch/frame.hpp"
#include "batch/fire_timing.hpp"
#include "batch/forward.hpp"
#include "batch/tp.hpp"
#include "kernels/kv_paged.hpp"
#include "store/kv_cache.hpp"
#include "store/elastic.hpp"
#include "store/mla_cache.hpp"
#include "store/dsa_cache.hpp"
#ifndef PIE_CUDA_QWEN_ONLY
#include "model/deepseek_v4/deepseek_v4_forward.hpp"
#include "model/gemma/gemma2.hpp"
#include "model/gemma4/gemma4.hpp"
#include "model/gemma4/gemma4_audio_adapter.hpp"
#include "model/gemma4/gemma4_vision_adapter.hpp"
#include "model/glm5/glm5_forward.hpp"
#include "model/kimi/kimi_forward.hpp"
#endif
#include "model/llama_like/llama_like.hpp"
#include "model/loaded_model.hpp"
#ifndef PIE_CUDA_QWEN_ONLY
#include "model/nemotron_h/nemotron_h.hpp"
#include "model/nemotron_h/nemotron_h_forward.hpp"
#endif
#include "model/qwen3_5/qwen3_5_config.hpp"
#include "model/qwen3_5/qwen3_5_forward.hpp"
#include "model/qwen3_5/qwen3_5_moe_forward.hpp"
#include "model/registry.hpp"
#include "model/workspace.hpp"
#include "ops/gemm.hpp"
#include "pipeline/registry.hpp"
#include "store/recurrent_state_cache.hpp"
#include "store/swap_pool.hpp"

namespace pie::cuda {
namespace {

struct OwnedValue {
    void* ptr = nullptr;
    void (*deleter)(void*) = nullptr;
};

struct AsyncCompletionContext {
    PieRuntimeCallbacks runtime{};
    PieCompletion completion{};
};

struct PendingAsyncResources {
    cudaEvent_t ready = nullptr;
    std::vector<OwnedValue> keepalive;

    ~PendingAsyncResources() {
        if (ready != nullptr) cudaEventDestroy(ready);
        for (auto it = keepalive.rbegin(); it != keepalive.rend(); ++it) {
            if (it->ptr != nullptr && it->deleter != nullptr) it->deleter(it->ptr);
        }
    }
};

template <typename T>
void keep_alive(std::vector<OwnedValue>& owners, T&& value) {
    using U = std::decay_t<T>;
    U* ptr = new U(std::forward<T>(value));
    owners.push_back(OwnedValue{ptr, [](void* p) { delete static_cast<U*>(p); }});
}

void publish_terminal(PieTerminalCell* cell, std::uint32_t outcome) {
    if (cell == nullptr) return;
    cell->reserved0 = 0;
    std::atomic_ref<std::uint32_t>(cell->outcome).store(
        outcome, std::memory_order_release);
}

std::size_t cuda_vmm_handle_bytes() {
    constexpr std::size_t kMib = 1024ull * 1024ull;
    constexpr std::size_t kDefaultMib = 32;
    const char* value = std::getenv("PIE_CUDA_VMM_HANDLE_MB");
    if (value == nullptr || *value == '\0') return kDefaultMib * kMib;
    char* end = nullptr;
    const unsigned long long mib = std::strtoull(value, &end, 10);
    if (end == value || end == nullptr || *end != '\0' ||
        mib < 2 || mib > 64) {
        throw std::runtime_error(
            "PIE_CUDA_VMM_HANDLE_MB must be an integer from 2 through 64");
    }
    return static_cast<std::size_t>(mib) * kMib;
}

void CUDART_CB finish_async_completion(void* userdata) {
    std::unique_ptr<AsyncCompletionContext> ctx(
        static_cast<AsyncCompletionContext*>(userdata));
    if (ctx == nullptr) return;
    publish_terminal(ctx->completion.terminal_cell, PIE_TERMINAL_OUTCOME_SUCCESS);
    if (ctx->runtime.notify != nullptr && ctx->completion.wait_id != 0) {
        ctx->runtime.notify(
            ctx->runtime.ctx,
            ctx->completion.wait_id,
            ctx->completion.target_epoch);
    }
}

struct LaunchScratch {
    std::vector<std::uint64_t> ptir_program_hashes;
    std::vector<std::uint64_t> ptir_program_instances;

    pie_native::LaunchView build(
        const pie_native::StepLaunch& launch,
        const std::vector<pie_cuda_driver::pipeline::InstanceRecord>& instances) {
        const std::size_t lanes = instances.size();
        ptir_program_hashes.clear();
        ptir_program_instances.clear();
        ptir_program_hashes.reserve(lanes);
        ptir_program_instances.reserve(lanes);
        for (const pie_cuda_driver::pipeline::InstanceRecord& inst : instances) {
            ptir_program_hashes.push_back(inst.program_hash);
            ptir_program_instances.push_back(inst.instance_id);
        }

        pie_native::LaunchView view{};
        view.terminal_cells = pie_native::slice_from(launch.terminal_cells.ptr, launch.terminal_cells.len);
        view.token_ids = pie_native::slice_from_u32(launch.token_ids.ptr, launch.token_ids.len);
        view.position_ids = pie_native::slice_from_u32(launch.position_ids.ptr, launch.position_ids.len);
        view.kv_page_indices = pie_native::slice_from_u32(launch.kv_page_indices.ptr, launch.kv_page_indices.len);
        view.kv_page_indptr = pie_native::slice_from_u32(launch.kv_page_indptr.ptr, launch.kv_page_indptr.len);
        view.kv_last_page_lens = pie_native::slice_from_u32(launch.kv_last_page_lens.ptr, launch.kv_last_page_lens.len);
        view.qo_indptr = pie_native::slice_from_u32(launch.qo_indptr.ptr, launch.qo_indptr.len);
        view.rs_slot_ids = pie_native::slice_from_u32(launch.rs_slot_ids.ptr, launch.rs_slot_ids.len);
        view.rs_slot_flags = pie_native::slice_from_u8(launch.rs_slot_flags.ptr, launch.rs_slot_flags.len);
        view.rs_fold_lens = pie_native::slice_from_u32(launch.rs_fold_lens.ptr, launch.rs_fold_lens.len);
        view.rs_buffer_slot_ids = pie_native::slice_from_u32(launch.rs_buffer_slot_ids.ptr, launch.rs_buffer_slot_ids.len);
        view.rs_buffer_slot_indptr = pie_native::slice_from_u32(launch.rs_buffer_slot_indptr.ptr, launch.rs_buffer_slot_indptr.len);
        view.flattened_masks = pie_native::slice_from_u32(launch.masks.words.ptr, launch.masks.words.len);
        view.mask_indptr = pie_native::slice_from_u32(launch.masks.word_indptr.ptr, launch.masks.word_indptr.len);
        view.sampling_indices = pie_native::slice_from_u32(launch.sampling_indices.ptr, launch.sampling_indices.len);
        view.sampling_indptr = pie_native::slice_from_u32(launch.sampling_indptr.ptr, launch.sampling_indptr.len);
        view.ptir_program_hashes = pie_native::slice_from_u64(ptir_program_hashes.data(), ptir_program_hashes.size());
        view.ptir_program_instances = pie_native::slice_from_u64(ptir_program_instances.data(), ptir_program_instances.size());
        view.kv_translation = pie_native::slice_from_u32(launch.kv_translation.ptr, launch.kv_translation.len);
        view.kv_translation_indptr = pie_native::slice_from_u32(launch.kv_translation_indptr.ptr, launch.kv_translation_indptr.len);
        view.ptir_program_row_indptr = pie_native::slice_from_u32(launch.ptir_program_row_indptr.ptr, launch.ptir_program_row_indptr.len);
        view.ptir_kv_write_lower_bounds = pie_native::slice_from_u64(
            launch.ptir_kv_write_lower_bounds.ptr,
            launch.ptir_kv_write_lower_bounds.len);
        view.ptir_kv_write_upper_bounds = pie_native::slice_from_u64(
            launch.ptir_kv_write_upper_bounds.ptr,
            launch.ptir_kv_write_upper_bounds.len);
        view.logical_fire_ids = pie_native::slice_from_u64(launch.logical_fire_ids.ptr, launch.logical_fire_ids.len);
        view.channel_expected_head = pie_native::slice_from_u64(launch.channel_expected_head.ptr, launch.channel_expected_head.len);
        view.channel_expected_tail = pie_native::slice_from_u64(launch.channel_expected_tail.ptr, launch.channel_expected_tail.len);
        view.channel_ticket_indptr = pie_native::slice_from_u32(launch.channel_ticket_indptr.ptr, launch.channel_ticket_indptr.len);
        view.has_user_mask = launch.has_user_mask != 0;
        view.image_grids = pie_native::slice_from_u32(launch.image_grids.ptr, launch.image_grids.len);
        view.image_pixels = pie_native::slice_from_u8(launch.image_pixels.ptr, launch.image_pixels.len);
        view.image_pixel_indptr = pie_native::slice_from_u32(launch.image_pixel_indptr.ptr, launch.image_pixel_indptr.len);
        view.image_mrope_positions = pie_native::slice_from_u32(launch.image_mrope_positions.ptr, launch.image_mrope_positions.len);
        view.image_mrope_indptr = pie_native::slice_from_u32(launch.image_mrope_indptr.ptr, launch.image_mrope_indptr.len);
        view.image_patch_positions = pie_native::slice_from_u32(launch.image_patch_positions.ptr, launch.image_patch_positions.len);
        view.image_anchor_rows = pie_native::slice_from_u32(launch.image_anchor_rows.ptr, launch.image_anchor_rows.len);
        view.audio_features = pie_native::slice_from_u8(launch.audio_features.ptr, launch.audio_features.len);
        view.audio_feature_indptr = pie_native::slice_from_u32(launch.audio_feature_indptr.ptr, launch.audio_feature_indptr.len);
        view.audio_anchor_rows = pie_native::slice_from_u32(launch.audio_anchor_rows.ptr, launch.audio_anchor_rows.len);
        view.embed_rows = pie_native::slice_from_u8(launch.embed_rows.ptr, launch.embed_rows.len);
        view.embed_indptr = pie_native::slice_from_u32(launch.embed_indptr.ptr, launch.embed_indptr.len);
        view.embed_shapes = pie_native::slice_from_u32(launch.embed_shapes.ptr, launch.embed_shapes.len);
        view.embed_dtypes = pie_native::slice_from_u8(launch.embed_dtypes.ptr, launch.embed_dtypes.len);
        view.embed_anchor_rows = pie_native::slice_from_u32(launch.embed_anchor_rows.ptr, launch.embed_anchor_rows.len);
        view.embed_block_indptr = pie_native::slice_from_u32(launch.embed_block_indptr.ptr, launch.embed_block_indptr.len);
        return view;
    }
};

// Frame → per-step expansion (Project Venus, ABI v14): materialize the
// step's instance ids from the frame roster and slice the frame's
// WorkingSet translation table to the step's members; every other field
// borrows the descriptor for the duration of the launch call.
struct StepExpansion {
    std::vector<std::uint64_t> instance_ids;
    std::vector<std::uint32_t> kv_translation;
    std::vector<std::uint32_t> kv_translation_indptr;
    pie_native::StepLaunch launch{};
};

void expand_step(
    const PieFrameDesc& frame,
    const PieStepDesc& step,
    StepExpansion* out) {
    out->instance_ids.clear();
    out->kv_translation.clear();
    out->kv_translation_indptr.clear();
    out->instance_ids.reserve(step.roster_rows.len);
    out->kv_translation_indptr.reserve(step.roster_rows.len + 1);
    out->kv_translation_indptr.push_back(0);
    const bool have_translation = frame.kv_translation_indptr.len != 0;
    for (std::size_t i = 0; i < step.roster_rows.len; ++i) {
        const std::uint32_t row = step.roster_rows.ptr[i];
        out->instance_ids.push_back(frame.instance_ids.ptr[row]);
        if (have_translation) {
            const std::uint32_t begin = frame.kv_translation_indptr.ptr[row];
            const std::uint32_t end = frame.kv_translation_indptr.ptr[row + 1];
            out->kv_translation.insert(
                out->kv_translation.end(),
                frame.kv_translation.ptr + begin,
                frame.kv_translation.ptr + end);
        }
        out->kv_translation_indptr.push_back(
            static_cast<std::uint32_t>(out->kv_translation.size()));
    }
    pie_native::StepLaunch& launch = out->launch;
    launch.instance_ids = {out->instance_ids.data(), out->instance_ids.size()};
    launch.terminal_cells = step.terminal_cells;
    launch.token_ids = step.token_ids;
    launch.position_ids = step.position_ids;
    launch.kv_page_indices = step.kv_page_indices;
    launch.kv_page_indptr = step.kv_page_indptr;
    launch.kv_last_page_lens = step.kv_last_page_lens;
    launch.qo_indptr = step.qo_indptr;
    launch.rs_slot_ids = step.rs_slot_ids;
    launch.rs_slot_flags = step.rs_slot_flags;
    launch.rs_fold_lens = step.rs_fold_lens;
    launch.rs_buffer_slot_ids = step.rs_buffer_slot_ids;
    launch.rs_buffer_slot_indptr = step.rs_buffer_slot_indptr;
    launch.masks = step.masks;
    launch.sampling_indices = step.sampling_indices;
    launch.sampling_indptr = step.sampling_indptr;
    launch.context_ids = step.context_ids;
    launch.single_token_mode = step.single_token_mode;
    launch.has_user_mask = step.has_user_mask;
    launch.required_kv_pages = frame.required_kv_pages;
    launch.image_indptr = step.image_indptr;
    launch.image_grids = step.image_grids;
    launch.image_anchor_positions = step.image_anchor_positions;
    launch.image_pixels = step.image_pixels;
    launch.image_pixel_indptr = step.image_pixel_indptr;
    launch.image_mrope_positions = step.image_mrope_positions;
    launch.image_mrope_indptr = step.image_mrope_indptr;
    launch.image_patch_positions = step.image_patch_positions;
    launch.image_anchor_rows = step.image_anchor_rows;
    launch.audio_features = step.audio_features;
    launch.audio_feature_indptr = step.audio_feature_indptr;
    launch.audio_anchor_rows = step.audio_anchor_rows;
    launch.audio_indptr = step.audio_indptr;
    launch.embed_rows = step.embed_rows;
    launch.embed_indptr = step.embed_indptr;
    launch.embed_shapes = step.embed_shapes;
    launch.embed_dtypes = step.embed_dtypes;
    launch.embed_anchor_rows = step.embed_anchor_rows;
    launch.embed_block_indptr = step.embed_block_indptr;
    launch.kv_len = step.kv_len;
    launch.kv_len_device = step.kv_len_device;
    launch.kv_translation = {
        out->kv_translation.data(), out->kv_translation.size()};
    launch.kv_translation_indptr = {
        out->kv_translation_indptr.data(), out->kv_translation_indptr.size()};
    launch.ptir_program_row_indptr = step.ptir_program_row_indptr;
    launch.ptir_kv_write_lower_bounds = step.ptir_kv_write_lower_bounds;
    launch.ptir_kv_write_upper_bounds = step.ptir_kv_write_upper_bounds;
    launch.logical_fire_ids = step.logical_fire_ids;
    launch.channel_expected_head = step.channel_expected_head;
    launch.channel_expected_tail = step.channel_expected_tail;
    launch.channel_ticket_indptr = step.channel_ticket_indptr;
}

struct TpStartupBarrier {
    std::mutex mu;
    std::condition_variable cv;
    int parties = 0;
    int arrived = 0;
    int generation = 0;
};

std::shared_ptr<TpStartupBarrier> tp_startup_barrier_for(
    const std::string& key,
    int parties)
{
    static std::mutex registry_mu;
    static std::unordered_map<std::string, std::shared_ptr<TpStartupBarrier>> registry;
    std::lock_guard<std::mutex> lock(registry_mu);
    auto& slot = registry[key];
    if (!slot) {
        slot = std::make_shared<TpStartupBarrier>();
        slot->parties = parties;
    } else if (slot->parties != parties) {
        throw std::runtime_error(
            "tp_startup_cpu_barrier: inconsistent tp_size for key " + key);
    }
    return slot;
}

void tp_startup_cpu_barrier(const pie_cuda_driver::Config& cfg) {
    if (cfg.distributed.tp_size <= 1 || cfg.distributed.nccl_unique_id_hex.empty()) {
        return;
    }
    auto barrier = tp_startup_barrier_for(
        cfg.distributed.nccl_unique_id_hex,
        cfg.distributed.tp_size);
    std::unique_lock<std::mutex> lock(barrier->mu);
    const int generation = barrier->generation;
    barrier->arrived += 1;
    if (barrier->arrived == barrier->parties) {
        barrier->arrived = 0;
        barrier->generation += 1;
        lock.unlock();
        barrier->cv.notify_all();
        return;
    }
    const bool released = barrier->cv.wait_for(
        lock,
        std::chrono::seconds(120),
        [&] { return barrier->generation != generation; });
    if (!released) {
        if (barrier->generation == generation && barrier->arrived > 0) {
            barrier->arrived -= 1;
        }
        throw std::runtime_error(
            "tp_startup_cpu_barrier: timed out waiting for TP ranks");
    }
}

int configured_mtp_num_drafts(const pie_cuda_driver::Config& cfg) {
    static const int forced = [] {
        const char* v = std::getenv("PIE_MTP_DRAFT_TOKENS");
        if (v == nullptr || v[0] == '\0') return -1;
        return std::clamp(std::atoi(v), 0, 32);
    }();
    if (forced >= 0) return forced;
    return std::clamp(cfg.model.mtp_num_drafts, 0, 32);
}

}  // namespace

class Context::Impl {
  public:
    Impl() = default;
    ~Impl() {
        drain_async_streams();
        if (kv_proportional_peak_required_pages_ > 0) {
            std::cerr
                << "[pie-driver-cuda] KV proportionality summary: "
                << "peak_required_pages="
                << kv_proportional_peak_required_pages_
                << " planned_pages=" << kv_proportional_planned_pages_
                << " peak_committed_bytes="
                << kv_proportional_peak_committed_bytes_
                << " capacity_bytes=" << kv_proportional_capacity_bytes_
                << "\n";
        }
        if (media_stream_ != nullptr) {
            cudaStreamDestroy(media_stream_);
            media_stream_ = nullptr;
        }
        try {
            if (tp_comm_ != nullptr && tp_size_ > 1 && tp_rank_ == 0) {
                pie_cuda_driver::tp_send_shutdown(*tp_comm_, tp_cpu_gate_key_);
            }
        } catch (const std::exception& e) {
            std::cerr << "[pie-driver-cuda] destroy: tp shutdown sentinel failed: "
                      << e.what() << "\n";
        } catch (...) {
            std::cerr << "[pie-driver-cuda] destroy: tp shutdown sentinel failed\n";
        }
        if (tp_follower_thread_.joinable()) {
            tp_follower_stop_.store(true);
            if (tp_comm_ != nullptr) tp_comm_->abort();
            tp_follower_thread_.join();
        }
        for (auto it = owners_.rbegin(); it != owners_.rend(); ++it) {
            if (it->ptr != nullptr && it->deleter != nullptr) it->deleter(it->ptr);
        }
    }

    int initialize(const std::string& config_path, const PieRuntimeCallbacks& runtime);
    void fill_device_facts(PieDriverCaps* caps) const {
        if (caps == nullptr) return;
        caps->json_bytes =
            reinterpret_cast<const std::uint8_t*>(device_facts_json_.data());
        caps->json_len = device_facts_json_.size();
    }
    int load_model(const PieModelLoadDesc& load, PieDriverCaps* caps);

    int register_program(const PieProgramDesc& program, std::uint64_t* program_id);
    int register_channel(const PieChannelDesc& channel,
                         PieChannelEndpointBinding* binding);
    int bind_instance(const PieInstanceDesc& instance, PieInstanceBinding* binding);
    int launch(const PieFrameDesc& frame, PieCompletion completion);
    int encode(const PieEncodeDesc& encode, PieCompletion completion);
    int copy_kv(const PieKvCopyDesc& copy, PieCompletion completion);
    int copy_state(const PieStateCopyDesc& copy, PieCompletion completion);
    int resize_pool(const PiePoolResizeDesc& resize, PieCompletion completion);
    int close_instance(std::uint64_t instance_id);
    int close_channel(std::uint64_t channel_id);

  private:
    template <typename T>
    T* own_value(T&& value) {
        using U = std::decay_t<T>;
        U* ptr = new U(std::forward<T>(value));
        owners_.push_back(OwnedValue{ptr, [](void* p) { delete static_cast<U*>(p); }});
        return ptr;
    }

    template <typename T, typename... Args>
    T* own_emplace(Args&&... args) {
        T* ptr = new T(std::forward<Args>(args)...);
        owners_.push_back(OwnedValue{ptr, [](void* p) { delete static_cast<T*>(p); }});
        return ptr;
    }

    template <typename T>
    static void keep_async_value(
        std::vector<OwnedValue>& owners,
        T&& value) {
        keep_alive(owners, std::forward<T>(value));
    }

    void retain_async_resources(
        cudaStream_t stream,
        std::vector<OwnedValue> keepalive) {
        if (keepalive.empty()) return;
        auto pending = std::make_unique<PendingAsyncResources>();
        pending->keepalive = std::move(keepalive);
        pending_async_resources_.push_back(std::move(pending));
        auto& slot = pending_async_resources_.back();
        CUDA_CHECK(cudaEventCreateWithFlags(&slot->ready, cudaEventDisableTiming));
        CUDA_CHECK(cudaEventRecord(slot->ready, stream));
    }

    void collect_ready_async_resources() {
        auto out = pending_async_resources_.begin();
        for (auto it = pending_async_resources_.begin();
             it != pending_async_resources_.end();
             ++it) {
            if ((*it)->ready == nullptr) {
                if (out != it) *out = std::move(*it);
                ++out;
                continue;
            }
            const cudaError_t status = cudaEventQuery((*it)->ready);
            if (status == cudaSuccess) {
                continue;
            }
            if (status != cudaErrorNotReady) {
                CUDA_CHECK(status);
            }
            if (out != it) *out = std::move(*it);
            ++out;
        }
        pending_async_resources_.erase(out, pending_async_resources_.end());
    }

    void enqueue_completion(
        cudaStream_t stream,
        PieCompletion completion) const {
        if (completion.wait_id == 0) return;
        auto ctx = std::make_unique<AsyncCompletionContext>();
        ctx->runtime = runtime_;
        ctx->completion = completion;
        CUDA_CHECK(cudaLaunchHostFunc(
            stream,
            finish_async_completion,
            ctx.release()));
    }

    void drain_async_streams() const noexcept {
        if (executor_ != nullptr) {
            cudaStream_t stream = executor_->cublas.stream();
            if (stream != nullptr) cudaStreamSynchronize(stream);
        }
        if (swap_pool_ != nullptr) {
            cudaStream_t stream = swap_pool_->stream();
            if (stream != nullptr) cudaStreamSynchronize(stream);
        }
        if (media_stream_ != nullptr) cudaStreamSynchronize(media_stream_);
    }

    bool is_tp_follower() const noexcept {
        return tp_size_ > 1 && tp_rank_ > 0;
    }

    int validate_finalized_launch(
        const pie_native::StepLaunch& launch,
        std::vector<pie_cuda_driver::pipeline::InstanceRecord>* instances,
        LaunchScratch* scratch,
        pie_native::LaunchView* view) const;
    int required_kv_pages(const pie_native::StepLaunch& launch) const;
    std::size_t required_state_slots(const pie_native::StepLaunch& launch) const;
    std::vector<pie_cuda_driver::CudaAllocatorTarget> frame_targets(
        int kv_required,
        std::size_t state_required,
        std::array<std::size_t, 4>* target_bytes) const;
    void recalibrate_elastic_budget(bool reset_hard_ceiling);

    PieRuntimeCallbacks runtime_{};
    pie_cuda_driver::Config* cfg_ = nullptr;
    std::vector<OwnedValue> owners_;
    pie_cuda_driver::BatchEngine* executor_ = nullptr;
    pie_cuda_driver::pipeline::Registry* registry_ = nullptr;
    pie_cuda_driver::model::IModel* model_ = nullptr;
#ifndef PIE_CUDA_QWEN_ONLY
    pie_cuda_driver::model::VisRawWeights* encode_vision_ = nullptr;
    pie_cuda_driver::model::AudioRawWeights* encode_audio_ = nullptr;
#endif
    pie_cuda_driver::KvCache* kv_cache_ = nullptr;
    pie_cuda_driver::SwapPool* swap_pool_ = nullptr;
    std::shared_ptr<pie_cuda_driver::CudaPhysicalPool> elastic_pool_;
    std::shared_ptr<pie_cuda_driver::CudaArenaAllocator> kv_allocator_;
    std::shared_ptr<pie_cuda_driver::CudaArenaAllocator> state_allocator_;
    std::shared_ptr<pie_cuda_driver::CudaArenaAllocator> workspace_allocator_;
    std::shared_ptr<pie_cuda_driver::CudaArenaAllocator> attention_allocator_;
    std::size_t elastic_safety_floor_bytes_ = 0;
    pie_cuda_driver::ops::RuntimeQuantContext runtime_quant_context_;
    pie_cuda_driver::NcclComm* tp_comm_ = nullptr;
    std::string caps_json_;
    std::string device_facts_json_;
    std::size_t kv_proportional_peak_required_pages_ = 0;
    std::size_t kv_proportional_planned_pages_ = 0;
    std::size_t kv_proportional_peak_committed_bytes_ = 0;
    std::size_t kv_proportional_capacity_bytes_ = 0;
    bool load_attempted_ = false;
    std::string tp_cpu_gate_key_;
    int device_ordinal_ = 0;
    int tp_size_ = 1;
    int tp_rank_ = 0;
    int media_hidden_size_ = 0;
    cudaStream_t media_stream_ = nullptr;
    pie_cuda_driver::abi::MultimodalLimits multimodal_limits_;
    std::atomic<bool> tp_follower_stop_{false};
    std::thread tp_follower_thread_;
    std::vector<std::unique_ptr<PendingAsyncResources>> pending_async_resources_;
};

int Context::Impl::initialize(
    const std::string& config_path, const PieRuntimeCallbacks& runtime) {
    using namespace pie_cuda_driver;
    runtime_ = runtime;

    cfg_ = own_value(load_config(config_path));
    Config& cfg = *cfg_;
    tp_size_ = std::max(1, cfg.distributed.tp_size);
    tp_rank_ = cfg.distributed.tp_rank;
    tp_cpu_gate_key_ = cfg.distributed.nccl_unique_id_hex;

    device_ordinal_ = parse_cuda_device_id(cfg.model.device);
    CUDA_CHECK(cudaSetDevice(device_ordinal_));

    // Retain the stream-ordered allocator's pool across frees: the per-step
    // transient staging (`cudaMallocAsync` lane/ticket arrays in the compose
    // and settle paths) otherwise shrinks the pool back to zero at every
    // sync point, and re-growing it makes `cudaMallocAsync` take the slow
    // synchronizing path — measured as the Σ316ms/run `h2d_prepare` and
    // Σ180ms `begin_pull_validate` lane stalls at k=2 (two steps' transient
    // churn in flight). The retained bytes are bounded by the transient
    // peak (lane descriptor arrays — KBs), never KV-scale.
    {
        cudaMemPool_t default_pool = nullptr;
        CUDA_CHECK(cudaDeviceGetDefaultMemPool(&default_pool, device_ordinal_));
        std::uint64_t threshold = std::numeric_limits<std::uint64_t>::max();
        CUDA_CHECK(cudaMemPoolSetAttribute(
            default_pool, cudaMemPoolAttrReleaseThreshold, &threshold));
    }

    pie_cuda_driver::NcclComm* tp_comm_ptr = nullptr;
    if (tp_size_ > 1) {
        const ncclUniqueId uid =
            pie_cuda_driver::nccl_unique_id_from_hex(cfg.distributed.nccl_unique_id_hex);
        auto* tp_comm_p = own_emplace<pie_cuda_driver::NcclComm>(
            tp_size_, tp_rank_, uid);
        tp_comm_ = tp_comm_p;
        tp_comm_ptr = tp_comm_p;
        tp_startup_cpu_barrier(cfg);

        cudaStream_t stream = nullptr;
        CUDA_CHECK(cudaStreamCreate(&stream));
        int* d_v = nullptr;
        try {
            CUDA_CHECK(cudaMalloc(&d_v, sizeof(int)));
            const int rank1 = tp_rank_ + 1;
            CUDA_CHECK(cudaMemcpyAsync(
                d_v, &rank1, sizeof(int), cudaMemcpyHostToDevice, stream));
            NCCL_CHECK_ASYNC(
                ncclAllReduce(
                    d_v, d_v, 1, ncclInt32, ncclSum, tp_comm_ptr->comm(), stream),
                tp_comm_ptr->comm());
            int h_v = 0;
            CUDA_CHECK(cudaMemcpyAsync(
                &h_v, d_v, sizeof(int), cudaMemcpyDeviceToHost, stream));
            CUDA_CHECK(cudaStreamSynchronize(stream));
            const int expected = tp_size_ * (tp_size_ + 1) / 2;
            if (h_v != expected) {
                std::cerr << "[pie-driver-cuda] NCCL smoke test failed: got "
                          << h_v << ", expected " << expected << "\n";
                CUDA_CHECK(cudaFree(d_v));
                CUDA_CHECK(cudaStreamDestroy(stream));
                return PIE_STATUS_DRIVER_ERROR;
            }
        } catch (...) {
            if (d_v != nullptr) cudaFree(d_v);
            if (stream != nullptr) cudaStreamDestroy(stream);
            throw;
        }
        CUDA_CHECK(cudaFree(d_v));
        CUDA_CHECK(cudaStreamDestroy(stream));
    }

    cudaDeviceProp dev_prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&dev_prop, device_ordinal_));
    const bool fp8_native = (dev_prop.major > 8) ||
                            (dev_prop.major == 8 && dev_prop.minor >= 9);
    const bool native_mxfp4_moe =
#ifdef PIE_CUDA_HAS_MARLIN
        dev_prop.major >= 10;
#else
        false;
#endif
    nlohmann::json facts = {
        {"abi_version", PIE_DRIVER_ABI_VERSION},
        {"backend", "cuda"},
        {"unified_memory", false},
        {"fp8_native", fp8_native},
        {"native_mxfp4_moe", native_mxfp4_moe},
        {"storage_alignment", 256},
        {"storage_max_tile_bytes", 64ull * 1024ull * 1024ull},
        {"storage_tile_map_mask", pie_load_planner::kCudaTileMapMask},
        {"page_size", 0},
    };
    device_facts_json_ = facts.dump();
    return PIE_STATUS_OK;
}

int Context::Impl::load_model(
    const PieModelLoadDesc& load,
    PieDriverCaps* caps_out) {
    using namespace pie_cuda_driver;
    ops::ScopedRuntimeQuantContext quant_scope(runtime_quant_context_);
    if (cfg_ == nullptr || load_attempted_) return PIE_STATUS_CLOSED;
    load_attempted_ = true;
    Config& cfg = *cfg_;
    const bool verbose = cfg.runtime.verbose;
    NcclComm* tp_comm_ptr = tp_comm_;
    cfg.model.snapshot_dir.assign(
        reinterpret_cast<const char*>(load.snapshot_dir.ptr),
        load.snapshot_dir.len);
    const std::span<const std::uint8_t> load_plan_bytes(
        load.load_plan_bytes.ptr, load.load_plan_bytes.len);

    auto* engine_p = own_value(LoadedModel::load(
        cfg, tp_comm_, load_plan_bytes, load.compiler_version));
    auto& engine = *engine_p;
    media_hidden_size_ = engine.hf_config().hidden_size;

    // THE arch table decides support: an unrecognized `model_type` is a
    // load error here, never a silent llama-like fallback (cpp-refact.md
    // Phase 5). `arch_entry->family` is the only thing this function
    // switches on below, for allocating family-specific stores/workspaces
    // that physically differ; the registry alone owns arch selection.
    const model::ArchEntry* arch_entry =
        model::find_arch_entry(engine.hf_config().model_type);
    if (arch_entry == nullptr) {
        std::cerr << "[pie-driver-cuda] unsupported arch '"
                  << engine.hf_config().model_type << "'\n";
        return PIE_STATUS_UNSUPPORTED;
    }
    if (arch_entry->validate_config) {
        if (auto config_error = arch_entry->validate_config(engine.hf_config())) {
            std::cerr << "[pie-driver-cuda] invalid config for arch '"
                      << engine.hf_config().model_type << "': " << *config_error
                      << "\n";
            return PIE_STATUS_UNSUPPORTED;
        }
    }
    const model::Family family = arch_entry->family;

    if (load.component == PIE_MODEL_COMPONENT_ENCODE) {
#ifdef PIE_CUDA_QWEN_ONLY
        return PIE_STATUS_UNSUPPORTED;
#else
        if (tp_size_ > 1 || family != model::Family::Gemma4 ||
            (!engine.hf_config().gemma_vision.has_value() &&
             !engine.hf_config().gemma_audio.has_value())) {
            return PIE_STATUS_UNSUPPORTED;
        }
        if (engine.hf_config().gemma_vision.has_value()) {
            auto vision = model::bind_gemma4_vision(engine);
            encode_vision_ = own_value(model::to_vis_raw(vision));
            multimodal_limits_.gemma4_pool_kernel =
                encode_vision_->pool_kernel;
            multimodal_limits_.gemma4_position_table =
                encode_vision_->pos_table_size;
        }
        if (engine.hf_config().gemma_audio.has_value()) {
            auto audio = model::bind_gemma4_audio(engine);
            encode_audio_ = own_value(model::to_audio_raw(audio));
            multimodal_limits_.audio_mel_bins = encode_audio_->n_mel;
        }
        CUDA_CHECK(cudaStreamCreateWithFlags(&media_stream_, cudaStreamNonBlocking));

        const auto c = engine.capabilities();
        nlohmann::json caps = {
            {"abi_version", PIE_DRIVER_ABI_VERSION},
            {"total_pages", 0},
            {"kv_page_size", 0},
            {"swap_pool_size", 0},
            {"kv_copy_domain_mask", 0},
            {"rs_cache_required", false},
            {"rs_cache_slots", 0},
            {"rs_cache_slot_bytes", 0},
            {"has_mtp_logits", false},
            {"has_mtp_drafts", false},
            {"has_value_head", false},
            {"max_forward_tokens",
             static_cast<std::uint32_t>(std::max(1, c.max_model_len))},
            {"max_forward_requests", 256},
            {"max_page_refs", 0},
            {"arch_name", c.arch_name},
            {"vocab_size", c.vocab_size},
            {"max_model_len", c.max_model_len},
            {"activation_dtype", c.activation_dtype},
            {"hidden_size",
             static_cast<std::uint32_t>(media_hidden_size_)},
            {"supports_media_encode", true},
            {"snapshot_dir", c.snapshot_dir},
            {"kv_handle", nullptr},
        };
        caps_json_ = caps.dump();
        if (caps_out != nullptr) {
            caps_out->json_bytes =
                reinterpret_cast<const std::uint8_t*>(caps_json_.data());
            caps_out->json_len = caps_json_.size();
        }
        return PIE_STATUS_OK;
#endif
    }
    if (load.component != PIE_MODEL_COMPONENT_FULL) {
        return PIE_STATUS_UNSUPPORTED;
    }

    // Bind the checkpoint's weights into a family-owned `ModelPlan`. The
    // plan stays alive (as a local `unique_ptr`) until `create_model`
    // below moves its concrete weights into the constructed `IModel`;
    // `plan_info()` is the narrow pre-construction surface every sizing
    // decision in between reads instead of downcasting the plan.
    std::unique_ptr<model::ModelPlan> plan = arch_entry->bind(engine, verbose);
    const model::PlanInfo& plan_info = plan->plan_info();

    if (plan_info.has_vision) {
        if (family == model::Family::Gemma4) {
            multimodal_limits_.gemma4_pool_kernel = plan_info.gemma4_pool_kernel;
            multimodal_limits_.gemma4_position_table = plan_info.gemma4_position_table;
        } else if (family == model::Family::Qwen3VL) {
            multimodal_limits_.qwen3_vl_patch_dim = plan_info.qwen3_vl_patch_dim;
            multimodal_limits_.qwen3_vl_merge_unit = plan_info.qwen3_vl_merge_unit;
        }
    }
    if (plan_info.has_audio) {
        multimodal_limits_.audio_mel_bins = plan_info.audio_mel_bins;
    }
    const int native_mtp_num_drafts = configured_mtp_num_drafts(cfg);
    const bool has_mtp_logits =
        plan_info.has_mtp && native_mtp_num_drafts > 0;

    const int local_tp_size = tp_size_;
    const int local_q_heads = engine.hf_config().num_attention_heads / local_tp_size;
    const int local_kv_heads = engine.hf_config().num_key_value_heads / local_tp_size;
    int max_mlp_intermediate = engine.hf_config().intermediate_size / local_tp_size;
    int max_Hq = local_q_heads * engine.hf_config().head_dim;
    int max_Hk = local_kv_heads * engine.hf_config().head_dim;
    switch (family) {
    case model::Family::Gemma4:
        for (int v : plan_info.per_layer_intermediate)
            max_mlp_intermediate = std::max(max_mlp_intermediate, v);
        for (int d : plan_info.per_layer_head_dim) {
            max_Hq = std::max(max_Hq, local_q_heads * d);
            max_Hk = std::max(max_Hk, local_kv_heads * d);
        }
        break;
    case model::Family::Gemma3n:
        for (int v : plan_info.per_layer_intermediate)
            max_mlp_intermediate = std::max(max_mlp_intermediate, v / local_tp_size);
        break;
    case model::Family::NemotronH: {
        const auto& hf_n = engine.hf_config();
        max_mlp_intermediate = std::max(
            max_mlp_intermediate,
            std::max(hf_n.moe_intermediate_size / local_tp_size,
                     hf_n.shared_expert_intermediate_size / local_tp_size));
        break;
    }
    default:
        break;
    }

    // Recurrent/linear-attention layer maps come straight from the bound
    // plan's pre-construction surface (populated once at bind time from
    // the actual layer weights, not re-derived here).
    const std::vector<bool>& qwen3_5_layer_is_linear = plan_info.layer_is_linear_attn;
    const int qwen3_5_linear_layers = static_cast<int>(std::count(
        qwen3_5_layer_is_linear.begin(), qwen3_5_layer_is_linear.end(), true));

#ifndef PIE_CUDA_QWEN_ONLY
    const std::vector<bool>& nemotron_h_layer_is_mamba =
        plan_info.layer_is_mamba;
    const int nemotron_h_mamba_layers = static_cast<int>(std::count(
        nemotron_h_layer_is_mamba.begin(),
        nemotron_h_layer_is_mamba.end(), true));
    const int nemotron_h_attention_layer_count =
        family == model::Family::NemotronH
            ? model::nemotron_h_attention_layers(engine.hf_config())
            : 0;
#else
    const int nemotron_h_mamba_layers = 0;
    const int nemotron_h_attention_layer_count = 0;
#endif

    const auto kv_format = kv_cache_format_from_string(
        cfg.batching.kv_cache_dtype, cfg.model.dtype);
    const bool use_cuda_graphs = true;
    const auto runtime_quant_scratch_base =
        runtime_quant_scratch_spec(engine, /*max_tokens=*/0);

    const CudaMemoryPlan mem_plan = plan_cuda_memory(
        cfg, engine.hf_config(), max_mlp_intermediate, max_Hq, max_Hk,
        family == model::Family::Gemma4, plan_info.per_layer_head_dim,
        plan_info.kv_source_layer, family == model::Family::Qwen3_5,
        family == model::Family::Qwen3_5Moe, qwen3_5_linear_layers,
        family == model::Family::NemotronH, nemotron_h_mamba_layers,
        family == model::Family::DeepSeekV4,
        family == model::Family::Kimi,
        family == model::Family::Glm5,
        kv_format, runtime_quant_scratch_base, verbose);
    std::size_t free_device_bytes = 0;
    std::size_t total_device_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_device_bytes, &total_device_bytes));
    elastic_safety_floor_bytes_ =
        std::min<std::size_t>(128ull << 20, total_device_bytes / 10);
    const std::size_t elastic_budget =
        free_device_bytes > elastic_safety_floor_bytes_
            ? free_device_bytes - elastic_safety_floor_bytes_
            : 0;
    elastic_pool_ = std::make_shared<CudaPhysicalPool>(
        device_ordinal_,
        elastic_budget,
        cuda_vmm_handle_bytes());
    kv_allocator_ = std::make_shared<CudaArenaAllocator>(
        elastic_pool_, "kv", false);
    state_allocator_ = std::make_shared<CudaArenaAllocator>(
        elastic_pool_, "state", true);
    workspace_allocator_ = std::make_shared<CudaArenaAllocator>(
        elastic_pool_, "workspace", false);
    attention_allocator_ = std::make_shared<CudaArenaAllocator>(
        elastic_pool_, "attention", false);
    const int max_workspace_tokens = mem_plan.max_workspace_tokens;
    if (native_mtp_num_drafts > 0 &&
        mem_plan.max_requests >
            std::numeric_limits<int>::max() / native_mtp_num_drafts) {
        std::cerr << "[pie-driver-cuda] aggregate MTP draft capacity overflows\n";
        return PIE_STATUS_UNSUPPORTED;
    }
    const int aggregate_mtp_draft_capacity =
        plan_info.has_mtp
            ? mem_plan.max_requests * native_mtp_num_drafts
            : 0;
    const long kv_page_cap = [&]() -> long {
        if (const char* e = std::getenv("PIE_KV_PAGE_CAP")) {
            const long v = std::atol(e);
            if (v > 0) return v;
        }
        return static_cast<long>(cfg.batching.total_pages);
    }();
    if (mem_plan.kv_page_bytes == 0) {
        std::cerr << "[pie-driver-cuda] zero KV page byte coefficient\n";
        return PIE_STATUS_UNSUPPORTED;
    }
    const int logical_kv_pages = static_cast<int>(
        std::min<std::size_t>(
            elastic_budget / mem_plan.kv_page_bytes,
            static_cast<std::size_t>(std::numeric_limits<int>::max())));
    const int runtime_kv_pages = (kv_page_cap > 0)
        ? std::min<int>(logical_kv_pages, static_cast<int>(kv_page_cap))
        : logical_kv_pages;
    const bool page_zero_dummy_safe =
        kv_format.is_native_bf16() &&
        (family == model::Family::LlamaLike ||
         family == model::Family::Qwen3VL ||
         family == model::Family::Qwen3_5 ||
         family == model::Family::Qwen3_5Moe ||
         family == model::Family::Gemma4 ||
         family == model::Family::NemotronH) ||
        family == model::Family::Kimi ||
        family == model::Family::Glm5;
    const int physical_kv_pages =
        runtime_kv_pages +
        (runtime_kv_pages > 0 && !page_zero_dummy_safe ? 1 : 0);
    const int graph_pad_page = runtime_kv_pages > 0
        ? (page_zero_dummy_safe ? 0 : runtime_kv_pages)
        : -1;
    const bool has_recurrent_state_cache =
        (family == model::Family::Qwen3_5 ||
         family == model::Family::Qwen3_5Moe) &&
        qwen3_5_linear_layers > 0
#ifndef PIE_CUDA_QWEN_ONLY
        || (family == model::Family::NemotronH &&
            nemotron_h_mamba_layers > 0)
#endif
        ;
    const int runtime_state_slots =
        has_recurrent_state_cache ? mem_plan.max_requests : 0;
    const int graph_pad_slot =
        has_recurrent_state_cache && runtime_state_slots > 0 && graph_pad_page >= 0
            ? runtime_state_slots
            : -1;
    const int allocated_state_slots =
        runtime_state_slots + (graph_pad_slot >= 0 ? 1 : 0);

    model::Workspace* ws_p = nullptr;
    {
        ScopedCudaArenaAllocator arena(*workspace_allocator_);
        ws_p = own_value(model::Workspace::allocate_full(
            engine.hf_config(), max_workspace_tokens,
            max_mlp_intermediate, max_Hq, max_Hk,
            mem_plan.capacity.max_logit_rows,
            aggregate_mtp_draft_capacity));
    }
    auto& ws = *ws_p;

    // KV-cache shape genuinely differs per family (MLA-backed families use
    // a 1x1 placeholder KvCache; Gemma-4 has per-layer head_dim + KV
    // sharing; Nemotron-H only allocates pages for its attention layers).
    // This is exactly the kind of family-specific store allocation
    // `Context` is allowed to switch on (cpp-refact.md Phase 5) — the
    // registry alone decided `family` above.
    KvCache* kv_cache_p = nullptr;
    {
        ScopedCudaArenaAllocator arena(*kv_allocator_);
        kv_cache_p = own_value([&]() -> KvCache {
            switch (family) {
        case model::Family::DeepSeekV4:
            return KvCache::allocate(
                engine.hf_config().num_hidden_layers,
                physical_kv_pages,
                mem_plan.kv_page_size,
                1,
                engine.hf_config().head_dim,
                kv_format);
        case model::Family::Kimi:
        case model::Family::Glm5:
            return KvCache::allocate(
                engine.hf_config().num_hidden_layers,
                physical_kv_pages,
                mem_plan.kv_page_size,
                1,
                1,
                kv_format);
        case model::Family::Gemma4:
            return KvCache::allocate_per_layer(
                engine.hf_config().num_hidden_layers,
                physical_kv_pages,
                mem_plan.kv_page_size,
                local_kv_heads,
                plan_info.per_layer_head_dim,
                plan_info.kv_source_layer,
                plan_info.per_layer_num_kv_heads,
                kv_format);
        case model::Family::NemotronH:
            return KvCache::allocate(
                nemotron_h_attention_layer_count,
                physical_kv_pages,
                mem_plan.kv_page_size,
                local_kv_heads,
                engine.hf_config().head_dim_kernel,
                kv_format);
        default:
            return KvCache::allocate(
                engine.hf_config().num_hidden_layers,
                physical_kv_pages,
                mem_plan.kv_page_size,
                local_kv_heads,
                engine.hf_config().head_dim_kernel,
                kv_format);
            }
        }());
    }
    auto& kv_cache = *kv_cache_p;
    kv_cache.set_elastic_allocator(kv_allocator_);

#ifdef PIE_CUDA_QWEN_ONLY
    auto* mla_cache_p = own_value(MlaCache{});
    auto* dsa_cache_p = own_value(DsaCache{});
#else
    MlaCache* mla_cache_p = nullptr;
    {
        ScopedCudaArenaAllocator arena(*kv_allocator_);
        mla_cache_p = own_value(
            (family == model::Family::Kimi || family == model::Family::Glm5)
                ? MlaCache::allocate(
                      engine.hf_config().num_hidden_layers,
                      physical_kv_pages,
                      mem_plan.kv_page_size,
                      engine.hf_config().kv_lora_rank,
                      engine.hf_config().qk_rope_head_dim,
                      DType::BF16)
                : MlaCache{});
    }

    auto* dsa_cache_p = own_value(
        family == model::Family::Glm5
            ? DsaCache::allocate(
                  engine.hf_config().num_hidden_layers,
                  engine.hf_config().max_position_embeddings,
                  engine.hf_config().index_head_dim)
            : DsaCache{});
#endif
    auto& mla_cache = *mla_cache_p;
    auto& dsa_cache = *dsa_cache_p;

    AttentionWorkspace* attn_ws_p = nullptr;
    {
        ScopedCudaArenaAllocator arena(*attention_allocator_);
        attn_ws_p = own_value(AttentionWorkspace::allocate(
            mem_plan.attn_float_workspace_bytes, 8ull * 1024 * 1024));
    }
    auto& attn_ws = *attn_ws_p;

    auto* qwen3_5_plan_state_p = own_emplace<model::Qwen3_5PlanState>();
    auto& qwen3_5_plan_state = *qwen3_5_plan_state_p;
    auto* qwen3_5_la_ws_p = own_emplace<model::Qwen3_5LinearAttnWorkspace>();
    auto& qwen3_5_la_ws = *qwen3_5_la_ws_p;
    auto* qwen3_5_state_cache_p = own_emplace<RecurrentStateCache>();
    auto& qwen3_5_state_cache = *qwen3_5_state_cache_p;
    auto* qwen3_5_moe_ws_p = own_emplace<model::Qwen3_5MoeMlpWorkspace>();
    auto& qwen3_5_moe_ws = *qwen3_5_moe_ws_p;
#ifndef PIE_CUDA_QWEN_ONLY
    auto* nemotron_h_ws_p = own_emplace<model::NemotronHWorkspace>();
    auto& nemotron_h_ws = *nemotron_h_ws_p;
    auto* nemotron_h_state_cache_p = own_emplace<RecurrentStateCache>();
    auto& nemotron_h_state_cache = *nemotron_h_state_cache_p;
#endif
    int qwen3_5_runtime_rs_slots = 0;

    if (family == model::Family::Qwen3_5 || family == model::Family::Qwen3_5Moe) {
        const auto& cfg_q = engine.hf_config();
        const int local_linear_key_heads =
            cfg_q.linear_num_key_heads / local_tp_size;
        const int local_linear_value_heads =
            cfg_q.linear_num_value_heads / local_tp_size;
        const int K_dim = local_linear_key_heads * cfg_q.linear_key_head_dim;
        const int V_dim = local_linear_value_heads * cfg_q.linear_value_head_dim;
        const int conv_dim = 2 * K_dim + V_dim;
        {
            ScopedCudaArenaAllocator arena(*workspace_allocator_);
            qwen3_5_la_ws = model::Qwen3_5LinearAttnWorkspace::allocate(
                max_workspace_tokens, conv_dim,
                local_linear_value_heads,
                local_linear_key_heads,
                cfg_q.linear_key_head_dim,
                cfg_q.linear_value_head_dim,
                (cfg_q.num_attention_heads / local_tp_size) * cfg_q.head_dim);
        }
        qwen3_5_runtime_rs_slots = std::max(1, runtime_state_slots);
        {
            ScopedCudaArenaAllocator arena(*state_allocator_);
            qwen3_5_state_cache = RecurrentStateCache::allocate(
                qwen3_5_layer_is_linear, conv_dim, cfg_q.linear_conv_kernel_dim,
                local_linear_value_heads,
                cfg_q.linear_key_head_dim,
                cfg_q.linear_value_head_dim,
                cfg_q.hidden_size,
                qwen3_5_runtime_rs_slots);
        }
        const int stash_width = conv_dim + 2 * local_linear_value_heads;
        {
            ScopedCudaArenaAllocator arena(*state_allocator_);
            qwen3_5_state_cache.configure_verify_hidden_stash(
                max_workspace_tokens, stash_width);
            qwen3_5_state_cache.configure_rs_buffer_pool(
                mem_plan.kv_page_size, stash_width, qwen3_5_runtime_rs_slots);
        }
        if (family == model::Family::Qwen3_5Moe) {
            ScopedCudaArenaAllocator arena(*workspace_allocator_);
            qwen3_5_moe_ws = model::Qwen3_5MoeMlpWorkspace::allocate(
                max_workspace_tokens,
                cfg_q.hidden_size,
                cfg_q.num_experts,
                cfg_q.num_experts_per_tok,
                cfg_q.moe_intermediate_size / local_tp_size,
                cfg_q.shared_expert_intermediate_size / local_tp_size);
        }
    }
#ifndef PIE_CUDA_QWEN_ONLY
    else if (family == model::Family::NemotronH) {
        const auto& cfg_n = engine.hf_config();
        {
            ScopedCudaArenaAllocator arena(*workspace_allocator_);
            nemotron_h_ws = model::NemotronHWorkspace::allocate(
                cfg_n, max_workspace_tokens, local_tp_size);
        }
        const bool shard_mamba =
            model::nemotron_h_tp_mamba_sharding_enabled(local_tp_size);
        const int local_mamba_heads = shard_mamba
            ? cfg_n.mamba_num_heads / local_tp_size
            : cfg_n.mamba_num_heads;
        const int local_mamba_groups = shard_mamba
            ? cfg_n.mamba_n_groups / local_tp_size
            : cfg_n.mamba_n_groups;
        const int m_intermediate = local_mamba_heads * cfg_n.mamba_head_dim;
        const int conv_dim =
            m_intermediate + 2 * local_mamba_groups * cfg_n.mamba_state_size;
        {
            ScopedCudaArenaAllocator arena(*state_allocator_);
            nemotron_h_state_cache = RecurrentStateCache::allocate_bf16_recurrent(
                nemotron_h_layer_is_mamba,
                conv_dim,
                cfg_n.mamba_conv_kernel,
                local_mamba_heads,
                cfg_n.mamba_head_dim,
                cfg_n.mamba_state_size,
                std::max<int>(1, allocated_state_slots));
        }
    }
#endif

    auto* swap_pool_p = own_value(SwapPool::allocate_for_cache(
        kv_cache, static_cast<int>(cfg.batching.swap_pool_size)));
    auto& swap_pool = *swap_pool_p;
    kv_cache_ = &kv_cache;
    swap_pool_ = &swap_pool;

    auto* cublas_p = own_emplace<ops::CublasHandle>();
    auto& cublas = *cublas_p;
    auto runtime_quant_scratch = runtime_quant_scratch_base;
    runtime_quant_scratch.max_tokens = static_cast<std::size_t>(max_workspace_tokens);
    if (!runtime_quant_scratch.empty()) {
        runtime_quant_context_.reset();
        {
            ScopedCudaArenaAllocator arena(*workspace_allocator_);
            ops::reserve_runtime_quant_scratch(runtime_quant_scratch, true);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto* persistent_inputs_p = own_value(PersistentInputs::allocate(
        max_workspace_tokens,
        mem_plan.max_requests,
        mem_plan.max_page_refs,
        mem_plan.capacity.max_custom_mask_bytes,
        aggregate_mtp_draft_capacity));
    auto& persistent_inputs = *persistent_inputs_p;

    model::LlamaLikeForwardCfg fwd_cfg{};
#ifndef PIE_CUDA_QWEN_ONLY
    model::Gemma2ForwardCfg gemma_fwd_cfg{};
    model::Gemma4ForwardCfg gemma4_fwd_cfg{};
#endif
    {
        const auto& hf = engine.hf_config();
        const std::string& mt = hf.model_type;
        fwd_cfg.use_qk_norm = hf.use_qk_norm;
        fwd_cfg.use_qkv_bias = hf.attention_bias;
        const bool is_olmo_post_norm = (mt == "olmo2" || mt == "olmo3");
        fwd_cfg.norm_placement = is_olmo_post_norm
            ? model::NormPlacement::Post
            : model::NormPlacement::Pre;
        if (is_olmo_post_norm) fwd_cfg.use_qk_norm = true;
        model::apply_rope_config(fwd_cfg, hf);
        if (family == model::Family::Qwen3VL && hf.qwen3_vl_mrope_interleaved &&
            hf.qwen3_vl_mrope_section.size() == 3) {
            fwd_cfg.rope_kind = model::RopeKind::MRopeInterleaved;
            fwd_cfg.mrope_section_t = hf.qwen3_vl_mrope_section[0];
            fwd_cfg.mrope_section_h = hf.qwen3_vl_mrope_section[1];
            fwd_cfg.mrope_section_w = hf.qwen3_vl_mrope_section[2];
        }
        fwd_cfg.sliding_window = hf.sliding_window;
        const int gqa = hf.num_attention_heads / hf.num_key_value_heads;
        const bool gqa_in_decode_set = flashinfer_decode_supports_gqa(gqa);
        fwd_cfg.force_prefill_path = !gqa_in_decode_set;
        fwd_cfg.decode_plan_cuda_graph = use_cuda_graphs;
        fwd_cfg.tp_size = local_tp_size;
        fwd_cfg.tp_comm = tp_comm_ptr;
        fwd_cfg.emit_logits = true;
        fwd_cfg.use_xqa_decode =
            xqa_decode_enabled_by_env() &&
            ops::xqa_decode_bf16_supported(
                local_q_heads, local_kv_heads, hf.head_dim_kernel,
                mem_plan.kv_page_size, hf.sliding_window,
                0.f, -1.f) &&
            !has_non_full_attention_layers(hf);
        if (fwd_cfg.use_xqa_decode) {
            fwd_cfg.force_prefill_path = false;
            if (local_q_heads > 0 && local_kv_heads > 0 &&
                local_q_heads % local_kv_heads == 0) {
                ops::xqa_decode_bf16_warmup_current_device(
                    local_q_heads / local_kv_heads, mem_plan.kv_page_size);
            }
        }
#ifndef PIE_CUDA_QWEN_ONLY
        gemma_fwd_cfg.query_pre_attn_scalar =
            hf.gemma_query_pre_attn_scalar;
        gemma_fwd_cfg.final_logit_softcap =
            hf.gemma_final_logit_softcap;
        gemma_fwd_cfg.attn_logit_softcap =
            hf.gemma_attn_logit_softcap;
        gemma_fwd_cfg.use_qk_norm =
            mt == "gemma3" || mt == "gemma3_text";
        gemma_fwd_cfg.force_prefill_path = !gqa_in_decode_set;
        gemma_fwd_cfg.tp_size = local_tp_size;
        gemma_fwd_cfg.tp_comm = tp_comm_ptr;
#endif
        const bool homogeneous = !has_non_full_attention_layers(hf);
        if (!homogeneous) {
#ifndef PIE_CUDA_QWEN_ONLY
            gemma_fwd_cfg.per_layer_window_left.reserve(hf.layer_types.size());
            gemma_fwd_cfg.per_layer_rope_theta.reserve(hf.layer_types.size());
#endif
            fwd_cfg.per_layer_window_left.reserve(hf.layer_types.size());
            for (const auto& t : hf.layer_types) {
                const bool is_sliding = (t == "sliding_attention");
                const int window = is_sliding ? hf.sliding_window : -1;
#ifndef PIE_CUDA_QWEN_ONLY
                gemma_fwd_cfg.per_layer_window_left.push_back(window);
#endif
                fwd_cfg.per_layer_window_left.push_back(window);
#ifndef PIE_CUDA_QWEN_ONLY
                const float theta =
                    (is_sliding && hf.rope_local_base_freq > 0.f)
                        ? hf.rope_local_base_freq
                        : hf.rope_theta;
                gemma_fwd_cfg.per_layer_rope_theta.push_back(theta);
#endif
            }
        }
        cudaDeviceProp serving_prop{};
        CUDA_CHECK(cudaGetDeviceProperties(&serving_prop, device_ordinal_));
        const bool prefill_decode_supported_head_dim =
            hf.head_dim_kernel == 64 || hf.head_dim_kernel == 128 ||
            hf.head_dim_kernel == 256 || hf.head_dim_kernel == 512;
        const bool force_prefill_decode_plan = [] {
            const char* v = std::getenv("PIE_CUDA_PREFILL_DECODE_PLAN");
            return v != nullptr && v[0] != '\0' && v[0] != '0';
        }();
        fwd_cfg.use_prefill_decode_plan =
            (serving_prop.major >= 9 || force_prefill_decode_plan) &&
            gqa_in_decode_set && !fwd_cfg.force_prefill_path &&
            prefill_decode_supported_head_dim &&
            fwd_cfg.sliding_window < 0 &&
            fwd_cfg.per_layer_window_left.empty();
        if (fwd_cfg.use_prefill_decode_plan) {
            const std::size_t rank_kv_token_bytes = kv_page_bytes_homogeneous(
                hf, 1, kv_format);
            const bool kv_heavy_attention =
                rank_kv_token_bytes >= 192ull * 1024ull;
            fwd_cfg.prefill_decode_min_kv_pages = kv_heavy_attention ? 1 : 7;
            fwd_cfg.prefill_decode_full_attention_min_requests = 256;
            fwd_cfg.prefill_decode_full_attention_min_kv_pages =
                kv_heavy_attention ? 1 : 7;
        }
    }
#ifndef PIE_CUDA_QWEN_ONLY
    if (family == model::Family::Gemma4) {
        const auto& hf = engine.hf_config();
        gemma4_fwd_cfg.final_logit_softcap = hf.gemma_final_logit_softcap;
        const int gqa = hf.num_attention_heads / hf.num_key_value_heads;
        gemma4_fwd_cfg.force_prefill_path = !flashinfer_decode_supports_gqa(gqa);
        gemma4_fwd_cfg.tp_size = local_tp_size;
        gemma4_fwd_cfg.tp_comm = tp_comm_ptr;
    }
#endif

    ForwardFn forward_fn;
    NativeSystemDrafter system_drafter;
    auto* graph_cache_p = use_cuda_graphs ? own_emplace<ForwardGraphCache>() : nullptr;

#ifndef PIE_CUDA_QWEN_ONLY
    model::DsV4Workspace* dsv4_ws_p = nullptr;
    model::KimiWorkspace* kimi_ws_p = nullptr;
    model::Glm5Workspace* glm5_ws_p = nullptr;
    {
        ScopedCudaArenaAllocator arena(*workspace_allocator_);
        dsv4_ws_p = own_value(
            family == model::Family::DeepSeekV4
                ? model::DsV4Workspace::allocate(
                      engine.hf_config(), max_workspace_tokens,
                      mem_plan.capacity.max_logit_rows, local_tp_size)
                : model::DsV4Workspace{});
        kimi_ws_p = own_value(
            family == model::Family::Kimi
                ? model::KimiWorkspace::allocate(
                      engine.hf_config(), max_workspace_tokens,
                      mem_plan.capacity.max_logit_rows, local_tp_size)
                : model::KimiWorkspace{});
        glm5_ws_p = own_value(
            family == model::Family::Glm5
                ? model::Glm5Workspace::allocate(
                      engine.hf_config(), max_workspace_tokens,
                      mem_plan.capacity.max_logit_rows,
                      engine.hf_config().max_position_embeddings, local_tp_size)
                : model::Glm5Workspace{});
    }
    auto& dsv4_ws = *dsv4_ws_p;
    auto& kimi_ws = *kimi_ws_p;
    auto& glm5_ws = *glm5_ws_p;
    auto* gemma4_moe_ws_p = own_emplace<model::Gemma4MoeMlpWorkspace>();
    auto& gemma4_moe_ws = *gemma4_moe_ws_p;
    const bool gemma4_selected = (family == model::Family::Gemma4);
    if (gemma4_selected && engine.hf_config().gemma4_enable_moe) {
        const auto& hf_cfg = engine.hf_config();
        ScopedCudaArenaAllocator arena(*workspace_allocator_);
        gemma4_moe_ws = model::Gemma4MoeMlpWorkspace::allocate(
            max_workspace_tokens,
            hf_cfg.hidden_size,
            hf_cfg.num_experts,
            hf_cfg.num_experts_per_tok,
            hf_cfg.moe_intermediate_size);
    }
    if (gemma4_selected) {
        ScopedCudaArenaAllocator arena(*workspace_allocator_);
        gemma4_moe_ws.allocate_row_decode(max_workspace_tokens);
    }
    if (gemma4_selected &&
        engine.hf_config().gemma_hidden_size_per_layer_input > 0) {
        const auto& hf_cfg = engine.hf_config();
        ScopedCudaArenaAllocator arena(*workspace_allocator_);
        gemma4_moe_ws.allocate_ple(
            max_workspace_tokens,
            hf_cfg.num_hidden_layers *
                hf_cfg.gemma_hidden_size_per_layer_input);
    }
#endif

    // The registry owns architecture selection: exactly one `create_model`
    // call constructs the `IModel` from the bound plan plus every
    // Context-owned resource a family's factory might read. No model_type
    // string chain, no per-arch construction blocks here anymore.
    model::ModelResources resources;
    resources.hf_config = &engine.hf_config();
    resources.kv_cache = &kv_cache;
    resources.mla_cache = &mla_cache;
    resources.dsa_cache = &dsa_cache;
    resources.tp_size = local_tp_size;
    resources.tp_rank = tp_rank_;
    resources.tp_comm = tp_comm_ptr;
    resources.verbose = verbose;
    resources.llama_fwd_cfg = &fwd_cfg;
    resources.max_workspace_tokens = max_workspace_tokens;
    resources.small_spec_graph_tokens =
        model::qwen35_small_spec_graph_tokens();
    resources.qwen3_5_la_ws = &qwen3_5_la_ws;
    resources.qwen3_5_moe_ws = &qwen3_5_moe_ws;
    resources.qwen3_5_plan_state = &qwen3_5_plan_state;
    resources.qwen3_5_state_cache = &qwen3_5_state_cache;
    resources.system_drafter = &system_drafter;
    resources.native_mtp_num_drafts = native_mtp_num_drafts;
#ifndef PIE_CUDA_QWEN_ONLY
    resources.gemma_fwd_cfg = &gemma_fwd_cfg;
    resources.gemma4_fwd_cfg = &gemma4_fwd_cfg;
    resources.gemma4_moe_ws = &gemma4_moe_ws;
    resources.nemotron_h_ws = &nemotron_h_ws;
    resources.nemotron_h_state_cache = &nemotron_h_state_cache;
    resources.dsv4_ws = &dsv4_ws;
    resources.kimi_ws = &kimi_ws;
    resources.glm5_ws = &glm5_ws;
#endif

    auto* model_holder =
        own_value(arch_entry->create_model(std::move(plan), resources));
    model_ = model_holder->get();
    forward_fn.attach_model(model_);

    // The pipeline registry (program/instance/channel ownership + the single
    // `Dispatch` instance) is constructed once, here, ahead of the batch
    // engine so the engine can hold a non-owning pointer into it for the
    // whole of its lifetime. It is emplaced before `BatchEngine` in
    // `owners_`, so it is torn down after the engine at destruction
    // (reverse-of-construction order): the engine's `dispatch` pointer is
    // never dangling.
    registry_ = own_emplace<pipeline::Registry>();
    // W2: size the channel registry for the fleet at load. Live slots
    // scale with concurrent instances (measured on the c0 bench: 256
    // processes x ~18 slots ~= 4.6k vs the 1024 default), and every
    // mid-ramp grow() costs a device-wide sync on the lane thread.
    // x48 = the audited per-request slot budget (two passes x 9 slots,
    // sampling rng included, x32) plus headroom for one cohort boundary's
    // close-lag overlap — teardown closes now drain BEHIND the fresh
    // cohort's binds, so both generations' slots are briefly live at
    // once. The registry rounds up to its power-of-two ladder; ~24k
    // slots stays under 2 MB of device arrays, and every mid-ramp grow()
    // costs a device-wide sync on the lane thread.
    registry_->dispatch().reserve_channel_slots(static_cast<std::uint32_t>(
        std::max(1024, mem_plan.capacity.max_forward_requests * 48)));
    const bool complete_attention_hook_coverage =
        tp_size_ == 1
#ifndef PIE_CUDA_QWEN_ONLY
        && family != model::Family::Csm &&
        !(family == model::Family::NemotronH &&
          nemotron_h_attention_layer_count !=
              engine.hf_config().num_hidden_layers)
#endif
        ;
    registry_->dispatch().set_attention_hook_coverage(
        complete_attention_hook_coverage,
        static_cast<std::uint32_t>(
            std::max(0, engine.hf_config().num_hidden_layers)));

    BatchEngine* executor_p = nullptr;
    {
        ScopedCudaArenaAllocator arena(*workspace_allocator_);
        executor_p = own_emplace<BatchEngine>(
            engine, ws, kv_cache, attn_ws, cublas,
            max_workspace_tokens,
            mem_plan.max_requests,
            graph_pad_page,
            graph_pad_slot,
            persistent_inputs,
            verbose,
            std::move(forward_fn),
            std::move(system_drafter),
            graph_cache_p,
            tp_comm_ptr,
            tp_cpu_gate_key_,
            &runtime_quant_context_,
            !has_recurrent_state_cache
                ? nullptr
                : ((family == model::Family::Qwen3_5 ||
                    family == model::Family::Qwen3_5Moe)
                       ? &qwen3_5_state_cache
#ifndef PIE_CUDA_QWEN_ONLY
                       : &nemotron_h_state_cache
#else
                       : nullptr
#endif
                  ));
    }
    executor_p->dispatch = &registry_->dispatch();
    executor_ = executor_p;
    const bool has_usable_mtp_logits =
        has_mtp_logits && static_cast<bool>(executor_p->system_drafter);

    tp_startup_cpu_barrier(cfg);
    // Upfront lattice capture for EVERY topology (V6 iteration 53). Lazy
    // first-use capture pays ~10 ms of capture+instantiate INSIDE
    // `driver.launch` for each decode bucket a cohort ramp touches — a
    // mid-run submit-tail tax (measured spikes to 6.7 ms) that also varies
    // run-to-run with the ramp shape. TP followers additionally need the
    // lattice synchronized across ranks. Startup pays once; the
    // PIE_CUDA_DISABLE_UPFRONT_GRAPHS escape (checked inside) restores the
    // lazy behavior.
    if (use_cuda_graphs) {
        workspace_allocator_->ensure_all();
        attention_allocator_->ensure_all();
        state_allocator_->ensure_all();
        capture_forward_graph_lattice(*executor_);
        if (!is_tp_follower()) {
            workspace_allocator_->trim_bytes(pie::elastic::kLogicalPageBytes);
            attention_allocator_->trim_bytes(0);
            if (tp_size_ == 1) {
                state_allocator_->trim_bytes(0);
            }
        }
    }
    tp_startup_cpu_barrier(cfg);
    recalibrate_elastic_budget(/*reset_hard_ceiling=*/true);

    if (is_tp_follower()) {
        tp_follower_stop_.store(false);
        tp_follower_thread_ = std::thread([this, verbose]() {
            if (verbose) {
                std::cerr << "[pie-driver-cuda] tp follower rank "
                         << tp_rank_
                         << " ready (waiting on rank-0 broadcasts"
                         << (tp_cpu_gate_key_.empty() ? ", cpu_gate=off" : ", cpu_gate=on")
                         << ")\n";
            }
            try {
                pie_cuda_driver::tp_follower_serve(*executor_, tp_follower_stop_);
            } catch (const std::exception& e) {
                if (!tp_follower_stop_.load()) {
                   std::cerr << "[pie-driver-cuda] tp follower rank "
                             << tp_rank_ << " exited: " << e.what() << "\n";
                }
            } catch (...) {
                if (!tp_follower_stop_.load()) {
                   std::cerr << "[pie-driver-cuda] tp follower rank "
                             << tp_rank_ << " exited with unknown error\n";
                }
            }
        });
    }

    auto c = engine.capabilities();
    c.total_pages = runtime_kv_pages;
    c.swap_pool_size = swap_pool.num_pages();
    const bool rs_cache_required = has_recurrent_state_cache && runtime_state_slots > 0;
#ifdef PIE_CUDA_QWEN_ONLY
    const std::uint64_t rs_cache_slots = rs_cache_required
        ? static_cast<std::uint64_t>(qwen3_5_runtime_rs_slots)
        : 0;
    const std::uint64_t rs_cache_slot_bytes = rs_cache_required
        ? static_cast<std::uint64_t>(qwen3_5_linear_layers) *
              (qwen3_5_state_cache.conv_slot_stride_bytes() +
               qwen3_5_state_cache.recurrent_slot_stride_bytes()) +
              static_cast<std::uint64_t>(
                  std::max(0, qwen3_5_state_cache.hidden_size())) *
                  sizeof(std::uint16_t)
        : 0;
#else
    const std::uint64_t rs_cache_slots = rs_cache_required
        && family == model::Family::NemotronH
            ? static_cast<std::uint64_t>(runtime_state_slots)
            : (rs_cache_required
                   ? static_cast<std::uint64_t>(qwen3_5_runtime_rs_slots)
                   : 0);
    const std::uint64_t rs_cache_slot_bytes = !rs_cache_required
        ? 0
        : family == model::Family::NemotronH
            ? static_cast<std::uint64_t>(model::nemotron_h_state_slot_bytes(
                  engine.hf_config(), nemotron_h_mamba_layers, local_tp_size))
        : static_cast<std::uint64_t>(qwen3_5_linear_layers) *
              (qwen3_5_state_cache.conv_slot_stride_bytes() +
               qwen3_5_state_cache.recurrent_slot_stride_bytes()) +
          static_cast<std::uint64_t>(
              std::max(0, qwen3_5_state_cache.hidden_size())) *
              sizeof(std::uint16_t);
#endif
    const auto max_forward_requests_caps = rs_cache_required
        ? std::min<std::uint64_t>(
              static_cast<std::uint64_t>(mem_plan.capacity.max_forward_requests),
              rs_cache_slots)
        : static_cast<std::uint64_t>(mem_plan.capacity.max_forward_requests);
    nlohmann::json kv_regions = nlohmann::json::array();
    nlohmann::json kv_region_page_bytes = nlohmann::json::array();
    std::unordered_set<std::uintptr_t> exported_kv_buffers;
    for (int layer = 0; layer < kv_cache.num_layers(); ++layer) {
        for (const auto& buffer : kv_cache.page_buffers(layer)) {
            const auto base = reinterpret_cast<std::uintptr_t>(buffer.data);
            if (base == 0 || buffer.page_bytes == 0 ||
                !exported_kv_buffers.insert(base).second) {
                continue;
            }
            kv_regions.push_back({
                {"base", static_cast<std::uint64_t>(base)},
                {"len", static_cast<std::uint64_t>(buffer.page_bytes) *
                            static_cast<std::uint64_t>(kv_cache.num_pages())},
                {"page_stride", static_cast<std::uint64_t>(buffer.page_bytes)},
                {"domain", {{"CudaDevice", device_ordinal_}}},
            });
            kv_region_page_bytes.push_back(
                static_cast<std::uint64_t>(buffer.page_bytes));
        }
    }
    nlohmann::json kv_handle = {
        {"regions", std::move(kv_regions)},
        {"layout",
         {
             {"num_layers", static_cast<std::uint32_t>(kv_cache.num_layers())},
             {"num_kv_heads", static_cast<std::uint32_t>(kv_cache.num_kv_heads())},
             {"head_dim", static_cast<std::uint32_t>(kv_cache.head_dim())},
             {"page_size", static_cast<std::uint32_t>(kv_cache.page_size())},
             {"dtype", kv_cache.format().is_native_bf16() ? "Bf16" : "I8"},
             {"kind", "KvSeparate"},
             {"storage_format",
              kv_cache.format().name + ":" +
                  std::to_string(static_cast<int>(kv_cache.format().scheme)) + ":" +
                  std::to_string(static_cast<int>(kv_cache.format().scale_layout)) + ":" +
                  std::to_string(static_cast<int>(kv_cache.format().storage_dtype)) + ":" +
                  std::to_string(kv_cache.format().block_size) + ":" +
                  (kv_cache.hnd_layout() ? "hnd" : "nhd")},
             {"region_page_bytes", std::move(kv_region_page_bytes)},
         }},
    };
    if (family == model::Family::Kimi || family == model::Family::Glm5) {
        // These families decode from MlaCache (and GLM5 also DsaCache);
        // kv_cache is only a 1x1 compatibility placeholder.
        kv_handle = nullptr;
    }
    nlohmann::json caps = {
        {"abi_version", PIE_DRIVER_ABI_VERSION},
        {"total_pages", c.total_pages},
        {"kv_page_size", mem_plan.kv_page_size},
        {"swap_pool_size", c.swap_pool_size},
        {"kv_copy_domain_mask", local_tp_size == 1 ? 15u : 1u},
        {"rs_cache_required", rs_cache_required},
        {"rs_cache_slots", rs_cache_slots},
        {"rs_cache_slot_bytes", rs_cache_slot_bytes},
        {"elastic_page_bytes", elastic_pool_->page_bytes()},
        {"elastic_budget_pages", elastic_pool_->budget_pages()},
        {"has_mtp_logits", has_usable_mtp_logits},
        {"has_mtp_drafts", false},
        {"has_value_head", false},
        // RV-26: PIE_DEVICE_PORT_ATTN_MASK is deliberately NOT advertised.
        // The runtime classifies masked device-carried decode into the
        // DecodeEnvelope class exactly when this mask claims the port, but
        // this driver cannot execute that class: the envelope verifier
        // (is_decode_envelope_trace) has no AttnMask arm and both envelope
        // compose paths (enqueue_fixed_decode / enqueue_decode_envelopes)
        // carry no per-lane mask state — so a claimed mask port turned the
        // classifier's loud Host fallback into a bind error. Re-advertise
        // only together with all three: a verifier arm for the port,
        // per-lane mask state through both composes, and per-row mask
        // application in the composed decode attention. (The general
        // resolved path does consume masks, but an envelope-class batch
        // must compose — there is no per-fire fallback past composition.)
        {"device_geometry_port_mask", PIE_DEVICE_GEOMETRY_PORTS},
        {"max_forward_tokens", mem_plan.capacity.max_forward_tokens},
        {"max_forward_requests", max_forward_requests_caps},
        {"max_page_refs", mem_plan.capacity.max_page_refs},
        {"arch_name", c.arch_name},
        {"vocab_size", c.vocab_size},
        {"max_model_len", c.max_model_len},
        {"activation_dtype", c.activation_dtype},
        {"hidden_size", static_cast<std::uint32_t>(engine.hf_config().hidden_size)},
        {"supports_media_encode", model_->capabilities().supports_media_encode},
        {"snapshot_dir", c.snapshot_dir},
        {"kv_handle", std::move(kv_handle)},
    };
    caps_json_ = caps.dump();
    if (caps_out != nullptr) {
        caps_out->json_bytes =
            reinterpret_cast<const std::uint8_t*>(caps_json_.data());
        caps_out->json_len = caps_json_.size();
    }
    return PIE_STATUS_OK;
}

void Context::Impl::recalibrate_elastic_budget(bool reset_hard_ceiling) {
    if (elastic_pool_ == nullptr) return;
    std::size_t free_device_bytes = 0;
    std::size_t total_device_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_device_bytes, &total_device_bytes));
    static_cast<void>(total_device_bytes);
    elastic_pool_->recalibrate_budget(
        free_device_bytes,
        elastic_safety_floor_bytes_,
        reset_hard_ceiling);
}

int Context::Impl::required_kv_pages(const pie_native::StepLaunch& launch) const {
    int pages = static_cast<int>(launch.required_kv_pages);
    if (executor_->graph_pad_page >= 0 && kv_cache_->num_pages() > 0) {
        pages = std::max(pages, executor_->graph_pad_page + 1);
    }
    if (launch.kv_page_indices.len > 0) {
        const auto* end =
            launch.kv_page_indices.ptr + launch.kv_page_indices.len;
        const std::uint32_t highest =
            *std::max_element(launch.kv_page_indices.ptr, end);
        pages = std::max(pages, static_cast<int>(highest) + 1);
    }
    return pages;
}

std::size_t Context::Impl::required_state_slots(
    const pie_native::StepLaunch& launch) const {
    if (executor_->rs_cache == nullptr) return 0;
    std::size_t slots = 0;
    auto include = [&slots](PieU32Slice ids) {
        if (ids.len == 0) return;
        const auto* end = ids.ptr + ids.len;
        slots = std::max<std::size_t>(
            slots,
            static_cast<std::size_t>(*std::max_element(ids.ptr, end)) + 1);
    };
    include(launch.rs_slot_ids);
    include(launch.rs_buffer_slot_ids);
    if (executor_->graph_pad_slot >= 0) {
        slots = std::max<std::size_t>(
            slots,
            static_cast<std::size_t>(executor_->graph_pad_slot) + 1);
    }
    return slots;
}

std::vector<pie_cuda_driver::CudaAllocatorTarget>
Context::Impl::frame_targets(
    int kv_required_pages,
    std::size_t state_required,
    std::array<std::size_t, 4>* target_bytes) const {
    const std::size_t kv_capacity =
        static_cast<std::size_t>(std::max(1, kv_cache_->num_pages()));
    const std::size_t kv_required =
        static_cast<std::size_t>(std::max(0, kv_required_pages));
    const std::size_t state_capacity =
        executor_->rs_cache == nullptr
            ? 1
            : static_cast<std::size_t>(
                  std::max(1, executor_->rs_cache->max_slots()));
    std::vector<pie_cuda_driver::CudaAllocatorTarget> targets = {
        {kv_allocator_.get(), kv_required, kv_capacity},
        {state_allocator_.get(), state_required, state_capacity},
        {workspace_allocator_.get(), 1, 1},
        {attention_allocator_.get(), 1, 1},
    };
    if (target_bytes != nullptr) {
        *target_bytes = {
            kv_allocator_->target_bytes(kv_required, kv_capacity),
            state_allocator_->target_bytes(state_required, state_capacity),
            workspace_allocator_->allocated_bytes(),
            attention_allocator_->allocated_bytes(),
        };
    }
    return targets;
}

int Context::Impl::validate_finalized_launch(
    const pie_native::StepLaunch& launch,
    std::vector<pie_cuda_driver::pipeline::InstanceRecord>* instances,
    LaunchScratch* scratch,
    pie_native::LaunchView* view) const {
    if (executor_ == nullptr) return PIE_STATUS_CLOSED;
    if (launch.embed_rows.len > 0 &&
        (model_ == nullptr ||
         !model_->capabilities().supports_media_encode)) {
        return PIE_STATUS_UNSUPPORTED;
    }
    if (is_tp_follower()) return PIE_STATUS_UNSUPPORTED;
    const int resolve_status =
        registry_->resolve_instances(launch.instance_ids, instances);
    if (resolve_status != PIE_STATUS_OK) return resolve_status;
    std::vector<std::uint32_t> program_geometry_classes;
    program_geometry_classes.reserve(instances->size());
    for (const auto& record : *instances) {
        program_geometry_classes.push_back(record.geometry_class);
    }
    const int resource_status = pie_cuda_driver::abi::validate_launch_resources(
        launch,
        kv_cache_->num_pages(),
        kv_cache_->page_size(),
        executor_->rs_cache != nullptr ? executor_->rs_cache->max_slots() : 0,
        executor_->rs_cache != nullptr
            ? executor_->rs_cache->rs_buffer_num_slots()
            : 0,
        multimodal_limits_,
        program_geometry_classes.data(),
        program_geometry_classes.size());
    if (resource_status != PIE_STATUS_OK) return resource_status;
    if (launch.required_kv_pages >
        static_cast<std::uint32_t>(kv_cache_->num_pages())) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    *view = scratch->build(launch, *instances);
    if (view->ptir_program_hashes.empty()) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    std::string ptir_error;
    const int ptir_status = registry_->validate_launch(*view, &ptir_error);
    if (ptir_status != PIE_STATUS_OK && !ptir_error.empty()) {
        std::cerr << "[pie-driver-cuda] launch rejected: "
                  << ptir_error << "\n";
    }
    return ptir_status;
}

// Post one sealed frame (ABI v14). Admission is folded into the call: the
// whole frame expands and validates FIRST (an admitted frame is atomic —
// post-time rejection happens before anything reaches the stream), then one
// atomic commit against the frame-union demand, then the steps enqueue in
// order as one closed system. The tail step carries the frame completion:
// settle host callbacks are stream-ordered, so the tail's notify implies
// every step's terminals are latched.
int Context::Impl::launch(const PieFrameDesc& frame, PieCompletion completion) {
    pie_cuda_driver::ops::ScopedRuntimeQuantContext quant_scope(
        runtime_quant_context_);
    const PieStepDesc* steps = frame.steps.ptr;
    const std::size_t step_count = frame.steps.len;
    if (steps == nullptr || step_count == 0) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    std::vector<StepExpansion> expansions(step_count);
    std::vector<std::vector<pie_cuda_driver::pipeline::InstanceRecord>>
        instances(step_count);
    std::vector<LaunchScratch> scratch(step_count);
    std::vector<pie_native::LaunchView> views(step_count);
    int kv_required = 0;
    std::size_t state_required = 0;
    for (std::size_t i = 0; i < step_count; ++i) {
        expand_step(frame, steps[i], &expansions[i]);
        const int status = validate_finalized_launch(
            expansions[i].launch, &instances[i], &scratch[i], &views[i]);
        if (status != PIE_STATUS_OK) return status;
        kv_required =
            std::max(kv_required, required_kv_pages(expansions[i].launch));
        state_required = std::max(
            state_required, required_state_slots(expansions[i].launch));
    }
    // Folded admission: EXHAUSTED/IMPOSSIBLE return as statuses with no
    // side effects; the engine's lane retries EXHAUSTED in place.
    if (kv_required > kv_cache_->num_pages()) {
        std::cerr << "[pie-driver-cuda] frame admission impossible: "
                  << "kv_required=" << kv_required
                  << " num_pages=" << kv_cache_->num_pages() << "\n";
        return PIE_STATUS_IMPOSSIBLE;
    }
    std::array<std::size_t, 4> target_bytes{};
    const auto targets =
        frame_targets(kv_required, state_required, &target_bytes);
    const std::array<std::size_t, 4> committed_bytes = {
        kv_allocator_->committed_bytes(),
        state_allocator_->committed_bytes(),
        workspace_allocator_->committed_bytes(),
        attention_allocator_->committed_bytes(),
    };
    const bool needs_growth = std::equal(
        target_bytes.begin(), target_bytes.end(),
        committed_bytes.begin(),
        [](std::size_t target, std::size_t committed) {
            return target <= committed;
        }) == false;
    if (needs_growth) {
        recalibrate_elastic_budget(/*reset_hard_ceiling=*/false);
    }
    const auto commit = pie_cuda_driver::commit_cuda_arena_targets_atomically(
        elastic_pool_, targets);
    if (commit.outcome == pie_cuda_driver::CudaCommitOutcome::Exhausted) {
        return PIE_STATUS_EXHAUSTED;
    }
    if (commit.outcome == pie_cuda_driver::CudaCommitOutcome::Impossible) {
        std::cerr << "[pie-driver-cuda] frame commit impossible: "
                  << "kv_required=" << kv_required
                  << " required_pages=" << commit.required_pages
                  << " budget_pages=" << commit.budget_pages << "\n";
        return PIE_STATUS_IMPOSSIBLE;
    }
    executor_->required_kv_pages = kv_required;
    const char* assert_proportional =
        std::getenv("PIE_CUDA_ASSERT_KV_COMMIT_PROPORTIONAL");
    if (assert_proportional != nullptr && assert_proportional[0] != '\0' &&
        assert_proportional[0] != '0' && kv_required > 0) {
        const std::size_t committed = kv_allocator_->committed_bytes();
        const std::size_t capacity = kv_allocator_->allocated_bytes();
        kv_proportional_peak_required_pages_ = std::max(
            kv_proportional_peak_required_pages_,
            static_cast<std::size_t>(kv_required));
        kv_proportional_planned_pages_ =
            static_cast<std::size_t>(kv_cache_->num_pages());
        kv_proportional_peak_committed_bytes_ = std::max(
            kv_proportional_peak_committed_bytes_, committed);
        kv_proportional_capacity_bytes_ = capacity;
        if (kv_required * 2 < kv_cache_->num_pages() &&
            committed * 2 >= capacity) {
            std::cerr << "[pie-driver-cuda] short-context KV commit is not "
                         "demand-proportional\n";
            return PIE_STATUS_DRIVER_ERROR;
        }
    }
    // Stream work is SUCCESS-only for admitted frames (P4): any exception
    // past this point is a driver fault. Latch FAILED on the affected
    // steps' fires (every step for a prepare fault — nothing was enqueued;
    // this and later steps for an enqueue fault — earlier steps are live
    // on the stream and settle normally), resolve the frame completion,
    // and report the frame as settled-synchronously.
    std::vector<pie_cuda_driver::PreparedStep> prepared(step_count);
    const auto fail_frame = [&](std::size_t failed_step,
                                const char* phase,
                                const std::exception& e,
                                std::size_t abort_from,
                                std::size_t fail_from) -> int {
        std::cerr << "[pie-driver-cuda] frame step " << failed_step
                  << " " << phase << ": " << e.what() << "\n";
        for (std::size_t j = abort_from; j < step_count; ++j) {
            pie_cuda_driver::abort_step(*executor_, prepared[j]);
        }
        std::vector<std::pair<std::uint64_t, std::uint64_t>>
            channel_notifications;
        try {
            channel_notifications =
                registry_->dispatch().settle_failed_launch(
                    views[failed_step], executor_->cublas.stream());
        } catch (const std::exception& settle_error) {
            std::cerr << "[pie-driver-cuda] frame failure settlement: "
                      << settle_error.what() << "\n";
        }
        for (std::size_t j = fail_from; j < step_count; ++j) {
            for (std::size_t cell = 0;
                 cell < steps[j].terminal_cells.len;
                 ++cell) {
                publish_terminal(
                    steps[j].terminal_cells.ptr[cell],
                    PIE_TERMINAL_OUTCOME_FAILED);
            }
        }
        if (runtime_.notify != nullptr) {
            for (const auto& [wait_id, epoch] : channel_notifications) {
                if (wait_id != 0 && epoch != 0) {
                    runtime_.notify(runtime_.ctx, wait_id, epoch);
                }
            }
        }
        if (runtime_.notify != nullptr && completion.wait_id != 0) {
            runtime_.notify(
                runtime_.ctx, completion.wait_id, completion.target_epoch);
        }
        return PIE_STATUS_OK;
    };
    // FramePrepare: every step's host work runs at frame entry, before
    // anything of this frame reaches the stream (the two-track model —
    // T_prepare amortizes once per frame). Step order matters: wave
    // admission applies channel sequence tickets in wave order, and each
    // wave freezes its channel-cursor window into its tickets.
    for (std::size_t i = 0; i < step_count; ++i) {
        try {
            pie_cuda_driver::prepare_step(*executor_, views[i], prepared[i]);
        } catch (const std::exception& e) {
            return fail_frame(i, "prepare", e, 0, 0);
        }
    }
    // StepEnqueue + FrameSettle, in step order: each step's settlement
    // rides the stream before the next step's pull-validate, which is what
    // lets step i+1 consume step i's device-published channel state. The
    // tail step carries the frame completion.
    for (std::size_t i = 0; i < step_count; ++i) {
        const bool tail = i + 1 == step_count;
        const PieCompletion step_completion =
            tail ? completion : PieCompletion{0, 0, nullptr};
        try {
            pie_cuda_driver::enqueue_step(*executor_, prepared[i]);
            pie_cuda_driver::settle_step(
                *executor_, runtime_, step_completion, prepared[i]);
        } catch (const std::exception& e) {
            return fail_frame(i, "launch", e, i, i);
        }
    }
    return PIE_STATUS_OK;
}

int Context::Impl::register_program(const PieProgramDesc& program, std::uint64_t* program_id) {
    if (registry_ == nullptr) return PIE_STATUS_CLOSED;
    if (is_tp_follower()) return PIE_STATUS_UNSUPPORTED;
    std::string err;
    const int rc = registry_->register_program(program, program_id, &err);
    if (rc != PIE_STATUS_OK && !err.empty()) {
        std::cerr << "[pie-driver-cuda] register_program: " << err << "\n";
    }
    return rc;
}

int Context::Impl::register_channel(
    const PieChannelDesc& channel,
    PieChannelEndpointBinding* binding) {
    if (registry_ == nullptr) return PIE_STATUS_CLOSED;
    if (is_tp_follower()) return PIE_STATUS_UNSUPPORTED;
    std::string err;
    const int rc = registry_->register_channel(channel, binding, &err);
    if (rc != PIE_STATUS_OK && !err.empty()) {
        std::cerr << "[pie-driver-cuda] register_channel: " << err << "\n";
    }
    return rc;
}

int Context::Impl::bind_instance(const PieInstanceDesc& instance, PieInstanceBinding* binding) {
    if (registry_ == nullptr) return PIE_STATUS_CLOSED;
    if (is_tp_follower()) return PIE_STATUS_UNSUPPORTED;
    std::string err;
    const int rc = registry_->bind_instance(instance, binding, &err);
    if (rc != PIE_STATUS_OK && !err.empty()) {
        std::cerr << "[pie-driver-cuda] bind_instance: " << err << "\n";
    }
    return rc;
}

int Context::Impl::encode(const PieEncodeDesc& encode, PieCompletion completion) {
    pie_cuda_driver::ops::ScopedRuntimeQuantContext quant_scope(
        runtime_quant_context_);
#ifdef PIE_CUDA_QWEN_ONLY
    (void)encode;
    (void)completion;
    return PIE_STATUS_UNSUPPORTED;
#else
    if (model_ == nullptr && encode_vision_ == nullptr &&
        encode_audio_ == nullptr) {
        return PIE_STATUS_CLOSED;
    }
    if (tp_size_ > 1) return PIE_STATUS_UNSUPPORTED;
    if (encode_vision_ == nullptr && encode_audio_ == nullptr &&
        !model_->capabilities().supports_media_encode) {
        return PIE_STATUS_UNSUPPORTED;
    }
    const int resource_status = pie_cuda_driver::abi::validate_encode_resources(
        encode, multimodal_limits_, media_hidden_size_);
    if (resource_status != PIE_STATUS_OK) return resource_status;
    collect_ready_async_resources();
    try {
        pie_cuda_driver::model::MediaEncodeInputs in;
        in.image_pixels_h =
            reinterpret_cast<const float*>(encode.image_pixels.ptr);
        in.image_pixel_byte_indptr_h = encode.image_pixel_indptr.ptr;
        in.image_patch_positions_h = encode.image_patch_positions.ptr;
        in.image_anchor_rows_h = encode.image_anchor_rows.ptr;
        in.num_images = static_cast<int>(encode.image_anchor_rows.len);
        in.audio_features_h =
            reinterpret_cast<const float*>(encode.audio_features.ptr);
        in.audio_feature_byte_indptr_h =
            encode.audio_feature_indptr.ptr;
        in.audio_anchor_rows_h = encode.audio_anchor_rows.ptr;
        in.num_clips = static_cast<int>(encode.audio_anchor_rows.len);
        in.output_rows_h =
            reinterpret_cast<std::uint16_t*>(encode.output_rows.ptr);
        in.output_bytes = encode.output_rows.len;
        in.output_row_indptr_h = encode.output_row_indptr.ptr;
        cudaStream_t stream = media_stream_ != nullptr
            ? media_stream_
            : executor_->cublas.stream();
        if (encode_vision_ != nullptr || encode_audio_ != nullptr) {
            std::size_t row_offset = 0;
            in.output_row_indptr_h[0] = 0;
            if (in.num_images > 0) {
                if (encode_vision_ == nullptr) return PIE_STATUS_UNSUPPORTED;
                pie_cuda_driver::model::Gemma4VisionInputs vision;
                vision.weights = encode_vision_;
                vision.pixels_h = in.image_pixels_h;
                vision.pixel_byte_indptr_h =
                    in.image_pixel_byte_indptr_h;
                vision.patch_positions_h = in.image_patch_positions_h;
                vision.anchor_rows_h = in.image_anchor_rows_h;
                vision.num_images = in.num_images;
                std::vector<std::uint32_t> boundaries(
                    static_cast<std::size_t>(in.num_images) + 1);
                pie_cuda_driver::model::encode_gemma4_vision(
                    vision, in.output_rows_h, in.output_bytes,
                    boundaries.data(), stream);
                row_offset = boundaries.back();
                for (int image = 0; image < in.num_images; ++image) {
                    in.output_row_indptr_h[image + 1] =
                        boundaries[image + 1];
                }
            }
            if (in.num_clips > 0) {
                if (encode_audio_ == nullptr) return PIE_STATUS_UNSUPPORTED;
                pie_cuda_driver::model::Gemma4AudioInputs audio;
                audio.weights = encode_audio_;
                audio.features_h = in.audio_features_h;
                audio.feature_byte_indptr_h =
                    in.audio_feature_byte_indptr_h;
                audio.anchor_rows_h = in.audio_anchor_rows_h;
                audio.n_mel = encode_audio_->n_mel;
                audio.num_clips = in.num_clips;
                const std::size_t consumed =
                    row_offset * media_hidden_size_ *
                    sizeof(std::uint16_t);
                std::vector<std::uint32_t> boundaries(
                    static_cast<std::size_t>(in.num_clips) + 1);
                pie_cuda_driver::model::encode_gemma4_audio(
                    audio,
                    in.output_rows_h +
                        row_offset * media_hidden_size_,
                    in.output_bytes - consumed, boundaries.data(),
                    stream);
                for (int clip = 0; clip < in.num_clips; ++clip) {
                    in.output_row_indptr_h[
                        in.num_images + clip + 1] =
                        static_cast<std::uint32_t>(row_offset) +
                        boundaries[clip + 1];
                }
            }
        } else if (!model_->encode_media(in, stream)) {
            return PIE_STATUS_UNSUPPORTED;
        }
        enqueue_completion(stream, completion);
        return PIE_STATUS_OK;
    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-cuda] encode: " << e.what() << "\n";
        return PIE_STATUS_DRIVER_ERROR;
    }
#endif
}

int Context::Impl::copy_kv(const PieKvCopyDesc& copy, PieCompletion completion) {
    if (executor_ == nullptr) return PIE_STATUS_CLOSED;
    // Typed control operations need an explicit TP control broadcast and
    // all-rank completion barrier. Never report rank-0-only work as complete.
    if (tp_size_ > 1) return PIE_STATUS_UNSUPPORTED;
    const int resource_status = pie_cuda_driver::abi::validate_kv_copy_resources(
        copy,
        static_cast<std::uint32_t>(device_ordinal_),
        kv_cache_->num_pages(),
        swap_pool_->num_pages(),
        kv_cache_->page_size(),
        kv_cache_->format().is_native_bf16());
    if (resource_status != PIE_STATUS_OK) return resource_status;
    collect_ready_async_resources();

    try {
        std::uint32_t highest_device_page = 0;
        bool needs_device_pages = false;
        auto include_pages = [&](const std::uint32_t* pages, std::size_t count) {
            if (count == 0) return;
            needs_device_pages = true;
            highest_device_page = std::max(
                highest_device_page,
                *std::max_element(pages, pages + count));
        };
        if (copy.src_domain == PIE_MEMORY_DOMAIN_CUDA_DEVICE) {
            include_pages(copy.src_page_ids.ptr, copy.src_page_ids.len);
        }
        if (copy.dst_domain == PIE_MEMORY_DOMAIN_CUDA_DEVICE) {
            include_pages(copy.dst_page_ids.ptr, copy.dst_page_ids.len);
        }
        if (copy.cells.len > 0) {
            needs_device_pages = true;
            for (std::size_t i = 0; i < copy.cells.len; ++i) {
                highest_device_page = std::max(
                    highest_device_page,
                    std::max(
                        copy.cells.ptr[i].src_page_id,
                        copy.cells.ptr[i].dst_page_id));
            }
        }
        if (needs_device_pages) {
            kv_cache_->ensure_pages(
                static_cast<int>(highest_device_page) + 1);
        }
        const bool has_page_copies =
            copy.src_page_ids.len > 0 || copy.dst_page_ids.len > 0;
        cudaStream_t completion_stream = executor_->cublas.stream();
        std::vector<OwnedValue> keepalive;
        if (copy.src_page_ids.len > 0 || copy.dst_page_ids.len > 0) {
            const auto src = std::span<const std::uint32_t>(copy.src_page_ids.ptr, copy.src_page_ids.len);
            const auto dst = std::span<const std::uint32_t>(copy.dst_page_ids.ptr, copy.dst_page_ids.len);
            if (copy.src_domain == PIE_MEMORY_DOMAIN_CUDA_DEVICE &&
                copy.dst_domain == PIE_MEMORY_DOMAIN_HOST_PINNED) {
                swap_pool_->copy_d2h_async(*kv_cache_, src, dst);
            } else if (copy.src_domain == PIE_MEMORY_DOMAIN_HOST_PINNED &&
                       copy.dst_domain == PIE_MEMORY_DOMAIN_CUDA_DEVICE) {
                swap_pool_->copy_h2d_async(*kv_cache_, src, dst);
            } else if (copy.src_domain == PIE_MEMORY_DOMAIN_CUDA_DEVICE &&
                       copy.dst_domain == PIE_MEMORY_DOMAIN_CUDA_DEVICE) {
                swap_pool_->copy_d2d_async(*kv_cache_, src, dst);
            } else if (copy.src_domain == PIE_MEMORY_DOMAIN_HOST_PINNED &&
                       copy.dst_domain == PIE_MEMORY_DOMAIN_HOST_PINNED) {
                swap_pool_->copy_h2h_async(src, dst);
            }
        }
        if (copy.cells.len > 0) {
            std::vector<std::uint32_t> dst_pages(copy.cells.len);
            std::vector<std::uint32_t> dst_offs(copy.cells.len);
            std::vector<std::uint32_t> src_pages(copy.cells.len);
            std::vector<std::uint32_t> src_offs(copy.cells.len);
            for (std::size_t i = 0; i < copy.cells.len; ++i) {
                dst_pages[i] = copy.cells.ptr[i].dst_page_id;
                dst_offs[i] = copy.cells.ptr[i].dst_token_offset;
                src_pages[i] = copy.cells.ptr[i].src_page_id;
                src_offs[i] = copy.cells.ptr[i].src_token_offset;
            }
            auto d_dst_page = pie_cuda_driver::DeviceBuffer<std::uint32_t>::from_host(dst_pages);
            auto d_dst_off = pie_cuda_driver::DeviceBuffer<std::uint32_t>::from_host(dst_offs);
            auto d_src_page = pie_cuda_driver::DeviceBuffer<std::uint32_t>::from_host(src_pages);
            auto d_src_off = pie_cuda_driver::DeviceBuffer<std::uint32_t>::from_host(src_offs);
            cudaStream_t stream = completion_stream;
            for (int l = 0; l < kv_cache_->num_layers(); ++l) {
                pie_cuda_driver::kernels::launch_copy_kv_cells_bf16(
                    kv_cache_->layer_view(l),
                    d_dst_page.data(), d_dst_off.data(),
                    d_src_page.data(), d_src_off.data(),
                    static_cast<int>(copy.cells.len), stream);
            }
            keep_async_value(keepalive, std::move(d_dst_page));
            keep_async_value(keepalive, std::move(d_dst_off));
            keep_async_value(keepalive, std::move(d_src_page));
            keep_async_value(keepalive, std::move(d_src_off));
        }
        if (has_page_copies && (!keepalive.empty() || completion.wait_id != 0)) {
            cudaEvent_t swap_done = nullptr;
            CUDA_CHECK(cudaEventCreateWithFlags(&swap_done, cudaEventDisableTiming));
            CUDA_CHECK(cudaEventRecord(swap_done, swap_pool_->stream()));
            CUDA_CHECK(cudaStreamWaitEvent(completion_stream, swap_done, 0));
            CUDA_CHECK(cudaEventDestroy(swap_done));
        }
        retain_async_resources(completion_stream, std::move(keepalive));
        enqueue_completion(completion_stream, completion);
        return PIE_STATUS_OK;
    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-cuda] copy_kv: " << e.what() << "\n";
        return PIE_STATUS_DRIVER_ERROR;
    }
}

int Context::Impl::copy_state(const PieStateCopyDesc& copy, PieCompletion completion) {
    if (executor_ == nullptr) return PIE_STATUS_CLOSED;
    if (tp_size_ > 1) return PIE_STATUS_UNSUPPORTED;
    const int resource_status = pie_cuda_driver::abi::validate_state_copy_resources(
        copy,
        executor_->rs_cache != nullptr ? executor_->rs_cache->max_slots() : 0);
    if (resource_status != PIE_STATUS_OK) return resource_status;
    collect_ready_async_resources();
    try {
        state_allocator_->ensure_all();
        for (std::size_t i = 0; i < copy.slot_ranges.len; ++i) {
            const PieStateCopyRange& range = copy.slot_ranges.ptr[i];
            executor_->rs_cache->copy_slot_d2d(
                static_cast<int>(range.src_slot_id),
                static_cast<int>(range.dst_slot_id),
                executor_->cublas.stream());
        }
        enqueue_completion(executor_->cublas.stream(), completion);
        return PIE_STATUS_OK;
    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-cuda] copy_state: " << e.what() << "\n";
        return PIE_STATUS_DRIVER_ERROR;
    }
}

int Context::Impl::resize_pool(const PiePoolResizeDesc& resize, PieCompletion completion) {
    if (executor_ == nullptr) return PIE_STATUS_CLOSED;
    if (tp_size_ > 1) return PIE_STATUS_UNSUPPORTED;
    collect_ready_async_resources();
    if (resize.pool_id > PIE_ELASTIC_POOL_WORKSPACE) {
        return PIE_STATUS_UNSUPPORTED;
    }
    try {
        cudaStream_t stream = executor_->cublas.stream();
        if (stream != nullptr) {
            const cudaError_t stream_status = cudaStreamQuery(stream);
            if (stream_status == cudaErrorNotReady) {
                return PIE_STATUS_UNSUPPORTED;
            }
            CUDA_CHECK(stream_status);
        }
        if (swap_pool_ != nullptr && swap_pool_->stream() != nullptr) {
            const cudaError_t stream_status =
                cudaStreamQuery(swap_pool_->stream());
            if (stream_status == cudaErrorNotReady) {
                return PIE_STATUS_UNSUPPORTED;
            }
            CUDA_CHECK(stream_status);
        }
        // The quiescence gate above IS the horizon-empty condition (Venus
        // D6): with the stream drained no frame is in flight, so no
        // admission floor constrains the trim.
        if (resize.pool_id == PIE_ELASTIC_POOL_KV) {
            if (resize.target_pages >
                static_cast<std::uint64_t>(kv_cache_->num_pages())) {
                return PIE_STATUS_INVALID_ARGUMENT;
            }
            const int pages =
                std::max<int>(1, static_cast<int>(resize.target_pages));
            kv_cache_->ensure_pages(pages);
            kv_cache_->trim_pages(pages);
        } else {
            std::size_t bytes = static_cast<std::size_t>(
                std::min<std::uint64_t>(
                    resize.target_pages,
                    std::numeric_limits<std::size_t>::max() /
                        pie::elastic::kLogicalPageBytes)) *
                pie::elastic::kLogicalPageBytes;
            auto& allocator =
                resize.pool_id == PIE_ELASTIC_POOL_STATE
                    ? state_allocator_
                    : workspace_allocator_;
            std::vector<pie_cuda_driver::CudaAllocatorTarget> targets = {
                {allocator.get(), bytes, allocator->allocated_bytes()},
            };
            if (resize.pool_id == PIE_ELASTIC_POOL_WORKSPACE) {
                const std::size_t attention_bytes =
                    std::min(bytes, attention_allocator_->allocated_bytes());
                targets.push_back({
                    attention_allocator_.get(),
                    attention_bytes,
                    attention_allocator_->allocated_bytes(),
                });
            }
            const auto commit =
                pie_cuda_driver::commit_cuda_arena_targets_atomically(
                    elastic_pool_, targets);
            if (commit.outcome !=
                pie_cuda_driver::CudaCommitOutcome::Committed) {
                return PIE_STATUS_DRIVER_ERROR;
            }
            allocator->trim_bytes(bytes);
            if (resize.pool_id == PIE_ELASTIC_POOL_WORKSPACE) {
                const std::size_t attention_bytes =
                    std::min(bytes, attention_allocator_->allocated_bytes());
                attention_allocator_->trim_bytes(attention_bytes);
            }
        }
        enqueue_completion(executor_->cublas.stream(), completion);
        return PIE_STATUS_OK;
    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-cuda] resize_pool: " << e.what() << "\n";
        return PIE_STATUS_DRIVER_ERROR;
    }
}

int Context::Impl::close_instance(std::uint64_t instance_id) {
    if (executor_ == nullptr) return PIE_STATUS_CLOSED;
    if (is_tp_follower()) return PIE_STATUS_UNSUPPORTED;
    collect_ready_async_resources();
    return registry_->close_instance(instance_id);
}

int Context::Impl::close_channel(std::uint64_t channel_id) {
    if (executor_ == nullptr) return PIE_STATUS_CLOSED;
    if (is_tp_follower()) return PIE_STATUS_UNSUPPORTED;
    collect_ready_async_resources();
    std::string err;
    const int rc = registry_->close_channel(channel_id, &err);
    if (rc != PIE_STATUS_OK && !err.empty()) {
        std::cerr << "[pie-driver-cuda] close_channel: " << err << "\n";
    }
    return rc;
}

Context::Context() : impl_(std::make_unique<Impl>()) {}
Context::~Context() = default;

int Context::initialize(const std::string& config_path, const PieRuntimeCallbacks& runtime) {
    return impl_->initialize(config_path, runtime);
}

void Context::fill_device_facts(PieDriverCaps* caps) const {
    impl_->fill_device_facts(caps);
}

int Context::load_model(const PieModelLoadDesc& load, PieDriverCaps* caps) {
    return impl_->load_model(load, caps);
}

int Context::register_program(const PieProgramDesc& program, std::uint64_t* program_id) {
    return impl_->register_program(program, program_id);
}

int Context::register_channel(
    const PieChannelDesc& channel, PieChannelEndpointBinding* binding) {
    return impl_->register_channel(channel, binding);
}

int Context::bind_instance(const PieInstanceDesc& instance, PieInstanceBinding* binding) {
    return impl_->bind_instance(instance, binding);
}

int Context::launch(const PieFrameDesc& frame, PieCompletion completion) {
    return impl_->launch(frame, completion);
}

int Context::encode(const PieEncodeDesc& encode, PieCompletion completion) {
    return impl_->encode(encode, completion);
}

int Context::copy_kv(const PieKvCopyDesc& copy, PieCompletion completion) {
    return impl_->copy_kv(copy, completion);
}

int Context::copy_state(const PieStateCopyDesc& copy, PieCompletion completion) {
    return impl_->copy_state(copy, completion);
}

int Context::resize_pool(const PiePoolResizeDesc& resize, PieCompletion completion) {
    return impl_->resize_pool(resize, completion);
}

int Context::close_instance(std::uint64_t instance_id) {
    return impl_->close_instance(instance_id);
}

int Context::close_channel(std::uint64_t channel_id) {
    return impl_->close_channel(channel_id);
}

}  // namespace pie::cuda
