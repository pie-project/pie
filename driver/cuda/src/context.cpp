#include "context.hpp"

#include <algorithm>
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
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <nlohmann/json.hpp>

#include "batch/workspace.hpp"
#include "config.hpp"
#include "distributed.hpp"
#include "entry_validation.hpp"
#include "cuda_check.hpp"
#include "store/memory_planner.hpp"
#include "device_buffer.hpp"
#include "batch/compose.hpp"
#include "batch/forward.hpp"
#include "batch/tp.hpp"
#include "kernels/kv_paged.hpp"
#include "store/kv_cache.hpp"
#include "store/mla_cache.hpp"
#include "store/dsa_cache.hpp"
#include "model/deepseek_v4/deepseek_v4_forward.hpp"
#include "model/gemma/gemma2.hpp"
#include "model/gemma4/gemma4.hpp"
#include "model/glm5/glm5_forward.hpp"
#include "model/kimi/kimi_forward.hpp"
#include "model/llama_like/llama_like.hpp"
#include "model/loaded_model.hpp"
#include "model/nemotron_h/nemotron_h.hpp"
#include "model/nemotron_h/nemotron_h_forward.hpp"
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
        const PieLaunchDesc& launch,
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
        view.logical_fire_ids = pie_native::slice_from_u64(launch.logical_fire_ids.ptr, launch.logical_fire_ids.len);
        view.retry_eligible = pie_native::slice_from_u8(launch.retry_eligible.ptr, launch.retry_eligible.len);
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
        return view;
    }
};

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
    return cfg.model.mtp_num_drafts;
}

}  // namespace

class Context::Impl {
  public:
    Impl() = default;
    ~Impl() {
        drain_async_streams();
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
    void fill_caps(PieDriverCaps* caps) const {
        if (caps == nullptr) return;
        caps->json_bytes = reinterpret_cast<const std::uint8_t*>(caps_json_.data());
        caps->json_len = caps_json_.size();
    }

    int register_program(const PieProgramDesc& program, std::uint64_t* program_id);
    int register_channel(const PieChannelDesc& channel,
                         PieChannelEndpointBinding* binding);
    int bind_instance(const PieInstanceDesc& instance, PieInstanceBinding* binding);
    int launch(const PieLaunchDesc& launch, PieCompletion completion);
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
    }

    bool is_tp_follower() const noexcept {
        return tp_size_ > 1 && tp_rank_ > 0;
    }

    PieRuntimeCallbacks runtime_{};
    std::vector<OwnedValue> owners_;
    pie_cuda_driver::BatchEngine* executor_ = nullptr;
    pie_cuda_driver::pipeline::Registry* registry_ = nullptr;
    pie_cuda_driver::KvCache* kv_cache_ = nullptr;
    pie_cuda_driver::SwapPool* swap_pool_ = nullptr;
    pie_cuda_driver::NcclComm* tp_comm_ = nullptr;
    std::string caps_json_;
    std::string tp_cpu_gate_key_;
    int device_ordinal_ = 0;
    int tp_size_ = 1;
    int tp_rank_ = 0;
    pie_cuda_driver::abi::MultimodalLimits multimodal_limits_;
    std::atomic<bool> tp_follower_stop_{false};
    std::thread tp_follower_thread_;
    std::vector<std::unique_ptr<PendingAsyncResources>> pending_async_resources_;
};

int Context::Impl::initialize(
    const std::string& config_path, const PieRuntimeCallbacks& runtime) {
    using namespace pie_cuda_driver;
    runtime_ = runtime;

    auto* cfg_p = own_value(load_config(config_path));
    Config& cfg = *cfg_p;
    const bool verbose = cfg.runtime.verbose;
    tp_size_ = std::max(1, cfg.distributed.tp_size);
    tp_rank_ = cfg.distributed.tp_rank;
    tp_cpu_gate_key_ = cfg.distributed.nccl_unique_id_hex;

    device_ordinal_ = parse_cuda_device_id(cfg.model.device);
    CUDA_CHECK(cudaSetDevice(device_ordinal_));

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

    auto* engine_p = own_value(LoadedModel::load(cfg, tp_comm_ptr));
    auto& engine = *engine_p;

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

    const std::vector<bool>& nemotron_h_layer_is_mamba = plan_info.layer_is_mamba;
    const int nemotron_h_mamba_layers = static_cast<int>(std::count(
        nemotron_h_layer_is_mamba.begin(), nemotron_h_layer_is_mamba.end(), true));
    const int nemotron_h_attention_layer_count = family == model::Family::NemotronH
        ? model::nemotron_h_attention_layers(engine.hf_config())
        : 0;

    const auto kv_format = kv_cache_format_from_string(
        cfg.batching.kv_cache_dtype, cfg.model.dtype);
    const bool use_cuda_graphs = true;
    const bool graph_capable_forward =
        use_cuda_graphs && family == model::Family::LlamaLike &&
        kv_format.is_native_bf16();
    const auto runtime_quant_scratch_base =
        graph_capable_forward
            ? runtime_quant_scratch_spec(engine, /*max_tokens=*/0)
            : ops::RuntimeQuantScratchSpec{};

    const CudaMemoryPlan mem_plan = plan_cuda_memory(
        cfg, engine.hf_config(), max_mlp_intermediate, max_Hq, max_Hk,
        family == model::Family::Gemma4, plan_info.per_layer_head_dim,
        plan_info.kv_source_layer, family == model::Family::Qwen3_5,
        family == model::Family::Qwen3_5Moe, qwen3_5_linear_layers,
        family == model::Family::NemotronH, nemotron_h_mamba_layers,
        kv_format, runtime_quant_scratch_base, verbose);
    const int max_workspace_tokens = mem_plan.max_workspace_tokens;
    const long kv_page_cap = [&]() -> long {
        if (const char* e = std::getenv("PIE_KV_PAGE_CAP")) {
            const long v = std::atol(e);
            if (v > 0) return v;
        }
        return static_cast<long>(cfg.batching.total_pages);
    }();
    const int runtime_kv_pages = (kv_page_cap > 0)
        ? std::min<int>(mem_plan.kv_pages, static_cast<int>(kv_page_cap))
        : mem_plan.kv_pages;
    const int physical_kv_pages =
        mem_plan.kv_pages > 0 ? mem_plan.kv_pages + 1 : mem_plan.kv_pages;
    const int graph_pad_page = mem_plan.kv_pages > 0 ? runtime_kv_pages : -1;
    const bool has_recurrent_state_cache =
        ((family == model::Family::Qwen3_5 || family == model::Family::Qwen3_5Moe) &&
         qwen3_5_linear_layers > 0) ||
        (family == model::Family::NemotronH && nemotron_h_mamba_layers > 0);
    const int runtime_state_slots = mem_plan.state_slots;
    const int graph_pad_slot =
        has_recurrent_state_cache && runtime_state_slots > 0 && graph_pad_page >= 0
            ? runtime_state_slots
            : -1;
    const int allocated_state_slots =
        runtime_state_slots + (graph_pad_slot >= 0 ? 1 : 0);

    auto* ws_p = own_value(model::Workspace::allocate_full(
        engine.hf_config(), max_workspace_tokens,
        max_mlp_intermediate, max_Hq, max_Hk,
        mem_plan.capacity.max_logit_rows));
    auto& ws = *ws_p;

    // KV-cache shape genuinely differs per family (MLA-backed families use
    // a 1x1 placeholder KvCache; Gemma-4 has per-layer head_dim + KV
    // sharing; Nemotron-H only allocates pages for its attention layers).
    // This is exactly the kind of family-specific store allocation
    // `Context` is allowed to switch on (cpp-refact.md Phase 5) — the
    // registry alone decided `family` above.
    auto* kv_cache_p = own_value([&]() -> KvCache {
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
    auto& kv_cache = *kv_cache_p;

    auto* mla_cache_p = own_value(
        (family == model::Family::Kimi || family == model::Family::Glm5)
            ? MlaCache::allocate(
                  engine.hf_config().num_hidden_layers,
                  physical_kv_pages,
                  mem_plan.kv_page_size,
                  engine.hf_config().kv_lora_rank,
                  engine.hf_config().qk_rope_head_dim,
                  DType::BF16)
            : MlaCache{});
    auto& mla_cache = *mla_cache_p;

    auto* dsa_cache_p = own_value(
        family == model::Family::Glm5
            ? DsaCache::allocate(
                  engine.hf_config().num_hidden_layers,
                  engine.hf_config().max_position_embeddings,
                  engine.hf_config().index_head_dim)
            : DsaCache{});
    auto& dsa_cache = *dsa_cache_p;

    auto* attn_ws_p = own_value(AttentionWorkspace::allocate(
        mem_plan.attn_float_workspace_bytes, 8ull * 1024 * 1024));
    auto& attn_ws = *attn_ws_p;

    auto* qwen3_5_plan_state_p = own_emplace<model::Qwen3_5PlanState>();
    auto& qwen3_5_plan_state = *qwen3_5_plan_state_p;
    auto* qwen3_5_la_ws_p = own_emplace<model::Qwen3_5LinearAttnWorkspace>();
    auto& qwen3_5_la_ws = *qwen3_5_la_ws_p;
    auto* qwen3_5_state_cache_p = own_emplace<RecurrentStateCache>();
    auto& qwen3_5_state_cache = *qwen3_5_state_cache_p;
    auto* qwen3_5_moe_ws_p = own_emplace<model::Qwen3_5MoeMlpWorkspace>();
    auto& qwen3_5_moe_ws = *qwen3_5_moe_ws_p;
    auto* nemotron_h_ws_p = own_emplace<model::NemotronHWorkspace>();
    auto& nemotron_h_ws = *nemotron_h_ws_p;
    auto* nemotron_h_state_cache_p = own_emplace<RecurrentStateCache>();
    auto& nemotron_h_state_cache = *nemotron_h_state_cache_p;
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
        qwen3_5_la_ws = model::Qwen3_5LinearAttnWorkspace::allocate(
            max_workspace_tokens, conv_dim,
            local_linear_value_heads,
            local_linear_key_heads,
            cfg_q.linear_key_head_dim,
            cfg_q.linear_value_head_dim,
            (cfg_q.num_attention_heads / local_tp_size) * cfg_q.head_dim);
        qwen3_5_runtime_rs_slots = std::max<int>(1, mem_plan.state_slots);
        qwen3_5_state_cache = RecurrentStateCache::allocate(
            qwen3_5_layer_is_linear, conv_dim, cfg_q.linear_conv_kernel_dim,
            local_linear_value_heads,
            cfg_q.linear_key_head_dim,
            cfg_q.linear_value_head_dim,
            cfg_q.hidden_size,
            qwen3_5_runtime_rs_slots);
        const int stash_width = conv_dim + 2 * local_linear_value_heads;
        qwen3_5_state_cache.configure_verify_hidden_stash(
            max_workspace_tokens, stash_width);
        qwen3_5_state_cache.configure_rs_buffer_pool(
            mem_plan.kv_page_size, stash_width, qwen3_5_runtime_rs_slots);
        if (family == model::Family::Qwen3_5Moe) {
            qwen3_5_moe_ws = model::Qwen3_5MoeMlpWorkspace::allocate(
                max_workspace_tokens,
                cfg_q.hidden_size,
                cfg_q.num_experts,
                cfg_q.num_experts_per_tok,
                cfg_q.moe_intermediate_size / local_tp_size,
                cfg_q.shared_expert_intermediate_size / local_tp_size);
        }
    } else if (family == model::Family::NemotronH) {
        const auto& cfg_n = engine.hf_config();
        nemotron_h_ws = model::NemotronHWorkspace::allocate(
            cfg_n, max_workspace_tokens, local_tp_size);
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
        nemotron_h_state_cache = RecurrentStateCache::allocate_bf16_recurrent(
            nemotron_h_layer_is_mamba,
            conv_dim,
            cfg_n.mamba_conv_kernel,
            local_mamba_heads,
            cfg_n.mamba_head_dim,
            cfg_n.mamba_state_size,
            std::max<int>(1, allocated_state_slots));
    }

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
        ops::reserve_runtime_quant_scratch(runtime_quant_scratch, true);
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    auto* persistent_inputs_p = own_value(PersistentInputs::allocate(
        max_workspace_tokens,
        mem_plan.max_requests,
        mem_plan.max_page_refs,
        mem_plan.capacity.max_custom_mask_bytes));
    auto& persistent_inputs = *persistent_inputs_p;

    model::LlamaLikeForwardCfg fwd_cfg{};
    model::Gemma2ForwardCfg gemma_fwd_cfg{};
    model::Gemma4ForwardCfg gemma4_fwd_cfg{};
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
        gemma_fwd_cfg.query_pre_attn_scalar = hf.gemma_query_pre_attn_scalar;
        gemma_fwd_cfg.final_logit_softcap = hf.gemma_final_logit_softcap;
        gemma_fwd_cfg.attn_logit_softcap = hf.gemma_attn_logit_softcap;
        gemma_fwd_cfg.use_qk_norm = (mt == "gemma3" || mt == "gemma3_text");
        gemma_fwd_cfg.force_prefill_path = !gqa_in_decode_set;
        gemma_fwd_cfg.tp_size = local_tp_size;
        gemma_fwd_cfg.tp_comm = tp_comm_ptr;
        const bool homogeneous = !has_non_full_attention_layers(hf);
        if (!homogeneous) {
            gemma_fwd_cfg.per_layer_window_left.reserve(hf.layer_types.size());
            gemma_fwd_cfg.per_layer_rope_theta.reserve(hf.layer_types.size());
            fwd_cfg.per_layer_window_left.reserve(hf.layer_types.size());
            for (const auto& t : hf.layer_types) {
                const bool is_sliding = (t == "sliding_attention");
                const int window = is_sliding ? hf.sliding_window : -1;
                gemma_fwd_cfg.per_layer_window_left.push_back(window);
                fwd_cfg.per_layer_window_left.push_back(window);
                const float theta =
                    (is_sliding && hf.rope_local_base_freq > 0.f)
                        ? hf.rope_local_base_freq
                        : hf.rope_theta;
                gemma_fwd_cfg.per_layer_rope_theta.push_back(theta);
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
    if (family == model::Family::Gemma4) {
        const auto& hf = engine.hf_config();
        gemma4_fwd_cfg.final_logit_softcap = hf.gemma_final_logit_softcap;
        const int gqa = hf.num_attention_heads / hf.num_key_value_heads;
        gemma4_fwd_cfg.force_prefill_path = !flashinfer_decode_supports_gqa(gqa);
        gemma4_fwd_cfg.tp_size = local_tp_size;
        gemma4_fwd_cfg.tp_comm = tp_comm_ptr;
    }

    ForwardFn forward_fn;
    NativeSystemDrafter system_drafter;
    auto* graph_cache_p = use_cuda_graphs ? own_emplace<ForwardGraphCache>() : nullptr;

    auto* dsv4_ws_p = own_value(
        family == model::Family::DeepSeekV4
            ? model::DsV4Workspace::allocate(
                  engine.hf_config(), max_workspace_tokens,
                  mem_plan.capacity.max_logit_rows, local_tp_size)
            : model::DsV4Workspace{});
    auto& dsv4_ws = *dsv4_ws_p;
    auto* kimi_ws_p = own_value(
        family == model::Family::Kimi
            ? model::KimiWorkspace::allocate(
                  engine.hf_config(), max_workspace_tokens,
                  mem_plan.capacity.max_logit_rows, local_tp_size)
            : model::KimiWorkspace{});
    auto& kimi_ws = *kimi_ws_p;
    auto* glm5_ws_p = own_value(
        family == model::Family::Glm5
            ? model::Glm5Workspace::allocate(
                  engine.hf_config(), max_workspace_tokens,
                  mem_plan.capacity.max_logit_rows,
                  engine.hf_config().max_position_embeddings, local_tp_size)
            : model::Glm5Workspace{});
    auto& glm5_ws = *glm5_ws_p;
    auto* gemma4_moe_ws_p = own_emplace<model::Gemma4MoeMlpWorkspace>();
    auto& gemma4_moe_ws = *gemma4_moe_ws_p;
    const bool gemma4_selected = (family == model::Family::Gemma4);
    if (gemma4_selected && engine.hf_config().gemma4_enable_moe) {
        const auto& hf_cfg = engine.hf_config();
        gemma4_moe_ws = model::Gemma4MoeMlpWorkspace::allocate(
            max_workspace_tokens,
            hf_cfg.hidden_size,
            hf_cfg.num_experts,
            hf_cfg.num_experts_per_tok,
            hf_cfg.moe_intermediate_size);
    }
    if (gemma4_selected) gemma4_moe_ws.allocate_row_decode(max_workspace_tokens);
    if (gemma4_selected &&
        engine.hf_config().gemma_hidden_size_per_layer_input > 0) {
        const auto& hf_cfg = engine.hf_config();
        gemma4_moe_ws.allocate_ple(
            max_workspace_tokens,
            hf_cfg.num_hidden_layers *
                hf_cfg.gemma_hidden_size_per_layer_input);
    }

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
    resources.gemma_fwd_cfg = &gemma_fwd_cfg;
    resources.gemma4_fwd_cfg = &gemma4_fwd_cfg;
    resources.max_workspace_tokens = max_workspace_tokens;
    resources.small_spec_graph_tokens = model::qwen35_small_spec_graph_tokens();
    resources.gemma4_moe_ws = &gemma4_moe_ws;
    resources.qwen3_5_la_ws = &qwen3_5_la_ws;
    resources.qwen3_5_moe_ws = &qwen3_5_moe_ws;
    resources.qwen3_5_plan_state = &qwen3_5_plan_state;
    resources.qwen3_5_state_cache = &qwen3_5_state_cache;
    resources.system_drafter = &system_drafter;
    resources.native_mtp_num_drafts = native_mtp_num_drafts;
    resources.nemotron_h_ws = &nemotron_h_ws;
    resources.nemotron_h_state_cache = &nemotron_h_state_cache;
    resources.dsv4_ws = &dsv4_ws;
    resources.kimi_ws = &kimi_ws;
    resources.glm5_ws = &glm5_ws;

    auto* model_holder =
        own_value(arch_entry->create_model(std::move(plan), resources));
    forward_fn.attach_model(model_holder->get());

    // The pipeline registry (program/instance/channel ownership + the single
    // `Dispatch` instance) is constructed once, here, ahead of the batch
    // engine so the engine can hold a non-owning pointer into it for the
    // whole of its lifetime. It is emplaced before `BatchEngine` in
    // `owners_`, so it is torn down after the engine at destruction
    // (reverse-of-construction order): the engine's `dispatch` pointer is
    // never dangling.
    registry_ = own_emplace<pipeline::Registry>();

    auto* executor_p = own_emplace<BatchEngine>(
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
        !has_recurrent_state_cache
            ? nullptr
            : ((family == model::Family::Qwen3_5 || family == model::Family::Qwen3_5Moe)
                   ? &qwen3_5_state_cache
                   : &nemotron_h_state_cache));
    executor_p->dispatch = &registry_->dispatch();
    executor_ = executor_p;

    tp_startup_cpu_barrier(cfg);
    // Rank-0 CUDA-graph replay is unreachable (direct PTIR launches always
    // dispatch through `run_forward_dispatch`'s non-graph path — see
    // batch/forward.hpp); only TP followers ever replay a captured graph
    // (`tp_follower_serve`). Precapturing the full decode lattice on a
    // single-GPU run therefore only costs startup time and VRAM for no
    // benefit, so gate it on `tp_size_ > 1`.
    if (use_cuda_graphs && tp_size_ > 1) capture_forward_graph_lattice(*executor_);
    tp_startup_cpu_barrier(cfg);

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
    const std::uint64_t rs_cache_slots = rs_cache_required
        ? (family == model::Family::NemotronH
               ? static_cast<std::uint64_t>(runtime_state_slots)
               : static_cast<std::uint64_t>(qwen3_5_runtime_rs_slots))
        : 0;
    const std::uint64_t rs_cache_slot_bytes = rs_cache_required
        ? (family == model::Family::NemotronH
               ? static_cast<std::uint64_t>(model::nemotron_h_state_slot_bytes(
                     engine.hf_config(), nemotron_h_mamba_layers, local_tp_size))
               : static_cast<std::uint64_t>(qwen3_5_linear_layers) *
                     (qwen3_5_state_cache.conv_slot_stride_bytes() +
                      qwen3_5_state_cache.recurrent_slot_stride_bytes()) +
                 static_cast<std::uint64_t>(std::max(0, qwen3_5_state_cache.hidden_size())) *
                     sizeof(std::uint16_t))
        : 0;
    const auto max_forward_requests_caps = rs_cache_required
        ? std::min<std::uint64_t>(
              static_cast<std::uint64_t>(mem_plan.capacity.max_forward_requests),
              rs_cache_slots)
        : static_cast<std::uint64_t>(mem_plan.capacity.max_forward_requests);
    nlohmann::json caps = {
        {"abi_version", PIE_DRIVER_ABI_VERSION},
        {"total_pages", c.total_pages},
        {"kv_page_size", mem_plan.kv_page_size},
        {"swap_pool_size", c.swap_pool_size},
        {"rs_cache_required", rs_cache_required},
        {"rs_cache_slots", rs_cache_slots},
        {"rs_cache_slot_bytes", rs_cache_slot_bytes},
        {"max_forward_tokens", mem_plan.capacity.max_forward_tokens},
        {"max_forward_requests", max_forward_requests_caps},
        {"max_page_refs", mem_plan.capacity.max_page_refs},
        {"arch_name", c.arch_name},
        {"vocab_size", c.vocab_size},
        {"max_model_len", c.max_model_len},
        {"activation_dtype", c.activation_dtype},
        {"snapshot_dir", c.snapshot_dir},
        {"storage_backend", c.storage_backend},
        {"max_tile_bytes", c.max_tile_bytes},
        {"preferred_alignment", c.preferred_alignment},
        {"mxfp4_moe_policy", c.mxfp4_moe_policy},
        {"native_mxfp4_moe", c.native_mxfp4_moe},
    };
    caps_json_ = caps.dump();
    return PIE_STATUS_OK;
}

int Context::Impl::register_program(const PieProgramDesc& program, std::uint64_t* program_id) {
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
    if (is_tp_follower()) return PIE_STATUS_UNSUPPORTED;
    std::string err;
    const int rc = registry_->register_channel(channel, binding, &err);
    if (rc != PIE_STATUS_OK && !err.empty()) {
        std::cerr << "[pie-driver-cuda] register_channel: " << err << "\n";
    }
    return rc;
}

int Context::Impl::bind_instance(const PieInstanceDesc& instance, PieInstanceBinding* binding) {
    if (is_tp_follower()) return PIE_STATUS_UNSUPPORTED;
    std::string err;
    const int rc = registry_->bind_instance(instance, binding, &err);
    if (rc != PIE_STATUS_OK && !err.empty()) {
        std::cerr << "[pie-driver-cuda] bind_instance: " << err << "\n";
    }
    return rc;
}

int Context::Impl::launch(const PieLaunchDesc& launch, PieCompletion completion) {
    if (executor_ == nullptr) return PIE_STATUS_CLOSED;
    if (is_tp_follower()) return PIE_STATUS_UNSUPPORTED;
    std::vector<pie_cuda_driver::pipeline::InstanceRecord> launch_instances;
    const int resolve_status =
        registry_->resolve_instances(launch.instance_ids, &launch_instances);
    if (resolve_status != PIE_STATUS_OK) return resolve_status;
    const int resource_status = pie_cuda_driver::abi::validate_launch_resources(
        launch,
        kv_cache_->num_pages(),
        kv_cache_->page_size(),
        executor_->rs_cache != nullptr ? executor_->rs_cache->max_slots() : 0,
        executor_->rs_cache != nullptr
            ? executor_->rs_cache->rs_buffer_num_slots()
            : 0,
        multimodal_limits_);
    if (resource_status != PIE_STATUS_OK) return resource_status;
    LaunchScratch scratch;
    pie_native::LaunchView view = scratch.build(launch, launch_instances);
    const bool has_ptir_launch = !view.ptir_program_hashes.empty();
    if (!has_ptir_launch) {
        return PIE_STATUS_INVALID_ARGUMENT;
    }
    std::string ptir_error;
    const int ptir_status = registry_->validate_launch(view, &ptir_error);
    if (ptir_status != PIE_STATUS_OK) {
        if (!ptir_error.empty()) {
            std::cerr << "[pie-driver-cuda] launch rejected: "
                      << ptir_error << "\n";
        }
        return ptir_status;
    }
    try {
        pie_cuda_driver::handle_fire_batch(
            0, view, *executor_, runtime_, completion);
        return PIE_STATUS_OK;
    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-cuda] launch: " << e.what() << "\n";
        std::vector<std::pair<std::uint64_t, std::uint64_t>>
            channel_notifications;
        try {
            channel_notifications =
                registry_->dispatch().settle_failed_launch(
                view, executor_->cublas.stream());
        } catch (const std::exception& settle_error) {
            std::cerr
                << "[pie-driver-cuda] launch failure settlement: "
                << settle_error.what() << "\n";
        }
        for (std::size_t i = 0; i < launch.terminal_cells.len; ++i) {
            publish_terminal(
                launch.terminal_cells.ptr[i],
                PIE_TERMINAL_OUTCOME_FAILED);
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
    }
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
    if (resize.pool_id != 0) return PIE_STATUS_UNSUPPORTED;
    try {
        if (swap_pool_ != nullptr && swap_pool_->stream() != nullptr) {
            const cudaError_t stream_status = cudaStreamQuery(swap_pool_->stream());
            if (stream_status == cudaErrorNotReady) return PIE_STATUS_UNSUPPORTED;
            CUDA_CHECK(stream_status);
        }
        *swap_pool_ = pie_cuda_driver::SwapPool::allocate_for_cache(
            *kv_cache_, static_cast<int>(resize.target_pages));
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
    drain_async_streams();
    collect_ready_async_resources();
    return registry_->close_instance(instance_id);
}

int Context::Impl::close_channel(std::uint64_t channel_id) {
    if (executor_ == nullptr) return PIE_STATUS_CLOSED;
    if (is_tp_follower()) return PIE_STATUS_UNSUPPORTED;
    drain_async_streams();
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

void Context::fill_caps(PieDriverCaps* caps) const {
    impl_->fill_caps(caps);
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

int Context::launch(const PieLaunchDesc& launch, PieCompletion completion) {
    return impl_->launch(launch, completion);
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
