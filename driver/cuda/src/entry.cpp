#include "model/mistral3.hpp"

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
#include <pie_driver_abi.h>

#include "attention_workspace.hpp"
#include "pie_native/abi_validation.hpp"
#include "config.hpp"
#include "distributed.hpp"
#include "entry_validation.hpp"
#include "cuda_check.hpp"
#include "cuda_memory_planner.hpp"
#include "device_buffer.hpp"
#include "executor/executor.hpp"
#include "hf_snapshot.hpp"
#include "kernels/kv_paged.hpp"
#include "kv_cache.hpp"
#include "mla_cache.hpp"
#include "dsa_cache.hpp"
#include "model/bound_model.hpp"
#include "model/csm_model.hpp"
#include "model/deepseek_v4_model.hpp"
#include "model/gemma2.hpp"
#include "model/gemma2_model.hpp"
#include "model/gemma3n.hpp"
#include "model/gemma3n_model.hpp"
#include "model/gemma4.hpp"
#include "model/gemma4_model.hpp"
#include "model/glm5_model.hpp"
#include "model/kimi_model.hpp"
#include "model/llama_like.hpp"
#include "model/llama_like_model.hpp"
#include "model/loaded_model.hpp"
#include "model/mixtral.hpp"
#include "model/mixtral_model.hpp"
#include "model/nemotron_h.hpp"
#include "model/nemotron_h_forward.hpp"
#include "model/nemotron_h_model.hpp"
#include "model/qwen3.hpp"
#include "model/qwen3_5.hpp"
#include "model/qwen3_5_config.hpp"
#include "model/qwen3_5_forward.hpp"
#include "model/qwen3_5_model.hpp"
#include "model/qwen3_5_moe.hpp"
#include "model/qwen3_5_moe_forward.hpp"
#include "model/qwen3_5_moe_model.hpp"
#include "model/qwen3_forward.hpp"
#include "model/qwen3_vl_model.hpp"
#include "ops/gemm.hpp"
#include "recurrent_state_cache.hpp"
#include "swap_pool.hpp"

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

void CUDART_CB finish_async_completion(void* userdata) {
    std::unique_ptr<AsyncCompletionContext> ctx(
        static_cast<AsyncCompletionContext*>(userdata));
    if (ctx == nullptr) return;
    if (ctx->runtime.notify != nullptr && ctx->completion.wait_id != 0) {
        ctx->runtime.notify(
            ctx->runtime.ctx,
            ctx->completion.wait_id,
            ctx->completion.target_epoch);
    }
}

struct ProgramRecord {
    std::uint64_t program_id = 0;
    std::uint64_t program_hash = 0;
    std::vector<std::uint8_t> canonical;
    std::vector<std::uint8_t> sidecar;
};

struct InstanceRecord {
    std::uint64_t instance_id = 0;
    std::uint64_t program_id = 0;
    std::uint64_t program_hash = 0;
};

struct LaunchScratch {
    std::vector<std::uint64_t> ptir_program_hashes;
    std::vector<std::uint64_t> ptir_program_instances;
    std::vector<std::uint64_t> ptir_host_put_channels;
    std::vector<std::uint8_t> ptir_host_put_blob;
    std::vector<std::uint32_t> ptir_host_put_lens;
    std::vector<std::uint32_t> ptir_host_put_indptr;

    pie_native::LaunchView build(
        const PieLaunchDesc& launch,
        const std::vector<InstanceRecord>& instances) {
        const std::size_t lanes = instances.size();
        ptir_program_hashes.clear();
        ptir_program_instances.clear();
        ptir_program_hashes.reserve(lanes);
        ptir_program_instances.reserve(lanes);
        for (const InstanceRecord& inst : instances) {
            ptir_program_hashes.push_back(inst.program_hash);
            ptir_program_instances.push_back(inst.instance_id);
        }

        ptir_host_put_channels.clear();
        ptir_host_put_blob.clear();
        ptir_host_put_lens.clear();
        if (launch.host_put_indptr.ptr != nullptr && launch.host_put_indptr.len == lanes + 1) {
            ptir_host_put_indptr.assign(
                launch.host_put_indptr.ptr,
                launch.host_put_indptr.ptr + launch.host_put_indptr.len);
        } else {
            ptir_host_put_indptr.assign(lanes + 1, 0);
        }
        if (launch.ptir_host_put_values.ptr != nullptr) {
            for (std::size_t i = 0; i < launch.ptir_host_put_values.len; ++i) {
                const PieChannelValueDesc& value = launch.ptir_host_put_values.ptr[i];
                ptir_host_put_channels.push_back(value.channel_id);
                ptir_host_put_lens.push_back(
                    static_cast<std::uint32_t>(value.bytes.len));
                if (value.bytes.ptr != nullptr && value.bytes.len > 0) {
                    ptir_host_put_blob.insert(
                        ptir_host_put_blob.end(),
                        value.bytes.ptr,
                        value.bytes.ptr + value.bytes.len);
                }
            }
        }

        pie_native::LaunchView view{};
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
        view.ptir_program_host_put_channels = pie_native::slice_from_u64(ptir_host_put_channels.data(), ptir_host_put_channels.size());
        view.ptir_program_host_put_blob = pie_native::slice_from_u8(ptir_host_put_blob.data(), ptir_host_put_blob.size());
        view.ptir_program_host_put_lens = pie_native::slice_from_u32(ptir_host_put_lens.data(), ptir_host_put_lens.size());
        view.ptir_program_host_put_indptr = pie_native::slice_from_u32(ptir_host_put_indptr.data(), ptir_host_put_indptr.size());
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

class CudaDriver {
  public:
    explicit CudaDriver(const PieDriverCreateDesc& desc) : runtime_(desc.runtime) {}
    ~CudaDriver() {
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

    int initialize(const std::string& config_path);
    void fill_caps(PieDriverCaps* caps) const {
        if (caps == nullptr) return;
        caps->json_bytes = reinterpret_cast<const std::uint8_t*>(caps_json_.data());
        caps->json_len = caps_json_.size();
    }

    int register_program(const PieProgramDesc& program, std::uint64_t* program_id);
    int bind_instance(const PieInstanceDesc& instance, PieInstanceBinding* binding);
    int launch(const PieLaunchDesc& launch, PieCompletion completion);
    int copy_kv(const PieKvCopyDesc& copy, PieCompletion completion);
    int copy_state(const PieStateCopyDesc& copy, PieCompletion completion);
    int resize_pool(const PiePoolResizeDesc& resize, PieCompletion completion);
    int close_instance(std::uint64_t instance_id);

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
    pie_cuda_driver::Executor* executor_ = nullptr;
    pie_cuda_driver::KvCache* kv_cache_ = nullptr;
    pie_cuda_driver::SwapPool* swap_pool_ = nullptr;
    pie_cuda_driver::NcclComm* tp_comm_ = nullptr;
    std::string caps_json_;
    std::string tp_cpu_gate_key_;
    int device_ordinal_ = 0;
    int tp_size_ = 1;
    int tp_rank_ = 0;
    pie_cuda_driver::abi::MultimodalLimits multimodal_limits_;
    std::uint64_t next_program_id_ = 1;
    std::uint64_t next_instance_id_ = 1;
    std::atomic<bool> tp_follower_stop_{false};
    std::thread tp_follower_thread_;
    std::vector<std::unique_ptr<PendingAsyncResources>> pending_async_resources_;
    std::unordered_map<std::uint64_t, ProgramRecord> programs_;
    std::unordered_map<std::uint64_t, std::uint64_t> program_ids_by_hash_;
    std::unordered_map<std::uint64_t, InstanceRecord> instances_;
};

int configured_mtp_num_drafts(const pie_cuda_driver::Config& cfg) {
    static const int forced = [] {
        const char* v = std::getenv("PIE_MTP_DRAFT_TOKENS");
        if (v == nullptr || v[0] == '\0') return -1;
        return std::clamp(std::atoi(v), 0, 32);
    }();
    if (forced >= 0) return forced;
    return cfg.model.mtp_num_drafts;
}

int CudaDriver::initialize(const std::string& config_path) {
    using namespace pie_cuda_driver;

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

    {
        const auto& mt = engine.hf_config().model_type;
        const bool supported =
            mt == "qwen3"
         || mt == "qwen3_5" || mt == "qwen3_5_text"
         || mt == "qwen3_5_moe" || mt == "qwen3_5_moe_text"
         || mt == "qwen3_moe"
         || mt == "qwen3_vl" || mt == "qwen3_vl_text"
         || mt == "qwen2"
         || mt == "llama" || mt == "llama3"
         || mt == "mistral" || mt == "mistral3" || mt == "ministral3"
         || mt == "mixtral"
         || mt == "gpt_oss"
         || mt == "phi3"
         || mt == "olmo2" || mt == "olmo3"
         || mt == "gemma2"
         || mt == "gemma3" || mt == "gemma3_text"
         || mt == "gemma4" || mt == "gemma4_text"
         || mt == "gemma3n" || mt == "gemma3n_text"
         || mt == "nemotron_h"
         || mt == "deepseek_v4"
         || mt == "deepseek_v2" || mt == "deepseek_v3"
         || mt == "kimi_k2"
         || mt == "glm_moe_dsa"
         || mt == "csm";
        if (!supported) {
            std::cerr << "[pie-driver-cuda] unsupported arch '" << mt << "'\n";
            return PIE_STATUS_UNSUPPORTED;
        }
    }

    auto* bound_model_p = own_value(model::bind_cuda_model(engine, verbose));
    auto& bound_model = *bound_model_p;
    auto& weights_llama = bound_model.llama;
    auto& weights_gemma = bound_model.gemma;
    auto& weights_gemma4 = bound_model.gemma4;
    auto& weights_gemma3n = bound_model.gemma3n;
    auto& weights_mixtral = bound_model.mixtral;
    auto& weights_qwen3_5 = bound_model.qwen3_5;
    auto& weights_qwen3_5_moe = bound_model.qwen3_5_moe;
    auto& weights_nemotron_h = bound_model.nemotron_h;
    auto& weights_dsv4 = bound_model.deepseek_v4;
    auto& weights_kimi = bound_model.kimi;
    auto& weights_glm5 = bound_model.glm5;

    const bool is_gemma_arch = bound_model.is_gemma();
    const bool is_gemma4_arch = bound_model.is_gemma4();
    const bool is_gemma3n_arch = bound_model.is_gemma3n();
    const bool is_mixtral_arch = bound_model.is_mixtral();
    const bool is_qwen3_5_arch = bound_model.is_qwen3_5();
    const bool is_qwen3_5_moe_arch = bound_model.is_qwen3_5_moe();
    const bool is_nemotron_h_arch = bound_model.is_nemotron_h();
    const bool is_dsv4_arch = bound_model.is_deepseek_v4();
    const bool is_kimi_arch = bound_model.is_kimi();
    const bool is_glm5_arch = bound_model.is_glm5();
    const bool is_qwen3_vl_arch = bound_model.is_qwen3_vl();
    if (is_gemma4_arch && bound_model.has_vision) {
        multimodal_limits_.gemma4_pool_kernel =
            bound_model.gemma4_vision.config.pooling_kernel_size;
        const auto* positions =
            bound_model.gemma4_vision.patch_position_embedding;
        if (positions != nullptr && positions->shape().size() > 1 &&
            positions->shape()[1] <=
                static_cast<std::size_t>(std::numeric_limits<int>::max())) {
            multimodal_limits_.gemma4_position_table =
                static_cast<int>(positions->shape()[1]);
        }
    }
    if (is_qwen3_vl_arch && bound_model.has_vision) {
        const auto& vision = bound_model.qwen3_vl_vision.config;
        const std::int64_t patch_dim =
            static_cast<std::int64_t>(vision.in_channels) *
            vision.temporal_patch_size *
            vision.patch_size *
            vision.patch_size;
        const std::int64_t merge_unit =
            static_cast<std::int64_t>(vision.spatial_merge_size) *
            vision.spatial_merge_size;
        if (patch_dim > 0 &&
            patch_dim <= std::numeric_limits<int>::max() &&
            merge_unit > 0 &&
            merge_unit <= std::numeric_limits<int>::max()) {
            multimodal_limits_.qwen3_vl_patch_dim =
                static_cast<int>(patch_dim);
            multimodal_limits_.qwen3_vl_merge_unit =
                static_cast<int>(merge_unit);
        }
    }
    if (bound_model.has_audio) {
        multimodal_limits_.audio_mel_bins = 128;
    }
    const int native_mtp_num_drafts = configured_mtp_num_drafts(cfg);

    const int local_tp_size = tp_size_;
    const int local_q_heads = engine.hf_config().num_attention_heads / local_tp_size;
    const int local_kv_heads = engine.hf_config().num_key_value_heads / local_tp_size;
    int max_mlp_intermediate = engine.hf_config().intermediate_size / local_tp_size;
    int max_Hq = local_q_heads * engine.hf_config().head_dim;
    int max_Hk = local_kv_heads * engine.hf_config().head_dim;
    if (is_gemma4_arch) {
        for (int v : weights_gemma4.per_layer_intermediate)
            max_mlp_intermediate = std::max(max_mlp_intermediate, v);
        for (int d : weights_gemma4.per_layer_head_dim) {
            max_Hq = std::max(max_Hq, local_q_heads * d);
            max_Hk = std::max(max_Hk, local_kv_heads * d);
        }
    } else if (is_gemma3n_arch) {
        for (int v : weights_gemma3n.per_layer_intermediate)
            max_mlp_intermediate = std::max(max_mlp_intermediate, v / local_tp_size);
    } else if (is_nemotron_h_arch) {
        const auto& hf_n = engine.hf_config();
        max_mlp_intermediate = std::max(
            max_mlp_intermediate,
            std::max(hf_n.moe_intermediate_size / local_tp_size,
                     hf_n.shared_expert_intermediate_size / local_tp_size));
    }

    std::vector<bool> qwen3_5_layer_is_linear;
    if (is_qwen3_5_arch || is_qwen3_5_moe_arch) {
        const std::size_t num_layers = is_qwen3_5_arch
            ? weights_qwen3_5.layers.size()
            : weights_qwen3_5_moe.layers.size();
        qwen3_5_layer_is_linear.resize(num_layers);
        for (std::size_t L = 0; L < num_layers; ++L) {
            qwen3_5_layer_is_linear[L] = is_qwen3_5_arch
                ? (weights_qwen3_5.layers[L].kind ==
                   model::Qwen3_5LayerWeights::Kind::LinearAttn)
                : (weights_qwen3_5_moe.layers[L].kind ==
                   model::Qwen3_5MoeLayerWeights::Kind::LinearAttn);
        }
    }
    const int qwen3_5_linear_layers = static_cast<int>(std::count(
        qwen3_5_layer_is_linear.begin(), qwen3_5_layer_is_linear.end(), true));

    std::vector<bool> nemotron_h_layer_is_mamba;
    if (is_nemotron_h_arch) {
        nemotron_h_layer_is_mamba.resize(weights_nemotron_h.layers.size());
        for (std::size_t L = 0; L < weights_nemotron_h.layers.size(); ++L) {
            nemotron_h_layer_is_mamba[L] =
                weights_nemotron_h.layers[L].kind ==
                model::NemotronHLayerWeights::Kind::Mamba;
        }
    }
    const int nemotron_h_mamba_layers = static_cast<int>(std::count(
        nemotron_h_layer_is_mamba.begin(), nemotron_h_layer_is_mamba.end(), true));
    const int nemotron_h_attention_layer_count = is_nemotron_h_arch
        ? model::nemotron_h_attention_layers(engine.hf_config())
        : 0;

    const auto kv_format = kv_cache_format_from_string(
        cfg.batching.kv_cache_dtype, cfg.model.dtype);
    const bool use_cuda_graphs = true;
    const bool graph_capable_forward =
        use_cuda_graphs && bound_model.is_llama_like() && kv_format.is_native_bf16();
    const auto runtime_quant_scratch_base =
        graph_capable_forward
            ? runtime_quant_scratch_spec(engine, /*max_tokens=*/0)
            : ops::RuntimeQuantScratchSpec{};

    const CudaMemoryPlan mem_plan = plan_cuda_memory(
        cfg, engine.hf_config(), max_mlp_intermediate, max_Hq, max_Hk,
        is_gemma4_arch, weights_gemma4.per_layer_head_dim,
        weights_gemma4.kv_source_layer, is_qwen3_5_arch,
        is_qwen3_5_moe_arch, qwen3_5_linear_layers,
        is_nemotron_h_arch, nemotron_h_mamba_layers,
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
        ((is_qwen3_5_arch || is_qwen3_5_moe_arch) && qwen3_5_linear_layers > 0) ||
        (is_nemotron_h_arch && nemotron_h_mamba_layers > 0);
    const int runtime_state_slots = mem_plan.state_slots;
    const int graph_pad_slot =
        has_recurrent_state_cache && runtime_state_slots > 0 && graph_pad_page >= 0
            ? runtime_state_slots
            : -1;
    const int allocated_state_slots =
        runtime_state_slots + (graph_pad_slot >= 0 ? 1 : 0);

    auto* ws_p = own_value(model::Qwen3Workspace::allocate_full(
        engine.hf_config(), max_workspace_tokens,
        max_mlp_intermediate, max_Hq, max_Hk,
        mem_plan.capacity.max_logit_rows));
    auto& ws = *ws_p;

    auto* kv_cache_p = own_value(
        is_dsv4_arch
            ? KvCache::allocate(
                  engine.hf_config().num_hidden_layers,
                  physical_kv_pages,
                  mem_plan.kv_page_size,
                  1,
                  engine.hf_config().head_dim,
                  kv_format)
            : (is_kimi_arch || is_glm5_arch)
                ? KvCache::allocate(
                      engine.hf_config().num_hidden_layers,
                      physical_kv_pages,
                      mem_plan.kv_page_size,
                      1,
                      1,
                      kv_format)
                : is_gemma4_arch
                    ? KvCache::allocate_per_layer(
                          engine.hf_config().num_hidden_layers,
                          physical_kv_pages,
                          mem_plan.kv_page_size,
                          local_kv_heads,
                          weights_gemma4.per_layer_head_dim,
                          weights_gemma4.kv_source_layer,
                          weights_gemma4.per_layer_num_kv_heads,
                          kv_format)
                    : is_nemotron_h_arch
                        ? KvCache::allocate(
                              nemotron_h_attention_layer_count,
                              physical_kv_pages,
                              mem_plan.kv_page_size,
                              local_kv_heads,
                              engine.hf_config().head_dim_kernel,
                              kv_format)
                        : KvCache::allocate(
                              engine.hf_config().num_hidden_layers,
                              physical_kv_pages,
                              mem_plan.kv_page_size,
                              local_kv_heads,
                              engine.hf_config().head_dim_kernel,
                              kv_format));
    auto& kv_cache = *kv_cache_p;

    auto* mla_cache_p = own_value(
        (is_kimi_arch || is_glm5_arch)
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
        is_glm5_arch
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

    if (is_qwen3_5_arch || is_qwen3_5_moe_arch) {
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
        if (is_qwen3_5_moe_arch) {
            qwen3_5_moe_ws = model::Qwen3_5MoeMlpWorkspace::allocate(
                max_workspace_tokens,
                cfg_q.hidden_size,
                cfg_q.num_experts,
                cfg_q.num_experts_per_tok,
                cfg_q.moe_intermediate_size / local_tp_size,
                cfg_q.shared_expert_intermediate_size / local_tp_size);
        }
    } else if (is_nemotron_h_arch) {
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
        if (is_qwen3_vl_arch && hf.qwen3_vl_mrope_interleaved &&
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
    if (is_gemma4_arch) {
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
        is_dsv4_arch
            ? model::DsV4Workspace::allocate(
                  engine.hf_config(), max_workspace_tokens,
                  mem_plan.capacity.max_logit_rows, local_tp_size)
            : model::DsV4Workspace{});
    auto& dsv4_ws = *dsv4_ws_p;
    auto* kimi_ws_p = own_value(
        is_kimi_arch
            ? model::KimiWorkspace::allocate(
                  engine.hf_config(), max_workspace_tokens,
                  mem_plan.capacity.max_logit_rows, local_tp_size)
            : model::KimiWorkspace{});
    auto& kimi_ws = *kimi_ws_p;
    auto* glm5_ws_p = own_value(
        is_glm5_arch
            ? model::Glm5Workspace::allocate(
                  engine.hf_config(), max_workspace_tokens,
                  mem_plan.capacity.max_logit_rows,
                  engine.hf_config().max_position_embeddings, local_tp_size)
            : model::Glm5Workspace{});
    auto& glm5_ws = *glm5_ws_p;
    auto* gemma4_moe_ws_p = own_emplace<model::Gemma4MoeMlpWorkspace>();
    auto& gemma4_moe_ws = *gemma4_moe_ws_p;
    if (is_gemma4_arch && engine.hf_config().gemma4_enable_moe) {
        const auto& hf_cfg = engine.hf_config();
        gemma4_moe_ws = model::Gemma4MoeMlpWorkspace::allocate(
            max_workspace_tokens,
            hf_cfg.hidden_size,
            hf_cfg.num_experts,
            hf_cfg.num_experts_per_tok,
            hf_cfg.moe_intermediate_size);
    }
    if (is_gemma4_arch) gemma4_moe_ws.allocate_row_decode(max_workspace_tokens);
    if (is_gemma4_arch &&
        engine.hf_config().gemma_hidden_size_per_layer_input > 0) {
        const auto& hf_cfg = engine.hf_config();
        gemma4_moe_ws.allocate_ple(
            max_workspace_tokens,
            hf_cfg.num_hidden_layers *
                hf_cfg.gemma_hidden_size_per_layer_input);
    }

    if (bound_model.is_csm()) {
        auto* csm_model = own_emplace<model::CsmModel>(std::move(bound_model.csm));
        forward_fn.attach_model(csm_model);
    } else if (is_qwen3_vl_arch) {
        auto* model_p = own_emplace<model::Qwen3VLModel>(
            weights_llama, engine.hf_config(), kv_cache,
            fwd_cfg, max_workspace_tokens,
            bound_model.has_vision ? &bound_model.qwen3_vl_vision : nullptr);
        forward_fn.attach_model(model_p);
    } else if (is_gemma4_arch) {
        auto* model_p = own_emplace<model::Gemma4Model>(
            weights_gemma4, engine.hf_config(),
            gemma4_moe_ws, kv_cache,
            gemma4_fwd_cfg, model::qwen35_small_spec_graph_tokens(),
            bound_model.has_vision ? &bound_model.gemma4_vision : nullptr,
            bound_model.has_audio ? &bound_model.gemma4_audio : nullptr);
        forward_fn.attach_model(model_p);
    } else if (is_gemma3n_arch) {
        model::Gemma3nForwardCfg gemma3n_fwd_cfg{};
        gemma3n_fwd_cfg.final_logit_softcap = engine.hf_config().gemma_final_logit_softcap;
        gemma3n_fwd_cfg.tp_size = local_tp_size;
        gemma3n_fwd_cfg.tp_comm = tp_comm_ptr;
        auto* model_p = own_emplace<model::Gemma3nModel>(
            weights_gemma3n, engine.hf_config(), gemma3n_fwd_cfg);
        forward_fn.attach_model(model_p);
    } else if (is_gemma_arch) {
        auto* model_p = own_emplace<model::Gemma2Model>(
            weights_gemma, engine.hf_config(), gemma_fwd_cfg);
        forward_fn.attach_model(model_p);
    } else if (is_mixtral_arch) {
        auto* model_p = own_emplace<model::MixtralModel>(
            weights_mixtral, engine.hf_config(), fwd_cfg,
            engine.hf_config().num_experts,
            engine.hf_config().num_experts_per_tok);
        forward_fn.attach_model(model_p);
    } else if (is_nemotron_h_arch) {
        auto* model_p = own_emplace<model::NemotronHModel>(
            weights_nemotron_h, engine.hf_config(),
            nemotron_h_ws, nemotron_h_state_cache, kv_cache,
            fwd_cfg, local_tp_size, tp_comm_ptr);
        forward_fn.attach_model(model_p);
    } else if (is_qwen3_5_arch) {
        auto* model_p = own_emplace<model::Qwen35Model>(
            weights_qwen3_5, engine.hf_config(),
            qwen3_5_la_ws, qwen3_5_state_cache, qwen3_5_plan_state,
            kv_cache, local_tp_size, tp_comm_ptr,
            !flashinfer_decode_supports_gqa(
                engine.hf_config().num_attention_heads /
                std::max(1, engine.hf_config().num_key_value_heads)),
            model::qwen35_small_spec_graph_tokens(),
            kv_cache.format().is_native_bf16() &&
                !model::qwen35_forward_profile_enabled(),
            kv_cache.format().is_native_bf16() && !kv_cache.hnd_layout() &&
                model::qwen35_small_spec_graph_tokens() > 0);
        forward_fn.attach_model(model_p);
        if (weights_qwen3_5.mtp.has_value() && native_mtp_num_drafts > 0) {
            model_p->wire_system_drafter(
                system_drafter, native_mtp_num_drafts,
                model::qwen35_mtp_draft_position_offset(),
                model::qwen35_mtp_prefix_global_cache(),
                model::qwen35_mtp_fused_gemv_enabled());
        }
    } else if (is_qwen3_5_moe_arch) {
        auto* model_p = own_emplace<model::Qwen35MoeModel>(
            weights_qwen3_5_moe, engine.hf_config(),
            qwen3_5_la_ws, qwen3_5_moe_ws,
            qwen3_5_state_cache, qwen3_5_plan_state,
            kv_cache, local_tp_size, tp_comm_ptr,
            !flashinfer_decode_supports_gqa(
                engine.hf_config().num_attention_heads /
                std::max(1, engine.hf_config().num_key_value_heads)),
            model::qwen35_small_spec_graph_tokens(),
            kv_cache.format().is_native_bf16(),
            kv_cache.format().is_native_bf16() && !kv_cache.hnd_layout() &&
                model::qwen35_small_spec_graph_tokens() > 0);
        forward_fn.attach_model(model_p);
        if (weights_qwen3_5_moe.mtp.has_value() && native_mtp_num_drafts > 0) {
            model_p->wire_system_drafter(
                system_drafter, native_mtp_num_drafts,
                model::qwen35_mtp_draft_position_offset(),
                model::qwen35_mtp_prefix_global_cache());
        }
    } else if (is_dsv4_arch) {
        auto* model_p = own_emplace<model::DsV4Model>(
            weights_dsv4, engine.hf_config(), dsv4_ws,
            local_tp_size, tp_rank_, tp_comm_ptr, true);
        forward_fn.attach_model(model_p);
    } else if (is_kimi_arch) {
        const bool supports_tp_greedy_argmax =
            local_tp_size > 1 && weights_kimi.lm_head_tp_sharded;
        auto* model_p = own_emplace<model::KimiModel>(
            weights_kimi, engine.hf_config(), kimi_ws, mla_cache,
            local_tp_size, tp_comm_ptr, true, supports_tp_greedy_argmax);
        forward_fn.attach_model(model_p);
    } else if (is_glm5_arch) {
        auto* model_p = own_emplace<model::Glm5Model>(
            weights_glm5, engine.hf_config(), glm5_ws, mla_cache, dsa_cache,
            local_tp_size, tp_comm_ptr, true);
        forward_fn.attach_model(model_p);
    } else {
        const bool supports_tp_greedy_argmax =
            local_tp_size > 1 && weights_llama.lm_head_tp_shard != nullptr;
        auto* model_p = own_emplace<model::LlamaLikeModel>(
            weights_llama, engine.hf_config(), kv_cache,
            fwd_cfg, supports_tp_greedy_argmax);
        forward_fn.attach_model(model_p);
    }

    auto* executor_p = own_emplace<Executor>(
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
            : ((is_qwen3_5_arch || is_qwen3_5_moe_arch)
                   ? &qwen3_5_state_cache
                   : &nemotron_h_state_cache));
    executor_ = executor_p;

    tp_startup_cpu_barrier(cfg);
    if (use_cuda_graphs) capture_forward_graph_lattice(*executor_);
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
        ? (is_nemotron_h_arch
               ? static_cast<std::uint64_t>(runtime_state_slots)
               : static_cast<std::uint64_t>(qwen3_5_runtime_rs_slots))
        : 0;
    const std::uint64_t rs_cache_slot_bytes = rs_cache_required
        ? (is_nemotron_h_arch
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

int CudaDriver::register_program(const PieProgramDesc& program, std::uint64_t* program_id) {
    if (is_tp_follower()) return PIE_STATUS_UNSUPPORTED;
    auto found = program_ids_by_hash_.find(program.program_hash);
    if (found != program_ids_by_hash_.end()) {
        if (program_id != nullptr) *program_id = found->second;
        return PIE_STATUS_OK;
    }
    if (program.canonical_bytes.len == 0) return PIE_STATUS_INVALID_ARGUMENT;
    if (!executor_->ptir_dispatch)
        executor_->ptir_dispatch = std::make_unique<pie_cuda_driver::ptir::PtirDispatch>();
    std::string err;
    const int rc = executor_->ptir_dispatch->register_program(
        program.program_hash,
        pie_native::ByteSlice{program.canonical_bytes.ptr, program.canonical_bytes.len},
        pie_native::ByteSlice{program.sidecar_bytes.ptr, program.sidecar_bytes.len},
        &err);
    if (rc != PIE_STATUS_OK) {
        if (!err.empty()) {
            std::cerr << "[pie-driver-cuda] register_program: " << err << "\n";
        }
        return rc;
    }

    ProgramRecord record;
    record.program_id = next_program_id_++;
    record.program_hash = program.program_hash;
    record.canonical.assign(program.canonical_bytes.ptr,
                            program.canonical_bytes.ptr + program.canonical_bytes.len);
    if (program.sidecar_bytes.ptr != nullptr && program.sidecar_bytes.len > 0) {
        record.sidecar.assign(program.sidecar_bytes.ptr,
                              program.sidecar_bytes.ptr + program.sidecar_bytes.len);
    }
    program_ids_by_hash_[record.program_hash] = record.program_id;
    if (program_id != nullptr) *program_id = record.program_id;
    programs_.emplace(record.program_id, std::move(record));
    return PIE_STATUS_OK;
}

int CudaDriver::bind_instance(const PieInstanceDesc& instance, PieInstanceBinding* binding) {
    if (is_tp_follower()) return PIE_STATUS_UNSUPPORTED;
    auto pit = programs_.find(instance.program_id);
    if (pit == programs_.end()) return PIE_STATUS_INVALID_ARGUMENT;
    std::vector<std::uint64_t> channel_ids(
        instance.channel_ids.ptr,
        instance.channel_ids.ptr + instance.channel_ids.len);
    std::vector<PieChannelWait> waits;
    if (instance.channel_waits.len > 0) {
        waits.assign(instance.channel_waits.ptr,
                     instance.channel_waits.ptr + instance.channel_waits.len);
    }
    if (waits.empty()) waits.assign(channel_ids.size(), PieChannelWait{});
    if (waits.size() != channel_ids.size()) return PIE_STATUS_INVALID_ARGUMENT;
    std::vector<PieChannelValueDesc> seeds;
    if (instance.seed_values.len > 0) {
        seeds.assign(instance.seed_values.ptr,
                     instance.seed_values.ptr + instance.seed_values.len);
    }

    if (!executor_->ptir_dispatch)
        executor_->ptir_dispatch = std::make_unique<pie_cuda_driver::ptir::PtirDispatch>();
    const std::uint64_t instance_id =
        instance.requested_instance_id != 0 ? instance.requested_instance_id : next_instance_id_++;
    std::string err;
    const int rc = executor_->ptir_dispatch->bind_instance(
        instance_id,
        pit->second.program_hash,
        instance.pacing_wait_id,
        channel_ids,
        waits,
        seeds,
        binding,
        &err);
    if (rc != PIE_STATUS_OK) {
        if (!err.empty()) {
            std::cerr << "[pie-driver-cuda] bind_instance: " << err << "\n";
        }
        return rc;
    }

    instances_[instance_id] = InstanceRecord{
        instance_id, instance.program_id, pit->second.program_hash};
    return PIE_STATUS_OK;
}

int CudaDriver::launch(const PieLaunchDesc& launch, PieCompletion completion) {
    if (executor_ == nullptr) return PIE_STATUS_CLOSED;
    if (is_tp_follower()) return PIE_STATUS_UNSUPPORTED;
    std::vector<InstanceRecord> launch_instances;
    launch_instances.reserve(launch.instance_ids.len);
    for (std::size_t i = 0; i < launch.instance_ids.len; ++i) {
        auto it = instances_.find(launch.instance_ids.ptr[i]);
        if (it == instances_.end()) return PIE_STATUS_INVALID_ARGUMENT;
        launch_instances.push_back(it->second);
    }
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
    try {
        if ((launch_instances.empty() || view.qo_indptr.empty() || view.token_ids.empty()) &&
            has_ptir_launch) {
            if (!executor_->ptir_dispatch)
                executor_->ptir_dispatch = std::make_unique<pie_cuda_driver::ptir::PtirDispatch>();
            executor_->ptir_dispatch->run(
                view, nullptr, 0, executor_->cublas.stream(), &runtime_, completion);
        } else {
            pie_cuda_driver::handle_fire_batch(
                0, view, *executor_, runtime_, completion);
        }
        return PIE_STATUS_OK;
    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-cuda] launch: " << e.what() << "\n";
        return PIE_STATUS_DRIVER_ERROR;
    }
}

int CudaDriver::copy_kv(const PieKvCopyDesc& copy, PieCompletion completion) {
    if (executor_ == nullptr) return PIE_STATUS_CLOSED;
    if (is_tp_follower()) return PIE_STATUS_UNSUPPORTED;
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

int CudaDriver::copy_state(const PieStateCopyDesc& copy, PieCompletion completion) {
    if (executor_ == nullptr) return PIE_STATUS_CLOSED;
    if (is_tp_follower()) return PIE_STATUS_UNSUPPORTED;
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

int CudaDriver::resize_pool(const PiePoolResizeDesc& resize, PieCompletion completion) {
    if (executor_ == nullptr) return PIE_STATUS_CLOSED;
    if (is_tp_follower()) return PIE_STATUS_UNSUPPORTED;
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

int CudaDriver::close_instance(std::uint64_t instance_id) {
    if (executor_ == nullptr) return PIE_STATUS_CLOSED;
    if (is_tp_follower()) return PIE_STATUS_UNSUPPORTED;
    drain_async_streams();
    collect_ready_async_resources();
    auto it = instances_.find(instance_id);
    if (it == instances_.end()) return PIE_STATUS_CLOSED;
    if (executor_->ptir_dispatch) executor_->ptir_dispatch->close_instance(instance_id);
    instances_.erase(it);
    return PIE_STATUS_OK;
}

PieDriver* create_driver_impl(const PieDriverCreateDesc* desc, PieDriverCaps* caps) {
    std::memset(caps, 0, sizeof(*caps));
    const std::string config_path(
        reinterpret_cast<const char*>(desc->config_bytes.ptr),
        desc->config_bytes.len);
    auto driver = std::make_unique<CudaDriver>(*desc);
    const int rc = driver->initialize(config_path);
    if (rc != PIE_STATUS_OK) return nullptr;
    driver->fill_caps(caps);
    return reinterpret_cast<PieDriver*>(driver.release());
}

CudaDriver* as_driver(PieDriver* driver) {
    return reinterpret_cast<CudaDriver*>(driver);
}

}  // namespace

extern "C" PieDriver* pie_cuda_create(const PieDriverCreateDesc* desc,
                                       PieDriverCaps* caps) {
    if (pie_native::abi::validate_create_desc(desc, caps) != PIE_STATUS_OK) {
        return nullptr;
    }
    try {
        return create_driver_impl(desc, caps);
    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-cuda] create: " << e.what() << "\n";
        return nullptr;
    } catch (...) {
        std::cerr << "[pie-driver-cuda] create: unknown exception\n";
        return nullptr;
    }
}

extern "C" int32_t pie_cuda_register_program(PieDriver* driver,
                                              const PieProgramDesc* program,
                                              std::uint64_t* program_id) {
    const int status = pie_native::abi::validate_program_desc(program, program_id);
    if (status != PIE_STATUS_OK) return status;
    if (driver == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    try {
        return as_driver(driver)->register_program(*program, program_id);
    } catch (...) {
        return PIE_STATUS_DRIVER_ERROR;
    }
}

extern "C" int32_t pie_cuda_bind_instance(PieDriver* driver,
                                           const PieInstanceDesc* instance,
                                           PieInstanceBinding* binding) {
    const int status = pie_native::abi::validate_instance_desc(instance, binding);
    if (status != PIE_STATUS_OK) return status;
    if (driver == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    try {
        return as_driver(driver)->bind_instance(*instance, binding);
    } catch (...) {
        return PIE_STATUS_DRIVER_ERROR;
    }
}

extern "C" int32_t pie_cuda_launch(PieDriver* driver,
                                    const PieLaunchDesc* launch,
                                    PieCompletion completion) {
    const int status = pie_native::abi::validate_launch_desc(launch);
    if (status != PIE_STATUS_OK) return status;
    if (driver == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    try {
        return as_driver(driver)->launch(*launch, completion);
    } catch (...) {
        return PIE_STATUS_DRIVER_ERROR;
    }
}

extern "C" int32_t pie_cuda_copy_kv(PieDriver* driver,
                                     const PieKvCopyDesc* copy,
                                     PieCompletion completion) {
    const int status = pie_native::abi::validate_kv_copy_desc(copy);
    if (status != PIE_STATUS_OK) return status;
    if (driver == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    try {
        return as_driver(driver)->copy_kv(*copy, completion);
    } catch (...) {
        return PIE_STATUS_DRIVER_ERROR;
    }
}

extern "C" int32_t pie_cuda_copy_state(PieDriver* driver,
                                        const PieStateCopyDesc* copy,
                                        PieCompletion completion) {
    const int status = pie_native::abi::validate_state_copy_desc(copy);
    if (status != PIE_STATUS_OK) return status;
    if (driver == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    try {
        return as_driver(driver)->copy_state(*copy, completion);
    } catch (...) {
        return PIE_STATUS_DRIVER_ERROR;
    }
}

extern "C" int32_t pie_cuda_resize_pool(PieDriver* driver,
                                         const PiePoolResizeDesc* resize,
                                         PieCompletion completion) {
    const int status = pie_native::abi::validate_pool_resize_desc(resize);
    if (status != PIE_STATUS_OK) return status;
    if (driver == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    try {
        return as_driver(driver)->resize_pool(*resize, completion);
    } catch (...) {
        return PIE_STATUS_DRIVER_ERROR;
    }
}

extern "C" int32_t pie_cuda_close_instance(PieDriver* driver,
                                            std::uint64_t instance_id) {
    if (driver == nullptr) return PIE_STATUS_INVALID_ARGUMENT;
    try {
        return as_driver(driver)->close_instance(instance_id);
    } catch (...) {
        return PIE_STATUS_DRIVER_ERROR;
    }
}

extern "C" void pie_cuda_destroy(PieDriver* driver) {
    delete as_driver(driver);
}
