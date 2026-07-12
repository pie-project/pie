#include "model/loaded_model.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <stdexcept>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "distributed.hpp"
#include "ops/gemm.hpp"
#include "loader/rust_loader_bridge.hpp"
#include "loader/rust_storage_executor.hpp"
#include "model/weight_artifact_cache.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver {

namespace {

// llama-like and assumes the standard name layout — gemma/mixtral/MoE need
// their own plan (per-expert weights, dual-norm, etc.).
bool supports_tp(const std::string& mt) {
    return mt == "qwen3"
        || mt == "qwen2"
        || mt == "llama" || mt == "llama3"
        || mt == "mistral" || mt == "mistral3" || mt == "ministral3"
        || mt == "phi3"
        || mt == "olmo2" || mt == "olmo3"
        || mt == "gemma2"
        || mt == "gemma3" || mt == "gemma3_text"
        || mt == "gemma4" || mt == "gemma4_text"
        || mt == "mixtral"
        || mt == "gemma3n" || mt == "gemma3n_text"
        || mt == "gpt_oss"
        || mt == "qwen3_5" || mt == "qwen3_5_text"
        || mt == "qwen3_5_moe" || mt == "qwen3_5_moe_text"
        || mt == "qwen3_moe"
        || mt == "nemotron_h"
        || mt == "kimi_k2"
        || mt == "deepseek_v2" || mt == "deepseek_v3" || mt == "deepseek_v4"
        || mt == "glm_moe_dsa";
}

// True for any MoE model whose forward path lives in qwen3_5_moe_forward.
// All members share an all-MoE MLP layout (no dense `intermediate_size`),
// so the engine's TP divisibility checks on `intermediate_size` should
// be skipped for them.
bool model_type_is_qwen3_5_moe(const std::string& mt) {
    return mt == "qwen3_5_moe" || mt == "qwen3_5_moe_text"
        || mt == "qwen3_moe";
}

struct LoadMemorySampler {
    LoadExecutionStats* stats = nullptr;

    static void sample(void* context) noexcept {
        auto* self = static_cast<LoadMemorySampler*>(context);
        if (self == nullptr || self->stats == nullptr) return;
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) return;
        auto& s = *self->stats;
        if (s.cuda_memory_samples == 0) {
            s.cuda_free_before_bytes = free_bytes;
            s.cuda_min_free_bytes = free_bytes;
            s.cuda_total_bytes = total_bytes;
        } else {
            s.cuda_min_free_bytes = std::min<std::uint64_t>(
                s.cuda_min_free_bytes, free_bytes);
        }
        s.cuda_total_bytes = total_bytes;
        s.cuda_memory_samples += 1;
    }
};

class ScopedDeviceTensorMemoryCallback {
public:
    explicit ScopedDeviceTensorMemoryCallback(LoadMemorySampler* sampler)
        : enabled_(sampler != nullptr)
    {
        if (enabled_) {
            set_device_tensor_memory_callback(&LoadMemorySampler::sample, sampler);
        }
    }

    ScopedDeviceTensorMemoryCallback(const ScopedDeviceTensorMemoryCallback&) = delete;
    ScopedDeviceTensorMemoryCallback& operator=(
        const ScopedDeviceTensorMemoryCallback&) = delete;

    ~ScopedDeviceTensorMemoryCallback() {
        if (enabled_) {
            set_device_tensor_memory_callback(nullptr, nullptr);
        }
    }

private:
    bool enabled_ = false;
};

}  // namespace

LoadedModel LoadedModel::load(
    const Config& boot_cfg,
    NcclComm* tp_comm,
    std::span<const std::uint8_t> program_bytes,
    std::uint64_t compiler_version) {
    (void)tp_comm;

    if (boot_cfg.model.snapshot_dir.empty()) {
        throw std::runtime_error(
            "engine: model.snapshot_dir is empty — pass it in dev.toml or "
            "let the wrapper resolve it via pie_driver.hf_utils");
    }

    LoadedModel e;
    e.boot_ = boot_cfg;
    const bool verbose = boot_cfg.runtime.verbose;
    const auto load_start = std::chrono::steady_clock::now();
    auto log_stage = [&](const char* stage) {
        if (!verbose) return;
        const auto now = std::chrono::steady_clock::now();
        const double ms =
            std::chrono::duration<double, std::milli>(now - load_start).count();
        std::cerr << "[pie-driver-cuda] load stage rank="
                  << boot_cfg.distributed.tp_rank << " +" << static_cast<int>(ms)
                  << "ms: " << stage << "\n";
    };

    const std::filesystem::path snapshot{boot_cfg.model.snapshot_dir};
    log_stage("parse hf config begin");
    e.hf_ = parse_hf_config(snapshot / "config.json");
    log_stage("parse hf config done");

    // Bind to the requested CUDA device before we allocate anything.
    int dev_id = 0;
    {
        const auto& d = boot_cfg.model.device;
        const auto colon = d.find(':');
        if (colon != std::string::npos) {
            dev_id = std::stoi(d.substr(colon + 1));
        }
    }
    CUDA_CHECK(cudaSetDevice(dev_id));

    // Compute capability — used by the runtime-quant skip (sm<89) and
    // by the eager FP8→bf16 dequant pass. cuBLASLt's native FP8 GEMM
    // requires sm89+ (Ada/Hopper); on Ampere (sm80) the dispatcher
    // falls back to dequant-then-bf16-GEMM.
    cudaDeviceProp dev_prop{};
    CUDA_CHECK(cudaGetDeviceProperties(&dev_prop, dev_id));
    const bool fp8_native = (dev_prop.major > 8) ||
                            (dev_prop.major == 8 && dev_prop.minor >= 9);
#ifdef PIE_CUDA_HAS_MARLIN
    const bool gptq_marlin_int4 = true;
    // Native MXFP4 expert execution requires a Blackwell-class FP4 path.
    // Older GPUs keep packed MXFP4 resident but use routed BF16 dequant
    // scratch for the selected experts.
    const bool mxfp4_native_gemm = dev_prop.major >= 10;
#else
    const bool gptq_marlin_int4 = false;
    const bool mxfp4_native_gemm = false;
#endif
    BackendTarget backend_target{
        .device_major = dev_prop.major,
        .device_minor = dev_prop.minor,
        .fp8_native = fp8_native,
        .gptq_marlin_int4 = gptq_marlin_int4,
        .mxfp4_native_gemm = mxfp4_native_gemm,
    };
    RustStorageProgram storage_program =
        deserialize_rust_storage_program(program_bytes, compiler_version);
    switch (storage_program.mxfp4_moe()) {
    case pie_weight_loader::PieLoaderMxfp4MoePolicy::NativeGemm:
        if (!backend_target.mxfp4_native_gemm) {
            throw std::runtime_error(
                "engine: StorageProgram requires native MXFP4 MoE, but this "
                "device/build does not provide it");
        }
        backend_target.mxfp4_moe = Mxfp4MoeLowering::NativeGemm;
        break;
    case pie_weight_loader::PieLoaderMxfp4MoePolicy::EagerBf16:
        backend_target.mxfp4_moe = Mxfp4MoeLowering::Bf16Dequant;
        break;
    case pie_weight_loader::PieLoaderMxfp4MoePolicy::RoutedDecode:
        backend_target.mxfp4_moe = Mxfp4MoeLowering::RoutedDequant;
        break;
    }
    e.mxfp4_moe_lowering_ = backend_target.mxfp4_moe;

    log_stage("open safetensors begin");
    auto loader = CheckpointSource::open(snapshot);
    log_stage("open safetensors done");

    const int tp_size = boot_cfg.distributed.tp_size;
    if (tp_size > 1 && !supports_tp(e.hf_.model_type)) {
        throw std::runtime_error(
            "engine: tensor-parallelism not yet supported for model_type='" +
            e.hf_.model_type +
            "'. Currently TP-enabled: qwen2/qwen3, llama/llama3, mistral/"
            "mistral3, phi3, olmo2/olmo3, gemma2/gemma3.");
    }
    // Sharding along the head dim requires the head/expert counts and
    // intermediate widths to all divide cleanly by tp_size. Reject early
    // with a useful message instead of failing inside `load_to_device_sharded`
    // (which sees only one tensor at a time and can't explain why).
    if (tp_size > 1) {
        const auto& hf = e.hf_;
        auto require_divisible = [&](int v, const char* name) {
            if (v <= 0 || v % tp_size != 0) {
                throw std::runtime_error(
                    std::string("engine: ") + name + "=" + std::to_string(v) +
                    " is not divisible by tp_size=" + std::to_string(tp_size) +
                    ". Sharding the head/intermediate axis requires this; "
                    "use a smaller tp_size or run single-GPU.");
            }
        };
        require_divisible(hf.num_attention_heads, "num_attention_heads");
        // V4 has num_key_value_heads=1 (MQA) — the single KV head is
        // replicated, not sharded. Skip the divisibility check.
        if (hf.model_type != "deepseek_v4") {
            require_divisible(hf.num_key_value_heads, "num_key_value_heads");
        }
        // Qwen3.5-MoE / Qwen3-MoE have no dense `intermediate_size`; the
        // MLP lives entirely in `moe_intermediate_size` (+ `shared_expert_
        // intermediate_size` for the 3.5/3.6 family — Qwen3-MoE has no
        // shared expert).
        const bool is_kimi_k2 = hf.model_type == "kimi_k2"
            || hf.model_type == "deepseek_v2" || hf.model_type == "deepseek_v3"
            || hf.model_type == "glm_moe_dsa";
        const bool is_dsv4 = hf.model_type == "deepseek_v4";
        const bool is_q35_moe = model_type_is_qwen3_5_moe(hf.model_type);
        const bool is_nemotron_h = hf.model_type == "nemotron_h";
        if (!is_q35_moe && !is_kimi_k2 && !is_dsv4) {
            require_divisible(hf.intermediate_size, "intermediate_size");
        }
        if (is_kimi_k2) {
            require_divisible(hf.q_lora_rank, "q_lora_rank");
            require_divisible(hf.kv_lora_rank, "kv_lora_rank");
            require_divisible(hf.moe_intermediate_size, "moe_intermediate_size");
            if (hf.shared_expert_intermediate_size > 0) {
                require_divisible(hf.shared_expert_intermediate_size,
                                  "shared_expert_intermediate_size");
            }
        }
        if (is_dsv4) {
            require_divisible(hf.q_lora_rank, "q_lora_rank");
            if (hf.dsv4_o_lora_rank > 0) {
                require_divisible(hf.dsv4_o_lora_rank, "o_lora_rank");
            }
            require_divisible(hf.moe_intermediate_size, "moe_intermediate_size");
        }
        // Qwen3.5 / 3.6-MoE: linear-attention head counts must shard too.
        // Qwen3-MoE has no linear-attn layers, so this check is skipped.
        const bool has_linear_attn =
            (hf.model_type == "qwen3_5" || hf.model_type == "qwen3_5_text" ||
             hf.model_type == "qwen3_5_moe" ||
             hf.model_type == "qwen3_5_moe_text");
        if (has_linear_attn) {
            require_divisible(hf.linear_num_key_heads, "linear_num_key_heads");
            require_divisible(hf.linear_num_value_heads, "linear_num_value_heads");
        }
        if (is_q35_moe || is_nemotron_h) {
            require_divisible(hf.moe_intermediate_size, "moe_intermediate_size");
            // shared_expert_intermediate_size is 0 for Qwen3-MoE (no shared
            // expert); only enforce divisibility when the family actually
            // has one.
            if (hf.shared_expert_intermediate_size > 0) {
                require_divisible(hf.shared_expert_intermediate_size,
                                  "shared_expert_intermediate_size");
            }
        }
    }

    if (e.hf_.kv_cache_scheme_present) {
        std::cerr << "[pie-driver-cuda] WARNING: ckpt's "
                  << "quantization_config.kv_cache_scheme is non-null but "
                  << "runtime-loaded checkpoint KV scales are unsupported. "
                  << "Configured kv_cache_dtype='" << boot_cfg.batching.kv_cache_dtype
                  << "' will use default or online dynamic scales, so generation "
                  << "may drift from the calibrated reference.\n";
    }

    const auto t0 = std::chrono::steady_clock::now();

    log_stage("decode storage program begin");
    RustLoaderCompileResult rust_plan =
        load_rust_storage_program(
            e.hf_,
            std::move(storage_program),
            program_bytes,
            compiler_version);
    log_stage("decode storage program done");
    WeightStoreBuilder(e.weights_).reserve(rust_plan.runtime_tensor_count);
    const auto rust_view = rust_plan.program.view();
    if (const char* dump_path =
            std::getenv("PIE_CUDA_RUST_LAYOUT_PLAN_DUMP");
        dump_path && dump_path[0] != '\0') {
        std::ofstream out(dump_path);
        if (!out) {
            throw std::runtime_error(
                "engine: failed to open PIE_CUDA_RUST_LAYOUT_PLAN_DUMP "
                "path: " + std::string(dump_path));
        }
        out << dump_rust_storage_program_json(
            rust_view,
            rust_plan.source_tensor_count,
            rust_plan.covered_contract_count,
            rust_plan.runtime_tensor_count);
    }
    if (verbose) {
        std::cerr
            << "[pie-driver-cuda] layout compiler: rust RuntimeABI -> "
               "algebra -> storage program\n";
        std::cerr << "[pie-driver-cuda] rust loader compiler: "
                  << describe_rust_storage_program(
                         rust_view,
                         rust_plan.source_tensor_count,
                         rust_plan.covered_contract_count,
                         rust_plan.runtime_tensor_count)
                  << "\n";
    }
    if (rust_plan.covered_contract_count != rust_plan.runtime_tensor_count) {
        throw std::runtime_error(
            "engine: Rust loader did not cover the full RuntimeABI; covered " +
            std::to_string(rust_plan.covered_contract_count) + "/" +
            std::to_string(rust_plan.runtime_tensor_count) +
            " runtime tensors. Add schema/RuntimeABI coverage before enabling "
            "this model.");
    }

    // Materialized-weight artifact cache (WEIGHT_LOADER_TODO.md A3.1). The
    // materialized weights are a deterministic function of rust_plan.cache_key,
    // so on a hit we reload them straight into device memory and skip the
    // executor pass below. The compile above is cheap (~tens of ms) and still
    // runs every boot, validating the key + full ABI coverage.
    LoadExecutionStats materialized;
    const auto weight_cache_dir = weight_artifact_cache_dir();
    bool weight_cache_hit = false;
    if (!weight_cache_dir.empty()) {
        try {
            WeightStoreBuilder cache_builder(e.weights_);
            weight_cache_hit = read_weight_artifact_cache(
                cache_builder, rust_plan.cache_key, weight_cache_dir);
        } catch (const std::exception& ex) {
            std::cerr << "[pie-driver-cuda] weight cache: reload failed ("
                      << ex.what() << "); falling back to materialize\n";
            weight_cache_hit = false;
        }
        log_stage(weight_cache_hit
                      ? "weight artifact cache hit (skipped materialize)"
                      : "weight artifact cache miss");
    }

    if (!weight_cache_hit) {
        // A miss can leave the store partially populated — a checksum mismatch
        // is only detected after the owned blobs are inserted, and a throwing
        // reload aborts mid-restore. WeightStore::insert rejects duplicate names,
        // so materialize would abort on the leftovers (e.g. the storage arena).
        // Reset to a clean slate; this also frees any stranded device tensors
        // (DeviceTensor RAII). A no-op when the restore left nothing.
        e.weights_ = WeightStore{};
        WeightStoreBuilder rust_builder(e.weights_);
        RustStorageProgramExecutor rust_executor(
            loader,
            rust_builder,
            std::move(rust_plan.quant_attachments));
        log_stage("materialize storage program begin");
        LoadExecutionStats load_memory_stats;
        const bool sample_load_memory =
            verbose || std::getenv("PIE_CUDA_PROFILE_LOAD_MEMORY") != nullptr;
        LoadMemorySampler load_memory_sampler{.stats = &load_memory_stats};
        if (sample_load_memory) {
            LoadMemorySampler::sample(&load_memory_sampler);
        }
        {
            ScopedDeviceTensorMemoryCallback callback(
                sample_load_memory ? &load_memory_sampler : nullptr);
            materialized = rust_executor.execute(rust_view);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        log_stage("materialize storage program done");
        if (sample_load_memory) {
            std::size_t free_after = 0;
            std::size_t total_after = 0;
            CUDA_CHECK(cudaMemGetInfo(&free_after, &total_after));
            load_memory_stats.cuda_free_after_bytes = free_after;
            load_memory_stats.cuda_total_bytes = total_after;
            if (load_memory_stats.cuda_memory_samples > 0 &&
                load_memory_stats.cuda_free_before_bytes >=
                    load_memory_stats.cuda_min_free_bytes) {
                load_memory_stats.cuda_actual_peak_delta_bytes =
                    load_memory_stats.cuda_free_before_bytes -
                    load_memory_stats.cuda_min_free_bytes;
            }
            materialized.cuda_total_bytes = load_memory_stats.cuda_total_bytes;
            materialized.cuda_free_before_bytes =
                load_memory_stats.cuda_free_before_bytes;
            materialized.cuda_min_free_bytes =
                load_memory_stats.cuda_min_free_bytes;
            materialized.cuda_free_after_bytes =
                load_memory_stats.cuda_free_after_bytes;
            materialized.cuda_actual_peak_delta_bytes =
                load_memory_stats.cuda_actual_peak_delta_bytes;
            materialized.cuda_memory_samples =
                load_memory_stats.cuda_memory_samples;
        }

        if (!weight_cache_dir.empty()) {
            log_stage("weight artifact cache write begin");
            bool wrote = false;
            try {
                wrote = write_weight_artifact_cache(
                    e.weights_, rust_plan.cache_key, weight_cache_dir);
            } catch (const std::exception& ex) {
                std::cerr << "[pie-driver-cuda] weight cache: write failed ("
                          << ex.what() << ")\n";
            }
            log_stage(wrote ? "weight artifact cache write done"
                            : "weight artifact cache write skipped");
        }
    }

    if (verbose && materialized.runtime_quantized_weights > 0) {
        const double mib_before =
            static_cast<double>(materialized.runtime_quant_bytes_before) /
            (1024.0 * 1024.0);
        const double mib_after =
            static_cast<double>(materialized.runtime_quant_bytes_after) /
            (1024.0 * 1024.0);
        std::cerr << "[pie-driver-cuda] storage program quantised "
                  << materialized.runtime_quantized_weights
                  << " projections: "
                  << static_cast<std::uint64_t>(mib_before) << " -> "
                  << static_cast<std::uint64_t>(mib_after) << " MiB ("
                  << static_cast<int>(
                         100.0 * mib_after / std::max(mib_before, 1.0))
                  << "% of original)\n";
    }
    if (verbose && materialized.axis_concat_groups > 0) {
        std::cerr << "[pie-driver-cuda] storage loader: "
                  << materialized.axis_concat_groups << " AxisConcat groups"
                  << " (raw projection weights exposed as non-owning views)\n";
    }
    if (verbose && materialized.cuda_memory_samples > 0) {
        const auto to_mib = [](std::uint64_t bytes) {
            return bytes / (1024ull * 1024ull);
        };
        std::cerr << "[pie-driver-cuda] load memory high-water: planned_peak~"
                  << to_mib(materialized.planned_storage_peak_bytes)
                  << " MiB, planned_temp<="
                  << to_mib(materialized.planned_storage_temp_bytes)
                  << " MiB, actual_cuda_delta~"
                  << to_mib(materialized.cuda_actual_peak_delta_bytes)
                  << " MiB, free "
                  << to_mib(materialized.cuda_free_before_bytes)
                  << " -> min "
                  << to_mib(materialized.cuda_min_free_bytes)
                  << " -> "
                  << to_mib(materialized.cuda_free_after_bytes)
                  << " MiB across "
                  << materialized.cuda_memory_samples << " samples\n";
    }
    if (const char* profile = std::getenv("PIE_WEIGHT_LOADER_PROFILE");
        profile != nullptr && profile[0] != '\0' && profile[0] != '0') {
        const auto to_mib = [](std::uint64_t bytes) {
            return bytes / (1024ull * 1024ull);
        };
        std::cerr << "[pie-driver-cuda] weight loader profile: h2d_copies="
                  << materialized.h2d_copy_count
                  << " bulk_copies=" << materialized.h2d_bulk_copy_count
                  << " pinned_copies="
                  << materialized.h2d_pinned_copy_count
                  << " slab_scatter="
                  << materialized.slab_scatter_count
                  << " slab_placements="
                  << materialized.slab_scatter_placements
                  << " h2d_bytes=" << to_mib(materialized.h2d_copy_bytes)
                  << " MiB bulk_bytes="
                  << to_mib(materialized.h2d_bulk_copy_bytes)
                  << " MiB pinned_bytes="
                  << to_mib(materialized.h2d_pinned_copy_bytes)
                  << " MiB slab_source_bytes="
                  << to_mib(materialized.slab_scatter_source_bytes)
                  << " MiB slab_payload_bytes="
                  << to_mib(materialized.slab_scatter_payload_bytes)
                  << " MiB copy_flushes="
                  << materialized.copy_stream_flushes
                  << " batch_calls="
                  << materialized.h2d_batch_calls
                  << " max_pending="
                  << materialized.max_pending_copies_seen << "\n";
        std::cerr << "[pie-driver-cuda] weight loader phases: alloc="
                  << static_cast<int>(materialized.phase_alloc_ms)
                  << "ms transfer=" << static_cast<int>(materialized.phase_transfer_ms)
                  << "ms (pinned_alloc="
                  << static_cast<int>(materialized.phase_pinned_alloc_ms)
                  << "ms) transform="
                  << static_cast<int>(materialized.phase_transform_ms) << "ms\n";
    }

    e.weights_.validate_quant_metadata();
    const std::uint64_t loaded_bytes = e.weights_.total_bytes();

    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double mib = static_cast<double>(loaded_bytes) / (1024.0 * 1024.0);

    if (verbose) {
        std::cerr << "[pie-driver-cuda] loaded " << e.weights_.size() << " tensors ("
                  << static_cast<std::uint64_t>(mib) << " MiB on this rank, "
                  << "tp=" << tp_size << ") in " << static_cast<int>(ms)
                  << " ms; arch=" << e.hf_.arch_name << " (" << e.hf_.model_type << ")\n";
    }

    return e;
}

LoadedModelCapabilities LoadedModel::capabilities() const {
    LoadedModelCapabilities c;
    c.total_pages = 0;  // populated in M1.2.2 once kv_cache lands
    c.kv_page_size = static_cast<int>(boot_.batching.kv_page_size);
    c.swap_pool_size = 0;
    // The runtime's `model::instruct::create` dispatches on the
    // PIE-arch key ("llama3", "gemma3", …) not HF's `architectures[0]`
    // ("LlamaForCausalLM") nor the raw HF model_type ("llama",
    // "gemma3_text"). The Python `pie_driver` normalises via the
    // `HF_TO_PIE_ARCH` table; we mirror that table here so the
    // runtime gets the same key from both backends.
    auto normalise_arch = [](const std::string& mt) -> std::string {
        if (mt == "llama")          return "llama3";
        if (mt == "gemma3_text")    return "gemma3";
        if (mt == "gemma4_text")    return "gemma4";
        if (mt == "ministral3")     return "mistral3";
        if (mt == "qwen3_vl_text")  return "qwen3_vl";
        return mt;  // qwen2 / qwen3 / gemma2 / olmo3 / phi3 / mistral3 / mixtral
    };
    c.arch_name = hf_.model_type.empty()
        ? hf_.arch_name
        : normalise_arch(hf_.model_type);
    c.vocab_size = hf_.vocab_size;
    c.max_model_len = hf_.max_position_embeddings;
    c.activation_dtype = boot_.model.dtype;
    c.snapshot_dir = boot_.model.snapshot_dir;
    return c;
}

std::uint64_t LoadedModel::total_weight_bytes() const noexcept {
    return weights_.total_bytes();
}

const DeviceTensor& LoadedModel::get(const std::string& name) const {
    return weights_.get(name);
}

std::size_t LoadedModel::erase_runtime_weight(const std::string& name) {
    return weights_.erase_runtime_weight(name);
}

std::optional<QuantMeta> LoadedModel::quant_meta(const std::string& name) const {
    return weights_.quant_meta(name);
}

ops::RuntimeQuantScratchSpec runtime_quant_scratch_spec(const LoadedModel& engine,
                                                       std::size_t max_tokens) {
    ops::RuntimeQuantScratchSpec spec;
    spec.max_tokens = max_tokens;

    const auto& store = engine.weight_store();
    for (const auto& item : store.quant_meta_map()) {
        const auto& name = item.first;
        auto it = store.find(name);
        if (it == store.end()) continue;
        const auto& tensor = it->second.tensor;
        if (tensor.shape().size() != 2) continue;

        if (tensor.dtype() == DType::FP8_E4M3) {
            spec.has_fp8 = true;
        } else if (tensor.dtype() == DType::INT8) {
            spec.has_int8 = true;
        } else {
            continue;
        }

        spec.max_weight_rows = std::max<std::size_t>(
            spec.max_weight_rows,
            static_cast<std::size_t>(std::max<std::int64_t>(
                0, tensor.shape()[0])));
        spec.max_weight_cols = std::max<std::size_t>(
            spec.max_weight_cols,
            static_cast<std::size_t>(std::max<std::int64_t>(
                0, tensor.shape()[1])));
    }

    return spec;
}

}  // namespace pie_cuda_driver
