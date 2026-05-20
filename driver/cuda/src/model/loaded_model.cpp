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
#include "loader/rust_loader_bridge.hpp"
#include "loader/rust_storage_executor.hpp"

namespace pie_cuda_driver {

namespace {

// Whitelist of model_types we currently support TP for. The shard plan is
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
        || mt == "qwen3_moe";
}

// True for any MoE model whose forward path lives in qwen3_5_moe_forward.
// All members share an all-MoE MLP layout (no dense `intermediate_size`),
// so the engine's TP divisibility checks on `intermediate_size` should
// be skipped for them.
bool is_qwen3_5_moe_arch(const std::string& mt) {
    return mt == "qwen3_5_moe" || mt == "qwen3_5_moe_text"
        || mt == "qwen3_moe";
}

Mxfp4MoeLowering select_mxfp4_moe_lowering(
    const ModelConfig& model_cfg,
    const BackendTarget& target)
{
    const std::string& policy = model_cfg.mxfp4_moe;
    if (policy.empty() || policy == "auto") {
        return target.mxfp4_native_gemm
            ? Mxfp4MoeLowering::NativeGemm
            : Mxfp4MoeLowering::RoutedDequant;
    }
    if (policy == "routed_dequant" || policy == "packed") {
        return Mxfp4MoeLowering::RoutedDequant;
    }
    if (policy == "bf16" || policy == "dequant" ||
        policy == "eager_bf16") {
        return Mxfp4MoeLowering::Bf16Dequant;
    }
    if (policy == "native") {
        if (!target.mxfp4_native_gemm) {
            throw std::runtime_error(
                "engine: model.mxfp4_moe='native' requested a true MXFP4 "
                "MoE GEMM backend, but this build has no registered native "
                "MXFP4 expert GEMM kernels");
        }
        return Mxfp4MoeLowering::NativeGemm;
    }
    throw std::runtime_error(
        "engine: model.mxfp4_moe must be one of "
        "{auto,routed_dequant,packed,bf16,dequant,eager_bf16,native}");
}

}  // namespace

LoadedModel LoadedModel::load(const Config& boot_cfg, NcclComm* tp_comm) {
    (void)tp_comm;

    if (boot_cfg.model.snapshot_dir.empty()) {
        throw std::runtime_error(
            "engine: model.snapshot_dir is empty — pass it in dev.toml or "
            "let the wrapper resolve it via pie_driver.hf_utils");
    }

    LoadedModel e;
    e.boot_ = boot_cfg;
    const bool verbose = boot_cfg.runtime.verbose;

    const std::filesystem::path snapshot{boot_cfg.model.snapshot_dir};
    e.hf_ = parse_hf_config(snapshot / "config.json");

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
    if (boot_cfg.model.runtime_quant == "fp8" && !fp8_native) {
        std::cerr << "[pie-driver-cuda] runtime_quant=fp8 skipped: "
                  << "sm" << dev_prop.major << dev_prop.minor
                  << " has no native FP8 GEMM. Weights stay bf16 "
                  << "(use runtime_quant=int8 or marlin Int4 / GPTQ "
                  << "for memory + perf wins on this generation).\n";
    }
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
    backend_target.mxfp4_moe =
        select_mxfp4_moe_lowering(boot_cfg.model, backend_target);
    e.mxfp4_moe_lowering_ = backend_target.mxfp4_moe;

    auto loader = SafetensorsCheckpointSource::open(snapshot);

    const int tp_size = boot_cfg.distributed.tp_size;
    const int tp_rank = boot_cfg.distributed.tp_rank;
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
        require_divisible(hf.num_key_value_heads, "num_key_value_heads");
        // Qwen3.5-MoE / Qwen3-MoE have no dense `intermediate_size`; the
        // MLP lives entirely in `moe_intermediate_size` (+ `shared_expert_
        // intermediate_size` for the 3.5/3.6 family — Qwen3-MoE has no
        // shared expert).
        const bool is_q35_moe = is_qwen3_5_moe_arch(hf.model_type);
        if (!is_q35_moe) {
            require_divisible(hf.intermediate_size, "intermediate_size");
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
        if (is_q35_moe) {
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

    WeightStoreBuilder(e.weights_).reserve(loader.num_tensors());

    std::string runtime_quant = boot_cfg.model.runtime_quant;
    if (runtime_quant == "fp8" && !fp8_native) {
        runtime_quant.clear();
    }
    RustLoaderCompileResult rust_plan =
        compile_rust_loader_plan_from_metadata(
            e.hf_, loader, runtime_quant, tp_rank, tp_size,
            64ull * 1024ull * 1024ull,
            /*preferred_alignment=*/256,
            backend_target);
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

    WeightStoreBuilder rust_builder(e.weights_);
    RustStorageProgramExecutor rust_executor(
        loader,
        rust_builder,
        std::move(rust_plan.source_tensor_names),
        std::move(rust_plan.quant_attachments));
    LoadExecutionStats materialized = rust_executor.execute(rust_view);
    CUDA_CHECK(cudaDeviceSynchronize());

    if (verbose && materialized.runtime_quantized_weights > 0) {
        const double mib_before =
            static_cast<double>(materialized.runtime_quant_bytes_before) /
            (1024.0 * 1024.0);
        const double mib_after =
            static_cast<double>(materialized.runtime_quant_bytes_after) /
            (1024.0 * 1024.0);
        std::cerr << "[pie-driver-cuda] runtime_quant="
                  << boot_cfg.model.runtime_quant << " quantised "
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
        if (mt == "llama")        return "llama3";
        if (mt == "gemma3_text")  return "gemma3";
        if (mt == "gemma4_text")  return "gemma4";
        if (mt == "ministral3")   return "mistral3";
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

std::optional<QuantMeta> LoadedModel::quant_meta(const std::string& name) const {
    return weights_.quant_meta(name);
}

}  // namespace pie_cuda_driver
