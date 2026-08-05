#include "model/loaded_model.hpp"

#include <algorithm>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <sstream>
#include <stdexcept>

#include <cuda_runtime.h>

#include "cuda_check.hpp"
#include "distributed.hpp"
#include "ops/gemm.hpp"
#include "loader/rust_author.hpp"
#include "loader/load_plan_executor.hpp"
#include "model/descriptor.hpp"
#include "model/registry.hpp"
#include "model/weight_artifact_cache.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver {

namespace {

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

/// How much device memory the group slab may take.
///
/// A configured `expert_cache_gb` is taken at its word -- someone who names a
/// number is answering a question about their card that this cannot. The
/// interesting case is zero, and the rule there is: ask for the whole group,
/// but never take more than half of what is free.
///
/// The first half of that is what makes streaming safe to leave on. If the
/// group fits, the slab holds all of it, no page-in ever misses after the
/// first sweep, and the run is exactly the resident run that the stacked
/// contract would have produced -- so turning streaming on costs nothing on a
/// card that did not need it. The second half is the fallback that makes it
/// useful on a card that did: the experts and the KV cache are the two things
/// competing for what is left after the resident weights, and with no way to
/// know how long the sequences will be, splitting it is the honest default.
std::uint64_t group_cache_budget(
    double configured_gb,
    pie_loader::PieLoaderGroupSlice groups,
    bool verbose)
{
    std::uint64_t wanted = 0;
    for (std::size_t i = 0; i < groups.len; ++i) {
        const auto& g = groups.ptr[i];
        if (g.plan == nullptr) continue;
        wanted += g.plan->memory.persistent_bytes *
                  static_cast<std::uint64_t>(g.arity);
    }
    if (configured_gb > 0.0) {
        return static_cast<std::uint64_t>(configured_gb * 1024.0 * 1024.0 * 1024.0);
    }
    std::size_t free_bytes = 0;
    std::size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    const std::uint64_t half_free = static_cast<std::uint64_t>(free_bytes) / 2;
    const std::uint64_t budget = std::min(wanted, half_free);
    if (verbose) {
        std::cerr << "[pie-driver-cuda] group slab: want "
                  << (wanted / (1024ull * 1024ull)) << " MiB, "
                  << (free_bytes / (1024ull * 1024ull)) << " MiB free, taking "
                  << (budget / (1024ull * 1024ull)) << " MiB\n";
    }
    return budget;
}

}  // namespace

LoadedModel LoadedModel::load(
    const Config& boot_cfg,
    NcclComm* tp_comm,
    std::string_view runtime_quant,
    model::Mxfp4MoeRequest mxfp4_moe,
    model::Component component) {
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
    // The model config arrives already normalized, whatever the worker was
    // pointed at: an artifact carries a `pie.model/1` descriptor, and a plain
    // HF snapshot is normalized into one by `worker/src/weights.rs` before any
    // driver is created. See model/descriptor.hpp.
    //
    // There used to be an `else` here — `parse_hf_config`, 855 lines and 25
    // `model_type` conditionals, reading `config.json` a second time in a
    // second language. It is gone, so this is an error rather than a fallback:
    // a driver with no descriptor cannot answer what the model is made of, and
    // guessing is what produced two answers that had to agree by coincidence.
    if (boot_cfg.model.descriptor.empty()) {
        throw std::runtime_error(
            "engine: model.descriptor is empty. Every boot is handed a "
            "pie.model/1 descriptor beside its startup TOML; a hand-written "
            "config must point `[model] descriptor` at one "
            "(`pie model import` writes an artifact that carries it, and "
            "`cargo run -p pie-model-config --bin descriptor config.json` "
            "compiles one from a snapshot).");
    }
    log_stage("read model descriptor begin");
    // Kept for the whole load: the compile request borrows this document
    // rather than a struct distilled from it, so it has to outlive
    // `prepare_load_plan_rust_author` below.
    const std::string descriptor_json = [&] {
        std::ifstream in(boot_cfg.model.descriptor);
        if (!in) {
            throw std::runtime_error("cannot open model descriptor: " +
                                     boot_cfg.model.descriptor);
        }
        std::ostringstream buffer;
        buffer << in.rdbuf();
        return buffer.str();
    }();
    e.hf_ = parse_pie_model_descriptor(descriptor_json);
    log_stage("read model descriptor done");

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
    // Native MXFP4 expert execution requires a Blackwell-class FP4 path.
    // Older GPUs keep packed MXFP4 resident but use routed BF16 dequant
    // scratch for the selected experts.
    const bool mxfp4_native_gemm = dev_prop.major >= 10;
#else
    const bool mxfp4_native_gemm = false;
#endif

    // Compile the plan for *this* device. The driver states what the device can
    // do and the loader answers with a plan that stays inside it, so there is
    // nothing to re-validate afterwards: the mxfp4 lowering below is read back
    // as the loader's decision, not checked against a second opinion
    // (`loader/architecture.md` §9).
    log_stage("compile LoadPlan begin");
    const pie_loader::DeviceTarget device_target = [&] {
        auto target = cuda_device_target();
        target.tp_rank = static_cast<std::uint32_t>(boot_cfg.distributed.tp_rank);
        target.tp_size = static_cast<std::uint32_t>(boot_cfg.distributed.tp_size);
        target.native_mxfp4_moe = mxfp4_native_gemm;
        return target;
    }();

    // Read the tensor table once. The contract is written against it and the
    // compile consumes the same handle, so the two cannot be about different
    // parses of the same directory.
    std::string open_error;
    pie_loader::Checkpoint checkpoint =
        pie_loader::Checkpoint::open(boot_cfg.model.snapshot_dir, &open_error);
    if (!checkpoint) {
        throw std::runtime_error("engine: failed to read checkpoint: " + open_error);
    }

    // The row is the same one `Context` will later call `bind` on; whether
    // the model is supported is answered here, before anything is loaded.
    // What to *build* is authored on the loader's side from the request
    // below, which is why a family this driver has never heard of fails
    // here by name rather than deep in a load.
    const model::ArchEntry* arch = model::find_arch_entry(e.hf_.model_type);
    if (arch == nullptr) {
        throw std::runtime_error("engine: unsupported model_type '" + e.hf_.model_type +
                                 "'; no row in the arch table declares what it binds");
    }
    const model::ModelFacts facts{
        .model_type = e.hf_.model_type,
        .quant_method = e.hf_.quant_method,
        .num_hidden_layers = static_cast<std::uint32_t>(std::max(0, e.hf_.num_hidden_layers)),
        .num_experts = static_cast<std::uint32_t>(std::max(0, e.hf_.num_experts)),
        .head_dim = static_cast<std::uint32_t>(std::max(0, e.hf_.head_dim)),
        .mamba_groups = static_cast<std::uint32_t>(std::max(0, e.hf_.mamba_n_groups)),
    };
    // One author, on the far side of the request boundary: facts and policy
    // in, plan out, the contract never crossing the ABI
    // (`plan/model-in-rust.md` §2). The C++ author this replaced was proven
    // byte-equal by the §8-3 differential — 17 synthetic cases, ten real
    // checkpoints, and a dual boot on this hardware — before it was
    // deleted.
    model::Mxfp4MoePolicy mxfp4_moe_policy = model::Mxfp4MoePolicy::RoutedDecode;
    LoadPlanResult planned_load = prepare_load_plan_rust_author(
        checkpoint, descriptor_json, device_target,
        model::resolve_runtime_quant(runtime_quant, fp8_native),
        mxfp4_moe, component, boot_cfg.model.stream_routed_experts,
        &mxfp4_moe_policy);

    // The policy is the *author's* answer, not the plan's — a family may
    // override the device rule — and it comes back through the entry's
    // out-parameter, so there is one answer rather than two that have to be
    // kept agreeing.
    e.mxfp4_moe_policy_ = mxfp4_moe_policy;
    log_stage("compile LoadPlan done");

    log_stage("open checkpoint source begin");
    // On the heap because streaming outlives this function: a group is paged
    // in by reading the same files the resident load read, so the handles have
    // to survive it, and the cache holds a reference to them.
    auto source = std::make_unique<pie_loader::CheckpointSource>(
        planned_load.plan.view());
    pie_loader::CheckpointSource& loader = *source;
    log_stage("open checkpoint source done");

    // What used to sit here: `supports_tp()` — a list of twenty-odd model_type
    // strings — followed by eighty lines of per-family divisibility rules read
    // off `config.json`. Every one of them restated something the loader had
    // already decided by the time this ran: it is the loader that partitions a
    // tensor, so it is the loader that discovers an axis tp_size does not
    // divide, and it names the tensor when it says so (`frontend.rs`,
    // `arch.rs::local_range`). A family missing from the list got no check at
    // all; a family in it got two.

    if (e.hf_.kv_cache_scheme_present) {
        std::cerr << "[pie-driver-cuda] WARNING: ckpt's "
                  << "quantization_config.kv_cache_scheme is non-null but "
                  << "runtime-loaded checkpoint KV scales are unsupported. "
                  << "Configured kv_cache_dtype='" << boot_cfg.batching.kv_cache_dtype
                  << "' will use default or online dynamic scales, so generation "
                  << "may drift from the calibrated reference.\n";
    }

    const auto t0 = std::chrono::steady_clock::now();

    // The store holds what the plan *produces*, which is more than the contract
    // demands: the declaration is a lower bound the loader must meet, not an
    // inventory (an undeclared family demands nothing at all).
    WeightStoreBuilder(e.weights_).reserve(planned_load.planned_tensor_count);
    const auto load_view = planned_load.plan.view();
    if constexpr (false) {
        const char* dump_path = nullptr;
        std::ofstream out(dump_path);
        if (!out) {
            throw std::runtime_error(
                "engine: failed to open PIE_CUDA_RUST_LAYOUT_PLAN_DUMP "
                "path: " + std::string(dump_path));
        }
        out << pie_loader::bytes_to_string(load_view.stats_json);
    }
    if (verbose) {
        std::cerr
            << "[pie-driver-cuda] layout compiler: rust RuntimeABI -> "
               "algebra -> LoadPlan\n";
        std::cerr << "[pie-driver-cuda] rust loader compiler: "
                  << pie_loader::bytes_to_string(load_view.summary) << "\n";
    }
    // Materialized-weight artifact cache (WEIGHT_LOADER_TODO.md A3.1). The
    // materialized weights are a deterministic function of the load-plan cache key,
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
                cache_builder, planned_load.cache_key, weight_cache_dir);
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
        LoadPlanExecutor load_executor(
            loader,
            rust_builder,
            std::move(planned_load.quant_attachments));
        log_stage("materialize LoadPlan begin");
        LoadExecutionStats load_memory_stats;
        const bool sample_load_memory = verbose;
        LoadMemorySampler load_memory_sampler{.stats = &load_memory_stats};
        if (sample_load_memory) {
            LoadMemorySampler::sample(&load_memory_sampler);
        }
        {
            ScopedDeviceTensorMemoryCallback callback(
                sample_load_memory ? &load_memory_sampler : nullptr);
            materialized = load_executor.execute(load_view);
        }
        CUDA_CHECK(cudaDeviceSynchronize());
        log_stage("materialize LoadPlan done");
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
                    e.weights_, planned_load.cache_key, weight_cache_dir);
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
        std::cerr << "[pie-driver-cuda] LoadPlan quantised "
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
    if constexpr (false) {
        const auto to_mib = [](std::uint64_t bytes) {
            return bytes / (1024ull * 1024ull);
        };
        std::cerr << "[pie-driver-cuda] load executor profile: h2d_copies="
                  << materialized.h2d_copy_count
                  << " bulk_copies=" << materialized.h2d_bulk_copy_count
                  << " pinned_copies="
                  << materialized.h2d_pinned_copy_count
                  << " h2d_bytes=" << to_mib(materialized.h2d_copy_bytes)
                  << " MiB bulk_bytes="
                  << to_mib(materialized.h2d_bulk_copy_bytes)
                  << " MiB pinned_bytes="
                  << to_mib(materialized.h2d_pinned_copy_bytes)
                  << " MiB copy_flushes="
                  << materialized.copy_stream_flushes
                  << " batch_calls="
                  << materialized.h2d_batch_calls
                  << " max_pending="
                  << materialized.max_pending_copies_seen << "\n";
        std::cerr << "[pie-driver-cuda] load executor phases: alloc="
                  << static_cast<int>(materialized.phase_alloc_ms)
                  << "ms transfer=" << static_cast<int>(materialized.phase_transfer_ms)
                  << "ms (pinned_alloc="
                  << static_cast<int>(materialized.phase_pinned_alloc_ms)
                  << "ms) transform="
                  << static_cast<int>(materialized.phase_transform_ms) << "ms\n";
    }

    // The groups, if the contract declared any.
    //
    // This happens here, after the resident weights are on the device and
    // before anything else measures the card, and that ordering is what makes
    // the slab cost nothing to account for: the KV pool sizes itself from
    // `cudaMemGetInfo` at context construction, so a slab taken now is already
    // subtracted from what it sees. There is no second budget to keep in step.
    if (boot_cfg.model.stream_routed_experts && load_view.groups.len > 0) {
        const std::uint64_t budget = group_cache_budget(
            boot_cfg.model.expert_cache_gb, load_view.groups, verbose);
        e.group_cache_ = std::make_unique<GroupStreamCache>(
            loader, load_view.groups, budget,
            static_cast<std::uint64_t>(
                boot_cfg.model.expert_host_cache_gb * 1024.0 * 1024.0 * 1024.0),
            verbose);
        e.stream_source_ = std::move(source);
        e.stream_plan_ = std::make_unique<pie_loader::LoadPlan>(
            std::move(planned_load.plan));
        if (verbose) {
            const auto& cache = *e.group_cache_;
            std::cerr << "[pie-driver-cuda] streaming " << load_view.groups.len
                      << " group(s), " << cache.total_instances()
                      << " instances of "
                      << (cache.slot_bytes() / (1024ull * 1024ull))
                      << " MiB in " << cache.num_slots() << " slots ("
                      << (cache.slab_bytes() / (1024ull * 1024ull)) << " MiB)"
                      << (cache.fully_resident() ? " -- fully resident\n"
                                                 : "\n");
        }
    }

    e.weights_.validate_quant_metadata();
    const std::uint64_t loaded_bytes = e.weights_.total_bytes();

    const auto t1 = std::chrono::steady_clock::now();
    const double ms = std::chrono::duration<double, std::milli>(t1 - t0).count();
    const double mib = static_cast<double>(loaded_bytes) / (1024.0 * 1024.0);

    if (verbose) {
        std::cerr << "[pie-driver-cuda] loaded " << e.weights_.size() << " tensors ("
                  << static_cast<std::uint64_t>(mib) << " MiB on this rank, "
                  << "tp=" << boot_cfg.distributed.tp_size << ") in " << static_cast<int>(ms)
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
        std::size_t rows = 0;
        std::size_t cols = 0;
        const bool is_mxfp4 =
            item.second.group_size == 32 &&
            (tensor.dtype() == DType::MXFP4_PACKED ||
             tensor.dtype() == DType::UINT8);
        if (is_mxfp4 && tensor.shape().size() == 1) {
            spec.has_fp8 = true;
            if (tensor.nbytes() >
                std::numeric_limits<std::size_t>::max() / 2) {
                throw std::runtime_error(
                    "runtime quant MXFP4 dimensions overflow");
            }
            spec.max_dequant_weight_elems = std::max(
                spec.max_dequant_weight_elems, tensor.nbytes() * 2);
            spec.max_weight_rows =
                std::max<std::size_t>(spec.max_weight_rows, 1);
            spec.max_weight_cols =
                std::max<std::size_t>(spec.max_weight_cols, 1);
            continue;
        } else if (is_mxfp4 && tensor.shape().size() == 2) {
            rows = static_cast<std::size_t>(std::max<std::int64_t>(
                0, tensor.shape()[0]));
            cols = static_cast<std::size_t>(std::max<std::int64_t>(
                0, tensor.shape()[1])) * 2;
            spec.has_fp8 = true;
        } else if (tensor.shape().size() != 2) {
            continue;
        } else if (tensor.dtype() == DType::FP8_E4M3) {
            spec.has_fp8 = true;
        } else if (tensor.dtype() == DType::INT8) {
            spec.has_int8 = true;
        } else {
            continue;
        }
        if (rows == 0) {
            rows = static_cast<std::size_t>(std::max<std::int64_t>(
                0, tensor.shape()[0]));
            cols = static_cast<std::size_t>(std::max<std::int64_t>(
                0, tensor.shape()[1]));
        }
        if (cols > 0 &&
            rows > std::numeric_limits<std::size_t>::max() / cols) {
            throw std::runtime_error(
                "runtime quant scratch dimensions overflow");
        }

        spec.max_weight_rows = std::max<std::size_t>(
            spec.max_weight_rows, rows);
        spec.max_weight_cols = std::max<std::size_t>(
            spec.max_weight_cols, cols);
        spec.max_dequant_weight_elems = std::max(
            spec.max_dequant_weight_elems, rows * cols);
    }

    return spec;
}

}  // namespace pie_cuda_driver
