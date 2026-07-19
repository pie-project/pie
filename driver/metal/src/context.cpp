#include <pie_native/step_launch.hpp>
#include "context.hpp"

#include <algorithm>
#include <atomic>
#include <cctype>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <exception>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>
#include <unistd.h>

#include <nlohmann/json.hpp>
#include <toml++/toml.hpp>

#include "pie_native/launch_view.hpp"
#include "pie_native/load_plan.hpp"
#include "pie_native/ptir_channels.hpp"
#include "pipeline/interp.hpp"
#include "pipeline/descriptor_resolve.hpp"
#include "pipeline/registry.hpp"
#include "batch/compose.hpp"
#include "batch/forward.hpp"
#include "batch/worker.hpp"
#include "decode_abi.hpp"
#include "observability.hpp"
#if defined(__APPLE__)
#include "mtl4_context.hpp"
#include "pipeline/m1_runtime.hpp"
#endif

namespace pie::metal {
namespace {

namespace interp = pie::metal::pipeline;
namespace executor = pie::metal::batch;
namespace backend = pie::metal;
using pipeline::ChannelRecord;
using pipeline::InstanceRecord;
using pipeline::ProgramRecord;
using executor::LaunchJobData;
using executor::LaunchMember;

enum class TicketPreparation { Ready, Retry, Failed };
enum class MemberRunOutcome { Committed, Retry, Failed };
enum class ForwardBuildResult { Ready, Retry, Failed };

// Directory the compiled-in default .metal kernel library resolves to
// (driver/metal/CMakeLists.txt), overridable at run time.
#ifndef PIE_METAL_KERNELS_DIR_DEFAULT
#define PIE_METAL_KERNELS_DIR_DEFAULT ""
#endif

std::string metal_kernels_dir() {
    if (const char* env = std::getenv("PIE_METAL_KERNELS_DIR")) return std::string(env);
    return PIE_METAL_KERNELS_DIR_DEFAULT;
}

struct RuntimeConfig {
    bool verbose = false;
};

struct ModelConfig {
    std::string hf_path;
    std::string backend = "metal:0";
};

struct BatchingConfig {
    std::uint32_t kv_page_size = 32;
    std::uint32_t total_pages = 1024;
    std::uint32_t max_forward_tokens = 10240;
    std::uint32_t max_forward_requests = 512;
    std::uint32_t cpu_pages = 0;
    std::string kv_cache_dtype = "auto";
};

struct Config {
    ModelConfig model;
    BatchingConfig batching;
    RuntimeConfig runtime;
};

Config load_config(const std::filesystem::path& path) {
    if (!std::filesystem::exists(path)) {
        throw std::runtime_error("config not found: " + path.string());
    }

    auto tbl = toml::parse_file(path.string());
    Config config;
    if (auto model = tbl["model"].as_table()) {
        config.model.hf_path = (*model)["hf_path"].value_or(std::string{});
        config.model.backend = (*model)["backend"].value_or(config.model.backend);
    }
    if (auto batching = tbl["batching"].as_table()) {
        constexpr std::string_view allowed[] = {
            "kv_page_size",
            "total_pages",
            "max_forward_tokens",
            "max_forward_requests",
            "cpu_pages",
            "kv_cache_dtype",
        };
        for (const auto& [key, _] : *batching) {
            const auto name = key.str();
            const bool known = std::any_of(
                std::begin(allowed), std::end(allowed),
                [name](std::string_view candidate) { return name == candidate; });
            if (!known) {
                throw std::runtime_error(
                    "config: unknown [batching] key: " + std::string{name});
            }
        }
        config.batching.kv_page_size =
            (*batching)["kv_page_size"].value_or<std::int64_t>(
                config.batching.kv_page_size);
        config.batching.total_pages =
            (*batching)["total_pages"].value_or<std::int64_t>(
                config.batching.total_pages);
        config.batching.max_forward_tokens =
            (*batching)["max_forward_tokens"].value_or<std::int64_t>(
                config.batching.max_forward_tokens);
        config.batching.max_forward_requests =
            (*batching)["max_forward_requests"].value_or<std::int64_t>(
                config.batching.max_forward_requests);
        config.batching.cpu_pages =
            (*batching)["cpu_pages"].value_or<std::int64_t>(
                config.batching.cpu_pages);
        config.batching.kv_cache_dtype =
            (*batching)["kv_cache_dtype"].value_or(config.batching.kv_cache_dtype);
    }
    if (auto runtime = tbl["runtime"].as_table()) {
        config.runtime.verbose =
            (*runtime)["verbose"].value_or(config.runtime.verbose);
    }

    const auto& kv = config.batching.kv_cache_dtype;
    if (!(kv == "auto" || kv == "bf16" || kv == "bfloat16" ||
          kv == "fp8_e4m3" || kv == "fp8_e5m2" ||
          kv == "int8_per_token_head" || kv == "fp8_per_token_head" ||
          kv == "fp4_e2m1" || kv == "nvfp4")) {
        throw std::runtime_error(
            "config: invalid [batching].kv_cache_dtype '" + kv +
            "'; expected one of: auto, bf16, bfloat16, fp8_e4m3, fp8_e5m2, "
            "int8_per_token_head, fp8_per_token_head, fp4_e2m1, nvfp4");
    }
    return config;
}

void publish_terminal(PieTerminalCell* cell, std::uint32_t outcome) {
    if (cell == nullptr) return;
    cell->reserved0 = 0;
    std::atomic_ref<std::uint32_t>(cell->outcome).store(outcome, std::memory_order_release);
}

// Slice member `m`'s forward fields out of the batch-wide CSR view (§5.2).
// `member_count` is `members.size()` for THIS launch — every CSR indptr
// slice must carry exactly `member_count + 1` entries when present.
//
// `resolved` (Phase 2, C3): when non-null, the member's token/position/KV-
// page/readout fields come from the descriptor-resolved `FireGeometry`
// INSTEAD of the wire CSR slices — a device-geometry program's wire span is
// an empty placeholder (`pie_native::LaunchView`'s own doc comment); the
// channel-resolved geometry is the only truth for those fields. The
// recurrent-state slot bookkeeping (rs_slot_id/rs_reset) is unrelated to
// device-geometry classification and always comes from the wire.
struct ModelFacts {
    std::uint32_t vocab_size = 32000;
    std::uint32_t max_model_len = 8192;
    std::string arch_name = "llama";
    bool has_linear_attn = false;
};

// Phase 1a (metal_ptir_plan.md §5.4, §12 "Caps honesty"): the Metal forward
// is ONE resident linear-sequence MetalExecutor — a fixed
// `max_ctx_ = 4096` KV/GDN ring (batch/forward.cpp), not the runtime's
// multi-tenant paged pool. Shared between `build_caps_json` (what we
// ADVERTISE) and the Phase 2 descriptor resolver's `validate_fire_geometry`
// page-range check (what we ENFORCE) so both always agree.
constexpr std::uint32_t kMetalPhase1aMaxCtxTokens = 4096;

std::uint32_t effective_total_pages(const Config& cfg, bool rs_cache_required) {
    const std::uint32_t kv_page_size = std::max<std::uint32_t>(1u, cfg.batching.kv_page_size);
    // Ceil-divide: the ring must hold kMetalPhase1aMaxCtxTokens tokens even
    // when kv_page_size does not divide it evenly (a floor division would
    // under-report by up to one page).
    return rs_cache_required ? (kMetalPhase1aMaxCtxTokens + kv_page_size - 1) / kv_page_size
                             : cfg.batching.total_pages;
}

ModelFacts read_model_facts(const std::string& hf_path) {
    ModelFacts facts;
    if (hf_path.empty()) return facts;
    const std::filesystem::path cfg =
        std::filesystem::path(hf_path) / "config.json";
    std::ifstream f(cfg);
    if (!f) return facts;
    try {
        nlohmann::json j;
        f >> j;
        if (j.contains("vocab_size") && j["vocab_size"].is_number_integer()) {
            facts.vocab_size = j["vocab_size"].get<std::uint32_t>();
        }
        if (j.contains("max_position_embeddings") &&
            j["max_position_embeddings"].is_number_integer()) {
            facts.max_model_len = j["max_position_embeddings"].get<std::uint32_t>();
        }
        if (j.contains("architectures") && j["architectures"].is_array() &&
            !j["architectures"].empty()) {
            std::string a = j["architectures"][0].get<std::string>();
            for (auto& c : a) c = static_cast<char>(std::tolower(c));
            const std::string suffix = "forcausallm";
            if (a.size() > suffix.size() &&
                a.compare(a.size() - suffix.size(), suffix.size(), suffix) == 0) {
                a.erase(a.size() - suffix.size());
            }
            if (!a.empty()) facts.arch_name = a;
        }
        const nlohmann::json& tc =
            (j.contains("text_config") && j["text_config"].is_object())
                ? j["text_config"]
                : j;
        if (tc.contains("linear_num_value_heads") &&
            tc["linear_num_value_heads"].is_number_integer() &&
            tc["linear_num_value_heads"].get<int>() > 0) {
            facts.has_linear_attn = true;
        }
        if (tc.contains("layer_types") && tc["layer_types"].is_array()) {
            for (const auto& t : tc["layer_types"]) {
                if (t.is_string() && t.get<std::string>() == "linear_attention") {
                    facts.has_linear_attn = true;
                    break;
                }
            }
        }
    } catch (const std::exception& e) {
        std::cerr << "[pie-driver-metal] warning: failed to parse "
                  << cfg.string() << ": " << e.what() << "\n";
    }
    return facts;
}

std::string build_caps_json(const Config& cfg,
                            const ModelFacts& facts) {
    const bool rs_cache_required = facts.has_linear_attn;
    // Reporting the generic multi-request config values here would let the
    // scheduler build launches this driver structurally cannot honor, so
    // cap them truthfully whenever the checkpoint needs the forward
    // (GDN-hybrid, i.e. this increment's qwen3.6 target) — capping only
    // ever narrows a config/model limit down to what the ring can hold,
    // never inflates a limit that was already smaller.
    //
    // The paged DAG is allocated/bound for exactly these bounded capacities.
    // Do not echo arbitrary config values: doing so would advertise IO/scratch
    // rows or GDN slots that do not exist in the resident decoder.
    constexpr std::uint32_t kMetalPagedMaxForwardRequests =
        executor::kPagedMaxForwardRequests;
    constexpr std::uint32_t kMetalPagedMaxForwardTokens =
        executor::kPagedMaxForwardTokens;
    // Phase 1b: the GDN recurrent-state region is genuinely sized for
    // `batch::kPhase1bRsSlots` addressable slots (heap_layout.hpp
    // plan_heap sizes State region as `max_slots * per_slot_bytes`, and
    // MetalExecutor::setup() sets `DecodeGeometry.max_slots` to exactly this
    // constant) — copy_state can genuinely address any of them. This does
    // NOT relax max_forward_requests: Phase 1b still runs one forward
    // request synchronously; the extra slots exist purely as addressable
    // copy_state destinations/sources (e.g. warm-starting/branching a
    // resident sequence's state), not concurrent forward execution.
    const std::uint32_t rs_cache_slots =
        rs_cache_required ? executor::kPhase1bRsSlots : 0u;
    // Static per-slot byte formula mirrors MetalExecutor::rs_slot_bytes()
    // exactly (conv_state + conv_state_out + recurrent_state per GDN layer),
    // computed here from the shipped qwen3.6 DecodeGeometry{} defaults
    // directly since no live executor/decoder exists yet at capabilities-
    // build time (mirrors how vocab_size is cross-checked without a live
    // decoder).
    std::uint32_t rs_cache_slot_bytes = 0u;
    if (rs_cache_required) {
        const backend::DecodeGeometry g{};
        const std::uint64_t conv_stride = g.gdn_conv_stride_bytes();
        const std::uint64_t recur_stride = g.gdn_recurrent_stride_bytes();
        int gdn_layers = 0;
        for (int l = 0; l < g.n_layers; ++l) {
            if (!backend::DecodeGeometry::is_full_attn(l)) ++gdn_layers;
        }
        rs_cache_slot_bytes = static_cast<std::uint32_t>(
            std::uint64_t(gdn_layers) * (2 * conv_stride + recur_stride));
    }
    const std::uint32_t max_forward_requests =
        rs_cache_required ? std::min(cfg.batching.max_forward_requests,
                                     kMetalPagedMaxForwardRequests)
                          : cfg.batching.max_forward_requests;
    const std::uint32_t max_forward_tokens =
        rs_cache_required
            ? std::min(cfg.batching.max_forward_tokens, kMetalPagedMaxForwardTokens)
            : cfg.batching.max_forward_tokens;
    const std::uint32_t max_model_len =
        rs_cache_required ? std::min(facts.max_model_len, kMetalPhase1aMaxCtxTokens)
                          : facts.max_model_len;
    const std::uint32_t total_pages = effective_total_pages(cfg, rs_cache_required);
    nlohmann::json caps = {
        {"abi_version", PIE_DRIVER_ABI_VERSION},
        {"total_pages", total_pages},
        {"kv_page_size", cfg.batching.kv_page_size},
        {"swap_pool_size", 0},
        {"kv_copy_domain_mask", 1},
        {"rs_cache_required", rs_cache_required},
        {"rs_cache_slots", rs_cache_slots},
        {"rs_cache_slot_bytes", rs_cache_slot_bytes},
        {"device_geometry_port_mask", 0},
        {"max_forward_tokens", max_forward_tokens},
        {"max_forward_requests", max_forward_requests},
        {"max_page_refs", rs_cache_required
                              ? total_pages * max_forward_requests
                              : total_pages},
        {"arch_name", facts.arch_name},
        {"vocab_size", facts.vocab_size},
        {"max_model_len", max_model_len},
        {"activation_dtype", "bf16"},
        {"snapshot_dir", cfg.model.hf_path},
    };
    return caps.dump();
}

}  // namespace

class Context::Impl {
  public:
    struct LaunchDemand {
        std::uint32_t kv_pages = 0;
        std::uint32_t state_slots = 0;
        std::uint32_t token_rows = 0;
    };

    // Tear the MetalExecutor (Metal device/heap/PSO objects) down ON the
    // worker thread that created + exclusively drove them (Phase 3, §7 thread-
    // affinity). `worker_.run` drains behind any still-queued jobs first (FIFO),
    // so all in-flight launches/control ops settle before the executor is freed;
    // both the null-check and the reset run on the worker so `executor_` is
    // never touched off that thread. The destructor body runs before member
    // destructors, so `executor_` is already null when its unique_ptr member is
    // destroyed and `worker_` is still alive here.
    ~Impl() {
        worker_.run([this] {
            if (executor_ != nullptr) executor_.reset();
#if defined(__APPLE__)
            if (m1_runtime_ != nullptr) m1_runtime_.reset();
#endif
        });
    }

    int initialize(const std::string& config_path, const PieRuntimeCallbacks& runtime) {
        cfg_ = load_config(config_path);
        runtime_ = runtime;
        std::uint32_t alignment = alignof(std::max_align_t);
        const long host_page_size = ::sysconf(_SC_PAGESIZE);
        std::uint32_t page_size =
            static_cast<std::uint32_t>(host_page_size > 0 ? host_page_size : 1);
#if defined(__APPLE__)
        const MetalStorageFacts storage = query_metal_storage_facts();
        alignment = storage.alignment;
        page_size = storage.page_size;
#endif
        device_facts_json_ = nlohmann::json{
            {"abi_version", PIE_DRIVER_ABI_VERSION},
            {"backend", "metal"},
            {"unified_memory", true},
            {"fp8_native", false},
            {"native_mxfp4_moe", false},
            {"storage_alignment", alignment},
            {"storage_max_tile_bytes", 64ull * 1024ull * 1024ull},
            {"storage_tile_map_mask", pie_load_planner::kMetalTileMapMask},
            {"page_size", page_size},
        }.dump();
        storage_page_size_ = page_size;
        return PIE_STATUS_OK;
    }

    void fill_device_facts(PieDriverCaps* caps) const {
        if (caps == nullptr) return;
        caps->json_bytes =
            reinterpret_cast<const std::uint8_t*>(device_facts_json_.data());
        caps->json_len = device_facts_json_.size();
    }

    int load_model(const PieModelLoadDesc& load, PieDriverCaps* caps) {
        if (load_attempted_) {
            return PIE_STATUS_CLOSED;
        }
        load_attempted_ = true;
        if (load.component != PIE_MODEL_COMPONENT_FULL) {
            return PIE_STATUS_UNSUPPORTED;
        }
        cfg_.model.hf_path.assign(
            reinterpret_cast<const char*>(load.snapshot_dir.ptr),
            load.snapshot_dir.len);
        load_plan_bytes_.assign(
            load.load_plan_bytes.ptr,
            load.load_plan_bytes.ptr + load.load_plan_bytes.len);
        compiler_version_ = load.compiler_version;
        facts_ = read_model_facts(cfg_.model.hf_path);
        std::string error;
        if (!ensure_executor(error)) {
            std::cerr << "[pie-driver-metal] load_model: " << error << "\n";
            return PIE_STATUS_UNSUPPORTED;
        }
        nlohmann::json capabilities =
            nlohmann::json::parse(build_caps_json(cfg_, facts_));
        capabilities["elastic_page_bytes"] =
            executor_->elastic_page_bytes();
        capabilities["elastic_budget_pages"] =
            executor_->elastic_budget_pages();
        caps_json_ = capabilities.dump();
        if (caps != nullptr) {
            caps->json_bytes =
                reinterpret_cast<const std::uint8_t*>(caps_json_.data());
            caps->json_len = caps_json_.size();
        }
        return PIE_STATUS_OK;
    }

    int register_program(const PieProgramDesc& program, std::uint64_t* program_id) {
        std::uint64_t id = 0;
        pipeline::ExecPlan compile_plan;
        std::vector<std::uint8_t> compile_canonical;
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            const int status = registry_.register_program(program, &id);
            if (status != PIE_STATUS_OK) return status;
            ProgramRecord& record = *registry_.find_program(id);
            if (record.m1_executable != nullptr) {
                if (program_id != nullptr) *program_id = id;
                return PIE_STATUS_OK;
            }
            if (!record.m1_error.empty()) {
                std::cerr << "[pie-driver-metal] register_program: "
                          << record.m1_error << "\n";
                return PIE_STATUS_UNSUPPORTED;
            }
            if (!record.plan.executable) {
                record.m1_error = record.plan.reject_reason;
                std::cerr << "[pie-driver-metal] register_program: "
                          << record.m1_error << "\n";
                return PIE_STATUS_UNSUPPORTED;
            }
            compile_plan = record.plan;
            compile_canonical = record.canonical_bytes;
        }

#if defined(__APPLE__)
        std::shared_ptr<pipeline::M1ProgramExecutable> executable;
        std::string compile_error;
        pipeline::M1CompileFailureKind compile_failure =
            pipeline::M1CompileFailureKind::Retryable;
        worker_.run([&] {
            if (m1_runtime_ == nullptr) {
                m1_runtime_ = pipeline::M1Runtime::create(
                    metal_kernels_dir(),
                    pipeline::default_m1_cache_dir(),
                    compile_error);
            }
            if (m1_runtime_ != nullptr) {
                executable = m1_runtime_->compile_program(
                    program.program_hash,
                    compile_plan,
                    compile_error,
                    compile_canonical,
                    &compile_failure);
            }
        });
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            ProgramRecord& record = *registry_.find_program(id);
            if (executable == nullptr) {
                const std::string message =
                    compile_error.empty()
                        ? "Metal M1 compilation failed"
                        : compile_error;
                if (compile_failure ==
                    pipeline::M1CompileFailureKind::Deterministic) {
                    record.m1_error = message;
                }
                std::cerr << "[pie-driver-metal] register_program: "
                          << message << "\n";
                return compile_failure ==
                               pipeline::M1CompileFailureKind::Deterministic
                           ? PIE_STATUS_UNSUPPORTED
                           : PIE_STATUS_DRIVER_ERROR;
            }
            record.m1_executable = std::move(executable);
        }
#endif
        if (program_id != nullptr) *program_id = id;
        return PIE_STATUS_OK;
    }

    int register_channel(const PieChannelDesc& channel,
                         PieChannelEndpointBinding* binding) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return registry_.register_channel(channel, binding);
    }

    int bind_instance(const PieInstanceDesc& instance, PieInstanceBinding* binding) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return registry_.bind_instance(instance, binding);
    }

    // ABI v2 launch (Phase 3, review item 1 — ASYNC). SYNCHRONOUS PREFLIGHT
    // under `state_mutex_`: validate instance/program ids, static rules, and
    // ticket shape. Then deep-copy the accepted batch into an
    // owned `LaunchJobData` (each forward member's descriptor is a fully-owned
    // copy; no launch-array pointer is retained), POST it to the executor
    // worker, and RETURN — `pie_metal_launch` never waits for the GPU forward
    // or settlement. The worker (`run_launch_job`) runs forward + interp step +
    // §4.4 publication (channel words → terminals → per-channel notifies → the
    // batch notify, exactly once) off the caller thread. Non-forward
    // (channel-plane C1) members settle the same way, just without a forward.
    static LaunchDemand launch_demand(const pie_native::StepLaunch& launch) {
        LaunchDemand demand;
        demand.kv_pages = launch.required_kv_pages;
        auto include_pages = [&demand](PieU32Slice pages) {
            if (pages.len == 0) return;
            demand.kv_pages = std::max(
                demand.kv_pages,
                *std::max_element(pages.ptr, pages.ptr + pages.len) + 1);
        };
        include_pages(launch.kv_page_indices);
        auto include_slots = [&demand](PieU32Slice slots) {
            if (slots.len == 0) return;
            demand.state_slots = std::max(
                demand.state_slots,
                *std::max_element(slots.ptr, slots.ptr + slots.len) + 1);
        };
        include_slots(launch.rs_slot_ids);
        include_slots(launch.rs_buffer_slot_ids);
        demand.token_rows = static_cast<std::uint32_t>(
            std::min<std::size_t>(
                std::numeric_limits<std::uint32_t>::max(),
                std::max(launch.token_ids.len, launch.instance_ids.len)));
        return demand;
    }

    static bool demand_covers(
        const LaunchDemand& reserved,
        const LaunchDemand& actual) {
        return reserved.kv_pages >= actual.kv_pages &&
               reserved.state_slots >= actual.state_slots &&
               reserved.token_rows >= actual.token_rows;
    }

    // Post one sealed frame (ABI v14): expand each step to the internal
    // batch shape (roster → instance ids, frame translation → per-step
    // slices) and run the per-batch path per step. The tail step carries
    // the frame completion; the executor worker is a FIFO, so the tail's
    // settlement implies every step settled.
    int launch(const PieFrameDesc& frame, PieCompletion completion) {
        const PieStepDesc* steps = frame.steps.ptr;
        const std::size_t step_count = frame.steps.len;
        if (steps == nullptr || step_count == 0) {
            return PIE_STATUS_INVALID_ARGUMENT;
        }
        for (std::size_t i = 0; i < step_count; ++i) {
            StepExpansion expansion;
            expand_step(frame, steps[i], &expansion);
            const bool tail = i + 1 == step_count;
            const PieCompletion step_completion =
                tail ? completion : PieCompletion{0, 0, nullptr};
            const int status =
                launch_impl(expansion.launch, step_completion);
            if (status != PIE_STATUS_OK) return status;
        }
        return PIE_STATUS_OK;
    }

    struct StepExpansion {
        std::vector<std::uint64_t> instance_ids;
        std::vector<std::uint32_t> kv_translation;
        std::vector<std::uint32_t> kv_translation_indptr;
        pie_native::StepLaunch launch{};
    };

    static void expand_step(
        const PieFrameDesc& frame,
        const PieStepDesc& step,
        StepExpansion* out) {
        out->instance_ids.reserve(step.roster_rows.len);
        out->kv_translation_indptr.reserve(step.roster_rows.len + 1);
        out->kv_translation_indptr.push_back(0);
        const bool have_translation = frame.kv_translation_indptr.len != 0;
        for (std::size_t i = 0; i < step.roster_rows.len; ++i) {
            const std::uint32_t row = step.roster_rows.ptr[i];
            out->instance_ids.push_back(frame.instance_ids.ptr[row]);
            if (have_translation) {
                const std::uint32_t begin = frame.kv_translation_indptr.ptr[row];
                const std::uint32_t end =
                    frame.kv_translation_indptr.ptr[row + 1];
                out->kv_translation.insert(
                    out->kv_translation.end(),
                    frame.kv_translation.ptr + begin,
                    frame.kv_translation.ptr + end);
            }
            out->kv_translation_indptr.push_back(
                static_cast<std::uint32_t>(out->kv_translation.size()));
        }
        pie_native::StepLaunch& launch = out->launch;
        launch.instance_ids = {
            out->instance_ids.data(), out->instance_ids.size()};
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
            out->kv_translation_indptr.data(),
            out->kv_translation_indptr.size()};
        launch.ptir_program_row_indptr = step.ptir_program_row_indptr;
        launch.ptir_kv_write_lower_bounds = step.ptir_kv_write_lower_bounds;
        launch.ptir_kv_write_upper_bounds = step.ptir_kv_write_upper_bounds;
        launch.logical_fire_ids = step.logical_fire_ids;
        launch.channel_expected_head = step.channel_expected_head;
        launch.channel_expected_tail = step.channel_expected_tail;
        launch.channel_ticket_indptr = step.channel_ticket_indptr;
    }

    int launch_impl(
        const pie_native::StepLaunch& launch,
        PieCompletion completion) {
        std::unique_lock<std::mutex> lock_holder(state_mutex_);
        std::vector<InstanceRecord*> members;
        members.reserve(launch.instance_ids.len);
        for (std::size_t i = 0; i < launch.instance_ids.len; ++i) {
            InstanceRecord* instance =
                registry_.find_instance(launch.instance_ids.ptr[i]);
            if (instance == nullptr) {
                return PIE_STATUS_INVALID_ARGUMENT;
            }
            members.push_back(instance);
        }
        for (const InstanceRecord* member : members) {
            const ProgramRecord& program =
                *registry_.find_program(member->program_id);
            if (!program.plan.executable) {
                std::cerr << "[pie-driver-metal] launch: instance " << member->instance_id
                          << ": " << program.plan.reject_reason << "\n";
                return PIE_STATUS_UNSUPPORTED;
            }
        }
        if (launch.has_user_mask != 0) {
            std::cerr
                << "[pie-driver-metal] launch: user-provided wire masks "
                   "require BRLE decoding, which Metal does not support; "
                   "refusing to run unmasked\n";
            return PIE_STATUS_UNSUPPORTED;
        }
        // Phase 2 (C3): at most one device-geometry program per launch batch
        // — the same structural constraint the runtime's scheduler already
        // upholds (metal_ptir_plan.md §6); a defensive re-check here so a
        // scheduling bug fails the launch loudly instead of resolving two
        // programs' geometry against one shared forward.
        {
            std::size_t device_geometry_count = 0;
            for (const InstanceRecord* member : members) {
                const ProgramRecord& program =
                    *registry_.find_program(member->program_id);
                if (interp::requires_descriptor_resolution(program.plan.trace)) {
                    ++device_geometry_count;
                }
            }
            if (device_geometry_count > 1) {
                std::cerr << "[pie-driver-metal] launch: " << device_geometry_count
                          << " device-geometry programs in one batch (at most one is supported)\n";
                return PIE_STATUS_INVALID_ARGUMENT;
            }
        }

        if (launch.channel_ticket_indptr.len != members.size() + 1 ||
            launch.channel_expected_head.len != launch.channel_expected_tail.len) {
            std::cerr << "[pie-driver-metal] launch: channel tickets are required\n";
            return PIE_STATUS_INVALID_ARGUMENT;
        }

        // Deep-copy the accepted batch and its immutable sequence tickets.
        // Availability is checked on the worker immediately before execution;
        // accepting the launch never consumes or reserves a host-ring entry.
        // launch into an OWNED job and POST it to the executor worker, then
        // return WITHOUT waiting. The worker runs the (possibly multi-ms) GPU
        // forward + interp step + settlement + publication; `pie_metal_launch`
        // must not block the engine's scheduler thread on any of that. Every
        // per-member forward descriptor is a fully-owned copy built here (under
        // the state mutex, instances alive), so the job borrows no launch-array
        // pointer after this returns. Instances are re-resolved by id on the
        // worker so a close racing the in-flight job is handled, not UAF'd.
        auto job = std::make_shared<LaunchJobData>();
        job->completion = completion;
        job->launch = executor::OwnedLaunchView::capture(launch);
        job->members.resize(members.size());
        for (std::size_t m = 0; m < members.size(); ++m) {
            LaunchMember& lm = job->members[m];
            lm.instance_id = members[m]->instance_id;
            lm.terminal_cell = launch.terminal_cells.ptr[m];
            const ProgramRecord& program =
                *registry_.find_program(members[m]->program_id);
            lm.needs_forward = program.plan.needs_forward();
            const std::size_t lo = launch.channel_ticket_indptr.ptr[m];
            const std::size_t hi = launch.channel_ticket_indptr.ptr[m + 1];
            if (hi < lo || hi - lo != members[m]->channel_ids.size() ||
                hi > launch.channel_expected_head.len) {
                return PIE_STATUS_INVALID_ARGUMENT;
            }
            lm.tickets.reserve(hi - lo);
            for (std::size_t dense = 0; dense < hi - lo; ++dense) {
                const std::uint64_t expected_head =
                    launch.channel_expected_head.ptr[lo + dense];
                const std::uint64_t expected_tail =
                    launch.channel_expected_tail.ptr[lo + dense];
                if ((program.plan.takes_channel(static_cast<std::uint32_t>(dense)) &&
                     expected_head == executor::kNoChannelTicket) ||
                    (program.plan.puts_channel(static_cast<std::uint32_t>(dense)) &&
                     expected_tail == executor::kNoChannelTicket)) {
                    return PIE_STATUS_INVALID_ARGUMENT;
                }
                lm.tickets.push_back(executor::ChannelTicket{
                    .channel_id = members[m]->channel_ids[dense],
                    .dense = dense,
                    .expected_head = expected_head,
                    .expected_tail = expected_tail,
                    .requires_input =
                        expected_head != executor::kNoChannelTicket ||
                        program.plan.requires_channel_input(
                            static_cast<std::uint32_t>(dense)),
                });
            }
        }
        lock_holder.unlock();  // never hold state_mutex_ across the worker post/return
        worker_.post([this, job] { run_launch_job(job); });
        return PIE_STATUS_OK;
    }

    // Phase 3 (review item 1): the OWNED, async settlement of one accepted
    // launch, executed on the executor worker thread. Publishes in the §4.4
    // order (channel words while settling → terminal cells → per-channel
    // notifies → the batch notify exactly once), regardless of any fault. The
    // GPU forward runs WITHOUT the state mutex (so it never blocks a concurrent
    // launch preflight); only the interp step + channel settlement take the
    // mutex. Item 3: exceptions from the forward or a member's settlement are
    // caught and translated to that member's terminal FAILED with the original
    // what() diagnostic — never swallowed, never left pending.
    void run_launch_job(std::shared_ptr<LaunchJobData> job) {
        const M0TimingSnapshot timing_before =
            m0_timing_counters().snapshot();
#if defined(__APPLE__)
        const pipeline::M3GroupStats m3_before =
            m1_runtime_ != nullptr ? m1_runtime_->m3_stats()
                                   : pipeline::M3GroupStats{};
#endif
        const std::size_t M = job->members.size();
        std::vector<std::uint32_t> outcomes(M, PIE_TERMINAL_OUTCOME_SUCCESS);
        std::vector<std::pair<std::uint64_t, std::uint64_t>> notifications;
        std::string kv_commit_error;
        const bool kv_commit_failed =
            job->launch.required_kv_pages != 0 &&
            (executor_ == nullptr ||
             !executor_->ensure_kv_pages(
                 job->launch.required_kv_pages, &kv_commit_error));
        if (kv_commit_failed && kv_commit_error.empty()) {
            kv_commit_error = "Metal KV commit failed";
        }
#if defined(__APPLE__)
        struct PendingM3Group {
            std::vector<std::size_t> members;
            std::vector<pipeline::M3LaneCandidate> candidates;
            std::vector<std::uint8_t> accepted;
            std::vector<std::size_t> accepted_members;
            std::shared_ptr<pipeline::M3GroupCommand> command;
            bool finalized = false;
            std::size_t leader = std::numeric_limits<std::size_t>::max();
        };
        std::vector<std::shared_ptr<pipeline::M1PreparedFire>> prepared(M);
        std::vector<std::shared_ptr<pipeline::M2CommandPlan>> m2_commands(M);
        std::vector<std::uint8_t> m2_active(M, 0);
        std::vector<std::shared_ptr<PendingM3Group>> m3_for_member(M);
        std::vector<pipeline::M1ExecuteOutcome> m3_outcomes(
            M, pipeline::M1ExecuteOutcome::Failed);
        std::vector<std::uint8_t> m3_active(M, 0);
        std::vector<std::shared_ptr<PendingM3Group>> m3_groups;
#endif

        // ── Phase 0: execution-time ticket validation directly against the
        // authoritative Shared-storage channel rings. ──
        const pie_native::LaunchView view = job->launch.view();
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            job->fwd_descs.clear();
            for (std::size_t m = 0; m < M; ++m) {
                LaunchMember& lm = job->members[m];
                if (kv_commit_failed) {
                    lm.build_err = kv_commit_error;
                    continue;
                }
                InstanceRecord* instance = registry_.find_instance(lm.instance_id);
                if (instance == nullptr) {
                    lm.build_err = "instance closed before execution";
                    continue;
                }
                InstanceRecord& member = *instance;
                const ProgramRecord& program =
                    *registry_.find_program(member.program_id);
#if defined(__APPLE__)
                lm.requires_m2 = m1_runtime_->requires_m2_placement(
                    program.m1_executable);
                lm.mtp_draft_row =
                    interp::bounded_mtp_row_base(
                        program.plan, facts_.vocab_size);
                if (program.plan.needs_mtp_logits &&
                    lm.mtp_draft_row < 0) {
                    lm.build_err =
                        "bounded MTP row base cannot be derived for this "
                        "model/program";
                    continue;
                }
#endif
                std::string prepare_error;
#if defined(__APPLE__)
                const pipeline::M1PrepareOutcome prepared_outcome =
                    m1_runtime_ != nullptr && program.m1_executable != nullptr
                        ? m1_runtime_->prepare(
                              program.m1_executable,
                              member.interp.channels,
                              lm.tickets,
                              prepared[m],
                              prepare_error)
                        : pipeline::M1PrepareOutcome::Failed;
                const TicketPreparation prepared_status =
                    prepared_outcome == pipeline::M1PrepareOutcome::Ready
                        ? TicketPreparation::Ready
                        : (prepared_outcome == pipeline::M1PrepareOutcome::Retry
                               ? TicketPreparation::Retry
                               : TicketPreparation::Failed);
#else
                const TicketPreparation prepared =
                    prepare_member_tickets(lm.tickets, prepare_error);
                const TicketPreparation prepared_status = prepared;
#endif
                if (prepared_status == TicketPreparation::Retry) {
                    outcomes[m] = PIE_TERMINAL_OUTCOME_RETRY;
                    continue;
                }
                if (prepared_status == TicketPreparation::Failed) {
                    if (prepare_error.empty()) {
                        prepare_error =
                            "Metal M1 executable is unavailable";
                    }
                    lm.build_err = std::move(prepare_error);
                    continue;
                }
                if (!lm.needs_forward) continue;
                executor::MemberForwardDesc desc;
                const ForwardBuildResult built = build_forward_desc_for_member(
                    view, m, M, member, program, desc, lm.build_err);
                if (built == ForwardBuildResult::Ready) {
                    lm.fwd_slot = static_cast<int>(job->fwd_descs.size());
                    job->fwd_descs.push_back(std::move(desc));
                } else if (built == ForwardBuildResult::Retry) {
                    outcomes[m] = PIE_TERMINAL_OUTCOME_RETRY;
                    lm.build_err.clear();
#if defined(__APPLE__)
                    if (m1_runtime_ != nullptr) {
                        m1_runtime_->release(prepared[m]);
                    }
#endif
                }
            }
        }

        // ── Phase 1: GPU forward (no mutex; executor is worker-owned) ──
        std::vector<executor::LogitsOut> fwd_outs;
        std::vector<std::uint8_t> fwd_ok;
        std::vector<std::string> fwd_err;
        std::string setup_err;
        bool executor_ready = true;
#if defined(__APPLE__)
        {
            std::unordered_map<std::string, std::vector<std::size_t>>
                channel_groups;
            for (std::size_t m = 0; m < M; ++m) {
                if (outcomes[m] != PIE_TERMINAL_OUTCOME_SUCCESS ||
                    job->members[m].needs_forward ||
                    prepared[m] == nullptr) {
                    continue;
                }
                channel_groups["all"].push_back(m);
            }
            for (auto& [key, members] : channel_groups) {
                static_cast<void>(key);
                std::vector<pipeline::M3LaneCandidate> candidates;
                for (const std::size_t member : members) {
                    candidates.push_back({
                        .fire = prepared[member],
                        .inputs = {},
                        .retry_ineligible = false,
                    });
                }
                std::shared_ptr<pipeline::M3GroupCommand> command;
                std::string group_error;
                if (!m1_runtime_->prepare_m3_group(
                        candidates,
                        m1_runtime_->context(),
                        command,
                        group_error)) {
                    if (cfg_.runtime.verbose && !group_error.empty()) {
                        std::cerr
                            << "[pie-driver-metal] channel M3 fallback: "
                            << group_error << "\n";
                    }
                    continue;
                }
                const StepTiming timing =
                    m1_runtime_->context().run_step(
                    [&](StepEncoder& encoder) {
                        m1_runtime_->encode_m3_pre(command, encoder);
                        m1_runtime_->encode_m3_post(command, encoder);
                    });
                auto grouped = m1_runtime_->finish_m3_group(
                    command, group_error);
                if (!timing.succeeded()) {
                    std::fill(
                        grouped.begin(),
                        grouped.end(),
                        pipeline::M1ExecuteOutcome::Failed);
                    group_error =
                        "Metal M3 command timed out before its completion fence";
                }
                for (std::size_t lane = 0;
                     lane < grouped.size() && lane < members.size();
                     ++lane) {
                    m3_active[members[lane]] = 1;
                    m3_outcomes[members[lane]] = grouped[lane];
                }
            }
        }
#endif
        bool has_forward = false;
        for (std::size_t m = 0; m < M; ++m)
            if (outcomes[m] == PIE_TERMINAL_OUTCOME_SUCCESS &&
                job->members[m].needs_forward &&
                job->members[m].build_err.empty())
                has_forward = true;
        if (has_forward) {
            try {
                executor_ready = ensure_executor(setup_err);  // inline on the worker
                if (executor_ready) {
#if defined(__APPLE__)
                    std::vector<executor::PtirCommandCallbacks> callbacks(
                        job->fwd_descs.size());
                    std::vector<std::uint32_t> token_bases(
                        job->fwd_descs.size(), 0);
                    std::uint32_t token_base = 0;
                    for (std::size_t slot = 0;
                         slot < job->fwd_descs.size();
                         ++slot) {
                        token_bases[slot] = token_base;
                        token_base += static_cast<std::uint32_t>(
                            job->fwd_descs[slot].token_ids.size());
                    }
                    const bool legacy_single =
                        job->fwd_descs.size() == 1 &&
                        !job->fwd_descs[0].requires_paged &&
                        !job->fwd_descs[0].has_write_desc;
                    RawMetalContext* command_context =
                        executor_->command_context();
                    std::vector<pipeline::M1DeviceInputs> member_inputs(M);
                    std::vector<std::uint8_t> direct_eligible(M, 0);
                    for (std::size_t m = 0; m < M; ++m) {
                        LaunchMember& member = job->members[m];
                        if (!member.needs_forward ||
                            member.fwd_slot < 0 ||
                            prepared[m] == nullptr) {
                            continue;
                        }
                        const std::size_t slot =
                            static_cast<std::size_t>(member.fwd_slot);
                        const auto& desc = job->fwd_descs[slot];
                        if (desc.readout_local_indices.empty() ||
                            (legacy_single &&
                             (desc.readout_local_indices.size() != 1 ||
                              desc.readout_local_indices[0] + 1 !=
                                  desc.token_ids.size()))) {
                            continue;
                        }
                        pipeline::M1DeviceInputs inputs;
                        inputs.logits_bf16 =
                            executor_->logits_device_slot();
                        inputs.logits_row_offset =
                            legacy_single
                                ? 0
                                : token_bases[slot] +
                                      desc.readout_local_indices[0];
                        inputs.logits_row_count =
                            static_cast<std::uint32_t>(
                                desc.readout_local_indices.size());
                        if (!legacy_single) {
                            for (const std::uint32_t local :
                                 desc.readout_local_indices) {
                                inputs.logits_rows.push_back(
                                    token_bases[slot] + local);
                            }
                        }
                        inputs.vocab = executor_->vocab();
                        inputs.mtp_draft_row = member.mtp_draft_row;
                        inputs.extents =
                            pipeline::m3_extents_from_forward_desc(desc);
                        member_inputs[m] = inputs;
                        direct_eligible[m] = 1;
                    }

                    std::unordered_map<std::string, std::vector<std::size_t>>
                        candidate_groups;
                    for (std::size_t m = 0; m < M; ++m) {
                        if (direct_eligible[m] == 0) continue;
                        candidate_groups["all"].push_back(m);
                    }
                    for (auto& [key, members] : candidate_groups) {
                        static_cast<void>(key);
                        auto group = std::make_shared<PendingM3Group>();
                        group->members = members;
                        group->accepted.assign(members.size(), 0);
                        for (const std::size_t member : members) {
                            const auto& desc = job->fwd_descs[
                                static_cast<std::size_t>(
                                    job->members[member].fwd_slot)];
                            group->candidates.push_back({
                                .fire = prepared[member],
                                .inputs = member_inputs[member],
                                .retry_ineligible =
                                    desc.has_rs_slot ||
                                    job->members[member].requires_m2,
                            });
                            m3_for_member[member] = group;
                        }
                        m3_groups.push_back(group);
                        for (std::size_t lane = 0;
                             lane < members.size();
                             ++lane) {
                            const std::size_t member = members[lane];
                            const std::size_t slot =
                                static_cast<std::size_t>(
                                    job->members[member].fwd_slot);
                            callbacks[slot].set_logits_row =
                                [group, lane](std::uint32_t row) {
                                    group->accepted[lane] = 1;
                                    group->candidates[lane]
                                        .inputs.logits_row_offset = row;
                                    group->candidates[lane]
                                        .inputs.logits_rows = {row};
                                    group->candidates[lane]
                                        .inputs.extents.sampled_rows = 1;
                                };
                            callbacks[slot].set_logits_rows =
                                [group, lane](
                                    const std::vector<std::uint32_t>& rows) {
                                    if (rows.empty()) return;
                                    group->accepted[lane] = 1;
                                    group->candidates[lane]
                                        .inputs.logits_row_offset =
                                        rows.front();
                                    group->candidates[lane]
                                        .inputs.logits_row_count =
                                        static_cast<std::uint32_t>(
                                            rows.size());
                                    group->candidates[lane]
                                        .inputs.logits_rows = rows;
                                    group->candidates[lane]
                                        .inputs.extents.sampled_rows =
                                        static_cast<std::uint32_t>(
                                            rows.size());
                                };
                            callbacks[slot].finalize_group =
                                [&, group] {
                                    if (group->finalized) return;
                                    group->finalized = true;
                                    std::vector<pipeline::M3LaneCandidate>
                                        accepted;
                                    for (std::size_t index = 0;
                                         index < group->members.size();
                                         ++index) {
                                        if (group->accepted[index] == 0)
                                            continue;
                                        accepted.push_back(
                                            group->candidates[index]);
                                        group->accepted_members.push_back(
                                            group->members[index]);
                                    }
                                    if (accepted.empty() ||
                                        command_context == nullptr) {
                                        return;
                                    }
                                    std::string group_error;
                                    if (!m1_runtime_->prepare_m3_group(
                                            accepted,
                                            *command_context,
                                            group->command,
                                            group_error)) {
                                        if (cfg_.runtime.verbose &&
                                            !group_error.empty()) {
                                            std::cerr
                                                << "[pie-driver-metal] M3 group fallback: "
                                                << group_error << "\n";
                                        }
                                        for (std::size_t index = 0;
                                             index <
                                             group->accepted_members.size();
                                             ++index) {
                                            const std::size_t fallback_member =
                                                group->accepted_members[index];
                                            std::string fallback_error;
                                            const auto& rows =
                                                accepted[index].inputs.logits_rows;
                                            const bool contiguous_rows =
                                                rows.empty() ||
                                                std::adjacent_find(
                                                    rows.begin(),
                                                    rows.end(),
                                                    [](std::uint32_t left,
                                                       std::uint32_t right) {
                                                        return right !=
                                                            left + 1;
                                                    }) == rows.end();
                                            if (contiguous_rows &&
                                                m1_runtime_->prepare_m2_command(
                                                    prepared[fallback_member],
                                                    accepted[index].inputs,
                                                    *command_context,
                                                    m2_commands[fallback_member],
                                                    fallback_error)) {
                                                m2_active[fallback_member] = 1;
                                                continue;
                                            }
                                            if (accepted[index]
                                                    .retry_ineligible) {
                                                if (!contiguous_rows) {
                                                    fallback_error =
                                                        "non-contiguous multi-row logits "
                                                        "require M3 row attribution";
                                                }
                                                throw std::runtime_error(
                                                    "definitively admitted Metal M3 lane "
                                                    "lost safe placement before launch: " +
                                                    group_error + "; M2 fallback: " +
                                                    fallback_error);
                                            }
                                        }
                                        return;
                                    }
                                    group->leader =
                                        group->accepted_members.front();
                                    for (const std::size_t accepted_member :
                                         group->accepted_members) {
                                        m3_active[accepted_member] = 1;
                                    }
                                };
                            callbacks[slot].pre_forward =
                                [this, group, member, &m2_active,
                                 &m2_commands](StepEncoder& encoder) {
                                    if (group->command &&
                                        group->leader == member) {
                                        m1_runtime_->encode_m3_pre(
                                            group->command, encoder);
                                    } else if (m2_active[member] != 0) {
                                        m1_runtime_->encode_m2_pre(
                                            m2_commands[member], encoder);
                                    }
                                };
                            callbacks[slot].post_forward =
                                [this, group, member, &m2_active,
                                 &m2_commands](StepEncoder& encoder) {
                                    if (group->command &&
                                        group->leader == member) {
                                        m1_runtime_->encode_m3_post(
                                            group->command, encoder);
                                    } else if (m2_active[member] != 0) {
                                        m1_runtime_->encode_m2_post(
                                            m2_commands[member], encoder);
                                    }
                                };
                            // Keep GPU staging enabled so a failed grouping
                            // attempt can safely use singleton fallback.
                            callbacks[slot].consumes_logits_directly = false;
                        }
                    }

                    for (std::size_t m = 0; m < M; ++m) {
                        LaunchMember& member = job->members[m];
                        if (direct_eligible[m] == 0 ||
                            m3_for_member[m] != nullptr) {
                            continue;
                        }
                        const std::size_t slot =
                            static_cast<std::size_t>(member.fwd_slot);
                        const auto& inputs = member_inputs[m];
                        std::string m2_error;
                        if (command_context == nullptr ||
                            !m1_runtime_->prepare_m2_command(
                                prepared[m],
                                inputs,
                                *command_context,
                                m2_commands[m],
                                m2_error)) {
                            if (member.requires_m2) {
                                executor_ready = false;
                                setup_err =
                                    "required Metal M2 placement failed: " +
                                    m2_error;
                                break;
                            }
                            if (cfg_.runtime.verbose && !m2_error.empty()) {
                                std::cerr
                                    << "[pie-driver-metal] M2 singleton fallback: "
                                    << m2_error << "\n";
                            }
                            continue;
                        }
                        callbacks[slot].pre_forward =
                            [this, command = m2_commands[m]](
                                StepEncoder& encoder) {
                                m1_runtime_->encode_m2_pre(
                                    command, encoder);
                            };
                        callbacks[slot].post_forward =
                            [this, command = m2_commands[m]](
                                StepEncoder& encoder) {
                                m1_runtime_->encode_m2_post(
                                    command, encoder);
                            };
                        callbacks[slot].set_logits_row =
                            [this, command = m2_commands[m]](
                                std::uint32_t row) {
                                m1_runtime_->set_m2_logits_row(
                                    command, row);
                            };
                        callbacks[slot].consumes_logits_directly = true;
                        m2_active[m] = 1;
                    }
                    if (executor_ready) {
                        executor_->forward_batch(
                            job->fwd_descs,
                            fwd_outs,
                            fwd_ok,
                            fwd_err,
                            &callbacks);
                    }
#else
                    executor_->forward_batch(
                        job->fwd_descs, fwd_outs, fwd_ok, fwd_err);
#endif
                }
            } catch (const std::exception& e) {
                executor_ready = false;
                setup_err = std::string("forward raised: ") + e.what();
            } catch (...) {
                executor_ready = false;
                setup_err = "forward raised: unknown exception";
            }
        }
#if defined(__APPLE__)
        for (const auto& group : m3_groups) {
            if (group->command == nullptr) continue;
            std::string group_error;
            const auto group_outcomes =
                m1_runtime_->finish_m3_group(
                    group->command, group_error);
            for (std::size_t lane = 0;
                 lane < group_outcomes.size() &&
                 lane < group->accepted_members.size();
                 ++lane) {
                m3_outcomes[group->accepted_members[lane]] =
                    group_outcomes[lane];
            }
            if (!group_error.empty() && cfg_.runtime.verbose) {
                std::cerr << "[pie-driver-metal] M3 finish: "
                          << group_error << "\n";
            }
        }
#endif

        // ── Phase 2: generated singleton execution + channel settlement. ──
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            for (std::size_t m = 0; m < M; ++m) {
                LaunchMember& lm = job->members[m];
                if (outcomes[m] == PIE_TERMINAL_OUTCOME_RETRY) continue;
                InstanceRecord* instance = registry_.find_instance(lm.instance_id);
                if (instance == nullptr) {
                    // Instance closed after acceptance (guest ordering bug): fail
                    // its terminal rather than dereference freed interp state.
                    std::cerr << "[pie-driver-metal] launch: instance " << lm.instance_id
                              << " was closed before its accepted fire settled\n";
                    outcomes[m] = PIE_TERMINAL_OUTCOME_FAILED;
#if defined(__APPLE__)
                    if (m1_runtime_ != nullptr) m1_runtime_->release(prepared[m]);
#endif
                    continue;
                }
                InstanceRecord& member = *instance;
#if !defined(__APPLE__)
                const ProgramRecord& program =
                    *registry_.find_program(member.program_id);
#endif
                member.fire_seq += 1;
                std::string failure;
#if !defined(__APPLE__)
                interp::PassInputs pass_in{};
#endif
                bool ok = lm.build_err.empty();
                if (!ok) failure = lm.build_err;
#if defined(__APPLE__)
                pipeline::M1DeviceInputs generated_inputs;
#endif
                if (lm.needs_forward) {
                    if (!ok) {
                        // Preserve the preparation/geometry failure.
                    } else if (!executor_ready) {
                        ok = false;
                        failure = "Metal executor setup failed: " + setup_err;
                    } else {
                        const int si = lm.fwd_slot;
                        if (si < 0 || fwd_ok[static_cast<std::size_t>(si)] == 0) {
                            ok = false;
                            failure = si >= 0 ? fwd_err[static_cast<std::size_t>(si)]
                                              : "forward member was not scheduled";
                        } else {
                            const executor::LogitsOut& lo = fwd_outs[static_cast<std::size_t>(si)];
#if defined(__APPLE__)
                            const auto& desc =
                                job->fwd_descs[static_cast<std::size_t>(si)];
                            generated_inputs =
                                pipeline::m1_singleton_fallback_inputs(
                                    lo, desc, lm.mtp_draft_row);
#else
                            pass_in.logits = lo.data.data();
                            pass_in.rows = lo.rows;
                            pass_in.vocab = lo.vocab;
                            pass_in.mtp_draft_row = -1;  // MtpLogits falls back to row 0
#endif
                        }
                    }
                }
                try {
                    if (ok) {
#if defined(__APPLE__)
                        const pipeline::M1ExecuteOutcome generated =
                            m3_active[m] != 0
                                ? m3_outcomes[m]
                            : m2_active[m] != 0
                                ? m1_runtime_->finish_m2_command(
                                      m2_commands[m], failure)
                                : m1_runtime_->execute(
                                      prepared[m],
                                      generated_inputs,
                                      failure);
                        if (generated ==
                            pipeline::M1ExecuteOutcome::Retry) {
                            if (cfg_.runtime.verbose) {
                                std::cerr << "[pie-driver-metal] M1 retry: "
                                          << failure << "\n";
                            }
                            outcomes[m] = PIE_TERMINAL_OUTCOME_RETRY;
                            m1_runtime_->release(prepared[m]);
                            continue;
                        }
                        if (generated ==
                            pipeline::M1ExecuteOutcome::Failed) {
                            ok = false;
                        } else {
                            queue_channel_notifications(
                                lm.tickets, notifications);
                        }
#else
                        const MemberRunOutcome run = run_member(
                            member, program, lm.tickets, pass_in, notifications, failure);
                        if (run == MemberRunOutcome::Retry) {
                            outcomes[m] = PIE_TERMINAL_OUTCOME_RETRY;
                            continue;
                        }
                        if (run == MemberRunOutcome::Failed) ok = false;
#endif
                    }
                } catch (const std::exception& e) {
                    ok = false;
                    failure = std::string("settlement raised: ") + e.what();
                } catch (...) {
                    ok = false;
                    failure = "settlement raised: unknown exception";
                }
                if (!ok) {
                    std::cerr << "[pie-driver-metal] instance " << member.instance_id
                              << " launch failed: " << failure << "\n";
                    poison_instance(member, notifications);
                    outcomes[m] = PIE_TERMINAL_OUTCOME_FAILED;
                }
#if defined(__APPLE__)
                if (m2_commands[m] != nullptr) {
                    std::string ignored;
                    (void)m1_runtime_->finish_m2_command(
                        m2_commands[m], ignored);
                }
                if (m1_runtime_ != nullptr) m1_runtime_->release(prepared[m]);
#endif
            }
        }

        // ── Phase 3: publication (no mutex — leased terminal cells + notify
        //    callbacks): terminals, per-channel notifies, then the batch notify
        //    exactly once. Always runs, so a fault above still settles here. ──
        for (std::size_t m = 0; m < M; ++m) {
            publish_terminal(job->members[m].terminal_cell, outcomes[m]);
        }
        for (const auto& [wait_id, epoch] : notifications) notify(wait_id, epoch);
        notify(job->completion.wait_id, job->completion.target_epoch);
        if (cfg_.runtime.verbose) {
            const M0TimingSnapshot timing =
                m0_timing_counters().snapshot() - timing_before;
            std::cerr
                << "[pie-driver-metal] ptir_m0_timing"
                << " cpu_epilogue_ns=" << timing.cpu_epilogue_ns
                << " cpu_epilogue_samples=" << timing.cpu_epilogue_samples
                << " bf16_conversion_ns=" << timing.bf16_conversion_ns
                << " bf16_conversion_samples="
                << timing.bf16_conversion_samples
                << " forward_wait_ns=" << timing.forward_wait_ns
                << " forward_wait_samples=" << timing.forward_wait_samples
                << " forward_wait_timeouts="
                << timing.forward_wait_timeouts
                << "\n";
#if defined(__APPLE__)
            if (m1_runtime_ != nullptr) {
                const pipeline::M3GroupStats m3_after =
                    m1_runtime_->m3_stats();
                std::cerr
                    << "[pie-driver-metal] ptir_m3"
                    << " grouped_readiness_launches="
                    << m3_after.readiness_launches -
                           m3_before.readiness_launches
                    << " grouped_body_launches="
                    << m3_after.body_launches - m3_before.body_launches
                    << " grouped_library_launches="
                    << m3_after.library_launches -
                           m3_before.library_launches
                    << " grouped_parallel_selection_launches="
                    << m3_after.parallel_selection_launches -
                           m3_before.parallel_selection_launches
                    << " grouped_fallback_launches="
                    << m3_after.singleton_fallback_launches -
                           m3_before.singleton_fallback_launches
                    << " grouped_commit_launches="
                    << m3_after.commit_launches -
                           m3_before.commit_launches
                    << " grouped_lanes="
                    << m3_after.lanes - m3_before.lanes
                    << " post_forward_critical_ns="
                    << m3_after.post_forward_critical_ns -
                           m3_before.post_forward_critical_ns
                    << "\n";
            }
#endif
        }
    }

    // Phase 1b/3 paged-KV bridge: real, page-addressable KV pool copy — see
    // MetalExecutor::copy_kv_pages/copy_kv_cells (memcpy over the SEPARATE
    // Shared-storage NHD standalone pool — genuinely page-addressable,
    // sized to exactly what caps advertises, see ensure_executor). Only the
    // Metal-resident domain is supported (no host-pinned swap pool exists
    // in this build) — a cross-domain request is honestly UNSUPPORTED
    // rather than silently misinterpreted.
    // Phase 3 (review items 1/2): control ops run their executor-touching body
    // on the worker (behind any queued launches, FIFO), so they never race an
    // in-flight forward and all MetalExecutor use stays on one thread. The
    // thin public wrapper just forwards to the worker and returns its status;
    // `worker_.run` rethrows a job exception, which the extern "C" wrapper maps
    // to DRIVER_ERROR.
    int copy_kv(const PieKvCopyDesc& copy, PieCompletion completion) {
        int status = PIE_STATUS_OK;
        worker_.run([&] { status = copy_kv_impl(copy, completion); });
        return status;
    }

    int copy_kv_impl(const PieKvCopyDesc& copy, PieCompletion completion) {
        if (!facts_.has_linear_attn) {
            // Non-hybrid architectures are out of scope for this Metal
            // increment entirely (MetalExecutor::setup refuses them) — no
            // executor, no pool, nothing to copy.
            std::cerr << "[pie-driver-metal] copy_kv: UNSUPPORTED — this increment only "
                         "supports the qwen3.6 (GDN-hybrid) checkpoint geometry\n";
            return PIE_STATUS_UNSUPPORTED;
        }
        if (copy.src_domain != PIE_MEMORY_DOMAIN_METAL_SHARED ||
            copy.dst_domain != PIE_MEMORY_DOMAIN_METAL_SHARED) {
            std::cerr << "[pie-driver-metal] copy_kv: UNSUPPORTED — only same-domain "
                         "(PIE_MEMORY_DOMAIN_METAL_SHARED) copies are supported; there is no "
                         "host-pinned swap pool in this build\n";
            return PIE_STATUS_UNSUPPORTED;
        }
        std::string err;
        if (!ensure_executor(err)) {
            std::cerr << "[pie-driver-metal] copy_kv: " << err << "\n";
            return PIE_STATUS_UNSUPPORTED;
        }
        if (executor_->kv_pool_total_pages() == 0) {
            std::cerr << "[pie-driver-metal] copy_kv: UNSUPPORTED — no paged KV pool is "
                         "allocated (config total_pages/kv_page_size produced a zero-sized "
                         "pool, or allocation failed at executor setup; see prior stderr)\n";
            return PIE_STATUS_UNSUPPORTED;
        }
        if (copy.src_page_ids.len != copy.dst_page_ids.len) {
            std::cerr << "[pie-driver-metal] copy_kv: src/dst page id count mismatch\n";
            return PIE_STATUS_INVALID_ARGUMENT;
        }
        if (copy.src_page_ids.len > 0) {
            const std::vector<std::uint32_t> src(copy.src_page_ids.ptr,
                                                  copy.src_page_ids.ptr + copy.src_page_ids.len);
            const std::vector<std::uint32_t> dst(copy.dst_page_ids.ptr,
                                                  copy.dst_page_ids.ptr + copy.dst_page_ids.len);
            std::string copy_err;
            if (!executor_->copy_kv_pages(src, dst, &copy_err)) {
                std::cerr << "[pie-driver-metal] copy_kv: " << copy_err << "\n";
                return PIE_STATUS_DRIVER_ERROR;
            }
        }
        if (copy.cells.len > 0) {
            std::vector<executor::MetalExecutor::KvMoveCell> cells;
            cells.reserve(copy.cells.len);
            for (std::size_t i = 0; i < copy.cells.len; ++i) {
                const PieKvMoveCell& c = copy.cells.ptr[i];
                cells.push_back({c.dst_page_id, c.dst_token_offset, c.src_page_id,
                                c.src_token_offset});
            }
            std::string cell_err;
            if (!executor_->copy_kv_cells(cells, &cell_err)) {
                std::cerr << "[pie-driver-metal] copy_kv: " << cell_err << "\n";
                return PIE_STATUS_DRIVER_ERROR;
            }
        }
        publish_terminal(completion.terminal_cell, PIE_TERMINAL_OUTCOME_SUCCESS);
        notify(completion.wait_id, completion.target_epoch);
        return PIE_STATUS_OK;
    }

    int copy_state(const PieStateCopyDesc& copy, PieCompletion completion) {
        int status = PIE_STATUS_OK;
        worker_.run([&] { status = copy_state_impl(copy, completion); });
        return status;
    }

    int copy_state_impl(const PieStateCopyDesc& copy, PieCompletion completion) {
        if (!facts_.has_linear_attn) {
            std::cerr << "[pie-driver-metal] copy_state: UNSUPPORTED — this checkpoint has "
                         "no GDN/linear-attention layers, so there is no recurrent state to "
                         "copy\n";
            return PIE_STATUS_UNSUPPORTED;
        }
        std::string err;
        if (!ensure_executor(err)) {
            std::cerr << "[pie-driver-metal] copy_state: " << err << "\n";
            return PIE_STATUS_UNSUPPORTED;
        }
        const std::uint32_t rs_slots = executor_->rs_slots();
        for (std::size_t i = 0; i < copy.slot_ranges.len; ++i) {
            const PieStateCopyRange& range = copy.slot_ranges.ptr[i];
            if (range.src_slot_id >= rs_slots || range.dst_slot_id >= rs_slots) {
                std::cerr << "[pie-driver-metal] copy_state: slot id out of range [0, "
                          << rs_slots << ")\n";
                return PIE_STATUS_INVALID_ARGUMENT;
            }
        }
        for (std::size_t i = 0; i < copy.slot_ranges.len; ++i) {
            const PieStateCopyRange& range = copy.slot_ranges.ptr[i];
            std::string copy_err;
            if (!executor_->copy_state(range.src_slot_id, range.dst_slot_id, &copy_err)) {
                std::cerr << "[pie-driver-metal] copy_state: " << copy_err << "\n";
                return PIE_STATUS_DRIVER_ERROR;
            }
        }
        publish_terminal(completion.terminal_cell, PIE_TERMINAL_OUTCOME_SUCCESS);
        notify(completion.wait_id, completion.target_epoch);
        return PIE_STATUS_OK;
    }

    int resize_pool(const PiePoolResizeDesc& resize, PieCompletion completion) {
        int status = PIE_STATUS_OK;
        worker_.run([&] { status = resize_pool_impl(resize, completion); });
        return status;
    }

    int resize_pool_impl(const PiePoolResizeDesc& resize, PieCompletion completion) {
        if (resize.pool_id > PIE_ELASTIC_POOL_WORKSPACE) {
            std::cerr << "[pie-driver-metal] resize_pool: unknown elastic pool id\n";
            return PIE_STATUS_UNSUPPORTED;
        }
        if (!facts_.has_linear_attn) {
            std::cerr << "[pie-driver-metal] resize_pool: UNSUPPORTED — this increment only "
                         "supports the qwen3.6 (GDN-hybrid) checkpoint geometry\n";
            return PIE_STATUS_UNSUPPORTED;
        }
        {
            std::lock_guard<std::mutex> lock(state_mutex_);
            if (false) {
                return PIE_STATUS_UNSUPPORTED;
            }
        }
        std::string err;
        if (!ensure_executor(err)) {
            std::cerr << "[pie-driver-metal] resize_pool: " << err << "\n";
            return PIE_STATUS_UNSUPPORTED;
        }
        if (resize.pool_id != PIE_ELASTIC_POOL_KV) {
            if (!executor_->resize_elastic_pool(
                    resize.pool_id,
                    resize.target_pages,
                    &err)) {
                std::cerr << "[pie-driver-metal] resize_pool: " << err << "\n";
                return PIE_STATUS_DRIVER_ERROR;
            }
            publish_terminal(
                completion.terminal_cell,
                PIE_TERMINAL_OUTCOME_SUCCESS);
            notify(completion.wait_id, completion.target_epoch);
            return PIE_STATUS_OK;
        }
        const std::uint32_t current_pages =
            executor_->kv_pool_committed_pages();
        if (current_pages == 0) {
            std::cerr << "[pie-driver-metal] resize_pool: UNSUPPORTED — no paged KV pool is "
                         "allocated (config total_pages/kv_page_size produced a zero-sized "
                         "pool, or allocation failed at executor setup; see prior stderr)\n";
            return PIE_STATUS_UNSUPPORTED;
        }
        if (resize.target_pages > std::numeric_limits<std::uint32_t>::max()) {
            std::cerr << "[pie-driver-metal] resize_pool: target_pages exceeds u32 range\n";
            return PIE_STATUS_INVALID_ARGUMENT;
        }
        const auto target_pages = static_cast<std::uint32_t>(resize.target_pages);
        const std::uint32_t allocated_pages =
            effective_total_pages(cfg_, facts_.has_linear_attn);
        if (target_pages > allocated_pages) {
            std::cerr
                << "[pie-driver-metal] resize_pool: growth beyond the fixed "
                   "paged-IO allocation is unsupported\n";
            return PIE_STATUS_UNSUPPORTED;
        }
        // A shrink is only accepted if the caller's `unmap_ranges` fully
        // covers every page in [target_pages, current_pages) — this driver
        // has no independent way to know which physical pages the runtime
        // still considers live, so it never silently truncates without
        // that explicit attestation (never "reject if REALLY needed"; a
        // partial attestation is rejected too, precisely).
        bool unmapped_tail_pages = true;
        if (target_pages < current_pages) {
            std::vector<bool> unmapped(current_pages - target_pages, false);
            for (std::size_t i = 0; i < resize.unmap_ranges.len; ++i) {
                const PiePoolRange& r = resize.unmap_ranges.ptr[i];
                for (std::uint64_t p = r.page_index; p < r.page_index + r.page_count; ++p) {
                    if (p >= target_pages && p < current_pages) unmapped[p - target_pages] = true;
                }
            }
            unmapped_tail_pages =
                std::all_of(unmapped.begin(), unmapped.end(), [](bool b) { return b; });
        }
        std::string resize_err;
        if (!executor_->resize_kv_pool(target_pages, unmapped_tail_pages, &resize_err)) {
            std::cerr << "[pie-driver-metal] resize_pool: " << resize_err << "\n";
            return target_pages < current_pages && !unmapped_tail_pages
                       ? PIE_STATUS_UNSUPPORTED
                       : PIE_STATUS_DRIVER_ERROR;
        }
        publish_terminal(completion.terminal_cell, PIE_TERMINAL_OUTCOME_SUCCESS);
        notify(completion.wait_id, completion.target_epoch);
        return PIE_STATUS_OK;
    }

    // Phase 3 (review items 1/2): a close runs on the worker BEHIND any queued
    // launches (FIFO), so `close_sequence` (executor state) runs on the owning
    // thread and can never race an in-flight forward or its settlement. The
    // map mutation takes the state mutex (shared with launch preflight).
    int close_instance(std::uint64_t instance_id) {
        int status = PIE_STATUS_OK;
        worker_.run([&] { status = close_instance_impl(instance_id); });
        return status;
    }

    int close_instance_impl(std::uint64_t instance_id) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        if (registry_.find_instance(instance_id) == nullptr) {
            return PIE_STATUS_CLOSED;
        }
        // Release Metal executor residency BEFORE erasing the instance so a
        // later, different sequence's fresh fire is not rejected as
        // "another sequence is resident" (executor::MetalExecutor
        // `validate_linear_sequence_geometry`; a no-op if this instance
        // never ran a forward, or a different sequence is resident).
        if (executor_ != nullptr) executor_->close_sequence(instance_id);
        return registry_.close_instance(instance_id);
    }

    int close_channel(std::uint64_t channel_id) {
        int status = PIE_STATUS_OK;
        worker_.run([&] { status = close_channel_impl(channel_id); });
        return status;
    }

    int close_channel_impl(std::uint64_t channel_id) {
        std::lock_guard<std::mutex> lock(state_mutex_);
        return registry_.close_channel(channel_id);
    }

  private:
    void notify(std::uint64_t wait_id, std::uint64_t epoch) const {
        if (wait_id == 0 || epoch == 0 || runtime_.notify == nullptr) return;
        runtime_.notify(runtime_.ctx, wait_id, epoch);
    }

#if !defined(__APPLE__)
    TicketPreparation prepare_member_tickets(
        const std::vector<executor::ChannelTicket>& tickets,
        std::string& failure) {
        for (const auto& ticket : tickets) {
            ChannelRecord* endpoint = registry_.find_channel(ticket.channel_id);
            if (endpoint == nullptr) {
                failure = "ticket references a closed channel";
                return TicketPreparation::Failed;
            }
            const std::uint64_t poison = endpoint->shared_state->poison();
            const std::uint64_t closed = endpoint->shared_state->closed();
            if (poison != 0 || closed != 0) {
                failure = poison != 0 ? "ticket channel is poisoned"
                                      : "ticket channel is closed";
                return TicketPreparation::Failed;
            }
            const std::uint64_t head = endpoint->shared_state->head();
            const std::uint64_t tail = endpoint->shared_state->tail();
            if (tail < head) {
                failure = "channel tail precedes head";
                return TicketPreparation::Failed;
            }
            if (ticket.expected_head != executor::kNoChannelTicket &&
                head != ticket.expected_head) {
                return TicketPreparation::Retry;
            }
            if (ticket.requires_input && tail <= head) {
                return TicketPreparation::Retry;
            }
            if (ticket.expected_tail != executor::kNoChannelTicket) {
                const std::uint64_t same_fire_consume =
                    ticket.expected_head != executor::kNoChannelTicket ? 1 : 0;
                if (tail != ticket.expected_tail ||
                    tail - head >=
                        static_cast<std::uint64_t>(endpoint->desc.capacity) +
                            same_fire_consume) {
                    return TicketPreparation::Retry;
                }
            }
        }

        return TicketPreparation::Ready;
    }
#endif

    // C2 (forward-needing) members only: lazily creates the executor on the
    // first forward-needing launch (mirrors CUDA's lazy `ptir_dispatch`,
    // §5.1). For a device-geometry (C3) program, resolves its descriptor-
    // port channels into a `FireGeometry` BEFORE the forward (Phase 2,
    // W1.1) and feeds that instead of the wire CSR fields; a not-ready
    // descriptor channel fails the fire here (no dummy-run) — the caller
    // poisons only this member. Otherwise slices the wire fields as before
    // (C2, Phase 1). Either way, runs the forward and binds `pass_in` to
    // the materialized logits. `logits_out` must outlive `pass_in`'s use.
    // Lazily create (and one-time-`setup()`) the MetalExecutor on demand —
    // shared by the forward path and the Phase 1b control ops (copy_state
    // et al.) which also need a live MetalExecutor to operate on, even if
    // no forward has run yet in this process.
    bool ensure_executor(std::string& err) {
        if (executor_ != nullptr) return true;
        executor::SetupConfig setup_cfg;
        setup_cfg.checkpoint_dir = cfg_.model.hf_path;
        setup_cfg.kernels_dir = metal_kernels_dir();
        setup_cfg.arch_name = facts_.arch_name;
        setup_cfg.vocab_size = facts_.vocab_size;
        setup_cfg.has_linear_attn = facts_.has_linear_attn;
        // Phase 1b/3 paged-KV bridge: size the REAL pool to exactly what
        // caps advertises (effective_total_pages — the same capped value
        // build_caps_json reports), so copy_kv/resize_pool never operate
        // over a pool bigger or smaller than what the driver claims to have.
        setup_cfg.total_pages = effective_total_pages(cfg_, facts_.has_linear_attn);
        setup_cfg.kv_page_size = cfg_.batching.kv_page_size;
        setup_cfg.max_forward_tokens = cfg_.batching.max_forward_tokens;
        setup_cfg.max_forward_requests = cfg_.batching.max_forward_requests;
        setup_cfg.load_plan = load_plan_bytes_;
        setup_cfg.compiler_version = compiler_version_;
        setup_cfg.storage_page_size = storage_page_size_;
        // Create + `setup()` the executor ON THE WORKER THREAD (Phase 3, §7):
        // MetalExecutor::setup builds the Metal device/heap/PSOs, which must
        // be created on the same thread that will later drive them.
        bool ok = false;
        std::string setup_err;
        std::unique_ptr<executor::MetalExecutor> candidate;
        worker_.run([&] {
            candidate = std::make_unique<executor::MetalExecutor>();
            ok = candidate->setup(setup_cfg, &setup_err);
        });
        if (!ok) {
            err = "Metal executor setup failed: " + setup_err;
            return false;
        }
        executor_ = std::move(candidate);
        return true;
    }

    // C2 (forward-needing) members only: builds the executor forward
    // descriptor for member `m`. For a device-geometry (C3) program, resolves
    // its descriptor-port channels into a `FireGeometry` BEFORE the forward
    // (Phase 2, W1.1) and feeds that instead of the wire CSR fields; a
    // not-ready descriptor channel returns a typed retry unless its producer
    // endpoint is poisoned. Otherwise slices the wire fields as
    // before (C2, Phase 1). The actual forward is run for the WHOLE batch at
    // once by `forward_batch` (Phase 3, §7); this only prepares one member's
    // descriptor + arbitrates its device geometry.
    ForwardBuildResult build_forward_desc_for_member(
        const pie_native::LaunchView& view,
        std::size_t m,
        std::size_t member_count,
        InstanceRecord& member,
        const ProgramRecord& program,
        executor::MemberForwardDesc& desc,
        std::string& failure) {
        desc.sequence_id = member.instance_id;

        pie_native::ptir::FireGeometry resolved;
        const bool device_geometry =
            interp::requires_descriptor_resolution(program.plan.trace);
        if (device_geometry) {
            const interp::GeometryResolveResult resolution =
                interp::resolve_fire_geometry_typed(
                    program.plan,
                    member.interp,
                    cfg_.batching.kv_page_size,
                    resolved,
                    &failure);
            if (resolution.status == interp::GeometryResolveStatus::NotReady) {
                if (resolution.channel >= member.channel_ids.size()) {
                    failure = "descriptor resolver returned an invalid channel id";
                    return ForwardBuildResult::Failed;
                }
                ChannelRecord* endpoint =
                    registry_.find_channel(member.channel_ids[resolution.channel]);
                if (endpoint == nullptr) {
                    failure = "descriptor channel is closed";
                    return ForwardBuildResult::Failed;
                }
                const std::uint64_t poison = endpoint->shared_state->poison();
                const std::uint64_t closed = endpoint->shared_state->closed();
                if (poison != 0 || closed != 0) {
                    failure =
                        poison != 0
                            ? "descriptor producer failed for channel " +
                                  std::to_string(resolution.channel)
                            : "descriptor channel is closed";
                    return ForwardBuildResult::Failed;
                }
                return ForwardBuildResult::Retry;
            }
            if (resolution.status == interp::GeometryResolveStatus::Failed) {
                return ForwardBuildResult::Failed;
            }
            // WorkingSet page translation (Phase 2 review fix A): apply
            // BEFORE validation, so validate_fire_geometry checks the
            // PHYSICAL page ids the forward will actually use.
            if (view.kv_translation_indptr.size() == member_count + 1) {
                const std::uint32_t lo = view.kv_translation_indptr.data()[m];
                const std::uint32_t hi = view.kv_translation_indptr.data()[m + 1];
                if (hi > lo && hi <= view.kv_translation.size()) {
                    interp::translate_kv_pages(view.kv_translation.data() + lo, hi - lo, resolved);
                }
                // hi <= lo (empty segment): legacy pass-through, ids stay physical.
            }
            const std::uint32_t device_pages =
                effective_total_pages(cfg_, facts_.has_linear_attn);
            if (!pie_native::ptir::validate_fire_geometry(
                    resolved, device_pages, cfg_.batching.kv_page_size, &failure)) {
                return ForwardBuildResult::Failed;
            }
        }
        return executor::build_member_forward_desc(
                   view,
                   m,
                   member_count,
                   facts_.has_linear_attn,
                   cfg_.batching.kv_page_size,
                   device_geometry ? &resolved : nullptr,
                   desc,
                   failure)
                   ? ForwardBuildResult::Ready
                   : ForwardBuildResult::Failed;
    }

    void queue_channel_notifications(
        const std::vector<executor::ChannelTicket>& tickets,
        std::vector<std::pair<std::uint64_t, std::uint64_t>>& notifications) {
        for (const auto& ticket : tickets) {
            ChannelRecord& endpoint =
                *registry_.find_channel(ticket.channel_id);
            if (ticket.expected_head != executor::kNoChannelTicket &&
                endpoint.desc.host_role == PIE_CHANNEL_HOST_ROLE_WRITER) {
                notifications.emplace_back(
                    endpoint.desc.writer_wait_id,
                    ticket.expected_head + 1);
            }
            if (ticket.expected_tail != executor::kNoChannelTicket &&
                endpoint.desc.host_role == PIE_CHANNEL_HOST_ROLE_READER) {
                notifications.emplace_back(
                    endpoint.desc.reader_wait_id,
                    ticket.expected_tail + 1);
            }
        }
    }

#if !defined(__APPLE__)
    MemberRunOutcome run_member(
        InstanceRecord& member,
        const ProgramRecord& program,
        const std::vector<executor::ChannelTicket>& tickets,
        const interp::PassInputs& pass_in,
        std::vector<std::pair<std::uint64_t, std::uint64_t>>& notifications,
        std::string& failure) {
        struct TimingRecorder {
            M0TimingCounters::Clock::time_point begin =
                M0TimingCounters::Clock::now();
            ~TimingRecorder() {
                m0_timing_counters().record_cpu_epilogue(
                    M0TimingCounters::Clock::now() - begin);
            }
        } timing;
        const interp::StepResult report = interp::step(member.interp, program.plan, pass_in);
        if (!report.ok) {
            failure = report.error;
            return MemberRunOutcome::Failed;
        }
        if (!report.committed) {
            return MemberRunOutcome::Retry;
        }

        // step() wrote pending values directly into the endpoint's authoritative
        // Shared-storage ring and release-published its head/tail words. Only
        // wake publication remains here.
        for (const auto& ticket : tickets) {
            ChannelRecord& endpoint = *registry_.find_channel(ticket.channel_id);
            if (ticket.expected_head != executor::kNoChannelTicket) {
                const std::uint64_t actual = ticket.expected_head + 1;
                if (endpoint.desc.host_role == PIE_CHANNEL_HOST_ROLE_WRITER) {
                    notifications.emplace_back(endpoint.desc.writer_wait_id, actual);
                }
            }
            if (ticket.expected_tail != executor::kNoChannelTicket) {
                const std::uint64_t actual = ticket.expected_tail + 1;
                if (endpoint.desc.host_role == PIE_CHANNEL_HOST_ROLE_READER) {
                    notifications.emplace_back(endpoint.desc.reader_wait_id, actual);
                }
            }
        }
        return MemberRunOutcome::Committed;
    }
#endif

    // D4 failure settlement: poison every attached channel's word (epoch
    // `tail + 1`, first poison wins device-side by construction) and queue the
    // role's wake so parked waiters observe Poisoned, not Empty.
    void poison_instance(InstanceRecord& member,
                         std::vector<std::pair<std::uint64_t, std::uint64_t>>& notifications) {
        member.interp.poisoned = true;
        for (const std::uint64_t channel_id : member.channel_ids) {
            ChannelRecord& endpoint = *registry_.find_channel(channel_id);
            const std::uint64_t tail = endpoint.shared_state->tail();
            const std::uint64_t poison_epoch = std::max<std::uint64_t>(tail + 1, 1);
            endpoint.shared_state->store_word(2, poison_epoch);
            const std::uint64_t wait_id =
                endpoint.desc.host_role == PIE_CHANNEL_HOST_ROLE_READER
                    ? endpoint.desc.reader_wait_id
                    : (endpoint.desc.host_role == PIE_CHANNEL_HOST_ROLE_WRITER
                           ? endpoint.desc.writer_wait_id
                           : 0);
            notifications.emplace_back(wait_id, poison_epoch);
        }
    }

    Config cfg_{};
    ModelFacts facts_{};
    std::string caps_json_;
    std::string device_facts_json_;
    std::vector<std::uint8_t> load_plan_bytes_;
    std::uint64_t compiler_version_ = 0;
    bool load_attempted_ = false;
    std::uint32_t storage_page_size_ = 1;
    PieRuntimeCallbacks runtime_{};
    pipeline::Registry registry_;
    // Phase 3 (§7, D4): coherent single mutex over ALL driver state (programs/
    // instances/channels + the executor pointer). Every ABI entry point locks
    // it, so registry/channel/instance mutation is race-free and a close/copy
    // can never race an in-flight forward's settlement. Declared BEFORE the
    // worker + executor so it outlives them at teardown.
    std::mutex state_mutex_;
    // Phase 3 (§7, D4): the single thread that owns every MetalExecutor /
    // RawMetalContext touch. Executor setup, forward_batch, and the copy/
    // resize control ops all run here (via `worker_.run`), giving Metal
    // command-queue thread-affinity + FIFO serialization. Declared BEFORE
    // `executor_` so the worker is still alive while `executor_` is destroyed.
    executor::ExecutorWorker worker_;
#if defined(__APPLE__)
    std::unique_ptr<pipeline::M1Runtime> m1_runtime_;
#endif
    // Lazily created on the first forward-needing (C2) launch (§5.1); zero
    // Metal dependencies leak into this translation unit beyond the plain
    // executor::MetalExecutor interface. Only ever touched on `worker_`'s
    // thread once created (see ensure_executor / the executor call sites).
    std::unique_ptr<executor::MetalExecutor> executor_;
};

Context::Context() : impl_(std::make_unique<Impl>()) {}
Context::~Context() = default;

int Context::initialize(
    const std::string& config_path,
    const PieRuntimeCallbacks& runtime) {
    return impl_->initialize(config_path, runtime);
}

void Context::fill_device_facts(PieDriverCaps* caps) const {
    impl_->fill_device_facts(caps);
}

int Context::load_model(const PieModelLoadDesc& load, PieDriverCaps* caps) {
    return impl_->load_model(load, caps);
}

int Context::register_program(
    const PieProgramDesc& program,
    std::uint64_t* program_id) {
    return impl_->register_program(program, program_id);
}

int Context::register_channel(
    const PieChannelDesc& channel,
    PieChannelEndpointBinding* binding) {
    return impl_->register_channel(channel, binding);
}

int Context::bind_instance(
    const PieInstanceDesc& instance,
    PieInstanceBinding* binding) {
    return impl_->bind_instance(instance, binding);
}

int Context::launch(const PieFrameDesc& frame, PieCompletion completion) {
    return impl_->launch(frame, completion);
}


int Context::copy_kv(const PieKvCopyDesc& copy, PieCompletion completion) {
    return impl_->copy_kv(copy, completion);
}

int Context::copy_state(const PieStateCopyDesc& copy, PieCompletion completion) {
    return impl_->copy_state(copy, completion);
}

int Context::resize_pool(
    const PiePoolResizeDesc& resize,
    PieCompletion completion) {
    return impl_->resize_pool(resize, completion);
}

int Context::close_instance(std::uint64_t instance_id) {
    return impl_->close_instance(instance_id);
}

int Context::close_channel(std::uint64_t channel_id) {
    return impl_->close_channel(channel_id);
}

}  // namespace pie::metal
