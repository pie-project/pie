#pragma once

#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iterator>
#include <memory>
#include <mutex>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include "pie_native/ptir/ptir_abi.h"
#include "pipeline/generated/fused_codegen.hpp"

namespace pie_cuda_driver::pipeline::generated {

enum class CompileFailureKind {
    None,
    Deterministic,
    Retryable,
};

struct FusedRegionExecutable {
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    std::string entry_name;

    ~FusedRegionExecutable() {
        if (module != nullptr) cuModuleUnload(module);
    }
    FusedRegionExecutable() = default;
    FusedRegionExecutable(const FusedRegionExecutable&) = delete;
    FusedRegionExecutable& operator=(const FusedRegionExecutable&) = delete;
};

struct FusedStageExecutable {
    std::uint64_t signature_hash = 0;
    std::vector<std::uint8_t> signature;
    std::vector<std::shared_ptr<FusedRegionExecutable>> regions;
};

struct FusedProgramExecutable {
    std::uint64_t program_hash = 0;
    std::vector<std::shared_ptr<FusedStageExecutable>> stages;
};

struct ModuleCacheStats {
    std::uint64_t memory_hits = 0;
    std::uint64_t compilations = 0;
    std::uint64_t disk_hits = 0;
    std::uint64_t disk_writes = 0;
    std::uint64_t disk_errors = 0;
    std::uint64_t negative_hits = 0;
    std::size_t stage_entries = 0;
    std::size_t program_entries = 0;
    std::size_t negative_entries = 0;
};

class ModuleCache {
  public:
    static constexpr std::size_t kMaximumStageEntries = 128;
    static constexpr std::size_t kMaximumProgramEntries = 128;
    static constexpr std::size_t kMaximumNegativeEntries = 128;

    ModuleCache()
        : disk_cache_directory_(default_cache_directory()) {}
    explicit ModuleCache(std::filesystem::path disk_cache_directory)
        : disk_cache_directory_(std::move(disk_cache_directory)) {}

    std::shared_ptr<const FusedProgramExecutable> compile_program(
        std::uint64_t program_hash,
        const std::vector<pie_native::ptir::plan::StagePlan>& plans,
        CompileFailureKind& failure_kind,
        std::string& error) {
        std::lock_guard<std::mutex> lock(mutex_);
        failure_kind = CompileFailureKind::None;
        error.clear();
        const auto existing_program = programs_.find(program_hash);
        if (existing_program != programs_.end()) {
            ++stats_.memory_hits;
            return existing_program->second;
        }
        if (programs_.size() >= kMaximumProgramEntries) {
            failure_kind = CompileFailureKind::Retryable;
            error = "CUDA fused program cache is at capacity";
            return nullptr;
        }

        int device = 0;
        cudaDeviceProp properties{};
        const cudaError_t device_status = cudaGetDevice(&device);
        const cudaError_t properties_status =
            device_status == cudaSuccess
                ? cudaGetDeviceProperties(&properties, device)
                : device_status;
        if (properties_status != cudaSuccess) {
            failure_kind = CompileFailureKind::Retryable;
            error = "CUDA fused compiler cannot query the active device: " +
                std::string(cudaGetErrorString(properties_status));
            return nullptr;
        }
        const std::string architecture =
            "--gpu-architecture=compute_" +
            std::to_string(properties.major) +
            std::to_string(properties.minor);
        int nvrtc_major = 0;
        int nvrtc_minor = 0;
        const nvrtcResult version_status =
            nvrtcVersion(&nvrtc_major, &nvrtc_minor);
        if (version_status != NVRTC_SUCCESS) {
            failure_kind = CompileFailureKind::Retryable;
            error = "CUDA fused compiler cannot query NVRTC version: " +
                std::string(nvrtcGetErrorString(version_status));
            return nullptr;
        }

        auto program = std::make_shared<FusedProgramExecutable>();
        program->program_hash = program_hash;
        std::vector<std::pair<std::string, std::shared_ptr<FusedStageExecutable>>>
            staged_entries;
        staged_entries.reserve(plans.size());
        for (const auto& plan : plans) {
            const std::string key = stage_key(
                plan,
                properties.major,
                properties.minor,
                nvrtc_major,
                nvrtc_minor);
            const auto negative = negative_stages_.find(key);
            if (negative != negative_stages_.end()) {
                ++stats_.negative_hits;
                failure_kind = CompileFailureKind::Deterministic;
                error = negative->second;
                return nullptr;
            }
            const auto existing_stage = stages_.find(key);
            if (existing_stage != stages_.end()) {
                if (existing_stage->second->signature_hash !=
                        plan.signature_hash ||
                    existing_stage->second->signature != plan.signature) {
                    failure_kind = CompileFailureKind::Deterministic;
                    error = "CUDA fused stage cache identity collision";
                    return nullptr;
                }
                ++stats_.memory_hits;
                program->stages.push_back(existing_stage->second);
                continue;
            }
            if (stages_.size() + staged_entries.size() >=
                kMaximumStageEntries) {
                failure_kind = CompileFailureKind::Retryable;
                error = "CUDA fused stage cache is at capacity";
                return nullptr;
            }
            auto stage = std::make_shared<FusedStageExecutable>();
            stage->signature_hash = plan.signature_hash;
            stage->signature = plan.signature;
            stage->regions.resize(plan.fused.regions.size());
            for (std::size_t region_index = 0;
                 region_index < plan.fused.regions.size();
                 ++region_index) {
                const auto& region = plan.fused.regions[region_index];
                if (region.library) continue;
                const std::string entry =
                    entry_name(plan.signature_hash, region_index);
                std::shared_ptr<FusedRegionExecutable> executable;
                if (auto ptx = load_cached_ptx(
                        key, region_index, entry)) {
                    executable = load_region(*ptx, entry, error);
                    if (executable != nullptr) {
                        ++stats_.disk_hits;
                    } else {
                        invalidate_cached_ptx(key, region_index);
                        ++stats_.disk_errors;
                        std::fprintf(
                            stderr,
                            "[pie-driver-cuda] ignoring invalid PTIR disk "
                            "cache entry: %s\n",
                            error.c_str());
                        error.clear();
                    }
                }
                if (executable == nullptr) {
                    GeneratedKernelSource source =
                        emit_fused_region_cuda(entry, plan, region);
                    if (!source.ok) {
                        failure_kind = CompileFailureKind::Deterministic;
                        error = source.error;
                        remember_negative(key, error);
                        return nullptr;
                    }
                    std::string compiled_ptx;
                    executable = compile_region(
                        source,
                        architecture,
                        failure_kind,
                        error,
                        &compiled_ptx);
                    if (executable == nullptr) {
                        if (failure_kind ==
                            CompileFailureKind::Deterministic) {
                            remember_negative(key, error);
                        }
                        return nullptr;
                    }
                    ++stats_.compilations;
                    store_cached_ptx(
                        key,
                        region_index,
                        entry,
                        compiled_ptx);
                }
                stage->regions[region_index] = std::move(executable);
            }
            if (!complete_stage_coverage(plan, *stage, error)) {
                failure_kind = CompileFailureKind::Deterministic;
                remember_negative(key, error);
                return nullptr;
            }
            staged_entries.emplace_back(key, stage);
            program->stages.push_back(std::move(stage));
        }

        for (auto& [key, stage] : staged_entries) {
            stages_.emplace(std::move(key), std::move(stage));
        }
        programs_.emplace(program_hash, program);
        update_sizes();
        return program;
    }

    std::shared_ptr<const FusedProgramExecutable> program(
        std::uint64_t program_hash) const {
        std::lock_guard<std::mutex> lock(mutex_);
        const auto found = programs_.find(program_hash);
        return found == programs_.end() ? nullptr : found->second;
    }

    ModuleCacheStats stats() const {
        std::lock_guard<std::mutex> lock(mutex_);
        ModuleCacheStats result = stats_;
        result.stage_entries = stages_.size();
        result.program_entries = programs_.size();
        result.negative_entries = negative_stages_.size();
        return result;
    }

  private:
    static std::filesystem::path default_cache_directory() {
        if (std::getenv("PIE_DISABLE_PTIR_DISK_CACHE") != nullptr) {
            return {};
        }
        if (const char* configured = std::getenv("PIE_PTIR_CACHE_DIR")) {
            return *configured == '\0'
                ? std::filesystem::path{}
                : std::filesystem::path(configured);
        }
        if (const char* xdg = std::getenv("XDG_CACHE_HOME")) {
            if (*xdg != '\0') {
                return std::filesystem::path(xdg) /
                    "pie" / "ptir-cuda";
            }
        }
        if (const char* home = std::getenv("HOME")) {
            if (*home != '\0') {
                return std::filesystem::path(home) /
                    ".cache" / "pie" / "ptir-cuda";
            }
        }
        return {};
    }

    static std::uint64_t cache_key_hash(const std::string& key) {
        std::uint64_t hash = 0xcbf29ce484222325ULL;
        for (const unsigned char value : key) {
            hash ^= value;
            hash *= 0x100000001b3ULL;
        }
        return hash;
    }

    std::filesystem::path cache_file(
        const std::string& key,
        std::size_t region_index) const {
        std::ostringstream name;
        name << std::hex << std::setfill('0') << std::setw(16)
             << cache_key_hash(key) << "-" << std::dec << region_index
             << ".ptx";
        return disk_cache_directory_ / name.str();
    }

    static void append_u32(
        std::string& output,
        std::uint32_t value) {
        for (unsigned shift = 0; shift < 32; shift += 8) {
            output.push_back(static_cast<char>(value >> shift));
        }
    }

    static void append_u64(
        std::string& output,
        std::uint64_t value) {
        for (unsigned shift = 0; shift < 64; shift += 8) {
            output.push_back(static_cast<char>(value >> shift));
        }
    }

    std::optional<std::string> load_cached_ptx(
        const std::string& key,
        std::size_t region_index,
        const std::string& entry) {
        if (disk_cache_directory_.empty()) return std::nullopt;
        const auto path = cache_file(key, region_index);
        std::error_code filesystem_error;
        if (!std::filesystem::exists(path, filesystem_error)) {
            if (filesystem_error) {
                report_disk_error(
                    "cannot inspect PTIR disk cache", path, filesystem_error);
            }
            return std::nullopt;
        }
        std::ifstream input(path, std::ios::binary | std::ios::ate);
        if (!input) {
            report_disk_error("cannot open PTIR disk cache", path, {});
            return std::nullopt;
        }
        const std::streamoff size = input.tellg();
        constexpr std::streamoff kMaximumCacheEntryBytes =
            128 * 1024 * 1024;
        if (size < 28 || size > kMaximumCacheEntryBytes) {
            invalidate_cached_ptx(key, region_index);
            report_disk_error("invalid PTIR disk cache size", path, {});
            return std::nullopt;
        }
        input.seekg(0);
        std::string bytes(static_cast<std::size_t>(size), '\0');
        input.read(bytes.data(), size);
        if (!input) {
            report_disk_error("cannot read PTIR disk cache", path, {});
            return std::nullopt;
        }
        std::size_t cursor = 0;
        auto take_u32 = [&]() -> std::optional<std::uint32_t> {
            if (cursor > bytes.size() || bytes.size() - cursor < 4) {
                return std::nullopt;
            }
            std::uint32_t value = 0;
            for (unsigned shift = 0; shift < 32; shift += 8) {
                value |= static_cast<std::uint32_t>(
                    static_cast<unsigned char>(bytes[cursor++])) << shift;
            }
            return value;
        };
        auto take_u64 = [&]() -> std::optional<std::uint64_t> {
            if (cursor > bytes.size() || bytes.size() - cursor < 8) {
                return std::nullopt;
            }
            std::uint64_t value = 0;
            for (unsigned shift = 0; shift < 64; shift += 8) {
                value |= static_cast<std::uint64_t>(
                    static_cast<unsigned char>(bytes[cursor++])) << shift;
            }
            return value;
        };
        constexpr char kMagic[] = "PTRPTX01";
        if (bytes.compare(0, 8, kMagic, 8) != 0) {
            invalidate_cached_ptx(key, region_index);
            report_disk_error("invalid PTIR disk cache magic", path, {});
            return std::nullopt;
        }
        cursor = 8;
        const auto stored_region = take_u32();
        const auto key_size = take_u32();
        const auto entry_size = take_u32();
        const auto ptx_size = take_u64();
        const bool valid_sizes =
            stored_region.has_value() &&
            key_size.has_value() &&
            entry_size.has_value() &&
            ptx_size.has_value() &&
            *stored_region == region_index &&
            *key_size == key.size() &&
            *entry_size == entry.size() &&
            *ptx_size <= bytes.size() &&
            cursor <= bytes.size() &&
            static_cast<std::uint64_t>(bytes.size() - cursor) ==
                static_cast<std::uint64_t>(*key_size) +
                *entry_size + *ptx_size;
        if (!valid_sizes ||
            bytes.compare(cursor, key.size(), key) != 0 ||
            bytes.compare(
                cursor + key.size(), entry.size(), entry) != 0) {
            invalidate_cached_ptx(key, region_index);
            report_disk_error("PTIR disk cache identity mismatch", path, {});
            return std::nullopt;
        }
        cursor += key.size() + entry.size();
        return bytes.substr(cursor, static_cast<std::size_t>(*ptx_size));
    }

    void store_cached_ptx(
        const std::string& key,
        std::size_t region_index,
        const std::string& entry,
        const std::string& ptx) {
        if (disk_cache_directory_.empty()) return;
        if (key.size() > std::numeric_limits<std::uint32_t>::max() ||
            entry.size() > std::numeric_limits<std::uint32_t>::max()) {
            ++stats_.disk_errors;
            std::fprintf(
                stderr,
                "[pie-driver-cuda] PTIR disk cache identity is too large\n");
            return;
        }
        std::error_code filesystem_error;
        std::filesystem::create_directories(
            disk_cache_directory_, filesystem_error);
        if (filesystem_error) {
            report_disk_error(
                "cannot create PTIR disk cache directory",
                disk_cache_directory_,
                filesystem_error);
            return;
        }
        std::string bytes("PTRPTX01", 8);
        append_u32(bytes, static_cast<std::uint32_t>(region_index));
        append_u32(bytes, static_cast<std::uint32_t>(key.size()));
        append_u32(bytes, static_cast<std::uint32_t>(entry.size()));
        append_u64(bytes, ptx.size());
        bytes.append(key);
        bytes.append(entry);
        bytes.append(ptx);

        const auto destination = cache_file(key, region_index);
        const std::uint64_t nonce =
            disk_cache_nonce_.fetch_add(1, std::memory_order_relaxed);
        const auto temporary = destination.string() +
            ".tmp-" + std::to_string(nonce);
        {
            std::ofstream output(
                temporary,
                std::ios::binary | std::ios::trunc);
            if (!output) {
                report_disk_error(
                    "cannot create PTIR disk cache entry", temporary, {});
                return;
            }
            output.write(bytes.data(), bytes.size());
            output.flush();
            if (!output) {
                output.close();
                std::filesystem::remove(temporary, filesystem_error);
                report_disk_error(
                    "cannot write PTIR disk cache entry", temporary, {});
                return;
            }
        }
        std::filesystem::rename(
            temporary, destination, filesystem_error);
        if (filesystem_error) {
            std::filesystem::remove(temporary, filesystem_error);
            report_disk_error(
                "cannot publish PTIR disk cache entry",
                destination,
                filesystem_error);
            return;
        }
        ++stats_.disk_writes;
    }

    void invalidate_cached_ptx(
        const std::string& key,
        std::size_t region_index) {
        if (disk_cache_directory_.empty()) return;
        std::error_code ignored;
        std::filesystem::remove(cache_file(key, region_index), ignored);
    }

    void report_disk_error(
        const char* message,
        const std::filesystem::path& path,
        const std::error_code& error) {
        ++stats_.disk_errors;
        std::fprintf(
            stderr,
            "[pie-driver-cuda] %s '%s'%s%s\n",
            message,
            path.string().c_str(),
            error ? ": " : "",
            error ? error.message().c_str() : "");
    }

    static std::string stage_key(
        const pie_native::ptir::plan::StagePlan& plan,
        int major,
        int minor,
        int nvrtc_major,
        int nvrtc_minor) {
        std::string key;
        auto add_u8 = [&](std::uint8_t value) {
            key.push_back(static_cast<char>(value));
        };
        auto add_u16 = [&](std::uint16_t value) {
            for (unsigned shift = 0; shift < 16; shift += 8) {
                add_u8(static_cast<std::uint8_t>(value >> shift));
            }
        };
        auto add_u32 = [&](std::uint32_t value) {
            for (unsigned shift = 0; shift < 32; shift += 8) {
                add_u8(static_cast<std::uint8_t>(value >> shift));
            }
        };
        auto add_u64 = [&](std::uint64_t value) {
            for (unsigned shift = 0; shift < 64; shift += 8) {
                add_u8(static_cast<std::uint8_t>(value >> shift));
            }
        };
        auto add_bytes = [&](const auto& bytes) {
            add_u32(static_cast<std::uint32_t>(bytes.size()));
            for (const auto value : bytes) {
                add_u8(static_cast<std::uint8_t>(value));
            }
        };
        const std::uint16_t compiler = PTIR_COMPILER_VERSION;
        const std::uint16_t region_plan = PTIR_REGION_PLAN_VERSION;
        const std::uint32_t lane_abi = PTIR_LANE_TABLE_ABI_VERSION;
        add_u16(compiler);
        add_u16(region_plan);
        add_u32(lane_abi);
        add_u16(kCudaGeneratedEmitterVersion);
        add_u32(static_cast<std::uint32_t>(major));
        add_u32(static_cast<std::uint32_t>(minor));
        add_u32(static_cast<std::uint32_t>(nvrtc_major));
        add_u32(static_cast<std::uint32_t>(nvrtc_minor));
        add_u8(plan.stage);
        add_u64(plan.signature_hash);
        add_bytes(plan.signature);
        add_u32(static_cast<std::uint32_t>(
            plan.channel_bindings.size()));
        add_u32(static_cast<std::uint32_t>(plan.names.size()));
        for (const auto& name : plan.names) add_bytes(name);
        add_u32(static_cast<std::uint32_t>(plan.ops.size()));
        for (const auto& normalized : plan.ops) {
            const auto& op = normalized.op;
            add_u8(op.tag);
            add_u64(static_cast<std::uint64_t>(op.chan));
            add_u16(op.name_idx);
            add_u32(static_cast<std::uint32_t>(op.args.size()));
            for (const auto argument : op.args) add_u32(argument);
            add_u32(op.results);
            add_u8(op.lit_dtype);
            add_u32(op.lit_bits);
            add_u16(op.intr);
            add_u8(op.dtype);
            add_u8(op.shape.rank);
            for (std::uint8_t dimension = 0;
                 dimension < op.shape.rank;
                 ++dimension) {
                add_u32(op.shape.dims[dimension]);
            }
            add_u32(op.imm);
            add_u32(op.imm2);
            add_u32(op.imm3);
            add_u8(op.kind);
            add_u8(op.pred_tag);
            add_u32(op.pred_payload);
        }
        add_u32(static_cast<std::uint32_t>(plan.value_types.size()));
        for (const auto& type : plan.value_types) {
            add_u8(type.dtype);
            add_u8(type.domain);
            add_u8(static_cast<std::uint8_t>(type.dims.size()));
            for (const auto& dimension : type.dims) {
                add_u8(dimension.symbolic ? 1u : 0u);
                add_u32(dimension.value);
            }
        }
        add_u8(plan.fused.kind);
        add_u8(plan.fused.whole_stage_fallback ? 1u : 0u);
        add_u32(static_cast<std::uint32_t>(
            plan.fused.regions.size()));
        for (const auto& region : plan.fused.regions) {
            add_u8(region.library ? 1u : 0u);
            add_u8(region.library_op);
            add_u8(region.schedule);
            add_u32(static_cast<std::uint32_t>(region.nodes.size()));
            for (const auto node : region.nodes) add_u32(node);
            add_u32(static_cast<std::uint32_t>(region.inputs.size()));
            for (const auto input : region.inputs) add_u32(input);
            add_u32(static_cast<std::uint32_t>(region.outputs.size()));
            for (const auto output : region.outputs) add_u32(output);
            add_u32(static_cast<std::uint32_t>(region.sinks.size()));
            for (const auto& sink : region.sinks) {
                add_u32(sink.channel_slot);
                add_u32(sink.value);
            }
        }
        return key;
    }

    static bool complete_stage_coverage(
        const pie_native::ptir::plan::StagePlan& plan,
        const FusedStageExecutable& stage,
        std::string& error) {
        if (plan.fused.whole_stage_fallback ||
            stage.regions.size() != plan.fused.regions.size()) {
            error = "CUDA fused stage requests fallback or has missing regions";
            return false;
        }
        for (const auto& normalized : plan.ops) {
            if (normalized.op.tag == PTIR_OP_INTRINSIC_VAL &&
                normalized.op.intr == PTIR_INTR_MTP_DRAFTS) {
                error =
                    "CUDA fused runtime has no direct MtpDrafts token binding";
                return false;
            }
        }
        for (std::size_t index = 0;
             index < plan.fused.regions.size();
             ++index) {
            const auto& region = plan.fused.regions[index];
            if (!region.library) {
                if (stage.regions[index] == nullptr ||
                    stage.regions[index]->function == nullptr) {
                    error = "CUDA generated fused region is unavailable";
                    return false;
                }
                continue;
            }
            if (!detail::library_region_valid(plan, region)) {
                error = "CUDA stock library region ABI is invalid";
                return false;
            }
            if (region.library_op == PTIR_LIBRARY_SECOND_PARTY) {
                error = "CUDA second-party library is unavailable";
                return false;
            }
            if (region.library_op == PTIR_LIBRARY_NUCLEUS_SAMPLE) {
                const auto& logits =
                    plan.value_types[region.inputs[0]];
                if (logits.dims.empty() ||
                    logits.dims.back().symbolic) {
                    error =
                        "CUDA nucleus library requires a static vocabulary";
                    return false;
                }
            }
        }
        return true;
    }

    static std::string entry_name(
        std::uint64_t signature_hash,
        std::size_t region_index) {
        std::ostringstream output;
        output << "ptir_fused_" << std::hex << std::setfill('0')
               << std::setw(16) << signature_hash << "_"
               << std::dec << region_index;
        return output.str();
    }

    static std::shared_ptr<FusedRegionExecutable> compile_region(
        const GeneratedKernelSource& generated,
        const std::string& architecture,
        CompileFailureKind& failure_kind,
        std::string& error,
        std::string* compiled_ptx) {
        nvrtcProgram program = nullptr;
        nvrtcResult status = nvrtcCreateProgram(
            &program,
            generated.source.c_str(),
            "ptir_fused_region.cu",
            0,
            nullptr,
            nullptr);
        if (status != NVRTC_SUCCESS) {
            failure_kind = CompileFailureKind::Retryable;
            error = "NVRTC program creation failed: " +
                std::string(nvrtcGetErrorString(status));
            return nullptr;
        }
        const char* options[] = {
            architecture.c_str(),
            "--std=c++17",
            "--fmad=false",
            "--prec-div=true",
            "--prec-sqrt=true",
        };
        status = nvrtcCompileProgram(program, std::size(options), options);
        if (status != NVRTC_SUCCESS) {
            std::size_t log_size = 0;
            nvrtcGetProgramLogSize(program, &log_size);
            std::string log(log_size, '\0');
            if (log_size != 0) nvrtcGetProgramLog(program, log.data());
            nvrtcDestroyProgram(&program);
            failure_kind =
                status == NVRTC_ERROR_COMPILATION
                    ? CompileFailureKind::Deterministic
                    : CompileFailureKind::Retryable;
            error = "NVRTC fused compilation failed: " + log;
            return nullptr;
        }
        std::size_t ptx_size = 0;
        status = nvrtcGetPTXSize(program, &ptx_size);
        if (status != NVRTC_SUCCESS) {
            nvrtcDestroyProgram(&program);
            failure_kind = CompileFailureKind::Retryable;
            error = "NVRTC fused PTX sizing failed: " +
                std::string(nvrtcGetErrorString(status));
            return nullptr;
        }
        std::string ptx(ptx_size, '\0');
        status = nvrtcGetPTX(program, ptx.data());
        nvrtcDestroyProgram(&program);
        if (status != NVRTC_SUCCESS) {
            failure_kind = CompileFailureKind::Retryable;
            error = "NVRTC fused PTX extraction failed: " +
                std::string(nvrtcGetErrorString(status));
            return nullptr;
        }
        if (compiled_ptx != nullptr) {
            *compiled_ptx = ptx;
        }
        auto executable =
            load_region(ptx, generated.entry_name, error);
        if (executable == nullptr) {
            failure_kind = CompileFailureKind::Retryable;
        }
        return executable;
    }

    static std::shared_ptr<FusedRegionExecutable> load_region(
        const std::string& ptx,
        const std::string& entry_name,
        std::string& error) {
        auto executable = std::make_shared<FusedRegionExecutable>();
        executable->entry_name = entry_name;
        CUresult driver_status =
            cuModuleLoadData(&executable->module, ptx.data());
        if (driver_status == CUDA_SUCCESS) {
            driver_status = cuModuleGetFunction(
                &executable->function,
                executable->module,
                executable->entry_name.c_str());
        }
        if (driver_status != CUDA_SUCCESS) {
            const char* message = nullptr;
            cuGetErrorString(driver_status, &message);
            error = "CUDA fused module load failed: " +
                std::string(
                    message == nullptr ? "unknown driver error" : message);
            return nullptr;
        }
        return executable;
    }

    void update_sizes() {
        stats_.stage_entries = stages_.size();
        stats_.program_entries = programs_.size();
        stats_.negative_entries = negative_stages_.size();
    }

    void remember_negative(
        const std::string& key,
        const std::string& error) {
        if (negative_stages_.size() >= kMaximumNegativeEntries) return;
        negative_stages_.try_emplace(key, error);
        stats_.negative_entries = negative_stages_.size();
    }

    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<FusedStageExecutable>>
        stages_;
    std::unordered_map<std::uint64_t, std::shared_ptr<FusedProgramExecutable>>
        programs_;
    std::unordered_map<std::string, std::string> negative_stages_;
    std::filesystem::path disk_cache_directory_;
    inline static std::atomic<std::uint64_t> disk_cache_nonce_{0};
    ModuleCacheStats stats_{};
};

}  // namespace pie_cuda_driver::pipeline::generated
