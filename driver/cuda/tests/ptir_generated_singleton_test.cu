#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iterator>
#include <limits>
#include <map>
#include <memory>
#include <set>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include <cuda.h>
#include <cuda_runtime.h>
#include <nvrtc.h>

#include "pie_native/ptir/bound.hpp"
#include "pie_native/ptir/plan.hpp"
#include "pipeline/channels.hpp"
#include "pipeline/generated/fused_codegen.hpp"
#include "pipeline/generated/fused_runtime.cuh"
#include "pipeline/generated/module_cache.hpp"
#include "pipeline/generated/singleton_codegen.hpp"

using namespace pie_cuda_driver::pipeline;
using namespace pie_cuda_driver::pipeline::generated;
using namespace pie_native::ptir;

namespace {

int failures = 0;

void expect(bool condition, const std::string& message) {
    if (condition) return;
    ++failures;
    std::fprintf(stderr, "FAIL: %s\n", message.c_str());
}

void check_cuda(cudaError_t result, const char* expression) {
    if (result == cudaSuccess) return;
    throw std::runtime_error(
        std::string(expression) + ": " + cudaGetErrorString(result));
}

void check_cu(CUresult result, const char* expression) {
    if (result == CUDA_SUCCESS) return;
    const char* message = nullptr;
    cuGetErrorString(result, &message);
    throw std::runtime_error(
        std::string(expression) + ": " +
        (message == nullptr ? "unknown CUDA driver error" : message));
}

void check_nvrtc(nvrtcResult result, const char* expression) {
    if (result == NVRTC_SUCCESS) return;
    throw std::runtime_error(
        std::string(expression) + ": " + nvrtcGetErrorString(result));
}

#define CUDA_OK(expression) check_cuda((expression), #expression)
#define CU_OK(expression) check_cu((expression), #expression)
#define NVRTC_OK(expression) check_nvrtc((expression), #expression)

std::string trim(const std::string& value) {
    const std::size_t first = value.find_first_not_of(" \t\r\n");
    if (first == std::string::npos) return {};
    const std::size_t last = value.find_last_not_of(" \t\r\n");
    return value.substr(first, last - first + 1);
}

std::vector<std::uint8_t> hex_bytes(const std::string& value) {
    std::vector<std::uint8_t> bytes;
    bytes.reserve(value.size() / 2);
    for (std::size_t index = 0; index + 1 < value.size(); index += 2) {
        bytes.push_back(static_cast<std::uint8_t>(
            std::stoul(value.substr(index, 2), nullptr, 16)));
    }
    return bytes;
}

struct Fixture {
    std::string name;
    std::string verdict;
    std::uint64_t hash = 0;
    std::vector<std::uint8_t> canonical;
    std::vector<std::uint8_t> sidecar;
    Trace trace;
    std::vector<plan::StagePlan> plans;
    bool trace_valid = false;
};

bool decode_fixture(Fixture& fixture) {
    if (fixture.verdict != "OK") return true;
    if (fixture.canonical.empty() || fixture.sidecar.empty()) return false;

    container::Container decoded_container;
    container::DecodeError container_error;
    if (!container::decode(
            fixture.canonical.data(),
            fixture.canonical.size(),
            decoded_container,
            &container_error)) {
        std::fprintf(
            stderr,
            "%s: container decode: %s\n",
            fixture.name.c_str(),
            container_error.detail.c_str());
        return false;
    }
    bound::Bound decoded_bound;
    std::string error;
    if (!bound::parse_sidecar(
            fixture.sidecar.data(),
            fixture.sidecar.size(),
            decoded_bound,
            &error) ||
        decoded_container.hash != fixture.hash ||
        decoded_bound.container_hash != fixture.hash) {
        std::fprintf(
            stderr,
            "%s: sidecar/identity decode: %s\n",
            fixture.name.c_str(),
            error.c_str());
        return false;
    }
    if (decoded_bound.plans.size() != decoded_container.stages.size()) {
        std::fprintf(
            stderr,
            "%s: plan/stage mismatch (plans=%zu stages=%zu)\n",
            fixture.name.c_str(),
            decoded_bound.plans.size(),
            decoded_container.stages.size());
        return false;
    }
    fixture.plans.reserve(decoded_bound.plans.size());
    for (const auto& encoded : decoded_bound.plans) {
        plan::StagePlan stage;
        if (!plan::decode(
                encoded.bytes.data(), encoded.bytes.size(), stage, &error)) {
            std::fprintf(
                stderr,
                "%s: PTRP decode: %s\n",
                fixture.name.c_str(),
                error.c_str());
            return false;
        }
        fixture.plans.push_back(std::move(stage));
    }
    auto translated =
        bound::container_to_trace(decoded_container, decoded_bound);
    if (translated.ok) {
        fixture.trace = std::move(translated.trace);
        fixture.trace_valid = true;
    } else if (
        translated.error.find("second-party kernel") ==
        std::string::npos) {
        std::fprintf(
            stderr,
            "%s: trace translation: %s\n",
            fixture.name.c_str(),
            translated.error.c_str());
        return false;
    }
    return true;
}

bool load_fixtures(
    const std::filesystem::path& path,
    std::vector<Fixture>& fixtures) {
    std::ifstream input(path);
    if (!input) return false;
    Fixture fixture;
    std::string canonical;
    std::string sidecar;
    auto finish = [&]() {
        if (fixture.name.empty()) return true;
        fixture.canonical = hex_bytes(canonical);
        fixture.sidecar = hex_bytes(sidecar);
        if (!decode_fixture(fixture)) return false;
        fixtures.push_back(std::move(fixture));
        fixture = {};
        canonical.clear();
        sidecar.clear();
        return true;
    };
    std::string line;
    while (std::getline(input, line)) {
        const std::size_t separator = line.find(':');
        if (separator == std::string::npos) continue;
        const std::string key = trim(line.substr(0, separator));
        const std::string value = trim(line.substr(separator + 1));
        if (key == "name") {
            if (!finish()) return false;
            fixture.name = value;
        } else if (key == "verdict") {
            fixture.verdict = value;
        } else if (key == "hash") {
            fixture.hash = std::stoull(value, nullptr, 16);
        } else if (key == "container") {
            canonical = value;
        } else if (key == "sidecar") {
            sidecar = value;
        }
    }
    return finish();
}

struct DeviceBuffer {
    void* pointer = nullptr;
    std::size_t bytes = 0;

    DeviceBuffer() = default;
    explicit DeviceBuffer(std::size_t size) : bytes(std::max<std::size_t>(size, 1)) {
        CUDA_OK(cudaMalloc(&pointer, bytes));
    }
    ~DeviceBuffer() {
        if (pointer != nullptr) cudaFree(pointer);
    }
    DeviceBuffer(const DeviceBuffer&) = delete;
    DeviceBuffer& operator=(const DeviceBuffer&) = delete;
    DeviceBuffer(DeviceBuffer&& other) noexcept
        : pointer(std::exchange(other.pointer, nullptr)),
          bytes(std::exchange(other.bytes, 0)) {}
    DeviceBuffer& operator=(DeviceBuffer&& other) noexcept {
        if (this == &other) return *this;
        if (pointer != nullptr) cudaFree(pointer);
        pointer = std::exchange(other.pointer, nullptr);
        bytes = std::exchange(other.bytes, 0);
        return *this;
    }
};

struct CompiledKernel {
    CUmodule module = nullptr;
    CUfunction function = nullptr;
    std::string entry_name;

    ~CompiledKernel() {
        if (module != nullptr) cuModuleUnload(module);
    }
    CompiledKernel(const CompiledKernel&) = delete;
    CompiledKernel& operator=(const CompiledKernel&) = delete;
    CompiledKernel() = default;
};

std::string architecture_option() {
    int device = 0;
    cudaDeviceProp properties{};
    CUDA_OK(cudaGetDevice(&device));
    CUDA_OK(cudaGetDeviceProperties(&properties, device));
    return "--gpu-architecture=compute_" +
        std::to_string(properties.major) +
        std::to_string(properties.minor);
}

std::unique_ptr<CompiledKernel> compile_source(
    const GeneratedKernelSource& generated,
    const std::string& label) {
    if (!generated.ok) {
        throw std::runtime_error(
            label + " source emission failed: " + generated.error);
    }

    nvrtcProgram program = nullptr;
    NVRTC_OK(nvrtcCreateProgram(
        &program,
        generated.source.c_str(),
        "ptir_generated.cu",
        0,
        nullptr,
        nullptr));
    const std::string architecture = architecture_option();
    const char* options[] = {
        architecture.c_str(),
        "--std=c++17",
        "--fmad=false",
        "--prec-div=true",
        "--prec-sqrt=true",
    };
    const nvrtcResult compile_result =
        nvrtcCompileProgram(program, std::size(options), options);
    if (compile_result != NVRTC_SUCCESS) {
        std::size_t log_size = 0;
        nvrtcGetProgramLogSize(program, &log_size);
        std::string log(log_size, '\0');
        if (log_size != 0) nvrtcGetProgramLog(program, log.data());
        nvrtcDestroyProgram(&program);
        throw std::runtime_error(
            "NVRTC " + label + " compile failed:\n" + log);
    }
    std::size_t ptx_size = 0;
    NVRTC_OK(nvrtcGetPTXSize(program, &ptx_size));
    std::string ptx(ptx_size, '\0');
    NVRTC_OK(nvrtcGetPTX(program, ptx.data()));
    NVRTC_OK(nvrtcDestroyProgram(&program));

    auto result = std::make_unique<CompiledKernel>();
    result->entry_name = generated.entry_name;
    CU_OK(cuModuleLoadData(&result->module, ptx.data()));
    CU_OK(cuModuleGetFunction(
        &result->function,
        result->module,
        result->entry_name.c_str()));
    return result;
}

std::unique_ptr<CompiledKernel> compile_kernel(std::uint8_t tag) {
    const std::string entry =
        "ptir_generated_singleton_" + std::to_string(tag);
    return compile_source(
        emit_singleton_region_cuda(entry, tag),
        "singleton tag " + std::to_string(tag));
}

GeneratedStatus launch_raw(
    const CompiledKernel& kernel,
    const std::vector<GeneratedValueDesc>& descriptors,
    const GeneratedOpParams& params,
    const void* a0,
    const void* a1,
    const void* a2,
    void* o0,
    void* o1) {
    DeviceBuffer descriptor_device(
        descriptors.size() * sizeof(GeneratedValueDesc));
    CUDA_OK(cudaMemcpy(
        descriptor_device.pointer,
        descriptors.data(),
        descriptor_device.bytes,
        cudaMemcpyHostToDevice));
    DeviceBuffer params_device(sizeof(params));
    CUDA_OK(cudaMemcpy(
        params_device.pointer,
        &params,
        sizeof(params),
        cudaMemcpyHostToDevice));
    DeviceBuffer status_device(sizeof(GeneratedStatus));
    GeneratedStatus status{.state = 1};
    CUDA_OK(cudaMemcpy(
        status_device.pointer,
        &status,
        sizeof(status),
        cudaMemcpyHostToDevice));
    std::size_t maximum_length = 1;
    for (const auto& descriptor : descriptors) {
        maximum_length =
            std::max<std::size_t>(maximum_length, descriptor.len);
    }
    DeviceBuffer temporary(maximum_length * 16);
    DeviceBuffer dummy(16);
    CUdeviceptr status_pointer =
        reinterpret_cast<CUdeviceptr>(status_device.pointer);
    CUdeviceptr descriptor_pointer =
        reinterpret_cast<CUdeviceptr>(descriptor_device.pointer);
    CUdeviceptr a0_pointer = reinterpret_cast<CUdeviceptr>(
        a0 == nullptr ? dummy.pointer : a0);
    CUdeviceptr a1_pointer = reinterpret_cast<CUdeviceptr>(
        a1 == nullptr ? dummy.pointer : a1);
    CUdeviceptr a2_pointer = reinterpret_cast<CUdeviceptr>(
        a2 == nullptr ? dummy.pointer : a2);
    CUdeviceptr o0_pointer = reinterpret_cast<CUdeviceptr>(
        o0 == nullptr ? dummy.pointer : o0);
    CUdeviceptr o1_pointer = reinterpret_cast<CUdeviceptr>(
        o1 == nullptr ? dummy.pointer : o1);
    CUdeviceptr temporary_pointer =
        reinterpret_cast<CUdeviceptr>(temporary.pointer);
    CUdeviceptr params_pointer =
        reinterpret_cast<CUdeviceptr>(params_device.pointer);
    void* arguments[] = {
        &status_pointer,
        &descriptor_pointer,
        &a0_pointer,
        &a1_pointer,
        &a2_pointer,
        &o0_pointer,
        &o1_pointer,
        &temporary_pointer,
        &params_pointer,
    };
    CU_OK(cuLaunchKernel(
        kernel.function,
        1, 1, 1,
        1, 1, 1,
        0,
        nullptr,
        arguments,
        nullptr));
    CU_OK(cuCtxSynchronize());
    CUDA_OK(cudaMemcpy(
        &status,
        status_device.pointer,
        sizeof(status),
        cudaMemcpyDeviceToHost));
    return status;
}

struct RuntimeExtents {
    std::uint32_t kv_len = 1;
    std::uint32_t page_count = 1;
    std::uint32_t row_count = 1;
    std::uint32_t token_count = 1;
    std::uint32_t sampled_rows = 1;
    std::uint32_t query_len = 1;
    std::uint32_t key_len = 1;
};

struct GeneratedInputs {
    const void* logits = nullptr;
    const void* mtp_logits = nullptr;
    const void* mtp_drafts = nullptr;
    const void* hidden = nullptr;
    const void* query = nullptr;
    const void* value_head = nullptr;
    const std::uint32_t* layer = nullptr;
    std::uint32_t intrinsic_dtype = 1;
    std::uint32_t auxiliary_intrinsic_dtype = 0;
    std::uint32_t vocab = 0;
    std::uint32_t row_stride = 0;
    std::uint32_t auxiliary_row_stride = 0;
    std::uint32_t logits_row_offset = 0;
    std::uint32_t mtp_logits_row_offset = 0;
    std::uint32_t hidden_row_offset = 0;
    std::uint32_t query_row_offset = 0;
    std::uint32_t value_head_row_offset = 0;
    std::uint64_t rng_seed = 0;
};

std::uint32_t resolve_extent(
    std::uint32_t role,
    const RuntimeExtents& extents) {
    switch (role) {
        case PTIR_EXTENT_KV_LEN: return extents.kv_len;
        case PTIR_EXTENT_PAGE_COUNT: return extents.page_count;
        case PTIR_EXTENT_ROW_COUNT: return extents.row_count;
        case PTIR_EXTENT_TOKEN_COUNT: return extents.token_count;
        case PTIR_EXTENT_SAMPLED_ROWS: return extents.sampled_rows;
        case PTIR_EXTENT_QUERY_LEN: return extents.query_len;
        case PTIR_EXTENT_KEY_LEN: return extents.key_len;
        default: return 0;
    }
}

bool describe_value(
    const plan::ValueType& type,
    const RuntimeExtents& extents,
    GeneratedValueDesc& descriptor) {
    descriptor = {};
    descriptor.dtype = type.dtype;
    descriptor.rank = static_cast<std::uint32_t>(type.dims.size());
    std::uint64_t length = 1;
    for (std::size_t index = 0; index < type.dims.size(); ++index) {
        const auto& dimension = type.dims[index];
        const std::uint32_t value = dimension.symbolic
            ? resolve_extent(dimension.value, extents)
            : dimension.value;
        if (value == 0 ||
            length > std::numeric_limits<std::uint32_t>::max() / value) {
            return false;
        }
        descriptor.dims[index] = value;
        length *= value;
    }
    descriptor.len = static_cast<std::uint32_t>(length);
    descriptor.rows = 1;
    if (descriptor.rank >= 2) {
        std::uint64_t rows = 1;
        for (std::uint32_t index = 0; index + 1 < descriptor.rank; ++index) {
            if (rows >
                std::numeric_limits<std::uint32_t>::max() /
                    descriptor.dims[index]) {
                return false;
            }
            rows *= descriptor.dims[index];
        }
        descriptor.rows = static_cast<std::uint32_t>(rows);
    }
    descriptor.last = descriptor.len / descriptor.rows;
    return true;
}

std::size_t value_bytes(const GeneratedValueDesc& descriptor) {
    return std::max<std::size_t>(
        descriptor.dtype == PTIR_DT_BOOL
            ? descriptor.len
            : static_cast<std::size_t>(descriptor.len) * 4,
        4);
}

void upload_indices(
    const std::vector<std::uint32_t>& values,
    DeviceBuffer& buffer) {
    if (values.empty()) return;
    buffer = DeviceBuffer(values.size() * sizeof(std::uint32_t));
    CUDA_OK(cudaMemcpy(
        buffer.pointer,
        values.data(),
        buffer.bytes,
        cudaMemcpyHostToDevice));
}

void deduplicate(std::vector<std::uint32_t>& values) {
    std::sort(values.begin(), values.end());
    values.erase(std::unique(values.begin(), values.end()), values.end());
}

class GeneratedSingletonExecutor {
  public:
    GeneratedSingletonExecutor(
        const Trace& trace,
        const std::vector<plan::StagePlan>& plans,
        const std::map<std::uint8_t, std::unique_ptr<CompiledKernel>>& kernels)
        : trace_(trace), plans_(plans), kernels_(kernels) {
        arena_.init(trace_.channels);
    }

    void seed(
        ChannelId channel,
        const void* bytes,
        std::size_t size) {
        arena_.seed_cell(channel, bytes, size);
    }

    void feed(
        ChannelId channel,
        const void* bytes,
        std::size_t size) {
        arena_.host_feed(channel, bytes, size);
    }

    bool output_ready(ChannelId channel) {
        return arena_.committed_full(channel);
    }

    void take(
        ChannelId channel,
        void* bytes,
        std::size_t size) {
        arena_.host_take(channel, bytes, size);
    }

    bool run(
        const void* logits,
        std::uint32_t intrinsic_dtype,
        std::uint32_t vocab,
        const RuntimeExtents& extents,
        std::uint64_t rng_seed,
        std::string& error) {
        return run(
            GeneratedInputs{
                .logits = logits,
                .intrinsic_dtype = intrinsic_dtype,
                .vocab = vocab,
                .row_stride = vocab,
                .rng_seed = rng_seed,
            },
            extents,
            error);
    }

    bool run(
        const GeneratedInputs& inputs,
        const RuntimeExtents& extents,
        std::string& error) {
        const std::uint32_t one = 1;
        CUDA_OK(cudaMemcpy(
            arena_.d_commit(),
            &one,
            sizeof(one),
            cudaMemcpyHostToDevice));

        std::vector<std::uint32_t> descriptor_full;
        std::vector<std::uint32_t> descriptor_taken;
        for (const PortBinding& binding : trace_.ports) {
            if (binding.is_const) continue;
            descriptor_full.push_back(binding.channel);
            if (port_consumes(binding.port)) {
                descriptor_taken.push_back(binding.channel);
            }
        }
        if (!launch_readiness(descriptor_full, {})) return false;

        std::vector<bool> pending(trace_.channels.size(), false);
        std::vector<bool> prior_taken(trace_.channels.size(), false);
        for (const std::uint32_t channel : descriptor_taken) {
            prior_taken[channel] = true;
        }
        std::vector<std::uint32_t> pass_taken = descriptor_taken;
        std::vector<std::uint32_t> pass_put;
        if (plans_.size() != trace_.stages.size()) {
            error = "trace/region-plan stage count mismatch";
            return false;
        }
        for (std::size_t stage_index = 0;
             stage_index < plans_.size();
             ++stage_index) {
            std::vector<std::uint32_t> need_full;
            std::vector<std::uint32_t> need_empty;
            std::vector<std::uint32_t> taken;
            collect_stage_channels(
                trace_.stages[stage_index],
                descriptor_taken,
                pending,
                prior_taken,
                need_full,
                need_empty,
                taken);
            if (!launch_readiness(need_full, need_empty)) return false;
            pass_taken.insert(
                pass_taken.end(), taken.begin(), taken.end());
            for (const std::uint32_t channel : taken) {
                prior_taken[channel] = true;
            }
            for (const ChannelPut& put :
                 trace_.stages[stage_index].puts) {
                pass_put.push_back(put.channel);
            }
            if (!execute_stage(
                    plans_[stage_index],
                    pending,
                    inputs,
                    extents,
                    error)) {
                const std::uint32_t zero = 0;
                CUDA_OK(cudaMemcpy(
                    arena_.d_commit(),
                    &zero,
                    sizeof(zero),
                    cudaMemcpyHostToDevice));
                return false;
            }
        }

        deduplicate(pass_taken);
        deduplicate(pass_put);
        DeviceBuffer taken_device;
        DeviceBuffer put_device;
        upload_indices(pass_taken, taken_device);
        upload_indices(pass_put, put_device);
        k_commit_bump<<<1, 1>>>(
            arena_.d_full(),
            arena_.d_head(),
            arena_.d_tail(),
            arena_.d_cap1(),
            static_cast<const std::uint32_t*>(taken_device.pointer),
            static_cast<std::uint32_t>(pass_taken.size()),
            static_cast<const std::uint32_t*>(put_device.pointer),
            static_cast<std::uint32_t>(pass_put.size()),
            arena_.d_commit());
        CUDA_OK(cudaGetLastError());
        std::uint32_t committed = 0;
        CUDA_OK(cudaMemcpy(
            &committed,
            arena_.d_commit(),
            sizeof(committed),
            cudaMemcpyDeviceToHost));
        if (committed != 0) arena_.sync_host_rings();
        return committed != 0;
    }

  private:
    void collect_stage_channels(
        const Stage& stage,
        const std::vector<std::uint32_t>& descriptor_taken,
        const std::vector<bool>& prior_put,
        const std::vector<bool>& prior_taken,
        std::vector<std::uint32_t>& need_full,
        std::vector<std::uint32_t>& need_empty,
        std::vector<std::uint32_t>& taken) const {
        std::vector<std::uint8_t> full(trace_.channels.size(), 0);
        std::vector<std::uint8_t> consumed(trace_.channels.size(), 0);
        std::vector<std::uint8_t> descriptor_consumed(
            trace_.channels.size(), 0);
        for (const std::uint32_t channel : descriptor_taken) {
            if (channel < descriptor_consumed.size()) {
                descriptor_consumed[channel] = 1;
            }
        }
        for (const Op& op : stage.ops) {
            for (const ValueId argument : op.args) {
                const Value* value = trace_.value(argument);
                if (value == nullptr) continue;
                if (value->source == ValueSource::ChannelTake) {
                    full[value->channel] = 1;
                    consumed[value->channel] = 1;
                } else if (
                    value->source == ValueSource::ChannelRead) {
                    full[value->channel] = 1;
                }
            }
        }
        for (const ChannelId channel : stage.takes) {
            full[channel] = 1;
            consumed[channel] = 1;
        }
        for (const ChannelId channel : stage.reads) {
            full[channel] = 1;
        }
        for (std::size_t channel = 0;
             channel < trace_.channels.size();
             ++channel) {
            if (full[channel] != 0 && !prior_put[channel]) {
                need_full.push_back(
                    static_cast<std::uint32_t>(channel));
            }
            if (consumed[channel] != 0) {
                taken.push_back(
                    static_cast<std::uint32_t>(channel));
            }
        }
        for (const ChannelPut& put : stage.puts) {
            if (full[put.channel] == 0 &&
                descriptor_consumed[put.channel] == 0 &&
                !prior_put[put.channel] &&
                !prior_taken[put.channel]) {
                need_empty.push_back(put.channel);
            }
        }
    }

    bool launch_readiness(
        const std::vector<std::uint32_t>& need_full,
        const std::vector<std::uint32_t>& need_empty) {
        DeviceBuffer full_device;
        DeviceBuffer empty_device;
        upload_indices(need_full, full_device);
        upload_indices(need_empty, empty_device);
        k_stage_readiness<<<1, 1>>>(
            arena_.d_full(),
            arena_.d_head(),
            arena_.d_tail(),
            arena_.d_cap1(),
            static_cast<const std::uint32_t*>(full_device.pointer),
            static_cast<std::uint32_t>(need_full.size()),
            static_cast<const std::uint32_t*>(empty_device.pointer),
            static_cast<std::uint32_t>(need_empty.size()),
            arena_.d_commit());
        CUDA_OK(cudaGetLastError());
        std::uint32_t ready = 0;
        CUDA_OK(cudaMemcpy(
            &ready,
            arena_.d_commit(),
            sizeof(ready),
            cudaMemcpyDeviceToHost));
        return ready != 0;
    }

    bool execute_stage(
        const plan::StagePlan& stage,
        std::vector<bool>& pending,
        const GeneratedInputs& inputs,
        const RuntimeExtents& extents,
        std::string& error) {
        std::vector<GeneratedOpMeta> operations;
        if (!validate_singleton_plan(stage, operations, error)) {
            return false;
        }
        if (operations.empty()) return true;
        std::vector<GeneratedValueDesc> descriptors;
        descriptors.reserve(stage.value_types.size());
        std::size_t maximum_length = 1;
        for (const auto& type : stage.value_types) {
            GeneratedValueDesc descriptor;
            if (!describe_value(type, extents, descriptor)) {
                error = "resolved generated value shape exceeds u32";
                return false;
            }
            maximum_length =
                std::max<std::size_t>(maximum_length, descriptor.len);
            descriptors.push_back(descriptor);
        }
        DeviceBuffer descriptor_device(
            descriptors.size() * sizeof(GeneratedValueDesc));
        CUDA_OK(cudaMemcpy(
            descriptor_device.pointer,
            descriptors.data(),
            descriptor_device.bytes,
            cudaMemcpyHostToDevice));
        std::vector<DeviceBuffer> values;
        values.reserve(descriptors.size());
        for (const auto& descriptor : descriptors) {
            values.emplace_back(value_bytes(descriptor));
            CUDA_OK(cudaMemset(
                values.back().pointer, 0, values.back().bytes));
        }
        DeviceBuffer temporary(maximum_length * 16);
        DeviceBuffer dummy(16);
        DeviceBuffer status_device(sizeof(GeneratedStatus));
        DeviceBuffer params_device(sizeof(GeneratedOpParams));
        GeneratedStatus status{};
        status.state = 1;
        CUDA_OK(cudaMemcpy(
            status_device.pointer,
            &status,
            sizeof(status),
            cudaMemcpyHostToDevice));

        for (const GeneratedOpMeta& metadata : operations) {
            const auto& op = metadata.op;
            GeneratedOpParams params{};
            params.tag = op.tag;
            params.a0 = !op.args.empty() ? op.args[0] : 0;
            params.a1 = op.args.size() > 1
                ? op.args[1]
                : (op.tag == PTIR_OP_PIVOT_THRESHOLD
                       ? op.pred_payload
                       : 0);
            params.a2 = op.args.size() > 2 ? op.args[2] : 0;
            params.o0 = op.results > 0
                ? metadata.result_base
                : params.a0;
            params.o1 = op.results > 1
                ? params.o0 + 1
                : params.o0;
            params.imm = op.imm;
            if (op.tag == PTIR_OP_INTRINSIC_VAL) {
                params.imm =
                    op.intr == PTIR_INTR_LOGITS ||
                            op.intr == PTIR_INTR_MTP_LOGITS
                        ? inputs.vocab
                        : descriptors[params.o0].last;
            }
            params.imm2 = op.imm2;
            params.imm3 = op.imm3;
            params.kind = op.kind;
            params.pred_tag = op.pred_tag;
            params.lit_dtype = op.lit_dtype;
            params.lit_bits = op.lit_bits;
            params.channel_slot =
                op.chan >= 0 ? static_cast<std::uint32_t>(op.chan) : 0;
            params.intr = op.intr;
            params.intrinsic_dtype = inputs.intrinsic_dtype;
            params.bool_storage = 0;
            params.intrinsic_row_stride =
                inputs.row_stride == 0
                    ? inputs.vocab
                    : inputs.row_stride;
            params.rng_seed = inputs.rng_seed;

            void* a0 = dummy.pointer;
            void* a1 = dummy.pointer;
            void* a2 = dummy.pointer;
            void* o0 = dummy.pointer;
            void* o1 = dummy.pointer;
            if (!op.args.empty()) a0 = values[params.a0].pointer;
            if (op.args.size() > 1 ||
                op.tag == PTIR_OP_PIVOT_THRESHOLD) {
                a1 = values[params.a1].pointer;
            }
            if (op.args.size() > 2) a2 = values[params.a2].pointer;
            if (op.results > 0) o0 = values[params.o0].pointer;
            if (op.results > 1) o1 = values[params.o1].pointer;

            if (op.tag == PTIR_OP_CHAN_TAKE ||
                op.tag == PTIR_OP_CHAN_READ) {
                const std::size_t dense =
                    stage.channel_bindings.at(op.chan);
                a0 = pending[dense]
                    ? arena_.pending_cell(dense)
                    : arena_.committed_cell(dense);
            } else if (op.tag == PTIR_OP_CHAN_PUT) {
                const std::size_t dense =
                    stage.channel_bindings.at(op.chan);
                params.sink_bytes = static_cast<std::uint32_t>(
                    arena_.cell_bytes(dense));
                o0 = arena_.pending_cell(dense);
            } else if (op.tag == PTIR_OP_INTRINSIC_VAL) {
                switch (op.intr) {
                    case PTIR_INTR_LOGITS:
                        a0 = const_cast<void*>(inputs.logits);
                        params.intrinsic_row_offset =
                            inputs.logits_row_offset;
                        break;
                    case PTIR_INTR_MTP_LOGITS:
                        a0 = const_cast<void*>(
                            inputs.mtp_logits != nullptr
                                ? inputs.mtp_logits
                                : inputs.logits);
                        params.intrinsic_row_offset =
                            inputs.mtp_logits_row_offset;
                        break;
                    case PTIR_INTR_MTP_DRAFTS:
                        a0 = const_cast<void*>(inputs.mtp_drafts);
                        params.intrinsic_dtype = 0;
                        params.intrinsic_row_stride = 1;
                        params.intrinsic_row_offset = 0;
                        break;
                    case PTIR_INTR_HIDDEN:
                        a0 = const_cast<void*>(inputs.hidden);
                        params.intrinsic_dtype =
                            inputs.auxiliary_intrinsic_dtype;
                        params.intrinsic_row_stride =
                            inputs.auxiliary_row_stride;
                        params.intrinsic_row_offset =
                            inputs.hidden_row_offset;
                        break;
                    case PTIR_INTR_QUERY:
                        a0 = const_cast<void*>(inputs.query);
                        params.intrinsic_dtype =
                            inputs.auxiliary_intrinsic_dtype;
                        params.intrinsic_row_stride =
                            inputs.auxiliary_row_stride;
                        params.intrinsic_row_offset =
                            inputs.query_row_offset;
                        break;
                    case PTIR_INTR_VALUE_HEAD:
                        a0 = const_cast<void*>(inputs.value_head);
                        params.intrinsic_dtype =
                            inputs.auxiliary_intrinsic_dtype;
                        params.intrinsic_row_stride =
                            inputs.auxiliary_row_stride;
                        params.intrinsic_row_offset =
                            inputs.value_head_row_offset;
                        break;
                    case PTIR_INTR_LAYER:
                        a0 = const_cast<std::uint32_t*>(inputs.layer);
                        params.intrinsic_dtype = 0;
                        params.intrinsic_row_stride = 1;
                        params.intrinsic_row_offset = 0;
                        break;
                    default:
                        error = "generated test intrinsic is not bound";
                        return false;
                }
                if (a0 == nullptr ||
                    (op.intr != PTIR_INTR_MTP_DRAFTS &&
                     params.imm == 0)) {
                    error = "generated test intrinsic is unbound";
                    return false;
                }
            }

            CUDA_OK(cudaMemcpy(
                params_device.pointer,
                &params,
                sizeof(params),
                cudaMemcpyHostToDevice));
            const auto found = kernels_.find(op.tag);
            if (found == kernels_.end()) {
                error = "generated singleton kernel is not compiled";
                return false;
            }
            CUdeviceptr status_pointer =
                reinterpret_cast<CUdeviceptr>(status_device.pointer);
            CUdeviceptr descriptor_pointer =
                reinterpret_cast<CUdeviceptr>(descriptor_device.pointer);
            CUdeviceptr a0_pointer = reinterpret_cast<CUdeviceptr>(a0);
            CUdeviceptr a1_pointer = reinterpret_cast<CUdeviceptr>(a1);
            CUdeviceptr a2_pointer = reinterpret_cast<CUdeviceptr>(a2);
            CUdeviceptr o0_pointer = reinterpret_cast<CUdeviceptr>(o0);
            CUdeviceptr o1_pointer = reinterpret_cast<CUdeviceptr>(o1);
            CUdeviceptr temporary_pointer =
                reinterpret_cast<CUdeviceptr>(temporary.pointer);
            CUdeviceptr params_pointer =
                reinterpret_cast<CUdeviceptr>(params_device.pointer);
            void* arguments[] = {
                &status_pointer,
                &descriptor_pointer,
                &a0_pointer,
                &a1_pointer,
                &a2_pointer,
                &o0_pointer,
                &o1_pointer,
                &temporary_pointer,
                &params_pointer,
            };
            CU_OK(cuLaunchKernel(
                found->second->function,
                1, 1, 1,
                1, 1, 1,
                0,
                nullptr,
                arguments,
                nullptr));
            CU_OK(cuCtxSynchronize());
            CUDA_OK(cudaMemcpy(
                &status,
                status_device.pointer,
                sizeof(status),
                cudaMemcpyDeviceToHost));
            if (status.state != 1) {
                error =
                    "generated singleton fault tag=" +
                    std::to_string(op.tag) +
                    " code=" + std::to_string(status.fault);
                return false;
            }
            if (op.tag == PTIR_OP_CHAN_PUT) {
                const std::size_t dense =
                    stage.channel_bindings.at(op.chan);
                pending[dense] = true;
            }
        }
        return true;
    }

    const Trace& trace_;
    const std::vector<plan::StagePlan>& plans_;
    const std::map<std::uint8_t, std::unique_ptr<CompiledKernel>>& kernels_;
    ChannelArena arena_;
};

std::vector<std::uint16_t> to_bf16(const std::vector<float>& values) {
    std::vector<std::uint16_t> result(values.size());
    for (std::size_t index = 0; index < values.size(); ++index) {
        std::uint32_t bits = 0;
        std::memcpy(&bits, &values[index], sizeof(bits));
        result[index] = static_cast<std::uint16_t>(bits >> 16);
    }
    return result;
}

DeviceBuffer upload_bf16(const std::vector<float>& values) {
    const auto encoded = to_bf16(values);
    DeviceBuffer device(encoded.size() * sizeof(std::uint16_t));
    CUDA_OK(cudaMemcpy(
        device.pointer,
        encoded.data(),
        device.bytes,
        cudaMemcpyHostToDevice));
    return device;
}

GeneratedValueDesc generated_desc(
    std::uint32_t dtype,
    std::initializer_list<std::uint32_t> dims) {
    GeneratedValueDesc descriptor{};
    descriptor.dtype = dtype;
    descriptor.rank = static_cast<std::uint32_t>(dims.size());
    descriptor.len = 1;
    std::size_t index = 0;
    for (const std::uint32_t dimension : dims) {
        descriptor.dims[index++] = dimension;
        descriptor.len *= dimension;
    }
    descriptor.rows = 1;
    if (descriptor.rank >= 2) {
        descriptor.rows = descriptor.len /
            descriptor.dims[descriptor.rank - 1];
    }
    descriptor.last = descriptor.len / descriptor.rows;
    return descriptor;
}

void run_ambient_rng_contract(
    const std::map<std::uint8_t, std::unique_ptr<CompiledKernel>>& kernels) {
    DeviceBuffer output(8 * sizeof(float));
    const GeneratedValueDesc descriptor =
        generated_desc(PTIR_DT_F32, {2, 4});
    GeneratedOpParams params{};
    params.tag = PTIR_OP_RNG;
    params.o0 = 0;
    params.imm = 3;
    params.kind = 0;
    params.rng_seed = 0;
    const GeneratedStatus status = launch_raw(
        *kernels.at(PTIR_OP_RNG),
        {descriptor},
        params,
        nullptr,
        nullptr,
        nullptr,
        output.pointer,
        nullptr);
    std::array<float, 8> actual{};
    CUDA_OK(cudaMemcpy(
        actual.data(),
        output.pointer,
        output.bytes,
        cudaMemcpyDeviceToHost));
    std::array<float, 8> expected{};
    const std::uint64_t effective =
        ptir_rng_seed_eff_stream(0, 3);
    for (std::uint32_t index = 0; index < expected.size(); ++index) {
        expected[index] =
            ptir_rng_hash_uniform(effective, index);
    }
    expect(
        status.state == 1 && actual == expected,
        "generated ambient RNG preserves the matrix flat-index axis");
}

void run_rank_threshold_contract(
    const std::map<std::uint8_t, std::unique_ptr<CompiledKernel>>& kernels) {
    const float scores[2] = {2.0f, 1.0f};
    const std::uint32_t threshold = 0x80000000u;
    DeviceBuffer scores_device(sizeof(scores));
    DeviceBuffer threshold_device(sizeof(threshold));
    DeviceBuffer output(2);
    CUDA_OK(cudaMemcpy(
        scores_device.pointer,
        scores,
        sizeof(scores),
        cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(
        threshold_device.pointer,
        &threshold,
        sizeof(threshold),
        cudaMemcpyHostToDevice));
    GeneratedOpParams params{};
    params.tag = PTIR_OP_PIVOT_THRESHOLD;
    params.a0 = 0;
    params.a1 = 1;
    params.o0 = 2;
    params.pred_tag = 0;
    const GeneratedStatus status = launch_raw(
        *kernels.at(PTIR_OP_PIVOT_THRESHOLD),
        {
            generated_desc(PTIR_DT_F32, {2}),
            generated_desc(PTIR_DT_U32, {}),
            generated_desc(PTIR_DT_BOOL, {2}),
        },
        params,
        scores_device.pointer,
        threshold_device.pointer,
        nullptr,
        output.pointer,
        nullptr);
    std::array<std::uint8_t, 2> actual{};
    CUDA_OK(cudaMemcpy(
        actual.data(),
        output.pointer,
        actual.size(),
        cudaMemcpyDeviceToHost));
    expect(
        status.state == 1 &&
            actual == std::array<std::uint8_t, 2>{1, 1},
        "generated RankLe treats U32 thresholds as unsigned");
}

void run_stock_library_numeric_contract() {
    DeviceBuffer commit_device(sizeof(std::uint32_t));
    const std::uint32_t one = 1;
    CUDA_OK(cudaMemcpy(
        commit_device.pointer,
        &one,
        sizeof(one),
        cudaMemcpyHostToDevice));
    PtirLaneTableHeader header{
        PTIR_LANE_TABLE_ABI_VERSION, 1, 0, 0};
    PtirLaneRecord lane{};
    lane.commit_slot =
        reinterpret_cast<std::uint64_t>(commit_device.pointer);
    DeviceBuffer header_device(sizeof(header));
    DeviceBuffer lane_device(sizeof(lane));
    CUDA_OK(cudaMemcpy(
        header_device.pointer,
        &header,
        sizeof(header),
        cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(
        lane_device.pointer,
        &lane,
        sizeof(lane),
        cudaMemcpyHostToDevice));

    const float scan_input[3] = {1.0e20f, -1.0e20f, 3.0f};
    DeviceBuffer scan_input_device(sizeof(scan_input));
    DeviceBuffer scan_output_device(sizeof(scan_input));
    CUDA_OK(cudaMemcpy(
        scan_input_device.pointer,
        scan_input,
        sizeof(scan_input),
        cudaMemcpyHostToDevice));
    std::uint64_t scan_values[2] = {
        reinterpret_cast<std::uint64_t>(scan_input_device.pointer),
        reinterpret_cast<std::uint64_t>(scan_output_device.pointer),
    };
    DeviceBuffer scan_values_device(sizeof(scan_values));
    CUDA_OK(cudaMemcpy(
        scan_values_device.pointer,
        scan_values,
        sizeof(scan_values),
        cudaMemcpyHostToDevice));
    GroupedRowShape scan_shape;
    scan_shape.rows.max_numel = 1;
    scan_shape.columns.max_numel = 3;
    scan_shape.max_rows = 1;
    scan_shape.max_columns = 3;
    k_generated_scan_f32<<<1, 1>>>(
        static_cast<const PtirLaneTableHeader*>(header_device.pointer),
        static_cast<const PtirLaneRecord*>(lane_device.pointer),
        static_cast<const std::uint64_t*>(scan_values_device.pointer),
        2,
        0,
        1,
        scan_shape,
        false);
    CUDA_OK(cudaGetLastError());
    std::array<float, 3> scan_output{};
    CUDA_OK(cudaMemcpy(
        scan_output.data(),
        scan_output_device.pointer,
        sizeof(scan_output),
        cudaMemcpyDeviceToHost));
    expect(
        scan_output ==
            std::array<float, 3>{1.0e20f, 0.0f, 3.0f},
        "stock scan library preserves increasing-index prefix order");

    const float zero = 0.0f;
    const float nan = std::numeric_limits<float>::quiet_NaN();
    DeviceBuffer left_device(sizeof(float));
    DeviceBuffer right_device(sizeof(float));
    DeviceBuffer output_device(sizeof(float));
    CUDA_OK(cudaMemcpy(
        left_device.pointer,
        &zero,
        sizeof(zero),
        cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(
        right_device.pointer,
        &nan,
        sizeof(nan),
        cudaMemcpyHostToDevice));
    const std::uint64_t matmul_values[3] = {
        reinterpret_cast<std::uint64_t>(left_device.pointer),
        reinterpret_cast<std::uint64_t>(right_device.pointer),
        reinterpret_cast<std::uint64_t>(output_device.pointer),
    };
    DeviceBuffer matmul_values_device(sizeof(matmul_values));
    CUDA_OK(cudaMemcpy(
        matmul_values_device.pointer,
        matmul_values,
        sizeof(matmul_values),
        cudaMemcpyHostToDevice));
    const GroupedRowShape scalar_matrix_shape{};
    k_generated_matmul_f32<<<1, 1>>>(
        static_cast<const PtirLaneTableHeader*>(header_device.pointer),
        static_cast<const PtirLaneRecord*>(lane_device.pointer),
        static_cast<const std::uint64_t*>(matmul_values_device.pointer),
        3,
        0,
        1,
        2,
        scalar_matrix_shape,
        scalar_matrix_shape);
    CUDA_OK(cudaGetLastError());
    float matmul_output = nan;
    CUDA_OK(cudaMemcpy(
        &matmul_output,
        output_device.pointer,
        sizeof(matmul_output),
        cudaMemcpyDeviceToHost));
    expect(
        matmul_output == 0.0f &&
            !std::isnan(matmul_output),
        "stock matmul library skips zero left operands before NaN");

    constexpr float canary = -1234.0f;
    const std::array<float, 2> left0{2.0f, 3.0f};
    const std::array<float, 4> left1{1.0f, 0.0f, 0.0f, 2.0f};
    const std::array<float, 2> right{5.0f, 7.0f};
    const std::array<float, 2> initial_output{canary, canary};
    DeviceBuffer left0_device(sizeof(left0));
    DeviceBuffer left1_device(sizeof(left1));
    DeviceBuffer right0_device(sizeof(right));
    DeviceBuffer right1_device(sizeof(right));
    DeviceBuffer output0_device(sizeof(initial_output));
    DeviceBuffer output1_device(sizeof(initial_output));
    CUDA_OK(cudaMemcpy(
        left0_device.pointer,
        left0.data(),
        sizeof(left0),
        cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(
        left1_device.pointer,
        left1.data(),
        sizeof(left1),
        cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(
        right0_device.pointer,
        right.data(),
        sizeof(right),
        cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(
        right1_device.pointer,
        right.data(),
        sizeof(right),
        cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(
        output0_device.pointer,
        initial_output.data(),
        sizeof(initial_output),
        cudaMemcpyHostToDevice));
    CUDA_OK(cudaMemcpy(
        output1_device.pointer,
        initial_output.data(),
        sizeof(initial_output),
        cudaMemcpyHostToDevice));
    const std::array<std::uint64_t, 6> ragged_values{
        reinterpret_cast<std::uint64_t>(left0_device.pointer),
        reinterpret_cast<std::uint64_t>(right0_device.pointer),
        reinterpret_cast<std::uint64_t>(output0_device.pointer),
        reinterpret_cast<std::uint64_t>(left1_device.pointer),
        reinterpret_cast<std::uint64_t>(right1_device.pointer),
        reinterpret_cast<std::uint64_t>(output1_device.pointer),
    };
    DeviceBuffer ragged_values_device(sizeof(ragged_values));
    CUDA_OK(cudaMemcpy(
        ragged_values_device.pointer,
        ragged_values.data(),
        sizeof(ragged_values),
        cudaMemcpyHostToDevice));
    const std::array<std::uint32_t, 2> commits{1, 1};
    DeviceBuffer ragged_commits_device(sizeof(commits));
    CUDA_OK(cudaMemcpy(
        ragged_commits_device.pointer,
        commits.data(),
        sizeof(commits),
        cudaMemcpyHostToDevice));
    std::array<PtirLaneRecord, 2> ragged_lanes{};
    ragged_lanes[0].commit_slot =
        reinterpret_cast<std::uint64_t>(ragged_commits_device.pointer);
    ragged_lanes[0].row_count = 1;
    ragged_lanes[1].commit_slot =
        reinterpret_cast<std::uint64_t>(
            static_cast<std::uint32_t*>(ragged_commits_device.pointer) + 1);
    ragged_lanes[1].row_count = 2;
    DeviceBuffer ragged_lanes_device(sizeof(ragged_lanes));
    CUDA_OK(cudaMemcpy(
        ragged_lanes_device.pointer,
        ragged_lanes.data(),
        sizeof(ragged_lanes),
        cudaMemcpyHostToDevice));
    PtirLaneTableHeader ragged_header{
        PTIR_LANE_TABLE_ABI_VERSION, 2, 0, 0};
    DeviceBuffer ragged_header_device(sizeof(ragged_header));
    CUDA_OK(cudaMemcpy(
        ragged_header_device.pointer,
        &ragged_header,
        sizeof(ragged_header),
        cudaMemcpyHostToDevice));
    GroupedRowShape left_shape;
    left_shape.rows.max_numel = 2;
    left_shape.rows.extent = PTIR_EXTENT_ROW_COUNT;
    left_shape.columns.max_numel = 2;
    left_shape.max_rows = 2;
    left_shape.max_columns = 2;
    GroupedRowShape right_shape;
    right_shape.rows.max_numel = 2;
    right_shape.columns.max_numel = 1;
    right_shape.max_rows = 2;
    right_shape.max_columns = 1;
    k_generated_matmul_f32<<<1, 32>>>(
        static_cast<const PtirLaneTableHeader*>(
            ragged_header_device.pointer),
        static_cast<const PtirLaneRecord*>(ragged_lanes_device.pointer),
        static_cast<const std::uint64_t*>(
            ragged_values_device.pointer),
        3,
        0,
        1,
        2,
        left_shape,
        right_shape);
    CUDA_OK(cudaGetLastError());
    std::array<float, 2> output0{};
    std::array<float, 2> output1{};
    CUDA_OK(cudaMemcpy(
        output0.data(),
        output0_device.pointer,
        sizeof(output0),
        cudaMemcpyDeviceToHost));
    CUDA_OK(cudaMemcpy(
        output1.data(),
        output1_device.pointer,
        sizeof(output1),
        cudaMemcpyDeviceToHost));
    expect(
        output0 == std::array<float, 2>{31.0f, canary} &&
            output1 == std::array<float, 2>{5.0f, 14.0f},
        "ragged grouped matmul uses each lane's dimensions");
}

void run_intrinsic_contract(
    const std::map<std::uint8_t, std::unique_ptr<CompiledKernel>>& kernels) {
    auto run_float = [&](
                         std::uint16_t intrinsic,
                         std::uint32_t storage,
                         const void* input,
                         std::uint32_t stride,
                         std::uint32_t offset,
                         std::vector<float> expected,
                         const std::string& label) {
        const GeneratedValueDesc descriptor =
            generated_desc(
                PTIR_DT_F32,
                expected.size() == 4
                    ? std::initializer_list<std::uint32_t>{1, 4}
                    : std::initializer_list<std::uint32_t>{
                          static_cast<std::uint32_t>(expected.size())});
        DeviceBuffer output(expected.size() * sizeof(float));
        GeneratedOpParams params{};
        params.tag = PTIR_OP_INTRINSIC_VAL;
        params.o0 = 0;
        params.imm = descriptor.last;
        params.intr = intrinsic;
        params.intrinsic_dtype = storage;
        params.intrinsic_row_stride = stride;
        params.intrinsic_row_offset = offset;
        const GeneratedStatus status = launch_raw(
            *kernels.at(PTIR_OP_INTRINSIC_VAL),
            {descriptor},
            params,
            input,
            nullptr,
            nullptr,
            output.pointer,
            nullptr);
        std::vector<float> actual(expected.size());
        CUDA_OK(cudaMemcpy(
            actual.data(),
            output.pointer,
            output.bytes,
            cudaMemcpyDeviceToHost));
        expect(
            status.state == 1 && actual == expected,
            "generated " + label + " intrinsic binding");
    };

    const std::vector<float> hidden{1.0f, -2.0f, 3.0f};
    DeviceBuffer hidden_device(hidden.size() * sizeof(float));
    CUDA_OK(cudaMemcpy(
        hidden_device.pointer,
        hidden.data(),
        hidden_device.bytes,
        cudaMemcpyHostToDevice));
    run_float(
        PTIR_INTR_HIDDEN,
        0,
        hidden_device.pointer,
        3,
        0,
        hidden,
        "Hidden");
    run_float(
        PTIR_INTR_VALUE_HEAD,
        0,
        hidden_device.pointer,
        3,
        0,
        hidden,
        "ValueHead");

    const std::vector<float> padded{
        100, 101, 102, 103, 104, 105,
        7, 8, 9, 10, 900, 901,
    };
    DeviceBuffer padded_device(padded.size() * sizeof(float));
    CUDA_OK(cudaMemcpy(
        padded_device.pointer,
        padded.data(),
        padded_device.bytes,
        cudaMemcpyHostToDevice));
    run_float(
        PTIR_INTR_LOGITS,
        0,
        padded_device.pointer,
        6,
        1,
        {7, 8, 9, 10},
        "FP32 padded Logits");

    const auto query_bf16 = to_bf16(padded);
    DeviceBuffer query_device(
        query_bf16.size() * sizeof(std::uint16_t));
    CUDA_OK(cudaMemcpy(
        query_device.pointer,
        query_bf16.data(),
        query_device.bytes,
        cudaMemcpyHostToDevice));
    run_float(
        PTIR_INTR_QUERY,
        1,
        query_device.pointer,
        6,
        1,
        {7, 8, 9, 10},
        "BF16 padded Query");

    const std::array<std::int32_t, 2> input_drafts{3, 5};
    DeviceBuffer draft_device(sizeof(input_drafts));
    CUDA_OK(cudaMemcpy(
        draft_device.pointer,
        input_drafts.data(),
        sizeof(input_drafts),
        cudaMemcpyHostToDevice));
    DeviceBuffer draft_output(2 * sizeof(std::int32_t));
    GeneratedOpParams draft_params{};
    draft_params.tag = PTIR_OP_INTRINSIC_VAL;
    draft_params.o0 = 0;
    draft_params.intr = PTIR_INTR_MTP_DRAFTS;
    const GeneratedValueDesc draft_descriptor =
        generated_desc(PTIR_DT_I32, {2});
    const GeneratedStatus draft_status = launch_raw(
        *kernels.at(PTIR_OP_INTRINSIC_VAL),
        {draft_descriptor},
        draft_params,
        draft_device.pointer,
        nullptr,
        nullptr,
        draft_output.pointer,
        nullptr);
    std::array<std::int32_t, 2> draft_tokens{};
    CUDA_OK(cudaMemcpy(
        draft_tokens.data(),
        draft_output.pointer,
        draft_output.bytes,
        cudaMemcpyDeviceToHost));
    expect(
        draft_status.state == 1 &&
            draft_tokens == input_drafts,
        "generated MtpDrafts intrinsic copies I32 draft tokens");
    const GeneratedStatus invalid_draft_status = launch_raw(
        *kernels.at(PTIR_OP_INTRINSIC_VAL),
        {generated_desc(PTIR_DT_F32, {2})},
        draft_params,
        draft_device.pointer,
        nullptr,
        nullptr,
        draft_output.pointer,
        nullptr);
    expect(
        invalid_draft_status.state == 3,
        "generated MtpDrafts rejects a non-I32 descriptor");

    const std::uint32_t layer = 7;
    DeviceBuffer layer_device(sizeof(layer));
    CUDA_OK(cudaMemcpy(
        layer_device.pointer,
        &layer,
        sizeof(layer),
        cudaMemcpyHostToDevice));
    DeviceBuffer layer_output(sizeof(layer));
    GeneratedOpParams params{};
    params.tag = PTIR_OP_INTRINSIC_VAL;
    params.o0 = 0;
    params.imm = 1;
    params.intr = PTIR_INTR_LAYER;
    const GeneratedStatus status = launch_raw(
        *kernels.at(PTIR_OP_INTRINSIC_VAL),
        {generated_desc(PTIR_DT_U32, {})},
        params,
        layer_device.pointer,
        nullptr,
        nullptr,
        layer_output.pointer,
        nullptr);
    std::uint32_t actual_layer = 0;
    CUDA_OK(cudaMemcpy(
        &actual_layer,
        layer_output.pointer,
        sizeof(actual_layer),
        cudaMemcpyDeviceToHost));
    expect(
        status.state == 1 && actual_layer == layer,
        "generated Layer intrinsic preserves U32 scalar semantics");
}

plan::StagePlan synthetic_stage_plan(
    std::uint8_t stage,
    std::string signature,
    std::vector<std::uint32_t> channel_bindings,
    std::vector<container::COp> ops,
    std::vector<plan::ValueType> value_types) {
    plan::StagePlan result;
    result.stage = stage;
    result.signature.assign(signature.begin(), signature.end());
    result.signature_hash = container::fnv1a64(
        result.signature.data(), result.signature.size());
    result.channel_bindings = std::move(channel_bindings);
    for (std::size_t index = 0; index < ops.size(); ++index) {
        result.ops.push_back({std::move(ops[index]), {}});
        plan::Region singleton;
        singleton.schedule = PTIR_SCHEDULE_EFFECTS;
        singleton.nodes = {static_cast<std::uint32_t>(index)};
        result.singleton.regions.push_back(std::move(singleton));
    }
    result.value_types = std::move(value_types);
    result.singleton.kind = 0;
    result.fused.kind = 1;
    if (!result.ops.empty()) {
        plan::Region fused;
        fused.schedule = PTIR_SCHEDULE_EFFECTS;
        for (std::size_t index = 0; index < result.ops.size(); ++index) {
            fused.nodes.push_back(static_cast<std::uint32_t>(index));
        }
        result.fused.regions.push_back(std::move(fused));
    }
    return result;
}

plan::ValueType fixed_type(
    std::uint8_t dtype,
    std::initializer_list<std::uint32_t> dimensions) {
    plan::ValueType type;
    type.dtype = dtype;
    for (const std::uint32_t dimension : dimensions) {
        type.dims.push_back({
            .symbolic = false,
            .value = dimension,
        });
    }
    return type;
}

void validate_direct_argmax_reshape_guard() {
    auto make_stage = [](std::initializer_list<std::uint32_t> source_dims,
                         std::initializer_list<std::uint32_t> reshape_dims) {
        container::COp logits;
        logits.tag = PTIR_OP_INTRINSIC_VAL;
        logits.intr = PTIR_INTR_LOGITS;
        logits.dtype = PTIR_DT_F32;
        container::COp reshape;
        reshape.tag = PTIR_OP_RESHAPE;
        reshape.args = {0};
        container::COp argmax;
        argmax.tag = PTIR_OP_REDUCE_ARGMAX;
        argmax.args = {1};
        return synthetic_stage_plan(
            PTIR_STAGE_EPILOGUE,
            "direct-argmax-reshape",
            {},
            {logits, reshape, argmax},
            {
                fixed_type(PTIR_DT_F32, source_dims),
                fixed_type(PTIR_DT_F32, reshape_dims),
                fixed_type(PTIR_DT_U32, {}),
            });
    };
    auto flattened = make_stage({2, 4}, {8});
    const GeneratedKernelSource flattened_generated = emit_fused_region_cuda(
        "ptir_flattened_argmax",
        flattened,
        flattened.fused.regions.front());
    expect(
        flattened_generated.ok &&
            flattened_generated.source.find(
            "const m1_u32 direct_intrinsic_index") == std::string::npos,
        "flattening reshape cannot direct-index a BF16 row table");
    expect(
        !generated_compact_argmax_value(
             flattened, {0, 1, 2})
             .has_value(),
        "flattening reshape cannot select compact argmax scratch");

    auto row_preserving = make_stage({1, 8}, {8});
    const GeneratedKernelSource row_preserving_generated =
        emit_fused_region_cuda(
        "ptir_row_preserving_argmax",
        row_preserving,
        row_preserving.fused.regions.front());
    expect(
        row_preserving_generated.ok &&
            row_preserving_generated.source.find(
                "const m1_u32 direct_intrinsic_index") !=
                std::string::npos,
        "row-preserving reshape retains direct BF16 argmax fusion");
    expect(
        generated_compact_argmax_value(
            row_preserving, {0, 1, 2}) == 2,
        "row-preserving reshape retains compact argmax scratch");
}

void validate_inactive_lane_state_codegen() {
    container::COp take;
    take.tag = PTIR_OP_CHAN_TAKE;
    take.chan = 0;
    container::COp one;
    one.tag = PTIR_OP_CONST;
    one.lit_dtype = PTIR_DT_U32;
    one.lit_bits = 1;
    container::COp add;
    add.tag = PTIR_OP_ADD;
    add.args = {0, 1};
    container::COp put;
    put.tag = PTIR_OP_CHAN_PUT;
    put.chan = 0;
    put.args = {2};
    put.results = 0;
    auto stage = synthetic_stage_plan(
        PTIR_STAGE_EPILOGUE,
        "inactive-lane-state",
        {0},
        {take, one, add, put},
        {
            fixed_type(PTIR_DT_U32, {1}),
            fixed_type(PTIR_DT_U32, {}),
            fixed_type(PTIR_DT_U32, {1}),
        });
    const GeneratedKernelSource generated = emit_fused_region_cuda(
        "ptir_inactive_lane_state",
        stage,
        stage.fused.regions.front());
    expect(
        generated.ok &&
            generated.source.find("channel.committed_cell") !=
                std::string::npos &&
            generated.source.find("committed[byte]") != std::string::npos,
        "inactive generated lanes preserve non-sample channel state");
}

void run_cross_stage_pending_contract(
    const std::map<std::uint8_t, std::unique_ptr<CompiledKernel>>& kernels) {
    Trace trace;
    Channel state;
    state.id = 0;
    state.type = {Shape::vec(1), DType::U32};
    state.capacity = 1;
    Channel output;
    output.id = 1;
    output.type = {Shape::vec(1), DType::U32};
    output.capacity = 1;
    output.host_visible = true;
    output.host_reader = true;
    trace.channels = {state, output};
    Stage producer;
    producer.kind = StageKind::Prologue;
    producer.puts = {{0, 0}};
    Stage consumer;
    consumer.kind = StageKind::Epilogue;
    consumer.takes = {0};
    consumer.puts = {{1, 0}};
    trace.stages = {producer, consumer};

    container::COp seven;
    seven.tag = PTIR_OP_CONST;
    seven.lit_dtype = PTIR_DT_U32;
    seven.lit_bits = 7;
    container::COp put_state;
    put_state.tag = PTIR_OP_CHAN_PUT;
    put_state.chan = 0;
    put_state.args = {0};
    put_state.results = 0;
    container::COp take_state;
    take_state.tag = PTIR_OP_CHAN_TAKE;
    take_state.chan = 0;
    container::COp put_output;
    put_output.tag = PTIR_OP_CHAN_PUT;
    put_output.chan = 1;
    put_output.args = {0};
    put_output.results = 0;
    plan::ValueType scalar;
    scalar.dtype = PTIR_DT_U32;
    scalar.dims = {{.symbolic = false, .value = 1}};
    const std::vector<plan::StagePlan> plans{
        synthetic_stage_plan(
            PTIR_STAGE_PROLOGUE,
            "cross-stage-producer",
            {0},
            {seven, put_state},
            {scalar}),
        synthetic_stage_plan(
            PTIR_STAGE_EPILOGUE,
            "cross-stage-consumer",
            {0, 1},
            {take_state, put_output},
            {scalar}),
    };
    GeneratedSingletonExecutor executor(trace, plans, kernels);
    RuntimeExtents extents;
    GeneratedInputs inputs;
    std::string error;
    const bool committed = executor.run(inputs, extents, error);
    std::uint32_t actual = 0;
    if (executor.output_ready(1)) {
        executor.take(1, &actual, sizeof(actual));
    }
    expect(
        committed && actual == 7,
        "generated later stage observes pending earlier-stage put: " +
            error);
}

void run_grouped_fused_greedy(
    const Fixture& fixture,
    std::uint32_t vocab = 8) {
    constexpr std::uint32_t lane_count = 2;
    auto stage = fixture.plans.front();
    for (auto& type : stage.value_types) {
        for (auto& dimension : type.dims) {
            if (!dimension.symbolic && dimension.value == 8) {
                dimension.value = vocab;
            }
        }
    }
    const auto generated_region = std::find_if(
        stage.fused.regions.begin(),
        stage.fused.regions.end(),
        [](const auto& region) {
            return !region.library;
        });
    expect(
        generated_region != stage.fused.regions.end(),
        "greedy plan has a generated fused region");
    if (generated_region == stage.fused.regions.end()) return;
    const GeneratedKernelSource generated_source =
        emit_fused_region_cuda(
            "ptir_grouped_fused_greedy",
            stage,
            *generated_region);
    std::vector<std::uint32_t> greedy_bases(stage.ops.size());
    std::uint32_t greedy_value = 0;
    std::uint32_t greedy_argmax_node = UINT32_MAX;
    for (std::size_t node = 0; node < stage.ops.size(); ++node) {
        greedy_bases[node] = greedy_value;
        greedy_value += stage.ops[node].op.results;
        if (stage.ops[node].op.tag == PTIR_OP_REDUCE_ARGMAX) {
            greedy_argmax_node = static_cast<std::uint32_t>(node);
        }
    }
    const DirectArgmaxAnalysis greedy_direct =
        analyze_direct_argmax(stage, *generated_region, greedy_bases);
    expect(
        generated_source.ok &&
            greedy_argmax_node != UINT32_MAX &&
            greedy_direct.intrinsic[greedy_argmax_node] ==
                PTIR_INTR_LOGITS &&
            greedy_direct.requires_single_row[greedy_argmax_node] != 0 &&
            generated_source.source.find(
                "const m1_u32 direct_intrinsic_index") !=
                std::string::npos,
        "compiler-produced symbolic logits retain direct argmax fusion");
    auto kernel = compile_source(
        generated_source,
        "grouped fused greedy");

    RuntimeExtents extents;
    std::vector<GeneratedValueDesc> one_lane_descriptors;
    std::size_t maximum_length = 1;
    for (const auto& type : stage.value_types) {
        GeneratedValueDesc descriptor;
        if (!describe_value(type, extents, descriptor)) {
            expect(false, "greedy fused descriptor resolution");
            return;
        }
        maximum_length =
            std::max<std::size_t>(maximum_length, descriptor.len);
        one_lane_descriptors.push_back(descriptor);
    }
    std::vector<GeneratedValueDesc> descriptors;
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        descriptors.insert(
            descriptors.end(),
            one_lane_descriptors.begin(),
            one_lane_descriptors.end());
    }
    const auto align = [](std::size_t value) {
        return (value + 255) / 256 * 256;
    };
    std::vector<std::uint32_t> offsets(
        one_lane_descriptors.size());
    std::size_t scratch_bytes = 256;
    for (std::size_t value = 0;
         value < one_lane_descriptors.size();
         ++value) {
        scratch_bytes = align(scratch_bytes);
        offsets[value] = static_cast<std::uint32_t>(scratch_bytes);
        scratch_bytes += align(value_bytes(one_lane_descriptors[value]));
    }
    const std::uint32_t temporary_offset =
        static_cast<std::uint32_t>(align(scratch_bytes));
    scratch_bytes =
        temporary_offset + align(maximum_length * 32);
    const std::uint32_t scratch_stride =
        static_cast<std::uint32_t>(scratch_bytes);

    std::vector<GeneratedOpMeta> operations;
    std::string validation_error;
    expect(
        validate_singleton_plan(stage, operations, validation_error),
        "greedy fused parameter plan validates: " + validation_error);
    std::vector<GeneratedOpParams> params(stage.ops.size());
    for (const GeneratedOpMeta& metadata : operations) {
        const auto& op = metadata.op;
        auto& param = params[metadata.node];
        param.tag = op.tag;
        param.a0 = !op.args.empty() ? op.args[0] : 0;
        param.a1 = op.args.size() > 1
            ? op.args[1]
            : (op.tag == PTIR_OP_PIVOT_THRESHOLD
                   ? op.pred_payload
                   : 0);
        param.a2 = op.args.size() > 2 ? op.args[2] : 0;
        param.o0 = op.results > 0
            ? metadata.result_base
            : param.a0;
        param.o1 = op.results > 1 ? param.o0 + 1 : param.o0;
        param.imm = op.imm;
        param.imm2 = op.imm2;
        param.imm3 = op.imm3;
        param.kind = op.kind;
        param.pred_tag = op.pred_tag;
        param.lit_dtype = op.lit_dtype;
        param.lit_bits = op.lit_bits;
        param.intr = op.intr;
        param.bool_storage = 0;
        if (op.tag == PTIR_OP_CHAN_PUT) {
            const std::uint32_t dense =
                stage.channel_bindings.at(op.chan);
            param.sink_bytes = static_cast<std::uint32_t>(
                fixture.trace.channels[dense].type.shape.numel() *
                dtype_size(fixture.trace.channels[dense].type.dtype));
        }
    }

    std::array<ChannelArena, lane_count> arenas;
    std::array<std::int32_t, lane_count> seeds{1, 1};
    std::vector<PtirLaneRecord> lane_records(lane_count);
    std::vector<PtirLaneChannelSlot> channel_slots(
        lane_count * stage.channel_bindings.size());
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        arenas[lane].init(fixture.trace.channels);
        arenas[lane].seed_cell(0, &seeds[lane], sizeof(seeds[lane]));
        const std::uint32_t one = 1;
        CUDA_OK(cudaMemcpy(
            arenas[lane].d_commit(),
            &one,
            sizeof(one),
            cudaMemcpyHostToDevice));
        lane_records[lane].channel_slot_offset =
            lane * stage.channel_bindings.size();
        lane_records[lane].commit_slot =
            reinterpret_cast<std::uint64_t>(
                arenas[lane].d_commit());
        for (std::size_t local = 0;
             local < stage.channel_bindings.size();
             ++local) {
            const std::uint32_t dense =
                stage.channel_bindings[local];
            auto& slot = channel_slots[
                lane_records[lane].channel_slot_offset + local];
            slot.committed_cell =
                reinterpret_cast<std::uint64_t>(
                    arenas[lane].committed_cell(dense));
            slot.pending_cell =
                reinterpret_cast<std::uint64_t>(
                    arenas[lane].pending_cell(dense));
        }
    }

    std::vector<float> logits(
        static_cast<std::size_t>(lane_count) * vocab, -100.0f);
    logits[2] = 100.0f;
    logits[vocab] = 100.0f;
    DeviceBuffer logits_device = upload_bf16(logits);
    const std::uint8_t row_valid[lane_count] = {1, 0};
    DeviceBuffer row_valid_device(sizeof(row_valid));
    CUDA_OK(cudaMemcpy(
        row_valid_device.pointer,
        row_valid,
        sizeof(row_valid),
        cudaMemcpyHostToDevice));
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        lane_records[lane].sample_output_channel_mask = 0b11;
        lane_records[lane].token_count = 1;
        lane_records[lane].row_valid =
            reinterpret_cast<std::uint64_t>(row_valid_device.pointer);
        lane_records[lane].row_valid_offset = lane;
    }
    std::vector<std::uint64_t> intrinsic_bases(lane_count * 7, 0);
    std::vector<std::uint32_t> intrinsic_modes(lane_count * 7, 0);
    std::vector<std::uint32_t> intrinsic_widths(lane_count * 7, 0);
    std::vector<std::uint32_t> intrinsic_strides(lane_count * 7, 0);
    std::vector<std::uint32_t> intrinsic_offsets(lane_count * 7, 0);
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        const std::size_t index = lane * 7 + PTIR_INTR_LOGITS;
        intrinsic_bases[index] = reinterpret_cast<std::uint64_t>(
            static_cast<std::uint16_t*>(logits_device.pointer) +
            lane * vocab);
        intrinsic_modes[index] = 1;
        intrinsic_widths[index] = vocab;
        intrinsic_strides[index] = vocab;
    }

    PtirLaneTableHeader header{
        PTIR_LANE_TABLE_ABI_VERSION,
        lane_count,
        static_cast<std::uint32_t>(stage.channel_bindings.size()),
        0,
    };
    DeviceBuffer header_device(sizeof(header));
    DeviceBuffer lanes_device(
        lane_records.size() * sizeof(PtirLaneRecord));
    DeviceBuffer channels_device(
        channel_slots.size() * sizeof(PtirLaneChannelSlot));
    DeviceBuffer descriptors_device(
        descriptors.size() * sizeof(GeneratedValueDesc));
    DeviceBuffer params_device(
        params.size() * sizeof(GeneratedOpParams));
    DeviceBuffer offsets_device(
        offsets.size() * sizeof(std::uint32_t));
    DeviceBuffer scratch_device(scratch_stride * lane_count);
    DeviceBuffer pending_device(channel_slots.size());
    DeviceBuffer bases_device(
        intrinsic_bases.size() * sizeof(std::uint64_t));
    DeviceBuffer modes_device(
        intrinsic_modes.size() * sizeof(std::uint32_t));
    DeviceBuffer widths_device(
        intrinsic_widths.size() * sizeof(std::uint32_t));
    DeviceBuffer strides_device(
        intrinsic_strides.size() * sizeof(std::uint32_t));
    DeviceBuffer intrinsic_offsets_device(
        intrinsic_offsets.size() * sizeof(std::uint32_t));
    auto upload = [](void* destination, const void* source, std::size_t bytes) {
        CUDA_OK(cudaMemcpy(
            destination, source, bytes, cudaMemcpyHostToDevice));
    };
    upload(header_device.pointer, &header, sizeof(header));
    upload(
        lanes_device.pointer,
        lane_records.data(),
        lanes_device.bytes);
    upload(
        channels_device.pointer,
        channel_slots.data(),
        channels_device.bytes);
    upload(
        descriptors_device.pointer,
        descriptors.data(),
        descriptors_device.bytes);
    upload(
        params_device.pointer,
        params.data(),
        params_device.bytes);
    upload(
        offsets_device.pointer,
        offsets.data(),
        offsets_device.bytes);
    upload(
        bases_device.pointer,
        intrinsic_bases.data(),
        bases_device.bytes);
    upload(
        modes_device.pointer,
        intrinsic_modes.data(),
        modes_device.bytes);
    upload(
        widths_device.pointer,
        intrinsic_widths.data(),
        widths_device.bytes);
    upload(
        strides_device.pointer,
        intrinsic_strides.data(),
        strides_device.bytes);
    upload(
        intrinsic_offsets_device.pointer,
        intrinsic_offsets.data(),
        intrinsic_offsets_device.bytes);
    CUDA_OK(cudaMemset(
        scratch_device.pointer, 0, scratch_device.bytes));
    CUDA_OK(cudaMemset(
        pending_device.pointer, 0, pending_device.bytes));

    CUdeviceptr header_pointer =
        reinterpret_cast<CUdeviceptr>(header_device.pointer);
    CUdeviceptr lanes_pointer =
        reinterpret_cast<CUdeviceptr>(lanes_device.pointer);
    CUdeviceptr channels_pointer =
        reinterpret_cast<CUdeviceptr>(channels_device.pointer);
    CUdeviceptr descriptors_pointer =
        reinterpret_cast<CUdeviceptr>(descriptors_device.pointer);
    CUdeviceptr params_pointer =
        reinterpret_cast<CUdeviceptr>(params_device.pointer);
    CUdeviceptr offsets_pointer =
        reinterpret_cast<CUdeviceptr>(offsets_device.pointer);
    CUdeviceptr scratch_pointer =
        reinterpret_cast<CUdeviceptr>(scratch_device.pointer);
    CUdeviceptr pending_pointer =
        reinterpret_cast<CUdeviceptr>(pending_device.pointer);
    CUdeviceptr bases_pointer =
        reinterpret_cast<CUdeviceptr>(bases_device.pointer);
    CUdeviceptr modes_pointer =
        reinterpret_cast<CUdeviceptr>(modes_device.pointer);
    CUdeviceptr widths_pointer =
        reinterpret_cast<CUdeviceptr>(widths_device.pointer);
    CUdeviceptr strides_pointer =
        reinterpret_cast<CUdeviceptr>(strides_device.pointer);
    CUdeviceptr intrinsic_offsets_pointer =
        reinterpret_cast<CUdeviceptr>(
            intrinsic_offsets_device.pointer);
    const std::uint32_t value_count =
        static_cast<std::uint32_t>(stage.value_types.size());
    void* arguments[] = {
        &header_pointer,
        &lanes_pointer,
        &channels_pointer,
        &descriptors_pointer,
        &params_pointer,
        &offsets_pointer,
        &scratch_pointer,
        const_cast<std::uint32_t*>(&value_count),
        const_cast<std::uint32_t*>(&scratch_stride),
        const_cast<std::uint32_t*>(&temporary_offset),
        &pending_pointer,
        &bases_pointer,
        &modes_pointer,
        &widths_pointer,
        &strides_pointer,
        &intrinsic_offsets_pointer,
    };
    CU_OK(cuLaunchKernel(
        kernel->function,
        lane_count, 1, 1,
        256, 1, 1,
        0,
        nullptr,
        arguments,
        nullptr));
    CU_OK(cuCtxSynchronize());

    const std::uint32_t taken = 0;
    const std::uint32_t puts[2] = {0, 1};
    DeviceBuffer taken_device(sizeof(taken));
    DeviceBuffer puts_device(sizeof(puts));
    upload(taken_device.pointer, &taken, sizeof(taken));
    upload(puts_device.pointer, puts, sizeof(puts));
    std::array<std::int32_t, lane_count> actual{};
    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
        k_commit_bump<<<1, 1>>>(
            arenas[lane].d_full(),
            arenas[lane].d_head(),
            arenas[lane].d_tail(),
            arenas[lane].d_cap1(),
            static_cast<const std::uint32_t*>(taken_device.pointer),
            1,
            static_cast<const std::uint32_t*>(puts_device.pointer),
            2,
            arenas[lane].d_commit());
        CUDA_OK(cudaGetLastError());
        arenas[lane].sync_host_rings();
        if (arenas[lane].committed_full(1)) {
            arenas[lane].host_take(
                1, &actual[lane], sizeof(actual[lane]));
        }
    }
    expect(
        actual == std::array<std::int32_t, lane_count>{2, -1},
        "inactive grouped greedy lane commits the -1 token sentinel");

    const int benchmark_iterations = vocab > 4096 ? 100 : 1000;
    cudaEvent_t start = nullptr;
    cudaEvent_t stop = nullptr;
    CUDA_OK(cudaEventCreate(&start));
    CUDA_OK(cudaEventCreate(&stop));
    CUDA_OK(cudaEventRecord(start));
    for (int iteration = 0;
         iteration < benchmark_iterations;
         ++iteration) {
        CU_OK(cuLaunchKernel(
            kernel->function,
            lane_count, 1, 1,
            256, 1, 1,
            0,
            nullptr,
            arguments,
            nullptr));
    }
    CUDA_OK(cudaEventRecord(stop));
    CUDA_OK(cudaEventSynchronize(stop));
    float elapsed_ms = 0.0f;
    CUDA_OK(cudaEventElapsedTime(&elapsed_ms, start, stop));
    std::printf(
        "generated fused greedy V=%u B=%u: %.3f us/body (%d iterations)\n",
        vocab,
        lane_count,
        elapsed_ms * 1000.0f / benchmark_iterations,
        benchmark_iterations);
    CUDA_OK(cudaEventDestroy(stop));
    CUDA_OK(cudaEventDestroy(start));
}

void run_counter(
    const Fixture& fixture,
    const std::map<std::uint8_t, std::unique_ptr<CompiledKernel>>& kernels) {
    GeneratedSingletonExecutor executor(
        fixture.trace, fixture.plans, kernels);
    const std::uint32_t seed = 10;
    executor.seed(0, &seed, sizeof(seed));
    RuntimeExtents extents;
    GeneratedInputs inputs;
    std::string error;
    expect(
        executor.run(inputs, extents, error),
        "generated counter step 0 commits: " + error);
    error.clear();
    expect(
        !executor.run(inputs, extents, error),
        "generated counter back-pressure is effect-free");
    std::uint32_t value = 0;
    if (executor.output_ready(1)) {
        executor.take(1, &value, sizeof(value));
    }
    expect(value == 11, "generated counter first value is 11");
    error.clear();
    expect(
        executor.run(inputs, extents, error),
        "generated counter step 2 commits: " + error);
    value = 0;
    if (executor.output_ready(1)) {
        executor.take(1, &value, sizeof(value));
    }
    expect(value == 12, "generated counter persisted value is 12");
}

void run_greedy(
    const Fixture& fixture,
    const std::map<std::uint8_t, std::unique_ptr<CompiledKernel>>& kernels) {
    GeneratedSingletonExecutor executor(
        fixture.trace, fixture.plans, kernels);
    const std::int32_t seed = 1;
    executor.seed(0, &seed, sizeof(seed));
    for (const auto& [logits, expected] :
         std::vector<std::pair<std::vector<float>, std::int32_t>>{
             {{0, 1, 9, 2, 0, 0, 0, 3}, 2},
             {{7, 1, 0, 2, 0, 0, 0, 3}, 0},
         }) {
        const auto bf16 = to_bf16(logits);
        DeviceBuffer device(bf16.size() * sizeof(std::uint16_t));
        CUDA_OK(cudaMemcpy(
            device.pointer,
            bf16.data(),
            device.bytes,
            cudaMemcpyHostToDevice));
        std::string error;
        RuntimeExtents extents;
        const bool committed = executor.run(
            device.pointer, 1, 8, extents, 0, error);
        std::int32_t token = -1;
        if (executor.output_ready(1)) {
            executor.take(1, &token, sizeof(token));
        }
        expect(
            committed && token == expected,
            "generated greedy token expected=" +
                std::to_string(expected) +
                " actual=" + std::to_string(token) +
                " error=" + error);
    }
}

void run_nucleus(
    const Fixture& fixture,
    const std::map<std::uint8_t, std::unique_ptr<CompiledKernel>>& kernels) {
    struct Case {
        std::vector<float> logits;
        float top_p = 1.0f;
        std::uint32_t counter = 0;
        std::int32_t expected = 0;
    };
    const float nan = std::numeric_limits<float>::quiet_NaN();
    const float neg_inf = -std::numeric_limits<float>::infinity();
    const std::vector<Case> cases{
        {{4.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f, -1.0f, nan},
         0.5f, 0, 0},
        {{nan, 1.0f, 1.0f, neg_inf, neg_inf, neg_inf, neg_inf, neg_inf},
         1.0f, 1, 1},
        {{0.0f, 0.0f, neg_inf, neg_inf, neg_inf, neg_inf, neg_inf, neg_inf},
         0.5f, 2, 0},
    };
    for (const Case& test : cases) {
        GeneratedSingletonExecutor executor(
            fixture.trace, fixture.plans, kernels);
        const std::uint32_t state[2] = {1234, test.counter};
        executor.seed(0, state, sizeof(state));
        executor.seed(1, &test.top_p, sizeof(test.top_p));
        DeviceBuffer logits = upload_bf16(test.logits);
        RuntimeExtents extents;
        GeneratedInputs inputs{
            .logits = logits.pointer,
            .intrinsic_dtype = 1,
            .vocab = 8,
            .row_stride = 8,
        };
        std::string error;
        const bool committed = executor.run(inputs, extents, error);
        std::int32_t token = -1;
        if (executor.output_ready(2)) {
            executor.take(2, &token, sizeof(token));
        }
        expect(
            committed && token == test.expected,
            "generated nucleus token expected=" +
                std::to_string(test.expected) +
                " actual=" + std::to_string(token) +
                " error=" + error);
    }
}

void run_masked_gumbel(
    const Fixture& fixture,
    const std::map<std::uint8_t, std::unique_ptr<CompiledKernel>>& kernels) {
    GeneratedSingletonExecutor executor(
        fixture.trace, fixture.plans, kernels);
    const std::int32_t token_seed = 1;
    const std::uint32_t length_seed = 1;
    const std::uint32_t rng_seed[2] = {1234, 0};
    executor.seed(0, &token_seed, sizeof(token_seed));
    executor.seed(3, &length_seed, sizeof(length_seed));
    executor.seed(4, rng_seed, sizeof(rng_seed));

    std::vector<float> logits(32, 0.0f);
    logits[7] = 100.0f;
    const auto bf16 = to_bf16(logits);
    DeviceBuffer device(bf16.size() * sizeof(std::uint16_t));
    CUDA_OK(cudaMemcpy(
        device.pointer,
        bf16.data(),
        device.bytes,
        cudaMemcpyHostToDevice));
    RuntimeExtents extents;
    std::string error;

    std::vector<std::uint8_t> mask(32, 1);
    executor.feed(2, mask.data(), mask.size());
    bool committed = executor.run(
        device.pointer, 1, 32, extents, 0, error);
    std::int32_t token = -1;
    if (executor.output_ready(1)) {
        executor.take(1, &token, sizeof(token));
    }
    expect(
        committed && token == 7,
        "generated masked Gumbel allow-all token expected=7 actual=" +
            std::to_string(token) + " error=" + error);

    error.clear();
    committed = executor.run(
        device.pointer, 1, 32, extents, 0, error);
    expect(
        !committed && !executor.output_ready(1),
        "generated masked Gumbel missing-mask retry is effect-free");

    std::fill(mask.begin(), mask.end(), 0);
    mask[3] = 1;
    executor.feed(2, mask.data(), mask.size());
    error.clear();
    committed = executor.run(
        device.pointer, 1, 32, extents, 0, error);
    token = -1;
    if (executor.output_ready(1)) {
        executor.take(1, &token, sizeof(token));
    }
    expect(
        committed && token == 3,
        "generated masked Gumbel constrained token expected=3 actual=" +
            std::to_string(token) + " error=" + error);
}

void run_matrix_select(
    const Fixture& fixture,
    const std::map<std::uint8_t, std::unique_ptr<CompiledKernel>>& kernels) {
    constexpr std::uint32_t rows = 4;
    constexpr std::uint32_t vocab = 8;
    GeneratedSingletonExecutor executor(
        fixture.trace, fixture.plans, kernels);
    std::vector<float> logits(rows * vocab, 0.0f);
    std::vector<std::uint8_t> allowed(rows * vocab, 0);
    for (std::uint32_t row = 0; row < rows; ++row) {
        logits[row * vocab + row] = 9.0f;
        const std::uint32_t accepted = (row + 2) % vocab;
        logits[row * vocab + accepted] = 1.0f;
        allowed[row * vocab + accepted] = 1;
    }
    executor.feed(0, allowed.data(), allowed.size());
    DeviceBuffer device = upload_bf16(logits);
    RuntimeExtents extents{
        .row_count = rows,
        .sampled_rows = rows,
    };
    GeneratedInputs inputs{
        .logits = device.pointer,
        .intrinsic_dtype = 1,
        .vocab = vocab,
        .row_stride = vocab,
    };
    std::string error;
    const bool committed = executor.run(inputs, extents, error);
    std::array<std::int32_t, rows> tokens{};
    if (executor.output_ready(1)) {
        executor.take(1, tokens.data(), sizeof(tokens));
    }
    expect(
        committed &&
            tokens == std::array<std::int32_t, rows>{2, 3, 4, 5},
        "generated matrix select mask returns [2,3,4,5]: " + error);
}

void run_matrix_mask_apply(
    const Fixture& fixture,
    const std::map<std::uint8_t, std::unique_ptr<CompiledKernel>>& kernels) {
    GeneratedSingletonExecutor executor(
        fixture.trace, fixture.plans, kernels);
    DeviceBuffer device = upload_bf16({
        0,0,9,1,0,2,0,0,
        0,0,0,4,0,3,0,9,
    });
    RuntimeExtents extents{
        .row_count = 2,
        .sampled_rows = 2,
    };
    GeneratedInputs inputs{
        .logits = device.pointer,
        .intrinsic_dtype = 1,
        .vocab = 8,
        .row_stride = 8,
    };
    std::string error;
    const bool committed = executor.run(inputs, extents, error);
    std::array<std::int32_t, 2> tokens{};
    if (executor.output_ready(0)) {
        executor.take(0, tokens.data(), sizeof(tokens));
    }
    expect(
        committed && tokens == std::array<std::int32_t, 2>{5, 3},
        "generated packed mask applies per row: " + error);
}

void run_beam(
    const Fixture& fixture,
    const std::map<std::uint8_t, std::unique_ptr<CompiledKernel>>& kernels) {
    constexpr std::uint32_t beams = 2;
    constexpr std::uint32_t vocab = 8;
    constexpr std::uint32_t pages = 3;
    constexpr std::uint32_t page_size = 4;
    GeneratedSingletonExecutor executor(
        fixture.trace, fixture.plans, kernels);
    auto seed_u32 = [&](ChannelId channel, std::vector<std::uint32_t> values) {
        executor.seed(
            channel, values.data(), values.size() * sizeof(std::uint32_t));
    };
    auto seed_i32 = [&](ChannelId channel, std::vector<std::int32_t> values) {
        executor.seed(
            channel, values.data(), values.size() * sizeof(std::int32_t));
    };
    auto seed_f32 = [&](ChannelId channel, std::vector<float> values) {
        executor.seed(
            channel, values.data(), values.size() * sizeof(float));
    };
    seed_u32(0, {5, 6, 0, 5, 6, 0});
    seed_u32(1, {4, 2, 0, 4, 2, 0});
    seed_u32(2, {6, 6});
    std::vector<std::uint8_t> kv_mask(beams * pages * page_size);
    const std::uint32_t lengths[3] = {4, 2, 0};
    for (std::uint32_t beam = 0; beam < beams; ++beam) {
        for (std::uint32_t page = 0; page < pages; ++page) {
            for (std::uint32_t offset = 0; offset < page_size; ++offset) {
                kv_mask[
                    beam * pages * page_size +
                    page * page_size + offset] =
                    offset < lengths[page] ? 1 : 0;
            }
        }
    }
    executor.seed(3, kv_mask.data(), kv_mask.size());
    seed_u32(4, {6, 6});
    seed_u32(5, {2, 2});
    seed_u32(6, {6, 6});
    seed_u32(7, {2, 2});
    seed_u32(8, {6, 6});
    seed_u32(9, {2, 2});
    seed_i32(10, {1, 2});
    seed_f32(11, {0.0f, 0.0f});

    std::vector<float> logits(beams * vocab, 0.0f);
    logits[3] = 8.0f;
    logits[vocab + 5] = 7.0f;
    DeviceBuffer device = upload_bf16(logits);
    RuntimeExtents extents{
        .row_count = beams,
        .sampled_rows = beams,
    };
    GeneratedInputs inputs{
        .logits = device.pointer,
        .intrinsic_dtype = 1,
        .vocab = vocab,
        .row_stride = vocab,
    };
    std::string error;
    expect(
        !executor.run(inputs, extents, error),
        "generated beam waits for the fresh-slot grant");
    const std::uint32_t fresh[2] = {7, 8};
    executor.feed(12, fresh, sizeof(fresh));
    error.clear();
    expect(
        executor.run(inputs, extents, error),
        "generated beam commits: " + error);
    std::int32_t tokens[2]{};
    std::uint32_t parents[2]{};
    float scores[2]{};
    if (executor.output_ready(13)) {
        executor.take(13, tokens, sizeof(tokens));
    }
    if (executor.output_ready(14)) {
        executor.take(14, parents, sizeof(parents));
    }
    if (executor.output_ready(15)) {
        executor.take(15, scores, sizeof(scores));
    }
    const auto near = [](float left, float right) {
        return std::fabs(left - right) <=
            1.0e-4f + 1.0e-4f * std::fabs(right);
    };
    expect(
        tokens[0] == 3 && tokens[1] == 5,
        "generated beam tokens are [3,5]");
    expect(
        parents[0] == 0 && parents[1] == 1,
        "generated beam parents are [0,1]");
    expect(
        near(scores[0], -0.0023454318f) &&
            near(scores[1], -0.006362776f),
        "generated beam scores match the Rust golden");
}

void run_mtp(
    const Fixture& fixture,
    const std::map<std::uint8_t, std::unique_ptr<CompiledKernel>>& kernels) {
    constexpr std::uint32_t vocab = 8;
    constexpr std::uint32_t verify_rows = 4;
    GeneratedSingletonExecutor executor(
        fixture.trace, fixture.plans, kernels);
    const std::int32_t drafts[3] = {3, 5, 6};
    executor.seed(0, drafts, sizeof(drafts));
    DeviceBuffer draft_tokens_device(sizeof(drafts));
    CUDA_OK(cudaMemcpy(
        draft_tokens_device.pointer,
        drafts,
        sizeof(drafts),
        cudaMemcpyHostToDevice));
    const std::vector<std::uint8_t> mask{
        1,1,1,1,1,1,1,1,
        1,1,1,1,1,1,1,1,
        0,0,1,0,0,0,0,0,
        1,1,1,1,1,1,1,1,
    };
    executor.feed(1, mask.data(), mask.size());
    const std::vector<float> packed{
        0,0,0,9,0,0,0,0,
        0,0,0,0,0,9,0,0,
        0,0,1,0,0,0,9,0,
        0,0,0,0,9,0,0,0,
        0,7,0,0,0,0,0,0,
        0,0,0,0,7,0,0,0,
        7,0,0,0,0,0,0,0,
    };
    DeviceBuffer device = upload_bf16(packed);
    RuntimeExtents extents{
        .row_count = verify_rows,
        .sampled_rows = verify_rows,
    };
    GeneratedInputs inputs{
        .logits = device.pointer,
        .mtp_logits = device.pointer,
        .mtp_drafts = draft_tokens_device.pointer,
        .intrinsic_dtype = 1,
        .vocab = vocab,
        .row_stride = vocab,
        .mtp_logits_row_offset = verify_rows,
    };
    std::string error;
    const bool committed = executor.run(inputs, extents, error);
    std::int32_t accepted[4]{};
    std::int32_t draft_tokens[3]{};
    if (executor.output_ready(2)) {
        executor.take(2, accepted, sizeof(accepted));
    }
    if (executor.output_ready(3)) {
        executor.take(3, draft_tokens, sizeof(draft_tokens));
    }
    expect(
        committed &&
            accepted[0] == 3 && accepted[1] == 5 &&
            accepted[2] == 2 && accepted[3] == -1,
        "generated MTP accepted prefix is [3,5,2,-1]: " + error);
    expect(
        draft_tokens[0] == 1 &&
            draft_tokens[1] == 4 &&
            draft_tokens[2] == 0,
        "generated MTP draft argmax is [1,4,0]");
}

void run_structured_masks(
    const Fixture& fixture,
    const std::map<std::uint8_t, std::unique_ptr<CompiledKernel>>& kernels) {
    GeneratedSingletonExecutor executor(
        fixture.trace, fixture.plans, kernels);
    const std::uint32_t positions[2] = {3, 5};
    const std::uint32_t ancestors[6] = {0, 1, 2, 1, 2, 3};
    const std::uint32_t owners[4] = {0, 1, 2, 3};
    executor.seed(0, positions, sizeof(positions));
    executor.seed(1, ancestors, sizeof(ancestors));
    executor.seed(2, owners, sizeof(owners));
    RuntimeExtents extents;
    GeneratedInputs inputs;
    std::string error;
    expect(
        executor.run(inputs, extents, error),
        "generated structured masks commit: " + error);
    const std::vector<std::vector<std::uint8_t>> expected{
        {1, 1, 1, 1, 0, 0, 1, 1, 1, 1, 1, 1},
        {0, 1, 1, 1, 0, 0, 0, 0, 0, 1, 1, 1},
        {1, 1, 1, 1, 0, 0, 1, 0, 0, 1, 1, 1},
        {1, 1, 1, 0, 0, 1, 1, 1},
    };
    for (std::size_t output = 0; output < expected.size(); ++output) {
        std::vector<std::uint8_t> actual(expected[output].size());
        if (executor.output_ready(output + 3)) {
            executor.take(
                output + 3, actual.data(), actual.size());
        }
        expect(
            actual == expected[output],
            "generated structured mask output " +
                std::to_string(output));
    }
}

void run_dfa(
    const Fixture& fixture,
    const std::map<std::uint8_t, std::unique_ptr<CompiledKernel>>& kernels) {
    GeneratedSingletonExecutor executor(
        fixture.trace, fixture.plans, kernels);
    const std::uint8_t allowed[24] = {
        0,1,1,0,0,0,0,0,
        0,0,0,1,0,0,0,0,
        1,0,0,0,0,0,0,0,
    };
    const std::uint32_t next[24] = {
        0,1,1,0,0,0,0,0,
        0,0,0,2,0,0,0,0,
        2,0,0,0,0,0,0,0,
    };
    const std::uint32_t state = 0;
    executor.seed(0, allowed, sizeof(allowed));
    executor.seed(1, next, sizeof(next));
    executor.seed(2, &state, sizeof(state));
    const std::vector<std::pair<std::vector<float>, std::int32_t>> cases{
        {{0, 0, 1, 0, 0, 9, 0, 0}, 2},
        {{0, 0, 0, 1, 0, 0, 9, 0}, 3},
        {{1, 0, 0, 0, 0, 0, 0, 9}, 0},
    };
    RuntimeExtents extents;
    for (const auto& [logits, expected] : cases) {
        DeviceBuffer device = upload_bf16(logits);
        GeneratedInputs inputs{
            .logits = device.pointer,
            .intrinsic_dtype = 1,
            .vocab = 8,
            .row_stride = 8,
        };
        std::string error;
        const bool committed = executor.run(inputs, extents, error);
        std::int32_t token = -1;
        if (executor.output_ready(3)) {
            executor.take(3, &token, sizeof(token));
        }
        expect(
            committed && token == expected,
            "generated DFA token expected=" +
                std::to_string(expected) +
                " actual=" + std::to_string(token) +
                " error=" + error);
    }
}

void run_pivots(
    const Fixture& fixture,
    const std::map<std::uint8_t, std::unique_ptr<CompiledKernel>>& kernels) {
    GeneratedSingletonExecutor executor(
        fixture.trace, fixture.plans, kernels);
    const float top_p = 0.999f;
    const float threshold = 0.0003f;
    executor.feed(0, &top_p, sizeof(top_p));
    executor.feed(1, &threshold, sizeof(threshold));
    DeviceBuffer device = upload_bf16(
        {0.0f, 1.0f, 9.0f, 2.0f, 0.0f, 0.0f, 0.0f, 3.0f});
    RuntimeExtents extents;
    GeneratedInputs inputs{
        .logits = device.pointer,
        .intrinsic_dtype = 1,
        .vocab = 8,
        .row_stride = 8,
    };
    std::string error;
    const bool committed = executor.run(inputs, extents, error);
    std::array<std::uint8_t, 8> mass{};
    std::array<std::uint8_t, 8> probability{};
    if (executor.output_ready(2)) {
        executor.take(2, mass.data(), mass.size());
    }
    if (executor.output_ready(3)) {
        executor.take(3, probability.data(), probability.size());
    }
    const std::array<std::uint8_t, 8> expected_mass{
        0, 0, 1, 1, 0, 0, 0, 1};
    const std::array<std::uint8_t, 8> expected_probability{
        0, 1, 1, 1, 0, 0, 0, 1};
    expect(
        committed &&
            mass == expected_mass &&
            probability == expected_probability,
        "generated dynamic pivot masks match the Rust golden: " + error);
}

void run_extern_contrastive(
    const Fixture& amateur_fixture,
    const Fixture& expert_fixture,
    const std::map<std::uint8_t, std::unique_ptr<CompiledKernel>>& kernels) {
    GeneratedSingletonExecutor amateur(
        amateur_fixture.trace, amateur_fixture.plans, kernels);
    GeneratedSingletonExecutor expert(
        expert_fixture.trace, expert_fixture.plans, kernels);
    const std::uint8_t allowed[8] = {1,1,1,1,1,1,1,1};
    expert.feed(1, allowed, sizeof(allowed));
    DeviceBuffer amateur_logits = upload_bf16(
        {0, 0, 9, 0, 0, 0, 0, 0});
    DeviceBuffer expert_logits = upload_bf16(
        {0, 0, 9, 0, 0, 8.5f, 0, 0});
    RuntimeExtents extents;
    GeneratedInputs amateur_inputs{
        .logits = amateur_logits.pointer,
        .intrinsic_dtype = 1,
        .vocab = 8,
        .row_stride = 8,
    };
    GeneratedInputs expert_inputs{
        .logits = expert_logits.pointer,
        .intrinsic_dtype = 1,
        .vocab = 8,
        .row_stride = 8,
    };
    std::string error;
    expect(
        !expert.run(expert_inputs, extents, error),
        "generated expert waits for imported amateur logits");
    error.clear();
    expect(
        amateur.run(amateur_inputs, extents, error),
        "generated amateur exports logits: " + error);
    error.clear();
    expect(
        !amateur.run(amateur_inputs, extents, error),
        "generated extern export observes cross-instance back-pressure");
    std::array<float, 8> transferred{};
    if (amateur.output_ready(0)) {
        amateur.take(0, transferred.data(), sizeof(transferred));
    }
    expert.feed(0, transferred.data(), sizeof(transferred));
    error.clear();
    const bool expert_committed =
        expert.run(expert_inputs, extents, error);
    std::int32_t token = -1;
    if (expert.output_ready(2)) {
        expert.take(2, &token, sizeof(token));
    }
    expect(
        expert_committed && token == 5,
        "generated extern contrastive expert picks token 5: " + error);
    error.clear();
    expect(
        amateur.run(amateur_inputs, extents, error),
        "generated amateur resumes after imported logits drain: " + error);
}

void validate_second_party_regions(const Fixture& fixture) {
    bool kernel = false;
    bool sink = false;
    bool exact = true;
    for (const auto& stage : fixture.plans) {
        exact = exact &&
            !stage.singleton.whole_stage_fallback &&
            !stage.fused.whole_stage_fallback;
        for (std::size_t node = 0; node < stage.ops.size(); ++node) {
            const std::uint8_t tag = stage.ops[node].op.tag;
            if (tag != PTIR_OP_KERNEL_CALL &&
                tag != PTIR_OP_SINK_CALL) {
                continue;
            }
            const auto& singleton = stage.singleton.regions[node];
            exact = exact &&
                singleton.library &&
                singleton.library_op == PTIR_LIBRARY_SECOND_PARTY &&
                singleton.schedule == PTIR_SCHEDULE_LIBRARY;
            kernel = kernel || tag == PTIR_OP_KERNEL_CALL;
            sink = sink || tag == PTIR_OP_SINK_CALL;
        }
    }
    expect(
        exact && kernel && sink,
        fixture.name +
            " uses explicit second-party library regions without fallback");
}

void run_module_cache_contract(const Fixture& fixture) {
    ModuleCache cache;
    CompileFailureKind failure = CompileFailureKind::None;
    std::string error;
    const auto first = cache.compile_program(
        fixture.hash, fixture.plans, failure, error);
    const ModuleCacheStats after_first = cache.stats();
    const auto second = cache.compile_program(
        fixture.hash, fixture.plans, failure, error);
    const ModuleCacheStats after_second = cache.stats();
    expect(
        first != nullptr && second == first &&
            after_first.compilations + after_first.disk_hits != 0 &&
            after_second.compilations == after_first.compilations &&
            after_second.disk_hits == after_first.disk_hits &&
            after_second.memory_hits > after_first.memory_hits,
        "generated module cache compiles at registration and reuses exact identity");

    auto remapped_channels = fixture.plans;
    for (auto& stage : remapped_channels) {
        for (auto& channel : stage.channel_bindings) {
            channel += 100;
        }
    }
    failure = CompileFailureKind::None;
    error.clear();
    const auto remapped = cache.compile_program(
        fixture.hash ^ 0x40,
        remapped_channels,
        failure,
        error);
    const ModuleCacheStats after_remapped = cache.stats();
    expect(
        remapped != nullptr &&
            remapped->stages.front() == first->stages.front() &&
            after_remapped.stage_entries == after_second.stage_entries &&
            after_remapped.compilations == after_second.compilations,
        "generated executable identity excludes program-local channel ids");

    bool shape_variants_cached = true;
    for (std::uint64_t variant = 0; variant < 129; ++variant) {
        failure = CompileFailureKind::None;
        error.clear();
        shape_variants_cached =
            cache.compile_program(
                fixture.hash ^ (0x10000 + variant),
                remapped_channels,
                failure,
                error) != nullptr &&
            shape_variants_cached;
    }
    const ModuleCacheStats after_shape_variants = cache.stats();
    expect(
        shape_variants_cached &&
            after_shape_variants.program_entries >= 131 &&
            after_shape_variants.stage_entries ==
                after_remapped.stage_entries,
        "generated program cache retains more than 128 shape variants "
        "without multiplying stage modules");

    auto different_semantics = fixture.plans;
    const auto changed = std::find_if(
        different_semantics.front().ops.begin(),
        different_semantics.front().ops.end(),
        [](const auto& normalized) {
            return normalized.op.tag == PTIR_OP_REDUCE_ARGMAX;
        });
    if (changed != different_semantics.front().ops.end()) {
        changed->op.tag = PTIR_OP_REDUCE_MAX;
    }
    failure = CompileFailureKind::None;
    error.clear();
    const auto distinct = cache.compile_program(
        fixture.hash ^ 0x80,
        different_semantics,
        failure,
        error);
    const ModuleCacheStats after_distinct = cache.stats();
    expect(
        changed != different_semantics.front().ops.end() &&
            distinct != nullptr &&
            distinct->stages.front() != first->stages.front() &&
            after_distinct.stage_entries > after_remapped.stage_entries &&
            after_distinct.compilations + after_distinct.disk_hits >
                after_remapped.compilations + after_remapped.disk_hits,
        "generated module cache keys the complete normalized stage semantics");

    auto invalid = fixture.plans;
    invalid.front().signature.push_back(0xff);
    invalid.front().signature_hash = container::fnv1a64(
        invalid.front().signature.data(),
        invalid.front().signature.size());
    invalid.front().ops.front().op.tag = PTIR_OP_KERNEL_CALL;
    invalid.front().ops.front().op.name_idx = 0;
    invalid.front().names = {"cuda.identity"};
    failure = CompileFailureKind::None;
    error.clear();
    const auto rejected = cache.compile_program(
        fixture.hash ^ 0x100, invalid, failure, error);
    const ModuleCacheStats after_reject = cache.stats();
    failure = CompileFailureKind::None;
    error.clear();
    const auto rejected_again = cache.compile_program(
        fixture.hash ^ 0x200, invalid, failure, error);
    const ModuleCacheStats after_negative_hit = cache.stats();
    expect(
        rejected == nullptr && rejected_again == nullptr &&
            failure == CompileFailureKind::Deterministic &&
            after_reject.negative_entries == 1 &&
            after_reject.program_entries ==
                after_distinct.program_entries &&
            after_reject.stage_entries ==
                after_distinct.stage_entries &&
            after_negative_hit.negative_hits >
                after_reject.negative_hits,
        "generated module cache negative-caches deterministic failures only");

    const auto cache_directory =
        std::filesystem::temp_directory_path() /
        ("pie-ptir-cache-" + std::to_string(fixture.hash));
    std::error_code filesystem_error;
    std::filesystem::remove_all(cache_directory, filesystem_error);
    {
        ModuleCache writer(cache_directory);
        failure = CompileFailureKind::None;
        error.clear();
        const auto compiled = writer.compile_program(
            fixture.hash, fixture.plans, failure, error);
        const auto writer_stats = writer.stats();
        expect(
            compiled != nullptr &&
                writer_stats.compilations != 0 &&
                writer_stats.disk_writes != 0,
            "generated module cache atomically persists compiled PTX");
    }
    {
        ModuleCache reader(cache_directory);
        failure = CompileFailureKind::None;
        error.clear();
        const auto loaded = reader.compile_program(
            fixture.hash, fixture.plans, failure, error);
        const auto reader_stats = reader.stats();
        expect(
            loaded != nullptr &&
                reader_stats.compilations == 0 &&
                reader_stats.disk_hits != 0,
            "generated module cache reloads PTX without NVRTC");
    }
    std::filesystem::remove_all(cache_directory, filesystem_error);
}

}  // namespace

int main(int argc, char** argv) {
    if (argc != 2) {
        std::fprintf(stderr, "usage: %s <golden-directory>\n", argv[0]);
        return 2;
    }
    try {
        CU_OK(cuInit(0));
        CUdevice device = 0;
        CUcontext context = nullptr;
        CU_OK(cuDeviceGet(&device, 0));
        CU_OK(cuDevicePrimaryCtxRetain(&context, device));
        CU_OK(cuCtxSetCurrent(context));
        validate_direct_argmax_reshape_guard();
        validate_inactive_lane_state_codegen();

        std::vector<Fixture> fixtures;
        for (const auto& entry :
             std::filesystem::directory_iterator(argv[1])) {
            if (!entry.is_regular_file() ||
                entry.path().extension() != ".txt") {
                continue;
            }
            std::vector<Fixture> decoded;
            expect(
                load_fixtures(entry.path(), decoded),
                "load generated-singleton fixture " +
                    entry.path().filename().string());
            for (Fixture& fixture : decoded) {
                if (fixture.verdict == "OK") {
                    fixtures.push_back(std::move(fixture));
                }
            }
        }
        std::sort(
            fixtures.begin(),
            fixtures.end(),
            [](const Fixture& left, const Fixture& right) {
                return left.name < right.name;
            });

        std::set<std::uint8_t> fixture_tags;
        for (const Fixture& fixture : fixtures) {
            for (const auto& stage : fixture.plans) {
                std::vector<GeneratedOpMeta> operations;
                std::string error;
                const bool valid =
                    validate_singleton_plan(stage, operations, error);
                expect(
                    valid,
                    fixture.name +
                        ": validates singleton partition (" +
                        error + ")");
                if (!valid) continue;
                for (const auto& operation : operations) {
                    fixture_tags.insert(operation.op.tag);
                }
            }
        }

#define PTIR_GENERATED_TAG(name, tag, arity, results) \
        std::uint8_t{tag},
        const std::vector<std::uint8_t> all_tags = {
            PTIR_OP_LIST(PTIR_GENERATED_TAG)
        };
#undef PTIR_GENERATED_TAG

        std::map<std::uint8_t, std::unique_ptr<CompiledKernel>> kernels;
        for (const std::uint8_t tag : all_tags) {
            GeneratedKernelSource generated =
                emit_singleton_region_cuda("coverage", tag);
            expect(
                generated.ok && !generated.source.empty(),
                "singleton emitter covers tag " +
                    std::to_string(tag));
            kernels.emplace(tag, compile_kernel(tag));
        }
        expect(
            kernels.size() == all_tags.size() &&
                std::all_of(
                    fixture_tags.begin(),
                    fixture_tags.end(),
                    [&](std::uint8_t tag) {
                        return kernels.contains(tag);
                    }),
            "every first-party and authoritative golden singleton tag "
            "NVRTC-compiles");
        std::size_t compiled_fused_regions = 0;
        for (const Fixture& fixture : fixtures) {
            for (std::size_t stage_index = 0;
                 stage_index < fixture.plans.size();
                 ++stage_index) {
                const auto& stage = fixture.plans[stage_index];
                for (std::size_t region_index = 0;
                     region_index < stage.fused.regions.size();
                     ++region_index) {
                    const auto& region = stage.fused.regions[region_index];
                    if (region.library) continue;
                    const std::string entry =
                        "ptir_fused_" +
                        std::to_string(compiled_fused_regions);
                    auto compiled = compile_source(
                        emit_fused_region_cuda(entry, stage, region),
                        fixture.name + " fused region " +
                            std::to_string(region_index));
                    ++compiled_fused_regions;
                }
            }
        }
        expect(
            compiled_fused_regions != 0,
            "every canonical generated fused region NVRTC-compiles");
        run_ambient_rng_contract(kernels);
        run_rank_threshold_contract(kernels);
        run_stock_library_numeric_contract();
        run_intrinsic_contract(kernels);
        run_cross_stage_pending_contract(kernels);

        auto fixture = [&](const std::string& name) -> const Fixture* {
            const auto found = std::find_if(
                fixtures.begin(),
                fixtures.end(),
                [&](const Fixture& candidate) {
                    return candidate.name == name;
                });
            expect(
                found != fixtures.end(),
                name + " generated singleton fixture exists");
            return found == fixtures.end() ? nullptr : &*found;
        };
        if (const Fixture* value = fixture("counter_pingpong")) {
            run_counter(*value, kernels);
        }
        if (const Fixture* value = fixture("greedy_argmax")) {
            run_greedy(*value, kernels);
            run_grouped_fused_greedy(*value);
            run_grouped_fused_greedy(*value, 151936);
            run_module_cache_contract(*value);
        }
        if (const Fixture* value = fixture("nucleus_sample")) {
            run_nucleus(*value, kernels);
        }
        if (const Fixture* value = fixture("section3_masked_gumbel")) {
            run_masked_gumbel(*value, kernels);
        }
        if (const Fixture* value = fixture("matrix_select_mask")) {
            run_matrix_select(*value, kernels);
        }
        if (const Fixture* value = fixture("matrix_mask_apply_packed")) {
            run_matrix_mask_apply(*value, kernels);
        }
        if (const Fixture* value = fixture("beam_epilogue")) {
            run_beam(*value, kernels);
        }
        if (const Fixture* value = fixture("mtp_verify_tail")) {
            run_mtp(*value, kernels);
        }
        if (const Fixture* value = fixture("structured_masks")) {
            run_structured_masks(*value, kernels);
        }
        if (const Fixture* value = fixture("dfa_ingraph")) {
            run_dfa(*value, kernels);
        }
        if (const Fixture* value = fixture("pivot_predicates_multistage")) {
            run_pivots(*value, kernels);
        }
        const Fixture* amateur = fixture("extern_amateur");
        const Fixture* expert = fixture("extern_expert");
        if (amateur != nullptr && expert != nullptr) {
            run_extern_contrastive(*amateur, *expert, kernels);
        }
        if (const Fixture* value = fixture("pentathlon_expand")) {
            validate_second_party_regions(*value);
        }
        if (const Fixture* value = fixture("pentathlon_rollout")) {
            validate_second_party_regions(*value);
        }

        kernels.clear();
        CU_OK(cuDevicePrimaryCtxRelease(device));
    } catch (const std::exception& exception) {
        std::fprintf(stderr, "FAIL: %s\n", exception.what());
        return 1;
    }
    std::printf(
        "PTIR generated singleton oracle: %d failure(s)\n",
        failures);
    return failures == 0 ? 0 : 1;
}
