#include "ops/flashinfer_moe.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <mutex>
#include <optional>
#include <stdexcept>
#include <string>
#include <vector>

#include <cuda_bf16.h>

#include "cutlass_fused_moe_kernels.cuh"

namespace pie_cuda_driver::ops {
namespace {

namespace ck = tensorrt_llm::kernels::cutlass_kernels;
namespace ce = tensorrt_llm::cutlass_extensions;
namespace tk = tensorrt_llm::kernels;

using Runner = ck::CutlassMoeFCRunner<__nv_bfloat16, __nv_bfloat16>;

bool env_truthy(const char* value) {
    if (value == nullptr || value[0] == '\0') return false;
    return value[0] == '1' || value[0] == 'y' || value[0] == 'Y' ||
           value[0] == 't' || value[0] == 'T' || value[0] == 'o' ||
           value[0] == 'O';
}

struct RunnerState {
    std::once_flag init_once;
    std::unique_ptr<Runner> runner;
    bool ready = false;
    std::exception_ptr init_error;
};

RunnerState& state() {
    constexpr int kMaxCudaDevices = 16;
    static std::array<RunnerState, kMaxCudaDevices> states;
    int device = 0;
    cudaError_t st = cudaGetDevice(&device);
    if (st != cudaSuccess) {
        throw std::runtime_error(
            std::string("flashinfer CUTLASS MoE: cudaGetDevice failed: ") +
            cudaGetErrorString(st));
    }
    if (device < 0 || device >= kMaxCudaDevices) {
        throw std::runtime_error(
            "flashinfer CUTLASS MoE: CUDA device index exceeds runner cache");
    }
    return states[device];
}

bool log_enabled() {
    return env_truthy(std::getenv("PIE_NEMOTRON_FLASHINFER_MOE_LOG"));
}

bool use_first_tactic_selection() {
    const char* v = std::getenv("PIE_NEMOTRON_FLASHINFER_MOE_SELECT");
    return v != nullptr && std::strcmp(v, "first") == 0;
}

bool use_raw_tactic_selection() {
    const char* v = std::getenv("PIE_NEMOTRON_FLASHINFER_MOE_SELECT");
    return v != nullptr && std::strcmp(v, "raw") == 0;
}

std::optional<ce::CutlassGemmConfig> select_first_profile(
    const std::vector<ce::CutlassGemmConfig>& configs,
    const char* name) {
    if (configs.empty()) return std::nullopt;
    if (log_enabled()) {
        std::fprintf(
            stderr,
            "[pie-driver-cuda] FlashInfer MoE %s selected first profile "
            "total=%zu\n",
            name, configs.size());
    }
    return configs.front();
}

std::optional<ce::CutlassGemmConfig> first_supported(
    Runner& runner,
    const std::vector<ce::CutlassGemmConfig>& configs,
    std::optional<ce::CutlassGemmConfig::EpilogueFusionType> fusion,
    int supported_index,
    const char* name) {
    int seen = 0;
    int supported = 0;
    int index = -1;
    int selected_index = -1;
    std::optional<ce::CutlassGemmConfig> selected;
    for (const auto& cfg : configs) {
        ++index;
        if (fusion && cfg.epilogue_fusion_type != *fusion) continue;
        ++seen;
        if (runner.queryOccupancyForConfig(cfg) > 0) {
            if (supported == supported_index) {
                selected = cfg;
                selected_index = index;
            }
            ++supported;
        }
    }
    if (log_enabled()) {
        if (selected) {
            std::fprintf(
                stderr,
                "[pie-driver-cuda] FlashInfer MoE %s selected "
                "supported_index=%d raw_index=%d supported=%d seen=%d total=%zu\n",
                name, supported_index, selected_index, supported, seen, configs.size());
        } else {
            std::fprintf(
                stderr,
                "[pie-driver-cuda] FlashInfer MoE %s no tactic for "
                "supported_index=%d supported=%d seen=%d total=%zu\n",
                name, supported_index, supported, seen, configs.size());
        }
    }
    return selected;
}

std::optional<ce::CutlassGemmConfig> select_raw_profile(
    const std::vector<ce::CutlassGemmConfig>& configs,
    int raw_index,
    const char* name) {
    if (configs.empty()) return std::nullopt;
    const int index = std::min(
        std::max(0, raw_index),
        static_cast<int>(configs.size()) - 1);
    if (log_enabled()) {
        std::fprintf(
            stderr,
            "[pie-driver-cuda] FlashInfer MoE %s selected raw_index=%d "
            "total=%zu\n",
            name, index, configs.size());
    }
    return configs[static_cast<std::size_t>(index)];
}

const char* fusion_name(ce::CutlassGemmConfig::EpilogueFusionType fusion) {
    switch (fusion) {
    case ce::CutlassGemmConfig::EpilogueFusionType::NONE: return "none";
    case ce::CutlassGemmConfig::EpilogueFusionType::FINALIZE: return "finalize";
    }
    return "unknown";
}

int env_index(const char* name) {
    const char* v = std::getenv(name);
    if (v == nullptr || v[0] == '\0') return 0;
    return std::max(0, std::atoi(v));
}

std::optional<ce::CutlassGemmConfig::EpilogueFusionType> requested_gemm2_fusion() {
    const char* v = std::getenv("PIE_NEMOTRON_FLASHINFER_MOE_GEMM2");
    if (v == nullptr || v[0] == '\0' || std::strcmp(v, "auto") == 0) {
        return std::nullopt;
    }
    if (std::strcmp(v, "none") == 0) {
        return ce::CutlassGemmConfig::EpilogueFusionType::NONE;
    }
    if (std::strcmp(v, "finalize") == 0) {
        return ce::CutlassGemmConfig::EpilogueFusionType::FINALIZE;
    }
    throw std::runtime_error(
        "PIE_NEMOTRON_FLASHINFER_MOE_GEMM2 must be auto, none, or finalize");
}

void log_config(const char* name, const ce::CutlassGemmConfig& cfg) {
    if (!log_enabled()) return;
    std::fprintf(
        stderr,
        "[pie-driver-cuda] FlashInfer MoE %s tactic: fusion=%s "
        "tma=%d swap_ab=%d sm=%d tile80=%d tile90=%d mainloop=%d "
        "epilogue=%d cluster=%d split_k=%d stages=%d\n",
        name,
        fusion_name(cfg.epilogue_fusion_type),
        cfg.is_tma_warp_specialized ? 1 : 0,
        cfg.swap_ab ? 1 : 0,
        cfg.sm_version,
        static_cast<int>(cfg.tile_config_sm80),
        static_cast<int>(cfg.tile_config_sm90),
        static_cast<int>(cfg.mainloop_schedule),
        static_cast<int>(cfg.epilogue_schedule),
        static_cast<int>(cfg.cluster_shape),
        cfg.split_k_factor,
        cfg.stages);
}

Runner& get_runner() {
    RunnerState& s = state();
    std::call_once(s.init_once, [&] {
        try {
            auto runner = std::make_unique<Runner>();
            auto gemm1 = runner->getTactics(ck::MoeGemmId::GEMM_1);
            auto gemm2 = runner->getTactics(ck::MoeGemmId::GEMM_2);
            std::optional<ce::CutlassGemmConfig> best_gemm1;
            std::optional<ce::CutlassGemmConfig> best_gemm2;
            if (use_raw_tactic_selection()) {
                best_gemm1 = select_raw_profile(
                    gemm1,
                    env_index("PIE_NEMOTRON_FLASHINFER_MOE_GEMM1_INDEX"),
                    "GEMM1");
                best_gemm2 = select_raw_profile(
                    gemm2,
                    env_index("PIE_NEMOTRON_FLASHINFER_MOE_GEMM2_INDEX"),
                    "GEMM2");
            } else if (use_first_tactic_selection()) {
                best_gemm1 = select_first_profile(gemm1, "GEMM1");
                best_gemm2 = select_first_profile(gemm2, "GEMM2");
            } else {
                // Default: first tactic the runner reports as occupancy-viable.
                //
                // For GEMM2 prefer the FINALIZE epilogue: it folds the topk
                // weighted reduction (and the unpermute) into the GEMM epilogue
                // instead of running a separate finalize pass. Measured at GLM
                // shapes (M=128, H=6144, I=2048, E=8, topk=8) this is 147.0 us
                // vs 174.6 us for the unfused epilogue -- the single largest
                // knob on this path. Fall back to any supported tactic if the
                // fused epilogue is unavailable for this problem.
                best_gemm1 = first_supported(
                    *runner, gemm1, std::nullopt,
                    env_index("PIE_NEMOTRON_FLASHINFER_MOE_GEMM1_INDEX"),
                    "GEMM1");
                const auto gemm2_fusion = requested_gemm2_fusion();
                const int gemm2_index =
                    env_index("PIE_NEMOTRON_FLASHINFER_MOE_GEMM2_INDEX");
                best_gemm2 = first_supported(
                    *runner, gemm2,
                    gemm2_fusion.value_or(
                        ce::CutlassGemmConfig::EpilogueFusionType::FINALIZE),
                    gemm2_index, "GEMM2");
                if (!best_gemm2 && !gemm2_fusion) {
                    best_gemm2 = first_supported(
                        *runner, gemm2, std::nullopt, gemm2_index, "GEMM2");
                }
            }
            if (!best_gemm1 || !best_gemm2) {
                throw std::runtime_error(
                    "flashinfer CUTLASS MoE: no supported BF16 tactics");
            }
            log_config("GEMM1", *best_gemm1);
            log_config("GEMM2", *best_gemm2);
            runner->setTactic(best_gemm1, best_gemm2);
            s.runner = std::move(runner);
            s.ready = true;
        } catch (...) {
            s.init_error = std::current_exception();
        }
    });

    if (!s.ready) {
        if (s.init_error) std::rethrow_exception(s.init_error);
        throw std::runtime_error("flashinfer CUTLASS MoE: runner not initialized");
    }
    return *s.runner;
}

ck::ActivationType to_cutlass_activation(MoeActivation a) {
    switch (a) {
        case MoeActivation::Swiglu: return ck::ActivationType::Swiglu;
        case MoeActivation::Relu2:
        default:                    return ck::ActivationType::Relu2;
    }
}

ck::MOEParallelismConfig parallelism_config(int tp_size, int tp_rank) {
    return ck::MOEParallelismConfig(std::max(1, tp_size), tp_rank, 1, 0);
}

}  // namespace

bool flashinfer_cutlass_moe_enabled() {
    // Two consumers, each with its own switch: nemotron_h (Relu2) and the
    // qwen3_5 MoE decode (Swiglu). Either one turns the runner on.
    static const bool enabled = [] {
        if (env_truthy(std::getenv("PIE_NEMOTRON_FLASHINFER_MOE"))) return true;
        const char* q = std::getenv("PIE_QWEN35_MOE_FLASHINFER");
        return q == nullptr || q[0] == '\0' || q[0] != '0';
    }();
    return enabled;
}

std::size_t flashinfer_cutlass_moe_workspace_bytes(
    MoeActivation activation,
    int num_rows,
    int hidden_size,
    int inter_size,
    int num_experts,
    int experts_per_token,
    int tp_size,
    int tp_rank) {
    if (num_rows <= 0 || hidden_size <= 0 || inter_size <= 0 ||
        num_experts <= 0 || experts_per_token <= 0) {
        return 0;
    }
    Runner& runner = get_runner();
    return runner.getWorkspaceSize(
        num_rows,
        hidden_size,
        inter_size,
        num_experts,
        experts_per_token,
        to_cutlass_activation(activation),
        parallelism_config(tp_size, tp_rank),
        false,
        false,
        false,
        false,
        false);
}

bool flashinfer_cutlass_moe_bf16(
    MoeActivation activation,
    const std::uint16_t* input,
    const std::int32_t* token_selected_experts,
    const float* token_final_scales,
    const std::uint16_t* fc1_expert_weights,
    const std::uint16_t* fc2_expert_weights,
    std::uint16_t* output,
    std::uint8_t* workspace,
    std::size_t workspace_bytes,
    std::int32_t* unpermuted_row_to_permuted_row,
    int num_rows,
    int hidden_size,
    int inter_size,
    int num_experts,
    int experts_per_token,
    int tp_size,
    int tp_rank,
    cudaStream_t stream) {
    if (input == nullptr || token_selected_experts == nullptr ||
        token_final_scales == nullptr || fc1_expert_weights == nullptr ||
        fc2_expert_weights == nullptr || output == nullptr ||
        workspace == nullptr || unpermuted_row_to_permuted_row == nullptr) {
        return false;
    }
    const std::size_t needed = flashinfer_cutlass_moe_workspace_bytes(
        activation, num_rows, hidden_size, inter_size, num_experts,
        experts_per_token, tp_size, tp_rank);
    if (needed == 0 || workspace_bytes < needed) return false;

    Runner& runner = get_runner();
    ck::QuantParams quant_params{};
    tk::LoraParams lora_params{};
    ck::MoeMinLatencyParams min_latency_params{};
    runner.runMoe(
        input,
        nullptr,
        false,
        token_selected_experts,
        token_final_scales,
        fc1_expert_weights,
        nullptr,
        ck::ActivationParams(to_cutlass_activation(activation)),
        fc2_expert_weights,
        nullptr,
        quant_params,
        num_rows,
        hidden_size,
        hidden_size,
        inter_size,
        num_experts,
        experts_per_token,
        reinterpret_cast<char*>(workspace),
        output,
        unpermuted_row_to_permuted_row,
        parallelism_config(tp_size, tp_rank),
        false,
        false,
        lora_params,
        false,
        false,
        false,
        min_latency_params,
        false,
        stream);
    return true;
}

}  // namespace pie_cuda_driver::ops
