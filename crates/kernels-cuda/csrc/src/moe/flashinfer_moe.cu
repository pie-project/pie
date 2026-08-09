#include "moe/flashinfer_moe.hpp"

// TWO STANDARD HEADERS, DOWN FROM FOURTEEN.
//
// What was here: <algorithm> <array> <cstdio> <cstdlib> <limits> <cstring>
// <memory> <mutex> <optional> <stdexcept> <string> <unordered_map> <utility>
// <vector>. A `std::mutex` and a `std::unordered_map` are not things NVRTC
// could compile; they were the evidence this file was never NVRTC's to
// compile, and they are `driver-cuda/src/fire/flashinfer_moe.rs` now.
//
// <cstring> is `std::strncpy`, for copying `what()` into the caller's buffer.
// <exception> is `std::exception`, for catching it. Both exist only because
// an exception may not cross the C ABI; there is no third reason for a
// standard header in this file and adding one is the host program coming
// back.
#include <cstring>
#include <exception>

#include <cuda_bf16.h>

// THE PURE KERNEL HEADER. 4,991 lines, 19 `__global__`, 52 `cutlass::`,
// CPM-fetched into `target/**/_deps/flashinfer-src` at configure time. It is
// upstream DEVICE TEXT, and it is the LAST thing in this family still
// compiled ahead of time. That is a state, not a settlement: the principle
// has no exception any more -- every CUDA kernel is compiled by NVRTC at run
// time, no nvcc under any circumstances -- so this target, its generated
// `_SM90_`/`_SM100_` lists and the CPM fetch that feeds them all have to go.
// Whether this header can go through NVRTC is MEASURED, and mostly yes: a
// concrete sm90 ptr-array grouped GEMM (`MainloopSm90ArrayTmaGmmaWarpSpecialized`,
// 128x128x64, bf16) compiles under NVRTC at compute_90a to 1,245,452 B of PTX
// with exactly one `.entry`. §13.6's "a FlashInfer patch set plus ~39
// bit-exact device intrinsics" was priced for FA2's prefill lattice and does
// NOT transfer: CUTLASS cost three names in namespace `std` (`is_pointer_v`,
// `void_t`, `max`), three specific cub includes instead of the umbrella, and
// two `griddepcontrol` asm lines. `driver-cuda/src/fire/flashinfer_moe.rs`'s
// module doc has the probe table and the one thing still unresolved (the
// FINALIZE scatter epilogue, which failed on parameterisation, not on NVRTC).
#include "cutlass_fused_moe_kernels.cuh"

namespace pie_cuda_driver::kernels::moe {
namespace {

namespace ck = tensorrt_llm::kernels::cutlass_kernels;
namespace ce = tensorrt_llm::cutlass_extensions;
namespace tk = tensorrt_llm::kernels;

using Runner = ck::CutlassMoeFCRunner<__nv_bfloat16, __nv_bfloat16>;

// The seam's object. A `Runner` by value and nothing beside it: the tactic
// vectors, the tuned map, the default pair and the once-flag that used to
// live in `RunnerState` are the Rust's, per device, in `fire/
// flashinfer_moe.rs`.
struct Seam {
    Runner runner;
};

void report(char* err, std::size_t cap, const char* what) {
    if (err == nullptr || cap == 0) return;
    std::strncpy(err, what, cap - 1);
    err[cap - 1] = '\0';
}

// `PIE_K_GUARD`'s job, at the one seam that is left. `...` catches what
// `std::exception` does not, because a launcher that throws an `int` still
// may not unwind through a Rust frame.
#define PIE_MOE_GUARD(body)                                     \
    try {                                                       \
        body                                                    \
    } catch (const ::std::exception& e) {                       \
        report(err, err_cap, e.what());                         \
        return -1;                                              \
    } catch (...) {                                             \
        report(err, err_cap, "unknown C++ exception");           \
        return -1;                                              \
    }

// The only decision left in this file, and it is here because both sides of
// it are C++ enumerators.
//
// Upstream's `Geglu` dispatches to `EpilogueOpDefaultFtGelu` and is accepted
// by `supportsFusedGatedActivation`; `GegluTanh` is neither, so it would drop
// to the unfused gate for the same math.
ck::ActivationType to_cutlass_activation(MoeActivation a) {
    switch (a) {
        case MoeActivation::Swiglu: return ck::ActivationType::Swiglu;
        case MoeActivation::Geglu:  return ck::ActivationType::Geglu;
        case MoeActivation::Relu2:
        default:                    return ck::ActivationType::Relu2;
    }
}

// `MOEParallelismConfig(tp_size, tp_rank, ep_size, ep_rank)`. The Rust
// supplies the first two -- INCLUDING the `max(1, tp_size)` clamp, which is a
// host decision and is spelled in `fire::flashinfer_moe::parallelism_config`.
// The trailing `1, 0` are structural: pie runs no expert parallelism, and a
// value the caller cannot vary is not an argument.
ck::MOEParallelismConfig parallelism(int tp_size, int tp_rank) {
    return ck::MOEParallelismConfig(tp_size, tp_rank, 1, 0);
}

int fusion_of(ce::CutlassGemmConfig::EpilogueFusionType f) {
    switch (f) {
        case ce::CutlassGemmConfig::EpilogueFusionType::NONE:
            return PIE_MOE_FUSION_NONE;
        case ce::CutlassGemmConfig::EpilogueFusionType::FINALIZE:
            return PIE_MOE_FUSION_FINALIZE;
    }
    return PIE_MOE_FUSION_UNKNOWN;
}

// `sm_version` decides which of the four tile fields is live; printing the
// wrong one shows a constant `heuristic` for every tactic. The four field
// NAMES are the whole reason this is not in Rust.
int tile_of(const ce::CutlassGemmConfig& cfg) {
    if (cfg.sm_version >= 120) return static_cast<int>(cfg.tile_config_sm120);
    if (cfg.sm_version >= 100) return static_cast<int>(cfg.tile_config_sm100);
    if (cfg.sm_version >= 90) return static_cast<int>(cfg.tile_config_sm90);
    return static_cast<int>(cfg.tile_config_sm80);
}

ck::MoeGemmId gemm_of(int gemm_id) {
    return gemm_id == PIE_MOE_GEMM_1 ? ck::MoeGemmId::GEMM_1
                                     : ck::MoeGemmId::GEMM_2;
}

}  // namespace

extern "C" {

void* pie_moe_cutlass_create(char* err, std::size_t err_cap) {
    try {
        return new Seam();
    } catch (const std::exception& e) {
        report(err, err_cap, e.what());
        return nullptr;
    } catch (...) {
        report(err, err_cap, "unknown C++ exception");
        return nullptr;
    }
}

int pie_moe_cutlass_tactics(void* runner, int gemm_id, PieMoeTactic* out,
                            int cap, char* err, std::size_t err_cap) {
    if (runner == nullptr) {
        report(err, err_cap, "null runner");
        return -1;
    }
    PIE_MOE_GUARD({
        Seam& seam = *static_cast<Seam*>(runner);
        // `auto`, never `std::vector<...>`: the type is the kernel header's
        // and naming it here would be this file owning a container again.
        auto configs = seam.runner.getTactics(gemm_of(gemm_id));
        const int total = static_cast<int>(configs.size());
        const int n = (out == nullptr || cap < 0) ? 0 : (cap < total ? cap : total);
        for (int i = 0; i < n; ++i) {
            const auto& cfg = configs[static_cast<std::size_t>(i)];
            PieMoeTactic& t = out[i];
            t.fusion = fusion_of(cfg.epilogue_fusion_type);
            t.is_tma_warp_specialized = cfg.is_tma_warp_specialized ? 1 : 0;
            t.swap_ab = cfg.swap_ab ? 1 : 0;
            t.sm_version = cfg.sm_version;
            t.tile = tile_of(cfg);
            t.mainloop_schedule = static_cast<int>(cfg.mainloop_schedule);
            t.epilogue_schedule = static_cast<int>(cfg.epilogue_schedule);
            t.cluster_shape = static_cast<int>(cfg.cluster_shape);
            t.dynamic_cluster_shape = static_cast<int>(cfg.dynamic_cluster_shape);
            t.fallback_cluster_shape = static_cast<int>(cfg.fallback_cluster_shape);
            t.split_k_factor = cfg.split_k_factor;
            t.stages = cfg.stages;
            t.occupancy = seam.runner.queryOccupancyForConfig(cfg);
        }
        return total;
    })
}

int pie_moe_cutlass_set_tactic(void* runner, int gemm1, int gemm2, char* err,
                               std::size_t err_cap) {
    if (runner == nullptr) {
        report(err, err_cap, "null runner");
        return -1;
    }
    PIE_MOE_GUARD({
        Seam& seam = *static_cast<Seam*>(runner);
        auto c1 = seam.runner.getTactics(ck::MoeGemmId::GEMM_1);
        auto c2 = seam.runner.getTactics(ck::MoeGemmId::GEMM_2);
        if (gemm1 < 0 || gemm2 < 0 ||
            static_cast<std::size_t>(gemm1) >= c1.size() ||
            static_cast<std::size_t>(gemm2) >= c2.size()) {
            report(err, err_cap, "tactic index out of range");
            return -1;
        }
        seam.runner.setTactic(c1[static_cast<std::size_t>(gemm1)],
                              c2[static_cast<std::size_t>(gemm2)]);
        return 0;
    })
}

int pie_moe_cutlass_workspace_size(void* runner, MoeActivation activation,
                                   int num_rows, int hidden_size,
                                   int inter_size, int num_experts,
                                   int experts_per_token, int tp_size,
                                   int tp_rank, std::size_t* out, char* err,
                                   std::size_t err_cap) {
    if (runner == nullptr || out == nullptr) {
        report(err, err_cap, "null runner or output");
        return -1;
    }
    PIE_MOE_GUARD({
        Seam& seam = *static_cast<Seam*>(runner);
        // The five trailing `false`s the C++ always passed: use_lora,
        // use_fp8_block_scaling, min_latency_mode, use_awq and the
        // enable_alltoall the OSS path does not run. Structural, like the
        // `1, 0` in `parallelism`.
        *out = seam.runner.getWorkspaceSize(
            num_rows, hidden_size, inter_size, num_experts, experts_per_token,
            to_cutlass_activation(activation), parallelism(tp_size, tp_rank),
            false, false, false, false, false);
        return 0;
    })
}

int pie_moe_cutlass_run(void* runner, MoeActivation activation,
                        const std::uint16_t* input,
                        const std::int32_t* token_selected_experts,
                        const float* token_final_scales,
                        const std::uint16_t* fc1_expert_weights,
                        const std::uint16_t* fc2_expert_weights,
                        std::uint16_t* output, std::uint8_t* workspace,
                        std::int32_t* unpermuted_row_to_permuted_row,
                        int num_rows, int hidden_size, int inter_size,
                        int num_experts, int experts_per_token, int tp_size,
                        int tp_rank, cudaStream_t stream, char* err,
                        std::size_t err_cap) {
    if (runner == nullptr) {
        report(err, err_cap, "null runner");
        return -1;
    }
    PIE_MOE_GUARD({
        Seam& seam = *static_cast<Seam*>(runner);
        ck::QuantParams quant_params{};
        tk::LoraParams lora_params{};
        ck::MoeMinLatencyParams min_latency_params{};
        seam.runner.runMoe(
            input, nullptr, false, token_selected_experts, token_final_scales,
            fc1_expert_weights, nullptr,
            ck::ActivationParams(to_cutlass_activation(activation)),
            fc2_expert_weights, nullptr, quant_params, num_rows, hidden_size,
            hidden_size, inter_size, num_experts, experts_per_token,
            reinterpret_cast<char*>(workspace), output,
            unpermuted_row_to_permuted_row, parallelism(tp_size, tp_rank),
            false, false, lora_params, false, false, false,
            min_latency_params, false, stream);
        return 0;
    })
}

}  // extern "C"

}  // namespace pie_cuda_driver::kernels::moe
