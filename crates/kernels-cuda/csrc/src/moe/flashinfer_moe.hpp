#pragma once

// The CUTLASS fused-MoE INSTANTIATION SEAM, and nothing else.
//
// This header used to declare five host functions -- a capability, a row
// window (two accessors), a workspace query and a dispatch. All five were
// host programs: a workspace calculation, a tuning cache, an autotuner and a
// dispatch, with `std::mutex` and `std::unordered_map` in the middle of them
// and not one line of device code. They are `driver-cuda/src/fire/
// flashinfer_moe.rs` now.
//
// What is left is what Rust cannot do at all. `CutlassMoeFCRunner<
// __nv_bfloat16, __nv_bfloat16>` is a C++ CLASS TEMPLATE whose kernels are
// templates in headers -- `${flashinfer_SOURCE_DIR}/csrc/fused_moe/
// cutlass_backend/cutlass_fused_moe_kernels.cuh`, 4,991 lines, 19
// `__global__`, CPM-fetched at configure time. Nothing in Rust, cudarc or
// NVRTC can name that type, so an `extern "C"` seam over it is the minimum
// C++ this file can be, and every function below exists because it names a
// C++ type or enumerator that has no spelling on the other side:
//
//   create        constructs the runner                  (a class template)
//   tactics       copies `CutlassGemmConfig` fields out  (a struct we cannot see)
//   set_tactic    indexes `getTactics()`'s own vector    (a std::vector)
//   workspace     `getWorkspaceSize`                     (a method that THROWS)
//   run           `runMoe`                               (a method that THROWS)
//
// Not one of them decides anything. Which tactic, when to tune, what to
// cache, which rows are eligible, whether the call is declined -- all of it
// is Rust. The seam reports FACTS and performs the two calls it is asked to.
//
// # Nothing here may throw
//
// An exception crossing the C ABI is undefined behaviour and in practice
// reaches SIGABRT with no message; FlashInfer's own dispatch macros end in
// `throw std::invalid_argument(...)`. So every entry point below is
// `noexcept` in effect: it catches, writes `what()` into the caller's buffer
// and answers a status. That is the same contract `abi::emit_c_shim`'s
// `PIE_K_GUARD` gives the generated entries, moved to the one seam that is
// left.

#include <cstddef>
#include <cstdint>

#include <cuda_runtime.h>

namespace pie_cuda_driver::kernels::moe {

// Which gated/ungated epilogue the fused MoE runs. Each value costs one
// more CUTLASS grouped-GEMM instantiation, so the set is declared in
// kernels.def and kept to what a shipped arch actually reaches.
//
// Note on Swiglu: the runner reads the gate half from the *second* half of the
// fc1 output and the linear half from the first, i.e. silu(w[I:]) * w[:I] --
// the opposite of pie's chunked_swiglu. fc1 weights must be stacked as
// [up; gate], not pie's usual [gate; up]. Geglu has the same convention; only
// the scalar function on the gate half differs.
//
// FIVE MODEL CONTRACTS CITE THIS PARAGRAPH BY FILE NAME
// (`model/src/{qwen_3_5,glm_5,kimi_k3,gemma_4}/contract.rs`), which is why it
// stays here in the header rather than moving to the Rust with the rest of
// the host program. `driver_cuda::bind::abi::MoeActivation` is the mirror,
// and `driver-cuda/tests/launch_abi.rs` static_asserts the three
// discriminants below against it.
enum class MoeActivation {
    Relu2,    // nemotron_h
    Swiglu,   // qwen3.5 / qwen3.6 MoE, glm5 / kimi / deepseek_v4
    Geglu,    // gemma-4 26B-A4B routed experts (GELU-tanh gate)
};

// ---------------------------------------------------------------------------
// The seam
// ---------------------------------------------------------------------------

extern "C" {

// Which of the runner's two grouped GEMMs a tactic query is about.
// `ck::MoeGemmId::GEMM_1` / `GEMM_2`, as the plain integers the enumerators
// carry.
enum {
    PIE_MOE_GEMM_1 = 1,
    PIE_MOE_GEMM_2 = 2,
};

// `ce::CutlassGemmConfig::EpilogueFusionType`, RENUMBERED onto values this
// header owns.
//
// The seam maps rather than casts, so a reordered upstream enumerator is a
// C++ compile error here instead of a silent change of meaning on the Rust
// side -- which matters more than usual, because GEMM2's fusion type is a
// NUMERICS decision (see `fire/flashinfer_moe.rs`: FINALIZE accumulates the
// topk sum in bf16, in CTA completion order) and the tuner is forbidden to
// cross it.
enum {
    PIE_MOE_FUSION_NONE = 0,
    PIE_MOE_FUSION_FINALIZE = 1,
    // An upstream enumerator this header has not been taught. The Rust
    // prints it as "unknown" and never selects across it, which is what the
    // C++'s own `fusion_name` did with its `return "unknown"` tail.
    PIE_MOE_FUSION_UNKNOWN = 2,
};

// One `ce::CutlassGemmConfig`, flattened to the fields the host program
// reads. Thirteen `int`s and no pointers, so it needs no layout proof beyond
// field order.
//
// `tile` is the ONE field `sm_version` decides: the config carries four tile
// enums and only one of them is live. The seam picks it (printing the wrong
// one shows a constant `heuristic` for every tactic), because picking it
// needs the four field names.
//
// `occupancy` is `runner.queryOccupancyForConfig(cfg)`, filled at query time.
// The C++ re-asked it inside every sweep; the query takes a config and
// nothing else, so the answer cannot depend on the problem and asking once is
// the same answer for less work.
struct PieMoeTactic {
    int fusion;                  // PIE_MOE_FUSION_*
    int is_tma_warp_specialized;
    int swap_ab;
    int sm_version;
    int tile;                    // the live one of the four tile_config_sm*
    int mainloop_schedule;
    int epilogue_schedule;
    int cluster_shape;
    int dynamic_cluster_shape;
    int fallback_cluster_shape;
    int split_k_factor;
    int stages;
    int occupancy;
};

/// Construct the bf16 runner. Null on failure, with `what()` in `err`.
///
/// THERE IS NO DESTROY, and the consumer set for one is empty by
/// construction: the C++ held its runners in a function-local
/// `static std::array<RunnerState, 16>` whose destructors ran at process
/// exit, against a CUDA context that may already be torn down. The Rust holds
/// them in a `OnceLock`, which never drops. Neither ever freed one while the
/// process could still use it, so the seam does not offer a way to.
void* pie_moe_cutlass_create(char* err, std::size_t err_cap);

/// Copy the runner's tactic list for `gemm_id` into `out`.
///
/// Answers the FULL count -- which may exceed `cap` -- or `-1` with `err` set.
/// `out` may be null when `cap` is 0, which is how the count is asked for on
/// its own.
int pie_moe_cutlass_tactics(void* runner, int gemm_id, PieMoeTactic* out,
                            int cap, char* err, std::size_t err_cap);

/// `runner.setTactic(gemm1_tactics[gemm1], gemm2_tactics[gemm2])`.
///
/// Indices into the same lists `pie_moe_cutlass_tactics` reported, in the same
/// order. 0 on success, non-zero with `err` set otherwise (including an index
/// out of range, which is the one failure the seam decides for itself).
int pie_moe_cutlass_set_tactic(void* runner, int gemm1, int gemm2, char* err,
                               std::size_t err_cap);

/// `runner.getWorkspaceSize(...)` -- and the arch probe, because it is the
/// call that THROWS when no TMA warp-specialized config has a compiled
/// launcher for this SM.
///
/// 0 on success with the size in `*out`; non-zero with `err` set otherwise.
/// The caller decides what a throw means -- it means "report zero and let the
/// caller take the unfused expert path", and that decision is in the Rust.
int pie_moe_cutlass_workspace_size(void* runner, MoeActivation activation,
                                   int num_rows, int hidden_size,
                                   int inter_size, int num_experts,
                                   int experts_per_token, int tp_size,
                                   int tp_rank, std::size_t* out, char* err,
                                   std::size_t err_cap);

/// `runner.runMoe(...)` on `stream`, with the currently installed tactic.
///
/// Every pointer is the caller's device address; the seam validates none of
/// them (the Rust already refused a null before reaching here). 0 on success,
/// non-zero with `err` set otherwise.
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
                        std::size_t err_cap);

}  // extern "C"

}  // namespace pie_cuda_driver::kernels::moe
