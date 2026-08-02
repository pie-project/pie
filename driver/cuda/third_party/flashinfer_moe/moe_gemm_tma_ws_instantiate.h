// Wrapper around FlashInfer's INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM used by
// third_party/flashinfer_generated/gemm_grouped/{90,100}/.
//
// Why this exists instead of calling the upstream macro directly:
//
// The compile-time dispatch in moe_gemm_template_dispatch_tma_ws.h is
// exhaustive -- it names a launcher for every (tile shape x cluster shape x
// epilogue schedule x fusion x swap_ab) combination the runtime *could* pick,
// so every one of them has to resolve at link time. Upstream's generator only
// emits the subset NVIDIA ships (for bf16: CTA_M=128, N in {128,192,256}),
// which does not cover pie's reduced dispatch, and the combinations it leaves
// out are not all compilable: with the TMA epilogue schedule and no FINALIZE
// fusion the launcher feeds `alpha_scale_ptr_array` (a `float const**`) into a
// plain LinearCombination whose Arguments start with a scalar `float alpha`,
// which is a hard type error.
//
// So this header routes each instantiation to either the real CUTLASS kernel or
// a definition that throws. getTmaWarpSpecializedConfigs() in
// moe_gemm_template_dispatch.h drops exactly the same (schedule, fusion) pairs
// from the runtime candidate list, so the throwing bodies are unreachable --
// they exist only to satisfy the linker. Keep the two lists in sync.
#pragma once

#include "launchers/moe_gemm_tma_ws_launcher.inl"

// 1 = a real CUTLASS instantiation, 0 = a throwing stub. Named by pasting the
// epilogue schedule and the fusion tag, so the selection is a pure token
// operation the preprocessor can do.
#define PIE_TMA_WS_OK_PtrArrayNoSmemWarpSpecialized_NONE 1
#define PIE_TMA_WS_OK_PtrArrayNoSmemWarpSpecialized_FINALIZE 1
#define PIE_TMA_WS_OK_PtrArrayTmaWarpSpecialized_NONE 0
#define PIE_TMA_WS_OK_PtrArrayTmaWarpSpecialized_FINALIZE 1

// Hopper (Sm90) does not parameterise the launcher on an epilogue schedule --
// the template takes `void` in that position. It splits the same way Blackwell's
// TMA schedule does: NONE hits the documented type error above (the launcher
// feeds `alpha_scale_ptr_array`, a `float const**`, into a LinearCombination
// whose Arguments open with a scalar `float alpha`), FINALIZE compiles.
// Verified by building both: NONE fails at moe_gemm_tma_ws_launcher.inl:65.
#define PIE_TMA_WS_OK_void_NONE 0
#define PIE_TMA_WS_OK_void_FINALIZE 1

#define PIE_TMA_WS_CAT_(a, b) a##b
#define PIE_TMA_WS_CAT(a, b) PIE_TMA_WS_CAT_(a, b)
#define PIE_TMA_WS_SUPPORTED(EpilogueSchedule_, FUSION_) \
  PIE_TMA_WS_CAT(PIE_TMA_WS_CAT(PIE_TMA_WS_OK_, EpilogueSchedule_), PIE_TMA_WS_CAT(_, FUSION_))

#define PIE_INSTANTIATE_TMA_WS_MOE_GEMM_1(...) \
  INSTANTIATE_TMA_WARP_SPECIALIZED_MOE_GEMM(__VA_ARGS__)

#define PIE_INSTANTIATE_TMA_WS_MOE_GEMM_0(                                                    \
    ArchTag_, DataType_, WeightType_, OutputType_, EpilogueSchedule_, EpilogueTag_, FUSION_,  \
    CTA_M_, CTA_N_, CTA_K_, CGA_M_, CGA_N_, CGA_K_, MXFPX_, DYNAMIC_CGA_, BIAS_, SWAP_AB_)    \
  template <>                                                                                 \
  void tma_warp_specialized_generic_moe_gemm_kernelLauncher<                                  \
      cutlass::arch::ArchTag_, DataType_, WeightType_, OutputType_, EpilogueSchedule_,        \
      tensorrt_llm::cutlass_extensions::EpilogueTag_, EpilogueFusion::FUSION_,                \
      cute::Shape<cute::Int<CTA_M_>, cute::Int<CTA_N_>, cute::Int<CTA_K_>>,                   \
      cute::Shape<cute::Int<CGA_M_>, cute::Int<CGA_N_>, cute::Int<CGA_K_>>, MXFPX_,           \
      DYNAMIC_CGA_, BIAS_, SWAP_AB_>(                                                         \
      TmaWarpSpecializedGroupedGemmInput, int, int, cudaStream_t, int*,                       \
      size_t* workspace_size, cute::Shape<int32_t, int32_t, cute::_1>,                        \
      cute::Shape<int32_t, int32_t, cute::_1>) {                                              \
    /* Reached only through the workspace-sizing sweep, which visits configs it never */      \
    /* selects; contributing 0 keeps the max over candidates correct. */                      \
    if (workspace_size) {                                                                     \
      *workspace_size = 0;                                                                    \
      return;                                                                                 \
    }                                                                                         \
    TLLM_THROW(                                                                               \
        "MoE grouped GEMM config " #EpilogueSchedule_ "/" #FUSION_                            \
        " has no CUTLASS instantiation; it should have been filtered out of the candidate "   \
        "list by getTmaWarpSpecializedConfigs().");                                           \
  }

#define PIE_INSTANTIATE_TMA_WS_MOE_GEMM(ArchTag_, DataType_, WeightType_, OutputType_,          \
                                        EpilogueSchedule_, EpilogueTag_, FUSION_, CTA_M_,       \
                                        CTA_N_, CTA_K_, CGA_M_, CGA_N_, CGA_K_, MXFPX_,         \
                                        DYNAMIC_CGA_, BIAS_, SWAP_AB_)                          \
  PIE_TMA_WS_CAT(PIE_INSTANTIATE_TMA_WS_MOE_GEMM_,                                              \
                 PIE_TMA_WS_SUPPORTED(EpilogueSchedule_, FUSION_))                              \
  (ArchTag_, DataType_, WeightType_, OutputType_, EpilogueSchedule_, EpilogueTag_, FUSION_,     \
   CTA_M_, CTA_N_, CTA_K_, CGA_M_, CGA_N_, CGA_K_, MXFPX_, DYNAMIC_CGA_, BIAS_, SWAP_AB_)
