#include "moe_gemm_tma_ws_instantiate.h"

namespace tensorrt_llm::kernels::cutlass_kernels_oss {

PIE_INSTANTIATE_TMA_WS_MOE_GEMM(Sm100, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, PtrArrayTmaWarpSpecialized, EpilogueOpDefault, FINALIZE, 128, 256, 64, 2, 1, 1, false, true, false, false);
PIE_INSTANTIATE_TMA_WS_MOE_GEMM(Sm100, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, PtrArrayTmaWarpSpecialized, EpilogueOpDefault, FINALIZE, 128, 256, 64, 2, 1, 1, false, true, false, true);
PIE_INSTANTIATE_TMA_WS_MOE_GEMM(Sm100, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, PtrArrayTmaWarpSpecialized, EpilogueOpDefault, FINALIZE, 128, 32, 64, 1, 1, 1, false, false, false, false);
PIE_INSTANTIATE_TMA_WS_MOE_GEMM(Sm100, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, PtrArrayTmaWarpSpecialized, EpilogueOpDefault, FINALIZE, 128, 32, 64, 1, 1, 1, false, false, false, true);
PIE_INSTANTIATE_TMA_WS_MOE_GEMM(Sm100, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, PtrArrayTmaWarpSpecialized, EpilogueOpDefault, FINALIZE, 128, 32, 64, 1, 1, 1, false, true, false, false);
PIE_INSTANTIATE_TMA_WS_MOE_GEMM(Sm100, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, PtrArrayTmaWarpSpecialized, EpilogueOpDefault, FINALIZE, 128, 32, 64, 1, 1, 1, false, true, false, true);
PIE_INSTANTIATE_TMA_WS_MOE_GEMM(Sm100, __nv_bfloat16, __nv_bfloat16, __nv_bfloat16, PtrArrayTmaWarpSpecialized, EpilogueOpDefault, FINALIZE, 128, 64, 64, 1, 1, 1, false, false, false, false);

}  // namespace tensorrt_llm::kernels::cutlass_kernels_oss
