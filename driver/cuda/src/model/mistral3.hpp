#pragma once

// Mistral-Small-3.1 (FP8 native checkpoint). The model schema is
// otherwise identical to the standard Llama-like layer, so we delegate
// the forward to `llama_like_forward_paged` once the FP8 weights are
// dequantized to bf16.
//
// Strategy: dequantize-on-load. Each FP8 projection ships with a
// scalar `weight_scale_inv`; we materialize a fresh bf16 tensor via
// `launch_dequant_fp8_e4m3_to_bf16` (a per-tensor-scaled cast), then
// register it in the engine under the canonical "self_attn.q_proj.weight"
// name `bind_llama_like` expects. Cost: 2× the FP8 footprint while we
// hold both copies, then we drop the FP8 source.
//
// For 24B+ checkpoints the right move is fused FP8 GEMM via cuBLAS's
// `cublasGemmEx(CUDA_R_8F_E4M3, …)` — that path also accepts a scalar
// scale and skips the dequant entirely. The dequant approach keeps
// the existing GEMM path uniform; large checkpoints can swap to fused
// FP8 by adding a `gemm_act_x_wt_fp8` overload alongside the bf16 one.

#include "model/loaded_model.hpp"
#include "model/qwen3.hpp"

namespace pie_cuda_driver::model {

// Walks the layer-projection tensors in `engine`, dequantizes them in
// place (registering bf16 views under the un-suffixed names), and then
// returns a `Qwen3Weights` ready for `llama_like_forward_paged`. The
// FP8 source tensors stay in the engine but are unreferenced.
Qwen3Weights bind_mistral3(LoadedModel& engine);

}  // namespace pie_cuda_driver::model
