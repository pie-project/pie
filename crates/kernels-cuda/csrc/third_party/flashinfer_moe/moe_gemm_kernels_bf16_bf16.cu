/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// ─── Vendored from FlashInfer ────────────────────────────────────────────────
// Source: v0.6.15, csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/
//         moe_gemm/moe_gemm_kernels_bf16_bf16.cu -- byte-identical to upstream.
//
// Copied verbatim only so the quoted #include below resolves against this
// directory: a quoted include searches the including file's own directory
// first, so the upstream .cu would keep picking up the upstream header and
// ignore our patched copy of moe_gemm_template_dispatch.h entirely.
//
// CMakeLists.txt builds this file instead of the upstream one. Nothing here
// needs to change on a FlashInfer bump unless upstream edits the file.
// ─────────────────────────────────────────────────────────────────────────────

#include "moe_gemm_template_dispatch.h"

namespace tensorrt_llm::kernels::cutlass_kernels {
#ifdef ENABLE_BF16
template class MoeGemmRunner<__nv_bfloat16, __nv_bfloat16, __nv_bfloat16>;
#endif
}  // namespace tensorrt_llm::kernels::cutlass_kernels
