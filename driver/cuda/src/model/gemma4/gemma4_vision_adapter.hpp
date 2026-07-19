#pragma once

// Host-side bridge from the bound `Gemma4VisionWeights` (DeviceTensor handles)
// to the cuda-only `VisRawWeights` (raw bf16 pointers) the encoder kernels
// consume. This header includes the model headers (toml++ etc.), so it is only
// ever included by host `.cpp` files — never by an nvcc-compiled `.cu`.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "model/gemma4/gemma4.hpp"                 // Gemma4VisionWeights
#include "model/gemma4/gemma4_vision_forward.hpp"  // VisRawWeights, run_gemma4_vision

namespace pie_cuda_driver::model {

// Extract raw device pointers + dims from the bound weights.
VisRawWeights to_vis_raw(const Gemma4VisionWeights& w);

// Convenience overload: build `VisRawWeights` and run the encoder.
void run_gemma4_vision(const Gemma4VisionWeights& w,
                       const __nv_bfloat16* pixel,
                       const float* pos,
                       const int* grp,
                       int n_patch,
                       int out_len,
                       __nv_bfloat16* out_proj,
                       cudaStream_t stream = 0);

}  // namespace pie_cuda_driver::model
