#pragma once

// Host-side bridge from the bound `Gemma4AudioWeights` (DeviceTensor handles)
// to the cuda-only `AudioRawWeights` (raw bf16 pointers) the encoder kernels
// consume. This header includes the model/tensor headers (toml++ etc.), so it
// is only ever included by host `.cpp` files — never by an nvcc-compiled `.cu`.
//
// Mirrors gemma4_vision_adapter.hpp.
//
// EXPECTED upstream types. These will live in the SHARED `model/gemma4.hpp`
// (which this task must NOT edit — a teammate wires `bind_gemma4_audio` there).
// To keep this adapter self-compiling in the meantime, the expected structs are
// declared HERE under `#ifndef PIE_HAS_GEMMA4_AUDIO_WEIGHTS`. When the real
// types land in gemma4.hpp, that header should `#define
// PIE_HAS_GEMMA4_AUDIO_WEIGHTS` (or this block be deleted and the include
// swapped to `model/gemma4.hpp`) so there is a single definition. The struct
// below is the contract `bind_gemma4_audio` must satisfy; adjust field names to
// match the final struct if they diverge.
//
// Bound from the `model.audio_tower.` + `model.embed_audio.` checkpoint
// prefixes (see the integration checklist), all by the real tensor names
// (verified present, missing=0, by scripts/gemma4_audio_parity_ref.py):
//   subsample_conv_projection.layer{0,1}.{conv.weight, norm.weight}
//   subsample_conv_projection.input_proj_linear.weight
//   layers.{i}.feed_forward{1,2}.{pre,post}_layer_norm.weight
//   layers.{i}.feed_forward{1,2}.ffw_layer_{1,2}  (clipped linear)
//   layers.{i}.{norm_pre_attn, norm_post_attn, norm_out}.weight
//   layers.{i}.self_attn.{q,k,v,post}_proj  (clipped; "post" is the out proj)
//   layers.{i}.self_attn.relative_k_proj.weight   (NOT clipped)
//   layers.{i}.self_attn.per_dim_scale            [head_dim]
//   layers.{i}.lconv1d.{pre_layer_norm, conv_norm}.weight
//   layers.{i}.lconv1d.{linear_start, linear_end} (clipped)
//   layers.{i}.lconv1d.depthwise_conv1d.weight    [hidden, 1, conv_kernel]
//   output_proj.{weight, bias}
//   model.embed_audio.embedding_projection.weight

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include "model/gemma4/gemma4.hpp"                 // Gemma4AudioWeights (defines PIE_HAS_GEMMA4_AUDIO_WEIGHTS)
#include "model/gemma4/gemma4_audio_forward.hpp"  // AudioRawWeights, run_gemma4_audio
#include "tensor.hpp"                       // DeviceTensor

namespace pie_cuda_driver::model {

#ifndef PIE_HAS_GEMMA4_AUDIO_WEIGHTS
// ── Expected bound-weight contract (teammate moves this to gemma4.hpp) ───────

// One clipped-linear: `<name>.linear.weight` + scalar clip ranges (the audio
// tower sets `use_clipped_linears=True`). The clip-range tensors are null when
// the checkpoint is not quantized — identical layout to Gemma4ClippedLinear.
struct Gemma4AudioClippedLinear {
    const DeviceTensor* weight     = nullptr;  // .linear.weight  [out, in]
    const DeviceTensor* input_min  = nullptr;  // scalar
    const DeviceTensor* input_max  = nullptr;  // scalar
    const DeviceTensor* output_min = nullptr;  // scalar
    const DeviceTensor* output_max = nullptr;  // scalar
};

// Macaron feed-forward (`Gemma4AudioFeedForward`).
struct Gemma4AudioFfnWeights {
    const DeviceTensor* pre_layer_norm  = nullptr;  // [hidden]
    const DeviceTensor* post_layer_norm = nullptr;  // [hidden]
    Gemma4AudioClippedLinear ffw_layer_1;            // [4*hidden, hidden]
    Gemma4AudioClippedLinear ffw_layer_2;            // [hidden, 4*hidden]
};

// One Conformer block (`Gemma4AudioLayer`).
struct Gemma4AudioLayerWeights {
    Gemma4AudioFfnWeights feed_forward1, feed_forward2;

    const DeviceTensor* norm_pre_attn  = nullptr;   // [hidden]
    const DeviceTensor* norm_post_attn = nullptr;   // [hidden]
    const DeviceTensor* norm_out       = nullptr;   // [hidden]

    // Chunked-local self-attention. `post` is the attention output projection.
    Gemma4AudioClippedLinear q_proj, k_proj, v_proj, post;
    const DeviceTensor* relative_k_proj = nullptr;  // [H*head_dim, hidden] (NOT clipped)
    const DeviceTensor* per_dim_scale   = nullptr;  // [head_dim]

    // Light depthwise-conv module (`lconv1d`).
    const DeviceTensor* lconv_pre_layer_norm = nullptr;  // [hidden]
    const DeviceTensor* lconv_conv_norm      = nullptr;  // [hidden]
    Gemma4AudioClippedLinear lconv_linear_start;          // [2*hidden, hidden] → GLU
    Gemma4AudioClippedLinear lconv_linear_end;            // [hidden, hidden]
    const DeviceTensor* lconv_depthwise_conv = nullptr;  // [hidden, 1, conv_kernel]
};

// Minimal config the adapter reads (subset of the future GemmaAudioConfig the
// teammate parses into HfConfig.gemma_audio). Filled by `bind_gemma4_audio`.
struct Gemma4AudioConfigLite {
    int hidden_size = 1024;
    int num_attention_heads = 8;
    int num_hidden_layers = 12;
    int conv_kernel_size = 5;
    int subsampling_conv_channels0 = 128;
    int subsampling_conv_channels1 = 32;
    int output_proj_dims = 1536;
    int attention_chunk_size = 12;
    int attention_context_left = 13;
    int attention_context_right = 0;
    int feature_size = 128;          // mel bins
    float attention_logit_cap = 50.0f;
    float residual_weight = 0.5f;
    float rms_norm_eps = 1e-6f;
};

struct Gemma4AudioWeights {
    // SSCP subsampling conv stack.
    const DeviceTensor* sscp_layer0_conv = nullptr;  // [c0, 1, 3, 3]
    const DeviceTensor* sscp_layer0_norm = nullptr;  // [c0]
    const DeviceTensor* sscp_layer1_conv = nullptr;  // [c1, c0, 3, 3]
    const DeviceTensor* sscp_layer1_norm = nullptr;  // [c1]
    const DeviceTensor* sscp_input_proj  = nullptr;  // [hidden, (c0/4)*c1]

    std::vector<Gemma4AudioLayerWeights> layers;

    const DeviceTensor* output_proj_weight = nullptr;  // [out_proj_dims, hidden]
    const DeviceTensor* output_proj_bias   = nullptr;  // [out_proj_dims]

    // Shared embedder (`embed_audio`): parameterless RMSNorm → projection.
    const DeviceTensor* embed_audio_projection = nullptr;  // [text_hidden, out_proj_dims]

    Gemma4AudioConfigLite config;
};
#endif  // PIE_HAS_GEMMA4_AUDIO_WEIGHTS

// Extract raw device pointers + dims from the bound weights.
AudioRawWeights to_audio_raw(const Gemma4AudioWeights& w);

// Convenience overload: build `AudioRawWeights` and run the encoder.
void run_gemma4_audio(const Gemma4AudioWeights& w,
                      const float* features,
                      int n_frames,
                      int n_mel,
                      int out_len,
                      __nv_bfloat16* out_proj,
                      cudaStream_t stream = 0);

}  // namespace pie_cuda_driver::model
