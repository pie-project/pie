#pragma once

// Gemma-4 audio encoder forward (bf16). Reproduces transformers
// `Gemma4AudioModel` (USM/Conformer) + the shared `Gemma4MultimodalEmbedder`.
// First-draft scaffold; mirrors gemma4_vision_forward.hpp. Parity TODOs are
// marked inline in the .cu (chunked-attention masking, conv-module GLU+norm,
// subsampling stride math) and the harness is scripts/gemma4_audio_parity_ref.py
// (RUN: 751 tensors, missing=0; 199 mel frames → 50 audio tokens → [50,2560]).
//
// CUDA-only header (no model/loader headers) so the `.cu` doesn't pull in the
// toml++-heavy config headers that nvcc cannot parse. The host call site builds
// `AudioRawWeights` from the bound `Gemma4AudioWeights` via `DeviceTensor::data()`
// (see gemma4_audio_adapter.{hpp,cpp}).
//
// Architecture (verified from `gemma-4-E4B` config + transformers 5.9
// `modeling_gemma4.py`):
//   input  : log-mel features [n_frames, 128]  (inferlet computes these; option B)
//   SSCP   : 2× Conv2d(k3,s2,p1, +LayerNorm-over-ch +ReLU) → 4× time downsample,
//            freq 128→32; reshape [T', 32*32] → input_proj_linear → hidden 1024
//   12×    : Conformer block, residual_weight 0.5:
//            ½·FFN  (RMSNorm → lin 1024→4096 → silu → lin 4096→1024, ×0.5 + res)
//            MHSA   (chunked LOCAL attention, chunk 12 / left 13 / right 0,
//                    relative-position bias, logit cap 50, per_dim_scale softplus)
//            LConv  (RMSNorm → lin 1024→2048 → GLU → depthwise causal conv k5 →
//                    clamp → RMSNorm → silu → lin 1024→1024 + res)
//            ½·FFN  (as above)
//            RMSNorm (norm_out)
//   output : output_proj (1024→1536, +bias)
//   embedder: parameterless RMSNorm(1536) → embedding_projection (1536→2560)
//
// All `*_proj` / ffw / lconv linears are "clipped linears": clamp(x,imin,imax) →
// matmul → clamp(y,omin,omax) — same pattern as the vision tower
// (`use_clipped_linears: true`). Reuse the vision clipped-linear kernel style.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <vector>

namespace pie_cuda_driver::model {

// Raw device pointers for one clipped-linear: weight + optional bf16 scalar
// clip ranges (null = no clamp on that side). Same shape as VisClipRaw.
struct AudioClipRaw {
    const __nv_bfloat16* w = nullptr;
    const __nv_bfloat16* imin = nullptr;
    const __nv_bfloat16* imax = nullptr;
    const __nv_bfloat16* omin = nullptr;
    const __nv_bfloat16* omax = nullptr;
};

// One Conformer block (`Gemma4AudioLayer`). All RMSNorms carry a learnable
// scale (`with_scale=True`). The macaron FFN appears twice (feed_forward1 /
// feed_forward2) with the same internal layout.
struct AudioFfnRaw {
    const __nv_bfloat16* pre_ln = nullptr;   // pre_layer_norm   [hidden]
    const __nv_bfloat16* post_ln = nullptr;  // post_layer_norm  [hidden]
    AudioClipRaw fc1;                         // ffw_layer_1  [4*hidden, hidden]
    AudioClipRaw fc2;                         // ffw_layer_2  [hidden, 4*hidden]
};

struct AudioLayerRaw {
    // Macaron FFNs (each contributes residual_weight · FFN(x)).
    AudioFfnRaw ff1, ff2;

    // Chunked-local self-attention with relative-position bias + logit cap.
    const __nv_bfloat16* norm_pre_attn = nullptr;   // [hidden]
    const __nv_bfloat16* norm_post_attn = nullptr;  // [hidden]
    AudioClipRaw q, k, v, post;                      // post == attn out proj
    const __nv_bfloat16* relative_k = nullptr;       // relative_k_proj.weight [H*hd, hidden] (NOT clipped)
    const __nv_bfloat16* per_dim_scale = nullptr;    // [head_dim] (softplus-gated)

    // Light depthwise-conv module (`lconv1d`).
    const __nv_bfloat16* lconv_pre_ln = nullptr;     // pre_layer_norm  [hidden]
    const __nv_bfloat16* lconv_conv_norm = nullptr;  // conv_norm       [hidden]
    AudioClipRaw lconv_start;                         // linear_start [2*hidden, hidden] → GLU
    AudioClipRaw lconv_end;                           // linear_end   [hidden, hidden]
    const __nv_bfloat16* depthwise_conv = nullptr;   // [hidden, 1, conv_kernel] (causal, left-pad k-1)

    // Final block RMSNorm.
    const __nv_bfloat16* norm_out = nullptr;         // [hidden]
};

struct AudioRawWeights {
    // SSCP subsampling conv stack (`subsample_conv_projection`).
    const __nv_bfloat16* sscp0_conv = nullptr;       // layer0.conv.weight  [c0, 1, 3, 3]
    const __nv_bfloat16* sscp0_norm = nullptr;       // layer0.norm.weight  [c0]  (LayerNorm, no bias)
    const __nv_bfloat16* sscp1_conv = nullptr;       // layer1.conv.weight  [c1, c0, 3, 3]
    const __nv_bfloat16* sscp1_norm = nullptr;       // layer1.norm.weight  [c1]
    const __nv_bfloat16* sscp_input_proj = nullptr;  // input_proj_linear.weight [hidden, (c0/4)*c1]

    std::vector<AudioLayerRaw> layers;

    // output_proj (Linear with bias) → output_proj_dims.
    const __nv_bfloat16* output_proj_w = nullptr;    // [out_proj_dims, hidden]
    const __nv_bfloat16* output_proj_b = nullptr;    // [out_proj_dims]

    // Shared embedder (`embed_audio`): parameterless RMSNorm → projection.
    const __nv_bfloat16* embed_proj = nullptr;       // embedding_projection.weight [text_hidden, out_proj_dims]

    int hidden = 1024;
    int heads = 8;              // head_dim = hidden / heads = 128
    int conv_kernel = 5;        // depthwise causal conv kernel
    int n_mel = 128;            // mel bins (= SSCP input freq)
    int sscp_ch0 = 128;
    int sscp_ch1 = 32;
    int out_proj_dims = 1536;
    int text_hidden = 2560;
    int chunk_size = 12;        // attention_chunk_size
    int context_left = 13;      // attention_context_left  (max_past_horizon = left-1)
    int context_right = 0;      // attention_context_right
    float logit_cap = 50.0f;    // attention_logit_cap
    float residual_weight = 0.5f;
    float eps = 1e-6f;
    // q_scale = (head_dim^-0.5)/log(2); k_scale = log(1+e)/log(2). Derived at
    // run time from `heads`/`hidden` to match transformers Gemma4AudioAttention.
};

// Per-forward audio inputs for the scatter. Host pointers into the request view
// (option-B log-mel path). `features_h` is the f32 log-mel of all clips
// concatenated; `feature_byte_indptr_h[i..i+1]` is clip i's byte range
// (n_frames_i * 128 * 4); `n_mel` is the bins per frame (128); `anchor_rows_h[i]`
// is the batch row where clip i's audio soft tokens live.
struct Gemma4AudioInputs {
    const AudioRawWeights* weights = nullptr;
    const float* features_h = nullptr;
    const std::uint32_t* feature_byte_indptr_h = nullptr;  // [num_clips + 1]
    const std::uint32_t* anchor_rows_h = nullptr;          // [num_clips]
    int n_mel = 128;
    int num_clips = 0;
};

// Encode each audio clip (audio tower → output_proj → embedder) and overwrite
// the soft-token rows of `hidden` (bf16 `[n_rows, text_hidden]`) at each clip's
// anchor row. No-op if `in.weights == nullptr` or `in.num_clips == 0`.
void scatter_gemma4_audio(const Gemma4AudioInputs& in,
                          __nv_bfloat16* hidden,
                          int n_rows,
                          int text_hidden,
                          cudaStream_t stream = 0);

void encode_gemma4_audio(const Gemma4AudioInputs& in,
                         std::uint16_t* output_rows_h,
                         std::size_t output_bytes,
                         std::uint32_t* output_row_indptr_h,
                         cudaStream_t stream = 0);

// Encode one clip's log-mel features → soft-token embeddings in the text hidden
// space.
//   features : f32 [n_frames, n_mel]      (log-mel, padding already stripped)
//   out_proj : bf16 [out_len, text_hidden] (written; out_len = n_audio_tokens)
// `out_len` must equal the post-subsampling frame count (see
// `gemma4_audio_subsampled_len`). Allocates internal scratch (first-cut; a
// workspace arena is a follow-up).
void run_gemma4_audio(const AudioRawWeights& w,
                      const float* features,
                      int n_frames,
                      int n_mel,
                      int out_len,
                      __nv_bfloat16* out_proj,
                      cudaStream_t stream = 0);

// Per-stage checkpoint hook (parity debugging only). When set, `run_gemma4_audio`
// invokes `fn(name, dev, numel, user)` after each named stage (`sscp_out`,
// `layer0`..`layer11`, `encoder_out`, `projected`) with the device bf16 buffer.
// Mirrors the mimi-decoder checkpoint mechanism. No-op when unset.
typedef void (*Gemma4AudioCkptFn)(const char* name, const __nv_bfloat16* dev,
                                  long numel, void* user);
void set_gemma4_audio_ckpt(Gemma4AudioCkptFn fn, void* user);

// Number of audio soft tokens for `n_frames` mel frames: two Conv2d(k3,s2,p1)
// applied along time. floor((n-1)/2)+1 twice. Host-callable (constexpr-ish).
inline int gemma4_audio_subsampled_len(int n_frames) {
    auto conv = [](int n) { return (n + 2 * 1 - 3) / 2 + 1; };
    return conv(conv(n_frames));
}

}  // namespace pie_cuda_driver::model
