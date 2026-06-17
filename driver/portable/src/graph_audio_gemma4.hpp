#pragma once
// Gemma-4 audio tower (USM / Conformer encoder) for the portable ggml driver.
// Mirrors the CUDA gemma4_audio_forward. Architecture (google/gemma-4-E2B/E4B,
// audio_config model_type "gemma4_audio"):
//   hidden 1024, 12 Conformer blocks, 8 heads (head_dim 128), conv_kernel 5,
//   subsampling_conv_channels [128, 32], output_proj_dims 1536, feature_size 128,
//   attention_chunk_size 12, attention_context_left 13 (max_past 12), right 0,
//   attention_logit_cap 50, residual_weight 0.5, rms_norm_eps 1e-6,
//   use_clipped_linears. Projected to text hidden via embed_audio.
//
// Pipeline (per clip):
//   SSCP  : log-mel [n_frames, 128] -> 2x Conv2d(k3,s2,p1)+LayerNorm-over-ch+ReLU
//           (4x time downsample, freq 128->32) -> flatten [T2, 32*32]
//           -> input_proj_linear -> hidden 1024.  N = T2 = subsampled frames.
//   12x   : Conformer block (residual_weight 0.5):
//           1/2 FFN  (RMSNorm -> lin 1024->4096 -> silu -> lin 4096->1024, x0.5 +res)
//           MHSA     (chunked LOCAL causal window past=12, relative-position bias,
//                     logit cap 50, per_dim_scale softplus)
//           LConv    (RMSNorm -> lin 1024->2048 -> GLU -> depthwise causal conv k5
//                     -> conv_norm RMSNorm -> silu -> lin 1024->1024 + res)
//           1/2 FFN  (as above)
//           RMSNorm  (norm_out)
//   out   : output_proj (1024->1536, +bias)
//   embed : parameterless RMSNorm(1536) -> embedding_projection (1536->text_hidden)
//
// All *_proj / ffw / lconv linears are clipped linears (clamp(x,imin,imax) ->
// matmul -> clamp(y,omin,omax)); bounds are per-projection scalars, like vision.
//
// Chunked-local attention realized as a fixed-width causal sliding window: query
// t attends keys j with 0 <= t-j <= max_past-1 (= 11; distance max_past=12 is
// excluded, matching CUDA k_local_attn `lo=i-(P-2)`). We express it as scores
// over Dwin = max_past relative offsets with a softmax over that offset axis,
// which avoids any 2D gather. matrix_bd[t,j] uses the relative-position embedding
// for distance d=(t-j): pe row (P-1)-d after relative_k_proj, contributing
// q_t . relk[(P-1)-d]. q is pre-scaled by q_scale*softplus(per_dim_scale), k by
// k_scale (q_scale=(hd^-0.5)/ln2, k_scale=ln(1+e)/ln2). Logit soft-cap
// cap*tanh(s/cap) before softmax.

#include <vector>

#include <ggml.h>

#include "hf_config.hpp"

namespace pie_portable_driver {

// Clipped linear: clamp(x,[imin,imax]) -> W x -> clamp([omin,omax]). Bounds are
// scalars read from the checkpoint at load time. has_* false => no clamp.
struct G4AudClipLinear {
    ggml_tensor* w = nullptr;   // [in, out] ggml
    float imin = 0, imax = 0;   bool has_in  = false;
    float omin = 0, omax = 0;   bool has_out = false;
};

// Macaron FFN (feed_forward1 / feed_forward2), identical internal layout.
struct G4AudFfn {
    ggml_tensor* pre_ln  = nullptr;  // pre_layer_norm.weight  [hidden]
    ggml_tensor* post_ln = nullptr;  // post_layer_norm.weight [hidden]
    G4AudClipLinear fc1;             // ffw_layer_1 [hidden, 4*hidden]
    G4AudClipLinear fc2;             // ffw_layer_2 [4*hidden, hidden]
};

struct G4AudLayer {
    G4AudFfn ff1, ff2;
    // Chunked-local self-attention.
    ggml_tensor* norm_pre_attn  = nullptr;  // [hidden]
    ggml_tensor* norm_post_attn = nullptr;  // [hidden]
    G4AudClipLinear q, k, v, post;           // post = attn out proj
    ggml_tensor* relative_k    = nullptr;    // relative_k_proj.weight [hidden, hidden] (NOT clipped)
    ggml_tensor* per_dim_scale = nullptr;    // [head_dim]
    // Light depthwise-conv module.
    ggml_tensor* lconv_pre_ln   = nullptr;   // pre_layer_norm  [hidden]
    ggml_tensor* lconv_conv_norm = nullptr;  // conv_norm       [hidden]
    G4AudClipLinear lconv_start, lconv_end;  // start [hidden,2*hidden] -> GLU; end [hidden,hidden]
    ggml_tensor* depthwise_conv = nullptr;   // depthwise_conv1d.weight, ggml [K,1,hidden]
    // Final block RMSNorm.
    ggml_tensor* norm_out = nullptr;         // [hidden]
};

struct Gemma4AudioWeights {
    ggml_tensor* sscp0_conv = nullptr;   // layer0.conv.weight  ggml [3,3,1,c0]
    ggml_tensor* sscp0_norm = nullptr;   // layer0.norm.weight  [c0]
    ggml_tensor* sscp1_conv = nullptr;   // layer1.conv.weight  ggml [3,3,c0,c1]
    ggml_tensor* sscp1_norm = nullptr;   // layer1.norm.weight  [c1]
    ggml_tensor* sscp_input_proj = nullptr;  // input_proj_linear.weight [(c1*f2), hidden]
    std::vector<G4AudLayer> layers;          // 12
    ggml_tensor* output_proj_w = nullptr;    // output_proj.weight [hidden, out_proj_dims]
    ggml_tensor* output_proj_b = nullptr;    // output_proj.bias   [out_proj_dims]
    ggml_tensor* embed_proj    = nullptr;    // embed_audio.embedding_projection.weight [out_proj_dims, text_hidden]
    bool present = false;
};

// Build the Gemma-4 audio encoder graph. Returns projected soft-token embeddings
// [text_hidden, n_token]. n_token == subsampled frames (= cdim(cdim(n_frames))).
//   features  [n_mel, n_frames]   f32 log-mel (n_mel innermost)
//   pe        [hidden, P]         sinusoidal relative-position encoding (host)
//   win_mask  [Dwin, n_token]     additive window mask (0 valid, -inf if i<d) (host)
ggml_tensor* build_gemma4_audio_graph(ggml_context* ctx,
                                      const Gemma4AudioWeights& w,
                                      const Hparams& h,
                                      ggml_tensor* features,
                                      ggml_tensor* pe,
                                      ggml_tensor* win_mask,
                                      std::int32_t n_frames,
                                      std::int32_t n_token);

// Host helper: subsampled frame count after 2x Conv2d(k3,s2,p1) over time.
inline std::int32_t gemma4_audio_subsampled_len(std::int32_t n_frames) {
    auto cdim = [](std::int32_t n) { return (n - 1) / 2 + 1; };
    return cdim(cdim(n_frames));
}

// Host helper: sinusoidal relative-position encoding pe[P*hidden] row-major,
// P = max_past+1. Row r holds position_id = (P-1)-r; pe[r] =
// concat(sin(pos*inv), cos(pos*inv)) over the two hidden halves.
std::vector<float> gemma4_audio_rel_pos_enc(std::int32_t P, std::int32_t hidden);

// Host helper: window additive mask [Dwin*n_token] row-major (mask[d*n_token+i]),
// 0 when key i-d is valid (i >= d), -inf otherwise.
std::vector<float> gemma4_audio_window_mask(std::int32_t dwin, std::int32_t n_token);

}  // namespace pie_portable_driver
