#pragma once
// CSM-1B (sesame/csm-1b) weight structs for the portable ggml driver. Native
// audio OUTPUT (TTS). Mirrors the CUDA bind_csm (driver/cuda/src/model/csm.cpp).
//
// Three sub-models:
//   * Backbone  — Llama-3.2-1B (hidden 2048, 16 layers, 32q/8kv, head_dim 64,
//                 llama3 rope theta 5e5). Embeds text + audio frames, produces
//                 the hidden state whose lm_head gives codebook-0 logits.
//   * Depth     — 4-layer Llama (hidden 1024) that samples codebooks 1..31 of a
//                 frame from the backbone hidden + the running codebook embeds.
//   * Mimi      — codec vocoder: 32 RVQ codebooks -> upsample ConvTranspose1d ->
//                 8-layer decoder transformer -> SEANet decoder -> 24 kHz PCM.
//
// Tensor names are root-level (no "model." prefix), declared directly via
// Model::declare_. CSM ships torch_dtype float32; the loader downcasts large
// matmul weights to bf16 and keeps norms/conv/scales in their loaded dtype.

#include <vector>

#include <ggml.h>

namespace pie_portable_driver {

// One decoder layer (backbone or depth): standard Llama block, no biases.
struct CsmDecLayer {
    ggml_tensor* in_ln   = nullptr;  // input_layernorm.weight
    ggml_tensor* post_ln = nullptr;  // post_attention_layernorm.weight
    ggml_tensor* q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr;
    ggml_tensor* gate = nullptr, *up = nullptr, *down = nullptr;
};

// A 1D conv (Conv1d or ConvTranspose1d) with bias. `weight`/`bias` are the
// loaded ggml tensors; the conv hyperparams are folded in at graph build.
struct CsmConv {
    ggml_tensor* weight = nullptr;
    ggml_tensor* bias   = nullptr;
    int stride = 1;
    int dilation = 1;
    int groups = 1;
};

// One Mimi decoder-transformer layer (pre-norm w/ bias, per-sublayer LayerScale).
struct MimiXfLayer {
    ggml_tensor* in_ln_w = nullptr, *in_ln_b = nullptr;
    ggml_tensor* q = nullptr, *k = nullptr, *v = nullptr, *o = nullptr;
    ggml_tensor* attn_scale = nullptr;      // self_attn_layer_scale.scale
    ggml_tensor* post_ln_w = nullptr, *post_ln_b = nullptr;
    ggml_tensor* fc1 = nullptr, *fc2 = nullptr;
    ggml_tensor* mlp_scale = nullptr;       // mlp_layer_scale.scale
};

// One SEANet upsampler stage: a ConvTranspose1d followed by a residual block
// (two Conv1d's). The residual is x + conv2(elu(conv1(elu(x)))).
struct MimiSeanetStage {
    CsmConv convtr;
    CsmConv resnet_conv1;
    CsmConv resnet_conv2;
};

struct CsmWeights {
    bool present = false;

    // ── Backbone ──
    ggml_tensor* embed_text  = nullptr;  // embed_text_tokens.weight [text_vocab, hidden]
    ggml_tensor* embed_audio = nullptr;  // backbone_model.embed_tokens.embed_audio_tokens.weight
    ggml_tensor* bb_norm     = nullptr;  // backbone_model.norm.weight
    ggml_tensor* lm_head     = nullptr;  // lm_head.weight [audio_vocab, hidden] (codebook-0 head)
    std::vector<CsmDecLayer> backbone_layers;

    // ── Depth decoder ──
    ggml_tensor* depth_embed_tokens = nullptr;  // depth_decoder.model.embed_tokens.weight
    ggml_tensor* depth_inputs_proj  = nullptr;  // depth_decoder.model.inputs_embeds_projector.weight
    ggml_tensor* depth_norm         = nullptr;  // depth_decoder.model.norm.weight
    ggml_tensor* depth_codebooks_head = nullptr;  // depth_decoder.codebooks_head.weight [(NCB-1)*vocab, hidden]
    std::vector<CsmDecLayer> depth_layers;

    // ── Mimi codec ──
    // RVQ codebooks: codebook[i] resolved from embed_sum/cluster_usage at graph
    // time. Stored as the raw pair per quantizer (index 0 = semantic, 1.. = acoustic).
    std::vector<ggml_tensor*> codebook_embed_sum;     // [NCB] each [codebook_size, codebook_dim]
    std::vector<ggml_tensor*> codebook_cluster_usage; // [NCB] each [codebook_size]
    ggml_tensor* semantic_output_proj = nullptr;
    ggml_tensor* acoustic_output_proj = nullptr;
    CsmConv upsample;                       // codec_model.upsample.conv (ConvTranspose1d, groups=512)
    std::vector<MimiXfLayer> xf_layers;     // decoder_transformer (8 layers)
    CsmConv seanet_in;                      // decoder.layers.0.conv (Conv1d k7)
    std::vector<MimiSeanetStage> seanet_stages;  // 4 upsampler stages
    CsmConv seanet_out;                     // decoder.layers.14.conv (Conv1d k3)
};

}  // namespace pie_portable_driver
