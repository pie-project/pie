#pragma once

// Host-side bridge from the bound `MimiDecoderWeights` (DeviceTensor handles)
// to the cuda-only `MimiDecoderRawWeights` (raw bf16 pointers) the Mimi decoder
// kernels consume. This header includes the model/tensor headers (toml++ etc.),
// so it is only ever included by host `.cpp` files — never by an nvcc-compiled
// `.cu`. Mirrors gemma4_audio_adapter.hpp.
//
// EXPECTED upstream types. These will live in a SHARED model header (this task
// must NOT edit shared files — a teammate wires `bind_mimi_decoder` there). To
// keep this adapter self-compiling in the meantime, the expected structs are
// declared HERE under `#ifndef PIE_HAS_MIMI_DECODER_WEIGHTS`. When the real
// types land, that header should `#define PIE_HAS_MIMI_DECODER_WEIGHTS` (or this
// block be deleted and the include swapped) so there is a single definition.
//
// Bound from the `kyutai/mimi` checkpoint (or CSM's `codec_model.` prefix), all
// by the real tensor names (verified present, missing=0, by
// scripts/mimi_decoder_parity_ref.py → 225 decoder weights + 32 resolved
// codebook embeds):
//   quantizer.semantic_residual_vector_quantizer.layers.0.codebook.{embed_sum,
//       cluster_usage}            → resolved embed [2048,256]
//   quantizer.semantic_residual_vector_quantizer.output_proj.weight  [512,256,1]
//   quantizer.acoustic_residual_vector_quantizer.layers.{0..30}.codebook.*
//   quantizer.acoustic_residual_vector_quantizer.{input_proj,output_proj}.weight
//   upsample.conv.weight          [512,1,4]  (groups=512, no bias)
//   decoder_transformer.layers.{i}.{input_layernorm,post_attention_layernorm}.{weight,bias}
//   decoder_transformer.layers.{i}.self_attn.{q,k,v,o}_proj.weight  [512,512]
//   decoder_transformer.layers.{i}.{self_attn,mlp}_layer_scale.scale  [512]
//   decoder_transformer.layers.{i}.mlp.{fc1,fc2}.weight  [2048,512]/[512,2048]
//   decoder.layers.0.conv.{weight,bias}        (Conv1d k7 512→1024)
//   decoder.layers.{2,5,8,11}.conv.{weight,bias}  (ConvTranspose1d upsamplers)
//   decoder.layers.{3,6,9,12}.block.{1,3}.conv.{weight,bias}  (Resnet blocks)
//   decoder.layers.14.conv.{weight,bias}       (Conv1d k3 64→1)
//
// NOTE on codebook embeds: the loader must materialize `embed = embed_sum /
// cluster_usage.clamp(min=1e-5)[:,None]` once at load time (the parity ref dumps
// this resolved table). The CUDA side then embeds directly. The 1×1 output_proj
// conv weight [512,256,1] is treated as [512,256] (kernel dim squeezed).

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <vector>

#include "model/mimi_decoder_forward.hpp"  // MimiDecoderRawWeights, run_mimi_decoder
#include "tensor.hpp"                        // DeviceTensor

namespace pie_cuda_driver::model {

#ifndef PIE_HAS_MIMI_DECODER_WEIGHTS
// ── Expected bound-weight contract (teammate moves this to the shared header) ─

struct MimiConvWeights {
    const DeviceTensor* weight = nullptr;   // [out, in, k]
    const DeviceTensor* bias   = nullptr;   // [out] (null for the groups=512 upsample)
    int stride = 1;
    int dilation = 1;
    int groups = 1;
};

struct MimiResnetWeights {
    MimiConvWeights conv1;   // block.1  (dim → dim/compress, k=residual_kernel_size)
    MimiConvWeights conv2;   // block.3  (dim/compress → dim, k=1)
};

struct MimiDecoderStageWeights {
    MimiConvWeights convtr;          // ConvTranspose1d (ratio*2 kernel, stride=ratio)
    MimiResnetWeights resnet;
};

struct MimiXfLayerWeights {
    const DeviceTensor* input_layernorm_weight = nullptr;
    const DeviceTensor* input_layernorm_bias   = nullptr;
    const DeviceTensor* q_proj = nullptr;
    const DeviceTensor* k_proj = nullptr;
    const DeviceTensor* v_proj = nullptr;
    const DeviceTensor* o_proj = nullptr;
    const DeviceTensor* self_attn_layer_scale = nullptr;
    const DeviceTensor* post_attention_layernorm_weight = nullptr;
    const DeviceTensor* post_attention_layernorm_bias   = nullptr;
    const DeviceTensor* mlp_fc1 = nullptr;
    const DeviceTensor* mlp_fc2 = nullptr;
    const DeviceTensor* mlp_layer_scale = nullptr;
};

struct MimiDecoderConfigLite {
    int hidden_size = 512;
    int codebook_dim = 256;          // vector_quantization_hidden_dimension
    int codebook_size = 2048;
    int num_quantizers = 32;
    int num_semantic_quantizers = 1;
    int num_filters = 64;
    std::vector<int> upsampling_ratios{8, 6, 5, 4};
    int xf_num_attention_heads = 8;
    int xf_num_key_value_heads = 8;
    int xf_head_dim = 64;
    int xf_intermediate_size = 2048;
    int xf_sliding_window = 250;
    float xf_rope_theta = 10000.0f;
    float norm_eps = 1e-5f;
    int sampling_rate = 24000;
    bool use_causal_conv = true;
    int upsample_groups = 512;
    int upsample_kernel = 4;
    int upsample_stride = 2;
};

struct MimiDecoderWeights {
    // Resolved per-codebook embedding tables [num_quantizers][2048,256], in
    // codebook order (index 0 = semantic, 1..31 = acoustic). Loader materializes
    // embed = embed_sum / cluster_usage.clamp(min=eps).
    std::vector<const DeviceTensor*> codebook_embed;
    const DeviceTensor* semantic_output_proj = nullptr;   // [512,256,1]
    const DeviceTensor* acoustic_output_proj = nullptr;   // [512,256,1]

    MimiConvWeights upsample;                              // ConvTranspose1d k4 s2 g512

    std::vector<MimiXfLayerWeights> xf_layers;
    const DeviceTensor* xf_final_ln_weight = nullptr;      // null if absent
    const DeviceTensor* xf_final_ln_bias   = nullptr;

    MimiConvWeights seanet_in;                             // Conv1d k7 512→1024
    std::vector<MimiDecoderStageWeights> seanet_stages;
    MimiConvWeights seanet_out;                            // Conv1d k3 64→1

    MimiDecoderConfigLite config;
};
#endif  // PIE_HAS_MIMI_DECODER_WEIGHTS

// Extract raw device pointers + dims from the bound weights.
MimiDecoderRawWeights to_mimi_decoder_raw(const MimiDecoderWeights& w);

// Convenience overload: build `MimiDecoderRawWeights` and run the decoder.
//   codes    : host i32 [num_codebooks, n_frames]
//   out_wave : f32 DEVICE buffer (caller-owned), length >= num_samples.
// Returns the number of samples written.
int run_mimi_decoder(const MimiDecoderWeights& w,
                     const std::int32_t* codes,
                     int n_frames,
                     float* out_wave,
                     cudaStream_t stream = 0);

}  // namespace pie_cuda_driver::model
