#pragma once

// CSM-1B native audio OUTPUT — bound weights + the host-side generation entry.
//
// CSM = backbone (stock Llama-3.2-1B, reuses the naive forward in
// csm_backbone_forward) + depth decoder (4-layer RVQ sampler,
// csm_depth_decoder_forward) + Mimi codec decoder (mimi_decoder_forward). This
// header owns the bound `DeviceTensor` handles for all three and the adapter
// that turns them into the cuda-only Raw weight structs the kernels consume.
//
// See AUDIO_OUTPUT.md. The bind reads the `eustlb/csm-1b` tensor names verified
// present in the checkpoint:
//   backbone_model.layers.{i}.{input_layernorm,post_attention_layernorm}.weight
//   backbone_model.layers.{i}.self_attn.{q,k,v,o}_proj.weight
//   backbone_model.layers.{i}.mlp.{gate,up,down}_proj.weight
//   backbone_model.norm.weight
//   backbone_model.embed_tokens.embed_audio_tokens.weight  [32*2051, 2048]
//   embed_text_tokens.weight  [128256, 2048]
//   lm_head.weight            [2051, 2048]
//   depth_decoder.model.* (+ codebooks_head, inputs_embeds_projector, embed_tokens)
//   codec_model.* (decoder, decoder_transformer, upsample, quantizer)

#include <cstdint>
#include <memory>
#include <vector>

#include "model/config.hpp"
#include "model/csm/csm_backbone_forward.hpp"
#include "model/csm/csm_depth_decoder_forward.hpp"
#include "model/csm/mimi_decoder_adapter.hpp"   // MimiDecoderWeights
#include "tensor.hpp"

namespace pie_cuda_driver {
class LoadedModel;
namespace model {

struct CsmBackboneLayerTensors {
    const DeviceTensor* in_ln = nullptr;
    const DeviceTensor* post_ln = nullptr;
    const DeviceTensor* q = nullptr;
    const DeviceTensor* k = nullptr;
    const DeviceTensor* v = nullptr;
    const DeviceTensor* o = nullptr;
    const DeviceTensor* gate = nullptr;
    const DeviceTensor* up = nullptr;
    const DeviceTensor* down = nullptr;
};

struct CsmDepthLayerTensors {
    const DeviceTensor* in_ln = nullptr;
    const DeviceTensor* post_ln = nullptr;
    const DeviceTensor* q = nullptr;
    const DeviceTensor* k = nullptr;
    const DeviceTensor* v = nullptr;
    const DeviceTensor* o = nullptr;
    const DeviceTensor* gate = nullptr;
    const DeviceTensor* up = nullptr;
    const DeviceTensor* down = nullptr;
};

struct CsmWeights {
    // Backbone.
    const DeviceTensor* embed_text = nullptr;     // embed_text_tokens.weight
    const DeviceTensor* embed_audio = nullptr;    // embed_audio_tokens.weight
    const DeviceTensor* norm = nullptr;           // backbone_model.norm.weight
    const DeviceTensor* lm_head = nullptr;        // lm_head.weight
    std::vector<CsmBackboneLayerTensors> backbone_layers;

    // Depth decoder.
    const DeviceTensor* depth_embed_tokens = nullptr;       // depth_decoder.model.embed_tokens.weight
    const DeviceTensor* depth_inputs_proj = nullptr;        // inputs_embeds_projector.weight
    const DeviceTensor* depth_norm = nullptr;               // depth_decoder.model.norm.weight
    const DeviceTensor* depth_codebooks_head = nullptr;     // depth_decoder.codebooks_head.weight
    std::vector<CsmDepthLayerTensors> depth_layers;

    // Mimi decoder — codebook embeds are RESOLVED at bind time
    // (embed = embed_sum / cluster_usage.clamp(eps)); the bind owns those
    // materialized DeviceTensors so they outlive the model.
    MimiDecoderWeights mimi;
    std::vector<std::unique_ptr<DeviceTensor>> mimi_owned;   // resolved codebook embeds

    // bf16 copies of any F32 checkpoint tensors (eustlb/csm-1b ships F32; the
    // loader preserves the on-disk dtype). Owned here so the bf16 storage that
    // the Raw weight pointers reference outlives the model.
    std::vector<std::unique_ptr<DeviceTensor>> bf16_owned;

    HfConfig config;   // copy (carries hf.csm + backbone dims)

    // Resolve a tensor by name as bf16, casting F32 -> bf16 (owned) if needed.
    const DeviceTensor* bf16_tensor(LoadedModel& engine, const std::string& name);

    // Build the cuda-only Raw weight structs for the kernels.
    CsmBackboneRawWeights backbone_raw() const;
    CsmDepthRawWeights depth_raw() const;
    MimiDecoderRawWeights mimi_raw() const;
};

CsmWeights bind_csm(LoadedModel& engine, bool verbose);

}  // namespace model
}  // namespace pie_cuda_driver
