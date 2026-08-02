#pragma once

// Mimi neural-codec DECODER forward (bf16): 32 RVQ codes/frame → 24 kHz
// waveform. This is the OUTPUT-modality inverse of the gemma4 audio ENCODER
// (gemma4_audio_forward.hpp): instead of waveform → soft tokens, it takes the
// codec tokens an autoregressive model EMITS and turns them back into PCM.
// See AUDIO_OUTPUT.md for the full output-modality design.
//
// Reproduces transformers 5.9 `MimiModel.decode` (the codes→waveform half):
//   codes [B, 32, T]
//     → quantizer.decode        (MimiSplitResidualVectorQuantizer.decode:
//                                 cb0 via the 1-layer SEMANTIC RVQ, cb1..31 via
//                                 the 31-layer ACOUSTIC RVQ; each codebook is a
//                                 2048×256 table, per-codebook embeddings SUMMED
//                                 across the RVQ depth, then a 1×1 conv
//                                 `output_proj` lifts 256→512 PER RVQ GROUP and
//                                 the two groups' 512-d results are added)
//     → upsample                (MimiConvTranspose1d k4 s2 groups=512, no bias;
//                                 depthwise-grouped → ×2 frames, 12.5→25 Hz)
//     → decoder_transformer     (8× MimiTransformerLayer: LayerNorm(+bias) →
//                                 RoPE attention (sliding window 250) ×
//                                 layer_scale → LayerNorm(+bias) → GELU MLP ×
//                                 layer_scale; pre-norm residual)
//     → decoder (SEANet)        (Conv1d k7 512→1024; then 4× [ELU →
//                                 ConvTranspose1d (ratios 8,6,5,4) →
//                                 MimiResnetBlock]; ELU; Conv1d k3 64→1) → 24 kHz
//
// CUDA-only header (no model/loader headers) so the `.cu` doesn't pull in the
// toml++-heavy config headers nvcc cannot parse — same convention as
// gemma4_{vision,audio}_forward.hpp. The host call site builds
// `MimiDecoderRawWeights` from the bound weights via `DeviceTensor::data()`
// (see mimi_decoder_adapter.{hpp,cpp}).
//
// Parity harness: scripts/mimi_decoder_parity_ref.py dumps codes →
// dequantized_embeddings → upsampled_embeddings → decoder_transformer_out →
// SEANet intermediates → output_waveform + all decoder weights to
// /tmp/mimi_decoder_parity/. Parity metric (MULTIMODAL.md §2.2): bf16-vs-bf16
// rel_rms + cosine; fp32 refs are dumped too for a tight standalone check.
//
// All convs/norms are bf16-stored + fp32-compute, matching the driver and the
// gemma4 encoders.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace pie_cuda_driver::model {

// One SEANet 1-D conv (causal `MimiConv1d`) with bias.
//   weight [out_ch, in_ch, k]  (PyTorch Conv1d layout)
//   bias   [out_ch]            (may be null)
// `stride`/`dilation` per the layer; padding is causal (left-pad
// padding_total = (k-1)*dilation + 1 - stride, then `_get_extra_padding`).
struct MimiConvRaw {
    const __nv_bfloat16* w = nullptr;
    const __nv_bfloat16* b = nullptr;
    int in_ch = 0;
    int out_ch = 0;
    int kernel = 1;
    int stride = 1;
    int dilation = 1;
};

// One SEANet transposed conv (`MimiConvTranspose1d`). For the SEANet upsamplers
// `groups=1`; for the codec `upsample` stage `groups = upsample_groups = 512`
// (depthwise). Causal trim on the right by `padding_total` (trim_right_ratio=1).
//   weight [in_ch, out_ch/groups, k]   (PyTorch ConvTranspose1d layout)
struct MimiConvTRaw {
    const __nv_bfloat16* w = nullptr;
    const __nv_bfloat16* b = nullptr;   // null for the groups=512 upsample
    int in_ch = 0;
    int out_ch = 0;
    int kernel = 1;
    int stride = 1;
    int groups = 1;
};

// One SEANet residual block (`MimiResnetBlock`):
//   out = shortcut(x) + conv2(ELU(conv1(ELU(x))))
// conv1: dim→dim/compress, kernel=residual_kernel_size, dilation=dilations[0]
// conv2: dim/compress→dim, kernel=1, dilation=1
// shortcut is Identity for Mimi (use_conv_shortcut=false), so it is omitted.
struct MimiResnetRaw {
    MimiConvRaw conv1;   // block.1
    MimiConvRaw conv2;   // block.3
};

// One upsampling stage of the SEANet decoder: ELU → transposed conv → resnet.
struct MimiDecoderStageRaw {
    MimiConvTRaw convtr;     // ConvTranspose1d (ratio*2 kernel, stride=ratio)
    MimiResnetRaw resnet;    // MimiResnetBlock at the upsampled channel count
};

// One MimiTransformerLayer of the decoder_transformer (post-quantize, pre-SEANet
// refinement). LayerNorms carry bias; attention/MLP outputs are scaled by a
// learned per-channel `layer_scale` before the residual add.
struct MimiXfLayerRaw {
    const __nv_bfloat16* in_ln_w = nullptr;    // input_layernorm.weight [512]
    const __nv_bfloat16* in_ln_b = nullptr;    // input_layernorm.bias   [512]
    const __nv_bfloat16* q = nullptr;          // self_attn.q_proj [512,512]
    const __nv_bfloat16* k = nullptr;          // self_attn.k_proj [512,512]
    const __nv_bfloat16* v = nullptr;          // self_attn.v_proj [512,512]
    const __nv_bfloat16* o = nullptr;          // self_attn.o_proj [512,512]
    const __nv_bfloat16* attn_scale = nullptr; // self_attn_layer_scale.scale [512]
    const __nv_bfloat16* post_ln_w = nullptr;  // post_attention_layernorm.weight
    const __nv_bfloat16* post_ln_b = nullptr;  // post_attention_layernorm.bias
    const __nv_bfloat16* fc1 = nullptr;        // mlp.fc1 [2048,512]
    const __nv_bfloat16* fc2 = nullptr;        // mlp.fc2 [512,2048]
    const __nv_bfloat16* mlp_scale = nullptr;  // mlp_layer_scale.scale [512]
};

struct MimiDecoderRawWeights {
    // ── RVQ dequantize (MimiSplitResidualVectorQuantizer) ───────────────────
    // Resolved per-codebook embedding tables `embed = embed_sum /
    // cluster_usage.clamp(min=eps)`, each [codebook_size, codebook_dim] = [2048,
    // 256]. Index 0 is the single semantic codebook; 1..31 the acoustic ones.
    // (The parity ref dumps these as quantizer.*.codebook.embed.npy so the CUDA
    // side embeds directly without recomputing the division.)
    std::vector<const __nv_bfloat16*> codebook_embed;   // [num_codebooks][2048,256]
    // 1×1 conv output projections 256→512 (one per RVQ GROUP). Applied to that
    // group's summed 256-d residual; the two 512-d outputs are then added.
    const __nv_bfloat16* semantic_output_proj = nullptr;   // [512,256,1]
    const __nv_bfloat16* acoustic_output_proj = nullptr;   // [512,256,1]
    int num_semantic = 1;        // codebooks 0 .. num_semantic-1 → semantic RVQ

    // ── upsample (12.5 → 25 Hz; ConvTranspose1d k4 s2 groups=512) ───────────
    MimiConvTRaw upsample;

    // ── decoder_transformer (8 layers) ──────────────────────────────────────
    std::vector<MimiXfLayerRaw> xf_layers;
    const __nv_bfloat16* xf_final_ln_w = nullptr;  // decoder_transformer norm
    const __nv_bfloat16* xf_final_ln_b = nullptr;  // (null if model has no final LN)

    // ── SEANet decoder ───────────────────────────────────────────────────────
    MimiConvRaw seanet_in;                       // layer0: Conv1d k7 512→1024
    std::vector<MimiDecoderStageRaw> seanet_stages;  // one per upsampling ratio
    MimiConvRaw seanet_out;                      // last: Conv1d k3 64→1

    // ── dims / hyperparams (from config.json; defaults = kyutai/mimi) ────────
    int hidden = 512;            // codec hidden_size (post output_proj / pre SEANet)
    int codebook_dim = 256;      // vector_quantization_hidden_dimension
    int codebook_size = 2048;
    int num_codebooks = 32;
    int num_filters = 64;
    std::vector<int> upsampling_ratios{8, 6, 5, 4};
    int xf_heads = 8;
    int xf_kv_heads = 8;
    int xf_head_dim = 64;
    int xf_intermediate = 2048;
    int xf_sliding_window = 250;
    float xf_rope_theta = 10000.0f;
    float norm_eps = 1e-5f;
    int sampling_rate = 24000;
    bool causal = true;          // use_causal_conv
};

// Decode one clip's RVQ codes → waveform.
//   codes      : i32 host pointer, row-major [num_codebooks, n_frames]
//                (codebook-major, matching the parity ref's codes[0])
//   n_frames   : number of Mimi frames (12.5 Hz)
//   out_wave   : f32 DEVICE buffer, length >= mimi_decoder_num_samples(...)
//                (24 kHz mono PCM). Caller owns/allocates it.
// Returns the number of output samples written. Allocates internal scratch
// (first-cut; a workspace arena is a follow-up, as in the gemma4 encoders).
int run_mimi_decoder(const MimiDecoderRawWeights& w,
                     const std::int32_t* codes,
                     int n_frames,
                     float* out_wave,
                     cudaStream_t stream = 0);

// Optional parity-debug hook: when set, `run_mimi_decoder` invokes `fn` after
// each named pipeline stage with that stage's DEVICE bf16 buffer + element
// count, letting a standalone harness compare against the HF staged dumps.
// No-op (and zero overhead) when unset. See tests/mimi_decoder_full_parity.cu.
typedef void (*MimiDecoderCkptFn)(const char* name, const __nv_bfloat16* dev,
                                  long numel, void* user);
void set_mimi_decoder_ckpt(MimiDecoderCkptFn fn, void* user);

// Output sample count for `n_frames` input frames: upsample (×2) then the
// SEANet upsampling product (∏ ratios). For kyutai/mimi: 2·8·6·5·4 = 1920
// samples per 12.5 Hz frame → 24 kHz (frame_rate 12.5 → 24000/12.5 = 1920). The
// causal convs preserve length, so this is exact before any padding-mask trim.
inline int mimi_decoder_num_samples(const MimiDecoderRawWeights& w, int n_frames) {
    long s = (long)n_frames * 2;  // upsample ConvTranspose1d stride 2
    for (int r : w.upsampling_ratios) s *= r;
    return (int)s;
}

}  // namespace pie_cuda_driver::model
