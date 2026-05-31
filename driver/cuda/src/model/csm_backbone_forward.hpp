#pragma once

// CSM backbone forward + the frame-stepped generation primitive (bf16 store /
// fp32 compute). The CSM backbone is a stock Llama-3.2-1B (16 layers / 2048
// hidden, 32q/8kv, head_dim 64, SiLU MLP 8192, RMSNorm, llama3-scaled RoPE
// theta 500000). It reuses the same naive kernel style as
// csm_depth_decoder_forward.cu (correctness + parity first; the sequences here
// are short — a text prompt + up to a few hundred audio frames).
//
// This file owns the OUTER loop of AUDIO_OUTPUT.md §1.1 (the genuinely-new
// control flow): prefill the backbone over the text prompt, then per frame:
//   1. backbone decode step over the current row -> sample codebook 0
//      (argmax of lm_head(last_hidden)) and grab last_hidden [2048],
//   2. run the depth decoder's 31-step inner loop (run_csm_depth_decoder_frame)
//      to get codebooks 1..31,
//   3. assemble the 32-code frame, re-embed it (sum of 32 offset codebook
//      embeds) as the next backbone input row,
//   4. stop on all-EOS (every codebook == codebook_eos_token_id) or max_frames.
// The collected codes [32, n_frames] then feed run_mimi_decoder -> PCM.
//
// The backbone KV cache is a plain contiguous [maxL, KV, hd] cache here (the
// outer loop); the depth-decoder cache is allocated/freed per frame internally
// (ephemeral, AUDIO_OUTPUT.md §5). CUDA-only header (no model/loader/toml++) so
// the `.cu` stays nvcc-parseable — same convention as the depth decoder + mimi.
//
// The host call site (csm.cpp) builds `CsmBackboneRawWeights` + the depth +
// mimi raw weights from the bound DeviceTensors and calls
// `csm_generate_audio` once per `generate_audio` request.

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

#include "model/csm_depth_decoder_forward.hpp"   // CsmDepthRawWeights
#include "model/mimi_decoder_forward.hpp"          // MimiDecoderRawWeights

namespace pie_cuda_driver::model {

// One backbone transformer layer (plain Llama block; no biases, no qk-norm).
struct CsmBackboneLayerRaw {
    const __nv_bfloat16* in_ln_w = nullptr;
    const __nv_bfloat16* post_ln_w = nullptr;
    const __nv_bfloat16* q = nullptr;     // [n_heads*hd, hidden]
    const __nv_bfloat16* k = nullptr;     // [n_kv*hd, hidden]
    const __nv_bfloat16* v = nullptr;     // [n_kv*hd, hidden]
    const __nv_bfloat16* o = nullptr;     // [hidden, n_heads*hd]
    const __nv_bfloat16* gate = nullptr;  // [inter, hidden]
    const __nv_bfloat16* up = nullptr;    // [inter, hidden]
    const __nv_bfloat16* down = nullptr;  // [hidden, inter]
};

struct CsmBackboneRawWeights {
    // Text token embedding [text_vocab, hidden] (embed_text_tokens.weight).
    const __nv_bfloat16* embed_text = nullptr;
    // Audio token embedding [num_codebooks*audio_vocab, hidden]
    // (backbone_model.embed_tokens.embed_audio_tokens.weight). Indexed by
    // id + codebook_idx*audio_vocab; an audio frame embed is the SUM over the
    // 32 codebooks of the previous frame.
    const __nv_bfloat16* embed_audio = nullptr;
    std::vector<CsmBackboneLayerRaw> layers;
    const __nv_bfloat16* norm_w = nullptr;   // backbone_model.norm.weight
    // codebook-0 head [audio_vocab, hidden] (lm_head.weight).
    const __nv_bfloat16* lm_head = nullptr;

    int hidden = 2048;
    int num_layers = 16;
    int num_heads = 32;
    int num_kv_heads = 8;
    int head_dim = 64;
    int intermediate = 8192;
    int text_vocab = 128256;
    int audio_vocab = 2051;       // per-codebook vocab (== top-level vocab_size)
    int num_codebooks = 32;
    int codebook_eos_token_id = 0;   // all-EOS frame -> stop
    float norm_eps = 1e-5f;
    // RoPE (llama3 YaRN scaling; backbone params: orig_max_pos 1024).
    float rope_theta = 500000.0f;
    float rope_factor = 32.0f;
    float rope_low_freq_factor = 0.125f;
    float rope_high_freq_factor = 0.5f;
    int rope_original_max_position = 1024;
};

// Run the full CSM generation loop and return 24 kHz mono PCM.
//   prompt_ids   : host i32 [n_prompt] text token ids (the "[speaker]text"
//                  prompt already tokenized by the SDK).
//   max_frames   : cap on the number of Mimi frames produced.
//   bb / depth / mimi : the three bound raw-weight sets.
//   out_pcm      : host f32 vector, resized to the produced sample count.
//   out_codes    : optional host i32 [32 * n_frames] (codebook-major), the
//                  emitted RVQ codes (for debugging/parity). May be nullptr.
// Returns the number of Mimi frames produced (n_frames). Greedy (argmax) for a
// reproducible reference, matching csm_generate_ref.py do_sample=False.
// Cast a device F32 buffer to bf16 (used by bind_csm to normalize an F32
// checkpoint — eustlb/csm-1b ships torch_dtype float32, and the default loader
// ABI preserves the source dtype, so the bound DeviceTensors are F32. The CSM
// kernels are bf16-store/fp32-compute, so each F32 weight is cast once at bind).
void csm_cast_f32_to_bf16(const float* in, __nv_bfloat16* out, long n,
                          cudaStream_t stream = 0);

// Resolve a Mimi RVQ codebook embedding table on device:
//   embed[row, d] = embed_sum[row, d] / max(cluster_usage[row], eps)
// `embed_sum` is [rows, dim] bf16, `cluster_usage` is [rows] bf16, `out` is a
// caller-allocated [rows, dim] bf16 buffer. Used once per codebook at bind time
// (MimiDecoderWeights.codebook_embed contract; mirrors the parity ref's
// `embed = embed_sum / cluster_usage.clamp(min=eps)`).
void mimi_resolve_codebook_embed(const __nv_bfloat16* embed_sum,
                                 const __nv_bfloat16* cluster_usage,
                                 __nv_bfloat16* out,
                                 int rows, int dim, float eps,
                                 cudaStream_t stream = 0);

int csm_generate_audio(const CsmBackboneRawWeights& bb,
                       const CsmDepthRawWeights& depth,
                       const MimiDecoderRawWeights& mimi,
                       const std::int32_t* prompt_ids,
                       int n_prompt,
                       int max_frames,
                       std::vector<float>& out_pcm,
                       std::vector<std::int32_t>* out_codes = nullptr,
                       cudaStream_t stream = 0);

}  // namespace pie_cuda_driver::model
