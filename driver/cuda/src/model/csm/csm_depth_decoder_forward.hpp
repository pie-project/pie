#pragma once

// CSM depth-decoder forward + RVQ frame sampler (bf16 store / fp32 compute).
//
// This is the genuinely-new OUTPUT-modality engine piece (AUDIO_OUTPUT.md §3):
// the CSM backbone is a stock Llama-3.2-1B (reuses Pie's verified `llama_like`)
// and the Mimi decoder (codes -> waveform) is parity-verified separately
// (mimi_decoder_forward). What remains is the depth decoder — a 4-layer /
// 1024-hidden Llama that, seeded by the backbone's last hidden state + the
// backbone-sampled codebook 0, autoregressively predicts codebooks 1..31 of one
// Mimi frame (the RVQ depth) — plus the per-frame 31-step inner loop.
//
// Reproduces transformers 5.9 `CsmDepthDecoderForCausalLM` + the per-frame inner
// loop of `CsmGenerationMixin._sample`:
//   * input embed: embed_tokens(input_id + codebook_idx*vocab_size), with depth
//     position 0 OVERWRITTEN by backbone_last_hidden_state (both [2048]); then
//     `inputs_embeds_projector` (Linear 2048->1024, no bias) projects every row,
//   * 4 Llama decoder layers (RMSNorm pre-norm, GQA 8q/2kv head_dim 128, RoPE
//     theta 500000 llama3-scaled orig_max_pos 16, SiLU MLP 8192), seq <= 33,
//   * position-specific head: at the step producing codebook `i` (1..31), apply
//     codebooks_head.weight[i-1] (a [1024,2051] slice) -> 2051 logits -> argmax
//     (greedy) -> next codebook id.  The just-sampled id is re-embedded at the
//     next depth position with offset `i*vocab_size`.
//
// CUDA-only header (no model/loader/toml++ headers) so the `.cu` stays
// nvcc-parseable — same convention as mimi_decoder_forward.hpp /
// gemma4_audio_forward.hpp. The host call site builds `CsmDepthRawWeights` from
// the bound DeviceTensors (see csm.cpp / the adapter).
//
// Parity: scripts/csm_depth_decoder_parity_ref.py dumps, for one frame, the
// backbone hidden seed + cb0 + the emitted cb1..31 + per-step logits + all
// depth weights (bf16) to /tmp/csm_depth_parity/. The standalone harness
// tests/csm_depth_decoder_parity.cu binds those and checks the 31 emitted codes
// match HF exactly (greedy argmax is the natural metric for discrete codes).

#include <cuda_bf16.h>
#include <cuda_runtime.h>

#include <cstdint>
#include <vector>

namespace pie_cuda_driver::model {

// One depth-decoder transformer layer (plain Llama block; no biases, no qk-norm).
struct CsmDepthLayerRaw {
    const __nv_bfloat16* in_ln_w = nullptr;   // input_layernorm.weight [1024]
    const __nv_bfloat16* post_ln_w = nullptr; // post_attention_layernorm.weight [1024]
    const __nv_bfloat16* q = nullptr;         // self_attn.q_proj [n_heads*hd, 1024]
    const __nv_bfloat16* k = nullptr;         // self_attn.k_proj [n_kv*hd, 1024]
    const __nv_bfloat16* v = nullptr;         // self_attn.v_proj [n_kv*hd, 1024]
    const __nv_bfloat16* o = nullptr;         // self_attn.o_proj [1024, n_heads*hd]
    const __nv_bfloat16* gate = nullptr;      // mlp.gate_proj [inter, 1024]
    const __nv_bfloat16* up = nullptr;        // mlp.up_proj   [inter, 1024]
    const __nv_bfloat16* down = nullptr;      // mlp.down_proj [1024, inter]
};

struct CsmDepthRawWeights {
    // Audio-token embedding table [num_codebooks*vocab_size, backbone_hidden].
    // Tied with the backbone's embed_audio_tokens. Indexed by
    // input_id + codebook_idx*vocab_size. Shape [32*2051, 2048] = [65632, 2048].
    const __nv_bfloat16* embed_tokens = nullptr;
    // inputs_embeds_projector [hidden, backbone_hidden] = [1024, 2048], no bias.
    const __nv_bfloat16* inputs_embeds_projector = nullptr;
    std::vector<CsmDepthLayerRaw> layers;     // num_hidden_layers (4)
    const __nv_bfloat16* norm_w = nullptr;    // final RMSNorm [1024]
    // codebooks_head [num_codebooks-1, hidden, vocab] = [31, 1024, 2051]. At the
    // step producing codebook i (1..31) we use weight[i-1] (a [1024,2051] slab),
    // applied as logits[v] = sum_h hidden[h] * weight[i-1][h][v].
    const __nv_bfloat16* codebooks_head = nullptr;

    // dims (defaults = eustlb/csm-1b depth_decoder_config)
    int hidden = 1024;
    int backbone_hidden = 2048;
    int num_layers = 4;
    int num_heads = 8;
    int num_kv_heads = 2;
    int head_dim = 128;
    int intermediate = 8192;
    int num_codebooks = 32;
    int vocab_size = 2051;
    float norm_eps = 1e-5f;
    // RoPE (llama3 YaRN scaling)
    float rope_theta = 500000.0f;
    float rope_factor = 32.0f;
    float rope_low_freq_factor = 0.001953125f;
    float rope_high_freq_factor = 0.0078125f;
    int rope_original_max_position = 16;
};

// Generate codebooks 1..31 for ONE frame, greedily (argmax).
//   bb_hidden : DEVICE bf16 [backbone_hidden] — the backbone's last hidden state
//               (the seed at depth position 0).
//   cb0       : the backbone-sampled codebook-0 id for this frame.
//   out_codes : host i32 [num_codebooks-1] = [31] — the emitted cb1..cb31.
// The depth KV cache is allocated/freed internally and is EPHEMERAL per frame
// (carries no cross-frame state), per AUDIO_OUTPUT.md §5.
void run_csm_depth_decoder_frame(const CsmDepthRawWeights& w,
                                 const __nv_bfloat16* bb_hidden,
                                 std::int32_t cb0,
                                 std::int32_t* out_codes,
                                 cudaStream_t stream = 0);

// Variant that also returns the per-step logits (host f32 [31, vocab]) — used by
// the parity harness to compare against HF logits, not just the argmax. Pass
// `out_logits = nullptr` to skip.
void run_csm_depth_decoder_frame_dbg(const CsmDepthRawWeights& w,
                                     const __nv_bfloat16* bb_hidden,
                                     std::int32_t cb0,
                                     std::int32_t* out_codes,
                                     float* out_logits,
                                     cudaStream_t stream = 0);

}  // namespace pie_cuda_driver::model
