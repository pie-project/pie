#pragma once
// CSM-1B audio generation (TTS) for the portable ggml driver. Drives the loaded
// CsmWeights (backbone + depth decoder + Mimi codec) through the autoregressive
// frame loop and the Mimi vocoder to produce 24 kHz PCM. Mirrors the CUDA
// csm_generate_audio (csm_backbone_forward.cu) + run_mimi_decoder.
//
// Per request: prefill the backbone over the text prompt, then per frame
//   1. backbone decode -> argmax(lm_head) = codebook 0,
//   2. depth decoder 31-step loop -> codebooks 1..31,
//   3. re-embed the 32-code frame (sum of offset codebook embeds) as the next
//      backbone row,
//   4. stop on all-EOS (first NCB-1 codebooks == codebook_eos) or max_frames.
// The collected codes [NCB, n_frames] then feed the Mimi decoder -> PCM. Greedy.

#include <cstdint>
#include <vector>

namespace pie_portable_driver {

class Model;

// Generate speech for `prompt` (token ids). Fills `out_pcm` with 24 kHz mono f32
// samples and (optionally) `out_codes` with the [NCB * n_frames] codebook-major
// codes. Returns the number of Mimi frames produced. Throws on a non-CSM model.
int csm_generate_audio(Model& model,
                       const std::int32_t* prompt, int n_prompt,
                       int max_frames,
                       std::vector<float>& out_pcm,
                       std::vector<std::int32_t>* out_codes);

}  // namespace pie_portable_driver
