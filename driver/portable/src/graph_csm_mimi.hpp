#pragma once
// Mimi codec vocoder for CSM-1B audio output: RVQ codes [num_quantizers,
// n_frames] -> 24 kHz PCM. Mirrors the CUDA run_mimi_decoder
// (mimi_decoder_forward.cu): dequantize the residual codes, project, run the
// 8-layer decoder transformer + upsample ConvTranspose1d, then the SEANet
// decoder (Conv1d + 4 ConvTranspose1d upsamplers w/ residual blocks) -> waveform.

#include <cstdint>
#include <vector>

namespace pie_portable_driver {

class Model;

// Decode `codes` (codebook-major [num_quantizers * n_frames]) to 24 kHz mono f32
// PCM, appended into `out_pcm`. Reads the loaded Mimi weights from `model`.
void csm_mimi_decode(Model& model, const std::int32_t* codes, int n_frames,
                     std::vector<float>& out_pcm);

}  // namespace pie_portable_driver
