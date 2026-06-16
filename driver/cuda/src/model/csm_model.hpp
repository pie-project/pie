#pragma once

// CSM IModel — holds the bound CSM weights (backbone + depth + Mimi) and owns
// the `generate_audio` entry the audio-out serving path invokes. Unlike the
// other arches, CSM does NOT participate in the per-step batched forward loop
// (it emits a 32-code frame per backbone step via a nested depth loop, breaking
// the one-token-per-step assumption — AUDIO_OUTPUT.md §3). So `body()` is not a
// supported text-generation path; the model is driven exclusively through
// `generate_audio`, a single synchronous call that runs prefill + the frame
// loop + Mimi decode and returns 24 kHz PCM.

#include <cstdint>
#include <vector>

#include "model/csm.hpp"
#include "model/imodel.hpp"

namespace pie_cuda_driver::model {

class CsmModel final : public IModel {
public:
    explicit CsmModel(CsmWeights weights) : weights_(std::move(weights)) {}

    void prepare(AttentionWorkspace&, const ForwardFn::PrepareInputs&) override {}
    void body(Qwen3Workspace&, KvCache&, AttentionWorkspace&,
              ops::CublasHandle&, const ForwardFn::ForwardInputs&) override;

    ModelCapabilities capabilities() const override { return {}; }

    // Run the full CSM generation: text prompt -> 24 kHz mono PCM.
    //   prompt_ids : the tokenized "[speaker]text" prompt.
    //   max_frames : cap on Mimi frames.
    //   out_codes  : optional emitted RVQ codes (frame-major [n_frames*32]).
    // Returns PCM samples (24 kHz mono, [-1, 1]).
    std::vector<float> generate_audio(const std::vector<std::int32_t>& prompt_ids,
                                      int max_frames,
                                      std::vector<std::int32_t>* out_codes = nullptr);

    const CsmWeights& weights() const noexcept { return weights_; }

private:
    CsmWeights weights_;
};

}  // namespace pie_cuda_driver::model
