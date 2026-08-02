// CSM IModel impl. See csm_model.hpp.

#include "model/csm/csm_model.hpp"

#include <stdexcept>

#include "model/csm/csm_backbone_forward.hpp"

namespace pie_cuda_driver::model {

void CsmModel::body(Workspace&, KvCache&, AttentionWorkspace&,
                    ops::CublasHandle&, const ForwardFn::ForwardInputs&) {
    // CSM does not support the per-step batched text forward. It is driven
    // exclusively through `generate_audio`. A regular ForwardRequest routed
    // here is a wiring error.
    throw std::runtime_error(
        "CsmModel::body: CSM has no per-step text forward; use generate_audio "
        "(pie:core/audio-out). See AUDIO_OUTPUT.md.");
}

std::vector<float> CsmModel::generate_audio(
    const std::vector<std::int32_t>& prompt_ids,
    int max_frames,
    std::vector<std::int32_t>* out_codes) {
    if (prompt_ids.empty()) {
        throw std::runtime_error("CsmModel::generate_audio: empty prompt");
    }
    const CsmBackboneRawWeights bb = weights_.backbone_raw();
    const CsmDepthRawWeights depth = weights_.depth_raw();
    const MimiDecoderRawWeights mimi = weights_.mimi_raw();

    std::vector<float> pcm;
    csm_generate_audio(bb, depth, mimi, prompt_ids.data(),
                       static_cast<int>(prompt_ids.size()), max_frames, pcm,
                       out_codes, /*stream=*/0);
    return pcm;
}

}  // namespace pie_cuda_driver::model
