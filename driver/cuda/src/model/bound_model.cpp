#include "model/bound_model.hpp"

#include <string>

namespace pie_cuda_driver::model {

std::size_t BoundCudaModel::num_layers() const noexcept {
    switch (kind) {
    case Kind::LlamaLike: return llama.layers.size();
    case Kind::Gemma: return gemma.layers.size();
    case Kind::Gemma4: return gemma4.layers.size();
    case Kind::Gemma3n: return gemma3n.layers.size();
    case Kind::Mixtral: return mixtral.layers.size();
    case Kind::Qwen3_5: return qwen3_5.layers.size();
    case Kind::Qwen3_5Moe: return qwen3_5_moe.layers.size();
    case Kind::Kimi: return kimi.layers.size();
    }
    return 0;
}

BoundCudaModel bind_cuda_model(const LoadedModel& engine, bool verbose) {
    const std::string& mt = engine.hf_config().model_type;
    BoundCudaModel bound;

    const bool is_gemma4 =
        (mt == "gemma4" || mt == "gemma4_text");
    const bool is_gemma3n =
        (mt == "gemma3n" || mt == "gemma3n_text");
    const bool is_gpt_oss = (mt == "gpt_oss");
    const bool is_qwen3_5 =
        (mt == "qwen3_5" || mt == "qwen3_5_text");
    const bool is_qwen3_5_moe =
        (mt == "qwen3_5_moe" || mt == "qwen3_5_moe_text" || mt == "qwen3_moe");

    if (mt == "kimi_k2") {
        bound.kind = BoundCudaModel::Kind::Kimi;
        bound.kimi = bind_kimi(engine);
    } else if (mt == "phi3") {
        bound.kind = BoundCudaModel::Kind::LlamaLike;
        bound.llama = bind_phi3(engine);
    } else if (mt == "olmo2" || mt == "olmo3") {
        bound.kind = BoundCudaModel::Kind::LlamaLike;
        bound.llama = bind_olmo3(engine);
    } else if (mt == "mistral3" || mt == "ministral3") {
        bound.kind = BoundCudaModel::Kind::LlamaLike;
        bound.llama = bind_mistral3(engine);
    } else if (mt == "gemma2") {
        bound.kind = BoundCudaModel::Kind::Gemma;
        bound.gemma = bind_gemma2(engine);
    } else if (mt == "gemma3" || mt == "gemma3_text") {
        bound.kind = BoundCudaModel::Kind::Gemma;
        bound.gemma = bind_gemma3(engine);
    } else if (is_gemma4) {
        bound.kind = BoundCudaModel::Kind::Gemma4;
        bound.gemma4 = bind_gemma4(engine);
    } else if (is_gemma3n) {
        bound.kind = BoundCudaModel::Kind::Gemma3n;
        bound.gemma3n = bind_gemma3n(engine);
    } else if (is_gpt_oss) {
        bound.kind = BoundCudaModel::Kind::Mixtral;
        bound.mixtral = bind_gpt_oss(engine);
    } else if (mt == "mixtral") {
        bound.kind = BoundCudaModel::Kind::Mixtral;
        bound.mixtral = bind_mixtral(engine);
    } else if (is_qwen3_5) {
        bound.kind = BoundCudaModel::Kind::Qwen3_5;
        bound.qwen3_5 = bind_qwen3_5(engine);
    } else if (is_qwen3_5_moe) {
        bound.kind = BoundCudaModel::Kind::Qwen3_5Moe;
        bound.qwen3_5_moe = bind_qwen3_5_moe(engine);
    } else {
        bound.kind = BoundCudaModel::Kind::LlamaLike;
        bound.llama = bind_llama_like(engine, verbose);
    }

    return bound;
}

}  // namespace pie_cuda_driver::model
