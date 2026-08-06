#include "model/gemma4/declared_forward.hpp"

#include <stdexcept>
#include <string>
#include <string_view>

namespace pie_cuda_driver::model {

namespace {

// The launcher registry — every kernel a gemma-4 class trace may STATE,
// one enum value per symbol. Deliberately EXHAUSTIVE against the traced
// decode plan: `gemma4_validate_stated_kernels` walks the plan at load
// and a symbol outside this list is a model-load failure, so this list
// and `family::gemma4_cuda` are two spellings of one vocabulary.
enum class G4Kernel {
    QkvPackedPost,
    QkRmsnormRopeRounded,
    RopeQOnly,
    RopeQOnlyPartial,
    RmsnormNoScale,
    WriteKvToPages,
    AttnFlashinferDecode,
    AttnFlashinferPrefill,
    GegluTanh,
    ChunkedGegluTanh,
    NormResidualScaleNorm,
    NormResidualAdd,
    ScalarMul,
    TransposeNldToLnd,
    LogitSoftcap,
};

G4Kernel resolve_g4_kernel(std::string_view k) {
    if (k == "launch_qkv_packed_qk_norm_rope_vnorm_write_kv_bf16")
        return G4Kernel::QkvPackedPost;
    if (k == "launch_qk_rmsnorm_rope_bf16_rounded")
        return G4Kernel::QkRmsnormRopeRounded;
    if (k == "launch_rope_bf16") return G4Kernel::RopeQOnly;
    if (k == "launch_rope_partial_bf16") return G4Kernel::RopeQOnlyPartial;
    if (k == "launch_rmsnorm_no_scale_bf16") return G4Kernel::RmsnormNoScale;
    if (k == "launch_write_kv_to_pages") return G4Kernel::WriteKvToPages;
    if (k == "dispatch_attention_flashinfer_decode")
        return G4Kernel::AttnFlashinferDecode;
    if (k == "dispatch_attention_flashinfer_prefill_bf16")
        return G4Kernel::AttnFlashinferPrefill;
    if (k == "launch_geglu_tanh_bf16") return G4Kernel::GegluTanh;
    if (k == "launch_chunked_geglu_tanh_bf16") return G4Kernel::ChunkedGegluTanh;
    if (k == "launch_rmsnorm_residual_add_scale_rmsnorm_bf16")
        return G4Kernel::NormResidualScaleNorm;
    if (k == "launch_rmsnorm_residual_add_bf16") return G4Kernel::NormResidualAdd;
    if (k == "launch_scalar_mul_bf16") return G4Kernel::ScalarMul;
    if (k == "launch_transpose_bf16_nld_to_lnd")
        return G4Kernel::TransposeNldToLnd;
    if (k == "launch_logit_softcap_bf16") return G4Kernel::LogitSoftcap;
    throw std::runtime_error(
        "declared gemma4: stated kernel '" + std::string(k) +
        "' is not in this executor's registry (the trace and the driver "
        "drifted)");
}

}  // namespace

void gemma4_validate_stated_kernels(const pie_forward::ForwardPlan& plan) {
    const std::size_t n = plan.op_count();
    for (std::size_t i = 0; i < n; ++i) {
        const auto& op = plan.op(i);
        if (op.kind != pie_forward::PieForwardOpKind::Launch) continue;
        (void)resolve_g4_kernel(plan.weight_name(op));
    }
}

}  // namespace pie_cuda_driver::model
