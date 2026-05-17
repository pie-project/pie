#include "loader/runtime_abi.hpp"

#include <stdexcept>
#include <utility>

namespace pie_cuda_driver {

RuntimeTensorContract RuntimeABI::tensor_contract(
    std::string name,
    DType dtype,
    std::vector<std::int64_t> shape,
    TensorLayoutKind layout,
    TensorOwnershipKind ownership,
    TensorParallelKind parallel,
    QuantSpec quant,
    RuntimeQuantPolicyKind quant_policy,
    std::string backing_tensor) const
{
    RuntimeTensorContract contract;
    contract.name = std::move(name);
    contract.dtype = dtype;
    contract.shape = std::move(shape);
    contract.encoding = encoding_for_quant(quant, dtype);
    contract.layout = layout;
    contract.ownership = ownership;
    contract.parallel = parallel;
    contract.backing_tensor = std::move(backing_tensor);
    contract.quant = std::move(quant);
    contract.quant_policy = quant_policy;
    return contract;
}

RuntimeTensorContract RuntimeABI::view_contract(
    std::string name,
    DType dtype,
    std::vector<std::int64_t> shape,
    std::string backing_tensor,
    TensorParallelKind parallel) const
{
    return tensor_contract(
        std::move(name),
        dtype,
        std::move(shape),
        TensorLayoutKind::View,
        TensorOwnershipKind::BorrowedView,
        parallel,
        {},
        RuntimeQuantPolicyKind::None,
        std::move(backing_tensor));
}

RuntimePackedProjectionDecl RuntimeABI::packed_projection(
    RuntimeProjectionPackKind kind,
    const std::string& runtime_base) const
{
    switch (kind) {
    case RuntimeProjectionPackKind::AttentionQkvRows:
        return RuntimePackedProjectionDecl{
            .storage_name = runtime_base + ".qkv_proj.fused.weight",
        };
    case RuntimeProjectionPackKind::MlpGateUpRows:
        return RuntimePackedProjectionDecl{
            .storage_name = runtime_base + ".gate_up_proj.fused.weight",
        };
    }
    throw std::runtime_error("runtime ABI: unknown packed projection kind");
}

RuntimeExpertBankDecl RuntimeABI::fused_expert_bank(
    const std::string& runtime_base) const
{
    return RuntimeExpertBankDecl{
        .gate_up_name = runtime_base + ".experts.gate_up_proj",
        .down_name = runtime_base + ".experts.down_proj",
    };
}

std::string RuntimeABI::quant_scale_inv_name(
    const std::string& runtime_weight_name) const
{
    return runtime_weight_name + "_scale_inv";
}

RuntimeEncodingKind RuntimeABI::encoding_for_quant(
    const QuantSpec& quant,
    DType dtype) const noexcept
{
    switch (quant.format) {
    case QuantFormat::RuntimeFp8E4M3:
        return RuntimeEncodingKind::RuntimeFp8E4M3;
    case QuantFormat::RuntimeInt8:
        return RuntimeEncodingKind::RuntimeInt8;
    case QuantFormat::GptqInt4:
        return RuntimeEncodingKind::GptqMarlinInt4;
    case QuantFormat::AwqInt4:
        return RuntimeEncodingKind::AwqMarlinInt4;
    case QuantFormat::Mxfp4E2M1E8M0:
        return RuntimeEncodingKind::Mxfp4E2M1E8M0;
    case QuantFormat::None:
    case QuantFormat::CompressedFp8E4M3:
    case QuantFormat::CompressedInt8:
        break;
    }
    if (dtype == DType::FP8_E4M3) {
        return RuntimeEncodingKind::RuntimeFp8E4M3;
    }
    if (dtype == DType::INT8) {
        return RuntimeEncodingKind::RuntimeInt8;
    }
    return RuntimeEncodingKind::Dense;
}

const char* runtime_encoding_kind_name(RuntimeEncodingKind kind) noexcept
{
    switch (kind) {
    case RuntimeEncodingKind::Dense: return "Dense";
    case RuntimeEncodingKind::RuntimeFp8E4M3: return "RuntimeFp8E4M3";
    case RuntimeEncodingKind::RuntimeInt8: return "RuntimeInt8";
    case RuntimeEncodingKind::GptqMarlinInt4: return "GptqMarlinInt4";
    case RuntimeEncodingKind::AwqMarlinInt4: return "AwqMarlinInt4";
    case RuntimeEncodingKind::Mxfp4E2M1E8M0: return "Mxfp4E2M1E8M0";
    }
    return "?";
}

const char* runtime_quant_policy_kind_name(
    RuntimeQuantPolicyKind kind) noexcept
{
    switch (kind) {
    case RuntimeQuantPolicyKind::None: return "None";
    case RuntimeQuantPolicyKind::NativePacked: return "NativePacked";
    case RuntimeQuantPolicyKind::RoutedDecode: return "RoutedDecode";
    case RuntimeQuantPolicyKind::EagerBf16: return "EagerBf16";
    }
    return "?";
}

const RuntimeABI& pie_cuda_runtime_abi() noexcept
{
    static const RuntimeABI abi;
    return abi;
}

}  // namespace pie_cuda_driver
