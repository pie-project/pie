#pragma once

#include <cstdint>
#include <string>
#include <vector>

#include "loader/layout_plan.hpp"

namespace pie_cuda_driver {

enum class RuntimeEncodingKind {
    Dense,
    RuntimeFp8E4M3,
    RuntimeInt8,
    GptqMarlinInt4,
    AwqMarlinInt4,
    Mxfp4E2M1E8M0,
};

enum class RuntimeQuantPolicyKind {
    None,
    NativePacked,
    RoutedDecode,
    EagerBf16,
};

enum class RuntimeProjectionPackKind {
    AttentionQkvRows,
    MlpGateUpRows,
};

struct RuntimeTensorContract {
    std::string name;
    DType dtype = DType::BF16;
    std::vector<std::int64_t> shape;
    RuntimeEncodingKind encoding = RuntimeEncodingKind::Dense;
    TensorLayoutKind layout = TensorLayoutKind::Dense;
    TensorOwnershipKind ownership = TensorOwnershipKind::Owned;
    TensorParallelKind parallel = TensorParallelKind::Replicated;
    std::uint64_t alignment_bytes = 256;
    std::string backing_tensor;
    int view_axis = -1;
    std::int64_t view_start = 0;
    std::int64_t view_length = 0;
    QuantSpec quant;
    RuntimeQuantPolicyKind quant_policy = RuntimeQuantPolicyKind::None;
};

struct RuntimePackedProjectionDecl {
    std::string storage_name;
    TensorLayoutKind storage_layout = TensorLayoutKind::AxisConcatenated;
    TensorLayoutKind view_layout = TensorLayoutKind::View;
    std::uint64_t alignment_bytes = 256;
};

struct RuntimeExpertBankDecl {
    std::string gate_up_name;
    std::string down_name;
    TensorLayoutKind layout = TensorLayoutKind::Grouped;
    std::uint64_t alignment_bytes = 256;
};

// RuntimeABI owns final CUDA tensor names and layout descriptors. Model
// adapters declare semantic groups; this object declares the concrete names
// and layouts that binders/kernels consume.
class RuntimeABI {
public:
    RuntimeTensorContract tensor_contract(
        std::string name,
        DType dtype,
        std::vector<std::int64_t> shape,
        TensorLayoutKind layout = TensorLayoutKind::Dense,
        TensorOwnershipKind ownership = TensorOwnershipKind::Owned,
        TensorParallelKind parallel = TensorParallelKind::Replicated,
        QuantSpec quant = {},
        RuntimeQuantPolicyKind quant_policy =
            RuntimeQuantPolicyKind::None,
        std::string backing_tensor = {}) const;

    RuntimeTensorContract view_contract(
        std::string name,
        DType dtype,
        std::vector<std::int64_t> shape,
        std::string backing_tensor,
        int axis,
        std::int64_t start,
        std::int64_t length,
        TensorParallelKind parallel =
            TensorParallelKind::Replicated) const;

    RuntimePackedProjectionDecl packed_projection(
        RuntimeProjectionPackKind kind,
        const std::string& runtime_base) const;

    RuntimeExpertBankDecl fused_expert_bank(
        const std::string& runtime_base) const;

    std::string quant_scale_inv_name(
        const std::string& runtime_weight_name) const;

    RuntimeEncodingKind encoding_for_quant(
        const QuantSpec& quant,
        DType dtype) const noexcept;
};

const char* runtime_encoding_kind_name(RuntimeEncodingKind kind) noexcept;
const char* runtime_quant_policy_kind_name(
    RuntimeQuantPolicyKind kind) noexcept;
const RuntimeABI& pie_cuda_runtime_abi() noexcept;

}  // namespace pie_cuda_driver
