#pragma once

#include <algorithm>
#include <cstdint>
#include <deque>
#include <string>
#include <utility>
#include <vector>

#include "../../../weight_loader/include/weight_loader.h"
#include "loader/checkpoint_source.hpp"
#include "loader/hf_config.hpp"

namespace pie_cuda_driver {

class RustLoaderInputBuilder {
public:
    pie_weight_loader::PieLoaderCompileInput view() const noexcept
    {
        return pie_weight_loader::PieLoaderCompileInput{
            .version = 1,
            .files = {
                .ptr = files_.data(),
                .len = files_.size(),
            },
            .tensors = {
                .ptr = tensors_.data(),
                .len = tensors_.size(),
            },
            .model = model_,
            .runtime_abi = runtime_abi_,
            .target = target_,
        };
    }

    std::uint32_t intern_string(std::string value)
    {
        strings_.push_back(std::move(value));
        return static_cast<std::uint32_t>(strings_.size() - 1);
    }

    pie_weight_loader::PieLoaderBytes bytes(std::uint32_t id) const noexcept
    {
        const auto& value = strings_[id];
        return pie_weight_loader::PieLoaderBytes{
            .ptr = reinterpret_cast<const std::uint8_t*>(value.data()),
            .len = value.size(),
        };
    }

    std::uint32_t intern_shape(std::vector<std::int64_t> shape)
    {
        shapes_.push_back(std::move(shape));
        return static_cast<std::uint32_t>(shapes_.size() - 1);
    }

    pie_weight_loader::PieLoaderI64Slice shape(std::uint32_t id) const noexcept
    {
        const auto& value = shapes_[id];
        return pie_weight_loader::PieLoaderI64Slice{
            .ptr = value.data(),
            .len = value.size(),
        };
    }

    std::uint32_t intern_u32_slice(std::vector<std::uint32_t> values)
    {
        u32_slices_.push_back(std::move(values));
        return static_cast<std::uint32_t>(u32_slices_.size() - 1);
    }

    pie_weight_loader::PieLoaderU32Slice u32_slice(
        std::uint32_t id) const noexcept
    {
        const auto& value = u32_slices_[id];
        return pie_weight_loader::PieLoaderU32Slice{
            .ptr = value.data(),
            .len = value.size(),
        };
    }

    void set_model(const HfConfig& hf)
    {
        model_type_ = intern_string(hf.model_type);
        quant_method_ = intern_string(hf.quant_method);
        model_ = pie_weight_loader::PieLoaderModelConfigView{
            .model_type = bytes(model_type_),
            .quant_method = bytes(quant_method_),
            .num_hidden_layers = static_cast<std::uint32_t>(
                std::max(hf.num_hidden_layers, 0)),
            .num_experts = static_cast<std::uint32_t>(
                std::max(hf.num_experts, 0)),
            .num_experts_per_tok = static_cast<std::uint32_t>(
                std::max(hf.num_experts_per_tok, 0)),
        };
    }

    void set_target(
        int tp_rank,
        int tp_size,
        std::uint64_t max_tile_bytes,
        std::uint32_t preferred_alignment)
    {
        target_ = pie_weight_loader::PieLoaderBackendTargetView{
            .backend = pie_weight_loader::PieLoaderBackendKind::Cuda,
            .tp_rank = static_cast<std::uint32_t>(std::max(tp_rank, 0)),
            .tp_size = static_cast<std::uint32_t>(std::max(tp_size, 1)),
            .max_tile_bytes = max_tile_bytes,
            .preferred_alignment = preferred_alignment,
            .mxfp4_moe =
                pie_weight_loader::PieLoaderMxfp4MoePolicy::RoutedDecode,
            .native_mxfp4_moe = false,
        };
    }

    void add_file(
        std::uint32_t id,
        std::string path,
        std::uint64_t size_bytes,
        pie_weight_loader::PieLoaderCheckpointFormat format)
    {
        const std::uint32_t path_id = intern_string(std::move(path));
        files_.push_back(pie_weight_loader::PieLoaderCheckpointFileView{
            .id = id,
            .path = bytes(path_id),
            .size_bytes = size_bytes,
            .format = format,
        });
    }

    void add_tensor(
        std::uint32_t id,
        std::string name,
        std::uint32_t file_id,
        std::uint64_t file_offset,
        std::uint64_t span_bytes,
        const TensorInfo& info,
        pie_weight_loader::PieLoaderEncodingKind encoding_kind =
            pie_weight_loader::PieLoaderEncodingKind::Raw,
        pie_weight_loader::PieLoaderQuantScheme quant_scheme =
            pie_weight_loader::PieLoaderQuantScheme::None)
    {
        const std::uint32_t name_id = intern_string(std::move(name));
        const std::uint32_t shape_id = intern_shape(info.shape);
        tensors_.push_back(pie_weight_loader::PieLoaderCheckpointTensorView{
            .id = id,
            .name = bytes(name_id),
            .file_id = file_id,
            .file_offset = file_offset,
            .span_bytes = span_bytes,
            .dtype = dtype_to_rust(info.dtype),
            .encoding_kind = encoding_kind,
            .quant_scheme = quant_scheme,
            .shape = shape(shape_id),
        });
    }

    void set_runtime_abi_name(std::string name, std::uint32_t version)
    {
        runtime_abi_name_ = intern_string(std::move(name));
        runtime_abi_.name = bytes(runtime_abi_name_);
        runtime_abi_.version = version;
        refresh_contract_slice();
    }

    void add_direct_contract(
        std::string output_name,
        std::uint32_t source_tensor_id,
        DType dtype,
        std::vector<std::int64_t> tensor_shape,
        std::uint32_t alignment,
        int shard_axis = -1,
        std::vector<std::uint32_t> metadata_tensor_ids = {})
    {
        const std::uint32_t name_id = intern_string(std::move(output_name));
        const std::uint32_t shape_id = intern_shape(std::move(tensor_shape));
        const std::uint32_t metadata_id =
            intern_u32_slice(std::move(metadata_tensor_ids));
        contracts_.push_back(
            pie_weight_loader::PieLoaderRuntimeTensorContractView{
                .output_name = bytes(name_id),
                .source_kind =
                    pie_weight_loader::PieLoaderRuntimeSourceKind::DirectTensor,
                .source_tensor_id = source_tensor_id,
                .source_tensor_ids = {},
                .byte_spans = {},
                .metadata_tensor_ids = u32_slice(metadata_id),
                .source_contract_id = UINT32_MAX,
                .semantic_role =
                    pie_weight_loader::PieLoaderSemanticRole::DirectTensor,
                .layer = 0,
                .has_layer = false,
                .expert = 0,
                .has_expert = false,
                .axis = -1,
                .start = 0,
                .length = 0,
                .dtype = dtype_to_rust(dtype),
                .encoding_kind = pie_weight_loader::PieLoaderEncodingKind::Raw,
                .quant_scheme = pie_weight_loader::PieLoaderQuantScheme::None,
                .shape = shape(shape_id),
                .alignment = alignment,
                .shard_axis = shard_axis,
            });
        refresh_contract_slice();
    }

    void add_byte_span_contract(
        std::string output_name,
        std::vector<pie_weight_loader::PieLoaderRuntimeByteSpanView> spans,
        DType dtype,
        std::vector<std::int64_t> tensor_shape,
        std::uint32_t alignment)
    {
        byte_spans_.push_back(std::move(spans));
        const std::uint32_t name_id = intern_string(std::move(output_name));
        const std::uint32_t shape_id = intern_shape(std::move(tensor_shape));
        const auto& stored_spans = byte_spans_.back();
        contracts_.push_back(
            pie_weight_loader::PieLoaderRuntimeTensorContractView{
                .output_name = bytes(name_id),
                .source_kind =
                    pie_weight_loader::PieLoaderRuntimeSourceKind::ByteSpans,
                .source_tensor_id = UINT32_MAX,
                .source_tensor_ids = {},
                .byte_spans = {
                    .ptr = stored_spans.data(),
                    .len = stored_spans.size(),
                },
                .metadata_tensor_ids = {},
                .source_contract_id = UINT32_MAX,
                .semantic_role =
                    pie_weight_loader::PieLoaderSemanticRole::DirectTensor,
                .layer = 0,
                .has_layer = false,
                .expert = 0,
                .has_expert = false,
                .axis = -1,
                .start = 0,
                .length = 0,
                .dtype = dtype_to_rust(dtype),
                .encoding_kind = pie_weight_loader::PieLoaderEncodingKind::Raw,
                .quant_scheme = pie_weight_loader::PieLoaderQuantScheme::None,
                .shape = shape(shape_id),
                .alignment = alignment,
                .shard_axis = -1,
            });
        refresh_contract_slice();
    }

    void add_join_contract(
        std::string output_name,
        std::vector<std::uint32_t> source_tensor_ids,
        int axis,
        DType dtype,
        std::vector<std::int64_t> tensor_shape,
        std::uint32_t alignment)
    {
        const std::uint32_t name_id = intern_string(std::move(output_name));
        const std::uint32_t sources_id =
            intern_u32_slice(std::move(source_tensor_ids));
        const std::uint32_t shape_id = intern_shape(std::move(tensor_shape));
        contracts_.push_back(
            pie_weight_loader::PieLoaderRuntimeTensorContractView{
                .output_name = bytes(name_id),
                .source_kind =
                    pie_weight_loader::PieLoaderRuntimeSourceKind::Join,
                .source_tensor_id = UINT32_MAX,
                .source_tensor_ids = u32_slice(sources_id),
                .byte_spans = {},
                .metadata_tensor_ids = {},
                .source_contract_id = UINT32_MAX,
                .semantic_role =
                    pie_weight_loader::PieLoaderSemanticRole::DirectTensor,
                .layer = 0,
                .has_layer = false,
                .expert = 0,
                .has_expert = false,
                .axis = axis,
                .start = 0,
                .length = 0,
                .dtype = dtype_to_rust(dtype),
                .encoding_kind = pie_weight_loader::PieLoaderEncodingKind::Raw,
                .quant_scheme = pie_weight_loader::PieLoaderQuantScheme::None,
                .shape = shape(shape_id),
                .alignment = alignment,
                .shard_axis = -1,
            });
        refresh_contract_slice();
    }

    void add_select_contract(
        std::string output_name,
        std::uint32_t source_contract_id,
        int axis,
        std::int64_t start,
        std::int64_t length,
        DType dtype,
        std::vector<std::int64_t> tensor_shape,
        std::uint32_t alignment)
    {
        const std::uint32_t name_id = intern_string(std::move(output_name));
        const std::uint32_t shape_id = intern_shape(std::move(tensor_shape));
        contracts_.push_back(
            pie_weight_loader::PieLoaderRuntimeTensorContractView{
                .output_name = bytes(name_id),
                .source_kind =
                    pie_weight_loader::PieLoaderRuntimeSourceKind::Select,
                .source_tensor_id = UINT32_MAX,
                .source_tensor_ids = {},
                .byte_spans = {},
                .metadata_tensor_ids = {},
                .source_contract_id = source_contract_id,
                .semantic_role =
                    pie_weight_loader::PieLoaderSemanticRole::DirectTensor,
                .layer = 0,
                .has_layer = false,
                .expert = 0,
                .has_expert = false,
                .axis = axis,
                .start = start,
                .length = length,
                .dtype = dtype_to_rust(dtype),
                .encoding_kind = pie_weight_loader::PieLoaderEncodingKind::Raw,
                .quant_scheme = pie_weight_loader::PieLoaderQuantScheme::None,
                .shape = shape(shape_id),
                .alignment = alignment,
                .shard_axis = -1,
            });
        refresh_contract_slice();
    }

private:
    static pie_weight_loader::PieLoaderDType dtype_to_rust(DType dtype)
    {
        switch (dtype) {
        case DType::BF16: return pie_weight_loader::PieLoaderDType::BF16;
        case DType::FP16: return pie_weight_loader::PieLoaderDType::F16;
        case DType::FP32: return pie_weight_loader::PieLoaderDType::F32;
        case DType::INT8: return pie_weight_loader::PieLoaderDType::I8;
        case DType::INT32: return pie_weight_loader::PieLoaderDType::I32;
        case DType::UINT8: return pie_weight_loader::PieLoaderDType::U8;
        case DType::FP8_E4M3:
            return pie_weight_loader::PieLoaderDType::F8E4M3;
        case DType::FP8_E5M2:
            return pie_weight_loader::PieLoaderDType::F8E5M2;
        case DType::INT64:
        case DType::INT4_PACKED:
            return pie_weight_loader::PieLoaderDType::U8;
        }
        return pie_weight_loader::PieLoaderDType::U8;
    }

    void refresh_contract_slice() noexcept
    {
        runtime_abi_.tensors =
            pie_weight_loader::PieLoaderRuntimeTensorContractSlice{
                .ptr = contracts_.data(),
                .len = contracts_.size(),
            };
    }

    std::deque<std::string> strings_;
    std::deque<std::vector<std::int64_t>> shapes_;
    std::deque<std::vector<std::uint32_t>> u32_slices_;
    std::deque<std::vector<pie_weight_loader::PieLoaderRuntimeByteSpanView>>
        byte_spans_;
    std::vector<pie_weight_loader::PieLoaderCheckpointFileView> files_;
    std::vector<pie_weight_loader::PieLoaderCheckpointTensorView> tensors_;
    std::vector<pie_weight_loader::PieLoaderRuntimeTensorContractView> contracts_;

    std::uint32_t model_type_ = 0;
    std::uint32_t quant_method_ = 0;
    std::uint32_t runtime_abi_name_ = 0;
    pie_weight_loader::PieLoaderModelConfigView model_{};
    pie_weight_loader::PieLoaderRuntimeAbiView runtime_abi_{};
    pie_weight_loader::PieLoaderBackendTargetView target_{};
};

}  // namespace pie_cuda_driver
