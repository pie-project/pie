#include "loader/model_schema.hpp"

#include <algorithm>
#include <cstring>
#include <utility>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>

#include <pie_driver_common/tensor_names.hpp>

#include "loader/safetensors.hpp"

namespace pie_cuda_driver {

namespace {

bool ends_with(const std::string& s, const char* suffix) {
    const auto n = std::char_traits<char>::length(suffix);
    return s.size() >= n && s.compare(s.size() - n, n, suffix) == 0;
}

bool is_qwen3_5_moe_arch(const std::string& mt) {
    return mt == "qwen3_5_moe" || mt == "qwen3_5_moe_text"
        || mt == "qwen3_moe";
}

bool supports_dense_llama_packed_load(const HfConfig& hf, const Config& boot_cfg) {
    if (!hf.quant_method.empty()) return false;
    if (!boot_cfg.model.runtime_quant.empty()) return false;

    const std::string& mt = hf.model_type;
    return mt == "qwen3"
        || mt == "qwen2"
        || mt == "llama" || mt == "llama3"
        || mt == "mistral" || mt == "mistral3" || mt == "ministral3"
        || mt == "olmo2" || mt == "olmo3";
}

bool can_pack_2d_bf16_group(
    const TensorMetadataSource& loader,
    const std::vector<std::string>& raw_names)
{
    if (raw_names.empty()) return false;
    std::int64_t cols = -1;
    for (const auto& raw : raw_names) {
        if (!loader.contains(raw)) return false;
        const auto& info = loader.info(raw);
        if (info.dtype != DType::BF16 || info.shape.size() != 2) return false;
        if (cols < 0) {
            cols = info.shape[1];
        } else if (info.shape[1] != cols) {
            return false;
        }
    }
    return true;
}

TensorParallelKind parallel_kind_from_axis(int axis) {
    if (axis == 0) return TensorParallelKind::Column;
    if (axis == 1) return TensorParallelKind::Row;
    return TensorParallelKind::Replicated;
}

std::vector<std::int64_t> sharded_shape(
    std::vector<std::int64_t> shape,
    int shard_axis,
    int tp_size,
    const std::string& name)
{
    if (tp_size <= 1 || shard_axis < 0) return shape;
    if (shard_axis >= static_cast<int>(shape.size())) {
        throw std::runtime_error(
            "load schema: shard axis out of range for '" + name + "'");
    }
    if (shape[shard_axis] % tp_size != 0) {
        throw std::runtime_error(
            "load schema: dimension " + std::to_string(shard_axis) +
            " of '" + name + "' is not divisible by tp_size=" +
            std::to_string(tp_size));
    }
    shape[shard_axis] /= tp_size;
    return shape;
}

void register_tensor_spec(
    LoadPlan& plan,
    std::string name,
    DType dtype,
    std::vector<std::int64_t> shape,
    TensorLayoutKind layout,
    TensorOwnershipKind ownership,
    TensorParallelKind parallel,
    std::string backing_tensor = {},
    QuantSpec quant = {})
{
    TensorSpec spec;
    spec.name = std::move(name);
    spec.dtype = dtype;
    spec.shape = std::move(shape);
    spec.layout = layout;
    spec.ownership = ownership;
    spec.parallel = parallel;
    spec.quant = std::move(quant);
    spec.backing_tensor = std::move(backing_tensor);
    const std::string key = spec.name;
    auto [it, inserted] = plan.tensors.emplace(key, std::move(spec));
    if (!inserted) {
        throw std::runtime_error(
            "load schema: duplicate runtime tensor spec '" + it->first + "'");
    }

    if (it->second.ownership == TensorOwnershipKind::Owned ||
        it->second.ownership == TensorOwnershipKind::Temporary) {
        std::uint64_t numel = 1;
        for (const auto dim : it->second.shape) {
            numel *= static_cast<std::uint64_t>(dim);
        }
        const std::uint64_t bytes =
            numel * static_cast<std::uint64_t>(dtype_bytes(it->second.dtype));
        if (it->second.ownership == TensorOwnershipKind::Owned) {
            plan.memory.persistent_bytes += bytes;
        } else {
            plan.memory.max_temporary_bytes =
                std::max(plan.memory.max_temporary_bytes, bytes);
        }
        plan.memory.estimated_peak_bytes = std::max(
            plan.memory.estimated_peak_bytes,
            plan.memory.persistent_bytes + plan.memory.max_temporary_bytes);
    }
}

void estimate_temporary_bytes(LoadPlan& plan, std::uint64_t bytes) {
    plan.memory.max_temporary_bytes =
        std::max(plan.memory.max_temporary_bytes, bytes);
    plan.memory.estimated_peak_bytes = std::max(
        plan.memory.estimated_peak_bytes,
        plan.memory.persistent_bytes + plan.memory.max_temporary_bytes);
}

std::uint64_t tensor_nbytes(
    DType dtype,
    const std::vector<std::int64_t>& shape);

bool normalizes_to_bf16(DType dtype) noexcept {
    return dtype == DType::FP16 || dtype == DType::FP32;
}

void add_owned_producer(
    LoadPlan& plan,
    LoadOp producer,
    const std::string& output_name,
    DType source_dtype,
    const std::vector<std::int64_t>& shape,
    TensorLayoutKind layout,
    TensorParallelKind parallel)
{
    if (!normalizes_to_bf16(source_dtype) || ends_with(output_name, "_scale_inv")) {
        set_load_op_output(producer, output_name);
        plan.ops.push_back(std::move(producer));
        register_tensor_spec(
            plan, output_name, source_dtype, shape, layout,
            TensorOwnershipKind::Owned, parallel);
        return;
    }

    const std::string tmp_name = output_name + ".__dtype_source";
    set_load_op_output(producer, tmp_name);
    plan.ops.push_back(std::move(producer));

    register_tensor_spec(
        plan, tmp_name, source_dtype, shape, layout,
        TensorOwnershipKind::Temporary, parallel);
    register_tensor_spec(
        plan, output_name, DType::BF16, shape, layout,
        TensorOwnershipKind::Owned, parallel);

    plan.ops.push_back(make_tensor_op(
        LoadOpKind::Cast, output_name, {tmp_name}));
    plan.ops.push_back(make_tensor_op(
        LoadOpKind::Drop, tmp_name + ".__drop", {tmp_name}));

    estimate_temporary_bytes(plan, tensor_nbytes(source_dtype, shape));
}

bool try_add_packed_axis_group(
    LoadPlan& plan,
    const TensorMetadataSource& loader,
    const LogicalTensorGroup& group,
    std::unordered_set<std::string>& consumed_raw,
    int tp_size)
{
    const bool is_qkv = group.kind == LogicalTensorGroupKind::PackedQkv;
    const bool is_gate_up = group.kind == LogicalTensorGroupKind::PackedGateUp;
    if (!is_qkv && !is_gate_up) return false;
    const std::size_t expected = is_qkv ? 3 : 2;
    if (group.raw_names.size() != expected ||
        group.runtime_names.size() != expected) {
        throw std::runtime_error(
            "load schema: packed logical group has wrong arity at '" +
            group.runtime_base + "'");
    }
    for (const auto& raw : group.raw_names) {
        if (consumed_raw.contains(raw)) return true;
    }
    if (!can_pack_2d_bf16_group(loader, group.raw_names)) return false;

    const std::string packed_name = is_qkv
        ? group.runtime_base + ".qkv_proj.fused.weight"
        : group.runtime_base + ".gate_up_proj.fused.weight";
    std::vector<TensorSourceRef> sources;
    sources.reserve(expected);
    for (std::size_t i = 0; i < expected; ++i) {
        sources.push_back({group.raw_names[i], group.runtime_names[i]});
    }

    plan.ops.push_back(make_axis_concat_op(
        packed_name, /*shard_axis=*/0, std::move(sources)));
    ++plan.axis_concat_groups;

    std::int64_t rows = 0;
    std::int64_t cols = -1;
    for (std::size_t i = 0; i < expected; ++i) {
        const auto& info = loader.info(group.raw_names[i]);
        if (cols < 0) cols = info.shape[1];
        rows += info.shape[0] / tp_size;
    }

    register_tensor_spec(
        plan, packed_name, loader.info(group.raw_names[0]).dtype,
        {rows, cols}, TensorLayoutKind::AxisConcatenated,
        TensorOwnershipKind::Owned, TensorParallelKind::Column);
    for (std::size_t i = 0; i < expected; ++i) {
        const auto& info = loader.info(group.raw_names[i]);
        register_tensor_spec(
            plan, group.runtime_names[i], info.dtype,
            {info.shape[0] / tp_size, info.shape[1]},
            TensorLayoutKind::View, TensorOwnershipKind::BorrowedView,
            TensorParallelKind::Column, packed_name);
        consumed_raw.insert(group.raw_names[i]);
    }
    return true;
}

const std::string& group_runtime_name_for_role(
    const LogicalTensorGroup& group,
    LogicalTensorRole role)
{
    for (std::size_t i = 0; i < group.runtime_roles.size(); ++i) {
        if (group.runtime_roles[i] == role && i < group.runtime_names.size()) {
            return group.runtime_names[i];
        }
    }
    throw std::runtime_error(
        "load schema: logical group '" + group.runtime_base +
        "' does not declare expected runtime role");
}

bool try_add_row_range_split_group(
    LoadPlan& plan,
    const HfConfig& hf,
    const TensorMetadataSource& loader,
    const LogicalTensorGroup& group,
    std::unordered_set<std::string>& consumed_raw,
    int tp_size)
{
    if (group.kind != LogicalTensorGroupKind::RowRangeSplit) return false;
    if (group.raw_names.size() != 1 || group.raw_roles.size() != 1) {
        throw std::runtime_error(
            "load schema: row-range split group has wrong source arity at '" +
            group.runtime_base + "'");
    }
    const std::string& raw_name = group.raw_names[0];
    if (consumed_raw.contains(raw_name)) return true;
    const TensorInfo& info = loader.info(raw_name);
    const LogicalTensorRole fused_role = group.raw_roles[0];

    struct SplitPart {
        LogicalTensorRole role;
        std::int64_t offset;
        std::int64_t rows;
    };
    std::vector<SplitPart> parts;
    if (fused_role == LogicalTensorRole::AttentionQkv) {
        const std::int64_t Hq =
            static_cast<std::int64_t>(hf.num_attention_heads) * hf.head_dim;
        const std::int64_t Hk =
            static_cast<std::int64_t>(hf.num_key_value_heads) * hf.head_dim;
        parts = {
            {LogicalTensorRole::AttentionQ, 0, Hq},
            {LogicalTensorRole::AttentionK, Hq, Hk},
            {LogicalTensorRole::AttentionV, Hq + Hk, Hk},
        };
    } else if (fused_role == LogicalTensorRole::MlpGateUp) {
        const std::int64_t I = hf.intermediate_size;
        parts = {
            {LogicalTensorRole::MlpGate, 0, I},
            {LogicalTensorRole::MlpUp, I, I},
        };
    } else {
        return false;
    }

    if (info.shape.size() != 2) {
        throw std::runtime_error(
            "load schema: row-range split expects 2-D source at '" +
            raw_name + "'");
    }
    for (const auto& part : parts) {
        if (part.rows % tp_size != 0) {
            throw std::runtime_error(
                "load schema: row-range split output is not divisible by "
                "tp_size at '" + raw_name + "'");
        }
        const std::string& output_name =
            group_runtime_name_for_role(group, part.role);
        LoadOp op = make_row_range_shard_op(
            /*output_name=*/{}, raw_name, part.offset, part.rows);
        add_owned_producer(
            plan, std::move(op), output_name, info.dtype,
            {part.rows / tp_size, info.shape[1]},
            TensorLayoutKind::Dense, TensorParallelKind::Column);
    }

    consumed_raw.insert(raw_name);
    return true;
}

void add_copy(LoadPlan& plan, const std::string& raw_name,
              const std::string& output_name, const TensorInfo& info,
              int shard_axis, int tp_size)
{
    LoadOp op = make_raw_load_op(
        LoadOpKind::Copy, /*output_name=*/{}, raw_name, shard_axis);
    add_owned_producer(
        plan, std::move(op), output_name, info.dtype,
        sharded_shape(info.shape, shard_axis, tp_size, output_name),
        TensorLayoutKind::Dense, parallel_kind_from_axis(shard_axis));
}

bool runtime_quant_model_supported(const std::string& mt) {
    return mt == "qwen3"
        || mt == "qwen2"
        || mt == "llama" || mt == "llama3"
        || mt == "mistral"
        || mt == "qwen3_5" || mt == "qwen3_5_text";
}

bool runtime_quantizable_role(LogicalTensorRole role) {
    switch (role) {
    case LogicalTensorRole::AttentionQ:
    case LogicalTensorRole::AttentionK:
    case LogicalTensorRole::AttentionV:
    case LogicalTensorRole::AttentionO:
    case LogicalTensorRole::MlpGate:
    case LogicalTensorRole::MlpUp:
    case LogicalTensorRole::MlpDown:
        return true;
    default:
        return false;
    }
}

bool runtime_quant_enabled_for_plan(
    const HfConfig& hf,
    const Config& boot_cfg,
    bool fp8_native)
{
    const auto& mode = boot_cfg.model.runtime_quant;
    if (mode.empty()) return false;
    const bool is_fp8 = (mode == "fp8");
    const bool is_int8 = (mode == "int8");
    if (!is_fp8 && !is_int8) {
        throw std::runtime_error(
            "load schema: unsupported runtime_quant '" + mode +
            "'. Currently supported: 'fp8' or 'int8'.");
    }
    if (!runtime_quant_model_supported(hf.model_type)) {
        throw std::runtime_error(
            "load schema: runtime_quant=" + mode + " is currently wired "
            "for {qwen2, qwen3, qwen3_5, qwen3_5_text, llama, llama3, "
            "mistral} (got '" + hf.model_type + "').");
    }
    if (!hf.quant_method.empty()) return false;
    if (is_fp8 && !fp8_native) return false;
    return true;
}

void add_runtime_quantized_copy(
    LoadPlan& plan,
    const LogicalTensor& logical,
    const TensorInfo& info,
    const Config& boot_cfg,
    int shard_axis,
    int tp_size)
{
    if (info.shape.size() != 2) {
        throw std::runtime_error(
            "load schema: runtime quant source is not 2-D: " +
            logical.runtime_name);
    }

    const std::string tmp_name =
        logical.runtime_name + ".__runtime_quant_source";
    plan.ops.push_back(make_raw_load_op(
        LoadOpKind::Copy, tmp_name, logical.raw_name, shard_axis));

    auto final_shape =
        sharded_shape(info.shape, shard_axis, tp_size, logical.runtime_name);
    const bool is_int8 = boot_cfg.model.runtime_quant == "int8";
    const DType q_dtype = is_int8 ? DType::INT8 : DType::FP8_E4M3;
    const QuantFormat q_format = is_int8
        ? QuantFormat::RuntimeInt8
        : QuantFormat::RuntimeFp8E4M3;
    const std::string scale_name = logical.runtime_name + "_scale_inv";
    const TensorParallelKind parallel = parallel_kind_from_axis(shard_axis);

    register_tensor_spec(
        plan, tmp_name, info.dtype, final_shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
        parallel);

    QuantSpec quant;
    quant.format = q_format;
    quant.granularity = QuantGranularity::PerChannel;
    quant.group_size = 0;
    quant.channel_axis = 0;
    quant.scale_tensor = scale_name;

    register_tensor_spec(
        plan, logical.runtime_name, q_dtype, final_shape,
        TensorLayoutKind::QuantPacked, TensorOwnershipKind::Owned,
        parallel, /*backing_tensor=*/{}, std::move(quant));
    register_tensor_spec(
        plan, scale_name, DType::FP32, {final_shape[0]},
        TensorLayoutKind::Dense, TensorOwnershipKind::Owned,
        parallel);

    plan.ops.push_back(make_tensor_op(
        LoadOpKind::QuantizeRuntime, logical.runtime_name,
        {tmp_name}, scale_name));
    plan.ops.push_back(make_tensor_op(
        LoadOpKind::BindMetadata, logical.runtime_name));
    plan.ops.push_back(make_tensor_op(
        LoadOpKind::Drop, tmp_name + ".__drop", {tmp_name}));

    std::uint64_t source_numel = 1;
    for (const auto dim : final_shape) {
        source_numel *= static_cast<std::uint64_t>(dim);
    }
    const std::uint64_t source_bytes =
        source_numel * static_cast<std::uint64_t>(dtype_bytes(info.dtype));
    const std::uint64_t bf16_scratch_bytes =
        (info.dtype == DType::BF16)
            ? 0
            : source_numel * static_cast<std::uint64_t>(dtype_bytes(DType::BF16));
    estimate_temporary_bytes(plan, source_bytes + bf16_scratch_bytes);
}

bool is_compressed_fp8_scale_companion(
    const HfConfig& hf,
    const TensorMetadataSource& loader,
    const LogicalTensor& logical)
{
    if (hf.quant_method != "compressed-tensors") return false;
    constexpr const char* scale_suffix = "_scale";
    if (!ends_with(logical.raw_name, scale_suffix)) return false;
    const std::string raw_weight =
        logical.raw_name.substr(0, logical.raw_name.size() -
                                      std::strlen(scale_suffix));
    return loader.contains(raw_weight) &&
           loader.info(raw_weight).dtype == DType::FP8_E4M3;
}

bool is_compressed_quant_companion(
    const HfConfig& hf,
    const TensorMetadataSource& loader,
    const LogicalTensor& logical)
{
    if (is_compressed_fp8_scale_companion(hf, loader, logical)) {
        return true;
    }
    if (hf.quant_method != "compressed-tensors") return false;
    constexpr const char* scale_suffix = "_scale";
    constexpr const char* zero_suffix = "_zero_point";
    const char* matched = nullptr;
    if (ends_with(logical.raw_name, scale_suffix)) matched = scale_suffix;
    if (ends_with(logical.raw_name, zero_suffix)) matched = zero_suffix;
    if (matched == nullptr) return false;

    const std::string raw_weight =
        logical.raw_name.substr(0, logical.raw_name.size() -
                                      std::strlen(matched));
    return loader.contains(raw_weight) &&
           (loader.info(raw_weight).dtype == DType::FP8_E4M3 ||
            loader.info(raw_weight).dtype == DType::INT8);
}

bool is_fp8_scale_inv_companion(
    const TensorMetadataSource& loader,
    const LogicalTensor& logical)
{
    constexpr const char* scale_suffix = "_scale_inv";
    if (!ends_with(logical.raw_name, scale_suffix)) return false;
    const std::string raw_weight =
        logical.raw_name.substr(0, logical.raw_name.size() -
                                      std::strlen(scale_suffix));
    return loader.contains(raw_weight) &&
           loader.info(raw_weight).dtype == DType::FP8_E4M3;
}

bool try_add_compressed_fp8_weight(
    LoadPlan& plan,
    const HfConfig& hf,
    const TensorMetadataSource& loader,
    const LogicalTensor& logical,
    const TensorInfo& info,
    int shard_axis,
    int tp_size,
    bool fp8_native)
{
    if (hf.quant_method != "compressed-tensors" ||
        info.dtype != DType::FP8_E4M3 ||
        !ends_with(logical.runtime_name, ".weight")) {
        return false;
    }

    const std::string scale_raw = logical.raw_name + "_scale";
    if (!loader.contains(scale_raw)) return false;
    const TensorInfo& scale_info = loader.info(scale_raw);
    if (scale_info.dtype != DType::BF16 && scale_info.dtype != DType::FP32) {
        throw std::runtime_error(
            "load schema: compressed-tensors FP8 scale '" + scale_raw +
            "' has unsupported dtype " +
            std::string(dtype_name(scale_info.dtype)));
    }

    const auto final_shape =
        sharded_shape(info.shape, shard_axis, tp_size, logical.runtime_name);
    const int scale_axis = (tp_size > 1)
        ? llama_like_shard_axis(logical.runtime_name + "_scale")
        : -1;
    const auto scale_shape = sharded_shape(
        scale_info.shape, scale_axis, tp_size, logical.runtime_name + "_scale");
    std::uint64_t scale_numel = 1;
    for (const auto dim : scale_shape) {
        scale_numel *= static_cast<std::uint64_t>(dim);
    }

    if (fp8_native) {
        const std::string scale_name = logical.runtime_name + "_scale_inv";

        QuantSpec quant;
        quant.format = QuantFormat::CompressedFp8E4M3;
        quant.granularity = (scale_numel == 1)
            ? QuantGranularity::PerTensor
            : QuantGranularity::PerChannel;
        quant.group_size = 0;
        quant.channel_axis = 0;
        quant.scale_tensor = scale_name;

        plan.ops.push_back(make_raw_load_op(
            LoadOpKind::Copy, logical.runtime_name,
            logical.raw_name, shard_axis));
        register_tensor_spec(
            plan, logical.runtime_name, DType::FP8_E4M3, final_shape,
            TensorLayoutKind::QuantPacked, TensorOwnershipKind::Owned,
            parallel_kind_from_axis(shard_axis),
            /*backing_tensor=*/{}, std::move(quant));

        register_tensor_spec(
            plan, scale_name, DType::FP32, scale_shape,
            TensorLayoutKind::Dense, TensorOwnershipKind::Owned,
            parallel_kind_from_axis(scale_axis));
        if (scale_info.dtype == DType::FP32) {
            plan.ops.push_back(make_raw_load_op(
                LoadOpKind::Copy, scale_name, scale_raw, scale_axis));
        } else {
            const std::string scale_tmp = scale_name + ".__source";
            plan.ops.push_back(make_raw_load_op(
                LoadOpKind::Copy, scale_tmp, scale_raw, scale_axis));

            register_tensor_spec(
                plan, scale_tmp, scale_info.dtype, scale_shape,
                TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
                parallel_kind_from_axis(scale_axis));

            plan.ops.push_back(make_tensor_op(
                LoadOpKind::Cast, scale_name, {scale_tmp}));
            plan.ops.push_back(make_tensor_op(
                LoadOpKind::Drop, scale_tmp + ".__drop", {scale_tmp}));
        }

        plan.ops.push_back(make_tensor_op(
            LoadOpKind::BindMetadata, logical.runtime_name));
    } else {
        const std::string weight_tmp =
            logical.runtime_name + ".__compressed_fp8_source";
        const std::string scale_tmp =
            logical.runtime_name + ".__compressed_fp8_scale";

        plan.ops.push_back(make_raw_load_op(
            LoadOpKind::Copy, weight_tmp, logical.raw_name, shard_axis));
        plan.ops.push_back(make_raw_load_op(
            LoadOpKind::Copy, scale_tmp, scale_raw, scale_axis));

        register_tensor_spec(
            plan, weight_tmp, DType::FP8_E4M3, final_shape,
            TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
            parallel_kind_from_axis(shard_axis));
        register_tensor_spec(
            plan, scale_tmp, scale_info.dtype, scale_shape,
            TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
            parallel_kind_from_axis(scale_axis));

        register_tensor_spec(
            plan, logical.runtime_name, DType::BF16, final_shape,
            TensorLayoutKind::Dense, TensorOwnershipKind::Owned,
            parallel_kind_from_axis(shard_axis));

        plan.ops.push_back(make_tensor_op(
            LoadOpKind::Dequantize, logical.runtime_name,
            {weight_tmp, scale_tmp}));
        plan.ops.push_back(make_tensor_op(
            LoadOpKind::Drop,
            logical.runtime_name + ".__compressed_fp8_drop",
            {weight_tmp, scale_tmp}));

        std::uint64_t source_numel = 1;
        for (const auto dim : final_shape) {
            source_numel *= static_cast<std::uint64_t>(dim);
        }
        const std::uint64_t scale_bytes =
            scale_numel * static_cast<std::uint64_t>(dtype_bytes(scale_info.dtype));
        const std::uint64_t fp32_scale_scratch =
            scale_info.dtype == DType::BF16
                ? scale_numel * static_cast<std::uint64_t>(dtype_bytes(DType::FP32))
                : 0;
        estimate_temporary_bytes(
            plan,
            source_numel * static_cast<std::uint64_t>(dtype_bytes(DType::FP8_E4M3)) +
            scale_bytes + fp32_scale_scratch);
    }
    return true;
}

bool try_add_fp8_scale_inv_weight(
    LoadPlan& plan,
    const TensorMetadataSource& loader,
    const LogicalTensor& logical,
    const TensorInfo& info,
    int shard_axis,
    int tp_size)
{
    if (info.dtype != DType::FP8_E4M3 ||
        !ends_with(logical.runtime_name, ".weight")) {
        return false;
    }

    const std::string scale_raw = logical.raw_name + "_scale_inv";
    if (!loader.contains(scale_raw)) return false;
    const TensorInfo& scale_info = loader.info(scale_raw);
    if (scale_info.dtype != DType::BF16 && scale_info.dtype != DType::FP32) {
        throw std::runtime_error(
            "load schema: FP8 scale_inv '" + scale_raw +
            "' has unsupported dtype " +
            std::string(dtype_name(scale_info.dtype)));
    }

    const auto final_shape =
        sharded_shape(info.shape, shard_axis, tp_size, logical.runtime_name);
    const std::string scale_name = logical.runtime_name + "_scale_inv";
    int scale_axis = -1;
    if (scale_info.shape.size() == 1 && final_shape.size() >= 1 &&
        scale_info.shape[0] == info.shape[0]) {
        scale_axis = shard_axis == 0 ? 0 : -1;
    }
    const auto scale_shape =
        sharded_shape(scale_info.shape, scale_axis, tp_size, scale_name);

    std::uint64_t scale_numel = 1;
    for (const auto dim : scale_shape) {
        scale_numel *= static_cast<std::uint64_t>(dim);
    }
    if (scale_numel != 1 &&
        (final_shape.empty() ||
         scale_numel != static_cast<std::uint64_t>(final_shape[0]))) {
        throw std::runtime_error(
            "load schema: FP8 scale_inv '" + scale_raw +
            "' must be scalar or one scale per output row for '" +
            logical.runtime_name + "'");
    }

    const std::string weight_tmp =
        logical.runtime_name + ".__fp8_scale_inv_source";
    const std::string scale_tmp =
        logical.runtime_name + ".__fp8_scale_inv_scale";

    plan.ops.push_back(make_raw_load_op(
        LoadOpKind::Copy, weight_tmp, logical.raw_name, shard_axis));
    plan.ops.push_back(make_raw_load_op(
        LoadOpKind::Copy, scale_tmp, scale_raw, scale_axis));

    register_tensor_spec(
        plan, weight_tmp, DType::FP8_E4M3, final_shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
        parallel_kind_from_axis(shard_axis));
    register_tensor_spec(
        plan, scale_tmp, scale_info.dtype, scale_shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
        parallel_kind_from_axis(scale_axis));
    register_tensor_spec(
        plan, logical.runtime_name, DType::BF16, final_shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Owned,
        parallel_kind_from_axis(shard_axis));

    plan.ops.push_back(make_tensor_op(
        LoadOpKind::Dequantize, logical.runtime_name,
        {weight_tmp, scale_tmp}));
    plan.ops.push_back(make_tensor_op(
        LoadOpKind::Drop,
        logical.runtime_name + ".__fp8_scale_inv_drop",
        {weight_tmp, scale_tmp}));

    estimate_temporary_bytes(
        plan,
        tensor_nbytes(DType::FP8_E4M3, final_shape) +
        tensor_nbytes(scale_info.dtype, scale_shape) +
        (scale_info.dtype == DType::BF16
            ? tensor_nbytes(DType::FP32, scale_shape)
            : 0));
    return true;
}

bool try_add_compressed_int8_weight(
    LoadPlan& plan,
    const HfConfig& hf,
    const TensorMetadataSource& loader,
    const LogicalTensor& logical,
    const TensorInfo& info,
    int shard_axis,
    int tp_size)
{
    if (hf.quant_method != "compressed-tensors" ||
        info.dtype != DType::INT8 ||
        !ends_with(logical.runtime_name, ".weight")) {
        return false;
    }

    const std::string scale_raw = logical.raw_name + "_scale";
    const std::string zero_raw = logical.raw_name + "_zero_point";
    if (!loader.contains(scale_raw)) return false;
    if (loader.contains(zero_raw)) {
        throw std::runtime_error(
            "load schema: compressed-tensors INT8 weight '" +
            logical.raw_name + "' has a zero-point companion. The scheduled "
            "INT8 runtime backend supports symmetric per-channel INT8; "
            "asymmetric compressed-tensors INT8 should lower through an "
            "explicit Dequantize op once that kernel is registered.");
    }

    const TensorInfo& scale_info = loader.info(scale_raw);
    if (scale_info.dtype != DType::BF16 && scale_info.dtype != DType::FP32) {
        throw std::runtime_error(
            "load schema: compressed-tensors INT8 scale '" + scale_raw +
            "' has unsupported dtype " +
            std::string(dtype_name(scale_info.dtype)));
    }

    const auto final_shape =
        sharded_shape(info.shape, shard_axis, tp_size, logical.runtime_name);
    if (final_shape.size() != 2) {
        throw std::runtime_error(
            "load schema: compressed-tensors INT8 weight is not 2-D: " +
            logical.runtime_name);
    }

    const int scale_axis = (tp_size > 1)
        ? llama_like_shard_axis(logical.runtime_name + "_scale")
        : -1;
    const auto scale_shape = sharded_shape(
        scale_info.shape, scale_axis, tp_size, logical.runtime_name + "_scale");
    std::uint64_t scale_numel = 1;
    for (const auto dim : scale_shape) {
        scale_numel *= static_cast<std::uint64_t>(dim);
    }
    if (scale_numel != static_cast<std::uint64_t>(final_shape[0])) {
        throw std::runtime_error(
            "load schema: compressed-tensors INT8 currently requires one "
            "scale per output row for '" + logical.runtime_name + "'");
    }

    const std::string scale_name = logical.runtime_name + "_scale_inv";

    QuantSpec quant;
    quant.format = QuantFormat::CompressedInt8;
    quant.granularity = QuantGranularity::PerChannel;
    quant.group_size = 0;
    quant.channel_axis = 0;
    quant.scale_tensor = scale_name;

    plan.ops.push_back(make_raw_load_op(
        LoadOpKind::Copy, logical.runtime_name,
        logical.raw_name, shard_axis));
    register_tensor_spec(
        plan, logical.runtime_name, DType::INT8, final_shape,
        TensorLayoutKind::QuantPacked, TensorOwnershipKind::Owned,
        parallel_kind_from_axis(shard_axis),
        /*backing_tensor=*/{}, std::move(quant));

    register_tensor_spec(
        plan, scale_name, DType::FP32, scale_shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Owned,
        parallel_kind_from_axis(scale_axis));
    if (scale_info.dtype == DType::FP32) {
        plan.ops.push_back(make_raw_load_op(
            LoadOpKind::Copy, scale_name, scale_raw, scale_axis));
    } else {
        const std::string scale_tmp = scale_name + ".__source";
        plan.ops.push_back(make_raw_load_op(
            LoadOpKind::Copy, scale_tmp, scale_raw, scale_axis));

        register_tensor_spec(
            plan, scale_tmp, scale_info.dtype, scale_shape,
            TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
            parallel_kind_from_axis(scale_axis));

        plan.ops.push_back(make_tensor_op(
            LoadOpKind::Cast, scale_name, {scale_tmp}));
        plan.ops.push_back(make_tensor_op(
            LoadOpKind::Drop, scale_tmp + ".__drop", {scale_tmp}));
    }

    plan.ops.push_back(make_tensor_op(
        LoadOpKind::BindMetadata, logical.runtime_name));
    return true;
}

bool can_lower_gptq_marlin_repack_to_plan(
    const HfConfig& hf,
    int tp_size)
{
    return hf.quant_method == "gptq" &&
           hf.quant_bits == 4 &&
           hf.quant_group_size > 0 &&
           !hf.quant_desc_act &&
           hf.quant_sym &&
           !hf.quant_zero_point &&
           tp_size >= 1;
}

bool can_lower_awq_dequant_to_plan(const HfConfig& hf) {
    return hf.quant_method == "awq" &&
           hf.quant_bits == 4 &&
           hf.quant_group_size > 0;
}

bool can_lower_awq_marlin_repack_to_plan(const HfConfig& hf) {
    return hf.quant_method == "awq" &&
           hf.quant_bits == 4 &&
           hf.quant_group_size > 0;
}

bool can_lower_gptq_dequant_to_plan(
    const HfConfig& hf,
    int tp_size)
{
    (void)tp_size;
    if (hf.quant_method != "gptq" ||
        hf.quant_bits != 4 ||
        hf.quant_group_size <= 0) {
        return false;
    }
    const bool needs_bf16_fallback =
        hf.quant_desc_act || hf.quant_zero_point || !hf.quant_sym;
    return needs_bf16_fallback;
}

enum class OfflineInt4Format {
    Awq,
    Gptq,
};

struct OfflineInt4ShardPlan {
    int canonical_axis = -1;
    int qweight_axis = -1;
    int qzeros_axis = -1;
    int scale_axis = -1;
    int gidx_axis = -1;
    TensorParallelKind parallel = TensorParallelKind::Replicated;
};

OfflineInt4ShardPlan offline_int4_shard_plan_for(
    OfflineInt4Format format,
    const std::string& canonical_weight_name,
    int tp_size)
{
    OfflineInt4ShardPlan out;
    if (tp_size <= 1) return out;

    out.canonical_axis = llama_like_shard_axis(canonical_weight_name);
    out.parallel = parallel_kind_from_axis(out.canonical_axis);
    if (out.canonical_axis == 0) {
        // Canonical dense weight is [N, K]. Column-parallel sharding cuts N.
        // GPTQ qweight is [K/8, N]; AWQ qweight/qzeros pack N by 8.
        out.qweight_axis = 1;
        out.qzeros_axis = 1;
        out.scale_axis = 1;
    } else if (out.canonical_axis == 1) {
        // Row-parallel sharding cuts K. GPTQ packs K by 8 on qweight axis 0;
        // AWQ stores K directly on qweight axis 0. Both use K-derived scale
        // groups, and GPTQ g_idx follows K when present.
        out.qweight_axis = 0;
        out.qzeros_axis = 0;
        out.scale_axis = 0;
        if (format == OfflineInt4Format::Gptq) {
            out.gidx_axis = 0;
        }
    }
    return out;
}

OfflineInt4ShardPlan gptq_shard_plan_for(
    const std::string& canonical_weight_name,
    int tp_size)
{
    return offline_int4_shard_plan_for(
        OfflineInt4Format::Gptq, canonical_weight_name, tp_size);
}

void add_awq_marlin_repack_ops(
    LoadPlan& plan,
    const HfConfig& hf,
    const TensorInfo& qweight_info,
    const TensorInfo& qzeros_info,
    const TensorInfo& scale_info,
    const std::string& raw_qweight,
    const std::string& raw_qzeros,
    const std::string& raw_scales,
    const std::string& canonical_w,
    const OfflineInt4ShardPlan& shard_plan,
    int tp_size)
{
    const auto local_qweight_shape = sharded_shape(
        qweight_info.shape, shard_plan.qweight_axis, tp_size, raw_qweight);
    const auto local_qzeros_shape = sharded_shape(
        qzeros_info.shape, shard_plan.qzeros_axis, tp_size, raw_qzeros);
    const auto local_scale_shape = sharded_shape(
        scale_info.shape, shard_plan.scale_axis, tp_size, raw_scales);

    const std::int64_t k_local = local_qweight_shape[0];
    const std::int64_t n_local = local_qweight_shape[1] * 8;
    if (k_local <= 0 || n_local <= 0 ||
        k_local % 16 != 0 ||
        n_local % 64 != 0 ||
        k_local % hf.quant_group_size != 0) {
        throw std::runtime_error(
            "load schema: AWQ qweight '" + raw_qweight +
            "' local shape is not compatible with Marlin repack");
    }
    const std::int64_t groups_local = k_local / hf.quant_group_size;
    if (local_qzeros_shape !=
            std::vector<std::int64_t>{groups_local, n_local / 8} ||
        local_scale_shape !=
            std::vector<std::int64_t>{groups_local, n_local}) {
        throw std::runtime_error(
            "load schema: AWQ qzeros/scales TP slices do not match "
            "qweight/group_size for Marlin repack at '" + raw_qweight + "'");
    }

    const std::string canonical_s = canonical_w + "_scale_inv";
    const std::string canonical_z = canonical_w + "_zero_point";
    const std::string tmp_qw = canonical_w + ".__awq_qweight";
    const std::string tmp_qz = canonical_w + ".__awq_qzeros";
    const std::string tmp_scales = canonical_w + ".__awq_scales";

    plan.ops.push_back(make_raw_load_op(
        LoadOpKind::Copy, tmp_qw, raw_qweight, shard_plan.qweight_axis));
    plan.ops.push_back(make_raw_load_op(
        LoadOpKind::Copy, tmp_qz, raw_qzeros, shard_plan.qzeros_axis));
    plan.ops.push_back(make_raw_load_op(
        LoadOpKind::Copy, tmp_scales, raw_scales, shard_plan.scale_axis));
    plan.ops.push_back(make_tensor_op(
        LoadOpKind::RepackLayout, canonical_w,
        {tmp_qw, tmp_qz, tmp_scales}, canonical_s,
        shard_plan.canonical_axis));
    plan.ops.push_back(make_tensor_op(
        LoadOpKind::BindMetadata, canonical_w));
    plan.ops.push_back(make_tensor_op(
        LoadOpKind::Drop, canonical_w + ".__drop_awq_sources",
        {tmp_qw, tmp_qz, tmp_scales}));

    register_tensor_spec(
        plan, tmp_qw, qweight_info.dtype, local_qweight_shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
        shard_plan.parallel);
    register_tensor_spec(
        plan, tmp_qz, qzeros_info.dtype, local_qzeros_shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
        shard_plan.parallel);
    register_tensor_spec(
        plan, tmp_scales, scale_info.dtype, local_scale_shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
        shard_plan.parallel);

    QuantSpec quant;
    quant.format = QuantFormat::AwqInt4;
    quant.granularity = QuantGranularity::PerGroup;
    quant.group_size = hf.quant_group_size;
    quant.channel_axis = 0;
    quant.scale_tensor = canonical_s;
    quant.zero_point_tensor = canonical_z;

    register_tensor_spec(
        plan, canonical_w, DType::INT4_PACKED,
        {k_local / 16, n_local * 8},
        TensorLayoutKind::QuantPacked, TensorOwnershipKind::Owned,
        shard_plan.parallel,
        /*backing_tensor=*/{}, std::move(quant));
    register_tensor_spec(
        plan, canonical_s, DType::BF16, local_scale_shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Owned,
        shard_plan.parallel);
    register_tensor_spec(
        plan, canonical_z, DType::INT32, local_qzeros_shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Owned,
        shard_plan.parallel);
    estimate_temporary_bytes(
        plan,
        tensor_nbytes(qweight_info.dtype, local_qweight_shape) +
        tensor_nbytes(qzeros_info.dtype, local_qzeros_shape) +
        tensor_nbytes(scale_info.dtype, local_scale_shape));
}

bool is_gptq_repack_companion(
    const HfConfig& hf,
    const TensorMetadataSource& loader,
    const LogicalTensor& logical,
    bool lowering_enabled)
{
    if (!lowering_enabled || hf.quant_method != "gptq") return false;

    constexpr const char* scale_suffix = ".scales";
    constexpr const char* zero_suffix = ".qzeros";
    constexpr const char* gidx_suffix = ".g_idx";

    const char* matched = nullptr;
    if (ends_with(logical.raw_name, scale_suffix)) matched = scale_suffix;
    if (ends_with(logical.raw_name, zero_suffix)) matched = zero_suffix;
    if (ends_with(logical.raw_name, gidx_suffix)) matched = gidx_suffix;
    if (matched == nullptr) return false;

    const std::string prefix =
        logical.raw_name.substr(0, logical.raw_name.size() -
                                      std::strlen(matched));
    return loader.contains(prefix + ".qweight");
}

bool is_offline_int4_dequant_companion(
    const HfConfig& hf,
    const TensorMetadataSource& loader,
    const LogicalTensor& logical,
    bool lowering_enabled)
{
    if (!lowering_enabled) return false;
    if (hf.quant_method != "awq" && hf.quant_method != "gptq") return false;

    constexpr const char* scale_suffix = ".scales";
    constexpr const char* zero_suffix = ".qzeros";
    constexpr const char* gidx_suffix = ".g_idx";

    const char* matched = nullptr;
    if (ends_with(logical.raw_name, scale_suffix)) matched = scale_suffix;
    if (ends_with(logical.raw_name, zero_suffix)) matched = zero_suffix;
    if (hf.quant_method == "gptq" &&
        ends_with(logical.raw_name, gidx_suffix)) {
        matched = gidx_suffix;
    }
    if (matched == nullptr) return false;

    const std::string prefix =
        logical.raw_name.substr(0, logical.raw_name.size() -
                                      std::strlen(matched));
    return loader.contains(prefix + ".qweight");
}

bool try_add_gptq_marlin_repack_weight(
    LoadPlan& plan,
    const HfConfig& hf,
    const TensorMetadataSource& loader,
    const LogicalTensor& logical,
    const TensorInfo& info,
    int tp_size,
    std::unordered_set<std::string>& consumed_raw)
{
    constexpr const char* qweight_suffix = ".qweight";
    if (!ends_with(logical.runtime_name, qweight_suffix)) return false;

    if (info.dtype != DType::INT32 || info.shape.size() != 2) {
        throw std::runtime_error(
            "load schema: GPTQ qweight '" + logical.raw_name +
            "' must be a 2-D int32 tensor");
    }

    const std::string raw_prefix =
        logical.raw_name.substr(0, logical.raw_name.size() -
                                      std::strlen(qweight_suffix));
    const std::string runtime_prefix =
        logical.runtime_name.substr(0, logical.runtime_name.size() -
                                          std::strlen(qweight_suffix));
    const std::string canonical_w = runtime_prefix + ".weight";
    const OfflineInt4ShardPlan shard_plan =
        gptq_shard_plan_for(canonical_w, tp_size);
    const std::string raw_scales = raw_prefix + ".scales";
    if (!loader.contains(raw_scales)) {
        throw std::runtime_error(
            "load schema: GPTQ qweight '" + logical.raw_name +
            "' is missing matching '.scales' tensor");
    }
    const TensorInfo& scale_info = loader.info(raw_scales);
    if (scale_info.dtype != DType::FP16 || scale_info.shape.size() != 2) {
        throw std::runtime_error(
            "load schema: GPTQ scales '" + raw_scales +
            "' must be a 2-D fp16 tensor for Marlin repack");
    }

    const std::int64_t k_full = info.shape[0] * 8;
    const std::int64_t n_full = info.shape[1];
    if (k_full <= 0 || n_full <= 0 || k_full % 16 != 0) {
        throw std::runtime_error(
            "load schema: GPTQ qweight '" + logical.raw_name +
            "' has unsupported packed shape");
    }
    if (k_full % hf.quant_group_size != 0) {
        throw std::runtime_error(
            "load schema: GPTQ qweight '" + logical.raw_name +
            "' K dimension is not divisible by group_size=" +
            std::to_string(hf.quant_group_size));
    }
    const std::int64_t expected_groups = k_full / hf.quant_group_size;
    if (scale_info.shape[0] != expected_groups ||
        scale_info.shape[1] != n_full) {
        throw std::runtime_error(
            "load schema: GPTQ scales '" + raw_scales +
            "' shape does not match qweight/group_size");
    }

    const std::string canonical_s = runtime_prefix + ".weight_scale_inv";
    const auto local_qweight_shape = sharded_shape(
        info.shape, shard_plan.qweight_axis, tp_size, logical.raw_name);
    const auto local_scale_shape = sharded_shape(
        scale_info.shape, shard_plan.scale_axis, tp_size, raw_scales);
    const std::int64_t k_local = local_qweight_shape[0] * 8;
    const std::int64_t n_local = local_qweight_shape[1];
    if (local_scale_shape[1] != n_local) {
        throw std::runtime_error(
            "load schema: GPTQ scale/qweight TP slices disagree for '" +
            logical.raw_name + "'");
    }
    if (k_local % hf.quant_group_size != 0 ||
        local_scale_shape[0] != k_local / hf.quant_group_size) {
        throw std::runtime_error(
            "load schema: GPTQ row-parallel slice for '" + logical.raw_name +
            "' does not preserve whole quantization groups");
    }
    const std::string tmp_qw = canonical_w + ".__gptq_qweight";
    const std::string tmp_scales = canonical_w + ".__gptq_scales";

    plan.ops.push_back(make_raw_load_op(
        LoadOpKind::Copy, tmp_qw, logical.raw_name,
        shard_plan.qweight_axis));
    plan.ops.push_back(make_raw_load_op(
        LoadOpKind::Copy, tmp_scales, raw_scales,
        shard_plan.scale_axis));
    plan.ops.push_back(make_tensor_op(
        LoadOpKind::RepackLayout, canonical_w,
        {tmp_qw, tmp_scales}, canonical_s,
        shard_plan.canonical_axis));
    plan.ops.push_back(make_tensor_op(
        LoadOpKind::BindMetadata, canonical_w));
    plan.ops.push_back(make_tensor_op(
        LoadOpKind::Drop, canonical_w + ".__drop_gptq_sources",
        {tmp_qw, tmp_scales}));

    register_tensor_spec(
        plan, tmp_qw, info.dtype, local_qweight_shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
        shard_plan.parallel);
    register_tensor_spec(
        plan, tmp_scales, scale_info.dtype, local_scale_shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
        shard_plan.parallel);

    QuantSpec quant;
    quant.format = QuantFormat::GptqInt4;
    quant.granularity = QuantGranularity::PerGroup;
    quant.group_size = hf.quant_group_size;
    quant.channel_axis = 0;
    quant.scale_tensor = canonical_s;

    register_tensor_spec(
        plan, canonical_w, DType::INT4_PACKED,
        {k_local / 16, n_local * 8},
        TensorLayoutKind::QuantPacked, TensorOwnershipKind::Owned,
        shard_plan.parallel,
        /*backing_tensor=*/{}, std::move(quant));
    register_tensor_spec(
        plan, canonical_s, DType::BF16, local_scale_shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Owned,
        shard_plan.parallel);
    estimate_temporary_bytes(
        plan,
        tensor_nbytes(info.dtype, local_qweight_shape) +
        tensor_nbytes(scale_info.dtype, local_scale_shape));

    consumed_raw.insert(logical.raw_name);
    consumed_raw.insert(raw_scales);
    if (loader.contains(raw_prefix + ".qzeros")) {
        consumed_raw.insert(raw_prefix + ".qzeros");
    }
    if (loader.contains(raw_prefix + ".g_idx")) {
        consumed_raw.insert(raw_prefix + ".g_idx");
    }
    return true;
}

bool scale_dtype_supported_for_int4_dequant(DType dtype) {
    return dtype == DType::BF16 || dtype == DType::FP16 || dtype == DType::FP32;
}

void add_int4_dequant_ops(
    LoadPlan& plan,
    const HfConfig& hf,
    const TensorInfo& qweight_info,
    const TensorInfo& qzeros_info,
    const TensorInfo& scale_info,
    const TensorInfo* gidx_info,
    const std::string& raw_qweight,
    const std::string& raw_qzeros,
    const std::string& raw_scales,
    const std::string& raw_gidx,
    const std::string& canonical_w,
    OfflineInt4Format format,
    const OfflineInt4ShardPlan& shard_plan,
    int tp_size)
{
    const bool is_awq = format == OfflineInt4Format::Awq;
    const QuantFormat quant_format =
        is_awq ? QuantFormat::AwqInt4 : QuantFormat::GptqInt4;
    const std::string quant_prefix = is_awq ? "awq" : "gptq";

    const auto local_qweight_shape = sharded_shape(
        qweight_info.shape, shard_plan.qweight_axis, tp_size, raw_qweight);
    const auto local_qzeros_shape = sharded_shape(
        qzeros_info.shape, shard_plan.qzeros_axis, tp_size, raw_qzeros);
    const auto local_scale_shape = sharded_shape(
        scale_info.shape, shard_plan.scale_axis, tp_size, raw_scales);

    const std::int64_t k_local =
        is_awq ? local_qweight_shape[0] : local_qweight_shape[0] * 8;
    const std::int64_t n_local =
        is_awq ? local_qweight_shape[1] * 8 : local_qweight_shape[1];
    if (k_local <= 0 || n_local <= 0 ||
        k_local % hf.quant_group_size != 0) {
        throw std::runtime_error(
            "load schema: " + quant_prefix + " qweight '" + raw_qweight +
            "' local shape is not compatible with group_size=" +
            std::to_string(hf.quant_group_size));
    }
    const std::int64_t groups_local = k_local / hf.quant_group_size;
    const std::vector<std::int64_t> expected_scale_shape =
        {groups_local, n_local};
    const std::vector<std::int64_t> expected_qzeros_shape =
        {groups_local, n_local / 8};
    if (n_local % 8 != 0) {
        throw std::runtime_error(
            "load schema: " + quant_prefix + " local N is not divisible "
            "by 8 for '" + raw_qweight + "'");
    }
    if (gidx_info == nullptr) {
        if (local_scale_shape != expected_scale_shape ||
            local_qzeros_shape != expected_qzeros_shape) {
            throw std::runtime_error(
                "load schema: " + quant_prefix + " qzeros/scales TP slices "
                "do not match qweight/group_size for '" + raw_qweight + "'");
        }
    } else {
        if (local_scale_shape.size() != 2 || local_qzeros_shape.size() != 2 ||
            local_scale_shape[1] != n_local ||
            local_qzeros_shape[1] != n_local / 8 ||
            local_scale_shape[0] != local_qzeros_shape[0] ||
            local_scale_shape[0] < groups_local) {
            throw std::runtime_error(
                "load schema: GPTQ act-order qzeros/scales slices do not "
                "cover local g_idx groups for '" + raw_qweight + "'");
        }
    }

    const std::string tmp_qw =
        canonical_w + ".__" + quant_prefix + "_qweight";
    const std::string tmp_qz =
        canonical_w + ".__" + quant_prefix + "_qzeros";
    const std::string tmp_scales =
        canonical_w + ".__" + quant_prefix + "_scales";
    const std::string tmp_gidx =
        canonical_w + ".__" + quant_prefix + "_g_idx";

    plan.ops.push_back(make_raw_load_op(
        LoadOpKind::Copy, tmp_qw, raw_qweight,
        shard_plan.qweight_axis));
    plan.ops.push_back(make_raw_load_op(
        LoadOpKind::Copy, tmp_qz, raw_qzeros,
        shard_plan.qzeros_axis));
    plan.ops.push_back(make_raw_load_op(
        LoadOpKind::Copy, tmp_scales, raw_scales,
        shard_plan.scale_axis));

    std::vector<std::string> dequant_inputs = {tmp_qw, tmp_qz, tmp_scales};
    std::vector<std::string> drop_inputs = dequant_inputs;
    if (gidx_info != nullptr) {
        const auto local_gidx_shape = sharded_shape(
            gidx_info->shape, shard_plan.gidx_axis, tp_size, raw_gidx);
        if (local_gidx_shape != std::vector<std::int64_t>{k_local}) {
            throw std::runtime_error(
                "load schema: GPTQ g_idx slice does not match local K for '" +
                raw_qweight + "'");
        }

        plan.ops.push_back(make_raw_load_op(
            LoadOpKind::Copy, tmp_gidx, raw_gidx,
            shard_plan.gidx_axis));
        dequant_inputs.push_back(tmp_gidx);
        drop_inputs.push_back(tmp_gidx);

        register_tensor_spec(
            plan, tmp_gidx, gidx_info->dtype, local_gidx_shape,
            TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
            shard_plan.parallel);
    }

    plan.ops.push_back(make_tensor_op(
        LoadOpKind::Dequantize, canonical_w,
        std::move(dequant_inputs), /*secondary_output_name=*/{},
        shard_plan.canonical_axis));
    plan.ops.push_back(make_tensor_op(
        LoadOpKind::Drop,
        canonical_w + ".__drop_" + quant_prefix + "_sources",
        std::move(drop_inputs)));

    QuantSpec source_quant;
    source_quant.format = quant_format;
    source_quant.granularity = QuantGranularity::PerGroup;
    source_quant.group_size = hf.quant_group_size;
    source_quant.channel_axis = 0;
    source_quant.scale_tensor = tmp_scales;
    source_quant.zero_point_tensor = tmp_qz;

    register_tensor_spec(
        plan, tmp_qw, qweight_info.dtype, local_qweight_shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
        shard_plan.parallel, /*backing_tensor=*/{}, std::move(source_quant));
    register_tensor_spec(
        plan, tmp_qz, qzeros_info.dtype, local_qzeros_shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
        shard_plan.parallel);
    register_tensor_spec(
        plan, tmp_scales, scale_info.dtype, local_scale_shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
        shard_plan.parallel);
    register_tensor_spec(
        plan, canonical_w, DType::BF16, {n_local, k_local},
        TensorLayoutKind::Dense, TensorOwnershipKind::Owned,
        shard_plan.parallel);

    const std::uint64_t scale_cast_scratch =
        scale_info.dtype == DType::BF16
            ? 0
            : tensor_nbytes(DType::BF16, local_scale_shape);
    std::uint64_t gidx_bytes = 0;
    if (gidx_info != nullptr) {
        gidx_bytes = tensor_nbytes(
            gidx_info->dtype,
            sharded_shape(gidx_info->shape, shard_plan.gidx_axis,
                          tp_size, raw_gidx));
    }
    estimate_temporary_bytes(
        plan,
        tensor_nbytes(qweight_info.dtype, local_qweight_shape) +
        tensor_nbytes(qzeros_info.dtype, local_qzeros_shape) +
        tensor_nbytes(scale_info.dtype, local_scale_shape) +
        scale_cast_scratch + gidx_bytes);
}

bool try_add_awq_dequant_weight(
    LoadPlan& plan,
    const HfConfig& hf,
    const TensorMetadataSource& loader,
    const LogicalTensor& logical,
    const TensorInfo& info,
    int tp_size,
    std::unordered_set<std::string>& consumed_raw)
{
    constexpr const char* qweight_suffix = ".qweight";
    if (!ends_with(logical.runtime_name, qweight_suffix)) return false;
    if (info.dtype != DType::INT32 || info.shape.size() != 2) {
        throw std::runtime_error(
            "load schema: AWQ qweight '" + logical.raw_name +
            "' must be a 2-D int32 tensor");
    }

    const std::string raw_prefix =
        logical.raw_name.substr(0, logical.raw_name.size() -
                                      std::strlen(qweight_suffix));
    const std::string runtime_prefix =
        logical.runtime_name.substr(0, logical.runtime_name.size() -
                                          std::strlen(qweight_suffix));
    const std::string raw_qzeros = raw_prefix + ".qzeros";
    const std::string raw_scales = raw_prefix + ".scales";
    if (!loader.contains(raw_qzeros) || !loader.contains(raw_scales)) {
        throw std::runtime_error(
            "load schema: AWQ qweight '" + logical.raw_name +
            "' is missing qzeros/scales companions");
    }
    const TensorInfo& qzeros_info = loader.info(raw_qzeros);
    const TensorInfo& scale_info = loader.info(raw_scales);
    if (qzeros_info.dtype != DType::INT32 || qzeros_info.shape.size() != 2 ||
        scale_info.shape.size() != 2 ||
        !scale_dtype_supported_for_int4_dequant(scale_info.dtype)) {
        throw std::runtime_error(
            "load schema: AWQ qzeros/scales for '" + logical.raw_name +
            "' have unsupported dtype or rank");
    }

    const std::int64_t k_full = info.shape[0];
    const std::int64_t n_full = info.shape[1] * 8;
    if (k_full <= 0 || n_full <= 0 ||
        k_full % hf.quant_group_size != 0) {
        throw std::runtime_error(
            "load schema: AWQ qweight '" + logical.raw_name +
            "' shape is incompatible with group_size=" +
            std::to_string(hf.quant_group_size));
    }
    const std::int64_t groups = k_full / hf.quant_group_size;
    if (qzeros_info.shape != std::vector<std::int64_t>{groups, n_full / 8} ||
        scale_info.shape != std::vector<std::int64_t>{groups, n_full}) {
        throw std::runtime_error(
            "load schema: AWQ qzeros/scales shape does not match qweight for '" +
            logical.raw_name + "'");
    }

    const std::string canonical_w = runtime_prefix + ".weight";
    const OfflineInt4ShardPlan shard_plan =
        offline_int4_shard_plan_for(
            OfflineInt4Format::Awq, canonical_w, tp_size);
    add_int4_dequant_ops(
        plan, hf, info, qzeros_info, scale_info, nullptr,
        logical.raw_name, raw_qzeros, raw_scales, {},
        canonical_w, OfflineInt4Format::Awq, shard_plan, tp_size);

    consumed_raw.insert(logical.raw_name);
    consumed_raw.insert(raw_qzeros);
    consumed_raw.insert(raw_scales);
    return true;
}

bool try_add_awq_marlin_repack_weight(
    LoadPlan& plan,
    const HfConfig& hf,
    const TensorMetadataSource& loader,
    const LogicalTensor& logical,
    const TensorInfo& info,
    int tp_size,
    std::unordered_set<std::string>& consumed_raw)
{
    constexpr const char* qweight_suffix = ".qweight";
    if (!ends_with(logical.runtime_name, qweight_suffix)) return false;
    if (info.dtype != DType::INT32 || info.shape.size() != 2) {
        throw std::runtime_error(
            "load schema: AWQ qweight '" + logical.raw_name +
            "' must be a 2-D int32 tensor");
    }

    const std::string raw_prefix =
        logical.raw_name.substr(0, logical.raw_name.size() -
                                      std::strlen(qweight_suffix));
    const std::string runtime_prefix =
        logical.runtime_name.substr(0, logical.runtime_name.size() -
                                          std::strlen(qweight_suffix));
    const std::string raw_qzeros = raw_prefix + ".qzeros";
    const std::string raw_scales = raw_prefix + ".scales";
    if (!loader.contains(raw_qzeros) || !loader.contains(raw_scales)) {
        throw std::runtime_error(
            "load schema: AWQ qweight '" + logical.raw_name +
            "' is missing qzeros/scales companions");
    }
    const TensorInfo& qzeros_info = loader.info(raw_qzeros);
    const TensorInfo& scale_info = loader.info(raw_scales);
    if (qzeros_info.dtype != DType::INT32 || qzeros_info.shape.size() != 2 ||
        scale_info.shape.size() != 2 ||
        !scale_dtype_supported_for_int4_dequant(scale_info.dtype)) {
        throw std::runtime_error(
            "load schema: AWQ qzeros/scales for '" + logical.raw_name +
            "' have unsupported dtype or rank");
    }

    const std::int64_t k_full = info.shape[0];
    const std::int64_t n_full = info.shape[1] * 8;
    if (k_full <= 0 || n_full <= 0 ||
        k_full % hf.quant_group_size != 0 ||
        k_full % 16 != 0 || n_full % 64 != 0) {
        throw std::runtime_error(
            "load schema: AWQ qweight '" + logical.raw_name +
            "' shape is incompatible with Marlin repack");
    }
    const std::int64_t groups = k_full / hf.quant_group_size;
    if (qzeros_info.shape != std::vector<std::int64_t>{groups, n_full / 8} ||
        scale_info.shape != std::vector<std::int64_t>{groups, n_full}) {
        throw std::runtime_error(
            "load schema: AWQ qzeros/scales shape does not match qweight for '" +
            logical.raw_name + "'");
    }

    const std::string canonical_w = runtime_prefix + ".weight";
    const OfflineInt4ShardPlan shard_plan =
        offline_int4_shard_plan_for(
            OfflineInt4Format::Awq, canonical_w, tp_size);
    add_awq_marlin_repack_ops(
        plan, hf, info, qzeros_info, scale_info,
        logical.raw_name, raw_qzeros, raw_scales,
        canonical_w, shard_plan, tp_size);

    consumed_raw.insert(logical.raw_name);
    consumed_raw.insert(raw_qzeros);
    consumed_raw.insert(raw_scales);
    return true;
}

bool try_add_gptq_dequant_weight(
    LoadPlan& plan,
    const HfConfig& hf,
    const TensorMetadataSource& loader,
    const LogicalTensor& logical,
    const TensorInfo& info,
    int tp_size,
    std::unordered_set<std::string>& consumed_raw)
{
    constexpr const char* qweight_suffix = ".qweight";
    if (!ends_with(logical.runtime_name, qweight_suffix)) return false;
    if (info.dtype != DType::INT32 || info.shape.size() != 2) {
        throw std::runtime_error(
            "load schema: GPTQ qweight '" + logical.raw_name +
            "' must be a 2-D int32 tensor");
    }

    const std::string raw_prefix =
        logical.raw_name.substr(0, logical.raw_name.size() -
                                      std::strlen(qweight_suffix));
    const std::string runtime_prefix =
        logical.runtime_name.substr(0, logical.runtime_name.size() -
                                          std::strlen(qweight_suffix));
    const std::string raw_qzeros = raw_prefix + ".qzeros";
    const std::string raw_scales = raw_prefix + ".scales";
    const std::string raw_gidx = raw_prefix + ".g_idx";
    if (!loader.contains(raw_qzeros) || !loader.contains(raw_scales)) {
        throw std::runtime_error(
            "load schema: GPTQ qweight '" + logical.raw_name +
            "' is missing qzeros/scales companions");
    }
    const TensorInfo& qzeros_info = loader.info(raw_qzeros);
    const TensorInfo& scale_info = loader.info(raw_scales);
    if (qzeros_info.dtype != DType::INT32 || qzeros_info.shape.size() != 2 ||
        scale_info.shape.size() != 2 ||
        !scale_dtype_supported_for_int4_dequant(scale_info.dtype)) {
        throw std::runtime_error(
            "load schema: GPTQ qzeros/scales for '" + logical.raw_name +
            "' have unsupported dtype or rank");
    }

    const std::int64_t k_full = info.shape[0] * 8;
    const std::int64_t n_full = info.shape[1];
    if (k_full <= 0 || n_full <= 0 ||
        k_full % hf.quant_group_size != 0 ||
        n_full % 8 != 0) {
        throw std::runtime_error(
            "load schema: GPTQ qweight '" + logical.raw_name +
            "' shape is incompatible with group_size=" +
            std::to_string(hf.quant_group_size));
    }
    const std::int64_t groups = k_full / hf.quant_group_size;
    if (qzeros_info.shape != std::vector<std::int64_t>{groups, n_full / 8} ||
        scale_info.shape != std::vector<std::int64_t>{groups, n_full}) {
        throw std::runtime_error(
            "load schema: GPTQ qzeros/scales shape does not match qweight for '" +
            logical.raw_name + "'");
    }

    const TensorInfo* gidx_info = nullptr;
    if (hf.quant_desc_act) {
        if (!loader.contains(raw_gidx)) {
            throw std::runtime_error(
                "load schema: GPTQ desc_act qweight '" + logical.raw_name +
                "' is missing g_idx companion");
        }
        gidx_info = &loader.info(raw_gidx);
        if (gidx_info->dtype != DType::INT32 ||
            gidx_info->shape != std::vector<std::int64_t>{k_full}) {
            throw std::runtime_error(
                "load schema: GPTQ g_idx shape does not match qweight for '" +
                logical.raw_name + "'");
        }
    }

    const std::string canonical_w = runtime_prefix + ".weight";
    OfflineInt4ShardPlan shard_plan =
        offline_int4_shard_plan_for(
            OfflineInt4Format::Gptq, canonical_w, tp_size);
    if (hf.quant_desc_act && shard_plan.canonical_axis == 1) {
        // Act-order row shards carry local qweight/g_idx, but g_idx values
        // are global group ids. Keep qzeros/scales unsharded so the generic
        // dequant kernel can dereference those global ids directly.
        shard_plan.qzeros_axis = -1;
        shard_plan.scale_axis = -1;
    }
    add_int4_dequant_ops(
        plan, hf, info, qzeros_info, scale_info, gidx_info,
        logical.raw_name, raw_qzeros, raw_scales, raw_gidx,
        canonical_w, OfflineInt4Format::Gptq, shard_plan, tp_size);

    consumed_raw.insert(logical.raw_name);
    consumed_raw.insert(raw_qzeros);
    consumed_raw.insert(raw_scales);
    if (loader.contains(raw_gidx)) {
        consumed_raw.insert(raw_gidx);
    }
    return true;
}

std::uint64_t tensor_nbytes(DType dtype, const std::vector<std::int64_t>& shape) {
    std::uint64_t numel = 1;
    for (const auto dim : shape) {
        numel *= static_cast<std::uint64_t>(dim);
    }
    return numel * static_cast<std::uint64_t>(dtype_bytes(dtype));
}

std::uint64_t tensor_nbytes(const TensorSpec& spec) {
    return tensor_nbytes(spec.dtype, spec.shape);
}

struct PerExpertMoeName {
    std::string base;
    int expert = -1;
    std::string projection;
};

bool try_parse_per_expert_moe_weight(
    const std::string& name,
    PerExpertMoeName& out)
{
    const std::string marker = ".experts.";
    const auto marker_pos = name.find(marker);
    if (marker_pos == std::string::npos) return false;
    const std::size_t idx_begin = marker_pos + marker.size();
    const auto idx_end = name.find('.', idx_begin);
    if (idx_end == std::string::npos || idx_end == idx_begin) return false;
    const std::string idx_text = name.substr(idx_begin, idx_end - idx_begin);
    if (!std::all_of(idx_text.begin(), idx_text.end(), [](unsigned char ch) {
            return ch >= '0' && ch <= '9';
        })) {
        return false;
    }

    const std::string suffix = name.substr(idx_end);
    std::string projection;
    if (suffix == ".gate_proj.weight") {
        projection = "gate";
    } else if (suffix == ".up_proj.weight") {
        projection = "up";
    } else if (suffix == ".down_proj.weight") {
        projection = "down";
    } else {
        return false;
    }

    out.base = name.substr(0, marker_pos);
    out.expert = std::stoi(idx_text);
    out.projection = std::move(projection);
    return true;
}

std::string per_expert_raw_name(
    const std::string& base,
    int expert,
    const char* projection)
{
    return base + ".experts." + std::to_string(expert) + "." +
           projection + "_proj.weight";
}

bool try_add_per_expert_moe_fusion(
    LoadPlan& plan,
    const HfConfig& hf,
    const TensorMetadataSource& loader,
    const LogicalTensorGroup& group,
    std::unordered_set<std::string>& consumed_raw,
    int tp_size)
{
    if (group.kind != LogicalTensorGroupKind::PerExpertMoe) return false;
    if (group.raw_names.empty() || group.raw_names.size() % 3 != 0 ||
        group.raw_roles.size() != group.raw_names.size() ||
        group.runtime_names.size() != 2) {
        throw std::runtime_error(
            "load schema: per-expert MoE group has wrong arity at '" +
            group.runtime_base + "'");
    }
    const std::string& raw_gate0 = group.raw_names[0];
    if (consumed_raw.contains(raw_gate0)) return true;

    int expert_count = static_cast<int>(group.raw_names.size() / 3);
    if (expert_count <= 0) {
        throw std::runtime_error(
            "load schema: could not determine MoE expert count for '" +
            group.runtime_base + "'");
    }
    if (hf.num_experts > 0 && hf.num_experts != expert_count) {
        throw std::runtime_error(
            "load schema: per-expert MoE group count disagrees with config at '" +
            group.runtime_base + "'");
    }

    std::vector<TensorSourceRef> fuse_sources;
    fuse_sources.reserve(static_cast<std::size_t>(expert_count) * 3);

    std::vector<std::int64_t> gate_shape0;
    std::vector<std::int64_t> down_shape0;
    for (int e = 0; e < expert_count; ++e) {
        const std::size_t base = static_cast<std::size_t>(e) * 3;
        if (group.raw_roles[base] != LogicalTensorRole::MoeExpertGate ||
            group.raw_roles[base + 1] != LogicalTensorRole::MoeExpertUp ||
            group.raw_roles[base + 2] != LogicalTensorRole::MoeExpertDown) {
            throw std::runtime_error(
                "load schema: per-expert MoE source roles are not gate/up/down "
                "triples at '" + group.runtime_base + "'");
        }
        const std::string& raw_gate = group.raw_names[base];
        const std::string& raw_up = group.raw_names[base + 1];
        const std::string& raw_down = group.raw_names[base + 2];
        if (!loader.contains(raw_gate) ||
            !loader.contains(raw_up) ||
            !loader.contains(raw_down)) {
            throw std::runtime_error(
                "load schema: incomplete per-expert MoE group under '" +
                group.runtime_base + "' at expert " + std::to_string(e));
        }

        const TensorInfo& gate_info = loader.info(raw_gate);
        const TensorInfo& up_info = loader.info(raw_up);
        const TensorInfo& down_info = loader.info(raw_down);
        if (gate_info.shape.size() != 2 ||
            up_info.shape.size() != 2 ||
            down_info.shape.size() != 2) {
            throw std::runtime_error(
                "load schema: per-expert MoE fusion expects 2-D expert "
                "weights under '" + group.runtime_base + "'");
        }

        const auto local_gate_shape =
            sharded_shape(gate_info.shape, 0, tp_size, raw_gate);
        const auto local_up_shape =
            sharded_shape(up_info.shape, 0, tp_size, raw_up);
        const auto local_down_shape =
            sharded_shape(down_info.shape, 1, tp_size, raw_down);
        if (local_gate_shape != local_up_shape ||
            local_down_shape[0] != local_gate_shape[1] ||
            local_down_shape[1] != local_gate_shape[0]) {
            throw std::runtime_error(
                "load schema: per-expert MoE gate/up/down shapes do not "
                "align under '" + group.runtime_base + "'");
        }
        if (e == 0) {
            gate_shape0 = local_gate_shape;
            down_shape0 = local_down_shape;
        } else if (local_gate_shape != gate_shape0 ||
                   local_down_shape != down_shape0) {
            throw std::runtime_error(
                "load schema: per-expert MoE shapes are not uniform under '" +
                group.runtime_base + "'");
        }

        if (gate_info.dtype != DType::BF16 ||
            up_info.dtype != DType::BF16 ||
            down_info.dtype != DType::BF16) {
            throw std::runtime_error(
                "load schema: direct MoE expert fusion currently requires "
                "bf16 sources under '" + group.runtime_base + "'");
        }

        fuse_sources.push_back({raw_gate, "gate"});
        fuse_sources.push_back({raw_up, "up"});
        fuse_sources.push_back({raw_down, "down"});

        consumed_raw.insert(raw_gate);
        consumed_raw.insert(raw_up);
        consumed_raw.insert(raw_down);
    }

    const std::string& gate_up_name =
        group_runtime_name_for_role(group, LogicalTensorRole::MoeExpertsGateUp);
    const std::string& down_name =
        group_runtime_name_for_role(group, LogicalTensorRole::MoeExpertsDown);
    const std::int64_t E = expert_count;
    register_tensor_spec(
        plan, gate_up_name, DType::BF16,
        {E, 2 * gate_shape0[0], gate_shape0[1]},
        TensorLayoutKind::Grouped, TensorOwnershipKind::Owned,
        TensorParallelKind::Expert);
    register_tensor_spec(
        plan, down_name, DType::BF16,
        {E, down_shape0[0], down_shape0[1]},
        TensorLayoutKind::Grouped, TensorOwnershipKind::Owned,
        TensorParallelKind::Expert);

    plan.ops.push_back(make_stack_groups_op(
        gate_up_name, down_name, {}, std::move(fuse_sources)));
    return true;
}

bool temporary_tensor_bytes(
    const std::unordered_map<std::string, std::uint64_t>& bytes,
    const std::string& name,
    std::uint64_t& out)
{
    const auto it = bytes.find(name);
    if (it == bytes.end()) return false;
    out = it->second;
    return true;
}

void finalize_memory_plan(LoadPlan& plan) {
    const std::uint64_t conservative_temp_floor =
        plan.memory.max_temporary_bytes;

    std::uint64_t persistent_bytes = 0;
    std::unordered_map<std::string, std::uint64_t> temp_bytes;
    temp_bytes.reserve(plan.tensors.size());
    for (const auto& [name, spec] : plan.tensors) {
        if (spec.ownership == TensorOwnershipKind::Owned) {
            persistent_bytes += tensor_nbytes(spec);
        } else if (spec.ownership == TensorOwnershipKind::Temporary) {
            temp_bytes.emplace(name, tensor_nbytes(spec));
        }
    }

    std::unordered_map<std::string, std::size_t> last_use;
    last_use.reserve(temp_bytes.size());
    for (std::size_t i = 0; i < plan.ops.size(); ++i) {
        const LoadOp& op = plan.ops[i];
        for (const auto& input : load_op_inputs(op)) {
            if (temp_bytes.contains(input)) {
                last_use[input] = i;
            }
        }
    }

    std::uint64_t live_temp_bytes = 0;
    std::uint64_t high_water_temp_bytes = 0;
    auto add_if_temporary = [&](const std::string& name) {
        std::uint64_t bytes = 0;
        if (temporary_tensor_bytes(temp_bytes, name, bytes)) {
            live_temp_bytes += bytes;
            high_water_temp_bytes =
                std::max(high_water_temp_bytes, live_temp_bytes);
        }
    };
    auto release_if_last_use = [&](const std::string& name, std::size_t op_idx) {
        const auto use_it = last_use.find(name);
        if (use_it == last_use.end() || use_it->second != op_idx) return;
        std::uint64_t bytes = 0;
        if (temporary_tensor_bytes(temp_bytes, name, bytes)) {
            live_temp_bytes -= std::min(live_temp_bytes, bytes);
        }
    };

    for (std::size_t i = 0; i < plan.ops.size(); ++i) {
        const LoadOp& op = plan.ops[i];
        add_if_temporary(load_op_output(op));
        add_if_temporary(load_op_secondary_output(op));
        for (const auto& input : load_op_inputs(op)) {
            release_if_last_use(input, i);
        }
    }

    plan.memory.persistent_bytes = persistent_bytes;
    plan.memory.max_temporary_bytes =
        std::max(conservative_temp_floor, high_water_temp_bytes);
    plan.memory.estimated_peak_bytes =
        plan.memory.persistent_bytes + plan.memory.max_temporary_bytes;
}

enum class Mxfp4ExpertProjection {
    GateUpInterleaved,
    Down,
};

struct Mxfp4ExpertGroup {
    Mxfp4ExpertProjection projection = Mxfp4ExpertProjection::Down;
    std::string raw_base;
    std::string runtime_base;
    std::string stem;
    std::string raw_blocks;
    std::string raw_scales;
    std::string raw_bias;
    std::string group_prefix;
    TensorInfo blocks;
    TensorInfo scales;
    TensorInfo bias;
    std::int64_t experts = 0;
    std::int64_t out_dim = 0;
    std::int64_t in_dim = 0;
    int shard_axis = -1;

    bool is_gate_up() const noexcept {
        return projection == Mxfp4ExpertProjection::GateUpInterleaved;
    }

    std::vector<std::int64_t> full_bf16_shape() const {
        return {experts, out_dim, in_dim};
    }
};

void push_copy_op(
    LoadPlan& plan,
    const std::string& raw_name,
    const std::string& output_name,
    int shard_axis = -1)
{
    plan.ops.push_back(make_raw_load_op(
        LoadOpKind::Copy, output_name, raw_name, shard_axis));
}

void push_dequant_op(
    LoadPlan& plan,
    const std::string& output_name,
    const std::string& blocks_name,
    const std::string& scales_name)
{
    plan.ops.push_back(make_tensor_op(
        LoadOpKind::Dequantize, output_name, {blocks_name, scales_name}));
}

void push_deinterleave_op(
    LoadPlan& plan,
    const std::string& first_output,
    const std::string& second_output,
    const std::string& input,
    int shard_axis)
{
    plan.ops.push_back(make_tensor_op(
        LoadOpKind::Deinterleave, first_output, {input},
        second_output, shard_axis));
}

void push_drop_op(
    LoadPlan& plan,
    const std::string& output_name,
    std::vector<std::string> inputs)
{
    plan.ops.push_back(make_tensor_op(
        LoadOpKind::Drop, output_name, std::move(inputs)));
}

bool try_describe_gpt_oss_mxfp4_group(
    const HfConfig& hf,
    const TensorMetadataSource& loader,
    const LogicalTensorGroup& logical_group,
    int tp_size,
    Mxfp4ExpertGroup& out)
{
    if (hf.model_type != "gpt_oss" || hf.quant_method != "mxfp4") return false;
    if (logical_group.kind != LogicalTensorGroupKind::GptOssMxfp4) return false;
    if (logical_group.raw_names.size() != 3 ||
        logical_group.runtime_names.size() != 3 ||
        logical_group.raw_roles.size() != 3 ||
        logical_group.runtime_roles.size() != 3 ||
        logical_group.raw_roles[0] != LogicalTensorRole::QuantPackedData ||
        logical_group.raw_roles[1] != LogicalTensorRole::QuantScale ||
        logical_group.raw_roles[2] != LogicalTensorRole::Bias) {
        throw std::runtime_error(
            "load schema: GPT-OSS MXFP4 group has wrong declaration at '" +
            logical_group.runtime_base + "'");
    }

    const bool is_gate_up =
        logical_group.runtime_base.find(".gate_up_proj") != std::string::npos;
    const bool is_down =
        logical_group.runtime_base.find(".down_proj") != std::string::npos;
    if (!is_gate_up && !is_down) {
        throw std::runtime_error(
            "load schema: GPT-OSS MXFP4 group has unknown projection at '" +
            logical_group.runtime_base + "'");
    }

    Mxfp4ExpertGroup group;
    group.projection = is_gate_up
        ? Mxfp4ExpertProjection::GateUpInterleaved
        : Mxfp4ExpertProjection::Down;
    group.raw_base = logical_group.raw_names[0].substr(
        0, logical_group.raw_names[0].size() - std::strlen("_blocks"));
    const std::string marker =
        is_gate_up ? ".mlp.experts.gate_up_proj" : ".mlp.experts.down_proj";
    const auto marker_pos = logical_group.runtime_base.rfind(marker);
    if (marker_pos == std::string::npos) {
        throw std::runtime_error(
            "load schema: GPT-OSS MXFP4 group base is inconsistent at '" +
            logical_group.runtime_base + "'");
    }
    group.runtime_base = logical_group.runtime_base.substr(0, marker_pos);
    group.stem = is_gate_up ? "gate_up_proj" : "down_proj";
    group.raw_blocks = logical_group.raw_names[0];
    group.raw_scales = logical_group.raw_names[1];
    group.raw_bias = logical_group.raw_names[2];
    group.group_prefix = logical_group.runtime_base;

    if (!loader.contains(group.raw_blocks) ||
        !loader.contains(group.raw_scales) ||
        !loader.contains(group.raw_bias)) {
        throw std::runtime_error(
            "load schema: GPT-OSS MXFP4 expert tensor group is incomplete at '" +
            group.raw_base + ".mlp.experts." + group.stem + "'");
    }

    group.blocks = loader.info(group.raw_blocks);
    group.scales = loader.info(group.raw_scales);
    group.bias = loader.info(group.raw_bias);
    if (group.blocks.dtype != DType::UINT8 ||
        group.scales.dtype != DType::UINT8 ||
        group.bias.dtype != DType::BF16) {
        throw std::runtime_error(
            "load schema: GPT-OSS MXFP4 group has unexpected dtypes at '" +
            group.raw_base + ".mlp.experts." + group.stem + "'");
    }
    if (group.blocks.shape.size() != 4 || group.scales.shape.size() != 3 ||
        group.blocks.shape[0] != group.scales.shape[0] ||
        group.blocks.shape[1] != group.scales.shape[1] ||
        group.blocks.shape[2] != group.scales.shape[2] ||
        group.blocks.shape[3] != 16) {
        throw std::runtime_error(
            "load schema: GPT-OSS MXFP4 packed/scales shape mismatch at '" +
            group.raw_blocks + "'");
    }

    group.experts = group.blocks.shape[0];
    group.out_dim = group.blocks.shape[1];
    group.in_dim = group.blocks.shape[2] * 32;
    group.shard_axis = (tp_size > 1) ? (group.is_gate_up() ? 1 : 2) : -1;
    out = std::move(group);
    return true;
}

void validate_gpt_oss_mxfp4_runtime_shape(
    const HfConfig& hf,
    const Mxfp4ExpertGroup& group,
    int tp_size,
    const char* lowering_name)
{
    if (group.is_gate_up()) {
        if (group.out_dim % 2 != 0 || (group.out_dim / 2) % tp_size != 0) {
            throw std::runtime_error(
                "load schema: GPT-OSS " + std::string(lowering_name) +
                " gate/up intermediate axis is not divisible by tp_size at '" +
                group.raw_blocks + "'");
        }
        if (group.bias.shape !=
            std::vector<std::int64_t>{group.experts, group.out_dim}) {
            throw std::runtime_error(
                "load schema: GPT-OSS gate/up bias shape mismatch at '" +
                group.raw_bias + "'");
        }
    } else {
        if (group.out_dim != hf.hidden_size || group.in_dim % tp_size != 0) {
            throw std::runtime_error(
                "load schema: GPT-OSS " + std::string(lowering_name) +
                " down projection shape cannot be tensor-parallel sharded at '" +
                group.raw_blocks + "'");
        }
        if (group.bias.shape !=
            std::vector<std::int64_t>{group.experts, group.out_dim}) {
            throw std::runtime_error(
                "load schema: GPT-OSS down bias shape mismatch at '" +
                group.raw_bias + "'");
        }
    }
}

void lower_mxfp4_expert_group_routed_dequant(
    LoadPlan& plan,
    const HfConfig& hf,
    const Mxfp4ExpertGroup& group,
    int tp_size)
{
    validate_gpt_oss_mxfp4_runtime_shape(
        hf, group, tp_size, "routed MXFP4 dequant");

    const std::string weight_name = group.group_prefix + ".weight";
    const std::string scale_name = group.group_prefix + ".weight_scale";
    const std::string bias_name = group.group_prefix + ".bias";
    const int bias_shard_axis = (tp_size > 1 && group.is_gate_up()) ? 1 : -1;

    QuantSpec quant;
    quant.format = QuantFormat::Mxfp4E2M1E8M0;
    quant.granularity = QuantGranularity::PerGroup;
    quant.group_size = 32;
    quant.channel_axis = 2;
    quant.scale_tensor = scale_name;

    register_tensor_spec(
        plan, weight_name, DType::UINT8,
        sharded_shape(group.blocks.shape, group.shard_axis, tp_size, weight_name),
        TensorLayoutKind::QuantPacked, TensorOwnershipKind::Owned,
        TensorParallelKind::Expert, /*backing_tensor=*/{}, std::move(quant));
    register_tensor_spec(
        plan, scale_name, DType::UINT8,
        sharded_shape(group.scales.shape, group.shard_axis, tp_size, scale_name),
        TensorLayoutKind::Dense, TensorOwnershipKind::Owned,
        TensorParallelKind::Expert);
    register_tensor_spec(
        plan, bias_name, DType::BF16,
        sharded_shape(group.bias.shape, bias_shard_axis, tp_size, bias_name),
        TensorLayoutKind::Grouped, TensorOwnershipKind::Owned,
        TensorParallelKind::Expert);

    push_copy_op(plan, group.raw_blocks, weight_name, group.shard_axis);
    push_copy_op(plan, group.raw_scales, scale_name, group.shard_axis);
    push_copy_op(plan, group.raw_bias, bias_name, bias_shard_axis);

    plan.ops.push_back(make_tensor_op(
        LoadOpKind::BindMetadata, weight_name));
}

void lower_mxfp4_gate_up_group_to_bf16(
    LoadPlan& plan,
    const Mxfp4ExpertGroup& group,
    int tp_size,
    const std::string& blocks_tmp,
    const std::string& scales_tmp)
{
    const std::vector<std::int64_t> full_bf16_shape = group.full_bf16_shape();
    const std::int64_t I_local = (group.out_dim / 2) / tp_size;
    const std::vector<std::int64_t> expert_w_shape =
        {group.experts, I_local, group.in_dim};
    const std::vector<std::int64_t> expert_b_shape =
        {group.experts, I_local};
    const std::string gate_w =
        group.runtime_base + ".mlp.experts.gate_proj.weight";
    const std::string up_w =
        group.runtime_base + ".mlp.experts.up_proj.weight";
    const std::string gate_b =
        group.runtime_base + ".mlp.experts.gate_proj.bias";
    const std::string up_b =
        group.runtime_base + ".mlp.experts.up_proj.bias";
    const std::string bf16_tmp = group.group_prefix + ".__mxfp4_bf16";

    register_tensor_spec(
        plan, gate_w, DType::BF16, expert_w_shape,
        TensorLayoutKind::Grouped, TensorOwnershipKind::Owned,
        TensorParallelKind::Expert);
    register_tensor_spec(
        plan, up_w, DType::BF16, expert_w_shape,
        TensorLayoutKind::Grouped, TensorOwnershipKind::Owned,
        TensorParallelKind::Expert);
    register_tensor_spec(
        plan, bf16_tmp, DType::BF16, full_bf16_shape,
        TensorLayoutKind::Grouped, TensorOwnershipKind::Temporary,
        TensorParallelKind::Expert);

    push_dequant_op(plan, bf16_tmp, blocks_tmp, scales_tmp);
    push_deinterleave_op(
        plan, gate_w, up_w, bf16_tmp, group.shard_axis);

    const std::string bias_tmp = group.group_prefix + ".__gate_up_bias";
    push_copy_op(plan, group.raw_bias, bias_tmp);
    register_tensor_spec(
        plan, bias_tmp, group.bias.dtype, group.bias.shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
        TensorParallelKind::Expert);
    register_tensor_spec(
        plan, gate_b, DType::BF16, expert_b_shape,
        TensorLayoutKind::Grouped, TensorOwnershipKind::Owned,
        TensorParallelKind::Expert);
    register_tensor_spec(
        plan, up_b, DType::BF16, expert_b_shape,
        TensorLayoutKind::Grouped, TensorOwnershipKind::Owned,
        TensorParallelKind::Expert);
    push_deinterleave_op(
        plan, gate_b, up_b, bias_tmp, group.shard_axis);

    push_drop_op(
        plan, group.group_prefix + ".__drop",
        {blocks_tmp, scales_tmp, bf16_tmp, bias_tmp});
    estimate_temporary_bytes(
        plan,
        group.blocks.nbytes + group.scales.nbytes + group.bias.nbytes +
        tensor_nbytes(DType::BF16, full_bf16_shape));
}

void lower_mxfp4_down_group_to_bf16(
    LoadPlan& plan,
    const Mxfp4ExpertGroup& group,
    int tp_size,
    const std::string& blocks_tmp,
    const std::string& scales_tmp)
{
    const std::vector<std::int64_t> full_bf16_shape = group.full_bf16_shape();
    const std::int64_t I_local = group.in_dim / tp_size;
    const std::string down_w =
        group.runtime_base + ".mlp.experts.down_proj.weight";
    const std::string down_b =
        group.runtime_base + ".mlp.experts.down_proj.bias";

    register_tensor_spec(
        plan, down_w, DType::BF16, {group.experts, group.out_dim, I_local},
        TensorLayoutKind::Grouped, TensorOwnershipKind::Owned,
        TensorParallelKind::Expert);
    if (tp_size == 1) {
        push_dequant_op(plan, down_w, blocks_tmp, scales_tmp);
    } else {
        const std::string bf16_tmp = group.group_prefix + ".__mxfp4_bf16";
        register_tensor_spec(
            plan, bf16_tmp, DType::BF16, full_bf16_shape,
            TensorLayoutKind::Grouped, TensorOwnershipKind::Temporary,
            TensorParallelKind::Expert);
        push_dequant_op(plan, bf16_tmp, blocks_tmp, scales_tmp);

        plan.ops.push_back(make_slice_op(
            down_w, bf16_tmp, /*slice_axis=*/2,
            /*slice_start=*/0, I_local, group.shard_axis));
    }

    push_copy_op(plan, group.raw_bias, down_b);
    register_tensor_spec(
        plan, down_b, DType::BF16, group.bias.shape,
        TensorLayoutKind::Grouped, TensorOwnershipKind::Owned,
        TensorParallelKind::Expert);

    std::vector<std::string> drop_inputs = {blocks_tmp, scales_tmp};
    if (tp_size > 1) {
        drop_inputs.push_back(group.group_prefix + ".__mxfp4_bf16");
    }
    push_drop_op(plan, group.group_prefix + ".__drop", std::move(drop_inputs));
    estimate_temporary_bytes(
        plan,
        group.blocks.nbytes + group.scales.nbytes +
        (tp_size > 1 ? tensor_nbytes(DType::BF16, full_bf16_shape) : 0));
}

void lower_mxfp4_expert_group_to_bf16(
    LoadPlan& plan,
    const HfConfig& hf,
    const Mxfp4ExpertGroup& group,
    int tp_size)
{
    validate_gpt_oss_mxfp4_runtime_shape(
        hf, group, tp_size, "BF16 fallback");

    const std::string blocks_tmp = group.group_prefix + ".__mxfp4_blocks";
    const std::string scales_tmp = group.group_prefix + ".__mxfp4_scales";
    push_copy_op(plan, group.raw_blocks, blocks_tmp);
    push_copy_op(plan, group.raw_scales, scales_tmp);
    register_tensor_spec(
        plan, blocks_tmp, group.blocks.dtype, group.blocks.shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
        TensorParallelKind::Expert);
    register_tensor_spec(
        plan, scales_tmp, group.scales.dtype, group.scales.shape,
        TensorLayoutKind::Dense, TensorOwnershipKind::Temporary,
        TensorParallelKind::Expert);

    if (group.is_gate_up()) {
        lower_mxfp4_gate_up_group_to_bf16(
            plan, group, tp_size, blocks_tmp, scales_tmp);
    } else {
        lower_mxfp4_down_group_to_bf16(
            plan, group, tp_size, blocks_tmp, scales_tmp);
    }
}

void mark_mxfp4_group_consumed(
    const Mxfp4ExpertGroup& group,
    std::unordered_set<std::string>& consumed_raw)
{
    consumed_raw.insert(group.raw_blocks);
    consumed_raw.insert(group.raw_scales);
    consumed_raw.insert(group.raw_bias);
}

bool try_add_gpt_oss_mxfp4_expert(
    LoadPlan& plan,
    const HfConfig& hf,
    const TensorMetadataSource& loader,
    const LogicalTensorGroup& logical_group,
    std::unordered_set<std::string>& consumed_raw,
    int tp_size,
    Mxfp4MoeLowering lowering)
{
    Mxfp4ExpertGroup group;
    if (!try_describe_gpt_oss_mxfp4_group(
            hf, loader, logical_group, tp_size, group)) {
        return false;
    }
    if (consumed_raw.contains(group.raw_blocks)) return true;

    if (lowering == Mxfp4MoeLowering::RoutedDequant) {
        lower_mxfp4_expert_group_routed_dequant(plan, hf, group, tp_size);
    } else if (lowering == Mxfp4MoeLowering::NativeGemm) {
        throw std::runtime_error(
            "load schema: GPT-OSS MXFP4 native GEMM target was selected, "
            "but no native MXFP4 MoE GEMM backend is registered yet");
    } else {
        lower_mxfp4_expert_group_to_bf16(plan, hf, group, tp_size);
    }

    mark_mxfp4_group_consumed(group, consumed_raw);
    return true;
}

}  // namespace

LogicalTensorRole infer_logical_tensor_role(const std::string& name) {
    if (ends_with(name, ".embed_tokens.weight")) {
        return LogicalTensorRole::Embedding;
    }
    if (name == "lm_head.weight" || ends_with(name, ".lm_head.weight")) {
        return LogicalTensorRole::LmHead;
    }
    if (ends_with(name, ".input_layernorm.weight") ||
        ends_with(name, ".post_attention_layernorm.weight") ||
        ends_with(name, ".norm.weight") ||
        ends_with(name, ".q_norm.weight") ||
        ends_with(name, ".k_norm.weight")) {
        return LogicalTensorRole::Norm;
    }
    if (ends_with(name, ".self_attn.q_proj.weight")) {
        return LogicalTensorRole::AttentionQ;
    }
    if (ends_with(name, ".self_attn.k_proj.weight")) {
        return LogicalTensorRole::AttentionK;
    }
    if (ends_with(name, ".self_attn.v_proj.weight")) {
        return LogicalTensorRole::AttentionV;
    }
    if (ends_with(name, ".self_attn.o_proj.weight")) {
        return LogicalTensorRole::AttentionO;
    }
    if (ends_with(name, ".self_attn.qkv_proj.weight")) {
        return LogicalTensorRole::AttentionQkv;
    }
    if (ends_with(name, ".mlp.gate_proj.weight")) {
        return LogicalTensorRole::MlpGate;
    }
    if (ends_with(name, ".mlp.up_proj.weight")) {
        return LogicalTensorRole::MlpUp;
    }
    if (ends_with(name, ".mlp.down_proj.weight")) {
        return LogicalTensorRole::MlpDown;
    }
    if (ends_with(name, ".mlp.gate_up_proj.weight")) {
        return LogicalTensorRole::MlpGateUp;
    }
    if (ends_with(name, ".experts.gate_up_proj")) {
        return LogicalTensorRole::MoeExpertsGateUp;
    }
    if (ends_with(name, ".experts.down_proj")) {
        return LogicalTensorRole::MoeExpertsDown;
    }
    if (ends_with(name, "_blocks")) {
        return LogicalTensorRole::QuantPackedData;
    }
    if (ends_with(name, "_bias") || ends_with(name, ".bias")) {
        return LogicalTensorRole::Bias;
    }
    if (ends_with(name, ".gate_proj.weight")) {
        return LogicalTensorRole::MoeExpertGate;
    }
    if (ends_with(name, ".up_proj.weight")) {
        return LogicalTensorRole::MoeExpertUp;
    }
    if (ends_with(name, ".down_proj.weight")) {
        return LogicalTensorRole::MoeExpertDown;
    }
    if (ends_with(name, ".w1.weight")) {
        return LogicalTensorRole::MoeExpertGate;
    }
    if (ends_with(name, ".w3.weight")) {
        return LogicalTensorRole::MoeExpertUp;
    }
    if (ends_with(name, ".w2.weight")) {
        return LogicalTensorRole::MoeExpertDown;
    }
    if (ends_with(name, ".weight_scale") ||
        ends_with(name, ".weight_scale_inv") ||
        ends_with(name, ".scales")) {
        return LogicalTensorRole::QuantScale;
    }
    if (ends_with(name, ".weight_zero_point") ||
        ends_with(name, ".qzeros") ||
        ends_with(name, ".zero_point")) {
        return LogicalTensorRole::QuantZeroPoint;
    }
    return LogicalTensorRole::Unknown;
}

namespace {

void add_logical_group_once(
    LogicalTensorGraph& graph,
    std::unordered_set<std::string>& seen,
    LogicalTensorGroupKind kind,
    std::string runtime_base,
    std::vector<std::string> raw_names,
    std::vector<std::string> runtime_names,
    std::vector<LogicalTensorRole> raw_roles = {},
    std::vector<LogicalTensorRole> runtime_roles = {})
{
    const std::string key =
        std::to_string(static_cast<int>(kind)) + ":" + runtime_base;
    if (!seen.insert(key).second) return;
    graph.groups.push_back(LogicalTensorGroup{
        .kind = kind,
        .runtime_base = std::move(runtime_base),
        .raw_names = std::move(raw_names),
        .runtime_names = std::move(runtime_names),
        .raw_roles = std::move(raw_roles),
        .runtime_roles = std::move(runtime_roles),
    });
}

void discover_logical_tensor_groups(
    LogicalTensorGraph& graph,
    const HfConfig& hf,
    const TensorMetadataSource& loader)
{
    std::unordered_set<std::string> seen;
    seen.reserve(graph.tensors.size());

    for (const auto& logical : graph.tensors) {
        constexpr const char* q_suffix = ".self_attn.q_proj.weight";
        constexpr const char* k_suffix = ".self_attn.k_proj.weight";
        constexpr const char* v_suffix = ".self_attn.v_proj.weight";
        constexpr const char* gate_suffix = ".mlp.gate_proj.weight";
        constexpr const char* up_suffix = ".mlp.up_proj.weight";

        if (ends_with(logical.runtime_name, q_suffix) ||
            ends_with(logical.runtime_name, k_suffix) ||
            ends_with(logical.runtime_name, v_suffix)) {
            const char* matched = ends_with(logical.runtime_name, q_suffix)
                ? q_suffix
                : (ends_with(logical.runtime_name, k_suffix) ? k_suffix : v_suffix);
            const std::string raw_base =
                logical.raw_name.substr(
                    0, logical.raw_name.size() - std::strlen(matched));
            const std::string runtime_base =
                logical.runtime_name.substr(
                    0, logical.runtime_name.size() - std::strlen(matched));
            const std::vector<std::string> raw_names = {
                raw_base + q_suffix,
                raw_base + k_suffix,
                raw_base + v_suffix,
            };
            if (can_pack_2d_bf16_group(loader, raw_names)) {
                add_logical_group_once(
                    graph, seen, LogicalTensorGroupKind::PackedQkv,
                    runtime_base + ".self_attn", raw_names,
                    {runtime_base + q_suffix,
                     runtime_base + k_suffix,
                     runtime_base + v_suffix},
                    {LogicalTensorRole::AttentionQ,
                     LogicalTensorRole::AttentionK,
                     LogicalTensorRole::AttentionV},
                    {LogicalTensorRole::AttentionQ,
                     LogicalTensorRole::AttentionK,
                     LogicalTensorRole::AttentionV});
            }
        }

        if (ends_with(logical.runtime_name, gate_suffix) ||
            ends_with(logical.runtime_name, up_suffix)) {
            const char* matched = ends_with(logical.runtime_name, gate_suffix)
                ? gate_suffix
                : up_suffix;
            const std::string raw_base =
                logical.raw_name.substr(
                    0, logical.raw_name.size() - std::strlen(matched));
            const std::string runtime_base =
                logical.runtime_name.substr(
                    0, logical.runtime_name.size() - std::strlen(matched));
            const std::vector<std::string> raw_names = {
                raw_base + gate_suffix,
                raw_base + up_suffix,
            };
            if (can_pack_2d_bf16_group(loader, raw_names)) {
                add_logical_group_once(
                    graph, seen, LogicalTensorGroupKind::PackedGateUp,
                    runtime_base + ".mlp", raw_names,
                    {runtime_base + gate_suffix,
                     runtime_base + up_suffix},
                    {LogicalTensorRole::MlpGate,
                     LogicalTensorRole::MlpUp},
                    {LogicalTensorRole::MlpGate,
                     LogicalTensorRole::MlpUp});
            }
        }

        if (logical.role == LogicalTensorRole::AttentionQkv) {
            constexpr const char* qkv_suffix = ".self_attn.qkv_proj.weight";
            if (ends_with(logical.runtime_name, qkv_suffix)) {
                const std::string runtime_base =
                    logical.runtime_name.substr(
                        0, logical.runtime_name.size() - std::strlen(qkv_suffix));
                add_logical_group_once(
                    graph, seen, LogicalTensorGroupKind::RowRangeSplit,
                    logical.runtime_name,
                    {logical.raw_name},
                    {runtime_base + ".self_attn.q_proj.weight",
                     runtime_base + ".self_attn.k_proj.weight",
                     runtime_base + ".self_attn.v_proj.weight"},
                    {LogicalTensorRole::AttentionQkv},
                    {LogicalTensorRole::AttentionQ,
                     LogicalTensorRole::AttentionK,
                     LogicalTensorRole::AttentionV});
            }
        }

        if (logical.role == LogicalTensorRole::MlpGateUp) {
            constexpr const char* gate_up_suffix = ".mlp.gate_up_proj.weight";
            if (ends_with(logical.runtime_name, gate_up_suffix)) {
                const std::string runtime_base =
                    logical.runtime_name.substr(
                        0, logical.runtime_name.size() -
                               std::strlen(gate_up_suffix));
                add_logical_group_once(
                    graph, seen, LogicalTensorGroupKind::RowRangeSplit,
                    logical.runtime_name,
                    {logical.raw_name},
                    {runtime_base + ".mlp.gate_proj.weight",
                     runtime_base + ".mlp.up_proj.weight"},
                    {LogicalTensorRole::MlpGateUp},
                    {LogicalTensorRole::MlpGate,
                     LogicalTensorRole::MlpUp});
            }
        }

        PerExpertMoeName expert;
        PerExpertMoeName raw_expert;
        if (try_parse_per_expert_moe_weight(logical.runtime_name, expert) &&
            try_parse_per_expert_moe_weight(logical.raw_name, raw_expert)) {
            int expert_count = hf.num_experts;
            if (expert_count <= 0) {
                expert_count = 0;
                for (;;) {
                    const std::string gate =
                        per_expert_raw_name(raw_expert.base, expert_count, "gate");
                    const std::string up =
                        per_expert_raw_name(raw_expert.base, expert_count, "up");
                    const std::string down =
                        per_expert_raw_name(raw_expert.base, expert_count, "down");
                    if (!loader.contains(gate) ||
                        !loader.contains(up) ||
                        !loader.contains(down)) {
                        break;
                    }
                    ++expert_count;
                }
            }
            std::vector<std::string> raw_names;
            std::vector<LogicalTensorRole> raw_roles;
            raw_names.reserve(static_cast<std::size_t>(std::max(expert_count, 0)) * 3);
            raw_roles.reserve(raw_names.capacity());
            for (int e = 0; e < expert_count; ++e) {
                raw_names.push_back(per_expert_raw_name(raw_expert.base, e, "gate"));
                raw_roles.push_back(LogicalTensorRole::MoeExpertGate);
                raw_names.push_back(per_expert_raw_name(raw_expert.base, e, "up"));
                raw_roles.push_back(LogicalTensorRole::MoeExpertUp);
                raw_names.push_back(per_expert_raw_name(raw_expert.base, e, "down"));
                raw_roles.push_back(LogicalTensorRole::MoeExpertDown);
            }
            add_logical_group_once(
                graph, seen, LogicalTensorGroupKind::PerExpertMoe,
                expert.base + ".experts", std::move(raw_names),
                {expert.base + ".experts.gate_up_proj",
                 expert.base + ".experts.down_proj"},
                std::move(raw_roles),
                {LogicalTensorRole::MoeExpertsGateUp,
                 LogicalTensorRole::MoeExpertsDown});
        }

        if (ends_with(logical.runtime_name, ".experts.gate_up_proj") ||
            ends_with(logical.runtime_name, ".experts.down_proj")) {
            const auto experts_pos = logical.runtime_name.rfind(".experts.");
            const std::string runtime_base =
                logical.runtime_name.substr(0, experts_pos);
            add_logical_group_once(
                graph, seen, LogicalTensorGroupKind::FusedMoeExperts,
                runtime_base + ".experts", {}, {},
                {},
                {LogicalTensorRole::MoeExpertsGateUp,
                 LogicalTensorRole::MoeExpertsDown});
        }

        if (hf.model_type == "gpt_oss" &&
            (ends_with(logical.runtime_name,
                       ".mlp.experts.gate_up_proj_blocks") ||
             ends_with(logical.runtime_name,
                       ".mlp.experts.down_proj_blocks"))) {
            const std::string raw_base =
                logical.raw_name.substr(
                    0, logical.raw_name.rfind("_blocks"));
            const std::string runtime_base =
                logical.runtime_name.substr(
                    0, logical.runtime_name.rfind("_blocks"));
            add_logical_group_once(
                graph, seen, LogicalTensorGroupKind::GptOssMxfp4,
                runtime_base,
                {raw_base + "_blocks",
                 raw_base + "_scales",
                 raw_base + "_bias"},
                {runtime_base + ".weight",
                 runtime_base + ".weight_scale",
                 runtime_base + ".bias"},
                {LogicalTensorRole::QuantPackedData,
                 LogicalTensorRole::QuantScale,
                 LogicalTensorRole::Bias},
                {LogicalTensorRole::QuantPackedData,
                 LogicalTensorRole::QuantScale,
                 LogicalTensorRole::Bias});
        }

        if (logical.checkpoint_dtype == DType::FP8_E4M3 &&
            loader.contains(logical.raw_name + "_scale_inv")) {
            add_logical_group_once(
                graph, seen, LogicalTensorGroupKind::Fp8ScaleInv,
                logical.runtime_name,
                {logical.raw_name, logical.raw_name + "_scale_inv"},
                {logical.runtime_name, logical.runtime_name + "_scale_inv"});
        }
    }
}

bool logical_graph_has_group(
    const LogicalTensorGraph& graph,
    LogicalTensorGroupKind kind)
{
    return std::any_of(
        graph.groups.begin(), graph.groups.end(),
        [kind](const LogicalTensorGroup& group) {
            return group.kind == kind;
        });
}

}  // namespace

LogicalTensorGraph build_logical_tensor_graph(
    const HfConfig& hf,
    const TensorMetadataSource& loader)
{
    LogicalTensorGraph graph;
    graph.tensors.reserve(loader.num_tensors());

    const std::string& mm_strip = hf.mm_lm_strip_prefix;
    const auto& mm_skip = hf.mm_skip_prefixes;
    for (const auto& raw_name : loader.tensor_names()) {
        if (pie_driver_common::starts_with_any(raw_name, mm_skip)) continue;
        const std::string runtime_name =
            pie_driver_common::strip_prefix(raw_name, mm_strip);
        const auto& info = loader.info(raw_name);
        graph.tensors.push_back(LogicalTensor{
            .raw_name = raw_name,
            .runtime_name = runtime_name,
            .role = infer_logical_tensor_role(runtime_name),
            .checkpoint_dtype = info.dtype,
            .checkpoint_shape = info.shape,
            .shard_axis = llama_like_shard_axis(runtime_name),
        });
    }
    discover_logical_tensor_groups(graph, hf, loader);
    return graph;
}

ModelSchema resolve_model_schema(
    const HfConfig& hf,
    const Config& boot_cfg,
    int tp_size)
{
    ModelSchema schema;
    schema.name = hf.model_type.empty() ? hf.arch_name : hf.model_type;
    schema.pack_dense_qkv_and_gate_up =
        supports_dense_llama_packed_load(hf, boot_cfg);
    schema.unfuse_phi3_for_tp = (hf.model_type == "phi3");
    schema.shard_fused_moe_experts_for_tp =
        (tp_size > 1) && is_qwen3_5_moe_arch(hf.model_type);
    schema.fuse_per_expert_moe_after_load = is_qwen3_5_moe_arch(hf.model_type);

    const std::string& mt = hf.model_type;
    if (mt == "phi3") {
        schema.family = ModelSchemaFamily::Phi3;
    } else if (is_qwen3_5_moe_arch(mt)) {
        schema.family = ModelSchemaFamily::QwenMoe;
    } else if (mt == "qwen3" || mt == "qwen2" ||
               mt == "llama" || mt == "llama3" ||
               mt == "mistral" || mt == "mistral3" || mt == "ministral3" ||
               mt == "olmo2" || mt == "olmo3") {
        schema.family = ModelSchemaFamily::DenseLlamaLike;
    }

    return schema;
}

int llama_like_shard_axis(const std::string& name) {
    // Column-parallel: shard along the leading output dim.
    if (ends_with(name, ".q_proj.weight") || ends_with(name, ".q_proj.bias") ||
        ends_with(name, ".k_proj.weight") || ends_with(name, ".k_proj.bias") ||
        ends_with(name, ".v_proj.weight") || ends_with(name, ".v_proj.bias") ||
        ends_with(name, ".gate_proj.weight") ||
        ends_with(name, ".up_proj.weight") ||
        ends_with(name, ".sinks")) {
        return 0;
    }
    // Row-parallel: shard along the inner input dim.
    if (ends_with(name, ".o_proj.weight") || ends_with(name, ".down_proj.weight")) {
        return 1;
    }
    // Mixtral / GPT-OSS expert weights.
    if (ends_with(name, ".w1.weight") || ends_with(name, ".w3.weight") ||
        ends_with(name, ".w1.bias")   || ends_with(name, ".w3.bias")) {
        return 0;
    }
    if (ends_with(name, ".w2.weight")) {
        return 1;
    }
    // Compressed-tensors FP8 per-channel weight_scale companion.
    if (ends_with(name, ".q_proj.weight_scale") ||
        ends_with(name, ".q_proj.weight_scale_inv") ||
        ends_with(name, ".k_proj.weight_scale") ||
        ends_with(name, ".k_proj.weight_scale_inv") ||
        ends_with(name, ".v_proj.weight_scale") ||
        ends_with(name, ".v_proj.weight_scale_inv") ||
        ends_with(name, ".gate_proj.weight_scale") ||
        ends_with(name, ".gate_proj.weight_scale_inv") ||
        ends_with(name, ".up_proj.weight_scale") ||
        ends_with(name, ".up_proj.weight_scale_inv")) {
        return 0;
    }
    // Qwen3.5 / Qwen3.6 linear-attention tensors that shard cleanly.
    if (ends_with(name, ".linear_attn.in_proj_z.weight") ||
        ends_with(name, ".linear_attn.in_proj_b.weight") ||
        ends_with(name, ".linear_attn.in_proj_a.weight") ||
        ends_with(name, ".linear_attn.dt_bias") ||
        ends_with(name, ".linear_attn.A_log")) {
        return 0;
    }
    if (ends_with(name, ".linear_attn.out_proj.weight")) {
        return 1;
    }
    return -1;
}

LoadPlan build_model_load_plan(
    const HfConfig& hf,
    const Config& boot_cfg,
    const TensorMetadataSource& loader,
    int tp_size,
    const LoadTarget& target)
{
    LoadPlan plan;
    std::unordered_set<std::string> consumed_raw;

    const ModelSchema schema = resolve_model_schema(hf, boot_cfg, tp_size);
    const LogicalTensorGraph logical_graph =
        build_logical_tensor_graph(hf, loader);
    std::unordered_map<std::string, const LogicalTensorGroup*> packed_group_by_raw;
    packed_group_by_raw.reserve(logical_graph.groups.size() * 3);
    std::unordered_map<std::string, const LogicalTensorGroup*> row_split_group_by_raw;
    row_split_group_by_raw.reserve(logical_graph.groups.size());
    std::unordered_map<std::string, const LogicalTensorGroup*> per_expert_group_by_raw;
    per_expert_group_by_raw.reserve(logical_graph.groups.size() * 3);
    std::unordered_map<std::string, const LogicalTensorGroup*> mxfp4_group_by_raw;
    mxfp4_group_by_raw.reserve(logical_graph.groups.size() * 3);
    for (const auto& group : logical_graph.groups) {
        if (group.kind == LogicalTensorGroupKind::PackedQkv ||
            group.kind == LogicalTensorGroupKind::PackedGateUp) {
            for (const auto& raw : group.raw_names) {
                packed_group_by_raw.emplace(raw, &group);
            }
        } else if (group.kind == LogicalTensorGroupKind::RowRangeSplit) {
            for (const auto& raw : group.raw_names) {
                row_split_group_by_raw.emplace(raw, &group);
            }
        } else if (group.kind == LogicalTensorGroupKind::PerExpertMoe) {
            for (const auto& raw : group.raw_names) {
                per_expert_group_by_raw.emplace(raw, &group);
            }
        } else if (group.kind == LogicalTensorGroupKind::GptOssMxfp4) {
            for (const auto& raw : group.raw_names) {
                mxfp4_group_by_raw.emplace(raw, &group);
            }
        }
    }
    const bool has_per_expert_moe_sources =
        schema.fuse_per_expert_moe_after_load &&
        logical_graph_has_group(
            logical_graph, LogicalTensorGroupKind::PerExpertMoe);
    const bool lower_runtime_quant =
        runtime_quant_enabled_for_plan(hf, boot_cfg, target.fp8_native);
    const bool can_repack_gptq_marlin =
        can_lower_gptq_marlin_repack_to_plan(hf, tp_size);
    const bool lower_gptq_marlin_repack =
        can_repack_gptq_marlin && target.gptq_marlin_int4;
    const bool can_repack_awq_marlin =
        can_lower_awq_marlin_repack_to_plan(hf);
    const bool lower_awq_marlin_repack =
        can_repack_awq_marlin && target.gptq_marlin_int4;
    const bool lower_awq_dequant =
        can_lower_awq_dequant_to_plan(hf) && !lower_awq_marlin_repack;
    const bool lower_gptq_dequant =
        can_lower_gptq_dequant_to_plan(hf, tp_size) ||
        (can_repack_gptq_marlin && !target.gptq_marlin_int4);
    const bool lower_offline_int4_dequant =
        lower_awq_dequant || lower_gptq_dequant;
    bool lowered_compressed_quant = false;
    bool lowered_gptq_marlin_repack = false;
    bool lowered_offline_int4_dequant = false;
    bool lowered_per_expert_moe_fusion = false;

    for (const auto& logical : logical_graph.tensors) {
        const std::string& raw_name = logical.raw_name;
        const std::string& name = logical.runtime_name;
        if (consumed_raw.contains(raw_name)) continue;
        if (is_compressed_quant_companion(hf, loader, logical)) {
            continue;
        }
        if (is_fp8_scale_inv_companion(loader, logical)) {
            continue;
        }
        if (is_gptq_repack_companion(
                hf, loader, logical, lower_gptq_marlin_repack)) {
            continue;
        }
        if (is_offline_int4_dequant_companion(
                hf, loader, logical, lower_offline_int4_dequant)) {
            continue;
        }
        if (is_offline_int4_dequant_companion(
                hf, loader, logical, lower_awq_marlin_repack)) {
            continue;
        }
        if (mxfp4_group_by_raw.contains(raw_name) &&
            raw_name != mxfp4_group_by_raw.at(raw_name)->raw_names.front()) {
            continue;
        }

        if (schema.shard_fused_moe_experts_for_tp &&
            logical.role == LogicalTensorRole::MoeExpertsGateUp) {
            LoadOp op = make_raw_load_op(
                LoadOpKind::GroupedSliceConcat, /*output_name=*/{},
                raw_name);
            const auto& info = loader.info(raw_name);
            auto shape = info.shape;
            if (shape.size() != 3 || shape[1] % (2 * tp_size) != 0) {
                throw std::runtime_error(
                    "load schema: MoE gate/up tensor has unsupported shape: " +
                    name);
            }
            shape[1] /= tp_size;
            add_owned_producer(
                plan, std::move(op), name, info.dtype, shape,
                TensorLayoutKind::Grouped,
                TensorParallelKind::Expert);
            consumed_raw.insert(raw_name);
            continue;
        }
        if (schema.shard_fused_moe_experts_for_tp &&
            logical.role == LogicalTensorRole::MoeExpertsDown) {
            LoadOp op = make_raw_load_op(
                LoadOpKind::GroupedSlice, /*output_name=*/{},
                raw_name);
            const auto& info = loader.info(raw_name);
            auto shape = info.shape;
            if (shape.size() != 3 || shape[2] % tp_size != 0) {
                throw std::runtime_error(
                    "load schema: MoE down tensor has unsupported shape: " +
                    name);
            }
            shape[2] /= tp_size;
            add_owned_producer(
                plan, std::move(op), name, info.dtype, shape,
                TensorLayoutKind::Grouped,
                TensorParallelKind::Expert);
            consumed_raw.insert(raw_name);
            continue;
        }

        if (schema.unfuse_phi3_for_tp) {
            const auto split_it = row_split_group_by_raw.find(raw_name);
            if (split_it != row_split_group_by_raw.end() &&
                try_add_row_range_split_group(
                    plan, hf, loader, *split_it->second,
                    consumed_raw, tp_size)) {
                continue;
            }
        }

        if (const auto mxfp4_it = mxfp4_group_by_raw.find(raw_name);
            mxfp4_it != mxfp4_group_by_raw.end()) {
            if (try_add_gpt_oss_mxfp4_expert(
                    plan, hf, loader, *mxfp4_it->second, consumed_raw, tp_size,
                    target.mxfp4_moe)) {
                continue;
            }
        }

        if (schema.pack_dense_qkv_and_gate_up) {
            const auto group_it = packed_group_by_raw.find(raw_name);
            if (group_it != packed_group_by_raw.end() &&
                try_add_packed_axis_group(
                    plan, loader, *group_it->second, consumed_raw, tp_size)) {
                continue;
            }
        }

        if (schema.fuse_per_expert_moe_after_load &&
            per_expert_group_by_raw.contains(raw_name) &&
            try_add_per_expert_moe_fusion(
                plan, hf, loader, *per_expert_group_by_raw.at(raw_name),
                consumed_raw, tp_size)) {
            lowered_per_expert_moe_fusion = true;
            continue;
        }

        const int axis = (tp_size > 1) ? logical.shard_axis : -1;
        if (try_add_compressed_fp8_weight(
                plan, hf, loader, logical, loader.info(raw_name),
                axis, tp_size, target.fp8_native)) {
            consumed_raw.insert(raw_name);
            consumed_raw.insert(raw_name + "_scale");
            lowered_compressed_quant = true;
            continue;
        }
        if (try_add_compressed_int8_weight(
                plan, hf, loader, logical, loader.info(raw_name),
                axis, tp_size)) {
            consumed_raw.insert(raw_name);
            consumed_raw.insert(raw_name + "_scale");
            lowered_compressed_quant = true;
            continue;
        }
        if (try_add_fp8_scale_inv_weight(
                plan, loader, logical, loader.info(raw_name),
                axis, tp_size)) {
            consumed_raw.insert(raw_name);
            consumed_raw.insert(raw_name + "_scale_inv");
            continue;
        }
        if (lower_awq_dequant &&
            try_add_awq_dequant_weight(
                plan, hf, loader, logical, loader.info(raw_name),
                tp_size, consumed_raw)) {
            lowered_offline_int4_dequant = true;
            continue;
        }
        if (lower_awq_marlin_repack &&
            try_add_awq_marlin_repack_weight(
                plan, hf, loader, logical, loader.info(raw_name),
                tp_size, consumed_raw)) {
            lowered_gptq_marlin_repack = true;
            continue;
        }
        if (lower_gptq_marlin_repack &&
            try_add_gptq_marlin_repack_weight(
                plan, hf, loader, logical, loader.info(raw_name),
                tp_size, consumed_raw)) {
            lowered_gptq_marlin_repack = true;
            continue;
        }
        if (lower_gptq_dequant &&
            try_add_gptq_dequant_weight(
                plan, hf, loader, logical, loader.info(raw_name),
                tp_size, consumed_raw)) {
            lowered_offline_int4_dequant = true;
            continue;
        }
        if (lower_runtime_quant && runtime_quantizable_role(logical.role)) {
            add_runtime_quantized_copy(
                plan, logical, loader.info(raw_name), boot_cfg,
                axis, tp_size);
            consumed_raw.insert(raw_name);
            continue;
        }
        add_copy(plan, raw_name, name, loader.info(raw_name), axis, tp_size);
        consumed_raw.insert(raw_name);
    }
    if (hf.quant_method == "compressed-tensors" && !lowered_compressed_quant) {
        throw std::runtime_error(
            "load schema: compressed-tensors checkpoint did not match any "
            "scheduled FP8/INT8 quantized weight layout");
    }
    if ((hf.quant_method == "awq" || hf.quant_method == "gptq") &&
        !lowered_gptq_marlin_repack && !lowered_offline_int4_dequant) {
        throw std::runtime_error(
            "load schema: " + hf.quant_method +
            " checkpoint did not match any scheduled INT4 dequant/repack "
            "layout");
    }
    if (schema.fuse_per_expert_moe_after_load &&
        has_per_expert_moe_sources && !lowered_per_expert_moe_fusion) {
        throw std::runtime_error(
            "load schema: per-expert MoE checkpoint layout requires "
            "StackGroups IR lowering, but no complete expert group was "
            "scheduled");
    }
    finalize_memory_plan(plan);
    return plan;
}

}  // namespace pie_cuda_driver
