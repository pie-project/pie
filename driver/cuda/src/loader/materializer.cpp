#include "loader/materializer.hpp"

#include <cuda_runtime.h>
#include <nccl.h>

#include <cstdint>
#include <limits>
#include <stdexcept>
#include <utility>
#include <vector>

#include "cuda_check.hpp"
#include "distributed.hpp"
#include "kernels/deinterleave.hpp"
#include "kernels/dequant_fp8.hpp"
#include "kernels/dequant_fp4.hpp"
#include "kernels/dtype_cast.hpp"
#include "kernels/quant_bf16_to_fp8.hpp"
#include "loader/safetensors.hpp"
#include "model/weight_store.hpp"
#include "tensor.hpp"

#ifdef PIE_CUDA_HAS_MARLIN
#include "marlin_wrapper.hpp"
#endif

namespace pie_cuda_driver {

namespace {

const TensorSpec& tensor_spec_for(
    const LoadPlan& plan,
    const std::string& name);

std::uint64_t copy_raw_to_output(
    const LoadOp& op,
    const LoadPlan& plan,
    SafetensorsLoader& loader,
    WeightStoreBuilder& weights,
    int tp_rank,
    int tp_size)
{
    const TensorSpec& out_spec = tensor_spec_for(plan, load_op_output(op));
    const TensorInfo& info = loader.info(load_op_raw_name(op));
    if (info.dtype != out_spec.dtype) {
        throw std::runtime_error(
            "load plan: raw copy dtype mismatch for '" + out_spec.name + "'");
    }
    DeviceTensor out = DeviceTensor::allocate(out_spec.dtype, out_spec.shape);
    loader.copy_shard_to_device(
        load_op_raw_name(op),
        load_op_shard_axis(op),
        tp_rank,
        tp_size,
        out.data(),
        out_spec.shape);
    const std::uint64_t bytes = out.nbytes();
    weights.insert(out_spec.name, std::move(out), out_spec);
    return bytes;
}

std::uint64_t copy_row_range_to_output(
    const LoadOp& op,
    const LoadPlan& plan,
    SafetensorsLoader& loader,
    WeightStoreBuilder& weights,
    int tp_rank,
    int tp_size)
{
    const TensorSpec& out_spec = tensor_spec_for(plan, load_op_output(op));
    const std::string& raw_name = load_op_raw_name(op);
    const std::int64_t row_offset = load_op_row_offset(op);
    const std::int64_t rows = load_op_rows(op);
    if (rows <= 0) {
        throw std::runtime_error(
            "load plan: row-range shard has non-positive row count for '" +
            raw_name + "'");
    }
    std::int64_t local_rows = rows;
    std::int64_t local_offset = row_offset;
    if (tp_size > 1) {
        if (rows % tp_size != 0) {
            throw std::runtime_error(
                "load plan: row range for '" + raw_name +
                "' is not divisible by tensor-parallel size");
        }
        local_rows = rows / tp_size;
        local_offset += static_cast<std::int64_t>(tp_rank) * local_rows;
    }
    DeviceTensor out = DeviceTensor::allocate(out_spec.dtype, out_spec.shape);
    loader.copy_slice_to_device(
        raw_name, /*axis=*/0, local_offset, local_rows,
        out.data(), out_spec.shape);
    const std::uint64_t bytes = out.nbytes();
    weights.insert(out_spec.name, std::move(out), out_spec);
    return bytes;
}

std::uint64_t copy_moe_gate_up_to_output(
    const LoadOp& op,
    const LoadPlan& plan,
    SafetensorsLoader& loader,
    WeightStoreBuilder& weights,
    int tp_rank,
    int tp_size)
{
    const TensorSpec& out_spec = tensor_spec_for(plan, load_op_output(op));
    const std::string& raw_name = load_op_raw_name(op);
    const TensorInfo& info = loader.info(raw_name);
    if (info.shape.size() != 3 || out_spec.shape.size() != 3) {
        throw std::runtime_error(
            "load plan: MoE gate/up shard expects rank-3 tensor '" +
            raw_name + "'");
    }
    const std::int64_t E = info.shape[0];
    const std::int64_t two_I = info.shape[1];
    const std::int64_t H = info.shape[2];
    if (two_I % 2 != 0 || (two_I / 2) % tp_size != 0) {
        throw std::runtime_error(
            "load plan: MoE gate/up shard intermediate axis does not divide "
            "tensor-parallel size for '" + raw_name + "'");
    }
    const std::int64_t I = two_I / 2;
    const std::int64_t I_local = I / tp_size;
    if (out_spec.dtype != info.dtype ||
        out_spec.shape != std::vector<std::int64_t>{E, 2 * I_local, H}) {
        throw std::runtime_error(
            "load plan: MoE gate/up output spec mismatch for '" +
            out_spec.name + "'");
    }

    DeviceTensor out = DeviceTensor::allocate(out_spec.dtype, out_spec.shape);
    const std::size_t expert_half_bytes =
        static_cast<std::size_t>(I_local) *
        static_cast<std::size_t>(H) *
        dtype_bytes(out_spec.dtype);
    const std::size_t expert_bytes = 2 * expert_half_bytes;
    const std::int64_t gate_start =
        static_cast<std::int64_t>(tp_rank) * I_local;
    const std::int64_t up_start = I + gate_start;
    auto* dst = static_cast<std::uint8_t*>(out.data());
    const std::vector<std::int64_t> slice_shape{1, I_local, H};
    for (std::int64_t e = 0; e < E; ++e) {
        loader.copy_strided_to_device(
            raw_name,
            {TensorSlice{0, e, 1}, TensorSlice{1, gate_start, I_local}},
            dst + static_cast<std::size_t>(e) * expert_bytes,
            slice_shape);
        loader.copy_strided_to_device(
            raw_name,
            {TensorSlice{0, e, 1}, TensorSlice{1, up_start, I_local}},
            dst + static_cast<std::size_t>(e) * expert_bytes + expert_half_bytes,
            slice_shape);
    }
    const std::uint64_t bytes = out.nbytes();
    weights.insert(out_spec.name, std::move(out), out_spec);
    return bytes;
}

std::uint64_t copy_moe_down_to_output(
    const LoadOp& op,
    const LoadPlan& plan,
    SafetensorsLoader& loader,
    WeightStoreBuilder& weights,
    int tp_rank,
    int tp_size)
{
    const TensorSpec& out_spec = tensor_spec_for(plan, load_op_output(op));
    const std::string& raw_name = load_op_raw_name(op);
    const TensorInfo& info = loader.info(raw_name);
    if (info.shape.size() != 3 || out_spec.shape.size() != 3) {
        throw std::runtime_error(
            "load plan: MoE down shard expects rank-3 tensor '" + raw_name + "'");
    }
    const std::int64_t I = info.shape[2];
    if (I % tp_size != 0) {
        throw std::runtime_error(
            "load plan: MoE down shard intermediate axis does not divide "
            "tensor-parallel size for '" + raw_name + "'");
    }
    const std::int64_t I_local = I / tp_size;
    if (out_spec.dtype != info.dtype ||
        out_spec.shape != std::vector<std::int64_t>{
            info.shape[0], info.shape[1], I_local}) {
        throw std::runtime_error(
            "load plan: MoE down output spec mismatch for '" + out_spec.name + "'");
    }
    DeviceTensor out = DeviceTensor::allocate(out_spec.dtype, out_spec.shape);
    loader.copy_strided_to_device(
        raw_name,
        {TensorSlice{2, static_cast<std::int64_t>(tp_rank) * I_local, I_local}},
        out.data(),
        out_spec.shape);
    const std::uint64_t bytes = out.nbytes();
    weights.insert(out_spec.name, std::move(out), out_spec);
    return bytes;
}

const TensorSpec& tensor_spec_for(
    const LoadPlan& plan,
    const std::string& name)
{
    auto it = plan.tensors.find(name);
    if (it == plan.tensors.end()) {
        throw std::runtime_error(
            "load plan: no TensorSpec for materialized tensor '" + name + "'");
    }
    return it->second;
}

const std::string& first_input(const LoadOp& op) {
    if (load_op_inputs(op).empty()) {
        throw std::runtime_error(
            "load plan: op for '" + load_op_output(op) +
            "' requires at least one input");
    }
    return load_op_inputs(op).front();
}

void insert_row_view(
    WeightStoreBuilder& weights,
    const TensorSpec& view_spec,
    const std::string& backing_name,
    const std::string& view_name,
    std::int64_t row_offset,
    std::int64_t rows,
    std::int64_t cols)
{
    const DeviceTensor& backing = weights.get(backing_name);
    const auto element_bytes =
        static_cast<std::int64_t>(dtype_bytes(backing.dtype()));
    auto* base = static_cast<std::uint8_t*>(const_cast<void*>(backing.data()));
    weights.insert(view_name, DeviceTensor::view(
        base + row_offset * cols * element_bytes,
        backing.dtype(),
        {rows, cols}), view_spec);
}

std::uint64_t copy_tensor_to_output(
    const DeviceTensor& src,
    WeightStoreBuilder& weights,
    const TensorSpec& out_spec)
{
    if (src.dtype() != out_spec.dtype || src.shape() != out_spec.shape) {
        throw std::runtime_error(
            "load plan: Materialize source does not match TensorSpec for '" +
            out_spec.name + "'");
    }
    DeviceTensor out = DeviceTensor::allocate(out_spec.dtype, out_spec.shape);
    CUDA_CHECK(cudaMemcpyAsync(
        out.data(), src.data(), src.nbytes(),
        cudaMemcpyDeviceToDevice, /*stream=*/0));
    const std::uint64_t bytes = out.nbytes();
    weights.insert(out_spec.name, std::move(out), out_spec);
    return bytes;
}

void cast_tensor_to_output(
    const DeviceTensor& src,
    WeightStoreBuilder& weights,
    const TensorSpec& out_spec)
{
    if (src.shape() != out_spec.shape) {
        throw std::runtime_error(
            "load plan: Cast source shape does not match TensorSpec for '" +
            out_spec.name + "'");
    }
    DeviceTensor out = DeviceTensor::allocate(out_spec.dtype, out_spec.shape);
    if (src.dtype() == out_spec.dtype) {
        CUDA_CHECK(cudaMemcpyAsync(
            out.data(), src.data(), src.nbytes(),
            cudaMemcpyDeviceToDevice, /*stream=*/0));
    } else if (src.dtype() == DType::FP16 && out_spec.dtype == DType::BF16) {
        kernels::launch_cast_fp16_to_bf16(
            src.data(), out.data(), src.numel(), /*stream=*/0);
    } else if (src.dtype() == DType::FP32 && out_spec.dtype == DType::BF16) {
        kernels::launch_cast_fp32_to_bf16(
            src.data(), out.data(), src.numel(), /*stream=*/0);
    } else if (src.dtype() == DType::BF16 && out_spec.dtype == DType::FP32) {
        kernels::launch_cast_bf16_to_fp32(
            src.data(), out.data(), src.numel(), /*stream=*/0);
    } else {
        throw std::runtime_error(
            "load plan: unsupported Cast " +
            std::string(dtype_name(src.dtype())) + " -> " +
            std::string(dtype_name(out_spec.dtype)) + " for '" +
            out_spec.name + "'");
    }
    weights.insert(out_spec.name, std::move(out), out_spec);
}

std::uint64_t concat_rows_to_output(
    const LoadOp& op,
    const LoadPlan& plan,
    WeightStoreBuilder& weights)
{
    const TensorSpec& out_spec = tensor_spec_for(plan, load_op_output(op));
    if (out_spec.shape.size() != 2) {
        throw std::runtime_error(
            "load plan: Concat output must be 2-D: " + load_op_output(op));
    }
    if (load_op_inputs(op).empty()) {
        throw std::runtime_error(
            "load plan: Concat op has no inputs for " + load_op_output(op));
    }

    const std::int64_t out_cols = out_spec.shape[1];
    std::int64_t total_rows = 0;
    for (const auto& name : load_op_inputs(op)) {
        const DeviceTensor& t = weights.get(name);
        if (t.dtype() != out_spec.dtype || t.shape().size() != 2 ||
            t.shape()[1] != out_cols) {
            throw std::runtime_error(
                "load plan: Concat input mismatch for '" + load_op_output(op) +
                "': " + name);
        }
        total_rows += t.shape()[0];
    }
    if (total_rows != out_spec.shape[0]) {
        throw std::runtime_error(
            "load plan: Concat row count does not match TensorSpec for '" +
            load_op_output(op) + "'");
    }

    DeviceTensor out = DeviceTensor::allocate(out_spec.dtype, out_spec.shape);
    auto* dst = static_cast<std::uint8_t*>(out.data());
    std::size_t byte_offset = 0;
    for (const auto& name : load_op_inputs(op)) {
        const DeviceTensor& t = weights.get(name);
        CUDA_CHECK(cudaMemcpyAsync(
            dst + byte_offset,
            t.data(),
            t.nbytes(),
            cudaMemcpyDeviceToDevice,
            /*stream=*/0));
        byte_offset += t.nbytes();
    }
    const std::uint64_t bytes = out.nbytes();
    weights.insert(load_op_output(op), std::move(out), out_spec);
    return bytes;
}

std::uint64_t slice_rows_to_output(
    const LoadOp& op,
    const LoadPlan& plan,
    WeightStoreBuilder& weights)
{
    const TensorSpec& out_spec = tensor_spec_for(plan, load_op_output(op));
    if (out_spec.shape.size() != 2) {
        throw std::runtime_error(
            "load plan: Slice output must be 2-D: " + load_op_output(op));
    }
    const DeviceTensor& src = weights.get(first_input(op));
    if (src.dtype() != out_spec.dtype || src.shape().size() != 2 ||
        src.shape()[1] != out_spec.shape[1]) {
        throw std::runtime_error(
            "load plan: Slice source mismatch for '" + load_op_output(op) + "'");
    }
    const std::int64_t rows =
        load_op_rows(op) > 0 ? load_op_rows(op) : out_spec.shape[0];
    if (rows != out_spec.shape[0] || load_op_row_offset(op) < 0 ||
        load_op_row_offset(op) + rows > src.shape()[0]) {
        throw std::runtime_error(
            "load plan: Slice range out of bounds for '" + load_op_output(op) + "'");
    }

    DeviceTensor out = DeviceTensor::allocate(out_spec.dtype, out_spec.shape);
    const std::size_t row_bytes =
        static_cast<std::size_t>(out_spec.shape[1]) *
        dtype_bytes(out_spec.dtype);
    const auto* src8 = static_cast<const std::uint8_t*>(src.data()) +
        static_cast<std::size_t>(load_op_row_offset(op)) * row_bytes;
    CUDA_CHECK(cudaMemcpyAsync(
        out.data(), src8, static_cast<std::size_t>(rows) * row_bytes,
        cudaMemcpyDeviceToDevice, /*stream=*/0));
    const std::uint64_t bytes = out.nbytes();
    weights.insert(load_op_output(op), std::move(out), out_spec);
    return bytes;
}

std::uint64_t slice_axis_to_output(
    const LoadOp& op,
    const LoadPlan& plan,
    WeightStoreBuilder& weights,
    int tp_rank,
    int tp_size)
{
    const TensorSpec& out_spec = tensor_spec_for(plan, load_op_output(op));
    const DeviceTensor& src = weights.get(first_input(op));
    if (src.dtype() != out_spec.dtype) {
        throw std::runtime_error(
            "load plan: Slice dtype mismatch for '" + load_op_output(op) + "'");
    }
    if (load_op_slice_axis(op) < 0 ||
        load_op_slice_axis(op) >= static_cast<int>(src.shape().size())) {
        throw std::runtime_error(
            "load plan: Slice axis out of range for '" + load_op_output(op) + "'");
    }

    auto expected = src.shape();
    std::int64_t length = load_op_slice_length(op) > 0
        ? load_op_slice_length(op)
        : out_spec.shape.at(static_cast<std::size_t>(load_op_slice_axis(op)));
    std::int64_t start = load_op_slice_start(op);
    if (tp_size > 1 && load_op_shard_axis(op) == load_op_slice_axis(op)) {
        start += static_cast<std::int64_t>(tp_rank) * length;
    }
    if (length <= 0 || start < 0 ||
        start + length > src.shape()[static_cast<std::size_t>(load_op_slice_axis(op))]) {
        throw std::runtime_error(
            "load plan: Slice range out of bounds for '" + load_op_output(op) + "'");
    }
    expected[static_cast<std::size_t>(load_op_slice_axis(op))] = length;
    if (expected != out_spec.shape) {
        throw std::runtime_error(
            "load plan: Slice output shape does not match TensorSpec for '" +
            load_op_output(op) + "'");
    }

    std::int64_t outer = 1;
    for (int i = 0; i < load_op_slice_axis(op); ++i) {
        outer *= src.shape()[static_cast<std::size_t>(i)];
    }
    std::int64_t inner = 1;
    for (std::size_t i = static_cast<std::size_t>(load_op_slice_axis(op)) + 1;
         i < src.shape().size(); ++i) {
        inner *= src.shape()[i];
    }
    const std::int64_t axis_dim =
        src.shape()[static_cast<std::size_t>(load_op_slice_axis(op))];
    const std::size_t elem = dtype_bytes(src.dtype());
    const std::size_t width =
        static_cast<std::size_t>(length) *
        static_cast<std::size_t>(inner) * elem;
    const std::size_t src_pitch =
        static_cast<std::size_t>(axis_dim) *
        static_cast<std::size_t>(inner) * elem;
    const std::size_t dst_pitch = width;
    const auto* src_base = static_cast<const std::uint8_t*>(src.data()) +
        static_cast<std::size_t>(start) *
        static_cast<std::size_t>(inner) * elem;

    DeviceTensor out = DeviceTensor::allocate(out_spec.dtype, out_spec.shape);
    CUDA_CHECK(cudaMemcpy2DAsync(
        out.data(), dst_pitch,
        src_base, src_pitch,
        width, static_cast<std::size_t>(outer),
        cudaMemcpyDeviceToDevice, /*stream=*/0));
    const std::uint64_t bytes = out.nbytes();
    weights.insert(load_op_output(op), std::move(out), out_spec);
    return bytes;
}

void view_or_alias_to_output(
    const LoadOp& op,
    const LoadPlan& plan,
    WeightStoreBuilder& weights)
{
    const TensorSpec& out_spec = tensor_spec_for(plan, load_op_output(op));
    const DeviceTensor& src = weights.get(first_input(op));
    if (src.dtype() != out_spec.dtype) {
        throw std::runtime_error(
            "load plan: View/Alias dtype mismatch for '" + load_op_output(op) + "'");
    }

    void* ptr = const_cast<void*>(src.data());
    if (load_op_rows(op) > 0 || load_op_row_offset(op) > 0) {
        if (src.shape().size() != 2 || out_spec.shape.size() != 2 ||
            src.shape()[1] != out_spec.shape[1]) {
            throw std::runtime_error(
                "load plan: row view requires matching 2-D tensors for '" +
                load_op_output(op) + "'");
        }
        const std::int64_t rows = load_op_rows(op) > 0 ? load_op_rows(op) : out_spec.shape[0];
        if (rows != out_spec.shape[0] || load_op_row_offset(op) < 0 ||
            load_op_row_offset(op) + rows > src.shape()[0]) {
            throw std::runtime_error(
                "load plan: row view range out of bounds for '" +
                load_op_output(op) + "'");
        }
        const std::size_t row_bytes =
            static_cast<std::size_t>(src.shape()[1]) * dtype_bytes(src.dtype());
        ptr = static_cast<std::uint8_t*>(ptr) +
            static_cast<std::size_t>(load_op_row_offset(op)) * row_bytes;
    } else if (src.shape() != out_spec.shape) {
        throw std::runtime_error(
            "load plan: Alias shape mismatch for '" + load_op_output(op) + "'");
    }

    weights.insert(load_op_output(op), DeviceTensor::view(
        ptr, out_spec.dtype, out_spec.shape), out_spec);
}

void attach_quant_meta_for_output(
    const LoadOp& op,
    const LoadPlan& plan,
    WeightStoreBuilder& weights)
{
    const TensorSpec& spec = tensor_spec_for(plan, load_op_output(op));
    if (spec.quant.format == QuantFormat::None ||
        spec.quant.scale_tensor.empty()) {
        throw std::runtime_error(
            "load plan: AttachQuantMeta needs quant TensorSpec for '" +
            load_op_output(op) + "'");
    }

    QuantMeta meta;
    switch (spec.quant.granularity) {
    case QuantGranularity::PerTensor:
        meta.kind = QuantMeta::Kind::PerTensor;
        break;
    case QuantGranularity::PerChannel:
        meta.kind = QuantMeta::Kind::PerChannel;
        break;
    case QuantGranularity::PerGroup:
        meta.kind = QuantMeta::Kind::PerGroup;
        break;
    case QuantGranularity::None:
        throw std::runtime_error(
            "load plan: AttachQuantMeta missing granularity for '" +
            load_op_output(op) + "'");
    }
    meta.scale = &weights.get(spec.quant.scale_tensor);
    meta.zero_point = spec.quant.zero_point_tensor.empty()
        ? nullptr
        : &weights.get(spec.quant.zero_point_tensor);
    meta.group_size = spec.quant.group_size;
    meta.channel_axis = spec.quant.channel_axis;
    weights.set_quant_meta(load_op_output(op), std::move(meta));
}

struct RuntimeQuantResult {
    std::uint64_t bytes_before = 0;
    std::uint64_t bytes_after = 0;
};

int checked_int_dim(std::int64_t value, const std::string& name);

RuntimeQuantResult quantize_runtime_to_output(
    const LoadOp& op,
    const LoadPlan& plan,
    WeightStoreBuilder& weights,
    NcclComm* tp_comm,
    int tp_size)
{
    const TensorSpec& out_spec = tensor_spec_for(plan, load_op_output(op));
    const DeviceTensor& src = weights.get(first_input(op));
    if (src.shape().size() != 2 || out_spec.shape.size() != 2 ||
        src.shape() != out_spec.shape) {
        throw std::runtime_error(
            "load plan: QuantizeRuntime requires matching 2-D source and "
            "output shapes for '" + load_op_output(op) + "'");
    }
    if (out_spec.quant.scale_tensor.empty()) {
        throw std::runtime_error(
            "load plan: QuantizeRuntime missing scale tensor for '" +
            load_op_output(op) + "'");
    }

    const bool is_int8 =
        out_spec.quant.format == QuantFormat::RuntimeInt8 &&
        out_spec.dtype == DType::INT8;
    const bool is_fp8 =
        out_spec.quant.format == QuantFormat::RuntimeFp8E4M3 &&
        out_spec.dtype == DType::FP8_E4M3;
    if (!is_int8 && !is_fp8) {
        throw std::runtime_error(
            "load plan: QuantizeRuntime has unsupported quant spec for '" +
            load_op_output(op) + "'");
    }
    if (out_spec.quant.granularity != QuantGranularity::PerChannel) {
        throw std::runtime_error(
            "load plan: QuantizeRuntime currently expects per-channel "
            "quantization for '" + load_op_output(op) + "'");
    }

    DeviceTensor bf16_scratch;
    const DeviceTensor* bf16_src = &src;
    if (src.dtype() == DType::FP16 || src.dtype() == DType::FP32) {
        bf16_scratch = DeviceTensor::allocate(DType::BF16, src.shape());
        if (src.dtype() == DType::FP16) {
            kernels::launch_cast_fp16_to_bf16(
                src.data(), bf16_scratch.data(), src.numel(), /*stream=*/0);
        } else {
            kernels::launch_cast_fp32_to_bf16(
                src.data(), bf16_scratch.data(), src.numel(), /*stream=*/0);
        }
        bf16_src = &bf16_scratch;
    } else if (src.dtype() != DType::BF16) {
        throw std::runtime_error(
            "load plan: QuantizeRuntime source '" + first_input(op) +
            "' has unsupported dtype " + dtype_name(src.dtype()));
    }

    const auto rows = static_cast<int>(bf16_src->shape()[0]);
    const auto cols = static_cast<int>(bf16_src->shape()[1]);
    DeviceTensor q = DeviceTensor::allocate(out_spec.dtype, out_spec.shape);
    DeviceTensor scale = DeviceTensor::allocate(DType::FP32, {rows});
    float* scale_ptr = static_cast<float*>(scale.data());

    kernels::launch_absmax_per_row_bf16(
        bf16_src->data(), scale_ptr, rows, cols, /*stream=*/0);
    if (tp_size > 1 && out_spec.parallel == TensorParallelKind::Row) {
        if (!tp_comm) {
            throw std::runtime_error(
                "load plan: QuantizeRuntime row-parallel tensor '" +
                load_op_output(op) + "' requires a tensor-parallel communicator");
        }
        tp_comm->all_reduce_fp32(
            scale_ptr, static_cast<std::size_t>(rows),
            ncclMax, /*stream=*/0);
    }
    if (is_int8) {
        kernels::launch_absmax_to_scale_inv_int8(
            scale_ptr, rows, /*stream=*/0);
        kernels::launch_cast_bf16_to_int8_per_channel(
            bf16_src->data(), static_cast<std::int8_t*>(q.data()),
            scale_ptr, rows, cols, /*stream=*/0);
    } else {
        kernels::launch_absmax_to_scale_inv(
            scale_ptr, rows, /*stream=*/0);
        kernels::launch_cast_bf16_to_fp8_e4m3_per_channel(
            bf16_src->data(), static_cast<std::uint8_t*>(q.data()),
            scale_ptr, rows, cols, /*stream=*/0);
    }

    RuntimeQuantResult result;
    result.bytes_before = src.nbytes();
    result.bytes_after = q.nbytes() + scale.nbytes();
    weights.insert(load_op_output(op), std::move(q), out_spec);
    weights.insert(
        out_spec.quant.scale_tensor,
        std::move(scale),
        tensor_spec_for(plan, out_spec.quant.scale_tensor));
    return result;
}

const TensorSpec* find_tensor_spec(
    const LoadPlan& plan,
    const std::string& name) noexcept
{
    const auto it = plan.tensors.find(name);
    return it == plan.tensors.end() ? nullptr : &it->second;
}

const void* bf16_scale_ptr_for_int4_dequant(
    const DeviceTensor& scale,
    DeviceTensor& bf16_scratch,
    const std::string& output_name)
{
    if (scale.dtype() == DType::BF16) {
        return scale.data();
    }
    bf16_scratch = DeviceTensor::allocate(DType::BF16, scale.shape());
    if (scale.dtype() == DType::FP16) {
        kernels::launch_cast_fp16_to_bf16(
            scale.data(), bf16_scratch.data(), scale.numel(), /*stream=*/0);
    } else if (scale.dtype() == DType::FP32) {
        kernels::launch_cast_fp32_to_bf16(
            scale.data(), bf16_scratch.data(), scale.numel(), /*stream=*/0);
    } else {
        throw std::runtime_error(
            "load plan: INT4 Dequantize scale for '" + output_name +
            "' must be bf16/fp16/fp32, got " +
            std::string(dtype_name(scale.dtype())));
    }
    return bf16_scratch.data();
}

std::uint64_t dequantize_awq_to_output(
    const LoadOp& op,
    const LoadPlan& plan,
    WeightStoreBuilder& weights,
    const TensorSpec& source_spec,
    const TensorSpec& out_spec)
{
    if (load_op_inputs(op).size() != 3) {
        throw std::runtime_error(
            "load plan: AWQ Dequantize op for '" + load_op_output(op) +
            "' requires qweight, qzeros, and scales inputs");
    }
    const DeviceTensor& qweight = weights.get(load_op_inputs(op)[0]);
    const DeviceTensor& qzeros = weights.get(load_op_inputs(op)[1]);
    const DeviceTensor& scales = weights.get(load_op_inputs(op)[2]);
    (void)plan;

    if (qweight.dtype() != DType::INT32 || qweight.shape().size() != 2 ||
        qzeros.dtype() != DType::INT32 || qzeros.shape().size() != 2 ||
        scales.shape().size() != 2 || out_spec.dtype != DType::BF16) {
        throw std::runtime_error(
            "load plan: AWQ Dequantize input dtype/rank mismatch for '" +
            load_op_output(op) + "'");
    }
    const std::int64_t size_k = qweight.shape()[0];
    const std::int64_t size_n = qweight.shape()[1] * 8;
    if (size_k <= 0 || size_n <= 0 ||
        source_spec.quant.group_size <= 0 ||
        size_k % source_spec.quant.group_size != 0 ||
        out_spec.shape != std::vector<std::int64_t>{size_n, size_k}) {
        throw std::runtime_error(
            "load plan: AWQ Dequantize output shape mismatch for '" +
            load_op_output(op) + "'");
    }
    const std::int64_t groups = size_k / source_spec.quant.group_size;
    if (qzeros.shape() != std::vector<std::int64_t>{groups, size_n / 8} ||
        scales.shape() != std::vector<std::int64_t>{groups, size_n}) {
        throw std::runtime_error(
            "load plan: AWQ Dequantize qzeros/scales shape mismatch for '" +
            load_op_output(op) + "'");
    }

    DeviceTensor scale_bf16;
    const void* scale_ptr =
        bf16_scale_ptr_for_int4_dequant(scales, scale_bf16, load_op_output(op));
    DeviceTensor out = DeviceTensor::allocate(out_spec.dtype, out_spec.shape);
    kernels::launch_awq_dequant_to_bf16(
        qweight.data(), qzeros.data(), scale_ptr, out.data(),
        checked_int_dim(size_k, load_op_output(op)),
        checked_int_dim(size_n, load_op_output(op)),
        source_spec.quant.group_size,
        /*stream=*/0);
    const std::uint64_t bytes = out.nbytes();
    weights.insert(load_op_output(op), std::move(out), out_spec);
    return bytes;
}

std::uint64_t dequantize_gptq_to_output(
    const LoadOp& op,
    const LoadPlan& plan,
    WeightStoreBuilder& weights,
    const TensorSpec& source_spec,
    const TensorSpec& out_spec)
{
    if (load_op_inputs(op).size() != 3 && load_op_inputs(op).size() != 4) {
        throw std::runtime_error(
            "load plan: GPTQ Dequantize op for '" + load_op_output(op) +
            "' requires qweight, qzeros, scales, and optional g_idx inputs");
    }
    const DeviceTensor& qweight = weights.get(load_op_inputs(op)[0]);
    const DeviceTensor& qzeros = weights.get(load_op_inputs(op)[1]);
    const DeviceTensor& scales = weights.get(load_op_inputs(op)[2]);
    (void)plan;

    if (qweight.dtype() != DType::INT32 || qweight.shape().size() != 2 ||
        qzeros.dtype() != DType::INT32 || qzeros.shape().size() != 2 ||
        scales.shape().size() != 2 || out_spec.dtype != DType::BF16) {
        throw std::runtime_error(
            "load plan: GPTQ Dequantize input dtype/rank mismatch for '" +
            load_op_output(op) + "'");
    }
    const std::int64_t size_k = qweight.shape()[0] * 8;
    const std::int64_t size_n = qweight.shape()[1];
    if (size_k <= 0 || size_n <= 0 || size_n % 8 != 0 ||
        source_spec.quant.group_size <= 0 ||
        size_k % source_spec.quant.group_size != 0 ||
        out_spec.shape != std::vector<std::int64_t>{size_n, size_k}) {
        throw std::runtime_error(
            "load plan: GPTQ Dequantize output shape mismatch for '" +
            load_op_output(op) + "'");
    }
    const std::int64_t groups = size_k / source_spec.quant.group_size;
    const bool has_gidx = load_op_inputs(op).size() == 4;
    if (!has_gidx) {
        if (qzeros.shape() != std::vector<std::int64_t>{groups, size_n / 8} ||
            scales.shape() != std::vector<std::int64_t>{groups, size_n}) {
            throw std::runtime_error(
                "load plan: GPTQ Dequantize qzeros/scales shape mismatch for '" +
                load_op_output(op) + "'");
        }
    } else {
        if (qzeros.shape().size() != 2 || scales.shape().size() != 2 ||
            qzeros.shape()[1] != size_n / 8 ||
            scales.shape()[1] != size_n ||
            qzeros.shape()[0] != scales.shape()[0] ||
            qzeros.shape()[0] < groups) {
            throw std::runtime_error(
                "load plan: GPTQ act-order Dequantize qzeros/scales do not "
                "cover local groups for '" + load_op_output(op) + "'");
        }
    }

    const void* gidx_ptr = nullptr;
    if (has_gidx) {
        const DeviceTensor& gidx = weights.get(load_op_inputs(op)[3]);
        if (gidx.dtype() != DType::INT32 ||
            gidx.shape() != std::vector<std::int64_t>{size_k}) {
            throw std::runtime_error(
                "load plan: GPTQ Dequantize g_idx shape mismatch for '" +
                load_op_output(op) + "'");
        }
        gidx_ptr = gidx.data();
    }

    DeviceTensor scale_bf16;
    const void* scale_ptr =
        bf16_scale_ptr_for_int4_dequant(scales, scale_bf16, load_op_output(op));
    DeviceTensor out = DeviceTensor::allocate(out_spec.dtype, out_spec.shape);
    kernels::launch_gptq_dequant_to_bf16(
        qweight.data(), qzeros.data(), scale_ptr, gidx_ptr, out.data(),
        checked_int_dim(size_k, load_op_output(op)),
        checked_int_dim(size_n, load_op_output(op)),
        source_spec.quant.group_size,
        /*stream=*/0);
    const std::uint64_t bytes = out.nbytes();
    weights.insert(load_op_output(op), std::move(out), out_spec);
    return bytes;
}

std::uint64_t dequantize_to_output(
    const LoadOp& op,
    const LoadPlan& plan,
    WeightStoreBuilder& weights)
{
    if (load_op_inputs(op).size() < 2) {
        throw std::runtime_error(
            "load plan: Dequantize op for '" + load_op_output(op) +
            "' requires weight and scale inputs");
    }
    const TensorSpec& out_spec = tensor_spec_for(plan, load_op_output(op));
    const TensorSpec* source_spec = find_tensor_spec(plan, load_op_inputs(op)[0]);
    if (source_spec != nullptr &&
        source_spec->quant.format == QuantFormat::AwqInt4) {
        return dequantize_awq_to_output(
            op, plan, weights, *source_spec, out_spec);
    }
    if (source_spec != nullptr &&
        source_spec->quant.format == QuantFormat::GptqInt4) {
        return dequantize_gptq_to_output(
            op, plan, weights, *source_spec, out_spec);
    }

    const DeviceTensor& src = weights.get(load_op_inputs(op)[0]);
    const DeviceTensor& scale = weights.get(load_op_inputs(op)[1]);
    if (src.dtype() == DType::UINT8 && scale.dtype() == DType::UINT8 &&
        out_spec.dtype == DType::BF16) {
        if (scale.shape().empty() || src.shape().size() != scale.shape().size() + 1) {
            throw std::runtime_error(
                "load plan: MXFP4 Dequantize expects packed blocks with one "
                "extra trailing dimension for '" + load_op_output(op) + "'");
        }
        const std::int64_t blocks_per_row = scale.shape().back();
        if (blocks_per_row <= 0 || src.shape().back() != 16) {
            throw std::runtime_error(
                "load plan: MXFP4 Dequantize expects 16 packed bytes per "
                "32-value block for '" + load_op_output(op) + "'");
        }
        std::int64_t rows = 1;
        for (std::size_t i = 0; i + 1 < scale.shape().size(); ++i) {
            rows *= scale.shape()[i];
        }
        const std::int64_t in_dim = blocks_per_row * 32;
        std::vector<std::int64_t> expected = scale.shape();
        expected.back() = in_dim;
        if (expected != out_spec.shape) {
            throw std::runtime_error(
                "load plan: MXFP4 Dequantize output shape mismatch for '" +
                load_op_output(op) + "'");
        }
        if (src.numel() != static_cast<std::size_t>(rows * blocks_per_row * 16)) {
            throw std::runtime_error(
                "load plan: MXFP4 Dequantize packed source shape mismatch for '" +
                load_op_output(op) + "'");
        }

        DeviceTensor out = DeviceTensor::allocate(out_spec.dtype, out_spec.shape);
        kernels::launch_dequant_mxfp4_to_bf16(
            static_cast<const std::uint8_t*>(src.data()),
            static_cast<const std::uint8_t*>(scale.data()),
            out.data(),
            static_cast<int>(rows),
            static_cast<int>(in_dim),
            /*stream=*/0);
        const std::uint64_t bytes = out.nbytes();
        weights.insert(load_op_output(op), std::move(out), out_spec);
        return bytes;
    }

    if (src.dtype() != DType::FP8_E4M3 || out_spec.dtype != DType::BF16) {
        throw std::runtime_error(
            "load plan: Dequantize currently supports FP8_E4M3, MXFP4, "
            "AWQ INT4, and GPTQ INT4 to BF16 for '" +
            load_op_output(op) + "'");
    }
    if (src.shape() != out_spec.shape || src.shape().empty()) {
        throw std::runtime_error(
            "load plan: Dequantize source shape does not match output spec for '" +
            load_op_output(op) + "'");
    }
    if (scale.dtype() != DType::FP32 && scale.dtype() != DType::BF16) {
        throw std::runtime_error(
            "load plan: Dequantize scale for '" + load_op_output(op) +
            "' must be fp32 or bf16, got " +
            std::string(dtype_name(scale.dtype())));
    }

    DeviceTensor fp32_scale_scratch;
    const DeviceTensor* fp32_scale = &scale;
    if (scale.dtype() == DType::BF16) {
        fp32_scale_scratch = DeviceTensor::allocate(DType::FP32, scale.shape());
        kernels::launch_cast_bf16_to_fp32(
            scale.data(), fp32_scale_scratch.data(),
            scale.numel(), /*stream=*/0);
        fp32_scale = &fp32_scale_scratch;
    }

    DeviceTensor out = DeviceTensor::allocate(out_spec.dtype, out_spec.shape);
    if (fp32_scale->numel() == 1) {
        float host_scale = 0.f;
        CUDA_CHECK(cudaMemcpyAsync(
            &host_scale, fp32_scale->data(), sizeof(float),
            cudaMemcpyDeviceToHost, /*stream=*/0));
        CUDA_CHECK(cudaStreamSynchronize(/*stream=*/0));
        kernels::launch_dequant_fp8_e4m3_to_bf16(
            static_cast<const std::uint8_t*>(src.data()),
            out.data(), host_scale, src.numel(), /*stream=*/0);
    } else {
        const auto rows = static_cast<int>(src.shape()[0]);
        const auto cols = static_cast<int>(src.numel() /
                                           static_cast<std::size_t>(rows));
        if (fp32_scale->numel() != static_cast<std::size_t>(rows)) {
            throw std::runtime_error(
                "load plan: Dequantize per-channel scale length does not "
                "match source rows for '" + load_op_output(op) + "'");
        }
        kernels::launch_dequant_fp8_e4m3_to_bf16_per_channel(
            static_cast<const std::uint8_t*>(src.data()),
            out.data(),
            static_cast<const float*>(fp32_scale->data()),
            rows, cols, /*stream=*/0);
    }

    const std::uint64_t bytes = out.nbytes();
    weights.insert(load_op_output(op), std::move(out), out_spec);
    return bytes;
}

std::uint64_t split_interleaved_to_outputs(
    const LoadOp& op,
    const LoadPlan& plan,
    WeightStoreBuilder& weights,
    int tp_rank,
    int tp_size)
{
    const TensorSpec& first_spec = tensor_spec_for(plan, load_op_output(op));
    const TensorSpec& second_spec = tensor_spec_for(plan, load_op_secondary_output(op));
    const DeviceTensor& src = weights.get(first_input(op));
    if (src.dtype() != DType::BF16 ||
        first_spec.dtype != DType::BF16 ||
        second_spec.dtype != DType::BF16) {
        throw std::runtime_error(
            "load plan: SplitInterleaved currently supports bf16 tensors for '" +
            load_op_output(op) + "'");
    }
    if (first_spec.shape != second_spec.shape) {
        throw std::runtime_error(
            "load plan: SplitInterleaved output shapes differ for '" +
            load_op_output(op) + "'");
    }
    if (src.shape().size() != first_spec.shape.size() ||
        (src.shape().size() != 2 && src.shape().size() != 3)) {
        throw std::runtime_error(
            "load plan: SplitInterleaved expects rank-2 or rank-3 source for '" +
            load_op_output(op) + "'");
    }
    const auto& src_shape = src.shape();
    const auto& out_shape = first_spec.shape;
    const std::int64_t E = src_shape[0];
    const std::int64_t two_I_full = src_shape[1];
    if (two_I_full % 2 != 0 || E != out_shape[0]) {
        throw std::runtime_error(
            "load plan: SplitInterleaved source shape mismatch for '" +
            load_op_output(op) + "'");
    }
    const std::int64_t I_full = two_I_full / 2;
    const std::int64_t I_local = out_shape[1];
    if (I_local <= 0 || I_full % I_local != 0) {
        throw std::runtime_error(
            "load plan: SplitInterleaved local size mismatch for '" +
            load_op_output(op) + "'");
    }
    std::int64_t start = 0;
    if (tp_size > 1 && load_op_shard_axis(op) == 1) {
        if (I_full % tp_size != 0 || I_local != I_full / tp_size) {
            throw std::runtime_error(
                "load plan: SplitInterleaved cannot shard intermediate axis for '" +
                load_op_output(op) + "'");
        }
        start = static_cast<std::int64_t>(tp_rank) * I_local;
    } else if (I_local != I_full) {
        throw std::runtime_error(
            "load plan: SplitInterleaved output is sharded but op has no "
            "matching shard_axis for '" + load_op_output(op) + "'");
    }

    DeviceTensor first = DeviceTensor::allocate(first_spec.dtype, first_spec.shape);
    DeviceTensor second = DeviceTensor::allocate(second_spec.dtype, second_spec.shape);
    const std::size_t elem = dtype_bytes(DType::BF16);
    if (src_shape.size() == 3) {
        const std::int64_t H = src_shape[2];
        if (out_shape.size() != 3 || out_shape[2] != H) {
            throw std::runtime_error(
                "load plan: SplitInterleaved matrix output mismatch for '" +
                load_op_output(op) + "'");
        }
        const std::size_t src_expert_bytes =
            static_cast<std::size_t>(two_I_full) *
            static_cast<std::size_t>(H) * elem;
        const std::size_t out_expert_bytes =
            static_cast<std::size_t>(I_local) *
            static_cast<std::size_t>(H) * elem;
        for (std::int64_t e = 0; e < E; ++e) {
            const auto* src_e = static_cast<const std::uint8_t*>(src.data()) +
                static_cast<std::size_t>(e) * src_expert_bytes +
                static_cast<std::size_t>(2 * start) *
                static_cast<std::size_t>(H) * elem;
            auto* first_e = static_cast<std::uint8_t*>(first.data()) +
                static_cast<std::size_t>(e) * out_expert_bytes;
            auto* second_e = static_cast<std::uint8_t*>(second.data()) +
                static_cast<std::size_t>(e) * out_expert_bytes;
            kernels::launch_deinterleave_rows_bf16(
                src_e, first_e, second_e,
                static_cast<int>(I_local),
                static_cast<int>(H),
                /*stream=*/0);
        }
    } else {
        if (out_shape.size() != 2) {
            throw std::runtime_error(
                "load plan: SplitInterleaved vector output mismatch for '" +
                load_op_output(op) + "'");
        }
        const std::size_t src_expert_bytes =
            static_cast<std::size_t>(two_I_full) * elem;
        const std::size_t out_expert_bytes =
            static_cast<std::size_t>(I_local) * elem;
        for (std::int64_t e = 0; e < E; ++e) {
            const auto* src_e = static_cast<const std::uint8_t*>(src.data()) +
                static_cast<std::size_t>(e) * src_expert_bytes +
                static_cast<std::size_t>(2 * start) * elem;
            auto* first_e = static_cast<std::uint8_t*>(first.data()) +
                static_cast<std::size_t>(e) * out_expert_bytes;
            auto* second_e = static_cast<std::uint8_t*>(second.data()) +
                static_cast<std::size_t>(e) * out_expert_bytes;
            kernels::launch_deinterleave_vec_bf16(
                src_e, first_e, second_e,
                static_cast<int>(I_local),
                /*stream=*/0);
        }
    }

    const std::uint64_t bytes = first.nbytes() + second.nbytes();
    weights.insert(load_op_output(op), std::move(first), first_spec);
    weights.insert(load_op_secondary_output(op), std::move(second), second_spec);
    return bytes;
}

std::uint64_t fuse_moe_experts_to_outputs(
    const LoadOp& op,
    const LoadPlan& plan,
    SafetensorsLoader& loader,
    WeightStoreBuilder& weights,
    int tp_rank,
    int tp_size)
{
    const TensorSpec& gate_up_spec = tensor_spec_for(plan, load_op_output(op));
    const TensorSpec& down_spec =
        tensor_spec_for(plan, load_op_secondary_output(op));
    if ((!load_op_inputs(op).empty() && load_op_inputs(op).size() % 3 != 0) ||
        (!load_op_row_sources(op).empty() && load_op_row_sources(op).size() % 3 != 0) ||
        (load_op_inputs(op).empty() == load_op_row_sources(op).empty())) {
        throw std::runtime_error(
            "load plan: FuseMoeExperts expects gate/up/down triples for '" +
            load_op_output(op) + "'");
    }
    const std::int64_t E =
        load_op_row_sources(op).empty()
            ? static_cast<std::int64_t>(load_op_inputs(op).size() / 3)
            : static_cast<std::int64_t>(load_op_row_sources(op).size() / 3);
    if (gate_up_spec.dtype != DType::BF16 ||
        down_spec.dtype != DType::BF16 ||
        gate_up_spec.shape.size() != 3 ||
        down_spec.shape.size() != 3 ||
        gate_up_spec.shape[0] != E ||
        down_spec.shape[0] != E) {
        throw std::runtime_error(
            "load plan: FuseMoeExperts output spec mismatch for '" +
            load_op_output(op) + "'");
    }

    const std::int64_t I = gate_up_spec.shape[1] / 2;
    const std::int64_t H = gate_up_spec.shape[2];
    const std::int64_t I_down = down_spec.shape[2];
    if (gate_up_spec.shape !=
            std::vector<std::int64_t>{E, 2 * I, H} ||
        down_spec.shape !=
            std::vector<std::int64_t>{E, H, I_down}) {
        throw std::runtime_error(
            "load plan: FuseMoeExperts output shapes do not match inputs for '" +
            load_op_output(op) + "'");
    }

    DeviceTensor gate_up =
        DeviceTensor::allocate(gate_up_spec.dtype, gate_up_spec.shape);
    DeviceTensor down =
        DeviceTensor::allocate(down_spec.dtype, down_spec.shape);

    const std::size_t elem = dtype_bytes(DType::BF16);
    const std::size_t proj_bytes =
        static_cast<std::size_t>(I) * static_cast<std::size_t>(H) * elem;
    const std::size_t gate_up_expert_bytes = 2 * proj_bytes;
    const std::size_t down_expert_bytes =
        static_cast<std::size_t>(H) *
        static_cast<std::size_t>(I_down) * elem;

    for (std::int64_t e = 0; e < E; ++e) {
        auto* gate_up_e = static_cast<std::uint8_t*>(gate_up.data()) +
            static_cast<std::size_t>(e) * gate_up_expert_bytes;
        auto* down_e = static_cast<std::uint8_t*>(down.data()) +
            static_cast<std::size_t>(e) * down_expert_bytes;

        if (!load_op_row_sources(op).empty()) {
            const auto& gate_src = load_op_row_sources(op)[static_cast<std::size_t>(e) * 3];
            const auto& up_src = load_op_row_sources(op)[static_cast<std::size_t>(e) * 3 + 1];
            const auto& down_src = load_op_row_sources(op)[static_cast<std::size_t>(e) * 3 + 2];
            loader.copy_strided_to_device(
                gate_src.raw_name,
                tp_size > 1
                    ? std::vector<TensorSlice>{
                          TensorSlice{0, static_cast<std::int64_t>(tp_rank) * I, I}}
                    : std::vector<TensorSlice>{},
                gate_up_e,
                {I, H});
            loader.copy_strided_to_device(
                up_src.raw_name,
                tp_size > 1
                    ? std::vector<TensorSlice>{
                          TensorSlice{0, static_cast<std::int64_t>(tp_rank) * I, I}}
                    : std::vector<TensorSlice>{},
                gate_up_e + proj_bytes,
                {I, H});
            loader.copy_strided_to_device(
                down_src.raw_name,
                tp_size > 1
                    ? std::vector<TensorSlice>{
                          TensorSlice{1, static_cast<std::int64_t>(tp_rank) * I_down, I_down}}
                    : std::vector<TensorSlice>{},
                down_e,
                {H, I_down});
        } else {
            const DeviceTensor& gate =
                weights.get(load_op_inputs(op)[static_cast<std::size_t>(e) * 3]);
            const DeviceTensor& up =
                weights.get(load_op_inputs(op)[static_cast<std::size_t>(e) * 3 + 1]);
            const DeviceTensor& down_src =
                weights.get(load_op_inputs(op)[static_cast<std::size_t>(e) * 3 + 2]);
            if (gate.dtype() != DType::BF16 || up.dtype() != DType::BF16 ||
                down_src.dtype() != DType::BF16 ||
                gate.shape() != std::vector<std::int64_t>{I, H} ||
                up.shape() != std::vector<std::int64_t>{I, H} ||
                down_src.shape() != std::vector<std::int64_t>{H, I_down}) {
                throw std::runtime_error(
                    "load plan: FuseMoeExperts expert input mismatch for '" +
                    load_op_output(op) + "'");
            }
            CUDA_CHECK(cudaMemcpyAsync(
                gate_up_e, gate.data(), proj_bytes,
                cudaMemcpyDeviceToDevice, /*stream=*/0));
            CUDA_CHECK(cudaMemcpyAsync(
                gate_up_e + proj_bytes, up.data(), proj_bytes,
                cudaMemcpyDeviceToDevice, /*stream=*/0));
            CUDA_CHECK(cudaMemcpyAsync(
                down_e, down_src.data(), down_expert_bytes,
                cudaMemcpyDeviceToDevice, /*stream=*/0));
        }
    }

    const std::uint64_t bytes = gate_up.nbytes() + down.nbytes();
    weights.insert(load_op_output(op), std::move(gate_up), gate_up_spec);
    weights.insert(load_op_secondary_output(op), std::move(down), down_spec);
    return bytes;
}

int checked_int_dim(std::int64_t value, const std::string& name) {
    if (value <= 0 ||
        value > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "load plan: dimension out of range for '" + name + "'");
    }
    return static_cast<int>(value);
}

std::uint64_t repack_quant_to_outputs(
    const LoadOp& op,
    const LoadPlan& plan,
    WeightStoreBuilder& weights)
{
#ifndef PIE_CUDA_HAS_MARLIN
    throw std::runtime_error(
        "load plan: RepackQuant for '" + load_op_output(op) +
        "' requires Marlin, but this build was configured without "
        "PIE_CUDA_BUILD_MARLIN=ON.");
#else
    const TensorSpec& out_spec = tensor_spec_for(plan, load_op_output(op));
    const TensorSpec& scale_spec =
        tensor_spec_for(plan, load_op_secondary_output(op));
    const bool is_gptq = out_spec.quant.format == QuantFormat::GptqInt4;
    const bool is_awq = out_spec.quant.format == QuantFormat::AwqInt4;
    if ((!is_gptq && !is_awq) ||
        out_spec.quant.granularity != QuantGranularity::PerGroup ||
        out_spec.quant.scale_tensor != load_op_secondary_output(op)) {
        throw std::runtime_error(
            "load plan: RepackQuant supports GPTQ/AWQ int4 with "
            "per-group scale metadata for '" + load_op_output(op) + "'");
    }
    if (out_spec.dtype != DType::INT4_PACKED ||
        out_spec.layout != TensorLayoutKind::QuantPacked) {
        throw std::runtime_error(
            "load plan: RepackQuant output spec must be QuantPacked int4 for '" +
            load_op_output(op) + "'");
    }
    if (scale_spec.dtype != DType::BF16 ||
        scale_spec.layout != TensorLayoutKind::Dense) {
        throw std::runtime_error(
            "load plan: RepackQuant scale output must be dense bf16 for '" +
            load_op_output(op) + "'");
    }

    const DeviceTensor& qweight = weights.get(load_op_inputs(op).at(0));
    if (qweight.dtype() != DType::INT32 || qweight.shape().size() != 2) {
        throw std::runtime_error(
            "load plan: RepackQuant source must be 2-D int32 for '" +
            load_op_output(op) + "'");
    }

    if (is_awq) {
        if (load_op_inputs(op).size() != 3) {
            throw std::runtime_error(
                "load plan: AWQ RepackQuant requires qweight, qzeros, "
                "and scales inputs for '" + load_op_output(op) + "'");
        }
        if (out_spec.quant.zero_point_tensor.empty()) {
            throw std::runtime_error(
                "load plan: AWQ RepackQuant output is missing zero-point "
                "metadata for '" + load_op_output(op) + "'");
        }
        const TensorSpec& zero_spec =
            tensor_spec_for(plan, out_spec.quant.zero_point_tensor);
        const DeviceTensor& qzeros = weights.get(load_op_inputs(op).at(1));
        const DeviceTensor& scales = weights.get(load_op_inputs(op).at(2));
        if (qzeros.dtype() != DType::INT32 || qzeros.shape().size() != 2 ||
            scales.shape().size() != 2) {
            throw std::runtime_error(
                "load plan: AWQ RepackQuant qzeros/scales mismatch for '" +
                load_op_output(op) + "'");
        }

        const int size_k =
            checked_int_dim(qweight.shape()[0], load_op_output(op));
        const int size_n =
            checked_int_dim(qweight.shape()[1] * 8, load_op_output(op));
        const int groups =
            checked_int_dim(scales.shape()[0], load_op_secondary_output(op));
        if (size_k % 16 != 0 || size_n % 64 != 0 ||
            qzeros.shape() !=
                std::vector<std::int64_t>{groups, size_n / 8} ||
            scales.shape()[1] != size_n ||
            scale_spec.shape != scales.shape() ||
            zero_spec.dtype != DType::INT32 ||
            zero_spec.shape != qzeros.shape() ||
            out_spec.shape != std::vector<std::int64_t>{
                static_cast<std::int64_t>(size_k / 16),
                static_cast<std::int64_t>(size_n) * 8}) {
            throw std::runtime_error(
                "load plan: AWQ RepackQuant shape mismatch for '" +
                load_op_output(op) + "'");
        }

        DeviceTensor gptq_qweight =
            DeviceTensor::allocate(DType::INT32, {size_k / 8, size_n});
        kernels::launch_awq_qweight_to_gptq_w4(
            qweight.data(), gptq_qweight.data(), size_k, size_n,
            /*stream=*/0);

        DeviceTensor packed =
            DeviceTensor::allocate(out_spec.dtype, out_spec.shape);
        marlin::launch_gptq_repack_w4_no_perm(
            gptq_qweight.data(), packed.data(), size_k, size_n,
            /*stream=*/0);

        DeviceTensor bf16_scales =
            DeviceTensor::allocate(scale_spec.dtype, scale_spec.shape);
        if (scales.dtype() == DType::BF16) {
            CUDA_CHECK(cudaMemcpyAsync(
                bf16_scales.data(), scales.data(), scales.nbytes(),
                cudaMemcpyDeviceToDevice, /*stream=*/0));
        } else if (scales.dtype() == DType::FP16) {
            kernels::launch_cast_fp16_to_bf16(
                scales.data(), bf16_scales.data(), scales.numel(),
                /*stream=*/0);
        } else if (scales.dtype() == DType::FP32) {
            kernels::launch_cast_fp32_to_bf16(
                scales.data(), bf16_scales.data(), scales.numel(),
                /*stream=*/0);
        } else {
            throw std::runtime_error(
                "load plan: AWQ RepackQuant scales must be bf16/fp16/fp32 "
                "for '" + load_op_output(op) + "'");
        }
        kernels::launch_marlin_permute_scales_bf16(
            bf16_scales.data(),
            groups, size_n, out_spec.quant.group_size, size_k,
            /*stream=*/0);

        DeviceTensor qzeros_marlin =
            DeviceTensor::allocate(zero_spec.dtype, zero_spec.shape);
        kernels::launch_awq_qzero_to_marlin_w4(
            qzeros.data(), qzeros_marlin.data(), groups, size_n,
            /*stream=*/0);

        const std::uint64_t bytes =
            packed.nbytes() + bf16_scales.nbytes() + qzeros_marlin.nbytes();
        weights.insert(load_op_output(op), std::move(packed), out_spec);
        weights.insert(load_op_secondary_output(op), std::move(bf16_scales), scale_spec);
        weights.insert(
            out_spec.quant.zero_point_tensor,
            std::move(qzeros_marlin),
            zero_spec);
        return bytes;
    }

    const DeviceTensor& scales = weights.get(load_op_inputs(op).at(1));
    if (scales.dtype() != DType::FP16 || scales.shape().size() != 2) {
        throw std::runtime_error(
            "load plan: RepackQuant GPTQ scales must be 2-D fp16 for '" +
            load_op_output(op) + "'");
    }

    const int size_k =
        checked_int_dim(qweight.shape()[0] * 8, load_op_output(op));
    const int size_n =
        checked_int_dim(qweight.shape()[1], load_op_output(op));
    const int groups =
        checked_int_dim(scales.shape()[0], load_op_secondary_output(op));
    if (scales.shape()[1] != size_n) {
        throw std::runtime_error(
            "load plan: RepackQuant scale columns do not match qweight for '" +
            load_op_output(op) + "'");
    }
    if (size_k % 16 != 0 ||
        out_spec.shape != std::vector<std::int64_t>{
            static_cast<std::int64_t>(size_k / 16),
            static_cast<std::int64_t>(size_n) * 8}) {
        throw std::runtime_error(
            "load plan: RepackQuant packed output shape mismatch for '" +
            load_op_output(op) + "'");
    }
    if (scale_spec.shape != scales.shape()) {
        throw std::runtime_error(
            "load plan: RepackQuant scale output shape mismatch for '" +
            load_op_output(op) + "'");
    }

    DeviceTensor packed =
        DeviceTensor::allocate(out_spec.dtype, out_spec.shape);
    marlin::launch_gptq_repack_w4_no_perm(
        qweight.data(), packed.data(), size_k, size_n, /*stream=*/0);

    DeviceTensor bf16_scales =
        DeviceTensor::allocate(scale_spec.dtype, scale_spec.shape);
    kernels::launch_cast_fp16_to_bf16(
        scales.data(), bf16_scales.data(), scales.numel(), /*stream=*/0);
    kernels::launch_marlin_permute_scales_bf16(
        bf16_scales.data(),
        groups, size_n, out_spec.quant.group_size, size_k,
        /*stream=*/0);

    const std::uint64_t bytes = packed.nbytes() + bf16_scales.nbytes();
    weights.insert(load_op_output(op), std::move(packed), out_spec);
    weights.insert(load_op_secondary_output(op), std::move(bf16_scales), scale_spec);
    return bytes;
#endif
}

void validate_materialized_tensors(
    const LoadPlan& plan,
    const WeightStore& weights)
{
    for (const auto& [name, spec] : plan.tensors) {
        if (spec.ownership == TensorOwnershipKind::Temporary) continue;
        if (weights.find(name) == weights.end()) {
            throw std::runtime_error(
                "load plan: materializer did not produce tensor '" + name + "'");
        }
        const DeviceTensor& tensor = weights.get(name);
        if (tensor.dtype() != spec.dtype) {
            throw std::runtime_error(
                "load plan: tensor '" + name + "' dtype mismatch: planned " +
                std::string(dtype_name(spec.dtype)) + ", got " +
                std::string(dtype_name(tensor.dtype())));
        }
        if (tensor.shape() != spec.shape) {
            throw std::runtime_error(
                "load plan: tensor '" + name +
                "' shape mismatch after materialization");
        }
        if (spec.ownership == TensorOwnershipKind::BorrowedView &&
            weights.find(spec.backing_tensor) == weights.end()) {
            throw std::runtime_error(
                "load plan: view '" + name + "' backing tensor '" +
                spec.backing_tensor + "' was not materialized");
        }
    }
}

std::uint64_t materialize_pack_rows(
    const LoadOp& op,
    const LoadPlan& plan,
    SafetensorsLoader& loader,
    WeightStoreBuilder& weights,
    int tp_rank,
    int tp_size)
{
    if (load_op_row_sources(op).empty()) {
        throw std::runtime_error("load plan: PackRows op has no sources for " +
                                 load_op_output(op));
    }

    std::int64_t total_rows = 0;
    std::int64_t cols = -1;
    DType dtype = DType::BF16;
    bool dtype_set = false;
    std::vector<std::int64_t> source_rows;
    source_rows.reserve(load_op_row_sources(op).size());

    for (const auto& src : load_op_row_sources(op)) {
        const auto& info = loader.info(src.raw_name);
        if (info.shape.size() != 2) {
            throw std::runtime_error("load plan: PackRows source is not 2-D: " +
                                     src.raw_name);
        }
        auto shape = info.shape;
        if (tp_size > 1 && load_op_shard_axis(op) >= 0) {
            if (load_op_shard_axis(op) >= static_cast<int>(shape.size()) ||
                shape[load_op_shard_axis(op)] % tp_size != 0) {
                throw std::runtime_error(
                    "load plan: PackRows source cannot be sharded cleanly: " +
                    src.raw_name);
            }
            shape[load_op_shard_axis(op)] /= tp_size;
        }
        if (!dtype_set) {
            dtype = info.dtype;
            cols = shape[1];
            dtype_set = true;
        } else if (info.dtype != dtype || shape[1] != cols) {
            throw std::runtime_error("load plan: PackRows source mismatch for " +
                                     load_op_output(op));
        }
        total_rows += shape[0];
        source_rows.push_back(shape[0]);
    }

    DeviceTensor packed = DeviceTensor::allocate(dtype, {total_rows, cols});
    auto* dst = static_cast<std::uint8_t*>(packed.data());
    std::int64_t row_offset = 0;
    std::size_t byte_offset = 0;

    for (std::size_t i = 0; i < load_op_row_sources(op).size(); ++i) {
        const auto& src = load_op_row_sources(op)[i];
        std::vector<TensorSlice> slices;
        std::vector<std::int64_t> expected_shape =
            loader.info(src.raw_name).shape;
        if (tp_size > 1 && load_op_shard_axis(op) >= 0) {
            const auto axis = static_cast<std::size_t>(load_op_shard_axis(op));
            const std::int64_t local = expected_shape[axis] / tp_size;
            slices.push_back(TensorSlice{
                load_op_shard_axis(op),
                static_cast<std::int64_t>(tp_rank) * local,
                local});
            expected_shape[axis] = local;
        }
        if (expected_shape != std::vector<std::int64_t>{source_rows[i], cols}) {
            throw std::runtime_error(
                "load plan: PackRows expected shape mismatch for " +
                src.raw_name);
        }
        loader.copy_strided_to_device(
            src.raw_name, slices, dst + byte_offset, expected_shape);
        byte_offset +=
            static_cast<std::size_t>(source_rows[i]) *
            static_cast<std::size_t>(cols) * dtype_bytes(dtype);
    }

    const std::uint64_t loaded_bytes = packed.nbytes();
    weights.insert(
        load_op_output(op),
        std::move(packed),
        tensor_spec_for(plan, load_op_output(op)));

    for (std::size_t i = 0; i < load_op_row_sources(op).size(); ++i) {
        insert_row_view(
            weights,
            tensor_spec_for(plan, load_op_row_sources(op)[i].view_name),
            load_op_output(op),
            load_op_row_sources(op)[i].view_name,
            row_offset,
            source_rows[i],
            cols);
        row_offset += source_rows[i];
    }

    return loaded_bytes;
}

}  // namespace

Materializer::Materializer(
    SafetensorsLoader& loader,
    WeightStore& weights,
    int tp_rank,
    int tp_size,
    NcclComm* tp_comm) noexcept
    : loader_(loader),
      weights_(weights),
      builder_(weights),
      tp_rank_(tp_rank),
      tp_size_(tp_size),
      tp_comm_(tp_comm)
{}

MaterializedLoadPlan Materializer::run(const LoadPlan& plan)
{
    validate_load_plan(plan);

    MaterializedLoadPlan result;
    result.packed_qkv_groups = plan.packed_qkv_groups;
    result.packed_gate_up_groups = plan.packed_gate_up_groups;
    result.planned_tensor_count = plan.tensors.size();

    for (const auto& op : plan.ops) {
        switch (op.kind) {
        case LoadOpKind::Copy:
            result.loaded_bytes += copy_raw_to_output(
                op, plan, loader_, builder_, tp_rank_, tp_size_);
            break;
        case LoadOpKind::Shard:
            result.loaded_bytes += copy_raw_to_output(
                op, plan, loader_, builder_, tp_rank_, tp_size_);
            break;
        case LoadOpKind::Read:
            result.loaded_bytes += copy_raw_to_output(
                op, plan, loader_, builder_, tp_rank_, tp_size_);
            break;
        case LoadOpKind::RowRangeShard:
            result.loaded_bytes += copy_row_range_to_output(
                op, plan, loader_, builder_, tp_rank_, tp_size_);
            break;
        case LoadOpKind::MoeGateUpShard:
            result.loaded_bytes += copy_moe_gate_up_to_output(
                op, plan, loader_, builder_, tp_rank_, tp_size_);
            break;
        case LoadOpKind::MoeDownShard:
            result.loaded_bytes += copy_moe_down_to_output(
                op, plan, loader_, builder_, tp_rank_, tp_size_);
            break;
        case LoadOpKind::PackRows:
            result.loaded_bytes += materialize_pack_rows(
                op, plan, loader_, builder_, tp_rank_, tp_size_);
            break;
        case LoadOpKind::Slice:
            result.loaded_bytes += (load_op_slice_axis(op) >= 0)
                ? slice_axis_to_output(
                      op, plan, builder_, tp_rank_, tp_size_)
                : slice_rows_to_output(op, plan, builder_);
            break;
        case LoadOpKind::Cast: {
            const TensorSpec& out_spec = tensor_spec_for(plan, load_op_output(op));
            cast_tensor_to_output(
                builder_.get(first_input(op)), builder_, out_spec);
            result.loaded_bytes += builder_.get(load_op_output(op)).nbytes();
            break;
        }
        case LoadOpKind::Concat:
            result.loaded_bytes += concat_rows_to_output(op, plan, builder_);
            break;
        case LoadOpKind::View:
        case LoadOpKind::Alias:
            view_or_alias_to_output(op, plan, builder_);
            break;
        case LoadOpKind::Drop:
            if (load_op_inputs(op).empty()) {
                builder_.erase(load_op_output(op));
            } else {
                for (const auto& input : load_op_inputs(op)) {
                    builder_.erase(input);
                }
            }
            break;
        case LoadOpKind::AttachQuantMeta:
            attach_quant_meta_for_output(op, plan, builder_);
            break;
        case LoadOpKind::QuantizeRuntime: {
            const RuntimeQuantResult qr = quantize_runtime_to_output(
                op, plan, builder_, tp_comm_, tp_size_);
            result.loaded_bytes += qr.bytes_after;
            result.runtime_quantized_weights += 1;
            result.runtime_quant_bytes_before += qr.bytes_before;
            result.runtime_quant_bytes_after += qr.bytes_after;
            break;
        }
        case LoadOpKind::Materialize: {
            const TensorSpec& out_spec = tensor_spec_for(plan, load_op_output(op));
            result.loaded_bytes += copy_tensor_to_output(
                builder_.get(first_input(op)), builder_, out_spec);
            break;
        }
        case LoadOpKind::Dequantize:
            result.loaded_bytes += dequantize_to_output(op, plan, builder_);
            break;
        case LoadOpKind::SplitInterleaved:
            result.loaded_bytes += split_interleaved_to_outputs(
                op, plan, builder_, tp_rank_, tp_size_);
            break;
        case LoadOpKind::RepackQuant:
            result.loaded_bytes += repack_quant_to_outputs(
                op, plan, builder_);
            break;
        case LoadOpKind::FuseMoeExperts:
            result.loaded_bytes += fuse_moe_experts_to_outputs(
                op, plan, loader_, builder_, tp_rank_, tp_size_);
            break;
        }
    }

    validate_materialized_tensors(plan, weights_);
    builder_.finalize();

    return result;
}

MaterializedLoadPlan materialize_load_plan(
    const LoadPlan& plan,
    SafetensorsLoader& loader,
    WeightStore& weights,
    int tp_rank,
    int tp_size,
    NcclComm* tp_comm)
{
    return Materializer(loader, weights, tp_rank, tp_size, tp_comm).run(plan);
}

}  // namespace pie_cuda_driver
