#include "loader/load_executor.hpp"

#include <cuda_runtime.h>
#include <nccl.h>

#include <cstdint>
#include <limits>
#include <algorithm>
#include <stdexcept>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "cuda_check.hpp"
#include "distributed.hpp"
#include "kernels/deinterleave.hpp"
#include "kernels/dequant_fp8.hpp"
#include "kernels/dequant_fp4.hpp"
#include "kernels/dtype_cast.hpp"
#include "kernels/quant_bf16_to_fp8.hpp"
#include "loader/byte_source.hpp"
#include "loader/storage_executor.hpp"
#include "loader/storage_program.hpp"
#include "loader/safetensors.hpp"
#include "model/weight_store.hpp"
#include "tensor.hpp"

#ifdef PIE_CUDA_HAS_MARLIN
#include "marlin_wrapper.hpp"
#endif

namespace pie_cuda_driver {

namespace {

const TensorDecl& tensor_spec_for(
    const LayoutPlan& plan,
    const std::string& name);

std::uint64_t shape_nbytes(
    DType dtype,
    const std::vector<std::int64_t>& shape)
{
    std::uint64_t n = 1;
    for (const auto dim : shape) {
        n *= static_cast<std::uint64_t>(dim);
    }
    return n * static_cast<std::uint64_t>(dtype_bytes(dtype));
}

struct CudaLoadMemoryTelemetry {
    std::uint64_t total = 0;
    std::uint64_t free_before = 0;
    std::uint64_t min_free = 0;
    std::uint64_t free_after = 0;
    std::size_t samples = 0;

    void sample() noexcept
    {
        std::size_t free_bytes = 0;
        std::size_t total_bytes = 0;
        if (cudaMemGetInfo(&free_bytes, &total_bytes) != cudaSuccess) {
            return;
        }
        if (samples == 0) {
            free_before = static_cast<std::uint64_t>(free_bytes);
            min_free = free_before;
            total = static_cast<std::uint64_t>(total_bytes);
        } else {
            min_free = std::min<std::uint64_t>(
                min_free,
                static_cast<std::uint64_t>(free_bytes));
            total = std::max<std::uint64_t>(
                total,
                static_cast<std::uint64_t>(total_bytes));
        }
        free_after = static_cast<std::uint64_t>(free_bytes);
        ++samples;
    }
};

void sample_cuda_load_memory(void* context) noexcept
{
    static_cast<CudaLoadMemoryTelemetry*>(context)->sample();
}

class ScopedDeviceTensorMemoryCallback {
public:
    explicit ScopedDeviceTensorMemoryCallback(CudaLoadMemoryTelemetry& telemetry)
    {
        set_device_tensor_memory_callback(sample_cuda_load_memory, &telemetry);
    }
    ~ScopedDeviceTensorMemoryCallback()
    {
        set_device_tensor_memory_callback(nullptr, nullptr);
    }

    ScopedDeviceTensorMemoryCallback(const ScopedDeviceTensorMemoryCallback&) = delete;
    ScopedDeviceTensorMemoryCallback& operator=(
        const ScopedDeviceTensorMemoryCallback&) = delete;
};

std::vector<ExtentWrite> extent_writes_for_instruction(
    const StorageProgram& storage_program,
    const StorageInstr& instr)
{
    std::vector<ExtentWrite> writes;
    writes.reserve(instr.extent_write_indices.size());
    for (const auto write_index : instr.extent_write_indices) {
        if (write_index >= storage_program.extent_writes.size()) {
            throw std::runtime_error(
                "storage program: instruction extent-write index out of range");
        }
        writes.push_back(storage_program.extent_writes[write_index]);
    }
    return writes;
}

std::uint64_t materialize_extent_write_instruction_run(
    const LayoutPlan& plan,
    const StorageProgram& storage_program,
    std::size_t first_step,
    std::size_t last_step_exclusive,
    StorageWriteExecutor& executor,
    WeightStoreBuilder& weights)
{
    std::vector<std::size_t> write_indices;
    for (std::size_t i = first_step; i < last_step_exclusive; ++i) {
        const auto& instr = storage_program.schedule[i];
        if (instr.kind != StorageInstrKind::ExtentWrite) {
            throw std::runtime_error(
                "storage program: extent-write run contains non-write instruction");
        }
        write_indices.insert(
            write_indices.end(),
            instr.extent_write_indices.begin(),
            instr.extent_write_indices.end());
    }
    if (write_indices.empty()) {
        throw std::runtime_error("storage program: empty extent-write run");
    }

    std::unordered_set<std::size_t> run_write_indices(
        write_indices.begin(), write_indices.end());
    std::unordered_map<std::string, DeviceTensor> outputs;
    std::uint64_t allocated_bytes = 0;
    for (const auto write_index : write_indices) {
        if (write_index >= storage_program.extent_writes.size()) {
            throw std::runtime_error(
                "storage program: extent-write run index out of range");
        }
        const auto& write = storage_program.extent_writes[write_index];
        if (!outputs.contains(write.output_name)) {
            const TensorDecl& spec = tensor_spec_for(plan, write.output_name);
            DeviceTensor out = DeviceTensor::allocate(spec.dtype, spec.shape);
            allocated_bytes += out.nbytes();
            outputs.emplace(write.output_name, std::move(out));
        }
    }

    std::vector<StorageWriteDestination> destinations;
    destinations.reserve(write_indices.size());
    for (const auto write_index : storage_program.scheduled_extent_writes) {
        if (!run_write_indices.contains(write_index)) continue;
        const auto& write = storage_program.extent_writes[write_index];
        auto out = outputs.find(write.output_name);
        if (out == outputs.end()) {
            throw std::runtime_error(
                "storage program: missing destination allocation for '" +
                write.output_name + "'");
        }
        destinations.push_back(StorageWriteDestination{
            .write = &write,
            .dst_base = out->second.data(),
        });
    }
    if (destinations.size() != write_indices.size()) {
        throw std::runtime_error(
            "storage program: extent-write run schedule coverage mismatch");
    }
    executor.execute(destinations);

    for (auto& [name, tensor] : outputs) {
        weights.insert(name, std::move(tensor), tensor_spec_for(plan, name));
    }
    return allocated_bytes;
}

void cast_tile_to_output(
    const DeviceTensor& scratch,
    void* out_ptr,
    DType out_dtype)
{
    if (scratch.dtype() == out_dtype) {
        CUDA_CHECK(cudaMemcpyAsync(
            out_ptr, scratch.data(), scratch.nbytes(),
            cudaMemcpyDeviceToDevice, /*stream=*/0));
    } else if (scratch.dtype() == DType::FP16 && out_dtype == DType::BF16) {
        kernels::launch_cast_fp16_to_bf16(
            scratch.data(), out_ptr, scratch.numel(), /*stream=*/0);
    } else if (scratch.dtype() == DType::FP32 && out_dtype == DType::BF16) {
        kernels::launch_cast_fp32_to_bf16(
            scratch.data(), out_ptr, scratch.numel(), /*stream=*/0);
    } else if (scratch.dtype() == DType::BF16 && out_dtype == DType::FP32) {
        kernels::launch_cast_bf16_to_fp32(
            scratch.data(), out_ptr, scratch.numel(), /*stream=*/0);
    } else {
        throw std::runtime_error(
            "storage program: unsupported TileMap Cast " +
            std::string(dtype_name(scratch.dtype())) + " -> " +
            std::string(dtype_name(out_dtype)));
    }
}

std::vector<TensorSlice> tile_slices_for_leading_axis(
    const ExtentWrite& base_write,
    std::int64_t tile_start,
    std::int64_t tile_rows)
{
    std::vector<TensorSlice> slices = base_write.slices;
    for (auto& slice : slices) {
        if (slice.axis == 0) {
            slice.start += tile_start;
            slice.length = tile_rows;
            return slices;
        }
    }
    slices.push_back(TensorSlice{0, tile_start, tile_rows});
    return slices;
}

std::uint64_t materialize_tile_raw_cast(
    const TileMap& tile,
    const std::vector<ExtentWrite>& writes,
    const LayoutPlan& plan,
    CheckpointByteSource& byte_source,
    WeightStoreBuilder& weights,
    std::uint64_t tile_bytes)
{
    if (tile.kind != TileMapKind::Cast || tile.inputs.size() != 1) {
        throw std::runtime_error(
            "storage program: raw TileMap requires one Cast input for '" +
            tile.output_name + "'");
    }
    const TensorDecl& out_spec = tensor_spec_for(plan, tile.output_name);
    const TensorDecl& src_spec = tensor_spec_for(plan, tile.inputs.front());
    if (src_spec.shape != out_spec.shape) {
        throw std::runtime_error(
            "storage program: TileMap Cast source/output shape mismatch for '" +
            out_spec.name + "'");
    }
    if (writes.size() != 1 || writes[0].dst_offset_bytes != 0 ||
        writes[0].dst_shape != src_spec.shape) {
        throw std::runtime_error(
            "storage program: TileMap Cast expects one compact raw write for '" +
            out_spec.name + "'");
    }

    DeviceTensor out = DeviceTensor::allocate(out_spec.dtype, out_spec.shape);
    const ExtentWrite& base_write = writes[0];
    if (src_spec.shape.empty()) {
        DeviceTensor scratch = DeviceTensor::allocate(src_spec.dtype, {});
        byte_source.write_to_device(base_write, scratch.data());
        cast_tile_to_output(scratch, out.data(), out_spec.dtype);
    } else {
        std::int64_t rows = src_spec.shape[0];
        std::uint64_t row_bytes =
            shape_nbytes(src_spec.dtype, std::vector<std::int64_t>(
                src_spec.shape.begin() + 1, src_spec.shape.end()));
        if (src_spec.shape.size() == 1) {
            row_bytes = static_cast<std::uint64_t>(dtype_bytes(src_spec.dtype));
        }
        const std::int64_t rows_per_tile = std::max<std::int64_t>(
            1,
            static_cast<std::int64_t>(tile_bytes / std::max<std::uint64_t>(row_bytes, 1)));
        for (std::int64_t row = 0; row < rows; row += rows_per_tile) {
            const std::int64_t tile_rows = std::min(rows_per_tile, rows - row);
            auto tile_shape = src_spec.shape;
            tile_shape[0] = tile_rows;
            ExtentWrite tile_write = base_write;
            tile_write.slices = tile_slices_for_leading_axis(
                base_write, row, tile_rows);
            tile_write.dst_shape = tile_shape;
            tile_write.dst_offset_bytes = 0;
            tile_write.bytes = shape_nbytes(src_spec.dtype, tile_shape);
            DeviceTensor scratch =
                DeviceTensor::allocate(src_spec.dtype, tile_shape);
            byte_source.write_to_device(tile_write, scratch.data());
            auto* out_ptr = static_cast<std::uint8_t*>(out.data()) +
                static_cast<std::uint64_t>(row) *
                    shape_nbytes(out_spec.dtype, std::vector<std::int64_t>(
                        out_spec.shape.begin() + 1, out_spec.shape.end()));
            cast_tile_to_output(scratch, out_ptr, out_spec.dtype);
        }
    }
    const std::uint64_t bytes = out.nbytes();
    weights.insert(out_spec.name, std::move(out), out_spec);
    return bytes;
}

const TensorDecl& tensor_spec_for(
    const LayoutPlan& plan,
    const std::string& name)
{
    auto it = plan.tensors.find(name);
    if (it == plan.tensors.end()) {
        throw std::runtime_error(
            "layout plan: no TensorDecl for materialized tensor '" + name + "'");
    }
    return it->second;
}

struct ExecOp {
    std::string output_name;
    std::string secondary_output_name;
    std::vector<std::string> inputs;
    int shard_axis = -1;
    int slice_axis = -1;
    std::int64_t row_offset = 0;
    std::int64_t rows = 0;
    std::int64_t slice_start = 0;
    std::int64_t slice_length = 0;
    std::vector<TensorSourceRef> sources;
};

const std::string& exec_op_output(const ExecOp& op) {
    return op.output_name;
}

const std::string& exec_op_secondary_output(const ExecOp& op) {
    return op.secondary_output_name;
}

const std::vector<std::string>& exec_op_inputs(const ExecOp& op) {
    return op.inputs;
}

int exec_op_shard_axis(const ExecOp& op) {
    return op.shard_axis;
}

int exec_op_slice_axis(const ExecOp& op) {
    return op.slice_axis;
}

std::int64_t exec_op_row_offset(const ExecOp& op) {
    return op.row_offset;
}

std::int64_t exec_op_rows(const ExecOp& op) {
    return op.rows;
}

std::int64_t exec_op_slice_start(const ExecOp& op) {
    return op.slice_start;
}

std::int64_t exec_op_slice_length(const ExecOp& op) {
    return op.slice_length;
}

const std::vector<TensorSourceRef>& exec_op_sources(const ExecOp& op) {
    return op.sources;
}

const std::string& first_input(const ExecOp& op) {
    if (exec_op_inputs(op).empty()) {
        throw std::runtime_error(
            "layout plan: op for '" + exec_op_output(op) +
            "' requires at least one input");
    }
    return exec_op_inputs(op).front();
}

void insert_row_view(
    WeightStoreBuilder& weights,
    const TensorDecl& view_spec,
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
    const TensorDecl& out_spec)
{
    if (src.dtype() != out_spec.dtype || src.shape() != out_spec.shape) {
        throw std::runtime_error(
            "layout plan: Materialize source does not match TensorDecl for '" +
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
    const TensorDecl& out_spec)
{
    if (src.shape() != out_spec.shape) {
        throw std::runtime_error(
            "layout plan: Cast source shape does not match TensorDecl for '" +
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
            "layout plan: unsupported Cast " +
            std::string(dtype_name(src.dtype())) + " -> " +
            std::string(dtype_name(out_spec.dtype)) + " for '" +
            out_spec.name + "'");
    }
    weights.insert(out_spec.name, std::move(out), out_spec);
}

std::uint64_t concat_rows_to_output(
    const ExecOp& op,
    const LayoutPlan& plan,
    WeightStoreBuilder& weights)
{
    const TensorDecl& out_spec = tensor_spec_for(plan, exec_op_output(op));
    if (out_spec.shape.size() != 2) {
        throw std::runtime_error(
            "layout plan: Concat output must be 2-D: " + exec_op_output(op));
    }
    if (exec_op_inputs(op).empty()) {
        throw std::runtime_error(
            "layout plan: Concat op has no inputs for " + exec_op_output(op));
    }

    const std::int64_t out_cols = out_spec.shape[1];
    std::int64_t total_rows = 0;
    for (const auto& name : exec_op_inputs(op)) {
        const DeviceTensor& t = weights.get(name);
        if (t.dtype() != out_spec.dtype || t.shape().size() != 2 ||
            t.shape()[1] != out_cols) {
            throw std::runtime_error(
                "layout plan: Concat input mismatch for '" + exec_op_output(op) +
                "': " + name);
        }
        total_rows += t.shape()[0];
    }
    if (total_rows != out_spec.shape[0]) {
        throw std::runtime_error(
            "layout plan: Concat row count does not match TensorDecl for '" +
            exec_op_output(op) + "'");
    }

    DeviceTensor out = DeviceTensor::allocate(out_spec.dtype, out_spec.shape);
    auto* dst = static_cast<std::uint8_t*>(out.data());
    std::size_t byte_offset = 0;
    for (const auto& name : exec_op_inputs(op)) {
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
    weights.insert(exec_op_output(op), std::move(out), out_spec);
    return bytes;
}

std::uint64_t slice_rows_to_output(
    const ExecOp& op,
    const LayoutPlan& plan,
    WeightStoreBuilder& weights)
{
    const TensorDecl& out_spec = tensor_spec_for(plan, exec_op_output(op));
    if (out_spec.shape.size() != 2) {
        throw std::runtime_error(
            "layout plan: Slice output must be 2-D: " + exec_op_output(op));
    }
    const DeviceTensor& src = weights.get(first_input(op));
    if (src.dtype() != out_spec.dtype || src.shape().size() != 2 ||
        src.shape()[1] != out_spec.shape[1]) {
        throw std::runtime_error(
            "layout plan: Slice source mismatch for '" + exec_op_output(op) + "'");
    }
    const std::int64_t rows =
        exec_op_rows(op) > 0 ? exec_op_rows(op) : out_spec.shape[0];
    if (rows != out_spec.shape[0] || exec_op_row_offset(op) < 0 ||
        exec_op_row_offset(op) + rows > src.shape()[0]) {
        throw std::runtime_error(
            "layout plan: Slice range out of bounds for '" + exec_op_output(op) + "'");
    }

    DeviceTensor out = DeviceTensor::allocate(out_spec.dtype, out_spec.shape);
    const std::size_t row_bytes =
        static_cast<std::size_t>(out_spec.shape[1]) *
        dtype_bytes(out_spec.dtype);
    const auto* src8 = static_cast<const std::uint8_t*>(src.data()) +
        static_cast<std::size_t>(exec_op_row_offset(op)) * row_bytes;
    CUDA_CHECK(cudaMemcpyAsync(
        out.data(), src8, static_cast<std::size_t>(rows) * row_bytes,
        cudaMemcpyDeviceToDevice, /*stream=*/0));
    const std::uint64_t bytes = out.nbytes();
    weights.insert(exec_op_output(op), std::move(out), out_spec);
    return bytes;
}

std::uint64_t slice_axis_to_output(
    const ExecOp& op,
    const LayoutPlan& plan,
    WeightStoreBuilder& weights,
    int tp_rank,
    int tp_size)
{
    const TensorDecl& out_spec = tensor_spec_for(plan, exec_op_output(op));
    const DeviceTensor& src = weights.get(first_input(op));
    if (src.dtype() != out_spec.dtype) {
        throw std::runtime_error(
            "layout plan: Slice dtype mismatch for '" + exec_op_output(op) + "'");
    }
    if (exec_op_slice_axis(op) < 0 ||
        exec_op_slice_axis(op) >= static_cast<int>(src.shape().size())) {
        throw std::runtime_error(
            "layout plan: Slice axis out of range for '" + exec_op_output(op) + "'");
    }

    auto expected = src.shape();
    std::int64_t length = exec_op_slice_length(op) > 0
        ? exec_op_slice_length(op)
        : out_spec.shape.at(static_cast<std::size_t>(exec_op_slice_axis(op)));
    std::int64_t start = exec_op_slice_start(op);
    if (tp_size > 1 && exec_op_shard_axis(op) == exec_op_slice_axis(op)) {
        start += static_cast<std::int64_t>(tp_rank) * length;
    }
    if (length <= 0 || start < 0 ||
        start + length > src.shape()[static_cast<std::size_t>(exec_op_slice_axis(op))]) {
        throw std::runtime_error(
            "layout plan: Slice range out of bounds for '" + exec_op_output(op) + "'");
    }
    expected[static_cast<std::size_t>(exec_op_slice_axis(op))] = length;
    if (expected != out_spec.shape) {
        throw std::runtime_error(
            "layout plan: Slice output shape does not match TensorDecl for '" +
            exec_op_output(op) + "'");
    }

    std::int64_t outer = 1;
    for (int i = 0; i < exec_op_slice_axis(op); ++i) {
        outer *= src.shape()[static_cast<std::size_t>(i)];
    }
    std::int64_t inner = 1;
    for (std::size_t i = static_cast<std::size_t>(exec_op_slice_axis(op)) + 1;
         i < src.shape().size(); ++i) {
        inner *= src.shape()[i];
    }
    const std::int64_t axis_dim =
        src.shape()[static_cast<std::size_t>(exec_op_slice_axis(op))];
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
    weights.insert(exec_op_output(op), std::move(out), out_spec);
    return bytes;
}

void view_or_alias_to_output(
    const ExecOp& op,
    const LayoutPlan& plan,
    WeightStoreBuilder& weights)
{
    const TensorDecl& out_spec = tensor_spec_for(plan, exec_op_output(op));
    const DeviceTensor& src = weights.get(first_input(op));
    if (src.dtype() != out_spec.dtype) {
        throw std::runtime_error(
            "layout plan: View/Alias dtype mismatch for '" + exec_op_output(op) + "'");
    }

    void* ptr = const_cast<void*>(src.data());
    if (exec_op_rows(op) > 0 || exec_op_row_offset(op) > 0) {
        if (src.shape().size() != 2 || out_spec.shape.size() != 2 ||
            src.shape()[1] != out_spec.shape[1]) {
            throw std::runtime_error(
                "layout plan: row view requires matching 2-D tensors for '" +
                exec_op_output(op) + "'");
        }
        const std::int64_t rows = exec_op_rows(op) > 0 ? exec_op_rows(op) : out_spec.shape[0];
        if (rows != out_spec.shape[0] || exec_op_row_offset(op) < 0 ||
            exec_op_row_offset(op) + rows > src.shape()[0]) {
            throw std::runtime_error(
                "layout plan: row view range out of bounds for '" +
                exec_op_output(op) + "'");
        }
        const std::size_t row_bytes =
            static_cast<std::size_t>(src.shape()[1]) * dtype_bytes(src.dtype());
        ptr = static_cast<std::uint8_t*>(ptr) +
            static_cast<std::size_t>(exec_op_row_offset(op)) * row_bytes;
    } else if (src.shape() != out_spec.shape) {
        throw std::runtime_error(
            "layout plan: Alias shape mismatch for '" + exec_op_output(op) + "'");
    }

    weights.insert(exec_op_output(op), DeviceTensor::view(
        ptr, out_spec.dtype, out_spec.shape), out_spec);
}

void bind_metadata_for_output(
    const ExecOp& op,
    const LayoutPlan& plan,
    WeightStoreBuilder& weights)
{
    const TensorDecl& spec = tensor_spec_for(plan, exec_op_output(op));
    if (spec.quant.format == QuantFormat::None ||
        spec.quant.scale_tensor.empty()) {
        throw std::runtime_error(
            "layout plan: AttachMetadata needs quant TensorDecl for '" +
            exec_op_output(op) + "'");
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
            "layout plan: AttachMetadata missing granularity for '" +
            exec_op_output(op) + "'");
    }
    meta.scale = &weights.get(spec.quant.scale_tensor);
    meta.zero_point = spec.quant.zero_point_tensor.empty()
        ? nullptr
        : &weights.get(spec.quant.zero_point_tensor);
    meta.group_size = spec.quant.group_size;
    meta.channel_axis = spec.quant.channel_axis;
    weights.set_quant_meta(exec_op_output(op), std::move(meta));
}

struct RuntimeQuantResult {
    std::uint64_t bytes_before = 0;
    std::uint64_t bytes_after = 0;
};

int checked_int_dim(std::int64_t value, const std::string& name);

RuntimeQuantResult quantize_runtime_to_output(
    const ExecOp& op,
    const LayoutPlan& plan,
    WeightStoreBuilder& weights,
    NcclComm* tp_comm,
    int tp_size)
{
    const TensorDecl& out_spec = tensor_spec_for(plan, exec_op_output(op));
    const DeviceTensor& src = weights.get(first_input(op));
    if (src.shape().size() != 2 || out_spec.shape.size() != 2 ||
        src.shape() != out_spec.shape) {
        throw std::runtime_error(
            "layout plan: QuantizeRuntime requires matching 2-D source and "
            "output shapes for '" + exec_op_output(op) + "'");
    }
    if (out_spec.quant.scale_tensor.empty()) {
        throw std::runtime_error(
            "layout plan: QuantizeRuntime missing scale tensor for '" +
            exec_op_output(op) + "'");
    }

    const bool is_int8 =
        out_spec.quant.format == QuantFormat::RuntimeInt8 &&
        out_spec.dtype == DType::INT8;
    const bool is_fp8 =
        out_spec.quant.format == QuantFormat::RuntimeFp8E4M3 &&
        out_spec.dtype == DType::FP8_E4M3;
    if (!is_int8 && !is_fp8) {
        throw std::runtime_error(
            "layout plan: QuantizeRuntime has unsupported quant spec for '" +
            exec_op_output(op) + "'");
    }
    if (out_spec.quant.granularity != QuantGranularity::PerChannel) {
        throw std::runtime_error(
            "layout plan: QuantizeRuntime currently expects per-channel "
            "quantization for '" + exec_op_output(op) + "'");
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
            "layout plan: QuantizeRuntime source '" + first_input(op) +
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
                "layout plan: QuantizeRuntime row-parallel tensor '" +
                exec_op_output(op) + "' requires a tensor-parallel communicator");
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
    weights.insert(exec_op_output(op), std::move(q), out_spec);
    weights.insert(
        out_spec.quant.scale_tensor,
        std::move(scale),
        tensor_spec_for(plan, out_spec.quant.scale_tensor));
    return result;
}

const TensorDecl* find_tensor_spec(
    const LayoutPlan& plan,
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
            "layout plan: INT4 Dequantize scale for '" + output_name +
            "' must be bf16/fp16/fp32, got " +
            std::string(dtype_name(scale.dtype())));
    }
    return bf16_scratch.data();
}

DeviceTensor dequantize_awq_tensor(
    const ExecOp& op,
    const LayoutPlan& plan,
    WeightStoreBuilder& weights,
    const TensorDecl& source_spec,
    const TensorDecl& out_spec)
{
    if (exec_op_inputs(op).size() != 3) {
        throw std::runtime_error(
            "layout plan: AWQ Dequantize op for '" + exec_op_output(op) +
            "' requires qweight, qzeros, and scales inputs");
    }
    const DeviceTensor& qweight = weights.get(exec_op_inputs(op)[0]);
    const DeviceTensor& qzeros = weights.get(exec_op_inputs(op)[1]);
    const DeviceTensor& scales = weights.get(exec_op_inputs(op)[2]);
    (void)plan;

    if (qweight.dtype() != DType::INT32 || qweight.shape().size() != 2 ||
        qzeros.dtype() != DType::INT32 || qzeros.shape().size() != 2 ||
        scales.shape().size() != 2 || out_spec.dtype != DType::BF16) {
        throw std::runtime_error(
            "layout plan: AWQ Dequantize input dtype/rank mismatch for '" +
            exec_op_output(op) + "'");
    }
    const std::int64_t size_k = qweight.shape()[0];
    const std::int64_t size_n = qweight.shape()[1] * 8;
    if (size_k <= 0 || size_n <= 0 ||
        source_spec.quant.group_size <= 0 ||
        size_k % source_spec.quant.group_size != 0 ||
        out_spec.shape != std::vector<std::int64_t>{size_n, size_k}) {
        throw std::runtime_error(
            "layout plan: AWQ Dequantize output shape mismatch for '" +
            exec_op_output(op) + "'");
    }
    const std::int64_t groups = size_k / source_spec.quant.group_size;
    if (qzeros.shape() != std::vector<std::int64_t>{groups, size_n / 8} ||
        scales.shape() != std::vector<std::int64_t>{groups, size_n}) {
        throw std::runtime_error(
            "layout plan: AWQ Dequantize qzeros/scales shape mismatch for '" +
            exec_op_output(op) + "'");
    }

    DeviceTensor scale_bf16;
    const void* scale_ptr =
        bf16_scale_ptr_for_int4_dequant(scales, scale_bf16, exec_op_output(op));
    DeviceTensor out = DeviceTensor::allocate(out_spec.dtype, out_spec.shape);
    kernels::launch_awq_dequant_to_bf16(
        qweight.data(), qzeros.data(), scale_ptr, out.data(),
        checked_int_dim(size_k, exec_op_output(op)),
        checked_int_dim(size_n, exec_op_output(op)),
        source_spec.quant.group_size,
        /*stream=*/0);
    return out;
}

DeviceTensor dequantize_gptq_tensor(
    const ExecOp& op,
    const LayoutPlan& plan,
    WeightStoreBuilder& weights,
    const TensorDecl& source_spec,
    const TensorDecl& out_spec)
{
    if (exec_op_inputs(op).size() != 3 && exec_op_inputs(op).size() != 4) {
        throw std::runtime_error(
            "layout plan: GPTQ Dequantize op for '" + exec_op_output(op) +
            "' requires qweight, qzeros, scales, and optional g_idx inputs");
    }
    const DeviceTensor& qweight = weights.get(exec_op_inputs(op)[0]);
    const DeviceTensor& qzeros = weights.get(exec_op_inputs(op)[1]);
    const DeviceTensor& scales = weights.get(exec_op_inputs(op)[2]);
    (void)plan;

    if (qweight.dtype() != DType::INT32 || qweight.shape().size() != 2 ||
        qzeros.dtype() != DType::INT32 || qzeros.shape().size() != 2 ||
        scales.shape().size() != 2 || out_spec.dtype != DType::BF16) {
        throw std::runtime_error(
            "layout plan: GPTQ Dequantize input dtype/rank mismatch for '" +
            exec_op_output(op) + "'");
    }
    const std::int64_t size_k = qweight.shape()[0] * 8;
    const std::int64_t size_n = qweight.shape()[1];
    if (size_k <= 0 || size_n <= 0 || size_n % 8 != 0 ||
        source_spec.quant.group_size <= 0 ||
        size_k % source_spec.quant.group_size != 0 ||
        out_spec.shape != std::vector<std::int64_t>{size_n, size_k}) {
        throw std::runtime_error(
            "layout plan: GPTQ Dequantize output shape mismatch for '" +
            exec_op_output(op) + "'");
    }
    const std::int64_t groups = size_k / source_spec.quant.group_size;
    const bool has_gidx = exec_op_inputs(op).size() == 4;
    if (!has_gidx) {
        if (qzeros.shape() != std::vector<std::int64_t>{groups, size_n / 8} ||
            scales.shape() != std::vector<std::int64_t>{groups, size_n}) {
            throw std::runtime_error(
                "layout plan: GPTQ Dequantize qzeros/scales shape mismatch for '" +
                exec_op_output(op) + "'");
        }
    } else {
        if (qzeros.shape().size() != 2 || scales.shape().size() != 2 ||
            qzeros.shape()[1] != size_n / 8 ||
            scales.shape()[1] != size_n ||
            qzeros.shape()[0] != scales.shape()[0] ||
            qzeros.shape()[0] < groups) {
            throw std::runtime_error(
                "layout plan: GPTQ act-order Dequantize qzeros/scales do not "
                "cover local groups for '" + exec_op_output(op) + "'");
        }
    }

    const void* gidx_ptr = nullptr;
    if (has_gidx) {
        const DeviceTensor& gidx = weights.get(exec_op_inputs(op)[3]);
        if (gidx.dtype() != DType::INT32 ||
            gidx.shape() != std::vector<std::int64_t>{size_k}) {
            throw std::runtime_error(
                "layout plan: GPTQ Dequantize g_idx shape mismatch for '" +
                exec_op_output(op) + "'");
        }
        gidx_ptr = gidx.data();
    }

    DeviceTensor scale_bf16;
    const void* scale_ptr =
        bf16_scale_ptr_for_int4_dequant(scales, scale_bf16, exec_op_output(op));
    DeviceTensor out = DeviceTensor::allocate(out_spec.dtype, out_spec.shape);
    kernels::launch_gptq_dequant_to_bf16(
        qweight.data(), qzeros.data(), scale_ptr, gidx_ptr, out.data(),
        checked_int_dim(size_k, exec_op_output(op)),
        checked_int_dim(size_n, exec_op_output(op)),
        source_spec.quant.group_size,
        /*stream=*/0);
    return out;
}

DeviceTensor dequantize_to_tensor(
    const ExecOp& op,
    const LayoutPlan& plan,
    WeightStoreBuilder& weights,
    const TensorDecl& out_spec)
{
    if (exec_op_inputs(op).size() < 2) {
        throw std::runtime_error(
            "layout plan: Dequantize op for '" + exec_op_output(op) +
            "' requires weight and scale inputs");
    }
    const TensorDecl* source_spec = find_tensor_spec(plan, exec_op_inputs(op)[0]);
    if (source_spec != nullptr &&
        source_spec->quant.format == QuantFormat::AwqInt4) {
        return dequantize_awq_tensor(
            op, plan, weights, *source_spec, out_spec);
    }
    if (source_spec != nullptr &&
        source_spec->quant.format == QuantFormat::GptqInt4) {
        return dequantize_gptq_tensor(
            op, plan, weights, *source_spec, out_spec);
    }

    const DeviceTensor& src = weights.get(exec_op_inputs(op)[0]);
    const DeviceTensor& scale = weights.get(exec_op_inputs(op)[1]);
    if (src.dtype() == DType::UINT8 && scale.dtype() == DType::UINT8 &&
        out_spec.dtype == DType::BF16) {
        if (scale.shape().empty() || src.shape().size() != scale.shape().size() + 1) {
            throw std::runtime_error(
                "layout plan: MXFP4 Dequantize expects packed blocks with one "
                "extra trailing dimension for '" + exec_op_output(op) + "'");
        }
        const std::int64_t blocks_per_row = scale.shape().back();
        if (blocks_per_row <= 0 || src.shape().back() != 16) {
            throw std::runtime_error(
                "layout plan: MXFP4 Dequantize expects 16 packed bytes per "
                "32-value block for '" + exec_op_output(op) + "'");
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
                "layout plan: MXFP4 Dequantize output shape mismatch for '" +
                exec_op_output(op) + "'");
        }
        if (src.numel() != static_cast<std::size_t>(rows * blocks_per_row * 16)) {
            throw std::runtime_error(
                "layout plan: MXFP4 Dequantize packed source shape mismatch for '" +
                exec_op_output(op) + "'");
        }

        DeviceTensor out = DeviceTensor::allocate(out_spec.dtype, out_spec.shape);
        const std::int64_t rows_per_tile = std::max<std::int64_t>(
            1,
            static_cast<std::int64_t>(
                (64ull * 1024ull * 1024ull) /
                std::max<std::uint64_t>(
                    static_cast<std::uint64_t>(in_dim) *
                        dtype_bytes(DType::BF16),
                    1)));
        const auto* src8 = static_cast<const std::uint8_t*>(src.data());
        const auto* scale8 = static_cast<const std::uint8_t*>(scale.data());
        auto* out8 = static_cast<std::uint8_t*>(out.data());
        for (std::int64_t row = 0; row < rows; row += rows_per_tile) {
            const std::int64_t tile_rows = std::min(rows_per_tile, rows - row);
            kernels::launch_dequant_mxfp4_to_bf16(
                src8 + static_cast<std::uint64_t>(row) *
                    static_cast<std::uint64_t>(blocks_per_row) * 16ull,
                scale8 + static_cast<std::uint64_t>(row) *
                    static_cast<std::uint64_t>(blocks_per_row),
                out8 + static_cast<std::uint64_t>(row) *
                    static_cast<std::uint64_t>(in_dim) *
                    dtype_bytes(DType::BF16),
                static_cast<int>(tile_rows),
                static_cast<int>(in_dim),
                /*stream=*/0);
        }
        return out;
    }

    if (src.dtype() != DType::FP8_E4M3 || out_spec.dtype != DType::BF16) {
        throw std::runtime_error(
            "layout plan: Dequantize currently supports FP8_E4M3, MXFP4, "
            "AWQ INT4, and GPTQ INT4 to BF16 for '" +
            exec_op_output(op) + "'");
    }
    if (src.shape() != out_spec.shape || src.shape().empty()) {
        throw std::runtime_error(
            "layout plan: Dequantize source shape does not match output spec for '" +
            exec_op_output(op) + "'");
    }
    if (scale.dtype() != DType::FP32 && scale.dtype() != DType::BF16) {
        throw std::runtime_error(
            "layout plan: Dequantize scale for '" + exec_op_output(op) +
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
        const std::size_t elems_per_tile = std::max<std::size_t>(
            1,
            (64ull * 1024ull * 1024ull) /
                std::max<std::size_t>(dtype_bytes(DType::BF16), 1));
        const auto* src8 = static_cast<const std::uint8_t*>(src.data());
        auto* out8 = static_cast<std::uint8_t*>(out.data());
        for (std::size_t offset = 0; offset < src.numel(); offset += elems_per_tile) {
            const std::size_t tile_elems =
                std::min(elems_per_tile, src.numel() - offset);
            kernels::launch_dequant_fp8_e4m3_to_bf16(
                src8 + offset,
                out8 + offset * dtype_bytes(DType::BF16),
                host_scale, tile_elems, /*stream=*/0);
        }
    } else {
        const auto rows = static_cast<int>(src.shape()[0]);
        const auto cols = static_cast<int>(src.numel() /
                                           static_cast<std::size_t>(rows));
        if (fp32_scale->numel() != static_cast<std::size_t>(rows)) {
            throw std::runtime_error(
                "layout plan: Dequantize per-channel scale length does not "
                "match source rows for '" + exec_op_output(op) + "'");
        }
        const int rows_per_tile = std::max<int>(
            1,
            static_cast<int>(
                (64ull * 1024ull * 1024ull) /
                std::max<std::uint64_t>(
                    static_cast<std::uint64_t>(cols) *
                        dtype_bytes(DType::BF16),
                    1)));
        const auto* src8 = static_cast<const std::uint8_t*>(src.data());
        auto* out8 = static_cast<std::uint8_t*>(out.data());
        const auto* scale_ptr = static_cast<const float*>(fp32_scale->data());
        for (int row = 0; row < rows; row += rows_per_tile) {
            const int tile_rows = std::min(rows_per_tile, rows - row);
            kernels::launch_dequant_fp8_e4m3_to_bf16_per_channel(
                src8 + static_cast<std::size_t>(row) *
                    static_cast<std::size_t>(cols),
                out8 + static_cast<std::size_t>(row) *
                    static_cast<std::size_t>(cols) *
                    dtype_bytes(DType::BF16),
                scale_ptr + row,
                tile_rows, cols, /*stream=*/0);
        }
    }

    return out;
}

std::uint64_t dequantize_to_output(
    const ExecOp& op,
    const LayoutPlan& plan,
    WeightStoreBuilder& weights)
{
    const TensorDecl& out_spec = tensor_spec_for(plan, exec_op_output(op));
    DeviceTensor out = dequantize_to_tensor(op, plan, weights, out_spec);
    const std::uint64_t bytes = out.nbytes();
    weights.insert(exec_op_output(op), std::move(out), out_spec);
    return bytes;
}

RuntimeQuantResult transcode_to_output(
    const ExecOp& op,
    const LayoutPlan& plan,
    WeightStoreBuilder& weights,
    NcclComm* tp_comm,
    int tp_size)
{
    const TensorDecl& out_spec = tensor_spec_for(plan, exec_op_output(op));
    if (out_spec.quant.format == QuantFormat::None) {
        throw std::runtime_error(
            "layout plan: Transcode output missing quant spec for '" +
            exec_op_output(op) + "'");
    }
    TensorDecl decoded_spec = out_spec;
    decoded_spec.name = exec_op_output(op) + ".__transcode_bf16";
    decoded_spec.dtype = DType::BF16;
    decoded_spec.layout = TensorLayoutKind::Dense;
    decoded_spec.ownership = TensorOwnershipKind::Temporary;
    decoded_spec.quant = {};

    DeviceTensor decoded = dequantize_to_tensor(op, plan, weights, decoded_spec);
    const std::uint64_t decoded_bytes = decoded.nbytes();
    const std::string decoded_name = decoded_spec.name;
    weights.insert(decoded_name, std::move(decoded), decoded_spec);

    ExecOp encode_op = op;
    encode_op.inputs = {decoded_name};
    RuntimeQuantResult result;
    try {
        result = quantize_runtime_to_output(
            encode_op, plan, weights, tp_comm, tp_size);
    } catch (...) {
        weights.erase(decoded_name);
        throw;
    }
    weights.erase(decoded_name);
    result.bytes_before = decoded_bytes;
    return result;
}

std::uint64_t deinterleave_to_outputs(
    const ExecOp& op,
    const LayoutPlan& plan,
    WeightStoreBuilder& weights,
    int tp_rank,
    int tp_size)
{
    const TensorDecl& first_spec = tensor_spec_for(plan, exec_op_output(op));
    const TensorDecl& second_spec = tensor_spec_for(plan, exec_op_secondary_output(op));
    const DeviceTensor& src = weights.get(first_input(op));
    if (src.dtype() != DType::BF16 ||
        first_spec.dtype != DType::BF16 ||
        second_spec.dtype != DType::BF16) {
        throw std::runtime_error(
            "layout plan: Deinterleave currently supports bf16 tensors for '" +
            exec_op_output(op) + "'");
    }
    if (first_spec.shape != second_spec.shape) {
        throw std::runtime_error(
            "layout plan: Deinterleave output shapes differ for '" +
            exec_op_output(op) + "'");
    }
    if (src.shape().size() != first_spec.shape.size() ||
        (src.shape().size() != 2 && src.shape().size() != 3)) {
        throw std::runtime_error(
            "layout plan: Deinterleave expects rank-2 or rank-3 source for '" +
            exec_op_output(op) + "'");
    }
    const auto& src_shape = src.shape();
    const auto& out_shape = first_spec.shape;
    const std::int64_t E = src_shape[0];
    const std::int64_t two_I_full = src_shape[1];
    if (two_I_full % 2 != 0 || E != out_shape[0]) {
        throw std::runtime_error(
            "layout plan: Deinterleave source shape mismatch for '" +
            exec_op_output(op) + "'");
    }
    const std::int64_t I_full = two_I_full / 2;
    const std::int64_t I_local = out_shape[1];
    if (I_local <= 0 || I_full % I_local != 0) {
        throw std::runtime_error(
            "layout plan: Deinterleave local size mismatch for '" +
            exec_op_output(op) + "'");
    }
    std::int64_t start = 0;
    if (tp_size > 1 && exec_op_shard_axis(op) == 1) {
        if (I_full % tp_size != 0 || I_local != I_full / tp_size) {
            throw std::runtime_error(
                "layout plan: Deinterleave cannot shard intermediate axis for '" +
                exec_op_output(op) + "'");
        }
        start = static_cast<std::int64_t>(tp_rank) * I_local;
    } else if (I_local != I_full) {
        throw std::runtime_error(
            "layout plan: Deinterleave output is sharded but op has no "
            "matching shard_axis for '" + exec_op_output(op) + "'");
    }

    DeviceTensor first = DeviceTensor::allocate(first_spec.dtype, first_spec.shape);
    DeviceTensor second = DeviceTensor::allocate(second_spec.dtype, second_spec.shape);
    const std::size_t elem = dtype_bytes(DType::BF16);
    if (src_shape.size() == 3) {
        const std::int64_t H = src_shape[2];
        if (out_shape.size() != 3 || out_shape[2] != H) {
            throw std::runtime_error(
                "layout plan: Deinterleave matrix output mismatch for '" +
                exec_op_output(op) + "'");
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
                "layout plan: Deinterleave vector output mismatch for '" +
                exec_op_output(op) + "'");
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
    weights.insert(exec_op_output(op), std::move(first), first_spec);
    weights.insert(exec_op_secondary_output(op), std::move(second), second_spec);
    return bytes;
}

std::uint64_t stack_groups_to_outputs(
    const ExecOp& op,
    const LayoutPlan& plan,
    WeightStoreBuilder& weights)
{
    const TensorDecl& gate_up_spec = tensor_spec_for(plan, exec_op_output(op));
    const TensorDecl& down_spec =
        tensor_spec_for(plan, exec_op_secondary_output(op));
    if (exec_op_inputs(op).empty() ||
        exec_op_inputs(op).size() % 3 != 0 ||
        !exec_op_sources(op).empty()) {
        throw std::runtime_error(
            "layout plan: StackGroups expects input tensor triples for '" +
            exec_op_output(op) + "'");
    }
    const std::int64_t E =
        static_cast<std::int64_t>(exec_op_inputs(op).size() / 3);
    if (gate_up_spec.dtype != DType::BF16 ||
        down_spec.dtype != DType::BF16 ||
        gate_up_spec.shape.size() != 3 ||
        down_spec.shape.size() != 3 ||
        gate_up_spec.shape[0] != E ||
        down_spec.shape[0] != E) {
        throw std::runtime_error(
            "layout plan: StackGroups output spec mismatch for '" +
            exec_op_output(op) + "'");
    }

    const std::int64_t I = gate_up_spec.shape[1] / 2;
    const std::int64_t H = gate_up_spec.shape[2];
    const std::int64_t I_down = down_spec.shape[2];
    if (gate_up_spec.shape !=
            std::vector<std::int64_t>{E, 2 * I, H} ||
        down_spec.shape !=
            std::vector<std::int64_t>{E, H, I_down}) {
        throw std::runtime_error(
            "layout plan: StackGroups output shapes do not match inputs for '" +
            exec_op_output(op) + "'");
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

        const DeviceTensor& gate =
            weights.get(exec_op_inputs(op)[static_cast<std::size_t>(e) * 3]);
        const DeviceTensor& up =
            weights.get(exec_op_inputs(op)[static_cast<std::size_t>(e) * 3 + 1]);
        const DeviceTensor& down_src =
            weights.get(exec_op_inputs(op)[static_cast<std::size_t>(e) * 3 + 2]);
        if (gate.dtype() != DType::BF16 || up.dtype() != DType::BF16 ||
            down_src.dtype() != DType::BF16 ||
            gate.shape() != std::vector<std::int64_t>{I, H} ||
            up.shape() != std::vector<std::int64_t>{I, H} ||
            down_src.shape() != std::vector<std::int64_t>{H, I_down}) {
            throw std::runtime_error(
                "layout plan: StackGroups expert input mismatch for '" +
                exec_op_output(op) + "'");
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

    const std::uint64_t bytes = gate_up.nbytes() + down.nbytes();
    weights.insert(exec_op_output(op), std::move(gate_up), gate_up_spec);
    weights.insert(exec_op_secondary_output(op), std::move(down), down_spec);
    return bytes;
}

int checked_int_dim(std::int64_t value, const std::string& name) {
    if (value <= 0 ||
        value > static_cast<std::int64_t>(std::numeric_limits<int>::max())) {
        throw std::runtime_error(
            "layout plan: dimension out of range for '" + name + "'");
    }
    return static_cast<int>(value);
}

std::uint64_t repack_layout_to_outputs(
    const ExecOp& op,
    const LayoutPlan& plan,
    WeightStoreBuilder& weights)
{
#ifndef PIE_CUDA_HAS_MARLIN
    throw std::runtime_error(
        "layout plan: RepackLayout for '" + exec_op_output(op) +
        "' requires Marlin, but this build was configured without "
        "PIE_CUDA_BUILD_MARLIN=ON.");
#else
    const TensorDecl& out_spec = tensor_spec_for(plan, exec_op_output(op));
    const TensorDecl& scale_spec =
        tensor_spec_for(plan, exec_op_secondary_output(op));
    const bool is_gptq = out_spec.quant.format == QuantFormat::GptqInt4;
    const bool is_awq = out_spec.quant.format == QuantFormat::AwqInt4;
    if ((!is_gptq && !is_awq) ||
        out_spec.quant.granularity != QuantGranularity::PerGroup ||
        out_spec.quant.scale_tensor != exec_op_secondary_output(op)) {
        throw std::runtime_error(
            "layout plan: RepackLayout supports GPTQ/AWQ int4 with "
            "per-group scale metadata for '" + exec_op_output(op) + "'");
    }
    if (out_spec.dtype != DType::INT4_PACKED ||
        out_spec.layout != TensorLayoutKind::QuantPacked) {
        throw std::runtime_error(
            "layout plan: RepackLayout output spec must be QuantPacked int4 for '" +
            exec_op_output(op) + "'");
    }
    if (scale_spec.dtype != DType::BF16 ||
        scale_spec.layout != TensorLayoutKind::Dense) {
        throw std::runtime_error(
            "layout plan: RepackLayout scale output must be dense bf16 for '" +
            exec_op_output(op) + "'");
    }

    const DeviceTensor& qweight = weights.get(exec_op_inputs(op).at(0));
    if (qweight.dtype() != DType::INT32 || qweight.shape().size() != 2) {
        throw std::runtime_error(
            "layout plan: RepackLayout source must be 2-D int32 for '" +
            exec_op_output(op) + "'");
    }

    if (is_awq) {
        if (exec_op_inputs(op).size() != 3) {
            throw std::runtime_error(
                "layout plan: AWQ RepackLayout requires qweight, qzeros, "
                "and scales inputs for '" + exec_op_output(op) + "'");
        }
        if (out_spec.quant.zero_point_tensor.empty()) {
            throw std::runtime_error(
                "layout plan: AWQ RepackLayout output is missing zero-point "
                "metadata for '" + exec_op_output(op) + "'");
        }
        const TensorDecl& zero_spec =
            tensor_spec_for(plan, out_spec.quant.zero_point_tensor);
        const DeviceTensor& qzeros = weights.get(exec_op_inputs(op).at(1));
        const DeviceTensor& scales = weights.get(exec_op_inputs(op).at(2));
        if (qzeros.dtype() != DType::INT32 || qzeros.shape().size() != 2 ||
            scales.shape().size() != 2) {
            throw std::runtime_error(
                "layout plan: AWQ RepackLayout qzeros/scales mismatch for '" +
                exec_op_output(op) + "'");
        }

        const int size_k =
            checked_int_dim(qweight.shape()[0], exec_op_output(op));
        const int size_n =
            checked_int_dim(qweight.shape()[1] * 8, exec_op_output(op));
        const int groups =
            checked_int_dim(scales.shape()[0], exec_op_secondary_output(op));
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
                "layout plan: AWQ RepackLayout shape mismatch for '" +
                exec_op_output(op) + "'");
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
                "layout plan: AWQ RepackLayout scales must be bf16/fp16/fp32 "
                "for '" + exec_op_output(op) + "'");
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
        weights.insert(exec_op_output(op), std::move(packed), out_spec);
        weights.insert(exec_op_secondary_output(op), std::move(bf16_scales), scale_spec);
        weights.insert(
            out_spec.quant.zero_point_tensor,
            std::move(qzeros_marlin),
            zero_spec);
        return bytes;
    }

    const DeviceTensor& scales = weights.get(exec_op_inputs(op).at(1));
    if (scales.dtype() != DType::FP16 || scales.shape().size() != 2) {
        throw std::runtime_error(
            "layout plan: RepackLayout GPTQ scales must be 2-D fp16 for '" +
            exec_op_output(op) + "'");
    }

    const int size_k =
        checked_int_dim(qweight.shape()[0] * 8, exec_op_output(op));
    const int size_n =
        checked_int_dim(qweight.shape()[1], exec_op_output(op));
    const int groups =
        checked_int_dim(scales.shape()[0], exec_op_secondary_output(op));
    if (scales.shape()[1] != size_n) {
        throw std::runtime_error(
            "layout plan: RepackLayout scale columns do not match qweight for '" +
            exec_op_output(op) + "'");
    }
    if (size_k % 16 != 0 ||
        out_spec.shape != std::vector<std::int64_t>{
            static_cast<std::int64_t>(size_k / 16),
            static_cast<std::int64_t>(size_n) * 8}) {
        throw std::runtime_error(
            "layout plan: RepackLayout packed output shape mismatch for '" +
            exec_op_output(op) + "'");
    }
    if (scale_spec.shape != scales.shape()) {
        throw std::runtime_error(
            "layout plan: RepackLayout scale output shape mismatch for '" +
            exec_op_output(op) + "'");
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
    weights.insert(exec_op_output(op), std::move(packed), out_spec);
    weights.insert(exec_op_secondary_output(op), std::move(bf16_scales), scale_spec);
    return bytes;
#endif
}

void validate_materialized_tensors(
    const LayoutPlan& plan,
    const WeightStore& weights)
{
    for (const auto& [name, spec] : plan.tensors) {
        if (spec.ownership == TensorOwnershipKind::Temporary) continue;
        if (weights.find(name) == weights.end()) {
            throw std::runtime_error(
                "layout plan: load_executor did not produce tensor '" + name + "'");
        }
        const DeviceTensor& tensor = weights.get(name);
        if (tensor.dtype() != spec.dtype) {
            throw std::runtime_error(
                "layout plan: tensor '" + name + "' dtype mismatch: planned " +
                std::string(dtype_name(spec.dtype)) + ", got " +
                std::string(dtype_name(tensor.dtype())));
        }
        if (tensor.shape() != spec.shape) {
            throw std::runtime_error(
                "layout plan: tensor '" + name +
                "' shape mismatch after materialization");
        }
        if (spec.ownership == TensorOwnershipKind::BorrowedView &&
            weights.find(spec.backing_tensor) == weights.end()) {
            throw std::runtime_error(
                "layout plan: view '" + name + "' backing tensor '" +
                spec.backing_tensor + "' was not materialized");
        }
    }
}

void register_axis_concat_views(
    const ExecOp& op,
    const std::vector<ExtentWrite>& writes,
    const LayoutPlan& plan,
    WeightStoreBuilder& weights)
{
    if (writes.size() != exec_op_sources(op).size()) {
        throw std::runtime_error(
            "storage program: AxisConcat extent-write count mismatch for '" +
            exec_op_output(op) + "'");
    }
    const TensorDecl& packed_spec = tensor_spec_for(plan, exec_op_output(op));
    if (packed_spec.shape.size() != 2) {
        throw std::runtime_error(
            "layout plan: AxisConcat output must be 2-D: " + exec_op_output(op));
    }
    const std::int64_t cols = packed_spec.shape[1];
    std::int64_t row_offset = 0;
    for (std::size_t i = 0; i < exec_op_sources(op).size(); ++i) {
        const auto& write = writes[i];
        if (write.dst_shape.size() != 2 || write.dst_shape[1] != cols) {
            throw std::runtime_error(
                "storage program: AxisConcat view shape mismatch for '" +
                exec_op_sources(op)[i].view_name + "'");
        }
        insert_row_view(
            weights,
            tensor_spec_for(plan, exec_op_sources(op)[i].view_name),
            exec_op_output(op),
            exec_op_sources(op)[i].view_name,
            row_offset,
            write.dst_shape[0],
            cols);
        row_offset += write.dst_shape[0];
    }
}

std::vector<ExtentWrite> storage_writes_for_output(
    const StorageProgram& storage_program,
    const std::string& output_name)
{
    std::vector<ExtentWrite> writes;
    for (const auto& write : storage_program.extent_writes) {
        if (write.output_name == output_name) {
            writes.push_back(write);
        }
    }
    return writes;
}

ExecOp exec_op_for_instruction(
    const StorageProgram& storage_program,
    const StorageInstr& instr)
{
    ExecOp op;
    op.inputs = instr.inputs;
    if (!instr.outputs.empty()) {
        op.output_name = instr.outputs[0];
    }
    if (instr.outputs.size() > 1) {
        op.secondary_output_name = instr.outputs[1];
    }

    if (instr.kind == StorageInstrKind::TileMap) {
        if (instr.tile_map_index >= storage_program.tile_maps.size()) {
            throw std::runtime_error(
                "storage program: TileMap instruction index out of range");
        }
        const TileMap& tile = storage_program.tile_maps[instr.tile_map_index];
        op.inputs = tile.inputs;
        op.output_name = tile.output_name;
        op.secondary_output_name =
            instr.outputs.size() > 1 ? instr.outputs[1] : std::string{};
    }

    if (const auto* slice =
            std::get_if<StorageSlicePayload>(&instr.payload)) {
        op.slice_axis = slice->axis;
        op.slice_start = slice->start;
        op.slice_length = slice->length;
        op.shard_axis = slice->shard_axis;
    } else if (const auto* view =
                   std::get_if<StorageViewPayload>(&instr.payload)) {
        op.row_offset = view->start;
        op.rows = view->length;
        if (view->axis >= 0) {
            op.slice_axis = view->axis;
        }
    } else if (const auto* axis =
                   std::get_if<StorageAxisPayload>(&instr.payload)) {
        op.shard_axis = axis->shard_axis;
    }

    if (instr.kind == StorageInstrKind::CreateView &&
        instr.outputs.size() > 1) {
        if (instr.inputs.empty()) {
            throw std::runtime_error(
                "storage program: CreateView group has no backing input");
        }
        op.output_name = instr.inputs.front();
        op.secondary_output_name.clear();
        op.sources.reserve(instr.outputs.size());
        for (const auto& output : instr.outputs) {
            op.sources.push_back(TensorSourceRef{
                .raw_name = {},
                .view_name = output,
            });
        }
    }

    if (op.output_name.empty() &&
        instr.kind != StorageInstrKind::Release) {
        throw std::runtime_error(
            "storage program: instruction has no output tensor");
    }
    return op;
}

RuntimeQuantResult execute_tile_map_instruction(
    const LayoutPlan& plan,
    const StorageProgram& storage_program,
    const StorageInstr& instr,
    CheckpointByteSource& byte_source,
    WeightStoreBuilder& weights,
    NcclComm* tp_comm,
    int tp_size)
{
    if (instr.tile_map_index >= storage_program.tile_maps.size()) {
        throw std::runtime_error(
            "storage program: TileMap instruction index out of range");
    }
    const TileMap& tile = storage_program.tile_maps[instr.tile_map_index];
    if (!instr.extent_write_indices.empty()) {
        RuntimeQuantResult result;
        result.bytes_after = materialize_tile_raw_cast(
            tile,
            extent_writes_for_instruction(storage_program, instr),
            plan,
            byte_source,
            weights,
            tile.tile_bytes);
        return result;
    }

    RuntimeQuantResult result;
    switch (tile.kind) {
    case TileMapKind::Cast: {
        if (tile.inputs.size() != 1) {
            throw std::runtime_error(
                "storage program: TileMap Cast needs one input for '" +
                tile.output_name + "'");
        }
        const TensorDecl& out_spec = tensor_spec_for(plan, tile.output_name);
        cast_tensor_to_output(weights.get(tile.inputs.front()), weights, out_spec);
        result.bytes_after = weights.get(tile.output_name).nbytes();
        return result;
    }
    case TileMapKind::Decode:
        result.bytes_after = dequantize_to_output(
            exec_op_for_instruction(storage_program, instr), plan, weights);
        return result;
    case TileMapKind::Encode:
        return quantize_runtime_to_output(
            exec_op_for_instruction(storage_program, instr),
            plan, weights, tp_comm, tp_size);
    case TileMapKind::Transcode:
        return transcode_to_output(
            exec_op_for_instruction(storage_program, instr),
            plan, weights, tp_comm, tp_size);
    case TileMapKind::Reblock:
    case TileMapKind::Reorder:
        throw std::runtime_error(
            "storage program: TileMap " +
            std::string(tile_map_kind_name(tile.kind)) +
            " execution is not implemented for '" + tile.output_name + "'");
    }
    throw std::runtime_error("storage program: unknown TileMap kind");
}

RuntimeQuantResult execute_transform_instruction(
    const LayoutPlan& plan,
    const StorageProgram& storage_program,
    const StorageInstr& instr,
    WeightStoreBuilder& weights,
    NcclComm* tp_comm,
    int tp_rank,
    int tp_size)
{
    const ExecOp op = exec_op_for_instruction(storage_program, instr);
    RuntimeQuantResult result;
    switch (instr.transform_kind) {
    case StorageTransformKind::Slice:
        result.bytes_after = (exec_op_slice_axis(op) >= 0)
            ? slice_axis_to_output(op, plan, weights, tp_rank, tp_size)
            : slice_rows_to_output(op, plan, weights);
        return result;
    case StorageTransformKind::Concat:
        result.bytes_after = concat_rows_to_output(op, plan, weights);
        return result;
    case StorageTransformKind::Quantize:
        return quantize_runtime_to_output(op, plan, weights, tp_comm, tp_size);
    case StorageTransformKind::Materialize: {
        const TensorDecl& out_spec =
            tensor_spec_for(plan, exec_op_output(op));
        result.bytes_after = copy_tensor_to_output(
            weights.get(first_input(op)), weights, out_spec);
        return result;
    }
    case StorageTransformKind::Decode:
        result.bytes_after = dequantize_to_output(op, plan, weights);
        return result;
    case StorageTransformKind::Deinterleave:
        result.bytes_after =
            deinterleave_to_outputs(op, plan, weights, tp_rank, tp_size);
        return result;
    case StorageTransformKind::Repack:
        result.bytes_after = repack_layout_to_outputs(op, plan, weights);
        return result;
    case StorageTransformKind::Stack:
        result.bytes_after = stack_groups_to_outputs(op, plan, weights);
        return result;
    case StorageTransformKind::None:
        throw std::runtime_error(
            "storage program: Transform instruction has no transform kind");
    }
    throw std::runtime_error("storage program: unknown transform kind");
}

void execute_create_view_instruction(
    const LayoutPlan& plan,
    const StorageProgram& storage_program,
    const StorageInstr& instr,
    WeightStoreBuilder& weights)
{
    const ExecOp op = exec_op_for_instruction(storage_program, instr);
    if (instr.outputs.size() > 1) {
        register_axis_concat_views(
            op, storage_writes_for_output(storage_program, op.output_name),
            plan, weights);
        return;
    }

    if (const auto* view =
            std::get_if<StorageViewPayload>(&instr.payload)) {
        if (instr.inputs.size() != 1 || instr.outputs.size() != 1) {
            throw std::runtime_error(
                "storage program: invalid CreateView payload shape");
        }
        const TensorDecl& view_spec =
            tensor_spec_for(plan, instr.outputs.front());
        const std::string& backing_name = view_spec.backing_tensor.empty()
            ? instr.inputs.front()
            : view_spec.backing_tensor;
        if (view->axis == 0 && view_spec.shape.size() == 2) {
            insert_row_view(
                weights, view_spec, backing_name, view_spec.name,
                view->start, view->length, view_spec.shape[1]);
            return;
        }
        if (view->axis < 0) {
            const DeviceTensor& backing = weights.get(backing_name);
            weights.insert(
                view_spec.name,
                DeviceTensor::view(
                    const_cast<void*>(backing.data()),
                    backing.dtype(),
                    view_spec.shape),
                view_spec);
            return;
        }
    }

    view_or_alias_to_output(op, plan, weights);
}

void execute_release_instruction(
    const StorageInstr& instr,
    WeightStoreBuilder& weights)
{
    for (const auto& input : instr.inputs) {
        weights.erase(input);
    }
}

void execute_attach_instruction(
    const LayoutPlan& plan,
    const StorageInstr& instr,
    WeightStoreBuilder& weights)
{
    bind_metadata_for_output(
        exec_op_for_instruction(StorageProgram{}, instr), plan, weights);
}

}  // namespace

LoadExecutor::LoadExecutor(
    SafetensorsCheckpointSource& loader,
    WeightStore& weights,
    int tp_rank,
    int tp_size,
    NcclComm* tp_comm,
    CheckpointByteSource* byte_source,
    const StorageProgram* storage_program) noexcept
    : loader_(loader),
      weights_(weights),
      builder_(weights),
      tp_rank_(tp_rank),
      tp_size_(tp_size),
      tp_comm_(tp_comm),
      byte_source_(byte_source),
      storage_program_(storage_program)
{}

LoadExecutionStats LoadExecutor::run(const LayoutPlan& plan)
{
    if (storage_program_ == nullptr) {
        throw std::runtime_error(
            "load executor: compiled StorageProgram is required");
    }
    validate_layout_plan(plan);

    LoadExecutionStats result;
    result.axis_concat_groups = plan.axis_concat_groups;
    result.planned_tensor_count = plan.tensors.size();
    if (storage_program_ != nullptr) {
        result.planned_storage_peak_bytes =
            storage_program_->memory.estimated_peak_bytes;
        result.planned_storage_temp_bytes =
            storage_program_->memory.max_temporary_bytes;
    }
    CudaLoadMemoryTelemetry cuda_memory;
    cuda_memory.sample();
    ScopedDeviceTensorMemoryCallback memory_callback(cuda_memory);
    MmapByteSource mmap_byte_source(loader_);
    CheckpointByteSource& byte_source =
        byte_source_ != nullptr ? *byte_source_ : mmap_byte_source;
    auto write_executor = make_storage_write_executor(byte_source);

    for (std::size_t step_index = 0;
         step_index < storage_program_->schedule.size();
         ++step_index) {
        const StorageInstr& instr = storage_program_->schedule[step_index];
        if (instr.kind == StorageInstrKind::ExtentWrite) {
            std::size_t run_end = step_index + 1;
            while (run_end < storage_program_->schedule.size() &&
                   storage_program_->schedule[run_end].kind ==
                       StorageInstrKind::ExtentWrite) {
                ++run_end;
            }
            result.loaded_bytes += materialize_extent_write_instruction_run(
                plan, *storage_program_, step_index, run_end,
                *write_executor, builder_);
            step_index = run_end - 1;
            cuda_memory.sample();
            continue;
        }

        switch (instr.kind) {
        case StorageInstrKind::Allocate:
            break;
        case StorageInstrKind::ExtentWrite:
            break;
        case StorageInstrKind::TileMap:
            {
                const RuntimeQuantResult qr = execute_tile_map_instruction(
                    plan, *storage_program_, instr, byte_source, builder_,
                    tp_comm_, tp_size_);
                result.loaded_bytes += qr.bytes_after;
                if (qr.bytes_before != 0 || qr.bytes_after != 0) {
                    const auto& tile =
                        storage_program_->tile_maps[instr.tile_map_index];
                    if (tile.kind == TileMapKind::Encode) {
                        result.runtime_quantized_weights += 1;
                        result.runtime_quant_bytes_before += qr.bytes_before;
                        result.runtime_quant_bytes_after += qr.bytes_after;
                    }
                }
            }
            break;
        case StorageInstrKind::Transform: {
            const RuntimeQuantResult qr = execute_transform_instruction(
                plan, *storage_program_, instr, builder_,
                tp_comm_, tp_rank_, tp_size_);
            result.loaded_bytes += qr.bytes_after;
            if (qr.bytes_before != 0 || qr.bytes_after != 0) {
                if (instr.transform_kind == StorageTransformKind::Quantize) {
                    result.runtime_quantized_weights += 1;
                    result.runtime_quant_bytes_before += qr.bytes_before;
                    result.runtime_quant_bytes_after += qr.bytes_after;
                }
            }
            break;
        }
        case StorageInstrKind::Attach:
            execute_attach_instruction(plan, instr, builder_);
            break;
        case StorageInstrKind::CreateView:
            execute_create_view_instruction(
                plan, *storage_program_, instr, builder_);
            break;
        case StorageInstrKind::Release:
            execute_release_instruction(instr, builder_);
            break;
        case StorageInstrKind::Finalize:
            break;
        }
        cuda_memory.sample();
    }

    validate_materialized_tensors(plan, weights_);
    builder_.finalize();
    cuda_memory.sample();
    result.cuda_total_bytes = cuda_memory.total;
    result.cuda_free_before_bytes = cuda_memory.free_before;
    result.cuda_min_free_bytes = cuda_memory.min_free;
    result.cuda_free_after_bytes = cuda_memory.free_after;
    result.cuda_memory_samples = cuda_memory.samples;
    if (cuda_memory.free_before >= cuda_memory.min_free) {
        result.cuda_actual_peak_delta_bytes =
            cuda_memory.free_before - cuda_memory.min_free;
    }

    return result;
}

LoadExecutionStats execute_layout_plan(
    const LayoutPlan& plan,
    SafetensorsCheckpointSource& loader,
    WeightStore& weights,
    int tp_rank,
    int tp_size,
    NcclComm* tp_comm,
    CheckpointByteSource* byte_source,
    const StorageProgram* storage_program)
{
    return LoadExecutor(
        loader, weights, tp_rank, tp_size, tp_comm,
        byte_source, storage_program).run(plan);
}

}  // namespace pie_cuda_driver
