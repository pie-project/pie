#include <cuda_runtime.h>

#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

#include <nlohmann/json.hpp>
#include <unistd.h>

#include "cuda_check.hpp"
#include "loader/layout_plan.hpp"
#include "loader/load_executor.hpp"
#include "loader/storage_compiler.hpp"
#include "loader/storage_program.hpp"
#include "loader/safetensors.hpp"
#include "model/weight_store.hpp"
#include "tensor.hpp"

using namespace pie_cuda_driver;

namespace {

#define CHECK_TRUE(x) do { \
    if (!(x)) throw std::runtime_error(std::string("check failed: ") + #x); \
} while (0)

#define CHECK_EQ(a, b) do { \
    const auto _a = (a); \
    const auto _b = (b); \
    if (!(_a == _b)) { \
        throw std::runtime_error( \
            std::string("check failed: ") + #a + " == " + #b); \
    } \
} while (0)

struct FixtureTensor {
    std::string dtype;
    std::vector<std::int64_t> shape;
    std::vector<std::uint8_t> bytes;
};

template <typename T>
std::vector<std::uint8_t> bytes_of(const std::vector<T>& values)
{
    std::vector<std::uint8_t> bytes(values.size() * sizeof(T));
    if (!values.empty()) {
        std::memcpy(bytes.data(), values.data(), bytes.size());
    }
    return bytes;
}

std::filesystem::path make_temp_snapshot(const std::string& name)
{
    auto dir = std::filesystem::temp_directory_path() /
        ("pie-loader-golden-" + name + "-" + std::to_string(::getpid()));
    std::filesystem::remove_all(dir);
    std::filesystem::create_directories(dir);
    return dir;
}

void write_safetensors(
    const std::filesystem::path& dir,
    const std::vector<std::pair<std::string, FixtureTensor>>& tensors)
{
    nlohmann::json header = nlohmann::json::object();
    std::uint64_t offset = 0;
    for (const auto& [name, tensor] : tensors) {
        const std::uint64_t end = offset + tensor.bytes.size();
        header[name] = {
            {"dtype", tensor.dtype},
            {"shape", tensor.shape},
            {"data_offsets", {offset, end}},
        };
        offset = end;
    }

    const std::string header_bytes = header.dump();
    const std::uint64_t header_size = header_bytes.size();
    std::ofstream out(dir / "model.safetensors", std::ios::binary);
    out.write(reinterpret_cast<const char*>(&header_size), sizeof(header_size));
    out.write(header_bytes.data(), static_cast<std::streamsize>(header_bytes.size()));
    for (const auto& [_, tensor] : tensors) {
        out.write(
            reinterpret_cast<const char*>(tensor.bytes.data()),
            static_cast<std::streamsize>(tensor.bytes.size()));
    }
    if (!out) {
        throw std::runtime_error("failed to write safetensors fixture");
    }
}

void register_spec(
    LayoutPlan& plan,
    std::string name,
    DType dtype,
    std::vector<std::int64_t> shape,
    TensorLayoutKind layout = TensorLayoutKind::Dense,
    TensorOwnershipKind ownership = TensorOwnershipKind::Owned,
    TensorParallelKind parallel = TensorParallelKind::Replicated,
    std::string backing = {},
    QuantSpec quant = {})
{
    TensorDecl spec;
    spec.name = std::move(name);
    spec.dtype = dtype;
    spec.shape = std::move(shape);
    spec.layout = layout;
    spec.ownership = ownership;
    spec.parallel = parallel;
    spec.backing_tensor = std::move(backing);
    spec.quant = std::move(quant);
    plan.tensors.emplace(spec.name, std::move(spec));
}

template <typename T>
std::vector<T> read_tensor(const DeviceTensor& tensor)
{
    std::vector<T> host(tensor.nbytes() / sizeof(T));
    CUDA_CHECK(cudaMemcpy(
        host.data(), tensor.data(), tensor.nbytes(), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());
    return host;
}

LoadExecutionStats run_plan(
    const std::filesystem::path& dir,
    LayoutPlan& plan,
    WeightStore& weights,
    int tp_rank = 0,
    int tp_size = 1)
{
    auto loader = SafetensorsCheckpointSource::open(dir);
    auto storage = compile_storage_program(plan, loader, tp_rank, tp_size);
    LoadExecutor load_executor(
        loader, weights, tp_rank, tp_size,
        /*tp_comm=*/nullptr,
        /*byte_source=*/nullptr,
        &storage);
    return load_executor.run(plan);
}

void test_axis_concat_exact()
{
    const auto dir = make_temp_snapshot("axis");
    write_safetensors(dir, {
        {"q", {"BF16", {2, 2}, bytes_of<std::uint16_t>({1, 2, 3, 4})}},
        {"k", {"BF16", {1, 2}, bytes_of<std::uint16_t>({5, 6})}},
        {"v", {"BF16", {1, 2}, bytes_of<std::uint16_t>({7, 8})}},
    });

    LayoutPlan plan;
    plan.ops.push_back(make_axis_concat_op(
        "packed", -1, {{"q", "q_view"}, {"k", "k_view"}, {"v", "v_view"}}));
    register_spec(
        plan, "packed", DType::BF16, {4, 2},
        TensorLayoutKind::AxisConcatenated);
    register_spec(
        plan, "q_view", DType::BF16, {2, 2}, TensorLayoutKind::View,
        TensorOwnershipKind::BorrowedView, TensorParallelKind::Column,
        "packed");
    register_spec(
        plan, "k_view", DType::BF16, {1, 2}, TensorLayoutKind::View,
        TensorOwnershipKind::BorrowedView, TensorParallelKind::Column,
        "packed");
    register_spec(
        plan, "v_view", DType::BF16, {1, 2}, TensorLayoutKind::View,
        TensorOwnershipKind::BorrowedView, TensorParallelKind::Column,
        "packed");

    WeightStore weights;
    run_plan(dir, plan, weights);
    CHECK_EQ(read_tensor<std::uint16_t>(weights.get("packed")),
             std::vector<std::uint16_t>({1, 2, 3, 4, 5, 6, 7, 8}));
    CHECK_EQ(read_tensor<std::uint16_t>(weights.get("k_view")),
             std::vector<std::uint16_t>({5, 6}));
    std::filesystem::remove_all(dir);
}

void test_grouped_slice_exact()
{
    const auto dir = make_temp_snapshot("grouped");
    std::vector<std::uint16_t> values(16);
    for (std::uint16_t i = 0; i < values.size(); ++i) values[i] = i + 1;
    write_safetensors(dir, {
        {"gate_up", {"BF16", {2, 4, 2}, bytes_of(values)}},
        {"down", {"BF16", {2, 2, 4}, bytes_of(values)}},
    });

    LayoutPlan plan;
    plan.ops.push_back(make_raw_load_op(
        LayoutOpKind::GroupedSliceConcat, "gate_up_local", "gate_up"));
    plan.ops.push_back(make_raw_load_op(
        LayoutOpKind::GroupedSlice, "down_local", "down"));
    register_spec(
        plan, "gate_up_local", DType::BF16, {2, 2, 2},
        TensorLayoutKind::Grouped, TensorOwnershipKind::Owned,
        TensorParallelKind::Expert);
    register_spec(
        plan, "down_local", DType::BF16, {2, 2, 2},
        TensorLayoutKind::Grouped, TensorOwnershipKind::Owned,
        TensorParallelKind::Expert);

    WeightStore weights;
    run_plan(dir, plan, weights, /*tp_rank=*/1, /*tp_size=*/2);
    CHECK_EQ(read_tensor<std::uint16_t>(weights.get("gate_up_local")),
             std::vector<std::uint16_t>({3, 4, 7, 8, 11, 12, 15, 16}));
    CHECK_EQ(read_tensor<std::uint16_t>(weights.get("down_local")),
             std::vector<std::uint16_t>({3, 4, 7, 8, 11, 12, 15, 16}));
    std::filesystem::remove_all(dir);
}

void test_stack_groups_exact()
{
    const auto dir = make_temp_snapshot("stack");
    write_safetensors(dir, {
        {"e0_gate", {"BF16", {2, 2}, bytes_of<std::uint16_t>({1, 2, 3, 4})}},
        {"e0_up", {"BF16", {2, 2}, bytes_of<std::uint16_t>({5, 6, 7, 8})}},
        {"e0_down", {"BF16", {2, 2}, bytes_of<std::uint16_t>({9, 10, 11, 12})}},
        {"e1_gate", {"BF16", {2, 2}, bytes_of<std::uint16_t>({13, 14, 15, 16})}},
        {"e1_up", {"BF16", {2, 2}, bytes_of<std::uint16_t>({17, 18, 19, 20})}},
        {"e1_down", {"BF16", {2, 2}, bytes_of<std::uint16_t>({21, 22, 23, 24})}},
    });

    LayoutPlan plan;
    plan.ops.push_back(make_stack_groups_op(
        "experts.gate_up", "experts.down", {},
        {{"e0_gate", "gate"}, {"e0_up", "up"}, {"e0_down", "down"},
         {"e1_gate", "gate"}, {"e1_up", "up"}, {"e1_down", "down"}}));
    register_spec(
        plan, "experts.gate_up", DType::BF16, {2, 4, 2},
        TensorLayoutKind::Grouped);
    register_spec(
        plan, "experts.down", DType::BF16, {2, 2, 2},
        TensorLayoutKind::Grouped);

    WeightStore weights;
    run_plan(dir, plan, weights);
    CHECK_EQ(read_tensor<std::uint16_t>(weights.get("experts.gate_up")),
             std::vector<std::uint16_t>(
                 {1, 2, 3, 4, 5, 6, 7, 8,
                  13, 14, 15, 16, 17, 18, 19, 20}));
    CHECK_EQ(read_tensor<std::uint16_t>(weights.get("experts.down")),
             std::vector<std::uint16_t>({9, 10, 11, 12, 21, 22, 23, 24}));
    std::filesystem::remove_all(dir);
}

void test_cast_exact()
{
    const auto dir = make_temp_snapshot("cast");
    write_safetensors(dir, {
        {"fp32", {"F32", {2}, bytes_of<std::uint32_t>({0x00000000u, 0x3f800000u})}},
    });

    LayoutPlan plan;
    plan.ops.push_back(make_raw_load_op(LayoutOpKind::Copy, "tmp", "fp32"));
    plan.ops.push_back(make_tensor_op(LayoutOpKind::Cast, "bf16", {"tmp"}));
    plan.ops.push_back(make_tensor_op(LayoutOpKind::Drop, "drop", {"tmp"}));
    register_spec(
        plan, "tmp", DType::FP32, {2}, TensorLayoutKind::Dense,
        TensorOwnershipKind::Temporary);
    register_spec(plan, "bf16", DType::BF16, {2});

    WeightStore weights;
    run_plan(dir, plan, weights);
    CHECK_EQ(read_tensor<std::uint16_t>(weights.get("bf16")),
             std::vector<std::uint16_t>({0x0000u, 0x3f80u}));
    std::filesystem::remove_all(dir);
}

void test_dequantize_exact()
{
    const auto dir = make_temp_snapshot("dequant");
    write_safetensors(dir, {
        {"fp8", {"F8_E4M3", {2, 2}, bytes_of<std::uint8_t>({0, 0, 0, 0})}},
        {"scale", {"F32", {1}, bytes_of<float>({1.0f})}},
    });

    LayoutPlan plan;
    plan.ops.push_back(make_raw_load_op(LayoutOpKind::Copy, "fp8_tmp", "fp8"));
    plan.ops.push_back(make_raw_load_op(LayoutOpKind::Copy, "scale_tmp", "scale"));
    plan.ops.push_back(make_tensor_op(
        LayoutOpKind::Dequantize, "bf16", {"fp8_tmp", "scale_tmp"}));
    plan.ops.push_back(make_tensor_op(
        LayoutOpKind::Drop, "drop", {"fp8_tmp", "scale_tmp"}));
    register_spec(
        plan, "fp8_tmp", DType::FP8_E4M3, {2, 2},
        TensorLayoutKind::Dense, TensorOwnershipKind::Temporary);
    register_spec(
        plan, "scale_tmp", DType::FP32, {1},
        TensorLayoutKind::Dense, TensorOwnershipKind::Temporary);
    register_spec(plan, "bf16", DType::BF16, {2, 2});

    WeightStore weights;
    run_plan(dir, plan, weights);
    CHECK_EQ(read_tensor<std::uint16_t>(weights.get("bf16")),
             std::vector<std::uint16_t>({0, 0, 0, 0}));
    std::filesystem::remove_all(dir);
}

void test_bind_metadata_exact()
{
    const auto dir = make_temp_snapshot("metadata");
    write_safetensors(dir, {
        {"w", {"I8", {2, 2}, bytes_of<std::int8_t>({1, 2, 3, 4})}},
        {"s", {"F32", {2}, bytes_of<float>({1.0f, 2.0f})}},
    });

    LayoutPlan plan;
    plan.ops.push_back(make_raw_load_op(LayoutOpKind::Copy, "w", "w"));
    plan.ops.push_back(make_raw_load_op(LayoutOpKind::Copy, "s", "s"));
    plan.ops.push_back(make_tensor_op(LayoutOpKind::AttachMetadata, "w"));

    QuantSpec quant;
    quant.format = QuantFormat::RuntimeInt8;
    quant.granularity = QuantGranularity::PerChannel;
    quant.channel_axis = 0;
    quant.scale_tensor = "s";
    register_spec(
        plan, "w", DType::INT8, {2, 2}, TensorLayoutKind::QuantPacked,
        TensorOwnershipKind::Owned, TensorParallelKind::Column,
        {}, quant);
    register_spec(plan, "s", DType::FP32, {2});

    WeightStore weights;
    run_plan(dir, plan, weights);
    const auto meta = weights.quant_meta("w");
    CHECK_TRUE(meta.has_value());
    CHECK_EQ(meta->kind, QuantMeta::Kind::PerChannel);
    CHECK_EQ(meta->channel_axis, 0);
    CHECK_TRUE(meta->scale == &weights.get("s"));
    CHECK_TRUE(meta->zero_point == nullptr);
    std::filesystem::remove_all(dir);
}

}  // namespace

int main()
{
    try {
        CUDA_CHECK(cudaSetDevice(0));
        test_axis_concat_exact();
        test_grouped_slice_exact();
        test_stack_groups_exact();
        test_cast_exact();
        test_dequantize_exact();
        test_bind_metadata_exact();
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    std::cout << "loader golden tests passed\n";
    return 0;
}
