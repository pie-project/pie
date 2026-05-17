#include <cuda_runtime.h>

#include <cmath>
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
#include "loader/layout_optimizer.hpp"
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

LayoutExprId add_expr(LayoutPlan& plan, LayoutExpr expr)
{
    const LayoutExprId id = plan.algebra.exprs.size();
    plan.algebra.exprs.push_back(std::move(expr));
    return id;
}

TensorDecl spec_for(const LayoutPlan& plan, const std::string& name)
{
    const auto it = plan.tensors.find(name);
    if (it == plan.tensors.end()) {
        throw std::runtime_error("missing spec: " + name);
    }
    return it->second;
}

LayoutExprId source(
    LayoutPlan& plan,
    std::string raw_name,
    TensorDecl decl)
{
    LayoutExpr expr;
    expr.kind = LayoutExprKind::Source;
    expr.raw_name = std::move(raw_name);
    expr.decl = std::move(decl);
    expr.dtype = expr.decl.dtype;
    expr.encoding = expr.decl.quant;
    return add_expr(plan, std::move(expr));
}

LayoutExprId realize(
    LayoutPlan& plan,
    std::string name,
    LayoutExprId input)
{
    TensorDecl decl = spec_for(plan, name);
    LayoutExpr expr;
    expr.kind = LayoutExprKind::Realize;
    expr.inputs = {input};
    expr.runtime_name = name;
    expr.decl = decl;
    expr.dtype = decl.dtype;
    expr.encoding = decl.quant;
    const LayoutExprId root = add_expr(plan, std::move(expr));
    plan.algebra.bindings.push_back(LayoutBinding{std::move(name), root});
    return root;
}

LayoutExprId select(
    LayoutPlan& plan,
    LayoutExprId input,
    TensorDecl decl,
    int axis,
    std::int64_t start,
    std::int64_t length)
{
    LayoutExpr expr;
    expr.kind = LayoutExprKind::Select;
    expr.inputs = {input};
    expr.decl = std::move(decl);
    expr.dtype = expr.decl.dtype;
    expr.encoding = expr.decl.quant;
    expr.axis = axis;
    expr.start = start;
    expr.length = length;
    return add_expr(plan, std::move(expr));
}

LayoutExprId partition(
    LayoutPlan& plan,
    LayoutExprId input,
    TensorDecl decl,
    int axis,
    int parts)
{
    LayoutExpr expr;
    expr.kind = LayoutExprKind::Partition;
    expr.inputs = {input};
    expr.decl = std::move(decl);
    expr.dtype = expr.decl.dtype;
    expr.encoding = expr.decl.quant;
    expr.axis = axis;
    expr.partitions = parts;
    return add_expr(plan, std::move(expr));
}

LayoutExprId join(
    LayoutPlan& plan,
    std::vector<LayoutExprId> inputs,
    TensorDecl decl,
    int axis)
{
    LayoutExpr expr;
    expr.kind = LayoutExprKind::Join;
    expr.inputs = std::move(inputs);
    expr.decl = std::move(decl);
    expr.dtype = expr.decl.dtype;
    expr.encoding = expr.decl.quant;
    expr.axis = axis;
    return add_expr(plan, std::move(expr));
}

LayoutExprId unary(
    LayoutPlan& plan,
    LayoutExprKind kind,
    LayoutExprId input,
    TensorDecl decl,
    std::string runtime_name = {},
    std::string secondary_name = {})
{
    LayoutExpr expr;
    expr.kind = kind;
    expr.inputs = {input};
    expr.decl = std::move(decl);
    expr.dtype = expr.decl.dtype;
    expr.encoding = expr.decl.quant;
    expr.runtime_name = std::move(runtime_name);
    expr.secondary_runtime_name = std::move(secondary_name);
    return add_expr(plan, std::move(expr));
}

void release(LayoutPlan& plan, std::vector<LayoutExprId> inputs)
{
    TensorDecl decl;
    decl.name = "__release";
    LayoutExpr expr;
    expr.kind = LayoutExprKind::Release;
    expr.inputs = std::move(inputs);
    expr.decl = std::move(decl);
    expr.runtime_name = "__release";
    (void)add_expr(plan, std::move(expr));
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
    const auto q = source(plan, "q", TensorDecl{
        .name = "q", .dtype = DType::BF16, .shape = {2, 2}});
    const auto k = source(plan, "k", TensorDecl{
        .name = "k", .dtype = DType::BF16, .shape = {1, 2}});
    const auto v = source(plan, "v", TensorDecl{
        .name = "v", .dtype = DType::BF16, .shape = {1, 2}});
    const auto packed = join(plan, {q, k, v}, spec_for(plan, "packed"), 0);
    realize(plan, "packed", packed);
    auto q_view_decl = spec_for(plan, "q_view");
    auto k_view_decl = spec_for(plan, "k_view");
    auto v_view_decl = spec_for(plan, "v_view");
    auto q_view = unary(plan, LayoutExprKind::View, packed, q_view_decl, "q_view");
    plan.algebra.exprs[q_view].axis = 0;
    plan.algebra.exprs[q_view].start = 0;
    plan.algebra.exprs[q_view].length = 2;
    plan.algebra.bindings.push_back(LayoutBinding{"q_view", q_view});
    auto k_view = unary(plan, LayoutExprKind::View, packed, k_view_decl, "k_view");
    plan.algebra.exprs[k_view].axis = 0;
    plan.algebra.exprs[k_view].start = 2;
    plan.algebra.exprs[k_view].length = 1;
    plan.algebra.bindings.push_back(LayoutBinding{"k_view", k_view});
    auto v_view = unary(plan, LayoutExprKind::View, packed, v_view_decl, "v_view");
    plan.algebra.exprs[v_view].axis = 0;
    plan.algebra.exprs[v_view].start = 3;
    plan.algebra.exprs[v_view].length = 1;
    plan.algebra.bindings.push_back(LayoutBinding{"v_view", v_view});

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
    register_spec(
        plan, "gate_up_local", DType::BF16, {2, 2, 2},
        TensorLayoutKind::Grouped, TensorOwnershipKind::Owned,
        TensorParallelKind::Expert);
    register_spec(
        plan, "down_local", DType::BF16, {2, 2, 2},
        TensorLayoutKind::Grouped, TensorOwnershipKind::Owned,
        TensorParallelKind::Expert);
    TensorDecl gate_up_source_decl;
    gate_up_source_decl.name = "gate_up";
    gate_up_source_decl.dtype = DType::BF16;
    gate_up_source_decl.shape = {2, 4, 2};
    TensorDecl half_decl = gate_up_source_decl;
    half_decl.name = "gate_up_half";
    half_decl.shape = {2, 2, 2};
    TensorDecl local_half_decl = half_decl;
    local_half_decl.shape = {2, 1, 2};
    const auto gate_up_src = source(plan, "gate_up", gate_up_source_decl);
    auto gate = select(plan, gate_up_src, half_decl, 1, 0, 2);
    gate = partition(plan, gate, local_half_decl, 1, 2);
    auto up = select(plan, gate_up_src, half_decl, 1, 2, 2);
    up = partition(plan, up, local_half_decl, 1, 2);
    realize(
        plan, "gate_up_local",
        join(plan, {gate, up}, spec_for(plan, "gate_up_local"), 1));

    TensorDecl down_source_decl;
    down_source_decl.name = "down";
    down_source_decl.dtype = DType::BF16;
    down_source_decl.shape = {2, 2, 4};
    const auto down_src = source(plan, "down", down_source_decl);
    realize(
        plan, "down_local",
        partition(plan, down_src, spec_for(plan, "down_local"), 2, 2));

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
    register_spec(
        plan, "experts.gate_up", DType::BF16, {2, 4, 2},
        TensorLayoutKind::Grouped);
    register_spec(
        plan, "experts.down", DType::BF16, {2, 2, 2},
        TensorLayoutKind::Grouped);
    TensorDecl expert_decl;
    expert_decl.dtype = DType::BF16;
    expert_decl.shape = {2, 2};
    std::vector<LayoutExprId> inputs;
    for (const auto& raw :
         {"e0_gate", "e0_up", "e0_down", "e1_gate", "e1_up", "e1_down"}) {
        expert_decl.name = raw;
        inputs.push_back(source(plan, raw, expert_decl));
    }
    LayoutExpr stack;
    stack.kind = LayoutExprKind::Stack;
    stack.inputs = std::move(inputs);
    stack.decl = spec_for(plan, "experts.gate_up");
    stack.dtype = DType::BF16;
    stack.runtime_name = "experts.gate_up";
    stack.secondary_runtime_name = "experts.down";
    stack.axis = 0;
    const auto stacked = add_expr(plan, std::move(stack));
    realize(plan, "experts.gate_up", stacked);
    realize(plan, "experts.down", stacked);

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
    register_spec(
        plan, "tmp", DType::FP32, {2}, TensorLayoutKind::Dense,
        TensorOwnershipKind::Temporary);
    register_spec(plan, "bf16", DType::BF16, {2});
    const auto fp32 = source(plan, "fp32", spec_for(plan, "tmp"));
    const auto cast = unary(
        plan, LayoutExprKind::Cast, fp32, spec_for(plan, "bf16"), "bf16");
    realize(plan, "bf16", cast);
    release(plan, {fp32});

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
    register_spec(
        plan, "fp8_tmp", DType::FP8_E4M3, {2, 2},
        TensorLayoutKind::Dense, TensorOwnershipKind::Temporary);
    register_spec(
        plan, "scale_tmp", DType::FP32, {1},
        TensorLayoutKind::Dense, TensorOwnershipKind::Temporary);
    register_spec(plan, "bf16", DType::BF16, {2, 2});
    const auto fp8 = source(plan, "fp8", spec_for(plan, "fp8_tmp"));
    const auto scale = source(plan, "scale", spec_for(plan, "scale_tmp"));
    LayoutExpr decode;
    decode.kind = LayoutExprKind::Decode;
    decode.inputs = {fp8, scale};
    decode.decl = spec_for(plan, "bf16");
    decode.dtype = DType::BF16;
    decode.runtime_name = "bf16";
    const auto decoded = add_expr(plan, std::move(decode));
    realize(plan, "bf16", decoded);
    release(plan, {fp8, scale});

    WeightStore weights;
    run_plan(dir, plan, weights);
    CHECK_EQ(read_tensor<std::uint16_t>(weights.get("bf16")),
             std::vector<std::uint16_t>({0, 0, 0, 0}));
    std::filesystem::remove_all(dir);
}

void test_encode_int8_exact()
{
    const auto dir = make_temp_snapshot("encode");
    write_safetensors(dir, {
        {"bf16", {"BF16", {1, 2}, bytes_of<std::uint16_t>({0x3f80u, 0xbf80u})}},
    });

    LayoutPlan plan;
    register_spec(
        plan, "tmp", DType::BF16, {1, 2}, TensorLayoutKind::Dense,
        TensorOwnershipKind::Temporary);
    QuantSpec quant;
    quant.format = QuantFormat::RuntimeInt8;
    quant.granularity = QuantGranularity::PerChannel;
    quant.channel_axis = 0;
    quant.scale_tensor = "q_scale";
    register_spec(
        plan, "q", DType::INT8, {1, 2}, TensorLayoutKind::QuantPacked,
        TensorOwnershipKind::Owned, TensorParallelKind::Column,
        {}, quant);
    register_spec(plan, "q_scale", DType::FP32, {1});

    const auto input = source(plan, "bf16", spec_for(plan, "tmp"));
    const auto encoded = unary(
        plan, LayoutExprKind::Encode, input, spec_for(plan, "q"),
        "q", "q_scale");
    realize(plan, "q", encoded);
    release(plan, {input});

    WeightStore weights;
    run_plan(dir, plan, weights);
    CHECK_EQ(read_tensor<std::int8_t>(weights.get("q")),
             std::vector<std::int8_t>({127, -127}));
    const auto scale = read_tensor<float>(weights.get("q_scale"));
    CHECK_TRUE(scale.size() == 1);
    CHECK_TRUE(scale.empty() || std::abs(scale[0] - (1.0f / 127.0f)) < 1e-6f);
    std::filesystem::remove_all(dir);
}

void test_transcode_fp8_to_int8_exact()
{
    const auto dir = make_temp_snapshot("transcode");
    write_safetensors(dir, {
        {"fp8", {"F8_E4M3", {2, 2}, bytes_of<std::uint8_t>({0, 0, 0, 0})}},
        {"scale", {"F32", {1}, bytes_of<float>({1.0f})}},
    });

    LayoutPlan plan;
    register_spec(
        plan, "fp8_tmp", DType::FP8_E4M3, {2, 2},
        TensorLayoutKind::Dense, TensorOwnershipKind::Temporary);
    register_spec(
        plan, "scale_tmp", DType::FP32, {1},
        TensorLayoutKind::Dense, TensorOwnershipKind::Temporary);
    register_spec(
        plan, "decoded_tmp", DType::BF16, {2, 2},
        TensorLayoutKind::Dense, TensorOwnershipKind::Temporary);
    QuantSpec quant;
    quant.format = QuantFormat::RuntimeInt8;
    quant.granularity = QuantGranularity::PerChannel;
    quant.channel_axis = 0;
    quant.scale_tensor = "q_scale";
    register_spec(
        plan, "q", DType::INT8, {2, 2}, TensorLayoutKind::QuantPacked,
        TensorOwnershipKind::Owned, TensorParallelKind::Column,
        {}, quant);
    register_spec(plan, "q_scale", DType::FP32, {2});

    const auto fp8 = source(plan, "fp8", spec_for(plan, "fp8_tmp"));
    const auto scale = source(plan, "scale", spec_for(plan, "scale_tmp"));
    LayoutExpr decode;
    decode.kind = LayoutExprKind::Decode;
    decode.inputs = {fp8, scale};
    decode.decl = spec_for(plan, "decoded_tmp");
    decode.dtype = DType::BF16;
    const auto decoded = add_expr(plan, std::move(decode));
    const auto encoded = unary(
        plan, LayoutExprKind::Encode, decoded, spec_for(plan, "q"),
        "q", "q_scale");
    realize(plan, "q", encoded);
    release(plan, {fp8, scale});
    (void)optimize_layout_algebra(plan);

    WeightStore weights;
    run_plan(dir, plan, weights);
    CHECK_EQ(read_tensor<std::int8_t>(weights.get("q")),
             std::vector<std::int8_t>({0, 0, 0, 0}));
    CHECK_EQ(read_tensor<float>(weights.get("q_scale")),
             std::vector<float>({1.0f, 1.0f}));
    std::filesystem::remove_all(dir);
}

void test_unzip_exact()
{
    const auto dir = make_temp_snapshot("unzip");
    write_safetensors(dir, {
        {"fused", {"BF16", {1, 4, 2},
                   bytes_of<std::uint16_t>({1, 2, 3, 4, 5, 6, 7, 8})}},
    });

    LayoutPlan plan;
    register_spec(
        plan, "fused_tmp", DType::BF16, {1, 4, 2},
        TensorLayoutKind::Dense, TensorOwnershipKind::Temporary);
    register_spec(plan, "gate", DType::BF16, {1, 2, 2});
    register_spec(plan, "up", DType::BF16, {1, 2, 2});
    const auto fused = source(plan, "fused", spec_for(plan, "fused_tmp"));
    LayoutExpr unzip;
    unzip.kind = LayoutExprKind::Unzip;
    unzip.inputs = {fused};
    unzip.decl = spec_for(plan, "gate");
    unzip.dtype = DType::BF16;
    unzip.runtime_name = "gate";
    unzip.secondary_runtime_name = "up";
    unzip.axis = -1;
    const auto unzipped = add_expr(plan, std::move(unzip));
    realize(plan, "gate", unzipped);
    realize(plan, "up", unzipped);
    release(plan, {fused});

    WeightStore weights;
    run_plan(dir, plan, weights);
    CHECK_EQ(read_tensor<std::uint16_t>(weights.get("gate")),
             std::vector<std::uint16_t>({1, 2, 5, 6}));
    CHECK_EQ(read_tensor<std::uint16_t>(weights.get("up")),
             std::vector<std::uint16_t>({3, 4, 7, 8}));
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
    const auto w = source(plan, "w", spec_for(plan, "w"));
    const auto s = source(plan, "s", spec_for(plan, "s"));
    realize(plan, "w", w);
    realize(plan, "s", s);
    LayoutExpr attach;
    attach.kind = LayoutExprKind::Attach;
    attach.inputs = {w};
    attach.decl = spec_for(plan, "w");
    attach.dtype = DType::INT8;
    attach.encoding = attach.decl.quant;
    attach.runtime_name = "w";
    (void)add_expr(plan, std::move(attach));

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
        test_encode_int8_exact();
        test_transcode_fp8_to_int8_exact();
        test_unzip_exact();
        test_bind_metadata_exact();
    } catch (const std::exception& e) {
        std::cerr << e.what() << "\n";
        return 1;
    }
    std::cout << "loader golden tests passed\n";
    return 0;
}
