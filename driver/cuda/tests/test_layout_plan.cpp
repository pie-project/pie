#include <algorithm>
#include <array>
#include <cstdint>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "loader/model_schema.hpp"
#include "loader/model_adapter.hpp"
#include "loader/checkpoint_source.hpp"
#include "loader/gguf_source.hpp"
#include "loader/layout_optimizer.hpp"
#include "loader/layout_plan.hpp"
#include "loader/layout_planner.hpp"
#include "loader/runtime_abi.hpp"
#include "loader/storage_compiler.hpp"
#include "loader/storage_program.hpp"

namespace {

int g_failures = 0;

#define CHECK(cond)                                                   \
    do {                                                              \
        if (!(cond)) {                                                \
            std::fprintf(stderr, "FAIL: %s:%d: %s\n",                 \
                         __FILE__, __LINE__, #cond);                 \
            ++g_failures;                                             \
        }                                                             \
    } while (0)

#define CHECK_EQ(a, b)                                                \
    do {                                                              \
        const auto _a = (a);                                          \
        const auto _b = (b);                                          \
        if (!(_a == _b)) {                                            \
            std::fprintf(stderr, "FAIL: %s:%d: %s == %s\n",           \
                         __FILE__, __LINE__, #a, #b);                 \
            ++g_failures;                                             \
        }                                                             \
    } while (0)

#define CHECK_THROWS(expr)                                            \
    do {                                                              \
        bool _threw = false;                                          \
        try {                                                         \
            (void)(expr);                                             \
        } catch (const std::exception&) {                             \
            _threw = true;                                            \
        }                                                             \
        if (!_threw) {                                                \
            std::fprintf(stderr, "FAIL: %s:%d: expected throw: %s\n", \
                         __FILE__, __LINE__, #expr);                 \
            ++g_failures;                                             \
        }                                                             \
    } while (0)

void write_u32(std::ofstream& out, std::uint32_t value) {
    char bytes[4] = {
        static_cast<char>(value & 0xffu),
        static_cast<char>((value >> 8) & 0xffu),
        static_cast<char>((value >> 16) & 0xffu),
        static_cast<char>((value >> 24) & 0xffu),
    };
    out.write(bytes, sizeof(bytes));
}

void write_u64(std::ofstream& out, std::uint64_t value) {
    char bytes[8] = {};
    for (int i = 0; i < 8; ++i) {
        bytes[i] = static_cast<char>((value >> (8 * i)) & 0xffu);
    }
    out.write(bytes, sizeof(bytes));
}

void write_gguf_string(std::ofstream& out, const std::string& value) {
    write_u64(out, static_cast<std::uint64_t>(value.size()));
    out.write(value.data(), static_cast<std::streamsize>(value.size()));
}

void align_stream(std::ofstream& out, std::uint64_t alignment) {
    const auto pos = static_cast<std::uint64_t>(
        static_cast<std::streamoff>(out.tellp()));
    const std::uint64_t padding = (alignment - (pos % alignment)) % alignment;
    for (std::uint64_t i = 0; i < padding; ++i) {
        out.put('\0');
    }
}

class FakeMetadata final : public pie_cuda_driver::CheckpointSource {
public:
    void add(std::string name,
             pie_cuda_driver::DType dtype,
             std::vector<std::int64_t> shape) {
        pie_cuda_driver::TensorInfo info;
        info.dtype = dtype;
        info.shape = std::move(shape);
        info.nbytes = pie_cuda_driver::dtype_bytes(dtype);
        for (const auto dim : info.shape) {
            info.nbytes *= static_cast<std::uint64_t>(dim);
        }
        info.data_offset = next_offset_;
        next_offset_ += info.nbytes;
        names_.push_back(name);
        tensors_.emplace(std::move(name), std::move(info));
        std::sort(names_.begin(), names_.end());
    }

    std::vector<std::string> tensor_names() const override {
        return names_;
    }

    std::size_t num_tensors() const noexcept override {
        return tensors_.size();
    }

    const pie_cuda_driver::TensorInfo& info(
        const std::string& name) const override {
        const auto it = tensors_.find(name);
        if (it == tensors_.end()) {
            throw std::runtime_error("missing fake tensor: " + name);
        }
        return it->second;
    }

    bool contains(const std::string& name) const noexcept override {
        return tensors_.find(name) != tensors_.end();
    }

private:
    std::vector<std::string> names_;
    std::unordered_map<std::string, pie_cuda_driver::TensorInfo> tensors_;
    std::uint64_t next_offset_ = 0;
};

std::size_t count_exprs(const pie_cuda_driver::LayoutPlan& plan,
                        pie_cuda_driver::LayoutExprKind kind) {
    return static_cast<std::size_t>(std::count_if(
        plan.algebra.exprs.begin(), plan.algebra.exprs.end(),
        [kind](const auto& expr) { return expr.kind == kind; }));
}

std::size_t count_storage_instrs(
    const pie_cuda_driver::StorageProgram& program,
    pie_cuda_driver::StorageInstrKind kind) {
    return static_cast<std::size_t>(std::count_if(
        program.schedule.begin(), program.schedule.end(),
        [kind](const auto& instr) { return instr.kind == kind; }));
}

std::size_t count_storage_transforms(
    const pie_cuda_driver::StorageProgram& program,
    pie_cuda_driver::StorageTransformKind kind) {
    return static_cast<std::size_t>(std::count_if(
        program.schedule.begin(), program.schedule.end(),
        [kind](const auto& instr) {
            return instr.kind == pie_cuda_driver::StorageInstrKind::Transform &&
                   instr.transform_kind == kind;
        }));
}

std::size_t count_tile_maps(
    const pie_cuda_driver::StorageProgram& program,
    pie_cuda_driver::TileMapKind kind) {
    return static_cast<std::size_t>(std::count_if(
        program.tile_maps.begin(), program.tile_maps.end(),
        [kind](const auto& tile) { return tile.kind == kind; }));
}

const pie_cuda_driver::TensorDecl* find_spec(
    const pie_cuda_driver::LayoutPlan& plan,
    const std::string& name) {
    const auto it = plan.tensors.find(name);
    return it == plan.tensors.end() ? nullptr : &it->second;
}

const pie_cuda_driver::LayoutExpr* find_expr(
    const pie_cuda_driver::LayoutPlan& plan,
    pie_cuda_driver::LayoutExprKind kind,
    const std::string& runtime_name) {
    const auto it = std::find_if(
        plan.algebra.exprs.begin(), plan.algebra.exprs.end(),
        [&](const auto& expr) {
            return expr.kind == kind && expr.runtime_name == runtime_name;
        });
    return it == plan.algebra.exprs.end() ? nullptr : &*it;
}

bool has_group(const pie_cuda_driver::SemanticGraph& graph,
               pie_cuda_driver::SemanticGroupKind kind) {
    return std::any_of(
        graph.groups.begin(), graph.groups.end(),
        [kind](const auto& group) { return group.kind == kind; });
}

const pie_cuda_driver::SemanticGroup* find_group(
    const pie_cuda_driver::SemanticGraph& graph,
    pie_cuda_driver::SemanticGroupKind kind) {
    const auto it = std::find_if(
        graph.groups.begin(), graph.groups.end(),
        [kind](const auto& group) { return group.kind == kind; });
    return it == graph.groups.end() ? nullptr : &*it;
}

pie_cuda_driver::HfConfig qwen3_config() {
    pie_cuda_driver::HfConfig hf{};
    hf.model_type = "qwen3";
    hf.torch_dtype = "bfloat16";
    return hf;
}

void test_runtime_abi_declares_final_layout_names() {
    const auto& abi = pie_cuda_driver::pie_cuda_runtime_abi();
    const auto qkv = abi.packed_projection(
        pie_cuda_driver::RuntimeProjectionPackKind::AttentionQkvRows,
        "model.layers.0.self_attn");
    CHECK_EQ(qkv.storage_name,
             std::string("model.layers.0.self_attn.qkv_proj.fused.weight"));
    CHECK(qkv.storage_layout ==
          pie_cuda_driver::TensorLayoutKind::AxisConcatenated);
    CHECK(qkv.view_layout == pie_cuda_driver::TensorLayoutKind::View);

    const auto gate_up = abi.packed_projection(
        pie_cuda_driver::RuntimeProjectionPackKind::MlpGateUpRows,
        "model.layers.0.mlp");
    CHECK_EQ(gate_up.storage_name,
             std::string("model.layers.0.mlp.gate_up_proj.fused.weight"));

    const auto experts =
        abi.fused_expert_bank("model.layers.0.mlp");
    CHECK_EQ(experts.gate_up_name,
             std::string("model.layers.0.mlp.experts.gate_up_proj"));
    CHECK_EQ(experts.down_name,
             std::string("model.layers.0.mlp.experts.down_proj"));
    CHECK(experts.layout == pie_cuda_driver::TensorLayoutKind::Grouped);

    CHECK_EQ(abi.quant_scale_inv_name("model.layers.0.w.weight"),
             std::string("model.layers.0.w.weight_scale_inv"));
    pie_cuda_driver::QuantSpec quant;
    quant.format = pie_cuda_driver::QuantFormat::RuntimeInt8;
    quant.granularity = pie_cuda_driver::QuantGranularity::PerChannel;
    quant.scale_tensor = "runtime.weight_scale";
    const auto contract = abi.tensor_contract(
        "runtime.weight",
        pie_cuda_driver::DType::INT8,
        {8, 8},
        pie_cuda_driver::TensorLayoutKind::QuantPacked,
        pie_cuda_driver::TensorOwnershipKind::Owned,
        pie_cuda_driver::TensorParallelKind::Row,
        quant,
        pie_cuda_driver::RuntimeQuantPolicyKind::NativePacked);
    CHECK_EQ(contract.name, std::string("runtime.weight"));
    CHECK(contract.encoding ==
          pie_cuda_driver::RuntimeEncodingKind::RuntimeInt8);
    CHECK(contract.quant_policy ==
          pie_cuda_driver::RuntimeQuantPolicyKind::NativePacked);
    CHECK_EQ(contract.alignment_bytes, std::uint64_t{256});
}

void test_gguf_checkpoint_source_reads_dense_tensor_extents() {
    const auto path =
        std::filesystem::temp_directory_path() /
        "pie_test_layout_plan_dense.gguf";
    std::filesystem::remove(path);
    {
        std::ofstream out(path, std::ios::binary);
        out.write("GGUF", 4);
        write_u32(out, 3);  // version
        write_u64(out, 1);  // tensor_count
        write_u64(out, 1);  // metadata_count

        write_gguf_string(out, "general.alignment");
        write_u32(out, 4);   // GGUF_METADATA_VALUE_TYPE_UINT32
        write_u32(out, 32);  // alignment

        write_gguf_string(out, "tensor.weight");
        write_u32(out, 2);  // rank
        write_u64(out, 2);
        write_u64(out, 3);
        write_u32(out, 0);  // GGML_TYPE_F32
        write_u64(out, 0);  // tensor data offset from data section

        align_stream(out, 32);
        for (int i = 0; i < 6; ++i) {
            write_u32(out, 0);
        }
    }

    const auto source = pie_cuda_driver::GgufCheckpointSource::open(path);
    CHECK_EQ(source.version(), std::uint32_t{3});
    CHECK_EQ(source.alignment(), std::uint64_t{32});
    CHECK_EQ(source.num_tensors(), std::size_t{1});
    CHECK(source.contains("tensor.weight"));
    const auto& info = source.info("tensor.weight");
    CHECK(info.dtype == pie_cuda_driver::DType::FP32);
    CHECK((info.shape == std::vector<std::int64_t>{2, 3}));
    CHECK_EQ(info.nbytes, std::uint64_t{24});
    CHECK_EQ(info.data_offset, std::uint64_t{0});

    const auto storage = source.storage_info("tensor.weight");
    CHECK_EQ(storage.path, path);
    CHECK_EQ(storage.nbytes, std::uint64_t{24});
    CHECK_EQ(storage.file_offset % 32, std::uint64_t{0});
    std::filesystem::remove(path);
}

void test_gguf_checkpoint_source_reads_q4_0_block_extents_and_decode() {
    const auto path =
        std::filesystem::temp_directory_path() /
        "pie_test_layout_plan_q4_0.gguf";
    std::filesystem::remove(path);
    std::array<std::uint8_t, 18> block{};
    block[0] = 0x00;  // fp16 1.0, little-endian
    block[1] = 0x3c;
    for (std::size_t i = 0; i < 16; ++i) {
        block[2 + i] = static_cast<std::uint8_t>(
            ((i & 0x0f) << 4) | ((15 - i) & 0x0f));
    }
    {
        std::ofstream out(path, std::ios::binary);
        out.write("GGUF", 4);
        write_u32(out, 3);  // version
        write_u64(out, 1);  // tensor_count
        write_u64(out, 1);  // metadata_count

        write_gguf_string(out, "general.alignment");
        write_u32(out, 4);   // GGUF_METADATA_VALUE_TYPE_UINT32
        write_u32(out, 32);  // alignment

        write_gguf_string(out, "tensor.q4_0");
        write_u32(out, 1);   // rank
        write_u64(out, 32);  // one Q4_0 block
        write_u32(out, 2);   // GGML_TYPE_Q4_0
        write_u64(out, 0);   // tensor data offset from data section

        align_stream(out, 32);
        out.write(
            reinterpret_cast<const char*>(block.data()),
            static_cast<std::streamsize>(block.size()));
    }

    const auto source = pie_cuda_driver::GgufCheckpointSource::open(path);
    CHECK(source.contains("tensor.q4_0"));
    const auto& info = source.info("tensor.q4_0");
    CHECK(info.dtype == pie_cuda_driver::DType::UINT8);
    CHECK((info.shape == std::vector<std::int64_t>{32}));
    CHECK_EQ(info.encoding, std::string("gguf.q4_0"));
    CHECK_EQ(info.block_elements, std::uint32_t{32});
    CHECK_EQ(info.block_bytes, std::uint32_t{18});
    CHECK_EQ(info.nbytes, std::uint64_t{18});

    const auto decoded =
        pie_cuda_driver::decode_gguf_q4_0_block(block.data(), block.size());
    CHECK_EQ(decoded.size(), std::size_t{32});
    CHECK((decoded == std::vector<float>{
        7.0f, 6.0f, 5.0f, 4.0f, 3.0f, 2.0f, 1.0f, 0.0f,
        -1.0f, -2.0f, -3.0f, -4.0f, -5.0f, -6.0f, -7.0f, -8.0f,
        -8.0f, -7.0f, -6.0f, -5.0f, -4.0f, -3.0f, -2.0f, -1.0f,
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f}));
    std::filesystem::remove(path);
}

FakeMetadata dense_llama_metadata() {
    using pie_cuda_driver::DType;
    FakeMetadata meta;
    meta.add("model.embed_tokens.weight", DType::BF16, {32, 8});
    meta.add("model.layers.0.input_layernorm.weight", DType::BF16, {8});
    meta.add("model.layers.0.post_attention_layernorm.weight", DType::BF16, {8});
    meta.add("model.layers.0.self_attn.q_proj.weight", DType::BF16, {8, 8});
    meta.add("model.layers.0.self_attn.k_proj.weight", DType::BF16, {4, 8});
    meta.add("model.layers.0.self_attn.v_proj.weight", DType::BF16, {4, 8});
    meta.add("model.layers.0.self_attn.o_proj.weight", DType::BF16, {8, 8});
    meta.add("model.layers.0.mlp.gate_proj.weight", DType::BF16, {16, 8});
    meta.add("model.layers.0.mlp.up_proj.weight", DType::BF16, {16, 8});
    meta.add("model.layers.0.mlp.down_proj.weight", DType::BF16, {8, 16});
    meta.add("model.norm.weight", DType::BF16, {8});
    meta.add("lm_head.weight", DType::BF16, {32, 8});
    return meta;
}

void test_dense_qwen_plan_packs_projection_groups() {
    auto hf = qwen3_config();
    pie_cuda_driver::Config cfg{};
    const auto meta = dense_llama_metadata();
    const pie_cuda_driver::BackendTarget target{};
    auto plan = pie_cuda_driver::build_model_layout_plan(
        hf, cfg, meta, /*tp_size=*/1, target);
    pie_cuda_driver::validate_layout_plan(plan);

    CHECK_EQ(plan.axis_concat_groups, std::size_t{2});
    CHECK_EQ(plan.memory.max_temporary_bytes, std::uint64_t{0});
    CHECK(!plan.algebra.bindings.empty());

    const auto* qkv =
        find_spec(plan, "model.layers.0.self_attn.qkv_proj.fused.weight");
    CHECK(qkv != nullptr);
    if (qkv != nullptr) {
        CHECK(qkv->layout == pie_cuda_driver::TensorLayoutKind::AxisConcatenated);
        CHECK((qkv->shape == std::vector<std::int64_t>{16, 8}));
    }
    const auto* q_view =
        find_spec(plan, "model.layers.0.self_attn.q_proj.weight");
    CHECK(q_view != nullptr);
    if (q_view != nullptr) {
        CHECK(q_view->ownership ==
              pie_cuda_driver::TensorOwnershipKind::BorrowedView);
        CHECK_EQ(q_view->backing_tensor,
                 std::string("model.layers.0.self_attn.qkv_proj.fused.weight"));
    }
}

void test_storage_program_lowers_packed_groups_to_extent_writes() {
    auto hf = qwen3_config();
    pie_cuda_driver::Config cfg{};
    const auto meta = dense_llama_metadata();
    const pie_cuda_driver::BackendTarget target{};
    auto plan = pie_cuda_driver::build_model_layout_plan(
        hf, cfg, meta, /*tp_size=*/1, target);
    pie_cuda_driver::StorageProgram storage =
        pie_cuda_driver::compile_storage_program(
        plan, meta, /*tp_rank=*/0, /*tp_size=*/1);

    CHECK_EQ(storage.memory.max_extent_temporary_bytes, std::uint64_t{0});
    CHECK(storage.memory.extent_write_count >= 5);
    CHECK(storage.memory.algebra_extent_write_count >= 5);
    std::vector<const pie_cuda_driver::ExtentWrite*> qkv_writes;
    for (const auto& write : storage.extent_writes) {
        if (write.output_name ==
            "model.layers.0.self_attn.qkv_proj.fused.weight") {
            qkv_writes.push_back(&write);
        }
    }
    CHECK_EQ(qkv_writes.size(), std::size_t{3});
    if (qkv_writes.size() == 3) {
        CHECK_EQ(qkv_writes[0]->dst_offset_bytes, std::uint64_t{0});
        CHECK_EQ(qkv_writes[1]->dst_offset_bytes, std::uint64_t{128});
        CHECK_EQ(qkv_writes[2]->dst_offset_bytes, std::uint64_t{192});
        CHECK((qkv_writes[0]->dst_shape == std::vector<std::int64_t>{8, 8}));
        CHECK((qkv_writes[1]->dst_shape == std::vector<std::int64_t>{4, 8}));
        CHECK((qkv_writes[2]->dst_shape == std::vector<std::int64_t>{4, 8}));
    }
    const std::string dump =
        pie_cuda_driver::dump_layout_plan_json(plan, storage);
    CHECK(dump.find("\"plan_kind\": \"LayoutPlan\"") != std::string::npos);
    CHECK(dump.find("\"program_kind\":\"StorageProgram\"") != std::string::npos);
    CHECK(dump.find("\"write_kind\":\"ExtentWrite\"") != std::string::npos);
    CHECK(dump.find("\"storage\"") != std::string::npos);
    CHECK(dump.find("axis-concatenated") != std::string::npos);
}

void test_storage_compiler_covers_every_semantic_expr() {
    auto hf = qwen3_config();
    pie_cuda_driver::Config cfg{};
    cfg.model.runtime_quant = "int8";
    const auto meta = dense_llama_metadata();
    pie_cuda_driver::BackendTarget target{};
    auto plan = pie_cuda_driver::build_model_layout_plan(
        hf, cfg, meta, /*tp_size=*/1, target);
    pie_cuda_driver::StorageProgram storage =
        pie_cuda_driver::compile_storage_program(
        plan, meta, /*tp_rank=*/0, /*tp_size=*/1);
    pie_cuda_driver::validate_storage_program(plan, storage);

    CHECK(storage.memory.extent_write_count > 0);
    CHECK_EQ(count_tile_maps(storage, pie_cuda_driver::TileMapKind::Encode),
             std::size_t{7});
    CHECK_EQ(count_storage_instrs(
                 storage, pie_cuda_driver::StorageInstrKind::Attach),
             std::size_t{7});

    const std::string dump =
        pie_cuda_driver::dump_layout_plan_json(plan, storage);
    CHECK(dump.find("ExtentWrite") != std::string::npos);
    CHECK(dump.find("TileMap") != std::string::npos);
    CHECK(dump.find("Attach") != std::string::npos);
}

void test_storage_schedule_orders_ready_extent_writes_by_file_offset() {
    using pie_cuda_driver::DType;

    FakeMetadata meta;
    meta.add("checkpoint.b", DType::BF16, {2});
    meta.add("checkpoint.a", DType::BF16, {2});

    pie_cuda_driver::LayoutPlan plan;
    plan.tensors.emplace(
        "runtime.a",
        pie_cuda_driver::TensorDecl{
            .name = "runtime.a",
            .dtype = DType::BF16,
            .shape = {2},
        });
    plan.tensors.emplace(
        "runtime.b",
        pie_cuda_driver::TensorDecl{
            .name = "runtime.b",
            .dtype = DType::BF16,
            .shape = {2},
        });
    auto add_source_binding =
        [&](const std::string& raw_name, const std::string& runtime_name) {
            const auto decl = plan.tensors.at(runtime_name);
            pie_cuda_driver::LayoutExpr source;
            source.kind = pie_cuda_driver::LayoutExprKind::Source;
            source.raw_name = raw_name;
            source.decl = decl;
            source.decl.name = raw_name;
            source.dtype = decl.dtype;
            const auto source_id = plan.algebra.exprs.size();
            plan.algebra.exprs.push_back(std::move(source));

            pie_cuda_driver::LayoutExpr realize;
            realize.kind = pie_cuda_driver::LayoutExprKind::Realize;
            realize.inputs = {source_id};
            realize.decl = decl;
            realize.runtime_name = runtime_name;
            realize.dtype = decl.dtype;
            const auto root = plan.algebra.exprs.size();
            plan.algebra.exprs.push_back(std::move(realize));
            plan.algebra.bindings.push_back(
                pie_cuda_driver::LayoutBinding{runtime_name, root});
        };
    add_source_binding("checkpoint.a", "runtime.a");
    add_source_binding("checkpoint.b", "runtime.b");

    const pie_cuda_driver::StorageProgram storage =
        pie_cuda_driver::compile_storage_program(
        plan, meta, /*tp_rank=*/0, /*tp_size=*/1);
    CHECK_EQ(storage.extent_writes.size(), std::size_t{2});
    CHECK_EQ(storage.scheduled_extent_writes.size(), std::size_t{2});
    if (storage.scheduled_extent_writes.size() == 2) {
        CHECK_EQ(storage.scheduled_extent_writes[0], std::size_t{1});
        CHECK_EQ(storage.scheduled_extent_writes[1], std::size_t{0});
    }
    CHECK(!storage.schedule.empty());
    CHECK(storage.memory.file_ordered_extent_write_count == 2);
    const std::string dump =
        pie_cuda_driver::dump_layout_plan_json(plan, storage);
    CHECK(dump.find("\"schedule\"") != std::string::npos);
    CHECK(dump.find("\"source_offset_bytes\"") != std::string::npos);
}

void test_storage_compiler_lowers_algebra_only_copy_plan() {
    using namespace pie_cuda_driver;

    FakeMetadata meta;
    meta.add("checkpoint.weight", DType::BF16, {2, 2});

    TensorDecl decl{
        .name = "runtime.weight",
        .dtype = DType::BF16,
        .shape = {2, 2},
    };

    LayoutPlan plan;
    plan.tensors.emplace(decl.name, decl);
    plan.memory.persistent_bytes = 8;
    plan.memory.estimated_peak_bytes = 8;

    LayoutExpr source;
    source.kind = LayoutExprKind::Source;
    source.decl = decl;
    source.decl.name = "checkpoint.weight";
    source.raw_name = "checkpoint.weight";
    source.dtype = DType::BF16;
    const LayoutExprId source_id = plan.algebra.exprs.size();
    plan.algebra.exprs.push_back(std::move(source));

    LayoutExpr realize;
    realize.kind = LayoutExprKind::Realize;
    realize.inputs = {source_id};
    realize.decl = decl;
    realize.runtime_name = decl.name;
    realize.dtype = DType::BF16;
    const LayoutExprId root = plan.algebra.exprs.size();
    plan.algebra.exprs.push_back(std::move(realize));
    plan.algebra.bindings.push_back(LayoutBinding{
        .runtime_name = decl.name,
        .root = root,
    });

    const StorageProgram storage =
        compile_storage_program(plan, meta, /*tp_rank=*/0, /*tp_size=*/1);
    CHECK_EQ(storage.extent_writes.size(), std::size_t{1});
    CHECK_EQ(storage.schedule.size(), std::size_t{1});
    CHECK_EQ(storage.memory.algebra_extent_write_count, std::uint64_t{1});
    if (!storage.extent_writes.empty()) {
        const auto& write = storage.extent_writes.front();
        CHECK_EQ(write.op_index, kInvalidStorageId);
        CHECK_EQ(write.expr_id, root);
        CHECK_EQ(write.binding_index, std::size_t{0});
        CHECK_EQ(write.raw_name, std::string("checkpoint.weight"));
        CHECK_EQ(write.output_name, std::string("runtime.weight"));
    }
    if (!storage.schedule.empty()) {
        CHECK_EQ(storage.schedule.front().op_index, kInvalidStorageId);
        CHECK_EQ(storage.schedule.front().expr_id, root);
        CHECK_EQ(storage.schedule.front().binding_index, std::size_t{0});
    }
    const std::string dump = dump_storage_program_json(storage);
    CHECK(dump.find("\"op_index\":null") != std::string::npos);
    CHECK(dump.find("\"expr_id\":1") != std::string::npos);
    CHECK(dump.find("\"binding_index\":0") != std::string::npos);
}

void test_native_layout_planner_builds_dense_packed_algebra_plan() {
    using namespace pie_cuda_driver;

    auto hf = qwen3_config();
    Config cfg{};
    const auto meta = dense_llama_metadata();
    const SemanticGraph graph = build_model_semantic_graph(hf, cfg, meta);
    LayoutPlan plan = build_native_dense_algebra_plan(
        graph, meta, /*tp_size=*/1);
    validate_layout_plan(plan);

    CHECK_EQ(plan.axis_concat_groups, std::size_t{2});
    CHECK(find_spec(
        plan,
        "model.layers.0.self_attn.qkv_proj.fused.weight") != nullptr);
    CHECK(find_spec(
        plan,
        "model.layers.0.mlp.gate_up_proj.fused.weight") != nullptr);
    const auto* q_view = find_spec(
        plan,
        "model.layers.0.self_attn.q_proj.weight");
    CHECK(q_view != nullptr);
    if (q_view != nullptr) {
        CHECK(q_view->ownership == TensorOwnershipKind::BorrowedView);
        CHECK_EQ(q_view->backing_tensor,
                 std::string("model.layers.0.self_attn.qkv_proj.fused.weight"));
    }

    const StorageProgram storage =
        compile_storage_program(plan, meta, /*tp_rank=*/0, /*tp_size=*/1);
    CHECK_EQ(storage.memory.algebra_extent_write_count, std::uint64_t{12});
    CHECK(std::any_of(
        storage.schedule.begin(), storage.schedule.end(),
        [](const auto& instr) {
            return instr.kind == StorageInstrKind::CreateView &&
                   instr.op_index == kInvalidStorageId;
        }));

    std::vector<const ExtentWrite*> qkv_writes;
    for (const auto& write : storage.extent_writes) {
        if (write.output_name ==
            "model.layers.0.self_attn.qkv_proj.fused.weight") {
            qkv_writes.push_back(&write);
        }
    }
    CHECK_EQ(qkv_writes.size(), std::size_t{3});
    if (qkv_writes.size() == 3) {
        CHECK_EQ(qkv_writes[0]->dst_offset_bytes, std::uint64_t{0});
        CHECK_EQ(qkv_writes[1]->dst_offset_bytes, std::uint64_t{128});
        CHECK_EQ(qkv_writes[2]->dst_offset_bytes, std::uint64_t{192});
    }
}

void test_layout_optimizer_pushes_select_and_sinks_cast() {
    using namespace pie_cuda_driver;
    LayoutPlan plan;
    auto add_expr = [&](LayoutExpr expr) {
        const LayoutExprId id = plan.algebra.exprs.size();
        plan.algebra.exprs.push_back(std::move(expr));
        return id;
    };
    auto source = [&](std::string raw, std::string name, DType dtype) {
        TensorDecl decl{
            .name = std::move(name),
            .dtype = dtype,
            .shape = {4, 2},
        };
        LayoutExpr expr;
        expr.kind = LayoutExprKind::Source;
        expr.decl = decl;
        expr.raw_name = std::move(raw);
        expr.dtype = dtype;
        return add_expr(std::move(expr));
    };

    const auto a = source("a", "a", DType::FP16);
    const auto b = source("b", "b", DType::FP16);
    TensorDecl joined{
        .name = "joined",
        .dtype = DType::FP16,
        .shape = {8, 2},
    };
    LayoutExpr join;
    join.kind = LayoutExprKind::Join;
    join.inputs = {a, b};
    join.decl = joined;
    join.axis = 0;
    join.dtype = DType::FP16;
    const auto j = add_expr(std::move(join));

    TensorDecl selected = joined;
    selected.name = "selected";
    selected.shape = {4, 2};
    LayoutExpr select;
    select.kind = LayoutExprKind::Select;
    select.inputs = {j};
    select.decl = selected;
    select.axis = 0;
    select.start = 2;
    select.length = 4;
    select.dtype = DType::FP16;
    const auto s = add_expr(std::move(select));

    TensorDecl casted = selected;
    casted.name = "casted";
    casted.dtype = DType::BF16;
    LayoutExpr cast;
    cast.kind = LayoutExprKind::Cast;
    cast.inputs = {s};
    cast.decl = casted;
    cast.dtype = DType::BF16;
    const auto c = add_expr(std::move(cast));

    LayoutExpr realize;
    realize.kind = LayoutExprKind::Realize;
    realize.inputs = {c};
    realize.decl = casted;
    realize.runtime_name = "casted";
    realize.dtype = DType::BF16;
    const auto r = add_expr(std::move(realize));
    plan.algebra.bindings.push_back({"casted", r});

    const auto result = optimize_layout_algebra(plan);
    CHECK(!result.passes.empty());
    CHECK(result.passes.front().rewrites >= std::size_t{1});

    bool saw_join_after_pushdown = false;
    bool saw_child_cast = false;
    for (const auto& expr : plan.algebra.exprs) {
        if (expr.kind == LayoutExprKind::Join &&
            expr.decl.dtype == DType::BF16 &&
            expr.decl.shape == std::vector<std::int64_t>{4, 2}) {
            saw_join_after_pushdown = true;
        }
        if (expr.kind == LayoutExprKind::Cast &&
            expr.decl.dtype == DType::BF16 &&
            expr.decl.shape == std::vector<std::int64_t>{2, 2}) {
            saw_child_cast = true;
        }
    }
    CHECK(saw_join_after_pushdown);
    CHECK(saw_child_cast);
}

void test_layout_optimizer_covers_proposal_rewrites() {
    using namespace pie_cuda_driver;

    auto add_expr = [](LayoutPlan& plan, LayoutExpr expr) {
        const LayoutExprId id = plan.algebra.exprs.size();
        plan.algebra.exprs.push_back(std::move(expr));
        return id;
    };
    auto source = [&](LayoutPlan& plan,
                      std::string raw,
                      std::string name,
                      DType dtype,
                      std::vector<std::int64_t> shape) {
        TensorDecl decl{
            .name = std::move(name),
            .dtype = dtype,
            .shape = std::move(shape),
        };
        plan.tensors.emplace(decl.name, decl);
        LayoutExpr expr;
        expr.kind = LayoutExprKind::Source;
        expr.decl = decl;
        expr.raw_name = std::move(raw);
        expr.dtype = dtype;
        return add_expr(plan, std::move(expr));
    };
    auto realize = [&](LayoutPlan& plan,
                       LayoutExprId input,
                       const TensorDecl& decl) {
        LayoutExpr expr;
        expr.kind = LayoutExprKind::Realize;
        expr.inputs = {input};
        expr.decl = decl;
        expr.runtime_name = decl.name;
        expr.dtype = decl.dtype;
        const auto id = add_expr(plan, std::move(expr));
        plan.algebra.bindings.push_back({decl.name, id});
        plan.tensors.emplace(decl.name, decl);
        return id;
    };

    {
        LayoutPlan plan;
        const auto a = source(plan, "a", "a.tmp", DType::BF16, {4, 2});
        const auto b = source(plan, "b", "b.tmp", DType::BF16, {4, 2});
        TensorDecl joined{.name = "joined", .dtype = DType::BF16, .shape = {8, 2}};
        LayoutExpr join;
        join.kind = LayoutExprKind::Join;
        join.inputs = {a, b};
        join.decl = joined;
        join.axis = 0;
        join.dtype = DType::BF16;
        const auto j = add_expr(plan, std::move(join));
        TensorDecl out{.name = "rank1", .dtype = DType::BF16, .shape = {4, 2}};
        LayoutExpr partition;
        partition.kind = LayoutExprKind::Partition;
        partition.inputs = {j};
        partition.decl = out;
        partition.axis = 0;
        partition.partitions = 2;
        partition.partition_index = 1;
        partition.dtype = DType::BF16;
        const auto p = add_expr(plan, std::move(partition));
        realize(plan, p, out);
        const auto result = optimize_layout_algebra(plan);
        CHECK(result.passes.front().rewrites >= std::size_t{1});
        CHECK_EQ(count_exprs(plan, LayoutExprKind::Partition), std::size_t{0});
    }

    {
        LayoutPlan plan;
        const auto q = source(plan, "q", "q.tmp", DType::FP8_E4M3, {8, 2});
        const auto scale = source(plan, "q.scale", "q.scale.tmp", DType::FP32, {8});
        TensorDecl decoded{.name = "decoded", .dtype = DType::BF16, .shape = {8, 2}};
        LayoutExpr decode;
        decode.kind = LayoutExprKind::Decode;
        decode.inputs = {q, scale};
        decode.decl = decoded;
        decode.dtype = DType::BF16;
        const auto d = add_expr(plan, std::move(decode));
        TensorDecl out{.name = "decoded.slice", .dtype = DType::BF16, .shape = {4, 2}};
        LayoutExpr select;
        select.kind = LayoutExprKind::Select;
        select.inputs = {d};
        select.decl = out;
        select.axis = 0;
        select.start = 2;
        select.length = 4;
        select.dtype = DType::BF16;
        const auto s = add_expr(plan, std::move(select));
        realize(plan, s, out);
        (void)optimize_layout_algebra(plan);
        CHECK(std::any_of(
            plan.algebra.exprs.begin(), plan.algebra.exprs.end(),
            [](const auto& expr) {
                return expr.kind == LayoutExprKind::Decode &&
                       expr.decl.shape == std::vector<std::int64_t>{4, 2};
            }));
    }

    {
        LayoutPlan plan;
        const auto f = source(plan, "f", "f.tmp", DType::BF16, {8, 2});
        TensorDecl selected{.name = "selected", .dtype = DType::BF16, .shape = {4, 2}};
        LayoutExpr select;
        select.kind = LayoutExprKind::Select;
        select.inputs = {f};
        select.decl = selected;
        select.axis = 0;
        select.start = 2;
        select.length = 4;
        select.dtype = DType::BF16;
        const auto s = add_expr(plan, std::move(select));
        TensorDecl encoded{.name = "encoded", .dtype = DType::INT8, .shape = {4, 2}};
        encoded.quant.format = QuantFormat::RuntimeInt8;
        encoded.quant.scale_tensor = "encoded.scale";
        plan.tensors.emplace(
            "encoded.scale",
            TensorDecl{.name = "encoded.scale", .dtype = DType::FP32, .shape = {1}});
        LayoutExpr encode;
        encode.kind = LayoutExprKind::Encode;
        encode.inputs = {s};
        encode.decl = encoded;
        encode.runtime_name = encoded.name;
        encode.dtype = DType::INT8;
        encode.encoding = encoded.quant;
        const auto e = add_expr(plan, std::move(encode));
        realize(plan, e, encoded);
        (void)optimize_layout_algebra(plan);
        CHECK(std::any_of(
            plan.algebra.exprs.begin(), plan.algebra.exprs.end(),
            [](const auto& expr) {
                return expr.kind == LayoutExprKind::Encode &&
                       expr.decl.shape == std::vector<std::int64_t>{8, 2};
            }));
    }

    {
        LayoutPlan plan;
        const auto q = source(plan, "raw.q", "q.tmp", DType::FP8_E4M3, {8, 2});
        const auto scale = source(
            plan, "raw.q.scale", "q.scale.tmp", DType::FP32, {8});
        TensorDecl decoded{.name = "decoded", .dtype = DType::BF16, .shape = {8, 2}};
        LayoutExpr decode;
        decode.kind = LayoutExprKind::Decode;
        decode.inputs = {q, scale};
        decode.decl = decoded;
        decode.dtype = DType::BF16;
        const auto d = add_expr(plan, std::move(decode));
        TensorDecl encoded{.name = "runtime.q", .dtype = DType::INT8, .shape = {8, 2}};
        encoded.quant.format = QuantFormat::RuntimeInt8;
        encoded.quant.scale_tensor = "runtime.q.scale";
        plan.tensors.emplace(
            "runtime.q.scale",
            TensorDecl{.name = "runtime.q.scale", .dtype = DType::FP32, .shape = {8}});
        LayoutExpr encode;
        encode.kind = LayoutExprKind::Encode;
        encode.inputs = {d};
        encode.decl = encoded;
        encode.runtime_name = encoded.name;
        encode.dtype = DType::INT8;
        encode.encoding = encoded.quant;
        const auto e = add_expr(plan, std::move(encode));
        realize(plan, e, encoded);
        (void)optimize_layout_algebra(plan);
        CHECK_EQ(count_exprs(plan, LayoutExprKind::Transcode), std::size_t{1});

        FakeMetadata meta;
        meta.add("raw.q", DType::FP8_E4M3, {8, 2});
        meta.add("raw.q.scale", DType::FP32, {8});
        const auto storage = compile_storage_program(
            plan, meta, /*tp_rank=*/0, /*tp_size=*/1);
        CHECK_EQ(count_tile_maps(storage, TileMapKind::Transcode), std::size_t{1});
    }

    {
        LayoutPlan plan;
        const auto a = source(plan, "a", "a.stack", DType::FP16, {2, 2});
        const auto b = source(plan, "b", "b.stack", DType::FP16, {2, 2});
        TensorDecl stacked{.name = "stacked", .dtype = DType::FP16, .shape = {2, 2}};
        LayoutExpr stack;
        stack.kind = LayoutExprKind::Stack;
        stack.inputs = {a, b};
        stack.decl = stacked;
        stack.axis = 0;
        stack.dtype = DType::FP16;
        const auto st = add_expr(plan, std::move(stack));
        TensorDecl casted{.name = "stacked.bf16", .dtype = DType::BF16, .shape = {2, 2}};
        LayoutExpr cast;
        cast.kind = LayoutExprKind::Cast;
        cast.inputs = {st};
        cast.decl = casted;
        cast.dtype = DType::BF16;
        const auto c = add_expr(plan, std::move(cast));
        realize(plan, c, casted);
        (void)optimize_layout_algebra(plan);
        CHECK(std::any_of(
            plan.algebra.exprs.begin(), plan.algebra.exprs.end(),
            [](const auto& expr) {
                return expr.kind == LayoutExprKind::Stack &&
                       expr.decl.dtype == DType::BF16;
            }));
    }
}

void test_dense_schema_adapters_pack_mistral_olmo_variants() {
    const auto meta = dense_llama_metadata();
    pie_cuda_driver::Config cfg{};
    const pie_cuda_driver::BackendTarget target{};
    for (const std::string model_type :
         {"mistral3", "ministral3", "olmo3"}) {
        auto hf = qwen3_config();
        hf.model_type = model_type;
        auto plan = pie_cuda_driver::build_model_layout_plan(
            hf, cfg, meta, /*tp_size=*/1, target);
        pie_cuda_driver::validate_layout_plan(plan);
        CHECK_EQ(plan.axis_concat_groups, std::size_t{2});
        CHECK(!plan.algebra.bindings.empty());
    }
}

void test_qwen36_dense_linear_attention_plan() {
    using pie_cuda_driver::DType;
    auto hf = qwen3_config();
    hf.model_type = "qwen3_5";
    FakeMetadata meta = dense_llama_metadata();
    meta.add("model.layers.0.linear_attn.in_proj_z.weight", DType::BF16, {8, 8});
    meta.add("model.layers.0.linear_attn.in_proj_b.weight", DType::BF16, {8, 8});
    meta.add("model.layers.0.linear_attn.in_proj_a.weight", DType::BF16, {8, 8});
    meta.add("model.layers.0.linear_attn.out_proj.weight", DType::BF16, {8, 8});
    meta.add("model.layers.0.linear_attn.dt_bias", DType::BF16, {8});
    meta.add("model.layers.0.linear_attn.A_log", DType::BF16, {8});

    pie_cuda_driver::Config cfg{};
    const pie_cuda_driver::BackendTarget target{};
    auto plan = pie_cuda_driver::build_model_layout_plan(
        hf, cfg, meta, /*tp_size=*/2, target);
    pie_cuda_driver::validate_layout_plan(plan);

    const auto* z = find_spec(
        plan, "model.layers.0.linear_attn.in_proj_z.weight");
    CHECK(z != nullptr);
    if (z != nullptr) {
        CHECK((z->shape == std::vector<std::int64_t>{4, 8}));
        CHECK(z->parallel == pie_cuda_driver::TensorParallelKind::Column);
    }
    const auto* out = find_spec(
        plan, "model.layers.0.linear_attn.out_proj.weight");
    CHECK(out != nullptr);
    if (out != nullptr) {
        CHECK((out->shape == std::vector<std::int64_t>{8, 4}));
        CHECK(out->parallel == pie_cuda_driver::TensorParallelKind::Row);
    }
}

void test_dump_plan_has_no_post_load_transform_channel() {
    auto hf = qwen3_config();
    pie_cuda_driver::Config cfg{};
    const auto meta = dense_llama_metadata();
    const pie_cuda_driver::BackendTarget target{};
    auto plan = pie_cuda_driver::build_model_layout_plan(
        hf, cfg, meta, /*tp_size=*/1, target);
    const std::string json = pie_cuda_driver::dump_layout_plan_json(plan);
    CHECK(json.find("\"transforms\"") == std::string::npos);
    CHECK(json.find("transform ops") == std::string::npos);
    CHECK(!plan.algebra.exprs.empty());
    CHECK(!plan.algebra.bindings.empty());
    CHECK(json.find("\"algebra\"") != std::string::npos);
    CHECK(json.find("\"expr_kind\":\"LayoutExpr\"") != std::string::npos);
}

void test_runtime_int8_lowers_to_scheduled_quant_ops() {
    auto hf = qwen3_config();
    pie_cuda_driver::Config cfg{};
    cfg.model.runtime_quant = "int8";
    const auto meta = dense_llama_metadata();
    const pie_cuda_driver::BackendTarget target{};
    auto plan = pie_cuda_driver::build_model_layout_plan(
        hf, cfg, meta, /*tp_size=*/1, target);
    pie_cuda_driver::validate_layout_plan(plan);

    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Encode),
             std::size_t{7});
    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Attach),
             std::size_t{7});
    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Release),
             std::size_t{7});
    const auto* q = find_spec(plan, "model.layers.0.self_attn.q_proj.weight");
    CHECK(q != nullptr);
    if (q != nullptr) {
        CHECK(q->dtype == pie_cuda_driver::DType::INT8);
        CHECK(q->layout == pie_cuda_driver::TensorLayoutKind::QuantPacked);
        CHECK(q->quant.format == pie_cuda_driver::QuantFormat::RuntimeInt8);
        CHECK_EQ(q->quant.scale_tensor,
                 std::string("model.layers.0.self_attn.q_proj.weight_scale_inv"));
    }
}

void test_compressed_tensors_int8_lowers_to_quant_packed() {
    using pie_cuda_driver::DType;
    auto hf = qwen3_config();
    hf.quant_method = "compressed-tensors";
    pie_cuda_driver::Config cfg{};
    FakeMetadata meta;
    meta.add("model.layers.0.self_attn.q_proj.weight", DType::INT8, {8, 8});
    meta.add("model.layers.0.self_attn.q_proj.weight_scale", DType::FP32, {8});

    const pie_cuda_driver::BackendTarget target{};
    auto plan = pie_cuda_driver::build_model_layout_plan(
        hf, cfg, meta, /*tp_size=*/1, target);
    pie_cuda_driver::validate_layout_plan(plan);

    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Attach),
             std::size_t{1});
    const auto* q = find_spec(
        plan, "model.layers.0.self_attn.q_proj.weight");
    CHECK(q != nullptr);
    if (q != nullptr) {
        CHECK(q->dtype == pie_cuda_driver::DType::INT8);
        CHECK(q->layout == pie_cuda_driver::TensorLayoutKind::QuantPacked);
        CHECK(q->quant.format == pie_cuda_driver::QuantFormat::CompressedInt8);
        CHECK(q->quant.granularity ==
              pie_cuda_driver::QuantGranularity::PerChannel);
        CHECK_EQ(q->quant.scale_tensor,
                 std::string("model.layers.0.self_attn.q_proj.weight_scale_inv"));
    }
}

void test_fp16_copy_lowers_to_scheduled_cast() {
    using pie_cuda_driver::DType;
    auto hf = qwen3_config();
    pie_cuda_driver::Config cfg{};
    FakeMetadata meta;
    meta.add("model.norm.weight", DType::FP16, {8});

    const pie_cuda_driver::BackendTarget target{};
    auto plan = pie_cuda_driver::build_model_layout_plan(
        hf, cfg, meta, /*tp_size=*/1, target);
    pie_cuda_driver::validate_layout_plan(plan);

    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Source),
             std::size_t{1});
    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Cast),
             std::size_t{1});
    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Release),
             std::size_t{1});
    const auto* source =
        find_spec(plan, "model.norm.weight.__dtype_source");
    CHECK(source != nullptr);
    if (source != nullptr) {
        CHECK(source->dtype == DType::FP16);
        CHECK(source->ownership == pie_cuda_driver::TensorOwnershipKind::Temporary);
    }
    const auto* final = find_spec(plan, "model.norm.weight");
    CHECK(final != nullptr);
    if (final != nullptr) {
        CHECK(final->dtype == DType::BF16);
        CHECK(final->ownership == pie_cuda_driver::TensorOwnershipKind::Owned);
    }
    const auto storage = pie_cuda_driver::compile_storage_program(
        plan, meta, /*tp_rank=*/0, /*tp_size=*/1);
    CHECK_EQ(storage.tile_maps.size(), std::size_t{1});
    if (!storage.tile_maps.empty()) {
        CHECK(storage.tile_maps[0].kind ==
              pie_cuda_driver::TileMapKind::Cast);
        CHECK_EQ(storage.tile_maps[0].output_name,
                 std::string("model.norm.weight"));
    }
}

void test_phi3_tp_uses_row_range_shards_for_fused_tensors() {
    using pie_cuda_driver::DType;
    pie_cuda_driver::HfConfig hf{};
    hf.model_type = "phi3";
    hf.num_attention_heads = 4;
    hf.num_key_value_heads = 4;
    hf.head_dim = 2;
    hf.intermediate_size = 8;

    FakeMetadata meta;
    meta.add("model.layers.0.self_attn.qkv_proj.weight", DType::BF16, {24, 8});
    meta.add("model.layers.0.mlp.gate_up_proj.weight", DType::BF16, {16, 8});
    pie_cuda_driver::Config cfg{};
    pie_cuda_driver::BackendTarget target{};
    target.gptq_marlin_int4 = false;
    auto plan = pie_cuda_driver::build_model_layout_plan(
        hf, cfg, meta, /*tp_size=*/2, target);
    pie_cuda_driver::validate_layout_plan(plan);

    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Select),
             std::size_t{5});
    CHECK(find_spec(plan, "model.layers.0.self_attn.qkv_proj.weight") == nullptr);
    CHECK(find_spec(plan, "model.layers.0.mlp.gate_up_proj.weight") == nullptr);

    const auto* q = find_spec(plan, "model.layers.0.self_attn.q_proj.weight");
    CHECK(q != nullptr);
    if (q != nullptr) {
        CHECK((q->shape == std::vector<std::int64_t>{4, 8}));
        CHECK(q->parallel == pie_cuda_driver::TensorParallelKind::Column);
    }
    const auto* gate = find_spec(plan, "model.layers.0.mlp.gate_proj.weight");
    CHECK(gate != nullptr);
    if (gate != nullptr) {
        CHECK((gate->shape == std::vector<std::int64_t>{4, 8}));
        CHECK(gate->parallel == pie_cuda_driver::TensorParallelKind::Column);
    }
}

void test_phi3_fp16_row_range_shards_cast_to_bf16() {
    using pie_cuda_driver::DType;
    pie_cuda_driver::HfConfig hf{};
    hf.model_type = "phi3";
    hf.num_attention_heads = 4;
    hf.num_key_value_heads = 4;
    hf.head_dim = 2;
    hf.intermediate_size = 8;

    FakeMetadata meta;
    meta.add("model.layers.0.self_attn.qkv_proj.weight", DType::FP16, {24, 8});
    pie_cuda_driver::Config cfg{};
    pie_cuda_driver::BackendTarget target{};
    target.gptq_marlin_int4 = false;
    auto plan = pie_cuda_driver::build_model_layout_plan(
        hf, cfg, meta, /*tp_size=*/2, target);
    pie_cuda_driver::validate_layout_plan(plan);

    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Select),
             std::size_t{3});
    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Cast),
             std::size_t{3});
    const auto* q =
        find_spec(plan, "model.layers.0.self_attn.q_proj.weight");
    CHECK(q != nullptr);
    if (q != nullptr) {
        CHECK(q->dtype == DType::BF16);
        CHECK((q->shape == std::vector<std::int64_t>{4, 8}));
    }
    const auto* q_tmp =
        find_spec(plan, "model.layers.0.self_attn.q_proj.weight.__dtype_source");
    CHECK(q_tmp != nullptr);
    if (q_tmp != nullptr) {
        CHECK(q_tmp->dtype == DType::FP16);
        CHECK(q_tmp->ownership == pie_cuda_driver::TensorOwnershipKind::Temporary);
    }
}

void test_gptq_symmetric_lowers_to_repack_layout() {
    using pie_cuda_driver::DType;
    pie_cuda_driver::HfConfig hf{};
    hf.model_type = "qwen3";
    hf.quant_method = "gptq";
    hf.quant_bits = 4;
    hf.quant_group_size = 16;
    hf.quant_sym = true;
    hf.quant_zero_point = false;
    hf.quant_desc_act = false;

    FakeMetadata meta;
    meta.add("model.layers.0.self_attn.q_proj.g_idx", DType::INT32, {32});
    meta.add("model.layers.0.self_attn.q_proj.qweight", DType::INT32, {4, 64});
    meta.add("model.layers.0.self_attn.q_proj.qzeros", DType::INT32, {2, 8});
    meta.add("model.layers.0.self_attn.q_proj.scales", DType::FP16, {2, 64});

    pie_cuda_driver::Config cfg{};
    pie_cuda_driver::BackendTarget target{};
    target.gptq_marlin_int4 = true;
    auto plan = pie_cuda_driver::build_model_layout_plan(
        hf, cfg, meta, /*tp_size=*/1, target);
    pie_cuda_driver::validate_layout_plan(plan);

    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Reorder),
             std::size_t{1});
    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Attach),
             std::size_t{1});
    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Release),
             std::size_t{1});
    CHECK(find_spec(plan, "model.layers.0.self_attn.q_proj.qweight") == nullptr);
    CHECK(find_spec(plan, "model.layers.0.self_attn.q_proj.qzeros") == nullptr);
    CHECK(find_spec(plan, "model.layers.0.self_attn.q_proj.g_idx") == nullptr);

    const auto* weight =
        find_spec(plan, "model.layers.0.self_attn.q_proj.weight");
    CHECK(weight != nullptr);
    if (weight != nullptr) {
        CHECK(weight->dtype == DType::INT4_PACKED);
        CHECK(weight->layout == pie_cuda_driver::TensorLayoutKind::QuantPacked);
        CHECK((weight->shape == std::vector<std::int64_t>{2, 512}));
        CHECK(weight->quant.format == pie_cuda_driver::QuantFormat::GptqInt4);
        CHECK(weight->quant.granularity ==
              pie_cuda_driver::QuantGranularity::PerGroup);
        CHECK_EQ(weight->quant.group_size, 16);
        CHECK_EQ(weight->quant.scale_tensor,
                 std::string("model.layers.0.self_attn.q_proj.weight_scale_inv"));
    }

    const auto* scale =
        find_spec(plan, "model.layers.0.self_attn.q_proj.weight_scale_inv");
    CHECK(scale != nullptr);
    if (scale != nullptr) {
        CHECK(scale->dtype == DType::BF16);
        CHECK((scale->shape == std::vector<std::int64_t>{2, 64}));
    }
}

void test_gptq_symmetric_tp_lowers_to_local_repack_layout() {
    using pie_cuda_driver::DType;
    pie_cuda_driver::HfConfig hf{};
    hf.model_type = "qwen3";
    hf.quant_method = "gptq";
    hf.quant_bits = 4;
    hf.quant_group_size = 16;
    hf.quant_sym = true;
    hf.quant_zero_point = false;
    hf.quant_desc_act = false;

    FakeMetadata meta;
    meta.add("model.layers.0.self_attn.q_proj.qweight", DType::INT32, {4, 64});
    meta.add("model.layers.0.self_attn.q_proj.scales", DType::FP16, {2, 64});
    meta.add("model.layers.0.self_attn.o_proj.qweight", DType::INT32, {4, 64});
    meta.add("model.layers.0.self_attn.o_proj.scales", DType::FP16, {2, 64});

    pie_cuda_driver::Config cfg{};
    pie_cuda_driver::BackendTarget target{};
    target.gptq_marlin_int4 = true;
    auto plan = pie_cuda_driver::build_model_layout_plan(
        hf, cfg, meta, /*tp_size=*/2, target);
    pie_cuda_driver::validate_layout_plan(plan);

    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Reorder),
             std::size_t{2});
    const auto* q_tmp =
        find_spec(plan, "model.layers.0.self_attn.q_proj.weight.__gptq_qweight");
    CHECK(q_tmp != nullptr);
    if (q_tmp != nullptr) {
        CHECK((q_tmp->shape == std::vector<std::int64_t>{4, 32}));
        CHECK(q_tmp->parallel == pie_cuda_driver::TensorParallelKind::Column);
    }
    const auto* q_scale =
        find_spec(plan, "model.layers.0.self_attn.q_proj.weight_scale_inv");
    CHECK(q_scale != nullptr);
    if (q_scale != nullptr) {
        CHECK((q_scale->shape == std::vector<std::int64_t>{2, 32}));
        CHECK(q_scale->parallel == pie_cuda_driver::TensorParallelKind::Column);
    }
    const auto* q_weight =
        find_spec(plan, "model.layers.0.self_attn.q_proj.weight");
    CHECK(q_weight != nullptr);
    if (q_weight != nullptr) {
        CHECK((q_weight->shape == std::vector<std::int64_t>{2, 256}));
        CHECK(q_weight->parallel == pie_cuda_driver::TensorParallelKind::Column);
    }
    const auto* o_tmp =
        find_spec(plan, "model.layers.0.self_attn.o_proj.weight.__gptq_qweight");
    CHECK(o_tmp != nullptr);
    if (o_tmp != nullptr) {
        CHECK((o_tmp->shape == std::vector<std::int64_t>{2, 64}));
        CHECK(o_tmp->parallel == pie_cuda_driver::TensorParallelKind::Row);
    }
    const auto* o_scale =
        find_spec(plan, "model.layers.0.self_attn.o_proj.weight_scale_inv");
    CHECK(o_scale != nullptr);
    if (o_scale != nullptr) {
        CHECK((o_scale->shape == std::vector<std::int64_t>{1, 64}));
        CHECK(o_scale->parallel == pie_cuda_driver::TensorParallelKind::Row);
    }
    const auto* o_weight =
        find_spec(plan, "model.layers.0.self_attn.o_proj.weight");
    CHECK(o_weight != nullptr);
    if (o_weight != nullptr) {
        CHECK((o_weight->shape == std::vector<std::int64_t>{1, 512}));
        CHECK(o_weight->parallel == pie_cuda_driver::TensorParallelKind::Row);
    }
}

void test_gptq_symmetric_without_marlin_lowers_to_dequant() {
    using pie_cuda_driver::DType;
    pie_cuda_driver::HfConfig hf{};
    hf.model_type = "qwen3";
    hf.quant_method = "gptq";
    hf.quant_bits = 4;
    hf.quant_group_size = 16;
    hf.quant_sym = true;
    hf.quant_zero_point = false;
    hf.quant_desc_act = false;

    FakeMetadata meta;
    meta.add("model.layers.0.self_attn.q_proj.qweight", DType::INT32, {4, 64});
    meta.add("model.layers.0.self_attn.q_proj.qzeros", DType::INT32, {2, 8});
    meta.add("model.layers.0.self_attn.q_proj.scales", DType::FP16, {2, 64});

    pie_cuda_driver::Config cfg{};
    pie_cuda_driver::BackendTarget target{};
    target.gptq_marlin_int4 = false;
    auto plan = pie_cuda_driver::build_model_layout_plan(
        hf, cfg, meta, /*tp_size=*/1, target);
    pie_cuda_driver::validate_layout_plan(plan);

    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Reorder),
             std::size_t{0});
    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Decode),
             std::size_t{1});
    const auto* weight =
        find_spec(plan, "model.layers.0.self_attn.q_proj.weight");
    CHECK(weight != nullptr);
    if (weight != nullptr) {
        CHECK(weight->dtype == DType::BF16);
        CHECK((weight->shape == std::vector<std::int64_t>{64, 32}));
    }
}

void test_awq_lowers_to_scheduled_dequant() {
    using pie_cuda_driver::DType;
    pie_cuda_driver::HfConfig hf{};
    hf.model_type = "qwen3";
    hf.quant_method = "awq";
    hf.quant_bits = 4;
    hf.quant_group_size = 16;

    FakeMetadata meta;
    meta.add("model.layers.0.self_attn.q_proj.qweight", DType::INT32, {32, 8});
    meta.add("model.layers.0.self_attn.q_proj.qzeros", DType::INT32, {2, 8});
    meta.add("model.layers.0.self_attn.q_proj.scales", DType::FP16, {2, 64});

    pie_cuda_driver::Config cfg{};
    pie_cuda_driver::BackendTarget target{};
    target.gptq_marlin_int4 = false;
    auto plan = pie_cuda_driver::build_model_layout_plan(
        hf, cfg, meta, /*tp_size=*/1, target);
    pie_cuda_driver::validate_layout_plan(plan);

    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Decode),
             std::size_t{1});
    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Release),
             std::size_t{1});
    CHECK(find_spec(plan, "model.layers.0.self_attn.q_proj.qweight") == nullptr);
    CHECK(find_spec(plan, "model.layers.0.self_attn.q_proj.qzeros") == nullptr);

    const auto* tmp = find_spec(
        plan, "model.layers.0.self_attn.q_proj.weight.__awq_qweight");
    CHECK(tmp != nullptr);
    if (tmp != nullptr) {
        CHECK(tmp->ownership == pie_cuda_driver::TensorOwnershipKind::Temporary);
        CHECK(tmp->quant.format == pie_cuda_driver::QuantFormat::AwqInt4);
        CHECK_EQ(tmp->quant.group_size, 16);
    }
    const auto* weight =
        find_spec(plan, "model.layers.0.self_attn.q_proj.weight");
    CHECK(weight != nullptr);
    if (weight != nullptr) {
        CHECK(weight->dtype == DType::BF16);
        CHECK(weight->layout == pie_cuda_driver::TensorLayoutKind::Dense);
        CHECK((weight->shape == std::vector<std::int64_t>{64, 32}));
    }
}

void test_awq_lowers_to_marlin_repack_when_target_supports_int4() {
    using pie_cuda_driver::DType;
    pie_cuda_driver::HfConfig hf{};
    hf.model_type = "qwen3";
    hf.quant_method = "awq";
    hf.quant_bits = 4;
    hf.quant_group_size = 16;

    FakeMetadata meta;
    meta.add("model.layers.0.self_attn.q_proj.qweight", DType::INT32, {32, 8});
    meta.add("model.layers.0.self_attn.q_proj.qzeros", DType::INT32, {2, 8});
    meta.add("model.layers.0.self_attn.q_proj.scales", DType::FP16, {2, 64});

    pie_cuda_driver::Config cfg{};
    pie_cuda_driver::BackendTarget target{};
    target.gptq_marlin_int4 = true;
    auto plan = pie_cuda_driver::build_model_layout_plan(
        hf, cfg, meta, /*tp_size=*/1, target);
    pie_cuda_driver::validate_layout_plan(plan);

    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Reorder),
             std::size_t{1});
    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Decode),
             std::size_t{0});
    const auto* weight =
        find_spec(plan, "model.layers.0.self_attn.q_proj.weight");
    CHECK(weight != nullptr);
    if (weight != nullptr) {
        CHECK(weight->dtype == DType::INT4_PACKED);
        CHECK(weight->layout == pie_cuda_driver::TensorLayoutKind::QuantPacked);
        CHECK(weight->quant.format == pie_cuda_driver::QuantFormat::AwqInt4);
        CHECK_EQ(weight->quant.zero_point_tensor,
                 std::string("model.layers.0.self_attn.q_proj.weight_zero_point"));
    }
    const auto* zero =
        find_spec(plan, "model.layers.0.self_attn.q_proj.weight_zero_point");
    CHECK(zero != nullptr);
    if (zero != nullptr) {
        CHECK(zero->dtype == DType::INT32);
        CHECK((zero->shape == std::vector<std::int64_t>{2, 8}));
    }
}

void test_awq_tp_lowers_to_local_dequant() {
    using pie_cuda_driver::DType;
    pie_cuda_driver::HfConfig hf{};
    hf.model_type = "qwen3";
    hf.quant_method = "awq";
    hf.quant_bits = 4;
    hf.quant_group_size = 16;

    FakeMetadata meta;
    meta.add("model.layers.0.self_attn.q_proj.qweight", DType::INT32, {32, 8});
    meta.add("model.layers.0.self_attn.q_proj.qzeros", DType::INT32, {2, 8});
    meta.add("model.layers.0.self_attn.q_proj.scales", DType::FP16, {2, 64});
    meta.add("model.layers.0.self_attn.o_proj.qweight", DType::INT32, {32, 8});
    meta.add("model.layers.0.self_attn.o_proj.qzeros", DType::INT32, {2, 8});
    meta.add("model.layers.0.self_attn.o_proj.scales", DType::FP16, {2, 64});

    pie_cuda_driver::Config cfg{};
    pie_cuda_driver::BackendTarget target{};
    target.gptq_marlin_int4 = false;
    auto plan = pie_cuda_driver::build_model_layout_plan(
        hf, cfg, meta, /*tp_size=*/2, target);
    pie_cuda_driver::validate_layout_plan(plan);

    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Decode),
             std::size_t{2});
    const auto* q_tmp = find_spec(
        plan, "model.layers.0.self_attn.q_proj.weight.__awq_qweight");
    CHECK(q_tmp != nullptr);
    if (q_tmp != nullptr) {
        CHECK((q_tmp->shape == std::vector<std::int64_t>{32, 4}));
        CHECK(q_tmp->parallel == pie_cuda_driver::TensorParallelKind::Column);
    }
    const auto* q_zero = find_spec(
        plan, "model.layers.0.self_attn.q_proj.weight.__awq_qzeros");
    CHECK(q_zero != nullptr);
    if (q_zero != nullptr) {
        CHECK((q_zero->shape == std::vector<std::int64_t>{2, 4}));
    }
    const auto* q_weight =
        find_spec(plan, "model.layers.0.self_attn.q_proj.weight");
    CHECK(q_weight != nullptr);
    if (q_weight != nullptr) {
        CHECK((q_weight->shape == std::vector<std::int64_t>{32, 32}));
        CHECK(q_weight->parallel == pie_cuda_driver::TensorParallelKind::Column);
    }
    const auto* o_tmp = find_spec(
        plan, "model.layers.0.self_attn.o_proj.weight.__awq_qweight");
    CHECK(o_tmp != nullptr);
    if (o_tmp != nullptr) {
        CHECK((o_tmp->shape == std::vector<std::int64_t>{16, 8}));
        CHECK(o_tmp->parallel == pie_cuda_driver::TensorParallelKind::Row);
    }
    const auto* o_zero = find_spec(
        plan, "model.layers.0.self_attn.o_proj.weight.__awq_qzeros");
    CHECK(o_zero != nullptr);
    if (o_zero != nullptr) {
        CHECK((o_zero->shape == std::vector<std::int64_t>{1, 8}));
    }
    const auto* o_weight =
        find_spec(plan, "model.layers.0.self_attn.o_proj.weight");
    CHECK(o_weight != nullptr);
    if (o_weight != nullptr) {
        CHECK((o_weight->shape == std::vector<std::int64_t>{64, 16}));
        CHECK(o_weight->parallel == pie_cuda_driver::TensorParallelKind::Row);
    }
}

void test_gptq_asymmetric_tp_lowers_to_local_dequant() {
    using pie_cuda_driver::DType;
    pie_cuda_driver::HfConfig hf{};
    hf.model_type = "qwen3";
    hf.quant_method = "gptq";
    hf.quant_bits = 4;
    hf.quant_group_size = 16;
    hf.quant_sym = false;
    hf.quant_zero_point = true;
    hf.quant_desc_act = false;

    FakeMetadata meta;
    meta.add("model.layers.0.self_attn.q_proj.qweight", DType::INT32, {4, 64});
    meta.add("model.layers.0.self_attn.q_proj.qzeros", DType::INT32, {2, 8});
    meta.add("model.layers.0.self_attn.q_proj.scales", DType::FP16, {2, 64});
    meta.add("model.layers.0.self_attn.o_proj.qweight", DType::INT32, {4, 64});
    meta.add("model.layers.0.self_attn.o_proj.qzeros", DType::INT32, {2, 8});
    meta.add("model.layers.0.self_attn.o_proj.scales", DType::FP16, {2, 64});

    pie_cuda_driver::Config cfg{};
    const pie_cuda_driver::BackendTarget target{};
    auto plan = pie_cuda_driver::build_model_layout_plan(
        hf, cfg, meta, /*tp_size=*/2, target);
    pie_cuda_driver::validate_layout_plan(plan);

    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Decode),
             std::size_t{2});
    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Reorder),
             std::size_t{0});
    const auto* q_tmp = find_spec(
        plan, "model.layers.0.self_attn.q_proj.weight.__gptq_qweight");
    CHECK(q_tmp != nullptr);
    if (q_tmp != nullptr) {
        CHECK((q_tmp->shape == std::vector<std::int64_t>{4, 32}));
        CHECK(q_tmp->quant.format == pie_cuda_driver::QuantFormat::GptqInt4);
        CHECK(q_tmp->parallel == pie_cuda_driver::TensorParallelKind::Column);
    }
    const auto* q_zero = find_spec(
        plan, "model.layers.0.self_attn.q_proj.weight.__gptq_qzeros");
    CHECK(q_zero != nullptr);
    if (q_zero != nullptr) {
        CHECK((q_zero->shape == std::vector<std::int64_t>{2, 4}));
    }
    const auto* q_weight =
        find_spec(plan, "model.layers.0.self_attn.q_proj.weight");
    CHECK(q_weight != nullptr);
    if (q_weight != nullptr) {
        CHECK((q_weight->shape == std::vector<std::int64_t>{32, 32}));
        CHECK(q_weight->parallel == pie_cuda_driver::TensorParallelKind::Column);
    }

    const auto* o_tmp = find_spec(
        plan, "model.layers.0.self_attn.o_proj.weight.__gptq_qweight");
    CHECK(o_tmp != nullptr);
    if (o_tmp != nullptr) {
        CHECK((o_tmp->shape == std::vector<std::int64_t>{2, 64}));
        CHECK(o_tmp->parallel == pie_cuda_driver::TensorParallelKind::Row);
    }
    const auto* o_zero = find_spec(
        plan, "model.layers.0.self_attn.o_proj.weight.__gptq_qzeros");
    CHECK(o_zero != nullptr);
    if (o_zero != nullptr) {
        CHECK((o_zero->shape == std::vector<std::int64_t>{1, 8}));
    }
    const auto* o_weight =
        find_spec(plan, "model.layers.0.self_attn.o_proj.weight");
    CHECK(o_weight != nullptr);
    if (o_weight != nullptr) {
        CHECK((o_weight->shape == std::vector<std::int64_t>{64, 16}));
        CHECK(o_weight->parallel == pie_cuda_driver::TensorParallelKind::Row);
    }
}

void test_gptq_desc_act_row_tp_keeps_full_group_metadata() {
    using pie_cuda_driver::DType;
    pie_cuda_driver::HfConfig hf{};
    hf.model_type = "qwen3";
    hf.quant_method = "gptq";
    hf.quant_bits = 4;
    hf.quant_group_size = 16;
    hf.quant_sym = false;
    hf.quant_zero_point = true;
    hf.quant_desc_act = true;

    FakeMetadata meta;
    meta.add("model.layers.0.self_attn.o_proj.g_idx", DType::INT32, {32});
    meta.add("model.layers.0.self_attn.o_proj.qweight", DType::INT32, {4, 64});
    meta.add("model.layers.0.self_attn.o_proj.qzeros", DType::INT32, {2, 8});
    meta.add("model.layers.0.self_attn.o_proj.scales", DType::FP16, {2, 64});

    pie_cuda_driver::Config cfg{};
    const pie_cuda_driver::BackendTarget target{};
    auto plan = pie_cuda_driver::build_model_layout_plan(
        hf, cfg, meta, /*tp_size=*/2, target);
    pie_cuda_driver::validate_layout_plan(plan);

    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Decode),
             std::size_t{1});
    const auto* qw = find_spec(
        plan, "model.layers.0.self_attn.o_proj.weight.__gptq_qweight");
    CHECK(qw != nullptr);
    if (qw != nullptr) {
        CHECK((qw->shape == std::vector<std::int64_t>{2, 64}));
        CHECK(qw->parallel == pie_cuda_driver::TensorParallelKind::Row);
    }
    const auto* qz = find_spec(
        plan, "model.layers.0.self_attn.o_proj.weight.__gptq_qzeros");
    CHECK(qz != nullptr);
    if (qz != nullptr) {
        CHECK((qz->shape == std::vector<std::int64_t>{2, 8}));
    }
    const auto* sc = find_spec(
        plan, "model.layers.0.self_attn.o_proj.weight.__gptq_scales");
    CHECK(sc != nullptr);
    if (sc != nullptr) {
        CHECK((sc->shape == std::vector<std::int64_t>{2, 64}));
    }
    const auto* gi = find_spec(
        plan, "model.layers.0.self_attn.o_proj.weight.__gptq_g_idx");
    CHECK(gi != nullptr);
    if (gi != nullptr) {
        CHECK((gi->shape == std::vector<std::int64_t>{16}));
    }
    const auto* dequant = find_expr(
        plan, pie_cuda_driver::LayoutExprKind::Decode,
        "model.layers.0.self_attn.o_proj.weight");
    CHECK(dequant != nullptr);
    if (dequant != nullptr) {
        CHECK_EQ(dequant->inputs.size(), std::size_t{4});
    }
}

void test_qwen_moe_tp_uses_expert_shard_ops() {
    using pie_cuda_driver::DType;
    pie_cuda_driver::HfConfig hf{};
    hf.model_type = "qwen3_moe";

    FakeMetadata meta;
    meta.add("model.layers.0.mlp.experts.gate_up_proj", DType::BF16, {2, 16, 8});
    meta.add("model.layers.0.mlp.experts.down_proj", DType::BF16, {2, 8, 16});

    pie_cuda_driver::Config cfg{};
    const pie_cuda_driver::BackendTarget target{};
    auto plan = pie_cuda_driver::build_model_layout_plan(
        hf, cfg, meta, /*tp_size=*/2, target);
    pie_cuda_driver::validate_layout_plan(plan);

    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Join),
             std::size_t{1});
    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Select),
             std::size_t{2});
    CHECK(find_spec(plan, "model.layers.0.mlp.experts.gate_up_proj") != nullptr);
    CHECK(find_spec(plan, "model.layers.0.mlp.experts.down_proj") != nullptr);

    const auto* gate_up =
        find_spec(plan, "model.layers.0.mlp.experts.gate_up_proj");
    CHECK(gate_up != nullptr);
    if (gate_up != nullptr) {
        CHECK((gate_up->shape == std::vector<std::int64_t>{2, 8, 8}));
        CHECK(gate_up->layout ==
              pie_cuda_driver::TensorLayoutKind::Grouped);
        CHECK(gate_up->parallel == pie_cuda_driver::TensorParallelKind::Expert);
    }
    const auto* down =
        find_spec(plan, "model.layers.0.mlp.experts.down_proj");
    CHECK(down != nullptr);
    if (down != nullptr) {
        CHECK((down->shape == std::vector<std::int64_t>{2, 8, 8}));
        CHECK(down->layout ==
              pie_cuda_driver::TensorLayoutKind::Grouped);
        CHECK(down->parallel == pie_cuda_driver::TensorParallelKind::Expert);
    }
}

void test_qwen_moe_per_expert_fusion_is_scheduled() {
    using pie_cuda_driver::DType;
    pie_cuda_driver::HfConfig hf{};
    hf.model_type = "qwen3_moe";
    hf.num_experts = 2;

    FakeMetadata meta;
    for (int e = 0; e < 2; ++e) {
        const std::string p =
            "model.layers.0.mlp.experts." + std::to_string(e) + ".";
        meta.add(p + "gate_proj.weight", DType::BF16, {16, 8});
        meta.add(p + "up_proj.weight", DType::BF16, {16, 8});
        meta.add(p + "down_proj.weight", DType::BF16, {8, 16});
    }

    pie_cuda_driver::Config cfg{};
    const pie_cuda_driver::BackendTarget target{};
    auto plan = pie_cuda_driver::build_model_layout_plan(
        hf, cfg, meta, /*tp_size=*/2, target);
    pie_cuda_driver::validate_layout_plan(plan);

    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Stack),
             std::size_t{1});
    const auto* gate_up =
        find_spec(plan, "model.layers.0.mlp.experts.gate_up_proj");
    CHECK(gate_up != nullptr);
    if (gate_up != nullptr) {
        CHECK((gate_up->shape == std::vector<std::int64_t>{2, 16, 8}));
        CHECK(gate_up->layout ==
              pie_cuda_driver::TensorLayoutKind::Grouped);
        CHECK(gate_up->ownership ==
              pie_cuda_driver::TensorOwnershipKind::Owned);
    }
    const auto* down =
        find_spec(plan, "model.layers.0.mlp.experts.down_proj");
    CHECK(down != nullptr);
    if (down != nullptr) {
        CHECK((down->shape == std::vector<std::int64_t>{2, 8, 8}));
        CHECK(down->layout ==
              pie_cuda_driver::TensorLayoutKind::Grouped);
    }
    CHECK(find_spec(
        plan, "model.layers.0.mlp.experts.0.gate_proj.weight") == nullptr);
}

void test_mixtral_schema_adapter_marks_and_shards_expert_weights() {
    using pie_cuda_driver::DType;
    pie_cuda_driver::HfConfig hf{};
    hf.model_type = "mixtral";

    FakeMetadata meta;
    meta.add("model.layers.0.block_sparse_moe.experts.0.w1.weight",
             DType::BF16, {16, 8});
    meta.add("model.layers.0.block_sparse_moe.experts.0.w3.weight",
             DType::BF16, {16, 8});
    meta.add("model.layers.0.block_sparse_moe.experts.0.w2.weight",
             DType::BF16, {8, 16});

    const auto graph = pie_cuda_driver::build_semantic_graph(hf, meta);
    CHECK(std::any_of(
        graph.tensors.begin(), graph.tensors.end(),
        [](const auto& t) {
            return t.runtime_name.ends_with(".w1.weight") &&
                   t.role == pie_cuda_driver::SemanticRole::MoeExpertGate;
        }));
    CHECK(std::any_of(
        graph.tensors.begin(), graph.tensors.end(),
        [](const auto& t) {
            return t.runtime_name.ends_with(".w3.weight") &&
                   t.role == pie_cuda_driver::SemanticRole::MoeExpertUp;
        }));

    pie_cuda_driver::Config cfg{};
    const pie_cuda_driver::BackendTarget target{};
    auto plan = pie_cuda_driver::build_model_layout_plan(
        hf, cfg, meta, /*tp_size=*/2, target);
    pie_cuda_driver::validate_layout_plan(plan);

    const auto* gate = find_spec(
        plan, "model.layers.0.block_sparse_moe.experts.0.w1.weight");
    CHECK(gate != nullptr);
    if (gate != nullptr) {
        CHECK((gate->shape == std::vector<std::int64_t>{8, 8}));
        CHECK(gate->parallel == pie_cuda_driver::TensorParallelKind::Column);
    }
    const auto* down = find_spec(
        plan, "model.layers.0.block_sparse_moe.experts.0.w2.weight");
    CHECK(down != nullptr);
    if (down != nullptr) {
        CHECK((down->shape == std::vector<std::int64_t>{8, 8}));
        CHECK(down->parallel == pie_cuda_driver::TensorParallelKind::Row);
    }
}

void test_gemma4_schema_adapter_declares_fused_expert_groups() {
    using pie_cuda_driver::DType;
    pie_cuda_driver::HfConfig hf{};
    hf.model_type = "gemma4";

    FakeMetadata meta;
    meta.add("model.layers.0.experts.gate_up_proj", DType::BF16, {2, 16, 8});
    meta.add("model.layers.0.experts.down_proj", DType::BF16, {2, 8, 16});

    const auto graph = pie_cuda_driver::build_semantic_graph(hf, meta);
    CHECK(has_group(
        graph, pie_cuda_driver::SemanticGroupKind::FusedMoeExperts));
}

void test_qwen36_moe_plan_shards_fused_experts() {
    using pie_cuda_driver::DType;
    pie_cuda_driver::HfConfig hf{};
    hf.model_type = "qwen3_5_moe";

    FakeMetadata meta;
    meta.add("model.layers.0.mlp.experts.gate_up_proj", DType::BF16, {2, 16, 8});
    meta.add("model.layers.0.mlp.experts.down_proj", DType::BF16, {2, 8, 8});

    pie_cuda_driver::Config cfg{};
    const pie_cuda_driver::BackendTarget target{};
    auto plan = pie_cuda_driver::build_model_layout_plan(
        hf, cfg, meta, /*tp_size=*/2, target);
    pie_cuda_driver::validate_layout_plan(plan);
    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Join),
             std::size_t{1});
    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Select),
             std::size_t{2});

    const auto* gate_up =
        find_spec(plan, "model.layers.0.mlp.experts.gate_up_proj");
    CHECK(gate_up != nullptr);
    if (gate_up != nullptr) {
        CHECK((gate_up->shape == std::vector<std::int64_t>{2, 8, 8}));
    }
    const auto* down =
        find_spec(plan, "model.layers.0.mlp.experts.down_proj");
    CHECK(down != nullptr);
    if (down != nullptr) {
        CHECK((down->shape == std::vector<std::int64_t>{2, 8, 4}));
    }
}

void test_gemma4_dense_and_moe_plan_coverage() {
    using pie_cuda_driver::DType;
    pie_cuda_driver::Config cfg{};
    const pie_cuda_driver::BackendTarget target{};

    auto dense_hf = qwen3_config();
    dense_hf.model_type = "gemma4";
    const auto dense_meta = dense_llama_metadata();
    auto dense_plan = pie_cuda_driver::build_model_layout_plan(
        dense_hf, cfg, dense_meta, /*tp_size=*/1, target);
    pie_cuda_driver::validate_layout_plan(dense_plan);
    CHECK(find_spec(dense_plan, "model.layers.0.self_attn.q_proj.weight") != nullptr);

    pie_cuda_driver::HfConfig moe_hf{};
    moe_hf.model_type = "gemma4";
    moe_hf.gemma4_enable_moe = true;
    FakeMetadata moe_meta;
    moe_meta.add("model.layers.0.experts.gate_up_proj", DType::BF16, {2, 16, 8});
    moe_meta.add("model.layers.0.experts.down_proj", DType::BF16, {2, 8, 16});
    auto moe_plan = pie_cuda_driver::build_model_layout_plan(
        moe_hf, cfg, moe_meta, /*tp_size=*/1, target);
    pie_cuda_driver::validate_layout_plan(moe_plan);
    CHECK(find_spec(moe_plan, "model.layers.0.experts.gate_up_proj") != nullptr);
    CHECK(find_spec(moe_plan, "model.layers.0.experts.down_proj") != nullptr);
}

FakeMetadata gpt_oss_mxfp4_metadata() {
    using pie_cuda_driver::DType;
    FakeMetadata meta;
    meta.add("model.layers.0.mlp.experts.gate_up_proj_blocks",
             DType::UINT8, {2, 8, 1, 16});
    meta.add("model.layers.0.mlp.experts.gate_up_proj_scales",
             DType::UINT8, {2, 8, 1});
    meta.add("model.layers.0.mlp.experts.gate_up_proj_bias",
             DType::BF16, {2, 8});
    meta.add("model.layers.0.mlp.experts.down_proj_blocks",
             DType::UINT8, {2, 8, 1, 16});
    meta.add("model.layers.0.mlp.experts.down_proj_scales",
             DType::UINT8, {2, 8, 1});
    meta.add("model.layers.0.mlp.experts.down_proj_bias",
             DType::BF16, {2, 8});
    return meta;
}

pie_cuda_driver::HfConfig gpt_oss_mxfp4_config() {
    pie_cuda_driver::HfConfig hf{};
    hf.model_type = "gpt_oss";
    hf.quant_method = "mxfp4";
    hf.hidden_size = 8;
    return hf;
}

void test_semantic_adapters_declare_group_roles() {
    {
        auto hf = qwen3_config();
        const auto meta = dense_llama_metadata();
        const auto graph = pie_cuda_driver::build_semantic_graph(hf, meta);
        const auto* qkv = find_group(
            graph, pie_cuda_driver::SemanticGroupKind::PackedQkv);
        CHECK(qkv != nullptr);
        if (qkv != nullptr) {
            CHECK((qkv->raw_roles == std::vector<pie_cuda_driver::SemanticRole>{
                pie_cuda_driver::SemanticRole::AttentionQ,
                pie_cuda_driver::SemanticRole::AttentionK,
                pie_cuda_driver::SemanticRole::AttentionV}));
            CHECK((qkv->runtime_roles == qkv->raw_roles));
        }
    }
    {
        pie_cuda_driver::HfConfig hf{};
        hf.model_type = "phi3";
        FakeMetadata meta;
        meta.add("model.layers.0.self_attn.qkv_proj.weight",
                 pie_cuda_driver::DType::BF16, {24, 8});
        const auto graph = pie_cuda_driver::build_semantic_graph(hf, meta);
        const auto* split = find_group(
            graph, pie_cuda_driver::SemanticGroupKind::RowRangeSplit);
        CHECK(split != nullptr);
        if (split != nullptr) {
            CHECK((split->raw_roles == std::vector<pie_cuda_driver::SemanticRole>{
                pie_cuda_driver::SemanticRole::AttentionQkv}));
            CHECK((split->runtime_roles == std::vector<pie_cuda_driver::SemanticRole>{
                pie_cuda_driver::SemanticRole::AttentionQ,
                pie_cuda_driver::SemanticRole::AttentionK,
                pie_cuda_driver::SemanticRole::AttentionV}));
        }
    }
    {
        auto hf = gpt_oss_mxfp4_config();
        const auto meta = gpt_oss_mxfp4_metadata();
        const auto graph = pie_cuda_driver::build_semantic_graph(hf, meta);
        const auto* mxfp4 = find_group(
            graph, pie_cuda_driver::SemanticGroupKind::GptOssMxfp4);
        CHECK(mxfp4 != nullptr);
        if (mxfp4 != nullptr) {
            CHECK((mxfp4->raw_roles == std::vector<pie_cuda_driver::SemanticRole>{
                pie_cuda_driver::SemanticRole::QuantPackedData,
                pie_cuda_driver::SemanticRole::QuantScale,
                pie_cuda_driver::SemanticRole::Bias}));
        }
    }
}

void test_gpt_oss_mxfp4_fallback_emits_dequantized_experts() {
    auto hf = gpt_oss_mxfp4_config();
    const auto meta = gpt_oss_mxfp4_metadata();
    pie_cuda_driver::Config cfg{};
    const pie_cuda_driver::BackendTarget target{};
    auto plan = pie_cuda_driver::build_model_layout_plan(
        hf, cfg, meta, /*tp_size=*/2, target);
    pie_cuda_driver::validate_layout_plan(plan);

    CHECK(!plan.algebra.bindings.empty());
    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Decode),
             std::size_t{2});
    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Unzip),
             std::size_t{2});
    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Select),
             std::size_t{1});
    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Release),
             std::size_t{2});

    const auto* gate =
        find_spec(plan, "model.layers.0.mlp.experts.gate_proj.weight");
    CHECK(gate != nullptr);
    if (gate != nullptr) {
        CHECK(gate->dtype == pie_cuda_driver::DType::BF16);
        CHECK((gate->shape == std::vector<std::int64_t>{2, 2, 32}));
        CHECK(gate->layout ==
              pie_cuda_driver::TensorLayoutKind::Grouped);
    }
    const auto* up =
        find_spec(plan, "model.layers.0.mlp.experts.up_proj.weight");
    CHECK(up != nullptr);
    if (up != nullptr) {
        CHECK((up->shape == std::vector<std::int64_t>{2, 2, 32}));
    }
    const auto* down =
        find_spec(plan, "model.layers.0.mlp.experts.down_proj.weight");
    CHECK(down != nullptr);
    if (down != nullptr) {
        CHECK(down->dtype == pie_cuda_driver::DType::BF16);
        CHECK((down->shape == std::vector<std::int64_t>{2, 8, 16}));
        CHECK(down->layout ==
              pie_cuda_driver::TensorLayoutKind::Grouped);
    }
    CHECK(find_spec(
        plan, "model.layers.0.mlp.experts.gate_up_proj.weight") == nullptr);

    const auto storage = pie_cuda_driver::compile_storage_program(
        plan, meta, /*tp_rank=*/0, /*tp_size=*/2);
    pie_cuda_driver::validate_storage_program(plan, storage);
    CHECK_EQ(count_tile_maps(storage, pie_cuda_driver::TileMapKind::Decode),
             std::size_t{2});
    CHECK_EQ(count_storage_transforms(
                 storage, pie_cuda_driver::StorageTransformKind::Deinterleave),
             std::size_t{2});
    CHECK_EQ(count_storage_transforms(
                 storage, pie_cuda_driver::StorageTransformKind::Slice),
             std::size_t{1});
    CHECK_EQ(count_storage_instrs(
                 storage, pie_cuda_driver::StorageInstrKind::Release),
             std::size_t{2});
    CHECK(storage.memory.algebra_extent_write_count >= 6);
    CHECK(storage.memory.tile_map_count == 2);
}

void test_gpt_oss_mxfp4_native_emits_quant_packed_experts() {
    auto hf = gpt_oss_mxfp4_config();
    const auto meta = gpt_oss_mxfp4_metadata();
    pie_cuda_driver::Config cfg{};
    pie_cuda_driver::BackendTarget target{};
    target.mxfp4_moe = pie_cuda_driver::Mxfp4MoeLowering::RoutedDequant;
    auto plan = pie_cuda_driver::build_model_layout_plan(
        hf, cfg, meta, /*tp_size=*/1, target);
    pie_cuda_driver::validate_layout_plan(plan);

    CHECK(!plan.algebra.bindings.empty());
    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Decode),
             std::size_t{0});
    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Attach),
             std::size_t{2});

    const auto* gate_up =
        find_spec(plan, "model.layers.0.mlp.experts.gate_up_proj.weight");
    CHECK(gate_up != nullptr);
    if (gate_up != nullptr) {
        CHECK(gate_up->dtype == pie_cuda_driver::DType::UINT8);
        CHECK(gate_up->layout ==
              pie_cuda_driver::TensorLayoutKind::QuantPacked);
        CHECK(gate_up->quant.format ==
              pie_cuda_driver::QuantFormat::Mxfp4E2M1E8M0);
        CHECK_EQ(gate_up->quant.group_size, 32);
        CHECK_EQ(gate_up->quant.scale_tensor,
                 std::string("model.layers.0.mlp.experts.gate_up_proj.weight_scale"));
    }

    const auto* down =
        find_spec(plan, "model.layers.0.mlp.experts.down_proj.weight");
    CHECK(down != nullptr);
    if (down != nullptr) {
        CHECK(down->dtype == pie_cuda_driver::DType::UINT8);
        CHECK(down->layout == pie_cuda_driver::TensorLayoutKind::QuantPacked);
        CHECK(down->quant.format ==
              pie_cuda_driver::QuantFormat::Mxfp4E2M1E8M0);
    }

    const auto storage = pie_cuda_driver::compile_storage_program(
        plan, meta, /*tp_rank=*/0, /*tp_size=*/1);
    pie_cuda_driver::validate_storage_program(plan, storage);
    CHECK_EQ(count_storage_instrs(
                 storage, pie_cuda_driver::StorageInstrKind::Attach),
             std::size_t{2});
    CHECK_EQ(count_tile_maps(storage, pie_cuda_driver::TileMapKind::Decode),
             std::size_t{0});
    CHECK(storage.memory.algebra_extent_write_count >= 6);
}

void test_gpt_oss_mxfp4_true_native_gemm_requires_backend() {
    auto hf = gpt_oss_mxfp4_config();
    const auto meta = gpt_oss_mxfp4_metadata();
    pie_cuda_driver::Config cfg{};
    pie_cuda_driver::BackendTarget target{};
    target.mxfp4_moe = pie_cuda_driver::Mxfp4MoeLowering::NativeGemm;
    target.mxfp4_native_gemm = false;

    CHECK_THROWS(pie_cuda_driver::build_model_layout_plan(
        hf, cfg, meta, /*tp_size=*/1, target));
}

void test_fp8_scale_inv_lowers_to_scheduled_dequant() {
    using pie_cuda_driver::DType;
    pie_cuda_driver::HfConfig hf{};
    hf.model_type = "mistral3";

    FakeMetadata meta;
    meta.add("model.layers.0.self_attn.q_proj.weight", DType::FP8_E4M3, {8, 8});
    meta.add("model.layers.0.self_attn.q_proj.weight_scale_inv", DType::FP32, {1});

    pie_cuda_driver::Config cfg{};
    pie_cuda_driver::BackendTarget target{};
    auto plan = pie_cuda_driver::build_model_layout_plan(
        hf, cfg, meta, /*tp_size=*/1, target);
    pie_cuda_driver::validate_layout_plan(plan);

    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Decode),
             std::size_t{1});
    CHECK_EQ(count_exprs(plan, pie_cuda_driver::LayoutExprKind::Release),
             std::size_t{1});
    const auto* weight =
        find_spec(plan, "model.layers.0.self_attn.q_proj.weight");
    CHECK(weight != nullptr);
    if (weight != nullptr) {
        CHECK(weight->dtype == DType::BF16);
        CHECK((weight->shape == std::vector<std::int64_t>{8, 8}));
    }
    CHECK(find_spec(
        plan,
        "model.layers.0.self_attn.q_proj.weight_scale_inv") == nullptr);
    CHECK(find_spec(
        plan,
        "model.layers.0.self_attn.q_proj.weight.__fp8_scale_inv_scale") != nullptr);
}

}  // namespace

int main() {
    test_runtime_abi_declares_final_layout_names();
    test_gguf_checkpoint_source_reads_dense_tensor_extents();
    test_gguf_checkpoint_source_reads_q4_0_block_extents_and_decode();
    test_dense_qwen_plan_packs_projection_groups();
    test_storage_program_lowers_packed_groups_to_extent_writes();
    test_storage_compiler_covers_every_semantic_expr();
    test_storage_schedule_orders_ready_extent_writes_by_file_offset();
    test_storage_compiler_lowers_algebra_only_copy_plan();
    test_native_layout_planner_builds_dense_packed_algebra_plan();
    test_layout_optimizer_pushes_select_and_sinks_cast();
    test_layout_optimizer_covers_proposal_rewrites();
    test_dense_schema_adapters_pack_mistral_olmo_variants();
    test_qwen36_dense_linear_attention_plan();
    test_dump_plan_has_no_post_load_transform_channel();
    test_runtime_int8_lowers_to_scheduled_quant_ops();
    test_compressed_tensors_int8_lowers_to_quant_packed();
    test_fp16_copy_lowers_to_scheduled_cast();
    test_phi3_tp_uses_row_range_shards_for_fused_tensors();
    test_phi3_fp16_row_range_shards_cast_to_bf16();
    test_gptq_symmetric_lowers_to_repack_layout();
    test_gptq_symmetric_tp_lowers_to_local_repack_layout();
    test_gptq_symmetric_without_marlin_lowers_to_dequant();
    test_awq_lowers_to_scheduled_dequant();
    test_awq_lowers_to_marlin_repack_when_target_supports_int4();
    test_awq_tp_lowers_to_local_dequant();
    test_gptq_asymmetric_tp_lowers_to_local_dequant();
    test_gptq_desc_act_row_tp_keeps_full_group_metadata();
    test_qwen_moe_tp_uses_expert_shard_ops();
    test_qwen_moe_per_expert_fusion_is_scheduled();
    test_mixtral_schema_adapter_marks_and_shards_expert_weights();
    test_gemma4_schema_adapter_declares_fused_expert_groups();
    test_qwen36_moe_plan_shards_fused_experts();
    test_gemma4_dense_and_moe_plan_coverage();
    test_semantic_adapters_declare_group_roles();
    test_gpt_oss_mxfp4_fallback_emits_dequantized_experts();
    test_gpt_oss_mxfp4_native_emits_quant_packed_experts();
    test_gpt_oss_mxfp4_true_native_gemm_requires_backend();
    test_fp8_scale_inv_lowers_to_scheduled_dequant();

    if (g_failures != 0) {
        std::fprintf(stderr, "%d layout-plan test failure(s)\n", g_failures);
        return 1;
    }
    return 0;
}
