#include <algorithm>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "loader/model_schema.hpp"
#include "loader/physical_load_plan.hpp"

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

class FakeMetadata final : public pie_cuda_driver::TensorMetadataSource {
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
};

std::size_t count_ops(const pie_cuda_driver::LoadPlan& plan,
                      pie_cuda_driver::LoadOpKind kind) {
    return static_cast<std::size_t>(std::count_if(
        plan.ops.begin(), plan.ops.end(),
        [kind](const auto& op) { return op.kind == kind; }));
}

const pie_cuda_driver::TensorSpec* find_spec(
    const pie_cuda_driver::LoadPlan& plan,
    const std::string& name) {
    const auto it = plan.tensors.find(name);
    return it == plan.tensors.end() ? nullptr : &it->second;
}

bool has_copy_output(const pie_cuda_driver::LoadPlan& plan,
                     const std::string& output_name) {
    return std::any_of(
        plan.ops.begin(), plan.ops.end(),
        [&](const auto& op) {
            return op.kind == pie_cuda_driver::LoadOpKind::Copy &&
                   load_op_output(op) == output_name;
        });
}

const pie_cuda_driver::LoadOp* find_op(
    const pie_cuda_driver::LoadPlan& plan,
    pie_cuda_driver::LoadOpKind kind,
    const std::string& output_name) {
    const auto it = std::find_if(
        plan.ops.begin(), plan.ops.end(),
        [&](const auto& op) {
            return op.kind == kind && load_op_output(op) == output_name;
        });
    return it == plan.ops.end() ? nullptr : &*it;
}

bool has_group(const pie_cuda_driver::LogicalTensorGraph& graph,
               pie_cuda_driver::LogicalTensorGroupKind kind) {
    return std::any_of(
        graph.groups.begin(), graph.groups.end(),
        [kind](const auto& group) { return group.kind == kind; });
}

pie_cuda_driver::HfConfig qwen3_config() {
    pie_cuda_driver::HfConfig hf{};
    hf.model_type = "qwen3";
    hf.torch_dtype = "bfloat16";
    return hf;
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
    const pie_cuda_driver::LoadTarget target{};
    auto plan = pie_cuda_driver::build_model_load_plan(
        hf, cfg, meta, /*tp_size=*/1, target);
    pie_cuda_driver::validate_load_plan(plan);

    CHECK_EQ(plan.axis_concat_groups, std::size_t{2});
    CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::AxisConcat),
             std::size_t{2});
    CHECK_EQ(plan.memory.max_temporary_bytes, std::uint64_t{0});
    CHECK(!has_copy_output(plan, "model.layers.0.self_attn.q_proj.weight"));
    CHECK(!has_copy_output(plan, "model.layers.0.self_attn.k_proj.weight"));
    CHECK(!has_copy_output(plan, "model.layers.0.self_attn.v_proj.weight"));
    CHECK(!has_copy_output(plan, "model.layers.0.mlp.gate_proj.weight"));
    CHECK(!has_copy_output(plan, "model.layers.0.mlp.up_proj.weight"));

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

void test_physical_plan_lowers_packed_groups_to_byte_writes() {
    auto hf = qwen3_config();
    pie_cuda_driver::Config cfg{};
    const auto meta = dense_llama_metadata();
    const pie_cuda_driver::LoadTarget target{};
    auto plan = pie_cuda_driver::build_model_load_plan(
        hf, cfg, meta, /*tp_size=*/1, target);
    auto physical = pie_cuda_driver::build_physical_load_plan(
        plan, meta, /*tp_rank=*/0, /*tp_size=*/1);

    CHECK_EQ(physical.memory.max_copy_temporary_bytes, std::uint64_t{0});
    CHECK(physical.memory.byte_write_count >= 5);
    std::vector<const pie_cuda_driver::ByteRangeWrite*> qkv_writes;
    for (const auto& write : physical.byte_writes) {
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
        pie_cuda_driver::dump_load_plan_json(plan, physical);
    CHECK(dump.find("\"physical\"") != std::string::npos);
    CHECK(dump.find("axis-concatenated") != std::string::npos);
}

void test_dense_schema_adapters_pack_mistral_olmo_variants() {
    const auto meta = dense_llama_metadata();
    pie_cuda_driver::Config cfg{};
    const pie_cuda_driver::LoadTarget target{};
    for (const std::string model_type :
         {"mistral3", "ministral3", "olmo3"}) {
        auto hf = qwen3_config();
        hf.model_type = model_type;
        auto plan = pie_cuda_driver::build_model_load_plan(
            hf, cfg, meta, /*tp_size=*/1, target);
        pie_cuda_driver::validate_load_plan(plan);
        CHECK_EQ(plan.axis_concat_groups, std::size_t{2});
        CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::AxisConcat),
                 std::size_t{2});
    }
}

void test_dump_plan_has_no_post_load_transform_channel() {
    auto hf = qwen3_config();
    pie_cuda_driver::Config cfg{};
    const auto meta = dense_llama_metadata();
    const pie_cuda_driver::LoadTarget target{};
    auto plan = pie_cuda_driver::build_model_load_plan(
        hf, cfg, meta, /*tp_size=*/1, target);
    const std::string json = pie_cuda_driver::dump_load_plan_json(plan);
    CHECK(json.find("\"transforms\"") == std::string::npos);
    CHECK(json.find("transform ops") == std::string::npos);
}

void test_runtime_int8_lowers_to_scheduled_quant_ops() {
    auto hf = qwen3_config();
    pie_cuda_driver::Config cfg{};
    cfg.model.runtime_quant = "int8";
    const auto meta = dense_llama_metadata();
    const pie_cuda_driver::LoadTarget target{};
    auto plan = pie_cuda_driver::build_model_load_plan(
        hf, cfg, meta, /*tp_size=*/1, target);
    pie_cuda_driver::validate_load_plan(plan);

    CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::QuantizeRuntime),
             std::size_t{7});
    CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::BindMetadata),
             std::size_t{7});
    CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::Drop),
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

    const pie_cuda_driver::LoadTarget target{};
    auto plan = pie_cuda_driver::build_model_load_plan(
        hf, cfg, meta, /*tp_size=*/1, target);
    pie_cuda_driver::validate_load_plan(plan);

    CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::BindMetadata),
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

    const pie_cuda_driver::LoadTarget target{};
    auto plan = pie_cuda_driver::build_model_load_plan(
        hf, cfg, meta, /*tp_size=*/1, target);
    pie_cuda_driver::validate_load_plan(plan);

    CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::Copy),
             std::size_t{1});
    CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::Cast),
             std::size_t{1});
    CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::Drop),
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
    const auto physical = pie_cuda_driver::build_physical_load_plan(
        plan, meta, /*tp_rank=*/0, /*tp_size=*/1);
    CHECK_EQ(physical.tiled_transforms.size(), std::size_t{1});
    if (!physical.tiled_transforms.empty()) {
        CHECK(physical.tiled_transforms[0].kind ==
              pie_cuda_driver::PhysicalTransformKind::Cast);
        CHECK_EQ(physical.tiled_transforms[0].output_name,
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
    pie_cuda_driver::LoadTarget target{};
    target.gptq_marlin_int4 = false;
    auto plan = pie_cuda_driver::build_model_load_plan(
        hf, cfg, meta, /*tp_size=*/2, target);
    pie_cuda_driver::validate_load_plan(plan);

    CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::RowRangeShard),
             std::size_t{5});
    CHECK(!has_copy_output(plan, "model.layers.0.self_attn.qkv_proj.weight"));
    CHECK(!has_copy_output(plan, "model.layers.0.mlp.gate_up_proj.weight"));

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
    pie_cuda_driver::LoadTarget target{};
    target.gptq_marlin_int4 = false;
    auto plan = pie_cuda_driver::build_model_load_plan(
        hf, cfg, meta, /*tp_size=*/2, target);
    pie_cuda_driver::validate_load_plan(plan);

    CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::RowRangeShard),
             std::size_t{3});
    CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::Cast),
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
    pie_cuda_driver::LoadTarget target{};
    target.gptq_marlin_int4 = true;
    auto plan = pie_cuda_driver::build_model_load_plan(
        hf, cfg, meta, /*tp_size=*/1, target);
    pie_cuda_driver::validate_load_plan(plan);

    CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::RepackLayout),
             std::size_t{1});
    CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::BindMetadata),
             std::size_t{1});
    CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::Drop),
             std::size_t{1});
    CHECK(!has_copy_output(plan, "model.layers.0.self_attn.q_proj.qweight"));
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
    pie_cuda_driver::LoadTarget target{};
    target.gptq_marlin_int4 = true;
    auto plan = pie_cuda_driver::build_model_load_plan(
        hf, cfg, meta, /*tp_size=*/2, target);
    pie_cuda_driver::validate_load_plan(plan);

    CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::RepackLayout),
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
    const auto* q_copy = find_op(
        plan, pie_cuda_driver::LoadOpKind::Copy,
        "model.layers.0.self_attn.q_proj.weight.__gptq_qweight");
    CHECK(q_copy != nullptr);
    if (q_copy != nullptr) CHECK_EQ(load_op_shard_axis(*q_copy), 1);

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
    const auto* o_copy = find_op(
        plan, pie_cuda_driver::LoadOpKind::Copy,
        "model.layers.0.self_attn.o_proj.weight.__gptq_qweight");
    CHECK(o_copy != nullptr);
    if (o_copy != nullptr) CHECK_EQ(load_op_shard_axis(*o_copy), 0);
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
    pie_cuda_driver::LoadTarget target{};
    target.gptq_marlin_int4 = false;
    auto plan = pie_cuda_driver::build_model_load_plan(
        hf, cfg, meta, /*tp_size=*/1, target);
    pie_cuda_driver::validate_load_plan(plan);

    CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::RepackLayout),
             std::size_t{0});
    CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::Dequantize),
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
    pie_cuda_driver::LoadTarget target{};
    target.gptq_marlin_int4 = false;
    auto plan = pie_cuda_driver::build_model_load_plan(
        hf, cfg, meta, /*tp_size=*/1, target);
    pie_cuda_driver::validate_load_plan(plan);

    CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::Dequantize),
             std::size_t{1});
    CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::Drop),
             std::size_t{1});
    CHECK(!has_copy_output(plan, "model.layers.0.self_attn.q_proj.qweight"));
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
    pie_cuda_driver::LoadTarget target{};
    target.gptq_marlin_int4 = true;
    auto plan = pie_cuda_driver::build_model_load_plan(
        hf, cfg, meta, /*tp_size=*/1, target);
    pie_cuda_driver::validate_load_plan(plan);

    CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::RepackLayout),
             std::size_t{1});
    CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::Dequantize),
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
    pie_cuda_driver::LoadTarget target{};
    target.gptq_marlin_int4 = false;
    auto plan = pie_cuda_driver::build_model_load_plan(
        hf, cfg, meta, /*tp_size=*/2, target);
    pie_cuda_driver::validate_load_plan(plan);

    CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::Dequantize),
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
    const auto* q_copy = find_op(
        plan, pie_cuda_driver::LoadOpKind::Copy,
        "model.layers.0.self_attn.q_proj.weight.__awq_qweight");
    CHECK(q_copy != nullptr);
    if (q_copy != nullptr) CHECK_EQ(load_op_shard_axis(*q_copy), 1);

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
    const auto* o_copy = find_op(
        plan, pie_cuda_driver::LoadOpKind::Copy,
        "model.layers.0.self_attn.o_proj.weight.__awq_qweight");
    CHECK(o_copy != nullptr);
    if (o_copy != nullptr) CHECK_EQ(load_op_shard_axis(*o_copy), 0);
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
    const pie_cuda_driver::LoadTarget target{};
    auto plan = pie_cuda_driver::build_model_load_plan(
        hf, cfg, meta, /*tp_size=*/2, target);
    pie_cuda_driver::validate_load_plan(plan);

    CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::Dequantize),
             std::size_t{2});
    CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::RepackLayout),
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
    const pie_cuda_driver::LoadTarget target{};
    auto plan = pie_cuda_driver::build_model_load_plan(
        hf, cfg, meta, /*tp_size=*/2, target);
    pie_cuda_driver::validate_load_plan(plan);

    CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::Dequantize),
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
    const auto* dequant = find_op(
        plan, pie_cuda_driver::LoadOpKind::Dequantize,
        "model.layers.0.self_attn.o_proj.weight");
    CHECK(dequant != nullptr);
    if (dequant != nullptr) {
        CHECK_EQ(load_op_inputs(*dequant).size(), std::size_t{4});
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
    const pie_cuda_driver::LoadTarget target{};
    auto plan = pie_cuda_driver::build_model_load_plan(
        hf, cfg, meta, /*tp_size=*/2, target);
    pie_cuda_driver::validate_load_plan(plan);

    CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::GroupedSliceConcat),
             std::size_t{1});
    CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::GroupedSlice),
             std::size_t{1});
    CHECK(!has_copy_output(plan, "model.layers.0.mlp.experts.gate_up_proj"));
    CHECK(!has_copy_output(plan, "model.layers.0.mlp.experts.down_proj"));

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
    const pie_cuda_driver::LoadTarget target{};
    auto plan = pie_cuda_driver::build_model_load_plan(
        hf, cfg, meta, /*tp_size=*/2, target);
    pie_cuda_driver::validate_load_plan(plan);

    CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::StackGroups),
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
    CHECK(!has_copy_output(
        plan, "model.layers.0.mlp.experts.0.gate_proj.weight"));
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

    const auto graph = pie_cuda_driver::build_logical_tensor_graph(hf, meta);
    CHECK(std::any_of(
        graph.tensors.begin(), graph.tensors.end(),
        [](const auto& t) {
            return t.runtime_name.ends_with(".w1.weight") &&
                   t.role == pie_cuda_driver::LogicalTensorRole::MoeExpertGate;
        }));
    CHECK(std::any_of(
        graph.tensors.begin(), graph.tensors.end(),
        [](const auto& t) {
            return t.runtime_name.ends_with(".w3.weight") &&
                   t.role == pie_cuda_driver::LogicalTensorRole::MoeExpertUp;
        }));

    pie_cuda_driver::Config cfg{};
    const pie_cuda_driver::LoadTarget target{};
    auto plan = pie_cuda_driver::build_model_load_plan(
        hf, cfg, meta, /*tp_size=*/2, target);
    pie_cuda_driver::validate_load_plan(plan);

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

    const auto graph = pie_cuda_driver::build_logical_tensor_graph(hf, meta);
    CHECK(has_group(
        graph, pie_cuda_driver::LogicalTensorGroupKind::FusedMoeExperts));
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

void test_gpt_oss_mxfp4_fallback_emits_dequantized_experts() {
    auto hf = gpt_oss_mxfp4_config();
    const auto meta = gpt_oss_mxfp4_metadata();
    pie_cuda_driver::Config cfg{};
    const pie_cuda_driver::LoadTarget target{};
    auto plan = pie_cuda_driver::build_model_load_plan(
        hf, cfg, meta, /*tp_size=*/2, target);
    pie_cuda_driver::validate_load_plan(plan);

    CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::Dequantize),
             std::size_t{2});
    CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::Deinterleave),
             std::size_t{2});
    CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::Slice),
             std::size_t{1});
    CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::Drop),
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
}

void test_gpt_oss_mxfp4_native_emits_quant_packed_experts() {
    auto hf = gpt_oss_mxfp4_config();
    const auto meta = gpt_oss_mxfp4_metadata();
    pie_cuda_driver::Config cfg{};
    pie_cuda_driver::LoadTarget target{};
    target.mxfp4_moe = pie_cuda_driver::Mxfp4MoeLowering::RoutedDequant;
    auto plan = pie_cuda_driver::build_model_load_plan(
        hf, cfg, meta, /*tp_size=*/1, target);
    pie_cuda_driver::validate_load_plan(plan);

    CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::Dequantize),
             std::size_t{0});
    CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::BindMetadata),
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
}

void test_gpt_oss_mxfp4_true_native_gemm_requires_backend() {
    auto hf = gpt_oss_mxfp4_config();
    const auto meta = gpt_oss_mxfp4_metadata();
    pie_cuda_driver::Config cfg{};
    pie_cuda_driver::LoadTarget target{};
    target.mxfp4_moe = pie_cuda_driver::Mxfp4MoeLowering::NativeGemm;
    target.mxfp4_native_gemm = false;

    CHECK_THROWS(pie_cuda_driver::build_model_load_plan(
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
    pie_cuda_driver::LoadTarget target{};
    auto plan = pie_cuda_driver::build_model_load_plan(
        hf, cfg, meta, /*tp_size=*/1, target);
    pie_cuda_driver::validate_load_plan(plan);

    CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::Dequantize),
             std::size_t{1});
    CHECK_EQ(count_ops(plan, pie_cuda_driver::LoadOpKind::Drop),
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
    test_dense_qwen_plan_packs_projection_groups();
    test_physical_plan_lowers_packed_groups_to_byte_writes();
    test_dense_schema_adapters_pack_mistral_olmo_variants();
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
    test_gpt_oss_mxfp4_fallback_emits_dequantized_experts();
    test_gpt_oss_mxfp4_native_emits_quant_packed_experts();
    test_gpt_oss_mxfp4_true_native_gemm_requires_backend();
    test_fp8_scale_inv_lowers_to_scheduled_dequant();

    if (g_failures != 0) {
        std::fprintf(stderr, "%d load-plan test failure(s)\n", g_failures);
        return 1;
    }
    return 0;
}
