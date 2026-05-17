#include "loader/load_plan.hpp"

#include <sstream>
#include <stdexcept>
#include <unordered_set>
#include <utility>

#include "tensor.hpp"

namespace pie_cuda_driver {

const char* load_op_kind_name(LoadOpKind kind) noexcept {
    switch (kind) {
    case LoadOpKind::Read: return "Read";
    case LoadOpKind::Copy: return "Copy";
    case LoadOpKind::Slice: return "Slice";
    case LoadOpKind::Shard: return "Shard";
    case LoadOpKind::RowRangeShard: return "RowRangeShard";
    case LoadOpKind::MoeGateUpShard: return "MoeGateUpShard";
    case LoadOpKind::MoeDownShard: return "MoeDownShard";
    case LoadOpKind::Cast: return "Cast";
    case LoadOpKind::Concat: return "Concat";
    case LoadOpKind::PackRows: return "PackRows";
    case LoadOpKind::View: return "View";
    case LoadOpKind::Alias: return "Alias";
    case LoadOpKind::Drop: return "Drop";
    case LoadOpKind::QuantizeRuntime: return "QuantizeRuntime";
    case LoadOpKind::Dequantize: return "Dequantize";
    case LoadOpKind::SplitInterleaved: return "SplitInterleaved";
    case LoadOpKind::RepackQuant: return "RepackQuant";
    case LoadOpKind::FuseMoeExperts: return "FuseMoeExperts";
    case LoadOpKind::AttachQuantMeta: return "AttachQuantMeta";
    case LoadOpKind::Materialize: return "Materialize";
    }
    return "?";
}

const char* tensor_layout_kind_name(TensorLayoutKind kind) noexcept {
    switch (kind) {
    case TensorLayoutKind::Dense: return "dense";
    case TensorLayoutKind::RowPacked: return "row-packed";
    case TensorLayoutKind::PackedQkv: return "packed-qkv";
    case TensorLayoutKind::PackedGateUp: return "packed-gate-up";
    case TensorLayoutKind::FusedMoeExperts: return "fused-moe-experts";
    case TensorLayoutKind::QuantPacked: return "quant-packed";
    case TensorLayoutKind::View: return "view";
    }
    return "?";
}

const char* quant_format_name(QuantFormat format) noexcept {
    switch (format) {
    case QuantFormat::None: return "none";
    case QuantFormat::RuntimeFp8E4M3: return "runtime-fp8-e4m3";
    case QuantFormat::RuntimeInt8: return "runtime-int8";
    case QuantFormat::GptqInt4: return "gptq-int4";
    case QuantFormat::AwqInt4: return "awq-int4";
    case QuantFormat::CompressedFp8E4M3: return "compressed-fp8-e4m3";
    case QuantFormat::CompressedInt8: return "compressed-int8";
    case QuantFormat::Mxfp4E2M1E8M0: return "mxfp4-e2m1-e8m0";
    }
    return "?";
}

LoadOp make_raw_load_op(
    LoadOpKind kind,
    std::string output_name,
    std::string raw_name,
    int shard_axis)
{
    return LoadOp{
        .kind = kind,
        .payload = RawLoadPayload{
            .output_name = std::move(output_name),
            .raw_name = std::move(raw_name),
            .shard_axis = shard_axis,
        },
    };
}

LoadOp make_row_range_shard_op(
    std::string output_name,
    std::string raw_name,
    std::int64_t row_offset,
    std::int64_t rows)
{
    return LoadOp{
        .kind = LoadOpKind::RowRangeShard,
        .payload = RowRangeShardPayload{
            .output_name = std::move(output_name),
            .raw_name = std::move(raw_name),
            .row_offset = row_offset,
            .rows = rows,
        },
    };
}

LoadOp make_tensor_op(
    LoadOpKind kind,
    std::string output_name,
    std::vector<std::string> inputs,
    std::string secondary_output_name,
    int shard_axis)
{
    return LoadOp{
        .kind = kind,
        .payload = TensorOpPayload{
            .output_name = std::move(output_name),
            .secondary_output_name = std::move(secondary_output_name),
            .inputs = std::move(inputs),
            .shard_axis = shard_axis,
        },
    };
}

LoadOp make_slice_op(
    std::string output_name,
    std::string input,
    int slice_axis,
    std::int64_t slice_start,
    std::int64_t slice_length,
    int shard_axis)
{
    return LoadOp{
        .kind = LoadOpKind::Slice,
        .payload = SlicePayload{
            .output_name = std::move(output_name),
            .inputs = {std::move(input)},
            .slice_axis = slice_axis,
            .slice_start = slice_start,
            .slice_length = slice_length,
            .shard_axis = shard_axis,
        },
    };
}

LoadOp make_pack_rows_op(
    std::string output_name,
    int shard_axis,
    std::vector<PackedRowSource> row_sources)
{
    return LoadOp{
        .kind = LoadOpKind::PackRows,
        .payload = PackRowsPayload{
            .output_name = std::move(output_name),
            .shard_axis = shard_axis,
            .row_sources = std::move(row_sources),
        },
    };
}

LoadOp make_fuse_moe_experts_op(
    std::string output_name,
    std::string secondary_output_name,
    std::vector<std::string> inputs,
    std::vector<PackedRowSource> row_sources)
{
    return LoadOp{
        .kind = LoadOpKind::FuseMoeExperts,
        .payload = FuseMoeExpertsPayload{
            .output_name = std::move(output_name),
            .secondary_output_name = std::move(secondary_output_name),
            .inputs = std::move(inputs),
            .row_sources = std::move(row_sources),
        },
    };
}

namespace {

const std::string kEmptyString;
const std::vector<std::string> kEmptyInputs;
const std::vector<PackedRowSource> kEmptyRowSources;

template <typename Payload>
const std::string& output_of(const Payload& payload) {
    return payload.output_name;
}

const char* ownership_kind_name(TensorOwnershipKind kind) noexcept {
    switch (kind) {
    case TensorOwnershipKind::Owned: return "owned";
    case TensorOwnershipKind::BorrowedView: return "borrowed-view";
    case TensorOwnershipKind::Alias: return "alias";
    case TensorOwnershipKind::Temporary: return "temporary";
    }
    return "?";
}

const char* parallel_kind_name(TensorParallelKind kind) noexcept {
    switch (kind) {
    case TensorParallelKind::Replicated: return "replicated";
    case TensorParallelKind::Column: return "column";
    case TensorParallelKind::Row: return "row";
    case TensorParallelKind::Expert: return "expert";
    case TensorParallelKind::Custom: return "custom";
    }
    return "?";
}

const char* quant_granularity_name(QuantGranularity granularity) noexcept {
    switch (granularity) {
    case QuantGranularity::None: return "none";
    case QuantGranularity::PerTensor: return "per-tensor";
    case QuantGranularity::PerChannel: return "per-channel";
    case QuantGranularity::PerGroup: return "per-group";
    }
    return "?";
}

bool op_produces_tensor(LoadOpKind kind) noexcept {
    return kind != LoadOpKind::Drop &&
           kind != LoadOpKind::AttachQuantMeta;
}

void json_string(std::ostream& out, const std::string& value) {
    out << '"';
    for (const char ch : value) {
        switch (ch) {
        case '\\': out << "\\\\"; break;
        case '"': out << "\\\""; break;
        case '\n': out << "\\n"; break;
        case '\r': out << "\\r"; break;
        case '\t': out << "\\t"; break;
        default: out << ch; break;
        }
    }
    out << '"';
}

void json_shape(std::ostream& out, const std::vector<std::int64_t>& shape) {
    out << '[';
    for (std::size_t i = 0; i < shape.size(); ++i) {
        if (i) out << ',';
        out << shape[i];
    }
    out << ']';
}

}  // namespace

const std::string& load_op_output(const LoadOp& op) {
    return std::visit(
        [](const auto& payload) -> const std::string& {
            return payload.output_name;
        },
        op.payload);
}

const std::string& load_op_secondary_output(const LoadOp& op) {
    return std::visit(
        [](const auto& payload) -> const std::string& {
            if constexpr (requires { payload.secondary_output_name; }) {
                return payload.secondary_output_name;
            } else {
                return kEmptyString;
            }
        },
        op.payload);
}

const std::string& load_op_raw_name(const LoadOp& op) {
    return std::visit(
        [](const auto& payload) -> const std::string& {
            if constexpr (requires { payload.raw_name; }) {
                return payload.raw_name;
            } else {
                return kEmptyString;
            }
        },
        op.payload);
}

const std::vector<std::string>& load_op_inputs(const LoadOp& op) {
    return std::visit(
        [](const auto& payload) -> const std::vector<std::string>& {
            if constexpr (requires { payload.inputs; }) {
                return payload.inputs;
            } else {
                return kEmptyInputs;
            }
        },
        op.payload);
}

int load_op_shard_axis(const LoadOp& op) {
    return std::visit(
        [](const auto& payload) -> int {
            if constexpr (requires { payload.shard_axis; }) {
                return payload.shard_axis;
            } else {
                return -1;
            }
        },
        op.payload);
}

int load_op_slice_axis(const LoadOp& op) {
    if (const auto* payload = std::get_if<SlicePayload>(&op.payload)) {
        return payload->slice_axis;
    }
    return -1;
}

std::int64_t load_op_row_offset(const LoadOp& op) {
    if (const auto* payload = std::get_if<RowRangeShardPayload>(&op.payload)) {
        return payload->row_offset;
    }
    return 0;
}

std::int64_t load_op_rows(const LoadOp& op) {
    if (const auto* payload = std::get_if<RowRangeShardPayload>(&op.payload)) {
        return payload->rows;
    }
    return 0;
}

std::int64_t load_op_slice_start(const LoadOp& op) {
    if (const auto* payload = std::get_if<SlicePayload>(&op.payload)) {
        return payload->slice_start;
    }
    return 0;
}

std::int64_t load_op_slice_length(const LoadOp& op) {
    if (const auto* payload = std::get_if<SlicePayload>(&op.payload)) {
        return payload->slice_length;
    }
    return 0;
}

const std::vector<PackedRowSource>& load_op_row_sources(const LoadOp& op) {
    return std::visit(
        [](const auto& payload) -> const std::vector<PackedRowSource>& {
            if constexpr (requires { payload.row_sources; }) {
                return payload.row_sources;
            } else {
                return kEmptyRowSources;
            }
        },
        op.payload);
}

void set_load_op_output(LoadOp& op, std::string output_name) {
    std::visit(
        [&](auto& payload) {
            payload.output_name = output_name;
        },
        op.payload);
}

void validate_load_plan(const LoadPlan& plan) {
    for (const auto& [name, spec] : plan.tensors) {
        if (name.empty() || spec.name.empty()) {
            throw std::runtime_error("load plan: tensor spec has empty name");
        }
        if (name != spec.name) {
            throw std::runtime_error(
                "load plan: tensor spec key/name mismatch for '" + name + "'");
        }
        for (const auto dim : spec.shape) {
            if (dim < 0) {
                throw std::runtime_error(
                    "load plan: tensor spec '" + name +
                    "' has a negative dimension");
            }
        }
        if (spec.ownership == TensorOwnershipKind::BorrowedView ||
            spec.ownership == TensorOwnershipKind::Alias) {
            if (spec.backing_tensor.empty()) {
                throw std::runtime_error(
                    "load plan: view/alias spec '" + name +
                    "' has no backing tensor");
            }
            if (!plan.tensors.contains(spec.backing_tensor)) {
                throw std::runtime_error(
                    "load plan: view/alias spec '" + name +
                    "' references missing backing tensor '" +
                    spec.backing_tensor + "'");
            }
        }
        if (spec.quant.format != QuantFormat::None) {
            if (spec.quant.scale_tensor.empty()) {
                throw std::runtime_error(
                    "load plan: quant tensor spec '" + name +
                    "' has no scale tensor");
            }
            if (!plan.tensors.contains(spec.quant.scale_tensor)) {
                throw std::runtime_error(
                    "load plan: quant tensor spec '" + name +
                    "' references missing scale tensor '" +
                    spec.quant.scale_tensor + "'");
            }
            if (!spec.quant.zero_point_tensor.empty() &&
                !plan.tensors.contains(spec.quant.zero_point_tensor)) {
                throw std::runtime_error(
                    "load plan: quant tensor spec '" + name +
                    "' references missing zero-point tensor '" +
                    spec.quant.zero_point_tensor + "'");
            }
        }
    }

    std::unordered_set<std::string> produced;
    produced.reserve(plan.ops.size() + plan.tensors.size());

    for (const auto& op : plan.ops) {
        if (load_op_output(op).empty() &&
            (op.kind != LoadOpKind::Drop || load_op_inputs(op).empty())) {
            throw std::runtime_error(
                std::string("load plan: ") + load_op_kind_name(op.kind) +
                " op has empty output name");
        }
        for (const auto& input : load_op_inputs(op)) {
            if (input.empty()) {
                throw std::runtime_error(
                    "load plan: " +
                    std::string(load_op_kind_name(op.kind)) +
                    " op for '" + load_op_output(op) + "' has an empty input");
            }
            if (!produced.contains(input)) {
                throw std::runtime_error(
                    "load plan: " +
                    std::string(load_op_kind_name(op.kind)) +
                    " op for '" + load_op_output(op) +
                    "' reads tensor before it is produced: '" + input + "'");
            }
        }
        if (op_produces_tensor(op.kind) &&
            !plan.tensors.contains(load_op_output(op))) {
            throw std::runtime_error(
                "load plan: " + std::string(load_op_kind_name(op.kind)) +
                " op produces tensor without TensorSpec: '" +
                load_op_output(op) + "'");
        }
        if (op_produces_tensor(op.kind) &&
            !produced.insert(load_op_output(op)).second) {
            throw std::runtime_error(
                "load plan: duplicate runtime tensor output '" +
                load_op_output(op) + "'");
        }
        if (!load_op_secondary_output(op).empty() &&
            !plan.tensors.contains(load_op_secondary_output(op))) {
            throw std::runtime_error(
                "load plan: " + std::string(load_op_kind_name(op.kind)) +
                " op produces secondary tensor without TensorSpec: '" +
                load_op_secondary_output(op) + "'");
        }
        if (!load_op_secondary_output(op).empty() &&
            !produced.insert(load_op_secondary_output(op)).second) {
            throw std::runtime_error(
                "load plan: duplicate secondary runtime tensor output '" +
                load_op_secondary_output(op) + "'");
        }
        if ((op.kind == LoadOpKind::Read ||
             op.kind == LoadOpKind::Copy ||
             op.kind == LoadOpKind::Shard ||
             op.kind == LoadOpKind::RowRangeShard ||
             op.kind == LoadOpKind::MoeGateUpShard ||
             op.kind == LoadOpKind::MoeDownShard) &&
            load_op_raw_name(op).empty()) {
            throw std::runtime_error(
                "load plan: op for '" + load_op_output(op) +
                "' has empty raw tensor name");
        }
        if (op.kind == LoadOpKind::PackRows) {
            if (load_op_row_sources(op).empty()) {
                throw std::runtime_error(
                    "load plan: PackRows op for '" + load_op_output(op) +
                    "' has no row sources");
            }
            std::unordered_set<std::string> view_names;
            for (const auto& src : load_op_row_sources(op)) {
                if (src.raw_name.empty() || src.view_name.empty()) {
                    throw std::runtime_error(
                        "load plan: PackRows op for '" + load_op_output(op) +
                        "' has an empty raw/view source");
                }
                if (!view_names.insert(src.view_name).second) {
                    throw std::runtime_error(
                        "load plan: PackRows op for '" + load_op_output(op) +
                        "' has duplicate view '" + src.view_name + "'");
                }
                if (!plan.tensors.contains(src.view_name)) {
                    throw std::runtime_error(
                        "load plan: PackRows op for '" + load_op_output(op) +
                        "' has no TensorSpec for view '" + src.view_name + "'");
                }
                if (!produced.insert(src.view_name).second) {
                    throw std::runtime_error(
                        "load plan: duplicate PackRows view output '" +
                        src.view_name + "'");
                }
            }
        }
        if ((op.kind == LoadOpKind::Slice ||
             op.kind == LoadOpKind::Cast ||
             op.kind == LoadOpKind::Concat ||
             op.kind == LoadOpKind::View ||
             op.kind == LoadOpKind::Alias ||
             op.kind == LoadOpKind::QuantizeRuntime ||
             op.kind == LoadOpKind::Dequantize ||
             op.kind == LoadOpKind::RepackQuant ||
             op.kind == LoadOpKind::SplitInterleaved ||
             op.kind == LoadOpKind::Materialize) &&
            load_op_inputs(op).empty()) {
            throw std::runtime_error(
                "load plan: " + std::string(load_op_kind_name(op.kind)) +
                " op for '" + load_op_output(op) + "' has no inputs");
        }
        if (op.kind == LoadOpKind::Dequantize && load_op_inputs(op).size() < 2) {
            throw std::runtime_error(
                "load plan: Dequantize op for '" + load_op_output(op) +
                "' requires weight and scale inputs");
        }
        if (op.kind == LoadOpKind::SplitInterleaved &&
            load_op_secondary_output(op).empty()) {
            throw std::runtime_error(
                "load plan: SplitInterleaved op for '" + load_op_output(op) +
                "' requires a secondary output name");
        }
        if (op.kind == LoadOpKind::RepackQuant) {
            if (load_op_inputs(op).size() < 2) {
                throw std::runtime_error(
                    "load plan: RepackQuant op for '" + load_op_output(op) +
                    "' requires weight and metadata inputs");
            }
            if (load_op_secondary_output(op).empty()) {
                throw std::runtime_error(
                    "load plan: RepackQuant op for '" + load_op_output(op) +
                    "' requires a secondary output name");
            }
            const auto spec_it = plan.tensors.find(load_op_output(op));
            if (spec_it != plan.tensors.end() &&
                !spec_it->second.quant.zero_point_tensor.empty() &&
                !produced.insert(spec_it->second.quant.zero_point_tensor).second) {
                throw std::runtime_error(
                    "load plan: duplicate RepackQuant zero-point output '" +
                    spec_it->second.quant.zero_point_tensor + "'");
            }
        }
        if (op.kind == LoadOpKind::FuseMoeExperts) {
            if (load_op_secondary_output(op).empty()) {
                throw std::runtime_error(
                    "load plan: FuseMoeExperts op for '" + load_op_output(op) +
                    "' requires a secondary output name");
            }
            const bool has_input_triples =
                !load_op_inputs(op).empty() && load_op_inputs(op).size() % 3 == 0;
            const bool has_raw_triples =
                !load_op_row_sources(op).empty() && load_op_row_sources(op).size() % 3 == 0;
            if (has_input_triples == has_raw_triples) {
                throw std::runtime_error(
                    "load plan: FuseMoeExperts op for '" + load_op_output(op) +
                    "' expects either input tensor triples or raw source triples");
            }
            if (has_raw_triples) {
                for (const auto& src : load_op_row_sources(op)) {
                    if (src.raw_name.empty() || src.view_name.empty()) {
                        throw std::runtime_error(
                            "load plan: FuseMoeExperts op for '" +
                            load_op_output(op) + "' has empty raw source metadata");
                    }
                }
            }
        }
        if (op.kind == LoadOpKind::AttachQuantMeta) {
            const auto spec_it = plan.tensors.find(load_op_output(op));
            if (spec_it == plan.tensors.end()) {
                throw std::runtime_error(
                    "load plan: AttachQuantMeta has no TensorSpec for '" +
                    load_op_output(op) + "'");
            }
            const auto& spec = spec_it->second;
            if (!produced.contains(load_op_output(op))) {
                throw std::runtime_error(
                    "load plan: AttachQuantMeta runs before weight tensor '" +
                    load_op_output(op) + "' is produced");
            }
            if (!produced.contains(spec.quant.scale_tensor)) {
                throw std::runtime_error(
                    "load plan: AttachQuantMeta for '" + load_op_output(op) +
                    "' runs before scale tensor '" +
                    spec.quant.scale_tensor + "' is produced");
            }
            if (!spec.quant.zero_point_tensor.empty() &&
                !produced.contains(spec.quant.zero_point_tensor)) {
                throw std::runtime_error(
                    "load plan: AttachQuantMeta for '" + load_op_output(op) +
                    "' runs before zero-point tensor '" +
                    spec.quant.zero_point_tensor + "' is produced");
            }
        }
    }
}

std::string describe_load_plan(const LoadPlan& plan) {
    std::ostringstream out;
    out << plan.ops.size() << " load ops, "
        << plan.tensors.size() << " runtime tensor specs"
        << ", persistent=" << (plan.memory.persistent_bytes / (1024 * 1024))
        << " MiB"
        << ", temp<=" << (plan.memory.max_temporary_bytes / (1024 * 1024))
        << " MiB"
        << ", peak~=" << (plan.memory.estimated_peak_bytes / (1024 * 1024))
        << " MiB";
    if (plan.packed_qkv_groups > 0 || plan.packed_gate_up_groups > 0) {
        out << ", packed groups: qkv=" << plan.packed_qkv_groups
            << ", gate/up=" << plan.packed_gate_up_groups;
    }
    return out.str();
}

std::string dump_load_plan_json(const LoadPlan& plan) {
    validate_load_plan(plan);

    std::ostringstream out;
    out << "{\n";
    out << "  \"summary\": ";
    json_string(out, describe_load_plan(plan));
    out << ",\n";
    out << "  \"memory\": {"
        << "\"persistent_bytes\":" << plan.memory.persistent_bytes << ','
        << "\"max_temporary_bytes\":" << plan.memory.max_temporary_bytes << ','
        << "\"estimated_peak_bytes\":" << plan.memory.estimated_peak_bytes
        << "},\n";

    out << "  \"ops\": [";
    for (std::size_t i = 0; i < plan.ops.size(); ++i) {
        const LoadOp& op = plan.ops[i];
        if (i) out << ',';
        out << "\n    {\"kind\":";
        json_string(out, load_op_kind_name(op.kind));
        out << ",\"output_name\":";
        json_string(out, load_op_output(op));
        out << ",\"secondary_output_name\":";
        json_string(out, load_op_secondary_output(op));
        out << ",\"raw_name\":";
        json_string(out, load_op_raw_name(op));
        out << ",\"inputs\":[";
        for (std::size_t j = 0; j < load_op_inputs(op).size(); ++j) {
            if (j) out << ',';
            json_string(out, load_op_inputs(op)[j]);
        }
        out << "],\"shard_axis\":" << load_op_shard_axis(op)
            << ",\"slice_axis\":" << load_op_slice_axis(op)
            << ",\"row_offset\":" << load_op_row_offset(op)
            << ",\"rows\":" << load_op_rows(op)
            << ",\"slice_start\":" << load_op_slice_start(op)
            << ",\"slice_length\":" << load_op_slice_length(op);
        if (!load_op_row_sources(op).empty()) {
            out << ",\"row_sources\":[";
            for (std::size_t j = 0; j < load_op_row_sources(op).size(); ++j) {
                if (j) out << ',';
                out << "{\"raw_name\":";
                json_string(out, load_op_row_sources(op)[j].raw_name);
                out << ",\"view_name\":";
                json_string(out, load_op_row_sources(op)[j].view_name);
                out << '}';
            }
            out << ']';
        }
        out << '}';
    }
    if (!plan.ops.empty()) out << '\n';
    out << "  ],\n";

    out << "  \"tensors\": [";
    bool first = true;
    for (const auto& [name, spec] : plan.tensors) {
        if (!first) out << ',';
        first = false;
        out << "\n    {\"name\":";
        json_string(out, name);
        out << ",\"dtype\":";
        json_string(out, dtype_name(spec.dtype));
        out << ",\"shape\":";
        json_shape(out, spec.shape);
        out << ",\"layout\":";
        json_string(out, tensor_layout_kind_name(spec.layout));
        out << ",\"ownership\":";
        json_string(out, ownership_kind_name(spec.ownership));
        out << ",\"parallel\":";
        json_string(out, parallel_kind_name(spec.parallel));
        out << ",\"backing_tensor\":";
        json_string(out, spec.backing_tensor);
        out << ",\"quant\":{";
        out << "\"format\":";
        json_string(out, quant_format_name(spec.quant.format));
        out << ",\"granularity\":";
        json_string(out, quant_granularity_name(spec.quant.granularity));
        out << ",\"group_size\":" << spec.quant.group_size
            << ",\"channel_axis\":" << spec.quant.channel_axis
            << ",\"scale_tensor\":";
        json_string(out, spec.quant.scale_tensor);
        out << ",\"zero_point_tensor\":";
        json_string(out, spec.quant.zero_point_tensor);
        out << "}}";
    }
    if (!plan.tensors.empty()) out << '\n';
    out << "  ]\n";
    out << "}\n";
    return out.str();
}

}  // namespace pie_cuda_driver
