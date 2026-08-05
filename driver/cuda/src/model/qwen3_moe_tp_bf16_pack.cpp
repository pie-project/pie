#include "model/expert_pack_build.hpp"
#include "model/expert_pack_cache.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_check.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver {
namespace {

// Contiguous TP-local BF16 sections for plain Qwen3-MoE under tp_size>1.
// HF: gate/up [I,H], down [H,I]. Pack densifies the down column shard
// (strided in HF) so the streamer can page contiguous extents.
struct Qwen3MoeTpBf16PackTraits {
    static constexpr int kSections = 3;

    static const char* miss_label()
    {
        return "streaming Qwen3-MoE TP BF16 pack";
    }

    static void require_build_support() {}

    struct Context {
        int full_intermediate = 0;
        int local_start = 0;
        int local_intermediate = 0;
        int hidden = 0;
        std::uint64_t gate_span = 0;
        std::uint64_t up_span = 0;
        std::uint64_t down_span = 0;
        DeviceBuf src_gate;
        DeviceBuf src_up;
        DeviceBuf src_down;
        DeviceBuf dst_gate;
        DeviceBuf dst_up;
        DeviceBuf dst_down;
    };

    static Context prepare(
        const StreamedExpertTable& table,
        SafetensorsCheckpointSource& checkpoint)
    {
        const int tp = std::max(1, table.tp_size);
        const int rank = table.tp_rank;
        if (tp <= 1) {
            throw std::runtime_error(
                "expert pack: Qwen3MoeTpBf16 requires tp_size>1");
        }
        const auto& sb = table.section_bytes;

        const std::string gate_name =
            "model.layers.0.mlp.experts.0.gate_proj.weight";
        const std::string up_name =
            "model.layers.0.mlp.experts.0.up_proj.weight";
        const std::string down_name =
            "model.layers.0.mlp.experts.0.down_proj.weight";
        const auto& gate_info = checkpoint.info(gate_name);
        const auto& up_info = checkpoint.info(up_name);
        const auto& down_info = checkpoint.info(down_name);
        if (gate_info.shape.size() != 2 || up_info.shape.size() != 2 ||
            down_info.shape.size() != 2) {
            throw std::runtime_error(
                "expert pack: unexpected Qwen3-MoE expert weight ranks");
        }

        Context ctx;
        ctx.full_intermediate = static_cast<int>(gate_info.shape[0]);
        ctx.hidden = static_cast<int>(gate_info.shape[1]);
        if (static_cast<int>(up_info.shape[0]) != ctx.full_intermediate ||
            static_cast<int>(up_info.shape[1]) != ctx.hidden) {
            throw std::runtime_error(
                "expert pack: Qwen3-MoE up shape must match gate");
        }
        if (static_cast<int>(down_info.shape[0]) != ctx.hidden ||
            static_cast<int>(down_info.shape[1]) != ctx.full_intermediate) {
            throw std::runtime_error(
                "expert pack: Qwen3-MoE down expected [H, I]");
        }
        if (ctx.full_intermediate % tp != 0) {
            throw std::runtime_error(
                "expert pack: intermediate not divisible by tp_size");
        }
        ctx.local_intermediate = ctx.full_intermediate / tp;
        ctx.local_start = rank * ctx.local_intermediate;

        const std::uint64_t i_local =
            static_cast<std::uint64_t>(ctx.local_intermediate);
        const std::uint64_t h = static_cast<std::uint64_t>(ctx.hidden);
        const std::uint64_t gate_bytes = i_local * h * 2;
        const std::uint64_t down_bytes = h * i_local * 2;
        const std::uint64_t expect[kSections] = {
            gate_bytes, gate_bytes, down_bytes};
        if (sb.size() != static_cast<std::size_t>(kSections)) {
            throw std::runtime_error(
                "expert pack: Qwen3-MoE TP expected " +
                std::to_string(kSections) + " section_bytes, got " +
                std::to_string(sb.size()));
        }
        for (int i = 0; i < kSections; ++i) {
            if (sb[static_cast<std::size_t>(i)] != expect[i]) {
                throw std::runtime_error(
                    "expert pack: Qwen3-MoE TP section_bytes[" +
                    std::to_string(i) + "]=" +
                    std::to_string(sb[static_cast<std::size_t>(i)]) +
                    " != expected " + std::to_string(expect[i]) +
                    " (I_local=" + std::to_string(ctx.local_intermediate) +
                    " H=" + std::to_string(ctx.hidden) + ")");
            }
        }

        ctx.gate_span = checkpoint.storage_info(gate_name).nbytes;
        ctx.up_span = checkpoint.storage_info(up_name).nbytes;
        ctx.down_span = checkpoint.storage_info(down_name).nbytes;
        ctx.src_gate = DeviceBuf(ctx.gate_span);
        ctx.src_up = DeviceBuf(ctx.up_span);
        ctx.src_down = DeviceBuf(ctx.down_span);
        ctx.dst_gate = DeviceBuf(sb[0]);
        ctx.dst_up = DeviceBuf(sb[1]);
        ctx.dst_down = DeviceBuf(sb[2]);
        return ctx;
    }

    static void load_expert(
        Context& ctx,
        SafetensorsCheckpointSource& checkpoint,
        int layer,
        int expert)
    {
        const std::string p =
            "model.layers." + std::to_string(layer) + ".mlp.experts." +
            std::to_string(expert) + ".";
        const auto gate = checkpoint.storage_info(p + "gate_proj.weight");
        const auto up = checkpoint.storage_info(p + "up_proj.weight");
        const auto down = checkpoint.storage_info(p + "down_proj.weight");
        checkpoint.copy_storage_bytes_to_device(
            gate.shard_id, gate.file_offset, ctx.gate_span, ctx.src_gate.ptr);
        checkpoint.copy_storage_bytes_to_device(
            up.shard_id, up.file_offset, ctx.up_span, ctx.src_up.ptr);
        checkpoint.copy_storage_bytes_to_device(
            down.shard_id, down.file_offset, ctx.down_span, ctx.src_down.ptr);
    }

    static void transform(Context& ctx)
    {
        // gate/up: [I, H] BF16 — contiguous row slice for local intermediate.
        const std::size_t row_bytes =
            static_cast<std::size_t>(ctx.hidden) * 2;
        const std::size_t row_offset =
            static_cast<std::size_t>(ctx.local_start) * row_bytes;
        const std::size_t copy_bytes =
            static_cast<std::size_t>(ctx.local_intermediate) * row_bytes;
        CUDA_CHECK(cudaMemcpy(
            ctx.dst_gate.ptr,
            static_cast<const std::uint8_t*>(ctx.src_gate.ptr) + row_offset,
            copy_bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(
            ctx.dst_up.ptr,
            static_cast<const std::uint8_t*>(ctx.src_up.ptr) + row_offset,
            copy_bytes, cudaMemcpyDeviceToDevice));

        // down: [H, I] BF16 — gather local columns into a dense [H, I_local].
        const std::size_t full_row_bytes =
            static_cast<std::size_t>(ctx.full_intermediate) * 2;
        const std::size_t local_row_bytes =
            static_cast<std::size_t>(ctx.local_intermediate) * 2;
        const std::size_t col_offset =
            static_cast<std::size_t>(ctx.local_start) * 2;
        CUDA_CHECK(cudaMemcpy2D(
            ctx.dst_down.ptr, local_row_bytes,
            static_cast<const std::uint8_t*>(ctx.src_down.ptr) + col_offset,
            full_row_bytes, local_row_bytes,
            static_cast<std::size_t>(ctx.hidden), cudaMemcpyDeviceToDevice));

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    static void emit(
        Context& ctx,
        ExpertPackWriter& writer,
        const StreamedExpertTable& table,
        std::vector<std::uint8_t>& host_bounce)
    {
        const DeviceBuf* sections[kSections] = {
            &ctx.dst_gate,
            &ctx.dst_up,
            &ctx.dst_down,
        };
        expert_pack_emit_slot_sections(
            writer, table, sections, kSections, host_bounce);
    }
};

}  // namespace

bool ensure_qwen3_moe_tp_bf16_expert_pack(
    StreamedExpertTable& table,
    const std::string& cache_key,
    SafetensorsCheckpointSource& checkpoint,
    bool verbose)
{
    return ensure_expert_pack<Qwen3MoeTpBf16PackTraits>(
        table, cache_key, checkpoint, verbose);
}

}  // namespace pie_cuda_driver
