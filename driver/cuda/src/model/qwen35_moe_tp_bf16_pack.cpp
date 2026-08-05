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

// Contiguous TP-local BF16 sections for Qwen3.5-MoE fused banks under tp>1.
// HF: gate_up [E, 2I, H] (rows [0,I)=gate, [I,2I)=up), down [E, H, I].
// Pack densifies gate/up halves and down columns into contiguous local-I
// sections so the streamer can page them.
struct Qwen35MoeTpBf16PackTraits {
    static constexpr int kSections = 2;

    static const char* miss_label()
    {
        return "streaming Qwen3.5-MoE TP BF16 pack";
    }

    static void require_build_support() {}

    struct Context {
        int full_intermediate = 0;
        int local_start = 0;
        int local_intermediate = 0;
        int hidden = 0;
        int num_experts = 0;
        std::string prefix_root;
        std::uint64_t gu_span = 0;
        std::uint64_t dn_span = 0;
        DeviceBuf src_gu;
        DeviceBuf src_dn;
        DeviceBuf dst_gu;
        DeviceBuf dst_dn;
    };

    static std::string resolve_prefix_root(
        SafetensorsCheckpointSource& checkpoint)
    {
        const std::string lm =
            "model.language_model.layers.0.mlp.experts.gate_up_proj";
        if (checkpoint.contains(lm)) {
            return "model.language_model.layers.";
        }
        return "model.layers.";
    }

    static Context prepare(
        const StreamedExpertTable& table,
        SafetensorsCheckpointSource& checkpoint)
    {
        const int tp = std::max(1, table.tp_size);
        const int rank = table.tp_rank;
        if (tp <= 1) {
            throw std::runtime_error(
                "expert pack: Qwen35MoeTpBf16 requires tp_size>1");
        }
        const auto& sb = table.section_bytes;

        Context ctx;
        ctx.prefix_root = resolve_prefix_root(checkpoint);
        ctx.num_experts = table.num_experts;
        const std::string gu_name =
            ctx.prefix_root + "0.mlp.experts.gate_up_proj";
        const std::string dn_name =
            ctx.prefix_root + "0.mlp.experts.down_proj";
        const auto& gu_info = checkpoint.info(gu_name);
        const auto& dn_info = checkpoint.info(dn_name);
        if (gu_info.shape.size() != 3 || dn_info.shape.size() != 3) {
            throw std::runtime_error(
                "expert pack: unexpected Qwen3.5-MoE fused expert ranks");
        }
        if (static_cast<int>(gu_info.shape[0]) != ctx.num_experts ||
            static_cast<int>(dn_info.shape[0]) != ctx.num_experts) {
            throw std::runtime_error(
                "expert pack: Qwen3.5-MoE fused bank expert count mismatch");
        }
        if (gu_info.shape[1] % 2 != 0) {
            throw std::runtime_error(
                "expert pack: Qwen3.5-MoE gate_up expected [E, 2I, H]");
        }
        ctx.full_intermediate = static_cast<int>(gu_info.shape[1] / 2);
        ctx.hidden = static_cast<int>(gu_info.shape[2]);
        if (static_cast<int>(dn_info.shape[1]) != ctx.hidden ||
            static_cast<int>(dn_info.shape[2]) != ctx.full_intermediate) {
            throw std::runtime_error(
                "expert pack: Qwen3.5-MoE down expected [E, H, I]");
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
        const std::uint64_t gu_bytes = 2 * i_local * h * 2;
        const std::uint64_t dn_bytes = h * i_local * 2;
        const std::uint64_t expect[kSections] = {gu_bytes, dn_bytes};
        if (sb.size() != static_cast<std::size_t>(kSections)) {
            throw std::runtime_error(
                "expert pack: Qwen3.5-MoE TP expected " +
                std::to_string(kSections) + " section_bytes, got " +
                std::to_string(sb.size()));
        }
        for (int i = 0; i < kSections; ++i) {
            if (sb[static_cast<std::size_t>(i)] != expect[i]) {
                throw std::runtime_error(
                    "expert pack: Qwen3.5-MoE TP section_bytes[" +
                    std::to_string(i) + "]=" +
                    std::to_string(sb[static_cast<std::size_t>(i)]) +
                    " != expected " + std::to_string(expect[i]) +
                    " (I_local=" + std::to_string(ctx.local_intermediate) +
                    " H=" + std::to_string(ctx.hidden) + ")");
            }
        }

        const auto gu_store = checkpoint.storage_info(gu_name);
        const auto dn_store = checkpoint.storage_info(dn_name);
        ctx.gu_span = gu_store.nbytes / static_cast<std::uint64_t>(ctx.num_experts);
        ctx.dn_span = dn_store.nbytes / static_cast<std::uint64_t>(ctx.num_experts);
        ctx.src_gu = DeviceBuf(ctx.gu_span);
        ctx.src_dn = DeviceBuf(ctx.dn_span);
        ctx.dst_gu = DeviceBuf(sb[0]);
        ctx.dst_dn = DeviceBuf(sb[1]);
        return ctx;
    }

    static void load_expert(
        Context& ctx,
        SafetensorsCheckpointSource& checkpoint,
        int layer,
        int expert)
    {
        const std::string p =
            ctx.prefix_root + std::to_string(layer) + ".mlp.experts.";
        const auto gu = checkpoint.storage_info(p + "gate_up_proj");
        const auto dn = checkpoint.storage_info(p + "down_proj");
        const auto e = static_cast<std::uint64_t>(expert);
        checkpoint.copy_storage_bytes_to_device(
            gu.shard_id, gu.file_offset + e * ctx.gu_span, ctx.gu_span,
            ctx.src_gu.ptr);
        checkpoint.copy_storage_bytes_to_device(
            dn.shard_id, dn.file_offset + e * ctx.dn_span, ctx.dn_span,
            ctx.src_dn.ptr);
    }

    static void transform(Context& ctx)
    {
        // gate_up: [2I, H] BF16 — copy gate then up local row halves into
        // contiguous [2*I_local, H] (matches resident ByteSpan TP layout).
        const std::size_t row_bytes =
            static_cast<std::size_t>(ctx.hidden) * 2;
        const std::size_t half_bytes =
            static_cast<std::size_t>(ctx.local_intermediate) * row_bytes;
        const std::size_t gate_off =
            static_cast<std::size_t>(ctx.local_start) * row_bytes;
        const std::size_t up_off =
            static_cast<std::size_t>(ctx.full_intermediate + ctx.local_start) *
            row_bytes;
        auto* dst = static_cast<std::uint8_t*>(ctx.dst_gu.ptr);
        const auto* src = static_cast<const std::uint8_t*>(ctx.src_gu.ptr);
        CUDA_CHECK(cudaMemcpy(
            dst, src + gate_off, half_bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(
            dst + half_bytes, src + up_off, half_bytes,
            cudaMemcpyDeviceToDevice));

        // down: [H, I] BF16 — gather local columns into dense [H, I_local].
        const std::size_t full_row_bytes =
            static_cast<std::size_t>(ctx.full_intermediate) * 2;
        const std::size_t local_row_bytes =
            static_cast<std::size_t>(ctx.local_intermediate) * 2;
        const std::size_t col_offset =
            static_cast<std::size_t>(ctx.local_start) * 2;
        CUDA_CHECK(cudaMemcpy2D(
            ctx.dst_dn.ptr, local_row_bytes,
            static_cast<const std::uint8_t*>(ctx.src_dn.ptr) + col_offset,
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
            &ctx.dst_gu,
            &ctx.dst_dn,
        };
        expert_pack_emit_slot_sections(
            writer, table, sections, kSections, host_bounce);
    }
};

}  // namespace

bool ensure_qwen35_moe_tp_bf16_expert_pack(
    StreamedExpertTable& table,
    const std::string& cache_key,
    SafetensorsCheckpointSource& checkpoint,
    bool verbose)
{
    return ensure_expert_pack<Qwen35MoeTpBf16PackTraits>(
        table, cache_key, checkpoint, verbose);
}

}  // namespace pie_cuda_driver
