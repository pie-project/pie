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

// Contiguous TP-local HF MXFP4 sections for RoutedDecode under tp_size>1.
// Gate/up fused rows are contiguous; down groups are strided in the HF bank
// and must be gathered into a dense local buffer.
struct GptOssRoutedMxfp4PackTraits {
    static constexpr int kSections = 4;

    static const char* miss_label()
    {
        return "streaming RoutedDecode TP MXFP4 pack";
    }

    static void require_build_support() {}

    struct Context {
        int full_intermediate = 0;
        int local_start = 0;
        int local_intermediate = 0;
        int hidden = 0;
        int gu_groups = 0;
        int down_groups_full = 0;
        int down_groups_local = 0;
        std::uint64_t gu_w_span = 0;
        std::uint64_t gu_s_span = 0;
        std::uint64_t dn_w_span = 0;
        std::uint64_t dn_s_span = 0;
        DeviceBuf src_gu_w;
        DeviceBuf src_gu_s;
        DeviceBuf src_dn_w;
        DeviceBuf src_dn_s;
        DeviceBuf dst_gu_w;
        DeviceBuf dst_gu_s;
        DeviceBuf dst_dn_w;
        DeviceBuf dst_dn_s;
    };

    static Context prepare(
        const StreamedExpertTable& table,
        SafetensorsCheckpointSource& checkpoint)
    {
        const int E = table.num_experts;
        const int tp = std::max(1, table.tp_size);
        const int rank = table.tp_rank;
        if (tp <= 1) {
            throw std::runtime_error(
                "expert pack: GptOssRoutedMxfp4 requires tp_size>1");
        }
        const auto& sb = table.section_bytes;

        const std::string gu_w_name =
            "model.layers.0.mlp.experts.gate_up_proj_blocks";
        const std::string dn_w_name =
            "model.layers.0.mlp.experts.down_proj_blocks";
        const auto& gu_w_info = checkpoint.info(gu_w_name);
        const auto& dn_w_info = checkpoint.info(dn_w_name);
        if (gu_w_info.shape.size() != 4 || dn_w_info.shape.size() != 4) {
            throw std::runtime_error(
                "expert pack: unexpected GPT-OSS expert block ranks");
        }

        Context ctx;
        const int fused_rows = static_cast<int>(gu_w_info.shape[1]);
        ctx.gu_groups = static_cast<int>(gu_w_info.shape[2]);
        ctx.full_intermediate = fused_rows / 2;
        if (ctx.full_intermediate % tp != 0) {
            throw std::runtime_error(
                "expert pack: intermediate not divisible by tp_size");
        }
        ctx.local_intermediate = ctx.full_intermediate / tp;
        ctx.local_start = rank * ctx.local_intermediate;
        if (ctx.local_start % 32 != 0 || ctx.local_intermediate % 32 != 0) {
            throw std::runtime_error(
                "expert pack: RoutedDecode TP shard must align to 32");
        }
        ctx.hidden = ctx.gu_groups * 32;
        ctx.down_groups_full = static_cast<int>(dn_w_info.shape[2]);
        if (ctx.down_groups_full != ctx.full_intermediate / 32) {
            throw std::runtime_error(
                "expert pack: RoutedDecode down groups " +
                std::to_string(ctx.down_groups_full) +
                " != full_intermediate/32 (" +
                std::to_string(ctx.full_intermediate / 32) + ")");
        }
        ctx.down_groups_local = ctx.local_intermediate / 32;

        const std::uint64_t i_local =
            static_cast<std::uint64_t>(ctx.local_intermediate);
        const std::uint64_t h = static_cast<std::uint64_t>(ctx.hidden);
        const std::uint64_t gu_w = i_local * h;
        const std::uint64_t gu_s = 2 * i_local * (h / 32);
        const std::uint64_t dn_w = h * i_local / 2;
        const std::uint64_t dn_s = h * (i_local / 32);
        const std::uint64_t expect[kSections] = {gu_w, gu_s, dn_w, dn_s};
        if (sb.size() != static_cast<std::size_t>(kSections)) {
            throw std::runtime_error(
                "expert pack: RoutedDecode TP expected " +
                std::to_string(kSections) + " section_bytes, got " +
                std::to_string(sb.size()));
        }
        for (int i = 0; i < kSections; ++i) {
            if (sb[static_cast<std::size_t>(i)] != expect[i]) {
                throw std::runtime_error(
                    "expert pack: RoutedDecode TP section_bytes[" +
                    std::to_string(i) + "]=" +
                    std::to_string(sb[static_cast<std::size_t>(i)]) +
                    " != expected " + std::to_string(expect[i]) +
                    " (I_local=" + std::to_string(ctx.local_intermediate) +
                    " H=" + std::to_string(ctx.hidden) + ")");
            }
        }

        const auto gu_w_store = checkpoint.storage_info(gu_w_name);
        const auto gu_s_store = checkpoint.storage_info(
            "model.layers.0.mlp.experts.gate_up_proj_scales");
        const auto dn_w_store = checkpoint.storage_info(dn_w_name);
        const auto dn_s_store = checkpoint.storage_info(
            "model.layers.0.mlp.experts.down_proj_scales");
        ctx.gu_w_span = gu_w_store.nbytes / static_cast<std::uint64_t>(E);
        ctx.gu_s_span = gu_s_store.nbytes / static_cast<std::uint64_t>(E);
        ctx.dn_w_span = dn_w_store.nbytes / static_cast<std::uint64_t>(E);
        ctx.dn_s_span = dn_s_store.nbytes / static_cast<std::uint64_t>(E);

        ctx.src_gu_w = DeviceBuf(ctx.gu_w_span);
        ctx.src_gu_s = DeviceBuf(ctx.gu_s_span);
        ctx.src_dn_w = DeviceBuf(ctx.dn_w_span);
        ctx.src_dn_s = DeviceBuf(ctx.dn_s_span);
        ctx.dst_gu_w = DeviceBuf(sb[0]);
        ctx.dst_gu_s = DeviceBuf(sb[1]);
        ctx.dst_dn_w = DeviceBuf(sb[2]);
        ctx.dst_dn_s = DeviceBuf(sb[3]);
        return ctx;
    }

    static void load_expert(
        Context& ctx,
        SafetensorsCheckpointSource& checkpoint,
        int layer,
        int expert)
    {
        const std::string p =
            "model.layers." + std::to_string(layer) + ".mlp.experts.";
        const auto gu_w = checkpoint.storage_info(p + "gate_up_proj_blocks");
        const auto gu_s = checkpoint.storage_info(p + "gate_up_proj_scales");
        const auto dn_w = checkpoint.storage_info(p + "down_proj_blocks");
        const auto dn_s = checkpoint.storage_info(p + "down_proj_scales");
        const auto e = static_cast<std::uint64_t>(expert);
        checkpoint.copy_storage_bytes_to_device(
            gu_w.shard_id, gu_w.file_offset + e * ctx.gu_w_span, ctx.gu_w_span,
            ctx.src_gu_w.ptr);
        checkpoint.copy_storage_bytes_to_device(
            gu_s.shard_id, gu_s.file_offset + e * ctx.gu_s_span, ctx.gu_s_span,
            ctx.src_gu_s.ptr);
        checkpoint.copy_storage_bytes_to_device(
            dn_w.shard_id, dn_w.file_offset + e * ctx.dn_w_span, ctx.dn_w_span,
            ctx.src_dn_w.ptr);
        checkpoint.copy_storage_bytes_to_device(
            dn_s.shard_id, dn_s.file_offset + e * ctx.dn_s_span, ctx.dn_s_span,
            ctx.src_dn_s.ptr);
    }

    static void transform(Context& ctx)
    {
        // Gate/up fused weight: [2I, H/32, 16] → contiguous row slice.
        // Each fused row is (H/32)*16 = H/2 bytes.
        const std::size_t gu_row_bytes =
            static_cast<std::size_t>(ctx.hidden) / 2;
        const std::size_t gu_row_offset =
            static_cast<std::size_t>(2 * ctx.local_start) * gu_row_bytes;
        const std::size_t gu_copy_bytes =
            static_cast<std::size_t>(2 * ctx.local_intermediate) * gu_row_bytes;
        CUDA_CHECK(cudaMemcpy(
            ctx.dst_gu_w.ptr,
            static_cast<const std::uint8_t*>(ctx.src_gu_w.ptr) + gu_row_offset,
            gu_copy_bytes, cudaMemcpyDeviceToDevice));

        // Gate/up scales: [2I, H/32] u8 → contiguous row slice.
        const std::size_t gu_s_row_bytes =
            static_cast<std::size_t>(ctx.gu_groups);
        const std::size_t gu_s_offset =
            static_cast<std::size_t>(2 * ctx.local_start) * gu_s_row_bytes;
        const std::size_t gu_s_copy =
            static_cast<std::size_t>(2 * ctx.local_intermediate) *
            gu_s_row_bytes;
        CUDA_CHECK(cudaMemcpy(
            ctx.dst_gu_s.ptr,
            static_cast<const std::uint8_t*>(ctx.src_gu_s.ptr) + gu_s_offset,
            gu_s_copy, cudaMemcpyDeviceToDevice));

        // Down weight: [H, I/32, 16] — gather local groups per hidden row.
        // Each group is 16 bytes; full row pitch = down_groups_full * 16.
        const std::size_t dn_group_bytes = 16;
        const std::size_t dn_full_pitch =
            static_cast<std::size_t>(ctx.down_groups_full) * dn_group_bytes;
        const std::size_t dn_local_pitch =
            static_cast<std::size_t>(ctx.down_groups_local) * dn_group_bytes;
        const std::size_t dn_col_offset =
            static_cast<std::size_t>(ctx.local_start / 32) * dn_group_bytes;
        CUDA_CHECK(cudaMemcpy2D(
            ctx.dst_dn_w.ptr, dn_local_pitch,
            static_cast<const std::uint8_t*>(ctx.src_dn_w.ptr) + dn_col_offset,
            dn_full_pitch, dn_local_pitch,
            static_cast<std::size_t>(ctx.hidden), cudaMemcpyDeviceToDevice));

        // Down scales: [H, I/32] u8 — same gather pattern.
        const std::size_t dn_s_full_pitch =
            static_cast<std::size_t>(ctx.down_groups_full);
        const std::size_t dn_s_local_pitch =
            static_cast<std::size_t>(ctx.down_groups_local);
        const std::size_t dn_s_col_offset =
            static_cast<std::size_t>(ctx.local_start / 32);
        CUDA_CHECK(cudaMemcpy2D(
            ctx.dst_dn_s.ptr, dn_s_local_pitch,
            static_cast<const std::uint8_t*>(ctx.src_dn_s.ptr) +
                dn_s_col_offset,
            dn_s_full_pitch, dn_s_local_pitch,
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
            &ctx.dst_gu_w,
            &ctx.dst_gu_s,
            &ctx.dst_dn_w,
            &ctx.dst_dn_s,
        };
        expert_pack_emit_slot_sections(
            writer, table, sections, kSections, host_bounce);
    }
};

}  // namespace

bool ensure_gpt_oss_routed_mxfp4_expert_pack(
    StreamedExpertTable& table,
    const std::string& cache_key,
    SafetensorsCheckpointSource& checkpoint,
    bool verbose)
{
    return ensure_expert_pack<GptOssRoutedMxfp4PackTraits>(
        table, cache_key, checkpoint, verbose);
}

}  // namespace pie_cuda_driver
