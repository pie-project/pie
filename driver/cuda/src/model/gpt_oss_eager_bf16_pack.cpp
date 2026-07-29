#include "model/expert_pack_build.hpp"
#include "model/expert_pack_cache.hpp"

#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_check.hpp"
#include "kernels/deinterleave.hpp"
#include "kernels/dequant_fp4.hpp"
#include "tensor.hpp"

namespace pie_cuda_driver {
namespace {

struct GptOssEagerBf16PackTraits {
    static constexpr int kSections = 3;

    static const char* miss_label()
    {
        return "streaming eager BF16 dequant";
    }

    static void require_build_support() {}

    struct Context {
        int intermediate = 0;
        int hidden = 0;
        std::uint64_t gu_w_span = 0;
        std::uint64_t gu_s_span = 0;
        std::uint64_t dn_w_span = 0;
        std::uint64_t dn_s_span = 0;
        DeviceBuf src_gu_w;
        DeviceBuf src_gu_s;
        DeviceBuf src_dn_w;
        DeviceBuf src_dn_s;
        DeviceBuf fused_bf16;
        DeviceBuf dst_gate;
        DeviceBuf dst_up;
        DeviceBuf dst_down;
    };

    static Context prepare(
        const StreamedExpertTable& table,
        SafetensorsCheckpointSource& checkpoint)
    {
        const int E = table.num_experts;
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
        const int fused_rows = static_cast<int>(gu_w_info.shape[1]);
        const int gu_groups = static_cast<int>(gu_w_info.shape[2]);

        Context ctx;
        ctx.intermediate = fused_rows / 2;
        ctx.hidden = gu_groups * 32;

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
        ctx.fused_bf16 = DeviceBuf(
            static_cast<std::uint64_t>(2 * ctx.intermediate) *
            static_cast<std::uint64_t>(ctx.hidden) * 2);
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
        kernels::launch_dequant_mxfp4_to_bf16(
            static_cast<const std::uint8_t*>(ctx.src_gu_w.ptr),
            static_cast<const std::uint8_t*>(ctx.src_gu_s.ptr),
            ctx.fused_bf16.ptr, 2 * ctx.intermediate, ctx.hidden,
            /*stream=*/0);
        kernels::launch_deinterleave_rows_bf16(
            ctx.fused_bf16.ptr, ctx.dst_gate.ptr, ctx.dst_up.ptr,
            ctx.intermediate, ctx.hidden, /*stream=*/0);
        kernels::launch_dequant_mxfp4_to_bf16(
            static_cast<const std::uint8_t*>(ctx.src_dn_w.ptr),
            static_cast<const std::uint8_t*>(ctx.src_dn_s.ptr),
            ctx.dst_down.ptr, ctx.hidden, ctx.intermediate, /*stream=*/0);
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

bool ensure_gpt_oss_eager_bf16_expert_pack(
    StreamedExpertTable& table,
    const std::string& cache_key,
    SafetensorsCheckpointSource& checkpoint,
    bool verbose)
{
    return ensure_expert_pack<GptOssEagerBf16PackTraits>(
        table, cache_key, checkpoint, verbose);
}

}  // namespace pie_cuda_driver
