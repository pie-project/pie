#include "model/expert_pack_build.hpp"
#include "model/expert_pack_cache.hpp"

#include <algorithm>
#include <stdexcept>
#include <string>
#include <vector>

#include "cuda_check.hpp"
#include "kernels/mxfp4_marlin.hpp"
#include "tensor.hpp"

#if defined(PIE_CUDA_HAS_MARLIN)
#include "marlin_wrapper.hpp"
#endif

namespace pie_cuda_driver {
namespace {

int align_up_int(int v, int a)
{
    return (v + a - 1) / a * a;
}

void repack_weight_one(
    const std::uint8_t* src,
    std::uint8_t* dst,
    std::uint8_t* gptq_stage,
    int source_rows,
    int source_row_offset,
    int target_rows,
    int valid_rows,
    int source_stride_k,
    int source_col_offset,
    int source_k,
    int target_k,
    kernels::Mxfp4RowSelect row_map)
{
#if defined(PIE_CUDA_HAS_MARLIN)
    kernels::launch_mxfp4_weight_to_gptq_w4(
        src, gptq_stage,
        source_rows, source_row_offset, target_rows, valid_rows,
        source_stride_k, source_col_offset, source_k, target_k,
        row_map, /*stream=*/0);
    marlin::launch_gptq_repack_w4_no_perm(
        gptq_stage, dst, target_k, target_rows, /*stream=*/0);
    CUDA_CHECK(cudaGetLastError());
#else
    (void)src; (void)dst; (void)gptq_stage;
    (void)source_rows; (void)source_row_offset;
    (void)target_rows; (void)valid_rows;
    (void)source_stride_k; (void)source_col_offset;
    (void)source_k; (void)target_k; (void)row_map;
    throw std::runtime_error(
        "expert pack: Marlin repack requires PIE_CUDA_HAS_MARLIN");
#endif
}

void repack_scale_one(
    const std::uint8_t* src,
    std::uint8_t* dst,
    int source_rows,
    int source_row_offset,
    int target_rows,
    int valid_rows,
    int source_stride_groups,
    int source_group_offset,
    int source_groups,
    int target_groups,
    kernels::Mxfp4RowSelect row_map)
{
    kernels::launch_mxfp4_scales_to_marlin_e8m0(
        src, dst,
        source_rows, source_row_offset, target_rows, valid_rows,
        source_stride_groups, source_group_offset, source_groups,
        target_groups, row_map, /*stream=*/0);
    CUDA_CHECK(cudaGetLastError());
}

struct GptOssNativeMarlinPackTraits {
    static constexpr int kSections = 6;

    static const char* miss_label()
    {
        return "streaming Marlin build";
    }

    static void require_build_support()
    {
#if !defined(PIE_CUDA_HAS_MARLIN)
        throw std::runtime_error(
            "expert pack: native GPT-OSS pack build requires Marlin");
#endif
    }

    struct Context {
        int fused_rows = 0;
        int full_intermediate = 0;
        int local_start = 0;
        int local_intermediate = 0;
        int intermediate_native = 0;
        int hidden = 0;
        int gu_groups = 0;
        int down_groups = 0;
        std::uint64_t gu_w_span = 0;
        std::uint64_t gu_s_span = 0;
        std::uint64_t dn_w_span = 0;
        std::uint64_t dn_s_span = 0;
        DeviceBuf src_gu_w;
        DeviceBuf src_gu_s;
        DeviceBuf src_dn_w;
        DeviceBuf src_dn_s;
        DeviceBuf dst_gate_w;
        DeviceBuf dst_gate_s;
        DeviceBuf dst_up_w;
        DeviceBuf dst_up_s;
        DeviceBuf dst_dn_w;
        DeviceBuf dst_dn_s;
        DeviceBuf gptq_stage;
    };

    static Context prepare(
        const StreamedExpertTable& table,
        SafetensorsCheckpointSource& checkpoint)
    {
        const int E = table.num_experts;
        const int tp = std::max(1, table.tp_size);
        const int rank = table.tp_rank;
        const auto& sb = table.section_bytes;

        const std::string gu_w_name =
            "model.layers.0.mlp.experts.gate_up_proj_blocks";
        const std::string gu_s_name =
            "model.layers.0.mlp.experts.gate_up_proj_scales";
        const std::string dn_w_name =
            "model.layers.0.mlp.experts.down_proj_blocks";
        const std::string dn_s_name =
            "model.layers.0.mlp.experts.down_proj_scales";
        const auto& gu_w_info = checkpoint.info(gu_w_name);
        const auto& dn_w_info = checkpoint.info(dn_w_name);
        if (gu_w_info.shape.size() != 4 || dn_w_info.shape.size() != 4) {
            throw std::runtime_error(
                "expert pack: unexpected GPT-OSS expert block ranks");
        }

        Context ctx;
        ctx.fused_rows = static_cast<int>(gu_w_info.shape[1]);
        ctx.gu_groups = static_cast<int>(gu_w_info.shape[2]);
        ctx.full_intermediate = ctx.fused_rows / 2;
        if (ctx.full_intermediate % tp != 0) {
            throw std::runtime_error(
                "expert pack: intermediate not divisible by tp_size");
        }
        ctx.local_intermediate = ctx.full_intermediate / tp;
        ctx.local_start = rank * ctx.local_intermediate;
        if (ctx.local_start % 32 != 0 || ctx.local_intermediate % 32 != 0) {
            throw std::runtime_error(
                "expert pack: native Marlin TP shard must align to 32");
        }
        ctx.intermediate_native = align_up_int(ctx.local_intermediate, 128);
        ctx.hidden = ctx.gu_groups * 32;
        ctx.down_groups = static_cast<int>(dn_w_info.shape[2]);

        // Match Rust gpt_oss_native_section_bytes / stream plan layout.
        const std::uint64_t gate_w =
            static_cast<std::uint64_t>(ctx.intermediate_native) *
            static_cast<std::uint64_t>(ctx.hidden) / 2;
        const std::uint64_t gate_s =
            static_cast<std::uint64_t>(ctx.intermediate_native) *
            static_cast<std::uint64_t>(ctx.gu_groups);
        const std::uint64_t down_w =
            static_cast<std::uint64_t>(ctx.hidden) *
            static_cast<std::uint64_t>(ctx.intermediate_native) / 2;
        const std::uint64_t down_s =
            static_cast<std::uint64_t>(ctx.hidden) *
            static_cast<std::uint64_t>(ctx.intermediate_native / 32);
        const std::uint64_t expect[kSections] = {
            gate_w, gate_s, gate_w, gate_s, down_w, down_s};
        if (sb.size() != static_cast<std::size_t>(kSections)) {
            throw std::runtime_error(
                "expert pack: native Marlin expected " +
                std::to_string(kSections) + " section_bytes, got " +
                std::to_string(sb.size()));
        }
        for (int i = 0; i < kSections; ++i) {
            if (sb[static_cast<std::size_t>(i)] != expect[i]) {
                throw std::runtime_error(
                    "expert pack: native Marlin section_bytes[" +
                    std::to_string(i) + "]=" +
                    std::to_string(sb[static_cast<std::size_t>(i)]) +
                    " != expected " + std::to_string(expect[i]) +
                    " (I_native=" + std::to_string(ctx.intermediate_native) +
                    " H=" + std::to_string(ctx.hidden) +
                    " groups=" + std::to_string(ctx.gu_groups) +
                    " tp=" + std::to_string(tp) + ")");
            }
        }

        const auto gu_w_store = checkpoint.storage_info(gu_w_name);
        const auto gu_s_store = checkpoint.storage_info(gu_s_name);
        const auto dn_w_store = checkpoint.storage_info(dn_w_name);
        const auto dn_s_store = checkpoint.storage_info(dn_s_name);
        ctx.gu_w_span = gu_w_store.nbytes / static_cast<std::uint64_t>(E);
        ctx.gu_s_span = gu_s_store.nbytes / static_cast<std::uint64_t>(E);
        ctx.dn_w_span = dn_w_store.nbytes / static_cast<std::uint64_t>(E);
        ctx.dn_s_span = dn_s_store.nbytes / static_cast<std::uint64_t>(E);

        ctx.src_gu_w = DeviceBuf(ctx.gu_w_span);
        ctx.src_gu_s = DeviceBuf(ctx.gu_s_span);
        ctx.src_dn_w = DeviceBuf(ctx.dn_w_span);
        ctx.src_dn_s = DeviceBuf(ctx.dn_s_span);
        ctx.dst_gate_w = DeviceBuf(sb[0]);
        ctx.dst_gate_s = DeviceBuf(sb[1]);
        ctx.dst_up_w = DeviceBuf(sb[2]);
        ctx.dst_up_s = DeviceBuf(sb[3]);
        ctx.dst_dn_w = DeviceBuf(sb[4]);
        ctx.dst_dn_s = DeviceBuf(sb[5]);
        const std::uint64_t gptq_bytes = std::max(
            std::max(sb[0], sb[4]),
            static_cast<std::uint64_t>(ctx.intermediate_native) *
                static_cast<std::uint64_t>(ctx.hidden) / 2);
        ctx.gptq_stage = DeviceBuf(gptq_bytes);
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
        auto* gu_w_ptr = static_cast<std::uint8_t*>(ctx.src_gu_w.ptr);
        auto* gu_s_ptr = static_cast<std::uint8_t*>(ctx.src_gu_s.ptr);
        auto* dn_w_ptr = static_cast<std::uint8_t*>(ctx.src_dn_w.ptr);
        auto* dn_s_ptr = static_cast<std::uint8_t*>(ctx.src_dn_s.ptr);

        // Gate (even rows) / up (odd rows); source_row_offset = TP local_start.
        repack_weight_one(
            gu_w_ptr, static_cast<std::uint8_t*>(ctx.dst_gate_w.ptr),
            static_cast<std::uint8_t*>(ctx.gptq_stage.ptr),
            ctx.fused_rows, ctx.local_start, ctx.intermediate_native,
            ctx.local_intermediate, ctx.hidden, /*source_col_offset=*/0,
            ctx.hidden, ctx.hidden, kernels::Mxfp4RowSelect::Even);
        repack_scale_one(
            gu_s_ptr, static_cast<std::uint8_t*>(ctx.dst_gate_s.ptr),
            ctx.fused_rows, ctx.local_start, ctx.intermediate_native,
            ctx.local_intermediate, ctx.gu_groups, /*source_group_offset=*/0,
            ctx.gu_groups, ctx.gu_groups, kernels::Mxfp4RowSelect::Even);

        repack_weight_one(
            gu_w_ptr, static_cast<std::uint8_t*>(ctx.dst_up_w.ptr),
            static_cast<std::uint8_t*>(ctx.gptq_stage.ptr),
            ctx.fused_rows, ctx.local_start, ctx.intermediate_native,
            ctx.local_intermediate, ctx.hidden, /*source_col_offset=*/0,
            ctx.hidden, ctx.hidden, kernels::Mxfp4RowSelect::Odd);
        repack_scale_one(
            gu_s_ptr, static_cast<std::uint8_t*>(ctx.dst_up_s.ptr),
            ctx.fused_rows, ctx.local_start, ctx.intermediate_native,
            ctx.local_intermediate, ctx.gu_groups, /*source_group_offset=*/0,
            ctx.gu_groups, ctx.gu_groups, kernels::Mxfp4RowSelect::Odd);

        // Down: Identity rows; source_col_offset = TP local_start.
        repack_weight_one(
            dn_w_ptr, static_cast<std::uint8_t*>(ctx.dst_dn_w.ptr),
            static_cast<std::uint8_t*>(ctx.gptq_stage.ptr),
            ctx.hidden, /*source_row_offset=*/0, ctx.hidden, ctx.hidden,
            ctx.full_intermediate, ctx.local_start, ctx.local_intermediate,
            ctx.intermediate_native, kernels::Mxfp4RowSelect::Identity);
        repack_scale_one(
            dn_s_ptr, static_cast<std::uint8_t*>(ctx.dst_dn_s.ptr),
            ctx.hidden, /*source_row_offset=*/0, ctx.hidden, ctx.hidden,
            ctx.down_groups, ctx.local_start / 32, ctx.local_intermediate / 32,
            ctx.intermediate_native / 32, kernels::Mxfp4RowSelect::Identity);

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    static void emit(
        Context& ctx,
        ExpertPackWriter& writer,
        const StreamedExpertTable& table,
        std::vector<std::uint8_t>& host_bounce)
    {
        const DeviceBuf* sections[kSections] = {
            &ctx.dst_gate_w,
            &ctx.dst_gate_s,
            &ctx.dst_up_w,
            &ctx.dst_up_s,
            &ctx.dst_dn_w,
            &ctx.dst_dn_s,
        };
        expert_pack_emit_slot_sections(
            writer, table, sections, kSections, host_bounce);
    }
};

}  // namespace

bool ensure_gpt_oss_native_expert_pack(
    StreamedExpertTable& table,
    const std::string& cache_key,
    SafetensorsCheckpointSource& checkpoint,
    bool verbose)
{
    return ensure_expert_pack<GptOssNativeMarlinPackTraits>(
        table, cache_key, checkpoint, verbose);
}

}  // namespace pie_cuda_driver
