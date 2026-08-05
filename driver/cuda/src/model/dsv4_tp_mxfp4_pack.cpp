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

// Contiguous TP-local MXFP4 sections for DeepSeek-V4 under tp_size>1.
// HF: w1/w3 [I, H/2] + scale [I, H/32]; w2 [H, I/2] + scale [H, I/32].
// Pack densifies w2 columns (strided in HF) so the streamer can page
// contiguous extents matching resident Partition layout.
struct Dsv4TpMxfp4PackTraits {
    static constexpr int kSections = 6;

    static const char* miss_label()
    {
        return "streaming DeepSeek-V4 TP MXFP4 pack";
    }

    static void require_build_support() {}

    struct Context {
        int full_intermediate = 0;
        int local_start = 0;
        int local_intermediate = 0;
        int hidden = 0;
        std::uint64_t w1_span = 0;
        std::uint64_t w1s_span = 0;
        std::uint64_t w2_span = 0;
        std::uint64_t w2s_span = 0;
        std::uint64_t w3_span = 0;
        std::uint64_t w3s_span = 0;
        DeviceBuf src_w1;
        DeviceBuf src_w1s;
        DeviceBuf src_w2;
        DeviceBuf src_w2s;
        DeviceBuf src_w3;
        DeviceBuf src_w3s;
        DeviceBuf dst_w1;
        DeviceBuf dst_w1s;
        DeviceBuf dst_w2;
        DeviceBuf dst_w2s;
        DeviceBuf dst_w3;
        DeviceBuf dst_w3s;
    };

    static Context prepare(
        const StreamedExpertTable& table,
        SafetensorsCheckpointSource& checkpoint)
    {
        const int tp = std::max(1, table.tp_size);
        const int rank = table.tp_rank;
        if (tp <= 1) {
            throw std::runtime_error(
                "expert pack: Dsv4TpMxfp4 requires tp_size>1");
        }
        const auto& sb = table.section_bytes;

        const std::string w1_name = "layers.0.ffn.experts.0.w1.weight";
        const std::string w1s_name = "layers.0.ffn.experts.0.w1.scale";
        const std::string w2_name = "layers.0.ffn.experts.0.w2.weight";
        const std::string w2s_name = "layers.0.ffn.experts.0.w2.scale";
        const std::string w3_name = "layers.0.ffn.experts.0.w3.weight";
        const std::string w3s_name = "layers.0.ffn.experts.0.w3.scale";
        const auto& w1_info = checkpoint.info(w1_name);
        const auto& w2_info = checkpoint.info(w2_name);
        if (w1_info.shape.size() != 2 || w2_info.shape.size() != 2) {
            throw std::runtime_error(
                "expert pack: unexpected DeepSeek-V4 expert weight ranks");
        }

        Context ctx;
        ctx.full_intermediate = static_cast<int>(w1_info.shape[0]);
        ctx.hidden = static_cast<int>(w1_info.shape[1]) * 2;
        if (ctx.hidden % 32 != 0) {
            throw std::runtime_error(
                "expert pack: DeepSeek-V4 hidden must be divisible by 32");
        }
        if (static_cast<int>(w2_info.shape[0]) != ctx.hidden ||
            static_cast<int>(w2_info.shape[1]) != ctx.full_intermediate / 2) {
            throw std::runtime_error(
                "expert pack: DeepSeek-V4 w2 expected [H, I/2]");
        }
        if (ctx.full_intermediate % tp != 0) {
            throw std::runtime_error(
                "expert pack: intermediate not divisible by tp_size");
        }
        ctx.local_intermediate = ctx.full_intermediate / tp;
        ctx.local_start = rank * ctx.local_intermediate;
        if (ctx.local_intermediate % 32 != 0) {
            throw std::runtime_error(
                "expert pack: DeepSeek-V4 I_local must be divisible by 32");
        }

        const std::uint64_t i_local =
            static_cast<std::uint64_t>(ctx.local_intermediate);
        const std::uint64_t h = static_cast<std::uint64_t>(ctx.hidden);
        const std::uint64_t w13 = i_local * h / 2;
        const std::uint64_t s13 = i_local * h / 32;
        const std::uint64_t w2b = h * i_local / 2;
        const std::uint64_t s2 = h * i_local / 32;
        const std::uint64_t expect[kSections] = {w13, s13, w2b, s2, w13, s13};
        if (sb.size() != static_cast<std::size_t>(kSections)) {
            throw std::runtime_error(
                "expert pack: DeepSeek-V4 TP expected " +
                std::to_string(kSections) + " section_bytes, got " +
                std::to_string(sb.size()));
        }
        for (int i = 0; i < kSections; ++i) {
            if (sb[static_cast<std::size_t>(i)] != expect[i]) {
                throw std::runtime_error(
                    "expert pack: DeepSeek-V4 TP section_bytes[" +
                    std::to_string(i) + "]=" +
                    std::to_string(sb[static_cast<std::size_t>(i)]) +
                    " != expected " + std::to_string(expect[i]) +
                    " (I_local=" + std::to_string(ctx.local_intermediate) +
                    " H=" + std::to_string(ctx.hidden) + ")");
            }
        }

        ctx.w1_span = checkpoint.storage_info(w1_name).nbytes;
        ctx.w1s_span = checkpoint.storage_info(w1s_name).nbytes;
        ctx.w2_span = checkpoint.storage_info(w2_name).nbytes;
        ctx.w2s_span = checkpoint.storage_info(w2s_name).nbytes;
        ctx.w3_span = checkpoint.storage_info(w3_name).nbytes;
        ctx.w3s_span = checkpoint.storage_info(w3s_name).nbytes;
        ctx.src_w1 = DeviceBuf(ctx.w1_span);
        ctx.src_w1s = DeviceBuf(ctx.w1s_span);
        ctx.src_w2 = DeviceBuf(ctx.w2_span);
        ctx.src_w2s = DeviceBuf(ctx.w2s_span);
        ctx.src_w3 = DeviceBuf(ctx.w3_span);
        ctx.src_w3s = DeviceBuf(ctx.w3s_span);
        ctx.dst_w1 = DeviceBuf(sb[0]);
        ctx.dst_w1s = DeviceBuf(sb[1]);
        ctx.dst_w2 = DeviceBuf(sb[2]);
        ctx.dst_w2s = DeviceBuf(sb[3]);
        ctx.dst_w3 = DeviceBuf(sb[4]);
        ctx.dst_w3s = DeviceBuf(sb[5]);
        return ctx;
    }

    static void load_expert(
        Context& ctx,
        SafetensorsCheckpointSource& checkpoint,
        int layer,
        int expert)
    {
        const std::string p =
            "layers." + std::to_string(layer) + ".ffn.experts." +
            std::to_string(expert) + ".";
        const auto w1 = checkpoint.storage_info(p + "w1.weight");
        const auto w1s = checkpoint.storage_info(p + "w1.scale");
        const auto w2 = checkpoint.storage_info(p + "w2.weight");
        const auto w2s = checkpoint.storage_info(p + "w2.scale");
        const auto w3 = checkpoint.storage_info(p + "w3.weight");
        const auto w3s = checkpoint.storage_info(p + "w3.scale");
        checkpoint.copy_storage_bytes_to_device(
            w1.shard_id, w1.file_offset, ctx.w1_span, ctx.src_w1.ptr);
        checkpoint.copy_storage_bytes_to_device(
            w1s.shard_id, w1s.file_offset, ctx.w1s_span, ctx.src_w1s.ptr);
        checkpoint.copy_storage_bytes_to_device(
            w2.shard_id, w2.file_offset, ctx.w2_span, ctx.src_w2.ptr);
        checkpoint.copy_storage_bytes_to_device(
            w2s.shard_id, w2s.file_offset, ctx.w2s_span, ctx.src_w2s.ptr);
        checkpoint.copy_storage_bytes_to_device(
            w3.shard_id, w3.file_offset, ctx.w3_span, ctx.src_w3.ptr);
        checkpoint.copy_storage_bytes_to_device(
            w3s.shard_id, w3s.file_offset, ctx.w3s_span, ctx.src_w3s.ptr);
    }

    static void transform(Context& ctx)
    {
        // w1/w3 weight: [I, H/2] — contiguous row slice.
        const std::size_t w_row = static_cast<std::size_t>(ctx.hidden) / 2;
        const std::size_t w_off =
            static_cast<std::size_t>(ctx.local_start) * w_row;
        const std::size_t w_copy =
            static_cast<std::size_t>(ctx.local_intermediate) * w_row;
        CUDA_CHECK(cudaMemcpy(
            ctx.dst_w1.ptr,
            static_cast<const std::uint8_t*>(ctx.src_w1.ptr) + w_off, w_copy,
            cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(
            ctx.dst_w3.ptr,
            static_cast<const std::uint8_t*>(ctx.src_w3.ptr) + w_off, w_copy,
            cudaMemcpyDeviceToDevice));

        // w1/w3 scale: [I, H/32] — contiguous row slice.
        const std::size_t s_row = static_cast<std::size_t>(ctx.hidden) / 32;
        const std::size_t s_off =
            static_cast<std::size_t>(ctx.local_start) * s_row;
        const std::size_t s_copy =
            static_cast<std::size_t>(ctx.local_intermediate) * s_row;
        CUDA_CHECK(cudaMemcpy(
            ctx.dst_w1s.ptr,
            static_cast<const std::uint8_t*>(ctx.src_w1s.ptr) + s_off, s_copy,
            cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(
            ctx.dst_w3s.ptr,
            static_cast<const std::uint8_t*>(ctx.src_w3s.ptr) + s_off, s_copy,
            cudaMemcpyDeviceToDevice));

        // w2 weight: [H, I/2] — gather local packed columns.
        const std::size_t w2_full = static_cast<std::size_t>(ctx.full_intermediate) / 2;
        const std::size_t w2_local =
            static_cast<std::size_t>(ctx.local_intermediate) / 2;
        const std::size_t w2_col =
            static_cast<std::size_t>(ctx.local_start) / 2;
        CUDA_CHECK(cudaMemcpy2D(
            ctx.dst_w2.ptr, w2_local,
            static_cast<const std::uint8_t*>(ctx.src_w2.ptr) + w2_col, w2_full,
            w2_local, static_cast<std::size_t>(ctx.hidden),
            cudaMemcpyDeviceToDevice));

        // w2 scale: [H, I/32] — gather local scale columns.
        const std::size_t s2_full = static_cast<std::size_t>(ctx.full_intermediate) / 32;
        const std::size_t s2_local =
            static_cast<std::size_t>(ctx.local_intermediate) / 32;
        const std::size_t s2_col =
            static_cast<std::size_t>(ctx.local_start) / 32;
        CUDA_CHECK(cudaMemcpy2D(
            ctx.dst_w2s.ptr, s2_local,
            static_cast<const std::uint8_t*>(ctx.src_w2s.ptr) + s2_col, s2_full,
            s2_local, static_cast<std::size_t>(ctx.hidden),
            cudaMemcpyDeviceToDevice));

        CUDA_CHECK(cudaDeviceSynchronize());
    }

    static void emit(
        Context& ctx,
        ExpertPackWriter& writer,
        const StreamedExpertTable& table,
        std::vector<std::uint8_t>& host_bounce)
    {
        const DeviceBuf* sections[kSections] = {
            &ctx.dst_w1,
            &ctx.dst_w1s,
            &ctx.dst_w2,
            &ctx.dst_w2s,
            &ctx.dst_w3,
            &ctx.dst_w3s,
        };
        expert_pack_emit_slot_sections(
            writer, table, sections, kSections, host_bounce);
    }
};

}  // namespace

bool ensure_dsv4_tp_mxfp4_expert_pack(
    StreamedExpertTable& table,
    const std::string& cache_key,
    SafetensorsCheckpointSource& checkpoint,
    bool verbose)
{
    return ensure_expert_pack<Dsv4TpMxfp4PackTraits>(
        table, cache_key, checkpoint, verbose);
}

}  // namespace pie_cuda_driver
