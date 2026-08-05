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

// Contiguous TP-local BF16 sections for Mixtral under tp_size>1.
// HF: w1/w3 [I,H] row-major, w2 [H,I]. Pack densifies the w2 column shard
// (strided in HF) so the streamer can page contiguous extents.
struct MixtralTpBf16PackTraits {
    static constexpr int kSections = 3;

    static const char* miss_label()
    {
        return "streaming Mixtral TP BF16 pack";
    }

    static void require_build_support() {}

    struct Context {
        int full_intermediate = 0;
        int local_start = 0;
        int local_intermediate = 0;
        int hidden = 0;
        std::uint64_t w1_span = 0;
        std::uint64_t w2_span = 0;
        std::uint64_t w3_span = 0;
        DeviceBuf src_w1;
        DeviceBuf src_w2;
        DeviceBuf src_w3;
        DeviceBuf dst_w1;
        DeviceBuf dst_w2;
        DeviceBuf dst_w3;
    };

    static Context prepare(
        const StreamedExpertTable& table,
        SafetensorsCheckpointSource& checkpoint)
    {
        const int tp = std::max(1, table.tp_size);
        const int rank = table.tp_rank;
        if (tp <= 1) {
            throw std::runtime_error(
                "expert pack: MixtralTpBf16 requires tp_size>1");
        }
        const auto& sb = table.section_bytes;

        const std::string w1_name =
            "model.layers.0.block_sparse_moe.experts.0.w1.weight";
        const std::string w2_name =
            "model.layers.0.block_sparse_moe.experts.0.w2.weight";
        const std::string w3_name =
            "model.layers.0.block_sparse_moe.experts.0.w3.weight";
        const auto& w1_info = checkpoint.info(w1_name);
        const auto& w2_info = checkpoint.info(w2_name);
        const auto& w3_info = checkpoint.info(w3_name);
        if (w1_info.shape.size() != 2 || w2_info.shape.size() != 2 ||
            w3_info.shape.size() != 2) {
            throw std::runtime_error(
                "expert pack: unexpected Mixtral expert weight ranks");
        }

        Context ctx;
        ctx.full_intermediate = static_cast<int>(w1_info.shape[0]);
        ctx.hidden = static_cast<int>(w1_info.shape[1]);
        if (static_cast<int>(w3_info.shape[0]) != ctx.full_intermediate ||
            static_cast<int>(w3_info.shape[1]) != ctx.hidden) {
            throw std::runtime_error(
                "expert pack: Mixtral w3 shape must match w1");
        }
        if (static_cast<int>(w2_info.shape[0]) != ctx.hidden ||
            static_cast<int>(w2_info.shape[1]) != ctx.full_intermediate) {
            throw std::runtime_error(
                "expert pack: Mixtral w2 expected [H, I]");
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
        const std::uint64_t w1_bytes = i_local * h * 2;
        const std::uint64_t w2_bytes = h * i_local * 2;
        const std::uint64_t expect[kSections] = {w1_bytes, w2_bytes, w1_bytes};
        if (sb.size() != static_cast<std::size_t>(kSections)) {
            throw std::runtime_error(
                "expert pack: Mixtral TP expected " +
                std::to_string(kSections) + " section_bytes, got " +
                std::to_string(sb.size()));
        }
        for (int i = 0; i < kSections; ++i) {
            if (sb[static_cast<std::size_t>(i)] != expect[i]) {
                throw std::runtime_error(
                    "expert pack: Mixtral TP section_bytes[" +
                    std::to_string(i) + "]=" +
                    std::to_string(sb[static_cast<std::size_t>(i)]) +
                    " != expected " + std::to_string(expect[i]) +
                    " (I_local=" + std::to_string(ctx.local_intermediate) +
                    " H=" + std::to_string(ctx.hidden) + ")");
            }
        }

        ctx.w1_span = checkpoint.storage_info(w1_name).nbytes;
        ctx.w2_span = checkpoint.storage_info(w2_name).nbytes;
        ctx.w3_span = checkpoint.storage_info(w3_name).nbytes;
        ctx.src_w1 = DeviceBuf(ctx.w1_span);
        ctx.src_w2 = DeviceBuf(ctx.w2_span);
        ctx.src_w3 = DeviceBuf(ctx.w3_span);
        ctx.dst_w1 = DeviceBuf(sb[0]);
        ctx.dst_w2 = DeviceBuf(sb[1]);
        ctx.dst_w3 = DeviceBuf(sb[2]);
        return ctx;
    }

    static void load_expert(
        Context& ctx,
        SafetensorsCheckpointSource& checkpoint,
        int layer,
        int expert)
    {
        const std::string p =
            "model.layers." + std::to_string(layer) +
            ".block_sparse_moe.experts." + std::to_string(expert) + ".";
        const auto w1 = checkpoint.storage_info(p + "w1.weight");
        const auto w2 = checkpoint.storage_info(p + "w2.weight");
        const auto w3 = checkpoint.storage_info(p + "w3.weight");
        checkpoint.copy_storage_bytes_to_device(
            w1.shard_id, w1.file_offset, ctx.w1_span, ctx.src_w1.ptr);
        checkpoint.copy_storage_bytes_to_device(
            w2.shard_id, w2.file_offset, ctx.w2_span, ctx.src_w2.ptr);
        checkpoint.copy_storage_bytes_to_device(
            w3.shard_id, w3.file_offset, ctx.w3_span, ctx.src_w3.ptr);
    }

    static void transform(Context& ctx)
    {
        // w1/w3: [I, H] BF16 — contiguous row slice for local intermediate.
        const std::size_t row_bytes =
            static_cast<std::size_t>(ctx.hidden) * 2;
        const std::size_t row_offset =
            static_cast<std::size_t>(ctx.local_start) * row_bytes;
        const std::size_t copy_bytes =
            static_cast<std::size_t>(ctx.local_intermediate) * row_bytes;
        CUDA_CHECK(cudaMemcpy(
            ctx.dst_w1.ptr,
            static_cast<const std::uint8_t*>(ctx.src_w1.ptr) + row_offset,
            copy_bytes, cudaMemcpyDeviceToDevice));
        CUDA_CHECK(cudaMemcpy(
            ctx.dst_w3.ptr,
            static_cast<const std::uint8_t*>(ctx.src_w3.ptr) + row_offset,
            copy_bytes, cudaMemcpyDeviceToDevice));

        // w2: [H, I] BF16 — gather local columns into a dense [H, I_local].
        const std::size_t full_row_bytes =
            static_cast<std::size_t>(ctx.full_intermediate) * 2;
        const std::size_t local_row_bytes =
            static_cast<std::size_t>(ctx.local_intermediate) * 2;
        const std::size_t col_offset =
            static_cast<std::size_t>(ctx.local_start) * 2;
        CUDA_CHECK(cudaMemcpy2D(
            ctx.dst_w2.ptr, local_row_bytes,
            static_cast<const std::uint8_t*>(ctx.src_w2.ptr) + col_offset,
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
            &ctx.dst_w1,
            &ctx.dst_w2,
            &ctx.dst_w3,
        };
        expert_pack_emit_slot_sections(
            writer, table, sections, kSections, host_bounce);
    }
};

}  // namespace

bool ensure_mixtral_tp_bf16_expert_pack(
    StreamedExpertTable& table,
    const std::string& cache_key,
    SafetensorsCheckpointSource& checkpoint,
    bool verbose)
{
    return ensure_expert_pack<MixtralTpBf16PackTraits>(
        table, cache_key, checkpoint, verbose);
}

}  // namespace pie_cuda_driver
