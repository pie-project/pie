#include <algorithm>
#include <cstdint>
#include <cstdio>
#include <exception>
#include <string>
#include <unordered_map>
#include <vector>

#include <cuda_runtime.h>

#include "loader/dtype_map.hpp"
#include "loader/transcode_engine.hpp"

#ifndef PIE_CUDA_HAS_MARLIN_MXFP4_REPACK
#error "Marlin MXFP4 repack test requires the repack capability"
#endif

namespace {

namespace lp = pie_loader;
using namespace pie_cuda_driver;

void cuda_check(cudaError_t status, const char* operation)
{
    if (status != cudaSuccess) {
        throw std::runtime_error(
            std::string(operation) + ": " + cudaGetErrorString(status));
    }
}

}  // namespace

int main()
{
    try {
        constexpr int rows = 64;
        constexpr int cols = 128;
        constexpr std::size_t packed_bytes = rows * cols / 2;

        lp::PieLoaderPlan plan{};
        lp::CheckpointSource loader(plan);
        WeightCopyEngine copy_engine(loader);
        lp::LoadPlanIndex plan_index("marlin mxfp4 repack test");
        WeightStore store;
        WeightStoreBuilder weights(store);
        std::unordered_map<std::uint32_t, DeviceTensor> buffers;
        std::unordered_map<std::uint32_t, std::string> finalized_names;
        BufferResolver resolver{buffers, finalized_names, weights};
        TranscodeEngine transcode(loader, copy_engine, plan_index, resolver);

        std::vector<std::uint8_t> source(packed_bytes);
        for (std::size_t i = 0; i < source.size(); ++i) {
            source[i] = static_cast<std::uint8_t>((i * 29 + 17) & 0xff);
        }
        DeviceTensor input =
            DeviceTensor::allocate(DType::UINT8, {static_cast<std::int64_t>(packed_bytes)});
        cuda_check(cudaMemcpy(input.data(), source.data(), packed_bytes,
                              cudaMemcpyHostToDevice),
                   "copy source");
        buffers.emplace(1, std::move(input));
        buffers.emplace(
            2, DeviceTensor::allocate(
                   DType::UINT8, {static_cast<std::int64_t>(packed_bytes)}));
        cuda_check(cudaMemset(buffers.at(2).data(), 0, packed_bytes),
                   "clear destination");

        std::uint32_t input_id = 1;
        std::uint32_t output_id = 2;
        lp::PieLoaderStorageOp::TileMap_Body instr{};
        instr.tile_kind = lp::PieLoaderTileMapKind::Repack;
        instr.has_source = false;
        instr.has_dest = true;
        instr.dest.buffer_id = output_id;
        instr.input_buffers = {&input_id, 1};
        instr.output_buffers = {&output_id, 1};
        instr.repack_layout = lp::PieLoaderRepackLayout::MarlinMxfp4Weight;
        instr.transform_batch = 1;
        instr.transform_source_rows = rows;
        instr.transform_target_rows = rows;
        instr.transform_source_cols = cols;
        instr.transform_target_cols = cols;

        LoadExecutionStats stats{};
        transcode.tile_map(instr, lp::PieLoaderSourceExtentView{}, stats);
        cuda_check(cudaDeviceSynchronize(), "execute MarlinMxfp4Weight repack");

        std::vector<std::uint8_t> output(packed_bytes);
        cuda_check(cudaMemcpy(output.data(), buffers.at(2).data(), packed_bytes,
                              cudaMemcpyDeviceToHost),
                   "copy destination");
        if (std::all_of(output.begin(), output.end(),
                        [](std::uint8_t byte) { return byte == 0; })) {
            std::fprintf(stderr,
                         "marlin_mxfp4_repack_test: destination was not written\n");
            return 1;
        }
        std::puts(
            "marlin_mxfp4_repack_test: MarlinMxfp4Weight tile map executed");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "marlin_mxfp4_repack_test: %s\n", error.what());
        return 1;
    }
}
