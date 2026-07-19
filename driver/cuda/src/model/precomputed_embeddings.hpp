#pragma once

#include <cuda_runtime.h>

#include <cstddef>
#include <cstdint>
#include <stdexcept>

#include "model/precomputed_embedding_inputs.hpp"

namespace pie_cuda_driver::model {

inline void scatter_precomputed_embeddings(
    const PrecomputedEmbeddingInputs& in,
    std::uint16_t* hidden,
    int total_rows,
    int hidden_size,
    cudaStream_t stream) {
    if (in.num_blocks == 0) return;
    if (in.rows_h == nullptr || in.byte_indptr_h == nullptr ||
        in.shapes_h == nullptr || in.dtypes_h == nullptr ||
        in.anchor_rows_h == nullptr || hidden == nullptr) {
        throw std::runtime_error("precomputed embeddings: missing input");
    }
    for (int block = 0; block < in.num_blocks; ++block) {
        if (in.dtypes_h[block] != 2) {
            throw std::runtime_error("precomputed embeddings: expected bf16");
        }
        const std::uint32_t rows = in.shapes_h[2 * block];
        const std::uint32_t width = in.shapes_h[2 * block + 1];
        const std::uint32_t anchor = in.anchor_rows_h[block];
        if (rows == 0 || width != static_cast<std::uint32_t>(hidden_size) ||
            static_cast<std::uint64_t>(anchor) + rows >
                static_cast<std::uint64_t>(total_rows)) {
            throw std::runtime_error("precomputed embeddings: invalid shape");
        }
        const std::size_t begin = in.byte_indptr_h[block];
        const std::size_t end = in.byte_indptr_h[block + 1];
        const std::size_t expected =
            static_cast<std::size_t>(rows) * hidden_size * sizeof(std::uint16_t);
        if (end < begin || end - begin != expected) {
            throw std::runtime_error("precomputed embeddings: invalid byte extent");
        }
        const cudaError_t status = cudaMemcpyAsync(
            hidden + static_cast<std::size_t>(anchor) * hidden_size,
            in.rows_h + begin,
            expected,
            cudaMemcpyHostToDevice,
            stream);
        if (status != cudaSuccess) {
            throw std::runtime_error("precomputed embeddings: cudaMemcpyAsync failed");
        }
    }
}

}  // namespace pie_cuda_driver::model
