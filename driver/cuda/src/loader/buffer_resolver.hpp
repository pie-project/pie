#pragma once

// Resolves a LoadPlan buffer id to its DeviceTensor. A buffer is either
// still live in the executor's working set (`buffers`) or has been finalized into
// the WeightStore (`finalized_names` -> `weights`). Shared by the executor
// (CreateView) and the transcode engine, which both need to read input buffers
// without owning the maps.

#include <cstdint>
#include <stdexcept>
#include <string>
#include <unordered_map>

#include "tensor.hpp"
#include "model/weight_store.hpp"

namespace pie_cuda_driver {

struct BufferResolver {
    std::unordered_map<std::uint32_t, DeviceTensor>& buffers;
    std::unordered_map<std::uint32_t, std::string>& finalized_names;
    WeightStoreBuilder& weights;

    DeviceTensor& tensor(std::uint32_t buffer_id)
    {
        auto it = buffers.find(buffer_id);
        if (it == buffers.end()) {
            throw std::runtime_error("rust storage executor: buffer missing");
        }
        return it->second;
    }

    const DeviceTensor& or_finalized(std::uint32_t buffer_id) const
    {
        auto it = buffers.find(buffer_id);
        if (it != buffers.end()) {
            return it->second;
        }
        auto finalized = finalized_names.find(buffer_id);
        if (finalized != finalized_names.end()) {
            return weights.get(finalized->second);
        }
        throw std::runtime_error(
            "rust storage executor: source buffer missing for CreateView");
    }
};

}  // namespace pie_cuda_driver
