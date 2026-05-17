#pragma once

#include "loader/layout_plan.hpp"
#include "model/weight_store.hpp"

namespace pie_cuda_driver {

class CheckpointByteSource;
struct StorageProgram;
class SafetensorsCheckpointSource;
class NcclComm;

class LoadExecutor {
public:
    LoadExecutor(
        SafetensorsCheckpointSource& loader,
        WeightStore& weights,
        int tp_rank,
        int tp_size,
        NcclComm* tp_comm,
        CheckpointByteSource* byte_source,
        const StorageProgram* storage_program) noexcept;

    LoadExecutionStats run(const LayoutPlan& plan);

private:
    SafetensorsCheckpointSource& loader_;
    WeightStore& weights_;
    WeightStoreBuilder builder_;
    int tp_rank_ = 0;
    int tp_size_ = 1;
    NcclComm* tp_comm_ = nullptr;
    CheckpointByteSource* byte_source_ = nullptr;
    const StorageProgram* storage_program_ = nullptr;
};

LoadExecutionStats execute_layout_plan(
    const LayoutPlan& plan,
    SafetensorsCheckpointSource& loader,
    WeightStore& weights,
    int tp_rank,
    int tp_size,
    NcclComm* tp_comm,
    CheckpointByteSource* byte_source,
    const StorageProgram* storage_program);

}  // namespace pie_cuda_driver
