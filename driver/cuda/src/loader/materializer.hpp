#pragma once

#include "loader/load_plan.hpp"
#include "model/weight_store.hpp"

namespace pie_cuda_driver {

class SafetensorsLoader;
class NcclComm;

class Materializer {
public:
    Materializer(
        SafetensorsLoader& loader,
        WeightStore& weights,
        int tp_rank,
        int tp_size,
        NcclComm* tp_comm = nullptr) noexcept;

    MaterializedLoadPlan run(const LoadPlan& plan);

private:
    SafetensorsLoader& loader_;
    WeightStore& weights_;
    WeightStoreBuilder builder_;
    int tp_rank_ = 0;
    int tp_size_ = 1;
    NcclComm* tp_comm_ = nullptr;
};

MaterializedLoadPlan materialize_load_plan(
    const LoadPlan& plan,
    SafetensorsLoader& loader,
    WeightStore& weights,
    int tp_rank,
    int tp_size,
    NcclComm* tp_comm = nullptr);

}  // namespace pie_cuda_driver
