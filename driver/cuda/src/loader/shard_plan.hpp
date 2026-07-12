#pragma once

#include <cstdint>
#include <stdexcept>
#include <string>
#include <vector>

namespace pie_driver_common {

struct AxisShardPlan {
    std::vector<std::int64_t> output_shape;
    std::int64_t shard_dim = 0;
    std::int64_t offset = 0;
};

inline void validate_rank(int rank, int world_size, const std::string& label) {
    if (world_size <= 0 || rank < 0 || rank >= world_size) {
        throw std::runtime_error(label + ": rank " + std::to_string(rank) +
                                 " out of range for world_size " +
                                 std::to_string(world_size));
    }
}

inline AxisShardPlan plan_axis_shard(const std::vector<std::int64_t>& shape,
                                     int axis,
                                     int rank,
                                     int world_size,
                                     const std::string& label) {
    validate_rank(rank, world_size, label);
    if (shape.empty() || shape.size() > 2) {
        throw std::runtime_error(label + ": unsupported rank " +
                                 std::to_string(shape.size()));
    }
    if (axis < 0 || axis >= static_cast<int>(shape.size())) {
        throw std::runtime_error(label + ": axis " + std::to_string(axis) +
                                 " out of range");
    }
    const std::int64_t dim = shape[axis];
    if (dim % world_size != 0) {
        throw std::runtime_error(label + ": dim " + std::to_string(dim) +
                                 " not divisible by world_size " +
                                 std::to_string(world_size));
    }
    AxisShardPlan plan;
    plan.output_shape = shape;
    plan.shard_dim = dim / world_size;
    plan.offset = static_cast<std::int64_t>(rank) * plan.shard_dim;
    plan.output_shape[axis] = plan.shard_dim;
    return plan;
}

struct RowRangeShardPlan {
    std::int64_t row_start = 0;
    std::int64_t rows = 0;
};

inline RowRangeShardPlan plan_row_range_shard(std::int64_t total_rows,
                                              std::int64_t row_offset,
                                              std::int64_t rows,
                                              int rank,
                                              int world_size,
                                              const std::string& label) {
    validate_rank(rank, world_size, label);
    if (rows <= 0 || rows % world_size != 0) {
        throw std::runtime_error(label + ": rows " + std::to_string(rows) +
                                 " not divisible by world_size " +
                                 std::to_string(world_size));
    }
    if (row_offset < 0 || row_offset + rows > total_rows) {
        throw std::runtime_error(label + ": row range [" +
                                 std::to_string(row_offset) + ", " +
                                 std::to_string(row_offset + rows) +
                                 ") out of bounds for " +
                                 std::to_string(total_rows));
    }
    const std::int64_t rows_per_rank = rows / world_size;
    return RowRangeShardPlan{
        row_offset + static_cast<std::int64_t>(rank) * rows_per_rank,
        rows_per_rank};
}

}  // namespace pie_driver_common
