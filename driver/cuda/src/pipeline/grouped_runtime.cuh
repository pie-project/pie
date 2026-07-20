#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <initializer_list>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <span>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <cub/block/block_reduce.cuh>
#include <cub/block/block_scan.cuh>

#include "cuda_check.hpp"
#include "pie_native/ptir/plan.hpp"
#include "pipeline/channels.hpp"
#include "pipeline/grouped_copy.hpp"
#include "pipeline/library_region.hpp"
#include "pipeline/program_runtime.hpp"
#include "pipeline/tier0/tier0_kernels.cuh"

namespace pie_cuda_driver::pipeline {

inline constexpr std::uint32_t kUnavailableGroupedExtent = UINT32_MAX;
inline constexpr std::uint32_t kMaxExactNucleusLibraryVocab = 4096;
inline constexpr std::uint32_t kFastNucleusThreads = 256;
inline constexpr std::uint32_t kFastNucleusChunks = 4;

struct GroupedLaneBinding {
    PtirInstance* instance = nullptr;
    const plan::StagePlan* plan = nullptr;
    std::uint64_t plan_identity = 0;
    std::vector<DeviceHostChannelTicket>* tickets = nullptr;
    const float* logits_base = nullptr;
    const void* query_base = nullptr;
    const void* layer_base = nullptr;
    const std::vector<std::uint64_t>* logits_bf16_rows = nullptr;
    const std::vector<std::uint64_t>* mtp_logits_bf16_rows = nullptr;
    std::uint64_t sample_output_channel_mask = 0;
    const std::uint8_t* row_valid = nullptr;
    std::uint32_t row_valid_offset = 0;
    const std::uint32_t* row_seeds = nullptr;
    // Launch-scoped staged execution overlays. A stage reads a channel written
    // by an earlier stage from its pending cell, and readiness treats prior
    // takes/puts as logical-fire-local rather than externally committed.
    const std::unordered_set<std::uint32_t>* prior_put_slots = nullptr;
    const std::unordered_set<std::uint32_t>* prior_take_slots = nullptr;
    std::uint32_t* commit_slot = nullptr;
    std::uint32_t logits_row_offset = 0;
    std::uint32_t logits_row_count = 0;
    std::uint32_t row_count = kUnavailableGroupedExtent;
    std::uint32_t token_count = kUnavailableGroupedExtent;
    std::uint32_t kv_len = kUnavailableGroupedExtent;
    std::uint32_t page_count = kUnavailableGroupedExtent;
    std::uint32_t query_len = kUnavailableGroupedExtent;
    std::uint32_t key_len = kUnavailableGroupedExtent;
    std::uint32_t vocab = 0;          // PTIR-visible logical vocabulary
    std::uint32_t logits_stride = 0; // physical model row stride
    std::uint32_t program_index = 0;
};

struct GroupedExecutionOptions {
    bool reset_commits = true;
    bool pull_tickets = true;
    bool finalize = true;
    // Fire-timing probe: fill GroupedLaunchResult's section times.
    bool time_sections = false;
};

struct GroupedLaunchResult {
    DeviceHostChannelTicket* device_tickets = nullptr;
    bool device_tickets_persistent = false;
    std::vector<std::uint32_t> ticket_offsets;
    std::vector<std::uint32_t> ticket_counts;
    bool used_nucleus_library = false;
    bool used_selection_library = false;
    std::uint32_t body_op_launches = 0;
    bool large_nucleus_scalable = false;
    // Section times (µs), filled only under
    // GroupedExecutionOptions::time_sections: host-side lane/value table
    // build, device workspace acquisition (including the fallback
    // cudaMallocAsync path), staging pack + H2D upload, and the launch
    // region (readiness, body kernels, settlement wiring).
    std::int64_t t_build_us = -1;
    std::int64_t t_workspace_us = -1;
    std::int64_t t_upload_us = -1;
    std::int64_t t_launch_us = -1;
};

struct GroupedReadinessLane {
    std::uint32_t full_offset = 0;
    std::uint32_t full_count = 0;
    std::uint32_t empty_offset = 0;
    std::uint32_t empty_count = 0;
};

struct GroupedDynamicShape {
    std::uint64_t max_numel = 1;
    std::uint32_t elements_per_extent = 1;
    std::uint8_t extent = 0xff;
    std::uint8_t reserved[3] = {};
};

struct GroupedRowShape {
    GroupedDynamicShape rows{};
    GroupedDynamicShape columns{};
    std::uint32_t max_rows = 1;
    std::uint32_t max_columns = 1;
};

struct GroupedNucleusLaunch {
    std::uint32_t logits = 0;
    std::uint32_t top_p = 0;
    std::uint32_t rng_state = 0;
    std::uint32_t output = 0;
    std::uint32_t output_channel = UINT32_MAX;
    std::uint32_t rows = 0;
    std::uint32_t len = 0;
    std::uint32_t top_p_numel = 0;
    std::uint8_t logits_kind = 0;
    std::uint32_t logit_scale = UINT32_MAX;
    std::uint32_t logit_scale_numel = 0;
};

__device__ __forceinline__ std::uint32_t* grouped_commit(
    const PtirLaneRecord* lanes, std::uint32_t lane) {
    return reinterpret_cast<std::uint32_t*>(lanes[lane].commit_slot);
}

__device__ __forceinline__ std::uint32_t grouped_lane_extent(
    const PtirLaneRecord& lane, std::uint8_t extent) {
    switch (extent) {
        case PTIR_EXTENT_KV_LEN: return lane.kv_len;
        case PTIR_EXTENT_PAGE_COUNT: return lane.page_count;
        case PTIR_EXTENT_ROW_COUNT: return lane.row_count;
        case PTIR_EXTENT_TOKEN_COUNT: return lane.token_count;
        case PTIR_EXTENT_SAMPLED_ROWS: return lane.sampled_rows;
        case PTIR_EXTENT_QUERY_LEN: return lane.query_len;
        case PTIR_EXTENT_KEY_LEN: return lane.key_len;
        default: return 1;
    }
}

__device__ __forceinline__ std::uint64_t grouped_lane_numel(
    const PtirLaneRecord& lane, GroupedDynamicShape shape) {
    return shape.extent == 0xff
        ? shape.max_numel
        : static_cast<std::uint64_t>(
            grouped_lane_extent(lane, shape.extent)) *
            shape.elements_per_extent;
}

__device__ __forceinline__ std::uint32_t grouped_lane_rows(
    const PtirLaneRecord& lane, const GroupedRowShape& shape) {
    return static_cast<std::uint32_t>(
        grouped_lane_numel(lane, shape.rows));
}

__device__ __forceinline__ std::uint32_t grouped_lane_columns(
    const PtirLaneRecord& lane, const GroupedRowShape& shape) {
    return static_cast<std::uint32_t>(
        grouped_lane_numel(lane, shape.columns));
}

template <class T>
__device__ __forceinline__ T* grouped_value(
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t lane,
    std::uint32_t value) {
    return reinterpret_cast<T*>(values[
        static_cast<std::size_t>(lane) * value_count + value]);
}

__device__ __forceinline__ float grouped_direct_bf16_load(
    const PtirLaneRecord* lanes,
    const std::uint64_t* mtp_rows,
    const std::uint32_t* mtp_offsets,
    std::uint32_t lane,
    std::uint64_t index,
    std::uint32_t vocab,
    std::uint8_t kind) {
    const std::uint32_t row =
        static_cast<std::uint32_t>(index / vocab);
    const std::uint32_t column =
        static_cast<std::uint32_t>(index % vocab);
    const auto* rows = kind == 2
        ? mtp_rows + mtp_offsets[lane]
        : reinterpret_cast<const std::uint64_t*>(
            lanes[lane].logits_base);
    const auto* source =
        reinterpret_cast<const std::uint16_t*>(rows[row]);
    return __uint_as_float(
        static_cast<std::uint32_t>(source[column]) << 16);
}

__device__ __forceinline__ float grouped_nucleus_logit_scale(
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t lane,
    std::uint32_t row,
    const GroupedNucleusLaunch& launch) {
    if (launch.logit_scale == UINT32_MAX) return 1.0f;
    const float divisor = grouped_value<float>(
        values, value_count, lane, launch.logit_scale)[
        launch.logit_scale_numel <= 1 ? 0 : row];
    return 1.0f / divisor;
}

static __global__ void k_grouped_stage_readiness(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const GroupedReadinessLane* readiness,
    const std::uint32_t* slots,
    const std::uint8_t* full,
    const std::uint32_t* head,
    const std::uint32_t* tail,
    const std::uint32_t* cap1) {
    const std::uint32_t lane =
        blockIdx.x * static_cast<std::uint32_t>(blockDim.x) + threadIdx.x;
    if (lane >= header->lane_count) return;
    const GroupedReadinessLane descriptor = readiness[lane];
    bool ready = true;
    for (std::uint32_t index = 0; index < descriptor.full_count; ++index) {
        const std::uint32_t slot = slots[descriptor.full_offset + index];
        ready = ready &&
            full[static_cast<std::size_t>(slot) * kMaxRing + head[slot]] != 0;
    }
    for (std::uint32_t index = 0; index < descriptor.empty_count; ++index) {
        const std::uint32_t slot = slots[descriptor.empty_offset + index];
        ready = ready && ((tail[slot] + 1) % cap1[slot]) != head[slot];
    }
    if (!ready) atomicAnd(grouped_commit(lanes, lane), 0u);
}

// Oracle helper used only by the synthetic Tier-0 regression. Production
// grouped and one-lane execution load BF16 rows directly in each consumer.
static __global__ void k_materialize_bf16_rows(
    const std::uint64_t* rows,
    float* output,
    std::uint32_t row_count,
    std::uint32_t vocab) {
    const std::uint64_t numel =
        static_cast<std::uint64_t>(row_count) * vocab;
    for (std::uint64_t index =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         index < numel;
         index += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const std::uint32_t row =
            static_cast<std::uint32_t>(index / vocab);
        const std::uint32_t column =
            static_cast<std::uint32_t>(index % vocab);
        const auto* source =
            reinterpret_cast<const std::uint16_t*>(rows[row]);
        output[index] = __uint_as_float(
            static_cast<std::uint32_t>(source[column]) << 16);
    }
}

static __global__ void k_grouped_topk(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* mtp_rows,
    const std::uint32_t* mtp_offsets,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t input,
    std::uint32_t output_values,
    std::uint32_t output_indices,
    std::uint32_t rows,
    std::uint32_t len,
    std::uint32_t k,
    bool flat_ragged,
    GroupedDynamicShape input_shape,
    GroupedRowShape row_shape,
    std::uint32_t vocab,
    std::uint8_t input_kind) {
    __shared__ float shared_values[kTier0Block];
    __shared__ std::uint32_t shared_indices[kTier0Block];
    __shared__ std::uint8_t shared_nan[kTier0Block];
    __shared__ float previous_value;
    __shared__ std::uint32_t previous_index;
    __shared__ std::uint8_t previous_nan;
    __shared__ std::uint8_t previous_valid;
    const std::uint32_t grouped_row = blockIdx.x;
    const std::uint32_t lane = grouped_row / row_shape.max_rows;
    const std::uint32_t row = grouped_row % row_shape.max_rows;
    const std::uint32_t actual_rows = lane < header->lane_count
        ? grouped_lane_rows(lanes[lane], row_shape)
        : 0;
    if (lane >= header->lane_count ||
        (!flat_ragged && row >= actual_rows) ||
        *grouped_commit(lanes, lane) == 0) {
        return;
    }
    if (flat_ragged) {
        len = static_cast<std::uint32_t>(
            grouped_lane_numel(lanes[lane], input_shape));
        k = len;
    } else {
        len = grouped_lane_columns(lanes[lane], row_shape);
    }
    const float* source = input_kind == 0
        ? grouped_value<float>(values, value_count, lane, input) +
            static_cast<std::uint64_t>(row) * len
        : nullptr;
    float* output_value =
        grouped_value<float>(values, value_count, lane, output_values) +
        static_cast<std::uint64_t>(row) * k;
    std::uint32_t* output_index =
        grouped_value<std::uint32_t>(
            values, value_count, lane, output_indices) +
        static_cast<std::uint64_t>(row) * k;
    if (threadIdx.x == 0) {
        previous_value = 0.0f;
        previous_index = 0;
        previous_nan = 0;
        previous_valid = 0;
    }
    __syncthreads();
    for (std::uint32_t pick = 0; pick < k; ++pick) {
        const float prior_value = previous_value;
        const std::uint32_t prior_index = previous_index;
        const bool prior_nan = previous_nan != 0;
        const bool prior_valid = previous_valid != 0;
        float best = 0.0f;
        std::uint32_t best_index = UINT32_MAX;
        bool best_nan = false;
        for (std::uint32_t index = threadIdx.x; index < len;
             index += blockDim.x) {
            const float value = input_kind == 0
                ? source[index]
                : grouped_direct_bf16_load(
                      lanes, mtp_rows, mtp_offsets, lane,
                      static_cast<std::uint64_t>(row) * len + index,
                      vocab, input_kind);
            const bool value_nan = isnan(value);
            const bool available =
                !prior_valid ||
                t0_desc_before(
                    prior_value, prior_index, prior_nan,
                    value, index, value_nan);
            if (available &&
                (best_index == UINT32_MAX ||
                 t0_desc_before(
                     value, index, value_nan,
                     best, best_index, best_nan))) {
                best = value;
                best_index = index;
                best_nan = value_nan;
            }
        }
        shared_values[threadIdx.x] = best;
        shared_indices[threadIdx.x] = best_index;
        shared_nan[threadIdx.x] = best_nan;
        __syncthreads();
        for (std::uint32_t stride = blockDim.x >> 1; stride > 0;
             stride >>= 1) {
            if (threadIdx.x < stride) {
                const std::uint32_t other_index =
                    shared_indices[threadIdx.x + stride];
                if (other_index != UINT32_MAX &&
                    (shared_indices[threadIdx.x] == UINT32_MAX ||
                     t0_desc_before(
                         shared_values[threadIdx.x + stride],
                         other_index,
                         shared_nan[threadIdx.x + stride] != 0,
                         shared_values[threadIdx.x],
                         shared_indices[threadIdx.x],
                         shared_nan[threadIdx.x] != 0))) {
                    shared_values[threadIdx.x] =
                        shared_values[threadIdx.x + stride];
                    shared_indices[threadIdx.x] = other_index;
                    shared_nan[threadIdx.x] =
                        shared_nan[threadIdx.x + stride];
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            output_value[pick] = shared_values[0];
            output_index[pick] = shared_indices[0];
            previous_value = shared_values[0];
            previous_index = shared_indices[0];
            previous_nan = shared_nan[0];
            previous_valid = 1;
        }
        __syncthreads();
    }
}

__device__ __forceinline__ void grouped_nucleus_reduce(
    const PtirLaneRecord* lanes,
    const std::uint64_t* mtp_rows,
    const std::uint32_t* mtp_offsets,
    std::uint32_t lane_index,
    std::uint32_t row_index,
    const float* source,
    std::uint32_t len,
    std::uint8_t logits_kind,
    RedKind kind,
    bool exponentials,
    float maximum,
    float logit_scale,
    float levels[kCanonicalReduceLevels][kCanonicalReduceWidth],
    float tile[kCanonicalReduceWidth],
    std::uint32_t counts[kCanonicalReduceLevels],
    float* result) {
    const std::uint32_t thread_lane = threadIdx.x;
    if (thread_lane < kCanonicalReduceLevels) counts[thread_lane] = 0;
    __syncwarp();
    for (std::uint32_t base = 0; base < len;
         base += kCanonicalReduceWidth) {
        const std::uint32_t index = base + thread_lane;
        float value = red_identity<float>(kind);
        if (index < len) {
            value = logits_kind == 0
                ? source[index]
                : grouped_direct_bf16_load(
                      lanes, mtp_rows, mtp_offsets, lane_index,
                      static_cast<std::uint64_t>(row_index) * len + index,
                      len, logits_kind);
            value *= logit_scale;
            if (exponentials) value = expf(value - maximum);
        }
        tile[thread_lane] = value;
        const std::uint32_t tile_count =
            len - base < kCanonicalReduceWidth
                ? len - base
                : kCanonicalReduceWidth;
        const float partial = reduce_canonical_slot(
            tile, thread_lane, tile_count, kind);
        if (thread_lane == 0) levels[0][counts[0]++] = partial;
        __syncwarp();
        for (std::uint32_t level = 0;
             level + 1 < kCanonicalReduceLevels;
             ++level) {
            if (counts[level] != kCanonicalReduceWidth) break;
            const float carry = reduce_canonical_slot(
                levels[level], thread_lane, kCanonicalReduceWidth, kind);
            if (thread_lane == 0) {
                counts[level] = 0;
                levels[level + 1][counts[level + 1]++] = carry;
            }
            __syncwarp();
        }
    }
    for (std::uint32_t level = 0;
         level + 1 < kCanonicalReduceLevels;
         ++level) {
        const std::uint32_t count = counts[level];
        if (count == 0) continue;
        bool higher = false;
        for (std::uint32_t next = level + 1;
             next < kCanonicalReduceLevels;
             ++next) {
            higher = higher || counts[next] != 0;
        }
        if (count == 1 && !higher) {
            if (thread_lane == 0) *result = levels[level][0];
            __syncwarp();
            return;
        }
        const float carry = reduce_canonical_slot(
            levels[level], thread_lane, count, kind);
        if (thread_lane == 0) {
            counts[level] = 0;
            levels[level + 1][counts[level + 1]++] = carry;
        }
        __syncwarp();
    }
    if (thread_lane == 0) {
        *result = counts[kCanonicalReduceLevels - 1] == 0
            ? red_identity<float>(kind)
            : levels[kCanonicalReduceLevels - 1][0];
    }
    __syncwarp();
}

struct NucleusFloatMax {
    __device__ float operator()(float lhs, float rhs) const {
        return fmaxf(lhs, rhs);
    }
};

struct NucleusCandidate {
    float score;
    std::uint32_t token;
    std::uint32_t have;
};

struct NucleusCandidateMax {
    __device__ NucleusCandidate operator()(
        const NucleusCandidate& lhs,
        const NucleusCandidate& rhs) const {
        if (lhs.have == 0) return rhs;
        if (rhs.have == 0) return lhs;
        if (rhs.score > lhs.score ||
            (rhs.score == lhs.score && rhs.token < lhs.token)) {
            return rhs;
        }
        return lhs;
    }
};

static __global__ void k_grouped_nucleus_max_chunks(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* mtp_rows,
    const std::uint32_t* mtp_offsets,
    const std::uint64_t* values,
    std::uint32_t value_count,
    float* partial_maxima,
    GroupedNucleusLaunch launch) {
    using Reduce = cub::BlockReduce<float, kFastNucleusThreads>;
    __shared__ typename Reduce::TempStorage reduce;
    const std::uint32_t grouped_chunk = blockIdx.x;
    const std::uint32_t segment =
        grouped_chunk / kFastNucleusChunks;
    const std::uint32_t chunk =
        grouped_chunk % kFastNucleusChunks;
    const std::uint32_t lane = segment / launch.rows;
    const std::uint32_t row = segment % launch.rows;
    if (lane >= header->lane_count ||
        row >= lanes[lane].sampled_rows ||
        *grouped_commit(lanes, lane) == 0) {
        return;
    }
    const auto* logits = launch.logits_kind == 0
        ? grouped_value<float>(
              values, value_count, lane, launch.logits) +
              static_cast<std::uint64_t>(row) * launch.len
        : nullptr;
    const float logit_scale = grouped_nucleus_logit_scale(
        values, value_count, lane, row, launch);
    float local_max = t0_neg_inf();
    for (std::uint32_t index =
             chunk * blockDim.x + threadIdx.x;
         index < launch.len;
         index += kFastNucleusChunks * blockDim.x) {
        const float logit = (launch.logits_kind == 0
            ? logits[index]
            : grouped_direct_bf16_load(
                  lanes, mtp_rows, mtp_offsets, lane,
                  static_cast<std::uint64_t>(row) * launch.len + index,
                  launch.len, launch.logits_kind)) * logit_scale;
        local_max = fmaxf(local_max, logit);
    }
    const float maximum = Reduce(reduce).Reduce(
        local_max, NucleusFloatMax{});
    if (threadIdx.x == 0) {
        partial_maxima[grouped_chunk] = maximum;
    }
}

static __global__ void k_grouped_nucleus_weight_chunks(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* mtp_rows,
    const std::uint32_t* mtp_offsets,
    const std::uint64_t* values,
    std::uint32_t value_count,
    const float* partial_maxima,
    float* weights,
    float* thread_masses,
    GroupedNucleusLaunch launch) {
    const std::uint32_t grouped_chunk = blockIdx.x;
    const std::uint32_t segment =
        grouped_chunk / kFastNucleusChunks;
    const std::uint32_t chunk =
        grouped_chunk % kFastNucleusChunks;
    const std::uint32_t lane = segment / launch.rows;
    const std::uint32_t row = segment % launch.rows;
    if (lane >= header->lane_count ||
        row >= lanes[lane].sampled_rows ||
        *grouped_commit(lanes, lane) == 0) {
        return;
    }
    float maximum = t0_neg_inf();
    for (std::uint32_t part = 0;
         part < kFastNucleusChunks;
         ++part) {
        maximum = fmaxf(
            maximum,
            partial_maxima[
                segment * kFastNucleusChunks + part]);
    }
    const auto* logits = launch.logits_kind == 0
        ? grouped_value<float>(
              values, value_count, lane, launch.logits) +
              static_cast<std::uint64_t>(row) * launch.len
        : nullptr;
    const float logit_scale = grouped_nucleus_logit_scale(
        values, value_count, lane, row, launch);
    float* row_weights =
        weights + static_cast<std::uint64_t>(segment) * launch.len;
    float local_mass = 0.0f;
    for (std::uint32_t index =
             chunk * blockDim.x + threadIdx.x;
         index < launch.len;
         index += kFastNucleusChunks * blockDim.x) {
        const float logit = (launch.logits_kind == 0
            ? logits[index]
            : grouped_direct_bf16_load(
                  lanes, mtp_rows, mtp_offsets, lane,
                  static_cast<std::uint64_t>(row) * launch.len + index,
                  launch.len, launch.logits_kind)) * logit_scale;
        const float weight = expf(logit - maximum);
        row_weights[index] = weight;
        local_mass += weight;
    }
    thread_masses[
        static_cast<std::uint64_t>(grouped_chunk) * blockDim.x +
        threadIdx.x] = local_mass;
}

static __global__ void k_grouped_nucleus_inverse_sample(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const PtirLaneChannelSlot* channels,
    const std::uint64_t* values,
    std::uint32_t value_count,
    float* weights,
    const float* thread_masses,
    GroupedNucleusLaunch launch) {
    using CandidateReduce =
        cub::BlockReduce<NucleusCandidate, kFastNucleusThreads>;
    using FloatReduce = cub::BlockReduce<float, kFastNucleusThreads>;
    using FloatScan = cub::BlockScan<float, kFastNucleusThreads>;
    __shared__ typename CandidateReduce::TempStorage candidate_reduce;
    __shared__ typename FloatReduce::TempStorage float_reduce;
    __shared__ typename FloatScan::TempStorage float_scan;
    __shared__ float chunk_masses[kFastNucleusChunks];
    __shared__ float total_mass;
    __shared__ float candidate_probability;
    __shared__ float chunk_target;
    __shared__ float owner_target;
    __shared__ std::uint32_t candidate_index;
    __shared__ std::uint32_t owner_thread;
    __shared__ std::uint32_t selected_chunk;
    __shared__ std::uint32_t selected_token;
    __shared__ std::uint8_t done;
    const std::uint32_t segment = blockIdx.x;
    const std::uint32_t lane = segment / launch.rows;
    const std::uint32_t row = segment % launch.rows;
    if (lane >= header->lane_count ||
        row >= lanes[lane].sampled_rows ||
        *grouped_commit(lanes, lane) == 0) {
        return;
    }
    auto* row_weights =
        weights +
        static_cast<std::uint64_t>(segment) * launch.len;
    const auto* state = grouped_value<std::uint32_t>(
        values, value_count, lane, launch.rng_state);
    const unsigned long long seed64 =
        ptir_rng_keyed_seed(state[0], state[1]);
    const float top_p = grouped_value<float>(
        values, value_count, lane, launch.top_p)[
        launch.top_p_numel <= 1 ? 0 : row];
    for (std::uint32_t chunk = 0;
         chunk < kFastNucleusChunks;
         ++chunk) {
        const float local_mass = thread_masses[
            (static_cast<std::uint64_t>(segment) *
                 kFastNucleusChunks +
             chunk) *
                blockDim.x +
            threadIdx.x];
        const float chunk_mass =
            FloatReduce(float_reduce).Sum(local_mass);
        if (threadIdx.x == 0) {
            chunk_masses[chunk] = chunk_mass;
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        total_mass = 0.0f;
        for (std::uint32_t chunk = 0;
             chunk < kFastNucleusChunks;
             ++chunk) {
            total_mass += chunk_masses[chunk];
        }
        done = 0;
        selected_token = 0;
    }
    __syncthreads();

    constexpr std::uint32_t kMaxAttempts = 128;
    for (std::uint32_t attempt = 0;
         attempt < kMaxAttempts && done == 0;
         ++attempt) {
        if (threadIdx.x == 0) {
            candidate_index = UINT32_MAX;
            owner_thread = UINT32_MAX;
            const float target = t0_hash_uniform(
                seed64,
                static_cast<int>(
                    static_cast<std::uint64_t>(row) * kMaxAttempts +
                    attempt)) * total_mass;
            float chunk_begin = 0.0f;
            selected_chunk = kFastNucleusChunks - 1;
            chunk_target = CUDART_INF_F;
            owner_target = CUDART_INF_F;
            for (std::uint32_t chunk = 0;
                 chunk < kFastNucleusChunks;
                 ++chunk) {
                const float chunk_end =
                    chunk_begin + chunk_masses[chunk];
                if (chunk_begin < target &&
                    chunk_end >= target) {
                    selected_chunk = chunk;
                    chunk_target = target - chunk_begin;
                    break;
                }
                chunk_begin = chunk_end;
            }
        }
        __syncthreads();
        const float thread_mass = thread_masses[
            (static_cast<std::uint64_t>(segment) *
                 kFastNucleusChunks +
             selected_chunk) *
                blockDim.x +
            threadIdx.x];
        float inclusive_mass = 0.0f;
        FloatScan(float_scan).InclusiveSum(
            thread_mass, inclusive_mass);
        const float thread_begin = inclusive_mass - thread_mass;
        if (thread_begin < chunk_target &&
            inclusive_mass >= chunk_target) {
            owner_thread = threadIdx.x;
            owner_target = chunk_target - thread_begin;
        }
        __syncthreads();
        if (threadIdx.x == 0 && owner_thread == UINT32_MAX) {
            owner_thread = blockDim.x - 1;
            owner_target = CUDART_INF_F;
        }
        __syncthreads();
        if (threadIdx.x == owner_thread) {
            float cumulative = 0.0f;
            for (std::uint32_t index =
                     selected_chunk * blockDim.x + threadIdx.x;
                 index < launch.len;
                 index += kFastNucleusChunks * blockDim.x) {
                cumulative += row_weights[index];
                if (cumulative >= owner_target) {
                    candidate_index = index;
                    break;
                }
            }
        }
        __syncthreads();
        if (threadIdx.x == 0) {
            if (candidate_index == UINT32_MAX) {
                candidate_index = launch.len - 1;
            }
            candidate_probability =
                row_weights[candidate_index];
        }
        __syncthreads();
        float local_exclusive = 0.0f;
        for (std::uint32_t index = threadIdx.x;
             index < launch.len;
             index += blockDim.x) {
            const float probability = row_weights[index];
            if (probability > candidate_probability ||
                (probability == candidate_probability &&
                 index < candidate_index)) {
                local_exclusive += probability;
            }
        }
        const float exclusive =
            FloatReduce(float_reduce).Sum(local_exclusive);
        if (threadIdx.x == 0 &&
            exclusive < top_p * total_mass) {
            selected_token = candidate_index;
            done = 1;
        }
        __syncthreads();
    }
    if (done == 0) {
        NucleusCandidate best{t0_neg_inf(), 0, 0};
        for (std::uint32_t index = threadIdx.x; index < launch.len;
             index += blockDim.x) {
            const float probability = row_weights[index];
            if (!isnan(probability) &&
                (best.have == 0 || probability > best.score ||
                 (probability == best.score && index < best.token))) {
                best = {probability, index, 1};
            }
        }
        const NucleusCandidate winner =
            CandidateReduce(candidate_reduce).Reduce(
                best, NucleusCandidateMax{});
        if (threadIdx.x == 0) {
            selected_token = winner.have != 0 ? winner.token : 0;
        }
    }
    if (threadIdx.x == 0) {
        grouped_value<std::uint32_t>(
            values, value_count, lane, launch.output)[row] =
            selected_token;
        if (launch.output_channel != UINT32_MAX) {
            auto* pending = reinterpret_cast<std::uint32_t*>(
                channels[
                    lanes[lane].channel_slot_offset + launch.output_channel]
                    .pending_cell);
            pending[row] = selected_token;
        }
    }
}

static __global__ void k_grouped_nucleus_sample(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const PtirLaneChannelSlot* channels,
    const std::uint64_t* mtp_rows,
    const std::uint32_t* mtp_offsets,
    const std::uint64_t* values,
    std::uint32_t value_count,
    GroupedNucleusLaunch launch) {
    __shared__ float levels[kCanonicalReduceLevels][kCanonicalReduceWidth];
    __shared__ float tile[kCanonicalReduceWidth];
    __shared__ std::uint32_t counts[kCanonicalReduceLevels];
    __shared__ float candidates[kCanonicalReduceWidth];
    __shared__ std::uint32_t candidate_indices[kCanonicalReduceWidth];
    __shared__ std::uint8_t candidate_nan[kCanonicalReduceWidth];
    __shared__ std::uint8_t candidate_have[kCanonicalReduceWidth];
    __shared__ float maximum;
    __shared__ float sum;
    __shared__ float previous_value;
    __shared__ float exclusive_mass;
    __shared__ std::uint32_t previous_index;
    __shared__ std::uint8_t previous_nan;
    __shared__ std::uint8_t selected_any;
    __shared__ std::uint8_t stop;

    const std::uint32_t grouped_row = blockIdx.x;
    const std::uint32_t lane = grouped_row / launch.rows;
    const std::uint32_t row = grouped_row % launch.rows;
    if (lane >= header->lane_count ||
        row >= lanes[lane].sampled_rows ||
        *grouped_commit(lanes, lane) == 0) {
        return;
    }
    const float* logits = launch.logits_kind == 0
        ? grouped_value<float>(values, value_count, lane, launch.logits) +
              static_cast<std::uint64_t>(row) * launch.len
        : nullptr;
    const float logit_scale = grouped_nucleus_logit_scale(
        values, value_count, lane, row, launch);
    grouped_nucleus_reduce(
        lanes, mtp_rows, mtp_offsets, lane, row, logits, launch.len,
        launch.logits_kind, RedKind::Max, false, 0.0f,
        logit_scale,
        levels, tile, counts, &maximum);
    grouped_nucleus_reduce(
        lanes, mtp_rows, mtp_offsets, lane, row, logits, launch.len,
        launch.logits_kind, RedKind::Sum, true, maximum,
        logit_scale,
        levels, tile, counts, &sum);

    const float p = grouped_value<float>(
        values, value_count, lane, launch.top_p)[
        launch.top_p_numel <= 1 ? 0 : row];
    if (threadIdx.x == 0) {
        previous_value = t0_pos_inf();
        previous_index = 0;
        previous_nan = 0;
        exclusive_mass = 0.0f;
        selected_any = 0;
        stop = 0;
    }
    __syncwarp();
    constexpr std::uint32_t none = UINT32_MAX;
    for (std::uint32_t pick = 0; pick < launch.len; ++pick) {
        if (stop) break;
        float best = 0.0f;
        std::uint32_t best_index = none;
        bool best_nan = false;
        for (std::uint32_t index = threadIdx.x; index < launch.len;
             index += kCanonicalReduceWidth) {
            const float logit = (launch.logits_kind == 0
                ? logits[index]
                : grouped_direct_bf16_load(
                      lanes, mtp_rows, mtp_offsets, lane,
                      static_cast<std::uint64_t>(row) * launch.len + index,
                      launch.len, launch.logits_kind)) * logit_scale;
            const float probability = expf(logit - maximum) / sum;
            const bool probability_nan = isnan(probability);
            if (selected_any != 0 &&
                !t0_desc_before(
                    previous_value, previous_index, previous_nan != 0,
                    probability, index, probability_nan)) {
                continue;
            }
            if (best_index == none ||
                t0_desc_before(
                    probability, index, probability_nan,
                    best, best_index, best_nan)) {
                best = probability;
                best_index = index;
                best_nan = probability_nan;
            }
        }
        candidates[threadIdx.x] = best;
        candidate_indices[threadIdx.x] = best_index;
        candidate_nan[threadIdx.x] = best_nan;
        __syncwarp();
        for (std::uint32_t offset = kCanonicalReduceWidth / 2;
             offset > 0;
             offset >>= 1) {
            if (threadIdx.x < offset) {
                const std::uint32_t other =
                    candidate_indices[threadIdx.x + offset];
                if (other != none &&
                    (candidate_indices[threadIdx.x] == none ||
                     t0_desc_before(
                         candidates[threadIdx.x + offset], other,
                         candidate_nan[threadIdx.x + offset] != 0,
                         candidates[threadIdx.x],
                         candidate_indices[threadIdx.x],
                         candidate_nan[threadIdx.x] != 0))) {
                    candidates[threadIdx.x] =
                        candidates[threadIdx.x + offset];
                    candidate_indices[threadIdx.x] = other;
                    candidate_nan[threadIdx.x] =
                        candidate_nan[threadIdx.x + offset];
                }
            }
            __syncwarp();
        }
        if (threadIdx.x == 0) {
            if (candidate_indices[0] == none || !(exclusive_mass < p)) {
                stop = 1;
            } else {
                selected_any = 1;
                exclusive_mass += candidates[0];
                previous_value = candidates[0];
                previous_index = candidate_indices[0];
                previous_nan = candidate_nan[0];
            }
        }
        __syncwarp();
    }

    const std::uint32_t* state = grouped_value<std::uint32_t>(
        values, value_count, lane, launch.rng_state);
    const unsigned long long seed64 =
        ptir_rng_keyed_seed(state[0], state[1]);
    float best_score = t0_neg_inf();
    std::uint32_t best_token = 0;
    bool have = false;
    for (std::uint32_t index = threadIdx.x; index < launch.len;
         index += kCanonicalReduceWidth) {
        const float logit = (launch.logits_kind == 0
            ? logits[index]
            : grouped_direct_bf16_load(
                  lanes, mtp_rows, mtp_offsets, lane,
                  static_cast<std::uint64_t>(row) * launch.len + index,
                  launch.len, launch.logits_kind)) * logit_scale;
        const float probability = expf(logit - maximum) / sum;
        const bool probability_nan = isnan(probability);
        const bool selected =
            selected_any != 0 &&
            (index == previous_index ||
             t0_desc_before(
                 probability, index, probability_nan,
                 previous_value, previous_index, previous_nan != 0));
        const float uniform = t0_hash_uniform(
            seed64, static_cast<int>(
                static_cast<std::uint64_t>(row) * launch.len + index));
        const float noise = -logf(-logf(uniform));
        const float score =
            (selected ? logit : t0_neg_inf()) + noise;
        if (!isnan(score) &&
            (!have || score > best_score ||
             (score == best_score && index < best_token))) {
            best_score = score;
            best_token = index;
            have = true;
        }
    }
    candidates[threadIdx.x] = best_score;
    candidate_indices[threadIdx.x] = best_token;
    candidate_have[threadIdx.x] = have;
    __syncwarp();
    for (std::uint32_t offset = kCanonicalReduceWidth / 2;
         offset > 0;
         offset >>= 1) {
        if (threadIdx.x < offset) {
            const float other_score = candidates[threadIdx.x + offset];
            const std::uint32_t other_token =
                candidate_indices[threadIdx.x + offset];
            const bool other_have =
                candidate_have[threadIdx.x + offset] != 0;
            const bool self_have = candidate_have[threadIdx.x] != 0;
            if (other_have &&
                (!self_have || other_score > candidates[threadIdx.x] ||
                 (other_score == candidates[threadIdx.x] &&
                  other_token < candidate_indices[threadIdx.x]))) {
                candidates[threadIdx.x] = other_score;
                candidate_indices[threadIdx.x] = other_token;
                candidate_have[threadIdx.x] = 1;
            }
        }
        __syncwarp();
    }
    if (threadIdx.x == 0) {
        const std::uint32_t token =
            candidate_have[0] ? candidate_indices[0] : 0;
        grouped_value<std::uint32_t>(
            values, value_count, lane, launch.output)[row] = token;
        if (launch.output_channel != UINT32_MAX) {
            auto* pending = reinterpret_cast<std::uint32_t*>(
                channels[
                    lanes[lane].channel_slot_offset + launch.output_channel]
                    .pending_cell);
            pending[row] = token;
        }
    }
}

inline std::uint32_t grouped_grid(std::uint64_t total) {
    return static_cast<std::uint32_t>(
        std::max<std::uint64_t>(
            1,
            std::min<std::uint64_t>(
                (total + kTier0Block - 1) / kTier0Block, 65535)));
}

inline std::size_t grouped_dtype_size(std::uint8_t dtype) {
    return dtype == PTIR_DT_BOOL ? 1u : 4u;
}

inline std::size_t grouped_channel_bytes(
    const GroupedLaneBinding& lane,
    std::uint32_t local_channel) {
    const auto dense = lane.plan->channel_bindings[local_channel];
    const Channel& channel = lane.instance->trace().channels[dense];
    return channel.type.shape.numel() *
        (channel.type.dtype == DType::Bool ? 1u : 4u);
}

inline std::uint32_t grouped_extent(
    std::uint8_t extent,
    const GroupedLaneBinding& lane) {
    switch (extent) {
        case PTIR_EXTENT_SAMPLED_ROWS:
            return lane.logits_row_count;
        case PTIR_EXTENT_ROW_COUNT:
            return lane.row_count;
        case PTIR_EXTENT_TOKEN_COUNT:
            return lane.token_count;
        case PTIR_EXTENT_KV_LEN:
            return lane.kv_len;
        case PTIR_EXTENT_PAGE_COUNT:
            return lane.page_count;
        case PTIR_EXTENT_QUERY_LEN:
            return lane.query_len;
        case PTIR_EXTENT_KEY_LEN:
            return lane.key_len;
        default:
            return 0;
    }
}

inline std::uint64_t grouped_numel(
    const plan::ValueType& type,
    const GroupedLaneBinding& lane) {
    std::uint64_t value = 1;
    for (const plan::Dimension& dimension : type.dims) {
        const std::uint32_t size = dimension.symbolic
            ? grouped_extent(static_cast<std::uint8_t>(dimension.value), lane)
            : dimension.value;
        value *= size;
    }
    return value;
}

inline GroupedDynamicShape grouped_dynamic_shape(
    const plan::ValueType& type,
    const std::vector<GroupedLaneBinding>& lanes) {
    GroupedDynamicShape shape{};
    std::uint32_t varying_count = 0;
    std::uint64_t static_numel = 1;
    for (const plan::Dimension& dimension : type.dims) {
        if (!dimension.symbolic) {
            static_numel *= dimension.value;
            continue;
        }
        const auto extent = static_cast<std::uint8_t>(dimension.value);
        const auto first = grouped_extent(extent, lanes.front());
        const bool varies = std::any_of(
            lanes.begin(), lanes.end(), [&](const GroupedLaneBinding& lane) {
                return grouped_extent(extent, lane) != first;
            });
        if (varies) {
            ++varying_count;
            shape.extent = extent;
        } else {
            static_numel *= first;
        }
    }
    if (varying_count == 0) {
        shape.max_numel = static_numel;
        shape.extent = 0xff;
        return shape;
    }
    if (varying_count > 1) {
        throw std::runtime_error(
            "ragged grouped value has multiple varying symbolic extents");
    }
    if (static_numel > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error("grouped dynamic value is too large");
    }
    shape.elements_per_extent = static_cast<std::uint32_t>(static_numel);
    shape.max_numel = 0;
    for (const auto& lane : lanes) {
        shape.max_numel =
            std::max(shape.max_numel, grouped_numel(type, lane));
    }
    return shape;
}

inline GroupedDynamicShape grouped_dimensions_shape(
    const plan::ValueType& type,
    std::size_t begin,
    std::size_t end,
    const std::vector<GroupedLaneBinding>& lanes) {
    GroupedDynamicShape shape;
    std::uint64_t static_numel = 1;
    for (std::size_t index = begin; index < end; ++index) {
        const auto& dimension = type.dims[index];
        if (dimension.symbolic) {
            if (shape.extent != 0xff) {
                throw std::runtime_error(
                    "grouped row layout has multiple symbolic dimensions");
            }
            shape.extent = static_cast<std::uint8_t>(dimension.value);
        } else {
            static_numel *= dimension.value;
        }
    }
    shape.elements_per_extent =
        static_cast<std::uint32_t>(static_numel);
    shape.max_numel = shape.extent == 0xff ? static_numel : 0;
    if (shape.extent != 0xff) {
        for (const auto& lane : lanes) {
            shape.max_numel = std::max(
                shape.max_numel,
                static_cast<std::uint64_t>(
                    grouped_extent(shape.extent, lane)) *
                    shape.elements_per_extent);
        }
    }
    return shape;
}

inline GroupedRowShape grouped_row_shape(
    const plan::ValueType& type,
    const std::vector<GroupedLaneBinding>& lanes) {
    GroupedRowShape result;
    if (type.dims.size() < 2) {
        result.rows.max_numel = 1;
        result.rows.elements_per_extent = 1;
        result.rows.extent = 0xff;
        result.columns =
            grouped_dimensions_shape(type, 0, type.dims.size(), lanes);
    } else {
        result.rows = grouped_dimensions_shape(
            type, 0, type.dims.size() - 1, lanes);
        result.columns = grouped_dimensions_shape(
            type, type.dims.size() - 1, type.dims.size(), lanes);
    }
    result.max_rows = std::max<std::uint32_t>(
        1, static_cast<std::uint32_t>(result.rows.max_numel));
    result.max_columns =
        static_cast<std::uint32_t>(result.columns.max_numel);
    return result;
}

inline std::uint32_t grouped_dimension(
    const plan::ValueType& type,
    std::size_t index,
    const GroupedLaneBinding& lane) {
    const plan::Dimension& dimension = type.dims[index];
    return dimension.symbolic
        ? grouped_extent(static_cast<std::uint8_t>(dimension.value), lane)
        : dimension.value;
}

inline std::uint32_t grouped_rows(
    const plan::ValueType& type,
    const GroupedLaneBinding& lane) {
    if (type.dims.size() < 2) return 1;
    std::uint64_t rows = 1;
    for (std::size_t index = 0; index + 1 < type.dims.size(); ++index) {
        const plan::Dimension& dimension = type.dims[index];
        rows *= dimension.symbolic
            ? grouped_extent(static_cast<std::uint8_t>(dimension.value), lane)
            : dimension.value;
    }
    return std::max<std::uint32_t>(
        1, static_cast<std::uint32_t>(rows));
}

inline bool grouped_supported_tag(std::uint8_t tag) {
    switch (tag) {
        case PTIR_OP_CONST:
        case PTIR_OP_CHAN_TAKE:
        case PTIR_OP_CHAN_READ:
        case PTIR_OP_CHAN_PUT:
        case PTIR_OP_SINK_CALL:
        case PTIR_OP_INTRINSIC_VAL:
        case PTIR_OP_RESHAPE:
        case PTIR_OP_BROADCAST:
        case PTIR_OP_TRANSPOSE:
        case PTIR_OP_CAST:
        case PTIR_OP_IOTA:
        case PTIR_OP_RNG:
        case PTIR_OP_RNG_KEYED:
        case PTIR_OP_ADD:
        case PTIR_OP_SUB:
        case PTIR_OP_MUL:
        case PTIR_OP_DIV:
        case PTIR_OP_REM:
        case PTIR_OP_MAX_ELEM:
        case PTIR_OP_MIN_ELEM:
        case PTIR_OP_GT:
        case PTIR_OP_GE:
        case PTIR_OP_EQ:
        case PTIR_OP_NE:
        case PTIR_OP_LT:
        case PTIR_OP_LE:
        case PTIR_OP_AND:
        case PTIR_OP_OR:
        case PTIR_OP_NOT:
        case PTIR_OP_NEG:
        case PTIR_OP_EXP:
        case PTIR_OP_LOG:
        case PTIR_OP_RECIP:
        case PTIR_OP_ABS:
        case PTIR_OP_SIGN:
        case PTIR_OP_SELECT:
        case PTIR_OP_REDUCE_SUM:
        case PTIR_OP_REDUCE_MAX:
        case PTIR_OP_REDUCE_MIN:
        case PTIR_OP_REDUCE_ARGMAX:
        case PTIR_OP_CUMSUM:
        case PTIR_OP_CUMPROD:
        case PTIR_OP_GATHER:
        case PTIR_OP_GATHER_ROW:
        case PTIR_OP_SCATTER_SET:
        case PTIR_OP_SCATTER_ADD:
        case PTIR_OP_MASK_APPLY_PACKED:
        case PTIR_OP_CAUSAL_MASK:
        case PTIR_OP_SLIDING_WINDOW_MASK:
        case PTIR_OP_SINK_WINDOW_MASK:
        case PTIR_OP_MATMUL:
        case PTIR_OP_SORT_DESC:
        case PTIR_OP_TOP_K:
        case PTIR_OP_PIVOT_THRESHOLD:
            return true;
        default:
            return false;
    }
}

struct GroupedStageStaticPlan {
    struct ChannelRule {
        std::uint32_t value = UINT32_MAX;
        std::uint32_t local = UINT32_MAX;
    };

    explicit GroupedStageStaticPlan(const plan::StagePlan& stage)
        : signature_hash(stage.signature_hash),
          signature(stage.signature),
          op_count(stage.ops.size()),
          value_count(stage.value_types.size()),
          channel_count(stage.channel_bindings.size()) {
        auto fail = [&](const char* message) {
            valid = false;
            error = message;
        };
        if (stage.stage > PTIR_STAGE_EPILOGUE) {
            fail("missing executable stage plan");
            return;
        }
        value_extents.resize(stage.value_types.size());
        std::array<bool, PTIR_EXTENT_KEY_LEN + 1> seen_extents{};
        for (std::size_t value = 0;
             value < stage.value_types.size();
             ++value) {
            for (const plan::Dimension& dimension :
                 stage.value_types[value].dims) {
                if (!dimension.symbolic) continue;
                if (dimension.value > PTIR_EXTENT_KEY_LEN) {
                    fail("stage uses unsupported runtime extents");
                    return;
                }
                const auto extent =
                    static_cast<std::uint8_t>(dimension.value);
                value_extents[value].push_back(extent);
                if (!seen_extents[extent]) {
                    seen_extents[extent] = true;
                    used_extents.push_back(extent);
                }
            }
        }

        std::vector<std::uint32_t> value_bases(stage.ops.size());
        std::uint32_t next_value = 0;
        for (std::size_t node = 0; node < stage.ops.size(); ++node) {
            value_bases[node] = next_value;
            next_value += stage.ops[node].op.results;
        }
        for (std::size_t node = 0; node < stage.ops.size(); ++node) {
            const container::COp& op = stage.ops[node].op;
            if (!grouped_supported_tag(op.tag)) {
                fail("stage contains an unsupported grouped op");
                return;
            }
            if (op.tag == PTIR_OP_INTRINSIC_VAL) {
                if (op.intr == PTIR_INTR_QUERY) {
                    requires_query = true;
                } else if (op.intr == PTIR_INTR_LAYER) {
                    requires_layer = true;
                } else if (op.intr == PTIR_INTR_MTP_LOGITS) {
                    const std::uint32_t value = value_bases[node];
                    if (value >= stage.value_types.size() ||
                        stage.value_types[value].dims.size() != 2 ||
                        stage.value_types[value].dims[0].symbolic) {
                        fail("MtpLogits has no static draft-row layout");
                        return;
                    }
                    const std::size_t required_rows =
                        stage.value_types[value].dims[0].value;
                    if (requires_mtp_rows &&
                        mtp_rows != required_rows) {
                        fail(
                            "MtpLogits stages declare incompatible draft-row layouts");
                        return;
                    }
                    requires_mtp_rows = true;
                    mtp_rows = required_rows;
                } else if (op.intr != PTIR_INTR_LOGITS) {
                    fail("stage uses an unsupported intrinsic");
                    return;
                }
            }

            std::uint32_t value = UINT32_MAX;
            if (op.tag == PTIR_OP_CHAN_TAKE ||
                op.tag == PTIR_OP_CHAN_READ) {
                value = value_bases[node];
            } else if (op.tag == PTIR_OP_CHAN_PUT &&
                       !op.args.empty()) {
                value = op.args[0];
            } else {
                continue;
            }
            if (value >= stage.value_types.size() ||
                op.chan < 0 ||
                static_cast<std::size_t>(op.chan) >=
                    stage.channel_bindings.size()) {
                fail("channel value is outside the grouped plan");
                return;
            }
            channel_rules.push_back(ChannelRule{
                .value = value,
                .local = static_cast<std::uint32_t>(op.chan),
            });
        }
    }

    bool matches(const plan::StagePlan& stage) const {
        return stage.signature_hash == signature_hash &&
            stage.signature == signature &&
            stage.ops.size() == op_count &&
            stage.value_types.size() == value_count &&
            stage.channel_bindings.size() == channel_count;
    }

    bool valid = true;
    std::string error;
    std::uint64_t signature_hash = 0;
    std::vector<std::uint8_t> signature;
    std::size_t op_count = 0;
    std::size_t value_count = 0;
    std::size_t channel_count = 0;
    bool requires_query = false;
    bool requires_layer = false;
    bool requires_mtp_rows = false;
    std::size_t mtp_rows = 0;
    std::vector<std::uint8_t> used_extents;
    std::vector<std::vector<std::uint8_t>> value_extents;
    std::vector<ChannelRule> channel_rules;
};

class GroupedStageAccumulator {
  public:
    GroupedStageAccumulator() = default;
    explicit GroupedStageAccumulator(
        const GroupedStageStaticPlan& static_plan)
        : static_plan_(&static_plan) {}

    bool try_add(
        const GroupedLaneBinding& lane,
        std::string* reason = nullptr) {
        auto fail = [&](const char* message) {
            if (reason != nullptr) *reason = message;
            return false;
        };
        bool provisional_plan = false;
        if (static_plan_ == nullptr) {
            if (lane.plan == nullptr) {
                return fail("missing executable stage plan");
            }
            owned_plan_ =
                std::make_unique<GroupedStageStaticPlan>(*lane.plan);
            static_plan_ = owned_plan_.get();
            provisional_plan = true;
        }
        if (!static_plan_->valid) {
            if (reason != nullptr) *reason = static_plan_->error;
            if (provisional_plan) reset_owned_plan();
            return false;
        }
        if (!validate_lane(lane, reason)) {
            if (provisional_plan) reset_owned_plan();
            return false;
        }
        if (!prepare_candidate_slots(lane, reason)) {
            if (provisional_plan) reset_owned_plan();
            return false;
        }
        if (first_ == nullptr) {
            initialize(lane);
            return true;
        }
        if (lane.vocab != first_lane_.vocab ||
            lane.logits_stride != first_lane_.logits_stride ||
            lane.plan_identity != first_lane_.plan_identity ||
            !static_plan_->matches(*lane.plan)) {
            return fail("group plan or static shape mismatch");
        }
        if ((lane.logits_bf16_rows != nullptr) !=
            (first_lane_.logits_bf16_rows != nullptr)) {
            return fail("group mixes FP32 and direct BF16 logits");
        }
        if ((lane.mtp_logits_bf16_rows != nullptr) !=
            (first_lane_.mtp_logits_bf16_rows != nullptr)) {
            return fail("group mixes MTP draft-row layouts");
        }

        auto next_min = extent_min_;
        auto next_max = extent_max_;
        for (const std::uint8_t extent :
             static_plan_->used_extents) {
            const std::uint32_t value = grouped_extent(extent, lane);
            next_min[extent] = std::min(next_min[extent], value);
            next_max[extent] = std::max(next_max[extent], value);
        }
        for (const auto& extents :
             static_plan_->value_extents) {
            std::uint32_t varying = 0;
            for (const std::uint8_t extent : extents) {
                varying += next_min[extent] != next_max[extent] ? 1u : 0u;
            }
            if (varying > 1) {
                return fail(
                    "stage has multiple varying symbolic dimensions in one value");
            }
        }

        for (std::size_t index = 0;
             index < static_plan_->channel_rules.size();
             ++index) {
            const auto& rule =
                static_plan_->channel_rules[index];
            const plan::ValueType& value_type =
                first_->value_types[rule.value];
            const std::size_t value_bytes =
                grouped_numel(value_type, lane) *
                grouped_dtype_size(value_type.dtype);
            const std::size_t channel_bytes =
                grouped_channel_bytes(lane, rule.local);
            if (std::max(channel_limits_[index].first, value_bytes) >
                std::min(channel_limits_[index].second, channel_bytes)) {
                return fail("channel value size does not match fixed cell");
            }
        }

        extent_min_ = next_min;
        extent_max_ = next_max;
        for (std::size_t index = 0;
             index < static_plan_->channel_rules.size();
             ++index) {
            const auto& rule =
                static_plan_->channel_rules[index];
            const plan::ValueType& value_type =
                first_->value_types[rule.value];
            channel_limits_[index].first = std::max(
                channel_limits_[index].first,
                grouped_numel(value_type, lane) *
                    grouped_dtype_size(value_type.dtype));
            channel_limits_[index].second = std::min(
                channel_limits_[index].second,
                grouped_channel_bytes(lane, rule.local));
        }
        for (const auto& [slot, mutates] : candidate_slots_) {
            if (slot >= slot_states_.size()) {
                slot_states_.resize(
                    static_cast<std::size_t>(slot) + 1, -1);
            }
            slot_states_[slot] = mutates ? 1 : 0;
        }
        ++size_;
        return true;
    }

    std::size_t size() const noexcept { return size_; }

  private:
    void reset_owned_plan() {
        static_plan_ = nullptr;
        owned_plan_.reset();
    }

    bool prepare_candidate_slots(
        const GroupedLaneBinding& lane,
        std::string* reason) {
        auto fail = [&](const char* message) {
            if (reason != nullptr) *reason = message;
            return false;
        };
        candidate_slots_.clear();
        candidate_slots_.reserve(lane.instance->view().slots().size());
        for (const std::uint32_t slot :
             lane.instance->view().slots()) {
            const auto ticket = std::find_if(
                lane.tickets->begin(),
                lane.tickets->end(),
                [slot](const DeviceHostChannelTicket& candidate) {
                    return candidate.slot == slot;
                });
            const bool mutates =
                ticket != lane.tickets->end() &&
                (ticket->flags & (kTicketConsume | kTicketPublish)) != 0;
            if (slot < slot_states_.size() &&
                slot_states_[slot] >= 0 &&
                (slot_states_[slot] != 0 || mutates)) {
                return fail(
                    "shared mutable device channel slot requires ordered execution");
            }
            const auto local = std::find_if(
                candidate_slots_.begin(),
                candidate_slots_.end(),
                [slot](const auto& entry) {
                    return entry.first == slot;
                });
            if (local != candidate_slots_.end() &&
                (local->second || mutates)) {
                return fail(
                    "shared mutable device channel slot requires ordered execution");
            }
            if (local == candidate_slots_.end()) {
                candidate_slots_.emplace_back(slot, mutates);
            }
        }
        return true;
    }

    bool validate_lane(
        const GroupedLaneBinding& lane,
        std::string* reason) const {
        auto fail = [&](const char* message) {
            if (reason != nullptr) *reason = message;
            return false;
        };
        if (lane.instance == nullptr ||
            lane.plan == nullptr ||
            lane.tickets == nullptr ||
            !static_plan_->matches(*lane.plan)) {
            return fail("group plan or static shape mismatch");
        }
        const std::uint32_t physical_logits_stride =
            lane.logits_stride == 0 ? lane.vocab : lane.logits_stride;
        if (physical_logits_stride < lane.vocab) {
            return fail("physical logits stride is shorter than logical vocab");
        }
        if (lane.instance->trace().stages.empty()) {
            return fail("program has no executable stages");
        }
        if (static_plan_->requires_query &&
            lane.query_base == nullptr) {
            return fail("Query intrinsic is unavailable");
        }
        if (static_plan_->requires_layer &&
            lane.layer_base == nullptr) {
            return fail("Layer intrinsic is unavailable");
        }
        if (static_plan_->requires_mtp_rows &&
            (lane.mtp_logits_bf16_rows == nullptr ||
             lane.mtp_logits_bf16_rows->size() !=
                 static_plan_->mtp_rows)) {
            return fail(
                "MtpLogits dedicated row count does not match plan");
        }
        for (const std::uint8_t extent :
             static_plan_->used_extents) {
            if (grouped_extent(extent, lane) ==
                kUnavailableGroupedExtent) {
                return fail("stage runtime extent is unavailable");
            }
        }
        for (const auto& rule : static_plan_->channel_rules) {
            const plan::ValueType& value_type =
                lane.plan->value_types[rule.value];
            const std::size_t value_bytes =
                grouped_numel(value_type, lane) *
                grouped_dtype_size(value_type.dtype);
            if (value_bytes >
                grouped_channel_bytes(lane, rule.local)) {
                return fail(
                    "channel value size does not match fixed cell");
            }
        }
        return true;
    }

    void initialize(const GroupedLaneBinding& lane) {
        first_ = lane.plan;
        first_lane_ = lane;
        for (const std::uint8_t extent :
             static_plan_->used_extents) {
            extent_min_[extent] = grouped_extent(extent, lane);
            extent_max_[extent] = extent_min_[extent];
        }
        channel_limits_.reserve(
            static_plan_->channel_rules.size());
        for (const auto& rule : static_plan_->channel_rules) {
            const plan::ValueType& value_type =
                first_->value_types[rule.value];
            channel_limits_.emplace_back(
                grouped_numel(value_type, lane) *
                    grouped_dtype_size(value_type.dtype),
                grouped_channel_bytes(lane, rule.local));
        }
        slot_states_.assign(
            lane.instance->view().registry()->slot_capacity(), -1);
        for (const auto& [slot, mutates] : candidate_slots_) {
            if (slot >= slot_states_.size()) {
                slot_states_.resize(
                    static_cast<std::size_t>(slot) + 1, -1);
            }
            slot_states_[slot] = mutates ? 1 : 0;
        }
        size_ = 1;
    }

    const plan::StagePlan* first_ = nullptr;
    GroupedLaneBinding first_lane_{};
    const GroupedStageStaticPlan* static_plan_ = nullptr;
    std::unique_ptr<GroupedStageStaticPlan> owned_plan_;
    std::array<std::uint32_t, PTIR_EXTENT_KEY_LEN + 1> extent_min_{};
    std::array<std::uint32_t, PTIR_EXTENT_KEY_LEN + 1> extent_max_{};
    std::vector<std::pair<std::size_t, std::size_t>> channel_limits_;
    std::vector<std::int8_t> slot_states_;
    std::vector<std::pair<std::uint32_t, bool>> candidate_slots_;
    std::size_t size_ = 0;
};

inline bool grouped_nucleus_region_supported(
    const plan::StagePlan& stage,
    const plan::Region& region) {
    if (!region.library ||
        region.library_op != PTIR_LIBRARY_NUCLEUS_SAMPLE ||
        (region.inputs.size() != 3 && region.inputs.size() != 5) ||
        region.nodes.size() != 13 ||
        region.outputs.size() != 1 ||
        !region.sinks.empty()) {
        return false;
    }
    const bool scaled = region.inputs.size() == 5;
    const auto logits = region.inputs[0];
    const auto top_p = region.inputs[scaled ? 3 : 1];
    const auto rng_state = region.inputs[scaled ? 4 : 2];
    const auto token = region.outputs[0];
    if (logits >= stage.value_types.size() ||
        top_p >= stage.value_types.size() ||
        rng_state >= stage.value_types.size() ||
        token >= stage.value_types.size()) {
        return false;
    }
    const auto& raw_logits_type = stage.value_types[logits];
    const auto& logits_type = stage.value_types[
        region.inputs[scaled ? 2 : 0]];
    return raw_logits_type.dtype == PTIR_DT_F32 &&
        logits_type.dtype == PTIR_DT_F32 &&
        !raw_logits_type.dims.empty() &&
        !logits_type.dims.empty() &&
        !logits_type.dims.back().symbolic &&
        raw_logits_type.dims.back().symbolic ==
            logits_type.dims.back().symbolic &&
        raw_logits_type.dims.back().value ==
            logits_type.dims.back().value &&
        (!scaled ||
         stage.value_types[region.inputs[1]].dtype == PTIR_DT_F32) &&
        (!scaled ||
         stage.value_types[region.inputs[2]].dtype == PTIR_DT_F32) &&
        stage.value_types[top_p].dtype == PTIR_DT_F32 &&
        stage.value_types[rng_state].dtype == PTIR_DT_U32 &&
        (stage.value_types[token].dtype == PTIR_DT_I32 ||
         stage.value_types[token].dtype == PTIR_DT_U32);
}

inline bool grouped_nucleus_library_supported(
    const plan::StagePlan& stage,
    const plan::Region& region) {
    return grouped_nucleus_region_supported(stage, region) &&
        stage.value_types[
            region.inputs[region.inputs.size() == 5 ? 2 : 0]]
            .dims.back().value <=
            kMaxExactNucleusLibraryVocab;
}

}  // namespace pie_cuda_driver::pipeline
