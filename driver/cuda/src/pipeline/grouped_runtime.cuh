#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <memory>
#include <mutex>
#include <numeric>
#include <optional>
#include <string>
#include <type_traits>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <cub/device/device_segmented_radix_sort.cuh>

#include "cuda_check.hpp"
#include "pie_native/ptir/plan.hpp"
#include "pipeline/channels.hpp"
#include "pipeline/grouped_copy.hpp"
#include "pipeline/library_region.hpp"
#include "pipeline/program_runtime.hpp"
#include "pipeline/tier0/tier0_kernels.cuh"

namespace pie_cuda_driver::pipeline {

inline constexpr std::uint32_t kUnavailableGroupedExtent = UINT32_MAX;
inline constexpr std::uint32_t kGroupedLaneFlagBf16Rows = 1u << 0;
inline constexpr std::uint32_t kMaxExactNucleusLibraryVocab = 4096;

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
};

struct GroupedFinalizeLane {
    std::uint32_t taken_offset = 0;
    std::uint32_t taken_count = 0;
    std::uint32_t put_offset = 0;
    std::uint32_t put_count = 0;
};

struct GroupedReadinessLane {
    std::uint32_t full_offset = 0;
    std::uint32_t full_count = 0;
    std::uint32_t empty_offset = 0;
    std::uint32_t empty_count = 0;
};

struct GroupedGraphUpload {
    void* destination = nullptr;
    const void* source = nullptr;
    std::size_t bytes = 0;
};

struct GroupedBroadcastMeta {
    std::uint32_t rank = 0;
    std::uint32_t out_dims[4] = {1, 1, 1, 1};
    std::uint32_t src_strides[4] = {0, 0, 0, 0};
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

struct GroupedPutDesc {
    std::uint32_t value = 0;
    std::uint32_t channel = 0;
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

static __global__ void k_grouped_reset(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes) {
    const std::uint32_t lane =
        blockIdx.x * static_cast<std::uint32_t>(blockDim.x) + threadIdx.x;
    if (lane < header->lane_count) {
        *grouped_commit(lanes, lane) = 1;
    }
}

static __global__ void k_grouped_pull_validate(
    const DeviceHostChannelTicket* tickets,
    const std::uint32_t* ticket_lanes,
    std::uint32_t count,
    const PtirLaneRecord* lanes,
    std::uint8_t* full) {
    const std::uint32_t index = blockIdx.x;
    if (index >= count) return;
    const DeviceHostChannelTicket ticket = tickets[index];
    if (ticket.flags == 0 || ticket.words == nullptr) return;
    const std::uint32_t lane = ticket_lanes[index];
    std::uint32_t* pass_commit = grouped_commit(lanes, lane);
    __shared__ std::uint32_t valid;
    if (threadIdx.x == 0) {
        const std::uint64_t head = load_system_acquire(ticket.words + 0);
        const std::uint64_t tail = load_system_acquire(ticket.words + 1);
        bool ok = true;
        if ((ticket.flags & kTicketConsume) != 0) {
            ok = head == ticket.expected_head;
        }

        if ((ticket.flags & kTicketRequireInput) != 0) {
            ok = ok && tail > head;
        }
        if ((ticket.flags & kTicketPublish) != 0) {
            const std::uint64_t same_fire_consume =
                (ticket.flags & kTicketConsume) != 0 ? 1u : 0u;
            ok = ok && tail == ticket.expected_tail &&
                 tail - head <
                     static_cast<std::uint64_t>(ticket.cap1 - 1) +
                         same_fire_consume;
        }
        valid = ok ? 1u : 0u;
        if (!ok) atomicAnd(pass_commit, 0u);
    }
    __syncthreads();
    if (valid == 0 || (ticket.flags & kTicketHostWriter) == 0 ||
        (ticket.flags & kTicketConsume) == 0) {
        return;
    }

    const std::uint32_t ring = static_cast<std::uint32_t>(
        ticket.expected_head % ticket.cap1);
    const std::uint8_t* source =
        ticket.mirror + static_cast<std::size_t>(ring) * ticket.wire_bytes;
    std::uint8_t* destination =
        ticket.cells + static_cast<std::size_t>(ring) * ticket.native_bytes;
    if ((ticket.flags & kTicketPackedBool) != 0) {
        for (std::uint32_t i = threadIdx.x; i < ticket.native_bytes;
             i += blockDim.x) {
            destination[i] = static_cast<std::uint8_t>(
                (source[i / 8] >> (i % 8)) & 1u);
        }
    } else {
        for (std::uint32_t i = threadIdx.x; i < ticket.native_bytes;
             i += blockDim.x) {
            destination[i] = source[i];
        }
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        full[static_cast<std::size_t>(ticket.slot) * kMaxRing + ring] = 1;
    }
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

static __global__ void k_grouped_copy_dynamic_root(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t result_value,
    const std::uint64_t* sources,
    GroupedDynamicShape shape,
    std::uint32_t element_bytes,
    std::uint32_t logical_row_width,
    std::uint32_t source_row_stride) {
    const std::uint64_t total =
        static_cast<std::uint64_t>(header->lane_count) * shape.max_numel;
    for (std::uint64_t flat =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         flat < total;
         flat += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
        const auto lane = static_cast<std::uint32_t>(flat / shape.max_numel);
        const auto index = flat % shape.max_numel;
        auto* dst = grouped_value<std::uint8_t>(
            values, value_count, lane, result_value);
        const auto* src = reinterpret_cast<const std::uint8_t*>(sources[lane]);
        if (dst == nullptr) continue;
        const auto valid = grouped_lane_numel(lanes[lane], shape);
        const std::uint64_t source_index =
            grouped_row_strided_source_index(
                index, logical_row_width, source_row_stride);
        for (std::uint32_t byte = 0; byte < element_bytes; ++byte) {
            dst[index * element_bytes + byte] =
                src != nullptr && index < valid
                    ? src[source_index * element_bytes + byte]
                    : 0;
        }
    }
}

static __global__ void k_grouped_materialize_reshape(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* mtp_rows,
    const std::uint32_t* mtp_offsets,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t source,
    std::uint32_t output,
    GroupedDynamicShape shape,
    std::uint32_t element_bytes,
    std::uint32_t vocab,
    std::uint8_t direct_kind) {
    const std::uint64_t total =
        static_cast<std::uint64_t>(header->lane_count) * shape.max_numel;
    for (std::uint64_t flat =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         flat < total;
         flat += static_cast<std::uint64_t>(blockDim.x) * gridDim.x) {
        const auto lane = static_cast<std::uint32_t>(
            flat / shape.max_numel);
        if (*grouped_commit(lanes, lane) == 0) continue;
        const std::uint64_t index = flat % shape.max_numel;
        if (index >= grouped_lane_numel(lanes[lane], shape)) continue;
        if (direct_kind != 0) {
            grouped_value<float>(
                values, value_count, lane, output)[index] =
                grouped_direct_bf16_load(
                    lanes, mtp_rows, mtp_offsets, lane, index, vocab,
                    direct_kind);
            continue;
        }
        const auto* input = grouped_value<const std::uint8_t>(
            values, value_count, lane, source);
        auto* destination = grouped_value<std::uint8_t>(
            values, value_count, lane, output);
        if (input == nullptr || destination == nullptr) continue;
        for (std::uint32_t byte = 0; byte < element_bytes; ++byte) {
            destination[index * element_bytes + byte] =
                input[index * element_bytes + byte];
        }
    }
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

template <class T>
static __global__ void k_grouped_constant(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t output,
    std::uint64_t numel,
    T literal) {
    const std::uint64_t total =
        static_cast<std::uint64_t>(header->lane_count) * numel;
    for (std::uint64_t flat =
             blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
         flat < total;
         flat += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const std::uint32_t lane = static_cast<std::uint32_t>(flat / numel);
        if (*grouped_commit(lanes, lane) == 0) continue;
        const std::uint64_t index = flat % numel;
        grouped_value<T>(values, value_count, lane, output)[index] = literal;
    }
}

template <class T>
__device__ __forceinline__ T grouped_binary_value(
    BinKind kind, T a, T b) {
    if constexpr (std::is_same_v<T, float>) {
        return t0_bin<float>(kind, a, b);
    } else {
        using U = std::make_unsigned_t<T>;
        switch (kind) {
            case BinKind::Add:
                return static_cast<T>(
                    static_cast<U>(a) + static_cast<U>(b));
            case BinKind::Sub:
                return static_cast<T>(
                    static_cast<U>(a) - static_cast<U>(b));
            case BinKind::Mul:
                return static_cast<T>(
                    static_cast<U>(a) * static_cast<U>(b));
            case BinKind::Div:
                if (b == 0) return 0;
                if constexpr (std::is_signed_v<T>) {
                    if (a == std::numeric_limits<T>::min() &&
                        b == static_cast<T>(-1)) {
                        return a;
                    }
                }
                return static_cast<T>(a / b);
            case BinKind::Rem:
                if (b == 0) return 0;
                if constexpr (std::is_signed_v<T>) {
                    if (a == std::numeric_limits<T>::min() &&
                        b == static_cast<T>(-1)) {
                        return 0;
                    }
                }
                return static_cast<T>(a % b);
            case BinKind::MaxElem:
                return a > b ? a : b;
            case BinKind::MinElem:
                return a < b ? a : b;
        }
        return a;
    }
}

template <class T>
static __global__ void k_grouped_binary(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t a,
    std::uint32_t b,
    std::uint32_t output,
    std::uint64_t numel,
    BinKind kind,
    bool a_scalar,
    bool b_scalar) {
    const std::uint64_t total =
        static_cast<std::uint64_t>(header->lane_count) * numel;
    for (std::uint64_t flat =
             blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
         flat < total;
         flat += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const std::uint32_t lane = static_cast<std::uint32_t>(flat / numel);
        if (*grouped_commit(lanes, lane) == 0) continue;
        const std::uint64_t index = flat % numel;
        const T* av = grouped_value<T>(values, value_count, lane, a);
        const T* bv = grouped_value<T>(values, value_count, lane, b);
        T* out = grouped_value<T>(values, value_count, lane, output);
        out[index] = grouped_binary_value<T>(
            kind, av[a_scalar ? 0 : index], bv[b_scalar ? 0 : index]);
    }
}

template <class T>
static __global__ void k_grouped_compare(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t a,
    std::uint32_t b,
    std::uint32_t output,
    std::uint64_t numel,
    CmpKind kind,
    bool a_scalar,
    bool b_scalar) {
    const std::uint64_t total =
        static_cast<std::uint64_t>(header->lane_count) * numel;
    for (std::uint64_t flat =
             blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
         flat < total;
         flat += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const std::uint32_t lane = static_cast<std::uint32_t>(flat / numel);
        if (*grouped_commit(lanes, lane) == 0) continue;
        const std::uint64_t index = flat % numel;
        const T av = grouped_value<T>(values, value_count, lane, a)[
            a_scalar ? 0 : index];
        const T bv = grouped_value<T>(values, value_count, lane, b)[
            b_scalar ? 0 : index];
        bool result = false;
        switch (kind) {
            case CmpKind::Eq: result = av == bv; break;
            case CmpKind::Ne: result = av != bv; break;
            case CmpKind::Lt: result = av < bv; break;
            case CmpKind::Le: result = av <= bv; break;
            case CmpKind::Gt: result = av > bv; break;
            case CmpKind::Ge: result = av >= bv; break;
        }
        grouped_value<std::uint8_t>(
            values, value_count, lane, output)[index] =
            result ? 1u : 0u;
    }
}

static __global__ void k_grouped_logic(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t a,
    std::uint32_t b,
    std::uint32_t output,
    std::uint64_t numel,
    LogicKind kind) {
    const std::uint64_t total =
        static_cast<std::uint64_t>(header->lane_count) * numel;
    for (std::uint64_t flat =
             blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
         flat < total;
         flat += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const std::uint32_t lane = static_cast<std::uint32_t>(flat / numel);
        if (*grouped_commit(lanes, lane) == 0) continue;
        const std::uint64_t index = flat % numel;
        const bool av =
            grouped_value<std::uint8_t>(values, value_count, lane, a)[index] != 0;
        const bool bv =
            grouped_value<std::uint8_t>(values, value_count, lane, b)[index] != 0;
        grouped_value<std::uint8_t>(
            values, value_count, lane, output)[index] =
            (kind == LogicKind::And ? av && bv : av || bv) ? 1u : 0u;
    }
}

static __global__ void k_grouped_not(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t input,
    std::uint32_t output,
    std::uint64_t numel) {
    const std::uint64_t total =
        static_cast<std::uint64_t>(header->lane_count) * numel;
    for (std::uint64_t flat =
             blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
         flat < total;
         flat += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const std::uint32_t lane = static_cast<std::uint32_t>(flat / numel);
        if (*grouped_commit(lanes, lane) == 0) continue;
        const std::uint64_t index = flat % numel;
        grouped_value<std::uint8_t>(
            values, value_count, lane, output)[index] =
            grouped_value<std::uint8_t>(
                values, value_count, lane, input)[index] == 0;
    }
}

template <class T>
__device__ __forceinline__ T grouped_unary_value(T value, UnKind kind) {
    return t0_unary_value(value, kind);
}

template <class T>
static __global__ void k_grouped_unary(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t input,
    std::uint32_t output,
    std::uint64_t numel,
    UnKind kind) {
    const std::uint64_t total =
        static_cast<std::uint64_t>(header->lane_count) * numel;
    for (std::uint64_t flat =
             blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
         flat < total;
         flat += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const std::uint32_t lane = static_cast<std::uint32_t>(flat / numel);
        if (*grouped_commit(lanes, lane) == 0) continue;
        const std::uint64_t index = flat % numel;
        const T value =
            grouped_value<T>(values, value_count, lane, input)[index];
        grouped_value<T>(values, value_count, lane, output)[index] =
            grouped_unary_value(value, kind);
    }
}

template <class T>
static __global__ void k_grouped_select(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t condition,
    std::uint32_t a,
    std::uint32_t b,
    std::uint32_t output,
    std::uint64_t numel,
    bool condition_scalar,
    bool a_scalar,
    bool b_scalar) {
    const std::uint64_t total =
        static_cast<std::uint64_t>(header->lane_count) * numel;
    for (std::uint64_t flat =
             blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
         flat < total;
         flat += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const std::uint32_t lane = static_cast<std::uint32_t>(flat / numel);
        if (*grouped_commit(lanes, lane) == 0) continue;
        const std::uint64_t index = flat % numel;
        const std::uint8_t* cv =
            grouped_value<std::uint8_t>(values, value_count, lane, condition);
        const T* av = grouped_value<T>(values, value_count, lane, a);
        const T* bv = grouped_value<T>(values, value_count, lane, b);
        grouped_value<T>(values, value_count, lane, output)[index] =
            cv[condition_scalar ? 0 : index]
                ? av[a_scalar ? 0 : index]
                : bv[b_scalar ? 0 : index];
    }
}

__device__ __forceinline__ float grouped_direct_or_f32(
    const PtirLaneRecord* lanes,
    const std::uint64_t* mtp_rows,
    const std::uint32_t* mtp_offsets,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t lane,
    std::uint32_t value,
    std::uint64_t index,
    std::uint32_t vocab,
    std::uint8_t direct_kind,
    bool scalar) {
    return direct_kind == 0
        ? grouped_value<float>(
              values, value_count, lane, value)[scalar ? 0 : index]
        : grouped_direct_bf16_load(
              lanes, mtp_rows, mtp_offsets, lane,
              scalar ? 0 : index, vocab, direct_kind);
}

static __global__ void k_grouped_direct_unary(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* mtp_rows,
    const std::uint32_t* mtp_offsets,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t input,
    std::uint32_t output,
    std::uint64_t numel,
    GroupedDynamicShape output_shape,
    std::uint32_t vocab,
    std::uint8_t input_kind,
    UnKind kind) {
    const std::uint64_t total =
        static_cast<std::uint64_t>(header->lane_count) * numel;
    for (std::uint64_t flat =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         flat < total;
         flat += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const std::uint32_t lane =
            static_cast<std::uint32_t>(flat / numel);
        if (*grouped_commit(lanes, lane) == 0) continue;
        const std::uint64_t index = flat % numel;
        if (index >= grouped_lane_numel(lanes[lane], output_shape)) continue;
        grouped_value<float>(
            values, value_count, lane, output)[index] =
            grouped_unary_value(
                grouped_direct_bf16_load(
                    lanes, mtp_rows, mtp_offsets, lane, index,
                    vocab, input_kind),
                kind);
    }
}

static __global__ void k_grouped_direct_binary(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* mtp_rows,
    const std::uint32_t* mtp_offsets,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t a,
    std::uint32_t b,
    std::uint32_t output,
    std::uint64_t numel,
    GroupedDynamicShape output_shape,
    std::uint32_t vocab,
    std::uint8_t a_kind,
    std::uint8_t b_kind,
    BinKind kind,
    bool a_scalar,
    bool b_scalar) {
    const std::uint64_t total =
        static_cast<std::uint64_t>(header->lane_count) * numel;
    for (std::uint64_t flat =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         flat < total;
         flat += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const std::uint32_t lane =
            static_cast<std::uint32_t>(flat / numel);
        if (*grouped_commit(lanes, lane) == 0) continue;
        const std::uint64_t index = flat % numel;
        if (index >= grouped_lane_numel(lanes[lane], output_shape)) continue;
        const float av = grouped_direct_or_f32(
            lanes, mtp_rows, mtp_offsets, values, value_count, lane,
            a, index, vocab, a_kind, a_scalar);
        const float bv = grouped_direct_or_f32(
            lanes, mtp_rows, mtp_offsets, values, value_count, lane,
            b, index, vocab, b_kind, b_scalar);
        grouped_value<float>(
            values, value_count, lane, output)[index] =
            grouped_binary_value(kind, av, bv);
    }
}

static __global__ void k_grouped_direct_compare(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* mtp_rows,
    const std::uint32_t* mtp_offsets,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t a,
    std::uint32_t b,
    std::uint32_t output,
    std::uint64_t numel,
    GroupedDynamicShape output_shape,
    std::uint32_t vocab,
    std::uint8_t a_kind,
    std::uint8_t b_kind,
    CmpKind kind,
    bool a_scalar,
    bool b_scalar) {
    const std::uint64_t total =
        static_cast<std::uint64_t>(header->lane_count) * numel;
    for (std::uint64_t flat =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         flat < total;
         flat += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const std::uint32_t lane =
            static_cast<std::uint32_t>(flat / numel);
        if (*grouped_commit(lanes, lane) == 0) continue;
        const std::uint64_t index = flat % numel;
        if (index >= grouped_lane_numel(lanes[lane], output_shape)) continue;
        const float av = grouped_direct_or_f32(
            lanes, mtp_rows, mtp_offsets, values, value_count, lane,
            a, index, vocab, a_kind, a_scalar);
        const float bv = grouped_direct_or_f32(
            lanes, mtp_rows, mtp_offsets, values, value_count, lane,
            b, index, vocab, b_kind, b_scalar);
        bool selected = false;
        switch (kind) {
            case CmpKind::Eq: selected = av == bv; break;
            case CmpKind::Ne: selected = av != bv; break;
            case CmpKind::Lt: selected = av < bv; break;
            case CmpKind::Le: selected = av <= bv; break;
            case CmpKind::Gt: selected = av > bv; break;
            case CmpKind::Ge: selected = av >= bv; break;
        }
        grouped_value<std::uint8_t>(
            values, value_count, lane, output)[index] = selected;
    }
}

static __global__ void k_grouped_direct_select(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* mtp_rows,
    const std::uint32_t* mtp_offsets,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t condition,
    std::uint32_t a,
    std::uint32_t b,
    std::uint32_t output,
    std::uint64_t numel,
    GroupedDynamicShape output_shape,
    std::uint32_t vocab,
    std::uint8_t a_kind,
    std::uint8_t b_kind,
    bool condition_scalar,
    bool a_scalar,
    bool b_scalar) {
    const std::uint64_t total =
        static_cast<std::uint64_t>(header->lane_count) * numel;
    for (std::uint64_t flat =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         flat < total;
         flat += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const std::uint32_t lane =
            static_cast<std::uint32_t>(flat / numel);
        if (*grouped_commit(lanes, lane) == 0) continue;
        const std::uint64_t index = flat % numel;
        if (index >= grouped_lane_numel(lanes[lane], output_shape)) continue;
        const bool selected = grouped_value<std::uint8_t>(
            values, value_count, lane, condition)[
                condition_scalar ? 0 : index] != 0;
        grouped_value<float>(
            values, value_count, lane, output)[index] =
            selected
                ? grouped_direct_or_f32(
                      lanes, mtp_rows, mtp_offsets, values, value_count,
                      lane, a, index, vocab, a_kind, a_scalar)
                : grouped_direct_or_f32(
                      lanes, mtp_rows, mtp_offsets, values, value_count,
                      lane, b, index, vocab, b_kind, b_scalar);
    }
}

template <class Source, class Target>
__device__ __forceinline__ Target grouped_cast_value(Source value) {
    if constexpr (
        std::is_same_v<Source, float> &&
        std::is_same_v<Target, std::int32_t>) {
        if (isnan(value)) return 0;
        if (value >= 2147483648.0f) {
            return std::numeric_limits<std::int32_t>::max();
        }
        if (value <= -2147483648.0f) {
            return std::numeric_limits<std::int32_t>::min();
        }
    } else if constexpr (
        std::is_same_v<Source, float> &&
        std::is_same_v<Target, std::uint32_t>) {
        if (isnan(value) || value <= 0.0f) return 0;
        if (value >= 4294967296.0f) {
            return std::numeric_limits<std::uint32_t>::max();
        }
    }
    return static_cast<Target>(value);
}

template <class Source, class Target>
static __global__ void k_grouped_cast(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t input,
    std::uint32_t output,
    std::uint64_t numel) {
    const std::uint64_t total =
        static_cast<std::uint64_t>(header->lane_count) * numel;
    for (std::uint64_t flat =
             blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
         flat < total;
         flat += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const std::uint32_t lane = static_cast<std::uint32_t>(flat / numel);
        if (*grouped_commit(lanes, lane) == 0) continue;
        const std::uint64_t index = flat % numel;
        grouped_value<Target>(values, value_count, lane, output)[index] =
            grouped_cast_value<Source, Target>(
                grouped_value<Source>(
                    values, value_count, lane, input)[index]);
    }
}

template <class Source>
static __global__ void k_grouped_cast_bool(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t input,
    std::uint32_t output,
    std::uint64_t numel) {
    const std::uint64_t total =
        static_cast<std::uint64_t>(header->lane_count) * numel;
    for (std::uint64_t flat =
             blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
         flat < total;
         flat += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const std::uint32_t lane = static_cast<std::uint32_t>(flat / numel);
        if (*grouped_commit(lanes, lane) == 0) continue;
        const std::uint64_t index = flat % numel;
        grouped_value<std::uint8_t>(
            values, value_count, lane, output)[index] =
            grouped_value<Source>(
                values, value_count, lane, input)[index] !=
            static_cast<Source>(0);
    }
}

static __global__ void k_grouped_iota(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t output,
    std::uint64_t numel,
    GroupedDynamicShape shape) {
    const std::uint64_t total =
        static_cast<std::uint64_t>(header->lane_count) * numel;
    for (std::uint64_t flat =
             blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
         flat < total;
         flat += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const std::uint32_t lane = static_cast<std::uint32_t>(flat / numel);
        if (*grouped_commit(lanes, lane) == 0) continue;
        const std::uint64_t index = flat % numel;
        if (index >= grouped_lane_numel(lanes[lane], shape)) continue;
        grouped_value<std::uint32_t>(
            values, value_count, lane, output)[index] =
            static_cast<std::uint32_t>(index);
    }
}

template <class T, class Index>
static __global__ void k_grouped_gather_axis0(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t source,
    std::uint32_t indices,
    std::uint32_t output,
    GroupedDynamicShape source_shape,
    GroupedDynamicShape axis0_shape,
    GroupedDynamicShape index_shape,
    GroupedDynamicShape output_shape) {
    const std::uint64_t numel = output_shape.max_numel;
    const std::uint64_t total =
        static_cast<std::uint64_t>(header->lane_count) * numel;
    for (std::uint64_t flat =
             blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
         flat < total;
         flat += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const std::uint32_t lane = static_cast<std::uint32_t>(flat / numel);
        if (*grouped_commit(lanes, lane) == 0) continue;
        const std::uint64_t index = flat % numel;
        const std::uint64_t output_numel =
            grouped_lane_numel(lanes[lane], output_shape);
        if (index >= output_numel) continue;
        const std::uint32_t axis0 = static_cast<std::uint32_t>(
            grouped_lane_numel(lanes[lane], axis0_shape));
        const std::uint64_t source_numel =
            grouped_lane_numel(lanes[lane], source_shape);
        if (axis0 == 0 || source_numel == 0) {
            grouped_value<T>(
                values, value_count, lane, output)[index] = T{};
            continue;
        }
        const std::uint32_t inner = static_cast<std::uint32_t>(
            source_numel / axis0);
        if (inner == 0) {
            grouped_value<T>(
                values, value_count, lane, output)[index] = T{};
            continue;
        }
        const std::uint32_t position =
            static_cast<std::uint32_t>(index / inner);
        if (position >= grouped_lane_numel(lanes[lane], index_shape)) continue;
        const std::uint32_t offset =
            static_cast<std::uint32_t>(index % inner);
        const Index raw =
            grouped_value<Index>(
                values, value_count, lane, indices)[position];
        std::uint32_t selected = 0;
        grouped_value<T>(values, value_count, lane, output)[index] =
            valid_axis0_index(raw, axis0, &selected)
                ? grouped_value<T>(
                      values, value_count, lane, source)[
                      static_cast<std::uint64_t>(selected) * inner + offset]
                : T{};
    }
}

template <class T, class Index>
static __global__ void k_grouped_gather_rows(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t source,
    std::uint32_t indices,
    std::uint32_t output,
    GroupedDynamicShape row_shape,
    GroupedDynamicShape column_shape) {
    const std::uint64_t numel = row_shape.max_numel;
    const std::uint64_t total =
        static_cast<std::uint64_t>(header->lane_count) * numel;
    for (std::uint64_t flat =
             blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
         flat < total;
         flat += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const std::uint32_t lane = static_cast<std::uint32_t>(flat / numel);
        if (*grouped_commit(lanes, lane) == 0) continue;
        const std::uint32_t row = static_cast<std::uint32_t>(flat % numel);
        const std::uint32_t rows = static_cast<std::uint32_t>(
            grouped_lane_numel(lanes[lane], row_shape));
        if (row >= rows) continue;
        const std::uint32_t columns = static_cast<std::uint32_t>(
            grouped_lane_numel(lanes[lane], column_shape));
        const Index raw =
            grouped_value<Index>(
                values, value_count, lane, indices)[row];
        std::uint32_t column = 0;
        grouped_value<T>(values, value_count, lane, output)[row] =
            valid_axis0_index(raw, columns, &column)
                ? grouped_value<T>(
                      values, value_count, lane, source)[
                      static_cast<std::uint64_t>(row) * columns + column]
                : T{};
    }
}

template <class Index>
static __global__ void k_grouped_direct_gather(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* mtp_rows,
    const std::uint32_t* mtp_offsets,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t indices,
    std::uint32_t output,
    GroupedDynamicShape source_shape,
    GroupedDynamicShape axis0_shape,
    GroupedDynamicShape index_shape,
    GroupedDynamicShape output_shape,
    std::uint32_t vocab,
    std::uint8_t source_kind,
    bool gather_row) {
    const std::uint64_t numel = output_shape.max_numel;
    const std::uint64_t total =
        static_cast<std::uint64_t>(header->lane_count) * numel;
    for (std::uint64_t flat =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         flat < total;
         flat += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const std::uint32_t lane =
            static_cast<std::uint32_t>(flat / numel);
        if (*grouped_commit(lanes, lane) == 0) continue;
        const std::uint64_t index = flat % numel;
        if (index >= grouped_lane_numel(lanes[lane], output_shape)) continue;
        const std::uint32_t axis0 = static_cast<std::uint32_t>(
            grouped_lane_numel(lanes[lane], axis0_shape));
        const std::uint64_t source_numel =
            grouped_lane_numel(lanes[lane], source_shape);
        if (axis0 == 0 || source_numel == 0) {
            grouped_value<float>(
                values, value_count, lane, output)[index] = 0.0f;
            continue;
        }
        const std::uint32_t inner = static_cast<std::uint32_t>(
            source_numel / axis0);
        if (inner == 0) {
            grouped_value<float>(
                values, value_count, lane, output)[index] = 0.0f;
            continue;
        }
        const std::uint32_t position = gather_row
            ? static_cast<std::uint32_t>(index)
            : static_cast<std::uint32_t>(index / inner);
        if (position >= grouped_lane_numel(lanes[lane], index_shape)) continue;
        const Index raw =
            grouped_value<Index>(
                values, value_count, lane, indices)[position];
        const std::uint32_t bound = gather_row ? inner : axis0;
        std::uint32_t selected = 0;
        if (!valid_axis0_index(raw, bound, &selected)) {
            grouped_value<float>(
                values, value_count, lane, output)[index] = 0.0f;
            continue;
        }
        const std::uint64_t source_index = gather_row
            ? static_cast<std::uint64_t>(position) * inner + selected
            : static_cast<std::uint64_t>(selected) * inner +
                  index % inner;
        grouped_value<float>(values, value_count, lane, output)[index] =
            grouped_direct_bf16_load(
                lanes, mtp_rows, mtp_offsets, lane, source_index,
                vocab, source_kind);
    }
}

template <class T, class Index, bool Add>
static __global__ void k_grouped_scatter(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t base,
    std::uint32_t indices,
    std::uint32_t updates,
    std::uint32_t output,
    GroupedDynamicShape base_shape,
    GroupedDynamicShape axis0_shape,
    GroupedDynamicShape index_shape,
    bool scalar_values) {
    const std::uint32_t lane = blockIdx.x;
    if (lane >= header->lane_count || *grouped_commit(lanes, lane) == 0) {
        return;
    }
    const T* base_value =
        grouped_value<T>(values, value_count, lane, base);
    T* destination =
        grouped_value<T>(values, value_count, lane, output);
    const std::uint64_t base_numel =
        grouped_lane_numel(lanes[lane], base_shape);
    const std::uint32_t axis0 = static_cast<std::uint32_t>(
        grouped_lane_numel(lanes[lane], axis0_shape));
    const std::uint32_t inner = static_cast<std::uint32_t>(
        base_numel / (axis0 == 0 ? 1u : axis0));
    const std::uint32_t update_count = static_cast<std::uint32_t>(
        grouped_lane_numel(lanes[lane], index_shape));
    for (std::uint64_t index = threadIdx.x; index < base_numel;
         index += blockDim.x) {
        destination[index] = base_value[index];
    }
    __syncthreads();
    if (threadIdx.x == 0) {
        const Index* index_values =
            grouped_value<Index>(
                values, value_count, lane, indices);
        const T* update_values =
            grouped_value<T>(values, value_count, lane, updates);
        for (std::uint32_t update = 0; update < update_count; ++update) {
            std::uint32_t target = 0;
            if (!valid_axis0_index(
                    index_values[update], axis0, &target)) {
                continue;
            }
            for (std::uint32_t offset = 0; offset < inner; ++offset) {
                const T value = update_values[
                    scalar_values
                        ? 0
                        : static_cast<std::uint64_t>(update) * inner + offset];
                T& target_value = destination[
                    static_cast<std::uint64_t>(target) * inner + offset];
                if constexpr (Add) {
                    target_value = grouped_binary_value<T>(
                        BinKind::Add, target_value, value);
                } else {
                    target_value = value;
                }
            }
        }
    }
}

template <class T>
static __global__ void k_grouped_transpose(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t input,
    std::uint32_t output,
    GroupedRowShape input_shape) {
    const std::uint64_t numel =
        static_cast<std::uint64_t>(input_shape.max_rows) *
        input_shape.max_columns;
    const std::uint64_t total =
        static_cast<std::uint64_t>(header->lane_count) * numel;
    for (std::uint64_t flat =
             blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
         flat < total;
         flat += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const std::uint32_t lane = static_cast<std::uint32_t>(flat / numel);
        if (*grouped_commit(lanes, lane) == 0) continue;
        const std::uint64_t index = flat % numel;
        const std::uint32_t actual_rows =
            grouped_lane_rows(lanes[lane], input_shape);
        const std::uint32_t actual_columns =
            grouped_lane_columns(lanes[lane], input_shape);
        const std::uint64_t actual_numel =
            static_cast<std::uint64_t>(actual_rows) * actual_columns;
        if (index >= actual_numel) continue;
        const std::uint32_t row =
            static_cast<std::uint32_t>(index / actual_columns);
        const std::uint32_t column =
            static_cast<std::uint32_t>(index % actual_columns);
        grouped_value<T>(values, value_count, lane, output)[
            static_cast<std::uint64_t>(column) * actual_rows + row] =
            grouped_value<T>(values, value_count, lane, input)[index];
    }
}

static __global__ void k_grouped_mask_apply_packed(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t logits,
    std::uint32_t mask,
    std::uint32_t output,
    std::uint32_t rows,
    std::uint32_t len) {
    const std::uint64_t numel =
        static_cast<std::uint64_t>(rows) * len;
    const std::uint64_t total =
        static_cast<std::uint64_t>(header->lane_count) * numel;
    const std::uint32_t mask_words = (len + 31) / 32;
    for (std::uint64_t flat =
             blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
         flat < total;
         flat += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const std::uint32_t lane = static_cast<std::uint32_t>(flat / numel);
        if (*grouped_commit(lanes, lane) == 0) continue;
        const std::uint64_t index = flat % numel;
        const std::uint32_t row = static_cast<std::uint32_t>(index / len);
        if (row >= lanes[lane].sampled_rows) {
            grouped_value<float>(
                values, value_count, lane, output)[index] = 0.0f;
            continue;
        }
        const std::uint32_t column = static_cast<std::uint32_t>(index % len);
        const std::uint32_t word =
            grouped_value<std::uint32_t>(
                values, value_count, lane, mask)[
                static_cast<std::uint64_t>(row) * mask_words +
                (column >> 5)];
        grouped_value<float>(values, value_count, lane, output)[index] =
            ((word >> (column & 31)) & 1u)
                ? grouped_value<float>(
                    values, value_count, lane, logits)[index]
                : t0_neg_inf();
    }
}

static __global__ void k_grouped_direct_mask_apply_packed(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* mtp_rows,
    const std::uint32_t* mtp_offsets,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t mask,
    std::uint32_t output,
    std::uint32_t rows,
    std::uint32_t len,
    std::uint32_t vocab,
    std::uint8_t logits_kind) {
    const std::uint64_t numel =
        static_cast<std::uint64_t>(rows) * len;
    const std::uint64_t total =
        static_cast<std::uint64_t>(header->lane_count) * numel;
    const std::uint32_t mask_words = (len + 31) / 32;
    for (std::uint64_t flat =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         flat < total;
         flat += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const std::uint32_t lane =
            static_cast<std::uint32_t>(flat / numel);
        if (*grouped_commit(lanes, lane) == 0) continue;
        const std::uint64_t index = flat % numel;
        const std::uint32_t row =
            static_cast<std::uint32_t>(index / len);
        if (row >= lanes[lane].sampled_rows) continue;
        const std::uint32_t column =
            static_cast<std::uint32_t>(index % len);
        const std::uint32_t word =
            grouped_value<std::uint32_t>(
                values, value_count, lane, mask)[
                static_cast<std::uint64_t>(row) * mask_words +
                (column >> 5)];
        grouped_value<float>(
            values, value_count, lane, output)[index] =
            ((word >> (column & 31)) & 1u)
                ? grouped_direct_bf16_load(
                      lanes, mtp_rows, mtp_offsets, lane, index,
                      vocab, logits_kind)
                : t0_neg_inf();
    }
}

static __global__ void k_grouped_structured_position_mask(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t positions,
    std::uint32_t output,
    std::uint32_t position_count,
    std::uint32_t key_count,
    Tier0StructuredMaskKind kind,
    std::uint32_t window,
    std::uint32_t sink) {
    const std::uint64_t numel =
        static_cast<std::uint64_t>(position_count) * key_count;
    const std::uint64_t total =
        static_cast<std::uint64_t>(header->lane_count) * numel;
    for (std::uint64_t flat =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         flat < total;
         flat += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const std::uint32_t lane =
            static_cast<std::uint32_t>(flat / numel);
        if (*grouped_commit(lanes, lane) == 0) continue;
        const std::uint64_t index = flat % numel;
        const std::uint32_t position_index =
            static_cast<std::uint32_t>(index / key_count);
        const std::uint32_t key =
            static_cast<std::uint32_t>(index % key_count);
        grouped_value<std::uint8_t>(
            values, value_count, lane, output)[index] =
            structured_position_allows(
                grouped_value<std::uint32_t>(
                    values, value_count, lane, positions)[position_index],
                key,
                kind,
                window,
                sink);
    }
}

static __global__ void k_grouped_matmul(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t a,
    std::uint32_t b,
    std::uint32_t output,
    GroupedRowShape a_shape,
    GroupedRowShape b_shape) {
    const std::uint64_t output_numel =
        static_cast<std::uint64_t>(a_shape.max_rows) *
        b_shape.max_columns;
    const std::uint64_t total =
        static_cast<std::uint64_t>(header->lane_count) * output_numel;
    for (std::uint64_t flat =
             blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
         flat < total;
         flat += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const std::uint32_t lane =
            static_cast<std::uint32_t>(flat / output_numel);
        if (*grouped_commit(lanes, lane) == 0) continue;
        const std::uint64_t index = flat % output_numel;
        const std::uint32_t m =
            grouped_lane_rows(lanes[lane], a_shape);
        const std::uint32_t k =
            grouped_lane_columns(lanes[lane], a_shape);
        const std::uint32_t n =
            grouped_lane_columns(lanes[lane], b_shape);
        if (index >= static_cast<std::uint64_t>(m) * n) continue;
        const std::uint32_t row = static_cast<std::uint32_t>(index / n);
        const std::uint32_t column = static_cast<std::uint32_t>(index % n);
        const float* av = grouped_value<float>(
            values, value_count, lane, a);
        const float* bv = grouped_value<float>(
            values, value_count, lane, b);
        float sum = 0.0f;
        for (std::uint32_t inner = 0; inner < k; ++inner) {
            sum += av[static_cast<std::uint64_t>(row) * k + inner] *
                bv[static_cast<std::uint64_t>(inner) * n + column];
        }
        grouped_value<float>(
            values, value_count, lane, output)[index] = sum;
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

template <class Threshold>
static __global__ void k_grouped_pivot_rank(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t input,
    std::uint32_t threshold,
    std::uint32_t output,
    GroupedRowShape shape,
    std::uint32_t threshold_numel) {
    const std::uint32_t grouped_row = blockIdx.x;
    const std::uint32_t lane = grouped_row / shape.max_rows;
    const std::uint32_t row = grouped_row % shape.max_rows;
    const std::uint32_t rows = lane < header->lane_count
        ? grouped_lane_rows(lanes[lane], shape) : 0;
    const std::uint32_t len = lane < header->lane_count
        ? grouped_lane_columns(lanes[lane], shape) : 0;
    if (lane >= header->lane_count || row >= rows ||
        *grouped_commit(lanes, lane) == 0) {
        return;
    }
    const float* source =
        grouped_value<float>(values, value_count, lane, input) +
        static_cast<std::uint64_t>(row) * len;
    std::uint8_t* destination =
        grouped_value<std::uint8_t>(
            values, value_count, lane, output) +
        static_cast<std::uint64_t>(row) * len;
    const auto* thresholds = grouped_value<Threshold>(
        values, value_count, lane, threshold);
    const std::int64_t raw = static_cast<std::int64_t>(
        thresholds[threshold_numel <= 1 ? 0 : row]);
    const std::int64_t k =
        raw < 0 ? 0 : (raw > static_cast<std::int64_t>(len)
            ? static_cast<std::int64_t>(len)
            : raw);
    for (std::uint32_t index = threadIdx.x; index < len;
         index += blockDim.x) {
        const float value = source[index];
        if (isnan(value)) {
            destination[index] = 0;
            continue;
        }
        std::int64_t greater = 0;
        for (std::uint32_t other = 0; other < len; ++other) {
            const float candidate = source[other];
            if (!isnan(candidate) && candidate > value) ++greater;
        }
        destination[index] = greater < k;
    }
}

static __global__ void k_grouped_pivot_prob(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t input,
    std::uint32_t threshold,
    std::uint32_t output,
    GroupedRowShape shape,
    std::uint32_t threshold_numel) {
    const std::uint64_t numel =
        static_cast<std::uint64_t>(shape.max_rows) * shape.max_columns;
    const std::uint64_t total =
        static_cast<std::uint64_t>(header->lane_count) * numel;
    for (std::uint64_t flat =
             blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
         flat < total;
         flat += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const std::uint32_t lane =         static_cast<std::uint32_t>(flat / numel);
        if (*grouped_commit(lanes, lane) == 0) continue;
        const std::uint64_t index = flat % numel;
        const std::uint32_t rows =
        grouped_lane_rows(lanes[lane], shape);
        const std::uint32_t len =
        grouped_lane_columns(lanes[lane], shape);
        if (index >= static_cast<std::uint64_t>(rows) * len) continue;
        const std::uint32_t row = static_cast<std::uint32_t>(index / len);
        const float threshold_value = grouped_value<float>(
            values, value_count, lane, threshold)[
            threshold_numel <= 1 ? 0 : row];
        grouped_value<std::uint8_t>(
            values, value_count, lane, output)[index] =
            grouped_value<float>(
                values, value_count, lane, input)[index] >= threshold_value;
    }
}

static __global__ void k_grouped_pivot_cummass(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t input,
    std::uint32_t threshold,
    std::uint32_t output,
    GroupedRowShape shape,
    std::uint32_t threshold_numel) {
    __shared__ float shared_values[kTier0Block];
    __shared__ std::uint32_t shared_indices[kTier0Block];
    __shared__ std::uint8_t shared_nan[kTier0Block];
    __shared__ float previous_value;
    __shared__ std::uint32_t previous_index;
    __shared__ std::uint8_t previous_nan;
    __shared__ std::uint8_t previous_valid;
    __shared__ float exclusive_mass;
    __shared__ std::uint8_t stop;
    constexpr std::uint32_t none = UINT32_MAX;
    const std::uint32_t grouped_row = blockIdx.x;
    const std::uint32_t lane = grouped_row / shape.max_rows;
    const std::uint32_t row = grouped_row % shape.max_rows;
    const std::uint32_t rows = lane < header->lane_count
        ? grouped_lane_rows(lanes[lane], shape) : 0;
    const std::uint32_t len = lane < header->lane_count
        ? grouped_lane_columns(lanes[lane], shape) : 0;
    if (lane >= header->lane_count || row >= rows ||
        *grouped_commit(lanes, lane) == 0) {
        return;
    }
    const float* source =
        grouped_value<float>(values, value_count, lane, input) +
        static_cast<std::uint64_t>(row) * len;
    std::uint8_t* destination =
        grouped_value<std::uint8_t>(
            values, value_count, lane, output) +
        static_cast<std::uint64_t>(row) * len;
    const float p = grouped_value<float>(
        values, value_count, lane, threshold)[
        threshold_numel <= 1 ? 0 : row];
    for (std::uint32_t index = threadIdx.x; index < len;
         index += blockDim.x) {
        destination[index] = 0;
    }
    if (threadIdx.x == 0) {
        previous_value = t0_pos_inf();
        previous_index = 0;
        previous_nan = 0;
        previous_valid = 0;
        exclusive_mass = 0.0f;
        stop = 0;
    }
    __syncthreads();
    for (std::uint32_t pick = 0; pick < len; ++pick) {
        if (stop) break;
        const float prior_value = previous_value;
        const std::uint32_t prior_index = previous_index;
        const bool prior_nan = previous_nan != 0;
        const bool prior_valid = previous_valid != 0;
        float best = 0.0f;
        std::uint32_t best_index = none;
        bool best_nan = false;
        for (std::uint32_t index = threadIdx.x; index < len;
             index += blockDim.x) {
            const float value = source[index];
            const bool value_nan = isnan(value);
            if (prior_valid &&
                !t0_desc_before(
                    prior_value, prior_index, prior_nan,
                    value, index, value_nan)) {
                continue;
            }
            if (best_index == none ||
                t0_desc_before(
                    value, index, value_nan,
                    best, best_index, best_nan)) {
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
                const std::uint32_t other =
                    shared_indices[threadIdx.x + stride];
                if (other != none &&
                    (shared_indices[threadIdx.x] == none ||
                     t0_desc_before(
                         shared_values[threadIdx.x + stride], other,
                         shared_nan[threadIdx.x + stride] != 0,
                         shared_values[threadIdx.x],
                         shared_indices[threadIdx.x],
                         shared_nan[threadIdx.x] != 0))) {
                    shared_values[threadIdx.x] =
                        shared_values[threadIdx.x + stride];
                    shared_indices[threadIdx.x] = other;
                    shared_nan[threadIdx.x] =
                        shared_nan[threadIdx.x + stride];
                }
            }
            __syncthreads();
        }
        if (threadIdx.x == 0) {
            const std::uint32_t selected = shared_indices[0];
            if (selected == none) {
                stop = 1;
            } else if (exclusive_mass < p) {
                destination[selected] = 1;
                exclusive_mass += source[selected];
                previous_value = shared_values[0];
                previous_index = selected;
                previous_nan = shared_nan[0];
                previous_valid = 1;
            } else {
                stop = 1;
            }
        }
        __syncthreads();
    }
}

static __global__ void k_grouped_rng_keyed(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t state,
    std::uint32_t output,
    std::uint64_t numel,
    bool gumbel) {
    const std::uint64_t total =
        static_cast<std::uint64_t>(header->lane_count) * numel;
    for (std::uint64_t flat =
             blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
         flat < total;
         flat += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const std::uint32_t lane = static_cast<std::uint32_t>(flat / numel);
        if (*grouped_commit(lanes, lane) == 0) continue;
        const std::uint64_t index = flat % numel;
        const std::uint32_t* state_value =
            grouped_value<std::uint32_t>(values, value_count, lane, state);
        const unsigned long long seed64 =
            ptir_rng_keyed_seed(state_value[0], state_value[1]);
        const float uniform =
            t0_hash_uniform(seed64, static_cast<int>(index));
        grouped_value<float>(values, value_count, lane, output)[index] =
            gumbel ? -logf(-logf(uniform)) : uniform;
    }
}

static __global__ void k_grouped_rng_ambient(
        const PtirLaneTableHeader* header,
        const PtirLaneRecord* lanes,
        const std::uint64_t* values,
        std::uint32_t value_count,
        std::uint32_t output,
        std::uint32_t rows,
        std::uint32_t len,
        std::uint32_t stream,
        bool gumbel) {
        const std::uint64_t numel =
            static_cast<std::uint64_t>(rows) * len;
        const std::uint64_t total =
            static_cast<std::uint64_t>(header->lane_count) * numel;
        for (std::uint64_t flat =
                 blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
             flat < total;
             flat += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
            const std::uint32_t lane = static_cast<std::uint32_t>(flat / numel);
            if (*grouped_commit(lanes, lane) == 0) continue;
            const std::uint64_t index = flat % numel;
            const std::uint32_t row = static_cast<std::uint32_t>(index / len);
            if (row >= lanes[lane].sampled_rows) {
                grouped_value<float>(
                    values, value_count, lane, output)[index] = 0.0f;
                continue;
            }
            const auto* seeds =
                reinterpret_cast<const std::uint32_t*>(lanes[lane].rng_state);
            const unsigned long long effective =
                t0_seed_eff_stream(seeds[row], stream);
            const float uniform =
                t0_hash_uniform(effective, static_cast<int>(index % len));
            grouped_value<float>(values, value_count, lane, output)[index] =
                gumbel ? -logf(-logf(uniform)) : uniform;
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

static __global__ void k_grouped_nucleus_probabilities(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* mtp_rows,
    const std::uint32_t* mtp_offsets,
    const std::uint64_t* values,
    std::uint32_t value_count,
    float* probabilities,
    GroupedNucleusLaunch launch) {
    __shared__ float levels[kCanonicalReduceLevels][kCanonicalReduceWidth];
    __shared__ float tile[kCanonicalReduceWidth];
    __shared__ std::uint32_t counts[kCanonicalReduceLevels];
    __shared__ float maximum;
    __shared__ float sum;
    const std::uint32_t segment = blockIdx.x;
    const std::uint32_t lane = segment / launch.rows;
    const std::uint32_t row = segment % launch.rows;
    float* destination =
        probabilities + static_cast<std::uint64_t>(segment) * launch.len;
    if (lane >= header->lane_count ||
        row >= lanes[lane].sampled_rows ||
        *grouped_commit(lanes, lane) == 0) {
        for (std::uint32_t index = threadIdx.x; index < launch.len;
             index += blockDim.x) {
            destination[index] = 0.0f;
        }
        return;
    }
    const float* logits = launch.logits_kind == 0
        ? grouped_value<float>(
              values, value_count, lane, launch.logits) +
              static_cast<std::uint64_t>(row) * launch.len
        : nullptr;
    grouped_nucleus_reduce(
        lanes, mtp_rows, mtp_offsets, lane, row, logits, launch.len,
        launch.logits_kind, RedKind::Max, false, 0.0f,
        levels, tile, counts, &maximum);
    grouped_nucleus_reduce(
        lanes, mtp_rows, mtp_offsets, lane, row, logits, launch.len,
        launch.logits_kind, RedKind::Sum, true, maximum,
        levels, tile, counts, &sum);
    for (std::uint32_t index = threadIdx.x; index < launch.len;
         index += blockDim.x) {
        const float logit = launch.logits_kind == 0
            ? logits[index]
            : grouped_direct_bf16_load(
                  lanes, mtp_rows, mtp_offsets, lane,
                  static_cast<std::uint64_t>(row) * launch.len + index,
                  launch.len, launch.logits_kind);
        destination[index] = expf(logit - maximum) / sum;
    }
}

template <class T>
__device__ __forceinline__ void grouped_canonical_reduce(
    const T* source,
    std::uint32_t len,
    RedKind kind,
    T levels[kCanonicalReduceLevels][kCanonicalReduceWidth],
    T tile[kCanonicalReduceWidth],
    std::uint32_t counts[kCanonicalReduceLevels],
    T* result) {
    const std::uint32_t lane = threadIdx.x;
    if (lane < kCanonicalReduceLevels) counts[lane] = 0;
    __syncwarp();
    for (std::uint32_t base = 0; base < len;
         base += kCanonicalReduceWidth) {
        const std::uint32_t index = base + lane;
        tile[lane] =
            index < len ? source[index] : red_identity<T>(kind);
        const std::uint32_t tile_count =
            len - base < kCanonicalReduceWidth
                ? len - base
                : kCanonicalReduceWidth;
        const T partial =
            reduce_canonical_slot(tile, lane, tile_count, kind);
        if (lane == 0) levels[0][counts[0]++] = partial;
        __syncwarp();
        for (std::uint32_t level = 0;
             level + 1 < kCanonicalReduceLevels;
             ++level) {
            if (counts[level] != kCanonicalReduceWidth) break;
            const T carry = reduce_canonical_slot(
                levels[level], lane, kCanonicalReduceWidth, kind);
            if (lane == 0) {
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
            if (lane == 0) *result = levels[level][0];
            __syncwarp();
            return;
        }
        const T carry =
            reduce_canonical_slot(levels[level], lane, count, kind);
        if (lane == 0) {
            counts[level] = 0;
            levels[level + 1][counts[level + 1]++] = carry;
        }
        __syncwarp();
    }
    if (lane == 0) {
        *result = counts[kCanonicalReduceLevels - 1] == 0
            ? red_identity<T>(kind)
            : levels[kCanonicalReduceLevels - 1][0];
    }
    __syncwarp();
}

template <class T>
static __global__ void k_grouped_reduce_chunks(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t input,
    T* partials,
    std::uint32_t rows,
    std::uint32_t len,
    std::uint32_t chunks,
    std::uint32_t chunk_len,
    RedKind kind) {
    __shared__ T levels[kCanonicalReduceLevels][kCanonicalReduceWidth];
    __shared__ T tile[kCanonicalReduceWidth];
    __shared__ std::uint32_t counts[kCanonicalReduceLevels];
    const std::uint32_t grouped_chunk = blockIdx.x;
    const std::uint32_t chunks_per_lane = rows * chunks;
    const std::uint32_t lane = grouped_chunk / chunks_per_lane;
    const std::uint32_t within_lane = grouped_chunk % chunks_per_lane;
    const std::uint32_t row = within_lane / chunks;
    const std::uint32_t chunk = within_lane % chunks;
    if (lane >= header->lane_count || *grouped_commit(lanes, lane) == 0) {
        return;
    }
    const T* source =
        grouped_value<T>(values, value_count, lane, input) +
        static_cast<std::uint64_t>(row) * len +
        static_cast<std::uint64_t>(chunk) * chunk_len;
    grouped_canonical_reduce(
        source,
        chunk_len,
        kind,
        levels,
        tile,
        counts,
        partials + static_cast<std::uint64_t>(grouped_chunk));
}

template <class T>
static __global__ void k_grouped_reduce_chunk_results(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* values,
    std::uint32_t value_count,
    const T* partials,
    std::uint32_t output,
    std::uint32_t rows,
    std::uint32_t chunks,
    RedKind kind) {
    __shared__ T levels[kCanonicalReduceLevels][kCanonicalReduceWidth];
    __shared__ T tile[kCanonicalReduceWidth];
    __shared__ std::uint32_t counts[kCanonicalReduceLevels];
    const std::uint32_t grouped_row = blockIdx.x;
    const std::uint32_t lane = grouped_row / rows;
    const std::uint32_t row = grouped_row % rows;
    if (lane >= header->lane_count || *grouped_commit(lanes, lane) == 0) {
        return;
    }
    T* destination =
        grouped_value<T>(values, value_count, lane, output) + row;
    grouped_canonical_reduce(
        partials + static_cast<std::uint64_t>(grouped_row) * chunks,
        chunks,
        kind,
        levels,
        tile,
        counts,
        destination);
}

static __global__ void k_grouped_nucleus_sort_keys(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const float* probabilities,
    std::uint64_t* keys,
    std::uint32_t* indices,
    std::uint32_t rows,
    std::uint32_t vocab) {
    const std::uint64_t segment_size = vocab;
    const std::uint64_t segments =
        static_cast<std::uint64_t>(header->lane_count) * rows;
    const std::uint64_t total = segments * segment_size;
    for (std::uint64_t flat =
             static_cast<std::uint64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
         flat < total;
         flat += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const std::uint32_t column =
            static_cast<std::uint32_t>(flat % segment_size);
        const float probability = probabilities[flat];
        std::uint32_t bits = __float_as_uint(probability);
        // The reference comparator treats signed zero as an ordinary tie and
        // places every NaN after every non-NaN. Canonicalize zero, then map the
        // IEEE numeric order to an ascending radix key and invert it for the
        // required descending order. The low word makes ties stable by source
        // index, independently of the radix sort implementation.
        if (probability == 0.0f) bits = 0;
        const std::uint32_t ascending =
            bits ^ ((bits & 0x80000000u) != 0
                        ? 0xffffffffu
                        : 0x80000000u);
        const std::uint32_t order =
            isnan(probability) ? UINT32_MAX : ~ascending;
        keys[flat] =
            (static_cast<std::uint64_t>(order) << 32) | column;
        indices[flat] = column;
    }
}

static __global__ void k_grouped_nucleus_sorted_finish(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const PtirLaneChannelSlot* channels,
    const std::uint64_t* mtp_rows,
    const std::uint32_t* mtp_offsets,
    const std::uint64_t* values,
    std::uint32_t value_count,
    const float* probabilities,
    const std::uint32_t* sorted_indices,
    GroupedNucleusLaunch launch) {
    const std::uint32_t segment =
        blockIdx.x * blockDim.x + threadIdx.x;
    const std::uint32_t segment_count =
        header->lane_count * launch.rows;
    if (segment >= segment_count) return;
    const std::uint32_t lane = segment / launch.rows;
    const std::uint32_t row = segment % launch.rows;
    if (*grouped_commit(lanes, lane) == 0 ||
        row >= lanes[lane].sampled_rows) {
        return;
    }
    const auto* logits = launch.logits_kind == 0
        ? grouped_value<float>(
              values, value_count, lane, launch.logits) +
              static_cast<std::uint64_t>(row) * launch.len
        : nullptr;
    probabilities += static_cast<std::uint64_t>(segment) * launch.len;
    const float p = grouped_value<float>(
        values, value_count, lane, launch.top_p)[
        launch.top_p_numel <= 1 ? 0 : row];
    const auto* state = grouped_value<std::uint32_t>(
        values, value_count, lane, launch.rng_state);
    const unsigned long long seed64 =
        ptir_rng_keyed_seed(state[0], state[1]);
    float exclusive = 0.0f;
    float best = t0_neg_inf();
    std::uint32_t token = 0;
    std::uint32_t first_excluded = UINT32_MAX;
    bool have = false;
    const auto sorted_base =
        static_cast<std::uint64_t>(segment) * launch.len;
    for (std::uint32_t rank = 0; rank < launch.len; ++rank) {
        const std::uint32_t index =
            sorted_indices[sorted_base + rank];
        if (!(exclusive < p)) {
            first_excluded =
                index < first_excluded ? index : first_excluded;
            continue;
        }
        exclusive += probabilities[index];
        const float uniform = t0_hash_uniform(
            seed64,
            static_cast<int>(
                static_cast<std::uint64_t>(row) * launch.len + index));
        const float logit = launch.logits_kind == 0
            ? logits[index]
            : grouped_direct_bf16_load(
                  lanes, mtp_rows, mtp_offsets, lane,
                  static_cast<std::uint64_t>(row) * launch.len + index,
                  launch.len, launch.logits_kind);
        const float score = logit - logf(-logf(uniform));
        if (!isnan(score) &&
            (!have || score > best ||
             (score == best && index < token))) {
            best = score;
            token = index;
            have = true;
        }
    }
    if (first_excluded != UINT32_MAX &&
        (!have ||
         (best == t0_neg_inf() && first_excluded < token))) {
        token = first_excluded;
        best = t0_neg_inf();
        have = true;
    }
    if (!have) token = 0;
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
    grouped_nucleus_reduce(
        lanes, mtp_rows, mtp_offsets, lane, row, logits, launch.len,
        launch.logits_kind, RedKind::Max, false, 0.0f,
        levels, tile, counts, &maximum);
    grouped_nucleus_reduce(
        lanes, mtp_rows, mtp_offsets, lane, row, logits, launch.len,
        launch.logits_kind, RedKind::Sum, true, maximum,
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
            const float logit = launch.logits_kind == 0
                ? logits[index]
                : grouped_direct_bf16_load(
                      lanes, mtp_rows, mtp_offsets, lane,
                      static_cast<std::uint64_t>(row) * launch.len + index,
                      launch.len, launch.logits_kind);
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
        const float logit = launch.logits_kind == 0
            ? logits[index]
            : grouped_direct_bf16_load(
                  lanes, mtp_rows, mtp_offsets, lane,
                  static_cast<std::uint64_t>(row) * launch.len + index,
                  launch.len, launch.logits_kind);
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

static __global__ void k_grouped_reduce_argmax_direct_bf16(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* mtp_rows,
    const std::uint32_t* mtp_offsets,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t output,
    GroupedRowShape shape,
    std::uint32_t vocab,
    std::uint8_t input_kind) {
    __shared__ float shared_values[kTier0Block];
    __shared__ std::uint32_t shared_indices[kTier0Block];
    __shared__ std::uint8_t shared_have[kTier0Block];
    const std::uint32_t lane = blockIdx.x / shape.max_rows;
    const std::uint32_t row = blockIdx.x % shape.max_rows;
    const std::uint32_t rows = lane < header->lane_count
        ? grouped_lane_rows(lanes[lane], shape)
        : 0;
    const std::uint32_t len = lane < header->lane_count
        ? grouped_lane_columns(lanes[lane], shape)
        : 0;
    if (lane >= header->lane_count ||
        row >= rows ||
        *grouped_commit(lanes, lane) == 0) {
        return;
    }
    float best = 0.0f;
    std::uint32_t best_index = 0;
    bool have = false;
    for (std::uint32_t index = threadIdx.x;
         index < len;
         index += blockDim.x) {
        const float value = grouped_direct_bf16_load(
            lanes, mtp_rows, mtp_offsets, lane,
            static_cast<std::uint64_t>(row) * len + index,
            vocab, input_kind);
        if (!isnan(value) &&
            (!have || value > best ||
             (value == best && index < best_index))) {
            best = value;
            best_index = index;
            have = true;
        }
    }
    shared_values[threadIdx.x] = best;
    shared_indices[threadIdx.x] = best_index;
    shared_have[threadIdx.x] = have;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x >> 1;
         stride > 0;
         stride >>= 1) {
        if (threadIdx.x < stride) {
            const bool other_have =
                shared_have[threadIdx.x + stride] != 0;
            const float other_value =
                shared_values[threadIdx.x + stride];
            const std::uint32_t other_index =
                shared_indices[threadIdx.x + stride];
            if (other_have &&
                (shared_have[threadIdx.x] == 0 ||
                 other_value > shared_values[threadIdx.x] ||
                 (other_value == shared_values[threadIdx.x] &&
                  other_index < shared_indices[threadIdx.x]))) {
                shared_values[threadIdx.x] = other_value;
                shared_indices[threadIdx.x] = other_index;
                shared_have[threadIdx.x] = 1;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        grouped_value<std::uint32_t>(
            values, value_count, lane, output)[row] =
            shared_have[0] ? shared_indices[0] : 0;
    }
}

template <class T>
static __global__ void k_grouped_reduce_argmax(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t input,
    std::uint32_t output,
    std::uint32_t rows,
    std::uint32_t len) {
    __shared__ T shared_values[kTier0Block];
    __shared__ std::uint32_t shared_indices[kTier0Block];
    __shared__ std::uint8_t shared_have[kTier0Block];
    const std::uint32_t lane = blockIdx.x / rows;
    const std::uint32_t row = blockIdx.x % rows;
    if (lane >= header->lane_count || *grouped_commit(lanes, lane) == 0) {
        return;
    }
    const T* source =
        grouped_value<T>(values, value_count, lane, input) +
        static_cast<std::uint64_t>(row) * len;
    T best{};
    std::uint32_t best_index = 0;
    bool have = false;
    for (std::uint32_t index = threadIdx.x; index < len; index += blockDim.x) {
        const T value = source[index];
        bool selectable = true;
        if constexpr (std::is_same_v<T, float>) {
            selectable = !isnan(value);
        }
        if (selectable &&
            (!have || value > best ||
             (value == best && index < best_index))) {
            best = value;
            best_index = index;
            have = true;
        }
    }
    shared_values[threadIdx.x] = best;
    shared_indices[threadIdx.x] = best_index;
    shared_have[threadIdx.x] = have;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            const T other_value = shared_values[threadIdx.x + stride];
            const std::uint32_t other_index =
                shared_indices[threadIdx.x + stride];
            const bool other_have = shared_have[threadIdx.x + stride] != 0;
            const bool self_have = shared_have[threadIdx.x] != 0;
            if (other_have &&
                (!self_have || other_value > shared_values[threadIdx.x] ||
                 (other_value == shared_values[threadIdx.x] &&
                  other_index < shared_indices[threadIdx.x]))) {
                shared_values[threadIdx.x] = other_value;
                shared_indices[threadIdx.x] = other_index;
                shared_have[threadIdx.x] = 1;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        grouped_value<std::uint32_t>(
            values, value_count, lane, output)[row] =
            shared_have[0] ? shared_indices[0] : 0;
    }
}

template <class T>
static __global__ void k_grouped_reduce_argmax_ragged(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t input,
    std::uint32_t output,
    GroupedRowShape shape) {
    __shared__ T shared_values[kTier0Block];
    __shared__ std::uint32_t shared_indices[kTier0Block];
    __shared__ std::uint8_t shared_have[kTier0Block];
    const std::uint32_t lane = blockIdx.x / shape.max_rows;
    const std::uint32_t row = blockIdx.x % shape.max_rows;
    const std::uint32_t rows = lane < header->lane_count
        ? grouped_lane_rows(lanes[lane], shape)
        : 0;
    const std::uint32_t len = lane < header->lane_count
        ? grouped_lane_columns(lanes[lane], shape)
        : 0;
    if (lane >= header->lane_count || row >= rows ||
        *grouped_commit(lanes, lane) == 0) {
        return;
    }
    const T* source =
        grouped_value<T>(values, value_count, lane, input) +
        static_cast<std::uint64_t>(row) * len;
    T best{};
    std::uint32_t best_index = 0;
    bool have = false;
    for (std::uint32_t index = threadIdx.x;
         index < len;
         index += blockDim.x) {
        const T value = source[index];
        bool selectable = true;
        if constexpr (std::is_same_v<T, float>) {
            selectable = !isnan(value);
        }
        if (selectable &&
            (!have || value > best ||
             (value == best && index < best_index))) {
            best = value;
            best_index = index;
            have = true;
        }
    }
    shared_values[threadIdx.x] = best;
    shared_indices[threadIdx.x] = best_index;
    shared_have[threadIdx.x] = have;
    __syncthreads();
    for (std::uint32_t stride = blockDim.x >> 1;
         stride > 0;
         stride >>= 1) {
        if (threadIdx.x < stride) {
            const bool other_have =
                shared_have[threadIdx.x + stride] != 0;
            const T other_value =
                shared_values[threadIdx.x + stride];
            const std::uint32_t other_index =
                shared_indices[threadIdx.x + stride];
            if (other_have &&
                (shared_have[threadIdx.x] == 0 ||
                 other_value > shared_values[threadIdx.x] ||
                 (other_value == shared_values[threadIdx.x] &&
                  other_index < shared_indices[threadIdx.x]))) {
                shared_values[threadIdx.x] = other_value;
                shared_indices[threadIdx.x] = other_index;
                shared_have[threadIdx.x] = 1;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        grouped_value<std::uint32_t>(
            values, value_count, lane, output)[row] =
            shared_have[0] ? shared_indices[0] : 0;
    }
}

static __global__ void k_grouped_argmax_chunks(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t input,
    float* partial_values,
    std::uint32_t* partial_indices,
    std::uint32_t rows,
    std::uint32_t len,
    std::uint32_t chunks,
    std::uint32_t chunk_len) {
    __shared__ float shared_values[kTier0Block];
    __shared__ std::uint32_t shared_indices[kTier0Block];
    const std::uint32_t grouped_chunk = blockIdx.x;
    const std::uint32_t chunks_per_lane = rows * chunks;
    const std::uint32_t lane = grouped_chunk / chunks_per_lane;
    const std::uint32_t within_lane = grouped_chunk % chunks_per_lane;
    const std::uint32_t row = within_lane / chunks;
    const std::uint32_t chunk = within_lane % chunks;
    if (lane >= header->lane_count || *grouped_commit(lanes, lane) == 0) {
        return;
    }
    const float* source =
        grouped_value<float>(values, value_count, lane, input) +
        static_cast<std::uint64_t>(row) * len;
    const std::uint32_t begin = chunk * chunk_len;
    const std::uint32_t end =
        begin + chunk_len < len ? begin + chunk_len : len;
    float best = t0_neg_inf();
    std::uint32_t best_index = UINT32_MAX;
    for (std::uint32_t index = begin + threadIdx.x;
         index < end;
         index += blockDim.x) {
        const float value = source[index];
        if (!isnan(value) &&
            (best_index == UINT32_MAX || value > best ||
             (value == best && index < best_index))) {
            best = value;
            best_index = index;
        }
    }
    shared_values[threadIdx.x] = best;
    shared_indices[threadIdx.x] = best_index;
    __syncthreads();
    for (std::uint32_t offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            const auto other_index =
                shared_indices[threadIdx.x + offset];
            const auto other_value =
                shared_values[threadIdx.x + offset];
            if (other_index != UINT32_MAX &&
                (shared_indices[threadIdx.x] == UINT32_MAX ||
                 other_value > shared_values[threadIdx.x] ||
                 (other_value == shared_values[threadIdx.x] &&
                  other_index < shared_indices[threadIdx.x]))) {
                shared_values[threadIdx.x] = other_value;
                shared_indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        partial_values[grouped_chunk] = shared_values[0];
        partial_indices[grouped_chunk] = shared_indices[0];
    }
}

static __global__ void k_grouped_argmax_chunk_results(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* values,
    std::uint32_t value_count,
    const float* partial_values,
    const std::uint32_t* partial_indices,
    std::uint32_t output,
    std::uint32_t rows,
    std::uint32_t chunks) {
    __shared__ float shared_values[kTier0Block];
    __shared__ std::uint32_t shared_indices[kTier0Block];
    const std::uint32_t grouped_row = blockIdx.x;
    const std::uint32_t lane = grouped_row / rows;
    const std::uint32_t row = grouped_row % rows;
    if (lane >= header->lane_count || *grouped_commit(lanes, lane) == 0) {
        return;
    }
    float best = t0_neg_inf();
    std::uint32_t best_index = UINT32_MAX;
    const auto base = static_cast<std::uint64_t>(grouped_row) * chunks;
    for (std::uint32_t chunk = threadIdx.x; chunk < chunks;
         chunk += blockDim.x) {
        const auto index = partial_indices[base + chunk];
        const auto value = partial_values[base + chunk];
        if (index != UINT32_MAX &&
            (best_index == UINT32_MAX || value > best ||
             (value == best && index < best_index))) {
            best = value;
            best_index = index;
        }
    }
    shared_values[threadIdx.x] = best;
    shared_indices[threadIdx.x] = best_index;
    __syncthreads();
    for (std::uint32_t offset = blockDim.x >> 1; offset > 0; offset >>= 1) {
        if (threadIdx.x < offset) {
            const auto other_index =
                shared_indices[threadIdx.x + offset];
            const auto other_value =
                shared_values[threadIdx.x + offset];
            if (other_index != UINT32_MAX &&
                (shared_indices[threadIdx.x] == UINT32_MAX ||
                 other_value > shared_values[threadIdx.x] ||
                 (other_value == shared_values[threadIdx.x] &&
                  other_index < shared_indices[threadIdx.x]))) {
                shared_values[threadIdx.x] = other_value;
                shared_indices[threadIdx.x] = other_index;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) {
        grouped_value<std::uint32_t>(
            values, value_count, lane, output)[row] =
            shared_indices[0] == UINT32_MAX ? 0 : shared_indices[0];
    }
}

static __global__ void k_grouped_reduce_direct_bf16(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* mtp_rows,
    const std::uint32_t* mtp_offsets,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t output,
    GroupedRowShape shape,
    std::uint32_t vocab,
    std::uint8_t input_kind,
    RedKind kind) {
    __shared__ float levels[kCanonicalReduceLevels][kCanonicalReduceWidth];
    __shared__ float tile[kCanonicalReduceWidth];
    __shared__ std::uint32_t counts[kCanonicalReduceLevels];
    const std::uint32_t grouped_row = blockIdx.x;
    const std::uint32_t lane = grouped_row / shape.max_rows;
    const std::uint32_t row = grouped_row % shape.max_rows;
    const std::uint32_t rows = lane < header->lane_count
        ? grouped_lane_rows(lanes[lane], shape)
        : 0;
    const std::uint32_t len = lane < header->lane_count
        ? grouped_lane_columns(lanes[lane], shape)
        : 0;
    if (lane >= header->lane_count ||
        row >= rows ||
        *grouped_commit(lanes, lane) == 0) {
        return;
    }
    const std::uint32_t thread_lane = threadIdx.x;
    if (thread_lane < kCanonicalReduceLevels) counts[thread_lane] = 0;
    __syncwarp();
    for (std::uint32_t base = 0; base < len;
         base += kCanonicalReduceWidth) {
        const std::uint32_t index = base + thread_lane;
        tile[thread_lane] = index < len
            ? grouped_direct_bf16_load(
                  lanes, mtp_rows, mtp_offsets, lane,
                  static_cast<std::uint64_t>(row) * len + index,
                  vocab, input_kind)
            : red_identity<float>(kind);
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
                levels[level], thread_lane,
                kCanonicalReduceWidth, kind);
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
            if (thread_lane == 0) {
                grouped_value<float>(
                    values, value_count, lane, output)[row] =
                    levels[level][0];
            }
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
        grouped_value<float>(
            values, value_count, lane, output)[row] =
            counts[kCanonicalReduceLevels - 1] == 0
                ? red_identity<float>(kind)
                : levels[kCanonicalReduceLevels - 1][0];
    }
}

template <class T>
static __global__ void k_grouped_reduce(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t input,
    std::uint32_t output,
    std::uint32_t rows,
    std::uint32_t len,
    RedKind kind) {
    __shared__ T levels[kCanonicalReduceLevels][kCanonicalReduceWidth];
    __shared__ T tile[kCanonicalReduceWidth];
    __shared__ std::uint32_t counts[kCanonicalReduceLevels];
    const std::uint32_t grouped_row = blockIdx.x;
    const std::uint32_t lane = grouped_row / rows;
    const std::uint32_t row = grouped_row % rows;
    if (lane >= header->lane_count || *grouped_commit(lanes, lane) == 0) {
        return;
    }
    const std::uint32_t thread_lane = threadIdx.x;
    const T* source = grouped_value<T>(values, value_count, lane, input) +
        static_cast<std::uint64_t>(row) * len;
    if (thread_lane < kCanonicalReduceLevels) counts[thread_lane] = 0;
    __syncwarp();
    for (std::uint32_t base = 0; base < len;
         base += kCanonicalReduceWidth) {
        const std::uint32_t index = base + thread_lane;
        tile[thread_lane] =
            index < len ? source[index] : red_identity<T>(kind);
        const std::uint32_t tile_count =
            len - base < kCanonicalReduceWidth
                ? len - base
                : kCanonicalReduceWidth;
        const T partial =
            reduce_canonical_slot(tile, thread_lane, tile_count, kind);
        if (thread_lane == 0) levels[0][counts[0]++] = partial;
        __syncwarp();
        for (std::uint32_t level = 0;
             level + 1 < kCanonicalReduceLevels;
             ++level) {
            if (counts[level] != kCanonicalReduceWidth) break;
            const T carry = reduce_canonical_slot(
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
            if (thread_lane == 0) {
                grouped_value<T>(
                    values, value_count, lane, output)[row] =
                    levels[level][0];
            }
            return;
        }
        const T carry =
            reduce_canonical_slot(levels[level], thread_lane, count, kind);
        if (thread_lane == 0) {
            counts[level] = 0;
            levels[level + 1][counts[level + 1]++] = carry;
        }
        __syncwarp();
    }
    if (thread_lane == 0) {
        grouped_value<T>(values, value_count, lane, output)[row] =
            counts[kCanonicalReduceLevels - 1] == 0
                ? red_identity<T>(kind)
                : levels[kCanonicalReduceLevels - 1][0];
    }
}

template <class T>
static __global__ void k_grouped_reduce_ragged(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t input,
    std::uint32_t output,
    GroupedRowShape shape,
    RedKind kind) {
    __shared__ T levels[kCanonicalReduceLevels][kCanonicalReduceWidth];
    __shared__ T tile[kCanonicalReduceWidth];
    __shared__ std::uint32_t counts[kCanonicalReduceLevels];
    const std::uint32_t grouped_row = blockIdx.x;
    const std::uint32_t lane = grouped_row / shape.max_rows;
    const std::uint32_t row = grouped_row % shape.max_rows;
    const std::uint32_t rows = lane < header->lane_count
        ? grouped_lane_rows(lanes[lane], shape)
        : 0;
    const std::uint32_t len = lane < header->lane_count
        ? grouped_lane_columns(lanes[lane], shape)
        : 0;
    if (lane >= header->lane_count || row >= rows ||
        *grouped_commit(lanes, lane) == 0) {
        return;
    }
    const T* source =
        grouped_value<T>(values, value_count, lane, input) +
        static_cast<std::uint64_t>(row) * len;
    grouped_canonical_reduce(
        source, len, kind, levels, tile, counts,
        grouped_value<T>(values, value_count, lane, output) + row);
}

template <class T>
static __global__ void k_grouped_scan(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t input,
    std::uint32_t output,
    GroupedRowShape shape,
    ScanKind kind) {
    __shared__ T shared[kTier0Block];
    const std::uint32_t grouped_row = blockIdx.x;
    const std::uint32_t lane = grouped_row / shape.max_rows;
    const std::uint32_t row = grouped_row % shape.max_rows;
    const std::uint32_t rows = lane < header->lane_count
        ? grouped_lane_rows(lanes[lane], shape)
        : 0;
    const std::uint32_t len = lane < header->lane_count
        ? grouped_lane_columns(lanes[lane], shape)
        : 0;
    if (lane >= header->lane_count || row >= rows ||
        *grouped_commit(lanes, lane) == 0) {
        return;
    }
    const T* source =
        grouped_value<T>(values, value_count, lane, input) +
        static_cast<std::uint64_t>(row) * len;
    T* destination =
        grouped_value<T>(values, value_count, lane, output) +
        static_cast<std::uint64_t>(row) * len;
    T carry = kind == ScanKind::Sum ? static_cast<T>(0) : static_cast<T>(1);
    for (std::uint32_t base = 0; base < len; base += blockDim.x) {
        const std::uint32_t index = base + threadIdx.x;
        const T identity =
            kind == ScanKind::Sum ? static_cast<T>(0) : static_cast<T>(1);
        shared[threadIdx.x] = index < len ? source[index] : identity;
        __syncthreads();
        for (std::uint32_t offset = 1; offset < blockDim.x; offset <<= 1) {
            const T prior =
                threadIdx.x >= offset ? shared[threadIdx.x - offset] : identity;
            __syncthreads();
            shared[threadIdx.x] =
                scan_combine(shared[threadIdx.x], prior, kind);
            __syncthreads();
        }
        if (index < len) {
            destination[index] =
                scan_combine(carry, shared[threadIdx.x], kind);
        }
        const T tile_total = shared[blockDim.x - 1];
        __syncthreads();
        carry = scan_combine(carry, tile_total, kind);
    }
}

template <class T>
static __global__ void k_grouped_broadcast(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const std::uint64_t* values,
    std::uint32_t value_count,
    std::uint32_t input,
    std::uint32_t output,
    std::uint64_t numel,
    GroupedBroadcastMeta meta) {
    const std::uint64_t total =
        static_cast<std::uint64_t>(header->lane_count) * numel;
    for (std::uint64_t flat =
             blockIdx.x * static_cast<std::uint64_t>(blockDim.x) + threadIdx.x;
         flat < total;
         flat += static_cast<std::uint64_t>(gridDim.x) * blockDim.x) {
        const std::uint32_t lane = static_cast<std::uint32_t>(flat / numel);
        if (*grouped_commit(lanes, lane) == 0) continue;
        const std::uint64_t output_index = flat % numel;
        std::uint64_t remaining = output_index;
        std::uint64_t source_index = 0;
        for (int dimension = static_cast<int>(meta.rank) - 1;
             dimension >= 0;
             --dimension) {
            const std::uint32_t size = meta.out_dims[dimension];
            const std::uint32_t coordinate =
                size == 0 ? 0 : static_cast<std::uint32_t>(remaining % size);
            if (size != 0) remaining /= size;
            source_index +=
                static_cast<std::uint64_t>(coordinate) *
                meta.src_strides[dimension];
        }
        grouped_value<T>(values, value_count, lane, output)[output_index] =
            grouped_value<T>(
                values, value_count, lane, input)[source_index];
    }
}

static __global__ void k_grouped_puts(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const PtirLaneChannelSlot* channels,
    const std::uint32_t* channel_bytes,
    const std::uint64_t* values,
    std::uint32_t value_count,
    const GroupedPutDesc* puts,
    std::uint32_t put_count) {
    const std::uint32_t lane = blockIdx.x;
    if (lane >= header->lane_count || *grouped_commit(lanes, lane) == 0) {
        return;
    }
    for (std::uint32_t put = 0; put < put_count; ++put) {
        const GroupedPutDesc descriptor = puts[put];
        const auto* source = grouped_value<std::uint8_t>(
            values, value_count, lane, descriptor.value);
        auto* destination = reinterpret_cast<std::uint8_t*>(
            channels[
                lanes[lane].channel_slot_offset + descriptor.channel]
                .pending_cell);
        const std::uint32_t bytes = channel_bytes[
            static_cast<std::size_t>(lane) *
                header->channel_slots_per_lane +
            descriptor.channel];
        for (std::uint32_t byte = threadIdx.x; byte < bytes;
             byte += blockDim.x) {
            destination[byte] = source[byte];
        }
        __syncthreads();
    }
}

static __global__ void k_grouped_finalize(
    const PtirLaneTableHeader* header,
    const PtirLaneRecord* lanes,
    const GroupedFinalizeLane* finalize,
    const std::uint32_t* slots,
    std::uint8_t* full,
    std::uint32_t* head,
    std::uint32_t* tail,
    const std::uint32_t* cap1) {
    const std::uint32_t lane =
        blockIdx.x * static_cast<std::uint32_t>(blockDim.x) + threadIdx.x;
    if (lane >= header->lane_count || *grouped_commit(lanes, lane) == 0) {
        return;
    }
    const GroupedFinalizeLane descriptor = finalize[lane];
    for (std::uint32_t index = 0; index < descriptor.put_count; ++index) {
        const std::uint32_t slot = slots[descriptor.put_offset + index];
        full[static_cast<std::size_t>(slot) * kMaxRing + tail[slot]] = 1;
        tail[slot] = (tail[slot] + 1) % cap1[slot];
    }
    for (std::uint32_t index = 0; index < descriptor.taken_count; ++index) {
        const std::uint32_t slot = slots[descriptor.taken_offset + index];
        full[static_cast<std::size_t>(slot) * kMaxRing + head[slot]] = 0;
        head[slot] = (head[slot] + 1) % cap1[slot];
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

inline bool grouped_stage_supported(
    const std::vector<GroupedLaneBinding>& lanes,
    std::string* reason = nullptr) {
    auto fail = [&](const char* message) {
        if (reason != nullptr) *reason = message;
        return false;
    };
    if (lanes.empty()) return fail("empty group");
    const plan::StagePlan* first = lanes.front().plan;
    if (first == nullptr || first->stage > PTIR_STAGE_EPILOGUE) {
        return fail("missing executable stage plan");
    }
    bool ragged = std::any_of(
        lanes.begin(), lanes.end(), [&](const GroupedLaneBinding& lane) {
            return lane.logits_row_count != lanes.front().logits_row_count;
        });
    for (const auto& value_type : first->value_types) {
        for (const auto& dimension : value_type.dims) {
            if (!dimension.symbolic) continue;
            const auto extent =
                static_cast<std::uint8_t>(dimension.value);
            const auto expected =
                grouped_extent(extent, lanes.front());
            ragged = ragged || std::any_of(
                lanes.begin() + 1,
                lanes.end(),
                [&](const GroupedLaneBinding& lane) {
                    return grouped_extent(extent, lane) != expected;
                });
        }
    }
    const bool bf16_logits =
        lanes.front().logits_bf16_rows != nullptr;
    std::vector<std::uint32_t> value_bases(first->ops.size());
    std::uint32_t next_value = 0;
    for (std::size_t node = 0; node < first->ops.size(); ++node) {
        value_bases[node] = next_value;
        next_value += first->ops[node].op.results;
    }
    for (const plan::NormalizedOp& op : first->ops) {
        if (!grouped_supported_tag(op.op.tag)) {
            return fail("stage contains an unsupported grouped op");
        }
        if (op.op.tag == PTIR_OP_INTRINSIC_VAL &&
            op.op.intr != PTIR_INTR_LOGITS &&
            op.op.intr != PTIR_INTR_MTP_LOGITS &&
            op.op.intr != PTIR_INTR_QUERY &&
            op.op.intr != PTIR_INTR_LAYER) {
            return fail("stage uses an unsupported intrinsic");
        }
        if (op.op.tag == PTIR_OP_INTRINSIC_VAL &&
            op.op.intr == PTIR_INTR_QUERY &&
            std::any_of(
                lanes.begin(), lanes.end(),
                [](const GroupedLaneBinding& lane) {
                    return lane.query_base == nullptr;
                })) {
            return fail("Query intrinsic is unavailable");
        }
        if (op.op.tag == PTIR_OP_INTRINSIC_VAL &&
            op.op.intr == PTIR_INTR_LAYER &&
            std::any_of(
                lanes.begin(), lanes.end(),
                [](const GroupedLaneBinding& lane) {
                    return lane.layer_base == nullptr;
                })) {
            return fail("Layer intrinsic is unavailable");
        }
        if (op.op.tag == PTIR_OP_INTRINSIC_VAL &&
            op.op.intr == PTIR_INTR_MTP_LOGITS &&
            std::any_of(
                lanes.begin(), lanes.end(),
                [](const GroupedLaneBinding& lane) {
                    return lane.mtp_logits_bf16_rows == nullptr;
                })) {
            return fail("MtpLogits dedicated rows are unavailable");
        }
        if (op.op.tag == PTIR_OP_INTRINSIC_VAL &&
            op.op.intr == PTIR_INTR_MTP_LOGITS) {
            const std::uint32_t value = value_bases[
                &op - first->ops.data()];
            if (value >= first->value_types.size() ||
                first->value_types[value].dims.size() != 2 ||
                first->value_types[value].dims[0].symbolic) {
                return fail("MtpLogits has no static draft-row layout");
            }
            const std::size_t expected =
                first->value_types[value].dims[0].value;
            if (std::any_of(
                    lanes.begin(), lanes.end(),
                    [expected](const GroupedLaneBinding& lane) {
                        return lane.mtp_logits_bf16_rows == nullptr ||
                            lane.mtp_logits_bf16_rows->size() != expected;
                    })) {
                return fail("MtpLogits dedicated row count does not match plan");
            }
        }
    }
    for (const plan::ValueType& value_type : first->value_types) {
        std::uint32_t varying_dimensions = 0;
        for (std::size_t dimension_index = 0;
             dimension_index < value_type.dims.size();
             ++dimension_index) {
            const plan::Dimension& dimension =
                value_type.dims[dimension_index];
            if (dimension.symbolic &&
                dimension.value != PTIR_EXTENT_KV_LEN &&
                dimension.value != PTIR_EXTENT_PAGE_COUNT &&
                dimension.value != PTIR_EXTENT_SAMPLED_ROWS &&
                dimension.value != PTIR_EXTENT_ROW_COUNT &&
                dimension.value != PTIR_EXTENT_TOKEN_COUNT &&
                dimension.value != PTIR_EXTENT_QUERY_LEN &&
                dimension.value != PTIR_EXTENT_KEY_LEN) {
                return fail("stage uses unsupported runtime extents");
            }
            if (dimension.symbolic) {
                const auto extent =
                    static_cast<std::uint8_t>(dimension.value);
                if (std::any_of(
                        lanes.begin(),
                        lanes.end(),
                        [&](const GroupedLaneBinding& lane) {
                            return grouped_extent(extent, lane) ==
                                kUnavailableGroupedExtent;
                        })) {
                    return fail("stage runtime extent is unavailable");
                }
            }
            if (ragged && dimension.symbolic) {
                const auto extent =
                    static_cast<std::uint8_t>(dimension.value);
                const auto first_extent =
                    grouped_extent(extent, lanes.front());
                varying_dimensions += std::any_of(
                    lanes.begin(),
                    lanes.end(),
                    [&](const GroupedLaneBinding& lane) {
                        return grouped_extent(extent, lane) != first_extent;
                    }) ? 1u : 0u;
            }
        }
        if (varying_dimensions > 1) {
            return fail(
                "stage has multiple varying symbolic dimensions in one value");
        }
    }
    for (const GroupedLaneBinding& lane : lanes) {
        const std::uint32_t physical_logits_stride =
            lane.logits_stride == 0 ? lane.vocab : lane.logits_stride;
        if (lane.instance == nullptr || lane.plan == nullptr ||
            lane.vocab != lanes.front().vocab ||
            lane.logits_stride != lanes.front().logits_stride ||
            lane.plan_identity != lanes.front().plan_identity ||
            lane.plan->signature_hash != first->signature_hash ||
            lane.plan->signature != first->signature ||
            lane.plan->ops.size() != first->ops.size() ||
            lane.plan->value_types.size() != first->value_types.size() ||
            lane.plan->channel_bindings.size() !=
                first->channel_bindings.size()) {
            return fail("group plan or static shape mismatch");
        }
        if (physical_logits_stride < lane.vocab) {
            return fail("physical logits stride is shorter than logical vocab");
        }
        if ((lane.logits_bf16_rows != nullptr) != bf16_logits) {
            return fail("group mixes FP32 and direct BF16 logits");
        }
        if ((lane.mtp_logits_bf16_rows != nullptr) !=
            (lanes.front().mtp_logits_bf16_rows != nullptr)) {
            return fail("group mixes MTP draft-row layouts");
        }
        if (lane.instance->trace().stages.empty()) {
            return fail("program has no executable stages");
        }
        for (std::size_t node = 0; node < first->ops.size(); ++node) {
            const container::COp& op = first->ops[node].op;
            std::uint32_t value = UINT32_MAX;
            if (op.tag == PTIR_OP_CHAN_TAKE ||
                op.tag == PTIR_OP_CHAN_READ) {
                value = value_bases[node];
            } else if (op.tag == PTIR_OP_CHAN_PUT && !op.args.empty()) {
                value = op.args[0];
            } else {
                continue;
            }
            if (value >= first->value_types.size()) {
                return fail("channel value is outside the grouped plan");
            }
            const plan::ValueType& value_type = first->value_types[value];
            const auto local = static_cast<std::uint32_t>(op.chan);
            const std::size_t value_bytes =
                grouped_numel(value_type, lane) *
                grouped_dtype_size(value_type.dtype);
            const std::size_t max_value_bytes = std::accumulate(
                lanes.begin(), lanes.end(), std::size_t{0},
                [&](std::size_t maximum,
                    const GroupedLaneBinding& candidate) {
                    return std::max(
                        maximum,
                        static_cast<std::size_t>(
                            grouped_numel(value_type, candidate)) *
                            grouped_dtype_size(value_type.dtype));
                });
            const std::size_t channel_bytes =
                grouped_channel_bytes(lane, local);
            if (value_bytes > channel_bytes ||
                max_value_bytes > channel_bytes) {
                return fail("channel value size does not match fixed cell");
            }
        }
    }
    std::unordered_map<std::uint32_t, bool> slots;
    for (const GroupedLaneBinding& lane : lanes) {
        for (std::uint32_t slot : lane.instance->view().slots()) {
            const auto ticket = std::find_if(
                lane.tickets->begin(),
                lane.tickets->end(),
                [slot](const DeviceHostChannelTicket& candidate) {
                    return candidate.slot == slot;
                });
            const bool mutates =
                ticket != lane.tickets->end() &&
                (ticket->flags & (kTicketConsume | kTicketPublish)) != 0;
            const auto [existing, inserted] =
                slots.emplace(slot, mutates);
            if (!inserted) {
                if (existing->second || mutates) {
                    return fail(
                        "shared mutable device channel slot requires ordered execution");
                }
            } else {
                existing->second = mutates;
            }
        }
    }
    return true;
}

inline bool grouped_dispatch_supported(
    const std::vector<GroupedLaneBinding>& lanes,
    bool has_recurrent_state,
    std::string* reason = nullptr) {
    if (has_recurrent_state) {
        if (reason != nullptr) {
            *reason = "recurrent-state lane is retry-ineligible";
        }
        return false;
    }
    return grouped_stage_supported(lanes, reason);
}

inline bool grouped_nucleus_region_supported(
    const plan::StagePlan& stage,
    const plan::Region& region) {
    if (!region.library ||
        region.library_op != PTIR_LIBRARY_NUCLEUS_SAMPLE ||
        region.nodes.size() != 13 ||
        region.inputs.size() != 3 ||
        region.outputs.size() != 1 ||
        !region.sinks.empty()) {
        return false;
    }
    const auto logits = region.inputs[0];
    const auto top_p = region.inputs[1];
    const auto rng_state = region.inputs[2];
    const auto token = region.outputs[0];
    if (logits >= stage.value_types.size() ||
        top_p >= stage.value_types.size() ||
        rng_state >= stage.value_types.size() ||
        token >= stage.value_types.size()) {
        return false;
    }
    const auto& logits_type = stage.value_types[logits];
    return logits_type.dtype == PTIR_DT_F32 &&
        !logits_type.dims.empty() &&
        !logits_type.dims.back().symbolic &&
        stage.value_types[top_p].dtype == PTIR_DT_F32 &&
        stage.value_types[rng_state].dtype == PTIR_DT_U32 &&
        (stage.value_types[token].dtype == PTIR_DT_I32 ||
         stage.value_types[token].dtype == PTIR_DT_U32);
}

inline bool grouped_nucleus_library_supported(
    const plan::StagePlan& stage,
    const plan::Region& region) {
    return grouped_nucleus_region_supported(stage, region) &&
        stage.value_types[region.inputs[0]].dims.back().value <=
            kMaxExactNucleusLibraryVocab;
}

class GroupedTier0GraphCache {
  public:
    struct Metrics {
        std::uint64_t captures = 0;
        std::uint64_t replays = 0;
        std::uint64_t evictions = 0;
        std::size_t max_nodes = 0;
        std::size_t retained_bytes = 0;
    };
    struct Entry {
        struct Buffer {
            void* pointer = nullptr;
            std::size_t bytes = 0;
        };
        struct HostBuffer {
            void* pointer = nullptr;
            std::size_t bytes = 0;
        };
        std::mutex mutex;
        cudaGraph_t graph = nullptr;
        cudaGraphExec_t exec = nullptr;
        cudaEvent_t done = nullptr;
        std::vector<std::uintptr_t> fingerprint;
        std::uint64_t captures = 0;
        std::uint64_t replays = 0;
        std::size_t node_count = 0;
        std::uint32_t body_op_launches = 0;
        std::vector<Buffer> buffers;
        std::vector<HostBuffer> host_buffers;
        std::uint64_t use_tick = 0;
        std::size_t retained_bytes = 0;

        ~Entry() {
            if (done != nullptr) cudaEventSynchronize(done);
            if (exec != nullptr) cudaGraphExecDestroy(exec);
            if (graph != nullptr) cudaGraphDestroy(graph);
            if (done != nullptr) cudaEventDestroy(done);
            for (const auto& buffer : buffers) {
                if (buffer.pointer != nullptr) cudaFree(buffer.pointer);
            }
            for (const auto& buffer : host_buffers) {
                if (buffer.pointer != nullptr) cudaFreeHost(buffer.pointer);
            }
        }
    };

    explicit GroupedTier0GraphCache(
        std::size_t capacity = 64,
        std::size_t byte_capacity = 512ULL << 20)
        : capacity_(std::max<std::size_t>(capacity, 1)),
          byte_capacity_(std::max<std::size_t>(byte_capacity, 1)) {}

    std::shared_ptr<Entry> get_or_create(const std::string& key) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = entries_.find(key);
        if (found != entries_.end()) {
            found->second->use_tick = ++tick_;
            return found->second;
        }
        while (entries_.size() >= capacity_ && evict_one(nullptr)) {}
        if (entries_.size() >= capacity_) return {};
        auto entry = std::make_shared<Entry>();
        entry->use_tick = ++tick_;
        entries_.emplace(key, entry);
        return entry;
    }

    bool account(
        const std::shared_ptr<Entry>& entry,
        std::size_t bytes) {
        std::lock_guard<std::mutex> lock(mutex_);
        auto found = std::find_if(
            entries_.begin(), entries_.end(),
            [&](const auto& candidate) {
                return candidate.second == entry;
            });
        if (found == entries_.end()) return false;
        retained_bytes_ -= entry->retained_bytes;
        entry->retained_bytes = bytes;
        retained_bytes_ += bytes;
        entry->use_tick = ++tick_;
        while (retained_bytes_ > byte_capacity_ &&
               evict_one(entry.get())) {}
        if (retained_bytes_ > byte_capacity_) {
            retained_bytes_ -= entry->retained_bytes;
            entries_.erase(found);
            ++evictions_;
            return false;
        }
        return true;
    }

    std::size_t size() const {
        std::lock_guard<std::mutex> lock(mutex_);
        return entries_.size();
    }

    Metrics metrics() const {
        Metrics result;
        std::vector<std::shared_ptr<Entry>> snapshot;
        {
            std::lock_guard<std::mutex> lock(mutex_);
            result.evictions = evictions_;
            result.retained_bytes = retained_bytes_;
            snapshot.reserve(entries_.size());
            for (const auto& [_, entry] : entries_) {
                snapshot.push_back(entry);
            }
        }
        for (const auto& entry : snapshot) {
            std::lock_guard<std::mutex> entry_lock(entry->mutex);
            result.captures += entry->captures;
            result.replays += entry->replays;
            result.max_nodes =
                std::max(result.max_nodes, entry->node_count);
        }
        return result;
    }

  private:
    bool evict_one(const Entry* keep) {
        auto victim = entries_.end();
        for (auto it = entries_.begin(); it != entries_.end(); ++it) {
            if (it->second.get() == keep || it->second.use_count() != 1) {
                continue;
            }
            if (victim == entries_.end() ||
                it->second->use_tick < victim->second->use_tick) {
                victim = it;
            }
        }
        if (victim == entries_.end()) return false;
        retained_bytes_ -= victim->second->retained_bytes;
        entries_.erase(victim);
        ++evictions_;
        return true;
    }

    std::size_t capacity_;
    std::size_t byte_capacity_;
    std::size_t retained_bytes_ = 0;
    std::uint64_t tick_ = 0;
    std::uint64_t evictions_ = 0;
    mutable std::mutex mutex_;
    std::unordered_map<std::string, std::shared_ptr<Entry>> entries_;
};

class GroupedTier0Executor {
  public:
    static GroupedLaunchResult run(
        const std::vector<GroupedLaneBinding>& lanes,
        cudaStream_t stream,
        GroupedTier0GraphCache* graph_cache = nullptr,
        GroupedExecutionOptions options = {}) {
        GroupedLaunchResult result;
        const plan::StagePlan& stage = *lanes.front().plan;
        const std::uint32_t lane_count =
            static_cast<std::uint32_t>(lanes.size());
        const std::uint32_t channel_count =
            static_cast<std::uint32_t>(stage.channel_bindings.size());
        const std::uint32_t value_count =
            static_cast<std::uint32_t>(stage.value_types.size());
        GroupedLaneBinding launch_shape = lanes.front();
        auto maximize_extent = [](std::uint32_t& target,
                                  std::uint32_t candidate) {
            if (candidate == kUnavailableGroupedExtent) return;
            if (target == kUnavailableGroupedExtent || candidate > target) {
                target = candidate;
            }
        };
        for (const auto& lane : lanes) {
            launch_shape.logits_row_count =
                std::max(launch_shape.logits_row_count, lane.logits_row_count);
            maximize_extent(launch_shape.row_count, lane.row_count);
            maximize_extent(launch_shape.token_count, lane.token_count);
            maximize_extent(launch_shape.kv_len, lane.kv_len);
            maximize_extent(launch_shape.page_count, lane.page_count);
            maximize_extent(launch_shape.query_len, lane.query_len);
            maximize_extent(launch_shape.key_len, lane.key_len);
        }
        const bool has_nucleus_region = std::any_of(
            stage.fused.regions.begin(),
            stage.fused.regions.end(),
            [](const plan::Region& region) {
                return region.library &&
                    region.library_op == PTIR_LIBRARY_NUCLEUS_SAMPLE;
            });
        const bool nucleus_library_enabled =
            has_nucleus_region &&
            std::getenv("PIE_CUDA_DISABLE_PTIR_NUCLEUS_LIBRARY") == nullptr;
        const bool direct_logits_rows =
            lanes.front().logits_bf16_rows != nullptr ||
            lanes.front().mtp_logits_bf16_rows != nullptr;
        const bool persistent_arena_candidate =
            graph_cache != nullptr && stream != nullptr &&
            std::none_of(
                stage.fused.regions.begin(), stage.fused.regions.end(),
                [](const plan::Region& region) {
                    return region.library &&
                        region.library_op !=
                            PTIR_LIBRARY_NUCLEUS_SAMPLE;
                }) &&
            std::none_of(
                stage.ops.begin(), stage.ops.end(),
                [](const plan::NormalizedOp& candidate) {
                    return candidate.op.tag == PTIR_OP_SORT_DESC ||
                        candidate.op.tag == PTIR_OP_TOP_K ||
                        candidate.op.tag == PTIR_OP_MATMUL;
                });
        const bool tier0_graph_candidate = persistent_arena_candidate;
        std::shared_ptr<GroupedTier0GraphCache::Entry> tier0_graph;
        std::unique_lock<std::mutex> tier0_graph_lock;
        if (persistent_arena_candidate) {
            std::string key(
                reinterpret_cast<const char*>(stage.signature.data()),
                stage.signature.size());
            const std::uint64_t plan_identity =
                lanes.front().plan_identity != 0
                    ? lanes.front().plan_identity
                    : compiled_stage_identity(stage);
            key.append(
                reinterpret_cast<const char*>(&plan_identity),
                sizeof(plan_identity));
            const std::uint8_t option_bits =
                (options.reset_commits ? 1u : 0u) |
                (options.pull_tickets ? 2u : 0u) |
                (options.finalize ? 4u : 0u) |
                (nucleus_library_enabled ? 8u : 0u) |
                (direct_logits_rows ? 16u : 0u);
            key.push_back(static_cast<char>(option_bits));
            key.append(
                reinterpret_cast<const char*>(&lane_count),
                sizeof(lane_count));
            const std::uint32_t graph_extents[] = {
                launch_shape.logits_row_count,
                launch_shape.row_count,
                launch_shape.token_count,
                launch_shape.kv_len,
                launch_shape.page_count,
                launch_shape.query_len,
                launch_shape.key_len,
                launch_shape.vocab,
                launch_shape.logits_stride,
            };
            key.append(
                reinterpret_cast<const char*>(graph_extents),
                sizeof(graph_extents));
            tier0_graph = graph_cache->get_or_create(key);
            if (tier0_graph != nullptr) {
                tier0_graph_lock =
                    std::unique_lock<std::mutex>(tier0_graph->mutex);
                if (tier0_graph->done != nullptr) {
                    CUDA_CHECK(cudaEventSynchronize(tier0_graph->done));
                }
            }
        }
        std::size_t graph_buffer_cursor = 0;
        const auto allocate = [&](void** pointer, std::size_t bytes) {
            if (tier0_graph == nullptr) {
                CUDA_CHECK(cudaMallocAsync(pointer, bytes, stream));
                return;
            }
            if (graph_buffer_cursor == tier0_graph->buffers.size()) {
                void* allocation = nullptr;
                CUDA_CHECK(cudaMalloc(&allocation, bytes));
                tier0_graph->buffers.push_back({allocation, bytes});
            } else if (
                tier0_graph->buffers[graph_buffer_cursor].bytes < bytes) {
                CUDA_CHECK(cudaFree(
                    tier0_graph->buffers[graph_buffer_cursor].pointer));
                CUDA_CHECK(cudaMalloc(
                    &tier0_graph->buffers[graph_buffer_cursor].pointer,
                    bytes));
                tier0_graph->buffers[graph_buffer_cursor].bytes = bytes;
                if (tier0_graph->exec != nullptr) {
                    CUDA_CHECK(cudaGraphExecDestroy(tier0_graph->exec));
                    tier0_graph->exec = nullptr;
                }
                if (tier0_graph->graph != nullptr) {
                    CUDA_CHECK(cudaGraphDestroy(tier0_graph->graph));
                    tier0_graph->graph = nullptr;
                }
            }
            *pointer =
                tier0_graph->buffers[graph_buffer_cursor++].pointer;
        };
        const auto release = [&](void* pointer) {
            if (tier0_graph == nullptr && pointer != nullptr) {
                CUDA_CHECK(cudaFreeAsync(pointer, stream));
            }
        };
        struct StagedMemset {
            void* destination = nullptr;
            std::size_t bytes = 0;
        };
        std::vector<GroupedGraphUpload> staged_uploads;
        std::vector<StagedMemset> staged_memsets;
        std::size_t host_buffer_cursor = 0;
        const auto upload = [&](void* destination,
                                const void* source,
                                std::size_t bytes) {
            if (tier0_graph == nullptr) {
                CUDA_CHECK(cudaMemcpyAsync(
                    destination, source, bytes,
                    cudaMemcpyHostToDevice, stream));
                return;
            }
            if (host_buffer_cursor == tier0_graph->host_buffers.size()) {
                void* pinned = nullptr;
                CUDA_CHECK(cudaMallocHost(&pinned, bytes));
                tier0_graph->host_buffers.push_back({pinned, bytes});
            } else if (
                tier0_graph->host_buffers[host_buffer_cursor].bytes < bytes) {
                CUDA_CHECK(cudaFreeHost(
                    tier0_graph->host_buffers[host_buffer_cursor].pointer));
                CUDA_CHECK(cudaMallocHost(
                    &tier0_graph->host_buffers[host_buffer_cursor].pointer,
                    bytes));
                tier0_graph->host_buffers[host_buffer_cursor].bytes = bytes;
            }
            void* pinned =
                tier0_graph->host_buffers[host_buffer_cursor++].pointer;
            std::memcpy(pinned, source, bytes);
            staged_uploads.push_back({destination, pinned, bytes});
        };
        const auto zero = [&](void* destination, std::size_t bytes) {
            if (tier0_graph == nullptr) {
                CUDA_CHECK(cudaMemsetAsync(
                    destination, 0, bytes, stream));
            } else {
                staged_memsets.push_back({destination, bytes});
            }
        };
        result.device_tickets_persistent = tier0_graph != nullptr;

        std::vector<PtirLaneRecord> host_lanes(lane_count);
        std::vector<PtirLaneChannelSlot> host_channels(
            static_cast<std::size_t>(lane_count) * channel_count);
        std::vector<std::uint32_t> host_channel_bytes(
            static_cast<std::size_t>(lane_count) * channel_count);
        std::vector<std::uint64_t> host_bf16_rows;
        std::vector<std::uint32_t> bf16_row_offsets(
            lane_count, UINT32_MAX);
        std::vector<std::uint64_t> host_mtp_bf16_rows;
        std::vector<std::uint32_t> mtp_bf16_offsets(
            lane_count, UINT32_MAX);
        std::vector<DeviceHostChannelTicket> host_tickets;
        std::vector<std::uint32_t> host_ticket_lanes;
        result.ticket_offsets.reserve(lane_count);
        result.ticket_counts.reserve(lane_count);

        for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
            const GroupedLaneBinding& binding = lanes[lane];
            PtirLaneRecord& record = host_lanes[lane];
            const std::uint32_t logits_stride = binding.logits_stride == 0
                ? binding.vocab
                : binding.logits_stride;
            record.logits_base =
                reinterpret_cast<std::uint64_t>(
                    binding.logits_base == nullptr
                        ? nullptr
                        : binding.logits_base +
                            static_cast<std::size_t>(
                                binding.logits_row_offset) *
                                logits_stride);
            if (binding.logits_bf16_rows != nullptr) {
                if (binding.logits_bf16_rows->size() !=
                    binding.logits_row_count) {
                    throw std::runtime_error(
                        "direct BF16 row table does not match lane rows");
                }
                if (binding.mtp_logits_bf16_rows != nullptr) {
                    mtp_bf16_offsets[lane] =
                        static_cast<std::uint32_t>(
                            host_mtp_bf16_rows.size());
                    host_mtp_bf16_rows.insert(
                        host_mtp_bf16_rows.end(),
                        binding.mtp_logits_bf16_rows->begin(),
                        binding.mtp_logits_bf16_rows->end());
                }
                bf16_row_offsets[lane] =
                    static_cast<std::uint32_t>(host_bf16_rows.size());
                host_bf16_rows.insert(
                    host_bf16_rows.end(),
                    binding.logits_bf16_rows->begin(),
                    binding.logits_bf16_rows->end());
            }
            record.logits_row_offset = 0;
            record.logits_row_count = binding.logits_row_count;
            record.kv_len = binding.kv_len;
            record.page_count = binding.page_count;
            record.row_count = binding.row_count;
            record.sampled_rows = binding.logits_row_count;
            record.token_count = binding.token_count;
            record.query_len = binding.query_len;
            record.key_len = binding.key_len;
            record.channel_slot_offset = lane * channel_count;
            record.rng_state = reinterpret_cast<std::uint64_t>(
                binding.row_seeds);
            record.commit_slot = reinterpret_cast<std::uint64_t>(
                binding.commit_slot != nullptr
                    ? binding.commit_slot
                    : binding.instance->commit_device_flag());
            record.active_row_mask =
                binding.logits_row_count >= 64
                    ? std::numeric_limits<std::uint64_t>::max()
                    : ((std::uint64_t{1} << binding.logits_row_count) - 1);

            for (std::uint32_t local = 0; local < channel_count; ++local) {
                const std::uint32_t dense =
                    binding.plan->channel_bindings[local];
                const std::uint32_t slot =
                    binding.instance->view().slot(dense);
                const auto ticket = std::find_if(
                    binding.tickets->begin(),
                    binding.tickets->end(),
                    [slot](const DeviceHostChannelTicket& candidate) {
                        return candidate.slot == slot;
                    });
                PtirLaneChannelSlot& channel =
                    host_channels[record.channel_slot_offset + local];
                const auto bytes = grouped_channel_bytes(binding, local);
                if (bytes > std::numeric_limits<std::uint32_t>::max()) {
                    throw std::runtime_error(
                        "grouped channel cell exceeds lane size table");
                }
                host_channel_bytes[record.channel_slot_offset + local] =
                    static_cast<std::uint32_t>(bytes);
                if (ticket != binding.tickets->end()) {
                    channel.expected_head = ticket->expected_head;
                    channel.expected_tail = ticket->expected_tail;
                    channel.committed_cell = reinterpret_cast<std::uint64_t>(
                        ticket->cells +
                        static_cast<std::size_t>(
                            (ticket->expected_head == kNoChannelTicket
                                 ? binding.instance->view().registry()->host_head(slot)
                                 : ticket->expected_head) %
                            ticket->cap1) *
                            ticket->native_bytes);
                    channel.pending_cell = reinterpret_cast<std::uint64_t>(
                        ticket->cells +
                        static_cast<std::size_t>(
                            (ticket->expected_tail == kNoChannelTicket
                                 ? binding.instance->view().registry()->host_tail(slot)
                                 : ticket->expected_tail) %
                            ticket->cap1) *
                            ticket->native_bytes);
                } else {
                    channel.expected_head = kNoChannelTicket;
                    channel.expected_tail = kNoChannelTicket;
                    channel.committed_cell = reinterpret_cast<std::uint64_t>(
                        binding.instance->view().committed_cell(dense));
                    channel.pending_cell = reinterpret_cast<std::uint64_t>(
                        binding.instance->view().pending_cell(dense));
                }
                if (binding.prior_put_slots != nullptr &&
                    binding.prior_put_slots->contains(slot)) {
                    channel.committed_cell = channel.pending_cell;
                }
            }

            result.ticket_offsets.push_back(
                static_cast<std::uint32_t>(host_tickets.size()));
            const std::uint32_t ticket_count = options.pull_tickets
                ? static_cast<std::uint32_t>(binding.tickets->size())
                : 0;
            result.ticket_counts.push_back(ticket_count);
            if (options.pull_tickets) {
                host_tickets.insert(
                    host_tickets.end(),
                    binding.tickets->begin(),
                    binding.tickets->end());
                host_ticket_lanes.insert(
                    host_ticket_lanes.end(), binding.tickets->size(), lane);
            }
        }

        PtirLaneTableHeader host_header{
            PTIR_LANE_TABLE_ABI_VERSION,
            lane_count,
            channel_count,
            host_bf16_rows.empty() && host_mtp_bf16_rows.empty()
                ? 0u
                : kGroupedLaneFlagBf16Rows,
        };
        PtirLaneTableHeader* device_header = nullptr;
        PtirLaneRecord* device_lanes = nullptr;
        PtirLaneChannelSlot* device_channels = nullptr;
        std::uint32_t* device_channel_bytes = nullptr;
        std::uint64_t* device_bf16_rows = nullptr;
        std::uint64_t* device_mtp_bf16_rows = nullptr;
        std::uint32_t* device_mtp_bf16_offsets = nullptr;
        std::uint64_t* device_values = nullptr;
        allocate(
            reinterpret_cast<void**>(&device_header),
            sizeof(host_header));
        allocate(
            reinterpret_cast<void**>(&device_lanes),
            host_lanes.size() * sizeof(PtirLaneRecord));
        allocate(
            reinterpret_cast<void**>(&device_channels),
            host_channels.size() * sizeof(PtirLaneChannelSlot));
        if (!host_channel_bytes.empty()) {
            allocate(
                reinterpret_cast<void**>(&device_channel_bytes),
                host_channel_bytes.size() * sizeof(std::uint32_t));
        }
        if (!host_bf16_rows.empty()) {
            allocate(
                reinterpret_cast<void**>(&device_bf16_rows),
                host_bf16_rows.size() * sizeof(std::uint64_t));
            upload(
                device_bf16_rows,
                host_bf16_rows.data(),
                host_bf16_rows.size() * sizeof(std::uint64_t));
            for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
                if (bf16_row_offsets[lane] != UINT32_MAX) {
                    host_lanes[lane].logits_base =
                        reinterpret_cast<std::uint64_t>(
                            device_bf16_rows + bf16_row_offsets[lane]);
                    host_lanes[lane].logits_row_offset = 0;
                }
            }
            if (!host_mtp_bf16_rows.empty()) {
                allocate(
                    reinterpret_cast<void**>(&device_mtp_bf16_rows),
                    host_mtp_bf16_rows.size() * sizeof(std::uint64_t));
                allocate(
                    reinterpret_cast<void**>(&device_mtp_bf16_offsets),
                    mtp_bf16_offsets.size() * sizeof(std::uint32_t));
                upload(
                    device_mtp_bf16_rows,
                    host_mtp_bf16_rows.data(),
                    host_mtp_bf16_rows.size() * sizeof(std::uint64_t));
                upload(
                    device_mtp_bf16_offsets,
                    mtp_bf16_offsets.data(),
                    mtp_bf16_offsets.size() * sizeof(std::uint32_t));
            }
        }
        upload(device_header, &host_header, sizeof(host_header));
        upload(
            device_lanes,
            host_lanes.data(),
            host_lanes.size() * sizeof(PtirLaneRecord));
        upload(
            device_channels,
            host_channels.data(),
            host_channels.size() * sizeof(PtirLaneChannelSlot));
        if (device_channel_bytes != nullptr) {
            upload(
                device_channel_bytes,
                host_channel_bytes.data(),
                host_channel_bytes.size() * sizeof(std::uint32_t));
        }

        std::uint32_t* device_ticket_lanes = nullptr;
        if (!host_tickets.empty()) {
            allocate(
                reinterpret_cast<void**>(&result.device_tickets),
                host_tickets.size() * sizeof(DeviceHostChannelTicket));
            allocate(
                reinterpret_cast<void**>(&device_ticket_lanes),
                host_ticket_lanes.size() * sizeof(std::uint32_t));
            upload(
                result.device_tickets,
                host_tickets.data(),
                host_tickets.size() * sizeof(DeviceHostChannelTicket));
            upload(
                device_ticket_lanes,
                host_ticket_lanes.data(),
                host_ticket_lanes.size() * sizeof(std::uint32_t));
        }

        std::vector<std::uint32_t> need_full_channels;
        std::vector<std::uint32_t> put_channels;
        std::vector<std::uint8_t> seen_full(channel_count, 0);
        std::vector<std::uint8_t> seen_put(channel_count, 0);
        for (const plan::NormalizedOp& normalized : stage.ops) {
            const container::COp& op = normalized.op;
            if (op.tag == PTIR_OP_CHAN_TAKE || op.tag == PTIR_OP_CHAN_READ) {
                const std::uint32_t channel =
                    static_cast<std::uint32_t>(op.chan);
                if (!seen_put[channel] && !seen_full[channel]) {
                    seen_full[channel] = 1;
                    need_full_channels.push_back(channel);
                }
            } else if (op.tag == PTIR_OP_CHAN_PUT) {
                const std::uint32_t channel =
                    static_cast<std::uint32_t>(op.chan);
                if (!seen_put[channel]) {
                    seen_put[channel] = 1;
                    put_channels.push_back(channel);
                }
            }
        }
        std::vector<GroupedReadinessLane> host_readiness(lane_count);
        std::vector<std::uint32_t> readiness_slots;
        for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
            GroupedReadinessLane& descriptor = host_readiness[lane];
            descriptor.full_offset =
                static_cast<std::uint32_t>(readiness_slots.size());
            for (std::uint32_t local : need_full_channels) {
                const std::uint32_t slot =
                    lanes[lane].instance->view().slot(
                        lanes[lane].plan->channel_bindings[local]);
                if (lanes[lane].prior_put_slots != nullptr &&
                    lanes[lane].prior_put_slots->contains(slot)) {
                    continue;
                }
                readiness_slots.push_back(slot);
                ++descriptor.full_count;
            }
            descriptor.empty_offset =
                static_cast<std::uint32_t>(readiness_slots.size());
            for (std::uint32_t local : put_channels) {
                if (seen_full[local]) continue;
                const std::uint32_t slot =
                    lanes[lane].instance->view().slot(
                        lanes[lane].plan->channel_bindings[local]);
                const bool prior_put =
                    lanes[lane].prior_put_slots != nullptr &&
                    lanes[lane].prior_put_slots->contains(slot);
                const bool prior_take =
                    lanes[lane].prior_take_slots != nullptr &&
                    lanes[lane].prior_take_slots->contains(slot);
                if (prior_put || prior_take) continue;
                readiness_slots.push_back(slot);
                ++descriptor.empty_count;
            }
        }
        GroupedReadinessLane* device_readiness = nullptr;
        std::uint32_t* device_readiness_slots = nullptr;
        allocate(
            reinterpret_cast<void**>(&device_readiness),
            host_readiness.size() * sizeof(GroupedReadinessLane));
        upload(
            device_readiness,
            host_readiness.data(),
            host_readiness.size() * sizeof(GroupedReadinessLane));
        if (!readiness_slots.empty()) {
            allocate(
                reinterpret_cast<void**>(&device_readiness_slots),
                readiness_slots.size() * sizeof(std::uint32_t));
            upload(
                device_readiness_slots,
                readiness_slots.data(),
                readiness_slots.size() * sizeof(std::uint32_t));
        }
        ChannelView& group_view = lanes.front().instance->view();

        std::vector<GroupedFinalizeLane> host_finalize(lane_count);
        std::vector<std::uint32_t> host_slots;
        for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
            GroupedFinalizeLane& descriptor = host_finalize[lane];
            descriptor.taken_offset =
                static_cast<std::uint32_t>(host_slots.size());
            descriptor.taken_count = static_cast<std::uint32_t>(
                lanes[lane].instance->commit_taken_slots().size());
            host_slots.insert(
                host_slots.end(),
                lanes[lane].instance->commit_taken_slots().begin(),
                lanes[lane].instance->commit_taken_slots().end());
            descriptor.put_offset =
                static_cast<std::uint32_t>(host_slots.size());
            descriptor.put_count = static_cast<std::uint32_t>(
                lanes[lane].instance->commit_put_slots().size());
            host_slots.insert(
                host_slots.end(),
                lanes[lane].instance->commit_put_slots().begin(),
                lanes[lane].instance->commit_put_slots().end());
        }
        GroupedFinalizeLane* device_finalize = nullptr;
        std::uint32_t* device_slots = nullptr;
        allocate(
            reinterpret_cast<void**>(&device_finalize),
            host_finalize.size() * sizeof(GroupedFinalizeLane));
        upload(
            device_finalize,
            host_finalize.data(),
            host_finalize.size() * sizeof(GroupedFinalizeLane));
        if (!host_slots.empty()) {
            allocate(
                reinterpret_cast<void**>(&device_slots),
                host_slots.size() * sizeof(std::uint32_t));
            upload(
                device_slots,
                host_slots.data(),
                host_slots.size() * sizeof(std::uint32_t));
        }

        const auto launch_prefix = [&] {
            if (options.reset_commits) {
                k_grouped_reset<<<
                    (lane_count + 127) / 128, 128, 0, stream>>>(
                    device_header, device_lanes);
                CUDA_CHECK(cudaGetLastError());
            }
            if (options.pull_tickets && !host_tickets.empty()) {
                k_grouped_pull_validate<<<
                    static_cast<std::uint32_t>(host_tickets.size()),
                    256,
                    0,
                    stream>>>(
                    result.device_tickets,
                    device_ticket_lanes,
                    static_cast<std::uint32_t>(host_tickets.size()),
                    device_lanes,
                    group_view.d_full());
                CUDA_CHECK(cudaGetLastError());
            }
            k_grouped_stage_readiness<<<
                (lane_count + 127) / 128, 128, 0, stream>>>(
                device_header,
                device_lanes,
                device_readiness,
                device_readiness_slots,
                group_view.d_full(),
                group_view.d_head(),
                group_view.d_tail(),
                group_view.d_cap1());
            CUDA_CHECK(cudaGetLastError());
        };

        std::vector<void*> scratch;
        GroupedPutDesc* device_puts = nullptr;
        std::vector<std::uintptr_t> graph_fingerprint;
        bool tier0_graph_replay = false;
        bool tier0_graph_capture = false;
        std::optional<StreamCaptureGuard> capture_guard;
        std::vector<std::uint64_t> host_values(
            static_cast<std::size_t>(lane_count) * value_count, 0);
        struct DynamicRootCopy {
            std::uint32_t value = 0;
            std::vector<std::uint64_t> sources;
            std::uint64_t* device_sources = nullptr;
            GroupedDynamicShape shape{};
            std::uint32_t element_bytes = 0;
            std::uint32_t logical_row_width = 0;
            std::uint32_t source_row_stride = 0;
        };
        std::vector<DynamicRootCopy> dynamic_roots;
        struct ReshapeMaterialization {
            bool enabled = false;
            std::uint32_t source = 0;
            std::uint32_t output = 0;
            GroupedDynamicShape shape{};
            std::uint32_t element_bytes = 0;
            std::uint8_t direct_kind = 0;
        };
        std::vector<ReshapeMaterialization> reshape_materializations(
            stage.ops.size());
        std::vector<std::uint8_t> direct_bf16_values(
            value_count, 0);
        std::vector<std::uint8_t> raw_direct_bf16_values(
            value_count, 0);
        std::vector<std::uint32_t> bases(stage.ops.size(), 0);
        std::vector<std::uint32_t> pending_sources(
            channel_count, UINT32_MAX);
        std::uint32_t next_value = 0;
        if (!tier0_graph_replay) {
        for (std::size_t node = 0; node < stage.ops.size(); ++node) {
            bases[node] = next_value;
            next_value += stage.ops[node].op.results;
        }
        if (next_value != value_count) {
            throw std::runtime_error("grouped plan value layout mismatch");
        }
        const plan::Region* nucleus_region = nullptr;
        bool multiple_nucleus_regions = false;
        for (const auto& region : stage.fused.regions) {
            if (region.library &&
                region.library_op == PTIR_LIBRARY_NUCLEUS_SAMPLE &&
                std::getenv(
                    "PIE_CUDA_DISABLE_PTIR_NUCLEUS_LIBRARY") == nullptr) {
                if (nucleus_region != nullptr) {
                    multiple_nucleus_regions = true;
                } else {
                    nucleus_region = &region;
                }
            }
            if (region.library &&
                region.library_op == PTIR_LIBRARY_TOP_K) {
                result.used_selection_library = true;
            }
        }
        if (multiple_nucleus_regions) nucleus_region = nullptr;
        const bool nucleus_region_supported =
            nucleus_region != nullptr &&
            grouped_nucleus_region_supported(stage, *nucleus_region);
        bool exact_nucleus =
            nucleus_region_supported &&
            grouped_nucleus_library_supported(stage, *nucleus_region);
        bool scalable_nucleus =
            nucleus_region_supported && !exact_nucleus;
        bool use_nucleus_library =
            exact_nucleus || scalable_nucleus;
        result.large_nucleus_scalable = scalable_nucleus;
        std::uint32_t nucleus_launch_node = !use_nucleus_library
            ? UINT32_MAX
            : library_region_launch_node(*nucleus_region);
        std::uint32_t nucleus_output = use_nucleus_library
            ? nucleus_region->outputs[0]
            : UINT32_MAX;
        std::vector<std::uint8_t> nucleus_nodes(stage.ops.size(), 0);
        if (use_nucleus_library) {
            for (const std::uint32_t node : nucleus_region->nodes) {
                if (node >= nucleus_nodes.size()) {
                    throw std::runtime_error(
                        "nucleus library node is outside the stage plan");
                }
                nucleus_nodes[node] = 1;
            }
        }
        std::vector<std::size_t> fixed_value_bytes(value_count, 0);
        for (std::size_t node = 0; node < stage.ops.size(); ++node) {
            const auto& op = stage.ops[node].op;
            std::uint32_t value = UINT32_MAX;
            if (op.tag == PTIR_OP_CHAN_TAKE ||
                op.tag == PTIR_OP_CHAN_READ) {
                value = bases[node];
            } else if (
                op.tag == PTIR_OP_CHAN_PUT && !op.args.empty()) {
                value = op.args[0];
            }
            if (value >= value_count || op.chan < 0) continue;
            const auto channel = static_cast<std::uint32_t>(op.chan);
            for (const auto& binding : lanes) {
                fixed_value_bytes[value] = std::max(
                    fixed_value_bytes[value],
                    grouped_channel_bytes(binding, channel));
            }
        }
        for (std::size_t node = 0; node < stage.ops.size(); ++node) {
            const container::COp& op = stage.ops[node].op;
            const std::uint32_t base = bases[node];
            if (op.tag == PTIR_OP_CHAN_PUT) {
                pending_sources[static_cast<std::uint32_t>(op.chan)] =
                    op.args[0];
                continue;
            }
            if (op.results == 0) continue;
            if (op.tag == PTIR_OP_RESHAPE) {
                if (fixed_value_bytes[base] != 0) {
                    const auto shape = grouped_dynamic_shape(
                        stage.value_types[base], lanes);
                    const std::size_t element_bytes =
                        grouped_dtype_size(stage.value_types[base].dtype);
                    const std::size_t bytes_per_lane = std::max({
                        static_cast<std::size_t>(shape.max_numel) *
                            element_bytes,
                        fixed_value_bytes[base],
                        element_bytes,
                    });
                    void* allocation = nullptr;
                    allocate(&allocation, bytes_per_lane * lane_count);
                    zero(allocation, bytes_per_lane * lane_count);
                    scratch.push_back(allocation);
                    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
                        host_values[
                            static_cast<std::size_t>(lane) * value_count +
                            base] = reinterpret_cast<std::uint64_t>(
                                static_cast<std::uint8_t*>(allocation) +
                                bytes_per_lane * lane);
                    }
                    reshape_materializations[node] = {
                        true,
                        op.args[0],
                        base,
                        shape,
                        static_cast<std::uint32_t>(element_bytes),
                        direct_bf16_values[op.args[0]],
                    };
                    continue;
                }
                direct_bf16_values[base] =
                    direct_bf16_values[op.args[0]];
                for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
                    host_values[
                        static_cast<std::size_t>(lane) * value_count + base] =
                        host_values[
                            static_cast<std::size_t>(lane) * value_count +
                            op.args[0]];
                }
                continue;
            }
            if (op.tag == PTIR_OP_CHAN_TAKE ||
                op.tag == PTIR_OP_CHAN_READ) {
                const std::uint32_t pending =
                    pending_sources[static_cast<std::uint32_t>(op.chan)];
                const auto shape =
                    grouped_dynamic_shape(stage.value_types[base], lanes);
                if (pending == UINT32_MAX && shape.extent != 0xff) {
                    const std::size_t element_bytes =
                        grouped_dtype_size(stage.value_types[base].dtype);
                    const std::size_t bytes_per_lane = std::max({
                        static_cast<std::size_t>(shape.max_numel) *
                            element_bytes,
                        fixed_value_bytes[base],
                        element_bytes,
                    });
                    void* allocation = nullptr;
                    allocate(&allocation, bytes_per_lane * lane_count);
                    zero(allocation, bytes_per_lane * lane_count);
                    scratch.push_back(allocation);
                    DynamicRootCopy copy;
                    copy.value = base;
                    copy.shape = shape;
                    copy.element_bytes =
                        grouped_dtype_size(stage.value_types[base].dtype);
                    copy.sources.reserve(lane_count);
                    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
                        copy.sources.push_back(
                            host_channels[
                                static_cast<std::size_t>(lane) * channel_count +
                                static_cast<std::uint32_t>(op.chan)]
                                .committed_cell);
                        host_values[
                            static_cast<std::size_t>(lane) * value_count + base] =
                            reinterpret_cast<std::uint64_t>(
                                static_cast<std::uint8_t*>(allocation) +
                                bytes_per_lane * lane);
                    }
                    allocate(
                        reinterpret_cast<void**>(&copy.device_sources),
                        copy.sources.size() * sizeof(std::uint64_t));
                    upload(
                        copy.device_sources,
                        copy.sources.data(),
                        copy.sources.size() * sizeof(std::uint64_t));
                    scratch.push_back(copy.device_sources);
                    dynamic_roots.push_back(std::move(copy));
                    continue;
                }
                for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
                    host_values[
                        static_cast<std::size_t>(lane) * value_count + base] =
                        pending != UINT32_MAX
                            ? host_values[
                                static_cast<std::size_t>(lane) * value_count +
                                pending]
                            : host_channels[
                                static_cast<std::size_t>(lane) * channel_count +
                                static_cast<std::uint32_t>(op.chan)]
                                .committed_cell;
                }
                continue;
            }
            if (op.tag == PTIR_OP_INTRINSIC_VAL) {
                if (op.intr == PTIR_INTR_QUERY ||
                    op.intr == PTIR_INTR_LAYER) {
                    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
                        const void* source = op.intr == PTIR_INTR_QUERY
                            ? lanes[lane].query_base
                            : lanes[lane].layer_base;
                        host_values[
                            static_cast<std::size_t>(lane) * value_count + base] =
                            reinterpret_cast<std::uint64_t>(source);
                    }
                    continue;
                }
                const auto shape =
                    grouped_dynamic_shape(stage.value_types[base], lanes);
                const bool mtp = op.intr == PTIR_INTR_MTP_LOGITS;
                const bool direct_bf16 = mtp
                    ? lanes.front().mtp_logits_bf16_rows != nullptr
                    : lanes.front().logits_bf16_rows != nullptr;
                if (direct_bf16) {
                    if (stage.value_types[base].dtype != PTIR_DT_F32 ||
                        shape.max_numel % lanes.front().vocab != 0) {
                        throw std::runtime_error(
                            "direct BF16 logits require an F32 row tensor "
                            "(intrinsic=" + std::to_string(op.intr) +
                            ", value=" + std::to_string(base) +
                            ", dtype=" + std::to_string(
                                stage.value_types[base].dtype) +
                            ", numel=" + std::to_string(shape.max_numel) +
                            ", vocab=" + std::to_string(
                                lanes.front().vocab) + ")");
                    }
                    direct_bf16_values[base] = mtp ? 2 : 1;
                    raw_direct_bf16_values[base] = mtp ? 2 : 1;
                    continue;
                }
                if (shape.extent != 0xff) {
                    if (stage.value_types[base].dtype != PTIR_DT_F32 ||
                        lanes.front().vocab == 0 ||
                        shape.max_numel % lanes.front().vocab != 0) {
                        throw std::runtime_error(
                            "FP32 logits require a logical row tensor");
                    }
                    const std::size_t element_bytes =
                        grouped_dtype_size(stage.value_types[base].dtype);
                    const std::size_t bytes_per_lane = std::max({
                        static_cast<std::size_t>(shape.max_numel) *
                            element_bytes,
                        fixed_value_bytes[base],
                        element_bytes,
                    });
                    void* allocation = nullptr;
                    allocate(&allocation, bytes_per_lane * lane_count);
                    zero(allocation, bytes_per_lane * lane_count);
                    scratch.push_back(allocation);
                    DynamicRootCopy copy;
                    copy.value = base;
                    copy.shape = shape;
                    copy.element_bytes =
                        grouped_dtype_size(stage.value_types[base].dtype);
                    copy.logical_row_width = lanes.front().vocab;
                    copy.source_row_stride =
                        lanes.front().logits_stride == 0
                            ? lanes.front().vocab
                            : lanes.front().logits_stride;
                    copy.sources.reserve(lane_count);
                    for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
                        copy.sources.push_back(reinterpret_cast<std::uint64_t>(
                            lanes[lane].logits_base +
                            static_cast<std::size_t>(
                                lanes[lane].logits_row_offset) *
                                (lanes[lane].logits_stride == 0
                                     ? lanes[lane].vocab
                                     : lanes[lane].logits_stride)));
                        host_values[
                            static_cast<std::size_t>(lane) * value_count + base] =
                            reinterpret_cast<std::uint64_t>(
                                static_cast<std::uint8_t*>(allocation) +
                                bytes_per_lane * lane);
                    }
                    allocate(
                        reinterpret_cast<void**>(&copy.device_sources),
                        copy.sources.size() * sizeof(std::uint64_t));
                    upload(
                        copy.device_sources,
                        copy.sources.data(),
                        copy.sources.size() * sizeof(std::uint64_t));
                    scratch.push_back(copy.device_sources);
                    dynamic_roots.push_back(std::move(copy));
                    continue;
                }
                for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
                    host_values[
                        static_cast<std::size_t>(lane) * value_count + base] =
                        reinterpret_cast<std::uint64_t>(
                            lanes[lane].logits_base +
                            static_cast<std::size_t>(
                                lanes[lane].logits_row_offset) *
                                (lanes[lane].logits_stride == 0
                                     ? lanes[lane].vocab
                                     : lanes[lane].logits_stride));
                }
                continue;
            }

            for (std::uint32_t result_index = 0;
                 result_index < op.results;
                 ++result_index) {
                const std::uint32_t value = base + result_index;
                const auto shape =
                    grouped_dynamic_shape(stage.value_types[value], lanes);
                const std::size_t element_bytes =
                    grouped_dtype_size(stage.value_types[value].dtype);
                const std::size_t bytes_per_lane = std::max({
                    static_cast<std::size_t>(shape.max_numel) *
                        element_bytes,
                    fixed_value_bytes[value],
                    element_bytes,
                });
                void* allocation = nullptr;
                allocate(&allocation, bytes_per_lane * lane_count);
                zero(allocation, bytes_per_lane * lane_count);
                scratch.push_back(allocation);
                for (std::uint32_t lane = 0; lane < lane_count; ++lane) {
                    host_values[
                        static_cast<std::size_t>(lane) * value_count + value] =
                        reinterpret_cast<std::uint64_t>(
                            static_cast<std::uint8_t*>(allocation) +
                            bytes_per_lane * lane);
                }
            }
        }
        if (use_nucleus_library) {
            const std::uint32_t logits = nucleus_region->inputs[0];
            const bool direct_alias_without_materialization =
                direct_bf16_values[logits] != 0 &&
                raw_direct_bf16_values[logits] == 0;
            if (direct_alias_without_materialization) {
                exact_nucleus = false;
                scalable_nucleus = false;
                use_nucleus_library = false;
                result.large_nucleus_scalable = false;
                nucleus_launch_node = UINT32_MAX;
                nucleus_output = UINT32_MAX;
                std::fill(
                    nucleus_nodes.begin(), nucleus_nodes.end(), 0);
            }
        }

        allocate(
            reinterpret_cast<void**>(&device_values),
            host_values.size() * sizeof(std::uint64_t));
        upload(
            device_values,
            host_values.data(),
            host_values.size() * sizeof(std::uint64_t));
        std::vector<GroupedPutDesc> puts;
        for (const plan::NormalizedOp& normalized : stage.ops) {
            const container::COp& put = normalized.op;
            if (put.tag != PTIR_OP_CHAN_PUT) continue;
            puts.push_back({
                put.args[0],
                static_cast<std::uint32_t>(put.chan),
            });
        }
        if (!puts.empty()) {
            allocate(
                reinterpret_cast<void**>(&device_puts),
                puts.size() * sizeof(GroupedPutDesc));
            upload(
                device_puts,
                puts.data(),
                puts.size() * sizeof(GroupedPutDesc));
        }

        GroupedNucleusLaunch nucleus_launch{};
        float* nucleus_probabilities = nullptr;
        std::uint64_t* nucleus_keys_in = nullptr;
        std::uint64_t* nucleus_keys_out = nullptr;
        std::uint32_t* nucleus_indices_in = nullptr;
        std::uint32_t* nucleus_indices_out = nullptr;
        std::uint32_t* nucleus_offsets = nullptr;
        void* nucleus_sort_temp = nullptr;
        std::size_t nucleus_sort_temp_bytes = 0;
        std::size_t nucleus_items = 0;
        std::uint32_t nucleus_segments = 0;
        if (use_nucleus_library) {
            const std::uint32_t logits = nucleus_region->inputs[0];
            const std::uint32_t top_p = nucleus_region->inputs[1];
            const std::uint32_t rng_state = nucleus_region->inputs[2];
            const auto& logits_type = stage.value_types[logits];
            const std::uint32_t rows =
                grouped_rows(logits_type, launch_shape);
            const std::uint32_t vocab = static_cast<std::uint32_t>(
                grouped_numel(logits_type, launch_shape) / rows);
            nucleus_launch = {
                logits,
                top_p,
                rng_state,
                nucleus_output,
                UINT32_MAX,
                rows,
                vocab,
                static_cast<std::uint32_t>(grouped_numel(
                    stage.value_types[top_p], launch_shape)),
                raw_direct_bf16_values[logits],
            };
            if (scalable_nucleus) {
                nucleus_segments = lane_count * rows;
                nucleus_items =
                    static_cast<std::size_t>(nucleus_segments) * vocab;
                allocate(
                    reinterpret_cast<void**>(&nucleus_probabilities),
                    nucleus_items * sizeof(float));
                allocate(
                    reinterpret_cast<void**>(&nucleus_keys_in),
                    nucleus_items * sizeof(std::uint64_t));
                allocate(
                    reinterpret_cast<void**>(&nucleus_keys_out),
                    nucleus_items * sizeof(std::uint64_t));
                allocate(
                    reinterpret_cast<void**>(&nucleus_indices_in),
                    nucleus_items * sizeof(std::uint32_t));
                allocate(
                    reinterpret_cast<void**>(&nucleus_indices_out),
                    nucleus_items * sizeof(std::uint32_t));
                allocate(
                    reinterpret_cast<void**>(&nucleus_offsets),
                    (static_cast<std::size_t>(nucleus_segments) + 1) *
                        sizeof(std::uint32_t));
                std::vector<std::uint32_t> host_offsets(
                    nucleus_segments + 1);
                for (std::uint32_t segment = 0;
                     segment <= nucleus_segments;
                     ++segment) {
                    host_offsets[segment] = segment * vocab;
                }
                upload(
                    nucleus_offsets,
                    host_offsets.data(),
                    host_offsets.size() * sizeof(std::uint32_t));
                CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(
                    nullptr,
                    nucleus_sort_temp_bytes,
                    nucleus_keys_in,
                    nucleus_keys_out,
                    nucleus_indices_in,
                    nucleus_indices_out,
                    static_cast<int>(nucleus_items),
                    static_cast<int>(nucleus_segments),
                    nucleus_offsets,
                    nucleus_offsets + 1,
                    0,
                    64,
                    stream));
                allocate(
                    &nucleus_sort_temp,
                    std::max<std::size_t>(nucleus_sort_temp_bytes, 1));
                scratch.insert(
                    scratch.end(),
                    {
                        nucleus_probabilities,
                        nucleus_keys_in,
                        nucleus_keys_out,
                        nucleus_indices_in,
                        nucleus_indices_out,
                        nucleus_offsets,
                        nucleus_sort_temp,
                    });
            }
        }

        if (tier0_graph != nullptr && graph_cache != nullptr) {
            std::size_t retained = 0;
            for (const auto& buffer : tier0_graph->buffers) {
                    retained += buffer.bytes;
            }
            for (const auto& buffer : tier0_graph->host_buffers) {
                    retained += buffer.bytes;
            }
            graph_cache->account(tier0_graph, retained);
        }

        if (tier0_graph_candidate && tier0_graph != nullptr) {
            auto add_pointer = [&](const void* pointer) {
                graph_fingerprint.push_back(
                    reinterpret_cast<std::uintptr_t>(pointer));
            };
            add_pointer(device_header);
            add_pointer(device_lanes);
            add_pointer(device_channels);
            add_pointer(device_channel_bytes);
            add_pointer(result.device_tickets);
            add_pointer(device_ticket_lanes);
            add_pointer(device_readiness);
            add_pointer(device_readiness_slots);
            add_pointer(device_finalize);
            add_pointer(device_slots);
            add_pointer(device_values);
            add_pointer(device_puts);
            add_pointer(device_bf16_rows);
            add_pointer(device_mtp_bf16_rows);
            add_pointer(device_mtp_bf16_offsets);
            for (const void* allocation : scratch) {
                add_pointer(allocation);
            }
            tier0_graph_replay =
                tier0_graph->exec != nullptr &&
                tier0_graph->fingerprint == graph_fingerprint;
            if (tier0_graph->exec != nullptr &&
                !tier0_graph_replay &&
                std::getenv("PIE_PTIR_TRACE") != nullptr) {
                for (std::size_t index = 0;
                     index < std::min(
                         tier0_graph->fingerprint.size(),
                         graph_fingerprint.size());
                     ++index) {
                    if (tier0_graph->fingerprint[index] !=
                        graph_fingerprint[index]) {
                        std::fprintf(
                            stderr,
                            "[ptir-tier0-graph] pointer %zu changed "
                            "%zx -> %zx\n",
                            index,
                            static_cast<std::size_t>(
                                tier0_graph->fingerprint[index]),
                            static_cast<std::size_t>(
                                graph_fingerprint[index]));
                    }
                }
            }
            if (!tier0_graph_replay) {
                if (tier0_graph->exec != nullptr) {
                    CUDA_CHECK(cudaGraphExecDestroy(tier0_graph->exec));
                    tier0_graph->exec = nullptr;
                }
                if (tier0_graph->graph != nullptr) {
                    CUDA_CHECK(cudaGraphDestroy(tier0_graph->graph));
                    tier0_graph->graph = nullptr;
                }
                capture_guard.emplace(stream);
                tier0_graph_capture = true;
            }
        }
        if (tier0_graph_capture) {
            for (const auto& staged : staged_uploads) {
                CUDA_CHECK(cudaMemcpyAsync(
                    staged.destination,
                    staged.source,
                    staged.bytes,
                    cudaMemcpyHostToDevice,
                    stream));
            }
            for (const auto& staged : staged_memsets) {
                CUDA_CHECK(cudaMemsetAsync(
                    staged.destination, 0, staged.bytes, stream));
            }
        }
        if (!tier0_graph_replay) {
        launch_prefix();
        for (const auto& copy : dynamic_roots) {
            const auto blocks = static_cast<std::uint32_t>(std::min<std::uint64_t>(
                (copy.shape.max_numel * lane_count + kTier0Block - 1) /
                    kTier0Block,
                65535));
            k_grouped_copy_dynamic_root<<<blocks, kTier0Block, 0, stream>>>(
                device_header,
                device_lanes,
                device_values,
                value_count,
                copy.value,
                copy.device_sources,
                copy.shape,
                copy.element_bytes,
                copy.logical_row_width,
                copy.source_row_stride);
            CUDA_CHECK(cudaGetLastError());
            ++result.body_op_launches;
        }
        for (std::size_t node = 0; node < stage.ops.size(); ++node) {
            const container::COp& op = stage.ops[node].op;
            const std::uint32_t output = bases[node];
            if (node == nucleus_launch_node && exact_nucleus) {
                k_grouped_nucleus_sample<<<
                    lane_count * nucleus_launch.rows,
                    kCanonicalReduceWidth,
                    0,
                    stream>>>(
                    device_header,
                    device_lanes,
                    device_channels,
                    device_mtp_bf16_rows,
                    device_mtp_bf16_offsets,
                    device_values,
                    value_count,
                    nucleus_launch);
                CUDA_CHECK(cudaGetLastError());
                result.used_nucleus_library = true;
                ++result.body_op_launches;
                continue;
            }
            if (node == nucleus_launch_node && scalable_nucleus) {
                k_grouped_nucleus_probabilities<<<
                    nucleus_segments,
                    kCanonicalReduceWidth,
                    0,
                    stream>>>(
                    device_header,
                    device_lanes,
                    device_mtp_bf16_rows,
                    device_mtp_bf16_offsets,
                    device_values,
                    value_count,
                    nucleus_probabilities,
                    nucleus_launch);
                CUDA_CHECK(cudaGetLastError());
                k_grouped_nucleus_sort_keys<<<
                    grouped_grid(nucleus_items),
                    kTier0Block,
                    0,
                    stream>>>(
                    device_header,
                    device_lanes,
                    nucleus_probabilities,
                    nucleus_keys_in,
                    nucleus_indices_in,
                    nucleus_launch.rows,
                    nucleus_launch.len);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cub::DeviceSegmentedRadixSort::SortPairs(
                    nucleus_sort_temp,
                    nucleus_sort_temp_bytes,
                    nucleus_keys_in,
                    nucleus_keys_out,
                    nucleus_indices_in,
                    nucleus_indices_out,
                    static_cast<int>(nucleus_items),
                    static_cast<int>(nucleus_segments),
                    nucleus_offsets,
                    nucleus_offsets + 1,
                    0,
                    64,
                    stream));
                k_grouped_nucleus_sorted_finish<<<
                    (nucleus_segments + 127) / 128,
                    128,
                    0,
                    stream>>>(
                    device_header,
                    device_lanes,
                    device_channels,
                    device_mtp_bf16_rows,
                    device_mtp_bf16_offsets,
                    device_values,
                    value_count,
                    nucleus_probabilities,
                    nucleus_indices_out,
                    nucleus_launch);
                CUDA_CHECK(cudaGetLastError());
                result.used_nucleus_library = true;
                result.body_op_launches += 3;
                continue;
            }
            if (use_nucleus_library && nucleus_nodes[node] != 0) {
                continue;
            }
            const auto& reshape = reshape_materializations[node];
            if (reshape.enabled) {
                k_grouped_materialize_reshape<<<
                    grouped_grid(
                        static_cast<std::uint64_t>(lane_count) *
                        reshape.shape.max_numel),
                    kTier0Block,
                    0,
                    stream>>>(
                    device_header,
                    device_lanes,
                    device_mtp_bf16_rows,
                    device_mtp_bf16_offsets,
                    device_values,
                    value_count,
                    reshape.source,
                    reshape.output,
                    reshape.shape,
                    reshape.element_bytes,
                    launch_shape.vocab,
                    reshape.direct_kind);
                CUDA_CHECK(cudaGetLastError());
                ++result.body_op_launches;
                continue;
            }
            if (op.results == 0 || op.tag == PTIR_OP_CHAN_TAKE ||
                op.tag == PTIR_OP_CHAN_READ ||
                op.tag == PTIR_OP_INTRINSIC_VAL ||
                op.tag == PTIR_OP_RESHAPE) {
                continue;
            }
            launch_op(
                op,
                stage,
                launch_shape,
                lanes,
                direct_bf16_values,
                device_mtp_bf16_rows,
                device_mtp_bf16_offsets,
                output,
                device_header,
                device_lanes,
                device_values,
                value_count,
                lane_count,
                stream);
            ++result.body_op_launches;
        }
        if (!puts.empty()) {
            k_grouped_puts<<<lane_count, 256, 0, stream>>>(
                device_header,
                device_lanes,
                device_channels,
                device_channel_bytes,
                device_values,
                value_count,
                device_puts,
                static_cast<std::uint32_t>(puts.size()));
            CUDA_CHECK(cudaGetLastError());
            ++result.body_op_launches;
        }
        }
        }

        if (options.finalize && !tier0_graph_replay) {
            k_grouped_finalize<<<
                (lane_count + 127) / 128, 128, 0, stream>>>(
                device_header,
                device_lanes,
                device_finalize,
                device_slots,
                group_view.d_full(),
                group_view.d_head(),
                group_view.d_tail(),
                group_view.d_cap1());
            CUDA_CHECK(cudaGetLastError());
        }
        if (tier0_graph_candidate && tier0_graph != nullptr) {
            if (tier0_graph_capture) {
                tier0_graph->graph = capture_guard->end();
                capture_guard.reset();
                CUDA_CHECK(cudaGraphInstantiate(
                    &tier0_graph->exec,
                    tier0_graph->graph,
                    nullptr,
                    nullptr,
                    0));
                CUDA_CHECK(cudaGraphGetNodes(
                    tier0_graph->graph,
                    nullptr,
                    &tier0_graph->node_count));
                tier0_graph->fingerprint = graph_fingerprint;
                tier0_graph->body_op_launches =
                    result.body_op_launches;
                ++tier0_graph->captures;
            } else if (tier0_graph_replay) {
                result.body_op_launches =
                    tier0_graph->body_op_launches;
                ++tier0_graph->replays;
            }
            CUDA_CHECK(cudaGraphLaunch(tier0_graph->exec, stream));
            if (tier0_graph->done == nullptr) {
                CUDA_CHECK(cudaEventCreateWithFlags(
                    &tier0_graph->done, cudaEventDisableTiming));
            }
        }
        if (tier0_graph != nullptr &&
            result.device_tickets != nullptr &&
            !host_tickets.empty()) {
            DeviceHostChannelTicket* callback_tickets = nullptr;
            CUDA_CHECK(cudaMallocAsync(
                reinterpret_cast<void**>(&callback_tickets),
                host_tickets.size() * sizeof(DeviceHostChannelTicket),
                stream));
            CUDA_CHECK(cudaMemcpyAsync(
                callback_tickets,
                result.device_tickets,
                host_tickets.size() * sizeof(DeviceHostChannelTicket),
                cudaMemcpyDeviceToDevice,
                stream));
            result.device_tickets = callback_tickets;
            result.device_tickets_persistent = false;
        }
        if (tier0_graph != nullptr) {
            if (tier0_graph->done == nullptr) {
                CUDA_CHECK(cudaEventCreateWithFlags(
                    &tier0_graph->done, cudaEventDisableTiming));
            }
            CUDA_CHECK(cudaEventRecord(tier0_graph->done, stream));
        }

        for (void* allocation : scratch) {
            release(allocation);
        }
        release(device_puts);
        release(device_ticket_lanes);
        release(device_readiness_slots);
        release(device_readiness);
        release(device_slots);
        release(device_finalize);
        release(device_values);
        release(device_channel_bytes);
        release(device_bf16_rows);
        release(device_mtp_bf16_offsets);
        release(device_mtp_bf16_rows);
        release(device_channels);
        release(device_lanes);
        release(device_header);
        return result;
    }

  public:
    static void launch_op(
        const container::COp& op,
        const plan::StagePlan& stage,
        const GroupedLaneBinding& lane,
        const std::vector<GroupedLaneBinding>& bindings,
        const std::vector<std::uint8_t>& direct_bf16_values,
        const std::uint64_t* mtp_bf16_rows,
        const std::uint32_t* mtp_bf16_offsets,
        std::uint32_t output,
        PtirLaneTableHeader* header,
        PtirLaneRecord* lanes,
        std::uint64_t* values,
        std::uint32_t value_count,
        std::uint32_t lane_count,
        cudaStream_t stream) {
        const plan::ValueType& output_type = stage.value_types[output];
        const auto output_shape =
            grouped_dynamic_shape(output_type, bindings);
        const std::uint64_t numel = grouped_numel(output_type, lane);
        const std::uint64_t total =
            static_cast<std::uint64_t>(lane_count) * numel;
        const std::uint32_t blocks = grouped_grid(total);
        const auto input_numel = [&](std::uint32_t value) {
            return grouped_numel(stage.value_types[value], lane);
        };
        const auto direct_kind = [&](std::uint32_t value) {
            return value < direct_bf16_values.size()
                ? direct_bf16_values[value]
                : std::uint8_t{0};
        };
        const std::uint8_t dtype = output_type.dtype;

        if (op.tag == PTIR_OP_CONST) {
            if (dtype == PTIR_DT_F32) {
                float literal;
                std::memcpy(&literal, &op.lit_bits, sizeof(literal));
                k_grouped_constant<<<blocks, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, output, numel, literal);
            } else if (dtype == PTIR_DT_I32) {
                k_grouped_constant<<<blocks, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, output, numel,
                    static_cast<std::int32_t>(op.lit_bits));
            } else if (dtype == PTIR_DT_U32) {
                k_grouped_constant<<<blocks, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, output, numel,
                    op.lit_bits);
            } else {
                k_grouped_constant<<<blocks, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, output, numel,
                    static_cast<std::uint8_t>(op.lit_bits != 0));
            }
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        if (op.tag == PTIR_OP_IOTA) {
            const auto shape = grouped_dynamic_shape(output_type, bindings);
            k_grouped_iota<<<blocks, kTier0Block, 0, stream>>>(
                header, lanes, values, value_count, output, numel, shape);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        if (op.tag == PTIR_OP_GATHER || op.tag == PTIR_OP_GATHER_ROW) {
            const plan::ValueType& source_type =
                stage.value_types[op.args[0]];
            const std::uint8_t source_dtype = source_type.dtype;
            const std::uint8_t index_dtype =
                stage.value_types[op.args[1]].dtype;
            const bool gather_row = op.tag == PTIR_OP_GATHER_ROW;
            const auto source_shape =
                grouped_dynamic_shape(source_type, bindings);
            const auto axis0_shape = grouped_dimensions_shape(
                source_type, 0, 1, bindings);
            const auto index_shape = grouped_dynamic_shape(
                stage.value_types[op.args[1]], bindings);
            const auto row_shape = grouped_dimensions_shape(
                source_type, 0, 1, bindings);
            const auto column_shape = grouped_dimensions_shape(
                source_type, 1, source_type.dims.size(), bindings);
            if (direct_kind(op.args[0]) != 0) {
#define PIE_GROUP_DIRECT_GATHER(INDEX)                                        \
                k_grouped_direct_gather<INDEX><<<                             \
                    grouped_grid(total), kTier0Block, 0, stream>>>(            \
                    header, lanes, mtp_bf16_rows, mtp_bf16_offsets,           \
                    values, value_count, op.args[1], output, source_shape,    \
                    axis0_shape, index_shape, output_shape, lane.vocab,       \
                    direct_kind(op.args[0]), gather_row)
                if (index_dtype == PTIR_DT_I32) {
                    PIE_GROUP_DIRECT_GATHER(std::int32_t);
                } else {
                    PIE_GROUP_DIRECT_GATHER(std::uint32_t);
                }
#undef PIE_GROUP_DIRECT_GATHER
                CUDA_CHECK(cudaGetLastError());
                return;
            }
#define PIE_GROUP_GATHER(TYPE, INDEX)                                         \
            do {                                                               \
                if (gather_row) {                                              \
                    k_grouped_gather_rows<TYPE, INDEX><<<                      \
                        grouped_grid(total), kTier0Block, 0, stream>>>(         \
                        header, lanes, values, value_count, op.args[0],         \
                        op.args[1], output, row_shape, column_shape);           \
                } else {                                                       \
                    k_grouped_gather_axis0<TYPE, INDEX><<<                     \
                        blocks, kTier0Block, 0, stream>>>(                      \
                        header, lanes, values, value_count, op.args[0],         \
                        op.args[1], output, source_shape, axis0_shape,          \
                        index_shape, output_shape);                             \
                }                                                              \
            } while (false)
            if (index_dtype == PTIR_DT_I32) {
                if (source_dtype == PTIR_DT_F32) {
                    PIE_GROUP_GATHER(float, std::int32_t);
                } else if (source_dtype == PTIR_DT_I32) {
                    PIE_GROUP_GATHER(std::int32_t, std::int32_t);
                } else if (source_dtype == PTIR_DT_U32) {
                    PIE_GROUP_GATHER(std::uint32_t, std::int32_t);
                } else {
                    PIE_GROUP_GATHER(std::uint8_t, std::int32_t);
                }
            } else {
                if (source_dtype == PTIR_DT_F32) {
                    PIE_GROUP_GATHER(float, std::uint32_t);
                } else if (source_dtype == PTIR_DT_I32) {
                    PIE_GROUP_GATHER(std::int32_t, std::uint32_t);
                } else if (source_dtype == PTIR_DT_U32) {
                    PIE_GROUP_GATHER(std::uint32_t, std::uint32_t);
                } else {
                    PIE_GROUP_GATHER(std::uint8_t, std::uint32_t);
                }
            }
#undef PIE_GROUP_GATHER
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        if (op.tag == PTIR_OP_SCATTER_SET ||
            op.tag == PTIR_OP_SCATTER_ADD) {
            const std::uint8_t source_dtype =
                stage.value_types[op.args[0]].dtype;
            const std::uint8_t index_dtype =
                stage.value_types[op.args[1]].dtype;
            const auto base_shape = grouped_dynamic_shape(
                stage.value_types[op.args[0]], bindings);
            const auto axis0_shape = grouped_dimensions_shape(
                stage.value_types[op.args[0]], 0, 1, bindings);
            const auto index_shape = grouped_dynamic_shape(
                stage.value_types[op.args[1]], bindings);
            const bool scalar_values =
                grouped_dynamic_shape(
                    stage.value_types[op.args[2]], bindings).max_numel == 1;
            const bool add = op.tag == PTIR_OP_SCATTER_ADD;
#define PIE_GROUP_SCATTER(TYPE, INDEX)                                        \
            do {                                                               \
                if (add) {                                                     \
                    k_grouped_scatter<TYPE, INDEX, true><<<                    \
                        lane_count, kTier0Block, 0, stream>>>(                  \
                        header, lanes, values, value_count, op.args[0],         \
                        op.args[1], op.args[2], output, base_shape,             \
                        axis0_shape, index_shape, scalar_values);               \
                } else {                                                       \
                    k_grouped_scatter<TYPE, INDEX, false><<<                   \
                        lane_count, kTier0Block, 0, stream>>>(                  \
                        header, lanes, values, value_count, op.args[0],         \
                        op.args[1], op.args[2], output, base_shape,             \
                        axis0_shape, index_shape, scalar_values);               \
                }                                                              \
            } while (false)
            if (index_dtype == PTIR_DT_I32) {
                if (source_dtype == PTIR_DT_F32) {
                    PIE_GROUP_SCATTER(float, std::int32_t);
                } else if (source_dtype == PTIR_DT_I32) {
                    PIE_GROUP_SCATTER(std::int32_t, std::int32_t);
                } else if (source_dtype == PTIR_DT_U32) {
                    PIE_GROUP_SCATTER(std::uint32_t, std::int32_t);
                } else {
                    PIE_GROUP_SCATTER(std::uint8_t, std::int32_t);
                }
            } else {
                if (source_dtype == PTIR_DT_F32) {
                    PIE_GROUP_SCATTER(float, std::uint32_t);
                } else if (source_dtype == PTIR_DT_I32) {
                    PIE_GROUP_SCATTER(std::int32_t, std::uint32_t);
                } else if (source_dtype == PTIR_DT_U32) {
                    PIE_GROUP_SCATTER(std::uint32_t, std::uint32_t);
                } else {
                    PIE_GROUP_SCATTER(std::uint8_t, std::uint32_t);
                }
            }
#undef PIE_GROUP_SCATTER
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        if (op.tag == PTIR_OP_TRANSPOSE) {
            const plan::ValueType& input_type =
                stage.value_types[op.args[0]];
            const auto input_shape =
                grouped_row_shape(input_type, bindings);
            if (dtype == PTIR_DT_F32) {
                k_grouped_transpose<float><<<blocks, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, op.args[0], output,
                    input_shape);
            } else if (dtype == PTIR_DT_I32) {
                k_grouped_transpose<std::int32_t><<<
                    blocks, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, op.args[0], output,
                    input_shape);
            } else if (dtype == PTIR_DT_U32) {
                k_grouped_transpose<std::uint32_t><<<
                    blocks, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, op.args[0], output,
                    input_shape);
            } else {
                k_grouped_transpose<std::uint8_t><<<
                    blocks, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, op.args[0], output,
                    input_shape);
            }
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        if (op.tag == PTIR_OP_MASK_APPLY_PACKED) {
            const std::uint32_t rows =
                grouped_rows(stage.value_types[op.args[0]], lane);
            const std::uint32_t len = static_cast<std::uint32_t>(
                grouped_numel(stage.value_types[op.args[0]], lane) / rows);
            if (direct_kind(op.args[0]) != 0) {
                k_grouped_direct_mask_apply_packed<<<
                    blocks, kTier0Block, 0, stream>>>(
                    header, lanes, mtp_bf16_rows, mtp_bf16_offsets,
                    values, value_count, op.args[1], output, rows, len,
                    lane.vocab, direct_kind(op.args[0]));
            } else {
                k_grouped_mask_apply_packed<<<
                    blocks, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, op.args[0],
                    op.args[1], output, rows, len);
            }
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        if (op.tag == PTIR_OP_CAUSAL_MASK ||
            op.tag == PTIR_OP_SLIDING_WINDOW_MASK ||
            op.tag == PTIR_OP_SINK_WINDOW_MASK) {
            const auto positions = static_cast<std::uint32_t>(
                input_numel(op.args[0]));
            const Tier0StructuredMaskKind kind =
                op.tag == PTIR_OP_CAUSAL_MASK
                    ? Tier0StructuredMaskKind::Causal
                    : op.tag == PTIR_OP_SLIDING_WINDOW_MASK
                        ? Tier0StructuredMaskKind::SlidingWindow
                        : Tier0StructuredMaskKind::SinkWindow;
            const std::uint32_t window =
                op.tag == PTIR_OP_SINK_WINDOW_MASK ? op.imm3 : op.imm2;
            const std::uint32_t sink =
                op.tag == PTIR_OP_SINK_WINDOW_MASK ? op.imm2 : 0;
            k_grouped_structured_position_mask<<<
                blocks, kTier0Block, 0, stream>>>(
                header,
                lanes,
                values,
                value_count,
                op.args[0],
                output,
                positions,
                op.imm,
                kind,
                window,
                sink);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        if (op.tag == PTIR_OP_MATMUL) {
            const plan::ValueType& a_type =
                stage.value_types[op.args[0]];
            const plan::ValueType& b_type =
                stage.value_types[op.args[1]];
            const auto a_shape = grouped_row_shape(a_type, bindings);
            const auto b_shape = grouped_row_shape(b_type, bindings);
            k_grouped_matmul<<<blocks, kTier0Block, 0, stream>>>(
                header, lanes, values, value_count, op.args[0], op.args[1],
                output, a_shape, b_shape);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        if (op.tag == PTIR_OP_RNG) {
            const std::uint32_t rows = grouped_rows(output_type, lane);
            const std::uint32_t len =
                static_cast<std::uint32_t>(numel / rows);
            k_grouped_rng_ambient<<<blocks, kTier0Block, 0, stream>>>(
                header, lanes, values, value_count, output, rows, len,
                op.imm, op.kind != 0);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        if (op.tag == PTIR_OP_RNG_KEYED) {
            k_grouped_rng_keyed<<<blocks, kTier0Block, 0, stream>>>(
                header, lanes, values, value_count, op.args[0], output,
                numel, op.kind != 0);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        if (op.tag == PTIR_OP_REDUCE_ARGMAX) {
            const plan::ValueType& input_type =
                stage.value_types[op.args[0]];
            const auto row_shape =
                grouped_row_shape(input_type, bindings);
            const std::uint32_t rows = row_shape.max_rows;
            const std::uint32_t len = row_shape.max_columns;
            const auto input_kind = direct_kind(op.args[0]);
            if (input_kind != 0) {
                k_grouped_reduce_argmax_direct_bf16<<<
                    lane_count * rows, kTier0Block, 0, stream>>>(
                    header, lanes, mtp_bf16_rows, mtp_bf16_offsets,
                    values, value_count, output, row_shape, lane.vocab,
                    input_kind);
                CUDA_CHECK(cudaGetLastError());
                return;
            }
            if (row_shape.rows.extent != 0xff ||
                row_shape.columns.extent != 0xff) {
                if (input_type.dtype == PTIR_DT_F32) {
                    k_grouped_reduce_argmax_ragged<float><<<
                        lane_count * rows, kTier0Block, 0, stream>>>(
                        header, lanes, values, value_count, op.args[0],
                        output, row_shape);
                } else if (input_type.dtype == PTIR_DT_I32) {
                    k_grouped_reduce_argmax_ragged<std::int32_t><<<
                        lane_count * rows, kTier0Block, 0, stream>>>(
                        header, lanes, values, value_count, op.args[0],
                        output, row_shape);
                } else {
                    k_grouped_reduce_argmax_ragged<std::uint32_t><<<
                        lane_count * rows, kTier0Block, 0, stream>>>(
                        header, lanes, values, value_count, op.args[0],
                        output, row_shape);
                }
                CUDA_CHECK(cudaGetLastError());
                return;
            }
            constexpr std::uint32_t hierarchical_chunk = 1024;
            if (input_type.dtype == PTIR_DT_F32 &&
                len >= hierarchical_chunk * 4 &&
                static_cast<std::uint64_t>(lane_count) * rows < 128) {
                const std::uint32_t chunks =
                    (len + hierarchical_chunk - 1) / hierarchical_chunk;
                const std::size_t partial_count =
                    static_cast<std::size_t>(lane_count) * rows * chunks;
                float* partial_values = nullptr;
                std::uint32_t* partial_indices = nullptr;
                CUDA_CHECK(cudaMallocAsync(
                    reinterpret_cast<void**>(&partial_values),
                    partial_count * sizeof(float),
                    stream));
                CUDA_CHECK(cudaMallocAsync(
                    reinterpret_cast<void**>(&partial_indices),
                    partial_count * sizeof(std::uint32_t),
                    stream));
                k_grouped_argmax_chunks<<<
                    static_cast<std::uint32_t>(partial_count),
                    kTier0Block,
                    0,
                    stream>>>(
                    header,
                    lanes,
                    values,
                    value_count,
                    op.args[0],
                    partial_values,
                    partial_indices,
                    rows,
                    len,
                    chunks,
                    hierarchical_chunk);
                CUDA_CHECK(cudaGetLastError());
                k_grouped_argmax_chunk_results<<<
                    lane_count * rows,
                    kTier0Block,
                    0,
                    stream>>>(
                    header,
                    lanes,
                    values,
                    value_count,
                    partial_values,
                    partial_indices,
                    output,
                    rows,
                    chunks);
                CUDA_CHECK(cudaGetLastError());
                CUDA_CHECK(cudaFreeAsync(partial_indices, stream));
                CUDA_CHECK(cudaFreeAsync(partial_values, stream));
                return;
            }
            if (input_type.dtype == PTIR_DT_F32) {
                k_grouped_reduce_argmax<float><<<
                    lane_count * rows, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, op.args[0], output,
                    rows, len);
            } else if (input_type.dtype == PTIR_DT_I32) {
                k_grouped_reduce_argmax<std::int32_t><<<
                    lane_count * rows, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, op.args[0], output,
                    rows, len);
            } else {
                k_grouped_reduce_argmax<std::uint32_t><<<
                    lane_count * rows, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, op.args[0], output,
                    rows, len);
            }
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        if (op.tag == PTIR_OP_SORT_DESC || op.tag == PTIR_OP_TOP_K) {
            const plan::ValueType& input_type =
                stage.value_types[op.args[0]];
            const auto input_row_shape =
                grouped_row_shape(input_type, bindings);
            const std::uint32_t rows = op.tag == PTIR_OP_SORT_DESC
                ? 1
                : input_row_shape.max_rows;
            const std::uint32_t len = op.tag == PTIR_OP_SORT_DESC
                ? static_cast<std::uint32_t>(
                      grouped_dynamic_shape(input_type, bindings).max_numel)
                : input_row_shape.max_columns;
            const std::uint32_t k =
                op.tag == PTIR_OP_SORT_DESC ? len : op.imm;
            const auto input_shape = grouped_dynamic_shape(
                input_type, bindings);
            k_grouped_topk<<<
                lane_count * rows, kTier0Block, 0, stream>>>(
                header, lanes, mtp_bf16_rows, mtp_bf16_offsets,
                values, value_count, op.args[0], output,
                output + 1, rows, len, k,
                op.tag == PTIR_OP_SORT_DESC, input_shape,
                input_row_shape, lane.vocab,
                direct_kind(op.args[0]));
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        if (op.tag == PTIR_OP_PIVOT_THRESHOLD) {
            const plan::ValueType& input_type =
                stage.value_types[op.args[0]];
            const auto row_shape =
                grouped_row_shape(input_type, bindings);
            const std::uint32_t rows = row_shape.max_rows;
            const std::uint32_t len = row_shape.max_columns;
            const std::uint32_t threshold_numel = static_cast<std::uint32_t>(
                grouped_numel(stage.value_types[op.pred_payload], lane));
            if (op.pred_tag == static_cast<std::uint8_t>(PredTag::RankLe)) {
                const std::uint8_t threshold_dtype =
                    stage.value_types[op.pred_payload].dtype;
                if (threshold_dtype == PTIR_DT_I32) {
                    k_grouped_pivot_rank<std::int32_t><<<
                        lane_count * rows, kTier0Block, 0, stream>>>(
                        header, lanes, values, value_count, op.args[0],
                        op.pred_payload, output, row_shape, threshold_numel);
                } else {
                    k_grouped_pivot_rank<std::uint32_t><<<
                        lane_count * rows, kTier0Block, 0, stream>>>(
                        header, lanes, values, value_count, op.args[0],
                        op.pred_payload, output, row_shape, threshold_numel);
                }
            } else if (
                op.pred_tag ==
                static_cast<std::uint8_t>(PredTag::CummassLe)) {
                k_grouped_pivot_cummass<<<
                    lane_count * rows, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, op.args[0],
                    op.pred_payload, output, row_shape, threshold_numel);
            } else {
                k_grouped_pivot_prob<<<
                    grouped_grid(
                        static_cast<std::uint64_t>(lane_count) * rows * len),
                    kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, op.args[0],
                    op.pred_payload, output, row_shape, threshold_numel);
            }
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        if (op.tag == PTIR_OP_REDUCE_SUM ||
            op.tag == PTIR_OP_REDUCE_MAX ||
            op.tag == PTIR_OP_REDUCE_MIN) {
            const plan::ValueType& input_type =
                stage.value_types[op.args[0]];
            const auto row_shape =
                grouped_row_shape(input_type, bindings);
            const std::uint32_t rows = row_shape.max_rows;
            const std::uint32_t len = row_shape.max_columns;
            const RedKind kind = op.tag == PTIR_OP_REDUCE_SUM
                ? RedKind::Sum
                : (op.tag == PTIR_OP_REDUCE_MAX
                    ? RedKind::Max
                    : RedKind::Min);
            const auto input_kind = direct_kind(op.args[0]);
            if (input_kind != 0) {
                k_grouped_reduce_direct_bf16<<<
                    lane_count * rows, kCanonicalReduceWidth, 0, stream>>>(
                    header, lanes, mtp_bf16_rows, mtp_bf16_offsets,
                    values, value_count, output, row_shape, lane.vocab,
                    input_kind, kind);
                CUDA_CHECK(cudaGetLastError());
                return;
            }
            if (row_shape.rows.extent != 0xff ||
                row_shape.columns.extent != 0xff) {
                if (input_type.dtype == PTIR_DT_F32) {
                    k_grouped_reduce_ragged<float><<<
                        lane_count * rows, kCanonicalReduceWidth, 0, stream>>>(
                        header, lanes, values, value_count, op.args[0],
                        output, row_shape, kind);
                } else if (input_type.dtype == PTIR_DT_I32) {
                    k_grouped_reduce_ragged<std::int32_t><<<
                        lane_count * rows, kCanonicalReduceWidth, 0, stream>>>(
                        header, lanes, values, value_count, op.args[0],
                        output, row_shape, kind);
                } else {
                    k_grouped_reduce_ragged<std::uint32_t><<<
                        lane_count * rows, kCanonicalReduceWidth, 0, stream>>>(
                        header, lanes, values, value_count, op.args[0],
                        output, row_shape, kind);
                }
                CUDA_CHECK(cudaGetLastError());
                return;
            }
            constexpr std::uint32_t hierarchical_chunk = 1024;
            if (len >= hierarchical_chunk * 4 &&
                len % hierarchical_chunk == 0 &&
                static_cast<std::uint64_t>(lane_count) * rows < 128) {
                const std::uint32_t chunks = len / hierarchical_chunk;
                const std::size_t partial_count =
                    static_cast<std::size_t>(lane_count) * rows * chunks;
#define PIE_GROUP_HIERARCHICAL_REDUCE(TYPE)                                  \
                do {                                                          \
                    TYPE* partials = nullptr;                                  \
                    CUDA_CHECK(cudaMallocAsync(                                \
                        reinterpret_cast<void**>(&partials),                   \
                        partial_count * sizeof(TYPE), stream));                \
                    k_grouped_reduce_chunks<TYPE><<<                           \
                        static_cast<std::uint32_t>(partial_count),              \
                        kCanonicalReduceWidth, 0, stream>>>(                    \
                        header, lanes, values, value_count, op.args[0],         \
                        partials, rows, len, chunks, hierarchical_chunk, kind); \
                    CUDA_CHECK(cudaGetLastError());                             \
                    k_grouped_reduce_chunk_results<TYPE><<<                    \
                        lane_count * rows, kCanonicalReduceWidth, 0, stream>>>( \
                        header, lanes, values, value_count, partials, output,   \
                        rows, chunks, kind);                                    \
                    CUDA_CHECK(cudaGetLastError());                             \
                    CUDA_CHECK(cudaFreeAsync(partials, stream));                \
                } while (false)
                if (input_type.dtype == PTIR_DT_F32) {
                    PIE_GROUP_HIERARCHICAL_REDUCE(float);
                } else if (input_type.dtype == PTIR_DT_I32) {
                    PIE_GROUP_HIERARCHICAL_REDUCE(std::int32_t);
                } else {
                    PIE_GROUP_HIERARCHICAL_REDUCE(std::uint32_t);
                }
#undef PIE_GROUP_HIERARCHICAL_REDUCE
                return;
            }
            if (input_type.dtype == PTIR_DT_F32) {
                k_grouped_reduce<float><<<
                    lane_count * rows, kCanonicalReduceWidth, 0, stream>>>(
                    header, lanes, values, value_count, op.args[0], output,
                    rows, len, kind);
            } else if (input_type.dtype == PTIR_DT_I32) {
                k_grouped_reduce<std::int32_t><<<
                    lane_count * rows, kCanonicalReduceWidth, 0, stream>>>(
                    header, lanes, values, value_count, op.args[0], output,
                    rows, len, kind);
            } else {
                k_grouped_reduce<std::uint32_t><<<
                    lane_count * rows, kCanonicalReduceWidth, 0, stream>>>(
                    header, lanes, values, value_count, op.args[0], output,
                    rows, len, kind);
            }
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        if (op.tag == PTIR_OP_CUMSUM || op.tag == PTIR_OP_CUMPROD) {
            const plan::ValueType& input_type =
                stage.value_types[op.args[0]];
            const auto row_shape =
                grouped_row_shape(input_type, bindings);
            const std::uint32_t rows = row_shape.max_rows;
            const ScanKind kind = op.tag == PTIR_OP_CUMSUM
                ? ScanKind::Sum
                : ScanKind::Prod;
            if (input_type.dtype == PTIR_DT_F32) {
                k_grouped_scan<float><<<
                    lane_count * rows, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, op.args[0], output,
                    row_shape, kind);
            } else if (input_type.dtype == PTIR_DT_I32) {
                k_grouped_scan<std::int32_t><<<
                    lane_count * rows, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, op.args[0], output,
                    row_shape, kind);
            } else {
                k_grouped_scan<std::uint32_t><<<
                    lane_count * rows, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, op.args[0], output,
                    row_shape, kind);
            }
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        if (op.tag == PTIR_OP_BROADCAST) {
            const plan::ValueType& source_type =
                stage.value_types[op.args[0]];
            GroupedBroadcastMeta meta;
            meta.rank = static_cast<std::uint32_t>(output_type.dims.size());
            std::uint32_t source_dims[4] = {1, 1, 1, 1};
            for (std::size_t index = 0; index < output_type.dims.size(); ++index) {
                const plan::Dimension& dimension = output_type.dims[index];
                meta.out_dims[index] = dimension.symbolic
                    ? grouped_extent(
                        static_cast<std::uint8_t>(dimension.value), lane)
                    : dimension.value;
            }
            for (std::size_t index = 0; index < source_type.dims.size(); ++index) {
                const plan::Dimension& dimension = source_type.dims[index];
                source_dims[index] = dimension.symbolic
                    ? grouped_extent(
                        static_cast<std::uint8_t>(dimension.value), lane)
                    : dimension.value;
            }
            std::uint32_t stride = 1;
            for (int index = static_cast<int>(meta.rank) - 1;
                 index >= 0;
                 --index) {
                meta.src_strides[index] =
                    source_dims[index] == 1 && meta.out_dims[index] > 1
                        ? 0
                        : stride;
                stride *= source_dims[index];
            }
            if (dtype == PTIR_DT_F32) {
                k_grouped_broadcast<float><<<blocks, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, op.args[0], output,
                    numel, meta);
            } else if (dtype == PTIR_DT_I32) {
                k_grouped_broadcast<std::int32_t><<<
                    blocks, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, op.args[0], output,
                    numel, meta);
            } else if (dtype == PTIR_DT_U32) {
                k_grouped_broadcast<std::uint32_t><<<
                    blocks, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, op.args[0], output,
                    numel, meta);
            } else {
                k_grouped_broadcast<std::uint8_t><<<
                    blocks, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, op.args[0], output,
                    numel, meta);
            }
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        if (op.tag == PTIR_OP_CAST) {
            launch_cast(
                stage.value_types[op.args[0]].dtype, dtype, header, lanes,
                values, value_count, op.args[0], output, numel, blocks,
                stream);
            return;
        }
        if (op.tag == PTIR_OP_SELECT) {
            const bool condition_scalar = input_numel(op.args[0]) == 1;
            const bool a_scalar = input_numel(op.args[1]) == 1;
            const bool b_scalar = input_numel(op.args[2]) == 1;
            const auto a_kind = direct_kind(op.args[1]);
            const auto b_kind = direct_kind(op.args[2]);
            if (dtype == PTIR_DT_F32 &&
                (a_kind != 0 || b_kind != 0)) {
                k_grouped_direct_select<<<
                    blocks, kTier0Block, 0, stream>>>(
                    header, lanes, mtp_bf16_rows, mtp_bf16_offsets,
                    values, value_count, op.args[0], op.args[1],
                    op.args[2], output, numel, output_shape, lane.vocab,
                    a_kind, b_kind,
                    condition_scalar, a_scalar, b_scalar);
            } else if (dtype == PTIR_DT_F32) {
                k_grouped_select<float><<<blocks, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, op.args[0], op.args[1],
                    op.args[2], output, numel, condition_scalar, a_scalar,
                    b_scalar);
            } else if (dtype == PTIR_DT_I32) {
                k_grouped_select<std::int32_t><<<
                    blocks, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, op.args[0], op.args[1],
                    op.args[2], output, numel, condition_scalar, a_scalar,
                    b_scalar);
            } else if (dtype == PTIR_DT_U32) {
                k_grouped_select<std::uint32_t><<<
                    blocks, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, op.args[0], op.args[1],
                    op.args[2], output, numel, condition_scalar, a_scalar,
                    b_scalar);
            } else {
                k_grouped_select<std::uint8_t><<<
                    blocks, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, op.args[0], op.args[1],
                    op.args[2], output, numel, condition_scalar, a_scalar,
                    b_scalar);
            }
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        if (op.tag == PTIR_OP_GT || op.tag == PTIR_OP_GE ||
            op.tag == PTIR_OP_EQ || op.tag == PTIR_OP_NE ||
            op.tag == PTIR_OP_LT || op.tag == PTIR_OP_LE) {
            const CmpKind kind = op.tag == PTIR_OP_GT
                ? CmpKind::Gt
                : op.tag == PTIR_OP_GE
                    ? CmpKind::Ge
                    : op.tag == PTIR_OP_EQ
                        ? CmpKind::Eq
                        : op.tag == PTIR_OP_NE
                            ? CmpKind::Ne
                            : op.tag == PTIR_OP_LT
                                ? CmpKind::Lt
                                : CmpKind::Le;
            const std::uint8_t input_dtype =
                stage.value_types[op.args[0]].dtype;
            const bool a_scalar = input_numel(op.args[0]) == 1 && numel > 1;
            const bool b_scalar = input_numel(op.args[1]) == 1 && numel > 1;
            const auto a_kind = direct_kind(op.args[0]);
            const auto b_kind = direct_kind(op.args[1]);
            if (input_dtype == PTIR_DT_F32 &&
                (a_kind != 0 || b_kind != 0)) {
                k_grouped_direct_compare<<<
                    blocks, kTier0Block, 0, stream>>>(
                    header, lanes, mtp_bf16_rows, mtp_bf16_offsets,
                    values, value_count, op.args[0], op.args[1], output,
                    numel, output_shape, lane.vocab, a_kind, b_kind, kind,
                    a_scalar, b_scalar);
            } else if (input_dtype == PTIR_DT_F32) {
                k_grouped_compare<float><<<blocks, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, op.args[0], op.args[1],
                    output, numel, kind, a_scalar, b_scalar);
            } else if (input_dtype == PTIR_DT_I32) {
                k_grouped_compare<std::int32_t><<<
                    blocks, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, op.args[0], op.args[1],
                    output, numel, kind, a_scalar, b_scalar);
            } else if (input_dtype == PTIR_DT_U32) {
                k_grouped_compare<std::uint32_t><<<
                    blocks, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, op.args[0], op.args[1],
                    output, numel, kind, a_scalar, b_scalar);
            } else {
                k_grouped_compare<std::uint8_t><<<
                    blocks, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, op.args[0], op.args[1],
                    output, numel, kind, a_scalar, b_scalar);
            }
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        if (op.tag == PTIR_OP_AND || op.tag == PTIR_OP_OR) {
            k_grouped_logic<<<blocks, kTier0Block, 0, stream>>>(
                header, lanes, values, value_count, op.args[0], op.args[1],
                output, numel,
                op.tag == PTIR_OP_AND ? LogicKind::And : LogicKind::Or);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        if (op.tag == PTIR_OP_NOT) {
            k_grouped_not<<<blocks, kTier0Block, 0, stream>>>(
                header, lanes, values, value_count, op.args[0], output,
                numel);
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        if (op.tag == PTIR_OP_ADD || op.tag == PTIR_OP_SUB ||
            op.tag == PTIR_OP_MUL || op.tag == PTIR_OP_DIV ||
            op.tag == PTIR_OP_REM ||
            op.tag == PTIR_OP_MAX_ELEM || op.tag == PTIR_OP_MIN_ELEM) {
            const BinKind kind = op.tag == PTIR_OP_ADD
                ? BinKind::Add
                : op.tag == PTIR_OP_SUB
                    ? BinKind::Sub
                    : op.tag == PTIR_OP_MUL
                        ? BinKind::Mul
                        : op.tag == PTIR_OP_DIV
                            ? BinKind::Div
                            : op.tag == PTIR_OP_REM
                                ? BinKind::Rem
                            : op.tag == PTIR_OP_MAX_ELEM
                                ? BinKind::MaxElem
                                : BinKind::MinElem;
            const bool a_scalar = input_numel(op.args[0]) == 1 && numel > 1;
            const bool b_scalar = input_numel(op.args[1]) == 1 && numel > 1;
            const auto a_kind = direct_kind(op.args[0]);
            const auto b_kind = direct_kind(op.args[1]);
            if (dtype == PTIR_DT_F32 &&
                (a_kind != 0 || b_kind != 0)) {
                k_grouped_direct_binary<<<
                    blocks, kTier0Block, 0, stream>>>(
                    header, lanes, mtp_bf16_rows, mtp_bf16_offsets,
                    values, value_count, op.args[0], op.args[1], output,
                    numel, output_shape, lane.vocab, a_kind, b_kind, kind,
                    a_scalar, b_scalar);
            } else if (dtype == PTIR_DT_F32) {
                k_grouped_binary<float><<<blocks, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, op.args[0], op.args[1],
                    output, numel, kind, a_scalar, b_scalar);
            } else if (dtype == PTIR_DT_I32) {
                k_grouped_binary<std::int32_t><<<
                    blocks, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, op.args[0], op.args[1],
                    output, numel, kind, a_scalar, b_scalar);
            } else {
                k_grouped_binary<std::uint32_t><<<
                    blocks, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, op.args[0], op.args[1],
                    output, numel, kind, a_scalar, b_scalar);
            }
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        if (op.tag == PTIR_OP_NEG || op.tag == PTIR_OP_EXP ||
            op.tag == PTIR_OP_LOG || op.tag == PTIR_OP_RECIP ||
            op.tag == PTIR_OP_ABS || op.tag == PTIR_OP_SIGN) {
            const UnKind kind = op.tag == PTIR_OP_NEG
                ? UnKind::Neg
                : op.tag == PTIR_OP_EXP
                    ? UnKind::Exp
                    : op.tag == PTIR_OP_LOG
                        ? UnKind::Log
                        : op.tag == PTIR_OP_RECIP
                            ? UnKind::Recip
                            : op.tag == PTIR_OP_ABS
                                ? UnKind::Abs
                                : UnKind::Sign;
            if (dtype == PTIR_DT_F32 &&
                direct_kind(op.args[0]) != 0) {
                k_grouped_direct_unary<<<
                    blocks, kTier0Block, 0, stream>>>(
                    header, lanes, mtp_bf16_rows, mtp_bf16_offsets,
                    values, value_count, op.args[0], output, numel,
                    output_shape, lane.vocab, direct_kind(op.args[0]),
                    kind);
            } else if (dtype == PTIR_DT_F32) {
                k_grouped_unary<float><<<blocks, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, op.args[0], output,
                    numel, kind);
            } else if (dtype == PTIR_DT_I32) {
                k_grouped_unary<std::int32_t><<<
                    blocks, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, op.args[0], output,
                    numel, kind);
            } else {
                k_grouped_unary<std::uint32_t><<<
                    blocks, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, op.args[0], output,
                    numel, kind);
            }
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        throw std::runtime_error("unsupported grouped Tier 0 op");
    }

    static void launch_cast(
        std::uint8_t source,
        std::uint8_t target,
        PtirLaneTableHeader* header,
        PtirLaneRecord* lanes,
        std::uint64_t* values,
        std::uint32_t value_count,
        std::uint32_t input,
        std::uint32_t output,
        std::uint64_t numel,
        std::uint32_t blocks,
        cudaStream_t stream) {
        if (target == PTIR_DT_BOOL && source != PTIR_DT_BOOL) {
            if (source == PTIR_DT_F32) {
                k_grouped_cast_bool<float><<<blocks, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, input, output, numel);
            } else if (source == PTIR_DT_I32) {
                k_grouped_cast_bool<std::int32_t><<<
                    blocks, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, input, output, numel);
            } else {
                k_grouped_cast_bool<std::uint32_t><<<
                    blocks, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, input, output, numel);
            }
            CUDA_CHECK(cudaGetLastError());
            return;
        }
#define PIE_GROUP_CAST(SRC_TAG, SRC_TYPE, DST_TAG, DST_TYPE)                  \
        if (source == SRC_TAG && target == DST_TAG) {                         \
            k_grouped_cast<SRC_TYPE, DST_TYPE><<<                             \
                blocks, kTier0Block, 0, stream>>>(                            \
                header, lanes, values, value_count, input, output, numel);    \
            CUDA_CHECK(cudaGetLastError());                                   \
            return;                                                           \
        }
        PIE_GROUP_CAST(PTIR_DT_F32, float, PTIR_DT_I32, std::int32_t)
        PIE_GROUP_CAST(PTIR_DT_F32, float, PTIR_DT_U32, std::uint32_t)
        PIE_GROUP_CAST(PTIR_DT_I32, std::int32_t, PTIR_DT_F32, float)
        PIE_GROUP_CAST(PTIR_DT_I32, std::int32_t, PTIR_DT_U32, std::uint32_t)
        PIE_GROUP_CAST(PTIR_DT_U32, std::uint32_t, PTIR_DT_F32, float)
        PIE_GROUP_CAST(PTIR_DT_U32, std::uint32_t, PTIR_DT_I32, std::int32_t)
        PIE_GROUP_CAST(PTIR_DT_BOOL, std::uint8_t, PTIR_DT_F32, float)
        PIE_GROUP_CAST(PTIR_DT_BOOL, std::uint8_t, PTIR_DT_I32, std::int32_t)
        PIE_GROUP_CAST(PTIR_DT_BOOL, std::uint8_t, PTIR_DT_U32, std::uint32_t)
#undef PIE_GROUP_CAST
        if (source == target) {
            if (target == PTIR_DT_F32) {
                k_grouped_cast<float, float><<<blocks, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, input, output, numel);
            } else if (target == PTIR_DT_I32) {
                k_grouped_cast<std::int32_t, std::int32_t><<<
                    blocks, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, input, output, numel);
            } else if (target == PTIR_DT_U32) {
                k_grouped_cast<std::uint32_t, std::uint32_t><<<
                    blocks, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, input, output, numel);
            } else {
                k_grouped_cast<std::uint8_t, std::uint8_t><<<
                    blocks, kTier0Block, 0, stream>>>(
                    header, lanes, values, value_count, input, output, numel);
            }
            CUDA_CHECK(cudaGetLastError());
            return;
        }
        throw std::runtime_error("unsupported grouped cast");
    }
};

}  // namespace pie_cuda_driver::pipeline
