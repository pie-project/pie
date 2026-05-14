#pragma once

// Zero-copy flat-schema decoder for `BatchedForwardPassRequest`.
// Mirrors `runtime/src/shmem_schema.rs`.
//
// Layout (little-endian, host x86_64):
//
//   [512-byte header]
//     0:  u32 magic = 0x42504951 ("BPIQ")
//     4:  u32 schema_version = 2
//     8:  u32 device_id
//     12: u32 flags  (bit 0 = single_token_mode)
//     16: u32 num_arrays = 29
//     20: u32 reserved
//     24: u64 reserved
//     32: 29 × (u32 offset, u32 len_in_elements) = 232 bytes
//     264..512: reserved
//     512: array data (concatenated, each array 8-byte aligned)
//
// v2 added `A_PREDICT_FLAGS` (u8 per request) for pass-level
// speculative execution and bumped HEADER_SIZE 256 → 512 so the
// offset/len table fits. See SPECULATIVE_EXECUTION_DESIGN.md.

#include <cstddef>
#include <cstdint>
#include <span>
#include <stdexcept>
#include <string>

namespace pie_portable_driver::schema {

inline constexpr std::uint32_t MAGIC = 0x42504951;  // 'BPIQ'
inline constexpr std::uint32_t SCHEMA_VERSION = 2;
inline constexpr std::size_t HEADER_SIZE = 512;
inline constexpr std::size_t NUM_ARRAYS = 29;
inline constexpr std::size_t FIXED_HEADER = 32;

// Array indices — must match `runtime/src/shmem_schema.rs` and
// `pie/src/pie_driver/shmem_schema.py`.
enum ArrayIndex : std::size_t {
    A_TOKEN_IDS = 0,
    A_POSITION_IDS = 1,
    A_KV_PAGE_INDICES = 2,
    A_KV_PAGE_INDPTR = 3,
    A_KV_LAST_PAGE_LENS = 4,
    A_QO_INDPTR = 5,
    A_FLATTENED_MASKS = 6,
    A_MASK_INDPTR = 7,
    A_LOGIT_MASKS = 8,
    A_LOGIT_MASK_INDPTR = 9,
    A_SAMPLING_INDICES = 10,
    A_SAMPLING_INDPTR = 11,
    A_SAMPLER_TEMPERATURES = 12,  // f32
    A_SAMPLER_TOP_K = 13,
    A_SAMPLER_TOP_P = 14,         // f32
    A_SAMPLER_MIN_P = 15,         // f32
    A_SAMPLER_TYPES = 16,
    A_SAMPLER_SEEDS = 17,
    A_REQUEST_NUM_SAMPLERS = 18,
    A_SAMPLER_LABEL_IDS = 19,
    A_SAMPLER_LABEL_INDPTR = 20,
    A_ADAPTER_INDICES = 21,       // i64, -1 = None
    A_ADAPTER_SEEDS = 22,         // i64, INT64_MIN = None
    A_SPEC_TOKEN_IDS = 23,
    A_SPEC_POSITION_IDS = 24,
    A_SPEC_INDPTR = 25,
    A_OUTPUT_SPEC_FLAGS = 26,     // u8
    A_CONTEXT_IDS = 27,           // u64
    A_PREDICT_FLAGS = 28,         // u8 (v2)
};

inline constexpr std::size_t ELEM_SIZE[NUM_ARRAYS] = {
    4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4, 4,
    4, 4, 4, 4, 4, 4, 4, 4, 4, 8, 8, 4, 4, 4, 1, 8, 1,
};

struct DecodedRequest {
    std::uint32_t device_id;
    bool single_token_mode;

    // Raw byte view of each array. Caller casts to the appropriate element
    // type using `ELEM_SIZE[i]`.
    std::span<const std::uint8_t> arrays[NUM_ARRAYS];

    template <typename T>
    std::span<const T> as(ArrayIndex i) const noexcept {
        return std::span<const T>(reinterpret_cast<const T*>(arrays[i].data()),
                                  arrays[i].size() / sizeof(T));
    }
};

// Decode a flat request from `buf`. Throws on magic / version mismatch or
// out-of-bounds offsets.
DecodedRequest decode_request(std::span<const std::uint8_t> buf);

}  // namespace pie_portable_driver::schema
