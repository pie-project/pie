#pragma once

// CUDA-FREE geometry POD for the PTIR pre-forward descriptor resolver (W1.1).
// Split out of descriptor_resolve.hpp so the host-side dispatch façade
// (driver/cuda's pipeline/dispatch.hpp, included by CUDA-free .cpp TUs) can
// name it without pulling device headers. The resolver
// (driver/cuda's pipeline/descriptor_resolve.hpp, nvcc TU) fills it; the
// executor feeds it into the standard batch assembly.

#include <cstddef>
#include <cstdint>
#include <limits>
#include <string>
#include <vector>

namespace pie_native::ptir {

enum class StructuredMaskKind : std::uint8_t {
    None = 0,
    Causal = 1,
    SlidingWindow = 2,
    SinkWindow = 3,
};

struct StructuredMaskDescriptor {
    StructuredMaskKind kind = StructuredMaskKind::None;
    std::uint32_t key_len = 0;
    std::uint32_t sink = 0;
    std::uint32_t window = 0;

    explicit operator bool() const noexcept {
        return kind != StructuredMaskKind::None;
    }
};

// The forward geometry a device-geometry PTIR fire contributes, resolved from
// its descriptor-port channels. Mirrors the runtime `ReqGeometry` fields plus
// the explicit-write descriptor (`w_page`/`w_off`) and the dense attention mask.
struct FireGeometry {
    std::vector<std::uint32_t> token_ids;         // embed_tokens
    std::vector<std::uint32_t> position_ids;      // positions (else 0..nnz)
    std::vector<std::uint32_t> qo_indptr;         // embed_indptr (else [0,nnz])
    std::vector<std::uint32_t> kv_page_indices;   // pages (CSR-prefix trimmed)
    std::vector<std::uint32_t> kv_page_indptr;    // page_indptr
    std::vector<std::uint32_t> kv_last_page_lens; // from kv_len
    std::vector<std::uint32_t> sampling_indices;  // readout (else last of each lane)
    std::vector<std::uint32_t> sampling_indptr;
    std::vector<std::uint32_t> w_page;            // w_slot (PHYSICAL page id per lane)
    std::vector<std::uint32_t> w_off;             // w_off (offset-in-page per lane)
    std::vector<std::uint8_t>  mask;              // attn_mask, dense [lanes, stride] bytes
    StructuredMaskDescriptor structured_mask;
    bool has_kv_family = false;                   // pages/page_indptr present
    bool has_write_desc = false;                  // w_slot/w_off present
    bool has_mask = false;                        // attn_mask present
};

// Pre-forward descriptor resolution over a whole batch: one entry per
// launched program, parallel to `ptir_program_instances`. Wire programs
// (geometry on the launch wire) keep an empty `per_program` entry with
// `is_device_geometry[p] == 0`; device-geometry programs carry their
// channel-resolved, translation-mapped `FireGeometry`.
struct ResolvedPrograms {
    std::vector<FireGeometry>  per_program;
    std::vector<std::uint8_t>  is_device_geometry;
    std::size_t                device_count = 0;
    bool                       device_composed = false;
    // Mixed [wire][envelope] step: the device-geometry entries are
    // envelope-class shape templates sized to their FULL envelope width
    // (the composed CSRs reserve device-write capacity); the executor
    // routes them through the offset fixed-decode compose after the
    // ordinary wire refill — never the readback fallback.
    bool                       mixed_envelope = false;
};

inline bool validate_fire_geometry(const FireGeometry& geometry,
                                   std::uint32_t device_pages,
                                   std::uint32_t page_size,
                                   std::string* error = nullptr) {
    auto fail = [&](const char* message) {
        if (error != nullptr) *error = message;
        return false;
    };
    if (page_size == 0 ||
        geometry.token_ids.size() >
            static_cast<std::size_t>(std::numeric_limits<int>::max()) ||
        geometry.position_ids.size() != geometry.token_ids.size() ||
        geometry.qo_indptr.size() < 2 ||
        geometry.qo_indptr.front() != 0 ||
        geometry.qo_indptr.back() != geometry.token_ids.size()) {
        return fail("ptir: invalid resolved token geometry");
    }
    const std::size_t requests = geometry.qo_indptr.size() - 1;
    for (std::size_t request = 0; request < requests; ++request) {
        if (geometry.qo_indptr[request] >
            geometry.qo_indptr[request + 1]) {
            return fail("ptir: resolved qo_indptr is not monotonic");
        }
    }

    if (!geometry.token_ids.empty()) {
        if (!geometry.has_kv_family)
            return fail("ptir: resolved KV geometry has no Pages port");
        if (geometry.kv_page_indptr.size() != requests + 1) {
            if (error != nullptr) {
                *error = "ptir: resolved PageIndptr has " +
                    std::to_string(geometry.kv_page_indptr.size()) +
                    " entries for " + std::to_string(requests) + " lanes";
            }
            return false;
        }
        if (geometry.kv_page_indptr.front() != 0)
            return fail("ptir: resolved PageIndptr must start at zero");
        if (geometry.kv_page_indptr.back() != geometry.kv_page_indices.size())
            return fail("ptir: resolved PageIndptr does not cover Pages");
        if (geometry.kv_last_page_lens.size() != requests)
            return fail("ptir: resolved KvLen shape does not match lanes");
        for (std::uint32_t page : geometry.kv_page_indices) {
            if (page >= device_pages) {
                return fail("ptir: resolved KV page is out of range");
            }
        }
        for (std::size_t request = 0; request < requests; ++request) {
            const std::uint32_t page_begin =
                geometry.kv_page_indptr[request];
            const std::uint32_t page_end =
                geometry.kv_page_indptr[request + 1];
            if (page_begin > page_end) {
                return fail("ptir: resolved kv_page_indptr is not monotonic");
            }
            const std::uint32_t query_rows =
                geometry.qo_indptr[request + 1] -
                geometry.qo_indptr[request];
            const std::uint32_t page_count = page_end - page_begin;
            const std::uint32_t last_len =
                geometry.kv_last_page_lens[request];
            if (page_count == 0 || last_len == 0 ||
                last_len > page_size) {
                return fail("ptir: invalid resolved KV extent");
            }
            const std::uint64_t kv_len =
                static_cast<std::uint64_t>(page_count - 1) * page_size +
                last_len;
            if (kv_len < query_rows) {
                return fail("ptir: resolved KV extent is shorter than query span");
            }
        }
    }

    if (geometry.sampling_indptr.size() != requests + 1 ||
        geometry.sampling_indptr.front() != 0 ||
        geometry.sampling_indptr.back() !=
            geometry.sampling_indices.size()) {
        return fail("ptir: invalid resolved sampling CSR");
    }
    for (std::size_t request = 0; request < requests; ++request) {
        if (geometry.sampling_indptr[request] >
            geometry.sampling_indptr[request + 1]) {
            return fail("ptir: resolved sampling CSR is not monotonic");
        }
        const std::uint32_t query_rows =
            geometry.qo_indptr[request + 1] -
            geometry.qo_indptr[request];
        for (std::uint32_t index = geometry.sampling_indptr[request];
             index < geometry.sampling_indptr[request + 1];
             ++index) {
            if (geometry.sampling_indices[index] >= query_rows) {
                return fail("ptir: resolved sampling row is out of range");
            }
        }
    }

    if (geometry.has_write_desc ||
        !geometry.w_page.empty() ||
        !geometry.w_off.empty()) {
        if (!geometry.has_write_desc ||
            geometry.w_page.size() != geometry.token_ids.size() ||
            geometry.w_off.size() != geometry.token_ids.size()) {
            return fail("ptir: invalid resolved write descriptor shape");
        }
        for (std::size_t i = 0; i < geometry.w_page.size(); ++i) {
            if (geometry.w_page[i] >= device_pages ||
                geometry.w_off[i] >= page_size) {
                return fail("ptir: resolved write descriptor is out of range");
            }
        }
    }

    if (geometry.has_mask) {
        if (geometry.mask.empty() || geometry.token_ids.empty() ||
            geometry.mask.size() % geometry.token_ids.size() != 0) {
            return fail("ptir: invalid resolved attention mask shape");
        }
        const std::size_t stride =
            geometry.mask.size() / geometry.token_ids.size();
        for (std::size_t request = 0; request < requests; ++request) {
            const std::uint32_t page_count =
                geometry.kv_page_indptr[request + 1] -
                geometry.kv_page_indptr[request];
            const std::uint64_t kv_len =
                static_cast<std::uint64_t>(page_count - 1) * page_size +
                geometry.kv_last_page_lens[request];
            if (stride < kv_len) {
                return fail("ptir: resolved attention mask is shorter than KV");
            }
        }
    }
    if (geometry.structured_mask) {
        const auto& mask = geometry.structured_mask;
        if (mask.key_len == 0 ||
            geometry.position_ids.size() != geometry.token_ids.size()) {
            return fail("ptir: invalid structured attention mask shape");
        }
        // Idle fires legitimately carry an empty query span and no KV
        // projection. The descriptor is still validated above, but there is
        // no per-request KV extent to compare against.
        if (geometry.token_ids.empty()) return true;
        if (geometry.kv_page_indptr.size() != requests + 1 ||
            geometry.kv_last_page_lens.size() != requests) {
            return fail(
                "ptir: structured attention mask is missing KV geometry");
        }
        if (geometry.has_mask) {
            const std::size_t stride =
                geometry.mask.size() / geometry.token_ids.size();
            if (stride < mask.key_len) {
                return fail(
                    "ptir: dense structured fallback is shorter than its descriptor");
            }
        }
        for (std::size_t request = 0; request < requests; ++request) {
            const std::uint32_t page_count =
                geometry.kv_page_indptr[request + 1] -
                geometry.kv_page_indptr[request];
            const std::uint64_t kv_len =
                static_cast<std::uint64_t>(page_count - 1) * page_size +
                geometry.kv_last_page_lens[request];
            if (mask.key_len < kv_len) {
                return fail(
                    "ptir: structured attention mask is shorter than KV");
            }
        }
    }
    return true;
}

inline bool validate_kv_write_containment(
    const FireGeometry& geometry,
    std::uint32_t page_size,
    std::uint64_t lower,
    std::uint64_t upper,
    std::string* error = nullptr) {
    if (!geometry.has_write_desc) return true;
    if (page_size == 0 ||
        geometry.w_page.size() != geometry.token_ids.size() ||
        geometry.w_off.size() != geometry.token_ids.size()) {
        if (error != nullptr) {
            *error = "ptir: invalid resolved write descriptor shape";
        }
        return false;
    }
    const auto row_in_bounds = [&](std::size_t row,
                                   std::uint64_t effective_lower) {
        if (geometry.token_ids[row] ==
            std::numeric_limits<std::uint32_t>::max()) {
            return true;
        }
        const std::uint64_t token =
            static_cast<std::uint64_t>(geometry.w_page[row]) * page_size +
            geometry.w_off[row];
        if (token < effective_lower || token >= upper) {
            if (error != nullptr) {
                *error =
                    "ptir: resolved KV write escapes containment bounds";
            }
            return false;
        }
        return true;
    };
    for (std::size_t row = 0; row < geometry.w_page.size(); ++row) {
        if (!row_in_bounds(row, lower)) return false;
    }
    return true;
}

}  // namespace pie_native::ptir
